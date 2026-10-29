//! Batched MoE with one expert wave.
//!
//! The chunk form of `R/moe.rs::step_paged` for `T` token rows. The router runs on the device and
//! one readback of `[T, E]` probabilities is the only synchronization. The host selects the experts
//! of each row with `moe::select`, merges the choices into the sorted union of the chunk
//! (`prefill_plan::union_and_remap`), and reads the union once into a page set of the prefill pool.
//! `gather_qmm_sorted` then runs over all `T * top_k` (token, choice) pairs: `split_waves` orders
//! the pairs of each wave by local rank, the non-decreasing order that MLX's `sorted_indices` flag
//! requires, and `pool.import_wave` proves that order on the host before the import, because the
//! method cannot read the device indices (on the Q36 model the flag cut the expert products of a
//! 512-row chunk from about 457 ms to 207 ms; the decode path keeps `gather_qmm` with `false`). The
//! gate-weighted rows of a token are summed in choice order, as the decode step sums them.

use super::super::super::{
    moe::{MoeShape, select},
    prefill_plan::{ChunkRoute, WavePlan, split_waves, union_and_remap},
};
use super::super::{
    ops::{Ctx, dim, values},
    weights::RMoe,
};
use super::pool::{PrefillPool, WavePages};
use crate::{
    EngError,
    kernels::{BinaryOp, ReduceOp, UnaryOp},
    tensor::ExecutionTensor,
    weights::StreamedCheckpoint,
};

/// Returns a `BadRequest` with `message`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// Reads the router probabilities `softmax(x · router_t)` `[T, E]` back, selects the experts of
/// every row on the host, and merges them into the route of the chunk. One synchronization.
fn chunk_route(
    ctx: &mut Ctx<'_>,
    shape: &MoeShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<ChunkRoute, EngError> {
    let &[tokens, _] = x.shape() else {
        return Err(bad("chunk MoE input must have rank two"));
    };
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    let host = values(ctx.kernels, &probs)?;
    let experts = shape.experts();
    let expected = usize::try_from(tokens)
        .ok()
        .and_then(|tokens| tokens.checked_mul(experts));
    if expected != Some(host.len()) {
        return Err(bad("router probabilities do not match the chunk"));
    }
    let selections = host
        .chunks_exact(experts)
        .map(|row| select(shape, row))
        .collect::<Result<Vec<_>, _>>()?;
    union_and_remap(&selections, dim(experts)?)
}

/// Returns the gate-weighted expert rows `[n, H]` of the `n` pairs of one wave.
///
/// Row `i` is the token row `tokens[i]` of `x` through the expert `ranks[i]` of the wave, times
/// `gates[i]`. The three expert products run `gather_qmm_sorted` with `pages.ranks`: the wave pairs
/// arrive from `split_waves` sorted by local rank, the non-decreasing order that MLX's
/// `sorted_indices` flag requires, and `pool.import_wave` proved that order before the import. A
/// rank above the wave's expert count would clamp to the last expert on the device, so the ranks
/// must come from `split_waves`.
fn wave_rows(
    ctx: &mut Ctx<'_>,
    pages: &WavePages,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let &[_, hidden] = x.shape() else {
        return Err(bad("chunk MoE input must have rank two"));
    };
    let &[pairs] = pages.ranks.shape() else {
        return Err(bad("wave ranks must have rank one"));
    };
    let [g, u, d] = &pages.stacks;
    let picked = ctx.kernels.take(x, &pages.tokens, 0)?;
    let x_pairs = ctx.kernels.reshape(&picked, &[pairs, 1, hidden])?;
    let hg = ctx.kernels.gather_qmm_sorted(&x_pairs, g, &pages.ranks)?;
    let hu = ctx.kernels.gather_qmm_sorted(&x_pairs, u, &pages.ranks)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx.kernels.gather_qmm_sorted(&h, d, &pages.ranks)?;
    let g3 = ctx.kernels.reshape(&pages.gates, &[pairs, 1, 1])?;
    let weighted = ctx.kernels.binary(BinaryOp::Mul, &eo, &g3)?;
    ctx.kernels.reshape(&weighted, &[pairs, hidden])
}

/// Returns the gate-weighted expert rows `[T, H]` of a chunk whose union runs in the waves of
/// `plan`.
///
/// Each wave reads its experts into a free set. A wave before the last is evaluated and its set
/// released at once (one synchronization), so the next wave can reuse a set. The last wave stays
/// lent until the router readback of the next layer, as the only wave of a small chunk does. The
/// rows of all waves are joined in wave order and restored to pair order with the inverse
/// permutation, whatever the wave count: `split_waves` lists the pairs of every wave, a single
/// wave included, in (local rank, pair) order, so one wave needs the inverse as much as several.
/// The restored rows are summed in choice order, as the decode step sums them.
fn routed_rows(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    w: &RMoe,
    pool: &mut PrefillPool,
    plan: &WavePlan,
    x: &ExecutionTensor,
    top_k: u32,
) -> Result<ExecutionTensor, EngError> {
    let &[tokens, hidden] = x.shape() else {
        return Err(bad("chunk MoE input must have rank two"));
    };
    let mut parts: Vec<ExecutionTensor> = Vec::with_capacity(plan.waves.len());
    let mut inverse: Option<ExecutionTensor> = None;
    for (index, wave) in plan.waves.iter().enumerate() {
        let last = index + 1 == plan.waves.len();
        let wanted = last.then_some(plan.inverse.as_slice());
        let pages = pool.import_wave(ctx.kernels, ckpt, w, wave, wanted)?;
        let rows = wave_rows(ctx, &pages, x)?;
        if last {
            inverse = pages.inverse;
        } else {
            ctx.kernels.evaluate(&[&rows])?;
            pool.release_lent(); // the evaluate above covers this wave
        }
        parts.push(rows);
    }
    let ordered = match inverse {
        Some(inverse) => {
            let refs: Vec<&ExecutionTensor> = parts.iter().collect();
            let joined = ctx.kernels.concatenate(&refs, 0)?;
            let ordered = ctx.kernels.take(&joined, &inverse, 0)?;
            ctx.kernels.reshape(&ordered, &[tokens, top_k, hidden])?
        }
        None => return Err(bad("chunk waves are inconsistent")),
    };
    ctx.kernels.reduce(ReduceOp::Sum, &ordered, 1, false)
}

/// Shared expert `sigmoid(x · s) * down(silu(gate(x)) * up(x))` for `x` `[T, H]`, giving `[T, H]`.
/// It is the chunk form of `R/moe.rs::shared`: `qmm` takes the `T` rows.
fn shared_rows(
    ctx: &mut Ctx<'_>,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let sg = ctx.kernels.qmm(x, &w.shared_gate)?;
    let su = ctx.kernels.qmm(x, &w.shared_up)?;
    let act = ctx.kernels.silu(&sg)?;
    let sh = ctx.kernels.binary(BinaryOp::Mul, &act, &su)?;
    let so = ctx.kernels.qmm(&sh, &w.shared_down)?;
    let raw = ctx.kernels.qmm(x, &w.shared_scalar)?;
    let gs = ctx.kernels.unary(UnaryOp::Sigmoid, &raw)?;
    ctx.kernels.binary(BinaryOp::Mul, &gs, &so)
}

/// Runs the MoE of one layer for the `T` rows of `x` `[T, H]` and returns `[T, H]` with the route
/// of the chunk. It synchronizes once for the router readback and once more for each wave after
/// the first (`wave_count(distinct, pool.set_experts()) - 1` evaluations).
///
/// The router readback evaluates `x`. Every set that `pool` lent before is released right after
/// it, so the caller must make every earlier MoE output that is not yet evaluated an ancestor of
/// `x`, as consecutive layers are. The set of the last wave stays lent until the next release.
/// The union splits into waves of at most `pool.set_experts()` experts.
pub(super) fn step_chunk(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    shape: &MoeShape,
    w: &RMoe,
    pool: &mut PrefillPool,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, ChunkRoute), EngError> {
    let route = chunk_route(ctx, shape, w, x)?;
    pool.release_lent(); // the router readback evaluated every earlier layer
    let plan = split_waves(&route, pool.set_experts())?;
    let experts = routed_rows(ctx, ckpt, w, pool, &plan, x, route.top_k)?;
    let shared = shared_rows(ctx, w, x)?;
    let out = ctx.kernels.binary(BinaryOp::Add, &experts, &shared)?;
    Ok((out, route))
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        fixtures::{Entry, config},
        moe::{
            step_paged,
            tests::{checkpoint_from, routed_entries},
        },
        slots::{ExpertSlots, POOL_SETS, SetLayout},
        weights::ResidentWeights,
    };
    use super::super::{assert_parity, import_f32};
    use super::*;
    use crate::{
        family::flash_next::load::FlashWeights,
        kernels::{HostPages, MlxKernels},
    };

    /// Hidden width of the fixture.
    const HIDDEN: usize = 64;
    /// The router of layer 0 in the fixture checkpoint.
    const ROUTER: &str = "language_model.model.layers.0.mlp.gate.weight";

    /// The routed fixture with a tied router. The router rows of experts 0 to 3 are the constants
    /// 0.25, 0.125, 0.125, and -0.25, so experts 1 and 2 always have equal logits.
    fn tied_entries() -> Vec<Entry> {
        let mut out = routed_entries();
        for (name, _, _, bytes) in &mut out {
            if name == ROUTER {
                bytes.clear();
                for value in [0.25f32, 0.125, 0.125, -0.25] {
                    for _ in 0..HIDDEN {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        out
    }

    /// The `(c, s)` of the four test tokens. Token input: `x[h] = c + (((h + s) % 4) - 1.5) / 8`.
    /// The second term sums to zero over the 64 positions, so the router logits are
    /// `64 * c * [0.25, 0.125, 0.125, -0.25]`: token 0 `[1, 0.5, 0.5, -1]`, token 1
    /// `[-1, -0.5, -0.5, 1]`, token 2 `[0, 0, 0, 0]`, and token 3 as token 0. All sums are exact
    /// in F32.
    const TOKENS: [(f32, usize); 4] = [(0.0625, 0), (-0.0625, 1), (0.0, 2), (0.0625, 2)];

    /// Input row of token `index`.
    fn token(index: usize) -> Vec<f32> {
        let (c, s) = TOKENS[index];
        (0..HIDDEN)
            .map(|h| c + ((((h + s) % 4) as f32) - 1.5) * 0.125)
            .collect()
    }

    /// The MoE output of each token of `TOKENS` through layer 0 of the tied fixture: the first
    /// four values and the sum of all 64 values. They come from a float64 script that follows the
    /// model definition and shares no code with the engine. Expert `e` has the weight
    /// `W[r][c] = (c % 8) * SCALES[e] + ((e + r + shift) % 4) / 4 - 1/2` for gate, up (`shift` 1),
    /// and down (`shift` 0). The shared expert has `W[r][c] = (c % 8) + [0, 1/4, 1/2, 3/4][r % 4]`
    /// and its scalar gate is `sigmoid(sum_c (c % 8) * x[c])`. The output is
    /// `sum_j gate_j * down(silu(gate(x)) * up(x)) + shared`, with the routes of the router test
    /// below: token 0 experts 0 and 1, token 1 experts 3 and 1, token 2 experts 0 and 1, and
    /// token 3 as token 0.
    const ANCHORS: [([f64; 4], f64); 4] = [
        (
            [110_764.122, 119_123.579, 127_483.036, 134_247.786],
            7_865_896.36,
        ),
        (
            [13.646_297_2, 5.320_916_88, 8.186_369_86, 10.780_844_2],
            606.950_851,
        ),
        (
            [14.541_332_5, 16.606_482_5, 18.671_632_5, 13.907_456_7],
            1_019.630_47,
        ),
        (
            [14_665.791, 15_711.052_1, 16_756.313_2, 17_661.766_4],
            1_036_718.76,
        ),
    ];

    /// Checks the output row of token `index` against `ANCHORS` within 1e-4 relative.
    fn assert_anchor(row: &[f32], index: usize, what: &str) {
        assert_eq!(row.len(), HIDDEN, "{what}: width");
        let (first, total) = ANCHORS[index];
        for (position, (got, want)) in row.iter().zip(first).enumerate() {
            let got = f64::from(*got);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs(),
                "{what}: output {position} is {got}, want {want}"
            );
        }
        let sum: f64 = row.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (sum - total).abs() <= 1e-4 * total.abs(),
            "{what}: sum {sum}, want {total}"
        );
    }

    /// Loads the tied fixture.
    fn load(kernels: &mut MlxKernels) -> (StreamedCheckpoint, ResidentWeights) {
        let ckpt = checkpoint_from(tied_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(kernels, &flash).unwrap();
        (ckpt, weights)
    }

    /// The rows of the tokens `indices` as one `[T, 64]` resident.
    fn chunk_input(kernels: &mut MlxKernels, indices: &[usize]) -> ExecutionTensor {
        let rows: Vec<f32> = indices.iter().flat_map(|&i| token(i)).collect();
        import_f32(
            kernels,
            vec![u32::try_from(indices.len()).unwrap(), 64],
            &rows,
        )
    }

    /// A prefill pool for the layers of `weights`.
    fn pool_for(
        ckpt: &StreamedCheckpoint,
        weights: &ResidentWeights,
        set_experts: u32,
        pairs: u32,
    ) -> PrefillPool {
        let layers = weights.layers.iter().map(|layer| &layer.moe);
        PrefillPool::new(ckpt, layers, set_experts, pairs).unwrap()
    }

    /// Runs `step_chunk` on layer 0 for the tokens `indices` and reads the output back, so the
    /// next call may release the set of this one.
    fn run_chunk(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        ckpt: &StreamedCheckpoint,
        pool: &mut PrefillPool,
        indices: &[usize],
    ) -> Result<(Vec<f32>, ChunkRoute), EngError> {
        let x = chunk_input(kernels, indices);
        let mut ctx = Ctx {
            kernels: &mut *kernels,
            consts: &weights.consts,
        };
        let moe_w = &weights.layers[0].moe;
        let (out, route) = step_chunk(&mut ctx, ckpt, &weights.shape.moe, moe_w, pool, &x)?;
        Ok((values(kernels, &out)?, route))
    }

    /// The decode pool of the existing tests: `POOL_SETS` sets of top-k experts.
    fn decode_slots(ckpt: &StreamedCheckpoint, weights: &ResidentWeights) -> ExpertSlots {
        let k = u32::try_from(weights.shape.moe.top_k()).unwrap();
        let moe_w = &weights.layers[0].moe;
        let layout = SetLayout::new(ckpt, moe_w, k, HostPages::page_size()).unwrap();
        ExpertSlots::new(POOL_SETS, layout.end).unwrap()
    }

    /// One decode MoE call of layer 0 on token `index`, read back.
    fn decode_row(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        ckpt: &StreamedCheckpoint,
        slots: &mut ExpertSlots,
        index: usize,
    ) -> Vec<f32> {
        let x = import_f32(kernels, vec![1, 64], &token(index));
        let mut ctx = Ctx {
            kernels: &mut *kernels,
            consts: &weights.consts,
        };
        let moe_w = &weights.layers[0].moe;
        let (out, _) = step_paged(&mut ctx, ckpt, &weights.shape.moe, moe_w, slots, &x).unwrap();
        values(kernels, &out).unwrap()
    }

    #[test]
    fn chunk_route_breaks_ties_to_the_lowest_index_and_ranks_pairs_in_token_order() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (_ckpt, weights) = load(&mut kernels);
        let x = chunk_input(&mut kernels, &[0, 1, 2, 3]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let moe_w = &weights.layers[0].moe;
        assert!(
            weights.shape.moe.renorm(),
            "the fixture renormalizes the gates"
        );
        let route = chunk_route(&mut ctx, &weights.shape.moe, moe_w, &x).unwrap();
        // Experts 1 and 2 tie in every token, and expert 1 wins. Token 2 ties all four experts
        // and takes experts 0 and 1. So expert 2 is never selected.
        assert_eq!(route.distinct, [0, 1, 3], "distinct experts");
        assert_eq!(route.top_k, 2, "top-k");
        // Token 0 selects experts 0, 1 (ranks 0, 1); token 1 selects 3, 1 (ranks 2, 1);
        // tokens 2 and 3 select 0, 1.
        assert_eq!(route.rank_idx, [0, 1, 2, 1, 0, 1, 0, 1], "ranks");
        // Gates: sigmoid(0.5) = 0.6224593 and 1 - sigmoid(0.5); sigmoid(1.5) = 0.8175745 and
        // 1 - sigmoid(1.5); a tie of two gives 0.5 each.
        let want = [
            0.622_459_3,
            0.377_540_7,
            0.817_574_5,
            0.182_425_5,
            0.5,
            0.5,
            0.622_459_3,
            0.377_540_7,
        ];
        assert_eq!(route.gates.len(), want.len(), "gate count");
        for (pair, (got, want)) in route.gates.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5,
                "gate {pair}: {got}, want {want}"
            );
        }
    }

    #[test]
    fn a_chunk_equals_one_decode_call_per_token_with_the_shared_expert() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let mut pool = pool_for(&ckpt, &weights, 4, 8);
        let (chunk, route) =
            run_chunk(&mut kernels, &weights, &ckpt, &mut pool, &[0, 1, 2, 3]).unwrap();
        assert_eq!(route.distinct, [0, 1, 3], "distinct experts");
        assert_eq!(chunk.len(), 4 * HIDDEN, "output size");
        let mut slots = decode_slots(&ckpt, &weights);
        for (t, row) in chunk.as_chunks::<HIDDEN>().0.iter().enumerate() {
            let want = decode_row(&mut kernels, &weights, &ckpt, &mut slots, t);
            assert_parity(row, &want, &format!("token {t} against its decode call"));
            assert_anchor(row, t, &format!("token {t} against the oracle"));
        }
    }

    /// Twelve token rows over the four test tokens make 24 pairs (the union is the experts 0, 1,
    /// and 3). Hand-derived pair counts: tokens 0, 2, and 3 take the experts 0 and 1 and token 1
    /// takes 3 and 1, so over the twelve rows (three copies of each token) rank 0 has 3 * 3 = 9
    /// pairs, rank 1 has 12 * 1 = 12, and rank 2 has 3 * 1 = 3, which the fixture route confirms
    /// (the `rank_idx` of `chunk_route_breaks_ties...` repeated three times). Cap 4 holds the union
    /// in one wave of those 24 pairs, which reaches the sorted kernel of the pinned MLX: 24 pairs
    /// are at least 16 and 24 / 3 = 8 pairs per expert are at least 4. Cap 2 splits the same pairs
    /// into the waves {0, 1} with 9 + 12 = 21 pairs over 2 experts (21 is at least 16 and 21 / 2 =
    /// 10 is at least 4, so this wave runs the sorted kernel) and {3} with 3 pairs (below 16, the
    /// unsorted kernel), so several waves, a sorted wave, and the inverse `take` run together.
    /// Every wave lists non-decreasing local ranks, and every output row matches its one-token
    /// decode block, which runs the unsorted `gather_qmm` of the decode path, within the parity
    /// tolerance, and the float64 anchors of the fixture.
    #[test]
    fn a_chunk_of_twenty_four_pairs_reaches_the_sorted_kernel_and_matches_the_decode_block() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let indices = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        for (cap, sizes) in [(4u32, &[24usize][..]), (2, &[21, 3])] {
            let mut pool = pool_for(&ckpt, &weights, cap, 24);
            let (rows, route) =
                run_chunk(&mut kernels, &weights, &ckpt, &mut pool, &indices).unwrap();
            assert_eq!(
                route.rank_idx.len(),
                24,
                "cap {cap}: 24 pairs of 12 tokens of top-k 2"
            );
            assert_eq!(route.distinct, [0, 1, 3], "cap {cap}: distinct experts");
            let plan = split_waves(&route, cap).unwrap();
            let got: Vec<usize> = plan.waves.iter().map(|wave| wave.pairs.len()).collect();
            assert_eq!(got, sizes, "cap {cap}: pairs per wave");
            for (wave, list) in plan.waves.iter().enumerate() {
                assert!(
                    list.local_ranks.is_sorted(),
                    "cap {cap}: wave {wave} lists non-decreasing local ranks"
                );
            }
            let mut slots = decode_slots(&ckpt, &weights);
            for (t, row) in rows.as_chunks::<HIDDEN>().0.iter().enumerate() {
                let want = decode_row(&mut kernels, &weights, &ckpt, &mut slots, indices[t]);
                assert_parity(
                    row,
                    &want,
                    &format!("cap {cap} token {t} against its decode call"),
                );
                assert_anchor(
                    row,
                    indices[t],
                    &format!("cap {cap} token {t} against the oracle"),
                );
            }
        }
    }

    /// Every byte of both sets starts as 0xFF (F32 NaN, U32 4,294,967,295). Call A uses set 0,
    /// call B uses set 1, and call C reuses set 0, which still holds the pages of call A with a
    /// different union, token count, and gates.
    #[test]
    fn poisoned_and_reused_pages_never_leak_into_the_output() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let mut pool = pool_for(&ckpt, &weights, 4, 8);
        pool.poison(0xFF);
        let calls: [&[usize]; 3] = [&[0, 1, 2, 3], &[2, 3], &[1]];
        for (call, tokens) in calls.into_iter().enumerate() {
            let (rows, _) = run_chunk(&mut kernels, &weights, &ckpt, &mut pool, tokens).unwrap();
            assert_eq!(
                rows.len(),
                tokens.len() * HIDDEN,
                "call {call}: output size"
            );
            for (row, &t) in rows.as_chunks::<HIDDEN>().0.iter().zip(tokens) {
                assert_anchor(row, t, &format!("call {call} token {t}"));
            }
        }
    }

    /// With one set, the second call finds the set of the first call released at its router
    /// readback but not yet flushed, so `acquire` fails. A release before the readback would
    /// free the set at that readback and the call would succeed.
    #[test]
    fn a_set_stays_lent_until_the_router_readback_and_one_more_flush() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let layers = weights.layers.iter().map(|layer| &layer.moe);
        let mut pool = PrefillPool::with_sets(&ckpt, layers, 4, 8, 1).unwrap();
        run_chunk(&mut kernels, &weights, &ckpt, &mut pool, &[0, 1]).unwrap();
        let second = run_chunk(&mut kernels, &weights, &ckpt, &mut pool, &[2]);
        assert!(
            matches!(
                &second,
                Err(EngError::Backend {
                    operation: "expert_slots",
                    ..
                })
            ),
            "{second:?}"
        );
    }

    /// The config of the tied fixture with `num_experts_per_tok` set to `top_k`.
    fn config_with_top_k(top_k: u32) -> Vec<u8> {
        let mut value: serde_json::Value = serde_json::from_slice(&config()).unwrap();
        value["text_config"]["num_experts_per_tok"] = serde_json::json!(top_k);
        serde_json::to_vec(&value).unwrap()
    }

    /// Loads the tied fixture with `top_k` choices per token.
    fn load_with_top_k(
        kernels: &mut MlxKernels,
        top_k: u32,
    ) -> (StreamedCheckpoint, ResidentWeights) {
        let ckpt = checkpoint_from(tied_entries());
        let flash = FlashWeights::load(&ckpt, &config_with_top_k(top_k)).unwrap();
        let weights = ResidentWeights::import(kernels, &flash).unwrap();
        (ckpt, weights)
    }

    /// Asserts that `got` and `want` have the same bits in every element. No wave size changes
    /// the value of a pair row, and the sum keeps the choice order, so a difference of one bit
    /// is a defect.
    fn assert_bitwise(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (position, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: element {position} is {g:e}, want {w:e}"
            );
        }
    }

    /// The union of tokens 0 to 3 is the experts 0, 1, and 3. The cap `c` splits them into
    /// `ceil(3 / c)` waves: cap 4 and cap 3 one wave, cap 2 the waves {0, 1} and {3}, and cap 1
    /// the waves {0}, {1}, and {3}. The single wave matches the oracle. Every split gives the
    /// same bits as the single wave: each pair row is one `take` row through `gather_qmm_sorted`,
    /// the join and the inverse `take` only move rows, and the sum runs in choice order. That
    /// bitwise equality holds while every wave holds fewer than 16 pairs, as the 8-pair waves of
    /// this route do, because the pinned MLX runs one kernel below that bound; a wave of 16 pairs
    /// or more takes the sorted kernel and its different tiling, so do not enlarge this fixture
    /// and read a bitwise failure as a defect.
    #[test]
    fn waves_of_every_size_equal_the_oracle_and_the_single_wave_result() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let mut single = pool_for(&ckpt, &weights, 4, 8);
        let (want, _) =
            run_chunk(&mut kernels, &weights, &ckpt, &mut single, &[0, 1, 2, 3]).unwrap();
        for (row, t) in want.as_chunks::<HIDDEN>().0.iter().zip(0..) {
            assert_anchor(row, t, &format!("single wave token {t}"));
        }
        for cap in [3, 2, 1] {
            let mut pool = pool_for(&ckpt, &weights, cap, 8);
            let (rows, route) =
                run_chunk(&mut kernels, &weights, &ckpt, &mut pool, &[0, 1, 2, 3]).unwrap();
            assert_eq!(route.distinct, [0, 1, 3], "cap {cap}: distinct experts");
            assert_bitwise(&rows, &want, &format!("cap {cap} against one wave"));
            for (row, t) in rows.as_chunks::<HIDDEN>().0.iter().zip(0..) {
                assert_anchor(row, t, &format!("cap {cap} token {t}"));
            }
        }
    }

    /// With three choices per token, `a + b + c` depends on the order of the sum, so a changed
    /// order shows in the last bits (two choices cannot show it, as `a + b` equals `b + a`).
    /// Tokens 0 and 3 select the experts 0, 1, 2, token 1 selects 3, 1, 2, and token 2 selects
    /// 0, 1, 2 (ties go to the lowest index). The union is 0, 1, 2, 3, so cap 4 is one wave, cap 3
    /// the waves {0, 1, 2} and {3}, cap 2 the waves {0, 1} and {2, 3}, and cap 1 four waves. The
    /// single wave matches one decode call per token within the parity tolerance. Every split
    /// gives the same bits as the single wave.
    #[test]
    fn waves_sum_three_choices_in_choice_order_for_every_wave_size() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load_with_top_k(&mut kernels, 3);
        let mut single = pool_for(&ckpt, &weights, 4, 12);
        let (want, route) =
            run_chunk(&mut kernels, &weights, &ckpt, &mut single, &[0, 1, 2, 3]).unwrap();
        assert_eq!(route.top_k, 3, "top-k");
        assert_eq!(route.distinct, [0, 1, 2, 3], "distinct experts");
        assert_eq!(
            route.rank_idx,
            [0, 1, 2, 3, 1, 2, 0, 1, 2, 0, 1, 2],
            "pair ranks"
        );
        let mut slots = decode_slots(&ckpt, &weights);
        for (t, row) in want.as_chunks::<HIDDEN>().0.iter().enumerate() {
            let decode = decode_row(&mut kernels, &weights, &ckpt, &mut slots, t);
            assert_parity(row, &decode, &format!("token {t} against its decode call"));
        }
        for cap in [3, 2, 1] {
            let mut pool = pool_for(&ckpt, &weights, cap, 12);
            let (rows, _) =
                run_chunk(&mut kernels, &weights, &ckpt, &mut pool, &[0, 1, 2, 3]).unwrap();
            assert_bitwise(&rows, &want, &format!("cap {cap} against one wave"));
        }
    }

    /// Cap 1 runs three waves per call and cap 1 with token 1 alone runs two waves ({1}, {3}).
    /// Every byte of both sets starts as 0xFF. The calls use the sets in the order A, B, A, then
    /// B, A, then B, A, B, so each wave after the first two reuses a set that an earlier wave of
    /// the same or the previous call used.
    #[test]
    fn waves_reuse_two_sets_across_waves_and_calls() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        let mut pool = pool_for(&ckpt, &weights, 1, 8);
        pool.poison(0xFF);
        let calls: [&[usize]; 3] = [&[0, 1, 2, 3], &[1], &[0, 1, 2, 3]];
        for (call, tokens) in calls.into_iter().enumerate() {
            let (rows, _) = run_chunk(&mut kernels, &weights, &ckpt, &mut pool, tokens).unwrap();
            assert_eq!(
                rows.len(),
                tokens.len() * HIDDEN,
                "call {call}: output size"
            );
            for (row, &t) in rows.as_chunks::<HIDDEN>().0.iter().zip(tokens) {
                assert_anchor(row, t, &format!("call {call} token {t}"));
            }
        }
    }

    /// A layer with `w` waves makes one router readback and `w - 1` evaluations, so `w`
    /// synchronizations: cap 4 and cap 3 one wave, cap 2 two waves, cap 1 three waves.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_chunk_synchronizes_once_per_wave_and_copies_no_pages() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, weights) = load(&mut kernels);
        for (cap, syncs) in [(4u32, 1u64), (3, 1), (2, 2), (1, 3)] {
            let mut pool = pool_for(&ckpt, &weights, cap, 8);
            let x = chunk_input(&mut kernels, &[0, 1, 2, 3]);
            let before = kernels.qualification_sync_count();
            let copies = kernels.qualification_page_copies();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let moe_w = &weights.layers[0].moe;
            let (out, _) =
                step_chunk(&mut ctx, &ckpt, &weights.shape.moe, moe_w, &mut pool, &x).unwrap();
            assert_eq!(
                ctx.kernels.qualification_sync_count(),
                before + syncs,
                "cap {cap}: one readback and one evaluation per extra wave"
            );
            assert_eq!(
                ctx.kernels.qualification_page_copies(),
                copies,
                "cap {cap}: no import took the copy path"
            );
            let host = values(&mut kernels, &out).unwrap();
            assert_eq!(host.len(), 4 * HIDDEN, "cap {cap}: output size");
            assert!(
                host.iter().all(|v| v.is_finite()),
                "cap {cap}: finite output"
            );
        }
    }
}
