//! Portable chunk planning of the Q36 prefill.
//!
//! The pair plan sorts the (token, choice) pairs of a chunk by expert. The chunk MoE gathers its
//! rows in that order and restores the token-major order with the inverse permutation. The ranges
//! split a prompt into chunks and a DeltaNet layer into sub-chunks. Nothing here calls the native
//! adapter, so the tests run on Linux.

use super::{super::settings::MIN_PREFILL_CHUNK_ROWS, kv_plan::SCORE_BUDGET_BYTES};
use crate::{EngError, family::route::Routed};

/// The reserve that `delta_subchunk` keeps below the retire cap, and the default cap of the
/// adapter: 64 MiB. The decoder's decode cap is `DECODE_RETIRE_CAP` (4 GiB).
pub(super) const RETIRE_CAP_BASE: u64 = 64 << 20;

/// The retire cap of the adapter while a prefill chunk runs: a flat 512 MiB. The adapter returns
/// to its previous cap after the chunk.
pub(super) const PREFILL_RETIRE_CAP: u64 = 512 << 20;

/// State-sized tensors that the unrolled DeltaNet loop drops per token and layer: the replaced
/// state, the decayed state, and the outer product.
const DELTA_RETIRED_STATES: u64 = 3;

/// The pairs of one chunk in expert order. Pair `p = t * top_k + j` is choice `j` of token `t`.
/// All four vectors have one entry per pair.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PairPlan {
    /// The expert of each sorted pair, ascending.
    pub(super) experts: Vec<u32>,
    /// The token row of each sorted pair.
    pub(super) tokens: Vec<u32>,
    /// The gate of each sorted pair.
    pub(super) gates: Vec<f32>,
    /// For each pair `p` in token-major order, its position in the sorted lists. Taking the sorted
    /// rows at these positions restores token-major order.
    pub(super) inverse: Vec<u32>,
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("q36 prefill plan: {message}"))
}

/// Sorts the pairs of `rows` by expert. Pairs of one expert keep their token-major order.
///
/// `rows` holds one `Routed` per token with exactly `top_k` choices. An expert at or above
/// `experts`, a different choice count, or an empty chunk returns `BadRequest`.
pub(super) fn pair_plan(
    rows: &[Routed],
    top_k: usize,
    experts: usize,
) -> Result<PairPlan, EngError> {
    if rows.is_empty() || top_k == 0 {
        return Err(bad("a chunk needs a row and a choice"));
    }
    let mut pairs: Vec<(usize, u32, usize, f32)> =
        Vec::with_capacity(rows.len().saturating_mul(top_k));
    for (token, row) in rows.iter().enumerate() {
        if row.indices.len() != top_k || row.weights.len() != top_k {
            return Err(bad("a row does not hold top-k choices"));
        }
        for (&expert, &gate) in row.indices.iter().zip(&row.weights) {
            if !usize::try_from(expert).is_ok_and(|e| e < experts) {
                return Err(bad("an expert is outside the stack"));
            }
            pairs.push((pairs.len(), expert, token, gate));
        }
    }
    // The sort is stable, so the pairs of one expert keep their token-major order.
    pairs.sort_by_key(|pair| pair.1);
    let mut plan = PairPlan {
        experts: Vec::with_capacity(pairs.len()),
        tokens: Vec::with_capacity(pairs.len()),
        gates: Vec::with_capacity(pairs.len()),
        inverse: vec![0; pairs.len()],
    };
    for (position, &(pair, expert, token, gate)) in pairs.iter().enumerate() {
        plan.experts.push(expert);
        plan.tokens
            .push(u32::try_from(token).map_err(|_| bad("token row exceeds u32"))?);
        plan.gates.push(gate);
        let slot = plan
            .inverse
            .get_mut(pair)
            .ok_or_else(|| bad("pair is missing"))?;
        *slot = u32::try_from(position).map_err(|_| bad("pair position exceeds u32"))?;
    }
    Ok(plan)
}

/// Proves on the host the order that `gather_qmm_sorted` requires: `experts` must be
/// non-decreasing. An inversion returns `BadRequest`; an empty slice passes. The caller is the
/// chunk MoE, which checks its pair plan before the import.
pub(super) fn require_sorted_experts(experts: &[u32]) -> Result<(), EngError> {
    if experts.is_sorted() {
        Ok(())
    } else {
        Err(EngError::BadRequest(
            "moe prefill: the expert pairs are not sorted".into(),
        ))
    }
}

/// Retired bytes of one chunk token in one DeltaNet layer: `3 * VH * KD * VD * 4` (F32 state).
/// A zero dimension or an overflow returns `BadRequest`.
pub(super) fn delta_retire_per_token(
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
) -> Result<u64, EngError> {
    u64::from(value_heads)
        .checked_mul(u64::from(key_dim))
        .and_then(|n| n.checked_mul(u64::from(value_dim)))
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_mul(DELTA_RETIRED_STATES))
        .filter(|&n| n > 0)
        .ok_or_else(|| bad("delta retire bytes are zero or overflow"))
}

/// Tokens per DeltaNet sub-chunk: `floor((cap - 64 MiB) / per_token)`, at least 1 and at most
/// `rows`. The layer state is evaluated between sub-chunks, so the retired bytes of one layer stay
/// near `cap`. `rows` and `per_token` must be positive.
pub(super) fn delta_subchunk(rows: u32, cap: u64, per_token: u64) -> Result<u32, EngError> {
    if rows == 0 || per_token == 0 {
        return Err(bad("a sub-chunk needs positive rows and bytes per token"));
    }
    let fit = cap.saturating_sub(RETIRE_CAP_BASE) / per_token;
    Ok(u32::try_from(fit).unwrap_or(u32::MAX).clamp(1, rows))
}

/// The chunk sizes that the admission rule tries, largest first (Q36 keeps its own copy of the
/// list). The largest is the most rows that `ResidentDecoder::set_chunk_rows` accepts
/// (`decoder::MAX_CHUNK_ROWS`). A compile-time check in `executor.rs` fails the build if the two
/// differ. A size of 1 is the one-token path and is never a chunk.
pub(super) const PREFILL_CANDIDATES: [u32; 11] = [
    2048,
    1024,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
    MIN_PREFILL_CHUNK_ROWS,
    1,
];

/// Activation bytes per chunk token outside the attention scores, a planning value [A]: the mixer
/// projections, the 8 routed pairs of each token (about 0.4 MB [D]), and the shared expert. The
/// real-checkpoint prefill test reports the workspace against the allocator peak of a real run.
const PLAN_PER_TOKEN: u64 = 1 << 20;

/// Attention score copies that live at once for each chunk token: the scores and their softmax.
const SCORE_COPIES: u64 = 2;

/// Chunk bytes that do not depend on the chunk size, a planning value [A]. The retire cap comes on top.
const PLAN_FIXED: u64 = 100_000_000;

/// The admission cost of a prefill chunk (the shared admission rule without the page sets: every
/// Q36 expert is resident). The workspace of `rows` rows is
/// `rows * activation_per_token + min(rows * score_per_token, score_cap) + fixed`.
///
/// The attention scores of a chunk grow with the chunk and with the context, but the prefill cuts
/// a chunk at great depth into parts whose score matrix stays within `SCORE_BUDGET_BYTES`
/// (`kv_plan::depth_ranges`), so the score term never passes `SCORE_COPIES` times that budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkCost {
    /// Bytes for each chunk token if the score rows are not capped: the planning activations and
    /// the score rows at the context limit. A report prints it.
    pub(super) per_token: u64,
    /// Bytes for each chunk token outside the attention scores.
    pub(super) activation_per_token: u64,
    /// Bytes of the score rows for each chunk token at the context limit.
    pub(super) score_per_token: u64,
    /// The most bytes the score term takes, whatever the chunk size.
    pub(super) score_cap: u64,
    /// Bytes that do not depend on the chunk size, including the retire cap of a chunk.
    pub(super) fixed: u64,
}

impl ChunkCost {
    /// The cost on a model with `heads_q` query heads and a context of `positions` positions. A
    /// score row covers every position, so the plan holds for the longest context.
    pub(super) fn new(heads_q: u64, positions: u32) -> Result<Self, EngError> {
        let overflow = || EngError::MemoryPlan("q36 prefill cost overflows u64".into());
        let scores = SCORE_COPIES
            .checked_mul(heads_q)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_mul(u64::from(positions)))
            .ok_or_else(overflow)?;
        Ok(Self {
            per_token: PLAN_PER_TOKEN.checked_add(scores).ok_or_else(overflow)?,
            activation_per_token: PLAN_PER_TOKEN,
            score_per_token: scores,
            score_cap: SCORE_COPIES
                .checked_mul(SCORE_BUDGET_BYTES)
                .ok_or_else(overflow)?,
            fixed: PLAN_FIXED
                .checked_add(PREFILL_RETIRE_CAP)
                .ok_or_else(overflow)?,
        })
    }

    /// Workspace bytes of a chunk of `rows` rows.
    pub(super) fn workspace(&self, rows: u32) -> Result<u64, EngError> {
        let overflow = || EngError::MemoryPlan("q36 prefill workspace overflows u64".into());
        let rows = u64::from(rows);
        let activations = rows
            .checked_mul(self.activation_per_token)
            .ok_or_else(overflow)?;
        let scores = rows
            .checked_mul(self.score_per_token)
            .ok_or_else(overflow)?
            .min(self.score_cap);
        activations
            .checked_add(scores)
            .and_then(|n| n.checked_add(self.fixed))
            .ok_or_else(overflow)
    }
}

/// The terms that one admission charged, in bytes. `Display` gives one `name=bytes` line per term
/// and ends with `remaining_bytes`, so a report can print it unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillTerms {
    /// The engine memory budget.
    pub(crate) budget: u64,
    /// The resident weights of the executor.
    pub(crate) weights: u64,
    /// The session state, the MLX cache limit, and the pressure margin, as the caller summed them.
    pub(crate) session_reserve: u64,
}

impl PrefillTerms {
    /// The bytes left for a chunk: the budget minus the weights and the reserve, never below zero.
    pub(crate) fn remaining(&self) -> u64 {
        self.budget
            .saturating_sub(self.weights)
            .saturating_sub(self.session_reserve)
    }
}

impl std::fmt::Display for PrefillTerms {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(out, "budget_bytes={}", self.budget)?;
        writeln!(out, "weights_bytes={}", self.weights)?;
        writeln!(out, "session_reserve_bytes={}", self.session_reserve)?;
        writeln!(out, "remaining_bytes={}", self.remaining())
    }
}

/// The result of one admission: the chunk size and the terms that were charged for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillPlan {
    /// The chunk size, or `None` when no size above one token fits and the prefill runs one-token steps.
    pub(crate) chunk: Option<u32>,
    /// The terms that were charged.
    pub(crate) terms: PrefillTerms,
}

/// The largest candidate of at most `max_rows` rows whose workspace fits `remaining`, or `None`
/// when only the one-token path fits. `max_rows` is the largest worker piece: a call never feeds
/// more rows than that, so a larger chunk would not run. A `max_rows` of zero returns `BadRequest`.
pub(super) fn admitted_chunk(
    cost: &ChunkCost,
    remaining: u64,
    max_rows: u32,
) -> Result<Option<u32>, EngError> {
    if max_rows == 0 {
        return Err(bad("the largest chunk must exceed zero rows"));
    }
    for rows in PREFILL_CANDIDATES
        .into_iter()
        .filter(|&rows| rows > 1 && rows <= max_rows)
    {
        if cost.workspace(rows)? <= remaining {
            return Ok(Some(rows));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routed(choices: &[(u32, f32)]) -> Routed {
        Routed {
            indices: choices.iter().map(|c| c.0).collect(),
            weights: choices.iter().map(|c| c.1).collect(),
        }
    }

    /// Three tokens, top 2, four experts. Pairs in token-major order:
    /// p0 (2, t0, .6), p1 (0, t0, .4), p2 (1, t1, .5), p3 (2, t1, .5), p4 (0, t2, .7), p5 (1, t2, .3).
    /// By expert, ties in token-major order: expert 0 holds p1 and p4, expert 1 holds p2 and p5,
    /// expert 2 holds p0 and p3. The sorted order is p1 p4 p2 p5 p0 p3, so p0 sits at position 4,
    /// p1 at 0, p2 at 2, p3 at 5, p4 at 1, and p5 at 3.
    #[test]
    fn pairs_sort_by_expert_and_ties_keep_token_order() {
        let rows = [
            routed(&[(2, 0.6), (0, 0.4)]),
            routed(&[(1, 0.5), (2, 0.5)]),
            routed(&[(0, 0.7), (1, 0.3)]),
        ];
        let plan = pair_plan(&rows, 2, 4).unwrap();
        assert_eq!(plan.experts, [0, 0, 1, 1, 2, 2]);
        assert_eq!(plan.tokens, [0, 2, 1, 2, 0, 1]);
        assert_eq!(plan.gates, [0.4, 0.7, 0.5, 0.3, 0.6, 0.5]);
        assert_eq!(plan.inverse, [4, 0, 2, 5, 1, 3]);
    }

    /// 64 tokens, top 8 of 256 experts from a fixed congruential sequence with repeats across
    /// tokens. The sorted experts ascend, the tokens ascend inside each expert, and the inverse
    /// restores the token-major experts and gates.
    #[test]
    fn the_inverse_restores_token_major_order_at_real_sizes() {
        let mut seed = 12_345u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (seed >> 8) % 256
        };
        let (top_k, count) = (8usize, 64usize);
        let rows: Vec<Routed> = (0..count)
            .map(|t| {
                let mut chosen: Vec<u32> = Vec::new();
                while chosen.len() < top_k {
                    let e = next();
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                Routed {
                    weights: (0..top_k).map(|j| (t * 8 + j) as f32).collect(),
                    indices: chosen,
                }
            })
            .collect();
        let plan = pair_plan(&rows, top_k, 256).unwrap();
        assert_eq!(plan.experts.len(), count * top_k);
        for at in 1..plan.experts.len() {
            let (e0, e1) = (plan.experts[at - 1], plan.experts[at]);
            let (t0, t1) = (plan.tokens[at - 1], plan.tokens[at]);
            assert!(e0 <= e1, "experts ascend at {at}");
            assert!(e0 < e1 || t0 < t1, "tokens ascend inside expert at {at}");
        }
        for (t, row) in rows.iter().enumerate() {
            for j in 0..top_k {
                let sorted = plan.inverse[t * top_k + j] as usize;
                assert_eq!(plan.experts[sorted], row.indices[j], "expert of ({t}, {j})");
                assert_eq!(plan.tokens[sorted] as usize, t, "token of ({t}, {j})");
                assert_eq!(plan.gates[sorted], row.weights[j], "gate of ({t}, {j})");
            }
        }
    }

    /// The order check of the sorted prefill gather, next to the plan that must satisfy it: a
    /// sorted slice passes, the empty slice passes, and one inversion is a bad request that
    /// names the moe prefill.
    #[test]
    fn the_order_check_passes_sorted_and_empty_and_fails_one_inversion() {
        assert!(
            require_sorted_experts(&[0, 1, 1, 3, 5]).is_ok(),
            "a sorted slice passes"
        );
        assert!(
            require_sorted_experts(&[]).is_ok(),
            "the empty slice passes"
        );
        let failed = require_sorted_experts(&[0, 2, 1]);
        assert!(
            matches!(&failed, Err(EngError::BadRequest(m)) if m.starts_with("moe prefill:")),
            "one inversion is a bad request that names the moe prefill, got {:?}",
            failed
        );
    }

    /// Two tokens of top-2 routes over four experts. In token-major order the pairs hold the
    /// experts [2, 1, 0, 3]; the plan that `chunk` imports must present them as [0, 1, 2, 3],
    /// the order that `gather_qmm_sorted` requires.
    #[test]
    fn the_imported_pair_plan_satisfies_the_order_check() {
        let rows = [routed(&[(2, 0.6), (1, 0.4)]), routed(&[(0, 0.7), (3, 0.3)])];
        let plan = pair_plan(&rows, 2, 4).unwrap();
        assert_eq!(plan.experts, [0, 1, 2, 3]);
        assert!(require_sorted_experts(&plan.experts).is_ok());
    }

    #[test]
    fn a_bad_route_is_rejected() {
        let good = routed(&[(1, 0.5), (2, 0.5)]);
        let cases = [
            ("empty chunk", pair_plan(&[], 2, 4)),
            ("zero top-k", pair_plan(std::slice::from_ref(&good), 0, 4)),
            (
                "one choice too many",
                pair_plan(&[routed(&[(1, 0.4), (2, 0.3), (3, 0.3)])], 2, 4),
            ),
            (
                "one weight too few",
                pair_plan(
                    &[Routed {
                        indices: vec![1, 2],
                        weights: vec![1.0],
                    }],
                    2,
                    4,
                ),
            ),
            (
                "expert at the stack size",
                pair_plan(&[routed(&[(1, 0.5), (4, 0.5)])], 2, 4),
            ),
        ];
        for (what, result) in cases {
            assert!(
                matches!(&result, Err(EngError::BadRequest(m)) if m.contains("q36 prefill plan")),
                "{what}: {result:?}"
            );
        }
    }

    /// Real dimensions: 3 states of 32 * 128 * 128 * 4 B = 6,291,456 B per token. The room above the
    /// 64 MiB base under the 512 MiB cap is 469,762,048 B, which holds 74 tokens (74.67 floored).
    #[test]
    fn the_delta_sub_chunk_follows_the_retire_room() {
        assert_eq!(delta_retire_per_token(32, 128, 128).unwrap(), 6_291_456);
        assert_eq!(
            PREFILL_RETIRE_CAP - RETIRE_CAP_BASE,
            469_762_048,
            "room above the base"
        );
        assert_eq!(
            delta_subchunk(512, PREFILL_RETIRE_CAP, 6_291_456).unwrap(),
            74
        );
        assert_eq!(
            delta_subchunk(50, PREFILL_RETIRE_CAP, 6_291_456).unwrap(),
            50
        );
        assert_eq!(delta_subchunk(512, RETIRE_CAP_BASE, 6_291_456).unwrap(), 1);
        assert_eq!(delta_subchunk(512, 0, 6_291_456).unwrap(), 1);
        assert!(delta_subchunk(0, PREFILL_RETIRE_CAP, 1).is_err());
        assert!(delta_subchunk(1, PREFILL_RETIRE_CAP, 0).is_err());
        assert!(delta_retire_per_token(0, 128, 128).is_err());
        assert!(delta_retire_per_token(u32::MAX, u32::MAX, u32::MAX).is_err());
    }

    /// The real shape (16 query heads) at the shipped limit of 32,768 positions and at 262,144:
    /// the score rows are 4,194,304 and 33,554,432 bytes per token. By hand, with 1 MiB of
    /// activations per token and `fixed` = 100,000,000 + 536,870,912 = 636,870,912:
    /// 256 rows at 32,768 take 268,435,456 + 1,073,741,824 + 636,870,912 = 1,979,048,192, and
    /// the score term sits exactly at its cap of 2 * 512 MiB; at 262,144 the uncapped term
    /// would be 8,589,934,592, and the cap holds the workspace at the same 1,979,048,192.
    #[test]
    fn the_score_term_of_a_chunk_is_capped_at_the_depth_budget() {
        let shipped = ChunkCost::new(16, 32_768).unwrap();
        let deep = ChunkCost::new(16, 262_144).unwrap();
        assert_eq!(shipped.score_per_token, 4_194_304);
        assert_eq!(deep.score_per_token, 33_554_432);
        assert_eq!(deep.score_cap, 1_073_741_824);
        assert_eq!(deep.fixed, 636_870_912);
        assert_eq!(shipped.workspace(256).unwrap(), 1_979_048_192);
        assert_eq!(deep.workspace(256).unwrap(), 1_979_048_192);
        // 64 rows at 262,144 positions: the cap binds (2,147,483,648 uncapped).
        assert_eq!(deep.workspace(64).unwrap(), 1_777_721_600);
        // 16 rows: below the cap (536,870,912), so the term is linear.
        assert_eq!(deep.workspace(16).unwrap(), 1_190_519_040);
        // The shipped plan is unchanged: at 32,768 positions the cap is where 256 rows end.
        assert_eq!(
            shipped.workspace(256).unwrap(),
            256 * shipped.per_token + shipped.fixed
        );
    }

    /// With the cap, a 262,144-position limit admits 256 rows when 1,979,048,192 bytes remain,
    /// and 128 rows (1,844,830,464 bytes) with one byte less. Before the cap the 256-row
    /// workspace was 9.5 GB and the same remainder admitted 32 rows at best.
    #[test]
    fn a_deep_context_limit_still_admits_a_full_chunk() {
        let deep = ChunkCost::new(16, 262_144).unwrap();
        assert_eq!(
            admitted_chunk(&deep, 1_979_048_192, 256).unwrap(),
            Some(256)
        );
        assert_eq!(
            admitted_chunk(&deep, 1_979_048_191, 256).unwrap(),
            Some(128)
        );
        assert_eq!(deep.workspace(128).unwrap(), 1_844_830_464);
        assert_eq!(admitted_chunk(&deep, 1_190_519_040, 256).unwrap(), Some(16));
    }
}
