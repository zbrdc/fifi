//! Resident QSA step (S1-C plan section 2.3; decision 0039 item 6).
//!
//! A step appends one position and attends to the positions that the host rule selects
//! (`flash_next/qsa.rs` lines 279-319). While every complete indexer block is selected (dense),
//! attention runs over the whole cache and the indexer scoring is skipped. Beyond that (sparse),
//! the graph scores every complete block, ranks the blocks with a stable sort, and builds an
//! additive mask that hides the dropped blocks. Attention then runs over the whole cache with that
//! mask. The mask needs no gather of the cache. Selection stays on the device and adds no
//! synchronization. The raw indexer keys are appended in both cases.
//!
//! Retired bytes. A layer's step drops its old cache and its temporary tensors before the router
//! readback of the same layer, which is the next synchronization (decision 0039). The retire list
//! counts the full `byte_len` of every dropped tensor, so a view (slice, reshape, transpose) counts
//! as much as a copy. The table gives the bytes per position of the real model (KVH 2, D 256, QH 24,
//! ID 128, IQ 4, ratio 4, F32). `T` is the visible positions, and `blocks` is about `T / 4`.
//!
//! | Tensor dropped in one layer step | Bytes per position |
//! | --- | --- |
//! | Old keys and values `[1, 2, T, 256]`, two tensors | 2 * 2 * 256 * 4 = 4,096 |
//! | Old raw indexer keys `[T, 128]` | 128 * 4 = 512 |
//! | Dense total (`grouped_attention` has no `T`-sized temporary) | 4,608 |
//! | Sparse: `slice(ik)` and its reshape, `[tail, 128]` each | 2 * 512 = 1,024 |
//! | Sparse: five pooled-key tensors, `[blocks, 128]` each (pool, norm, reshape, rope, reshape) | 5 * 128 = 640 |
//! | Sparse: attention logits `[1, 2, T, 12]`: product, transpose, add mask, softmax | 4 * 96 = 384 |
//! | Sparse: mask `[tail]` and `[T]` | 2 * 4 = 8 |
//! | Sparse: dot products and relu `[blocks, 4]`, then eight `[blocks]` vectors | 2 * 4 + 8 = 16 |
//! | Sparse total | 4,608 + 2,072 = 6,680 |
//!
//! The retire cap is 64 MiB (67,108,864 bytes). A dense step passes it at about 14,560 positions.
//! A sparse step passes it at about 10,046 positions (67,108,864 / 6,680), because the mask design
//! adds about 2 KB per position to the dense figure. Above that position, each QSA
//! layer step forces at least one extra retire flush synchronization, and a longer context adds
//! more of them. The synchronization budget of `L + 2` (`Prompt`), `L + 3` (`Generate`), and
//! `L + 5` (`Record`) holds only below that position. The model test checks the budget at 2,070
//! positions, where one layer step retires about 14 MB.

use super::super::qsa::{QsaDims, QsaShape};
use super::ops::{Ctx, dim, linear};
use super::state::LayerState;
use super::weights::RQsa;
use crate::{
    EngError,
    kernels::{BinaryOp, ReduceOp, RopeSpec, UnaryOp},
    tensor::ExecutionTensor,
};

/// The largest context of the resident QSA path: the 32,768 positions of the memory table in the
/// S1-C plan. A step at this position or beyond returns `ContextLimit`. The load-time block
/// constants cover exactly this many positions.
pub(super) const MAX_POSITIONS: u32 = 32_768;

/// The additive attention mask of a dropped block. `exp` of it underflows to 0 in F32.
pub(super) const MASK_OFF: f32 = -1.0e30;

/// One resident QSA step: the layer output and the replacement layer state.
pub(super) struct QsaStep {
    /// Layer output `[1, H]`.
    pub(super) out: ExecutionTensor,
    /// `LayerState::Qsa` with one more position. The caller replaces the old state with it.
    pub(super) state: LayerState,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident qsa dimension overflows".into())
}

fn mismatch() -> EngError {
    EngError::BadRequest("resident qsa cache does not match its shape".into())
}

/// The positions that the step at `pos` attends to, by the host rule (qsa.rs:286-290).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    /// Positions visible to the step, including the new one: `pos + 1`.
    visible: u32,
    /// Complete indexer blocks: `visible / ratio`.
    blocks: u32,
    /// Positions inside complete blocks: `blocks * ratio`. The positions from `tail` on always attend.
    tail: u32,
    /// Blocks that attend: `min(blocks, budget / ratio)`.
    take: u32,
}

impl Plan {
    /// Computes the plan with checked arithmetic. `QsaShape::new` guarantees a nonzero ratio.
    fn new(dims: &QsaDims, pos: u32) -> Result<Self, EngError> {
        let visible = pos.checked_add(1).ok_or_else(overflow)?;
        let blocks = visible.checked_div(dims.ratio).ok_or_else(overflow)?;
        let tail = blocks.checked_mul(dims.ratio).ok_or_else(overflow)?;
        let limit = dims.budget.checked_div(dims.ratio).ok_or_else(overflow)?;
        Ok(Self {
            visible,
            blocks,
            tail,
            take: blocks.min(limit),
        })
    }

    /// True when some complete block does not attend. Otherwise every position attends (dense).
    fn sparse(&self) -> bool {
        self.take < self.blocks
    }
}

/// Checks, before any native call, that the cache holds `pos` positions (qsa.rs:202-207) and that
/// `pos` is inside the context of the load-time block constants.
pub(super) fn check_cache(len: usize, pos: u32) -> Result<(), EngError> {
    if usize::try_from(pos).ok() != Some(len) {
        return Err(EngError::BadRequest(format!(
            "qsa cache length {len} does not match position {pos}"
        )));
    }
    if pos >= MAX_POSITIONS {
        return Err(EngError::ContextLimit(format!(
            "resident QSA supports {MAX_POSITIONS} positions"
        )));
    }
    Ok(())
}

/// The grown caches of one layer: keys and values `[1, KVH, T, D]`, raw indexer keys `[T, ID]`.
struct Kv<'a> {
    k: &'a ExecutionTensor,
    v: &'a ExecutionTensor,
    ik: &'a ExecutionTensor,
}

/// What block selection needs besides the caches: the plan, the position, and the indexer
/// projection `[1, (IQ + 1) * ID]` of the new input.
struct Selection<'a> {
    plan: Plan,
    pos: u32,
    ip: &'a ExecutionTensor,
}

/// Scores every complete block as the host does (qsa.rs:279-308): pool the raw indexer keys of the
/// block, normalize and rotate the pooled key at the block start, and sum the rectified dot
/// products with the rotated indexer queries. The host divides the sum by `sqrt(ID)`. This graph
/// does not: a positive factor keeps the order and the ties. Returns `[blocks]`.
fn block_scores(
    ctx: &mut Ctx<'_>,
    shape: &QsaShape,
    w: &RQsa,
    sel: &Selection<'_>,
    ik: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let d = shape.dims();
    let (heads, width) = (d.idx_heads, d.idx_dim);
    let plan = &sel.plan;
    let spec = |scale: f32, offset: u32| RopeSpec {
        dims: d.rotary_dim,
        base: d.theta,
        scale,
        offset,
    };
    // Pooled keys: the mean of the `ratio` rows of each block.
    let rows = ctx.kernels.slice(ik, &[0, 0], &[plan.tail, width])?;
    let rows = ctx.kernels.reshape(&rows, &[plan.blocks, d.ratio, width])?;
    let pool = ctx.kernels.reduce(ReduceOp::Mean, &rows, 1, false)?;
    let pk = ctx.kernels.rms_norm(&pool, &w.idx_k_norm1p, d.eps)?;
    let pk = ctx.kernels.reshape(&pk, &[1, plan.blocks, width])?;
    // Block `b` starts at position `ratio * b`, so the rotation scale is `ratio` from offset 0.
    let pk = ctx.kernels.rope_scaled(&pk, &spec(d.ratio as f32, 0))?;
    let pk = ctx.kernels.reshape(&pk, &[plan.blocks, width])?;
    // Indexer queries: the first `IQ * ID` values of the projection, one row per head.
    let stop = heads.checked_mul(width).ok_or_else(overflow)?;
    let qi = ctx.kernels.slice(sel.ip, &[0, 0], &[1, stop])?;
    let qi = ctx.kernels.reshape(&qi, &[heads, width])?;
    let qi = ctx.kernels.rms_norm(&qi, &w.idx_q_norm1p, d.eps)?;
    let qi = ctx.kernels.reshape(&qi, &[heads, 1, width])?;
    let qi = ctx.kernels.rope_scaled(&qi, &spec(1.0, sel.pos))?;
    let qi = ctx.kernels.reshape(&qi, &[heads, width])?;
    let qi = ctx.kernels.transpose(&qi, &[1, 0])?;
    let dots = ctx.kernels.mm(&pk, &qi)?;
    let relu = ctx
        .kernels
        .binary(BinaryOp::Maximum, &dots, &ctx.consts.zero)?;
    ctx.kernels.reduce(ReduceOp::Sum, &relu, 1, false)
}

/// Returns the block indices `[blocks]` (U32) in descending score order. Equal scores keep the
/// lower block index, as the host sort does (qsa.rs:312). `argsort` is stable.
fn rank_blocks(ctx: &mut Ctx<'_>, scores: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
    let negated = ctx.kernels.unary(UnaryOp::Neg, scores)?;
    ctx.kernels.argsort(&negated, 0)
}

/// The additive attention mask `[visible]` of a sparse step: 0 for the positions of the top blocks
/// and of the tail, `MASK_OFF` for the positions of the dropped blocks (qsa.rs:313-319).
/// `ratio` is the positions per block.
///
/// `rank_blocks` gives the block of each rank. The mask needs the rank of each block, which is the
/// inverse permutation. Only F32 inputs sort, so the order becomes F32 through `take` on the
/// load-time block numbers, and a second `argsort` inverts it. The load-time `rank_keep` vector is
/// 0 for the ranks that attend and `MASK_OFF` for the others.
fn block_mask(
    ctx: &mut Ctx<'_>,
    scores: &ExecutionTensor,
    plan: &Plan,
    ratio: u32,
) -> Result<ExecutionTensor, EngError> {
    let order = rank_blocks(ctx, scores)?;
    let ids = ctx
        .kernels
        .slice(&ctx.consts.block_ids, &[0], &[plan.blocks])?;
    let order_f = ctx.kernels.take(&ids, &order, 0)?;
    let rank = ctx.kernels.argsort(&order_f, 0)?;
    let keep = ctx
        .kernels
        .slice(&ctx.consts.rank_keep, &[0], &[plan.blocks])?;
    let keep = ctx.kernels.take(&keep, &rank, 0)?;
    let mask = ctx.kernels.repeat(&keep, ratio, 0)?;
    if plan.tail == plan.visible {
        return Ok(mask);
    }
    let one = ctx.kernels.expand_dims(&ctx.consts.zero, 0)?;
    let rest_len = plan.visible.checked_sub(plan.tail).ok_or_else(|| {
        EngError::BadRequest(format!(
            "resident qsa tail {} is past the visible positions {}",
            plan.tail, plan.visible
        ))
    })?;
    let rest = ctx.kernels.repeat(&one, rest_len, 0)?;
    ctx.kernels.concatenate(&[&mask, &rest], 0)
}

/// Sparse attention: score, rank, mask, then attention over the whole cache with the mask.
/// The queries `q` are `[1, QH, 1, D]`. The result has `QH * D` values (`[1, KVH, QH / KVH, D]`).
/// The graph holds no readback. The fused `grouped_attention` takes no mask, so this uses
/// matrix products: the keys stay the left operand, so no cache copy is made.
fn attend_sparse(
    ctx: &mut Ctx<'_>,
    shape: &QsaShape,
    w: &RQsa,
    sel: &Selection<'_>,
    q: &ExecutionTensor,
    kv: &Kv<'_>,
) -> Result<ExecutionTensor, EngError> {
    let d = shape.dims();
    let (kv_heads, width) = (d.heads_kv, d.head_dim);
    let rep = d
        .heads_q
        .checked_div(kv_heads)
        .ok_or_else(|| EngError::BadRequest("resident qsa key-value head count is zero".into()))?;
    let scores = block_scores(ctx, shape, w, sel, kv.ik)?;
    let mask = block_mask(ctx, &scores, &sel.plan, d.ratio)?;
    // Head h reads KV head h / rep, so the heads reshape to [KVH, rep]. The scale folds into q.
    let q = ctx
        .kernels
        .binary(BinaryOp::Mul, q, &ctx.consts.qsa_scale)?;
    let q = ctx.kernels.reshape(&q, &[1, kv_heads, rep, width])?;
    let q = ctx.kernels.transpose(&q, &[0, 1, 3, 2])?;
    let logits = ctx.kernels.batched_mm(kv.k, &q)?;
    let logits = ctx.kernels.transpose(&logits, &[0, 1, 3, 2])?;
    let logits = ctx.kernels.binary(BinaryOp::Add, &logits, &mask)?;
    let probs = ctx.kernels.softmax(&logits)?;
    ctx.kernels.batched_mm(&probs, kv.v)
}

/// Runs one QSA decode step at `pos` on `x` (`H` values, any shape). It builds a lazy graph
/// only: no readback and no evaluate. `state` is never changed; on error nothing is returned.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &QsaShape,
    w: &RQsa,
    state: &LayerState,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<QsaStep, EngError> {
    let LayerState::Qsa { k, v, ik, len } = state else {
        return Err(EngError::BadRequest(
            "qsa step needs a QSA layer state".into(),
        ));
    };
    check_cache(*len, pos)?;
    let d = shape.dims();
    let plan = Plan::new(&d, pos)?;
    let (heads, kv_heads, width) = (d.heads_q, d.heads_kv, d.head_dim);
    let t = dim(*len)?;
    let cache = match (k, v, ik) {
        (None, None, None) if *len == 0 => None,
        (Some(k), Some(v), Some(ik)) if *len > 0 => {
            let kv_shape = [1, kv_heads, t, width];
            let ik_shape = [t, d.idx_dim];
            if k.shape() != kv_shape.as_slice()
                || v.shape() != kv_shape.as_slice()
                || ik.shape() != ik_shape.as_slice()
            {
                return Err(mismatch());
            }
            Some((k, v, ik))
        }
        _ => return Err(mismatch()),
    };
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let idx_start = d.idx_heads.checked_mul(d.idx_dim).ok_or_else(overflow)?;
    let idx_stop = idx_start.checked_add(d.idx_dim).ok_or_else(overflow)?;
    let next_len = len.checked_add(1).ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: d.rotary_dim,
        base: d.theta,
        scale: 1.0,
        offset: pos,
    };

    // 1. Query and gate, interleaved per head: head h is qp[h * 2D..(h + 1) * 2D] (qsa.rs:325, :345).
    let qp = linear(ctx, &w.q, x)?;
    let qp = ctx.kernels.reshape(&qp, &[heads, pair])?;
    let q_raw = ctx.kernels.slice(&qp, &[0, 0], &[heads, width])?;
    let gate = ctx.kernels.slice(&qp, &[0, width], &[heads, pair])?;
    // 2. Query norm with `1 + w`, then RoPE at `pos` (qsa.rs:326-327).
    let q = ctx.kernels.rms_norm(&q_raw, &w.q_norm1p, d.eps)?;
    let q = ctx.kernels.reshape(&q, &[1, heads, 1, width])?;
    let q = ctx.kernels.rope_scaled(&q, &rope)?;
    // 3. New key: norm and RoPE per KV head (qsa.rs:224, :231-236).
    let kp = linear(ctx, &w.k, x)?;
    let kp = ctx.kernels.reshape(&kp, &[kv_heads, width])?;
    let k_new = ctx.kernels.rms_norm(&kp, &w.k_norm1p, d.eps)?;
    let k_new = ctx.kernels.reshape(&k_new, &[1, kv_heads, 1, width])?;
    let k_new = ctx.kernels.rope_scaled(&k_new, &rope)?;
    // 4. New value, and the raw indexer key: the last `ID` projection values (qsa.rs:226, :228, :241).
    let v_new = linear(ctx, &w.v, x)?;
    let v_new = ctx.kernels.reshape(&v_new, &[1, kv_heads, 1, width])?;
    let ip = linear(ctx, &w.idx, x)?;
    let ik_new = ctx.kernels.slice(&ip, &[0, idx_start], &[1, idx_stop])?;
    // 5. Append on the time axis, or start the cache (qsa.rs:238-242).
    let (k_all, v_all, ik_all) = match cache {
        None => (k_new, v_new, ik_new),
        Some((k, v, ik)) => (
            ctx.kernels.concatenate(&[k, &k_new], 2)?,
            ctx.kernels.concatenate(&[v, &v_new], 2)?,
            ctx.kernels.concatenate(&[ik, &ik_new], 0)?,
        ),
    };
    // 6. Attention (qsa.rs:316-343). Head h reads KV head h / (QH / KVH), the same grouping as SDPA.
    // Dense: every position attends. Sparse: the top blocks and the tail attend.
    let att = if plan.sparse() {
        let sel = Selection { plan, pos, ip: &ip };
        let kv = Kv {
            k: &k_all,
            v: &v_all,
            ik: &ik_all,
        };
        attend_sparse(ctx, shape, w, &sel, &q, &kv)?
    } else {
        let scale = 1.0 / (width as f32).sqrt();
        ctx.kernels
            .grouped_attention(&q, &k_all, &v_all, scale, false)?
    };
    // 7. Sigmoid output gate and output projection (qsa.rs:344-350).
    let att = ctx.kernels.reshape(&att, &[heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    let out = linear(ctx, &w.o, &o)?;
    Ok(QsaStep {
        out,
        state: LayerState::Qsa {
            k: Some(k_all),
            v: Some(v_all),
            ik: Some(ik_all),
            len: next_len,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::load::FlashWeights;
    use super::super::super::math::{rms0, rope_half};
    use super::super::super::mlx::MlxProjector;
    use super::super::super::mlx::tests::{
        COS_BLOCK, COS_END, Gate, REL_BLOCK, REL_END, env_path, open_checkpoint, top5, write_report,
    };
    use super::super::super::qsa::{self as host, QsaCache};
    use super::super::super::step::{AttnWeights, ModelState, step as host_step};
    use super::super::fixtures;
    use super::super::ops::{f32_tensor, values, zeros};
    use super::super::weights::{RAttn, ResidentWeights, synthetic};
    use super::super::{ResidentDecoder, StepMode};
    use super::*;
    use crate::family::project::Projector;
    use crate::kernels::MlxKernels;
    use crate::tensor::Dtype;

    /// Input of step `pos`: `x[i] = ((i + 5 pos) % 11 - 5) / 16`, exact in f32.
    fn input(pos: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((i + 5 * pos) % 11) as f32 - 5.0) * 0.0625)
            .collect()
    }

    /// Asserts cos >= 0.99999 and max |a - e| / max |e| <= 1e-4 (plan section 4, option B).
    fn assert_parity(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        let (mut dot, mut aa, mut ee, mut diff, mut top) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, &e) in actual.iter().zip(expected) {
            let (a, e) = (f64::from(a), f64::from(e));
            dot += a * e;
            aa += a * a;
            ee += e * e;
            diff = diff.max((a - e).abs());
            top = top.max(e.abs());
        }
        let cos = dot / (aa.sqrt() * ee.sqrt());
        assert!(cos >= 0.99999, "{what}: cos {cos}");
        assert!(diff / top <= 1e-4, "{what}: relative error {}", diff / top);
    }

    /// A QSA state of `len` zero positions in the resident layouts (fixture KVH 1, D 64, ID 32).
    fn zero_cache(kernels: &mut MlxKernels, len: u32) -> LayerState {
        let kv = zeros(vec![1, 1, len, 64]).unwrap();
        let ik = zeros(vec![len, 32]).unwrap();
        LayerState::Qsa {
            k: Some(kernels.import_tensor(&kv).unwrap()),
            v: Some(kernels.import_tensor(&kv).unwrap()),
            ik: Some(kernels.import_tensor(&ik).unwrap()),
            len: len as usize,
        }
    }

    #[test]
    fn dense_and_sparse_steps_match_the_host_block() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        let view = flash.view();
        let AttnWeights::Qsa(hq) = &view.layers[1].attn else {
            panic!("host layer 1 is not QSA");
        };
        let qshape = flash.shape().qsa;
        let dims = qshape.dims();
        assert_eq!(
            (dims.heads_kv, dims.ratio, dims.budget),
            (1, 4, 16),
            "fixture dims"
        );
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut cache = QsaCache::new();
        let mut state = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        };
        let (mut exp_k, mut exp_v, mut exp_ik) = (Vec::new(), Vec::new(), Vec::new());
        // Positions 0 to 18 are dense: (pos + 1) / 4 <= 16 / 4. Positions 19 to 44 keep 4 of 5 to 11
        // blocks. `c6-margin.awk` (C6 packet appendix) computes the host rule in double precision for
        // these inputs. Each dropped block scores exactly 0, or the 4th and 5th scores differ by
        // 0.0046 or more.
        for pos in 0..45u32 {
            let x = input(pos);
            let want = host::step(&mut proj, &qshape, hq, &mut cache, pos, &x).unwrap();
            let xr = kernels
                .import_tensor(&f32_tensor(vec![1, 64], &x).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let next = step(&mut ctx, &qshape, rq, &state, pos, &xr).unwrap();
            state = next.state;
            assert_parity(
                &values(&mut kernels, &next.out).unwrap(),
                &want,
                &format!("pos {pos} output"),
            );
            // Host cache rows for KVH 1 (qsa.rs:224-241).
            let kp = proj.linear(hq.k, &x).unwrap();
            let mut kn = rms0(&kp, hq.k_norm, dims.eps).unwrap();
            rope_half(&mut kn, dims.rotary_dim as usize, pos, dims.theta).unwrap();
            exp_k.extend_from_slice(&kn);
            exp_v.extend(proj.linear(hq.v, &x).unwrap());
            let ip = proj.linear(hq.idx, &x).unwrap();
            exp_ik.extend_from_slice(&ip[32..]);
            let LayerState::Qsa {
                k: Some(k),
                v: Some(v),
                ik: Some(ik),
                len,
            } = &state
            else {
                panic!("pos {pos}: state has no cache");
            };
            let t = pos + 1;
            assert_eq!(*len, t as usize, "pos {pos} resident length");
            assert_eq!(cache.len(), *len, "pos {pos} host length");
            assert_eq!(k.shape(), &[1, 1, t, 64][..], "pos {pos} key shape");
            assert_eq!(v.shape(), &[1, 1, t, 64][..], "pos {pos} value shape");
            assert_eq!(ik.shape(), &[t, 32][..], "pos {pos} indexer key shape");
            assert_parity(
                &values(&mut kernels, k).unwrap(),
                &exp_k,
                &format!("pos {pos} keys"),
            );
            assert_parity(
                &values(&mut kernels, v).unwrap(),
                &exp_v,
                &format!("pos {pos} values"),
            );
            assert_parity(
                &values(&mut kernels, ik).unwrap(),
                &exp_ik,
                &format!("pos {pos} indexer keys"),
            );
        }
    }

    #[test]
    fn plan_follows_the_host_block_rule() {
        // Ratio 4 and budget 2048 are the QSA values of the real model: 512 blocks attend.
        let real = QsaDims {
            heads_q: 24,
            heads_kv: 2,
            head_dim: 256,
            idx_heads: 4,
            idx_dim: 128,
            ratio: 4,
            budget: 2048,
            rotary_dim: 64,
            theta: 10_000.0,
            hidden: 2560,
            eps: 1e-6,
        };
        let fixture = QsaDims { budget: 16, ..real };
        // (dims, pos, visible, blocks, tail, take, sparse). `flash_next/qsa.rs` lines 286-290 give each row.
        let cases = [
            (real, 2050, 2051, 512, 2048, 512, false),
            (real, 2051, 2052, 513, 2052, 512, true),
            (real, 2052, 2053, 513, 2052, 512, true),
            (fixture, 18, 19, 4, 16, 4, false),
            (fixture, 19, 20, 5, 20, 4, true),
            (fixture, 20, 21, 5, 20, 4, true),
        ];
        for (dims, pos, visible, blocks, tail, take, sparse) in cases {
            let plan = Plan::new(&dims, pos).unwrap();
            let want = Plan {
                visible,
                blocks,
                tail,
                take,
            };
            assert_eq!(plan, want, "plan at position {pos}");
            assert_eq!(plan.sparse(), sparse, "sparse flag at position {pos}");
        }
        assert!(
            matches!(Plan::new(&real, u32::MAX), Err(EngError::BadRequest(_))),
            "position u32::MAX was not rejected as a bad request"
        );
    }

    #[test]
    fn equal_scores_rank_by_lowest_block_index() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = synthetic(&mut kernels);
        // Descending score order. Each group of equal scores keeps its index order.
        let cases: [(&[f32], &[u32]); 3] = [
            (&[2.0, 5.0, 2.0, 5.0, 0.0, 5.0, 2.0], &[1, 3, 5, 0, 2, 6, 4]),
            (&[3.0; 9], &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (&[0.0, 1.0, 0.0], &[1, 0, 2]),
        ];
        for (scores, want) in cases {
            let host = f32_tensor(vec![scores.len() as u32], scores).unwrap();
            let resident = kernels.import_tensor(&host).unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let order = rank_blocks(&mut ctx, &resident).unwrap();
            assert_eq!(words(&mut kernels, &order), want, "order of {scores:?}");
        }
    }

    #[test]
    fn block_mask_keeps_the_top_blocks_and_the_tail() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = synthetic(&mut kernels);
        // Fixture ratio 4 and budget 16: 4 of 7 complete blocks attend, then 2 tail positions.
        let plan = Plan {
            visible: 30,
            blocks: 7,
            tail: 28,
            take: 4,
        };
        // The ranks are blocks 1, 3, 5, 0, 2, 6, 4. Equal scores keep the lower block index.
        let host = f32_tensor(vec![7], &[2.0, 5.0, 2.0, 5.0, 0.0, 5.0, 2.0]).unwrap();
        let scores = kernels.import_tensor(&host).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let mask = block_mask(&mut ctx, &scores, &plan, 4).unwrap();
        // Blocks 1, 3, 5, and 0 attend. Blocks 2, 4, and 6 do not.
        let mut want = Vec::new();
        for attends in [true, true, false, true, false, true, false] {
            want.extend([if attends { 0.0 } else { MASK_OFF }; 4]);
        }
        want.extend([0.0; 2]);
        assert_eq!(values(&mut kernels, &mask).unwrap(), want, "mask");
    }

    #[test]
    fn block_scores_follow_the_independent_reference() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        let qshape = flash.shape().qsa;
        // (position, score of each complete block, block order). A double-precision computation of
        // the host rule on the same inputs gives these values (`c6-margin.awk` in the C6 packet appendix).
        // They omit the host division by sqrt(32). Blocks with a negative dot product score 0.
        let reference: [(u32, &[f32], &[u32]); 2] = [
            (
                23,
                &[19.324, 0.0, 16.9072, 21.6795, 0.0, 24.6513],
                &[5, 3, 0, 2, 1, 4],
            ),
            (
                35,
                &[
                    21.107, 0.0, 17.0112, 19.6284, 0.0, 17.0378, 0.0, 0.0, 25.5287,
                ],
                &[8, 0, 3, 5, 2, 1, 4, 6, 7],
            ),
        ];
        let mut state = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        };
        for pos in 0..36u32 {
            let xr = kernels
                .import_tensor(&f32_tensor(vec![1, 64], &input(pos)).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let next = step(&mut ctx, &qshape, rq, &state, pos, &xr).unwrap();
            state = next.state;
            let Some((_, want_scores, want_order)) = reference.iter().find(|r| r.0 == pos) else {
                continue;
            };
            let LayerState::Qsa { ik: Some(ik), .. } = &state else {
                panic!("pos {pos}: state has no indexer keys");
            };
            let ip = linear(&mut ctx, &rq.idx, &xr).unwrap();
            let plan = Plan::new(&qshape.dims(), pos).unwrap();
            let sel = Selection { plan, pos, ip: &ip };
            let scores = block_scores(&mut ctx, &qshape, rq, &sel, ik).unwrap();
            let order = rank_blocks(&mut ctx, &scores).unwrap();
            let got = values(&mut kernels, &scores).unwrap();
            assert_eq!(got.len(), want_scores.len(), "pos {pos}: block count");
            for (b, (a, e)) in got.iter().zip(*want_scores).enumerate() {
                assert!(
                    (a - e).abs() <= 5e-3,
                    "pos {pos} block {b}: actual {a}, expected {e}"
                );
            }
            assert_eq!(words(&mut kernels, &order), *want_order, "pos {pos}: order");
        }
    }

    #[test]
    fn blocks_tied_at_zero_are_selected_by_lowest_index() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        let view = flash.view();
        let AttnWeights::Qsa(hq) = &view.layers[1].attn else {
            panic!("host layer 1 is not QSA");
        };
        let qshape = flash.shape().qsa;
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut cache = QsaCache::new();
        let mut state = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        };
        // Positions 0 to 18 fill 19 cached positions with distinct values.
        for pos in 0..19u32 {
            let x = input(pos);
            host::step(&mut proj, &qshape, hq, &mut cache, pos, &x).unwrap();
            let xr = kernels
                .import_tensor(&f32_tensor(vec![1, 64], &x).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            state = step(&mut ctx, &qshape, rq, &state, pos, &xr).unwrap().state;
        }
        // Position 19 sees 5 complete blocks and keeps 4. This input makes every projection exactly
        // 0, so every block scores exactly 0 and blocks 0 to 3 attend. Block 4 holds the new position.
        let x = tie_input();
        let want = host::step(&mut proj, &qshape, hq, &mut cache, 19, &x).unwrap();
        let xr = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &x).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let next = step(&mut ctx, &qshape, rq, &state, 19, &xr).unwrap();
        let LayerState::Qsa { ik: Some(ik), .. } = &next.state else {
            panic!("state has no indexer keys");
        };
        let ip = linear(&mut ctx, &rq.idx, &xr).unwrap();
        let plan = Plan::new(&qshape.dims(), 19).unwrap();
        let sel = Selection {
            plan,
            pos: 19,
            ip: &ip,
        };
        let scores = block_scores(&mut ctx, &qshape, rq, &sel, ik).unwrap();
        let order = rank_blocks(&mut ctx, &scores).unwrap();
        assert_eq!(
            values(&mut kernels, &scores).unwrap(),
            vec![0.0; 5],
            "every block scores 0"
        );
        assert_eq!(
            words(&mut kernels, &order),
            [0, 1, 2, 3, 4],
            "tied blocks keep their index order"
        );
        assert_parity(
            &values(&mut kernels, &next.out).unwrap(),
            &want,
            "tied output",
        );
    }

    #[test]
    fn sparse_model_steps_match_the_host_step() {
        let _serial = crate::native_test_lock();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut host = ModelState::new(&shape, &view);
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        let mut state = decoder.zero_state().unwrap();
        // Ratio 4, budget 16: positions 0 to 18 are dense. Positions 19 to 26 keep 4 of 5 to 6 blocks.
        for pos in 0..27u32 {
            let token = pos % 7;
            let want = host_step(&mut proj, &shape, &view, &mut host, token, true).unwrap();
            let before = syncs(&decoder.kernels);
            let got = decoder.step(&mut state, token, StepMode::Record).unwrap();
            if let (Some(before), Some(after)) = (before, syncs(&decoder.kernels)) {
                assert_eq!(
                    after - before,
                    7,
                    "position {pos}: Record synchronizations (L + 5 with L = 2)"
                );
            }
            assert_eq!(state.pos(), pos + 1, "position {pos}: committed position");
            assert_parity(
                &got.logits.unwrap(),
                &want.logits,
                &format!("position {pos} logits"),
            );
            let rows = got.layer_out.unwrap();
            let want_rows = want.layer_out.unwrap();
            assert_eq!(rows.len(), want_rows.len(), "position {pos}: layer rows");
            for (l, (g, w)) in rows.iter().zip(&want_rows).enumerate() {
                assert_parity(g, w, &format!("position {pos} layer {l}"));
            }
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn sparse_step_does_not_synchronize() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        let qshape = flash.shape().qsa;
        let x = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(0)).unwrap())
            .unwrap();
        // Ratio 4, budget 16: position 19 sees 5 complete blocks and keeps 4.
        let state = zero_cache(&mut kernels, 19);
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let sparse = step(&mut ctx, &qshape, rq, &state, 19, &x).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "a sparse step builds a graph only"
        );
        assert_eq!(sparse.out.shape(), &[1, 64][..], "output shape");
    }

    #[test]
    fn step_rejects_a_mismatched_cache() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        let qshape = flash.shape().qsa;
        let x = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(0)).unwrap())
            .unwrap();
        let empty = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        };
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let early = step(&mut ctx, &qshape, rq, &empty, 1, &x);
        assert!(
            matches!(early, Err(EngError::BadRequest(_))),
            "empty cache at position 1 was not rejected as a bad request"
        );
        let LayerState::Qsa { k, v, ik, .. } = zero_cache(&mut kernels, 3) else {
            panic!("zero cache is not QSA");
        };
        let wrong = LayerState::Qsa { k, v, ik, len: 2 };
        let delta = LayerState::Delta {
            rec: kernels
                .import_tensor(&zeros(vec![2, 32, 32]).unwrap())
                .unwrap(),
            ring: kernels
                .import_tensor(&zeros(vec![128, 3]).unwrap())
                .unwrap(),
        };
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        assert!(
            matches!(
                step(&mut ctx, &qshape, rq, &wrong, 2, &x),
                Err(EngError::BadRequest(_))
            ),
            "3 cached rows with length 2 were not rejected"
        );
        assert!(
            matches!(
                step(&mut ctx, &qshape, rq, &delta, 0, &x),
                Err(EngError::BadRequest(_))
            ),
            "a DeltaNet state was not rejected"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn step_does_not_synchronize() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[1].attn else {
            panic!("resident layer 1 is not QSA");
        };
        let qshape = flash.shape().qsa;
        let x0 = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(0)).unwrap())
            .unwrap();
        let x1 = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(1)).unwrap())
            .unwrap();
        let empty = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        };
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let first = step(&mut ctx, &qshape, rq, &empty, 0, &x0).unwrap();
        let second = step(&mut ctx, &qshape, rq, &first.state, 1, &x1).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "two steps build graphs only"
        );
        assert_eq!(second.out.shape(), &[1, 64][..], "output shape");
    }

    /// The step input at which every fixture projection is exactly 0: `sum x = 0` and
    /// `sum x[c] * (c % 8) = 0`. A fixture projection row is `(c % 8) + bias` (fixtures.rs:1-6).
    fn tie_input() -> Vec<f32> {
        let pattern = [0.5, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        (0..64usize).map(|c| pattern[c % 8]).collect()
    }

    /// Reads a U32 resident back as words.
    fn words(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<u32> {
        let host = kernels.readback(tensor).unwrap();
        assert_eq!(host.dtype(), Dtype::U32, "readback dtype");
        host.host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }

    /// The process-wide synchronization count in qualification builds. Other builds return `None`.
    fn syncs(kernels: &MlxKernels) -> Option<u64> {
        #[cfg(feature = "native-fault-qualification")]
        {
            Some(kernels.qualification_sync_count())
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = kernels;
            None
        }
    }

    /// Forced retire flushes (process total) and the peak retired bytes since the last reset, in
    /// qualification builds. Other builds return `None`.
    fn retire_stats(kernels: &MlxKernels) -> Option<(u64, u64)> {
        #[cfg(feature = "native-fault-qualification")]
        {
            Some(kernels.qualification_retire_stats())
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = kernels;
            None
        }
    }

    /// Resets the peak retired bytes in qualification builds. Other builds do nothing.
    fn reset_retire_peak(kernels: &mut MlxKernels) {
        #[cfg(feature = "native-fault-qualification")]
        {
            kernels.qualification_reset_retire_peak();
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = kernels;
        }
    }

    /// `count` pseudo-random activations in [-0.5, 0.5) from a 64-bit linear congruential generator.
    fn pseudo_input(state: &mut u64, count: usize) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (*state >> 40) as f32 / 16_777_216.0 - 0.5
            })
            .collect()
    }

    /// Token `n` of the long synthetic prompt, spread over the vocabulary.
    fn long_token(n: usize, vocab: usize) -> u32 {
        u32::try_from((n as u64 * 7919 + 760) % vocab as u64).unwrap()
    }

    /// Reads a number from the environment, or returns `default` when the variable is unset.
    fn env_number(name: &str, default: Option<u64>) -> u64 {
        match std::env::var(name) {
            Ok(text) => text
                .parse()
                .unwrap_or_else(|_| panic!("{name} is not a number")),
            Err(_) => default.unwrap_or_else(|| panic!("{name} is not set")),
        }
    }

    /// Decision 0039 gate 4, layer level. The first QSA layer of the real model runs on
    /// pseudo-random inputs through the host path and the resident path for more than 2,060
    /// positions. The test compares outputs at the block tolerance and classifies each sparse step by
    /// its selection margin: the score gap between the last kept and the first dropped block, in the
    /// host score scale (the device sum divided by sqrt(ID)). A margin above 1e-5 is strict. Two
    /// exact zeros are an exact tie, which both paths resolve to the lower block index. Any other gap
    /// is ambiguous and fails the test, because float rounding could rank such blocks differently.
    #[test]
    #[ignore = "needs the Flash-Next artifact on the Mac"]
    fn resident_sparse_qsa_layer_matches_the_host_past_2060_positions() {
        let _serial = crate::native_test_lock();
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let report_path = env_path("FLASH_NEXT_QSA_LAYER_REPORT");
        let cache_limit = env_number("FLASH_NEXT_CACHE_LIMIT", None);
        let positions = u32::try_from(env_number("FLASH_NEXT_QSA_POSITIONS", Some(2400))).unwrap();
        let seed = env_number("FLASH_NEXT_QSA_SEED", Some(1));
        assert!(positions > 2060, "gate 4 needs more than 2,060 positions");
        let ckpt = open_checkpoint(&artifact);
        let config = std::fs::read(artifact.join("config.json")).unwrap();
        let flash = FlashWeights::load(&ckpt, &config).unwrap();
        let qshape = flash.shape().qsa;
        let dims = qshape.dims();
        let view = flash.view();
        let layer = view
            .layers
            .iter()
            .position(|l| matches!(l.attn, AttnWeights::Qsa(_)))
            .expect("the model has no QSA layer");
        let AttnWeights::Qsa(hq) = &view.layers[layer].attn else {
            panic!("host layer {layer} is not QSA");
        };
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.set_cache_limit(cache_limit).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let RAttn::Qsa(rq) = &weights.layers[layer].attn else {
            panic!("resident layer {layer} is not QSA");
        };
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut cache = QsaCache::new();
        let mut state = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: 0,
        };
        let mut gate = Gate::default();
        let mut rng = seed;
        let (mut sparse, mut strict, mut zero_ties) = (0u32, 0u32, 0u32);
        let mut ambiguous: Vec<u32> = Vec::new();
        let mut smallest = f32::INFINITY;
        for pos in 0..positions {
            let x = pseudo_input(&mut rng, dims.hidden as usize);
            let want = host::step(&mut proj, &qshape, hq, &mut cache, pos, &x).unwrap();
            let host_x = f32_tensor(vec![1, dims.hidden], &x).unwrap();
            let xr = kernels.import_tensor(&host_x).unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let next = step(&mut ctx, &qshape, rq, &state, pos, &xr).unwrap();
            state = next.state;
            let plan = Plan::new(&dims, pos).unwrap();
            if plan.sparse() {
                let LayerState::Qsa { ik: Some(ik), .. } = &state else {
                    panic!("pos {pos}: state has no indexer keys");
                };
                let ip = linear(&mut ctx, &rq.idx, &xr).unwrap();
                let sel = Selection { plan, pos, ip: &ip };
                let scores = block_scores(&mut ctx, &qshape, rq, &sel, ik).unwrap();
                let mut ranked = values(ctx.kernels, &scores).unwrap();
                ranked.sort_by(|a, b| b.total_cmp(a));
                let take = plan.take as usize;
                let (kept, dropped) = (ranked[take - 1], ranked[take]);
                let margin = (kept - dropped) / (dims.idx_dim as f32).sqrt();
                sparse += 1;
                if margin > 1e-5 {
                    strict += 1;
                    smallest = smallest.min(margin);
                } else if kept == 0.0 && dropped == 0.0 {
                    zero_ties += 1;
                } else {
                    ambiguous.push(pos);
                }
            }
            if pos >= 2040 || pos.is_multiple_of(256) {
                let got = values(&mut kernels, &next.out).unwrap();
                let label = format!("qsa layer {layer} pos {pos}");
                gate.check(&label, &got, &want, COS_BLOCK, REL_BLOCK);
            }
        }
        gate.report.push_str(&format!(
            "layer={layer}\npositions={positions}\nseed={seed}\nsparse_steps={sparse}\n\
             strict_steps={strict}\nsmallest_strict_margin={smallest}\nzero_tie_steps={zero_ties}\n\
             ambiguous_steps={ambiguous:?}\nchecks={}\n",
            gate.checks
        ));
        write_report(&report_path, &gate);
        assert!(
            ambiguous.is_empty(),
            "ambiguous selection margins at {ambiguous:?}; choose another seed or length"
        );
        assert!(sparse >= 16, "only {sparse} sparse steps ran");
        assert!(strict >= 1, "no step had a margin above 1e-5");
        assert!(gate.failures.is_empty(), "{:?}", gate.failures);
    }

    /// Decision 0039 gate 4, model level. The resident path and the host path run the same
    /// synthetic prompt of `FLASH_NEXT_QSA_TOKENS` tokens (default 2,070). The last 4 resident steps
    /// run in `Record` mode and the others in `Prompt` mode. Qualification builds check the
    /// synchronization budget of every sparse step: `L + 2` in `Prompt` and `L + 5` in `Record`.
    /// The test compares the logits of the last 4 steps at the end-to-end tolerance, and the greedy
    /// token when the host margin is above 0.1. It also checks that each step charges 55,296 bytes
    /// per position to the session budget. It does not measure the selection margin of the 12 QSA
    /// layers; the layer test does.
    #[test]
    #[ignore = "needs the Flash-Next artifact on the Mac"]
    fn resident_sparse_qsa_model_matches_the_host_past_2060_positions() {
        const COMPARED: usize = 4;
        let _serial = crate::native_test_lock();
        let artifact = env_path("FLASH_NEXT_ARTIFACT");
        let report_path = env_path("FLASH_NEXT_QSA_MODEL_REPORT");
        let cache_limit = env_number("FLASH_NEXT_CACHE_LIMIT", None);
        let count = usize::try_from(env_number("FLASH_NEXT_QSA_TOKENS", Some(2070))).unwrap();
        assert!(count > 2060, "gate 4 needs more than 2,060 positions");
        let ckpt = open_checkpoint(&artifact);
        let config = std::fs::read(artifact.join("config.json")).unwrap();
        let mut gate = Gate::default();
        gate.report.push_str(&format!("tokens={count}\n"));

        // The resident phase.
        let flash = FlashWeights::load(&ckpt, &config).unwrap();
        let vocab = flash.shape().vocab;
        let tokens: Vec<u32> = (0..count).map(|n| long_token(n, vocab)).collect();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.set_cache_limit(cache_limit).unwrap();
        let mut decoder = ResidentDecoder::new(kernels, &ckpt, &flash).unwrap();
        drop(flash);
        let layers = decoder.weights.layers.len() as u64;
        let mut state = decoder.zero_state().unwrap();
        let mut resident: Vec<Vec<f32>> = Vec::new();
        for (n, &token) in tokens.iter().enumerate() {
            let record = n + COMPARED >= count;
            let mode = if record {
                StepMode::Record
            } else {
                StepMode::Prompt
            };
            reset_retire_peak(&mut decoder.kernels);
            let (s0, f0) = (syncs(&decoder.kernels), retire_stats(&decoder.kernels));
            let out = decoder.step(&mut state, token, mode).unwrap();
            let (s1, f1) = (syncs(&decoder.kernels), retire_stats(&decoder.kernels));
            // The pinned model has 12 QSA layers with KVH 2, D 256, and ID 128. One position stores
            // 12 * (2 * 2 * 256 + 128) * 4 = 55,296 bytes, so n + 1 positions hold 55,296 * (n + 1).
            assert_eq!(
                decoder.session_used(),
                55_296 * (n as u64 + 1),
                "session charge after step {n}"
            );
            if n >= 2051
                && let (Some(s0), Some(s1), Some(f0), Some(f1)) = (s0, s1, f0, f1)
            {
                let budget = layers + if record { 5 } else { 2 };
                gate.report.push_str(&format!(
                    "sparse_step{n}={mode:?},{},{},{}\n",
                    s1 - s0,
                    f1.0 - f0.0,
                    f1.1
                ));
                if s1 - s0 != budget || f1.0 != f0.0 {
                    gate.failures.push(format!("syncs at step {n}"));
                }
            }
            if let Some(logits) = out.logits {
                resident.push(logits);
            }
        }
        gate.report
            .push_str(&format!("session_charge={}\n", decoder.session_used()));
        drop(state);
        drop(decoder);

        // The host phase.
        let flash = FlashWeights::load(&ckpt, &config).unwrap();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut host = ModelState::new(&shape, &view);
        let mut want: Vec<Vec<f32>> = Vec::new();
        for (n, &token) in tokens.iter().enumerate() {
            let out = host_step(&mut proj, &shape, &view, &mut host, token, false).unwrap();
            if n + COMPARED >= count {
                want.push(out.logits);
            }
        }

        assert_eq!(resident.len(), COMPARED, "recorded resident steps");
        assert_eq!(want.len(), COMPARED, "recorded host steps");
        for (i, (got, exp)) in resident.iter().zip(&want).enumerate() {
            let n = count - COMPARED + i;
            gate.check(&format!("logits n{n}"), got, exp, COS_END, REL_END);
            let (g, w) = (top5(got), top5(exp));
            let margin = w[0].1 - w[1].1;
            gate.report.push_str(&format!(
                "argmax n{n} resident={} host={} host_margin={margin:.6}\n",
                g[0].0, w[0].0
            ));
            if margin > 0.1 && g[0].0 != w[0].0 {
                gate.failures.push(format!("argmax n{n}"));
            }
        }
        write_report(&report_path, &gate);
        assert_eq!(gate.checks, COMPARED, "check count");
        assert!(gate.failures.is_empty(), "{:?}", gate.failures);
    }
}
