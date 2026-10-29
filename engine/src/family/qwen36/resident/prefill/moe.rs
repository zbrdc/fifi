//! Batched MoE over a chunk with every expert resident.
//!
//! The chunk form of `moe::step` for `T` token rows. Both routing modes of the decoder select the
//! experts of every row. `Routing::Host` (the reference) reads the `[T, 256]` probabilities once
//! and routes every row with `route_experts`, the function that the decode block and the host
//! reference use (the lowest index wins a tie; the gates renormalize over the selection). The
//! `T * top_k` (token, choice) pairs then go to the device sorted by expert
//! (`chunk_plan::pair_plan`) with one `import_many`. `Routing::Device` (the decoder default) builds
//! the whole route on the device as lazy graph nodes: the top-k by a stable `argsort` of the
//! negated probabilities, the gates by `take_along_axis` and the renormalization, the pair sort by
//! an `argsort` of the exact small integer keys, and the un-sort by its inverse permutation. It
//! reads nothing back and imports nothing (the host side of the routing measured 140 ms of
//! the 855 ms 512-row chunk). Its health check replaces the per-row finite check of
//! `route_experts`: each layer returns the sum of its whole probability block, and the chunk step
//! reads the stacked sums back once and checks them with `route::check_health`.
//!
//! `gather_qmm` reads the selected experts in place, so a chunk has no page pool and no expert
//! waves. The weighted rows return to token-major order with the inverse permutation and sum in
//! choice order, as the decode step sums them. Sorting keeps the rows of one expert next to each
//! other, and the three expert products call `gather_qmm_sorted_bounded`, which passes MLX's
//! `sorted_indices` flag over ids that one `bound_indices` clamp covers for the whole chunk (the
//! flag cuts the expert product time of a 512-row chunk from about 457 ms to 207 ms). The host path
//! proves the non-decreasing order of `plan.experts` before the import, because the method cannot
//! read the device indices; the device path needs no proof, because its order comes from `argsort`
//! by definition. The decode path keeps the unsorted flag (`gather_qmm_bounded` with `false`).

use super::super::chunk_plan::{PairPlan, pair_plan, require_sorted_experts};
use super::super::ops::{Ctx, dim, f32_tensor, short_import, to_act, u32_tensor, values};
use super::super::weights::RMoe;
use crate::{
    EngError,
    family::{
        flash_next::math::expect_len,
        qwen36::host::HostShape,
        route::{Routed, route_experts},
        spec::Router,
    },
    kernels::{BinaryOp, ReduceOp, UnaryOp},
    tensor::{ExecutionTensor, Tensor},
};

/// Shared expert `sigmoid(x · s) * down(silu(gate(x)) * up(x))` for `x` `[T, H]`, as `[T, H]`.
/// The shared expert in kernel-ready parts: the down projection `so` `[T, H]` and the scalar
/// logit `raw` `[T, 1]` (the fused MoE combine folds `sigmoid(raw) * so` itself).
fn shared_parts(
    ctx: &mut Ctx<'_>,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let sg = ctx.kernels.qmm(x, &w.shared_gate)?;
    let su = ctx.kernels.qmm(x, &w.shared_up)?;
    let act = ctx.kernels.silu(&sg)?;
    let sh = ctx.kernels.binary(BinaryOp::Mul, &act, &su)?;
    let so = ctx.kernels.qmm(&sh, &w.shared_down)?;
    let raw = ctx.kernels.mm(x, &w.shared_scalar_t)?;
    Ok((so, raw))
}

fn shared_rows(
    ctx: &mut Ctx<'_>,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let (so, raw) = shared_parts(ctx, w, x)?;
    let factor = ctx.kernels.unary(UnaryOp::Sigmoid, &raw)?;
    ctx.kernels.binary(BinaryOp::Mul, &factor, &so)
}

/// Imports the four index tensors of a plan with one synchronization: the sorted experts `[P]`,
/// the token of each sorted pair `[P]`, the inverse permutation `[P]` (all U32), and the sorted
/// gates `[P, 1, 1]` (F32). It returns them in that order.
fn import_plan(ctx: &mut Ctx<'_>, plan: &PairPlan) -> Result<[ExecutionTensor; 4], EngError> {
    let pairs = dim(plan.experts.len())?;
    let sources: [Tensor; 4] = [
        u32_tensor(vec![pairs], &plan.experts)?,
        u32_tensor(vec![pairs], &plan.tokens)?,
        u32_tensor(vec![pairs], &plan.inverse)?,
        f32_tensor(vec![pairs, 1, 1], &plan.gates)?,
    ];
    let refs: Vec<&Tensor> = sources.iter().collect();
    ctx.kernels
        .import_many(&refs)?
        .try_into()
        .map_err(|_| short_import())
}

/// Runs the MoE of one layer for the `T` rows of `x` `[T, H]` and returns `[T, H]` with the route
/// of every row. It synchronizes twice: the probability readback (which also evaluates `x`) and the
/// import of the sorted pairs.
///
/// An input that is not `[T, H]` with `T > 0` returns `BadRequest` before the first native call.
/// A pair plan whose experts are not non-decreasing also returns `BadRequest` before the import:
/// `gather_qmm_sorted` requires that order and cannot read the device indices.
/// A router row that holds a NaN, an infinite value, or a negative value returns `Backend` from
/// `route_experts`. That error ends one job and does not make the adapter terminal.
pub(super) fn chunk(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, Vec<Routed>), EngError> {
    let m = &shape.moe;
    let hidden = dim(shape.hidden)?;
    let &[rows, width] = x.shape() else {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H]".into(),
        ));
    };
    if rows == 0 || width != hidden {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H] with T > 0".into(),
        ));
    }
    // Router probabilities softmax(x · router_t) `[T, E]`, read back once.
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    let host = values(ctx.kernels, &probs)?;
    expect_len(
        &host,
        m.experts
            .checked_mul(usize::try_from(rows).unwrap_or(usize::MAX))
            .ok_or_else(|| EngError::BadRequest("moe chunk size overflows".into()))?,
        "moe chunk router",
    )?;
    let top_k = u32::try_from(m.top_k)
        .map_err(|_| EngError::BadRequest("resident moe top-k exceeds u32".into()))?;
    let router = Router::SoftmaxTopK { renorm: m.renorm };
    let routed = host
        .chunks_exact(m.experts)
        .map(|row| route_experts(row, &router, top_k))
        .collect::<Result<Vec<_>, _>>()?;
    let plan = pair_plan(&routed, m.top_k, m.experts)?;
    require_sorted_experts(&plan.experts)?;
    let pairs = dim(plan.experts.len())?;
    let [ids, tokens, inverse, gates] = import_plan(ctx, &plan)?;
    // One clamp for the expert ids of the whole chunk: the three sorted
    // gathered products share it, and the clamp of the already in-range plan ids is the identity.
    // The `tokens` and `inverse` takes keep their own clamps: each is consumed once, so a shared
    // clamp would save no node.
    let bounded = ctx.kernels.bound_indices(&ids, dim(m.experts)?)?;
    // Selected experts: gate and up on the input row of each pair, SwiGLU, then down per pair.
    let picked = ctx.kernels.take(x, &tokens, 0)?;
    let x_pairs = ctx.kernels.reshape(&picked, &[pairs, 1, hidden])?;
    let hg = ctx
        .kernels
        .gather_qmm_sorted_bounded(&x_pairs, &w.gate, &bounded)?;
    let hu = ctx
        .kernels
        .gather_qmm_sorted_bounded(&x_pairs, &w.up, &bounded)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx
        .kernels
        .gather_qmm_sorted_bounded(&h, &w.down, &bounded)?;
    let weighted = ctx.kernels.binary(BinaryOp::Mul, &eo, &gates)?;
    // Back to token-major order, then the sum over the choices of each token.
    let flat = ctx.kernels.reshape(&weighted, &[pairs, hidden])?;
    let ordered = ctx.kernels.take(&flat, &inverse, 0)?;
    let ordered = ctx.kernels.reshape(&ordered, &[rows, top_k, hidden])?;
    let experts = ctx.kernels.reduce(ReduceOp::Sum, &ordered, 1, false)?;
    let shared_out = shared_rows(ctx, w, x)?;
    let out = ctx.kernels.binary(BinaryOp::Add, &experts, &shared_out)?;
    // As in the decode combine of `moe.rs`, the gates are F32, so the weighted expert sum and the
    // shared expert product are F32. The residual add takes the activation dtype.
    let out = to_act(ctx, &out)?;
    Ok((out, routed))
}

/// The routing trace of a device chunk that only tests read back (all U32 except the F32 gates):
/// the rank-ordered experts `[T, K]` with their gates `[T, K]`, the sorted expert ids `[P]`, and
/// the sort permutation and its inverse `[P]`. Production code drops the whole trace: the chunk
/// step consumes only the output and the health of `DeviceChunk`.
#[cfg(test)]
pub(super) struct ChunkTrace {
    /// The chosen experts `[T, K]` (U32) in rank order, the lowest index first on ties.
    pub(super) ids: ExecutionTensor,
    /// The gates `[T, K]` (F32), renormalized over the selection when the family asks for it.
    pub(super) gates: ExecutionTensor,
    /// The expert of each sorted pair `[P]` (U32), non-decreasing.
    pub(super) sorted_ids: ExecutionTensor,
    /// The pair-sort permutation `[P]` (U32): sorted position `i` holds token-major pair
    /// `order_p[i]`.
    pub(super) order_p: ExecutionTensor,
    /// The inverse permutation `[P]` (U32): `inverse[order_p[i]] == i`.
    pub(super) inverse: ExecutionTensor,
}

/// The outcome of one device-routed chunk MoE call.
pub(super) struct DeviceChunk {
    /// The layer output `[T, H]` in the activation dtype.
    pub(super) out: ExecutionTensor,
    /// The sum of the whole `[T, E]` probability block `[1]` (F32). A softmax block sums to `T`
    /// within rounding, and a NaN or infinite entry makes the sum non-finite. The chunk step
    /// stacks one per layer, reads the stack back once, and checks it with `route::check_health`.
    pub(super) health: ExecutionTensor,
    /// The routing trace that only tests read (see `ChunkTrace`).
    #[cfg(test)]
    pub(super) trace: ChunkTrace,
}

/// Runs the MoE of one layer for the `T` rows of `x` `[T, H]` entirely on the device and returns
/// `[T, H]` with the health of the router block. It reads nothing back
/// and imports nothing, so it makes no synchronization; the caller bounds the lazy graph with its
/// evaluation cadence and checks the stacked health at the chunk end.
///
/// Sortedness: `gather_qmm_sorted` requires non-decreasing expert ids along the flat pair axis.
/// The host path proves the order of its imported plan (`require_sorted_experts`); this path needs
/// no proof, because the order comes from `argsort` by definition: `sorted_ids` is the gather of
/// the keys at the positions of a stable ascending sort of the keys.
///
/// Which errors come before the first native call: the `[T, H]` input check and the top-k bound
/// return their `BadRequest` before the router `mm`. The router-width `BadRequest` (the check of
/// `step_device`) comes after it, and the pair-count overflow `BadRequest` after the top-k. A
/// router row that holds a NaN or an infinite value is not an error here: it makes the health
/// sum non-finite, and the chunk step returns the `Backend` error of `route_experts` at the chunk
/// end, as the host path does immediately.
pub(super) fn chunk_device(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<DeviceChunk, EngError> {
    let m = &shape.moe;
    let hidden = dim(shape.hidden)?;
    let &[rows, width] = x.shape() else {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H]".into(),
        ));
    };
    if rows == 0 || width != hidden {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H] with T > 0".into(),
        ));
    }
    let top_k = u32::try_from(m.top_k)
        .map_err(|_| EngError::BadRequest("resident moe top-k exceeds u32".into()))?;
    // a. Router probabilities softmax(x · router_t) `[T, E]`, and the health of the whole block.
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    if probs.shape() != [rows, dim(m.experts)?] {
        let count = probs
            .shape()
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)));
        return Err(EngError::BadRequest(format!(
            "moe router has {} values, expected {}",
            count.unwrap_or(u64::MAX),
            m.experts
        )));
    }
    let health = ctx.kernels.reduce(ReduceOp::Sum, &probs, 1, false)?;
    let health = ctx.kernels.reduce(ReduceOp::Sum, &health, 0, true)?;
    // b. Top-k per row: a stable ascending sort of the negated probabilities lists the largest
    // probability first and keeps the lower index first among equal values, the rule of
    // `route_experts`. The gates renormalize over the selection when the family asks for it.
    let negated = ctx.kernels.unary(UnaryOp::Neg, &probs)?;
    let order = ctx.kernels.argsort(&negated, 1)?;
    let ids = ctx.kernels.slice(&order, &[0, 0], &[rows, top_k])?;
    let top = ctx.kernels.take_along_axis(&probs, &ids, 1)?;
    let gates = if m.renorm {
        let sum = ctx.kernels.reduce(ReduceOp::Sum, &top, 1, true)?;
        ctx.kernels.binary(BinaryOp::Div, &top, &sum)?
    } else {
        top
    };
    // c. The pair sort: P = T * top_k pairs in token-major order (pair p = t * top_k + k), sorted
    // by expert. `u32_to_f32` is exact for the expert ids and the permutation below 2^24, and the
    // stable sort keeps the token-major order inside one expert, the tie order of `pair_plan`.
    let pairs = rows
        .checked_mul(top_k)
        .ok_or_else(|| EngError::BadRequest("moe chunk pair count overflows".into()))?;
    let keys = ctx.kernels.reshape(&ids, &[pairs])?;
    let keys_f = ctx.kernels.u32_to_f32(&keys)?;
    let order_p = ctx.kernels.argsort(&keys_f, 0)?;
    let sorted_ids = ctx.kernels.take_along_axis(&keys, &order_p, 0)?;
    // One clamp for the sorted expert ids of the whole chunk: the three
    // sorted gathered products share it. The clamp of the argsort output is the identity, so the
    // chunk keeps its values bit for bit.
    let sorted_bounded = ctx.kernels.bound_indices(&sorted_ids, dim(m.experts)?)?;
    // shortcut: the direct pair gather (take(x, order_p / top_k)) needs an index remap the
    // engine cannot express yet: take rejects U32 sources and BinaryOp has no floor divide,
    // so the composite repeat+take stays until one of those lands.
    let x_rep = ctx.kernels.repeat(x, top_k, 0)?;
    let x_sorted = ctx.kernels.take(&x_rep, &order_p, 0)?;
    let x_pairs = ctx.kernels.reshape(&x_sorted, &[pairs, 1, hidden])?;
    // d. Selected experts in the sorted order, SwiGLU, down per pair, then the un-sort back to
    // token-major order and the sum over the choices of each token. The gate product is the same
    // elementwise product as in the host path; only its place moves (token-major, not sorted).
    let hg = ctx
        .kernels
        .gather_qmm_sorted_bounded(&x_pairs, &w.gate, &sorted_bounded)?;
    let hu = ctx
        .kernels
        .gather_qmm_sorted_bounded(&x_pairs, &w.up, &sorted_bounded)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx
        .kernels
        .gather_qmm_sorted_bounded(&h, &w.down, &sorted_bounded)?;
    let order_p_f = ctx.kernels.u32_to_f32(&order_p)?;
    let inverse = ctx.kernels.argsort(&order_p_f, 0)?;
    let eo_flat = ctx.kernels.reshape(&eo, &[pairs, hidden])?;
    // The fused MoE combine (Q36_FUSED_PREFILL=1): the spike-validated kernel takes the
    // un-sort, the F32 gate product, the choice sum, the shared fold, and the activation cast
    // in one pass (no [T, K, H] F32 intermediate). The spike body hardcodes K 8 and HID 2048,
    // so only the production shape engages; anything else keeps the composite tail.
    let fused_combine = super::super::moe::fused_prefill()
        && std::env::var_os("Q36_NO_MOECOMBINE").is_none()
        && top_k == 8
        && hidden == 2048;
    let out = if fused_combine {
        let (so, raw) = shared_parts(ctx, w, x)?;
        ctx.kernels
            .moe_combine(&eo_flat, &inverse, &gates, &so, &raw, rows)?
    } else {
        let ordered = ctx.kernels.take(&eo_flat, &inverse, 0)?;
        let ordered = ctx.kernels.reshape(&ordered, &[rows, top_k, hidden])?;
        let gates3 = ctx.kernels.reshape(&gates, &[rows, top_k, 1])?;
        let weighted = ctx.kernels.binary(BinaryOp::Mul, &ordered, &gates3)?;
        let experts = ctx.kernels.reduce(ReduceOp::Sum, &weighted, 1, false)?;
        let shared_out = shared_rows(ctx, w, x)?;
        // As in the host chunk and the decode combine, the gates are F32, so the weighted expert
        // sum and the shared expert product are F32. The residual add takes the activation dtype.
        ctx.kernels.binary(BinaryOp::Add, &experts, &shared_out)?
    };
    let out = to_act(ctx, &out)?;
    Ok(DeviceChunk {
        out,
        health,
        #[cfg(test)]
        trace: ChunkTrace {
            ids,
            gates,
            sorted_ids,
            order_p,
            inverse,
        },
    })
}

/// The phase split of one device-routed chunk MoE call: routing (router matmul, softmax,
/// top-k, both sorts) vs the expert GEMMs (x gather, gate/up/down gathered products, SwiGLU)
/// vs the tail (un-sort, gates, choice sum, shared expert, cast). The phase boundary is the
/// barrier's evaluated frontier: routing ends at the clamped sorted ids (forcing both argsorts
/// and the clamp), the expert GEMM phase ends at the un-evaluated `eo`, and the tail ends at the
/// cast output - so each phase clock covers exactly its own graph segment. ONLY the ignored prefill
/// stage harness reads this (the MoE roofline seam); production keeps the single
/// `chunk_device` entry. The operations mirror `chunk_device` line for line in phase order,
/// with one barrier evaluate per phase, so each phase clock covers its own graph segment
/// only (the previous barrier flushed the earlier work). The three extra evaluates per layer
/// inflate the stage sum slightly; the harness reports the inflation as a barrier-to-
/// reference wall ratio.
#[cfg(test)]
pub(super) struct MoePhases {
    pub(super) route_s: f64,
    pub(super) gemm_s: f64,
    pub(super) tail_s: f64,
}

#[cfg(test)]
pub(super) fn chunk_device_phased(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<(DeviceChunk, MoePhases), EngError> {
    use std::time::Instant;

    let m = &shape.moe;
    let hidden = dim(shape.hidden)?;
    let &[rows, width] = x.shape() else {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H]".into(),
        ));
    };
    if rows == 0 || width != hidden {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H] with T > 0".into(),
        ));
    }
    let top_k = u32::try_from(m.top_k)
        .map_err(|_| EngError::BadRequest("resident moe top-k exceeds u32".into()))?;
    let pairs = rows
        .checked_mul(top_k)
        .ok_or_else(|| EngError::BadRequest("moe chunk pair count overflows".into()))?;
    // Routing phase: router probabilities, health, top-k, the renormalized gates, and the
    // pair sort with its clamped sorted ids.
    let clock = Instant::now();
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    if probs.shape() != [rows, dim(m.experts)?] {
        return Err(EngError::BadRequest(format!(
            "moe router has {} values, expected {}",
            probs.shape().iter().product::<u32>(),
            m.experts
        )));
    }
    let health = ctx.kernels.reduce(ReduceOp::Sum, &probs, 1, false)?;
    let health = ctx.kernels.reduce(ReduceOp::Sum, &health, 0, true)?;
    let negated = ctx.kernels.unary(UnaryOp::Neg, &probs)?;
    let order = ctx.kernels.argsort(&negated, 1)?;
    let ids = ctx.kernels.slice(&order, &[0, 0], &[rows, top_k])?;
    let top = ctx.kernels.take_along_axis(&probs, &ids, 1)?;
    let gates = if m.renorm {
        let sum = ctx.kernels.reduce(ReduceOp::Sum, &top, 1, true)?;
        ctx.kernels.binary(BinaryOp::Div, &top, &sum)?
    } else {
        top
    };
    let keys = ctx.kernels.reshape(&ids, &[pairs])?;
    let keys_f = ctx.kernels.u32_to_f32(&keys)?;
    let order_p = ctx.kernels.argsort(&keys_f, 0)?;
    let sorted_ids = ctx.kernels.take_along_axis(&keys, &order_p, 0)?;
    let sorted_bounded = ctx.kernels.bound_indices(&sorted_ids, dim(m.experts)?)?;
    // The barrier evaluates the clamped sorted ids (which forces the full pair-sort chain:
    // the reshape, the u32-to-f32 cast, both argsorts, and the clamp), the gates, and the
    // health - the entire routing frontier, so no routing work leaks into the gemm phase.
    ctx.kernels
        .evaluate(&[sorted_bounded.ids(), &gates, &health])?;
    let route_s = clock.elapsed().as_secs_f64();
    // Expert GEMM phase: the pair gather of x and the three sorted gathered products with
    // the SwiGLU between them.
    let clock = Instant::now();
    let x_rep = ctx.kernels.repeat(x, top_k, 0)?;
    let x_sorted = ctx.kernels.take(&x_rep, &order_p, 0)?;
    let x_pairs = ctx.kernels.reshape(&x_sorted, &[pairs, 1, hidden])?;
    let hg = ctx
        .kernels
        .gather_qmm_sorted_bounded(&x_pairs, &w.gate, &sorted_bounded)?;
    let hu = ctx
        .kernels
        .gather_qmm_sorted_bounded(&x_pairs, &w.up, &sorted_bounded)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx
        .kernels
        .gather_qmm_sorted_bounded(&h, &w.down, &sorted_bounded)?;
    ctx.kernels.evaluate(&[&eo])?;
    let gemm_s = clock.elapsed().as_secs_f64();
    // Tail phase: the un-sort, the gated choice sum, the shared expert, and the cast, fused
    // or composite exactly as `chunk_device` picks.
    let clock = Instant::now();
    let order_p_f = ctx.kernels.u32_to_f32(&order_p)?;
    let inverse = ctx.kernels.argsort(&order_p_f, 0)?;
    let eo_flat = ctx.kernels.reshape(&eo, &[pairs, hidden])?;
    let fused_combine = super::super::moe::fused_prefill()
        && std::env::var_os("Q36_NO_MOECOMBINE").is_none()
        && top_k == 8
        && hidden == 2048;
    let out = if fused_combine {
        let (so, raw) = shared_parts(ctx, w, x)?;
        ctx.kernels
            .moe_combine(&eo_flat, &inverse, &gates, &so, &raw, rows)?
    } else {
        let ordered = ctx.kernels.take(&eo_flat, &inverse, 0)?;
        let ordered = ctx.kernels.reshape(&ordered, &[rows, top_k, hidden])?;
        let gates3 = ctx.kernels.reshape(&gates, &[rows, top_k, 1])?;
        let weighted = ctx.kernels.binary(BinaryOp::Mul, &ordered, &gates3)?;
        let experts = ctx.kernels.reduce(ReduceOp::Sum, &weighted, 1, false)?;
        let shared_out = shared_rows(ctx, w, x)?;
        ctx.kernels.binary(BinaryOp::Add, &experts, &shared_out)?
    };
    let out = to_act(ctx, &out)?;
    ctx.kernels.evaluate(&[&out])?;
    let tail_s = clock.elapsed().as_secs_f64();
    Ok((
        DeviceChunk {
            out,
            health,
            trace: ChunkTrace {
                ids,
                gates,
                sorted_ids,
                order_p,
                inverse,
            },
        },
        MoePhases {
            route_s,
            gemm_s,
            tail_s,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::super::moe::{step as decode_step, step_device as decode_step_device};
    use super::super::super::ops::{assert_parity, values};
    use super::super::super::weights::ResidentWeights;
    use super::*;
    use crate::family::qwen36::host::{self, MoeW};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY};
    use crate::kernels::MlxKernels;
    use crate::tensor::Dtype;
    use crate::weights::StreamedCheckpoint;

    /// Input `n`: `x[i] = (((3 i + 7 n) % 13) - 6) / 8`, an exact multiple of 1/8.
    fn input(n: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((3 * i + 7 * n) % 13) as f32 - 6.0) / 8.0)
            .collect()
    }

    /// Top-2 experts of layer 0 for inputs 0 to 7, from the CPU reference with f64 softmax (the
    /// table of the decode block test in `moe.rs`). The smallest gap between the second and third
    /// probability is 0.0071 (input 2). The expert sets cover all four experts.
    const LAYER0_ROUTES: [[u32; 2]; 8] = [
        [2, 1],
        [0, 3],
        [1, 0],
        [3, 0],
        [1, 3],
        [0, 2],
        [1, 3],
        [3, 0],
    ];

    fn setup() -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(&TINY);
        let config = tiny::config(&TINY);
        let oracle = HostWeights::load(&ckpt, &TINY, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, &TINY, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, w, kernels)
    }

    fn upload(kernels: &mut MlxKernels, shape: Vec<u32>, x: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(shape, x).unwrap())
            .unwrap()
    }

    fn host_moe(
        oracle: &HostWeights,
        layer: usize,
    ) -> MoeW<'_, crate::family::flash_next::stored::StoredWeight> {
        let view = oracle.view();
        let m = view.layers[layer].as_moe().unwrap();
        MoeW {
            router: m.router,
            gate: m.gate,
            up: m.up,
            down: m.down,
            shared_gate: m.shared_gate,
            shared_up: m.shared_up,
            shared_down: m.shared_down,
            shared_scalar: m.shared_scalar,
        }
    }

    #[test]
    fn a_chunk_of_eight_rows_matches_the_host_block_and_the_expected_experts() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut want = Vec::new();
        let mut data = Vec::new();
        for n in 0..8u32 {
            let x = input(n);
            want.extend(host::moe_step(&mut cpu, &shape, &host_moe(&oracle, 0), &x).unwrap());
            data.extend(x);
        }
        let xr = upload(&mut kernels, vec![8, 64], &data);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (out, routes) = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
        assert_eq!(out.shape(), [8, 64], "output shape");
        let experts: Vec<[u32; 2]> = routes
            .iter()
            .map(|r| [r.indices[0], r.indices[1]])
            .collect();
        assert_eq!(experts, LAYER0_ROUTES, "experts of the eight rows");
        for (n, route) in routes.iter().enumerate() {
            let gate_sum: f32 = route.weights.iter().sum();
            assert!(
                (gate_sum - 1.0).abs() <= 1e-6,
                "row {n} gates sum to {gate_sum}"
            );
        }
        assert_parity(&values(&mut kernels, &out).unwrap(), &want, "chunk output");
    }

    #[test]
    fn a_chunk_of_one_row_equals_the_decode_block() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let x = input(3);
        let one = upload(&mut kernels, vec![1, 64], &x);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (got, routes) = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &one).unwrap();
        let (want, (selection, _)) =
            decode_step(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &one).unwrap();
        let chosen: Vec<(u32, f32)> = routes[0]
            .indices
            .iter()
            .copied()
            .zip(routes[0].weights.iter().copied())
            .collect();
        assert_eq!(chosen, selection, "the route of the decode block");
        assert_parity(
            &values(&mut kernels, &got).unwrap(),
            &values(&mut kernels, &want).unwrap(),
            "one row",
        );
    }

    /// Reads a U32 resident back as words.
    fn read_u32(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<u32> {
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

    /// Asserts `max |a - b| / max |b| <= tol`, the device-against-host gate (1e-6 relative in F32:
    /// the two paths differ only in the F32 summation order of the gate renormalization and the
    /// place of the gate product).
    fn assert_close_relative(actual: &[f32], expected: &[f32], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        let (mut diff, mut top) = (0f64, 0f64);
        for (&a, &b) in actual.iter().zip(expected) {
            diff = diff.max((f64::from(a) - f64::from(b)).abs());
            top = top.max(f64::from(b).abs());
        }
        assert!(top > 0.0, "{what}: expected values are all zero");
        assert!(diff / top <= tol, "{what}: relative error {}", diff / top);
    }

    /// The device-routed chunk of 8 rows selects the LAYER0_ROUTES
    /// experts (the set per row, and with the gates the rank order), gates them like the host
    /// path, and matches the host-routed chunk of the same rows within 1e-6 relative and the CPU
    /// oracle at the standard parity gate.
    #[test]
    fn chunk_device_phased_matches_production_bitwise() {
        // The mirror may only differ in WHERE the evaluates sit, never in what is computed:
        // both entries on the same input must read back bit-identical outputs and health.
        // The sweep driver runs this first and aborts the sweep on a mismatch.
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut data = Vec::new();
        for n in 0..8u32 {
            data.extend(input(n));
        }
        let xr = upload(&mut kernels, vec![8, 64], &data);
        let (plain, phased) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let plain = chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            let (phased, _phases) =
                chunk_device_phased(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            (plain, phased)
        };
        let bits = |v: Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let out_plain = values(&mut kernels, &plain.out).unwrap();
        let out_phased = values(&mut kernels, &phased.out).unwrap();
        let health_plain = values(&mut kernels, &plain.health).unwrap();
        let health_phased = values(&mut kernels, &phased.health).unwrap();
        assert_eq!(bits(out_plain), bits(out_phased), "phased moe output");
        assert_eq!(bits(health_plain), bits(health_phased), "phased moe health");
    }

    #[test]
    fn a_device_chunk_of_eight_rows_matches_the_host_chunk_and_the_expected_experts() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut want = Vec::new();
        let mut data = Vec::new();
        for n in 0..8u32 {
            let x = input(n);
            want.extend(host::moe_step(&mut cpu, &shape, &host_moe(&oracle, 0), &x).unwrap());
            data.extend(x);
        }
        let xr = upload(&mut kernels, vec![8, 64], &data);
        let device = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap()
        };
        assert_eq!(device.out.shape(), [8, 64], "output shape");
        // The ids come back in rank order: the set per row is the table row, and the gates
        // descend along the row, so the rank order is the table order.
        let ids = read_u32(&mut kernels, &device.trace.ids);
        let gates = values(&mut kernels, &device.trace.gates).unwrap();
        for (n, table) in LAYER0_ROUTES.iter().enumerate() {
            let (row, row_gates) = (&ids[n * 2..n * 2 + 2], &gates[n * 2..n * 2 + 2]);
            let (mut set, mut table_set) = (row.to_vec(), table.to_vec());
            set.sort();
            table_set.sort();
            assert_eq!(set, table_set, "row {n} expert set");
            assert!(
                row_gates[0] >= row_gates[1],
                "row {n} gates descend along the rank order: {row_gates:?}"
            );
            assert_eq!(row, table.as_slice(), "row {n} rank order with the gates");
            let gate_sum: f32 = row_gates.iter().sum();
            assert!(
                (gate_sum - 1.0).abs() <= 1e-6,
                "row {n} gates sum to {gate_sum}"
            );
        }
        // The health of the block: eight softmax rows sum to 8.
        let health = values(&mut kernels, &device.health).unwrap()[0];
        assert!((health - 8.0).abs() <= 1e-4, "health {health}");
        // The host-routed chunk of the same rows agrees, and routed the same experts.
        let (host_out, host_routes) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap()
        };
        for (n, routed) in host_routes.iter().enumerate() {
            assert_eq!(
                &ids[n * 2..n * 2 + 2],
                routed.indices.as_slice(),
                "row {n} against the host route"
            );
        }
        assert_close_relative(
            &values(&mut kernels, &device.out).unwrap(),
            &values(&mut kernels, &host_out).unwrap(),
            1e-6,
            "device against host chunk",
        );
        assert_parity(
            &values(&mut kernels, &device.out).unwrap(),
            &want,
            "device chunk against the oracle",
        );
    }

    /// A one-row device chunk equals the device decode block on the same row.
    #[test]
    fn a_device_chunk_of_one_row_equals_the_decode_block() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let x = input(3);
        let one = upload(&mut kernels, vec![1, 64], &x);
        let (device, decode) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let device =
                chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &one).unwrap();
            let decode =
                decode_step_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &one).unwrap();
            (device, decode)
        };
        let (decode, route) = decode;
        assert_eq!(
            read_u32(&mut kernels, &device.trace.ids),
            read_u32(&mut kernels, route.ids.ids()),
            "the ids of the decode block"
        );
        let health = values(&mut kernels, &device.health).unwrap()[0];
        assert!((health - 1.0).abs() <= 1e-5, "health {health}");
        assert_close_relative(
            &values(&mut kernels, &device.out).unwrap(),
            &values(&mut kernels, &decode).unwrap(),
            1e-6,
            "one row against the decode block",
        );
    }

    /// Sixteen rows, past the 8-row table. The expected routes come from the
    /// host path inside the test, not from hard-coded numbers.
    #[test]
    fn a_device_chunk_of_sixteen_rows_matches_the_host_derived_routes() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let data: Vec<f32> = (0..16u32).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![16, 64], &data);
        let (device, host) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let device =
                chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            let host = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            (device, host)
        };
        assert_eq!(device.out.shape(), [16, 64], "output shape");
        let (host_out, host_routes) = host;
        let ids = read_u32(&mut kernels, &device.trace.ids);
        assert_eq!(ids.len(), 32, "sixteen rows of two choices");
        for (n, routed) in host_routes.iter().enumerate() {
            assert_eq!(
                &ids[n * 2..n * 2 + 2],
                routed.indices.as_slice(),
                "row {n} against the host route"
            );
        }
        assert_close_relative(
            &values(&mut kernels, &device.out).unwrap(),
            &values(&mut kernels, &host_out).unwrap(),
            1e-6,
            "sixteen rows",
        );
    }

    /// A chunk of 1100 rows runs the multi-block merge sort of the pair
    /// keys. The top-k is 2, so the pair axis holds 2200 > 2048 keys and the MLX sort runs its
    /// multi-block merge, as a real 512-row chunk (4096 pairs) does; the rows come from the
    /// `input` pattern (period 13), so the keys hold many duplicate expert ids. The direct
    /// `chunk_device` call admits any row count (like the 16-row test, no state is built; the
    /// decoder would cap a chunk at its context limit), and the size stays below
    /// `MAX_CHUNK_ROWS` (2048) anyway. The device chunk matches the host chunk of the same rows
    /// within 1e-6 relative and routes the same experts per row, and the sorted ids read back
    /// non-decreasing over all 2200 pairs.
    #[test]
    fn a_device_chunk_of_1100_rows_runs_the_multi_block_pair_sort() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        const ROWS: u32 = 1100;
        let data: Vec<f32> = (0..ROWS).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![ROWS, 64], &data);
        let (device, host) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let device =
                chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            let host = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            (device, host)
        };
        assert_eq!(device.out.shape(), [ROWS, 64], "output shape");
        let (host_out, host_routes) = host;
        assert_eq!(host_routes.len(), ROWS as usize, "one route per row");
        let ids = read_u32(&mut kernels, &device.trace.ids);
        assert_eq!(ids.len(), 2 * ROWS as usize, "two choices per row");
        for (n, routed) in host_routes.iter().enumerate() {
            assert_eq!(
                &ids[n * 2..n * 2 + 2],
                routed.indices.as_slice(),
                "row {n} against the host route"
            );
        }
        let sorted = read_u32(&mut kernels, &device.trace.sorted_ids);
        assert_eq!(sorted.len(), ids.len(), "one sorted id per pair");
        assert!(
            sorted.is_sorted(),
            "the multi-block merge keeps the ids non-decreasing"
        );
        assert_close_relative(
            &values(&mut kernels, &device.out).unwrap(),
            &values(&mut kernels, &host_out).unwrap(),
            1e-6,
            "1100 rows through the multi-block pair sort",
        );
    }

    /// A shape whose router does not renormalize the top-k
    /// gates. `HostShape` is plain data, so a copy with `renorm` set to false builds the case
    /// with no production change; the weights are the same and only the gate rule differs. With
    /// no renormalization the gates are the raw top probabilities, the same F32 bits on both
    /// paths, and the device chunk matches the host chunk within 1e-6 relative.
    #[test]
    fn a_device_chunk_of_a_shape_without_renormalization_matches_the_host_chunk() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let mut shape = oracle.shape().clone();
        assert!(shape.moe.renorm, "the tiny model renormalizes by default");
        shape.moe.renorm = false;
        let data: Vec<f32> = (0..8u32).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![8, 64], &data);
        let (device, host) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let device =
                chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            let host = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
            (device, host)
        };
        let (host_out, host_routes) = host;
        let gates = values(&mut kernels, &device.trace.gates).unwrap();
        for (n, routed) in host_routes.iter().enumerate() {
            assert_eq!(
                &gates[n * 2..n * 2 + 2],
                routed.weights.as_slice(),
                "row {n} gates are the raw probabilities"
            );
        }
        assert_close_relative(
            &values(&mut kernels, &device.out).unwrap(),
            &values(&mut kernels, &host_out).unwrap(),
            1e-6,
            "renorm false",
        );
    }

    /// The sorted ids are non-decreasing (the order that
    /// `gather_qmm_sorted` requires, from `argsort` by definition) and the inverse permutation
    /// restores the token-major order.
    #[test]
    fn the_device_pair_sort_is_sorted_and_the_inverse_restores_token_order() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let data: Vec<f32> = (0..8u32).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![8, 64], &data);
        let device = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap()
        };
        let ids = read_u32(&mut kernels, &device.trace.ids);
        let sorted = read_u32(&mut kernels, &device.trace.sorted_ids);
        let order = read_u32(&mut kernels, &device.trace.order_p);
        let inverse = read_u32(&mut kernels, &device.trace.inverse);
        let pairs = ids.len();
        assert_eq!(pairs, 16, "eight rows of two choices");
        assert_eq!(
            sorted.len(),
            pairs,
            "the sorted ids hold one entry per pair"
        );
        assert_eq!(order.len(), pairs, "the order holds one entry per pair");
        assert_eq!(inverse.len(), pairs, "the inverse holds one entry per pair");
        assert!(
            sorted.is_sorted(),
            "the sorted ids are non-decreasing: {sorted:?}"
        );
        for i in 0..pairs {
            assert_eq!(
                inverse[order[i] as usize], i as u32,
                "inverse[order[{i}]] == {i}"
            );
            assert_eq!(
                sorted[i], ids[order[i] as usize],
                "sorted[{i}] is the key at position order[{i}]"
            );
        }
        let (mut by_row, mut by_expert) = (ids.clone(), sorted.clone());
        by_row.sort();
        by_expert.sort();
        assert_eq!(by_row, by_expert, "the sort keeps every expert");
    }

    /// The device tie rule of the argsort: x = 0 gives probabilities exactly 1/4, so the lowest
    /// two experts win with gates of 1/2, and the other row of the chunk keeps its own route.
    #[test]
    fn a_device_chunk_gives_a_tied_row_to_the_lowest_experts() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut data = input(0);
        data.extend([0.0; 64]);
        let xr = upload(&mut kernels, vec![2, 64], &data);
        let device = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap()
        };
        let ids = read_u32(&mut kernels, &device.trace.ids);
        let gates = values(&mut kernels, &device.trace.gates).unwrap();
        assert_eq!(&ids[0..2], &[2, 1], "row 0 keeps its own route");
        assert_eq!(&ids[2..4], &[0, 1], "the tied row takes the lowest experts");
        let tied: Vec<f32> = gates[2..4].to_vec();
        assert_eq!(tied, [0.5, 0.5], "the tied gates halve");
    }

    /// The device path builds graph nodes only: no import, no readback, no evaluate.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_device_chunk_makes_no_synchronization() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let data: Vec<f32> = (0..4u32).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![4, 64], &data);
        let before = kernels.qualification_sync_count();
        let device = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            chunk_device(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap()
        };
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "the device chunk makes no synchronization"
        );
        drop(device);
    }

    #[test]
    fn a_tied_row_inside_a_chunk_selects_the_lowest_experts() {
        // x = 0 gives logits 0 and probabilities exactly 1/4: the lowest two experts win, with
        // gates of 1/2. The other row of the chunk keeps its own route.
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut data = input(0);
        data.extend([0.0; 64]);
        let xr = upload(&mut kernels, vec![2, 64], &data);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (_, routes) = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
        assert_eq!(routes[0].indices, [2, 1]);
        assert_eq!(routes[1].indices, [0, 1]);
        assert_eq!(routes[1].weights, [0.5, 0.5]);
    }

    #[test]
    fn an_input_of_the_wrong_shape_is_a_bad_request() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let flat = upload(&mut kernels, vec![64], &[1.0; 64]);
        let narrow = upload(&mut kernels, vec![2, 32], &[1.0; 64]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        for (what, x) in [("rank one", &flat), ("width 32", &narrow)] {
            let failed = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), x);
            assert!(
                matches!(&failed, Err(EngError::BadRequest(m)) if m.contains("moe chunk input")),
                "{what}: {:?}",
                failed.as_ref().err()
            );
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn one_call_synchronizes_exactly_twice() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let data: Vec<f32> = (0..4u32).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![4, 64], &data);
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (out, _) = chunk(&mut ctx, &shape, w.layers[0].as_moe().unwrap(), &xr).unwrap();
        assert_eq!(kernels.qualification_sync_count(), before + 2);
        drop(out);
    }
}
