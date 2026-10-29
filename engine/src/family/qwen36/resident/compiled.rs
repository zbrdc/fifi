//! The compiled decode blocks of the resident Q36 decoder.
//!
//! The engine setting `qwen36.compile` (default false) turns the seam on: `delta::step` and the
//! device-routed `moe::step_device` run through the compiled closures of the kernels seam, and the
//! decoder admits the whole model at load — one warm-up trace per distinct program, so a trace
//! failure or a model the seam cannot compile is a typed load error, never a failure in the middle
//! of a request. B2/4/8 require explicit decoder preparation after that B1 warm-up; default loading
//! does not trace those row counts. The environment variable `QWEN36_G2A_COMPILE` stays as a
//! TEST-only override (it is read by `step_requested` in test builds alone and never reaches a
//! production build): a test decoder built with the setting off still compiles when the variable is
//! `1`, which is how the measurement harness keeps running. The HOST-ROUTED `moe::step` never
//! compiles the MoE block: it is the parity reference — the DeltaNet block compiles on a
//! host-routed decoder too, because its step ignores the routing — and the chunked prefill never
//! compiles (its chunk shapes vary).
//!
//! The compiled closures replay the two block op sequences in
//! `engine/src/kernels/mlx/native/guard.cpp`, calling the same native shims in the same order as
//! the uncompiled blocks of THIS branch, so the measurement compares like with like in BOTH
//! activation dtypes: the `bb` cast to F32 before the sigmoid, the F32 recurrence state and ring,
//! the `to_act` before the output projection, the BF16 (or F32) scales of the packed weights, the
//! F32 router, the F32 gates, and the ONE `mlx_minimum` clamp of the expert ids per layer. The
//! cache key mixes every baked constant, the activation dtype, and the scale dtype of each packed
//! weight (`kernels::compile_key`), so the F32 and BF16 decoders keep distinct traces. The block
//! outputs are row-contiguous MLX arrays (every block ends in a quantized matmul, a reshape view of
//! one, or a fresh elementwise node), so later uncompiled code reads them directly and none goes
//! through the readback net.
//!
//! What the seam does NOT compile, and why. The attention layers, the head, the embedding, and
//! the whole prefill stay uncompiled: the attention layers bake the RoPE position as a scalar of
//! each step (a different constant per token, so a compiled closure would retrace every position
//! and the cache would never hit), the head and the embedding serve shapes the seam did not
//! trace ([1, vocab] against the resident row reads), and the prefill runs chunk shapes that
//! differ from the decode shapes (a shapeless trace would change the numerics the gates compare).

use super::compile_plan::{DeltaLayerFacts, MoeLayerFacts, WeightFacts};
use super::delta::DeltaStep;
use super::moe::DeviceRoute;
use super::ops::{Consts, Ctx, dim};
use super::weights::{RDelta, RMoe};
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    kernels::mlx::native::compile::{CompileDeltaArgs, CompileMoeArgs, CompiledDeltaRows},
    kernels::{ResidentPackedWeight, compile_key},
    tensor::Dtype,
};

/// The load-time facts of one packed weight for the warm-up plan: the scale dtype key, the
/// quantization triple, and whether the weight carries biases (a weight without biases has no
/// compiled form, so the plan refuses it at load).
fn weight_facts(weight: &ResidentPackedWeight) -> Result<WeightFacts, EngError> {
    Ok(WeightFacts {
        scale_key: compile_key::dtype_key(weight.scales().dtype()),
        quant: crate::kernels::mlx::native::compile::quant_triple(weight)?,
        biases: weight.biases().is_some(),
    })
}

/// The load-time facts of one DeltaNet layer (index `layer`) for the warm-up plan: the baked
/// constants of its compiled step over the checked block dimensions and the weight facts of
/// qkv, z, b, a, out.
pub(super) fn delta_facts(
    layer: usize,
    shape: &HostShape,
    w: &RDelta,
    act: Dtype,
    eps: f32,
) -> Result<DeltaLayerFacts, EngError> {
    let d = &shape.delta;
    let taps = d
        .conv_kernel
        .checked_sub(1)
        .ok_or_else(|| EngError::BadRequest("resident delta kernel is empty".into()))?;
    let dims = compile_key::checked_delta_dims([
        dim(d.key_heads)?,
        dim(d.key_dim)?,
        dim(d.value_heads)?,
        dim(d.value_dim)?,
        dim(d.conv_dim())?,
        dim(taps)?,
    ])?;
    Ok(DeltaLayerFacts {
        layer,
        hidden: dim(shape.hidden)?,
        dims,
        eps_bits: eps.to_bits(),
        act: compile_key::dtype_key(act),
        weights: [
            weight_facts(&w.qkv)?,
            weight_facts(&w.z)?,
            weight_facts(&w.b)?,
            weight_facts(&w.a)?,
            weight_facts(&w.out)?,
        ],
    })
}

/// The load-time facts of one MoE layer (index `layer`) for the warm-up plan: the baked
/// constants of its compiled device-routed step and the weight facts of the shared gate, up,
/// down weights and the gate, up, down expert stacks.
pub(super) fn moe_facts(
    layer: usize,
    shape: &HostShape,
    w: &RMoe,
    act: Dtype,
) -> Result<MoeLayerFacts, EngError> {
    let m = &shape.moe;
    let experts = dim(m.experts)?;
    Ok(MoeLayerFacts {
        layer,
        hidden: dim(shape.hidden)?,
        experts,
        top_k: dim(m.top_k)?,
        renorm: u32::from(m.renorm),
        clamp: experts
            .checked_sub(1)
            .ok_or_else(|| EngError::BadRequest("resident moe needs experts".into()))?,
        act: compile_key::dtype_key(act),
        weights: [
            weight_facts(&w.shared_gate)?,
            weight_facts(&w.shared_up)?,
            weight_facts(&w.shared_down)?,
            weight_facts(&w.gate)?,
            weight_facts(&w.up)?,
            weight_facts(&w.down)?,
        ],
    })
}

/// Whether this step runs its compiled form: the engine setting, or — in test builds only — the
/// setting OR the `QWEN36_G2A_COMPILE` measurement override. A production build reads no
/// environment variable here.
#[cfg(not(test))]
pub(super) fn step_requested(setting: bool) -> bool {
    setting
}

/// The test form of [`step_requested`]: the `QWEN36_G2A_COMPILE` measurement override keeps
/// working on decoders built with the setting off, so the measurement harness and the gates run
/// unchanged. Unset is off, `1` is on, any other value panics like the other Q36 test knobs.
#[cfg(test)]
pub(super) fn step_requested(setting: bool) -> bool {
    setting || requested()
}

/// Reads `QWEN36_G2A_COMPILE` once per process (test builds only): unset is off, `1` is on, any
/// other value panics like the other Q36 test knobs.
#[cfg(test)]
pub(super) fn requested() -> bool {
    use std::sync::OnceLock;
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| match std::env::var("QWEN36_G2A_COMPILE") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        other => panic!("QWEN36_G2A_COMPILE must be unset or 1, got {other:?}"),
    })
}

/// Runs one DeltaNet decode step through the compiled closure. The caller (`delta::step`) has
/// already checked the state shapes. The step runs in the activation dtype of the decoder
/// (`ctx.consts.act`), which the seam bakes into the closure and the cache key.
pub(super) fn delta_step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RDelta,
    rec: &crate::tensor::ExecutionTensor,
    ring: &crate::tensor::ExecutionTensor,
    x: &crate::tensor::ExecutionTensor,
) -> Result<DeltaStep, EngError> {
    let args = delta_args(ctx.consts, shape, w, rec, ring, x)?;
    let got = ctx.kernels.compiled_delta_step(&args)?;
    Ok(DeltaStep {
        out: got.out,
        rec: got.rec,
        ring: got.ring,
    })
}

/// Runs B2/4/8 rows with distinct persistent state pairs in row order.
/// Slot zero belongs in the base arguments; the extras contain exactly B-1 pairs.
pub(super) fn delta_rows(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RDelta,
    states: &[(
        &crate::tensor::ExecutionTensor,
        &crate::tensor::ExecutionTensor,
    )],
    x: &crate::tensor::ExecutionTensor,
) -> Result<CompiledDeltaRows, EngError> {
    if !matches!(states.len(), 2 | 4 | 8) || x.shape() != [dim(states.len())?, dim(shape.hidden)?] {
        return Err(EngError::BadRequest(
            "compiled DeltaNet rows need B state pairs and [B, H] input for B in 2, 4, or 8".into(),
        ));
    }
    let (rec, ring) = states[0];
    let args = delta_args(ctx.consts, shape, w, rec, ring, x)?;
    ctx.kernels.compiled_delta_rows(&args, &states[1..])
}

/// Shares the B1 constructor without changing its graph or packing persistent state.
fn delta_args<'a>(
    consts: &'a Consts,
    shape: &HostShape,
    w: &'a RDelta,
    rec: &'a crate::tensor::ExecutionTensor,
    ring: &'a crate::tensor::ExecutionTensor,
    x: &'a crate::tensor::ExecutionTensor,
) -> Result<CompileDeltaArgs<'a>, EngError> {
    let d = &shape.delta;
    let hidden = dim(shape.hidden)?;
    Ok(CompileDeltaArgs {
        x,
        rec,
        ring,
        packed: [&w.qkv, &w.z, &w.b, &w.a, &w.out],
        plain: [
            &w.conv_taps,
            &w.conv_last,
            &w.neg_exp_a_log,
            &w.dt_bias,
            &w.norm,
        ],
        consts: [&consts.zero, &consts.qk_eps, &consts.sqrt_kd],
        hidden,
        dims: [
            dim(d.key_heads)?,
            dim(d.key_dim)?,
            dim(d.value_heads)?,
            dim(d.value_dim)?,
            dim(d.conv_dim())?,
            dim(d
                .conv_kernel
                .checked_sub(1)
                .ok_or_else(|| EngError::BadRequest("resident delta kernel is empty".into()))?)?,
        ],
        eps: shape.eps,
        act: consts.act,
    })
}

/// Runs B1/2/4/8 device-routed MoE rows through the existing compiled seam.
/// B1 retains its legacy graph and route ranks. Multirow routes retain every health row,
/// which the caller must check before publication. The activation dtype is `ctx.consts.act`.
pub(super) fn moe_step_device(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &crate::tensor::ExecutionTensor,
) -> Result<(crate::tensor::ExecutionTensor, DeviceRoute), EngError> {
    let m = &shape.moe;
    let (hidden, experts) = (dim(shape.hidden)?, dim(m.experts)?);
    let supported_rows = matches!(x.shape(), &[1 | 2 | 4 | 8, width] if width == hidden);
    if !supported_rows || w.router_t.shape() != [hidden, experts] {
        return Err(EngError::BadRequest(
            "compile seam inputs differ from the moe block shapes".into(),
        ));
    }
    let args = CompileMoeArgs {
        x,
        router_t: &w.router_t,
        shared: [&w.shared_gate, &w.shared_up, &w.shared_down],
        shared_scalar_t: &w.shared_scalar_t,
        stacks: [&w.gate, &w.up, &w.down],
        hidden,
        experts,
        top_k: dim(m.top_k)?,
        renorm: m.renorm,
        act: ctx.consts.act,
    };
    let got = ctx.kernels.compiled_moe_step(&args)?;
    Ok((
        got.out,
        DeviceRoute {
            // The closure body clamps the ids below the expert count with its ONE `mlx_minimum`
            // node, so the route wraps them as already clamped instead of building a second clamp
            // node.
            ids: crate::kernels::BoundedIndices::from_clamped(got.ids, experts),
            gates: got.gates,
            health: got.health,
        },
    ))
}

/// The `compile_counters` object of the reports: the traced, cached, and
/// total calls over both blocks, then the per-block rows (cache key, calls, traces, the seconds
/// of the first calls) and the cache clears. `kernels` supplies the process-wide counters.
#[cfg(test)]
fn counters(kernels: &crate::kernels::MlxKernels) -> serde_json::Value {
    let stats = kernels.compile_counters();
    let block = |b: &crate::kernels::mlx::compile_stats::Block| {
        serde_json::json!({
            "fun_id": b.fun_id.map(|id| format!("{id:#x}")),
            "calls": b.calls,
            "traces": b.traces,
            "cache_hits": b.calls - b.traces,
            "call_seconds": b.call_seconds,
        })
    };
    let (calls, traces, hits) = stats.totals();
    serde_json::json!({
        "calls": calls,
        "traces": traces,
        "cache_hits": hits,
        "delta": block(&stats.delta),
        "moe": block(&stats.moe),
        "cache_clears": stats.cache_clears,
        "note": "traces are the native closure-body counts: a body entry means MLX traced the \
                 closure (or returned the uncompiled function, which runs the body on every \
                 call), a cache hit replays the compiled tape without entering the body, and \
                 cache_hits = calls - traces",
    })
}

/// The value of the test switch and the `compile_counters` object, the two fields that the
/// `decode_timing` and `graph_count` reports carry.
#[cfg(test)]
pub(super) fn switch_and_counters(
    kernels: &crate::kernels::MlxKernels,
) -> (bool, serde_json::Value) {
    (requested(), counters(kernels))
}

/// The compile rows of the real-checkpoint reports: the switch, the counters, and the per-block
/// detail. `kernels` supplies the process-wide counters.
#[cfg(test)]
pub(super) fn report(kernels: &crate::kernels::MlxKernels) -> serde_json::Value {
    serde_json::json!({
        "enabled": requested(),
        "compile_counters": counters(kernels),
    })
}
