//! Staged Flint graph composition over the existing MLX operations, not an executor.
//! This module needs a native macOS compile/test gate before integration. No fused
//! no-sink attention or softmax-only CUDA router is used for Flint.

use crate::{
    EngError,
    kernels::{BinaryOp, MlxKernels, ResidentPackedWeight, RopeStep},
    tensor::{Dtype, ExecutionTensor},
};

/// Dense weights are already transposed `[in, out]`; packed weights retain
/// checkpoint `[out, in]` layout, as required by the existing `mm`/`qmm` seams.
pub enum Projection<'a> {
    Dense(&'a ExecutionTensor),
    Packed(&'a ResidentPackedWeight),
}

pub struct SwiGluWeights<'a> {
    pub gate: Projection<'a>,
    pub up: Projection<'a>,
    pub down: Projection<'a>,
}

fn project(
    k: &mut MlxKernels,
    w: &Projection<'_>,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    match w {
        Projection::Dense(w) => k.mm(x, w),
        Projection::Packed(w) => k.qmm(x, w),
    }
}

/// Routed, dense and shared experts all use down(SiLU(gate(x))*up(x)).
/// In particular there is NO fourth/shared scalar sigmoid projection.
pub fn expert(
    k: &mut MlxKernels,
    w: &SwiGluWeights<'_>,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let gate = project(k, &w.gate, x)?;
    let up = project(k, &w.up, x)?;
    let activated = k.silu(&gate)?;
    let hidden = k.binary(BinaryOp::Mul, &activated, &up)?;
    project(k, &w.down, &hidden)
}

/// Reuse the weighted-combine primitive; shared always has coefficient one.
/// `expert_rows` is `[K,H]`, `weights` is `[K,1,1]`, shared is `[1,H]`, all F32.
pub fn combine(
    k: &mut MlxKernels,
    expert_rows: &ExecutionTensor,
    weights: &ExecutionTensor,
    shared: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let routed = k.weighted_row_sum(expert_rows, weights)?;
    k.binary(BinaryOp::Add, &routed, shared)
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// QK head RMSNorm uses gamma directly, not Gemma's 1+gamma or V normalization.
/// Q `[32,T,128]`, K `[4,T,128]`, per-family gamma vectors `[128]`.
/// `rope=None` is REQUIRED on full layers; local layers pass their absolute
/// chunk start. The caller obtains that layer policy from validated metadata.
pub fn normalize_qk(
    kernels: &mut MlxKernels,
    qk: (&ExecutionTensor, &ExecutionTensor),
    gamma: (&ExecutionTensor, &ExecutionTensor),
    eps: f32,
    rope: Option<u32>,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let (q, k) = qk;
    let &[32, rows, 128] = q.shape() else {
        return Err(bad("Flint Q must be [32,T,128]"));
    };
    if rows == 0
        || k.shape() != [4, rows, 128]
        || gamma.0.shape() != [128]
        || gamma.1.shape() != [128]
    {
        return Err(bad("Flint QK head/gamma shapes differ"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(bad("Flint RMSNorm epsilon must be finite and positive"));
    }
    let q = kernels.cast(q, Dtype::F32)?;
    let k = kernels.cast(k, Dtype::F32)?;
    let gq = kernels.cast(gamma.0, Dtype::F32)?;
    let gk = kernels.cast(gamma.1, Dtype::F32)?;
    let mut q = kernels.rms_norm(&q, &gq, eps)?;
    let mut k = kernels.rms_norm(&k, &gk, eps)?;
    if let Some(pos) = rope {
        kernels.rope(
            &mut q,
            &mut k,
            &RopeStep {
                pos,
                theta: 10_000.0,
                part: 128,
            },
        )?;
    }
    Ok((q, k))
}

/// Scaled QK scores plus an additive absolute-position causal/window mask.
/// Mask `[T,S]` uses 0 for visible keys and -inf for masked keys. Build it
/// from `reference::ChunkWindow`, NEVER from ring-slot order or query-zero.
/// `scale` is a resident scalar `1/sqrt(128)`, shared by every layer/chunk.
pub fn qk_scores(
    k: &mut MlxKernels,
    q: &ExecutionTensor,
    keys: &ExecutionTensor,
    scale: &ExecutionTensor,
    mask: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let &[32, rows, 128] = q.shape() else {
        return Err(bad("Flint Q must be [32,T,128]"));
    };
    let &[4, slots, 128] = keys.shape() else {
        return Err(bad("Flint K must be [4,S,128]"));
    };
    if rows == 0 || slots == 0 || mask.shape() != [rows, slots] || scale.shape() != [1] {
        return Err(bad("Flint score mask/scale shapes differ"));
    }
    let keys = k.repeat(keys, 8, 0)?;
    let kt = k.transpose(&keys, &[0, 2, 1])?;
    let scores = k.batched_mm(q, &kt)?;
    let scaled = k.binary(BinaryOp::Mul, &scores, scale)?;
    k.binary(BinaryOp::Add, &scaled, mask)
}

/// Scores `[32,T,S]` are already QK-scaled and masked; V is `[4,S,128]`.
/// Sink `[32]` is the raw checkpoint `softmax_offset`, NOT multiplied by the
/// QK scale. Concatenate it once, softmax once, discard its probability before
/// the value product. Thus the sink never contributes V, including all-masked
/// rows. Pass the COMPLETE key window/prefix, never call this independently on
/// partitions then combine outputs. Partitioned kernels must merge sink-free
/// statistics before adding the sink, as `reference::AttentionPartial` does.
/// This composition materializes scores and is NOT a memory/performance claim.
pub fn sink_attention(
    k: &mut MlxKernels,
    scores: &ExecutionTensor,
    values: &ExecutionTensor,
    sink: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let &[32, rows, slots] = scores.shape() else {
        return Err(bad("Flint scores must be [32,T,S]"));
    };
    if rows == 0
        || slots == 0
        || slots == u32::MAX
        || values.shape() != [4, slots, 128]
        || sink.shape() != [32]
    {
        return Err(bad("Flint sink/value shapes differ"));
    }
    // F32 softmax/accumulation even when stored activations or sinks are BF16.
    let scores = k.cast(scores, Dtype::F32)?;
    let sink = k.cast(sink, Dtype::F32)?;
    let sink = k.reshape(&sink, &[32, 1, 1])?;
    let sink = k.repeat(&sink, rows, 1)?;
    let joined = k.concatenate(&[&scores, &sink], 2)?;
    let probs = k.softmax(&joined)?;
    let real = k.slice(&probs, &[0, 0, 0], &[32, rows, slots])?;
    let values = k.cast(values, Dtype::F32)?;
    let values = k.repeat(&values, 8, 0)?;
    k.batched_mm(&real, &values)
}

/// Host-routed bridge until a separately qualified sigmoid device router exists.
/// Finite validation happens on logits before sigmoid. No CUDA softmax reuse.
pub fn route(
    k: &mut MlxKernels,
    logits: &ExecutionTensor,
) -> Result<crate::family::route::Routed, EngError> {
    if logits.shape() != [1, 256] {
        return Err(bad("Flint logits must be [1,256]"));
    }
    let logits = k.cast(logits, Dtype::F32)?;
    let host = k.readback(&logits)?;
    let bytes = host
        .host_bytes()
        .ok_or_else(|| bad("Flint router readback is not host memory"))?;
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    super::math::route(&values)
}

#[cfg(test)]
#[path = "mlx_math_tests.rs"]
mod tests;
