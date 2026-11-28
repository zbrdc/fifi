//! The resident Gemma 4 Humanizer decode step on one CUDA device.
//!
//! This module composes the checked resident operations over a Q8_0
//! checkpoint: a Q8_0 embedding gather, forty-eight decoder layers (attention over
//! the KV8 planes, split-half RoPE, the GELU-tanh gated feed-forward, the residual
//! sums and the layer scalar), the final RMSNorm, the tied-embedding logit
//! projection and the logit softcap.
//!
//! Every launch is checked by its wrapper before the native call. This module only
//! orders the launches and owns the buffers. It uploads the id, position and cache-length
//! words each step (so it does allocate per step, and that free synchronizes the device).
//!
//! Limits:
//! - Prefill runs bounded batches through the batched step (single tokens take the m = 1
//!   path of the same step).
//! - The per-head RMSNorm still runs one launch per head.
//! - The logits stay on the device; the device sampler draws from them and only the chosen
//!   id, its logprob and the sampler status word come back.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "native-cuda resident helpers beyond the serving path"
    )
)]

pub(crate) mod bolt;
mod bolt_graph;
mod step;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod window_tests;

use crate::EngError;
use crate::family::kv_ring::KvRing;
use crate::family::project::cuda::matrix::{
    upload_matrix as matrix, upload_matrix_bytes as matrix_from_bytes,
};
use crate::kernels::cuda::prefill::attn_prefill::{WideAttention, WideGeometry};
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Shape};
use crate::kernels::cuda::resident::matvec_q::MatvecCall;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, NativeWeight};
use crate::weights::{GgufType, GgufWeights};

pub(crate) const LAYERS: usize = 48;
/// Values per hidden row.
pub(crate) const HIDDEN: usize = 3840;
/// Rows of the tied embedding and of the logit projection.
pub(crate) const VOCAB: usize = 262_144;
pub(crate) const HEADS: usize = 16;
/// KV cells held per head: the 8,192-token combined context of the goal.
pub(crate) const CELLS: usize = super::gguf::HUMANIZER_CELLS as usize;
pub(crate) const SOFT_CAP: f32 = 30.0;
pub(crate) const FFN: usize = 15360;

pub(crate) struct Vector {
    pub(crate) buf: DeviceBuffer,
    len: usize,
}

impl Vector {
    fn bytes(&self) -> usize {
        self.len * 4
    }
}

pub(crate) struct Kv {
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
    shape: Kv8Shape,
    /// The ring geometry the planes were built from, and the context their cells were counted for.
    geometry: KvRing,
    context: usize,
}

pub(crate) struct Layer {
    attn_norm: Vector,
    q_norm: Vector,
    k_norm: Vector,
    q: NativeWeight,
    k: NativeWeight,
    v: Option<NativeWeight>,
    out: NativeWeight,
    post_attn_norm: Vector,
    ffn_norm: Vector,
    gate: NativeWeight,
    up: NativeWeight,
    down: NativeWeight,
    post_ffw_norm: Vector,
    scale: f32,
    global: bool,
    head: usize,
    n_kv: usize,
    kv: Kv,
}

/// The per-step buffers, sized once at load.
pub(crate) struct Scratch {
    pub(crate) h: [Vector; 4],
    pub(crate) nrm: Vector,
    pub(crate) nrm2: Vector,
    pub(crate) q_normed: Vector,
    pub(crate) k_normed: Vector,
    pub(crate) v_normed: Vector,
    pub(crate) ones: Vector,
    pub(crate) q: Vector,
    pub(crate) k: Vector,
    pub(crate) v: Vector,
    pub(crate) ctx_out: Vector,
    pub(crate) proj: Vector,
    pub(crate) gate: Vector,
    pub(crate) up: Vector,
    pub(crate) mixed: Vector,
    pub(crate) down: Vector,
    pub(crate) q8_hidden: DeviceBuffer,
    pub(crate) q8_ctx: DeviceBuffer,
    pub(crate) q8_mixed: DeviceBuffer,
    pub(crate) q8_head: DeviceBuffer,
    pub(crate) ids: DeviceBuffer,
    pub(crate) pos: DeviceBuffer,
    pub(crate) kv_len: DeviceBuffer,
    /// The gather's status word: a bad token id sets a bit here.
    pub(crate) status: DeviceBuffer,
}

pub(crate) struct Gemma {
    pub(crate) ctx: CudaContext,
    pub(crate) stream: DeviceStream,
    pub(crate) layers: Vec<Layer>,
    pub(crate) embed: NativeWeight,
    pub(crate) out_norm: Vector,
    pub(crate) rope_freqs: Option<Vector>,
    pub(crate) scratch: Scratch,
    pub(crate) logits: Vector,
    pub(crate) logits_capped: Vector,
    pub(crate) eps: f32,
    /// The wide prefill attention of the global layers, when the device runs it.
    pub(crate) wide_prefill: Option<WideAttention>,
    /// The sliding window of the local layers, from the file's header.
    pub(crate) window: usize,
}

/// What the wide prefill attention sees of the global layers: one kv head of 512-wide planes
/// read by every query head, at most one prefill batch of rows.
fn wide_geometry() -> WideGeometry {
    WideGeometry::unscaled(HEADS, 1, CELLS)
}

pub(crate) fn load(path: &std::path::Path) -> Result<Gemma, EngError> {
    let super::HumanizerFile { weights, window } = super::open_humanizer_weights(path)?;
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())?;
    let stream = ctx.stream()?;
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let global = layer % 6 == 5;
        let head = if global { 512 } else { 256 };
        let n_kv = if global { 1 } else { 8 };
        let base = format!("blk.{layer}.");
        // A local layer reaches the last `window` positions, so it keeps a ring of that many
        // cells plus one prefill batch; a global layer keeps every position.
        let geometry = KvRing::sliding(if global { 0 } else { window }, Gemma::PREFILL_BATCH);
        let shape = Kv8Shape::of_layer(geometry, CELLS, 1, n_kv, head, Kv8CodeScale::StoredHalf);
        // The planes take the shape's cells: there is one derivation of a ring's cell count.
        let cells = shape.cells;
        let kv = Kv {
            k_codes: zero(&ctx, n_kv * cells * head)?,
            k_scales: zero(&ctx, n_kv * cells * (head / 32) * 2)?,
            v_codes: zero(&ctx, n_kv * cells * head)?,
            v_scales: zero(&ctx, n_kv * cells * (head / 32) * 2)?,
            shape,
            geometry,
            context: CELLS,
        };
        layers.push(Layer {
            attn_norm: vector(&weights, &ctx, &format!("{base}attn_norm.weight"), HIDDEN)?,
            q_norm: vector(&weights, &ctx, &format!("{base}attn_q_norm.weight"), head)?,
            k_norm: vector(&weights, &ctx, &format!("{base}attn_k_norm.weight"), head)?,
            q: matrix(
                &weights,
                &ctx,
                &format!("{base}attn_q.weight"),
                HIDDEN,
                HEADS * head,
            )?,
            k: matrix(
                &weights,
                &ctx,
                &format!("{base}attn_k.weight"),
                HIDDEN,
                n_kv * head,
            )?,
            v: if global {
                None
            } else {
                Some(matrix(
                    &weights,
                    &ctx,
                    &format!("{base}attn_v.weight"),
                    HIDDEN,
                    n_kv * head,
                )?)
            },
            out: matrix(
                &weights,
                &ctx,
                &format!("{base}attn_output.weight"),
                HEADS * head,
                HIDDEN,
            )?,
            post_attn_norm: vector(
                &weights,
                &ctx,
                &format!("{base}post_attention_norm.weight"),
                HIDDEN,
            )?,
            ffn_norm: vector(&weights, &ctx, &format!("{base}ffn_norm.weight"), HIDDEN)?,
            gate: matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_gate.weight"),
                HIDDEN,
                FFN,
            )?,
            up: matrix(&weights, &ctx, &format!("{base}ffn_up.weight"), HIDDEN, FFN)?,
            down: matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_down.weight"),
                FFN,
                HIDDEN,
            )?,
            post_ffw_norm: vector(
                &weights,
                &ctx,
                &format!("{base}post_ffw_norm.weight"),
                HIDDEN,
            )?,
            scale: scalar(&weights, &format!("{base}layer_output_scale.weight"))?,
            global,
            head,
            n_kv,
            kv,
        });
    }
    let embed = matrix(&weights, &ctx, "token_embd.weight", HIDDEN, VOCAB)?;
    let out_norm = vector(&weights, &ctx, "output_norm.weight", HIDDEN)?;
    let rope_freqs = Some(vector(&weights, &ctx, "rope_freqs.weight", 256)?);
    let eps = epsilon(&weights)?;
    let h = [
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
    ];
    let scratch = Scratch {
        h,
        nrm: vector_filled(&ctx, HIDDEN, 0.0)?,
        nrm2: vector_filled(&ctx, HIDDEN, 0.0)?,
        q_normed: vector_filled(&ctx, HEADS * 512, 0.0)?,
        k_normed: vector_filled(&ctx, 8 * 256, 0.0)?,
        v_normed: vector_filled(&ctx, 8 * 512, 0.0)?,
        ones: vector_filled(&ctx, 512, 1.0)?,
        q: vector_filled(&ctx, HEADS * 512, 0.0)?,
        k: vector_filled(&ctx, 8 * 256, 0.0)?,
        v: vector_filled(&ctx, 8 * 256, 0.0)?,
        ctx_out: vector_filled(&ctx, HEADS * 512, 0.0)?,
        proj: vector_filled(&ctx, HIDDEN, 0.0)?,
        gate: vector_filled(&ctx, FFN, 0.0)?,
        up: vector_filled(&ctx, FFN, 0.0)?,
        mixed: vector_filled(&ctx, FFN, 0.0)?,
        down: vector_filled(&ctx, HIDDEN, 0.0)?,
        q8_hidden: zero(&ctx, q8_1_bytes(HIDDEN, 1).expect("hidden q8 size"))?,
        q8_ctx: zero(&ctx, q8_1_bytes(HEADS * 512, 1).expect("context q8 size"))?,
        q8_mixed: zero(&ctx, q8_1_bytes(FFN, 1).expect("mixed q8 size"))?,
        q8_head: zero(&ctx, q8_1_bytes(512, 1).expect("head q8 size"))?,
        ids: zero(&ctx, 4)?,
        pos: zero(&ctx, 4)?,
        kv_len: zero(&ctx, 4)?,
        status: zero(&ctx, 4)?,
    };
    let logits = vector_filled(&ctx, VOCAB, 0.0)?;
    let logits_capped = vector_filled(&ctx, VOCAB, 0.0)?;
    // The wide prefill attention is built wherever the device gate passes and starts as the
    // override or the default says.
    let wide_prefill = WideAttention::load(&ctx, "gemma4 humanizer", wide_geometry())?;
    Ok(Gemma {
        ctx,
        stream,
        layers,
        embed,
        out_norm,
        rope_freqs,
        scratch,
        logits,
        logits_capped,
        eps,
        wide_prefill,
        window,
    })
}

fn zero(ctx: &CudaContext, bytes: usize) -> Result<DeviceBuffer, EngError> {
    ctx.upload(&vec![0u8; bytes])
}

fn vector_filled(ctx: &CudaContext, len: usize, value: f32) -> Result<Vector, EngError> {
    let bytes: Vec<u8> = value.to_le_bytes().repeat(len);
    Ok(Vector {
        buf: ctx.upload(&bytes)?,
        len,
    })
}

/// The shared matvec of the first `rows` rows of `weight` over the `m` q8_1 rows `x_q8`, into the
/// rows of `y`, in launches of up to 8 columns; the column-invariant entry makes a column the same
/// bytes whatever launch it sits in, which the window and chunk equalities of the step rely on.
fn matvec(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &NativeWeight,
    rows: usize,
    x_q8: &DeviceBuffer,
    y: &Vector,
    m: usize,
) -> Result<(), EngError> {
    let k = weight.k();
    let row_bytes = q8_1_bytes(k, 1).expect("activation row size");
    for first in (0..m).step_by(8) {
        let columns = 8.min(m - first);
        let call = MatvecCall {
            ty: weight.quant(),
            w: weight.buffer().range(0, weight.buffer().bytes())?,
            x_q8: x_q8.range(first * row_bytes, columns * row_bytes)?,
            y: y.buf.range(first * rows * 4, columns * rows * 4)?,
            k,
            rows,
            ncols: columns,
        };
        ctx.matvec_q_invariant(stream, call)?;
    }
    Ok(())
}

fn vector(
    weights: &GgufWeights,
    ctx: &CudaContext,
    name: &str,
    len: usize,
) -> Result<Vector, EngError> {
    let (ty, bytes) = weights.read_tensor(name)?;
    if ty != GgufType::F32 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} is {}, expected F32",
            ty.name()
        )));
    }
    if bytes.len() != len * 4 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} holds {} bytes, expected {}",
            bytes.len(),
            len * 4
        )));
    }
    Ok(Vector {
        buf: ctx.upload(&bytes)?,
        len,
    })
}

fn scalar(weights: &GgufWeights, name: &str) -> Result<f32, EngError> {
    let (ty, bytes) = weights.read_tensor(name)?;
    if ty != GgufType::F32 || bytes.len() != 4 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} is not one F32 value"
        )));
    }
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn epsilon(weights: &GgufWeights) -> Result<f32, EngError> {
    use crate::weights::MetaValue;
    match weights
        .header()
        .meta("gemma4.attention.layer_norm_rms_epsilon")
    {
        Some(MetaValue::F32(value)) => Ok(*value),
        Some(MetaValue::F64(value)) => Ok(*value as f32),
        _ => Err(EngError::Checkpoint(
            "gguf model: the normalization epsilon is missing".to_owned(),
        )),
    }
}
