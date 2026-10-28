//! The resident Gemma 4 Humanizer decode step on one CUDA device.
//!
//! This module composes the checked resident operations over the pinned Q8_0
//! checkpoint: a Q8_0 embedding gather, forty-eight decoder layers (attention over
//! the KV8 planes, split-half RoPE, the GELU-tanh gated feed-forward, the residual
//! sums and the layer scalar), the final RMSNorm, the tied-embedding logit
//! projection and the logit softcap.
//!
//! Every launch is checked by its wrapper before the native call. This module only
//! orders the launches and owns the buffers. It uploads the id, position and cache-length
//! words each step (so it does allocate per step, and that free synchronizes the device).
//!
//! Limits of this slice:
//! - Prefill runs bounded batches through the batched step (single tokens take the m = 1
//!   path of the same step).
//! - The per-head RMSNorm still runs one launch per head.
//! - The logits stay on the device; the device sampler draws from them and only the chosen
//!   id, its logprob and the sampler status word come back.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged until the executor is wired to the worker")
)]

mod step;
#[cfg(test)]
mod tests;

use crate::EngError;
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Shape};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};
use crate::weights::{GgufType, GgufWeights};

/// Decoder layers of the pinned checkpoint.
pub(crate) const LAYERS: usize = 48;
/// Values per hidden row.
pub(crate) const HIDDEN: usize = 3840;
/// Rows of the tied embedding and of the logit projection.
pub(crate) const VOCAB: usize = 262_144;
/// Query heads of every layer.
pub(crate) const HEADS: usize = 16;
/// KV cells held per head: the 8,192-token combined context of the goal.
pub(crate) const CELLS: usize = 8192;
/// Sliding window of a local layer.
pub(crate) const LOCAL_WINDOW: usize = 1024;
/// Logit softcap of the pinned checkpoint.
pub(crate) const SOFT_CAP: f32 = 30.0;
/// Feed-forward width.
pub(crate) const FFN: usize = 15360;

/// One Q8_0 matrix resident on the device, raw checkpoint blocks.
pub(crate) struct Matrix {
    pub(crate) buf: DeviceBuffer,
    rows: usize,
    k: usize,
}

impl Matrix {
    fn bytes(&self) -> usize {
        self.rows * (self.k / 32) * 34
    }
}

/// One Q8_0 matrix resident on the device in the split layout gemma_matvec reads: one
/// int8 quant plane per row, then one half scale per 32-value block.
pub(crate) struct PackedMatrix {
    pub(crate) qs: DeviceBuffer,
    pub(crate) scales: DeviceBuffer,
    rows: usize,
    k: usize,
}

impl PackedMatrix {
    fn qs_bytes(&self) -> usize {
        self.rows * self.k
    }

    fn scales_bytes(&self) -> usize {
        self.rows * (self.k / 32) * 2
    }
}

/// One F32 vector resident on the device.
pub(crate) struct Vector {
    pub(crate) buf: DeviceBuffer,
    len: usize,
}

impl Vector {
    fn bytes(&self) -> usize {
        self.len * 4
    }
}

/// The four KV8 planes of one layer.
pub(crate) struct Kv {
    k_codes: DeviceBuffer,
    k_scales: DeviceBuffer,
    v_codes: DeviceBuffer,
    v_scales: DeviceBuffer,
    shape: Kv8Shape,
}

/// One decoder layer's weights and cache planes.
pub(crate) struct Layer {
    attn_norm: Vector,
    q_norm: Vector,
    k_norm: Vector,
    q: PackedMatrix,
    k: PackedMatrix,
    v: Option<PackedMatrix>,
    out: PackedMatrix,
    post_attn_norm: Vector,
    ffn_norm: Vector,
    gate: PackedMatrix,
    up: PackedMatrix,
    down: PackedMatrix,
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
}

/// The loaded model and its device state.
pub(crate) struct Gemma {
    pub(crate) ctx: CudaContext,
    pub(crate) stream: DeviceStream,
    pub(crate) layers: Vec<Layer>,
    pub(crate) embed: Matrix,
    pub(crate) out_norm: Vector,
    pub(crate) rope_freqs: Option<Vector>,
    pub(crate) scratch: Scratch,
    pub(crate) logits: Vector,
    pub(crate) logits_capped: Vector,
    pub(crate) eps: f32,
}

/// Loads the pinned Humanizer Q8_0 checkpoint onto the device.
pub(crate) fn load(path: &std::path::Path, header_sha256: Option<&str>) -> Result<Gemma, EngError> {
    let weights = super::open_humanizer_weights(path, header_sha256)?;
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())?;
    let stream = ctx.stream()?;
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let global = layer % 6 == 5;
        let head = if global { 512 } else { 256 };
        let n_kv = if global { 1 } else { 8 };
        let base = format!("blk.{layer}.");
        let kv = Kv {
            k_codes: zero(&ctx, n_kv * CELLS * head)?,
            k_scales: zero(&ctx, n_kv * CELLS * (head / 32) * 2)?,
            v_codes: zero(&ctx, n_kv * CELLS * head)?,
            v_scales: zero(&ctx, n_kv * CELLS * (head / 32) * 2)?,
            shape: Kv8Shape {
                rows: 1,
                n_kv_head: n_kv,
                head_dim: head,
                cells: CELLS,
                code_scale: Kv8CodeScale::UnroundedF32,
            },
        };
        layers.push(Layer {
            attn_norm: vector(&weights, &ctx, &format!("{base}attn_norm.weight"), HIDDEN)?,
            q_norm: vector(&weights, &ctx, &format!("{base}attn_q_norm.weight"), head)?,
            k_norm: vector(&weights, &ctx, &format!("{base}attn_k_norm.weight"), head)?,
            q: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}attn_q.weight"),
                HIDDEN,
                HEADS * head,
            )?,
            k: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}attn_k.weight"),
                HIDDEN,
                n_kv * head,
            )?,
            v: if global {
                None
            } else {
                Some(packed_matrix(
                    &weights,
                    &ctx,
                    &format!("{base}attn_v.weight"),
                    HIDDEN,
                    n_kv * head,
                )?)
            },
            out: packed_matrix(
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
            gate: packed_matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_gate.weight"),
                HIDDEN,
                FFN,
            )?,
            up: packed_matrix(&weights, &ctx, &format!("{base}ffn_up.weight"), HIDDEN, FFN)?,
            down: packed_matrix(
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
    };
    let logits = vector_filled(&ctx, VOCAB, 0.0)?;
    let logits_capped = vector_filled(&ctx, VOCAB, 0.0)?;
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

fn matrix(
    weights: &GgufWeights,
    ctx: &CudaContext,
    name: &str,
    k: usize,
    rows: usize,
) -> Result<Matrix, EngError> {
    let (ty, bytes) = weights.read_tensor(name)?;
    if ty != GgufType::Q80 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} is {}, expected Q8_0",
            ty.name()
        )));
    }
    if bytes.len() != rows * (k / 32) * 34 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} holds {} bytes, expected {}",
            bytes.len(),
            rows * (k / 32) * 34
        )));
    }
    Ok(Matrix {
        buf: ctx.upload(&bytes)?,
        rows,
        k,
    })
}

/// Reads one Q8_0 tensor and splits each 34-byte block into the quant plane and the
/// half scale plane that gemma_matvec reads.
fn packed_matrix(
    weights: &GgufWeights,
    ctx: &CudaContext,
    name: &str,
    k: usize,
    rows: usize,
) -> Result<PackedMatrix, EngError> {
    let (ty, bytes) = weights.read_tensor(name)?;
    if ty != GgufType::Q80 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} is {}, expected Q8_0",
            ty.name()
        )));
    }
    let blocks = k / 32;
    if bytes.len() != rows * blocks * 34 {
        return Err(EngError::Checkpoint(format!(
            "gguf model: tensor {name} holds {} bytes, expected {}",
            bytes.len(),
            rows * blocks * 34
        )));
    }
    let mut qs = vec![0u8; rows * k];
    let mut scales = vec![0u8; rows * blocks * 2];
    // The split is a pure copy, so it runs over row chunks: 11.6 GB serially cost 23 s
    // of load on the Jetson.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(16);
    let per = rows.div_ceil(workers.max(1));
    let source = &bytes;
    std::thread::scope(|scope| {
        let chunks = qs
            .chunks_mut(per * k)
            .zip(scales.chunks_mut(per * blocks * 2))
            .enumerate();
        for (index, (qs_chunk, scales_chunk)) in chunks {
            let first = index * per;
            scope.spawn(move || {
                for (local, r) in (first..rows.min(first + per)).enumerate() {
                    for b in 0..blocks {
                        let at = (r * blocks + b) * 34;
                        let q_at = local * k + b * 32;
                        let s_at = local * blocks * 2 + b * 2;
                        scales_chunk[s_at] = source[at];
                        scales_chunk[s_at + 1] = source[at + 1];
                        qs_chunk[q_at..q_at + 32].copy_from_slice(&source[at + 2..at + 34]);
                    }
                }
            });
        }
    });
    Ok(PackedMatrix {
        qs: ctx.upload(&qs)?,
        scales: ctx.upload(&scales)?,
        rows,
        k,
    })
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
