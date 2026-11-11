//! Batched dense attention over a chunk.
//!
//! `chunk` is the chunk form of `attn::step`. The `T` rows of a chunk attend to the `P` cached
//! positions and to the chunk rows before them. The mask is causal and aligned to the last key, so
//! row `i` sees the positions `0..P + i`: `grouped_attention_prefix` keeps the mask mode string of
//! `grouped_attention`, and the new operation's test proved that the causal prefix view equals the
//! contiguous formulation bit for bit. The rules are those of `attn.rs`: the query projection
//! holds the query rows and then the gate rows per head, the query and key norms use the weights
//! as stored, RoPE rotates the first `rotary_dim` dimensions, and the head output is multiplied
//! by `sigmoid(gate)`. There is no indexer and no block selection.
//!
//! The cache keeps the decode layout — capacity buffers written in place by `kv_append` — so a
//! chunk and a decode step share it. The F32 control converts its new keys and values to F32 before
//! the append (as the decode step of `attn.rs`); the Q8 form keeps the activation dtype and
//! `quantize_rows` packs the rows. The attention runs `quantized_attention_prefix` over the
//! quantized buffers, causal for a chunk of more than one row, and the block output converts to the
//! activation dtype at the o-projection.

use super::super::attn::{AttnStep, KvBuffers, KvCache, append};
use super::super::kv_plan::{
    flash_dense_pad_enabled, flash_prefill_enabled, query_block_ranges, query_slice_rows,
};
use super::super::ops::{Ctx, cast_to, dim, to_act};
use super::super::weights::RAttn;
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    family::qwen36::settings::{KvDtype, Q8_BITS, Q8_GROUP},
    kernels::{BinaryOp, MlxKernels, QuantizedKv, RopeSpec, UnaryOp},
    tensor::{Dtype, ExecutionTensor},
};

fn overflow() -> EngError {
    EngError::BadRequest("resident attention dimension overflows".into())
}

fn mismatch() -> EngError {
    EngError::BadRequest("resident attention cache does not match its shape".into())
}

/// Runs one dense attention chunk of `x` `[T, H]` at position `pos`. It builds a lazy graph only:
/// no readback and no evaluate. `cache` is CONSUMED: `kv_append` writes the chunk rows into its
/// buffer in place at `[0, 0, len, 0]` (a growth replaces the buffer), so the caller hands the
/// layer's cache over and keeps only the returned one. The host validation — the input rank, the
/// position, the buffer shapes, and the capacity invariants — runs before the first native call
/// and before any buffer is consumed; once a buffer is consumed, any error leaves the caller's
/// state poisoned until `reset` (the rule of the chunk step, which takes the layer states before
/// it runs). A cache that holds a number of positions other than `pos`, tensors of the wrong
/// shape, or an input that is not `[T, H]` with `T > 0` returns `BadRequest` before the first
/// native call.
pub(super) fn chunk(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: KvCache,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<AttnStep, EngError> {
    chunk_sliced(ctx, shape, w, cache, pos, x, query_slice_rows()?)
}

/// `chunk` with the query-block rows of the composite attention given explicitly: `None` is one
/// attention call over the whole chunk (the default); `Some(n)` runs the calls on blocks of at
/// most `n` query rows, which `Q36_PREFILL_QUERY_SLICE` selects. The projections, norms, RoPE,
/// cache append, and output gate stay at the full chunk width either way. The flash kernel
/// ignores the setting: it never builds the score matrix the blocks bound.
pub(super) fn chunk_sliced(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: KvCache,
    pos: u32,
    x: &ExecutionTensor,
    slice: Option<u32>,
) -> Result<AttnStep, EngError> {
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    let &[rows, _] = x.shape() else {
        return Err(EngError::BadRequest(
            "attention chunk input must be [T, H]".into(),
        ));
    };
    if rows == 0 {
        return Err(EngError::BadRequest("attention chunk needs a row".into()));
    }
    if usize::try_from(pos).ok() != Some(cache.len) {
        return Err(EngError::BadRequest(format!(
            "attention cache holds {} positions, chunk is at {pos}",
            cache.len
        )));
    }
    cache.check_buffers(kv_heads, width, cache.kv)?;
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let q_width = heads.checked_mul(width).ok_or_else(overflow)?;
    let m = &mut *ctx.kernels;
    // 1. Query and gate, interleaved per head: head h is row[h * 2D..(h + 1) * 2D] of each row.
    let qp = m.qmm(x, &w.q)?;
    let qp = m.reshape(&qp, &[rows, heads, pair])?;
    let q_raw = m.slice(&qp, &[0, 0, 0], &[rows, heads, width])?;
    let gate = m.slice(&qp, &[0, 0, width], &[rows, heads, pair])?;
    // 2. Query norm with the weight as stored, time on axis -2; RoPE runs in `chunk_core`.
    let q = m.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
    let q = m.transpose(&q, &[1, 0, 2])?;
    let q = m.reshape(&q, &[1, heads, rows, width])?;
    // 3. New keys: norm per key-value head.
    let kp = m.qmm(x, &w.k)?;
    let kp = m.reshape(&kp, &[rows, kv_heads, width])?;
    let k_new = m.rms_norm(&kp, &w.k_norm, shape.eps)?;
    let k_new = m.transpose(&k_new, &[1, 0, 2])?;
    let k_new = m.reshape(&k_new, &[1, kv_heads, rows, width])?;
    // 4. New values.
    let vp = m.qmm(x, &w.v)?;
    let vp = m.reshape(&vp, &[rows, kv_heads, width])?;
    let v_new = m.transpose(&vp, &[1, 0, 2])?;
    let v_new = m.reshape(&v_new, &[1, kv_heads, rows, width])?;
    // 5-6. RoPE, the cache-form casts, the capacity append, and the causal attention core
    // over the valid prefix (the eager middle shared with the B1v2 compiled layer).
    let (att, cache) = chunk_core(ctx, shape, cache, pos, rows, &q, &k_new, &v_new, slice)?;
    // 7. Back to rows, sigmoid output gate, and output projection. The attention output is F32
    // over the F32 cache and follows the query dtype over the Q8 cache; the projection takes the
    // activation dtype.
    // The attn_gate fused kernel (Q36_FUSED_PREFILL=1) replaces the transpose+sigmoid+mul
    // chain in one kernel; the composite stays the default.
    let o = if crate::family::qwen36::resident::prefill::delta::fused_prefill_enabled()
        && std::env::var_os("Q36_NO_ATTNGATE").is_none()
        && width == 256
        && heads == 16
    {
        ctx.kernels.attn_gate(&att, &qp)?
    } else {
        let att = ctx.kernels.transpose(&att, &[0, 2, 1, 3])?;
        let att = ctx.kernels.reshape(&att, &[rows, heads, width])?;
        let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
        ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?
    };
    let o = ctx.kernels.reshape(&o, &[rows, q_width])?;
    let o = to_act(ctx, &o)?;
    let out = ctx.kernels.qmm(&o, &w.o)?;
    Ok(AttnStep { out, cache })
}

/// The eager middle of one attention chunk between the B1v2 compiled closures: RoPE on the
/// normed queries and keys, the cache-form casts, the capacity append, and the causal
/// attention core over the valid prefix. The projections, head norms, and gate slice (the
/// PRE closure) and the gate chain plus output projection (the POST closure) stay outside.
/// `q`, `k_new`, and `v_new` arrive normed and batched (`[1, heads, rows, width]` and
/// `[1, kv_heads, rows, width]`), unroped; returns the attention core output
/// `[1, heads, rows, width]` and the grown cache.
pub(super) fn chunk_core(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    cache: KvCache,
    pos: u32,
    rows: u32,
    q: &ExecutionTensor,
    k_new: &ExecutionTensor,
    v_new: &ExecutionTensor,
    slice: Option<u32>,
) -> Result<(ExecutionTensor, KvCache), EngError> {
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    if usize::try_from(pos).ok() != Some(cache.len) {
        return Err(EngError::BadRequest(format!(
            "attention cache holds {} positions, chunk is at {pos}",
            cache.len
        )));
    }
    cache.check_buffers(kv_heads, width, cache.kv)?;
    let next_len = usize::try_from(rows)
        .ok()
        .and_then(|rows| cache.len.checked_add(rows))
        .ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: dim(a.rotary_dim)?,
        base: a.theta,
        scale: 1.0,
        offset: pos,
    };
    let m = &mut *ctx.kernels;
    let q = m.rope_scaled(q, &rope)?;
    let k_new = m.rope_scaled(k_new, &rope)?;
    let v_new = v_new.share();
    // The cache form (as the decode step of `attn.rs`): the F32 control keeps its cast to F32
    // whatever the activation dtype (the cast runs before the append so that a cache which starts
    // with this chunk is F32 too), and the Q8 form keeps the activation dtype, so the quantized
    // scales and biases hold it.
    let (k_new, v_new) = match cache.kv {
        KvDtype::F32 => (
            cast_to(ctx, &k_new, Dtype::F32)?,
            cast_to(ctx, &v_new, Dtype::F32)?,
        ),
        KvDtype::Bf16 => (
            cast_to(ctx, &k_new, Dtype::Bf16)?,
            cast_to(ctx, &v_new, Dtype::Bf16)?,
        ),
        KvDtype::Q8 => (k_new, v_new),
    };
    // Append the rows into the capacity buffers (a growth builds the larger buffers first).
    let cache = append(ctx, kv_heads, width, cache, rows, &k_new, &v_new)?;
    // Causal attention over the valid prefix, as a copy-free view of the capacity buffers;
    // head h reads KV head h / (QH / KVH). The Q8 form runs the mlx-lm quantized attention over
    // the codes, scales, and biases - causal for a chunk of more than one row, and a one-row
    // chunk is the decode case.
    let scale = 1.0 / (width as f32).sqrt();
    let blocks = query_block_ranges(rows, slice);
    let att = match cache.buffers.as_ref() {
        Some(KvBuffers::F32 { k, v }) => {
            attend_blocks(ctx.kernels, &q, pos, &blocks, |kernels, q_block, len, _| {
                kernels.grouped_attention_prefix(q_block, k, v, len, scale, true)
            })?
        }
        // The BF16 form: the same causal prefix attention over the BF16 buffers.
        Some(KvBuffers::Bf16 { k, v }) => {
            ctx.kernels
                .grouped_attention_prefix(&q, k, v, dim(next_len)?, scale, true)?
        }
        Some(KvBuffers::Q8 { k, v }) => {
            let keys = QuantizedKv {
                codes: &k.codes,
                scales: &k.scales,
                biases: &k.biases,
            };
            let values = QuantizedKv {
                codes: &v.codes,
                scales: &v.scales,
                biases: &v.biases,
            };
            // The fused flash kernel replaces the composite under the flag, inside
            // its validated D envelope; the composite stays the default (and the D 64 form).
            // REP 8 runs the kernel directly; a REP 6 shape (the dense 24q/4kv) runs the
            // SAME kernel on a duplicate-padded query - each kv group of 6 real heads padded
            // to the kernel's 8 with its own first 2 heads, whose output rows are discarded:
            // a pad head reads real memory, its scores are finite, and its rows never reach
            // the projection (measured 1.21x over the composite at rows 512 depth 16K, with
            // flash error below the composite's against the F32 reference).
            let rep = heads / kv_heads;
            if flash_prefill_enabled()
                && width >= 128
                && (rep == 8 || (rep == 6 && flash_dense_pad_enabled()))
                && crate::kernels::mlx::q8_flash_envelope(width, Q8_GROUP)
            {
                let (flash_q, padded_heads) = if rep == 8 {
                    (q.share(), heads)
                } else {
                    (
                        pad_rep6_query(ctx.kernels, &q, heads, kv_heads)?,
                        kv_heads * 8,
                    )
                };
                let flash_out = ctx.kernels.q8_flash_prefill(
                    &flash_q,
                    &keys,
                    &values,
                    dim(next_len)?,
                    scale,
                    Q8_GROUP,
                    Q8_BITS,
                    None,
                    0,
                )?;
                if padded_heads == heads {
                    flash_out
                } else {
                    unpad_rep6_output(ctx.kernels, &flash_out, heads, kv_heads)?
                }
            } else {
                attend_blocks(ctx.kernels, &q, pos, &blocks, |kernels, q_block, len, n| {
                    kernels.quantized_attention_prefix(
                        q_block,
                        &keys,
                        &values,
                        len,
                        scale,
                        n > 1,
                        Q8_GROUP,
                        Q8_BITS,
                    )
                })?
            }
        }
        None => return Err(mismatch()),
    };
    Ok((att, cache))
}

/// Runs `attend(kernels, queries, len, block_rows)` once per query block of `blocks` and joins
/// Pads a REP 6 query (`[1, heads, rows, width]`, heads = 6 * kv_heads, the dense 24q/4kv)
/// to the flash kernel's REP 8 head contract: each kv group of 6 real heads is padded to 8
/// with its own first 2 heads, so a pad row reads real memory, produces finite scores, and
/// its output row is simply discarded by the caller's slice back to `heads` heads. Five lazy
/// nodes (four 6-head slices, four 2-head slices, one concatenate), measured noise against
/// the attention FLOPs at serving depths.
fn pad_rep6_query(
    kernels: &mut MlxKernels,
    q: &ExecutionTensor,
    heads: u32,
    kv_heads: u32,
) -> Result<ExecutionTensor, EngError> {
    let &[1, _, rows, width] = q.shape() else {
        return Err(mismatch());
    };
    if heads != kv_heads * 6 {
        return Err(mismatch());
    }
    let mut parts: Vec<ExecutionTensor> = Vec::with_capacity(8);
    for group in 0..kv_heads {
        let base = group * 6;
        parts.push(kernels.slice(q, &[0, base, 0, 0], &[1, base + 6, rows, width])?);
        parts.push(kernels.slice(q, &[0, base, 0, 0], &[1, base + 2, rows, width])?);
    }
    let views: Vec<&ExecutionTensor> = parts.iter().collect();
    kernels.concatenate(&views, 1)
}

/// The inverse of `pad_rep6_query`: keeps the FIRST 6 heads of each 8-head kv group of the
/// padded flash output and concatenates them back to `heads` in order. A head-range slice
/// of the first `heads` positions would interleave the discarded pad rows into the result
/// (the padded layout is `[real 6, pad 2]` per group), so the real heads are re-assembled
/// group by group.
fn unpad_rep6_output(
    kernels: &mut MlxKernels,
    padded: &ExecutionTensor,
    heads: u32,
    kv_heads: u32,
) -> Result<ExecutionTensor, EngError> {
    let &[1, _, rows, width] = padded.shape() else {
        return Err(mismatch());
    };
    if heads != kv_heads * 6 {
        return Err(mismatch());
    }
    let mut parts: Vec<ExecutionTensor> = Vec::with_capacity(kv_heads as usize);
    for group in 0..kv_heads {
        let base = group * 8;
        parts.push(kernels.slice(padded, &[0, base, 0, 0], &[1, base + 6, rows, width])?);
    }
    let views: Vec<&ExecutionTensor> = parts.iter().collect();
    kernels.concatenate(&views, 1)
}

/// Causal attention over the valid prefix in the 256-row blocks of `Q36_PREFILL_QUERY_SLICE`,
/// concatenating the outputs on the query axis. A block `[start, stop)` of a chunk at position `pos` attends
/// the first `len = pos + stop` positions with `block_rows = stop - start` queries, and the
/// causal mask aligned to the last key makes its first query the row `pos + start`: the rows
/// the whole-chunk call gave. A single block covering the chunk is that whole-chunk call on
/// `q` itself, so the default builds the graph it always built.
fn attend_blocks<F>(
    kernels: &mut MlxKernels,
    q: &ExecutionTensor,
    pos: u32,
    blocks: &[(u32, u32)],
    mut attend: F,
) -> Result<ExecutionTensor, EngError>
where
    F: FnMut(&mut MlxKernels, &ExecutionTensor, u32, u32) -> Result<ExecutionTensor, EngError>,
{
    let &[batch, heads, _, width] = q.shape() else {
        return Err(mismatch());
    };
    if let &[(0, stop)] = blocks {
        return attend(
            kernels,
            q,
            pos.checked_add(stop).ok_or_else(overflow)?,
            stop,
        );
    }
    let mut outputs = Vec::with_capacity(blocks.len());
    for &(start, stop) in blocks {
        let q_block = kernels.slice(q, &[0, 0, start, 0], &[batch, heads, stop, width])?;
        let len = pos.checked_add(stop).ok_or_else(overflow)?;
        outputs.push(attend(kernels, &q_block, len, stop - start)?);
    }
    kernels.concatenate(&outputs.iter().collect::<Vec<_>>(), 2)
}

#[cfg(test)]
mod tests {
    use super::super::super::attn::{ConcatCache, concat_step, growth_bytes, step};
    use super::super::super::ops::{assert_parity, f32_tensor, values};
    use super::super::super::weights::{RMixer, ResidentWeights};
    use super::*;
    use crate::family::flash_next::math::rope_half;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, KvCache as HostCache, MixerW, rms_plain};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::settings::Activation;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;
    use crate::weights::StreamedCheckpoint;

    /// Input of position `n`: `x[i] = (((i + 5 n) % 11) - 5) / 16`, exact in `f32`.
    fn input(n: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((i + 5 * n) % 11) as f32 - 5.0) / 16.0)
            .collect()
    }

    /// Layer 1 of the four-layer tiny model: 4 query heads over 2 key-value heads, head dimension 32,
    /// rotary dimension 16.
    fn setup() -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let config = tiny::config(&TINY_GQA);
        let oracle = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, w, kernels)
    }

    fn upload(kernels: &mut MlxKernels, shape: Vec<u32>, data: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(shape, data).unwrap())
            .unwrap()
    }

    /// The F32 pair of a cache of the F32 form (test observer).
    fn f32_pair(cache: &KvCache) -> (&ExecutionTensor, &ExecutionTensor) {
        match &cache.buffers {
            Some(KvBuffers::F32 { k, v }) => (k, v),
            _ => panic!("the cache is not the F32 form"),
        }
    }

    /// The input rows of positions `first..first + count` as `[count, 64]`.
    fn rows(kernels: &mut MlxKernels, first: u32, count: u32) -> ExecutionTensor {
        let data: Vec<f32> = (first..first + count).flat_map(input).collect();
        upload(kernels, vec![count, 64], &data)
    }

    /// The first `len` positions of each key-value head of a `[1, KVH, cap, D]` buffer readback,
    /// in the `[1, KVH, len, D]` layout.
    fn prefix_rows(full: &[f32], heads: usize, cap: usize, len: usize, width: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for g in 0..heads {
            let head = &full[g * cap * width..(g + 1) * cap * width];
            out.extend_from_slice(&head[..len * width]);
        }
        out
    }

    /// The host attention of layer 1 over some positions: the output of every position and, by
    /// definition, the key and the value of every position (the weighted RMS norm of the key
    /// projection and then RoPE; the value projection as is).
    struct HostRun {
        outputs: Vec<Vec<f32>>,
        keys: Vec<Vec<f32>>,
        vals: Vec<Vec<f32>>,
    }

    fn host_run(ckpt: &StreamedCheckpoint, oracle: &HostWeights, count: u32) -> HostRun {
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Attn(host_w) = &view.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let mut cpu = CpuProjector::new(ckpt, 1);
        let mut host_cache = HostCache::default();
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let (mut outputs, mut keys, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for n in 0..count {
            let x = input(n);
            // The oracle models the mixer over pre-normed rows; the chunk seam folds the input
            // norm at the caller and keeps the output residual out, so bridge the norm here.
            let normed_x = rms_plain(&x, view.layers[1].input_norm, shape.eps).unwrap();
            outputs.push(
                host::attn_step(&mut cpu, &shape, host_w, &mut host_cache, n, &normed_x).unwrap(),
            );
            let kp = cpu.linear(host_w.k, &normed_x).unwrap();
            let mut key = Vec::new();
            for g in 0..hkv {
                let mut head =
                    rms_plain(&kp[g * hd..(g + 1) * hd], host_w.k_norm, shape.eps).unwrap();
                rope_half(&mut head, shape.attn.rotary_dim, n, shape.attn.theta).unwrap();
                key.extend(head);
            }
            keys.push(key);
            vals.push(cpu.linear(host_w.v, &normed_x).unwrap());
        }
        HostRun {
            outputs,
            keys,
            vals,
        }
    }

    /// Layout `[1, KVH, T, D]`: element (g, t, d) sits at (g * T + t) * D + d.
    fn cache_layout(per_position: &[Vec<f32>], heads: usize, width: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for g in 0..heads {
            for position in per_position {
                out.extend_from_slice(&position[g * width..(g + 1) * width]);
            }
        }
        out
    }

    /// The chunk step exactly as it was before the capacity buffers (test-only): the cache is
    /// `[1, KVH, len, D]` (see `attn::ConcatCache`) and every chunk rebuilds it with
    /// `concatenate`. The bit equality of this formulation and the capacity buffers proves that
    /// the new cache changes no numerics.
    fn concat_chunk(
        ctx: &mut Ctx<'_>,
        shape: &HostShape,
        w: &RAttn,
        cache: &ConcatCache,
        pos: u32,
        x: &ExecutionTensor,
    ) -> Result<(ExecutionTensor, ConcatCache), EngError> {
        let a = &shape.attn;
        let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
        let &[rows, _] = x.shape() else {
            return Err(EngError::BadRequest(
                "attention chunk input must be [T, H]".into(),
            ));
        };
        let cached = dim(cache.len)?;
        let past = match (&cache.k, &cache.v) {
            (None, None) if cache.len == 0 => None,
            (Some(k), Some(v)) if cache.len > 0 => {
                let kv_shape = [1, kv_heads, cached, width];
                if k.shape() != kv_shape.as_slice() || v.shape() != kv_shape.as_slice() {
                    return Err(mismatch());
                }
                Some((k, v))
            }
            _ => return Err(mismatch()),
        };
        let pair = width.checked_mul(2).ok_or_else(overflow)?;
        let q_width = heads.checked_mul(width).ok_or_else(overflow)?;
        let next_len = usize::try_from(rows)
            .ok()
            .and_then(|rows| cache.len.checked_add(rows))
            .ok_or_else(overflow)?;
        let rope = RopeSpec {
            dims: dim(a.rotary_dim)?,
            base: a.theta,
            scale: 1.0,
            offset: pos,
        };
        let m = &mut *ctx.kernels;
        let qp = m.qmm(x, &w.q)?;
        let qp = m.reshape(&qp, &[rows, heads, pair])?;
        let q_raw = m.slice(&qp, &[0, 0, 0], &[rows, heads, width])?;
        let gate = m.slice(&qp, &[0, 0, width], &[rows, heads, pair])?;
        let q = m.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
        let q = m.transpose(&q, &[1, 0, 2])?;
        let q = m.reshape(&q, &[1, heads, rows, width])?;
        let q = m.rope_scaled(&q, &rope)?;
        let kp = m.qmm(x, &w.k)?;
        let kp = m.reshape(&kp, &[rows, kv_heads, width])?;
        let k_new = m.rms_norm(&kp, &w.k_norm, shape.eps)?;
        let k_new = m.transpose(&k_new, &[1, 0, 2])?;
        let k_new = m.reshape(&k_new, &[1, kv_heads, rows, width])?;
        let k_new = m.rope_scaled(&k_new, &rope)?;
        let vp = m.qmm(x, &w.v)?;
        let vp = m.reshape(&vp, &[rows, kv_heads, width])?;
        let v_new = m.transpose(&vp, &[1, 0, 2])?;
        let v_new = m.reshape(&v_new, &[1, kv_heads, rows, width])?;
        let (k_new, v_new) = (
            cast_to(ctx, &k_new, Dtype::F32)?,
            cast_to(ctx, &v_new, Dtype::F32)?,
        );
        let m = &mut *ctx.kernels;
        let (k_all, v_all) = match past {
            None => (k_new, v_new),
            Some((k, v)) => (
                m.concatenate(&[k, &k_new], 2)?,
                m.concatenate(&[v, &v_new], 2)?,
            ),
        };
        let scale = 1.0 / (width as f32).sqrt();
        let att = m.grouped_attention(&q, &k_all, &v_all, scale, true)?;
        let att = m.transpose(&att, &[0, 2, 1, 3])?;
        let att = m.reshape(&att, &[rows, heads, width])?;
        let gate = m.unary(UnaryOp::Sigmoid, &gate)?;
        let o = m.binary(BinaryOp::Mul, &att, &gate)?;
        let o = m.reshape(&o, &[rows, q_width])?;
        let o = to_act(ctx, &o)?;
        let out = ctx.kernels.qmm(&o, &w.o)?;
        Ok((
            out,
            ConcatCache {
                k: Some(k_all),
                v: Some(v_all),
                len: next_len,
            },
        ))
    }

    /// Runs positions `warm..warm + count` as one chunk after `warm` decode steps and compares the
    /// outputs and the whole valid prefix with the host run over `warm + count` positions. The
    /// caches grow in quanta of 4, and `want_cap` is the exact hand-derived capacity after the
    /// chunk: round_up(warm + count, 4).
    fn chunk_matches_the_host(warm: u32, count: u32, want_cap: usize) {
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let host = host_run(&ckpt, &oracle, warm + count);
        // The cached positions come from the decode step, which the tests of `attn.rs` compared
        // with the host.
        let mut cache = KvCache::with_step(4);
        for n in 0..warm {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            cache = step(&mut ctx, &shape, rw, cache, n, &xr, &w.layers[1].input_norm)
                .unwrap()
                .cache;
        }
        let x = rows(&mut kernels, warm, count);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // The oracle models the mixer over pre-normed rows; the chunk seam keeps the raw input
        // and lets the caller norm, so bridge the folded contract here.
        let x = ctx
            .kernels
            .rms_norm(&x, &w.layers[1].input_norm, shape.eps)
            .unwrap();
        let next = chunk(&mut ctx, &shape, rw, cache, warm, &x).unwrap();
        let total = warm as usize + count as usize;
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        assert_eq!(next.out.shape(), [count, 64], "output shape");
        assert_eq!(next.cache.len, total, "cache length");
        assert_eq!(
            next.cache.cap, want_cap,
            "the exact capacity after the chunk"
        );
        assert!(
            want_cap >= total,
            "the capacity {want_cap} must hold the valid rows {total}"
        );
        assert_eq!(
            f32_pair(&next.cache).0.shape(),
            [1, hkv as u32, next.cache.cap as u32, hd as u32],
            "key shape"
        );
        assert_eq!(
            next.cache.byte_len(),
            next.cache.cap as u64 * growth_bytes(&shape, KvDtype::F32, Activation::F32).unwrap(),
            "stored bytes follow the capacity"
        );
        let want_out: Vec<f32> = host.outputs[warm as usize..].concat();
        assert_parity(
            &values(&mut kernels, &next.out).unwrap(),
            &want_out,
            "chunk output",
        );
        let (k_buf, v_buf) = f32_pair(&next.cache);
        assert_parity(
            &prefix_rows(
                &values(&mut kernels, k_buf).unwrap(),
                hkv,
                next.cache.cap,
                total,
                hd,
            ),
            &cache_layout(&host.keys, hkv, hd),
            "cached keys",
        );
        assert_parity(
            &prefix_rows(
                &values(&mut kernels, v_buf).unwrap(),
                hkv,
                next.cache.cap,
                total,
                hd,
            ),
            &cache_layout(&host.vals, hkv, hd),
            "cached values",
        );
    }

    #[test]
    fn a_chunk_from_an_empty_cache_matches_the_host_steps() {
        let _serial = crate::native_test_lock();
        // Six rows of an empty cache grow in quanta of 4: round_up(6, 4) = 8.
        chunk_matches_the_host(0, 6, 8);
    }

    #[test]
    fn a_chunk_after_cached_positions_matches_the_host_steps() {
        // Five cached positions: RoPE starts at position 5 and row i sees 5 + i + 1 positions.
        // Five decode steps fill 4 and grow to 8; the chunk of 4 needs 9, so round_up(9, 4) = 12.
        let _serial = crate::native_test_lock();
        chunk_matches_the_host(5, 4, 12);
    }

    /// A chunk of 5 rows that crosses a growth (the buffers hold 2
    /// of 4 positions, the chunk needs 7, so they grow to 8), then decode steps that fill 8
    /// exactly and grow to 12. Every output is bit-equal to the old concatenate formulation and
    /// within the parity bounds of the host run.
    #[test]
    fn a_chunk_of_five_rows_crossing_a_growth_then_decode_steps_match_the_old_formulation() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let host = host_run(&ckpt, &oracle, 12);
        let mut cache = KvCache::with_step(4);
        let mut concat = ConcatCache {
            k: None,
            v: None,
            len: 0,
        };
        // Two decode steps hold 2 of the 4 first positions.
        for n in 0..2u32 {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr, &w.layers[1].input_norm).unwrap();
            let old = concat_step(
                &mut ctx,
                &shape,
                rw,
                &concat,
                n,
                &xr,
                &w.layers[1].input_norm,
            )
            .unwrap();
            assert_eq!(
                values(&mut kernels, &next.out).unwrap(),
                values(&mut kernels, &old.0).unwrap(),
                "warm step {n} differs from the concatenate formulation"
            );
            cache = next.cache;
            concat = old.1;
        }
        assert_eq!(cache.cap, 4, "the warm steps hold capacity 4");
        // The chunk of 5 rows: 2 + 5 = 7 passes 4, so the buffers grow to 8.
        let x = rows(&mut kernels, 2, 5);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // The oracle rows are pre-normed (the folded contract); the chunk seam keeps raw input.
        let x = ctx
            .kernels
            .rms_norm(&x, &w.layers[1].input_norm, shape.eps)
            .unwrap();
        let next = chunk(&mut ctx, &shape, rw, cache, 2, &x).unwrap();
        let old = concat_chunk(&mut ctx, &shape, rw, &concat, 2, &x).unwrap();
        let got = values(&mut kernels, &next.out).unwrap();
        assert_eq!(
            got,
            values(&mut kernels, &old.0).unwrap(),
            "the chunk output differs from the concatenate formulation"
        );
        assert_parity(
            &got,
            &host.outputs[2..7].concat(),
            "chunk output against the host",
        );
        assert_eq!(next.cache.len, 7, "cache length after the chunk");
        assert_eq!(next.cache.cap, 8, "the chunk crossed a growth");
        cache = next.cache;
        concat = old.1;
        // Decode steps fill 8 exactly and then grow to 12.
        for n in 7..12u32 {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr, &w.layers[1].input_norm).unwrap();
            let old = concat_step(
                &mut ctx,
                &shape,
                rw,
                &concat,
                n,
                &xr,
                &w.layers[1].input_norm,
            )
            .unwrap();
            let got = values(&mut kernels, &next.out).unwrap();
            assert_eq!(
                got,
                values(&mut kernels, &old.0).unwrap(),
                "decode step {n} after the chunk differs from the concatenate formulation"
            );
            // The decode seam folds the output residual, so bridge it onto the host row.
            let want: Vec<f32> = host.outputs[n as usize]
                .iter()
                .zip(&input(n))
                .map(|(a, b)| a + b)
                .collect();
            assert_parity(&got, &want, &format!("decode step {n} against the host"));
            cache = next.cache;
            concat = old.1;
        }
        assert_eq!(cache.len, 12, "final length");
        assert_eq!(cache.cap, 12, "the decode steps crossed the second growth");
        // The caches agree bit for bit on the valid prefix.
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let (k_buf, v_buf) = f32_pair(&cache);
        for (name, new_tensor, old_tensor) in [
            ("keys", k_buf, concat.k.as_ref().unwrap()),
            ("values", v_buf, concat.v.as_ref().unwrap()),
        ] {
            let got = values(&mut kernels, new_tensor).unwrap();
            let want = values(&mut kernels, old_tensor).unwrap();
            assert_eq!(
                prefix_rows(&got, hkv, cache.cap, 12, hd),
                want,
                "cached {name} differ from the concatenate formulation"
            );
        }
    }

    /// Review B finding 2 (the chunk half of it): a chunk of MORE than one row appended into
    /// existing slack with NO growth — two decode steps hold 2 of 4 positions and a chunk of 2
    /// fills the capacity exactly, in place — then a second chunk of 3 rows that grows the
    /// buffers to 8, then decode steps that fill 8 exactly and grow to 12. Every output is
    /// bit-equal to the old concatenate formulation, and the capacity assertions pin the growth
    /// schedule (no growth inside the slack, growth at the boundary). Whether the append runs in
    /// place is not visible here: a branch that always grows keeps every capacity and every
    /// result; only the donation test of `attn.rs` on the Mac can see it.
    #[test]
    fn two_chunks_in_slack_then_a_growth_then_decode_match_the_old_formulation() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let host = host_run(&ckpt, &oracle, 9);
        let mut cache = KvCache::with_step(4);
        let mut concat = ConcatCache {
            k: None,
            v: None,
            len: 0,
        };
        // Two decode steps hold 2 of the first 4 positions.
        for n in 0..2u32 {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr, &w.layers[1].input_norm).unwrap();
            let old = concat_step(
                &mut ctx,
                &shape,
                rw,
                &concat,
                n,
                &xr,
                &w.layers[1].input_norm,
            )
            .unwrap();
            assert_eq!(
                values(&mut kernels, &next.out).unwrap(),
                values(&mut kernels, &old.0).unwrap(),
                "warm step {n} differs from the concatenate formulation"
            );
            cache = next.cache;
            concat = old.1;
        }
        assert_eq!(cache.cap, 4, "the warm steps hold capacity 4");
        // The first chunk of 2 rows: 2 + 2 = 4 fits, so the append runs in place with no growth.
        let x = rows(&mut kernels, 2, 2);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // The oracle rows are pre-normed (the folded contract); the chunk seam keeps raw input.
        let x = ctx
            .kernels
            .rms_norm(&x, &w.layers[1].input_norm, shape.eps)
            .unwrap();
        let next = chunk(&mut ctx, &shape, rw, cache, 2, &x).unwrap();
        let old = concat_chunk(&mut ctx, &shape, rw, &concat, 2, &x).unwrap();
        let got = values(&mut kernels, &next.out).unwrap();
        assert_eq!(
            got,
            values(&mut kernels, &old.0).unwrap(),
            "the in-place chunk output differs from the concatenate formulation"
        );
        assert_parity(
            &got,
            &host.outputs[2..4].concat(),
            "the in-place chunk output against the host",
        );
        assert_eq!(next.cache.len, 4, "length after the in-place chunk");
        assert_eq!(next.cache.cap, 4, "the in-place chunk grows nothing");
        cache = next.cache;
        concat = old.1;
        // The second chunk of 3 rows: 4 + 3 = 7 passes 4, so the buffers grow to 8.
        let x = rows(&mut kernels, 4, 3);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // The oracle rows are pre-normed (the folded contract); the chunk seam keeps raw input.
        let x = ctx
            .kernels
            .rms_norm(&x, &w.layers[1].input_norm, shape.eps)
            .unwrap();
        let next = chunk(&mut ctx, &shape, rw, cache, 4, &x).unwrap();
        let old = concat_chunk(&mut ctx, &shape, rw, &concat, 4, &x).unwrap();
        let got = values(&mut kernels, &next.out).unwrap();
        assert_eq!(
            got,
            values(&mut kernels, &old.0).unwrap(),
            "the second chunk output differs from the concatenate formulation"
        );
        assert_parity(
            &got,
            &host.outputs[4..7].concat(),
            "the second chunk output against the host",
        );
        assert_eq!(next.cache.len, 7, "length after the second chunk");
        assert_eq!(next.cache.cap, 8, "the second chunk crossed a growth");
        cache = next.cache;
        concat = old.1;
        // Decode steps fill 8 exactly and then grow to 12.
        for n in 7..9u32 {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr, &w.layers[1].input_norm).unwrap();
            let old = concat_step(
                &mut ctx,
                &shape,
                rw,
                &concat,
                n,
                &xr,
                &w.layers[1].input_norm,
            )
            .unwrap();
            let got = values(&mut kernels, &next.out).unwrap();
            assert_eq!(
                got,
                values(&mut kernels, &old.0).unwrap(),
                "decode step {n} after the chunks differs from the concatenate formulation"
            );
            // The decode seam folds the output residual, so bridge it onto the host row.
            let want: Vec<f32> = host.outputs[n as usize]
                .iter()
                .zip(&input(n))
                .map(|(a, b)| a + b)
                .collect();
            assert_parity(&got, &want, &format!("decode step {n} against the host"));
            cache = next.cache;
            concat = old.1;
        }
        assert_eq!(cache.len, 9, "final length");
        assert_eq!(cache.cap, 12, "the last decode step crossed the growth");
    }

    #[test]
    fn a_cache_of_the_wrong_length_or_shape_or_an_input_of_the_wrong_rank_is_rejected() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let x = rows(&mut kernels, 0, 2);
        let flat = upload(&mut kernels, vec![128], &[0.0; 128]);
        let one_head = |kernels: &mut MlxKernels| upload(kernels, vec![1, 1, 1, 32], &[0.0; 32]);
        let bad = KvCache {
            buffers: Some(KvBuffers::F32 {
                k: one_head(&mut kernels),
                v: one_head(&mut kernels),
            }),
            kv: KvDtype::F32,
            len: 1,
            cap: 1,
            step: 4,
        };
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let cases = [
            (
                "cache holds 0 positions",
                chunk(&mut ctx, &shape, rw, KvCache::empty(), 3, &x),
            ),
            (
                "does not match its shape",
                chunk(&mut ctx, &shape, rw, bad, 1, &x),
            ),
            (
                "chunk input must be [T, H]",
                chunk(&mut ctx, &shape, rw, KvCache::empty(), 0, &flat),
            ),
        ];
        for (needle, result) in cases {
            assert!(
                matches!(&result, Err(EngError::BadRequest(m)) if m.contains(needle)),
                "{needle}: {:?}",
                result.as_ref().err()
            );
        }
    }

    /// Layer 1 of the four-layer tiny KV8 model: head dimension 64, the
    /// multiple of the Q8 group, so the Q8 form is available.
    fn setup_kv8() -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(&tiny::TINY_KV8);
        let config = tiny::config(&tiny::TINY_KV8);
        let oracle = HostWeights::load(&ckpt, &tiny::TINY_KV8, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, &tiny::TINY_KV8, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, w, kernels)
    }

    /// A Q8 cache (growth quantum 4) that holds the decode-step positions `0..warm`, with
    /// `perturbed` (if any) given a different input.
    fn q8_warm_cache(
        kernels: &mut MlxKernels,
        w: &ResidentWeights,
        shape: &HostShape,
        rw: &RAttn,
        warm: u32,
        perturbed: Option<u32>,
    ) -> KvCache {
        let mut cache = KvCache::with_form(KvDtype::Q8, 4);
        for n in 0..warm {
            let mut data = input(n);
            if perturbed == Some(n) {
                data = data.iter().map(|v| 0.0625 - v).collect();
            }
            let xr = upload(kernels, vec![1, 64], &data);
            let mut ctx = Ctx {
                kernels: &mut *kernels,
                consts: &w.consts,
                compile: false,
            };
            cache = step(&mut ctx, shape, rw, cache, n, &xr, &w.layers[1].input_norm)
                .unwrap()
                .cache;
        }
        cache
    }

    /// The `[count, 64]` output of a Q8 chunk of the rows `data` after `warm` Q8 decode steps.
    fn q8_chunk_output(
        kernels: &mut MlxKernels,
        w: &ResidentWeights,
        shape: &HostShape,
        rw: &RAttn,
        warm: u32,
        perturbed: Option<u32>,
        data: &[f32],
    ) -> Vec<f32> {
        let cache = q8_warm_cache(kernels, w, shape, rw, warm, perturbed);
        chunk_output(kernels, w, shape, rw, cache, warm, data, None)
    }

    /// The `[count, 64]` output of the chunk of the rows `data` on `cache` (holding `warm`
    /// positions), with the attention queries in blocks of `slice` rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "a test helper over the chunk's own inputs"
    )]
    fn chunk_output(
        kernels: &mut MlxKernels,
        w: &ResidentWeights,
        shape: &HostShape,
        rw: &RAttn,
        cache: KvCache,
        warm: u32,
        data: &[f32],
        slice: Option<u32>,
    ) -> Vec<f32> {
        let count = u32::try_from(data.len() / 64).unwrap();
        let x = upload(kernels, vec![count, 64], data);
        let mut ctx = Ctx {
            kernels: &mut *kernels,
            consts: &w.consts,
            compile: false,
        };
        // The decode steps norm inside (the folded contract); the chunk keeps raw input.
        let x = ctx
            .kernels
            .rms_norm(&x, &w.layers[1].input_norm, shape.eps)
            .unwrap();
        let next = chunk_sliced(&mut ctx, shape, rw, cache, warm, &x, slice).unwrap();
        assert_eq!(next.cache.len, warm as usize + count as usize);
        values(kernels, &next.out).unwrap()
    }

    /// An F32 cache that holds the decode-step positions `0..warm`.
    fn f32_warm_cache(
        kernels: &mut MlxKernels,
        w: &ResidentWeights,
        shape: &HostShape,
        rw: &RAttn,
        warm: u32,
    ) -> KvCache {
        let mut cache = KvCache::with_step(4);
        for n in 0..warm {
            let xr = upload(kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut *kernels,
                consts: &w.consts,
                compile: false,
            };
            cache = step(&mut ctx, shape, rw, cache, n, &xr, &w.layers[1].input_norm)
                .unwrap()
                .cache;
        }
        cache
    }

    /// The largest absolute difference of two equal-length outputs.
    fn max_gap(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// Seven rows after three cached positions: the query blocks to compare with the whole-chunk
    /// call. 1, 2 and 3 rows leave a short last block; 7 is the chunk itself and 100 is wider.
    const SLICES: [u32; 5] = [1, 2, 3, 7, 100];

    /// The float noise allowed between a whole-chunk attention call and the same rows in blocks:
    /// a block changes the row count of the score matmul, hence its reduction order (measured
    /// 2e-7 on outputs of about 0.3), the way `a_q8_chunk_agrees_with_five_q8_decode_steps...`
    /// allows 1e-5. A block with the wrong length or causal alignment lets a row read later
    /// keys and moves the output by orders of magnitude more.
    const SLICE_NOISE: f32 = 1e-5;

    /// Query blocks change the shape of each attention call, not its causal alignment: the block
    /// `[start, stop)` attends `pos + stop` positions and its first query is the row
    /// `pos + start`. Blocks of 1, 2 and 3 rows give the output of the whole-chunk call to
    /// float noise, and a block that covers the chunk is that call, bit for bit.
    fn sliced_blocks_match_the_whole_chunk(q8: bool) {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = if q8 { setup_kv8() } else { setup() };
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let (warm, count) = (3u32, 7u32);
        let data: Vec<f32> = (warm..warm + count).flat_map(input).collect();
        let mut run = |slice: Option<u32>| {
            let cache = if q8 {
                q8_warm_cache(&mut kernels, &w, &shape, rw, warm, None)
            } else {
                f32_warm_cache(&mut kernels, &w, &shape, rw, warm)
            };
            chunk_output(&mut kernels, &w, &shape, rw, cache, warm, &data, slice)
        };
        let whole = run(None);
        assert!(whole.iter().all(|value| value.is_finite()));
        let mut worst = 0f32;
        for slice in SLICES {
            let sliced = run(Some(slice));
            let gap = max_gap(&sliced, &whole);
            println!("q8 {q8}, slice {slice}: gap {gap}");
            if slice >= count {
                assert_eq!(
                    sliced, whole,
                    "a block covering the chunk is the whole call"
                );
            }
            worst = worst.max(gap);
        }
        assert!(worst <= SLICE_NOISE, "blocks move the output by {worst}");
    }

    #[test]
    fn sliced_q8_query_blocks_match_the_whole_chunk_call() {
        sliced_blocks_match_the_whole_chunk(true);
    }

    #[test]
    fn sliced_f32_query_blocks_match_the_whole_chunk_call() {
        sliced_blocks_match_the_whole_chunk(false);
    }

    /// The causality test of `a_q8_chunk_row_never_reads_a_later_row` with the queries in blocks
    /// of two rows: rows 0 to 2 are bit-identical when rows 3 and 4 change, and rows 3 and 4
    /// differ, so a block that attended past its own last row would fail the first check.
    #[test]
    fn a_sliced_q8_chunk_row_never_reads_a_later_row() {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let (warm, count) = (3u32, 5usize);
        let base: Vec<f32> = (warm..warm + count as u32).flat_map(input).collect();
        let mut changed = base.clone();
        for value in &mut changed[3 * 64..] {
            *value = 0.0625 - *value;
        }
        let run = |kernels: &mut MlxKernels, data: &[f32]| {
            let cache = q8_warm_cache(kernels, &w, &shape, rw, warm, None);
            chunk_output(kernels, &w, &shape, rw, cache, warm, data, Some(2))
        };
        let a = run(&mut kernels, &base);
        let b = run(&mut kernels, &changed);
        assert_eq!(
            a[..3 * 64],
            b[..3 * 64],
            "rows 0 to 2 read nothing of rows 3 and 4"
        );
        assert_ne!(
            a[3 * 64..],
            b[3 * 64..],
            "rows 3 and 4 read their own input"
        );
    }

    /// Causality: in the Q8 chunk a row reads no later row. Two
    /// chunks of five rows after three cached positions have the same rows 0 to 2 and different
    /// rows 3 and 4. The masked keys get the probability 0 exactly and every other step of a
    /// row (projection, norm, RoPE, quantization of its own K and V, softmax, output
    /// projection) depends on that row alone, so the outputs of rows 0 to 2 are bit-identical.
    /// A chunk that attends without the causal mask lets rows 0 to 2 read rows 3 and 4 and
    /// differ. The perturbation reaches its own rows (rows 3 and 4 differ), so the equality is
    /// not an artifact of equal outputs.
    #[test]
    fn a_q8_chunk_row_never_reads_a_later_row() {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let (warm, count) = (3u32, 5usize);
        let base: Vec<f32> = (warm..warm + count as u32).flat_map(input).collect();
        let mut changed = base.clone();
        for value in &mut changed[3 * 64..] {
            *value = 0.0625 - *value;
        }
        let a = q8_chunk_output(&mut kernels, &w, &shape, rw, warm, None, &base);
        let b = q8_chunk_output(&mut kernels, &w, &shape, rw, warm, None, &changed);
        assert_eq!(a.len(), count * 64);
        assert!(a.iter().chain(&b).all(|value| value.is_finite()));
        assert_eq!(
            a[..3 * 64],
            b[..3 * 64],
            "rows 0 to 2 read nothing of rows 3 and 4"
        );
        assert_ne!(
            a[3 * 64..],
            b[3 * 64..],
            "rows 3 and 4 read their own input"
        );
    }

    /// Alignment: a Q8 chunk row sees EVERY cached position and
    /// the chunk rows up to itself (the mask is aligned to the last key). Changing the input of
    /// cached position 1 changes the output of every chunk row, row 0 included: with a mask
    /// aligned to the first key, row 0 would see position 0 alone and ignore position 1.
    #[test]
    fn a_q8_chunk_row_reads_every_cached_position() {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let (warm, count) = (3u32, 5u32);
        let data: Vec<f32> = (warm..warm + count).flat_map(input).collect();
        let a = q8_chunk_output(&mut kernels, &w, &shape, rw, warm, None, &data);
        let b = q8_chunk_output(&mut kernels, &w, &shape, rw, warm, Some(1), &data);
        for row in 0..count as usize {
            assert_ne!(
                a[row * 64..(row + 1) * 64],
                b[row * 64..(row + 1) * 64],
                "row {row} ignores cached position 1"
            );
        }
    }

    /// Agreement: the Q8 chunk of five rows after three cached
    /// positions and five Q8 decode steps from the same cache quantize the same rows, so they
    /// differ only by float reduction order and, rarely, by one code that a one-ulp change moves
    /// across a rounding boundary. Both paths must agree at least as well as the Q8 chunk agrees
    /// with the F32 chunk, whose gap is the quantization error itself (a floor of 1e-5 covers a
    /// run where the two forms happen to agree to that level). The causal-off and the
    /// wrong-length mutants differ by far more than that gap.
    #[test]
    fn a_q8_chunk_agrees_with_five_q8_decode_steps_at_least_as_well_as_with_the_f32_chunk() {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let (warm, count) = (3u32, 5u32);
        let data: Vec<f32> = (warm..warm + count).flat_map(input).collect();
        let chunked = q8_chunk_output(&mut kernels, &w, &shape, rw, warm, None, &data);
        // Five Q8 decode steps from the same warm-up.
        let mut cache = q8_warm_cache(&mut kernels, &w, &shape, rw, warm, None);
        let mut stepped = Vec::new();
        for n in warm..warm + count {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr, &w.layers[1].input_norm).unwrap();
            cache = next.cache;
            // The decode seam folds the output residual; the chunk keeps the mixer output, so
            // remove the residual row before comparing the two paths.
            stepped.extend(
                values(&mut kernels, &next.out)
                    .unwrap()
                    .iter()
                    .zip(&input(n))
                    .map(|(v, r)| v - r),
            );
        }
        assert_eq!(cache.len, 8);
        // The F32 chunk after three F32 decode steps.
        let mut f32_cache = KvCache::with_step(4);
        for n in 0..warm {
            let xr = upload(&mut kernels, vec![1, 64], &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            f32_cache = step(
                &mut ctx,
                &shape,
                rw,
                f32_cache,
                n,
                &xr,
                &w.layers[1].input_norm,
            )
            .unwrap()
            .cache;
        }
        let x = upload(&mut kernels, vec![count, 64], &data);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // The decode steps norm inside (the folded contract); the chunk keeps raw input.
        let x = ctx
            .kernels
            .rms_norm(&x, &w.layers[1].input_norm, shape.eps)
            .unwrap();
        let control = chunk(&mut ctx, &shape, rw, f32_cache, warm, &x).unwrap();
        let control = values(&mut kernels, &control.out).unwrap();
        let gap = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let quantization = gap(&chunked, &control);
        let paths = gap(&chunked, &stepped);
        println!("q8 chunk against f32 chunk {quantization}, against q8 steps {paths}");
        assert!(stepped.iter().all(|value| value.is_finite()));
        assert!(
            paths <= quantization.max(1e-5),
            "the Q8 chunk differs from five Q8 decode steps by {paths}, more than from the F32 \
             chunk ({quantization})"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_chunk_builds_lazy_outputs_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let x = rows(&mut kernels, 0, 3);
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let next = chunk(&mut ctx, &shape, rw, KvCache::empty(), 0, &x).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "chunk synchronized"
        );
        let (k, v) = f32_pair(&next.cache);
        kernels.evaluate(&[&next.out, k, v]).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "evaluate adds one"
        );
    }
    /// The flash-prefill kernel at the Qwen3.6-35B-A3B production shape (QH 16, KVH 2,
    /// D 256, REP 8) matches an F32 reference at least as well as the composite prefix
    /// attention. One chunk of 16,389 rows (not a multiple of eight; q_offset 0; depth
    /// 16K+) over a synthetic Q8 cache. The reference is the attention over the
    /// PRE-quantization F32 keys and values, so the quantization error is identical for
    /// both kernels and the comparison isolates kernel numerics. The composite and the
    /// reference run in the 256-row blocks of the integration (their score tensors would
    /// reach 1 GB in one call); the flash kernel takes the whole chunk in one call, which
    /// also exercises the non-multiple-of-eight threadgroup tail.
    #[test]
    fn q8_flash_prefill_d256_tolerance() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (h, kvh, d, rows) = (16u32, 2u32, 256u32, 16389u32);
        let (group, bits) = (64u32, 8u32);
        let block: u32 = 256;
        let kv: Vec<f32> = (0..kvh * rows * d)
            .map(|i| (((i * 37) % 509) as f32 - 254.0) / 254.0)
            .collect();
        let k = upload(&mut kernels, vec![1, kvh, rows, d], &kv);
        let v = upload(&mut kernels, vec![1, kvh, rows, d], &kv);
        let (kc, ks, kb) = kernels.quantize_rows(&k, group, bits).unwrap();
        let (vc, vs, vb) = kernels.quantize_rows(&v, group, bits).unwrap();
        let keys = QuantizedKv {
            codes: &kc,
            scales: &ks,
            biases: &kb,
        };
        let qvalues = QuantizedKv {
            codes: &vc,
            scales: &vs,
            biases: &vb,
        };
        let qd: Vec<f32> = (0..h * rows * d)
            .map(|i| (((i * 29) % 389) as f32 - 194.0) / 300.0)
            .collect();
        let q = upload(&mut kernels, vec![1, h, rows, d], &qd);

        let mut run_blocks = |f32_path: bool| -> Vec<f32> {
            let mut blocks: Vec<ExecutionTensor> = Vec::new();
            let mut start = 0u32;
            while start < rows {
                let end = (start + block).min(rows);
                let q_block = kernels
                    .slice(&q, &[0, 0, start, 0], &[1, h, end, d])
                    .unwrap();
                let out = if f32_path {
                    kernels.grouped_attention_prefix(&q_block, &k, &v, end, 1.0, end - start > 1)
                } else {
                    kernels.quantized_attention_prefix(
                        &q_block,
                        &keys,
                        &qvalues,
                        end,
                        1.0,
                        end - start > 1,
                        group,
                        bits,
                    )
                }
                .unwrap();
                blocks.push(out);
                start = end;
            }
            let views: Vec<&ExecutionTensor> = blocks.iter().collect();
            let whole = kernels.concatenate(&views, 2).unwrap();
            values(&mut kernels, &whole).unwrap()
        };

        let reference = run_blocks(true);
        let composite = run_blocks(false);
        let flash_out = kernels
            .q8_flash_prefill(&q, &keys, &qvalues, rows, 1.0, group, bits, None, 0)
            .unwrap();
        let flash = values(&mut kernels, &flash_out).unwrap();

        let max_err = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let composite_err = max_err(&composite, &reference);
        let flash_err = max_err(&flash, &reference);
        eprintln!("D256 tolerance: composite err {composite_err:.6}, flash err {flash_err:.6}");
        assert!(
            flash_err <= composite_err * 1.10 + 1e-6,
            "flash err {flash_err:.6} vs composite err {composite_err:.6}"
        );
    }

    /// Dense-shape (QH 24, KVH 4, D 256, REP 6) bench: the composite Q8 prefix attention in
    /// 256-row blocks against the flash kernel on a duplicate-padded query (24 -> 32 heads,
    /// each kv group of 6 padded with its own first 2 heads; the pad rows' outputs are
    /// discarded). Also checks the flash output's 24 real rows against the F32 reference.
    /// Temporary probe for the dense prefill headroom decision: run with --ignored.
    #[test]
    #[ignore]
    fn dense_rep6_flash_pad_bench() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (h, kvh, d, rows, cap) = (24u32, 4u32, 256u32, 512u32, 16_384u32);
        let (group, bits) = (64u32, 8u32);
        let block: u32 = 256;
        let kv: Vec<f32> = (0..kvh * cap * d)
            .map(|i| (((i * 37) % 509) as f32 - 254.0) / 254.0)
            .collect();
        let k = upload(&mut kernels, vec![1, kvh, cap, d], &kv);
        let v = upload(&mut kernels, vec![1, kvh, cap, d], &kv);
        let (kc, ks, kb) = kernels.quantize_rows(&k, group, bits).unwrap();
        let (vc, vs, vb) = kernels.quantize_rows(&v, group, bits).unwrap();
        let keys = QuantizedKv {
            codes: &kc,
            scales: &ks,
            biases: &kb,
        };
        let values_q = QuantizedKv {
            codes: &vc,
            scales: &vs,
            biases: &vb,
        };
        let qd: Vec<f32> = (0..h * rows * d)
            .map(|i| (((i * 29) % 389) as f32 - 194.0) / 300.0)
            .collect();
        let q = upload(&mut kernels, vec![1, h, rows, d], &qd);
        let scale = 1.0 / (d as f32).sqrt();

        // MLX caches lazy graphs by inputs: identical iterations are free. Each timed
        // iteration uploads a fresh distinct q so the attention really runs each time.
        let variants: Vec<ExecutionTensor> = (0..6)
            .map(|n| {
                let data: Vec<f32> = (0..h * rows * d)
                    .map(|i| (((i * 29 + n * 7) % 389) as f32 - 194.0) / 300.0)
                    .collect();
                upload(&mut kernels, vec![1, h, rows, d], &data)
            })
            .collect();

        // The duplicate pad: per kv group g, [q[g*6..g*6+6], q[g*6..g*6+2]] -> 32 heads.
        let build_pad = |kernels: &mut MlxKernels, q: &ExecutionTensor| {
            let mut parts: Vec<ExecutionTensor> = Vec::with_capacity(8);
            for g in 0..kvh {
                let base = g * 6;
                parts.push(
                    kernels
                        .slice(q, &[0, base, 0, 0], &[1, base + 6, rows, d])
                        .unwrap(),
                );
                parts.push(
                    kernels
                        .slice(q, &[0, base, 0, 0], &[1, base + 2, rows, d])
                        .unwrap(),
                );
            }
            let views: Vec<&ExecutionTensor> = parts.iter().collect();
            kernels.concatenate(&views, 1).unwrap()
        };

        // Timing: each iteration forces the whole output to host (a full readback is an
        // unambiguous GPU sync; evaluate alone enqueues async and a probe element did not
        // force the producers). The readback itself (~12 MB) is inside both arms equally.
        let sync = |kernels: &mut MlxKernels, out: &ExecutionTensor| {
            let _ = values(kernels, out).unwrap();
        };

        let (cf, composite_ms, composite) = {
            let mut best = f64::INFINITY;
            let mut first = 0.0;
            let mut whole = None;
            for i in 0..5 {
                let start = std::time::Instant::now();
                let q_i = &variants[i + 1];
                let mut blocks: Vec<ExecutionTensor> = Vec::new();
                let mut start_row = 0u32;
                while start_row < rows {
                    let end = (start_row + block).min(rows);
                    // Tail-of-prefix causality: a block ending at row `end` of a chunk of
                    // `rows` rows at depth `cap` sees `cap - rows + end` keys - the same
                    // geometry the flash call gives the whole chunk.
                    let len_b = cap - rows + end;
                    let q_block = kernels
                        .slice(q_i, &[0, 0, start_row, 0], &[1, h, end, d])
                        .unwrap();
                    blocks.push(
                        kernels
                            .quantized_attention_prefix(
                                &q_block, &keys, &values_q, len_b, scale, true, group, bits,
                            )
                            .unwrap(),
                    );
                    start_row = end;
                }
                let views: Vec<&ExecutionTensor> = blocks.iter().collect();
                let out = kernels.concatenate(&views, 2).unwrap();
                sync(&mut kernels, &out);
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                if i == 0 {
                    first = elapsed;
                } else {
                    best = best.min(elapsed);
                }
                whole = Some(out);
            }
            (first, best, whole.unwrap())
        };
        let (ff, flash_ms, flash) = {
            let mut best = f64::INFINITY;
            let mut first = 0.0;
            let mut sliced = None;
            for i in 0..5 {
                let start = std::time::Instant::now();
                let pad_q = build_pad(&mut kernels, &variants[i + 1]);
                let out = kernels
                    .q8_flash_prefill(&pad_q, &keys, &values_q, cap, scale, group, bits, None, 0)
                    .unwrap();
                let out = unpad_rep6_output(&mut kernels, &out, h, kvh).unwrap();
                sync(&mut kernels, &out);
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                if i == 0 {
                    first = elapsed;
                } else {
                    best = best.min(elapsed);
                }
                sliced = Some(out);
            }
            (first, best, sliced.unwrap())
        };
        eprintln!(
            "dense REP6 rows={rows} depth={cap}: composite {composite_ms:.1}ms (first {cf:.1}), flash+pad {flash_ms:.1}ms (first {ff:.1})"
        );

        // Numerics: the F32 reference in the same blocks, over the real 24 heads, on the
        // same query variant the timed arms' final outputs hold (variants[5]).
        let mut reference: Vec<f32> = Vec::new();
        let mut start = 0u32;
        while start < rows {
            let end = (start + block).min(rows);
            let len_b = cap - rows + end;
            let q_block = kernels
                .slice(&variants[5], &[0, 0, start, 0], &[1, h, end, d])
                .unwrap();
            let out = kernels
                .grouped_attention_prefix(&q_block, &k, &v, len_b, scale, end - start > 1)
                .unwrap();
            let mut block_values = values(&mut kernels, &out).unwrap();
            reference.append(&mut block_values);
            start = end;
        }
        let composite_flat = values(&mut kernels, &composite).unwrap();
        let flash_flat = values(&mut kernels, &flash).unwrap();
        let max_err = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max)
        };
        let composite_err = max_err(&composite_flat, &reference);
        let flash_err = max_err(&flash_flat, &reference);
        eprintln!(
            "dense REP6 tolerance: composite err {composite_err:.6}, flash err {flash_err:.6}"
        );
        assert!(
            flash_err <= composite_err * 1.10 + 1e-6,
            "flash err {flash_err:.6} vs composite err {composite_err:.6}"
        );
    }

    /// The flash-prefill kernel through the INTEGRATION pad path (`pad_rep6_query` + the
    /// slice back to 24 heads) at the dense shape (QH 24, KVH 4, D 256, REP 6) matches an
    /// F32 reference at least as well as the composite prefix attention. Small depth, so it
    /// runs in the suite; the deep measurement lives in `dense_rep6_flash_pad_bench`.
    #[test]
    fn the_q8_flash_prefill_dense_rep6_pad_tolerance() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (h, kvh, d, rows) = (24u32, 4u32, 256u32, 96u32);
        let (group, bits) = (64u32, 8u32);
        let block: u32 = 256;
        let kv: Vec<f32> = (0..kvh * rows * d)
            .map(|i| (((i * 37) % 509) as f32 - 254.0) / 254.0)
            .collect();
        let k = upload(&mut kernels, vec![1, kvh, rows, d], &kv);
        let v = upload(&mut kernels, vec![1, kvh, rows, d], &kv);
        let (kc, ks, kb) = kernels.quantize_rows(&k, group, bits).unwrap();
        let (vc, vs, vb) = kernels.quantize_rows(&v, group, bits).unwrap();
        let keys = QuantizedKv {
            codes: &kc,
            scales: &ks,
            biases: &kb,
        };
        let values_q = QuantizedKv {
            codes: &vc,
            scales: &vs,
            biases: &vb,
        };
        let qd: Vec<f32> = (0..h * rows * d)
            .map(|i| (((i * 29) % 389) as f32 - 194.0) / 300.0)
            .collect();
        let q = upload(&mut kernels, vec![1, h, rows, d], &qd);
        let scale = 1.0 / (d as f32).sqrt();

        let mut run_blocks = |f32_path: bool| -> Vec<f32> {
            let mut start = 0u32;
            let mut blocks: Vec<ExecutionTensor> = Vec::new();
            while start < rows {
                let end = (start + block).min(rows);
                let q_block = kernels
                    .slice(&q, &[0, 0, start, 0], &[1, h, end, d])
                    .unwrap();
                let out = if f32_path {
                    kernels.grouped_attention_prefix(&q_block, &k, &v, end, scale, end - start > 1)
                } else {
                    kernels.quantized_attention_prefix(
                        &q_block,
                        &keys,
                        &values_q,
                        end,
                        scale,
                        end - start > 1,
                        group,
                        bits,
                    )
                }
                .unwrap();
                blocks.push(out);
                start = end;
            }
            let views: Vec<&ExecutionTensor> = blocks.iter().collect();
            let whole = kernels.concatenate(&views, 2).unwrap();
            values(&mut kernels, &whole).unwrap()
        };

        let reference = run_blocks(true);
        let composite = run_blocks(false);
        let padded = pad_rep6_query(&mut kernels, &q, h, kvh).unwrap();
        assert_eq!(padded.shape(), &[1, kvh * 8, rows, d]);
        let flash_out = kernels
            .q8_flash_prefill(&padded, &keys, &values_q, rows, scale, group, bits, None, 0)
            .unwrap();
        let flash = unpad_rep6_output(&mut kernels, &flash_out, h, kvh).unwrap();
        let flash = values(&mut kernels, &flash).unwrap();
        // A last synchronizing evaluate flushes the retire list so the adapter teardown
        // releases the process owner before the next native test initializes (without it,
        // the leftover lazy arrays hold the owner and sliced_* fails with "already holds").
        kernels.evaluate(&[&flash_out]).unwrap();

        let max_err = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max)
        };
        let composite_err = max_err(&composite, &reference);
        let flash_err = max_err(&flash, &reference);
        eprintln!(
            "dense REP6 tolerance: composite err {composite_err:.6}, flash err {flash_err:.6}"
        );
        assert!(
            flash_err <= composite_err * 1.10 + 1e-6,
            "flash err {flash_err:.6} vs composite err {composite_err:.6}"
        );
    }
}
