//! Resident dense attention of the Q36 family (packet Q36-C1; decision 0046).
//!
//! The graph follows the dense branch of `flash_next/resident/qsa.rs` and the rules of
//! `qwen36/host.rs::attn_step`: the query projection holds, per head, the query rows and then the
//! gate rows; the query and key take a weighted RMS norm with the weight as stored (Flash-Next adds
//! 1); RoPE rotates the first `rotary_dim` dimensions; the head output is multiplied by
//! `sigmoid(gate)` before the output projection. There is no indexer and no block selection.
//!
//! The cache (packet Q36-KV-INTEGRATE) is a pair of F32 capacity buffers `[1, KVH, cap, D]` for
//! keys and for values, preallocated in `KV_STEP` quanta and written in place by the consuming
//! `kv_append`: the first `len` positions are valid, the rest is slack with no value. A decode
//! step appends one position at `[0, 0, len, 0]`; a chunk appends its rows the same way. When
//! `len + rows` passes `cap`, the layer builds a larger buffer of `round_up(len + rows, KV_STEP)`
//! positions — the valid prefix `[0, len)` copied once by a `slice` (a copy is acceptable once
//! per growth), zero slack, `concatenate` — and appends into that. Attention runs over the
//! copy-free prefix view of the buffers (`grouped_attention_prefix`), so a step no longer copies
//! the whole cache per token as the `concatenate` formulation did (about 671 MB per token at
//! depth 8,192 on the real model). A snapshot shares the buffers; the next in-place append on a
//! shared buffer copies it once (the operation falls back when the handle is not unique), so the
//! snapshot keeps its own rows. The admission charge of the session budget stays LOGICAL, `len`
//! positions of `growth_bytes` per layer; the capacity slack of a whole session stays below one
//! `KV_STEP` of growth bytes per attention layer (256 * 4,096 * 10 = 10,485,760 bytes on the real
//! model). That slack bound EXCLUDES the growth transient — the old buffer, its slice copy, and
//! the new buffer alive together at a growth, about 100 MB per layer at depth 8,192 — which the
//! session reserve margin and the retire cap cover. The step builds a lazy graph only.

use super::kv_plan;
use super::ops::{Ctx, cast_to, dim, linear, to_act};
use super::weights::RAttn;
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    kernels::{BinaryOp, RopeSpec, UnaryOp},
    tensor::{Dtype, ExecutionTensor},
};

/// Bytes of one F32 element, the resident cache dtype. A BF16 cache is parked (it needs a cast
/// operation in `MlxKernels`).
const F32_BYTES: u64 = 4;

/// Keys and values of one dense attention layer, as capacity buffers. `None` while `len` is 0.
pub(super) struct KvCache {
    /// Keys after the norm and RoPE, `[1, KVH, cap, D]`; the first `len` positions are valid.
    pub(super) k: Option<ExecutionTensor>,
    /// Values, `[1, KVH, cap, D]`; the first `len` positions are valid.
    pub(super) v: Option<ExecutionTensor>,
    /// Valid positions.
    pub(super) len: usize,
    /// Allocated positions of each buffer. Zero while the cache holds no buffers.
    pub(super) cap: usize,
    /// The growth quantum in positions (packet Q36-KV-INTEGRATE). The default is `KV_STEP`;
    /// tests use a small step so a run crosses a growth boundary within a few tokens.
    pub(super) step: usize,
}

impl KvCache {
    /// A cache with no position that grows in `KV_STEP` quanta.
    pub(super) fn empty() -> Self {
        Self::with_step(kv_plan::KV_STEP)
    }

    /// A cache with no position that grows in `step`-position quanta.
    pub(super) fn with_step(step: usize) -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            cap: 0,
            step,
        }
    }

    /// Stored bytes of the keys and values, from the tensors' own sizes: the capacity slack is
    /// included. Reports and tests read this; the admission charge is logical (`state::kv_bytes`).
    pub(super) fn byte_len(&self) -> u64 {
        [&self.k, &self.v]
            .into_iter()
            .flatten()
            .map(ExecutionTensor::byte_len)
            .fold(0, u64::saturating_add)
    }

    /// Validates the buffers against `kv_heads` and `width` with no native call: a pair of
    /// `[1, KVH, cap, D]` tensors with `cap >= len`, or no pair at a zero `len` and `cap`.
    /// A mismatch is the `BadRequest` of a cache that does not match its shape.
    pub(super) fn check_buffers(&self, kv_heads: u32, width: u32) -> Result<(), EngError> {
        match (&self.k, &self.v) {
            (None, None) if self.len == 0 && self.cap == 0 => Ok(()),
            (Some(k), Some(v)) if self.len > 0 => {
                let kv_shape = [1, kv_heads, dim(self.cap)?, width];
                if self.cap < self.len
                    || k.shape() != kv_shape.as_slice()
                    || v.shape() != kv_shape.as_slice()
                {
                    return Err(mismatch());
                }
                Ok(())
            }
            _ => Err(mismatch()),
        }
    }
}

/// Bytes that one more position adds to the cache of one layer: keys and values, `KVH * D` F32
/// each. The real model adds 2 * 2 * 256 * 4 = 4,096 bytes per layer and 40,960 bytes over 10
/// layers. This is the LOGICAL admission charge of one position (packet Q36-KV-INTEGRATE): the
/// capacity buffers hold up to one `KV_STEP` of slack per layer on top of it, so the slack of a
/// whole session stays below `KV_STEP * growth_bytes` summed over the attention layers —
/// 256 * 4,096 * 10 = 10,485,760 bytes on the real model — and `state::kv_bytes` charges only
/// this formula, never the stored bytes. The bound excludes the growth transient (the old
/// buffer, its slice copy, and the new buffer alive at a growth), which the session reserve
/// margin and the retire cap cover.
pub(super) fn growth_bytes(shape: &HostShape) -> Result<u64, EngError> {
    let a = &shape.attn;
    u64::try_from(a.heads_kv)
        .ok()
        .zip(u64::try_from(a.head_dim).ok())
        .and_then(|(heads, width)| {
            heads
                .checked_mul(width)?
                .checked_mul(2)?
                .checked_mul(F32_BYTES)
        })
        .ok_or_else(|| EngError::MemoryPlan("attention growth bytes overflow u64".into()))
}

/// One resident attention step: the layer output and the replacement cache.
pub(super) struct AttnStep {
    /// Layer output `[1, H]` (decode) or `[T, H]` (chunk).
    pub(super) out: ExecutionTensor,
    /// The cache with the appended rows. The caller replaces the old cache with it.
    pub(super) cache: KvCache,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident attention dimension overflows".into())
}

fn mismatch() -> EngError {
    EngError::BadRequest("resident attention cache does not match its shape".into())
}

/// A lazy F32 zero buffer `[1, kv_heads, positions, width]`, built from the rank-0 zero constant
/// with `reshape` and `repeat`. No host import runs, so building or growing a cache adds no
/// synchronization to a step.
fn zero_rows(
    ctx: &mut Ctx<'_>,
    kv_heads: u32,
    positions: u32,
    width: u32,
) -> Result<ExecutionTensor, EngError> {
    let seed = ctx.kernels.reshape(&ctx.consts.zero, &[1, 1, 1, 1])?;
    let seed = ctx.kernels.repeat(&seed, kv_heads, 1)?;
    let seed = ctx.kernels.repeat(&seed, positions, 2)?;
    ctx.kernels.repeat(&seed, width, 3)
}

/// The buffer of a growth: the valid prefix of `old` (`[0, len)` of the time axis, copied once
/// by the `slice`) followed by zero slack, so the result holds `cap` positions.
fn grown(
    ctx: &mut Ctx<'_>,
    old: &ExecutionTensor,
    kv_heads: u32,
    width: u32,
    len: u32,
    cap: u32,
) -> Result<ExecutionTensor, EngError> {
    let prefix = ctx
        .kernels
        .slice(old, &[0, 0, 0, 0], &[1, kv_heads, len, width])?;
    let zeros = zero_rows(ctx, kv_heads, cap - len, width)?;
    ctx.kernels.concatenate(&[&prefix, &zeros], 2)
}

/// Appends `rows` new positions — `k_new` and `v_new`, each `[1, KVH, rows, D]` F32 — to the
/// validated `cache` and returns the next cache (packet Q36-KV-INTEGRATE). When `len + rows`
/// passes `cap`, the buffers grow to `kv_plan::next_capacity(len, rows, cap, step)` positions:
/// the valid prefix copied once and zero slack, for keys and for values. The append itself is
/// the consuming `kv_append` at `[0, 0, len, 0]`: a unique buffer is written in place (MLX
/// donates it), a buffer shared with a snapshot is copied once. Every call here builds a lazy
/// graph only; the caller validated the cache before its first native call, and after this
/// consumed a buffer any error leaves the caller's state poisoned until `reset`.
pub(super) fn append(
    ctx: &mut Ctx<'_>,
    kv_heads: u32,
    width: u32,
    cache: KvCache,
    rows: u32,
    k_new: &ExecutionTensor,
    v_new: &ExecutionTensor,
) -> Result<KvCache, EngError> {
    let rows = usize::try_from(rows).map_err(|_| overflow())?;
    let next_len = cache.len.checked_add(rows).ok_or_else(overflow)?;
    let cap = kv_plan::next_capacity(cache.len, rows, cache.cap, cache.step)?;
    let len = dim(cache.len)?;
    let (k, v) = (cache.k, cache.v);
    let (k, v) = match (k, v) {
        (Some(k), Some(v)) if cap == cache.cap => (
            ctx.kernels.kv_append(k, k_new, &[0, 0, len, 0])?,
            ctx.kernels.kv_append(v, v_new, &[0, 0, len, 0])?,
        ),
        (Some(k), Some(v)) => {
            let cap_dim = dim(cap)?;
            let k_buf = grown(ctx, &k, kv_heads, width, len, cap_dim)?;
            let v_buf = grown(ctx, &v, kv_heads, width, len, cap_dim)?;
            (
                ctx.kernels.kv_append(k_buf, k_new, &[0, 0, len, 0])?,
                ctx.kernels.kv_append(v_buf, v_new, &[0, 0, len, 0])?,
            )
        }
        (None, None) => {
            // The first rows of an empty cache allocate round_up(rows, step) positions.
            let cap_dim = dim(cap)?;
            let k_buf = zero_rows(ctx, kv_heads, cap_dim, width)?;
            let v_buf = zero_rows(ctx, kv_heads, cap_dim, width)?;
            (
                ctx.kernels.kv_append(k_buf, k_new, &[0, 0, 0, 0])?,
                ctx.kernels.kv_append(v_buf, v_new, &[0, 0, 0, 0])?,
            )
        }
        _ => return Err(mismatch()),
    };
    Ok(KvCache {
        k: Some(k),
        v: Some(v),
        len: next_len,
        cap,
        step: cache.step,
    })
}

/// Runs one dense attention decode step at `pos` on `x` (`H` values, any shape). It builds a lazy
/// graph only: no readback and no evaluate. `cache` is CONSUMED: `kv_append` writes the new
/// position into its buffer in place (a growth replaces the buffer), so the caller hands the
/// layer's cache over and keeps only the returned one. The host validation — the position, the
/// buffer shapes, and the capacity invariants — runs before the first native call and before any
/// buffer is consumed; once a buffer is consumed, any error leaves the caller's state poisoned
/// until `reset` (the rule of the decode step, which takes the layer states before it runs). A
/// cache that holds a number of positions other than `pos`, or tensors of the wrong shape,
/// returns `BadRequest` before the first native call.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: KvCache,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<AttnStep, EngError> {
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    let len = cache.len;
    if usize::try_from(pos).ok() != Some(len) {
        return Err(EngError::BadRequest(format!(
            "attention cache holds {} positions, step is at {pos}",
            len
        )));
    }
    cache.check_buffers(kv_heads, width)?;
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let next_len = len.checked_add(1).ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: dim(a.rotary_dim)?,
        base: a.theta,
        scale: 1.0,
        offset: pos,
    };
    // 1. Query and gate, interleaved per head: head h is qp[h * 2D..(h + 1) * 2D].
    let qp = linear(ctx, &w.q, x)?;
    let qp = ctx.kernels.reshape(&qp, &[heads, pair])?;
    let q_raw = ctx.kernels.slice(&qp, &[0, 0], &[heads, width])?;
    let gate = ctx.kernels.slice(&qp, &[0, width], &[heads, pair])?;
    // 2. Query norm with the weight as stored, then RoPE at `pos`.
    let q = ctx.kernels.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
    let q = ctx.kernels.reshape(&q, &[1, heads, 1, width])?;
    let q = ctx.kernels.rope_scaled(&q, &rope)?;
    // 3. New key: norm and RoPE per key-value head.
    let kp = linear(ctx, &w.k, x)?;
    let kp = ctx.kernels.reshape(&kp, &[kv_heads, width])?;
    let k_new = ctx.kernels.rms_norm(&kp, &w.k_norm, shape.eps)?;
    let k_new = ctx.kernels.reshape(&k_new, &[1, kv_heads, 1, width])?;
    let k_new = ctx.kernels.rope_scaled(&k_new, &rope)?;
    // 4. New value.
    let v_new = linear(ctx, &w.v, x)?;
    let v_new = ctx.kernels.reshape(&v_new, &[1, kv_heads, 1, width])?;
    // The cache stays F32 whatever the activation dtype (packet Q36-BF16-a; a BF16 cache is a
    // parked lever of decision 0046 item 5).
    let k_new = cast_to(ctx, &k_new, Dtype::F32)?;
    let v_new = cast_to(ctx, &v_new, Dtype::F32)?;
    // 5. Append into the capacity buffer (a growth builds the larger buffer first).
    let cache = append(ctx, kv_heads, width, cache, 1, &k_new, &v_new)?;
    // 6. Attention over every valid position, as a copy-free prefix view of the capacity
    // buffers. Head h reads key-value head h / (QH / KVH).
    let (k_buf, v_buf) = match (&cache.k, &cache.v) {
        (Some(k), Some(v)) => (k, v),
        _ => return Err(mismatch()),
    };
    let scale = 1.0 / (width as f32).sqrt();
    let att =
        ctx.kernels
            .grouped_attention_prefix(&q, k_buf, v_buf, dim(next_len)?, scale, false)?;
    // 7. Sigmoid output gate and output projection.
    let att = ctx.kernels.reshape(&att, &[heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    // The attention output is F32 over the F32 cache; the projection takes the activation dtype.
    let o = to_act(ctx, &o)?;
    let out = linear(ctx, &w.o, &o)?;
    Ok(AttnStep { out, cache })
}

/// The cache of the old formulation (test-only): exactly the valid positions, `[1, KVH, len, D]`.
#[cfg(test)]
pub(super) struct ConcatCache {
    /// Keys, `[1, KVH, len, D]`.
    pub(super) k: Option<ExecutionTensor>,
    /// Values, `[1, KVH, len, D]`.
    pub(super) v: Option<ExecutionTensor>,
    /// Valid positions.
    pub(super) len: usize,
}

/// The decode step exactly as it was before packet Q36-KV-INTEGRATE (test-only): the cache is
/// `[1, KVH, len, D]` and every step rebuilds it with `concatenate`. The bit equality of this
/// formulation and the capacity buffers — asserted on every step by the tests of this file and
/// of `prefill/attn.rs` — proves that the new cache changes no numerics.
#[cfg(test)]
pub(super) fn concat_step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: &ConcatCache,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, ConcatCache), EngError> {
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    let t = dim(cache.len)?;
    let past = match (&cache.k, &cache.v) {
        (None, None) if cache.len == 0 => None,
        (Some(k), Some(v)) if cache.len > 0 => {
            let kv_shape = [1, kv_heads, t, width];
            if k.shape() != kv_shape.as_slice() || v.shape() != kv_shape.as_slice() {
                return Err(mismatch());
            }
            Some((k, v))
        }
        _ => return Err(mismatch()),
    };
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let next_len = cache.len.checked_add(1).ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: dim(a.rotary_dim)?,
        base: a.theta,
        scale: 1.0,
        offset: pos,
    };
    // 1. Query and gate, interleaved per head.
    let qp = linear(ctx, &w.q, x)?;
    let qp = ctx.kernels.reshape(&qp, &[heads, pair])?;
    let q_raw = ctx.kernels.slice(&qp, &[0, 0], &[heads, width])?;
    let gate = ctx.kernels.slice(&qp, &[0, width], &[heads, pair])?;
    // 2. Query norm, then RoPE at `pos`.
    let q = ctx.kernels.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
    let q = ctx.kernels.reshape(&q, &[1, heads, 1, width])?;
    let q = ctx.kernels.rope_scaled(&q, &rope)?;
    // 3. New key: norm and RoPE per key-value head.
    let kp = linear(ctx, &w.k, x)?;
    let kp = ctx.kernels.reshape(&kp, &[kv_heads, width])?;
    let k_new = ctx.kernels.rms_norm(&kp, &w.k_norm, shape.eps)?;
    let k_new = ctx.kernels.reshape(&k_new, &[1, kv_heads, 1, width])?;
    let k_new = ctx.kernels.rope_scaled(&k_new, &rope)?;
    // 4. New value.
    let v_new = linear(ctx, &w.v, x)?;
    let v_new = ctx.kernels.reshape(&v_new, &[1, kv_heads, 1, width])?;
    let k_new = cast_to(ctx, &k_new, Dtype::F32)?;
    let v_new = cast_to(ctx, &v_new, Dtype::F32)?;
    // 5. Rebuild the cache with `concatenate`.
    let (k_all, v_all) = match past {
        None => (k_new, v_new),
        Some((k, v)) => (
            ctx.kernels.concatenate(&[k, &k_new], 2)?,
            ctx.kernels.concatenate(&[v, &v_new], 2)?,
        ),
    };
    // 6. Attention over every position.
    let scale = 1.0 / (width as f32).sqrt();
    let att = ctx
        .kernels
        .grouped_attention(&q, &k_all, &v_all, scale, false)?;
    // 7. Sigmoid output gate and output projection.
    let att = ctx.kernels.reshape(&att, &[heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    let o = to_act(ctx, &o)?;
    let out = linear(ctx, &w.o, &o)?;
    Ok((
        out,
        ConcatCache {
            k: Some(k_all),
            v: Some(v_all),
            len: next_len,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::ops::{assert_parity, f32_tensor, values};
    use super::super::weights::{RMixer, ResidentWeights};
    use super::*;
    use crate::family::flash_next::math::rope_half;
    use crate::family::flash_next::stored::StoredWeight;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, AttnW, KvCache as HostCache, MixerW, rms_plain};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;
    use crate::weights::StreamedCheckpoint;

    /// Input of step `n`: `x[i] = (((i + 5 n) % 11) - 5) / 16`, exact in `f32`.
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

    fn upload(kernels: &mut MlxKernels, x: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(vec![1, x.len() as u32], x).unwrap())
            .unwrap()
    }

    /// The first `len` positions of each key-value head of a `[1, KVH, cap, D]` buffer readback,
    /// in the `[1, KVH, len, D]` layout: element (g, t, d) of the capacity layout sits at
    /// (g * cap + t) * D + d, so head g keeps its rows only below `len`.
    fn prefix_rows(full: &[f32], heads: usize, cap: usize, len: usize, width: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for g in 0..heads {
            let head = &full[g * cap * width..(g + 1) * cap * width];
            out.extend_from_slice(&head[..len * width]);
        }
        out
    }

    /// Nine decode steps over a growth quantum of 3 (the buffers grow three times: caps 3, 6, 9)
    /// match the host oracle within the parity bounds, are bit-equal to the old
    /// concatenate-based step on every step, and keep the rotated keys and values in the valid
    /// prefix of the capacity buffers (packet Q36-KV-INTEGRATE 6(b)).
    #[test]
    fn nine_steps_over_three_growths_match_the_host_and_the_old_formulation_bitwise() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Attn(host_w) = &view.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut host_cache = HostCache::default();
        let mut cache = KvCache::with_step(3);
        let mut concat = ConcatCache {
            k: None,
            v: None,
            len: 0,
        };
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let growth = growth_bytes(&shape).unwrap();
        // Keys and values by definition, one entry per position: the weighted RMS norm of the key
        // projection, then RoPE; the value projection as is.
        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut vals: Vec<Vec<f32>> = Vec::new();
        for n in 0..9u32 {
            let x = input(n);
            let want = host::attn_step(&mut cpu, &shape, host_w, &mut host_cache, n, &x).unwrap();
            let mut key = Vec::new();
            let kp = cpu.linear(host_w.k, &x).unwrap();
            for g in 0..hkv {
                let mut head =
                    rms_plain(&kp[g * hd..(g + 1) * hd], host_w.k_norm, shape.eps).unwrap();
                rope_half(&mut head, shape.attn.rotary_dim, n, shape.attn.theta).unwrap();
                key.extend(head);
            }
            keys.push(key);
            vals.push(cpu.linear(host_w.v, &x).unwrap());
            let xr = upload(&mut kernels, &x);
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            let old = concat_step(&mut ctx, &shape, rw, &concat, n, &xr).unwrap();
            let t = n as usize + 1;
            // The cache arithmetic: one growth every three positions.
            let want_cap = [3, 3, 3, 6, 6, 6, 9, 9, 9][n as usize];
            assert_eq!(next.out.shape(), [1, 64], "step {n} output shape");
            assert_eq!(next.cache.len, t, "step {n} cache length");
            assert_eq!(next.cache.cap, want_cap, "step {n} capacity");
            assert_eq!(
                next.cache.k.as_ref().unwrap().shape(),
                [1, hkv as u32, want_cap as u32, hd as u32],
                "step {n} key shape"
            );
            // The host oracle within the parity bounds, and the old formulation bit for bit.
            let out = values(&mut kernels, &next.out).unwrap();
            assert_parity(&out, &want, &format!("step {n} output"));
            assert_eq!(
                out,
                values(&mut kernels, &old.0).unwrap(),
                "step {n} output differs from the concatenate formulation"
            );
            // The caches agree bit for bit on the valid prefix.
            let got_k = values(&mut kernels, next.cache.k.as_ref().unwrap()).unwrap();
            let got_v = values(&mut kernels, next.cache.v.as_ref().unwrap()).unwrap();
            assert_eq!(
                prefix_rows(&got_k, hkv, want_cap, t, hd),
                values(&mut kernels, old.1.k.as_ref().unwrap()).unwrap(),
                "step {n} cached keys differ from the concatenate formulation"
            );
            assert_eq!(
                prefix_rows(&got_v, hkv, want_cap, t, hd),
                values(&mut kernels, old.1.v.as_ref().unwrap()).unwrap(),
                "step {n} cached values differ from the concatenate formulation"
            );
            // The storage stays within the logical bytes plus one step of slack.
            let logical = kv_plan::logical_bytes(t, growth).unwrap();
            assert_eq!(
                next.cache.byte_len(),
                want_cap as u64 * growth,
                "step {n} stored bytes follow the capacity"
            );
            assert!(
                next.cache.byte_len() <= logical + 3 * growth,
                "step {n}: storage {} exceeds the logical {logical} plus one step of slack",
                next.cache.byte_len()
            );
            cache = next.cache;
            concat = old.1;
        }
        // The valid prefix holds the rotated keys and the values by definition.
        let t = keys.len();
        let (mut want_k, mut want_v) = (Vec::new(), Vec::new());
        for g in 0..hkv {
            for pos in 0..t {
                want_k.extend_from_slice(&keys[pos][g * hd..(g + 1) * hd]);
                want_v.extend_from_slice(&vals[pos][g * hd..(g + 1) * hd]);
            }
        }
        let got_k = values(&mut kernels, cache.k.as_ref().unwrap()).unwrap();
        let got_v = values(&mut kernels, cache.v.as_ref().unwrap()).unwrap();
        assert_parity(
            &prefix_rows(&got_k, hkv, cache.cap, t, hd),
            &want_k,
            "cached keys",
        );
        assert_parity(
            &prefix_rows(&got_v, hkv, cache.cap, t, hd),
            &want_v,
            "cached values",
        );
    }

    #[test]
    fn growth_bytes_follow_the_cache_dimensions() {
        // Tiny GQA: 2 key-value heads * 32 * 2 tensors * 4 bytes = 512. The real model: 2 * 256 * 2 * 4 = 4,096.
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let oracle = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
        assert_eq!(growth_bytes(oracle.shape()).unwrap(), 512);
        let real = HostShape::from_spec(&crate::family::spec::QWEN3_6_35B_A3B, 64).unwrap();
        assert_eq!(growth_bytes(&real).unwrap(), 4_096);
        assert_eq!(
            10 * growth_bytes(&real).unwrap(),
            40_960,
            "ten attention layers"
        );
    }

    #[test]
    fn a_cache_of_the_wrong_length_shape_or_capacity_is_rejected_before_any_native_call() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let xr = upload(&mut kernels, &input(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        // An empty cache at position 3.
        let wrong_len = step(&mut ctx, &shape, rw, KvCache::empty(), 3, &xr);
        assert!(
            matches!(&wrong_len, Err(EngError::BadRequest(m)) if m.contains("cache holds 0 positions")),
            "{:?}",
            wrong_len.as_ref().err()
        );
        // Keys with one key-value head instead of two, at the right length.
        let one_head = |ctx: &mut Ctx<'_>| {
            ctx.kernels
                .import_tensor(&f32_tensor(vec![1, 1, 1, 32], &[0.0; 32]).unwrap())
                .unwrap()
        };
        let bad = KvCache {
            k: Some(one_head(&mut ctx)),
            v: Some(one_head(&mut ctx)),
            len: 1,
            cap: 1,
            step: 3,
        };
        let wrong_shape = step(&mut ctx, &shape, rw, bad, 1, &xr);
        assert!(
            matches!(&wrong_shape, Err(EngError::BadRequest(m)) if m.contains("does not match its shape")),
            "{:?}",
            wrong_shape.as_ref().err()
        );
        // A capacity below the valid length, with MATCHING buffer shapes (review B finding 5):
        // `[1, 2, 1, 32]` buffers of one position that claim two valid positions pass the shape
        // comparison and reach the `cap < len` clause of `check_buffers`.
        let two_heads = |ctx: &mut Ctx<'_>| {
            ctx.kernels
                .import_tensor(&f32_tensor(vec![1, 2, 1, 32], &[0.0; 64]).unwrap())
                .unwrap()
        };
        let bad = KvCache {
            k: Some(two_heads(&mut ctx)),
            v: Some(two_heads(&mut ctx)),
            len: 2,
            cap: 1,
            step: 3,
        };
        let wrong_cap = step(&mut ctx, &shape, rw, bad, 2, &xr);
        assert!(
            matches!(&wrong_cap, Err(EngError::BadRequest(m)) if m.contains("does not match its shape")),
            "{:?}",
            wrong_cap.as_ref().err()
        );
    }

    /// The key and value row of position `n` by definition (the derivation of the nine-step
    /// test): the weighted RMS norm of the key projection and RoPE per key-value head, and the
    /// value projection as is. Pure host arithmetic, no state.
    fn host_kv_rows(
        cpu: &mut CpuProjector<'_>,
        shape: &HostShape,
        host_w: &AttnW<'_, StoredWeight>,
        n: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let x = input(n);
        let kp = cpu.linear(host_w.k, &x).unwrap();
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let mut key = Vec::new();
        for g in 0..hkv {
            let mut head = rms_plain(&kp[g * hd..(g + 1) * hd], host_w.k_norm, shape.eps).unwrap();
            rope_half(&mut head, shape.attn.rotary_dim, n, shape.attn.theta).unwrap();
            key.extend(head);
        }
        (key, cpu.linear(host_w.v, &x).unwrap())
    }

    /// Packet Q36-KV-INTEGRATE 6(e), review B finding 2: after warm-up, appending to a unique
    /// cache does not grow the MLX peak by a buffer — the method of
    /// `kv_append_donates_the_buffer_over_200_appends` (engine/tests/native_kv_ops.rs) driven
    /// through the real attention step. The step quantum 65,536 gives each capacity buffer of
    /// the tiny layer (2 key-value heads of width 32) 16 MiB, the order of the real layer's
    /// `[1, 2, 16384, 256]` buffer of 32 MiB. A donating loop adds only the small temporaries of
    /// the projections, so the bound is one quarter of one buffer (a copy of one buffer per
    /// append already moves the peak past it); the measured growth is printed for the record,
    /// and the cached rows are read back at the end and checked against the host derivation.
    #[test]
    fn two_hundred_decode_steps_donate_the_capacity_buffer() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Attn(host_w) = &view.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let buffer_bytes: u64 = 65_536 * 2 * 32 * 4;
        assert!(
            buffer_bytes >= 16_000_000,
            "one buffer must be at least 16 MB"
        );
        // A cap that no forced flush of the loop below can reach, so the bound cannot pass on a
        // copy per append (the method of the native test).
        let previous_cap = kernels.set_retire_cap(8 << 30).unwrap();
        // The host derivation of every cached row, kept for the readback at the end.
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut vals: Vec<Vec<f32>> = Vec::new();
        for n in 0..200u32 {
            let (key, val) = host_kv_rows(&mut cpu, &shape, host_w, n);
            keys.push(key);
            vals.push(val);
        }
        // Warm-up: the first step allocates both capacity buffers.
        let warm = upload(&mut kernels, &input(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let first = step(&mut ctx, &shape, rw, KvCache::with_step(65_536), 0, &warm).unwrap();
        kernels
            .evaluate(&[
                &first.out,
                first.cache.k.as_ref().unwrap(),
                first.cache.v.as_ref().unwrap(),
            ])
            .unwrap();
        let mut cache = first.cache;
        kernels.reset_peak_memory().unwrap();
        let start = kernels.memory_stats().unwrap().active;
        for n in 1..200u32 {
            let xr = upload(&mut kernels, &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            if n % 8 == 7 {
                kernels
                    .evaluate(&[
                        &next.out,
                        next.cache.k.as_ref().unwrap(),
                        next.cache.v.as_ref().unwrap(),
                    ])
                    .unwrap();
            }
            cache = next.cache;
        }
        assert_eq!(cache.len, 200, "the loop appended 199 more rows");
        assert_eq!(cache.cap, 65_536, "no step of the loop grew the buffers");
        let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
        println!(
            "decode steps: peak growth {growth} bytes over a {buffer_bytes}-byte buffer, bound {}",
            buffer_bytes / 4
        );
        assert!(
            growth < buffer_bytes / 4,
            "the decode step peak grew by {growth} bytes; a donating loop stays below one \
             quarter of one buffer ({})",
            buffer_bytes / 4
        );
        // Read the whole valid prefix of both buffers back and check it against the host.
        let (mut want_k, mut want_v) = (Vec::new(), Vec::new());
        for g in 0..hkv {
            for pos in 0..200 {
                want_k.extend_from_slice(&keys[pos][g * hd..(g + 1) * hd]);
                want_v.extend_from_slice(&vals[pos][g * hd..(g + 1) * hd]);
            }
        }
        kernels
            .evaluate(&[cache.k.as_ref().unwrap(), cache.v.as_ref().unwrap()])
            .unwrap();
        let got_k = values(&mut kernels, cache.k.as_ref().unwrap()).unwrap();
        let got_v = values(&mut kernels, cache.v.as_ref().unwrap()).unwrap();
        assert_parity(
            &prefix_rows(&got_k, hkv, cache.cap, 200, hd),
            &want_k,
            "cached keys after 200 steps",
        );
        assert_parity(
            &prefix_rows(&got_v, hkv, cache.cap, 200, hd),
            &want_v,
            "cached values after 200 steps",
        );
        kernels.set_retire_cap(previous_cap).unwrap();
    }

    /// Review B finding 6: the growth branch of `append` (`slice`, `zero_rows`, `concatenate`)
    /// adds no synchronization either. Three steps over the quantum 2 — the first allocates the
    /// buffers, the second appends in place, the third grows 2 to 4 — each build their graph
    /// lazily, and one `evaluate` of the final state synchronizes once.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_growth_step_builds_its_lazy_graph_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let inputs: Vec<_> = (0..3u32).map(|n| upload(&mut kernels, &input(n))).collect();
        let mut cache = KvCache::with_step(2);
        let before = kernels.qualification_sync_count();
        for (n, xr) in inputs.iter().enumerate() {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
            };
            let next = step(&mut ctx, &shape, rw, cache, n as u32, xr).unwrap();
            assert_eq!(
                kernels.qualification_sync_count(),
                before,
                "step {n} (the third grows the buffers) synchronized"
            );
            cache = next.cache;
        }
        assert_eq!(cache.cap, 4, "the third step grew the capacity 2 to 4");
        kernels
            .evaluate(&[cache.k.as_ref().unwrap(), cache.v.as_ref().unwrap()])
            .unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "evaluate adds one"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn step_builds_lazy_outputs_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let xr = upload(&mut kernels, &input(0));
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let next = step(&mut ctx, &shape, rw, KvCache::empty(), 0, &xr).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "step synchronized"
        );
        let (k, v) = (
            next.cache.k.as_ref().unwrap(),
            next.cache.v.as_ref().unwrap(),
        );
        kernels.evaluate(&[&next.out, k, v]).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "evaluate adds one"
        );
    }
}
