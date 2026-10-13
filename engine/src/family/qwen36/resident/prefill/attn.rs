//! Batched dense attention over a chunk (packet Q36-E4; decision 0044 item 11).
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
//! The cache keeps the decode layout — capacity buffers `[1, KVH, cap, D]` written in place by
//! `kv_append` (packet Q36-KV-INTEGRATE) — so a chunk and a decode step share it. The cache is
//! F32 in every activation dtype: the new keys and values convert to F32 before the append
//! (packet Q36-BF16-b, as the decode step of `attn.rs`), and the block output converts to the
//! activation dtype at the o-projection.

use super::super::attn::{AttnStep, KvCache, append};
use super::super::ops::{Ctx, cast_to, dim, to_act};
use super::super::weights::RAttn;
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    kernels::{BinaryOp, RopeSpec, UnaryOp},
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
    cache.check_buffers(kv_heads, width)?;
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
    // 1. Query and gate, interleaved per head: head h is row[h * 2D..(h + 1) * 2D] of each row.
    let qp = m.qmm(x, &w.q)?;
    let qp = m.reshape(&qp, &[rows, heads, pair])?;
    let q_raw = m.slice(&qp, &[0, 0, 0], &[rows, heads, width])?;
    let gate = m.slice(&qp, &[0, 0, width], &[rows, heads, pair])?;
    // 2. Query norm with the weight as stored, time on axis -2, then RoPE from position `pos`.
    let q = m.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
    let q = m.transpose(&q, &[1, 0, 2])?;
    let q = m.reshape(&q, &[1, heads, rows, width])?;
    let q = m.rope_scaled(&q, &rope)?;
    // 3. New keys: norm and RoPE per key-value head.
    let kp = m.qmm(x, &w.k)?;
    let kp = m.reshape(&kp, &[rows, kv_heads, width])?;
    let k_new = m.rms_norm(&kp, &w.k_norm, shape.eps)?;
    let k_new = m.transpose(&k_new, &[1, 0, 2])?;
    let k_new = m.reshape(&k_new, &[1, kv_heads, rows, width])?;
    let k_new = m.rope_scaled(&k_new, &rope)?;
    // 4. New values.
    let vp = m.qmm(x, &w.v)?;
    let vp = m.reshape(&vp, &[rows, kv_heads, width])?;
    let v_new = m.transpose(&vp, &[1, 0, 2])?;
    let v_new = m.reshape(&v_new, &[1, kv_heads, rows, width])?;
    // The cache stays F32 whatever the activation dtype (packet Q36-BF16-b, as the decode step
    // of `attn.rs`; a BF16 cache is a parked lever of decision 0046 item 5). The cast runs before
    // the append so that a cache which starts with this chunk is F32 too, not only one that
    // appends to F32 keys and values.
    let (k_new, v_new) = (
        cast_to(ctx, &k_new, Dtype::F32)?,
        cast_to(ctx, &v_new, Dtype::F32)?,
    );
    // 5. Append the rows into the capacity buffer (a growth builds the larger buffer first).
    let cache = append(ctx, kv_heads, width, cache, rows, &k_new, &v_new)?;
    let (k_buf, v_buf) = match (&cache.k, &cache.v) {
        (Some(k), Some(v)) => (k, v),
        _ => return Err(mismatch()),
    };
    // 6. Causal attention over the valid prefix, as a copy-free view of the capacity buffers;
    // head h reads KV head h / (QH / KVH).
    let scale = 1.0 / (width as f32).sqrt();
    let att =
        ctx.kernels
            .grouped_attention_prefix(&q, k_buf, v_buf, dim(next_len)?, scale, true)?;
    // 7. Back to rows, sigmoid output gate, and output projection. The attention output is F32
    // over the F32 cache; the projection takes the activation dtype (packet Q36-BF16-b).
    let att = ctx.kernels.transpose(&att, &[0, 2, 1, 3])?;
    let att = ctx.kernels.reshape(&att, &[rows, heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    let o = ctx.kernels.reshape(&o, &[rows, q_width])?;
    let o = to_act(ctx, &o)?;
    let out = ctx.kernels.qmm(&o, &w.o)?;
    Ok(AttnStep { out, cache })
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

    /// Runs the host attention of layer 1 over positions `0..count`.
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
            outputs
                .push(host::attn_step(&mut cpu, &shape, host_w, &mut host_cache, n, &x).unwrap());
            let kp = cpu.linear(host_w.k, &x).unwrap();
            let mut key = Vec::new();
            for g in 0..hkv {
                let mut head =
                    rms_plain(&kp[g * hd..(g + 1) * hd], host_w.k_norm, shape.eps).unwrap();
                rope_half(&mut head, shape.attn.rotary_dim, n, shape.attn.theta).unwrap();
                key.extend(head);
            }
            keys.push(key);
            vals.push(cpu.linear(host_w.v, &x).unwrap());
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

    /// The chunk step exactly as it was before packet Q36-KV-INTEGRATE (test-only): the cache is
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
    /// chunk (review B finding 4): round_up(warm + count, 4).
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
            };
            cache = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap().cache;
        }
        let x = rows(&mut kernels, warm, count);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let next = chunk(&mut ctx, &shape, rw, cache, warm, &x).unwrap();
        let total = warm as usize + count as usize;
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        assert_eq!(next.out.shape(), [count, 64], "output shape");
        assert_eq!(next.cache.len, total, "cache length");
        assert_eq!(
            next.cache.cap, want_cap,
            "the exact capacity after the chunk (review B finding 4)"
        );
        assert!(
            want_cap >= total,
            "the capacity {want_cap} must hold the valid rows {total}"
        );
        assert_eq!(
            next.cache.k.as_ref().unwrap().shape(),
            [1, hkv as u32, next.cache.cap as u32, hd as u32],
            "key shape"
        );
        assert_eq!(
            next.cache.byte_len(),
            next.cache.cap as u64 * growth_bytes(&shape).unwrap(),
            "stored bytes follow the capacity"
        );
        let want_out: Vec<f32> = host.outputs[warm as usize..].concat();
        assert_parity(
            &values(&mut kernels, &next.out).unwrap(),
            &want_out,
            "chunk output",
        );
        assert_parity(
            &prefix_rows(
                &values(&mut kernels, next.cache.k.as_ref().unwrap()).unwrap(),
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
                &values(&mut kernels, next.cache.v.as_ref().unwrap()).unwrap(),
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

    /// Packet Q36-KV-INTEGRATE 6(b): a chunk of 5 rows that crosses a growth (the buffers hold 2
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
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            let old = concat_step(&mut ctx, &shape, rw, &concat, n, &xr).unwrap();
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
        };
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
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            let old = concat_step(&mut ctx, &shape, rw, &concat, n, &xr).unwrap();
            let got = values(&mut kernels, &next.out).unwrap();
            assert_eq!(
                got,
                values(&mut kernels, &old.0).unwrap(),
                "decode step {n} after the chunk differs from the concatenate formulation"
            );
            assert_parity(
                &got,
                &host.outputs[n as usize],
                &format!("decode step {n} against the host"),
            );
            cache = next.cache;
            concat = old.1;
        }
        assert_eq!(cache.len, 12, "final length");
        assert_eq!(cache.cap, 12, "the decode steps crossed the second growth");
        // The caches agree bit for bit on the valid prefix.
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        for (name, new_tensor, old_tensor) in [
            ("keys", &cache.k, &concat.k),
            ("values", &cache.v, &concat.v),
        ] {
            let (new_tensor, old_tensor) = (new_tensor.as_ref(), old_tensor.as_ref());
            let got = values(&mut kernels, new_tensor.unwrap()).unwrap();
            let want = values(&mut kernels, old_tensor.unwrap()).unwrap();
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
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            let old = concat_step(&mut ctx, &shape, rw, &concat, n, &xr).unwrap();
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
        };
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
        };
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
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            let old = concat_step(&mut ctx, &shape, rw, &concat, n, &xr).unwrap();
            let got = values(&mut kernels, &next.out).unwrap();
            assert_eq!(
                got,
                values(&mut kernels, &old.0).unwrap(),
                "decode step {n} after the chunks differs from the concatenate formulation"
            );
            assert_parity(
                &got,
                &host.outputs[n as usize],
                &format!("decode step {n} against the host"),
            );
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
            k: Some(one_head(&mut kernels)),
            v: Some(one_head(&mut kernels)),
            len: 1,
            cap: 1,
            step: 4,
        };
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
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
        };
        let next = chunk(&mut ctx, &shape, rw, KvCache::empty(), 0, &x).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "chunk synchronized"
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
