//! Resident dense attention of the Q36 family (packet Q36-C1; decision 0046).
//!
//! The graph follows the dense branch of `flash_next/resident/qsa.rs` and the rules of
//! `qwen36/host.rs::attn_step`: the query projection holds, per head, the query rows and then the
//! gate rows; the query and key take a weighted RMS norm with the weight as stored (Flash-Next adds
//! 1); RoPE rotates the first `rotary_dim` dimensions; the head output is multiplied by
//! `sigmoid(gate)` before the output projection. There is no indexer and no block selection.
//!
//! The cache is F32 `[1, KVH, T, D]` for keys and for values. A step appends one position with
//! `concatenate` on the time axis. A capacity buffer written with `slice_update` is not used: the
//! resident `slice_update` returns a new array, so every step would copy the whole capacity, and
//! reading the prefix needs a `slice`, which copies again (it ends with `mlx_contiguous`). One
//! `concatenate` copies `T` positions once. The step builds a lazy graph only.

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

/// Keys and values of one dense attention layer. `None` while `len` is 0.
pub(super) struct KvCache {
    /// Keys after the norm and RoPE, `[1, KVH, len, D]`.
    pub(super) k: Option<ExecutionTensor>,
    /// Values, `[1, KVH, len, D]`.
    pub(super) v: Option<ExecutionTensor>,
    /// Cached positions.
    pub(super) len: usize,
}

impl KvCache {
    /// A cache with no position.
    pub(super) fn empty() -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
        }
    }

    /// Stored bytes of keys and values, from the tensors' own sizes.
    pub(super) fn byte_len(&self) -> u64 {
        [&self.k, &self.v]
            .into_iter()
            .flatten()
            .map(ExecutionTensor::byte_len)
            .fold(0, u64::saturating_add)
    }
}

/// Bytes that one more position adds to the cache of one layer: keys and values, `KVH * D` F32 each.
/// The real model adds 2 * 2 * 256 * 4 = 4,096 bytes per layer and 40,960 bytes over 10 layers.
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
    /// Layer output `[1, H]`.
    pub(super) out: ExecutionTensor,
    /// The cache with one more position. The caller replaces the old cache with it.
    pub(super) cache: KvCache,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident attention dimension overflows".into())
}

fn mismatch() -> EngError {
    EngError::BadRequest("resident attention cache does not match its shape".into())
}

/// Runs one dense attention decode step at `pos` on `x` (`H` values, any shape). It builds a lazy
/// graph only: no readback and no evaluate. `cache` is never changed. A cache that holds a number of
/// positions other than `pos`, or tensors of the wrong shape, returns `BadRequest` before the
/// first native call.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: &KvCache,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<AttnStep, EngError> {
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    if usize::try_from(pos).ok() != Some(cache.len) {
        return Err(EngError::BadRequest(format!(
            "attention cache holds {} positions, step is at {pos}",
            cache.len
        )));
    }
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
    // 5. Append on the time axis, or start the cache.
    let (k_all, v_all) = match past {
        None => (k_new, v_new),
        Some((k, v)) => (
            ctx.kernels.concatenate(&[k, &k_new], 2)?,
            ctx.kernels.concatenate(&[v, &v_new], 2)?,
        ),
    };
    // 6. Attention over every position. Head h reads key-value head h / (QH / KVH).
    let scale = 1.0 / (width as f32).sqrt();
    let att = ctx
        .kernels
        .grouped_attention(&q, &k_all, &v_all, scale, false)?;
    // 7. Sigmoid output gate and output projection.
    let att = ctx.kernels.reshape(&att, &[heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    // The attention output is F32 over the F32 cache; the projection takes the activation dtype.
    let o = to_act(ctx, &o)?;
    let out = linear(ctx, &w.o, &o)?;
    Ok(AttnStep {
        out,
        cache: KvCache {
            k: Some(k_all),
            v: Some(v_all),
            len: next_len,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::ops::{assert_parity, f32_tensor, values};
    use super::super::weights::{RMixer, ResidentWeights};
    use super::*;
    use crate::family::flash_next::math::rope_half;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, KvCache as HostCache, MixerW, rms_plain};
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

    #[test]
    fn seven_steps_match_the_host_block_and_the_cache_holds_the_rotated_keys() {
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
        let mut cache = KvCache::empty();
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        // Keys and values by definition, one entry per position: the weighted RMS norm of the key
        // projection, then RoPE; the value projection as is.
        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut vals: Vec<Vec<f32>> = Vec::new();
        for n in 0..7u32 {
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
            let next = step(&mut ctx, &shape, rw, &cache, n, &xr).unwrap();
            assert_eq!(next.out.shape(), [1, 64], "step {n} output shape");
            assert_eq!(next.cache.len, n as usize + 1, "step {n} cache length");
            let t = n + 1;
            assert_eq!(
                next.cache.k.as_ref().unwrap().shape(),
                [1, hkv as u32, t, hd as u32],
                "step {n} key shape"
            );
            let out = values(&mut kernels, &next.out).unwrap();
            assert_parity(&out, &want, &format!("step {n} output"));
            cache = next.cache;
            assert_eq!(
                cache.byte_len(),
                u64::from(t) * growth_bytes(&shape).unwrap(),
                "step {n} stored bytes follow the growth rule"
            );
        }
        // Layout [1, KVH, T, D]: element (g, t, d) sits at (g * T + t) * D + d.
        let t = keys.len();
        let got_k = values(&mut kernels, cache.k.as_ref().unwrap()).unwrap();
        let got_v = values(&mut kernels, cache.v.as_ref().unwrap()).unwrap();
        let (mut want_k, mut want_v) = (Vec::new(), Vec::new());
        for g in 0..hkv {
            for pos in 0..t {
                want_k.extend_from_slice(&keys[pos][g * hd..(g + 1) * hd]);
                want_v.extend_from_slice(&vals[pos][g * hd..(g + 1) * hd]);
            }
        }
        assert_parity(&got_k, &want_k, "cached keys");
        assert_parity(&got_v, &want_v, "cached values");
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
    fn a_cache_of_the_wrong_length_or_shape_is_rejected_before_any_native_call() {
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
        let wrong_len = step(&mut ctx, &shape, rw, &KvCache::empty(), 3, &xr);
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
        };
        let wrong_shape = step(&mut ctx, &shape, rw, &bad, 1, &xr);
        assert!(
            matches!(&wrong_shape, Err(EngError::BadRequest(m)) if m.contains("does not match its shape")),
            "{:?}",
            wrong_shape.as_ref().err()
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
        let next = step(&mut ctx, &shape, rw, &KvCache::empty(), 0, &xr).unwrap();
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
