//! Resident DeltaNet step (S1-C plan section 2.2). It mirrors the host step in `flash_next/deltanet.rs`.
//!
//! The step only builds lazy graph nodes. It never reads back or evaluates (decision 0039 item 2).

use super::super::deltanet::DeltaShape;
use super::ops::{Ctx, dim, linear};
use super::weights::RDelta;
use crate::{
    EngError,
    kernels::{BinaryOp, MlxKernels, ReduceOp, UnaryOp},
    tensor::ExecutionTensor,
};

/// Output and replacement state of one resident DeltaNet step. Every tensor is lazy.
pub(super) struct DeltaStep {
    /// Layer output `[1, H]`.
    pub(super) out: ExecutionTensor,
    /// New recurrent state `[VH, KD, VD]`, the host layout.
    pub(super) rec: ExecutionTensor,
    /// New convolution ring `[CD, CK - 1]`, the host layout.
    pub(super) ring: ExecutionTensor,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident delta dimensions overflow".into())
}

/// Returns `sqrt(sum(x * x) + 1e-6)` for each row of a `[rows, width]` tensor, as `[rows, 1]`.
fn l2_den(
    m: &mut MlxKernels,
    x: &ExecutionTensor,
    eps6: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let sq = m.binary(BinaryOp::Mul, x, x)?;
    let sum = m.reduce(ReduceOp::Sum, &sq, 1, true)?;
    let sum = m.binary(BinaryOp::Add, &sum, eps6)?;
    m.unary(UnaryOp::Sqrt, &sum)
}

/// Builds one DeltaNet decode step over `x` (`hidden` elements) and the old `rec` and `ring`.
///
/// It performs no synchronization. The caller replaces its state with the returned `rec` and `ring`.
/// A state shape error returns `BadRequest` before the first native call.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &DeltaShape,
    w: &RDelta,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
    x: &ExecutionTensor,
) -> Result<DeltaStep, EngError> {
    let kh = dim(shape.key_heads())?;
    let kd = dim(shape.key_dim())?;
    let vh = dim(shape.value_heads())?;
    let vd = dim(shape.value_dim())?;
    let cd = dim(shape.conv_dim())?;
    let taps = dim(shape.conv_kernel())?
        .checked_sub(1)
        .ok_or_else(overflow)?;
    let kw = kh.checked_mul(kd).ok_or_else(overflow)?;
    let kw2 = kw.checked_mul(2).ok_or_else(overflow)?;
    let rep = vh.checked_div(kh).ok_or_else(overflow)?;
    // deltanet.rs:188-189 checks the state lengths. A wrong rank-3 shape could broadcast, so check exactly.
    if rec.shape() != [vh, kd, vd] {
        return Err(EngError::BadRequest(
            "resident delta rec shape differs from [VH, KD, VD]".into(),
        ));
    }
    if ring.shape() != [cd, taps] {
        return Err(EngError::BadRequest(
            "resident delta ring shape differs from [CD, CK - 1]".into(),
        ));
    }
    // Projections, deltanet.rs:191-198.
    let u = linear(ctx, &w.qkv, x)?;
    let z = linear(ctx, &w.z, x)?;
    let bb = linear(ctx, &w.b, x)?;
    let aa = linear(ctx, &w.a, x)?;
    let consts = ctx.consts;
    let m = &mut *ctx.kernels;
    let u = m.reshape(&u, &[cd])?;
    // Causal convolution and SiLU, deltanet.rs:202-207 and math.rs:11-13.
    let now = m.binary(BinaryOp::Mul, &w.conv_last, &u)?;
    let past = m.binary(BinaryOp::Mul, &w.conv_taps, ring)?;
    let past = m.reduce(ReduceOp::Sum, &past, 1, false)?;
    let val = m.binary(BinaryOp::Add, &now, &past)?;
    let y = m.silu(&val)?;
    // New ring: drop the oldest tap and append u, deltanet.rs:208-211.
    let column = m.reshape(&u, &[cd, 1])?;
    let ring_next = if taps > 1 {
        let kept = m.slice(ring, &[0, 1], &[cd, taps])?;
        m.concatenate(&[&kept, &column], 1)?
    } else {
        column
    };
    // Split y into q, k, v, deltanet.rs:216-221 and 243.
    let q = m.slice(&y, &[0], &[kw])?;
    let q = m.reshape(&q, &[kh, kd])?;
    let k = m.slice(&y, &[kw], &[kw2])?;
    let k = m.reshape(&k, &[kh, kd])?;
    let v = m.slice(&y, &[kw2], &[cd])?;
    let v = m.reshape(&v, &[vh, 1, vd])?;
    // L2 normalization with eps inside the root; q also divides by sqrt(KD), deltanet.rs:218-223.
    let qd = l2_den(m, &q, &consts.eps6)?;
    let qd = m.binary(BinaryOp::Mul, &qd, &consts.sqrt_kd)?;
    let q = m.binary(BinaryOp::Div, &q, &qd)?;
    let kden = l2_den(m, &k, &consts.eps6)?;
    let k = m.binary(BinaryOp::Div, &k, &kden)?;
    // Value head h uses key head h / rep, deltanet.rs:226 and 230.
    let q3 = m.repeat(&q, rep, 0)?;
    let k3 = m.repeat(&k, rep, 0)?;
    // alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b), deltanet.rs:231-232, math.rs:6-8 and 16-18.
    let pre = m.binary(BinaryOp::Add, &aa, &w.dt_bias)?;
    let soft = m.binary(BinaryOp::LogAddExp, &pre, &consts.zero)?;
    let decay = m.binary(BinaryOp::Mul, &w.neg_exp_a_log, &soft)?;
    let alpha = m.unary(UnaryOp::Exp, &decay)?;
    let alpha = m.reshape(&alpha, &[vh, 1, 1])?;
    let beta = m.unary(UnaryOp::Sigmoid, &bb)?;
    let beta = m.reshape(&beta, &[vh, 1, 1])?;
    // Decay the state, deltanet.rs:233-236.
    let s1 = m.binary(BinaryOp::Mul, rec, &alpha)?;
    // Delta-rule error, deltanet.rs:237-244.
    let k_row = m.reshape(&k3, &[vh, 1, kd])?;
    let prev = m.batched_mm(&k_row, &s1)?;
    let err = m.binary(BinaryOp::Sub, &v, &prev)?;
    let delta = m.binary(BinaryOp::Mul, &err, &beta)?;
    // Rank-one update, deltanet.rs:245-249.
    let k_col = m.reshape(&k3, &[vh, kd, 1])?;
    let outer = m.binary(BinaryOp::Mul, &k_col, &delta)?;
    let rec_next = m.binary(BinaryOp::Add, &s1, &outer)?;
    // Read out with the updated state, deltanet.rs:250-257.
    let q_row = m.reshape(&q3, &[vh, 1, kd])?;
    let core = m.batched_mm(&q_row, &rec_next)?;
    let core = m.reshape(&core, &[vh, vd])?;
    // RMS norm with the plain weight, then the sigmoid(z) gate, deltanet.rs:258-261.
    let normed = m.rms_norm(&core, &w.norm, shape.eps())?;
    let z = m.reshape(&z, &[vh, vd])?;
    let gate = m.unary(UnaryOp::Sigmoid, &z)?;
    let o = m.binary(BinaryOp::Mul, &normed, &gate)?;
    // Output projection, deltanet.rs:264.
    let out = linear(ctx, &w.out, &o)?;
    Ok(DeltaStep {
        out,
        rec: rec_next,
        ring: ring_next,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::deltanet::{self, DeltaState};
    use super::super::super::load::FlashWeights;
    use super::super::super::mlx::MlxProjector;
    use super::super::super::step::AttnWeights;
    use super::super::fixtures;
    use super::super::ops::{f32_tensor, values, zeros};
    use super::super::weights::{RAttn, ResidentWeights};
    use super::*;

    /// Asserts cosine >= 0.99999 and max error / max host magnitude <= 1e-4 (the metric of `mlx.rs:167-198`).
    fn assert_parity(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let (mut dot, mut gg, mut ww, mut err, mut scale) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            let (g, w) = (f64::from(*g), f64::from(*w));
            dot += g * w;
            gg += g * g;
            ww += w * w;
            err = err.max((g - w).abs());
            scale = scale.max(w.abs());
        }
        let cos = if gg == 0.0 && ww == 0.0 {
            1.0
        } else {
            dot / (gg.sqrt() * ww.sqrt())
        };
        let rel = if scale == 0.0 { err } else { err / scale };
        assert!(cos >= 0.99999, "{what}: cos {cos}");
        assert!(rel <= 1e-4, "{what}: rel {rel}");
    }

    /// `x[i] = ((i % 7) - 3) / 8` for `i` in `0..64` (H = 64).
    fn ramp7() -> Vec<f32> {
        (0..64).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect()
    }

    /// `x[i] = ((i % 5) - 2) / 16` for `i` in `0..64`.
    fn ramp5() -> Vec<f32> {
        (0..64).map(|i| ((i % 5) as f32 - 2.0) * 0.0625).collect()
    }

    /// Imports `x` as `[1, 64]`.
    fn input(kernels: &mut MlxKernels, x: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(vec![1, 64], x).unwrap())
            .unwrap()
    }

    /// Imports a zero `rec` `[2, 32, 32]` and `ring` `[128, 3]`: the fixture's VH, KD, VD, CD, and CK - 1.
    fn zero_state(kernels: &mut MlxKernels) -> (ExecutionTensor, ExecutionTensor) {
        let rec = zeros(vec![2, 32, 32]).unwrap();
        let ring = zeros(vec![128, 3]).unwrap();
        let mut got = kernels.import_many(&[&rec, &ring]).unwrap().into_iter();
        (got.next().unwrap(), got.next().unwrap())
    }

    #[test]
    fn two_steps_match_the_host_block_and_state() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = weights.shape.delta;
        let view = flash.view();
        let AttnWeights::Delta(host_w) = &view.layers[0].attn else {
            panic!("host layer 0 is not DeltaNet");
        };
        let RAttn::Delta(rw) = &weights.layers[0].attn else {
            panic!("resident layer 0 is not DeltaNet");
        };
        let mut projector = MlxProjector::new(&ckpt).unwrap();
        let mut host_state = DeltaState::new(&shape);
        let (mut rec, mut ring) = zero_state(&mut kernels);
        for (n, x) in [ramp7(), ramp5()].iter().enumerate() {
            let want = deltanet::step(&mut projector, &shape, host_w, &mut host_state, x).unwrap();
            let xr = input(&mut kernels, x);
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let next = step(&mut ctx, &shape, rw, &rec, &ring, &xr).unwrap();
            assert_eq!(next.out.shape(), &[1, 64][..], "step {n} output shape");
            assert_eq!(next.rec.shape(), &[2, 32, 32][..], "step {n} rec shape");
            assert_eq!(next.ring.shape(), &[128, 3][..], "step {n} ring shape");
            let out = values(&mut kernels, &next.out).unwrap();
            assert_parity(&out, &want, &format!("step {n} output"));
            let got_rec = values(&mut kernels, &next.rec).unwrap();
            assert_parity(&got_rec, host_state.rec(), &format!("step {n} rec"));
            let got_ring = values(&mut kernels, &next.ring).unwrap();
            assert_parity(&got_ring, host_state.ring(), &format!("step {n} ring"));
            rec = next.rec;
            ring = next.ring;
        }
    }

    #[test]
    fn step_rejects_state_of_the_wrong_shape() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = weights.shape.delta;
        let RAttn::Delta(rw) = &weights.layers[0].attn else {
            panic!("resident layer 0 is not DeltaNet");
        };
        let (rec, ring) = zero_state(&mut kernels);
        // [1, 32, 32] would broadcast against alpha [2, 1, 1]; [1, 3] would broadcast against the taps.
        let flat_rec = kernels
            .import_tensor(&zeros(vec![1, 32, 32]).unwrap())
            .unwrap();
        let short_ring = kernels.import_tensor(&zeros(vec![1, 3]).unwrap()).unwrap();
        let x = input(&mut kernels, &ramp7());
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let bad_rec = step(&mut ctx, &shape, rw, &flat_rec, &ring, &x);
        assert!(
            matches!(&bad_rec, Err(EngError::BadRequest(m)) if m.contains("delta rec shape")),
            "rec [1, 32, 32]: {:?}",
            bad_rec.as_ref().err()
        );
        let bad_ring = step(&mut ctx, &shape, rw, &rec, &short_ring, &x);
        assert!(
            matches!(&bad_ring, Err(EngError::BadRequest(m)) if m.contains("delta ring shape")),
            "ring [1, 3]: {:?}",
            bad_ring.as_ref().err()
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn step_builds_lazy_outputs_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = weights.shape.delta;
        let RAttn::Delta(rw) = &weights.layers[0].attn else {
            panic!("resident layer 0 is not DeltaNet");
        };
        let (rec, ring) = zero_state(&mut kernels);
        let x = input(&mut kernels, &ramp7());
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let next = step(&mut ctx, &shape, rw, &rec, &ring, &x).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "step synchronized"
        );
        kernels
            .evaluate(&[&next.out, &next.rec, &next.ring])
            .unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "evaluate adds one"
        );
    }
}
