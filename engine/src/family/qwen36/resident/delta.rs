//! Resident DeltaNet step of the Q36 family.
//!
//! The graph follows `flash_next/resident/deltanet.rs` with the rules of
//! `qwen36/host.rs::delta_step`: the output gate is `silu(z)` (Flash-Next: `sigmoid(z)`), and the
//! query and key norm adds `key_dim * 1e-6` to the sum of squares (Flash-Next: `1e-6`). The delta
//! rule itself runs in the gated delta kernel, which takes the key heads without repetition and
//! keeps the state as `[VH, VD, KD]`, the layout of the host reference.
//!
//! The step only builds lazy graph nodes. It never reads back or evaluates.

use super::ops::{Ctx, cast_to, dim, linear, to_act};
use super::weights::RDelta;
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    kernels::{BinaryOp, MlxKernels, ReduceOp, UnaryOp},
    tensor::{Dtype, ExecutionTensor},
};

/// Output and replacement state of one resident DeltaNet step. Every tensor is lazy.
pub(super) struct DeltaStep {
    /// Layer output `[1, H]`.
    pub(super) out: ExecutionTensor,
    /// New recurrent state `[VH, VD, KD]`.
    pub(super) rec: ExecutionTensor,
    /// New convolution ring `[CD, CK - 1]`, oldest tap first.
    pub(super) ring: ExecutionTensor,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident delta dimensions overflow".into())
}

/// Returns `sqrt(sum(x * x) + eps)` for each row of a `[rows, width]` tensor, as `[rows, 1]`.
fn l2_den(
    m: &mut MlxKernels,
    x: &ExecutionTensor,
    eps: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let sq = m.binary(BinaryOp::Mul, x, x)?;
    let sum = m.reduce(ReduceOp::Sum, &sq, 1, true)?;
    let sum = m.binary(BinaryOp::Add, &sum, eps)?;
    m.unary(UnaryOp::Sqrt, &sum)
}

/// Builds one DeltaNet decode step over `x` (`hidden` elements) and the old `rec` and `ring`.
///
/// It performs no synchronization. The caller replaces its state with the returned `rec` and `ring`.
/// A state shape error returns `BadRequest` before the first native call.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RDelta,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
    x: &ExecutionTensor,
) -> Result<DeltaStep, EngError> {
    check_state_shapes(shape, rec, ring)?;
    // The compiled decode closure: the `qwen36.compile` setting (or the test override of
    // `QWEN36_G2A_COMPILE`) runs this step through ONE compiled closure instead of the op sequence
    // below, in the activation dtype of the step. There is no fallback: a failure of the compiled
    // path is a native failure of the adapter.
    if super::compiled::step_requested(ctx.compile) {
        return super::compiled::delta_step(ctx, shape, w, rec, ring, x);
    }
    // Projections: the batched path (`batch::step_rows`) meets these four
    // weights in ONE qmm each over `[B, hidden]`; only the recurrence below
    // stays per slot (the compiled closure above stays the solo path only).
    let u = linear(ctx, &w.qkv, x)?;
    let z = linear(ctx, &w.z, x)?;
    let bb = linear(ctx, &w.b, x)?;
    let aa = linear(ctx, &w.a, x)?;
    let (o, rec_next, ring_next) = step_projected(ctx, shape, w, rec, ring, [&u, &z, &bb, &aa])?;
    let out = linear(ctx, &w.out, &o)?;
    Ok(DeltaStep {
        out,
        rec: rec_next,
        ring: ring_next,
    })
}

/// Rejects malformed state before a same-element-count reshape can normalize it.
pub(super) fn check_state_shapes(
    shape: &HostShape,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
) -> Result<(), EngError> {
    let d = &shape.delta;
    let kd = dim(d.key_dim)?;
    let vh = dim(d.value_heads)?;
    let vd = dim(d.value_dim)?;
    let cd = dim(d.conv_dim())?;
    let taps = dim(d.conv_kernel)?.checked_sub(1).ok_or_else(overflow)?;
    // A state of another shape but the same element count would reshape without an error, so
    // check the state shapes exactly.
    if rec.shape() != [vh, vd, kd] {
        return Err(EngError::BadRequest(
            "resident delta rec shape differs from [VH, VD, KD]".into(),
        ));
    }
    if ring.shape() != [cd, taps] {
        return Err(EngError::BadRequest(
            "resident delta ring shape differs from [CD, CK - 1]".into(),
        ));
    }
    Ok(())
}

/// Everything between the projections and the out-projection: the F32 promote
/// of `bb`, the causal convolution over each slot's own ring, the q/k/v split,
/// the decay/beta gates, the gated-delta recurrence and the silu(z) gate. Takes
/// the RAW projection outputs (each `[1, N]`); returns the gated output as ONE
/// flat `[1, VH * VD]` row BEFORE `w.out`, the new recurrence and the new ring -
/// so the batched path can stack all rows and project once.
pub(super) fn step_projected(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RDelta,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
    projections: [&ExecutionTensor; 4],
) -> Result<(ExecutionTensor, ExecutionTensor, ExecutionTensor), EngError> {
    check_state_shapes(shape, rec, ring)?;
    let [u, z, bb, aa] = projections;
    let d = &shape.delta;
    let kh = dim(d.key_heads)?;
    let kd = dim(d.key_dim)?;
    let vh = dim(d.value_heads)?;
    let vd = dim(d.value_dim)?;
    let output_width = vh.checked_mul(vd).ok_or_else(overflow)?;
    let cd = dim(d.conv_dim())?;
    let taps = dim(d.conv_kernel)?.checked_sub(1).ok_or_else(overflow)?;
    let kw = kh.checked_mul(kd).ok_or_else(overflow)?;
    let kw2 = kw.checked_mul(2).ok_or_else(overflow)?;
    // Under a BF16 activation the recurrence, its state, and the convolution ring stay F32. `u`,
    // `z`, and `aa` meet an F32 weight or state first and promote on their own. `bb` feeds a
    // sigmoid alone and the gated delta kernel takes F32 `beta`, so it converts here.
    let bb = cast_to(ctx, bb, Dtype::F32)?;
    let consts = ctx.consts;
    let m = &mut *ctx.kernels;
    let u = m.reshape(u, &[cd])?;
    // Causal depthwise convolution over the ring and the new row, then SiLU.
    let now = m.binary(BinaryOp::Mul, &w.conv_last, &u)?;
    let past = m.binary(BinaryOp::Mul, &w.conv_taps, ring)?;
    let past = m.reduce(ReduceOp::Sum, &past, 1, false)?;
    let val = m.binary(BinaryOp::Add, &now, &past)?;
    let y = m.silu(&val)?;
    // New ring: drop the oldest tap and append u.
    let column = m.reshape(&u, &[cd, 1])?;
    let ring_next = if taps > 1 {
        let kept = m.slice(ring, &[0, 1], &[cd, taps])?;
        m.concatenate(&[&kept, &column], 1)?
    } else if column.dtype() == Dtype::F32 {
        column
    } else {
        m.cast(&column, Dtype::F32)?
    };
    // Split y into q, k, v.
    let q = m.slice(&y, &[0], &[kw])?;
    let q = m.reshape(&q, &[kh, kd])?;
    let k = m.slice(&y, &[kw], &[kw2])?;
    let k = m.reshape(&k, &[kh, kd])?;
    let v = m.slice(&y, &[kw2], &[cd])?;
    let v = m.reshape(&v, &[1, 1, vh, vd])?;
    // rms_norm(x, None, 1e-6) times 1/D (query) or 1/sqrt(D) (key) equals x / sqrt(sum + D * 1e-6)
    // with an extra divisor sqrt(D) on the query. `qk_eps` holds D * 1e-6.
    let qd = l2_den(m, &q, &consts.qk_eps)?;
    let qd = m.binary(BinaryOp::Mul, &qd, &consts.sqrt_kd)?;
    let q = m.binary(BinaryOp::Div, &q, &qd)?;
    let kden = l2_den(m, &k, &consts.qk_eps)?;
    let k = m.binary(BinaryOp::Div, &k, &kden)?;
    let q = m.reshape(&q, &[1, 1, kh, kd])?;
    let k = m.reshape(&k, &[1, 1, kh, kd])?;
    // alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
    let pre = m.binary(BinaryOp::Add, aa, &w.dt_bias)?;
    let soft = m.binary(BinaryOp::LogAddExp, &pre, &consts.zero)?;
    let decay = m.binary(BinaryOp::Mul, &w.neg_exp_a_log, &soft)?;
    let alpha = m.unary(UnaryOp::Exp, &decay)?;
    let alpha = m.reshape(&alpha, &[1, 1, vh])?;
    let beta = m.unary(UnaryOp::Sigmoid, &bb)?;
    let beta = m.reshape(&beta, &[1, 1, vh])?;
    // The delta rule for one token. Value head h reads key head h / (VH / KH) inside the kernel.
    let state = m.reshape(rec, &[1, vh, vd, kd])?;
    let kernel = m.gated_delta(&q, &k, &v, &alpha, &beta, &state)?;
    let core = m.reshape(&kernel.y, &[vh, vd])?;
    let rec_next = m.reshape(&kernel.state, &[vh, vd, kd])?;
    // Output: silu(z) * rms_norm(core, norm). The norm weight applies as stored.
    let normed = m.rms_norm(&core, &w.norm, shape.eps)?;
    let z = m.reshape(z, &[vh, vd])?;
    let gate = m.silu(&z)?;
    let o = m.binary(BinaryOp::Mul, &normed, &gate)?;
    // The gated norm and the gate promote to F32; the out-projection takes the activation dtype.
    let o = to_act(ctx, &o)?;
    // One flat row `[1, VH * VD]`: the stacked tail (`linear_rows`) needs rank-2
    // rows whose column count is the weight input width. (`ctx.kernels`, not the
    // `m` alias: `m` must not borrow across `to_act`'s mutable `ctx`.)
    let o = ctx.kernels.reshape(&o, &[1, output_width])?;
    Ok((o, rec_next, ring_next))
}

#[cfg(test)]
mod tests {
    use super::super::ops::{assert_parity, f32_tensor, values, zeros};
    use super::super::weights::{RMixer, ResidentWeights};
    use super::*;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, DeltaState, MixerW};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY, TINY_WIDE};
    use crate::family::spec::FamilySpec;
    use crate::weights::StreamedCheckpoint;

    /// Input of step `n`: `x[i] = (((i + 5 n) % 11) - 5) / 160`, exact in `f32` up to rounding of the
    /// division. The small scale keeps the query and key sums of squares near the epsilon `D * 1e-6`,
    /// so that the epsilon rule changes the output by several percent (checked on the host
    /// reference: 9 percent of the largest output at step 0).
    fn input(n: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((i + 5 * n) % 11) as f32 - 5.0) / 160.0)
            .collect()
    }

    /// The host load for the oracle, the resident import, and the checkpoint reader.
    fn setup(spec: &FamilySpec) -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(spec);
        let config = tiny::config(spec);
        let oracle = HostWeights::load(&ckpt, spec, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let resident = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, spec, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, resident, kernels)
    }

    fn zero_state(
        kernels: &mut MlxKernels,
        shape: &HostShape,
    ) -> (ExecutionTensor, ExecutionTensor) {
        let d = &shape.delta;
        let rec = zeros(vec![
            d.value_heads as u32,
            d.value_dim as u32,
            d.key_dim as u32,
        ])
        .unwrap();
        let ring = zeros(vec![d.conv_dim() as u32, d.conv_kernel as u32 - 1]).unwrap();
        let mut got = kernels.import_many(&[&rec, &ring]).unwrap().into_iter();
        (got.next().unwrap(), got.next().unwrap())
    }

    /// Runs five steps on the layer-0 DeltaNet of `spec` and compares the output, the recurrent state,
    /// and the ring with the host reference after every step.
    fn five_steps_match_the_host(spec: &FamilySpec) {
        let (ckpt, oracle, w, mut kernels) = setup(spec);
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Delta(host_w) = &view.layers[0].mixer else {
            panic!("layer 0 is DeltaNet");
        };
        let RMixer::Delta(rw) = &w.layers[0].mixer else {
            panic!("resident layer 0 is DeltaNet");
        };
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut host_state = DeltaState::new(&shape.delta);
        let (mut rec, mut ring) = zero_state(&mut kernels, &shape);
        let mut projected = Vec::new();
        for n in 0..5u32 {
            let x = input(n);
            let want = host::delta_step(&mut cpu, &shape, host_w, &mut host_state, &x).unwrap();
            projected.push(cpu.linear(host_w.qkv, &x).unwrap());
            let xr = kernels
                .import_tensor(&f32_tensor(vec![1, 64], &x).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, &rec, &ring, &xr).unwrap();
            assert_eq!(next.out.shape(), [1, 64], "step {n} output shape");
            assert_eq!(next.rec.shape(), rec.shape(), "step {n} rec shape");
            assert_eq!(next.ring.shape(), ring.shape(), "step {n} ring shape");
            let out = values(&mut kernels, &next.out).unwrap();
            assert_parity(&out, &want, &format!("step {n} output"));
            let got_rec = values(&mut kernels, &next.rec).unwrap();
            assert_parity(&got_rec, host_state.rec(), &format!("step {n} rec"));
            rec = next.rec;
            ring = next.ring;
        }
        // After five steps the ring holds the qkv projections of steps 2, 3, and 4, oldest first.
        let cd = shape.delta.conv_dim();
        let want_ring: Vec<f32> = (0..cd)
            .flat_map(|c| (2..5).map(|t| projected[t][c]).collect::<Vec<_>>())
            .collect();
        assert_parity(&values(&mut kernels, &ring).unwrap(), &want_ring, "ring");
    }

    #[test]
    fn five_steps_match_the_host_block_state_and_ring() {
        let _serial = crate::native_test_lock();
        five_steps_match_the_host(&TINY);
    }

    #[test]
    fn value_heads_follow_their_key_head_with_two_key_heads_and_ratio_two() {
        // TINY_WIDE has 2 key heads and 4 value heads: value heads 0 and 1 read key head 0, and
        // value heads 2 and 3 read key head 1. Tiling the key heads instead would read 0, 1, 0, 1.
        let _serial = crate::native_test_lock();
        five_steps_match_the_host(&TINY_WIDE);
    }

    #[test]
    fn step_rejects_state_of_the_wrong_shape() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup(&TINY);
        let shape = oracle.shape().clone();
        let RMixer::Delta(rw) = &w.layers[0].mixer else {
            panic!("resident layer 0 is DeltaNet");
        };
        let (rec, ring) = zero_state(&mut kernels, &shape);
        // [1, 32, 32] has one head instead of two. [1, 3] has one channel instead of the full width.
        let flat_rec = kernels
            .import_tensor(&zeros(vec![1, 32, 32]).unwrap())
            .unwrap();
        let short_ring = kernels.import_tensor(&zeros(vec![1, 3]).unwrap()).unwrap();
        let x = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(0)).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let bad_rec = step(&mut ctx, &shape, rw, &flat_rec, &ring, &x);
        assert!(
            matches!(&bad_rec, Err(EngError::BadRequest(m)) if m.contains("delta rec shape")),
            "{:?}",
            bad_rec.as_ref().err()
        );
        let bad_ring = step(&mut ctx, &shape, rw, &rec, &short_ring, &x);
        assert!(
            matches!(&bad_ring, Err(EngError::BadRequest(m)) if m.contains("delta ring shape")),
            "{:?}",
            bad_ring.as_ref().err()
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn step_builds_lazy_outputs_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup(&TINY);
        let shape = oracle.shape().clone();
        let RMixer::Delta(rw) = &w.layers[0].mixer else {
            panic!("resident layer 0 is DeltaNet");
        };
        let (rec, ring) = zero_state(&mut kernels, &shape);
        let x = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &input(0)).unwrap())
            .unwrap();
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
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
