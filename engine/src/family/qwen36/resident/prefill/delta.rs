//! Batched DeltaNet chunk of the Q36 family (packet Q36-E3; decision 0044 item 2).
//!
//! The projections, the causal convolution, the L2 norms, and the gates run once over `[T, ...]`
//! tensors. The delta-rule recurrence then runs inside `gated_delta_recurrence`, which equals `T`
//! steps of `delta::step`. The rules are those of `delta.rs`: the output gate is `silu(z)`, and the
//! query and key norm adds `key_dim * 1e-6` to the sum of squares.
//!
//! The activation dtype (packet Q36-BF16-b): the projections over `x` run in the activation
//! dtype, while the recurrence, its state, and the convolution ring stay F32, as the decode step
//! of `delta.rs` keeps them. `bb` converts to F32 before its sigmoid because the gated delta
//! kernel takes an F32 `beta`; the query, the key, the value, and the decay meet an F32 constant
//! or the F32 state first and promote on their own. The block output converts back to the
//! activation dtype at the out-projection.
//!
//! `gated_delta_recurrence` is the one seam of this file. The normalized query and key, the value,
//! the write strength `beta`, the decay factor `g`, and the state go in; the readout of every token
//! and the new state come out. It is one call of the gated delta kernel adapted from mlx-lm 0.31.0
//! (decision 0050), which loops over the tokens inside the kernel. The recurrent state is
//! `[VH, VD, KD]`, the layout of the kernel and of the host reference.

use super::super::ops::{Ctx, dim, to_act};
use super::super::weights::RDelta;
use super::conv::causal_conv;
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    kernels::{BinaryOp, MlxKernels, ReduceOp, UnaryOp},
    tensor::{Dtype, ExecutionTensor},
};

/// Output and replacement state of one DeltaNet chunk. Every tensor is lazy.
pub(super) struct DeltaChunk {
    /// Layer output `[T, H]`.
    pub(super) out: ExecutionTensor,
    /// New recurrent state `[VH, VD, KD]`.
    pub(super) rec: ExecutionTensor,
    /// New convolution ring `[CD, CK - 1]`, oldest tap first.
    pub(super) ring: ExecutionTensor,
}

/// The data inputs of `gated_delta_recurrence`. All are lazy F32. See that function for the layouts.
pub(super) struct GatedInputs<'a> {
    /// Query after its L2 norm and scale, `[T, KH, KD]`.
    pub(super) q: &'a ExecutionTensor,
    /// Key after its L2 norm, `[T, KH, KD]`.
    pub(super) k: &'a ExecutionTensor,
    /// Value, `[T, VH, VD]`.
    pub(super) v: &'a ExecutionTensor,
    /// Write strength, `[T, VH]`.
    pub(super) beta: &'a ExecutionTensor,
    /// Decay factor, `[T, VH]`.
    pub(super) g: &'a ExecutionTensor,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident delta dimensions overflow".into())
}

fn shape_error(what: &str) -> EngError {
    EngError::BadRequest(format!("resident delta recurrence: {what}"))
}

/// Returns `sqrt(sum(x * x) + eps)` over the last axis of a `[T, heads, KD]` tensor, as `[T, heads, 1]`.
fn l2_den(
    m: &mut MlxKernels,
    x: &ExecutionTensor,
    eps: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let sq = m.binary(BinaryOp::Mul, x, x)?;
    let sum = m.reduce(ReduceOp::Sum, &sq, 2, true)?;
    let sum = m.binary(BinaryOp::Add, &sum, eps)?;
    m.unary(UnaryOp::Sqrt, &sum)
}

/// Whether the flash-delta gated rule (R4-6) replaces the recurrent kernel. DEFAULT ON
/// since the gates passed (cohort: dNLL -0.000281, top-1 98.534%, KL 0.003540; robust
/// depths: KL 2-4e-6, top-1 8/8; the T >= 16 threshold keeps decode identical): set
/// `Q36_FLASH_DELTA=0` as the kill switch. The chunked WY-form kernel (4.1x) runs for
/// T >= 16; the recurrent kernel serves decode.
fn flash_delta_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var_os("Q36_FLASH_DELTA").is_some_and(|v| v == "0"))
}

/// Whether the fused prefill kernels (R4-7) replace the composite ops: set
/// Default ON (the coordinator-approved ship flip on the gate evidence: every kernel
/// bit-exact in isolation, all gates green); `Q36_FUSED_PREFILL=0` is the kill switch.
pub(super) fn fused_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("Q36_FUSED_PREFILL").is_none_or(|v| v != "0"))
}

/// The gated delta rule over a chunk of `T` tokens as one call of the gated delta kernel
/// (decision 0050).
///
/// Inputs `GatedInputs`, all lazy F32 (`KH` key heads, `VH` value heads, `VH` a multiple of `KH`):
/// - `q` and `k`, `[T, KH, KD]`: the query after its L2 norm and the `1 / sqrt(KD)` scale, and the
///   key after its L2 norm. Value head `h` reads key head `h / (VH / KH)`.
/// - `v`, `[T, VH, VD]`.
/// - `beta`, `[T, VH]`: the write strength `sigmoid(b)`.
/// - `g`, `[T, VH]`: the decay factor `exp(-exp(A_log) * softplus(a + dt_bias))`, not its logarithm.
/// - `state`, `[VH, VD, KD]`: the recurrent state before the chunk.
///
/// Per token and value head: `s = g * s`, `err = (v - s k) * beta`, `s = s + err (x) k`, and
/// `y = s q`. It returns `y`, `[T, VH, VD]`, and the new state, `[VH, VD, KD]`. The result equals
/// `T` calls of `delta::step`. The call is lazy and makes no synchronization.
///
/// A shape error returns `BadRequest` before the first native call.
pub(super) fn gated_delta_recurrence(
    m: &mut MlxKernels,
    inputs: &GatedInputs<'_>,
    state: &ExecutionTensor,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let GatedInputs { q, k, v, beta, g } = *inputs;
    let &[t, kh, kd] = q.shape() else {
        return Err(shape_error("q must be [T, KH, KD]"));
    };
    let &[_, vh, vd] = v.shape() else {
        return Err(shape_error("v must be [T, VH, VD]"));
    };
    if t == 0
        || kh == 0
        || !vh.is_multiple_of(kh)
        || k.shape() != [t, kh, kd]
        || v.shape() != [t, vh, vd]
        || beta.shape() != [t, vh]
        || g.shape() != [t, vh]
        || state.shape() != [vh, vd, kd]
    {
        return Err(shape_error("shapes disagree"));
    }
    let q4 = m.reshape(q, &[1, t, kh, kd])?;
    let k4 = m.reshape(k, &[1, t, kh, kd])?;
    let v4 = m.reshape(v, &[1, t, vh, vd])?;
    let g3 = m.reshape(g, &[1, t, vh])?;
    let beta3 = m.reshape(beta, &[1, t, vh])?;
    let s4 = m.reshape(state, &[1, vh, vd, kd])?;
    // The WY kernel engages only inside its whole envelope: the flag, T >= 16 (one full
    // chunk; decode T=1 stays on the recurrent kernel, bit-identical to flag-off), and the
    // spike's shape support (Dk % 64 == 0, Dv % 128 == 0, Hv % Hk == 0). Any other shape
    // - the tiny model's smaller dims or non-dividing head counts, for example - keeps
    // the recurrent kernel.
    let wy_eligible =
        flash_delta_enabled() && t >= 16 && kd % 64 == 0 && vd % 128 == 0 && vh % kh == 0;
    let out = if wy_eligible {
        m.flash_delta(&q4, &k4, &v4, &g3, &beta3, &s4)?
    } else {
        m.gated_delta(&q4, &k4, &v4, &g3, &beta3, &s4)?
    };
    let y = m.reshape(&out.y, &[t, vh, vd])?;
    let state = m.reshape(&out.state, &[vh, vd, kd])?;
    Ok((y, state))
}

/// Builds one DeltaNet chunk over `x` `[T, H]` and the old `rec` and `ring`.
///
/// Token `t` of the output and the final `rec` and `ring` equal what `T` calls of `delta::step`
/// give, in order. The chunk makes no synchronization. The caller replaces its state with the
/// returned `rec` and `ring`. `_sub` is the sub-chunk size of the plain-ops recurrence. The kernel
/// ignores it, and the callers pass it until the sub-chunk plan is removed.
/// A shape error returns `BadRequest` before the first native call.
pub(super) fn chunk(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RDelta,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
    x: &ExecutionTensor,
    _sub: u32,
) -> Result<DeltaChunk, EngError> {
    let d = &shape.delta;
    let kh = dim(d.key_heads)?;
    let kd = dim(d.key_dim)?;
    let vh = dim(d.value_heads)?;
    let vd = dim(d.value_dim)?;
    let cd = dim(d.conv_dim())?;
    let taps = dim(d.conv_kernel)?.checked_sub(1).ok_or_else(overflow)?;
    let kw = kh.checked_mul(kd).ok_or_else(overflow)?;
    let kw2 = kw.checked_mul(2).ok_or_else(overflow)?;
    let vw = vh.checked_mul(vd).ok_or_else(overflow)?;
    // A state of another shape but the same element count would reshape without an error, so
    // check the shapes exactly, as `delta::step` does.
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
    let &[t, _] = x.shape() else {
        return Err(EngError::BadRequest(
            "resident delta chunk input must be [T, H]".into(),
        ));
    };
    if t == 0 {
        return Err(EngError::BadRequest(
            "resident delta chunk needs a row".into(),
        ));
    }
    let rows = t.checked_mul(vh).ok_or_else(overflow)?;
    // Projections over all rows. Each `qmm` takes `[T, H]`.
    let u = ctx.kernels.qmm(x, &w.qkv)?;
    let z = ctx.kernels.qmm(x, &w.z)?;
    let bb = ctx.kernels.qmm(x, &w.b)?;
    // Packet Q36-BF16-b, as the decode step of `delta.rs`: the recurrence, its state, and the
    // convolution ring stay F32. `u`, `z`, and `aa` meet an F32 weight or state first and promote
    // on their own. `bb` feeds a sigmoid alone and the gated delta kernel takes F32 `beta`, so it
    // converts here.
    let aa = ctx.kernels.qmm(x, &w.a)?;
    // Share the scalar tensors before the mutable kernels borrow (the else branch's
    // cast_to also borrows ctx mutably).
    let qk_eps = ctx.consts.qk_eps.share();
    let sqrt_kd = ctx.consts.sqrt_kd.share();
    let consts = ctx.consts;
    let m = &mut *ctx.kernels;
    // The fused gdn_prep kernel (R4-7, Q36_FUSED_PREFILL=1) replaces the ~20-op
    // prep chain with one Metal kernel; the composite stays the default.
    let (q, k, v, g, beta, ring_next): (
        ExecutionTensor,
        ExecutionTensor,
        ExecutionTensor,
        ExecutionTensor,
        ExecutionTensor,
        ExecutionTensor,
    ) = if fused_prefill_enabled()
        && std::env::var_os("Q36_NO_GDNPREP").is_none()
        && kh == 16
        && kd == 128
        && vh == 32
        && vd == 128
    {
        let out = m.gdn_prep(
            &u,
            &aa,
            &bb,
            ring,
            &w.conv_taps,
            &w.conv_last,
            &w.neg_exp_a_log,
            &w.dt_bias,
            &qk_eps,
            &sqrt_kd,
        )?;
        (out.q, out.k, out.v, out.g, out.beta, out.ring)
    } else {
        let bb = m.cast(&bb, Dtype::F32)?;
        let (pre, ring_next) = causal_conv(m, &u, ring, &w.conv_taps, &w.conv_last)?;
        let y = m.silu(&pre)?;
        let q = m.slice(&y, &[0, 0], &[t, kw])?;
        let q = m.reshape(&q, &[t, kh, kd])?;
        let k = m.slice(&y, &[0, kw], &[t, kw2])?;
        let k = m.reshape(&k, &[t, kh, kd])?;
        let v = m.slice(&y, &[0, kw2], &[t, cd])?;
        let v = m.reshape(&v, &[t, vh, vd])?;
        let qd = l2_den(m, &q, &consts.qk_eps)?;
        let qd = m.binary(BinaryOp::Mul, &qd, &consts.sqrt_kd)?;
        let q = m.binary(BinaryOp::Div, &q, &qd)?;
        let kden = l2_den(m, &k, &consts.qk_eps)?;
        let k = m.binary(BinaryOp::Div, &k, &kden)?;
        let pre_a = m.binary(BinaryOp::Add, &aa, &w.dt_bias)?;
        let soft = m.binary(BinaryOp::LogAddExp, &pre_a, &consts.zero)?;
        let decay = m.binary(BinaryOp::Mul, &w.neg_exp_a_log, &soft)?;
        let g = m.unary(UnaryOp::Exp, &decay)?;
        let beta = m.unary(UnaryOp::Sigmoid, &bb)?;
        (q, k, v, g, beta, ring_next)
    };
    let inputs = GatedInputs {
        q: &q,
        k: &k,
        v: &v,
        beta: &beta,
        g: &g,
    };
    let (core, state) = gated_delta_recurrence(m, &inputs, rec)?;
    let core = m.reshape(&core, &[rows, vd])?;
    // Output: silu(z) * rms_norm(core, norm). The norm weight applies as stored.
    // The gdn_post fused kernel (R4-7, Q36_FUSED_PREFILL=1) replaces the
    // norm+gate chain in one kernel; the composite stays the default.
    let o = if fused_prefill_enabled()
        && std::env::var_os("Q36_NO_GDNPOST").is_none()
        && vd == 128
        && vh == 32
    {
        m.gdn_post(&core, &z, &w.norm, shape.eps)?
    } else {
        let normed = m.rms_norm(&core, &w.norm, shape.eps)?;
        let zr = m.reshape(&z, &[rows, vd])?;
        let gate = m.silu(&zr)?;
        let o = m.binary(BinaryOp::Mul, &normed, &gate)?;
        m.reshape(&o, &[t, vw])?
    };
    // The out-projection takes the activation dtype (packet Q36-BF16-b).
    let o = to_act(ctx, &o)?;
    let out = ctx.kernels.qmm(&o, &w.out)?;
    Ok(DeltaChunk {
        out,
        rec: state,
        ring: ring_next,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::ops::{assert_parity, f32_tensor, values};
    use super::super::super::weights::{RMixer, ResidentWeights};
    use super::*;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, DeltaState, MixerW};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY, TINY_WIDE};
    use crate::family::spec::FamilySpec;
    use crate::weights::StreamedCheckpoint;

    /// Input of position `n`: `x[i] = (((i + 5 n) % 11) - 5) / 160`. The small scale keeps the query
    /// and key sums of squares near the epsilon `D * 1e-6`, so that the epsilon rule changes the
    /// output by several percent (packet Q36-C2 measured 9 percent of the largest output).
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

    /// The host DeltaNet layer 0 run over positions `0..count`, with its outputs and the qkv
    /// projection of every position (the ring holds the last three).
    struct HostRun {
        state: DeltaState,
        projected: Vec<Vec<f32>>,
        outputs: Vec<Vec<f32>>,
    }

    fn host_run(ckpt: &StreamedCheckpoint, oracle: &HostWeights, count: u32) -> HostRun {
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Delta(host_w) = &view.layers[0].mixer else {
            panic!("layer 0 is DeltaNet");
        };
        let mut cpu = CpuProjector::new(ckpt, 1);
        let mut run = HostRun {
            state: DeltaState::new(&shape.delta),
            projected: Vec::new(),
            outputs: Vec::new(),
        };
        for n in 0..count {
            let x = input(n);
            run.projected.push(cpu.linear(host_w.qkv, &x).unwrap());
            let out = host::delta_step(&mut cpu, &shape, host_w, &mut run.state, &x).unwrap();
            run.outputs.push(out);
        }
        run
    }

    /// The ring `[CD, 3]` after the positions in `projected`, oldest first, zero-padded in front.
    fn ring_after(projected: &[Vec<f32>], cd: usize) -> Vec<f32> {
        (0..cd)
            .flat_map(|c| {
                (0..3).map(move |k| {
                    projected
                        .len()
                        .checked_sub(3 - k)
                        .map_or(0.0, |t| projected[t][c])
                })
            })
            .collect()
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

    /// The resident start state after `warm` host positions.
    fn start_state(
        kernels: &mut MlxKernels,
        shape: &HostShape,
        warm: &HostRun,
    ) -> (ExecutionTensor, ExecutionTensor) {
        let d = &shape.delta;
        let (vh, kd, vd, cd) = (d.value_heads, d.key_dim, d.value_dim, d.conv_dim());
        (
            upload(
                kernels,
                vec![vh as u32, vd as u32, kd as u32],
                warm.state.rec(),
            ),
            upload(
                kernels,
                vec![cd as u32, 3],
                &ring_after(&warm.projected, cd),
            ),
        )
    }

    /// The `_sub` argument of `chunk`, which the kernel ignores.
    const SUB: u32 = 1;

    /// Runs `count` positions from position `warm` as one `chunk` and compares the outputs, the
    /// state, and the ring with the host run over `warm + count` positions.
    fn chunk_matches_the_host(spec: &FamilySpec, warm: u32, count: u32) {
        let (ckpt, oracle, w, mut kernels) = setup(spec);
        let shape = oracle.shape().clone();
        let RMixer::Delta(rw) = &w.layers[0].mixer else {
            panic!("resident layer 0 is DeltaNet");
        };
        let before = host_run(&ckpt, &oracle, warm);
        let all = host_run(&ckpt, &oracle, warm + count);
        let (rec, ring) = start_state(&mut kernels, &shape, &before);
        let x = rows(&mut kernels, warm, count);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let next = chunk(&mut ctx, &shape, rw, &rec, &ring, &x, SUB).unwrap();
        assert_eq!(next.out.shape(), [count, 64], "output shape");
        assert_eq!(next.rec.shape(), rec.shape(), "rec shape");
        assert_eq!(next.ring.shape(), ring.shape(), "ring shape");
        let want_out: Vec<f32> = all.outputs[warm as usize..].concat();
        assert_parity(
            &values(&mut kernels, &next.out).unwrap(),
            &want_out,
            "chunk output",
        );
        assert_parity(
            &values(&mut kernels, &next.rec).unwrap(),
            all.state.rec(),
            "rec",
        );
        let want_ring = ring_after(&all.projected, shape.delta.conv_dim());
        assert_parity(
            &values(&mut kernels, &next.ring).unwrap(),
            &want_ring,
            "ring",
        );
    }

    #[test]
    fn a_chunk_from_a_zero_state_matches_the_host_steps() {
        let _serial = crate::native_test_lock();
        chunk_matches_the_host(&TINY, 0, 7);
    }

    #[test]
    fn a_chunk_continues_from_a_nonzero_state_and_ring() {
        // Three host positions leave a full ring of real projections and a nonzero state.
        let _serial = crate::native_test_lock();
        chunk_matches_the_host(&TINY, 3, 4);
    }

    #[test]
    fn value_heads_follow_their_key_head_in_a_chunk() {
        // TINY_WIDE has 2 key heads and 4 value heads: value heads 0 and 1 read key head 0, and
        // value heads 2 and 3 read key head 1. Tiling the key heads would read 0, 1, 0, 1.
        let _serial = crate::native_test_lock();
        chunk_matches_the_host(&TINY_WIDE, 0, 5);
    }

    #[test]
    fn a_forty_token_chunk_matches_the_host_steps() {
        // The kernel carries the state across 40 tokens; the gate and the strength advance per token.
        let _serial = crate::native_test_lock();
        chunk_matches_the_host(&TINY, 3, 40);
    }

    #[test]
    fn a_chunk_rejects_state_of_the_wrong_shape_and_an_input_of_the_wrong_rank() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup(&TINY);
        let shape = oracle.shape().clone();
        let RMixer::Delta(rw) = &w.layers[0].mixer else {
            panic!("resident layer 0 is DeltaNet");
        };
        let warm = host_run(&tiny::checkpoint(&TINY), &oracle, 0);
        let (rec, ring) = start_state(&mut kernels, &shape, &warm);
        // [1, 32, 32] has one head instead of two. [1, 3] has one channel instead of the full width.
        let flat_rec = upload(&mut kernels, vec![1, 32, 32], &[0.0; 1024]);
        let short_ring = upload(&mut kernels, vec![1, 3], &[0.0; 3]);
        let x = rows(&mut kernels, 0, 2);
        let flat = upload(&mut kernels, vec![128], &[0.0; 128]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let cases = [
            (
                "delta rec shape",
                chunk(&mut ctx, &shape, rw, &flat_rec, &ring, &x, SUB),
            ),
            (
                "delta ring shape",
                chunk(&mut ctx, &shape, rw, &rec, &short_ring, &x, SUB),
            ),
            (
                "chunk input must be [T, H]",
                chunk(&mut ctx, &shape, rw, &rec, &ring, &flat, SUB),
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
    fn a_chunk_builds_a_lazy_graph_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup(&TINY);
        let shape = oracle.shape().clone();
        let RMixer::Delta(rw) = &w.layers[0].mixer else {
            panic!("resident layer 0 is DeltaNet");
        };
        let warm = host_run(&ckpt, &oracle, 0);
        let (rec, ring) = start_state(&mut kernels, &shape, &warm);
        let x = rows(&mut kernels, 0, 7);
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let next = chunk(&mut ctx, &shape, rw, &rec, &ring, &x, SUB).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "the chunk built a lazy graph"
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
    /// The envelope must-fix: Hv 3 with Hk 2 (Dk 64, Dv 128, batch one) must fail the WY
    /// envelope before any native call instead of reaching the kernel body.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn flash_delta_rejects_mismatched_head_counts() {
        use crate::kernels::MlxKernels;
        let (kh, vh, kd, vd) = (2u32, 3u32, 64u32, 128u32);
        let rows = 4u32;
        let mut kernels = MlxKernels::try_new().unwrap();
        let mk = |kernels: &mut MlxKernels, shape: Vec<u32>| {
            let n = shape.iter().map(|&d| d as usize).product::<usize>();
            let data: Vec<f32> = (0..n).map(|i| (i as f32 % 17.0) / 17.0 - 0.5).collect();
            kernels
                .import_tensor(&f32_tensor(shape, &data).unwrap())
                .unwrap()
        };
        let q4 = mk(&mut kernels, vec![1, rows, kh, kd]);
        let k4 = mk(&mut kernels, vec![1, rows, kh, kd]);
        let v4 = mk(&mut kernels, vec![1, rows, vh, vd]);
        let g3 = mk(&mut kernels, vec![1, rows, vh]);
        let b3 = mk(&mut kernels, vec![1, rows, vh]);
        let s4 = mk(&mut kernels, vec![1, vh, vd, kd]);
        let out = kernels.flash_delta(&q4, &k4, &v4, &g3, &b3, &s4);
        assert!(out.is_err(), "Hv 3 with Hk 2 must fail the WY envelope");
    }

    /// R4-6: the flash-delta chain (Q36_FLASH_DELTA, the chunked WY-form kernel) matches the
    /// recurrent kernel within the spike's tolerance bound on the production shape (KH 16,
    /// VH 32, KD 128, VD 128) with a NON-ZERO initial state, at rows 1, 37, 256, and 1024 -
    /// including the non-multiple-of-eight tails. The WY form reassociates the math, so the
    /// outputs differ from the recurrent kernel's by a few e-6 absolute (the spike's
    /// measured 1.2e-6 on its data; this harness measures 2.4e-6 at rows 37 - the same f32 noise at a different scale; the 5e-6 bound is synthetic-data-only, real acceptance is the cohort KL gate).
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn flash_delta_matches_the_recurrent_kernel() {
        use crate::kernels::MlxKernels;
        use crate::kernels::mlx::GatedDeltaOutput;
        let (kh, vh, kd, vd) = (16u32, 32u32, 128u32, 128u32);
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut run = |rows: u32, seed: u32| {
            let idx = |i: u32, m: u32| {
                ((seed
                    .wrapping_mul(1103515245)
                    .wrapping_add(i.wrapping_mul(12345)))
                    % m) as f32
            };
            let q: Vec<f32> = (0..rows * kh * kd)
                .map(|i| idx(i, 97) / 97.0 - 0.5)
                .collect();
            let k: Vec<f32> = (0..rows * kh * kd)
                .map(|i| idx(i + 7, 97) / 97.0 - 0.5)
                .collect();
            let v: Vec<f32> = (0..rows * vh * vd)
                .map(|i| idx(i + 13, 97) / 97.0 - 0.5)
                .collect();
            let g: Vec<f32> = (0..rows * vh).map(|i| 0.05 + idx(i, 97) / 500.0).collect();
            let beta: Vec<f32> = (0..rows * vh)
                .map(|i| 0.5 + idx(i + 3, 97) / 300.0)
                .collect();
            let s0: Vec<f32> = (0..vh * vd * kd)
                .map(|i| idx(i + 29, 97) / 97.0 - 0.5)
                .collect();
            let mk = |kernels: &mut MlxKernels, shape: Vec<u32>, data: &[f32]| {
                kernels
                    .import_tensor(&f32_tensor(shape, data).unwrap())
                    .unwrap()
            };
            let q4 = mk(&mut kernels, vec![1, rows, kh, kd], &q);
            let k4 = mk(&mut kernels, vec![1, rows, kh, kd], &k);
            let v4 = mk(&mut kernels, vec![1, rows, vh, vd], &v);
            let g3 = mk(&mut kernels, vec![1, rows, vh], &g);
            let b3 = mk(&mut kernels, vec![1, rows, vh], &beta);
            let s4 = mk(&mut kernels, vec![1, vh, vd, kd], &s0);
            let GatedDeltaOutput {
                y: y_old,
                state: s_old,
            } = kernels.gated_delta(&q4, &k4, &v4, &g3, &b3, &s4).unwrap();
            let GatedDeltaOutput {
                y: y_new,
                state: s_new,
            } = kernels.flash_delta(&q4, &k4, &v4, &g3, &b3, &s4).unwrap();
            let (y_old, s_old, y_new, s_new) = (
                values(&mut kernels, &y_old).unwrap(),
                values(&mut kernels, &s_old).unwrap(),
                values(&mut kernels, &y_new).unwrap(),
                values(&mut kernels, &s_new).unwrap(),
            );
            assert_eq!(y_old.len(), y_new.len());
            assert_eq!(s_old.len(), s_new.len());
            let max = y_old
                .iter()
                .zip(&y_new)
                .chain(s_old.iter().zip(&s_new))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max <= 5.0e-6,
                "rows {rows}: the flash-delta chain differs from the recurrent kernel by {max:e}"
            );
        };
        for (rows, seed) in [(1u32, 3u32), (37, 5), (256, 7), (1024, 11)] {
            run(rows, seed);
        }
    }
}
