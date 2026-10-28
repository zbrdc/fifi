//! Batched DeltaNet chunk, option A (plan S4 sections 3.2 and 3.3).
//!
//! Projections, the causal convolution, the L2 norms, and the gates run once over `[T, ...]`
//! tensors. The delta-rule recurrence then runs as `T` steps of the operations of `step` in
//! `resident/deltanet.rs`, on slices of those tensors, inside the same lazy graph. The result
//! equals `T` decode steps. Nothing here imports, reads back, or evaluates.

use super::super::{
    ops::{Ctx, dim},
    weights::RDelta,
};
use super::conv::causal_conv;
use crate::{
    EngError,
    family::flash_next::deltanet::DeltaShape,
    kernels::{BinaryOp, MlxKernels, ReduceOp, UnaryOp},
    tensor::ExecutionTensor,
};

/// Output and replacement state of one DeltaNet chunk. Every tensor is lazy.
pub(super) struct DeltaChunk {
    /// Layer output `[T, H]`.
    pub(super) out: ExecutionTensor,
    /// New recurrent state `[VH, KD, VD]`, the host layout.
    pub(super) rec: ExecutionTensor,
    /// New convolution ring `[CD, CK - 1]`, the host layout.
    pub(super) ring: ExecutionTensor,
}

/// The per-token operands of the delta rule. Axis 0 of every tensor is the token. All are lazy F32.
struct Operands<'a> {
    /// Decay `alpha`, `[T, VH, 1, 1]`.
    alpha: &'a ExecutionTensor,
    /// Write strength `beta`, `[T, VH, 1, 1]`.
    beta: &'a ExecutionTensor,
    /// Normalized key as a row, `[T, VH, 1, KD]`.
    k_row: &'a ExecutionTensor,
    /// Normalized and scaled query as a row, `[T, VH, 1, KD]`.
    q_row: &'a ExecutionTensor,
    /// Value as a row, `[T, VH, 1, VD]`.
    v: &'a ExecutionTensor,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident delta dimensions overflow".into())
}

/// Returns `sqrt(sum(x * x) + 1e-6)` over the last axis of a `[T, heads, KD]` tensor, as `[T, heads, 1]`.
fn l2_den(
    m: &mut MlxKernels,
    x: &ExecutionTensor,
    eps6: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let sq = m.binary(BinaryOp::Mul, x, x)?;
    let sum = m.reduce(ReduceOp::Sum, &sq, 2, true)?;
    let sum = m.binary(BinaryOp::Add, &sum, eps6)?;
    m.unary(UnaryOp::Sqrt, &sum)
}

/// Slices token `i` of a rank-4 tensor and keeps axis 0 with size 1: `[1, a, b, c]`.
fn token(m: &mut MlxKernels, x: &ExecutionTensor, i: u32) -> Result<ExecutionTensor, EngError> {
    let &[_, a, b, c] = x.shape() else {
        return Err(EngError::BadRequest(
            "resident delta operand must be rank 4".into(),
        ));
    };
    let next = i.checked_add(1).ok_or_else(overflow)?;
    m.slice(x, &[i, 0, 0, 0], &[next, a, b, c])
}

/// Runs the delta rule over the tokens of `ops`, starting from `rec` `[VH, KD, VD]`.
///
/// Each step is the operation sequence of `step` in `resident/deltanet.rs`, on operands with a
/// leading axis of size 1. It returns the readout of every token, `[T, VH, 1, VD]`, and the final
/// state `[1, VH, KD, VD]`. At real dimensions each token drops about 9 MiB of state-sized
/// tensors (the decayed state, the outer product, and the replaced state), which the retire list
/// counts against its cap.
fn recurrence(
    m: &mut MlxKernels,
    rec: &ExecutionTensor,
    ops: &Operands<'_>,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let &[vh, kd, vd] = rec.shape() else {
        return Err(EngError::BadRequest(
            "resident delta rec must be rank 3".into(),
        ));
    };
    let Some(&t) = ops.alpha.shape().first() else {
        return Err(EngError::BadRequest(
            "resident delta operands need a token axis".into(),
        ));
    };
    let count = usize::try_from(t).map_err(|_| overflow())?;
    let mut state = m.reshape(rec, &[1, vh, kd, vd])?;
    let mut cores = Vec::with_capacity(count);
    for i in 0..t {
        let a_i = token(m, ops.alpha, i)?;
        let b_i = token(m, ops.beta, i)?;
        let kr = token(m, ops.k_row, i)?;
        let qr = token(m, ops.q_row, i)?;
        let v_i = token(m, ops.v, i)?;
        let kc = m.reshape(&kr, &[1, vh, kd, 1])?;
        // Decay the state, deltanet.rs:233-236.
        let s1 = m.binary(BinaryOp::Mul, &state, &a_i)?;
        // Delta-rule error, deltanet.rs:237-244.
        let prev = m.batched_mm(&kr, &s1)?;
        let err = m.binary(BinaryOp::Sub, &v_i, &prev)?;
        let delta = m.binary(BinaryOp::Mul, &err, &b_i)?;
        // Rank-one update, deltanet.rs:245-249.
        let outer = m.binary(BinaryOp::Mul, &kc, &delta)?;
        state = m.binary(BinaryOp::Add, &s1, &outer)?;
        // Read out with the updated state, deltanet.rs:250-257.
        cores.push(m.batched_mm(&qr, &state)?);
    }
    let parts: Vec<&ExecutionTensor> = cores.iter().collect();
    let joined = m.concatenate(&parts, 0)?;
    Ok((joined, state))
}

/// Builds one DeltaNet chunk over `x` `[T, H]` and the old `rec` and `ring`.
///
/// Token `t` of the output and the final `rec` and `ring` equal what `T` calls of `step` in
/// `resident/deltanet.rs` give, in order. It performs no synchronization. The caller replaces
/// its state with the returned `rec` and `ring`. A shape error returns `BadRequest` before the
/// first native call.
pub(super) fn chunk(
    ctx: &mut Ctx<'_>,
    shape: &DeltaShape,
    w: &RDelta,
    rec: &ExecutionTensor,
    ring: &ExecutionTensor,
    x: &ExecutionTensor,
) -> Result<DeltaChunk, EngError> {
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
    let vw = vh.checked_mul(vd).ok_or_else(overflow)?;
    let rep = vh.checked_div(kh).ok_or_else(overflow)?;
    // A wrong rank-3 or rank-2 shape could broadcast, so check exactly, as deltanet.rs:188-189 does.
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
    let &[t, _] = x.shape() else {
        return Err(EngError::BadRequest(
            "resident delta chunk input must be [T, H]".into(),
        ));
    };
    let rows = t.checked_mul(vh).ok_or_else(overflow)?;
    // Projections over all rows, deltanet.rs:191-198. Each `qmm` takes `[T, H]` (packet PF-K).
    let u = ctx.kernels.qmm(x, &w.qkv)?;
    let z = ctx.kernels.qmm(x, &w.z)?;
    let bb = ctx.kernels.qmm(x, &w.b)?;
    let aa = ctx.kernels.qmm(x, &w.a)?;
    let consts = ctx.consts;
    let m = &mut *ctx.kernels;
    // Causal convolution over time and SiLU, deltanet.rs:202-211.
    let (pre, ring_next) = causal_conv(m, &u, ring, &w.conv_taps, &w.conv_last, 1)?;
    let y = m.silu(&pre)?;
    // Split y into q, k, v, deltanet.rs:216-221 and 243.
    let q = m.slice(&y, &[0, 0], &[t, kw])?;
    let q = m.reshape(&q, &[t, kh, kd])?;
    let k = m.slice(&y, &[0, kw], &[t, kw2])?;
    let k = m.reshape(&k, &[t, kh, kd])?;
    let v = m.slice(&y, &[0, kw2], &[t, cd])?;
    let v = m.reshape(&v, &[t, vh, 1, vd])?;
    // L2 normalization with eps inside the root; q also divides by sqrt(KD), deltanet.rs:218-223.
    let qd = l2_den(m, &q, &consts.eps6)?;
    let qd = m.binary(BinaryOp::Mul, &qd, &consts.sqrt_kd)?;
    let q = m.binary(BinaryOp::Div, &q, &qd)?;
    let kden = l2_den(m, &k, &consts.eps6)?;
    let k = m.binary(BinaryOp::Div, &k, &kden)?;
    // Value head h uses key head h / rep, deltanet.rs:226 and 230.
    let q3 = m.repeat(&q, rep, 1)?;
    let k3 = m.repeat(&k, rep, 1)?;
    let q_row = m.reshape(&q3, &[t, vh, 1, kd])?;
    let k_row = m.reshape(&k3, &[t, vh, 1, kd])?;
    // alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b), deltanet.rs:231-232.
    let pre_a = m.binary(BinaryOp::Add, &aa, &w.dt_bias)?;
    let soft = m.binary(BinaryOp::LogAddExp, &pre_a, &consts.zero)?;
    let decay = m.binary(BinaryOp::Mul, &w.neg_exp_a_log, &soft)?;
    let alpha = m.unary(UnaryOp::Exp, &decay)?;
    let alpha = m.reshape(&alpha, &[t, vh, 1, 1])?;
    let beta = m.unary(UnaryOp::Sigmoid, &bb)?;
    let beta = m.reshape(&beta, &[t, vh, 1, 1])?;
    let ops = Operands {
        alpha: &alpha,
        beta: &beta,
        k_row: &k_row,
        q_row: &q_row,
        v: &v,
    };
    let (joined, state) = recurrence(m, rec, &ops)?;
    let core = m.reshape(&joined, &[rows, vd])?;
    // RMS norm with the plain weight, then the sigmoid(z) gate, deltanet.rs:258-261.
    let normed = m.rms_norm(&core, &w.norm, shape.eps())?;
    let zr = m.reshape(&z, &[rows, vd])?;
    let gate = m.unary(UnaryOp::Sigmoid, &zr)?;
    let o = m.binary(BinaryOp::Mul, &normed, &gate)?;
    let o = m.reshape(&o, &[t, vw])?;
    let rec_next = m.reshape(&state, &[vh, kd, vd])?;
    // Output projection, deltanet.rs:264.
    let out = ctx.kernels.qmm(&o, &w.out)?;
    Ok(DeltaChunk {
        out,
        rec: rec_next,
        ring: ring_next,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::deltanet::step as decode_step;
    use super::super::super::ops::{assert_close, values, zeros};
    use super::super::super::weights::{RAttn, ResidentWeights, synthetic};
    use super::super::{assert_parity, import_f32};
    use super::*;

    /// The DeltaNet weights of resident layer 0.
    fn delta_weights(weights: &ResidentWeights) -> &RDelta {
        let RAttn::Delta(rw) = &weights.layers[0].attn else {
            panic!("resident layer 0 is not DeltaNet");
        };
        rw
    }

    /// Input row of token `t`: element `i` is `(((i + 3t) % 7) - 3) / 8` for `i` in `0..64` (H = 64).
    fn x_row(t: usize) -> Vec<f32> {
        (0..64)
            .map(|i| (((i + 3 * t) % 7) as f32 - 3.0) * 0.125)
            .collect()
    }

    /// Imports the input rows of tokens `first..first + count` as `[count, 64]`.
    fn rows(kernels: &mut MlxKernels, first: usize, count: usize) -> ExecutionTensor {
        let data: Vec<f32> = (first..first + count).flat_map(x_row).collect();
        import_f32(kernels, vec![count as u32, 64], &data)
    }

    /// Imports a zero `rec` `[2, 32, 32]` and `ring` `[128, 3]`: the fixture's VH, KD, VD, CD, and CK - 1.
    fn zero_state(kernels: &mut MlxKernels) -> (ExecutionTensor, ExecutionTensor) {
        let rec = zeros(vec![2, 32, 32]).unwrap();
        let ring = zeros(vec![128, 3]).unwrap();
        let mut got = kernels.import_many(&[&rec, &ring]).unwrap().into_iter();
        (got.next().unwrap(), got.next().unwrap())
    }

    /// Outputs (concatenated), final `rec`, and final `ring` of `count` decode steps from a zero state.
    fn decode_run(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        count: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let shape = weights.shape.delta;
        let rw = delta_weights(weights);
        let (mut rec, mut ring) = zero_state(kernels);
        let mut outs = Vec::new();
        for t in 0..count {
            let x = import_f32(kernels, vec![1, 64], &x_row(t));
            let mut ctx = Ctx {
                kernels: &mut *kernels,
                consts: &weights.consts,
            };
            let next = decode_step(&mut ctx, &shape, rw, &rec, &ring, &x).unwrap();
            outs.extend(values(kernels, &next.out).unwrap());
            rec = next.rec;
            ring = next.ring;
        }
        let rec_values = values(kernels, &rec).unwrap();
        let ring_values = values(kernels, &ring).unwrap();
        (outs, rec_values, ring_values)
    }

    /// VH = 2, KD = 2, VD = 2, T = 3. Operands are token-major, then head. Head 1 has `alpha = 1`
    /// and `beta = 0`, so its state stays `[[5, 6], [7, 8]]` and its readout is `q . state`.
    /// Head 0 by hand, with rows of `rec` written `[r0, r1]` and `rec = [[1, 2], [3, 4]]`:
    /// - Token 0 (`alpha 1/2`, `beta 1/2`, `k (1,0)`, `q (0,1)`, `v (2,4)`): the decayed state is
    ///   `[[1/2, 1], [3/2, 2]]`; `prev = k . s1 = (1/2, 1)`; `err = (3/2, 3)`; `delta = (3/4, 3/2)`;
    ///   the state becomes `[[5/4, 5/2], [3/2, 2]]`; the readout is row 1: `(3/2, 2)`.
    /// - Token 1 (`alpha 1/4`, `beta 1/2`, `k (1,1)`, `q (1,2)`, `v (1,2)`): the decayed state is
    ///   `[[5/16, 5/8], [3/8, 1/2]]`; `prev = (11/16, 9/8)`; `err = (5/16, 7/8)`;
    ///   `delta = (5/32, 7/16)`; the state becomes `[[15/32, 17/16], [17/32, 15/16]]`; the readout
    ///   is row 0 plus twice row 1: `(49/32, 47/16)`.
    /// - Token 2 (`alpha 1/2`, `beta 1`, `k (0,1)`, `q (1,0)`, `v (1,1)`): the decayed state is
    ///   `[[15/64, 17/32], [17/64, 15/32]]`; `prev` is its row 1; `err = (47/64, 17/32)`; with
    ///   `beta 1` and `k = (0,1)` row 1 becomes `v`, so the state is `[[15/64, 17/32], [1, 1]]`;
    ///   the readout is row 0: `(15/64, 17/32)`.
    ///
    /// Head 1 readouts: `(1,1)` gives `(12, 14)`; `(2,0)` gives `(10, 12)`; `(0,1)` gives `(7, 8)`.
    /// Every value is a dyadic rational, so F32 holds each one exactly.
    #[test]
    fn recurrence_matches_hand_values() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let rec = import_f32(
            &mut kernels,
            vec![2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let alpha = import_f32(
            &mut kernels,
            vec![3, 2, 1, 1],
            &[0.5, 1.0, 0.25, 1.0, 0.5, 1.0],
        );
        let beta = import_f32(
            &mut kernels,
            vec![3, 2, 1, 1],
            &[0.5, 0.0, 0.5, 0.0, 1.0, 0.0],
        );
        let k_row = import_f32(
            &mut kernels,
            vec![3, 2, 1, 2],
            &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        );
        let q_row = import_f32(
            &mut kernels,
            vec![3, 2, 1, 2],
            &[0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        );
        let v = import_f32(
            &mut kernels,
            vec![3, 2, 1, 2],
            &[
                2.0, 4.0, 9.0, 10.0, 1.0, 2.0, 11.0, 12.0, 1.0, 1.0, 13.0, 14.0,
            ],
        );
        let ops = Operands {
            alpha: &alpha,
            beta: &beta,
            k_row: &k_row,
            q_row: &q_row,
            v: &v,
        };

        let (cores, state) = recurrence(&mut kernels, &rec, &ops).unwrap();

        assert_eq!(cores.shape(), &[3, 2, 1, 2][..], "readout shape");
        assert_eq!(state.shape(), &[1, 2, 2, 2][..], "state shape");
        let got = values(&mut kernels, &cores).unwrap();
        let want = [
            1.5, 2.0, 12.0, 14.0, 1.53125, 2.9375, 10.0, 12.0, 0.234375, 0.53125, 7.0, 8.0,
        ];
        assert_close(&got, &want, "readouts");
        let got_state = values(&mut kernels, &state).unwrap();
        let want_state = [0.234375, 0.53125, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0];
        assert_close(&got_state, &want_state, "state");
    }

    /// A chunk of `T` tokens from a zero state equals `T` decode steps: outputs, `rec`, and `ring`.
    #[test]
    fn chunk_matches_decode_steps_for_each_chunk_length() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = synthetic(&mut kernels);
        let shape = weights.shape.delta;
        let rw = delta_weights(&weights);
        for count in [1usize, 2, 3, 4, 7] {
            let (want_out, want_rec, want_ring) = decode_run(&mut kernels, &weights, count);
            let x = rows(&mut kernels, 0, count);
            let (rec, ring) = zero_state(&mut kernels);
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };

            let got = chunk(&mut ctx, &shape, rw, &rec, &ring, &x).unwrap();

            assert_eq!(
                got.out.shape(),
                &[count as u32, 64][..],
                "T {count} output shape"
            );
            assert_eq!(got.rec.shape(), &[2, 32, 32][..], "T {count} rec shape");
            assert_eq!(got.ring.shape(), &[128, 3][..], "T {count} ring shape");
            let out = values(&mut kernels, &got.out).unwrap();
            assert_parity(&out, &want_out, &format!("T {count} output"));
            let rec_values = values(&mut kernels, &got.rec).unwrap();
            assert_parity(&rec_values, &want_rec, &format!("T {count} rec"));
            let ring_values = values(&mut kernels, &got.ring).unwrap();
            assert_parity(&ring_values, &want_ring, &format!("T {count} ring"));
        }
    }

    /// A chunk of 3 and a chunk of 4 that carry the state equal 7 decode steps. The second chunk
    /// starts from a nonzero `rec` and a full `ring`, so a decay or a ring shift that is skipped at
    /// its first token shows.
    #[test]
    fn chunks_that_carry_the_state_equal_one_run() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = synthetic(&mut kernels);
        let shape = weights.shape.delta;
        let rw = delta_weights(&weights);
        let (want_out, want_rec, want_ring) = decode_run(&mut kernels, &weights, 7);
        let first_rows = rows(&mut kernels, 0, 3);
        let second_rows = rows(&mut kernels, 3, 4);
        let (rec, ring) = zero_state(&mut kernels);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let first = chunk(&mut ctx, &shape, rw, &rec, &ring, &first_rows).unwrap();
        let second = chunk(&mut ctx, &shape, rw, &first.rec, &first.ring, &second_rows).unwrap();

        let out_a = values(&mut kernels, &first.out).unwrap();
        assert_parity(&out_a, &want_out[..192], "first chunk output");
        let out_b = values(&mut kernels, &second.out).unwrap();
        assert_parity(&out_b, &want_out[192..], "second chunk output");
        let rec_values = values(&mut kernels, &second.rec).unwrap();
        assert_parity(&rec_values, &want_rec, "rec after both chunks");
        let ring_values = values(&mut kernels, &second.ring).unwrap();
        assert_parity(&ring_values, &want_ring, "ring after both chunks");
    }

    #[test]
    fn chunk_rejects_state_and_input_of_the_wrong_shape() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = synthetic(&mut kernels);
        let shape = weights.shape.delta;
        let rw = delta_weights(&weights);
        let (rec, ring) = zero_state(&mut kernels);
        // [1, 32, 32] would broadcast against alpha [T, 2, 1, 1]; [1, 3] would broadcast against the taps.
        let flat_rec = kernels
            .import_tensor(&zeros(vec![1, 32, 32]).unwrap())
            .unwrap();
        let short_ring = kernels.import_tensor(&zeros(vec![1, 3]).unwrap()).unwrap();
        let x = rows(&mut kernels, 0, 2);
        let flat_x = import_f32(&mut kernels, vec![64], &x_row(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let cases = [
            (
                "rec [1, 32, 32]",
                chunk(&mut ctx, &shape, rw, &flat_rec, &ring, &x),
                "delta rec shape",
            ),
            (
                "ring [1, 3]",
                chunk(&mut ctx, &shape, rw, &rec, &short_ring, &x),
                "delta ring shape",
            ),
            (
                "input [64]",
                chunk(&mut ctx, &shape, rw, &rec, &ring, &flat_x),
                "chunk input",
            ),
        ];

        for (what, result, word) in cases {
            assert!(
                matches!(&result, Err(EngError::BadRequest(m)) if m.contains(word)),
                "{what}: {:?}",
                result.as_ref().err()
            );
        }
        let ok = chunk(&mut ctx, &shape, rw, &rec, &ring, &x);
        assert!(ok.is_ok(), "valid call: {:?}", ok.as_ref().err());
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn chunk_builds_lazy_outputs_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = synthetic(&mut kernels);
        let shape = weights.shape.delta;
        let rw = delta_weights(&weights);
        let (rec, ring) = zero_state(&mut kernels);
        let x = rows(&mut kernels, 0, 7);
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let next = chunk(&mut ctx, &shape, rw, &rec, &ring, &x).unwrap();

        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "chunk synchronized"
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
