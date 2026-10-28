//! Batched PLE block (plan S4 sections 3.1 and 3.2).
//!
//! The chunk form of `block` in the parent module. The per-token parts (keys, gate, gated update)
//! gain a leading `T` axis. The convolution runs over time through `causal_conv`, with the taps `N`
//! rows apart because the ring stores `N` tokens per tap. The function builds lazy graphs only.

use super::super::{
    ops::{Ctx, dim, rms0_rows},
    ple::gate,
    weights::RPle,
};
use super::conv::causal_conv;
use crate::{
    EngError,
    family::flash_next::step::ModelShape,
    kernels::{BinaryOp, ReduceOp},
    tensor::ExecutionTensor,
};

/// Output of the chunk PLE block: the new hyper state `[T, C, H]` and the new ring `[W, (PCK - 1) * N]`.
pub(super) struct PleRowsOut {
    pub(super) hyper: ExecutionTensor,
    pub(super) ring: ExecutionTensor,
}

/// Returns a `BadRequest` with `message`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// Runs the PLE block on `hyper` `[T, C, H]` with `emb` `[T, E]` and the ring from the previous token.
///
/// Row `t` sees the ring that the `t` earlier rows of the chunk would have left. A shape error
/// returns `BadRequest` before the first native call.
pub(super) fn block_rows(
    ctx: &mut Ctx<'_>,
    w: &RPle,
    shape: &ModelShape,
    emb: &ExecutionTensor,
    hyper: &ExecutionTensor,
    ring: &ExecutionTensor,
) -> Result<PleRowsOut, EngError> {
    let dims = shape.ple.dims();
    let branches = dim(shape.hyper.branches())?;
    let hidden = dim(shape.hyper.hidden())?;
    let width = dim(shape.hyper.width())?;
    let &[t, _] = emb.shape() else {
        return Err(bad("ple emb must be [T, E]"));
    };
    if hyper.shape() != [t, branches, hidden] {
        return Err(bad("ple hyper must be [T, C, H]"));
    }
    let consts = ctx.consts;
    // Keys [T, C, H] and value [T, 1, H] (ple.rs:300-303).
    let keys = ctx.kernels.qmm(emb, &w.key)?;
    let keys = ctx.kernels.reshape(&keys, &[t, branches, hidden])?;
    let value = ctx.kernels.qmm(emb, &w.value)?;
    let value = ctx.kernels.reshape(&value, &[t, 1, hidden])?;
    // Per-branch gate [T, C, 1] (ple.rs:310-314).
    let kn = rms0_rows(ctx, &keys, &w.norm_key1p, dims.eps)?;
    let qn = rms0_rows(ctx, hyper, &w.norm_query1p, dims.eps)?;
    let product = ctx.kernels.binary(BinaryOp::Mul, &kn, &qn)?;
    let dot = ctx.kernels.reduce(ReduceOp::Sum, &product, 2, true)?;
    let d = ctx
        .kernels
        .binary(BinaryOp::Div, &dot, &consts.sqrt_hidden)?;
    let g = gate(ctx, &d)?;
    // Gated [T, C, H] and normed [T, W] (ple.rs:315-319).
    let gated = ctx.kernels.binary(BinaryOp::Mul, &g, &value)?;
    let normed = rms0_rows(ctx, &gated, &w.norm_conv1p, dims.eps)?;
    let normed = ctx.kernels.reshape(&normed, &[t, width])?;
    // Convolution over time, taps N rows apart (ple.rs:322-331), then the residual update.
    let (pre, new_ring) = causal_conv(
        &mut *ctx.kernels,
        &normed,
        ring,
        &w.conv_taps,
        &w.conv_last,
        dims.ngram,
    )?;
    let conv = ctx.kernels.silu(&pre)?;
    let flat = ctx.kernels.reshape(&gated, &[t, width])?;
    let out = ctx.kernels.binary(BinaryOp::Add, &flat, &conv)?;
    let out = ctx.kernels.reshape(&out, &[t, branches, hidden])?;
    let new_hyper = ctx.kernels.binary(BinaryOp::Add, hyper, &out)?;
    Ok(PleRowsOut {
        hyper: new_hyper,
        ring: new_ring,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::fixtures;
    use super::super::super::ops::{values, zeros};
    use super::super::super::weights::ResidentWeights;
    use super::super::{assert_parity, import_f32};
    use super::*;
    use crate::family::flash_next::{load::FlashWeights, mlx::MlxProjector, ple};
    use crate::family::project::Projector;
    use crate::kernels::MlxKernels;
    use crate::weights::StreamedCheckpoint;

    /// Tokens that warm the PLE ring before the chunk. Nine tokens fill the nine ring columns.
    const WARM: [u32; 9] = [1, 2, 3, 4, 5, 6, 0, 1, 2];
    /// The chunk. Token 7 is the end-of-sequence token of the fixture and clears the history.
    const CHUNK: [u32; 4] = [3, 5, 7, 2];

    /// Hyper input of the token with 1-based global index `t`: `x[i] = (((i + 3t) % 7) - 3) * 0.125`.
    /// Branch 1 is zero when `t % 5 == 2`, so the gate input `d` is exactly 0 and the sign rule
    /// matters. In `WARM` followed by `CHUNK` this holds for `t = 2`, `7`, and `12`.
    fn hyper_at(t: usize) -> Vec<f32> {
        (0..128)
            .map(|i| {
                if t % 5 == 2 && i >= 64 {
                    0.0
                } else {
                    (((i + 3 * t) % 7) as f32 - 3.0) * 0.125
                }
            })
            .collect()
    }

    /// Host results for `CHUNK` after the host ran `WARM`.
    struct Reference {
        ring_in: Vec<f32>,
        hyper_in: Vec<f32>,
        want_emb: Vec<f32>,
        want_hyper: Vec<f32>,
        want_ring: Vec<f32>,
    }

    /// Runs the host PLE step over `WARM` and `CHUNK`. It records the ring before the first chunk
    /// token and, for each chunk token, the hyper input, the PLE embedding row, and the new hyper.
    fn reference(ckpt: &StreamedCheckpoint, flash: &FlashWeights) -> Reference {
        let view = flash.view();
        let shape = *flash.shape();
        let mut proj = MlxProjector::new(ckpt).unwrap();
        let mut state = ple::PleState::new(&shape.ple);
        let mut reference = Reference {
            ring_in: Vec::new(),
            hyper_in: Vec::new(),
            want_emb: Vec::new(),
            want_hyper: Vec::new(),
            want_ring: Vec::new(),
        };
        for (index, &token) in WARM.iter().chain(&CHUNK).enumerate() {
            if index == WARM.len() {
                reference.ring_in = state.ring().to_vec();
            }
            let x = hyper_at(index + 1);
            let plan = ple::plan_rows(&shape.ple, &view.ple, state.history(), token).unwrap();
            let update =
                ple::step(&mut proj, &shape.ple, &view.ple, &mut state, token, &x).unwrap();
            if index < WARM.len() {
                continue;
            }
            for &(table, local) in &plan.rows {
                let row = proj.rows(&view.ple.shards[table], &[local]).unwrap();
                reference
                    .want_emb
                    .extend(row.iter().map(|v| v * view.ple.weight_scale));
            }
            reference
                .want_hyper
                .extend(x.iter().zip(&update).map(|(h, u)| h + u));
            reference.hyper_in.extend(x);
        }
        reference.want_ring = state.ring().to_vec();
        reference
    }

    /// The four-token chunk starts from the ring that nine host steps left, so every tap of the
    /// convolution reads a nonzero value. By hand, the new ring holds the last nine columns of
    /// the old ring followed by the four new columns, so ring columns 0 to 4 of each channel equal
    /// the old columns 4 to 8 exactly.
    #[test]
    fn block_rows_match_the_host_block_from_a_warm_ring() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let want = reference(&ckpt, &flash);
        assert_eq!(want.ring_in.len(), 128 * 9, "host ring length");
        assert!(
            want.ring_in.iter().any(|v| *v != 0.0),
            "the warm ring is not all zero"
        );
        let hyper = import_f32(&mut kernels, vec![4, 2, 64], &want.hyper_in);
        let emb = import_f32(&mut kernels, vec![4, 64], &want.want_emb);
        let ring = import_f32(&mut kernels, vec![128, 9], &want.ring_in);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let out = block_rows(&mut ctx, &weights.ple, &weights.shape, &emb, &hyper, &ring).unwrap();

        assert_eq!(out.hyper.shape(), &[4, 2, 64][..], "hyper shape");
        assert_eq!(out.ring.shape(), &[128, 9][..], "ring shape");
        let got = values(&mut kernels, &out.hyper).unwrap();
        assert_parity(&got, &want.want_hyper, "hyper rows");
        let got_ring = values(&mut kernels, &out.ring).unwrap();
        assert_parity(&got_ring, &want.want_ring, "ring");
        for channel in 0..128 {
            assert_eq!(
                got_ring[channel * 9..channel * 9 + 5],
                want.ring_in[channel * 9 + 4..channel * 9 + 9],
                "channel {channel}: the old columns 4 to 8 shift to the front"
            );
        }
    }

    /// One chunk of 1 token and one of 3 tokens, with the ring carried, equal the host over 4 tokens.
    #[test]
    fn block_rows_chunks_that_carry_the_ring_equal_one_run() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let want = reference(&ckpt, &flash);
        let hyper_a = import_f32(&mut kernels, vec![1, 2, 64], &want.hyper_in[..128]);
        let emb_a = import_f32(&mut kernels, vec![1, 64], &want.want_emb[..64]);
        let hyper_b = import_f32(&mut kernels, vec![3, 2, 64], &want.hyper_in[128..]);
        let emb_b = import_f32(&mut kernels, vec![3, 64], &want.want_emb[64..]);
        let ring = import_f32(&mut kernels, vec![128, 9], &want.ring_in);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let first = block_rows(
            &mut ctx,
            &weights.ple,
            &weights.shape,
            &emb_a,
            &hyper_a,
            &ring,
        )
        .unwrap();
        let second = block_rows(
            &mut ctx,
            &weights.ple,
            &weights.shape,
            &emb_b,
            &hyper_b,
            &first.ring,
        )
        .unwrap();

        let got_a = values(&mut kernels, &first.hyper).unwrap();
        assert_parity(&got_a, &want.want_hyper[..128], "first chunk hyper");
        let got_b = values(&mut kernels, &second.hyper).unwrap();
        assert_parity(&got_b, &want.want_hyper[128..], "second chunk hyper");
        let got_ring = values(&mut kernels, &second.ring).unwrap();
        assert_parity(&got_ring, &want.want_ring, "ring after both chunks");
    }

    #[test]
    fn block_rows_reject_misshapen_inputs() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let emb = import_f32(&mut kernels, vec![4, 64], &[0.0; 256]);
        let flat_emb = import_f32(&mut kernels, vec![64], &[0.0; 64]);
        let hyper = import_f32(&mut kernels, vec![4, 2, 64], &[0.0; 512]);
        let short_hyper = import_f32(&mut kernels, vec![3, 2, 64], &[0.0; 384]);
        let ring = kernels
            .import_tensor(&zeros(vec![128, 9]).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let flat = block_rows(
            &mut ctx,
            &weights.ple,
            &weights.shape,
            &flat_emb,
            &hyper,
            &ring,
        );
        assert!(
            matches!(&flat, Err(EngError::BadRequest(m)) if m.contains("ple emb")),
            "rank 1 emb: {:?}",
            flat.as_ref().err()
        );
        let short = block_rows(
            &mut ctx,
            &weights.ple,
            &weights.shape,
            &emb,
            &short_hyper,
            &ring,
        );
        assert!(
            matches!(&short, Err(EngError::BadRequest(m)) if m.contains("ple hyper")),
            "hyper with 3 rows for 4 tokens: {:?}",
            short.as_ref().err()
        );
        let good = block_rows(&mut ctx, &weights.ple, &weights.shape, &emb, &hyper, &ring);
        assert!(good.is_ok(), "valid call: {:?}", good.as_ref().err());
    }
}
