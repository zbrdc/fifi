//! Resident PLE: the token-start row import and the PLE block.
//!
//! Hashing and row reads stay on the host. The embedding row and the PLE rows import packed with one
//! synchronization and dequantize on the device. The block performs no synchronization.

use super::super::{
    ple::{self, PleWeights},
    step::ModelShape,
    stored::{self, SlicedWeight},
};
use super::ops::{Ctx, dim, linear, rms0_rows, short_import};
use super::weights::{RPle, ResidentWeights};
use crate::{
    EngError,
    kernels::{BinaryOp, PackedWeight, ReduceOp, UnaryOp},
    tensor::ExecutionTensor,
    weights::StreamedCheckpoint,
};

/// Device inputs of one token and the PLE history after it.
pub(super) struct TokenStart {
    /// Hyper state `[C, H]`: the embedding row repeated over the branches (step.rs:180-185).
    pub(super) hyper: ExecutionTensor,
    /// Scaled PLE embedding `[1, heads * head_dim]` (ple.rs:291-297).
    pub(super) emb: ExecutionTensor,
    /// The n-gram history after the token; the step commits it.
    pub(super) history: Vec<u32>,
}

/// Output of the PLE block: the updated hyper state `[C, H]` and the new ring `[W, (PCK - 1) * N]`.
pub(super) struct PleOut {
    /// `hyper + update` (step.rs:207-210).
    pub(super) hyper: ExecutionTensor,
    /// The shifted convolution ring (ple.rs:333-339).
    pub(super) ring: ExecutionTensor,
}

/// A `PleWeights` view for `ple::plan_rows`, which reads only the hash tables and the shard rows
/// (ple.rs:207-237). The projection, norm, and convolution fields are unread placeholders.
pub(super) fn table_view(w: &RPle) -> Result<PleWeights<'_, SlicedWeight>, EngError> {
    let first = w
        .shards
        .first()
        .ok_or_else(|| EngError::Checkpoint("ple has no row tables".into()))?;
    Ok(PleWeights {
        multipliers: &w.multipliers,
        vocab: &w.vocab,
        offsets: &w.offsets,
        shards: &w.shards,
        shard_rows: &w.shard_rows,
        weight_scale: 0.0,
        key: first,
        value: first,
        norm_key: &[],
        norm_query: &[],
        norm_conv: &[],
        conv: &[],
    })
}

/// Plans and reads the embedding row and the PLE rows of `token`, then imports them with one synchronization.
pub(super) fn import_token(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    weights: &ResidentWeights,
    history: &[u32],
    token: u32,
) -> Result<TokenStart, EngError> {
    let plan = ple::plan_rows(
        &weights.shape.ple,
        &table_view(&weights.ple)?,
        history,
        token,
    )?;
    let mut host: Vec<PackedWeight> = Vec::with_capacity(plan.rows.len());
    host.push(stored::read_slice(ckpt, &weights.embed, token)?);
    for &(table, local) in &plan.rows {
        let shard = weights
            .ple
            .shards
            .get(table)
            .ok_or_else(|| EngError::Checkpoint("ple row table is missing".into()))?;
        host.push(stored::read_slice(ckpt, shard, local)?);
    }
    let refs: Vec<&PackedWeight> = host.iter().collect();
    let residents = ctx.kernels.import_packed_many(&refs)?;
    let (embed_row, rows) = residents.split_first().ok_or_else(short_import)?;
    let row = ctx.kernels.dequantize(embed_row)?;
    let branches = dim(weights.shape.hyper.branches())?;
    let hyper = ctx.kernels.repeat(&row, branches, 0)?;
    let mut parts = Vec::with_capacity(rows.len());
    for part in rows {
        parts.push(ctx.kernels.dequantize(part)?);
    }
    let part_refs: Vec<&ExecutionTensor> = parts.iter().collect();
    let joined = ctx.kernels.concatenate(&part_refs, 1)?;
    let consts = ctx.consts;
    let emb = ctx
        .kernels
        .binary(BinaryOp::Mul, &joined, &consts.ple_scale)?;
    Ok(TokenStart {
        hyper,
        emb,
        history: plan.history,
    })
}

/// `sigmoid(s * sqrt(max(|d|, 1e-6)))` with `s = sign(sign(d) + 0.5)`: `d = 0` gives `s = +1`, as the
/// host `copysign` does (ple.rs:313-314; plan finding 0.4).
pub(super) fn gate(ctx: &mut Ctx<'_>, d: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
    let consts = ctx.consts;
    let sign = ctx.kernels.unary(UnaryOp::Sign, d)?;
    let shifted = ctx.kernels.binary(BinaryOp::Add, &sign, &consts.half)?;
    let sign = ctx.kernels.unary(UnaryOp::Sign, &shifted)?;
    let magnitude = ctx.kernels.unary(UnaryOp::Abs, d)?;
    let magnitude = ctx
        .kernels
        .binary(BinaryOp::Maximum, &magnitude, &consts.eps6)?;
    let magnitude = ctx.kernels.unary(UnaryOp::Sqrt, &magnitude)?;
    let shaped = ctx.kernels.binary(BinaryOp::Mul, &sign, &magnitude)?;
    ctx.kernels.unary(UnaryOp::Sigmoid, &shaped)
}

/// Runs the PLE block on `hyper` `[C, H]` with `emb` and `ring`, and returns the new hyper state and ring.
pub(super) fn block(
    ctx: &mut Ctx<'_>,
    w: &RPle,
    shape: &ModelShape,
    emb: &ExecutionTensor,
    hyper: &ExecutionTensor,
    ring: &ExecutionTensor,
) -> Result<PleOut, EngError> {
    let dims = shape.ple.dims();
    let branches = dim(shape.hyper.branches())?;
    let hidden = dim(shape.hyper.hidden())?;
    let width = dim(shape.hyper.width())?;
    let overflow = || EngError::BadRequest("ple ring width overflows".into());
    let taps = dims.conv_kernel.checked_sub(1).ok_or_else(overflow)?;
    let stride = taps.checked_mul(dims.ngram).ok_or_else(overflow)?;
    let consts = ctx.consts;
    // Keys [C, H] and value [1, H] (ple.rs:300-303).
    let keys = linear(ctx, &w.key, emb)?;
    let keys = ctx.kernels.reshape(&keys, &[branches, hidden])?;
    let value = linear(ctx, &w.value, emb)?;
    // Per-branch gate [C, 1] (ple.rs:310-314).
    let kn = rms0_rows(ctx, &keys, &w.norm_key1p, dims.eps)?;
    let qn = rms0_rows(ctx, hyper, &w.norm_query1p, dims.eps)?;
    let product = ctx.kernels.binary(BinaryOp::Mul, &kn, &qn)?;
    let dot = ctx.kernels.reduce(ReduceOp::Sum, &product, 1, true)?;
    let d = ctx
        .kernels
        .binary(BinaryOp::Div, &dot, &consts.sqrt_hidden)?;
    let g = gate(ctx, &d)?;
    // Gated [C, H] and normed [W] (ple.rs:315-319).
    let gated = ctx.kernels.binary(BinaryOp::Mul, &g, &value)?;
    let normed = rms0_rows(ctx, &gated, &w.norm_conv1p, dims.eps)?;
    let normed = ctx.kernels.reshape(&normed, &[width])?;
    // Taps at ring columns k * ngram, then the convolution (ple.rs:322-331).
    let ring3 = ctx.kernels.reshape(ring, &[width, taps, dims.ngram])?;
    let old = ctx.kernels.slice(&ring3, &[0, 0, 0], &[width, taps, 1])?;
    let old = ctx.kernels.reshape(&old, &[width, taps])?;
    let weighted = ctx.kernels.binary(BinaryOp::Mul, &w.conv_taps, &old)?;
    let past = ctx.kernels.reduce(ReduceOp::Sum, &weighted, 1, false)?;
    let current = ctx.kernels.binary(BinaryOp::Mul, &w.conv_last, &normed)?;
    let total = ctx.kernels.binary(BinaryOp::Add, &current, &past)?;
    let conv = ctx.kernels.silu(&total)?;
    let flat = ctx.kernels.reshape(&gated, &[width])?;
    let out = ctx.kernels.binary(BinaryOp::Add, &flat, &conv)?;
    let out = ctx.kernels.reshape(&out, &[branches, hidden])?;
    let new_hyper = ctx.kernels.binary(BinaryOp::Add, hyper, &out)?;
    // Shift the ring by one column and append `normed` (ple.rs:333-339).
    let kept = ctx.kernels.slice(ring, &[0, 1], &[width, stride])?;
    let column = ctx.kernels.reshape(&normed, &[width, 1])?;
    let new_ring = ctx.kernels.concatenate(&[&kept, &column], 1)?;
    Ok(PleOut {
        hyper: new_hyper,
        ring: new_ring,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::{load::FlashWeights, mlx::MlxProjector};
    use super::super::fixtures;
    use super::super::ops::{assert_close, f32_tensor, values, zeros};
    use super::*;
    use crate::family::project::Projector;
    use crate::kernels::MlxKernels;

    /// Asserts cosine >= 0.99999 and max relative error <= 1e-4.
    fn assert_parity(got: &[f32], want: &[f32], what: &str) {
        let (mut err, mut scale, mut dot, mut gg, mut ww) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&g, &w) in got.iter().zip(want) {
            let (g, w) = (f64::from(g), f64::from(w));
            err = err.max((g - w).abs());
            scale = scale.max(w.abs());
            dot += g * w;
            gg += g * g;
            ww += w * w;
        }
        let rel = if scale == 0.0 { err } else { err / scale };
        let cos = if gg == 0.0 && ww == 0.0 {
            1.0
        } else {
            dot / (gg.sqrt() * ww.sqrt())
        };
        assert!(
            rel <= 1e-4 && cos >= 0.99999,
            "{what}: rel {rel:e}, cos {cos}"
        );
    }

    /// Hyper input of step `t`: `x[i] = (((i + 3t) % 7) - 3) * 0.125`. Branch 1 is zero at step 2, so `d = 0`.
    fn hyper_at(t: usize) -> Vec<f32> {
        (0..128)
            .map(|i| {
                if t == 2 && i >= 64 {
                    0.0
                } else {
                    (((i + 3 * t) % 7) as f32 - 3.0) * 0.125
                }
            })
            .collect()
    }

    #[test]
    fn import_token_matches_the_host_rows() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let view = flash.view();
        let shape = *flash.shape();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let start = import_token(&mut ctx, &ckpt, &weights, &[3, 5], 2).unwrap();
        let plan = ple::plan_rows(&shape.ple, &view.ple, &[3, 5], 2).unwrap();
        assert_eq!(plan.rows, [(0, 7), (0, 10)], "host rows");
        assert_eq!(start.history, plan.history, "history equals the host plan");
        assert_eq!(start.history, [5, 2], "history");
        assert_eq!(start.hyper.shape(), &[2, 64][..], "hyper shape");
        assert_eq!(start.emb.shape(), &[1, 64][..], "emb shape");
        let hyper = values(&mut kernels, &start.hyper).unwrap();
        let want_hyper: Vec<f32> = (0..128).map(|i| (i % 8) as f32 + 0.5).collect();
        assert_close(&hyper, &want_hyper, "hyper hand values");
        let row = proj.rows(view.embed, &[2]).unwrap();
        let host_hyper: Vec<f32> = row.iter().chain(&row).copied().collect();
        assert_parity(&hyper, &host_hyper, "hyper against host");
        let emb = values(&mut kernels, &start.emb).unwrap();
        let want_emb: Vec<f32> = (0..64)
            .map(|j| 0.5 * ((j % 8) as f32 + if j < 32 { 0.75 } else { 0.5 }))
            .collect();
        assert_close(&emb, &want_emb, "emb hand values");
        let mut host_emb = Vec::new();
        for &(table, local) in &plan.rows {
            let r = proj.rows(&view.ple.shards[table], &[local]).unwrap();
            host_emb.extend(r.iter().map(|v| v * view.ple.weight_scale));
        }
        assert_parity(&emb, &host_emb, "emb against host");
    }

    #[test]
    fn gate_keeps_the_host_sign_at_zero() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let d_values = [0.0f32, 1e-8, -1e-8, 0.25, -2.0];
        let d = kernels
            .import_tensor(&f32_tensor(vec![5, 1], &d_values).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let g = gate(&mut ctx, &d).unwrap();
        let got = values(&mut kernels, &g).unwrap();
        assert_eq!(got.len(), 5, "gate length");
        for (i, (&x, &a)) in d_values.iter().zip(&got).enumerate() {
            let x = f64::from(x);
            let shaped = x.abs().max(1e-6).sqrt().copysign(x);
            let want = (1.0 / (1.0 + (-shaped).exp())) as f32;
            assert!(
                (a - want).abs() <= 1e-6,
                "gate {i} (d = {x}): actual {a}, expected {want}"
            );
        }
    }

    #[test]
    fn four_steps_match_the_host_block() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let view = flash.view();
        let shape = *flash.shape();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut host_state = ple::PleState::new(&shape.ple);
        let mut ring = kernels
            .import_tensor(&zeros(vec![128, 9]).unwrap())
            .unwrap();
        let mut history: Vec<u32> = Vec::new();
        for (index, token) in [3u32, 5, 7, 2].into_iter().enumerate() {
            let t = index + 1;
            let hyper_in = hyper_at(t);
            let update = ple::step(
                &mut proj,
                &shape.ple,
                &view.ple,
                &mut host_state,
                token,
                &hyper_in,
            )
            .unwrap();
            let want: Vec<f32> = hyper_in.iter().zip(&update).map(|(h, u)| h + u).collect();
            let hyper = kernels
                .import_tensor(&f32_tensor(vec![2, 64], &hyper_in).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let start = import_token(&mut ctx, &ckpt, &weights, &history, token).unwrap();
            let out = block(
                &mut ctx,
                &weights.ple,
                &weights.shape,
                &start.emb,
                &hyper,
                &ring,
            )
            .unwrap();
            history = start.history;
            ring = out.ring;
            assert_eq!(out.hyper.shape(), &[2, 64][..], "step {t} hyper shape");
            assert_eq!(ring.shape(), &[128, 9][..], "step {t} ring shape");
            let got = values(&mut kernels, &out.hyper).unwrap();
            assert_parity(&got, &want, &format!("step {t} hyper"));
            let got_ring = values(&mut kernels, &ring).unwrap();
            assert_parity(&got_ring, host_state.ring(), &format!("step {t} ring"));
            assert_eq!(history.as_slice(), host_state.history(), "step {t} history");
        }
        assert_eq!(history, [2], "history after eos and token 2");
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn token_start_takes_one_synchronization_and_the_block_none() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let ring = kernels
            .import_tensor(&zeros(vec![128, 9]).unwrap())
            .unwrap();
        let hyper = kernels
            .import_tensor(&f32_tensor(vec![2, 64], &hyper_at(1)).unwrap())
            .unwrap();
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let start = import_token(&mut ctx, &ckpt, &weights, &[], 3).unwrap();
        let after_import = ctx.kernels.qualification_sync_count();
        let out = block(
            &mut ctx,
            &weights.ple,
            &weights.shape,
            &start.emb,
            &hyper,
            &ring,
        )
        .unwrap();
        let after_block = ctx.kernels.qualification_sync_count();
        assert_eq!(after_import, before + 1, "token start synchronizations");
        assert_eq!(after_block, after_import, "the block stays lazy");
        assert_eq!(out.ring.shape(), &[128, 9][..], "ring shape");
    }
}
