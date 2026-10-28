//! Resident hyper-connection read, mixer read, and apply (packet C3a; S1-C plan section 2.1).
//!
//! The residual stays `[C, H]` for the whole step. These functions mirror `read` and `apply` in
//! `flash_next/hyper.rs`. They build lazy resident graphs and never read back or evaluate.

use super::super::hyper::HyperShape;
use super::ops::{Ctx, dim, rms0_rows};
use super::weights::RHyper;
use crate::{
    EngError,
    kernels::{BinaryOp, ReduceOp, ResidentPackedWeight, UnaryOp},
    tensor::ExecutionTensor,
};

/// Output of a resident hyper read: `mixed` `[1, H]` and, with an inject projection, `gate` `[C, 1]`.
pub(super) struct ResidentRead {
    pub(super) mixed: ExecutionTensor,
    pub(super) gate: Option<ExecutionTensor>,
}

/// Returns a `BadRequest` with `message`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// Checks that `h` is `[C, H]` and that the weights match `shape`, before any native call.
fn check_read(shape: &HyperShape, w: &RHyper, h: &ExecutionTensor) -> Result<(), EngError> {
    let c = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    if h.shape() != [c, hidden] {
        return Err(bad("resident hyper residual must be [C, H]"));
    }
    check_weights(shape, w)
}

/// Checks that the weights match `shape`, before any native call. The chunk form reuses it.
pub(super) fn check_weights(shape: &HyperShape, w: &RHyper) -> Result<(), EngError> {
    let c = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    let rank = dim(shape.rank())?;
    let width = dim(shape.width())?;
    if w.norm1p.shape() != [c, hidden]
        || w.down.logical_shape() != [rank, width]
        || w.up.logical_shape() != [width, rank]
        || w.inject
            .as_ref()
            .is_some_and(|i| i.logical_shape() != [c, width])
    {
        return Err(bad("resident hyper weights do not match the hyper shape"));
    }
    Ok(())
}

/// Plan 2.1 steps 1-5. Returns the flat normalized residual `nf` `[1, W]` and `mixed` `[1, H]`.
fn mix(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    h: &ExecutionTensor,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let branches = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    let width = dim(shape.width())?;
    // hyper.rs:102-110: rms0 of each branch row; `norm1p` holds `1 + w`.
    let n = rms0_rows(ctx, h, &w.norm1p, shape.eps())?;
    let consts = ctx.consts;
    let k = &mut *ctx.kernels;
    // hyper.rs:112-116: low = silu((n · downᵀ) / C).
    let nf = k.reshape(&n, &[1, width])?;
    let down = k.qmm(&nf, &w.down)?;
    let scaled = k.binary(BinaryOp::Div, &down, &consts.branches)?;
    let low = k.silu(&scaled)?;
    // hyper.rs:118-119: mixing = low · upᵀ, viewed as [C, H].
    let up = k.qmm(&low, &w.up)?;
    let mixing = k.reshape(&up, &[branches, hidden])?;
    // hyper.rs:120-127: mixed[d] = sum over b of sigmoid(mixing[b, d]) * n[b, d], divided by C.
    let weight = k.unary(UnaryOp::Sigmoid, &mixing)?;
    let product = k.binary(BinaryOp::Mul, &weight, &n)?;
    let sum = k.reduce(ReduceOp::Sum, &product, 0, true)?;
    let mixed = k.binary(BinaryOp::Div, &sum, &consts.branches)?;
    Ok((nf, mixed))
}

/// hyper.rs:129-140: gate = 2 * sigmoid((n · injectᵀ) / C), as `[C, 1]`.
fn inject_gate(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    inject: &ResidentPackedWeight,
    nf: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let branches = dim(shape.branches())?;
    let consts = ctx.consts;
    let k = &mut *ctx.kernels;
    let raw = k.qmm(nf, inject)?;
    let scaled = k.binary(BinaryOp::Div, &raw, &consts.branches)?;
    let unit = k.unary(UnaryOp::Sigmoid, &scaled)?;
    let doubled = k.binary(BinaryOp::Mul, &consts.two, &unit)?;
    k.reshape(&doubled, &[branches, 1])
}

/// Reads a hyper-connection (hyper.rs:89-143, plan 2.1 steps 1-6). `h` is the `[C, H]` residual.
pub(super) fn read(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    h: &ExecutionTensor,
) -> Result<ResidentRead, EngError> {
    check_read(shape, w, h)?;
    let (nf, mixed) = mix(ctx, shape, w, h)?;
    let gate = w
        .inject
        .as_ref()
        .map(|inject| inject_gate(ctx, shape, inject, &nf))
        .transpose()?;
    Ok(ResidentRead { mixed, gate })
}

/// Mixer read for the head (step.rs:244): `mixed` `[1, H]` only. It never computes an inject gate.
pub(super) fn mixer_read(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    h: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    check_read(shape, w, h)?;
    let (_nf, mixed) = mix(ctx, shape, w, h)?;
    Ok(mixed)
}

/// Returns `h + gate * block` as `[C, H]` (hyper.rs:145-163, plan 2.1 step 7).
pub(super) fn apply(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    h: &ExecutionTensor,
    gate: &ExecutionTensor,
    block: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let c = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    if h.shape() != [c, hidden] {
        return Err(bad("resident hyper residual must be [C, H]"));
    }
    if gate.shape() != [c, 1] {
        return Err(bad("resident hyper gate must be [C, 1]"));
    }
    if block.shape() != [1, hidden] {
        return Err(bad("resident hyper block must be [1, H]"));
    }
    let scaled = ctx.kernels.binary(BinaryOp::Mul, gate, block)?;
    ctx.kernels.binary(BinaryOp::Add, h, &scaled)
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{checkpoint, config};
    use super::super::ops::{f32_tensor, values};
    use super::super::weights::ResidentWeights;
    use super::*;
    use crate::family::flash_next::{hyper as host, load::FlashWeights, mlx::MlxProjector};
    use crate::kernels::MlxKernels;

    /// The `[2, 64]` residual of every parity test: `x[i] = ((i % 5) - 1) * 1e-7`.
    ///
    /// The fixture hyper weights are `c % 8 + bias` in every row, so a residual of unit scale
    /// saturates every sigmoid. At this scale `rms0` is dominated by eps (1e-6), the sums have
    /// mostly positive terms, the host gates are near 1.017, and `max |mixed|` is near 3.5e-4.
    fn residual() -> Vec<f32> {
        (0..128).map(|i| ((i % 5) as f32 - 1.0) * 1.0e-7).collect()
    }

    /// Imports host values as a resident F32 tensor of `shape`.
    fn resident(kernels: &mut MlxKernels, shape: Vec<u32>, data: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(shape, data).unwrap())
            .unwrap()
    }

    /// Asserts cosine >= 0.99999 and max |a - e| / max |e| <= 1e-4 (plan section 4, option B).
    fn assert_parity(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        let (mut dot, mut aa, mut ee, mut err, mut scale) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (a, e) in actual.iter().zip(expected) {
            let (a, e) = (f64::from(*a), f64::from(*e));
            dot += a * e;
            aa += a * a;
            ee += e * e;
            err = err.max((a - e).abs());
            scale = scale.max(e.abs());
        }
        assert!(scale > 0.0, "{what}: expected values are all zero");
        let cos = dot / (aa.sqrt() * ee.sqrt());
        let rel = err / scale;
        assert!(cos >= 0.99999, "{what}: cos {cos}");
        assert!(rel <= 1e-4, "{what}: rel {rel}");
    }

    #[test]
    fn read_with_inject_matches_the_host_block() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let x = residual();
        let want = host::read(&mut proj, &shape.hyper, &view.layers[0].attn_hyper, &x).unwrap();
        let want_gate = want.gate.unwrap();
        assert!(
            want_gate.iter().all(|g| (1.0..1.1).contains(g)),
            "host gate regime {want_gate:?}"
        );
        let h = resident(&mut kernels, vec![2, 64], &x);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let got = read(&mut ctx, &shape.hyper, &weights.layers[0].attn_hyper, &h).unwrap();
        assert_eq!(got.mixed.shape(), &[1, 64][..], "mixed shape");
        let gate = got.gate.unwrap();
        assert_eq!(gate.shape(), &[2, 1][..], "gate shape");
        assert_parity(
            &values(&mut kernels, &got.mixed).unwrap(),
            &want.mixed,
            "mixed",
        );
        assert_parity(&values(&mut kernels, &gate).unwrap(), &want_gate, "gate");
    }

    #[test]
    fn read_without_inject_matches_the_host_block() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let x = residual();
        let want = host::read(&mut proj, &shape.hyper, &view.mixer, &x).unwrap();
        assert!(want.gate.is_none(), "host mixer has no gate");
        assert!(
            want.mixed.iter().any(|v| v.abs() > 1.0e-4),
            "host mixed regime {:?}",
            want.mixed
        );
        let h = resident(&mut kernels, vec![2, 64], &x);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let got = read(&mut ctx, &shape.hyper, &weights.mixer, &h).unwrap();
        assert!(got.gate.is_none(), "resident mixer has no gate");
        assert_eq!(got.mixed.shape(), &[1, 64][..], "mixed shape");
        assert_parity(
            &values(&mut kernels, &got.mixed).unwrap(),
            &want.mixed,
            "mixed",
        );
    }

    #[test]
    fn mixer_read_matches_the_host_head_input() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let x = residual();
        let want = host::read(&mut proj, &shape.hyper, &view.mixer, &x)
            .unwrap()
            .mixed;
        let h = resident(&mut kernels, vec![2, 64], &x);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let mixed = mixer_read(&mut ctx, &shape.hyper, &weights.mixer, &h).unwrap();
        assert_eq!(mixed.shape(), &[1, 64][..], "mixed shape");
        assert_parity(&values(&mut kernels, &mixed).unwrap(), &want, "mixed");
    }

    #[test]
    fn apply_chain_matches_the_host_blocks_over_two_reads() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let block1: Vec<f32> = (0..64).map(|d| ((d % 3) as f32 - 1.0) * 1.0e-7).collect();
        let block2: Vec<f32> = (0..64).map(|d| ((d % 4) as f32 - 1.5) * 1.0e-7).collect();
        // Host chain: attention hyper read and apply, then MLP hyper read and apply.
        let mut host_h = residual();
        let read1 =
            host::read(&mut proj, &shape.hyper, &view.layers[0].attn_hyper, &host_h).unwrap();
        host::apply(&shape.hyper, &mut host_h, &read1.gate.unwrap(), &block1).unwrap();
        let host_h1 = host_h.clone();
        let read2 =
            host::read(&mut proj, &shape.hyper, &view.layers[0].mlp_hyper, &host_h).unwrap();
        let host_gate2 = read2.gate.unwrap();
        host::apply(&shape.hyper, &mut host_h, &host_gate2, &block2).unwrap();
        // Resident chain on the same inputs.
        let h0 = resident(&mut kernels, vec![2, 64], &residual());
        let b1 = resident(&mut kernels, vec![1, 64], &block1);
        let b2 = resident(&mut kernels, vec![1, 64], &block2);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let got1 = read(&mut ctx, &shape.hyper, &weights.layers[0].attn_hyper, &h0).unwrap();
        let h1 = apply(&mut ctx, &shape.hyper, &h0, &got1.gate.unwrap(), &b1).unwrap();
        let got2 = read(&mut ctx, &shape.hyper, &weights.layers[0].mlp_hyper, &h1).unwrap();
        let gate2 = got2.gate.unwrap();
        let h2 = apply(&mut ctx, &shape.hyper, &h1, &gate2, &b2).unwrap();
        assert_eq!(h2.shape(), &[2, 64][..], "residual shape");
        assert_parity(
            &values(&mut kernels, &h1).unwrap(),
            &host_h1,
            "residual after step 1",
        );
        assert_parity(
            &values(&mut kernels, &got2.mixed).unwrap(),
            &read2.mixed,
            "mixed of step 2",
        );
        assert_parity(
            &values(&mut kernels, &gate2).unwrap(),
            &host_gate2,
            "gate of step 2",
        );
        assert_parity(
            &values(&mut kernels, &h2).unwrap(),
            &host_h,
            "residual after step 2",
        );
    }

    #[test]
    fn read_rejects_a_residual_that_is_not_c_by_h() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = *flash.shape();
        let bad_h = resident(&mut kernels, vec![1, 64], &[1.0e-7; 64]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let result = read(
            &mut ctx,
            &shape.hyper,
            &weights.layers[0].attn_hyper,
            &bad_h,
        );
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "read: {:?}",
            result.as_ref().err()
        );
        let result = mixer_read(&mut ctx, &shape.hyper, &weights.mixer, &bad_h);
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "mixer read: {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn apply_rejects_a_gate_or_block_of_the_wrong_shape() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let shape = *flash.shape();
        let h = resident(&mut kernels, vec![2, 64], &residual());
        let gate = resident(&mut kernels, vec![2, 1], &[1.0, 1.0]);
        let bad_gate = resident(&mut kernels, vec![1, 1], &[1.0]);
        let block = resident(&mut kernels, vec![1, 64], &[1.0e-7; 64]);
        let bad_block = resident(&mut kernels, vec![1, 1], &[1.0e-7]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let result = apply(&mut ctx, &shape.hyper, &h, &bad_gate, &block);
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "gate [1, 1]: {:?}",
            result.as_ref().err()
        );
        let result = apply(&mut ctx, &shape.hyper, &h, &gate, &bad_block);
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "block [1, 1]: {:?}",
            result.as_ref().err()
        );
        let good = apply(&mut ctx, &shape.hyper, &h, &gate, &block).unwrap();
        assert_eq!(good.shape(), &[2, 64][..], "valid apply shape");
    }
}
