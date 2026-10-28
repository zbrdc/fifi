//! Batched hyper-connection read and apply (plan S4 section 3.1).
//!
//! The residual of a chunk is `[T, C, H]`. These functions mirror `read` and `apply` in the parent
//! module with a leading `T` axis. They build lazy graphs and never read back or evaluate.

use super::super::{
    hyper::check_weights,
    ops::{Ctx, dim, rms0_rows},
    weights::RHyper,
};
use crate::{
    EngError,
    family::flash_next::hyper::HyperShape,
    kernels::{BinaryOp, ReduceOp, ResidentPackedWeight, UnaryOp},
    tensor::ExecutionTensor,
};

/// Output of a batched hyper read: `mixed` `[T, H]` and, with an inject projection, `gate` `[T, C, 1]`.
pub(super) struct RowsRead {
    pub(super) mixed: ExecutionTensor,
    pub(super) gate: Option<ExecutionTensor>,
}

/// Returns a `BadRequest` with `message`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// Checks that `h` is `[T, C, H]` with `T >= 1` and returns `T`.
fn check_rows(shape: &HyperShape, h: &ExecutionTensor) -> Result<u32, EngError> {
    let c = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    let &[t, branches, width] = h.shape() else {
        return Err(bad("resident hyper residual must be [T, C, H]"));
    };
    if t == 0 || branches != c || width != hidden {
        return Err(bad("resident hyper residual must be [T, C, H]"));
    }
    Ok(t)
}

/// Returns the flat normalized residual `nf` `[T, W]` and `mixed` `[T, H]` (`mix` in the parent module).
fn mix_rows(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    h: &ExecutionTensor,
    t: u32,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let branches = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    let width = dim(shape.width())?;
    let n = rms0_rows(ctx, h, &w.norm1p, shape.eps())?;
    let consts = ctx.consts;
    let k = &mut *ctx.kernels;
    let nf = k.reshape(&n, &[t, width])?;
    let down = k.qmm(&nf, &w.down)?;
    let scaled = k.binary(BinaryOp::Div, &down, &consts.branches)?;
    let low = k.silu(&scaled)?;
    let up = k.qmm(&low, &w.up)?;
    let mixing = k.reshape(&up, &[t, branches, hidden])?;
    let weight = k.unary(UnaryOp::Sigmoid, &mixing)?;
    let product = k.binary(BinaryOp::Mul, &weight, &n)?;
    let sum = k.reduce(ReduceOp::Sum, &product, 1, false)?;
    let mixed = k.binary(BinaryOp::Div, &sum, &consts.branches)?;
    Ok((nf, mixed))
}

/// gate = 2 * sigmoid((n · injectᵀ) / C), as `[T, C, 1]` (`inject_gate` in the parent module).
fn inject_gate_rows(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    inject: &ResidentPackedWeight,
    nf: &ExecutionTensor,
    t: u32,
) -> Result<ExecutionTensor, EngError> {
    let branches = dim(shape.branches())?;
    let consts = ctx.consts;
    let k = &mut *ctx.kernels;
    let raw = k.qmm(nf, inject)?;
    let scaled = k.binary(BinaryOp::Div, &raw, &consts.branches)?;
    let unit = k.unary(UnaryOp::Sigmoid, &scaled)?;
    let doubled = k.binary(BinaryOp::Mul, &consts.two, &unit)?;
    k.reshape(&doubled, &[t, branches, 1])
}

/// Reads a hyper-connection for every row of the chunk residual `h` `[T, C, H]`.
///
/// Without an inject projection (the root mixer) `gate` is `None`. A shape error returns
/// `BadRequest` before the first native call.
pub(super) fn read_rows(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    w: &RHyper,
    h: &ExecutionTensor,
) -> Result<RowsRead, EngError> {
    check_weights(shape, w)?;
    let t = check_rows(shape, h)?;
    let (nf, mixed) = mix_rows(ctx, shape, w, h, t)?;
    let gate = w
        .inject
        .as_ref()
        .map(|inject| inject_gate_rows(ctx, shape, inject, &nf, t))
        .transpose()?;
    Ok(RowsRead { mixed, gate })
}

/// Returns `h + gate * block` as `[T, C, H]` for `h` `[T, C, H]`, `gate` `[T, C, 1]`, and `block` `[T, H]`.
pub(super) fn apply_rows(
    ctx: &mut Ctx<'_>,
    shape: &HyperShape,
    h: &ExecutionTensor,
    gate: &ExecutionTensor,
    block: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let t = check_rows(shape, h)?;
    let c = dim(shape.branches())?;
    let hidden = dim(shape.hidden())?;
    if gate.shape() != [t, c, 1] {
        return Err(bad("resident hyper gate must be [T, C, 1]"));
    }
    if block.shape() != [t, hidden] {
        return Err(bad("resident hyper block must be [T, H]"));
    }
    let block3 = ctx.kernels.reshape(block, &[t, 1, hidden])?;
    let scaled = ctx.kernels.binary(BinaryOp::Mul, gate, &block3)?;
    ctx.kernels.binary(BinaryOp::Add, h, &scaled)
}

#[cfg(test)]
mod tests {
    use super::super::super::fixtures::{checkpoint, config};
    use super::super::super::ops::values;
    use super::super::super::weights::ResidentWeights;
    use super::super::{assert_parity, import_f32};
    use super::*;
    use crate::family::flash_next::{hyper as host, load::FlashWeights, mlx::MlxProjector};
    use crate::kernels::MlxKernels;
    use crate::weights::StreamedCheckpoint;

    /// Rows of the chunk residual: `x[i] = (((i + 2t) % 5) - 1) * 1e-7 * (t + 1)`. The decode tests use
    /// this scale because at larger scales the fixture's sigmoids saturate. Each row differs in shift and amplitude.
    fn residual_row(t: usize) -> Vec<f32> {
        (0..128)
            .map(|i| (((i + 2 * t) % 5) as f32 - 1.0) * 1.0e-7 * (t + 1) as f32)
            .collect()
    }

    /// Block rows: `b[d] = (((d + t) % 3) - 1) * 1e-7`.
    fn block_row(t: usize) -> Vec<f32> {
        (0..64)
            .map(|d| (((d + t) % 3) as f32 - 1.0) * 1.0e-7)
            .collect()
    }

    const ROWS: usize = 3;

    /// Loads the fixture checkpoint and imports its resident weights.
    fn load(kernels: &mut MlxKernels) -> (StreamedCheckpoint, FlashWeights, ResidentWeights) {
        let ckpt = checkpoint();
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(kernels, &flash).unwrap();
        (ckpt, flash, weights)
    }

    fn stacked(rows: impl Fn(usize) -> Vec<f32>) -> Vec<f32> {
        (0..ROWS).flat_map(rows).collect()
    }

    #[test]
    fn read_rows_with_inject_match_the_host_per_row() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, flash, weights) = load(&mut kernels);
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let (mut want_mixed, mut want_gate) = (Vec::new(), Vec::new());
        for t in 0..ROWS {
            let want = host::read(
                &mut proj,
                &shape.hyper,
                &view.layers[0].attn_hyper,
                &residual_row(t),
            )
            .unwrap();
            assert!(
                want.mixed.iter().any(|v| v.abs() > 1.0e-4),
                "host mixed regime of row {t}: {:?}",
                want.mixed
            );
            want_mixed.extend(want.mixed);
            want_gate.extend(want.gate.unwrap());
        }
        assert_ne!(want_mixed[..64], want_mixed[128..], "rows 0 and 2 differ");
        let h = import_f32(&mut kernels, vec![3, 2, 64], &stacked(residual_row));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let got = read_rows(&mut ctx, &shape.hyper, &weights.layers[0].attn_hyper, &h).unwrap();

        assert_eq!(got.mixed.shape(), &[3, 64][..], "mixed shape");
        let gate = got.gate.unwrap();
        assert_eq!(gate.shape(), &[3, 2, 1][..], "gate shape");
        assert_parity(
            &values(&mut kernels, &got.mixed).unwrap(),
            &want_mixed,
            "mixed",
        );
        assert_parity(&values(&mut kernels, &gate).unwrap(), &want_gate, "gate");
    }

    #[test]
    fn read_rows_without_inject_match_the_host_mixer() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, flash, weights) = load(&mut kernels);
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut want_mixed = Vec::new();
        for t in 0..ROWS {
            let want = host::read(&mut proj, &shape.hyper, &view.mixer, &residual_row(t)).unwrap();
            assert!(want.gate.is_none(), "host mixer has no gate");
            assert!(
                want.mixed.iter().any(|v| v.abs() > 1.0e-4),
                "host mixed regime of row {t}: {:?}",
                want.mixed
            );
            want_mixed.extend(want.mixed);
        }
        let h = import_f32(&mut kernels, vec![3, 2, 64], &stacked(residual_row));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let got = read_rows(&mut ctx, &shape.hyper, &weights.mixer, &h).unwrap();

        assert!(got.gate.is_none(), "resident mixer has no gate");
        assert_eq!(got.mixed.shape(), &[3, 64][..], "mixed shape");
        assert_parity(
            &values(&mut kernels, &got.mixed).unwrap(),
            &want_mixed,
            "mixed",
        );
    }

    #[test]
    fn apply_rows_match_the_host_apply_per_row() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ckpt, flash, weights) = load(&mut kernels);
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut want_h = Vec::new();
        for t in 0..ROWS {
            let x = residual_row(t);
            let read = host::read(&mut proj, &shape.hyper, &view.layers[0].attn_hyper, &x).unwrap();
            let mut host_h = x;
            host::apply(
                &shape.hyper,
                &mut host_h,
                &read.gate.unwrap(),
                &block_row(t),
            )
            .unwrap();
            want_h.extend(host_h);
        }
        let h = import_f32(&mut kernels, vec![3, 2, 64], &stacked(residual_row));
        let block = import_f32(&mut kernels, vec![3, 64], &stacked(block_row));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let read = read_rows(&mut ctx, &shape.hyper, &weights.layers[0].attn_hyper, &h).unwrap();

        let next = apply_rows(&mut ctx, &shape.hyper, &h, &read.gate.unwrap(), &block).unwrap();

        assert_eq!(next.shape(), &[3, 2, 64][..], "residual shape");
        assert_parity(&values(&mut kernels, &next).unwrap(), &want_h, "residual");
    }

    /// `h[t][c][d] = 100 t + 10 c + d`, `gate[t][c] = 2 t + c + 1`, and `block[t][d] = 2 d + t`.
    /// By hand: `(t, c, d) = (1, 1, 3)` is `113 + 4 * 7 = 141`, `(0, 1, 5)` is `15 + 2 * 10 = 35`,
    /// and `(1, 0, 0)` is `100 + 3 * 1 = 103`. Every value is an exact small integer in F32.
    #[test]
    fn apply_rows_match_the_hand_values() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (_ckpt, flash, weights) = load(&mut kernels);
        let shape = *flash.shape();
        let (mut h_values, mut gate_values, mut block_values) =
            (Vec::new(), Vec::new(), Vec::new());
        for t in 0..2usize {
            for c in 0..2usize {
                gate_values.push((2 * t + c + 1) as f32);
                h_values.extend((0..64usize).map(|d| (100 * t + 10 * c + d) as f32));
            }
            block_values.extend((0..64usize).map(|d| (2 * d + t) as f32));
        }
        let h = import_f32(&mut kernels, vec![2, 2, 64], &h_values);
        let gate = import_f32(&mut kernels, vec![2, 2, 1], &gate_values);
        let block = import_f32(&mut kernels, vec![2, 64], &block_values);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };

        let next = apply_rows(&mut ctx, &shape.hyper, &h, &gate, &block).unwrap();

        let got = values(&mut kernels, &next).unwrap();
        assert_eq!(got.len(), 256, "length");
        assert_eq!(got[3 * 64 + 3], 141.0, "(t, c, d) = (1, 1, 3)");
        assert_eq!(got[64 + 5], 35.0, "(t, c, d) = (0, 1, 5)");
        assert_eq!(got[2 * 64], 103.0, "(t, c, d) = (1, 0, 0)");
        for t in 0..2usize {
            for c in 0..2usize {
                for d in 0..64usize {
                    let want =
                        (100 * t + 10 * c + d) as f32 + (2 * t + c + 1) as f32 * (2 * d + t) as f32;
                    assert_eq!(
                        got[(t * 2 + c) * 64 + d],
                        want,
                        "(t, c, d) = ({t}, {c}, {d})"
                    );
                }
            }
        }
    }

    #[test]
    fn rows_functions_reject_misshapen_inputs() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (_ckpt, flash, weights) = load(&mut kernels);
        let shape = *flash.shape();
        let rank2 = import_f32(&mut kernels, vec![2, 64], &[1.0e-7; 128]);
        let one_branch = import_f32(&mut kernels, vec![3, 1, 64], &[1.0e-7; 192]);
        let h = import_f32(&mut kernels, vec![3, 2, 64], &stacked(residual_row));
        let gate = import_f32(&mut kernels, vec![3, 2, 1], &[1.0; 6]);
        let bad_gate = import_f32(&mut kernels, vec![2, 1], &[1.0; 2]);
        let block = import_f32(&mut kernels, vec![3, 64], &[1.0e-7; 192]);
        let bad_block = import_f32(&mut kernels, vec![1, 64], &[1.0e-7; 64]);
        let wrong_rank = HyperShape::new(2, 64, 16, 1.0e-6).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let hyper_w = &weights.layers[0].attn_hyper;
        let results = [
            (
                "rank 2 residual",
                read_rows(&mut ctx, &shape.hyper, hyper_w, &rank2).err(),
            ),
            (
                "one branch",
                read_rows(&mut ctx, &shape.hyper, hyper_w, &one_branch).err(),
            ),
            (
                "weights of another rank",
                read_rows(&mut ctx, &wrong_rank, hyper_w, &h).err(),
            ),
            (
                "apply residual",
                apply_rows(&mut ctx, &shape.hyper, &rank2, &gate, &block).err(),
            ),
            (
                "apply gate",
                apply_rows(&mut ctx, &shape.hyper, &h, &bad_gate, &block).err(),
            ),
            (
                "apply block",
                apply_rows(&mut ctx, &shape.hyper, &h, &gate, &bad_block).err(),
            ),
        ];
        let words = [
            "[T, C, H]",
            "[T, C, H]",
            "weights",
            "[T, C, H]",
            "gate",
            "block",
        ];
        for ((what, error), word) in results.into_iter().zip(words) {
            assert!(
                matches!(&error, Some(EngError::BadRequest(m)) if m.contains(word)),
                "{what}: {error:?}"
            );
        }
        let good = apply_rows(&mut ctx, &shape.hyper, &h, &gate, &block).unwrap();
        assert_eq!(good.shape(), &[3, 2, 64][..], "valid apply shape");
    }
}
