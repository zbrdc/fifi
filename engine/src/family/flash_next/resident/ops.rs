//! Shared resident helpers: constants, the operation context, projections, and host tensors.

use super::super::step::ModelShape;
use super::qsa::{MASK_OFF, MAX_POSITIONS};
use crate::{
    EngError,
    kernels::{BinaryOp, MlxKernels, ResidentPackedWeight},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

/// Reports a batched import that returned a different tensor count than requested.
pub(super) fn short_import() -> EngError {
    EngError::Backend {
        operation: "import_many",
        message: "resident import returned a different tensor count".into(),
    }
}

/// Converts a host dimension to a resident shape dimension.
pub(super) fn dim(n: usize) -> Result<u32, EngError> {
    u32::try_from(n).map_err(|_| EngError::BadRequest("resident dimension exceeds u32".into()))
}

/// Builds an F32 host tensor. `Tensor::try_from_host` rejects a length that differs from the shape.
pub(super) fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Result<Tensor, EngError> {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// Builds a U32 host tensor.
pub(super) fn u32_tensor(shape: Vec<u32>, values: &[u32]) -> Result<Tensor, EngError> {
    Tensor::try_from_host(
        Dtype::U32,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// Builds an F32 host tensor of zeros.
pub(super) fn zeros(shape: Vec<u32>) -> Result<Tensor, EngError> {
    let bytes = shape
        .iter()
        .try_fold(4usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| EngError::MemoryPlan("resident zero tensor size overflows".into()))?;
    Tensor::try_from_host(Dtype::F32, shape, vec![0u8; bytes])
}

/// Reads an F32 resident back as host values in row-major order. One synchronization.
pub(super) fn values(
    kernels: &mut MlxKernels,
    tensor: &ExecutionTensor,
) -> Result<Vec<f32>, EngError> {
    let host = kernels.readback(tensor)?;
    let bytes = host
        .host_bytes()
        .filter(|_| host.dtype() == Dtype::F32)
        .ok_or_else(|| EngError::Backend {
            operation: "flash_next",
            message: "readback is not host F32".into(),
        })?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// Checked byte total of resident tensors. `byte_len` is an upper bound for shared buffers.
#[derive(Debug, Default)]
pub(super) struct Bytes(u64);

impl Bytes {
    /// Adds `bytes`.
    pub(super) fn add(&mut self, bytes: u64) -> Result<(), EngError> {
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or_else(|| EngError::MemoryPlan("resident byte count overflows u64".into()))?;
        Ok(())
    }
    /// Adds one resident tensor.
    pub(super) fn tensor(&mut self, tensor: &ExecutionTensor) -> Result<(), EngError> {
        self.add(tensor.byte_len())
    }
    /// Adds the payload, scales, and biases of a packed weight.
    pub(super) fn packed(&mut self, weight: &ResidentPackedWeight) -> Result<(), EngError> {
        self.tensor(weight.payload())?;
        self.tensor(weight.scales())?;
        if let Some(biases) = weight.biases() {
            self.tensor(biases)?;
        }
        Ok(())
    }
    /// The total in bytes.
    pub(super) fn total(&self) -> u64 {
        self.0
    }
}

/// Rank-0 F32 constants, the U32 `gather_qmm` indices, and the QSA block vectors, imported once at load.
pub(super) struct Consts {
    pub(super) zero: ExecutionTensor,
    pub(super) half: ExecutionTensor,
    pub(super) two: ExecutionTensor,
    pub(super) eps6: ExecutionTensor,
    pub(super) branches: ExecutionTensor,
    pub(super) sqrt_kd: ExecutionTensor,
    pub(super) sqrt_hidden: ExecutionTensor,
    pub(super) ple_scale: ExecutionTensor,
    pub(super) expert_ids: ExecutionTensor,
    /// `1 / sqrt(head_dim)` of the QSA attention.
    pub(super) qsa_scale: ExecutionTensor,
    /// F32 `[0, 1, ..., blocks - 1]` for the blocks of `MAX_POSITIONS` positions.
    pub(super) block_ids: ExecutionTensor,
    /// F32 `[blocks]`: 0 for the first `budget / ratio` ranks, then `MASK_OFF`.
    pub(super) rank_keep: ExecutionTensor,
}

impl Consts {
    /// Imports every constant with one `import_many` (one synchronization).
    pub(super) fn import(
        kernels: &mut MlxKernels,
        shape: &ModelShape,
        ple_scale: f32,
    ) -> Result<Self, EngError> {
        let scalars = [
            0.0,
            0.5,
            2.0,
            1e-6,
            shape.hyper.branches() as f32,
            (shape.delta.key_dim() as f32).sqrt(),
            (shape.hyper.hidden() as f32).sqrt(),
            ple_scale,
        ];
        let mut sources = scalars
            .iter()
            .map(|&v| f32_tensor(Vec::new(), &[v]))
            .collect::<Result<Vec<_>, _>>()?;
        let top_k = dim(shape.moe.top_k())?;
        sources.push(u32_tensor(vec![top_k], &(0..top_k).collect::<Vec<u32>>())?);
        let qsa = shape.qsa.dims();
        let bad_ratio = || EngError::BadRequest("resident qsa block ratio is zero".into());
        let whole = MAX_POSITIONS.checked_div(qsa.ratio).ok_or_else(bad_ratio)?;
        let partial = MAX_POSITIONS.checked_rem(qsa.ratio).ok_or_else(bad_ratio)?;
        let blocks = if partial == 0 {
            whole
        } else {
            whole.checked_add(1).ok_or_else(|| {
                EngError::BadRequest("resident qsa block count overflows u32".into())
            })?
        };
        let attending = qsa.budget.checked_div(qsa.ratio).ok_or_else(bad_ratio)?;
        let ids: Vec<f32> = (0..blocks).map(|b| b as f32).collect();
        let keep: Vec<f32> = (0..blocks)
            .map(|r| if r < attending { 0.0 } else { MASK_OFF })
            .collect();
        sources.push(f32_tensor(
            Vec::new(),
            &[1.0 / (qsa.head_dim as f32).sqrt()],
        )?);
        sources.push(f32_tensor(vec![blocks], &ids)?);
        sources.push(f32_tensor(vec![blocks], &keep)?);
        let refs: Vec<&Tensor> = sources.iter().collect();
        let mut got = kernels.import_many(&refs)?.into_iter();
        let mut next = || got.next().ok_or_else(short_import);
        Ok(Self {
            zero: next()?,
            half: next()?,
            two: next()?,
            eps6: next()?,
            branches: next()?,
            sqrt_kd: next()?,
            sqrt_hidden: next()?,
            ple_scale: next()?,
            expert_ids: next()?,
            qsa_scale: next()?,
            block_ids: next()?,
            rank_keep: next()?,
        })
    }
    /// Adds the constants' bytes to `total`.
    pub(super) fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        for tensor in [
            &self.zero,
            &self.half,
            &self.two,
            &self.eps6,
            &self.branches,
            &self.sqrt_kd,
            &self.sqrt_hidden,
            &self.ple_scale,
            &self.expert_ids,
            &self.qsa_scale,
            &self.block_ids,
            &self.rank_keep,
        ] {
            total.tensor(tensor)?;
        }
        Ok(())
    }
}

/// The adapter and the constants that every resident block uses.
pub(super) struct Ctx<'k> {
    pub(super) kernels: &'k mut MlxKernels,
    pub(super) consts: &'k Consts,
}

/// Returns `x · Wᵀ` as `[1, out]`. `x` may have any shape with `in` elements. `qmm` takes only `[1, in]`.
pub(super) fn linear(
    ctx: &mut Ctx<'_>,
    weight: &ResidentPackedWeight,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let &[_, input] = weight.logical_shape() else {
        return Err(EngError::BadRequest(
            "resident linear needs a rank-2 weight".into(),
        ));
    };
    let count = x
        .shape()
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)));
    if count != Some(u64::from(input)) {
        return Err(EngError::BadRequest(
            "resident linear input width differs from the weight".into(),
        ));
    }
    let flat = ctx.kernels.reshape(x, &[1, input])?;
    ctx.kernels.qmm(&flat, weight)
}

/// Zero-centered RMS norm of each row of the last axis: `rms_norm_plain(x) * (1 + w)`, with `1 + w` from load.
pub(super) fn rms0_rows(
    ctx: &mut Ctx<'_>,
    x: &ExecutionTensor,
    w1p: &ExecutionTensor,
    eps: f32,
) -> Result<ExecutionTensor, EngError> {
    let unit = ctx.kernels.rms_norm_plain(x, eps)?;
    ctx.kernels.binary(BinaryOp::Mul, &unit, w1p)
}

/// Asserts element-wise closeness within 1e-5 (test helper).
#[cfg(test)]
pub(super) fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5,
            "{what} element {i}: actual {a}, expected {e}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consts_read_back_their_values() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let consts = &weights.consts;
        for tensor in [
            &consts.zero,
            &consts.half,
            &consts.two,
            &consts.eps6,
            &consts.branches,
            &consts.sqrt_kd,
            &consts.sqrt_hidden,
            &consts.ple_scale,
        ] {
            assert!(
                tensor.shape().is_empty(),
                "scalar shape {:?}",
                tensor.shape()
            );
        }
        assert_close(&values(&mut kernels, &consts.zero).unwrap(), &[0.0], "zero");
        assert_close(&values(&mut kernels, &consts.half).unwrap(), &[0.5], "half");
        assert_close(&values(&mut kernels, &consts.two).unwrap(), &[2.0], "two");
        assert_close(
            &values(&mut kernels, &consts.branches).unwrap(),
            &[2.0],
            "branches",
        );
        assert_close(
            &values(&mut kernels, &consts.sqrt_kd).unwrap(),
            &[32f32.sqrt()],
            "sqrt key dim",
        );
        assert_close(
            &values(&mut kernels, &consts.sqrt_hidden).unwrap(),
            &[8.0],
            "sqrt hidden",
        );
        assert_close(
            &values(&mut kernels, &consts.ple_scale).unwrap(),
            &[0.5],
            "ple scale",
        );
        let eps6 = values(&mut kernels, &consts.eps6).unwrap();
        assert!((eps6[0] - 1e-6).abs() <= 1e-12, "eps6 actual {}", eps6[0]);
        assert_eq!(consts.expert_ids.shape(), &[2][..], "expert ids shape");
        let host = kernels.readback(&consts.expert_ids).unwrap();
        assert_eq!(host.dtype(), Dtype::U32, "expert ids dtype");
        let words: Vec<u32> = host
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(words, [0, 1], "expert ids words");
        assert_close(
            &values(&mut kernels, &consts.qsa_scale).unwrap(),
            &[0.125],
            "qsa scale",
        );
        let ids = values(&mut kernels, &consts.block_ids).unwrap();
        assert_eq!(ids.len(), 8192, "block id count");
        assert_eq!((ids[0], ids[1], ids[8191]), (0.0, 1.0, 8191.0), "block ids");
        let keep = values(&mut kernels, &consts.rank_keep).unwrap();
        assert_eq!(keep.len(), 8192, "rank keep count");
        assert_eq!(&keep[..4], &[0.0; 4][..], "ranks that attend");
        assert_eq!(keep[4], MASK_OFF, "first rank that does not attend");
    }

    #[test]
    fn linear_flattens_the_input_before_qmm() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let host = f32_tensor(vec![2, 64], &[1.0; 128]).unwrap();
        let x = kernels.import_tensor(&host).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let y = linear(&mut ctx, &weights.mixer.down, &x);
        assert!(y.is_ok(), "{:?}", y.as_ref().err());
        let y = y.unwrap();
        assert_eq!(y.shape(), &[1, 32][..], "linear output shape");
        let expected: Vec<f32> = (0..32).map(|r| 448.0 + 32.0 * (r % 4) as f32).collect();
        assert_close(
            &values(&mut kernels, &y).unwrap(),
            &expected,
            "linear values",
        );
        let bad_host = f32_tensor(vec![1, 64], &[1.0; 64]).unwrap();
        let bad = kernels.import_tensor(&bad_host).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        assert!(matches!(
            linear(&mut ctx, &weights.mixer.down, &bad),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn rms0_rows_scales_unit_rows() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let x = kernels
            .import_tensor(&f32_tensor(vec![2, 2], &[3.0, 4.0, 0.0, 2.0]).unwrap())
            .unwrap();
        let w1p = kernels
            .import_tensor(&f32_tensor(vec![2, 2], &[1.0, 2.0, 1.0, 1.0]).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let out = rms0_rows(&mut ctx, &x, &w1p, 1e-6).unwrap();
        let r = 12.5f32.sqrt();
        let expected = [3.0 / r, 8.0 / r, 0.0, std::f32::consts::SQRT_2];
        assert_close(&values(&mut kernels, &out).unwrap(), &expected, "rms0 rows");
    }
}
