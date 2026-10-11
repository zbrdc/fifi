//! Shared resident helpers: constants, the operation context, projections, and host tensors.
//!
//! These helpers follow `flash_next/resident/ops.rs`. Q36 keeps its own copy so that no packet edits
//! a Flash-Next file (decision 0046). The constants differ: DeltaNet scales the query and key
//! epsilon by the key dimension.

use crate::{
    EngError,
    family::qwen36::host::HostShape,
    kernels::{MlxKernels, ResidentPackedWeight},
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
            operation: "qwen36",
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

/// Rank-0 F32 constants, imported once at load.
pub(super) struct Consts {
    /// 0, the second operand of `logaddexp` (softplus).
    pub(super) zero: ExecutionTensor,
    /// `key_dim * 1e-6`: the epsilon of the DeltaNet query and key norm against the sum of squares.
    pub(super) qk_eps: ExecutionTensor,
    /// `sqrt(key_dim)`: the extra divisor of the DeltaNet query.
    pub(super) sqrt_kd: ExecutionTensor,
    /// The dtype of the decode activations (packet Q36-BF16-a): the hidden state between blocks,
    /// the inputs of the quantized matmuls, and the converted layer norms. `F32` at import; the
    /// decoder constructor sets `Bf16` for a BF16 engine, and nothing mutates it afterwards.
    pub(super) act: Dtype,
}

impl Consts {
    /// Imports every constant with one `import_many` (one synchronization).
    pub(super) fn import(kernels: &mut MlxKernels, shape: &HostShape) -> Result<Self, EngError> {
        let key_dim = shape.delta.key_dim as f32;
        let sources = [0.0, key_dim * 1e-6, key_dim.sqrt()]
            .iter()
            .map(|&v| f32_tensor(Vec::new(), &[v]))
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<&Tensor> = sources.iter().collect();
        let mut got = kernels.import_many(&refs)?.into_iter();
        let mut next = || got.next().ok_or_else(short_import);
        Ok(Self {
            zero: next()?,
            qk_eps: next()?,
            sqrt_kd: next()?,
            act: Dtype::F32,
        })
    }
    /// Adds the constants' bytes to `total`.
    pub(super) fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        for tensor in [&self.zero, &self.qk_eps, &self.sqrt_kd] {
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

/// Converts `x` to `dtype`. A tensor that has `dtype` already is shared and makes no native call, so
/// a step with F32 activations builds the same graph as before packet Q36-BF16-a.
pub(super) fn cast_to(
    ctx: &mut Ctx<'_>,
    x: &ExecutionTensor,
    dtype: Dtype,
) -> Result<ExecutionTensor, EngError> {
    if x.dtype() == dtype {
        return Ok(x.share());
    }
    ctx.kernels.cast(x, dtype)
}

/// Converts `x` to the activation dtype of the step (`Consts::act`).
pub(super) fn to_act(ctx: &mut Ctx<'_>, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
    let act = ctx.consts.act;
    cast_to(ctx, x, act)
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

/// Asserts element-wise closeness within `tol` (test helper).
#[cfg(test)]
pub(super) fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what} element {i}: actual {a}, expected {e}"
        );
    }
}

/// Asserts cosine at least 0.99999 and max error over max magnitude at most 1e-4 (test helper).
/// This is the metric of the Flash-Next resident tests and of decision 0046 item 4.
#[cfg(test)]
pub(super) fn assert_parity(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let (mut dot, mut aa, mut ee, mut diff, mut top) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e) = (f64::from(a), f64::from(e));
        dot += a * e;
        aa += a * a;
        ee += e * e;
        diff = diff.max((a - e).abs());
        top = top.max(e.abs());
    }
    assert!(top > 0.0, "{what}: expected values are all zero");
    let cos = dot / (aa.sqrt() * ee.sqrt());
    assert!(cos >= 0.99999, "{what}: cos {cos}");
    assert!(diff / top <= 1e-4, "{what}: relative error {}", diff / top);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{qwen36::host::HostShape, spec::QWEN3_6_35B_A3B};

    fn real_shape() -> HostShape {
        HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap()
    }

    #[test]
    fn consts_read_back_their_values() {
        // Real key dimension 128: 128 * 1e-6 = 1.28e-4 and sqrt(128) = 11.3137085.
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let consts = Consts::import(&mut kernels, &real_shape()).unwrap();
        for tensor in [&consts.zero, &consts.qk_eps, &consts.sqrt_kd] {
            assert!(
                tensor.shape().is_empty(),
                "scalar shape {:?}",
                tensor.shape()
            );
        }
        assert_close(
            &values(&mut kernels, &consts.zero).unwrap(),
            &[0.0],
            0.0,
            "zero",
        );
        assert_close(
            &values(&mut kernels, &consts.qk_eps).unwrap(),
            &[1.28e-4],
            1e-9,
            "qk eps",
        );
        assert_close(
            &values(&mut kernels, &consts.sqrt_kd).unwrap(),
            &[11.313_708],
            1e-5,
            "sqrt key dim",
        );
    }

    #[test]
    fn linear_flattens_the_input_and_rejects_a_wrong_width() {
        // A [2, 64] input with 128 elements against a weight with 64 inputs must fail,
        // and a [1, 2, 32] input with 64 elements must pass the width check.
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let consts = Consts::import(&mut kernels, &real_shape()).unwrap();
        let tiny = super::super::super::tiny::checkpoint(&super::super::super::tiny::TINY);
        let weight = super::super::super::host_load::HostWeights::load(
            &tiny,
            &super::super::super::tiny::TINY,
            &super::super::super::tiny::config(&super::super::super::tiny::TINY),
        )
        .unwrap();
        let view = weight.view();
        let super::super::super::host::MixerW::Delta(d) = &view.layers[0].mixer else {
            panic!("layer 0 is DeltaNet");
        };
        let crate::family::flash_next::stored::StoredWeight::Packed(packed) = d.out else {
            panic!("out_proj is packed");
        };
        let resident = kernels.import_packed(packed).unwrap();
        let wide = kernels
            .import_tensor(&f32_tensor(vec![2, 64], &[1.0; 128]).unwrap())
            .unwrap();
        let ok = kernels
            .import_tensor(&f32_tensor(vec![1, 2, 32], &[1.0; 64]).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &consts,
        };
        assert!(matches!(
            linear(&mut ctx, &resident, &wide),
            Err(EngError::BadRequest(_))
        ));
        let y = linear(&mut ctx, &resident, &ok).unwrap();
        assert_eq!(y.shape(), &[1, 64][..]);
    }
}
