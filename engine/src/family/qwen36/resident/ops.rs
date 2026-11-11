//! Shared resident helpers: constants, the operation context, projections, and host tensors.
//!
//! These helpers follow `flash_next/resident/ops.rs`. Q36 keeps its own copy so that a Q36 change
//! never edits a Flash-Next file. The constants differ: DeltaNet scales the query and key epsilon
//! by the key dimension.

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

pub(super) fn u32_tensor(shape: Vec<u32>, values: &[u32]) -> Result<Tensor, EngError> {
    Tensor::try_from_host(
        Dtype::U32,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

pub(super) fn zeros(shape: Vec<u32>) -> Result<Tensor, EngError> {
    let bytes = shape
        .iter()
        .try_fold(4usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| EngError::MemoryPlan("resident zero tensor size overflows".into()))?;
    Tensor::try_from_host(Dtype::F32, shape, vec![0u8; bytes])
}

/// Reads an F32 resident back as host values in row-major order. One synchronization.
pub(in crate::family::qwen36) fn values(
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
    pub(super) fn add(&mut self, bytes: u64) -> Result<(), EngError> {
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or_else(|| EngError::MemoryPlan("resident byte count overflows u64".into()))?;
        Ok(())
    }
    pub(super) fn tensor(&mut self, tensor: &ExecutionTensor) -> Result<(), EngError> {
        self.add(tensor.byte_len())
    }
    pub(super) fn packed(&mut self, weight: &ResidentPackedWeight) -> Result<(), EngError> {
        self.tensor(weight.payload())?;
        self.tensor(weight.scales())?;
        if let Some(biases) = weight.biases() {
            self.tensor(biases)?;
        }
        Ok(())
    }
    pub(super) fn total(&self) -> u64 {
        self.0
    }
}

/// The scalar and vector constants of the decode and prefill steps, imported once at load.
pub(super) struct Consts {
    /// 0, the second operand of `logaddexp` (softplus).
    pub(super) zero: ExecutionTensor,
    /// `key_dim * 1e-6`: the epsilon of the DeltaNet query and key norm against the sum of
    /// squares. The chunked prefill keeps the square/sum/divide rule and reads this; the decode
    /// step's fused form is the pair below.
    pub(super) qk_eps: ExecutionTensor,
    /// `sqrt(key_dim)`: the extra divisor of the DeltaNet query in the prefill rule.
    pub(super) sqrt_kd: ExecutionTensor,
    /// The query norm weight `[key_dim]` of the DeltaNet DECODE head norm: `1 / key_dim` per
    /// channel. `rms_norm(q, q_norm_w, 1e-6)` equals the host rule
    /// `q / (sqrt(sum(q * q) + key_dim * 1e-6) * sqrt(key_dim))` exactly over the reals, in ONE
    /// fused kernel instead of the square/sum/divide chain (the decode step is dispatch-bound,
    /// so the chain's extra kernel launches cost more than the math they do).
    pub(super) q_norm_w: ExecutionTensor,
    /// The key norm weight `[key_dim]` of the decode head norm: `1 / sqrt(key_dim)` per channel,
    /// so `rms_norm(k, k_norm_w, 1e-6)` equals `k / sqrt(sum(k * k) + key_dim * 1e-6)`.
    pub(super) k_norm_w: ExecutionTensor,
    /// The dtype of the decode activations: the hidden state between blocks,
    /// the inputs of the quantized matmuls, and the converted layer norms. `F32` at import; the
    /// decoder constructor sets `Bf16` for a BF16 engine, and nothing mutates it afterwards.
    pub(super) act: Dtype,
}

impl Consts {
    /// The three trailing slots of the compiled delta closure: the softplus zero, then either
    /// the `[kd]` norm weights (unfused) or the `qk_eps` and `sqrt_kd` scalars
    /// (`Q36_FUSED_DECODE=1` with three taps, when the ONE `delta_middle` kernel replaces the
    /// op sequence and the empty shape marks the mode for the key and the traced body). Both
    /// arg builders (`compiled::delta_args`, `compile_gate::delta_seam_args`) must agree with
    /// `delta::step_projected`, so the rule lives here once.
    pub(super) fn middle_consts(&self, fused: bool) -> [&ExecutionTensor; 3] {
        if fused {
            [&self.zero, &self.qk_eps, &self.sqrt_kd]
        } else {
            [&self.zero, &self.q_norm_w, &self.k_norm_w]
        }
    }

    /// Imports every constant with one `import_many` (one synchronization).
    pub(super) fn import(kernels: &mut MlxKernels, shape: &HostShape) -> Result<Self, EngError> {
        let key_dim = shape.delta.key_dim;
        let kd = key_dim as f32;
        // 1 / key_dim is exact in f32 for the power-of-two key dims of the family; 1 / sqrt(D)
        // rounds, within the parity tolerance of the step tests.
        let q_fill = 1.0 / kd;
        let k_fill = 1.0 / kd.sqrt();
        let sources = [
            f32_tensor(Vec::new(), &[0.0])?,
            f32_tensor(Vec::new(), &[kd * 1e-6])?,
            f32_tensor(Vec::new(), &[kd.sqrt()])?,
            f32_tensor(vec![key_dim as u32], &vec![q_fill; key_dim])?,
            f32_tensor(vec![key_dim as u32], &vec![k_fill; key_dim])?,
        ];
        let refs: Vec<&Tensor> = sources.iter().collect();
        let mut got = kernels.import_many(&refs)?.into_iter();
        let mut next = || got.next().ok_or_else(short_import);
        Ok(Self {
            zero: next()?,
            qk_eps: next()?,
            sqrt_kd: next()?,
            q_norm_w: next()?,
            k_norm_w: next()?,
            act: Dtype::F32,
        })
    }
    pub(super) fn bytes(&self, total: &mut Bytes) -> Result<(), EngError> {
        for tensor in [
            &self.zero,
            &self.qk_eps,
            &self.sqrt_kd,
            &self.q_norm_w,
            &self.k_norm_w,
        ] {
            total.tensor(tensor)?;
        }
        Ok(())
    }
}

pub(in crate::family::qwen36) struct Ctx<'k> {
    /// `kernels` is read by the sibling trace module's Mac-gated readback (the only field it
    /// borrows), so it carries the module-group scope; `consts` and `compile` stay resident-only.
    pub(in crate::family::qwen36) kernels: &'k mut MlxKernels,
    pub(super) consts: &'k Consts,
    /// `mlx_compile` on the DeltaNet and device-routed MoE decode steps:
    /// the decoder's `qwen36.compile` setting, copied by `step::run`. `delta::step` and
    /// `moe::step_device` run their compiled closure when it is true; every other block — the
    /// attention layers, the head, the embedding, and every chunk of the chunked prefill, whose
    /// shapes vary — ignores it and never compiles.
    pub(super) compile: bool,
}

/// Converts `x` to `dtype`. A tensor that has `dtype` already is shared and makes no native call, so
/// a step with F32 activations builds no extra node.
pub(in crate::family::qwen36) fn cast_to(
    ctx: &mut Ctx<'_>,
    x: &ExecutionTensor,
    dtype: Dtype,
) -> Result<ExecutionTensor, EngError> {
    if x.dtype() == dtype {
        return Ok(x.share());
    }
    ctx.kernels.cast(x, dtype)
}

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

/// The rank-2 form of `linear`: every row of `[rows, input]` meets the weight in
/// ONE qmm - the row-batched projection (the head and the dense projections).
/// Requires a non-empty rank-2 activation with the weight's input width.
pub(super) fn linear_rows(
    ctx: &mut Ctx<'_>,
    weight: &ResidentPackedWeight,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let &[_, input] = weight.logical_shape() else {
        return Err(EngError::BadRequest(
            "resident linear needs a rank-2 weight".into(),
        ));
    };
    let &[rows, cols] = x.shape() else {
        return Err(EngError::BadRequest(
            "linear_rows needs a rank-2 input".into(),
        ));
    };
    if rows == 0 || cols != input {
        return Err(EngError::BadRequest(format!(
            "linear_rows: expected non-empty rows of width {input}, got {:?}",
            x.shape()
        )));
    }
    ctx.kernels.qmm(x, weight)
}

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

/// This is the metric of the Flash-Next resident tests and of the Q36 per-layer parity gate.
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
        // Real key dimension 128: the prefill scalars 128 * 1e-6 = 1.28e-4 and
        // sqrt(128) = 11.3137085, and the decode norm weights 1/128 = 0.0078125 exactly and
        // 1/sqrt(128) = 0.08838835.
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
        for tensor in [&consts.q_norm_w, &consts.k_norm_w] {
            assert_eq!(tensor.shape(), &[128], "norm weight shape");
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
        assert_close(
            &values(&mut kernels, &consts.q_norm_w).unwrap(),
            &[0.0078125; 128],
            0.0,
            "query norm weight",
        );
        assert_close(
            &values(&mut kernels, &consts.k_norm_w).unwrap(),
            &[0.088_388_35; 128],
            1e-7,
            "key norm weight",
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
            compile: false,
        };
        assert!(matches!(
            linear(&mut ctx, &resident, &wide),
            Err(EngError::BadRequest(_))
        ));
        let y = linear(&mut ctx, &resident, &ok).unwrap();
        assert_eq!(y.shape(), &[1, 64][..]);
    }
}
