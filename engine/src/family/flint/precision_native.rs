//! Fixed Flint BF16 boundaries. F32 primitive reference fixtures remain separate.
use crate::{
    EngError,
    kernels::{BinaryOp, MlxKernels, ReduceOp, RopeStep},
    tensor::{Dtype, ExecutionTensor},
};

pub(super) fn normalize_qk(
    k: &mut MlxKernels,
    qk: (&ExecutionTensor, &ExecutionTensor),
    gamma: (&ExecutionTensor, &ExecutionTensor),
    position: Option<u32>,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let q = k.cast(qk.0, Dtype::Bf16)?;
    let keys = k.cast(qk.1, Dtype::Bf16)?;
    let mut q = k.rms_norm(&q, gamma.0, 1e-6)?;
    let mut keys = k.rms_norm(&keys, gamma.1, 1e-6)?;
    if let Some(pos) = position {
        // The existing RoPE seam accepts F32 only. Its inputs have already rounded
        // at the BF16 norm boundary; publish the rotation back in BF16.
        q = k.cast(&q, Dtype::F32)?;
        keys = k.cast(&keys, Dtype::F32)?;
        k.rope(
            &mut q,
            &mut keys,
            &RopeStep {
                pos,
                theta: 10000.0,
                part: 128,
            },
        )?;
        q = k.cast(&q, Dtype::Bf16)?;
        keys = k.cast(&keys, Dtype::Bf16)?;
    }
    Ok((q, keys))
}

pub(super) fn combine(
    k: &mut MlxKernels,
    rows: &ExecutionTensor,
    weights: &ExecutionTensor,
    shared: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let (tokens, count, hidden) = match *rows.shape() {
        [count, hidden] => (1, count, hidden),
        [tokens, count, hidden] => (tokens, count, hidden),
        _ => {
            return Err(EngError::BadRequest(
                "Flint expert rows must be [K,H] or [T,K,H]".into(),
            ));
        }
    };
    if tokens == 0 || count == 0 || hidden == 0 || shared.shape() != [tokens, hidden] {
        return Err(EngError::BadRequest(
            "Flint expert/shared geometry differs".into(),
        ));
    }
    let rows = k.reshape(rows, &[tokens, count, hidden])?;
    let weights = k.cast(weights, Dtype::Bf16)?;
    let weights = k.reshape(&weights, &[tokens, count, 1])?;
    let weighted = k.binary(BinaryOp::Mul, &rows, &weights)?;
    let total = k.reduce(ReduceOp::Sum, &weighted, 1, false)?;
    k.binary(BinaryOp::Add, &total, shared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    fn input(k: &mut MlxKernels, shape: &[u32], values: &[f32]) -> ExecutionTensor {
        let host = Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let x = k.import_tensor(&host).unwrap();
        k.cast(&x, Dtype::Bf16).unwrap()
    }
    #[test]
    #[ignore = "requires parent-owned Mac GPU window; no model weights"]
    fn bf16_norm_rope_and_expert_combine_keep_publication_boundaries() {
        let _serial = crate::native_test_lock();
        let mut k = MlxKernels::try_new().unwrap();
        k.allow_bf16(true);
        let q = input(
            &mut k,
            &[32, 1, 128],
            &(0..4096)
                .map(|i| (i % 19) as f32 / 16.0 - 0.5)
                .collect::<Vec<_>>(),
        );
        let keys = input(
            &mut k,
            &[4, 1, 128],
            &(0..512)
                .map(|i| (i % 17) as f32 / 16.0 - 0.5)
                .collect::<Vec<_>>(),
        );
        let gamma = input(&mut k, &[128], &[1.125; 128]);
        for pos in [None, Some(513)] {
            let (q, keys) = normalize_qk(&mut k, (&q, &keys), (&gamma, &gamma), pos).unwrap();
            assert_eq!(q.dtype(), Dtype::Bf16);
            assert_eq!(keys.dtype(), Dtype::Bf16);
            k.evaluate(&[&q, &keys]).unwrap();
        }
        let rows = input(&mut k, &[2, 2], &[1.5, -2.0, 3.0, 0.5]);
        let weights = input(&mut k, &[2, 1, 1], &[0.25, 0.75]);
        let weights = k.cast(&weights, Dtype::F32).unwrap();
        let projected = k.mm(&rows, &rows).unwrap();
        assert_eq!(projected.dtype(), Dtype::Bf16);
        let activated = k.silu(&projected).unwrap();
        assert_eq!(activated.dtype(), Dtype::Bf16);
        let product = k.binary(BinaryOp::Mul, &activated, &rows).unwrap();
        assert_eq!(product.dtype(), Dtype::Bf16);
        let shared = input(&mut k, &[1, 2], &[1.0, -1.0]);
        let out = combine(&mut k, &rows, &weights, &shared).unwrap();
        assert_eq!(out.dtype(), Dtype::Bf16);
        let out = k.cast(&out, Dtype::F32).unwrap();
        let host = k.readback(&out).unwrap();
        let got: Vec<_> = host
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(got, [3.625, -1.125]);
    }
}
