#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use inference_engine::{
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static_assertions::assert_not_impl_any!(ExecutionTensor: Send, Sync);

#[test]
fn imported_values_keep_bytes_and_owner_lease() {
    let mut kernels = MlxKernels::try_new().unwrap();
    let cases = [
        (Dtype::U8, vec![2, 2], vec![7, 3, 5, 9]),
        (Dtype::F16, vec![2], vec![0, 60, 0, 64]),
        (Dtype::Bf16, vec![2], vec![128, 63, 0, 64]),
        (Dtype::F32, vec![2], vec![0, 0, 128, 63, 0, 0, 0, 64]),
        (Dtype::U32, vec![], vec![4, 3, 2, 1]),
        (Dtype::I64, vec![1], vec![255; 8]),
        (Dtype::I16, vec![2], vec![1, 128, 254, 127]),
        (Dtype::I32, vec![], vec![45, 209, 220, 131]),
    ];
    let mut retained = Vec::new();
    for (dtype, shape, bytes) in cases {
        let source = Tensor::try_from_host(dtype, shape.clone(), bytes.clone()).unwrap();
        let value = kernels.import_tensor(&source).unwrap();
        drop(source);
        assert_eq!(value.dtype(), dtype);
        assert_eq!(value.shape(), shape);
        assert_eq!(value.byte_len(), bytes.len() as u64);
        let shared = value.share();
        drop(value);
        let readback = kernels.readback(&shared).unwrap();
        assert_eq!(readback.dtype(), dtype);
        assert_eq!(readback.shape(), shape);
        assert_eq!(readback.host_bytes(), Some(bytes.as_slice()));
        retained.push(shared);
    }
    drop(kernels);
    let mut second = MlxKernels::try_new().unwrap();
    assert!(matches!(
        second.readback(&retained[0]),
        Err(inference_engine::EngError::BadRequest(_))
    ));
    assert!(matches!(
        MlxKernels::try_new(),
        Err(inference_engine::EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
    drop(retained);
    let replacement = MlxKernels::try_new().unwrap();
    assert_ne!(replacement.owner_key(), second.owner_key());
    drop(replacement);
}
