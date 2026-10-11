#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

static_assertions::assert_not_impl_any!(MlxKernels: Send, Sync);

fn affine(bits: u32, group_size: u32) -> QuantizationSpec {
    QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits,
        group_size,
    }
}

fn values(numbers: &[f32]) -> Vec<u8> {
    numbers
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn words(numbers: &[u32]) -> Vec<u8> {
    numbers
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn tensor(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Tensor {
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}

fn affine_expert_weight() -> PackedWeight {
    let payload = tensor(
        Dtype::U32,
        vec![2, 2, 4],
        [0x11111111u32, 0x22222222, 0x11111111, 0x22222222]
            .into_iter()
            .flat_map(|word| std::iter::repeat_n(word, 4))
            .flat_map(u32::to_le_bytes)
            .collect(),
    );
    let scales = tensor(Dtype::F32, vec![2, 2, 1], values(&[1.0, 1.0, 2.0, 2.0]));
    let biases = tensor(Dtype::F32, vec![2, 2, 1], values(&[0.0, 0.0, 1.0, 1.0]));
    PackedWeight::try_new(payload, scales, Some(biases), vec![2, 2, 32], affine(4, 32)).unwrap()
}

fn ones(rows: u32, columns: u32) -> Tensor {
    let bytes = (0..rows * columns)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();
    tensor(Dtype::F32, vec![rows, columns], bytes)
}

#[test]
fn routed_affine_experts_produce_their_selected_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let expert_indices = tensor(Dtype::U32, vec![2], words(&[1, 0]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels
        .gather_dequant_matmul(&ones(2, 32), &affine_expert_weight(), &expert_indices)
        .unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = values(&[96.0, 160.0, 32.0, 64.0]);
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn routed_mxfp4_experts_produce_their_selected_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let payload = tensor(
        Dtype::U32,
        vec![2, 2, 4],
        [0x22222222u32, 0x44444444, 0x22222222, 0x44444444]
            .into_iter()
            .flat_map(|word| std::iter::repeat_n(word, 4))
            .flat_map(u32::to_le_bytes)
            .collect(),
    );
    let scales = tensor(Dtype::U8, vec![2, 2, 1], vec![127, 127, 128, 128]);
    let weight = PackedWeight::try_new(
        payload,
        scales,
        None,
        vec![2, 2, 32],
        QuantizationSpec {
            mode: QuantizationMode::Mxfp4,
            bits: 4,
            group_size: 32,
        },
    )
    .unwrap();
    let expert_indices = tensor(Dtype::U32, vec![2], words(&[1, 0]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels
        .gather_dequant_matmul(&ones(2, 32), &weight, &expert_indices)
        .unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = values(&[64.0, 128.0, 32.0, 64.0]);
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn routed_float_experts_produce_their_selected_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let activation = tensor(
        Dtype::F32,
        vec![2, 3],
        values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let weight = tensor(
        Dtype::F32,
        vec![2, 3, 2],
        values(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0]),
    );
    let expert_indices = tensor(Dtype::U32, vec![2], words(&[1, 0]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels
        .gather_matmul(&activation, &weight, &expert_indices)
        .unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = values(&[5.0, 7.0, 10.0, 11.0]);
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn embedding_rows_produce_the_requested_table_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let table = tensor(
        Dtype::F32,
        vec![4, 3],
        values(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]),
    );
    let indices = tensor(Dtype::U32, vec![3], words(&[2, 0, 3]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels.embedding_rows(&table, &indices).unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [3, 3]);
    let expected = values(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn quantized_embedding_rows_dequantize_the_selected_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let payload = tensor(
        Dtype::U32,
        vec![4, 4],
        [0x11111111u32, 0x22222222, 0x11111111, 0x22222222]
            .into_iter()
            .flat_map(|word| std::iter::repeat_n(word, 4))
            .flat_map(u32::to_le_bytes)
            .collect(),
    );
    let scales = tensor(Dtype::F32, vec![4, 1], values(&[1.0, 1.0, 2.0, 2.0]));
    let biases = tensor(Dtype::F32, vec![4, 1], values(&[0.0, 0.0, 1.0, 1.0]));
    let weight =
        PackedWeight::try_new(payload, scales, Some(biases), vec![4, 32], affine(4, 32)).unwrap();
    let indices = tensor(Dtype::U32, vec![2], words(&[3, 1]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels.quantized_embedding_rows(&weight, &indices).unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 32]);
    let mut expected = vec![5.0f32; 32];
    expected.extend([2.0; 32]);
    assert_eq!(result.host_bytes().unwrap(), values(&expected));
}

#[test]
fn gather_rejects_expert_indices_beyond_the_expert_count() {
    let _serial = GPU_TEST.lock().unwrap();
    let expert_indices = tensor(Dtype::U32, vec![2], words(&[0, 2]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result =
        kernels.gather_dequant_matmul(&ones(2, 32), &affine_expert_weight(), &expert_indices);

    assert!(matches!(result, Err(EngError::BadRequest(_))));
}

#[test]
fn embedding_rows_rejects_indices_beyond_the_table() {
    let _serial = GPU_TEST.lock().unwrap();
    let table = tensor(Dtype::F32, vec![2, 2], values(&[1.0, 2.0, 3.0, 4.0]));
    let indices = tensor(Dtype::U32, vec![1], words(&[2]));
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels.embedding_rows(&table, &indices);

    assert!(matches!(result, Err(EngError::BadRequest(_))));
}

#[test]
fn gather_rejects_an_index_count_that_differs_from_the_token_count() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let float_weight = tensor(Dtype::F32, vec![2, 32, 2], values(&[1.0; 128]));

    // The activation has two rows. One index would broadcast and three indices cannot pair.
    for count in [1usize, 3] {
        let expert_indices = tensor(Dtype::U32, vec![count as u32], words(&vec![0; count]));
        let quantized =
            kernels.gather_dequant_matmul(&ones(2, 32), &affine_expert_weight(), &expert_indices);
        assert!(
            matches!(&quantized, Err(EngError::BadRequest(message)) if message.contains("index count")),
            "{count} indices for two tokens: {:?}",
            quantized.as_ref().err()
        );
        let dense = kernels.gather_matmul(&ones(2, 32), &float_weight, &expert_indices);
        assert!(
            matches!(&dense, Err(EngError::BadRequest(message)) if message.contains("index count")),
            "{count} indices for two tokens: {:?}",
            dense.as_ref().err()
        );
    }
    // One token with two indices would widen the output past the requested shape.
    let two_indices = tensor(Dtype::U32, vec![2], words(&[0, 1]));
    let widened =
        kernels.gather_dequant_matmul(&ones(1, 32), &affine_expert_weight(), &two_indices);
    assert!(
        matches!(&widened, Err(EngError::BadRequest(message)) if message.contains("index count")),
        "two indices for one token: {:?}",
        widened.as_ref().err()
    );

    // The rejection happens before dispatch, so the adapter still serves a valid call.
    let matching = tensor(Dtype::U32, vec![2], words(&[1, 0]));
    let result = kernels
        .gather_dequant_matmul(&ones(2, 32), &affine_expert_weight(), &matching)
        .unwrap();
    assert_eq!(result.shape(), [2, 2]);
}
