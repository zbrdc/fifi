#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::{Arc, Mutex};

use inference_engine::{
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, WeightSource},
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

#[test]
fn mxfp4_weights_use_their_e2m1_and_e8m0_values_exactly() {
    let _serial = GPU_TEST.lock().unwrap();
    let words = [0x22222222u32, 0xaaaaaaaa]
        .into_iter()
        .flat_map(|word| std::iter::repeat_n(word, 8))
        .flat_map(u32::to_le_bytes)
        .collect();
    let scales = Tensor::try_from_host(
        Dtype::U8,
        vec![2, 2],
        [127u8, 127, 127, 127]
            .into_iter()
            .flat_map(u8::to_le_bytes)
            .collect(),
    )
    .unwrap();
    let weight = PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, vec![2, 8], words).unwrap(),
        scales,
        None,
        vec![2, 64],
        QuantizationSpec {
            mode: QuantizationMode::Mxfp4,
            bits: 4,
            group_size: 32,
        },
    )
    .unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels
        .dequant_matmul(&fixture_activation(64), &weight)
        .unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = [64.0f32, -64.0, 1.0, -1.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn affine_bf16_weights_produce_the_pinned_f32_matrix() {
    let _serial = GPU_TEST.lock().unwrap();
    let source = ResidentWeights::from_bytes(Arc::from(
        include_bytes!("fixtures/packed.safetensors").as_slice(),
    ))
    .unwrap();
    let weight = PackedWeight::try_new(
        source.tensor("weight").unwrap(),
        source.tensor("scales").unwrap(),
        Some(source.tensor("biases").unwrap()),
        vec![2, 64],
        affine(4, 64),
    )
    .unwrap();
    let mut values = vec![1.0f32; 64];
    values.push(1.0);
    values.extend([0.0; 63]);
    let activation = Tensor::try_from_host(
        Dtype::F32,
        vec![2, 64],
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels.dequant_matmul(&activation, &weight).unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = [-288.0f32, -1216.0, -8.0, -12.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(result.host_bytes().unwrap(), expected);
}

fn fixture_weight(dtype: Dtype, group_size: u32) -> PackedWeight {
    let width = group_size.max(64);
    let words = [0x76543210u32, 0xfedcba98]
        .into_iter()
        .flat_map(|word| std::iter::repeat_n(word, (width / 8) as usize))
        .flat_map(u32::to_le_bytes)
        .collect();
    let (scale_bytes, bias_bytes) = match dtype {
        Dtype::F32 => (
            [1.0f32, -2.0].map(|value| value.to_le_bytes().to_vec()),
            [-8.0f32, 4.0].map(|value| value.to_le_bytes().to_vec()),
        ),
        Dtype::F16 => (
            [0x3c00u16, 0xc000].map(|value| value.to_le_bytes().to_vec()),
            [0xc800u16, 0x4400].map(|value| value.to_le_bytes().to_vec()),
        ),
        Dtype::Bf16 => (
            [0x3f80u16, 0xc000].map(|value| value.to_le_bytes().to_vec()),
            [0xc100u16, 0x4080].map(|value| value.to_le_bytes().to_vec()),
        ),
        _ => panic!("unsupported fixture dtype"),
    };
    let sidecar = |rows: [Vec<u8>; 2]| {
        Tensor::try_from_host(
            dtype,
            vec![2, width / group_size],
            rows.into_iter()
                .flat_map(|row| row.repeat((width / group_size) as usize))
                .collect(),
        )
        .unwrap()
    };
    PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, vec![2, width / 8], words).unwrap(),
        sidecar(scale_bytes),
        Some(sidecar(bias_bytes)),
        vec![2, width],
        affine(4, group_size),
    )
    .unwrap()
}

#[test]
fn affine_eight_bit_weights_produce_the_pinned_f32_matrix() {
    let _serial = GPU_TEST.lock().unwrap();
    let width = 64u32;
    let words = [0x0a0a0a0au32, 0x06060606]
        .into_iter()
        .flat_map(|word| std::iter::repeat_n(word, (width / 4) as usize))
        .flat_map(u32::to_le_bytes)
        .collect();
    let sidecar = |row_values: [f32; 2]| {
        Tensor::try_from_host(
            Dtype::F32,
            vec![2, 1],
            row_values.into_iter().flat_map(f32::to_le_bytes).collect(),
        )
        .unwrap()
    };
    let weight = PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, vec![2, width / 4], words).unwrap(),
        sidecar([1.0, -2.0]),
        Some(sidecar([-8.0, 4.0])),
        vec![2, width],
        affine(8, 64),
    )
    .unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels
        .dequant_matmul(&fixture_activation(64), &weight)
        .unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = [128.0f32, -512.0, 2.0, -8.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(result.host_bytes().unwrap(), expected);
}

fn fixture_activation(width: u32) -> Tensor {
    let mut values = vec![0.0f32; 2 * width as usize];
    values[..64].fill(1.0);
    values[width as usize] = 1.0;
    Tensor::try_from_host(
        Dtype::F32,
        vec![2, width],
        values.into_iter().flat_map(f32::to_le_bytes).collect(),
    )
    .unwrap()
}

fn assert_fixture_output(result: &Tensor) {
    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = [-288.0f32, -1216.0, -8.0, -12.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn unquantized_f32_matmul_matches_the_exact_products() {
    let _serial = GPU_TEST.lock().unwrap();
    let activation = Tensor::try_from_host(
        Dtype::F32,
        vec![2, 3],
        [1.0f32, 2.0, 3.0, 0.5, 0.25, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
    .unwrap();
    let weight = Tensor::try_from_host(
        Dtype::F32,
        vec![3, 2],
        [2.0f32, 0.0, 1.0, 1.0, 0.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
    .unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();

    let result = kernels.matmul(&activation, &weight).unwrap();

    assert_eq!(result.dtype(), Dtype::F32);
    assert_eq!(result.shape(), [2, 2]);
    let expected = [4.0f32, 14.0, 1.25, 16.25]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(result.host_bytes().unwrap(), expected);
}

#[test]
fn f16_scale_storage_matches_the_pinned_f32_output() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let result = kernels
        .dequant_matmul(&fixture_activation(64), &fixture_weight(Dtype::F16, 64))
        .unwrap();
    assert_fixture_output(&result);
}

#[test]
fn group_32_uses_its_declared_layout() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let result = kernels
        .dequant_matmul(&fixture_activation(64), &fixture_weight(Dtype::Bf16, 32))
        .unwrap();
    assert_fixture_output(&result);
}

#[test]
fn group_128_uses_its_declared_layout() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let result = kernels
        .dequant_matmul(&fixture_activation(128), &fixture_weight(Dtype::Bf16, 128))
        .unwrap();
    assert_fixture_output(&result);
}

#[test]
fn f32_scale_storage_matches_the_pinned_output() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let result = kernels
        .dequant_matmul(&fixture_activation(64), &fixture_weight(Dtype::F32, 64))
        .unwrap();
    assert_fixture_output(&result);
}

#[test]
fn invalid_activation_dtypes_leave_the_adapter_usable() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = fixture_weight(Dtype::Bf16, 64);
    for (dtype, bytes) in [
        (Dtype::F16, 256),
        (Dtype::Bf16, 256),
        (Dtype::U8, 128),
        (Dtype::U32, 512),
        (Dtype::I64, 1024),
    ] {
        let activation = Tensor::try_from_host(dtype, vec![2, 64], vec![0; bytes]).unwrap();
        assert!(matches!(
            kernels.dequant_matmul(&activation, &weight),
            Err(inference_engine::EngError::BadRequest(_))
        ));
    }
    assert_fixture_output(
        &kernels
            .dequant_matmul(&fixture_activation(64), &weight)
            .unwrap(),
    );
}

#[test]
fn invalid_activation_shapes_leave_the_adapter_usable() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = fixture_weight(Dtype::Bf16, 64);
    for shape in [
        vec![],
        vec![64],
        vec![1, 2, 64],
        vec![0, 64],
        vec![2, 0],
        vec![2, 32],
    ] {
        let elements = shape
            .iter()
            .map(|&dimension| dimension as usize)
            .product::<usize>();
        let activation = Tensor::try_from_host(Dtype::F32, shape, vec![0; elements * 4]).unwrap();
        assert!(matches!(
            kernels.dequant_matmul(&activation, &weight),
            Err(inference_engine::EngError::BadRequest(_))
        ));
    }
    assert_fixture_output(
        &kernels
            .dequant_matmul(&fixture_activation(64), &weight)
            .unwrap(),
    );
}

#[test]
fn batched_storage_is_rejected_without_claiming_batched_execution() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = fixture_weight(Dtype::Bf16, 64);
    let add_batch = |tensor: &Tensor| {
        let mut shape = vec![1];
        shape.extend_from_slice(tensor.shape());
        Tensor::try_from_host(tensor.dtype(), shape, tensor.host_bytes().unwrap().to_vec()).unwrap()
    };
    let batched = PackedWeight::try_new(
        add_batch(weight.payload()),
        add_batch(weight.scales()),
        weight.biases().map(add_batch),
        vec![1, 2, 64],
        affine(4, 64),
    )
    .unwrap();
    assert!(matches!(
        kernels.dequant_matmul(&fixture_activation(64), &batched),
        Err(inference_engine::EngError::BadRequest(_))
    ));
    assert_fixture_output(
        &kernels
            .dequant_matmul(&fixture_activation(64), &weight)
            .unwrap(),
    );
}

#[test]
fn nonfinite_inputs_are_rejected_before_native_execution() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = fixture_weight(Dtype::F32, 64);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for role in ["activation", "scales", "biases"] {
            let replace_first = |tensor: &Tensor| {
                let mut bytes = tensor.host_bytes().unwrap().to_vec();
                bytes[..4].copy_from_slice(&value.to_le_bytes());
                Tensor::try_from_host(Dtype::F32, tensor.shape().to_vec(), bytes).unwrap()
            };
            let activation = fixture_activation(64);
            let activation = if role == "activation" {
                replace_first(&activation)
            } else {
                activation
            };
            let scales = if role == "scales" {
                replace_first(weight.scales())
            } else {
                weight.scales().clone()
            };
            let biases = if role == "biases" {
                replace_first(weight.biases().unwrap())
            } else {
                weight.biases().unwrap().clone()
            };
            let invalid = PackedWeight::try_new(
                weight.payload().clone(),
                scales,
                Some(biases),
                vec![2, 64],
                affine(4, 64),
            )
            .unwrap();
            assert!(
                matches!(
                    kernels.dequant_matmul(&activation, &invalid),
                    Err(inference_engine::EngError::BadRequest(_))
                ),
                "{role}"
            );
        }
    }
    assert_fixture_output(
        &kernels
            .dequant_matmul(&fixture_activation(64), &weight)
            .unwrap(),
    );
}

#[test]
fn completed_output_survives_its_inputs_and_adapter() {
    let _serial = GPU_TEST.lock().unwrap();
    let result = {
        let activation = fixture_activation(64);
        let weight = fixture_weight(Dtype::Bf16, 64);
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.dequant_matmul(&activation, &weight).unwrap()
    };
    assert_fixture_output(&result);
}

#[test]
fn repeated_calls_return_their_own_results() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = fixture_weight(Dtype::Bf16, 64);
    let first = kernels
        .dequant_matmul(&fixture_activation(64), &weight)
        .unwrap();
    let zero = Tensor::try_from_host(Dtype::F32, vec![2, 64], vec![0; 512]).unwrap();
    let second = kernels.dequant_matmul(&zero, &weight).unwrap();
    assert_eq!(second.shape(), [2, 2]);
    assert_eq!(second.host_bytes().unwrap(), [0; 16]);
    assert_fixture_output(&first);
    assert_fixture_output(
        &kernels
            .dequant_matmul(&fixture_activation(64), &weight)
            .unwrap(),
    );
}

#[test]
fn two_owner_slots_release_after_completed_teardown() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut first = MlxKernels::try_new().unwrap();
    let mut second = MlxKernels::try_new().unwrap();
    assert_ne!(first.owner_key(), second.owner_key());
    assert!(matches!(
        MlxKernels::try_new(),
        Err(inference_engine::EngError::Backend {
            operation: "initialize",
            message,
        }) if message == "this process already holds two native adapter leases"
    ));
    for owner in [&mut first, &mut second] {
        assert_fixture_output(
            &owner
                .dequant_matmul(&fixture_activation(64), &fixture_weight(Dtype::Bf16, 64))
                .unwrap(),
        );
    }
    drop(first);
    let mut next = MlxKernels::try_new().unwrap();
    assert_ne!(next.owner_key(), second.owner_key());
    assert_fixture_output(
        &next
            .dequant_matmul(&fixture_activation(64), &fixture_weight(Dtype::Bf16, 64))
            .unwrap(),
    );
}

fn rows_tensor(rows: &[Vec<f32>]) -> Tensor {
    let width = rows.first().map_or(0, Vec::len);
    Tensor::try_from_host(
        Dtype::F32,
        vec![rows.len() as u32, width as u32],
        rows.iter()
            .flatten()
            .copied()
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
    .unwrap()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn assert_rows_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "output length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "element {index}: actual {got}, expected {want}"
        );
    }
}

/// The fixture weight row 0 is `q - 8` over 64 columns (sum -288 for an all-ones row, -8 for a
/// row with only column 0 set). Row 1 is `4 - 2q` for `q = 8..15` repeated (sum -1216, column 0 is -12).
#[test]
fn resident_qmm_multiplies_every_activation_row() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = kernels
        .import_packed(&fixture_weight(Dtype::Bf16, 64))
        .unwrap();
    let ones = vec![1.0f32; 64];
    let mut first_only = vec![0.0f32; 64];
    first_only[0] = 1.0;
    let twos = vec![2.0f32; 64];
    let activation = kernels
        .import_tensor(&rows_tensor(&[ones, first_only, twos]))
        .unwrap();

    let product = kernels.qmm(&activation, &weight).unwrap();

    assert_eq!(product.shape(), [3, 2]);
    let output = kernels.readback(&product).unwrap();
    assert_rows_close(
        &read_f32(&output),
        &[-288.0, -1216.0, -8.0, -12.0, -576.0, -2432.0],
    );
}

#[test]
fn resident_qmm_handles_more_rows_than_one_tile() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = kernels
        .import_packed(&fixture_weight(Dtype::Bf16, 64))
        .unwrap();
    let rows: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32 + 1.0; 64]).collect();
    let activation = kernels.import_tensor(&rows_tensor(&rows)).unwrap();

    let product = kernels.qmm(&activation, &weight).unwrap();

    assert_eq!(product.shape(), [40, 2]);
    let expected: Vec<f32> = (0..40)
        .flat_map(|i| {
            let scale = i as f32 + 1.0;
            [-288.0 * scale, -1216.0 * scale]
        })
        .collect();
    let output = kernels.readback(&product).unwrap();
    assert_rows_close(&read_f32(&output), &expected);
}

#[test]
fn resident_qmm_rejects_misshapen_activations_and_stays_usable() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = kernels
        .import_packed(&fixture_weight(Dtype::Bf16, 64))
        .unwrap();
    for shape in [vec![0u32, 64], vec![2, 32], vec![1, 2, 64], vec![64]] {
        let elements: usize = shape.iter().map(|&dimension| dimension as usize).product();
        let host = Tensor::try_from_host(Dtype::F32, shape.clone(), vec![0; elements * 4]).unwrap();
        let activation = kernels.import_tensor(&host).unwrap();

        let result = kernels.qmm(&activation, &weight);

        assert!(
            matches!(&result, Err(inference_engine::EngError::BadRequest(_))),
            "shape {shape:?}: {:?}",
            result.as_ref().err()
        );
    }
    let valid = kernels
        .import_tensor(&rows_tensor(&[vec![1.0; 64]]))
        .unwrap();
    let product = kernels.qmm(&valid, &weight).unwrap();
    assert_eq!(product.shape(), [1, 2]);
    let output = kernels.readback(&product).unwrap();
    assert_rows_close(&read_f32(&output), &[-288.0, -1216.0]);
}
