#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

fn f32_input(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn u32_input(shape: Vec<u32>, values: &[u32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::U32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn values_of(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "actual values {actual:?}");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            "element {index}: actual {a}, expected {b}"
        );
    }
}

/// An affine 4-bit weight with group size 64. Every payload word is `word`, so element `c` of a
/// row is nibble `c % 8` of `word`. `scales` and `biases` give one value per row of 64 columns.
fn affine4(logical: Vec<u32>, word: u32, scales: &[f32], biases: &[f32]) -> PackedWeight {
    let mut payload_shape = logical.clone();
    let last = payload_shape.len() - 1;
    payload_shape[last] /= 8;
    let mut group_shape = logical.clone();
    group_shape[last] /= 64;
    let words = payload_shape.iter().product::<u32>() as usize;
    PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, payload_shape, word.to_le_bytes().repeat(words)).unwrap(),
        f32_input(group_shape.clone(), scales),
        Some(f32_input(group_shape, biases)),
        logical,
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .unwrap()
}

#[test]
fn gather_qmm_selects_experts_by_index() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = affine4(
        vec![3, 2, 64],
        0x7654_3210,
        &[1.0; 6],
        &[0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
    );
    let stack = kernels.import_packed(&weight).unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![1, 1, 64], &[1.0; 64]))
        .unwrap();
    let idx = kernels
        .import_tensor(&u32_input(vec![3], &[2, 0, 2]))
        .unwrap();

    let gathered = kernels.gather_qmm(&x, &stack, &idx).unwrap();
    assert_eq!(gathered.shape(), [3, 1, 2]);
    assert_eq!(gathered.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&gathered).unwrap()),
        &[13024.0, 13664.0, 224.0, 864.0, 13024.0, 13664.0],
    );

    let far = kernels.import_tensor(&u32_input(vec![1], &[7])).unwrap();
    let clamped = kernels.gather_qmm(&x, &stack, &far).unwrap();
    assert_eq!(clamped.shape(), [1, 1, 2]);
    assert_close(
        &values_of(&kernels.readback(&clamped).unwrap()),
        &[13024.0, 13664.0],
    );
}

#[test]
fn gather_qmm_uses_one_activation_row_per_index() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = affine4(
        vec![3, 2, 64],
        0x7654_3210,
        &[1.0; 6],
        &[0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
    );
    let stack = kernels.import_packed(&weight).unwrap();
    let mut rows = Vec::with_capacity(192);
    for i in 0..3_u32 {
        rows.extend([(i + 1) as f32; 64]);
    }
    let x = kernels
        .import_tensor(&f32_input(vec![3, 1, 64], &rows))
        .unwrap();
    let idx = kernels
        .import_tensor(&u32_input(vec![3], &[2, 0, 2]))
        .unwrap();

    let gathered = kernels.gather_qmm(&x, &stack, &idx).unwrap();
    assert_eq!(gathered.shape(), [3, 1, 2]);
    assert_eq!(gathered.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&gathered).unwrap()),
        &[13024.0, 13664.0, 448.0, 1728.0, 39072.0, 40992.0],
    );
}

#[test]
fn dequantize_expands_affine_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let w = kernels.import_packed(&weight).unwrap();

    let expanded = kernels.dequantize(&w).unwrap();
    assert_eq!(expanded.shape(), [2, 64]);
    assert_eq!(expanded.dtype(), Dtype::F32);
    let mut expected = Vec::with_capacity(128);
    for row in 0..2_u32 {
        for column in 0..64_u32 {
            let nibble = (column % 8) as f32;
            expected.push(if row == 0 { nibble } else { 2.0 * nibble + 0.5 });
        }
    }
    assert_close(&values_of(&kernels.readback(&expanded).unwrap()), &expected);
}

#[test]
fn dequantize_expands_a_rank_three_stack() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let weight = affine4(
        vec![3, 2, 64],
        0x7654_3210,
        &[1.0; 6],
        &[0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
    );
    let stack = kernels.import_packed(&weight).unwrap();

    let expanded = kernels.dequantize(&stack).unwrap();
    assert_eq!(expanded.shape(), [3, 2, 64]);
    assert_eq!(expanded.dtype(), Dtype::F32);
    let bias = [[0.0_f32, 10.0], [100.0, 110.0], [200.0, 210.0]];
    let mut expected = Vec::with_capacity(384);
    for expert_bias in bias {
        for row_bias in expert_bias {
            for column in 0..64_u32 {
                expected.push((column % 8) as f32 + row_bias);
            }
        }
    }
    assert_close(&values_of(&kernels.readback(&expanded).unwrap()), &expected);
}

#[test]
fn batched_matmul_broadcasts_batches() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_input(vec![2, 1, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_input(
            vec![2, 3, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ))
        .unwrap();

    let product = kernels.batched_mm(&a, &b).unwrap();
    assert_eq!(product.shape(), [2, 1, 2]);
    assert_eq!(product.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&product).unwrap()),
        &[22.0, 28.0, 139.0, 154.0],
    );

    let r = kernels
        .import_tensor(&f32_input(vec![1, 3], &[1.0, 2.0, 3.0]))
        .unwrap();
    let spread = kernels.batched_mm(&r, &b).unwrap();
    assert_eq!(spread.shape(), [2, 1, 2]);
    assert_close(
        &values_of(&kernels.readback(&spread).unwrap()),
        &[22.0, 28.0, 58.0, 64.0],
    );
}

#[test]
#[cfg(feature = "native-fault-qualification")]
fn import_packed_many_uses_one_synchronization() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let w1 = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let w2 = affine4(
        vec![3, 2, 64],
        0x7654_3210,
        &[1.0; 6],
        &[0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
    );

    let before = kernels.qualification_sync_count();
    let imported = kernels.import_packed_many(&[&w1, &w2]).unwrap();
    assert_eq!(kernels.qualification_sync_count(), before + 1);

    let expanded = kernels.dequantize(&imported[0]).unwrap();
    assert_eq!(expanded.shape(), [2, 64]);
    let mut expected = Vec::with_capacity(128);
    for row in 0..2_u32 {
        for column in 0..64_u32 {
            let nibble = (column % 8) as f32;
            expected.push(if row == 0 { nibble } else { 2.0 * nibble + 0.5 });
        }
    }
    assert_close(&values_of(&kernels.readback(&expanded).unwrap()), &expected);
    assert_eq!(imported[1].logical_shape(), [3, 2, 64]);
}

#[test]
fn import_packed_many_keeps_order_and_values() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let w1 = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let w2 = affine4(
        vec![3, 2, 64],
        0x7654_3210,
        &[1.0; 6],
        &[0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
    );

    let imported = kernels.import_packed_many(&[&w1, &w2]).unwrap();
    assert_eq!(imported.len(), 2);

    let first = kernels.dequantize(&imported[0]).unwrap();
    assert_eq!(first.shape(), [2, 64]);
    let mut rows = Vec::with_capacity(128);
    for row in 0..2_u32 {
        for column in 0..64_u32 {
            let nibble = (column % 8) as f32;
            rows.push(if row == 0 { nibble } else { 2.0 * nibble + 0.5 });
        }
    }
    assert_close(&values_of(&kernels.readback(&first).unwrap()), &rows);

    let second = kernels.dequantize(&imported[1]).unwrap();
    assert_eq!(second.shape(), [3, 2, 64]);
    let bias = [[0.0_f32, 10.0], [100.0, 110.0], [200.0, 210.0]];
    let mut expected = Vec::with_capacity(384);
    for expert_bias in bias {
        for row_bias in expert_bias {
            for column in 0..64_u32 {
                expected.push((column % 8) as f32 + row_bias);
            }
        }
    }
    assert_close(&values_of(&kernels.readback(&second).unwrap()), &expected);
}

#[test]
fn dequantize_rejects_a_foreign_weight() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let w1 = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let foreign = other.import_packed(&w1).unwrap();

    assert!(
        matches!(kernels.dequantize(&foreign), Err(EngError::BadRequest(_))),
        "dequantize of a weight imported by another owner must be BadRequest"
    );

    let own = kernels.import_packed(&w1).unwrap();
    let expanded = kernels.dequantize(&own).unwrap();
    assert_eq!(expanded.shape(), [2, 64]);
    let mut expected = Vec::with_capacity(128);
    for row in 0..2_u32 {
        for column in 0..64_u32 {
            let nibble = (column % 8) as f32;
            expected.push(if row == 0 { nibble } else { 2.0 * nibble + 0.5 });
        }
    }
    assert_close(&values_of(&kernels.readback(&expanded).unwrap()), &expected);
}

#[test]
fn quantized_operations_reject_bad_requests() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let weight = affine4(
        vec![3, 2, 64],
        0x7654_3210,
        &[1.0; 6],
        &[0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
    );
    let stack = kernels.import_packed(&weight).unwrap();
    let foreign_stack = other.import_packed(&weight).unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![1, 1, 64], &[1.0; 64]))
        .unwrap();
    let idx = kernels
        .import_tensor(&u32_input(vec![3], &[0, 1, 2]))
        .unwrap();
    let float_idx = kernels
        .import_tensor(&f32_input(vec![3], &[0.0, 1.0, 2.0]))
        .unwrap();
    let integer = kernels
        .import_tensor(&u32_input(vec![2, 3], &[0, 1, 2, 3, 4, 5]))
        .unwrap();
    let u32_activation = kernels
        .import_tensor(&u32_input(vec![1, 1, 64], &[0; 64]))
        .unwrap();
    let m = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();

    assert!(
        matches!(
            kernels.gather_qmm(&x, &foreign_stack, &idx),
            Err(EngError::BadRequest(_))
        ),
        "1: gather_qmm with a foreign stack must be BadRequest"
    );
    assert!(
        matches!(
            kernels.gather_qmm(&x, &stack, &float_idx),
            Err(EngError::BadRequest(_))
        ),
        "2: gather_qmm with F32 indices must be BadRequest"
    );
    assert!(
        matches!(
            kernels.gather_qmm(&u32_activation, &stack, &idx),
            Err(EngError::BadRequest(_))
        ),
        "3: gather_qmm with a U32 activation must be BadRequest"
    );
    assert!(
        matches!(
            kernels.batched_mm(&integer, &m),
            Err(EngError::BadRequest(_))
        ),
        "4: batched_mm with a U32 operand must be BadRequest"
    );
    assert!(
        matches!(kernels.batched_mm(&m, &m), Err(EngError::BadRequest(_))),
        "5: batched_mm with differing inner dimensions must be BadRequest"
    );

    let gathered = kernels.gather_qmm(&x, &stack, &idx).unwrap();
    assert_eq!(gathered.shape(), [3, 1, 2]);
    assert_close(
        &values_of(&kernels.readback(&gathered).unwrap()),
        &[224.0, 864.0, 6624.0, 7264.0, 13024.0, 13664.0],
    );
}

/// BF16 tensor of `values`, which must be exact in BF16 (their low 16 bits are zero).
fn bf16_input(shape: Vec<u32>, values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            assert_eq!(bits & 0xffff, 0, "{value} is not exact in BF16");
            ((bits >> 16) as u16).to_le_bytes()
        })
        .collect();
    Tensor::try_from_host(Dtype::Bf16, shape, bytes).unwrap()
}

#[test]
fn dequantize_with_bf16_scales_keeps_f32_precision() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // One affine 8-bit row of 64 columns. Every payload word holds the bytes 0, 3, 7, 255, so
    // column `c` holds code [0, 3, 7, 255][c % 4]. Scale 1.0078125 and bias 0.5 are exact in BF16.
    // The products 3.0234375, 7.0546875, and 256.99219 need more than BF16's 8 significant bits.
    let weight = PackedWeight::try_new(
        Tensor::try_from_host(
            Dtype::U32,
            vec![1, 16],
            0xff07_0300_u32.to_le_bytes().repeat(16),
        )
        .unwrap(),
        bf16_input(vec![1, 1], &[1.0078125]),
        Some(bf16_input(vec![1, 1], &[0.5])),
        vec![1, 64],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 8,
            group_size: 64,
        },
    )
    .unwrap();
    let w = kernels.import_packed(&weight).unwrap();
    let expanded = kernels.dequantize(&w).unwrap();
    assert_eq!(expanded.dtype(), Dtype::F32);
    let codes = [0.0_f32, 3.0, 7.0, 255.0];
    let expected: Vec<f32> = (0..64).map(|c| 1.0078125 * codes[c % 4] + 0.5).collect();
    assert_close(&values_of(&kernels.readback(&expanded).unwrap()), &expected);
}
