#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
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
fn cache_limit_round_trips() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let previous = kernels.set_cache_limit(1 << 20).unwrap();
    assert_eq!(
        kernels.set_cache_limit(previous).unwrap(),
        1 << 20,
        "restoring {previous} must report it back"
    );
}

#[test]
fn memory_stats_count_a_resident() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.reset_peak_memory().unwrap();
    let before = kernels.memory_stats().unwrap();
    let big = kernels
        .import_tensor(&f32_input(vec![1 << 20], &vec![1.0; 1 << 20]))
        .unwrap();
    let after = kernels.memory_stats().unwrap();
    assert!(
        after.active >= before.active + (4 << 20),
        "active {after:?} must grow by 4 MiB over {before:?}"
    );
    assert!(
        after.peak >= after.active,
        "peak {after:?} must cover the active bytes"
    );
    drop(big);
}

#[test]
fn import_packed_many_with_returns_weights_and_plain_tensors() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let w = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let gates = f32_input(vec![3], &[0.25, 0.5, 0.75]);

    let (weights, tensors) = kernels.import_packed_many_with(&[&w], &[&gates]).unwrap();
    assert_eq!(weights.len(), 1, "one packed weight must return");
    assert_eq!(tensors.len(), 1, "one plain tensor must return");
    assert_close(
        &values_of(&kernels.readback(&tensors[0]).unwrap()),
        &[0.25, 0.5, 0.75],
    );

    let expanded = kernels.dequantize(&weights[0]).unwrap();
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
#[cfg(feature = "native-fault-qualification")]
fn import_packed_many_with_uses_one_synchronization() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let w = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let gates = f32_input(vec![3], &[0.25, 0.5, 0.75]);

    let before = kernels.qualification_sync_count();
    let (weights, tensors) = kernels.import_packed_many_with(&[&w], &[&gates]).unwrap();
    assert_eq!(weights.len(), 1);
    assert_eq!(tensors.len(), 1);
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "weights and plain tensors must share one synchronization"
    );
}

#[test]
fn stack_logical_elements_count_every_axis() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let stack = affine4(vec![3, 2, 64], 0x7654_3210, &[1.0; 6], &[0.0; 6]);
    let resident_stack = kernels.import_packed(&stack).unwrap();
    assert_eq!(
        resident_stack.logical_elements(),
        384,
        "a [3, 2, 64] stack holds 3 x 2 x 64 logical elements"
    );

    let matrix = affine4(vec![2, 64], 0x7654_3210, &[1.0, 2.0], &[0.0, 0.5]);
    let resident_matrix = kernels.import_packed(&matrix).unwrap();
    assert_eq!(
        resident_matrix.logical_elements(),
        128,
        "a rank-2 weight keeps its row-by-column count"
    );
}
