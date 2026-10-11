#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::{MlxKernels, ReduceOp},
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

fn u32_values(tensor: &Tensor) -> Vec<u32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
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

#[test]
fn reductions_match_known_values() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();

    let sum = kernels.reduce(ReduceOp::Sum, &x, 1, false).unwrap();
    let mean = kernels.reduce(ReduceOp::Mean, &x, 0, true).unwrap();
    let max = kernels.reduce(ReduceOp::Max, &x, 1, true).unwrap();

    assert_eq!(sum.shape(), [2]);
    assert_close(&values_of(&kernels.readback(&sum).unwrap()), &[6.0, 15.0]);
    assert_eq!(mean.shape(), [1, 3]);
    assert_close(
        &values_of(&kernels.readback(&mean).unwrap()),
        &[2.5, 3.5, 4.5],
    );
    assert_eq!(max.shape(), [2, 1]);
    assert_close(&values_of(&kernels.readback(&max).unwrap()), &[3.0, 6.0]);
}

#[test]
fn reshape_and_expand_dims_keep_element_order() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();

    let reshaped = kernels.reshape(&x, &[3, 2]).unwrap();
    let expanded = kernels.expand_dims(&x, 1).unwrap();

    assert_eq!(reshaped.shape(), [3, 2]);
    assert_close(
        &values_of(&kernels.readback(&reshaped).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    assert_eq!(expanded.shape(), [2, 1, 3]);
    assert_close(
        &values_of(&kernels.readback(&expanded).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn transpose_reads_back_in_row_major_order() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();

    let transposed = kernels.transpose(&x, &[1, 0]).unwrap();

    assert_eq!(transposed.shape(), [3, 2]);
    assert_close(
        &values_of(&kernels.readback(&transposed).unwrap()),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
}

#[test]
fn repeat_repeats_each_row() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let y = kernels
        .import_tensor(&f32_input(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    let repeated = kernels.repeat(&y, 3, 0).unwrap();

    assert_eq!(repeated.shape(), [6, 2]);
    assert_close(
        &values_of(&kernels.readback(&repeated).unwrap()),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
    );
}

#[test]
fn shape_operations_reject_bad_requests() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let valid = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let foreign = other
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let integer_host =
        Tensor::try_from_host(Dtype::U32, vec![2], 1u32.to_le_bytes().repeat(2)).unwrap();
    let integer = kernels.import_tensor(&integer_host).unwrap();

    assert!(
        matches!(
            kernels.reduce(ReduceOp::Sum, &foreign, 0, false),
            Err(EngError::BadRequest(_))
        ),
        "1: reduce of a foreign owner must be BadRequest"
    );
    assert!(
        matches!(
            kernels.reduce(ReduceOp::Max, &integer, 0, false),
            Err(EngError::BadRequest(_))
        ),
        "2: reduce of a U32 input must be BadRequest"
    );
    assert!(
        matches!(
            kernels.reshape(&foreign, &[6]),
            Err(EngError::BadRequest(_))
        ),
        "3: reshape of a foreign owner must be BadRequest"
    );
    assert!(
        matches!(kernels.reshape(&valid, &[4]), Err(EngError::BadRequest(_))),
        "4: reshape with another element count must be BadRequest"
    );
    assert!(
        matches!(
            kernels.transpose(&valid, &[0, 0]),
            Err(EngError::BadRequest(_))
        ),
        "5: transpose with a repeated axis must be BadRequest"
    );
    assert!(
        matches!(
            kernels.expand_dims(&integer, 0),
            Err(EngError::BadRequest(_))
        ),
        "6: expand_dims of a U32 input must be BadRequest"
    );
    assert!(
        matches!(kernels.repeat(&foreign, 2, 0), Err(EngError::BadRequest(_))),
        "7: repeat of a foreign owner must be BadRequest"
    );
    assert!(
        matches!(kernels.repeat(&valid, 2, 2), Err(EngError::BadRequest(_))),
        "8: repeat along an out-of-range axis must be BadRequest"
    );

    let still_usable = kernels.reshape(&valid, &[6]).unwrap();
    assert_close(
        &values_of(&kernels.readback(&still_usable).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn slice_selects_a_window() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();

    let window = kernels.slice(&x, &[0, 1], &[2, 3]).unwrap();

    assert_eq!(window.shape(), [2, 2]);
    assert_close(
        &values_of(&kernels.readback(&window).unwrap()),
        &[2.0, 3.0, 5.0, 6.0],
    );
}

#[test]
fn slice_of_a_u32_tensor_keeps_the_dtype_and_the_exact_values() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // 16_777_217 (2^24 + 1) and 4_000_000_000 do not survive a trip through F32.
    let ids = kernels
        .import_tensor(&u32_input(
            vec![2, 3],
            &[1, 16_777_217, 4_000_000_000, 7, 8, 9],
        ))
        .unwrap();

    let window = kernels.slice(&ids, &[0, 1], &[1, 3]).unwrap();

    assert_eq!(window.dtype(), Dtype::U32, "the slice keeps the dtype");
    assert_eq!(window.shape(), [1, 2]);
    let host = kernels.readback(&window).unwrap();
    assert_eq!(host.dtype(), Dtype::U32, "readback dtype");
    assert_eq!(u32_values(&host), [16_777_217, 4_000_000_000]);
}

#[test]
fn slice_rejects_a_dtype_other_than_f32_and_u32() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let wide = kernels
        .import_tensor(&Tensor::try_from_host(Dtype::Bf16, vec![2, 2], vec![0u8; 8]).unwrap())
        .unwrap();

    let rejected = kernels.slice(&wide, &[0, 0], &[1, 1]);

    match rejected {
        Err(EngError::BadRequest(message)) => {
            assert_eq!(message, "slice: input dtype must be F32 or U32");
        }
        other => panic!("expected BadRequest, got {:?}", other.err()),
    }
}

#[test]
fn slice_update_returns_a_new_tensor() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let update = kernels
        .import_tensor(&f32_input(vec![1, 2], &[10.0, 20.0]))
        .unwrap();

    let updated = kernels.slice_update(&x, &update, &[1, 1]).unwrap();

    assert_eq!(updated.shape(), [2, 3]);
    assert_close(
        &values_of(&kernels.readback(&updated).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
    );
    assert_close(
        &values_of(&kernels.readback(&x).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn concatenate_joins_parts_in_order() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let a = kernels
        .import_tensor(&f32_input(vec![1, 3], &[1.0, 2.0, 3.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_input(vec![2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
        .unwrap();
    let c = kernels
        .import_tensor(&f32_input(vec![2, 1], &[7.0, 8.0]))
        .unwrap();

    let rows = kernels.concatenate(&[&a, &b], 0).unwrap();
    let columns = kernels.concatenate(&[&x, &c], 1).unwrap();

    assert_eq!(rows.shape(), [3, 3]);
    assert_close(
        &values_of(&kernels.readback(&rows).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    assert_eq!(columns.shape(), [2, 4]);
    assert_close(
        &values_of(&kernels.readback(&columns).unwrap()),
        &[1.0, 2.0, 3.0, 7.0, 4.0, 5.0, 6.0, 8.0],
    );
}

#[test]
fn slice_and_concatenate_reject_bad_requests() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let valid = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let foreign = other
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let integer_row_host =
        Tensor::try_from_host(Dtype::U32, vec![1, 2], 1u32.to_le_bytes().repeat(2)).unwrap();
    let integer_row = kernels.import_tensor(&integer_row_host).unwrap();
    let integer_block_host =
        Tensor::try_from_host(Dtype::U32, vec![2, 3], 1u32.to_le_bytes().repeat(6)).unwrap();
    let integer_block = kernels.import_tensor(&integer_block_host).unwrap();

    assert!(
        matches!(
            kernels.slice(&foreign, &[0, 0], &[1, 1]),
            Err(EngError::BadRequest(_))
        ),
        "1: slice of a foreign owner must be BadRequest"
    );
    assert!(
        matches!(
            kernels.slice(&valid, &[0, 0], &[3, 1]),
            Err(EngError::BadRequest(_))
        ),
        "2: slice past the input bounds must be BadRequest"
    );
    assert!(
        matches!(
            kernels.slice_update(&valid, &foreign, &[0, 0]),
            Err(EngError::BadRequest(_))
        ),
        "3: slice update with a foreign update must be BadRequest"
    );
    assert!(
        matches!(
            kernels.slice_update(&valid, &integer_row, &[0, 0]),
            Err(EngError::BadRequest(_))
        ),
        "4: slice update with a U32 update must be BadRequest"
    );
    assert!(
        matches!(
            kernels.concatenate(&[&valid, &foreign], 0),
            Err(EngError::BadRequest(_))
        ),
        "5: concatenate with a foreign part must be BadRequest"
    );
    assert!(
        matches!(kernels.concatenate(&[], 0), Err(EngError::BadRequest(_))),
        "6: concatenate of no parts must be BadRequest"
    );
    assert!(
        matches!(
            kernels.concatenate(&[&valid, &integer_block], 0),
            Err(EngError::BadRequest(_))
        ),
        "7: concatenate with a U32 part must be BadRequest"
    );

    let joined = kernels.concatenate(&[&valid], 0).unwrap();
    assert_close(
        &values_of(&kernels.readback(&joined).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn transpose_moves_three_axes() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let t = kernels
        .import_tensor(&f32_input(vec![2, 3, 4], &values))
        .unwrap();

    let transposed = kernels.transpose(&t, &[2, 0, 1]).unwrap();

    assert_eq!(transposed.shape(), [4, 2, 3]);
    assert_close(
        &values_of(&kernels.readback(&transposed).unwrap()),
        &[
            0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0,
            18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
        ],
    );
}

#[test]
fn repeat_uses_the_named_middle_axis() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let y = kernels
        .import_tensor(&f32_input(
            vec![2, 2, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ))
        .unwrap();

    let repeated = kernels.repeat(&y, 2, 1).unwrap();

    assert_eq!(repeated.shape(), [2, 4, 2]);
    assert_close(
        &values_of(&kernels.readback(&repeated).unwrap()),
        &[
            0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 7.0,
        ],
    );
}

#[test]
fn repeat_of_a_unit_axis_reads_back() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let row = kernels
        .import_tensor(&f32_input(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    // MLX can return a broadcast view for a size-1 axis; readback needs a row-contiguous array.
    let repeated = kernels.repeat(&row, 3, 0).unwrap();

    assert_eq!(repeated.shape(), [3, 4]);
    assert_close(
        &values_of(&kernels.readback(&repeated).unwrap()),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
    );
}
