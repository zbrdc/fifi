#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::{BinaryOp, MlxKernels},
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
fn argmax_returns_u32_indices_and_the_first_tie() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(
            vec![2, 4],
            &[1.0, 7.0, 7.0, 2.0, 9.0, 0.0, 9.0, 3.0],
        ))
        .unwrap();

    let rows = kernels.argmax(&x, 1, false).unwrap();
    assert_eq!(rows.shape(), [2]);
    assert_eq!(rows.dtype(), Dtype::U32);
    assert_eq!(
        u32_values(&kernels.readback(&rows).unwrap()),
        &[1, 0],
        "ties must keep the lowest index"
    );

    let columns = kernels.argmax(&x, 0, true).unwrap();
    assert_eq!(columns.shape(), [1, 4]);
    assert_eq!(columns.dtype(), Dtype::U32);
    assert_eq!(
        u32_values(&kernels.readback(&columns).unwrap()),
        &[1, 0, 1, 1]
    );
}

#[test]
fn argsort_is_ascending_and_stable() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let s = kernels
        .import_tensor(&f32_input(vec![6], &[3.0, 1.0, 2.0, 1.0, 3.0, 0.0]))
        .unwrap();

    let ascending = kernels.argsort(&s, 0).unwrap();
    assert_eq!(ascending.shape(), [6]);
    assert_eq!(ascending.dtype(), Dtype::U32);
    assert_eq!(
        u32_values(&kernels.readback(&ascending).unwrap()),
        &[5, 1, 3, 2, 0, 4],
        "equal values must keep input order"
    );

    let minus_one = kernels.import_tensor(&f32_input(vec![], &[-1.0])).unwrap();
    let negated = kernels.binary(BinaryOp::Mul, &s, &minus_one).unwrap();
    let descending = kernels.argsort(&negated, 0).unwrap();
    assert_eq!(descending.shape(), [6]);
    assert_eq!(descending.dtype(), Dtype::U32);
    assert_eq!(
        u32_values(&kernels.readback(&descending).unwrap()),
        &[0, 4, 2, 1, 3, 5],
        "negated scores must give descending order with ties in input order"
    );
}

#[test]
fn take_gathers_rows_by_u32_index() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let table = kernels
        .import_tensor(&f32_input(
            vec![4, 2],
            &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        ))
        .unwrap();
    let idx = kernels
        .import_tensor(&u32_input(vec![3], &[3, 0, 3]))
        .unwrap();

    let rows = kernels.take(&table, &idx, 0).unwrap();
    assert_eq!(rows.shape(), [3, 2]);
    assert_eq!(rows.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&rows).unwrap()),
        &[30.0, 31.0, 0.0, 1.0, 30.0, 31.0],
    );

    let col = kernels.import_tensor(&u32_input(vec![1], &[1])).unwrap();
    let columns = kernels.take(&table, &col, 1).unwrap();
    assert_eq!(columns.shape(), [4, 1]);
    assert_eq!(columns.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&columns).unwrap()),
        &[1.0, 11.0, 21.0, 31.0],
    );
}

#[test]
fn selection_rejects_bad_requests() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(
            vec![2, 4],
            &[1.0, 7.0, 7.0, 2.0, 9.0, 0.0, 9.0, 3.0],
        ))
        .unwrap();
    let foreign_idx = other.import_tensor(&u32_input(vec![1], &[0])).unwrap();
    let float_idx = kernels.import_tensor(&f32_input(vec![1], &[0.0])).unwrap();
    let integer = kernels.import_tensor(&u32_input(vec![2], &[0, 1])).unwrap();

    assert!(
        matches!(
            kernels.take(&x, &foreign_idx, 0),
            Err(EngError::BadRequest(_))
        ),
        "1: take with foreign indices must be BadRequest"
    );
    assert!(
        matches!(
            kernels.take(&x, &float_idx, 0),
            Err(EngError::BadRequest(_))
        ),
        "2: take with F32 indices must be BadRequest"
    );
    assert!(
        matches!(
            kernels.argmax(&integer, 0, false),
            Err(EngError::BadRequest(_))
        ),
        "3: argmax of a U32 input must be BadRequest"
    );
    assert!(
        matches!(kernels.argsort(&x, 2), Err(EngError::BadRequest(_))),
        "4: argsort with an out-of-range axis must be BadRequest"
    );
    assert!(
        matches!(
            kernels.take(&integer, &integer, 0),
            Err(EngError::BadRequest(_))
        ),
        "5: take with a U32 input must be BadRequest"
    );

    let picked = kernels.argmax(&x, 1, false).unwrap();
    assert_eq!(picked.shape(), [2]);
    assert_eq!(picked.dtype(), Dtype::U32);
    assert_eq!(u32_values(&kernels.readback(&picked).unwrap()), &[1, 0]);
}

#[test]
fn take_on_axis_one_reads_back_contiguous() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let table = kernels
        .import_tensor(&f32_input(
            vec![4, 2],
            &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        ))
        .unwrap();
    let idx = kernels.import_tensor(&u32_input(vec![2], &[1, 0])).unwrap();

    let swapped = kernels.take(&table, &idx, 1).unwrap();
    assert_eq!(swapped.shape(), [4, 2]);
    assert_eq!(swapped.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&swapped).unwrap()),
        &[1.0, 0.0, 11.0, 10.0, 21.0, 20.0, 31.0, 30.0],
    );
}

#[test]
fn take_clamps_out_of_range_indices() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let table = kernels
        .import_tensor(&f32_input(
            vec![4, 2],
            &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        ))
        .unwrap();
    let idx = kernels.import_tensor(&u32_input(vec![2], &[9, 0])).unwrap();

    let clamped = kernels.take(&table, &idx, 0).unwrap();
    assert_eq!(clamped.shape(), [2, 2]);
    assert_eq!(clamped.dtype(), Dtype::F32);
    assert_close(
        &values_of(&kernels.readback(&clamped).unwrap()),
        &[30.0, 31.0, 0.0, 1.0],
    );
}

#[test]
fn take_along_axis_gathers_one_value_per_index_cell() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // x [3, 5], one value per cell:
    //    0   1   2   3   4
    //   10  11  12  13  14
    //   20  21  22  23  24
    let x = kernels
        .import_tensor(&f32_input(
            vec![3, 5],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        ))
        .unwrap();

    // Axis 1 with indices [3, 2]: out[t, k] = x[t, idx[t, k]]. By hand, row 0 picks columns
    // 1 and 3 (1, 3), row 1 picks columns 4 and 0 (14, 10), row 2 picks columns 2 and 4 (22, 24).
    let idx = kernels
        .import_tensor(&u32_input(vec![3, 2], &[1, 3, 4, 0, 2, 4]))
        .unwrap();
    let picked = kernels.take_along_axis(&x, &idx, 1).unwrap();
    assert_eq!(picked.shape(), [3, 2], "axis 1 output shape");
    assert_eq!(picked.dtype(), Dtype::F32, "axis 1 output dtype");
    assert_close(
        &values_of(&kernels.readback(&picked).unwrap()),
        &[1.0, 3.0, 14.0, 10.0, 22.0, 24.0],
    );

    // Axis 0 with indices [2, 5]: out[a, j] = x[idx[a, j], j]. By hand, index row (2, 0, 2, 1, 1)
    // gives 20, 1, 22, 13, 14 and index row (0, 2, 1, 2, 0) gives 0, 21, 12, 23, 4.
    let rows = kernels
        .import_tensor(&u32_input(vec![2, 5], &[2, 0, 2, 1, 1, 0, 2, 1, 2, 0]))
        .unwrap();
    let picked = kernels.take_along_axis(&x, &rows, 0).unwrap();
    assert_eq!(picked.shape(), [2, 5], "axis 0 output shape");
    assert_eq!(picked.dtype(), Dtype::F32, "axis 0 output dtype");
    assert_close(
        &values_of(&kernels.readback(&picked).unwrap()),
        &[20.0, 1.0, 22.0, 13.0, 14.0, 0.0, 21.0, 12.0, 23.0, 4.0],
    );
}

#[test]
fn take_along_axis_clamps_out_of_range_indices() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(
            vec![3, 5],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        ))
        .unwrap();

    // Axis 1 spans five columns, so the index 9 clamps to column 4: by hand the rows pick
    // (0, 4), (14, 11), and (24, 24).
    let idx = kernels
        .import_tensor(&u32_input(vec![3, 2], &[0, 9, 9, 1, 4, 4]))
        .unwrap();
    let clamped = kernels.take_along_axis(&x, &idx, 1).unwrap();
    assert_eq!(clamped.shape(), [3, 2]);
    assert_close(
        &values_of(&kernels.readback(&clamped).unwrap()),
        &[0.0, 4.0, 14.0, 11.0, 24.0, 24.0],
    );

    // Axis 0 spans three rows, so the index 7 clamps to row 2: by hand the index rows
    // (2, 7, 0, 7, 7) and (7, 2, 1, 7, 0) give 20, 21, 2, 23, 24 and 20, 21, 12, 23, 4.
    let rows = kernels
        .import_tensor(&u32_input(vec![2, 5], &[2, 7, 0, 7, 7, 7, 2, 1, 7, 0]))
        .unwrap();
    let clamped = kernels.take_along_axis(&x, &rows, 0).unwrap();
    assert_eq!(clamped.shape(), [2, 5]);
    assert_close(
        &values_of(&kernels.readback(&clamped).unwrap()),
        &[20.0, 21.0, 2.0, 23.0, 24.0, 20.0, 21.0, 12.0, 23.0, 4.0],
    );
}

#[test]
fn take_along_axis_keeps_u32_values_and_bf16_operands() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // A U32 input keeps U32: the device chunk MoE gathers its sorted expert ids this way.
    // 16_777_217 (2^24 + 1) and 4_000_000_000 do not survive a trip through F32.
    let ids = kernels
        .import_tensor(&u32_input(
            vec![6],
            &[3, 16_777_217, 1, 0, 3, 4_000_000_000],
        ))
        .unwrap();
    let order = kernels
        .import_tensor(&u32_input(vec![6], &[3, 0, 1, 2, 4, 5]))
        .unwrap();
    let sorted = kernels.take_along_axis(&ids, &order, 0).unwrap();
    assert_eq!(sorted.shape(), [6]);
    assert_eq!(sorted.dtype(), Dtype::U32, "a U32 input keeps U32");
    assert_eq!(
        u32_values(&kernels.readback(&sorted).unwrap()),
        &[0, 3, 16_777_217, 1, 3, 4_000_000_000],
    );

    // A BF16 operand after `allow_bf16`: values that BF16 holds exactly survive the gather.
    kernels.allow_bf16(true);
    let source = kernels
        .import_tensor(&f32_input(vec![4], &[1.0, 2.0, 4.0, 8.0]))
        .unwrap();
    let bf16 = kernels.cast(&source, Dtype::Bf16).unwrap();
    let idx = kernels.import_tensor(&u32_input(vec![2], &[3, 0])).unwrap();
    let picked = kernels.take_along_axis(&bf16, &idx, 0).unwrap();
    assert_eq!(picked.shape(), [2]);
    assert_eq!(picked.dtype(), Dtype::Bf16, "a BF16 operand keeps BF16");
    assert_eq!(picked.byte_len(), 4, "two BF16 elements are 4 bytes");
    let host = kernels.readback(&picked).unwrap();
    let bytes = host.host_bytes().unwrap();
    let decoded: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect();
    assert_eq!(
        decoded,
        [8.0, 1.0],
        "the BF16 gather reads the picked cells"
    );

    // With the switch off, a BF16 operand is a bad request with the text of `slice`.
    kernels.allow_bf16(false);
    let refused = kernels.take_along_axis(&bf16, &idx, 0);
    assert_eq!(
        message_of(refused),
        "take_along_axis: input dtype must be F32 or U32"
    );
}

#[test]
fn take_along_axis_rejects_mismatched_layouts() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![3, 5], &[0.5; 15]))
        .unwrap();
    let rank = kernels.import_tensor(&u32_input(vec![2], &[0, 1])).unwrap();
    let width = kernels
        .import_tensor(&u32_input(vec![4, 2], &[0, 1, 0, 1, 0, 1, 0, 1]))
        .unwrap();
    let float_idx = kernels
        .import_tensor(&f32_input(vec![3, 2], &[0.0; 6]))
        .unwrap();
    assert_eq!(
        message_of(kernels.take_along_axis(&x, &rank, 1)),
        "take_along_axis: indices rank must match the input"
    );
    assert_eq!(
        message_of(kernels.take_along_axis(&x, &width, 1)),
        "take_along_axis: indices differ outside the axis"
    );
    assert_eq!(
        message_of(kernels.take_along_axis(&x, &float_idx, 1)),
        "take_along_axis: indices dtype must be U32"
    );
    assert_eq!(
        message_of(kernels.take_along_axis(&x, &rank, 2)),
        "take_along_axis: axis is out of range"
    );
    let good = kernels
        .import_tensor(&u32_input(vec![3, 2], &[0, 1, 2, 3, 4, 0]))
        .unwrap();
    assert!(
        kernels.take_along_axis(&x, &good, 1).is_ok(),
        "the matching layout is accepted"
    );
}

/// The message of a `BadRequest`, or a description of any other outcome.
fn message_of<T>(result: Result<T, EngError>) -> String {
    match result {
        Err(EngError::BadRequest(message)) => message,
        Err(other) => format!("{other:?}"),
        Ok(_) => "accepted".to_string(),
    }
}

#[test]
fn argmax_of_a_vector_is_a_scalar() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let v = kernels
        .import_tensor(&f32_input(vec![5], &[0.0, 3.0, 9.0, 9.0, 1.0]))
        .unwrap();

    let best = kernels.argmax(&v, 0, false).unwrap();
    assert!(best.shape().is_empty());
    assert_eq!(best.dtype(), Dtype::U32);
    assert_eq!(u32_values(&kernels.readback(&best).unwrap()), &[2]);
}

#[test]
fn u32_to_f32_converts_exact_values_and_keeps_the_shape() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let ids = kernels
        .import_tensor(&u32_input(vec![2, 3], &[0, 1, 255, 16_777_216, 248_319, 7]))
        .unwrap();
    let scores = kernels
        .import_tensor(&f32_input(vec![3], &[3.0, 1.0, 2.0]))
        .unwrap();
    let order = kernels.argsort(&scores, 0).unwrap();

    let converted = kernels.u32_to_f32(&ids);
    assert!(converted.is_ok(), "{:?}", converted.as_ref().err());
    let converted = converted.unwrap();
    let from_graph = kernels.u32_to_f32(&order).unwrap();

    assert_eq!(converted.dtype(), Dtype::F32);
    assert_eq!(converted.shape(), [2, 3]);
    assert_close(
        &values_of(&kernels.readback(&converted).unwrap()),
        &[0.0, 1.0, 255.0, 16_777_216.0, 248_319.0, 7.0],
    );
    // The input of the second conversion is an unevaluated argsort result: [1, 2, 0].
    assert_eq!(from_graph.shape(), [3]);
    assert_close(
        &values_of(&kernels.readback(&from_graph).unwrap()),
        &[1.0, 2.0, 0.0],
    );
}

#[test]
fn u32_to_f32_rejects_bad_requests() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let foreign = other.import_tensor(&u32_input(vec![2], &[1, 2])).unwrap();
    let floats = kernels
        .import_tensor(&f32_input(vec![2], &[1.0, 2.0]))
        .unwrap();
    let scalar = kernels.import_tensor(&u32_input(vec![], &[5])).unwrap();

    let message = |result: Result<_, EngError>| match result {
        Err(EngError::BadRequest(message)) => message,
        Ok(_) => "accepted".to_string(),
        Err(other) => format!("{other:?}"),
    };

    assert_eq!(
        message(kernels.u32_to_f32(&foreign)),
        "convert: input belongs to another native owner",
        "1: foreign owner"
    );
    assert_eq!(
        message(kernels.u32_to_f32(&floats)),
        "convert: input dtype must be U32",
        "2: F32 input"
    );
    assert!(
        message(kernels.u32_to_f32(&scalar)).contains("input shape is invalid"),
        "3: a scalar has no axis to keep"
    );
}

#[test]
fn argsort_is_stable_across_blocks() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let values: Vec<f32> = (0..5000).map(|i| (i % 3) as f32).collect();
    let s = kernels
        .import_tensor(&f32_input(vec![5000], &values))
        .unwrap();

    let ordered = kernels.argsort(&s, 0).unwrap();
    assert_eq!(ordered.shape(), [5000]);
    assert_eq!(ordered.dtype(), Dtype::U32);
    let mut expected = Vec::with_capacity(5000);
    for residue in 0..3_u32 {
        for i in 0..5000_u32 {
            if i % 3 == residue {
                expected.push(i);
            }
        }
    }
    assert_eq!(u32_values(&kernels.readback(&ordered).unwrap()), expected);
}
