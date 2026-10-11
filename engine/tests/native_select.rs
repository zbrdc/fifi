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
