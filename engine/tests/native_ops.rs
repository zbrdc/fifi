#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::{BinaryOp, MlxKernels, UnaryOp},
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

// 1e-6 relative is about 8 F32 ulp. It covers the Metal exp, sigmoid, rsqrt, and logaddexp approximations.
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
fn unary_operations_match_known_values() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![4], &[0.0, 2.0, -3.0, 4.0]))
        .unwrap();
    let roots = kernels
        .import_tensor(&f32_input(vec![4], &[1.0, 4.0, 9.0, 16.0]))
        .unwrap();

    let sigmoid = kernels.unary(UnaryOp::Sigmoid, &x).unwrap();
    let exp = kernels.unary(UnaryOp::Exp, &x).unwrap();
    let negative = kernels.unary(UnaryOp::Neg, &x).unwrap();
    let abs = kernels.unary(UnaryOp::Abs, &x).unwrap();
    let sign = kernels.unary(UnaryOp::Sign, &x).unwrap();
    let sqrt = kernels.unary(UnaryOp::Sqrt, &roots).unwrap();
    let rsqrt = kernels.unary(UnaryOp::Rsqrt, &roots).unwrap();

    assert_eq!(sigmoid.shape(), [4]);
    assert_close(
        &values_of(&kernels.readback(&sigmoid).unwrap()),
        &[0.5, 0.880_797_1, 0.047_425_87, 0.982_013_8],
    );
    assert_close(
        &values_of(&kernels.readback(&exp).unwrap()),
        &[1.0, 7.389_056, 0.049_787_07, 54.598_15],
    );
    assert_close(
        &values_of(&kernels.readback(&negative).unwrap()),
        &[-0.0, -2.0, 3.0, -4.0],
    );
    assert!(
        values_of(&kernels.readback(&negative).unwrap())[0].is_sign_negative(),
        "negative of 0.0 must be -0.0"
    );
    assert_close(
        &values_of(&kernels.readback(&abs).unwrap()),
        &[0.0, 2.0, 3.0, 4.0],
    );
    assert_close(
        &values_of(&kernels.readback(&sign).unwrap()),
        &[0.0, 1.0, -1.0, 1.0],
    );
    assert_close(
        &values_of(&kernels.readback(&sqrt).unwrap()),
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_close(
        &values_of(&kernels.readback(&rsqrt).unwrap()),
        &[1.0, 0.5, 0.333_333_34, 0.25],
    );
}

#[test]
fn binary_operations_broadcast_a_row() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_input(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let row = kernels
        .import_tensor(&f32_input(vec![3], &[10.0, 20.0, 30.0]))
        .unwrap();

    let add = kernels.binary(BinaryOp::Add, &a, &row).unwrap();
    let sub = kernels.binary(BinaryOp::Sub, &a, &row).unwrap();
    let mul = kernels.binary(BinaryOp::Mul, &a, &row).unwrap();
    let div = kernels.binary(BinaryOp::Div, &a, &row).unwrap();
    let maximum = kernels.binary(BinaryOp::Maximum, &a, &row).unwrap();

    assert_eq!(add.shape(), [2, 3]);
    assert_close(
        &values_of(&kernels.readback(&add).unwrap()),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
    );
    assert_close(
        &values_of(&kernels.readback(&sub).unwrap()),
        &[-9.0, -18.0, -27.0, -6.0, -15.0, -24.0],
    );
    assert_close(
        &values_of(&kernels.readback(&mul).unwrap()),
        &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0],
    );
    assert_close(
        &values_of(&kernels.readback(&div).unwrap()),
        &[0.1, 0.1, 0.1, 0.4, 0.25, 0.2],
    );
    assert_close(
        &values_of(&kernels.readback(&maximum).unwrap()),
        &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
    );
}

#[test]
fn binary_operations_broadcast_a_scalar() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_input(vec![6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let scalar = kernels.import_tensor(&f32_input(vec![], &[3.5])).unwrap();

    let maximum = kernels.binary(BinaryOp::Maximum, &a, &scalar).unwrap();

    assert_eq!(maximum.shape(), [6]);
    assert_close(
        &values_of(&kernels.readback(&maximum).unwrap()),
        &[3.5, 3.5, 3.5, 4.0, 5.0, 6.0],
    );

    let x = kernels
        .import_tensor(&f32_input(vec![2], &[1.0, 2.0]))
        .unwrap();
    let zero = kernels.import_tensor(&f32_input(vec![], &[0.0])).unwrap();

    let log_add_exp = kernels.binary(BinaryOp::LogAddExp, &x, &zero).unwrap();

    assert_close(
        &values_of(&kernels.readback(&log_add_exp).unwrap()),
        &[1.313_261_7, 2.126_928],
    );
}

#[test]
fn binary_rejects_shapes_that_do_not_broadcast() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_input(vec![2, 3], &[0.0; 6]))
        .unwrap();
    let mismatch = kernels
        .import_tensor(&f32_input(vec![2], &[0.0; 2]))
        .unwrap();

    assert!(matches!(
        kernels.binary(BinaryOp::Add, &a, &mismatch),
        Err(EngError::BadRequest(_))
    ));

    let one = kernels.import_tensor(&f32_input(vec![1], &[1.0])).unwrap();
    let sum = kernels.add(&one, &one).unwrap();

    assert_close(&values_of(&kernels.readback(&sum).unwrap()), &[2.0]);
}

#[test]
fn binary_broadcasts_the_left_operand() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels.import_tensor(&f32_input(vec![], &[10.0])).unwrap();
    let b = kernels
        .import_tensor(&f32_input(vec![3], &[1.0, 2.0, 4.0]))
        .unwrap();

    let sub = kernels.binary(BinaryOp::Sub, &a, &b).unwrap();
    let div = kernels.binary(BinaryOp::Div, &a, &b).unwrap();

    assert_eq!(sub.shape(), &[3]);
    assert_close(
        &values_of(&kernels.readback(&sub).unwrap()),
        &[9.0, 8.0, 6.0],
    );
    assert_eq!(div.shape(), &[3]);
    assert_close(
        &values_of(&kernels.readback(&div).unwrap()),
        &[10.0, 5.0, 2.5],
    );
}

#[test]
fn binary_broadcasts_both_operands() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_input(vec![2, 1], &[1.0, 2.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_input(vec![1, 3], &[10.0, 20.0, 30.0]))
        .unwrap();

    let mul = kernels.binary(BinaryOp::Mul, &a, &b).unwrap();
    let sub = kernels.binary(BinaryOp::Sub, &a, &b).unwrap();

    assert_eq!(mul.shape(), &[2, 3]);
    assert_close(
        &values_of(&kernels.readback(&mul).unwrap()),
        &[10.0, 20.0, 30.0, 20.0, 40.0, 60.0],
    );
    assert_eq!(sub.shape(), &[2, 3]);
    assert_close(
        &values_of(&kernels.readback(&sub).unwrap()),
        &[-9.0, -19.0, -29.0, -8.0, -18.0, -28.0],
    );
}

#[test]
fn unary_and_binary_reject_foreign_and_non_f32_operands() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let valid = kernels
        .import_tensor(&f32_input(vec![2], &[1.0, 2.0]))
        .unwrap();
    let foreign = other
        .import_tensor(&f32_input(vec![2], &[1.0, 2.0]))
        .unwrap();
    let integer_host =
        Tensor::try_from_host(Dtype::U32, vec![2], 1u32.to_le_bytes().repeat(2)).unwrap();
    let integer = kernels.import_tensor(&integer_host).unwrap();
    let empty_host = Tensor::try_from_host(Dtype::F32, vec![0], Vec::new()).unwrap();
    let empty = kernels.import_tensor(&empty_host).unwrap();

    let case_1 = kernels.unary(UnaryOp::Sigmoid, &foreign);
    assert!(matches!(case_1, Err(EngError::BadRequest(_))), "case 1");
    let case_2 = kernels.unary(UnaryOp::Exp, &integer);
    assert!(matches!(case_2, Err(EngError::BadRequest(_))), "case 2");
    let case_3 = kernels.unary(UnaryOp::Abs, &empty);
    assert!(matches!(case_3, Err(EngError::BadRequest(_))), "case 3");
    let case_4 = kernels.binary(BinaryOp::Sub, &valid, &foreign);
    assert!(matches!(case_4, Err(EngError::BadRequest(_))), "case 4");
    let case_5 = kernels.binary(BinaryOp::Sub, &foreign, &valid);
    assert!(matches!(case_5, Err(EngError::BadRequest(_))), "case 5");
    let case_6 = kernels.binary(BinaryOp::Div, &valid, &integer);
    assert!(matches!(case_6, Err(EngError::BadRequest(_))), "case 6");
    let case_7 = kernels.binary(BinaryOp::Div, &integer, &valid);
    assert!(matches!(case_7, Err(EngError::BadRequest(_))), "case 7");
    let case_8 = kernels.binary(BinaryOp::Add, &empty, &empty);
    assert!(matches!(case_8, Err(EngError::BadRequest(_))), "case 8");

    let sum = kernels.binary(BinaryOp::Add, &valid, &valid).unwrap();
    assert_close(&values_of(&kernels.readback(&sum).unwrap()), &[2.0, 4.0]);
}
