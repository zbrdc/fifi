#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::{
    collections::BTreeSet,
    env, fs,
    sync::{Mutex, MutexGuard, PoisonError},
};

use inference_engine::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use sha2::{Digest, Sha256};

const ROWS_FIXTURE_SHA256: &str =
    "28b3bee672e2cb6bdb370b7864efeb84cf726c673edffddd87470e9c0308156a";

static GPU_TEST: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    GPU_TEST.lock().unwrap_or_else(PoisonError::into_inner)
}

fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
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

fn f32_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn json_shape(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("fixture shape must be an array")
        .iter()
        .map(|dim| u32::try_from(dim.as_u64().expect("fixture dimension")).unwrap())
        .collect()
}

fn fixture_bytes(value: &serde_json::Value) -> Vec<u8> {
    assert_eq!(value["dtype"], "float32", "fixture dtype must be float32");
    value["bits"]
        .as_array()
        .expect("fixture bits must be an array")
        .iter()
        .flat_map(|word| {
            let word = u32::try_from(word.as_u64().expect("fixture bit pattern")).unwrap();
            word.to_le_bytes()
        })
        .collect()
}

fn import_fixture_tensor(kernels: &mut MlxKernels, value: &serde_json::Value) -> ExecutionTensor {
    let source = Tensor::try_from_host(
        Dtype::F32,
        json_shape(&value["shape"]),
        fixture_bytes(value),
    )
    .expect("fixture tensor must be well formed");
    kernels.import_tensor(&source).unwrap()
}

fn run_fixture_operation(
    kernels: &mut MlxKernels,
    cell: &serde_json::Value,
    inputs: &[ExecutionTensor],
) -> ExecutionTensor {
    match (cell["op"].as_str().expect("fixture operation"), inputs) {
        ("broadcast_rows", [x]) => {
            let rows = u32::try_from(cell["rows"].as_u64().expect("broadcast row count")).unwrap();
            kernels.broadcast_rows(x, rows)
        }
        ("weighted_row_sum", [rows, weights]) => kernels.weighted_row_sum(rows, weights),
        (op, _) => panic!(
            "unsupported row operation {op} with {} inputs",
            inputs.len()
        ),
    }
    .unwrap()
}

#[test]
fn broadcast_rows_repeats_a_row_and_rejects_zero_rows() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_tensor(vec![1, 2], &[1.0, 2.0]))
        .unwrap();

    let output = kernels.broadcast_rows(&x, 3).unwrap();

    assert_eq!(output.shape(), [3, 2]);
    assert_eq!(
        f32_values(&kernels.readback(&output).unwrap()),
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
    assert!(matches!(
        kernels.broadcast_rows(&x, 0),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn row_operations_match_frozen_pinned_mlx_outputs_bit_exactly() {
    let _serial = serial();
    let Some(path) = env::var_os("ROWS_FIXTURES") else {
        eprintln!(
            "SKIPPED: row_operations_match_frozen_pinned_mlx_outputs_bit_exactly: ROWS_FIXTURES is not set; frozen row-operation cells were not compared"
        );
        return;
    };
    let bytes = fs::read(path).expect("the row fixture file must be readable");
    let actual_hash = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        actual_hash, ROWS_FIXTURE_SHA256,
        "row fixture hash mismatch"
    );
    let fixture: serde_json::Value =
        serde_json::from_slice(&bytes).expect("the row fixture must be valid JSON");
    assert_eq!(fixture["schema"], "resident-rows-1");
    let cells = fixture["cells"].as_array().expect("fixture cells");
    assert_eq!(cells.len(), 4, "the pinned fixture must contain four cells");
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut operations = BTreeSet::new();

    for cell in cells {
        let inputs = cell["inputs"]
            .as_array()
            .expect("fixture inputs")
            .iter()
            .map(|input| import_fixture_tensor(&mut kernels, input))
            .collect::<Vec<_>>();
        let output = run_fixture_operation(&mut kernels, cell, &inputs);
        let expected = &cell["output"];
        assert_eq!(output.dtype(), Dtype::F32, "{} dtype", cell["id"]);
        assert_eq!(
            output.shape(),
            json_shape(&expected["shape"]),
            "{} shape",
            cell["id"]
        );
        let actual = kernels.readback(&output).unwrap();
        assert_eq!(
            actual.host_bytes().unwrap(),
            fixture_bytes(expected),
            "{} bits",
            cell["id"]
        );
        operations.insert(cell["op"].as_str().unwrap().to_owned());
    }
    assert_eq!(
        operations.len(),
        2,
        "the fixture must cover both row operations"
    );
    eprintln!(
        "compared {} frozen row-operation cells bit-exactly",
        cells.len()
    );
}

#[test]
fn weighted_row_sum_matches_hand_cell() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let rows = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let weights = kernels
        .import_tensor(&f32_tensor(vec![2, 1, 1], &[0.25, 0.75]))
        .unwrap();

    let output = kernels.weighted_row_sum(&rows, &weights).unwrap();

    assert_eq!(output.shape(), [1, 2]);
    assert_eq!(f32_values(&kernels.readback(&output).unwrap()), [2.5, 3.5]);
}

#[test]
fn row_operations_reject_foreign_and_non_f32_inputs() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let valid = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let foreign = other
        .import_tensor(&f32_tensor(vec![1, 2], &[1.0, 2.0]))
        .unwrap();
    let integer_source =
        Tensor::try_from_host(Dtype::U32, vec![1, 2], 1u32.to_le_bytes().repeat(2)).unwrap();
    let integer = kernels.import_tensor(&integer_source).unwrap();
    let weights = kernels
        .import_tensor(&f32_tensor(vec![2, 1, 1], &[0.25, 0.75]))
        .unwrap();

    assert!(matches!(
        kernels.broadcast_rows(&foreign, 2),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.broadcast_rows(&integer, 2),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.weighted_row_sum(&foreign, &weights),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.weighted_row_sum(&valid, &integer),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn row_operations_reject_shape_mismatches() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let rows = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let wrong_k = kernels
        .import_tensor(&f32_tensor(vec![3, 1, 1], &[0.25, 0.5, 0.75]))
        .unwrap();
    let wrong_weight_shape = kernels
        .import_tensor(&f32_tensor(vec![2, 1], &[0.25, 0.75]))
        .unwrap();
    let empty_row = kernels.import_tensor(&f32_tensor(vec![1, 0], &[])).unwrap();
    let empty_rows = kernels.import_tensor(&f32_tensor(vec![0, 2], &[])).unwrap();
    let empty_weights = kernels
        .import_tensor(&f32_tensor(vec![0, 1, 1], &[]))
        .unwrap();

    assert!(matches!(
        kernels.broadcast_rows(&x, 2),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.broadcast_rows(&empty_row, 2),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.weighted_row_sum(&rows, &wrong_k),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.weighted_row_sum(&rows, &wrong_weight_shape),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        kernels.weighted_row_sum(&empty_rows, &empty_weights),
        Err(EngError::BadRequest(_))
    ));
}
