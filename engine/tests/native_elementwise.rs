#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::{env, fs, sync::Mutex};

use sha2::{Digest, Sha256};

use inference_engine::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());
const ELEMENTWISE_FIXTURE_SHA256: &str =
    "09641c6406a45c197bb5587d594afe55fc6c53bea715458f32f46bb0173bf287";

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

#[test]
fn add_matches_ones_cell_and_rejects_shape_mismatch() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let ones = kernels
        .import_tensor(&f32_tensor(vec![2, 3], &[1.0; 6]))
        .unwrap();
    let zeros = kernels
        .import_tensor(&f32_tensor(vec![2, 3], &[0.0; 6]))
        .unwrap();
    let mismatch = kernels
        .import_tensor(&f32_tensor(vec![6], &[0.0; 6]))
        .unwrap();

    let output = kernels.add(&ones, &zeros).unwrap();

    assert_eq!(output.shape(), [2, 3]);
    assert_eq!(f32_values(&kernels.readback(&output).unwrap()), [1.0; 6]);
    assert!(matches!(
        kernels.add(&ones, &mismatch),
        Err(EngError::BadRequest(message)) if message.contains("add")
    ));
}

#[test]
fn mul_matches_products_and_rejects_shape_mismatch() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let a = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let b = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[5.0, 6.0, 7.0, 8.0]))
        .unwrap();
    let mismatch = kernels
        .import_tensor(&f32_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    let output = kernels.mul(&a, &b).unwrap();

    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(
        f32_values(&kernels.readback(&output).unwrap()),
        [5.0, 12.0, 21.0, 32.0]
    );
    assert!(matches!(
        kernels.mul(&a, &mismatch),
        Err(EngError::BadRequest(message)) if message.contains("multiply")
    ));
}

#[test]
fn softmax_matches_hand_vector_and_rejects_scalar() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_tensor(vec![2], &[0.0, 1.0]))
        .unwrap();
    let scalar = kernels.import_tensor(&f32_tensor(vec![], &[1.0])).unwrap();

    let output = kernels.softmax(&x).unwrap();
    let actual = f32_values(&kernels.readback(&output).unwrap());

    assert_eq!(output.shape(), [2]);
    assert!((actual[0] - 0.26894143f32).abs() <= 1e-7);
    assert!((actual[1] - 0.7310586f32).abs() <= 1e-7);
    assert!(matches!(
        kernels.softmax(&scalar),
        Err(EngError::BadRequest(message)) if message.contains("softmax")
    ));
}

#[test]
fn silu_matches_hand_cells() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_tensor(vec![2], &[0.0, 1.0]))
        .unwrap();

    let output = kernels.silu(&x).unwrap();
    let actual = f32_values(&kernels.readback(&output).unwrap());

    assert_eq!(output.dtype(), Dtype::F32);
    assert_eq!(output.shape(), [2]);
    assert_eq!(actual[0], 0.0);
    assert!((actual[1] - 0.7310586f32).abs() <= 1e-7);
}

fn integer_tensor(kernels: &mut MlxKernels) -> ExecutionTensor {
    let bytes = 1u32.to_le_bytes().repeat(2);
    let source = Tensor::try_from_host(Dtype::U32, vec![2], bytes).unwrap();
    kernels.import_tensor(&source).unwrap()
}

#[test]
fn elementwise_rejects_foreign_and_integer_inputs_before_dispatch() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let valid = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let foreign = other
        .import_tensor(&f32_tensor(vec![2], &[1.0, 2.0]))
        .unwrap();
    let integer = integer_tensor(&mut kernels);

    for result in [
        kernels.silu(&foreign),
        kernels.silu(&integer),
        kernels.softmax(&foreign),
        kernels.softmax(&integer),
        kernels.mul(&valid, &foreign),
        kernels.mul(&integer, &valid),
        kernels.add(&foreign, &valid),
        kernels.add(&valid, &integer),
    ] {
        assert!(matches!(result, Err(EngError::BadRequest(_))));
    }
    let output = kernels.add(&valid, &valid).unwrap();
    assert_eq!(f32_values(&kernels.readback(&output).unwrap()), [2.0, 4.0]);
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
    op: &str,
    inputs: &[ExecutionTensor],
) -> ExecutionTensor {
    match (op, inputs) {
        ("silu", [x]) => kernels.silu(x),
        ("softmax", [x]) => kernels.softmax(x),
        ("mul", [a, b]) => kernels.mul(a, b),
        ("add", [a, b]) => kernels.add(a, b),
        _ => panic!(
            "unsupported fixture operation {op} with {} inputs",
            inputs.len()
        ),
    }
    .unwrap()
}

#[test]
fn elementwise_matches_frozen_pinned_mlx_outputs_bit_exactly() {
    let _serial = GPU_TEST.lock().unwrap();
    let Some(path) = env::var_os("ELEMENTWISE_FIXTURES") else {
        eprintln!(
            "SKIPPED: elementwise_matches_frozen_pinned_mlx_outputs_bit_exactly: ELEMENTWISE_FIXTURES is not set; frozen elementwise cells were not compared"
        );
        return;
    };
    let bytes = fs::read(&path).expect("the elementwise fixture file must be readable");
    let actual_hash = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        actual_hash, ELEMENTWISE_FIXTURE_SHA256,
        "elementwise fixture hash mismatch"
    );
    let fixture: serde_json::Value =
        serde_json::from_slice(&bytes).expect("the elementwise fixture must be valid JSON");
    assert_eq!(fixture["schema"], "resident-elementwise-1");
    let cells = fixture["cells"].as_array().expect("fixture cells");
    // The pinned fixture holds nine cells. The count comes from the frozen file, not from this code.
    assert_eq!(cells.len(), 9, "the pinned fixture must contain nine cells");
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut operations = std::collections::BTreeSet::new();

    for cell in cells {
        let op = cell["op"].as_str().expect("fixture operation");
        let inputs = cell["inputs"]
            .as_array()
            .expect("fixture inputs")
            .iter()
            .map(|input| import_fixture_tensor(&mut kernels, input))
            .collect::<Vec<_>>();
        let output = run_fixture_operation(&mut kernels, op, &inputs);
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
        operations.insert(op.to_owned());
    }
    assert_eq!(
        operations.len(),
        4,
        "the fixture must cover all four operations"
    );
    eprintln!(
        "compared {} frozen elementwise cells bit-exactly",
        cells.len()
    );
}

#[test]
fn lazy_chain_release_before_readback_matches_the_frozen_output_and_keeps_lease_rules() {
    let _serial = GPU_TEST.lock().unwrap();
    let Some(path) = env::var_os("ELEMENTWISE_FIXTURES") else {
        eprintln!(
            "SKIPPED: lazy_chain_release_before_readback_matches_the_frozen_output_and_keeps_lease_rules: ELEMENTWISE_FIXTURES is not set"
        );
        return;
    };
    let bytes = fs::read(path).expect("the elementwise fixture file must be readable");
    let actual_hash = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        actual_hash, ELEMENTWISE_FIXTURE_SHA256,
        "elementwise fixture hash mismatch"
    );
    let fixture: serde_json::Value =
        serde_json::from_slice(&bytes).expect("the elementwise fixture must be valid JSON");
    assert_eq!(fixture["schema"], "resident-elementwise-1");
    let cell = fixture["cells"]
        .as_array()
        .expect("fixture cells")
        .iter()
        .find(|cell| cell["id"] == "silu-hand")
        .expect("frozen silu-hand cell");
    assert_eq!(cell["op"], "silu");
    let expected = &cell["output"];

    let mut kernels = MlxKernels::try_new().unwrap();
    let source = import_fixture_tensor(&mut kernels, &cell["inputs"][0]);
    let intermediate = kernels.silu(&source).unwrap();
    let ones_host = f32_tensor(vec![2], &[1.0, 1.0]);
    let ones = kernels.import_tensor(&ones_host).unwrap();
    drop(ones_host);
    let output = kernels.mul(&intermediate, &ones).unwrap();

    drop(intermediate);
    drop(source);
    drop(ones);
    assert_eq!(output.dtype(), Dtype::F32);
    assert_eq!(output.shape(), json_shape(&expected["shape"]));
    let actual = kernels.readback(&output).unwrap();
    assert_eq!(actual.host_bytes().unwrap(), fixture_bytes(expected));
    eprintln!(
        "released lazy elementwise inputs before readback; compared the frozen silu-hand output bit-exactly"
    );

    let retained = output.share();
    drop(output);
    drop(kernels);
    let mut second = MlxKernels::try_new().unwrap();
    assert!(matches!(
        second.readback(&retained),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        MlxKernels::try_new(),
        Err(EngError::Backend {
            operation: "initialize",
            ..
        })
    ));
    drop(retained);
    let replacement = MlxKernels::try_new().unwrap();
    assert_ne!(replacement.owner_key(), second.owner_key());
    drop(replacement);
    drop(second);
}

#[test]
fn scalar_inputs_round_trip_and_keep_the_adapter_usable() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let one = kernels.import_tensor(&f32_tensor(vec![], &[1.0])).unwrap();
    let two = kernels.import_tensor(&f32_tensor(vec![], &[2.0])).unwrap();

    let silu = kernels.silu(&one).unwrap();
    let sum = kernels.add(&one, &two).unwrap();
    let product = kernels.mul(&two, &two).unwrap();

    for (output, expected) in [(&silu, 0.7310586f32), (&sum, 3.0), (&product, 4.0)] {
        assert!(output.shape().is_empty());
        let values = f32_values(&kernels.readback(output).unwrap());
        assert_eq!(values.len(), 1);
        assert!((values[0] - expected).abs() <= 1e-7);
    }
    let later = kernels.add(&sum, &product).unwrap();
    assert_eq!(f32_values(&kernels.readback(&later).unwrap()), [7.0]);
}

#[test]
fn zero_element_inputs_are_rejected_before_dispatch() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let empty = kernels.import_tensor(&f32_tensor(vec![0], &[])).unwrap();
    let empty_rows = kernels.import_tensor(&f32_tensor(vec![0, 3], &[])).unwrap();
    let probe = kernels.import_tensor(&f32_tensor(vec![1], &[1.0])).unwrap();

    for (label, result) in [
        ("SiLU", kernels.silu(&empty)),
        ("elementwise add", kernels.add(&empty, &empty)),
        ("elementwise multiply", kernels.mul(&empty, &empty)),
        ("softmax", kernels.softmax(&empty_rows)),
    ] {
        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains(label)),
            "{label}: {:?}",
            result.err()
        );
    }
    let output = kernels.add(&probe, &probe).unwrap();
    assert_eq!(f32_values(&kernels.readback(&output).unwrap()), [2.0]);
}
