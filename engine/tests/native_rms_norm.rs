#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::{
    env, fs,
    sync::{Mutex, MutexGuard, PoisonError},
};

use sha2::{Digest, Sha256};

use inference_engine::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

/// Serializes the GPU tests. A panic in one test must not fail the others with a poison error.
fn serial() -> MutexGuard<'static, ()> {
    GPU_TEST.lock().unwrap_or_else(PoisonError::into_inner)
}
const RMS_FIXTURE_SHA256: &str = "60a5a22b832d30d910ffade0947930e830ada71350d8ee6dc278182d0089d976";

fn tensor(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Tensor {
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}

fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
    tensor(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn json_dtype(value: &serde_json::Value) -> Dtype {
    match value.as_str().unwrap() {
        "float32" => Dtype::F32,
        "float16" => Dtype::F16,
        "bfloat16" => Dtype::Bf16,
        other => panic!("unsupported RMS fixture dtype: {other}"),
    }
}

fn json_shape(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|dim| u32::try_from(dim.as_u64().unwrap()).unwrap())
        .collect()
}

fn decode_hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).unwrap();
            let low = (pair[1] as char).to_digit(16).unwrap();
            ((high << 4) | low) as u8
        })
        .collect()
}

fn import_fixture_tensor(
    kernels: &mut MlxKernels,
    value: &serde_json::Value,
) -> inference_engine::tensor::ExecutionTensor {
    let source = tensor(
        json_dtype(&value["dtype"]),
        json_shape(&value["shape"]),
        decode_hex(value["bytes"].as_str().unwrap()),
    );
    kernels.import_tensor(&source).unwrap()
}

#[test]
fn rms_norm_preserves_shape_and_uses_the_supplied_weight() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_tensor(vec![2, 2], &[3.0, 4.0, 1.0, 2.0]))
        .unwrap();
    let weight = kernels
        .import_tensor(&f32_tensor(vec![2], &[2.0, 2.0]))
        .unwrap();

    let output = kernels.rms_norm(&x, &weight, 3.5).unwrap();

    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output.dtype(), Dtype::F32);
    let actual = kernels.readback(&output).unwrap();
    let values = actual
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    let expected = [1.5f32, 2.0, 2.0 / 6.0f32.sqrt(), 4.0 / 6.0f32.sqrt()];
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() <= 5e-7 + 5e-7 * expected.abs());
    }
}

#[test]
fn rms_norm_matches_frozen_pinned_mlx_outputs_bit_exactly() {
    let _serial = serial();
    let Some(path) = env::var_os("RMS_NORM_FIXTURES") else {
        eprintln!(
            "SKIPPED: rms_norm_matches_frozen_pinned_mlx_outputs_bit_exactly: RMS_NORM_FIXTURES is not set; frozen RMS normalization cells were not compared"
        );
        return;
    };
    let bytes = fs::read(path).expect("the RMS normalization fixture file must be readable");
    let actual_hash = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(actual_hash, RMS_FIXTURE_SHA256, "RMS fixture hash mismatch");
    let fixture: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fixture["schema"], "resident-rms-norm-1");
    let cells = fixture["cells"].as_array().expect("fixture cells");
    // The pinned fixture holds two cells. The count comes from the frozen file, not from this code.
    assert_eq!(cells.len(), 2, "the pinned fixture must contain two cells");
    let mut kernels = MlxKernels::try_new().unwrap();

    for cell in cells {
        let x = import_fixture_tensor(&mut kernels, &cell["x"]);
        let weight = import_fixture_tensor(&mut kernels, &cell["weight"]);
        let output = kernels
            .rms_norm(&x, &weight, cell["eps"].as_f64().unwrap() as f32)
            .unwrap();
        let expected = &cell["output"];
        assert_eq!(output.shape(), json_shape(&expected["shape"]));
        assert_eq!(output.dtype(), json_dtype(&expected["dtype"]));
        let actual = kernels.readback(&output).unwrap();
        assert_eq!(
            actual.host_bytes().unwrap(),
            decode_hex(expected["bytes"].as_str().unwrap())
        );
    }
    eprintln!(
        "compared {} frozen RMS normalization cells bit-exactly",
        cells.len()
    );
}

#[test]
fn rms_norm_rejects_either_integer_operand_without_poisoning_the_adapter() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let float_x = kernels
        .import_tensor(&f32_tensor(vec![2, 3], &[1.0; 6]))
        .unwrap();
    let float_weight = kernels
        .import_tensor(&f32_tensor(vec![3], &[1.0; 3]))
        .unwrap();

    for (dtype, word) in [
        (Dtype::I16, 1i16.to_le_bytes().to_vec()),
        (Dtype::I32, 1i32.to_le_bytes().to_vec()),
    ] {
        let integer_x = kernels
            .import_tensor(&tensor(dtype, vec![2, 3], word.repeat(6)))
            .unwrap();
        let integer_weight = kernels
            .import_tensor(&tensor(dtype, vec![3], word.repeat(3)))
            .unwrap();
        for (x, weight) in [(&float_x, &integer_weight), (&integer_x, &float_weight)] {
            let rejected = kernels.rms_norm(x, weight, 1e-6);
            assert!(
                matches!(&rejected, Err(EngError::BadRequest(message)) if message.contains("floating")),
                "{dtype:?} operand must be refused before dispatch, got {:?}",
                rejected.as_ref().err()
            );
            let output = kernels.rms_norm(&float_x, &float_weight, 1e-6).unwrap();
            assert_eq!(kernels.readback(&output).unwrap().shape(), [2, 3]);
        }
    }
}

#[test]
fn rms_norm_rejects_invalid_inputs_before_dispatch() {
    let _serial = serial();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_tensor(vec![2, 3], &[1.0; 6]))
        .unwrap();
    let wrong_width = kernels
        .import_tensor(&f32_tensor(vec![2], &[1.0; 2]))
        .unwrap();
    let non_vector = kernels
        .import_tensor(&f32_tensor(vec![1, 3], &[1.0; 3]))
        .unwrap();
    let integer_x = kernels
        .import_tensor(&tensor(
            Dtype::U32,
            vec![2, 3],
            1u32.to_le_bytes().repeat(6),
        ))
        .unwrap();
    let integer_weight = kernels
        .import_tensor(&tensor(Dtype::U32, vec![3], 1u32.to_le_bytes().repeat(3)))
        .unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let foreign_weight = other
        .import_tensor(&f32_tensor(vec![3], &[1.0; 3]))
        .unwrap();
    let scalar = kernels
        .import_tensor(&tensor(Dtype::F32, vec![], 1.0f32.to_le_bytes().to_vec()))
        .unwrap();
    let valid_weight = kernels
        .import_tensor(&f32_tensor(vec![3], &[1.0; 3]))
        .unwrap();

    for result in [
        kernels.rms_norm(&x, &wrong_width, 1e-6),
        kernels.rms_norm(&x, &non_vector, 1e-6),
        kernels.rms_norm(&integer_x, &integer_weight, 1e-6),
        kernels.rms_norm(&x, &foreign_weight, 1e-6),
        kernels.rms_norm(&scalar, &valid_weight, 1e-6),
    ] {
        assert!(matches!(result, Err(EngError::BadRequest(_))));
    }
}
