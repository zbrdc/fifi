//! Compares the resident `qmm` and `mm` seams against the frozen layer-parity fixtures.

//! Portable checks load a tiny committed fixture copied verbatim from the
//! frozen bulk fixture. Native checks require the frozen external fixture
//! through `LAYER_PARITY_FIXTURES` and verify its pinned hash.

const PORTABLE_FIXTURE: &str = include_str!("fixtures/layer-parity-portable.json");

fn portable_fixture() -> serde_json::Value {
    let fixture: serde_json::Value =
        serde_json::from_str(PORTABLE_FIXTURE).expect("portable fixture JSON parses");
    assert_eq!(fixture["schema"], "layer-parity-1");
    fixture
}

fn tolerance(fixture: &serde_json::Value, family: &str) -> (f64, f64) {
    let entry = &fixture["tolerances"][family];
    assert!(!entry.is_null(), "unknown tolerance family {family}");
    (
        entry["abs"].as_f64().unwrap(),
        entry["rel"].as_f64().unwrap(),
    )
}

fn compare(actual: &[f32], expected: &[f64], abs: f64, rel: f64) -> (f64, f64, usize) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut exceeding = 0_usize;
    for (left, right) in actual.iter().zip(expected.iter()) {
        let difference = (f64::from(*left) - *right).abs();
        let relative = if *right == 0.0 {
            difference
        } else {
            difference / right.abs()
        };
        let allowed = abs + rel * right.abs();
        if difference > max_abs {
            max_abs = difference;
        }
        if relative > max_rel {
            max_rel = relative;
        }
        if difference > allowed {
            exceeding += 1;
        }
    }
    (max_abs, max_rel, exceeding)
}

#[test]
fn committed_fixture_declares_every_cell_and_tolerance() {
    let fixture = portable_fixture();
    let cells = fixture["cells"].as_array().expect("cells");
    let probes = fixture["probes"].as_array().expect("probes");
    for entry in cells.iter().chain(probes) {
        let id = entry["id"].as_str().expect("id");
        let family = entry["family"].as_str().expect("family");
        tolerance(&fixture, family);
        assert!(
            matches!(entry["kind"].as_str().expect("kind"), "qmm" | "mm"),
            "{id} kind"
        );
        assert!(
            !entry["reference_op"]
                .as_str()
                .expect("reference op")
                .is_empty(),
            "{id}"
        );
        let expected = entry["expected"]["values"].as_array().expect("expected");
        let shape: usize = entry["expected"]["shape"]
            .as_array()
            .expect("expected shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as usize)
            .product();
        assert_eq!(shape, expected.len(), "{id} shape");
        for value in expected {
            assert!(value.as_f64().expect("finite number").is_finite(), "{id}");
        }
    }
}

#[test]
fn tolerance_comparison_rejects_and_accepts_known_differences() {
    assert_eq!(compare(&[1.0], &[1.0], 0.0, 0.0), (0.0, 0.0, 0));
    assert_eq!(compare(&[1.0], &[1.0 + 1e-9], 0.0, 0.0).2, 1);
    assert_eq!(compare(&[1.0], &[1.000001], 2e-5, 2e-5).2, 0);
    assert_eq!(compare(&[1.0], &[1.5], 2e-5, 2e-5).2, 1);
    assert_eq!(compare(&[0.0], &[1e-12], 0.0, 0.0).2, 1);
}

#[test]
fn hand_probe_carries_the_pinned_expectation() {
    let fixture = portable_fixture();
    let probe = &fixture["probes"][0];
    assert_eq!(probe["id"].as_str().expect("probe id"), "qmm-uniform-hand");
    let expected = probe["expected"]["values"].as_array().expect("probe");
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].as_f64().expect("probe value"), 144.0);
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mac {
    use super::*;
    use inference_engine::{
        EngError,
        kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
        tensor::{Dtype, Tensor},
    };
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static GPU_TEST: Mutex<()> = Mutex::new(());

    const FROZEN_FIXTURE_SHA256: &str =
        "eec14e9b114d0fb08cfc0166f08a65fb706013efe5a5904cb5fcd1c21cb985e4";

    /// The frozen fixture, or `None` after a `SKIPPED` line when `LAYER_PARITY_FIXTURES` is unset.
    /// A skipped test is not gate evidence. A gate run sets the variable and shows no `SKIPPED` line.
    fn frozen_fixture(test: &str) -> Option<serde_json::Value> {
        let Some(path) = std::env::var_os("LAYER_PARITY_FIXTURES") else {
            eprintln!(
                "SKIPPED: {test}: LAYER_PARITY_FIXTURES is not set; the frozen layer-parity \
                 fixture (sha256 {FROZEN_FIXTURE_SHA256}) was not compared"
            );
            return None;
        };
        Some(load_frozen_fixture(&PathBuf::from(path)))
    }

    fn load_frozen_fixture(path: &Path) -> serde_json::Value {
        let bytes = std::fs::read(path).unwrap_or_else(|error| {
            panic!(
                "cannot read frozen layer parity fixture {} ({error}); \
                 regenerate it with the layer-parity qualification harness",
                path.display()
            )
        });
        let digest = Sha256::digest(&bytes);
        assert_eq!(
            format!("{digest:x}"),
            FROZEN_FIXTURE_SHA256,
            "frozen layer parity fixture hash mismatch for {}",
            path.display()
        );
        let fixture: serde_json::Value =
            serde_json::from_slice(&bytes).expect("fixture JSON parses");
        assert_eq!(fixture["schema"], "layer-parity-1");
        fixture
    }

    fn shape_of(item: &Value) -> Vec<u32> {
        item["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as u32)
            .collect()
    }

    fn values_of(item: &Value) -> Vec<f64> {
        item["values"]
            .as_array()
            .expect("values")
            .iter()
            .map(|value| value.as_f64().expect("finite number"))
            .collect()
    }

    fn host_bytes(values: &[f64], convert: fn(f64) -> Vec<u8>) -> Vec<u8> {
        values.iter().flat_map(|value| convert(*value)).collect()
    }

    fn exact_f32(value: f64) -> Vec<u8> {
        let narrowed = value as f32;
        assert_eq!(
            f64::from(narrowed),
            value,
            "fixture value {value} is not exact"
        );
        narrowed.to_le_bytes().to_vec()
    }

    fn exact_bf16(value: f64) -> Vec<u8> {
        let narrowed = value as f32;
        let stored = half::bf16::from_f32(narrowed);
        assert_eq!(
            f64::from(stored.to_f32()),
            value,
            "fixture value {value} is not bf16-representable"
        );
        stored.to_bits().to_le_bytes().to_vec()
    }

    fn float_tensor(item: &Value) -> Tensor {
        let (dtype, convert) = match item["dtype"].as_str().expect("float dtype") {
            "float32" => (Dtype::F32, exact_f32 as fn(f64) -> Vec<u8>),
            "bfloat16" => (Dtype::Bf16, exact_bf16 as fn(f64) -> Vec<u8>),
            other => panic!("unexpected float dtype {other}"),
        };
        let bytes = host_bytes(&values_of(item), convert);
        Tensor::try_from_host(dtype, shape_of(item), bytes).expect("float tensor")
    }

    fn quant_spec(quant: &Value) -> QuantizationSpec {
        assert_eq!(quant["mode"].as_str().expect("quant mode"), "affine");
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: quant["bits"].as_u64().expect("quant bits") as u32,
            group_size: quant["group_size"].as_u64().expect("quant group") as u32,
        }
    }

    fn packed_weight(weight: &Value, quant: &Value) -> PackedWeight {
        let payload = &weight["payload"];
        let words: Vec<u8> = payload["words"]
            .as_array()
            .expect("payload words")
            .iter()
            .flat_map(|word| {
                u32::try_from(word.as_u64().expect("word"))
                    .expect("word fits u32")
                    .to_le_bytes()
            })
            .collect();
        let payload = Tensor::try_from_host(Dtype::U32, shape_of(payload), words).expect("payload");
        let scales = float_tensor(&weight["scales"]);
        let biases = float_tensor(&weight["biases"]);
        let logical: Vec<u32> = weight["logical"]
            .as_array()
            .expect("logical")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as u32)
            .collect();
        PackedWeight::try_new(payload, scales, Some(biases), logical, quant_spec(quant))
            .expect("packed weight")
    }

    fn output_values(out: &Tensor) -> Vec<f32> {
        assert_eq!(out.dtype(), Dtype::F32, "caller output must be F32");
        let bytes = out.host_bytes().expect("output bytes");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    fn run_qmm(kernels: &mut MlxKernels, entry: &Value) -> Result<Vec<f32>, String> {
        let activation = float_tensor(&entry["activation"]);
        let weight = packed_weight(&entry["weight"], &entry["quant"]);
        let resident_activation = kernels
            .import_tensor(&activation)
            .map_err(|error| format!("import activation: {error:?}"))?;
        let resident_weight = kernels
            .import_packed(&weight)
            .map_err(|error| format!("import weight: {error:?}"))?;
        let out = kernels
            .qmm(&resident_activation, &resident_weight)
            .map_err(|error| format!("qmm: {error:?}"))?;
        let host = kernels
            .readback(&out)
            .map_err(|error| format!("readback: {error:?}"))?;
        Ok(output_values(&host))
    }

    fn run_mm(kernels: &mut MlxKernels, entry: &Value) -> Result<Vec<f32>, String> {
        let activation = float_tensor(&entry["activation"]);
        let weight = float_tensor(&entry["weight"]);
        let resident_activation = kernels
            .import_tensor(&activation)
            .map_err(|error| format!("import activation: {error:?}"))?;
        let resident_weight = kernels
            .import_tensor(&weight)
            .map_err(|error| format!("import weight: {error:?}"))?;
        let out = kernels
            .mm(&resident_activation, &resident_weight)
            .map_err(|error| format!("mm: {error:?}"))?;
        let host = kernels
            .readback(&out)
            .map_err(|error| format!("readback: {error:?}"))?;
        Ok(output_values(&host))
    }

    fn write_report_if_requested(cells: &[Value]) {
        let Ok(path) = std::env::var("LAYER_PARITY_REPORT") else {
            return;
        };
        let mut rows: Vec<Value> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .filter(|report| report["schema"] == "layer-parity-report-1")
            .and_then(|report| report["cells"].as_array().cloned())
            .unwrap_or_default();
        for cell in cells {
            let id = cell["id"].as_str().expect("report id");
            rows.retain(|row| row["id"].as_str() != Some(id));
            rows.push(cell.clone());
        }
        let report = json!({
            "schema": "layer-parity-report-1",
            "cells": rows,
        });
        let text = serde_json::to_string_pretty(&report).expect("report serializes");
        std::fs::write(&path, text).expect("report written");
    }

    #[test]
    fn qmm_matches_frozen_cells_bit_exact() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(fixture) = frozen_fixture("qmm_matches_frozen_cells_bit_exact") else {
            return;
        };
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut report = Vec::new();
        let cells = fixture["cells"].as_array().expect("cells");
        let probes = fixture["probes"].as_array().expect("probes");
        for entry in cells
            .iter()
            .chain(probes)
            .filter(|entry| entry["kind"] == "qmm")
        {
            let id = entry["id"].as_str().expect("id").to_string();
            let family = entry["family"].as_str().expect("family");
            let (abs, rel) = tolerance(&fixture, family);
            let actual =
                run_qmm(&mut kernels, entry).unwrap_or_else(|error| panic!("{id}: {error}"));
            let expected = values_of(&entry["expected"]);
            let (max_abs, max_rel, exceeding) = compare(&actual, &expected, abs, rel);
            assert_eq!(
                exceeding, 0,
                "{id} exceeded tolerance (max_abs {max_abs}, max_rel {max_rel})"
            );
            report.push(json!({
                "id": id,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "exceeding": exceeding,
            }));
        }
        // The pinned fixture holds two qmm cells and one qmm probe. The count comes from the frozen file.
        assert_eq!(report.len(), 3, "two qmm cells and one qmm probe");
        eprintln!(
            "compared {} frozen layer-parity qmm cells within tolerance",
            report.len()
        );
        write_report_if_requested(&report);
    }

    #[test]
    fn mm_matches_frozen_cells_bit_exact() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(fixture) = frozen_fixture("mm_matches_frozen_cells_bit_exact") else {
            return;
        };
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut report = Vec::new();
        let cells = fixture["cells"].as_array().expect("cells");
        for entry in cells.iter().filter(|entry| entry["kind"] == "mm") {
            let id = entry["id"].as_str().expect("id").to_string();
            let family = entry["family"].as_str().expect("family");
            let (abs, rel) = tolerance(&fixture, family);
            let actual =
                run_mm(&mut kernels, entry).unwrap_or_else(|error| panic!("{id}: {error}"));
            let expected = values_of(&entry["expected"]);
            let (max_abs, max_rel, exceeding) = compare(&actual, &expected, abs, rel);
            assert_eq!(
                exceeding, 0,
                "{id} exceeded tolerance (max_abs {max_abs}, max_rel {max_rel})"
            );
            report.push(json!({
                "id": id,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "exceeding": exceeding,
            }));
        }
        assert_eq!(report.len(), 3, "three f32 mm cells");
        eprintln!(
            "compared {} frozen layer-parity mm cells within tolerance",
            report.len()
        );
        write_report_if_requested(&report);
    }

    fn probe_weight(fixture: &serde_json::Value) -> PackedWeight {
        let probe = &fixture["probes"][0];
        packed_weight(&probe["weight"], &probe["quant"])
    }

    #[test]
    fn mm_rejects_bad_shape() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut kernels = MlxKernels::try_new().unwrap();
        let activation_bytes = host_bytes(&[1.0; 7], exact_f32);
        let activation =
            Tensor::try_from_host(Dtype::F32, vec![1, 7], activation_bytes).expect("activation");
        let weight_bytes = host_bytes(&[1.0; 32], exact_f32);
        let weight = Tensor::try_from_host(Dtype::F32, vec![8, 4], weight_bytes).expect("weight");
        let resident_activation = kernels.import_tensor(&activation).unwrap();
        let resident_weight = kernels.import_tensor(&weight).unwrap();

        let result = kernels.mm(&resident_activation, &resident_weight);

        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains("shape")),
            "expected BadRequest shape error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn mm_rejects_non_f32_operand() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut kernels = MlxKernels::try_new().unwrap();
        let bytes = host_bytes(&[1.0; 32], exact_bf16);
        let activation =
            Tensor::try_from_host(Dtype::Bf16, vec![1, 32], bytes).expect("activation");
        let weight_bytes = host_bytes(&[1.0; 64], exact_f32);
        let weight = Tensor::try_from_host(Dtype::F32, vec![32, 2], weight_bytes).expect("weight");
        let resident_activation = kernels.import_tensor(&activation).unwrap();
        let resident_weight = kernels.import_tensor(&weight).unwrap();

        let result = kernels.mm(&resident_activation, &resident_weight);

        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains("dtype")),
            "expected BadRequest dtype error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn mm_rejects_foreign_owner() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut other = MlxKernels::try_new().unwrap();
        let bytes = host_bytes(&[1.0; 8], exact_f32);
        let activation = Tensor::try_from_host(Dtype::F32, vec![1, 8], bytes).expect("activation");
        let weight_bytes = host_bytes(&[1.0; 16], exact_f32);
        let weight = Tensor::try_from_host(Dtype::F32, vec![8, 2], weight_bytes).expect("weight");
        let foreign = other.import_tensor(&activation).unwrap();
        let resident_weight = kernels.import_tensor(&weight).unwrap();

        let result = kernels.mm(&foreign, &resident_weight);

        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains("owner")),
            "expected BadRequest owner error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn qmm_rejects_bad_shape() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(fixture) = frozen_fixture("qmm_rejects_bad_shape") else {
            return;
        };
        let mut kernels = MlxKernels::try_new().unwrap();
        let bytes = host_bytes(&[1.0; 7], exact_f32);
        let activation = Tensor::try_from_host(Dtype::F32, vec![1, 7], bytes).expect("activation");
        let resident_activation = kernels.import_tensor(&activation).unwrap();
        let resident_weight = kernels.import_packed(&probe_weight(&fixture)).unwrap();

        let result = kernels.qmm(&resident_activation, &resident_weight);

        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains("shape")),
            "expected BadRequest shape error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn qmm_rejects_non_f32_activation() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(fixture) = frozen_fixture("qmm_rejects_non_f32_activation") else {
            return;
        };
        let mut kernels = MlxKernels::try_new().unwrap();
        let bytes = host_bytes(&[1.0; 64], exact_bf16);
        let activation =
            Tensor::try_from_host(Dtype::Bf16, vec![1, 64], bytes).expect("activation");
        let resident_activation = kernels.import_tensor(&activation).unwrap();
        let resident_weight = kernels.import_packed(&probe_weight(&fixture)).unwrap();

        let result = kernels.qmm(&resident_activation, &resident_weight);

        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains("dtype")),
            "expected BadRequest dtype error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn qmm_rejects_foreign_owner() {
        let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(fixture) = frozen_fixture("qmm_rejects_foreign_owner") else {
            return;
        };
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut other = MlxKernels::try_new().unwrap();
        let probe = &fixture["probes"][0];
        let activation = float_tensor(&probe["activation"]);
        let foreign = other.import_tensor(&activation).unwrap();
        let resident_weight = kernels.import_packed(&probe_weight(&fixture)).unwrap();

        let result = kernels.qmm(&foreign, &resident_weight);

        assert!(
            matches!(&result, Err(EngError::BadRequest(message)) if message.contains("owner")),
            "expected BadRequest owner error, got {:?}",
            result.as_ref().err()
        );
    }
}
