//! Compares every OQ04 parity matrix cell through the `MlxKernels` caller.

use std::path::{Path, PathBuf};

fn fixture_path() -> PathBuf {
    std::env::var_os("OQ04_PARITY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../qualification/oq04-parity/fixtures.json")
        })
}

fn load_fixture(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!(
            "missing parity fixture {} ({error}); run qualification/oq04-parity/prepare.py first",
            path.display()
        )
    });
    let fixture: serde_json::Value = serde_json::from_slice(&bytes).expect("fixture JSON parses");
    assert_eq!(fixture["schema"], "oq04-parity-matrix-1");
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
    for (index, pair) in actual.iter().zip(expected.iter()).enumerate() {
        let difference = (f64::from(*pair.0) - pair.1).abs();
        let relative = if *pair.1 == 0.0 {
            difference
        } else {
            difference / pair.1.abs()
        };
        let allowed = abs + rel * pair.1.abs();
        if difference > max_abs {
            max_abs = difference;
        }
        if relative > max_rel {
            max_rel = relative;
        }
        if difference > allowed {
            exceeding += 1;
            let _ = index;
        }
    }
    (max_abs, max_rel, exceeding)
}

fn expected_values(cell: &serde_json::Value) -> Vec<f64> {
    cell["expected"]["values"]
        .as_array()
        .expect("expected values")
        .iter()
        .map(|value| value.as_f64().expect("finite number"))
        .collect()
}

#[test]
fn committed_fixture_declares_every_cell_and_tolerance() {
    let fixture = load_fixture(&fixture_path());
    let cells = fixture["cells"].as_array().expect("cells array");
    assert!(cells.len() >= 25, "matrix too small: {}", cells.len());
    let kinds = [
        "dequant_matmul",
        "matmul",
        "gather_dequant_matmul",
        "gather_matmul",
        "embedding_rows",
        "quantized_embedding_rows",
    ];
    for cell in cells {
        let id = cell["id"].as_str().expect("cell id");
        assert!(
            kinds.contains(&cell["kind"].as_str().expect("kind")),
            "{id} kind"
        );
        tolerance(&fixture, cell["family"].as_str().expect("family"));
        assert!(
            !cell["reference_op"]
                .as_str()
                .expect("reference op")
                .is_empty(),
            "{id}"
        );
        let expected = expected_values(cell);
        let shape: usize = cell["expected"]["shape"]
            .as_array()
            .expect("expected shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as usize)
            .product();
        assert_eq!(shape, expected.len(), "{id} shape");
        for value in &expected {
            assert!(value.is_finite(), "{id} nonfinite expected value");
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

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mac {
    use super::*;
    use inference_engine::{
        kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
        tensor::{Dtype, Tensor},
    };
    use serde_json::{Value, json};
    use std::sync::Mutex;

    static GPU_TEST: Mutex<()> = Mutex::new(());

    fn shape_of(item: &Value) -> Vec<u32> {
        item["shape"]
            .as_array()
            .expect("input shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as u32)
            .collect()
    }

    fn values_of(item: &Value) -> Vec<f64> {
        item["values"]
            .as_array()
            .expect("input values")
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

    fn exact_f16(value: f64) -> Vec<u8> {
        let narrowed = value as f32;
        let stored = half::f16::from_f32(narrowed);
        assert_eq!(
            f64::from(stored.to_f32()),
            value,
            "fixture value {value} is not f16-representable"
        );
        stored.to_bits().to_le_bytes().to_vec()
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
        let dtype = match item["dtype"].as_str().expect("float dtype") {
            "f32" => Dtype::F32,
            "f16" => Dtype::F16,
            "bf16" => Dtype::Bf16,
            other => panic!("unexpected float dtype {other}"),
        };
        let values = values_of(item);
        let bytes: Vec<u8> = match dtype {
            Dtype::F32 => host_bytes(&values, exact_f32),
            Dtype::F16 => host_bytes(&values, exact_f16),
            Dtype::Bf16 => host_bytes(&values, exact_bf16),
            _ => unreachable!(),
        };
        Tensor::try_from_host(dtype, shape_of(item), bytes).expect("float tensor")
    }

    fn index_tensor(item: &Value) -> Tensor {
        let values = values_of(item);
        let bytes = host_bytes(&values, |value| {
            assert_eq!(value.fract(), 0.0, "index {value} is not an integer");
            assert!(
                (0.0..f64::from(u32::MAX)).contains(&value),
                "index {value} out of range"
            );
            (value as u32).to_le_bytes().to_vec()
        });
        let shape = vec![values.len() as u32];
        Tensor::try_from_host(Dtype::U32, shape, bytes).expect("index tensor")
    }

    fn scale_tensor(item: &Value) -> Tensor {
        if item["dtype"].as_str().expect("scale dtype") == "u8" {
            let values = values_of(item);
            let bytes = values
                .iter()
                .map(|value| {
                    assert_eq!(value.fract(), 0.0, "u8 scale {value} is not an integer");
                    *value as u8
                })
                .collect();
            return Tensor::try_from_host(Dtype::U8, shape_of(item), bytes).expect("scale tensor");
        }
        float_tensor(item)
    }

    fn packed_weight(item: &Value, quant: &Value) -> PackedWeight {
        let payload = &item["payload"];
        let words = payload["words"].as_array().expect("payload words");
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|word| {
                let value = word.as_u64().expect("u32 word");
                assert!(
                    value <= u64::from(u32::MAX),
                    "payload word {value} out of range"
                );
                (value as u32).to_le_bytes()
            })
            .collect();
        let payload = Tensor::try_from_host(Dtype::U32, shape_of(payload), bytes).expect("payload");
        let scales = scale_tensor(&item["scales"]);
        let biases = (!item["biases"].is_null()).then(|| float_tensor(&item["biases"]));
        let logical: Vec<u32> = item["logical_shape"]
            .as_array()
            .expect("logical shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as u32)
            .collect();
        let spec = QuantizationSpec {
            mode: match quant["mode"].as_str().expect("mode") {
                "affine" => QuantizationMode::Affine,
                "mxfp4" => QuantizationMode::Mxfp4,
                "mxfp8" => QuantizationMode::Mxfp8,
                other => panic!("unexpected mode {other}"),
            },
            bits: quant["bits"].as_u64().expect("bits") as u32,
            group_size: quant["group_size"].as_u64().expect("group") as u32,
        };
        PackedWeight::try_new(payload, scales, biases, logical, spec).expect("packed weight")
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

    fn run_cell(kernels: &mut MlxKernels, cell: &Value) -> Result<Vec<f32>, String> {
        let out = match cell["kind"].as_str().expect("kind") {
            "dequant_matmul" => {
                let activation = float_tensor(&cell["activation"]);
                let weight = packed_weight(&cell["weight"], &cell["quant"]);
                kernels.dequant_matmul(&activation, &weight)
            }
            "matmul" => {
                let activation = float_tensor(&cell["activation"]);
                let weight = float_tensor(&cell["weight"]);
                kernels.matmul(&activation, &weight)
            }
            "gather_dequant_matmul" => {
                let activation = float_tensor(&cell["activation"]);
                let weight = packed_weight(&cell["weight"], &cell["quant"]);
                let indices = index_tensor(&cell["indices"]);
                kernels.gather_dequant_matmul(&activation, &weight, &indices)
            }
            "gather_matmul" => {
                let activation = float_tensor(&cell["activation"]);
                let weight = float_tensor(&cell["weight"]);
                let indices = index_tensor(&cell["indices"]);
                kernels.gather_matmul(&activation, &weight, &indices)
            }
            "embedding_rows" => {
                let table = float_tensor(&cell["table"]);
                let indices = index_tensor(&cell["indices"]);
                kernels.embedding_rows(&table, &indices)
            }
            "quantized_embedding_rows" => {
                let weight = packed_weight(&cell["weight"], &cell["quant"]);
                let indices = index_tensor(&cell["indices"]);
                kernels.quantized_embedding_rows(&weight, &indices)
            }
            other => return Err(format!("unknown cell kind {other}")),
        }
        .map_err(|error| format!("{error:?}"))?;
        Ok(output_values(&out))
    }

    fn diff_records(
        cell: &Value,
        actual: &[f32],
        expected: &[f64],
        abs: f64,
        rel: f64,
    ) -> Vec<Value> {
        let mut records = Vec::new();
        for (index, pair) in actual.iter().zip(expected.iter()).enumerate() {
            let difference = (f64::from(*pair.0) - pair.1).abs();
            if difference > abs + rel * pair.1.abs() && records.len() < 32 {
                records.push(json!({"index": index, "expected": pair.1, "actual": pair.0, "abs_diff": difference}));
            }
        }
        let _ = cell;
        records
    }

    fn write_report_if_requested(fixture: &Value, cells: &[Value]) {
        let Ok(path) = std::env::var("OQ04_PARITY_REPORT") else {
            return;
        };
        let report = json!({
            "schema": "oq04-parity-report-1",
            "provenance": fixture["provenance"],
            "tolerances": fixture["tolerances"],
            "cells": cells,
        });
        let target = PathBuf::from(path);
        let temp = target.with_extension("json.tmp");
        std::fs::write(
            &temp,
            serde_json::to_vec_pretty(&report).expect("encode report"),
        )
        .expect("write report");
        std::fs::rename(&temp, &target).expect("publish report");
    }

    #[test]
    fn every_parity_cell_matches_the_pinned_reference() {
        let _serial = GPU_TEST.lock().unwrap();
        let fixture = load_fixture(&fixture_path());
        let cells = fixture["cells"].as_array().expect("cells array");
        let mut kernels = MlxKernels::try_new().expect("native kernels");
        let mut failures = Vec::new();
        let mut report_cells = Vec::new();
        for cell in cells {
            let id = cell["id"].as_str().expect("cell id");
            let actual =
                run_cell(&mut kernels, cell).unwrap_or_else(|error| panic!("{id}: {error}"));
            let expected = expected_values(cell);
            let (abs, rel) = tolerance(&fixture, cell["family"].as_str().expect("family"));
            let (max_abs, max_rel, exceeding) = compare(&actual, &expected, abs, rel);
            if exceeding > 0 {
                failures.push(id);
            }
            report_cells.push(json!({
                "id": id,
                "kind": cell["kind"],
                "family": cell["family"],
                "reference_op": cell["reference_op"],
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "exceeding": exceeding,
                "zero_diff": max_abs == 0.0,
                "differences": diff_records(cell, &actual, &expected, abs, rel),
            }));
        }
        write_report_if_requested(&fixture, &report_cells);
        assert!(
            failures.is_empty(),
            "cells outside declared tolerance: {failures:?}"
        );
    }
}
