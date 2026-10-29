//! Compares the frozen real-group fixture through the `MlxKernels` caller.

use std::path::{Path, PathBuf};

use inference_engine::family::quantization::{ModuleQuantization, QuantizationMap};

const SCHEMA: &str = "p0a05-real-group-1";
const MANIFEST_SCHEMA: &str = "p0a05-real-group-manifest-1";
const GROUP_SIZE: usize = 64;
const WORDS_PER_GROUP: usize = 8;

fn fixture_path() -> PathBuf {
    std::env::var_os("REAL_GROUP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../qualification/real-groups/real-groups.json")
        })
}

fn load_fixture(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!(
            "missing real-group fixture {} ({error}); run qualification/real-groups/prepare-real-groups.py first",
            path.display()
        )
    });
    let fixture: serde_json::Value = serde_json::from_slice(&bytes).expect("fixture JSON parses");
    assert_eq!(fixture["schema"], SCHEMA);
    fixture
}

fn load_manifest(fixture_path: &Path) -> serde_json::Value {
    let path = fixture_path.with_file_name("manifest.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!("missing real-group manifest {} ({error})", path.display());
    });
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).expect("manifest JSON parses");
    assert_eq!(manifest["schema"], MANIFEST_SCHEMA);
    manifest
}

fn zero_tolerance(fixture: &serde_json::Value) {
    let entry = &fixture["tolerances"]["real-group"];
    assert!(!entry.is_null(), "missing real-group tolerance");
    assert_eq!(entry["abs"].as_f64().unwrap(), 0.0, "absolute tolerance");
    assert_eq!(entry["rel"].as_f64().unwrap(), 0.0, "relative tolerance");
}

struct Comparison {
    mismatches: Vec<usize>,
    signed_zeros: usize,
}

fn parsed_f32_bits(parsed: f64, what: &str) -> u32 {
    let narrowed = parsed as f32;
    let drift = (f64::from(narrowed) - parsed).abs();
    assert!(
        drift <= f64::from(narrowed).abs() * 1e-6 + f64::MIN_POSITIVE,
        "{} value {} is not near f32",
        what,
        parsed
    );
    narrowed.to_bits()
}

fn checked_sidecar_bits(side: &serde_json::Value, name: &str) -> u16 {
    let bits = side["bf16_bits"].as_u64().expect("bf16 bits");
    assert!(bits <= u64::from(u16::MAX), "{} bits out of range", name);
    let widened = bf16_bits_to_f32(bits as u16);
    let stored = parsed_f32_bits(side["f32"].as_f64().expect("sidecar f32"), "sidecar");
    assert_eq!(
        widened.to_bits(),
        stored,
        "{} does not match its stored bits",
        name
    );
    assert!(side["f32"].as_f64().expect("f32").is_finite(), "{}", name);
    bits as u16
}

fn compare_exact(actual: &[f32], expected: &[f64]) -> Comparison {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    let mut mismatches = Vec::new();
    let mut signed_zeros = 0_usize;
    for (index, pair) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            pair.1.is_finite(),
            "expected value {} is not finite",
            pair.1
        );
        if parsed_f32_bits(*pair.1, "expected") == pair.0.to_bits() {
            continue;
        }
        if (*pair.1 as f32) == *pair.0 {
            signed_zeros += 1;
            continue;
        }
        mismatches.push(index);
    }
    Comparison {
        mismatches,
        signed_zeros,
    }
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn shape_product(item: &serde_json::Value) -> usize {
    item["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|dim| dim.as_u64().expect("dim") as usize)
        .product()
}

fn check_sidecar(cell: &serde_json::Value) {
    for name in ["scales", "biases"] {
        checked_sidecar_bits(&cell[name], name);
    }
}

fn check_coordinates(cell: &serde_json::Value) {
    let shape = cell["payload_shape"]
        .as_array()
        .expect("payload shape")
        .iter()
        .map(|dim| dim.as_u64().expect("dim") as usize)
        .collect::<Vec<usize>>();
    let rank = cell["rank"].as_u64().expect("rank") as usize;
    assert_eq!(shape.len(), rank, "{} rank", cell["id"]);
    assert!(
        rank == 2 || rank == 3,
        "{} rank must be two or three",
        cell["id"]
    );
    assert!(
        shape[rank - 1] >= WORDS_PER_GROUP && shape[rank - 1] % WORDS_PER_GROUP == 0,
        "{} width must be a positive multiple of {} words",
        cell["id"],
        WORDS_PER_GROUP
    );
    let per_row = cell["groups_per_row"].as_u64().expect("groups per row") as usize;
    assert_eq!(
        per_row,
        shape[rank - 1] / WORDS_PER_GROUP,
        "{} groups",
        cell["id"]
    );
    let coordinates = &cell["coordinates"];
    let group = coordinates["group"].as_u64().expect("group") as usize;
    assert!(group < per_row, "{} group out of range", cell["id"]);
    let row = coordinates["row"].as_u64().expect("row") as usize;
    assert!(row < shape[rank - 2], "{} row out of range", cell["id"]);
    if rank == 3 {
        let expert = coordinates["expert"].as_u64().expect("expert") as usize;
        assert!(expert < shape[0], "{} expert out of range", cell["id"]);
    } else {
        assert!(
            coordinates["expert"].is_null(),
            "{} expert on rank two",
            cell["id"]
        );
    }
}

fn check_counts(fixture: &serde_json::Value, cells: &[serde_json::Value]) {
    let counts = &fixture["counts"];
    let groups = counts["groups"].as_u64().expect("groups") as usize;
    assert_eq!(groups, cells.len(), "cell count");
    assert_eq!(
        counts["matmul_calls"].as_u64().expect("matmul calls") as usize,
        2 * cells.len(),
        "matmul call count"
    );
    assert_eq!(
        counts["compared_values"].as_u64().expect("compared") as usize,
        (GROUP_SIZE + 1) * cells.len(),
        "compared value count"
    );
    let negative = cells
        .iter()
        .filter(|cell| cell["negative_bias"].as_bool().expect("negative bias"))
        .count();
    assert_eq!(
        counts["negative_bias_groups"].as_u64().expect("negative") as usize,
        negative,
        "negative bias count"
    );
    assert!(negative > 0, "the fixture must cover negative biases");
}

fn check_cell(cell: &serde_json::Value) {
    let id = cell["id"].as_str().expect("cell id");
    assert_eq!(
        cell["payload_tensor"].as_str().expect("tensor"),
        format!("{}.weight", cell["module"].as_str().expect("module")),
        "{id} tensor"
    );
    assert_eq!(
        cell["dtypes"],
        serde_json::json!({"payload": "U32", "scales": "BF16", "biases": "BF16"}),
        "{id} dtypes"
    );
    let words = cell["weight"]["words"].as_array().expect("words");
    assert_eq!(words.len(), WORDS_PER_GROUP, "{id} word count");
    for word in words {
        assert!(word.as_u64().expect("word") <= u64::from(u32::MAX), "{id}");
    }
    check_sidecar(cell);
    check_coordinates(cell);
    assert_eq!(
        cell["negative_bias"].as_bool().expect("negative bias"),
        cell["biases"]["f32"].as_f64().expect("bias") < 0.0,
        "{id} negative bias flag"
    );
    for name in ["basis", "dense"] {
        let expected = &cell["expected"][name];
        let values = expected["values"].as_array().expect("expected values");
        assert_eq!(shape_product(expected), values.len(), "{id} {name} shape");
        for value in values {
            assert!(value.as_f64().expect("value").is_finite(), "{id} {name}");
        }
    }
    assert!(
        cell["reference_op"]
            .as_str()
            .expect("reference op")
            .contains("group_size=64, bits=4"),
        "{id} reference op"
    );
}

#[test]
fn committed_fixture_declares_every_group_and_tolerance() {
    let fixture = load_fixture(&fixture_path());
    zero_tolerance(&fixture);
    assert_eq!(
        fixture["quantization"]["default"],
        serde_json::json!({"mode": "affine", "bits": 4, "group_size": 64}),
        "default spec"
    );
    let cells = fixture["cells"].as_array().expect("cells array");
    assert_eq!(cells.len(), 1000, "the gate samples 1,000 groups");
    for cell in cells {
        check_cell(cell);
    }
    check_counts(&fixture, cells);
    let manifest = load_manifest(&fixture_path());
    assert_eq!(manifest["counts"], fixture["counts"], "manifest counts");
    let recorded = manifest["fixture"]["bytes"]
        .as_u64()
        .expect("fixture bytes");
    let observed = std::fs::metadata(fixture_path())
        .expect("fixture metadata")
        .len();
    assert_eq!(
        recorded, observed,
        "fixture length differs from the manifest"
    );
}

#[test]
fn quantization_metadata_resolves_the_sampled_modules() {
    let fixture = load_fixture(&fixture_path());
    let manifest = load_manifest(&fixture_path());
    let config = serde_json::to_vec(&fixture["quantization"]["config"]).expect("encode config");
    let wrapped = serde_json::json!({"quantization": serde_json::from_slice::<serde_json::Value>(&config).unwrap()});
    let map = QuantizationMap::from_json(&serde_json::to_vec(&wrapped).unwrap())
        .expect("quantization map");
    let default = ModuleQuantization::Quantized(inference_engine::kernels::QuantizationSpec {
        mode: inference_engine::kernels::QuantizationMode::Affine,
        bits: 4,
        group_size: GROUP_SIZE as u32,
    });
    for cell in fixture["cells"].as_array().expect("cells") {
        let module = cell["module"].as_str().expect("module");
        assert_eq!(map.resolve(module).expect("resolve"), default, "{module}");
    }
    for module in manifest["excluded_int8"].as_array().expect("excluded") {
        let name = module.as_str().expect("module name");
        let ModuleQuantization::Quantized(spec) = map.resolve(name).expect("resolve") else {
            panic!("{name} must resolve to a quantized spec");
        };
        assert_eq!(spec.bits, 8, "{name} bits");
    }
}

#[test]
fn signed_zero_comparison_counts_bits_without_failing() {
    let clean = compare_exact(&[1.0, 0.0], &[1.0, 0.0]);
    assert!(clean.mismatches.is_empty() && clean.signed_zeros == 0);
    let signed = compare_exact(&[0.0], &[-0.0]);
    assert!(signed.mismatches.is_empty() && signed.signed_zeros == 1);
    assert_eq!(compare_exact(&[1.0], &[2.0]).mismatches, vec![0_usize]);
}

#[test]
#[should_panic(expected = "output length mismatch")]
fn comparison_rejects_different_lengths() {
    compare_exact(&[1.0], &[1.0, 2.0]);
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

    fn exact_f32(value: f64) -> Vec<u8> {
        let narrowed = value as f32;
        assert_eq!(
            f64::from(narrowed),
            value,
            "fixture value {value} is not exact"
        );
        narrowed.to_le_bytes().to_vec()
    }

    fn activation_tensor(item: &Value) -> Tensor {
        assert_eq!(item["dtype"].as_str().expect("dtype"), "f32");
        let values = item["values"].as_array().expect("values");
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| exact_f32(value.as_f64().expect("finite number")))
            .collect();
        let shape = item["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim") as u32)
            .collect();
        Tensor::try_from_host(Dtype::F32, shape, bytes).expect("activation tensor")
    }

    fn sidecar_tensor(side: &Value) -> Tensor {
        let bits = checked_sidecar_bits(side, "sidecar");
        Tensor::try_from_host(Dtype::Bf16, vec![1, 1], bits.to_le_bytes().to_vec())
            .expect("sidecar tensor")
    }

    fn packed_row(cell: &Value) -> PackedWeight {
        let words = cell["weight"]["words"].as_array().expect("words");
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|word| (word.as_u64().expect("word") as u32).to_le_bytes())
            .collect();
        let payload = Tensor::try_from_host(Dtype::U32, vec![1, WORDS_PER_GROUP as u32], bytes)
            .expect("payload");
        let scales = sidecar_tensor(&cell["scales"]);
        let biases = sidecar_tensor(&cell["biases"]);
        PackedWeight::try_new(
            payload,
            scales,
            Some(biases),
            vec![1, GROUP_SIZE as u32],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: GROUP_SIZE as u32,
            },
        )
        .expect("packed row")
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

    fn expected_of(cell: &Value, stream: &str) -> Vec<f64> {
        cell["expected"][stream]["values"]
            .as_array()
            .expect("expected values")
            .iter()
            .map(|value| value.as_f64().expect("finite number"))
            .collect()
    }

    fn diff_details(
        stream: &str,
        actual: &[f32],
        expected: &[f64],
        done: &Comparison,
    ) -> Vec<Value> {
        done.mismatches
            .iter()
            .take(32)
            .map(|index| {
                json!({
                    "stream": stream,
                    "index": index,
                    "expected": expected[*index],
                    "actual": actual[*index],
                    "expected_bits": (expected[*index] as f32).to_bits(),
                    "actual_bits": actual[*index].to_bits(),
                })
            })
            .collect()
    }

    fn write_report_if_requested(fixture: &Value, manifest: &Value, cells: &[Value]) {
        let Ok(path) = std::env::var("REAL_GROUP_REPORT") else {
            return;
        };
        let report = json!({
            "schema": "p0a05-real-group-report-1",
            "provenance": fixture["provenance"],
            "counts": fixture["counts"],
            "fixture": manifest["fixture"],
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
    fn every_real_group_matches_the_pinned_reference() {
        let _serial = GPU_TEST.lock().unwrap();
        let fixture = load_fixture(&fixture_path());
        zero_tolerance(&fixture);
        let manifest = load_manifest(&fixture_path());
        let basis = activation_tensor(&fixture["activations"]["basis"]);
        let dense = activation_tensor(&fixture["activations"]["dense"]);
        let mut kernels = MlxKernels::try_new().expect("native kernels");
        let mut failures = Vec::new();
        let mut report_cells = Vec::new();
        for cell in fixture["cells"].as_array().expect("cells") {
            let id = cell["id"].as_str().expect("cell id");
            let weight = packed_row(cell);
            let basis_out = kernels
                .dequant_matmul(&basis, &weight)
                .unwrap_or_else(|error| panic!("{id} basis: {error:?}"));
            let dense_out = kernels
                .dequant_matmul(&dense, &weight)
                .unwrap_or_else(|error| panic!("{id} dense: {error:?}"));
            let basis_done = compare_exact(&output_values(&basis_out), &expected_of(cell, "basis"));
            let dense_done = compare_exact(&output_values(&dense_out), &expected_of(cell, "dense"));
            if !basis_done.mismatches.is_empty() || !dense_done.mismatches.is_empty() {
                failures.push(id);
            }
            let mut differences = diff_details(
                "basis",
                &output_values(&basis_out),
                &expected_of(cell, "basis"),
                &basis_done,
            );
            differences.extend(diff_details(
                "dense",
                &output_values(&dense_out),
                &expected_of(cell, "dense"),
                &dense_done,
            ));
            report_cells.push(json!({
                "id": id,
                "module": cell["module"],
                "coordinates": cell["coordinates"],
                "negative_bias": cell["negative_bias"],
                "signed_zeros": basis_done.signed_zeros + dense_done.signed_zeros,
                "zero_diff": differences.is_empty(),
                "differences": differences,
            }));
        }
        write_report_if_requested(&fixture, &manifest, &report_cells);
        assert!(
            failures.is_empty(),
            "groups with f32 differences: {failures:?}"
        );
    }
}
