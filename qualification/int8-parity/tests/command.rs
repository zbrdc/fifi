//! Portable command tests with a tiny synthetic fixture (case 3 of decision 0031).

use std::{path::Path, process::Command};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const BIN: &str = env!("CARGO_BIN_EXE_int8-parity-qualification");
const SHARD: &str = "int8-fixture-00001-of-00001.safetensors";

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn safetensors(tensors: &[(String, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut sorted: Vec<_> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let (mut header, mut data, mut offset) = (serde_json::Map::new(), Vec::new(), 0);
    for (name, dtype, shape, bytes) in sorted {
        header.insert(
            name.clone(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[offset, offset + bytes.len()]}),
        );
        offset += bytes.len();
        data.extend_from_slice(bytes);
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    out
}

fn inventory(tensors: &[(String, &str, Vec<u64>, Vec<u8>)]) -> Vec<Value> {
    let mut v: Vec<_> = tensors.iter().collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v.iter()
        .map(|(n, d, s, b)| json!({"name":n,"dtype":d,"shape":s,"sha256":hex(b)}))
        .collect()
}

fn asset(dir: &Path, name: &str, bytes: &[u8]) -> Value {
    std::fs::write(dir.join(name), bytes).unwrap();
    json!({"path":name,"bytes":bytes.len(),"sha256":hex(bytes)})
}

/// Builds a two-layer synthetic fixture. `mutate` edits the manifest before it is written.
fn build(dir: &Path, mutate: impl FnOnce(&mut Value)) -> String {
    let kinds = [("mlp.gate", 256u64), ("mlp.shared_expert_gate", 1)];
    let mut shard = Vec::new();
    let mut overrides = serde_json::Map::new();
    let mut modules = Vec::new();
    for layer in 0..2 {
        for (kind, rows) in kinds {
            let module = format!("language_model.model.layers.{layer}.{kind}");
            shard.push((
                format!("{module}.weight"),
                "U32",
                vec![rows, 512],
                vec![7u8; rows as usize * 2048],
            ));
            for role in ["scales", "biases"] {
                shard.push((
                    format!("{module}.{role}"),
                    "BF16",
                    vec![rows, 32],
                    [0x80u8, 0x3f].repeat(rows as usize * 32),
                ));
            }
            overrides.insert(module.clone(), json!({"bits":8,"group_size":64}));
            modules.push((module, rows));
        }
    }
    let mut config = json!({"quantization":{"mode":"affine","bits":4,"group_size":64}});
    config["quantization"]
        .as_object_mut()
        .unwrap()
        .extend(overrides);
    let weight_map: serde_json::Map<String, Value> =
        shard.iter().map(|t| (t.0.clone(), json!(SHARD))).collect();
    let index = json!({"metadata":{}, "weight_map": weight_map});
    let mut plan: Vec<(String, u32, bool)> =
        modules.iter().map(|(m, _)| (m.clone(), 1, false)).collect();
    for m in [5, 16] {
        for layer in [0, 1] {
            for kind in ["mlp.gate", "mlp.shared_expert_gate"] {
                plan.push((
                    format!("language_model.model.layers.{layer}.{kind}"),
                    m,
                    false,
                ));
            }
        }
    }
    plan.push(("language_model.model.layers.0.mlp.gate".into(), 1, true));
    plan.push((
        "language_model.model.layers.0.mlp.shared_expert_gate".into(),
        1,
        true,
    ));
    let (mut cases_tensors, mut records) = (Vec::new(), Vec::new());
    for (i, (module, m, zero)) in plan.iter().enumerate() {
        let rows = if module.ends_with("mlp.gate") { 256 } else { 1 };
        let mut act = vec![0u8; *m as usize * 2048 * 4];
        if *zero {
            for j in 0..512 {
                act[(4 * j + 1) * 4 + 3] = 0x80;
            }
        }
        let id = format!("c{i:03}");
        cases_tensors.push((format!("act.{id}"), "F32", vec![*m as u64, 2048], act));
        cases_tensors.push((
            format!("ref.{id}"),
            "F32",
            vec![*m as u64, rows],
            vec![0u8; *m as usize * rows as usize * 4],
        ));
        let kernel = match (module.ends_with("mlp.gate"), *m) {
            (true, 1) => "affine_qmv_fast",
            (false, 1) => "affine_qmv",
            (_, 5) => "affine_qmv_wide",
            _ => "affine_qmm_splitk",
        };
        records.push(json!({"id":id,"module":module,"m":m,"seed":31000 + i,"signed_zero":zero,"expected_kernel":kernel,
            "bits":8,"group_size":64,"mode":"affine","activation":format!("act.{id}"),"reference":format!("ref.{id}"),"float64_max_abs_diff":0.0}));
    }
    let shard_bytes = safetensors(&shard);
    let cases_bytes = safetensors(&cases_tensors);
    let mut manifest = json!({
        "schema":"int8-reference/1","synthetic":true,"layer_count":2,"reference_pin":"0".repeat(64),
        "versions":{},"recorded_binaries":{},"build_flags":{},"env":{},"device":{"cpu_brand":"synthetic"},
        "source_files":[],"sources":[],
        "assets":{
            "config":asset(dir,"config.json",&serde_json::to_vec(&config).unwrap()),
            "index":asset(dir,"model.safetensors.index.json",&serde_json::to_vec(&index).unwrap()),
            "shard":asset(dir,SHARD,&shard_bytes),
            "cases":asset(dir,"cases.safetensors",&cases_bytes)},
        "shard_tensors":inventory(&shard),"case_tensors":inventory(&cases_tensors),
        "cases":records,"float64_worst_relative_diff":0.0});
    mutate(&mut manifest);
    let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    std::fs::write(dir.join("manifest.json"), &bytes).unwrap();
    hex(&bytes)
}

fn run(dir: &Path, digest: &str, extra: &[&str]) -> (i32, String, String) {
    let out = Command::new(BIN)
        .args([
            "--cases",
            dir.to_str().unwrap(),
            "--manifest-sha256",
            digest,
        ])
        .args(extra)
        .output()
        .unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8_lossy(&out.stdout).into(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn requires_manifest_digest() {
    let out = Command::new(BIN)
        .args(["--cases", "x", "--output", "y"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("manifest-sha256"));
}

#[test]
fn synthetic_fixture_passes_integrity() {
    let dir = tempfile::tempdir().unwrap();
    let digest = build(dir.path(), |_| {});
    let (code, stdout, stderr) = run(dir.path(), &digest, &["--integrity-only"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("integrity_ok") && stdout.contains("\"cases\":14"));
}

#[test]
fn changed_manifest_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let digest = build(dir.path(), |_| {});
    let path = dir.path().join("manifest.json");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&path, bytes).unwrap();
    let (code, _, stderr) = run(dir.path(), &digest, &["--integrity-only"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("manifest digest mismatch"));
}

#[test]
fn tampered_shard_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let digest = build(dir.path(), |_| {});
    let path = dir.path().join(SHARD);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    let (code, _, stderr) = run(dir.path(), &digest, &["--integrity-only"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("digest mismatch"), "{stderr}");
}

#[test]
fn wrong_seed_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let digest = build(dir.path(), |m| m["cases"][3]["seed"] = json!(1));
    let (code, _, stderr) = run(dir.path(), &digest, &["--integrity-only"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("differs from decision 0031"), "{stderr}");
}

#[test]
fn missing_signed_zero_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let digest = build(dir.path(), |m| m["cases"][12]["signed_zero"] = json!(false));
    let (code, _, _) = run(dir.path(), &digest, &["--integrity-only"]);
    assert_eq!(code, 2);
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn native_run_reports_error_off_mac() {
    let dir = tempfile::tempdir().unwrap();
    let digest = build(dir.path(), |_| {});
    let report = dir.path().join("report.json");
    let (code, _, _) = run(dir.path(), &digest, &["--output", report.to_str().unwrap()]);
    assert_eq!(code, 2);
    let value: Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    assert_eq!(value["status"], "error");
    assert_eq!(value["qualified"], false);
}
