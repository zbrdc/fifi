use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::{Command, Output},
};

fn shard_bytes(tensors: &[(&str, &str, Vec<u32>, &[u8])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            (*name).into(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Fixture {
    dir: tempfile::TempDir,
    index_sha256: String,
    manifest: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let shard_one = shard_bytes(&[
        ("a.weight", "F32", vec![2], &[0u8; 8]),
        ("a.bias", "U8", vec![1], &[7u8]),
    ]);
    let shard_two = shard_bytes(&[("b.weight", "F16", vec![2], &[0u8; 4])]);
    std::fs::write(
        dir.path().join("model-00001-of-00002.safetensors"),
        &shard_one,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("model-00002-of-00002.safetensors"),
        &shard_two,
    )
    .unwrap();
    let index = json!({"weight_map": {
        "a.weight": "model-00001-of-00002.safetensors",
        "a.bias": "model-00001-of-00002.safetensors",
        "b.weight": "model-00002-of-00002.safetensors"
    }});
    let index = serde_json::to_vec(&index).unwrap();
    std::fs::write(dir.path().join("model.safetensors.index.json"), &index).unwrap();
    let manifest: BTreeMap<&str, String> = BTreeMap::from([
        ("model-00001-of-00002.safetensors", sha256(&shard_one)),
        ("model-00002-of-00002.safetensors", sha256(&shard_two)),
    ]);
    let manifest_path = dir.path().join("shards.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&serde_json::Value::Object(
            manifest
                .into_iter()
                .map(|(k, v)| (k.to_owned(), Value::String(v)))
                .collect(),
        ))
        .unwrap(),
    )
    .unwrap();
    Fixture {
        index_sha256: sha256(&index),
        manifest: manifest_path,
        dir,
    }
}

fn run(fixture: &Fixture, budget: &str, index_hash: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_model-load"))
        .arg("--model-dir")
        .arg(fixture.dir.path())
        .args(["--index-sha256", index_hash])
        .arg("--shard-manifest")
        .arg(&fixture.manifest)
        .args(["--max-model-bytes", budget])
        .arg("--output")
        .arg(fixture.dir.path().join("report.json"))
        .arg("--inventory")
        .arg(fixture.dir.path().join("inventory.json"))
        .output()
        .unwrap()
}

fn report(fixture: &Fixture) -> Value {
    serde_json::from_slice(&std::fs::read(fixture.dir.path().join("report.json")).unwrap()).unwrap()
}

#[test]
fn loads_an_indexed_checkpoint_and_publishes_the_inventory() {
    let fixture = fixture();
    let result = run(&fixture, "4096", &fixture.index_sha256);
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&fixture);
    assert_eq!(report["status"], "passed");
    assert_eq!(report["index_sha256"], fixture.index_sha256);
    assert_eq!(report["shard_count"], 2);
    assert_eq!(report["tensor_count"], 3);
    assert_eq!(report["resident_bytes"], 13);
    assert_eq!(report["max_model_bytes"], 4096);
    assert_eq!(report["memory_budget_ok"], true);
    let inventory: Value =
        serde_json::from_slice(&std::fs::read(fixture.dir.path().join("inventory.json")).unwrap())
            .unwrap();
    let tensors = inventory["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 3);
    assert_eq!(tensors[0]["name"], "a.bias");
    assert_eq!(tensors[0]["dtype"], "U8");
    assert_eq!(tensors[0]["byte_len"], 1);
    assert_eq!(tensors[1]["name"], "a.weight");
    assert_eq!(tensors[1]["shape"], json!([2]));
    assert_eq!(tensors[2]["name"], "b.weight");
    assert_eq!(tensors[2]["dtype"], "F16");
    assert_eq!(inventory["total_byte_len"], 13);
    assert_eq!(
        report["inventory_sha256"],
        sha256(&std::fs::read(fixture.dir.path().join("inventory.json")).unwrap())
    );
}

#[test]
fn rejects_an_index_digest_mismatch() {
    let fixture = fixture();
    let wrong = "0".repeat(64);
    let result = run(&fixture, "4096", &wrong);
    assert_eq!(result.status.code(), Some(1));
    let report = report(&fixture);
    assert_eq!(report["status"], "failed");
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("index digest mismatch")
    );
}

#[test]
fn rejects_a_shard_digest_mismatch() {
    let fixture = fixture();
    let mut manifest: BTreeMap<String, Value> =
        serde_json::from_slice(&std::fs::read(&fixture.manifest).unwrap()).unwrap();
    manifest.insert(
        "model-00002-of-00002.safetensors".into(),
        "1".repeat(64).into(),
    );
    std::fs::write(
        &fixture.manifest,
        serde_json::to_vec(&Value::Object(manifest.into_iter().collect())).unwrap(),
    )
    .unwrap();
    let result = run(&fixture, "4096", &fixture.index_sha256);
    assert_eq!(result.status.code(), Some(1));
    assert!(
        report(&fixture)["error"]
            .as_str()
            .unwrap()
            .contains("shard digest mismatch")
    );
}

#[test]
fn rejects_a_model_over_the_memory_budget() {
    let fixture = fixture();
    let result = run(&fixture, "12", &fixture.index_sha256);
    assert_eq!(result.status.code(), Some(1));
    assert!(
        report(&fixture)["error"]
            .as_str()
            .unwrap()
            .contains("exceeds the memory budget")
    );
}

#[test]
fn rejects_an_escaping_shard_name_before_reading_it() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    std::fs::create_dir(&model).unwrap();
    let outside = shard_bytes(&[("a.weight", "F32", vec![2], &[0u8; 8])]);
    std::fs::write(root.path().join("outside.safetensors"), &outside).unwrap();
    let name = "../outside.safetensors";
    let index = serde_json::to_vec(&json!({"weight_map": {"a.weight": name}})).unwrap();
    std::fs::write(model.join("model.safetensors.index.json"), &index).unwrap();
    let manifest = root.path().join("shards.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&json!({ name: sha256(&outside) })).unwrap(),
    )
    .unwrap();
    let inventory = root.path().join("inventory.json");
    let result = Command::new(env!("CARGO_BIN_EXE_model-load"))
        .arg("--model-dir")
        .arg(&model)
        .args(["--index-sha256", &sha256(&index)])
        .arg("--shard-manifest")
        .arg(&manifest)
        .args(["--max-model-bytes", "1"])
        .arg("--output")
        .arg(root.path().join("report.json"))
        .arg("--inventory")
        .arg(&inventory)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("report.json")).unwrap()).unwrap();
    assert_eq!(
        report["error"].as_str(),
        Some("invalid shard basename \"../outside.safetensors\"")
    );
    assert!(!inventory.exists());
}
