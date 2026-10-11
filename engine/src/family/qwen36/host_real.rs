//! Reference run of the host forward over the real checkpoint (Q36-HR). The test is ignored: it
//! needs the checkpoint on the Mac, uses the CPU only, and writes its evidence to `Q36_OUT_DIR`.
//!
//! ```text
//! Q36_CKPT_DIR=/Users/dan/edge0/models/edge0-35b Q36_OUT_DIR=<a path that does not exist yet> \
//!   cargo test --release --lib family::qwen36::host_real -- --ignored --nocapture
//! ```
//! Optional: `Q36_PROMPT` (default `The capital of France is`), `Q36_GREEDY_TOKENS` (default 32),
//! `Q36_THREADS` (default: all cores), `Q36_SOURCE_COMMIT` (default: `git rev-parse HEAD` of the checkout).
//!
//! Input identity: before any compute, the run hashes `config.json`, the index, `tokenizer.json`, and
//! every shard, and compares size and SHA-256 with the pins of `engine/testdata/mac-phase0a.json`
//! (included at compile time). A mismatch fails the run. `host-ref.json` records every file name, size,
//! and SHA-256 and the source commit.
//!
//! Limit of this evidence: the oracle, the unit tests, and a later resident decoder all follow one
//! reading of the mlx-lm sources. A shared misreading passes all of them. Examples are the plain norm
//! convention on the real values, the value-to-key head map, the per-head `[query, gate]` row order,
//! the rotary pair layout at rotary 64, and the packed-slice offsets shared with the Flash-Next path.
//! The external check is the one-time mlx-lm top-5 sanity check of decision 0046, which the user
//! approved as a sanity check only. A Q36 result does not satisfy Phase 0B or 1B.

use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use sha2::{Digest, Sha256};

use super::{
    host::{HostState, step},
    host_cpu::CpuProjector,
    host_load::HostWeights,
};
use crate::{
    family::spec::QWEN3_6_35B_A3B,
    tokenizer::Tokenizer,
    weights::{ShardIo, StreamedCheckpoint},
};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

/// The five largest logits as `(token, logit)`, the lowest token first on ties.
fn top5(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, *v))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(5);
    ranked
}

/// The pins of the checkpoint, from the environment record (`checkpoint.files`). The record is
/// included at compile time, so the pins have one source and no copy in code.
const ENVIRONMENT_RECORD: &str = include_str!("../../../testdata/mac-phase0a.json");

/// Name, size in bytes, and SHA-256 of one checkpoint file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FileId {
    name: String,
    bytes: u64,
    sha256: String,
}

/// The sorted shard names of an index.
fn shard_names(index: &[u8]) -> Vec<String> {
    let map: serde_json::Value = serde_json::from_slice(index).unwrap();
    let mut names: Vec<String> = map["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Hashes a file in 1 MiB reads and returns its identity.
fn hash_file(path: &Path, name: &str) -> FileId {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let (mut hasher, mut buffer, mut bytes) = (Sha256::new(), vec![0u8; 1 << 20], 0u64);
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        bytes += n as u64;
    }
    FileId {
        name: name.to_string(),
        bytes,
        sha256: format!("{:x}", hasher.finalize()),
    }
}

/// Compares the files found with the pins of `record`. The found set must be exactly `config.json`,
/// the index, `tokenizer.json`, and the base weight files of the record, each with the pinned size
/// and SHA-256.
fn check_pins(record: &str, found: &[FileId]) -> Result<(), String> {
    let record: serde_json::Value = serde_json::from_str(record).map_err(|e| e.to_string())?;
    let checkpoint = &record["checkpoint"];
    let pins = checkpoint["files"]
        .as_array()
        .ok_or("no checkpoint files")?;
    let mut required: Vec<String> = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ]
    .map(String::from)
    .into();
    for name in checkpoint["base_weight_files"]
        .as_array()
        .ok_or("no base weight files")?
    {
        required.push(name.as_str().ok_or("bad shard name")?.to_string());
    }
    for name in &required {
        if !found.iter().any(|f| &f.name == name) {
            return Err(format!("{name} is not among the files found"));
        }
    }
    for file in found {
        if !required.contains(&file.name) {
            return Err(format!("{} is not a pinned file", file.name));
        }
        let pin = pins
            .iter()
            .find(|p| p["path"].as_str() == Some(&file.name))
            .ok_or_else(|| format!("{} has no pin", file.name))?;
        if pin["bytes"].as_u64() != Some(file.bytes) {
            return Err(format!(
                "{}: size {} differs from the pin",
                file.name, file.bytes
            ));
        }
        if pin["sha256"].as_str() != Some(&file.sha256) {
            return Err(format!(
                "{}: SHA-256 {} differs from the pin",
                file.name, file.sha256
            ));
        }
    }
    Ok(())
}

/// The source commit: `Q36_SOURCE_COMMIT`, else `git rev-parse HEAD` in the checkout.
fn source_commit() -> String {
    if let Ok(commit) = std::env::var("Q36_SOURCE_COMMIT") {
        return commit;
    }
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run git, or set Q36_SOURCE_COMMIT");
    assert!(
        output.status.success(),
        "git rev-parse failed; set Q36_SOURCE_COMMIT"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn open_checkpoint(dir: &Path, index: &[u8]) -> StreamedCheckpoint {
    let names = shard_names(index);
    let shards = names
        .into_iter()
        .map(|name| {
            let file = File::open(dir.join(&name)).unwrap();
            let len = file.metadata().unwrap().len();
            let reader: Box<dyn ShardIo + Send> = Box::new(file);
            (name, reader, len)
        })
        .collect();
    StreamedCheckpoint::open(index, shards, 1 << 20).unwrap()
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn real_checkpoint_reference_run() {
    let (dir, out) = (env_path("Q36_CKPT_DIR"), env_path("Q36_OUT_DIR"));
    let prompt = std::env::var("Q36_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let greedy: usize = std::env::var("Q36_GREEDY_TOKENS")
        .ok()
        .map_or(32, |v| v.parse().unwrap());
    let threads: usize = std::env::var("Q36_THREADS").ok().map_or_else(
        || std::thread::available_parallelism().map_or(1, usize::from),
        |v| v.parse().unwrap(),
    );
    std::fs::create_dir(&out).expect("Q36_OUT_DIR must not exist yet");

    let config = std::fs::read(dir.join("config.json")).unwrap();
    let index = std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let tokenizer_json = std::fs::read(dir.join("tokenizer.json")).unwrap();
    let identity_of = |name: &str, bytes: &[u8]| FileId {
        name: name.into(),
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    };
    let mut files = vec![
        identity_of("config.json", &config),
        identity_of("model.safetensors.index.json", &index),
        identity_of("tokenizer.json", &tokenizer_json),
    ];
    for name in shard_names(&index) {
        files.push(hash_file(&dir.join(&name), &name));
    }
    // Fail before any compute when an input differs from the pinned checkpoint.
    if let Err(message) = check_pins(ENVIRONMENT_RECORD, &files) {
        panic!("input identity mismatch: {message}");
    }
    let source_commit = source_commit();
    let tokenizer = Tokenizer::from_bytes(&tokenizer_json).unwrap();
    let ckpt = open_checkpoint(&dir, &index);

    let started = Instant::now();
    let weights = HostWeights::load(&ckpt, &QWEN3_6_35B_A3B, &config).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    assert_eq!(
        weights.keys().len(),
        1757,
        "the load reads every tensor of the index"
    );
    let (shape, model) = (weights.shape(), weights.view());
    let mut projector = CpuProjector::new(&ckpt, threads);
    let mut state = HostState::new(shape);

    let prompt_ids = tokenizer.encode(&prompt).unwrap();
    assert!(!prompt_ids.is_empty());
    let (mut layers, mut steps, mut last_logits) = (Vec::new(), Vec::new(), Vec::new());
    let mut next = None;
    for (i, &token) in prompt_ids.iter().enumerate() {
        let started = Instant::now();
        let result = step(&mut projector, shape, &model, &mut state, token, true).unwrap();
        let seconds = started.elapsed().as_secs_f64();
        for layer in &result.layer_outputs {
            layers.extend(f32_bytes(layer.iter().copied()));
        }
        let ranked = top5(&result.logits);
        steps.push(serde_json::json!({"phase": "prompt", "index": i, "token": token, "top5": ranked, "seconds": seconds}));
        next = Some(ranked[0].0);
        if i + 1 == prompt_ids.len() {
            last_logits = f32_bytes(result.logits.iter().copied());
        }
    }
    let mut generated = Vec::new();
    for i in 0..greedy {
        let token = next.expect("a prompt step ran");
        generated.push(token);
        let started = Instant::now();
        let result = step(&mut projector, shape, &model, &mut state, token, false).unwrap();
        let ranked = top5(&result.logits);
        steps.push(serde_json::json!({"phase": "greedy", "index": i, "token": token, "top5": ranked, "seconds": started.elapsed().as_secs_f64()}));
        next = Some(ranked[0].0);
    }

    std::fs::write(out.join("layers.f32"), &layers).unwrap();
    std::fs::write(out.join("logits-last.f32"), &last_logits).unwrap();
    let report = serde_json::json!({
        "checkpoint": dir.display().to_string(),
        "source_commit": source_commit,
        "pins_source": "engine/testdata/mac-phase0a.json",
        "config_sha256": sha256(&config),
        "index_sha256": sha256(&index),
        "tokenizer_sha256": sha256(&tokenizer_json),
        "files": files.iter().map(|f| serde_json::json!({"name": f.name, "bytes": f.bytes, "sha256": f.sha256})).collect::<Vec<_>>(),
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "greedy_ids": generated,
        "greedy_text": tokenizer.decode(&generated).ok(),
        "layers_file": {"name": "layers.f32", "layout": "[prompt token][layer][hidden] little-endian f32", "sha256": sha256(&layers)},
        "logits_file": {"name": "logits-last.f32", "layout": "last prompt token, vocab little-endian f32", "sha256": sha256(&last_logits)},
        "shape": {"layers": shape.kinds.len(), "hidden": shape.hidden, "vocab": shape.vocab, "rotary_dim": shape.attn.rotary_dim},
        "threads": threads,
        "load_seconds": load_seconds,
        "steps": steps,
    });
    std::fs::write(
        out.join("host-ref.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out.join("host-ref.json").display());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned files of the record, as a run over the pinned checkpoint would find them.
    fn pinned_files() -> Vec<FileId> {
        let record: serde_json::Value = serde_json::from_str(ENVIRONMENT_RECORD).unwrap();
        let wanted = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ];
        wanted
            .iter()
            .map(|name| {
                let pin = record["checkpoint"]["files"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| p["path"] == *name)
                    .unwrap_or_else(|| panic!("{name} is pinned"));
                FileId {
                    name: (*name).into(),
                    bytes: pin["bytes"].as_u64().unwrap(),
                    sha256: pin["sha256"].as_str().unwrap().into(),
                }
            })
            .collect()
    }

    #[test]
    fn the_pinned_files_pass_and_the_pins_match_decision_0046() {
        let files = pinned_files();
        assert_eq!(check_pins(ENVIRONMENT_RECORD, &files), Ok(()));
        // Prefixes and suffixes of decision 0046 item 2.
        let sha = |name: &str| {
            files
                .iter()
                .find(|f| f.name == name)
                .unwrap()
                .sha256
                .clone()
        };
        assert!(sha("config.json").starts_with("a822a9e4"));
        assert!(sha("model.safetensors.index.json").ends_with("534d1"));
        assert!(sha("tokenizer.json").starts_with("87a7830d"));
        assert!(sha("model-00004-of-00004.safetensors").ends_with("a5422"));
    }

    /// The rejection message of `check_pins`; fails the test when the files are accepted.
    fn rejection(files: &[FileId]) -> String {
        let result = check_pins(ENVIRONMENT_RECORD, files);
        assert!(result.is_err(), "the files must be rejected");
        result.unwrap_err()
    }

    #[test]
    fn a_changed_hash_a_changed_size_a_missing_file_and_an_extra_file_fail() {
        let files = pinned_files();
        for victim in 0..files.len() {
            let mut changed = files.clone();
            changed[victim].sha256.replace_range(0..1, "z");
            let error = rejection(&changed);
            assert!(error.contains(&files[victim].name), "{error}");
            assert!(error.contains("SHA-256"), "{error}");
        }
        let mut resized = files.clone();
        resized[3].bytes += 1;
        assert!(rejection(&resized).contains("size"));
        assert!(rejection(&files[..6]).contains("not among the files found"));
        let mut extra = files.clone();
        extra.push(FileId {
            name: "lora_edge0_35b.safetensors".into(),
            bytes: 1,
            sha256: "00".into(),
        });
        assert!(rejection(&extra).contains("not a pinned file"));
    }

    #[test]
    fn hash_file_returns_the_size_and_sha256_of_the_bytes() {
        let dir = std::env::temp_dir().join(format!("q36-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abc.bin");
        std::fs::write(&path, b"abc").unwrap();
        let id = hash_file(&path, "abc.bin");
        std::fs::remove_dir_all(&dir).unwrap();
        // SHA-256("abc") from FIPS 180-2 appendix B.1.
        assert_eq!(
            id,
            FileId {
                name: "abc.bin".into(),
                bytes: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            }
        );
    }

    #[test]
    fn the_shard_names_of_an_index_are_sorted_and_unique() {
        let index = br#"{"weight_map": {"a": "s2.safetensors", "b": "s1.safetensors", "c": "s2.safetensors"}}"#;
        assert_eq!(shard_names(index), ["s1.safetensors", "s2.safetensors"]);
    }
}
