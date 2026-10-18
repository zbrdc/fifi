//! Real-header check for the Flash-Next IQ3_XXS GGUF shards.
//!
//! The command reads only header prefixes. It never reads tensor payloads.
//! Environment variables:
//! - `GGUF_SHARD1`, `GGUF_SHARD2`: paths of the shard header prefixes.
//! - `GGUF_LEN1`, `GGUF_LEN2`: true full file lengths in bytes.
//! - `GGUF_EVIDENCE`: path of the recorded per-type byte JSON.
//! - `GGUF_HEADER_LIMIT`: optional header byte limit (default 64 MiB).
//!
//! The command prints one JSON report and exits nonzero on any mismatch.

use std::collections::BTreeMap;
use std::fs::File;

use inference_engine::weights::{GgufHeader, GgufType, MetaValue, sha256_hex, sha256_stream};
use serde_json::{Value, json};

const SHARD2_TENSOR: &str = "per_layer_token_embd.weight";
const SHARD2_BYTES: u64 = 28_800_138_240;
const SHARD2_OFFSET: u64 = 192;
const SHARD2_ROWS: u64 = 320_001_536;
const SHARD2_COLS: u64 = 160;
/// Numeric metadata arrays the check expects. All are small model parameters.
const KNOWN_NUMERIC_ARRAYS: [&str; 7] = [
    "rope.dimension_sections",
    "attention.compress_ratios",
    "ple.layers",
    "ple.layer_multipliers",
    "ple.head_offsets",
    "ple.head_vocab_sizes",
    "tokenizer.ggml.token_type",
];
const SHARD1_TENSORS: usize = 1_223;
const TABLE_WORDS: [&str; 8] = [
    "grid", "table", "kmask", "ksigns", "codebook", "lut", "kvalues", "signs",
];

fn env(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|_| format!("set {name}"))
}

fn env_u64(name: &str) -> Result<u64, String> {
    env(name)?
        .parse()
        .map_err(|_| format!("{name} is not a number"))
}

struct Shard {
    header: GgufHeader,
    prefix_len: u64,
    prefix_sha256: String,
}

fn load(path: &str, file_len: u64, limit: u64) -> Result<Shard, String> {
    let mut file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let prefix_len = file.metadata().map_err(|e| e.to_string())?.len();
    let header =
        GgufHeader::parse(&mut file, file_len, limit).map_err(|e| format!("{path}: {e:?}"))?;
    let mut again = File::open(path).map_err(|e| e.to_string())?;
    let (digest, hashed) = sha256_stream(&mut again).map_err(|e| format!("{e:?}"))?;
    if hashed != prefix_len {
        return Err("prefix length changed while hashing".into());
    }
    Ok(Shard {
        header,
        prefix_len,
        prefix_sha256: sha256_hex(&digest),
    })
}

fn class(name: &str) -> &'static str {
    if name.contains("_exps") {
        "experts"
    } else if name.starts_with("per_layer_token_embd") {
        "ngram"
    } else {
        "non_expert"
    }
}

fn totals(h: &GgufHeader) -> (BTreeMap<String, u64>, BTreeMap<String, u64>) {
    let mut by_class = BTreeMap::new();
    let mut by_type = BTreeMap::new();
    for t in h.tensors() {
        let c = class(t.name());
        *by_class.entry(c.to_string()).or_insert(0) += t.byte_len();
        *by_type
            .entry(format!("{c}:{}", t.ggml_type().name()))
            .or_insert(0) += t.byte_len();
    }
    (by_class, by_type)
}

fn table_hits(h: &GgufHeader) -> Vec<String> {
    let mut hits = Vec::new();
    for t in h.tensors() {
        let lower = t.name().to_ascii_lowercase();
        if TABLE_WORDS.iter().any(|w| lower.contains(w)) {
            hits.push(format!("tensor {}", t.name()));
        }
    }
    for (key, value) in h.metadata() {
        let lower = key.to_ascii_lowercase();
        if TABLE_WORDS.iter().any(|w| lower.contains(w)) {
            hits.push(format!("metadata {key}"));
        }
        if let MetaValue::Array(a) = value
            && is_numeric(a.elem_type)
            && !KNOWN_NUMERIC_ARRAYS.iter().any(|k| key.contains(k))
        {
            hits.push(format!(
                "unexpected numeric metadata array {key} (type {}, {} elements)",
                a.elem_type, a.len
            ));
        }
    }
    hits
}

/// Numeric metadata array element types: integers, floats. Excludes bool, string, array.
fn is_numeric(elem_type: u32) -> bool {
    matches!(elem_type, 0..=6 | 10..=12)
}

/// Every numeric metadata array in a header, as `key`, element type, and count.
fn numeric_arrays(h: &GgufHeader) -> Vec<Value> {
    h.metadata()
        .iter()
        .filter_map(|(key, value)| match value {
            MetaValue::Array(a) if is_numeric(a.elem_type) => {
                Some(json!({"key": key, "elem_type": a.elem_type, "len": a.len}))
            }
            _ => None,
        })
        .collect()
}

fn map_u64(v: &Value) -> BTreeMap<String, u64> {
    v.as_object()
        .map(|o| {
            o.iter()
                .filter_map(|(k, x)| x.as_u64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default()
}

fn run() -> Result<Value, String> {
    let limit = env("GGUF_HEADER_LIMIT").map_or(Ok(64 << 20), |s| {
        s.parse()
            .map_err(|_| "GGUF_HEADER_LIMIT is not a number".to_string())
    })?;
    let len1 = env_u64("GGUF_LEN1")?;
    let len2 = env_u64("GGUF_LEN2")?;
    let s1 = load(&env("GGUF_SHARD1")?, len1, limit)?;
    let s2 = load(&env("GGUF_SHARD2")?, len2, limit)?;
    let mut failures: Vec<String> = Vec::new();
    let mut check = |ok: bool, what: &str| {
        if !ok {
            failures.push(what.to_string());
        }
    };

    let (c1, t1) = totals(&s1.header);
    let (c2, t2) = totals(&s2.header);
    check(
        s1.header.tensors().len() == SHARD1_TENSORS,
        "shard 1 has 1223 tensors",
    );
    check(s2.header.tensors().len() == 1, "shard 2 has one tensor");
    let split1 = s1.header.split();
    let split2 = s2.header.split();
    check(
        split1.is_some_and(|s| s.count == 2 && s.no == 0),
        "shard 1 split.count 2 and split.no 0",
    );
    check(
        split2.is_some_and(|s| s.count == 2 && s.no == 1),
        "shard 2 split.count 2 and split.no 1",
    );
    check(
        inference_engine::weights::check_split_set(&[&s1.header, &s2.header]).is_ok(),
        "split set is consistent",
    );
    check(
        s1.header.data_start() == 11_024_256 && s1.header.header_len() <= s1.header.data_start(),
        "shard 1 data starts at 11024256",
    );
    check(
        s1.header.data_start() + c1.values().sum::<u64>() == len1,
        "shard 1 data start plus tensor bytes equals the file length",
    );
    match s2.header.tensor(SHARD2_TENSOR) {
        Some(t) => {
            check(t.ggml_type() == GgufType::Iq4Nl, "ngram type is IQ4_NL");
            check(
                t.dims() == [SHARD2_COLS, SHARD2_ROWS],
                "ngram dims are [160, 320001536]",
            );
            check(t.abs_offset() == SHARD2_OFFSET, "ngram data offset is 192");
            check(
                t.byte_len() == SHARD2_BYTES,
                "ngram byte size is 28800138240",
            );
            check(
                t.abs_range().end == len2 && len2 == SHARD2_BYTES + SHARD2_OFFSET,
                "ngram range ends at the file end",
            );
        }
        None => check(false, "ngram tensor exists"),
    }

    let evidence: Value = serde_json::from_str(
        &std::fs::read_to_string(env("GGUF_EVIDENCE")?).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut matched = Vec::new();
    for (idx, (class_map, type_map)) in [(&c1, &t1), (&c2, &t2)].into_iter().enumerate() {
        let suffix = format!("0000{}-of-00002.gguf", idx + 1);
        let entry = evidence
            .as_object()
            .and_then(|o| o.iter().find(|(k, _)| k.ends_with(&suffix)))
            .map(|(_, v)| v);
        match entry {
            Some(e) => {
                let eb = map_u64(&e["bytes"]);
                let et = map_u64(&e["by_type"]);
                check(
                    &eb == class_map,
                    &format!("shard {} class totals match evidence", idx + 1),
                );
                check(
                    &et == type_map,
                    &format!("shard {} per-type totals match evidence", idx + 1),
                );
                matched.push(idx + 1);
            }
            None => check(false, &format!("evidence entry for shard {}", idx + 1)),
        }
    }
    let hits1 = table_hits(&s1.header);
    let hits2 = table_hits(&s2.header);
    check(
        hits1.is_empty() && hits2.is_empty(),
        "no grid or lookup tables in either header",
    );

    Ok(json!({
        "ok": failures.is_empty(),
        "failures": failures,
        "evidence_shards_matched": matched,
        "shard1": {
            "file_len": len1, "prefix_len": s1.prefix_len, "prefix_sha256": s1.prefix_sha256,
            "header_len": s1.header.header_len(), "data_start": s1.header.data_start(),
            "header_sha256": sha256_hex(&s1.header.header_sha256()),
            "tensors": s1.header.tensors().len(), "metadata_keys": s1.header.metadata().len(),
            "split": split1.map(|s| json!({"count": s.count, "no": s.no, "tensors_count": s.tensors_count})),
            "class_bytes": c1, "type_bytes": t1, "table_hits": hits1, "numeric_arrays": numeric_arrays(&s1.header),
        },
        "shard2": {
            "file_len": len2, "prefix_len": s2.prefix_len, "prefix_sha256": s2.prefix_sha256,
            "header_len": s2.header.header_len(), "data_start": s2.header.data_start(),
            "header_sha256": sha256_hex(&s2.header.header_sha256()),
            "tensors": s2.header.tensors().len(), "metadata_keys": s2.header.metadata().len(),
            "split": split2.map(|s| json!({"count": s.count, "no": s.no, "tensors_count": s.tensors_count})),
            "class_bytes": c2, "type_bytes": t2, "table_hits": hits2, "numeric_arrays": numeric_arrays(&s2.header),
        },
    }))
}

fn main() {
    match run() {
        Ok(report) => {
            let text = serde_json::to_string_pretty(&report).unwrap_or_default();
            match env("GGUF_REPORT") {
                Ok(path) => {
                    if let Err(e) = std::fs::write(&path, format!("{text}\n")) {
                        eprintln!("gguf-header: cannot write {path}: {e}");
                        std::process::exit(2);
                    }
                }
                Err(_) => println!("{text}"),
            }
            if report["ok"] != true {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("gguf-header: {e}");
            std::process::exit(2);
        }
    }
}
