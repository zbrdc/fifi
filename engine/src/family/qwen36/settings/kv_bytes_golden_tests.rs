//! The K/V bytes golden of the host settings arithmetic: per-layer and per-position growth for
//! the full `kv_dtype` × `activation_dtype` grid at the 35B spec's geometry, the Q8 staging
//! cache, and the `kv_dtype` name/parse round trip. A change that moves any of these numbers
//! must re-capture deliberately and name the move in the commit message.
//! `FIFI_GOLDEN_UPDATE=1` rewrites this family's golden file beside the module.

use super::{
    Activation, KvDtype, activation_name, cache_growth_per_position, kv_dtype_name,
    kv_growth_per_position, parse_activation, parse_kv_dtype,
};
use crate::family::qwen36::admission_sketch::kv8_staging_bytes;
use crate::family::spec::QWEN3_6_35B_A3B;
use std::fmt::Write as _;

fn digest(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// The per-layer growth grid at the 35B spec's KV geometry (heads_kv=2, head_dim=256).
fn growth_text() -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "spec heads_kv={} head_dim={}",
        QWEN3_6_35B_A3B.heads_kv, QWEN3_6_35B_A3B.head_dim
    );
    for kv in [KvDtype::F32, KvDtype::Bf16, KvDtype::Q8] {
        for act in [Activation::F32, Activation::Bf16] {
            let bytes =
                kv_growth_per_position(QWEN3_6_35B_A3B.heads_kv, QWEN3_6_35B_A3B.head_dim, kv, act)
                    .unwrap();
            let _ = writeln!(
                text,
                "kv={} act={} per_layer_bytes={bytes}",
                kv_dtype_name(kv),
                activation_name(act)
            );
        }
    }
    for kv in [KvDtype::F32, KvDtype::Bf16, KvDtype::Q8] {
        let bytes = cache_growth_per_position(&QWEN3_6_35B_A3B, kv, Activation::Bf16).unwrap();
        let _ = writeln!(
            text,
            "kv=bf16-act cache_per_position_bytes={bytes} kv={}",
            kv_dtype_name(kv)
        );
    }
    // The preflight figure for the CUDA f32 arm doubles this row (40,960 where the
    // real per-position growth at the bf16 default is 20,480); pinned as is until
    // preflight is unified.
    let _ = writeln!(
        text,
        "preflight_cuda_f32_figure=40960 doubles the bf16 default row above; pinned as is"
    );
    text
}

/// The Q8 time-major FP16 staging cache at a real ring capacity.
fn staging_text() -> String {
    let mut text = String::new();
    for capacity in [16_384usize, 45_056] {
        let bytes = kv8_staging_bytes(capacity, QWEN3_6_35B_A3B.heads_kv as usize, 256).unwrap();
        let _ = writeln!(text, "capacity {capacity} staging_bytes={bytes}");
    }
    text
}

/// The `kv_dtype` and `activation_dtype` member round trip: every name parses to its value and
/// names back.
fn dtype_text() -> String {
    let mut text = String::new();
    for kv in [KvDtype::F32, KvDtype::Bf16, KvDtype::Q8] {
        let name = kv_dtype_name(kv);
        let _ = writeln!(
            text,
            "{name} parses={} round_trip={}",
            parse_kv_dtype(name).is_ok(),
            kv_dtype_name(parse_kv_dtype(name).unwrap()) == name
        );
    }
    let _ = writeln!(
        text,
        "f32 activation parses={}",
        parse_activation("f32").is_ok()
    );
    let _ = writeln!(
        text,
        "bf16 activation parses={}",
        parse_activation("bf16").is_ok()
    );
    text
}

/// The digests of this family's arms.
fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("growth", growth_text()),
        ("staging", staging_text()),
        ("dtype", dtype_text()),
    ] {
        map.insert(name.into(), digest(&text).into());
    }
    map
}

/// Serializes `FIFI_GOLDEN_UPDATE` captures under a `create_new` guard beside the golden.
/// This family's golden file is ours alone.
fn with_golden_lock<R>(path: &std::path::Path, update: impl FnOnce() -> R) -> R {
    let lock = path.with_extension("lock");
    let began = std::time::Instant::now();
    while std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .is_err()
    {
        if began.elapsed() > std::time::Duration::from_secs(60) {
            panic!(
                "the kv-bytes update lock {} is held or stale; delete it if no capture runs",
                lock.display()
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let result = update();
    let _ = std::fs::remove_file(&lock);
    result
}

#[test]
fn the_kv_bytes_hold_their_golden_digests() {
    const GOLDENS_TEXT: &str = include_str!("testdata/kv-bytes-goldens.json");
    let path = std::path::Path::new(file!())
        .parent()
        .unwrap()
        .join("testdata/kv-bytes-goldens.json");
    let sections = sections();
    if std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty()) {
        with_golden_lock(&path, || {
            let mut goldens: serde_json::Value = serde_json::from_str(GOLDENS_TEXT)
                .unwrap_or_else(|_| serde_json::json!({ "family": "qwen36-host" }));
            goldens["family"] = "qwen36-host".into();
            let mut arms: std::collections::BTreeMap<&str, serde_json::Value> = goldens
                .get("arms")
                .and_then(|v| v.as_object())
                .map(|arms| arms.iter().map(|(k, v)| (k.as_str(), v.clone())).collect())
                .unwrap_or_default();
            for (name, sha) in &sections {
                arms.insert(name, sha.clone());
            }
            goldens["arms"] = serde_json::Value::Object(
                arms.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            );
            // Atomic: a capture that dies mid-write never leaves a half-written
            // golden; the rename is the commit point.
            let rendered = serde_json::to_string_pretty(&goldens).unwrap()
                + "
";
            let temp = path.with_extension("json.tmp");
            std::fs::write(&temp, rendered).unwrap();
            std::fs::rename(&temp, &path).unwrap();
            println!("GOLDEN UPDATED qwen36-host kv bytes");
        });
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed kv bytes goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no qwen36-host kv arms; capture with FIFI_GOLDEN_UPDATE=1"));
    // Report every drift in one assert, not just the first failure.
    let mut drifted: Vec<String> = Vec::new();
    for (name, sha) in &sections {
        let pinned_sha = pinned_arms.get(name).and_then(|v| v.as_str()).unwrap_or("");
        if pinned_sha != sha {
            drifted.push(format!("arm {name}: pinned {pinned_sha}, rendered {sha}"));
        }
    }
    for name in pinned_arms.keys() {
        if !sections.contains_key(name) {
            drifted.push(format!("arm {name}: pinned but no longer renders"));
        }
    }
    assert!(
        drifted.is_empty(),
        "kv bytes goldens moved (capture deliberately and name the move in the commit):
{}",
        drifted.join(
            "
"
        )
    );
    println!("GOLDEN MATCH qwen36-host kv bytes");
}
