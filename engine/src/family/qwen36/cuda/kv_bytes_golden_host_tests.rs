//! The K/V bytes golden of the device-side planning arithmetic: the pool's per-token figure for
//! both modes, the prefill window bytes, the loader's admission term at real contexts, the
//! placement stride and cache figures, and the `KvMode` `Debug` text the activation fingerprint
//! embeds (`step_activation.rs`). A change that moves any of these numbers must re-capture
//! deliberately and name the move in the commit message. `FIFI_GOLDEN_UPDATE=1` rewrites this
//! family's golden file beside the module.

use crate::family::gguf_model::Inventory;
use crate::family::qwen36::cuda::dims::{KvMode, StepDims};
use crate::family::qwen36::cuda::placement;
use crate::family::qwen36::cuda::prefill_plan;
use crate::family::qwen36::cuda::start;
use crate::family::qwen36::cuda::state_layout::KvPool;
use crate::family::qwen36::host::HostShape;
use crate::family::spec::QWEN3_6_35B_A3B;
use std::fmt::Write as _;

fn dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("real shape");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(&shape, &inventory).expect("dims")
}

fn digest(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// The pool's per-token figure for both modes at the 35B spec's geometry.
fn pool_text() -> String {
    let mut text = String::new();
    let dims = dims();
    let _ = writeln!(
        text,
        "heads_kv={} head_dim={}",
        dims.shape.attn.heads_kv, dims.shape.attn.head_dim
    );
    // Local labels, not {mode:?}: the golden must move on bytes, not on a Debug
    // name. The dtype arm below is the one deliberate exception: it pins the
    // Debug text itself.
    for (label, mode) in [("fp16", KvMode::Fp16), ("int8g32", KvMode::Int8G32)] {
        let bytes = KvPool::bytes_per_token(&dims, mode).unwrap();
        let _ = writeln!(text, "{label} bytes_per_token={bytes}");
    }
    // The pool the decoder actually reserves: aligned per-layer stride and whole-pool
    // bytes at a real capacity, both modes, full heads.
    for (label, mode) in [("fp16", KvMode::Fp16), ("int8g32", KvMode::Int8G32)] {
        let pool = KvPool::new(&dims, mode, 16_384).unwrap();
        let _ = writeln!(
            text,
            "{label} capacity 16384 aligned_layer_stride={} total={}",
            pool.layer_stride(),
            pool.total
        );
    }
    let staging = KvPool::q8_staging_layer(&dims, 16_384).unwrap();
    let _ = writeln!(
        text,
        "int8g32 staging_layer_stride={} staging_total={}",
        staging.layer_stride(),
        staging.total
    );
    text
}

/// The prefill window bytes and the loader's admission term at real contexts.
fn admission_text() -> String {
    let mut text = String::new();
    let dims = dims();
    for window in [512usize, 4_096, 45_056] {
        for (label, mode) in [("fp16", KvMode::Fp16), ("int8g32", KvMode::Int8G32)] {
            let bytes = prefill_plan::kv_bytes(&dims, mode, window).unwrap();
            let _ = writeln!(text, "window {window} {label} kv_bytes={bytes}");
        }
    }
    for context in [4_096usize, 32_768, 262_144] {
        for (label, mode) in [("fp16", KvMode::Fp16), ("int8g32", KvMode::Int8G32)] {
            let bytes = start::kv_cache_bytes(context, mode).unwrap();
            let _ = writeln!(text, "context {context} {label} kv_cache_bytes={bytes}");
        }
    }
    text
}

/// The placement stride and whole-cache figures the placement plan charges.
fn placement_text() -> String {
    let mut text = String::new();
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("real shape");
    let stride = placement::kv_layer_stride_bytes(&shape).unwrap();
    let _ = writeln!(text, "stride_bytes={stride}");
    for capacity in [16_384usize, 45_056] {
        let bytes = placement::kv_cache_bytes(&shape, 40, capacity).unwrap();
        let _ = writeln!(text, "layers 40 capacity {capacity} kv_cache_bytes={bytes}");
    }
    text
}

/// The `KvMode` `Debug` text the activation execution fingerprint embeds.
fn dtype_text() -> String {
    let mut text = String::new();
    let _ = writeln!(text, "Fp16 debug={:?}", KvMode::Fp16);
    let _ = writeln!(text, "Int8G32 debug={:?}", KvMode::Int8G32);
    text
}

/// The digests of this family's arms.
fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("pool", pool_text()),
        ("admission", admission_text()),
        ("placement", placement_text()),
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
                .unwrap_or_else(|_| serde_json::json!({ "family": "qwen36-cuda" }));
            goldens["family"] = "qwen36-cuda".into();
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
            println!("GOLDEN UPDATED qwen36-cuda kv bytes");
        });
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed kv bytes goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no qwen36-cuda kv arms; capture with FIFI_GOLDEN_UPDATE=1"));
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
    println!("GOLDEN MATCH qwen36-cuda kv bytes");
}
