//! The K/V bytes golden of the flint plan: per-plane row bytes for both storage modes, the
//! full K+V row bytes, and the uniform/mixed plans' own bookkeeping at real ring sizes.
//! A change that moves any of these numbers must re-capture deliberately and name the move
//! in the commit message. `FIFI_GOLDEN_UPDATE=1` rewrites this family's golden file beside
//! the module.

use super::{config::FULL_LAYERS, kv_plan::KvPlan, kv_plan::KvStorage};
use std::fmt::Write as _;

fn storage_label(storage: KvStorage) -> &'static str {
    match storage {
        KvStorage::F16 => "f16",
        KvStorage::Q8G32 => "q8g32",
    }
}

fn digest(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Per-plane payload and scale bytes, then the full row, for both storage modes.
fn plane_text() -> String {
    let mut text = String::new();
    // Local labels, not {storage:?}: the P5 ladder replaces this enum, and the golden
    // must move on bytes, not on a Debug name.
    for (label, storage) in [("f16", KvStorage::F16), ("q8g32", KvStorage::Q8G32)] {
        let (data, scales) = storage.plane_row_bytes();
        let _ = writeln!(
            text,
            "{label} plane_data_bytes={data} plane_scale_bytes={scales} row_bytes={}",
            storage.row_bytes()
        );
    }
    text
}

/// The ring plan's own bytes bookkeeping at the sizes the executor tests use: uniform F16,
/// uniform Q8G32, and the K/V mixed plan.
fn plan_text() -> String {
    let mut text = String::new();
    let full_layers = FULL_LAYERS.len();
    let _ = writeln!(text, "full_layers={full_layers}");
    let plans = [
        ("uniform", KvPlan::new(262_144, 512, true).unwrap()),
        (
            "uniform_q8",
            KvPlan::new(262_144, 512, true)
                .unwrap()
                .with_storage(KvStorage::Q8G32),
        ),
        (
            "mixed",
            KvPlan::new(262_144, 512, true)
                .unwrap()
                .with_kv_storage(KvStorage::Q8G32, KvStorage::F16),
        ),
    ];
    for (name, plan) in plans {
        let (k_bytes, v_bytes) = plan.per_plane_row_bytes();
        let _ = writeln!(
            text,
            "{name} cache_rows[0]={} cache_rows[48]={} k_row={k_bytes} v_row={v_bytes} k={} v={}",
            plan.cache_rows(0).unwrap(),
            plan.cache_rows(48).unwrap(),
            storage_label(plan.k_storage()),
            storage_label(plan.v_storage())
        );
    }
    text
}

/// The digests of this family's arms.
fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [("plane", plane_text()), ("plan", plan_text())] {
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
                .unwrap_or_else(|_| serde_json::json!({ "family": "flint" }));
            goldens["family"] = "flint".into();
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
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&goldens).unwrap() + "\n",
            )
            .unwrap();
            println!("GOLDEN UPDATED flint kv bytes");
        });
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed kv bytes goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no flint kv arms; capture with FIFI_GOLDEN_UPDATE=1"));
    for (name, sha) in &sections {
        let pinned_sha = pinned_arms.get(name).and_then(|v| v.as_str()).unwrap_or("");
        assert_eq!(pinned_sha, sha, "arm {name} moved; capture deliberately");
    }
    for name in pinned_arms.keys() {
        assert!(
            sections.contains_key(name),
            "pinned arm {name} no longer renders"
        );
    }
    println!("GOLDEN MATCH flint kv bytes");
}
