//! The K/V bytes golden of the Flash-Next planning arithmetic: what one position costs in
//! each format, the f32 row-scale term isolated on its own line, the second copy of the
//! element arithmetic at `tier_plan`, and the slot counts a real context yields. A change that
//! moves any of these numbers must re-capture deliberately and name the move in
//! the commit message. `FIFI_GOLDEN_UPDATE=1` rewrites this family's golden file beside the
//! module.

use super::*;
use crate::family::flash_next::moe::tier_plan::TierShape;
use std::fmt::Write as _;

const MIB: u64 = 1 << 20;
const EXPERT_BYTES: u64 = 1_881_600;
const TRUNK_BYTES: u64 = 3_854_000_000;
const STATE_BYTES: u64 = 36 * 3 * MIB;

const FLASH_NEXT: KvGeometry = KvGeometry {
    cached_layers: 12,
    heads_kv: 2,
    head_dim: 256,
    idx_dim: 128,
    ratio: 4,
};

fn budget() -> ContextBudget {
    let shape = TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES).unwrap();
    let placeholder = KvPlan {
        positions: 1,
        bytes_per_position: 1,
    };
    ContextBudget::new(
        TierInputs::cost_model_3090(shape, (TRUNK_BYTES, STATE_BYTES), placeholder),
        FLASH_NEXT,
    )
}

fn digest(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Renders the two formats' position bytes with the Int8 scale term on its own line.
fn position_text() -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "geometry layers={} heads_kv={} head_dim={} idx_dim={} ratio={}",
        FLASH_NEXT.cached_layers,
        FLASH_NEXT.heads_kv,
        FLASH_NEXT.head_dim,
        FLASH_NEXT.idx_dim,
        FLASH_NEXT.ratio
    );
    // Local labels, not {format:?}: the P5 ladder replaces this enum, and the golden
    // must move on bytes, not on a Debug name.
    for (label, format) in [("f16", KvFormat::F16), ("int8", KvFormat::Int8)] {
        let total = FLASH_NEXT.position_bytes(format).unwrap();
        let scales = FLASH_NEXT
            .cached_layers
            .checked_mul(2 * FLASH_NEXT.heads_kv)
            .and_then(|rows| rows.checked_mul(format.row_scale_bytes()))
            .unwrap();
        let _ = writeln!(
            text,
            "{label} position_bytes={total} row_scale_bytes={}",
            scales
        );
    }
    text
}

/// Renders the second copy of the element arithmetic: tier-level position bytes per element
/// width (no scale bytes — the tier plan's own doc says the budget adds them).
fn tier_text() -> String {
    let mut text = String::new();
    for elem in [2u64, 1] {
        let bytes = crate::family::flash_next::moe::tier_plan::position_bytes(
            FLASH_NEXT.cached_layers,
            FLASH_NEXT.heads_kv,
            FLASH_NEXT.head_dim,
            elem,
            (FLASH_NEXT.idx_dim, FLASH_NEXT.ratio),
        )
        .unwrap();
        let _ = writeln!(text, "elem {elem} tier_position_bytes={bytes}");
    }
    text
}

/// Renders the slot counts a real context yields per format: the load plan's K/V shape term.
fn slots_text() -> String {
    let mut text = String::new();
    for positions in [262_144u64, 131_072, 32_768] {
        for (label, format) in [("f16", KvFormat::F16), ("int8", KvFormat::Int8)] {
            let plan = budget().plan_at(positions, format).unwrap();
            let _ = writeln!(
                text,
                "positions {positions} {label} slots_per_layer={}",
                plan.slots_per_layer()
            );
        }
    }
    text
}

/// The digests of this family's arms.
fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("position", position_text()),
        ("tier", tier_text()),
        ("slots", slots_text()),
    ] {
        map.insert(name.into(), digest(&text).into());
    }
    map
}

/// Serializes `FIFI_GOLDEN_UPDATE` captures under a `create_new` guard beside the golden
/// (atomic everywhere we run; a capture crashed mid-update leaves the guard behind and the
/// next capture panics after a minute — delete the stale file then). This family's golden
/// file is ours alone.
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
    const GOLDENS_TEXT: &str = include_str!("../testdata/kv-bytes-goldens.json");
    let path = std::path::Path::new(file!())
        .parent()
        .unwrap()
        .join("../testdata/kv-bytes-goldens.json");
    let sections = sections();
    if std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty()) {
        with_golden_lock(&path, || {
            let mut goldens: serde_json::Value = serde_json::from_str(GOLDENS_TEXT)
                .unwrap_or_else(|_| serde_json::json!({ "family": "flash-next" }));
            goldens["family"] = "flash-next".into();
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
            println!("GOLDEN UPDATED flash-next kv bytes");
        });
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed kv bytes goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no flash-next kv arms; capture with FIFI_GOLDEN_UPDATE=1"));
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
    println!("GOLDEN MATCH flash-next kv bytes");
}
