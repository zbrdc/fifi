//! The K/V bytes golden of the Flash-Next planning arithmetic: what one position costs in
//! each format, the f32 row-scale term isolated on its own line, the second copy of the
//! element arithmetic at `tier_plan`, and the slot counts a real context yields. A change that
//! moves any of these numbers must re-capture deliberately and name the move in
//! the commit message. `FIFI_GOLDEN_UPDATE=1` rewrites this family's golden file beside the
//! module.

use super::*;
use crate::family::exl3_pack::{Marker, PackReport};
use crate::family::flash_next::load::exl3_pack::FlashPackReport;
use crate::family::flash_next::moe::load_plan::FlashLoadPlan;
use crate::family::flash_next::moe::tier_plan::TierShape;
use std::collections::BTreeMap;
use std::fmt::Write as _;

const CONFIG: &[u8] =
    include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");

fn pack_report(trunk: u64) -> FlashPackReport {
    let experts = 24_576 * EXPERT_BYTES;
    let embedding = 248_320u64 * 2560 * 2;
    FlashPackReport {
        pack: PackReport {
            device_bytes: experts + embedding + trunk,
            expert_bytes: experts,
            matrices: 74_041,
            rates: BTreeMap::new(),
            codebook: Marker::Mul1,
        },
    }
}

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
    // Local labels, not {format:?}: the golden must move on bytes, not on a
    // Debug name.
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

/// Renders the slot counts a real context yields per format, through the load plan's
/// own geometry (the plan the loader computes, not the budget helper alone).
fn slots_text() -> String {
    let mut text = String::new();
    let shape = TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES).unwrap();
    let card = TierInputs::cost_model_3090(
        shape,
        (0, 0),
        KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
    );
    let load =
        FlashLoadPlan::new(&pack_report(3_854_000_000), CONFIG, 32_640_183_408, card).unwrap();
    for positions in [262_144u64, 131_072, 32_768] {
        let plan = load.at_context(positions, KvFormat::F16, 100).unwrap();
        let _ = writeln!(
            text,
            "positions {positions} f16 slots_per_layer={} kv_bytes={}",
            plan.context.plan.slots_per_layer(),
            plan.kv_bytes
        );
    }
    // The load plan refuses Int8 until the attention kernel exists; the refusal text
    // is real behavior and pins with the numbers.
    let refused = load
        .at_context(262_144, KvFormat::Int8, 100)
        .unwrap_err()
        .describe();
    let _ = writeln!(text, "int8 refused: {refused}");
    text
}

/// The settings member the user actually writes: the f16 default resolves and the
/// int8 refusal text stays verbatim. Pinned before any step touches the parse or
/// the planning arithmetic.
fn settings_text() -> String {
    let mut text = String::new();
    let base = crate::family::flash_next::settings::FlashOptions {
        manifest: Some(std::path::PathBuf::from("/art/download-manifest.json")),
        // cache_limit_bytes stays unset: it is an MLX key and refuses to combine
        // with cuda_decode, which is the arm this pin drives.
        cuda_decode: Some(true),
        ..crate::family::flash_next::settings::FlashOptions::default()
    };
    let resolved = crate::family::flash_next::settings::FlashSettings::new(
        std::path::Path::new("/art"),
        std::path::Path::new("/tok"),
        Some(&base),
        &crate::family::flash_next::settings::FlashEnv::default(),
    );
    let _ = writeln!(text, "default kv_format resolves={}", resolved.is_ok());
    let int8 = crate::family::flash_next::settings::FlashOptions {
        kv_format: Some(crate::family::flash_next::settings::KvFormatName::Int8),
        ..base
    };
    let refused = crate::family::flash_next::settings::FlashSettings::new(
        std::path::Path::new("/art"),
        std::path::Path::new("/tok"),
        Some(&int8),
        &crate::family::flash_next::settings::FlashEnv::default(),
    )
    .unwrap_err()
    .describe();
    // The arm is self-checking: the rendered lines must actually be the resolution
    // and the refusal, not some other error path.
    assert!(
        resolved.is_ok(),
        "the f16 default must resolve: {resolved:?}"
    );
    assert!(
        refused.contains("no attention kernel"),
        "the refusal text changed: {refused}"
    );
    let _ = writeln!(text, "int8 refused: {refused}");
    text
}

/// The digests of this family's arms.
fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("position", position_text()),
        ("tier", tier_text()),
        ("slots", slots_text()),
        ("settings", settings_text()),
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
            // Atomic: a capture that dies mid-write never leaves a half-written
            // golden; the rename is the commit point.
            let rendered = serde_json::to_string_pretty(&goldens).unwrap()
                + "
";
            let temp = path.with_extension("json.tmp");
            std::fs::write(&temp, rendered).unwrap();
            std::fs::rename(&temp, &path).unwrap();
            println!("GOLDEN UPDATED flash-next kv bytes");
        });
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed kv bytes goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no flash-next kv arms; capture with FIFI_GOLDEN_UPDATE=1"));
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
    println!("GOLDEN MATCH flash-next kv bytes");
}
