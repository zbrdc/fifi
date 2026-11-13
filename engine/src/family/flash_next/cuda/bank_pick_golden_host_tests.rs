//! The pick-replay golden of the Flash-Next bank policy: `ReplaceOnMissLru` over a
//! popularity-filled bank (`from_fill`), replayed over deterministic pick streams with router
//! faults (-1) included, recording per call the picks, the plan (distinct, hits, moves), every
//! slots and the row after (incoming/outgoing expert), the refill a `DeviceBank` would issue
//! for it (one record, offset and length), the classifier's slot row after every call, and
//! one over-capacity `admit` with repeats on the replayed layer (the union-admit entry point
//! a prefill chunk uses). The error arm pins `plan_pass`'s atomicity ("an Err changes
//! nothing"): the row after a refused call is byte-identical.
//!
//! Host-only, modelless, GPUless. `FIFI_GOLDEN_UPDATE=1` rewrites this family's section of
//! `testdata/pick-goldens.json` (the file the Q36 pick golden in
//! `qwen36/cuda/exl3_bank_pick_golden_host_tests.rs` also owns a subtree of; one golden
//! mechanism, same switch as the prefill golden harness) atomically; without it the compile-time
//! pin is compared.

use super::bank::*;
use sha2::{Digest, Sha256};
use std::fmt::Write;

const LAYERS: usize = 2;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const SLOTS: usize = 48;
const RECORD: usize = 1_881_600;
const CALLS: usize = 200;
const PICKS: usize = 10;

fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// A deterministic popularity profile: expert counts in 0..10_000, seed per arm.
fn counts(seed: u64) -> Vec<Vec<u64>> {
    let mut state = seed;
    (0..LAYERS)
        .map(|layer| {
            (0..EXPERTS)
                .map(|expert| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) % 10_000 + u64::from((layer * 31 + expert) as u16)
                })
                .collect()
        })
        .collect()
}

/// One call's picks: `PICKS` raw values mapped to faults (-1) and in-range experts.
fn call_picks(seed: u64, call: usize) -> Vec<i32> {
    let mut state = seed ^ (call as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (0..PICKS)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let raw = (state >> 33) as i64 % 532 - 10;
            if raw < 0 {
                -1
            } else if raw > 511 {
                (raw - 512) as i32
            } else {
                raw as i32
            }
        })
        .collect()
}

fn policy(seed: u64) -> ReplaceOnMissLru {
    let geometry =
        BankGeometry::new(LAYERS, EXPERTS, TOP_K, SLOTS, RECORD).expect("a legal geometry");
    let profile = counts(seed);
    // Expert `i` in slot `i`: full, as `TierPlan::hot_set` tables are; the profile only sets
    // the initial victim order.
    let tables: Vec<Vec<i32>> = (0..LAYERS)
        .map(|_| {
            let mut row = vec![-1i32; EXPERTS];
            for (expert, slot) in row.iter_mut().enumerate().take(SLOTS) {
                *slot = expert as i32;
            }
            row
        })
        .collect();
    let rows: Vec<&[i32]> = tables.iter().map(|row| row.as_slice()).collect();
    ReplaceOnMissLru::from_fill(geometry, &rows, &profile, None).expect("a filled bank")
}

/// Renders one arm: `CALLS` calls on layer 0, then one `admit` on layer 1.
fn arm_text(seed: u64) -> String {
    let mut bank = policy(seed);
    let mut text = String::new();
    let (mut hits, mut moves) = (0usize, 0usize);
    for call in 0..CALLS {
        let picks = call_picks(seed, call);
        let plan = bank
            .plan_pass(0, &picks)
            .expect("in-range picks and faults");
        hits += plan.hits;
        moves += plan.moves.len();
        let _ = writeln!(
            text,
            "call {call} picks={picks:?} distinct={} hits={} moves={}",
            plan.distinct,
            plan.hits,
            plan.moves.len()
        );
        for (i, m) in plan.moves.iter().enumerate() {
            let _ = writeln!(
                text,
                "move {call} {i} slot={} in={} out={}",
                m.slot, m.incoming, m.outgoing
            );
            let _ = writeln!(
                text,
                "refill {call} {i} part=record off={} len={RECORD}",
                usize::from(m.slot) * RECORD
            );
        }
        let _ = writeln!(text, "table {call} layer=0 row={:?}", bank.slot_row(0));
    }
    let admit: Vec<u16> = (0..60u16).map(|i| (i % 50) * 5 + 1).collect();
    let plan = bank
        .admit(0, &admit)
        .expect("the admit contract on the replayed layer");
    let _ = writeln!(
        text,
        "admit layer=0 entries={} distinct={} hits={} moves={}",
        admit.len(),
        plan.distinct,
        plan.hits,
        plan.moves.len()
    );
    for m in &plan.moves {
        let _ = writeln!(
            text,
            "amove slot={} in={} out={}",
            m.slot, m.incoming, m.outgoing
        );
    }
    let _ = writeln!(text, "table admit layer=0 row={:?}", bank.slot_row(0));
    let _ = writeln!(text, "counters calls={CALLS} hits={hits} moves={moves}");
    text
}

/// Renders the error arm: a refused pick leaves the table untouched (plan_pass is atomic).
fn error_text(seed: u64) -> String {
    let mut bank = policy(seed);
    let mut text = String::new();
    for call in 0..4 {
        let picks = call_picks(seed, call);
        let plan = bank.plan_pass(0, &picks).expect("valid calls first");
        let _ = writeln!(
            text,
            "call {call} picks={picks:?} moves={}",
            plan.moves.len()
        );
    }
    let before = format!("{:?}", bank.slot_row(0));
    let bad = [7, 999, -1];
    let error = bank
        .plan_pass(0, &bad)
        .expect_err("pick 999 is outside the layer");
    let after = format!("{:?}", bank.slot_row(0));
    let _ = writeln!(text, "call 4 picks={bad:?} error={error:?}");
    let _ = writeln!(text, "table 4 unchanged={}", before == after);
    text
}

/// Renders one state arm: the same pick stream and admit as the fill arm, driven through
/// `ExpertBankState` (the host owner a `DeviceBank` wraps) in one host mode, recording the
/// copy operations the device would run — the demote write-back places included, which the
/// policy-level arms cannot see.
fn state_text(mode: HostMode, seed: u64) -> String {
    let geometry =
        BankGeometry::new(LAYERS, EXPERTS, TOP_K, SLOTS, RECORD).expect("a legal geometry");
    let mut state =
        ExpertBankState::with_policy(geometry, mode, policy(seed)).expect("a filled bank");
    let mut text = String::new();
    for call in 0..CALLS {
        let picks = call_picks(seed, call);
        let copies = state
            .begin_pass(0, &picks)
            .expect("in-range picks and faults");
        let _ = writeln!(
            text,
            "call {call} picks={picks:?} distinct={} hits={}",
            copies.distinct, copies.hits
        );
        for (i, op) in copies.ops.iter().enumerate() {
            let _ = writeln!(
                text,
                "op {call} {i} slot={} in={} src={} demote={:?}",
                op.slot, op.incoming, op.src_place, op.demote
            );
        }
    }
    let admit: Vec<u16> = (0..60u16).map(|i| (i % 50) * 5 + 1).collect();
    let copies = state.admit(0, &admit).expect("the admit contract");
    let _ = writeln!(
        text,
        "admit distinct={} hits={}",
        copies.distinct, copies.hits
    );
    for (i, op) in copies.ops.iter().enumerate() {
        let _ = writeln!(
            text,
            "aop {i} slot={} in={} src={} demote={:?}",
            op.slot, op.incoming, op.src_place, op.demote
        );
    }
    let stats = state.stats();
    let _ = writeln!(
        text,
        "counters passes={} picks={} hits={} misses={} demotions={}",
        stats.passes, stats.picks, stats.hits, stats.misses, stats.demotions
    );
    text
}

/// The digests of this family's arms.
pub(super) fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("fill", arm_text(0xfeed_0001)),
        ("error", error_text(0xfeed_0002)),
        (
            "state-exclusive",
            state_text(HostMode::Exclusive, 0xfeed_0003),
        ),
        (
            "state-inclusive",
            state_text(HostMode::Inclusive, 0xfeed_0004),
        ),
    ] {
        map.insert(name.into(), digest(&text).into());
    }
    map
}

/// Serializes `FIFI_GOLDEN_UPDATE` captures with the Q36 pick-golden test (same file, parallel
/// libtest threads — see its copy of this helper). Duplicated until the shared policy seam
/// gives both policies one home.
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
                "the pick-goldens update lock {} is held or stale; delete it if no capture runs",
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
fn the_pick_replay_holds_its_golden_digests() {
    const GOLDENS_TEXT: &str = include_str!("../../qwen36/cuda/testdata/pick-goldens.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/family/qwen36/cuda/testdata/pick-goldens.json");
    let sections = sections();
    if std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty()) {
        with_golden_lock(&path, || {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let mut goldens: serde_json::Value =
                serde_json::from_str(&text).expect("the committed pick goldens on disk");
            goldens["fn"] = serde_json::Value::Object(sections);
            let tmp = path.with_extension("json.tmp");
            std::fs::write(
                &tmp,
                serde_json::to_string_pretty(&goldens).expect("serialize"),
            )
            .expect("write the update beside the golden");
            std::fs::rename(&tmp, &path).expect("the atomic replace");
        });
        println!("UPDATED fn pick goldens in {}", path.display());
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed pick goldens");
    assert!(
        !pinned["fn"].is_null(),
        "no fn pick goldens; capture with FIFI_GOLDEN_UPDATE=1"
    );
    let mut drifted: Vec<String> = Vec::new();
    for (name, sha) in &sections {
        let pinned_sha = pinned["fn"][name].as_str().unwrap_or("");
        if pinned_sha != sha.as_str().unwrap_or("") {
            drifted.push(format!("{name}: pinned {pinned_sha:?}, computed {sha}"));
        }
    }
    assert!(
        drifted.is_empty(),
        "the Flash-Next bank policy moved; sections:\n{}",
        drifted.join("\n")
    );
    println!("GOLDEN MATCH fn picks");
}
