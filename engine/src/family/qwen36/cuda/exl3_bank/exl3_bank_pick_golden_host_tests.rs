//! The pick-replay golden of the Q36 bank policy: `LayerSlots` (LRU, and LRU with the sticky
//! decayed-hot set) replayed over deterministic pick streams, recording per call the picks,
//! the returned slots, the miss list, the gated pick set (the picks whose expert refilled this
//! call), the event flag a `resolve_split` would raise, the refill plan (three parts, part
//! offset and length per miss, in miss order), the whole slot table after every call, and the
//! counters. The error arm pins the RECORDED DEFECT: `assign` is not atomic (a mid-stream
//! `Err` leaves `tick` advanced and earlier assignments standing), and a failed refill write in
//! `resolve_mapped` leaves the table ahead of the device bytes — the fix is a later change
//! of its own, and this golden is what proves it unchanged until then and moved after.
//!
//! Host-only, modelless, GPUless, but compiled under `--features native-cuda` in the
//! container (the qwen36 cuda module is native-cuda-gated): name this test in every
//! native-cuda test battery. `FIFI_GOLDEN_UPDATE=1` rewrites this family's golden file
//! `testdata/pick-goldens.json` atomically with sorted keys (same mechanism and switch as
//! the prefill golden harness); without it the compile-time pin is compared, so a drifted
//! checkout cannot pass a compare it should fail. The Flash-Next pick golden
//! (`flash_next/cuda/bank_pick_golden_host_tests.rs`) owns its own file, same format.

use super::{DECAY_CALLS, EMPTY, LayerSlots, LayerTables, NO_SLOT, refill_plan};
use crate::family::expert_bank::SlotTable;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

const EXPERTS: usize = 512;
const SLOTS: usize = 48;
/// The three part lengths of the 3.05 pack's record (gate/up/down), 1,881,600 bytes
/// together. Mirrors `parts_strides` (the derivation `resolve_mapped` feeds `refill_plan`)
/// for that pack; the refill lines below pin the strides, offsets and experts against it.
const PART_LEN: [usize; 3] = [626_864, 626_864, 626_872];
const CALLS: usize = 200;
const PICKS: usize = 8;

/// An xorshift stream: `CALLS` calls of `PICKS` distinct experts each, deterministic per seed.
fn call_picks(seed: u64, call: usize) -> Vec<usize> {
    let mut state = seed ^ (call as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut picks = Vec::with_capacity(PICKS);
    while picks.len() < PICKS {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let expert = (state >> 33) as usize % EXPERTS;
        if !picks.contains(&expert) {
            picks.push(expert);
        }
    }
    picks
}

fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Renders one arm: the replay of `CALLS` calls through a fresh table. The refill lines
/// render the engine's own plan (`refill_plan`, the function `resolve_mapped` issues its
/// copies from), not a reimplementation: the format matches what an earlier version of
/// this harness recomputed, so a plan that agrees leaves the pinned digests untouched.
fn arm_text(slots: usize, sticky: usize, seed: u64) -> String {
    let mut table = LayerSlots::new(EXPERTS, slots, sticky);
    let mut text = String::new();
    let (mut hits, mut misses) = (0usize, 0usize);
    for call in 0..CALLS {
        let picks = call_picks(seed, call);
        let (slots, miss_slots) = table.assign(&picks).expect("in-range distinct picks");
        let gated: Vec<usize> = (0..picks.len())
            .filter(|&p| miss_slots.contains(&slots[p]))
            .collect();
        hits += slots.len() - gated.len();
        misses += miss_slots.len();
        let _ = writeln!(
            text,
            "call {call} picks={picks:?} slots={slots:?} misses={miss_slots:?} \
             gated={gated:?} event={}",
            !miss_slots.is_empty()
        );
        for (i, ops) in refill_plan(&table, &miss_slots, PART_LEN)
            .chunks(3)
            .enumerate()
        {
            for op in ops {
                let _ = writeln!(
                    text,
                    "refill {call} {i} part={} expert={} off={} len={}",
                    op.part,
                    op.expert,
                    op.offset(),
                    op.record
                );
            }
        }
        // The row is already the seam's shape (`NO_SLOT = -1`): the golden and the
        // shared `SlotTable::slot_row` read the same array.
        let row: Vec<i32> = table.slot_of.clone();
        let _ = writeln!(text, "table {call} tick={} row={row:?}", table.tick);
    }
    let decays = table.tick / DECAY_CALLS;
    // The seam's two directions agree after the replay: a slot holds an expert exactly
    // when the row points back at that slot, and the id is the one the row names.
    let seam_tables = LayerTables(vec![table]);
    let row = seam_tables.slot_row(0);
    for slot in 0..SLOTS {
        match seam_tables.expert_in_slot(0, slot) {
            Some(expert) => assert_eq!(
                row[usize::from(expert)],
                slot as i32,
                "slot {slot} holds {expert} but the row disagrees"
            ),
            None => assert!(
                !row.contains(&(slot as i32)),
                "slot {slot} holds no expert but the row points to it"
            ),
        }
    }
    let _ = writeln!(
        text,
        "counters calls={CALLS} hits={hits} misses={misses} decays={decays}"
    );
    text
}

/// Renders the error arm: a mid-stream out-of-range pick, and the atomicity the fix pins —
/// the failed call leaves the table untouched (tick and row as the last good call left
/// them, asserted against a snapshot), so the follow-up re-assignment of the pick the
/// failed call would have taken is a fresh miss, not a hit on a half-applied table. The
/// refill order, the gated set, the event flag and the counters are recomputed here from
/// the policy's returns, not read from the engine.
fn error_text(seed: u64) -> String {
    let mut table = LayerSlots::new(EXPERTS, SLOTS, 8);
    let mut text = String::new();
    let mut resident: Vec<usize> = Vec::new();
    for call in 0..4 {
        let picks = call_picks(seed, call);
        resident.extend_from_slice(&picks);
        let (slots, _) = table.assign(&picks).expect("valid calls first");
        let _ = writeln!(text, "call {call} picks={picks:?} slots={slots:?}");
    }
    let bad_expert = (0..EXPERTS)
        .find(|&e| !resident.contains(&e))
        .expect("512 experts against at most 32 assigned leaves an absent one");
    let bad = vec![bad_expert, EXPERTS, bad_expert];
    let before = table.clone();
    let error = table
        .assign(&bad)
        .expect_err("an out-of-range pick exceeds the layer");
    assert_eq!(table, before, "the refused call changes nothing");
    let _ = writeln!(text, "call 4 picks={bad:?} error={error:?}",);
    let follow = table
        .assign(&[bad_expert])
        .expect("the table still serves after a refusal");
    let _ = writeln!(
        text,
        "call 5 picks=[{bad_expert}] slots={:?} misses={:?} refilled={}",
        follow.0,
        follow.1,
        !follow.1.is_empty()
    );
    let row: Vec<i32> = table.slot_of.clone();
    let _ = writeln!(text, "table 5 tick={} row={row:?}", table.tick);
    text
}

/// Renders the torn-refill response arm. The failing device write itself needs the device
/// bank and is not host-drivable, so this drives the response alone: a call whose misses
/// would have refilled, then [`LayerSlots::invalidate`] over exactly those missed slots,
/// then the same picks again — every pick that had missed re-misses, and the refill of
/// that second call names the same experts the torn one would have. What is pinned is the
/// invalidation function and its round trip, not the write failure's trigger.
fn invalidate_text(seed: u64) -> String {
    let mut table = LayerSlots::new(EXPERTS, SLOTS, 8);
    let mut text = String::new();
    for call in 0..4 {
        let picks = call_picks(seed, call);
        let (slots, _) = table.assign(&picks).expect("valid calls first");
        let _ = writeln!(text, "call {call} picks={picks:?} slots={slots:?}");
    }
    let picks = call_picks(seed, 4);
    let (_, misses) = table.assign(&picks).expect("a miss-heavy call");
    let _ = writeln!(text, "call 4 picks={picks:?} misses={misses:?}");
    // The incoming expert of each missed slot, before invalidate drops it.
    let incoming: Vec<usize> = misses.iter().map(|&slot| table.expert_of(slot)).collect();
    table.invalidate(&misses);
    for (&slot, &expert) in misses.iter().zip(&incoming) {
        assert_eq!(table.expert_of[slot], EMPTY, "slot {slot} reads absent");
        assert_eq!(
            table.slot_of[expert], NO_SLOT,
            "expert {expert} reads absent"
        );
    }
    let row: Vec<i32> = table.slot_of.clone();
    let _ = writeln!(text, "invalidated tick={} row={row:?}", table.tick);
    let (_, again) = table.assign(&picks).expect("the same picks serve again");
    assert_eq!(
        again.len(),
        misses.len(),
        "every pick that had missed re-misses"
    );
    let _ = writeln!(
        text,
        "call 5 picks={picks:?} misses={again:?} refill_again={}",
        again.len()
    );
    text
}

/// Renders the duplicates-and-capacity arm: a stream with picks repeated inside one call,
/// then a call with more distinct experts than the table holds slots, recording whatever
/// the policy answers — repeats are pinned as they land, and the over-capacity refusal as
/// the engine states it.
fn dupes_text(slots: usize, seed: u64) -> String {
    let mut table = LayerSlots::new(EXPERTS, slots, 8);
    let mut text = String::new();
    for call in 0..8 {
        let base = call_picks(seed, call);
        let mut picks = base.clone();
        picks.extend_from_slice(&base[..4]);
        match table.assign(&picks) {
            Ok((assigned, miss_slots)) => {
                let _ = writeln!(
                    text,
                    "call {call} picks={picks:?} slots={assigned:?} misses={miss_slots:?}"
                );
            }
            Err(error) => {
                let _ = writeln!(text, "call {call} picks={picks:?} error={error:?}");
            }
        }
    }
    let over: Vec<usize> = (0..slots + 1).collect();
    match table.assign(&over) {
        Ok((assigned, miss_slots)) => {
            let _ = writeln!(
                text,
                "over picks={over:?} slots={assigned:?} misses={miss_slots:?}"
            );
        }
        Err(error) => {
            let _ = writeln!(text, "over picks={} distinct error={error:?}", over.len());
        }
    }
    text
}

/// The digests of this family's arms.
pub(super) fn sections() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, text) in [
        ("lru", arm_text(SLOTS, 0, 0x5eed_0001)),
        ("sticky", arm_text(SLOTS, 8, 0x5eed_0002)),
        ("error", error_text(0x5eed_0003)),
        ("lru32", arm_text(32, 0, 0x5eed_0004)),
        ("sticky3", arm_text(SLOTS, 3, 0x5eed_0005)),
        ("dupes", dupes_text(SLOTS, 0x5eed_0006)),
        ("invalidate", invalidate_text(0x5eed_0007)),
    ] {
        map.insert(name.into(), digest(&text).into());
    }
    map
}

/// Serializes `FIFI_GOLDEN_UPDATE` captures: the arms are written as a sorted-key map so the
/// file is byte-stable whatever crate features order `serde_json`, and the read-merge-write
/// of a capture runs under a `create_new` guard beside the golden (atomic everywhere we
/// run). A capture crashed mid-update leaves the guard behind, and the next capture panics
/// after a minute instead of racing — delete the stale file then. This family's golden file
/// is ours alone; the Flash-Next family writes its own file.
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
    const GOLDENS_TEXT: &str = include_str!("../testdata/pick-goldens.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/family/qwen36/cuda/testdata/pick-goldens.json");
    let sections = sections();
    if std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty()) {
        with_golden_lock(&path, || {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let mut goldens: serde_json::Value =
                serde_json::from_str(&text).expect("the committed pick goldens on disk");
            let mut arms: std::collections::BTreeMap<&str, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (name, sha) in &sections {
                arms.insert(name.as_str(), sha.clone());
            }
            goldens["arms"] = serde_json::Value::Object(
                arms.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            );
            let tmp = path.with_extension("json.tmp");
            let mut text = serde_json::to_string_pretty(&goldens).expect("serialize");
            text.push('\n');
            std::fs::write(&tmp, text).expect("write the update beside the golden");
            std::fs::rename(&tmp, &path).expect("the atomic replace");
        });
        println!("UPDATED q36 pick goldens in {}", path.display());
        return;
    }
    let pinned: serde_json::Value =
        serde_json::from_str(GOLDENS_TEXT).expect("the committed pick goldens");
    let pinned_arms = pinned["arms"]
        .as_object()
        .unwrap_or_else(|| panic!("no q36 pick arms; capture with FIFI_GOLDEN_UPDATE=1"));
    let mut drifted: Vec<String> = Vec::new();
    for (name, sha) in &sections {
        let pinned_sha = pinned_arms.get(name).and_then(|v| v.as_str()).unwrap_or("");
        if pinned_sha != sha.as_str().unwrap_or("") {
            drifted.push(format!("{name}: pinned {pinned_sha:?}, computed {sha}"));
        }
    }
    for name in pinned_arms.keys() {
        if !sections.contains_key(name) {
            drifted.push(format!("{name}: pinned but no longer computed — stale arm"));
        }
    }
    assert!(
        drifted.is_empty(),
        "the Q36 bank policy moved; sections:\n{}",
        drifted.join("\n")
    );
    println!("GOLDEN MATCH q36 picks");
}
