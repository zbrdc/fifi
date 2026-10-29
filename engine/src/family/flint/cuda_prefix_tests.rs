//! CUDA prefix-store qualification tests for the Flint CUDA executor (slice P1).
//!
//! Pure tests pin the fingerprint and the q8 refusal without weights or a
//! device; the retention-cap parsing tests live in tests.rs
//! (flint_prefix_retention_cap_math_admits_and_fails_closed). The ignored tests are the qualification
//! gates: they need the real loader and an exclusive 3090 window (the
//! FLINT_QUAL_MODEL / FLINT_QUAL_TOKENS environment of the other probes).
//!
//! The gates run the arms SEQUENTIALLY on ONE executor: the CUDA session
//! owns the device planes, so a second live executor would double the KV
//! charge. A reset between arms is the same quiesce the serving path sees
//! between requests, which is exactly the perturbation surface the gates
//! must cover.
//!
//! * GATE 1 (a): save transparency — same partition, with vs without a
//!   save: bit-identical pending-token logits and greedy tokens.
//! * GATE 1b (c): partition invariance — one sweep vs split at a
//!   non-multiple-of-8 internal boundary; measured either way
//!   (bit-identical ⇒ P1 output equals a cold one-sweep prefill; not ⇒
//!   partition-fixed determinism, a finding to report, not a save defect).
//! * GATE 2 (b): restore == cold prefill SPLIT AT THE SAME BOUNDARY —
//!   bit-identical logits rows and greedy tokens at 8K/16K/32K, N>=2.
//! * GATE 3: a rewritten tail refuses the restore (no wrong-prefix reuse)
//!   and the session then cold-prefills correctly.
//!
//! Three arms, in narrowing order (a)/(b)/(c): the Mac's save is
//! transparent and its divergence is the prefill partition.
//!
//! kv=f16 only: the q8 scale planes are dead code today, so both save and
//! restore refuse under kv_profile=q8 with a clear message.

use super::cuda_executor::{FlintExecutor, FlintSession};
use super::cuda_prefix::{CudaPrefixEnv, CudaPrefixPath};
use crate::family::{executor::ExecutorOps, session::SaveKind};
use std::path::Path;

/// The fingerprint binds every numeric-path input the CUDA executor reads:
/// context limit, prefill chunk size, kv profile, the FROZEN effective
/// alloc-cache and late-health selectors (effective values, never raw
/// env strings), and the model root. One change makes a stored prefix
/// unrestorable under the new setting.
#[test]
fn a_cuda_path_binds_every_flag_it_reads() {
    fn env(alloc_cache: bool, late_health: bool) -> CudaPrefixEnv {
        CudaPrefixEnv {
            alloc_cache,
            late_health,
            packed_pointwise: false,
            exl3_legacy: false,
            expert_cache: false,
        }
    }
    // One selector flip per arm, including the three bound by the respin
    // (packed pointwise, midrows legacy, expert cache): a snapshot taken
    // on one of those paths must not restore onto another.
    fn flip(env_value: &CudaPrefixEnv, field: &str) -> CudaPrefixEnv {
        let mut next = env_value.clone();
        match field {
            "packed_pointwise" => next.packed_pointwise = !next.packed_pointwise,
            "exl3_legacy" => next.exl3_legacy = !next.exl3_legacy,
            "expert_cache" => next.expert_cache = !next.expert_cache,
            _ => unreachable!("a bound selector"),
        }
        next
    }
    let off = env(false, false);
    for field in ["packed_pointwise", "exl3_legacy", "expert_cache"] {
        assert_ne!(
            CudaPrefixPath::from_values(Path::new("/model"), 4096, 8, "f16", off.clone()),
            CudaPrefixPath::from_values(Path::new("/model"), 4096, 8, "f16", flip(&off, field)),
            "the fingerprint must bind {field}"
        );
    }
    let root = Path::new("/model");
    let base = CudaPrefixPath::from_values(root, 4096, 8, "f16", env(false, false));
    assert_eq!(
        base,
        CudaPrefixPath::from_values(root, 4096, 8, "f16", env(false, false))
    );
    assert_ne!(
        base,
        CudaPrefixPath::from_values(root, 4097, 8, "f16", env(false, false))
    );
    assert_ne!(
        base,
        CudaPrefixPath::from_values(root, 4096, 4, "f16", env(false, false))
    );
    assert_ne!(
        base,
        CudaPrefixPath::from_values(root, 4096, 8, "q8", env(false, false))
    );
    assert_ne!(
        base,
        CudaPrefixPath::from_values(root, 4096, 8, "f16", env(true, false))
    );
    assert_ne!(
        base,
        CudaPrefixPath::from_values(root, 4096, 8, "f16", env(false, true))
    );
    assert_ne!(
        base,
        CudaPrefixPath::from_values(Path::new("/other-model"), 4096, 8, "f16", env(false, false))
    );
}

/// q8 refuses at the executor boundary with a message naming the limit, on
/// both save and restore, before any device work.
#[test]
fn q8_profile_refuses_prefix_reuse() {
    use crate::EngError;
    assert!(super::cuda_prefix::profile_allows_prefix("q8").is_err());
    assert!(super::cuda_prefix::profile_allows_prefix("f16").is_ok());
    let EngError::BadRequest(message) =
        super::cuda_prefix::profile_allows_prefix("q8").unwrap_err()
    else {
        panic!("the q8 refusal must be a BadRequest naming the profiles");
    };
    assert!(message.contains("q8"), "{message}");
    assert!(message.contains("f16"), "{message}");
}

/// GATE 1 (red-first): saving at the tip must not perturb the live session.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn save_then_continue_is_bit_identical_to_no_save() {
    let t0 = std::time::Instant::now();
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "f16");
    let prompt = qual_prompt(qual_fit(8192));

    // Arm A: prefill, save the tip, continue. The key is the FULL prompt:
    // the framework wants fed + 1 == key.len(), and fed counts the
    // committed rows (every token but the pending last).
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("A prefill");
    mark(&t0, "[COLD] g1a-A");
    let saved = ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip);
    assert!(
        matches!(saved, Ok(true)),
        "tip save must succeed: {saved:?}"
    );
    let pending_a = session.pending_token().expect("A pending token");
    let row_a = engine
        .qual_logits(&mut session, pending_a)
        .expect("A logits");
    let tokens_a = greedy_from_row(&mut engine, &mut session, &row_a, 64);

    // Arm B: reset, same prefill, no save.
    ExecutorOps::reset(&mut engine, &mut session).expect("B reset");
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("B prefill");
    mark(&t0, "[COLD] g1a-B");
    let pending_b = session.pending_token().expect("B pending token");
    let row_b = engine
        .qual_logits(&mut session, pending_b)
        .expect("B logits");
    let tokens_b = greedy_from_row(&mut engine, &mut session, &row_b, 64);

    assert_eq!(pending_a, pending_b);
    assert!(
        row_a
            .iter()
            .zip(&row_b)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "SAVE perturbed the logits"
    );
    assert_eq!(
        tokens_a, tokens_b,
        "SAVE perturbed the live session (bug class)"
    );
}

/// GATE 2: restore-vs-cold bitwise at 8K/16K/32K, N>=2 per size. One size
/// per test so an exclusive lease runs one size at a time (two sizes cannot
/// fit the 900 s lease cap).
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn restored_append_is_bit_identical_to_cold_8k() {
    restored_append_at(8192);
}

#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn restored_append_is_bit_identical_to_cold_16k() {
    restored_append_at(16384);
}

#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn restored_append_is_bit_identical_to_cold_32k() {
    restored_append_at(32768);
}

fn restored_append_at(size: u32) {
    let t0 = std::time::Instant::now();
    // FLINT_QUAL_REPEATS lets a lease run N=1 (two independent leases give
    // N=2 as independent processes when the 900 s cap cannot hold both).
    // Zero is configuration damage, not a valid N: a zero
    // repeat loop skips every assertion and would report a green gate.
    let repeats: u32 = std::env::var("FLINT_QUAL_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    assert!(
        repeats >= 1,
        "FLINT_QUAL_REPEATS must be >= 1, got {repeats}"
    );
    for _ in 0..repeats {
        let (mut engine, mut session) = FlintExecutor::open_qualification(size, "f16");
        let prompt = qual_prompt(qual_fit(size));
        let mut extended = prompt.clone();
        extended.extend_from_slice(&qual_suffix(64));

        // Arm A: prefill, save the tip, restore into the reset session,
        // prefill the remaining suffix.
        ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("A prefill");
        mark(&t0, &format!("[COLD] gate2-A size={size}"));
        assert!(matches!(
            ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
            Ok(true)
        ));
        ExecutorOps::reset(&mut engine, &mut session).expect("A arm reset");
        let reused =
            ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("restore");
        assert_eq!(reused, prompt.len(), "reused the saved key at size {size}");
        ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..])
            .expect("A suffix prefill");
        let pending_a = session.pending_token().expect("A pending token");
        let row_a = engine
            .qual_logits(&mut session, pending_a)
            .expect("A logits");
        let tokens_a = greedy_from_row(&mut engine, &mut session, &row_a, 32);

        // Arm B: reset, cold prefill SPLIT AT THE SAME BOUNDARY: the
        // restore arm's piece decomposition is
        // [prompt sweep][suffix rows], so the cold arm must reproduce the
        // same piece shapes — a one-sweep cold arm would conflate any
        // partition effect with restore correctness.
        ExecutorOps::reset(&mut engine, &mut session).expect("B arm reset");
        ExecutorOps::prefill(&mut engine, &mut session, &extended[..prompt.len()])
            .expect("B cold first part");
        mark(&t0, &format!("[COLD] gate2-B-part1 size={size}"));
        ExecutorOps::prefill(&mut engine, &mut session, &extended[prompt.len()..])
            .expect("B cold second part");
        mark(&t0, &format!("[COLD] gate2-B-part2 size={size}"));
        let pending_b = session.pending_token().expect("B pending token");
        let row_b = engine
            .qual_logits(&mut session, pending_b)
            .expect("B logits");
        let tokens_b = greedy_from_row(&mut engine, &mut session, &row_b, 32);

        assert_eq!(pending_a, pending_b);
        assert!(
            row_a
                .iter()
                .zip(&row_b)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "restore diverged at size {size}"
        );
        assert_eq!(tokens_a, tokens_b, "restore diverged at size {size}");

        // Arm C (— the aliasing property): after the session
        // has CONTINUED from the stored entry, the entry itself must be
        // unchanged. Reset, restore the SAME entry again, and the
        // boundary logits must equal arm A's first restored row bitwise.
        ExecutorOps::reset(&mut engine, &mut session).expect("C arm reset");
        let reused_again =
            ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("C restore");
        assert_eq!(
            reused_again,
            prompt.len(),
            "C reused the entry at size {size}"
        );
        ExecutorOps::prefill(&mut engine, &mut session, &extended[reused_again..])
            .expect("C suffix prefill");
        let pending_c = session.pending_token().expect("C pending token");
        let row_c = engine
            .qual_logits(&mut session, pending_c)
            .expect("C logits");
        assert_eq!(pending_a, pending_c);
        assert!(
            row_a
                .iter()
                .zip(&row_c)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the stored entry changed while a session continued from it (size {size})"
        );
    }
}

/// GATE 1b (partition invariance — the Mac finding that the save is
/// transparent there and the divergence is the PREFILL PARTITION): the same
/// prompt in one sweep vs split at an internal boundary that is NOT a
/// multiple of the 8-row prefill chunk. This gate MEASURES and RECORDS the
/// CUDA answer — a failure here is a partition-noninvariance FINDING to
/// report (P1 then ships 'deterministic under a fixed partition'
/// semantics), not a save/restore defect; if it passes, P1's output is
/// bit-equal to a cold one-sweep prefill, stronger than the Mac.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn split_prefill_partitions_are_bit_identical() {
    let t0 = std::time::Instant::now();
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "f16");
    let prompt = qual_prompt(qual_fit(8192));

    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("one-sweep prefill");
    mark(&t0, "[COLD] partition-sweep");
    let pending_one = session.pending_token().expect("pending token");
    let row_one = engine
        .qual_logits(&mut session, pending_one)
        .expect("logits");
    let tokens_one = greedy_from_row(&mut engine, &mut session, &row_one, 16);

    // The second part's row batches start at a shifted alignment.
    let split = prompt.len() / 2 + 3;
    ExecutorOps::reset(&mut engine, &mut session).expect("arm reset");
    ExecutorOps::prefill(&mut engine, &mut session, &prompt[..split]).expect("first part");
    ExecutorOps::prefill(&mut engine, &mut session, &prompt[split..]).expect("second part");
    mark(&t0, "[COLD] partition-split");
    let pending_two = session.pending_token().expect("pending token");
    let row_two = engine
        .qual_logits(&mut session, pending_two)
        .expect("logits");
    let tokens_two = greedy_from_row(&mut engine, &mut session, &row_two, 16);

    assert_eq!(pending_one, pending_two);
    assert!(
        row_one
            .iter()
            .zip(&row_two)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the split prefill partition changed the boundary logits"
    );
    assert_eq!(
        tokens_one, tokens_two,
        "the split partition changed the greedy stream"
    );
}

/// GATE 2-32K split-reference: the cold arm prefills the prompt, saves at
/// the boundary, then appends the 64-row suffix - the restore arm does
/// the same through the restore path, and the two must agree bitwise and
/// on the greedy walk. This is a SPLIT reference, not a one-sweep one:
/// a lease can hold only ONE full 32K prefill under the 900 s cap, and
/// no equivalence between the split and a single sweep is claimed here;
/// one-sweep invariance (a single cold prefill of the whole extended
/// prompt, no split) is proven separately by gate 1b at 8K and by
/// restored_append_vs_one_sweep_16k at 16K. The save must precede the
/// suffix append and the greedy walk:
/// the store binds the key to the HELD prompt (history/prompt_len/fed+1),
/// and the walk overwrites ring rows. Save transparency itself is already
/// proven by gates 1a/1b at 8K. Phase markers localize silent-phase hangs.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn restored_append_vs_split_reference_32k() {
    let t0 = std::time::Instant::now();
    let repeats: u32 = std::env::var("FLINT_QUAL_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    assert!(
        repeats >= 1,
        "FLINT_QUAL_REPEATS must be >= 1, got {repeats}"
    );
    for _ in 0..repeats {
        let (mut engine, mut session) = FlintExecutor::open_qualification(32768, "f16");
        let prompt = qual_prompt(qual_fit(32768));
        let suffix = qual_suffix(64);
        let mut extended = prompt.clone();
        extended.extend_from_slice(&suffix);

        // Cold reference: one sweep split at the save boundary.
        ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("prompt sweep");
        mark(&t0, "[COLD] 32k-sweep");
        mark(&t0, "[SAVE] begin");
        assert!(matches!(
            ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
            Ok(true)
        ));
        mark(&t0, "[SAVE] done");
        ExecutorOps::prefill(&mut engine, &mut session, &suffix).expect("cold suffix append");
        mark(&t0, "[APPEND] cold-done");
        let pending_cold = session.pending_token().expect("cold pending");
        let row_cold = engine
            .qual_logits(&mut session, pending_cold)
            .expect("cold logits");
        let tokens_cold = greedy_from_row(&mut engine, &mut session, &row_cold, 32);

        // Restore arm: restore the saved tip and replay the suffix.
        ExecutorOps::reset(&mut engine, &mut session).expect("arm reset");
        mark(&t0, "[RESTORE] begin");
        let reused =
            ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("restore");
        assert_eq!(reused, prompt.len(), "reused the saved key");
        mark(&t0, "[RESTORE] done");
        mark(&t0, "[APPEND] begin");
        ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..])
            .expect("suffix prefill");
        mark(&t0, "[APPEND] done");
        let pending_r = session.pending_token().expect("restore pending");
        let row_r = engine.qual_logits(&mut session, pending_r).expect("logits");
        let tokens_r = greedy_from_row(&mut engine, &mut session, &row_r, 32);
        assert_eq!(pending_cold, pending_r);
        assert!(
            row_cold
                .iter()
                .zip(&row_r)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "restore diverged from the split cold reference"
        );
        assert_eq!(tokens_cold, tokens_r);

        // Arm C: the entry is unchanged after a continuation.
        ExecutorOps::reset(&mut engine, &mut session).expect("C reset");
        let again =
            ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("C restore");
        assert_eq!(again, prompt.len());
        ExecutorOps::prefill(&mut engine, &mut session, &extended[again..]).expect("C suffix");
        let pending_c = session.pending_token().expect("C pending");
        let row_c = engine
            .qual_logits(&mut session, pending_c)
            .expect("C logits");
        assert_eq!(pending_r, pending_c);
        assert!(
            row_r
                .iter()
                .zip(&row_c)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the stored entry changed while a session continued from it"
        );
    }
}

/// GATE 2-16K one-sweep: the cold reference is a SINGLE prefill of the
/// whole extended prompt (no split), the strongest form of the reference.
/// Together with gate 1b at 8K this proves one-sweep invariance: the
/// 32K gate compares against a split reference to fit its lease, so the
/// one-sweep property is pinned here. Two 16K-class
/// prefills plus the restore fit one 900 s lease. Phase markers localize
/// silent-phase hangs.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader, prefix env SET"]
fn restored_append_vs_one_sweep_16k() {
    let t0 = std::time::Instant::now();
    let (mut engine, mut session) = FlintExecutor::open_qualification(16384, "f16");
    let prompt = qual_prompt(qual_fit(16384));
    let suffix = qual_suffix(64);
    let mut extended = prompt.clone();
    extended.extend_from_slice(&suffix);

    // Cold reference: ONE full sweep of the extended prompt.
    ExecutorOps::prefill(&mut engine, &mut session, &extended).expect("one-sweep cold prefill");
    mark(&t0, "[COLD] 16k-one-sweep");
    let pending_cold = session.pending_token().expect("cold pending");
    let row_cold = engine
        .qual_logits(&mut session, pending_cold)
        .expect("cold logits");
    let tokens_cold = greedy_from_row(&mut engine, &mut session, &row_cold, 32);

    // Save/restore arm: save the tip at the prompt boundary, restore,
    // replay only the suffix.
    ExecutorOps::reset(&mut engine, &mut session).expect("arm reset");
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("save prefill");
    mark(&t0, "[SAVE] begin");
    assert!(matches!(
        ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
        Ok(true)
    ));
    mark(&t0, "[SAVE] done");
    ExecutorOps::reset(&mut engine, &mut session).expect("pre-restore reset");
    mark(&t0, "[RESTORE] begin");
    let reused =
        ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("restore");
    assert_eq!(reused, prompt.len(), "reused the saved key");
    mark(&t0, "[RESTORE] done");
    ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..]).expect("suffix prefill");
    mark(&t0, "[APPEND] done");
    let pending_r = session.pending_token().expect("restore pending");
    let row_r = engine.qual_logits(&mut session, pending_r).expect("logits");
    let tokens_r = greedy_from_row(&mut engine, &mut session, &row_r, 32);
    assert_eq!(pending_cold, pending_r);
    assert!(
        row_cold
            .iter()
            .zip(&row_r)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "restore diverged from the independent one-sweep reference"
    );
    assert_eq!(tokens_cold, tokens_r, "the greedy walk must match");
    drop(session);
    drop(engine);
}

/// ENVELOPE gate (cache + pool under the 40 GiB guard, with the
/// reporting duty): save/restore at 32K under the lease's cache env;
/// asserts the device envelope stays under the card budget with headroom
/// and the snapshot's host charge fits the host/cgroup budget, and prints
/// the prefix_report entries/bytes plus the KV-peak delta for the record.
/// Runs in a lease that exports FIFI_FLINT_PREFIX_REUSE=1 (an armed f16
/// engine is the only one that publishes a report).
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader, prefix env SET"]
fn prefix_save_restore_envelope_under_guard() {
    let t0 = std::time::Instant::now();
    let (mut engine, mut session) = FlintExecutor::open_qualification(32768, "f16");
    let used = |engine: &FlintExecutor| {
        let (free, total) = engine.qual_memory_info();
        total - free
    };
    let (_, device_total) = engine.qual_memory_info();
    let before = used(&engine);
    let prompt = qual_prompt(qual_fit(32768));
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("prefill");
    mark(&t0, "[COLD] env-prefill");
    let mid = used(&engine);
    mark(&t0, "[SAVE] begin");
    assert!(matches!(
        ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
        Ok(true)
    ));
    mark(&t0, "[SAVE] done");
    let after_save = used(&engine);
    ExecutorOps::reset(&mut engine, &mut session).expect("reset");
    mark(&t0, "[RESTORE] begin");
    // An exact-key restore reuses 0 BY DESIGN: restore candidates are
    // entries whose key is STRICTLY SHORTER than the request and a prefix
    // of it (session.rs restore_prefix), so the envelope gate must extend
    // the prompt like its sibling gates before restoring.
    let mut extended = prompt.clone();
    extended.extend_from_slice(&qual_suffix(64));
    let reused =
        ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("restore");
    assert_eq!(reused, prompt.len());
    mark(&t0, "[RESTORE] done");
    ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..]).expect("suffix prefill");
    mark(&t0, "[APPEND] done");
    let after_restore = used(&engine);
    let report = ExecutorOps::prefix_report(&engine).expect("a prefix report");
    // Signed deltas: the unsigned subtraction could
    // panic when a later sample reads LOWER than an earlier one.
    let save_delta =
        i64::try_from(after_save).expect("sizes fit") - i64::try_from(mid).expect("sizes fit");
    let restore_delta = i64::try_from(after_restore).expect("sizes fit")
        - i64::try_from(after_save).expect("sizes fit");
    let charge = i64::try_from(report.bytes).expect("sizes fit");
    println!(
        "envelope record: used before={before} prefill={mid} after-save={after_save} after-restore={after_restore}; entries={} bytes={}; save delta {save_delta}, restore delta {restore_delta}",
        report.entries, report.bytes,
    );
    // The gate must be ABLE TO FAIL: the old 40 GiB
    // host constant could never fail on a 24 GiB card. The right budgets:
    // the DEVICE series stays under the card total with headroom, and the
    // snapshot's HOST charge (report.bytes — the figure the save admitted
    // against HostBudget) still fits that same host/cgroup budget.
    for (label, value) in [
        ("before", before),
        ("prefill", mid),
        ("after-save", after_save),
        ("after-restore", after_restore),
    ] {
        assert!(
            i64::try_from(value).expect("sizes fit")
                < i64::try_from(device_total).expect("sizes fit") - (64 << 20),
            "device envelope breach at {label}: {value} bytes against total {device_total}"
        );
    }
    // Save/restore pin HOST memory, not device: neither may grow device
    // use beyond allocator noise.
    assert!(
        save_delta <= 64 << 20,
        "save grew device use by {save_delta} bytes (snapshots pin host memory)"
    );
    assert!(
        restore_delta <= 64 << 20,
        "restore grew device use by {restore_delta} bytes (a restore refills existing planes)"
    );
    assert!(report.entries >= 1, "the saved tip must be reported");
    assert!(charge > 0, "the snapshot must pin a positive host charge");
    // Post-save FLOOR check only: the charge was admitted
    // BEFORE allocating (that is the admission the save answers to); this
    // snapshot is now LIVE, so re-admitting the full charge here would
    // demand headroom for a SECOND snapshot and can fail for a healthy
    // save. What must still hold after allocation is the floor: the OS
    // headroom the budget always requires (admit with charge 0 asserts
    // exactly os_floor <= available).
    let budget =
        crate::family::flint::exl3_resident::HostBudget::probe().expect("probe the host budget");
    budget
        .admit_for(0, "post-save floor")
        .expect("the host floor must still hold after the snapshot is live");
}

/// Default-OFF is bit-for-bit unobservable
/// WITHOUT FIFI_FLINT_PREFIX_REUSE the store answers a
/// closed (0, 0) limits pair and NO report — the request log carries no
/// prefix_hit field for any CUDA request. Runs in a lease that does NOT
/// export the env.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader, NO prefix env"]
fn default_off_leaves_the_store_unobservable() {
    let (engine, _session) = FlintExecutor::open_qualification(8192, "f16");
    assert_eq!(ExecutorOps::prefix_limits(&engine).roots, 0);
    assert_eq!(ExecutorOps::prefix_limits(&engine).entries, 0);
    assert!(
        ExecutorOps::prefix_report(&engine).is_none(),
        "default OFF must not publish a prefix report"
    );
}

/// The q8 negative arm (planned no-op by design):
/// WITH the opt-in env set and a non-f16 profile, the opt-in resolves to
/// OFF and the BOUNDARY REQUEST PATH SERVES COLD — save reports "not
/// saved" (Ok(false)), restore reuses 0 without an error, and the cold
/// prefill + logits read completes. Runs in a lease that DOES export
/// FIFI_FLINT_PREFIX_REUSE=1.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader, prefix env SET"]
fn q8_opt_in_resolves_off_and_serves_cold() {
    // The gate must prove it RAN with the env set:
    // without this assert a lease that forgot the env would pass on the
    // unarmed path twice over.
    assert!(
        crate::family::flint::prefix_reuse_opt_in(),
        "the q8-negative arm requires FIFI_FLINT_PREFIX_REUSE to be set"
    );
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "q8");
    assert_eq!(ExecutorOps::prefix_limits(&engine).roots, 0);
    assert_eq!(ExecutorOps::prefix_limits(&engine).entries, 0);
    assert!(ExecutorOps::prefix_report(&engine).is_none());
    let prompt = qual_prompt(qual_fit(8192));
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("cold q8 prefill");
    // The boundary request path under an unarmed engine: the save the
    // worker would plan reports not-saved, and the restore it would try
    // reuses 0 — BOTH without an error (the old code BadRequest here).
    assert!(
        matches!(
            ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
            Ok(false)
        ),
        "an unarmed save must report not-saved, not fail"
    );
    let mut extended = prompt.clone();
    extended.extend_from_slice(&qual_suffix(64));
    // Reset before the second request : the worker resets
    // before a restore (sched::worker::start); without it the session still
    // holds the 8,064-token history, the unarmed restore reuses 0, and the
    // cold replay appends 8,128 more = 16,192 > the 8,192 limit -> the
    // gate would die of PromptTooLong, not of anything it measures.
    ExecutorOps::reset(&mut engine, &mut session).expect("second-request reset");
    let reused = ExecutorOps::restore_prefix(&mut engine, &mut session, &extended)
        .expect("restore must not error");
    assert_eq!(
        reused, 0,
        "an unarmed restore must reuse nothing and serve cold"
    );
    ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..])
        .expect("cold replay of the whole request");
    let pending = session.pending_token().expect("cold pending token");
    let _ = engine
        .qual_logits(&mut session, pending)
        .expect("cold q8 logits");
}

/// A 1-token boundary key declines to cache and the request serves cold
/// : a single fed token leaves ZERO committed
/// rows, and the committed-boundary check used to fail the REQUEST; the
/// save now reports not-saved (Ok(false)) and the conversation continues
/// without error. Runs with the prefix env SET on a fresh 8192 f16 ring.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader, prefix env SET"]
fn one_token_boundary_declines_and_serves_cold() {
    // Arming preconditions: the gate must
    // prove it ran ARMED — env set AND the f16 store's one-entry shape —
    // or the unarmed Ok(false)/Ok(0) arms would pass it without ever
    // reaching the zero-committed branch it exists to test.
    assert!(
        crate::family::flint::prefix_reuse_opt_in(),
        "this gate requires FIFI_FLINT_PREFIX_REUSE to be set"
    );
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "f16");
    let limits = ExecutorOps::prefix_limits(&engine);
    assert_eq!(
        (limits.roots, limits.entries),
        (0, 1),
        "an armed f16 store answers the addendum-1 shape"
    );
    let refusals_before = ExecutorOps::prefix_report(&engine)
        .expect("an armed engine reports")
        .save_side
        .save_refusals;
    let prompt = qual_prompt(1);
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("one-token prefill");
    assert!(
        matches!(
            ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
            Ok(false)
        ),
        "a zero-committed key must decline to cache, not fail the request"
    );
    // The decline is counted executor-side and rides save_refusals with
    // an EXACT increment; the store saw nothing, so no
    // double-count is possible.
    assert_eq!(
        ExecutorOps::prefix_report(&engine)
            .expect("an armed engine reports")
            .save_side
            .save_refusals,
        refusals_before + 1,
        "the zero-committed decline must increment save_refusals exactly once"
    );
    let mut extended = prompt.clone();
    extended.extend_from_slice(&qual_suffix(64));
    ExecutorOps::reset(&mut engine, &mut session).expect("second-request reset");
    let reused =
        ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("restore");
    assert_eq!(reused, 0, "nothing was saved, so nothing can reuse");
    ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..])
        .expect("cold replay of the whole request");
    let pending = session.pending_token().expect("cold pending token");
    let _ = engine
        .qual_logits(&mut session, pending)
        .expect("cold logits");
}

/// NON-WRAPPED SLIDING RESTORE: every other gate
/// saves at 8064+ rows, so the committed-bounds copy path below the ring
/// size (committed << cache_rows) is never exercised. This gate saves a
/// ~300-token prompt on a fresh 8192 ring, DIRTIES the saved row range
/// with a 600-token sequence that differs at every position (so the
/// sliding slots wrap and a no-op restore cannot match), extends,
/// restores, replays the suffix, and compares boundary row + greedy walk
/// bitwise against a cold one-sweep arm of the same extended prompt.
/// Runs with the prefix env SET (the save must arm).
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader, prefix env SET"]
fn non_wrapped_short_prompt_restores_bit_identical() {
    let t0 = std::time::Instant::now();
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "f16");
    let prompt = qual_prompt(300);
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("short prefill");
    mark(&t0, "[COLD] slide-save");
    assert!(matches!(
        ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
        Ok(true)
    ));
    // Load-bearing restore: reset alone
    // leaves the saved rows in the rings, so a no-op restore would still
    // match bitwise — and tiling the SAME corpus would recompute row i
    // identically, which is exactly what the first dirty attempt did
    // (its first 300 tokens equaled the saved prompt). The dirty prompt is
    // therefore built from the LCG suffix stream, adjusted so it differs
    // from the saved prompt at EVERY position, and is >= 600 tokens so the
    // sliding slots wrap. A restore that uploads nothing leaves foreign
    // rows in [0, 300) — different token, same position, different KV —
    // and the comparison below FAILS. This is the mutation check: the
    // gate cannot pass on a no-op.
    let dirty = {
        let mut d = qual_suffix(600);
        for (i, slot) in d.iter_mut().enumerate().take(prompt.len()) {
            if *slot == prompt[i] {
                *slot = (*slot + 1) % 100_000;
            }
        }
        d
    };
    assert!(
        dirty.iter().zip(&prompt).all(|(d, p)| d != p),
        "precondition: the dirty prompt must differ at every saved position"
    );
    assert!(
        dirty.len() >= 600,
        "precondition: the dirty prompt wraps the sliding slots"
    );
    ExecutorOps::reset(&mut engine, &mut session).expect("dirty reset");
    ExecutorOps::prefill(&mut engine, &mut session, &dirty).expect("dirty prefill");
    mark(&t0, "[COLD] slide-dirty");
    // Observable proof the dirtying happened before the restore: prefill
    // feeds len-1 rows and keeps the last token pending (the rule the wrap
    // gate asserts), so fed == 599 proves all dirty rows landed.
    assert_eq!(
        session.fed(),
        dirty.len() - 1,
        "the dirty sweep must feed every row but the pending one"
    );
    let mut extended = prompt.clone();
    extended.extend_from_slice(&qual_suffix(64));
    // Restore arm.
    ExecutorOps::reset(&mut engine, &mut session).expect("arm reset");
    let reused =
        ExecutorOps::restore_prefix(&mut engine, &mut session, &extended).expect("restore");
    assert_eq!(reused, prompt.len(), "the short key must fully reuse");
    ExecutorOps::prefill(&mut engine, &mut session, &extended[reused..]).expect("suffix prefill");
    let pending_r = session.pending_token().expect("restore pending");
    let row_r = engine.qual_logits(&mut session, pending_r).expect("logits");
    let tokens_r = greedy_from_row(&mut engine, &mut session, &row_r, 16);
    // Cold one-sweep reference of the SAME extended prompt (its own reset
    // again leaves dirty rows beyond the sweep; attention only reads the
    // committed range, so the comparison stays exact).
    ExecutorOps::reset(&mut engine, &mut session).expect("cold reset");
    ExecutorOps::prefill(&mut engine, &mut session, &extended).expect("cold prefill");
    mark(&t0, "[COLD] slide-reference");
    let pending_c = session.pending_token().expect("cold pending");
    let row_c = engine.qual_logits(&mut session, pending_c).expect("logits");
    let tokens_c = greedy_from_row(&mut engine, &mut session, &row_c, 16);
    assert_eq!(pending_r, pending_c);
    assert!(
        row_r
            .iter()
            .zip(&row_c)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the non-wrapped boundary row must be bit-identical"
    );
    assert_eq!(tokens_r, tokens_c, "the non-wrapped greedy walk must match");
}

/// Shared replay: restore the saved tip, ASSERT the reused length equals
/// the saved key (a silent Ok(0) would degrade this into a cold full
/// prefill and both arms would pass), then prefill only the suffix,
/// returning the boundary pending token and its logits row.
fn restored_row(
    engine: &mut FlintExecutor,
    session: &mut FlintSession,
    extended: &[u32],
    expected_reused: usize,
) -> (u32, Vec<f32>) {
    let reused = ExecutorOps::restore_prefix(engine, session, extended).expect("restore");
    assert_eq!(
        reused, expected_reused,
        "restore must reuse the saved key, not fall back to a cold prefill"
    );
    ExecutorOps::prefill(engine, session, &extended[reused..]).expect("suffix prefill");
    let pending = session.pending_token().expect("pending token");
    let row = engine.qual_logits(session, pending).expect("logits row");
    (pending, row)
}

/// RING-STATE belt-and-braces: CUDA restores can only enter a quiesced
/// ring by construction (restore_committed refuses a non-reset cursor;
/// reset_after_quiescence clears the ring), so the possible ring states
/// collapse to one here. This gate closes
/// it with evidence: the same save/restore replay in an engine whose ring
/// had ALREADY reached the tip in an earlier conversation (then reset)
/// must match a fresh engine's replay bitwise. The wrap arm pays one extra
/// cold prefill so the two engines have genuinely different histories.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn restore_after_a_wrapped_ring_matches_a_fresh_engine() {
    let t0 = std::time::Instant::now();
    let prompt = qual_prompt(qual_fit(8192));
    let mut extended = prompt.clone();
    extended.extend_from_slice(&qual_suffix(64));

    // Wrapped engine: an earlier conversation reached the cap (ring at
    // tip), was reset, and only then the save/restore cycle runs. The wrap
    // arm prefills limit-1 rows explicitly (qual_fit stops 128 short) and
    // PROVES the tip was reached twice: limit-1 tokens feed limit-2 rows
    // (the last token stays pending), and the pending-token logits read
    // commits the tip row - so fed must be 8190 after prefill and 8191
    // after the read. A silent truncation or a not-quite-wrapped arm fails
    // the gate instead of degrading it into a duplicate-fresh comparison.
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "f16");
    let wrap_prompt = qual_prompt(8191);
    ExecutorOps::prefill(&mut engine, &mut session, &wrap_prompt).expect("wrap prefill to the cap");
    assert_eq!(
        session.fed(),
        wrap_prompt.len() - 1,
        "limit-1 tokens feed limit-2 rows while the last stays pending"
    );
    let wrap_pending = session.pending_token().expect("wrap pending token");
    let _ = engine
        .qual_logits(&mut session, wrap_pending)
        .expect("wrap logits read");
    assert_eq!(session.fed(), 8191, "the logits read commits the tip row");
    println!(
        "[WRAP] ring fed rows: after-prefill {} after-logits {} (tip = 8191)",
        wrap_prompt.len() - 1,
        session.fed()
    );
    mark(&t0, "[COLD] ring-wrap");
    ExecutorOps::reset(&mut engine, &mut session).expect("post-wrap reset");
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("save prefill");
    mark(&t0, "[COLD] ring-save");
    assert!(matches!(
        ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
        Ok(true)
    ));
    ExecutorOps::reset(&mut engine, &mut session).expect("pre-restore reset");
    let (pending_w, row_w) = restored_row(&mut engine, &mut session, &extended, prompt.len());

    // Fresh engine: identical replay, no prior wrap history. Two 21.5 GiB
    // weight payloads cannot coexist on the 24 GiB device, so release the
    // wrapped engine first; the captured (pending, row) values carry the
    // comparison and each engine exercised its own store.
    drop(session);
    drop(engine);
    let (mut engine2, mut session2) = FlintExecutor::open_qualification(8192, "f16");
    ExecutorOps::prefill(&mut engine2, &mut session2, &prompt).expect("fresh save prefill");
    mark(&t0, "[COLD] ring-fresh");
    assert!(matches!(
        ExecutorOps::save_prefix(&mut engine2, &session2, &prompt, SaveKind::Tip),
        Ok(true)
    ));
    ExecutorOps::reset(&mut engine2, &mut session2).expect("fresh pre-restore reset");
    let (pending_f, row_f) = restored_row(&mut engine2, &mut session2, &extended, prompt.len());

    assert_eq!(pending_w, pending_f);
    assert!(
        row_w
            .iter()
            .zip(&row_f)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a previously wrapped ring changed the restore"
    );
}

/// GATE 3: a rewritten tail refuses the restore (no wrong-prefix reuse) and
/// the session then cold-prefills correctly.
#[test]
#[ignore = "qualification gate: exclusive 3090 window, real loader"]
fn rewritten_tail_refuses_and_recovers() {
    let t0 = std::time::Instant::now();
    let (mut engine, mut session) = FlintExecutor::open_qualification(8192, "f16");
    let prompt = qual_prompt(qual_fit(8192));
    ExecutorOps::prefill(&mut engine, &mut session, &prompt).expect("prefill");
    mark(&t0, "[COLD] g3-save");
    assert!(matches!(
        ExecutorOps::save_prefix(&mut engine, &session, &prompt, SaveKind::Tip),
        Ok(true)
    ));
    let mut diverged = prompt.clone();
    // In-vocab divergent tail: the divergence only needs a different valid
    // id (restore must refuse on any tail mismatch); an out-of-vocab
    // sentinel like u32::MAX - 1 can never complete the recovered-logits read
    // because the FFN embedding lookup rejects it.
    let tail = diverged.pop().expect("non-empty qualification prompt");
    let alt = if tail == 0 { 1 } else { tail - 1 };
    diverged.push(alt);
    // LONGER than the saved key with the divergence inside the key region
    // A same-length rewritten prompt is refused by
    // the strictly-shorter candidate rule before any token compare, so the
    // gate only proved the length rule. The extension admits the lookup;
    // the token compare is what refuses.
    diverged.extend_from_slice(&qual_suffix(64));
    ExecutorOps::reset(&mut engine, &mut session).expect("arm reset");
    let reused =
        ExecutorOps::restore_prefix(&mut engine, &mut session, &diverged).expect("restore lookup");
    assert_eq!(reused, 0, "a diverged tail must not reuse");
    // Recovery: the session cold-prefills the diverged conversation whole,
    // and the recovered row/walk must be BITWISE the cold reference of the
    // same diverged prompt (a recovery that only
    // reads logits proves nothing about its correctness).
    ExecutorOps::prefill(&mut engine, &mut session, &diverged).expect("cold recovery prefill");
    mark(&t0, "[COLD] g3-recover");
    let pending_r = session.pending_token().expect("recovered pending token");
    let row_r = engine
        .qual_logits(&mut session, pending_r)
        .expect("recovered logits");
    let tokens_r = greedy_from_row(&mut engine, &mut session, &row_r, 16);
    ExecutorOps::reset(&mut engine, &mut session).expect("reference reset");
    ExecutorOps::prefill(&mut engine, &mut session, &diverged).expect("reference prefill");
    mark(&t0, "[COLD] g3-reference");
    let pending_c = session.pending_token().expect("reference pending token");
    let row_c = engine
        .qual_logits(&mut session, pending_c)
        .expect("reference logits");
    let tokens_c = greedy_from_row(&mut engine, &mut session, &row_c, 16);
    assert_eq!(pending_r, pending_c);
    assert!(
        row_r
            .iter()
            .zip(&row_c)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the recovered boundary row must be bit-identical to cold"
    );
    assert_eq!(tokens_r, tokens_c, "the recovered walk must match cold");
}

/// The FLINT_QUAL_TOKENS corpus (teacher-forced IDs), tiled the way the
/// other qualification probes tile it, or an empty marker when the
/// environment is absent so the gate refuses with a clear panic.
/// The largest prompt that fits a context limit with room for the 64-token
/// suffix arms and the boundary row: the executor refuses prompt_tokens >
/// limit_tokens - 1 (PromptTooLong, seen live in L1 attempt 2). The logits
/// read also refuses a full context (ContextLimit, seen live in L1a
/// attempt 1 at limit-65): leave 64 suffix + 32 greedy rows + boundary.
fn qual_fit(limit: u32) -> usize {
    limit as usize - 128
}

/// Wall-clock marker for the lease controller: with --nocapture the
/// controller streams these lines, computes per-cold/per-arm walls, and
/// projects the remaining lease budget for the early-abort rule.
fn mark(t0: &std::time::Instant, label: &str) {
    println!("[ARM] {label} +{:.1}s", t0.elapsed().as_secs());
}

fn qual_prompt(size: usize) -> Vec<u32> {
    let ids: Vec<u32> = std::env::var("FLINT_QUAL_TOKENS")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    assert!(
        !ids.is_empty(),
        "FLINT_QUAL_TOKENS must name a token corpus"
    );
    (0..).flat_map(|_| ids.iter().copied()).take(size).collect()
}

/// `count` greedy tokens continuing from `row`'s argmax. Single-feed (the
/// CUDA twin of the Mac fix): the boundary token was fed exactly once by the
/// caller's logits read — the walk seeds from that row and feeds each
/// sampled token once, instead of re-reading a stale pending token and
/// feeding the boundary twice.
fn greedy_from_row(
    engine: &mut FlintExecutor,
    session: &mut FlintSession,
    row: &[f32],
    count: usize,
) -> Vec<u32> {
    let mut next = argmax(row);
    let mut out = Vec::with_capacity(count);
    for step in 0..count {
        if step > 0 {
            let row = engine.qual_logits(session, next).expect("a logits row");
            next = argmax(&row);
        }
        out.push(next);
    }
    out
}

fn argmax(row: &[f32]) -> u32 {
    let (best, _) = row
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
            if *v > bv { (i, *v) } else { (bi, bv) }
        });
    u32::try_from(best).expect("vocab fits u32")
}

fn qual_suffix(n: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(n);
    let mut x = 0x5eed_1234u32;
    for _ in 0..n {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(x % 100_000);
    }
    v
}
