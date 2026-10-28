//! Prefix-store qualification tests for the Flint MLX executor (slice 1).
//!
//! The pure tests pin the fingerprint and the store shape without weights.
//! The ignored tests are the qualification gates on the real loader path:
//! restore identity (bitwise boundary logits + identical greedy tokens), the
//! appended turn, the diverged tail, and the side request. They require an
//! explicit Mac GPU allocation and run with the FLINT_QUAL_* environment of
//! the other qualification probes. IDENTITY GATES RUN TWICE in the
//! qualification window, once per EXPLICIT ring arm (RING_VIEW=1 and
//! RING_VIEW=0, both set on the command line; Sol wave-1 SF4: the arms
//! must be explicit because the DEFAULT moved - before the ring-flip
//! merge unset resolved OFF (two disabled arms), since it unset resolves
//! ON; either way an unset process collapses the two arms into one, so
//! the gates ASSERT the explicit arm below — the gates ASSERT the explicit arm below): the flag is
//! process-cached, so the two arms are two separate test-bin processes,
//! exactly like t1_f32_routing.

use super::factory;
use super::mlx_executor::{FlintExecutor, FlintPath, FlintSession};
use crate::family::{executor::ExecutorOps, session::SaveKind};
use std::path::Path;

/// The fingerprint is exactly the numeric-path surface: one EFFECTIVE
/// flag change makes a stored prefix unrestorable under the new setting.
/// The bind is by frozen effective value (Opus r1 SF-1), so raw-string
/// variants that resolve to the same flag ("1" vs "true" vs "on") are the
/// same path BY CONSTRUCTION - the type is bool/usize, not OsString.
#[test]
fn a_path_binds_every_flag_it_reads() {
    use crate::family::flint::mlx_executor::FlintEnv;
    // Pure construction with explicit values: the crate denies unsafe
    // (edition 2024 makes set_var unsafe), so the fingerprint contract is
    // tested without touching the process env; real-env coverage stays
    // with the per-arm test-bin processes of the qualification window.
    let root = Path::new("/model");
    let env = |batch_prefill,
               decode_attn,
               ring_view,
               prefill_chain,
               compile_layer,
               t1_flash,
               async_chain,
               coalesce_residual| FlintEnv {
        batch_prefill,
        decode_attn,
        ring_view,
        prefill_chain,
        compile_layer,
        t1_flash,
        async_chain,
        coalesce_residual,
    };
    let bare = || env(1024, true, false, false, false, false, false, false);
    let base = FlintPath::from_env_values(root, 32_768, bare());
    let same = FlintPath::from_env_values(root, 32_768, bare());
    assert_eq!(base, same, "same values = same path");
    // Every input moves the fingerprint, one at a time, by name.
    for (name, moved) in [
        (
            "batch_prefill",
            FlintPath::from_env_values(
                root,
                32_768,
                env(2048, true, false, false, false, false, false, false),
            ),
        ),
        (
            "decode_attn",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, false, false, false, false, false, false, false),
            ),
        ),
        (
            "ring_view",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, true, true, false, false, false, false, false),
            ),
        ),
        (
            "prefill_chain",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, true, false, true, false, false, false, false),
            ),
        ),
        (
            "compile_layer",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, true, false, false, true, false, false, false),
            ),
        ),
        (
            "t1_flash",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, true, false, false, false, true, false, false),
            ),
        ),
        (
            "async_chain",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, true, false, false, false, false, true, false),
            ),
        ),
        (
            "coalesce_residual",
            FlintPath::from_env_values(
                root,
                32_768,
                env(1024, true, false, false, false, false, false, true),
            ),
        ),
        ("limit", FlintPath::from_env_values(root, 16_384, bare())),
        (
            "model root",
            FlintPath::from_env_values(Path::new("/other-model"), 32_768, bare()),
        ),
    ] {
        assert_ne!(
            base, moved,
            "{name} must move the fingerprint so a snapshot refuses restore"
        );
    }
}

/// Slice-1 store shape: the addendum 1 limits (one tip entry, no roots), so
/// the worker plans exactly one tip save per job. Structural: the serving
/// path stays off unless the config flag AND the FIFI_FLINT_PREFIX_REUSE
/// opt-in both arm it (run.rs keeps the Flint family exclusion).
#[test]
fn the_store_is_one_tip_entry_no_roots() {
    assert_eq!(
        super::mlx_executor::PREFIX_STORE_LIMITS,
        crate::family::session::StoreLimits {
            roots: 0,
            entries: 1
        },
        "the executor's pinned shape is zero roots and one tip entry"
    );
}

/// The serving piece size the alignment assert needs, from the same
/// resolver the executor uses (pure: no engine, no model load).
fn serving_piece_rows() -> usize {
    crate::family::flint::factory::prefill_chunk_rows(crate::family::flint::factory::os_setting(
        std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref(),
    ))
}

/// Opens the real loader with the tokenizer's own EOS set and vocabulary.
/// Every real gate goes through here, so this is the single choke point
/// for the run's two environment facts: the serving opt-in must be armed
/// (Sol wave-1 SF3: without this a gate run that forgot
/// FIFI_FLINT_PREFIX_REUSE=1 would pass while testing the raw ops only),
/// and the ring arm must be an EXPLICIT enable/disable word whose
/// effective state is asserted (Sol wave-1 SF4: an unset run rides the
/// DEFAULT — OFF before the ring-flip merge, ON since — so it would
/// silently collapse the two documented arms into whichever the default
/// happens to be).
fn open_real(root: &Path) -> (FlintExecutor, FlintSession) {
    assert!(
        super::prefix_reuse_opt_in(),
        "the qualification gates run with FIFI_FLINT_PREFIX_REUSE=1 exported"
    );
    assert_explicit_ring_arm();
    let bundle = factory::load_tokenizer(root).expect("load the real tokenizer");
    let eos = bundle.eos_token_ids().to_vec();
    let vocab = bundle.vocab_size() as usize;
    FlintExecutor::open(root, eos, vocab).expect("open the real loader")
}

/// The ring arm this gate process runs under, asserted against the
/// effective resolvers (Sol wave-2 SF-4: the wave-1 check accepted
/// arbitrary non-enable strings as an OFF arm, and an ON arm with
/// ASYNC_CHAIN=0, where the ring route never runs). Both envs must be the
/// literal "0" or "1"; an ON ring arm requires the chain ON, and both
/// effective flags must match their env. Returns the arm.
fn assert_explicit_ring_arm() -> bool {
    let ring = std::env::var("FIFI_FLINT_MLX_RING_VIEW")
        .expect("the gates run under an explicit RING_VIEW arm");
    let chain = std::env::var("FIFI_FLINT_MLX_ASYNC_CHAIN")
        .expect("the gates run under an explicit ASYNC_CHAIN arm");
    assert!(
        (ring == "0" || ring == "1") && (chain == "0" || chain == "1"),
        "both arms are the literal 0 or 1 (got RING_VIEW={ring:?} ASYNC_CHAIN={chain:?})"
    );
    assert_eq!(
        super::mlx_forward::ring_view_flagged(),
        ring == "1",
        "the effective ring view must match the explicit arm"
    );
    assert_eq!(
        super::mlx_forward::async_chain_flagged(),
        chain == "1",
        "the effective async chain must match the explicit arm"
    );
    assert!(
        ring == "0" || chain == "1",
        "a ring-ON arm with the chain off is inert: the route would never run"
    );
    ring == "1"
}

/// Feeds a prompt the way the server does: in `prefill_piece`-sized pieces.
/// The chunk cache plans 1..=1024 rows per prefill call, so one 8192-token
/// call would be rejected ("Flint chunk carries 1..=1024 rows") even though
/// the served path never makes one.
fn feed(engine: &mut FlintExecutor, session: &mut FlintSession, tokens: &[u32]) {
    let piece = engine.prefill_piece().unwrap_or(1024).max(1);
    for chunk in tokens.chunks(piece) {
        engine.prefill(session, chunk).expect("prefill piece");
    }
}

/// The frozen FLINT_QUAL_TOKENS corpus (teacher-forced IDs), tiled the way
/// the qualification probes tile it (the frozen file is 2049 IDs; the
/// identity gates need 4200+ tokens to cross the first growth step), or an
/// empty marker when the environment is absent so the caller can refuse.
fn corpus() -> Vec<u32> {
    let ids: Vec<u32> = std::env::var("FLINT_QUAL_TOKENS")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    if ids.is_empty() {
        return ids;
    }
    (0..).flat_map(|_| ids.iter().copied()).take(8448).collect()
}

/// Gate: save at the tip, reset, restore, and the boundary row is
/// BITWISE-equal to a cold full prefill of the same conversation, and the
/// 32 greedy tokens after the boundary are identical. The corpus spans
/// 4096+ tokens, so ring planes have wrapped many times past the first
/// 513-row window and the boundary row sits across the first 4096-row
/// full-attention growth step.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn a_restored_prefix_is_bitwise_identical_to_a_cold_prefill() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(
        tokens.len() >= 8256,
        "the identity gate needs 8192 saved tokens plus a longer restore prompt"
    );
    let prompt = &tokens[..8192];

    let (mut warm, mut warm_session) = open_real(&root);
    let (mut cold, mut cold_session) = open_real(&root);

    feed(&mut cold, &mut cold_session, prompt); // cold prefill
    let (cold_row, cold_draws) = row_and_greedy(&mut cold, &mut cold_session, 32);

    feed(&mut warm, &mut warm_session, prompt); // warm prefill
    assert!(
        warm.save_prefix(&warm_session, prompt, SaveKind::Tip)
            .expect("save at the tip"),
        "the one-entry store accepts the tip save"
    );
    warm.reset(&mut warm_session).expect("reset");
    // The exact-key contract only offers candidates strictly shorter than
    // the prompt; restoring the saved conversation means asking for it PLUS
    // an appended turn, exactly like turn 2 of a served conversation.
    let covered = warm
        .restore_prefix(&mut warm_session, &tokens)
        .expect("restore the saved tip");
    assert_eq!(covered, prompt.len(), "the tip covers the saved key");
    let (warm_row, warm_draws) = row_and_greedy(&mut warm, &mut warm_session, 32);

    assert_eq!(
        cold_row, warm_row,
        "restored boundary logits must equal the cold prefill bitwise"
    );
    assert_eq!(
        cold_draws, warm_draws,
        "restored continuation must draw the same 32 greedy tokens"
    );
}

/// Gate 2, the normal agent case: restore the tip, prefill ONLY the
/// appended tokens, and the greedy stream equals a cold full prefill of the
/// whole longer conversation.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn an_appended_turn_restores_the_tip_and_prefills_only_the_new_tokens() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(tokens.len() >= 4400, "needs head + appended room");
    let (head, tail) = tokens.split_at(4096);
    let piece = serving_piece_rows();
    assert_eq!(
        4096 % piece,
        0,
        "gate 2's cold-vs-appended identity needs the head piece-aligned ({piece})"
    );

    let (mut warm, mut warm_session) = open_real(&root);
    let (mut cold, mut cold_session) = open_real(&root);

    feed(&mut warm, &mut warm_session, head); // head prefill
    assert!(
        warm.save_prefix(&warm_session, head, SaveKind::Tip)
            .expect("tip save")
    );
    warm.reset(&mut warm_session).expect("reset");
    // Restoring the tip means asking for the whole longer conversation; the
    // store covers head.len() and only the appended tokens are prefilled.
    let covered = warm
        .restore_prefix(&mut warm_session, &tokens)
        .expect("restore head");
    assert_eq!(covered, head.len());
    feed(&mut warm, &mut warm_session, tail); // prefill only the appended tokens

    feed(&mut cold, &mut cold_session, &tokens); // cold full prefill
    assert_eq!(
        logits_row(&mut warm, &mut warm_session),
        logits_row(&mut cold, &mut cold_session),
        "the appended turn must land on the cold boundary bitwise"
    );
}

/// Gate 3, diverged tail: a request that departs from the stored prompt
/// BEFORE the tip must never serve the snapshot. The store's candidate walk
/// only offers entries whose key is a prefix of the new prompt, so the
/// diverged head covers zero tokens and the request recomputes - and still
/// succeeds.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn a_diverged_tail_never_serves_the_tip_and_recomputes() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(tokens.len() >= 4096);
    let (head, _) = tokens.split_at(4096);

    let (mut engine, mut session) = open_real(&root);
    feed(&mut engine, &mut session, head); // head prefill
    assert!(
        engine
            .save_prefix(&session, head, SaveKind::Tip)
            .expect("tip save")
    );

    let mut diverged = head.to_vec();
    diverged[8] = diverged[8].wrapping_add(1);
    // Append a tail so the prompt is longer than the stored key: the walk
    // must refuse on divergence, not on the strictly-shorter rule.
    diverged.extend_from_slice(&tokens[4096..4160]);
    engine.reset(&mut session).expect("reset");
    let covered = engine
        .restore_prefix(&mut session, &diverged)
        .expect("the diverged lookup must refuse, not error");
    assert_eq!(
        covered, 0,
        "a divergence before the tip must cover nothing: full recompute"
    );
    // The diverged request still succeeds via ordinary piecewise prefill.
    feed(&mut engine, &mut session, &diverged);
}

/// Gate 4, side request: a stranger prompt between two turns of the first
/// conversation covers zero and never disturbs the stored tip.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn a_side_request_covers_zero_and_keeps_the_tip_intact() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(tokens.len() >= 4160);
    let (head, tail) = tokens.split_at(4096);

    let (mut engine, mut session) = open_real(&root);
    feed(&mut engine, &mut session, head); // head prefill
    assert!(
        engine
            .save_prefix(&session, head, SaveKind::Tip)
            .expect("tip save")
    );
    // The boundary row the tip must reproduce later, bitwise (Opus
    // wave-2 Nit: "keeps the tip intact" was never compared).
    let (boundary_row, _) = row_and_greedy(&mut engine, &mut session, 1);

    let mut stranger: Vec<u32> = head.iter().rev().copied().collect();
    stranger.extend_from_slice(&tail[..64]);
    engine.reset(&mut session).expect("reset");
    let covered = engine
        .restore_prefix(&mut session, &stranger)
        .expect("stranger lookup");
    assert_eq!(covered, 0, "a stranger conversation never mixes states");
    feed(&mut engine, &mut session, &stranger); // stranger prefill

    engine.reset(&mut session).expect("reset");
    let recovered = engine
        .restore_prefix(&mut session, &tokens)
        .expect("the first conversation's tip survives the side request");
    assert_eq!(recovered, head.len());
    let (recovered_row, _) = row_and_greedy(&mut engine, &mut session, 1);
    assert_eq!(
        boundary_row, recovered_row,
        "the tip changed while a stranger request ran on the same executor"
    );
    feed(&mut engine, &mut session, tail); // append after restore
}

/// Gate 5 (run-3ab finding, variants): a tip save must not perturb the
/// LIVE session. One plain engine and two variant engines prefill the same
/// 2048-token prompt; V1 saves through the prefix store (snapshot KEPT
/// ALIVE, refcount > 1 during decode, exactly the served ON arm); V3 takes
/// the same snapshot and DROPS it immediately (refcount back to 1, no
/// store). 32 decode steps compare full logits rows as raw f32 bits plus
/// the greedy draw. If V1 diverges and V3 does not, the live refcount
/// (kv_append fresh allocation instead of buffer donation) is the cause;
/// if V3 also diverges, the capture itself perturbs. The first divergent
/// step, word, and bit are reported. Run in an EXPLICIT ring arm (the A/B
/// server arms that diverged had the ring view default ON).
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn a_tip_save_then_continue_variants() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(tokens.len() >= 2048, "needs a 2048-token prompt");
    let prompt = &tokens[..2048];

    use crate::family::session::TokenStepper;
    let draw = |row: &[f32], vocab: usize| {
        row.iter()
            .enumerate()
            .take(vocab)
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if *v > bv { (i, *v) } else { (bi, bv) }
            })
            .0
    };
    let first_divergence = |name: &str, step: usize, a: &[u8], b: &[u8]| -> String {
        for (wi, (wa, wb)) in a.iter().zip(b).enumerate() {
            if wa != wb {
                let word = wi / 4;
                let bit = (wa ^ wb).trailing_zeros();
                return format!(
                    "{name} diverges at step {step}: word {wi} (f32 #{word}), bit {bit}"
                );
            }
        }
        format!("{name} diverges at step {step}: length differs")
    };
    let bits =
        |r: &[f32]| -> Vec<u8> { r.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect() };
    let row_of =
        |engine: &mut FlintExecutor, session: &mut FlintSession, pending: u32| -> Vec<f32> {
            let mut stepper = engine.stepper(session);
            stepper.logits(pending).expect("a logits row")
        };
    let vocab_of;
    // ---- phase 1: plain vs V1 (store save, snapshot kept alive) ----
    // At most two native adapter leases per process: phase 1 holds plain+V1,
    // then frees both before phase 2 opens V3.
    let mut plain_rows: Vec<Vec<f32>> = Vec::with_capacity(32);
    {
        let (mut plain, mut plain_session) = open_real(&root);
        let (mut v1, mut v1_session) = open_real(&root);
        feed(&mut plain, &mut plain_session, prompt);
        feed(&mut v1, &mut v1_session, prompt);
        assert!(
            v1.save_prefix(&v1_session, prompt, SaveKind::Tip)
                .expect("tip save")
        );
        vocab_of = engine_vocab(&plain);
        let mut plain_pending = plain_session
            .pending_token()
            .expect("plain pending boundary token");
        let mut v1_pending = v1_session
            .pending_token()
            .expect("v1 pending boundary token");
        assert_eq!(plain_pending, v1_pending, "same boundary token");
        for step in 0..32 {
            let plain_row = row_of(&mut plain, &mut plain_session, plain_pending);
            let v1_row = row_of(&mut v1, &mut v1_session, v1_pending);
            let (plain_bits, v1_bits) = (bits(&plain_row), bits(&v1_row));
            if plain_bits != v1_bits {
                panic!(
                    "{}",
                    first_divergence("V1(store, kept)", step, &plain_bits, &v1_bits)
                );
            }
            let best = draw(&plain_row, vocab_of);
            assert_eq!(
                best,
                draw(&v1_row, vocab_of),
                "V1 draw diverges at step {step}"
            );
            plain_pending = u32::try_from(best).expect("vocab fits u32");
            v1_pending = plain_pending;
            plain_rows.push(plain_row);
        }
        drop(plain);
        drop(plain_session);
        drop(v1);
        drop(v1_session);
    }
    println!("V1-OK: store-save continuation bitwise-identical for 32 steps");

    // ---- phase 2: V3 (share-only capture, dropped immediately) vs the
    // recorded plain rows. In-process cross-engine determinism is proven by
    // the identity gates (restored row == cold row, two engines), so the
    // recording is a valid plain reference. ----
    {
        let (mut v3, mut v3_session) = open_real(&root);
        feed(&mut v3, &mut v3_session, prompt);
        drop(v3_session.snapshot_prefix_now());
        let mut v3_pending = v3_session
            .pending_token()
            .expect("v3 pending boundary token");
        for (step, plain_row) in plain_rows.iter().enumerate() {
            let v3_row = row_of(&mut v3, &mut v3_session, v3_pending);
            let (plain_bits, v3_bits) = (bits(plain_row), bits(&v3_row));
            if plain_bits != v3_bits {
                panic!(
                    "{}",
                    first_divergence("V3(share, dropped)", step, &plain_bits, &v3_bits)
                );
            }
            let best = draw(plain_row, vocab_of);
            assert_eq!(
                best,
                draw(&v3_row, vocab_of),
                "V3 draw diverges at step {step}"
            );
            v3_pending = u32::try_from(best).expect("vocab fits u32");
        }
    }
    println!("VARIANTS-OK: V1 and V3 both bitwise-identical to plain for 32 steps");
}

/// Gate 6 (H3, run-3ab t1 divergence; owner ruling 2026-10-28): the served ON arm splits turn-1
/// prefill at the internal tip boundary (chat.rs `with_prefix_boundary`);
/// the OFF arm sweeps once. If a boundary-split prefill yields different
/// numerics than a sweep of the same ids, the A/B turn-1 text divergence is
/// the chunk partition, not a save defect. V2 (split with a tip save) vs V3
/// (split without the save) separates the partition effect from any save
/// perturbation. One engine live at a time (adapter lease cap).
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn a_save_is_transparent_and_partition_is_numerics() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(tokens.len() >= 2048, "needs a 2048-token prompt");
    let prompt = &tokens[..2048];
    // An interior split near the end mirrors the served shape: the derived
    // tip sits just before the short generation suffix, so the tail piece
    // is small. b is piece-interior (not a multiple of 1024) on purpose -
    // the served tail is never piece-aligned either.
    let b = 1984;
    let (head, tail) = prompt.split_at(b);

    use crate::family::session::TokenStepper;
    let draw = |row: &[f32], vocab: usize| {
        row.iter()
            .enumerate()
            .take(vocab)
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if *v > bv { (i, *v) } else { (bi, bv) }
            })
            .0
    };
    let bits =
        |r: &[f32]| -> Vec<u8> { r.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect() };

    // V1: one sweep. Its final row and 8 draws are the reference, built
    // with the shared single-feed helper (Opus delta SF1: the hand-rolled
    // loop re-fed the pending boundary token, so sweep_draws[0] came from
    // a different sequence than v2_draws[0] for a test-shape reason).
    let (sweep_bits, sweep_draws, vocab) = {
        let (mut e, mut sess) = open_real(&root);
        feed(&mut e, &mut sess, prompt);
        let vocab = engine_vocab(&e);
        let (bits, draws) = row_and_greedy(&mut e, &mut sess, 8);
        (bits, draws, vocab)
    };

    let run_split = |with_save: bool, name: &str| -> (Vec<f32>, Vec<u32>) {
        let (mut e, mut sess) = open_real(&root);
        feed(&mut e, &mut sess, head);
        if with_save {
            assert!(
                e.save_prefix(&sess, head, SaveKind::Tip)
                    .expect("tip save at the boundary"),
                "the store accepts the boundary tip"
            );
        }
        feed(&mut e, &mut sess, tail);
        let pending = sess.pending_token().expect("split pending token");
        let mut stepper = e.stepper(&mut sess);
        let row = stepper.logits(pending).expect("split final row");
        // Single feed (Nit-2): the boundary token is fed ONCE, above; each
        // drawn token is fed exactly once below.
        let mut draws = Vec::with_capacity(8);
        let mut feed = u32::try_from(draw(&row, vocab)).expect("vocab fits u32");
        draws.push(feed);
        for _ in 1..8 {
            let row = stepper.logits(feed).expect("split draw row");
            feed = u32::try_from(draw(&row, vocab)).expect("vocab fits u32");
            draws.push(feed);
        }
        let _ = name;
        (row, draws)
    };

    let (v2_row, v2_draws) = run_split(true, "V2");
    let (v3_row, v3_draws) = run_split(false, "V3");

    let (v2_bits, v3_bits) = (bits(&v2_row), bits(&v3_row));
    let first_diff = |a: &[u8], b_: &[u8]| -> Option<(usize, u32)> {
        a.iter()
            .zip(b_)
            .enumerate()
            .find_map(|(i, (x, y))| (x != y).then(|| (i / 4, u32::from(x ^ y))))
    };
    println!(
        "H3 sweep-vs-split: v2_diff={:?} v3_diff={:?} v2_vs_v3={:?}",
        first_diff(&sweep_bits, &v2_bits),
        first_diff(&sweep_bits, &v3_bits),
        first_diff(&v2_bits, &v3_bits)
    );
    println!("H3 draws: sweep={sweep_draws:?} v2={v2_draws:?} v3={v3_draws:?}");
    // The partition difference vs a sweep is measured physics, not a defect:
    // the served ON arm splits at the tip boundary, OFF sweeps once. The
    // merge-blocking question is whether the SAVE adds any perturbation of
    // its own beyond the partition - the split-with-save row must be BITWISE
    // the split-without-save row, draws included.
    assert_eq!(
        v2_bits, v3_bits,
        "save perturbs the split prefill: V2(split+save) != V3(split)"
    );
    assert_eq!(
        v2_draws, v3_draws,
        "save perturbs the split draws: V2 != V3"
    );
}

/// Gate 7 (Opus r1 SF-2): a stored entry is still BIT-EXACT after a
/// session has continued from it. The snapshot shares planes through Rc,
/// so a continuation's `kv_append` must copy (never donate) the shared
/// planes; this gate re-reads the stored tip AFTER the live session has
/// decoded past it and compares the restored row to the row the live
/// session had at the boundary, bitwise.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate"]
fn a_stored_entry_survives_a_continued_session() {
    let _serial = crate::native_test_lock();
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(tokens.len() >= 4400, "needs head + continuation room");
    let prompt = &tokens[..4096];

    let (mut e, mut sess) = open_real(&root);
    feed(&mut e, &mut sess, prompt);
    assert!(
        e.save_prefix(&sess, prompt, SaveKind::Tip)
            .expect("tip save at the boundary")
    );
    // The row the live session has at the boundary, from ONE feeding of
    // the pending token (Nit-2 shape).
    let (boundary_row, _draws) = row_and_greedy(&mut e, &mut sess, 1);
    // The live session now CONTINUES from the save: more prefilled rows
    // force kv_append (and, past the saved planes' capacity, growth) on
    // state that the store also references. A DELIBERATE MUTATION PROBE,
    // not a served shape (Opus delta Nit-6): row_and_greedy bypasses
    // DecodeSession, so the next feed re-feeds prompt[4095] at position
    // 4096 — the point is to mutate the shared planes maximally, then
    // prove the stored entry did not move.
    feed(&mut e, &mut sess, &tokens[4096..4104]);
    let _ = row_and_greedy(&mut e, &mut sess, 1);

    // Re-read the stored tip on a fresh session of the same executor.
    e.reset(&mut sess).expect("reset");
    let covered = e
        .restore_prefix(&mut sess, &tokens)
        .expect("restore the continued-from tip");
    assert_eq!(covered, prompt.len(), "the tip still covers its key");
    let (restored_row, _) = row_and_greedy(&mut e, &mut sess, 1);
    assert_eq!(
        boundary_row, restored_row,
        "the stored entry changed while a session continued from it"
    );
}

/// Opus wave-3 MF-1, the wiring proof (the only test that exercises the
/// seam through open()): with the reuse opt-in unset, open() loads the
/// real model UNCHANGED - a garbage budget env that would abort an armed
/// engine is never even parsed, and the admitted cap is 0, so every save
/// refuses. Run in the qualification window with FIFI_FLINT_PREFIX_REUSE
/// unset (the one gate that must NOT set it) and the budget env garbage.
#[test]
#[ignore = "explicit Mac GPU allocation; real loader; run via the window gate with the reuse env unset and a garbage budget env"]
fn default_off_open_is_inert_to_a_garbage_budget_env() {
    let _serial = crate::native_test_lock();
    assert!(
        std::env::var_os("FIFI_FLINT_PREFIX_REUSE").is_none(),
        "this gate runs with FIFI_FLINT_PREFIX_REUSE unset"
    );
    assert_eq!(
        std::env::var("FIFI_FLINT_PREFIX_BUDGET_BYTES").as_deref(),
        Ok("garbage-1G"),
        "this gate runs with the budget env pinned to garbage"
    );
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let bundle = factory::load_tokenizer(&root).expect("load the real tokenizer");
    let (engine, _session) = FlintExecutor::open(
        &root,
        bundle.eos_token_ids().to_vec(),
        bundle.vocab_size() as usize,
    )
    .expect("default-OFF open ignores the garbage budget env");
    assert_eq!(engine.prefix_retention_cap, 0, "no cap derivation ran");
}

/// Sol wave-1 MF1: a retention budget that cannot hold the snapshot
/// turns the save into a PLANNED refusal — `Ok(false)` before the store
/// is touched, counted in `save_refusals` — and the session continues
/// correctly with an empty store. Run in the qualification window with
/// FIFI_FLINT_PREFIX_BUDGET_BYTES=1 beside the usual FLINT_QUAL_* env.
#[test]
#[ignore = "explicit tiny Mac GPU allocation; real loader; run via the window gate with FIFI_FLINT_PREFIX_BUDGET_BYTES=1"]
fn a_too_small_budget_refuses_the_save_planned() {
    let _serial = crate::native_test_lock();
    assert_eq!(
        std::env::var("FIFI_FLINT_PREFIX_BUDGET_BYTES").as_deref(),
        Ok("1"),
        "this gate runs with the retention budget pinned to one byte"
    );
    let root = std::env::var_os("FLINT_QUAL_MODEL")
        .map_or_else(|| Path::new("/not-set").to_path_buf(), Into::into);
    let tokens = corpus();
    assert!(
        tokens.len() >= 4400,
        "needs a save prompt plus continuation"
    );
    let prompt = &tokens[..4096];

    let (mut e, mut sess) = open_real(&root);
    feed(&mut e, &mut sess, prompt);
    assert!(
        !e.save_prefix(&sess, prompt, SaveKind::Tip)
            .expect("the budget refusal is a planned Ok(false), not an error"),
        "a one-byte budget cannot retain the snapshot, so the save must refuse"
    );
    let report = e.prefix_report().expect("the Flint prefix report");
    assert!(
        report.save_side.save_refusals >= 1,
        "the refusal rides save_refusals"
    );
    assert_eq!(report.entries, 0, "nothing was stored");

    // The refusal must not corrupt the live session (Sol wave-2 SF-3 /
    // Opus wave-2 SF-2): the continued row is compared BITWISE against a
    // plain engine fed the same tokens[..4104] - same partition, same
    // position, no store - after a reset, never against a different token
    // history.
    feed(&mut e, &mut sess, &tokens[4096..4104]);
    let (continued, _) = row_and_greedy(&mut e, &mut sess, 1);
    e.reset(&mut sess).expect("reset before the cold reference");
    let covered = e
        .restore_prefix(&mut sess, &tokens)
        .expect("restore on the empty store");
    assert_eq!(covered, 0, "the empty store covers nothing");
    feed(&mut e, &mut sess, &tokens[..4104]);
    let (cold, _) = row_and_greedy(&mut e, &mut sess, 1);
    assert_eq!(
        continued, cold,
        "a refused save must leave the session bitwise identical to a plain prefill"
    );
}

/// The boundary logits row after the fed prompt, as raw f32 bits.
fn logits_row(engine: &mut FlintExecutor, session: &mut FlintSession) -> Vec<u8> {
    use crate::family::session::TokenStepper;
    let pending = session
        .pending_token()
        .expect("a pending boundary token after prefill");
    let mut step = engine.stepper(session);
    step.logits(pending)
        .expect("the boundary logits row")
        .iter()
        .flat_map(|f| f.to_bits().to_le_bytes())
        .collect()
}

/// The boundary row AND `count` greedy tokens from ONE feeding of the
/// pending token (Opus r1 Nit-2: the old `logits_row` + `greedy` pair fed
/// the boundary token twice - once for the row, once for the first draw.
/// Same-numerics comparisons stayed valid, but this is the served
/// continuation: feed the boundary token, take its row, then feed each
/// drawn token exactly once).
fn row_and_greedy(
    engine: &mut FlintExecutor,
    session: &mut FlintSession,
    count: usize,
) -> (Vec<u8>, Vec<u32>) {
    use crate::family::session::TokenStepper;
    let vocab = engine_vocab(engine);
    let pending = session
        .pending_token()
        .expect("a pending boundary token after prefill");
    let mut step = engine.stepper(session);
    let row = step
        .logits(pending)
        .expect("the boundary logits row")
        .to_vec();
    let bits = row.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .take(vocab)
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if *v > bv { (i, *v) } else { (bi, bv) }
            })
            .0
    };
    let mut out = Vec::with_capacity(count);
    let mut feed = u32::try_from(argmax(&row)).expect("vocab fits u32");
    out.push(feed);
    for _ in 1..count {
        let row = step
            .logits(feed)
            .expect("a drawn-token logits row")
            .to_vec();
        feed = u32::try_from(argmax(&row)).expect("vocab fits u32");
        out.push(feed);
    }
    (bits, out)
}

fn engine_vocab(engine: &FlintExecutor) -> usize {
    // The executor's own vocabulary, not a hardcoded constant (Opus
    // wave-2 Nit): `open` validates vocab against the tokenizer's.
    engine.vocab
}
