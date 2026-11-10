//! The chunk driver's device gate: one chunk through BOTH cards vs the
//! per-token path over the same tokens. The chunk path runs MMQ GEMMs
//! where the per-token path runs native matvecs, so the gate is the
//! greedy token and a top-2 margin bound (the single-device chunk suite's
//! own standard for the same comparison), not bit equality.

use super::card_prefill::{
    CardChunkInputs, build_card_chunk, build_card_chunk_with_stage_rows, run_card_chunk,
};
use super::card_session_device_tests::rig_spec;
use super::dims::{KvMode, StepDims, StepOptions};
use super::layer_table::LayerTable;
use crate::EngError;
use crate::family::executor::ExecutorOps;
use crate::family::gguf_model::Inventory;
use crate::family::qwen36::host::HostShape;
use crate::family::session::TokenStepper;
use crate::kernels::cuda::device_present;

/// One full chunk pass and one per-token pass over the same rows: the
/// greedy winner must agree and the top-2 margin stay inside the slack
/// the single-device chunk suite uses for MMQ-vs-matvec.
#[test]
#[ignore = "holds its CUDA adapter leases for the process lifetime; run alone"]
fn card_chunk_matches_the_per_token_path() {
    if !device_present("card_chunk_matches_the_per_token_path") {
        return;
    }
    check_card_chunk([0, 0], 8, None);
}

#[test]
#[ignore = "requires two physical peer-enabled GPUs and holds their adapter leases; run alone"]
fn card_chunk_matches_the_per_token_path_on_physical_peers() {
    assert!(device_present(
        "card_chunk_matches_the_per_token_path_on_physical_peers"
    ));
    // The device runner must expose only the intended peer pair, in the
    // desired order, via CUDA_VISIBLE_DEVICES; these are logical ordinals.
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    check_card_chunk(devices, 8, None);
}

#[test]
#[ignore = "requires two physical peer-enabled GPUs and holds their adapter leases; run alone"]
fn large_card_chunk_matches_the_per_token_path_on_physical_peers() {
    assert!(device_present(
        "large_card_chunk_matches_the_per_token_path_on_physical_peers"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    check_card_chunk(devices, 96, None);
}

#[test]
#[ignore = "requires two physical peer-enabled GPUs and holds their adapter leases; run alone"]
fn max_card_chunk_matches_the_per_token_path_on_physical_peers() {
    assert!(device_present(
        "max_card_chunk_matches_the_per_token_path_on_physical_peers"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    check_card_chunk(devices, 512, None);
}

#[test]
#[ignore = "requires two physical peer-enabled GPUs and holds their adapter leases; run alone"]
fn wide_card_chunk_matches_the_per_token_path_on_physical_peers() {
    assert!(device_present(
        "wide_card_chunk_matches_the_per_token_path_on_physical_peers"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    check_card_chunk(devices, 1024, None);
}

#[test]
#[ignore = "requires two physical peer-enabled GPUs and holds their adapter leases; run alone"]
fn wider_card_chunk_matches_the_per_token_path_on_physical_peers() {
    assert!(device_present(
        "wider_card_chunk_matches_the_per_token_path_on_physical_peers"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    check_card_chunk(devices, 2048, None);
}

#[test]
#[ignore = "requires two physical peer-enabled GPUs and holds their adapter leases; run alone"]
fn segmented_card_chunk_matches_the_per_token_path_on_physical_peers() {
    assert!(device_present(
        "segmented_card_chunk_matches_the_per_token_path_on_physical_peers"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    check_card_chunk(devices, 96, Some(32));
}

fn check_card_chunk(devices: [usize; 2], rows: usize, stage_rows: Option<usize>) {
    const NAME: &str = "card_chunk_matches_the_per_token_path";
    // The serving-hook parity check below consumes two complete chunks and a
    // final token. Keep the original small fixtures unchanged, but give the
    // larger fixtures enough KV capacity for that entire sequence.
    let capacity = if rows > 1024 {
        8192
    } else if rows > 512 {
        4096
    } else if rows > 127 {
        2048
    } else {
        256
    };
    let (mut session, stack, loaded, names, vocab) =
        crate::family::qwen36::cuda::card_session_device_tests::build_rig_session_for_chunk(
            devices,
            capacity,
            crate::family::session::StoreLimits::default(),
            rows,
        )
        .expect("the rig session");
    assert_eq!(ExecutorOps::prefill_piece(&session), None);
    let spec = rig_spec();
    let inv = Inventory::for_spec(&spec).expect("inventory");
    let shape = HostShape::from_spec(&spec, 64).expect("shape");
    let d = StepDims::new(&shape, &inv).expect("dims");
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: capacity,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: crate::family::project::cuda::WeightLayout::Native,
    };
    let table = LayerTable::new(&d, &inv).expect("layers");
    // The rig requests the same mailbox sizing rule as production, so the
    // larger peer test exercises a real multi-row exchange, not a truncation.
    assert!(rows <= super::card_prefill::max_chunk_rows(d.shape.hidden));
    if rows == 2048 {
        assert_eq!(rows, super::card_prefill::max_chunk_rows(d.shape.hidden));
    }
    let inputs = || CardChunkInputs {
        weights: loaded,
        embed_name: Box::leak(names.embed.clone().into_boxed_str()),
    };
    let chunk = match stage_rows {
        Some(stage_rows) => build_card_chunk_with_stage_rows(
            stack,
            &d,
            &table,
            inputs(),
            &options,
            rows,
            stage_rows,
        ),
        None => build_card_chunk(stack, &d, &table, inputs(), &options, rows),
    }
    .unwrap_or_else(|error| panic!("{NAME}: the chunk build refused: {error:?}"));
    assert_eq!(
        chunk.staging_for_test().expect("staging views"),
        vec![Some((stage_rows.unwrap_or(capacity), true)); 2],
        "{NAME}: both cards must exercise the requested staging path"
    );

    // Deterministic, in-range token ids.
    let tokens: Vec<u32> = (0..rows as u32 + 1).map(|i| (i * 37 + 11) % 900).collect();
    let (chunk_rows, [last]) = tokens.split_at(tokens.len() - 1) else {
        unreachable!("len checked above");
    };

    // The per-token control: every row through run_token.
    TokenStepper::reset(&mut session).expect("reset control");
    let mut control_rows = Vec::new();
    for &token in chunk_rows {
        control_rows.push(TokenStepper::logits(&mut session, token).expect("control logits"));
    }
    let control_last = TokenStepper::logits(&mut session, *last).expect("control last");

    // The chunk pass: rows through the driver, the last token per-token.
    TokenStepper::reset(&mut session).expect("reset chunk");
    let fed = TokenStepper::fed(&session);
    run_card_chunk(stack, session.states_mut(), &chunk, chunk_rows, fed)
        .unwrap_or_else(|error| panic!("{NAME}: the chunk run refused: {error:?}"));
    session.note_chunk_rows(chunk_rows.len());
    let chunk_last = TokenStepper::logits(&mut session, *last).expect("chunk last");

    // The gate: the same greedy winner, margin within the single-device
    // chunk suite's slack (2 * (peak * 1e-3 + 1e-6)).
    let winner = |row: &[f32]| -> (usize, f32, f32) {
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut runner = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best.1 {
                runner = best.1;
                best = (i, v);
            } else if v > runner {
                runner = v;
            }
        }
        (best.0, best.1, runner)
    };
    let (want_tok, want_best, want_runner) = winner(&control_last);
    let (got_tok, got_best, got_runner) = winner(&chunk_last);
    let peak = control_last.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let slack = 2.0 * (peak * 1e-3 + 1e-6);
    assert!(
        (got_best - want_best).abs() <= slack && (got_runner - want_runner).abs() <= slack,
        "{NAME}: the last-token margins diverge: chunk ({got_best:.4} vs {got_runner:.4}) \
         control ({want_best:.4} vs {want_runner:.4}), slack {slack:.5}"
    );
    // A hard disagreement on the winner is a real failure even when both
    // margins hold: report it distinctly.
    if got_tok != want_tok {
        panic!(
            "{NAME}: the greedy winner differs: chunk {got_tok} vs control {want_tok} \
             (peak {peak:.3}, slack {slack:.5})"
        );
    }
    // The driver does not expose intermediate logits. Check that the control
    // consumed the expected rows; only the final logits are compared here.
    assert_eq!(
        control_rows.len(),
        chunk_rows.len(),
        "{NAME}: the control fed every row"
    );

    // Exercise the serving hook across two full chunks plus an eager tail.
    let longer: Vec<u32> = (0..(rows as u32 * 2 + 2))
        .map(|i| (i * 29 + 17) % 900)
        .collect();
    let (prompt, [last]) = longer.split_at(longer.len() - 1) else {
        unreachable!("the prompt has a final token");
    };
    TokenStepper::reset(&mut session).expect("reset long control");
    for &token in prompt {
        TokenStepper::feed(&mut session, token).expect("long control feed");
    }
    let long_control = TokenStepper::logits(&mut session, *last).expect("long control logits");
    TokenStepper::reset(&mut session).expect("reset long chunk");
    session.set_chunk_for_test(chunk);
    assert_eq!(ExecutorOps::prefill_piece(&session), Some(rows));

    // The embedding table has a final padded row outside the sampler vocabulary.
    // Refuse invalid IDs anywhere in the input, including a later piece and the
    // eager tail, before any earlier row can change the session.
    let sampler_vocab = vocab - 1;
    session.set_sampler_vocab_for_test(sampler_vocab);
    let padded = u32::try_from(sampler_vocab).expect("rig vocabulary fits u32");
    let mut late = vec![11; rows + 3];
    late[rows + 1] = padded;
    for bad in [vec![padded, 11], vec![11, padded], late, vec![u32::MAX, 11]] {
        let error = TokenStepper::feed_rows(&mut session, &bad).expect_err("invalid ID refused");
        assert!(
            matches!(error, EngError::BadRequest(ref text) if text.contains("outside the sampler vocabulary")),
            "{NAME}: unexpected refusal: {error:?}"
        );
        assert_eq!(
            TokenStepper::fed(&session),
            0,
            "{NAME}: refusal must not feed any row"
        );
    }
    session.set_sampler_vocab_for_test(vocab);
    TokenStepper::feed_rows(&mut session, prompt).expect("long chunk feed");
    let long_chunk = TokenStepper::logits(&mut session, *last).expect("long chunk logits");
    let (want_tok, want_best, want_runner) = winner(&long_control);
    let (got_tok, got_best, got_runner) = winner(&long_chunk);
    let peak = long_control.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let slack = 2.0 * (peak * 1e-3 + 1e-6);
    assert_eq!(got_tok, want_tok, "{NAME}: winner after multiple chunks");
    assert!(
        (got_best - want_best).abs() <= slack && (got_runner - want_runner).abs() <= slack,
        "{NAME}: margins after multiple chunks exceed {slack:.5}"
    );
    let executor = super::card_session::Q36CudaStep::Cards(Box::new(session));
    assert_eq!(ExecutorOps::prefill_piece(&executor), Some(rows));
    println!("{NAME}: {rows} rows, winner {want_tok}, chunk margin {got_best:.4}/{got_runner:.4}");
}
