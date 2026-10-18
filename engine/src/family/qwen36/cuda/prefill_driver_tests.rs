//! Host tests of the chunked-prefill driver's pure pieces (CPF-7b-C D.1): the admission
//! inputs and their planned-against-allocated equality, the cap, decode-arena and extras
//! terms of the admission, the pre-arena verdict, the zero spans of the arena fill, the
//! gate's fifth condition, and the start line of the prefill path. Real dimensions from
//! `common_tests`; no device anywhere — the device rows are the `prefill_step_device_tests`
//! of T1/T2.

use super::common_tests;
use super::dims::{KvMode, StepOptions};
use super::prefill_driver::{
    AdmitFacts, ZERO_CHUNK, admit_guard, admit_m_sess, drive_chunks, needs_arena, zero_spans,
};
use super::prefill_plan::{
    ChunkCost, Extra, M_LADDER, PART_CHARGE, PrefillStep, SPARE_MIN, chunk_costs, chunk_list,
    kv_bytes, state_bytes,
};
use super::start::prefill_start_line;
use super::state_layout::StateLayout;
use super::step::{ChunkFacts, ChunkVerdict, chunk_verdict};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;

/// The SM count of the 3090, the device the costs rows of the packet run on.
const NSM: usize = 82;

/// The options of a plain session of `window` positions under `mode`: the fields the state
/// layout reads (the KV mode, the capacity, the DeltaNet slots), the rest at their
/// stage-1 defaults.
fn options_of(mode: KvMode, window: usize) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens: window,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

/// H-B1: the admission state term is the WHOLE resident layout — the planned figure equals
/// the allocated one — and a planned term that is not the allocated state (the KV bytes
/// alone, the 7b-A1 input, about 62.8 MiB short at the 16K window) is refused by its text
/// before anything else. Kills 7b-A1b (the equality guard dropped) and keeps the equality
/// text pinned for the device refusal kill of 7b-A1 (DV-B1).
#[test]
fn admission_charges_the_whole_resident_state() {
    let dims = common_tests::dims();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        for window in [1024usize, 32_768] {
            let state = state_bytes(&dims, mode, window, 1).expect("the planned state bytes");
            let layout = StateLayout::new(&dims, &options_of(mode, window)).expect("the layout");
            assert_eq!(
                layout.bytes().expect("the allocated bytes"),
                state,
                "{mode:?} x {window}: the planned term is the allocated layout's bytes"
            );
            // The equal pair admits the top of the ladder.
            let costs = chunk_costs(&dims, NSM).expect("the costs");
            assert_eq!(
                admit_m_sess(
                    &AdmitFacts {
                        budget: 1 << 35,
                        decode_arena: 0,
                        state_planned: state,
                        state_allocated: state,
                        cap: M_LADDER[0],
                    },
                    &[],
                    &costs
                ),
                Ok(Some(M_LADDER[0])),
                "{mode:?} x {window}: the equal pair admits"
            );
            // The mutant's input: the KV bytes alone as the planned term.
            let kv = kv_bytes(&dims, mode, window).expect("the kv bytes");
            match admit_m_sess(
                &AdmitFacts {
                    budget: 1 << 35,
                    decode_arena: 0,
                    state_planned: kv,
                    state_allocated: state,
                    cap: M_LADDER[0],
                },
                &[],
                &costs,
            ) {
                Err(EngError::Checkpoint(message)) => {
                    assert!(
                        message.contains("prefill admission: the planned state")
                            && message.contains(&format!("is not the allocated state {state}")),
                        "{mode:?} x {window}: {message}"
                    );
                }
                other => {
                    panic!("{mode:?} x {window}: expected the equality refusal, got {other:?}")
                }
            }
        }
    }
}

/// H-B2: the admission takes the cap, the decode arena and the extras. Every budget
/// boundary is computed from the `chunk_costs` rows in this test, never a literal. Kills
/// 7b-A2 (the cap dropped) and 7b-A6 (the decode-arena charge dropped).
#[test]
fn admission_takes_the_cap_the_decode_arena_and_the_extras() {
    let dims = common_tests::dims();
    let costs = chunk_costs(&dims, NSM).expect("the costs");
    let cost_of = |m: usize| {
        *costs
            .iter()
            .find(|cost| cost.m == m)
            .expect("a ladder cost")
    };
    let state = state_bytes(&dims, KvMode::Fp16, 1024, 1).expect("the state bytes");
    // The device sum of `cost` with `decode` bytes of decode arena: the state, the arena,
    // the larger of the flat and the real part charge, and the decode arena itself.
    let charged =
        |cost: &ChunkCost, decode: usize| decode + state + cost.arena + PART_CHARGE.max(cost.part);
    // The least budget that admits the top of the ladder with `decode` charged.
    let boundary = |decode: usize| charged(&cost_of(M_LADDER[0]), decode) + SPARE_MIN;
    let admits = |budget: usize, decode: usize, cap: usize, extras: &[Extra]| {
        admit_m_sess(
            &AdmitFacts {
                budget,
                decode_arena: decode,
                state_planned: state,
                state_allocated: state,
                cap,
            },
            extras,
            &costs,
        )
        .expect("the admission runs")
    };
    // A budget that admits 2048 with cap 2048 gives 2048 ...
    assert_eq!(admits(boundary(0), 0, M_LADDER[0], &[]), Some(2048));
    // ... and with cap 128 gives 128 (the cap narrows the ladder, 7b-A2).
    assert_eq!(admits(boundary(0), 0, 128, &[]), Some(128));
    // Charging the decode arena lowers the budget the ladder sees by exactly that many
    // bytes, so the SAME budget admits the next size down only when the arena is charged
    // (7b-A6: with the charge dropped, the row below returns the top again).
    let decode = 1 << 20;
    assert_eq!(
        admits(boundary(0), decode, M_LADDER[0], &[]),
        Some(1024),
        "charged: the top of the ladder no longer fits"
    );
    assert_eq!(
        admits(boundary(0), 0, M_LADDER[0], &[]),
        Some(2048),
        "uncharged: the same budget still admits the top"
    );
    // An extra of one fixed byte at the 2048 boundary flips it to 1024.
    let probe = [Extra {
        name: "probe",
        fixed: 1,
        bytes_per_token: 0,
    }];
    assert_eq!(
        admits(boundary(0), 0, M_LADDER[0], &probe),
        Some(1024),
        "the extra eats the spare minimum of the top size"
    );
}

/// H-B3: the zero spans cover `[0, total)` exactly once, contiguous from 0, each span
/// `1..=ZERO_CHUNK` long; a zero-byte span chunk is refused. Kills 7b-A4 (one span short).
#[test]
fn zero_spans_cover_the_arena_once() {
    for total in [
        1usize,
        ZERO_CHUNK - 1,
        ZERO_CHUNK,
        ZERO_CHUNK + 1,
        777_224_192,
    ] {
        let spans = zero_spans(total, ZERO_CHUNK).expect("the spans");
        let mut at = 0usize;
        for &(start, len) in &spans {
            assert_eq!(start, at, "{total}: the spans are contiguous from 0");
            assert!(
                (1..=ZERO_CHUNK).contains(&len),
                "{total}: span {start}+{len} is not 1..=ZERO_CHUNK"
            );
            at += len;
        }
        assert_eq!(
            at, total,
            "{total}: the spans cover [0, total) exactly once"
        );
    }
    // The 741.2 MiB arena of the ladder's top: 92 whole spans and one tail (93).
    assert_eq!(
        zero_spans(777_224_192, ZERO_CHUNK)
            .expect("the spans")
            .len(),
        93
    );
    match zero_spans(1024, 0) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains("zero-byte spans"), "{message}");
        }
        other => panic!("expected the zero-span refusal, got {other:?}"),
    }
}

/// H-B4: the gate refuses without an arena AFTER the Q6_K scan, and the accept verdict
/// carries the ADMITTED size, never the cap. Kills 7b-V1 and 7b-V2.
#[test]
fn the_gate_refuses_without_an_arena_after_the_q6k_check() {
    let otherwise_ready = || ChunkFacts {
        layout: WeightLayout::Native,
        kv_mode: KvMode::Fp16,
        tiered: false,
        mmq_q6k: false,
        m_sess_cap: 2048,
        arena: None,
    };
    assert_eq!(
        chunk_verdict(&otherwise_ready()),
        ChunkVerdict::NoArena,
        "an otherwise-ready session with no arena keeps the per-token path"
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            arena: Some(512),
            ..otherwise_ready()
        }),
        ChunkVerdict::Ready { m_sess: 512 },
        "the accept verdict carries the admitted size, never the cap (7b-V2)"
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            mmq_q6k: true,
            ..otherwise_ready()
        }),
        ChunkVerdict::Q6KWeight,
        "NoArena is checked after the Q6_K scan"
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            kv_mode: KvMode::Int8G32,
            arena: Some(512),
            ..otherwise_ready()
        }),
        ChunkVerdict::Kv8,
        "an arena never overrides an earlier refusal"
    );
}

/// H-B5: only an otherwise-ready session plans an arena — one false row per condition of
/// the pre-arena half. Kills 7b-V3.
#[test]
fn only_an_otherwise_ready_session_plans_an_arena() {
    assert!(needs_arena(
        WeightLayout::Native,
        KvMode::Fp16,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Tiled,
        KvMode::Fp16,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Int8G32,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Fp16,
        true,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Fp16,
        false,
        true
    ));
}

/// H-B6 (R3): the start line names the prefill path of every verdict, the accept with its
/// admitted size. Kills 7b-R3.
#[test]
fn the_start_line_names_the_prefill_path() {
    assert_eq!(
        prefill_start_line(&ChunkVerdict::Ready { m_sess: 2048 }),
        "q36 cuda: prefill chunk M_sess 2048"
    );
    assert_eq!(
        prefill_start_line(&ChunkVerdict::TiledLayout),
        "q36 cuda: prefill per token: tiled layout"
    );
    assert_eq!(
        prefill_start_line(&ChunkVerdict::Kv8),
        "q36 cuda: prefill per token: 8-bit cache"
    );
    assert_eq!(
        prefill_start_line(&ChunkVerdict::Tiered),
        "q36 cuda: prefill per token: tiered mode"
    );
    assert_eq!(
        prefill_start_line(&ChunkVerdict::Q6KWeight),
        "q36 cuda: prefill per token: Q6_K weight"
    );
    assert_eq!(
        prefill_start_line(&ChunkVerdict::NoArena),
        "q36 cuda: prefill per token: no arena"
    );
}

/// H-B13 (condition 8a): the two admission guards of `admit_prefill` refuse by their exact
/// texts, each with the other guard satisfied; both clear is `Ok`. Kills 7b-A7 (guard 2
/// dropped: the second row would return `Ok`).
#[test]
fn the_admission_guards_refuse_by_text() {
    assert_eq!(
        admit_guard(true, false),
        Err(EngError::BadRequest(
            "step: the prefill arena is admitted before the first token".to_owned()
        ))
    );
    assert_eq!(
        admit_guard(false, true),
        Err(EngError::BadRequest(
            "step: the prefill arena is already admitted".to_owned()
        ))
    );
    assert_eq!(admit_guard(false, false), Ok(()));
}

// CPF-7b-C B.4: continuation must start at the committed position.
#[test]
fn chunk_start_must_equal_the_committed_position() {
    use super::prefill_driver::check_chunk_start;

    assert_eq!(check_chunk_start(64, 64), Ok(()));
    assert_eq!(
        check_chunk_start(63, 64),
        Err(EngError::BadRequest(
            "prefill chunk: p0 63 is not the committed rows 64".to_owned()
        ))
    );
    assert_eq!(
        check_chunk_start(65, 64),
        Err(EngError::BadRequest(
            "prefill chunk: p0 65 is not the committed rows 64".to_owned()
        ))
    );
}

// H-B9: sampler padding is invalid even when an embedding row exists.
#[test]
fn chunk_ids_check_vocabulary_before_signed_conversion() {
    use super::prefill_driver::chunk_ids;

    assert_eq!(
        chunk_ids(&[1, 3], 3),
        Err(EngError::BadRequest(
            "token id 3 is outside the sampler vocabulary 0..3".to_owned()
        ))
    );
    assert_eq!(chunk_ids(&[2, 0, 1], 3), Ok(vec![2, 0, 1]));
    assert_eq!(
        chunk_ids(&[u32::MAX], 3),
        Err(EngError::BadRequest(
            "token id 4294967295 is outside the sampler vocabulary 0..3".to_owned()
        ))
    );
    assert_eq!(
        chunk_ids(&[2_147_483_648], usize::MAX),
        Err(EngError::BadRequest(
            "prefill chunk: token id 2147483648 does not fit i32".to_owned()
        ))
    );
}

/// H-B8 (CPF-7b-C D.1): every step gets its own rows at its own position, a feed step exactly
/// one, and an empty call drives nothing.
#[test]
fn drive_chunks_slices_rows_at_their_positions() {
    for &start in &[0usize, 130] {
        for &count in &[0usize, 1, 2, 129, 300] {
            let tokens: Vec<u32> = (0..count as u32).map(|at| at + 7).collect();
            let mut log: Vec<(PrefillStep, Vec<u32>)> = Vec::new();
            let ran = drive_chunks(start, &tokens, 128, |step, rows| {
                log.push((step, rows.to_vec()));
                Ok(())
            })
            .expect("a ladder chunk size plans every count");
            let expected = chunk_list(start, count, 128).expect("the same plan");
            assert_eq!(
                ran, expected,
                "the plan that ran is returned: start {start}, {count} rows"
            );
            assert_eq!(
                log.len(),
                expected.len(),
                "one call per step: start {start}, {count} rows"
            );
            for ((step, rows), want) in log.iter().zip(expected.iter()) {
                let (at, len) = match *want {
                    PrefillStep::Chunk { p0, rows } => (p0, rows),
                    PrefillStep::Feed { p0 } => (p0, 1),
                };
                assert_eq!(*step, *want, "the closure sees the planned step");
                assert_eq!(rows.len(), len, "the step gets its own row count");
                if matches!(*want, PrefillStep::Feed { .. }) {
                    assert_eq!(rows.len(), 1, "a feed step gets exactly one row");
                }
                let offset = at - start;
                assert_eq!(
                    rows.as_slice(),
                    &tokens[offset..offset + len],
                    "the rows are the prompt rows of that position"
                );
            }
        }
    }
}
