//! Host tests of the chunked-prefill driver's pure pieces: the admission inputs and their
//! planned-against-allocated equality, the cap, decode-arena and extras terms of the admission, the
//! pre-arena verdict, the zero spans of the arena fill, the gate's fifth condition, and the start
//! line of the prefill path. Real dimensions from `common_tests`; no device anywhere — the device
//! rows are in `prefill_step_device_tests`.

use super::common_tests;
use super::dims::{KvMode, StepOptions};
use super::prefill_driver::{
    AdmitFacts, FIXED_ADMISSION_MARGIN, FULL_STAGE_TIER_ROWS, Kv8StagingPlan, LedgerFigures,
    StageChoice, ZERO_CHUNK, admit_fixed, admit_fixed_staged, admit_guard, admit_m_sess,
    admit_with_staging, drive_chunks, fixed_chunk_rows, fixed_staging_rows, ledger_line,
    needs_arena, require_fixed_room, stage_label, wide_stage_candidate, zero_spans,
};
use super::prefill_plan::{
    ChunkCost, Extra, M_LADDER, PART_CHARGE, SPARE_MIN, chunk_costs, chunk_list, kv_bytes,
    state_bytes,
};
use super::start::prefill_start_line;
use super::state_layout::StateLayout;
use super::step::{ChunkFacts, ChunkVerdict, chunk_verdict};
use super::step_plan::{KernelScratch, StepPlan};
use crate::EngError;
use crate::family::prefill::arena::PrefillStep;
use crate::family::project::cuda::WeightLayout;

fn native_staging_owner() -> Result<[bool; 7], EngError> {
    use super::prefill_driver::{PrefillArena, io};
    use crate::kernels::cuda::CudaContext;
    use std::sync::atomic::Ordering;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let facts = ctx.mmq_device_facts()?;
    io::reset();
    let mut arena =
        PrefillArena::allocate(&ctx, &stream, &common_tests::dims(), 128, facts, usize::MAX)?;
    let base = arena.bytes();
    let base_allocs = io::ALLOCS.load(Ordering::Relaxed);
    let zero_writes = io::SPAN_WRITES.load(Ordering::Relaxed);
    let plan = Kv8StagingPlan::new(7, 2, 64)?;
    arena.attach_kv8_staging(&ctx, plan, false)?;
    let one = arena.kv8_staging()?.unwrap();
    let two = arena.kv8_staging()?.unwrap();
    let reused = std::ptr::eq(one.planes[0].buffer(), two.planes[0].buffer());
    let separate = !std::ptr::eq(one.planes[0].buffer(), arena.part());
    let split = one.planes[0].offset() == 0
        && one.planes[1].offset() == plan.bytes / 2
        && one
            .planes
            .iter()
            .all(|plane| plane.bytes() == plan.bytes / 2);
    let reported = arena.bytes() == base + plan.bytes
        && arena.part_bytes() == super::prefill_plan::PART_CHARGE;
    let refused_twice = matches!(
        arena.attach_kv8_staging(&ctx, plan, false),
        Err(EngError::BadRequest(_))
    );
    Ok([
        reused,
        separate,
        split,
        reported,
        refused_twice,
        base_allocs == 2 && io::ALLOCS.load(Ordering::Relaxed) == 3,
        io::SPAN_WRITES.load(Ordering::Relaxed) == zero_writes,
    ])
}

#[test]
#[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
fn kv8_staging_has_one_reused_allocation_separate_from_part() {
    if !crate::kernels::cuda::device_present(
        "kv8_staging_has_one_reused_allocation_separate_from_part",
    ) {
        return;
    }
    assert_eq!(native_staging_owner().unwrap(), [true; 7]);
}

#[test]
fn kv8_staging_is_bounded_and_crossing_the_cap_uses_fallback() {
    for (capacity, staged, bytes) in [
        (33_280, 33_280, 68_157_440),
        (65_536, 65_536, 134_217_728),
        (262_144, 65_536, 134_217_728),
    ] {
        let plan = Kv8StagingPlan::new(capacity, 2, 256).unwrap();
        assert_eq!((plan.capacity, plan.bytes), (staged, bytes));
    }
    let plan = Kv8StagingPlan::new(262_144, 2, 256).unwrap();
    for (p0, m, covered) in [
        (65_024, 512, true),
        (65_280, 512, false),
        (65_536, 16, false),
    ] {
        let rows = super::prefill::ChunkRows::new(p0, m, 262_144, 512).unwrap();
        assert_eq!(plan.covers(rows.end()), covered);
    }
    assert!(Kv8StagingPlan::new(0, 2, 256).is_err());
}

#[test]
fn kv8_context_staging_covers_context_and_rejects_invalid_sizes() {
    for capacity in [65_536, 65_537, 131_072, 262_144] {
        let plan = Kv8StagingPlan::for_context(capacity, 2, 256).unwrap();
        assert_eq!((plan.capacity, plan.bytes), (capacity, capacity * 2048));
        assert!(plan.covers(capacity));
        assert!(!plan.covers(capacity + 1));
    }
    assert!(Kv8StagingPlan::for_context(0, 2, 256).is_err());
    assert!(Kv8StagingPlan::for_context(usize::MAX, 2, 256).is_err());
}

#[test]
fn kv8_staging_is_charged_before_admission_and_retry_keeps_caller_extras() {
    let dims = common_tests::dims();
    let costs = chunk_costs(&dims, NSM).unwrap();
    let cost = costs.iter().find(|cost| cost.m == 128).unwrap();
    let state = state_bytes(&dims, KvMode::Int8G32, 262_144, 1).unwrap();
    let extra = Extra {
        name: "caller",
        fixed: 16 << 20,
        bytes_per_token: 64,
    };
    let base = state
        + cost.arena
        + PART_CHARGE.max(cost.part)
        + extra.fixed
        + extra.bytes_per_token * 128
        + SPARE_MIN;
    let mut facts = AdmitFacts {
        budget: base + 134_217_728,
        decode_arena: 0,
        state_planned: state,
        state_allocated: state,
        cap: 128,
    };
    let plan = Kv8StagingPlan::new(262_144, 2, 256).unwrap();
    assert_eq!(
        admit_with_staging(&facts, &[extra], &costs, Some(plan)).unwrap(),
        (Some(128), Some(plan))
    );
    facts.budget -= 1;
    assert_eq!(
        admit_with_staging(&facts, &[extra], &costs, Some(plan)).unwrap(),
        (Some(128), None)
    );
    facts.budget = base - 1;
    assert_eq!(
        admit_with_staging(&facts, &[extra], &costs, Some(plan)).unwrap(),
        (None, None)
    );
    facts.state_allocated += 1;
    assert!(matches!(
        admit_with_staging(&facts, &[extra], &costs, Some(plan)),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn kv8_extended_staging_fits_the_available_budget() {
    use crate::kernels::cuda::prefill::kv8_dequant::staging_bytes;

    let dims = common_tests::dims();
    let costs = chunk_costs(&dims, NSM).unwrap();
    let cost = costs.iter().find(|cost| cost.m == 128).unwrap();
    let state = state_bytes(&dims, KvMode::Int8G32, 131_072, 1).unwrap();
    let extra = Extra {
        name: "caller",
        fixed: 16 << 20,
        bytes_per_token: 64,
    };
    let decode = 4096;
    let base = state
        + decode
        + cost.arena
        + PART_CHARGE.max(cost.part)
        + extra.fixed
        + extra.bytes_per_token * 128
        + SPARE_MIN;
    let wanted = Kv8StagingPlan {
        capacity: 131_072,
        bytes: staging_bytes(131_072, 2, 256).unwrap(),
    };
    let mut facts = AdmitFacts {
        budget: base + staging_bytes(98_304, 2, 256).unwrap(),
        decode_arena: decode,
        state_planned: state,
        state_allocated: state,
        cap: 128,
    };
    for (less, capacity) in [(0, 98_304), (1, 98_303)] {
        facts.budget = base + staging_bytes(98_304, 2, 256).unwrap() - less;
        let expected = Kv8StagingPlan {
            capacity,
            bytes: staging_bytes(capacity, 2, 256).unwrap(),
        };
        assert_eq!(
            admit_with_staging(&facts, &[extra], &costs, Some(wanted)).unwrap(),
            (Some(128), Some(expected)),
            "the admitted prefix must fit both planes, caller extras and spare bytes"
        );
    }
}

/// The SM count of the 3090, the device the cost rows run on.
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

/// The admission state term is the WHOLE resident layout — the planned figure equals the allocated
/// one — and a planned term that is not the allocated state (the KV bytes alone, about 62.8 MiB
/// short at the 16K window) is refused by its text before anything else. Dropping the equality
/// guard fails this test, and the pinned equality text is the one the device refusal row expects.
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

/// The admission takes the cap, the decode arena and the extras. Every budget boundary is computed
/// from the `chunk_costs` rows in this test, never a literal. Dropping the cap or the decode-arena
/// charge fails it.
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
    assert_eq!(admits(boundary(0), 0, M_LADDER[0], &[]), Some(2048));
    assert_eq!(admits(boundary(0), 0, 128, &[]), Some(128));
    // Charging the decode arena lowers the budget the ladder sees by exactly that many
    // bytes, so the SAME budget admits the next size down only when the arena is charged
    // (with the charge dropped, the row below returns the top again).
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

/// The zero spans cover `[0, total)` exactly once, contiguous from 0, each span `1..=ZERO_CHUNK`
/// long; a zero-byte span chunk is refused. A fill one span short fails it.
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

/// The gate refuses without an arena AFTER the Q6_K scan, and the accept verdict carries the
/// ADMITTED size, never the cap.
#[test]
fn the_gate_refuses_without_an_arena_after_the_q6k_check() {
    let otherwise_ready = || ChunkFacts {
        layout: WeightLayout::Native,
        kv_mode: KvMode::Fp16,
        kv8_chunked_prefill: false,
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
        "the accept verdict carries the admitted size, never the cap"
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

/// An EXL3 session takes the chunk gate like a Native one: it needs an admitted arena, and the
/// same KV rule and refusals apply.
#[test]
fn an_exl3_session_takes_the_chunk_gate_of_a_native_one() {
    let exl3 = || ChunkFacts {
        layout: WeightLayout::Exl3,
        kv_mode: KvMode::Fp16,
        kv8_chunked_prefill: false,
        tiered: false,
        mmq_q6k: false,
        m_sess_cap: 2048,
        arena: None,
    };
    assert_eq!(chunk_verdict(&exl3()), ChunkVerdict::NoArena);
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            arena: Some(1024),
            ..exl3()
        }),
        ChunkVerdict::Ready { m_sess: 1024 }
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            kv_mode: KvMode::Int8G32,
            arena: Some(1024),
            ..exl3()
        }),
        ChunkVerdict::Kv8,
        "Int8G32 needs the opt-in for an EXL3 pack too"
    );
    assert_eq!(
        chunk_verdict(&ChunkFacts {
            layout: WeightLayout::Tiled,
            arena: Some(1024),
            ..exl3()
        }),
        ChunkVerdict::TiledLayout
    );
}

/// Only an otherwise-ready session plans an arena — one false row per condition of the pre-arena
/// half.
#[test]
fn only_an_otherwise_ready_session_plans_an_arena() {
    assert!(needs_arena(
        WeightLayout::Native,
        KvMode::Fp16,
        false,
        false,
        false
    ));
    assert!(needs_arena(
        WeightLayout::Exl3,
        KvMode::Fp16,
        false,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Tiled,
        KvMode::Fp16,
        false,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Int8G32,
        false,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Exl3,
        KvMode::Int8G32,
        false,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Fp16,
        true,
        false,
        false
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Fp16,
        false,
        true,
        false
    ));
}

#[test]
fn kv8_opt_in_plans_an_arena_without_bypassing_layout_or_weight_guards() {
    assert!(needs_arena(
        WeightLayout::Native,
        KvMode::Int8G32,
        false,
        false,
        true
    ));
    assert!(needs_arena(
        WeightLayout::Exl3,
        KvMode::Int8G32,
        false,
        false,
        true
    ));
    assert!(!needs_arena(
        WeightLayout::Tiled,
        KvMode::Int8G32,
        false,
        false,
        true
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Int8G32,
        true,
        false,
        true
    ));
    assert!(!needs_arena(
        WeightLayout::Native,
        KvMode::Int8G32,
        false,
        true,
        true
    ));
}

/// The start line names the prefill path of every verdict, the accept with its admitted size.
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

/// The two admission guards of `admit_prefill` refuse by their exact texts, each with the other
/// guard satisfied; both clear is `Ok`. Without the second guard the second row would return `Ok`.
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

// Continuation must start at the committed position.
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

// Sampler padding is invalid even when an embedding row exists.
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

/// Every step gets its own rows at its own position, a feed step exactly one, and an empty call
/// drives nothing.
#[test]
fn drive_chunks_slices_rows_at_their_positions() {
    for &start in &[0usize, 130] {
        for &count in &[0usize, 1, 2, 129, 300] {
            let tokens: Vec<u32> = (0..count as u32).map(|at| at + 7).collect();
            let mut log: Vec<(PrefillStep, Vec<u32>)> = Vec::new();
            let ran = drive_chunks(start, &tokens, 128, false, |step, rows| {
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

#[test]
fn tiered_wide_deep_coalesces_only_contiguous_rows_through_8k() {
    use super::prefill_driver::coalesce_tiered_deep;

    let original = chunk_list(0, 8_288, 2_048).unwrap();
    let widened = coalesce_tiered_deep(original, 2_048, 0);
    assert_eq!(
        widened,
        vec![
            PrefillStep::Chunk { p0: 0, rows: 2_048 },
            PrefillStep::Chunk {
                p0: 2_048,
                rows: 2_048
            },
            PrefillStep::Chunk {
                p0: 4_096,
                rows: 2_048
            },
            PrefillStep::Chunk {
                p0: 6_144,
                rows: 2_048
            },
            PrefillStep::Chunk {
                p0: 8_192,
                rows: 96
            },
        ]
    );
}

#[test]
fn tiered_wide_staged_stops_at_the_allocated_end_and_caps_deep_chunks() {
    use super::prefill_driver::coalesce_tiered_deep;
    use PrefillStep::Chunk;

    let start = 30_719;
    let end = start + 4_096;
    for arena in [2_048, 4_096] {
        let original = chunk_list(start, 6_144, arena).unwrap();
        let mut expected = vec![
            Chunk {
                p0: start,
                rows: 2_048,
            },
            Chunk {
                p0: start + 2_048,
                rows: 2_048,
            },
        ];
        expected.extend(chunk_list(end, 2_048, arena).unwrap());
        assert_eq!(coalesce_tiered_deep(original.clone(), arena, end), expected);
        assert_eq!(coalesce_tiered_deep(original.clone(), arena, 0), original);
    }
}

#[test]
fn tiered_wide_staged_preserves_scheduler_pieces_and_imported_positions() {
    use super::prefill_driver::drive_tiered_wide_chunks;

    let start = 16_383;
    let prompt = (0..4_097).collect::<Vec<u32>>();
    let mut seen = Vec::new();
    let mut steps = Vec::new();
    for (offset, piece) in prompt.chunks(2_048).enumerate() {
        let at = start + 2_048 * offset;
        steps.extend(
            drive_tiered_wide_chunks(at, piece, 2_048, false, 65_536, |_, rows| {
                seen.extend_from_slice(rows);
                Ok(())
            })
            .unwrap(),
        );
    }
    assert_eq!(seen, prompt);
    assert_eq!(
        steps,
        [
            PrefillStep::Chunk {
                p0: start,
                rows: 2_048
            },
            PrefillStep::Chunk {
                p0: start + 2_048,
                rows: 2_048
            },
            PrefillStep::Feed { p0: start + 4_096 },
        ]
    );
}

#[test]
fn tiered_wide_deep_respects_arena_gaps_feeds_and_depth_limit() {
    use super::prefill_driver::coalesce_tiered_deep;

    let source = vec![
        PrefillStep::Chunk {
            p0: 4_096,
            rows: 1_024,
        },
        PrefillStep::Chunk {
            p0: 5_120,
            rows: 1_024,
        },
        PrefillStep::Feed { p0: 6_144 },
        PrefillStep::Chunk {
            p0: 7_168,
            rows: 1_024,
        },
        PrefillStep::Chunk {
            p0: 8_192,
            rows: 1_024,
        },
    ];
    assert_eq!(coalesce_tiered_deep(source.clone(), 1_024, 0), source);
    let got = coalesce_tiered_deep(source, 2_048, 0);
    assert_eq!(
        got,
        vec![
            PrefillStep::Chunk {
                p0: 4_096,
                rows: 2_048
            },
            PrefillStep::Feed { p0: 6_144 },
            PrefillStep::Chunk {
                p0: 7_168,
                rows: 1_024
            },
            PrefillStep::Chunk {
                p0: 8_192,
                rows: 1_024
            },
        ]
    );
}

#[test]
fn fixed_drive_limits_depth_without_shrinking_the_session_arena() {
    let tokens: Vec<u32> = (0..1024).collect();
    let mut seen = Vec::new();
    let mut copied = Vec::new();
    drive_chunks(155_136, &tokens, 512, true, |step, rows| {
        seen.push(step);
        copied.extend_from_slice(rows);
        Ok(())
    })
    .unwrap();
    assert_eq!(copied, tokens);
    assert_eq!(
        seen,
        vec![
            PrefillStep::Chunk {
                p0: 155_136,
                rows: 512
            },
            PrefillStep::Chunk {
                p0: 155_648,
                rows: 256
            },
            PrefillStep::Chunk {
                p0: 155_904,
                rows: 256
            },
        ]
    );
}

const FIXED_TEST_FREE: usize = 3_663_265_792;

struct FixedCase {
    context: usize,
    facts: AdmitFacts,
    extras: Vec<Extra>,
    costs: Vec<ChunkCost>,
    staging: Kv8StagingPlan,
}

fn fixed_case(capacity: usize) -> FixedCase {
    let dims = common_tests::dims();
    let options = super::start::step_options(
        u32::try_from(capacity).unwrap(),
        WeightLayout::Native,
        KvMode::Int8G32,
    )
    .unwrap();
    let layout = StateLayout::new(&dims, &options).unwrap();
    let scratch = KernelScratch::for_dims(&dims, 1, KvMode::Int8G32).unwrap();
    FixedCase {
        context: capacity,
        facts: AdmitFacts {
            budget: FIXED_TEST_FREE - super::start::prefill_headroom(true),
            decode_arena: StepPlan::new(&dims, 1, &options, &scratch)
                .unwrap()
                .arena_bytes(),
            state_planned: state_bytes(&dims, KvMode::Int8G32, capacity, 1).unwrap(),
            state_allocated: layout.bytes().unwrap(),
            cap: fixed_chunk_rows(capacity, 1024),
        },
        extras: vec![Extra {
            name: "device-prefix-delta",
            fixed: layout.delta.total,
            bytes_per_token: 0,
        }],
        costs: chunk_costs(&dims, NSM).unwrap(),
        staging: Kv8StagingPlan::for_context(fixed_staging_rows(capacity), 2, 256).unwrap(),
    }
}

#[test]
fn fixed_admission_ledger_lists_every_charge_at_target_capacities() {
    use crate::kernels::cuda::plan as load_plan;
    for (capacity, rows) in [(131_072, 1024), (196_608, 512), (262_144, 512)] {
        let mut case = fixed_case(capacity);
        assert_eq!(case.facts.cap, rows);
        let cost = case.costs.iter().find(|cost| cost.m == rows).unwrap();
        let part = PART_CHARGE.max(cost.part);
        let charges = [
            case.facts.state_allocated,
            case.facts.decode_arena,
            cost.arena,
            part,
            case.extras[0].fixed,
            case.staging.bytes,
            SPARE_MIN,
            FIXED_ADMISSION_MARGIN,
        ];
        let total = charges
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .unwrap();
        assert!(total <= FIXED_TEST_FREE);
        let kv = super::start::kv_cache_bytes(capacity, KvMode::Int8G32).unwrap();
        let load = super::start::load_options(kv, true);
        assert_eq!(load.reserve.headroom, FIXED_ADMISSION_MARGIN as u64);
        assert_eq!(
            super::start::load_options(kv, false).reserve.headroom,
            1 << 30
        );
        let tensors = 20_690_501_632u64;
        let demand = load_plan::Demand {
            tensors,
            kv_cache: load.reserve.kv_cache,
            work: load.reserve.work,
            headroom: load.reserve.headroom,
        };
        let before_weights = tensors + total as u64;
        let load_required = load_plan::admit(&demand, before_weights).unwrap();
        assert_eq!(load_required, tensors + kv + FIXED_ADMISSION_MARGIN as u64);
        assert!(load_plan::admit(&demand, load_required - 1).is_err());
        // The loader compares its reserve; it uploads weights, not a second headroom buffer.
        let after_weights = usize::try_from(before_weights - tensors).unwrap();
        case.facts.budget = after_weights - super::start::prefill_headroom(true);
        let selected = admit_fixed(&case.facts, &case.extras, &case.costs, case.staging).unwrap();
        assert_eq!(selected.m, rows);
        case.facts.budget -= 1;
        assert!(
            matches!(
                admit_fixed(&case.facts, &case.extras, &case.costs, case.staging),
                Err(EngError::MemoryPlan(_))
            ),
            "no silent downgrade when the exact row loses one byte"
        );
        println!(
            "\n{}",
            serde_json::json!({
                "capacity": capacity, "rows": rows, "measured_free": FIXED_TEST_FREE,
                "state": case.facts.state_allocated, "decode_arena": case.facts.decode_arena,
                "chunk_arena": cost.arena, "partial": part,
                "device_prefix_delta": case.extras[0].fixed,
                "stage_rows": case.staging.capacity, "stage": case.staging.bytes,
                "spare": SPARE_MIN, "margin": FIXED_ADMISSION_MARGIN,
                "total": total, "remaining": FIXED_TEST_FREE - total,
                "load_required": load_required, "weights": tensors,
                "combined_required": before_weights, "headroom_charges": 1
            })
        );
    }
}

#[test]
fn fixed_display_ambient_budget_ledger() {
    // Existing CUDA snapshot at 651 MiB ambient; this test performs no device call.
    const FREE_AT_651_MIB: usize = 3_614_048_256;
    let base = fixed_case(262_144);
    for cost in &base.costs {
        println!(
            "\n{}",
            serde_json::json!({
                "kind": "chunk_cost", "rows": cost.m,
                "arena": cost.arena, "real_partial": cost.part,
            })
        );
    }
    for stage_rows in [8192, 16_384, 32_768] {
        let stage = Kv8StagingPlan::for_context(stage_rows, 2, 256).unwrap();
        println!(
            "\n{}",
            serde_json::json!({
                "kind": "stage_cost", "rows": stage_rows, "bytes": stage.bytes,
            })
        );
    }
    for (ambient_mib, expected_max) in [(807, 253_952), (1536, 180_224)] {
        let available = FREE_AT_651_MIB - (ambient_mib - 651) * (1 << 20);
        let mut maximum = 0;
        for capacity in (8192..=262_144).step_by(8192) {
            let mut case = fixed_case(capacity);
            let cost = case.costs.iter().find(|c| c.m == case.facts.cap).unwrap();
            let total = case.facts.state_allocated
                + case.facts.decode_arena
                + cost.arena
                + PART_CHARGE.max(cost.part)
                + case.extras[0].fixed
                + case.staging.bytes
                + SPARE_MIN
                + FIXED_ADMISSION_MARGIN;
            case.facts.budget = available - FIXED_ADMISSION_MARGIN;
            let fits = admit_fixed(&case.facts, &case.extras, &case.costs, case.staging).is_ok();
            assert_eq!(
                fits,
                total <= available,
                "ambient {ambient_mib}, capacity {capacity}"
            );
            if fits {
                maximum = capacity;
            }
            println!(
                "\n{}",
                serde_json::json!({
                    "kind": "display_budget", "ambient_mib": ambient_mib,
                    "available_after_weights": available, "capacity": capacity,
                    "rows": case.facts.cap, "total": total, "fits": fits,
                })
            );
        }
        assert_eq!(maximum, expected_max);
    }
}

#[test]
fn fixed_chunk_and_staging_rows_are_context_bound() {
    for (capacity, rows, stage) in [
        (65_536, 1024, 65_536),
        (131_072, 1024, 131_072),
        (131_073, 512, 32_768),
        (196_608, 512, 32_768),
        (196_609, 512, 32_768),
        (262_144, 512, 32_768),
    ] {
        assert_eq!(fixed_chunk_rows(capacity, 1024), rows);
        assert_eq!(fixed_chunk_rows(capacity, 512), rows.min(512));
        assert_eq!(fixed_chunk_rows(capacity, 256), rows.min(256));
        assert_eq!(fixed_chunk_rows(capacity, 128), 128);
        assert_eq!(fixed_staging_rows(capacity), stage);
    }
}

#[test]
fn fixed_admission_retains_rows_across_noise_and_refuses_gigabyte_drops() {
    for capacity in [131_072, 196_608, 262_144] {
        let mut case = fixed_case(capacity);
        for drop in [0usize, 40 << 20, 110 << 20] {
            case.facts.budget = FIXED_TEST_FREE - drop - FIXED_ADMISSION_MARGIN;
            let cost = admit_fixed(&case.facts, &case.extras, &case.costs, case.staging).unwrap();
            assert_eq!(cost.m, case.facts.cap);
            let allocated =
                case.facts.state_allocated + case.facts.decode_arena + case.extras[0].fixed;
            require_fixed_room(FIXED_TEST_FREE - drop - allocated, &cost, case.staging).unwrap();
        }
        case.facts.budget = FIXED_TEST_FREE - (2usize << 30) - FIXED_ADMISSION_MARGIN;
        assert!(matches!(
            admit_fixed(&case.facts, &case.extras, &case.costs, case.staging),
            Err(EngError::MemoryPlan(_))
        ));
        let cost = case
            .costs
            .iter()
            .find(|cost| cost.m == case.facts.cap)
            .unwrap();
        assert!(matches!(
            require_fixed_room(0, cost, case.staging),
            Err(EngError::MemoryPlan(_))
        ));
    }
}

#[test]
fn fixed_live_room_has_an_exact_boundary_without_recharging_resident_owners() {
    let case = fixed_case(262_144);
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap();
    let required = cost.arena
        + PART_CHARGE.max(cost.part)
        + case.staging.bytes
        + SPARE_MIN
        + FIXED_ADMISSION_MARGIN;
    require_fixed_room(required, cost, case.staging).unwrap();
    assert!(matches!(
        require_fixed_room(required - 1, cost, case.staging),
        Err(EngError::MemoryPlan(_))
    ));
    admit_fixed(&case.facts, &case.extras, &case.costs, case.staging).unwrap();
    let allocated = case.facts.state_allocated + case.facts.decode_arena + case.extras[0].fixed;
    let live_before_game = FIXED_TEST_FREE - allocated;
    require_fixed_room(live_before_game, cost, case.staging).unwrap();
    assert!(
        matches!(
            require_fixed_room(
                live_before_game.saturating_sub(2usize << 30),
                cost,
                case.staging
            ),
            Err(EngError::MemoryPlan(_))
        ),
        "a stale post-weight snapshot cannot hide a later multi-GB drop"
    );
}

// ---- Q36_CUDA_CPF_STAGE_128K (wide CPF staging candidate, on by default; =0 rolls back) ----

/// The live bytes `require_fixed_room` demands for a given stage: the pinned charge law.
fn live_required(cost: &ChunkCost, staging_bytes: usize) -> usize {
    cost.arena + PART_CHARGE.max(cost.part) + staging_bytes + SPARE_MIN + FIXED_ADMISSION_MARGIN
}

#[test]
fn wide_stage_candidate_is_only_wider_than_the_tier_boundary() {
    for (capacity, flag, expected) in [
        (131_072, true, false),
        (131_073, true, true),
        (180_224, true, true),
        (262_144, true, true),
        (131_073, false, false),
        (180_224, false, false),
    ] {
        assert_eq!(
            wide_stage_candidate(capacity, flag),
            expected,
            "capacity {capacity}, flag {flag}"
        );
    }
}

#[test]
fn wide_flag_admits_the_131072_stage_when_roomy() {
    let mut case = fixed_case(180_224);
    let wide = Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, 2, 256).unwrap();
    // Room for the default stage as-is, plus exactly the wide-stage delta.
    case.facts.budget += wide.bytes - case.staging.bytes;
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap();
    let free = live_required(cost, wide.bytes);
    let (cost, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        free,
        case.context,
    )
    .expect("a roomy snapshot with the wide flag admits the wide stage");
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (FULL_STAGE_TIER_ROWS, wide.bytes)
    );
    assert_eq!(choice, StageChoice::Wide);
}

#[test]
fn wide_flag_falls_back_to_the_default_stage_one_byte_short() {
    let mut case = fixed_case(180_224);
    let wide = Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, 2, 256).unwrap();
    // Pin the budget from the charge law itself, so the default stage sits exactly at
    // the SPARE_MIN boundary and the wide stage misses by exactly one byte.
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap();
    let charged = [
        case.facts.decode_arena,
        case.facts.state_allocated,
        cost.arena,
        PART_CHARGE.max(cost.part),
        case.extras[0].fixed,
        case.staging.bytes,
    ]
    .into_iter()
    .fold(0usize, usize::saturating_add);
    let delta = wide.bytes - case.staging.bytes;
    // The budget misses the wide stage by exactly one byte, so the default stage keeps
    // SPARE_MIN + delta - 1 spare bytes of its own.
    case.facts.budget = charged + SPARE_MIN + delta - 1;
    let free = live_required(cost, wide.bytes);
    let (cost, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        free,
        case.context,
    )
    .expect("a wide miss must keep the boot on the default stage");
    assert_eq!(
        (cost.m, plan.capacity, plan.bytes),
        (case.facts.cap, case.staging.capacity, case.staging.bytes)
    );
    assert_eq!(choice, StageChoice::Default);
    // Both-sided at the same boundary: one byte more budget keeps the wide stage.
    case.facts.budget += 1;
    let (cost, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        free,
        case.context,
    )
    .expect("exactly the charged wide boundary admits the wide stage");
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(plan.capacity, FULL_STAGE_TIER_ROWS);
    assert_eq!(choice, StageChoice::Wide);
}

#[test]
fn wide_flag_refusal_at_the_default_floor_keeps_todays_error_text() {
    // Wide miss AND default miss: today's default refusal surfaces AS IS.
    let mut case = fixed_case(180_224);
    let wide = Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, 2, 256).unwrap();
    // The ladder halves the stage down to the composed floor (the measured 8192-row
    // performance floor for this context) before refusing, so the floor needs a
    // starvation beyond even the smallest stage's charge.
    case.facts.budget = case.facts.budget.saturating_sub(2 << 30);
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        FIXED_TEST_FREE,
        case.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the default refusal must surface, got {error:?}")
    };
    assert_eq!(
        text,
        format!(
            "fixed prefill admission: {} rows and {} staging bytes do not fit the budget",
            case.facts.cap, case.staging.bytes
        )
    );
}

#[test]
fn wide_flag_propagates_non_memory_errors_from_the_wide_admission() {
    let mut case = fixed_case(180_224);
    let wide = Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, 2, 256).unwrap();
    // A non-Memory error from the wide arm must surface unchanged — pinned here as a
    // text/class regression pin: this BadRequest is identical to the one the default
    // arm would produce, so it cannot discriminate fallback from propagation on its
    // own; it does pin that the wide attempt never swallows or rewraps the tier error.
    case.facts.cap = 1;
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        FIXED_TEST_FREE,
        case.context,
    )
    .unwrap_err();
    assert_eq!(
        error,
        EngError::BadRequest("fixed prefill admission: unsupported row tier".into())
    );
}

#[test]
fn wide_plan_arithmetic_overflow_falls_back_to_the_default_stage() {
    // A wide plan whose charge overflows usize is "does not fit": the selector
    // treats every MemoryPlan alike, so the overflow lands on the default stage,
    // which admits on its own bounded budget.
    let case = fixed_case(180_224);
    let wide = Kv8StagingPlan {
        capacity: FULL_STAGE_TIER_ROWS,
        bytes: usize::MAX,
    };
    let (_, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        FIXED_TEST_FREE,
        case.context,
    )
    .expect("an overflowing wide charge falls back to the default stage");
    assert_eq!(
        (plan.capacity, choice),
        (case.staging.capacity, StageChoice::Default)
    );
}

#[test]
fn wide_flag_falls_back_when_only_the_live_room_misses() {
    // A live-free snapshot between the wide and default needs must keep the boot on
    // the default stage, not refuse a boot the default stage would have kept.
    let mut case = fixed_case(180_224);
    let wide = Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, 2, 256).unwrap();
    case.facts.budget += wide.bytes - case.staging.bytes;
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let wide_needed = live_required(&cost, wide.bytes);
    let default_needed = live_required(&cost, case.staging.bytes);
    let free = wide_needed - 1;
    assert!(
        free >= default_needed,
        "the pin must sit between the two needs"
    );
    let (_, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        free,
        case.context,
    )
    .expect("a live-room miss falls back to the default stage");
    assert_eq!(
        (plan.capacity, choice),
        (case.staging.capacity, StageChoice::Default)
    );
    // One byte more and the wide stage is kept: the exact live boundary.
    let (_, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        wide_needed,
        case.context,
    )
    .expect("exactly the wide need admits the wide stage");
    assert_eq!(
        (plan.capacity, choice),
        (FULL_STAGE_TIER_ROWS, StageChoice::Wide)
    );
}

#[test]
fn the_default_stage_degrades_when_only_live_room_misses() {
    // A live-room miss degrades instead of refusing: room for exactly the half
    // stage admits at 16384 rows. The floor refusal keeps the original text in its
    // own test below.
    let mut case = fixed_case(180_224);
    case.facts.budget = usize::MAX;
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let free = live_required(&cost, case.staging.bytes / 2);
    let (cost, plan) = admit_default_or_shrink(&case, free, case.context);
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (case.staging.capacity / 2, case.staging.bytes / 2)
    );
}

#[test]
fn ladder_rungs_from_every_shipped_stage_halve_exactly_to_the_floor() {
    // The halving guard keeps bytes exact, so from every shipped stage the ladder
    // must walk a chain of even rungs that lands exactly on the floor.
    for capacity in [32_768, 65_536, 131_072, 180_224, 196_608, 262_144] {
        let floor = floor_for(capacity);
        let mut rung = super::prefill_driver::fixed_staging_rows(capacity);
        assert!(
            rung.is_multiple_of(2) && rung >= floor,
            "stage {rung} for {capacity}"
        );
        while rung > floor {
            let next = rung / 2;
            assert_eq!(next * 2, rung, "rung {rung} cannot halve exactly");
            rung = next;
        }
        assert_eq!(rung, floor, "capacity {capacity} ladder missed the floor");
    }
}

/// The stage-independent charges: everything the ledger bills besides the staging
/// bytes of the current attempt.
fn base_charges(case: &FixedCase, cost: &ChunkCost) -> usize {
    let part = PART_CHARGE.max(cost.part);
    [
        case.facts.state_allocated,
        case.facts.decode_arena,
        cost.arena,
        part,
        case.extras[0].fixed,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .unwrap()
}

#[test]
fn fixed_admission_shrinks_past_an_overflowing_full_stage_charge() {
    // The 131072 stage's charge sum overflows while the half stage fits exactly:
    // the ladder must ADMIT at the half stage instead of surfacing the overflow.
    // Overflow on the full stage must not surface: a smaller stage can un-overflow
    // the sum. The exact budget and free pins charge the half-stage bytes
    // (the same bytes the allocation will use), not the full stage's — no
    // minus-one probe is run here, so this is the at-exact-boundary pin, not
    // a proof of the just-below refusal (the floor tests carry that side).
    let mut case = fixed_case(131_072);
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let base = base_charges(&case, &cost);
    let half_bytes = case.staging.bytes / 2;
    // Overflow the full stage by a small, exact amount: base + tip + full staging
    // exceeds usize::MAX while base + tip + half staging lands exactly on budget.
    case.extras.push(Extra {
        name: "overflow-tip",
        fixed: usize::MAX - base - half_bytes - SPARE_MIN,
        bytes_per_token: 0,
    });
    // Precondition: the full-stage attempt alone returns the overflow error.
    let error = admit_fixed(&case.facts, &case.extras, &case.costs, case.staging).unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the full stage must overflow, got {error:?}")
    };
    assert_eq!(text, "prefill plan: a size overflows");
    // The pins sit exactly at the half-stage bytes on both gates.
    case.facts.budget = base + case.extras[1].fixed + half_bytes + SPARE_MIN;
    let free = live_required(&cost, half_bytes);
    let (cost, plan, _) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        free,
        case.context,
    )
    .expect("the ladder must admit the half stage");
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (case.staging.capacity / 2, half_bytes)
    );
}

#[test]
fn fixed_admission_overflow_exhaust_refuses_with_the_first_attempt_error() {
    // The full stage overflows the charge sum; every intermediate rung overflows
    // too; exactly the floor rung reaches usize::MAX and misses the budget the
    // ordinary way. The refusal must be the FIRST attempt's overflow error, not
    // the floor rung's budget-miss text.
    let mut case = fixed_case(262_144);
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let base = base_charges(&case, &cost);
    let floor = floor_for(case.context);
    let floor_staging = case.staging.bytes * floor / case.staging.capacity;
    case.extras.push(Extra {
        name: "overflow-tip",
        fixed: usize::MAX - base - floor_staging,
        bytes_per_token: 0,
    });
    case.facts.budget = usize::MAX;
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        usize::MAX,
        case.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the overflow refusal must surface, got {error:?}")
    };
    assert_eq!(text, "prefill plan: a size overflows");
}

#[test]
fn fixed_admission_keeps_the_first_refusal_when_the_stage_is_odd() {
    // An odd stage cannot halve with exact bytes, so the ladder stops after the
    // first attempt and surfaces ITS budget text, unchanged: the boot refuses
    // rather than rounding a rung. The budget would admit the rounded half stage
    // (base + spare + half the staging), so the refusal can only come from the
    // evenness guard.
    let mut case = fixed_case(65_536);
    case.staging = Kv8StagingPlan::for_context(65_537, 2, 256).unwrap();
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    case.facts.budget = base_charges(&case, &cost) + SPARE_MIN + case.staging.bytes / 2;
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        usize::MAX,
        case.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the odd-stage refusal must surface, got {error:?}")
    };
    assert_eq!(
        text,
        format!(
            "fixed prefill admission: {} rows and {} staging bytes do not fit the budget",
            case.facts.cap, case.staging.bytes
        )
    );
}

#[test]
fn fixed_admission_keeps_the_first_refusal_when_a_rung_goes_odd() {
    // 32770 halves exactly to the odd 16385, whose own half (8192, rounded) is
    // at the floor and would fit the budget: the refusal is the FIRST
    // attempt's text (the 32770-row bytes), not a rounded 8192 attempt's, so this
    // pins both the evenness guard and the error source on a chain at the floor.
    let mut case = fixed_case(32_768);
    case.staging = Kv8StagingPlan::for_context(32_770, 2, 256).unwrap();
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    case.facts.budget = base_charges(&case, &cost) + SPARE_MIN + case.staging.bytes / 4;
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        usize::MAX,
        case.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the odd-rung refusal must surface, got {error:?}")
    };
    assert_eq!(
        text,
        format!(
            "fixed prefill admission: {} rows and {} staging bytes do not fit the budget",
            case.facts.cap, case.staging.bytes
        )
    );
}

#[test]
fn fixed_admission_lands_on_the_last_even_rung_above_the_floor() {
    // An even, non-power-of-two stage (3 x 8192) with a budget that fits exactly
    // its half: the ladder admits 12288 rows — its last rung above the floor:
    // halving again gives 6144, below the 8192 floor.
    let mut case = fixed_case(24_576);
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let base = base_charges(&case, &cost);
    case.facts.budget = base + case.staging.bytes / 2 + SPARE_MIN;
    let (cost, plan) = admit_default_or_shrink(&case, usize::MAX, case.context);
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (12_288, case.staging.bytes / 2)
    );
}

#[test]
fn fixed_admission_terminates_when_every_rung_overflows() {
    // Two near-usize::MAX extras overflow every rung's charge sum; the ladder
    // must terminate and refuse rather than loop or panic.
    let mut case = fixed_case(262_144);
    case.extras.push(Extra {
        name: "overflow-a",
        fixed: usize::MAX / 2,
        bytes_per_token: 0,
    });
    case.extras.push(Extra {
        name: "overflow-b",
        fixed: usize::MAX / 2,
        bytes_per_token: 0,
    });
    case.facts.budget = usize::MAX;
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        usize::MAX,
        case.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the overflow refusal must surface, got {error:?}")
    };
    // Every rung's overflow text is identical, so this pins the error class
    // and the named extra, not which rung surfaced it.
    assert_eq!(text, "prefill plan: extra overflow-b overflows");
}

#[test]
fn wide_flag_off_is_byte_for_byte_today_without_a_wide_candidate() {
    let case = fixed_case(180_224);
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let free = live_required(&cost, case.staging.bytes);
    let (staged_cost, staged_plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        free,
        case.context,
    )
    .unwrap();
    let plain_cost = admit_fixed(&case.facts, &case.extras, &case.costs, case.staging).unwrap();
    assert_eq!(
        (staged_cost, staged_plan.capacity, staged_plan.bytes),
        (plain_cost, case.staging.capacity, case.staging.bytes)
    );
    assert_eq!(choice, StageChoice::Default);
}

#[test]
fn stage_choice_names_the_boot_line_provenance() {
    assert_eq!(
        (StageChoice::Wide.as_str(), StageChoice::Default.as_str()),
        ("wide", "default")
    );
}

// ---- the tight-live-snapshot shrink ladder ----

/// The charges the ledger bills at the observed live free (977,141,760 B
/// after every allocation, the 2026-10-24 snapshot). The 180224 test pins the
/// verdict this baseline produces (not the sum itself); the 262144 ladder
/// tests reuse it as their baseline sum.
fn budget_at_observed_free(case: &FixedCase) -> usize {
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap();
    let part = PART_CHARGE.max(cost.part);
    let charges = [
        case.facts.state_allocated,
        case.facts.decode_arena,
        cost.arena,
        part,
        case.extras[0].fixed,
        case.staging.bytes,
        SPARE_MIN,
    ];
    charges
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .unwrap()
        + 977_141_760
}

/// The production floor expression itself: the split bound composed with the
/// performance floor. admit_fixed_staged computes this internally from the KV
/// capacity; step.rs has no other floor computation.
fn floor_for(context_capacity: usize) -> usize {
    super::prefill_driver::fixed_stage_min_rows(context_capacity)
}

/// The default arm under the production floor: `free` is the live-room
/// snapshot (usize::MAX for a roomy card), `kv_capacity` is the KV capacity
/// the floor is computed from.
fn admit_default_or_shrink(
    case: &FixedCase,
    free: usize,
    kv_capacity: usize,
) -> (ChunkCost, Kv8StagingPlan) {
    let (cost, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        free,
        kv_capacity,
    )
    .expect("the default arm must decide");
    assert_eq!(choice, StageChoice::Default);
    (cost, plan)
}

#[test]
fn ledger_line_pins_every_field_and_labels_a_wide_boot_not_shrunk() {
    // The figures of a 262144 prefix-off ledger, plus the shipped
    // device-prefix extra and a second distinct extra (nonzero
    // bytes_per_token) so sum/first/last/max/+bpt extras mutants all
    // differ, and a small part term so part_charge proves the
    // PART_CHARGE.max() law on both sides (the second call uses a part
    // term above it). The wide plan is 65536 rows, so wide_staging
    // (128 MiB) differs from live_margin (FIXED_ADMISSION_MARGIN) and
    // neither can print for the other.
    let extras = [
        Extra {
            name: "device-prefix-delta",
            fixed: 65_863_680,
            bytes_per_token: 0,
        },
        Extra {
            name: "second-extra",
            fixed: 1_048_576,
            bytes_per_token: 4_096,
        },
    ];
    let staging = fixed_case(262_144).staging;
    let wide = Kv8StagingPlan::for_context(65_536, 2, 256).unwrap();
    let line = ledger_line(&LedgerFigures {
        budget_post_margin: 2_802_778_112,
        live_free: 149_880_832,
        state_allocated: 2_917_991_168,
        decode_arena: 1_159_168,
        chunk_rows: 512,
        part: 3_000_000,
        arena: 61_229_056,
        extras: &extras,
        staging: &staging,
        wide: Some(&wide),
    });
    assert_eq!(
        line,
        "q36 cuda: fixed admission ledger: budget_post_margin=2802778112 live_free=149880832 \
live_margin=268435456 state=2917991168 decode_arena=1159168 arena=61229056 \
part_charge=67108864 extras_fixed=66912256 spare_min=67108864 default_staging=67108864 \
default_rows=32768 chunk_rows=512 wide_staging=134217728 wide_rows=65536"
    );
    // part above PART_CHARGE on a 16384-row stage (32 MiB, not 64 MiB):
    // the charge follows the chunk's own term, and part_charge,
    // spare_min, and default_staging are three distinct values, so
    // swapping any two of these three fails it; call 1 separates the rest.
    let narrow = Kv8StagingPlan::for_context(16_384, 2, 256).unwrap();
    let line = ledger_line(&LedgerFigures {
        budget_post_margin: 1,
        live_free: 1,
        state_allocated: 1,
        decode_arena: 1,
        chunk_rows: 512,
        part: 100_000_000,
        arena: 1,
        extras: &[],
        staging: &narrow,
        wide: None,
    });
    assert_eq!(
        line,
        "q36 cuda: fixed admission ledger: budget_post_margin=1 live_free=1 \
live_margin=268435456 state=1 decode_arena=1 arena=1 part_charge=100000000 \
extras_fixed=0 spare_min=67108864 default_staging=33554432 default_rows=16384 chunk_rows=512"
    );

    // Labels: a wide boot is NOT shrunk; a rung below the default is.
    assert_eq!(stage_label(StageChoice::Wide, 131_072, 32_768), "wide");
    assert_eq!(stage_label(StageChoice::Default, 16_384, 32_768), "shrunk");
    assert_eq!(stage_label(StageChoice::Default, 32_768, 32_768), "default");
}

#[test]
fn the_split_bound_binds_the_ladder_above_one_million_rows() {
    // kv_capacity 2,097,152: the composed floor is max(8192, ceil(kv / 128))
    // = 16,384, the only range where the split bound outranks the measured
    // performance floor. A 32,768-row default stage may halve exactly once.
    let mut case = fixed_case(262_144);
    case.facts.budget = usize::MAX;
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let kv = 2_097_152usize;

    // Live room for only 8192 rows: 16384 does not fit and the floor forbids
    // halving further, so the boot refuses (every rung's live-room text is the
    // same; the budget-exhaustion test pins the text itself).
    let free_8k = live_required(&cost, case.staging.bytes / 4);
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        free_8k,
        kv,
    )
    .unwrap_err();
    assert!(matches!(error, EngError::MemoryPlan(_)));

    // Live room for exactly 16384 rows: the ladder takes its one rung.
    let free_16k = live_required(&cost, case.staging.bytes / 2);
    let (cost, plan) = admit_default_or_shrink(&case, free_16k, kv);
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (case.staging.capacity / 2, case.staging.bytes / 2)
    );
}

#[test]
fn fixed_stage_floor_is_the_split_bound_for_every_context() {
    use crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_MAX_SPLITS;
    assert_eq!(super::prefill_driver::fixed_stage_floor(262_144), 2_048);
    assert_eq!(super::prefill_driver::fixed_stage_floor(180_224), 1_408);
    // Ceiling division at a non-multiple of the split width.
    assert_eq!(super::prefill_driver::fixed_stage_floor(131_073), 1_025);
    for capacity in [32_768, 131_072, 180_224, 196_608, 262_144] {
        let floor = super::prefill_driver::fixed_stage_floor(capacity);
        assert!(
            capacity.div_ceil(floor) <= CPF_ATTN_MAX_SPLITS,
            "capacity {capacity} floor {floor} exceeds the split bound"
        );
    }
    // The production floor composes the split bound with the performance
    // floor; no rung below either bound may be admitted. For every shipped
    // context the measured performance floor (8192) is the binding side; the
    // split bound binds the composition itself only above 1,048,576 rows (the
    // ladder test above pins that range end to end).
    assert_eq!(floor_for(131_073), 8_192);
    assert_eq!(floor_for(180_224), 8_192);
    assert_eq!(floor_for(393_216), 8_192);
    assert_eq!(
        super::prefill_driver::fixed_stage_min_rows(2_097_152),
        16_384
    );
}

#[test]
fn wide_refused_default_refused_the_ladder_admits_the_rung() {
    // The shipped 262144 path with CPF_STAGE_128K=1: a wide stage is offered,
    // the live snapshot refuses it, refuses the default stage too, and the
    // ladder still admits the 16384 rung as the default-stage choice.
    let mut case = fixed_case(262_144);
    case.facts.budget = usize::MAX;
    let wide = Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, 2, 256).unwrap();
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let free = live_required(&cost, case.staging.bytes / 2);
    let (cost, plan, choice) = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        free,
        case.context,
    )
    .expect("a refused wide and default must still boot on the rung");
    assert_eq!(cost.m, case.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (case.staging.capacity / 2, case.staging.bytes / 2)
    );
    assert_eq!(choice, StageChoice::Default);
}

#[test]
fn fixed_admission_admits_the_full_180224_plan_at_the_observed_budget() {
    let mut case = fixed_case(180_224);
    case.facts.budget = budget_at_observed_free(&case);
    let (cost, plan) = admit_default_or_shrink(&case, usize::MAX, case.context);
    assert_eq!(
        (cost.m, plan.capacity, plan.bytes),
        (case.facts.cap, case.staging.capacity, case.staging.bytes)
    );
}

#[test]
fn fixed_admission_roomy_card_keeps_the_32k_stage_at_262144() {
    let case = fixed_case(262_144);
    let (cost, plan) = admit_default_or_shrink(&case, usize::MAX, case.context);
    assert_eq!(
        (cost.m, plan.capacity, plan.bytes),
        (case.facts.cap, case.staging.capacity, case.staging.bytes)
    );
}

#[test]
fn fixed_admission_shrinks_the_stage_when_the_budget_is_tight() {
    // A baseline shift leaves room for half the stage but not the whole: the boot must
    // degrade the stage, not refuse. The budget pins the charges exactly at the half
    // stage, so the ladder lands on 16384 rows deterministically.
    let mut full = fixed_case(262_144);
    // Strip the calibration headroom too: the snapshot leaves room for exactly the half stage.
    let base = budget_at_observed_free(&full) - full.staging.bytes - SPARE_MIN - 977_141_760;
    full.facts.budget = base + SPARE_MIN + full.staging.bytes / 2;
    let (cost, plan) = admit_default_or_shrink(&full, usize::MAX, full.context);
    assert_eq!(cost.m, full.facts.cap);
    assert_eq!(
        (plan.capacity, plan.bytes),
        (full.staging.capacity / 2, full.staging.bytes / 2)
    );
}

#[test]
fn fixed_admission_admits_at_the_floor_when_only_the_floor_fits() {
    // Pins the composed floor itself — which at a 32,768-row default stage IS the
    // quarter stage, so this doubles as the live-room quarter verdict: when live
    // room fits the floor stage and nothing larger, the ladder stops exactly
    // there.
    let mut full = fixed_case(262_144);
    full.facts.budget = usize::MAX;
    let cost = full
        .costs
        .iter()
        .find(|cost| cost.m == full.facts.cap)
        .unwrap()
        .to_owned();
    let floor = floor_for(full.context);
    let floor_staging = full.staging.bytes * floor / full.staging.capacity;
    let free = live_required(&cost, floor_staging);
    let (cost, plan) = admit_default_or_shrink(&full, free, full.context);
    assert_eq!(cost.m, full.facts.cap);
    assert_eq!((plan.capacity, plan.bytes), (floor, floor_staging));
}

#[test]
fn fixed_admission_refuses_one_byte_below_the_floor_rung() {
    // One byte under the floor must still refuse. The live-room error text
    // names no stage bytes and every rung shares it, so this pins termination
    // at the floor and the error class.
    let mut full = fixed_case(262_144);
    full.facts.budget = usize::MAX;
    let cost = full
        .costs
        .iter()
        .find(|cost| cost.m == full.facts.cap)
        .unwrap()
        .to_owned();
    let floor = floor_for(full.context);
    let floor_staging = full.staging.bytes * floor / full.staging.capacity;
    let free = live_required(&cost, floor_staging) - 1;
    let error = admit_fixed_staged(
        &full.facts,
        &full.extras,
        &full.costs,
        full.staging,
        None,
        free,
        full.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the floor refusal must surface, got {error:?}")
    };
    assert_eq!(
        text,
        format!(
            "fixed prefill admission: {free} live bytes cannot retain {} rows, staging and the margin",
            cost.m
        )
    );
}

#[test]
fn fixed_admission_refuses_below_the_smallest_stage_with_the_original_text() {
    let mut starved = fixed_case(262_144);
    // Below even the smallest stage rung: the floor must keep the full-stage error verbatim.
    starved.facts.budget = starved.facts.budget.saturating_sub(2 << 30);
    let error = admit_fixed_staged(
        &starved.facts,
        &starved.extras,
        &starved.costs,
        starved.staging,
        None,
        usize::MAX,
        starved.context,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the floor must refuse with MemoryPlan, got {error:?}")
    };
    assert_eq!(
        text,
        format!(
            "fixed prefill admission: {} rows and {} staging bytes do not fit the budget",
            starved.facts.cap, starved.staging.bytes
        )
    );
}

#[test]
fn tiered_4096_requires_opt_in_and_retains_admission_reserves() {
    use super::prefill_driver::tiered_large_cap;
    use super::prefill_plan::chunk_cost;
    use std::ffi::OsStr;
    for (value, eligible, cap, expected) in [
        (None, true, 2048, Some(2048)),
        (Some("0"), true, 2048, Some(2048)),
        (Some("1"), true, 2048, Some(4096)),
        (Some("1"), false, 2048, None),
        (Some("1"), true, 1024, None),
        (Some("2"), true, 2048, None),
    ] {
        assert_eq!(
            tiered_large_cap(value.map(OsStr::new), eligible, cap).ok(),
            expected
        );
    }
    let dims = common_tests::dims();
    let mut costs = chunk_costs(&dims, 68).unwrap();
    assert!(
        chunk_cost(&dims, 4096, 68).is_err(),
        "an all-M FP16 partial does not fit"
    );
    let large = super::prefill_driver::tiered_large_cost(&dims, 68).unwrap();
    let base = costs[0];
    assert!(large.arena > base.arena);
    costs.insert(0, large);
    let extras = [Extra {
        name: "partial",
        fixed: 134_217_728,
        bytes_per_token: 8192,
    }];
    let state = 123_456;
    let decode = 654_321;
    let need = state
        + decode
        + large.arena
        + large.part.max(PART_CHARGE)
        + extras[0].fixed
        + extras[0].bytes_per_token * 4096
        + SPARE_MIN;
    let facts = AdmitFacts {
        budget: need,
        decode_arena: decode,
        state_planned: state,
        state_allocated: state,
        cap: 4096,
    };
    assert_eq!(admit_m_sess(&facts, &extras, &costs).unwrap(), Some(4096));
    assert_eq!(
        admit_m_sess(
            &AdmitFacts {
                budget: need - 1,
                ..facts
            },
            &extras,
            &costs
        )
        .unwrap(),
        Some(2048)
    );
    assert_eq!(
        admit_m_sess(&AdmitFacts { cap: 2048, ..facts }, &extras, &costs).unwrap(),
        Some(2048)
    );
    assert!(
        admit_m_sess(
            &AdmitFacts {
                state_allocated: state - 1,
                ..facts
            },
            &extras,
            &costs
        )
        .is_err()
    );
    println!(
        "4096 arena {} bytes versus 2048 arena {}; extra arena plus tiered partial {} bytes",
        large.arena,
        base.arena,
        large.arena - base.arena + 2048 * 8192
    );
}

#[test]
fn tiered_4096_chunks_preserve_tokens_and_the_8k_boundary() {
    use super::executor::piece_of;
    use super::prefill_driver::drive_tiered_wide_chunks;
    for (start, total) in [
        (0, 7903),
        (4095, 3808),
        (8191, 34),
        (8192, 4096),
        (73720, 32),
    ] {
        let tokens: Vec<u32> = (0..total as u32).collect();
        let mut seen = Vec::new();
        let mut at = start;
        while at < start + total {
            let size = piece_of(&ChunkVerdict::Ready { m_sess: 4096 }, at)
                .unwrap()
                .min(start + total - at);
            assert!(size > 0 && size <= 4096);
            if at >= 8192 {
                assert!(size <= 2048);
            }
            let chunk = &tokens[at - start..at - start + size];
            drive_tiered_wide_chunks(at, chunk, 4096, false, 0, |step, rows| {
                let (p0, count) = match step {
                    PrefillStep::Feed { p0 } => (p0, 1),
                    PrefillStep::Chunk { p0, rows } => (p0, rows),
                };
                assert_eq!(p0, start + seen.len());
                assert_eq!(count, rows.len());
                if count > 2048 {
                    assert!(p0 + count <= 8192);
                }
                seen.extend_from_slice(rows);
                Ok(())
            })
            .unwrap();
            at += size;
        }
        assert_eq!(seen, tokens);
    }
    let steps =
        super::prefill_driver::coalesce_tiered_deep(chunk_list(0, 7903, 4096).unwrap(), 4096, 0);
    assert_eq!(
        steps,
        vec![
            PrefillStep::Chunk { p0: 0, rows: 4096 },
            PrefillStep::Chunk {
                p0: 4096,
                rows: 3807
            }
        ]
    );
}
