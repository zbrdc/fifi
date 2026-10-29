//! Host tests of the chunked-prefill driver's pure pieces: the admission inputs and their
//! planned-against-allocated equality, the cap, decode-arena and extras terms of the admission, the
//! pre-arena verdict, the zero spans of the arena fill, the gate's fifth condition, and the start
//! line of the prefill path. Real dimensions from `common_tests`; no device anywhere — the device
//! rows are in `prefill_step_device_tests`.

use super::common_tests;
use super::dims::{KvMode, StepOptions};
use super::prefill_driver::{
    AdmitFacts, FIXED_ADMISSION_MARGIN, FULL_STAGE_TIER_ROWS, Kv8StagingPlan, StageChoice,
    ZERO_CHUNK, admit_fixed, admit_fixed_staged, admit_guard, admit_m_sess, admit_with_staging,
    drive_chunks, fixed_chunk_rows, fixed_staging_rows, needs_arena, require_fixed_room,
    wide_stage_candidate, zero_spans,
};
use super::prefill_plan::{
    ChunkCost, Extra, M_LADDER, PART_CHARGE, PrefillStep, SPARE_MIN, chunk_costs, chunk_list,
    kv_bytes, state_bytes,
};
use super::start::prefill_start_line;
use super::state_layout::StateLayout;
use super::step::{ChunkFacts, ChunkVerdict, chunk_verdict};
use super::step_plan::{KernelScratch, StepPlan};
use crate::EngError;
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
    // A budget that admits 2048 with cap 2048 gives 2048 ...
    assert_eq!(admits(boundary(0), 0, M_LADDER[0], &[]), Some(2048));
    // ... and with cap 128 gives 128 (the cap narrows the ladder).
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
fn fixed_rows_and_stage_are_context_bound_not_free_memory_bound() {
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
    case.facts.budget = case.facts.budget.saturating_sub(1 << 30);
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        Some(wide),
        FIXED_TEST_FREE,
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
    )
    .expect("exactly the wide need admits the wide stage");
    assert_eq!(
        (plan.capacity, choice),
        (FULL_STAGE_TIER_ROWS, StageChoice::Wide)
    );
}

#[test]
fn the_default_stage_still_refuses_when_only_live_room_misses() {
    // The pre-existing invariant, now enforced inside the selection: without room for
    // the default stage the boot refuses with today's text, wide candidate or not.
    let case = fixed_case(180_224);
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap()
        .to_owned();
    let free = live_required(&cost, case.staging.bytes) - 1;
    let error = admit_fixed_staged(
        &case.facts,
        &case.extras,
        &case.costs,
        case.staging,
        None,
        free,
    )
    .unwrap_err();
    let EngError::MemoryPlan(text) = error else {
        panic!("the live-room refusal must surface, got {error:?}")
    };
    let cost = case
        .costs
        .iter()
        .find(|cost| cost.m == case.facts.cap)
        .unwrap();
    assert_eq!(
        text,
        format!(
            "fixed prefill admission: {free} live bytes cannot retain {} rows, staging and the margin",
            cost.m
        )
    );
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
