//! Host tests of the step plan: sizes, lifetimes, aliasing, no overlap between live tensors, and
//! the device footprint at 32K, 128K and 256K.

use super::{
    state::{FlashGeometry, StatePlan},
    state_tests::real,
    step_plan::{
        DEFAULT_QSA_SPLITS, MAX_STEP_ROWS, PHASES, Phase, PlannedSlot, Site, Slot, StepOptions,
        StepPlan, footprint,
    },
};
use crate::{EngError, family::flash_next::moe::context_budget::KvFormat};

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;

fn plan_at(g: &FlashGeometry, rows: usize, positions: usize) -> StepPlan {
    StepPlan::new(g, rows, positions, &StepOptions::default()).expect("the step plan")
}

fn slot(plan: &StepPlan, slot: Slot) -> &PlannedSlot {
    &plan.slots()[plan
        .index(slot)
        .unwrap_or_else(|| panic!("{slot:?} is planned"))]
}

fn live(slot: &PlannedSlot, phase: Phase) -> bool {
    slot.first <= phase as usize && phase as usize <= slot.last
}

#[test]
fn the_phase_table_is_in_order() {
    for (position, phase) in PHASES.iter().enumerate() {
        assert_eq!(*phase as usize, position);
    }
}

#[test]
fn the_sizes_of_one_row_follow_the_geometry() {
    let (_, g) = real();
    let plan = plan_at(&g, 1, 32_768);
    let bytes = |s| slot(&plan, s).bytes;
    // The hyper stream: 4 x 2560 f32 per row.
    assert_eq!(bytes(Slot::Stream), 40 * KIB);
    assert_eq!(bytes(Slot::Mixed(Site::Attn)), 2560 * 4);
    assert_eq!(bytes(Slot::Gate(Site::Attn)), 4 * 4);
    assert_eq!(bytes(Slot::HyperNormed(Site::Mlp)), 40 * KIB);
    assert_eq!(bytes(Slot::HyperLow(Site::Mlp)), 320 * 4);
    assert_eq!(bytes(Slot::DeltaProj), (10240 + 6144 + 96) * 4);
    assert_eq!(bytes(Slot::DeltaConv), 10240 * 4);
    assert_eq!(bytes(Slot::DeltaCore), 6144 * 4);
    assert_eq!(bytes(Slot::AttnProj), (12288 + 1024 + 640) * 4);
    assert_eq!(bytes(Slot::AttnQ), 24 * 256 * 4);
    assert_eq!(bytes(Slot::IndexQueries), 4 * 128 * 4);
    assert_eq!(bytes(Slot::Scores), 8192 * 4);
    assert_eq!(bytes(Slot::Chosen), 512 * 4);
    assert_eq!(bytes(Slot::AttnPart), 24 * DEFAULT_QSA_SPLITS * 258 * 4);
    assert_eq!(bytes(Slot::RouterLogits), 513 * 4);
    assert_eq!(bytes(Slot::TopkId), 10 * 4);
    assert_eq!(bytes(Slot::ExpertGu), 10 * 1280 * 4);
    assert_eq!(bytes(Slot::ExpertAct), 10 * 640 * 4);
    assert_eq!(bytes(Slot::ExpertDown), 10 * 2560 * 4);
    assert_eq!(bytes(Slot::Logits), 248_320 * 4);
    // The rotated inputs of the 20 gate and up pairs and the lm_head partial are the largest.
    assert_eq!(bytes(Slot::Exl3Xh), 20 * 2560 * 4);
    assert_eq!(bytes(Slot::Exl3Z), 248_320 * 4);
    assert_eq!(bytes(Slot::Exl3Flag), 4);
}

#[test]
fn the_exl3_scratch_spans_the_chunk_at_every_width() {
    let (_, g) = real();
    // Independent of exl3_scratch's loop: at every width the rotated-input face is the routed
    // job table's one launch over rows * 10 pairs (its input width is hidden = 2560), and the
    // partial face is lm_head's (rows, 248_320) partial - every other launch (the shared
    // expert's 2 * rows groups, DeltaNet's conv/value, QSA's projections, PLE's keys) stays
    // under both at these widths. Rows 1 keeps the 20 gate/up pairs of the grouped launch.
    for (rows, pairs) in [(1usize, 20usize), (32, 320), (64, 640)] {
        let plan = plan_at(&g, rows, 32_768);
        let bytes = |s| slot(&plan, s).bytes;
        assert_eq!(bytes(Slot::Exl3Xh), pairs * 2560 * 4, "xh at rows {rows}");
        assert_eq!(bytes(Slot::Exl3Z), rows * 248_320 * 4, "z at rows {rows}");
    }
}

#[test]
fn sizes_grow_with_rows_and_the_context_only_where_they_should() {
    let (_, g) = real();
    let one = plan_at(&g, 1, 32_768);
    let four = plan_at(&g, 4, 32_768);
    assert_eq!(
        slot(&four, Slot::Stream).bytes,
        4 * slot(&one, Slot::Stream).bytes
    );
    assert_eq!(
        slot(&four, Slot::Logits).bytes,
        4 * slot(&one, Slot::Logits).bytes
    );
    let long = plan_at(&g, 1, 262_144);
    for planned in one.slots() {
        let other = slot(&long, planned.slot);
        match planned.slot {
            Slot::Scores => assert_eq!(other.bytes, 8 * planned.bytes),
            Slot::AttnPart => assert_eq!(other.bytes, planned.bytes, "the part is per split"),
            _ => assert_eq!(other.bytes, planned.bytes, "{:?}", planned.slot),
        }
    }
}

#[test]
fn lifetimes_cover_producers_and_consumers() {
    let (_, g) = real();
    let plan = plan_at(&g, 1, 32_768);
    let span = |s| {
        let p = slot(&plan, s);
        (p.first, p.last)
    };
    let n = |p: Phase| p as usize;
    // The stream is born with the embedding and read by the final mixer.
    assert_eq!(span(Slot::Stream), (n(Phase::Embed), n(Phase::FinalMix)));
    // A gate lives from its read to its apply, one per site, so a read never overwrites a gate
    // whose apply is still to come.
    assert_eq!(
        span(Slot::Gate(Site::Attn)),
        (n(Phase::HyperReadA), n(Phase::HyperApplyA))
    );
    assert_eq!(
        span(Slot::Gate(Site::Mlp)),
        (n(Phase::HyperReadM), n(Phase::HyperApplyM))
    );
    assert!(plan.index(Slot::Gate(Site::Final)).is_none());
    // Mixed rows live to the last reader of the block input.
    assert_eq!(
        span(Slot::Mixed(Site::Attn)),
        (n(Phase::HyperReadA), n(Phase::MixerIn))
    );
    assert_eq!(span(Slot::Mixed(Site::Mlp)).1, n(Phase::Combine));
    // The QSA chain.
    assert_eq!(span(Slot::Scores), (n(Phase::Score), n(Phase::Select)));
    assert_eq!(span(Slot::Chosen), (n(Phase::Select), n(Phase::AttnCore)));
    assert_eq!(
        span(Slot::AttnProj).1,
        n(Phase::AttnCore),
        "the gates are read by the attention"
    );
    assert_eq!(
        span(Slot::DeltaProj).1,
        n(Phase::DeltaRule),
        "z, beta and alpha are read by the scan"
    );
    // The n-gram row survives every layer before the PLE block (see the hazard test).
    assert_eq!(
        span(Slot::PleEmbed),
        (n(Phase::StepBegin), n(Phase::FinalMix))
    );
    // The EXL3 flag is cleared at the start and folded in before the argmax.
    assert_eq!(
        span(Slot::Exl3Flag),
        (n(Phase::StepBegin), n(Phase::Argmax))
    );
    // The block outputs live from the out-projection or combine to the apply.
    assert_eq!(
        span(Slot::BlockOut(Site::Attn)),
        (n(Phase::MixerOut), n(Phase::HyperApplyA))
    );
    assert_eq!(
        span(Slot::BlockOut(Site::Mlp)),
        (n(Phase::Combine), n(Phase::HyperApplyM))
    );
    // The logits are read by the argmax.
    assert_eq!(span(Slot::Logits), (n(Phase::LmHead), n(Phase::Argmax)));
    for planned in plan.slots() {
        assert!(planned.first <= planned.last, "{:?}", planned.slot);
        assert!(planned.last < PHASES.len(), "{:?}", planned.slot);
    }
}

#[test]
fn tensors_live_together_never_overlap() {
    let (_, g) = real();
    for (rows, positions) in [(1, 32_768), (1, 262_144), (8, 131_072)] {
        let plan = plan_at(&g, rows, positions);
        let placed: Vec<(PlannedSlot, usize)> = plan
            .slots()
            .iter()
            .map(|p| (*p, plan.placement(p.slot).unwrap().0))
            .collect();
        for (i, (a, a_off)) in placed.iter().enumerate() {
            assert_eq!(a_off % 256, 0, "{:?}", a.slot);
            assert!(a_off + a.bytes <= plan.arena_bytes(), "{:?}", a.slot);
            for (b, b_off) in &placed[i + 1..] {
                let together = a.first <= b.last && b.first <= a.last;
                let apart = a_off + a.bytes <= *b_off || b_off + b.bytes <= *a_off;
                assert!(!together || apart, "{:?} and {:?} overlap", a.slot, b.slot);
            }
        }
    }
}

#[test]
fn tensors_of_different_phases_share_addresses() {
    let (_, g) = real();
    let plan = plan_at(&g, 1, 32_768);
    assert!(plan.arena_bytes() < plan.unaliased_bytes().unwrap());
    // At every phase the live tensors fit the arena, and some phase sets are smaller than it.
    // The two gates are never live together, so the planner is free to share an address; they
    // only need to stay apart where they overlap in time, which the overlap test pins.
    let (a, b) = (
        slot(&plan, Slot::Gate(Site::Attn)),
        slot(&plan, Slot::Gate(Site::Mlp)),
    );
    assert!(a.last < b.first, "the two gates are never live together");
    // The PLE scratch is born and dead in one phase and is aliased by a later tensor.
    let ple_scratch = plan.placement(Slot::PleKeys).unwrap().0;
    let later = plan
        .slots()
        .iter()
        .filter(|p| p.first > Phase::Ple as usize)
        .any(|p| plan.placement(p.slot).unwrap().0 == ple_scratch);
    assert!(later, "something after the PLE phase reuses its scratch");
    // A DeltaNet tensor and a QSA tensor, never live in the same layer, may share an address.
    let delta_conv = slot(&plan, Slot::DeltaConv);
    let attn_q = slot(&plan, Slot::AttnQ);
    assert!(!live(delta_conv, Phase::AttnPrep) && !live(attn_q, Phase::DeltaConv));
}

#[test]
fn the_decode_arena_is_a_few_megabytes() {
    let (_, g) = real();
    let plan = plan_at(&g, 1, 32_768);
    // Logits, the lm_head partial and the activation rows dominate; the weights and the state are
    // not in this arena.
    let total = plan.arena_bytes();
    assert!(total > 2 * 248_320 * 4, "{total}");
    assert!(total < 8 * MIB, "{total}");
}

#[test]
fn rows_and_splits_outside_the_kernels_are_refused() {
    let (_, g) = real();
    for rows in [0, MAX_STEP_ROWS + 1] {
        let error = StepPlan::new(&g, rows, 4096, &StepOptions::default()).expect_err("rows");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    }
    for qsa_splits in [0, 129] {
        let error = StepPlan::new(&g, 1, 4096, &StepOptions { qsa_splits }).expect_err("splits");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    }
    let error = StepPlan::new(&g, 1, 0, &StepOptions::default()).expect_err("context");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
}

#[test]
fn the_footprint_at_32k_128k_and_256k() {
    let (_, g) = real();
    let options = StepOptions::default();
    let mut totals = Vec::new();
    for positions in [32_768usize, 131_072, 262_144] {
        let f = footprint(&g, positions, 1, &options, KvFormat::F16).unwrap();
        let state = StatePlan::new(&g, positions, KvFormat::F16)
            .unwrap()
            .bytes()
            .unwrap();
        assert_eq!(f.state, state);
        let step = plan_at(&g, 1, positions);
        assert_eq!(f.scratch_bytes, step.arena_bytes());
        assert_eq!(
            f.total_bytes().unwrap(),
            state.arena_bytes + step.arena_bytes()
        );
        assert_eq!(
            f.non_context_bytes().unwrap(),
            state.fixed_bytes() + step.arena_bytes()
        );
        totals.push(f.total_bytes().unwrap());
    }
    // The table: state at 256K is 6.0 GiB of K/V, 0.375 GiB of keys, 112 MiB of DeltaNet state,
    // plus rings and words below 1 MiB.
    let f = footprint(&g, 262_144, 1, &options, KvFormat::F16).unwrap();
    let state_gib = f.state.arena_bytes as f64 / (1u64 << 30) as f64;
    assert!((6.4..6.6).contains(&state_gib), "{state_gib}");
    // The scratch is a few MiB and grows by about one byte per position: it does not move the
    // expert hot set by a slot (one slot is 1.8 MiB x 48 layers).
    let growth = totals[2] - totals[0] - (262_144 - 32_768) * 26_112;
    assert!(growth < MIB, "scratch growth {growth}");
    assert!(totals[0] < totals[1] && totals[1] < totals[2]);
}

#[test]
fn the_state_and_step_plans_agree_on_the_context() {
    let (_, g) = real();
    let state = StatePlan::new(&g, 65_536, KvFormat::F16).unwrap();
    let step = plan_at(&g, 1, 65_536);
    // One score row per completed block, one cache block per `ratio` positions.
    assert_eq!(
        slot(&step, Slot::Scores).bytes,
        state.bytes().unwrap().cap_blocks * 4
    );
}
