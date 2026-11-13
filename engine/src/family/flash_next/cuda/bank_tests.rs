//! Tests of the expert bank: the geometry, the host budget at the sizes of the 3.05 bpw pack on the
//! 62 GiB host, the popularity profile, the replace-on-miss policy, the host places (no record is
//! ever lost or duplicated), the filler rule, and, ignored, the replay of the T0 route trace and
//! the device bank. Host tests need no model and no GPU.

use super::bank::*;
use crate::{
    EngError,
    family::{
        flash_next::moe::{
            slot_bank::SlotBankLayout,
            tier_plan::{TierInputs, TierPlan, TierShape, position_bytes},
        },
        qwen36::tiered::{
            budget::{HostBudget, KvPlan, VramBudget},
            classify::classify_reference,
        },
    },
};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const EXPERT_BYTES: u64 = 1_881_600;
const TRUNK_BYTES: u64 = 3_854_000_000;
const STATE_BYTES: u64 = 36 * 3 * MIB;
/// `MemTotal` of this host (65,467,544 kB): 62.43 GiB.
const HOST_TOTAL: u64 = 65_467_544 * 1024;
const COUNTS: &str =
    include_str!("../../../../tests/fixtures/flash_next_routing/t0-decode-counts.tsv");

fn message(error: EngError) -> String {
    error.describe()
}

fn real_layout() -> SlotBankLayout {
    SlotBankLayout::new(2560, 640, [48, 48, 48]).unwrap()
}

fn shape() -> TierShape {
    TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES).unwrap()
}

/// A plan with exactly `slots` slots per layer and room for the cold experts.
fn plan_with_slots(slots: usize) -> TierPlan {
    TierPlan::new(&TierInputs {
        shape: shape(),
        vram: VramBudget {
            usable: 1 + slots as u64 * shape().slot_cost(),
            non_expert: 0,
            state: 0,
            arena: 0,
            desktop_reserve: 0,
        },
        kv: KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
        host: HostBudget {
            total: 256 * GIB,
            non_reclaimable: 0,
            engine_heap: 0,
            staging: 0,
            margin: 0,
        },
    })
    .unwrap()
}

/// The tier plan of the 24 GiB card at a context, as the tier plan's own tests build it.
fn plan_at(positions: u64) -> TierPlan {
    let kv_position = position_bytes(12, 2, 256, 2, (128, 4)).unwrap();
    TierPlan::new(&TierInputs::cost_model_3090(
        shape(),
        (TRUNK_BYTES, STATE_BYTES),
        KvPlan {
            positions,
            bytes_per_position: kv_position,
        },
    ))
    .unwrap()
}

fn geometry_of(plan: &TierPlan) -> BankGeometry {
    BankGeometry::from_plan(plan, &real_layout()).unwrap()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn host(available: u64) -> HostMemory {
    HostMemory {
        total: HOST_TOTAL,
        available,
    }
}

// ---------------------------------------------------------------------------------------------
// Geometry.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_bank_needs_a_slot_for_the_filler_beyond_the_picks() {
    assert!(BankGeometry::new(48, 512, 10, 10, 1_881_600).is_err());
    let error = message(BankGeometry::new(48, 512, 10, 10, 1_881_600).unwrap_err());
    assert!(error.contains("11 or more"), "{error}");
    assert!(BankGeometry::new(48, 512, 10, 11, 1_881_600).is_ok());
    assert!(BankGeometry::new(48, 512, 10, 513, 1_881_600).is_err());
    assert!(BankGeometry::new(48, 70_000, 10, 100, 1_881_600).is_err());
    // 65,535 is the policy's empty-slot sentinel, so a layer holds at most 65,535 experts.
    assert!(BankGeometry::new(48, 65_536, 10, 100, 1_881_600).is_err());
    assert!(BankGeometry::new(48, 65_535, 10, 100, 1_881_600).is_ok());
    assert!(BankGeometry::new(48, 512, 10, 100, 1_881_601).is_err());
    assert!(BankGeometry::new(0, 512, 10, 100, 1_881_600).is_err());
}

#[test]
fn the_geometry_takes_the_plan_and_refuses_another_record_size() {
    let plan = plan_with_slots(143);
    let geometry = geometry_of(&plan);
    assert_eq!(
        (
            geometry.layers,
            geometry.experts,
            geometry.top_k,
            geometry.slots
        ),
        (48, 512, 10, 143)
    );
    assert_eq!(geometry.offset(142), 142 * 1_881_600);
    assert_eq!(geometry.device_bytes_per_layer(), 143 * 1_881_600);
    let other = SlotBankLayout::new(2560, 640, [80, 80, 80]).unwrap();
    let error = message(BankGeometry::from_plan(&plan, &other).unwrap_err());
    assert!(error.contains("tier plan counts"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// The host budget.
// ---------------------------------------------------------------------------------------------

#[test]
fn meminfo_gives_the_machine_and_the_os_floor() {
    let text = "MemTotal:       65467544 kB\nMemFree:  1 kB\nMemAvailable:   50000000 kB\n";
    let memory = HostMemory::from_meminfo(text).unwrap();
    assert_eq!(memory.total, HOST_TOTAL);
    assert_eq!(memory.available, 50_000_000 * 1024);
    assert_eq!(
        memory.os_floor(),
        8 * GIB,
        "a tenth of 62 GiB is under the 8 GiB floor"
    );
    let big = HostMemory {
        total: 200 * GIB,
        available: 100 * GIB,
    };
    assert_eq!(big.os_floor(), 20 * GIB);
    for bad in [
        "MemTotal: 10 kB\n",
        "MemAvailable: 10 kB\n",
        "MemTotal: 10 MB\nMemAvailable: 1 kB\n",
        "MemTotal: 10 kB\nMemAvailable: 11 kB\n",
        "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: x kB\nMemAvailable: 1 kB\n",
    ] {
        assert!(HostMemory::from_meminfo(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn the_pinned_store_is_the_cold_experts_plus_one_spare_place_per_layer() {
    // 62.43 GiB host with 55 GiB available. Cold experts and the spare place, in GiB, against the
    // cost model's 31.04 / 27.59 / 25.6 GiB of cold experts at 143 / 184 / about 208 slots.
    for (slots, cold_gib) in [(143usize, 31.04), (184, 27.59), (208, 25.6)] {
        let geometry = geometry_of(&plan_with_slots(slots));
        let plan = HostPinPlan::new(
            &geometry,
            HostMode::Exclusive,
            &host(55 * GIB),
            &HostTerms::DEFAULT,
        )
        .unwrap();
        let cold = 48 * (512 - slots as u64) * EXPERT_BYTES;
        assert_eq!(plan.places_per_layer, 512 - slots + 1);
        assert_eq!(plan.store_bytes, cold + 48 * EXPERT_BYTES);
        assert!(
            (gib(cold) - cold_gib).abs() < 0.05,
            "{slots} slots: {} GiB cold",
            gib(cold)
        );
        // The spare places cost 48 records, 0.084 GiB: the range is 25.6 to 31.1 GiB.
        assert!((gib(plan.store_bytes) - gib(cold) - 0.084).abs() < 0.001);
        assert_eq!(plan.os_floor, 8 * GIB);
        assert_eq!(
            plan.process_bytes,
            plan.store_bytes + plan.scratch_bytes + plan.ngram_staging_bytes + 3 * GIB + 2 * GIB
        );
        // 55 GiB available minus the store, the scratch, 3 + 2 GiB and the 8 GiB floor.
        assert_eq!(
            plan.page_cache_left,
            55 * GIB - plan.process_bytes - 8 * GIB
        );
        assert!(plan.page_cache_left >= 8 * GIB);
    }
}

#[test]
fn the_budget_at_the_contexts_of_the_tier_plan() {
    // 262,144 positions: 143 slots, the largest store.
    let long = geometry_of(&plan_at(262_144));
    assert!((140..=146).contains(&long.slots), "{} slots", long.slots);
    let plan = HostPinPlan::new(
        &long,
        HostMode::Exclusive,
        &host(55 * GIB),
        &HostTerms::DEFAULT,
    )
    .unwrap();
    assert!(
        (31.0..31.3).contains(&gib(plan.store_bytes)),
        "{} GiB",
        gib(plan.store_bytes)
    );
    // The page cache left for the 30.4 GiB n-gram table: 55 available - 8 floor - 31.2 - 5 = 10.8 GiB.
    assert!(
        (10.5..11.1).contains(&gib(plan.page_cache_left)),
        "{} GiB",
        gib(plan.page_cache_left)
    );
    let short = geometry_of(&plan_at(32_768));
    let plan = HostPinPlan::new(
        &short,
        HostMode::Exclusive,
        &host(55 * GIB),
        &HostTerms::DEFAULT,
    )
    .unwrap();
    assert!(
        (25.0..26.0).contains(&gib(plan.store_bytes)),
        "{} GiB at {} slots",
        gib(plan.store_bytes),
        short.slots
    );
}

/// The terms with the 8 GiB page-cache floor the plan used before the n-gram reads were measured, to
/// exercise the floor logic itself.
fn floor_terms() -> HostTerms {
    HostTerms {
        ngram_cache_min: 8 * GIB,
        ..HostTerms::DEFAULT
    }
}

#[test]
fn an_inclusive_store_fits_the_host_by_availability_and_by_the_floor_if_one_is_set() {
    let geometry = geometry_of(&plan_with_slots(143));
    // No n-gram floor (measured: cold rows are cheap): 43.07 GiB + 86 MB staging + 3 + 2 + 8 floor
    // is 56.2 GiB, which 60 GiB available holds with 3.8 GiB of page cache left.
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Inclusive,
        &host(60 * GIB),
        &HostTerms::DEFAULT,
    )
    .unwrap();
    assert!((gib(plan.store_bytes) - 43.07).abs() < 0.01);
    assert!((3.5..4.1).contains(&gib(plan.page_cache_left)));
    // 55 GiB available does not hold it.
    let error = message(
        HostPinPlan::new(
            &geometry,
            HostMode::Inclusive,
            &host(55 * GIB),
            &HostTerms::DEFAULT,
        )
        .unwrap_err(),
    );
    assert!(error.contains("are available"), "{error}");
    // With an 8 GiB floor the 3.8 GiB left is refused.
    let error = message(
        HostPinPlan::new(
            &geometry,
            HostMode::Inclusive,
            &host(60 * GIB),
            &floor_terms(),
        )
        .unwrap_err(),
    );
    assert!(error.contains("page-cache floor"), "{error}");
}

#[test]
fn the_default_floor_is_the_measured_zero() {
    assert_eq!(HostTerms::DEFAULT.ngram_cache_min, 0);
}

#[test]
fn the_budget_refuses_when_other_lanes_hold_the_memory_and_never_shrinks() {
    // MemAvailable was 14.9 GiB with other lanes running: the 31 GiB store is refused whole.
    let geometry = geometry_of(&plan_with_slots(143));
    let error = message(
        HostPinPlan::new(
            &geometry,
            HostMode::Exclusive,
            &host(15 * GIB),
            &HostTerms::DEFAULT,
        )
        .unwrap_err(),
    );
    assert!(error.contains("are available"), "{error}");
    // Exactly enough available memory passes, one byte less refuses (the n-gram floor is judged
    // in `the_page_cache_left_follows_what_other_processes_hold`, so it is off here).
    let terms = HostTerms {
        ngram_cache_min: 0,
        ..HostTerms::DEFAULT
    };
    let ok = HostPinPlan::new(&geometry, HostMode::Exclusive, &host(55 * GIB), &terms).unwrap();
    let needed = ok.process_bytes + ok.os_floor;
    assert!(HostPinPlan::new(&geometry, HostMode::Exclusive, &host(needed), &terms).is_ok());
    assert!(HostPinPlan::new(&geometry, HostMode::Exclusive, &host(needed - 1), &terms).is_err());
}

#[test]
fn the_page_cache_left_follows_what_other_processes_hold() {
    // 46 GiB available on the 62 GiB host: the store passes the availability guard (44 needed), but
    // only about 2 GiB of page cache would remain, under an 8 GiB floor (the default floor is 0).
    let geometry = geometry_of(&plan_with_slots(143));
    let ok = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &host(55 * GIB),
        &floor_terms(),
    )
    .unwrap();
    let needed = ok.process_bytes + ok.os_floor;
    assert!(needed < 46 * GIB);
    let error = message(
        HostPinPlan::new(
            &geometry,
            HostMode::Exclusive,
            &host(46 * GIB),
            &floor_terms(),
        )
        .unwrap_err(),
    );
    assert!(error.contains("page-cache floor"), "{error}");
    // Exactly the floor above the need passes; one byte less refuses.
    let floor = floor_terms().ngram_cache_min;
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &host(needed + floor),
        &floor_terms(),
    )
    .unwrap();
    assert_eq!(plan.page_cache_left, floor);
    assert!(
        HostPinPlan::new(
            &geometry,
            HostMode::Exclusive,
            &host(needed + floor - 1),
            &floor_terms()
        )
        .is_err()
    );
}

#[test]
fn the_mirrored_ngram_figures_equal_the_feed_constants() {
    use super::feeds::{NGRAM_PAGE_CACHE_RESERVE_BYTES, ngram_staging_bytes};
    assert_eq!(
        HostTerms::DEFAULT.ngram_cache_min,
        NGRAM_PAGE_CACHE_RESERVE_BYTES
    );
    for rows in [0u64, 1, 65_536, 131_072] {
        assert_eq!(
            rows * NGRAM_STAGING_BYTES_PER_ROW,
            ngram_staging_bytes(rows)
        );
    }
}

#[test]
fn the_ngram_staging_is_a_host_transient_charged_to_the_process() {
    assert_eq!(NGRAM_STAGING_BYTES_PER_ROW, 656);
    let geometry = geometry_of(&plan_with_slots(143));
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &host(55 * GIB),
        &HostTerms::DEFAULT,
    )
    .unwrap();
    // Two 4096-token chunks of 16 rows each in flight: 2 x 65,536 rows of 656 bytes, 86 MB.
    assert_eq!(plan.ngram_staging_bytes, 2 * 65_536 * 656);
    // More rows in flight move the need byte for byte, so the availability boundary moves with them.
    let more = HostTerms {
        ngram_staging_rows: 1 << 21,
        ..HostTerms::DEFAULT
    };
    let big = HostPinPlan::new(&geometry, HostMode::Exclusive, &host(55 * GIB), &more).unwrap();
    assert_eq!(big.ngram_staging_bytes, (1 << 21) * 656);
    assert_eq!(
        big.process_bytes - plan.process_bytes,
        big.ngram_staging_bytes - plan.ngram_staging_bytes
    );
    let needed = big.process_bytes + big.os_floor;
    assert!(HostPinPlan::new(&geometry, HostMode::Exclusive, &host(needed), &more).is_ok());
    assert!(HostPinPlan::new(&geometry, HostMode::Exclusive, &host(needed - 1), &more).is_err());
    // The default terms would still pass at `needed - 1`: the staging is what moved the boundary.
    assert!(
        HostPinPlan::new(
            &geometry,
            HostMode::Exclusive,
            &host(needed - 1),
            &HostTerms::DEFAULT
        )
        .is_ok()
    );
    // A row count that overflows is refused, not wrapped.
    let absurd = HostTerms {
        ngram_staging_rows: u64::MAX,
        ..HostTerms::DEFAULT
    };
    assert!(HostPinPlan::new(&geometry, HostMode::Exclusive, &host(55 * GIB), &absurd).is_err());
}

#[test]
fn the_stream_scratch_charges_both_slots_a_full_record_each() {
    // A record above 1 MiB: a once-per-slot allowance of record + 1 MiB would have been half.
    let geometry = geometry_of(&plan_with_slots(143));
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &host(55 * GIB),
        &HostTerms::DEFAULT,
    )
    .unwrap();
    const { assert!(EXPERT_BYTES > MIB) };
    assert_eq!(plan.scratch_bytes, 2 * (EXPERT_BYTES + MIB));
}

// ---------------------------------------------------------------------------------------------
// The popularity profile.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_t0_profile_parses_to_one_thousand_three_hundred_three_tokens_of_ten_picks_per_layer() {
    let counts = parse_route_counts(COUNTS, 48, 512).unwrap();
    assert_eq!(counts.len(), 48);
    for (layer, row) in counts.iter().enumerate() {
        assert_eq!(row.len(), 512);
        assert_eq!(
            row.iter().sum::<u64>(),
            13_030,
            "layer {layer}: 1,303 tokens x 10 picks"
        );
    }
}

#[test]
fn the_profile_parser_refuses_a_wrong_shape_a_repeat_and_a_stray_line() {
    let head = "# schema=flash-next-route-counts/1\n# layers=2\n# experts=4\n";
    let ok = format!("{head}0\t1\t5\n1\t3\t2\n");
    let counts = parse_route_counts(&ok, 2, 4).unwrap();
    assert_eq!(counts, vec![vec![0, 5, 0, 0], vec![0, 0, 0, 2]]);
    assert!(
        parse_route_counts(&ok, 3, 4).is_err(),
        "layer count differs"
    );
    assert!(
        parse_route_counts(&ok, 2, 5).is_err(),
        "expert count differs"
    );
    assert!(parse_route_counts(&format!("{head}0\t4\t1\n"), 2, 4).is_err());
    assert!(parse_route_counts(&format!("{head}2\t0\t1\n"), 2, 4).is_err());
    assert!(parse_route_counts(&format!("{head}0\t1\t1\n0\t1\t2\n"), 2, 4).is_err());
    assert!(parse_route_counts(&format!("{head}0\t1\n"), 2, 4).is_err());
    assert!(parse_route_counts(&format!("{head}0\t1\t1\t9\n"), 2, 4).is_err());
    assert!(parse_route_counts("0\t1\t1\n", 2, 4).is_err(), "no header");
    assert!(
        parse_route_counts("# layers=2\n", 2, 4).is_err(),
        "no schema"
    );
    // A header alone is an empty profile of that shape; another shape or no shape is refused.
    assert_eq!(
        parse_route_counts(head, 2, 4).unwrap(),
        vec![vec![0u64; 4]; 2]
    );
    assert!(parse_route_counts(head, 3, 4).is_err());
    assert!(parse_route_counts("# schema=flash-next-route-counts/1\n", 2, 4).is_err());
}

// ---------------------------------------------------------------------------------------------
// The static fill.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_static_fill_holds_the_most_picked_experts_and_the_classifier_agrees() {
    let counts = parse_route_counts(COUNTS, 48, 512).unwrap();
    let plan = plan_with_slots(143);
    let state = ExpertBankState::new(&plan, &real_layout(), HostMode::Exclusive, &counts).unwrap();
    for layer in [0usize, 3, 47] {
        let row = state.slot_row(layer);
        let held: Vec<usize> = (0..512).filter(|&e| row[e] >= 0).collect();
        assert_eq!(held.len(), 143);
        let mut ranked: Vec<usize> = (0..512).collect();
        ranked.sort_by_key(|&e| (std::cmp::Reverse(counts[layer][e]), e));
        let mut want = ranked[..143].to_vec();
        want.sort_unstable();
        assert_eq!(
            held, want,
            "layer {layer}: the 143 most picked, ties to the lowest id"
        );
        for slot in 0..143 {
            let expert = usize::from(state.expert_in_slot(layer, slot).unwrap());
            assert_eq!(row[expert], slot as i32);
        }
        // A pick off the hot set is a miss to slot 0 with weight +0.0.
        let cold = (0..512).find(|&e| row[e] < 0).unwrap() as i32;
        let hot = held[0] as i32;
        let got = classify_reference(row, 143, &[hot, cold], &[0.6, 0.4]).unwrap();
        assert_eq!(got.slot_ids, vec![row[hot as usize], 0]);
        assert_eq!(got.weights[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(got.misses, vec![(cold, 0.4)]);
    }
    // The store holds the other 369 experts and a free place, in every layer.
    assert_eq!(state.places().occupied(0).len(), 512 - 143);
}

// ---------------------------------------------------------------------------------------------
// The policy, on a small bank.
// ---------------------------------------------------------------------------------------------

/// 2 layers, 16 experts, top-3, 5 slots; experts 0..5 start in slots 0..5, expert 0 the least
/// popular so it is the first victim (counts decrease with the expert's rank).
fn small(mode: HostMode) -> ExpertBankState<ReplaceOnMissLru> {
    let geometry = BankGeometry::new(2, 16, 3, 5, 64).unwrap();
    let row: Vec<i32> = (0..16).map(|e| if e < 5 { e } else { -1 }).collect();
    let rows = [row.as_slice(), row.as_slice()];
    // Expert e holds slot e; popularity rises with e, so slot 0 is the first victim.
    let counts: Vec<Vec<u64>> = (0..2)
        .map(|_| (0..16).map(|e| e as u64).collect())
        .collect();
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    ExpertBankState::with_policy(geometry, mode, policy).unwrap()
}

fn moved(copies: &PassCopies) -> Vec<(u16, u16)> {
    copies.ops.iter().map(|op| (op.slot, op.incoming)).collect()
}

#[test]
fn a_pass_of_hits_moves_nothing_and_counts_the_hits() {
    let mut state = small(HostMode::Exclusive);
    let copies = state.begin_pass(0, &[4, 2, 3]).unwrap();
    assert!(copies.ops.is_empty());
    assert_eq!((copies.distinct, copies.hits), (3, 3));
    let stats = state.stats();
    assert_eq!(
        (stats.passes, stats.picks, stats.hits, stats.misses),
        (1, 3, 3, 0)
    );
    assert_eq!(stats.miss_histogram[0], 1);
}

#[test]
fn a_miss_takes_the_least_recent_slot_that_the_pass_does_not_use() {
    let mut state = small(HostMode::Exclusive);
    // Recency, oldest first: slots 0 1 2 3 4 (expert = slot). Slot 0 is also the filler slot, which a
    // pass with a miss protects, so the victim is slot 1.
    let copies = state.begin_pass(0, &[9, 2, 4]).unwrap();
    assert_eq!(
        moved(&copies),
        vec![(1, 9)],
        "slot 0 is protected, slot 1 is the least recent"
    );
    assert_eq!(state.slot_row(0)[9], 1);
    assert_eq!(state.slot_row(0)[1], -1);
    // Layers are independent.
    assert_eq!(state.slot_row(1)[9], -1);
    assert_eq!(state.expert_in_slot(1, 1), Some(1));
}

#[test]
fn the_experts_of_the_pass_are_never_evicted_even_when_they_are_the_oldest() {
    // Recency, oldest first: slots 0 1 2 3 4. The pass uses 1 and 2 and misses 8; slot 0 is the filler
    // and slots 1 and 2 are the pass's own, so the victim is slot 3, the next oldest.
    let mut state = small(HostMode::Exclusive);
    let copies = state.begin_pass(0, &[1, 2, 8]).unwrap();
    assert_eq!(moved(&copies), vec![(3, 8)]);
    let copies = state.begin_pass(0, &[1, 2, 9]).unwrap();
    // Recency now (old to new): 0 4 1 2 3 -> 1 and 2 are protected, slot 0 is the filler, so slot 4.
    assert_eq!(moved(&copies), vec![(4, 9)]);
}

#[test]
fn every_miss_of_a_pass_is_admitted_with_no_swap_limit() {
    let mut state = small(HostMode::Exclusive);
    // Slot 0 stays (filler); the three misses take slots 1, 2, 3 in pick order.
    let copies = state.begin_pass(0, &[10, 11, 12]).unwrap();
    assert_eq!(moved(&copies), vec![(1, 10), (2, 11), (3, 12)]);
    assert_eq!(state.stats().misses, 3);
    assert_eq!(state.stats().layers_with_3_plus_misses, 1);
    assert_eq!(state.stats().miss_histogram[3], 1);
}

#[test]
fn slot_zero_is_protected_only_while_a_pass_misses() {
    let mut state = small(HostMode::Exclusive);
    // No miss: slot 0's expert is simply a hit or untouched; nothing moves and nothing is protected.
    state.begin_pass(0, &[1, 2, 3]).unwrap();
    // Recency: 0 4 1 2 3. A miss with picks that avoid expert 0 must not evict it (filler), so slot 4 goes.
    let copies = state.begin_pass(0, &[5, 1, 2]).unwrap();
    assert_eq!(moved(&copies), vec![(4, 5)]);
    assert_eq!(state.expert_in_slot(0, 0), Some(0));
}

#[test]
fn a_hit_makes_the_slot_the_most_recent() {
    let mut state = small(HostMode::Exclusive);
    state.begin_pass(0, &[1]).unwrap(); // slot 1 newest; recency 0 2 3 4 1
    let copies = state.begin_pass(0, &[9]).unwrap(); // victim: slot 0 is the filler, so slot 2
    assert_eq!(moved(&copies), vec![(2, 9)]);
}

#[test]
fn repeats_count_once_a_router_fault_is_skipped_and_a_bad_id_changes_nothing() {
    let mut state = small(HostMode::Exclusive);
    let copies = state.begin_pass(0, &[9, 9, -1, 2]).unwrap();
    assert_eq!((copies.distinct, copies.hits, copies.ops.len()), (2, 1, 1));
    let row_before = state.slot_row(0).to_vec();
    let stats_before = state.stats().clone();
    for bad in [16, -2, 99] {
        assert!(state.begin_pass(0, &[3, bad]).is_err(), "{bad}");
    }
    assert!(state.begin_pass(2, &[1]).is_err(), "no layer 2");
    assert_eq!(state.slot_row(0), row_before.as_slice());
    assert_eq!(state.stats(), &stats_before);
    // More distinct experts than slots cannot be protected.
    assert!(state.begin_pass(0, &[0, 1, 2, 3, 4, 5]).is_err());
    assert_eq!(state.slot_row(0), row_before.as_slice());
}

#[test]
fn admitted_experts_keep_the_newest_use_and_leave_the_recency_in_that_order() {
    let mut state = small(HostMode::Exclusive);
    // A chunk used 6 experts, the newest last; five slots keep the last five distinct, 6 twice.
    let used = [14u16, 6, 7, 8, 6, 9, 10];
    let copies = state.admit(0, &used).unwrap();
    let kept: Vec<u16> = (0..5)
        .map(|s| state.expert_in_slot(0, s).unwrap())
        .collect();
    let mut kept_sorted = kept.clone();
    kept_sorted.sort_unstable();
    assert_eq!(kept_sorted, vec![6, 7, 8, 9, 10]);
    assert_eq!(copies.ops.len(), 5, "all five were cold");
    assert_eq!(
        state.stats().passes,
        0,
        "an admission is not decode traffic"
    );
    // Slot 0 holds expert 7, the oldest use, but it is the filler slot of a pass with a miss: the
    // next oldest, expert 8, is evicted.
    assert_eq!(state.expert_in_slot(0, 0), Some(7));
    let copies = state.begin_pass(0, &[1]).unwrap();
    assert_eq!(copies.ops[0].demote.unwrap().expert, 8);
}

#[test]
fn a_policy_row_must_hold_exactly_the_slots() {
    let geometry = BankGeometry::new(1, 16, 3, 5, 64).unwrap();
    let counts = vec![vec![0u64; 16]];
    let fill = |row: Vec<i32>| {
        let policy = ReplaceOnMissLru::from_fill(geometry, &[row.as_slice()], &counts, None)?;
        ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).map(|_| ())
    };
    let row = |held: usize| -> Vec<i32> {
        (0..16)
            .map(|e| if e < held { e as i32 } else { -1 })
            .collect()
    };
    assert!(fill(row(5)).is_ok());
    assert!(fill(row(4)).is_err(), "a short row");
    assert!(fill(row(6)).is_err(), "a slot past the bank");
    // A policy that reports a short row after its own checks is caught by `with_policy`.
    struct Short(Vec<i32>);
    impl SlotTable for Short {
        fn slot_row(&self, _: usize) -> &[i32] {
            &self.0
        }
        fn expert_in_slot(&self, _: usize, _: usize) -> Option<u16> {
            None
        }
    }
    impl SlotPolicy for Short {
        fn plan_pass(&mut self, _: usize, _: &[i32]) -> Result<PassPlan, EngError> {
            Ok(PassPlan::default())
        }
        fn admit(&mut self, _: usize, _: &[u16]) -> Result<PassPlan, EngError> {
            Ok(PassPlan::default())
        }
    }
    let error = message(
        ExpertBankState::with_policy(geometry, HostMode::Exclusive, Short(row(3)))
            .err()
            .expect("refused"),
    );
    assert!(error.contains("holds 3 experts"), "{error}");
}

#[test]
fn a_poisoned_bank_refuses_every_later_pass() {
    let mut state = small(HostMode::Exclusive);
    state.begin_pass(0, &[9]).unwrap();
    assert!(!state.is_poisoned());
    state.poison();
    assert!(state.is_poisoned());
    assert!(state.begin_pass(0, &[1]).is_err());
    assert!(state.admit(0, &[1]).is_err());
}

#[test]
fn an_unfilled_start_is_refused() {
    let geometry = BankGeometry::new(1, 8, 2, 4, 64).unwrap();
    let counts = vec![vec![0u64; 8]];
    let short: Vec<i32> = vec![0, 1, 2, -1, -1, -1, -1, -1];
    assert!(ReplaceOnMissLru::from_fill(geometry, &[short.as_slice()], &counts, None).is_err());
    let twice: Vec<i32> = vec![0, 1, 2, 2, -1, -1, -1, -1];
    assert!(ReplaceOnMissLru::from_fill(geometry, &[twice.as_slice()], &counts, None).is_err());
    let outside: Vec<i32> = vec![0, 1, 2, 4, -1, -1, -1, -1];
    assert!(ReplaceOnMissLru::from_fill(geometry, &[outside.as_slice()], &counts, None).is_err());
}

// ---------------------------------------------------------------------------------------------
// Host places: the record of an expert is never lost or duplicated.
// ---------------------------------------------------------------------------------------------

/// A host model of the device and the pinned store: every record is its expert's number.
struct Sim {
    device: Vec<Vec<u16>>,
    host: Vec<Vec<Option<u16>>>,
    exclusive: bool,
}

impl Sim {
    fn new(state: &ExpertBankState<ReplaceOnMissLru>) -> Self {
        let geometry = *state.geometry();
        let places = state.mode().places_per_layer(&geometry);
        let device = (0..geometry.layers)
            .map(|layer| {
                (0..geometry.slots)
                    .map(|slot| state.expert_in_slot(layer, slot).unwrap())
                    .collect()
            })
            .collect();
        let host = (0..geometry.layers)
            .map(|layer| {
                let mut row = vec![None; places];
                for (place, expert) in state.places().occupied(layer) {
                    row[place as usize] = Some(expert);
                }
                row
            })
            .collect();
        Self {
            device,
            host,
            exclusive: state.mode() == HostMode::Exclusive,
        }
    }

    /// The ops in stream order: write-back into a free place, then the copy out of the incoming
    /// expert's place.
    fn apply(&mut self, copies: &PassCopies) {
        let (device, host) = (&mut self.device[copies.layer], &mut self.host[copies.layer]);
        for op in &copies.ops {
            if let Some(demote) = op.demote {
                assert_eq!(
                    device[usize::from(op.slot)],
                    demote.expert,
                    "the slot holds the evicted expert"
                );
                assert_eq!(
                    host[demote.dst_place as usize], None,
                    "the write-back lands on a free place"
                );
                host[demote.dst_place as usize] = Some(demote.expert);
            }
            assert_eq!(
                host[op.src_place as usize],
                Some(op.incoming),
                "the copy reads the expert's place"
            );
            device[usize::from(op.slot)] = op.incoming;
            if self.exclusive {
                host[op.src_place as usize] = None;
            }
        }
    }

    fn check(&self, state: &ExpertBankState<ReplaceOnMissLru>) {
        let experts = state.geometry().experts;
        for (layer, (device, host)) in self.device.iter().zip(&self.host).enumerate() {
            let row = state.slot_row(layer);
            for (slot, &expert) in device.iter().enumerate() {
                assert_eq!(
                    row[usize::from(expert)],
                    slot as i32,
                    "the row follows the device"
                );
            }
            for expert in 0..experts as u16 {
                let on_device = device.contains(&expert);
                let in_store = host.iter().filter(|&&held| held == Some(expert)).count();
                if self.exclusive {
                    assert_eq!(
                        usize::from(on_device) + in_store,
                        1,
                        "layer {layer} expert {expert}"
                    );
                } else {
                    assert_eq!(in_store, 1, "inclusive: expert {expert} is always stored");
                }
            }
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

fn run_random_passes(mode: HostMode) {
    let geometry = BankGeometry::new(3, 64, 6, 20, 64).unwrap();
    let row: Vec<i32> = (0..64).map(|e| if e < 20 { e } else { -1 }).collect();
    let rows = vec![row.as_slice(); 3];
    let counts = vec![vec![0u64; 64]; 3];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let mut state = ExpertBankState::with_policy(geometry, mode, policy).unwrap();
    let mut sim = Sim::new(&state);
    sim.check(&state);
    let mut rng = Lcg(7);
    for pass in 0..3000 {
        let layer = rng.next(3);
        // Squared draws skew the picks to low numbers, so hits and misses both happen.
        let picks: Vec<i32> = (0..6)
            .map(|_| {
                let u = rng.next(64);
                (u * u / 64) as i32
            })
            .collect();
        let picks = if pass % 7 == 0 {
            (0..6).map(|_| rng.next(64) as i32).collect()
        } else {
            picks
        };
        let copies = state.begin_pass(layer, &picks).unwrap();
        sim.apply(&copies);
        sim.check(&state);
        for &pick in &picks {
            assert!(
                state.slot_row(layer)[pick as usize] >= 0,
                "every pick is resident after the pass"
            );
        }
        if pass % 400 == 399 {
            let used: Vec<u16> = (0..30).map(|_| rng.next(64) as u16).collect();
            let copies = state.admit(layer, &used).unwrap();
            sim.apply(&copies);
            sim.check(&state);
        }
    }
    let stats = state.stats();
    assert_eq!(stats.passes, 3000);
    assert_eq!(stats.hits + stats.misses, stats.picks);
    assert!(stats.hits > 0 && stats.misses > 0);
    assert_eq!(stats.miss_histogram.iter().sum::<u64>(), 3000);
    match mode {
        HostMode::Exclusive => assert_eq!(stats.demotions, stats.misses),
        HostMode::Inclusive => {
            assert_eq!(stats.demotions, 0, "an inclusive store needs no write-back")
        }
    }
}

#[test]
fn an_exclusive_store_never_loses_or_duplicates_a_record() {
    run_random_passes(HostMode::Exclusive);
}

#[test]
fn an_inclusive_store_needs_no_write_back() {
    run_random_passes(HostMode::Inclusive);
}

#[test]
fn the_exclusive_store_never_needs_more_than_one_spare_place() {
    let state = small(HostMode::Exclusive);
    assert_eq!(state.mode().places_per_layer(state.geometry()), 16 - 5 + 1);
    // A pass of three misses reuses the places its own copies free.
    let mut state = small(HostMode::Exclusive);
    let copies = state.begin_pass(0, &[10, 11, 12]).unwrap();
    let dst: Vec<u32> = copies
        .ops
        .iter()
        .map(|op| op.demote.unwrap().dst_place)
        .collect();
    let src: Vec<u32> = copies.ops.iter().map(|op| op.src_place).collect();
    assert_eq!(dst[0], 11, "the first write-back takes the spare place");
    assert_eq!(
        dst[1], src[0],
        "the second takes the place the first copy left"
    );
    assert_eq!(dst[2], src[1]);
}

// ---------------------------------------------------------------------------------------------
// The filler rule.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_two_passes_split_the_router_weights_between_resident_and_missed_picks() {
    let plan = plan_with_slots(143);
    let counts = parse_route_counts(COUNTS, 48, 512).unwrap();
    let mut state =
        ExpertBankState::new(&plan, &real_layout(), HostMode::Exclusive, &counts).unwrap();
    let layer = 5;
    let row = state.slot_row(layer).to_vec();
    let resident: Vec<i32> = (0..512)
        .filter(|&e| row[e] >= 0)
        .map(|e| e as i32)
        .collect();
    let cold: Vec<i32> = (0..512).filter(|&e| row[e] < 0).map(|e| e as i32).collect();
    // Ten picks: seven resident, three missed, interleaved.
    let ids = [
        resident[0],
        cold[0],
        resident[1],
        resident[2],
        cold[1],
        resident[3],
        resident[4],
        cold[2],
        resident[5],
        resident[6],
    ];
    let weights: Vec<f32> = (1..=10).map(|k| k as f32 / 55.0).collect();
    let classified = classify_reference(&row, 143, &ids, &weights).unwrap();
    assert_eq!(classified.misses.len(), 3);

    let copies = state.begin_pass(layer, &ids).unwrap();
    assert_eq!(copies.ops.len(), 3);
    assert_eq!(copies.hits, 7);
    let (slots2, weights2) =
        miss_pass_inputs(&ids, &weights, &classified, state.slot_row(layer)).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        let missed = cold.contains(&id);
        let first = classified.weights[i].to_bits();
        let second = weights2[i].to_bits();
        // Each pick appears once: the other pass carries +0.0 on the filler.
        if missed {
            assert_eq!(first, 0.0f32.to_bits());
            assert_eq!(second, weights[i].to_bits());
            assert_eq!(slots2[i], state.slot_row(layer)[id as usize]);
            assert!(
                slots2[i] > 0,
                "a missed pick never lands on the filler slot"
            );
        } else {
            assert_eq!(first, weights[i].to_bits());
            assert_eq!((slots2[i], second), (0, 0.0f32.to_bits()));
        }
        assert_eq!(f32::from_bits(first) + f32::from_bits(second), weights[i]);
    }
    // The filler slot kept its expert through the pass.
    assert_eq!(
        state.expert_in_slot(layer, 0),
        Some(row.iter().position(|&s| s == 0).unwrap() as u16)
    );
}

#[test]
fn the_second_pass_refuses_what_has_no_second_pass() {
    let row = vec![0, -1, 1, -1];
    let ids = [0, 1];
    let weights = [0.5f32, 0.5];
    let classified = classify_reference(&row, 2, &ids, &weights).unwrap();
    // The missed expert 1 still has no slot after the "copy".
    assert!(miss_pass_inputs(&ids, &weights, &classified, &row).is_err());
    let after = vec![0, 1, 2, -1];
    let (slots, out) = miss_pass_inputs(&ids, &weights, &classified, &after).unwrap();
    assert_eq!((slots, out), (vec![0, 1], vec![0.0, 0.5]));
    assert!(miss_pass_inputs(&ids, &weights[..1], &classified, &after).is_err());
    let fault = classify_reference(&row, 2, &[-1, 1], &weights).unwrap();
    assert!(miss_pass_inputs(&[-1, 1], &weights, &fault, &after).is_err());
    let invalid = classify_reference(&row, 2, &[0, 0], &weights).unwrap();
    assert!(miss_pass_inputs(&[0, 0], &weights, &invalid, &after).is_err());
    // A non-finite weight is flagged by the classifier (word != 0), so there is no second pass.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let w = [0.5f32, bad];
        let flagged = classify_reference(&row, 2, &ids, &w).unwrap();
        assert_ne!(flagged.word, 0, "{bad}");
        assert!(
            miss_pass_inputs(&ids, &w, &flagged, &after).is_err(),
            "{bad}"
        );
    }
    // -0.0 is the one finite weight whose bits change: -0.0 + 0.0 is +0.0. A router softmax cannot
    // produce it; the second pass carries it through as given.
    let w = [0.5f32, -0.0];
    let negative = classify_reference(&row, 2, &ids, &w).unwrap();
    let (_, second) = miss_pass_inputs(&ids, &w, &negative, &after).unwrap();
    assert_eq!(second[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(negative.weights[1].to_bits(), 0.0f32.to_bits());
}

// ---------------------------------------------------------------------------------------------
// The route trace (ignored: needs the 109 MB trace file).
// ---------------------------------------------------------------------------------------------

/// Replays the T0 route trace through the policy as the replay tool did: a cache per layer, a fresh
/// bank per prompt, prompt steps admitted, hits counted over the decode steps of prompts 1 to 6.
/// The receipt: LRU hit 0.870195 at 143 slots and 0.914977 at 184 (`flash-next-routing-locality-t0-trace`).
/// This policy lands 0.0007 and 0.0004 under it: the gap is the slot-0 protection of a pass with a
/// miss, which the replay tool's cache (and Flint's) does not have; without it an independent replay
/// gives 0.870107 and 0.914921.
fn replay(trace: &str, slots: usize) -> f64 {
    let geometry = BankGeometry::new(48, 512, 10, slots, 1_881_600).unwrap();
    let fresh = || {
        let row: Vec<i32> = (0..512)
            .map(|e| if e < slots { e as i32 } else { -1 })
            .collect();
        let rows = vec![row.as_slice(); 48];
        let counts = vec![vec![0u64; 512]; 48];
        ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap()
    };
    let (mut policy, mut current) = (fresh(), String::new());
    let (mut hits, mut picks) = (0u64, 0u64);
    for line in trace.lines() {
        if line.starts_with('#') || line.starts_with("prompt\t") {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields[0] != current {
            current = fields[0].to_owned();
            policy = fresh();
        }
        let layer: usize = fields[3].parse().unwrap();
        let ids: Vec<i32> = fields[4..14].iter().map(|id| id.parse().unwrap()).collect();
        let plan = policy.plan_pass(layer, &ids).unwrap();
        if fields[1] == "g" && fields[0] != "0" {
            hits += plan.hits as u64;
            picks += plan.distinct as u64;
        }
    }
    hits as f64 / picks as f64
}

#[test]
#[ignore = "needs FIFI_FLASH_NEXT_ROUTE_TRACE, the 109 MB T0 trace.tsv; host only, no GPU"]
fn the_policy_reproduces_the_recorded_lru_hit_rates_on_the_t0_trace() {
    let path = std::env::var("FIFI_FLASH_NEXT_ROUTE_TRACE").expect("FIFI_FLASH_NEXT_ROUTE_TRACE");
    let trace = std::fs::read_to_string(path).expect("the trace");
    for (slots, recorded) in [(143, 0.870195), (184, 0.914977)] {
        let rate = replay(&trace, slots);
        eprintln!("{slots} slots: hit {rate:.6} (recorded {recorded})");
        assert!(
            (rate - recorded).abs() < 0.002,
            "{slots} slots: {rate} against {recorded}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The device bank (ignored: needs a GPU).
// ---------------------------------------------------------------------------------------------

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod device_tests {
    use super::super::bank::{
        BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
        ReplaceOnMissLru, device::*,
    };
    use crate::{EngError, family::project::cuda::CudaProjector, kernels::cuda::device_present};

    /// The record of an expert is a pattern of its layer and number, distinct per byte position.
    struct Pattern;

    impl RecordSource for Pattern {
        fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
            for (index, byte) in out.iter_mut().enumerate() {
                *byte = (index as u32 * 31 + u32::from(expert) * 7 + layer as u32 * 101) as u8;
            }
            Ok(())
        }
    }

    fn expected(layer: usize, expert: u16, record: usize) -> Vec<u8> {
        let mut out = vec![0u8; record];
        Pattern.record(layer, expert, &mut out).unwrap();
        out
    }

    /// 2 layers, 12 experts, top-2, 4 slots of 9,216-byte records (hidden 128, inter 128, 16 words).
    #[test]
    #[ignore = "needs a GPU (under 1 GiB)"]
    fn a_small_bank_fills_copies_on_a_miss_and_writes_the_evicted_expert_back() {
        assert!(
            device_present("Flash-Next expert bank"),
            "a GPU is required"
        );
        let record = 9216;
        let geometry = BankGeometry::new(2, 12, 2, 4, record).unwrap();
        let row: Vec<i32> = (0..12).map(|e| if e < 4 { e } else { -1 }).collect();
        let counts = vec![(0..12u64).collect::<Vec<_>>(); 2];
        let policy =
            ReplaceOnMissLru::from_fill(geometry, &[row.as_slice(), row.as_slice()], &counts, None)
                .unwrap();
        let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();
        let places = state.mode().places_per_layer(&geometry);

        let projector = CudaProjector::new().expect("projector");
        let ctx = projector.context();
        let pinned_before = ctx.live_owners().pinned;
        let plan = HostPinPlan::new(
            &geometry,
            HostMode::Exclusive,
            &HostMemory {
                total: 62 * (1 << 30),
                available: 55 * (1 << 30),
            },
            &HostTerms::DEFAULT,
        )
        .expect("the host plan");
        let mut bank = DeviceBank::build(ctx, state, &Pattern, &plan).expect("build");
        assert_eq!(
            ctx.live_owners().pinned - pinned_before,
            2 * places * record,
            "the pinned store is the cold places of both layers"
        );
        let read_slot = |bank: &DeviceBank, layer: usize, slot: usize| {
            let mut out = vec![0u8; record];
            bank.slots(layer)
                .unwrap()
                .read_at(bank.stream(), geometry.offset(slot), &mut out)
                .expect("read");
            out
        };
        for slot in 0..4 {
            assert_eq!(
                read_slot(&bank, 0, slot),
                expected(0, slot as u16, record),
                "fill slot {slot}"
            );
        }

        // Experts 8 and 9 miss in layer 0; slot 0 is the filler, so slots 1 and 2 are refilled.
        let copies = bank.step(ctx, 0, &[8, 9]).unwrap();
        assert_eq!(copies.ops.len(), 2);
        for op in &copies.ops {
            assert_eq!(
                read_slot(&bank, 0, usize::from(op.slot)),
                expected(0, op.incoming, record)
            );
            let demote = op.demote.expect("an exclusive store writes back");
            let mut host = vec![0u8; record];
            bank.read_place(0, demote.dst_place, &mut host)
                .expect("the place");
            assert_eq!(
                host,
                expected(0, demote.expert, record),
                "the evicted expert is in its new place"
            );
        }
        // Layer 1 is untouched, and a pass of hits copies nothing.
        assert_eq!(read_slot(&bank, 1, 1), expected(1, 1, record));
        let copies = bank.step(ctx, 0, &[8, 9]).unwrap();
        assert!(copies.ops.is_empty());

        // The failure branch of `step`: a context of another adapter fails in the first copy, after
        // the host model has moved. The bank is poisoned and every later use of it is refused.
        assert_eq!((bank.stats().passes, bank.stats().misses), (2, 2));
        assert_eq!(bank.geometry().slots, 4);
        let other = CudaProjector::new().expect("a second context");
        assert!(!bank.is_poisoned());
        assert!(bank.slot_row(0).is_ok());
        assert!(bank.step(other.context(), 0, &[10, 11]).is_err());
        assert!(bank.is_poisoned());
        assert!(
            bank.step(ctx, 0, &[8, 9]).is_err(),
            "a poisoned bank refuses a step"
        );
        assert!(bank.admit(ctx, 0, &[1]).is_err(), "and an admission");
        assert!(bank.slot_row(0).is_err(), "and the slot row");
        assert!(bank.slots(0).is_err(), "and the slot buffer");
        assert!(
            bank.read_place(0, 0, &mut [0u8; 16]).is_err(),
            "and a place read"
        );
        // A plan for another store size is refused before anything is allocated.
        let state = {
            let row: Vec<i32> = (0..12).map(|e| if e < 4 { e } else { -1 }).collect();
            let counts = vec![(0..12u64).collect::<Vec<_>>(); 2];
            let policy = ReplaceOnMissLru::from_fill(
                geometry,
                &[row.as_slice(), row.as_slice()],
                &counts,
                None,
            )
            .unwrap();
            ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap()
        };
        let mut wrong = plan;
        wrong.store_bytes += 1;
        let error = DeviceBank::build(ctx, state, &Pattern, &wrong)
            .err()
            .expect("refused");
        assert!(
            error.describe().contains("store bytes"),
            "{}",
            error.describe()
        );
        assert_eq!(
            ctx.live_owners().pinned,
            pinned_before + 2 * places * record,
            "the refused build pinned nothing"
        );

        // The pinned store is released with the bank: observe the drop, not an output.
        drop(bank);
        assert_eq!(
            ctx.live_owners().pinned,
            pinned_before,
            "the pinned store was freed"
        );
    }
}
