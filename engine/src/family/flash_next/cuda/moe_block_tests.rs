//! Host tests of the split MoE block's logic: the private scratch layout, the classify log, the
//! spans of the bank buffer, and the two passes of a layer against one pass over an all-resident
//! bank. The expert outputs are synthetic and deterministic; the device test (ignored) runs the
//! real kernels on a real layer.

use super::{
    bank::{BankGeometry, ExpertBankState, HostMode, ReplaceOnMissLru},
    moe_block::{
        PassInputs, ScratchLayout, SlotSpans, classify_picks, first_poisoned_filler, parse_log,
        pick_union, plan_second_pass,
    },
};
use crate::{
    EngError,
    family::{flash_next::moe::slot_bank::SlotBankLayout, project::exl3_jobs_plan::EXL3_JOB_WORDS},
};
// The planning items sit in the native-cuda device module; the scratch-size tests above run on
// every host, so only these imports carry the gate.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use {
    super::moe_block::{source_offsets, source_plan},
    crate::family::project::cuda::exl3_jobs::PairPlan,
};

const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const HIDDEN: usize = 8;
const RECORD: usize = 1_881_600;

fn layout() -> SlotBankLayout {
    SlotBankLayout::new(2560, 640, [48, 48, 48]).unwrap()
}

/// A one-layer bank holding `held` (slot `s` holds `held[s]`), and the state around it.
fn bank(held: &[u16]) -> ExpertBankState<ReplaceOnMissLru> {
    let geometry = BankGeometry::new(1, EXPERTS, TOP_K, held.len(), RECORD).unwrap();
    let mut row = vec![-1i32; EXPERTS];
    for (slot, &expert) in held.iter().enumerate() {
        row[usize::from(expert)] = slot as i32;
    }
    // The least popular holder is the first victim: count by id, so a low id is evicted last.
    let counts = vec![(0..EXPERTS as u64).rev().collect::<Vec<u64>>()];
    let policy = ReplaceOnMissLru::from_fill(geometry, &[row.as_slice()], &counts, None).unwrap();
    ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap()
}

/// The sum the combine kernel forms for one row: each product rounded alone, the picks added in
/// pick order, then the gated shared term, then the residual (`moe_combine.rs`).
fn combine_reference(
    residual: &[f32],
    down: &[Vec<f32>],
    weights: &[f32],
    shared: Option<(f32, &[f32])>,
) -> Vec<f32> {
    (0..residual.len())
        .map(|j| {
            let mut sum = weights[0] * down[0][j];
            for (weight, row) in weights.iter().zip(down).skip(1) {
                sum += weight * row[j];
            }
            if let Some((gate, row)) = shared {
                sum += gate * row[j];
            }
            residual[j] + sum
        })
        .collect()
}

/// Router weights that sum to one, falling by rank.
fn weights() -> Vec<f32> {
    let raw: Vec<f32> = (0..TOP_K).map(|rank| 1.0 / (rank as f32 + 2.0)).collect();
    let total: f32 = raw.iter().sum();
    raw.iter().map(|w| w / total).collect()
}

/// The down output of `expert` at column `j`, as a deterministic finite f32.
fn down_of(expert: u16, j: usize) -> f32 {
    ((usize::from(expert) * 31 + j * 7) % 97) as f32 / 13.0 - 3.5
}

fn down_rows(experts: impl Iterator<Item = u16>) -> Vec<Vec<f32>> {
    experts
        .map(|expert| (0..HIDDEN).map(|j| down_of(expert, j)).collect())
        .collect()
}

fn shared_row() -> Vec<f32> {
    (0..HIDDEN).map(|j| 0.25 + j as f32 * 0.125).collect()
}

fn holders(state: &ExpertBankState<ReplaceOnMissLru>, slots: usize) -> Vec<u16> {
    (0..slots)
        .map(|slot| state.expert_in_slot(0, slot).expect("a filled slot"))
        .collect()
}

/// Runs the two passes of one layer on the host: the picks go through the classifier and the
/// policy, then the combine of each pass over the experts the slots held at that moment.
/// Returns the split result and the pass inputs.
fn split(
    state: &mut ExpertBankState<ReplaceOnMissLru>,
    ids: &[i32],
    weights: &[f32],
    gate: f32,
) -> (Vec<f32>, PassInputs) {
    let slots = state.geometry().slots;
    let before = holders(state, slots);
    let row_before = state.slot_row(0).to_vec();
    let classified = classify_picks(&row_before, slots, ids, weights).unwrap();
    // Pass one reads the slots as they are before the copies.
    let first_down = down_rows(classified.slot_ids.iter().map(|&s| before[s as usize]));
    let zero = vec![0.0f32; HIDDEN];
    let shared = shared_row();
    let first = combine_reference(
        &zero,
        &first_down,
        &classified.weights,
        Some((gate, &shared)),
    );
    state.begin_pass(0, ids).unwrap();
    let after = holders(state, slots);
    let second = plan_second_pass(ids, weights, &classified, state.slot_row(0)).unwrap();
    let result = match &second.miss {
        None => first,
        Some((miss_slots, miss_weights)) => {
            let down = down_rows(miss_slots.iter().map(|&s| after[s as usize]));
            combine_reference(&first, &down, miss_weights, None)
        }
    };
    (result, second)
}

/// The same picks over a bank that holds every expert: one combine in pick order.
fn all_resident(ids: &[i32], weights: &[f32], gate: f32) -> Vec<f32> {
    let down = down_rows(ids.iter().map(|&id| id as u16));
    combine_reference(&[0.0; HIDDEN], &down, weights, Some((gate, &shared_row())))
}

fn close(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum();
    let norm: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (diff / norm).sqrt()
}

#[test]
fn the_private_scratch_is_aligned_disjoint_and_small() {
    let layout = ScratchLayout::new(512, 10, 2560, 1).unwrap();
    let regions = [
        (layout.slot_row, 512 * 4),
        (layout.slot_ids, 40),
        (layout.weights, 40),
        (layout.log, 80),
        (layout.ctrl, 64),
        (layout.word, 4),
        (layout.mbox_req, 128),
        (layout.mbox_row, 2560 * 4),
        (layout.miss_ids, 40),
        (layout.miss_weights, 40),
        (layout.partial, 2560 * 4),
        // The job-table perm region: the two sources' lists and job tables together —
        // 2 list words plus one 4-word job record per pair (a source's jobs are at most its
        // pairs) — at routed = rows·top_k = 10.
        (layout.perm, 240),
    ];
    for (index, &(offset, bytes)) in regions.iter().enumerate() {
        assert_eq!(offset % 256, 0, "region {index}");
        assert!(offset + bytes <= layout.total, "region {index}");
        for &(other, other_bytes) in &regions[index + 1..] {
            assert!(offset + bytes <= other || other + other_bytes <= offset);
        }
    }
    assert_eq!(layout.total, 24_832, "24.25 KiB at the real sizes");
}

/// The perm region is the last region and covers the widest split of the chunk's pairs over
/// the two sources even when every pair is its own job; the allocation grows with the planned
/// chunk width (the row-indexed regions scale with it).
#[test]
fn the_private_scratch_covers_the_widest_chunk_plan() {
    let one = ScratchLayout::new(512, 10, 2560, 1).unwrap();
    for (rows, routed) in [(32usize, 320usize), (64, 640)] {
        let layout = ScratchLayout::new(512, 10, 2560, rows).unwrap();
        assert_eq!(layout.rows, rows);
        // The perm region is last; its size is 2 list words plus one job record (4 words) per
        // pair — the worst case is every pair in one source with a job of its own, and the two
        // sources sum to the same bound.
        assert_eq!(
            layout.total - layout.perm,
            (2 + EXL3_JOB_WORDS) * routed * 4
        );
        assert!(layout.total > one.total);
        // The row-indexed regions grow by more than the perm region alone.
        assert!(layout.total - one.total > (routed - 10) * (2 + EXL3_JOB_WORDS) * 4);
    }
}

/// One source's counting sort: sorted positions map back to the chunk's pair positions, and
/// a slot id outside the source's slots refuses the chunk.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn source_plan_maps_sorted_positions_to_chunk_pairs() {
    // A chunk of 2 rows, top_k 2: pair = row·2 + rank. Pairs 0, 2 and 3 pick bank slots
    // (5, 5, 0); pair 1 picks the ring (slot 9).
    let ids = [5i32, 9, 5, 0];
    let bank = source_plan(&ids, &[0, 2, 3], 8, 2).unwrap();
    assert_eq!(bank.counts, {
        let mut counts = vec![0usize; 8];
        counts[0] = 1;
        counts[5] = 2;
        counts
    });
    // Slot 0 owns pair 3 (row 1); slot 5 owns pairs 0 and 2 (rows 0 and 1), chunk order kept.
    assert_eq!(bank.order, [1, 0, 1]);
    assert_eq!(bank.scatter, [3, 0, 2]);
    let ring = source_plan(&ids, &[1], 16, 2).unwrap();
    assert_eq!(ring.counts, {
        let mut counts = vec![0usize; 16];
        counts[9] = 1;
        counts
    });
    assert_eq!(ring.order, [0]);
    assert_eq!(ring.scatter, [1]);
    // A slot id outside the source's slots is a router fault, refused.
    assert!(matches!(
        source_plan(&ids, &[1], 8, 2),
        Err(EngError::ModelOutput(_))
    ));
}

/// The two sources' lists and job tables pack into the perm region back to back, and their
/// gate/up segments tile the pair faces in source order.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn source_offsets_pack_the_perm_region() {
    // Counts drive the job tables: one job per expert with rows (the 64-row tile never splits
    // these small counts), so the bank (3 + 1 rows over two experts) takes two jobs.
    let bank = PairPlan {
        counts: vec![3, 1],
        order: vec![],
        scatter: vec![],
    };
    let ring = PairPlan {
        counts: vec![2],
        order: vec![],
        scatter: vec![],
    };
    let (sub_bank, sub_ring) = ([0usize, 1, 2, 3], [4usize, 5]);
    let (at_of, seg_of) = source_offsets(1_000, [&bank, &ring], [&sub_bank, &sub_ring]);
    // Bank: 4 pairs and 2 jobs — (2·4 + 4·2)·4 bytes of lists and table.
    assert_eq!(at_of, [1_000, 1_000 + (2 * 4 + EXL3_JOB_WORDS * 2) * 4]);
    assert_eq!(seg_of, [0, 4]);
    // Ring: 2 pairs and 1 job; the packing ends inside the perm formula's bound
    // (2 words + one job record a pair over all 6 pairs, 3 jobs).
    let end = at_of[1] + (2 * 2 + EXL3_JOB_WORDS) * 4;
    assert_eq!(end - 1_000, (2 * 6 + EXL3_JOB_WORDS * 3) * 4);
    assert!(end - 1_000 <= (2 + EXL3_JOB_WORDS) * 6 * 4);
}

#[test]
fn the_classify_log_round_trips() {
    let ids: Vec<i32> = vec![5, 400, 7, 2, 511, 0, 31, 9, 8, 100];
    let weights = weights();
    let mut raw = Vec::new();
    for (id, weight) in ids.iter().zip(&weights) {
        raw.extend_from_slice(&id.to_le_bytes());
        raw.extend_from_slice(&weight.to_bits().to_le_bytes());
    }
    let (got_ids, got_weights) = parse_log(&raw, TOP_K).unwrap();
    assert_eq!(got_ids, ids);
    assert!(
        got_weights
            .iter()
            .zip(&weights)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(parse_log(&raw[..raw.len() - 1], TOP_K).is_err());
    assert!(parse_log(&raw, TOP_K - 1).is_err());
}

#[test]
fn a_router_fault_and_an_invalid_route_have_no_second_pass() {
    let held: Vec<u16> = (0..16).collect();
    let state = bank(&held);
    let row = state.slot_row(0).to_vec();
    let good: Vec<i32> = (0..10).collect();
    assert!(classify_picks(&row, 16, &good, &weights()).is_ok());
    let mut fault = good.clone();
    fault[3] = -1;
    let error = classify_picks(&row, 16, &fault, &weights()).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    let mut twice = good.clone();
    twice[4] = 2;
    assert!(classify_picks(&row, 16, &twice, &weights()).is_err());
    let mut outside = good.clone();
    outside[0] = 512;
    assert!(classify_picks(&row, 16, &outside, &weights()).is_err());
    assert!(classify_picks(&row, 16, &good, &weights()[..9]).is_err());
}

#[test]
fn the_spans_address_the_bank_buffer_at_the_record_stride() {
    let layout = layout();
    assert_eq!(layout.record_bytes(), RECORD);
    for slots in [11usize, 16, 32] {
        let spans = SlotSpans::new(&layout, slots).unwrap();
        assert_eq!(spans.trellis_stride, RECORD);
        assert_eq!(spans.scale_stride, RECORD / 4);
        let bank_bytes = layout.bank_bytes(slots).unwrap();
        assert_eq!(bank_bytes, slots * RECORD);
        for (index, part) in spans.parts.iter().enumerate() {
            assert_eq!(
                part.shape,
                if index == 2 { [640, 2560] } else { [2560, 640] }
            );
            assert_eq!(part.bits_twice, 6, "3 bits per weight");
            for range in [&part.trellis, &part.suh, &part.svh] {
                assert!(range.end <= bank_bytes, "slots {slots} part {index}");
            }
            // Member `i` of the span is slot `i`'s component: the last member ends the span.
            let own = layout.part(index);
            assert_eq!(
                part.trellis.end - part.trellis.start,
                own.trellis.len() + (slots - 1) * RECORD
            );
            assert_eq!(part.suh.start, own.suh.start);
            assert_eq!(part.svh.start, own.svh.start);
        }
        // The three parts do not overlap within one record.
        let one = &spans.parts;
        assert!(one[0].svh.start < one[1].trellis.start);
        assert!(one[1].svh.start < one[2].trellis.start);
    }
    assert!(SlotSpans::new(&layout, 0).is_err());
}

#[test]
fn all_resident_picks_are_one_pass() {
    let held: Vec<u16> = (0..16).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![3, 9, 0, 14, 1, 7, 12, 5, 2, 15];
    let w = weights();
    let (result, inputs) = split(&mut state, &ids, &w, 0.6);
    assert_eq!(inputs.misses, 0);
    assert!(inputs.miss.is_none());
    // A layer without a miss is one launch over the same slots: the resident pass IS the result,
    // and with the identity fill it is bit for bit the all-resident combine.
    assert_eq!(result, all_resident(&ids, &w, 0.6));
}

#[test]
fn a_miss_pass_adds_each_missed_pick_once() {
    let held: Vec<u16> = (0..16).collect();
    let mut state = bank(&held);
    // Eight resident picks and two cold ones, in rank positions 2 and 7.
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let classified = classify_picks(state.slot_row(0), 16, &ids, &w).unwrap();
    assert_eq!(classified.misses.len(), 2);
    let (result, inputs) = split(&mut state, &ids, &w, 0.6);
    let (slots, miss_weights) = inputs.miss.expect("two misses");
    // The weights of the two passes add to the router's weights bit for bit, one pick per pick.
    for (pick, &weight) in w.iter().enumerate() {
        let sum = classified.weights[pick] + miss_weights[pick];
        assert_eq!(sum.to_bits(), weight.to_bits(), "pick {pick}");
        let missed = pick == 2 || pick == 7;
        assert_eq!(miss_weights[pick] != 0.0, missed, "pick {pick}");
        assert_eq!(classified.weights[pick] != 0.0, !missed, "pick {pick}");
        if !missed {
            assert_eq!(slots[pick], 0, "a filler pick reads slot 0");
            assert_eq!(miss_weights[pick].to_bits(), 0.0f32.to_bits());
        }
    }
    // The missed picks sit on slots the copies filled, never on slot 0.
    for &pick in &[2usize, 7] {
        assert_ne!(slots[pick], 0);
        assert_eq!(
            state.expert_in_slot(0, slots[pick] as usize),
            Some(ids[pick] as u16)
        );
    }
    // The split equals one pass over all-resident experts up to the association of the sums.
    let want = all_resident(&ids, &w, 0.6);
    let error = close(&result, &want);
    assert!(error < 1e-6, "relative L2 {error}");
}

#[test]
fn every_pick_missing_is_all_filler_then_all_picks() {
    // The bank holds sixteen experts none of which the router picks.
    let held: Vec<u16> = (100..116).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let classified = classify_picks(state.slot_row(0), 16, &ids, &w).unwrap();
    assert_eq!(classified.misses.len(), 10);
    assert!(classified.slot_ids.iter().all(|&s| s == 0));
    assert!(classified.weights.iter().all(|w| w.to_bits() == 0));
    let (result, inputs) = split(&mut state, &ids, &w, 0.6);
    let (slots, miss_weights) = inputs.miss.expect("ten misses");
    assert_eq!(inputs.misses, 10);
    // Ten picks of ten distinct experts need ten slots besides slot 0.
    let mut seen = slots.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 10);
    assert!(!slots.contains(&0));
    assert!(
        miss_weights
            .iter()
            .zip(&w)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let error = close(&result, &all_resident(&ids, &w, 0.6));
    assert!(error < 1e-6, "relative L2 {error}");
}

#[test]
fn a_second_use_of_the_same_picks_is_one_pass() {
    let held: Vec<u16> = (0..16).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![3, 9, 300, 14, 1, 7, 12, 411, 2, 15];
    let w = weights();
    let (first, inputs) = split(&mut state, &ids, &w, 0.6);
    assert_eq!(inputs.misses, 2);
    let (again, inputs) = split(&mut state, &ids, &w, 0.6);
    assert_eq!(inputs.misses, 0, "the missed experts now sit in the bank");
    assert!(close(&again, &first) < 1e-6);
}

#[test]
fn a_filler_output_that_is_not_finite_poisons_the_sum() {
    // Why the block can verify the filler rows: +0.0 times an infinity is a NaN, and the NaN
    // reaches every column the infinity touches.
    let zero = vec![0.0f32; 4];
    let mut filler = vec![1.0f32; 4];
    filler[2] = f32::INFINITY;
    let good = vec![0.5f32; 4];
    let out = combine_reference(&zero, &[good.clone(), filler], &[0.5, 0.0], None);
    assert!(out[2].is_nan());
    assert!(out[0].is_finite() && out[1].is_finite() && out[3].is_finite());
    let fine = combine_reference(&zero, &[good.clone(), good], &[0.5, 0.0], None);
    assert!(fine.iter().all(|v| v.is_finite()));
}

#[test]
fn the_bank_state_after_a_pass_is_what_the_second_pass_reads() {
    let held: Vec<u16> = (0..11).collect();
    let mut state = bank(&held);
    let ids: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 200];
    let w = weights();
    let row_before = state.slot_row(0).to_vec();
    let classified = classify_picks(&row_before, 11, &ids, &w).unwrap();
    let copies = state.begin_pass(0, &ids).unwrap();
    assert_eq!((copies.hits, copies.ops.len()), (9, 1));
    let second = plan_second_pass(&ids, &w, &classified, state.slot_row(0)).unwrap();
    let (slots, _) = second.miss.unwrap();
    assert_eq!(slots[9], i32::from(copies.ops[0].slot));
    // The classification made before the copies is stale after them: a second use of the old row
    // would send the missed pick to the filler.
    assert_eq!(row_before[200], -1);
    assert!(state.slot_row(0)[200] >= 0);
}

fn le_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn a_non_finite_filler_row_is_found_by_pick() {
    let finite = vec![0.5f32; HIDDEN];
    let mut nan = finite.clone();
    nan[3] = f32::NAN;
    let mut inf = finite.clone();
    inf[7] = f32::NEG_INFINITY;
    // Picks 1 and 3 are fillers (weight +0.0); picks 0 and 2 are real picks.
    let weights = [0.6f32, 0.0, 0.4, 0.0];
    let rows = |bad1: &Vec<f32>, bad3: &Vec<f32>| {
        le_rows(&[finite.clone(), bad1.clone(), finite.clone(), bad3.clone()])
    };
    assert_eq!(
        first_poisoned_filler(&rows(&finite, &finite), HIDDEN, &weights),
        None
    );
    assert_eq!(
        first_poisoned_filler(&rows(&nan, &finite), HIDDEN, &weights),
        Some(1),
        "a NaN row at weight +0.0"
    );
    assert_eq!(
        first_poisoned_filler(&rows(&finite, &inf), HIDDEN, &weights),
        Some(3),
        "an infinite row at weight +0.0"
    );
    assert_eq!(
        first_poisoned_filler(&rows(&inf, &nan), HIDDEN, &weights),
        Some(1),
        "the first of two"
    );
    // A non-finite row on a real pick is not a filler problem: the sum is wrong for another reason.
    let real_nan = le_rows(&[nan.clone(), finite.clone(), finite.clone(), finite.clone()]);
    assert_eq!(first_poisoned_filler(&real_nan, HIDDEN, &weights), None);
    // A negative zero is the same product: -0.0 == 0.0.
    let negative = [-0.0f32, 0.0, 0.4, 0.0];
    let first = le_rows(&[nan, finite.clone(), finite.clone(), finite.clone()]);
    assert_eq!(first_poisoned_filler(&first, HIDDEN, &negative), Some(0));
}

#[test]
fn pick_union_keeps_each_distinct_id_first_use_first() {
    let rows: Vec<&[i32]> = vec![&[7, 3, 7], &[3, 9], &[]];
    assert_eq!(pick_union(&rows, 16), vec![7, 3, 9]);
    // Duplicates inside one row and across rows collapse to the first use.
    assert_eq!(pick_union(&[&[5, 0, 5], &[0, 5]], 8), vec![5, 0]);
}
