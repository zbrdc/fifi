//! The host tests of the chunk-admit contract: [`AdmitPlan::of`] decisions and the union-admit
//! identity — admitting the chunk's union produces the same plan as admitting the raw list with
//! repeats, which is the mapping the golden's admit arm pins. Cardless: everything here runs on
//! the policy and the slot row, the parts of the contract that live on the host.

use super::bank::{BankGeometry, ReplaceOnMissLru, SlotPolicy, SlotTable};
use super::prefill_adapter::AdmitPlan;

const LAYERS: usize = 2;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const SLOTS: usize = 48;
const RECORD: usize = 1_881_600;

/// The bank fixture of the pick-golden tests: expert `i` in slot `i`, a popularity profile for
/// the victim order, layer 0 fully resident.
fn policy() -> ReplaceOnMissLru {
    let geometry =
        BankGeometry::new(LAYERS, EXPERTS, TOP_K, SLOTS, RECORD).expect("a legal geometry");
    let profile: Vec<Vec<u64>> = (0..LAYERS)
        .map(|layer| {
            (0..EXPERTS)
                .map(|expert| (expert * 31 + layer) as u64 % 977)
                .collect()
        })
        .collect();
    let tables: Vec<Vec<i32>> = (0..LAYERS)
        .map(|_| {
            let mut row = vec![-1i32; EXPERTS];
            for (expert, slot) in row.iter_mut().enumerate().take(SLOTS) {
                *slot = expert as i32;
            }
            row
        })
        .collect();
    let rows: Vec<&[i32]> = tables.iter().map(|row| row.as_slice()).collect();
    ReplaceOnMissLru::from_fill(geometry, &rows, &profile, None).expect("a filled bank")
}

/// `AdmitPlan::of` computes the union in first-seen order, counts the missing against the slot
/// row, and an over-capacity union does not fit.
#[test]
fn the_admit_plan_decides_from_the_slot_row() {
    let bank = policy();
    let row = bank.slot_row(0);
    // Residents 0..=47, so picks in 48..512 are missing and picks in 0..=47 are not.
    let fits_all: Vec<&[i32]> = vec![&[100, 101, 100], &[48, 49, 5]];
    let plan = AdmitPlan::of(&fits_all, EXPERTS, SLOTS, row);
    assert_eq!(plan.union, vec![100, 101, 48, 49, 5]);
    assert_eq!(plan.missing, 4); // 100, 101, 48, 49 are missing; 5 is resident
    assert!(plan.fits);
    assert!(plan.admits());

    // Everything already resident: nothing to admit.
    let resident: Vec<&[i32]> = vec![&[3, 7], &[11, 3]];
    let plan = AdmitPlan::of(&resident, EXPERTS, SLOTS, row);
    assert_eq!(plan.union, vec![3, 7, 11]);
    assert_eq!(plan.missing, 0);
    assert!(!plan.admits());

    // More distinct experts than slots: the union is never admitted whole.
    let over: Vec<i32> = (0..(SLOTS + 10) as i32).collect();
    let over_rows: Vec<&[i32]> = vec![&over];
    let plan = AdmitPlan::of(&over_rows, EXPERTS, SLOTS, row);
    assert_eq!(plan.union.len(), SLOTS + 10);
    assert!(!plan.fits);
    assert!(!plan.admits());
}

/// The union-admit identity: admitting the chunk's union (first-seen order, repeats gone) and
/// admitting the raw pick list with repeats produce the same plan — distinct, hits and every
/// move — because the policy counts an expert once either way; the slot rows agree
/// afterwards. A second admit of the same list on both banks must then also agree: it
/// exercises the recency order the first admit left, where a policy that counted repeats
/// differently would diverge.
#[test]
fn admitting_the_union_moves_the_same_slots_as_admitting_the_raw_list() {
    let mut by_union = policy();
    let mut by_raw = policy();

    // Three rows with a cross-row repeat: two blocks of missing experts and one of residents,
    // 43 distinct against 48 slots so the union admits.
    let rows: Vec<Vec<i32>> = vec![
        (100..111).collect(),
        (200..211).collect(),
        (0..21).chain(0..5).collect(),
    ];
    let refs: Vec<&[i32]> = rows.iter().map(|r| r.as_slice()).collect();
    let flat: Vec<u16> = rows
        .iter()
        .flat_map(|r| r.iter().map(|&id| id as u16))
        .collect();

    let plan_of = AdmitPlan::of(&refs, EXPERTS, SLOTS, by_union.slot_row(0));
    assert!(
        plan_of.admits(),
        "43 distinct (22 missing, 21 resident) fits 48 slots"
    );
    let through_union = by_union
        .admit(0, &plan_of.union)
        .expect("the union fits the slots");
    let direct = by_raw.admit(0, &flat).expect("the raw list fits the slots");

    assert_eq!(through_union.distinct, direct.distinct);
    assert_eq!(through_union.hits, direct.hits);
    assert_eq!(through_union.moves, direct.moves);
    // And the tables agree afterwards, not just the plans.
    assert_eq!(by_union.slot_row(0), by_raw.slot_row(0));

    // The second admit rides the recency order the first one left.
    let again_union = by_union
        .admit(0, &plan_of.union)
        .expect("the union still fits the slots");
    let again_raw = by_raw.admit(0, &flat).expect("the raw list still fits");
    assert_eq!(again_union.moves, again_raw.moves);
    assert_eq!(by_union.slot_row(0), by_raw.slot_row(0));
}
