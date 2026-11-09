//! The prefill ring's stream plan: layer-major, skipping the banked experts, ring slots by
//! round-robin.

use super::bank::NO_PLACE;
use super::prefill_ring::{StreamEntry, plan_from_rows};

fn entry(layer: u16, expert: u16, place: u32, slot: u16) -> StreamEntry {
    StreamEntry {
        layer,
        expert,
        place,
        slot,
    }
}

#[test]
fn plans_layer_major_skipping_banked() {
    let slots = [vec![0, -1, -1], vec![-1, -1, 2]];
    let places = [vec![NO_PLACE, 1, 2], vec![0, 1, NO_PLACE]];
    let rows: Vec<&[i32]> = slots.iter().map(|v| v.as_slice()).collect();
    let prows: Vec<&[u32]> = places.iter().map(|v| v.as_slice()).collect();
    // Ring of 3: entries wrap 0, 1, 2, 0.
    assert_eq!(
        plan_from_rows(&rows, &prows, 3, 3),
        vec![
            entry(0, 1, 1, 0),
            entry(0, 2, 2, 1),
            entry(1, 0, 0, 2),
            entry(1, 1, 1, 0),
        ]
    );
}

#[test]
fn empty_when_everything_resident() {
    let rows: Vec<&[i32]> = vec![[0i32, 1].as_slice()];
    let prows: Vec<&[u32]> = vec![[NO_PLACE, NO_PLACE].as_slice()];
    assert!(plan_from_rows(&rows, &prows, 2, 8).is_empty());
}

#[test]
fn skips_experts_without_a_place() {
    // A banked expert (slot >= 0) is skipped regardless of place; an unbanked expert without
    // a place cannot be streamed and is skipped too.
    let rows: Vec<&[i32]> = vec![[-1i32, 0, -1].as_slice()];
    let prows: Vec<&[u32]> = vec![[NO_PLACE, 5, NO_PLACE].as_slice()];
    assert!(plan_from_rows(&rows, &prows, 3, 4).is_empty());
}
