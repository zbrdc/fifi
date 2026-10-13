//! The reference of the device classification of one layer (TCAP C2): what the `tiered_classify`
//! kernel computes from the router's picks and the layer's slot table row.
//!
//! WHAT: [`classify_reference`] maps each pick to a bank slot or to the dummy read (slot 0,
//! weight +0.0), collects the misses in pick order, logs the raw picks, and flags a route the
//! host would refuse. WHY: the host classification (`TieredHost::begin_layer`) moves onto the
//! device so that the tiered token can be captured; this function is the host-side oracle that
//! the kernel, the device tests and the host must agree with. No CUDA type appears here.
//! Staged for the tiered step of the TCAP design (commits C2 and C6).

/// Bit of the tiered word the classification sets for a route the host would refuse: an id
/// outside the layer, a pick twice, a slot outside the bank, or a non-finite weight.
pub const ROUTE_INVALID: i32 = 4;

/// What the classification of one layer produces.
#[derive(Clone, Debug, PartialEq)]
pub struct Classified {
    /// The bank slot of each pick for the expert launches: a resident pick maps to its slot, a
    /// miss to slot 0, the router-fault id -1 and an invalid pick to -1.
    pub slot_ids: Vec<i32>,
    /// The combine weight of each pick: the router weight, +0.0 for a miss.
    pub weights: Vec<f32>,
    /// The router's picks as they came, `(id, weight)` in rank order: the route log.
    pub log: Vec<(i32, f32)>,
    /// The misses `(expert id, router weight)` in pick order: the request of the CPU server.
    pub misses: Vec<(i32, f32)>,
    /// [`ROUTE_INVALID`] when a pick was refused, else 0.
    pub word: i32,
}

/// Classifies one layer. `slot_row[e]` is the bank slot of expert `e` or -1 (see
/// `TieredHost::slot_row`); `slots` is the bank size. A pick with `id == -1` (the router fault)
/// passes through as `(-1, w)`; an id below -1 or at or above `slot_row.len()`, an expert picked
/// twice, a table slot outside `0..slots` and a non-finite weight give `(-1, w)` and set
/// [`ROUTE_INVALID`]. The caller never gives more than the picks of one layer.
pub fn classify_reference(
    slot_row: &[i32],
    slots: usize,
    ids: &[i32],
    weights: &[f32],
) -> Classified {
    let mut out = Classified {
        slot_ids: Vec::with_capacity(ids.len()),
        weights: Vec::with_capacity(ids.len()),
        log: Vec::with_capacity(ids.len()),
        misses: Vec::new(),
        word: 0,
    };
    for (i, (&id, &weight)) in ids.iter().zip(weights).enumerate() {
        out.log.push((id, weight));
        if id == -1 {
            out.slot_ids.push(-1);
            out.weights.push(weight);
            continue;
        }
        let expert = usize::try_from(id).ok().filter(|&e| e < slot_row.len());
        let repeated = ids[..i].contains(&id);
        let slot = expert.map(|e| slot_row[e]);
        let invalid = expert.is_none()
            || repeated
            || !weight.is_finite()
            || slot.is_some_and(|s| s < -1 || s >= slots as i32);
        if invalid {
            out.slot_ids.push(-1);
            out.weights.push(weight);
            out.word |= ROUTE_INVALID;
        } else if slot == Some(-1) {
            out.slot_ids.push(0);
            out.weights.push(0.0);
            out.misses.push((id, weight));
        } else {
            out.slot_ids.push(slot.unwrap_or(-1));
            out.weights.push(weight);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: [i32; 6] = [2, -1, 0, -1, 1, -1];

    fn run(ids: &[i32], weights: &[f32]) -> Classified {
        classify_reference(&ROW, 3, ids, weights)
    }

    #[test]
    fn hits_map_to_their_slot_and_misses_to_the_dummy_read_in_pick_order() {
        let got = run(&[4, 1, 0, 5], &[0.4, 0.3, 0.2, 0.1]);
        assert_eq!(got.slot_ids, vec![1, 0, 2, 0]);
        assert_eq!(got.weights, vec![0.4, 0.0, 0.2, 0.0]);
        assert_eq!(got.misses, vec![(1, 0.3), (5, 0.1)]);
        assert_eq!(got.log, vec![(4, 0.4), (1, 0.3), (0, 0.2), (5, 0.1)]);
        assert_eq!(got.word, 0);
        assert_eq!(got.weights[1].to_bits(), 0.0f32.to_bits(), "+0.0, not -0.0");
    }

    #[test]
    fn the_router_fault_id_passes_through_without_the_invalid_bit() {
        let got = run(&[-1, -1], &[0.0, f32::NAN]);
        assert_eq!(got.slot_ids, vec![-1, -1]);
        assert_eq!(
            got.word, 0,
            "a faulted row is the router's own status, not an invalid route"
        );
        assert!(got.misses.is_empty());
        let logged: Vec<i32> = got.log.iter().map(|&(id, _)| id).collect();
        assert_eq!(logged, vec![-1, -1], "the log keeps the fault id");
    }

    #[test]
    fn every_refused_route_sets_the_invalid_bit_and_leaves_the_other_picks_alone() {
        let nan = f32::NAN;
        let cases: [(&str, [i32; 3], [f32; 3]); 6] = [
            ("id below -1", [4, -2, 0], [0.5, 0.25, 0.25]),
            ("id past the layer", [4, 6, 0], [0.5, 0.25, 0.25]),
            ("pick twice", [4, 0, 4], [0.5, 0.25, 0.25]),
            ("non-finite weight", [4, 0, 2], [0.5, nan, 0.25]),
            ("infinite weight", [4, 0, 2], [0.5, f32::INFINITY, 0.25]),
            ("slot outside the bank", [4, 0, 2], [0.5, 0.25, 0.25]),
        ];
        for (what, ids, weights) in cases {
            let mut row = ROW;
            if what == "slot outside the bank" {
                row[0] = 3; // the bank has 3 slots: 0..3
            }
            let got = classify_reference(&row, 3, &ids, &weights);
            assert_eq!(got.word, ROUTE_INVALID, "{what}");
            let bad = got.slot_ids.iter().filter(|&&s| s == -1).count();
            assert_eq!(
                bad, 1,
                "{what}: exactly the refused pick is -1: {:?}",
                got.slot_ids
            );
            let logged: Vec<i32> = got.log.iter().map(|&(id, _)| id).collect();
            assert_eq!(logged, ids, "{what}: the log keeps every pick as it came");
        }
        let neg = classify_reference(&ROW, 3, &[4, 0], &[0.5, 0.5]);
        assert_eq!(neg.word, 0, "the control route is valid");
    }

    #[test]
    fn a_slot_in_the_table_below_minus_one_is_refused() {
        let mut row = ROW;
        row[4] = -2;
        let got = classify_reference(&row, 3, &[4], &[1.0]);
        assert_eq!((got.slot_ids, got.word), (vec![-1], ROUTE_INVALID));
    }
}
