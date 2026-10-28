use crate::EngError;

use super::spec::Router;

const OPERATION: &str = "route_experts";

#[derive(Debug, Clone, PartialEq)]
pub struct Routed {
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Selects the top-k experts from softmax probabilities or ungrouped sigmoid scores.
/// The caller applies the scoring function; this seam never softmaxes logits.
///
/// Error variants:
/// - `Checkpoint` when the family declares a router kind this seam cannot route. The router kind
///   comes from the family specification, so the defect is in the family and not in the request.
/// - `BadRequest` when the arguments break the call contract: an empty row, a zero `top_k`, or a
///   `top_k` beyond the row width.
/// - `Backend` when the row holds a NaN, an infinite value, or a negative value, or when the
///   selected values cannot renormalize. A softmax row never holds these, so the model execution
///   that produced the row failed.
///   This `Backend` error ends one job only and is not terminal: the engine stays ready and no
///   native lease becomes unusable. It differs from the terminal `Backend` errors of the MLX adapter
///   under decision 0023.
pub fn route_experts(probs: &[f32], router: &Router, top_k: u32) -> Result<Routed, EngError> {
    let renorm = match *router {
        Router::SoftmaxTopK { renorm } => renorm,
        Router::SigmoidTopK => true,
        _ => {
            return Err(EngError::Checkpoint(
                "router kind is not ungrouped top-k".into(),
            ));
        }
    };
    if probs.is_empty() {
        return Err(EngError::BadRequest(
            "router probabilities must not be empty".into(),
        ));
    }
    if probs.len() > u32::MAX as usize {
        return Err(EngError::BadRequest(
            "router width exceeds the u32 index range".into(),
        ));
    }
    if top_k == 0 {
        return Err(EngError::BadRequest("top_k must exceed zero".into()));
    }
    if u64::from(top_k) > probs.len() as u64 {
        return Err(EngError::BadRequest(
            "top_k exceeds the router width".into(),
        ));
    }
    if probs.iter().any(|prob| !prob.is_finite() || *prob < 0.0) {
        return Err(backend_fault(
            "router probabilities must be finite and non-negative",
        ));
    }
    let mut ranked: Vec<(u32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(index, prob)| (index as u32, *prob))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("non-finite probabilities are rejected before sorting")
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(top_k as usize);
    let indices = ranked.iter().map(|(index, _)| *index).collect();
    let weights = weights_for(&ranked, renorm)?;
    Ok(Routed { indices, weights })
}

fn weights_for(ranked: &[(u32, f32)], renorm: bool) -> Result<Vec<f32>, EngError> {
    if !renorm {
        return Ok(ranked.iter().map(|(_, prob)| *prob).collect());
    }
    let sum: f32 = ranked.iter().map(|(_, prob)| *prob).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(backend_fault(
            "selected router probabilities must sum to a finite positive value",
        ));
    }
    Ok(ranked.iter().map(|(_, prob)| prob / sum).collect())
}

fn backend_fault(message: &str) -> EngError {
    EngError::Backend {
        operation: OPERATION,
        message: message.into(),
    }
}

/// Checks the stacked router-health values of a device-routed chunk (packet Q36-PF-DEVROUTE).
/// Each layer of the chunk sums its whole `[T, E]` probability block into one F32 value, the chunk
/// stacks the values and reads them back once, and this check replaces the finite check that
/// `route_experts` makes per row under host routing: any non-finite value returns the same
/// `Backend` error with the same operation label, and a finite vector passes. A softmax row sums
/// to 1, so a NaN or infinite entry makes its layer's sum non-finite, as in `route_device`.
pub fn check_health(values: &[f32]) -> Result<(), EngError> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(backend_fault(
        "router probabilities must be finite and non-negative",
    ))
}

/// One beyond-tie difference that [`compare_top_k`] found between two selections of the same
/// probability row (packet Q36-BF16-a fix3; decision 0046 A6).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteFault {
    /// The compared rank position, or `None` when the fault is a set difference.
    pub rank: Option<usize>,
    /// The expert of the oracle selection: at `rank`, or an exchanged expert that only the
    /// oracle holds.
    pub oracle: u32,
    /// The expert of the routed selection at the same rank, or the exchanged expert nearest to
    /// `oracle` that only the routed selection holds. When the other side holds no exchanged
    /// expert at all, this is `oracle` itself.
    pub routed: u32,
    /// The probability gap of `oracle` and `routed` on the compared row, or `None` when the row
    /// cannot index one of them: no epsilon excuses that.
    pub gap: Option<f32>,
}

/// The verdict of [`compare_top_k`] on one pair of selections.
#[derive(Debug, Clone, PartialEq)]
pub struct TopKComparison {
    /// Every beyond-tie difference of the pair. An empty vector accepts the pair.
    pub faults: Vec<RouteFault>,
    /// The largest probability gap among the pair's differing ranks and exchanged experts,
    /// tolerated or not. Zero when the two selections are equal.
    pub max_gap: f32,
}

/// The probability gap of experts `a` and `b` on `probs`, or `None` when the row cannot index
/// one of them.
fn probability_gap(probs: &[f32], a: u32, b: u32) -> Option<f32> {
    let (a, b) = (a as usize, b as usize);
    (a < probs.len() && b < probs.len()).then(|| (probs[a] - probs[b]).abs())
}

/// The experts of `ids` that `others` does not hold, in order.
fn exclusive(ids: &[u32], others: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .filter(|id| !others.contains(id))
        .collect()
}

/// Compares two top-k selections of one softmax router row under a near-tie epsilon. `oracle`
/// and `routed` hold the expert ids in rank order of the two routings of the same token (the
/// host oracle and the device routing of the Q36 parity gate), and `probs` is the probability
/// row that ranked `oracle`.
///
/// The rule (fix5 of packet Q36-BF16-a; decision 0046 A6 item 8): a set difference or a rank
/// swap passes only when the host-row probability gap to the nearest exchanged (or swapped)
/// expert is <= `epsilon`; a missing exchange partner is a fault; ties at exactly equal
/// probability must agree with the lower index; any other difference fails. An id that `probs`
/// cannot index and a selection-width mismatch are faults whatever the epsilon. `max_gap`
/// carries the largest gap the rule saw, tolerated or not, so a caller can report the gap
/// distribution of the mismatching selections.
pub fn compare_top_k(
    oracle: &[u32],
    routed: &[u32],
    probs: &[f32],
    epsilon: f32,
) -> TopKComparison {
    let mut comparison = TopKComparison {
        faults: Vec::new(),
        max_gap: 0.0,
    };
    if oracle.len() != routed.len() {
        comparison.faults.push(RouteFault {
            rank: None,
            oracle: oracle.len() as u32,
            routed: routed.len() as u32,
            gap: None,
        });
        return comparison;
    }
    // The set rule: every expert that one selection holds and the other does not must sit within
    // `epsilon` of the nearest exchanged expert of the other selection.
    for (oracle_side, (mine, theirs)) in [(true, (oracle, routed)), (false, (routed, oracle))] {
        let exchanged = exclusive(theirs, mine);
        for id in exclusive(mine, theirs) {
            let mut nearest: Option<(u32, f32)> = None;
            for other in exchanged.iter().copied() {
                if let Some(gap) = probability_gap(probs, id, other)
                    && nearest.is_none_or(|(_, best)| gap < best)
                {
                    nearest = Some((other, gap));
                }
            }
            match nearest {
                Some((other, gap)) => {
                    if gap <= epsilon {
                        comparison.max_gap = comparison.max_gap.max(gap);
                    } else {
                        comparison.faults.push(RouteFault {
                            rank: None,
                            oracle: if oracle_side { id } else { other },
                            routed: if oracle_side { other } else { id },
                            gap: Some(gap),
                        });
                    }
                }
                None => comparison.faults.push(RouteFault {
                    rank: None,
                    oracle: id,
                    routed: id,
                    gap: None,
                }),
            }
        }
    }
    // The rank rule: a differing rank is a fault unless its two experts sit within `epsilon`.
    for (rank, (&a, &b)) in oracle.iter().zip(routed).enumerate() {
        if a == b {
            continue;
        }
        match probability_gap(probs, a, b) {
            Some(gap) if gap <= epsilon => comparison.max_gap = comparison.max_gap.max(gap),
            Some(gap) => {
                comparison.max_gap = comparison.max_gap.max(gap);
                comparison.faults.push(RouteFault {
                    rank: Some(rank),
                    oracle: a,
                    routed: b,
                    gap: Some(gap),
                });
            }
            None => comparison.faults.push(RouteFault {
                rank: Some(rank),
                oracle: a,
                routed: b,
                gap: None,
            }),
        }
    }
    comparison
}

/// The per-cell measurements of one differing top-k pair under the BF16 co-conditions (fix5
/// of packet Q36-BF16-a; decision 0046 A6 item 8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteCellDeltas {
    /// The largest `|p_a - p_b| / max(p_a, p_b)` over the pair's differing positions, where
    /// `p_a` and `p_b` are the probabilities of the oracle's and the routed expert at that
    /// position on the compared row. Zero when the selections are equal.
    pub max_relative_gap: f32,
    /// The sum of `p_a - p_b` over all of the pair's positions: equal positions cancel, a rank
    /// swap cancels itself (its two positions carry opposite signs), and an exchanged tail
    /// expert leaves the host-minus-device probability of the exchange. Zero when the
    /// selections are equal.
    pub signed_gap: f32,
}

/// Measures one differing pair of selections on the row that ranked `oracle`: the relative
/// gap and the signed host-minus-device probability sum of its differing positions (fix5 of
/// packet Q36-BF16-a; decision 0046 A6 item 8). The pair is one cell of the BF16 route gate;
/// `compare_top_k` decides its faults, this decides its bias report and its relative-gap gate.
/// A position whose ids the row cannot index is skipped (`compare_top_k` already faults it),
/// and a selection width the other lacks is ignored.
pub fn route_cell_deltas(oracle: &[u32], routed: &[u32], probs: &[f32]) -> RouteCellDeltas {
    let mut deltas = RouteCellDeltas {
        max_relative_gap: 0.0,
        signed_gap: 0.0,
    };
    for (&a, &b) in oracle.iter().zip(routed) {
        if a == b {
            continue;
        }
        let (ia, ib) = (a as usize, b as usize);
        if ia >= probs.len() || ib >= probs.len() {
            continue;
        }
        let signed = probs[ia] - probs[ib];
        deltas.signed_gap += signed;
        let max = probs[ia].max(probs[ib]);
        if max > 0.0 {
            deltas.max_relative_gap = deltas.max_relative_gap.max(signed.abs() / max);
        }
    }
    deltas
}

/// True when a differing cell at (`step`, `layer`) breaks the origin rule of the BF16 run (fix5
/// of packet Q36-BF16-a; decision 0046 A6 item 8): the F32 summation-order noise starts after
/// the first layers, so no cell of layers 0 to 2 of step 0 may differ. The d1 run first
/// mismatched at step 0 layer 3.
pub fn origin_fault(step: usize, layer: usize) -> bool {
    step == 0 && layer <= 2
}

/// True when the run's fraction of differing cells passes its gate (fix5 of packet
/// Q36-BF16-a; decision 0046 A6 item 8). A run that compared nothing differs in nothing.
pub fn differing_fraction_passes(compared: usize, differing: usize, limit: f64) -> bool {
    if compared == 0 {
        return true;
    }
    differing as f64 / compared as f64 <= limit
}

/// True when the run's signed mean of the per-cell host-minus-device probability sums passes
/// its gate (fix5 of packet Q36-BF16-a; decision 0046 A6 item 8). A NaN fails.
pub fn signed_mean_passes(signed_mean: f64, limit: f64) -> bool {
    signed_mean.is_finite() && signed_mean.abs() <= limit
}

/// True for a finite cosine at or above `floor` (fix5 of packet Q36-BF16-a): the shared
/// NaN-rejecting form of the logits-cosine floors of the route gate, the exact 0.99999 floor
/// for the `f32` control and 0.997 for BF16. A NaN fails both.
pub fn cosine_passes(cosine: f64, floor: f64) -> bool {
    cosine.is_finite() && cosine >= floor
}

/// True when a measured minimum keeps its floored anchor (fix5 of packet Q36-BF16-a; decision
/// 0046 A6): finite, at least `floor`, and no more than `regression` below `anchor`. The BF16
/// parity minimum keeps the anchor 0.9799 that the d1 run observed with floor 0.95 and
/// regression 0.01; a report-only rule is not allowed to ratchet.
pub fn keeps_floored_anchor(value: f64, floor: f64, anchor: f64, regression: f64) -> bool {
    value.is_finite() && value >= floor && value >= anchor - regression
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The production near-tie epsilon of the BF16 route rule (`BF16_TIE_EPSILON` in
    /// `qwen36/resident/route_parity.rs`, fix4 of packet Q36-BF16-a): a probability gap about
    /// twice the 2.55e-3 maximum that the d1 BF16 gate observed over its 600 mismatching cells
    /// of 1,440. A gap of 4e-3 passes it and a gap of 6e-3 fails it.
    const TIE_EPSILON: f32 = 5e-3;
    /// The production relative-gap gate of the BF16 route rule (`BF16_RELATIVE_GAP_LIMIT` in
    /// `qwen36/resident/route_parity.rs`, fix5 of packet Q36-BF16-a): 0.5, the first gate. The
    /// d1 evidence records absolute gaps only, so the observed maximum relative gap is to
    /// tighten it after the next Mac run.
    const RELATIVE_GAP_LIMIT: f32 = 0.5;
    /// The production differing-fraction gate of the BF16 route rule
    /// (`BF16_DIFFERING_FRACTION_LIMIT` in `qwen36/resident/route_parity.rs`, fix5 of packet
    /// Q36-BF16-a): 60% against the 42% that the d1 run observed.
    const DIFFERING_FRACTION_LIMIT: f64 = 0.6;
    /// The production signed-mean gate of the BF16 route rule (`BF16_SIGNED_MEAN_LIMIT` in
    /// `qwen36/resident/route_parity.rs`, fix5 of packet Q36-BF16-a): epsilon/5, 1e-3.
    const SIGNED_MEAN_LIMIT: f64 = 1e-3;
    /// The production logits-cosine floors of the route gate (`MIN_COSINE` and
    /// `BF16_MIN_COSINE` in `qwen36/resident/route_parity.rs`): the exact 0.99999 floor of the
    /// `f32` control and the 0.997 BF16 floor of fix5, 1.64e-3 under the 0.99864 minimum that
    /// the d1 run observed.
    const F32_LOGITS_FLOOR: f64 = 0.999_99;
    const BF16_LOGITS_FLOOR: f64 = 0.997;
    /// The production floor, anchor, and regression of the BF16 parity minimum
    /// (`BF16_LAYER_COSINE_FLOOR`, `D1_BF16_MIN_LAYER_COSINE`, and
    /// `BF16_LAYER_COSINE_REGRESSION` in `qwen36/resident/parity.rs`, fix5 of packet
    /// Q36-BF16-a): 0.95 under the 0.9799 that the d1 run observed, with 0.01 of allowed
    /// regression.
    const LAYER_FLOOR: f64 = 0.95;
    const D1_LAYER_MINIMUM: f64 = 0.979_9;
    const LAYER_REGRESSION: f64 = 0.01;

    /// A probability row of 16 experts whose top four are 7 (0.5), 5 (0.3), 3 (0.2), and
    /// 11 (0.2 - 4e-3) in rank order: 3 and 11 are a near-tie of 4e-3, every other pair is
    /// clear, and expert 2 (0.001) sits far below the boundary.
    fn row() -> Vec<f32> {
        let mut probs = vec![0.001f32; 16];
        probs[7] = 0.5;
        probs[5] = 0.3;
        probs[3] = 0.2;
        probs[11] = 0.2 - 4e-3;
        probs
    }

    /// The gap of the 3/11 near-tie, computed the way `compare_top_k` computes it.
    fn tie_gap(probs: &[f32]) -> f32 {
        (probs[3] - probs[11]).abs()
    }

    #[test]
    fn the_health_check_rejects_a_non_finite_layer_and_passes_a_finite_stack() {
        // A finite stack of layer sums, as a healthy chunk of four layers reads back: every value
        // is 1 within rounding and the check passes.
        let healthy = [1.0, 0.9999999, 1.0000001, 1.0];
        assert_eq!(check_health(&healthy), Ok(()), "a finite stack passes");
        assert_eq!(check_health(&[]), Ok(()), "an empty stack passes");
        // A NaN, either infinity, and a NaN anywhere but the first layer all give the same
        // `Backend` error that `route_experts` gives for a bad row. The middle and last cases
        // keep the first layer finite, so a check that reads only the first value fails here.
        for (what, bad) in [
            ("a NaN first", [f32::NAN, 1.0, 1.0, 1.0]),
            ("a positive infinity last", [1.0, 1.0, 1.0, f32::INFINITY]),
            (
                "a negative infinity middle",
                [1.0, f32::NEG_INFINITY, 1.0, 1.0],
            ),
            ("a NaN middle", [1.0, 1.0, f32::NAN, 1.0]),
            ("a NaN last", [1.0, 1.0, 1.0, f32::NAN]),
        ] {
            let failed = check_health(&bad);
            assert_eq!(
                failed,
                Err(EngError::Backend {
                    operation: "route_experts",
                    message: "router probabilities must be finite and non-negative".into(),
                }),
                "{what}"
            );
        }
    }

    #[test]
    fn the_near_tie_rule_tolerates_only_ties() {
        let probs = row();
        // Equal selections carry no fault and see no gap.
        let equal = compare_top_k(&[7, 3, 11, 5], &[7, 3, 11, 5], &probs, TIE_EPSILON);
        assert_eq!(
            equal,
            TopKComparison {
                faults: Vec::new(),
                max_gap: 0.0
            }
        );
        // A rank swap of the near-tied 3 and 11, a 4e-3 gap, passes the production epsilon, and
        // the report sees the gap.
        let swapped = compare_top_k(&[7, 3, 11, 5], &[7, 11, 3, 5], &probs, TIE_EPSILON);
        assert_eq!(swapped.faults, Vec::new(), "the near-tie swap is tolerated");
        assert!(
            (swapped.max_gap - tie_gap(&probs)).abs() <= 1e-9,
            "{swapped:?}"
        );
        // The same swap on a 6e-3 gap is beyond the production epsilon: a fault at both of its
        // ranks, so the gate fails for any gap above the epsilon.
        let mut wide = row();
        wide[11] = 0.2 - 6e-3;
        let beyond = compare_top_k(&[7, 3, 11, 5], &[7, 11, 3, 5], &wide, TIE_EPSILON);
        assert_eq!(
            beyond.faults,
            vec![
                RouteFault {
                    rank: Some(1),
                    oracle: 3,
                    routed: 11,
                    gap: Some(tie_gap(&wide))
                },
                RouteFault {
                    rank: Some(2),
                    oracle: 11,
                    routed: 3,
                    gap: Some(tie_gap(&wide))
                },
            ],
            "a 6e-3 gap is beyond the epsilon"
        );
        // The same 4e-3 swap against a smaller epsilon is a rank fault at both of its ranks.
        let strict = compare_top_k(&[7, 3, 11, 5], &[7, 11, 3, 5], &probs, 1e-5);
        assert_eq!(
            strict.faults,
            vec![
                RouteFault {
                    rank: Some(1),
                    oracle: 3,
                    routed: 11,
                    gap: Some(tie_gap(&probs))
                },
                RouteFault {
                    rank: Some(2),
                    oracle: 11,
                    routed: 3,
                    gap: Some(tie_gap(&probs))
                },
            ],
            "beyond the epsilon the swap is a fault"
        );
        // A swap of clearly separated experts (7 and 5) is a fault at both ranks.
        let clear_gap = (probs[7] - probs[5]).abs();
        let clear = compare_top_k(&[7, 3, 11, 5], &[5, 3, 11, 7], &probs, TIE_EPSILON);
        assert_eq!(
            clear
                .faults
                .iter()
                .map(|fault| (fault.rank, fault.gap))
                .collect::<Vec<_>>(),
            vec![(Some(0), Some(clear_gap)), (Some(3), Some(clear_gap))],
            "a clear swap is a fault"
        );
        // An exchanged expert at the selection boundary (5 leaves, 2 enters, gap 0.299) is a
        // tolerated tie within a wide epsilon and three faults beyond a narrow one.
        let exchanged = compare_top_k(&[7, 3, 11, 5], &[7, 3, 11, 2], &probs, 1.0);
        assert_eq!(
            exchanged.faults,
            Vec::new(),
            "a tolerated boundary exchange"
        );
        let refused = compare_top_k(&[7, 3, 11, 5], &[7, 3, 11, 2], &probs, TIE_EPSILON);
        assert_eq!(refused.faults.len(), 3, "{refused:?}");
        assert_eq!(
            refused
                .faults
                .iter()
                .filter(|fault| fault.rank.is_none())
                .count(),
            2,
            "one set fault per side"
        );
        // An id the row cannot index is always a fault, whatever the epsilon.
        let outside = compare_top_k(&[7, 3, 11, 16], &[7, 3, 11, 5], &probs, 100.0);
        assert_eq!(outside.faults.len(), 3, "{outside:?}");
        assert!(outside.faults.iter().all(|fault| fault.gap.is_none()));
        // A width mismatch is a fault of its own.
        let widths = compare_top_k(&[7, 3], &[7, 3, 11], &probs, 100.0);
        assert_eq!(widths.faults.len(), 1, "{widths:?}");
        assert_eq!(widths.faults[0].rank, None);
    }

    #[test]
    fn the_cell_deltas_report_the_relative_gap_and_the_signed_bias() {
        let probs = row();
        // Equal selections carry no relative gap and no signed bias.
        assert_eq!(
            route_cell_deltas(&[7, 3, 11, 5], &[7, 3, 11, 5], &probs),
            RouteCellDeltas {
                max_relative_gap: 0.0,
                signed_gap: 0.0
            }
        );
        // The 3/11 near-tie swap: the relative gap of the swapped pair (4e-3 over 0.2) and no
        // signed bias, because the two swapped positions carry opposite signs.
        let swapped = route_cell_deltas(&[7, 3, 11, 5], &[7, 11, 3, 5], &probs);
        assert!(
            (swapped.max_relative_gap - tie_gap(&probs) / probs[3]).abs() <= 1e-9,
            "{swapped:?}"
        );
        assert!(swapped.signed_gap.abs() <= 1e-9, "{swapped:?}");
        assert!(
            swapped.max_relative_gap <= RELATIVE_GAP_LIMIT,
            "the near tie passes the relative gate"
        );
        // The 5/2 boundary exchange: a relative gap (0.299 over 0.3) above the 0.5 gate, and
        // the host-minus-device probability of the exchange as its signed bias.
        let exchanged = route_cell_deltas(&[7, 3, 11, 5], &[7, 3, 11, 2], &probs);
        assert!(
            (exchanged.max_relative_gap - (probs[5] - probs[2]) / probs[5]).abs() <= 1e-6,
            "{exchanged:?}"
        );
        assert!(
            (exchanged.signed_gap - (probs[5] - probs[2])).abs() <= 1e-9,
            "{exchanged:?}"
        );
        assert!(
            exchanged.max_relative_gap > RELATIVE_GAP_LIMIT,
            "the exchange fails the relative gate"
        );
        // A position whose ids the row cannot index is skipped: `compare_top_k` faults it.
        assert_eq!(
            route_cell_deltas(&[7, 3, 11, 5], &[7, 3, 11, 16], &probs),
            RouteCellDeltas {
                max_relative_gap: 0.0,
                signed_gap: 0.0
            }
        );
    }

    #[test]
    fn the_origin_rule_faults_only_layers_0_to_2_of_step_0() {
        // The d1 BF16 run first mismatched at step 0 layer 3 (evidence 907030d6): the rule
        // faults every differing cell before that and nothing else.
        for layer in 0..3 {
            assert!(
                origin_fault(0, layer),
                "step 0 layer {layer} is within the origin"
            );
        }
        assert!(
            !origin_fault(0, 3),
            "layer 3 is the first mismatch of the d1 run"
        );
        for layer in 0..40 {
            assert!(
                !origin_fault(1, layer),
                "step 1 layer {layer} is past the origin"
            );
        }
    }

    #[test]
    fn the_bias_gates_hold_the_differing_fraction_and_the_signed_mean() {
        // 600 of 1,440 cells (42%) differed in the d1 run; the gate is 60%, and its boundary
        // passes.
        assert!(differing_fraction_passes(
            1_440,
            600,
            DIFFERING_FRACTION_LIMIT
        ));
        assert!(
            differing_fraction_passes(1_440, 864, DIFFERING_FRACTION_LIMIT),
            "864 of 1,440 is exactly the gate"
        );
        assert!(!differing_fraction_passes(
            1_440,
            865,
            DIFFERING_FRACTION_LIMIT
        ));
        assert!(
            differing_fraction_passes(0, 0, DIFFERING_FRACTION_LIMIT),
            "nothing compared differs in nothing"
        );
        // The signed-mean gate is epsilon/5 = 1e-3 on the absolute value; a NaN fails it.
        assert!(signed_mean_passes(1e-3, SIGNED_MEAN_LIMIT));
        assert!(signed_mean_passes(-1e-3, SIGNED_MEAN_LIMIT));
        assert!(!signed_mean_passes(1.1e-3, SIGNED_MEAN_LIMIT));
        assert!(!signed_mean_passes(-1.1e-3, SIGNED_MEAN_LIMIT));
        assert!(!signed_mean_passes(f64::NAN, SIGNED_MEAN_LIMIT));
    }

    #[test]
    fn the_logits_cosine_floor_is_finite_and_above_the_floor_of_the_dtype() {
        // The f32 control keeps the exact 0.99999 floor of decision 0046 item 4.
        assert!(cosine_passes(F32_LOGITS_FLOOR, F32_LOGITS_FLOOR));
        assert!(!cosine_passes(0.999_98, F32_LOGITS_FLOOR));
        // The BF16 floor is 0.997 (fix5): the d1 minimum 0.99864 passes with a 1.64e-3 margin,
        // and a cosine below the floor fails.
        assert!(cosine_passes(0.998_64, BF16_LOGITS_FLOOR));
        assert!(cosine_passes(BF16_LOGITS_FLOOR, BF16_LOGITS_FLOOR));
        assert!(!cosine_passes(0.996_9, BF16_LOGITS_FLOOR));
        // A NaN fails both floors.
        assert!(!cosine_passes(f64::NAN, F32_LOGITS_FLOOR));
        assert!(!cosine_passes(f64::NAN, BF16_LOGITS_FLOOR));
    }

    #[test]
    fn the_bf16_layer_floor_admits_the_d1_minimum_and_refuses_a_regression() {
        // The d1 BF16 parity run saw a minimum layer cosine of 0.9799: it passes, and so does
        // the exact 0.01 regression boundary under it.
        assert!(keeps_floored_anchor(
            D1_LAYER_MINIMUM,
            LAYER_FLOOR,
            D1_LAYER_MINIMUM,
            LAYER_REGRESSION
        ));
        assert!(keeps_floored_anchor(
            D1_LAYER_MINIMUM - LAYER_REGRESSION,
            LAYER_FLOOR,
            D1_LAYER_MINIMUM,
            LAYER_REGRESSION
        ));
        // A regression beyond 0.01 under the d1 minimum fails, and so does the 0.95 floor and
        // a NaN.
        assert!(!keeps_floored_anchor(
            0.969_8,
            LAYER_FLOOR,
            D1_LAYER_MINIMUM,
            LAYER_REGRESSION
        ));
        assert!(!keeps_floored_anchor(
            0.94,
            LAYER_FLOOR,
            D1_LAYER_MINIMUM,
            LAYER_REGRESSION
        ));
        assert!(!keeps_floored_anchor(
            f64::NAN,
            LAYER_FLOOR,
            D1_LAYER_MINIMUM,
            LAYER_REGRESSION
        ));
    }
}
