//! One logits row to one scored position (M3, registration v4).
//!
//! Pure host math shared by every arm's device loop: the true-token
//! log-probability and the greedy argmax of one f32 logits row, through the
//! shared row math of `family::teacher_forced`. Each arm is
//! scored against its OWN full distribution: normalization runs over every
//! head row the arm computes, padding rows included (the head width pads the
//! true vocabulary to the same width in both arms, so the effect is
//! identical, and the true id is always inside the true vocabulary —
//! disclosed in the registration). All sums run in f64. Non-finite logits
//! are a hard error in the runner; here NaN simply never ranks and NaN
//! propagates into the log-probability the runner checks.

// Pure host math, but its only callers are test-gated (the scorer's tests
// and the device harness), so a non-test build sees it unreferenced.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the M3 harness legs")
)]

use super::m3_quality::PositionRow;
pub(in super::super) use crate::family::teacher_forced::argmax_low_id;
use crate::family::teacher_forced::logprob_true;

/// The scored-position row of one logits row: true-token log-probability
/// (f32 for storage; the scorer's bootstrap re-widens to f64) and the greedy
/// argmax equality on the true token.
pub(in super::super) fn position_row(
    row: &[f32],
    true_id: u32,
    cluster: u32,
) -> Result<PositionRow, &'static str> {
    Ok(PositionRow {
        cluster,
        logprob_true: logprob_true(row, true_id)? as f32,
        top1_correct: argmax_low_id(row)? == true_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runner's storage type round-trips the scored row: f32 storage is
    /// the registration's precision for per-position rows.
    #[test]
    fn position_row_carries_cluster_and_greedy_equality() {
        let row = [0.0f32, 2.0f32.ln(), 0.0];
        let good = position_row(&row, 1, 7).unwrap();
        assert_eq!(good.cluster, 7);
        assert!(good.top1_correct);
        assert!((good.logprob_true - (0.5f32).ln()).abs() < 1e-6);
        let bad = position_row(&row, 0, 8).unwrap();
        assert!(!bad.top1_correct);
    }
}
