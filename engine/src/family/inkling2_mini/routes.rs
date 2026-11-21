//! Portable routing for the fixed Inkling 2 Mini expert layout.

use super::{EXPERTS, ROUTE_SCALE, TOP_K};
use crate::EngError;

/// The selected expert IDs and weights in token-major order, with eight entries per token.
#[derive(Debug)]
pub struct InklingRoutes {
    expert_ids: Vec<usize>,
    weights: Vec<f32>,
}

impl InklingRoutes {
    /// Selects experts from FP32 router logits and a shared FP32 selection bias.
    pub fn from_logits(logits: &[[f32; EXPERTS]], bias: &[f32; EXPERTS]) -> Result<Self, EngError> {
        Self::from_rows(logits.iter().copied(), bias)
    }

    pub(crate) fn from_rows(
        logits: impl ExactSizeIterator<Item = [f32; EXPERTS]>,
        bias: &[f32; EXPERTS],
    ) -> Result<Self, EngError> {
        if bias.iter().any(|value| !value.is_finite()) {
            return Err(EngError::BadRequest(
                "Inkling router selection bias must be finite".into(),
            ));
        }
        let capacity_error =
            || EngError::MemoryPlan("Inkling route storage exceeds host limits".into());
        let count = logits.len().checked_mul(TOP_K).ok_or_else(capacity_error)?;
        let mut expert_ids = Vec::new();
        let mut weights = Vec::new();
        expert_ids
            .try_reserve_exact(count)
            .map_err(|_| capacity_error())?;
        weights
            .try_reserve_exact(count)
            .map_err(|_| capacity_error())?;

        for row in logits {
            if row.iter().any(|value| !value.is_finite()) {
                return Err(EngError::BadRequest(
                    "Inkling router logits must be finite".into(),
                ));
            }
            let mut ranked: [(usize, f32); EXPERTS] = std::array::from_fn(|expert| {
                let sigmoid = 1.0 / (1.0 + (-row[expert]).exp());
                (expert, sigmoid + bias[expert])
            });
            let compare = |a: &(usize, f32), b: &(usize, f32)| {
                b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0))
            };
            let (selected, _, _) = ranked.select_nth_unstable_by(TOP_K, compare);
            selected.sort_unstable_by(compare);

            let log_scores: [f32; TOP_K] = std::array::from_fn(|slot| {
                let logit = row[selected[slot].0];
                logit.min(0.0) - (-logit.abs()).exp().ln_1p()
            });
            let max = log_scores.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let shifted = log_scores.map(|log_score| (log_score - max).exp());
            // Match the reference top-eight reduction tree. Normalize in the shifted
            // domain so large negative logits cannot swallow the normalization term.
            let sum = ((shifted[0] + shifted[2]) + (shifted[1] + shifted[3]))
                + ((shifted[4] + shifted[6]) + (shifted[5] + shifted[7]));
            expert_ids.extend(selected.iter().map(|&(id, _)| id));
            weights.extend(shifted.map(|score| (score / sum) * ROUTE_SCALE));
        }

        Ok(Self {
            expert_ids,
            weights,
        })
    }

    pub fn rows(&self) -> usize {
        self.expert_ids.len() / TOP_K
    }

    pub fn expert_ids(&self) -> &[usize] {
        &self.expert_ids
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_rank_by_corrected_sigmoid_with_lower_id_ties() {
        let mut row = [-10.0; EXPERTS];
        row[9] = 1.0;
        row[11] = 1.0;
        row[12] = 2.0;
        let mut bias = [0.0; EXPERTS];
        bias[20] = 1.0;
        bias[11] = 0.01;
        let routes = InklingRoutes::from_logits(&[row], &bias).unwrap();
        assert_eq!(routes.expert_ids(), &[20, 12, 11, 9, 0, 1, 2, 3]);
        assert_eq!(routes.weights()[2], routes.weights()[3]);
        assert_eq!(routes.weights()[0], routes.weights()[4]);
        assert!(routes.weights()[1] > routes.weights()[2]);
        assert!((routes.weights().iter().sum::<f32>() - ROUTE_SCALE).abs() < 1e-6);
    }

    #[test]
    fn routes_keep_unbiased_weights_in_token_major_order() {
        let mut rows = [[-10.0; EXPERTS]; 2];
        for expert in 0..TOP_K {
            rows[0][expert] = (expert + 1) as f32;
            rows[1][EXPERTS - 1 - expert] = (expert + 1) as f32;
        }
        let mut bias = [0.0; EXPERTS];
        bias[0] = 0.5;
        let baseline = InklingRoutes::from_logits(&rows, &[0.0; EXPERTS]).unwrap();
        let routes = InklingRoutes::from_logits(&rows, &bias).unwrap();
        assert_eq!(routes.rows(), 2);
        assert_eq!(
            routes.expert_ids(),
            &[
                0, 7, 6, 5, 4, 3, 2, 1, 120, 121, 122, 123, 124, 125, 126, 127
            ]
        );
        let mut expected = baseline.weights().to_vec();
        expected[..TOP_K].rotate_right(1);
        for (&actual, expected) in routes.weights().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn routes_normalize_extreme_finite_logits() {
        let mut varied = [-1000.0; EXPERTS];
        varied[7] = -998.0;
        let routes = InklingRoutes::from_logits(
            &[[-f32::MAX; EXPERTS], [f32::MAX; EXPERTS], varied],
            &[0.0; EXPERTS],
        )
        .unwrap();
        assert_eq!(&routes.weights()[..2 * TOP_K], &[1.0; 2 * TOP_K]);
        assert_eq!(&routes.expert_ids()[2 * TOP_K..], &[0, 1, 2, 3, 4, 5, 6, 7]);
        let varied_weights = &routes.weights()[2 * TOP_K..];
        assert!((varied_weights[7] / varied_weights[0] - 2.0_f32.exp()).abs() < 1e-6);
    }

    #[test]
    fn routes_reject_nonfinite_inputs_and_accept_empty_batches() {
        let mut bias = [0.0; EXPERTS];
        bias[0] = f32::NAN;
        assert!(InklingRoutes::from_logits(&[], &bias).is_err());
        let empty = InklingRoutes::from_logits(&[], &[0.0; EXPERTS]).unwrap();
        assert_eq!(empty.rows(), 0);
        assert!(empty.expert_ids().is_empty());
        assert!(empty.weights().is_empty());
        assert!(InklingRoutes::from_logits(&[[f32::INFINITY; EXPERTS]], &[0.0; EXPERTS]).is_err());
    }
}
