//! Portable Flint routing contract. Inputs are router logits, not softmax probabilities.

use crate::{
    EngError,
    family::{
        route::{Routed, route_experts},
        spec::Router,
    },
};

/// Normalized sigmoid top-6 over the published 256 experts. Equal sigmoid scores
/// choose the lower index, including ties caused by F32 sigmoid saturation.
/// Reject non-finite logits *before* sigmoid can hide an infinity.
pub fn route(logits: &[f32]) -> Result<Routed, EngError> {
    if logits.len() != 256 {
        return Err(EngError::BadRequest(
            "Flint router requires 256 logits".into(),
        ));
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err(EngError::Backend {
            operation: "route_experts",
            message: "Flint router logits must be finite".into(),
        });
    }
    let scores: Vec<_> = logits
        .iter()
        .map(|&x| super::super::flash_next::math::sigmoid(x))
        .collect();
    route_experts(&scores, &Router::SigmoidTopK, 6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_top6_matches_independent_f64_oracle() {
        let logits: Vec<_> = (0..256)
            .map(|i| ((i * 37) % 256) as f32 / 32.0 - 4.0)
            .collect();
        let actual = route(&logits).unwrap();
        let mut ids: Vec<usize> = (0..256).collect();
        ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        ids.truncate(6);
        assert_eq!(
            actual.indices,
            ids.iter().map(|&i| i as u32).collect::<Vec<_>>()
        );
        let scores: Vec<_> = ids
            .iter()
            .map(|&i| 1.0 / (1.0 + (-(logits[i] as f64)).exp()))
            .collect();
        let sum: f64 = scores.iter().sum();
        for (got, score) in actual.weights.iter().zip(scores) {
            assert!((*got as f64 - score / sum).abs() < 3e-8);
        }
        // Softmax of these logits is not sigmoid routing.
        let softmax_ratio = (logits[ids[0]] - logits[ids[5]]).exp();
        assert!((actual.weights[0] / actual.weights[5] - softmax_ratio).abs() > 0.1);
    }

    #[test]
    fn ties_saturation_and_normalization_are_deterministic() {
        for logit in [0.0, 100.0] {
            let actual = route(&[logit; 256]).unwrap();
            assert_eq!(actual.indices, [0, 1, 2, 3, 4, 5]);
            assert_eq!(actual.weights, [1.0 / 6.0; 6]);
        }
    }

    #[test]
    fn invalid_logits_and_zero_selected_mass_fail_closed() {
        assert!(matches!(route(&[0.0; 255]), Err(EngError::BadRequest(_))));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut row = [0.0; 256];
            row[255] = bad;
            assert!(matches!(route(&row), Err(EngError::Backend { .. })));
        }
        assert!(matches!(
            route(&[-1000.0; 256]),
            Err(EngError::Backend { .. })
        ));
    }

    #[test]
    fn qwen_probability_routing_is_unchanged() {
        let row = [0.1, 0.4, 0.3, 0.2];
        let raw = route_experts(&row, &Router::SoftmaxTopK { renorm: false }, 2).unwrap();
        assert_eq!(raw.indices, [1, 2]);
        assert_eq!(raw.weights, [0.4, 0.3]);
        let normalized = route_experts(&row, &Router::SoftmaxTopK { renorm: true }, 2).unwrap();
        assert!((normalized.weights[0] - 4.0 / 7.0).abs() < 1e-7);
        assert!(route_experts(&row, &Router::SigmoidGroup { groups: 2 }, 2).is_err());
    }
}
