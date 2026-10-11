use crate::EngError;

use super::spec::Router;

const OPERATION: &str = "route_experts";

#[derive(Debug, Clone, PartialEq)]
pub struct Routed {
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Selects the top-k experts from softmax router probabilities.
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
    let Router::SoftmaxTopK { renorm } = *router else {
        return Err(EngError::Checkpoint(
            "router kind is not softmax top-k".into(),
        ));
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
