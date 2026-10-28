//! Portable F32 NemotronH block forwards, without cache or backend state.

mod attention;

pub use attention::attention_block_forward;
pub(crate) use attention::attention_block_step_with_projector;
pub mod mamba;
pub mod moe;

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

/// NemotronHRMSNorm.forward lines 585-590: F32 RMS, then weight*x, not (1+weight)*x.
pub(crate) fn rms_norm(hidden: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, EngError> {
    let invalid = |message: &str| EngError::BadRequest(format!("NemotronH norm: {message}"));
    let &[tokens, width] = hidden.shape() else {
        return Err(invalid("hidden must have shape [T, hidden_size]"));
    };
    if tokens == 0 || width == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(invalid(
            "dimensions and epsilon must be positive and finite",
        ));
    }
    if hidden.dtype() != Dtype::F32 || weight.dtype() != Dtype::F32 || weight.shape() != [width] {
        return Err(invalid("expected F32 hidden and norm weight [hidden_size]"));
    }
    let len = (tokens as usize)
        .checked_mul(width as usize)
        .filter(|n| {
            n.checked_mul(4)
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| invalid("tensor byte length exceeds host limits"))?;
    let input = hidden
        .host_bytes()
        .ok_or_else(|| invalid("hidden is not host-backed"))?;
    let weights = weight
        .host_bytes()
        .ok_or_else(|| invalid("weight is not host-backed"))?;
    let input: Vec<_> = input
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let weights: Vec<_> = weights
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if input.iter().chain(&weights).any(|v| !v.is_finite()) {
        return Err(invalid("input tensors must contain finite values"));
    }
    let mut normalized = Vec::with_capacity(len * 4);
    for row in input.chunks_exact(width as usize) {
        let mean = row.iter().map(|x| x * x).sum::<f32>() / width as f32;
        let variance = mean + eps;
        if !variance.is_finite() {
            return Err(invalid("RMS variance is not finite"));
        }
        let inverse_rms = variance.sqrt().recip();
        normalized.extend(
            row.iter()
                .zip(&weights)
                .flat_map(|(x, w)| (w * (x * inverse_rms)).to_le_bytes()),
        );
    }
    Tensor::try_from_host(Dtype::F32, vec![tokens, width], normalized)
}
