//! The routed-expert compensation of Flint EXL3 packs.
//!
//! The pack stores every routed expert's `up_proj` divided by `expert_interm_div` so that the
//! SwiGLU activation (about 2.6e5 in the last layers) stays inside binary16. The runtime undoes
//! it by multiplying the normalized routing weights of the selected experts by the same factor.
//! The shared expert, the dense MLPs and every other projection are stored undivided.

use crate::{EngError, checkpoint_json::UniqueValue, family::Routed, weights::exl3::Config};

/// The divisor of the published Flint packs.
const FLINT_EXPERT_INTERM_DIV: u64 = 128;

/// The factor that multiplies the routing weights of the routed experts, after normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RoutedScale(f32);

impl RoutedScale {
    /// Reads `quantization_config.expert_interm_div` from the bytes of the model `config.json`.
    ///
    /// `Config::from_json` runs first: size cap, unique member names, `quant_method`, version
    /// and the 1-or-128 domain. It reads an absent member as 1, which for a Flint pack would be
    /// a silent 1/128 error, so this reader also requires the member to be present and equal to
    /// 128, and refuses a member of the same name at the top level (a wrong location).
    pub(crate) fn from_model_config(model_config: &[u8]) -> Result<Self, EngError> {
        Config::from_json(model_config)?;
        let root = serde_json::from_slice::<UniqueValue>(model_config)
            .map_err(|error| invalid(&format!("config JSON: {error}")))?
            .0;
        if root.get("expert_interm_div").is_some() {
            return Err(invalid(
                "expert_interm_div must be nested in quantization_config, not at the top level",
            ));
        }
        let stated = root
            .get("quantization_config")
            .and_then(|config| config.get("expert_interm_div"))
            .ok_or_else(|| invalid("expert_interm_div is missing from quantization_config"))?;
        if stated.as_u64() != Some(FLINT_EXPERT_INTERM_DIV) {
            return Err(invalid(&format!(
                "expert_interm_div must be {FLINT_EXPERT_INTERM_DIV}, found {stated}"
            )));
        }
        Ok(Self(FLINT_EXPERT_INTERM_DIV as f32))
    }

    pub(crate) fn factor(self) -> f32 {
        self.0
    }

    /// Multiplies the normalized weights of an existing route; the route itself is not changed.
    ///
    /// The factor is a power of two, so each product is exact in F32.
    pub(super) fn compensate(self, routed: &Routed) -> Result<Routed, EngError> {
        if routed.indices.len() != routed.weights.len() {
            return Err(EngError::BadRequest(format!(
                "route has {} indices and {} weights",
                routed.indices.len(),
                routed.weights.len()
            )));
        }
        let weights = routed
            .weights
            .iter()
            .map(|&weight| {
                let scaled = weight * self.0;
                if weight.is_finite() && weight >= 0.0 && scaled.is_finite() {
                    Ok(scaled)
                } else {
                    Err(EngError::Backend {
                        operation: "route_experts",
                        message: "routing weights must be finite and non-negative".into(),
                    })
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Routed {
            indices: routed.indices.clone(),
            weights,
        })
    }
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 Flint config: {message}"))
}
