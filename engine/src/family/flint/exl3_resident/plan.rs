use std::{collections::HashMap, path::Path};

use super::{FfnRoles, FlintResidentLimits, LayerRoles, ResidentPlan};
use crate::{
    EngError,
    weights::exl3::{
        Checkpoint,
        flint_pack::{Geometry, Kind, Limits, Plan, RoutedScale, audit},
    },
};

pub(super) fn plan_resident(
    directory: &Path,
    checkpoint: &Checkpoint,
    geometry: &Geometry,
    limits: FlintResidentLimits,
) -> Result<ResidentPlan, EngError> {
    let layout = geometry.plan();
    let (stored_payload_bytes, converted_payload_bytes) = payload_bytes(&layout)?;
    if converted_payload_bytes > limits.max_weight_bytes {
        return Err(EngError::MemoryPlan(format!(
            "Flint EXL3 converted payload requires {converted_payload_bytes} bytes, budget {}",
            limits.max_weight_bytes
        )));
    }
    let mut audit_limits = Limits::default();
    audit_limits.batch_bytes = audit_limits.batch_bytes.min(limits.max_staging_bytes);
    audit_limits.matrix_bytes = audit_limits.matrix_bytes.min(limits.max_staging_bytes);
    audit(directory, checkpoint, geometry, &audit_limits)?;
    let factor = RoutedScale::from_model_config(&checkpoint.model_config)?.factor();
    let names = Names::new(&layout);
    let layers = (0..geometry.layers)
        .map(|layer| names.layer(geometry, layer, factor))
        .collect::<Result<_, _>>()?;
    let embedding = names.plain("model.embed_tokens.weight")?;
    let final_norm = names.plain("model.norm.weight")?;
    let head = names.matrix("lm_head")?;
    Ok(ResidentPlan {
        layout,
        layers,
        embedding,
        final_norm,
        head,
        stored_payload_bytes,
        converted_payload_bytes,
    })
}

fn payload_bytes(layout: &Plan) -> Result<(u64, u64), EngError> {
    let overflow = || EngError::MemoryPlan("Flint EXL3 payload byte count overflow".into());
    let (mut stored, mut converted) = (0u64, 0u64);
    for spec in layout.specs() {
        let raw = spec.bytes();
        let device = match spec.kind {
            Kind::Mul1 => 0,
            Kind::Trellis => raw,
            _ => raw.checked_mul(2).ok_or_else(overflow)?,
        };
        stored = stored.checked_add(raw).ok_or_else(overflow)?;
        converted = converted.checked_add(device).ok_or_else(overflow)?;
    }
    Ok((stored, converted))
}

// Index the existing schema once; expert banks must not perform quadratic name searches.
struct Names<'a> {
    matrices: HashMap<&'a str, usize>,
    plain: HashMap<&'a str, usize>,
}

impl<'a> Names<'a> {
    fn new(layout: &'a Plan) -> Self {
        Self {
            matrices: layout
                .modules
                .iter()
                .enumerate()
                .map(|(index, module)| (module.prefix.as_str(), index))
                .collect(),
            plain: layout
                .plain
                .iter()
                .enumerate()
                .map(|(index, tensor)| (tensor.name.as_str(), index))
                .collect(),
        }
    }

    fn matrix(&self, name: &str) -> Result<usize, EngError> {
        required(&self.matrices, name)
    }

    fn plain(&self, name: &str) -> Result<usize, EngError> {
        required(&self.plain, name)
    }

    fn expert(&self, prefix: &str) -> Result<[usize; 3], EngError> {
        Ok([
            self.matrix(&format!("{prefix}.gate_proj"))?,
            self.matrix(&format!("{prefix}.up_proj"))?,
            self.matrix(&format!("{prefix}.down_proj"))?,
        ])
    }

    fn ffn(&self, geometry: &Geometry, layer: usize, factor: f32) -> Result<FfnRoles, EngError> {
        let prefix = format!("model.layers.{layer}.mlp");
        if geometry.dense_layers.contains(&layer) {
            return Ok(FfnRoles::Dense(self.expert(&prefix)?));
        }
        let experts = (0..geometry.experts)
            .map(|expert| self.expert(&format!("{prefix}.experts.{expert}")))
            .collect::<Result<_, _>>()?;
        Ok(FfnRoles::Moe {
            router: self.plain(&format!("{prefix}.gate.weight"))?,
            experts,
            shared: self.expert(&format!("{prefix}.shared_expert"))?,
            routing_weight_multiplier: factor,
        })
    }

    fn layer(
        &self,
        geometry: &Geometry,
        layer: usize,
        factor: f32,
    ) -> Result<LayerRoles, EngError> {
        let prefix = format!("model.layers.{layer}");
        Ok(LayerRoles {
            qkvo: [
                self.matrix(&format!("{prefix}.self_attn.q_proj"))?,
                self.matrix(&format!("{prefix}.self_attn.k_proj"))?,
                self.matrix(&format!("{prefix}.self_attn.v_proj"))?,
                self.matrix(&format!("{prefix}.self_attn.o_proj"))?,
            ],
            norms: [
                self.plain(&format!("{prefix}.input_layernorm.weight"))?,
                self.plain(&format!("{prefix}.post_attention_layernorm.weight"))?,
                self.plain(&format!("{prefix}.self_attn.q_norm.weight"))?,
                self.plain(&format!("{prefix}.self_attn.k_norm.weight"))?,
            ],
            sinks: self.plain(&format!("{prefix}.self_attn.softmax_offset"))?,
            ffn: self.ffn(geometry, layer, factor)?,
        })
    }
}

fn required(names: &HashMap<&str, usize>, name: &str) -> Result<usize, EngError> {
    names
        .get(name)
        .copied()
        .ok_or_else(|| EngError::Checkpoint(format!("Flint EXL3 resident role {name} is missing")))
}
