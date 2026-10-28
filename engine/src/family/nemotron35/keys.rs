use super::{NemotronBlockKind, NemotronTextSpec};
use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NemotronKeyClass {
    Embedding,
    FinalNorm,
    LmHead,
    BlockNorm,
    MambaStateDecay,
    MambaSkip,
    MambaTimeBias,
    MambaConvolution,
    MambaInputProjection,
    MambaNorm,
    MambaOutputProjection,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    MoeRouter,
    MoeRouterCorrection,
    MoeLatentInput,
    MoeLatentOutput,
    MoeExpertUp,
    MoeExpertDown,
    MoeSharedUp,
    MoeSharedDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronTensorPart {
    Weight,
    Bias,
    InputScale,
    WeightScale,
    WeightScale2,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronPlannedKey {
    pub block: Option<usize>,
    pub class: NemotronKeyClass,
    pub expert: Option<u32>,
    pub part: NemotronTensorPart,
}

impl NemotronBlockKind {
    /// Key classes observed in the pinned index, independent of quantization tensor parts.
    pub fn expected_key_classes(self) -> &'static [NemotronKeyClass] {
        use NemotronKeyClass::*;
        match self {
            Self::Mamba => &[
                BlockNorm,
                MambaStateDecay,
                MambaSkip,
                MambaTimeBias,
                MambaConvolution,
                MambaInputProjection,
                MambaNorm,
                MambaOutputProjection,
            ],
            Self::Attention => &[
                BlockNorm,
                AttentionQuery,
                AttentionKey,
                AttentionValue,
                AttentionOutput,
            ],
            Self::Moe => &[
                BlockNorm,
                MoeRouter,
                MoeRouterCorrection,
                MoeLatentInput,
                MoeLatentOutput,
                MoeExpertUp,
                MoeExpertDown,
                MoeSharedUp,
                MoeSharedDown,
            ],
        }
    }
}

impl NemotronTextSpec {
    /// Classify one index name; this does not validate payloads or inventory completeness.
    pub fn key_plan(&self, name: &str) -> Result<NemotronPlannedKey, EngError> {
        use NemotronKeyClass::*;
        use NemotronTensorPart::*;
        for (prefix, member) in [
            ("language_model.mtp.", "llm_config.num_nextn_predict_layers"),
            ("vision_model.", "vision_config"),
            ("vision_projector.", "vision_config"),
            ("mlp1.", "projector_hidden_size"),
            ("sound_model.", "sound_config"),
            ("audio_model.", "audio_config"),
        ] {
            if name.starts_with(prefix) {
                return Err(key_error(
                    name,
                    &format!("{member} is out of scope for text execution"),
                ));
            }
        }
        let global = match name {
            "language_model.backbone.embeddings.weight" => Some((Embedding, Weight)),
            "language_model.backbone.norm_f.weight" => Some((FinalNorm, Weight)),
            _ => name
                .strip_prefix("language_model.lm_head.")
                .and_then(|part| projection_part(part, false))
                .map(|part| (LmHead, part)),
        };
        if let Some((class, part)) = global {
            return Ok(NemotronPlannedKey {
                block: None,
                class,
                expert: None,
                part,
            });
        }
        let tail = name
            .strip_prefix("language_model.backbone.layers.")
            .ok_or_else(|| key_error(name, "unknown text tensor namespace"))?;
        let (index, tail) = tail
            .split_once('.')
            .ok_or_else(|| key_error(name, "missing block component"))?;
        let index_value: usize = index
            .parse()
            .map_err(|_| key_error(name, "invalid block index"))?;
        if index_value.to_string() != index {
            return Err(key_error(name, "noncanonical block index"));
        }
        let index = index_value;
        let kind = self
            .block_order()
            .get(index)
            .ok_or_else(|| key_error(name, "block index out of range"))?;
        let pieces: Vec<_> = tail.split('.').collect();
        let mut expert = None;
        let (class, part) = match pieces.as_slice() {
            ["norm", "weight"] => (BlockNorm, Weight),
            ["mixer", "A_log"] => (MambaStateDecay, Value),
            ["mixer", "D"] => (MambaSkip, Value),
            ["mixer", "dt_bias"] => (MambaTimeBias, Value),
            ["mixer", "conv1d", "weight"] => (MambaConvolution, Weight),
            ["mixer", "conv1d", "bias"] => (MambaConvolution, Bias),
            ["mixer", "norm", "weight"] => (MambaNorm, Weight),
            ["mixer", "gate", "weight"] => (MoeRouter, Weight),
            ["mixer", "gate", "e_score_correction_bias"] => (MoeRouterCorrection, Value),
            ["mixer", "fc1_latent_proj", "weight"] => (MoeLatentInput, Weight),
            ["mixer", "fc2_latent_proj", "weight"] => (MoeLatentOutput, Weight),
            ["mixer", projection, part] => {
                let class = match *projection {
                    "in_proj" => MambaInputProjection,
                    "out_proj" => MambaOutputProjection,
                    "q_proj" => AttentionQuery,
                    "k_proj" => AttentionKey,
                    "v_proj" => AttentionValue,
                    "o_proj" => AttentionOutput,
                    _ => return Err(key_error(name, "unknown projection")),
                };
                (
                    class,
                    projection_part(part, false)
                        .ok_or_else(|| key_error(name, "unknown projection tensor part"))?,
                )
            }
            ["mixer", "experts", number, projection, part] => {
                let number_value: u32 = number
                    .parse()
                    .map_err(|_| key_error(name, "invalid expert index"))?;
                if number_value.to_string() != *number {
                    return Err(key_error(name, "noncanonical expert index"));
                }
                let number = number_value;
                if number >= self.moe_geometry().routed_experts {
                    return Err(key_error(name, "expert index out of range"));
                }
                expert = Some(number);
                let class = match *projection {
                    "up_proj" => MoeExpertUp,
                    "down_proj" => MoeExpertDown,
                    _ => return Err(key_error(name, "unknown expert projection")),
                };
                (
                    class,
                    projection_part(part, true)
                        .ok_or_else(|| key_error(name, "unknown expert tensor part"))?,
                )
            }
            ["mixer", "shared_experts", projection, part] => {
                let class = match *projection {
                    "up_proj" => MoeSharedUp,
                    "down_proj" => MoeSharedDown,
                    _ => return Err(key_error(name, "unknown shared expert projection")),
                };
                (
                    class,
                    projection_part(part, true)
                        .ok_or_else(|| key_error(name, "unknown shared expert tensor part"))?,
                )
            }
            _ => return Err(key_error(name, "unknown block tensor")),
        };
        if !kind.expected_key_classes().contains(&class) {
            return Err(key_error(name, "tensor class does not match block type"));
        }
        Ok(NemotronPlannedKey {
            block: Some(index),
            class,
            expert,
            part,
        })
    }
}

fn projection_part(part: &str, second_scale: bool) -> Option<NemotronTensorPart> {
    use NemotronTensorPart::*;
    match part {
        "weight" => Some(Weight),
        "input_scale" => Some(InputScale),
        "weight_scale" => Some(WeightScale),
        "weight_scale_2" if second_scale => Some(WeightScale2),
        _ => None,
    }
}

fn key_error(name: &str, detail: &str) -> EngError {
    EngError::Checkpoint(format!("Nemotron 3.5 text key {name}: {detail}"))
}
