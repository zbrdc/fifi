use serde_json::Value;

use crate::{EngError, checkpoint_json::UniqueValue, kernels::QuantizationMode};

use super::{
    quantization::QuantizationMap,
    spec::{Act, AttentionKind, FamilySpec, Router},
};

/// The reference default for `full_attention_interval` (`qwen3_5.py:43`).
const DEFAULT_FULL_ATTENTION_INTERVAL: u32 = 4;

impl FamilySpec {
    /// Checks architecture before enforcing this family's quantization metadata.
    pub fn check_config(&self, config: &[u8]) -> Result<(), EngError> {
        self.check_architecture_config(config)?;
        self.check_quantization(config)
    }

    /// Checks checkpoint architecture against this family, without a weight encoding.
    ///
    /// Executor load paths call this before resolving weights. Any value that differs from
    /// this constant, or that the configuration omits, returns `EngError::Checkpoint`.
    /// The check reads `text_config` when it exists and the root object otherwise.
    /// It rejects duplicate members and a null `head_dim`.
    /// An absent `full_attention_interval` defaults to 4, as in the pinned reference, and a
    /// present `layer_types` must equal the order that interval gives.
    /// The configuration vocabulary names only `full_attention` and `linear_attention`, so a
    /// family may execute the full-attention positions as `AttentionKind::Qsa`.
    /// The `mtp` flag selects an execution mode and has no configuration field to check.
    ///
    /// TODO(S7): the check does not compare `partial_rotary_factor` or the rope dimension.
    /// The S7 lane owns that field and its correction.
    pub(crate) fn check_architecture_config(&self, config: &[u8]) -> Result<(), EngError> {
        let root = serde_json::from_slice::<UniqueValue>(config)
            .map_err(|error| EngError::Checkpoint(format!("configuration JSON: {error}")))?
            .0;
        let text = root.get("text_config").unwrap_or(&root);
        if !text.is_object() {
            return Err(EngError::Checkpoint(
                "configuration requires a JSON object".into(),
            ));
        }
        let fields = Fields {
            family: self.name,
            object: text,
        };
        super::keys::validate_family(self)?;
        self.check_model_type(&root, &fields)?;
        self.check_dimensions(&fields)?;
        self.check_switches(&fields)?;
        self.check_routing(&fields)?;
        self.check_layer_kinds(&fields)?;
        self.check_linear(&fields)?;
        Ok(())
    }

    fn check_dimensions(&self, fields: &Fields) -> Result<(), EngError> {
        let hidden = fields.expect_u32("hidden_size", self.hidden)?;
        let heads_q = fields.expect_u32("num_attention_heads", self.heads_q)?;
        fields.expect_u32("num_hidden_layers", self.layers)?;
        fields.expect_u32("num_key_value_heads", self.heads_kv)?;
        fields.expect_u32("vocab_size", self.vocab)?;
        if let Some(dense) = self.feed_forward {
            fields.expect_u32("intermediate_size", dense.intermediate)?;
            fields.expect_u32("max_position_embeddings", dense.context_limit)?;
            for name in [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "norm_topk_prob",
            ] {
                fields.reject(name)?;
            }
        } else {
            // Existing MoE families tolerate a stray intermediate_size; only the dense arm
            // reads it, and refusing it here would change what main accepts.
            fields.expect_u32("num_experts", self.experts)?;
            fields.expect_u32("num_experts_per_tok", self.top_k)?;
            fields.expect_u32("moe_intermediate_size", self.moe_inter)?;
            fields.expect_u32("shared_expert_intermediate_size", self.shared_inter)?;
        }
        if fields.object.get("head_dim").is_some() {
            fields.expect_u32("head_dim", self.head_dim)?;
        } else {
            // The reference derives an absent head dimension from the hidden and head counts.
            let derived = hidden.checked_div(heads_q);
            fields.mismatch("head_dim", derived.map(u64::from), u64::from(self.head_dim))?;
        }
        let theta = fields.rope_theta()?;
        if theta != f64::from(self.rope_theta) {
            return Err(fields.differs("rope_theta", theta, self.rope_theta));
        }
        Ok(())
    }

    /// The reference defaults differ by model type: `qwen3_5.py:51` sets `norm_topk_prob` true,
    /// and `qwen3_next.py:51` sets it false. The check therefore names the model type first.
    fn check_model_type(&self, root: &Value, fields: &Fields) -> Result<(), EngError> {
        if self.feed_forward.is_some() {
            fields.string("model_type")?;
        }
        let mut found = false;
        for level in [root, fields.object] {
            let Some(value) = level.get("model_type") else {
                continue;
            };
            found = true;
            let accepted = value
                .as_str()
                .is_some_and(|name| self.model_types.contains(&name));
            if !accepted {
                return Err(EngError::Checkpoint(format!(
                    "configuration model_type {value} is not one of {:?} for family {}",
                    self.model_types, self.name
                )));
            }
        }
        if found {
            Ok(())
        } else {
            Err(EngError::Checkpoint(
                "configuration lacks model_type".into(),
            ))
        }
    }

    fn check_switches(&self, fields: &Fields) -> Result<(), EngError> {
        // The reference gates the attention output unconditionally (qwen3_next.py:123-158),
        // and the resolver expects the gate rows, so only an explicit false is wrong.
        if !fields.optional_bool("attn_output_gate")?.unwrap_or(true) {
            return Err(fields.differs("attn_output_gate", false, true));
        }
        for name in ["tie_word_embeddings", "attention_bias"] {
            if fields.optional_bool(name)?.unwrap_or(false) {
                return Err(EngError::Checkpoint(format!(
                    "configuration {name} is true, family {} loads untied embeddings and bias-free attention",
                    self.name
                )));
            }
        }
        let eps = fields.f64("rms_norm_eps")?;
        if eps as f32 != self.rms_norm_eps {
            return Err(fields.differs("rms_norm_eps", eps, self.rms_norm_eps));
        }
        Ok(())
    }

    fn check_routing(&self, fields: &Fields) -> Result<(), EngError> {
        let act = match fields.string("hidden_act")? {
            "silu" => Act::Silu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Act::GeluApprox,
            other => {
                return Err(EngError::Checkpoint(format!(
                    "configuration hidden_act {other:?} is unsupported"
                )));
            }
        };
        if act != self.act {
            return Err(fields.differs(
                "hidden_act",
                format!("{act:?}"),
                format!("{:?}", self.act),
            ));
        }
        if self.feed_forward.is_some() {
            return Ok(());
        }
        let Router::SoftmaxTopK { renorm } = self.router else {
            return Err(EngError::Checkpoint(format!(
                "family {} router {:?} has no configuration mapping",
                self.name, self.router
            )));
        };
        // The reference defaults an absent norm_topk_prob to true.
        let found = match fields.object.get("norm_topk_prob") {
            None => true,
            Some(Value::Bool(flag)) => *flag,
            Some(_) => {
                return Err(EngError::Checkpoint(
                    "configuration norm_topk_prob must be a boolean".into(),
                ));
            }
        };
        if found != renorm {
            return Err(fields.differs("norm_topk_prob", found, renorm));
        }
        Ok(())
    }

    fn check_layer_kinds(&self, fields: &Fields) -> Result<(), EngError> {
        // The pinned reference derives the order from the interval alone and defaults the
        // interval to 4 (qwen3_5.py:43,203). A present layer_types must equal that order.
        let interval = fields
            .optional_u32("full_attention_interval")?
            .unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL);
        if interval == 0 {
            return Err(EngError::Checkpoint(
                "configuration full_attention_interval must exceed zero".into(),
            ));
        }
        let derived: Vec<AttentionKind> = (1..=fields
            .expect_u32("num_hidden_layers", self.layers)?)
            .map(|position| {
                if position % interval == 0 {
                    AttentionKind::Full
                } else {
                    AttentionKind::DeltaNet
                }
            })
            .collect();
        let declared: Vec<AttentionKind> = self
            .layers_attention
            .iter()
            .map(|kind| config_layer_kind(*kind))
            .collect();
        if derived != declared {
            return Err(EngError::Checkpoint(format!(
                "configuration full_attention_interval {interval} gives a layer order that differs from family {}",
                self.name
            )));
        }
        match fields.object.get("layer_types") {
            None => Ok(()),
            Some(Value::Array(types)) => {
                let listed = types
                    .iter()
                    .map(|kind| match kind.as_str() {
                        Some("linear_attention") => Ok(AttentionKind::DeltaNet),
                        Some("full_attention") => Ok(AttentionKind::Full),
                        _ => Err(EngError::Checkpoint(format!(
                            "configuration layer_types entry {kind} is unsupported"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if listed == derived {
                    Ok(())
                } else {
                    Err(EngError::Checkpoint(format!(
                        "configuration layer_types differ from the order that full_attention_interval {interval} gives, family {}",
                        self.name
                    )))
                }
            }
            Some(_) => Err(EngError::Checkpoint(
                "configuration layer_types must be an array".into(),
            )),
        }
    }

    fn check_linear(&self, fields: &Fields) -> Result<(), EngError> {
        let Some(linear) = self.linear else {
            return Ok(());
        };
        fields.expect_u32("linear_num_key_heads", linear.key_heads)?;
        fields.expect_u32("linear_key_head_dim", linear.key_head_dim)?;
        fields.expect_u32("linear_num_value_heads", linear.value_heads)?;
        fields.expect_u32("linear_value_head_dim", linear.value_head_dim)?;
        fields.expect_u32("linear_conv_kernel_dim", linear.conv_kernel)?;
        Ok(())
    }

    fn check_quantization(&self, config: &[u8]) -> Result<(), EngError> {
        let defaults = QuantizationMap::from_json(config)?.defaults();
        if defaults.mode != QuantizationMode::Affine
            || defaults.bits != self.quant_bits
            || defaults.group_size != self.quant_group
        {
            return Err(EngError::Checkpoint(format!(
                "configuration quantization is {:?} {} bits in groups of {}, family {} declares affine {} bits in groups of {}",
                defaults.mode,
                defaults.bits,
                defaults.group_size,
                self.name,
                self.quant_bits,
                self.quant_group
            )));
        }
        Ok(())
    }
}

/// The configuration vocabulary distinguishes only `full_attention` and
/// `linear_attention`. A family executes the full-attention positions either as
/// `AttentionKind::Full` or as the sparse `AttentionKind::Qsa`, so both declare
/// the same configuration kind.
fn config_layer_kind(kind: AttentionKind) -> AttentionKind {
    match kind {
        AttentionKind::Full | AttentionKind::Qsa => AttentionKind::Full,
        AttentionKind::DeltaNet => AttentionKind::DeltaNet,
    }
}

struct Fields<'a> {
    family: &'static str,
    object: &'a Value,
}

impl Fields<'_> {
    fn reject(&self, name: &str) -> Result<(), EngError> {
        if self.object.get(name).is_some() {
            return Err(EngError::Checkpoint(format!(
                "configuration {name} is not used by family {}",
                self.family
            )));
        }
        Ok(())
    }

    fn optional_u32(&self, name: &str) -> Result<Option<u32>, EngError> {
        let Some(value) = self.object.get(name) else {
            return Ok(None);
        };
        value
            .as_u64()
            .and_then(|number| u32::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| {
                EngError::Checkpoint(format!("configuration {name} must be a u32 integer"))
            })
    }

    fn u32(&self, name: &str) -> Result<u32, EngError> {
        self.optional_u32(name)?
            .ok_or_else(|| EngError::Checkpoint(format!("configuration lacks {name}")))
    }

    fn string(&self, name: &str) -> Result<&str, EngError> {
        self.object
            .get(name)
            .ok_or_else(|| EngError::Checkpoint(format!("configuration lacks {name}")))?
            .as_str()
            .ok_or_else(|| EngError::Checkpoint(format!("configuration {name} must be a string")))
    }

    fn optional_bool(&self, name: &str) -> Result<Option<bool>, EngError> {
        match self.object.get(name) {
            None => Ok(None),
            Some(Value::Bool(flag)) => Ok(Some(*flag)),
            Some(_) => Err(EngError::Checkpoint(format!(
                "configuration {name} must be a boolean"
            ))),
        }
    }

    fn f64(&self, name: &str) -> Result<f64, EngError> {
        self.object
            .get(name)
            .ok_or_else(|| EngError::Checkpoint(format!("configuration lacks {name}")))?
            .as_f64()
            .ok_or_else(|| EngError::Checkpoint(format!("configuration {name} must be a number")))
    }

    /// Reads `rope_parameters.rope_theta`, the only source the reference uses
    /// (`qwen3_5.py:54-61,72-82`). A missing nested value makes the reference fall back to
    /// 1e5, so the check rejects it. A top-level copy must agree with the nested value.
    fn rope_theta(&self) -> Result<f64, EngError> {
        let nested = self
            .object
            .get("rope_parameters")
            .and_then(|parameters| parameters.get("rope_theta"))
            .ok_or_else(|| {
                EngError::Checkpoint("configuration lacks rope_parameters.rope_theta".into())
            })?
            .as_f64()
            .ok_or_else(|| {
                EngError::Checkpoint("configuration rope_theta must be a number".into())
            })?;
        if let Some(top) = self.object.get("rope_theta")
            && top.as_f64() != Some(nested)
        {
            return Err(EngError::Checkpoint(format!(
                "configuration rope_theta {top} disagrees with rope_parameters.rope_theta {nested}"
            )));
        }
        Ok(nested)
    }

    fn expect_u32(&self, name: &str, declared: u32) -> Result<u32, EngError> {
        let found = self.u32(name)?;
        self.mismatch(name, Some(u64::from(found)), u64::from(declared))?;
        Ok(found)
    }

    fn mismatch(&self, name: &str, found: Option<u64>, declared: u64) -> Result<(), EngError> {
        match found {
            Some(found) if found == declared => Ok(()),
            Some(found) => Err(self.differs(name, found, declared)),
            None => Err(EngError::Checkpoint(format!(
                "configuration {name} cannot be derived for family {}",
                self.family
            ))),
        }
    }

    fn differs(
        &self,
        name: &str,
        found: impl std::fmt::Display,
        declared: impl std::fmt::Display,
    ) -> EngError {
        EngError::Checkpoint(format!(
            "configuration {name} is {found}, family {} declares {declared}",
            self.family
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::{QWEN3_6_35B_A3B, QWEN3_8_27B_DENSE};
    use serde_json::json;

    #[test]
    fn exl3_architecture_is_checked_without_mlx_quantization() {
        let mut config: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/qwen3_6_35b_a3b_config.json"
        ))
        .unwrap();
        config.as_object_mut().unwrap().remove("quantization");
        config["quantization_config"] = json!({"quant_method": "exl3", "version": "1.4.9"});
        let bytes = config.to_string();
        crate::weights::exl3::Config::from_json(bytes.as_bytes()).expect("supported EXL3 format");
        QWEN3_6_35B_A3B
            .check_architecture_config(bytes.as_bytes())
            .expect("Qwen architecture must not require MLX affine quantization");
        assert!(matches!(
            QWEN3_6_35B_A3B.check_config(bytes.as_bytes()),
            Err(EngError::Checkpoint(message))
                if message.contains("quant_method is not MLX quantization metadata")
        ));
        config["text_config"]["hidden_size"] = json!(1);
        assert!(matches!(
            QWEN3_6_35B_A3B.check_architecture_config(config.to_string().as_bytes()),
            Err(EngError::Checkpoint(message)) if message.contains("hidden_size")
        ));
    }

    // Checkpoint fields supplied by the qualification owner, not generated from FamilySpec.
    fn dense_config() -> Value {
        json!({
            "model_type": "qwen3_5",
            "quantization": {"mode": "affine", "bits": 6, "group_size": 64},
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120, "intermediate_size": 17408,
                "num_hidden_layers": 64, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "max_position_embeddings": 262144, "full_attention_interval": 4,
                "attn_output_gate": true, "tie_word_embeddings": false,
                "attention_bias": false, "hidden_act": "silu", "rms_norm_eps": 1e-6,
                "linear_num_key_heads": 16, "linear_key_head_dim": 128,
                "linear_num_value_heads": 48, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true, "mrope_section": [11, 11, 10]
                },
                "layer_types": (0..64).map(|i| if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }).collect::<Vec<_>>()
            }
        })
    }

    #[test]
    fn dense_checkpoint_text_config_passes() {
        QWEN3_8_27B_DENSE
            .check_config(dense_config().to_string().as_bytes())
            .expect("dense checkpoint dimensions");
    }

    #[test]
    fn dense_dimensions_and_hybrid_order_are_checked() {
        for field in [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "full_attention_interval",
            "linear_num_key_heads",
            "linear_key_head_dim",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ] {
            let mut config = dense_config();
            config["text_config"][field] = json!(1);
            assert!(
                QWEN3_8_27B_DENSE
                    .check_config(config.to_string().as_bytes())
                    .is_err(),
                "{field}"
            );
        }
        let mut config = dense_config();
        config["text_config"]["layer_types"][0] = json!("full_attention");
        assert!(
            QWEN3_8_27B_DENSE
                .check_config(config.to_string().as_bytes())
                .is_err()
        );
    }

    #[test]
    fn dense_config_requires_model_type_and_intermediate_size() {
        for field in ["model_type", "intermediate_size"] {
            let mut config = dense_config();
            config["text_config"].as_object_mut().unwrap().remove(field);
            assert!(
                QWEN3_8_27B_DENSE
                    .check_config(config.to_string().as_bytes())
                    .is_err()
            );
        }
        let mut config = dense_config();
        config["model_type"] = json!("qwen3_5_moe");
        assert!(
            QWEN3_8_27B_DENSE
                .check_config(config.to_string().as_bytes())
                .is_err()
        );
    }

    #[test]
    fn dense_config_rejects_moe_fields_and_moe_config_tolerates_intermediate_size() {
        for field in [
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "norm_topk_prob",
        ] {
            let mut config = dense_config();
            config["text_config"][field] = json!(0);
            assert!(
                QWEN3_8_27B_DENSE
                    .check_config(config.to_string().as_bytes())
                    .is_err()
            );
        }
        let mut config: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/qwen3_6_35b_a3b_config.json"
        ))
        .unwrap();
        QWEN3_6_35B_A3B
            .check_config(config.to_string().as_bytes())
            .unwrap();
        // Main tolerates a stray intermediate_size on a MoE family: only the dense arm reads
        // it, and rejecting it here would change what main accepts.
        config["text_config"]["intermediate_size"] = json!(17408);
        QWEN3_6_35B_A3B
            .check_config(config.to_string().as_bytes())
            .unwrap();
    }
}
