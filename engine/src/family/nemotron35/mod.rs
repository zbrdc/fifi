mod admission;
pub(crate) mod cpu_experts;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) mod cuda_projection;
mod diagnostics;
pub mod exec;
#[allow(
    dead_code,
    reason = "Nemotron executor wiring is scoped separately from this adapter packet"
)]
pub(crate) mod executor;
pub(crate) mod factory;
pub mod forward;
mod keys;
pub mod loader;
pub(crate) mod projection;
pub mod runner;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) mod transport_accounting;
pub use keys::{NemotronKeyClass, NemotronPlannedKey, NemotronTensorPart};

use crate::{EngError, checkpoint_json::UniqueValue};
use serde_json::Value;

/// The pinned config declares no RoPE parameters; the reference attention does not rotate Q or K.
pub const NEMOTRON_ROTATION_PROVENANCE: &str = "transformers 5.10.0.dev0, models/nemotron_h/modeling_nemotron_h.py, NemotronHAttention.forward (lines 851-893); SHA-256 859ff78fae71fa3f8857e4ea0620c43aebea624e310e8422c124c507131398d2. No rotary application; apply_rotary_pos_emb has no callers.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronRotation {
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronBlockKind {
    Mamba,
    Moe,
    Attention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronActivation {
    Silu,
    Relu2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NemotronGeometry {
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u32,
    pub layer_norm_epsilon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronMambaGeometry {
    pub groups: u32,
    pub state_size: u32,
    pub conv_kernel: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub expand: u32,
    pub inner_size: u32,
    pub chunk_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NemotronMoeGeometry {
    pub intermediate_size: u32,
    pub moe_intermediate_size: u32,
    pub latent_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub routed_experts: u32,
    pub shared_experts: u32,
    pub experts_per_token: u32,
    pub n_group: u32,
    pub n_groups: u32,
    pub topk_group: u32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronActivations {
    pub mamba: NemotronActivation,
    pub mlp: NemotronActivation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NemotronTextSpec {
    geometry: NemotronGeometry,
    mamba: NemotronMambaGeometry,
    moe: NemotronMoeGeometry,
    activations: NemotronActivations,
    block_order: Vec<NemotronBlockKind>,
    rotation: NemotronRotation,
    excluded_members: Vec<String>,
    tie_word_embeddings: bool,
    rescale_prenorm_residual: bool,
}

impl NemotronTextSpec {
    pub fn from_config(config: &[u8]) -> Result<Self, EngError> {
        let root = serde_json::from_slice::<UniqueValue>(config)
            .map_err(|error| config_error("JSON", error.to_string()))?
            .0;
        if !root.is_object() {
            return Err(config_error("root", "must be a JSON object"));
        }
        expect_string(&root, "model_type", "nemotron_h_omni")?;
        expect_architecture(&root, "architectures", "NemotronH_Omni_Reasoning_V3")?;
        let llm = required_object(&root, "llm_config")?;
        expect_string(llm, "model_type", "nemotron_h")?;
        expect_architecture(llm, "architectures", "NemotronHForCausalLM")?;

        let geometry = NemotronGeometry {
            hidden_size: required_u32(llm, "hidden_size")?,
            vocab_size: required_u32(llm, "vocab_size")?,
            num_attention_heads: required_u32(llm, "num_attention_heads")?,
            num_key_value_heads: required_u32(llm, "num_key_value_heads")?,
            head_dim: required_u32(llm, "head_dim")?,
            max_position_embeddings: required_u32(llm, "max_position_embeddings")?,
            layer_norm_epsilon: required_f64(llm, "layer_norm_epsilon")?,
        };
        validate_geometry(&geometry)?;
        let mamba_heads = required_u32(llm, "mamba_num_heads")?;
        let mamba_head_dim = required_u32(llm, "mamba_head_dim")?;
        let mamba = NemotronMambaGeometry {
            groups: required_u32(llm, "n_groups")?,
            state_size: required_u32(llm, "ssm_state_size")?,
            conv_kernel: required_u32(llm, "conv_kernel")?,
            heads: mamba_heads,
            head_dim: mamba_head_dim,
            expand: required_u32(llm, "expand")?,
            inner_size: checked_product(mamba_heads, mamba_head_dim, "mamba head extent")?,
            chunk_size: required_u32(llm, "chunk_size")?,
        };
        validate_mamba(&geometry, &mamba)?;
        let moe = NemotronMoeGeometry {
            intermediate_size: required_u32(llm, "intermediate_size")?,
            moe_intermediate_size: required_u32(llm, "moe_intermediate_size")?,
            latent_size: required_u32(llm, "moe_latent_size")?,
            shared_expert_intermediate_size: required_u32(
                llm,
                "moe_shared_expert_intermediate_size",
            )?,
            routed_experts: required_u32(llm, "n_routed_experts")?,
            shared_experts: required_u32(llm, "n_shared_experts")?,
            experts_per_token: required_u32(llm, "num_experts_per_tok")?,
            n_group: required_u32(llm, "n_group")?,
            n_groups: required_u32(llm, "n_groups")?,
            topk_group: required_u32(llm, "topk_group")?,
            norm_topk_prob: required_bool(llm, "norm_topk_prob")?,
            routed_scaling_factor: required_f64(llm, "routed_scaling_factor")?,
        };
        validate_moe(&moe)?;
        let activations = NemotronActivations {
            mamba: parse_activation(llm, "mamba_hidden_act")?,
            mlp: parse_activation(llm, "mlp_hidden_act")?,
        };
        if activations.mamba != NemotronActivation::Silu {
            return Err(config_error("mamba_hidden_act", "must be silu"));
        }
        if activations.mlp != NemotronActivation::Relu2 {
            return Err(config_error("mlp_hidden_act", "must be relu2"));
        }
        let block_order = parse_blocks(llm)?;
        let tie_word_embeddings = required_bool(llm, "tie_word_embeddings")?;
        let rescale_prenorm_residual = required_bool(llm, "rescale_prenorm_residual")?;
        if tie_word_embeddings {
            return Err(config_error("tie_word_embeddings", "must be false"));
        }
        if !rescale_prenorm_residual {
            return Err(config_error("rescale_prenorm_residual", "must be true"));
        }

        let mut excluded_members: Vec<String> =
            ["num_nextn_predict_layers", "mtp_layers_block_type"]
                .into_iter()
                .filter(|member| llm.get(member).is_some())
                .map(|member| format!("llm_config.{member}"))
                .collect();
        excluded_members.extend(
            root.as_object()
                .into_iter()
                .flat_map(|root| root.keys())
                .filter(|member| {
                    is_vision_member(member)
                        || member.starts_with("sound_")
                        || member.starts_with("audio_")
                })
                .cloned(),
        );
        Ok(Self {
            geometry,
            mamba,
            moe,
            activations,
            block_order,
            rotation: NemotronRotation::None,
            excluded_members,
            tie_word_embeddings,
            rescale_prenorm_residual,
        })
    }

    pub fn rotation(&self) -> NemotronRotation {
        self.rotation
    }

    /// Validate text architecture readiness, not backend support or checkpoint tensor payloads.
    pub fn validate_execution_readiness(&self) -> Result<(), EngError> {
        match self.rotation {
            NemotronRotation::None => Ok(()),
        }
    }

    pub fn excluded_members(&self) -> &[String] {
        &self.excluded_members
    }

    /// Check execution scope only; this does not resolve a configuration member.
    pub fn validate_execution_member(&self, member: &str) -> Result<(), EngError> {
        for excluded in &self.excluded_members {
            if member == excluded
                || member
                    .strip_prefix(excluded)
                    .is_some_and(|tail| tail.starts_with('.'))
            {
                return Err(config_error(excluded, "out of scope for text execution"));
            }
        }
        Ok(())
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    pub fn rescale_prenorm_residual(&self) -> bool {
        self.rescale_prenorm_residual
    }

    pub fn geometry(&self) -> NemotronGeometry {
        self.geometry
    }

    pub fn mamba_geometry(&self) -> NemotronMambaGeometry {
        self.mamba
    }

    pub fn moe_geometry(&self) -> NemotronMoeGeometry {
        self.moe
    }

    pub fn activations(&self) -> NemotronActivations {
        self.activations
    }

    pub fn block_order(&self) -> &[NemotronBlockKind] {
        &self.block_order
    }
}

fn required_object<'a>(object: &'a Value, field: &str) -> Result<&'a Value, EngError> {
    object
        .get(field)
        .filter(|value| value.is_object())
        .ok_or_else(|| config_error(field, "must be a JSON object"))
}

fn validate_moe(moe: &NemotronMoeGeometry) -> Result<(), EngError> {
    if moe.n_groups == 0 {
        return Err(config_error("n_groups", "must exceed zero"));
    }
    if !moe.routed_experts.is_multiple_of(moe.n_groups) {
        return Err(config_error(
            "n_groups",
            "must divide n_routed_experts evenly",
        ));
    }
    if moe.experts_per_token > moe.routed_experts || moe.topk_group > moe.n_groups {
        return Err(config_error(
            "num_experts_per_tok/topk_group",
            "routing selection exceeds the declared experts or groups",
        ));
    }
    let routed_extent = checked_product(
        moe.routed_experts,
        moe.moe_intermediate_size,
        "routed expert extent",
    )?;
    let latent_extent =
        checked_product(moe.routed_experts, moe.latent_size, "latent expert extent")?;
    let shared_extent = checked_product(
        moe.shared_experts,
        moe.shared_expert_intermediate_size,
        "shared expert extent",
    )?;
    if routed_extent == 0 || latent_extent == 0 || shared_extent == 0 {
        return Err(config_error("MoE extent", "must exceed zero"));
    }
    for (field, found, expected) in [
        ("intermediate_size", moe.intermediate_size, 2688),
        ("moe_intermediate_size", moe.moe_intermediate_size, 2688),
        ("moe_latent_size", moe.latent_size, 1024),
        (
            "moe_shared_expert_intermediate_size",
            moe.shared_expert_intermediate_size,
            5376,
        ),
        ("n_routed_experts", moe.routed_experts, 512),
        ("n_shared_experts", moe.shared_experts, 1),
        ("num_experts_per_tok", moe.experts_per_token, 22),
        ("n_group", moe.n_group, 1),
        ("n_groups", moe.n_groups, 8),
        ("topk_group", moe.topk_group, 1),
    ] {
        expect_u32_value(field, found, expected)?;
    }
    if !moe.norm_topk_prob {
        return Err(config_error("norm_topk_prob", "must be true"));
    }
    if !moe.routed_scaling_factor.is_finite() || moe.routed_scaling_factor != 5.0 {
        return Err(config_error(
            "routed_scaling_factor",
            format!("expected 5, got {}", moe.routed_scaling_factor),
        ));
    }
    Ok(())
}

fn validate_mamba(
    geometry: &NemotronGeometry,
    mamba: &NemotronMambaGeometry,
) -> Result<(), EngError> {
    let expanded_width = checked_product(
        geometry.hidden_size,
        mamba.expand,
        "Mamba expanded hidden extent",
    )?;
    let head_width = checked_product(mamba.heads, mamba.head_dim, "Mamba head extent")?;
    let conv_extent = checked_product(
        expanded_width,
        mamba.conv_kernel,
        "Mamba convolution extent",
    )?;
    let state_extent = checked_product(mamba.state_size, mamba.head_dim, "Mamba state extent")?;
    for (field, found, expected) in [
        ("n_groups", mamba.groups, 8),
        ("ssm_state_size", mamba.state_size, 128),
        ("conv_kernel", mamba.conv_kernel, 4),
        ("mamba_num_heads", mamba.heads, 128),
        ("mamba_head_dim", mamba.head_dim, 64),
        ("expand", mamba.expand, 2),
        ("chunk_size", mamba.chunk_size, 128),
    ] {
        expect_u32_value(field, found, expected)?;
    }
    if expanded_width != head_width || mamba.inner_size != expanded_width {
        return Err(config_error(
            "mamba_num_heads",
            format!("Mamba width {head_width} differs from expanded hidden width {expanded_width}"),
        ));
    }
    if conv_extent == 0 || state_extent == 0 {
        return Err(config_error("Mamba state", "extent must exceed zero"));
    }
    Ok(())
}

fn validate_geometry(geometry: &NemotronGeometry) -> Result<(), EngError> {
    checked_product(
        geometry.num_attention_heads,
        geometry.head_dim,
        "attention projection extent",
    )?;
    for (field, found, expected) in [
        ("hidden_size", geometry.hidden_size, 4096),
        ("vocab_size", geometry.vocab_size, 131_072),
        ("num_attention_heads", geometry.num_attention_heads, 32),
        ("num_key_value_heads", geometry.num_key_value_heads, 2),
        ("head_dim", geometry.head_dim, 128),
        (
            "max_position_embeddings",
            geometry.max_position_embeddings,
            1_048_576,
        ),
    ] {
        expect_u32_value(field, found, expected)?;
    }
    if !geometry.layer_norm_epsilon.is_finite() || geometry.layer_norm_epsilon != 1e-5 {
        return Err(config_error(
            "layer_norm_epsilon",
            format!("expected 0.00001, got {}", geometry.layer_norm_epsilon),
        ));
    }
    Ok(())
}

fn expect_u32_value(field: &str, found: u32, expected: u32) -> Result<(), EngError> {
    if found == expected {
        Ok(())
    } else {
        Err(config_error(
            field,
            format!("expected {expected}, got {found}"),
        ))
    }
}

fn required_u32(object: &Value, field: &str) -> Result<u32, EngError> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| config_error(field, "must be a u32 integer"))
}

fn required_bool(object: &Value, field: &str) -> Result<bool, EngError> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| config_error(field, "must be a boolean"))
}

fn required_f64(object: &Value, field: &str) -> Result<f64, EngError> {
    object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| config_error(field, "must be a number"))
}

fn required_str<'a>(object: &'a Value, field: &str) -> Result<&'a str, EngError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| config_error(field, "must be a string"))
}

fn expect_string(object: &Value, field: &str, expected: &str) -> Result<(), EngError> {
    let found = required_str(object, field)?;
    if found == expected {
        Ok(())
    } else {
        Err(config_error(
            field,
            format!("expected {expected:?}, got {found:?}"),
        ))
    }
}

fn expect_architecture(object: &Value, field: &str, expected: &str) -> Result<(), EngError> {
    let architectures = object
        .get(field)
        .and_then(Value::as_array)
        .filter(|values| values.len() == 1)
        .ok_or_else(|| config_error(field, "must contain one architecture name"))?;
    let found = architectures[0]
        .as_str()
        .ok_or_else(|| config_error(field, "architecture name must be a string"))?;
    if found == expected {
        Ok(())
    } else {
        Err(config_error(
            field,
            format!("expected {expected:?}, got {found:?}"),
        ))
    }
}

fn parse_activation(object: &Value, field: &str) -> Result<NemotronActivation, EngError> {
    match required_str(object, field)? {
        "silu" => Ok(NemotronActivation::Silu),
        "relu2" => Ok(NemotronActivation::Relu2),
        value => Err(config_error(
            field,
            format!("unsupported activation {value:?}"),
        )),
    }
}

fn parse_blocks(object: &Value) -> Result<Vec<NemotronBlockKind>, EngError> {
    let blocks = object
        .get("layers_block_type")
        .and_then(Value::as_array)
        .ok_or_else(|| config_error("layers_block_type", "must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let kind = value.as_str().ok_or_else(|| {
                config_error(format!("layers_block_type[{index}]"), "must be a string")
            })?;
            match kind {
                "mamba" => Ok(NemotronBlockKind::Mamba),
                "moe" => Ok(NemotronBlockKind::Moe),
                "attention" => Ok(NemotronBlockKind::Attention),
                _ => Err(config_error(
                    format!("layers_block_type[{index}]"),
                    format!("unsupported block kind {kind:?}"),
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_block_order(&blocks)?;
    Ok(blocks)
}

fn validate_block_order(blocks: &[NemotronBlockKind]) -> Result<(), EngError> {
    if blocks.len() != 88 {
        return Err(config_error(
            "layers_block_type",
            format!("must contain 88 blocks, got {}", blocks.len()),
        ));
    }
    const ATTENTION_POSITIONS: [usize; 8] = [7, 16, 25, 36, 47, 58, 69, 78];
    let mut non_attention_index: usize = 0;
    for (index, kind) in blocks.iter().enumerate() {
        let expected = if ATTENTION_POSITIONS.contains(&index) {
            NemotronBlockKind::Attention
        } else {
            let expected = if non_attention_index.is_multiple_of(2) {
                NemotronBlockKind::Mamba
            } else {
                NemotronBlockKind::Moe
            };
            non_attention_index += 1;
            expected
        };
        if *kind != expected {
            return Err(config_error(
                format!("layers_block_type[{index}]"),
                format!("expected {expected:?}, got {kind:?}"),
            ));
        }
    }
    Ok(())
}

fn checked_product(a: u32, b: u32, field: &str) -> Result<u32, EngError> {
    a.checked_mul(b)
        .ok_or_else(|| config_error(field, "extent overflows u32"))
}

fn config_error(field: impl std::fmt::Display, detail: impl std::fmt::Display) -> EngError {
    EngError::Checkpoint(format!("Nemotron 3.5 configuration {field}: {detail}"))
}

fn is_vision_member(member: &str) -> bool {
    ["vision_", "vit_", "image_", "img_", "video_"]
        .iter()
        .any(|prefix| member.starts_with(prefix))
        || matches!(
            member,
            "downsample_ratio"
                | "force_image_size"
                | "norm_mean"
                | "norm_std"
                | "patch_size"
                | "projector_hidden_size"
                | "ps_version"
                | "use_thumbnail"
        )
}
