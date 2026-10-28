//! Source adaptation for one full-attention EXL3 mixer, not a whole-model loader.
//!
//! Stock `architecture/qwen3_5.py:324-328,355-392` defines these source names and
//! zero-centered norms. Stock `exllamav3_ext/norm.cu:229-244` converts weights to
//! float4 BEFORE adding constant_bias. Supply that effective F32 weight once:
//! our add_norm_pack and attn_prep kernels multiply their supplied weights directly.
//! Query/gate rows remain in the source's per-head [query | gate] order.

use crate::family::project::cuda::WeightLayout;
use crate::family::qwen36::exl3_admission::{effective_norm_bytes, norm_layout};

use super::{
    layer_table::LayerTable,
    mixer_geometry::{MixerGeometry, check_dense},
};
use crate::{
    EngError,
    family::{
        AttentionKind,
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
    },
    weights::{
        GgufType, WeightSource,
        exl3::{Checkpoint, Matrix},
    },
};

/// Loads only the four projections and four norms of a full-attention mixer.
/// `table` owns logical names; checkpoint headers and `Matrix` own stored metadata.
/// Each matrix is checked against geometry before its first device allocation.
pub(super) fn load_attention(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    table: &LayerTable,
    layer: usize,
    geometry: &MixerGeometry,
    reserve: Reserve,
) -> Result<CudaWeights, EngError> {
    // Refuse DeltaNet (including its incompatible head mapping) before any staging.
    if table.slot(layer)?.kind != AttentionKind::Full {
        return Err(EngError::Checkpoint(format!(
            "EXL3 attention source: layer {layer} is not full attention"
        )));
    }
    projector.context().ensure_usable()?;
    check_source_geometry(&checkpoint.model_config, layer, geometry)?;
    let dense = geometry.dense_specs(WeightLayout::Exl3, AttentionKind::Full)?;
    let raw = geometry.raw_specs(AttentionKind::Full)?;
    let capacity = dense
        .len()
        .checked_add(raw.len())
        .ok_or_else(|| EngError::MemoryPlan("EXL3 attention weight count overflow".into()))?;
    let mut weights = Vec::new();
    weights.try_reserve_exact(capacity).map_err(|error| {
        EngError::MemoryPlan(format!("EXL3 attention weight allocation: {error}"))
    })?;
    for spec in dense {
        let name = table.name(layer, spec.role)?.to_owned();
        let prefix = format!(
            "model.language_model.layers.{layer}.{}",
            projection_suffix(spec.role)?
        );
        // Match the existing single-matrix upload seam's selected component set.
        // Include unsupported old representations so Matrix::load can refuse them.
        let names =
            crate::weights::exl3::COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
        let keys = names.each_ref().map(String::as_str);
        let selected = checkpoint.source.read_selected(&keys, 1 << 30)?;
        let matrix = Matrix::load(&selected, &prefix, &checkpoint.config)?;
        let [k, n] = matrix.shape();
        check_dense(&prefix, (n, k, None), &spec)?;
        let weight = projector.upload_exl3(&matrix, reserve)?;
        weights.push(CudaWeight::Exl3 { name, weight });
    }
    for spec in raw {
        let name = table.name(layer, spec.role)?.to_owned();
        let source_name = format!(
            "model.language_model.layers.{layer}.{}",
            norm_suffix(spec.role)?
        );
        let (dtype, shape) = checkpoint.source.stored(&source_name).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "EXL3 attention source: missing tensor {source_name}"
            ))
        })?;
        // Validate the exact source shape and type before reading even a tiny norm.
        let output_len = norm_layout(dtype, &shape, spec.values)?;
        let budget = u64::try_from(output_len)
            .map_err(|_| EngError::MemoryPlan("EXL3 norm read budget overflow".into()))?;
        let selected = checkpoint.source.read_selected(&[&source_name], budget)?;
        let tensor = selected.tensor(&source_name).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "EXL3 attention source: missing tensor {source_name}"
            ))
        })?;
        let bytes = effective_norm_bytes(&tensor, spec.values)?;
        projector.admit(budget, reserve)?;
        let buffer = projector.context().upload(&bytes)?;
        weights.push(CudaWeight::Raw {
            name,
            ty: GgufType::F32,
            buffer,
        });
    }
    CudaWeights::from_exl3(weights)
}

/// Refuse interpreting source attention with the caller's different geometry or epsilon.
fn check_source_geometry(bytes: &[u8], layer: usize, g: &MixerGeometry) -> Result<(), EngError> {
    if bytes.len() as u64 > crate::weights::exl3::CONFIG_LIMIT {
        return Err(EngError::MemoryPlan(
            "EXL3 model config exceeds 1 MiB".into(),
        ));
    }
    let crate::checkpoint_json::UniqueValue(root) = serde_json::from_slice(bytes)
        .map_err(|error| EngError::Checkpoint(format!("EXL3 model config: {error}")))?;
    let text = root
        .get("text_config")
        .ok_or_else(|| EngError::Checkpoint("EXL3 attention source: missing text_config".into()))?;
    let mismatch = || {
        EngError::Checkpoint(
            "EXL3 attention source: config does not match mixer geometry or full-attention layer"
                .into(),
        )
    };
    for (key, expected) in [
        ("hidden_size", g.hidden),
        ("head_dim", g.attn.head_dim),
        ("num_attention_heads", g.attn.heads_q),
        ("num_key_value_heads", g.attn.heads_kv),
    ] {
        if text
            .get(key)
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            != Some(expected)
        {
            return Err(mismatch());
        }
    }
    let epsilon = text
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .ok_or_else(mismatch)?;
    if !epsilon.is_finite()
        || epsilon <= 0.0
        || epsilon as f32 != g.eps
        || text.get("attn_output_gate").and_then(|v| v.as_bool()) != Some(true)
        || text
            .get("layer_types")
            .and_then(|v| v.as_array())
            .and_then(|v| v.get(layer))
            .and_then(|v| v.as_str())
            != Some("full_attention")
    {
        return Err(mismatch());
    }
    let rope = text.get("rope_parameters").ok_or_else(mismatch)?;
    let theta = rope
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .ok_or_else(mismatch)?;
    let fraction = rope
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .ok_or_else(mismatch)?;
    let rotary = fraction * g.attn.head_dim as f64;
    if !theta.is_finite()
        || theta <= 0.0
        || theta as f32 != g.attn.rope_base
        || !fraction.is_finite()
        || fraction <= 0.0
        || fraction > 1.0
        || rotary != g.attn.n_rot as f64
        || rope.get("rope_type").and_then(|v| v.as_str()) != Some("default")
    {
        return Err(mismatch());
    }
    Ok(())
}

fn projection_suffix(role: Role) -> Result<&'static str, EngError> {
    match role {
        Role::AttnQ => Ok("self_attn.q_proj"),
        Role::AttnK => Ok("self_attn.k_proj"),
        Role::AttnV => Ok("self_attn.v_proj"),
        Role::AttnOutput => Ok("self_attn.o_proj"),
        _ => Err(EngError::Checkpoint(format!(
            "EXL3 attention source: unsupported projection role {role:?}"
        ))),
    }
}

fn norm_suffix(role: Role) -> Result<&'static str, EngError> {
    match role {
        Role::AttnNorm => Ok("input_layernorm.weight"),
        Role::PostAttentionNorm => Ok("post_attention_layernorm.weight"),
        Role::AttnQNorm => Ok("self_attn.q_norm.weight"),
        Role::AttnKNorm => Ok("self_attn.k_norm.weight"),
        _ => Err(EngError::Checkpoint(format!(
            "EXL3 attention source: unsupported norm role {role:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_config() -> (MixerGeometry, serde_json::Value) {
        let geometry = MixerGeometry::new(&super::super::common_tests::dims()).unwrap();
        let a = geometry.attn;
        let config = serde_json::json!({"text_config": {
            "hidden_size": geometry.hidden, "head_dim": a.head_dim,
            "num_attention_heads": a.heads_q, "num_key_value_heads": a.heads_kv,
            "rms_norm_eps": geometry.eps, "attn_output_gate": true,
            "layer_types": ["full_attention"],
            "rope_parameters": {"rope_theta": a.rope_base, "rope_type": "default",
                "partial_rotary_factor": a.n_rot as f64 / a.head_dim as f64}
        }});
        (geometry, config)
    }

    #[test]
    fn source_config_must_match_geometry_epsilon_and_layer_kind() {
        let (geometry, config) = source_config();
        let bytes = serde_json::to_vec(&config).unwrap();
        check_source_geometry(&bytes, 0, &geometry).unwrap();
        assert!(matches!(
            check_source_geometry(&bytes, 1, &geometry),
            Err(EngError::Checkpoint(_))
        ));
        for (key, wrong) in [
            ("hidden_size", serde_json::json!(0)),
            ("num_key_value_heads", serde_json::json!(0)),
            ("head_dim", serde_json::json!(128)),
            ("num_attention_heads", serde_json::json!(32)),
            ("rms_norm_eps", serde_json::json!(0.001)),
            ("attn_output_gate", serde_json::json!(false)),
            ("layer_types", serde_json::json!(["linear_attention"])),
        ] {
            let mut bad = config.clone();
            bad["text_config"][key] = wrong;
            assert!(matches!(
                check_source_geometry(&serde_json::to_vec(&bad).unwrap(), 0, &geometry),
                Err(EngError::Checkpoint(_))
            ));
        }
    }

    #[test]
    fn source_config_refuses_oversize_duplicates_and_missing_fields() {
        let (geometry, config) = source_config();
        let mut oversized = serde_json::to_vec(&config).unwrap();
        oversized.resize(1024 * 1024 + 1, b' ');
        assert!(matches!(
            check_source_geometry(&oversized, 0, &geometry),
            Err(EngError::MemoryPlan(_))
        ));
        let text = serde_json::to_string(&config["text_config"]).unwrap();
        let duplicate = format!(r#"{{"text_config":{text},"text_config":{text}}}"#);
        for bytes in [b"{".as_slice(), b"{}".as_slice(), duplicate.as_bytes()] {
            assert!(matches!(
                check_source_geometry(bytes, 0, &geometry),
                Err(EngError::Checkpoint(_))
            ));
        }
        for key in [
            "hidden_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "attn_output_gate",
            "layer_types",
            "rope_parameters",
        ] {
            let mut bad = config.clone();
            bad["text_config"].as_object_mut().unwrap().remove(key);
            assert!(
                matches!(
                    check_source_geometry(&serde_json::to_vec(&bad).unwrap(), 0, &geometry),
                    Err(EngError::Checkpoint(_))
                ),
                "missing {key}"
            );
        }
    }

    #[test]
    fn source_config_refuses_missing_or_incompatible_rotary_parameters() {
        let (geometry, config) = source_config();
        let fraction = config["text_config"]["rope_parameters"]["partial_rotary_factor"]
            .as_f64()
            .unwrap();
        for (key, wrong) in [
            ("rope_theta", serde_json::json!(0)),
            (
                "rope_theta",
                serde_json::json!(geometry.attn.rope_base as f64 * 2.0),
            ),
            ("partial_rotary_factor", serde_json::json!(0)),
            ("partial_rotary_factor", serde_json::json!(2)),
            ("partial_rotary_factor", serde_json::json!(fraction / 2.0)),
            ("rope_type", serde_json::json!("dynamic")),
        ] {
            let mut bad = config.clone();
            bad["text_config"]["rope_parameters"][key] = wrong;
            assert!(
                matches!(
                    check_source_geometry(&serde_json::to_vec(&bad).unwrap(), 0, &geometry),
                    Err(EngError::Checkpoint(_))
                ),
                "wrong {key}"
            );
        }
        for key in ["rope_theta", "partial_rotary_factor", "rope_type"] {
            let mut bad = config.clone();
            bad["text_config"]["rope_parameters"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(
                matches!(
                    check_source_geometry(&serde_json::to_vec(&bad).unwrap(), 0, &geometry),
                    Err(EngError::Checkpoint(_))
                ),
                "missing {key}"
            );
        }
    }

    #[test]
    fn maps_only_full_attention_source_roles() {
        for (role, suffix) in [
            (Role::AttnQ, "self_attn.q_proj"),
            (Role::AttnK, "self_attn.k_proj"),
            (Role::AttnV, "self_attn.v_proj"),
            (Role::AttnOutput, "self_attn.o_proj"),
        ] {
            assert_eq!(projection_suffix(role).unwrap(), suffix);
        }
        for (role, suffix) in [
            (Role::AttnNorm, "input_layernorm.weight"),
            (Role::PostAttentionNorm, "post_attention_layernorm.weight"),
            (Role::AttnQNorm, "self_attn.q_norm.weight"),
            (Role::AttnKNorm, "self_attn.k_norm.weight"),
        ] {
            assert_eq!(norm_suffix(role).unwrap(), suffix);
        }
        assert!(matches!(
            projection_suffix(Role::AttnQkv),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            norm_suffix(Role::SsmNorm),
            Err(EngError::Checkpoint(_))
        ));
    }
}
