use serde_json::{Value, json};

use crate::{EngError, checkpoint_json::UniqueValue};

pub(super) const FULL_LAYERS: [u64; 14] = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48];

/// Checks the pinned Flint inference metadata, not weights or backend availability.
pub fn check_config(json: &[u8]) -> Result<(), EngError> {
    let root = serde_json::from_slice::<UniqueValue>(json)
        .map_err(|error| EngError::Checkpoint(format!("Flint configuration JSON: {error}")))?
        .0;
    if !root.is_object() {
        return Err(EngError::Checkpoint(
            "Flint configuration must be a JSON object".into(),
        ));
    }
    expect(&root, "model_type", json!("flint"))?;
    expect(&root, "architectures", json!(["FlintForCausalLM"]))?;
    check_dimensions(&root)?;
    check_math(&root)?;
    check_token_ids(&root)?;
    check_layers(&root)?;
    check_aliases(&root)?;
    check_rope(&root)?;
    check_quantization(&root)?;
    Ok(())
}

fn check_dimensions(root: &Value) -> Result<(), EngError> {
    for (name, wanted) in [
        ("hidden_size", 2048_u64),
        ("intermediate_size", 6144),
        ("head_dim", 128),
        ("num_hidden_layers", 49),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 4),
        ("num_experts", 256),
        ("num_experts_per_tok", 6),
        ("moe_intermediate_size", 512),
        ("shared_expert_intermediate_size", 1024),
        ("decoder_sparse_step", 1),
        ("sliding_window", 513),
        ("max_position_embeddings", 262144),
        ("vocab_size", 150144),
    ] {
        if root.get(name).and_then(Value::as_u64) != Some(wanted) {
            return Err(mismatch(name));
        }
    }
    Ok(())
}

fn check_math(root: &Value) -> Result<(), EngError> {
    for (name, wanted) in [
        ("hidden_act", "silu"),
        ("router_score_function", "sigmoid"),
        ("softmax_type", "learnable"),
        ("torch_dtype", "bfloat16"),
    ] {
        expect(root, name, json!(wanted))?;
    }
    for (name, wanted) in [
        ("attention_bias", false),
        ("norm_topk_prob", true),
        ("tie_word_embeddings", false),
        ("use_cache", true),
        ("use_sliding_window", true),
        ("output_router_logits", false),
    ] {
        expect(root, name, json!(wanted))?;
    }
    for (name, wanted) in [
        ("attention_dropout", 0.0),
        ("rms_norm_eps", 1e-6),
        ("rope_theta", 10000.0),
    ] {
        if root.get(name).and_then(Value::as_f64) != Some(wanted) {
            return Err(mismatch(name));
        }
    }
    Ok(())
}

fn check_token_ids(root: &Value) -> Result<(), EngError> {
    expect(root, "bos_token_id", json!(150000))?;
    expect(root, "pad_token_id", json!(150001))?;
    expect(
        root,
        "eos_token_id",
        json!([150001, 150003, 150005, 150018]),
    )
}

fn check_layers(root: &Value) -> Result<(), EngError> {
    let layers = root
        .get("layer_types")
        .and_then(Value::as_array)
        .filter(|layers| layers.len() == 49)
        .ok_or_else(|| mismatch("layer_types"))?;
    for (index, layer) in layers.iter().enumerate() {
        let wanted = if FULL_LAYERS.contains(&(index as u64)) {
            "full_attention"
        } else {
            "sliding_attention"
        };
        if layer.as_str() != Some(wanted) {
            return Err(mismatch("layer_types"));
        }
    }
    expect_indices(root, "no_rope_layer_indices", &FULL_LAYERS)?;
    expect_indices(root, "mlp_only_layers", &[0, 1])
}

fn check_aliases(root: &Value) -> Result<(), EngError> {
    for (name, wanted) in [
        ("dtype", json!("bfloat16")),
        ("scoring_func", json!("sigmoid")),
        ("num_local_experts", json!(256)),
        ("attention_sink", json!(true)),
        ("router_expert_bias", json!(false)),
    ] {
        if root.get(name).is_some() {
            expect(root, name, wanted)?;
        }
    }
    Ok(())
}

fn check_rope(root: &Value) -> Result<(), EngError> {
    if let Some(parameters) = root.get("rope_parameters")
        && (parameters.get("rope_theta").and_then(Value::as_f64) != Some(10000.0)
            || parameters.get("rope_type").and_then(Value::as_str) != Some("default"))
    {
        return Err(mismatch("rope_parameters"));
    }
    if root
        .get("rope_scaling")
        .is_some_and(|value| !value.is_null())
    {
        return Err(mismatch("rope_scaling"));
    }
    Ok(())
}

fn check_quantization(root: &Value) -> Result<(), EngError> {
    for (name, wanted) in [
        ("kv_key_bits", 8),
        ("kv_value_bits", 8),
        ("kv_group_size", 64),
    ] {
        if root.get(name).is_some() {
            expect(root, name, json!(wanted))?;
        }
    }
    let Some(quantization) = root.get("quantization") else {
        if root.get("mlx_export").is_some() {
            return Err(mismatch("quantization"));
        }
        return Ok(());
    };
    if !quantization.is_object() {
        return Err(mismatch("quantization"));
    }
    expect(quantization, "bits", json!(4))?;
    expect(quantization, "group_size", json!(64))?;
    if quantization.get("mode").is_some() {
        expect(quantization, "mode", json!("affine"))?;
    }
    let export = root
        .get("mlx_export")
        .filter(|value| value.is_object())
        .ok_or_else(|| mismatch("mlx_export"))?;
    for (name, wanted) in [
        ("bits", 4),
        ("routed_expert_bits", 4),
        ("embedding_bits", 8),
    ] {
        expect(export, name, json!(wanted))?;
    }
    expect_indices(export, "full_attention_layers", &FULL_LAYERS)
}

fn expect_indices(object: &Value, name: &str, wanted: &[u64]) -> Result<(), EngError> {
    let found = object
        .get(name)
        .and_then(Value::as_array)
        .filter(|values| values.len() == wanted.len())
        .ok_or_else(|| mismatch(name))?;
    if found
        .iter()
        .zip(wanted)
        .any(|(value, &wanted)| value.as_u64() != Some(wanted))
    {
        return Err(mismatch(name));
    }
    Ok(())
}

fn expect(object: &Value, name: &str, wanted: Value) -> Result<(), EngError> {
    if object.get(name) == Some(&wanted) {
        Ok(())
    } else {
        Err(mismatch(name))
    }
}

fn mismatch(name: &str) -> EngError {
    EngError::Checkpoint(format!(
        "Flint configuration {name} differs from the pinned profile"
    ))
}
