//! Flash-Next family checks against the artifact configuration values.
//!
//! The fixture holds the pinned artifact `config.json` values recorded by the
//! P0B-R3 inspection of `Qwen/Qwen3.8-Flash-Next-oQ4e-mtp`
//! (`docs/environment/flashnext-pin.json`, `docs/design/flash-next-executor.md`).

use inference_engine::{
    EngError,
    family::{Act, AttentionKind, QWEN3_8_FLASH_NEXT, Router, qwen3_8_flash_next},
};
use sha2::{Digest, Sha256};

const CONFIG_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen3_8_flash_next_config.json"
);
const CONFIG_SHA256: &str = "9519de5d50cddd7ed57434810f775c1f93989d0ac59e18700ef5377221da1193";

fn config_root() -> serde_json::Value {
    let text = std::fs::read_to_string(CONFIG_PATH).expect("fixture config");
    serde_json::from_str(&text).expect("fixture json")
}

fn text_config() -> serde_json::Value {
    config_root()
        .get("text_config")
        .expect("text_config")
        .clone()
}

fn field_u32(value: &serde_json::Value, key: &str) -> u32 {
    value.get(key).and_then(|v| v.as_u64()).expect(key) as u32
}

fn fixture_bytes() -> Vec<u8> {
    std::fs::read(CONFIG_PATH).expect("fixture config")
}

#[test]
fn flash_next_fixture_is_hash_pinned() {
    let digest = Sha256::digest(fixture_bytes());
    assert_eq!(format!("{digest:x}"), CONFIG_SHA256);
}

#[test]
fn flash_next_constants_match_the_artifact_config() {
    let root = config_root();
    let t = text_config();
    let spec = qwen3_8_flash_next();
    assert_eq!(spec, QWEN3_8_FLASH_NEXT);
    assert_eq!(spec.name, "qwen3_8_flash_next");
    assert_eq!(spec.layers, field_u32(&t, "num_hidden_layers"));
    assert_eq!(spec.hidden, field_u32(&t, "hidden_size"));
    assert_eq!(spec.heads_q, field_u32(&t, "num_attention_heads"));
    assert_eq!(spec.heads_kv, field_u32(&t, "num_key_value_heads"));
    assert_eq!(spec.head_dim, field_u32(&t, "head_dim"));
    assert_eq!(spec.vocab, field_u32(&t, "vocab_size"));
    assert_eq!(spec.experts, field_u32(&t, "num_experts"));
    assert_eq!(spec.top_k, field_u32(&t, "num_experts_per_tok"));
    assert_eq!(spec.moe_inter, field_u32(&t, "moe_intermediate_size"));
    assert_eq!(
        spec.shared_inter,
        field_u32(&t, "shared_expert_intermediate_size")
    );
    let rope_expected = t
        .get("rope_parameters")
        .and_then(|v| v.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .expect("rope_parameters.rope_theta");
    assert_eq!(f64::from(spec.rope_theta), rope_expected);
    assert_eq!(f64::from(spec.rope_theta), 10_000_000.0);
    assert_eq!(t["rms_norm_eps"].as_f64(), Some(1e-6));
    assert_eq!(f64::from(spec.rms_norm_eps), 1e-6_f32 as f64);
    let quant = root.get("quantization").expect("quantization");
    assert_eq!(spec.quant_bits, field_u32(quant, "bits"));
    assert_eq!(spec.quant_group, field_u32(quant, "group_size"));
    assert_eq!(spec.act, Act::Silu);
    assert_eq!(t["hidden_act"].as_str(), Some("silu"));
    let renorm = t
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    assert_eq!(spec.router, Router::SoftmaxTopK { renorm });
    assert!(spec.mtp);
    assert_eq!(field_u32(&t, "mtp_num_hidden_layers"), 1);
    for model_type in [
        root["model_type"].as_str().expect("root model_type"),
        t["model_type"].as_str().expect("text model_type"),
    ] {
        assert!(
            spec.model_types.contains(&model_type),
            "{model_type} is not declared by the family"
        );
    }
    let linear = spec.linear.expect("the family has delta layers");
    assert_eq!(linear.key_heads, field_u32(&t, "linear_num_key_heads"));
    assert_eq!(linear.key_head_dim, field_u32(&t, "linear_key_head_dim"));
    assert_eq!(linear.value_heads, field_u32(&t, "linear_num_value_heads"));
    assert_eq!(
        linear.value_head_dim,
        field_u32(&t, "linear_value_head_dim")
    );
    assert_eq!(linear.conv_kernel, field_u32(&t, "linear_conv_kernel_dim"));
    let hyper = spec.hyper.expect("the family widens the residual stream");
    assert_eq!(hyper.branches, field_u32(&t, "hc_count"));
    assert_eq!(hyper.rank, field_u32(&t, "hc_lowrank"));
    assert_eq!(
        hyper.width(spec.hidden).expect("hyper width"),
        4 * spec.hidden
    );
}

#[test]
fn flash_next_passes_the_config_check() {
    let spec = qwen3_8_flash_next();
    spec.check_config(&fixture_bytes())
        .expect("the artifact configuration matches the constant");
}

#[test]
fn flash_next_layer_order_matches_config_layer_types() {
    let t = text_config();
    let listed = t
        .get("layer_types")
        .and_then(|v| v.as_array())
        .expect("layer_types");
    let spec = qwen3_8_flash_next();
    assert_eq!(spec.layers_attention.len(), listed.len());
    for (index, kind) in spec.layers_attention.iter().enumerate() {
        let expected = match listed[index].as_str().expect("layer kind") {
            "linear_attention" => AttentionKind::DeltaNet,
            // The configuration vocabulary cannot express the sparse indexer;
            // those layers execute as Qsa.
            "full_attention" => AttentionKind::Qsa,
            other => panic!("unknown layer type {other}"),
        };
        assert_eq!(*kind, expected, "layer {index}");
    }
}

fn checked(edit: impl FnOnce(&mut serde_json::Value)) -> Result<(), EngError> {
    let mut root = config_root();
    edit(&mut root);
    qwen3_8_flash_next().check_config(root.to_string().as_bytes())
}

fn checkpoint_message(result: Result<(), EngError>) -> String {
    match result {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a checkpoint error, got {other:?}"),
    }
}

#[test]
fn flash_next_config_check_rejects_changed_values() {
    use serde_json::json;
    for (pointer, value, field) in [
        (
            "/text_config/num_hidden_layers",
            json!(41),
            "num_hidden_layers",
        ),
        ("/text_config/hidden_size", json!(1024), "hidden_size"),
        ("/text_config/num_experts", json!(256), "num_experts"),
        (
            "/text_config/num_experts_per_tok",
            json!(8),
            "num_experts_per_tok",
        ),
        (
            "/text_config/moe_intermediate_size",
            json!(512),
            "moe_intermediate_size",
        ),
        (
            "/text_config/shared_expert_intermediate_size",
            json!(512),
            "shared_expert_intermediate_size",
        ),
        (
            "/text_config/rope_parameters/rope_theta",
            json!(100_000),
            "rope_theta",
        ),
        (
            "/text_config/linear_num_value_heads",
            json!(32),
            "linear_num_value_heads",
        ),
        ("/text_config/model_type", json!("qwen3_next"), "model_type"),
        (
            "/text_config/full_attention_interval",
            json!(3),
            "full_attention_interval",
        ),
        (
            "/text_config/layer_types/3",
            json!("linear_attention"),
            "layer_types",
        ),
    ] {
        let message = checkpoint_message(checked(|root| {
            *root.pointer_mut(pointer).expect(pointer) = value.clone();
        }));
        assert!(message.contains(field), "{pointer}: {message}");
        assert!(
            message.contains("qwen3_8_flash_next"),
            "{pointer}: {message}"
        );
    }
    // The quantization block is listed twice in the artifact; both copies must move together.
    let message = checkpoint_message(checked(|root| {
        for block in ["quantization", "quantization_config"] {
            root[block]["bits"] = json!(8);
            root[block]["group_size"] = json!(32);
        }
    }));
    assert!(message.contains("quantization"), "{message}");
    assert!(message.contains("qwen3_8_flash_next"), "{message}");
}

#[test]
fn flash_next_layer_order_follows_the_interval_without_layer_types() {
    checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config object")
            .remove("layer_types");
    })
    .expect("interval 4 reproduces the qsa order");
    let message = checkpoint_message(checked(|root| {
        let text = root["text_config"].as_object_mut().expect("text_config");
        text.remove("layer_types");
        text.insert("full_attention_interval".into(), serde_json::json!(2));
    }));
    assert!(message.contains("full_attention_interval"), "{message}");
    assert!(message.contains("qwen3_8_flash_next"), "{message}");
}
