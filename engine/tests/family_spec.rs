use inference_engine::{
    EngError,
    family::{
        Act, AttentionKind, FamilySpec, LinearAttention, QWEN3_6_35B_A3B, Router, TierAssignment,
    },
};
use sha2::{Digest, Sha256};

const CONFIG_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen3_6_35b_a3b_config.json"
);
const CONFIG_SHA256: &str = "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767";

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

#[test]
fn fixture_is_hash_pinned() {
    let bytes = std::fs::read(CONFIG_PATH).expect("fixture config");
    let digest = Sha256::digest(&bytes);
    assert_eq!(format!("{digest:x}"), CONFIG_SHA256);
}

#[test]
fn constants_match_checkpoint_config() {
    let root = config_root();
    let t = text_config();
    let spec = QWEN3_6_35B_A3B;
    assert_eq!(spec.name, "qwen3_6_35b_a3b");
    assert_eq!(spec.layers, field_u32(&t, "num_hidden_layers"));
    assert_eq!(spec.hidden, field_u32(&t, "hidden_size"));
    assert_eq!(spec.heads_q, field_u32(&t, "num_attention_heads"));
    assert_eq!(spec.heads_kv, field_u32(&t, "num_key_value_heads"));
    assert_eq!(spec.head_dim, field_u32(&t, "head_dim"));
    let rope_expected = t
        .get("rope_parameters")
        .and_then(|v| v.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .expect("rope_parameters.rope_theta");
    assert_eq!(f64::from(spec.rope_theta), rope_expected);
    assert_eq!(spec.vocab, field_u32(&t, "vocab_size"));
    assert_eq!(spec.experts, field_u32(&t, "num_experts"));
    assert_eq!(spec.top_k, field_u32(&t, "num_experts_per_tok"));
    assert_eq!(spec.moe_inter, field_u32(&t, "moe_intermediate_size"));
    assert_eq!(
        spec.shared_inter,
        field_u32(&t, "shared_expert_intermediate_size")
    );
    let quant = root.get("quantization").expect("quantization");
    assert_eq!(spec.quant_bits, field_u32(quant, "bits"));
    assert_eq!(spec.quant_group, field_u32(quant, "group_size"));
    assert_eq!(spec.act, Act::Silu);
    let renorm = t
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    assert_eq!(spec.router, Router::SoftmaxTopK { renorm });
    assert!(!spec.mtp);
    assert_eq!(f64::from(spec.rms_norm_eps), 1e-6_f32 as f64);
    assert_eq!(t["rms_norm_eps"].as_f64(), Some(1e-6));
    assert!(
        spec.model_types
            .contains(&root["model_type"].as_str().unwrap())
    );
    assert!(
        spec.model_types
            .contains(&t["model_type"].as_str().unwrap())
    );
    let linear = spec.linear.expect("the first family has delta layers");
    assert_eq!(linear.key_heads, field_u32(&t, "linear_num_key_heads"));
    assert_eq!(linear.key_head_dim, field_u32(&t, "linear_key_head_dim"));
    assert_eq!(linear.value_heads, field_u32(&t, "linear_num_value_heads"));
    assert_eq!(
        linear.value_head_dim,
        field_u32(&t, "linear_value_head_dim")
    );
    assert_eq!(linear.conv_kernel, field_u32(&t, "linear_conv_kernel_dim"));
}

#[test]
fn linear_widths_follow_the_pinned_inventory() {
    // Widths from docs/design/stepping-stone-key-table.md: 16 x 128 keys, 32 x 128 values.
    let linear = QWEN3_6_35B_A3B.linear.expect("linear dimensions");
    assert_eq!(linear.key_dim().expect("key width"), 2048);
    assert_eq!(linear.value_dim().expect("value width"), 4096);
    assert_eq!(linear.conv_dim().expect("fused width"), 8192);
}

#[test]
fn constants_pass_the_config_check() {
    let bytes = std::fs::read(CONFIG_PATH).expect("fixture config");
    QWEN3_6_35B_A3B
        .check_config(&bytes)
        .expect("the pinned configuration matches the constant");
}

fn checked(edit: impl FnOnce(&mut serde_json::Value)) -> Result<(), EngError> {
    let mut root = config_root();
    edit(&mut root);
    QWEN3_6_35B_A3B.check_config(root.to_string().as_bytes())
}

fn checkpoint_message(result: Result<(), EngError>) -> String {
    match result {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a checkpoint error, got {other:?}"),
    }
}

#[test]
fn every_checked_field_rejects_a_changed_config() {
    use serde_json::json;
    for (pointer, value, field) in [
        (
            "/text_config/num_hidden_layers",
            json!(41),
            "num_hidden_layers",
        ),
        ("/text_config/hidden_size", json!(1024), "hidden_size"),
        (
            "/text_config/num_attention_heads",
            json!(32),
            "num_attention_heads",
        ),
        (
            "/text_config/num_key_value_heads",
            json!(4),
            "num_key_value_heads",
        ),
        ("/text_config/head_dim", json!(128), "head_dim"),
        ("/text_config/vocab_size", json!(248_321), "vocab_size"),
        ("/text_config/num_experts", json!(128), "num_experts"),
        (
            "/text_config/num_experts_per_tok",
            json!(4),
            "num_experts_per_tok",
        ),
        (
            "/text_config/moe_intermediate_size",
            json!(256),
            "moe_intermediate_size",
        ),
        (
            "/text_config/shared_expert_intermediate_size",
            json!(1024),
            "shared_expert_intermediate_size",
        ),
        (
            "/text_config/rope_parameters/rope_theta",
            json!(10_000),
            "rope_theta",
        ),
        (
            "/text_config/hidden_act",
            json!("gelu_pytorch_tanh"),
            "hidden_act",
        ),
        (
            "/text_config/linear_num_key_heads",
            json!(8),
            "linear_num_key_heads",
        ),
        (
            "/text_config/linear_key_head_dim",
            json!(64),
            "linear_key_head_dim",
        ),
        (
            "/text_config/linear_num_value_heads",
            json!(16),
            "linear_num_value_heads",
        ),
        (
            "/text_config/linear_value_head_dim",
            json!(64),
            "linear_value_head_dim",
        ),
        (
            "/text_config/linear_conv_kernel_dim",
            json!(3),
            "linear_conv_kernel_dim",
        ),
        (
            "/text_config/layer_types/3",
            json!("linear_attention"),
            "layer_types",
        ),
        ("/quantization/bits", json!(8), "quantization"),
        ("/quantization/group_size", json!(32), "quantization"),
    ] {
        let message = checkpoint_message(checked(|root| {
            // The pinned file lists the quantization block twice, and the parser rejects a conflict.
            if let Some(rest) = pointer.strip_prefix("/quantization/") {
                *root
                    .pointer_mut(&format!("/quantization_config/{rest}"))
                    .expect(pointer) = value.clone();
            }
            *root.pointer_mut(pointer).expect(pointer) = value;
        }));
        assert!(message.contains(field), "{pointer}: {message}");
        assert!(message.contains("qwen3_6_35b_a3b"), "{pointer}: {message}");
    }
}

#[test]
fn router_renormalization_defaults_to_true_and_follows_the_config() {
    // The pinned configuration omits norm_topk_prob, and the reference defaults it to true.
    checked(|root| {
        root["text_config"]["norm_topk_prob"] = serde_json::json!(true);
    })
    .expect("an explicit true matches the constant");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["norm_topk_prob"] = serde_json::json!(false);
    }));
    assert!(message.contains("norm_topk_prob"), "{message}");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["norm_topk_prob"] = serde_json::json!("yes");
    }));
    assert!(message.contains("norm_topk_prob"), "{message}");
}

#[test]
fn a_missing_or_mistyped_field_is_rejected() {
    for field in [
        "hidden_size",
        "head_dim",
        "vocab_size",
        "num_experts",
        "linear_conv_kernel_dim",
    ] {
        let message = checkpoint_message(checked(|root| {
            root["text_config"]
                .as_object_mut()
                .expect("text_config object")
                .remove(field);
        }));
        assert!(message.contains(field), "{field}: {message}");
    }
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["vocab_size"] = serde_json::json!("248320");
    }));
    assert!(message.contains("vocab_size"), "{message}");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["rope_parameters"]
            .as_object_mut()
            .expect("rope object")
            .remove("rope_theta");
    }));
    assert!(message.contains("rope_theta"), "{message}");
}

#[test]
fn layer_order_follows_the_full_attention_interval() {
    // The pinned reference reads only the interval (qwen3_5.py:43,203). The config file
    // has both fields, and layer_types must agree with the interval.
    checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config object")
            .remove("layer_types");
    })
    .expect("interval 4 reproduces the pinned order");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["full_attention_interval"] = serde_json::json!(2);
    }));
    assert!(message.contains("full_attention_interval"), "{message}");
    let message = checkpoint_message(checked(|root| {
        let text = root["text_config"].as_object_mut().expect("text_config");
        text.remove("layer_types");
        text.insert("full_attention_interval".into(), serde_json::json!(2));
    }));
    assert!(message.contains("full_attention_interval"), "{message}");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["layer_types"][0] = serde_json::json!("full_attention");
    }));
    assert!(message.contains("layer_types"), "{message}");
}

#[test]
fn a_missing_full_attention_interval_defaults_to_four() {
    // The pinned reference defaults the interval to 4 (qwen3_5.py:43,203).
    checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove("full_attention_interval");
    })
    .expect("layer_types that follow every fourth layer agree with the default");
    checked(|root| {
        let text = root["text_config"].as_object_mut().expect("text_config");
        text.remove("full_attention_interval");
        text.remove("layer_types");
    })
    .expect("the default interval alone reproduces the pinned order");
    let message = checkpoint_message(checked(|root| {
        let text = root["text_config"].as_object_mut().expect("text_config");
        text.remove("full_attention_interval");
        text["layer_types"][3] = serde_json::json!("linear_attention");
    }));
    assert!(message.contains("layer_types"), "{message}");
    assert!(message.contains("interval 4"), "{message}");
}

#[test]
fn a_flat_config_is_read_without_text_config() {
    let mut root = config_root();
    let text = root["text_config"].clone();
    let flat = root.as_object_mut().expect("root object");
    flat.remove("text_config");
    for (key, value) in text.as_object().expect("text object") {
        flat.insert(key.clone(), value.clone());
    }
    QWEN3_6_35B_A3B
        .check_config(root.to_string().as_bytes())
        .expect("flat configuration matches");
}

#[test]
fn duplicate_members_and_bad_json_are_rejected() {
    let text = std::fs::read_to_string(CONFIG_PATH).expect("fixture config");
    let duplicated = text.replacen("\"vocab_size\"", "\"vocab_size\": 1, \"vocab_size\"", 1);
    let message = checkpoint_message(QWEN3_6_35B_A3B.check_config(duplicated.as_bytes()));
    assert!(message.contains("duplicate"), "{message}");
    let message = checkpoint_message(QWEN3_6_35B_A3B.check_config(b"[1, 2]"));
    assert!(message.contains("object"), "{message}");
    let message = checkpoint_message(QWEN3_6_35B_A3B.check_config(b"{"));
    assert!(message.contains("configuration JSON"), "{message}");
}

fn with_linear(linear: Option<LinearAttention>) -> FamilySpec {
    FamilySpec {
        linear,
        ..QWEN3_6_35B_A3B
    }
}

fn family_error(spec: &FamilySpec) -> String {
    match TierAssignment::from_keys(&[], spec) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a checkpoint error, got {other:?}"),
    }
}

#[test]
fn delta_families_require_consistent_linear_dimensions() {
    let pinned = QWEN3_6_35B_A3B.linear.expect("linear dimensions");
    assert!(family_error(&with_linear(None)).contains("linear attention dimensions"));
    for (broken, expected) in [
        (
            LinearAttention {
                key_heads: 0,
                ..pinned
            },
            "key heads must exceed zero",
        ),
        (
            LinearAttention {
                key_head_dim: 0,
                ..pinned
            },
            "key head dimension",
        ),
        (
            LinearAttention {
                value_heads: 0,
                ..pinned
            },
            "value heads must exceed zero",
        ),
        (
            LinearAttention {
                value_head_dim: 0,
                ..pinned
            },
            "value head dimension",
        ),
        (
            LinearAttention {
                conv_kernel: 0,
                ..pinned
            },
            "convolution kernel",
        ),
        (
            LinearAttention {
                conv_kernel: 1,
                ..pinned
            },
            "at least 2",
        ),
        (
            LinearAttention {
                value_heads: 24,
                ..pinned
            },
            "must divide by key heads",
        ),
        (
            LinearAttention {
                key_heads: 65_536,
                key_head_dim: 65_536,
                value_heads: 65_536,
                value_head_dim: 1,
                ..pinned
            },
            "overflows u32",
        ),
        (
            LinearAttention {
                value_heads: 65_536,
                value_head_dim: 65_536,
                ..pinned
            },
            "overflows u32",
        ),
        (
            LinearAttention {
                key_heads: 1,
                key_head_dim: 2_147_483_648,
                value_heads: 1,
                value_head_dim: 1,
                conv_kernel: 4,
            },
            "overflows u32",
        ),
        (
            LinearAttention {
                key_heads: 1,
                key_head_dim: 1,
                value_heads: 1,
                value_head_dim: u32::MAX - 1,
                conv_kernel: 4,
            },
            "overflows u32",
        ),
    ] {
        let message = family_error(&with_linear(Some(broken)));
        assert!(message.contains(expected), "{broken:?}: {message}");
        assert!(message.contains("qwen3_6_35b_a3b"), "{message}");
    }
}

#[test]
fn layers_attention_order_matches_config_layer_types() {
    let t = text_config();
    let listed = t
        .get("layer_types")
        .and_then(|v| v.as_array())
        .expect("layer_types");
    let spec = QWEN3_6_35B_A3B;
    assert_eq!(spec.layers_attention.len(), listed.len());
    for (index, kind) in spec.layers_attention.iter().enumerate() {
        let expected = match listed[index].as_str().expect("layer kind") {
            "linear_attention" => AttentionKind::DeltaNet,
            "full_attention" => AttentionKind::Full,
            other => panic!("unknown layer type {other}"),
        };
        assert_eq!(*kind, expected, "layer {index}");
    }
    let full = spec
        .layers_attention
        .iter()
        .filter(|k| **k == AttentionKind::Full)
        .count();
    assert_eq!(full, 10);
    assert_eq!(spec.layers_attention.len(), 40);
}

#[test]
fn unsupported_config_values_are_named() {
    use serde_json::json;
    for (pointer, value, expected) in [
        ("/text_config/hidden_act", json!("relu"), "hidden_act"),
        ("/text_config/hidden_act", json!(3), "hidden_act"),
        (
            "/text_config/layer_types",
            json!("linear_attention"),
            "layer_types",
        ),
        (
            "/text_config/layer_types/0",
            json!("sliding_attention"),
            "layer_types entry",
        ),
        (
            "/text_config/rope_parameters/rope_theta",
            json!("1e7"),
            "rope_theta",
        ),
    ] {
        let message = checkpoint_message(checked(|root| {
            *root.pointer_mut(pointer).expect(pointer) = value;
        }));
        assert!(message.contains(expected), "{pointer}: {message}");
    }
    let message = checkpoint_message(checked(|root| {
        let text = root["text_config"].as_object_mut().expect("text_config");
        text.remove("layer_types");
        text.insert("full_attention_interval".into(), json!(0));
    }));
    assert!(message.contains("full_attention_interval"), "{message}");
}

#[test]
fn rope_theta_comes_only_from_the_nested_parameters() {
    use serde_json::json;
    // qwen3_5.py:54-61,72-82 reads rope_parameters.rope_theta and defaults to 1e5 without it.
    let message = checkpoint_message(checked(|root| {
        let text = root["text_config"].as_object_mut().expect("text_config");
        let theta = text["rope_parameters"]
            .as_object_mut()
            .expect("rope object")
            .remove("rope_theta")
            .expect("pinned rope_theta");
        text.insert("rope_theta".into(), theta);
    }));
    assert!(message.contains("rope_parameters.rope_theta"), "{message}");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["rope_theta"] = json!(100_000);
    }));
    assert!(message.contains("rope_theta"), "{message}");
    checked(|root| {
        root["text_config"]["rope_theta"] = json!(10_000_000);
    })
    .expect("an agreeing top-level copy is harmless");
    // The f32 constant must match exactly, not after rounding.
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["rope_parameters"]["rope_theta"] = json!(10_000_000.4);
    }));
    assert!(message.contains("rope_theta"), "{message}");
}

#[test]
fn the_model_type_must_belong_to_the_family() {
    use serde_json::json;
    for (pointer, value) in [
        ("/text_config/model_type", json!("qwen3_next")),
        ("/model_type", json!("qwen3_next")),
        ("/text_config/model_type", json!(7)),
    ] {
        let message = checkpoint_message(checked(|root| {
            *root.pointer_mut(pointer).expect(pointer) = value;
        }));
        assert!(message.contains("model_type"), "{pointer}: {message}");
    }
    let message = checkpoint_message(checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove("model_type");
        root.as_object_mut().expect("root").remove("model_type");
    }));
    assert!(message.contains("model_type"), "{message}");
    // Either level alone names the family.
    checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove("model_type");
    })
    .expect("the root model_type suffices");
}

#[test]
fn execution_switches_must_match_the_constant() {
    use serde_json::json;
    for (field, value) in [
        ("attn_output_gate", json!(false)),
        ("rms_norm_eps", json!(1e-5)),
        ("tie_word_embeddings", json!(true)),
        ("attention_bias", json!(true)),
    ] {
        let message = checkpoint_message(checked(|root| {
            root["text_config"][field] = value;
        }));
        assert!(message.contains(field), "{field}: {message}");
    }
    // The pinned file omits attention_bias; the reference default is false.
    checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove("attention_bias");
    })
    .expect("attention_bias defaults to false");
    // The reference always gates the attention output, so an absent flag means true.
    checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove("attn_output_gate");
    })
    .expect("attn_output_gate defaults to true");
    let message = checkpoint_message(checked(|root| {
        root["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove("rms_norm_eps");
    }));
    assert!(message.contains("rms_norm_eps"), "{message}");
}

#[test]
fn a_null_head_dim_is_rejected() {
    // mlx-lm derives hidden // heads for a null head_dim. The check fails closed instead.
    let message = checkpoint_message(checked(|root| {
        root["text_config"]["head_dim"] = serde_json::Value::Null;
    }));
    assert!(message.contains("head_dim"), "{message}");
}

#[test]
fn the_default_quantization_must_be_affine_and_present() {
    let message = checkpoint_message(checked(|root| {
        for block in ["quantization", "quantization_config"] {
            root[block]["mode"] = serde_json::json!("mxfp4");
            root[block]["bits"] = serde_json::json!(4);
            root[block]["group_size"] = serde_json::json!(32);
        }
    }));
    assert!(message.contains("quantization"), "{message}");
    let message = checkpoint_message(checked(|root| {
        let object = root.as_object_mut().expect("root object");
        object.remove("quantization");
        object.remove("quantization_config");
    }));
    assert!(message.contains("quantization"), "{message}");
}

#[test]
fn a_family_without_delta_layers_skips_the_linear_fields() {
    static FULL_ONLY: [AttentionKind; 1] = [AttentionKind::Full];
    let spec = FamilySpec {
        layers: 1,
        layers_attention: &FULL_ONLY,
        linear: None,
        ..QWEN3_6_35B_A3B
    };
    let mut root = config_root();
    let text = root["text_config"].as_object_mut().expect("text_config");
    text.insert("num_hidden_layers".into(), serde_json::json!(1));
    text.insert("layer_types".into(), serde_json::json!(["full_attention"]));
    text.insert("full_attention_interval".into(), serde_json::json!(1));
    for field in [
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ] {
        text.remove(field);
    }
    spec.check_config(root.to_string().as_bytes())
        .expect("no linear fields are required");
}

#[test]
fn a_sigmoid_router_has_no_config_mapping() {
    let spec = FamilySpec {
        router: Router::SigmoidGroup { groups: 8 },
        ..QWEN3_6_35B_A3B
    };
    let bytes = std::fs::read(CONFIG_PATH).expect("fixture config");
    let message = checkpoint_message(spec.check_config(&bytes));
    assert!(message.contains("router"), "{message}");
}

#[test]
fn non_boolean_flags_are_rejected_with_the_flag_name() {
    use serde_json::json;
    for name in ["attn_output_gate", "tie_word_embeddings", "attention_bias"] {
        for value in [json!("true"), json!(1), json!(null)] {
            let message = checkpoint_message(checked(|root| {
                root["text_config"][name] = value.clone();
            }));
            assert_eq!(
                message,
                format!("configuration {name} must be a boolean"),
                "{name}: {value}"
            );
        }
    }
}
