use super::{check_bolt_config, check_humanizer_config, open_bolt_q8, open_humanizer_q8};
use crate::family::spec::NEMOTRON_BOLT_E4B;

const HUMANIZER_CONFIG: &[u8] = include_bytes!("../../../testdata/gemma4/humanizer-config.json");
const BOLT_CONFIG: &[u8] = include_bytes!("../../../testdata/gemma4/bolt-config.json");

#[test]
fn accepts_the_pinned_bolt_configuration() {
    check_bolt_config(BOLT_CONFIG).unwrap();
}

#[test]
fn the_pinned_bolt_configuration_agrees_with_the_family_descriptor() {
    check_bolt_config(BOLT_CONFIG).unwrap();
    let spec = NEMOTRON_BOLT_E4B;
    let root: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    let text = &root["text_config"];
    let field = |name: &str| text[name].as_u64().expect("integer field") as u32;
    assert_eq!(field("num_hidden_layers"), spec.layers);
    assert_eq!(field("hidden_size"), spec.hidden);
    assert_eq!(field("num_attention_heads"), spec.heads_q);
    assert_eq!(field("num_key_value_heads"), spec.heads_kv);
    assert_eq!(field("head_dim"), spec.head_dim);
    assert_eq!(field("global_head_dim"), spec.global_head_dim);
    assert_eq!(field("sliding_window"), spec.sliding_window);
    assert_eq!(field("num_kv_shared_layers"), spec.kv_shared_layers);
    assert_eq!(field("vocab_size"), spec.vocab);
    assert_eq!(field("intermediate_size"), 10_240);
    assert_eq!(
        spec.feed_forward.expect("dense feed-forward").intermediate,
        10_240
    );
}

#[test]
fn rejects_altered_bolt_identity_and_shared_k_v_geometry() {
    let mut config: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    config["model_type"] = "gemma4_unified".into();
    assert_bolt_field(&serde_json::to_vec(&config).unwrap(), "model_type");

    let mut config: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    config["architectures"][0] = "Gemma4UnifiedForConditionalGeneration".into();
    assert_bolt_field(&serde_json::to_vec(&config).unwrap(), "architectures");

    let mut config: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    config["text_config"]["num_kv_shared_layers"] = 0.into();
    assert_bolt_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.num_kv_shared_layers",
    );

    let mut config: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    config["text_config"]["layer_types"][5] = "sliding_attention".into();
    assert_bolt_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.layer_types",
    );

    let mut config: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    config["text_config"]["attention_k_eq_v"] = true.into();
    assert_bolt_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.attention_k_eq_v",
    );

    let mut config: serde_json::Value = serde_json::from_slice(BOLT_CONFIG).unwrap();
    config["text_config"]["hidden_size"] = 3840.into();
    assert_bolt_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.hidden_size",
    );
}

fn assert_bolt_field(config: &[u8], field: &str) {
    match check_bolt_config(config) {
        Err(crate::EngError::Checkpoint(message)) => assert!(
            message.contains(field),
            "checkpoint error {message:?} does not name {field:?}"
        ),
        Err(error) => panic!("expected Checkpoint error naming {field:?}, got {error:?}"),
        Ok(()) => panic!("expected rejection naming {field:?}"),
    }
}

#[test]
fn accepts_the_pinned_humanizer_configuration() {
    check_humanizer_config(HUMANIZER_CONFIG).unwrap();
}

#[test]
fn rejects_a_wrong_root_model_type_and_names_the_field() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["model_type"] = "other_model".into();
    let bytes = serde_json::to_vec(&config).unwrap();
    assert_checkpoint_field(&bytes, "model_type");
}

fn assert_checkpoint_field(config: &[u8], field: &str) {
    match check_humanizer_config(config) {
        Err(crate::EngError::Checkpoint(message)) => assert!(
            message.contains(field),
            "checkpoint error {message:?} does not name {field:?}"
        ),
        Err(error) => panic!("expected Checkpoint error naming {field:?}, got {error:?}"),
        Ok(()) => panic!("expected rejection naming {field:?}"),
    }
}

#[test]
fn checks_the_pinned_architecture_and_both_tied_embedding_flags() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["architectures"][0] = "OtherForConditionalGeneration".into();
    assert_checkpoint_field(&serde_json::to_vec(&config).unwrap(), "architectures");

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["tie_word_embeddings"] = false.into();
    assert_checkpoint_field(&serde_json::to_vec(&config).unwrap(), "tie_word_embeddings");

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["model_type"] = "other_text".into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.model_type",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["tie_word_embeddings"] = false.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.tie_word_embeddings",
    );
}

#[test]
fn rejects_altered_dimensions_and_text_token_ids() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["hidden_size"] = 3841.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.hidden_size",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["num_global_key_value_heads"] = 2.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.num_global_key_value_heads",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["eos_token_id"] = 3.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.eos_token_id",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("max_position_embeddings");
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.max_position_embeddings",
    );
}

#[test]
fn enforces_five_sliding_then_one_full_layer_order() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["layer_types"][0] = "full_attention".into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.layer_types",
    );
}

#[test]
fn checks_both_rope_parameter_sets() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["rope_parameters"]["full_attention"]["rope_theta"] = 999_999.0.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.rope_parameters.full_attention.rope_theta",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["rope_parameters"]["sliding_attention"]["rope_type"] = "linear".into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.rope_parameters.sliding_attention.rope_type",
    );
}

#[test]
fn checks_activation_normalization_and_logit_softcap() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["hidden_activation"] = "gelu".into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.hidden_activation",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["rms_norm_eps"] = 1e-5.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.rms_norm_eps",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["final_logit_softcapping"] = 0.0.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.final_logit_softcapping",
    );
}

#[test]
fn checks_dense_causal_and_nonsharing_architecture_switches() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["attention_bias"] = true.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.attention_bias",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["attention_k_eq_v"] = false.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.attention_k_eq_v",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["enable_moe_block"] = true.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.enable_moe_block",
    );

    for field in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
        let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
        config["text_config"][field] = serde_json::Value::from(8);
        assert_checkpoint_field(
            &serde_json::to_vec(&config).unwrap(),
            &format!("text_config.{field}"),
        );
    }

    for (field, value) in [
        ("num_kv_shared_layers", serde_json::Value::from(1)),
        ("hidden_size_per_layer_input", serde_json::Value::from(1)),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
        config["text_config"][field] = value;
        assert_checkpoint_field(
            &serde_json::to_vec(&config).unwrap(),
            &format!("text_config.{field}"),
        );
    }

    for (field, value) in [
        ("use_double_wide_mlp", serde_json::Value::from(true)),
        ("use_cache", serde_json::Value::from(false)),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
        config["text_config"][field] = value;
        assert_checkpoint_field(
            &serde_json::to_vec(&config).unwrap(),
            &format!("text_config.{field}"),
        );
    }

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["use_bidirectional_attention"] = "always".into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.use_bidirectional_attention",
    );
}

#[test]
fn rejects_duplicate_json_members() {
    assert_checkpoint_field(
        br#"{"model_type":"gemma4_unified","model_type":"gemma4_unified"}"#,
        "model_type",
    );
    assert_checkpoint_field(
        br#"{"text_config":{"model_type":"gemma4_unified_text","model_type":"gemma4_unified_text"}}"#,
        "model_type",
    );
}

#[test]
fn rejects_wrong_object_types_and_missing_or_invalid_required_values() {
    assert_checkpoint_field(b"[]", "configuration");

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"] = serde_json::json!([]);
    assert_checkpoint_field(&serde_json::to_vec(&config).unwrap(), "text_config");

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config.as_object_mut().unwrap().remove("architectures");
    assert_checkpoint_field(&serde_json::to_vec(&config).unwrap(), "architectures");

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("model_type");
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.model_type",
    );

    for value in [
        serde_json::Value::Null,
        serde_json::Value::String("3840".into()),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
        config["text_config"]["hidden_size"] = value;
        assert_checkpoint_field(
            &serde_json::to_vec(&config).unwrap(),
            "text_config.hidden_size",
        );
    }

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("enable_moe_block");
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.enable_moe_block",
    );

    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["attention_k_eq_v"] = serde_json::Value::Null;
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.attention_k_eq_v",
    );
}

#[test]
fn rejects_a_layer_order_with_the_wrong_length() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["layer_types"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.layer_types",
    );
}

#[test]
fn rejects_non_unit_full_rope_factor() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["rope_parameters"]["full_attention"]["factor"] = 2.0.into();
    assert_checkpoint_field(
        &serde_json::to_vec(&config).unwrap(),
        "text_config.rope_parameters.full_attention.factor",
    );
}

#[test]
fn accepts_unit_rope_factor_and_rejects_present_non_numbers() {
    let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
    config["text_config"]["rope_parameters"]["full_attention"]["factor"] = 1.0.into();
    check_humanizer_config(&serde_json::to_vec(&config).unwrap()).unwrap();

    for value in [
        serde_json::Value::Null,
        serde_json::Value::Bool(true),
        serde_json::Value::String("1.0".into()),
    ] {
        let mut config: serde_json::Value = serde_json::from_slice(HUMANIZER_CONFIG).unwrap();
        config["text_config"]["rope_parameters"]["full_attention"]["factor"] = value;
        assert_checkpoint_field(
            &serde_json::to_vec(&config).unwrap(),
            "text_config.rope_parameters.full_attention.factor",
        );
    }
}

fn minimal_gguf_v3() -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&sized("payload.weight"));
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&[0u8; 32]);
    out
}

#[test]
fn rejects_a_file_without_the_gemma4_architecture_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.gguf");
    std::fs::write(&path, minimal_gguf_v3()).expect("write fixture");
    match open_humanizer_q8(&path, None) {
        Err(crate::EngError::Checkpoint(message)) => assert!(
            message.contains("general.architecture"),
            "checkpoint error {message:?} does not name general.architecture"
        ),
        other => panic!("expected Checkpoint naming general.architecture, got {other:?}"),
    }
}

const REAL_HEADER_SHA256: &str = "ca46011b7b65ad6dcc3944d86319d8d051c61cffc2b98551bbcf133888fb77b1";

struct Kv {
    key: String,
    bytes: Vec<u8>,
}

fn sized(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

fn kv_string(key: &str, value: &str) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&sized(value));
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

fn kv_u32(key: &str, value: u32) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

fn kv_f32(key: &str, value: f32) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&6u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

fn kv_f64(key: &str, value: f64) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&12u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

fn kv_bool(key: &str, value: bool) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&7u32.to_le_bytes());
    bytes.push(u8::from(value));
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

fn kv_i32_array(key: &str, values: &[i32]) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&5u32.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

fn kv_bool_array(key: &str, values: &[bool]) -> Kv {
    let mut bytes = sized(key);
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&7u32.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.push(u8::from(*value));
    }
    Kv {
        key: key.to_owned(),
        bytes,
    }
}

/// The scalar metadata of the pinned checkpoint, minus excluded keys.
fn base_kvs(exclude: &[&str]) -> Vec<Kv> {
    let mut heads = Vec::new();
    let mut pattern = Vec::new();
    for layer in 0..48 {
        let local = layer % 6 != 5;
        heads.push(if local { 8 } else { 1 });
        pattern.push(local);
    }
    let kvs = vec![
        kv_string("general.architecture", "gemma4"),
        kv_u32("gemma4.block_count", 48),
        kv_u32("gemma4.context_length", 262144),
        kv_u32("gemma4.embedding_length", 3840),
        kv_u32("gemma4.feed_forward_length", 15360),
        kv_u32("gemma4.attention.head_count", 16),
        kv_u32("gemma4.attention.key_length", 512),
        kv_u32("gemma4.attention.key_length_swa", 256),
        kv_u32("gemma4.attention.value_length", 512),
        kv_u32("gemma4.attention.value_length_swa", 256),
        kv_u32("gemma4.attention.sliding_window", 1024),
        kv_u32("gemma4.attention.shared_kv_layers", 0),
        kv_u32("gemma4.rope.dimension_count", 512),
        kv_u32("gemma4.rope.dimension_count_swa", 256),
        kv_f32("gemma4.rope.freq_base", 1_000_000.0),
        kv_f32("gemma4.rope.freq_base_swa", 10_000.0),
        kv_f32("gemma4.final_logit_softcapping", 30.0),
        kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6),
        kv_bool("tokenizer.ggml.add_bos_token", true),
        kv_u32("tokenizer.ggml.bos_token_id", 2),
        kv_u32("tokenizer.ggml.eos_token_id", 1),
        kv_u32("tokenizer.ggml.unknown_token_id", 3),
        kv_i32_array("gemma4.attention.head_count_kv", &heads),
        kv_bool_array("gemma4.attention.sliding_window_pattern", &pattern),
        kv_i32_array("tokenizer.ggml.suppress_tokens", &[258883, 258882]),
    ];
    kvs.into_iter()
        .filter(|kv| !exclude.contains(&kv.key.as_str()))
        .collect()
}

fn payload_len(ty: crate::weights::GgufType, dims: &[u64]) -> usize {
    let elements: u64 = dims.iter().product();
    match ty {
        crate::weights::GgufType::F32 => (elements * 4) as usize,
        _ => (elements / 32 * 34) as usize,
    }
}

/// Builds one GGUF v3 file from metadata and named tensors with zero payloads.
fn build_gguf(kvs: &[Kv], tensors: &[(&str, crate::weights::GgufType, &[u64])]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        out.extend_from_slice(&kv.bytes);
    }
    let mut offset = 0usize;
    for (name, ty, dims) in tensors {
        out.extend_from_slice(&sized(name));
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in *dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&ty.id().to_le_bytes());
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        offset = (offset + payload_len(*ty, dims)).div_ceil(32) * 32;
    }
    for (name, ty, dims) in tensors {
        out.resize(out.len().div_ceil(32) * 32, 0);
        let len = payload_len(*ty, dims);
        out.extend(std::iter::repeat_n(0u8, len));
        let _ = name;
    }
    out
}

fn globals() -> Vec<(&'static str, crate::weights::GgufType, &'static [u64])> {
    vec![
        ("output_norm.weight", crate::weights::GgufType::F32, &[3840]),
        ("rope_freqs.weight", crate::weights::GgufType::F32, &[256]),
    ]
}

fn open(bytes: &[u8]) -> Result<(), crate::EngError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("humanizer.gguf");
    std::fs::write(&path, bytes).expect("write fixture");
    open_humanizer_q8(&path, None)
}

fn expect_rejection(result: Result<(), crate::EngError>, expect: &str) {
    match result {
        Err(crate::EngError::Checkpoint(message)) => {
            assert!(message.contains(expect), "{expect:?} not in {message:?}");
        }
        other => panic!("expected Checkpoint with {expect:?}, got {other:?}"),
    }
}

#[test]
fn rejects_a_wrong_architecture_string() {
    let kvs: Vec<Kv> = base_kvs(&["general.architecture"])
        .into_iter()
        .chain([kv_string("general.architecture", "gemma3")])
        .collect();
    expect_rejection(open(&build_gguf(&kvs, &globals())), "general.architecture");
}

#[test]
fn rejects_wrong_scalar_metadata_values() {
    let kvs = base_kvs(&["gemma4.block_count"]);
    let kvs: Vec<Kv> = kvs
        .into_iter()
        .chain([kv_u32("gemma4.block_count", 47)])
        .collect();
    expect_rejection(open(&build_gguf(&kvs, &globals())), "gemma4.block_count");

    let kvs = base_kvs(&["gemma4.rope.freq_base"]);
    let kvs: Vec<Kv> = kvs
        .into_iter()
        .chain([kv_f32("gemma4.rope.freq_base", 10000.0)])
        .collect();
    expect_rejection(open(&build_gguf(&kvs, &globals())), "gemma4.rope.freq_base");
}

#[test]
fn rejects_broken_per_layer_attention_arrays() {
    let mut heads = Vec::new();
    for layer in 0..48 {
        heads.push(if layer % 6 != 5 { 8 } else { 1 });
    }
    heads[5] = 8;
    let kvs: Vec<Kv> = base_kvs(&["gemma4.attention.head_count_kv"])
        .into_iter()
        .chain([kv_i32_array("gemma4.attention.head_count_kv", &heads)])
        .collect();
    expect_rejection(
        open(&build_gguf(&kvs, &globals())),
        "head_count_kv is 8 at layer 5",
    );

    let mut pattern = Vec::new();
    for layer in 0..48 {
        pattern.push(layer % 6 != 5);
    }
    pattern[0] = false;
    let kvs: Vec<Kv> = base_kvs(&["gemma4.attention.sliding_window_pattern"])
        .into_iter()
        .chain([kv_bool_array(
            "gemma4.attention.sliding_window_pattern",
            &pattern,
        )])
        .collect();
    expect_rejection(
        open(&build_gguf(&kvs, &globals())),
        "sliding_window_pattern is 0 at layer 0",
    );

    let kvs: Vec<Kv> = base_kvs(&["tokenizer.ggml.suppress_tokens"])
        .into_iter()
        .chain([kv_i32_array("tokenizer.ggml.suppress_tokens", &[1, 2])])
        .collect();
    expect_rejection(open(&build_gguf(&kvs, &globals())), "suppress_tokens");
}

#[test]
fn rejects_missing_unexpected_and_mistyped_tensors() {
    let kvs = base_kvs(&[]);
    expect_rejection(
        open(&build_gguf(&kvs, &globals())),
        "missing tensor blk.0.attn_k.weight",
    );

    let mut tensors = globals();
    tensors.push(("blk.0.extra.weight", crate::weights::GgufType::F32, &[1]));
    expect_rejection(
        open(&build_gguf(&kvs, &tensors)),
        "unexpected tensor blk.0.extra.weight",
    );

    let mut tensors = globals();
    tensors.push((
        "blk.0.attn_q.weight",
        crate::weights::GgufType::F32,
        &[32, 32],
    ));
    expect_rejection(
        open(&build_gguf(&kvs, &tensors)),
        "tensor blk.0.attn_q.weight has type F32, expected Q8_0",
    );

    let mut tensors = globals();
    tensors.push((
        "blk.0.attn_q.weight",
        crate::weights::GgufType::Q80,
        &[32, 32],
    ));
    expect_rejection(
        open(&build_gguf(&kvs, &tensors)),
        "tensor blk.0.attn_q.weight has dims [32, 32], expected [3840, 4096]",
    );
}

#[test]
fn accepts_the_pinned_humanizer_q8_file_when_provided() {
    let Ok(path) = std::env::var("GEMMA4_Q8_FILE") else {
        eprintln!("GEMMA4_Q8_FILE not set; skipping the real-file check");
        return;
    };
    open_humanizer_q8(std::path::Path::new(&path), Some(REAL_HEADER_SHA256))
        .expect("the pinned Humanizer Q8_0 file opens");
    let wrong = open_humanizer_q8(std::path::Path::new(&path), Some(&"0".repeat(64)));
    expect_rejection(wrong, "header");
}

#[test]
fn accepts_the_pinned_bolt_q8_file_when_provided() {
    let Ok(path) = std::env::var("BOLT_Q8_FILE") else {
        eprintln!("BOLT_Q8_FILE not set; skipping the real-file check");
        return;
    };
    let pinned = std::env::var("BOLT_Q8_HEADER_SHA256").ok();
    open_bolt_q8(std::path::Path::new(&path), pinned.as_deref())
        .expect("the pinned Bolt Q8_0 file opens");
    let wrong = open_bolt_q8(std::path::Path::new(&path), Some(&"0".repeat(64)));
    expect_rejection(wrong, "header");
}

#[test]
fn rejects_an_unsupported_gguf_rms_epsilon() {
    const KEY: &str = "gemma4.attention.layer_norm_rms_epsilon";
    for entry in [
        kv_f32(KEY, 1.0e-5),
        kv_f32(KEY, 0.0),
        kv_f32(KEY, -1.0e-6),
        kv_f32(KEY, f32::NAN),
        kv_f32(KEY, f32::INFINITY),
        kv_f32(KEY, f32::from_bits(1.0e-6_f32.to_bits() + 1)),
        kv_f64(KEY, 1.0e-5),
        kv_f64(KEY, f64::from_bits(1.0e-6_f64.to_bits() + 1)),
        kv_u32(KEY, 1),
    ] {
        let kvs: Vec<Kv> = base_kvs(&[KEY]).into_iter().chain([entry]).collect();
        expect_rejection(open(&build_gguf(&kvs, &globals())), KEY);
    }
    expect_rejection(open(&build_gguf(&base_kvs(&[KEY]), &globals())), KEY);
}

/// Writes a Bolt-shaped GGUF file with sparse zero payloads: the full
/// inventory's Q8_0 matrices would otherwise occupy ~5 GiB of fixture bytes.
/// `payloads` overrides tensor bytes by index; an empty entry stays zero.
fn write_bolt_gguf_payload(
    kvs: &[Kv],
    tensors: &[(String, crate::weights::GgufType, Vec<u64>)],
    payloads: &[Vec<u8>],
    path: &std::path::Path,
) {
    use std::io::{Seek, Write as _};
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        out.extend_from_slice(&kv.bytes);
    }
    let mut offset: u64 = 0;
    for (name, ty, dims) in tensors {
        out.extend_from_slice(&sized(name));
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&ty.id().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset = (offset + payload_len(*ty, dims) as u64).div_ceil(32) * 32;
    }
    let data_start = out.len().div_ceil(32) * 32;
    let mut file = std::fs::File::create(path).expect("create fixture");
    file.write_all(&out).expect("write header");
    file.seek(std::io::SeekFrom::Start(data_start as u64))
        .expect("seek to payload");
    for (index, (_, ty, dims)) in tensors.iter().enumerate() {
        let padded = payload_len(*ty, dims) as u64;
        if let Some(bytes) = payloads.get(index).filter(|bytes| !bytes.is_empty()) {
            assert_eq!(bytes.len(), padded as usize, "payload must fill the tensor");
            file.write_all(bytes).expect("write payload");
            file.write_all(&vec![0u8; padded as usize - bytes.len()])
                .expect("pad payload");
        } else {
            file.seek(std::io::SeekFrom::Current(padded as i64))
                .expect("skip zero payload");
        }
    }
    file.set_len(data_start as u64 + offset)
        .expect("size fixture");
}

fn write_bolt_gguf(
    kvs: &[Kv],
    tensors: &[(String, crate::weights::GgufType, Vec<u64>)],
    path: &std::path::Path,
) {
    write_bolt_gguf_payload(kvs, tensors, &[], path);
}

/// The scalar metadata of the Bolt conversion, minus excluded keys.
fn bolt_base_kvs(exclude: &[&str]) -> Vec<Kv> {
    let mut heads = Vec::new();
    let mut pattern = Vec::new();
    for layer in 0..42 {
        heads.push(if layer < 24 { 2 } else { 0 });
        pattern.push(layer % 6 != 5);
    }
    let kvs = vec![
        kv_string("general.architecture", "gemma4"),
        kv_u32("gemma4.block_count", 42),
        kv_u32("gemma4.context_length", 65536),
        kv_u32("gemma4.embedding_length", 2560),
        kv_u32("gemma4.feed_forward_length", 10240),
        kv_u32("gemma4.attention.head_count", 8),
        kv_u32("gemma4.attention.key_length", 512),
        kv_u32("gemma4.attention.key_length_swa", 256),
        kv_u32("gemma4.attention.value_length", 512),
        kv_u32("gemma4.attention.value_length_swa", 256),
        kv_u32("gemma4.attention.sliding_window", 512),
        kv_u32("gemma4.attention.shared_kv_layers", 18),
        kv_u32("gemma4.rope.dimension_count", 512),
        kv_u32("gemma4.rope.dimension_count_swa", 256),
        kv_f32("gemma4.rope.freq_base", 1_000_000.0),
        kv_f32("gemma4.rope.freq_base_swa", 10_000.0),
        kv_f32("gemma4.final_logit_softcapping", 30.0),
        kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6),
        kv_bool("tokenizer.ggml.add_bos_token", true),
        kv_u32("tokenizer.ggml.bos_token_id", 2),
        kv_u32("tokenizer.ggml.eos_token_id", 1),
        kv_u32("tokenizer.ggml.unknown_token_id", 3),
        kv_i32_array("gemma4.attention.head_count_kv", &heads),
        kv_bool_array("gemma4.attention.sliding_window_pattern", &pattern),
        kv_i32_array("tokenizer.ggml.suppress_tokens", &[258883, 258882]),
    ];
    kvs.into_iter()
        .filter(|kv| !exclude.contains(&kv.key.as_str()))
        .collect()
}

/// The same metadata with the canonical all-twos KV head array: the
/// reference spelling carries the two heads on every layer instead of
/// zeros on the 18 shared layers.
fn bolt_base_kvs_canonical_heads(exclude: &[&str]) -> Vec<Kv> {
    let mut kvs = bolt_base_kvs(exclude);
    if let Some(at) = kvs
        .iter()
        .position(|kv| kv.key == "gemma4.attention.head_count_kv")
    {
        let heads = vec![2; 42];
        kvs[at] = kv_i32_array("gemma4.attention.head_count_kv", &heads);
    }
    kvs
}

/// The canonical llama.cpp gemma4 spelling of the Bolt inventory: the six
/// renamed per-layer tensors, no per-layer tables, and one merged
/// [PLE * layers, vocab] per_layer_token_embd in their place.
fn bolt_tensors_canonical() -> Vec<(String, crate::weights::GgufType, Vec<u64>)> {
    let mut out = Vec::new();
    for (name, ty, dims) in bolt_tensors() {
        if name.ends_with(".per_layer_embd.weight") {
            continue;
        }
        let name = match name.as_str() {
            "per_layer_model_projection.weight" => "per_layer_model_proj.weight".to_owned(),
            "per_layer_projection_norm.weight" => "per_layer_proj_norm.weight".to_owned(),
            other => {
                // Per-layer names are blk.N.suffix; split off N so the suffix
                // match cannot confuse N with the tensor suffix.
                match other
                    .strip_prefix("blk.")
                    .and_then(|rest| rest.split_once('.'))
                {
                    Some((index, suffix)) => match suffix {
                        "per_layer_gate.weight" => format!("blk.{index}.inp_gate.weight"),
                        "per_layer_proj.weight" => format!("blk.{index}.proj.weight"),
                        "post_per_layer_input_norm.weight" => {
                            format!("blk.{index}.post_norm.weight")
                        }
                        _ => other.to_owned(),
                    },
                    None => other.to_owned(),
                }
            }
        };
        out.push((name, ty, dims));
    }
    out.push((
        "per_layer_token_embd.weight".to_owned(),
        crate::weights::GgufType::Q80,
        vec![256 * 42, 262_144],
    ));
    out
}

/// The 707-tensor legacy Bolt inventory the reader expects.
fn bolt_tensors() -> Vec<(String, crate::weights::GgufType, Vec<u64>)> {
    use crate::weights::GgufType;
    let mut out = vec![
        (
            "token_embd.weight".to_owned(),
            GgufType::Q80,
            vec![2560, 262_144],
        ),
        ("output_norm.weight".to_owned(), GgufType::F32, vec![2560]),
        ("rope_freqs.weight".to_owned(), GgufType::F32, vec![256]),
        (
            "per_layer_model_projection.weight".to_owned(),
            GgufType::Q80,
            vec![2560, 10_752],
        ),
        (
            "per_layer_projection_norm.weight".to_owned(),
            GgufType::F32,
            vec![256],
        ),
    ];
    for layer in 0..42 {
        let local = layer % 6 != 5;
        let physical = layer < 24;
        let head: u64 = if local { 256 } else { 512 };
        let base = format!("blk.{layer}.");
        let norm =
            |suffix: &str, width: u64| (format!("{base}{suffix}"), GgufType::F32, vec![width]);
        let matrix = |suffix: &str, rows: u64, cols: u64| {
            (format!("{base}{suffix}"), GgufType::Q80, vec![rows, cols])
        };
        out.push(norm("attn_norm.weight", 2560));
        out.push(matrix("attn_q.weight", 2560, 8 * head));
        out.push(norm("attn_q_norm.weight", head));
        if physical {
            out.push(matrix("attn_k.weight", 2560, 2 * head));
            out.push(norm("attn_k_norm.weight", head));
            out.push(matrix("attn_v.weight", 2560, 2 * head));
        }
        out.push(matrix("attn_output.weight", 8 * head, 2560));
        out.push(norm("post_attention_norm.weight", 2560));
        out.push(norm("ffn_norm.weight", 2560));
        out.push(matrix("ffn_gate.weight", 2560, 10_240));
        out.push(matrix("ffn_up.weight", 2560, 10_240));
        out.push(matrix("ffn_down.weight", 10_240, 2560));
        out.push(norm("post_ffw_norm.weight", 2560));
        out.push(matrix("per_layer_gate.weight", 2560, 256));
        out.push(matrix("per_layer_proj.weight", 256, 2560));
        out.push(norm("post_per_layer_input_norm.weight", 2560));
        out.push(matrix("per_layer_embd.weight", 256, 262_144));
        out.push(norm("layer_output_scale.weight", 1));
    }
    out
}

fn open_bolt(
    kvs: &[Kv],
    tensors: &[(String, crate::weights::GgufType, Vec<u64>)],
) -> Result<(), crate::EngError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bolt.gguf");
    write_bolt_gguf(kvs, tensors, &path);
    open_bolt_q8(&path, None)
}

#[test]
fn accepts_a_synthetic_bolt_q8_gguf() {
    open_bolt(&bolt_base_kvs(&[]), &bolt_tensors()).expect("opens");
}

#[test]
fn rejects_a_bolt_gguf_without_shared_kv_layers() {
    let kvs: Vec<Kv> = bolt_base_kvs(&["gemma4.attention.shared_kv_layers"])
        .into_iter()
        .chain([kv_u32("gemma4.attention.shared_kv_layers", 0)])
        .collect();
    expect_rejection(
        open_bolt(&kvs, &bolt_tensors()),
        "gemma4.attention.shared_kv_layers",
    );
}

#[test]
fn rejects_bolt_kv_heads_on_a_shared_layer() {
    let mut heads = Vec::new();
    let mut pattern = Vec::new();
    for layer in 0..42 {
        heads.push(if layer < 24 { 2 } else { 0 });
        pattern.push(layer % 6 != 5);
    }
    heads[30] = 2;
    let kvs: Vec<Kv> = bolt_base_kvs(&["gemma4.attention.head_count_kv"])
        .into_iter()
        .chain([kv_i32_array("gemma4.attention.head_count_kv", &heads)])
        .collect();
    expect_rejection(
        open_bolt(&kvs, &bolt_tensors()),
        "head_count_kv is 2 at layer 30",
    );
}

/// Runs the converter's in-tree self-check so the battery exercises the
/// Python layout code too. Skips if the self-check is absent in a public
/// checkout, or if Python or NumPy is unavailable.
#[test]
fn the_bolt_converter_selfcheck_runs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let selfcheck = "scripts/convert_bolt_gguf_selfcheck.py";
    // Check for the script before probing NumPy so missing development
    // tooling cannot be hidden by a dependency skip.
    if !root
        .join(selfcheck)
        .try_exists()
        .expect("checking for the Bolt converter self-check")
    {
        assert!(
            std::env::var("FIFI_TEST_REQUIRE_DATA").as_deref() != Ok("1"),
            "FIFI_TEST_REQUIRE_DATA=1 but the Bolt converter self-check is absent"
        );
        let publisher_hook = root.join(".githooks/publisher-integration");
        assert!(
            !publisher_hook
                .try_exists()
                .expect("checking for the publisher integration hook"),
            "the Bolt converter self-check is absent in the development checkout"
        );
        eprintln!("skipping: no Bolt converter self-check in the public checkout");
        return;
    }
    // numpy is the self-check's only third-party need; probe it so a host
    // without the module skips instead of failing for a missing tool.
    let probe = std::process::Command::new("python3")
        .args(["-c", "import numpy"])
        .output();
    match probe {
        Ok(check) if check.status.success() => {}
        Ok(_) => {
            println!("SKIPPED: python3 present without numpy");
            return;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("SKIPPED: python3 not present");
            return;
        }
        Err(error) => panic!("python3 failed to start: {error}"),
    }
    let output = std::process::Command::new("python3")
        .arg(selfcheck)
        .current_dir(&root)
        .output();
    match output {
        Ok(run) if run.status.success() => {}
        Ok(run) => panic!(
            "convert_bolt_gguf_selfcheck failed:\n{}\n{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("SKIPPED: python3 not present");
        }
        Err(error) => panic!("python3 failed to start: {error}"),
    }
}
#[test]
fn rejects_a_bolt_gguf_whose_shared_kv_layers_disagree_with_the_spec() {
    // The resident's sharing map comes from the spec, so a file declaring a
    // different map (19 shared layers, one off the spec's 18) must be
    // refused: the shared layers' K/V sources are not negotiable per file.
    let kvs: Vec<Kv> = bolt_base_kvs(&["gemma4.attention.shared_kv_layers"])
        .into_iter()
        .chain([kv_u32("gemma4.attention.shared_kv_layers", 19)])
        .collect();
    expect_rejection(open_bolt(&kvs, &bolt_tensors()), "shared_kv_layers");
}

#[test]
fn accepts_a_canonical_bolt_q8_gguf() {
    open_bolt(
        &bolt_base_kvs_canonical_heads(&[]),
        &bolt_tensors_canonical(),
    )
    .expect("opens");
}

/// The tensor spelling and the KV head spelling are independent: a stock
/// reference-format file whose converter still writes legacy zeros opens
/// too, because the sharing map derives from shared_kv_layers either way.
#[test]
fn accepts_canonical_tensors_with_legacy_kv_heads() {
    open_bolt(&bolt_base_kvs(&[]), &bolt_tensors_canonical()).expect("opens");
}

/// Both spellings describe one model: the KV head arrays differ (2 versus
/// 0 on the shared layers) yet both files validate, and the sharing map the
/// resident builds - physical layer count and each shared layer's source,
/// from kv_shared_layers via the descriptor - is the same constant for both.
#[test]
fn bolt_both_spellings_share_one_kv_map() {
    use crate::family::spec::NEMOTRON_BOLT_E4B;

    let legacy_heads: Vec<i32> = (0..42)
        .map(|layer| if layer < 24 { 2 } else { 0 })
        .collect();
    let canonical_heads = [2; 42];
    assert_ne!(legacy_heads[24], canonical_heads[24]);

    open_bolt(&bolt_base_kvs(&[]), &bolt_tensors()).expect("the legacy spelling opens");
    open_bolt(
        &bolt_base_kvs_canonical_heads(&[]),
        &bolt_tensors_canonical(),
    )
    .expect("the canonical spelling opens");

    // One sharing map serves both: it never reads the KV head array.
    assert_eq!(NEMOTRON_BOLT_E4B.first_shared_layer(), 24);
    for layer in 24..42 {
        let source = NEMOTRON_BOLT_E4B
            .shared_kv_source(layer)
            .expect("every shared layer has a source");
        assert!(source < 24, "a shared source must be physical");
    }
}

#[test]
fn the_merged_table_and_per_layer_tables_carry_the_same_weights() {
    use crate::family::gemma4::gguf::ple_rows_from_merged;
    use crate::weights::{GgufLimits, GgufType, GgufWeights};

    let ple = 256;
    let layers = 42;
    let vocab = 64;
    let row = |layer: usize, token: usize| -> Vec<u8> {
        (0..(ple / 32) * 34)
            .map(|byte| (layer * 31 + token * 17 + byte) as u8)
            .collect()
    };

    // Legacy spelling: one [PLE, VOCAB] table per layer, rows back to back.
    let mut legacy_tensors = Vec::new();
    let mut legacy_payloads = Vec::new();
    for layer in 0..layers {
        let mut bytes = Vec::new();
        for token in 0..vocab {
            bytes.extend_from_slice(&row(layer, token));
        }
        legacy_tensors.push((
            format!("blk.{layer}.per_layer_embd.weight"),
            GgufType::Q80,
            vec![ple as u64, vocab as u64],
        ));
        legacy_payloads.push(bytes);
    }

    // Canonical spelling: the merged table in the reference layout — each
    // row is one token with every layer's PLE block contiguous.
    let mut merged_payload = Vec::new();
    for token in 0..vocab {
        for layer in 0..layers {
            merged_payload.extend_from_slice(&row(layer, token));
        }
    }
    let canonical_tensors = vec![(
        "per_layer_token_embd.weight".to_owned(),
        GgufType::Q80,
        vec![(ple * layers) as u64, vocab as u64],
    )];

    let limits = GgufLimits {
        max_header_bytes: 1 << 20,
        max_tensor_bytes: 1 << 20,
    };
    let dir = std::env::temp_dir();
    // fixed names would collide across parallel runs: pid + test tag keep
    // every invocation on its own pair of scratch files.
    let unique = format!("bolt-tables-{}-{}", std::process::id(), "gather");
    let legacy_path = dir.join(format!("{unique}-legacy.gguf"));
    let canonical_path = dir.join(format!("{unique}-merged.gguf"));
    write_bolt_gguf_payload(&[], &legacy_tensors, &legacy_payloads, &legacy_path);
    write_bolt_gguf_payload(&[], &canonical_tensors, &[merged_payload], &canonical_path);

    let legacy = GgufWeights::open(&legacy_path, limits, None).expect("legacy opens");
    let canonical = GgufWeights::open(&canonical_path, limits, None).expect("canonical opens");
    let merged = canonical
        .read_tensor("per_layer_token_embd.weight")
        .expect("merged table reads");
    for layer in [0, 5, 24, 41] {
        let from_merged = ple_rows_from_merged(&merged.1, layer, ple, layers, vocab);
        let (_, from_legacy) = legacy
            .read_tensor(&format!("blk.{layer}.per_layer_embd.weight"))
            .expect("legacy table reads");
        let source_of_truth: Vec<u8> = (0..vocab).flat_map(|t| row(layer, t)).collect();
        assert_eq!(
            from_legacy, source_of_truth,
            "layer {layer} via its own table"
        );
        assert_eq!(
            from_merged, source_of_truth,
            "layer {layer} via the merged table"
        );
    }

    // One sharing map serves both spellings: the file's declared map is
    // pinned to the spec by check_int_scalars (see the wrong-value test),
    // never free to diverge per spelling.

    std::fs::remove_file(&legacy_path).ok();
    std::fs::remove_file(&canonical_path).ok();
}

#[test]
fn rejects_a_bolt_gguf_with_both_spelling_tables() {
    let mut tensors = bolt_tensors();
    tensors.push((
        "per_layer_token_embd.weight".to_owned(),
        crate::weights::GgufType::Q80,
        vec![256 * 42, 262_144],
    ));
    expect_rejection(open_bolt(&bolt_base_kvs(&[]), &tensors), "are both present");
}

#[test]
fn rejects_a_missing_attn_v_on_a_physical_bolt_layer() {
    let tensors: Vec<_> = bolt_tensors()
        .into_iter()
        .filter(|(name, _, _)| name != "blk.2.attn_v.weight")
        .collect();
    expect_rejection(
        open_bolt(&bolt_base_kvs(&[]), &tensors),
        "missing tensor blk.2.attn_v.weight",
    );
}

#[test]
fn rejects_an_attn_k_on_a_shared_bolt_layer() {
    let mut tensors = bolt_tensors();
    tensors.push((
        "blk.30.attn_k.weight".to_owned(),
        crate::weights::GgufType::Q80,
        vec![2560, 512],
    ));
    expect_rejection(
        open_bolt(&bolt_base_kvs(&[]), &tensors),
        "unexpected tensor blk.30.attn_k.weight",
    );
}

#[test]
fn rejects_a_wrong_full_layer_query_width() {
    let mut tensors = bolt_tensors();
    for (name, _, dims) in &mut tensors {
        if name == "blk.5.attn_q.weight" {
            *dims = vec![2560, 2048];
        }
    }
    expect_rejection(
        open_bolt(&bolt_base_kvs(&[]), &tensors),
        "tensor blk.5.attn_q.weight has dims [2560, 2048], expected [2560, 4096]",
    );
}

#[test]
fn rejects_a_bolt_gguf_with_a_wrong_sliding_window() {
    let kvs: Vec<Kv> = bolt_base_kvs(&["gemma4.attention.sliding_window"])
        .into_iter()
        .chain([kv_u32("gemma4.attention.sliding_window", 1024)])
        .collect();
    expect_rejection(
        open_bolt(&kvs, &bolt_tensors()),
        "gemma4.attention.sliding_window",
    );
}

#[test]
fn rejects_a_bolt_gguf_with_a_wrong_sliding_window_pattern() {
    let mut pattern = Vec::new();
    for layer in 0..42 {
        pattern.push(layer % 6 != 5);
    }
    pattern[0] = false;
    let kvs: Vec<Kv> = bolt_base_kvs(&["gemma4.attention.sliding_window_pattern"])
        .into_iter()
        .chain([kv_bool_array(
            "gemma4.attention.sliding_window_pattern",
            &pattern,
        )])
        .collect();
    expect_rejection(
        open_bolt(&kvs, &bolt_tensors()),
        "sliding_window_pattern is 0 at layer 0",
    );
}

#[test]
fn rejects_a_bolt_gguf_with_wrong_suppress_tokens() {
    let kvs: Vec<Kv> = bolt_base_kvs(&["tokenizer.ggml.suppress_tokens"])
        .into_iter()
        .chain([kv_i32_array("tokenizer.ggml.suppress_tokens", &[1, 2])])
        .collect();
    expect_rejection(open_bolt(&kvs, &bolt_tensors()), "suppress_tokens");
}

#[test]
fn rejects_a_bolt_gguf_with_a_bad_header_hash() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bolt.gguf");
    write_bolt_gguf(&bolt_base_kvs(&[]), &bolt_tensors(), &path);
    let wrong = open_bolt_q8(&path, Some(&"0".repeat(64)));
    expect_rejection(wrong, "header");
}

#[test]
fn rejects_a_missing_per_layer_table_on_a_shared_layer() {
    let tensors: Vec<_> = bolt_tensors()
        .into_iter()
        .filter(|(name, _, _)| name != "blk.30.per_layer_embd.weight")
        .collect();
    expect_rejection(
        open_bolt(&bolt_base_kvs(&[]), &tensors),
        "missing tensor blk.30.per_layer_embd.weight",
    );
}

#[test]
fn rejects_a_wrong_per_layer_gate_width() {
    let mut tensors = bolt_tensors();
    for (name, _, dims) in &mut tensors {
        if name == "blk.7.per_layer_gate.weight" {
            *dims = vec![2560, 512];
        }
    }
    expect_rejection(
        open_bolt(&bolt_base_kvs(&[]), &tensors),
        "tensor blk.7.per_layer_gate.weight has dims [2560, 512], expected [2560, 256]",
    );
}

#[test]
fn accepts_gguf_rms_epsilon_in_its_stored_float_precision() {
    const KEY: &str = "gemma4.attention.layer_norm_rms_epsilon";
    for entry in [kv_f32(KEY, 1.0e-6), kv_f64(KEY, 1.0e-6)] {
        let kvs: Vec<Kv> = base_kvs(&[KEY]).into_iter().chain([entry]).collect();
        // These small fixtures omit layer tensors. A valid epsilon reaches the inventory check.
        expect_rejection(
            open(&build_gguf(&kvs, &globals())),
            "missing tensor blk.0.attn_k.weight",
        );
    }
}
