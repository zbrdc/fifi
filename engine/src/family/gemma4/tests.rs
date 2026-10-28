use super::{check_humanizer_config, open_humanizer_q8};

const HUMANIZER_CONFIG: &[u8] = include_bytes!("../../../testdata/gemma4/humanizer-config.json");

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
