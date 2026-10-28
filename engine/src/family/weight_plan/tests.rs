use std::io::Cursor;

use serde_json::{Value, json};

use crate::{
    EngError,
    family::{
        KeyTier, WeightPlan,
        quantization::{ModuleQuantization, QuantizationMap},
        spec::{Act, AttentionKind, FamilySpec, Router},
    },
    kernels::{QuantizationMode, QuantizationSpec},
    weights::ShardHeader,
};

static SPEC: FamilySpec = FamilySpec {
    name: "plan-test",
    layers: 2,
    hidden: 64,
    heads_q: 8,
    heads_kv: 2,
    head_dim: 32,
    rope_theta: 10000.0,
    rms_norm_eps: 1e-6,
    model_types: &["test_model"],
    vocab: 96,
    feed_forward: None,
    experts: 4,
    top_k: 2,
    moe_inter: 16,
    shared_inter: 16,
    quant_bits: 4,
    quant_group: 64,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &[AttentionKind::Full, AttentionKind::Full],
    linear: None,
    hyper: None,
};

const INT4: ModuleQuantization = ModuleQuantization::Quantized(QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 4,
    group_size: 64,
});

const INT8: ModuleQuantization = ModuleQuantization::Quantized(QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 8,
    group_size: 64,
});

const DENSE: ModuleQuantization = ModuleQuantization::Unquantized;

const EXPECTED: [(&str, KeyTier, ModuleQuantization); 12] = [
    ("language_model.lm_head.weight", KeyTier::Resident, INT4),
    (
        "language_model.model.embed_tokens.weight",
        KeyTier::Resident,
        INT4,
    ),
    (
        "language_model.model.layers.0.mlp.gate.weight",
        KeyTier::Resident,
        INT8,
    ),
    (
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
        KeyTier::ExpertStacked,
        INT4,
    ),
    (
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
        KeyTier::ExpertStacked,
        INT4,
    ),
    (
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        KeyTier::ExpertStacked,
        INT4,
    ),
    (
        "language_model.model.layers.0.self_attn.o_proj.scales",
        KeyTier::Resident,
        INT4,
    ),
    (
        "language_model.model.layers.0.self_attn.q_proj.weight",
        KeyTier::Resident,
        INT4,
    ),
    (
        "language_model.model.layers.1.input_layernorm.weight",
        KeyTier::Resident,
        DENSE,
    ),
    (
        "language_model.model.layers.1.mlp.gate.biases",
        KeyTier::Resident,
        INT8,
    ),
    (
        "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight",
        KeyTier::ExpertStacked,
        INT4,
    ),
    ("language_model.model.norm.weight", KeyTier::Resident, DENSE),
];

fn width(dtype: &str) -> usize {
    match dtype {
        "BF16" => 2,
        "F32" | "U32" => 4,
        other => panic!("unsupported fixture dtype {other}"),
    }
}

fn shard(parts: &[(&str, &str, &[usize])]) -> ShardHeader {
    let mut map = serde_json::Map::new();
    let mut start = 0usize;
    for &(key, dtype, shape) in parts {
        let bytes = shape.iter().product::<usize>() * width(dtype);
        map.insert(
            key.into(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": (start, start + bytes)}),
        );
        start += bytes;
    }
    let json = serde_json::to_string(&Value::Object(map)).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    bytes.resize(8 + json.len() + start, 0);
    ShardHeader::read(
        &mut Cursor::new(bytes.as_slice()),
        bytes.len() as u64,
        65536,
    )
    .unwrap()
}

fn shards() -> [ShardHeader; 2] {
    [
        shard(&[
            (
                "language_model.model.embed_tokens.weight",
                "BF16",
                &[96, 64],
            ),
            ("language_model.lm_head.weight", "U32", &[96, 8]),
            ("language_model.model.norm.weight", "BF16", &[64]),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "U32",
                &[64, 8],
            ),
            (
                "language_model.model.layers.0.self_attn.o_proj.scales",
                "BF16",
                &[64, 1],
            ),
            (
                "language_model.model.layers.0.mlp.gate.weight",
                "U32",
                &[64, 16],
            ),
        ]),
        shard(&[
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                "U32",
                &[256, 8],
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
                "BF16",
                &[256, 1],
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
                "BF16",
                &[256, 1],
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight",
                "U32",
                &[256, 8],
            ),
            (
                "language_model.model.layers.1.mlp.gate.biases",
                "BF16",
                &[64, 1],
            ),
            (
                "language_model.model.layers.1.input_layernorm.weight",
                "BF16",
                &[64],
            ),
        ]),
    ]
}

fn base_quantization() -> Value {
    json!({
        "mode": "affine",
        "bits": 4,
        "group_size": 64,
        "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
        "language_model.model.layers.1.mlp.gate": {"bits": 8, "group_size": 64},
    })
}

fn build_plan(quantization: Value) -> Result<WeightPlan, EngError> {
    let config = json!({ "quantization": quantization }).to_string();
    let quant = QuantizationMap::from_json(config.as_bytes()).unwrap();
    WeightPlan::build(&SPEC, &quant, &shards())
}

fn checkpoint_message(error: EngError) -> String {
    match error {
        EngError::Checkpoint(message) => message,
        other => panic!("expected a checkpoint error, got {other:?}"),
    }
}

#[test]
fn assigns_the_expected_tier_for_every_key() {
    let plan = build_plan(base_quantization()).unwrap();
    let mut keys: Vec<&str> = plan.iter().map(|(key, _)| key).collect();
    keys.sort_unstable();
    let mut expected: Vec<&str> = EXPECTED.iter().map(|row| row.0).collect();
    expected.sort_unstable();
    assert_eq!(keys, expected);
    for (key, tier, _) in EXPECTED {
        assert_eq!(plan.tier(key), Some(tier), "{key}");
    }
}

#[test]
fn resolves_the_expected_quantization_for_every_key() {
    let plan = build_plan(base_quantization()).unwrap();
    for (key, _, quantization) in EXPECTED {
        assert_eq!(plan.quantization(key), Some(quantization), "{key}");
    }
}

#[test]
fn rejects_an_unmapped_override_key() {
    let mut quantization = base_quantization();
    quantization["language_model.model.layers.7.mlp.gate"] = json!({"bits": 8, "group_size": 64});
    let message = checkpoint_message(build_plan(quantization).unwrap_err());
    assert!(message.contains("override key"), "{message}");
}

#[test]
fn rejects_a_packed_key_with_no_quantization() {
    let mut quantization = base_quantization();
    quantization["language_model.model.layers.0.self_attn.q_proj"] = json!(false);
    let message = checkpoint_message(build_plan(quantization).unwrap_err());
    assert!(message.contains("has no quantization"), "{message}");
}

#[test]
fn dense_keys_ignore_the_default_and_incidental_overrides() {
    let dense = [
        "language_model.model.norm.weight",
        "language_model.model.layers.1.input_layernorm.weight",
    ];
    let mut quantization = base_quantization();
    quantization["language_model.model.norm"] = json!(false);
    quantization["language_model.model.layers.1.input_layernorm"] =
        json!({"bits": 8, "group_size": 64});
    let plan = build_plan(quantization).unwrap();
    for key in dense {
        assert_eq!(plan.quantization(key), Some(DENSE), "{key}");
    }
    assert_eq!(
        plan.quantization("language_model.model.embed_tokens.weight"),
        Some(INT4)
    );
}

#[test]
fn rejects_a_key_present_in_two_headers() {
    let key = "language_model.model.norm.weight";
    let [first, _] = shards();
    let second = shard(&[(key, "BF16", &[64])]);
    let quant = QuantizationMap::from_json(
        json!({ "quantization": base_quantization() })
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    let message =
        checkpoint_message(WeightPlan::build(&SPEC, &quant, &[first, second]).unwrap_err());
    assert_eq!(message, format!("duplicate weight key {key:?}"));
}
