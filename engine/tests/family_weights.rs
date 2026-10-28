use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use inference_engine::{
    EngError,
    family::{
        Act, AttentionKind, FamilySpec, FeedForwardWeights, LayerWeights, LinearAttention,
        MixerWeights, ProjectionWeights, Router, WeightPlan,
        quantization::{ModuleQuantization, QuantizationMap},
    },
    kernels::{QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{IndexedWeights, ResidentWeights, ShardHeader, Tier, WeightSource},
};

const MANIFEST: &str = include_str!("fixtures/family-fixtures.json");
const FULL: &[u8] = include_bytes!("fixtures/family-full.safetensors");
const FULL_INDEX: &[u8] = include_bytes!("fixtures/family-full.index.json");
const MISSING: &[u8] = include_bytes!("fixtures/family-missing-router.safetensors");
const MISSING_INDEX: &[u8] = include_bytes!("fixtures/family-missing-router.index.json");
const WRONG_RANK: &[u8] = include_bytes!("fixtures/family-wrong-rank.safetensors");
const WRONG_RANK_INDEX: &[u8] = include_bytes!("fixtures/family-wrong-rank.index.json");
const BAD_PAYLOAD: &[u8] = include_bytes!("fixtures/family-bad-payload-dtype.safetensors");
const BAD_PAYLOAD_INDEX: &[u8] = include_bytes!("fixtures/family-bad-payload-dtype.index.json");
const BAD_EXPERT: &[u8] = include_bytes!("fixtures/family-bad-expert-dtype.safetensors");
const BAD_EXPERT_INDEX: &[u8] = include_bytes!("fixtures/family-bad-expert-dtype.index.json");
const BAD_SIDECAR: &[u8] = include_bytes!("fixtures/family-bad-sidecar-dtype.safetensors");
const BAD_SIDECAR_INDEX: &[u8] = include_bytes!("fixtures/family-bad-sidecar-dtype.index.json");
const BAD_DENSE: &[u8] = include_bytes!("fixtures/family-bad-dense-dtype.safetensors");
const BAD_DENSE_INDEX: &[u8] = include_bytes!("fixtures/family-bad-dense-dtype.index.json");
const RATIO_BYTES: &[u8] = include_bytes!("fixtures/family-delta-ratio.safetensors");
const RATIO_INDEX: &[u8] = include_bytes!("fixtures/family-delta-ratio.index.json");
const RATIO_MANIFEST: &str = include_str!("fixtures/family-delta-ratio.json");
const RATIO_SHARD: &str = "family-delta-ratio.safetensors";
const FULL_SHARD: &str = "family-full.safetensors";
const MISSING_SHARD: &str = "family-missing-router.safetensors";
const MISSING_KEY: &str = "language_model.model.layers.0.mlp.gate.weight";

// Hand-derived expectations for the fixture geometry H=128, Q=4, KV=2, D=16,
// E=4, I=64, S=64, AUX=64 and the quantization below. They do not call the
// implementation under test.
const CONFIG: &str = r#"{
  "quantization": {
    "mode": "affine",
    "bits": 4,
    "group_size": 32,
    "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
    "language_model.model.layers.0.mlp.shared_expert_gate": {"bits": 8, "group_size": 64},
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj": {"bits": 4, "group_size": 128},
    "language_model.model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32}
  }
}"#;

const Q4_32: QuantizationSpec = QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 4,
    group_size: 32,
};
const Q4_128: QuantizationSpec = QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 4,
    group_size: 128,
};
const Q8_32: QuantizationSpec = QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 8,
    group_size: 32,
};
const Q8_64: QuantizationSpec = QuantizationSpec {
    mode: QuantizationMode::Affine,
    bits: 8,
    group_size: 64,
};

const TINY_LAYERS: [AttentionKind; 2] = [AttentionKind::Full, AttentionKind::DeltaNet];

const TINY: FamilySpec = FamilySpec {
    name: "tiny_family",
    layers: 2,
    hidden: 128,
    heads_q: 4,
    heads_kv: 2,
    head_dim: 16,
    rope_theta: 10_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["test_model"],
    vocab: 32,
    feed_forward: None,
    experts: 4,
    top_k: 2,
    moe_inter: 64,
    shared_inter: 64,
    quant_bits: 4,
    quant_group: 32,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &TINY_LAYERS,
    // Key width 128 and value width 256 reproduce the fixture rows: 512 fused, 256 gate, 64 heads.
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 8,
        value_heads: 64,
        value_head_dim: 4,
        conv_kernel: 4,
    }),
    hyper: None,
};

fn load_index(shard: &str, index: &'static [u8], data: &'static [u8]) -> IndexedWeights {
    let shards = BTreeMap::from([(shard.to_string(), Arc::from(data))]);
    IndexedWeights::from_shards(index, shards).expect("fixture index loads")
}

fn header(data: &'static [u8]) -> ShardHeader {
    ShardHeader::read(&mut Cursor::new(data), data.len() as u64, 1 << 20)
        .expect("fixture header reads")
}

fn plan_for(spec: &FamilySpec, config: &[u8], headers: &[ShardHeader]) -> WeightPlan {
    let quantization = QuantizationMap::from_json(config).expect("configuration parses");
    WeightPlan::build(spec, &quantization, headers).expect("weight plan builds")
}

fn plan(config: &[u8], headers: &[ShardHeader]) -> WeightPlan {
    plan_for(&TINY, config, headers)
}

fn full_plan() -> WeightPlan {
    plan(CONFIG.as_bytes(), &[header(FULL)])
}

fn resolve(
    layer: u32,
    index: &IndexedWeights,
    spec: &FamilySpec,
    plan: &WeightPlan,
) -> LayerWeights {
    match LayerWeights::resolve(index, spec, layer, plan) {
        Ok(weights) => weights,
        Err(err) => panic!("resolve({layer}) failed: {err:?}"),
    }
}

fn fixture(key: &str) -> (Vec<u32>, u64) {
    let doc: serde_json::Value = serde_json::from_str(MANIFEST).expect("manifest parses");
    manifest_entry(&doc["files"][FULL_SHARD]["tensors"][key], key)
}

fn ratio_fixture(key: &str) -> (Vec<u32>, u64) {
    let doc: serde_json::Value = serde_json::from_str(RATIO_MANIFEST).expect("manifest parses");
    manifest_entry(&doc["file"]["tensors"][key], key)
}

fn manifest_entry(entry: &serde_json::Value, key: &str) -> (Vec<u32>, u64) {
    assert!(!entry.is_null(), "manifest lacks {key}");
    let shape = entry["shape"]
        .as_array()
        .expect("shape array")
        .iter()
        .map(|dim| dim.as_u64().expect("dim") as u32)
        .collect();
    (shape, entry["fill"].as_u64().expect("fill"))
}

fn assert_fill(tensor: &Tensor, fill: u64, key: &str) {
    let bytes = tensor
        .host_bytes()
        .unwrap_or_else(|| panic!("{key} has no host bytes"));
    let filled = match tensor.dtype() {
        Dtype::U32 => {
            let want = (fill as u32).to_le_bytes();
            bytes.chunks(4).all(|chunk| chunk == want.as_slice())
        }
        Dtype::Bf16 => {
            let encoded = (fill as f32).to_le_bytes();
            let want = &encoded[2..4];
            bytes.chunks(2).all(|chunk| chunk == want)
        }
        _ => panic!("{key} has an unexpected dtype"),
    };
    assert!(filled, "{key} does not hold fill {fill}");
}

fn assert_role(tensor: &Tensor, key: &str) {
    assert_role_of(tensor, key, fixture(key));
}

fn assert_ratio_role(tensor: &Tensor, key: &str) {
    assert_role_of(tensor, key, ratio_fixture(key));
}

fn assert_role_of(tensor: &Tensor, key: &str, (shape, fill): (Vec<u32>, u64)) {
    assert_eq!(tensor.shape(), shape.as_slice(), "shape of {key}");
    assert_fill(tensor, fill, key);
}

fn assert_projection(
    projection: &ProjectionWeights,
    stem: &str,
    quantization: QuantizationSpec,
    weight: &[u32],
    sidecar: &[u32],
) {
    check_projection(
        projection,
        stem,
        quantization,
        (weight, sidecar),
        assert_role,
    );
}

fn check_projection(
    projection: &ProjectionWeights,
    stem: &str,
    quantization: QuantizationSpec,
    (weight, sidecar): (&[u32], &[u32]),
    role: fn(&Tensor, &str),
) {
    let packed = projection.packed();
    assert_eq!(packed.quantization(), quantization, "mode of {stem}");
    assert_eq!(packed.payload().shape(), weight, "weight of {stem}");
    assert_eq!(packed.scales().shape(), sidecar, "scales of {stem}");
    let biases = packed.biases().expect("affine projections keep biases");
    assert_eq!(biases.shape(), sidecar, "biases of {stem}");
    role(packed.payload(), &format!("{stem}.weight"));
    role(packed.scales(), &format!("{stem}.scales"));
    role(biases, &format!("{stem}.biases"));
}

#[test]
fn full_layer_resolves_all_roles() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = full_plan();
    let layer = resolve(0, &index, &TINY, &plan);
    let FeedForwardWeights::Moe {
        router,
        experts,
        shared,
    } = &layer.feed_forward
    else {
        panic!("layer 0 must use MoE")
    };
    let stem = "language_model.model.layers.0.";
    assert_role(&layer.input_norm, &format!("{stem}input_layernorm.weight"));
    assert_role(
        &layer.post_attention_norm,
        &format!("{stem}post_attention_layernorm.weight"),
    );
    assert_projection(router, &format!("{stem}mlp.gate"), Q8_64, &[4, 32], &[4, 2]);
    assert_projection(
        &shared.shared_gate,
        &format!("{stem}mlp.shared_expert_gate"),
        Q8_64,
        &[1, 32],
        &[1, 2],
    );
    assert_projection(
        &experts.gate,
        &format!("{stem}mlp.switch_mlp.gate_proj"),
        Q4_128,
        &[4, 64, 16],
        &[4, 64, 1],
    );
    assert_projection(
        &experts.up,
        &format!("{stem}mlp.switch_mlp.up_proj"),
        Q4_32,
        &[4, 64, 16],
        &[4, 64, 4],
    );
    assert_projection(
        &experts.down,
        &format!("{stem}mlp.switch_mlp.down_proj"),
        Q4_32,
        &[4, 128, 8],
        &[4, 128, 2],
    );
    assert_projection(
        &shared.gate,
        &format!("{stem}mlp.shared_expert.gate_proj"),
        Q4_32,
        &[64, 16],
        &[64, 4],
    );
    assert_projection(
        &shared.up,
        &format!("{stem}mlp.shared_expert.up_proj"),
        Q4_32,
        &[64, 16],
        &[64, 4],
    );
    assert_projection(
        &shared.down,
        &format!("{stem}mlp.shared_expert.down_proj"),
        Q4_32,
        &[128, 8],
        &[128, 2],
    );
    let MixerWeights::Attention(attn) = &layer.mixer else {
        panic!("layer 0 must use attention")
    };
    assert_projection(
        &attn.q,
        &format!("{stem}self_attn.q_proj"),
        Q8_32,
        &[128, 32],
        &[128, 4],
    );
    assert_projection(
        &attn.k,
        &format!("{stem}self_attn.k_proj"),
        Q4_32,
        &[32, 16],
        &[32, 4],
    );
    assert_projection(
        &attn.v,
        &format!("{stem}self_attn.v_proj"),
        Q4_32,
        &[32, 16],
        &[32, 4],
    );
    assert_projection(
        &attn.o,
        &format!("{stem}self_attn.o_proj"),
        Q4_32,
        &[128, 8],
        &[128, 2],
    );
    assert_role(&attn.q_norm, &format!("{stem}self_attn.q_norm.weight"));
    assert_role(&attn.k_norm, &format!("{stem}self_attn.k_norm.weight"));
}

#[test]
fn delta_layer_resolves_all_roles() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = full_plan();
    let layer = resolve(1, &index, &TINY, &plan);
    let FeedForwardWeights::Moe {
        router,
        experts,
        shared,
    } = &layer.feed_forward
    else {
        panic!("layer 1 must use MoE")
    };
    let stem = "language_model.model.layers.1.";
    assert_role(&layer.input_norm, &format!("{stem}input_layernorm.weight"));
    assert_role(
        &layer.post_attention_norm,
        &format!("{stem}post_attention_layernorm.weight"),
    );
    assert_projection(router, &format!("{stem}mlp.gate"), Q4_32, &[4, 16], &[4, 4]);
    assert_projection(
        &shared.shared_gate,
        &format!("{stem}mlp.shared_expert_gate"),
        Q4_32,
        &[1, 16],
        &[1, 4],
    );
    assert_projection(
        &experts.gate,
        &format!("{stem}mlp.switch_mlp.gate_proj"),
        Q4_32,
        &[4, 64, 16],
        &[4, 64, 4],
    );
    assert_projection(
        &experts.down,
        &format!("{stem}mlp.switch_mlp.down_proj"),
        Q4_32,
        &[4, 128, 8],
        &[4, 128, 2],
    );
    let MixerWeights::DeltaNet(delta) = &layer.mixer else {
        panic!("layer 1 must use delta net")
    };
    for (role, name, rows) in [
        (&delta.in_proj_qkv, "in_proj_qkv", 512),
        (&delta.in_proj_z, "in_proj_z", 256),
        (&delta.in_proj_a, "in_proj_a", 64),
        (&delta.in_proj_b, "in_proj_b", 64),
    ] {
        assert_projection(
            role,
            &format!("{stem}linear_attn.{name}"),
            Q4_32,
            &[rows, 16],
            &[rows, 4],
        );
    }
    assert_projection(
        &delta.out_proj,
        &format!("{stem}linear_attn.out_proj"),
        Q4_32,
        &[128, 32],
        &[128, 8],
    );
    assert_role(&delta.conv1d, &format!("{stem}linear_attn.conv1d.weight"));
    assert_role(&delta.a_log, &format!("{stem}linear_attn.A_log"));
    assert_role(&delta.dt_bias, &format!("{stem}linear_attn.dt_bias"));
    assert_role(&delta.norm, &format!("{stem}linear_attn.norm.weight"));
}

#[test]
fn missing_role_fails_checkpoint() {
    let index = load_index(MISSING_SHARD, MISSING_INDEX, MISSING);
    let plan = full_plan();
    let err = match LayerWeights::resolve(&index, &TINY, 0, &plan) {
        Ok(_) => panic!("resolve must fail without the router weight"),
        Err(err) => err,
    };
    match err {
        EngError::Checkpoint(message) => {
            assert!(message.contains(MISSING_KEY), "message: {message}")
        }
        err => panic!("expected Checkpoint, got {err:?}"),
    }
}

#[test]
fn wrong_width_fails_checkpoint() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = full_plan();
    let wider = FamilySpec {
        hidden: 256,
        ..TINY
    };
    let err = match LayerWeights::resolve(&index, &wider, 0, &plan) {
        Ok(_) => panic!("resolve must fail on a width mismatch"),
        Err(err) => err,
    };
    assert!(
        matches!(err, EngError::Checkpoint(_)),
        "expected Checkpoint"
    );
}

#[test]
fn wrong_rank_fails_checkpoint() {
    let index = load_index(
        "family-wrong-rank.safetensors",
        WRONG_RANK_INDEX,
        WRONG_RANK,
    );
    let plan = full_plan();
    let err = match LayerWeights::resolve(&index, &TINY, 0, &plan) {
        Ok(_) => panic!("resolve must fail on a rank mismatch"),
        Err(err) => err,
    };
    match err {
        EngError::Checkpoint(message) => assert!(
            message.contains("language_model.model.layers.0.mlp.gate.weight"),
            "message: {message}"
        ),
        err => panic!("expected Checkpoint, got {err:?}"),
    }
}

#[test]
fn f32_packed_payload_fails_checkpoint() {
    let index = load_index(
        "family-bad-payload-dtype.safetensors",
        BAD_PAYLOAD_INDEX,
        BAD_PAYLOAD,
    );
    let plan = full_plan();
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("mlp.gate.weight") && message.contains("payload dtype must be U32"),
        "message: {message}"
    );
}

#[test]
fn f32_expert_payload_fails_checkpoint() {
    let index = load_index(
        "family-bad-expert-dtype.safetensors",
        BAD_EXPERT_INDEX,
        BAD_EXPERT,
    );
    let plan = full_plan();
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("mlp.switch_mlp.gate_proj.weight")
            && message.contains("payload dtype must be U32"),
        "message: {message}"
    );
}

#[test]
fn mismatched_sidecar_dtypes_fail_checkpoint() {
    let index = load_index(
        "family-bad-sidecar-dtype.safetensors",
        BAD_SIDECAR_INDEX,
        BAD_SIDECAR,
    );
    let plan = full_plan();
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("mlp.gate.weight")
            && message.contains("affine biases dtype must match scales"),
        "message: {message}"
    );
}

#[test]
fn non_floating_dense_role_fails_checkpoint() {
    let index = load_index(
        "family-bad-dense-dtype.safetensors",
        BAD_DENSE_INDEX,
        BAD_DENSE,
    );
    let plan = full_plan();
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("input_layernorm.weight") && message.contains("expected a floating dtype"),
        "message: {message}"
    );
}

const PLAIN_CONFIG: &str = r#"{"quantization": {"mode": "affine", "bits": 4, "group_size": 32}}"#;
const MXFP4_CONFIG: &str = r#"{"quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32}}"#;
const GATE_64_CONFIG: &str = r#"{
  "quantization": {
    "mode": "affine",
    "bits": 4,
    "group_size": 32,
    "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
    "language_model.model.layers.0.mlp.shared_expert_gate": {"bits": 8, "group_size": 64},
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj": {"bits": 4, "group_size": 64},
    "language_model.model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32}
  }
}"#;
const INCIDENTAL_CONFIG: &str = r#"{
  "quantization": {
    "mode": "affine",
    "bits": 4,
    "group_size": 32,
    "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
    "language_model.model.layers.0.mlp.shared_expert_gate": {"bits": 8, "group_size": 64},
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj": {"bits": 4, "group_size": 128},
    "language_model.model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32},
    "language_model.model.layers.0.input_layernorm": {"bits": 8, "group_size": 64},
    "language_model.model.layers.1.linear_attn.norm": {"bits": 8, "group_size": 64}
  }
}"#;
const PARTIAL: &[u8] = include_bytes!("fixtures/family-partial.safetensors");

fn resolve_error(
    layer: u32,
    index: &IndexedWeights,
    spec: &FamilySpec,
    plan: &WeightPlan,
) -> EngError {
    match LayerWeights::resolve(index, spec, layer, plan) {
        Ok(_) => panic!("resolve({layer}) must fail"),
        Err(err) => err,
    }
}

fn checkpoint_message(err: EngError) -> String {
    match err {
        EngError::Checkpoint(message) => message,
        err => panic!("expected Checkpoint, got {err:?}"),
    }
}

#[test]
fn missing_plan_entry_fails_checkpoint() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = plan(PLAIN_CONFIG.as_bytes(), &[header(PARTIAL)]);
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("weight plan lacks language_model.model.layers.0.mlp.gate.weight"),
        "message: {message}"
    );
}

#[test]
fn default_plan_against_overridden_shard_fails() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = plan(PLAIN_CONFIG.as_bytes(), &[header(FULL)]);
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("mlp.gate.weight") && message.contains("expected [4, 16]"),
        "message: {message}"
    );
}

#[test]
fn changed_group_against_unchanged_sidecar_fails() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = plan(GATE_64_CONFIG.as_bytes(), &[header(FULL)]);
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("mlp.switch_mlp.gate_proj.scales")
            && message.contains("expected [4, 64, 2]"),
        "message: {message}"
    );
}

#[test]
fn mxfp4_plan_rejects_non_affine() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = plan(MXFP4_CONFIG.as_bytes(), &[header(FULL)]);
    let message = checkpoint_message(resolve_error(0, &index, &TINY, &plan));
    assert!(
        message.contains("mlp.gate.weight")
            && message.contains("uses unsupported Mxfp4 quantization"),
        "message: {message}"
    );
}

#[test]
fn incidental_plan_entries_leave_dense_roles_dense() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = plan(INCIDENTAL_CONFIG.as_bytes(), &[header(FULL)]);
    assert_eq!(
        plan.quantization("language_model.model.layers.0.input_layernorm.weight"),
        Some(ModuleQuantization::Unquantized)
    );
    assert_eq!(
        plan.quantization("language_model.model.layers.1.linear_attn.norm.weight"),
        Some(ModuleQuantization::Unquantized)
    );
    let full = resolve(0, &index, &TINY, &plan);
    assert_eq!(full.input_norm.shape(), &[128]);
    assert_eq!(full.post_attention_norm.shape(), &[128]);
    let delta = resolve(1, &index, &TINY, &plan);
    let MixerWeights::DeltaNet(weights) = &delta.mixer else {
        panic!("layer 1 must use delta net")
    };
    assert_eq!(weights.norm.shape(), &[4]);
}

const RATIO_LAYERS: [AttentionKind; 1] = [AttentionKind::DeltaNet];

// Hand-derived from the mlx-lm GatedDeltaNet constructor: key width 2 * 8 = 16, value width
// 4 * 16 = 64, fused width 2 * 16 + 64 = 96. Every count differs, so a swapped head count and
// head width shows. Hidden width 128 makes every old multiple (512, 256) wrong.
const RATIO_LINEAR: LinearAttention = LinearAttention {
    key_heads: 2,
    key_head_dim: 8,
    value_heads: 4,
    value_head_dim: 16,
    conv_kernel: 3,
};

const RATIO: FamilySpec = FamilySpec {
    name: "ratio_family",
    layers: 1,
    hidden: 128,
    heads_q: 4,
    heads_kv: 2,
    head_dim: 16,
    rope_theta: 10_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["test_model"],
    vocab: 32,
    feed_forward: None,
    experts: 4,
    top_k: 2,
    moe_inter: 64,
    shared_inter: 64,
    quant_bits: 4,
    quant_group: 32,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &RATIO_LAYERS,
    linear: Some(RATIO_LINEAR),
    hyper: None,
};

fn ratio_plan() -> WeightPlan {
    plan_for(&RATIO, PLAIN_CONFIG.as_bytes(), &[header(RATIO_BYTES)])
}

fn ratio_index() -> IndexedWeights {
    load_index(RATIO_SHARD, RATIO_INDEX, RATIO_BYTES)
}

fn assert_ratio_projection(
    projection: &ProjectionWeights,
    stem: &str,
    weight: &[u32],
    sidecar: &[u32],
) {
    check_projection(
        projection,
        stem,
        Q4_32,
        (weight, sidecar),
        assert_ratio_role,
    );
}

#[test]
fn delta_shapes_follow_the_declared_linear_dimensions() {
    let layer = resolve(0, &ratio_index(), &RATIO, &ratio_plan());
    let stem = "language_model.model.layers.0.linear_attn.";
    let MixerWeights::DeltaNet(delta) = &layer.mixer else {
        panic!("layer 0 must use delta net")
    };
    for (role, name, rows) in [
        (&delta.in_proj_qkv, "in_proj_qkv", 96),
        (&delta.in_proj_z, "in_proj_z", 64),
        (&delta.in_proj_a, "in_proj_a", 4),
        (&delta.in_proj_b, "in_proj_b", 4),
    ] {
        assert_ratio_projection(role, &format!("{stem}{name}"), &[rows, 16], &[rows, 4]);
    }
    assert_ratio_projection(
        &delta.out_proj,
        &format!("{stem}out_proj"),
        &[128, 8],
        &[128, 2],
    );
    assert_ratio_role(&delta.conv1d, &format!("{stem}conv1d.weight"));
    assert_ratio_role(&delta.a_log, &format!("{stem}A_log"));
    assert_ratio_role(&delta.dt_bias, &format!("{stem}dt_bias"));
    assert_ratio_role(&delta.norm, &format!("{stem}norm.weight"));
    assert_eq!(delta.conv1d.shape(), &[96, 3, 1]);
    assert_eq!(delta.norm.shape(), &[16]);
}

#[test]
fn a_convolution_kernel_that_differs_from_the_checkpoint_fails() {
    let spec = FamilySpec {
        linear: Some(LinearAttention {
            conv_kernel: 4,
            ..RATIO_LINEAR
        }),
        ..RATIO
    };
    let message = checkpoint_message(resolve_error(0, &ratio_index(), &spec, &ratio_plan()));
    assert!(
        message.contains("linear_attn.conv1d.weight") && message.contains("expected [96, 4, 1]"),
        "message: {message}"
    );
}

#[test]
fn a_value_head_count_that_differs_from_the_checkpoint_fails() {
    // The fused and gate widths still match, so only the per-head roles disagree.
    let spec = FamilySpec {
        linear: Some(LinearAttention {
            value_heads: 8,
            value_head_dim: 8,
            ..RATIO_LINEAR
        }),
        ..RATIO
    };
    let message = checkpoint_message(resolve_error(0, &ratio_index(), &spec, &ratio_plan()));
    assert!(
        message.contains("linear_attn.in_proj_a.weight") && message.contains("expected [8, 16]"),
        "message: {message}"
    );
}

#[test]
fn a_delta_layer_without_linear_dimensions_fails() {
    let spec = FamilySpec {
        linear: None,
        ..RATIO
    };
    let message = checkpoint_message(resolve_error(0, &ratio_index(), &spec, &ratio_plan()));
    assert!(
        message.contains("ratio_family") && message.contains("linear attention dimensions"),
        "message: {message}"
    );
}

#[test]
fn projections_keep_the_checked_packed_weight() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let layer = resolve(0, &index, &TINY, &full_plan());
    let stem = "language_model.model.layers.0.";
    let FeedForwardWeights::Moe {
        router, experts, ..
    } = &layer.feed_forward
    else {
        panic!("layer 0 must use MoE")
    };
    let router = router.packed();
    assert_eq!(router.logical_shape(), [4, 128]);
    assert_eq!(router.quantization(), Q8_64);
    let stored = index
        .tensor(&format!("{stem}mlp.gate.weight"))
        .expect("router payload");
    assert_eq!(
        router.payload().host_bytes().map(<[u8]>::as_ptr),
        stored.host_bytes().map(<[u8]>::as_ptr),
        "the packed weight retains the stored payload"
    );
    assert_eq!(experts.gate.packed().logical_shape(), [4, 64, 128]);
    assert_eq!(experts.down.packed().logical_shape(), [4, 128, 64]);
    let MixerWeights::Attention(attn) = &layer.mixer else {
        panic!("layer 0 must use attention")
    };
    assert_eq!(attn.q.packed().logical_shape(), [128, 128]);
    assert_eq!(attn.o.packed().logical_shape(), [128, 64]);
}

#[test]
fn layers_resolve_from_any_weight_source() {
    let single = ResidentWeights::from_bytes(Arc::from(FULL)).expect("fixture loads");
    let layer = resolve_with(0, &single, &TINY, &full_plan());
    assert_eq!(layer.input_norm.shape(), &[128]);
    let indexed = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let dynamic: &dyn WeightSource = &indexed;
    let layer = resolve_with(1, dynamic, &TINY, &full_plan());
    assert!(matches!(layer.mixer, MixerWeights::DeltaNet(_)));
}

fn resolve_with<S: WeightSource + ?Sized>(
    layer: u32,
    source: &S,
    spec: &FamilySpec,
    plan: &WeightPlan,
) -> LayerWeights {
    match LayerWeights::resolve(source, spec, layer, plan) {
        Ok(weights) => weights,
        Err(err) => panic!("resolve({layer}) failed: {err:?}"),
    }
}

#[test]
fn only_projection_files_carry_a_quantization() {
    // Dense roles by name: norms, convolution, and the recurrent auxiliary vectors.
    const DENSE_SUFFIXES: [&str; 9] = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "linear_attn.norm.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "model.norm.weight",
    ];
    for (name, bytes, spec) in [
        (FULL_SHARD, FULL, &TINY),
        (RATIO_SHARD, RATIO_BYTES, &RATIO),
    ] {
        let plan = plan_for(spec, PLAIN_CONFIG.as_bytes(), &[header(bytes)]);
        let mut dense = 0;
        for (key, entry) in plan.iter() {
            if DENSE_SUFFIXES.iter().any(|suffix| key.ends_with(suffix)) {
                dense += 1;
                assert_eq!(entry.quantization, ModuleQuantization::Unquantized, "{key}");
            } else {
                assert_eq!(
                    entry.quantization,
                    ModuleQuantization::Quantized(Q4_32),
                    "{key}"
                );
            }
        }
        assert!(dense >= 3, "{name} holds dense roles");
    }
}

const QSA_LAYERS: [AttentionKind; 2] = [AttentionKind::Qsa, AttentionKind::DeltaNet];

#[test]
fn a_qsa_layer_is_rejected_as_unsupported() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = full_plan();
    let spec = FamilySpec {
        layers_attention: &QSA_LAYERS,
        ..TINY
    };
    let message = checkpoint_message(resolve_error(0, &index, &spec, &plan));
    assert_eq!(message, "layer 0 uses unsupported qsa attention");
}

#[test]
fn a_qsa_layer_is_rejected_before_its_moe_tensors_load() {
    let index = load_index(MISSING_SHARD, MISSING_INDEX, MISSING);
    let plan = full_plan();
    let spec = FamilySpec {
        layers_attention: &QSA_LAYERS,
        ..TINY
    };
    let counting = CountingSource {
        inner: &index,
        calls: AtomicUsize::new(0),
    };
    let err = match LayerWeights::resolve(&counting, &spec, 0, &plan) {
        Ok(_) => panic!("resolve(0) must fail"),
        Err(err) => err,
    };
    assert_eq!(
        checkpoint_message(err),
        "layer 0 uses unsupported qsa attention"
    );
    assert_eq!(counting.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn a_layer_beyond_the_family_is_rejected() {
    let index = load_index(FULL_SHARD, FULL_INDEX, FULL);
    let plan = full_plan();
    let beyond = TINY_LAYERS.len() as u32;
    let message = checkpoint_message(resolve_error(beyond, &index, &TINY, &plan));
    assert_eq!(message, format!("layer {beyond} is out of range"));
}

struct CountingSource<'a> {
    inner: &'a IndexedWeights,
    calls: AtomicUsize,
}

impl WeightSource for CountingSource<'_> {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.tensor(key)
    }
    fn resident_bytes(&self) -> u64 {
        self.inner.resident_bytes()
    }
    fn tier(&self) -> Tier {
        self.inner.tier()
    }
}
