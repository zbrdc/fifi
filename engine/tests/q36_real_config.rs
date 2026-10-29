//! Checks the stepping-stone specification against the recorded real checkpoint.
//!
//! Source: `<model_dir>` on the Mac (base shards only). The tests read the
//! committed `config.json` copy and the committed shard headers (the 8-byte prefix and the JSON
//! header of each shard). They read no tensor data. Every expected number below comes from the
//! shard headers, the index, and `config.json`. None comes from the code under test.

use std::{collections::BTreeMap, io::Cursor};

use inference_engine::{
    EngError,
    family::{
        AttentionKind, KeyTier, QWEN3_6_35B_A3B, WeightPlan,
        quantization::{ModuleQuantization, QuantizationMap},
    },
    kernels::{QuantizationMode, QuantizationSpec},
    tensor::Dtype,
    weights::ShardHeader,
};
use sha2::{Digest, Sha256};

const CONFIG: &[u8] = include_bytes!("fixtures/qwen3_6_35b_a3b_config.json");
const CONFIG_SHA256: &str = "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767";

/// Name, committed header bytes, header SHA-256, real shard byte length, tensor count.
struct Shard {
    name: &'static str,
    header: &'static [u8],
    header_sha256: &'static str,
    byte_len: u64,
    tensors: usize,
}

const SHARDS: [Shard; 4] = [
    Shard {
        name: "model-00001-of-00004.safetensors",
        header: include_bytes!(
            "fixtures/qwen3_6_35b_a3b_shard_headers/model-00001-of-00004.header"
        ),
        header_sha256: "d52a5010481fdee5e24648209c5af874104d0f358a9c1398fb13af9365ac74c3",
        byte_len: 4_395_019_264,
        tensors: 383,
    },
    Shard {
        name: "model-00002-of-00004.safetensors",
        header: include_bytes!(
            "fixtures/qwen3_6_35b_a3b_shard_headers/model-00002-of-00004.header"
        ),
        header_sha256: "52c2ade7f85d3c1aa9bb67174d52acf97759b61a434d3d8bc9f1caae4ac66134",
        byte_len: 5_368_472_749,
        tensors: 498,
    },
    Shard {
        name: "model-00003-of-00004.safetensors",
        header: include_bytes!(
            "fixtures/qwen3_6_35b_a3b_shard_headers/model-00003-of-00004.header"
        ),
        header_sha256: "67b3b3b59495a8183721a1189990efe8aeadcc8476428bee3ad9a1c0d3a33818",
        byte_len: 5_368_324_139,
        tensors: 504,
    },
    Shard {
        name: "model-00004-of-00004.safetensors",
        header: include_bytes!(
            "fixtures/qwen3_6_35b_a3b_shard_headers/model-00004-of-00004.header"
        ),
        header_sha256: "40f89330260b304448daa201c250dd55385ddbece1d94d3b658214b784ec98dd",
        byte_len: 4_377_211_365,
        tensors: 372,
    },
];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn headers() -> Vec<ShardHeader> {
    SHARDS
        .iter()
        .map(|shard| {
            ShardHeader::read(&mut Cursor::new(shard.header), shard.byte_len, 1 << 20)
                .unwrap_or_else(|error| panic!("{}: {error:?}", shard.name))
        })
        .collect()
}

fn plan_for(config: &[u8], headers: &[ShardHeader]) -> WeightPlan {
    let quant = QuantizationMap::from_json(config).expect("quantization map");
    WeightPlan::build(&QWEN3_6_35B_A3B, &quant, headers).expect("weight plan")
}

/// Every stored tensor as `(dtype, shape)`, keyed by name, with its shard number (1 to 4).
fn stored(headers: &[ShardHeader]) -> BTreeMap<String, (Dtype, Vec<u32>, usize)> {
    let mut map = BTreeMap::new();
    for (number, header) in headers.iter().enumerate() {
        for (key, entry) in header.iter() {
            let previous = map.insert(
                key.to_string(),
                (entry.dtype(), entry.shape().to_vec(), number + 1),
            );
            assert!(previous.is_none(), "{key} appears in two shards");
        }
    }
    map
}

#[test]
fn recorded_files_are_hash_pinned() {
    assert_eq!(sha256(CONFIG), CONFIG_SHA256);
    for shard in &SHARDS {
        assert_eq!(sha256(shard.header), shard.header_sha256, "{}", shard.name);
    }
}

#[test]
fn headers_hold_the_recorded_counts_and_bytes() {
    let headers = headers();
    for (shard, header) in SHARDS.iter().zip(&headers) {
        assert_eq!(header.iter().count(), shard.tensors, "{}", shard.name);
    }
    let all = stored(&headers);
    assert_eq!(all.len(), 1757);
    assert!(
        all.values()
            .all(|(dtype, ..)| matches!(dtype, Dtype::Bf16 | Dtype::U32))
    );
    // Payload bytes: 19,508,787,456 in all, of which the 120 expert stacks (three arrays each)
    // hold 18,119,393,280. The rest is 1,389,394,176.
    let mut total = 0u64;
    let mut experts = 0u64;
    for header in &headers {
        for (key, entry) in header.iter() {
            let bytes = entry.offsets().end - entry.offsets().start;
            total += bytes;
            if key.contains(".mlp.switch_mlp.") {
                experts += bytes;
            }
        }
    }
    assert_eq!(total, 19_508_787_456);
    assert_eq!(experts, 18_119_393_280);
    assert_eq!(total - experts, 1_389_394_176);
}

#[test]
fn three_layers_have_tensors_in_two_shards() {
    // The loader must not assume that one layer sits in one shard.
    let mut shards_of: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (key, (_, _, shard)) in stored(&headers()) {
        let Some(rest) = key.strip_prefix("language_model.model.layers.") else {
            continue;
        };
        let layer: u32 = rest.split('.').next().unwrap().parse().unwrap();
        let list = shards_of.entry(layer).or_default();
        if !list.contains(&shard) {
            list.push(shard);
        }
    }
    let split: BTreeMap<u32, Vec<usize>> = shards_of
        .into_iter()
        .filter(|(_, list)| list.len() > 1)
        .map(|(layer, mut list)| {
            list.sort_unstable();
            (layer, list)
        })
        .collect();
    assert_eq!(
        split,
        BTreeMap::from([(8, vec![1, 2]), (20, vec![2, 3]), (31, vec![3, 4])])
    );
}

#[test]
fn the_checkpoint_has_no_mtp_and_no_vision_tensors() {
    let all = stored(&headers());
    assert!(all.keys().all(|key| key.starts_with("language_model.")));
    assert!(
        all.keys()
            .all(|key| !key.contains("mtp") && !key.contains("visual"))
    );
    // The configuration still declares one MTP layer. The family has `mtp: false`.
    let root: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    assert_eq!(root["text_config"]["mtp_num_hidden_layers"], 1);
    const { assert!(!QWEN3_6_35B_A3B.mtp) };
}

/// One group of tensors, named by the key without the `layers.N.` prefix.
enum Group {
    /// `D(key, shape)`: a dense BF16 tensor.
    D(&'static str, &'static [u32]),
    /// `M(stem, rows, input, bits)`: a packed matrix. U32 payload `[rows, input * bits / 32]`,
    /// BF16 scales and biases `[rows, input / 64]`, groups of 64.
    M(&'static str, u32, u32, u32),
    /// `S(stem, rows, input)`: a packed stack of 256 experts at 4 bits, groups of 64.
    S(&'static str, u32, u32),
}

use Group::{D, M, S};

/// Groups present in every one of the 40 layers.
const EVERY_LAYER: [Group; 10] = [
    D("input_layernorm.weight", &[2048]),
    D("post_attention_layernorm.weight", &[2048]),
    M("mlp.gate", 256, 2048, 8),
    M("mlp.shared_expert_gate", 1, 2048, 8),
    M("mlp.shared_expert.gate_proj", 512, 2048, 4),
    M("mlp.shared_expert.up_proj", 512, 2048, 4),
    M("mlp.shared_expert.down_proj", 2048, 512, 4),
    S("mlp.switch_mlp.gate_proj", 512, 2048),
    S("mlp.switch_mlp.up_proj", 512, 2048),
    S("mlp.switch_mlp.down_proj", 2048, 512),
];

/// Groups of the 30 `linear_attention` layers (indices 0, 1, 2, 4, 5, 6, 8, and so on).
const DELTA_LAYER: [Group; 9] = [
    M("linear_attn.in_proj_qkv", 8192, 2048, 4),
    M("linear_attn.in_proj_z", 4096, 2048, 4),
    M("linear_attn.in_proj_a", 32, 2048, 4),
    M("linear_attn.in_proj_b", 32, 2048, 4),
    M("linear_attn.out_proj", 2048, 4096, 4),
    D("linear_attn.conv1d.weight", &[8192, 4, 1]),
    D("linear_attn.A_log", &[32]),
    D("linear_attn.dt_bias", &[32]),
    D("linear_attn.norm.weight", &[128]),
];

/// Groups of the 10 `full_attention` layers (indices 3, 7, and so on up to 39).
const FULL_LAYER: [Group; 6] = [
    M("self_attn.q_proj", 8192, 2048, 4),
    M("self_attn.k_proj", 512, 2048, 4),
    M("self_attn.v_proj", 512, 2048, 4),
    M("self_attn.o_proj", 2048, 4096, 4),
    D("self_attn.q_norm.weight", &[256]),
    D("self_attn.k_norm.weight", &[256]),
];

/// Groups outside the layers. Their names carry the full prefix.
const ROOT: [Group; 3] = [
    M("language_model.model.embed_tokens", 248_320, 2048, 4),
    M("language_model.lm_head", 248_320, 2048, 4),
    D("language_model.model.norm.weight", &[2048]),
];

/// The storage the plan must resolve for a key: `None` for a dense tensor, else the bit width.
type Expected = (Dtype, Vec<u32>, Option<u32>);

fn add_group(out: &mut BTreeMap<String, Expected>, prefix: &str, group: &Group) {
    match *group {
        D(name, shape) => {
            out.insert(
                format!("{prefix}{name}"),
                (Dtype::Bf16, shape.to_vec(), None),
            );
        }
        M(stem, rows, input, bits) => {
            add_packed(out, &format!("{prefix}{stem}"), &[rows], input, bits)
        }
        S(stem, rows, input) => add_packed(out, &format!("{prefix}{stem}"), &[256, rows], input, 4),
    }
}

fn add_packed(
    out: &mut BTreeMap<String, Expected>,
    stem: &str,
    lead: &[u32],
    input: u32,
    bits: u32,
) {
    let mut payload = lead.to_vec();
    payload.push(input * bits / 32);
    let mut sidecar = lead.to_vec();
    sidecar.push(input / 64);
    out.insert(format!("{stem}.weight"), (Dtype::U32, payload, Some(bits)));
    out.insert(
        format!("{stem}.scales"),
        (Dtype::Bf16, sidecar.clone(), Some(bits)),
    );
    out.insert(format!("{stem}.biases"), (Dtype::Bf16, sidecar, Some(bits)));
}

/// The inventory that the specification implies for the real checkpoint.
fn inventory() -> BTreeMap<String, Expected> {
    let mut out = BTreeMap::new();
    for (layer, kind) in QWEN3_6_35B_A3B.layers_attention.iter().enumerate() {
        let prefix = format!("language_model.model.layers.{layer}.");
        let own: &[Group] = match kind {
            AttentionKind::DeltaNet => &DELTA_LAYER,
            AttentionKind::Full => &FULL_LAYER,
            AttentionKind::Qsa => panic!("the stepping stone has no QSA layer"),
        };
        for group in EVERY_LAYER.iter().chain(own) {
            add_group(&mut out, &prefix, group);
        }
    }
    for group in &ROOT {
        add_group(&mut out, "", group);
    }
    out
}

#[test]
fn stored_tensors_equal_the_inventory_the_spec_implies() {
    let expected = inventory();
    let actual = stored(&headers());
    assert_eq!(expected.len(), 1757);
    for (key, (dtype, shape, _)) in &expected {
        let found = actual
            .get(key)
            .unwrap_or_else(|| panic!("{key} is missing"));
        assert_eq!((found.0, &found.1), (*dtype, shape), "{key}");
    }
    for key in actual.keys() {
        assert!(expected.contains_key(key), "{key} is not in the inventory");
    }
}

#[test]
fn the_weight_plan_resolves_tier_and_quantization_for_every_key() {
    let plan = plan_for(CONFIG, &headers());
    let expected = inventory();
    assert_eq!(plan.len(), 1757);
    let (mut int4, mut int8, mut dense, mut stacked) = (0, 0, 0, 0);
    for (key, (_, _, bits)) in &expected {
        let entry = plan
            .quantization(key)
            .unwrap_or_else(|| panic!("{key}: no plan entry"));
        match bits {
            None => {
                assert_eq!(entry, ModuleQuantization::Unquantized, "{key}");
                dense += 1;
            }
            Some(bits) => {
                let spec = QuantizationSpec {
                    mode: QuantizationMode::Affine,
                    bits: *bits,
                    group_size: 64,
                };
                assert_eq!(entry, ModuleQuantization::Quantized(spec), "{key}");
                if *bits == 8 { int8 += 1 } else { int4 += 1 }
            }
        }
        let tier = if key.contains(".mlp.switch_mlp.") {
            KeyTier::ExpertStacked
        } else {
            KeyTier::Resident
        };
        assert_eq!(plan.tier(key), Some(tier), "{key}");
        if tier == KeyTier::ExpertStacked {
            stacked += 1;
        }
    }
    // 80 int8 modules with three keys each, 120 stacks with three keys each, 221 dense tensors.
    assert_eq!((int8, int4, dense, stacked), (240, 1296, 221, 360));
}

/// Counts the packed `.weight` keys whose stored width disagrees with the plan's bit width.
fn width_mismatches(plan: &WeightPlan, headers: &[ShardHeader]) -> usize {
    let all = stored(headers);
    let mut count = 0;
    for (key, (_, shape, _)) in &all {
        let Some(stem) = key.strip_suffix(".weight") else {
            continue;
        };
        let Some(ModuleQuantization::Quantized(spec)) = plan.quantization(key) else {
            continue;
        };
        let scales = &all[&format!("{stem}.scales")].1;
        let input = scales.last().unwrap() * spec.group_size;
        if *shape.last().unwrap() != input * spec.bits / 32 {
            count += 1;
        }
    }
    count
}

#[test]
fn the_int8_overrides_are_what_makes_the_stored_widths_consistent() {
    let headers = headers();
    assert_eq!(width_mismatches(&plan_for(CONFIG, &headers), &headers), 0);
    // Without the 80 overrides the plan resolves the gate modules at 4 bits. Their stored
    // payload has twice the width that 4 bits would give.
    let mut root: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    for name in ["quantization", "quantization_config"] {
        root[name]
            .as_object_mut()
            .unwrap()
            .retain(|_, value| !value.is_object());
    }
    let stripped = serde_json::to_vec(&root).unwrap();
    assert_eq!(
        width_mismatches(&plan_for(&stripped, &headers), &headers),
        80
    );
}

#[test]
fn the_config_declares_the_recorded_overrides_rope_and_token_values() {
    let root: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    let text = &root["text_config"];
    let quant = root["quantization"].as_object().unwrap();
    assert_eq!(root["quantization"], root["quantization_config"]);
    assert_eq!(quant.len(), 83);
    let modules: Vec<&String> = quant
        .iter()
        .filter(|(_, v)| v.is_object())
        .map(|(k, _)| k)
        .collect();
    assert_eq!(modules.len(), 80);
    for layer in 0..40 {
        for name in ["mlp.gate", "mlp.shared_expert_gate"] {
            let key = format!("language_model.model.layers.{layer}.{name}");
            assert_eq!(
                quant[&key],
                serde_json::json!({"group_size": 64, "bits": 8}),
                "{key}"
            );
        }
    }
    // `check_config` skips these fields (TODO S7). The Q36 loader checks them itself.
    assert_eq!(text["partial_rotary_factor"], 0.25);
    assert_eq!(text["rope_parameters"]["partial_rotary_factor"], 0.25);
    assert_eq!(text["head_dim"], 256);
    assert_eq!(256.0 * 0.25, 64.0, "rotary dimension");
    assert_eq!(
        text["rope_parameters"]["mrope_section"],
        serde_json::json!([11, 11, 10])
    );
    assert_eq!(
        11 + 11 + 10,
        64 / 2,
        "mrope sections cover half the rotary dimension"
    );
    assert_eq!(text["rope_parameters"]["mrope_interleaved"], true);
    assert_eq!(text["rope_parameters"]["rope_type"], "default");
    assert_eq!(text["rope_parameters"]["rope_theta"], 10_000_000);
    assert_eq!(root["eos_token_id"], serde_json::json!([248046, 248044]));
    assert_eq!(text["eos_token_id"], 248044);
    assert_eq!(text["max_position_embeddings"], 262_144);
    assert_eq!(text["mamba_ssm_dtype"], "float32");
    assert_eq!(text["dtype"], "bfloat16");
}

#[test]
fn the_pinned_config_passes_the_family_check_and_a_changed_one_does_not() {
    QWEN3_6_35B_A3B
        .check_config(CONFIG)
        .expect("pinned configuration");
    let mut root: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    root["text_config"]["num_experts_per_tok"] = 10.into();
    let changed = serde_json::to_vec(&root).unwrap();
    assert!(matches!(
        QWEN3_6_35B_A3B.check_config(&changed),
        Err(EngError::Checkpoint(message)) if message.contains("num_experts_per_tok")
    ));
}
