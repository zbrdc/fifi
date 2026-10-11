//! Test support: small in-memory models with the key layout of `Qwen3.6-35B-A3B`.
//!
//! `TINY` is the two-layer model of the host reference tests: layer 0 is DeltaNet and layer 1 is dense
//! attention with 2 query heads and 1 key-value head. `TINY_GQA` has four layers (DeltaNet, attention,
//! DeltaNet, attention) and 4 query heads over 2 key-value heads, so that the head grouping is visible.
//! Both use hidden 64, quantization groups of 32, vocabulary 8, 4 experts with top 2, head dimension 32,
//! and a rotary dimension of 16. Router and shared expert gate are 8-bit. Every other module is 4-bit.

use serde_json::Value;

use super::fixtures::{self, Entry, dense, val};
use crate::family::spec::{Act, AttentionKind, FamilySpec, LinearAttention, Router};
use crate::weights::StreamedCheckpoint;

const fn model(
    name: &'static str,
    layers: u32,
    (heads_q, heads_kv): (u32, u32),
    (key_heads, value_heads): (u32, u32),
    kinds: &'static [AttentionKind],
) -> FamilySpec {
    FamilySpec {
        name,
        layers,
        hidden: 64,
        heads_q,
        heads_kv,
        head_dim: 32,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        model_types: &["qwen3_5_moe", "qwen3_5_moe_text"],
        vocab: 8,
        experts: 4,
        top_k: 2,
        moe_inter: 64,
        shared_inter: 64,
        quant_bits: 4,
        quant_group: 32,
        act: Act::Silu,
        router: Router::SoftmaxTopK { renorm: true },
        mtp: false,
        layers_attention: kinds,
        linear: Some(LinearAttention {
            key_heads,
            key_head_dim: 32,
            value_heads,
            value_head_dim: 32,
            conv_kernel: 4,
        }),
        hyper: None,
    }
}

/// Two layers: DeltaNet, then attention with 2 query heads and 1 key-value head.
pub(super) const TINY: FamilySpec = model(
    "q36_tiny",
    2,
    (2, 1),
    (1, 2),
    &[AttentionKind::DeltaNet, AttentionKind::Full],
);

/// Two layers like `TINY`, but DeltaNet has 2 key heads and 4 value heads (ratio 2), so that the map from
/// value head to key head is visible. Its convolution has 256 channels.
pub(super) const TINY_WIDE: FamilySpec = model(
    "q36_tiny_wide",
    2,
    (2, 1),
    (2, 4),
    &[AttentionKind::DeltaNet, AttentionKind::Full],
);

/// Four layers: DeltaNet, attention, DeltaNet, attention, with 4 query heads and 2 key-value heads.
pub(super) const TINY_GQA: FamilySpec = model(
    "q36_tiny_gqa",
    4,
    (4, 2),
    (1, 2),
    &[
        AttentionKind::DeltaNet,
        AttentionKind::Full,
        AttentionKind::DeltaNet,
        AttentionKind::Full,
    ],
);

/// The `config.json` of `spec`, in the real file's shape: quantization overrides for the router
/// and the shared expert gate (8 bits), and the text configuration under `text_config`.
pub(super) fn config(spec: &FamilySpec) -> Vec<u8> {
    let mut quantization = serde_json::json!({"mode": "affine", "bits": 4, "group_size": 32});
    for layer in 0..spec.layers {
        for name in ["mlp.gate", "mlp.shared_expert_gate"] {
            quantization[format!("language_model.model.layers.{layer}.{name}")] =
                serde_json::json!({"bits": 8, "group_size": 32});
        }
    }
    let lin = spec.linear.unwrap();
    let value: Value = serde_json::json!({
        "model_type": "qwen3_5_moe",
        "quantization": quantization,
        "text_config": {
            "model_type": "qwen3_5_moe_text", "hidden_size": 64,
            "num_attention_heads": spec.heads_q, "num_hidden_layers": spec.layers,
            "num_key_value_heads": spec.heads_kv, "vocab_size": 8,
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64, "head_dim": 32, "rms_norm_eps": 1e-6,
            "hidden_act": "silu", "full_attention_interval": 2,
            "linear_num_key_heads": lin.key_heads, "linear_key_head_dim": 32,
            "linear_num_value_heads": lin.value_heads,
            "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                "rope_type": "default", "mrope_section": [3, 3, 2]
            }
        }
    });
    serde_json::to_vec(&value).unwrap()
}

/// Adds a packed module with hash-derived words, scales, and biases. Scales are multiples of
/// `2^-(bits + 5)` and biases are multiples of 1/64, so every value is exact in BF16.
fn module(out: &mut Vec<Entry>, tag: &mut u64, stem: &str, lead: &[u32], cols: u32, bits: u32) {
    *tag += 1;
    let tag = *tag;
    let scale_unit = 1.0 / f32::from(1u16 << (bits + 5));
    fixtures::packed(
        out,
        (stem, lead, cols),
        (bits, 32),
        |i| fixtures::hash32(tag, i as u64),
        |g| (8.0 + 8.0 * val(tag, g as u64)) * scale_unit,
        |g| val(tag + 1000, g as u64) / 8.0,
    );
}

fn floats(out: &mut Vec<Entry>, tag: &mut u64, key: &str, len: u32, f: impl Fn(f32) -> f32) {
    *tag += 1;
    let values: Vec<f32> = (0..u64::from(len)).map(|i| f(val(*tag, i))).collect();
    dense(out, key, vec![len], &values);
}

fn one_plus_quarter(v: f32) -> f32 {
    1.0 + v / 4.0
}

/// Every tensor of the model, with a different hash tag per tensor.
pub(super) fn entries(spec: &FamilySpec) -> Vec<Entry> {
    let (mut out, tag) = (Vec::new(), &mut 0u64);
    let (hq, hkv) = (spec.heads_q, spec.heads_kv);
    let lin = spec.linear.unwrap();
    let head = |name: &str| format!("language_model.{name}");
    module(
        &mut out,
        tag,
        "language_model.model.embed_tokens",
        &[8],
        64,
        4,
    );
    module(&mut out, tag, &head("lm_head"), &[8], 64, 4);
    floats(
        &mut out,
        tag,
        &head("model.norm.weight"),
        64,
        one_plus_quarter,
    );
    for (layer, kind) in spec.layers_attention.iter().enumerate() {
        let p = format!("language_model.model.layers.{layer}.");
        floats(
            &mut out,
            tag,
            &format!("{p}input_layernorm.weight"),
            64,
            one_plus_quarter,
        );
        floats(
            &mut out,
            tag,
            &format!("{p}post_attention_layernorm.weight"),
            64,
            one_plus_quarter,
        );
        if *kind == AttentionKind::DeltaNet {
            let (qkv_rows, z_rows) = (
                2 * lin.key_heads * 32 + lin.value_heads * 32,
                lin.value_heads * 32,
            );
            for (name, rows) in [
                ("in_proj_qkv", qkv_rows),
                ("in_proj_z", z_rows),
                ("in_proj_b", lin.value_heads),
                ("in_proj_a", lin.value_heads),
            ] {
                module(
                    &mut out,
                    tag,
                    &format!("{p}linear_attn.{name}"),
                    &[rows],
                    64,
                    4,
                );
            }
            module(
                &mut out,
                tag,
                &format!("{p}linear_attn.out_proj"),
                &[64],
                z_rows,
                4,
            );
            floats(
                &mut out,
                tag,
                &format!("{p}linear_attn.conv1d.weight"),
                qkv_rows * 4,
                |v| v,
            );
            floats(
                &mut out,
                tag,
                &format!("{p}linear_attn.A_log"),
                lin.value_heads,
                |v| v / 2.0,
            );
            floats(
                &mut out,
                tag,
                &format!("{p}linear_attn.dt_bias"),
                lin.value_heads,
                |v| v / 2.0,
            );
            floats(
                &mut out,
                tag,
                &format!("{p}linear_attn.norm.weight"),
                32,
                one_plus_quarter,
            );
        } else {
            for (name, rows, cols) in [
                ("q_proj", hq * 64, 64),
                ("k_proj", hkv * 32, 64),
                ("v_proj", hkv * 32, 64),
                ("o_proj", 64, hq * 32),
            ] {
                module(
                    &mut out,
                    tag,
                    &format!("{p}self_attn.{name}"),
                    &[rows],
                    cols,
                    4,
                );
            }
            floats(
                &mut out,
                tag,
                &format!("{p}self_attn.q_norm.weight"),
                32,
                one_plus_quarter,
            );
            floats(
                &mut out,
                tag,
                &format!("{p}self_attn.k_norm.weight"),
                32,
                one_plus_quarter,
            );
        }
        module(&mut out, tag, &format!("{p}mlp.gate"), &[4], 64, 8);
        module(
            &mut out,
            tag,
            &format!("{p}mlp.shared_expert_gate"),
            &[1],
            64,
            8,
        );
        for name in ["gate_proj", "up_proj", "down_proj"] {
            module(
                &mut out,
                tag,
                &format!("{p}mlp.switch_mlp.{name}"),
                &[4, 64],
                64,
                4,
            );
            module(
                &mut out,
                tag,
                &format!("{p}mlp.shared_expert.{name}"),
                &[64],
                64,
                4,
            );
        }
    }
    out
}

/// The tensors of `spec` in one in-memory shard, read through the same reader as a real checkpoint.
pub(super) fn checkpoint(spec: &FamilySpec) -> StreamedCheckpoint {
    fixtures::checkpoint(entries(spec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::host_load::HostWeights;

    #[test]
    fn both_models_pass_the_config_check_and_the_host_load_reads_every_tensor() {
        for spec in [&TINY, &TINY_GQA, &TINY_WIDE] {
            let mut names: Vec<String> = entries(spec).iter().map(|e| e.0.clone()).collect();
            names.sort();
            let ckpt = checkpoint(spec);
            let weights = HostWeights::load(&ckpt, spec, &config(spec)).unwrap();
            assert_eq!(weights.keys(), names.as_slice(), "{}", spec.name);
            assert_eq!(weights.shape().kinds.len(), spec.layers as usize);
        }
    }

    #[test]
    fn the_gqa_model_has_four_query_heads_over_two_key_value_heads() {
        let weights =
            HostWeights::load(&checkpoint(&TINY_GQA), &TINY_GQA, &config(&TINY_GQA)).unwrap();
        let attn = weights.shape().attn;
        assert_eq!((attn.heads_q, attn.heads_kv, attn.rotary_dim), (4, 2, 16));
    }
}
