//! In-memory packed Flash-Next checkpoint for the resident tests.
//!
//! H 64, C 2, W 128, R 32, vocabulary 8; layer 0 DeltaNet, layer 1 QSA, PLE at layer 1. Every packed
//! tensor is affine 4-bit, group 32. Every payload word is `0x7654_3210`, so element `c` of a row is
//! `c % 8`. Scales are 1.0. Row `r` (over the flattened leading axes) has bias `BIASES[r % 4]`.

use std::io::Cursor;

use crate::weights::{ShardIo, StreamedCheckpoint};

const MODEL: &str = "language_model.model.";
const WORD: u32 = 0x7654_3210;
/// BF16 1.0.
const ONE: u16 = 0x3F80;
/// BF16 0.0, 0.25, 0.5, 0.75.
const BIASES: [u16; 4] = [0x0000, 0x3E80, 0x3F00, 0x3F40];
/// BF16 2^23, the embedding scale of `Embedding::TokenDependent`.
const WIDE_SCALE: u16 = 0x4B00;

/// How the synthetic embedding table is built.
///
/// The blocks of this model give residual values near 1e8 (the DeltaNet convolution weights run
/// from 0 to 511 and the router weights from 0 to 255). Every row of `Flat` is the same pattern
/// plus a bias of at most 0.75, so a token changes the logits by about one part in 1e7. That is
/// rounding noise: a host run gives logits that differ in the last bits for two prompts, and the
/// resident path on the Mac gave bitwise equal logits for the keys [1, 2, 3, 4] and [6, 4, 7, 1].
/// `TokenDependent` rotates the pattern of row `t` by `t` nibbles and scales the table by 2^23.
/// The embedding is then as large as the block outputs, and the same two keys differ by about 9
/// percent in the logits (the portable test `a_token_dependent_embedding_changes_the_logits_by_more_than_rounding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Embedding {
    /// Every row has the same pattern (the default of all earlier tests).
    Flat,
    /// Row `t` is the pattern rotated by `t` nibbles, with scale 2^23.
    TokenDependent,
}

/// One stored tensor: name, safetensors dtype, shape, little-endian bytes.
pub(super) type Entry = (String, &'static str, Vec<u32>, Vec<u8>);

/// The `config.json` bytes of the synthetic model.
pub(super) fn config() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "quantization": {"mode": "affine", "bits": 4, "group_size": 32},
        "text_config": {
            "hidden_size": 64, "rms_norm_eps": 1e-6, "hc_count": 2, "hc_lowrank": 32, "vocab_size": 8,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            "indexer_n_heads": 1, "indexer_head_dim": 32, "indexer_budget": 16, "indexer_compress_ratio": 4,
            "linear_num_key_heads": 1, "linear_key_head_dim": 32, "linear_num_value_heads": 2,
            "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "ngram_size": 3, "heads_per_ngram": 1,
            "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "ple_layer_ids": [2], "eos_token_id": 7
        }
    }))
    .unwrap()
}

/// Pushes `{stem}.weight`, `.scales`, and `.biases` for `logical` (last axis a multiple of 32).
fn packed(out: &mut Vec<Entry>, stem: &str, logical: &[u32]) {
    let (&cols, lead) = logical.split_last().unwrap();
    let rows = lead.iter().product::<u32>() as usize;
    let groups = (cols / 32) as usize;
    let mut payload_shape = lead.to_vec();
    payload_shape.push(cols / 8);
    let mut group_shape = lead.to_vec();
    group_shape.push(cols / 32);
    let payload = WORD.to_le_bytes().repeat(rows * (cols / 8) as usize);
    let scales = ONE.to_le_bytes().repeat(rows * groups);
    let biases: Vec<u8> = (0..rows)
        .flat_map(|r| BIASES[r % 4].to_le_bytes().repeat(groups))
        .collect();
    out.push((format!("{stem}.weight"), "U32", payload_shape, payload));
    out.push((
        format!("{stem}.scales"),
        "BF16",
        group_shape.clone(),
        scales,
    ));
    out.push((format!("{stem}.biases"), "BF16", group_shape, biases));
}

/// Pushes an F32 tensor.
fn floats(out: &mut Vec<Entry>, key: &str, shape: Vec<u32>, values: &[f32]) {
    out.push((
        key.to_string(),
        "F32",
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
}

/// Pushes an I64 vector.
fn longs(out: &mut Vec<Entry>, key: &str, values: &[i64]) {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.push((key.to_string(), "I64", vec![values.len() as u32], bytes));
}

fn hyper(out: &mut Vec<Entry>, prefix: &str, inject: bool) {
    floats(
        out,
        &format!("{prefix}hc_norm.weight"),
        vec![128],
        &[0.5; 128],
    );
    packed(out, &format!("{prefix}input_mix_weight_down"), &[32, 128]);
    packed(out, &format!("{prefix}input_mix_weight_up"), &[128, 32]);
    if inject {
        packed(out, &format!("{prefix}block_inject_weight"), &[2, 128]);
    }
}

fn delta(out: &mut Vec<Entry>, prefix: &str) {
    for (name, rows) in [
        ("in_proj_qkv", 128),
        ("in_proj_z", 64),
        ("in_proj_b", 2),
        ("in_proj_a", 2),
        ("out_proj", 64),
    ] {
        packed(out, &format!("{prefix}{name}"), &[rows, 64]);
    }
    let conv: Vec<f32> = (0..512).map(|i| i as f32).collect();
    floats(
        out,
        &format!("{prefix}conv1d.weight"),
        vec![128, 1, 4],
        &conv,
    );
    floats(out, &format!("{prefix}A_log"), vec![2], &[0.0, 1.0]);
    floats(out, &format!("{prefix}dt_bias"), vec![2], &[0.5, -0.5]);
    floats(out, &format!("{prefix}norm.weight"), vec![32], &[1.0; 32]);
}

fn attention(out: &mut Vec<Entry>, prefix: &str) {
    for (name, shape) in [
        ("q_proj", [256, 64]),
        ("k_proj", [64, 64]),
        ("v_proj", [64, 64]),
        ("o_proj", [64, 128]),
        ("indexer.index_qk_proj", [64, 64]),
    ] {
        packed(out, &format!("{prefix}{name}"), &shape);
    }
    floats(
        out,
        &format!("{prefix}q_norm.weight"),
        vec![64],
        &[0.25; 64],
    );
    floats(
        out,
        &format!("{prefix}k_norm.weight"),
        vec![64],
        &[-0.25; 64],
    );
    floats(
        out,
        &format!("{prefix}indexer.q_layernorm.weight"),
        vec![32],
        &[0.0; 32],
    );
    floats(
        out,
        &format!("{prefix}indexer.k_layernorm.weight"),
        vec![32],
        &[0.0; 32],
    );
}

fn moe(out: &mut Vec<Entry>, prefix: &str) {
    let router: Vec<f32> = (0..256).map(|i| i as f32).collect();
    floats(out, &format!("{prefix}gate.weight"), vec![4, 64], &router);
    packed(out, &format!("{prefix}switch_mlp.gate_proj"), &[4, 32, 64]);
    packed(out, &format!("{prefix}switch_mlp.up_proj"), &[4, 32, 64]);
    packed(out, &format!("{prefix}switch_mlp.down_proj"), &[4, 64, 32]);
    packed(out, &format!("{prefix}shared_expert.gate_proj"), &[32, 64]);
    packed(out, &format!("{prefix}shared_expert.up_proj"), &[32, 64]);
    packed(out, &format!("{prefix}shared_expert.down_proj"), &[64, 32]);
    packed(out, &format!("{prefix}shared_expert_gate"), &[1, 64]);
}

fn ple(out: &mut Vec<Entry>, prefix: &str) {
    let table = format!("{prefix}ple_embedding.");
    longs(out, &format!("{table}layer_multipliers"), &[3, 5, 7]);
    longs(out, &format!("{table}ngram_heads_vocab_sizes"), &[8, 8]);
    longs(out, &format!("{table}ngram_heads_offsets"), &[0, 8]);
    floats(
        out,
        &format!("{table}ngram_embedding.weight_scale"),
        vec![],
        &[0.5],
    );
    packed(out, &format!("{table}ngram_embedding.shards.0"), &[16, 32]);
    packed(out, &format!("{prefix}key_proj"), &[128, 64]);
    packed(out, &format!("{prefix}value_proj"), &[64, 64]);
    for name in ["norm_key", "norm_query", "norm_conv"] {
        floats(
            out,
            &format!("{prefix}{name}.weight"),
            vec![128],
            &[0.5; 128],
        );
    }
    let conv: Vec<f32> = (0..512).map(|i| i as f32).collect();
    floats(
        out,
        &format!("{prefix}conv1d.weight"),
        vec![128, 1, 4],
        &conv,
    );
}

/// Every tensor of the synthetic checkpoint (152 tensors).
pub(super) fn tensors() -> Vec<Entry> {
    tensors_with(Embedding::Flat)
}

/// Every tensor of the synthetic checkpoint with the given embedding table.
pub(super) fn tensors_with(embedding: Embedding) -> Vec<Entry> {
    let mut out = Vec::new();
    packed(&mut out, &format!("{MODEL}embed_tokens"), &[8, 64]);
    if embedding == Embedding::TokenDependent {
        // `packed` pushed the payload, the scales, and the biases of the table in this order.
        if let [payload, scales, _biases] = &mut out[..] {
            payload.3 = (0..8u32)
                .flat_map(|token| WORD.rotate_left(4 * token).to_le_bytes().repeat(8))
                .collect();
            scales.3 = WIDE_SCALE.to_le_bytes().repeat(8 * 2);
        }
    }
    packed(&mut out, "language_model.lm_head", &[8, 64]);
    hyper(&mut out, &format!("{MODEL}hyper_connection_mixer."), false);
    for (index, is_qsa) in [(0, false), (1, true)] {
        let layer = format!("{MODEL}layers.{index}.");
        hyper(&mut out, &format!("{layer}attn_hyper_connection."), true);
        if is_qsa {
            attention(&mut out, &format!("{layer}self_attn."));
        } else {
            delta(&mut out, &format!("{layer}linear_attn."));
        }
        hyper(&mut out, &format!("{layer}mlp_hyper_connection."), true);
        moe(&mut out, &format!("{layer}mlp."));
    }
    ple(&mut out, &format!("{MODEL}layers.1.ple."));
    out
}

/// The synthetic checkpoint, read through the same streamed reader as a real artifact.
pub(super) fn checkpoint() -> StreamedCheckpoint {
    checkpoint_with(Embedding::Flat)
}

/// The synthetic checkpoint with the given embedding table.
pub(super) fn checkpoint_with(embedding: Embedding) -> StreamedCheckpoint {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut index = serde_json::Map::new();
    for (name, dtype, shape, bytes) in tensors_with(embedding) {
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
        );
        index.insert(name, serde_json::json!("m.safetensors"));
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
    let len = file.len() as u64;
    let reader: Box<dyn ShardIo + Send> = Box::new(Cursor::new(file));
    StreamedCheckpoint::open(
        &index,
        vec![("m.safetensors".to_string(), reader, len)],
        1 << 20,
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{Embedding, checkpoint, checkpoint_with, config, tensors};
    use crate::family::{
        flash_next::{
            load::FlashWeights,
            step::{AttnWeights, ModelState, step},
        },
        qwen36::host_cpu::CpuProjector,
    };

    #[test]
    fn synthetic_checkpoint_loads_every_tensor() {
        let ckpt = checkpoint();
        let loaded = FlashWeights::load(&ckpt, &config());
        assert!(loaded.is_ok(), "load failed: {:?}", loaded.as_ref().err());
        let weights = loaded.unwrap();
        let names: Vec<String> = tensors()
            .into_iter()
            .map(|(name, ..)| name)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(names.len(), 152, "fixture tensor count");
        assert_eq!(
            weights.keys(),
            names.as_slice(),
            "the loader reads every fixture tensor"
        );
        let shape = weights.shape();
        assert_eq!(
            (
                shape.hyper.branches(),
                shape.hyper.hidden(),
                shape.hyper.rank()
            ),
            (2, 64, 32),
            "hyper"
        );
        assert!((shape.hyper.eps() - 1e-6).abs() <= 1e-12, "hyper eps");
        let d = shape.delta;
        assert_eq!(
            (
                d.key_heads(),
                d.key_dim(),
                d.value_heads(),
                d.value_dim(),
                d.conv_kernel(),
                d.conv_dim()
            ),
            (1, 32, 2, 32, 4, 128),
            "delta"
        );
        assert_eq!(
            (shape.moe.experts(), shape.moe.top_k(), shape.moe.renorm()),
            (4, 2, true),
            "moe"
        );
        assert_eq!(
            (shape.qsa.dims().head_dim, shape.qsa.dims().rotary_dim),
            (64, 32),
            "qsa"
        );
        let p = shape.ple.dims();
        assert_eq!((p.conv_kernel, p.ngram, p.head_dim), (4, 3, 32), "ple");
        assert_eq!(
            (shape.ple_layer, shape.vocab),
            (1, 8),
            "ple layer and vocabulary"
        );
        let view = weights.view();
        assert!(
            matches!(view.layers.as_slice(), [a, b] if matches!(a.attn, AttnWeights::Delta(_)) && matches!(b.attn, AttnWeights::Qsa(_))),
            "layer kinds"
        );
        assert!(
            (view.ple.weight_scale - 0.5).abs() <= 1e-6,
            "ple weight scale"
        );
    }

    /// The logits row after `prompt` from the host forward pass, which shares no code with the
    /// resident path. The row is the full vocabulary of 8.
    fn host_logits(embedding: Embedding, prompt: &[u32]) -> Vec<f32> {
        let ckpt = checkpoint_with(embedding);
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = flash.view();
        let mut projector = CpuProjector::new(&ckpt, 1);
        let mut state = ModelState::new(flash.shape(), &weights);
        let mut logits = Vec::new();
        for &token in prompt {
            logits = step(
                &mut projector,
                flash.shape(),
                &weights,
                &mut state,
                token,
                false,
            )
            .unwrap()
            .logits;
        }
        logits
    }

    /// The largest relative distance between two logits rows.
    fn relative_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()))
            .fold(0.0, f32::max)
    }

    #[test]
    fn a_token_dependent_embedding_changes_the_logits_by_more_than_rounding() {
        // The two keys of the engine prefix test. Both have four tokens, so only the ids differ.
        let (key_a, key_b) = ([1, 2, 3, 4], [6, 4, 7, 1]);
        let flat = relative_distance(
            &host_logits(Embedding::Flat, &key_a),
            &host_logits(Embedding::Flat, &key_b),
        );
        let wide = relative_distance(
            &host_logits(Embedding::TokenDependent, &key_a),
            &host_logits(Embedding::TokenDependent, &key_b),
        );
        // A measured host run gives 1.8e-7 and 9.0e-2. An f32 rounding difference between two
        // backends is far below 1e-4, and a state that differs shows far above it.
        assert!(flat < 1e-5, "the flat embedding gave {flat}");
        assert!(wide > 5e-2, "the token-dependent embedding gave {wide}");
    }
}
