//! Test packs: the pinned Flint config and a tiny Flint-shaped EXL3 pack written to disk.
//!
//! The tiny pack names its tensors by literal patterns, independent of `schema.rs`, so the
//! audit is checked against a second statement of the layout and not against itself.

use std::{collections::BTreeMap, path::Path};

use serde_json::{Value, json};
use tempfile::TempDir;

use super::{Audit, Geometry, Limits, audit};
use crate::{EngError, weights::exl3::source::Checkpoint};

/// `config.json` of 0xBakeer/TandemLLM-Flint-EXL3-4.0 at revision e7ab11242b0b.
pub(crate) const PINNED_CONFIG: &[u8] =
    include_bytes!("../../../../testdata/flint-exl3/config-4.0-e7ab1124.json");
/// SHA-256 of that file, also listed in the pack's own `SHA256SUMS`.
pub(crate) const PINNED_CONFIG_SHA256: &str =
    "9d776686ac4c8ace5aad0ba087220aea6c376f32315b06b95a8f48642b223858";
pub(crate) const MARKER: u32 = 0x83dc_d12d;
pub(crate) const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

pub(crate) struct Entry {
    pub(crate) dtype: &'static str,
    pub(crate) shape: Vec<u32>,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) struct PackBuilder {
    pub(crate) config: Value,
    pub(crate) tensors: BTreeMap<String, Entry>,
}

pub(crate) fn tiny_config() -> Value {
    json!({
        "model_type": "flint",
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "num_experts": 4,
        "num_hidden_layers": 3,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "vocab_size": 256,
        "quantization_config": {
            "quant_method": "exl3", "version": "1.6.2", "bits": 4.0, "head_bits": 5,
            "codebook": "mul1", "expert_interm_div": 128
        }
    })
}

fn repeat16(word: u16, count: usize) -> Vec<u8> {
    word.to_le_bytes().repeat(count)
}

impl PackBuilder {
    /// Three layers (0 dense, 1 and 2 sparse), four experts, hidden 128, vocabulary 256.
    pub(crate) fn tiny() -> Self {
        Self::tiny_with_bits(4)
    }

    /// As [`Self::tiny`] with `bits` bits per weight in every body projection.
    pub(crate) fn tiny_with_bits(bits: u32) -> Self {
        let mut pack = Self {
            config: tiny_config(),
            tensors: BTreeMap::new(),
        };
        pack.config["quantization_config"]["bits"] = json!(f64::from(bits));
        pack.plain("model.embed_tokens.weight", "BF16", &[256, 128], 0x3f80);
        pack.plain("model.norm.weight", "BF16", &[128], 0x3f80);
        pack.matrix("lm_head", 128, 256, 5);
        for layer in 0..3 {
            pack.layer(layer, bits);
        }
        pack
    }

    fn layer(&mut self, layer: usize, bits: u32) {
        let p = format!("model.layers.{layer}");
        self.plain(
            &format!("{p}.input_layernorm.weight"),
            "BF16",
            &[128],
            0x3f80,
        );
        self.plain(
            &format!("{p}.post_attention_layernorm.weight"),
            "BF16",
            &[128],
            0x3f80,
        );
        self.plain(
            &format!("{p}.self_attn.q_norm.weight"),
            "BF16",
            &[128],
            0x3f80,
        );
        self.plain(
            &format!("{p}.self_attn.k_norm.weight"),
            "BF16",
            &[128],
            0x3f80,
        );
        self.plain(
            &format!("{p}.self_attn.softmax_offset"),
            "F16",
            &[2],
            0x3c00,
        );
        self.matrix(&format!("{p}.self_attn.q_proj"), 128, 256, bits);
        self.matrix(&format!("{p}.self_attn.k_proj"), 128, 128, bits);
        self.matrix(&format!("{p}.self_attn.v_proj"), 128, 128, bits);
        self.matrix(&format!("{p}.self_attn.o_proj"), 256, 128, bits);
        if layer == 0 {
            self.matrix(&format!("{p}.mlp.gate_proj"), 128, 256, bits);
            self.matrix(&format!("{p}.mlp.up_proj"), 128, 256, bits);
            self.matrix(&format!("{p}.mlp.down_proj"), 256, 128, bits);
            return;
        }
        self.plain(&format!("{p}.mlp.gate.weight"), "F16", &[4, 128], 0);
        for projection in ["gate", "up", "down"] {
            self.matrix(
                &format!("{p}.mlp.shared_expert.{projection}_proj"),
                128,
                128,
                bits,
            );
            for expert in 0..4 {
                let prefix = format!("{p}.mlp.experts.{expert}.{projection}_proj");
                self.matrix(&prefix, 128, 128, bits);
            }
        }
    }

    pub(crate) fn plain(&mut self, name: &str, dtype: &'static str, shape: &[u32], word: u16) {
        let count: u32 = shape.iter().product();
        let bytes = if dtype == "F32" {
            1.0f32.to_le_bytes().repeat(count as usize)
        } else {
            repeat16(word, count as usize)
        };
        self.put(name, dtype, shape, bytes);
    }

    pub(crate) fn put(&mut self, name: &str, dtype: &'static str, shape: &[u32], bytes: Vec<u8>) {
        let entry = Entry {
            dtype,
            shape: shape.to_vec(),
            bytes,
        };
        self.tensors.insert(name.into(), entry);
    }

    /// A stored `[k, n]` matrix of `bits` bits per weight with unit scales and a zero trellis.
    pub(crate) fn matrix(&mut self, prefix: &str, k: u32, n: u32, bits: u32) {
        let words = 16 * bits;
        let trellis = vec![0u8; (k / 16 * (n / 16) * words * 2) as usize];
        self.put(
            &format!("{prefix}.trellis"),
            "I16",
            &[k / 16, n / 16, words],
            trellis,
        );
        self.plain(&format!("{prefix}.suh"), "F16", &[k], 0x3c00);
        self.plain(&format!("{prefix}.svh"), "F16", &[n], 0x3c00);
        self.put(
            &format!("{prefix}.mul1"),
            "I32",
            &[],
            MARKER.to_le_bytes().to_vec(),
        );
    }

    pub(crate) fn remove(&mut self, name: &str) {
        assert!(self.tensors.remove(name).is_some(), "no tensor {name}");
    }

    pub(crate) fn entry(&mut self, name: &str) -> &mut Entry {
        self.tensors
            .get_mut(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
    }

    /// Writes `config.json`, the index and two shards (the sorted names split in half).
    pub(crate) fn write(&self) -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&self.config).unwrap(),
        )
        .unwrap();
        let names: Vec<_> = self.tensors.keys().collect();
        let mut weight_map = serde_json::Map::new();
        for (shard, chunk) in SHARDS.iter().zip(names.chunks(names.len().div_ceil(2))) {
            self.write_shard(&dir, shard, chunk);
            for name in chunk {
                weight_map.insert((*name).clone(), json!(shard));
            }
        }
        let index = json!({"metadata": {"total_size": 0}, "weight_map": weight_map});
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        dir
    }

    fn write_shard(&self, dir: &TempDir, shard: &str, names: &[&String]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for name in names {
            let entry = &self.tensors[*name];
            let start = payload.len();
            payload.extend_from_slice(&entry.bytes);
            header.insert(
                (*name).clone(),
                json!({"dtype": entry.dtype, "shape": entry.shape, "data_offsets": [start, payload.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        std::fs::write(dir.path().join(shard), file).unwrap();
    }
}

/// Opens `dir` with the house reader and audits it against the geometry its own config states.
pub(crate) fn run(dir: &Path) -> Result<Audit, EngError> {
    let checkpoint = Checkpoint::open(dir)?;
    let geometry = Geometry::from_config(&checkpoint.model_config)?;
    audit(dir, &checkpoint, &geometry, &Limits::default())
}

/// Asserts a refusal whose message names `needle`; a stub or an unrelated error fails.
pub(crate) fn expect_refusal<T: std::fmt::Debug>(result: Result<T, EngError>, needle: &str) {
    match result {
        Err(EngError::Checkpoint(message) | EngError::MemoryPlan(message))
            if message.contains(needle) => {}
        other => panic!("expected a refusal naming {needle:?}, got {other:?}"),
    }
}

/// Builds the tiny pack, applies `mutate`, and returns the audit's refusal message.
pub(crate) fn refusal(mutate: impl FnOnce(&mut PackBuilder)) -> String {
    let mut pack = PackBuilder::tiny();
    mutate(&mut pack);
    let dir = pack.write();
    match run(dir.path()) {
        Err(EngError::Checkpoint(message) | EngError::MemoryPlan(message)) => message,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// As [`refusal`], asserting that the message contains `needle`.
pub(crate) fn refuses(mutate: impl FnOnce(&mut PackBuilder), needle: &str) {
    let message = refusal(mutate);
    assert!(message.contains(needle), "{needle:?} not in {message:?}");
}
