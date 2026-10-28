//! The tensor layout a Flint EXL3 pack must have, derived from the pack's own `config.json`.
//!
//! `Geometry::published` first runs `check_config`, the single source of the pinned Flint
//! dimensions, and then reads those same fields; nothing here restates them. `from_config`
//! reads the fields without the pin, so tests can describe a small pack.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::{EngError, checkpoint_json::UniqueValue, family::flint::check_config, tensor::Dtype};

// Upper bounds keep a hostile config from sizing the plan: the published pack has 49 layers
// and 256 experts.
const MAX_LAYERS: u64 = 128;
const MAX_EXPERTS: u64 = 1024;
const MAX_HEADS: u64 = 4096;
const MAX_WIDTH: u64 = 1 << 22;
const TILE: usize = 128;

#[derive(Debug, Clone)]
pub(crate) struct Geometry {
    pub(crate) hidden: usize,
    pub(crate) dense_ff: usize,
    pub(crate) moe_ff: usize,
    pub(crate) shared_ff: usize,
    pub(crate) experts: usize,
    pub(crate) layers: usize,
    pub(crate) dense_layers: Vec<usize>,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab: usize,
    pub(crate) body_bits: usize,
    pub(crate) head_bits: usize,
}

/// One quantized projection: `{prefix}.{trellis,suh,svh,mul1}` for an input `k` and output `n`.
#[derive(Debug, Clone)]
pub(crate) struct Module {
    pub(crate) prefix: String,
    pub(crate) role: &'static str,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) words: usize,
}

/// A tensor stored as plain 16-bit values. `bulk` tensors are checked by header only.
#[derive(Debug, Clone)]
pub(crate) struct Plain {
    pub(crate) name: String,
    pub(crate) role: &'static str,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<u32>,
    pub(crate) bulk: bool,
}

#[derive(Debug, Default)]
pub(crate) struct Plan {
    pub(crate) modules: Vec<Module>,
    pub(crate) plain: Vec<Plain>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Trellis,
    Suh,
    Svh,
    Mul1,
    Plain,
    Bulk,
}

/// One expected stored tensor.
#[derive(Debug, Clone)]
pub(crate) struct Spec {
    pub(crate) name: String,
    pub(crate) role: &'static str,
    pub(crate) kind: Kind,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<u32>,
}

impl Spec {
    pub(crate) fn bytes(&self) -> u64 {
        let elements: u64 = self.shape.iter().map(|&d| u64::from(d)).product();
        elements * self.dtype.byte_width()
    }
}

impl Module {
    pub(crate) fn specs(&self) -> [Spec; 4] {
        let spec = |suffix: &str, kind, dtype, shape: Vec<u32>| Spec {
            name: format!("{}.{suffix}", self.prefix),
            role: self.role,
            kind,
            dtype,
            shape,
        };
        let tiles = |extent: usize| (extent / 16) as u32;
        [
            spec(
                "trellis",
                Kind::Trellis,
                Dtype::I16,
                vec![tiles(self.k), tiles(self.n), self.words as u32],
            ),
            spec("suh", Kind::Suh, Dtype::F16, vec![self.k as u32]),
            spec("svh", Kind::Svh, Dtype::F16, vec![self.n as u32]),
            spec("mul1", Kind::Mul1, Dtype::I32, vec![]),
        ]
    }
}

impl Plan {
    pub(crate) fn specs(&self) -> Vec<Spec> {
        let plain = self.plain.iter().map(|p| Spec {
            name: p.name.clone(),
            role: p.role,
            kind: if p.bulk { Kind::Bulk } else { Kind::Plain },
            dtype: p.dtype,
            shape: p.shape.clone(),
        });
        self.modules
            .iter()
            .flat_map(Module::specs)
            .chain(plain)
            .collect()
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.modules.len() * 4 + self.plain.len()
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.specs().iter().map(Spec::bytes).sum()
    }

    fn matrix(&mut self, prefix: String, role: &'static str, (k, n): (usize, usize), bits: usize) {
        self.modules.push(Module {
            prefix,
            role,
            k,
            n,
            words: 16 * bits,
        });
    }

    fn half(&mut self, name: String, role: &'static str, dtype: Dtype, shape: &[usize]) {
        self.push_plain(name, role, dtype, shape, false);
    }

    /// A tensor too large to read in the audit (the embedding): header checks only.
    fn bulk(&mut self, name: String, role: &'static str, dtype: Dtype, shape: &[usize]) {
        self.push_plain(name, role, dtype, shape, true);
    }

    fn push_plain(
        &mut self,
        name: String,
        role: &'static str,
        dtype: Dtype,
        shape: &[usize],
        bulk: bool,
    ) {
        let shape = shape.iter().map(|&d| d as u32).collect();
        self.plain.push(Plain {
            name,
            role,
            dtype,
            shape,
            bulk,
        });
    }
}

impl Geometry {
    /// Pinned Flint dimensions (`check_config`), then the same fields as a geometry.
    pub(crate) fn published(model_config: &[u8]) -> Result<Self, EngError> {
        check_config(model_config)?;
        Self::from_config(model_config)
    }

    /// Reads the layout fields without checking them against the published Flint pin.
    pub(crate) fn from_config(model_config: &[u8]) -> Result<Self, EngError> {
        let root = serde_json::from_slice::<UniqueValue>(model_config)
            .map_err(|error| invalid(&format!("config JSON: {error}")))?
            .0;
        let quant = root
            .get("quantization_config")
            .ok_or_else(|| invalid("quantization_config is missing"))?;
        let layers = integer(&root, "num_hidden_layers", MAX_LAYERS)?;
        let heads = integer(&root, "num_attention_heads", MAX_HEADS)?;
        let kv_heads = integer(&root, "num_key_value_heads", MAX_HEADS)?;
        let head_dim = integer(&root, "head_dim", MAX_HEADS)?;
        // Every non-listed layer is sparse only when the step is 1.
        integer(&root, "decoder_sparse_step", 1)?;
        if heads % kv_heads != 0 {
            return Err(invalid(
                "num_key_value_heads must divide num_attention_heads",
            ));
        }
        let geometry = Self {
            hidden: tiled(&root, "hidden_size", MAX_WIDTH)?,
            dense_ff: tiled(&root, "intermediate_size", MAX_WIDTH)?,
            moe_ff: tiled(&root, "moe_intermediate_size", MAX_WIDTH)?,
            shared_ff: tiled(&root, "shared_expert_intermediate_size", MAX_WIDTH)?,
            experts: integer(&root, "num_experts", MAX_EXPERTS)?,
            layers,
            dense_layers: dense_layers(&root, layers)?,
            heads,
            kv_heads,
            head_dim,
            vocab: tiled(&root, "vocab_size", MAX_WIDTH)?,
            body_bits: bits(quant, "bits")?,
            head_bits: bits(quant, "head_bits")?,
        };
        if (heads * head_dim) % TILE != 0 || (kv_heads * head_dim) % TILE != 0 {
            return Err(invalid(
                "head_dim must make the attention widths multiples of 128",
            ));
        }
        Ok(geometry)
    }

    pub(crate) fn plan(&self) -> Plan {
        let mut plan = Plan::default();
        let embed = [self.vocab, self.hidden];
        plan.bulk(
            "model.embed_tokens.weight".into(),
            "embed_tokens",
            Dtype::Bf16,
            &embed,
        );
        plan.half(
            "model.norm.weight".into(),
            "model.norm",
            Dtype::Bf16,
            &[self.hidden],
        );
        let head = (self.hidden, self.vocab);
        plan.matrix("lm_head".into(), "lm_head", head, self.head_bits);
        for layer in 0..self.layers {
            self.layer(&mut plan, layer);
        }
        plan
    }

    fn layer(&self, plan: &mut Plan, layer: usize) {
        let p = format!("model.layers.{layer}");
        let norm = |name: &str| format!("{p}.{name}.weight");
        plan.half(
            norm("input_layernorm"),
            "input_layernorm",
            Dtype::Bf16,
            &[self.hidden],
        );
        plan.half(
            norm("post_attention_layernorm"),
            "post_attention_layernorm",
            Dtype::Bf16,
            &[self.hidden],
        );
        plan.half(
            norm("self_attn.q_norm"),
            "q_norm",
            Dtype::Bf16,
            &[self.head_dim],
        );
        plan.half(
            norm("self_attn.k_norm"),
            "k_norm",
            Dtype::Bf16,
            &[self.head_dim],
        );
        plan.half(
            format!("{p}.self_attn.softmax_offset"),
            "softmax_offset",
            Dtype::F16,
            &[self.heads],
        );
        let (q, kv) = (self.heads * self.head_dim, self.kv_heads * self.head_dim);
        let bits = self.body_bits;
        for (name, role, shape) in [
            ("q_proj", "self_attn.q_proj", (self.hidden, q)),
            ("k_proj", "self_attn.k_proj", (self.hidden, kv)),
            ("v_proj", "self_attn.v_proj", (self.hidden, kv)),
            ("o_proj", "self_attn.o_proj", (q, self.hidden)),
        ] {
            plan.matrix(format!("{p}.self_attn.{name}"), role, shape, bits);
        }
        if self.dense_layers.contains(&layer) {
            self.dense_mlp(plan, &p);
        } else {
            self.sparse_mlp(plan, &p);
        }
    }

    fn dense_mlp(&self, plan: &mut Plan, p: &str) {
        let (h, f, bits) = (self.hidden, self.dense_ff, self.body_bits);
        for (name, role, shape) in [
            ("gate_proj", "mlp.gate_proj", (h, f)),
            ("up_proj", "mlp.up_proj", (h, f)),
            ("down_proj", "mlp.down_proj", (f, h)),
        ] {
            plan.matrix(format!("{p}.mlp.{name}"), role, shape, bits);
        }
    }

    fn sparse_mlp(&self, plan: &mut Plan, p: &str) {
        let (h, bits) = (self.hidden, self.body_bits);
        plan.half(
            format!("{p}.mlp.gate.weight"),
            "router",
            Dtype::F16,
            &[self.experts, h],
        );
        let projections = [
            (
                "gate_proj",
                "shared_expert.gate_proj",
                "experts.gate_proj",
                false,
            ),
            ("up_proj", "shared_expert.up_proj", "experts.up_proj", false),
            (
                "down_proj",
                "shared_expert.down_proj",
                "experts.down_proj",
                true,
            ),
        ];
        for (name, shared_role, expert_role, down) in projections {
            let shape = |f: usize| if down { (f, h) } else { (h, f) };
            plan.matrix(
                format!("{p}.mlp.shared_expert.{name}"),
                shared_role,
                shape(self.shared_ff),
                bits,
            );
            for expert in 0..self.experts {
                let prefix = format!("{p}.mlp.experts.{expert}.{name}");
                plan.matrix(prefix, expert_role, shape(self.moe_ff), bits);
            }
        }
    }

    /// Matrices whose trellis payload the audit admits through `Matrix::load`: attention and a
    /// dense MLP of the first layers, and the first and last experts of the first and last
    /// sparse layers. The head's trellis is the largest tensor and stays header-only.
    pub(crate) fn sample(&self) -> BTreeSet<String> {
        let mut wanted = BTreeSet::new();
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            wanted.insert(format!("model.layers.0.self_attn.{name}"));
        }
        if let Some(dense) = self.dense_layers.first() {
            wanted.insert(format!("model.layers.{dense}.mlp.down_proj"));
        }
        let mut sparse = (0..self.layers).filter(|l| !self.dense_layers.contains(l));
        let last_expert = self.experts - 1;
        if let Some(first) = sparse.next() {
            add_sparse(&mut wanted, first, &[0, last_expert], true);
        }
        if let Some(last) = sparse.next_back() {
            add_sparse(&mut wanted, last, &[last_expert], false);
        }
        wanted
    }
}

fn add_sparse(wanted: &mut BTreeSet<String>, layer: usize, experts: &[usize], shared: bool) {
    for name in ["gate_proj", "up_proj", "down_proj"] {
        if shared {
            wanted.insert(format!("model.layers.{layer}.mlp.shared_expert.{name}"));
        }
        for expert in experts {
            wanted.insert(format!("model.layers.{layer}.mlp.experts.{expert}.{name}"));
        }
    }
}

fn integer(root: &Value, name: &str, max: u64) -> Result<usize, EngError> {
    root.get(name)
        .and_then(Value::as_u64)
        .filter(|value| (1..=max).contains(value))
        .map(|value| value as usize)
        .ok_or_else(|| invalid(&format!("{name} must be an integer from 1 to {max}")))
}

/// An integer that is also a whole number of 128-wide transform blocks.
fn tiled(root: &Value, name: &str, max: u64) -> Result<usize, EngError> {
    let value = integer(root, name, max)?;
    if value % TILE != 0 {
        return Err(invalid(&format!("{name} must be a multiple of {TILE}")));
    }
    Ok(value)
}

fn dense_layers(root: &Value, layers: usize) -> Result<Vec<usize>, EngError> {
    let bad = || invalid("mlp_only_layers must list distinct layer indices");
    let listed = root
        .get("mlp_only_layers")
        .and_then(Value::as_array)
        .ok_or_else(bad)?;
    let indices = listed
        .iter()
        .map(|v| v.as_u64().map(|i| i as usize).filter(|&i| i < layers))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(bad)?;
    if indices.iter().collect::<BTreeSet<_>>().len() != indices.len() {
        return Err(bad());
    }
    Ok(indices)
}

/// Bits per weight as an integer, which may be written `4.0`.
fn bits(quant: &Value, name: &str) -> Result<usize, EngError> {
    quant
        .get(name)
        .and_then(Value::as_f64)
        .filter(|v| v.fract() == 0.0 && (1.0..=8.0).contains(v))
        .map(|v| v as usize)
        .ok_or_else(|| {
            invalid(&format!(
                "quantization_config.{name} must be an integer from 1 to 8"
            ))
        })
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 Flint config: {message}"))
}
