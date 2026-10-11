//! Builds the Flash-Next model shape and weights from a streamed checkpoint.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::{
    deltanet::{DeltaShape, DeltaWeights},
    hyper::{HyperShape, HyperWeights},
    moe::{MoeShape, MoeWeights},
    ple::{PleDims, PleShape, PleWeights},
    qsa::{QsaDims, QsaShape, QsaWeights},
    step::{AttnWeights, FlashLayerWeights, ModelShape, ModelWeights},
    stored::{self, StoredWeight},
};
use crate::{
    EngError,
    checkpoint_json::UniqueValue,
    family::quantization::{ModuleQuantization, QuantizationMap},
    kernels::{QuantizationMode, QuantizationSpec},
    weights::StreamedCheckpoint,
};

/// Prefix of every language-model tensor name.
const MODEL: &str = "language_model.model.";

/// Model shape and per-layer attention kind (true for QSA) read from `config.json`.
pub(crate) fn model_shape(config: &[u8]) -> Result<(ModelShape, Vec<bool>), EngError> {
    let dims = Dims::read(config)?;
    let shape = dims.model_shape()?;
    Ok((shape, dims.kinds))
}

/// Configuration values that the loader reads, parsed once from `text_config`.
struct Dims {
    hidden: u32,
    eps: f32,
    branches: u32,
    rank: u32,
    width: u32,
    vocab: u32,
    kinds: Vec<bool>,
    heads_q: u32,
    heads_kv: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    idx_heads: u32,
    idx_dim: u32,
    budget: u32,
    ratio: u32,
    key_heads: u32,
    key_dim: u32,
    value_heads: u32,
    value_dim: u32,
    conv_kernel: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    shared_inter: u32,
    renorm: bool,
    ngram: u32,
    heads_per_ngram: u32,
    ple_heads: u32,
    ple_head_dim: u32,
    ple_embed: u32,
    ple_conv: u32,
    ple_layer: usize,
    ngram_parts: Option<u32>,
    eos: u32,
}

/// Reports a config field that is absent or has the wrong type.
fn invalid(name: &str) -> EngError {
    EngError::Checkpoint(format!("config field {name} is missing or invalid"))
}

/// Reports a derived dimension that overflows its integer type.
fn overflow() -> EngError {
    EngError::Checkpoint("config dimension overflows".into())
}

/// Multiplies configuration dimensions with overflow checks.
fn product(factors: &[u32]) -> Result<u32, EngError> {
    factors
        .iter()
        .try_fold(1u32, |n, &factor| n.checked_mul(factor))
        .ok_or_else(overflow)
}

/// Looks up a dotted path such as `rope_parameters.rope_theta`.
fn lookup<'a>(text: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    let mut parts = path.split('.');
    let first = parts.next()?;
    parts.try_fold(text.get(first)?, |value, part| value.get(part))
}

/// Reads an unsigned integer field that fits `u32`.
fn number(text: &Map<String, Value>, path: &str) -> Result<u32, EngError> {
    lookup(text, path)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| invalid(path))
}

/// Reads a numeric field as f64.
fn real(text: &Map<String, Value>, path: &str) -> Result<f64, EngError> {
    lookup(text, path)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(path))
}

impl Dims {
    /// Parses and checks every configuration value that the loader reads.
    fn read(config: &[u8]) -> Result<Self, EngError> {
        let root = serde_json::from_slice::<UniqueValue>(config)
            .map_err(|error| EngError::Checkpoint(format!("configuration JSON: {error}")))?
            .0;
        let text = root
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("text_config"))?;

        let hidden = number(text, "hidden_size")?;
        let eps = real(text, "rms_norm_eps")? as f32;
        let branches = number(text, "hc_count")?;
        let rank = number(text, "hc_lowrank")?;
        let width = product(&[branches, hidden])?;
        let vocab = number(text, "vocab_size")?;
        let layers = number(text, "num_hidden_layers")?;
        let layer_types = lookup(text, "layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("layer_types"))?;
        if layer_types.len() != layers as usize {
            return Err(invalid("layer_types"));
        }
        let kinds = layer_types
            .iter()
            .map(|kind| {
                kind.as_str()
                    .map(|name| name != "linear_attention")
                    .ok_or_else(|| invalid("layer_types"))
            })
            .collect::<Result<Vec<bool>, EngError>>()?;

        let heads_q = number(text, "num_attention_heads")?;
        let heads_kv = number(text, "num_key_value_heads")?;
        let head_dim = number(text, "head_dim")?;
        let theta = real(text, "rope_parameters.rope_theta")? as f32;
        let partial = real(text, "rope_parameters.partial_rotary_factor")?;
        let rotary_dim = (f64::from(head_dim) * partial) as u32;
        let idx_heads = number(text, "indexer_n_heads")?;
        let idx_dim = number(text, "indexer_head_dim")?;
        let budget = number(text, "indexer_budget")?;
        let ratio = number(text, "indexer_compress_ratio")?;
        let key_heads = number(text, "linear_num_key_heads")?;
        let key_dim = number(text, "linear_key_head_dim")?;
        let value_heads = number(text, "linear_num_value_heads")?;
        let value_dim = number(text, "linear_value_head_dim")?;
        let conv_kernel = number(text, "linear_conv_kernel_dim")?;
        let experts = number(text, "num_experts")?;
        let top_k = number(text, "num_experts_per_tok")?;
        let moe_inter = number(text, "moe_intermediate_size")?;
        let shared_inter = number(text, "shared_expert_intermediate_size")?;
        let renorm = match lookup(text, "norm_topk_prob") {
            None => true,
            Some(value) => value.as_bool().ok_or_else(|| invalid("norm_topk_prob"))?,
        };
        let ngram = number(text, "ngram_size")?;
        let heads_per_ngram = number(text, "heads_per_ngram")?;
        let ple_embed = number(text, "ple_embed_dim")?;
        let ple_conv = number(text, "ple_conv_kernel_size")?;
        let first_layer = lookup(text, "ple_layer_ids")
            .and_then(Value::as_array)
            .and_then(|ids| ids.first())
            .and_then(Value::as_u64)
            .filter(|&id| id >= 1)
            .ok_or_else(|| invalid("ple_layer_ids"))?;
        let ple_layer = usize::try_from(first_layer - 1).map_err(|_| invalid("ple_layer_ids"))?;
        if ple_layer >= layers as usize {
            return Err(EngError::Checkpoint(
                "config field ple_layer_ids is out of range".into(),
            ));
        }
        let ngram_parts = lookup(text, "split_ngram_parts")
            .map(|_| number(text, "split_ngram_parts"))
            .transpose()?;
        let eos = number(text, "eos_token_id")?;

        let ple_heads = ngram
            .checked_sub(1)
            .and_then(|n| n.checked_mul(heads_per_ngram))
            .ok_or_else(overflow)?;
        if ple_embed.checked_rem(ple_heads) != Some(0) {
            return Err(EngError::Checkpoint(
                "ple embedding width must divide by the ple head count".into(),
            ));
        }
        let ple_head_dim = ple_embed / ple_heads;

        Ok(Self {
            hidden,
            eps,
            branches,
            rank,
            width,
            vocab,
            kinds,
            heads_q,
            heads_kv,
            head_dim,
            rotary_dim,
            theta,
            idx_heads,
            idx_dim,
            budget,
            ratio,
            key_heads,
            key_dim,
            value_heads,
            value_dim,
            conv_kernel,
            experts,
            top_k,
            moe_inter,
            shared_inter,
            renorm,
            ngram,
            heads_per_ngram,
            ple_heads,
            ple_head_dim,
            ple_embed,
            ple_conv,
            ple_layer,
            ngram_parts,
            eos,
        })
    }

    /// Builds the checked block shapes from these values.
    fn model_shape(&self) -> Result<ModelShape, EngError> {
        Ok(ModelShape {
            hyper: HyperShape::new(self.branches, self.hidden, self.rank, self.eps)?,
            delta: DeltaShape::new(
                self.key_heads,
                self.key_dim,
                self.value_heads,
                self.value_dim,
                self.conv_kernel,
                self.hidden,
                self.eps,
            )?,
            qsa: QsaShape::new(QsaDims {
                heads_q: self.heads_q,
                heads_kv: self.heads_kv,
                head_dim: self.head_dim,
                idx_heads: self.idx_heads,
                idx_dim: self.idx_dim,
                ratio: self.ratio,
                budget: self.budget,
                rotary_dim: self.rotary_dim,
                theta: self.theta,
                hidden: self.hidden,
                eps: self.eps,
            })?,
            moe: MoeShape::new(
                self.experts,
                self.top_k,
                self.moe_inter,
                self.shared_inter,
                self.hidden,
                self.renorm,
            )?,
            ple: PleShape::new(PleDims {
                branches: self.branches,
                hidden: self.hidden,
                ngram: self.ngram,
                heads_per_ngram: self.heads_per_ngram,
                head_dim: self.ple_head_dim,
                conv_kernel: self.ple_conv,
                eos: self.eos,
                eps: self.eps,
            })?,
            ple_layer: self.ple_layer,
            vocab: self.vocab as usize,
        })
    }
}

/// Owned hyper-connection weights.
struct OwnedHyper {
    norm: Vec<f32>,
    down: StoredWeight,
    up: StoredWeight,
    inject: Option<StoredWeight>,
}

impl OwnedHyper {
    /// Borrows these weights for `step`.
    fn view(&self) -> HyperWeights<'_, StoredWeight> {
        HyperWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: self.inject.as_ref(),
        }
    }
}

/// Owned DeltaNet weights.
struct OwnedDelta {
    qkv: StoredWeight,
    z: StoredWeight,
    b: StoredWeight,
    a: StoredWeight,
    out: StoredWeight,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

impl OwnedDelta {
    /// Borrows these weights for `step`.
    fn view(&self) -> DeltaWeights<'_, StoredWeight> {
        DeltaWeights {
            qkv: &self.qkv,
            z: &self.z,
            b: &self.b,
            a: &self.a,
            out: &self.out,
            conv: &self.conv,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
        }
    }
}

/// Owned QSA weights.
struct OwnedQsa {
    q: StoredWeight,
    k: StoredWeight,
    v: StoredWeight,
    o: StoredWeight,
    idx: StoredWeight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    idx_q_norm: Vec<f32>,
    idx_k_norm: Vec<f32>,
}

impl OwnedQsa {
    /// Borrows these weights for `step`.
    fn view(&self) -> QsaWeights<'_, StoredWeight> {
        QsaWeights {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            o: &self.o,
            idx: &self.idx,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            idx_q_norm: &self.idx_q_norm,
            idx_k_norm: &self.idx_k_norm,
        }
    }
}

/// Owned attention weights of one layer.
enum OwnedAttn {
    /// A DeltaNet layer.
    Delta(OwnedDelta),
    /// A QSA layer.
    Qsa(OwnedQsa),
}

/// Owned MoE weights.
struct OwnedMoe {
    router: Vec<f32>,
    gate: StoredWeight,
    up: StoredWeight,
    down: StoredWeight,
    shared_gate: StoredWeight,
    shared_up: StoredWeight,
    shared_down: StoredWeight,
    shared_scalar: StoredWeight,
}

impl OwnedMoe {
    /// Borrows these weights for `step`.
    fn view(&self) -> MoeWeights<'_, StoredWeight> {
        MoeWeights {
            router: &self.router,
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
            shared_gate: &self.shared_gate,
            shared_up: &self.shared_up,
            shared_down: &self.shared_down,
            shared_scalar: &self.shared_scalar,
        }
    }
}

/// Owned PLE weights.
struct OwnedPle {
    multipliers: Vec<i64>,
    vocab: Vec<i64>,
    offsets: Vec<i64>,
    shards: Vec<StoredWeight>,
    shard_rows: Vec<u64>,
    weight_scale: f32,
    key: StoredWeight,
    value: StoredWeight,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
}

impl OwnedPle {
    /// Borrows these weights for `step`.
    fn view(&self) -> PleWeights<'_, StoredWeight> {
        PleWeights {
            multipliers: &self.multipliers,
            vocab: &self.vocab,
            offsets: &self.offsets,
            shards: &self.shards,
            shard_rows: &self.shard_rows,
            weight_scale: self.weight_scale,
            key: &self.key,
            value: &self.value,
            norm_key: &self.norm_key,
            norm_query: &self.norm_query,
            norm_conv: &self.norm_conv,
            conv: &self.conv,
        }
    }
}

/// Owned weights of one decoder layer.
struct OwnedLayer {
    attn_hyper: OwnedHyper,
    attn: OwnedAttn,
    mlp_hyper: OwnedHyper,
    moe: OwnedMoe,
}

/// Owned Flash-Next weights.
pub(crate) struct FlashWeights {
    shape: ModelShape,
    embed: StoredWeight,
    lm_head: StoredWeight,
    mixer: OwnedHyper,
    layers: Vec<OwnedLayer>,
    ple: OwnedPle,
    keys: Vec<String>,
}

/// Reads checkpoint tensors through the quantization map and records each key read.
struct Loader<'a> {
    ckpt: &'a StreamedCheckpoint,
    quant: &'a QuantizationMap,
    keys: BTreeSet<String>,
}

impl Loader<'_> {
    /// Returns the quantization of a module that must be quantized.
    fn spec(&self, stem: &str) -> Result<QuantizationSpec, EngError> {
        match self.quant.resolve(stem)? {
            ModuleQuantization::Quantized(spec) => Ok(spec),
            ModuleQuantization::Unquantized => Err(EngError::Checkpoint(format!(
                "module {stem} is not quantized"
            ))),
        }
    }

    /// Records the keys of one packed or sliced module.
    fn record_module(&mut self, stem: &str, spec: QuantizationSpec) {
        self.keys.insert(format!("{stem}.weight"));
        self.keys.insert(format!("{stem}.scales"));
        if spec.mode == QuantizationMode::Affine {
            self.keys.insert(format!("{stem}.biases"));
        }
    }

    /// Loads the packed matrix `stem` with logical shape `[out, input]`.
    fn packed(&mut self, stem: &str, out: u32, input: u32) -> Result<StoredWeight, EngError> {
        let spec = self.spec(stem)?;
        let weight = stored::packed(self.ckpt, stem, spec, out, input)?;
        self.record_module(stem, spec);
        Ok(StoredWeight::Packed(Box::new(weight)))
    }

    /// Describes the packed stack or table `stem` without reading it.
    fn sliced(&mut self, stem: &str, slice_rows: u32, cols: u32) -> Result<StoredWeight, EngError> {
        let spec = self.spec(stem)?;
        let weight = stored::sliced(self.ckpt, stem, spec, slice_rows, cols)?;
        self.record_module(stem, spec);
        Ok(StoredWeight::Sliced(weight))
    }

    /// Reads a host f32 tensor of `len` elements.
    fn host_f32(&mut self, key: &str, len: usize) -> Result<Vec<f32>, EngError> {
        let values = stored::host_f32(self.ckpt, key, len)?;
        self.keys.insert(key.to_string());
        Ok(values)
    }

    /// Reads a host i64 tensor of `len` elements.
    fn host_i64(&mut self, key: &str, len: usize) -> Result<Vec<i64>, EngError> {
        let values = stored::host_i64(self.ckpt, key, len)?;
        self.keys.insert(key.to_string());
        Ok(values)
    }
}

/// Loads one hyper-connection. The mixer has no inject projection.
fn hyper(
    loader: &mut Loader<'_>,
    prefix: &str,
    dims: &Dims,
    inject: bool,
) -> Result<OwnedHyper, EngError> {
    let norm = loader.host_f32(&format!("{prefix}hc_norm.weight"), dims.width as usize)?;
    let down = loader.packed(
        &format!("{prefix}input_mix_weight_down"),
        dims.rank,
        dims.width,
    )?;
    let up = loader.packed(
        &format!("{prefix}input_mix_weight_up"),
        dims.width,
        dims.rank,
    )?;
    let inject = if inject {
        Some(loader.packed(
            &format!("{prefix}block_inject_weight"),
            dims.branches,
            dims.width,
        )?)
    } else {
        None
    };
    Ok(OwnedHyper {
        norm,
        down,
        up,
        inject,
    })
}

/// Loads one DeltaNet layer from `prefix`, which ends with `linear_attn.`.
fn delta(loader: &mut Loader<'_>, prefix: &str, dims: &Dims) -> Result<OwnedDelta, EngError> {
    let key_width = product(&[dims.key_heads, dims.key_dim])?;
    let value_width = product(&[dims.value_heads, dims.value_dim])?;
    let conv_dim = product(&[2, key_width])?
        .checked_add(value_width)
        .ok_or_else(overflow)?;
    let qkv = loader.packed(&format!("{prefix}in_proj_qkv"), conv_dim, dims.hidden)?;
    let z = loader.packed(&format!("{prefix}in_proj_z"), value_width, dims.hidden)?;
    let b = loader.packed(&format!("{prefix}in_proj_b"), dims.value_heads, dims.hidden)?;
    let a = loader.packed(&format!("{prefix}in_proj_a"), dims.value_heads, dims.hidden)?;
    let out = loader.packed(&format!("{prefix}out_proj"), dims.hidden, value_width)?;
    let conv = loader.host_f32(
        &format!("{prefix}conv1d.weight"),
        product(&[conv_dim, dims.conv_kernel])? as usize,
    )?;
    let a_log = loader.host_f32(&format!("{prefix}A_log"), dims.value_heads as usize)?;
    let dt_bias = loader.host_f32(&format!("{prefix}dt_bias"), dims.value_heads as usize)?;
    let norm = loader.host_f32(&format!("{prefix}norm.weight"), dims.value_dim as usize)?;
    Ok(OwnedDelta {
        qkv,
        z,
        b,
        a,
        out,
        conv,
        a_log,
        dt_bias,
        norm,
    })
}

/// Loads one QSA layer from `prefix`, which ends with `self_attn.`.
fn qsa(loader: &mut Loader<'_>, prefix: &str, dims: &Dims) -> Result<OwnedQsa, EngError> {
    let q_width = product(&[dims.heads_q, dims.head_dim])?;
    let kv_width = product(&[dims.heads_kv, dims.head_dim])?;
    let idx_rows = dims.idx_heads.checked_add(1).ok_or_else(overflow)?;
    let q = loader.packed(
        &format!("{prefix}q_proj"),
        product(&[2, q_width])?,
        dims.hidden,
    )?;
    let k = loader.packed(&format!("{prefix}k_proj"), kv_width, dims.hidden)?;
    let v = loader.packed(&format!("{prefix}v_proj"), kv_width, dims.hidden)?;
    let o = loader.packed(&format!("{prefix}o_proj"), dims.hidden, q_width)?;
    let idx = loader.packed(
        &format!("{prefix}indexer.index_qk_proj"),
        product(&[idx_rows, dims.idx_dim])?,
        dims.hidden,
    )?;
    let q_norm = loader.host_f32(&format!("{prefix}q_norm.weight"), dims.head_dim as usize)?;
    let k_norm = loader.host_f32(&format!("{prefix}k_norm.weight"), dims.head_dim as usize)?;
    let idx_q_norm = loader.host_f32(
        &format!("{prefix}indexer.q_layernorm.weight"),
        dims.idx_dim as usize,
    )?;
    let idx_k_norm = loader.host_f32(
        &format!("{prefix}indexer.k_layernorm.weight"),
        dims.idx_dim as usize,
    )?;
    Ok(OwnedQsa {
        q,
        k,
        v,
        o,
        idx,
        q_norm,
        k_norm,
        idx_q_norm,
        idx_k_norm,
    })
}

/// Loads one MoE block from `prefix`, which ends with `mlp.`.
fn moe(loader: &mut Loader<'_>, prefix: &str, dims: &Dims) -> Result<OwnedMoe, EngError> {
    let router = loader.host_f32(
        &format!("{prefix}gate.weight"),
        product(&[dims.experts, dims.hidden])? as usize,
    )?;
    let gate = loader.sliced(
        &format!("{prefix}switch_mlp.gate_proj"),
        dims.moe_inter,
        dims.hidden,
    )?;
    let up = loader.sliced(
        &format!("{prefix}switch_mlp.up_proj"),
        dims.moe_inter,
        dims.hidden,
    )?;
    let down = loader.sliced(
        &format!("{prefix}switch_mlp.down_proj"),
        dims.hidden,
        dims.moe_inter,
    )?;
    let shared_gate = loader.packed(
        &format!("{prefix}shared_expert.gate_proj"),
        dims.shared_inter,
        dims.hidden,
    )?;
    let shared_up = loader.packed(
        &format!("{prefix}shared_expert.up_proj"),
        dims.shared_inter,
        dims.hidden,
    )?;
    let shared_down = loader.packed(
        &format!("{prefix}shared_expert.down_proj"),
        dims.hidden,
        dims.shared_inter,
    )?;
    let shared_scalar = loader.packed(&format!("{prefix}shared_expert_gate"), 1, dims.hidden)?;
    Ok(OwnedMoe {
        router,
        gate,
        up,
        down,
        shared_gate,
        shared_up,
        shared_down,
        shared_scalar,
    })
}

/// Loads the PLE block of the layer whose prefix is `layer`.
fn ple(loader: &mut Loader<'_>, layer: &str, dims: &Dims) -> Result<OwnedPle, EngError> {
    let prefix = format!("{layer}ple.");
    let ngram = dims.ngram as usize;
    let heads = dims.ple_heads as usize;
    let width = dims.width;
    let multipliers =
        loader.host_i64(&format!("{prefix}ple_embedding.layer_multipliers"), ngram)?;
    let vocab = loader.host_i64(
        &format!("{prefix}ple_embedding.ngram_heads_vocab_sizes"),
        heads,
    )?;
    let offsets = loader.host_i64(&format!("{prefix}ple_embedding.ngram_heads_offsets"), heads)?;
    let weight_scale = loader
        .host_f32(
            &format!("{prefix}ple_embedding.ngram_embedding.weight_scale"),
            1,
        )?
        .first()
        .copied()
        .ok_or_else(|| invalid("ple_embedding.ngram_embedding.weight_scale"))?;

    let mut shards = Vec::new();
    let mut shard_rows = Vec::new();
    for index in 0u32.. {
        let stem = format!("{prefix}ple_embedding.ngram_embedding.shards.{index}");
        if !loader.ckpt.contains(&format!("{stem}.weight")) {
            break;
        }
        let rows = loader
            .ckpt
            .stored(&format!("{stem}.weight"))
            .and_then(|(_, shape)| shape.first().copied())
            .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {stem}.weight")))?;
        shards.push(loader.sliced(&stem, 1, dims.ple_head_dim)?);
        shard_rows.push(u64::from(rows));
    }
    if shards.is_empty() {
        return Err(EngError::Checkpoint(format!(
            "missing tensor {prefix}ple_embedding.ngram_embedding.shards.0.weight"
        )));
    }
    if dims
        .ngram_parts
        .is_some_and(|parts| usize::try_from(parts).ok() != Some(shards.len()))
    {
        return Err(EngError::Checkpoint(
            "ple shard count differs from split_ngram_parts".into(),
        ));
    }
    let total = shard_rows
        .iter()
        .try_fold(0u64, |sum, &rows| sum.checked_add(rows))
        .ok_or_else(overflow)?;
    for (head, (&offset, &size)) in offsets.iter().zip(vocab.iter()).enumerate() {
        let covered =
            offset >= 0 && size > 0 && i128::from(offset) + i128::from(size) <= i128::from(total);
        if !covered {
            return Err(EngError::Checkpoint(format!(
                "ple shards do not cover head {head}"
            )));
        }
    }

    let key = loader.packed(&format!("{prefix}key_proj"), width, dims.ple_embed)?;
    let value = loader.packed(&format!("{prefix}value_proj"), dims.hidden, dims.ple_embed)?;
    let norm_key = loader.host_f32(&format!("{prefix}norm_key.weight"), width as usize)?;
    let norm_query = loader.host_f32(&format!("{prefix}norm_query.weight"), width as usize)?;
    let norm_conv = loader.host_f32(&format!("{prefix}norm_conv.weight"), width as usize)?;
    let conv = loader.host_f32(
        &format!("{prefix}conv1d.weight"),
        product(&[width, dims.ple_conv])? as usize,
    )?;
    Ok(OwnedPle {
        multipliers,
        vocab,
        offsets,
        shards,
        shard_rows,
        weight_scale,
        key,
        value,
        norm_key,
        norm_query,
        norm_conv,
        conv,
    })
}

impl FlashWeights {
    /// Loads every language-model tensor that the forward pass reads.
    pub(crate) fn load(ckpt: &StreamedCheckpoint, config: &[u8]) -> Result<Self, EngError> {
        let dims = Dims::read(config)?;
        let shape = dims.model_shape()?;
        let quant = QuantizationMap::from_json(config)?;
        let mut loader = Loader {
            ckpt,
            quant: &quant,
            keys: BTreeSet::new(),
        };

        let embed = loader.sliced(&format!("{MODEL}embed_tokens"), 1, dims.hidden)?;
        let lm_head = loader.packed("language_model.lm_head", dims.vocab, dims.hidden)?;
        let mixer = hyper(
            &mut loader,
            &format!("{MODEL}hyper_connection_mixer."),
            &dims,
            false,
        )?;

        let mut layers = Vec::with_capacity(dims.kinds.len());
        for (index, is_qsa) in dims.kinds.iter().copied().enumerate() {
            let layer = format!("{MODEL}layers.{index}.");
            let attn_hyper = hyper(
                &mut loader,
                &format!("{layer}attn_hyper_connection."),
                &dims,
                true,
            )?;
            let attn = if is_qsa {
                OwnedAttn::Qsa(qsa(&mut loader, &format!("{layer}self_attn."), &dims)?)
            } else {
                OwnedAttn::Delta(delta(&mut loader, &format!("{layer}linear_attn."), &dims)?)
            };
            let mlp_hyper = hyper(
                &mut loader,
                &format!("{layer}mlp_hyper_connection."),
                &dims,
                true,
            )?;
            let moe_block = moe(&mut loader, &format!("{layer}mlp."), &dims)?;
            layers.push(OwnedLayer {
                attn_hyper,
                attn,
                mlp_hyper,
                moe: moe_block,
            });
        }

        let ple_block = ple(
            &mut loader,
            &format!("{MODEL}layers.{}.", dims.ple_layer),
            &dims,
        )?;
        let keys = loader.keys.into_iter().collect();
        Ok(Self {
            shape,
            embed,
            lm_head,
            mixer,
            layers,
            ple: ple_block,
            keys,
        })
    }

    /// The model shape.
    pub(crate) fn shape(&self) -> &ModelShape {
        &self.shape
    }

    /// A borrowed view for `step::step`.
    pub(crate) fn view(&self) -> ModelWeights<'_, StoredWeight> {
        ModelWeights {
            embed: &self.embed,
            layers: self
                .layers
                .iter()
                .map(|layer| FlashLayerWeights {
                    attn_hyper: layer.attn_hyper.view(),
                    attn: match &layer.attn {
                        OwnedAttn::Delta(weights) => AttnWeights::Delta(weights.view()),
                        OwnedAttn::Qsa(weights) => AttnWeights::Qsa(weights.view()),
                    },
                    mlp_hyper: layer.mlp_hyper.view(),
                    moe: layer.moe.view(),
                })
                .collect(),
            ple: self.ple.view(),
            mixer: self.mixer.view(),
            lm_head: &self.lm_head,
        }
    }

    /// Every checkpoint key the loader read or checked, sorted and without duplicates.
    pub(crate) fn keys(&self) -> &[String] {
        &self.keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::ShardIo;

    const CONFIG: &[u8] = include_bytes!("../../../tests/fixtures/qwen3_8_flash_next_config.json");

    fn checkpoint(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> StreamedCheckpoint {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name.to_string(), serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        let len = file.len() as u64;
        let reader: Box<dyn ShardIo + Send> = Box::new(std::io::Cursor::new(file));
        StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }

    #[test]
    fn model_shape_reads_the_flash_next_config() {
        let (shape, kinds) = model_shape(CONFIG).unwrap();
        assert_eq!(shape.hyper.width(), 10240, "hyper width");
        assert_eq!(shape.hyper.branches(), 4, "hyper branches");
        assert_eq!(shape.ple_layer, 1, "ple layer");
        assert_eq!(shape.vocab, 248_320, "vocabulary size");
        assert_eq!(kinds.len(), 48, "layer kind count");
        let qsa: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter(|(_, is_qsa)| **is_qsa)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            qsa,
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47],
            "QSA layer indices"
        );
    }

    #[test]
    fn model_shape_rejects_a_missing_field() {
        let mut value: Value = serde_json::from_slice(CONFIG).unwrap();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("hidden_size");
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(model_shape(&bytes), Err(EngError::Checkpoint(_))));
    }

    #[test]
    fn load_reports_the_first_missing_tensor() {
        let ckpt = checkpoint(&[(
            "language_model.model.hyper_connection_mixer.hc_norm.weight",
            "BF16",
            vec![10240],
            vec![0u8; 10240 * 2],
        )]);
        match FlashWeights::load(&ckpt, CONFIG) {
            Err(EngError::Checkpoint(message)) => {
                assert!(
                    message.contains("missing tensor"),
                    "message was {message:?}"
                );
            }
            Err(other) => panic!("expected a checkpoint error, got {other:?}"),
            Ok(_) => panic!("load unexpectedly succeeded with one tensor"),
        }
    }

    #[test]
    fn ple_layer_outside_the_model_is_rejected() {
        let ckpt = checkpoint(&[(
            "language_model.model.hyper_connection_mixer.hc_norm.weight",
            "BF16",
            vec![10240],
            vec![0u8; 10240 * 2],
        )]);
        let mut value: Value = serde_json::from_slice(CONFIG).unwrap();
        value["text_config"]["ple_layer_ids"] = serde_json::json!([49]);
        let config = serde_json::to_vec(&value).unwrap();
        match FlashWeights::load(&ckpt, &config) {
            Err(EngError::Checkpoint(message)) => {
                assert!(message.contains("ple_layer_ids"), "message was {message:?}");
            }
            Err(other) => panic!("expected a checkpoint error, got {other:?}"),
            Ok(_) => panic!("load unexpectedly accepted ple_layer_ids [49]"),
        }
    }
}
