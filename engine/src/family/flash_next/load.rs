use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::{
    deltanet::{DeltaShape, DeltaWeights},
    hyper::{HyperShape, HyperWeights},
    moe::{MoeShape, MoeWeights},
    ple::{PleDims, PleShape, PleWeights},
    qsa::{QsaDims, QsaShape, QsaWeights},
    step::{AttnWeights, FlashLayerWeights, ModelShape, ModelWeights},
};
use crate::family::stored::{self, StoredWeight};
use crate::{
    EngError,
    checkpoint_json::UniqueValue,
    family::quantization::{ModuleQuantization, QuantizationMap},
    kernels::{QuantizationMode, QuantizationSpec},
    weights::StreamedCheckpoint,
};

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(super) mod exl3_pack;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(super) mod pack_files;

const MODEL: &str = "language_model.model.";

/// Model shape and per-layer attention kind (true for QSA) read from `config.json`.
pub(crate) fn model_shape(config: &[u8]) -> Result<(ModelShape, Vec<bool>), EngError> {
    let dims = Dims::read(config)?;
    let shape = dims.model_shape()?;
    Ok((shape, dims.kinds))
}

/// Packed payloads that the resident MLX importer always loads.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn resident_packed_payload(key: &str, kinds: &[bool], ple_layer: usize) -> bool {
    if key == "language_model.lm_head.weight" {
        return true;
    }
    let Some(module) = key
        .strip_prefix(MODEL)
        .and_then(|name| name.strip_suffix(".weight"))
    else {
        return false;
    };
    if matches!(
        module,
        "hyper_connection_mixer.input_mix_weight_down"
            | "hyper_connection_mixer.input_mix_weight_up"
    ) {
        return true;
    }
    let Some((layer, role)) = module
        .strip_prefix("layers.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    let Ok(index) = layer.parse::<usize>() else {
        return false;
    };
    let Some(&qsa) = kinds.get(index).filter(|_| layer == index.to_string()) else {
        return false;
    };
    if let Some(projection) = role
        .strip_prefix("attn_hyper_connection.")
        .or_else(|| role.strip_prefix("mlp_hyper_connection."))
    {
        return matches!(
            projection,
            "input_mix_weight_down" | "input_mix_weight_up" | "block_inject_weight"
        );
    }
    match role {
        "self_attn.q_proj"
        | "self_attn.k_proj"
        | "self_attn.v_proj"
        | "self_attn.o_proj"
        | "self_attn.indexer.index_qk_proj" => qsa,
        "linear_attn.in_proj_qkv"
        | "linear_attn.in_proj_z"
        | "linear_attn.in_proj_b"
        | "linear_attn.in_proj_a"
        | "linear_attn.out_proj" => !qsa,
        "mlp.shared_expert.gate_proj"
        | "mlp.shared_expert.up_proj"
        | "mlp.shared_expert.down_proj"
        | "mlp.shared_expert_gate" => true,
        "ple.key_proj" | "ple.value_proj" => index == ple_layer,
        _ => false,
    }
}

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

fn invalid(name: &str) -> EngError {
    EngError::Checkpoint(format!("config field {name} is missing or invalid"))
}

fn overflow() -> EngError {
    EngError::Checkpoint("config dimension overflows".into())
}

fn product(factors: &[u32]) -> Result<u32, EngError> {
    factors
        .iter()
        .try_fold(1u32, |n, &factor| n.checked_mul(factor))
        .ok_or_else(overflow)
}

fn lookup<'a>(text: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    let mut parts = path.split('.');
    let first = parts.next()?;
    parts.try_fold(text.get(first)?, |value, part| value.get(part))
}

fn number(text: &Map<String, Value>, path: &str) -> Result<u32, EngError> {
    lookup(text, path)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| invalid(path))
}

fn real(text: &Map<String, Value>, path: &str) -> Result<f64, EngError> {
    lookup(text, path)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(path))
}

impl Dims {
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

struct OwnedHyper {
    norm: Vec<f32>,
    down: StoredWeight,
    up: StoredWeight,
    inject: Option<StoredWeight>,
}

impl OwnedHyper {
    fn view(&self) -> HyperWeights<'_, StoredWeight> {
        HyperWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: self.inject.as_ref(),
        }
    }
}

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

enum OwnedAttn {
    Delta(OwnedDelta),
    Qsa(OwnedQsa),
}

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

struct OwnedLayer {
    attn_hyper: OwnedHyper,
    attn: OwnedAttn,
    mlp_hyper: OwnedHyper,
    moe: OwnedMoe,
}

pub(crate) struct FlashWeights {
    shape: ModelShape,
    embed: StoredWeight,
    lm_head: StoredWeight,
    mixer: OwnedHyper,
    layers: Vec<OwnedLayer>,
    ple: OwnedPle,
    keys: Vec<String>,
}

struct Loader<'a> {
    ckpt: &'a StreamedCheckpoint,
    quant: &'a QuantizationMap,
    keys: BTreeSet<String>,
}

impl Loader<'_> {
    fn spec(&self, stem: &str) -> Result<QuantizationSpec, EngError> {
        match self.quant.resolve(stem)? {
            ModuleQuantization::Quantized(spec) => Ok(spec),
            ModuleQuantization::Unquantized => Err(EngError::Checkpoint(format!(
                "module {stem} is not quantized"
            ))),
        }
    }

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

    fn host_f32(&mut self, key: &str, len: usize) -> Result<Vec<f32>, EngError> {
        let values = stored::host_f32(self.ckpt, key, len)?;
        self.keys.insert(key.to_string());
        Ok(values)
    }

    fn host_i64(&mut self, key: &str, len: usize) -> Result<Vec<i64>, EngError> {
        let values = stored::host_i64(self.ckpt, key, len)?;
        self.keys.insert(key.to_string());
        Ok(values)
    }
}

/// The mixer has no inject projection.
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

    pub(crate) fn shape(&self) -> &ModelShape {
        &self.shape
    }

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

/// The capability preflight's facts-in-rows-out entry for this family: the
/// same settings resolution, open, walk and plan the loader runs, fed by
/// probed facts instead of the live card, rendered as the shared decision
/// rows with a one-line note the report appends. `context` and `kv_dtype`
/// are the candidate's values; the settings resolution accepts and refuses
/// exactly what the loader would, with its own texts and key names. The
/// ring budget is read from the same environment variable and clamped the
/// same way; the prefill arena is start-only policy, so the expert row is
/// taken before it — the note says so. When the probed facts cannot carry
/// a plan (no free-bytes or host reading), the entry answers with no rows
/// and the reason: a fact the probe could not read is not the pack's
/// fault.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn preflight_rows(
    request: &crate::family::plan_rows::RowsRequest,
) -> Result<
    (
        Vec<crate::family::plan_decision::PlanDecision>,
        String,
        crate::family::plan_rows::RowsFit,
    ),
    EngError,
> {
    use crate::capability::Probe;
    use crate::family::flash_next::moe::tier_plan::{TierInputs, TierShape};
    use crate::family::plan_decision::{PlanAlternative, PlanFixity};
    use crate::family::plan_rows::{PlanRows, RowsFit};
    // The fields this family reads; the request's shape is shared.
    let dir = request.model_dir;
    let context = request.context;
    let kv_dtype = request.kv_dtype;
    let facts = request.facts;

    // The loader's own settings resolution: same keys, same texts, same
    // bounds (a multiple of 4 from 4 to 262144; f16 only, int8 has no
    // attention kernel yet) — refused, never clamped.
    let options = super::settings::FlashOptions {
        cuda_decode: Some(true),
        context,
        prefill_rows: request.prefill_rows,
        kv_format: match kv_dtype {
            None => None,
            Some(name) => Some(
                serde_json::from_value::<super::settings::KvFormatName>(serde_json::Value::String(
                    name.to_string(),
                ))
                .map_err(|error| EngError::BadRequest(format!("kv_format: {error}")))?,
            ),
        },
        ..super::settings::FlashOptions::default()
    };
    let settings = match super::settings::cuda_settings(&options)? {
        Some(settings) => settings,
        // Unreachable in practice: the options above set cuda_decode. A
        // report path refuses; it never panics.
        None => {
            return Err(EngError::BadRequest(
                "cuda_decode is set but the settings resolution returned none".into(),
            ));
        }
    };

    // The loader's own switch check, called the same way `build_driver` calls it.
    super::cuda::driver::chunk::prof::check_switch_spelling()?;
    // The two PLE switches refuse on spelling the same way at start; their
    // values change no row, so only their refusal is shared.
    super::settings::ple_device()?;
    super::settings::ple_batch()?;

    let free_vram = match &facts.cuda.first_device_free_bytes {
        Probe::Measured(free) => *free,
        Probe::Unavailable(reason) => {
            return Ok((
                Vec::new(),
                format!("the device's free memory could not be probed: {reason}"),
                RowsFit::Unknown,
            ));
        }
    };
    let host = match (&facts.host.mem_total_bytes, &facts.host.mem_available_bytes) {
        (Probe::Measured(total), Probe::Measured(available)) => (*total, *available),
        _ => {
            return Ok((
                Vec::new(),
                "the host's memory could not be probed, so the host side of the plan cannot be checked".to_string(),
                RowsFit::Unknown,
            ));
        }
    };

    // The loader's own open: the storage guard judges the directory, every
    // shard, the metadata files and the n-gram table before anything is
    // read, refusing a network-backed path by name.
    let pack_files::FlashPackFiles {
        checkpoint,
        ngram: _,
        report,
    } = pack_files::FlashPackFiles::open(dir)?;
    let (model, kinds) = model_shape(&checkpoint.model_config)?;
    let ngram_bytes = std::fs::metadata(dir.join(pack_files::NGRAM_FILE))
        .map_err(|error| EngError::Checkpoint(format!("{}: {error}", pack_files::NGRAM_FILE)))?
        .len();
    let tier_shape = TierShape::from_pack(
        kinds.len(),
        model.moe.experts(),
        model.moe.top_k(),
        report.expert_bytes,
    )?;
    let ring_budget = super::settings::ring_budget_bytes(
        std::env::var("INFERENCE_FLASH_NEXT_RING_BUDGET")
            .ok()
            .as_deref(),
    )?;
    let card = TierInputs::measured(
        tier_shape,
        (0, 0),
        crate::family::device_budget::KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
        free_vram.saturating_sub(ring_budget),
        host,
        // No ring source on this path: the pinned-store refusal stands.
        0,
    );
    let plan = crate::family::flash_next::moe::load_plan::FlashLoadPlan::new(
        &report,
        &checkpoint.model_config,
        ngram_bytes,
        card,
    )?;
    let requested = u64::from(settings.context);
    let kv = settings.kv_format.format();
    let min_slots = model.moe.top_k() + 1;
    let note = "the rows assume cuda_decode: true and a manifest in the config; the expert row is before the prefill arena the load carves out; the rows do not run the bank-mode switch spelling, the pack-record and geometry constructions, the PLE-vector and n-gram reads, the live-host pin admission, the tokenizer vocabulary pin, an expert_stats profile parse, or the device-time bank and kernel registration — load-time admission re-runs all of them";
    match plan.at_context(requested, kv, min_slots) {
        Ok(load) => Ok((load.plan_rows(), note.to_string(), RowsFit::Fits)),
        Err(refusal) => {
            // The context row keeps the configured truth and carries the
            // largest fitting context as a proposal, never applied; every
            // size row is the largest fitting plan's own, and the
            // alternative says so.
            let Some(largest) = largest_fitting_context(&plan, requested, kv, min_slots) else {
                return Err(refusal);
            };
            let largest = u32::try_from(largest).map_err(|_| {
                EngError::BadRequest(
                    "the largest fitting context overflows the settings range".into(),
                )
            })?;
            let load = plan.at_context(u64::from(largest), kv, min_slots)?;
            let mut rows = load.plan_rows();
            if let Some(row) = rows.iter_mut().find(|row| row.name == "context") {
                *row = crate::family::plan_decision::PlanDecision {
                    name: row.name.clone(),
                    chosen: format!("{requested} positions"),
                    alternatives: vec![PlanAlternative {
                        option: format!("{largest} positions"),
                        verdict: "fits".to_string(),
                        reason: "largest multiple of 1024 the same plan admits; a proposal, never applied — the size rows are this plan's".to_string(),
                    }],
                    fixed_at: PlanFixity::Start,
                };
            }
            // The two rows the plan scales with the context say which
            // context they are for; the fixed rows (trunk, embedding,
            // n-gram) do not scale and stay context-free.
            for row in rows
                .iter_mut()
                .filter(|row| matches!(row.name.as_str(), "K/V" | "expert placement"))
            {
                row.chosen = format!("{} at {largest} positions", row.chosen);
            }
            Ok((
                rows,
                note.to_string(),
                RowsFit::Constrained {
                    requested: settings.context,
                    largest,
                },
            ))
        }
    }
}

/// The largest multiple-of-1024 context at or below `requested` the plan
/// still admits at `min_slots`, found by binary search over the same
/// refusal the loader raises (the plan's fit is monotone in the context).
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn largest_fitting_context(
    plan: &crate::family::flash_next::moe::load_plan::FlashLoadPlan,
    requested: u64,
    kv: crate::family::flash_next::moe::context_budget::KvFormat,
    min_slots: usize,
) -> Option<u64> {
    // Search over multiples of 1024, like the qwen36 proposal. The caller
    // already knows `requested` does not fit, so `hi` starts refused.
    let mut lo = 0u64;
    let mut hi = requested / 1024;
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if plan.at_context(mid * 1024, kv, min_slots).is_ok() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    (lo > 0).then(|| lo * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::plan_decision::{PlanAlternative, PlanFixity};
    use crate::family::plan_rows::RowsFit;
    use crate::weights::ShardIo;

    fn rows_request<'a>(
        dir: &'a std::path::Path,
        context: Option<u32>,
        kv_dtype: Option<&'a str>,
        facts: &'a crate::capability::probe::HardwareFacts,
    ) -> crate::family::plan_rows::RowsRequest<'a> {
        crate::family::plan_rows::RowsRequest {
            model_dir: dir,
            tokenizer_dir: dir,
            context,
            kv_dtype,
            prefill_rows: None,
            facts,
        }
    }

    pub(super) const CONFIG: &[u8] =
        include_bytes!("../../../tests/fixtures/qwen3_8_flash_next_config.json");

    pub(super) fn checkpoint(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> StreamedCheckpoint {
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

    // ----- the preflight rows entry -----

    /// The committed real n-gram table header: sparse-file layout, so the
    /// 32.6 GB it describes never touch the disk here.
    const NGRAM_HEADER: &[u8] =
        include_bytes!("../../../tests/fixtures/flash_next_exl3_ngram/ngram_embedding.header");

    fn width_of(dtype: &str) -> usize {
        match dtype {
            "F16" | "BF16" | "I16" => 2,
            _ => 4,
        }
    }

    /// A pack of the small stock layout on disk, its PLE widened so the
    /// n-gram table's layout is the committed real one (heads 16, rows of
    /// 160 values), plus the n-gram header as the table file.
    fn pack_dir() -> tempfile::TempDir {
        use crate::weights::exl3::flint_pack::PackBuilder;

        let mut value: serde_json::Value =
            serde_json::from_slice(&super::exl3_pack::tests::config()).unwrap();
        let text = value["text_config"].as_object_mut().unwrap();
        text.insert("heads_per_ngram".into(), serde_json::json!(8));
        text.insert("ple_embed_dim".into(), serde_json::json!(2560));
        // The pack's own header block: the model config above carries the
        // HF-style quantization block, not the pack reader's.
        value["quantization_config"] =
            serde_json::json!({"quant_method": "exl3", "version": "1.6.2"});
        let mut entries = super::exl3_pack::tests::pack();
        for (name, _, shape) in &mut entries {
            if name.ends_with("ple.key_proj.weight") {
                *shape = vec![256, 2560];
            }
            if name.ends_with("ple.value_proj.weight") {
                *shape = vec![128, 2560];
            }
        }
        let mut builder = PackBuilder {
            config: value,
            tensors: std::collections::BTreeMap::new(),
        };
        for (name, dtype, shape) in entries {
            let bytes = shape.iter().map(|&d| d as usize).product::<usize>() * width_of(dtype);
            builder.put(&name, dtype, &shape, vec![0u8; bytes]);
        }
        let dir = builder.write();
        let ngram_path = dir.path().join(super::pack_files::NGRAM_FILE);
        std::fs::write(&ngram_path, NGRAM_HEADER).unwrap();
        // The sparse-file trick of the n-gram table's own tests: the header
        // describes the real 32.6 GB table, and the hole never touches disk.
        std::fs::File::options()
            .write(true)
            .open(&ngram_path)
            .unwrap()
            .set_len(32_640_183_408)
            .unwrap();
        dir
    }

    fn flash_facts(free_vram: u64) -> crate::capability::probe::HardwareFacts {
        use crate::capability::Probe;
        use crate::capability::probe::{CompiledBackends, CudaFacts, HardwareFacts, HostFacts};
        HardwareFacts {
            backends: CompiledBackends {
                cuda: true,
                mlx: false,
            },
            cuda: CudaFacts {
                device_count: Probe::Measured(1),
                first_device_integrated: Probe::Measured(false),
                first_device_free_bytes: Probe::Measured(free_vram),
                first_device_total_bytes: Probe::Measured(24u64 << 30),
            },
            host: HostFacts {
                mem_total_bytes: Probe::Measured(64u64 << 30),
                mem_available_bytes: Probe::Measured(32u64 << 30),
                cpu_threads: 8,
            },
        }
    }

    #[test]
    fn preflight_rows_render_the_loaders_plan_from_probed_facts() {
        let dir = pack_dir();
        let (rows, note, fit) = preflight_rows(&rows_request(
            dir.path(),
            None,
            None,
            &flash_facts(24 << 30),
        ))
        .unwrap();
        assert!(matches!(fit, RowsFit::Fits));
        assert_eq!(
            note,
            "the rows assume cuda_decode: true and a manifest in the config; the expert row is before the prefill arena the load carves out; the rows do not run the bank-mode switch spelling, the pack-record and geometry constructions, the PLE-vector and n-gram reads, the live-host pin admission, the tokenizer vocabulary pin, an expert_stats profile parse, or the device-time bank and kernel registration — load-time admission re-runs all of them"
        );
        let rendered: Vec<(&str, &str, PlanFixity)> = rows
            .iter()
            .map(|row| (row.name.as_str(), row.chosen.as_str(), row.fixed_at))
            .collect();
        assert_eq!(
            rendered,
            vec![
                ("context", "4096 positions", PlanFixity::Start),
                ("K/V", "F16 0.00 GiB", PlanFixity::Start),
                ("trunk", "0.00 GiB on the device", PlanFixity::Load),
                (
                    "expert placement",
                    "3 of 3 experts per layer in VRAM (0.00 GiB), 0.00 GiB pinned in host RAM",
                    PlanFixity::Load
                ),
                ("embedding", "0.00 GiB host", PlanFixity::Load),
                ("n-gram table", "30.40 GiB local NVMe", PlanFixity::Load),
            ]
        );
        assert!(rows.iter().all(|row| row.alternatives.is_empty()));
    }

    #[test]
    fn preflight_rows_take_an_explicit_width_over_the_default() {
        // The request names prefill_rows 64, the way a caller that knows its
        // chunked width would, under the same pinned shared mode as the default
        // resolution. A valid width resolves through the loader's own ranged
        // bounds; an invalid one is refused THERE, naming prefill_rows with the
        // same rule startup uses — the width demonstrably reaches the loader's
        // resolution, not a shallow echo of the default.
        let dir = pack_dir();
        let facts = flash_facts(24 << 30);
        let mut request = rows_request(dir.path(), None, None, &facts);
        request.prefill_rows = Some(64);
        let (_rows, _note, fit) = preflight_rows(&request).unwrap();
        assert!(matches!(fit, RowsFit::Fits));
        // Above the shared bound the loader's own ranged resolution refuses,
        // naming prefill_rows: the width reached the real bounds, and a value
        // no path accepts cannot pass the rows arm.
        let mut refused = rows_request(dir.path(), None, None, &facts);
        refused.prefill_rows = Some(1_000_000);
        let text = match preflight_rows(&refused) {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected a BadRequest refusal, got {other:?}"),
        };
        assert!(
            text.contains("prefill_rows"),
            "the refusal names the width rule: {text}"
        );
    }

    #[test]
    fn preflight_rows_name_the_largest_fitting_context_when_the_card_is_small() {
        let dir = pack_dir();
        let (rows, _note, fit) = preflight_rows(&rows_request(
            dir.path(),
            Some(262_144),
            None,
            &flash_facts(1_000_000_000),
        ))
        .unwrap();
        let context = rows.first().expect("the context row");
        assert_eq!(context.chosen, "262144 positions");
        assert_eq!(context.fixed_at, PlanFixity::Start);
        assert_eq!(
            context.alternatives,
            vec![PlanAlternative {
                option: "97280 positions".to_string(),
                verdict: "fits".to_string(),
                reason: "largest multiple of 1024 the same plan admits; a proposal, never applied — the size rows are this plan's"
                    .to_string(),
            }]
        );
        assert!(
            matches!(
                fit,
                RowsFit::Constrained {
                    requested: 262_144,
                    largest: 97_280
                }
            ),
            "{fit:?}"
        );
        let kv = &rows[1];
        // The context-scaled rows say which context they are for.
        assert_eq!(kv.chosen, "F16 0.05 GiB at 97280 positions");
        let placement = rows
            .iter()
            .find(|row| row.name == "expert placement")
            .expect("the placement row");
        assert!(placement.chosen.ends_with(" at 97280 positions"));
    }

    #[test]
    fn the_switch_checks_the_loader_runs_are_the_preflight_runs_too() {
        use super::super::cuda::driver::chunk::prof::check_switch_value;
        // A value the switches do not read, with the loader's own text.
        let error = check_switch_value("INFERENCE_FLASH_NEXT_PROFILE", Some("yes"))
            .unwrap_err()
            .describe();
        assert!(
            error.contains("must be 1, true, on, 0, false or off, not yes"),
            "{error}"
        );
    }

    #[test]
    fn preflight_rows_refuse_a_kv_name_the_settings_refuse() {
        let dir = pack_dir();
        // The same serde text a config file with this value would print.
        assert_eq!(
            preflight_rows(&rows_request(
                dir.path(),
                None,
                Some("q8"),
                &flash_facts(24 << 30)
            ))
            .unwrap_err()
            .describe(),
            "kv_format: unknown variant `q8`, expected `f16` or `int8`"
        );
    }

    #[test]
    fn preflight_rows_refuse_kv_int8_with_the_loaders_own_text() {
        let dir = pack_dir();
        assert_eq!(
            preflight_rows(&rows_request(
                dir.path(),
                None,
                Some("int8"),
                &flash_facts(24 << 30)
            ))
            .unwrap_err()
            .describe(),
            "engine config: kv_format int8 has no attention kernel yet; use f16"
        );
    }

    #[test]
    fn preflight_rows_refuse_an_unmultiple_context_instead_of_clamping() {
        let dir = pack_dir();
        assert_eq!(
            preflight_rows(&rows_request(
                dir.path(),
                Some(4097),
                None,
                &flash_facts(24 << 30)
            ))
            .unwrap_err()
            .describe(),
            "engine config: context must be a multiple of 4, found 4097"
        );
        // Above the family's own ceiling is refused, not clamped down.
        assert_eq!(
            preflight_rows(&rows_request(
                dir.path(),
                Some(262_148),
                None,
                &flash_facts(24 << 30)
            ))
            .unwrap_err()
            .describe(),
            "engine config: context must be between 4 and 262144, found 262148"
        );
    }

    #[test]
    fn the_entry_runs_the_loader_switch_checks_before_it_probes_the_card() {
        // The entry cannot take a variable through an env reader (the crate
        // denies `unsafe_code`), so the calls are pinned in source order,
        // the same pattern as the chunk tests use for start's order.
        let source = include_str!("load.rs");
        let entry = &source[source
            .find("pub(crate) fn preflight_rows(")
            .expect("the rows entry")..];
        let settings = entry
            .find("super::settings::cuda_settings(&options)?")
            .expect("settings resolution");
        let spelling = entry
            .find("prof::check_switch_spelling()?")
            .expect("switch spelling check");
        let ple_device = entry
            .find("super::settings::ple_device()?")
            .expect("PLE device switch check");
        let ple_batch = entry
            .find("super::settings::ple_batch()?")
            .expect("PLE batch switch check");
        let probe = entry
            .find("facts.cuda.first_device_free_bytes")
            .expect("the card probe read");
        assert!(settings < spelling && spelling < ple_device);
        assert!(ple_device < ple_batch && ple_batch < probe);
    }

    #[test]
    fn preflight_rows_answer_no_rows_when_the_probe_could_not_read_the_card() {
        let dir = pack_dir();
        let mut facts = flash_facts(24 << 30);
        facts.cuda.first_device_free_bytes =
            crate::capability::Probe::Unavailable("no driver".to_string());
        let (rows, note, fit) =
            preflight_rows(&rows_request(dir.path(), None, None, &facts)).unwrap();
        assert!(matches!(fit, RowsFit::Unknown));
        assert!(rows.is_empty());
        assert!(
            note.contains("the device's free memory could not be probed: no driver"),
            "{note}"
        );
    }
}
