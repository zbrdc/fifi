//! Loads the stepping-stone checkpoint for the host reference (Q36-HR).
//!
//! Resident tensors (the dense weights, the packed projections, the router, and the output head) are
//! read whole at load. The expert stacks and the embedding table stay in the shard files and are read
//! one slice at a time by [`super::host_cpu::CpuProjector`]. Norm weights load as stored: this
//! checkpoint holds plain weights, not zero-centered ones.

use std::collections::BTreeSet;

use serde_json::Value;

use super::host::{AttnW, DeltaW, DenseW, FeedForwardW, HostShape, LayerW, MixerW, ModelW, MoeW};
use crate::{
    EngError,
    checkpoint_json::UniqueValue,
    family::{
        flash_next::stored::{self, StoredWeight},
        quantization::{ModuleQuantization, QuantizationMap},
        spec::{AttentionKind, FamilySpec},
    },
    kernels::{QuantizationMode, QuantizationSpec},
    weights::StreamedCheckpoint,
};

/// Prefix of every language-model tensor name.
const MODEL: &str = "language_model.model.";

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

struct OwnedAttn {
    q: StoredWeight,
    k: StoredWeight,
    v: StoredWeight,
    o: StoredWeight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

enum OwnedMixer {
    Delta(OwnedDelta),
    Attn(OwnedAttn),
}

struct OwnedMoe {
    router: StoredWeight,
    gate: StoredWeight,
    up: StoredWeight,
    down: StoredWeight,
    shared_gate: StoredWeight,
    shared_up: StoredWeight,
    shared_down: StoredWeight,
    shared_scalar: StoredWeight,
}

struct OwnedDense {
    gate: StoredWeight,
    up: StoredWeight,
    down: StoredWeight,
}

enum OwnedFeedForward {
    Dense(Box<OwnedDense>),
    Moe(Box<OwnedMoe>),
}

struct OwnedLayer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: OwnedMixer,
    feed_forward: OwnedFeedForward,
}

/// Owned weights of the host reference.
pub(crate) struct HostWeights {
    shape: HostShape,
    embed: StoredWeight,
    lm_head: StoredWeight,
    final_norm: Vec<f32>,
    layers: Vec<OwnedLayer>,
    keys: Vec<String>,
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("rope configuration: {message}"))
}

/// Reads the RoPE rotary dimension from the configuration and checks the fields that
/// [`FamilySpec::check_config`] skips (TODO S7): the partial rotary factor must give a whole even
/// dimension, `mrope_section` must cover half of it, and the rope type must be `default`.
/// Text positions are equal in all three mrope axes, so interleaved mrope equals plain RoPE.
pub(crate) fn rotary_dim(config: &[u8], head_dim: u32) -> Result<u32, EngError> {
    let root = serde_json::from_slice::<UniqueValue>(config)
        .map_err(|error| EngError::Checkpoint(format!("configuration JSON: {error}")))?
        .0;
    let text = root.get("text_config").unwrap_or(&root);
    let rope = text
        .get("rope_parameters")
        .ok_or_else(|| invalid("rope_parameters is missing"))?;
    let factor = rope
        .get("partial_rotary_factor")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("rope_parameters.partial_rotary_factor is missing"))?;
    if let Some(top) = text.get("partial_rotary_factor")
        && top.as_f64() != Some(factor)
    {
        return Err(invalid(
            "partial_rotary_factor differs between the two levels",
        ));
    }
    if let Some(kind) = rope.get("rope_type")
        && kind.as_str() != Some("default")
    {
        return Err(invalid("only the default rope type is supported"));
    }
    let exact = f64::from(head_dim) * factor;
    let rotary = exact.round();
    if (exact - rotary).abs() > 1e-9 || rotary < 2.0 || rotary > f64::from(head_dim) {
        return Err(invalid(
            "head_dim * partial_rotary_factor is not a whole dimension",
        ));
    }
    let rotary = rotary as u32;
    if !rotary.is_multiple_of(2) {
        return Err(invalid("the rotary dimension must be even"));
    }
    if let Some(sections) = rope.get("mrope_section") {
        let total: Option<u64> = sections.as_array().and_then(|list| {
            list.iter()
                .try_fold(0u64, |total, section| total.checked_add(section.as_u64()?))
        });
        if total != Some(u64::from(rotary / 2)) {
            return Err(invalid(
                "mrope_section must sum to half the rotary dimension",
            ));
        }
    }
    Ok(rotary)
}

/// Reads checkpoint tensors through the quantization map and records each key read.
struct Loader<'a> {
    ckpt: &'a StreamedCheckpoint,
    quant: QuantizationMap,
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

    fn record(&mut self, stem: &str, spec: QuantizationSpec) {
        self.keys.insert(format!("{stem}.weight"));
        self.keys.insert(format!("{stem}.scales"));
        if spec.mode == QuantizationMode::Affine {
            self.keys.insert(format!("{stem}.biases"));
        }
    }

    /// A packed matrix `[out, input]`, read whole.
    fn packed(&mut self, stem: &str, out: usize, input: usize) -> Result<StoredWeight, EngError> {
        let spec = self.spec(stem)?;
        let weight = stored::packed(self.ckpt, stem, spec, dim(out)?, dim(input)?)?;
        self.record(stem, spec);
        Ok(StoredWeight::Packed(Box::new(weight)))
    }

    /// A packed stack `[slices, rows, cols]` or table `[slices, cols]`, described and not read.
    fn sliced(&mut self, stem: &str, rows: usize, cols: usize) -> Result<StoredWeight, EngError> {
        let spec = self.spec(stem)?;
        let weight = stored::sliced(self.ckpt, stem, spec, dim(rows)?, dim(cols)?)?;
        self.record(stem, spec);
        Ok(StoredWeight::Sliced(weight))
    }

    /// A dense tensor of `len` elements as host `f32`.
    fn floats(&mut self, key: &str, len: usize) -> Result<Vec<f32>, EngError> {
        let values = stored::host_f32(self.ckpt, key, len)?;
        self.keys.insert(key.to_string());
        Ok(values)
    }
}

fn dim(value: usize) -> Result<u32, EngError> {
    u32::try_from(value).map_err(|_| EngError::Checkpoint("dimension exceeds u32".into()))
}

impl HostWeights {
    /// Checks `config` against `spec`, then loads the resident tensors from `ckpt`.
    pub(crate) fn load(
        ckpt: &StreamedCheckpoint,
        spec: &FamilySpec,
        config: &[u8],
    ) -> Result<Self, EngError> {
        spec.check_config(config)?;
        let shape = HostShape::from_spec(spec, rotary_dim(config, spec.head_dim)?)?;
        let mut loader = Loader {
            ckpt,
            quant: QuantizationMap::from_json(config)?,
            keys: BTreeSet::new(),
        };
        let hidden = shape.hidden;
        let embed = loader.sliced(&format!("{MODEL}embed_tokens"), 1, hidden)?;
        let lm_head = loader.packed("language_model.lm_head", shape.vocab, hidden)?;
        let final_norm = loader.floats(&format!("{MODEL}norm.weight"), hidden)?;
        let mut layers = Vec::with_capacity(shape.kinds.len());
        for (index, kind) in shape.kinds.iter().enumerate() {
            let prefix = format!("{MODEL}layers.{index}.");
            let mixer = match kind {
                AttentionKind::DeltaNet => {
                    OwnedMixer::Delta(load_delta(&mut loader, &shape, &prefix)?)
                }
                AttentionKind::Full => OwnedMixer::Attn(load_attn(&mut loader, &shape, &prefix)?),
                AttentionKind::Qsa => {
                    return Err(EngError::Checkpoint(format!(
                        "layer {index} is a QSA layer"
                    )));
                }
            };
            layers.push(OwnedLayer {
                input_norm: loader.floats(&format!("{prefix}input_layernorm.weight"), hidden)?,
                post_norm: loader
                    .floats(&format!("{prefix}post_attention_layernorm.weight"), hidden)?,
                mixer,
                feed_forward: if let Some(inter) = shape.dense_inter {
                    OwnedFeedForward::Dense(Box::new(load_dense(
                        &mut loader,
                        hidden,
                        inter,
                        &prefix,
                    )?))
                } else {
                    OwnedFeedForward::Moe(Box::new(load_moe(&mut loader, &shape, &prefix)?))
                },
            });
        }
        Ok(Self {
            shape,
            embed,
            lm_head,
            final_norm,
            layers,
            keys: loader.keys.into_iter().collect(),
        })
    }

    /// The checked dimensions.
    pub(crate) fn shape(&self) -> &HostShape {
        &self.shape
    }

    /// Every tensor key that the load read, sorted.
    pub(crate) fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Borrowed weights for [`super::host::step`].
    pub(crate) fn view(&self) -> ModelW<'_, StoredWeight> {
        ModelW {
            embed: &self.embed,
            layers: self
                .layers
                .iter()
                .map(|layer| LayerW {
                    input_norm: &layer.input_norm,
                    post_norm: &layer.post_norm,
                    mixer: match &layer.mixer {
                        OwnedMixer::Delta(d) => MixerW::Delta(DeltaW {
                            qkv: &d.qkv,
                            z: &d.z,
                            b: &d.b,
                            a: &d.a,
                            out: &d.out,
                            conv: &d.conv,
                            a_log: &d.a_log,
                            dt_bias: &d.dt_bias,
                            norm: &d.norm,
                        }),
                        OwnedMixer::Attn(a) => MixerW::Attn(AttnW {
                            q: &a.q,
                            k: &a.k,
                            v: &a.v,
                            o: &a.o,
                            q_norm: &a.q_norm,
                            k_norm: &a.k_norm,
                        }),
                    },
                    feed_forward: match &layer.feed_forward {
                        OwnedFeedForward::Dense(dense) => FeedForwardW::Dense(DenseW {
                            gate: &dense.gate,
                            up: &dense.up,
                            down: &dense.down,
                        }),
                        OwnedFeedForward::Moe(moe) => FeedForwardW::Moe(MoeW {
                            router: &moe.router,
                            gate: &moe.gate,
                            up: &moe.up,
                            down: &moe.down,
                            shared_gate: &moe.shared_gate,
                            shared_up: &moe.shared_up,
                            shared_down: &moe.shared_down,
                            shared_scalar: &moe.shared_scalar,
                        }),
                    },
                })
                .collect(),
            final_norm: &self.final_norm,
            lm_head: &self.lm_head,
        }
    }
}

fn load_delta(
    loader: &mut Loader<'_>,
    shape: &HostShape,
    prefix: &str,
) -> Result<OwnedDelta, EngError> {
    let (d, hidden) = (&shape.delta, shape.hidden);
    let value_width = d.value_heads * d.value_dim;
    let stem = |name: &str| format!("{prefix}linear_attn.{name}");
    Ok(OwnedDelta {
        qkv: loader.packed(&stem("in_proj_qkv"), d.conv_dim(), hidden)?,
        z: loader.packed(&stem("in_proj_z"), value_width, hidden)?,
        b: loader.packed(&stem("in_proj_b"), d.value_heads, hidden)?,
        a: loader.packed(&stem("in_proj_a"), d.value_heads, hidden)?,
        out: loader.packed(&stem("out_proj"), hidden, value_width)?,
        conv: loader.floats(&stem("conv1d.weight"), d.conv_dim() * d.conv_kernel)?,
        a_log: loader.floats(&stem("A_log"), d.value_heads)?,
        dt_bias: loader.floats(&stem("dt_bias"), d.value_heads)?,
        norm: loader.floats(&stem("norm.weight"), d.value_dim)?,
    })
}

fn load_attn(
    loader: &mut Loader<'_>,
    shape: &HostShape,
    prefix: &str,
) -> Result<OwnedAttn, EngError> {
    let (a, hidden) = (&shape.attn, shape.hidden);
    let stem = |name: &str| format!("{prefix}self_attn.{name}");
    Ok(OwnedAttn {
        q: loader.packed(&stem("q_proj"), a.heads_q * a.head_dim * 2, hidden)?,
        k: loader.packed(&stem("k_proj"), a.heads_kv * a.head_dim, hidden)?,
        v: loader.packed(&stem("v_proj"), a.heads_kv * a.head_dim, hidden)?,
        o: loader.packed(&stem("o_proj"), hidden, a.heads_q * a.head_dim)?,
        q_norm: loader.floats(&stem("q_norm.weight"), a.head_dim)?,
        k_norm: loader.floats(&stem("k_norm.weight"), a.head_dim)?,
    })
}

fn load_dense(
    loader: &mut Loader<'_>,
    hidden: usize,
    inter: usize,
    prefix: &str,
) -> Result<OwnedDense, EngError> {
    Ok(OwnedDense {
        gate: loader.packed(&format!("{prefix}mlp.gate_proj"), inter, hidden)?,
        up: loader.packed(&format!("{prefix}mlp.up_proj"), inter, hidden)?,
        down: loader.packed(&format!("{prefix}mlp.down_proj"), hidden, inter)?,
    })
}

fn load_moe(
    loader: &mut Loader<'_>,
    shape: &HostShape,
    prefix: &str,
) -> Result<OwnedMoe, EngError> {
    let (m, hidden) = (&shape.moe, shape.hidden);
    let stem = |name: &str| format!("{prefix}mlp.{name}");
    Ok(OwnedMoe {
        router: loader.packed(&stem("gate"), m.experts, hidden)?,
        gate: loader.sliced(&stem("switch_mlp.gate_proj"), m.inter, hidden)?,
        up: loader.sliced(&stem("switch_mlp.up_proj"), m.inter, hidden)?,
        down: loader.sliced(&stem("switch_mlp.down_proj"), hidden, m.inter)?,
        shared_gate: loader.packed(&stem("shared_expert.gate_proj"), m.shared_inter, hidden)?,
        shared_up: loader.packed(&stem("shared_expert.up_proj"), m.shared_inter, hidden)?,
        shared_down: loader.packed(&stem("shared_expert.down_proj"), hidden, m.shared_inter)?,
        shared_scalar: loader.packed(&stem("shared_expert_gate"), 1, hidden)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::Projector;
    use crate::family::qwen36::fixtures::{self, Entry, checkpoint, dense, val};
    use crate::family::qwen36::host::{HostState, step};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::spec::{Act, LinearAttention, Router};
    use crate::kernels::PackedWeight;

    /// A two-layer model with the stepping-stone key layout: layer 0 is DeltaNet, layer 1 is dense
    /// attention. Hidden 64, quantization groups of 32, vocabulary 8, 4 experts with top 2.
    const TINY: FamilySpec = FamilySpec {
        name: "q36_tiny",
        layers: 2,
        hidden: 64,
        heads_q: 2,
        heads_kv: 1,
        head_dim: 32,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        model_types: &["qwen3_5_moe", "qwen3_5_moe_text"],
        vocab: 8,
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
        layers_attention: &[AttentionKind::DeltaNet, AttentionKind::Full],
        linear: Some(LinearAttention {
            key_heads: 1,
            key_head_dim: 32,
            value_heads: 2,
            value_head_dim: 32,
            conv_kernel: 4,
        }),
        hyper: None,
    };

    fn config_json() -> Value {
        let mut quantization = serde_json::json!({"mode": "affine", "bits": 4, "group_size": 32});
        for layer in 0..2 {
            for name in ["mlp.gate", "mlp.shared_expert_gate"] {
                quantization[format!("language_model.model.layers.{layer}.{name}")] =
                    serde_json::json!({"bits": 8, "group_size": 32});
            }
        }
        serde_json::json!({
            "model_type": "qwen3_5_moe",
            "quantization": quantization,
            "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64, "num_attention_heads": 2,
                "num_hidden_layers": 2, "num_key_value_heads": 1, "vocab_size": 8,
                "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64, "head_dim": 32, "rms_norm_eps": 1e-6,
                "hidden_act": "silu", "full_attention_interval": 2,
                "linear_num_key_heads": 1, "linear_key_head_dim": 32, "linear_num_value_heads": 2,
                "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                "partial_rotary_factor": 0.5,
                "rope_parameters": {
                    "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                    "rope_type": "default", "mrope_section": [3, 3, 2]
                }
            }
        })
    }

    fn config() -> Vec<u8> {
        serde_json::to_vec(&config_json()).unwrap()
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

    /// Every tensor of the tiny model, with a different hash tag per tensor.
    fn entries() -> Vec<Entry> {
        let (mut out, tag) = (Vec::new(), &mut 0u64);
        module(
            &mut out,
            tag,
            "language_model.model.embed_tokens",
            &[8],
            64,
            4,
        );
        module(&mut out, tag, "language_model.lm_head", &[8], 64, 4);
        floats(&mut out, tag, "language_model.model.norm.weight", 64, |v| {
            1.0 + v / 4.0
        });
        for layer in 0..2 {
            let p = format!("language_model.model.layers.{layer}.");
            floats(
                &mut out,
                tag,
                &format!("{p}input_layernorm.weight"),
                64,
                |v| 1.0 + v / 4.0,
            );
            floats(
                &mut out,
                tag,
                &format!("{p}post_attention_layernorm.weight"),
                64,
                |v| 1.0 + v / 4.0,
            );
            if layer == 0 {
                for (name, rows) in [
                    ("in_proj_qkv", 128),
                    ("in_proj_z", 64),
                    ("in_proj_b", 2),
                    ("in_proj_a", 2),
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
                    64,
                    4,
                );
                floats(
                    &mut out,
                    tag,
                    &format!("{p}linear_attn.conv1d.weight"),
                    128 * 4,
                    |v| v,
                );
                floats(&mut out, tag, &format!("{p}linear_attn.A_log"), 2, |v| {
                    v / 2.0
                });
                floats(&mut out, tag, &format!("{p}linear_attn.dt_bias"), 2, |v| {
                    v / 2.0
                });
                floats(
                    &mut out,
                    tag,
                    &format!("{p}linear_attn.norm.weight"),
                    32,
                    |v| 1.0 + v / 4.0,
                );
            } else {
                for (name, rows) in [
                    ("q_proj", 128),
                    ("k_proj", 32),
                    ("v_proj", 32),
                    ("o_proj", 64),
                ] {
                    module(
                        &mut out,
                        tag,
                        &format!("{p}self_attn.{name}"),
                        &[rows],
                        64,
                        4,
                    );
                }
                floats(
                    &mut out,
                    tag,
                    &format!("{p}self_attn.q_norm.weight"),
                    32,
                    |v| 1.0 + v / 4.0,
                );
                floats(
                    &mut out,
                    tag,
                    &format!("{p}self_attn.k_norm.weight"),
                    32,
                    |v| 1.0 + v / 4.0,
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

    fn entry<'a>(entries: &'a [Entry], key: &str) -> &'a Entry {
        entries
            .iter()
            .find(|e| e.0 == key)
            .unwrap_or_else(|| panic!("no entry {key}"))
    }

    /// Dequantizes every row of a packed matrix by the definition `scale * code + bias`.
    fn naive_rows(weight: &PackedWeight) -> Vec<Vec<f64>> {
        let (bits, group) = (
            weight.quantization().bits as usize,
            weight.quantization().group_size as usize,
        );
        let &[rows, cols] = weight.logical_shape() else {
            panic!("rank")
        };
        let (rows, cols) = (rows as usize, cols as usize);
        let words: Vec<u32> = weight
            .payload()
            .host_bytes()
            .unwrap()
            .chunks(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let half = |tensor: &crate::tensor::Tensor| -> Vec<f64> {
            tensor
                .host_bytes()
                .unwrap()
                .chunks(2)
                .map(|b| {
                    f64::from(f32::from_bits(
                        u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
                    ))
                })
                .collect()
        };
        let (scales, biases) = (half(weight.scales()), half(weight.biases().unwrap()));
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let bit = c * bits;
                        let word = words[r * (cols * bits / 32) + bit / 32];
                        let code = (word >> (bit % 32)) & ((1 << bits) - 1);
                        let g = r * (cols / group) + c / group;
                        scales[g] * f64::from(code) + biases[g]
                    })
                    .collect()
            })
            .collect()
    }

    /// Projections by the definition, one dequantized matrix at a time, summed in `f64`.
    struct NaiveProjector<'c>(&'c crate::weights::StreamedCheckpoint);

    impl NaiveProjector<'_> {
        fn times(rows: &[Vec<f64>], x: &[f32]) -> Vec<f32> {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .zip(x)
                        .map(|(w, v)| w * f64::from(*v))
                        .sum::<f64>() as f32
                })
                .collect()
        }
    }

    impl Projector for NaiveProjector<'_> {
        type Weight = StoredWeight;

        fn linear(&mut self, weight: &StoredWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
            let StoredWeight::Packed(p) = weight else {
                panic!("linear on a slice")
            };
            Ok(Self::times(&naive_rows(p), x))
        }

        fn expert_linear(
            &mut self,
            stack: &StoredWeight,
            expert: u32,
            x: &[f32],
        ) -> Result<Vec<f32>, EngError> {
            let StoredWeight::Sliced(s) = stack else {
                panic!("expert_linear on a matrix")
            };
            Ok(Self::times(
                &naive_rows(&stored::read_slice(self.0, s, expert)?),
                x,
            ))
        }

        fn rows(&mut self, table: &StoredWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
            let StoredWeight::Sliced(s) = table else {
                panic!("rows on a matrix")
            };
            let mut out = Vec::new();
            for &row in rows {
                out.extend(
                    naive_rows(&stored::read_slice(self.0, s, row)?)[0]
                        .iter()
                        .map(|v| *v as f32),
                );
            }
            Ok(out)
        }
    }

    #[test]
    fn dense_load_reads_three_packed_projections_without_moe_keys() {
        use crate::family::qwen36::tiny::{self, TINY_DENSE};

        let all = tiny::entries(&TINY_DENSE);
        let ckpt = tiny::checkpoint(&TINY_DENSE);
        let weights = HostWeights::load(&ckpt, &TINY_DENSE, &tiny::config(&TINY_DENSE)).unwrap();
        let mut names: Vec<String> = all.iter().map(|e| e.0.clone()).collect();
        names.sort();
        assert_eq!(weights.keys(), names.as_slice());
        for (index, layer) in weights.view().layers.iter().enumerate() {
            assert!(matches!(layer.as_moe(), Err(EngError::BadRequest(_))));
            let FeedForwardW::Dense(dense) = &layer.feed_forward else {
                panic!("expected dense feed-forward");
            };
            for (name, weight, shape) in [
                ("gate_proj", dense.gate, [96, 64]),
                ("up_proj", dense.up, [96, 64]),
                ("down_proj", dense.down, [64, 96]),
            ] {
                let StoredWeight::Packed(packed) = weight else {
                    panic!("expected a packed projection");
                };
                assert_eq!(packed.logical_shape(), shape);
                let key = format!("{MODEL}layers.{index}.mlp.{name}.weight");
                assert_eq!(
                    packed.payload().host_bytes().unwrap(),
                    entry(&all, &key).3.as_slice()
                );
            }
        }
    }

    #[test]
    fn load_reads_every_tensor_once_and_wires_each_role_to_its_key() {
        let all = entries();
        let ckpt = checkpoint(all.clone());
        let weights = HostWeights::load(&ckpt, &TINY, &config()).unwrap();
        let mut names: Vec<String> = all.iter().map(|e| e.0.clone()).collect();
        names.sort();
        assert_eq!(
            weights.keys(),
            names.as_slice(),
            "the load reads every fixture tensor and no other"
        );
        assert_eq!(weights.shape().attn.rotary_dim, 16);
        let model = weights.view();
        let p = "language_model.model.layers.";
        let payload = |w: &StoredWeight| match w {
            StoredWeight::Packed(packed) => packed.payload().host_bytes().unwrap().to_vec(),
            StoredWeight::Sliced(_) => panic!("expected a matrix"),
        };
        let first_slice = |w: &StoredWeight| match w {
            StoredWeight::Sliced(s) => {
                let packed = stored::read_slice(&ckpt, s, 0).unwrap();
                packed.payload().host_bytes().unwrap().to_vec()
            }
            StoredWeight::Packed(_) => panic!("expected slices"),
        };
        let stored_payload = |key: &str| entry(&all, key).3.clone();
        let stored_first = |key: &str| {
            let bytes = &entry(&all, key).3;
            bytes[..bytes.len() / 4].to_vec()
        };
        // Matrices are compared whole; stacks and the table compare slice 0 (of 4 and of 8).
        assert_eq!(
            payload(model.lm_head),
            stored_payload("language_model.lm_head.weight")
        );
        let table = entry(&all, "language_model.model.embed_tokens.weight")
            .3
            .len()
            / 8;
        assert_eq!(
            first_slice(model.embed),
            entry(&all, "language_model.model.embed_tokens.weight").3[..table].to_vec()
        );
        let (MixerW::Delta(d), MixerW::Attn(a)) = (&model.layers[0].mixer, &model.layers[1].mixer)
        else {
            panic!("layer kinds");
        };
        for (weight, name) in [
            (d.qkv, "in_proj_qkv"),
            (d.z, "in_proj_z"),
            (d.b, "in_proj_b"),
            (d.a, "in_proj_a"),
            (d.out, "out_proj"),
        ] {
            assert_eq!(
                payload(weight),
                stored_payload(&format!("{p}0.linear_attn.{name}.weight")),
                "{name}"
            );
        }
        for (weight, name) in [
            (a.q, "q_proj"),
            (a.k, "k_proj"),
            (a.v, "v_proj"),
            (a.o, "o_proj"),
        ] {
            assert_eq!(
                payload(weight),
                stored_payload(&format!("{p}1.self_attn.{name}.weight")),
                "{name}"
            );
        }
        for layer in 0..2 {
            let m = model.layers[layer].as_moe().unwrap();
            let stem = format!("{p}{layer}.mlp.");
            assert_eq!(
                payload(m.router),
                stored_payload(&format!("{stem}gate.weight"))
            );
            assert_eq!(
                payload(m.shared_scalar),
                stored_payload(&format!("{stem}shared_expert_gate.weight"))
            );
            assert_eq!(
                first_slice(m.gate),
                stored_first(&format!("{stem}switch_mlp.gate_proj.weight"))
            );
            assert_eq!(
                first_slice(m.up),
                stored_first(&format!("{stem}switch_mlp.up_proj.weight"))
            );
            assert_eq!(
                first_slice(m.down),
                stored_first(&format!("{stem}switch_mlp.down_proj.weight"))
            );
            assert_eq!(
                payload(m.shared_gate),
                stored_payload(&format!("{stem}shared_expert.gate_proj.weight"))
            );
            assert_eq!(
                payload(m.shared_up),
                stored_payload(&format!("{stem}shared_expert.up_proj.weight"))
            );
            assert_eq!(
                payload(m.shared_down),
                stored_payload(&format!("{stem}shared_expert.down_proj.weight"))
            );
        }
        let floats = |key: &str| -> Vec<f32> {
            entry(&all, key)
                .3
                .chunks(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect()
        };
        assert_eq!(a.q_norm, floats(&format!("{p}1.self_attn.q_norm.weight")));
        assert_eq!(a.k_norm, floats(&format!("{p}1.self_attn.k_norm.weight")));
        assert_eq!(d.conv, floats(&format!("{p}0.linear_attn.conv1d.weight")));
        assert_eq!(d.a_log, floats(&format!("{p}0.linear_attn.A_log")));
        assert_eq!(d.dt_bias, floats(&format!("{p}0.linear_attn.dt_bias")));
        assert_eq!(d.norm, floats(&format!("{p}0.linear_attn.norm.weight")));
        assert_eq!(
            model.layers[0].input_norm,
            floats(&format!("{p}0.input_layernorm.weight"))
        );
        assert_eq!(
            model.layers[1].post_norm,
            floats(&format!("{p}1.post_attention_layernorm.weight"))
        );
        assert_eq!(model.final_norm, floats("language_model.model.norm.weight"));
    }

    #[test]
    fn the_cpu_projector_matches_projections_by_definition_over_three_steps() {
        let ckpt = checkpoint(entries());
        let weights = HostWeights::load(&ckpt, &TINY, &config()).unwrap();
        let (shape, model) = (weights.shape(), weights.view());
        let mut cpu = CpuProjector::new(&ckpt, 2);
        let mut naive = NaiveProjector(&ckpt);
        let (mut cpu_state, mut naive_state) = (HostState::new(shape), HostState::new(shape));
        for token in [3u32, 1, 7] {
            let a = step(&mut cpu, shape, &model, &mut cpu_state, token, true).unwrap();
            let b = step(&mut naive, shape, &model, &mut naive_state, token, true).unwrap();
            assert!(a.logits.iter().all(|v| v.is_finite()));
            assert!(
                a.logits.iter().any(|v| (v - a.logits[0]).abs() > 1e-3),
                "logits are not constant"
            );
            for (i, (x, y)) in a.logits.iter().zip(&b.logits).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-4 * (1.0 + y.abs()),
                    "token {token} logit {i}: {x} vs {y}"
                );
            }
            for (layer, (x, y)) in a.layer_outputs.iter().zip(&b.layer_outputs).enumerate() {
                for (i, (u, v)) in x.iter().zip(y).enumerate() {
                    assert!(
                        (u - v).abs() <= 1e-4 * (1.0 + v.abs()),
                        "layer {layer}[{i}]: {u} vs {v}"
                    );
                }
            }
        }
        assert_eq!(cpu_state.pos(), 3);
    }

    #[test]
    fn overflowing_mrope_sections_return_a_typed_refusal() {
        let mut root: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/qwen3_6_35b_a3b_config.json"
        ))
        .unwrap();
        let head_dim = crate::family::spec::QWEN3_6_35B_A3B.head_dim;
        assert_eq!(
            rotary_dim(&serde_json::to_vec(&root).unwrap(), head_dim).unwrap(),
            64
        );
        root["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([u64::MAX, 33]);
        assert!(matches!(
            rotary_dim(&serde_json::to_vec(&root).unwrap(), head_dim),
            Err(EngError::Checkpoint(message)) if message.contains("mrope_section")
        ));
    }

    #[test]
    fn rotary_dim_checks_the_fields_that_check_config_skips() {
        let edit = |f: &dyn Fn(&mut Value)| {
            let mut root = config_json();
            f(&mut root["text_config"]);
            rotary_dim(&serde_json::to_vec(&root).unwrap(), 32)
        };
        assert_eq!(edit(&|_| {}).unwrap(), 16);
        let message = |r: Result<u32, EngError>| match r {
            Err(EngError::Checkpoint(m)) => m,
            other => panic!("expected a checkpoint error, got {other:?}"),
        };
        assert!(message(edit(&|t| t["partial_rotary_factor"] = 0.25.into())).contains("differs"));
        assert!(
            message(edit(&|t| t["rope_parameters"]["rope_type"] = "yarn".into()))
                .contains("default rope")
        );
        assert!(
            message(edit(
                &|t| t["rope_parameters"]["mrope_section"] = serde_json::json!([3, 3, 3])
            ))
            .contains("mrope_section")
        );
        assert!(
            message(edit(&|t| {
                t["partial_rotary_factor"] = 0.3.into();
                t["rope_parameters"]["partial_rotary_factor"] = 0.3.into();
            }))
            .contains("whole dimension")
        );
        assert!(
            message(edit(
                &|t| t["rope_parameters"] = serde_json::json!({"rope_theta": 10000.0})
            ))
            .contains("partial_rotary_factor")
        );
    }

    #[test]
    fn load_rejects_a_changed_config_a_missing_tensor_and_a_wrong_width() {
        let ckpt = checkpoint(entries());
        let mut changed = config_json();
        changed["text_config"]["num_experts"] = 8.into();
        let result = HostWeights::load(&ckpt, &TINY, &serde_json::to_vec(&changed).unwrap());
        assert!(matches!(result, Err(EngError::Checkpoint(m)) if m.contains("num_experts")));

        let mut missing = entries();
        missing.retain(|e| e.0 != "language_model.model.layers.1.self_attn.v_proj.biases");
        let result = HostWeights::load(&checkpoint(missing), &TINY, &config());
        assert!(matches!(result, Err(EngError::Checkpoint(m)) if m.contains("v_proj")));

        // Without the router override the plan reads the 8-bit router payload as 4-bit.
        let mut no_override = config_json();
        no_override["quantization"]
            .as_object_mut()
            .unwrap()
            .retain(|_, v| !v.is_object());
        let result = HostWeights::load(&ckpt, &TINY, &serde_json::to_vec(&no_override).unwrap());
        assert!(matches!(
            result,
            Err(EngError::Checkpoint(m)) if m.contains("payload shape must be [4, 8], received [4, 16]")
        ));
    }
}
