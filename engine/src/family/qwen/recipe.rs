//! The layer list of a model of the Qwen lineage, built from what its GGUF header says.
//!
//! The lineage shares one block: a mixer that is the delta rule or the gated attention, then a
//! gated feed-forward, each behind an RMS norm. Models differ in which blocks attend, in whether
//! a range of blocks is walked more than once, and in whether a range routes two residual
//! streams. Those differences are data here; no model has a walk of its own.

use crate::EngError;
use crate::family::qwen::sensenova::{LoopShape, RopePlan};
use crate::family::step::delta::DeltaNames;
use crate::family::step::moe;
use crate::family::step::recipe::{Forward, Layer, Mixer, Site};
use crate::weights::{GgufHeader, MetaValue};

/// One visit of the trunk, in the order the walk takes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Visit {
    /// The block whose tensors the visit reads (`blk.<block>`).
    pub(crate) block: usize,
    /// Whether the mixer is the delta rule; otherwise it is the gated attention.
    pub(crate) recurrent: bool,
    pub(crate) streams: Streams,
}

/// How a visit holds its residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Streams {
    One,
    /// Two routed streams; `fork` starts both from the one residual.
    Two {
        fork: bool,
    },
}

/// The visits of a trunk whose every `interval`-th block attends and which has no repeated
/// range: `block_count` visits, one per block.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the device test that runs the routed-expert model on the lineage step; the served start takes it when that model is cut over"
    )
)]
pub(crate) fn every_nth_attends(block_count: usize, interval: usize) -> Vec<Visit> {
    (0..block_count)
        .map(|block| Visit {
            block,
            recurrent: interval == 0 || (block + 1) % interval != 0,
            streams: Streams::One,
        })
        .collect()
}

/// What a file of the lineage stores in every block, beside the visit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Blocks {
    /// Whether the delta rule's four input projections are stored apart; otherwise they are
    /// fused into two matrices.
    pub(crate) split_delta_inputs: bool,
    /// Whether the feed-forward is a routed block of experts with a gated shared expert;
    /// otherwise it is one dense gated feed-forward.
    pub(crate) experts: bool,
}

impl Blocks {
    fn forward(self, stem: &str) -> Forward {
        let three = |tail: &str| {
            ["gate", "up", "down"].map(|part| format!("{stem}.ffn_{part}{tail}.weight"))
        };
        if self.experts {
            Forward::Routed(moe::Names {
                router: format!("{stem}.ffn_gate_inp.weight"),
                rule: moe::Router::Softmax {
                    shared_gate: format!("{stem}.ffn_gate_inp_shexp.weight"),
                },
                stacks: three("_exps"),
                shared: three("_shexp"),
            })
        } else {
            Forward::Dense(three(""))
        }
    }
}

/// A model of the lineage as its header describes it: what the step is opened with.
pub(crate) struct Model {
    pub(crate) family: &'static str,
    pub(crate) shape: LoopShape,
    pub(crate) visits: Vec<Visit>,
    pub(crate) blocks: Blocks,
    /// The experts of a routed feed-forward and how many a row keeps.
    pub(crate) experts: Option<(usize, usize)>,
    pub(crate) rope: RopePlan,
}

/// The architecture name of the routed-expert models of the lineage.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the device test that runs the routed-expert model on the lineage step; the served start takes it when that model is cut over"
    )
)]
pub(crate) const ROUTED_EXPERTS: &str = "qwen35moe";

/// The routed-expert model of `header`: every `full_attention_interval`-th block attends, the
/// delta inputs are stored apart, the feed-forward is a block of experts beside a gated shared
/// expert, and the rotary table has no scaling. Refuses a header of another architecture.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the device test that runs the routed-expert model on the lineage step; the served start takes it when that model is cut over"
    )
)]
pub(crate) fn routed_experts(header: &GgufHeader) -> Result<Model, EngError> {
    let arch = ROUTED_EXPERTS;
    let reject = |what: String| EngError::Checkpoint(format!("{arch}: {what}"));
    if header.meta("general.architecture") != Some(&MetaValue::String(arch.to_owned())) {
        return Err(reject("general.architecture is another model's".to_owned()));
    }
    let count = |key: &str| -> Result<usize, EngError> {
        header
            .meta_u32(&format!("{arch}.{key}"))
            .map(|v| v as usize)
            .filter(|v| *v > 0)
            .ok_or_else(|| reject(format!("{key} is missing, zero or not a u32")))
    };
    let real = |key: &str| -> Result<f32, EngError> {
        match header.meta(&format!("{arch}.{key}")) {
            Some(MetaValue::F32(v)) if v.is_finite() && *v > 0.0 => Ok(*v),
            _ => Err(reject(format!("{key} is missing or not a positive f32"))),
        }
    };
    let (value_heads, inner) = (count("ssm.time_step_rank")?, count("ssm.inner_size")?);
    let expert_width = count("expert_feed_forward_length")?;
    if expert_width != count("expert_shared_feed_forward_length")? {
        return Err(reject(
            "the shared expert is not as wide as the experts".to_owned(),
        ));
    }
    if !inner.is_multiple_of(value_heads) {
        return Err(reject(
            "the delta value heads do not divide its inner width".to_owned(),
        ));
    }
    let head_dim = count("attention.key_length")?;
    if head_dim != count("attention.value_length")? {
        return Err(reject("attention key and value lengths differ".to_owned()));
    }
    let vocab = match header.meta("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(meta)) => meta.len as usize,
        _ => return Err(reject("tokenizer.ggml.tokens is missing".to_owned())),
    };
    let context = count("context_length")?;
    let shape = LoopShape {
        hidden: count("embedding_length")?,
        feed_forward: expert_width,
        heads_q: count("attention.head_count")?,
        heads_kv: count("attention.head_count_kv")?,
        head_dim,
        delta_key_heads: count("ssm.group_count")?,
        delta_key_dim: count("ssm.state_size")?,
        delta_value_heads: value_heads,
        delta_value_dim: inner / value_heads,
        conv_kernel: count("ssm.conv_kernel")?,
        vocab,
        context,
        eps: real("attention.layer_norm_rms_epsilon")?,
    };
    let dims = count("rope.dimension_count")?;
    if !dims.is_multiple_of(2) || dims > head_dim {
        return Err(reject(
            "rope.dimension_count is odd or wider than a head".to_owned(),
        ));
    }
    Ok(Model {
        family: "qwen",
        shape,
        visits: every_nth_attends(count("block_count")?, count("full_attention_interval")?),
        blocks: Blocks {
            split_delta_inputs: true,
            experts: true,
        },
        experts: Some((count("expert_count")?, count("expert_used_count")?)),
        // No scaling: a factor of one leaves every frequency as the base gives it.
        rope: RopePlan {
            dims,
            sections: [dims / 2, 0, 0, 0],
            freq_base: real("rope.freq_base")?,
            factor: 1.0,
            original_context: context,
        },
    })
}

/// The slots a layer list needs: one delta state per visit that carries one, one attention
/// cache per visit that attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotCount {
    pub(crate) delta: usize,
    pub(crate) attend: usize,
}

/// The layers of `visits`. A delta visit takes the slot of its position in the walk; the
/// attention visits take cache slots in walk order.
pub(crate) fn layers(visits: &[Visit], blocks: Blocks) -> (Vec<Layer>, SlotCount) {
    let mut attend = 0;
    let layers = visits
        .iter()
        .enumerate()
        .map(|(at, visit)| {
            let stem = format!("blk.{}", visit.block);
            let n = |name: &str| format!("{stem}.{name}");
            let router = |part: &str| [n(&format!("{part}.weight")), n(&format!("{part}.bias"))];
            Layer {
                norms: [n("attn_norm.weight"), n("post_attention_norm.weight")].map(Some),
                site: match visit.streams {
                    Streams::One => Site::Plain,
                    Streams::Two { fork } => Site::Routed {
                        routers: [router("hc_attn_fn"), router("hc_ffn_fn")],
                        fork,
                    },
                },
                mixer: if visit.recurrent {
                    Mixer::Delta {
                        names: Box::new(if blocks.split_delta_inputs {
                            DeltaNames::split(&stem)
                        } else {
                            DeltaNames::fused(&stem)
                        }),
                        slot: at,
                    }
                } else {
                    attend += 1;
                    Mixer::Attend {
                        stem: stem.clone(),
                        slot: attend - 1,
                    }
                },
                forward: blocks.forward(&stem),
            }
        })
        .collect();
    let slots = SlotCount {
        delta: visits.len(),
        attend,
    };
    (layers, slots)
}

/// Every tensor a card holds for `layers`, in load order: the head, each block's tensors once
/// (a block walked twice is loaded once), the final norm. The embedding stays on the host.
pub(crate) fn tensors(layers: &[Layer]) -> Vec<String> {
    let mut names = vec!["output.weight".to_owned()];
    let mut seen = std::collections::HashSet::new();
    for name in layers.iter().flat_map(Layer::tensors) {
        if seen.insert(name.clone()) {
            names.push(name);
        }
    }
    names.push("output_norm.weight".to_owned());
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_interval_of_four_puts_attention_on_every_fourth_block() {
        let visits = every_nth_attends(40, 4);
        let attending: Vec<usize> = visits
            .iter()
            .filter(|v| !v.recurrent)
            .map(|v| v.block)
            .collect();
        assert_eq!(attending, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
        assert!(visits.iter().all(|v| v.streams == Streams::One));
    }

    #[test]
    fn attention_visits_take_cache_slots_in_walk_order() {
        let blocks = Blocks {
            split_delta_inputs: true,
            experts: true,
        };
        let (layers, slots) = layers(&every_nth_attends(8, 4), blocks);
        let Forward::Routed(names) = &layers[0].forward else {
            panic!("a block of experts is a routed feed-forward");
        };
        assert_eq!(names.stacks[2], "blk.0.ffn_down_exps.weight");
        assert_eq!(names.shared[0], "blk.0.ffn_gate_shexp.weight");
        assert_eq!(names.rule.side(), "blk.0.ffn_gate_inp_shexp.weight");
        assert_eq!(
            slots,
            SlotCount {
                delta: 8,
                attend: 2
            }
        );
        let caches: Vec<(usize, &str)> = layers
            .iter()
            .filter_map(|layer| match &layer.mixer {
                Mixer::Attend { stem, slot } => Some((*slot, stem.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(caches, [(0, "blk.3"), (1, "blk.7")]);
        assert!(matches!(&layers[2].mixer, Mixer::Delta { slot: 2, .. }));
        assert_eq!(
            layers[5].norms[1].as_deref(),
            Some("blk.5.post_attention_norm.weight")
        );
    }
}
