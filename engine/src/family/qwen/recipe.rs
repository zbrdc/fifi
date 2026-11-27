//! The layer list of a model of the Qwen lineage, built from what its GGUF header says.
//!
//! The lineage shares one block: a mixer that is the delta rule or the gated attention, then a
//! gated feed-forward, each behind an RMS norm. Models differ in which blocks attend, in whether
//! a range of blocks is walked more than once, and in whether a range routes two residual
//! streams. Those differences are data here; no model has a walk of its own.

use crate::family::step::delta::DeltaNames;
use crate::family::step::recipe::{Forward, Layer, Mixer, Site};

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
/// range: `block_count` visits, one per block. The first model to take it is the routed-expert
/// one, once the step has its block.
#[cfg(test)]
pub(crate) fn every_nth_attends(block_count: usize, interval: usize) -> Vec<Visit> {
    (0..block_count)
        .map(|block| Visit {
            block,
            recurrent: interval == 0 || (block + 1) % interval != 0,
            streams: Streams::One,
        })
        .collect()
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
pub(crate) fn layers(visits: &[Visit]) -> (Vec<Layer>, SlotCount) {
    let mut attend = 0;
    let layers = visits
        .iter()
        .enumerate()
        .map(|(at, visit)| {
            let stem = format!("blk.{}", visit.block);
            let n = |name: &str| format!("{stem}.{name}");
            let router = |part: &str| [n(&format!("{part}.weight")), n(&format!("{part}.bias"))];
            Layer {
                norms: [n("attn_norm.weight"), n("post_attention_norm.weight")],
                site: match visit.streams {
                    Streams::One => Site::Plain,
                    Streams::Two { fork } => Site::Routed {
                        routers: [router("hc_attn_fn"), router("hc_ffn_fn")],
                        fork,
                    },
                },
                mixer: if visit.recurrent {
                    Mixer::Delta {
                        names: Box::new(DeltaNames::fused(&stem)),
                        slot: at,
                    }
                } else {
                    attend += 1;
                    Mixer::Attend {
                        stem: stem.clone(),
                        slot: attend - 1,
                    }
                },
                forward: Forward::Dense(
                    ["gate", "up", "down"].map(|t| n(&format!("ffn_{t}.weight"))),
                ),
            }
        })
        .collect();
    let slots = SlotCount {
        delta: visits.len(),
        attend,
    };
    (layers, slots)
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
        let (layers, slots) = layers(&every_nth_attends(8, 4));
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
        assert_eq!(layers[5].norms[1], "blk.5.post_attention_norm.weight");
    }
}
