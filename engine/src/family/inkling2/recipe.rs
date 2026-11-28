//! The model's recipe for the shared step: one layer per visit, on the depth-attention site,
//! with shared relative attention and stacked routed experts.

use super::TextSpec;
use crate::family::step::{
    self,
    recipe::{Forward, Layer, Mixer, Repeat, Site},
};

/// The layers of `spec`'s visits, in visit order.
pub(super) fn layers(spec: &TextSpec) -> Vec<Layer> {
    let core = &spec.core;
    let marks = passes(
        core.core_start,
        core.core_end - core.core_start,
        core.num_passes,
    );
    spec.visits()
        .iter()
        .enumerate()
        .map(|(visit, v)| {
            let layer = v.weight_layer;
            let blk = |name: &str| format!("blk.{layer}.{name}_res_query.weight");
            Layer {
                // The model's halves norm their own inputs (per role for attention).
                norms: [None, None],
                site: Site::DepthAttention {
                    queries: [(layer > 0).then(|| blk("attn")), Some(blk("ffn"))],
                    seal: layer.is_multiple_of(spec.n_layers_per_attn_res_block),
                    repeat: marks(visit),
                },
                mixer: Mixer::Relative { slot: v.state_slot },
                forward: Forward::StackedRouted,
            }
        })
        .collect()
}

/// The repeat place of each visit index for `passes` passes over `width` layers from visit
/// `start`: the mark at the first pass's start, a rewind at each later one and the leave at the
/// first visit after the last.
pub(super) fn passes(start: usize, width: usize, passes: usize) -> impl Fn(usize) -> Repeat {
    let end = start + width * passes;
    move |visit| match visit {
        _ if visit == start => Repeat::Mark,
        _ if visit > start && visit < end && (visit - start).is_multiple_of(width) => {
            Repeat::Rewind
        }
        _ if visit == end => Repeat::Leave,
        _ => Repeat::Step,
    }
}

/// The site of a depth-attention layer: whether its visit seals and where it stands in the
/// passes.
pub(super) fn depth(layer: &Layer) -> (bool, Repeat) {
    match layer.site {
        Site::DepthAttention { seal, repeat, .. } => (seal, repeat),
        _ => (false, Repeat::Step),
    }
}

/// The most sealed blocks a state holds over `layers`, starting from `blocks`. A pass start
/// goes back to the blocks held at the first pass start the walk saw, which is the mark, or
/// the walk's own first pass start when it begins inside the passes.
pub(super) fn peak_blocks<'a>(layers: impl Iterator<Item = &'a Layer>, blocks: usize) -> usize {
    let (mut now, mut peak, mut prelude) = (blocks, blocks, None);
    for (seal, pass) in layers.map(depth) {
        if matches!(pass, Repeat::Mark | Repeat::Rewind) {
            match prelude {
                None => prelude = Some(now),
                Some(blocks) => now = blocks,
            }
        }
        if seal {
            now += 1;
            peak = peak.max(now);
        }
    }
    peak
}

/// What the token chain needs of each visit: whether it saves the prelude, whether it restarts
/// a pass, the sealed blocks it reads, the bank it reads (the blocks of the last pass and the
/// prelude's count plus one) and which of the two banks is active.
pub(super) type Visit = (bool, bool, usize, Option<(usize, usize)>, usize);

/// [`Visit`] for every layer, from no sealed blocks: the counts are the shared depth site's.
pub(super) fn token_layout(layers: &[Layer]) -> Vec<Visit> {
    // `end`: the sealed blocks after the previous visit, which a rewind banks.
    let (mut end, mut bank, mut active) = (0, None, 0);
    layers
        .iter()
        .zip(step::depth::blocks(layers))
        .map(|(layer, blocks)| {
            let (seal, repeat) = depth(layer);
            let restart = repeat == Repeat::Rewind;
            if restart {
                bank = Some((end, blocks + 1));
                active ^= 1;
            }
            if repeat == Repeat::Leave {
                bank = None;
            }
            end = blocks + usize::from(seal);
            (repeat == Repeat::Mark, restart, blocks, bank, active)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layers of `prelude` visits, `passes` passes over `core` layers and `coda` layers, with a
    /// block of `block` layers.
    fn shape(prelude: usize, core: usize, passes: usize, coda: usize, block: usize) -> Vec<Layer> {
        let marks = super::passes(prelude, core, passes);
        let weights = (0..prelude)
            .chain((0..passes).flat_map(|_| prelude..prelude + core))
            .chain(prelude + core..prelude + core + coda);
        weights
            .enumerate()
            .map(|(visit, layer)| Layer {
                norms: [None, None],
                site: Site::DepthAttention {
                    queries: [None, None],
                    seal: layer.is_multiple_of(block),
                    repeat: marks(visit),
                },
                mixer: Mixer::Relative { slot: visit },
                forward: Forward::StackedRouted,
            })
            .collect()
    }

    /// The token chain's layout as it was computed from the visit list, before the recipe.
    fn old_layout(
        visits: &[(usize, usize)],
        (start, end, passes): (usize, usize, usize),
        block: usize,
    ) -> Vec<Visit> {
        let width = end - start;
        let suffix = start + width * passes;
        let (mut blocks, mut pre, mut bank, mut active) = (0, 0, None, 0);
        visits
            .iter()
            .map(|&(i, layer)| {
                let save = i == start;
                if save {
                    pre = blocks;
                }
                let restart = i > start && i < suffix && (i - start).is_multiple_of(width);
                if restart {
                    bank = Some((blocks, pre + 1));
                    blocks = pre;
                    active ^= 1;
                }
                if i == suffix {
                    bank = None;
                }
                let info = (save, restart, blocks, bank, active);
                blocks += usize::from(layer.is_multiple_of(block));
                info
            })
            .collect()
    }

    #[test]
    fn the_recipe_gives_the_token_layout_of_the_visit_list() {
        for (prelude, core, passes, coda, block) in [
            (6, 12, 3, 6, 4),
            (4, 8, 2, 4, 4),
            (2, 6, 8, 1, 3),
            (3, 5, 2, 0, 2),
        ] {
            let layers = shape(prelude, core, passes, coda, block);
            let visits: Vec<(usize, usize)> = (0..prelude)
                .chain((0..passes).flat_map(|_| prelude..prelude + core))
                .chain(prelude + core..prelude + core + coda)
                .enumerate()
                .collect();
            let old = old_layout(&visits, (prelude, prelude + core, passes), block);
            assert_eq!(
                token_layout(&layers),
                old,
                "{prelude} + {passes} x {core} + {coda}, block {block}"
            );
        }
    }

    #[test]
    fn peak_blocks_restart_each_pass_from_the_prelude() {
        let layers = shape(6, 12, 3, 6, 4);
        // Seals at layers 0, 4 (prelude), 8, 12, 16 (each pass) and 20 (coda): two before the
        // passes, five at each pass's end, six after the coda.
        assert_eq!(peak_blocks(layers.iter(), 0), 6);
        assert_eq!(peak_blocks(layers.iter(), 3), 9);
        // From the second pass's start the prelude is that pass's own start.
        assert_eq!(peak_blocks(layers[18..].iter(), 2), 6);
        // One pass: no restart.
        assert_eq!(peak_blocks(shape(6, 12, 1, 0, 4).iter(), 0), 5);
    }

    #[test]
    fn every_visit_names_its_depth_queries() {
        let layers = shape(2, 3, 2, 1, 2);
        assert!(layers.iter().all(|l| l.tensors().is_empty()));
        let named = Layer {
            norms: [None, None],
            site: Site::DepthAttention {
                queries: [None, Some("q".into())],
                seal: false,
                repeat: Repeat::Step,
            },
            mixer: Mixer::Relative { slot: 0 },
            forward: Forward::StackedRouted,
        };
        assert_eq!(named.tensors(), ["q"]);
    }
}
