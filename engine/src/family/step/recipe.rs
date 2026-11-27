//! The recipe of a model: one [`Layer`] per visit, each naming its tensors in the weight map,
//! and the one walk over them.
//!
//! A visit has two halves, a mixer and a feed-forward. Each half reads its input at the layer's
//! residual site, norms it, runs its branch and applies the branch at the site. A model
//! supplies the list, its widths, the rows it works in and the mixers nobody else has;
//! [`Walk`] is the walk every model runs.

use super::attend::{self, At, AttendRows};
use super::delta::{self, ChannelRows, DeltaNames, DeltaRows};
use super::depth::{DepthSite, Mix};
use super::rows::{Map, Norm, StepIo, q8, raw};
use super::scratch::Core;
use super::site::{self, MixWeights};
use super::state::State;
use super::walk::{Half, Visits};
use super::{ffn, moe};
use crate::EngError;
use crate::kernels::cuda::DeviceBuffer;

/// One visit of a model.
pub(crate) struct Layer {
    /// The norm weights of the mixer half and of the feed-forward half. `None`: the site's read
    /// writes the normed branch input itself, and the walk launches no norm for that half (a
    /// plain or routed site always takes one).
    pub(crate) norms: [Option<String>; 2],
    pub(crate) site: Site,
    pub(crate) mixer: Mixer,
    pub(crate) forward: Forward,
}

/// How a layer joins its branches to the residual.
pub(crate) enum Site {
    /// One stream: the branches add onto it.
    Plain,
    /// Two streams a router mixes before each branch and merges after it. `routers` names the
    /// weight and the bias of the router of each half; `fork` starts both streams from the
    /// residual.
    Routed {
        routers: [[String; 2]; 2],
        fork: bool,
    },
    /// Four streams mixed by learned weights. `halves` names the mix, base and scale tensors of
    /// each half.
    Mixed { halves: [[String; 3]; 2] },
    /// Attention residuals over depth: one current row and an append-only list of sealed,
    /// RMS-normed block rows. Each half reads a softmax mix of the sealed rows and the normed
    /// current row under its learned query; a `None` query means that half reads the current
    /// row as it stands, with no mix (a first layer has no sealed rows for its attention half to
    /// mix). The forward half's branch adds onto the current row. The mixer half's branch adds
    /// onto it too unless `seal` is set: then the current row is sealed (appended, normed) and
    /// the branch output becomes the new current row, so a block starts at its first visit's
    /// attention. `repeat` places the visit in a range of layers the model visits more than
    /// once. A site with no query, no seal and no place in a repeat is [`Site::Plain`], and
    /// [`Layer::check`] refuses it.
    DepthAttention {
        queries: [Option<String>; 2],
        seal: bool,
        repeat: Repeat,
    },
}

/// Where a visit stands in a range of layers that the model visits more than once (passes over
/// the same weights). A fact of the layer list, carried by the site that needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Repeat {
    /// Any visit not at a pass boundary, inside the range or outside it.
    Step,
    /// The first visit of the first pass: the state held here is the state every pass starts
    /// from.
    Mark,
    /// The first visit of a later pass: the state goes back to the one held at the mark.
    Rewind,
    /// The first visit after the last pass.
    Leave,
}

pub(crate) enum Mixer {
    /// The delta rule, carrying its state in slot `slot`.
    Delta { names: Box<DeltaNames>, slot: usize },
    /// The gated attention of block `stem`, caching in slot `slot`.
    Attend { stem: String, slot: usize },
    /// The low-rank attention: one cached latent row per position, the layer's cache and
    /// factors in slot `slot`.
    Latent {
        names: Box<attend::latent::Names>,
        slot: usize,
    },
    /// A mixer of the model's own.
    Own,
}

pub(crate) enum Forward {
    /// The gate, up and down matrices of a gated feed-forward.
    Dense([String; 3]),
    Routed(moe::Names),
    /// A feed-forward of the model's own.
    Own,
}

/// Where a half stands when a model may look at its rows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// The normed input of the branch is written.
    Read,
    Branch,
    /// The residual holds the branch.
    Apply,
}

impl Layer {
    /// Every tensor the layer names, in the order a load takes them. A mixer or a feed-forward
    /// of the model's own names none here.
    pub(crate) fn tensors(&self) -> Vec<String> {
        let mut names: Vec<String> = self.norms.iter().flatten().cloned().collect();
        match &self.site {
            Site::Routed { routers, .. } => names.extend(routers.iter().flatten().cloned()),
            Site::Mixed { halves } => names.extend(halves.iter().flatten().cloned()),
            Site::DepthAttention { queries, .. } => {
                names.extend(queries.iter().flatten().cloned());
            }
            Site::Plain => {}
        }
        match &self.mixer {
            Mixer::Delta { names: delta, .. } => {
                names.extend(delta.tensors().into_iter().map(str::to_owned));
            }
            Mixer::Attend { stem, .. } => names.extend(attend::tensors(stem)),
            Mixer::Latent { names: latent, .. } => {
                names.extend(latent.tensors().into_iter().map(str::to_owned));
            }
            Mixer::Own => {}
        }
        match &self.forward {
            Forward::Dense(dense) => names.extend(dense.iter().cloned()),
            Forward::Routed(routed) => {
                names.push(routed.router.clone());
                names.push(routed.rule.side().to_owned());
                names.extend(routed.stacks.iter().chain(&routed.shared).cloned());
            }
            Forward::Own => {}
        }
        names
    }

    /// Resolves every tensor the layer names, so a missing one stops a load and not a step.
    pub(crate) fn check(
        &self,
        map: &Map<'_>,
        delta: delta::Dims,
        attend: Option<attend::Dims>,
    ) -> Result<(), EngError> {
        let site_norms = !matches!(self.site, Site::Plain | Site::Routed { .. });
        if !site_norms && self.norms.iter().any(Option::is_none) {
            return Err(EngError::BadRequest(format!(
                "{}: a plain or routed site needs the norm of both halves",
                map.family
            )));
        }
        for name in self.norms.iter().flatten() {
            map.raw(name)?;
        }
        let raws: Vec<&String> = match &self.site {
            Site::Plain => Vec::new(),
            Site::Routed { routers, .. } => routers.iter().flatten().collect(),
            Site::Mixed { halves } => halves.iter().flatten().collect(),
            Site::DepthAttention {
                queries: [None, None],
                seal: false,
                repeat: Repeat::Step,
            } => {
                return Err(EngError::BadRequest(format!(
                    "{}: a depth-attention site with no query, no seal and no repeat is a plain \
                     site",
                    map.family
                )));
            }
            Site::DepthAttention { queries, .. } => queries.iter().flatten().collect(),
        };
        for name in raws {
            map.raw(name)?;
        }
        match (&self.mixer, attend) {
            (Mixer::Delta { names, .. }, _) => drop(names.resolve(map, delta)?),
            (Mixer::Attend { stem, .. }, Some(dims)) => drop(attend::resolve(map, stem, dims)?),
            (Mixer::Attend { .. }, None) => {
                return Err(EngError::BadRequest(format!(
                    "{}: an attention layer needs the attention widths",
                    map.family
                )));
            }
            (Mixer::Latent { names, .. }, _) => names.check(map)?,
            (Mixer::Own, _) => {}
        }
        match &self.forward {
            Forward::Dense(names) => names.iter().try_for_each(|n| map.native(n).map(drop)),
            Forward::Routed(names) => {
                map.raw(&names.router)?;
                map.raw(names.rule.side())?;
                let stacks = names.stacks.iter().chain(&names.shared);
                stacks.into_iter().try_for_each(|n| map.native(n).map(drop))
            }
            Forward::Own => Ok(()),
        }
    }
}

/// The sealed rows the states of `site` hold as visit `visit` starts.
fn site_blocks(site: &DepthSite<'_>, visit: usize) -> Result<usize, EngError> {
    site.blocks.get(visit).copied().ok_or_else(|| {
        EngError::BadRequest(format!(
            "the depth site has no sealed-row count for visit {visit}"
        ))
    })
}

fn unsupported<T>(family: &str, what: &str) -> Result<T, EngError> {
    Err(EngError::BadRequest(format!(
        "{family}: the recipe names {what}, which this model does not run"
    )))
}

/// What a model gives the walk beside its layers.
pub(crate) trait Model {
    fn family(&self) -> &'static str;
    fn layers(&self) -> &[Layer];
    /// What the projections share; `exact` puts them on the decoding routes.
    fn io(&self, exact: bool) -> StepIo<'_>;
    /// Whether the branches of an `n`-row chunk take the decoding routes.
    fn exact(&self, _n: usize) -> bool {
        false
    }
    fn core(&self) -> &Core;
    fn state(&self) -> &State;
    fn hidden(&self) -> usize;
    /// The eps of the norms of both halves.
    fn eps(&self) -> f32;
    /// What a chunk of `n` rows needs before its first visit.
    fn begin(&self, _n: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// What visit `visit` of a chunk of `n` rows at `p0` needs before its first half (see
    /// [`Visits::enter`]).
    fn enter(&self, _visit: usize, _n: usize, _p0: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// The widths of the delta-rule layers and the rows of the per-channel rule.
    fn delta(&self) -> (delta::Dims, Option<ChannelRows<'_>>);
    /// The widths, the rows and the place of an attention visit over `n` rows at `p0`.
    fn attend(
        &self,
        _n: usize,
        _p0: usize,
    ) -> Result<(attend::Dims, AttendRows<'_>, At<'_>), EngError> {
        unsupported(self.family(), "an attention layer")
    }
    /// The widths and the rows of a low-rank attention visit over `n` rows, with the cache and
    /// the factors of slot `slot`.
    fn latent(
        &self,
        _slot: usize,
        _n: usize,
    ) -> Result<(attend::latent::Dims, attend::latent::LatentRows<'_>), EngError> {
        unsupported(self.family(), "a low-rank attention layer")
    }
    /// The mixer of visit `visit`, for a layer whose mixer is the model's own.
    fn own_mixer(&self, _visit: usize, _n: usize, _p0: usize) -> Result<(), EngError> {
        unsupported(self.family(), "a mixer of the model's own")
    }
    /// The feed-forward of visit `visit`, for a layer whose feed-forward is the model's own: it
    /// reads the normed rows and writes the forward rows, as the shared blocks do.
    fn own_forward(&self, _visit: usize, _n: usize, _p0: usize) -> Result<(), EngError> {
        unsupported(self.family(), "a feed-forward of the model's own")
    }
    /// The activation and the scratch of a dense feed-forward over `n` rows.
    fn dense(&self, n: usize) -> Result<(ffn::Join, ffn::Scratch<'_>), EngError>;
    /// The widths, the activation, the scratch and the shared expert's scratch of a routed
    /// feed-forward over `n` rows.
    #[allow(clippy::type_complexity)]
    fn routed(
        &self,
        _n: usize,
    ) -> Result<((moe::Dims, ffn::Join), &moe::Scratch, ffn::Scratch<'_>), EngError> {
        unsupported(self.family(), "a routed feed-forward")
    }
    fn routed_site(&self) -> Result<site::Routed<'_>, EngError> {
        unsupported(self.family(), "a routed residual site")
    }
    fn mixed_site(&self) -> Result<site::Mixed<'_>, EngError> {
        unsupported(self.family(), "a mixed residual site")
    }
    fn depth_site(&self) -> Result<DepthSite<'_>, EngError> {
        unsupported(self.family(), "a depth-attention residual site")
    }
    /// Called after every stage of every half.
    fn tap(&self, _visit: usize, _half: Half, _stage: Stage) -> Result<(), EngError> {
        Ok(())
    }
}

/// The walk of a model over its layers.
pub(crate) struct Walk<'m, M>(pub(crate) &'m M);

impl<M: Model> Walk<'_, M> {
    fn layer(&self, visit: usize) -> Result<&Layer, EngError> {
        self.0.layers().get(visit).ok_or_else(|| {
            EngError::BadRequest(format!("{}: no layer for visit {visit}", self.0.family()))
        })
    }

    /// The rows the branch of `half` writes.
    fn branch_rows(&self, half: Half) -> &DeviceBuffer {
        match half {
            Half::Mixer => &self.0.core().mix_out,
            Half::Forward => &self.0.core().forward_out,
        }
    }
}

impl<M: Model> Visits for Walk<'_, M> {
    fn count(&self) -> usize {
        self.0.layers().len()
    }

    fn begin(&self, n: usize) -> Result<(), EngError> {
        self.0.core().cur.set(0);
        self.0.begin(n)
    }

    /// The model's own entry runs first, so it still sees the states a rewind replaces (the
    /// last pass's, for a model that reads across passes); then a depth site marks or rewinds.
    fn enter(&self, visit: usize, n: usize, p0: usize) -> Result<(), EngError> {
        self.0.enter(visit, n, p0)?;
        let Site::DepthAttention { repeat, .. } = self.layer(visit)?.site else {
            return Ok(());
        };
        let io = self.0.io(false);
        match repeat {
            Repeat::Mark => self.0.depth_site()?.depth.mark(io.ctx, io.stream, n),
            Repeat::Rewind => self.0.depth_site()?.depth.rewind(io.ctx, io.stream, n),
            Repeat::Step | Repeat::Leave => Ok(()),
        }
    }

    /// A plain site reads the residual, and its feed-forward read adds the mixer rows in the
    /// same launch; the other sites read the input their mix writes.
    fn read(&self, visit: usize, half: Half, n: usize) -> Result<(), EngError> {
        let (m, layer) = (self.0, self.layer(visit)?);
        let (core, io, h) = (m.core(), m.io(false), m.hidden());
        let cur = &core.resid[core.cur.get()];
        let (x, delta) = match &layer.site {
            Site::Plain => {
                let branch = match half {
                    Half::Mixer => None,
                    Half::Forward => Some(core.mix_out.range(0, n * h * 4)?),
                };
                (cur.range(0, n * h * 4)?, branch)
            }
            Site::Routed { routers, fork } => {
                let routed = m.routed_site()?;
                if *fork && half == Half::Mixer {
                    routed.fork(&io, cur, n)?;
                }
                let [weight, bias] = &routers[half as usize];
                let tensor = |name: &str| raw(io.weights, io.family, name);
                routed.route(&io, tensor(weight)?, tensor(bias)?, n)?;
                (routed.input.range(0, n * h * 4)?, None)
            }
            Site::Mixed { halves } => {
                let mixed = m.mixed_site()?;
                let [mix, base, scale] = &halves[half as usize];
                let weights = MixWeights {
                    mix: io.raw(mix)?,
                    base: io.raw(base)?,
                    scale: io.raw(scale)?,
                };
                mixed.read(&io, weights, cur, n)?;
                (mixed.input.range(0, n * h * 4)?, None)
            }
            Site::DepthAttention { queries, seal, .. } => {
                let site = m.depth_site()?;
                let blocks = site_blocks(&site, visit)?;
                // The forward half reads after the mixer half's seal.
                let blocks = blocks + usize::from(*seal && half == Half::Forward);
                let query = match &queries[half as usize] {
                    // A query may hold several rows (one per input of the model's own mixer):
                    // the walk mixes under the first, and such a mixer mixes the others itself.
                    Some(name) => Some(io.raw(name)?.slice(0, h * 4)?),
                    None => None,
                };
                let mix = Mix {
                    query,
                    norm: None,
                    first_block: 0,
                    out: site.input.range(0, n * h * 4)?,
                    stride: h,
                };
                site.depth.mix(io.ctx, io.stream, mix, blocks, n)?;
                (site.input.range(0, n * h * 4)?, None)
            }
        };
        if let Some(weight) = &layer.norms[half as usize] {
            let norm = Norm {
                d: h,
                eps: m.eps(),
                x,
                delta,
                weight: io.raw(weight)?.slice(0, h * 4)?,
            };
            io.norm(
                n,
                norm,
                core.normed.range(0, n * h * 4)?,
                core.normed_q8.range(0, q8(h, n))?,
            )?;
        }
        m.tap(visit, half, Stage::Read)
    }

    fn branch(&self, visit: usize, half: Half, n: usize, p0: usize) -> Result<(), EngError> {
        let (m, layer) = (self.0, self.layer(visit)?);
        let (core, io) = (m.core(), m.io(m.exact(n)));
        match half {
            Half::Mixer => match &layer.mixer {
                Mixer::Delta { names, slot } => {
                    let (dims, channel) = m.delta();
                    let resolved = names.resolve(&io.map(), dims)?;
                    let rows = DeltaRows {
                        channel,
                        ..core.delta_rows()
                    };
                    let state = m.state().delta(*slot, &resolved)?;
                    delta::mix(&io, &resolved, &rows, state, n, p0)?;
                }
                Mixer::Attend { stem, slot } => {
                    let (dims, rows, at) = m.attend(n, p0)?;
                    let resolved = attend::resolve(&io.map(), stem, dims)?;
                    let cache = m.state().cache(*slot)?;
                    attend::mix(&io, &resolved, &rows, cache, n, at)?;
                }
                Mixer::Latent { names, slot } => {
                    let (dims, rows) = m.latent(*slot, n)?;
                    attend::latent::mix(&io, names, dims, &rows, n, p0)?;
                }
                Mixer::Own => m.own_mixer(visit, n, p0)?,
            },
            Half::Forward => match &layer.forward {
                Forward::Dense([gate, up, down]) => {
                    let (join, scratch) = m.dense(n)?;
                    let weights = ffn::Weights::named(&io, gate, up, down)?;
                    let x = core.normed_rows();
                    ffn::gated(&io, weights, join, x, n, scratch, &core.forward_out)?;
                }
                Forward::Routed(names) => {
                    let (dims, scratch, shared) = m.routed(n)?;
                    let (x, out) = (core.normed_rows(), &core.forward_out);
                    moe::forward(&io, names, dims, scratch, shared, x, n, out)?;
                }
                Forward::Own => m.own_forward(visit, n, p0)?,
            },
        }
        m.tap(visit, half, Stage::Branch)
    }

    /// A plain site adds the feed-forward rows into the other residual plane. A routed site
    /// merges each branch into both streams and ends on their mean. A mixed site writes the
    /// other plane after each branch.
    fn apply(&self, visit: usize, half: Half, n: usize) -> Result<(), EngError> {
        let (m, layer) = (self.0, self.layer(visit)?);
        let (core, io, h) = (m.core(), m.io(false), m.hidden());
        let (cur, next) = (core.cur.get(), core.cur.get() ^ 1);
        let branch = self.branch_rows(half);
        match &layer.site {
            Site::Plain if half == Half::Mixer => {}
            Site::Plain => {
                let len = n * h;
                let out = core.resid[next].range(0, len * 4)?;
                let x = core.resid[cur].range(0, len * 4)?;
                site::add(&io, out, x, branch.range(0, len * 4)?, len)?;
                core.cur.set(next);
            }
            Site::Routed { .. } => {
                let routed = m.routed_site()?;
                routed.merge(&io, branch, n)?;
                if half == Half::Forward {
                    routed.mean(&io, core.resid[cur].range(0, n * h * 4)?, n)?;
                }
            }
            Site::Mixed { .. } => {
                let mixed = m.mixed_site()?;
                mixed.apply(&io, &core.resid[cur], &core.resid[next], branch, n)?;
                core.cur.set(next);
            }
            Site::DepthAttention { seal, .. } => {
                let site = m.depth_site()?;
                let blocks = site_blocks(&site, visit)?;
                let rows = branch.range(0, n * h * 4)?;
                site.depth
                    .apply(io.ctx, io.stream, half, *seal, rows, blocks, n)?;
            }
        }
        m.tap(visit, half, Stage::Apply)
    }
}
