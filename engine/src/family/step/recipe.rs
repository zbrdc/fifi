//! The recipe of a model: one [`Layer`] per visit, each naming its tensors in the weight map,
//! and the one walk over them.
//!
//! A visit has two halves, a mixer and a feed-forward. Each half reads its input at the layer's
//! residual site, norms it, runs its branch and applies the branch at the site. A model
//! supplies the list, its widths, the rows it works in and the mixers nobody else has;
//! [`Walk`] is the walk every model runs.

use super::attend::{self, At, AttendRows};
use super::delta::{self, ChannelRows, DeltaNames, DeltaRows};
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
    /// The norm weights of the mixer half and of the feed-forward half.
    pub(crate) norms: [String; 2],
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
}

pub(crate) enum Mixer {
    /// The delta rule, carrying its state in slot `slot`.
    Delta { names: Box<DeltaNames>, slot: usize },
    /// The gated attention of block `stem`, caching in slot `slot`.
    Attend { stem: String, slot: usize },
    /// A mixer of the model's own.
    Own,
}

pub(crate) enum Forward {
    /// The gate, up and down matrices of a gated feed-forward.
    Dense([String; 3]),
    Routed(moe::Names),
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
    /// Resolves every tensor the layer names, so a missing one stops a load and not a step.
    pub(crate) fn check(
        &self,
        map: &Map<'_>,
        delta: delta::Dims,
        attend: Option<attend::Dims>,
    ) -> Result<(), EngError> {
        for name in &self.norms {
            map.raw(name)?;
        }
        let raws: Vec<&String> = match &self.site {
            Site::Plain => Vec::new(),
            Site::Routed { routers, .. } => routers.iter().flatten().collect(),
            Site::Mixed { halves } => halves.iter().flatten().collect(),
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
            (Mixer::Own, _) => {}
        }
        match &self.forward {
            Forward::Dense(names) => names.iter().try_for_each(|n| map.native(n).map(drop)),
            Forward::Routed(names) => {
                map.raw(&names.router)?;
                map.raw(&names.bias)?;
                let stacks = names.stacks.iter().chain(&names.shared);
                stacks.into_iter().try_for_each(|n| map.native(n).map(drop))
            }
        }
    }
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
    /// The mixer of visit `visit`, for a layer whose mixer is the model's own.
    fn own_mixer(&self, _visit: usize, _n: usize, _p0: usize) -> Result<(), EngError> {
        unsupported(self.family(), "a mixer of the model's own")
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

    fn enter(&self, visit: usize, n: usize, p0: usize) -> Result<(), EngError> {
        self.0.enter(visit, n, p0)
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
        };
        let norm = Norm {
            d: h,
            eps: m.eps(),
            x,
            delta,
            weight: io.raw(&layer.norms[half as usize])?.slice(0, h * 4)?,
        };
        io.norm(
            n,
            norm,
            core.normed.range(0, n * h * 4)?,
            core.normed_q8.range(0, q8(h, n))?,
        )?;
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
        }
        m.tap(visit, half, Stage::Apply)
    }
}
