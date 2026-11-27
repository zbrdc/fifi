//! The depth-attention residual site ([`Site::DepthAttention`](super::recipe::Site)): attention
//! residuals over depth, kept on the device.
//!
//! Each of the chunk's `n` rows owns one state, `1 + capacity` rows of `width` floats: its
//! current row, then its sealed rows, each stored RMS-normed. Every row of a chunk crosses the
//! same visits, so all states hold the same number of sealed rows, and that count at each visit
//! is a fact of the layer list ([`blocks`]): the launches take it as an argument and keep no
//! count of their own, so a recorded step replays them as recorded.
//!
//! A read mixes the sealed rows and the normed current row of every state under a learned query
//! ([`Depth::mix`]). The mixer half's branch then either adds onto the current row or, at a
//! sealing visit, becomes the new current row after the old one is sealed ([`Depth::apply`]).
//! The forward half's branch adds. A range of layers visited more than once marks the states at
//! its first visit and rewinds them at each later pass's first visit ([`Depth::mark`],
//! [`Depth::rewind`]). Each state's launches are a block per state, so a chunk row has the bits
//! of a one-row step of the same inputs.

use super::recipe::{Layer, Repeat, Site};
use super::walk::Half;
use crate::EngError;
use crate::kernels::cuda::resident::depth_attention::{
    AddCall, MixStatesCall, MixStatesShape, SealCall, States, StreamShape,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

const F32: usize = size_of::<f32>();

/// The sealed rows every state holds as each visit of `layers` starts, from no sealed rows: a
/// seal adds one after the visit's mixer half, a mark keeps the count and a rewind returns to it.
/// A layer of another site keeps the count.
pub(crate) fn blocks(layers: &[Layer]) -> Vec<usize> {
    let (mut now, mut marked) = (0, 0);
    layers
        .iter()
        .map(|layer| {
            let Site::DepthAttention { seal, repeat, .. } = layer.site else {
                return now;
            };
            match repeat {
                Repeat::Mark => marked = now,
                Repeat::Rewind => now = marked,
                Repeat::Step | Repeat::Leave => {}
            }
            let at = now;
            now += usize::from(seal);
            at
        })
        .collect()
}

/// The depth site of a model: its states and the rows a read writes.
pub(crate) struct Depth<'a> {
    pub(crate) width: usize,
    /// Sealed rows each state has room for.
    pub(crate) capacity: usize,
    /// The eps of the mix's current-row norm and of a seal's norm.
    pub(crate) eps: f32,
    /// Every row's state, `1 + capacity` rows apart.
    pub(crate) states: &'a DeviceBuffer,
    /// A copy of the states, as the mark left them; `None` for a model with no repeated range.
    pub(crate) mark: Option<&'a DeviceBuffer>,
    /// One float: left alone, or quiet NaN once a launch saw a value that was not finite.
    pub(crate) health: &'a DeviceBuffer,
}

/// The depth site a model gives the walk: the site, the sealed rows at each visit
/// ([`blocks`] of its layers) and the rows a read writes, `width` floats per chunk row.
pub(crate) struct DepthSite<'a> {
    pub(crate) depth: Depth<'a>,
    pub(crate) blocks: &'a [usize],
    pub(crate) input: &'a DeviceBuffer,
}

/// What one mix reads besides the states, and where it writes.
pub(crate) struct Mix<'a> {
    /// The depth query, `width` floats; without one the output is the raw current row.
    pub(crate) query: Option<DeviceRange<'a>>,
    /// A norm weight and eps applied to the mix, fused into the same launch.
    pub(crate) norm: Option<(DeviceRange<'a>, f32)>,
    /// The first sealed row the mix reads.
    pub(crate) first_block: usize,
    /// Row `r` of the output starts `r * stride` floats into `out`.
    pub(crate) out: DeviceRange<'a>,
    pub(crate) stride: usize,
}

impl Depth<'_> {
    fn states_of(&self, n: usize) -> States {
        States {
            rows: n,
            stride: 1 + self.capacity,
        }
    }

    /// The states of `n` rows through `used` rows of the last one: the exact range a launch
    /// takes.
    fn used(&self, n: usize, used: usize) -> Result<DeviceRange<'_>, EngError> {
        let rows = (n - 1) * (1 + self.capacity) + used;
        self.states.range(0, rows * self.width * F32)
    }

    /// The whole states of `n` rows.
    fn whole<'b>(&self, buffer: &'b DeviceBuffer, n: usize) -> Result<DeviceRange<'b>, EngError> {
        buffer.range(0, n * (1 + self.capacity) * self.width * F32)
    }

    fn health(&self) -> Result<DeviceRange<'_>, EngError> {
        self.health.range(0, F32)
    }

    /// The mix of every one of `n` states holding `blocks` sealed rows.
    pub(crate) fn mix(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        mix: Mix<'_>,
        blocks: usize,
        n: usize,
    ) -> Result<(), EngError> {
        let (gamma, norm_eps) = mix.norm.unzip();
        ctx.depth_attention_mix_states(
            stream,
            MixStatesCall {
                state: self.used(n, blocks + 1)?,
                query: mix.query,
                gamma,
                out: mix.out,
                health: self.health()?,
            },
            MixStatesShape {
                first_block: mix.first_block,
                blocks,
                width: self.width,
                out_stride: mix.stride,
                mix_eps: self.eps,
                norm_eps: norm_eps.unwrap_or(1.0),
            },
            self.states_of(n),
        )
    }

    /// The branch rows `branch` (one row per state, packed) of `half` return to `n` states
    /// holding `blocks` sealed rows: they add onto the current rows, or at a sealing mixer half
    /// the current rows are sealed and the branch rows become the current rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        half: Half,
        seal: bool,
        branch: DeviceRange<'_>,
        blocks: usize,
        n: usize,
    ) -> Result<(), EngError> {
        if !(seal && half == Half::Mixer) {
            return ctx.depth_attention_add_states(
                stream,
                AddCall {
                    state: self.used(n, 1)?,
                    row: branch,
                    health: self.health()?,
                },
                self.width,
                self.states_of(n),
            );
        }
        if blocks >= self.capacity {
            return Err(EngError::BadRequest(format!(
                "the depth site has room for {} sealed rows and a visit seals row {blocks}",
                self.capacity
            )));
        }
        ctx.depth_attention_seal_states(
            stream,
            SealCall {
                state: self.used(n, blocks + 2)?,
                next: branch,
                health: self.health()?,
            },
            StreamShape {
                blocks,
                width: self.width,
                eps: self.eps,
            },
            self.states_of(n),
        )
    }

    fn marked(&self) -> Result<&DeviceBuffer, EngError> {
        self.mark.ok_or_else(|| {
            EngError::BadRequest("a depth site with no mark buffer meets a repeat".into())
        })
    }

    /// The states of `n` rows are copied to the mark.
    pub(crate) fn mark(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        n: usize,
    ) -> Result<(), EngError> {
        let mark = self.marked()?;
        ctx.copy_device(stream, self.whole(self.states, n)?, self.whole(mark, n)?)
    }

    /// The states of `n` rows go back to the mark.
    pub(crate) fn rewind(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        n: usize,
    ) -> Result<(), EngError> {
        let mark = self.marked()?;
        ctx.copy_device(stream, self.whole(mark, n)?, self.whole(self.states, n)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::step::recipe::{Forward, Mixer};

    fn layer(seal: bool, repeat: Repeat) -> Layer {
        Layer {
            norms: [None, None],
            site: Site::DepthAttention {
                queries: [None, Some("q".into())],
                seal,
                repeat,
            },
            mixer: Mixer::Own,
            forward: Forward::Own,
        }
    }

    #[test]
    fn blocks_count_seals_and_return_to_the_mark() {
        use Repeat::{Leave, Mark, Rewind, Step};
        // A sealing prelude visit, two passes of a sealing and a plain visit, a sealing coda.
        let layers = [
            layer(true, Step),
            layer(true, Mark),
            layer(false, Step),
            layer(true, Rewind),
            layer(false, Step),
            layer(true, Leave),
        ];
        assert_eq!(blocks(&layers), [0, 1, 2, 1, 2, 2]);
        let plain = Layer {
            site: Site::Plain,
            ..layer(false, Step)
        };
        assert_eq!(
            blocks(&[layer(true, Step), plain, layer(false, Step)]),
            [0, 1, 1]
        );
    }
}
