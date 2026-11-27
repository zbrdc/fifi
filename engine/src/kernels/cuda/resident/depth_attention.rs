//! The resident attention-residual depth stream, launch-only: the weighted mix over the sealed blocks,
//! the seal of the current row and the add into it.
//!
//! A stream state packs the current row and then `blocks` sealed rows, `width` floats each
//! (`native/resident/depth_attention.h`). `health` is one float per launch: left alone, or quiet
//! NaN when an input or an output of that launch was not finite.

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

use super::super::native::resident::depth_attention::{self as native};
pub(crate) use super::super::native::resident::depth_attention::{
    MAX_BLOCKS, MAX_MIX_JOBS, MAX_NORM_ROWS, MAX_STATES, States,
};

const MAX_WIDTH: usize = 8192;

/// The ranges of a mix launch. `state` holds the current row and then `shape.blocks` sealed rows.
pub(crate) struct MixCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// One mix of [`CudaContext::depth_attention_mix_jobs`]: a state of `blocks + 1` rows whose
/// sealed rows `first_block..blocks` join the mix, its query and its output row.
pub(crate) struct MixJob<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) first_block: usize,
    pub(crate) blocks: usize,
}

/// The sealed rows `first_block..blocks` join the mix, newest first; the current row comes last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MixShape {
    pub(crate) first_block: usize,
    pub(crate) blocks: usize,
    pub(crate) width: usize,
    pub(crate) eps: f32,
}

/// The shape of a seal or add launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StreamShape {
    pub(crate) blocks: usize,
    pub(crate) width: usize,
    pub(crate) eps: f32,
}

/// The ranges of a seal launch. `state` must hold the new sealed row: `blocks + 2` rows.
pub(crate) struct SealCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) next: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The ranges of an add launch.
pub(crate) struct AddCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) row: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The ranges of a mix of several states. Without `query` the output is the raw current row;
/// with `gamma` it is `norm(mix) * gamma` (`CpuForward::norm`).
pub(crate) struct MixStatesCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: Option<DeviceRange<'a>>,
    pub(crate) gamma: Option<DeviceRange<'a>>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The shape of a mix of several states; output rows are `out_stride` floats apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MixStatesShape {
    pub(crate) first_block: usize,
    pub(crate) blocks: usize,
    pub(crate) width: usize,
    pub(crate) out_stride: usize,
    pub(crate) mix_eps: f32,
    pub(crate) norm_eps: f32,
}

/// The ranges of a norm launch: `dst = src * rms_scale(src)`.
pub(crate) struct NormCall<'a> {
    pub(crate) src: DeviceRange<'a>,
    pub(crate) dst: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

fn check_shape(
    what: &str,
    first_block: usize,
    blocks: usize,
    width: usize,
    eps: f32,
) -> Result<(), EngError> {
    if first_block <= blocks
        && blocks <= MAX_BLOCKS
        && (1..=MAX_WIDTH).contains(&width)
        && eps.is_finite()
        && eps > 0.0
    {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA depth stream {what} takes first_block <= blocks <= {MAX_BLOCKS}, 1 to \
         {MAX_WIDTH} columns and a finite positive eps, got first_block {first_block}, blocks \
         {blocks}, width {width}, eps {eps}"
    )))
}

fn state_bytes(states: States, used: usize, width: usize) -> Result<usize, EngError> {
    states.bytes(used, width).ok_or_else(|| {
        EngError::BadRequest(format!(
            "the CUDA depth stream takes 1 to {MAX_STATES} states at least {used} rows apart, \
             got {} states {} rows apart",
            states.rows, states.stride
        ))
    })
}

fn check_row(name: &'static str, range: &DeviceRange<'_>, bytes: usize) -> Result<(), EngError> {
    require_fit(name, range, bytes, align_of::<f32>())?;
    if range.bytes() == bytes {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA {name} range holds {} bytes but the call needs exactly {bytes}",
        range.bytes()
    )))
}

/// `out` and `health` share no byte with each other or with a range in `reads`.
fn writes_apart<'a>(
    writes: [Claim<'a>; 2],
    reads: impl IntoIterator<Item = Claim<'a>>,
) -> Result<(), EngError> {
    let [out, health] = writes;
    require_apart(&out, &health)?;
    for read in reads {
        require_apart(&out, &read)?;
        require_apart(&health, &read)?;
    }
    Ok(())
}

impl CudaContext {
    /// `Stream::mix(first_block, query)` of the host: the softmax-weighted mean of the sealed
    /// rows `first_block..blocks` (newest first) and the normalized current row, each scored by
    /// `dot(row, query) / sqrt(width)`. Agrees with the host within floating-point error.
    pub(crate) fn depth_attention_mix(
        &self,
        stream: &DeviceStream,
        call: MixCall<'_>,
        shape: MixShape,
    ) -> Result<(), EngError> {
        self.depth_attention_mix_states(
            stream,
            MixStatesCall {
                state: call.state,
                query: Some(call.query),
                gamma: None,
                out: call.out,
                health: call.health,
            },
            MixStatesShape {
                first_block: shape.first_block,
                blocks: shape.blocks,
                width: shape.width,
                out_stride: shape.width,
                mix_eps: shape.eps,
                norm_eps: 1.0,
            },
            States::one(shape.blocks + 1),
        )
    }

    /// [`Self::depth_attention_mix`] of independent jobs in one launch, a block per job: each
    /// output has the bits of its own mix. The jobs share `health`; several may read one state.
    pub(crate) fn depth_attention_mix_jobs(
        &self,
        stream: &DeviceStream,
        jobs: &[MixJob<'_>],
        health: &DeviceRange<'_>,
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        if !(1..=MAX_MIX_JOBS).contains(&jobs.len()) {
            return Err(EngError::BadRequest(format!(
                "the CUDA depth stream mix takes 1 to {MAX_MIX_JOBS} jobs per launch, got {}",
                jobs.len()
            )));
        }
        let row = width * size_of::<f32>();
        let health_bytes = size_of::<f32>();
        check_row("depth stream health", health, health_bytes)?;
        let mut reads = Vec::with_capacity(2 * jobs.len());
        for job in jobs {
            check_shape("mix", job.first_block, job.blocks, width, eps)?;
            let state = (job.blocks + 1) * row;
            check_row("depth stream state", &job.state, state)?;
            check_row("depth stream query", &job.query, row)?;
            check_row("depth stream mix output", &job.out, row)?;
            let state = Claim::of("depth stream state", &job.state, state);
            let query = Claim::of("depth stream query", &job.query, row);
            require_apart(&state, &query)?;
            reads.extend([state, query]);
        }
        let mut writes = vec![Claim::of("depth stream health", health, health_bytes)];
        writes.extend(
            jobs.iter()
                .map(|job| Claim::of("depth stream mix output", &job.out, row)),
        );
        for (at, write) in writes.iter().enumerate() {
            for other in reads.iter().chain(&writes[at + 1..]) {
                require_apart(write, other)?;
            }
        }
        self.ensure_usable()?;
        let raw = jobs
            .iter()
            .map(|job| {
                Ok(native::MixJobRaw {
                    state: self.raw_range(&job.state)?,
                    query: self.raw_range(&job.query)?,
                    out: self.raw_range(&job.out)?,
                    first_block: job.first_block,
                    blocks: job.blocks,
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let health = self.raw_range(health)?;
        self.launch_on(stream, |s| native::mix_jobs(s, &raw, health, (width, eps)))
    }

    /// `Stream::seal(next)` of the host: the current row is RMS-normalized into the new sealed
    /// row, then replaced by `next`. The sealed row agrees with the host within rounding.
    pub(crate) fn depth_attention_seal(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
    ) -> Result<(), EngError> {
        self.depth_attention_seal_states(stream, call, shape, States::one(shape.blocks + 2))
    }

    /// [`Self::depth_attention_seal`] of each of `states`, the next rows packed one row apart.
    pub(crate) fn depth_attention_seal_states(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
        states: States,
    ) -> Result<(), EngError> {
        check_shape("seal", 0, shape.blocks, shape.width, shape.eps)?;
        let row = shape.width * size_of::<f32>();
        let state = state_bytes(states, shape.blocks + 2, shape.width)?;
        let next = states.rows * row;
        let health = size_of::<f32>();
        check_row("depth stream state", &call.state, state)?;
        check_row("depth stream next row", &call.next, next)?;
        check_row("depth stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("depth stream state", &call.state, state),
                Claim::of("depth stream health", &call.health, health),
            ],
            [Claim::of("depth stream next row", &call.next, next)],
        )?;
        self.ensure_usable()?;
        let (state, next, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.next)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            native::seal(
                raw,
                state,
                next,
                health,
                [shape.blocks, shape.width],
                shape.eps,
                states,
            )
        })
    }

    /// [`Self::depth_attention_mix`] of each of `states`, with the optional operands of
    /// [`MixStatesCall`]. Each output row has the bits of the same state mixed on its own, and a
    /// norm fused by `gamma` those of `enqueue_row_weighted_norm` on the mix.
    pub(crate) fn depth_attention_mix_states(
        &self,
        stream: &DeviceStream,
        call: MixStatesCall<'_>,
        shape: MixStatesShape,
        states: States,
    ) -> Result<(), EngError> {
        let eps = |set: bool, eps: f32| if set { eps } else { 1.0 };
        check_shape(
            "mix",
            shape.first_block,
            shape.blocks,
            shape.width,
            eps(call.query.is_some(), shape.mix_eps),
        )?;
        check_shape(
            "mix norm",
            0,
            0,
            shape.width,
            eps(call.gamma.is_some(), shape.norm_eps),
        )?;
        if shape.out_stride < shape.width {
            return Err(EngError::BadRequest(format!(
                "the CUDA depth stream mix takes an output stride of at least the width, got {} \
                 for {} columns",
                shape.out_stride, shape.width
            )));
        }
        let row = shape.width * size_of::<f32>();
        let state = state_bytes(states, shape.blocks + 1, shape.width)?;
        let out = ((states.rows - 1) * shape.out_stride + shape.width) * size_of::<f32>();
        let health = size_of::<f32>();
        check_row("depth stream state", &call.state, state)?;
        for (name, range) in [("query", &call.query), ("gamma", &call.gamma)] {
            if let Some(range) = range {
                check_row(name, range, row)?;
            }
        }
        check_row("depth stream mix output", &call.out, out)?;
        check_row("depth stream health", &call.health, health)?;
        let mut reads = vec![Claim::of("depth stream state", &call.state, state)];
        reads.extend(call.query.iter().map(|q| Claim::of("query", q, row)));
        reads.extend(call.gamma.iter().map(|g| Claim::of("gamma", g, row)));
        writes_apart(
            [
                Claim::of("depth stream mix output", &call.out, out),
                Claim::of("depth stream health", &call.health, health),
            ],
            reads,
        )?;
        self.ensure_usable()?;
        let raw = native::MixRaw {
            state: self.raw_range(&call.state)?,
            query: call.query.as_ref().map(|q| self.raw_range(q)).transpose()?,
            gamma: call.gamma.as_ref().map(|g| self.raw_range(g)).transpose()?,
            out: self.raw_range(&call.out)?,
            health: self.raw_range(&call.health)?,
        };
        self.launch_on(stream, |stream| {
            native::mix(
                stream,
                raw,
                [
                    shape.first_block,
                    shape.blocks,
                    shape.width,
                    shape.out_stride,
                ],
                [shape.mix_eps, shape.norm_eps],
                states,
            )
        })
    }

    /// `Stream::add_current(row)` of the host: `current = current + row`, bit-identical.
    pub(crate) fn depth_attention_add(
        &self,
        stream: &DeviceStream,
        call: AddCall<'_>,
        width: usize,
    ) -> Result<(), EngError> {
        self.depth_attention_add_states(stream, call, width, States::one(1))
    }

    /// [`Self::depth_attention_add`] into the current row of each of `states`, the addends packed
    /// one row apart.
    pub(crate) fn depth_attention_add_states(
        &self,
        stream: &DeviceStream,
        call: AddCall<'_>,
        width: usize,
        states: States,
    ) -> Result<(), EngError> {
        check_shape("add", 0, 0, width, 1.0)?;
        let row = width * size_of::<f32>();
        let state = state_bytes(states, 1, width)?;
        let rows = states.rows * row;
        let health = size_of::<f32>();
        check_row("depth stream state", &call.state, state)?;
        check_row("depth stream addend row", &call.row, rows)?;
        check_row("depth stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("depth stream state", &call.state, state),
                Claim::of("depth stream health", &call.health, health),
            ],
            [Claim::of("depth stream addend row", &call.row, rows)],
        )?;
        self.ensure_usable()?;
        let (state, row, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.row)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            native::add(raw, state, row, health, width, states)
        })
    }

    /// The gate halves of the cross-pass controllers: `dst = src * rms_scale(src)` with `eps`
    /// (the host's `rms_norm_with_scale`) for each row in one launch, a block per row. Agrees
    /// with the host within rounding; rows may share one health word.
    pub(crate) fn depth_attention_norm_rows(
        &self,
        stream: &DeviceStream,
        rows: &[NormCall<'_>],
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        if !(1..=MAX_NORM_ROWS).contains(&rows.len()) {
            return Err(EngError::BadRequest(format!(
                "the CUDA depth stream norm takes 1 to {MAX_NORM_ROWS} rows per launch, got {}",
                rows.len()
            )));
        }
        check_shape("norm", 0, 0, width, eps)?;
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        for call in rows {
            check_row("depth stream norm source", &call.src, row)?;
            check_row("depth stream norm output", &call.dst, row)?;
            check_row("depth stream health", &call.health, health)?;
        }
        for (at, call) in rows.iter().enumerate() {
            let dst = Claim::of("depth stream norm output", &call.dst, row);
            let words = Claim::of("depth stream health", &call.health, health);
            for (other_at, other) in rows.iter().enumerate() {
                let src = Claim::of("depth stream norm source", &other.src, row);
                let other_dst = Claim::of("depth stream norm output", &other.dst, row);
                require_apart(&dst, &src)?;
                require_apart(&words, &src)?;
                require_apart(&words, &other_dst)?;
                if at != other_at {
                    require_apart(&dst, &other_dst)?;
                }
            }
        }
        self.ensure_usable()?;
        let raw = rows
            .iter()
            .map(|call| {
                Ok(native::NormRowRaw {
                    src: self.raw_range(&call.src)?,
                    dst: self.raw_range(&call.dst)?,
                    health: self.raw_range(&call.health)?,
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        self.launch_on(stream, |s| native::norm_rows(s, &raw, width, eps))
    }
}
