//! The resident Inkling depth stream, launch-only: the weighted mix over the sealed blocks,
//! the seal of the current row and the add into it.
//!
//! A stream state packs the current row and then `blocks` sealed rows, `width` floats each
//! (`native/resident/inkling_stream.h`). `health` is one float per launch: left alone, or quiet
//! NaN when an input or an output of that launch was not finite.

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

use super::super::native::resident::inkling_stream::{self as native};
pub(crate) use super::super::native::resident::inkling_stream::{
    MAX_BLOCKS, MAX_MIX_JOBS, MAX_NORM_ROWS, MAX_ORDERED_WIDTH, MAX_STATES, States,
};

const MAX_WIDTH: usize = 8192;

/// The ranges of a mix launch. `state` holds the current row and then `shape.blocks` sealed rows.
pub(crate) struct MixCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// One mix of [`CudaContext::inkling_stream_mix_jobs`]: a state of `blocks + 1` rows whose
/// sealed rows `first_block..blocks` join the mix, its query and its output row.
#[derive(Clone, Copy)]
pub(crate) struct MixJob<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) first_block: usize,
    pub(crate) blocks: usize,
    /// The decode glue (`FIFI_INKLING_CUDA_DECODE_GLUE`): a row added into the current row of
    /// `state` before the mix, the host's residual add ([`CudaContext::inkling_stream_add`]).
    pub(crate) addend: Option<DeviceRange<'a>>,
    /// The decode glue: the weighted norm of the job's output, written by the same block.
    pub(crate) norm: Option<MixNorm<'a>>,
}

/// The norm epilogue of a [`MixJob`] under the decode glue: `normed = (out * rms_scale(out)) *
/// gamma` with `eps`, the host's `rms_norm_with_scale` (the row
/// [`CudaContext::enqueue_row_weighted_norm`] writes).
#[derive(Clone, Copy)]
pub(crate) struct MixNorm<'a> {
    pub(crate) gamma: DeviceRange<'a>,
    pub(crate) normed: DeviceRange<'a>,
    pub(crate) eps: f32,
}

/// One job's claims: what it reads, what it writes, and (for a job with an addend) the current
/// row of its state that the addend writes into.
struct MixClaims<'a> {
    reads: Vec<Claim<'a>>,
    writes: Vec<Claim<'a>>,
    current: Option<Claim<'a>>,
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

/// The ranges of a host-ordered mix. Without `query` the output is the raw current row; with
/// `gamma` it is `norm(mix) * gamma` (`CpuForward::norm`).
pub(crate) struct OrderedMixCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: Option<DeviceRange<'a>>,
    pub(crate) gamma: Option<DeviceRange<'a>>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The shape of a host-ordered mix; output rows are `out_stride` floats apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OrderedMixShape {
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
        "the CUDA Inkling stream {what} takes first_block <= blocks <= {MAX_BLOCKS}, 1 to \
         {MAX_WIDTH} columns and a finite positive eps, got first_block {first_block}, blocks \
         {blocks}, width {width}, eps {eps}"
    )))
}

fn state_bytes(states: States, used: usize, width: usize) -> Result<usize, EngError> {
    states.bytes(used, width).ok_or_else(|| {
        EngError::BadRequest(format!(
            "the CUDA Inkling stream takes 1 to {MAX_STATES} states at least {used} rows apart, \
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

/// The claims of one mix job: the state, the query and the output of every job, then the gamma
/// and the normed row of a norm epilogue and the addend row, whose sum the job writes into the
/// first row of the state it reads. Every extra range holds exactly one row.
fn mix_claims<'a>(job: &MixJob<'a>, (width, eps): (usize, f32)) -> Result<MixClaims<'a>, EngError> {
    check_shape("mix", job.first_block, job.blocks, width, eps)?;
    let row = width * size_of::<f32>();
    let state = (job.blocks + 1) * row;
    check_row("Inkling stream state", &job.state, state)?;
    check_row("Inkling stream query", &job.query, row)?;
    check_row("Inkling stream mix output", &job.out, row)?;
    let mut claims = MixClaims {
        reads: vec![
            Claim::of("Inkling stream state", &job.state, state),
            Claim::of("Inkling stream query", &job.query, row),
        ],
        writes: vec![Claim::of("Inkling stream mix output", &job.out, row)],
        current: None,
    };
    if let Some(norm) = &job.norm {
        if !norm.eps.is_finite() || norm.eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the CUDA Inkling stream mix norm takes a finite positive eps".into(),
            ));
        }
        check_row("Inkling stream norm gamma", &norm.gamma, row)?;
        check_row("Inkling stream normed row", &norm.normed, row)?;
        claims
            .reads
            .push(Claim::of("Inkling stream norm gamma", &norm.gamma, row));
        claims
            .writes
            .push(Claim::of("Inkling stream normed row", &norm.normed, row));
    }
    if let Some(addend) = &job.addend {
        check_row("Inkling stream addend row", addend, row)?;
        claims
            .reads
            .push(Claim::of("Inkling stream addend row", addend, row));
        claims.current = Some(Claim::of(
            "Inkling stream current row",
            &job.state.slice(0, row)?,
            row,
        ));
    }
    Ok(claims)
}

/// Every write of a job (its output, its normed row, the health word and, under the glue, the
/// current row an addend writes) shares no byte with any row any job reads or writes, except
/// the state that holds the current row it writes.
fn require_claims_apart(
    claims: &[MixClaims<'_>],
    health: &DeviceRange<'_>,
) -> Result<(), EngError> {
    let health = Claim::of("Inkling stream health", health, size_of::<f32>());
    for (j, claim) in claims.iter().enumerate() {
        for write in &claim.writes {
            require_apart(write, &health)?;
            for read in &claim.reads {
                require_apart(write, read)?;
            }
        }
        // Within one job the state shares no byte with the query; with the glue's extras it
        // shares none with the addend, the gamma, the output or the normed row either (the
        // native entry refuses both, and a native refusal is terminal).
        let state = &claim.reads[0];
        require_apart(state, &claim.reads[1])?;
        if let Some(current) = &claim.current {
            require_apart(current, &health)?;
            // The current row an addend writes is part of the state; the rest is apart.
            for read in claim.reads.iter().skip(2) {
                require_apart(state, read)?;
                require_apart(current, read)?;
            }
            for write in &claim.writes {
                require_apart(state, write)?;
            }
        } else if claim.reads.len() > 2 {
            for read in claim.reads.iter().skip(2) {
                require_apart(state, read)?;
            }
            for write in &claim.writes {
                require_apart(state, write)?;
            }
        }
        let others = claims
            .iter()
            .take(j)
            .chain(&claims[j + 1..])
            .flat_map(|other| {
                other
                    .reads
                    .iter()
                    .chain(&other.writes)
                    .chain(&other.current)
            })
            .collect::<Vec<_>>();
        for write in claim.writes.iter().chain(claim.current.iter()) {
            for row in &others {
                require_apart(write, row)?;
            }
        }
    }
    Ok(())
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
    pub(crate) fn inkling_stream_mix(
        &self,
        stream: &DeviceStream,
        call: MixCall<'_>,
        shape: MixShape,
    ) -> Result<(), EngError> {
        self.inkling_stream_mix_states(stream, call, shape, States::one(shape.blocks + 1))
    }

    /// [`Self::inkling_stream_mix`] of independent jobs in one launch, a block per job: each
    /// output has the bits of its own mix. The jobs share `health`; several may read one state.
    pub(crate) fn inkling_stream_mix_jobs(
        &self,
        stream: &DeviceStream,
        jobs: &[MixJob<'_>],
        health: &DeviceRange<'_>,
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        if !(1..=MAX_MIX_JOBS).contains(&jobs.len()) {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling stream mix takes 1 to {MAX_MIX_JOBS} jobs per launch, got {}",
                jobs.len()
            )));
        }
        check_row("Inkling stream health", health, size_of::<f32>())?;
        let claims = jobs
            .iter()
            .map(|job| mix_claims(job, (width, eps)))
            .collect::<Result<Vec<_>, EngError>>()?;
        require_claims_apart(&claims, health)?;
        self.ensure_usable()?;
        let raw = jobs
            .iter()
            .map(|job| self.raw_mix_job(job))
            .collect::<Result<Vec<_>, EngError>>()?;
        let health = self.raw_range(health)?;
        self.launch_on(stream, move |s| {
            native::mix_jobs(s, &raw, health, (width, eps))
        })
    }

    /// The native job: its three rows and its extras, every range checked by `mix_claims`.
    fn raw_mix_job<'b>(&self, job: &MixJob<'b>) -> Result<native::MixJobRaw<'b>, EngError> {
        let extras = match (&job.addend, &job.norm) {
            (None, None) => None,
            (addend, norm) => Some(native::MixExtrasRaw {
                addend: addend
                    .as_ref()
                    .map(|range| self.raw_range(range))
                    .transpose()?,
                norm: norm
                    .as_ref()
                    .map(|norm| {
                        Ok::<_, EngError>(native::MixNormRaw {
                            gamma: self.raw_range(&norm.gamma)?,
                            normed: self.raw_range(&norm.normed)?,
                            eps: norm.eps,
                        })
                    })
                    .transpose()?,
            }),
        };
        Ok(native::MixJobRaw {
            state: self.raw_range(&job.state)?,
            query: self.raw_range(&job.query)?,
            out: self.raw_range(&job.out)?,
            first_block: job.first_block,
            blocks: job.blocks,
            extras,
        })
    }

    /// [`Self::inkling_stream_mix`] of each of `states`, the outputs packed one row apart.
    pub(crate) fn inkling_stream_mix_states(
        &self,
        stream: &DeviceStream,
        call: MixCall<'_>,
        shape: MixShape,
        states: States,
    ) -> Result<(), EngError> {
        check_shape(
            "mix",
            shape.first_block,
            shape.blocks,
            shape.width,
            shape.eps,
        )?;
        let row = shape.width * size_of::<f32>();
        let state = state_bytes(states, shape.blocks + 1, shape.width)?;
        let out = states.rows * row;
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream query", &call.query, row)?;
        check_row("Inkling stream mix output", &call.out, out)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream mix output", &call.out, out),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [
                Claim::of("Inkling stream state", &call.state, state),
                Claim::of("Inkling stream query", &call.query, row),
            ],
        )?;
        self.ensure_usable()?;
        let (state, query, out, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.query)?,
            self.raw_range(&call.out)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            native::mix(
                raw,
                state,
                query,
                out,
                health,
                [shape.first_block, shape.blocks, shape.width],
                (shape.eps, states),
            )
        })
    }

    /// `Stream::seal(next)` of the host: the current row is RMS-normalized into the new sealed
    /// row, then replaced by `next`. The sealed row agrees with the host within rounding.
    pub(crate) fn inkling_stream_seal(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
    ) -> Result<(), EngError> {
        self.inkling_stream_seal_states(stream, call, shape, States::one(shape.blocks + 2))
    }

    /// [`Self::inkling_stream_seal`] of each of `states`, the next rows packed one row apart.
    pub(crate) fn inkling_stream_seal_states(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
        states: States,
    ) -> Result<(), EngError> {
        self.seal_states_with(false, stream, call, shape, states)
    }

    /// [`Self::inkling_stream_seal_states`] with the scale summed in the host's element order:
    /// the sealed rows are the host's bits.
    pub(crate) fn inkling_stream_seal_ordered(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
        states: States,
    ) -> Result<(), EngError> {
        if shape.width > MAX_ORDERED_WIDTH {
            return Err(EngError::BadRequest(format!(
                "the ordered CUDA Inkling stream seal takes at most {MAX_ORDERED_WIDTH} columns"
            )));
        }
        self.seal_states_with(true, stream, call, shape, states)
    }

    /// `Stream::mix` of the host for each of `states`, summed in the host's element order on one
    /// lane, so the output is the host's bits; see [`OrderedMixCall`] for the optional operands.
    pub(crate) fn inkling_stream_mix_ordered(
        &self,
        stream: &DeviceStream,
        call: OrderedMixCall<'_>,
        shape: OrderedMixShape,
        states: States,
    ) -> Result<(), EngError> {
        let eps = |set: bool, eps: f32| if set { eps } else { 1.0 };
        check_shape(
            "ordered mix",
            shape.first_block,
            shape.blocks,
            shape.width,
            eps(call.query.is_some(), shape.mix_eps),
        )?;
        check_shape(
            "ordered mix norm",
            0,
            0,
            shape.width,
            eps(call.gamma.is_some(), shape.norm_eps),
        )?;
        if shape.width > MAX_ORDERED_WIDTH || shape.out_stride < shape.width {
            return Err(EngError::BadRequest(format!(
                "the ordered CUDA Inkling stream mix takes at most {MAX_ORDERED_WIDTH} columns and \
                 an output stride of at least the width, got {} and {}",
                shape.width, shape.out_stride
            )));
        }
        let row = shape.width * size_of::<f32>();
        let state = state_bytes(states, shape.blocks + 1, shape.width)?;
        let out = ((states.rows - 1) * shape.out_stride + shape.width) * size_of::<f32>();
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, state)?;
        for (name, range) in [("query", &call.query), ("gamma", &call.gamma)] {
            if let Some(range) = range {
                check_row(name, range, row)?;
            }
        }
        check_row("Inkling ordered mix output", &call.out, out)?;
        check_row("Inkling stream health", &call.health, health)?;
        let mut reads = vec![Claim::of("Inkling stream state", &call.state, state)];
        reads.extend(call.query.iter().map(|q| Claim::of("query", q, row)));
        reads.extend(call.gamma.iter().map(|g| Claim::of("gamma", g, row)));
        writes_apart(
            [
                Claim::of("Inkling ordered mix output", &call.out, out),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            reads,
        )?;
        self.ensure_usable()?;
        let raw = native::OrderedMixRaw {
            state: self.raw_range(&call.state)?,
            query: call.query.as_ref().map(|q| self.raw_range(q)).transpose()?,
            gamma: call.gamma.as_ref().map(|g| self.raw_range(g)).transpose()?,
            out: self.raw_range(&call.out)?,
            health: self.raw_range(&call.health)?,
        };
        self.launch_on(stream, |stream| {
            native::mix_ordered(
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

    fn seal_states_with(
        &self,
        ordered: bool,
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
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream next row", &call.next, next)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream state", &call.state, state),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream next row", &call.next, next)],
        )?;
        self.ensure_usable()?;
        let (state, next, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.next)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            if ordered {
                native::seal_ordered(
                    raw,
                    [state, next, health],
                    [shape.blocks, shape.width],
                    shape.eps,
                    states,
                )
            } else {
                native::seal(
                    raw,
                    state,
                    next,
                    health,
                    [shape.blocks, shape.width],
                    shape.eps,
                    states,
                )
            }
        })
    }

    /// `Stream::add_current(row)` of the host: `current = current + row`, bit-identical.
    pub(crate) fn inkling_stream_add(
        &self,
        stream: &DeviceStream,
        call: AddCall<'_>,
        width: usize,
    ) -> Result<(), EngError> {
        self.inkling_stream_add_states(stream, call, width, States::one(1))
    }

    /// [`Self::inkling_stream_add`] into the current row of each of `states`, the addends packed
    /// one row apart.
    pub(crate) fn inkling_stream_add_states(
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
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream addend row", &call.row, rows)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream state", &call.state, state),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream addend row", &call.row, rows)],
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
    pub(crate) fn inkling_stream_norm_rows(
        &self,
        stream: &DeviceStream,
        rows: &[NormCall<'_>],
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        if !(1..=MAX_NORM_ROWS).contains(&rows.len()) {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling stream norm takes 1 to {MAX_NORM_ROWS} rows per launch, got {}",
                rows.len()
            )));
        }
        check_shape("norm", 0, 0, width, eps)?;
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        for call in rows {
            check_row("Inkling stream norm source", &call.src, row)?;
            check_row("Inkling stream norm output", &call.dst, row)?;
            check_row("Inkling stream health", &call.health, health)?;
        }
        for (at, call) in rows.iter().enumerate() {
            let dst = Claim::of("Inkling stream norm output", &call.dst, row);
            let words = Claim::of("Inkling stream health", &call.health, health);
            for (other_at, other) in rows.iter().enumerate() {
                let src = Claim::of("Inkling stream norm source", &other.src, row);
                let other_dst = Claim::of("Inkling stream norm output", &other.dst, row);
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
