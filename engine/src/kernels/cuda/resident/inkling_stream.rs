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
    native::resident::RawRange,
};

pub(crate) use super::super::native::resident::inkling_stream::MAX_BLOCKS;
use super::super::native::resident::inkling_stream::{
    self as native, MAX_MIX_JOBS, MAX_NORM_ROWS, MAX_WORD_ROWS, MixJobRaw,
};

const MAX_WIDTH: usize = 8192;

/// The ranges of a mix launch. `state` holds the current row and then `shape.blocks` sealed rows.
pub(crate) struct MixCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The norm epilogue of a mix job: `normed = out * rms_scale(out, eps) * gamma`, the host's
/// weighted RMSNorm of the mix output, written by the mix itself.
#[derive(Clone, Copy)]
pub(crate) struct MixNorm<'a> {
    pub(crate) gamma: DeviceRange<'a>,
    pub(crate) normed: DeviceRange<'a>,
    pub(crate) eps: f32,
}

/// One mix of a `inkling_stream_mix_jobs` launch: the ranges of [`MixCall`] with the first and
/// last sealed rows of [`MixShape`]; the width and eps belong to the launch. `addend` is added
/// into the current row of `state` before the mix (the host's `add_current`); `norm` makes the
/// job write the weighted norm of its output too.
pub(crate) struct MixJob<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
    pub(crate) addend: Option<DeviceRange<'a>>,
    pub(crate) norm: Option<MixNorm<'a>>,
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
    /// Zero every range in one launch, one block per range. Each range holds whole words.
    pub(crate) fn inkling_zero_words(
        &self,
        stream: &DeviceStream,
        ranges: &[DeviceRange<'_>],
    ) -> Result<(), EngError> {
        if ranges.is_empty() || ranges.len() > MAX_WORD_ROWS {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling word rows take 1 to {MAX_WORD_ROWS} ranges, got {}",
                ranges.len()
            )));
        }
        for range in ranges {
            require_fit("Inkling word row", range, 1, align_of::<f32>())?;
            if !range.bytes().is_multiple_of(size_of::<f32>()) {
                return Err(EngError::BadRequest(
                    "the CUDA Inkling word row holds a whole number of words".into(),
                ));
            }
        }
        self.ensure_usable()?;
        let raw = ranges
            .iter()
            .map(|range| self.raw_range(range))
            .collect::<Result<Vec<_>, EngError>>()?;
        self.launch_on(stream, |s| native::zero_words(s, &raw))
    }

    /// Copy every `(source, destination)` pair in one launch, one block per pair; each pair
    /// holds the same whole number of words and shares no byte.
    pub(crate) fn inkling_copy_words(
        &self,
        stream: &DeviceStream,
        pairs: &[(DeviceRange<'_>, DeviceRange<'_>)],
    ) -> Result<(), EngError> {
        if pairs.is_empty() || pairs.len() > MAX_WORD_ROWS {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling word rows take 1 to {MAX_WORD_ROWS} pairs, got {}",
                pairs.len()
            )));
        }
        for (source, destination) in pairs {
            require_fit("Inkling word row source", source, 1, align_of::<f32>())?;
            require_fit(
                "Inkling word row destination",
                destination,
                1,
                align_of::<f32>(),
            )?;
            if source.bytes() != destination.bytes()
                || !source.bytes().is_multiple_of(size_of::<f32>())
            {
                return Err(EngError::BadRequest(
                    "the CUDA Inkling word row copies hold the same whole number of words".into(),
                ));
            }
            require_apart(
                &Claim::of("Inkling word row source", source, source.bytes()),
                &Claim::of(
                    "Inkling word row destination",
                    destination,
                    destination.bytes(),
                ),
            )?;
        }
        self.ensure_usable()?;
        let raw = pairs
            .iter()
            .map(|(source, destination)| {
                Ok((self.raw_range(source)?, self.raw_range(destination)?))
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        self.launch_on(stream, |s| native::copy_words(s, &raw))
    }
}

/// True when both ranges start at the same byte of the same allocation.
fn same_start(a: &DeviceRange<'_>, b: &DeviceRange<'_>) -> bool {
    std::ptr::eq(a.buffer(), b.buffer()) && a.offset() == b.offset()
}

/// The claim of a range for its whole extent.
fn claim<'a>(name: &'static str, range: &'a DeviceRange<'a>) -> Claim<'a> {
    Claim::of(name, range, range.bytes())
}

mod baseline;

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
        check_shape(
            "mix",
            shape.first_block,
            shape.blocks,
            shape.width,
            shape.eps,
        )?;
        let row = shape.width * size_of::<f32>();
        let state = (shape.blocks + 1) * row;
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream query", &call.query, row)?;
        check_row("Inkling stream mix output", &call.out, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream mix output", &call.out, row),
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
            let mix = if self.dense_group_policy().decode_glue {
                native::mix
            } else {
                super::super::native::resident::inkling_stream_baseline::mix
            };
            mix(
                raw,
                [state, query, out, health],
                [shape.first_block, shape.blocks, shape.width],
                shape.eps,
            )
        })
    }

    /// Independent [`Self::inkling_stream_mix`] launches in one: one block per job, every job
    /// over the same `width` and `eps`. Each job passes the checks of the single mix, and its
    /// optional addend row and norm epilogue (gamma read, normed row written) hold the row and
    /// share no byte with the job. No range a job writes (its output, health word, normed row
    /// and, with an addend, the current row of its state) shares a byte with any range of
    /// another job, unless jobs name the same health word (each would store the same NaN).
    pub(crate) fn inkling_stream_mix_jobs(
        &self,
        stream: &DeviceStream,
        jobs: &[MixJob<'_>],
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        if jobs.is_empty() || jobs.len() > MAX_MIX_JOBS {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling stream mixes 1 to {MAX_MIX_JOBS} jobs at once, got {}",
                jobs.len()
            )));
        }
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        let mut checked_jobs = Vec::with_capacity(jobs.len());
        for job in jobs {
            check_shape("mix", job.first_block, job.blocks, width, eps)?;
            let state = (job.blocks + 1) * row;
            check_row("Inkling stream state", &job.state, state)?;
            check_row("Inkling stream query", &job.query, row)?;
            check_row("Inkling stream mix output", &job.out, row)?;
            check_row("Inkling stream health", &job.health, health)?;
            if let Some(addend) = &job.addend {
                check_row("Inkling stream mix addend", addend, row)?;
            }
            if let Some(norm) = &job.norm {
                if !norm.eps.is_finite() || norm.eps <= 0.0 {
                    return Err(EngError::BadRequest(
                        "the CUDA Inkling stream mix norm needs a finite positive eps".into(),
                    ));
                }
                check_row("Inkling stream mix gamma", &norm.gamma, row)?;
                check_row("Inkling stream mix normed row", &norm.normed, row)?;
            }
            // Named ranges; `claim` names a range in the error of `require_apart`.
            let mut reads: Vec<(&'static str, DeviceRange<'_>)> = vec![
                ("Inkling stream state", job.state),
                ("Inkling stream query", job.query),
            ];
            if let Some(addend) = &job.addend {
                reads.push(("Inkling stream mix addend", *addend));
            }
            if let Some(norm) = &job.norm {
                reads.push(("Inkling stream mix gamma", norm.gamma));
            }
            // Named ranges and whether a written range is the current row of the state.
            let mut writes: Vec<(&'static str, DeviceRange<'_>, bool)> = vec![
                ("Inkling stream mix output", job.out, false),
                ("Inkling stream health", job.health, false),
            ];
            if let Some(norm) = &job.norm {
                writes.push(("Inkling stream mix normed row", norm.normed, false));
            }
            // With an addend the job writes the current row, the first row of its state.
            if job.addend.is_some() {
                writes.push(("Inkling stream current row", job.state.slice(0, row)?, true));
            }
            let state_range = job.state;
            for (i, write) in writes.iter().enumerate() {
                for (_, other, _) in &writes[i + 1..] {
                    require_apart(&claim(write.0, &write.1), &claim(write.0, other))?;
                }
                for (name, read) in &reads {
                    // The current row is the first row of the state the job reads.
                    if write.2 && same_start(read, &state_range) {
                        continue;
                    }
                    require_apart(&claim(write.0, &write.1), &claim(name, read))?;
                }
            }
            checked_jobs.push((reads, writes, job.health));
        }
        for (i, (_, writes, health)) in checked_jobs.iter().enumerate() {
            for (reads, other_writes, other_health) in &checked_jobs[i + 1..] {
                let shared_health = same_start(health, other_health);
                let both_health = |a: &DeviceRange<'_>, b: &DeviceRange<'_>| {
                    shared_health && same_start(a, health) && same_start(b, other_health)
                };
                let other_claims = reads
                    .iter()
                    .map(|(name, range)| (*name, *range))
                    .chain(other_writes.iter().map(|(name, range, _)| (*name, *range)))
                    .collect::<Vec<_>>();
                let my_claims = checked_jobs[i]
                    .0
                    .iter()
                    .map(|(name, range)| (*name, *range))
                    .chain(writes.iter().map(|(name, range, _)| (*name, *range)))
                    .collect::<Vec<_>>();
                for (name, write) in writes.iter().map(|(n, r, _)| (*n, *r)) {
                    for (other_name, other) in &other_claims {
                        if !both_health(&write, other) {
                            require_apart(&claim(name, &write), &claim(other_name, other))?;
                        }
                    }
                }
                for (name, write, _) in other_writes {
                    for (other_name, other) in &my_claims {
                        if !both_health(write, other) {
                            require_apart(&claim(name, write), &claim(other_name, other))?;
                        }
                    }
                }
            }
        }
        self.ensure_usable()?;
        let raw = jobs
            .iter()
            .map(|job| {
                let [state, query, out, health] = [&job.state, &job.query, &job.out, &job.health]
                    .each_ref()
                    .map(|r| self.raw_range(r));
                let ranges = [state?, query?, out?, health?];
                let addend = job.addend.as_ref().map(|r| self.raw_range(r)).transpose()?;
                let norm = match &job.norm {
                    Some(n) => Some((
                        [self.raw_range(&n.gamma)?, self.raw_range(&n.normed)?],
                        n.eps,
                    )),
                    None => None,
                };
                Ok(MixJobRaw {
                    ranges,
                    addend,
                    norm,
                    shape: [job.first_block, job.blocks],
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        self.launch_on(stream, |s| native::mix_jobs(s, &raw, (width, eps)))
    }

    /// `Stream::seal(next)` of the host: the current row is RMS-normalized into the new sealed
    /// row, then replaced by `next`. The sealed row agrees with the host within rounding.
    pub(crate) fn inkling_stream_seal(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
    ) -> Result<(), EngError> {
        check_shape("seal", 0, shape.blocks, shape.width, shape.eps)?;
        let row = shape.width * size_of::<f32>();
        let state = (shape.blocks + 2) * row;
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream next row", &call.next, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream state", &call.state, state),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream next row", &call.next, row)],
        )?;
        self.ensure_usable()?;
        let (state, next, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.next)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            let seal = if self.dense_group_policy().decode_glue {
                native::seal
            } else {
                super::super::native::resident::inkling_stream_baseline::seal
            };
            seal(
                raw,
                state,
                next,
                health,
                [shape.blocks, shape.width],
                shape.eps,
            )
        })
    }

    /// `Stream::add_current(row)` of the host: `current = current + row`, bit-identical. The
    /// resident chain folds its adds into the convolution and the mix jobs; this entry stays
    /// as the reference the tests measure those fusions against.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the reference for the fused adds")
    )]
    pub(crate) fn inkling_stream_add(
        &self,
        stream: &DeviceStream,
        call: AddCall<'_>,
        width: usize,
    ) -> Result<(), EngError> {
        check_shape("add", 0, 0, width, 1.0)?;
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, row)?;
        check_row("Inkling stream addend row", &call.row, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream state", &call.state, row),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream addend row", &call.row, row)],
        )?;
        self.ensure_usable()?;
        let (state, row, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.row)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            let add = if self.dense_group_policy().decode_glue {
                native::add
            } else {
                super::super::native::resident::inkling_stream_baseline::add
            };
            add(raw, state, row, health, width)
        })
    }

    /// `dst = src * rms_scale(src)` with `eps`, the host's `rms_norm_with_scale`, for every
    /// row in one launch, one block per row, all over the same `width` and `eps` (the gate
    /// halves of the cross-pass controllers). A row's output and health word share no byte
    /// with its source; no row's output or health word shares a byte with another row's ranges,
    /// unless rows name the same health word (each stores the same quiet NaN). Agrees with the
    /// host within floating-point error.
    pub(crate) fn inkling_stream_norm_rows(
        &self,
        stream: &DeviceStream,
        rows: &[NormCall<'_>],
        (width, eps): (usize, f32),
    ) -> Result<(), EngError> {
        if rows.is_empty() || rows.len() > MAX_NORM_ROWS {
            return Err(EngError::BadRequest(format!(
                "the CUDA Inkling stream norms 1 to {MAX_NORM_ROWS} rows at once, got {}",
                rows.len()
            )));
        }
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        let mut claims = Vec::with_capacity(rows.len());
        for call in rows {
            check_shape("norm", 0, 0, width, eps)?;
            check_row("Inkling stream norm source", &call.src, row)?;
            check_row("Inkling stream norm output", &call.dst, row)?;
            check_row("Inkling stream health", &call.health, health)?;
            let src = Claim::of("Inkling stream norm source", &call.src, row);
            let (out, health) = (
                Claim::of("Inkling stream norm output", &call.dst, row),
                Claim::of("Inkling stream health", &call.health, health),
            );
            writes_apart([out, health], [src])?;
            claims.push([src, out, health]);
        }
        for (r, [src, out, health]) in claims.iter().enumerate() {
            for (other, [other_src, other_out, other_health]) in claims[r + 1..].iter().enumerate()
            {
                require_apart(out, other_src)?;
                require_apart(other_out, src)?;
                require_apart(out, other_out)?;
                if same_start(&rows[r].health, &rows[r + 1 + other].health) {
                    continue;
                }
                require_apart(out, other_health)?;
                require_apart(other_out, health)?;
                require_apart(health, other_health)?;
                require_apart(health, other_src)?;
                require_apart(other_health, src)?;
            }
        }
        self.ensure_usable()?;
        let raw: Vec<[RawRange<'_>; 3]> = rows
            .iter()
            .map(|call| {
                Ok([
                    self.raw_range(&call.src)?,
                    self.raw_range(&call.dst)?,
                    self.raw_range(&call.health)?,
                ])
            })
            .collect::<Result<_, EngError>>()?;
        self.launch_on(stream, |s| native::norm_rows(s, &raw, (width, eps)))
    }
}
