//! The EXL3 arms of the chunk stages: the dense projections of a prefill chunk through the EXL3
//! job-table projection.
//!
//! A dense projection of `m` rows is ONE job-table call (three launches): the rows are cut into
//! jobs of at most `tile_rows` rows of the one matrix, and the tensor-core kernel decodes the
//! weights once per job, whatever the number of rows. The table is the same for every
//! projection of a chunk, so the chunk driver uploads it once ([`write_dense_jobs`]).
//!
//! The DeltaNet and attention input rows hold several projections side by side; the job-table
//! output lands in its columns through the output row stride and column offset, so no scatter is
//! needed. Only the two plain gates, which are not EXL3 matrices, go through the router's dot
//! kernel and `scatter_cols`.
//!
//! Nothing here allocates. The arithmetic flag of the projections is sticky: the chunk driver
//! clears it before the first layer and reads it once after the last.
//!
//! The chunk glue stays with its family: the tile height and the flag's device range are
//! parameters, so no family name travels through this module.

use crate::{
    EngError,
    family::{
        exl3::launch::Launch,
        project::cuda::{Exl3JobsIo, exl3::Exl3Weight},
    },
    kernels::cuda::{
        DeviceRange,
        prefill::scatter_cols::{ScatterColsCall, ScatterColsShape},
        resident::{
            exl3::Exl3Job,
            router::{RouterLogitsCall, RouterShape},
        },
    },
};

pub(in crate::family) const F32: usize = size_of::<f32>();

/// The scratch of the projections of one chunk, caller-owned and disjoint from every operand.
#[derive(Clone, Copy)]
pub(in crate::family) struct Exl3Scratch<'b> {
    /// F16 `[m][k]` for the widest `k` of a dense projection: the rotated input rows.
    pub(in crate::family) xh: DeviceRange<'b>,
    /// F32 `[m][n]` for the widest `n`: the partials.
    pub(in crate::family) z: DeviceRange<'b>,
    /// F32 `[m][n]` of a plain gate, which waits here for `scatter_cols`.
    pub(in crate::family) y: DeviceRange<'b>,
    /// The sticky arithmetic flag, four bytes.
    pub(in crate::family) flag: DeviceRange<'b>,
    /// The job table of the dense projections, written by [`write_dense_jobs`].
    pub(in crate::family) jobs: DeviceRange<'b>,
}

#[derive(Clone, Copy)]
pub(in crate::family) enum Output<'b> {
    /// The destination is exactly `[rows][n]`.
    Rows(DeviceRange<'b>),
    /// Columns `col_off..col_off + n` of a destination whose rows are `ld` floats apart.
    Columns {
        dst: DeviceRange<'b>,
        ld: usize,
        col_off: usize,
    },
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("prefill chunk: EXL3: {detail}"))
}

pub(in crate::family) fn window<'b>(
    range: DeviceRange<'b>,
    offset: usize,
    bytes: usize,
) -> Result<DeviceRange<'b>, EngError> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > range.bytes())
    {
        return Err(refuse(format!(
            "{bytes} bytes at {offset} pass a range of {}",
            range.bytes()
        )));
    }
    range.buffer().range(range.offset() + offset, bytes)
}

/// The jobs of a dense projection of `rows` rows: member 0, tiles of at most `tile_rows` rows.
pub(in crate::family) fn dense_jobs(rows: usize, tile_rows: usize) -> Vec<Exl3Job> {
    debug_assert!(tile_rows > 0);
    (0..rows)
        .step_by(tile_rows)
        .map(|first| Exl3Job {
            member: 0,
            row0: first as u32,
            rows: tile_rows.min(rows - first) as u32,
        })
        .collect()
}

/// Writes the job table of a chunk of `rows` rows. The write waits for the stream, so it runs once
/// per chunk, before the first layer, and not once per projection.
pub(in crate::family) fn write_dense_jobs(
    launch: &Launch<'_>,
    scratch: &Exl3Scratch<'_>,
    rows: usize,
    tile_rows: usize,
) -> Result<(), EngError> {
    let bytes = Exl3Job::table(&dense_jobs(rows, tile_rows));
    let table = window(scratch.jobs, 0, bytes.len())?;
    table
        .buffer()
        .write_at(launch.stream, table.offset(), &bytes)
}

/// The launch, scratch and rows of one chunk for which the dense job table was written.
pub(in crate::family) struct Projector<'a> {
    pub(in crate::family) launch: &'a Launch<'a>,
    pub(in crate::family) scratch: Exl3Scratch<'a>,
    pub(in crate::family) rows: usize,
    /// The most rows of one job: the tallest CTA of the job-table kernel, the family's own
    /// constant.
    pub(in crate::family) tile_rows: usize,
}

impl Projector<'_> {
    pub(in crate::family) fn project(
        &self,
        weight: &Exl3Weight,
        x: DeviceRange<'_>,
        out: Output<'_>,
    ) -> Result<(), EngError> {
        split::count("dense job-table projections");
        let jobs = dense_jobs(self.rows, self.tile_rows);
        let (out, out_ld, out_col) = match out {
            Output::Rows(dst) => (dst, weight.shape[1], 0),
            Output::Columns { dst, ld, col_off } => (dst, ld, col_off),
        };
        let scratch = &self.scratch;
        let io = Exl3JobsIo {
            x,
            xh: scratch.xh,
            z: scratch.z,
            out,
            flag: scratch.flag,
            jobs: window(scratch.jobs, 0, jobs.len() * 16)?,
            jobs_count: jobs.len(),
            max_rows: jobs.iter().map(|job| job.rows as usize).max().unwrap_or(1),
            total_rows: self.rows,
            out_ld,
            out_col,
        };
        let members = weight.members()?;
        self.launch
            .ctx
            .exl3_jobs(self.launch.stream, members.jobs_call(&io))
    }

    fn scatter(
        &self,
        src: DeviceRange<'_>,
        dst: DeviceRange<'_>,
        (rows, n, ld, col_off): (usize, usize, usize, usize),
    ) -> Result<(), EngError> {
        split::count("scatter_cols calls");
        self.launch.ctx.scatter_cols(
            self.launch.stream,
            ScatterColsCall {
                shape: ScatterColsShape {
                    rows,
                    n,
                    ld,
                    col_off,
                },
                src,
                dst,
            },
        )
    }

    /// The plain F32 gate `w` (`n` rows of `k` values) over `rows` rows of `x`, into columns
    /// `col_off..col_off + n` of `dst`. The plain gates are not EXL3 matrices; the router's dot
    /// kernel evaluates them at any row count.
    pub(in crate::family) fn plain(
        &self,
        w: DeviceRange<'_>,
        x: DeviceRange<'_>,
        shape: RouterShape,
        out: (DeviceRange<'_>, usize, usize),
    ) -> Result<(), EngError> {
        let (dst, ld, col_off) = out;
        let y = window(self.scratch.y, 0, shape.m * shape.n_expert * F32)?;
        self.launch.ctx.router_logits_rows(
            self.launch.stream,
            RouterLogitsCall {
                shape,
                x,
                w,
                logits: y,
            },
        )?;
        self.scatter(y, dst, (shape.m, shape.n_expert, ld, col_off))
    }
}

/// Clears the sticky arithmetic flag before the first layer. The write waits for the stream.
pub(in crate::family) fn clear_flag(
    launch: &Launch<'_>,
    flag: DeviceRange<'_>,
) -> Result<(), EngError> {
    flag.buffer()
        .write_at(launch.stream, flag.offset(), &0u32.to_le_bytes())
}

/// Refuses a chunk whose projections flagged an invalid value: one read after the last
/// layer, which waits for the stream.
pub(in crate::family) fn check_flag(
    launch: &Launch<'_>,
    flag: DeviceRange<'_>,
) -> Result<(), EngError> {
    let mut word = [0u8; 4];
    flag.buffer()
        .read_at(launch.stream, flag.offset(), &mut word)?;
    if u32::from_le_bytes(word) != 0 {
        return Err(EngError::ModelOutput(
            "EXL3 chunk projection flagged invalid arithmetic".to_owned(),
        ));
    }
    Ok(())
}

/// Test-only section timer of the EXL3 chunk: once enabled, each mark waits for the stream and
/// adds the time since the previous mark to its bucket, and the projections count their calls.
/// A timed run is slower than an untimed one; the buckets locate time, they do not measure the
/// untimed chunk. The family's stage names map onto the buckets in its own delegation; this
/// module owns only the thread-local machinery.
#[cfg(test)]
pub(in crate::family) mod split {
    use crate::kernels::cuda::DeviceStream;
    use std::{cell::RefCell, collections::BTreeMap, time::Instant};

    #[derive(Default)]
    struct State {
        last: Option<Instant>,
        seconds: BTreeMap<&'static str, f64>,
        calls: BTreeMap<&'static str, usize>,
    }

    thread_local! {
        static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
    }

    pub(in crate::family) fn enable() {
        STATE.with(|state| *state.borrow_mut() = Some(State::default()));
    }

    /// Ends timing and returns the seconds and the call counts per bucket.
    pub(in crate::family) fn take() -> (BTreeMap<&'static str, f64>, BTreeMap<&'static str, usize>)
    {
        STATE
            .with(|state| state.borrow_mut().take())
            .map(|state| (state.seconds, state.calls))
            .unwrap_or_default()
    }

    /// Waits for `stream` and charges the time since the last mark to `bucket`.
    pub(in crate::family) fn mark(stream: &DeviceStream, bucket: &'static str) {
        if STATE.with(|state| state.borrow().is_none()) {
            return;
        }
        stream.synchronize().expect("the timed stream drains");
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                let now = Instant::now();
                if let Some(last) = state.last {
                    *state.seconds.entry(bucket).or_default() += (now - last).as_secs_f64();
                }
                state.last = Some(now);
            }
        });
    }

    pub(in crate::family) fn count(bucket: &'static str) {
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                *state.calls.entry(bucket).or_default() += 1;
            }
        });
    }
}

/// Production build: the section timer does nothing.
#[cfg(not(test))]
pub(in crate::family) mod split {
    use crate::kernels::cuda::DeviceStream;

    #[inline(always)]
    pub(in crate::family) fn mark(_stream: &DeviceStream, _bucket: &'static str) {}

    #[inline(always)]
    pub(in crate::family) fn count(_bucket: &'static str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jobs_of(rows: usize) -> Vec<(u32, u32, u32)> {
        dense_jobs(rows, 256)
            .iter()
            .map(|job| (job.member, job.row0, job.rows))
            .collect()
    }

    /// A dense projection of `rows` rows is one member cut into jobs of at most 256 rows, back to
    /// back, the last one short: 1, 255 and 256 rows are one job, 257 rows are a full job and one
    /// row, a 1,024-row chunk is four jobs.
    /// Mutation: a tile height of 255 or 257 fails the 256- and 257-row cases; a `row0` that does
    /// not advance fails the multi-job cases.
    #[test]
    fn dense_jobs_cut_the_rows_into_tiles_of_the_job_height() {
        assert_eq!(jobs_of(1), [(0, 0, 1)]);
        assert_eq!(jobs_of(255), [(0, 0, 255)]);
        assert_eq!(jobs_of(256), [(0, 0, 256)]);
        assert_eq!(jobs_of(257), [(0, 0, 256), (0, 256, 1)]);
        assert_eq!(
            jobs_of(1024),
            [(0, 0, 256), (0, 256, 256), (0, 512, 256), (0, 768, 256)]
        );
        assert_eq!(
            jobs_of(1000),
            [(0, 0, 256), (0, 256, 256), (0, 512, 256), (0, 768, 232)]
        );
        assert!(dense_jobs(0, 256).is_empty());
    }
}
