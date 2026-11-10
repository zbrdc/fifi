//! The EXL3 arms of the chunk stages: the dense projections of a prefill chunk through the EXL3
//! job-table projection.
//!
//! A dense projection of `m` rows is ONE job-table call (three launches): the rows are cut into
//! jobs of at most [`TILE_ROWS`] rows of the one matrix, and the tensor-core kernel decodes the
//! weights once per job, whatever the number of rows. The projections of the GGUF chunk read a
//! q8_1 copy of the rows instead, so the EXL3 chunk has no quantize stages. The table is the same
//! for every projection of a chunk, so the chunk driver uploads it once ([`write_dense_jobs`]).
//!
//! The DeltaNet and attention input rows hold several projections side by side; the job-table
//! output lands in its columns through the output row stride and column offset, so no scatter is
//! needed. Only the two plain DeltaNet gates, which are not EXL3 matrices, go through the router's
//! dot kernel and `scatter_cols`.
//!
//! Nothing here allocates. The arithmetic flag of the projections is sticky: the chunk driver
//! clears it before the first layer and reads it once after the last.

use super::{
    ChunkRun, ChunkSlot, Launch, LayerCtx, LayerMoe, Role, attn_proj_cols, delta_proj_cols,
};
use crate::{
    EngError,
    family::project::cuda::{Exl3JobsIo, exl3::Exl3Weight},
    kernels::cuda::{
        DeviceRange,
        prefill::scatter_cols::{ScatterColsCall, ScatterColsShape},
        resident::{
            exl3::Exl3Job,
            router::{RouterLogitsCall, RouterShape},
        },
    },
};

const F32: usize = size_of::<f32>();

// The most rows of one job: the tallest CTA of the job-table kernel.
use super::super::prefill_plan::EXL3_TILE_ROWS as TILE_ROWS;

/// The scratch of the projections of one chunk, caller-owned and disjoint from every operand.
#[derive(Clone, Copy)]
pub(super) struct Exl3Scratch<'b> {
    /// F16 `[m][k]` for the widest `k` of a dense projection: the rotated input rows.
    pub(super) xh: DeviceRange<'b>,
    /// F32 `[m][n]` for the widest `n`: the partials.
    pub(super) z: DeviceRange<'b>,
    /// F32 `[m][n]` of a plain gate, which waits here for `scatter_cols`.
    pub(super) y: DeviceRange<'b>,
    /// The sticky arithmetic flag, four bytes.
    pub(super) flag: DeviceRange<'b>,
    /// The job table of the dense projections, written by [`write_dense_jobs`].
    pub(super) jobs: DeviceRange<'b>,
}

#[derive(Clone, Copy)]
pub(super) enum Output<'b> {
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

fn window<'b>(
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

/// The jobs of a dense projection of `rows` rows: member 0, tiles of at most [`TILE_ROWS`] rows.
pub(super) fn dense_jobs(rows: usize) -> Vec<Exl3Job> {
    (0..rows)
        .step_by(TILE_ROWS)
        .map(|first| Exl3Job {
            member: 0,
            row0: first as u32,
            rows: TILE_ROWS.min(rows - first) as u32,
        })
        .collect()
}

/// Writes the job table of a chunk of `rows` rows. The write waits for the stream, so it runs once
/// per chunk, before the first layer, and not once per projection.
pub(super) fn write_dense_jobs(
    launch: &Launch<'_>,
    scratch: &Exl3Scratch<'_>,
    rows: usize,
) -> Result<(), EngError> {
    let bytes = Exl3Job::table(&dense_jobs(rows));
    let table = window(scratch.jobs, 0, bytes.len())?;
    table
        .buffer()
        .write_at(launch.stream, table.offset(), &bytes)
}

/// The launch, scratch and rows of one chunk for which the dense job table was written.
pub(super) struct Projector<'a> {
    pub(super) launch: &'a Launch<'a>,
    pub(super) scratch: Exl3Scratch<'a>,
    pub(super) rows: usize,
}

impl Projector<'_> {
    pub(super) fn project(
        &self,
        weight: &Exl3Weight,
        x: DeviceRange<'_>,
        out: Output<'_>,
    ) -> Result<(), EngError> {
        split::count("dense job-table projections");
        let jobs = dense_jobs(self.rows);
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
    pub(super) fn plain(
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

#[cfg(test)]
#[path = "prefill_exl3_device_tests.rs"]
mod device_tests;
#[path = "prefill_exl3_moe.rs"]
mod moe;

pub(super) fn projector<'a>(
    run: &'a ChunkRun<'_, '_>,
    ctx: &'a LayerCtx<'_, '_>,
) -> Result<Projector<'a>, EngError> {
    Ok(Projector {
        launch: ctx.launch,
        scratch: run.exl3_scratch()?,
        rows: run.rows.m(),
    })
}

/// Stage 3d, the DeltaNet in-projections: the two EXL3 matrices into their columns of `ProjD`,
/// the two plain gates through the router's dot kernel.
pub(super) fn delta_proj(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let (m, g) = (run.rows.m(), run.model.geometry);
    let (x, dst) = (run.v(ChunkSlot::X)?, run.v(ChunkSlot::ProjD)?);
    let projector = projector(run, ctx)?;
    for (role, n, col_off) in delta_proj_cols(g) {
        if matches!(role, Role::SsmBeta | Role::SsmAlpha) {
            let shape = RouterShape {
                m,
                hidden: g.hidden,
                n_expert: n,
                top_k: 1,
            };
            projector.plain(ctx.weights.raw(role)?, x, shape, (dst, g.delta.ld, col_off))?;
            continue;
        }
        let weight = ctx.weights.exl3(role)?;
        let out = Output::Columns {
            dst,
            ld: g.delta.ld,
            col_off,
        };
        projector.project(weight, x, out)?;
    }
    Ok(())
}

/// Stage 7d, the DeltaNet output projection of the scan rows into `Delta`.
pub(super) fn ssm_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let weight = ctx.weights.exl3(Role::SsmOut)?;
    let out = Output::Rows(run.v(ChunkSlot::Delta)?);
    projector(run, ctx)?.project(weight, run.v(ChunkSlot::OOut)?, out)
}

/// Stage 3a, the three attention projections into their columns of `ProjA`.
pub(super) fn attn_proj(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let g = run.model.geometry;
    let (x, dst) = (run.v(ChunkSlot::X)?, run.v(ChunkSlot::ProjA)?);
    let projector = projector(run, ctx)?;
    for (role, _n, col_off) in attn_proj_cols(g) {
        let weight = ctx.weights.exl3(role)?;
        let out = Output::Columns {
            dst,
            ld: g.attn.ld,
            col_off,
        };
        projector.project(weight, x, out)?;
    }
    Ok(())
}

/// Stage 7a, the attention output projection of the flash rows into `Delta`.
pub(super) fn attn_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let weight = ctx.weights.exl3(Role::AttnOutput)?;
    let out = Output::Rows(run.v(ChunkSlot::Delta)?);
    projector(run, ctx)?.project(weight, run.v(ChunkSlot::AttnOut)?, out)
}

/// Stages 10-17 of one layer: the shared expert, the routed experts and the combine.
pub(super) fn moe_block(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let LayerMoe::Exl3(weights) = ctx.moe else {
        return Err(refuse(
            "the MoE weights of the layer are not EXL3".to_owned(),
        ));
    };
    let view = |slot| run.v(slot);
    let bufs = moe::Bufs {
        x2: view(ChunkSlot::X2)?,
        ids: view(ChunkSlot::TopkId)?,
        weights: view(ChunkSlot::TopkWeight)?,
        shared_gate: view(ChunkSlot::SharedGate)?,
        hidden: view(ChunkSlot::Hidden)?,
        perm: view(ChunkSlot::ExpertPerm)?,
        gu: view(ChunkSlot::Gu)?,
        act: view(ChunkSlot::Act)?,
        paired: view(ChunkSlot::Down)?,
        expert_xh: view(ChunkSlot::ExpertXh)?,
        expert_z: view(ChunkSlot::ExpertZ)?,
        shared_gu: view(ChunkSlot::GuSh)?,
        shared_act: view(ChunkSlot::ActSh)?,
        shared_out: view(ChunkSlot::ShOut)?,
    };
    let dims = run.model.dims_at(run.rows.m());
    // A banked chunk runs the shared expert once for the whole chunk, then the routed experts
    // and the combine per sub-block whose whole pick set fits the slots together, mapping each
    // sub-block's picks through the layer's slot table first.
    if let Some(cell) = run.bank {
        let top_k = dims.top_k;
        let sub_rows = (cell.borrow().slots() / top_k).max(1);
        let m = dims.rows;
        let sub = (0..m)
            .step_by(sub_rows)
            .map(|r0| (r0, run.model.dims_at((m - r0).min(sub_rows))))
            .collect::<Vec<_>>();
        let mut resolver = cell.borrow_mut();
        let mut bank = resolver.layer(ctx.layer);
        moe::run_banked(
            &projector(run, ctx)?,
            weights,
            &bufs,
            &dims,
            &sub,
            &mut bank,
        )
    } else {
        moe::run(&projector(run, ctx)?, weights, &bufs, &dims)
    }
}

/// Clears the sticky arithmetic flag before the first layer. The write waits for the stream.
pub(super) fn clear_flag(launch: &Launch<'_>, run: &ChunkRun<'_, '_>) -> Result<(), EngError> {
    let flag = run.v(ChunkSlot::Exl3Flag)?;
    flag.buffer()
        .write_at(launch.stream, flag.offset(), &0u32.to_le_bytes())
}

/// Refuses a chunk whose projections flagged an invalid value: one read after the last
/// layer, which waits for the stream.
pub(super) fn check_flag(launch: &Launch<'_>, run: &ChunkRun<'_, '_>) -> Result<(), EngError> {
    let flag = run.v(ChunkSlot::Exl3Flag)?;
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
/// untimed chunk. Production code carries no timer.
#[cfg(test)]
pub(in crate::family::qwen36::cuda) mod split {
    use super::super::Stage;
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

    pub(in crate::family::qwen36::cuda) fn enable() {
        STATE.with(|state| *state.borrow_mut() = Some(State::default()));
    }

    /// Ends timing and returns the seconds and the call counts per bucket.
    pub(in crate::family::qwen36::cuda) fn take()
    -> (BTreeMap<&'static str, f64>, BTreeMap<&'static str, usize>) {
        STATE
            .with(|state| state.borrow_mut().take())
            .map(|state| (state.seconds, state.calls))
            .unwrap_or_default()
    }

    /// Waits for `stream` and charges the time since the last mark to `bucket`.
    pub(in crate::family::qwen36::cuda) fn mark(stream: &DeviceStream, bucket: &'static str) {
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

    /// Charges the time since the last mark to the bucket of `stage`.
    pub(in crate::family::qwen36::cuda) fn mark_stage(stream: &DeviceStream, stage: Stage) {
        mark(
            stream,
            match stage {
                Stage::NormA => "norm (input)",
                Stage::NormF => "norm (post mixer)",
                Stage::DeltaProj => "GDN input projections",
                Stage::Conv => "GDN conv",
                Stage::Scan => "GDN scan",
                Stage::SsmOut => "GDN output projection",
                Stage::AttnProj => "attention input projections",
                Stage::Prep => "attention prep",
                Stage::Flash => "attention flash",
                Stage::AttnOut => "attention output projection",
                Stage::QuantA | Stage::QuantDOut | Stage::QuantAOut => "quantize",
                Stage::Route => "router",
                Stage::Moe => "moe tail",
                // Timing labels recorded inside the tiered MoE arm; EXL3 never
                // charges them (no EXL3 tiered chunk path).
                Stage::MoeStart
                | Stage::MoeShared
                | Stage::MoeStage
                | Stage::MoeScatter
                | Stage::MoeBankUp
                | Stage::MoeColdUp
                | Stage::MoeSwiglu
                | Stage::MoeGather
                | Stage::MoeBankDown
                | Stage::MoeColdDown
                | Stage::MoeCombine => "tiered moe (labels)",
            },
        );
    }

    pub(super) fn count(bucket: &'static str) {
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                *state.calls.entry(bucket).or_default() += 1;
            }
        });
    }
}

/// Production build: the section timer does nothing.
#[cfg(not(test))]
pub(in crate::family::qwen36::cuda) mod split {
    use super::super::Stage;
    use crate::kernels::cuda::DeviceStream;

    #[inline(always)]
    pub(in crate::family::qwen36::cuda) fn mark_stage(_stream: &DeviceStream, _stage: Stage) {}

    #[inline(always)]
    pub(super) fn mark(_stream: &DeviceStream, _bucket: &'static str) {}

    #[inline(always)]
    pub(super) fn count(_bucket: &'static str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jobs_of(rows: usize) -> Vec<(u32, u32, u32)> {
        dense_jobs(rows)
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
        assert_eq!(TILE_ROWS, 256);
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
        assert!(dense_jobs(0).is_empty());
    }
}
