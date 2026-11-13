//! The EXL3 arms of the chunk stages: this family's glue over the family-neutral projection
//! core. The dense job-table projection ([`Projector`]), the plain-gate fallback and the routed
//! MoE block live in [`crate::family::exl3::prefill`] and [`crate::family::exl3::prefill_moe`];
//! this module binds them to the chunk driver: the tile height ([`TILE_ROWS`]), the sticky
//! flag's chunk slot, and the stage functions ([`delta_proj`], [`attn_proj`], [`ssm_out`],
//! [`attn_out`], [`moe_block`]) the driver calls by name.
//!
//! The projections of the GGUF chunk read a q8_1 copy of the rows instead, so the EXL3 chunk has
//! no quantize stages.

use super::{
    ChunkRun, ChunkSlot, Launch, LayerCtx, LayerMoe, Role, attn_proj_cols, delta_proj_cols,
};
use crate::{
    EngError,
    kernels::cuda::resident::{exl3::Exl3Job, router::RouterShape},
};
// The moved core's types, re-exported for the driver, the device tests and the golden host
// test, which reach them through this module.
use crate::family::exl3::prefill as core;
#[cfg(test)]
pub(super) use crate::family::exl3::prefill::F32;
pub(super) use crate::family::exl3::prefill::{Exl3Scratch, Output, Projector};

// The most rows of one job: the tallest CTA of the job-table kernel, this family's own
// constant, passed to every call of the moved core.
use super::super::prefill_plan::EXL3_TILE_ROWS as TILE_ROWS;

/// The jobs of a dense projection of `rows` rows at this family's tile height.
/// `pub(in ...::cuda)` so the golden host test can digest the table.
pub(in crate::family::qwen36::cuda) fn dense_jobs(rows: usize) -> Vec<Exl3Job> {
    core::dense_jobs(rows, TILE_ROWS)
}

/// Writes the job table of a chunk of `rows` rows at this family's tile height.
pub(super) fn write_dense_jobs(
    launch: &Launch<'_>,
    scratch: &Exl3Scratch<'_>,
    rows: usize,
) -> Result<(), EngError> {
    core::write_dense_jobs(launch, scratch, rows, TILE_ROWS)
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
        tile_rows: TILE_ROWS,
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
        return Err(EngError::BadRequest(
            "prefill chunk: EXL3: the MoE weights of the layer are not EXL3".to_owned(),
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
    core::clear_flag(launch, run.v(ChunkSlot::Exl3Flag)?)
}

/// Refuses a chunk whose projections flagged an invalid value: one read after the last
/// layer, which waits for the stream.
pub(super) fn check_flag(launch: &Launch<'_>, run: &ChunkRun<'_, '_>) -> Result<(), EngError> {
    core::check_flag(launch, run.v(ChunkSlot::Exl3Flag)?)
}

/// Test-only section timer of the EXL3 chunk: the buckets live in the family-neutral core's
/// thread-local ([`core::split`]); this delegation maps the driver's stage names onto them.
/// Production code carries no timer of its own here; the serve-phase hook of the parent module
/// (`FIFI_Q36_SERVE_PHASE`) is the one production timer and marks stages around this loop when
/// armed.
#[cfg(test)]
pub(in crate::family::qwen36::cuda) mod split {
    use super::super::Stage;
    use crate::{family::exl3::prefill as core, kernels::cuda::DeviceStream};
    use std::collections::BTreeMap;

    pub(in crate::family::qwen36::cuda) use core::split::enable;

    /// Ends timing and returns the seconds and the call counts per bucket.
    pub(in crate::family::qwen36::cuda) fn take()
    -> (BTreeMap<&'static str, f64>, BTreeMap<&'static str, usize>) {
        core::split::take()
    }

    /// Charges the time since the last mark to the bucket of `stage`.
    pub(in crate::family::qwen36::cuda) fn mark_stage(stream: &DeviceStream, stage: Stage) {
        core::split::mark(
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
            },
        );
    }
}

/// Production build: the section timer does nothing.
#[cfg(not(test))]
pub(in crate::family::qwen36::cuda) mod split {
    use super::super::Stage;
    use crate::kernels::cuda::DeviceStream;

    #[inline(always)]
    pub(in crate::family::qwen36::cuda) fn mark_stage(_stream: &DeviceStream, _stage: Stage) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile height this family passes to every call of the moved core: the tallest CTA of
    /// the job-table kernel, 256 rows. The core's own tiling test covers the cutting.
    #[test]
    fn the_tile_height_is_256() {
        assert_eq!(TILE_ROWS, 256);
    }
}
