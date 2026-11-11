//! The MoE block of a prefill chunk on an EXL3 pack: the shared expert and the routed experts of
//! `m` rows, then the combine.
//!
//! The job table of the routed projections lives on the device but is built on the host, so a
//! chunk reads the router's picks back once per layer (the router has queued them on the device),
//! sorts its `m * top_k` pairs by expert, and uploads the two row lists and the table in one
//! write. Gate, up and down then run each as ONE job-table launch over every expert of the
//! chunk (three kernels, whatever the number of experts), and one SwiGLU runs over every pair.
//! A job is at most [`EXL3_EXPERT_TILE_ROWS`](crate::family::project::exl3_jobs_plan::EXL3_EXPERT_TILE_ROWS)
//! rows of one expert; the expert's weights are read once per job,
//! which makes the routed pass layer-major: its weight traffic follows the experts a chunk picks,
//! not its rows.
//!
//! No row is copied to sort it. The input rotation of gate and up reads the rows of the normalized
//! chunk through the first list (the row each sorted pair reads), and the output rotation of down
//! writes each sorted pair straight to its `(row, rank)` place of `paired`, the layout the combine
//! reads, through the second list. `perm` holds the two lists and the job table, written by one
//! host upload.
//!
//! Stream rules the table relies on. Every launch of the block, the upload of the lists and the
//! table (`write_at` waits for the stream, so it never overtakes earlier work) and the read of the
//! picks run on the one stream of the step, in this order; no other stream touches the faces or the
//! scratch while a chunk runs, and the table is read by the kernels when they run, so it must not
//! be rewritten before the stream has passed the down projection (the next layer's upload comes
//! after this layer's combine on the same stream). A second stream, or a graph replay that
//! overlaps two layers, would need one table per layer in flight.
//!
//! Gate and up write their halves of the `[gate | up]` rows of `gu` directly, through the row
//! stride and column offset of the job-table output, the layout the SwiGLU reads.

use super::{Output, Projector, window};
use crate::family::project::cuda::exl3_jobs::{
    Lists, PairPlan, Scratch, Table, jobs_call, paired_scratch, plan_jobs, plan_pairs, read_ids,
    write_lists,
};
use crate::{
    EngError,
    family::{
        project::cuda::exl3::Exl3Bank,
        qwen36::cuda::{
            exl3_moe::Exl3MoeWeights,
            moe::MoeDims,
            prefill::{chunk_combine_shape, routed_swiglu, shared_swiglu},
        },
    },
    kernels::cuda::{
        DeviceRange,
        prefill::swiglu_rows::SwigluRowsCall,
        resident::moe_combine::{CombineCall, CombineOut, SharedTerm},
    },
};

const I32: usize = size_of::<i32>();
const F32: usize = size_of::<f32>();

pub(super) struct Bufs<'b> {
    /// F32 `[m][hidden]`: the normalized rows, read only.
    pub(super) x2: DeviceRange<'b>,
    /// i32 `[m][top_k]`: the router's picks, read only.
    pub(super) ids: DeviceRange<'b>,
    /// F32 `[m][top_k]`: the router's pick weights, read only.
    pub(super) weights: DeviceRange<'b>,
    /// F32 `[m]`: the shared expert's gate, read only.
    pub(super) shared_gate: DeviceRange<'b>,
    /// F32 `[m][hidden]`: the residual, updated in place by the combine.
    pub(super) hidden: DeviceRange<'b>,
    /// u32 `[2][m * top_k]`: for each sorted pair the row it reads, then the pair row it lands in;
    /// then the job table.
    pub(super) perm: DeviceRange<'b>,
    /// F32 `[m * top_k][2 * inter]`.
    pub(super) gu: DeviceRange<'b>,
    /// F32 `[m * top_k][inter]`.
    pub(super) act: DeviceRange<'b>,
    /// F32 `[m * top_k][hidden]`.
    pub(super) paired: DeviceRange<'b>,
    /// F16 `[m * top_k][max(hidden, inter)]`: the rotated input rows of one routed projection.
    pub(super) expert_xh: DeviceRange<'b>,
    /// F32 `[m * top_k][max(hidden, inter)]`: its partials.
    pub(super) expert_z: DeviceRange<'b>,
    /// F32 `[m][2 * shared_inter]`.
    pub(super) shared_gu: DeviceRange<'b>,
    /// F32 `[m][shared_inter]`.
    pub(super) shared_act: DeviceRange<'b>,
    /// F32 `[m][hidden]`.
    pub(super) shared_out: DeviceRange<'b>,
}

/// Reads the picks of the chunk back, which waits for the stream, and sorts them. Keeps the
/// device-test probe of the readback around the shared reader.
fn read_plan(
    projector: &Projector<'_>,
    ids: DeviceRange<'_>,
    dims: &MoeDims,
) -> Result<PairPlan, EngError> {
    #[cfg(test)]
    let began = std::time::Instant::now();
    let picks = read_ids(projector.launch, ids, dims)?;
    #[cfg(test)]
    crate::family::qwen36::cuda::exl3_bank::probe::readback(began.elapsed());
    // A fault-marked id (-1) falls through to the sort's refusal.
    plan_pairs(&picks, dims.n_expert, dims.top_k)
}

/// The shared expert: gate and up into the halves of `shared_gu`, the SwiGLU, down.
fn shared(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let (m, s) = (dims.rows, dims.shared_inter);
    for (half, weight) in [weights.shared_gate, weights.shared_up]
        .into_iter()
        .enumerate()
    {
        let out = Output::Columns {
            dst: bufs.shared_gu,
            ld: 2 * s,
            col_off: half * s,
        };
        projector.project(weight, bufs.x2, out)?;
    }
    projector.launch.ctx.swiglu_rows(
        projector.launch.stream,
        SwigluRowsCall {
            shape: shared_swiglu(m, dims),
            gu: bufs.shared_gu,
            act: bufs.shared_act,
        },
    )?;
    projector.project(
        weights.shared_down,
        bufs.shared_act,
        Output::Rows(bufs.shared_out),
    )
}

struct Routed<'a, 'b> {
    bank: &'a Exl3Bank,
    x: DeviceRange<'b>,
    /// The output face, its row stride and its first column.
    out: (DeviceRange<'b>, usize, usize),
    /// The row lists of the call: `x` is read through the first (over `x_rows` rows) and the
    /// output is written through the second (over `out_rows` rows); `None` is the table order.
    lists: Lists<'b>,
}

/// One routed projection over every job: three launches whatever the number of experts. Keeps
/// the split tap the test seams count around the shared call.
fn project_jobs(
    projector: &Projector<'_>,
    bufs: &Bufs<'_>,
    table: Table<'_, '_>,
    call: Routed<'_, '_>,
) -> Result<(), EngError> {
    #[cfg(test)]
    super::split::count("routed job-table projections");
    let scratch = Scratch {
        xh: bufs.expert_xh,
        z: bufs.expert_z,
    };
    let members = call.bank.members()?;
    let launch = jobs_call(
        projector.scratch.flag,
        scratch,
        table,
        &members,
        call.x,
        call.out,
        call.lists,
    )?;
    projector
        .launch
        .ctx
        .exl3_jobs(projector.launch.stream, launch)
}

/// The routed experts: the sort, gate and up reading the chunk rows through a list, one SwiGLU,
/// down writing each pair row to its place through a list.
fn routed(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    routed_with(
        projector,
        weights,
        bufs,
        dims,
        projector.launch.ctx.exl3_jobs_pair_enabled()?,
    )
    .map(|_| ())
}

/// [`routed`] with the pair request given; returns whether the gate and up projections rotated
/// their rows together (the request was on, the scratch fit, and the buffers were aligned).
fn routed_with(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
    pair: bool,
) -> Result<bool, EngError> {
    let plan = read_plan(projector, bufs.ids, dims)?;
    routed_plan(projector, weights, bufs, dims, plan, pair)
}

/// [`routed_with`] with the pair plan given: the caller has already read (and possibly
/// mapped) the picks, so this stages only the sort lists and the projections.
fn routed_plan(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
    plan: PairPlan,
    pair: bool,
) -> Result<bool, EngError> {
    let (m, pairs, i) = (dims.rows, dims.rows * dims.top_k, dims.inter);
    let jobs = plan_jobs(&plan.counts);
    write_lists(projector.launch, bufs.perm, (&plan, &jobs))?;
    super::split::mark(projector.launch.stream, "ids read and sort");
    let list = |at: usize| window(bufs.perm, at * pairs * I32, pairs * I32);
    let table = Table {
        jobs: &jobs,
        range: window(bufs.perm, 2 * pairs * I32, jobs.len() * 16)?,
        rows: pairs,
    };
    // Gate and up read the rows of the chunk through the first list, each sorted pair once.
    let listed = Lists {
        input: Some(list(0)?),
        output: None,
        x_rows: m,
        out_rows: pairs,
    };
    // With the pair request on, the rows are read and rotated once for both projections.
    let paired = if pair {
        paired_scratch(bufs.expert_xh, bufs.expert_z, dims)?
    } else {
        None
    };
    let mut taken = false;
    if let Some((first, second)) = paired {
        #[cfg(test)]
        for _ in 0..2 {
            super::split::count("routed job-table projections");
        }
        let (gate_members, up_members) = (weights.gate.members()?, weights.up.members()?);
        let calls = (
            jobs_call(
                projector.scratch.flag,
                first,
                table,
                &gate_members,
                bufs.x2,
                (bufs.gu, 2 * i, 0),
                listed,
            )?,
            jobs_call(
                projector.scratch.flag,
                second,
                table,
                &up_members,
                bufs.x2,
                (bufs.gu, 2 * i, i),
                listed,
            )?,
        );
        taken = projector.launch.ctx.exl3_jobs_pair_mode(
            projector.launch.stream,
            calls,
            (true, None),
        )?;
    } else {
        for (half, bank) in [weights.gate, weights.up].into_iter().enumerate() {
            project_jobs(
                projector,
                bufs,
                table,
                Routed {
                    bank,
                    x: bufs.x2,
                    out: (bufs.gu, 2 * i, half * i),
                    lists: listed,
                },
            )?;
        }
    }
    super::split::mark(projector.launch.stream, "routed gate/up");
    projector.launch.ctx.swiglu_rows(
        projector.launch.stream,
        SwigluRowsCall {
            shape: routed_swiglu(m, dims),
            gu: bufs.gu,
            act: bufs.act,
        },
    )?;
    super::split::mark(projector.launch.stream, "routed swiglu");
    // Down writes each sorted pair to its (row, rank) place through the second list.
    let call = Routed {
        bank: weights.down,
        x: bufs.act,
        out: (bufs.paired, dims.hidden, 0),
        lists: Lists {
            input: None,
            output: Some(list(1)?),
            x_rows: pairs,
            out_rows: pairs,
        },
    };
    project_jobs(projector, bufs, table, call)?;
    super::split::mark(projector.launch.stream, "routed down");
    Ok(taken)
}

/// `hidden = hidden + (picks in rank order + gate * shared)` for every row, in place.
fn combine(projector: &Projector<'_>, bufs: &Bufs<'_>, dims: &MoeDims) -> Result<(), EngError> {
    projector.launch.ctx.moe_combine(
        projector.launch.stream,
        CombineCall {
            shape: chunk_combine_shape(dims, dims.rows),
            res: bufs.hidden,
            down: bufs.paired,
            weights: bufs.weights,
            shared: Some(SharedTerm {
                gate: bufs.shared_gate,
                row: bufs.shared_out,
            }),
            out: CombineOut::InPlace,
        },
    )
}

/// Stages 10-17 of one layer of a chunk, after the router has written the picks: the shared
/// expert, the routed experts and the combine. It waits for the stream once, to read the picks.
pub(super) fn run(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    shared(projector, weights, bufs, dims)?;
    super::split::mark(projector.launch.stream, "shared expert");
    routed(projector, weights, bufs, dims)?;
    combine(projector, bufs, dims)?;
    super::split::mark(projector.launch.stream, "combine");
    Ok(())
}

/// The per-row slice of every row-scaled buffer of `bufs` at chunk row `r0`, for `sub.rows`
/// rows. The `perm` scratch stays whole: each sub-block rewrites and consumes it on the one
/// stream before the next sub-block's launches, so no slice is needed.
fn slice_at<'b>(
    bufs: &Bufs<'b>,
    full: &MoeDims,
    sub: &MoeDims,
    r0: usize,
) -> Result<Bufs<'b>, EngError> {
    let (h, i, k, s) = (full.hidden, full.inter, full.top_k, full.shared_inter);
    let wide = h.max(i);
    let f = |range: &DeviceRange<'b>, row: usize| {
        range
            .buffer()
            .range(range.offset() + r0 * row, sub.rows * row)
    };
    Ok(Bufs {
        x2: f(&bufs.x2, h * F32)?,
        ids: f(&bufs.ids, k * I32)?,
        weights: f(&bufs.weights, k * F32)?,
        shared_gate: f(&bufs.shared_gate, F32)?,
        hidden: f(&bufs.hidden, h * F32)?,
        perm: bufs.perm,
        gu: f(&bufs.gu, k * 2 * i * F32)?,
        act: f(&bufs.act, k * i * F32)?,
        paired: f(&bufs.paired, k * h * F32)?,
        expert_xh: f(&bufs.expert_xh, k * wide * 2)?,
        expert_z: f(&bufs.expert_z, k * wide * F32)?,
        shared_gu: f(&bufs.shared_gu, 2 * s * F32)?,
        shared_act: f(&bufs.shared_act, s * F32)?,
        shared_out: f(&bufs.shared_out, h * F32)?,
    })
}

/// Stages 10-17 of one banked chunk layer: the shared expert once for the whole chunk, then the
/// routed experts and the combine per sub-block, because the launches of one block read every
/// pick of its rows together and the bank holds only `slots` members. The whole chunk's picks
/// are read back once; each sub-block's picks are then mapped through the slot table
/// (refilling missed slots from the host store) and grouped as slot ids. `sub` carries each
/// sub-block's first row and its dims, built by the caller from its own chunk model.
pub(super) fn run_banked(
    projector: &Projector<'_>,
    weights: &Exl3MoeWeights<'_>,
    bufs: &Bufs<'_>,
    dims: &MoeDims,
    sub: &[(usize, MoeDims)],
    bank: &mut crate::family::qwen36::cuda::exl3_bank::LayerResolver<'_, '_>,
) -> Result<(), EngError> {
    shared(projector, weights, bufs, dims)?;
    super::split::mark(projector.launch.stream, "shared expert");
    // One readback for the whole chunk: each sub-block's mapping then runs host-side from
    // the picks already in hand, and only its mapped slot ids return to the device — the
    // per-sub-block readback, a blocking drain of the stream, is gone. The mapping stays
    // per sub-block: one sub-block's picks fit the slots together, a whole chunk's need not.
    let picks = read_ids(projector.launch, bufs.ids, dims)?;
    let pair = projector.launch.ctx.exl3_jobs_pair_enabled()?;
    let k = dims.top_k;
    for &(r0, ref sub_dims) in sub {
        let raw = &picks[r0 * k..r0 * k + sub_dims.rows * k];
        let sub_bufs = slice_at(bufs, dims, sub_dims, r0)?;
        let plan = if raw.iter().any(|&id| id < 0) {
            // A fault-marked pick reaches the sort's refusal, as on the eager path.
            plan_pairs(raw, dims.n_expert, k)?
        } else {
            let wanted = raw.iter().map(|&id| id as usize).collect::<Vec<_>>();
            let mapped = bank.resolve(projector.launch.stream, &wanted)?;
            let bytes: Vec<u8> = mapped
                .iter()
                .flat_map(|&slot| (slot as i32).to_le_bytes())
                .collect();
            sub_bufs.ids.buffer().write_at(
                projector.launch.stream,
                sub_bufs.ids.offset(),
                &bytes,
            )?;
            let slots: Vec<i32> = mapped.iter().map(|&slot| slot as i32).collect();
            plan_pairs(&slots, dims.n_expert, k)?
        };
        routed_plan(projector, weights, &sub_bufs, sub_dims, plan, pair)?;
        combine(projector, &sub_bufs, sub_dims)?;
    }
    super::split::mark(projector.launch.stream, "combine");
    Ok(())
}

#[cfg(test)]
#[path = "prefill_exl3_moe_device_tests.rs"]
mod device_tests;
