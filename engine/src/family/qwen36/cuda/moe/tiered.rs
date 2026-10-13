//! The tiered MoE block of one layer (OFF-2 D3, design `ledger/plans/OFF-2-DEVICE-WIRING.md`).
//!
//! WHAT: [`run_moe_block_tiered`] runs the native ten-op block with a host decision between the
//! router and the expert launches. The router runs as usual; its normed row, ids and weights
//! are copied to a pinned buffer and the host (a [`TieredLayerHost`]) maps each pick to a bank
//! slot (resident) or to slot 0 with weight 0.0 (missed, served on the CPU); the slot ids and
//! the device weights are written to two small device buffers; the nine remaining native ops
//! then run UNCHANGED over the bank stacks with those buffers in place of the router's ids and
//! weights; the host joins its CPU experts and, only when there are any, their partial sum is
//! added to the block output by a second `moe_combine` launch (top_k 1, weight 1.0, in place).
//! WHY: the existing launches, kernels and arena phases stay untouched; the only new device
//! objects are plain buffers (`TieredScratch`).
//!
//! The block syncs with the host: the event wait after the router outputs are copied, and the
//! blocking writes of the slot list and of the CPU partial. It is therefore eager only (a
//! captured window refuses every one of these calls with a typed error) and never part of the
//! captured step (decision 0047, amendment of 2026-10-13).
//!
//! `check_block` stays as it is. Its router checks run against the real expert count with the
//! stack facts of the bank replaced by full-depth facts (the real stacks are not in VRAM); the
//! bank stacks get their own checks in [`check_bank`].

use super::super::run_report::HostSyncs;
use super::{
    ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, RouteBufs, StackFacts, WeightFacts,
    check_block, launch_op, moe_sequence, op_router, refuse,
};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::resident::moe_combine::{CombineCall, CombineOut, CombineShape};
use crate::kernels::cuda::{DeviceBuffer, DeviceEvent, PinnedBuf};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();

/// What the router decided for this layer, read back from the device.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::family::qwen36::cuda) struct RouteRead {
    /// The pick ids in rank order (`-1` is the router-fault value).
    pub(in crate::family::qwen36::cuda) ids: Vec<i32>,
    /// The renormalized pick weights.
    pub(in crate::family::qwen36::cuda) weights: Vec<f32>,
    /// The normed row the CPU experts read.
    pub(in crate::family::qwen36::cuda) hidden: Vec<f32>,
}

/// What the device launches of the layer need, per pick.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::family::qwen36::cuda) struct LayerPlan {
    /// The bank slot of each pick: `-1` passes the router fault through, a missed pick reads 0.
    pub(in crate::family::qwen36::cuda) slot_ids: Vec<i32>,
    /// The combine weight of each pick: 0.0 for the picks served on the CPU.
    pub(in crate::family::qwen36::cuda) device_weights: Vec<f32>,
}

/// The host side of one layer: classifies the picks and runs the CPU experts.
pub(in crate::family::qwen36::cuda) trait TieredLayerHost {
    /// Classifies `route`, starts the CPU jobs of the missed picks and returns the device plan.
    fn begin_layer(&mut self, layer: usize, route: &RouteRead) -> Result<LayerPlan, EngError>;
    /// Joins the CPU jobs of `layer`: the partial sum (length `hidden`), or `None` when no pick
    /// was served on the CPU.
    fn finish_layer(&mut self, layer: usize) -> Result<Option<Vec<f32>>, EngError>;
}

/// The per-step objects of the tiered block: plain buffers, one set for all layers (the stream
/// orders every use), and not arena slots.
pub(in crate::family::qwen36::cuda) struct TieredScratch {
    slot_ids: DeviceBuffer,
    slot_weights: DeviceBuffer,
    cpu_partial: DeviceBuffer,
    ones: DeviceBuffer,
    host: PinnedBuf,
    event: DeviceEvent,
    hidden: usize,
    top_k: usize,
}

impl TieredScratch {
    /// Allocates the scratch for `hidden` values per row and `top_k` picks.
    pub(in crate::family::qwen36::cuda) fn new(
        ctx: &crate::kernels::cuda::CudaContext,
        hidden: usize,
        top_k: usize,
    ) -> Result<Self, EngError> {
        if hidden == 0 || top_k == 0 {
            return Err(refuse(format!(
                "tiered scratch needs hidden {hidden} and top_k {top_k} above zero"
            )));
        }
        let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
        Ok(Self {
            slot_ids: zeros(top_k * I32)?,
            slot_weights: zeros(top_k * F32)?,
            cpu_partial: zeros(hidden * F32)?,
            ones: ctx.upload(&1.0f32.to_le_bytes())?,
            host: ctx.pinned_buf((hidden + 2 * top_k) * F32)?,
            event: ctx.event()?,
            hidden,
            top_k,
        })
    }
}

/// The bank facts: every bank stack holds `slots` experts of the dims' shape, and `slots` is at
/// least `top_k` (the expert launch needs `n_used <= n_expert`) and at most the layer's experts.
fn check_bank(dims: &MoeDims, facts: &WeightFacts) -> Result<usize, EngError> {
    let slots = facts.gate.experts;
    if slots < dims.top_k || slots > dims.n_expert {
        return Err(refuse(format!(
            "the bank holds {slots} slots, a layer needs {}..={}",
            dims.top_k, dims.n_expert
        )));
    }
    let stacks = [
        ("gate", facts.gate, dims.inter, dims.hidden),
        ("up", facts.up, dims.inter, dims.hidden),
        ("down", facts.down, dims.hidden, dims.inter),
    ];
    for (name, stack, rows_per_slot, k) in stacks {
        if stack.experts != slots || stack.rows != slots * rows_per_slot || stack.k != k {
            return Err(refuse(format!(
                "the {name} bank holds {} rows of {} values for {} slots, {slots} slots need {} rows of {k}",
                stack.rows,
                stack.k,
                stack.experts,
                slots * rows_per_slot
            )));
        }
    }
    Ok(slots)
}

/// Every pure refusal of the tiered block: the bank facts, then `check_block` against the real
/// expert count with the stacks replaced by full-depth facts (they decide nothing else).
fn check_tiered(
    dims: &MoeDims,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
) -> Result<usize, EngError> {
    let facts = weights.facts();
    let slots = check_bank(dims, &facts)?;
    check_block(dims, &router_view(dims, facts), &bufs.facts())?;
    Ok(slots)
}

/// The weight facts `check_block` judges the router and the buffers against: the real facts
/// with the three bank stacks replaced by stacks of the layer's full depth.
fn router_view(dims: &MoeDims, facts: WeightFacts) -> WeightFacts {
    let full = |rows_per_expert: usize, k: usize| StackFacts {
        rows: dims.n_expert * rows_per_expert,
        k,
        experts: dims.n_expert,
    };
    WeightFacts {
        gate: full(dims.inter, dims.hidden),
        up: full(dims.inter, dims.hidden),
        down: full(dims.hidden, dims.inter),
        ..facts
    }
}

/// The plan the host returned: one entry per pick, a slot id of -1 or inside the bank, finite
/// weights. A refusal queues nothing.
fn check_plan(plan: &LayerPlan, top_k: usize, slots: usize) -> Result<(), EngError> {
    if plan.slot_ids.len() != top_k || plan.device_weights.len() != top_k {
        return Err(refuse(format!(
            "the layer plan holds {} ids and {} weights for {top_k} picks",
            plan.slot_ids.len(),
            plan.device_weights.len()
        )));
    }
    let outside = |id: &i32| *id < -1 || usize::try_from(*id).is_ok_and(|slot| slot >= slots);
    if let Some(id) = plan.slot_ids.iter().find(|id| outside(id)) {
        return Err(refuse(format!(
            "the layer plan names slot {id}, the bank has {slots}"
        )));
    }
    if plan.device_weights.iter().any(|w| !w.is_finite()) {
        return Err(refuse(
            "the layer plan holds a non-finite weight".to_owned(),
        ));
    }
    Ok(())
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

fn to_bytes<T: Copy>(values: &[T], one: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    values.iter().flat_map(|&v| one(v)).collect()
}

/// Runs the router, copies its outputs to the host and waits: the first host sync of the layer.
fn route_to_host(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    scratch: &mut TieredScratch,
    dims: &MoeDims,
) -> Result<RouteRead, EngError> {
    op_router(launch, weights, bufs, *dims)?;
    let (ids_at, weights_at) = (scratch.hidden * F32, (scratch.hidden + scratch.top_k) * F32);
    let route = &bufs.route;
    let ctx = launch.ctx;
    ctx.copy_to_pinned(launch.stream, route.norm_row, &mut scratch.host, 0)?;
    ctx.copy_to_pinned(launch.stream, route.ids, &mut scratch.host, ids_at)?;
    ctx.copy_to_pinned(launch.stream, route.weights, &mut scratch.host, weights_at)?;
    launch.stream.record(&scratch.event)?;
    scratch.host.wait(&scratch.event)?;
    let mut raw = vec![0u8; scratch.host.len()];
    scratch.host.read(0, &mut raw)?;
    let (id_words, _) = raw[ids_at..weights_at].as_chunks::<4>();
    let ids = id_words
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect();
    Ok(RouteRead {
        ids,
        weights: floats(&raw[weights_at..]),
        hidden: floats(&raw[..ids_at]),
    })
}

/// What one tiered block call works on: the real router and shared weights with the layer's BANK
/// stacks, the real buffer ranges, the scratch and the real dims (the router's expert count).
pub(in crate::family::qwen36::cuda) struct TieredBlock<'a, 'w, 'b> {
    pub(in crate::family::qwen36::cuda) weights: &'a MoeLayerWeights<'w>,
    pub(in crate::family::qwen36::cuda) bufs: &'a MoeBuffers<'b>,
    pub(in crate::family::qwen36::cuda) scratch: &'a mut TieredScratch,
    pub(in crate::family::qwen36::cuda) dims: &'a MoeDims,
}

/// The buffers of the nine ops after the router: the real ranges with the pick ids and weights
/// replaced by the scratch buffers the host plan was written to.
fn tiered_bufs<'x>(
    bufs: &MoeBuffers<'x>,
    scratch: &'x TieredScratch,
) -> Result<MoeBuffers<'x>, EngError> {
    let (route, expert) = (&bufs.route, &bufs.expert);
    Ok(MoeBuffers {
        route: RouteBufs {
            resid_in: route.resid_in,
            resid_out: route.resid_out,
            norm_row: route.norm_row,
            logits: route.logits,
            ids: scratch.slot_ids.range(0, scratch.top_k * I32)?,
            weights: scratch.slot_weights.range(0, scratch.top_k * F32)?,
            shared_gate: route.shared_gate,
            status: route.status,
        },
        expert: ExpertBufs {
            packs: expert.packs,
            expert_gu: expert.expert_gu,
            expert_down: expert.expert_down,
            shared_gu: expert.shared_gu,
            shared_out: expert.shared_out,
        },
    })
}

/// Adds the CPU partial to the block output: `resid_out += 1.0 * partial` through a second
/// `moe_combine` (top_k 1, in place; `1.0 * x` is exact and the kernel has no FMA).
fn add_cpu_partial(
    launch: &Launch<'_>,
    block: &mut TieredBlock<'_, '_, '_>,
    partial: &[f32],
) -> Result<(), EngError> {
    let hidden = block.dims.hidden;
    if partial.len() != hidden || partial.iter().any(|v| !v.is_finite()) {
        return Err(refuse(format!(
            "the CPU partial holds {} values (hidden {hidden}) or a non-finite one",
            partial.len()
        )));
    }
    let scratch = &block.scratch;
    scratch
        .cpu_partial
        .write_at(launch.stream, 0, &to_bytes(partial, f32::to_le_bytes))?;
    launch.ctx.moe_combine(
        launch.stream,
        CombineCall {
            shape: CombineShape {
                m: 1,
                n: hidden,
                top_k: 1,
                ld_down: hidden,
            },
            res: block.bufs.route.resid_out,
            down: scratch.cpu_partial.range(0, hidden * F32)?,
            weights: scratch.ones.range(0, F32)?,
            shared: None,
            out: CombineOut::InPlace,
        },
    )
}

/// The MoE block of one layer with a host decision between the router and the expert launches.
///
/// The caller keeps the block eager (it syncs with the host); a capture window refuses it. An
/// error from the host or from a device call is returned as it is: the step then poisons the
/// token, so nothing here undoes the launches already queued. On success it returns the host
/// waits it made: the route event wait, the two blocking plan writes, and the blocking write of
/// the CPU partial when a pick was served on the CPU.
pub(in crate::family::qwen36::cuda) fn run_moe_block_tiered(
    launch: &Launch<'_>,
    host: &mut dyn TieredLayerHost,
    layer: usize,
    block: &mut TieredBlock<'_, '_, '_>,
) -> Result<HostSyncs, EngError> {
    let (dims, weights) = (block.dims, block.weights);
    let slots = check_tiered(dims, weights, block.bufs)?;
    if block.scratch.hidden != dims.hidden || block.scratch.top_k != dims.top_k {
        return Err(refuse(format!(
            "the tiered scratch is for hidden {} and top_k {}, the block has {} and {}",
            block.scratch.hidden, block.scratch.top_k, dims.hidden, dims.top_k
        )));
    }
    let route = route_to_host(launch, weights, block.bufs, block.scratch, dims)?;
    let mut syncs = HostSyncs {
        waits: 1,
        writes: 0,
    };
    let plan = host.begin_layer(layer, &route)?;
    check_plan(&plan, dims.top_k, slots)?;
    let scratch = &*block.scratch;
    scratch.slot_ids.write_at(
        launch.stream,
        0,
        &to_bytes(&plan.slot_ids, i32::to_le_bytes),
    )?;
    syncs.writes += 1;
    scratch.slot_weights.write_at(
        launch.stream,
        0,
        &to_bytes(&plan.device_weights, f32::to_le_bytes),
    )?;
    syncs.writes += 1;
    let tiered = tiered_bufs(block.bufs, scratch)?;
    for op in moe_sequence(dims.top_k, WeightLayout::Native)
        .into_iter()
        .skip(1)
    {
        launch_op(launch, weights, &tiered, op)?;
    }
    if let Some(partial) = host.finish_layer(layer)? {
        add_cpu_partial(launch, block, &partial)?;
        syncs.writes += 1;
    }
    Ok(syncs)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_bad_request, buffer_facts, moe_dims, weight_facts};
    use super::*;

    /// The weight facts of the fixture with a bank of `slots` slots in place of the stacks.
    fn bank_facts(dims: &MoeDims, slots: usize) -> WeightFacts {
        let stack = |rows_per_slot: usize, k: usize| StackFacts {
            rows: slots * rows_per_slot,
            k,
            experts: slots,
        };
        WeightFacts {
            gate: stack(dims.inter, dims.hidden),
            up: stack(dims.inter, dims.hidden),
            down: stack(dims.hidden, dims.inter),
            ..weight_facts(dims)
        }
    }

    #[test]
    fn the_tiered_block_runs_the_native_ten_ops_split_after_the_router() {
        let seq = moe_sequence(8, WeightLayout::Native);
        assert_eq!(seq.len(), 10);
        assert_eq!(seq[0], super::super::MoeOp::Router);
        assert!(seq[1..].iter().all(|op| *op != super::super::MoeOp::Router));
        assert_eq!(seq[9], super::super::MoeOp::Combine);
    }

    #[test]
    fn the_router_view_passes_check_block_for_a_valid_bank_and_the_bank_is_checked_apart() {
        let dims = moe_dims(1);
        for slots in [dims.top_k, 40, dims.n_expert] {
            let facts = bank_facts(&dims, slots);
            assert_eq!(check_bank(&dims, &facts).unwrap(), slots);
            let view = router_view(&dims, facts);
            check_block(&dims, &view, &buffer_facts(&dims)).expect("the router view is valid");
        }
        // The real check_block alone would refuse the bank (its depth is not the expert count):
        // the reason the router view exists.
        let alone = check_block(&dims, &bank_facts(&dims, 40), &buffer_facts(&dims));
        assert_bad_request(alone, "a bank against the real expert count", "stack depth");
    }

    #[test]
    fn a_bank_geometry_that_does_not_match_the_layer_is_refused_before_any_launch() {
        let dims = moe_dims(1);
        let refuse_with = |facts: WeightFacts, case: &str, needle: &str| {
            assert_bad_request(check_bank(&dims, &facts).map(|_| ()), case, needle);
        };
        refuse_with(
            bank_facts(&dims, dims.top_k - 1),
            "fewer slots than picks",
            "slots",
        );
        refuse_with(
            bank_facts(&dims, dims.n_expert + 1),
            "more slots than experts",
            "slots",
        );
        let mut up_depth = bank_facts(&dims, 40);
        up_depth.up.experts = 39;
        refuse_with(up_depth, "an up bank of another depth", "up bank");
        let mut gate_rows = bank_facts(&dims, 40);
        gate_rows.gate.rows += dims.inter;
        refuse_with(
            gate_rows,
            "a gate bank with an extra slot of rows",
            "gate bank",
        );
        let mut down_k = bank_facts(&dims, 40);
        down_k.down.k = dims.inter + 256;
        refuse_with(down_k, "a down bank of the wrong width", "down bank");
        let mut hidden = bank_facts(&dims, 40);
        hidden.gate.k = dims.hidden / 2;
        refuse_with(hidden, "a gate bank of the wrong width", "gate bank");
    }

    #[test]
    fn an_invalid_plan_is_refused_before_any_copy() {
        let good = LayerPlan {
            slot_ids: vec![0, 3, -1, 7, 2, 2, 0, 1],
            device_weights: vec![0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0],
        };
        check_plan(&good, 8, 8).expect("a valid plan");
        let refuse_with = |plan: LayerPlan, case: &str, needle: &str| {
            assert_bad_request(check_plan(&plan, 8, 8), case, needle);
        };
        let short = LayerPlan {
            slot_ids: good.slot_ids[..7].to_vec(),
            ..good.clone()
        };
        refuse_with(short, "a short id list", "7 ids");
        let few_weights = LayerPlan {
            device_weights: good.device_weights[..7].to_vec(),
            ..good.clone()
        };
        refuse_with(few_weights, "a short weight list", "7 weights");
        let mut past = good.clone();
        past.slot_ids[2] = 8;
        refuse_with(past, "a slot past the bank", "slot 8");
        let mut below = good.clone();
        below.slot_ids[0] = -2;
        refuse_with(below, "an id below the fault value", "slot -2");
        let mut nan = good;
        nan.device_weights[1] = f32::NAN;
        refuse_with(nan, "a non-finite weight", "non-finite");
    }
}
