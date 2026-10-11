//! The MoE block of ONE layer of the resident Qwen3.6 decode step at ONE token (`m = 1`): a fixed
//! launch order on the one owned stream (decision 0047 item 3, packet RD-9c). The router (RD-5)
//! writes the expert ids, the pick weights and the shared gate on the device; every expert op
//! reads its id from device memory through `matmul_expert` (RD-0b), so the host never reads an id
//! and no host value depends on device data. [`MoeOp`] is the sequence as data, [`moe_sequence`]
//! the order, [`op_slots`] the slots each op touches and the phase it runs in (RD-9a's plan), and
//! [`check_block`] every pure refusal, before the first launch.
//!
//! The order (one stream, phase numbers of `step_plan`): ROUTE the router; GATE_UP the gate and
//! up matmul of every pick slot, then the shared gate and up; SWIGLU the shared pack first (the
//! shared `gu` slot may share bytes with `ExpertDown`, whose first write comes later), then per
//! slot the pack into the one `XfDown` followed by that slot's down matmul (the interleave is why
//! one pack buffer is enough); DOWN the shared down; COMBINE the residual plus the picks plus the
//! gated shared row into the other residual buffer. [`run_moe_block`] queues exactly this.
//!
//! The block makes no allocation, copies nothing, never syncs and never maps the status
//! word: the router ORs its fault bit on the device, and mapping is the step loop's job (RD-9d).
//! Eager and captured steps call the same function; there is no capture branch.
//!
//! Design guidance only, no code copied. The device-side expert routing this block relies on is
//! RD-5's (credited there from MegaCapybara v1.35 static RE, section 6.1 C9).

use super::dims::StepDims;
use super::step_plan::{Phase, Slot};
use crate::EngError;
use crate::kernels::cuda::resident::matmul_expert::{ExpertCall, ExpertSelect, ExpertStack};
use crate::kernels::cuda::resident::moe_combine::{
    CombineCall, CombineOut, CombineShape, SharedTerm,
};
use crate::kernels::cuda::resident::router::{
    RouterCall, RouterOut, RouterShape, RouterWeights, router_logits_bytes,
};
use crate::kernels::cuda::resident::swiglu_pack::SwigluPackCall;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream, MatmulIo, Projection, TiledWeight,
    xf_bytes,
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();
/// The width quantum of `swiglu_pack` (`k_pad`, RD-4b's `WIDTH_QUANTUM`).
const PACK_WIDTH: usize = 256;

/// The dims of one MoE block: the row count of the step and the MoE geometry. No model number
/// lives here; [`Self::new`] reads the checked step dims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MoeDims {
    /// Token rows; stage 1 runs this block at one row (verify windows get their own packet).
    pub(super) rows: usize,
    /// Hidden size: the router and shared-projection width, the residual width.
    pub(super) hidden: usize,
    /// Routed experts of the layer.
    pub(super) n_expert: usize,
    /// Experts picked per row.
    pub(super) top_k: usize,
    /// Rows of one expert's gate and up projections.
    pub(super) inter: usize,
    /// Rows of the shared expert's gate and up projections.
    pub(super) shared_inter: usize,
}

impl MoeDims {
    /// The block dims of the checked step `dims` at `rows` rows.
    pub(super) fn new(dims: &StepDims, rows: usize) -> Self {
        let moe = &dims.shape.moe;
        Self {
            rows,
            hidden: dims.shape.hidden,
            n_expert: moe.experts,
            top_k: moe.top_k,
            inter: moe.inter,
            shared_inter: moe.shared_inter,
        }
    }
}

/// One launch of the MoE block, as data. The launcher dispatches on it; the tests check it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MoeOp {
    /// The router: ids, pick weights, shared gate, fault bit.
    Router,
    /// The gate matmul of pick slot `usize` against `XfMoe`.
    ExpertGate(usize),
    /// The up matmul of pick slot `usize` against `XfMoe`.
    ExpertUp(usize),
    /// The shared expert's gate matmul.
    SharedGate,
    /// The shared expert's up matmul.
    SharedUp,
    /// The shared expert's SwiGLU pack.
    SwigluShared,
    /// Pick slot `usize`'s SwiGLU pack into the one down pack.
    SwigluExpert(usize),
    /// Pick slot `usize`'s down matmul.
    ExpertDown(usize),
    /// The shared expert's down matmul.
    SharedDown,
    /// The combine into the other residual buffer.
    Combine,
}

/// The documented order of 4.2 as the launch list: router; every slot's gate and up; the shared
/// gate and up; the shared pack; per slot the pack then the down matmul; the shared down; the
/// combine. `top_k` slots make `3 * top_k + 14` ops.
pub(super) fn moe_sequence(top_k: usize) -> Vec<MoeOp> {
    let mut ops = vec![MoeOp::Router];
    for slot in 0..top_k {
        ops.push(MoeOp::ExpertGate(slot));
        ops.push(MoeOp::ExpertUp(slot));
    }
    ops.push(MoeOp::SharedGate);
    ops.push(MoeOp::SharedUp);
    ops.push(MoeOp::SwigluShared);
    for slot in 0..top_k {
        ops.push(MoeOp::SwigluExpert(slot));
        ops.push(MoeOp::ExpertDown(slot));
    }
    ops.push(MoeOp::SharedDown);
    ops.push(MoeOp::Combine);
    ops
}

/// The arena slots one op reads and writes, and the phase it runs in. The status word is not an
/// arena slot: the router alone writes it, and this block never maps it.
pub(super) struct OpSlots {
    /// Slots the op reads; live in the plan from `first` on.
    pub(super) reads: &'static [Slot],
    /// Slots the op writes; live in the plan through `last`.
    pub(super) writes: &'static [Slot],
    /// The phase of `step_plan` the op runs in.
    pub(super) phase: usize,
}

/// The slots and phase of `op`. The pure view fixes the residual parity as `ResidA` in and
/// `ResidB` out; both slots share one lifetime, and the launcher takes the real ranges as data.
pub(super) fn op_slots(op: MoeOp) -> OpSlots {
    match op {
        MoeOp::Router => OpSlots {
            reads: &[Slot::NormRow, Slot::RouterLogits],
            writes: &[
                Slot::RouterLogits,
                Slot::TopkId,
                Slot::TopkWeight,
                Slot::SharedGate,
            ],
            phase: Phase::Route as usize,
        },
        MoeOp::ExpertGate(_) | MoeOp::ExpertUp(_) => OpSlots {
            reads: &[Slot::XfMoe, Slot::TopkId],
            writes: &[Slot::ExpertGu],
            phase: Phase::GateUp as usize,
        },
        MoeOp::SharedGate | MoeOp::SharedUp => OpSlots {
            reads: &[Slot::XfMoe],
            writes: &[Slot::SharedGu],
            phase: Phase::GateUp as usize,
        },
        MoeOp::SwigluShared => OpSlots {
            reads: &[Slot::SharedGu],
            writes: &[Slot::XfShared],
            phase: Phase::Swiglu as usize,
        },
        MoeOp::SwigluExpert(_) => OpSlots {
            reads: &[Slot::ExpertGu],
            writes: &[Slot::XfDown(0)],
            phase: Phase::Swiglu as usize,
        },
        MoeOp::ExpertDown(_) => OpSlots {
            reads: &[Slot::XfDown(0), Slot::TopkId],
            writes: &[Slot::ExpertDown],
            phase: Phase::Down as usize,
        },
        MoeOp::SharedDown => OpSlots {
            reads: &[Slot::XfShared],
            writes: &[Slot::SharedOut],
            phase: Phase::Down as usize,
        },
        MoeOp::Combine => OpSlots {
            reads: &[
                Slot::ResidA,
                Slot::ExpertDown,
                Slot::TopkWeight,
                Slot::SharedGate,
                Slot::SharedOut,
            ],
            writes: &[Slot::ResidB],
            phase: Phase::Combine as usize,
        },
    }
}

/// One device range as plain numbers: `offset` and `bytes` inside the allocation named `buffer`.
/// Ranges of different allocations never overlap, so `buffer` is compared by identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RangeFacts {
    /// Byte offset inside the allocation.
    pub(super) offset: usize,
    /// Bytes of the range.
    pub(super) bytes: usize,
    /// Identity of the allocation.
    pub(super) buffer: usize,
}

/// The geometry of one expert stack: the whole tiled weight's rows and width, and its depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StackFacts {
    /// Rows of the whole stack, `experts` experts back to back.
    pub(super) rows: usize,
    /// Values per row.
    pub(super) k: usize,
    /// Experts of the stack.
    pub(super) experts: usize,
}

/// The geometry of one dense projection weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProjFacts {
    /// Rows of the weight.
    pub(super) rows: usize,
    /// Values per row.
    pub(super) k: usize,
}

/// The weights of the block as plain facts, for [`check_block`] without a device.
pub(super) struct WeightFacts {
    /// The raw F32 router matrix allocation.
    pub(super) router: RangeFacts,
    /// The raw F32 shared-gate vector allocation.
    pub(super) router_shared: RangeFacts,
    /// The gate stack.
    pub(super) gate: StackFacts,
    /// The up stack.
    pub(super) up: StackFacts,
    /// The down stack.
    pub(super) down: StackFacts,
    /// The shared expert's gate projection.
    pub(super) shared_gate: ProjFacts,
    /// The shared expert's up projection.
    pub(super) shared_up: ProjFacts,
    /// The shared expert's down projection.
    pub(super) shared_down: ProjFacts,
}

/// The routing and residual ranges of the block as plain facts.
pub(super) struct RouteFacts {
    /// f32 `[hidden]`: the residual after the mixer add, the block input.
    pub(super) resid_in: RangeFacts,
    /// f32 `[hidden]`: the other residual buffer, the block output.
    pub(super) resid_out: RangeFacts,
    /// f32 `[hidden]`: the normalized row, the router input.
    pub(super) norm_row: RangeFacts,
    /// The router's logits scratch.
    pub(super) logits: RangeFacts,
    /// i32 `[top_k]`: the pick ids, read on the device only.
    pub(super) ids: RangeFacts,
    /// f32 `[top_k]`: the renormalized pick weights.
    pub(super) weights: RangeFacts,
    /// f32: the shared gate, sigmoid-ed by the router.
    pub(super) shared_gate: RangeFacts,
    /// The status word the router ORs its fault bit into.
    pub(super) status: RangeFacts,
}

/// The expert ranges of the block as plain facts.
pub(super) struct ExpertFacts {
    /// The fp16 pack of the normalized row, every gate and up matmul's input.
    pub(super) xf_moe: RangeFacts,
    /// f32 `[top_k][gate | up]`, slot `s`'s halves contiguous for its pack.
    pub(super) expert_gu: RangeFacts,
    /// The fp16 pack one slot's down matmul reads.
    pub(super) xf_down: RangeFacts,
    /// f32 `[top_k][hidden]` in pick order, the combine's expert term.
    pub(super) expert_down: RangeFacts,
    /// f32 `[gate | up]` of the shared expert.
    pub(super) shared_gu: RangeFacts,
    /// The fp16 pack the shared down matmul reads.
    pub(super) xf_shared: RangeFacts,
    /// f32 `[hidden]`, the shared expert's down output.
    pub(super) shared_out: RangeFacts,
}

/// The fifteen block buffers as plain facts.
pub(super) struct BufferFacts {
    /// Routing and residual ranges.
    pub(super) route: RouteFacts,
    /// Expert and shared-expert ranges.
    pub(super) expert: ExpertFacts,
}

/// The bytes each range of the block must hold, from the dims alone.
struct NeedBytes {
    resid: usize,
    norm_row: usize,
    logits: usize,
    picks: usize,
    ids: usize,
    gate: usize,
    status: usize,
    slot_gu: usize,
    expert_gu: usize,
    xf_moe: usize,
    xf_down: usize,
    xf_shared: usize,
    expert_down: usize,
    shared_gu: usize,
    shared_out: usize,
}

impl NeedBytes {
    fn of(dims: &MoeDims) -> Result<Self, EngError> {
        let row = |width: usize| product(&[width, F32]);
        let xf = |k: usize| {
            xf_bytes(dims.rows, k)
                .ok_or_else(|| refuse(format!("the pack of {k} values overflows")))
        };
        Ok(Self {
            resid: row(dims.hidden)?,
            norm_row: row(dims.hidden)?,
            logits: router_logits_bytes(RouterShape {
                m: dims.rows,
                hidden: dims.hidden,
                n_expert: dims.n_expert,
                top_k: dims.top_k,
            })?,
            picks: product(&[dims.top_k, F32])?,
            ids: product(&[dims.top_k, I32])?,
            gate: row(1)?,
            status: product(&[I32])?,
            slot_gu: row(2 * dims.inter)?,
            expert_gu: product(&[dims.top_k, dims.inter, 2, F32])?,
            xf_moe: xf(dims.hidden)?,
            xf_down: xf(dims.inter)?,
            xf_shared: xf(dims.shared_inter)?,
            expert_down: product(&[dims.top_k, dims.hidden, F32])?,
            shared_gu: row(2 * dims.shared_inter)?,
            shared_out: row(dims.hidden)?,
        })
    }
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("moe block: {detail}"))
}

fn product(parts: &[usize]) -> Result<usize, EngError> {
    parts.iter().try_fold(1usize, |total, &part| {
        total
            .checked_mul(part)
            .ok_or_else(|| refuse("a size overflows".to_owned()))
    })
}

/// Every pure refusal of the block, checked before the first launch so a captured block cannot be
/// half queued: the row count, the shape rules, the weight geometry, the byte size of every range
/// and the overlap freedom the router and combine wrappers require.
pub(super) fn check_block(
    dims: &MoeDims,
    weights: &WeightFacts,
    bufs: &BufferFacts,
) -> Result<(), EngError> {
    check_dims(dims)?;
    check_weights(dims, weights)?;
    let need = NeedBytes::of(dims)?;
    check_sizes(&need, bufs)?;
    check_apart(bufs)
}

fn check_dims(dims: &MoeDims) -> Result<(), EngError> {
    if dims.rows != 1 {
        return Err(refuse(format!(
            "the block takes one row, got {}",
            dims.rows
        )));
    }
    if dims.top_k > dims.n_expert {
        return Err(refuse(format!(
            "top_k {} is above the {} experts",
            dims.top_k, dims.n_expert
        )));
    }
    if !dims.inter.is_multiple_of(PACK_WIDTH) || !dims.shared_inter.is_multiple_of(PACK_WIDTH) {
        return Err(refuse(format!(
            "inter {} and shared_inter {} must be multiples of {PACK_WIDTH}",
            dims.inter, dims.shared_inter
        )));
    }
    Ok(())
}

fn check_weights(dims: &MoeDims, weights: &WeightFacts) -> Result<(), EngError> {
    let router = product(&[dims.n_expert, dims.hidden, F32])?;
    if weights.router.bytes < router {
        return Err(refuse(format!(
            "the router weight holds {} bytes, {} experts of {} values need {router}",
            weights.router.bytes, dims.n_expert, dims.hidden
        )));
    }
    let shared = product(&[dims.hidden, F32])?;
    if weights.router_shared.bytes < shared {
        return Err(refuse(format!(
            "the shared router vector holds {} bytes, {} values need {shared}",
            weights.router_shared.bytes, dims.hidden
        )));
    }
    let gu_rows = product(&[dims.n_expert, dims.inter])?;
    let down_rows = product(&[dims.n_expert, dims.hidden])?;
    let stacks = [
        ("gate", weights.gate, gu_rows, dims.hidden),
        ("up", weights.up, gu_rows, dims.hidden),
        ("down", weights.down, down_rows, dims.inter),
    ];
    for (name, stack, rows, k) in stacks {
        if stack.experts != dims.n_expert {
            return Err(refuse(format!(
                "the {name} stack depth {} does not equal the {} experts",
                stack.experts, dims.n_expert
            )));
        }
        if stack.rows != rows || stack.k != k {
            return Err(refuse(format!(
                "the {name} stack holds {} rows of {} values, the dims need {rows} of {k}",
                stack.rows, stack.k
            )));
        }
    }
    let projections = [
        (
            "shared gate",
            weights.shared_gate,
            dims.shared_inter,
            dims.hidden,
        ),
        (
            "shared up",
            weights.shared_up,
            dims.shared_inter,
            dims.hidden,
        ),
        (
            "shared down",
            weights.shared_down,
            dims.hidden,
            dims.shared_inter,
        ),
    ];
    for (name, proj, rows, k) in projections {
        if proj.rows != rows || proj.k != k {
            return Err(refuse(format!(
                "the {name} weight holds {} rows of {} values, the dims need {rows} of {k}",
                proj.rows, proj.k
            )));
        }
    }
    Ok(())
}

fn check_sizes(need: &NeedBytes, bufs: &BufferFacts) -> Result<(), EngError> {
    let (route, expert) = (&bufs.route, &bufs.expert);
    let ranges: [(&str, &RangeFacts, usize); 15] = [
        ("resid_in", &route.resid_in, need.resid),
        ("resid_out", &route.resid_out, need.resid),
        ("norm_row", &route.norm_row, need.norm_row),
        ("logits", &route.logits, need.logits),
        ("ids", &route.ids, need.ids),
        ("weights", &route.weights, need.picks),
        ("shared_gate", &route.shared_gate, need.gate),
        ("status", &route.status, need.status),
        ("xf_moe", &expert.xf_moe, need.xf_moe),
        ("expert_gu", &expert.expert_gu, need.expert_gu),
        ("xf_down", &expert.xf_down, need.xf_down),
        ("xf_shared", &expert.xf_shared, need.xf_shared),
        ("expert_down", &expert.expert_down, need.expert_down),
        ("shared_gu", &expert.shared_gu, need.shared_gu),
        ("shared_out", &expert.shared_out, need.shared_out),
    ];
    for (name, got, want) in ranges {
        if got.bytes != want {
            return Err(refuse(format!(
                "the {name} range holds {} bytes, the block needs {want}",
                got.bytes
            )));
        }
    }
    Ok(())
}

/// True when two fact ranges share a byte: one allocation and intersecting spans.
fn overlaps(a: &RangeFacts, b: &RangeFacts) -> bool {
    a.buffer == b.buffer
        && a.offset < b.offset.saturating_add(b.bytes)
        && b.offset < a.offset.saturating_add(a.bytes)
}

/// `BadRequest` when the range the block writes and another range of the same call share a byte.
fn apart(written: (&str, &RangeFacts), other: (&str, &RangeFacts)) -> Result<(), EngError> {
    if overlaps(written.1, other.1) {
        return Err(refuse(format!(
            "the {} and {} ranges overlap",
            written.0, other.0
        )));
    }
    Ok(())
}

fn check_apart(bufs: &BufferFacts) -> Result<(), EngError> {
    let route = &bufs.route;
    let read = ("norm_row", &route.norm_row);
    let outputs = [
        ("logits", &route.logits),
        ("ids", &route.ids),
        ("weights", &route.weights),
        ("shared_gate", &route.shared_gate),
        ("status", &route.status),
    ];
    for (index, output) in outputs.iter().enumerate() {
        apart(*output, read)?;
        for other in &outputs[index + 1..] {
            apart(*output, *other)?;
        }
    }
    let written = ("resid_out", &route.resid_out);
    let combine_reads = [
        ("resid_in", &route.resid_in),
        ("expert_down", &bufs.expert.expert_down),
        ("weights", &route.weights),
        ("shared_gate", &route.shared_gate),
        ("shared_out", &bufs.expert.shared_out),
    ];
    for input in combine_reads {
        apart(written, input)?;
    }
    Ok(())
}

/// The weights of the MoE block of one layer, resolved by the layer driver (RD-9b's
/// `(layer, role)` lookup): the raw F32 router matrices and the tiled expert stacks and
/// shared-expert projections.
pub(super) struct MoeLayerWeights<'w> {
    /// The raw F32 `[n_expert][hidden]` router matrix (`ffn_gate_inp`).
    pub(super) router: &'w DeviceBuffer,
    /// The raw F32 `[hidden]` shared-gate vector (`ffn_gate_inp_shexp`).
    pub(super) router_shared: &'w DeviceBuffer,
    /// The gate stack of all routed experts.
    pub(super) gate: ExpertStack<'w>,
    /// The up stack of all routed experts.
    pub(super) up: ExpertStack<'w>,
    /// The down stack of all routed experts.
    pub(super) down: ExpertStack<'w>,
    /// The shared expert's gate projection.
    pub(super) shared_gate: &'w TiledWeight,
    /// The shared expert's up projection.
    pub(super) shared_up: &'w TiledWeight,
    /// The shared expert's down projection.
    pub(super) shared_down: &'w TiledWeight,
}

/// The routing and residual ranges of the block: views of the RD-9a arena and the status word.
pub(super) struct RouteBufs<'b> {
    /// f32 `[hidden]`: the residual after the mixer add, the block input.
    pub(super) resid_in: DeviceRange<'b>,
    /// f32 `[hidden]`: the other residual buffer, the block output.
    pub(super) resid_out: DeviceRange<'b>,
    /// f32 `[hidden]`: the normalized row, the router input.
    pub(super) norm_row: DeviceRange<'b>,
    /// The router's logits scratch (`RouterLogits`).
    pub(super) logits: DeviceRange<'b>,
    /// i32 `[top_k]`: the pick ids, read on the device only.
    pub(super) ids: DeviceRange<'b>,
    /// f32 `[top_k]`: the renormalized pick weights.
    pub(super) weights: DeviceRange<'b>,
    /// f32: the shared gate, sigmoid-ed by the router.
    pub(super) shared_gate: DeviceRange<'b>,
    /// The status word the router ORs its fault bit into.
    pub(super) status: DeviceRange<'b>,
}

/// The expert and shared-expert ranges of the block: views of the RD-9a arena.
pub(super) struct ExpertBufs<'b> {
    /// The fp16 pack of the normalized row, every gate and up matmul's input.
    pub(super) xf_moe: DeviceRange<'b>,
    /// f32 `[top_k][gate | up]`, slot `s`'s halves contiguous for its pack.
    pub(super) expert_gu: DeviceRange<'b>,
    /// The fp16 pack one slot's down matmul reads.
    pub(super) xf_down: DeviceRange<'b>,
    /// f32 `[top_k][hidden]` in pick order, the combine's expert term.
    pub(super) expert_down: DeviceRange<'b>,
    /// f32 `[gate | up]` of the shared expert.
    pub(super) shared_gu: DeviceRange<'b>,
    /// The fp16 pack the shared down matmul reads.
    pub(super) xf_shared: DeviceRange<'b>,
    /// f32 `[hidden]`, the shared expert's down output.
    pub(super) shared_out: DeviceRange<'b>,
}

/// The fifteen device ranges of the block, grouped so no function takes more than four arguments.
pub(super) struct MoeBuffers<'b> {
    /// Routing and residual ranges.
    pub(super) route: RouteBufs<'b>,
    /// Expert and shared-expert ranges.
    pub(super) expert: ExpertBufs<'b>,
}

fn facts_of(range: &DeviceRange<'_>) -> RangeFacts {
    RangeFacts {
        offset: range.offset(),
        bytes: range.bytes(),
        buffer: std::ptr::from_ref(range.buffer()) as usize,
    }
}

fn buffer_facts(buffer: &DeviceBuffer) -> RangeFacts {
    RangeFacts {
        offset: 0,
        bytes: buffer.bytes(),
        buffer: std::ptr::from_ref(buffer) as usize,
    }
}

impl MoeLayerWeights<'_> {
    /// The pure facts of the weight set, for [`check_block`] without a device.
    pub(super) fn facts(&self) -> WeightFacts {
        let stack = |s: &ExpertStack<'_>| StackFacts {
            rows: s.weight.n_rows(),
            k: s.weight.k(),
            experts: s.experts,
        };
        let proj = |w: &TiledWeight| ProjFacts {
            rows: w.n_rows(),
            k: w.k(),
        };
        WeightFacts {
            router: buffer_facts(self.router),
            router_shared: buffer_facts(self.router_shared),
            gate: stack(&self.gate),
            up: stack(&self.up),
            down: stack(&self.down),
            shared_gate: proj(self.shared_gate),
            shared_up: proj(self.shared_up),
            shared_down: proj(self.shared_down),
        }
    }
}

impl MoeBuffers<'_> {
    /// The pure facts of the fifteen ranges, for [`check_block`] without a device.
    pub(super) fn facts(&self) -> BufferFacts {
        let route = &self.route;
        let expert = &self.expert;
        BufferFacts {
            route: RouteFacts {
                resid_in: facts_of(&route.resid_in),
                resid_out: facts_of(&route.resid_out),
                norm_row: facts_of(&route.norm_row),
                logits: facts_of(&route.logits),
                ids: facts_of(&route.ids),
                weights: facts_of(&route.weights),
                shared_gate: facts_of(&route.shared_gate),
                status: facts_of(&route.status),
            },
            expert: ExpertFacts {
                xf_moe: facts_of(&expert.xf_moe),
                expert_gu: facts_of(&expert.expert_gu),
                xf_down: facts_of(&expert.xf_down),
                expert_down: facts_of(&expert.expert_down),
                shared_gu: facts_of(&expert.shared_gu),
                xf_shared: facts_of(&expert.xf_shared),
                shared_out: facts_of(&expert.shared_out),
            },
        }
    }
}

/// The context and the one owned stream the block queues on. Grouping them keeps every function
/// of this file at four arguments or fewer (AGENTS.md).
pub(super) struct Launch<'a> {
    /// The adapter that owns every buffer of the block.
    pub(super) ctx: &'a CudaContext,
    /// The one owned stream: every op queues on it, and none of them makes the host wait.
    pub(super) stream: &'a DeviceStream,
}

/// The block dims back from the facts [`check_block`] pins exactly, so that the launcher carries
/// no extra parameter. `rows` is one: the block refuses every other row count.
fn dims_of(weights: &MoeLayerWeights<'_>, bufs: &MoeBuffers<'_>) -> MoeDims {
    MoeDims {
        rows: 1,
        hidden: bufs.route.resid_in.bytes() / F32,
        n_expert: weights.gate.experts,
        top_k: bufs.route.weights.bytes() / F32,
        inter: weights.gate.weight.n_rows() / weights.gate.experts.max(1),
        shared_inter: weights.shared_gate.n_rows(),
    }
}

/// The stack by value: [`ExpertCall`] takes it by value and it is two `Copy` fields.
fn stack_of<'a>(stack: &ExpertStack<'a>) -> ExpertStack<'a> {
    ExpertStack {
        weight: stack.weight,
        experts: stack.experts,
    }
}

/// The whole allocation behind a weight buffer, as the router's weight range.
fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The `bytes` bytes of `range` from byte `at` on: a sub-range of the same allocation.
fn slice<'b>(
    range: &DeviceRange<'b>,
    at: usize,
    bytes: usize,
) -> Result<DeviceRange<'b>, EngError> {
    let start = range
        .offset()
        .checked_add(at)
        .ok_or_else(|| refuse("a sub-range offset overflows".to_owned()))?;
    range.buffer().range(start, bytes)
}

/// The `index`th row of `width` bytes of `range`, a sub-range of the same allocation.
fn row_at<'b>(
    range: &DeviceRange<'b>,
    width: usize,
    index: usize,
) -> Result<DeviceRange<'b>, EngError> {
    slice(range, index * width, width)
}

/// The router of `op` ([`MoeOp::Router`]): ids, pick weights and the shared gate on the device.
fn op_router(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    dims: MoeDims,
) -> Result<(), EngError> {
    let shape = RouterShape {
        m: dims.rows,
        hidden: dims.hidden,
        n_expert: dims.n_expert,
        top_k: dims.top_k,
    };
    launch.ctx.router(
        launch.stream,
        RouterCall {
            shape,
            x: bufs.route.norm_row,
            weights: RouterWeights {
                w: whole(weights.router)?,
                w_shared: Some(whole(weights.router_shared)?),
            },
            logits: bufs.route.logits,
            out: RouterOut {
                ids: bufs.route.ids,
                weights: bufs.route.weights,
                shared_gate: Some(bufs.route.shared_gate),
                status: bufs.route.status,
            },
        },
    )
}

/// The expert matmul of `op` ([`MoeOp::ExpertGate`], [`MoeOp::ExpertUp`] or [`MoeOp::ExpertDown`]):
/// one `matmul_expert` against the id the kernel reads on the device.
fn op_expert(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    op: MoeOp,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    let width = dims.inter * F32;
    let (slot, stack, xf, y) = match op {
        MoeOp::ExpertGate(slot) => (
            slot,
            &weights.gate,
            bufs.expert.xf_moe,
            row_at(&bufs.expert.expert_gu, width, 2 * slot)?,
        ),
        MoeOp::ExpertUp(slot) => (
            slot,
            &weights.up,
            bufs.expert.xf_moe,
            row_at(&bufs.expert.expert_gu, width, 2 * slot + 1)?,
        ),
        MoeOp::ExpertDown(slot) => (
            slot,
            &weights.down,
            bufs.expert.xf_down,
            row_at(&bufs.expert.expert_down, dims.hidden * F32, slot)?,
        ),
        _ => {
            return Err(refuse(
                "the expert matmul takes a pick of the block".to_owned(),
            ));
        }
    };
    launch.ctx.matmul_expert(
        launch.stream,
        ExpertCall {
            stack: stack_of(stack),
            select: ExpertSelect {
                ids: bufs.route.ids,
                slot,
                count: dims.n_expert,
            },
            io: MatmulIo {
                xf,
                y,
                m: dims.rows,
            },
        },
    )
}

/// The dense matmul of `op` ([`MoeOp::SharedGate`], [`MoeOp::SharedUp`] or [`MoeOp::SharedDown`]):
/// one shared-expert projection, one row.
fn op_shared(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    op: MoeOp,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    let width = dims.shared_inter * F32;
    let (weight, xf, y) = match op {
        MoeOp::SharedGate => (
            weights.shared_gate,
            bufs.expert.xf_moe,
            row_at(&bufs.expert.shared_gu, width, 0)?,
        ),
        MoeOp::SharedUp => (
            weights.shared_up,
            bufs.expert.xf_moe,
            row_at(&bufs.expert.shared_gu, width, 1)?,
        ),
        MoeOp::SharedDown => (
            weights.shared_down,
            bufs.expert.xf_shared,
            bufs.expert.shared_out,
        ),
        _ => return Err(refuse("the shared matmul takes a shared op".to_owned())),
    };
    launch.ctx.matmul(
        launch.stream,
        Projection {
            weight,
            first_row: 0,
            rows: weight.n_rows(),
        },
        MatmulIo {
            xf,
            y,
            m: dims.rows,
        },
    )
}

/// The SwiGLU pack of `op` ([`MoeOp::SwigluShared`] or [`MoeOp::SwigluExpert`]): the shared
/// expert into `XfShared`, or one pick into the one `XfDown`.
fn op_pack(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    op: MoeOp,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    let (gu, xf, n_ff) = match op {
        MoeOp::SwigluShared => (
            bufs.expert.shared_gu,
            bufs.expert.xf_shared,
            dims.shared_inter,
        ),
        MoeOp::SwigluExpert(slot) => (
            slice(
                &bufs.expert.expert_gu,
                slot * 2 * dims.inter * F32,
                2 * dims.inter * F32,
            )?,
            bufs.expert.xf_down,
            dims.inter,
        ),
        _ => return Err(refuse("the pack takes a SwiGLU op".to_owned())),
    };
    launch.ctx.swiglu_pack(
        launch.stream,
        SwigluPackCall {
            gu,
            xf,
            m: dims.rows,
            n_ff,
        },
    )
}

/// The combine: `resid_out = resid_in + (picks in pick order + gate * shared)` out of place
/// (RD-4e, `CombineOut::To`), the weights and the gate exactly as the router wrote them.
fn op_combine(launch: &Launch<'_>, bufs: &MoeBuffers<'_>, dims: MoeDims) -> Result<(), EngError> {
    launch.ctx.moe_combine(
        launch.stream,
        CombineCall {
            shape: CombineShape {
                m: dims.rows,
                n: dims.hidden,
                top_k: dims.top_k,
                ld_down: dims.hidden,
            },
            res: bufs.route.resid_in,
            down: bufs.expert.expert_down,
            weights: bufs.route.weights,
            shared: Some(SharedTerm {
                gate: bufs.route.shared_gate,
                row: bufs.expert.shared_out,
            }),
            out: CombineOut::To(bufs.route.resid_out),
        },
    )
}

/// One op of the sequence as one wrapper call. The match is the op table: one arm per variant,
/// each arm one call (the arm count is the op count of [`MoeOp`], inspected and kept as one match
/// on purpose: a table of closures would hide the arguments of every call).
fn launch_op(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    op: MoeOp,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    match op {
        MoeOp::Router => op_router(launch, weights, bufs, dims),
        MoeOp::ExpertGate(_) => op_expert(launch, weights, bufs, op),
        MoeOp::ExpertUp(_) => op_expert(launch, weights, bufs, op),
        MoeOp::ExpertDown(_) => op_expert(launch, weights, bufs, op),
        MoeOp::SharedGate => op_shared(launch, weights, bufs, op),
        MoeOp::SharedUp => op_shared(launch, weights, bufs, op),
        MoeOp::SwigluShared => op_pack(launch, weights, bufs, op),
        MoeOp::SwigluExpert(_) => op_pack(launch, weights, bufs, op),
        MoeOp::SharedDown => op_shared(launch, weights, bufs, op),
        MoeOp::Combine => op_combine(launch, bufs, dims),
    }
}

/// The MoE block of one layer at one token on the one stream: every pure refusal first
/// ([`check_block`], so a captured block cannot be half queued), then the fixed list of
/// [`moe_sequence`], one wrapper call per op. The function makes no allocation, copies nothing
/// and never syncs; eager and captured steps call the same one, and no host value depends on
/// device data (RD-0b).
pub(super) fn run_moe_block(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    check_block(dims, &weights.facts(), &bufs.facts())?;
    for op in moe_sequence(dims.top_k) {
        launch_op(launch, weights, bufs, op)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
