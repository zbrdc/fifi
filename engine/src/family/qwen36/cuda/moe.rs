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
//! MATVEC-PAR S3b: the Native layout replaces the sixteen per-slot gate/up matmuls and the
//! sixteen per-slot packs and down matmuls with ONE merged launch each (`matvec_expert` over the
//! device ids, `swiglu_q8` over the eight slots), reading the padded q8_1 packs of the NATIVE
//! plan; the router keeps its place and its bytes (the OFF-T hook reads them right after it).
//! [`Packs`] carries the two views of the input packs and [`check_block`] sizes them per kind.
//!
//! Design guidance only, no code copied. The device-side expert routing this block relies on is
//! RD-5's (credited there from MegaCapybara v1.35 static RE, section 6.1 C9).

use super::dims::StepDims;
use super::mixer_weights::DenseWeight;
use super::step_plan::{Phase, Slot};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::resident::matmul_expert::{ExpertCall, ExpertSelect, ExpertStack};
use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
use crate::kernels::cuda::resident::matvec_q::MatvecCall;
use crate::kernels::cuda::resident::moe_combine::{
    CombineCall, CombineOut, CombineShape, SharedTerm,
};
use crate::kernels::cuda::resident::router::{
    RouterCall, RouterOut, RouterShape, RouterWeights, router_logits_bytes,
};
use crate::kernels::cuda::resident::swiglu_pack::SwigluPackCall;
use crate::kernels::cuda::resident::swiglu_q8::SwigluQ8Call;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream, MatmulIo, NativeWeight, Projection,
    TiledWeight, resident::quantize_q8_1::q8_1_bytes, xf_bytes,
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
    /// The ONE merged gate matvec of every pick slot against `Q8Moe` (`matvec_expert`,
    /// `x_per_slot` false, the ids on the device; MATVEC-PAR S3b).
    ExpertGateQ,
    /// The ONE merged up matmul of every pick slot (the twin of [`MoeOp::ExpertGateQ`]).
    ExpertUpQ,
    /// The shared expert's gate matvec against `Q8Moe` (`matvec_q`).
    SharedGateQ,
    /// The shared expert's up matvec against `Q8Moe`.
    SharedUpQ,
    /// The shared expert's SwiGLU and q8_1 quantize (`swiglu_q8`, one slot) into `Q8Shared`.
    SwigluSharedQ,
    /// The batched SwiGLU and q8_1 quantize of the eight slots into `Q8Down` (`swiglu_q8`).
    SwigluExperts,
    /// The ONE merged down matvec of every pick slot (`matvec_expert`, `x_per_slot` true, the
    /// per-slot rows of `Q8Down`).
    ExpertDownQ,
    /// The shared expert's down matvec against `Q8Shared` (`matvec_q`).
    SharedDownQ,
}

/// The documented order of 4.2 as the launch list: router; every slot's gate and up; the shared
/// gate and up; the shared pack; per slot the pack then the down matmul; the shared down; the
/// combine. `top_k` slots make `3 * top_k + 14` ops.
pub(super) fn moe_sequence(top_k: usize, layout: WeightLayout) -> Vec<MoeOp> {
    if layout == WeightLayout::Native {
        // The router keeps the first place and its bytes (the OFF-T hook reads them there); the
        // per-slot triples collapse to one merged launch each, so a Native block is 10 ops.
        return vec![
            MoeOp::Router,
            MoeOp::ExpertGateQ,
            MoeOp::ExpertUpQ,
            MoeOp::SharedGateQ,
            MoeOp::SharedUpQ,
            MoeOp::SwigluSharedQ,
            MoeOp::SwigluExperts,
            MoeOp::ExpertDownQ,
            MoeOp::SharedDownQ,
            MoeOp::Combine,
        ];
    }
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
        MoeOp::ExpertGateQ | MoeOp::ExpertUpQ => OpSlots {
            reads: &[Slot::Q8Moe, Slot::TopkId],
            writes: &[Slot::ExpertGu],
            phase: Phase::GateUp as usize,
        },
        MoeOp::SharedGateQ | MoeOp::SharedUpQ => OpSlots {
            reads: &[Slot::Q8Moe],
            writes: &[Slot::SharedGu],
            phase: Phase::GateUp as usize,
        },
        MoeOp::SwigluSharedQ => OpSlots {
            reads: &[Slot::SharedGu],
            writes: &[Slot::Q8Shared],
            phase: Phase::Swiglu as usize,
        },
        MoeOp::SwigluExperts => OpSlots {
            reads: &[Slot::ExpertGu],
            writes: &[Slot::Q8Down],
            phase: Phase::Swiglu as usize,
        },
        MoeOp::ExpertDownQ => OpSlots {
            reads: &[Slot::Q8Down, Slot::TopkId],
            writes: &[Slot::ExpertDown],
            phase: Phase::Down as usize,
        },
        MoeOp::SharedDownQ => OpSlots {
            reads: &[Slot::Q8Shared],
            writes: &[Slot::SharedOut],
            phase: Phase::Down as usize,
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
    /// The input packs of the block (`xf_moe`/`xf_down`/`xf_shared` of the Tiled block, the
    /// q8_1 packs `Q8Moe`/`Q8Down`/`Q8Shared` of the Native block), with their kind.
    pub(super) packs: PackFacts,
    /// f32 `[top_k][gate | up]`, slot `s`'s halves contiguous for its pack.
    pub(super) expert_gu: RangeFacts,
    /// f32 `[top_k][hidden]` in pick order, the combine's expert term.
    pub(super) expert_down: RangeFacts,
    /// f32 `[gate | up]` of the shared expert.
    pub(super) shared_gu: RangeFacts,
    /// f32 `[hidden]`, the shared expert's down output.
    pub(super) shared_out: RangeFacts,
}

/// The three input-pack views of one layout (MATVEC-PAR S3b): the fp16 packs of the Tiled
/// block or the padded q8_1 packs of the Native block. The wrong-kind accessor is a typed
/// refusal, so a Native list cannot launch against Tiled packs or the other way round.
#[derive(Clone, Copy)]
pub(super) enum Packs<'b> {
    Xf {
        moe: DeviceRange<'b>,
        down: DeviceRange<'b>,
        shared: DeviceRange<'b>,
    },
    Q8 {
        moe: DeviceRange<'b>,
        down: DeviceRange<'b>,
        shared: DeviceRange<'b>,
    },
}

/// The kind of a pack view, for the pure size checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PackKind {
    Xf,
    Q8,
}

impl PackKind {
    /// The step layout the pack kind serves.
    pub(super) fn layout(self) -> WeightLayout {
        match self {
            Self::Xf => WeightLayout::Tiled,
            Self::Q8 => WeightLayout::Native,
        }
    }
}

impl Packs<'_> {
    pub(super) fn kind(&self) -> PackKind {
        match self {
            Self::Xf { .. } => PackKind::Xf,
            Self::Q8 { .. } => PackKind::Q8,
        }
    }

    /// The fp16 pack triple of the Tiled block.
    pub(super) fn xf(
        &self,
    ) -> Result<(DeviceRange<'_>, DeviceRange<'_>, DeviceRange<'_>), EngError> {
        match self {
            Self::Xf { moe, down, shared } => Ok((*moe, *down, *shared)),
            Self::Q8 { .. } => Err(refuse(
                "the block holds q8_1 packs, the tiled launch reads fp16 packs".to_owned(),
            )),
        }
    }

    /// The padded q8_1 pack triple of the Native block.
    pub(super) fn q8(
        &self,
    ) -> Result<(DeviceRange<'_>, DeviceRange<'_>, DeviceRange<'_>), EngError> {
        match self {
            Self::Q8 { moe, down, shared } => Ok((*moe, *down, *shared)),
            Self::Xf { .. } => Err(refuse(
                "the block holds fp16 packs, the native launch reads q8_1 packs".to_owned(),
            )),
        }
    }
}

/// The pack views as plain facts: the kind beside the three ranges.
pub(super) struct PackFacts {
    pub(super) kind: PackKind,
    pub(super) moe: RangeFacts,
    pub(super) down: RangeFacts,
    pub(super) shared: RangeFacts,
}

/// The fifteen block buffers as plain facts.
pub(super) struct BufferFacts {
    /// Routing and residual ranges.
    pub(super) route: RouteFacts,
    /// Expert and shared-expert ranges.
    pub(super) expert: ExpertFacts,
}

/// The bytes each range of the block must hold, from the dims and the pack kind alone.
struct NeedBytes {
    resid: usize,
    norm_row: usize,
    logits: usize,
    picks: usize,
    ids: usize,
    gate: usize,
    status: usize,
    expert_gu: usize,
    pack_moe: usize,
    pack_down: usize,
    pack_shared: usize,
    expert_down: usize,
    shared_gu: usize,
    shared_out: usize,
}

impl NeedBytes {
    fn of(dims: &MoeDims, packs: PackKind) -> Result<Self, EngError> {
        let row = |width: usize| product(&[width, F32]);
        // The fp16 packs of the Tiled block (one per row) or the padded q8_1 packs of the Native
        // block (one row for moe/shared, `top_k` rows for the expert slots; MATVEC-PAR S3b).
        let pack = |k: usize, rows: usize| {
            match packs {
                PackKind::Xf => xf_bytes(dims.rows, k),
                PackKind::Q8 => q8_1_bytes(k, rows),
            }
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
            expert_gu: product(&[dims.top_k, dims.inter, 2, F32])?,
            pack_moe: pack(dims.hidden, 1)?,
            pack_down: pack(dims.inter, dims.top_k)?,
            pack_shared: pack(dims.shared_inter, 1)?,
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
    let need = NeedBytes::of(dims, bufs.expert.packs.kind)?;
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
        ("xf_moe", &expert.packs.moe, need.pack_moe),
        ("expert_gu", &expert.expert_gu, need.expert_gu),
        ("xf_down", &expert.packs.down, need.pack_down),
        ("xf_shared", &expert.packs.shared, need.pack_shared),
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

/// One expert stack of the block (MATVEC-PAR S3b): the tiles of the Tiled load (the per-slot
/// `matmul_expert` launches) or the file's own blocks of the Native load (the ONE merged
/// `matvec_expert` launch; the depth rides `NativeWeight::experts`).
pub(super) enum StackWeights<'w> {
    Tiled(ExpertStack<'w>),
    Native(&'w NativeWeight),
}

/// The weights of the MoE block of one layer, resolved by the layer driver (RD-9b's
/// `(layer, role)` lookup): the raw F32 router matrices and the expert stacks and
/// shared-expert projections of the session's layout (tiles or native blocks).
pub(super) struct MoeLayerWeights<'w> {
    /// The raw F32 `[n_expert][hidden]` router matrix (`ffn_gate_inp`).
    pub(super) router: &'w DeviceBuffer,
    /// The raw F32 `[hidden]` shared-gate vector (`ffn_gate_inp_shexp`).
    pub(super) router_shared: &'w DeviceBuffer,
    /// The gate stack of all routed experts.
    pub(super) gate: StackWeights<'w>,
    /// The up stack of all routed experts.
    pub(super) up: StackWeights<'w>,
    /// The down stack of all routed experts.
    pub(super) down: StackWeights<'w>,
    /// The shared expert's gate projection.
    pub(super) shared_gate: DenseWeight<'w>,
    /// The shared expert's up projection.
    pub(super) shared_up: DenseWeight<'w>,
    /// The shared expert's down projection.
    pub(super) shared_down: DenseWeight<'w>,
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

/// The expert and shared-expert ranges of the block: views of the step arena, the input packs
/// grouped in [`Packs`] (MATVEC-PAR S3b).
pub(super) struct ExpertBufs<'b> {
    pub(super) packs: Packs<'b>,
    /// f32 `[top_k][gate | up]`, slot `s`'s halves contiguous for its pack.
    pub(super) expert_gu: DeviceRange<'b>,
    /// f32 `[top_k][hidden]` in pick order, the combine's expert term.
    pub(super) expert_down: DeviceRange<'b>,
    /// f32 `[gate | up]` of the shared expert.
    pub(super) shared_gu: DeviceRange<'b>,
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

fn pack_facts(packs: &Packs<'_>) -> PackFacts {
    let (kind, moe, down, shared) = match packs {
        Packs::Xf { moe, down, shared } => (PackKind::Xf, moe, down, shared),
        Packs::Q8 { moe, down, shared } => (PackKind::Q8, moe, down, shared),
    };
    PackFacts {
        kind,
        moe: facts_of(moe),
        down: facts_of(down),
        shared: facts_of(shared),
    }
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

impl StackWeights<'_> {
    /// The tiles of the Tiled load for the per-slot launches (a Native stack refuses).
    fn tiled(&self) -> Result<&ExpertStack<'_>, EngError> {
        match self {
            Self::Tiled(stack) => Ok(stack),
            Self::Native(_) => Err(refuse(
                "the stack is loaded native; the per-slot launch reads tiles".to_owned(),
            )),
        }
    }

    /// The native blocks of the Native load for the merged launches.
    fn native(&self) -> Result<&NativeWeight, EngError> {
        match self {
            Self::Native(weight) => Ok(weight),
            Self::Tiled(_) => Err(refuse(
                "the stack is loaded tiled; the merged launch reads native blocks".to_owned(),
            )),
        }
    }
}

impl MoeLayerWeights<'_> {
    /// The pure facts of the weight set, for [`check_block`] without a device: the layout of a
    /// stack or projection does not change its shape facts (MATVEC-PAR S3b).
    pub(super) fn facts(&self) -> WeightFacts {
        let stack = |s: &StackWeights<'_>| StackFacts {
            rows: match s {
                StackWeights::Tiled(t) => t.weight.n_rows(),
                StackWeights::Native(n) => n.n_rows(),
            },
            k: match s {
                StackWeights::Tiled(t) => t.weight.k(),
                StackWeights::Native(n) => n.k(),
            },
            experts: match s {
                StackWeights::Tiled(t) => t.experts,
                StackWeights::Native(n) => n.experts().unwrap_or(0),
            },
        };
        let proj = |w: &DenseWeight<'_>| ProjFacts {
            rows: match w {
                DenseWeight::Tiled(t) => t.n_rows(),
                DenseWeight::Native(n) => n.n_rows(),
            },
            k: match w {
                DenseWeight::Tiled(t) => t.k(),
                DenseWeight::Native(n) => n.k(),
            },
        };
        WeightFacts {
            router: buffer_facts(self.router),
            router_shared: buffer_facts(self.router_shared),
            gate: stack(&self.gate),
            up: stack(&self.up),
            down: stack(&self.down),
            shared_gate: proj(&self.shared_gate),
            shared_up: proj(&self.shared_up),
            shared_down: proj(&self.shared_down),
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
                packs: pack_facts(&expert.packs),
                expert_gu: facts_of(&expert.expert_gu),
                expert_down: facts_of(&expert.expert_down),
                shared_gu: facts_of(&expert.shared_gu),
                shared_out: facts_of(&expert.shared_out),
            },
        }
    }
}

/// The context and the one owned stream the block queues on. Grouping them keeps every function
/// of this file at four arguments or fewer (the project rules).
pub(super) struct Launch<'a> {
    /// The adapter that owns every buffer of the block.
    pub(super) ctx: &'a CudaContext,
    /// The one owned stream: every op queues on it, and none of them makes the host wait.
    pub(super) stream: &'a DeviceStream,
}

/// The block dims back from the facts [`check_block`] pins exactly, so that the launcher carries
/// no extra parameter. `rows` is one: the block refuses every other row count.
fn dims_of(weights: &MoeLayerWeights<'_>, bufs: &MoeBuffers<'_>) -> MoeDims {
    let gate = weights.facts().gate;
    MoeDims {
        rows: 1,
        hidden: bufs.route.resid_in.bytes() / F32,
        n_expert: gate.experts,
        top_k: bufs.route.weights.bytes() / F32,
        inter: gate.rows / gate.experts.max(1),
        shared_inter: weights.facts().shared_gate.rows,
    }
}

/// The stack by value: [`ExpertCall`] takes it by value and it is two `Copy` fields.
fn stack_of<'a>(stack: &ExpertStack<'a>) -> ExpertStack<'a> {
    ExpertStack {
        weight: stack.weight,
        experts: stack.experts,
    }
}

/// The whole native stack buffer as the merged launch's weight range.
fn native_stack(weight: &NativeWeight) -> Result<DeviceRange<'_>, EngError> {
    weight.buffer().range(0, weight.buffer().bytes())
}

/// The tiles of a shared-expert projection for a tiled launch (a native entry refuses by name of
/// the projection's role at the call site).
fn shared_tiled<'w>(w: &DenseWeight<'w>) -> Result<&'w TiledWeight, EngError> {
    match w {
        DenseWeight::Tiled(weight) => Ok(weight),
        DenseWeight::Native(_) => Err(refuse(
            "the shared projection is loaded native; the tiled matmul reads tiles".to_owned(),
        )),
    }
}

/// The native blocks of a shared-expert projection for the merged launch (the mirror refusal).
fn shared_native<'w>(w: &DenseWeight<'w>) -> Result<&'w NativeWeight, EngError> {
    match w {
        DenseWeight::Native(weight) => Ok(weight),
        DenseWeight::Tiled(_) => Err(refuse(
            "the shared projection is loaded tiled; the native matvec reads blocks".to_owned(),
        )),
    }
}

/// The whole allocation behind a weight buffer, as the router's weight range.
fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

/// The whole range of an arena view (its allocation holds more; the view is exact).
fn whole_of<'b>(range: &DeviceRange<'b>) -> Result<DeviceRange<'b>, EngError> {
    range.buffer().range(range.offset(), range.bytes())
}

/// The tail of `range` from f32 column `col` (the A2 halves of `ExpertGu`/`SharedGu`): the tail
/// ends at the end of the view, so the merged launch's `(n_used - 1) * stride + rows` footprint
/// fits it exactly.
fn tail<'b>(range: &DeviceRange<'b>, col: usize) -> Result<DeviceRange<'b>, EngError> {
    let offset = col
        .checked_mul(F32)
        .filter(|offset| *offset < range.bytes())
        .ok_or_else(|| refuse("the slot half lies outside its range".to_owned()))?;
    range
        .buffer()
        .range(range.offset() + offset, range.bytes() - offset)
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
    let (moe, down, _) = bufs.expert.packs.xf()?;
    let (slot, stack, xf, y) = match op {
        MoeOp::ExpertGate(slot) => (
            slot,
            weights.gate.tiled()?,
            moe,
            row_at(&bufs.expert.expert_gu, width, 2 * slot)?,
        ),
        MoeOp::ExpertUp(slot) => (
            slot,
            weights.up.tiled()?,
            moe,
            row_at(&bufs.expert.expert_gu, width, 2 * slot + 1)?,
        ),
        MoeOp::ExpertDown(slot) => (
            slot,
            weights.down.tiled()?,
            down,
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
    let (moe, _, shared) = bufs.expert.packs.xf()?;
    let (weight, xf, y) = match op {
        MoeOp::SharedGate => (
            shared_tiled(&weights.shared_gate)?,
            moe,
            row_at(&bufs.expert.shared_gu, width, 0)?,
        ),
        MoeOp::SharedUp => (
            shared_tiled(&weights.shared_up)?,
            moe,
            row_at(&bufs.expert.shared_gu, width, 1)?,
        ),
        MoeOp::SharedDown => (
            shared_tiled(&weights.shared_down)?,
            shared,
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
    let (_, down, shared) = bufs.expert.packs.xf()?;
    let (gu, xf, n_ff) = match op {
        MoeOp::SwigluShared => (bufs.expert.shared_gu, shared, dims.shared_inter),
        MoeOp::SwigluExpert(slot) => (
            slice(
                &bufs.expert.expert_gu,
                slot * 2 * dims.inter * F32,
                2 * dims.inter * F32,
            )?,
            down,
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

/// The ONE merged gate/up matvec of the native block ([`MoeOp::ExpertGateQ`],
/// [`MoeOp::ExpertUpQ`]): a single `matvec_expert` launch over the device ids, every slot reading
/// the same `Q8Moe` row (`x_per_slot` false), writing slot `s`'s `rows` floats at column offset
/// 0 (gate) or `inter` (up) of `ExpertGu` with the A2 stride `2 * inter`.
fn op_expert_gu(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    op: MoeOp,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    let (stack, offset) = match op {
        MoeOp::ExpertGateQ => (&weights.gate, 0),
        MoeOp::ExpertUpQ => (&weights.up, dims.inter),
        _ => return Err(refuse("the merged gate/up launch takes a Q op".to_owned())),
    };
    let native = stack.native()?;
    let (moe, _, _) = bufs.expert.packs.q8()?;
    launch.ctx.matvec_expert(
        launch.stream,
        ExpertMatvecCall {
            ty: native.quant(),
            w_stack: native_stack(native)?,
            ids: bufs.route.ids,
            x_q8: moe,
            x_per_slot: false,
            y: tail(&bufs.expert.expert_gu, offset)?,
            y_slot_stride: 2 * dims.inter,
            k: dims.hidden,
            rows: dims.inter,
            n_expert: dims.n_expert,
            n_used: dims.top_k,
        },
    )
}

/// The ONE merged down matvec of the native block ([`MoeOp::ExpertDownQ`]): `matvec_expert` with
/// `x_per_slot` true (slot `s` reads `Q8Down`'s own padded row), writing slot `s`'s `hidden`
/// floats of `ExpertDown` with the stride `hidden`.
fn op_expert_down(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    let native = weights.down.native()?;
    let (_, down, _) = bufs.expert.packs.q8()?;
    launch.ctx.matvec_expert(
        launch.stream,
        ExpertMatvecCall {
            ty: native.quant(),
            w_stack: native_stack(native)?,
            ids: bufs.route.ids,
            x_q8: down,
            x_per_slot: true,
            y: whole_of(&bufs.expert.expert_down)?,
            y_slot_stride: dims.hidden,
            k: dims.inter,
            rows: dims.hidden,
            n_expert: dims.n_expert,
            n_used: dims.top_k,
        },
    )
}

/// The shared gate/up/down matvec of the native block ([`MoeOp::SharedGateQ`],
/// [`MoeOp::SharedUpQ`], [`MoeOp::SharedDownQ`]): one `matvec_q` against the native blocks, the
/// shared `[gate | up]` row written at column 0 / `shared_inter`.
fn op_shared_q(
    launch: &Launch<'_>,
    weights: &MoeLayerWeights<'_>,
    bufs: &MoeBuffers<'_>,
    op: MoeOp,
) -> Result<(), EngError> {
    let dims = dims_of(weights, bufs);
    let (moe, _, shared) = bufs.expert.packs.q8()?;
    let (weight, x_q8, y, k, rows) = match op {
        MoeOp::SharedGateQ => (
            shared_native(&weights.shared_gate)?,
            moe,
            tail(&bufs.expert.shared_gu, 0)?,
            dims.hidden,
            dims.shared_inter,
        ),
        MoeOp::SharedUpQ => (
            shared_native(&weights.shared_up)?,
            moe,
            tail(&bufs.expert.shared_gu, dims.shared_inter)?,
            dims.hidden,
            dims.shared_inter,
        ),
        MoeOp::SharedDownQ => (
            shared_native(&weights.shared_down)?,
            shared,
            whole_of(&bufs.expert.shared_out)?,
            dims.shared_inter,
            dims.hidden,
        ),
        _ => return Err(refuse("the native shared launch takes a Q op".to_owned())),
    };
    launch.ctx.matvec_q(
        launch.stream,
        MatvecCall {
            ty: weight.quant(),
            w: native_stack(weight)?,
            x_q8,
            y,
            k,
            rows,
            ncols: 1,
        },
    )
}

/// The batched SwiGLU of the native block ([`MoeOp::SwigluExperts`], the eight slots into the
/// `Q8Down` rows, and [`MoeOp::SwigluSharedQ`], the shared row into `Q8Shared`).
fn op_swiglu_q(
    launch: &Launch<'_>,
    bufs: &MoeBuffers<'_>,
    dims: MoeDims,
    op: MoeOp,
) -> Result<(), EngError> {
    let (_, down, shared) = bufs.expert.packs.q8()?;
    let (gu, q8, n_ff, y_slot_stride, slots) = match op {
        MoeOp::SwigluExperts => (
            whole_of(&bufs.expert.expert_gu)?,
            down,
            dims.inter,
            2 * dims.inter,
            dims.top_k,
        ),
        MoeOp::SwigluSharedQ => (
            whole_of(&bufs.expert.shared_gu)?,
            shared,
            dims.shared_inter,
            2 * dims.shared_inter,
            1,
        ),
        _ => return Err(refuse("the native swiglu takes a Q op".to_owned())),
    };
    launch.ctx.swiglu_q8(
        launch.stream,
        SwigluQ8Call {
            gu,
            q8,
            n_ff,
            y_slot_stride,
            slots,
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
        MoeOp::ExpertGateQ | MoeOp::ExpertUpQ => op_expert_gu(launch, weights, bufs, op),
        MoeOp::SharedGateQ | MoeOp::SharedUpQ | MoeOp::SharedDownQ => {
            op_shared_q(launch, weights, bufs, op)
        }
        MoeOp::SwigluSharedQ | MoeOp::SwigluExperts => op_swiglu_q(launch, bufs, dims, op),
        MoeOp::ExpertDownQ => op_expert_down(launch, weights, bufs),
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
    for op in moe_sequence(dims.top_k, bufs.expert.packs.kind().layout()) {
        launch_op(launch, weights, bufs, op)?;
    }
    Ok(())
}

pub(super) mod tiered;

#[cfg(test)]
mod tests;
