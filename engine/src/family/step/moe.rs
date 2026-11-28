//! The routed feed-forward block of a chunk.
//!
//! A router scores every expert for every token row and keeps the best few with their
//! weights. A shared expert runs on every row. Each kept expert runs the gated feed-forward on
//! its row: its gate and up projections, the activation, its down projection. The block's
//! output is the shared expert's rows plus the weighted rows of the kept experts. A few rows
//! go through the experts token by token; a chunk goes grouped by expert where the stacks
//! allow it, so every expert's weights are read once.

use super::experts::{self, Picks};
use super::ffn;
use super::rows::{Out, Rows, StepIo, q8};
use super::scratch::Alloc;
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as product_q8_bytes;
use crate::kernels::cuda::resident::dense_gemv::{
    DenseDtype, GgufMoeCall, GgufMoePicks, GgufMoeSchedule, GgufMoeShape, GgufMoeWeights,
};
use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
use crate::kernels::cuda::resident::matvec_gguf_f32::ExpertF32Call;
use crate::kernels::cuda::resident::moe_combine::{
    CombineCall, CombineOut, CombineShape, SharedTerm,
};
use crate::kernels::cuda::resident::router::{
    RouterCall, RouterLogitsCall, RouterOut, RouterShape, RouterSigmoidBias, RouterSigmoidCall,
    RouterWeights,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// Rows from which the experts run grouped.
const GROUPED_FROM: usize = 9;

/// The tensors of one routed block by their names in the weight map.
pub(crate) struct Names {
    pub(crate) router: String,
    pub(crate) rule: Router,
    /// The gate, up and down stacks of the experts.
    pub(crate) stacks: [String; 3],
    /// The gate, up and down matrices of the shared expert.
    pub(crate) shared: [String; 3],
}

/// How a router turns its logits into the kept experts and their weights.
pub(crate) enum Router {
    /// A sigmoid per expert; `bias` names what is added to every expert's probability before
    /// the choice. The shared expert's output enters whole.
    Sigmoid { bias: String },
    /// A softmax over the experts, renormalized over the kept ones. The shared expert's output
    /// enters scaled by the sigmoid of the row's product with the vector `shared_gate`.
    Softmax { shared_gate: String },
}

impl Router {
    fn gates_shared(&self) -> bool {
        matches!(self, Self::Softmax { .. })
    }

    /// The f32 tensor the rule reads beside the router matrix.
    pub(crate) fn side(&self) -> &str {
        match self {
            Self::Sigmoid { bias } => bias,
            Self::Softmax { shared_gate } => shared_gate,
        }
    }
}

/// The widths of a routed block.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub(crate) hidden: usize,
    /// The inner width of an expert.
    pub(crate) inner: usize,
    pub(crate) experts: usize,
    pub(crate) used: usize,
    /// Whether the router gates the shared expert ([`Router::Softmax`]).
    pub(crate) gated_shared: bool,
}

/// The rows of a routed block, for chunks of up to `rows` token rows.
pub(crate) struct Scratch<B = DeviceBuffer> {
    logits: B,
    /// The shared expert's gate of every row and the zero rows its sum starts from, for a
    /// router that gates it.
    gated: Option<(B, B)>,
    status: B,
    /// The kept experts of every row, i32 `[rows][used]`, and their weights.
    pub(crate) ids: B,
    weights: B,
    /// The `[gate | up]` rows of every kept expert.
    gate_up: B,
    act: B,
    act_q8: B,
    down: B,
    /// The shared expert's output rows.
    shared: B,
    routed_q8: B,
    act_product_q8: B,
    ids_g: B,
    ids_src1: B,
    ids_dst: B,
    expert_bounds: B,
}

impl<B> Scratch<B> {
    pub(crate) fn new<A: Alloc<Buf = B>>(a: &A, d: Dims, rows: usize) -> Result<Self, EngError> {
        let slots = rows * d.used;
        Ok(Self {
            logits: a.floats(rows * (d.experts + usize::from(d.gated_shared)))?,
            gated: if d.gated_shared {
                Some((a.floats(rows)?, a.zeroed(rows * d.hidden * 4)?))
            } else {
                None
            },
            status: a.zeroed(4)?,
            ids: a.floats(slots)?,
            weights: a.floats(slots)?,
            gate_up: a.floats(slots * 2 * d.inner)?,
            act: a.floats(slots * d.inner)?,
            act_q8: a.q8(d.inner, slots)?,
            down: a.floats(slots * d.hidden)?,
            shared: a.floats(rows * d.hidden)?,
            routed_q8: a.bytes(product_q8_bytes(d.hidden, slots)?)?,
            act_product_q8: a.bytes(product_q8_bytes(d.inner, slots)?)?,
            ids_g: a.floats(slots)?,
            ids_src1: a.floats(slots)?,
            ids_dst: a.floats(Picks::dst_ints(d.used, rows)?)?,
            expert_bounds: a.floats(d.experts + 1)?,
        })
    }
}

/// The rows a per-token expert product reads.
#[derive(Clone, Copy)]
enum Input<'a> {
    Q8(DeviceRange<'a>),
    F32(DeviceRange<'a>),
}

/// One per-token launch through the kept experts of the stack `name`.
#[allow(clippy::too_many_arguments)]
fn per_token(
    io: &StepIo<'_>,
    d: Dims,
    name: &str,
    x: Input<'_>,
    x_per_slot: bool,
    ids: DeviceRange<'_>,
    y: DeviceRange<'_>,
    (stride, rows, k): (usize, usize, usize),
) -> Result<(), EngError> {
    let weight = io.native(name)?;
    let x_q8 = match x {
        Input::Q8(rows) => rows,
        // The exact route: every weight value decoded over the f32 rows.
        Input::F32(x) => {
            return io.ctx.matvec_gguf_f32_expert(
                io.stream,
                ExpertF32Call {
                    ty: weight.quant(),
                    w_stack: StepIo::whole(weight.buffer())?,
                    ids,
                    x,
                    x_per_slot,
                    y,
                    y_slot_stride: stride,
                    k,
                    rows,
                    n_expert: d.experts,
                    n_used: d.used,
                    glu_gate: None,
                },
            );
        }
    };
    io.ctx.matvec_expert(
        io.stream,
        ExpertMatvecCall {
            ty: weight.quant(),
            w_stack: StepIo::whole(weight.buffer())?,
            ids,
            x_q8,
            x_per_slot,
            y,
            y_slot_stride: stride,
            k,
            rows,
            n_expert: d.experts,
            n_used: d.used,
            glu_gate: None,
        },
    )
}

/// The block over `n` normed rows `x` into `out`, rows back to back. `join` is the activation
/// of every expert, `shared` the scratch of the shared expert's feed-forward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    io: &StepIo<'_>,
    names: &Names,
    (d, join): (Dims, ffn::Join),
    sc: &Scratch,
    shared: ffn::Scratch<'_>,
    x: Rows<'_>,
    n: usize,
    out: &DeviceBuffer,
) -> Result<(), EngError> {
    let (hidden, inner, used) = (d.hidden, d.inner, d.used);
    let Some(x_f32) = x.f32 else {
        return Err(EngError::BadRequest(format!(
            "{}: the router needs f32 rows",
            io.family
        )));
    };
    let shape = RouterShape {
        m: n,
        hidden,
        n_expert: d.experts,
        top_k: used,
    };
    if names.rule.gates_shared() != d.gated_shared {
        return Err(EngError::BadRequest(format!(
            "{}: the routed block's widths and its router rule disagree on the shared expert's gate",
            io.family
        )));
    }
    let gated = match (&names.rule, &sc.gated) {
        (Router::Sigmoid { bias }, _) => {
            io.ctx.router_logits_rows(
                io.stream,
                RouterLogitsCall {
                    shape,
                    x: x_f32.range(0, n * hidden * 4)?,
                    w: io.raw(&names.router)?,
                    logits: sc.logits.range(0, n * d.experts * 4)?,
                },
            )?;
            io.ctx.router_sigmoid_select_rows(
                io.stream,
                RouterSigmoidCall {
                    shape,
                    logits: sc.logits.range(0, n * d.experts * 4)?,
                    ids: sc.ids.range(0, n * used * 4)?,
                    weights: sc.weights.range(0, n * used * 4)?,
                    status: StepIo::whole(&sc.status)?,
                    bias: Some(RouterSigmoidBias::KernelRounded(io.raw(bias)?)),
                },
            )?;
            None
        }
        (Router::Softmax { shared_gate }, Some((gate, zero))) => {
            let gate = gate.range(0, n * 4)?;
            io.ctx.router(
                io.stream,
                RouterCall {
                    shape,
                    x: x_f32.range(0, n * hidden * 4)?,
                    weights: RouterWeights {
                        w: io.raw(&names.router)?,
                        w_shared: Some(io.raw(shared_gate)?),
                    },
                    logits: sc.logits.range(0, n * (d.experts + 1) * 4)?,
                    out: RouterOut {
                        ids: sc.ids.range(0, n * used * 4)?,
                        weights: sc.weights.range(0, n * used * 4)?,
                        shared_gate: Some(gate),
                        status: StepIo::whole(&sc.status)?,
                    },
                },
            )?;
            Some((gate, zero.range(0, n * hidden * 4)?))
        }
        (Router::Softmax { .. }, None) => {
            return Err(EngError::BadRequest(format!(
                "{}: the routed block has no rows for the shared expert's gate",
                io.family
            )));
        }
    };
    // The shared expert first: its output is the base the kept experts are summed onto.
    let [shared_gate, shared_up, shared_down] = &names.shared;
    ffn::gated(
        io,
        ffn::Weights::named(io, shared_gate, shared_up, shared_down)?,
        join,
        x,
        n,
        shared,
        &sc.shared,
    )?;
    let [gate, up, down] = &names.stacks;
    let gu_row = used * 2 * inner * 4;
    let stride = 2 * inner;
    let grouped = |name: &str| !io.exact && n >= GROUPED_FROM && experts::serves(io, name);
    let gate_up = grouped(gate) && grouped(up);
    let down_grouped = grouped(down);
    let picks = Picks {
        experts: d.experts,
        used,
        ids: &sc.ids,
        ids_g: &sc.ids_g,
        ids_src1: &sc.ids_src1,
        ids_dst: &sc.ids_dst,
        expert_bounds: &sc.expert_bounds,
    };
    let ids = if gate_up || down_grouped {
        Some(picks.ids(io, n)?)
    } else {
        None
    };
    if let Some(ids) = ids.as_ref().filter(|_| gate_up) {
        let rows = experts::token_rows(
            ids,
            x_f32.range(0, n * hidden * 4)?,
            hidden,
            sc.routed_q8.range(0, product_q8_bytes(hidden, n * used)?)?,
        );
        for (name, offset) in [(gate, 0), (up, inner * 4)] {
            let dst = sc.gate_up.range(offset, n * gu_row - offset)?;
            experts::grouped(io, name, ids, &rows, stride, dst)?;
        }
    } else {
        let gu_span = ((used - 1) * stride + inner) * 4;
        let x_stride = q8(hidden, 1);
        for t in 0..n {
            let ids = sc.ids.range(t * used * 4, used * 4)?;
            let row = if io.exact {
                Input::F32(x_f32.range(t * hidden * 4, hidden * 4)?)
            } else {
                Input::Q8(x.q8.range(t * x_stride, x_stride)?)
            };
            for (name, offset) in [(gate, 0), (up, inner * 4)] {
                let y = sc.gate_up.range(t * gu_row + offset, gu_span)?;
                per_token(io, d, name, row, false, ids, y, (stride, inner, hidden))?;
            }
        }
    }
    let gu = sc.gate_up.range(0, n * gu_row)?;
    let scratch = ffn::Scratch {
        gate: Out {
            dst: gu,
            ld: stride,
            col_off: 0,
        },
        up: Out {
            dst: gu,
            ld: stride,
            col_off: inner,
        },
        // A swiglu writes its f32 rows first; the grouped down product and the exact route
        // read them too.
        act: if down_grouped || io.exact || matches!(join, ffn::Join::Swiglu) {
            Rows::both(&sc.act_q8, &sc.act)
        } else {
            Rows::q8(&sc.act_q8)
        },
    };
    // The grouped down product reads the f32 rows; only the per-token one reads q8_1 rows.
    ffn::joined(
        io,
        join,
        (n * used, inner),
        scratch,
        !down_grouped && !io.exact,
    )?;
    let e_stride = q8(inner, 1);
    if let Some(ids) = ids.as_ref().filter(|_| down_grouped) {
        let rows = experts::slot_rows(
            ids,
            sc.act.range(0, n * used * inner * 4)?,
            inner,
            sc.act_product_q8
                .range(0, product_q8_bytes(inner, n * used)?)?,
        );
        let dst = sc.down.range(0, n * used * hidden * 4)?;
        experts::grouped(io, down, ids, &rows, hidden, dst)?;
    } else {
        for t in 0..n {
            per_token(
                io,
                d,
                down,
                if io.exact {
                    Input::F32(sc.act.range(t * used * inner * 4, used * inner * 4)?)
                } else {
                    Input::Q8(sc.act_q8.range(t * used * e_stride, used * e_stride)?)
                },
                true,
                sc.ids.range(t * used * 4, used * 4)?,
                sc.down.range(t * used * hidden * 4, used * hidden * 4)?,
                (hidden, hidden, inner),
            )?;
        }
    }
    let shared_rows = sc.shared.range(0, n * hidden * 4)?;
    io.ctx.moe_combine(
        io.stream,
        CombineCall {
            shape: CombineShape {
                m: n,
                n: hidden,
                top_k: used,
                ld_down: hidden,
            },
            // A gated shared expert is a term of the sum; an ungated one is what the sum
            // starts from.
            res: match gated {
                Some((_, zero)) => zero,
                None => shared_rows,
            },
            down: sc.down.range(0, n * used * hidden * 4)?,
            weights: sc.weights.range(0, n * used * 4)?,
            shared: gated.map(|(gate, _)| SharedTerm {
                gate,
                row: shared_rows,
            }),
            out: CombineOut::To(out.range(0, n * hidden * 4)?),
        },
    )
}

/// One sigmoid router launch with a host-rounded per-expert bias, writing logits, top-k ids and
/// weights.
#[derive(Clone, Copy)]
pub(crate) struct HostRoundedRoute<'a> {
    pub(crate) shape: RouterShape,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) w: DeviceRange<'a>,
    pub(crate) bias: DeviceRange<'a>,
    pub(crate) floor: f32,
    pub(crate) scale: f32,
    pub(crate) logits: DeviceRange<'a>,
    pub(crate) ids: DeviceRange<'a>,
    pub(crate) weights: DeviceRange<'a>,
    pub(crate) status: DeviceRange<'a>,
}

/// Scores experts and picks top-k ids and weights with a host-rounded sigmoid bias.
pub(crate) fn route_host_rounded(
    ctx: &CudaContext,
    stream: &DeviceStream,
    route: HostRoundedRoute<'_>,
) -> Result<(), EngError> {
    ctx.router_logits_rows(
        stream,
        RouterLogitsCall {
            shape: route.shape,
            x: route.x,
            w: route.w,
            logits: route.logits,
        },
    )?;
    ctx.router_sigmoid_select_rows(
        stream,
        RouterSigmoidCall {
            shape: route.shape,
            bias: Some(RouterSigmoidBias::HostRounded {
                values: route.bias,
                floor: route.floor,
                scale: route.scale,
            }),
            logits: route.logits,
            ids: route.ids,
            weights: route.weights,
            status: route.status,
        },
    )
}

/// One routed feed-forward row over fused gate-up and down GGUF stacks with per-expert scales and
/// a host-rounded sigmoid router bias.
pub(crate) struct StackedRow<'a> {
    pub(crate) route: HostRoundedRoute<'a>,
    pub(crate) moe: GgufMoeShape,
    pub(crate) dtypes: [DenseDtype; 2],
    pub(crate) caps: [f32; 2],
    pub(crate) stacks: [DeviceRange<'a>; 2],
    pub(crate) scales: [DeviceRange<'a>; 2],
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) schedule: GgufMoeSchedule,
}

/// Routes one row with a host-rounded sigmoid bias and runs its kept experts through fused
/// gate-up and down GGUF stacks with per-expert scales.
pub(crate) fn stacked_row(
    ctx: &CudaContext,
    stream: &DeviceStream,
    row: StackedRow<'_>,
) -> Result<GgufMoeSchedule, EngError> {
    route_host_rounded(ctx, stream, row.route)?;
    ctx.gguf_moe_scheduled(
        stream,
        GgufMoeCall {
            shape: row.moe,
            dtypes: row.dtypes,
            caps: row.caps,
            picks: GgufMoePicks::Device {
                count: row.route.shape.top_k,
                ids: row.route.ids,
                weights: row.route.weights,
                scales: row.scales,
            },
            weights: GgufMoeWeights::Stacks(row.stacks),
            x: row.route.x,
            scratch: row.scratch,
            y: row.out,
        },
        row.schedule,
    )
}
