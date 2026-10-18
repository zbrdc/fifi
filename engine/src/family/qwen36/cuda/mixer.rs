//! The eager mixer blocks of the resident step (decision 0047, packets RD-9b3a and RD-9b3b): one
//! DeltaNet layer or one attention layer of one decode step, queued on ONE stream.
//!
//! A block runs from the residual stream to the MoE input. WHAT it launches is the op list of
//! `mixer_ops.rs`, and every non-slot argument of a wrapper call comes from `mixer_shapes.rs`;
//! this file only resolves ranges and turns each op into one wrapper call. Everything is queued
//! and nothing is synchronized; a block never reads a device word and never copies to the host.
//!
//! Errors. The geometry, the weight shapes and the ranges are proved when the session builds
//! `MixerGeometry`, `MixerWeights` and `MixerViews`. `run_block` first checks that the stream owns
//! the state and that the state is valid (`BadRequest`, nothing queued). **Any `Err` after that
//! means the caller poisons the state** (`ResidentState::poison`): a wrapper refusal during a
//! block is a defect and the earlier kernels of the block, of the layers before it and of the
//! step are already queued. `cancel_enqueue` is only for a failure before `run_block` was called.
//!
//! Status writers in a block, in queue order: `attn_prep` ORs mask 4 into `StepParams.status` on
//! a refused position (RD-8a), and under Int8G32 the rows entry holds the same rule while
//! `kv8_append` ORs 4 only on a word that entry already refused (KV8-B). They run after
//! `step_begin` and before the router and `argmax`, as `step_plan.rs` requires. A block takes the
//! `Pos`, `KvLen` and `Status` words as 4-byte ranges (`ResidentState::step_field`), never the
//! 32-byte block.
//!
//! Design guidance only, no code copied.

use super::{
    dims::KvMode,
    layer_table::LayerTable,
    mixer_geometry::MixerGeometry,
    mixer_ops::{BLOCK_ROWS, BlockOp, block_ops},
    mixer_shapes::{attn_decode_shape, attn_prep_shape},
    mixer_shapes::{attn_q8_shape, check_delta_eps, conv_shape, delta_shape, kv8_shape},
    mixer_weights::{LayerWeights, MixerWeights},
    state::ResidentState,
    state_layout::Plane,
    step_plan::{Slot, StepPlan},
};
use crate::{
    EngError,
    family::{AttentionKind, project::cuda::WeightLayout},
    kernels::cuda::{
        CudaContext, DeviceArena, DeviceRange, DeviceStream, MatmulIo, PackCall, Projection,
        resident::{
            add_norm_pack::AddNormCall,
            add_rmsnorm_q8::AddNormQ8Call,
            attn_decode::{AttnDecodeCall, AttnSpan, AttnSplit},
            attn_decode_q8::AttnDecodeQ8Call,
            attn_prep::{AttnPrepCall, AttnPrepRowsCall, RopeParams},
            gdn_conv::ConvCall,
            gdn_delta::{DeltaCall, DeltaWeights},
            kv8_append::Kv8Data,
            matvec_q::MatvecCall,
            quantize_q8_1::QuantizeCall,
        },
    },
};

const F32: usize = size_of::<f32>();

/// The ops of every supported layer kind, built once.
struct KindOps {
    delta: Vec<BlockOp>,
    attn: Vec<BlockOp>,
}

impl KindOps {
    fn new(g: &MixerGeometry, layout: WeightLayout, kv_mode: KvMode) -> Result<Self, EngError> {
        Ok(Self {
            delta: block_ops(AttentionKind::DeltaNet, g, layout, kv_mode)?,
            attn: block_ops(AttentionKind::Full, g, layout, kv_mode)?,
        })
    }

    fn of(&self, kind: AttentionKind) -> Result<&[BlockOp], EngError> {
        match kind {
            AttentionKind::DeltaNet => Ok(&self.delta),
            AttentionKind::Full => Ok(&self.attn),
            other => Err(EngError::Checkpoint(format!(
                "mixer: attention kind {other:?} has no resident mixer"
            ))),
        }
    }

    fn all(&self) -> impl Iterator<Item = &BlockOp> {
        self.delta.iter().chain(self.attn.iter())
    }
}

/// The arena ranges the blocks use, resolved once per session from the slots the ops name.
pub(super) struct MixerViews<'a> {
    views: Vec<(Slot, DeviceRange<'a>)>,
}

impl<'a> MixerViews<'a> {
    pub(super) fn new(
        plan: &StepPlan,
        arena: &'a DeviceArena,
        g: &MixerGeometry,
        layout: WeightLayout,
        kv_mode: KvMode,
    ) -> Result<Self, EngError> {
        let ops = KindOps::new(g, layout, kv_mode)?;
        let mut views: Vec<(Slot, DeviceRange<'a>)> = Vec::new();
        for op in ops.all() {
            for slot in op.reads().into_iter().chain(op.writes()) {
                if views.iter().any(|(known, _)| *known == slot) {
                    continue;
                }
                let index = plan.index(slot).ok_or_else(|| {
                    EngError::BadRequest(format!("mixer views: the plan has no slot {slot:?}"))
                })?;
                views.push((slot, arena.tensor(index)?));
            }
        }
        Ok(Self { views })
    }

    fn get(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        self.views
            .iter()
            .find(|(known, _)| *known == slot)
            .map(|(_, range)| *range)
            .ok_or_else(|| {
                EngError::BadRequest(format!("mixer views: slot {slot:?} was not resolved"))
            })
    }
}

/// The residual stream of one block and the position mirror.
#[derive(Clone, Copy)]
pub(super) struct BlockIo<'a> {
    /// `f32 [hidden]`; holds the residual on entry and, on exit, the residual after the mixer add
    /// (this is `resid_in` of the MoE block).
    pub(super) resid: DeviceRange<'a>,
    /// Host mirror of the device `pos` word at the start of the step (`ResidentState::enqueue_rows`).
    pub(super) admit_pos: usize,
}

/// What a block runs against, bundled for the op executor.
struct Run<'r> {
    stream: &'r DeviceStream,
    state: &'r ResidentState,
    weights: &'r LayerWeights<'r>,
    ordinal: usize,
    io: BlockIo<'r>,
}

/// The pieces a session builds once.
pub(super) struct MixerParts<'a> {
    pub(super) geometry: MixerGeometry,
    pub(super) table: &'a LayerTable,
    pub(super) weights: MixerWeights<'a>,
    pub(super) views: MixerViews<'a>,
    /// The layout the op lists and the weight lookups were built for (MATVEC-PAR S3b): taken
    /// from the session options, which the step has checked against the loaded weights.
    pub(super) layout: WeightLayout,
    /// The KV mode the op lists were built for (KV8-B B3): the attention core is the fp16 pair
    /// under `Fp16` and the KV8 rows/append/q8-decode triple under `Int8G32`.
    pub(super) kv_mode: KvMode,
}

pub(super) struct Mixers<'a> {
    context: &'a CudaContext,
    parts: MixerParts<'a>,
    ops: KindOps,
}

/// `(offset, length)` in bytes of the tail of a slot of `slot_bytes` bytes that starts at column
/// `col` of the shared f32 row. The tail ends at the end of the slot, so a matmul's `y` range is
/// never shorter than its need.
pub(super) fn tail_extent(slot_bytes: usize, col: usize) -> Result<(usize, usize), EngError> {
    let offset = col
        .checked_mul(F32)
        .filter(|offset| *offset < slot_bytes)
        .ok_or_else(|| {
            EngError::BadRequest(format!(
                "mixer: column {col} lies outside a row slot of {slot_bytes} bytes"
            ))
        })?;
    Ok((offset, slot_bytes - offset))
}

/// The tail of `range` from column `col`.
fn tail<'b>(range: DeviceRange<'b>, col: usize) -> Result<DeviceRange<'b>, EngError> {
    let (offset, len) = tail_extent(range.bytes(), col)?;
    range.buffer().range(range.offset() + offset, len)
}

impl<'a> Mixers<'a> {
    /// The layer weights the chunk model borrows (CPF-7b-C B.4).
    pub(super) fn weights(&self) -> &MixerWeights<'a> {
        &self.parts.weights
    }

    pub(super) fn new(context: &'a CudaContext, parts: MixerParts<'a>) -> Result<Self, EngError> {
        check_delta_eps(parts.geometry.eps)?;
        let ops = KindOps::new(&parts.geometry, parts.layout, parts.kv_mode)?;
        Ok(Self {
            context,
            parts,
            ops,
        })
    }

    /// Queues layer `layer` on `stream` up to the MoE input. One row only (see
    /// `mixer_geometry.rs`). Any `Err` after the owner and health guards means: poison the state.
    pub(super) fn run_block(
        &self,
        stream: &DeviceStream,
        state: &ResidentState,
        layer: usize,
        io: BlockIo<'_>,
    ) -> Result<(), EngError> {
        state.check_block(stream)?;
        let slot = self.parts.table.slot(layer)?;
        let weights = self.parts.weights.layer_weights(layer)?;
        let run = Run {
            stream,
            state,
            weights,
            ordinal: slot.ordinal,
            io,
        };
        for op in self.ops.of(slot.kind)? {
            self.run_op(&run, op)?;
        }
        Ok(())
    }

    fn view(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        self.parts.views.get(slot)
    }

    fn run_op(&self, run: &Run<'_>, op: &BlockOp) -> Result<(), EngError> {
        let (ctx, stream, g) = (self.context, run.stream, &self.parts.geometry);
        match *op {
            BlockOp::NormPack {
                weight,
                delta,
                xf,
                norm_out,
                ..
            } => ctx.add_norm_pack(
                stream,
                AddNormCall {
                    m: BLOCK_ROWS,
                    n: g.hidden,
                    eps: g.eps,
                    h: run.io.resid,
                    delta: delta.map(|slot| self.view(slot)).transpose()?,
                    weight: run.weights.raw(weight)?,
                    xf: self.view(xf)?,
                    norm_out: norm_out.map(|slot| self.view(slot)).transpose()?,
                },
            ),
            BlockOp::Project {
                role, xf, y, col, ..
            } => {
                let weight = run.weights.dense(role)?;
                let projection = Projection {
                    weight,
                    first_row: 0,
                    rows: weight.n_rows(),
                };
                let io = MatmulIo {
                    xf: self.view(xf)?,
                    y: tail(self.view(y)?, col)?,
                    m: BLOCK_ROWS,
                };
                ctx.matmul(stream, projection, io)
            }
            BlockOp::Pack { x, xf, k, .. } => ctx.pack(
                stream,
                PackCall {
                    x: self.view(x)?,
                    xf: self.view(xf)?,
                    m: BLOCK_ROWS,
                    k,
                },
            ),
            BlockOp::NormQ8 {
                weight,
                delta,
                q8,
                norm_out,
                ..
            } => ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: g.hidden,
                    eps: g.eps,
                    h: run.io.resid,
                    delta: delta.map(|slot| self.view(slot)).transpose()?,
                    weight: run.weights.raw(weight)?,
                    norm_out: self.view(norm_out)?,
                    q8: self.view(q8)?,
                },
            ),
            BlockOp::ProjectQ {
                role, x_q8, y, col, ..
            } => {
                let weight = run.weights.native(role)?;
                let ty = weight.quant();
                ctx.matvec_q(
                    stream,
                    MatvecCall {
                        ty,
                        w: weight.buffer().range(0, weight.buffer().bytes())?,
                        x_q8: self.view(x_q8)?,
                        y: tail(self.view(y)?, col)?,
                        k: weight.k(),
                        rows: weight.n_rows(),
                        ncols: BLOCK_ROWS,
                    },
                )
            }
            BlockOp::QuantizeQ8 { x, q8, k, .. } => ctx.quantize_q8_1(
                stream,
                QuantizeCall {
                    x: self.view(x)?,
                    y: self.view(q8)?,
                    k,
                    ncols: BLOCK_ROWS,
                },
            ),
            BlockOp::Conv { proj, out, weights } => ctx.gdn_conv(
                stream,
                ConvCall {
                    shape: conv_shape(g),
                    proj: self.view(proj)?,
                    weights: run.weights.raw(weights)?,
                    state: run.state.delta_conv(0, run.ordinal)?,
                    out: self.view(out)?,
                },
            ),
            BlockOp::Delta {
                conv,
                proj,
                out,
                a,
                dt_bias,
                norm,
            } => ctx.gdn_delta(
                stream,
                DeltaCall {
                    shape: delta_shape(g),
                    eps: g.eps,
                    conv: self.view(conv)?,
                    proj: self.view(proj)?,
                    weights: DeltaWeights {
                        a: run.weights.raw(a)?,
                        dt_bias: run.weights.raw(dt_bias)?,
                        norm_w: run.weights.raw(norm)?,
                    },
                    state: run.state.delta_rec(0, run.ordinal)?,
                    out: self.view(out)?,
                },
            ),
            BlockOp::AttnPrep {
                proj,
                q,
                q_norm,
                k_norm,
                pos,
                status,
            } => ctx.attn_prep(
                stream,
                AttnPrepCall {
                    shape: attn_prep_shape(g, run.state.layout().kv.capacity),
                    rope: RopeParams {
                        eps: g.eps,
                        rope_base: g.attn.rope_base,
                    },
                    admit_pos: run.io.admit_pos,
                    proj: self.view(proj)?,
                    q_norm: run.weights.raw(q_norm)?,
                    k_norm: run.weights.raw(k_norm)?,
                    k_cache: run.state.kv_plane(run.ordinal, Plane::K)?,
                    v_cache: run.state.kv_plane(run.ordinal, Plane::V)?,
                    q: self.view(q)?,
                    pos: run.state.step_field(pos)?,
                    status: run.state.step_field(status)?,
                },
            ),
            BlockOp::AttnDecode {
                q,
                proj,
                part,
                out,
                kv_len,
            } => ctx.attn_decode(
                stream,
                AttnDecodeCall {
                    shape: attn_decode_shape(g, run.state.layout().kv.capacity),
                    span: AttnSpan::FULL,
                    split: AttnSplit::STAGE1,
                    scale: g.attn.scale,
                    q: self.view(q)?,
                    proj: Some(self.view(proj)?),
                    k_cache: run.state.kv_plane(run.ordinal, Plane::K)?,
                    v_cache: run.state.kv_plane(run.ordinal, Plane::V)?,
                    kv_len: run.state.step_field(kv_len)?,
                    part: self.view(part)?,
                    out: self.view(out)?,
                },
            ),
            // The KV8 core of the Int8G32 mode (KV8-B B3): the rows entry writes the f32 k and v
            // rows, `kv8_append` quantizes them into the layer's code and scale planes, and the
            // q8 decode reads the planes back at the split the plan sized its partials with.
            BlockOp::AttnPrepRows {
                proj,
                q,
                q_norm,
                k_norm,
                k_rows,
                v_rows,
                pos,
                status,
            } => ctx.attn_prep_rows(
                stream,
                AttnPrepRowsCall {
                    shape: attn_prep_shape(g, run.state.layout().kv.capacity),
                    rope: RopeParams {
                        eps: g.eps,
                        rope_base: g.attn.rope_base,
                    },
                    admit_pos: run.io.admit_pos,
                    proj: self.view(proj)?,
                    q_norm: run.weights.raw(q_norm)?,
                    k_norm: run.weights.raw(k_norm)?,
                    k_rows: self.view(k_rows)?,
                    v_rows: self.view(v_rows)?,
                    q: self.view(q)?,
                    pos: run.state.step_field(pos)?,
                    status: run.state.step_field(status)?,
                },
            ),
            BlockOp::Kv8Append {
                k,
                v,
                k_codes,
                k_scales,
                v_codes,
                v_scales,
                pos,
                status,
            } => ctx.kv8_append(
                stream,
                kv8_shape(g, run.state.layout().kv.capacity),
                Kv8Data {
                    k: self.view(k)?,
                    v: self.view(v)?,
                    k_codes: run.state.kv_plane(run.ordinal, k_codes)?,
                    k_scales: run.state.kv_plane(run.ordinal, k_scales)?,
                    v_codes: run.state.kv_plane(run.ordinal, v_codes)?,
                    v_scales: run.state.kv_plane(run.ordinal, v_scales)?,
                },
                run.state.step_field(pos)?,
                run.state.step_field(status)?,
            ),
            BlockOp::AttnDecodeQ8 {
                q,
                proj,
                k_codes,
                k_scales,
                v_codes,
                v_scales,
                kv_len,
                part,
                out,
                split,
            } => ctx.attn_decode_q8(
                stream,
                AttnDecodeQ8Call {
                    shape: attn_q8_shape(g, run.state.layout().kv.capacity),
                    span: AttnSpan::FULL,
                    split,
                    scale: g.attn.scale,
                    q: self.view(q)?,
                    proj: Some(self.view(proj)?),
                    k_codes: run.state.kv_plane(run.ordinal, k_codes)?,
                    k_scales: run.state.kv_plane(run.ordinal, k_scales)?,
                    v_codes: run.state.kv_plane(run.ordinal, v_codes)?,
                    v_scales: run.state.kv_plane(run.ordinal, v_scales)?,
                    kv_len: run.state.step_field(kv_len)?,
                    part: self.view(part)?,
                    out: self.view(out)?,
                },
            ),
        }
    }
}
