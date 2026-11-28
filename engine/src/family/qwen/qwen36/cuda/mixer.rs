//! The eager mixer blocks of the resident step: one DeltaNet layer or one attention layer of one
//! decode step, queued on ONE stream.
//!
//! A block runs from the residual stream to the MoE input. WHAT it launches is the op list of
//! `mixer_ops.rs`, and every non-slot argument of a wrapper call comes from `mixer_shapes.rs`;
//! this file only resolves ranges and turns each op into one wrapper call. Everything is queued
//! on the resident path. Paged attention synchronizes its bounded host/device transfer slots.
//!
//! Errors. The geometry, the weight shapes and the ranges are proved when the session builds
//! `MixerGeometry`, `MixerWeights` and `MixerViews`. `run_block` first checks that the stream owns
//! the state and that the state is valid (`BadRequest`, nothing queued). **Any `Err` after that
//! means the caller poisons the state** (`ResidentState::poison`): a wrapper refusal during a
//! block is a defect and the earlier kernels of the block, of the layers before it and of the
//! step are already queued. `cancel_enqueue` is only for a failure before `run_block` was called.
//!
//! Status writers in a block, in queue order: `attn_prep` ORs mask 4 into `StepParams.status` on a
//! refused position, and under Int8G32 the rows entry holds the same rule while `kv8_append` ORs 4
//! only on a word that entry already refused. They run after `step_begin` and before the router and
//! `argmax`, as `step_plan.rs` requires. A block takes the `Pos`, `KvLen` and `Status` words as
//! 4-byte ranges (`ResidentState::step_field`), never the 32-byte block.
use crate::family::exl3::layer_table::LayerTable;

use super::{
    dims::KvMode,
    mixer_ops::{BLOCK_ROWS, BlockOp, CardOp, block_ops, block_ops_card},
    mixer_shapes::{attn_decode_shape, attn_prep_shape},
    mixer_shapes::{attn_q8_shape, check_delta_eps, conv_shape, delta_shape, kv8_shape},
    mixer_weights::{LayerWeights, MixerWeights},
    paging_store::{PageAttention, PageId, PageToken},
    resources::PagedResources,
    sparse_exec::SparseExec,
    state::{ResidentState, StepField},
    step_plan::{Slot, StepPlan},
};
use crate::family::decode_step::mixer_geometry::MixerGeometry;
use crate::family::decode_step::shard::ShardTable;
use crate::family::decode_step::state_layout::Plane;
use crate::family::state_words::Word;
use crate::{
    EngError,
    family::{AttentionKind, project::cuda::WeightLayout},
    kernels::cuda::{
        CudaContext, DeviceArena, DeviceRange, DeviceStream, MatmulIo, PackCall, Projection,
        TimingEvent,
        resident::{
            add_norm_pack::AddNormCall,
            add_rmsnorm_q8::AddNormQ8Call,
            attn_decode::{AttnDecodeCall, AttnSpan, AttnSplit},
            attn_decode_q8::{AttnDecodeQ8Call, AttnQ8Shape},
            attn_prep::{AttnPrepCall, AttnPrepRowsCall, RopeParams},
            gdn_conv::ConvCall,
            gdn_delta::{DeltaCall, DeltaWeights},
            kv_f16_append::{KvF16AppendData, KvF16AppendShape},
            kv8_append::Kv8Data,
            quantize_q8_1::QuantizeCall,
            sparse_decode::SparseDecodeData,
            sparse_decode::SparseDecodeShape,
            sparse_score::{SparseScoreData, SparseScoreShape},
            sparse_sketch::{SparseSketchData, SparseSketchShape},
            sparse_topk::{SparseTopkData, SparseTopkShape},
        },
    },
};

const F32: usize = size_of::<f32>();

struct KindOps {
    delta: Vec<BlockOp>,
    attn: Vec<BlockOp>,
}

impl KindOps {
    fn new(
        g: &MixerGeometry,
        layout: WeightLayout,
        kv_mode: KvMode,
        moe_tiled: bool,
    ) -> Result<Self, EngError> {
        Ok(Self {
            delta: block_ops(AttentionKind::DeltaNet, g, layout, kv_mode, moe_tiled)?,
            attn: block_ops(AttentionKind::Full, g, layout, kv_mode, moe_tiled)?,
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

/// The card op lists of every layer kind (Native weights), for one card of the multi-device
/// step: the local launches of `block_ops_card` with the ring markers in place. The coordinator
/// runs these against the card's own stream and serves the markers from the peer ring.
pub(super) struct CardKindOps {
    pub(super) delta: Vec<CardOp>,
    pub(super) attn: Vec<CardOp>,
}

impl CardKindOps {
    /// The card `card` of `table`: its own geometry `g` (`MixerGeometry::for_card`) and its
    /// absolute K/V row range into the full cache row.
    pub(super) fn new(
        table: &ShardTable,
        card: usize,
        g: &MixerGeometry,
        kv_mode: KvMode,
    ) -> Result<Self, EngError> {
        let rows = table.kv_rows(card)?;
        let k_range = (rows.start, rows.end);
        Ok(Self {
            delta: block_ops_card(
                crate::family::AttentionKind::DeltaNet,
                g,
                k_range,
                WeightLayout::Native,
                kv_mode,
            )?,
            attn: block_ops_card(
                crate::family::AttentionKind::Full,
                g,
                k_range,
                WeightLayout::Native,
                kv_mode,
            )?,
        })
    }

    pub(super) fn of(&self, kind: crate::family::AttentionKind) -> Result<&[CardOp], EngError> {
        match kind {
            crate::family::AttentionKind::DeltaNet => Ok(&self.delta),
            crate::family::AttentionKind::Full => Ok(&self.attn),
            other => Err(EngError::Checkpoint(format!(
                "card mixer: attention kind {other:?} has no resident mixer"
            ))),
        }
    }

    fn all(&self) -> impl Iterator<Item = &CardOp> {
        self.delta.iter().chain(self.attn.iter())
    }
}

/// The arena ranges the blocks use, resolved once per session from the slots the ops name.
pub(super) struct MixerViews<'a> {
    views: Vec<(Slot, DeviceRange<'a>)>,
}

/// The slots every op of `ops` names, deduplicated in first-seen order and resolved against
/// `plan` (the shared enumeration of the single-device and card view builders).
fn named_slots<'s>(
    plan: &StepPlan,
    ops: impl Iterator<Item = &'s BlockOp>,
) -> Result<Vec<(Slot, usize)>, EngError> {
    let mut indices: Vec<(Slot, usize)> = Vec::new();
    for op in ops {
        for slot in op.reads().into_iter().chain(op.writes()) {
            if indices.iter().any(|(known, _)| *known == slot) {
                continue;
            }
            let index = plan.index(slot).ok_or_else(|| {
                EngError::BadRequest(format!("mixer views: the plan has no slot {slot:?}"))
            })?;
            indices.push((slot, index));
        }
    }
    Ok(indices)
}

impl<'a> MixerViews<'a> {
    pub(super) fn new(
        plan: &StepPlan,
        arena: &'a DeviceArena,
        g: &MixerGeometry,
        layout: WeightLayout,
        kv_mode: KvMode,
        moe_tiled: bool,
    ) -> Result<Self, EngError> {
        let views = Self::slot_indices(plan, g, layout, kv_mode, moe_tiled)?
            .into_iter()
            .map(|(slot, index)| Ok((slot, arena.tensor(index)?)))
            .collect::<Result<Vec<_>, EngError>>()?;
        Ok(Self { views })
    }

    /// The card variant of [`Self::new`] for the multi-device shards: the slots come from the
    /// CARD op lists (whose gathers and fp16 append name KRows/VRows under both modes, which
    /// the single-device Fp16 list never does), resolved against the card plan
    /// `StepPlan::for_card` builds. Every card of the table builds its own views.
    pub(super) fn for_card(
        plan: &StepPlan,
        arena: &'a DeviceArena,
        ops: &CardKindOps,
    ) -> Result<Self, EngError> {
        let locals = || {
            ops.all().filter_map(|op| match op {
                CardOp::Local(op) => Some(op),
                CardOp::GatherK | CardOp::GatherV | CardOp::MixerSum => None,
            })
        };
        let views = named_slots(plan, locals())?
            .into_iter()
            .map(|(slot, index)| Ok((slot, arena.tensor(index)?)))
            .collect::<Result<Vec<_>, EngError>>()?;
        Ok(Self { views })
    }

    /// The pure slot enumeration of [`Self::for_card`]: every LOCAL op of the card lists names
    /// a slot the card plan holds (the markers carry no slots; the ring moves ranges the
    /// coordinator already holds).
    #[cfg(test)]
    pub(super) fn card_slot_indices(
        plan: &StepPlan,
        ops: &CardKindOps,
    ) -> Result<Vec<(Slot, usize)>, EngError> {
        let locals = || {
            ops.all().filter_map(|op| match op {
                CardOp::Local(op) => Some(op),
                CardOp::GatherK | CardOp::GatherV | CardOp::MixerSum => None,
            })
        };
        named_slots(plan, locals())
    }

    pub(super) fn slot_indices(
        plan: &StepPlan,
        g: &MixerGeometry,
        layout: WeightLayout,
        kv_mode: KvMode,
        moe_tiled: bool,
    ) -> Result<Vec<(Slot, usize)>, EngError> {
        let ops = KindOps::new(g, layout, kv_mode, moe_tiled)?;
        let mut indices: Vec<(Slot, usize)> = Vec::new();
        for op in ops.all() {
            for slot in op.reads().into_iter().chain(op.writes()) {
                if indices.iter().any(|(known, _)| *known == slot) {
                    continue;
                }
                let index = match plan.index(slot) {
                    Some(index) => index,
                    None if slot == Slot::AttnPart
                        && kv_mode == KvMode::Int8G32
                        && layout == WeightLayout::Native =>
                    {
                        continue;
                    }
                    None => {
                        return Err(EngError::BadRequest(format!(
                            "mixer views: the plan has no slot {slot:?}"
                        )));
                    }
                };
                indices.push((slot, index));
            }
        }
        Ok(indices)
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

#[derive(Clone, Copy)]
pub(super) struct BlockIo<'a> {
    /// `f32 [hidden]`; holds the residual on entry and, on exit, the residual after the mixer add
    /// (this is `resid_in` of the MoE block).
    pub(super) resid: DeviceRange<'a>,
    /// Host mirror of the device `pos` word at the start of the step (`ResidentState::enqueue_rows`).
    pub(super) admit_pos: usize,
}

struct Run<'r> {
    stream: &'r DeviceStream,
    state: &'r ResidentState,
    weights: &'r LayerWeights<'r>,
    ordinal: usize,
    io: BlockIo<'r>,
}

pub(super) struct MixerParts<'a> {
    pub(super) geometry: MixerGeometry,
    pub(super) table: &'a LayerTable,
    pub(super) weights: MixerWeights<'a>,
    pub(super) views: MixerViews<'a>,
    /// The layout the op lists and the weight lookups were built for: taken from the session
    /// options, which the step has checked against the loaded weights.
    pub(super) layout: WeightLayout,
    /// The KV mode the op lists were built for: the attention core is the fp16 pair under `Fp16`
    /// and the KV8 rows/append/q8-decode triple under `Int8G32`.
    pub(super) kv_mode: KvMode,
    /// Whether the MoE epilogue must produce the fp16 pack: the tiled-experts hybrid overrides
    /// the expert stacks to tiles beside the native session, so their launches read `XfMoe`
    /// instead of `Q8Moe`.
    pub(super) moe_tiled: bool,
    pub(super) paged: Option<&'a PagedResources>,
    /// Present only when armed and admitted: sparse decode resources, else every op runs dense.
    pub(super) sparse: Option<&'a SparseExec>,
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

fn tail<'b>(range: DeviceRange<'b>, col: usize) -> Result<DeviceRange<'b>, EngError> {
    let (offset, len) = tail_extent(range.bytes(), col)?;
    range.buffer().range(range.offset() + offset, len)
}

impl<'a> Mixers<'a> {
    /// The layer weights the chunk model borrows.
    pub(super) fn weights(&self) -> &MixerWeights<'a> {
        &self.parts.weights
    }

    pub(super) fn new(context: &'a CudaContext, parts: MixerParts<'a>) -> Result<Self, EngError> {
        check_delta_eps(parts.geometry.eps)?;
        let ops = KindOps::new(
            &parts.geometry,
            parts.layout,
            parts.kv_mode,
            parts.moe_tiled,
        )?;
        Ok(Self {
            context,
            parts,
            ops,
        })
    }

    /// Queues layer `layer` on `stream` up to the MoE input. One row only (see
    /// `mixer_geometry.rs`).
    pub(super) fn run_block(
        &self,
        stream: &DeviceStream,
        state: &ResidentState,
        layer: usize,
        io: BlockIo<'_>,
        core_marks: Option<(&TimingEvent, &TimingEvent)>,
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
        // The finer split's attention-core window: opens at the block's first core op
        // (`Conv` of the delta list, `AttnPrep*` of the full list) and closes after its last
        // (`Delta`, `AttnDecode*`), so the pair brackets the core whatever the list's exact
        // shape (the KV-append ops sit inside it).
        let mut core_open = false;
        for op in self.ops.of(slot.kind)? {
            let opens_core = matches!(
                op,
                BlockOp::Conv { .. }
                    | BlockOp::AttnPrep { .. }
                    | BlockOp::AttnPrepRows { .. }
                    | BlockOp::AttnPrepCard { .. }
            );
            let closes_core = matches!(
                op,
                BlockOp::Delta { .. }
                    | BlockOp::AttnDecode { .. }
                    | BlockOp::AttnDecodeQ8 { .. }
                    | BlockOp::AttnDecodeQ8Card { .. }
            );
            if let Some((core_start, _)) = core_marks
                && opens_core
                && !core_open
            {
                stream.record_timing(core_start)?;
                core_open = true;
            }
            self.run_op(&run, op)?;
            if let Some((_, core_end)) = core_marks
                && closes_core
            {
                stream.record_timing(core_end)?;
            }
        }
        Ok(())
    }

    /// The arena view of `slot` (the card block driver resolves the rows the
    /// ring moves through this).
    pub(super) fn slot_view(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        self.view(slot)
    }

    fn view(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        self.parts.views.get(slot)
    }

    /// One LOCAL op of a card list against this mixer's parts: the card
    /// geometry, the card weight table and the card arena views. The card
    /// block driver walks the op list and routes the exchange markers to
    /// the ring instead.
    pub(super) fn run_local_op(
        &self,
        stream: &DeviceStream,
        state: &ResidentState,
        layer: usize,
        ordinal: usize,
        io: BlockIo<'_>,
        op: &BlockOp,
    ) -> Result<(), EngError> {
        let weights = self.parts.weights.layer_weights(layer)?;
        let run = Run {
            stream,
            state,
            weights,
            ordinal,
            io,
        };
        self.run_op(&run, op)
    }

    fn append_paged(&self, run: &Run<'_>, call: PageToken<'_>) -> Result<(), EngError> {
        if let Some(store) = run.state.cached_store() {
            return store
                .try_borrow_mut()
                .map_err(|_| EngError::BadRequest("cached KV store is already borrowed".into()))?
                .append(self.context, run.stream, call);
        }
        let mut store = run
            .state
            .paged_store()?
            .try_borrow_mut()
            .map_err(|_| EngError::BadRequest("paged KV store is already borrowed".into()))?;
        // The preceding layer's attention drained both slots; no resident page is aliased.
        store.load(self.context, run.stream, 0, call.page)?;
        store.append_decode(self.context, run.stream, 0, call)
    }

    fn attend_paged(
        &self,
        run: &Run<'_>,
        paged: &PagedResources,
        call: PageAttention<'_>,
        kv_len: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        if let Some(store) = run.state.cached_store() {
            let mut scratch = paged.scratch.try_borrow_mut().map_err(|_| {
                EngError::BadRequest("paged attention scratch is already borrowed".into())
            })?;
            return store
                .try_borrow_mut()
                .map_err(|_| EngError::BadRequest("cached KV store is already borrowed".into()))?
                .attend(
                    self.context,
                    run.stream,
                    &mut scratch,
                    super::paging_store::device_first::CachedAttention { page: call, kv_len },
                );
        }
        let mut store = run
            .state
            .paged_store()?
            .try_borrow_mut()
            .map_err(|_| EngError::BadRequest("paged KV store is already borrowed".into()))?;
        let mut scratch = paged.scratch.try_borrow_mut().map_err(|_| {
            EngError::BadRequest("paged attention scratch is already borrowed".into())
        })?;
        store.attend(self.context, run.stream, &mut scratch, call)
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
                    rows: 1,
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
                let x_f32 = (x_q8 == Slot::Q8Mix)
                    .then(|| self.view(Slot::NormRow))
                    .transpose()?;
                ctx.project_1row_native(
                    stream,
                    role.dq_tag(),
                    weight,
                    x_f32,
                    self.view(x_q8)?,
                    tail(self.view(y)?, col)?,
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
                    inv_freq: None,
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
            // The KV8 core of the Int8G32 mode: the rows entry writes the f32 k and v rows,
            // `kv8_append` quantizes them into the layer's code and scale planes, and the q8 decode
            // reads the planes back at the split the plan sized its partials with.
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
                    inv_freq: None,
                },
            ),
            BlockOp::Kv8Append {
                k,
                v,
                k_at,
                v_at,
                k_codes,
                k_scales,
                v_codes,
                v_scales,
                pos,
                status,
            } => {
                if let Some(paged) = self.parts.paged {
                    let page = PageId {
                        full_layer: run.ordinal,
                        group: run.io.admit_pos / paged.state.page_cells(),
                    };
                    self.append_paged(
                        run,
                        PageToken {
                            page,
                            position: run.io.admit_pos,
                            params: run.state.word(Word::StepParams)?,
                            k: self.view(k)?,
                            v: self.view(v)?,
                        },
                    )
                } else {
                    // The card's slice of the gathered full-width row: its
                    // K/V heads at `k_at` f32 values in, `heads_kv` heads
                    // wide, quantized into its sliced planes.
                    let kv_width = g.attn.heads_kv * g.attn.head_dim;
                    let slice = |slot: Slot, at: usize| -> Result<_, EngError> {
                        let view = self.view(slot)?;
                        view.buffer()
                            .range(view.offset() + at * F32, kv_width * F32)
                    };
                    ctx.kv8_append(
                        stream,
                        kv8_shape(g, run.state.layout().kv.capacity),
                        Kv8Data {
                            k: slice(k, k_at)?,
                            v: slice(v, v_at)?,
                            k_codes: run.state.kv_plane(run.ordinal, k_codes)?,
                            k_scales: run.state.kv_plane(run.ordinal, k_scales)?,
                            v_codes: run.state.kv_plane(run.ordinal, v_codes)?,
                            v_scales: run.state.kv_plane(run.ordinal, v_scales)?,
                        },
                        run.state.step_field(pos)?,
                        run.state.step_field(status)?,
                    )?;
                    // R1: the sketch is derived state written exactly once, right after the
                    // one append, keyed by the same pos word; a refused launch sets the step
                    // status mask and poisons the step, so no partially committed sketch is
                    // ever consumed. Only full-attention layers carry a plane.
                    if let Some(sp) = self.parts.sparse
                        && let Some(plane) = sp.plane_of(run.ordinal)
                    {
                        ctx.sparse_sketch(
                            stream,
                            SparseSketchShape {
                                rows: 1,
                                n_kv_head: sp.n_kv_head(),
                                head_dim: sp.head_dim(),
                                rank: sp.rank(),
                                cells: sp.cells(),
                            },
                            SparseSketchData {
                                k: self.view(k)?,
                                basis: sp.basis_range()?,
                                sketch: sp.sketch_plane(plane)?,
                            },
                            run.state.step_field(pos)?,
                            run.state.step_field(status)?,
                        )?;
                        // R3: the append op just enqueued this row's sketch cell; the
                        // attention op of the same step checks sketch_covers against it.
                        sp.note_appended(run.io.admit_pos);
                    }
                    Ok(())
                }
            }
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
            } => {
                if let Some(paged) = self.parts.paged {
                    let visible_keys = run.io.admit_pos.checked_add(1).ok_or_else(|| {
                        EngError::BadRequest("paged KV position overflows".into())
                    })?;
                    self.attend_paged(
                        run,
                        paged,
                        PageAttention {
                            full_layer: run.ordinal,
                            visible_keys,
                            scale: g.attn.scale,
                            q: self.view(q)?,
                            proj: Some(self.view(proj)?),
                            out: self.view(out)?,
                        },
                        run.state.step_field(kv_len)?,
                    )
                } else {
                    // The authoritative dense call, verbatim: the pre-launch bypass and the
                    // post-refusal fallback share it (R2).
                    let dense = || {
                        ctx.attn_decode_q8(
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
                        )
                    };
                    // R2: the sparse chain replaces the dense call only when the resources
                    // exist, the layer is full-attention, and the selection basis is valid for
                    // the visible window. Otherwise the dense call runs, unmodified. Host-side
                    // chain errors (status range, launches) do NOT fall back: they propagate
                    // and poison the step — only the kernel's own refusal bit routes dense.
                    let visible_keys = run.io.admit_pos.checked_add(1).ok_or_else(|| {
                        EngError::BadRequest("resident KV position overflows".into())
                    })?;
                    // `cells` must equal the session's KV stride: the decode indexes the real
                    // code/scale planes with it, so any drift reads misaligned rows.
                    let sparse = match self.parts.sparse {
                        Some(sp)
                            if sp.cfg().basis_valid_for(visible_keys)
                                && sp.sketch_covers(visible_keys)
                                && sp.cells() == run.state.layout().kv.capacity =>
                        {
                            sp.plane_of(run.ordinal).map(|plane| (sp, plane))
                        }
                        _ => None,
                    };
                    match sparse {
                        None => dense(),
                        Some((sp, plane)) => {
                            sp.note_launch(plane);
                            let status = sp.status_range()?;
                            sp.clear_status(stream)?;
                            ctx.sparse_score(
                                stream,
                                SparseScoreShape {
                                    n_q_head: sp.n_head(),
                                    n_kv_head: sp.n_kv_head(),
                                    head_dim: sp.head_dim(),
                                    rank: sp.rank(),
                                    cells: sp.cells(),
                                    sink: sp.sink(),
                                    recent: sp.recent(),
                                    scale: g.attn.scale,
                                },
                                SparseScoreData {
                                    q: self.view(q)?,
                                    basis: sp.basis_range()?,
                                    sketch: sp.sketch_plane(plane)?,
                                    scores: sp.scores_range()?,
                                },
                                // The sparse kernels' device word is the POS of the token
                                // being appended (kv_len = *pos + 1, the kv8_append word),
                                // not the dense decode's kv_len count word.
                                run.state.step_field(StepField::Pos)?,
                                status,
                            )?;
                            ctx.sparse_topk(
                                stream,
                                SparseTopkShape {
                                    n_kv_head: sp.n_kv_head(),
                                    cells: sp.cells(),
                                    top_k: sp.top_k(),
                                    sink: sp.sink(),
                                    recent: sp.recent(),
                                },
                                SparseTopkData {
                                    scores: sp.scores_range()?,
                                    indices: sp.indices_range()?,
                                    counts: sp.counts_range()?,
                                },
                                run.state.step_field(StepField::Pos)?,
                                status,
                            )?;
                            ctx.sparse_decode(
                                stream,
                                SparseDecodeShape {
                                    n_kv_head: sp.n_kv_head(),
                                    n_head: sp.n_head(),
                                    head_dim: sp.head_dim(),
                                    cells: sp.cells(),
                                    row: sp.row(),
                                    scale: g.attn.scale,
                                },
                                SparseDecodeData {
                                    q: self.view(q)?,
                                    k_codes: run.state.kv_plane(run.ordinal, k_codes)?,
                                    k_scales: run.state.kv_plane(run.ordinal, k_scales)?,
                                    v_codes: run.state.kv_plane(run.ordinal, v_codes)?,
                                    v_scales: run.state.kv_plane(run.ordinal, v_scales)?,
                                    indices: sp.indices_range()?,
                                    counts: sp.counts_range()?,
                                    out: self.view(out)?,
                                    proj: Some(self.view(proj)?),
                                    status,
                                },
                            )?;
                            // R2, transactional: a refused decode wrote nothing to `out`, and
                            // this blocking readback orders the whole chain; the dense
                            // overwrite below is complete before any consumer runs.
                            if sp.decode_refused(stream)? {
                                dense()
                            } else {
                                Ok(())
                            }
                        }
                    }
                }
            }
            // The card q8 decode: THIS card's query heads over ITS kv-head
            // range of the replicated planes. The head-major kv8 planes
            // make the range contiguous; the kernel's fixed GROUP mapping
            // then sees whole groups, relative head 0 being the card's
            // first (its fused row and q slot hold exactly its heads).
            BlockOp::AttnDecodeQ8Card {
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
                kv_heads,
            } => {
                let shape = AttnQ8Shape {
                    m: BLOCK_ROWS,
                    n_head: g.attn.heads_q,
                    n_kv_head: kv_heads.1 - kv_heads.0,
                    head_dim: g.attn.head_dim,
                    cells: run.state.layout().kv.capacity,
                    ld: g.attn.ld,
                };
                // The sub-range of one kv-head span inside the layer's
                // plane: the card geometry keeps the FULL kv head count,
                // so the per-head stride divides the plane evenly.
                let plane_extent = |plane: Plane| -> Result<DeviceRange<'_>, EngError> {
                    let full = run.state.kv_plane(run.ordinal, plane)?;
                    let per_head = full.bytes() / g.attn.heads_kv;
                    full.buffer().range(
                        full.offset() + kv_heads.0 * per_head,
                        (kv_heads.1 - kv_heads.0) * per_head,
                    )
                };
                ctx.attn_decode_q8(
                    stream,
                    AttnDecodeQ8Call {
                        shape,
                        span: AttnSpan::FULL,
                        split,
                        scale: g.attn.scale,
                        q: self.view(q)?,
                        proj: Some(self.view(proj)?),
                        k_codes: plane_extent(k_codes)?,
                        k_scales: plane_extent(k_scales)?,
                        v_codes: plane_extent(v_codes)?,
                        v_scales: plane_extent(v_scales)?,
                        kv_len: run.state.step_field(kv_len)?,
                        part: self.view(part)?,
                        out: self.view(out)?,
                    },
                )
            }
            // The card prep: the SAME rows kernel over the card's head
            // counts, writing the card's K/V extent at its GLOBAL rows of
            // the full replicated row buffers (the gathered row is then
            // assembled in place). Only a card mixer emits this op, and its
            // parts carry the card geometry, so the launch below is the
            // card's own.
            BlockOp::AttnPrepCard {
                proj,
                q,
                q_norm,
                k_norm,
                k_rows,
                v_rows,
                k_range,
                pos,
                status,
            } => {
                let kv_heads = super::card_attn::card_kv_heads(k_range, g.attn.head_dim)?;
                fn row_extent(
                    view: DeviceRange<'_>,
                    k_range: (usize, usize),
                ) -> Result<DeviceRange<'_>, EngError> {
                    view.buffer().range(
                        view.offset() + super::card_attn::card_slice_offset(k_range),
                        (k_range.1 - k_range.0) * F32,
                    )
                }
                ctx.attn_prep_rows(
                    stream,
                    AttnPrepRowsCall {
                        shape: super::card_attn::card_prep_shape(
                            g,
                            kv_heads,
                            run.state.layout().kv.capacity,
                        ),
                        rope: RopeParams {
                            eps: g.eps,
                            rope_base: g.attn.rope_base,
                        },
                        admit_pos: run.io.admit_pos,
                        proj: self.view(proj)?,
                        q_norm: run.weights.raw(q_norm)?,
                        k_norm: run.weights.raw(k_norm)?,
                        k_rows: row_extent(self.view(k_rows)?, k_range)?,
                        v_rows: row_extent(self.view(v_rows)?, k_range)?,
                        q: self.view(q)?,
                        pos: run.state.step_field(pos)?,
                        status: run.state.step_field(status)?,
                        inv_freq: None,
                    },
                )
            }
            // The card F16 append: the card's slice of the gathered full
            // rows into the card's sliced fp16 cache pair of this layer.
            BlockOp::KvAppendF16 {
                k,
                v,
                k_at,
                v_at,
                k_plane,
                v_plane,
                pos,
                status,
            } => {
                let shape = KvF16AppendShape {
                    rows: 1,
                    n_kv_head: g.attn.heads_kv,
                    head_dim: g.attn.head_dim,
                    cells: run.state.layout().kv.capacity,
                };
                let kv_width = g.attn.heads_kv * g.attn.head_dim;
                let slice = |slot: Slot, at: usize| -> Result<_, EngError> {
                    let view = self.view(slot)?;
                    view.buffer()
                        .range(view.offset() + at * F32, kv_width * F32)
                };
                ctx.kv_f16_append(
                    stream,
                    shape,
                    KvF16AppendData {
                        k: slice(k, k_at)?,
                        v: slice(v, v_at)?,
                        k_cache: run.state.kv_plane(run.ordinal, k_plane)?,
                        v_cache: run.state.kv_plane(run.ordinal, v_plane)?,
                    },
                    run.state.step_field(pos)?,
                    run.state.step_field(status)?,
                )
            }
        }
    }
}
