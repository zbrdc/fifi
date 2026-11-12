//! The persistent device state of the Flash-Next CUDA step: what survives from one token to the
//! next, its sizes, and the arena that holds it.
//!
//! Pure host arithmetic; the device owner is a later slice. One [`StatePlan`] is a list of
//! [`ArenaTensor`]s for [`crate::kernels::cuda::CudaContext::arena`], so the state is ONE
//! allocation planned by `plan_arena`, with checked views. Every state tensor lives for the whole
//! session, so the planner packs them end to end and aliases nothing.
//!
//! # What is state
//!
//! - DeltaNet layers (36 of 48): the recurrent state f32 `[value_heads][key_dim][value_dim]` (the
//!   layout of `gdn_scan_seq`, value column fastest) and the conv ring, planar f32
//!   `[conv_kernel - 1][conv_dim]`, oldest row first (the layout of `gdn_conv`). The host ring of
//!   `deltanet.rs` is `[channel][tap]`, the transpose; the byte count is the same, so the host
//!   `DeltaState::byte_len` is the size oracle and an export must transpose the ring. The conv
//!   ring row order is oldest first.
//! - QSA layers (12 of 48): the fp16 K and V caches `[position][kv head][head_dim]` (the layout
//!   `qsa_attn` reads), the pooled indexer keys f32 `[cap_blocks][idx_dim]` (one per completed
//!   block of `ratio` positions) and the raw-key ring f32 `[ratio][idx_dim]` of the open block.
//!   The raw ring is indexed by `position % ratio` (`qsa_index.h`), and the key cache holds
//!   `ceil(positions / ratio)` blocks, so `cap_blocks * ratio >= positions` (the index kernel
//!   admits `position + m <= cap_blocks * ratio`).
//! - The PLE conv ring f32 `[hyper width][conv_kernel - 1][ngram]` (host layout
//!   `PleState::ring`; the token history stays on the host).
//! - The words block: `StepParams`, the result pair and a prefill status word, one 256-byte slot
//!   each, at the offsets of Qwen3.6's `state_layout::Word` (widened to `crate::family`, used
//!   here so the two families share one contract).
//!
//! The hyper stream (4 x 2560 f32 per row) is an activation, not state: every token starts it from
//! the embedding. It is in `step_plan.rs`.
//!
//! # K/V format
//!
//! The format is the context budget's own [`KvFormat`] (`moe/context_budget.rs`, widened to
//! `flash_next` visibility for this slice), not a type of this module. Element widths and per
//! position bytes come from it and from [`KvGeometry`], so the state plan and the budget cannot
//! disagree. [`StatePlan::new`] refuses a format without a QSA kernel ([`KvFormat::has_kernels`]):
//! today only `F16`, which `qsa_attn` reads.
//!
//! # Names shared with the context budget
//!
//! [`StateBytes::bytes_per_position`] is `KvGeometry::position_bytes(format)` (26,112 bytes for
//! `F16`) and equals the planned K/V plus indexer bytes per position when the context is a multiple
//! of the indexer ratio, and [`StateBytes::delta_state_bytes`]
//! equals `FlashLoadPlan::delta_state_bytes`. The fixed state that the budget does not charge yet is
//! [`StateBytes::other_fixed_bytes`].

pub(super) use super::driver::StateSlot;
use crate::{
    EngError,
    family::{
        AttentionKind,
        flash_next::{
            moe::context_budget::{KvFormat, KvGeometry},
            step::ModelShape,
        },
        qwen36::cuda::{book::StateBook, state_layout::Word},
    },
    kernels::{
        ARENA_ALIGN,
        cuda::{
            ArenaPlan, ArenaTensor, plan_arena,
            resident::{
                gdn_conv::{ConvShape, conv_bytes},
                step_params::POS_MAX,
            },
        },
    },
};

const F32: usize = size_of::<f32>();

/// Head size of the QSA attention kernel (`QSA_ATTN_HEAD`).
const QSA_HEAD_DIM: usize = 256;
/// Head size of the Gated DeltaNet scan (`GDN_SCAN_HEAD_DIM`).
const GDN_HEAD_DIM: usize = 128;
/// Limits of the QSA kernels (`qsa_index.h`, `qsa_attn.h`) the geometry is checked against, so a
/// shape the kernels refuse fails at plan time.
const QSA_MAX_IDX_DIM: usize = 1024;
const QSA_MAX_QUERY_FLOATS: usize = 8192;
const QSA_MAX_RATIO: usize = 64;
const QSA_MAX_GROUP: usize = 32;
/// Conv taps of the resident conv kernel; its ring holds `GDN_CONV_KERNEL - 1` input rows.
const GDN_CONV_KERNEL: usize = 4;
/// The words block: one arena-aligned slot per word, at the offsets of `Word`.
pub(super) const WORDS_BYTES: usize = Word::PrefillStatus.offset() + ARENA_ALIGN;
/// Byte offsets of the words inside the words tensor.
pub(super) const WORD_STEP_PARAMS: usize = Word::StepParams.offset();
pub(super) const WORD_RESULT_PAIR: usize = Word::ResultPair.offset();
pub(super) const WORD_PREFILL_STATUS: usize = Word::PrefillStatus.offset();

pub(super) fn overflow() -> EngError {
    EngError::MemoryPlan("Flash-Next state plan: a size overflows".to_owned())
}

pub(super) fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |total, &factor| total.checked_mul(factor))
        .ok_or_else(overflow)
}

pub(super) fn sum(terms: &[usize]) -> Result<usize, EngError> {
    terms
        .iter()
        .try_fold(0usize, |total, &term| total.checked_add(term))
        .ok_or_else(overflow)
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next state plan: {detail}"))
}

/// Gated DeltaNet dimensions the device state needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeltaGeometry {
    pub(super) key_heads: usize,
    pub(super) key_dim: usize,
    pub(super) value_heads: usize,
    pub(super) value_dim: usize,
    /// Channels of the conv: `2 * key_heads * key_dim + value_heads * value_dim`.
    pub(super) conv_dim: usize,
}

impl DeltaGeometry {
    pub(super) fn value_width(&self) -> usize {
        self.value_heads * self.value_dim
    }

    /// Width of one projection row the scan reads: qkv, then z, then beta and alpha
    /// (one value per value head each).
    pub(super) fn proj_ld(&self) -> usize {
        self.conv_dim + self.value_width() + 2 * self.value_heads
    }

    /// Bytes of the recurrent state of one layer.
    fn rec_bytes(&self) -> Result<usize, EngError> {
        product(&[self.value_heads, self.key_dim, self.value_dim, F32])
    }

    /// Bytes of the conv ring of one layer, from the conv kernel's own sizing.
    fn conv_ring_bytes(&self) -> Result<usize, EngError> {
        Ok(conv_bytes(ConvShape {
            m: 1,
            channels: self.conv_dim,
            ld: self.conv_dim,
        })?
        .state)
    }
}

/// QSA dimensions. The first five names are the fields of the context budget's `KvGeometry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QsaGeometry {
    /// Layers that cache K/V and indexer keys (12 of 48).
    pub(super) cached_layers: usize,
    pub(super) heads_kv: usize,
    pub(super) head_dim: usize,
    /// Width of one f32 pooled indexer key.
    pub(super) idx_dim: usize,
    /// Positions pooled into one indexer key.
    pub(super) ratio: usize,
    pub(super) heads_q: usize,
    pub(super) idx_heads: usize,
    /// Positions the indexer may select; the selection holds `budget / ratio` blocks.
    pub(super) budget: usize,
}

impl QsaGeometry {
    pub(super) fn q_width(&self) -> usize {
        self.heads_q * self.head_dim
    }

    pub(super) fn kv_width(&self) -> usize {
        self.heads_kv * self.head_dim
    }

    pub(super) fn idx_proj(&self) -> usize {
        (self.idx_heads + 1) * self.idx_dim
    }

    /// Width of one projection row: per query head `[q | gate]`, then k, then v, then the indexer
    /// row (queries, then the raw key).
    pub(super) fn proj_ld(&self) -> usize {
        2 * self.q_width() + 2 * self.kv_width() + self.idx_proj()
    }

    /// Blocks the selection may name.
    pub(super) fn budget_blocks(&self) -> usize {
        self.budget / self.ratio
    }

    /// Indexer key blocks that hold `positions` positions.
    pub(super) fn cap_blocks(&self, positions: usize) -> usize {
        positions.div_ceil(self.ratio)
    }

    /// Bytes of one K (or V) plane of one layer.
    fn plane_bytes(&self, positions: usize, format: KvFormat) -> Result<usize, EngError> {
        let element = usize::try_from(format.element_bytes()).map_err(|_| overflow())?;
        product(&[positions, self.heads_kv, self.head_dim, element])
    }

    /// The geometry in the context budget's names.
    pub(super) fn kv_geometry(&self) -> Result<KvGeometry, EngError> {
        let wide = |value: usize| u64::try_from(value).map_err(|_| overflow());
        Ok(KvGeometry {
            cached_layers: wide(self.cached_layers)?,
            heads_kv: wide(self.heads_kv)?,
            head_dim: wide(self.head_dim)?,
            idx_dim: wide(self.idx_dim)?,
            ratio: wide(self.ratio)?,
        })
    }

    fn index_key_bytes(&self, positions: usize) -> Result<usize, EngError> {
        product(&[self.cap_blocks(positions), self.idx_dim, F32])
    }

    fn raw_ring_bytes(&self) -> Result<usize, EngError> {
        product(&[self.ratio, self.idx_dim, F32])
    }
}

/// The model facts the state and the step plan read, taken from the host reference's
/// [`ModelShape`] and the per-layer kinds of `load::model_shape`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FlashGeometry {
    /// One entry per layer: [`AttentionKind::Qsa`] or [`AttentionKind::DeltaNet`].
    pub(super) layers: Vec<AttentionKind>,
    pub(super) hidden: usize,
    pub(super) branches: usize,
    pub(super) rank: usize,
    pub(super) delta: DeltaGeometry,
    pub(super) qsa: QsaGeometry,
    pub(super) experts: usize,
    pub(super) top_k: usize,
    pub(super) inter: usize,
    pub(super) shared_inter: usize,
    pub(super) vocab: usize,
    /// Width of the PLE conv ring: `[hyper width][conv_kernel - 1][ngram]` values.
    pub(super) ple_ring_floats: usize,
    /// Width of the PLE embedding row the feeds supply (`heads * head_dim`).
    pub(super) ple_embed: usize,
    /// Layer whose attention read follows the PLE update.
    pub(super) ple_layer: usize,
}

impl FlashGeometry {
    /// `kinds[layer]` is true for a QSA layer, as `load::model_shape` returns it. Refuses a shape
    /// the kernels do not take: DeltaNet heads of other than 128 values, a conv kernel other than
    /// 4, QSA heads of other than 256 values.
    pub(super) fn new(shape: &ModelShape, kinds: &[bool]) -> Result<Self, EngError> {
        let delta = &shape.delta;
        let dims = shape.qsa.dims();
        let ple = shape.ple.dims();
        let to_usize = |value: u32| usize::try_from(value).map_err(|_| overflow());
        if delta.key_dim() != GDN_HEAD_DIM || delta.value_dim() != GDN_HEAD_DIM {
            return Err(refuse(format!(
                "the DeltaNet scan takes heads of {GDN_HEAD_DIM} values, got key {} and value {}",
                delta.key_dim(),
                delta.value_dim()
            )));
        }
        if delta.conv_kernel() != GDN_CONV_KERNEL {
            return Err(refuse(format!(
                "the resident conv takes a kernel of {GDN_CONV_KERNEL}, got {}",
                delta.conv_kernel()
            )));
        }
        if to_usize(dims.head_dim)? != QSA_HEAD_DIM {
            return Err(refuse(format!(
                "the QSA attention takes heads of {QSA_HEAD_DIM} values, got {}",
                dims.head_dim
            )));
        }
        let (idx_dim, ratio) = (to_usize(dims.idx_dim)?, to_usize(dims.ratio)?);
        if !(32..=QSA_MAX_IDX_DIM).contains(&idx_dim)
            || !idx_dim.is_multiple_of(32)
            || to_usize(dims.idx_heads)? * idx_dim > QSA_MAX_QUERY_FLOATS
            || !(1..=QSA_MAX_RATIO).contains(&ratio)
            || to_usize(dims.heads_q)? / to_usize(dims.heads_kv)? > QSA_MAX_GROUP
        {
            return Err(refuse(format!(
                "the QSA kernels take an indexer head of a multiple of 32 up to {QSA_MAX_IDX_DIM}, \
                 at most {QSA_MAX_QUERY_FLOATS} query values, a ratio of 1 to {QSA_MAX_RATIO} and \
                 at most {QSA_MAX_GROUP} query heads per kv head, got idx_dim {idx_dim}, \
                 idx_heads {}, ratio {ratio}, heads_q {}, heads_kv {}",
                dims.idx_heads, dims.heads_q, dims.heads_kv
            )));
        }
        if dims.ratio == 0 || !(to_usize(dims.idx_dim)? * F32).is_multiple_of(to_usize(dims.ratio)?)
        {
            return Err(refuse(format!(
                "the indexer key of {} f32 values does not divide over {} positions",
                dims.idx_dim, dims.ratio
            )));
        }
        if shape.ple_layer >= kinds.len() {
            return Err(refuse(format!(
                "the PLE layer {} is outside the {} layers",
                shape.ple_layer,
                kinds.len()
            )));
        }
        let layers: Vec<AttentionKind> = kinds
            .iter()
            .map(|&qsa| {
                if qsa {
                    AttentionKind::Qsa
                } else {
                    AttentionKind::DeltaNet
                }
            })
            .collect();
        let cached_layers = layers
            .iter()
            .filter(|&&kind| kind == AttentionKind::Qsa)
            .count();
        let ple_heads = product(&[to_usize(ple.ngram - 1)?, to_usize(ple.heads_per_ngram)?])?;
        let hyper_width = shape.hyper.width();
        Ok(Self {
            layers,
            hidden: shape.hyper.hidden(),
            branches: shape.hyper.branches(),
            rank: shape.hyper.rank(),
            delta: DeltaGeometry {
                key_heads: delta.key_heads(),
                key_dim: delta.key_dim(),
                value_heads: delta.value_heads(),
                value_dim: delta.value_dim(),
                conv_dim: delta.conv_dim(),
            },
            qsa: QsaGeometry {
                cached_layers,
                heads_kv: to_usize(dims.heads_kv)?,
                head_dim: to_usize(dims.head_dim)?,
                idx_dim: to_usize(dims.idx_dim)?,
                ratio: to_usize(dims.ratio)?,
                heads_q: to_usize(dims.heads_q)?,
                idx_heads: to_usize(dims.idx_heads)?,
                budget: to_usize(dims.budget)?,
            },
            experts: shape.moe.experts(),
            top_k: shape.moe.top_k(),
            inter: shape.moe.inter(),
            shared_inter: shape.moe.shared_inter(),
            vocab: shape.vocab,
            ple_ring_floats: product(&[
                hyper_width,
                to_usize(ple.conv_kernel - 1)?,
                to_usize(ple.ngram)?,
            ])?,
            ple_embed: product(&[ple_heads, to_usize(ple.head_dim)?])?,
            ple_layer: shape.ple_layer,
        })
    }

    /// Width `branches * hidden` of the hyper stream.
    pub(super) fn width(&self) -> usize {
        self.branches * self.hidden
    }

    pub(super) fn delta_layers(&self) -> usize {
        self.layers.len() - self.qsa.cached_layers
    }

    /// Bytes of the PLE conv ring.
    fn ple_ring_bytes(&self) -> Result<usize, EngError> {
        product(&[self.ple_ring_floats, F32])
    }
}

/// The fixed and the context-dependent bytes of a state plan, in the names the context budget uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StateBytes {
    pub(super) positions: usize,
    /// Indexer key blocks per QSA layer.
    pub(super) cap_blocks: usize,
    /// Recurrent state and conv ring of every DeltaNet layer
    /// (`FlashLoadPlan::delta_state_bytes`).
    pub(super) delta_state_bytes: usize,
    /// Raw-key rings, the PLE ring and the words block: fixed, and not yet charged by the budget.
    pub(super) other_fixed_bytes: usize,
    /// fp16 K and V planes of every QSA layer, `positions` rows.
    pub(super) kv_bytes: usize,
    /// Pooled indexer keys of every QSA layer.
    pub(super) index_key_bytes: usize,
    /// K/V plus indexer keys per position (`KvPlan::bytes_per_position`); exact when `positions`
    /// is a multiple of the ratio.
    pub(super) bytes_per_position: usize,
    /// The arena total, which is the sum of the four parts plus alignment padding (none for the
    /// real shape).
    pub(super) arena_bytes: usize,
}

impl StateBytes {
    /// The fixed part of the state: everything that does not grow with the context.
    pub(super) fn fixed_bytes(&self) -> usize {
        self.delta_state_bytes + self.other_fixed_bytes
    }

    /// The part that grows with the context.
    pub(super) fn context_bytes(&self) -> usize {
        self.kv_bytes + self.index_key_bytes
    }
}

/// The planned state of a session of `positions` context positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StatePlan {
    geometry: FlashGeometry,
    positions: usize,
    format: KvFormat,
    slots: Vec<(StateSlot, usize)>,
    arena: ArenaPlan,
}

impl StatePlan {
    /// `positions` is the context the K/V and indexer caches hold; the arena never grows.
    pub(super) fn new(
        geometry: &FlashGeometry,
        positions: usize,
        format: KvFormat,
    ) -> Result<Self, EngError> {
        let qsa = &geometry.qsa;
        if !format.has_kernels() {
            return Err(refuse(format!(
                "{format:?} K/V has no attention kernel yet"
            )));
        }
        if positions == 0 || positions > POS_MAX as usize {
            return Err(refuse(format!(
                "a context of {positions} positions is outside 1..={POS_MAX}"
            )));
        }
        let mut slots = Vec::new();
        let rec = geometry.delta.rec_bytes()?;
        let ring = geometry.delta.conv_ring_bytes()?;
        let plane = qsa.plane_bytes(positions, format)?;
        let keys = qsa.index_key_bytes(positions)?;
        let raw = qsa.raw_ring_bytes()?;
        for (layer, kind) in geometry.layers.iter().enumerate() {
            match kind {
                AttentionKind::DeltaNet => {
                    slots.push((StateSlot::DeltaRec(layer), rec));
                    slots.push((StateSlot::DeltaConv(layer), ring));
                }
                AttentionKind::Qsa => {
                    slots.push((StateSlot::QsaK(layer), plane));
                    slots.push((StateSlot::QsaV(layer), plane));
                    slots.push((StateSlot::QsaIndexKeys(layer), keys));
                    slots.push((StateSlot::QsaRawRing(layer), raw));
                }
                AttentionKind::Full => {
                    return Err(refuse(format!(
                        "layer {layer} is dense attention, which Flash-Next does not have"
                    )));
                }
            }
        }
        slots.push((StateSlot::PleRing, geometry.ple_ring_bytes()?));
        slots.push((StateSlot::Words, WORDS_BYTES));
        let arena = plan_arena(&Self::arena_tensors_of(&slots))?;
        Ok(Self {
            geometry: geometry.clone(),
            positions,
            format,
            slots,
            arena,
        })
    }

    fn arena_tensors_of(slots: &[(StateSlot, usize)]) -> Vec<ArenaTensor> {
        // Persistent: live for the whole session, so nothing aliases.
        slots
            .iter()
            .map(|&(_, bytes)| ArenaTensor {
                bytes,
                first: 0,
                last: 0,
            })
            .collect()
    }

    /// The tensors to hand to `CudaContext::arena`, in slot order.
    pub(super) fn arena_tensors(&self) -> Vec<ArenaTensor> {
        Self::arena_tensors_of(&self.slots)
    }

    pub(super) fn positions(&self) -> usize {
        self.positions
    }

    pub(super) fn slots(&self) -> &[(StateSlot, usize)] {
        &self.slots
    }

    /// The tensor index of `slot` in the arena (`DeviceArena::tensor`).
    pub(super) fn index(&self, slot: StateSlot) -> Option<usize> {
        self.slots.iter().position(|&(found, _)| found == slot)
    }

    /// Whether `slot` is position state a resume leaves in place under the valid length (the
    /// QSA K/V planes and the indexer keys): no kernel reads a position at or past the valid
    /// length, so a fresh sequence never observes them. The raw-key ring is NOT position state:
    /// slot `p % ratio` accumulates the open block's raw keys, so the saving job's own
    /// continuation overwrites the slots a mid-block boundary needs — the ring is copied with
    /// the fixed state (saved, restored and zeroed on reset).
    pub(super) fn is_position_state(slot: StateSlot) -> bool {
        matches!(
            slot,
            StateSlot::QsaK(_) | StateSlot::QsaV(_) | StateSlot::QsaIndexKeys(_)
        )
    }

    /// `(offset, bytes)` of `slot` in the arena.
    pub(super) fn placement(&self, slot: StateSlot) -> Option<(usize, usize)> {
        let index = self.index(slot)?;
        Some((self.arena.offsets[index], self.slots[index].1))
    }

    pub(super) fn arena_bytes(&self) -> usize {
        self.arena.total
    }

    /// The bookkeeping book for this state: its capacity is the context, `owner` the id of the
    /// stream that owns the state (`DeviceStream::id`). The book starts `Poisoned` until the first
    /// reset, as in Qwen3.6.
    pub(super) fn new_book(&self, owner: u64) -> StateBook {
        StateBook::new(self.positions, owner)
    }

    /// The bytes by part, in the names of the context budget.
    pub(super) fn bytes(&self) -> Result<StateBytes, EngError> {
        let qsa = &self.geometry.qsa;
        let (mut delta, mut other, mut kv, mut keys) = (0usize, 0usize, 0usize, 0usize);
        for &(slot, bytes) in &self.slots {
            let part = match slot {
                StateSlot::DeltaRec(_) | StateSlot::DeltaConv(_) => &mut delta,
                StateSlot::QsaK(_) | StateSlot::QsaV(_) => &mut kv,
                StateSlot::QsaIndexKeys(_) => &mut keys,
                StateSlot::QsaRawRing(_) | StateSlot::PleRing | StateSlot::Words => &mut other,
            };
            *part = part.checked_add(bytes).ok_or_else(overflow)?;
        }
        let per_position = usize::try_from(qsa.kv_geometry()?.position_bytes(self.format)?)
            .map_err(|_| overflow())?;
        Ok(StateBytes {
            positions: self.positions,
            cap_blocks: qsa.cap_blocks(self.positions),
            delta_state_bytes: delta,
            other_fixed_bytes: other,
            kv_bytes: kv,
            index_key_bytes: keys,
            bytes_per_position: per_position,
            arena_bytes: self.arena.total,
        })
    }
}
