//! Where the persistent state of the resident step lives: byte offsets of the DeltaNet pool, the
//! KV pool and the persistent words, and the order of every element inside them (decision 0047
//! items 3 and 10).
//!
//! Pure host arithmetic; the device owner is `state.rs`. The orders below are the CONTRACT that
//! the DeltaNet kernels (RD-7a, RD-7b), the attention kernels (RD-8a, RD-8b, RD-8k) and
//! `export_state` / `import_state` share. They follow the GGUF semantics of llama.cpp
//! `acecd5603` (tiled key heads, `ssm_a` as stored, `silu(z)`), but two orders are the
//! TRANSPOSE of the reference ones (the kernels' own documentation, `gdn_delta.h` and
//! `gdn_conv.h`, says the same):
//!
//! - DeltaNet recurrent state, per layer, f32, element `(h, i, j)` of value head `h`, key row `i`,
//!   value column `j` at index `(h * key_dim + i) * value_dim + j` (the value column is the
//!   fastest). llama.cpp stores `state[h][j][i]` (the key row fastest). A llama.cpp state dump
//!   is the transpose of this layout: `llama[h][j][i] = state[h][i][j]`.
//! - DeltaNet convolution state, per layer, f32 `[tap][channel]` with `conv_kernel - 1` taps,
//!   oldest first, channel fastest. ggml and the host ring of the Q36 reference are
//!   `[channel][tap]`.
//! - KV, `Fp16`: per attention layer a K plane and a V plane, fp16 `[position][kv head][head_dim]`.
//! - KV, `Int8G32` (lossy; allocated from RD-8k): codes `i8 [kv head][cell][head_dim]` and
//!   scales fp16 `[kv head][cell][head_dim / 32]`, separate K and V planes, `cell = position`,
//!   `scale = amax / 127`.
//!
//! The kernel headers `gdn_delta.h` and `gdn_conv.h` say an import or export "transposes": that
//! means a conversion TO the reference order transposes. The export in this contract copies the
//! device order verbatim and never converts; no converter exists, so a llama.cpp or Q36-HR state
//! cannot be imported, and a transposing export would break the layout tag.
//!
//! # Persistent words
//!
//! Three small regions live in their own block, not in a step arena, because their address must be
//! the same for every captured graph of every row count and for the prefill path: `StepParams`
//! (RD-4d, 32 bytes), the result pair `{token, status}` (RD-4d, 8 bytes, 8-byte aligned; room
//! for 32 pairs) and the prefill status word (decision 0049, 4 bytes, bit 8). Each region is
//! `STATE_ALIGN` bytes apart, so a later word is appended at the end and never moves another.
//! `reset` zeroes the block, which is an invalid `StepParams` (`kv_len != pos + 1`): the host
//! writes a valid block before the first step.
//!
//! # Export contract (REPLY-43 item 4)
//!
//! The export carries the device orders above verbatim and the layout tag `LAYOUT_TAG`
//! (`qwen36-gguf-xq-1`, in `state.rs`) with the KV mode and the geometry. `import_state` refuses a
//! different tag, mode or geometry before the first device write. No code converts to the host
//! order of llama.cpp or of Q36-HR implicitly; such a conversion is a separate, explicit step.
//!
//! # Prefill and verify windows
//!
//! Prefill (decision 0049) has its own arena, sized by `family/qwen36/cuda/prefill_plan.rs`, with
//! no overlay on the decode arena. It writes the KV rows `[p0, p0 + M)` and the DeltaNet state of
//! slot 0 IN PLACE, through the offsets of this file, so nothing here changes when it lands. The
//! handoff to decode is: the committed token count `ResidentState::tokens()` equals the `pos` of
//! the first decode token, and the host writes `StepParams` with that `pos`; prefill reads and
//! writes no `StepParams` field. A verify window of M rows (MEL-PLAN 6.5, M up to 8) writes KV
//! rows `pos .. pos + M` before it knows how many are accepted; a rollback is a lower committed
//! count (the KV cache is never cleared) plus the DeltaNet slot the window needs (RD-7c).
//! `KvPool::capacity` is the `rows` argument of RD-4d's `ensure_rows`.
//!
//! Design guidance only, no code copied; the per-component sizing follows the idea of
//! MegaCapybara C4, and the slot-indexed DeltaNet pool is the addressing hook of C12
//! (research/megacapybara-re-lessons.md section 6.1).

use super::dims::{KvMode, StepDims, StepOptions};
use crate::{
    EngError,
    family::AttentionKind,
    kernels::cuda::resident::step_params::{RESULT_PAIR_BYTES, STEP_PARAMS_BYTES},
};

/// Offsets inside a state pool are multiples of this many bytes (the `cudaMalloc` alignment, as
/// in the arena planner).
pub(super) const STATE_ALIGN: usize = 256;
/// Values per fp16 scale in the `Int8G32` format.
pub(super) const KV8_GROUP: usize = 32;
/// Bytes per fp16 value or fp16 scale.
const HALF: usize = 2;

fn overflow() -> EngError {
    EngError::MemoryPlan("state layout: a size overflows".to_owned())
}

pub(super) fn align_up(value: usize) -> Option<usize> {
    value.checked_next_multiple_of(STATE_ALIGN)
}

fn aligned(value: usize) -> Result<usize, EngError> {
    align_up(value).ok_or_else(overflow)
}

fn mul(a: usize, b: usize) -> Result<usize, EngError> {
    a.checked_mul(b).ok_or_else(overflow)
}

fn add(a: usize, b: usize) -> Result<usize, EngError> {
    a.checked_add(b).ok_or_else(overflow)
}

/// The DeltaNet pool: `[slot][DeltaNet layer][recurrent | convolution]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaPool {
    pub(crate) slots: usize,
    pub(crate) layers: usize,
    /// f32 values of one layer's recurrent state and convolution state.
    pub(crate) rec_values: usize,
    pub(crate) conv_values: usize,
    rec_region: usize,
    layer_stride: usize,
    pub(crate) total: usize,
}

impl DeltaPool {
    fn new(dims: &StepDims, slots: usize) -> Result<Self, EngError> {
        let d = &dims.shape.delta;
        let layers = dims.layers_of(AttentionKind::DeltaNet);
        let rec_values = mul(mul(d.value_heads, d.key_dim)?, d.value_dim)?;
        let conv_values = mul(d.conv_kernel - 1, d.conv_dim())?;
        let rec_region = aligned(mul(rec_values, size_of::<f32>())?)?;
        let conv_region = aligned(mul(conv_values, size_of::<f32>())?)?;
        let layer_stride = add(rec_region, conv_region)?;
        let total = mul(mul(layer_stride, layers)?, slots)?;
        Ok(Self {
            slots,
            layers,
            rec_values,
            conv_values,
            rec_region,
            layer_stride,
            total,
        })
    }

    fn base(&self, slot: usize, ordinal: usize) -> Result<usize, EngError> {
        if slot >= self.slots || ordinal >= self.layers {
            return Err(EngError::BadRequest(format!(
                "state layout: DeltaNet slot {slot} layer {ordinal} is outside {} slots of {} layers",
                self.slots, self.layers
            )));
        }
        Ok((slot * self.layers + ordinal) * self.layer_stride)
    }

    /// Byte offset of the recurrent state of the `ordinal`-th DeltaNet layer in `slot`.
    pub(crate) fn rec_offset(&self, slot: usize, ordinal: usize) -> Result<usize, EngError> {
        self.base(slot, ordinal)
    }

    /// Byte offset of the convolution state of the same layer.
    pub(crate) fn conv_offset(&self, slot: usize, ordinal: usize) -> Result<usize, EngError> {
        Ok(self.base(slot, ordinal)? + self.rec_region)
    }
}

/// The four planes of one attention layer's KV cache (the scale planes are empty for `Fp16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Plane {
    K,
    KScales,
    V,
    VScales,
}

/// One contiguous run of a plane inside the KV pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyRange {
    pub(crate) plane: Plane,
    pub(crate) device_offset: usize,
    pub(crate) bytes: usize,
}

/// The KV pool: `[attention layer][K | K scales | V | V scales]`, sized for `capacity` positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvPool {
    pub(crate) mode: KvMode,
    pub(crate) capacity: usize,
    pub(crate) layers: usize,
    kv_heads: usize,
    head_dim: usize,
    plane_bytes: usize,
    scale_bytes: usize,
    layer_stride: usize,
    pub(crate) total: usize,
}

impl KvPool {
    fn new(dims: &StepDims, mode: KvMode, capacity: usize) -> Result<Self, EngError> {
        let a = &dims.shape.attn;
        if capacity == 0 {
            return Err(EngError::BadRequest(
                "state layout: zero KV positions".to_owned(),
            ));
        }
        let values = mul(mul(capacity, a.heads_kv)?, a.head_dim)?;
        let (plane, scale) = match mode {
            KvMode::Fp16 => (mul(values, HALF)?, 0),
            KvMode::Int8G32 => {
                if !a.head_dim.is_multiple_of(KV8_GROUP) {
                    return Err(EngError::Checkpoint(format!(
                        "state layout: head_dim {} is not a multiple of {KV8_GROUP}",
                        a.head_dim
                    )));
                }
                (values, mul(values / KV8_GROUP, HALF)?)
            }
        };
        let (plane_bytes, scale_bytes) = (aligned(plane)?, aligned(scale)?);
        let layer_stride = mul(add(plane_bytes, scale_bytes)?, 2)?;
        let layers = dims.layers_of(AttentionKind::Full);
        Ok(Self {
            mode,
            capacity,
            layers,
            kv_heads: a.heads_kv,
            head_dim: a.head_dim,
            plane_bytes,
            scale_bytes,
            layer_stride,
            total: mul(layer_stride, layers)?,
        })
    }

    /// KV bytes for one position over all attention layers (exact, not rounded to the alignment).
    pub(crate) fn bytes_per_token(dims: &StepDims, mode: KvMode) -> Result<usize, EngError> {
        let a = &dims.shape.attn;
        let width = mul(a.heads_kv, a.head_dim)?;
        let per_plane = match mode {
            KvMode::Fp16 => mul(width, HALF)?,
            KvMode::Int8G32 => add(width, mul(width / KV8_GROUP, HALF)?)?,
        };
        mul(mul(per_plane, 2)?, dims.layers_of(AttentionKind::Full))
    }

    fn plane_start(&self, ordinal: usize, plane: Plane) -> Result<usize, EngError> {
        if ordinal >= self.layers {
            return Err(EngError::BadRequest(format!(
                "state layout: attention layer {ordinal} is outside {} layers",
                self.layers
            )));
        }
        let within = match plane {
            Plane::K => 0,
            Plane::KScales => self.plane_bytes,
            Plane::V => self.plane_bytes + self.scale_bytes,
            Plane::VScales => 2 * self.plane_bytes + self.scale_bytes,
        };
        Ok(ordinal * self.layer_stride + within)
    }

    /// `(offset, bytes)` of one whole plane (all `capacity` positions, rounded up to the
    /// alignment) of the `ordinal`-th attention layer. Empty for a scale plane of `Fp16`.
    pub(crate) fn plane_extent(
        &self,
        ordinal: usize,
        plane: Plane,
    ) -> Result<(usize, usize), EngError> {
        let bytes = match plane {
            Plane::K | Plane::V => self.plane_bytes,
            Plane::KScales | Plane::VScales => self.scale_bytes,
        };
        Ok((self.plane_start(ordinal, plane)?, bytes))
    }

    /// Planes the mode stores, in export order.
    pub(crate) fn planes(&self) -> &'static [Plane] {
        match self.mode {
            KvMode::Fp16 => &[Plane::K, Plane::V],
            KvMode::Int8G32 => &[Plane::K, Plane::KScales, Plane::V, Plane::VScales],
        }
    }

    /// `(runs, stride between runs, bytes per run)` of the first `tokens` positions of a plane.
    /// An `Fp16` plane is position-major, so it is one run. An `Int8G32` plane is head-major, so
    /// it is one run per KV head, `capacity` cells apart.
    fn plane_runs(&self, plane: Plane, tokens: usize) -> Result<(usize, usize, usize), EngError> {
        let width = mul(self.kv_heads, self.head_dim)?;
        let groups = self.head_dim / KV8_GROUP;
        Ok(match (self.mode, plane) {
            (KvMode::Fp16, _) => (1, 0, mul(mul(tokens, width)?, HALF)?),
            (KvMode::Int8G32, Plane::K | Plane::V) => (
                self.kv_heads,
                mul(self.capacity, self.head_dim)?,
                mul(tokens, self.head_dim)?,
            ),
            (KvMode::Int8G32, _) => (
                self.kv_heads,
                mul(mul(self.capacity, groups)?, HALF)?,
                mul(mul(tokens, groups)?, HALF)?,
            ),
        })
    }

    /// The runs that hold the first `tokens` positions of the `ordinal`-th attention layer,
    /// plane by plane. Export concatenates the runs of a plane, which makes the exported planes
    /// independent of the capacity.
    pub(crate) fn prefix_ranges(
        &self,
        ordinal: usize,
        tokens: usize,
    ) -> Result<Vec<CopyRange>, EngError> {
        if tokens > self.capacity {
            return Err(EngError::BadRequest(format!(
                "state layout: {tokens} positions exceed the capacity {}",
                self.capacity
            )));
        }
        let mut ranges = Vec::new();
        for &plane in self.planes() {
            let start = self.plane_start(ordinal, plane)?;
            let (runs, stride, bytes) = self.plane_runs(plane, tokens)?;
            for run in 0..runs {
                ranges.push(CopyRange {
                    plane,
                    device_offset: start + run * stride,
                    bytes,
                });
            }
        }
        Ok(ranges)
    }
}

/// A persistent device word (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Word {
    /// RD-4d `StepParams`.
    StepParams,
    /// RD-4d result pair `{token, status}`; the host copies these 8 bytes after a step.
    ResultPair,
    /// Decision 0049 prefill status word; the first prefill kernel of a chunk zeroes it.
    PrefillStatus,
}

// The merged ABI: the pair must be 8-aligned in its slot and every word must fit one slot.
const _: () = assert!(
    Word::StepParams.bytes() == 32 && Word::ResultPair.bytes() == 8,
    "the words block is laid out for StepParams of 32 bytes and a result pair of 8 bytes"
);
const _: () = assert!(
    Word::StepParams.bytes() <= STATE_ALIGN
        && Word::ResultPair.bytes() <= STATE_ALIGN
        && Word::PrefillStatus.bytes() <= STATE_ALIGN
        && Word::ResultPair.offset().is_multiple_of(8)
);

impl Word {
    /// Bytes the kernels use: RD-4d's `STEP_PARAMS_BYTES` (32) and `RESULT_PAIR_BYTES` (8), taken
    /// from the merged wrapper, and one i32 for the prefill status.
    pub(crate) const fn bytes(self) -> usize {
        match self {
            Self::StepParams => STEP_PARAMS_BYTES,
            Self::ResultPair => RESULT_PAIR_BYTES,
            Self::PrefillStatus => 4,
        }
    }

    /// Byte offset in the words block: one `STATE_ALIGN` slot per word, in declaration order.
    pub(crate) const fn offset(self) -> usize {
        match self {
            Self::StepParams => 0,
            Self::ResultPair => STATE_ALIGN,
            Self::PrefillStatus => 2 * STATE_ALIGN,
        }
    }
}

/// The block of persistent words: the end of the last word's slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WordsBlock {
    pub(crate) total: usize,
}

impl WordsBlock {
    fn new() -> Self {
        Self {
            total: Word::PrefillStatus.offset() + STATE_ALIGN,
        }
    }
}

/// All persistent state of one session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateLayout {
    pub(crate) delta: DeltaPool,
    pub(crate) kv: KvPool,
    pub(crate) words: WordsBlock,
}

impl StateLayout {
    /// Device bytes of the pools and the words, the state term of the admission sum (decision
    /// 0047 item 6). The decode arena (`StepPlan::arena_bytes`), the workspace (`Admission`) and
    /// a prefill arena are separate terms.
    pub(crate) fn bytes(&self) -> Result<usize, EngError> {
        add(add(self.delta.total, self.kv.total)?, self.words.total)
    }

    pub(crate) fn new(dims: &StepDims, options: &StepOptions) -> Result<Self, EngError> {
        if options.delta_slots == 0 {
            return Err(EngError::BadRequest(
                "state layout: zero DeltaNet slots".to_owned(),
            ));
        }
        Ok(Self {
            delta: DeltaPool::new(dims, options.delta_slots)?,
            kv: KvPool::new(dims, options.kv_mode, options.context_tokens)?,
            words: WordsBlock::new(),
        })
    }
}
