//! The resident Nemotron Bolt E4B decode step on one CUDA device.
//!
//! This module composes the checked resident operations over the converted Bolt
//! Q8_0 checkpoint. Each of the forty-two decoder layers runs three residual
//! blocks: attention over the KV8 planes, the dense GeLU-tanh feed-forward and
//! the per-layer-input block. The last eighteen layers hold no k/v projections
//! and no cache planes; their attention reads the cache of the last physical
//! layer of their own kind (layer 22 for sliding, 23 for full), so only the
//! twenty-four physical layers consume cache memory.
//!
//! The per-layer inputs are computed once per step from the scaled embedding:
//! the model projection times hidden^-0.5, one 256-wide RMSNorm slice per
//! layer, plus sixteen times the token's per-layer embedding row, all scaled by
//! 2^-0.5. The step ends with the final RMSNorm, the tied-embedding logit
//! projection and the logit softcap.
//!
//! Limits:
//! - The four full-attention layers hold the planned number of cells (up to the model's 65,536
//!   maximum positions), chosen from the free card at load; each sliding layer holds a ring of
//!   the window plus one prefill chunk, whatever the context.
//! - Prefill runs bounded batches through the batched step.
//! - The per-head RMSNorm runs one batched launch across heads.
//! - Shared layers rotate one scratch k row, because the rope kernel requires
//!   at least one; the row is never read.
//! - A decode step's capped logits are read back to the host; host sampling
//!   draws from that row.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "native-cuda resident helpers beyond the serving path"
    )
)]

use super::bolt_graph::{
    DecodeGraphFacts, DecodeGraphMode, check_decode_graph, decode_graph_wanted, parse_switch,
};
use super::step::layer_buffers;
use super::{
    Kv, Vector, epsilon, matrix, matrix_from_bytes, matvec, scalar, vector, vector_filled, zero,
};
use crate::EngError;
use crate::family::card_fit::CellFigures;
use crate::family::cell_pool::{CellLease, CellPool, fit_planes, fund_growth};
use crate::family::gemma::gemma4::gguf::ple_rows_from_merged;
use crate::family::kv_ring::KvRing;
use crate::family::spec::{AttentionKind, NEMOTRON_BOLT_E4B};
use crate::kernels::cuda::DeviceRange;
use crate::kernels::cuda::prefill::attn_prefill::{
    WideAttention, WideCall, WideGeometry, WideLoad,
};
use crate::kernels::cuda::prefill::dense_chunk_gemm::DenseChunkGemm;
use crate::kernels::cuda::prefill::mmq_dense::takes;
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::attn_decode_gemma::{
    AttnDecodeGemmaCall, AttnDecodeGemmaSplit,
};
use crate::kernels::cuda::resident::embed::EmbedNativeCall;
use crate::kernels::cuda::resident::gelu_ffn::GeluFfnCall;
use crate::kernels::cuda::resident::kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::resident::multi_row::{
    AppendMultiCall, DecodeGemmaMultiCall, KvRow, KvRows, ROW_RECORD_BYTES, STEP_BLOCK_BYTES,
};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::resident::rope_gemma::RopeGemmaCall;
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::resident::row_scale::RowScaleCall;
use crate::kernels::cuda::resident::shortlist;
use crate::kernels::cuda::resident::softcap::SoftcapCall;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceStream, GraphExec, is_capture_refusal};
use crate::kernels::cuda::{DeviceBuffer, NativeWeight};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Geometry is read from the family descriptor, the single source of truth;
// the executor pins only the cache cell count and the checkpoint properties
// the descriptor does not carry (PLE width, soft cap).
pub(crate) const LAYERS: usize = NEMOTRON_BOLT_E4B.layers as usize;
/// The physical k/v layers; the shared layers attend into their caches.
pub(crate) const PHYSICAL_LAYERS: usize =
    (NEMOTRON_BOLT_E4B.layers - NEMOTRON_BOLT_E4B.kv_shared_layers) as usize;
/// Values per hidden row.
pub(crate) const HIDDEN: usize = NEMOTRON_BOLT_E4B.hidden as usize;
/// Rows of the tied embedding and of the logit projection.
pub(crate) const VOCAB: usize = NEMOTRON_BOLT_E4B.vocab as usize;
pub(crate) const HEADS: usize = NEMOTRON_BOLT_E4B.heads_q as usize;
pub(crate) const KV_HEADS: usize = NEMOTRON_BOLT_E4B.heads_kv as usize;
pub(crate) const LOCAL_HEAD: usize = NEMOTRON_BOLT_E4B.head_dim as usize;
pub(crate) const GLOBAL_HEAD: usize = NEMOTRON_BOLT_E4B.global_head_dim as usize;
/// Width of one per-layer input and of the gated mixing inside every layer.
pub(crate) const PLE: usize = 256;
pub(crate) const FFN: usize = 10_240;
/// The fewest prompt rows of a chunk that run on the tensor cores; a shorter chunk keeps the matvec.
const MMQ_MIN_ROWS: usize = 16;
/// The widest activation a chunk projects.
const MMQ_K_MAX: usize = FFN;

/// Whether a group of projections over one activation runs on the tensor cores: the chunk is wide
/// enough and every weight of the group rests in one type the dense product takes by default, since
/// one quantized activation serves one layout.
fn on_tensor_cores(mut types: impl Iterator<Item = QuantType>, m: usize) -> bool {
    let Some(first) = types.next() else {
        return false;
    };
    m >= MMQ_MIN_ROWS && takes(first) && types.all(|ty| ty == first)
}
/// The model's maximum positions: the ceiling of the planned KV cells per physical layer.
pub(crate) const MAX_CELLS: usize = match NEMOTRON_BOLT_E4B.feed_forward {
    Some(dense) => dense.context_limit as usize,
    None => 0,
};
/// The cell count a refusal is worded for when the card carries no planned count.
const FALLBACK_CELLS: usize = 32_768;
pub(crate) const SOFT_CAP: f32 = 30.0;
/// The identity and mixing scales applied to the per-layer inputs.
const PLE_IDENTITY_SCALE: f32 = 16.0;
const PLE_COMBINED_SCALE: f32 = std::f32::consts::SQRT_2.recip();

/// The device bytes the scratch can hold at once: every growable activation
/// vector and q8 buffer sized for the planned prefill rows (the maximum
/// `grow_scratch` allocates), plus the fixed ones/logits/head/qk buffers and
/// the three input cells. The starter admits this against free memory before
/// loading. `None` on overflow. Mirrors `grow_scratch` so the charge cannot
/// undercount the prefill working set. The attention split's partial buffer
/// is charged only when the split is armed, at the device's split count (the split is the default since it
/// landed; the admission must refuse an armed load before allocating), and the wide
/// prefill attention's only when `wide_prefill` (the device gate passed).
pub(crate) fn scratch_bytes(splits: Option<usize>, wide: bool, rows: usize) -> Option<u64> {
    let rows64 = rows as u64;
    let growable_f32 = [
        8 * HIDDEN,                 // h[4], nrm, nrm2, proj, down
        3 * HEADS * GLOBAL_HEAD,    // q_normed, q, ctx_out
        4 * KV_HEADS * GLOBAL_HEAD, // k_normed, v_normed, k, v
        3 * FFN,                    // gate, up, mixed
        2 * LAYERS * PLE,           // ple_all, ple_scaled
        3 * PLE,                    // ple_nrm, ple_tbl, ple_mixed
    ];
    let mut total = 0u64;
    for elements in growable_f32 {
        total = total.checked_add((elements as u64).checked_mul(rows64)?.checked_mul(4)?)?;
    }
    // Fixed f32 buffers: ones, logits, logits_capped.
    total = total.checked_add(((GLOBAL_HEAD + 2 * VOCAB) as u64).checked_mul(4)?)?;
    // Growable q8 buffers at the planned prefill columns.
    for width in [HIDDEN, HEADS * GLOBAL_HEAD, FFN, PLE] {
        total = total.checked_add(q8_1_bytes(width, rows)? as u64)?;
    }
    // The q8 buffers sized once: the head rows and the qk dump.
    total = total.checked_add(q8_1_bytes(GLOBAL_HEAD, rows)? as u64)?;
    total = total.checked_add(q8_1_bytes(GLOBAL_HEAD, rows * HEADS)? as u64)?;
    // ids, pos, kv_len input cells.
    total = total.checked_add(12)?;
    // The split partial buffer, present only when the attention split is armed.
    if let Some(splits) = splits {
        total = total.checked_add(attn_part_bytes(splits, rows) as u64)?;
    }
    // The wide prefill attention's partial buffer, present only when the device gate passed; the
    // buffer holds at most 128 splits, so the model maximum bounds it for every planned cell count.
    total = total.checked_add(wide_geometry(MAX_CELLS).charge(wide)?)?;
    total.checked_add(DenseChunkGemm::bytes(MMQ_K_MAX, rows)?)
}

/// The parity dump directory, read from BOLT_TRACE_DIR once. The trace hook
/// runs per checkpoint per token; caching the environment read keeps the
/// off-path free of repeated allocations.
fn trace_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("BOLT_TRACE_DIR").ok())
        .as_deref()
}

/// The environment variable that selects the split decode attention: one
/// part+combine launch pair per attention instead of the single-pass kernel.
/// The split changes the summation order, matches the f64 reference within
/// its tolerance, and is bit-identical only between its eager and graphed
/// runs (the KV8 combine lift is the bit-identical piece). Armed by default
/// since it qualified (11.7x on the full-attention call at 30K depth);
/// `0` or `off` is the rollback to the single-pass summation order.
pub(crate) const ATTN_SPLIT_VAR: &str = "FIFI_BOLT_ATTN_SPLIT";

/// Split policy: at most `splits` over the visible range, at least 128 keys per split, so a short
/// context keeps one split and a long one fills the device. The count is 32 on the 82-SM card it
/// was tuned on (8 q heads x 32 splits = 256 blocks) and scales with the device's multiprocessor
/// count ([`device_attn_splits`]).
pub(crate) const ATTN_SPLIT_MAX: usize = 32;
pub(crate) const ATTN_SPLIT_MIN_KEYS: usize = 128;

/// The split partial buffer of the resident's attention: stats (max, sum) plus
/// the accumulator vector, per (token, q head, split). Sized on the 512-dim
/// global head and the full prefill batch, because the eager step runs the
/// same kernels for m = 1..=rows; the 256-dim local layers need less.
pub(crate) const fn attn_part_bytes(splits: usize, rows: usize) -> usize {
    rows * HEADS * splits * (2 + GLOBAL_HEAD) * 4
}

/// The split count on `ctx`'s device: the tuned count scaled by the device query's multiprocessor
/// count. A driver that refuses the query keeps the tuned count.
///
/// [`ATTN_SPLITS_VAR`] overrides the count (1 to 128), for a comparison on one binary; a bad value
/// fails the load.
fn device_attn_splits(ctx: &CudaContext) -> Result<usize, EngError> {
    crate::capability::tuning::attention_splits_or_env(
        ATTN_SPLITS_VAR,
        ATTN_SPLIT_MAX,
        ctx.sm_count(),
    )
}

/// The environment variable that sets the split count of the decode attention instead of the
/// device-derived one.
pub(crate) const ATTN_SPLITS_VAR: &str = "FIFI_BOLT_ATTN_SPLITS";

/// The armed state the environment text selects: unset arms the qualified
/// split (the default); `0`/`off` disarms it, `1`/`on` arms it, and anything
/// else is refused. Split from the env read so the default rule has a
/// portable test: edition 2024 makes `std::env::set_var` unsafe and the crate
/// denies `unsafe_code`, so a test cannot carry the variable into the read.
pub(crate) fn split_armed_from(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        None => Ok(true),
        Some(v) => parse_switch(Some(v), &format!("bolt attention split: {ATTN_SPLIT_VAR}")),
    }
}

/// Whether the environment arms the attention split; a bad value fails the
/// load instead of arming or disarming by accident.
pub(crate) fn attn_split_wanted() -> Result<bool, EngError> {
    split_armed_from(std::env::var(ATTN_SPLIT_VAR).ok().as_deref())
}

/// What the wide prefill attention sees of the global layers: the prefill batch at most, over
/// the whole cache.
fn wide_geometry(cells: usize) -> WideGeometry {
    WideGeometry::unscaled(HEADS, KV_HEADS, cells)
}

pub(crate) struct BoltLayer {
    attn_norm: Vector,
    q_norm: Vector,
    k_norm: Option<Vector>,
    q: NativeWeight,
    k: Option<NativeWeight>,
    v: Option<NativeWeight>,
    out: NativeWeight,
    post_attn_norm: Vector,
    ffn_norm: Vector,
    ffn_gate: NativeWeight,
    ffn_up: NativeWeight,
    ffn_down: NativeWeight,
    post_ffw_norm: Vector,
    ple_gate: NativeWeight,
    ple_proj: NativeWeight,
    post_ple_norm: Vector,
    table: NativeWeight,
    scale: f32,
    global: bool,
    head: usize,
    /// For a layer without its own cache planes, the physical layer whose planes it attends into.
    kv_source: Option<usize>,
}

impl BoltLayer {
    /// Whether the layer writes cache planes of its own.
    fn physical(&self) -> bool {
        self.kv_source.is_none()
    }
}

/// The device state of one sequence: its KV8 planes for the physical layers and the two words
/// the position and attention kernels read. A session owns one; the model owns none, so any
/// number of sequences can step through the same weights.
pub(crate) struct BoltSeq {
    id: u64,
    /// The cells of each head's plane: what this session's request was sized to.
    cells: usize,
    /// The pool cells this session holds; they return when the session drops.
    lease: CellLease,
    /// One entry per layer; `Some` for the physical layers.
    kv: Vec<Option<Kv>>,
    /// The position of the first row of the step (`StepParams.pos`).
    pos: DeviceBuffer,
    /// The key count the first row attends over (`pos + 1`).
    kv_len: DeviceBuffer,
    /// The 32-byte step block a window's rows read: the position at byte 4 and the key count at
    /// byte 8 (the layout of `StepParams`), so one launch can read the words of every row. A
    /// window writes this block and not the two words above; a chunk or a lone token writes
    /// those and not this block.
    step: DeviceBuffer,
}

/// A copy of the cache of one sequence below a position, in host memory. The planes of the 24
/// physical layers are stored layer by layer, plane by plane (k codes, k scales, v codes, v
/// scales) and kv head by kv head, each head's kept positions in position order: `0..fed` for a
/// full-attention layer, the last window for a sliding layer. An entry holds `fed` times the cost
/// of one full-attention position plus one window of the sliding layers. It restores into any
/// sequence, whatever its ring.
pub(crate) struct BoltSnapshot {
    fed: usize,
    /// Shared so the disk tier can read it in chunks while the store still owns the entry.
    data: std::rc::Rc<Vec<u8>>,
}

impl BoltSnapshot {
    /// A snapshot of `fed` cells whose bytes came from another tier (a disk read), laid out as
    /// `Bolt::snapshot_layer_bytes(fed)` says.
    pub(crate) fn from_parts(fed: usize, data: Vec<u8>) -> Self {
        Self {
            fed,
            data: std::rc::Rc::new(data),
        }
    }

    /// A second owner of the bytes, for the disk tier to copy out of.
    pub(crate) fn hold(&self) -> std::rc::Rc<Vec<u8>> {
        std::rc::Rc::clone(&self.data)
    }

    /// The positions the snapshot covers.
    pub(crate) fn fed(&self) -> usize {
        self.fed
    }

    /// The host bytes the snapshot holds.
    pub(crate) fn bytes(&self) -> usize {
        self.data.len()
    }
}

/// The logits rows of a window: the raw head output and its softcap, one vocabulary row per
/// sequence. Allocated only for an engine that serves more than one sequence.
pub(crate) struct WindowBuffers {
    rows: usize,
    logits: Vector,
    capped: Vector,
    /// The row records of every physical layer of one step, rewritten each step.
    tables: DeviceBuffer,
}

/// What the rows of one step are.
enum Rows<'a> {
    /// `m` consecutive positions of one sequence from `pos`: a prompt chunk or one decode token.
    Chunk { seq: &'a BoltSeq, pos: usize },
    /// One token of each of several sequences, in window order.
    Window {
        seqs: &'a [&'a BoltSeq],
        /// The row records of each layer's own planes, by layer; `None` for a layer that reads
        /// another layer's cache.
        tables: &'a [Option<KvRows<'a>>],
    },
}

impl Rows<'_> {
    /// The position label of the trace files.
    fn trace_pos(&self) -> usize {
        match self {
            Self::Chunk { pos, .. } => *pos,
            Self::Window { .. } => 0,
        }
    }
}

/// The columns of one head launch in a window. The vendored matvec keeps one summation order
/// for up to four columns and switches to another above that, so a window of more rows runs the
/// head in launches of at most this many columns and every row sees the order a lone row sees.
pub(crate) const HEAD_COLUMNS: usize = 4;

/// The bytes of one zero write when a sequence clears its planes.
const ZERO_CHUNK: usize = 8 << 20;

static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

/// The per-step buffers, sized once at load.
pub(crate) struct Scratch {
    pub(crate) h: [Vector; 4],
    pub(crate) nrm: Vector,
    pub(crate) nrm2: Vector,
    pub(crate) q_normed: Vector,
    pub(crate) k_normed: Vector,
    pub(crate) v_normed: Vector,
    pub(crate) ones: Vector,
    pub(crate) q: Vector,
    pub(crate) k: Vector,
    pub(crate) v: Vector,
    pub(crate) ctx_out: Vector,
    pub(crate) proj: Vector,
    pub(crate) gate: Vector,
    pub(crate) up: Vector,
    pub(crate) mixed: Vector,
    pub(crate) down: Vector,
    pub(crate) ple_all: Vector,
    pub(crate) ple_scaled: Vector,
    pub(crate) ple_nrm: Vector,
    pub(crate) ple_tbl: Vector,
    pub(crate) ple_mixed: Vector,
    /// Present only when the attention split is armed: the split partial
    /// buffer, sized by [attn_part_bytes].
    pub(crate) attn_part: Option<DeviceBuffer>,
    pub(crate) q8_hidden: DeviceBuffer,
    pub(crate) q8_ctx: DeviceBuffer,
    pub(crate) q8_mixed: DeviceBuffer,
    pub(crate) q8_ple_mixed: DeviceBuffer,
    pub(crate) q8_head: DeviceBuffer,
    /// The QK-norm dump: rows of quantized heads nobody reads back, sized for a
    /// full prefill batch of every q head at the wider head width.
    pub(crate) q8_qk: DeviceBuffer,
    pub(crate) ids: DeviceBuffer,
    /// The KV8 decode-append status word. The launch contract requires it;
    /// the caller's host range checks make the kernel's refuse path
    /// unreachable, so nothing reads it back.
    pub(crate) status: DeviceBuffer,
}

pub(crate) struct Bolt {
    /// The most KV cells one session holds per physical layer: the model's context, or less when
    /// the card carries less.
    cells: usize,
    /// The splits of one decode attention launch on this device.
    attn_splits: usize,
    /// The rows of one prefill chunk: the planned rows of this load. It is also the widest append
    /// any caller makes (`prefill_batch` refuses more), so it sizes the sliding layers' rings.
    pub(crate) rows: usize,
    /// The positions a sliding layer reaches, from the spec the file's header was checked against.
    local_window: usize,
    /// The geometry of a sliding layer's cache: a ring of `local_window + rows - 1` cells. A test
    /// can set it linear, which keeps every position in the sliding layers' planes as the
    /// control the ring must equal bit for bit.
    local_ring: KvRing,
    /// The cells every session of this load draws from: what the card carries beside the weights.
    pool: Arc<CellPool>,
    /// The captured one-token decode body, armed by [`bolt_graph`]. Declared
    /// FIRST so it drops first: fields drop in declaration order, and the
    /// graph must be destroyed before any buffer whose address it baked (a
    /// pin test holds the order). The graph pins nothing (`CapturePlan::pins`
    /// is empty, the flash_next bench pattern): safety is by ownership
    /// instead - no buffer the body touches is reallocated (the step words
    /// are written in place, the scratch sits at its load-time capacity) and a
    /// scratch grow drops the graph before any replacement.
    graph: Option<GraphExec<'static>>,
    /// The sequence whose planes and words the captured body baked.
    graph_seq: u64,
    pub(crate) ctx: CudaContext,
    pub(crate) stream: DeviceStream,
    pub(crate) layers: Vec<BoltLayer>,
    pub(crate) embed: NativeWeight,
    pub(crate) ple_model: NativeWeight,
    pub(crate) ple_model_norm: Vector,
    pub(crate) out_norm: Vector,
    pub(crate) rope_freqs: Option<Vector>,
    pub(crate) scratch: Scratch,
    /// The tensor-core GEMM of a prompt chunk's projections.
    mmq: Option<DenseChunkGemm>,
    pub(crate) logits: Vector,
    pub(crate) logits_capped: Vector,
    pub(crate) eps: f32,
    /// The sticky decode-graph mode: wanted from the environment, refused on
    /// the first reason, always counted. Every capture refusal pins the mode
    /// to eager; the eager body stays the reference.
    pub(crate) graph_mode: DecodeGraphMode,
    /// The logits rows of a window, present once `reserve_window` ran.
    window: Option<WindowBuffers>,
    draw: crate::family::decode_step::draw::DrawBuffers,
    /// What names the checkpoint to an entry kept outside this process: its header digest, a
    /// sample of every tensor and its length and modification time.
    pub(crate) weights_identity: String,
    /// The wide prefill attention of the global layers, when the device runs it.
    wide_prefill: Option<WideAttention>,
}

fn embed_scale() -> f32 {
    (HIDDEN as f32).sqrt()
}

/// The device demand of a Bolt load: the weights file bytes, the KV8 cache of
/// the physical layers at `cells` positions, the prefill scratch, and the
/// shared headroom reserve. Pure arithmetic, so the admission refusal is
/// testable on the host without a device.
pub(crate) fn bolt_demand(
    weights_bytes: u64,
    attn_splits: Option<usize>,
    cells: usize,
    wide_prefill: bool,
    prefill_rows: usize,
) -> Result<crate::kernels::cuda::plan::Demand, EngError> {
    let kv = super::super::budget::bolt_kv_plan(cells as u64)
        .ok_or_else(|| EngError::MemoryPlan("the Bolt KV budget overflows".into()))?;
    // The first session: the full-attention layers' cells and every sliding layer's ring.
    let kv_bytes = kv
        .positions
        .checked_mul(kv.bytes_per_position)
        .and_then(|bytes| bytes.checked_add(super::super::budget::bolt_ring_bytes(prefill_rows)?))
        .ok_or_else(|| EngError::MemoryPlan("the Bolt KV cache charge overflows".into()))?;
    let scratch = scratch_bytes(attn_splits, wide_prefill, prefill_rows)
        .ok_or_else(|| EngError::MemoryPlan("the Bolt scratch charge overflows".into()))?;
    Ok(crate::kernels::cuda::plan::Demand {
        tensors: weights_bytes,
        kv_cache: kv_bytes,
        work: scratch,
        headroom: super::super::budget::BOLT_HEADROOM_BYTES,
    })
}

/// The cells, prefill rows and demand of one load.
type Planned = (usize, usize, usize, crate::kernels::cuda::plan::Demand);

/// The working set shared by every session and the first session's sliding rings, which are
/// charged with it: the rings do not grow with the context, so they are not part of the pool.
fn shared_bytes(splits: Option<usize>, wide: bool, rows: usize) -> Option<u64> {
    scratch_bytes(splits, wide, rows)?.checked_add(super::super::budget::bolt_ring_bytes(rows)?)
}

/// The byte figures of Bolt's per-session cell sizing, planned by the shared card-fit rule.
fn cell_figures(splits: Option<usize>, wide: bool, rows: usize) -> Result<CellFigures, EngError> {
    CellFigures::checked(
        super::super::budget::bolt_kv_bytes_per_position(),
        shared_bytes(splits, wide, rows),
        (
            super::super::budget::BOLT_HEADROOM_BYTES,
            super::super::budget::BOLT_SEQ_FIXED_BYTES
                .saturating_add(super::super::budget::bolt_ring_bytes(rows).unwrap_or(u64::MAX)),
        ),
    )
    .ok_or_else(|| EngError::MemoryPlan("the Bolt cell figures overflow".into()))
}

/// The positions a weights file's header gives as its context, at most the model maximum. `None`
/// when the header cannot be read or gives no positive count.
fn header_context(path: &std::path::Path) -> Option<usize> {
    use crate::weights::MetaValue;
    let header = crate::weights::shared_header(path, 256 << 20).ok()?;
    let value = match header.meta("gemma4.context_length")? {
        MetaValue::U32(value) => u64::from(*value),
        MetaValue::U64(value) => *value,
        MetaValue::I32(value) => u64::try_from(*value).ok()?,
        MetaValue::I64(value) => u64::try_from(*value).ok()?,
        _ => return None,
    };
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .map(|value| value.min(MAX_CELLS))
}

/// The sessions a start plans its cells for: the count the config names, else the most the
/// planner would ever fund (one when the decode graph is asked for, which serves one session).
fn planned_sessions(seqs: super::super::config::MaxSeqs) -> Result<usize, EngError> {
    use super::super::config::MaxSeqs;
    Ok(match seqs {
        MaxSeqs::Count(count) => count,
        MaxSeqs::Auto if decode_graph_wanted()? => 1,
        MaxSeqs::Auto => Bolt::WINDOW_ROWS,
    })
}

pub(crate) fn load(path: &std::path::Path) -> Result<Bolt, EngError> {
    let bytes = std::fs::metadata(path)
        .map_err(|error| EngError::MemoryPlan(format!("the Bolt weights file: {error}")))?
        .len();
    load_choosing(
        &[crate::family::gguf_select::GgufFile {
            path: path.to_path_buf(),
            bytes,
        }],
        super::super::config::MaxSeqs::Count(1),
        &|_| {},
    )
}

/// Loads the first of `files` (highest precision first) whose device demand the free memory
/// admits, and prints which file it chose and why when there was a choice. `planned` is told the
/// KV cells the load serves as soon as they are known, before any weight is uploaded.
pub(crate) fn load_choosing(
    files: &[crate::family::gguf_select::GgufFile],
    seqs: super::super::config::MaxSeqs,
    planned: &dyn Fn(usize),
) -> Result<Bolt, EngError> {
    let sessions = planned_sessions(seqs)?;
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())?;
    let stream = ctx.stream()?;
    // Admit the device budget before allocating: weights + KV cache + scratch
    // plus headroom against free device memory, through the shared plan::admit
    // the other resident executors use. An exact fit of the allocations alone
    // is refused; the shared error names the needed and free bytes.
    let free = ctx.memory_info()?.free as u64;
    let split = attn_split_wanted()?
        .then(|| device_attn_splits(&ctx))
        .transpose()?;
    // The wide prefill attention is built wherever the device gate passes and starts as the
    // override or the default says; the admission charges its buffer when it is built.
    let wide_load = WideLoad::read(&ctx, "gemma4 bolt", wide_geometry(MAX_CELLS))?;
    let wide = wide_load.passed();
    // The cells a card carries for a file, the rows of one prefill chunk it takes, and the demand
    // of that load: one session holds at most the cells the file's header gives as its context,
    // and every session leases its cells from the pool of what the card carries.
    let plan = |file: &crate::family::gguf_select::GgufFile| -> Result<Planned, EngError> {
        let rows = cell_figures(split, wide, Bolt::WINDOW_ROWS)?.prefill_rows(
            free,
            file.bytes,
            (super::super::budget::BOLT_PREFILL_LADDER, FALLBACK_CELLS),
            &|rows| shared_bytes(split, wide, rows),
            "FIFI_BOLT_PREFILL_ROWS",
        );
        let figures = cell_figures(split, wide, rows)?;
        let (cells, pool) =
            figures.planned(free, file.bytes, sessions, (MAX_CELLS, FALLBACK_CELLS));
        let cells = cells.min(header_context(&file.path).unwrap_or(MAX_CELLS));
        Ok((
            cells,
            pool,
            rows,
            bolt_demand(file.bytes, split, cells, wide, rows)?,
        ))
    };
    let choice = crate::family::gguf_select::choose(files, &|file| {
        plan(file)
            .and_then(|(_, _, _, demand)| crate::kernels::cuda::plan::admit(&demand, free))
            .map(|_| ())
            .map_err(|error| error.describe())
    })?;
    if let Some(line) = &choice.line {
        eprintln!("{line}");
    }
    let path = choice.chosen.path.as_path();
    let (weights, spelling) = super::super::open_bolt_weights(path)?;
    let weights_identity = crate::weights::sha256_hex(
        &crate::weights::gguf_identity_digest(weights.header(), path, None)
            .map_err(|error| EngError::Checkpoint(format!("prefix identity: {error}")))?,
    )
    .to_string();
    if header_context(&choice.chosen.path).is_none() {
        eprintln!(
            "bolt cuda: {} states no context length; serving up to the model maximum of {MAX_CELLS} cells",
            choice.chosen.path.display()
        );
    }
    let (cells, pool, rows, demand) = plan(&choice.chosen)?;
    crate::kernels::cuda::plan::admit(&demand, free)?;
    planned(cells);
    eprintln!(
        "bolt cuda: {pool} KV cells per layer and {rows} prefill rows per chunk chosen from the \
         card, shared by every session up to {cells} each ({free} bytes free, {} bytes of \
         weights; the model maximum is {MAX_CELLS})",
        choice.chosen.bytes
    );
    let mut layers = Vec::with_capacity(LAYERS);
    // The canonical spelling's merged per_layer_token_embd is read once and
    // sliced per layer below; the legacy spelling reads each layer's own
    // table instead.
    let merged = match spelling.ple_table(0) {
        Some(_) => None,
        None => {
            let name = spelling.per_layer_token_embd().ok_or_else(|| {
                EngError::Checkpoint(
                    "gguf model: the canonical spelling carries the merged table".to_string(),
                )
            })?;
            let (ty, bytes) = weights.read_tensor(name)?;
            Some((bytes, ty))
        }
    };
    for layer in 0..LAYERS {
        let global = NEMOTRON_BOLT_E4B.layers_attention[layer] == AttentionKind::Full;
        let head = if global { GLOBAL_HEAD } else { LOCAL_HEAD };
        let physical = layer < PHYSICAL_LAYERS;
        // A shared layer attends into the cache of the last physical layer of its own kind.
        let kv_source = if physical {
            None
        } else {
            Some(NEMOTRON_BOLT_E4B.shared_kv_source(layer as u32)? as usize)
        };
        let base = format!("blk.{layer}.");
        layers.push(BoltLayer {
            attn_norm: vector(&weights, &ctx, &format!("{base}attn_norm.weight"), HIDDEN)?,
            q_norm: vector(&weights, &ctx, &format!("{base}attn_q_norm.weight"), head)?,
            k_norm: if physical {
                Some(vector(
                    &weights,
                    &ctx,
                    &format!("{base}attn_k_norm.weight"),
                    head,
                )?)
            } else {
                None
            },
            q: matrix(
                &weights,
                &ctx,
                &format!("{base}attn_q.weight"),
                HIDDEN,
                HEADS * head,
            )?,
            k: if physical {
                Some(matrix(
                    &weights,
                    &ctx,
                    &format!("{base}attn_k.weight"),
                    HIDDEN,
                    KV_HEADS * head,
                )?)
            } else {
                None
            },
            v: if physical {
                Some(matrix(
                    &weights,
                    &ctx,
                    &format!("{base}attn_v.weight"),
                    HIDDEN,
                    KV_HEADS * head,
                )?)
            } else {
                None
            },
            out: matrix(
                &weights,
                &ctx,
                &format!("{base}attn_output.weight"),
                HEADS * head,
                HIDDEN,
            )?,
            post_attn_norm: vector(
                &weights,
                &ctx,
                &format!("{base}post_attention_norm.weight"),
                HIDDEN,
            )?,
            ffn_norm: vector(&weights, &ctx, &format!("{base}ffn_norm.weight"), HIDDEN)?,
            ffn_gate: matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_gate.weight"),
                HIDDEN,
                FFN,
            )?,
            ffn_up: matrix(&weights, &ctx, &format!("{base}ffn_up.weight"), HIDDEN, FFN)?,
            ffn_down: matrix(
                &weights,
                &ctx,
                &format!("{base}ffn_down.weight"),
                FFN,
                HIDDEN,
            )?,
            post_ffw_norm: vector(
                &weights,
                &ctx,
                &format!("{base}post_ffw_norm.weight"),
                HIDDEN,
            )?,
            ple_gate: matrix(
                &weights,
                &ctx,
                &format!("{base}{}", spelling.ple_gate()),
                HIDDEN,
                PLE,
            )?,
            ple_proj: matrix(
                &weights,
                &ctx,
                &format!("{base}{}", spelling.ple_proj()),
                PLE,
                HIDDEN,
            )?,
            post_ple_norm: vector(
                &weights,
                &ctx,
                &format!("{base}{}", spelling.ple_post_norm()),
                HIDDEN,
            )?,
            table: match spelling.ple_table(layer) {
                Some(name) => matrix(&weights, &ctx, &name, PLE, VOCAB)?,
                None => {
                    let (merged, merged_ty) = merged.as_ref().ok_or_else(|| {
                        EngError::Checkpoint(
                            "gguf model: the merged table was hoisted for the canonical spelling"
                                .to_string(),
                        )
                    })?;
                    let bytes =
                        ple_rows_from_merged(merged, *merged_ty, layer, PLE, LAYERS, VOCAB)?;
                    matrix_from_bytes(
                        &ctx,
                        "per_layer_token_embd.weight",
                        *merged_ty,
                        PLE,
                        VOCAB,
                        &bytes,
                    )?
                }
            },
            scale: scalar(&weights, &format!("{base}layer_output_scale.weight"))?,
            global,
            head,
            kv_source,
        });
    }
    let embed = matrix(&weights, &ctx, "token_embd.weight", HIDDEN, VOCAB)?;
    let ple_model = matrix(
        &weights,
        &ctx,
        spelling.ple_model_proj(),
        HIDDEN,
        LAYERS * PLE,
    )?;
    let ple_model_norm = vector(&weights, &ctx, spelling.ple_proj_norm(), PLE)?;
    let out_norm = vector(&weights, &ctx, "output_norm.weight", HIDDEN)?;
    let rope_freqs = Some(vector(&weights, &ctx, "rope_freqs.weight", 256)?);
    let eps = epsilon(&weights)?;
    let h = [
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
        vector_filled(&ctx, HIDDEN, 0.0)?,
    ];
    // The attention split refuses a bad switch value before any allocation.
    let attn_split = attn_split_wanted()?;
    let attn_splits = device_attn_splits(&ctx)?;
    let scratch = Scratch {
        h,
        nrm: vector_filled(&ctx, HIDDEN, 0.0)?,
        nrm2: vector_filled(&ctx, HIDDEN, 0.0)?,
        q_normed: vector_filled(&ctx, HEADS * GLOBAL_HEAD, 0.0)?,
        k_normed: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        v_normed: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        ones: vector_filled(&ctx, GLOBAL_HEAD, 1.0)?,
        q: vector_filled(&ctx, HEADS * GLOBAL_HEAD, 0.0)?,
        k: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        v: vector_filled(&ctx, KV_HEADS * GLOBAL_HEAD, 0.0)?,
        ctx_out: vector_filled(&ctx, HEADS * GLOBAL_HEAD, 0.0)?,
        proj: vector_filled(&ctx, HIDDEN, 0.0)?,
        gate: vector_filled(&ctx, FFN, 0.0)?,
        up: vector_filled(&ctx, FFN, 0.0)?,
        mixed: vector_filled(&ctx, FFN, 0.0)?,
        down: vector_filled(&ctx, HIDDEN, 0.0)?,
        ple_all: vector_filled(&ctx, LAYERS * PLE, 0.0)?,
        ple_scaled: vector_filled(&ctx, LAYERS * PLE, 0.0)?,
        ple_nrm: vector_filled(&ctx, PLE, 0.0)?,
        ple_tbl: vector_filled(&ctx, PLE, 0.0)?,
        ple_mixed: vector_filled(&ctx, PLE, 0.0)?,
        q8_hidden: zero(&ctx, q8_1_bytes(HIDDEN, 1).expect("hidden q8 size"))?,
        q8_ctx: zero(
            &ctx,
            q8_1_bytes(HEADS * GLOBAL_HEAD, 1).expect("context q8 size"),
        )?,
        q8_mixed: zero(&ctx, q8_1_bytes(FFN, 1).expect("mixed q8 size"))?,
        q8_ple_mixed: zero(&ctx, q8_1_bytes(PLE, 1).expect("ple mixed q8 size"))?,
        q8_head: zero(&ctx, q8_1_bytes(GLOBAL_HEAD, rows).expect("head q8 size"))?,
        q8_qk: zero(
            &ctx,
            q8_1_bytes(GLOBAL_HEAD, rows * HEADS).expect("qk dump size"),
        )?,
        ids: zero(&ctx, rows * 4)?,
        status: zero(&ctx, 4)?,
        attn_part: if attn_split {
            Some(zero(&ctx, attn_part_bytes(attn_splits, rows))?)
        } else {
            None
        },
    };
    let logits = vector_filled(&ctx, VOCAB, 0.0)?;
    let logits_capped = vector_filled(&ctx, VOCAB, 0.0)?;
    let wide_prefill = wide_load.build(&ctx, wide_geometry(cells))?;
    let mmq = DenseChunkGemm::new(&ctx, MMQ_K_MAX, rows)
        .inspect_err(|error| eprintln!("bolt cuda: no tensor-core prefill: {}", error.describe()))
        .ok();
    // The pool was planned from the free memory before the weights went up; what the load
    // allocated beyond its demand is already gone, so the pool is cut to what the card holds now.
    let held = u64::try_from(ctx.memory_info()?.free).unwrap_or(0);
    let pool = cell_figures(split, wide, rows)?.held_pool(pool, held, sessions);
    // The memory receipt: a report, never a reason to fail a good start.
    let receipt = ctx.memory_info().map(|info| {
        let per_position = super::super::budget::bolt_kv_bytes_per_position().unwrap_or(0);
        let cells = u64::try_from(pool).unwrap_or(0);
        crate::family::mem::fill::TierReceipt::dense_only(
            (
                u64::try_from(info.total).unwrap_or(0),
                u64::try_from(info.free).unwrap_or(0),
            ),
            choice.chosen.bytes,
            Some(crate::family::mem::fill::CacheFigure {
                cells,
                bytes: per_position.saturating_mul(cells),
            }),
        )
    });
    eprintln!(
        "{}",
        crate::family::mem::fill::report_line("bolt cuda", receipt)
    );
    let draw = crate::family::decode_step::draw::DrawBuffers::new(
        &ctx,
        Bolt::WINDOW_ROWS,
        VOCAB,
        Bolt::DRAW_WIDTH,
    )?;
    Ok(Bolt {
        cells: cells.min(pool),
        attn_splits,
        rows,
        local_window: NEMOTRON_BOLT_E4B.sliding_window as usize,
        local_ring: NEMOTRON_BOLT_E4B.kv_ring(AttentionKind::Sliding, rows),
        pool: CellPool::new(pool),
        ctx,
        stream,
        layers,
        embed,
        ple_model,
        ple_model_norm,
        out_norm,
        rope_freqs,
        scratch,
        mmq,
        logits,
        logits_capped,
        eps,
        graph_mode: DecodeGraphMode::new(decode_graph_wanted()?),
        graph: None,
        graph_seq: 0,
        window: None,
        draw,
        weights_identity,
        wide_prefill,
    })
}

impl BoltSeq {
    /// The cells of each head's plane this sequence holds.
    pub(crate) fn cells(&self) -> usize {
        self.cells
    }

    /// The most cells this sequence can be resized to: its own and every free cell of the pool.
    pub(crate) fn reachable(&self) -> usize {
        self.lease.reachable()
    }
}

impl Bolt {
    /// The most sessions one decode window steps together: the decode attention and the head
    /// launch are planned for this many rows.
    pub(crate) const WINDOW_ROWS: usize = 8;

    /// The KV cells per physical layer this load serves.
    pub(crate) fn cells(&self) -> usize {
        self.cells
    }

    /// The splits of one decode attention launch on this device.
    pub(crate) fn attn_splits(&self) -> usize {
        self.attn_splits
    }

    /// Makes the sliding layers of sequences created from now on keep every position in a linear
    /// plane, or a ring again: the control a ring must equal bit for bit. A test hook; the ring is
    /// what every start uses.
    #[cfg(test)]
    pub(crate) fn set_linear_local_layers(&mut self, linear: bool) {
        self.local_ring = if linear {
            KvRing::LINEAR
        } else {
            NEMOTRON_BOLT_E4B.kv_ring(AttentionKind::Sliding, self.rows)
        };
    }

    /// The cells the card carries for every session together.
    pub(crate) fn pool(&self) -> &Arc<CellPool> {
        &self.pool
    }

    /// The sessions that hold cells of the pool now.
    pub(crate) fn live_sessions(&self) -> usize {
        Arc::strong_count(&self.pool).saturating_sub(1)
    }

    /// The free VRAM now, `None` when the card cannot be read.
    pub(crate) fn free_vram(&self) -> Option<u64> {
        self.ctx
            .memory_info()
            .ok()
            .map(|info| u64::try_from(info.free).unwrap_or(0))
    }

    /// Allocates the device state of one more sequence: zeroed KV8 planes for the physical
    /// layers at `cells` positions and its two position words. The planes are read only below
    /// the position word, so a sequence that is reset needs no clearing.
    pub(crate) fn new_seq(&self) -> Result<BoltSeq, EngError> {
        self.new_seq_cells(self.cells)
    }

    /// A sequence over `cells` cells per head, leased from the pool.
    pub(crate) fn new_seq_cells(&self, cells: usize) -> Result<BoltSeq, EngError> {
        let mut seq = BoltSeq {
            id: 0,
            cells: 0,
            lease: self.pool.lease_for(cells, self.cells)?,
            kv: Vec::new(),
            pos: zero(&self.ctx, 4)?,
            kv_len: zero(&self.ctx, 4)?,
            step: zero(&self.ctx, STEP_BLOCK_BYTES)?,
        };
        self.fill_seq(&mut seq)?;
        Ok(seq)
    }

    /// Resizes `seq` to `cells` per head: its planes are released first and the new ones
    /// allocated and zeroed, so the peak is one session's planes, not two. The contents are not
    /// kept: a session is resized only at the start of a request, before any prefix restore. A
    /// pool that cannot fund the growth refuses with a memory-plan error and leaves `seq` as it
    /// was.
    pub(crate) fn resize_seq(&self, seq: &mut BoltSeq, cells: usize) -> Result<(), EngError> {
        if cells == seq.cells {
            return Ok(());
        }
        seq.lease.resize_for(cells, self.cells)?;
        self.fill_seq(seq)
    }

    /// Gives `seq` room for `positions` positions with the `fed` cells it has kept: leases the
    /// cells from the pool (the one growth rule of `family::cell_pool`) and re-lays the planes in
    /// larger ones. A pool or a card that cannot fund it refuses with a memory-plan error and
    /// leaves `seq` as it was: its planes, its lease and its id. No other sequence is touched.
    pub(crate) fn make_room(
        &self,
        seq: &mut BoltSeq,
        fed: usize,
        positions: usize,
    ) -> Result<(), EngError> {
        let held = seq.cells;
        if positions <= held {
            return Ok(());
        }
        let who = seq.id;
        let cells = fund_growth(&mut seq.lease, who, held, positions, self.cells)?;
        // The old planes stay until the copy ends: the card must hold both.
        let per_position = crate::family::gemma::gemma4::budget::bolt_seq_cost(self.rows)
            .map_or(0, |cost| cost.per_position);
        let cells = fit_planes(
            &mut seq.lease,
            who,
            held,
            cells,
            positions,
            self.ctx.memory_info().ok().map(|info| info.free as u64),
            |cells| per_position.saturating_mul(cells as u64),
        )?;
        let began = std::time::Instant::now();
        match self.grow_planes(seq, cells, fed) {
            Ok(()) => {
                eprintln!(
                    "gemma4 bolt: session {who} grew from {held} to {cells} cells in {:.1} ms ({} of {} cells free)",
                    began.elapsed().as_secs_f64() * 1000.0,
                    self.pool.free(),
                    self.pool.total()
                );
                Ok(())
            }
            Err(error) => {
                seq.lease.resize(held);
                Err(match error {
                    EngError::MemoryPlan(detail) => EngError::MemoryPlan(format!(
                        "session {who} cannot grow from {held} to {cells} cells on the card: \
                         {detail}; the request ends here and the others go on"
                    )),
                    other => other,
                })
            }
        }
    }

    /// Re-lays the planes of `seq` in `cells` cells per head, keeping the first `fed` cells of
    /// every head of every plane of a full-attention layer. A sliding layer's ring has the same
    /// cells whatever the context, so its planes are moved, not copied. The old planes are
    /// released after the copies on the stream, so the peak is both sets of the full-attention
    /// planes together. The sequence takes a new id, so a captured decode graph never replays the
    /// old planes.
    fn grow_planes(&self, seq: &mut BoltSeq, cells: usize, fed: usize) -> Result<(), EngError> {
        let zeros = zero(&self.ctx, ZERO_CHUNK)?;
        let built = self.build_kv(&zeros, cells, Some(&seq.kv));
        let drained = self.stream.synchronize();
        let mut grown = built?;
        drained?;
        let keep = fed.min(seq.cells);
        for (old, new) in seq.kv.iter().zip(grown.iter_mut()) {
            let (Some(old), Some(new)) = (old, new) else {
                continue;
            };
            let layer_head = old.shape.head_dim;
            let (from_cells, to_cells) = (old.shape.cells, new.shape.cells);
            for (from, to, unit) in [
                (&old.k_codes, &new.k_codes, layer_head),
                (&old.k_scales, &new.k_scales, (layer_head / 32) * 2),
                (&old.v_codes, &new.v_codes, layer_head),
                (&old.v_scales, &new.v_scales, (layer_head / 32) * 2),
            ] {
                for kv_head in 0..KV_HEADS {
                    let bytes = keep.min(from_cells).min(to_cells) * unit;
                    if bytes == 0 {
                        continue;
                    }
                    self.ctx.copy_device(
                        &self.stream,
                        from.range(kv_head * from_cells * unit, bytes)?,
                        to.range(kv_head * to_cells * unit, bytes)?,
                    )?;
                }
            }
        }
        // The copies read the old planes: the stream drains before they drop.
        self.stream.synchronize()?;
        // The rings keep their planes: only the layers `build_kv` left empty move across.
        for (old, new) in seq.kv.iter_mut().zip(grown.iter_mut()) {
            if new.is_none() {
                *new = old.take();
            }
        }
        seq.kv = grown;
        seq.cells = cells;
        seq.id = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Builds the planes of `seq` at the cells its lease holds. The sequence takes a new id, so a
    /// captured decode graph never replays old planes.
    fn fill_seq(&self, seq: &mut BoltSeq) -> Result<(), EngError> {
        seq.kv = Vec::new();
        seq.cells = 0;
        let cells = seq.lease.cells();
        // One zero chunk on the device, copied down every plane: the planes are cleared at
        // device speed and no host memory the size of a plane is touched.
        let zeros = zero(&self.ctx, ZERO_CHUNK)?;
        let built = self.build_kv(&zeros, cells, None);
        // The queued copies read `zeros`: the stream drains before it drops, whether or not the
        // build succeeded.
        let drained = self.stream.synchronize();
        seq.kv = built?;
        drained?;
        seq.cells = cells;
        seq.id = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The cache geometry of one physical layer and the context its cells are counted for: a
    /// full-attention layer keeps a plane of the session's cells, a sliding layer a ring whose
    /// cells do not depend on the session (the linear control keeps the session's cells too).
    fn layer_geometry(&self, layer: &BoltLayer, seq_cells: usize) -> (KvRing, usize) {
        if layer.global || self.local_ring == KvRing::LINEAR {
            (KvRing::LINEAR, seq_cells)
        } else {
            (self.local_ring, MAX_CELLS)
        }
    }

    /// The planes of every physical layer at `cells` cells for the full-attention layers. With the
    /// planes a sequence already `holds`, the layers whose planes are rings are left out (`None`):
    /// a ring has the same cells at any context, so the caller moves those across.
    fn build_kv(
        &self,
        zeros: &DeviceBuffer,
        cells: usize,
        holds: Option<&[Option<Kv>]>,
    ) -> Result<Vec<Option<Kv>>, EngError> {
        let ctx = &self.ctx;
        let plane = |bytes: usize| -> Result<DeviceBuffer, EngError> {
            let buffer = ctx.allocate(bytes)?;
            let mut at = 0;
            while at < bytes {
                let len = ZERO_CHUNK.min(bytes - at);
                ctx.copy_device(&self.stream, zeros.range(0, len)?, buffer.range(at, len)?)?;
                at += len;
            }
            Ok(buffer)
        };
        let mut kv = Vec::with_capacity(LAYERS);
        for (index, layer) in self.layers.iter().enumerate() {
            let head = layer.head;
            let held_ring = holds
                .and_then(|planes| planes.get(index))
                .and_then(Option::as_ref)
                .is_some_and(|plane| plane.geometry != KvRing::LINEAR);
            kv.push(if !layer.physical() || held_ring {
                None
            } else {
                let (geometry, context) = self.layer_geometry(layer, cells);
                let shape = Kv8Shape::of_layer(
                    geometry,
                    context,
                    1,
                    KV_HEADS,
                    head,
                    Kv8CodeScale::UnroundedF32,
                );
                let plane_cells = shape.cells;
                Some(Kv {
                    k_codes: plane(KV_HEADS * plane_cells * head)?,
                    k_scales: plane(KV_HEADS * plane_cells * (head / 32) * 2)?,
                    v_codes: plane(KV_HEADS * plane_cells * head)?,
                    v_scales: plane(KV_HEADS * plane_cells * (head / 32) * 2)?,
                    shape,
                    geometry,
                    context,
                })
            });
        }
        Ok(kv)
    }

    /// The `(offset, bytes)` ranges of the positions a snapshot of `fed` tokens keeps, of every
    /// plane of one layer's cache, in the order a snapshot stores them: plane by plane (k codes,
    /// k scales, v codes, v scales), kv head by kv head, a head's cells in position order. A
    /// full-attention layer keeps cells `0..fed`; a sliding layer keeps its last window, which
    /// sits in ring cells (two runs when it wraps).
    fn snapshot_ranges(
        geometry: KvRing,
        context: usize,
        head: usize,
        fed: usize,
    ) -> [Vec<(usize, usize)>; 4] {
        let (codes, scales) = (head, (head / 32) * 2);
        let plane = |unit: usize| geometry.plane_ranges(context, fed, KV_HEADS, unit);
        [plane(codes), plane(scales), plane(codes), plane(scales)]
    }

    /// The bytes of each physical layer's part of a snapshot of `fed` cells, in the order the
    /// snapshot stores them. Their sum is the snapshot's size; each is one object of a disk entry.
    pub(crate) fn snapshot_layer_bytes(&self, fed: usize) -> Vec<usize> {
        self.layers
            .iter()
            .filter(|layer| layer.physical())
            .map(|layer| {
                let (geometry, context) = self.layer_geometry(layer, self.cells);
                Self::snapshot_ranges(geometry, context, layer.head, fed)
                    .iter()
                    .map(|plane| plane.iter().map(|range| range.1).sum::<usize>())
                    .sum()
            })
            .collect()
    }

    /// Copies the kept positions of every plane of `seq` into host memory: `0..fed` of a
    /// full-attention layer, the last window of a sliding layer (read out of its ring cells, in
    /// position order). The copy is what a later request restores into any sequence, so it is
    /// allocated only for the positions it covers; a host whose available memory (less the OS
    /// floor) cannot hold it refuses with a memory-plan error and nothing is stored.
    pub(crate) fn snapshot_seq(&self, seq: &BoltSeq, fed: usize) -> Result<BoltSnapshot, EngError> {
        if fed > seq.cells {
            return Err(EngError::ContextLimit(
                "the Bolt snapshot exceeds the context".to_owned(),
            ));
        }
        let mut total = 0usize;
        for planes in seq.kv.iter().flatten() {
            for plane in
                Self::snapshot_ranges(planes.geometry, planes.context, planes.shape.head_dim, fed)
            {
                total += plane.iter().map(|range| range.1).sum::<usize>();
            }
        }
        // Linux grants an allocation it cannot back and kills the process later, so the
        // allocation cannot be what refuses: the host's available memory (with the OS floor and
        // the enclosing cgroup's limit) is checked first.
        crate::family::host_store::admit_entry(
            "a Bolt prefix snapshot",
            total,
            crate::family::mem::HostFacts::read("a Bolt prefix snapshot", "a Bolt prefix snapshot"),
        )?;
        let mut data = Vec::new();
        data.try_reserve_exact(total).map_err(|_| {
            EngError::MemoryPlan(format!(
                "the host cannot hold a {total} byte prefix snapshot"
            ))
        })?;
        data.resize(total, 0);
        let mut at = 0;
        for kv in seq.kv.iter().flatten() {
            let buffers = [&kv.k_codes, &kv.k_scales, &kv.v_codes, &kv.v_scales];
            for (buffer, ranges) in buffers.into_iter().zip(Self::snapshot_ranges(
                kv.geometry,
                kv.context,
                kv.shape.head_dim,
                fed,
            )) {
                for (offset, bytes) in ranges {
                    if bytes > 0 {
                        buffer.read_at(&self.stream, offset, &mut data[at..at + bytes])?;
                        at += bytes;
                    }
                }
            }
        }
        Ok(BoltSnapshot::from_parts(fed, data))
    }

    /// Writes a snapshot into `seq`: the positions `0..fed` of a full-attention layer and the last
    /// window of a sliding layer, each into the cell its geometry gives (a ring cell is
    /// `position % ring`). The cells above are left as they are, which is safe because the
    /// position word bounds every read.
    pub(crate) fn restore_seq(
        &self,
        seq: &BoltSeq,
        snapshot: &BoltSnapshot,
    ) -> Result<(), EngError> {
        if snapshot.fed > seq.cells {
            return Err(EngError::ContextLimit(
                "the Bolt snapshot exceeds the session's cells".to_owned(),
            ));
        }
        let mut at = 0;
        for kv in seq.kv.iter().flatten() {
            let buffers = [&kv.k_codes, &kv.k_scales, &kv.v_codes, &kv.v_scales];
            for (buffer, ranges) in buffers.into_iter().zip(Self::snapshot_ranges(
                kv.geometry,
                kv.context,
                kv.shape.head_dim,
                snapshot.fed,
            )) {
                for (offset, bytes) in ranges {
                    if bytes > 0 {
                        buffer.write_at(&self.stream, offset, &snapshot.data[at..at + bytes])?;
                        at += bytes;
                    }
                }
            }
        }
        Ok(())
    }

    /// Allocates the logits rows that a window of up to `rows` sequences needs, once. A request
    /// for fewer rows than are held keeps the buffers.
    pub(crate) fn reserve_window(&mut self, rows: usize) -> Result<(), EngError> {
        if !(2..=Self::WINDOW_ROWS).contains(&rows) {
            return Err(EngError::BadRequest(format!(
                "a Bolt window holds 2 to {} rows, got {rows}",
                Self::WINDOW_ROWS
            )));
        }
        if self.window.as_ref().is_some_and(|held| held.rows >= rows) {
            return Ok(());
        }
        self.window = Some(WindowBuffers {
            rows,
            logits: vector_filled(&self.ctx, rows * VOCAB, 0.0)?,
            capped: vector_filled(&self.ctx, rows * VOCAB, 0.0)?,
            tables: self
                .ctx
                .allocate(rows * PHYSICAL_LAYERS * ROW_RECORD_BYTES)?,
        });
        Ok(())
    }

    /// Steps one token of each sequence together and returns the capped logits row of each, in
    /// order. The projections, norms, feed-forward and per-layer-input blocks run once over all
    /// rows; rotation, cache append and attention run once per sequence against its own planes
    /// and words, which is the launch a lone sequence makes. The head runs in launches of at
    /// most [`HEAD_COLUMNS`] columns. `positions[i]` is the cell the token of `seqs[i]` is
    /// written to.
    pub(crate) fn decode_rows(
        &mut self,
        seqs: &[&BoltSeq],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.step_window(seqs, tokens, positions)?;
        let m = seqs.len();
        let window = self.window.as_ref().expect("checked above");
        let mut bytes = vec![0u8; m * VOCAB * 4];
        window.capped.buf.read_at(&self.stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<{ VOCAB * 4 }>()
            .0
            .iter()
            .map(|row| {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            })
            .collect())
    }

    /// The row records of a window: for each physical layer, one record per sequence naming that
    /// sequence's planes of the layer, their cells and its step block, written in one copy into the
    /// window's table buffer. The cache append and the attention of every layer read them, so each
    /// is one launch over all rows.
    fn window_tables<'a>(
        &'a self,
        seqs: &'a [&'a BoltSeq],
    ) -> Result<Vec<Option<KvRows<'a>>>, EngError> {
        let held = self
            .window
            .as_ref()
            .expect("the window buffers are reserved");
        let mut per_layer: Vec<Vec<KvRow<'a>>> = Vec::with_capacity(PHYSICAL_LAYERS);
        let mut index_of = Vec::with_capacity(PHYSICAL_LAYERS);
        for (index, layer) in self.layers.iter().enumerate() {
            if !layer.physical() {
                continue;
            }
            let mut rows = Vec::with_capacity(seqs.len());
            for seq in seqs {
                let own = seq.kv[index].as_ref().expect("a physical layer has planes");
                rows.push(KvRow {
                    k_codes: own.k_codes.range(0, own.k_codes.bytes())?,
                    k_scales: own.k_scales.range(0, own.k_scales.bytes())?,
                    v_codes: own.v_codes.range(0, own.v_codes.bytes())?,
                    v_scales: own.v_scales.range(0, own.v_scales.bytes())?,
                    step: seq.step.range(0, STEP_BLOCK_BYTES)?,
                    cells: own.shape.cells,
                });
            }
            per_layer.push(rows);
            index_of.push(index);
        }
        let built = self
            .ctx
            .kv_rows_layers(&self.stream, &held.tables, &per_layer)?;
        let mut tables: Vec<Option<KvRows<'a>>> = (0..LAYERS).map(|_| None).collect();
        for (index, table) in index_of.into_iter().zip(built) {
            tables[index] = Some(table);
        }
        Ok(tables)
    }

    /// Steps one token of each sequence together, as [`Self::decode_rows`], and leaves the capped
    /// logits rows on the device.
    fn step_window(
        &mut self,
        seqs: &[&BoltSeq],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<(), EngError> {
        let m = seqs.len();
        if m != tokens.len() || m != positions.len() {
            return Err(EngError::BadRequest(
                "a Bolt window needs one token and one position per sequence".to_owned(),
            ));
        }
        if !(2..=Self::WINDOW_ROWS).contains(&m) {
            return Err(EngError::BadRequest(format!(
                "a Bolt window holds 2 to {} rows, got {m}",
                Self::WINDOW_ROWS
            )));
        }
        if self.window.as_ref().is_none_or(|held| held.rows < m) {
            return Err(EngError::BadRequest(
                "the Bolt window buffers are not reserved for this many rows".to_owned(),
            ));
        }
        if seqs
            .iter()
            .zip(positions)
            .any(|(seq, &pos)| pos >= seq.cells)
        {
            return Err(EngError::ContextLimit(
                "the Bolt decode window exceeds the context".to_owned(),
            ));
        }
        self.grow_scratch(m)?;
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.scratch.ids.write_at_enqueue(&self.stream, 0, &ids)?;
        for (seq, &pos) in seqs.iter().zip(positions) {
            // Position at byte 4 and key count at byte 8 of the step block, in one copy.
            let mut words = [0u8; 8];
            words[..4].copy_from_slice(&(pos as i32).to_le_bytes());
            words[4..].copy_from_slice(&((pos as i32) + 1).to_le_bytes());
            seq.step.write_at_enqueue(&self.stream, 4, &words)?;
        }
        {
            let tables = self.window_tables(seqs)?;
            self.decode_body(
                &Rows::Window {
                    seqs,
                    tables: &tables,
                },
                true,
                m,
            )?;
        }
        self.graph_mode.note_eager();
        Ok(())
    }

    /// The candidates a draw ranks per row: Bolt's default `top_k` fits whole.
    pub(crate) const DRAW_WIDTH: usize = shortlist::WIDE_K;

    /// Steps one token and draws the next on the device: the capped logits are ranked and drawn
    /// where they are, and one word leaves the card instead of the row.
    pub(crate) fn decode_draw(
        &mut self,
        seq: &BoltSeq,
        token: u32,
        pos: usize,
        sampler: &crate::family::sampler::Sampler,
    ) -> Result<u32, EngError> {
        self.decode(seq, token, pos)?;
        let route =
            crate::family::decode_step::draw::route(sampler, pos as u64, Self::DRAW_WIDTH, VOCAB);
        Ok(self
            .draw
            .draw(&self.ctx, &self.stream, &self.logits_capped.buf, &[route])?[0])
    }

    /// Steps one token of each sequence together, as [`Self::decode_rows`], and draws the next token
    /// of each on the device, each from its own sampler at its own position.
    pub(crate) fn decode_rows_draw(
        &mut self,
        seqs: &[&BoltSeq],
        tokens: &[u32],
        positions: &[usize],
        samplers: &[&crate::family::sampler::Sampler],
    ) -> Result<Vec<u32>, EngError> {
        if samplers.len() != seqs.len() {
            return Err(EngError::BadRequest(
                "a Bolt window needs one sampler per sequence".to_owned(),
            ));
        }
        self.step_window(seqs, tokens, positions)?;
        let routes: Vec<_> = positions
            .iter()
            .zip(samplers)
            .map(|(&pos, sampler)| {
                crate::family::decode_step::draw::route(
                    sampler,
                    pos as u64,
                    Self::DRAW_WIDTH,
                    VOCAB,
                )
            })
            .collect();
        let window = self.window.as_ref().expect("checked above");
        self.draw
            .draw(&self.ctx, &self.stream, &window.capped.buf, &routes)
    }

    /// Arms or disarms the decode graph for this model (the environment
    /// sets the load-time default; the tests flip it between runs).
    pub(crate) fn use_decode_graph(&mut self, on: bool) {
        self.graph_mode.set_on(on);
    }

    /// Arms or disarms the split decode attention for this model. The
    /// environment sets the load-time default; the qualification tests flip
    /// it between runs. Arming outside capture allocates the partial buffer
    /// on first use (the env-armed load already counted it in the device
    /// budget; this post-load path is test and qualification traffic). Any
    /// captured decode graph is dropped: it was captured against the previous
    /// split state, and its replay would read a freed partial buffer when the
    /// split is disarmed (or never split at all when armed after capture).
    pub(crate) fn use_attn_split(&mut self, on: bool) -> Result<(), EngError> {
        self.ctx.ensure_usable()?;
        self.ctx.ensure_uncaptured("attn split partial buffer")?;
        self.graph = None;
        if on {
            if self.scratch.attn_part.is_none() {
                self.scratch.attn_part = Some(zero(
                    &self.ctx,
                    attn_part_bytes(self.attn_splits, self.rows),
                )?);
            }
        } else {
            self.scratch.attn_part = None;
        }
        Ok(())
    }

    /// Whether a captured decode graph is held (the split invalidation pin).
    pub(crate) fn graph_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// Whether the attention split is armed: the partial buffer exists. The
    /// disarm assert pins on this, not on `graph_captured`, which the disarm
    /// does not change (arming already dropped any graph).
    pub(crate) fn split_armed(&self) -> bool {
        self.scratch.attn_part.is_some()
    }

    /// The capped logits of the newest logits-bearing forward, for host
    /// sampling. The tie to `logits_capped` (not `logits`) matches the
    /// model's own output distribution.
    pub(crate) fn capped_logits(&self) -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; VOCAB * 4];
        self.ctx.download(&self.logits_capped.buf, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    /// Debug parity hook: with BOLT_TRACE_DIR set, one raw f32 vector lands per
    /// checkpoint (embed, each layer's output) as p{pos}_{name}.f32. No-op
    /// without the variable; a parity comparison reads the dumps.
    fn trace(&self, pos: usize, name: &str, v: &Vector, bytes: usize) {
        let Some(dir) = trace_dir() else {
            return;
        };
        let mut raw = vec![0u8; bytes];
        if self.ctx.download(&v.buf, &mut raw).is_ok() {
            let _ = std::fs::write(
                std::path::Path::new(dir).join(format!("p{pos}_{name}.f32")),
                raw,
            );
        }
    }

    /// Runs a bounded prompt batch. Only its last row can produce logits.
    /// Scratch grows once; each batch synchronizes once at the end when no
    /// logits are drawn.
    pub(crate) fn prefill_batch(
        &mut self,
        seq: &BoltSeq,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<(), EngError> {
        if !(1..=self.rows).contains(&tokens.len()) {
            return Err(EngError::BadRequest(
                "a Bolt prefill batch needs 1 to its planned rows".to_owned(),
            ));
        }
        if pos
            .checked_add(tokens.len())
            .is_none_or(|end| end > seq.cells)
        {
            return Err(EngError::ContextLimit(
                "the Bolt prefill batch exceeds the context".to_owned(),
            ));
        }
        self.grow_scratch(tokens.len())?;
        self.decode_with_logits(seq, tokens, pos, logits)?;
        if !logits {
            self.stream.synchronize()?;
        }
        Ok(())
    }

    fn grow_scratch(&mut self, rows: usize) -> Result<(), EngError> {
        // A buffer replacement invalidates a captured body (it baked the old
        // address), so the graph drops when a replacement actually happens -
        // not on every call, or no prefill would ever leave the graph armed.
        // (The load-time capacity already covers every planned prefill chunk, so
        // nothing replaces in practice.)
        let replaced = self.grow_scratch_buffers(rows)?;
        if replaced {
            self.graph = None;
        }
        Ok(())
    }

    /// Grows every scratch buffer whose capacity is below `rows` up to the
    /// planned prefill capacity, reporting whether any buffer was replaced.
    fn grow_scratch_buffers(&mut self, rows: usize) -> Result<bool, EngError> {
        let mut replaced = false;
        let sc = &mut self.scratch;
        for (v, width) in sc.h.iter_mut().map(|v| (v, HIDDEN)).chain([
            (&mut sc.nrm, HIDDEN),
            (&mut sc.nrm2, HIDDEN),
            (&mut sc.q_normed, HEADS * GLOBAL_HEAD),
            (&mut sc.k_normed, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.v_normed, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.q, HEADS * GLOBAL_HEAD),
            (&mut sc.k, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.v, KV_HEADS * GLOBAL_HEAD),
            (&mut sc.ctx_out, HEADS * GLOBAL_HEAD),
            (&mut sc.proj, HIDDEN),
            (&mut sc.gate, FFN),
            (&mut sc.up, FFN),
            (&mut sc.mixed, FFN),
            (&mut sc.down, HIDDEN),
            (&mut sc.ple_all, LAYERS * PLE),
            (&mut sc.ple_scaled, LAYERS * PLE),
            (&mut sc.ple_nrm, PLE),
            (&mut sc.ple_tbl, PLE),
            (&mut sc.ple_mixed, PLE),
        ]) {
            if v.len < rows * width {
                let len = self.rows * width;
                v.buf = self.ctx.allocate(len * 4)?;
                v.len = len;
                replaced = true;
            }
        }
        for (buffer, width) in [
            (&mut sc.q8_hidden, HIDDEN),
            (&mut sc.q8_ctx, HEADS * GLOBAL_HEAD),
            (&mut sc.q8_mixed, FFN),
            (&mut sc.q8_ple_mixed, PLE),
        ] {
            if buffer.bytes() < q8_1_bytes(width, rows).expect("bounded q8 rows") {
                *buffer = self
                    .ctx
                    .allocate(q8_1_bytes(width, self.rows).expect("bounded q8 rows"))?;
                replaced = true;
            }
        }
        Ok(replaced)
    }

    /// The projections of one group over one activation, `x` as f32 rows and `x_q8` as q8_1 rows:
    /// one `(weight, y)` per projection. A chunk of 16 rows or more runs the group on the tensor
    /// cores when every weight of it rests in one type that has a dense case, since one quantized
    /// activation serves one layout; any other group runs each weight through the matvec.
    fn project_group(
        &self,
        group: &[(&NativeWeight, &Vector)],
        x: &Vector,
        x_q8: &DeviceBuffer,
        m: usize,
    ) -> Result<(), EngError> {
        let first = group[0].0;
        let wide = on_tensor_cores(group.iter().map(|(weight, _)| weight.quant()), m);
        match self.mmq.as_ref().filter(|_| wide) {
            Some(gemm) => {
                let rows = x.buf.range(0, m * first.k() * 4)?;
                gemm.input(&self.ctx, &self.stream, first, rows, m)?;
                group.iter().try_for_each(|(weight, y)| {
                    let y = y.buf.range(0, m * weight.n_rows() * 4)?;
                    gemm.project(&self.ctx, &self.stream, weight, m, y)
                })
            }
            None => group
                .iter()
                .try_for_each(|(weight, y)| self.matvec(weight, weight.n_rows(), x_q8, y, m)),
        }
    }

    fn matvec(
        &self,
        weight: &NativeWeight,
        rows: usize,
        x_q8: &DeviceBuffer,
        y: &Vector,
        m: usize,
    ) -> Result<(), EngError> {
        matvec(&self.ctx, &self.stream, weight, rows, x_q8, y, m)
    }

    /// Keeps the existing norm reduction and quantization order for every activation row.
    fn norm_rows<'b>(
        &self,
        stream: &DeviceStream,
        h: DeviceRange<'b>,
        weight: DeviceRange<'b>,
        norm_out: DeviceRange<'b>,
        q8: DeviceRange<'b>,
        rows: usize,
    ) -> Result<(), EngError> {
        self.ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h,
                delta: None,
                weight,
                norm_out,
                q8,
                rows,
            },
        )
    }

    /// Runs one decode step for the token at the position and leaves the capped
    /// logits in the device buffer. Synchronizes once at the end.
    pub(crate) fn decode(&mut self, seq: &BoltSeq, token: u32, pos: usize) -> Result<(), EngError> {
        self.decode_with_logits(seq, &[token], pos, true)
    }

    /// Runs one step without the output norm and the logits projection, for prompt tokens
    /// whose logits are never read. It queues the step and does not synchronize.
    /// The shared-KV layers are skipped: they write no cache cells, and their
    /// hidden state only feeds the final norm, so a token whose logits are
    /// never read does not need them.
    pub(crate) fn decode_hidden(
        &mut self,
        seq: &BoltSeq,
        token: u32,
        pos: usize,
    ) -> Result<(), EngError> {
        self.decode_with_logits(seq, &[token], pos, false)
    }

    fn decode_with_logits(
        &mut self,
        seq: &BoltSeq,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<(), EngError> {
        let m = tokens.len();
        if pos.checked_add(m).is_none_or(|end| end > seq.cells) {
            return Err(EngError::ContextLimit(
                "the Bolt decode step exceeds the context".to_owned(),
            ));
        }
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        // The step words are written in place, never reallocated: a captured
        // decode graph bakes their addresses, and the eager path saves the
        // three per-step allocations the uploads made.
        self.scratch.ids.write_at_enqueue(&self.stream, 0, &ids)?;
        seq.pos
            .write_at_enqueue(&self.stream, 0, &(pos as i32).to_le_bytes())?;
        seq.kv_len
            .write_at_enqueue(&self.stream, 0, &((pos as i32) + 1).to_le_bytes())?;
        let rows = Rows::Chunk { seq, pos };
        let facts = DecodeGraphFacts {
            trace_armed: trace_dir().is_some(),
            multi_token: m > 1,
            hidden: !logits,
        };
        if check_decode_graph(facts).is_ok() {
            // The mode gates the replay too: a disarmed (or refused) mode must
            // not keep replaying a graph it captured earlier.
            if let Some(exec) = self
                .graph
                .as_ref()
                .filter(|_| self.graph_mode.active() && self.graph_seq == seq.id)
            {
                exec.launch(&self.stream)?;
                self.graph_mode.note_replay();
                return self.stream.synchronize();
            }
            if self.graph_mode.active() && self.graph_mode.warm() {
                match self.capture_decode_graph(seq, pos) {
                    Ok(()) => {
                        self.graph_mode.note_replay();
                        return self.stream.synchronize();
                    }
                    Err(failure) if is_capture_refusal(&failure) => {
                        // The body never ran; the sticky refusal makes every
                        // later step eager, and this one runs below.
                        self.graph_mode.refuse(&failure.describe());
                    }
                    Err(failure) => return Err(failure),
                }
            }
        }
        self.decode_body(&rows, logits, m)?;
        self.graph_mode.note_eager();
        if check_decode_graph(facts).is_ok() {
            // One clean eager step warms every kernel the capture records
            // (a first launch loads its module, which a capture forbids).
            self.graph_mode.note_warm();
        }
        self.stream.synchronize()
    }

    /// The launches of one decode step, from the embedding to the capped
    /// logits: everything a replay re-runs, and the body a capture records.
    /// No per-step scalar but the trace label's position reaches the kernels
    /// through arguments; the kernels that depend on the position or the KV
    /// length read the step words.
    fn decode_body(&self, rows: &Rows<'_>, logits: bool, m: usize) -> Result<(), EngError> {
        let pos = rows.trace_pos();
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let h0 = &sc.h[0];
        let h1 = &sc.h[1];
        let h2 = &sc.h[2];
        let h3 = &sc.h[3];
        ctx.embed_native(
            stream,
            EmbedNativeCall {
                weight: &self.embed,
                ids: sc.ids.range(0, m * 4)?,
                x: h0.buf.range(0, h0.bytes())?,
                status: sc.status.range(0, 4)?,
                m,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: h1.buf.range(0, h1.bytes())?,
                x: h0.buf.range(0, h0.bytes())?,
                scale: embed_scale(),
                m,
                n: HIDDEN,
            },
        )?;
        // K/V-only prefill: only the cache-writing prefix of layers runs when
        // no logits are read; the shared-KV layers after it neither write cache
        // cells nor feed anything but the final norm.
        let kv_prefix = if logits {
            LAYERS
        } else {
            self.layers
                .iter()
                .position(|layer| !layer.physical())
                .unwrap_or(LAYERS)
        };
        self.per_layer_inputs(h1, m, kv_prefix * PLE)?;
        self.trace(pos, "embed", h1, m * HIDDEN * 4);
        let h = [h0, h1, h2, h3];
        let mut current = 1usize;
        for index in 0..kv_prefix {
            let (input, r1, r2, out) = layer_buffers(current);
            self.decode_layer(
                index,
                h[input],
                h[r1],
                h[r2],
                h[out],
                rows,
                kv_prefix * PLE,
                m,
            )?;
            current = out;
        }
        if !logits {
            // Only the last prompt token contributes logits, so the output norm and the
            // tied-embedding projection are skipped for the rest.
            return Ok(());
        }
        let input = h[current];
        if let Rows::Window { .. } = rows {
            return self.window_head(input, m);
        }
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: input.buf.range((m - 1) * HIDDEN * 4, HIDDEN * 4)?,
                delta: None,
                weight: self.out_norm.buf.range(0, self.out_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, sc.nrm.bytes())?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                rows: 1,
            },
        )?;
        ctx.matvec_native(
            stream,
            &self.embed,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            self.logits.buf.range(0, self.logits.bytes())?,
            1,
        )?;
        ctx.softcap(
            stream,
            SoftcapCall {
                out: self
                    .logits_capped
                    .buf
                    .range(0, self.logits_capped.bytes())?,
                x: self.logits.buf.range(0, self.logits.bytes())?,
                cap: SOFT_CAP,
                n: VOCAB,
            },
        )?;
        self.trace(pos, "logits", &self.logits_capped, VOCAB * 4);
        Ok(())
    }

    /// Captures the one-token logits body and replays it once: a capture
    /// records the work without running it, so this step's token is produced
    /// by the first replay. A refusal propagates to the caller, which notes it
    /// sticky and runs the body eagerly.
    fn capture_decode_graph(&mut self, seq: &BoltSeq, pos: usize) -> Result<(), EngError> {
        let plan = CapturePlan {
            pins: &[],
            holds: Vec::new(),
        };
        let rows = Rows::Chunk { seq, pos };
        let (exec, ()) = self
            .ctx
            .capture(&self.stream, plan, || self.decode_body(&rows, true, 1))?;
        exec.launch(&self.stream)?;
        self.graph = Some(exec);
        self.graph_seq = seq.id;
        Ok(())
    }

    /// The head of a window: the output norm over every row, the tied-embedding projection in
    /// launches of at most [`HEAD_COLUMNS`] columns, and the softcap over all rows.
    fn window_head(&self, input: &Vector, m: usize) -> Result<(), EngError> {
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the Bolt window buffers are absent".to_owned()))?;
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: HIDDEN,
                eps: self.eps,
                h: input.buf.range(0, m * HIDDEN * 4)?,
                delta: None,
                weight: self.out_norm.buf.range(0, self.out_norm.bytes())?,
                norm_out: sc.nrm.buf.range(0, m * HIDDEN * 4)?,
                q8: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                rows: m,
            },
        )?;
        let row_bytes = q8_1_bytes(HIDDEN, 1).expect("hidden q8 row size");
        for first in (0..m).step_by(HEAD_COLUMNS) {
            let columns = HEAD_COLUMNS.min(m - first);
            ctx.matvec_native(
                stream,
                &self.embed,
                sc.q8_hidden.range(first * row_bytes, columns * row_bytes)?,
                window
                    .logits
                    .buf
                    .range(first * VOCAB * 4, columns * VOCAB * 4)?,
                columns,
            )?;
        }
        ctx.softcap(
            stream,
            SoftcapCall {
                out: window.capped.buf.range(0, m * VOCAB * 4)?,
                x: window.logits.buf.range(0, m * VOCAB * 4)?,
                cap: SOFT_CAP,
                n: m * VOCAB,
            },
        )
    }

    /// Projects the scaled embedding into the per-layer planes and scales by
    /// hidden^-0.5; the per-layer 256-wide norms and table sums run inside each
    /// layer, where the result is consumed.
    fn per_layer_inputs(&self, embed: &Vector, m: usize, ple_rows: usize) -> Result<(), EngError> {
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: embed.buf.range(0, m * HIDDEN * 4)?,
                y: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                k: HIDDEN,
                ncols: m,
            },
        )?;
        self.matvec(&self.ple_model, ple_rows, &sc.q8_hidden, &sc.ple_all, m)?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: sc.ple_scaled.buf.range(0, sc.ple_scaled.bytes())?,
                x: sc.ple_all.buf.range(0, sc.ple_all.bytes())?,
                scale: (HIDDEN as f32).recip().sqrt(),
                m,
                n: ple_rows,
            },
        )
    }

    /// Rotation, cache append and attention of one layer. A chunk runs them once over its
    /// consecutive rows against its sequence. A window rotates each row against its own position
    /// word, then runs the append and the attention once over all rows, each reading its own
    /// planes, cells, position and key count from its record.
    fn rope_append_attend(&self, index: usize, rows: &Rows<'_>, m: usize) -> Result<(), EngError> {
        let layer = &self.layers[index];
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let head = layer.head;
        let q_rows = HEADS * head;
        let kv_rows = KV_HEADS * head;
        let physical = layer.physical();
        let source = layer.kv_source.unwrap_or(index);
        let pos = rows.trace_pos();
        let attn_split = sc.attn_part.as_ref().map(|part| {
            let bytes = attn_part_bytes(self.attn_splits, self.rows);
            AttnDecodeGemmaSplit {
                // Sized at load for the full prefill batch; the range cannot fail.
                part: part.range(0, bytes).expect("partial buffer"),
                max_splits: self.attn_splits,
                min_keys: ATTN_SPLIT_MIN_KEYS,
            }
        });
        // The rotation of `count` rows starting at row `row` of the scratch, the position read
        // from `pos_word`.
        let rotate =
            |row: usize, count: usize, pos_word: DeviceRange<'_>| -> Result<(), EngError> {
                let (at, bytes) = (row * q_rows * 4, count * q_rows * 4);
                let (kat, kbytes) = (row * kv_rows * 4, count * kv_rows * 4);
                // A shared layer rotates its query only; the rope kernel requires at least one k
                // row, so it rotates one scratch row that no later launch reads.
                ctx.rope_gemma(
                    stream,
                    RopeGemmaCall {
                        q: sc.q_normed.buf.range(at, bytes)?,
                        k: if physical {
                            sc.k_normed.buf.range(kat, kbytes)?
                        } else {
                            sc.k_normed.buf.range(0, kv_rows * 4)?
                        },
                        pos: pos_word,
                        factors: match (&self.rope_freqs, layer.global) {
                            (Some(freqs), true) => Some(freqs.buf.range(0, freqs.bytes())?),
                            _ => None,
                        },
                        rows_q: count,
                        rows_k: if physical { count } else { 1 },
                        n_q_head: HEADS,
                        n_kv_head: KV_HEADS,
                        head_dim: head,
                        global_mode: layer.global,
                    },
                )
            };
        // `count` consecutive rows of `seq` starting at row `row` of the scratch, whose first
        // row sits at cell `first`.
        let group = |seq: &BoltSeq,
                     row: usize,
                     count: usize,
                     first: usize|
         -> Result<(), EngError> {
            let own = seq.kv[source]
                .as_ref()
                .expect("the shared-KV source layer is physical");
            // One chunk is appended whole before it is attended, so a sliding layer's ring
            // must hold the window plus the chunk: a wider chunk would overwrite cells its own
            // first rows still read. The ring is sized from `rows`, the widest chunk
            // `prefill_batch` takes, so this refuses a caller that bypassed that limit.
            if !own.geometry.admits(count) {
                return Err(EngError::BadRequest(format!(
                    "a Bolt append of {count} rows is wider than the sliding ring's margin of {}",
                    own.geometry.widest_append()
                )));
            }
            let (at, bytes) = (row * q_rows * 4, count * q_rows * 4);
            let (kat, kbytes) = (row * kv_rows * 4, count * kv_rows * 4);
            rotate(row, count, seq.pos.range(0, 4)?)?;
            if physical {
                let data = Kv8Data {
                    k: sc.k_normed.buf.range(kat, kbytes)?,
                    v: sc.v_normed.buf.range(kat, kbytes)?,
                    k_codes: own.k_codes.range(0, own.k_codes.bytes())?,
                    k_scales: own.k_scales.range(0, own.k_scales.bytes())?,
                    v_codes: own.v_codes.range(0, own.v_codes.bytes())?,
                    v_scales: own.v_scales.range(0, own.v_scales.bytes())?,
                };
                if count == 1 {
                    // The decode entry: the device pos word is authoritative, so a captured
                    // replay writes the cell of the step it replays instead of the
                    // capture-time cell. The status word satisfies the launch contract; the
                    // caller's host check that `pos + m` leaves the context makes the
                    // kernel's refuse path unreachable, so the word is never read back.
                    ctx.kv8_append(
                        stream,
                        Kv8Shape {
                            rows: 1,
                            ..own.shape
                        },
                        data,
                        seq.pos.range(0, 4)?,
                        sc.status.range(0, 4)?,
                    )?;
                } else {
                    // The prefill chunk entry takes its position as a host scalar; it is never
                    // captured (a multi-token step refuses arming), so the baked-scalar hazard
                    // does not arise.
                    ctx.kv8_chunk(
                        stream,
                        Kv8Shape {
                            rows: count,
                            ..own.shape
                        },
                        data,
                        first,
                    )?;
                }
            }
            self.trace(
                pos,
                &format!("L{index}_v"),
                &sc.v_normed,
                (row + count) * kv_rows * 4,
            );
            // A prompt chunk of two or more rows on a global layer runs the wide prefill
            // attention when it is on; a lone row, and so every decode step, keeps the decode
            // attention.
            let call = WideCall::new(
                [&own.k_codes, &own.k_scales, &own.v_codes, &own.v_scales],
                sc.q_normed.buf.range(at, bytes)?,
                sc.ctx_out.buf.range(at, bytes)?,
                count,
                first,
            )
            .with_cells(own.shape.cells);
            let take = layer.global && count > 1;
            if WideAttention::try_attend(self.wide_prefill.as_ref(), take, ctx, stream, call)? {
                return Ok(());
            }
            ctx.attn_decode_gemma(
                stream,
                AttnDecodeGemmaCall {
                    out: sc.ctx_out.buf.range(at, bytes)?,
                    status: None,
                    q: sc.q_normed.buf.range(at, bytes)?,
                    k_codes: own.k_codes.range(0, own.k_codes.bytes())?,
                    k_scales: own.k_scales.range(0, own.k_scales.bytes())?,
                    v_codes: own.v_codes.range(0, own.v_codes.bytes())?,
                    v_scales: own.v_scales.range(0, own.v_scales.bytes())?,
                    kv_len: seq.kv_len.range(0, 4)?,
                    m: count,
                    n_q_head: HEADS,
                    n_kv_head: KV_HEADS,
                    head_dim: head,
                    cells: own.shape.cells,
                    window: if layer.global { 0 } else { self.local_window },
                    scale: 1.0,
                    split: attn_split,
                    ring: own.shape.ring,
                },
            )
        };
        match rows {
            Rows::Chunk { seq, pos } => group(seq, 0, m, *pos),
            Rows::Window { seqs, tables } => {
                // Each row rotates against its own position word; the cache append and the
                // attention then run once over all rows, each reading its own planes, cells,
                // position and key count from its record.
                for (row, seq) in seqs.iter().enumerate() {
                    rotate(row, 1, seq.step.range(4, 4)?)?;
                }
                let table = tables[source]
                    .as_ref()
                    .expect("a window table for every layer that owns planes");
                let rows = seqs.len();
                let own = seqs[0].kv[source]
                    .as_ref()
                    .expect("the shared-KV source layer is physical");
                if physical {
                    ctx.kv8_append_multi(
                        stream,
                        AppendMultiCall {
                            shape: Kv8Shape { rows, ..own.shape },
                            k: sc.k_normed.buf.range(0, rows * kv_rows * 4)?,
                            v: sc.v_normed.buf.range(0, rows * kv_rows * 4)?,
                            rows: table,
                        },
                    )?;
                }
                ctx.attn_decode_gemma_multi(
                    stream,
                    DecodeGemmaMultiCall {
                        n_q_head: HEADS,
                        n_kv_head: KV_HEADS,
                        head_dim: head,
                        window: if layer.global { 0 } else { self.local_window },
                        ring: own.shape.ring,
                        scale: 1.0,
                        q: sc.q_normed.buf.range(0, rows * q_rows * 4)?,
                        rows: table,
                        split: attn_split,
                        out: sc.ctx_out.buf.range(0, rows * q_rows * 4)?,
                    },
                )
            }
        }
    }

    /// Turns the wide prefill attention on or off for this model, for a comparison on one
    /// load; returns whether the device has it at all.
    #[cfg(test)]
    pub(crate) fn use_wide_prefill(&self, on: bool) -> bool {
        self.wide_prefill.as_ref().is_some_and(|wide| {
            wide.set_on(on);
            true
        })
    }

    /// How many wide prefill attention launches this model has queued.
    #[cfg(test)]
    pub(crate) fn wide_prefill_launches(&self) -> u64 {
        self.wide_prefill
            .as_ref()
            .map_or(0, WideAttention::launches)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the residual rotation keeps four separate buffers"
    )]
    fn decode_layer(
        &self,
        index: usize,
        input: &Vector,
        r1: &Vector,
        r2: &Vector,
        out: &Vector,
        rows: &Rows<'_>,
        ple_stride: usize,
        m: usize,
    ) -> Result<(), EngError> {
        let pos = rows.trace_pos();
        let layer = &self.layers[index];
        let ctx = &self.ctx;
        let stream = &self.stream;
        let sc = &self.scratch;
        let head = layer.head;
        let q_rows = HEADS * head;
        let kv_rows = KV_HEADS * head;
        let shared = !layer.physical();
        self.norm_rows(
            stream,
            input.buf.range(0, input.bytes())?,
            layer.attn_norm.buf.range(0, layer.attn_norm.bytes())?,
            sc.nrm.buf.range(0, sc.nrm.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        match (&layer.k, &layer.v) {
            (Some(k), Some(v)) => self.project_group(
                &[(&layer.q, &sc.q), (k, &sc.k), (v, &sc.v)],
                &sc.nrm,
                &sc.q8_hidden,
                m,
            )?,
            _ => self.project_group(&[(&layer.q, &sc.q)], &sc.nrm, &sc.q8_hidden, m)?,
        }
        // The QK-norm of the reference graph: each head of q and of k is
        // RMS-normalized and scaled by its own weight before RoPE. Values get the
        // reference's unweighted per-head RMS norm before their cache write; the
        // ones vector makes the weighted kernel compute the unweighted norm. The
        // head rows are contiguous, so each loop is ONE batched launch; the q8
        // side of the kernel is a dump nobody reads.
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n: head,
                eps: self.eps,
                h: sc.q.buf.range(0, m * q_rows * 4)?,
                delta: None,
                weight: layer.q_norm.buf.range(0, head * 4)?,
                norm_out: sc.q_normed.buf.range(0, m * q_rows * 4)?,
                q8: sc.q8_qk.range(0, sc.q8_qk.bytes())?,
                rows: m * HEADS,
            },
        )?;
        if !shared {
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.k.buf.range(0, m * kv_rows * 4)?,
                    delta: None,
                    weight: layer
                        .k_norm
                        .as_ref()
                        .expect("physical layers carry a k norm")
                        .buf
                        .range(0, head * 4)?,
                    norm_out: sc.k_normed.buf.range(0, m * kv_rows * 4)?,
                    q8: sc.q8_qk.range(0, sc.q8_qk.bytes())?,
                    rows: m * KV_HEADS,
                },
            )?;
            ctx.add_rmsnorm_q8(
                stream,
                AddNormQ8Call {
                    n: head,
                    eps: self.eps,
                    h: sc.v.buf.range(0, m * kv_rows * 4)?,
                    delta: None,
                    weight: sc.ones.buf.range(0, head * 4)?,
                    norm_out: sc.v_normed.buf.range(0, m * kv_rows * 4)?,
                    q8: sc.q8_qk.range(0, sc.q8_qk.bytes())?,
                    rows: m * KV_HEADS,
                },
            )?;
        }
        self.rope_append_attend(index, rows, m)?;
        self.trace(pos, &format!("L{index}_ctx"), &sc.ctx_out, m * q_rows * 4);
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.ctx_out.buf.range(0, m * q_rows * 4)?,
                y: sc.q8_ctx.range(0, sc.q8_ctx.bytes())?,
                k: q_rows,
                ncols: m,
            },
        )?;
        self.project_group(&[(&layer.out, &sc.proj)], &sc.ctx_out, &sc.q8_ctx, m)?;
        self.trace(pos, &format!("L{index}_proj"), &sc.proj, m * HIDDEN * 4);
        self.norm_rows(
            stream,
            sc.proj.buf.range(0, m * HIDDEN * 4)?,
            layer
                .post_attn_norm
                .buf
                .range(0, layer.post_attn_norm.bytes())?,
            sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        self.trace(pos, &format!("L{index}_nrm"), &sc.nrm2, m * HIDDEN * 4);
        ctx.row_add(
            stream,
            RowAddCall {
                out: r1.buf.range(0, r1.bytes())?,
                x: input.buf.range(0, input.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
                flag: None,
                fused: false,
            },
        )?;
        self.trace(pos, &format!("L{index}_r1"), r1, m * HIDDEN * 4);
        self.norm_rows(
            stream,
            r1.buf.range(0, r1.bytes())?,
            layer.ffn_norm.buf.range(0, layer.ffn_norm.bytes())?,
            sc.nrm.buf.range(0, sc.nrm.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        self.project_group(
            &[(&layer.ffn_gate, &sc.gate), (&layer.ffn_up, &sc.up)],
            &sc.nrm,
            &sc.q8_hidden,
            m,
        )?;
        ctx.gelu_ffn(
            stream,
            GeluFfnCall {
                out: sc.mixed.buf.range(0, m * FFN * 4)?,
                gate: sc.gate.buf.range(0, m * FFN * 4)?,
                up: sc.up.buf.range(0, m * FFN * 4)?,
                m,
                n: FFN,
            },
        )?;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.mixed.buf.range(0, m * FFN * 4)?,
                y: sc.q8_mixed.range(0, sc.q8_mixed.bytes())?,
                k: FFN,
                ncols: m,
            },
        )?;
        self.project_group(&[(&layer.ffn_down, &sc.down)], &sc.mixed, &sc.q8_mixed, m)?;
        self.norm_rows(
            stream,
            sc.down.buf.range(0, m * HIDDEN * 4)?,
            layer
                .post_ffw_norm
                .buf
                .range(0, layer.post_ffw_norm.bytes())?,
            sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: r2.buf.range(0, r2.bytes())?,
                x: r1.buf.range(0, r1.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
                flag: None,
                fused: false,
            },
        )?;
        self.trace(pos, &format!("L{index}_r2"), r2, m * HIDDEN * 4);
        // The per-layer-input block: the gated mixing of the second residual with
        // this layer's token-identity input, projected back to the hidden width.
        // The skipped-shared prefill packs the per-layer planes at the prefix stride, not the
        // full 42-layer stride: the rows of this layer's slice sit `ple_stride` apart.
        ctx.add_rmsnorm_q8_strided(
            stream,
            AddNormQ8Call {
                n: PLE,
                eps: self.eps,
                h: sc
                    .ple_scaled
                    .buf
                    .range(index * PLE * 4, ((m - 1) * ple_stride + PLE) * 4)?,
                delta: None,
                weight: self
                    .ple_model_norm
                    .buf
                    .range(0, self.ple_model_norm.bytes())?,
                norm_out: sc.ple_nrm.buf.range(0, m * PLE * 4)?,
                q8: sc.q8_head.range(0, sc.q8_head.bytes())?,
                rows: m,
            },
            ple_stride,
        )?;
        ctx.embed_native(
            stream,
            EmbedNativeCall {
                weight: &layer.table,
                ids: sc.ids.range(0, m * 4)?,
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                status: sc.status.range(0, 4)?,
                m,
            },
        )?;
        self.trace(pos, &format!("L{index}_plep"), &sc.ple_nrm, m * PLE * 4);
        self.trace(pos, &format!("L{index}_pletbl"), &sc.ple_tbl, m * PLE * 4);
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                scale: PLE_IDENTITY_SCALE,
                m,
                n: PLE,
            },
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                x: sc.ple_nrm.buf.range(0, m * PLE * 4)?,
                y: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                n: m * PLE,
                flag: None,
                fused: false,
            },
        )?;
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                scale: PLE_COMBINED_SCALE,
                m,
                n: PLE,
            },
        )?;
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: r2.buf.range(0, m * HIDDEN * 4)?,
                y: sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
                k: HIDDEN,
                ncols: m,
            },
        )?;
        self.matvec(&layer.ple_gate, PLE, &sc.q8_hidden, &sc.ple_nrm, m)?;
        ctx.gelu_ffn(
            stream,
            GeluFfnCall {
                out: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                gate: sc.ple_nrm.buf.range(0, m * PLE * 4)?,
                up: sc.ple_mixed.buf.range(0, m * PLE * 4)?,
                m,
                n: PLE,
            },
        )?;
        self.trace(pos, &format!("L{index}_plegate"), &sc.ple_nrm, m * PLE * 4);
        self.trace(pos, &format!("L{index}_plemix"), &sc.ple_tbl, m * PLE * 4);
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: sc.ple_tbl.buf.range(0, m * PLE * 4)?,
                y: sc.q8_ple_mixed.range(0, sc.q8_ple_mixed.bytes())?,
                k: PLE,
                ncols: m,
            },
        )?;
        self.matvec(&layer.ple_proj, HIDDEN, &sc.q8_ple_mixed, &sc.proj, m)?;
        self.trace(pos, &format!("L{index}_pled"), &sc.proj, m * HIDDEN * 4);
        self.norm_rows(
            stream,
            sc.proj.buf.range(0, m * HIDDEN * 4)?,
            layer
                .post_ple_norm
                .buf
                .range(0, layer.post_ple_norm.bytes())?,
            sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
            sc.q8_hidden.range(0, sc.q8_hidden.bytes())?,
            m,
        )?;
        ctx.row_add(
            stream,
            RowAddCall {
                out: input.buf.range(0, input.bytes())?,
                x: r2.buf.range(0, r2.bytes())?,
                y: sc.nrm2.buf.range(0, sc.nrm2.bytes())?,
                n: m * HIDDEN,
                flag: None,
                fused: false,
            },
        )?;
        // The scaled layer output lands in `out`; `input` held the dead pre-layer
        // residual, so the scale has a distinct source buffer.
        ctx.row_scale(
            stream,
            RowScaleCall {
                out: out.buf.range(0, out.bytes())?,
                x: input.buf.range(0, input.bytes())?,
                scale: layer.scale,
                m,
                n: HIDDEN,
            },
        )?;
        self.trace(pos, &format!("L{index}"), out, m * HIDDEN * 4);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_file_serves_no_more_cells_than_its_header_gives_as_context() {
        use crate::tokenizer::gguf::tests::{gguf_file, kv_u32_for_tests};
        let dir = tempfile::tempdir().expect("tempdir");
        let at = |name: &str, kvs: Vec<Vec<u8>>| {
            let path = dir.path().join(name);
            std::fs::write(&path, gguf_file(&kvs)).expect("write");
            header_context(&path)
        };
        assert_eq!(
            at(
                "short.gguf",
                vec![kv_u32_for_tests("gemma4.context_length", 4096)]
            ),
            Some(4096)
        );
        assert_eq!(
            at(
                "long.gguf",
                vec![kv_u32_for_tests("gemma4.context_length", 1 << 20)]
            ),
            Some(MAX_CELLS),
            "never past the model maximum"
        );
        assert_eq!(
            at(
                "zero.gguf",
                vec![kv_u32_for_tests("gemma4.context_length", 0)]
            ),
            None
        );
        assert_eq!(at("none.gguf", vec![]), None);
        assert_eq!(header_context(&dir.path().join("missing.gguf")), None);
    }

    use super::*;

    fn planned_cells(
        free: u64,
        weights: u64,
        splits: Option<usize>,
        wide: bool,
        sessions: usize,
    ) -> Result<(usize, usize), EngError> {
        Ok(cell_figures(splits, wide, 8)?.planned(
            free,
            weights,
            sessions,
            (MAX_CELLS, FALLBACK_CELLS),
        ))
    }

    fn pool_cells(free: u64, weights: u64, split: bool, wide: bool, sessions: usize) -> usize {
        planned_cells(
            free,
            weights,
            split.then_some(ATTN_SPLIT_MAX),
            wide,
            sessions,
        )
        .expect("plan")
        .1
    }

    #[test]
    fn the_split_default_is_armed_and_zero_disarms() {
        // Unset arms the qualified split; `0`/`off` is the rollback to the
        // single-pass summation order; any other spelling refuses the load.
        assert!(split_armed_from(None).expect("unset arms"));
        assert!(split_armed_from(Some("1")).expect("1 arms"));
        assert!(!split_armed_from(Some("0")).expect("zero disarms"));
        assert!(!split_armed_from(Some("off")).expect("off disarms"));
        assert!(split_armed_from(Some("yes")).is_err());
    }

    #[test]
    fn a_group_runs_on_the_tensor_cores_only_in_one_type_with_a_dense_case() {
        use QuantType::{Iq4Nl, Q4K, Q6K, Q80};
        let wide = MMQ_MIN_ROWS;
        assert!(on_tensor_cores([Q80, Q80, Q80].into_iter(), wide));
        assert!(on_tensor_cores([Q4K, Q4K].into_iter(), wide));
        assert!(on_tensor_cores([Q6K].into_iter(), 64));
        // A mixed group keeps each weight on the matvec: one activation layout serves one type.
        assert!(!on_tensor_cores([Q4K, Q6K, Q6K].into_iter(), wide));
        assert!(!on_tensor_cores([Q80, Q4K].into_iter(), wide));
        // A short chunk, a type without a dense case and an empty group never do.
        assert!(!on_tensor_cores([Q80, Q80].into_iter(), wide - 1));
        assert!(!on_tensor_cores([Iq4Nl].into_iter(), wide));
        assert!(!on_tensor_cores(std::iter::empty(), wide));
    }

    #[test]
    fn scratch_bytes_charges_the_prefill_maximum() {
        // Growable f32 vectors at 8 rows (89,856 elements x 8 x 4),
        // fixed f32 ones + two logits rows (524,800 x 4), growable q8 at eight
        // columns plus the head and qk buffers, plus the 12 input-cell bytes;
        // the armed split adds its partial buffer on top.
        let disarmed = scratch_bytes(None, false, 8).expect("the disarmed scratch computes");
        let armed =
            scratch_bytes(Some(ATTN_SPLIT_MAX), false, 8).expect("the armed scratch computes");
        let gemm = DenseChunkGemm::bytes(MMQ_K_MAX, 8).expect("the gemm charge computes");
        assert_eq!(disarmed, 5_172_748 + gemm);
        assert_eq!(armed, 9_383_436 + gemm);
        assert_eq!(armed - disarmed, attn_part_bytes(ATTN_SPLIT_MAX, 8) as u64);
        // The wide prefill attention's partial buffer is charged only where the gate passed.
        let wide = wide_geometry(MAX_CELLS)
            .part_bytes()
            .expect("the wide partial size computes") as u64;
        assert_eq!(wide, 16_842_752);
        assert_eq!(scratch_bytes(None, true, 8), Some(disarmed + wide));
        assert_eq!(
            scratch_bytes(Some(ATTN_SPLIT_MAX), true, 8),
            Some(armed + wide)
        );
    }

    #[test]
    fn the_bolt_demand_is_refused_without_the_headroom() {
        // A device holding exactly the allocations (weights + KV + scratch) with
        // no room for the headroom reserve must be refused by the shared admit;
        // the same demand fits once the reserve is included.
        let weights_bytes = 3_000_000_000u64;
        let demand = bolt_demand(weights_bytes, Some(ATTN_SPLIT_MAX), 32_768, true, 8)
            .expect("the Bolt demand computes");
        assert_eq!(demand.tensors, weights_bytes);
        assert!(demand.kv_cache > 0 && demand.work > 0 && demand.headroom > 0);
        let allocations = demand.tensors + demand.kv_cache + demand.work;
        let error = crate::kernels::cuda::plan::admit(&demand, allocations)
            .expect_err("an exact fit of the allocations alone must be refused");
        let EngError::MemoryPlan(message) = error else {
            panic!("expected MemoryPlan, got {error:?}");
        };
        assert!(message.contains("headroom"), "{message}");
        assert!(
            message.ends_with(&format!("device has {allocations} free")),
            "{message}"
        );
        assert_eq!(
            crate::kernels::cuda::plan::admit(&demand, allocations + demand.headroom),
            Ok(allocations + demand.headroom)
        );
    }

    #[test]
    fn the_pool_is_the_largest_count_the_card_carries() {
        let weights = 7_578_800_000u64;
        let free_for = |cells: usize| {
            let demand =
                bolt_demand(weights, Some(ATTN_SPLIT_MAX), cells, false, 8).expect("demand");
            demand.tensors + demand.kv_cache + demand.work + demand.headroom
        };
        // A card that carries 40,960 cells and not one step more has exactly those, and one
        // session may hold all of them.
        let free = free_for(40_960);
        assert_eq!(
            planned_cells(free, weights, Some(ATTN_SPLIT_MAX), false, 1).expect("plan"),
            (40_960, 40_960)
        );
        assert_eq!(pool_cells(free - 1, weights, true, false, 1), 39_936);
        // The pool passes the shared admission, and a card short of one cell plans the fallback
        // that the admission refuses with its own figures.
        let demand = bolt_demand(weights, Some(ATTN_SPLIT_MAX), 40_960, false, 8).expect("demand");
        assert!(crate::kernels::cuda::plan::admit(&demand, free).is_ok());
        let short = free_for(0) - 1;
        assert_eq!(pool_cells(short, weights, true, false, 1), FALLBACK_CELLS);
        let refused =
            bolt_demand(weights, Some(ATTN_SPLIT_MAX), FALLBACK_CELLS, false, 8).expect("demand");
        assert!(crate::kernels::cuda::plan::admit(&refused, short).is_err());
        assert_eq!(MAX_CELLS, 65_536);
    }

    /// One session holds up to the model's context whatever the planned session count; the pool
    /// is what the card carries for all of them together. On a 3090 the pool funds the full
    /// context for a lone prompt and an even share of it for each of eight sessions.
    #[test]
    fn a_session_holds_the_model_context_and_the_pool_is_what_the_card_carries() {
        let weights = 7_946_927_552u64;
        let free = 23_448_322_048u64;
        let (cells, pool) = planned_cells(
            free,
            weights,
            Some(ATTN_SPLIT_MAX),
            false,
            Bolt::WINDOW_ROWS,
        )
        .expect("plan");
        assert_eq!(cells, MAX_CELLS);
        // The sliding layers' rings are charged once, not per session, so the pool carries the
        // full-attention planes of many contexts. It must hold a lone session's whole context,
        // and the planned pool itself must pass the shared admission against the same card.
        assert!(pool >= MAX_CELLS, "{pool}");
        let demand = bolt_demand(
            weights,
            Some(ATTN_SPLIT_MAX),
            pool,
            false,
            Bolt::WINDOW_ROWS,
        )
        .expect("demand");
        assert!(
            crate::kernels::cuda::plan::admit(&demand, free).is_ok(),
            "{pool}"
        );
        eprintln!(
            "3090 default start: pool {pool} cells, one session up to {cells}, an even share of \
             {} for each of {} sessions",
            pool / Bolt::WINDOW_ROWS,
            Bolt::WINDOW_ROWS
        );
        // A smaller card still holds the pool of what it carries; one session never holds more
        // than the pool.
        let small =
            planned_cells(12_000_000_000, weights, Some(ATTN_SPLIT_MAX), false, 2).expect("plan");
        assert!(small.0 <= small.1 && small.1 < pool);
    }

    #[test]
    fn the_planned_cells_fund_the_wide_prefill_buffer_when_the_gate_passes() {
        let weights = 7_600_000_000u64;
        for free in [11_000_000_000u64, 14_000_000_000, 21_000_000_000] {
            let plain = planned_cells(free, weights, Some(ATTN_SPLIT_MAX), false, 1)
                .expect("cells")
                .0;
            let wide = planned_cells(free, weights, Some(ATTN_SPLIT_MAX), true, 1)
                .expect("cells")
                .0;
            assert!(wide <= plain, "{free}: {wide} against {plain}");
            let demand = bolt_demand(weights, Some(ATTN_SPLIT_MAX), wide, true, 8).expect("demand");
            crate::kernels::cuda::plan::admit(&demand, free)
                .unwrap_or_else(|error| panic!("{free}: {wide} cells: {error:?}"));
        }
    }

    fn logits(model: &Bolt) -> Vec<f32> {
        let mut bytes = vec![0u8; VOCAB * 4];
        model
            .ctx
            .download(&model.logits_capped.buf, &mut bytes)
            .expect("logits");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn assert_logits(got: &[f32], want: &[f32]) {
        let mut max_relative = 0.0f32;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite() && w.is_finite(), "non-finite logit {i}");
            let error = (g - w).abs();
            assert!(error <= 2e-5 * w.abs(), "logit {i}: {g} versus {w}");
            if w != 0.0 {
                max_relative = max_relative.max(error / w.abs());
            }
        }
        eprintln!("logits max relative error {max_relative}");
    }

    /// Every launch shape of the Bolt step, replayed with tiny buffers: the wrappers
    /// and native entries must admit the whole geometry before any model runs. This
    /// test exists because a validation reject surfaced only on the device path.
    #[test]
    #[ignore = "requires a CUDA device; use --release --test-threads=1"]
    fn bolt_step_shapes_are_admitted() {
        use crate::kernels::cuda::CudaContext;
        use crate::kernels::cuda::resident::attn_decode_gemma::AttnDecodeGemmaCall;
        use crate::kernels::cuda::resident::kv8_append::{Kv8Data, Kv8Shape};
        let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())
            .expect("context");
        let stream = ctx.stream().expect("stream");
        let zero = |n: usize| vec![0u8; n];
        let f32s = |n: usize| ctx.upload(&zero(n * 4)).expect("f32 upload");
        let u16s = |n: usize| ctx.upload(&zero(n * 2)).expect("u16 upload");
        let i8s = |n: usize| ctx.upload(&zero(n)).expect("i8 upload");
        let mut word_bytes = Vec::with_capacity(8);
        word_bytes.extend_from_slice(&0i32.to_le_bytes());
        word_bytes.extend_from_slice(&1i32.to_le_bytes());
        let words = ctx.upload(&word_bytes).expect("words");
        let status = ctx.upload(&zero(4)).expect("status");
        let q8_matrix = |rows: usize, k: usize| {
            super::matrix_from_bytes(
                &ctx,
                "zeros",
                crate::weights::GgufType::Q80,
                k,
                rows,
                &zero(rows * (k / 32) * 34),
            )
            .expect("matrix")
        };
        // Embedding gather and the hidden-path ops at full Bolt widths.
        ctx.embed_native(
            &stream,
            EmbedNativeCall {
                weight: &q8_matrix(1024, 2560),
                ids: words.range(0, 4).unwrap(),
                x: f32s(2560).range(0, 2560 * 4).unwrap(),
                status: status.range(0, 4).unwrap(),
                m: 1,
            },
        )
        .expect("embed 2560");
        for (m, n) in [(1, 2560), (1, 256 * 42), (1, 256), (1, 2560)] {
            ctx.row_scale(
                &stream,
                RowScaleCall {
                    out: f32s(n).range(0, n * 4).unwrap(),
                    x: f32s(n).range(0, n * 4).unwrap(),
                    scale: 0.5,
                    m,
                    n,
                },
            )
            .expect("row_scale {n}");
            ctx.row_add(
                &stream,
                RowAddCall {
                    out: f32s(n).range(0, n * 4).unwrap(),
                    x: f32s(n).range(0, n * 4).unwrap(),
                    y: f32s(n).range(0, n * 4).unwrap(),
                    n,
                    flag: None,
                    fused: false,
                },
            )
            .expect("row_add {n}");
        }
        // The quantize/matvec pairs at every k the step uses.
        for k in [2560, 2048, 1024, 256] {
            ctx.quantize_q8_1(
                &stream,
                QuantizeCall {
                    x: f32s(k).range(0, k * 4).unwrap(),
                    y: i8s(q8_1_bytes(k, 1).unwrap())
                        .range(0, q8_1_bytes(k, 1).unwrap())
                        .unwrap(),
                    k,
                    ncols: 1,
                },
            )
            .expect("quantize {k}");
        }
        // The per-head norm at 256 and 512, including the interior-slice pattern.
        let ple = f32s(256 * 42);
        for n in [256usize, 512] {
            ctx.add_rmsnorm_q8(
                &stream,
                AddNormQ8Call {
                    n,
                    eps: 1e-6,
                    h: f32s(n).range(0, n * 4).unwrap(),
                    delta: None,
                    weight: f32s(n).range(0, n * 4).unwrap(),
                    norm_out: f32s(n).range(0, n * 4).unwrap(),
                    q8: i8s(q8_1_bytes(n, 1).unwrap())
                        .range(0, q8_1_bytes(n, 1).unwrap())
                        .unwrap(),
                    rows: 1,
                },
            )
            .expect("head norm {n}");
        }
        ctx.add_rmsnorm_q8(
            &stream,
            AddNormQ8Call {
                n: 256,
                eps: 1e-6,
                h: ple.range(256 * 4, 256 * 4).unwrap(),
                delta: None,
                weight: f32s(256).range(0, 256 * 4).unwrap(),
                norm_out: f32s(256).range(0, 256 * 4).unwrap(),
                q8: i8s(q8_1_bytes(256, 1).unwrap())
                    .range(0, q8_1_bytes(256, 1).unwrap())
                    .unwrap(),
                rows: 1,
            },
        )
        .expect("ple slice norm");
        // The PLE table gather and the gated mixing at 256.
        ctx.embed_native(
            &stream,
            EmbedNativeCall {
                weight: &q8_matrix(1024, 256),
                ids: words.range(0, 4).unwrap(),
                x: f32s(256).range(0, 256 * 4).unwrap(),
                status: status.range(0, 4).unwrap(),
                m: 1,
            },
        )
        .expect("ple table gather");
        ctx.gelu_ffn(
            &stream,
            GeluFfnCall {
                out: f32s(256).range(0, 256 * 4).unwrap(),
                gate: f32s(256).range(0, 256 * 4).unwrap(),
                up: f32s(256).range(0, 256 * 4).unwrap(),
                m: 1,
                n: 256,
            },
        )
        .expect("gelu 256");
        // The attention path at both head widths, 8 q heads, 2 kv heads.
        for head in [256usize, 512] {
            let shape = Kv8Shape::linear(
                1,
                2,
                head,
                32768,
                crate::kernels::cuda::resident::kv8_append::Kv8CodeScale::UnroundedF32,
            );
            let codes = 2 * 32768 * head;
            let scales = 2 * 32768 * (head / 32);
            ctx.kv8_chunk(
                &stream,
                shape,
                Kv8Data {
                    k: f32s(2 * head).range(0, 2 * head * 4).unwrap(),
                    v: f32s(2 * head).range(0, 2 * head * 4).unwrap(),
                    k_codes: i8s(codes).range(0, codes).unwrap(),
                    k_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    v_codes: i8s(codes).range(0, codes).unwrap(),
                    v_scales: u16s(scales).range(0, scales * 2).unwrap(),
                },
                0,
            )
            .expect("kv8 chunk {head}");
            ctx.rope_gemma(
                &stream,
                RopeGemmaCall {
                    q: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    k: f32s(2 * head).range(0, 2 * head * 4).unwrap(),
                    pos: words.range(0, 4).unwrap(),
                    factors: None,
                    rows_q: 1,
                    rows_k: 1,
                    n_q_head: 8,
                    n_kv_head: 2,
                    head_dim: head,
                    global_mode: head == 512,
                },
            )
            .expect("rope {head}");
            ctx.attn_decode_gemma(
                &stream,
                AttnDecodeGemmaCall {
                    out: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    status: None,
                    q: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    k_codes: i8s(codes).range(0, codes).unwrap(),
                    k_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    v_codes: i8s(codes).range(0, codes).unwrap(),
                    v_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    kv_len: words.range(4, 4).unwrap(),
                    m: 1,
                    n_q_head: 8,
                    n_kv_head: 2,
                    head_dim: head,
                    cells: 32768,
                    window: if head == 512 { 0 } else { 512 },
                    scale: 1.0,
                    split: None,
                    ring: 0,
                },
            )
            .expect("attn {head}");
            // The split path of the same geometry: the wrapper must admit the
            // part+combine plan at both head widths when the switch arms.
            let part_floats = 8 * super::ATTN_SPLIT_MAX * (2 + head);
            ctx.attn_decode_gemma(
                &stream,
                AttnDecodeGemmaCall {
                    out: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    status: None,
                    q: f32s(8 * head).range(0, 8 * head * 4).unwrap(),
                    k_codes: i8s(codes).range(0, codes).unwrap(),
                    k_scales: u16s(scales).range(0, scales * 2).unwrap(),
                    v_codes: i8s(codes).range(0, codes).unwrap(),
                    v_scales: u16s(scales * 2).range(0, scales * 2).unwrap(),
                    kv_len: words.range(4, 4).unwrap(),
                    m: 1,
                    n_q_head: 8,
                    n_kv_head: 2,
                    head_dim: head,
                    cells: 32768,
                    window: if head == 512 { 0 } else { 512 },
                    scale: 1.0,
                    split: Some(AttnDecodeGemmaSplit {
                        part: ctx
                            .upload(&zero(part_floats * 4))
                            .expect("part upload")
                            .range(0, part_floats * 4)
                            .unwrap(),
                        max_splits: super::ATTN_SPLIT_MAX,
                        min_keys: super::ATTN_SPLIT_MIN_KEYS,
                    }),
                    ring: 0,
                },
            )
            .expect("attn split {head}");
        }
        stream.synchronize().expect("sync");
    }

    /// Single-step parity vehicle: with BOLT_TRACE_DIR set, writes one f32
    /// checkpoint per layer (and the scaled embedding) for the first two
    /// positions, unmodified by any batched replay. The torch reference writes
    /// the same names; the first diverging layer names the wrong op.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn trace_two_single_steps() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let mut model = super::load(std::path::Path::new(&path)).expect("load");
        let seq = model.new_seq().expect("sequence");
        for (pos, token) in [2u32, 19].into_iter().enumerate() {
            if pos == 0 {
                model
                    .decode(&seq, token, pos)
                    .expect("first token with logits");
            } else {
                model
                    .decode_hidden(&seq, token, pos)
                    .expect("second token hidden");
            }
        }
    }

    /// Greedy-quality gate: the full reference prompt, logits traced at every step.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn trace_full_prompt_logits() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let mut model = super::load(std::path::Path::new(&path)).expect("load");
        let seq = model.new_seq().expect("sequence");
        for pos in 0..32u32 {
            let token = 2 + pos * 17;
            model.decode(&seq, token, pos as usize).expect("step");
        }
    }

    /// The default start's pool serves every session the planner then funds, each at its even
    /// share and up to its last cell, and one session alone takes the model's context.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn the_default_pool_serves_every_planned_session_to_the_last_cell() {
        use crate::family::cell_pool::CELL_GRANULE;
        use crate::family::seq_plan::{SeqInputs, plan_seqs};
        let path = std::path::PathBuf::from(std::env::var("BOLT_Q8_FILE").expect("model path"));
        let bytes = std::fs::metadata(&path).expect("model file").len();
        let mut model = load_choosing(
            &[crate::family::gguf_select::GgufFile { path, bytes }],
            crate::family::gemma::gemma4::config::MaxSeqs::Auto,
            &|_| {},
        )
        .expect("load");
        let pool = model.pool().total();
        let initial = CELL_GRANULE;
        let first = model.new_seq_cells(initial).expect("first session");
        let free = u64::try_from(model.ctx.memory_info().expect("memory").free).unwrap_or(u64::MAX);
        let plan = plan_seqs(
            SeqInputs {
                free,
                headroom: crate::family::gemma::gemma4::budget::BOLT_HEADROOM_BYTES,
                cost: crate::family::gemma::gemma4::budget::bolt_seq_cost(model.rows)
                    .expect("cost"),
                context: initial as u64,
                row_cap: Bolt::WINDOW_ROWS,
                ceiling: Bolt::WINDOW_ROWS,
            },
            None,
        )
        .expect("session plan");
        let share = pool / plan.seqs / CELL_GRANULE * CELL_GRANULE;
        eprintln!(
            "pool {pool} cells, one session up to {}, planned sessions {} ({}), even share {share}",
            model.cells(),
            plan.seqs,
            plan.describe()
        );
        assert_eq!(plan.seqs, Bolt::WINDOW_ROWS);
        let mut seqs = vec![first];
        for _ in 1..plan.seqs {
            seqs.push(model.new_seq_cells(initial).expect("a planned session"));
        }
        for seq in &mut seqs {
            model.resize_seq(seq, share).expect("an even share");
        }
        assert!(
            model.pool().free() < share,
            "the shares leave less than one share free"
        );
        model.reserve_window(plan.seqs).expect("the window rows");
        let refs: Vec<&BoltSeq> = seqs.iter().collect();
        let tokens = vec![2u32; plan.seqs];
        let positions = vec![share - 1; plan.seqs];
        let rows = model
            .decode_rows(&refs, &tokens, &positions)
            .expect("every session decodes at the last cell");
        assert_eq!(rows.len(), plan.seqs);
        assert!(rows.iter().all(|row| row.len() == VOCAB));
        // The pool is spent: another session queues.
        assert!(matches!(
            model.new_seq_cells(share),
            Err(EngError::MemoryPlan(_))
        ));
        // One session alone takes the model's context: the others return their cells.
        drop(refs);
        seqs.truncate(1);
        let mut lone = seqs.pop().expect("one session");
        model
            .resize_seq(&mut lone, model.cells())
            .expect("the whole context");
        assert_eq!(lone.cells(), model.cells());
        model
            .prefill_batch(&lone, &[2], model.cells() - 1, false)
            .expect("the last cell of the whole context");
    }

    /// A sequence grown in the middle of a decode gives, at every step, the logits of a sequence
    /// sized for the whole run from the start, bit for bit: the growth crosses the sliding window
    /// of the local layers and re-lays every plane. A wrong copy would serve wrong text.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn a_sequence_grown_mid_decode_gives_the_logits_of_one_sized_for_the_run() {
        use crate::family::cell_pool::CELL_GRANULE;
        let path = std::path::PathBuf::from(std::env::var("BOLT_Q8_FILE").expect("model path"));
        let bytes = std::fs::metadata(&path).expect("model file").len();
        let mut model = load_choosing(
            &[crate::family::gguf_select::GgufFile { path, bytes }],
            crate::family::gemma::gemma4::config::MaxSeqs::Auto,
            &|_| {},
        )
        .expect("load");
        // A run under a sanitizer is slow: BOLT_GROW_STEPS shortens it (it must pass one granule).
        let steps = std::env::var("BOLT_GROW_STEPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1600usize);
        let mut grown = model.new_seq_cells(CELL_GRANULE).expect("small sequence");
        let full = model
            .new_seq_cells(steps.next_multiple_of(CELL_GRANULE))
            .expect("whole run");
        let mut growths = 0;
        for pos in 0..steps {
            let token = 2 + (pos as u32 * 17) % 5000;
            let before = grown.cells();
            model.make_room(&mut grown, pos, pos + 1).expect("room");
            growths += usize::from(grown.cells() != before);
            model.decode(&grown, token, pos).expect("grown step");
            let a = model.capped_logits().expect("grown logits");
            model.decode(&full, token, pos).expect("full step");
            let b = model.capped_logits().expect("full logits");
            let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(
                same,
                "the logits differ at position {pos} (grown holds {} cells)",
                grown.cells()
            );
        }
        assert!(growths >= 1, "the sequence grew during the run");
        assert!(grown.cells() >= steps);
    }

    fn load_q8() -> Bolt {
        let path = std::path::PathBuf::from(std::env::var("BOLT_Q8_FILE").expect("model path"));
        let bytes = std::fs::metadata(&path).expect("model file").len();
        load_choosing(
            &[crate::family::gguf_select::GgufFile { path, bytes }],
            crate::family::gemma::gemma4::config::MaxSeqs::Auto,
            &|_| {},
        )
        .expect("load")
    }

    /// The bits of the capped logits of the last step.
    fn logit_bits(model: &Bolt) -> Vec<u32> {
        model
            .capped_logits()
            .expect("logits")
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// A session whose sliding layers keep a ring gives, at every chunk and every decode step, the
    /// logits of a session that keeps every position in linear planes, bit for bit: the prompt
    /// crosses the 512-position window, the chunk widths run from one to the planned rows, and the
    /// positions straddle the ring's end at every offset.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn a_ring_session_gives_the_logits_of_the_linear_control_bit_for_bit() {
        use crate::family::cell_pool::CELL_GRANULE;
        let mut model = load_q8();
        assert_ne!(
            model.local_ring,
            KvRing::LINEAR,
            "the ring is what a start builds"
        );
        let cells = 2 * CELL_GRANULE;
        let ring = model.new_seq_cells(cells).expect("ring session");
        model.set_linear_local_layers(true);
        let linear = model.new_seq_cells(cells).expect("linear session");
        model.set_linear_local_layers(false);
        // The ring planes are the fixed small size; the linear control keeps every cell.
        let ring_cells = |seq: &BoltSeq| {
            seq.kv
                .iter()
                .flatten()
                .map(|kv| kv.shape.cells)
                .collect::<Vec<_>>()
        };
        assert!(
            ring_cells(&ring).iter().any(|&c| c < cells),
            "{:?}",
            ring_cells(&ring)
        );
        assert!(ring_cells(&linear).iter().all(|&c| c == cells));
        let widths = [
            1usize,
            4,
            8.min(model.rows),
            3,
            6,
            5.min(model.rows),
            2,
            7.min(model.rows),
            1,
            model.rows,
        ];
        let mut pos = 0usize;
        let mut turn = 0usize;
        while pos < 1_700 {
            let width = widths[turn % widths.len()].min(1_700 - pos);
            let tokens: Vec<u32> = (0..width)
                .map(|i| 2 + ((pos + i) as u32 * 37) % 9_000)
                .collect();
            model
                .prefill_batch(&ring, &tokens, pos, true)
                .expect("ring chunk");
            let a = logit_bits(&model);
            model
                .prefill_batch(&linear, &tokens, pos, true)
                .expect("linear chunk");
            let b = logit_bits(&model);
            assert_eq!(
                a,
                b,
                "the logits differ after the chunk ending at {}",
                pos + width
            );
            pos += width;
            turn += 1;
        }
        for step in 0..40usize {
            let token = 3 + (step as u32 * 11) % 5_000;
            model.decode(&ring, token, pos + step).expect("ring step");
            let a = logit_bits(&model);
            model
                .decode(&linear, token, pos + step)
                .expect("linear step");
            assert_eq!(
                a,
                logit_bits(&model),
                "the logits differ at decode step {step}"
            );
        }
    }

    /// A snapshot keeps the sliding layers' last window and a restore writes it back where the
    /// ring wants it: for every fed count around the window and the ring's end, a fresh session
    /// restored from the snapshot continues with the logits of the session it was taken from.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn a_snapshot_of_a_ring_session_restores_into_another_session_bit_for_bit() {
        use crate::family::cell_pool::CELL_GRANULE;
        let mut model = load_q8();
        let cells = 2 * CELL_GRANULE;
        let source = model.new_seq_cells(cells).expect("source");
        let target = model.new_seq_cells(cells).expect("target");
        let total = 1_600usize;
        let token_at = |pos: usize| 2 + (pos as u32 * 29) % 9_000;
        let tokens: Vec<u32> = (0..total).map(token_at).collect();
        // Feed the source one token at a time, saving at the counts that straddle the window.
        let mut snapshots = Vec::new();
        for pos in 0..total {
            model
                .prefill_batch(&source, &tokens[pos..=pos], pos, false)
                .expect("feed");
            let fed = pos + 1;
            if [100, 511, 512, 513, 519, 520, 1_030, 1_600].contains(&fed) {
                snapshots.push((fed, model.snapshot_seq(&source, fed).expect("snapshot")));
            }
        }
        for (fed, snapshot) in &snapshots {
            let layers = model.snapshot_layer_bytes(*fed);
            assert_eq!(
                layers.iter().sum::<usize>(),
                snapshot.bytes(),
                "size at {fed}"
            );
            // The budget's closed form is the same number: every full-attention position and one
            // window of the sliding layers.
            assert_eq!(
                crate::family::gemma::gemma4::budget::bolt_snapshot_bytes(*fed),
                Some(snapshot.bytes() as u64),
                "budget at {fed}"
            );
            // Past the window the entry stops growing with the prefix on the sliding layers.
            let probe = model.new_seq_cells(cells).expect("probe");
            for pos in 0..*fed {
                model
                    .prefill_batch(&probe, &tokens[pos..=pos], pos, false)
                    .expect("refeed");
            }
            // A fresh session and a grown one take the same entry and continue like the probe: the
            // grown session moved its rings and re-laid its full-attention planes first.
            model.restore_seq(&target, snapshot).expect("restore");
            let mut grown = model.new_seq_cells(CELL_GRANULE).expect("small session");
            model
                .make_room(&mut grown, 0, *fed + 8)
                .expect("room for the entry");
            model
                .restore_seq(&grown, snapshot)
                .expect("restore into a grown session");
            for step in 0..6usize {
                let token = 5 + step as u32;
                model.decode(&probe, token, fed + step).expect("probe step");
                let a = logit_bits(&model);
                model
                    .decode(&target, token, fed + step)
                    .expect("restored step");
                assert_eq!(a, logit_bits(&model), "fed {fed} step {step}");
                model.decode(&grown, token, fed + step).expect("grown step");
                assert_eq!(a, logit_bits(&model), "grown, fed {fed} step {step}");
            }
        }
        // The window caps the sliding layers: a long entry is not larger than a window-long one by
        // more than the full-attention layers' cells.
        let (short, long) = (
            model.snapshot_layer_bytes(600),
            model.snapshot_layer_bytes(1_600),
        );
        let sliding = |layers: &[usize]| layers[0];
        assert_eq!(
            sliding(&short),
            sliding(&long),
            "a sliding layer stores its window only"
        );
        assert!(
            long[5] > short[5],
            "a full-attention layer stores every position"
        );
    }

    /// The ring a start builds is exactly as wide as the widest chunk a step takes, and its
    /// admission rule refuses one row more (the append itself is refused by `rope_append_attend`
    /// and by the kernel launch, tested there).
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn the_ring_margin_is_the_widest_chunk_a_step_takes() {
        let model = load_q8();
        let seq = model.new_seq().expect("session");
        let margin = model.local_ring.widest_append();
        assert_eq!(
            margin, model.rows,
            "the ring is sized for the widest chunk a step takes"
        );
        assert!(!model.local_ring.admits(margin + 1));
        assert!(model.local_ring.admits(margin));
        drop(seq);
    }

    /// One model owns the device. Replaying from position zero overwrites every live cache cell.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn batched_prefill_matches_single_steps() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let mut model = super::load(std::path::Path::new(&path)).expect("load");
        // This test pins the row attention's batched chunks to its single steps within 2e-5; the
        // wide attention rounds to fp16 and has its own comparison below.
        model.use_wide_prefill(false);
        let seq = model.new_seq().expect("sequence");
        assert!(matches!(
            model.prefill_batch(&seq, &[], 0, false),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            model.prefill_batch(&seq, &vec![2; model.rows + 1], 0, false),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            model.prefill_batch(&seq, &[2], model.cells(), false),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.prefill_batch(&seq, &[2], usize::MAX, false),
            Err(EngError::ContextLimit(_))
        ));
        // Decode bounds its position like prefill: the last valid cell is one below the cap.
        assert!(matches!(
            model.decode(&seq, 2, model.cells()),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.decode_hidden(&seq, 2, model.cells()),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            model.decode(&seq, 2, usize::MAX),
            Err(EngError::ContextLimit(_))
        ));
        let prompt: Vec<u32> = (0..150).map(|i| 2 + i * 17).collect();
        let single_started = std::time::Instant::now();
        for (pos, &token) in prompt.iter().enumerate() {
            if pos + 1 == prompt.len() {
                model.decode(&seq, token, pos).expect("single last token");
            } else {
                model
                    .decode_hidden(&seq, token, pos)
                    .expect("single hidden token");
            }
        }
        let single_elapsed = single_started.elapsed();
        let expected = logits(&model);
        // Parity hook: write this run's single-step logits for the reference
        // comparison (scratch tooling reads it; no assertion depends on it).
        if let Ok(path) = std::env::var("BOLT_DUMP_LOGITS") {
            let bytes: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).expect("dump logits");
            eprintln!("logits dumped to {path}");
        }
        model
            .decode(&seq, 2, prompt.len())
            .expect("single continuation");
        let continued = logits(&model);
        eprintln!("timing {} single steps {single_elapsed:?}", prompt.len());
        // The planned rows are the shipped cap; 33 and 3 also exercise a partial final batch.
        for width in [1, Bolt::WINDOW_ROWS, 3, 33, model.rows] {
            let started = std::time::Instant::now();
            let mut pos = 0;
            for batch in prompt.chunks(width) {
                model
                    .prefill_batch(&seq, batch, pos, pos + batch.len() == prompt.len())
                    .expect("batch");
                pos += batch.len();
            }
            let elapsed = started.elapsed();
            // A chunk of 16 rows or more runs on the tensor cores, whose f32 sums add in another
            // order: its logits stay within the distribution tolerance instead of the matvec's.
            let check = |got: &[f32], want: &[f32]| {
                if width < MMQ_MIN_ROWS {
                    return assert_logits(got, want);
                }
                let (kl, max_abs, same_top) = compare_logits(want, got);
                eprintln!("tensor cores m={width}: kl {kl:.3e} max|dlogit| {max_abs:.3e}");
                assert!(kl < 0.1, "m={width}: a different distribution");
                let _ = (same_top, max_abs);
            };
            check(&logits(&model), &expected);
            model
                .decode(&seq, 2, prompt.len())
                .expect("batch continuation");
            check(&logits(&model), &continued);
            eprintln!(
                "timing {} batched tokens m={width} {elapsed:?}; speedup {:.3}",
                prompt.len(),
                single_elapsed.as_secs_f64() / elapsed.as_secs_f64()
            );
        }
    }

    /// How far two next-token distributions are apart: the KL divergence of the second from
    /// the first in nats, the largest absolute logit difference, and whether the arg-max agrees.
    fn compare_logits(old: &[f32], new: &[f32]) -> (f64, f32, bool) {
        let softmax = |v: &[f32]| -> Vec<f64> {
            let top = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f64> = v.iter().map(|&x| f64::from(x - top).exp()).collect();
            let sum: f64 = e.iter().sum();
            e.iter().map(|x| x / sum).collect()
        };
        let (p, q) = (softmax(old), softmax(new));
        let kl = p
            .iter()
            .zip(&q)
            .filter(|(p, _)| **p > 0.0)
            .map(|(p, q)| p * (p / q.max(1e-300)).ln())
            .sum();
        let max_abs = old
            .iter()
            .zip(new)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
        };
        (kl, max_abs, argmax(old) == argmax(new))
    }

    /// The wide prefill attention against the row attention, final-token logits of the same
    /// prompts on one load: lengths cross the sliding window (512) and a partial last chunk.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn wide_prefill_matches_the_row_attention_logits() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let lens: [usize; 3] = [40, 600, 1100];
        let widths: [usize; 2] = [5, 3];
        let mut model = super::load(std::path::Path::new(&path)).expect("load");
        assert!(
            model.use_wide_prefill(true),
            "this device has the wide path"
        );
        let seq = model.new_seq().expect("sequence");
        for len in lens {
            let prompt: Vec<u32> = (0..len as u32)
                .map(|i| 2 + i.wrapping_mul(7919) % 200_000)
                .collect();
            let run_width = |model: &mut Bolt,
                             wide: bool,
                             width: usize|
             -> (Vec<f32>, u64, std::time::Duration) {
                model.use_wide_prefill(wide);
                let before = model.wide_prefill_launches();
                let started = std::time::Instant::now();
                let mut pos = 0;
                for batch in prompt.chunks(width) {
                    model
                        .prefill_batch(&seq, batch, pos, pos + batch.len() == len)
                        .expect("batch");
                    pos += batch.len();
                }
                let elapsed = started.elapsed();
                (
                    logits(model),
                    model.wide_prefill_launches() - before,
                    elapsed,
                )
            };
            let run = |model: &mut Bolt, wide: bool| run_width(model, wide, Bolt::WINDOW_ROWS);
            let (old, old_launches, old_time) = run(&mut model, false);
            let (new, new_launches, new_time) = run(&mut model, true);
            // Pieces of any width whose rows all sit in chunks of two or more leave the logits of
            // pieces of 8 bit for bit: the wide attention of a row does not depend on the cut.
            for &width in &widths {
                // A lone row at the end of a prompt takes the decode attention, so a width that
                // leaves one differs from pieces of 8 by the attention paths' own difference.
                if len % width == 1 || len % Bolt::WINDOW_ROWS == 1 {
                    continue;
                }
                let (other, _, _) = run_width(&mut model, true, width);
                assert!(
                    other
                        .iter()
                        .zip(&new)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "len {len}: pieces of {width} differ from pieces of 8"
                );
            }
            assert_eq!(old_launches, 0, "the row attention ran");
            assert!(new_launches > 0, "the wide attention ran");
            // The scale of the comparison: the same row attention without its key split, which
            // only changes the summation order.
            model.use_attn_split(false).expect("split off");
            let (unsplit, _, _) = run(&mut model, false);
            model.use_attn_split(true).expect("split on");
            let (kl, max_abs, same_top) = compare_logits(&old, &new);
            let (kl_order, max_order, same_order) = compare_logits(&old, &unsplit);
            eprintln!(
                "WIDE len {len}: kl {kl:.3e} max|dlogit| {max_abs:.3e} top1 same {same_top} \
                 launches {new_launches} row {old_time:?} wide {new_time:?}; summation-order \
                 reference: kl {kl_order:.3e} max|dlogit| {max_order:.3e} top1 same {same_order}"
            );
        }
    }

    /// The widths, 1 to 8, in which a prompt of `len` tokens is cut for the scoring test: a fixed
    /// pseudo-random sequence, so lone rows and every alignment of a chunk end occur and both
    /// paths of one document see the same cut. With `lone_heavy`, half of the pieces are a lone
    /// row, which gives the lone-row positions a stratum of their own to judge.
    fn score_widths(len: usize, seed: u64, lone_heavy: bool) -> Vec<usize> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut widths = Vec::new();
        let mut left = len;
        while left > 0 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let pick = (state >> 33) as usize;
            let width = if lone_heavy && state >> 63 == 0 {
                1
            } else if lone_heavy {
                2 + pick % (Bolt::WINDOW_ROWS - 1)
            } else {
                1 + pick % Bolt::WINDOW_ROWS
            };
            widths.push(width.min(left));
            left -= width.min(left);
        }
        widths
    }

    /// True-token scores of the wide prefill attention against the row attention on real text.
    /// Each document is cut into pieces of 1 to 8 tokens (`score_widths`; `BOLT_SCORE_CUT=lone-heavy`
    /// makes half of them lone rows and scores every lone row); every third piece ends with
    /// logits, and its last position records the negative log-likelihood of the next true
    /// token and the arg-max under both paths. The text comes from `BOLT_SCORE_TOKENS`, one
    /// document per line as space-separated token ids; the records are written to
    /// `BOLT_SCORE_OUT` as tab-separated rows (document, position, true token, nll under the row
    /// attention, nll under the wide attention, arg-max of each, KL of the wide distribution from
    /// the row one, width of the piece).
    #[test]
    #[ignore = "requires the converted Bolt file, token documents and a device; use --release"]
    fn wide_prefill_scores_real_text_like_the_row_attention() {
        use std::fmt::Write as _;
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let docs = std::fs::read_to_string(std::env::var("BOLT_SCORE_TOKENS").expect("token file"))
            .expect("token documents");
        let out_path = std::env::var("BOLT_SCORE_OUT").expect("output file");
        let mut model = super::load(std::path::Path::new(&path)).expect("load");
        assert!(
            model.use_wide_prefill(true),
            "this device has the wide path"
        );
        let seq = model.new_seq().expect("sequence");
        let log_softmax = |v: &[f32]| -> Vec<f64> {
            let top = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = v.iter().map(|&x| f64::from(x - top).exp()).sum();
            let log_sum = sum.ln() + f64::from(top);
            v.iter().map(|&x| f64::from(x) - log_sum).collect()
        };
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i)
        };
        let lone_heavy = std::env::var("BOLT_SCORE_CUT").is_ok_and(|cut| cut == "lone-heavy");
        let tensor_mode = std::env::var("BOLT_SCORE_MODE").is_ok_and(|mode| mode == "tensor");
        let mut stash = None;
        let mut report = String::new();
        for (doc, line) in docs.lines().enumerate() {
            let tokens: Vec<u32> = line
                .split_whitespace()
                .map(|t| t.parse().expect("a token id"))
                .collect();
            let widths = if tensor_mode {
                // Chunks of the planned rows, which run on the tensor cores, and the remainder.
                let mut widths = vec![model.rows; tokens.len() / model.rows];
                widths.extend(Some(tokens.len() % model.rows).filter(|&rest| rest > 0));
                widths
            } else {
                score_widths(tokens.len(), doc as u64 + 1, lone_heavy)
            };
            // Scored positions: the last token of every third piece, whose next token is known.
            let mut rows: Vec<(usize, usize, [Vec<f32>; 2])> = Vec::new();
            for (pass, wide) in [false, true].into_iter().enumerate() {
                if tensor_mode {
                    // The two passes differ in the projections only: the matvec, then the tensor cores.
                    model.use_wide_prefill(true);
                    if pass == 0 {
                        stash = model.mmq.take();
                    } else {
                        model.mmq = stash.take();
                    }
                } else {
                    model.use_wide_prefill(wide);
                }
                let mut pos = 0;
                for (index, &width) in widths.iter().enumerate() {
                    let batch = &tokens[pos..pos + width];
                    let scored = (index % 3 == 2 || (lone_heavy && width == 1))
                        && pos + width < tokens.len();
                    model
                        .prefill_batch(&seq, batch, pos, scored)
                        .expect("batch");
                    pos += width;
                    if scored {
                        let row = logits(&model);
                        if pass == 0 {
                            rows.push((pos, width, [row, Vec::new()]));
                        } else {
                            let at = rows
                                .iter_mut()
                                .find(|(p, ..)| *p == pos)
                                .expect("the same positions");
                            at.2[1] = row;
                        }
                    }
                }
            }
            for (pos, width, [row, wide]) in rows {
                let truth = tokens[pos] as usize;
                let (lp_row, lp_wide) = (log_softmax(&row), log_softmax(&wide));
                let kl: f64 = lp_row
                    .iter()
                    .zip(&lp_wide)
                    .map(|(a, b)| a.exp() * (a - b))
                    .sum();
                writeln!(
                    report,
                    "{doc}\t{pos}\t{truth}\t{:.6}\t{:.6}\t{}\t{}\t{kl:.6e}\t{width}",
                    -lp_row[truth],
                    -lp_wide[truth],
                    argmax(&row),
                    argmax(&wide)
                )
                .expect("string write");
            }
        }
        std::fs::write(out_path, report).expect("scores written");
    }

    /// The skipped-shared prefill must leave the same final logits as a
    /// full-layer prefill: a hidden batch runs only the cache-writing physical
    /// prefix, while a logits batch runs every layer. The shared layers write
    /// no cache cells and feed only the final norm, so both paths agree.
    /// Each width reloads a fresh (zeroed) cache and runs the skip path FIRST,
    /// so a missing skip-path cache write at any width corrupts its logits
    /// instead of being masked by an earlier full write; the full path then
    /// overwrites the same cells from position 0.
    #[test]
    #[ignore = "requires the converted Bolt file and a device; use --release --test-threads=1"]
    fn skip_prefill_matches_full_layer_prefill() {
        let path = std::env::var("BOLT_Q8_FILE").expect("converted model path");
        let prompt: Vec<u32> = (0..150).map(|i| 2 + i * 17).collect();
        let n = prompt.len();
        // The planned rows (the shipped cap), 33 and 3 (partial final batches) and 1.
        for width in [
            crate::family::gemma::gemma4::budget::BOLT_PREFILL_LADDER[0],
            33,
            Bolt::WINDOW_ROWS,
            3,
            1,
        ] {
            // Fresh load per width: the cache starts zeroed, so a missing
            // skip-path write here is not masked by the last width's cells.
            let mut model = super::load(std::path::Path::new(&path)).expect("load");
            // Both prefills use the row attention here, so the 2e-5 bar compares the skip
            // structure alone.
            model.use_wide_prefill(false);
            let seq = model.new_seq().expect("sequence");
            // Skip-prefill: non-final batches hide logits and run only the
            // physical prefix; the final batch reads logits over the cache.
            let mut pos = 0;
            for batch in prompt.chunks(width) {
                model
                    .prefill_batch(&seq, batch, pos, pos + batch.len() == n)
                    .expect("skip batch");
                pos += batch.len();
            }
            let skip = logits(&model);
            model.decode(&seq, 2, n).expect("skip continuation");
            let skip_continued = logits(&model);
            // Full-layer prefill overwrites the same cells from position 0:
            // every batch reads logits, so every layer runs.
            let mut pos = 0;
            for batch in prompt.chunks(width) {
                model
                    .prefill_batch(&seq, batch, pos, true)
                    .expect("full batch");
                pos += batch.len();
            }
            assert_logits(&logits(&model), &skip);
            model.decode(&seq, 2, n).expect("full continuation");
            assert_logits(&logits(&model), &skip_continued);
        }
    }
}
