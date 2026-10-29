//! The resident state owner.
//!
//! [`ResidentState`] holds the three device buffers of [`StateLayout`] for one session: DeltaNet
//! recurrent and convolution state, the KV cache, and the persistent words (`StepParams`, the
//! result pair, the prefill status word). Addresses never change after allocation (captured graphs
//! bake them in), so there is no grow and no reallocation; `reset` and `import_state` write in
//! place. The type is thread confined because its buffers are. Slot 0 is the committed state. The
//! rows, the pending steps and the health live in the pure [`StateBook`]; the host export lives in
//! `export`.
//!
//! Every host copy goes through `DeviceBuffer::read_at` / `write_at` (`buffer_io.rs`), which take
//! the `DeviceStream` that owns the work on the buffer, queue the copy behind that work and wait
//! for that stream only. Every state call checks that the stream is the one the state was allocated
//! with (`DeviceStream::id`, `context/stream.rs`). Bytes produced on another stream (a forked
//! expert branch) must be joined into the owning stream with [`join_branches`] first.
//!
//! The device is little endian, so every value is read and written as little-endian bytes.

use std::{cell::RefCell, rc::Rc};

use super::{
    book::{Admitted, CommitRefusal, Health, StateBook},
    dims::{KvMode, StepDims, StepOptions},
    paging_plan::PagingPlan,
    paging_store::{
        PagedKv,
        device_first::{DeviceCachePlan, DeviceFirstKv},
    },
    state_layout::{Plane, StateLayout, Word},
};
use crate::{
    EngError,
    family::{AttentionKind, project::cuda::WeightLayout},
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream,
        resident::step_params::{
            OFFSET_KV_LEN, OFFSET_KV_SLOT, OFFSET_POS, OFFSET_STATUS, OFFSET_TOKEN_ID, StepParams,
        },
    },
};

pub(super) mod device_tip;
pub(super) mod export;
pub(super) mod pinned;
pub(super) mod prefix;

#[cfg(all(test, feature = "native-cuda"))]
mod paged_prefix_tests;
#[cfg(all(test, feature = "native-cuda"))]
mod prefix_refusal_tests;
#[cfg(test)]
mod tests;

/// Names the element orders of `state_layout.rs`. Change it whenever one of them changes.
pub(crate) const LAYOUT_TAG: &str = "qwen36-gguf-xq-1";
const F32: usize = size_of::<f32>();

// `poison_device` writes `pos`, `kv_len` and `kv_slot` as one run of 12 bytes.
const _: () = assert!(OFFSET_KV_LEN == OFFSET_POS + 4 && OFFSET_KV_SLOT == OFFSET_POS + 8);

/// One 4-byte field of the `StepParams` block. A kernel that reads or writes a single field takes
/// that 4-byte range and never the 32-byte block: the router would otherwise OR its bit into
/// `token_id` (byte 0). `TokenId` is the id the embedding gathers; `kv_slot` gets a variant when a
/// kernel takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepField {
    TokenId,
    Pos,
    KvLen,
    Status,
}

impl StepField {
    /// Byte offset inside `StepParams`, from the shared `StepParams` offset constants.
    pub(crate) const fn offset(self) -> usize {
        match self {
            Self::TokenId => OFFSET_TOKEN_ID,
            Self::Pos => OFFSET_POS,
            Self::KvLen => OFFSET_KV_LEN,
            Self::Status => OFFSET_STATUS,
        }
    }
}

/// What must match for a checkpoint to be importable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateIdentity {
    pub(crate) layout: &'static str,
    pub(crate) kv_mode: KvMode,
    delta_layers: usize,
    attn_layers: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_taps: usize,
    conv_dim: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl StateIdentity {
    pub(crate) fn new(dims: &StepDims, kv_mode: KvMode) -> Self {
        let (d, a) = (&dims.shape.delta, &dims.shape.attn);
        Self {
            layout: LAYOUT_TAG,
            kv_mode,
            delta_layers: dims.layers_of(AttentionKind::DeltaNet),
            attn_layers: dims.layers_of(AttentionKind::Full),
            value_heads: d.value_heads,
            key_dim: d.key_dim,
            value_dim: d.value_dim,
            conv_taps: d.conv_kernel - 1,
            conv_dim: d.conv_dim(),
            kv_heads: a.heads_kv,
            head_dim: a.head_dim,
        }
    }
}

/// Checked physical state sizing, retained from admission through allocation.
pub(crate) struct PagedStatePlan {
    layout: StateLayout,
    identity: StateIdentity,
    paging: PagingPlan,
    host_limit_bytes: usize,
    cache: Option<DeviceCachePlan>,
}

impl PagedStatePlan {
    pub(super) fn check_options(options: &StepOptions) -> Result<(), EngError> {
        if options.kv_mode != KvMode::Int8G32
            || options.layout != WeightLayout::Native
            || options.tiered.is_some()
            || options.forked_experts
        {
            return Err(EngError::BadRequest(
                "paged state requires Native Int8G32 without tiered or forked experts".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn new_cached(
        dims: &StepDims,
        options: &StepOptions,
        cache: DeviceCachePlan,
    ) -> Result<Self, EngError> {
        Self::check_options(options)?;
        cache.check_geometry(dims, options.context_tokens)?;
        let layout = StateLayout::new(dims, options)?;
        if layout.delta.total == 0 || layout.kv.total == 0 {
            return Err(EngError::Checkpoint(
                "the family needs both DeltaNet and attention layers".into(),
            ));
        }
        let plan = Self {
            layout,
            identity: StateIdentity::new(dims, options.kv_mode),
            paging: PagingPlan::new(dims, options.context_tokens)?,
            host_limit_bytes: cache.host_limit(),
            cache: Some(cache),
        };
        plan.device_bytes()?;
        Ok(plan)
    }

    pub(super) fn capacity(&self) -> usize {
        self.layout.kv.capacity
    }
    pub(super) fn cache_rows(&self) -> Option<usize> {
        self.cache.as_ref().map(DeviceCachePlan::rows)
    }

    pub(crate) fn new(
        dims: &StepDims,
        options: &StepOptions,
        host_limit_bytes: usize,
    ) -> Result<Self, EngError> {
        Self::check_options(options)?;
        let layout = StateLayout::new(dims, options)?;
        if layout.delta.total == 0 || layout.kv.total == 0 {
            return Err(EngError::Checkpoint(
                "the family needs both DeltaNet and attention layers".to_owned(),
            ));
        }
        let paging = PagingPlan::new(dims, options.context_tokens)?;
        let minimum = PagedKv::minimum_host_bytes(&paging)?;
        if minimum > host_limit_bytes {
            return Err(EngError::MemoryPlan(format!(
                "paged state needs at least {minimum} host bytes; limit is {host_limit_bytes}"
            )));
        }
        let plan = Self {
            layout,
            identity: StateIdentity::new(dims, options.kv_mode),
            paging,
            host_limit_bytes,
            cache: None,
        };
        plan.device_bytes()?;
        Ok(plan)
    }

    pub(super) fn page_cells(&self) -> usize {
        self.paging.group_cells
    }

    pub(crate) fn device_bytes(&self) -> Result<usize, EngError> {
        let kv_bytes = match &self.cache {
            Some(cache) => cache.device_bytes()?,
            None => self.paging.device_bytes,
        };
        self.layout
            .delta
            .total
            .checked_add(self.layout.words.total)
            .and_then(|bytes| bytes.checked_add(kv_bytes))
            .ok_or_else(|| EngError::MemoryPlan("paged state device size overflows".to_owned()))
    }
}

enum KvStorage {
    Resident(DeviceBuffer),
    Paged {
        store: Box<RefCell<PagedKv>>,
        device_bytes: usize,
    },
    Cached {
        store: Box<RefCell<DeviceFirstKv>>,
        device_bytes: usize,
    },
}

/// The logical state and its selected physical KV owner. Paged serving/snapshots are staged.
pub(crate) struct ResidentState {
    // Declaration order is drop order: the larger KV pool goes first.
    kv: KvStorage,
    delta: DeviceBuffer,
    words: DeviceBuffer,
    layout: StateLayout,
    identity: StateIdentity,
    // Import-compatible layouts do not identify the same physical KV owner.
    instance: Rc<()>,
    book: StateBook,
    resident_tip: Option<device_tip::DeviceTipKey>,
}

impl ResidentState {
    /// Allocates the buffers and zeroes the DeltaNet state and the words, ordered on `stream`,
    /// which becomes the owning stream. `Int8G32` allocates the four KV8 planes of
    /// `state_layout.rs` and zeroes their codes and scales, so a cell that no step appended
    /// holds scale 0 and code 0 (it dequantizes to 0, never NaN); the `Fp16` path is unchanged
    /// (its KV plane holds no valid row until a step appends one, as before).
    pub(crate) fn allocate(
        context: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        let layout = StateLayout::new(dims, options)?;
        if layout.delta.total == 0 || layout.kv.total == 0 {
            return Err(EngError::Checkpoint(
                "the family needs both DeltaNet and attention layers".to_owned(),
            ));
        }
        let delta = context.allocate(layout.delta.total)?;
        let kv = KvStorage::Resident(context.allocate(layout.kv.total)?);
        let words = context.allocate(layout.words.total)?;
        let book = StateBook::new(layout.kv.capacity, stream.id());
        let mut state = Self {
            kv,
            delta,
            words,
            layout,
            identity: StateIdentity::new(dims, options.kv_mode),
            instance: Rc::new(()),
            book,
            resident_tip: None,
        };
        if matches!(options.kv_mode, KvMode::Int8G32) {
            state.clear_kv(stream)?;
        }
        state.reset(stream)?;
        Ok(state)
    }

    pub(crate) fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Physical KV device bytes; paged mode holds only its two transfer slots.
    pub(crate) fn kv_bytes(&self) -> usize {
        match &self.kv {
            KvStorage::Resident(buffer) => buffer.bytes(),
            KvStorage::Paged { device_bytes, .. } | KvStorage::Cached { device_bytes, .. } => {
                *device_bytes
            }
        }
    }

    pub(super) fn cached_store(&self) -> Option<&RefCell<DeviceFirstKv>> {
        match &self.kv {
            KvStorage::Cached { store, .. } => Some(store),
            _ => None,
        }
    }

    pub(super) fn paged_store(&self) -> Result<&RefCell<PagedKv>, EngError> {
        match &self.kv {
            KvStorage::Paged { store, .. } => Ok(store),
            KvStorage::Resident(_) | KvStorage::Cached { .. } => Err(EngError::BadRequest(
                "paged mixer resources require a paged state".to_owned(),
            )),
        }
    }

    fn resident_kv(&self) -> Result<&DeviceBuffer, EngError> {
        match &self.kv {
            KvStorage::Resident(buffer) => Ok(buffer),
            KvStorage::Paged { .. } | KvStorage::Cached { .. } => Err(EngError::BadRequest(
                "paged state has no full resident KV plane; snapshot/mixer integration is pending"
                    .to_owned(),
            )),
        }
    }

    pub(crate) fn identity(&self) -> &StateIdentity {
        &self.identity
    }

    pub(crate) fn health(&self) -> Health {
        self.book.health()
    }

    /// Committed rows.
    pub(crate) fn tokens(&self) -> usize {
        self.book.tokens()
    }

    /// The recurrent state of the `ordinal`-th DeltaNet layer in `slot`, as a checked range.
    pub(crate) fn delta_rec(
        &self,
        slot: usize,
        ordinal: usize,
    ) -> Result<DeviceRange<'_>, EngError> {
        let pool = self.layout.delta;
        let bytes = pool.rec_values * F32;
        self.delta.range(pool.rec_offset(slot, ordinal)?, bytes)
    }

    /// The convolution state of the same layer, as a checked range.
    pub(crate) fn delta_conv(
        &self,
        slot: usize,
        ordinal: usize,
    ) -> Result<DeviceRange<'_>, EngError> {
        let pool = self.layout.delta;
        let bytes = pool.conv_values * F32;
        self.delta.range(pool.conv_offset(slot, ordinal)?, bytes)
    }

    /// One whole plane (capacity positions) of the `ordinal`-th attention layer's KV cache.
    pub(crate) fn kv_plane(
        &self,
        ordinal: usize,
        plane: Plane,
    ) -> Result<DeviceRange<'_>, EngError> {
        let (offset, bytes) = self.layout.kv.plane_extent(ordinal, plane)?;
        self.resident_kv()?.range(offset, bytes)
    }

    /// A persistent word, as a checked range of the bytes the kernels use.
    pub(crate) fn word(&self, word: Word) -> Result<DeviceRange<'_>, EngError> {
        self.words.range(word.offset(), word.bytes())
    }

    /// One 4-byte field of `StepParams` as a checked range, for the kernels that take a single
    /// word: `attn_prep` (`Pos`, `Status`), `attn_decode` (`KvLen`), the router (`Status`).
    pub(crate) fn step_field(&self, field: StepField) -> Result<DeviceRange<'_>, EngError> {
        self.words
            .range(Word::StepParams.offset() + field.offset(), 4)
    }

    /// The `status` word: every status writer ORs its bit into this range only.
    pub(crate) fn step_status(&self) -> Result<DeviceRange<'_>, EngError> {
        self.step_field(StepField::Status)
    }

    /// Admits one more step of `advance` rows (`ensure_rows` over the whole look-ahead window)
    /// and returns the host mirror of the device `pos` word at its start, for `attn_prep`'s
    /// `admit_pos`. Per-row order in stage 1: `write_step_params`, `enqueue_rows`, `run_block`,
    /// `commit_step`. Any `Err` from `run_block` means `poison`.
    pub(crate) fn enqueue_rows(&mut self, advance: usize) -> Result<Admitted, EngError> {
        let admitted = self.book.enqueue(advance)?;
        self.invalidate_tip_before_write();
        Ok(admitted)
    }

    /// Takes back the newest `enqueue_rows` when the step failed before `run_block` queued
    /// anything. After an `Err` from `run_block` call `poison` instead.
    pub(crate) fn cancel_enqueue(&mut self) -> Result<(), EngError> {
        self.book.cancel_last()
    }

    /// Commits the oldest pending step after the host read its result pair; `status` is the pair's
    /// status word. **In stage 1 any non-zero status bit (1, 2, 4, 8 or unknown) poisons the
    /// state** and commits nothing; the refusal is the wrapper's status class (`Backend`,
    /// `StepStatus::error()`, precedence 4 > 8 > 1 > 2) behind `CommitRefusal::Status`, which
    /// converts to `EngError` with that class. Stage 1 commits whole steps only (`accepted` equals
    /// the step's advance): a lower count means the step already ran, the DeltaNet slot is ahead of
    /// `tokens` (there is no per-row snapshot) and the step cannot be undone, so
    /// `CommitRefusal::Partial` POISONS too. Any other refusal changes nothing.
    pub(crate) fn commit_step(
        &mut self,
        accepted: usize,
        status: i32,
    ) -> Result<(), CommitRefusal> {
        self.book.commit(accepted, status)
    }

    /// Marks the state untrustworthy until the next `reset` or a successful `import_state`: the
    /// executor calls it for ANY non-zero status bit of a step or prefill report (1, 2, 4, 8 or
    /// unknown bits) and for any error after the first launch of a step. Host flag only: it stops
    /// new work at `enqueue_rows` and gates commit, export and `revalidate`. It cannot cancel
    /// kernels already queued; see `poison_device`.
    pub(crate) fn poison(&mut self) {
        self.resident_tip = None;
        self.book.poison();
    }

    /// `poison`, plus a write of -1 into `pos`, `kv_len` and `kv_slot` (12 bytes; `status` and
    /// `token_id` stay, so mask bits the host still needs survive). The write is queued on
    /// `stream` BEHIND everything already queued, so it cannot stop replays that were queued
    /// before it: those run on valid words and their results are discarded. It protects launches
    /// queued after it (graph replays that bypass `enqueue_rows`: their consumers find an invalid
    /// word and stop, the poison-consumer rule). The caller must stop enqueuing, synchronize the
    /// stream before it reads anything, and `reset`. The call blocks the host (`write_at`).
    #[expect(dead_code, reason = "first caller: prefill (0049)")]
    pub(crate) fn poison_device(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        self.book.begin_poison_device(stream.id())?;
        self.resident_tip = None;
        let poison = [0xFF_u8; 12];
        self.words
            .write_at(stream, Word::StepParams.offset() + OFFSET_POS, &poison)
    }

    /// Opens a prefill chunk: decode steps are refused until `revalidate`.
    pub(crate) fn begin_prefill(&mut self) -> Result<(), EngError> {
        self.book.begin_prefill()?;
        self.invalidate_tip_before_write();
        Ok(())
    }

    /// Publishes `tokens` committed rows after a clean prefill chunk. Allowed only from a prefill
    /// opened by `begin_prefill`: a state poisoned by a status report stays poisoned until reset.
    /// `prefill_status` is the prefill status word the caller read after joining every other
    /// stream into `stream`; non-zero poisons. This call waits for `stream` itself, so the
    /// chunk's writes have finished; a failed wait poisons. A start block is owed afterwards.
    #[cfg_attr(not(test), expect(dead_code, reason = "first caller: prefill (0049)"))]
    pub(crate) fn revalidate(
        &mut self,
        stream: &DeviceStream,
        tokens: usize,
        prefill_status: i32,
    ) -> Result<(), EngError> {
        self.book.begin_revalidate(stream.id())?;
        stream.synchronize().inspect_err(|_| self.book.poison())?;
        self.book.revalidate(tokens, prefill_status)
    }

    /// Zeroes the prefill status word: a 4-byte `write_at` of `Word::PrefillStatus`, ordered on
    /// `stream` behind the work already queued there. Called at chunk open, right after
    /// `begin_prefill`: the stream is drained at a chunk boundary (the previous chunk ended with
    /// its status read), so the write queues behind nothing, and a bit left by an earlier chunk
    /// cannot fail this one. Blocking like every `write_at` of this type. A failed write leaves the
    /// state `PrefillOpen`, not poisoned: an open chunk already refuses decode, commit, export and
    /// a new `begin_prefill`, `reset` accepts any health, and the driver poisons on any error of
    /// the chunk body.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "first caller: the chunked prefill driver")
    )]
    pub(crate) fn clear_prefill_status(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.book
            .begin_prefill_word(stream.id(), "clear_prefill_status")?;
        self.words
            .write_at(stream, Word::PrefillStatus.offset(), &[0u8; 4])
    }

    /// Reads the prefill status word after the chunk's work, once per chunk, at chunk close:
    /// one ordered 4-byte D2H on `stream` (`read_at` queues the copy behind the chunk's
    /// launches and waits for the stream, the sync pattern of every host copy of this type),
    /// after joining every other stream into `stream`, as `revalidate` requires of its caller.
    /// The value goes to `revalidate` as `prefill_status`, which maps it (`status_error`,
    /// precedence 4 > 8 > 1 > 2). A failed read leaves the state `PrefillOpen`, not poisoned,
    /// for the same reasons as a failed clear.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "first caller: the chunked prefill driver")
    )]
    pub(crate) fn read_prefill_status(&self, stream: &DeviceStream) -> Result<i32, EngError> {
        self.book
            .begin_prefill_word(stream.id(), "read_prefill_status")?;
        let mut word = [0u8; 4];
        self.words
            .read_at(stream, Word::PrefillStatus.offset(), &mut word)?;
        Ok(i32::from_le_bytes(word))
    }

    /// Writes the start block of a step chain (`StepParams::new(token, pos)`), ordered on
    /// `stream`. Only at a committed boundary with `pos == tokens`: with replays in flight the
    /// device owns the words (token chaining), and a `pos` that differs from the host mirror
    /// attends over rows that were never written. Stage 1 per-row order: this call, then
    /// `enqueue_rows`, `run_block`, `commit_step`. Required after a reset, an import and a
    /// prefill: `enqueue_rows` refuses until it has run. A failed write poisons the state (rule 3).
    pub(crate) fn write_step_params(
        &mut self,
        stream: &DeviceStream,
        params: &StepParams,
    ) -> Result<(), EngError> {
        self.book.begin_write_params(stream.id(), params)?;
        self.words
            .write_at(stream, Word::StepParams.offset(), &params.to_bytes())
            .inspect_err(|_| self.book.poison())?;
        self.book.params_written();
        Ok(())
    }

    /// The guards of a block: `stream` owns the state and the state is valid.
    pub(crate) fn check_block(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.book.begin_block(stream.id())
    }

    /// The book guard of the chunk body: the owning stream and an open prefill chunk; `check_block`
    /// requires `Valid` and refuses every chunk.
    pub(super) fn check_chunk(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.book.begin_prefill_word(stream.id(), "run_chunk")
    }

    /// Zeroes every DeltaNet slot and the words in place, ordered on `stream`, and forgets the KV
    /// positions and the pending steps (the copies run behind all work queued on the stream).
    ///
    /// Reset invariant under Int8G32: a reset leaves the STALE codes and scales in every cell >=
    /// the new kv_len — nothing rewrites or clears them. That is correct only because
    /// `attn_decode_q8` reads cells `[0, kv_len)` and every cell it does read was appended in the
    /// CURRENT session before it is read: `pos` restarts at 0 after the reset, so the decode at
    /// position p (kv_len = p + 1) reads exactly the cells the tokens 0..=p of this session
    /// appended. The step device test pins it: 64 tokens of a foreign prompt, a reset, then P5 is
    /// bitwise a fresh session's P5.
    pub(crate) fn reset(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        let valid = self.health() == Health::Valid;
        self.book.begin_reset(stream.id())?;
        if let KvStorage::Paged { store, .. } = &self.kv {
            store
                .try_borrow_mut()
                .map_err(|_| EngError::BadRequest("paged state is already borrowed".to_owned()))?
                .invalidate(stream)?;
        }
        if let KvStorage::Cached { store, .. } = &self.kv {
            store
                .try_borrow_mut()
                .map_err(|_| EngError::BadRequest("cached state is already borrowed".into()))?
                .invalidate(stream)?;
        }
        if !valid {
            self.resident_tip = None;
        }
        self.clear_delta(stream, 0)?;
        self.clear_words(stream)?;
        self.book.finish_overwrite(0);
        Ok(())
    }

    /// Zeroes the DeltaNet slots `first_slot..` (both the recurrent and the convolution state).
    fn clear_delta(&self, stream: &DeviceStream, first_slot: usize) -> Result<(), EngError> {
        let pool = self.layout.delta;
        if matches!(&self.kv, KvStorage::Paged { .. } | KvStorage::Cached { .. }) {
            // Fixed read-only staging keeps reset/import independent of retained prefix size.
            const ZEROS: [u8; 8192] = [0; 8192];
            for slot in first_slot..pool.slots {
                for ordinal in 0..pool.layers {
                    for (offset, bytes) in [
                        (pool.rec_offset(slot, ordinal)?, pool.rec_values * F32),
                        (pool.conv_offset(slot, ordinal)?, pool.conv_values * F32),
                    ] {
                        for at in (0..bytes).step_by(ZEROS.len()) {
                            let len = ZEROS.len().min(bytes - at);
                            self.delta.write_at(stream, offset + at, &ZEROS[..len])?;
                        }
                    }
                }
            }
            return Ok(());
        }
        let zeros = vec![0u8; pool.rec_values.max(pool.conv_values) * F32];
        for slot in first_slot..pool.slots {
            for ordinal in 0..pool.layers {
                let rec = pool.rec_offset(slot, ordinal)?;
                self.delta
                    .write_at(stream, rec, &zeros[..pool.rec_values * F32])?;
                let conv = pool.conv_offset(slot, ordinal)?;
                self.delta
                    .write_at(stream, conv, &zeros[..pool.conv_values * F32])?;
            }
        }
        Ok(())
    }

    /// Zeroes the whole KV pool in place, ordered on `stream` (the copies run behind all work
    /// queued on the stream), in bounded host chunks: the Int8G32 codes and scales start at 0.
    fn clear_kv(&self, stream: &DeviceStream) -> Result<(), EngError> {
        const CHUNK: usize = 1 << 24; // 16 MiB of staging per copy
        let zeros = vec![0u8; CHUNK.min(self.layout.kv.total)];
        for at in (0..self.layout.kv.total).step_by(zeros.len()) {
            let end = (at + zeros.len()).min(self.layout.kv.total);
            self.resident_kv()?
                .write_at(stream, at, &zeros[..end - at])?;
        }
        Ok(())
    }

    /// Rewrites the whole words block with zeros: `StepParams` (an invalid block, `status` 0), the
    /// result pair and the prefill status word.
    fn clear_words(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.words
            .write_at(stream, 0, &vec![0u8; self.layout.words.total])
    }
}

/// Orders work queued on branch streams before whatever the owning stream queues next: each
/// branch records its event and `main` waits for it (fork and join, `DeviceStream::record` and
/// `wait`). `main` must own `state`. The copies of this module wait for their own stream only, so
/// a caller that ran a forked step (stage 1b) joins first, also before a reset after a poison.
/// The record and its wait are queued next to each other, so one event may serve several
/// branches.
#[expect(dead_code, reason = "first caller: the forked step (stage 1b)")]
pub(crate) fn join_branches(
    state: &ResidentState,
    main: &DeviceStream,
    branches: &[(&DeviceStream, &DeviceEvent)],
) -> Result<(), EngError> {
    state.book.begin_join(main.id())?;
    for &(branch, event) in branches {
        branch.record(event)?;
        main.wait(event)?;
    }
    Ok(())
}

impl ResidentState {
    /// Chooses host-paged KV before any full resident KV allocation.
    /// Step/mixer and packed snapshot integration remain separate delivery slices.
    pub(crate) fn allocate_paged(
        context: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        options: &StepOptions,
        host_limit_bytes: usize,
    ) -> Result<Self, EngError> {
        context.ensure_uncaptured("paged state construction")?;
        context.check_launch_stream(stream)?;
        let plan = PagedStatePlan::new(dims, options, host_limit_bytes)?;
        Self::allocate_from_plan(context, stream, &plan)
    }

    pub(crate) fn allocate_from_plan(
        context: &CudaContext,
        stream: &DeviceStream,
        plan: &PagedStatePlan,
    ) -> Result<Self, EngError> {
        context.ensure_uncaptured("paged state construction")?;
        context.check_launch_stream(stream)?;
        let kv = match &plan.cache {
            Some(cache) => KvStorage::Cached {
                store: Box::new(RefCell::new(DeviceFirstKv::allocate(
                    context, stream, cache,
                )?)),
                device_bytes: cache.device_bytes()?,
            },
            None => KvStorage::Paged {
                store: Box::new(RefCell::new(PagedKv::allocate_planned(
                    context,
                    &plan.paging,
                    plan.host_limit_bytes,
                )?)),
                device_bytes: plan.paging.device_bytes,
            },
        };
        let delta = context.allocate(plan.layout.delta.total)?;
        let words = context.allocate(plan.layout.words.total)?;
        let mut state = Self {
            kv,
            delta,
            words,
            layout: plan.layout,
            identity: plan.identity.clone(),
            instance: Rc::new(()),
            book: StateBook::new(plan.capacity(), stream.id()),
            resident_tip: None,
        };
        state.reset(stream)?;
        Ok(state)
    }
}
