//! The resident state owner (decision 0047 items 3 and 10, packet RD-9b1; the device half of the
//! former RD-9a2).
//!
//! [`ResidentState`] holds the three device buffers of [`StateLayout`] for one session: DeltaNet
//! recurrent and convolution state, the KV cache, and the persistent words (`StepParams`, the
//! result pair, the prefill status word). Addresses never change after allocation (captured graphs
//! bake them in), so there is no grow and no reallocation; `reset` and `import_state` write in
//! place. The type is thread confined because its buffers are. Slot 0 is the committed state. The
//! rows, the pending steps and the health live in the pure [`StateBook`]; the host export lives in
//! `export`.
//!
//! Every host copy goes through `DeviceBuffer::read_at` / `write_at` (RD-2c, `buffer_io.rs`), which
//! take the `DeviceStream` that owns the work on the buffer, queue the copy behind that work and
//! wait for that stream only. Every state call checks that the stream is the one the state was
//! allocated with (`DeviceStream::id`, `context/stream.rs`). Bytes produced on another stream (a
//! forked expert branch) must be joined into the owning stream with [`join_branches`] first.
//!
//! The device is little endian, so every value is read and written as little-endian bytes.
//!
//! Design guidance only, no code copied.

use super::{
    book::{Admitted, CommitRefusal, Health, StateBook},
    dims::{KvMode, StepDims, StepOptions},
    state_layout::{Plane, StateLayout, Word},
};
use crate::{
    EngError,
    family::AttentionKind,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream,
        resident::step_params::{
            OFFSET_KV_LEN, OFFSET_KV_SLOT, OFFSET_POS, OFFSET_STATUS, OFFSET_TOKEN_ID, StepParams,
        },
    },
};

pub(super) mod export;

/// Names the element orders of `state_layout.rs`. Change it whenever one of them changes.
pub(crate) const LAYOUT_TAG: &str = "qwen36-gguf-xq-1";
const F32: usize = size_of::<f32>();

// `poison_device` writes `pos`, `kv_len` and `kv_slot` as one run of 12 bytes.
const _: () = assert!(OFFSET_KV_LEN == OFFSET_POS + 4 && OFFSET_KV_SLOT == OFFSET_POS + 8);

/// One 4-byte field of the `StepParams` block. A kernel that reads or writes a single field takes
/// that 4-byte range and never the 32-byte block: the router would otherwise OR its bit into
/// `token_id` (byte 0). `TokenId` is the id the
/// embedding of RD-4c gathers; `kv_slot` gets a variant when a kernel takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepField {
    TokenId,
    Pos,
    KvLen,
    Status,
}

impl StepField {
    /// Byte offset inside `StepParams`, from the merged RD-4d constants.
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

/// The device buffers of one session.
pub(crate) struct ResidentState {
    // Declaration order is drop order: the larger KV pool goes first.
    kv: DeviceBuffer,
    delta: DeviceBuffer,
    words: DeviceBuffer,
    layout: StateLayout,
    identity: StateIdentity,
    book: StateBook,
}

impl ResidentState {
    /// Allocates the buffers and zeroes the DeltaNet state and the words, ordered on `stream`,
    /// which becomes the owning stream. `Int8G32` allocates from RD-8k.
    pub(crate) fn allocate(
        context: &CudaContext,
        stream: &DeviceStream,
        dims: &StepDims,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        if options.kv_mode != KvMode::Fp16 {
            return Err(EngError::BadRequest(
                "the Int8G32 KV pool is allocated from RD-8k".to_owned(),
            ));
        }
        let layout = StateLayout::new(dims, options)?;
        if layout.delta.total == 0 || layout.kv.total == 0 {
            return Err(EngError::Checkpoint(
                "the family needs both DeltaNet and attention layers".to_owned(),
            ));
        }
        let delta = context.allocate(layout.delta.total)?;
        let kv = context.allocate(layout.kv.total)?;
        let words = context.allocate(layout.words.total)?;
        let book = StateBook::new(layout.kv.capacity, stream.id());
        let mut state = Self {
            kv,
            delta,
            words,
            layout,
            identity: StateIdentity::new(dims, options.kv_mode),
            book,
        };
        state.reset(stream)?;
        Ok(state)
    }

    pub(crate) fn layout(&self) -> &StateLayout {
        &self.layout
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
        self.kv.range(offset, bytes)
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
        self.book.enqueue(advance)
    }

    /// Takes back the newest `enqueue_rows` when the step failed before `run_block` queued
    /// anything. After an `Err` from `run_block` call `poison` instead.
    pub(crate) fn cancel_enqueue(&mut self) -> Result<(), EngError> {
        self.book.cancel_last()
    }

    /// Commits the oldest pending step after the host read its result pair; `status` is the
    /// pair's status word. **In stage 1 any non-zero status bit (1, 2, 4, 8 or unknown) poisons
    /// the state** and commits nothing; the refusal is the wrapper's status class (`Backend`,
    /// `StepStatus::error()`, precedence 4 > 8 > 1 > 2) behind `CommitRefusal::Status`, which
    /// converts to `EngError` with that class. Stage 1 commits whole steps only (`accepted`
    /// equals the step's advance): a lower count means the step already ran, the DeltaNet slot
    /// is ahead of `tokens` (no per-row snapshot until RD-7c) and the step cannot be undone, so
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
    /// unknown bits; REPLY-76 Q2) and for any error after the first launch of a step. Host flag
    /// only: it stops new work at `enqueue_rows` and gates commit, export and `revalidate`. It
    /// cannot cancel kernels already queued; see `poison_device`.
    pub(crate) fn poison(&mut self) {
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
        let poison = [0xFF_u8; 12];
        self.words
            .write_at(stream, Word::StepParams.offset() + OFFSET_POS, &poison)
    }

    /// Opens a prefill chunk (decision 0049): decode steps are refused until `revalidate`.
    pub(crate) fn begin_prefill(&mut self) -> Result<(), EngError> {
        self.book.begin_prefill()
    }

    /// Publishes `tokens` committed rows after a clean prefill chunk. Allowed only from a prefill
    /// opened by `begin_prefill`: a state poisoned by a status report stays poisoned until reset.
    /// `prefill_status` is the prefill status word the caller read after joining every other
    /// stream into `stream`; non-zero poisons. This call waits for `stream` itself, so the
    /// chunk's writes have finished; a failed wait poisons. A start block is owed afterwards.
    #[expect(dead_code, reason = "first caller: prefill (0049)")]
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

    /// Zeroes every DeltaNet slot and the words in place, ordered on `stream`, and forgets the KV
    /// positions and the pending steps (the copies run behind all work queued on the stream).
    pub(crate) fn reset(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        self.book.begin_reset(stream.id())?;
        self.clear_delta(stream, 0)?;
        self.clear_words(stream)?;
        self.book.finish_overwrite(0);
        Ok(())
    }

    /// Zeroes the DeltaNet slots `first_slot..` (both the recurrent and the convolution state).
    fn clear_delta(&self, stream: &DeviceStream, first_slot: usize) -> Result<(), EngError> {
        let pool = self.layout.delta;
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
#[expect(dead_code, reason = "first caller: RD-11b (stage 1b)")]
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
