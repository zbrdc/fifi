//! The `ExecutorOps` adapter of the Flash-Next token driver on a CUDA card.
//!
//! `FlashCudaExecutor` is the engine thread's executor. It follows the Qwen3.6 CUDA executor
//! (`qwen36/cuda/executor.rs`): the session is the shared `DecodeSession`, which owns the committed
//! tokens, the pending token, the finish flag, EOS, the output limit, the context limit and
//! cancellation; the model side is the [`TokenDriver`], a `TokenStepper`. This file holds no finish
//! rule of its own.
//!
//! The prompt is fed through the default `TokenStepper::feed_rows` loop, one token per step: the
//! chunked prefill is a later change and replaces that loop without changing this adapter.
//! Prefix reuse is the `INFERENCE_FLASH_NEXT_PREFIX_RESUME` switch (see `resume.rs`): unarmed,
//! the defaults of `ExecutorOps` hold — zero roots, nothing saved or restored; armed, the tip
//! boundary is saved together with one rolling block below it (two entries, the arena
//! generation as the path).
//!
//! Errors. A failure inside the step poisons the driver and the original error comes back
//! unchanged; the worker resets the session after any error, as for the other executors.

use std::sync::Arc;

use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, PrefixStore, SaveKind, StoreLimits, check_prompt},
    },
    tokenizer::DecodeBatch,
};

use super::driver::{ResumeSnapshot, ResumesState};
use super::resume;

type Result<T> = std::result::Result<T, EngError>;

/// The limits that the caller reads from the tokenizer bundle. The driver supplies the position
/// limit.
pub(in crate::family::flash_next) struct CudaRules {
    /// Every id that ends generation (`TokenizerBundle::eos_token_ids`).
    pub(in crate::family::flash_next) eos: Arc<[u32]>,
    /// The number of logits the sampler may choose from (`TokenizerBundle::vocab_size`). The head
    /// is wider (padded ids above the tokenizer's vocabulary), and those ids have no text.
    pub(in crate::family::flash_next) sampler_vocab: usize,
}

/// The executor of one Flash-Next model on a card. It lives on the engine thread.
pub(in crate::family::flash_next) struct FlashCudaExecutor<S: ResumesState> {
    driver: S,
    rules: CudaRules,
    /// The prefix store of the resume switch (`INFERENCE_FLASH_NEXT_PREFIX_RESUME`): the tip
    /// plus one rolling block cut at `RESUME_BLOCK_SPACING` below it (`RESUME_LIMITS`, no
    /// roots), both stamped with the arena generation that wrote their rows (the store's
    /// path) so an entry never matches after a fresh sequence overwrote them. Unarmed,
    /// nothing is saved or restored.
    prefix: PrefixStore<ResumeSnapshot, u64>,
    /// The host bytes the resume store may hold across its entries (see
    /// [`resume::HOST_BUDGET_BYTES`]); a save that would cross it is skipped.
    host_budget: u64,
    /// The rolling block lattice in tokens; a field, not the constant, so the worker-driven
    /// tests can shrink it below the fake device's position count. Production sets it from
    /// `RESUME_BLOCK_SPACING`.
    block_spacing: usize,
    /// Whether `INFERENCE_FLASH_NEXT_PREFIX_RESUME` was armed when the executor was built;
    /// the trait's store-shape answers read this snapshot, not the live environment, so a
    /// test can pin the unarmed answers without touching the process-global switch.
    armed: bool,
}

/// The resume store's shape: no roots and two entries — the tip and one rolling block. A
/// mid-turn suffix re-renders the turn tail, so its prompt diverges from the tip key a couple
/// of tokens inside the boundary (the chat template eats the trailing space and the tokenizer
/// merges the final stop with `<|im_end|>`); the block key sits far enough below the tip that
/// the suffixed prompt still prefixes it, and the restore re-prefills only the short tail.
/// The worker-visible store shape for a given switch state: the armed two-entry store
/// (tip plus one rolling block), or the mainline one-entry default unarmed — the unarmed
/// worker must plan no block save point and print no block note for an engine that saves
/// nothing.
fn resume_limits(on: bool) -> StoreLimits {
    if on {
        RESUME_LIMITS
    } else {
        StoreLimits::default()
    }
}

fn resume_spacing(on: bool, spacing: usize) -> Option<usize> {
    if on {
        Some(spacing)
    } else {
        // The trait default (family/executor.rs), which reads the mainline
        // ENGINESRV_PREFIX_BLOCK_MATCH / _TOKENS pair: literally what an unarmed
        // executor answered before this switch existed.
        crate::family::session::prefix_block_spacing()
    }
}

const RESUME_LIMITS: StoreLimits = StoreLimits {
    roots: 0,
    entries: 2,
};

/// The rolling block's lattice. The worker cuts the last block boundary strictly below the
/// tip, so the block sits `((tip - 1) % spacing) + 1` tokens under it; the re-rendered seam
/// sits exactly two tokens under the tip (see `RESUME_LIMITS`). A fresh prefill cuts on the
/// lattice, and only the `tip % spacing <= 2` residue classes fall through the seam
/// (2 in 64 at the default spacing). A turn that STARTS from a restored position is
/// different: the worker moves the cut up to the next prefill piece end (it never splits a
/// piece), so with the default 16-row pieces the block sits 0-15 tokens ABOVE the lattice
/// and the gap to the tip falls within the two-token seam for about 13 percent of tips —
/// those suffixes fall back to the tip or a full prefill. A lattice-aware plan (splitting a
/// piece at the lattice point, or cutting below the restored start) is worker-shared code
/// and deliberately its own slice.
const RESUME_BLOCK_SPACING: usize = 64;

impl<S: ResumesState> FlashCudaExecutor<S> {
    /// The executor and its first session. Refuses a sampler vocabulary of 0.
    pub(in crate::family::flash_next) fn new(
        driver: S,
        rules: CudaRules,
    ) -> Result<(Self, DecodeSession)> {
        if rules.sampler_vocab == 0 {
            return Err(EngError::Checkpoint(
                "Flash-Next CUDA: the sampler vocabulary must be above 0".to_string(),
            ));
        }
        Ok((
            Self {
                driver,
                rules,
                // The recent-longer protection keeps a restored Tip ancestor from being the
                // room victim of the very Block save that follows it, so a conversation's
                // block entry survives the turn that restored the tip (S2).
                prefix: PrefixStore::with_limits(RESUME_LIMITS).with_recent_longer_protection(),
                host_budget: resume::HOST_BUDGET_BYTES,
                block_spacing: RESUME_BLOCK_SPACING,
                armed: resume::on(),
            },
            DecodeSession::default(),
        ))
    }

    /// The number of positions the caches accept.
    pub(in crate::family::flash_next) fn context_limit(&self) -> usize {
        self.driver.positions()
    }

    /// The driver, for the device tests that drive the stepper directly.
    #[cfg(test)]
    pub(in crate::family::flash_next) fn stepper(&mut self) -> &mut S {
        &mut self.driver
    }

    /// Builds with a forced resume host budget, for the denial test (the default budget
    /// cannot deny at the default two-entry store).
    #[cfg(test)]
    pub(in crate::family::flash_next) fn with_host_budget_for_tests(
        driver: S,
        rules: CudaRules,
        host_budget: u64,
    ) -> Result<(Self, DecodeSession)> {
        let (mut executor, session) = Self::new(driver, rules)?;
        executor.host_budget = host_budget;
        Ok((executor, session))
    }

    /// The worker-driven tests' scaled store: a smaller block lattice (the fake device holds
    /// six positions) over the same budget ratio the serve pins.
    #[cfg(test)]
    pub(in crate::family::flash_next) fn with_block_spacing_for_tests(
        mut self,
        spacing: usize,
    ) -> Self {
        self.block_spacing = spacing;
        self
    }

    /// Pins the store-shape answers the trait reports (see `armed`), without touching the
    /// process-global switch: the unarmed answers must hold with the switch ON elsewhere.
    #[cfg(test)]
    pub(in crate::family::flash_next) fn with_armed_for_tests(mut self, armed: bool) -> Self {
        self.armed = armed;
        self
    }
}

impl<S: ResumesState> ExecutorOps for FlashCudaExecutor<S> {
    type Session = DecodeSession;

    /// Refuses a prompt that cannot fit before any state changes.
    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        check_prompt(prompt_tokens, self.context_limit())
    }

    fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
        st.history()
    }

    /// The resume store's shape (see `RESUME_LIMITS`); the worker plans its save points with
    /// it. Unarmed the worker sees the default one-entry store — the armed shape would make
    /// it plan a block save point and print the block note for an engine that saves nothing.
    fn prefix_limits(&self) -> StoreLimits {
        resume_limits(self.armed)
    }

    /// The rolling block lattice below the tip (see `RESUME_BLOCK_SPACING`): one block entry
    /// a mid-turn suffix can still prefix when the re-rendered tail breaks the tip key.
    /// Unarmed this answers the trait default (`session::prefix_block_spacing()`), as every
    /// executor did before the switch; no consumer can see it, because the unarmed
    /// one-entry `cuts_block` store suppresses both the block save point and the note.
    fn prefix_block_spacing(&self) -> Option<usize> {
        resume_spacing(self.armed, self.block_spacing)
    }

    fn reset(&mut self, st: &mut DecodeSession) -> Result<()> {
        st.reset(&mut self.driver)
    }

    fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<()> {
        let context_limit = self.context_limit();
        st.prefill(&mut self.driver, toks, context_limit)
    }

    fn decode(
        &mut self,
        st: &mut DecodeSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        let limits = Limits {
            eos: &self.rules.eos,
            sampler_vocab: self.rules.sampler_vocab,
            context_limit: self.driver.positions(),
        };
        st.decode(&mut self.driver, generation, &limits)
    }

    /// No transient routing or draft state exists between turns.
    fn boundary(&mut self, _st: &mut DecodeSession) {}

    /// Saves the boundary state when the resume switch is armed: the fixed device state read
    /// back to the host (the GDN recurrent states, their conv rings, the PLE ring), the position
    /// and the PLE feed's host half. The QSA rows stay on the card; a restore rewinds the valid
    /// length onto them. The core session method enforces the pending-token contract. The entry
    /// carries the arena generation as its path: a restore at any other generation — a fresh
    /// sequence has overwritten the rows since — never matches it and falls back to a full
    /// prefill.
    fn save_prefix(&mut self, st: &DecodeSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        if !resume::on() {
            return Ok(false);
        }
        let generation = self.driver.resume_generation();
        // Dead generations first: entries of another arena generation never candidate
        // again (the restore walk matches the path exactly), so their host bytes come off
        // the line before the charge — one stale entry must not wedge the budget until
        // restart.
        self.prefix.retain_path(&generation);
        // The charge runs after the store has made room, on the bytes the survivors hold
        // plus the would-be entry: a pre-room charge counts the entry this save evicts, so
        // a full two-entry store would deny every save from turn 2 on and reuse would stop
        // advancing where the unbudgeted store rolls forward.
        let budget = self.host_budget;
        let entry_bytes = self.driver.resume_host_bytes();
        let driver = &mut self.driver;
        st.save_prefix_budgeted(
            &mut self.prefix,
            key,
            generation,
            driver.fed(),
            kind,
            &ResumeSnapshot::host_bytes,
            &mut move |held| held.saturating_add(entry_bytes) <= budget,
            || driver.save_resume(),
        )
    }

    /// Restores the saved boundary when the new prompt has it as a prefix: the snapshot goes
    /// back to the device, the position rewinds and the session re-holds the boundary's pending
    /// token, so the prefill of the tail continues the sequence. The core session method runs
    /// the store's candidate walk; only an entry of the current arena generation is a candidate.
    fn restore_prefix(&mut self, st: &mut DecodeSession, prompt: &[u32]) -> Result<usize> {
        if !resume::on() {
            return Ok(0);
        }
        let generation = self.driver.resume_generation();
        let driver = &mut self.driver;
        let restored = st.restore_prefix(&mut self.prefix, prompt, generation, |snapshot| {
            driver.restore_resume(snapshot)
        })?;
        // The tail prefill is about to overwrite every row above the restored boundary, and a
        // restore does not start a fresh sequence, so the arena's generation does not advance
        // for it (the bump sits at the first chunk of a fresh sequence only, `chunk.rs`). An
        // entry with a longer key would keep matching at the same generation while the rows it
        // describes are replaced — the stale-entry defect one boundary up. Drop every longer
        // entry here, before any row is written; the restored entry and any deeper one stay.
        if restored > 0 {
            self.prefix.drop_above(restored);
        }
        Ok(restored)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod worker_tests;
