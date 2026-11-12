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
//! the defaults of `ExecutorOps` hold — zero roots, nothing saved or restored; armed, the last
//! tip boundary is saved and a later prompt with that prefix restores it (one entry, the arena
//! generation as its path).
//!
//! Errors. A failure inside the step poisons the driver and the original error comes back
//! unchanged; the worker resets the session after any error, as for the other executors.

use std::sync::Arc;

use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, PrefixStore, SaveKind, check_prompt},
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
    /// The prefix store of the resume switch (`INFERENCE_FLASH_NEXT_PREFIX_RESUME`): the last
    /// tip only (the default limits, one entry, no roots), stamped with the arena generation
    /// that wrote its rows (the store's path) so an entry never matches after a fresh sequence
    /// overwrote them. Unarmed, nothing is saved or restored.
    prefix: PrefixStore<ResumeSnapshot, u64>,
}

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
                prefix: PrefixStore::default(),
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
        st.save_prefix(
            &mut self.prefix,
            key,
            generation,
            self.driver.fed(),
            kind,
            || self.driver.save_resume(),
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
        st.restore_prefix(&mut self.prefix, prompt, generation, |snapshot| {
            driver.restore_resume(snapshot)
        })
    }
}

#[cfg(test)]
mod tests;
