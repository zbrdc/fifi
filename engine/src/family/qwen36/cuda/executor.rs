//! The `ExecutorOps` adapter of the resident Qwen3.6 step (decision 0033, packet RD-9d).
//!
//! `QwenResidentStep` is the adapter itself, not a new trait. It is the model side of the shared
//! decode session (`family/session.rs`, packet Q36-F1): it implements [`TokenStepper`], and the
//! session type is the shared [`DecodeSession`], which owns the committed tokens, the pending
//! token, the finish flag, EOS, the output limit, the context limit and cancellation. This file
//! holds no finish rule of its own. The device state lives in the step.
//!
//! The call order and behaviour are those of the Flash-Next and Q36 executors: `reset`,
//! `boundary`, `prefill` for each chunk, then `decode` until a finish. A prompt feeds every token
//! but the last through `feed` (the logits are computed and not read: one code path), and each
//! `decode` feeds the pending token through `logits`, whose row the host sampler draws from. The
//! device argmax token of the pair is the chain value of the gate, not the sampler's choice
//! (decision 0033 item 5: no device sampler).
//!
//! Errors. A failure inside the step poisons the state (`step.rs`) and the original error is
//! returned unchanged. After any error the caller resets the session before it reuses it; the
//! engine worker owns that recovery (`sched/worker.rs`), exactly as for the Q36 executor.
//!
//! `Projector` stays the parity oracle for `linear` and `expert_linear`, and `rows` keeps
//! returning `BadRequest`: nothing here touches either.
//!
//! Design guidance only, no code copied.

use super::step::QwenResidentStep;
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, TokenStepper, check_prompt},
    },
    tokenizer::DecodeBatch,
};
use std::sync::Arc;

type Result<T> = std::result::Result<T, EngError>;

impl TokenStepper for QwenResidentStep<'_> {
    const OPERATION: &'static str = "qwen36_cuda";

    /// Feeds one prompt token: the whole step runs, nothing is read back.
    fn feed(&mut self, token: u32) -> Result<()> {
        self.run_token(token, false).map(|_| ())
    }

    /// Feeds `token` and returns the full logits row of the next token.
    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.run_token(token, true)?;
        Ok(self.logits_values())
    }

    fn fed(&self) -> usize {
        self.tokens()
    }

    fn reset(&mut self) -> Result<()> {
        self.reset_state()
    }
}

impl ExecutorOps for QwenResidentStep<'_> {
    type Session = DecodeSession;

    /// Refuses a prompt that cannot fit before any state changes. The limit is the same KV
    /// capacity that `prefill` enforces chunk by chunk (`self.context_limit()`).
    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        check_prompt(prompt_tokens, self.context_limit())
    }

    fn history<'s>(&self, st: &'s DecodeSession) -> &'s [u32] {
        st.history()
    }

    fn reset(&mut self, st: &mut DecodeSession) -> Result<()> {
        st.reset(self)
    }

    fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<()> {
        let context_limit = self.context_limit();
        st.prefill(self, toks, context_limit)
    }

    fn decode(
        &mut self,
        st: &mut DecodeSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        // The ids are shared, so the limits do not borrow the step that `decode` borrows mutably.
        let eos = Arc::clone(&self.rules.eos);
        let limits = Limits {
            eos: &eos,
            sampler_vocab: self.rules.sampler_vocab,
            context_limit: self.context_limit(),
        };
        st.decode(self, generation, &limits)
    }

    /// No transient routing or draft state exists: the step keeps nothing between turns.
    fn boundary(&mut self, _st: &mut DecodeSession) {}
}
