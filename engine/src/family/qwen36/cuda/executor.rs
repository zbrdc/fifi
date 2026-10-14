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
//! Prefix reuse (decision 0036, addendum 1) adds no rule of its own: the two methods below are
//! the same glue as the Mac arm, and the shared session rules decide everything. What differs is
//! the state value. The CUDA export COPIES slot 0 and the first `tokens` KV rows to host bytes
//! and the restore copies them back into the same buffers (the base decision's item 5), so a
//! restored turn is bitwise against a cold one. A tiered step never saves: its numbers follow
//! the bank state, which follows the history. A capture needs no invalidation: the graph bakes
//! addresses, not positions, and the next token rewrites the `pos`/`kv_len` words from the
//! imported `tokens` count.
//!
//! Design guidance only, no code copied.

use super::prefix::{CudaPrefillPath, entry_bytes};
use super::step::QwenResidentStep;
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        session::{DecodeSession, Limits, SaveKind, TokenStepper, check_prompt},
    },
    tokenizer::DecodeBatch,
};
use std::sync::Arc;
use std::time::Instant;

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

    /// Exports the committed state as the one prefix entry (decision 0036, addendum 1), saved
    /// as `kind` (the caller names the save point). The session rules decide whether the call
    /// is valid. A tiered step cannot save: its numbers follow the bank state, so a restored
    /// turn would not match a cold one bitwise.
    fn save_prefix(&mut self, st: &DecodeSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        if self.is_tiered() {
            return Ok(false);
        }
        let fed = self.tokens();
        let began = Instant::now();
        // The slot leaves the step for the call, so the snapshot closure can borrow the step.
        let mut slot = std::mem::take(&mut self.prefix);
        let saved = st.save_prefix(&mut slot, key, self.prefill_path(), fed, kind, || {
            self.export_state()
        });
        self.prefix = slot;
        if matches!(saved, Ok(true)) {
            self.log_prefix("saved", began);
        }
        saved
    }

    /// Copies the saved entry into the reset state when the session rules accept the prompt.
    fn restore_prefix(&mut self, st: &mut DecodeSession, prompt: &[u32]) -> Result<usize> {
        let began = Instant::now();
        let mut slot = std::mem::take(&mut self.prefix);
        let covered = st.restore_prefix(&mut slot, prompt, self.prefill_path(), |saved| {
            self.import_state(saved)
        });
        self.prefix = slot;
        if matches!(covered, Ok(n) if n > 0) {
            self.log_prefix("restored", began);
        }
        covered
    }
}

impl QwenResidentStep<'_> {
    /// The prefill path every entry of this step is saved and restored under: the one value a
    /// `CudaPrefillPath` holds today, from one place, so a second variant (chunked prefill,
    /// decision 0049) changes only this body — exactly like the Mac glue.
    fn prefill_path(&self) -> CudaPrefillPath {
        CudaPrefillPath::Steps
    }

    /// One stderr line per saved or restored entry; the G3 turn gate parses the copy times. The
    /// save or restore it reports has already completed, so the line cannot fail the call: a
    /// charge that cannot be computed is named in the line instead. The line's `positions` is
    /// the entry's COMMITTED row count, `key.len() - 1` (the pending token is not a row), while
    /// the request log's `reused_tokens` is `key.len()`: G3's parser adds one to this line's
    /// `positions` to get the reused count.
    fn log_prefix(&self, what: &str, began: Instant) {
        let positions = self.tokens();
        let charge = entry_bytes(self.state_layout(), positions)
            .map_or_else(|error| format!("{error:?}"), |bytes| format!("{bytes} B"));
        eprintln!(
            "q36 cuda: prefix {what} {positions} positions, {charge}, {:.1} ms",
            began.elapsed().as_secs_f64() * 1000.0
        );
    }
}
