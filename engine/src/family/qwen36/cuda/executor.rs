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
use super::step::{ChunkVerdict, QwenResidentStep};
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

/// The prefill arm `feed_rows` takes for one chunk-gate verdict. THE ONE SWITCH of the chunk
/// path (plan §C.4): until the chunk driver lands with CPF-7a, every verdict — each refusal
/// and the accept — maps to the per-token arm, so no session's behaviour changes. CPF-7a's
/// integration commit adds the `Chunked` arm (the `m_sess` cap of the accept verdict) and the
/// driver call in `feed_rows`; the exhaustive match there is what makes the switch one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkPath {
    /// One token per step: the default `TokenStepper::feed_rows` loop's own arm.
    PerToken,
    /// The chunk driver, at the admitted M_sess of the arena.
    Chunked { m_sess: usize },
}

/// The prefill path of one chunk-gate verdict: every verdict keeps the per-token loop until
/// the chunk driver lands with CPF-7a (decision 0049; see `ChunkPath`).
pub(super) fn chunk_path_of(verdict: &ChunkVerdict) -> ChunkPath {
    match verdict {
        ChunkVerdict::Ready { m_sess } => ChunkPath::Chunked { m_sess: *m_sess },
        _ => ChunkPath::PerToken,
    }
}

/// The worker piece of the chunk path (plan CPF-7b-C B.9): the ADMITTED M_sess when the
/// verdict is Ready, else None, so the caller own fallback stands on every refusal.
pub(super) fn piece_of(verdict: &ChunkVerdict) -> Option<usize> {
    match verdict {
        ChunkVerdict::Ready { m_sess } => Some(*m_sess),
        _ => None,
    }
}

/// Feeds `tokens` one by one, stopping at the first failure: the default
/// `TokenStepper::feed_rows` loop's own body, shared by the CUDA override so the two cannot
/// drift while the chunk path is staged.
pub(super) fn feed_per_token(step: &mut impl TokenStepper, tokens: &[u32]) -> Result<()> {
    tokens.iter().try_for_each(|&token| step.feed(token))
}

impl TokenStepper for QwenResidentStep<'_> {
    const OPERATION: &'static str = "qwen36_cuda";

    /// Feeds one prompt token: the whole step runs, nothing is read back.
    fn feed(&mut self, token: u32) -> Result<()> {
        self.run_token(token, false).map(|_| ())
    }

    /// Feeds `tokens` in order as prompt rows, routed through the chunk gate (plan §C.4; the
    /// Flash-Next precedent). Until the chunk driver lands with CPF-7a every verdict takes the
    /// per-token arm — the default loop's own body — so every session behaves exactly as the
    /// default `TokenStepper::feed_rows`; when it lands, `chunk_path_of` is the one call that
    /// flips the accept verdict to the chunk arm.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let verdict = self.chunk_ready();
        self.note_feed_rows(&verdict, tokens.len());
        match chunk_path_of(&verdict) {
            ChunkPath::PerToken => {
                // The per-token arm records its rows too, so the report rows sum to the
                // prompt on every path and the job line can carry them.
                for &token in tokens {
                    let p0 = self.tokens();
                    let opened = Instant::now();
                    TokenStepper::feed(self, token)?;
                    let ns = u64::try_from(opened.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    self.record_feed(p0, ns);
                }
                Ok(())
            }
            ChunkPath::Chunked { m_sess } => self.feed_rows_chunked(tokens, m_sess),
        }
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
        // A job whose line was never taken must not cross into the next job.
        self.log_prefill_job();
        st.reset(self)
    }

    fn prefill(&mut self, st: &mut DecodeSession, toks: &[u32]) -> Result<()> {
        let context_limit = self.context_limit();
        let began = Instant::now();
        let out = st.prefill(self, toks, context_limit);
        self.record_prefill_piece(toks.len(), began);
        out
    }

    /// The worker piece of the chunk path (plan CPF-7b-C B.9, R1): the ADMITTED M_sess when
    /// the verdict is Ready, else None, so the caller own fallback stands on every refusal.
    ///
    /// Cancel-latency trade: the worker checks cancellation only between pieces, so a cancel
    /// waits up to one piece of M_sess rows. MEASURED on the 3090 at cap 2,048: the P1 release
    /// chunk of 1,023 rows took 262.3 ms, status word 0 (
    /// qrig/qb-run-01/result.json, sha256 656a4fe47864f5ad8a73adb9d54a5ebc8edbb7efdde276d03ddd22d172786af2, field steps[0].ns).
    /// A 2,048-row piece is about 525 ms at 1K depth. Up to about 2.9 s at 32K depth
    /// (CPF-8A section B.10 model); the worst case is the deepest admitted fp16 context, near
    /// 100K, where a 2,048-token piece is 4 chunks of 512 at about 8 s (4 x (about 1.81 s
    /// attention + about 0.19 s other); ESTIMATE from the PLAN C.6 FLOP model at 4.56 TFLOP/s,
    /// not measured. The CPF-8a follow-up replaces the 2,048-row figure with
    /// CPF8A.piece_ms_p50 (owner lane S; trigger: the CPF-8a report lands).
    fn prefill_piece(&self) -> Option<usize> {
        piece_of(&self.chunk_ready())
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
        self.log_prefill_job();
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
    /// Writes the one prefill line of a job, once: the first decode after the prefill, or
    /// the next reset when no decode ran (plan CPF-7b-C B.8, OQ-13).
    fn log_prefill_job(&mut self) {
        // A session that has not fed a row yet has nothing to report: an empty line at the
        // first reset would name no path and no pieces.
        if self.last_prefill_report().feed_rows_calls().is_empty() {
            return;
        }
        if let Some(line) = self.take_job_line() {
            eprintln!("{line}");
        }
    }

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
