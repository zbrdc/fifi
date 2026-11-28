//! The scripted test fake for the `ExecutorOps` seam lives in `crate::exec_fake`.

use crate::EngError;
use crate::family::sampler::Sampler;
use crate::family::session::{
    self, DecodeSession, Kept, Limits, PrefixStore, SaveKind, SnapshotBudget, StoreLimits,
    TokenStepper,
};
use crate::family::speculate::budget::DraftBudget;
use crate::sched::job::{PrefillBankReport, PrefixReport};
use crate::tokenizer::{DecodeBatch, IncrementalDecoder};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

type Result<T> = std::result::Result<T, EngError>;

/// What the batch worker does beyond its shared rules, per executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchPolicy {
    /// While no stream decodes, a tick feeds each cold prompt a whole planned piece, for up to
    /// this many cold prompts. 1 keeps the shared budget of one piece per tick for all of them.
    pub(crate) idle_pieces: usize,
    /// A pooled session that has idled this long is released, while one session stays when the
    /// engine is idle. `None` keeps every session for the life of the engine.
    pub(crate) idle_trim: Option<std::time::Duration>,
}

impl Default for BatchPolicy {
    fn default() -> Self {
        Self {
            idle_pieces: 1,
            idle_trim: None,
        }
    }
}

/// The most cold prompts that take a whole piece in one idle tick when an executor opts in.
pub(crate) const IDLE_PIECES_PER_TICK: usize = 8;

/// How long a batch engine sits with no request before it keeps one session. A chat user pauses
/// for seconds between turns, so a pool that outlives that spares the next burst its allocations;
/// a pool that outlives minutes holds the memory of a burst (seven sessions on a Bolt engine is
/// 6.5 GiB) for the life of the process.
pub(crate) const DEFAULT_SESSION_IDLE_TRIM: std::time::Duration =
    std::time::Duration::from_secs(30);

/// Per-job generation state. The executor feeds the decoder inside `decode`.
pub(crate) struct GenerationState {
    pub(crate) decoder: IncrementalDecoder,
    pub(crate) cancel: Arc<AtomicBool>,
    /// The sampler of this job. The executor draws from it inside `decode`.
    pub(crate) sampler: Sampler,
    /// Reset when the worker creates a generation, including on prefix hits.
    pub(crate) draft_budget: DraftBudget,
    max_tokens: Option<u32>,
    generated: u32,
}

impl GenerationState {
    pub(crate) fn new(decoder: IncrementalDecoder) -> Self {
        Self::with_sampler(decoder, Sampler::greedy())
    }

    pub(crate) fn with_sampler(decoder: IncrementalDecoder, sampler: Sampler) -> Self {
        Self {
            decoder,
            cancel: Arc::new(AtomicBool::new(false)),
            sampler,
            draft_budget: DraftBudget::default(),
            max_tokens: None,
            generated: 0,
        }
    }

    /// A limit of zero ends the job on its first `decode` call.
    pub(crate) fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub(crate) fn remaining_tokens(&self) -> Option<u32> {
        self.max_tokens
            .map(|limit| limit.saturating_sub(self.generated))
    }

    /// The tokens that count toward `max_tokens`; an EOS token never counts.
    pub(crate) fn record_generated(&mut self, count: u32) {
        self.generated = self.generated.saturating_add(count);
    }
}

pub(crate) enum PrefillPairOutcome {
    Declined,
    Completed,
    Cancelled,
}

/// The calls the engine worker makes on an executor, in the worker's call order. `check_prompt`
/// comes first: the worker asks it at the start of every job, before it resets a session or
/// feeds a token, so a prompt that cannot fit is refused before any engine work.
pub(crate) trait ExecutorOps {
    type Session;
    /// Existing executors keep their single-session path.
    fn max_seqs(&self) -> usize {
        1
    }
    /// Allocates independent mutable state on this executor's engine thread.
    fn new_session(&mut self) -> Result<Self::Session> {
        Err(EngError::BadRequest(
            "this executor does not support independent batch sessions".into(),
        ))
    }
    /// Advances a row-ordered window. Outer errors invalidate every affected session; inner
    /// errors refuse only that row before its feed. A Backend error at either level is terminal
    /// before any publication. This default never simulates batching.
    fn decode_window(
        &mut self,
        _sessions: &mut [&mut Self::Session],
        _generations: &mut [&mut GenerationState],
    ) -> Result<Vec<Result<DecodeBatch>>> {
        Err(EngError::BadRequest(
            "this executor does not support decode windows".into(),
        ))
    }
    /// Whether a new session can be allocated now. The batch worker asks before a timed retry of a
    /// job that the executor refused for lack of memory, so a retry that cannot fit does not pay an
    /// allocation, a zero fill and a stream drain while other streams decode. The default says yes.
    fn can_admit_session(&self) -> bool {
        true
    }
    /// Whether evicting prefix entries can free the memory a new session needs. Executors whose
    /// entries share the session budget (Qwen3.6) say yes, and the worker evicts one entry at a
    /// time before it reports a denial. An executor whose entries live in host memory while its
    /// sessions live on the device says no: a device shortage is not cured by dropping them, and
    /// they stay.
    fn eviction_frees_session_memory(&self) -> bool {
        true
    }
    /// The choices of the batch worker that an executor opts into once it has measured them. The
    /// default keeps the worker's shared budget and keeps every session for the life of the
    /// engine.
    fn batch_policy(&self) -> BatchPolicy {
        BatchPolicy::default()
    }
    /// The prompt rows the batch worker feeds per tick, over all cold slots together, while a
    /// stream decodes, beside the decode rows. `None` keeps the worker's rule: the iteration's
    /// rows, never below one row per live slot, less the decode rows; the cold prompts are
    /// dealt whole pieces from that budget, at least one a tick and at most one each, and they
    /// take turns. `Some(n)` makes the share independent of the piece size: the worker feeds at
    /// least one planned piece and at most `n` rows when `n` is larger, and a cold slot may run
    /// several pieces in a tick until its part is spent. With no stream decoding every cold
    /// prompt takes whole pieces whatever the share. Only the batch worker reads it.
    fn prompt_rows_per_tick(&self) -> Option<usize> {
        None
    }
    /// An executor with a context limit returns the typed over-length refusal for a prompt
    /// that cannot fit; it changes no state. Executors without a limit accept every prompt.
    fn check_prompt(&self, _prompt_tokens: usize) -> Result<()> {
        Ok(())
    }
    /// The committed tokens of the session.
    fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32];
    fn reset(&mut self, st: &mut Self::Session) -> Result<()>;
    /// Resets a session for an admitted request before any prefix restore or prefill.
    /// A fresh request with a known generation limit may size its initial buffers here.
    fn reset_for_request(
        &mut self,
        st: &mut Self::Session,
        _prompt_tokens: usize,
        _completion_limit: Option<u32>,
        _has_prefix_boundary: bool,
    ) -> Result<()> {
        self.reset(st)
    }
    /// Consumes one non-empty prompt chunk.
    fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<()>;
    /// Consumes the final piece for a request that will decode next. The worker has finished
    /// saving prefixes and will call only decode or reset on this session.
    fn prefill_for_decode(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<()> {
        self.prefill(st, toks)
    }
    /// Tries to consume a fresh prompt as two pieces in one call. A declined
    /// pair leaves the session untouched. A cancelled pair requires a reset.
    fn try_prefill_pair(
        &mut self,
        _st: &mut Self::Session,
        _tokens: &[u32],
        _first_piece: usize,
        _cancel: &AtomicBool,
    ) -> Result<PrefillPairOutcome> {
        Ok(PrefillPairOutcome::Declined)
    }
    /// Commits one batch of tokens, or reports `Cancelled`.
    fn decode(
        &mut self,
        st: &mut Self::Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch>;
    /// Completes deferred work after a batch is sent, before another decode or job reset.
    fn after_emit(&mut self) -> Result<()> {
        Ok(())
    }
    /// Clears transient routing and draft state at a turn boundary.
    fn boundary(&mut self, st: &mut Self::Session);
    /// The number of prompt tokens that one worker piece feeds when this executor plans its own
    /// chunks. The worker refreshes this hint between pieces. `None` means the caller's
    /// fallback; `Some(0)` is invalid and is checked at startup and between pieces.
    fn prefill_piece(&self) -> Option<usize> {
        None
    }
    /// A request's prefill width after reset. The worker bounds it by the startup ceiling
    /// and applies it to save points and every prefill piece of this session.
    fn request_prefill_piece(&self, _st: &Self::Session) -> Option<std::num::NonZeroUsize> {
        None
    }
    /// An optional report from startup after the executor has applied its plan.
    fn mlx_execution_plan(&self) -> Option<&super::mlx_report::MlxExecutionPlan> {
        None
    }
    /// The routed experts' placement after startup has built the bank.
    fn expert_placement(&self) -> Option<&super::expert_placement::ExpertPlacement> {
        None
    }
    /// The limits of this executor's prefix store: how many pinned roots and how many LRU entries
    /// it holds. The worker asks once per job and plans its save points with the answer: it emits a
    /// root point only when `roots > 0`, so a job with a root boundary behaves like a job without
    /// one on an engine that keeps no roots. It emits periodic points only when `entries >= 2`,
    /// because with one entry the tip would replace a periodic entry at once. The default is a
    /// store of zero roots and one entry. An executor that enables the set overrides this method
    /// and must answer with the limits of its own store (the Qwen3.6 engine will answer two roots
    /// and six entries).
    fn prefix_limits(&self) -> StoreLimits {
        StoreLimits::default()
    }
    /// The block prefix-match spacing (default ON): `None` only on the fail-safe rollback (a
    /// set value keeps the feature on solely for `1`/`true`/`on`, trimmed and case-folded;
    /// everything else, including non-UTF-8, disables); `Some(tokens)` cuts one rolling
    /// intermediate entry per stored path at the last block boundary below the tip. Each stored
    /// path holds at most one rolling Block. A protected store may retain a recent longer Block
    /// instead of a shorter replacement. The default reads `ENGINESRV_PREFIX_BLOCK_MATCH` and
    /// `ENGINESRV_PREFIX_BLOCK_TOKENS` once; a test fake overrides this to control the spacing
    /// without the environment.
    fn prefix_block_spacing(&self) -> Option<usize> {
        crate::family::session::prefix_block_spacing()
    }
    /// Saves the session as an entry of the engine's prefix store. `kind` names the save point:
    /// `SaveKind::Root` a pinned root at the end of the first system message, `SaveKind::Periodic`
    /// a periodic checkpoint of a long prompt, and `SaveKind::Tip` the last message boundary. `key`
    /// holds the committed prompt tokens of the session. Returns `Ok(false)` when this executor
    /// cannot save or when the memory plan denies the charge. It never changes the session.
    /// Executors that keep no prefix state use this default.
    fn save_prefix(&mut self, _st: &Self::Session, _key: &[u32], _kind: SaveKind) -> Result<bool> {
        Ok(false)
    }
    /// Replaces the reset session with the saved prefix state. It does so only when the saved key is
    /// shorter than `prompt`, equals its first tokens, and comes from the current numeric path.
    /// Returns the number of prompt tokens that the state covers, or 0 with the session unchanged.
    fn restore_prefix(&mut self, _st: &mut Self::Session, _prompt: &[u32]) -> Result<usize> {
        Ok(0)
    }
    /// [`Self::restore_prefix`] for a job whose last message boundary is `tip` prompt tokens. The
    /// worker calls this one. An executor whose restore depends on where the worker cuts the
    /// prompt overrides it; the default does not read the tip.
    fn restore_prefix_under(
        &mut self,
        st: &mut Self::Session,
        prompt: &[u32],
        _tip: usize,
    ) -> Result<usize> {
        self.restore_prefix(st, prompt)
    }
    /// The prefix-store report of this executor, which the worker takes once per job after the
    /// prefill and carries in `JobUsage`: the entries the store holds, the charged bytes of their
    /// snapshots, the kind of the entry the job restored, and the prompt tokens shared with a
    /// stored key beyond the restored prefix. Executors that keep no prefix report use this
    /// default: `None`, and the request log then omits its four `prefix_*` keys even for a request
    /// with a boundary (no false zeros). The Flash-Next executor keeps this default: it reports
    /// nothing.
    fn prefix_report(&self) -> Option<PrefixReport> {
        None
    }
    /// The expert-bank admit-path counters of this executor, which the worker snapshots
    /// before and after the prefill and logs as a per-request delta in `JobUsage`: admit
    /// passes and misses, the miss and rows-per-expert route histograms, and the upload
    /// bytes and wall time of the admit copy spans. Decode traffic never enters them.
    /// Executors without an expert bank use this default: `None`, and the request log
    /// omits the `prefill_bank_*` keys (no false zeros).
    fn prefill_bank_report(&self) -> Option<PrefillBankReport> {
        None
    }
    /// Evicts ONE of the engine's evictable prefix entries per the incremental denial order: dead
    /// tips first, then redundant single-chain interior anchors, then anchor leaves, then branch
    /// points (a shared header), oldest first within each tier; `protect` (the denied job's
    /// restored anchor) goes last; roots stay pinned. Returns the number of bytes the eviction
    /// released from the session budget, or `None` when no evictable entry remains - the caller's
    /// eviction loop ends there and the denial stands for classification. Executors that keep no
    /// prefix state use this default: nothing to free.
    // shortcut: the freed-byte count is not logged anywhere - the sched worker has no log
    // channel; log it where the server's request log lives when a failing test needs it.
    fn evict_one_prefix_entry(&mut self, _protect: Option<u64>) -> Option<u64> {
        None
    }
    /// The entry the denying session restored, to protect it last in
    /// [`Self::evict_one_prefix_entry`]. Executors that keep no prefix state - or callers
    /// with no session yet, like an admission rebuild - use this default: no protected
    /// lineage.
    fn restored_anchor_of(&self, _st: &Self::Session) -> Option<u64> {
        None
    }
    /// The entry a restart of `prompt` would restore first: an admission-rebuild eviction protects
    /// it, because no session exists there to carry the lineage and the rebuild must not free the
    /// very anchor its next attempt restores. Executors that keep no prefix state use this default:
    /// no candidate.
    fn restore_candidate(&self, _prompt: &[u32]) -> Option<u64> {
        None
    }
    /// Whether the executor has background work (a write-behind of saved prefixes) that the engine
    /// thread should advance while it waits for the next job. Executors without any use this
    /// default: nothing pending.
    fn background_pending(&self) -> bool {
        false
    }
    /// Advances the background work by one bounded step. The worker calls it between jobs, never
    /// during one.
    fn background_step(&mut self) {}
    /// Finishes the background work before the engine thread ends, within a bound.
    fn background_drain(&mut self) {}
    /// The largest prompt, in tokens, that the session budget could still admit for a fresh attempt
    /// of `st`: the pool-derived effective ceiling, not the configured context limit. The worker
    /// reports it as `PromptTooLong::limit_tokens` when an admission cannot be served even after
    /// the eviction, so a client can compact to a size that fits. The ceiling counts the pool's
    /// free bytes plus the bytes `st` itself would release on its next reset, because every attempt
    /// begins from a reset session. `None` means the executor budgets nothing and reports no
    /// ceiling.
    fn pool_prompt_ceiling(&self, _st: &Self::Session) -> Option<u64> {
        None
    }
}

/// Runs one model state through the shared decode policy. With [`Self::with_reuse`], a request
/// whose prompt begins like an earlier one starts from a kept state instead of position 0.
///
/// # Prompt reuse
///
/// A reused request has the bits of a cold start: the same request, with the same boundary
/// mark, served by this engine from an empty store. Chunks of more than a few rows give a row
/// bits that depend on the chunk it sits in, so the two runs must cut the prompt alike. They do,
/// because a state is kept only at a multiple of the worker's piece and put back only below the
/// tip of the request:
///
/// - a cold start cuts every piece at a multiple of the piece below its tip, cuts once at the
///   tip and starts its pieces again there;
/// - a kept key is a multiple of the piece, so a cold start of any prompt whose tip lies above
///   it ends a chunk there, and the pieces after a restore fall where the cold ones fall;
/// - a key at or above the tip is not taken: the cold start would have cut at its tip below it.
///
/// The rows a cache holds by position stay on the device, so one sequence is held at a time.
/// The model counts a generation up when a sequence starts at position 0, and a restore drops
/// every longer entry before a row is written over.
pub(crate) struct SerialDecode<S> {
    step: S,
    eos: Vec<u32>,
    sampler_vocab: usize,
    context_limit: usize,
    reuse: Option<Reuse>,
}

/// The kept states of one executor: each is tagged with the generation of the model that
/// kept it and with the piece.
struct Reuse {
    store: PrefixStore<Kept, (u64, usize)>,
    piece: usize,
    /// The host bytes the kept states may take together.
    budget: u64,
    /// Asks the host whether one more copy of this many bytes fits now.
    admit: HostAdmit,
}

/// The host admission of one kept state: a `MemoryPlan` refusal skips the save.
pub(crate) type HostAdmit = fn(usize) -> Result<()>;

/// The fewest entries with which the worker plans a cut below the tip. One conversation uses
/// one: a later cut of a prompt replaces the earlier one.
const KEPT_LIMITS: StoreLimits = StoreLimits {
    roots: 0,
    entries: 2,
};

impl<S: TokenStepper> SerialDecode<S> {
    pub(crate) fn new(step: S, eos: Vec<u32>, sampler_vocab: usize, context_limit: usize) -> Self {
        Self {
            step,
            eos,
            sampler_vocab,
            context_limit,
            reuse: None,
        }
    }

    /// Turns prompt reuse on for a model that keeps its state. `piece` is the prompt piece of
    /// the worker that drives this executor, `budget` the host bytes the kept states may take
    /// together (`None` when the host's memory is unknown: each save then asks `admit` alone),
    /// and `admit` the host admission of one more state.
    pub(crate) fn with_reuse(
        mut self,
        piece: usize,
        budget: Option<u64>,
        admit: HostAdmit,
    ) -> Self {
        if piece > 0 && self.step.kept_bytes().is_some() {
            self.reuse = Some(Reuse {
                store: PrefixStore::with_limits(KEPT_LIMITS),
                piece,
                budget: budget.unwrap_or(u64::MAX),
                admit,
            });
        }
        self
    }
}

impl<S: TokenStepper> ExecutorOps for SerialDecode<S> {
    type Session = DecodeSession;

    fn check_prompt(&self, prompt_tokens: usize) -> Result<()> {
        session::check_prompt(prompt_tokens, self.context_limit)
    }

    fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
        st.history()
    }

    fn reset(&mut self, st: &mut Self::Session) -> Result<()> {
        st.reset(&mut self.step)
    }

    fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<()> {
        st.prefill(&mut self.step, toks, self.context_limit)
    }

    fn decode(
        &mut self,
        st: &mut Self::Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        st.decode(
            &mut self.step,
            generation,
            &Limits {
                eos: &self.eos,
                sampler_vocab: self.sampler_vocab,
                context_limit: self.context_limit,
            },
        )
    }

    fn boundary(&mut self, _st: &mut Self::Session) {}

    fn prefix_limits(&self) -> StoreLimits {
        match self.reuse {
            Some(_) => KEPT_LIMITS,
            None => StoreLimits::default(),
        }
    }

    /// Keeps the state at `key` when a cold start of a longer prompt ends a chunk there. A
    /// host without the room skips the save; the request is served the same.
    fn save_prefix(&mut self, st: &Self::Session, key: &[u32], kind: SaveKind) -> Result<bool> {
        let (Some(reuse), Some(incoming)) = (self.reuse.as_mut(), self.step.kept_bytes()) else {
            return Ok(false);
        };
        if !key.len().is_multiple_of(reuse.piece) {
            return Ok(false);
        }
        let path = (self.step.kept_generation(), reuse.piece);
        // The states of an earlier generation never match again: their bytes go first.
        reuse.store.retain_path(&path);
        match (reuse.admit)(usize::try_from(incoming).unwrap_or(usize::MAX)) {
            Ok(()) => {}
            Err(EngError::MemoryPlan(_)) => return Ok(false),
            Err(other) => return Err(other),
        }
        let budget = SnapshotBudget {
            limit: reuse.budget,
            incoming,
            size: &Kept::bytes,
        };
        let step = &mut self.step;
        let fed = step.fed();
        st.save_prefix_budgeted(&mut reuse.store, key, path, fed, kind, budget, || {
            step.keep()
        })
    }

    /// Puts back the longest kept state whose key the prompt begins with below `tip`, and
    /// drops every longer one: the rest of the prompt is written over the rows those describe.
    fn restore_prefix_under(
        &mut self,
        st: &mut Self::Session,
        prompt: &[u32],
        tip: usize,
    ) -> Result<usize> {
        let Some(reuse) = self.reuse.as_mut() else {
            return Ok(0);
        };
        let path = (self.step.kept_generation(), reuse.piece);
        reuse.store.retain_path(&path);
        let below = &prompt[..tip.min(prompt.len())];
        let step = &mut self.step;
        let restored =
            st.restore_prefix(&mut reuse.store, below, path, |kept| step.put_back(kept))?;
        if restored > 0 {
            reuse.store.drop_above(restored);
        }
        Ok(restored)
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchPolicy, ExecutorOps, GenerationState, SerialDecode};
    use crate::EngError;
    use crate::family::sampler::{Sampler, SamplingParams};
    use crate::family::session::{DecodeSession, Kept, SaveKind, StoreLimits, TokenStepper};
    use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0},"merges":[]}}"#;

    fn new_decoder() -> IncrementalDecoder {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
        IncrementalDecoder::new(&tokenizer)
    }

    fn new_state() -> GenerationState {
        GenerationState::new(new_decoder())
    }

    #[test]
    fn the_helper_builds_a_state_whose_decoder_decodes() {
        let mut state = new_state();
        assert_eq!(state.decoder.push(0).expect("push token 0"), "a");
    }

    #[test]
    fn a_new_generation_state_starts_uncancelled_and_owns_its_flag() {
        let first = new_state();
        let second = new_state();
        assert!(!first.cancel.load(Ordering::SeqCst), "first starts set");
        first.cancel.store(true, Ordering::SeqCst);
        assert!(first.cancel.load(Ordering::SeqCst), "store is lost");
        assert!(
            !second.cancel.load(Ordering::SeqCst),
            "second shares the flag of first"
        );
    }

    #[test]
    fn a_new_generation_state_samples_greedily() {
        let mut state = new_state();
        for _ in 0..5 {
            let id = state.sampler.sample(&[0.0, 2.0, 1.0]).expect("sample");
            assert_eq!(id, 1);
        }
    }

    #[derive(Default)]
    struct Step(Vec<u32>);

    impl TokenStepper for Step {
        const OPERATION: &'static str = "test_serial_decode";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.0.push(token);
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            self.feed(token)?;
            assert_eq!(self.0, [0, 0, 0]);
            Ok(vec![1.0])
        }

        fn fed(&self) -> usize {
            self.0.len()
        }

        fn reset(&mut self) -> Result<(), EngError> {
            self.0.clear();
            Ok(())
        }
    }

    #[test]
    fn serial_decode_feeds_each_prompt_token_once_and_honors_the_budget() {
        for pieces in [&[&[0, 0, 0][..]][..], &[&[0][..], &[0, 0][..]][..]] {
            let mut executor = SerialDecode::new(Step::default(), vec![], 1, 4);
            let mut session = DecodeSession::default();
            for piece in pieces {
                executor.prefill(&mut session, piece).unwrap();
            }
            assert_eq!(executor.step.fed(), 2);
            let mut generation = new_state().with_max_tokens(1);
            let batch = executor.decode(&mut session, &mut generation).unwrap();
            assert_eq!(batch.tokens, [0]);
            assert_eq!(batch.text, "a");
            assert!(matches!(batch.finish, Some(DecodeFinish::Length)));
            assert_eq!(executor.history(&session), [0, 0, 0, 0]);
            executor.reset(&mut session).unwrap();
            assert_eq!(executor.step.fed(), 0);
            assert!(executor.history(&session).is_empty());
        }
    }

    struct Bare;

    impl ExecutorOps for Bare {
        type Session = Vec<u32>;
        fn history<'s>(&self, st: &'s Vec<u32>) -> &'s [u32] {
            st
        }
        fn reset(&mut self, st: &mut Vec<u32>) -> Result<(), EngError> {
            st.clear();
            Ok(())
        }
        fn prefill(&mut self, st: &mut Vec<u32>, toks: &[u32]) -> Result<(), EngError> {
            st.extend_from_slice(toks);
            Ok(())
        }
        fn decode(
            &mut self,
            _st: &mut Vec<u32>,
            _generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            Err(EngError::BadRequest(
                "bare executor does not decode".to_string(),
            ))
        }
        fn boundary(&mut self, _st: &mut Vec<u32>) {}
    }

    #[test]
    fn an_executor_without_window_support_keeps_solo_defaults() {
        let mut executor = Bare;
        assert_eq!(executor.max_seqs(), 1);
        // Nor does it opt into the batch worker's measured choices.
        assert_eq!(executor.batch_policy(), BatchPolicy::default());
        assert_eq!(executor.batch_policy().idle_pieces, 1);
        assert_eq!(executor.batch_policy().idle_trim, None);
        assert!(matches!(
            executor.new_session(),
            Err(EngError::BadRequest(_))
        ));
        let mut session = vec![1, 2, 3];
        let mut generation = new_state();
        assert!(matches!(
            executor.decode_window(&mut [&mut session], &mut [&mut generation]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(session, [1, 2, 3]);
    }

    #[test]
    fn an_executor_without_prefix_support_saves_nothing_and_restores_nothing() {
        let mut executor = Bare;
        let mut session = vec![1, 2, 3];
        assert!(
            !executor
                .save_prefix(&session, &[1, 2, 3], SaveKind::Tip)
                .expect("save")
        );
        assert_eq!(
            executor
                .restore_prefix(&mut session, &[1, 2, 3, 4])
                .expect("restore"),
            0
        );
        assert_eq!(session, vec![1, 2, 3], "the session is unchanged");
    }

    #[test]
    fn an_executor_without_prefix_support_reports_the_addendum_one_limits() {
        assert_eq!(
            Bare.prefix_limits(),
            StoreLimits {
                roots: 0,
                entries: 1
            },
            "zero roots and one entry (a literal, not the default under test)"
        );
    }

    /// The default `prefix_report` of an executor without a prefix report is `None`: its request
    /// lines omit the four `prefix_*` keys instead of carrying false zeros.
    #[test]
    fn an_executor_without_prefix_support_reports_no_prefix_report() {
        assert_eq!(
            Bare.prefix_report(),
            None,
            "no report, taken with `&self`, after any call sequence"
        );
    }

    #[test]
    fn a_generation_state_keeps_the_sampler_it_is_given() {
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let mut state = GenerationState::with_sampler(new_decoder(), Sampler::new(params, 0));
        let id = state.sampler.sample(&[0.0; 4]).expect("sample");
        assert_eq!(id, 3);
    }

    #[test]
    fn a_new_generation_state_has_no_token_limit() {
        let mut state = new_state();
        assert_eq!(state.remaining_tokens(), None, "fresh state");
        state.record_generated(3);
        assert_eq!(
            state.remaining_tokens(),
            None,
            "recording keeps it unlimited"
        );
    }

    #[test]
    fn with_max_tokens_sets_the_limit_and_keeps_the_sampler() {
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let mut state = GenerationState::with_sampler(new_decoder(), Sampler::new(params, 0))
            .with_max_tokens(5);
        assert_eq!(state.remaining_tokens(), Some(5), "limit");
        let id = state.sampler.sample(&[0.0; 4]).expect("sample");
        assert_eq!(id, 3, "the sampler given to the state is kept");
    }

    #[test]
    fn recorded_tokens_reduce_the_remaining_limit_and_saturate() {
        let mut state = new_state().with_max_tokens(5);
        state.record_generated(2);
        assert_eq!(state.remaining_tokens(), Some(3), "after two tokens");
        state.record_generated(10);
        assert_eq!(state.remaining_tokens(), Some(0), "past the limit");
    }

    #[test]
    fn the_generated_counter_saturates_instead_of_wrapping() {
        let mut state = new_state().with_max_tokens(5);
        state.record_generated(u32::MAX);
        state.record_generated(2);
        assert_eq!(state.remaining_tokens(), Some(0), "counter does not wrap");
    }

    /// A model whose bits depend on how its rows were cut into chunks, as the batched products
    /// of a device step do: a running state that folds every chunk with its length, and a row
    /// per position that holds where in its chunk the token sat. Its "logits" are a fingerprint
    /// of the running state and of the rows below the position, so a row at or above it is
    /// never read, and a stale row below it is.
    struct Chunky {
        running: u64,
        rows: Vec<u64>,
        pos: usize,
        generation: u64,
        log: Arc<std::sync::Mutex<Chunks>>,
    }

    /// What a [`Chunky`] ran: every chunk as (first position, rows, fingerprint after it), and
    /// the fingerprint behind every decoded token.
    #[derive(Default, Clone, PartialEq, Debug)]
    struct Chunks {
        chunks: Vec<(usize, usize, u64)>,
        decoded: Vec<u64>,
    }

    fn mix(a: u64, b: u64) -> u64 {
        (a ^ b).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ b
    }

    impl Chunky {
        fn new(context: usize, log: &Arc<std::sync::Mutex<Chunks>>) -> Self {
            Self {
                running: 0,
                rows: vec![0; context],
                pos: 0,
                generation: 0,
                log: Arc::clone(log),
            }
        }

        fn fingerprint(&self) -> u64 {
            self.rows[..self.pos]
                .iter()
                .fold(self.running, |h, &row| mix(h, row))
        }

        fn run(&mut self, tokens: &[u32]) {
            if self.pos == 0 {
                self.generation += 1;
            }
            let n = tokens.len() as u64;
            self.running = mix(self.running, n);
            for (at, &token) in tokens.iter().enumerate() {
                self.running = mix(self.running, u64::from(token));
                self.rows[self.pos + at] = mix(self.running, (n << 32) | at as u64);
            }
            let first = self.pos;
            self.pos += tokens.len();
            let after = self.fingerprint();
            let mut log = self.log.lock().unwrap();
            log.chunks.push((first, tokens.len(), after));
        }
    }

    impl TokenStepper for Chunky {
        const OPERATION: &'static str = "test_chunky";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.feed_rows(&[token])
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            self.run(tokens);
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            self.run(&[token]);
            let row = self.fingerprint();
            self.log.lock().unwrap().decoded.push(row);
            Ok(vec![1.0])
        }

        fn fed(&self) -> usize {
            self.pos
        }

        fn reset(&mut self) -> Result<(), EngError> {
            (self.running, self.pos) = (0, 0);
            Ok(())
        }

        fn kept_bytes(&self) -> Option<u64> {
            Some(8)
        }

        fn keep(&mut self) -> Result<Kept, EngError> {
            Ok(Kept {
                rows: self.pos,
                parts: vec![self.running.to_le_bytes().to_vec()],
            })
        }

        fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
            let bytes: [u8; 8] = kept.parts[0][..].try_into().expect("eight bytes");
            (self.running, self.pos) = (u64::from_le_bytes(bytes), kept.rows);
            Ok(())
        }

        fn kept_generation(&self) -> u64 {
            self.generation
        }
    }

    /// The prompt piece of the worker in the reuse tests, and the context of the model.
    const PIECE: usize = 256;
    const CONTEXT: usize = 8192;
    /// The tokens after the last message boundary of a test prompt.
    const TAIL: usize = 4;

    /// An engine over a [`Chunky`] on the real worker. `kept_piece` is the piece the executor
    /// keeps states at (`None` leaves reuse off); the worker's piece is [`PIECE`].
    struct Served {
        engine: crate::sched::worker::Engine,
        log: Arc<std::sync::Mutex<Chunks>>,
    }

    impl Served {
        fn start(kept_piece: Option<usize>) -> Self {
            let log = Arc::new(std::sync::Mutex::new(Chunks::default()));
            let thread_log = Arc::clone(&log);
            let engine = crate::sched::worker::spawn_engine(PIECE, move |_| {
                let step = Chunky::new(CONTEXT, &thread_log);
                let mut executor = SerialDecode::new(step, vec![], 1, CONTEXT);
                if let Some(piece) = kept_piece {
                    executor = executor.with_reuse(piece, None, |_| Ok(()));
                }
                Ok((executor, DecodeSession::default()))
            })
            .expect("the engine starts");
            Self { engine, log }
        }

        /// Serves `prompt`, whose last message ends [`TAIL`] tokens before its end, and three
        /// tokens after it. Returns what the model ran, the tokens and the reused count.
        fn request(&self, prompt: &[u32]) -> (Chunks, Vec<u32>, u32) {
            *self.log.lock().unwrap() = Chunks::default();
            let generation = new_state().with_max_tokens(3);
            let (job, mut handle) =
                crate::sched::job::Job::new(prompt.to_vec(), generation).expect("a job");
            let job = job
                .with_prefix_boundary(prompt.len() - TAIL)
                .expect("a boundary inside the prompt");
            assert!(
                self.engine.handle().try_submit(job).is_ok(),
                "the job queues"
            );
            let (mut tokens, mut reused) = (Vec::new(), None);
            while let Some(event) = handle.events.blocking_recv() {
                match event {
                    crate::sched::job::JobEvent::Batch(batch) => tokens.extend(batch.tokens),
                    crate::sched::job::JobEvent::Finished(usage) => {
                        reused = Some(usage.reused_tokens);
                    }
                    crate::sched::job::JobEvent::Error(error) => panic!("{error:?}"),
                    crate::sched::job::JobEvent::Started => {}
                }
            }
            let ran = self.log.lock().unwrap().clone();
            (ran, tokens, reused.expect("the job finished"))
        }

        /// What a cold start runs: the same request on an engine that has served nothing.
        fn cold(kept_piece: Option<usize>, prompt: &[u32]) -> (Chunks, Vec<u32>) {
            let (ran, tokens, reused) = Self::start(kept_piece).request(prompt);
            assert_eq!(reused, 0, "a cold start reuses nothing");
            (ran, tokens)
        }
    }

    fn ids(count: usize, salt: u32) -> Vec<u32> {
        (0..count as u32)
            .map(|i| (i ^ salt).wrapping_mul(2_654_435_761) >> 12)
            .collect()
    }

    /// The reused request ran exactly the chunks of the cold one above `kept`, on the same
    /// bits, and decoded from the same bits.
    fn assert_the_chunks_of_the_cold_start(warm: &Chunks, cold: &Chunks, kept: usize) {
        // A key of `kept` tokens is the state after `kept - 1` rows.
        let above: Vec<_> = cold
            .chunks
            .iter()
            .copied()
            .filter(|&(first, _, _)| first + 1 >= kept)
            .collect();
        assert_eq!(
            warm.chunks.first().map(|c| c.0),
            Some(kept - 1),
            "the first row fed"
        );
        assert_eq!(warm.chunks, above, "the chunks above the kept position");
        assert_eq!(
            warm.decoded, cold.decoded,
            "the rows behind the decoded tokens"
        );
    }

    /// The acceptance case: request 1, then request 2 = request 1, its completion and a new
    /// turn. Request 2 starts from the state kept at the last block boundary below request 1's
    /// tip, feeds only the rows above it, and runs the chunks its cold start runs.
    #[test]
    fn a_reused_request_runs_the_chunks_of_its_cold_start_with_the_same_boundary_mark() {
        let served = Served::start(Some(PIECE));
        let first = ids(2_600, 1);
        let (_, completion, reused) = served.request(&first);
        assert_eq!(reused, 0);
        let mut second = first.clone();
        second.extend(&completion);
        second.extend(ids(300, 2));
        let (warm, warm_tokens, reused) = served.request(&second);
        // The tip of request 1 is 2,596 tokens: its block spacing is 1,024, its cut 2,048.
        assert_eq!(reused, 2_048);
        let (cold, cold_tokens) = Served::cold(Some(PIECE), &second);
        assert_the_chunks_of_the_cold_start(&warm, &cold, 2_048);
        assert_eq!(warm_tokens, cold_tokens);
        // A third turn starts from the cut request 2 left, wherever it is.
        let mut third = second.clone();
        third.extend(&warm_tokens);
        third.extend(ids(1_500, 3));
        let (warm, _, reused) = served.request(&third);
        assert_eq!(reused, 2_048, "request 2 cut where request 1 had");
        let (cold, _) = Served::cold(Some(PIECE), &third);
        assert_the_chunks_of_the_cold_start(&warm, &cold, 2_048);
        let mut fourth = third.clone();
        fourth.extend(ids(40, 4));
        let (warm, _, reused) = served.request(&fourth);
        assert_eq!(reused, 4_096, "request 3 cut one block further");
        let (cold, _) = Served::cold(Some(PIECE), &fourth);
        assert_the_chunks_of_the_cold_start(&warm, &cold, 4_096);
    }

    /// The model of these tests can see a wrong cut: an executor that keeps the state at the
    /// tip itself, which is no multiple of the worker's piece, serves other bits than the cold
    /// start. This is the restore the executor refuses.
    #[test]
    fn a_state_kept_off_the_piece_lattice_would_not_have_the_bits_of_a_cold_start() {
        let served = Served::start(Some(1));
        let first = ids(2_600, 1);
        let (_, completion, _) = served.request(&first);
        let mut second = first.clone();
        second.extend(&completion);
        second.extend(ids(300, 2));
        let (warm, _, reused) = served.request(&second);
        assert_eq!(reused, 2_596, "the tip of request 1");
        let (cold, _) = Served::cold(Some(1), &second);
        assert_ne!(warm.decoded, cold.decoded);
    }

    /// A request repeated, and a second conversation that shares the head of the first, both
    /// start from the kept state and both have the chunks of their cold start.
    #[test]
    fn a_repeated_request_and_a_fork_below_the_tip_reuse_the_kept_state() {
        let served = Served::start(Some(PIECE));
        let first = ids(2_600, 1);
        served.request(&first);
        let (warm, _, reused) = served.request(&first);
        assert_eq!(reused, 2_048);
        let (cold, _) = Served::cold(Some(PIECE), &first);
        assert_the_chunks_of_the_cold_start(&warm, &cold, 2_048);
        let mut fork = first[..2_300].to_vec();
        fork.extend(ids(900, 5));
        let (warm, _, reused) = served.request(&fork);
        assert_eq!(reused, 2_048);
        let (cold, _) = Served::cold(Some(PIECE), &fork);
        assert_the_chunks_of_the_cold_start(&warm, &cold, 2_048);
    }

    /// What is not reused, each served with the bits of its cold start: a prompt whose tip is
    /// not above the kept key (the cold start cuts at its tip below the key), a prompt after
    /// another sequence wrote over the rows, and a prompt that leaves the kept one below the
    /// key.
    #[test]
    fn a_kept_state_is_refused_at_or_above_the_tip_and_after_another_sequence() {
        let first = ids(2_600, 1);
        let all_cold = |prompt: &[u32], served: &Served| {
            let (ran, tokens, reused) = served.request(prompt);
            assert_eq!(reused, 0);
            assert_eq!((ran, tokens), Served::cold(Some(PIECE), prompt));
        };
        // The generation tail spans the kept key: tip 2,046, key 2,048, prompt 2,050.
        let served = Served::start(Some(PIECE));
        served.request(&first);
        all_cold(&first[..2_050], &served);
        // Another sequence from position 0, then the first conversation again.
        let served = Served::start(Some(PIECE));
        served.request(&first);
        served.request(&ids(40, 9));
        all_cold(&first, &served);
        // A prompt that differs below the key.
        let served = Served::start(Some(PIECE));
        served.request(&first);
        let mut other = first.clone();
        other[2_000] ^= 1;
        all_cold(&other, &served);
    }

    /// Without `with_reuse`, and for a model that keeps no state, the executor is the one it
    /// was: the default limits, nothing saved, nothing restored.
    #[test]
    fn reuse_is_off_unless_asked_for_and_the_model_keeps_its_state() {
        let off = SerialDecode::new(
            Chunky::new(8, &Arc::new(std::sync::Mutex::new(Chunks::default()))),
            vec![],
            1,
            8,
        );
        assert_eq!(off.prefix_limits(), StoreLimits::default());
        let keeps_none =
            SerialDecode::new(Step::default(), vec![], 1, 4).with_reuse(PIECE, None, |_| Ok(()));
        assert_eq!(keeps_none.prefix_limits(), StoreLimits::default());
        let served = Served::start(None);
        let first = ids(2_600, 1);
        served.request(&first);
        let (_, _, reused) = served.request(&first);
        assert_eq!(reused, 0);
    }
}
