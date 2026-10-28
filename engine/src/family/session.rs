//! The decode session shared by the resident executors (decision 0033; packet Q36-F1).
//!
//! The Flash-Next executor and the Q36 executor use one `DecodeSession`. It owns the token side of
//! a session: the committed tokens, the pending token, and the finish flag. The model side stays
//! with each executor behind `TokenStepper`.
//!
//! Call order: `reset`, `prefill` for each chunk, then `decode` until a finish. A prompt feeds
//! every token except the last one through `TokenStepper::feed`. The last token stays pending.
//! Each `decode` feeds the pending token through `TokenStepper::next_token`, which by default reads
//! `TokenStepper::logits` and draws the next token from the job's sampler, and keeps that token
//! pending. One token leaves per call.
//!
//! The finish rules are pure functions in this file. The session owns EOS, the output limit, the
//! context limit, and cancellation (decision 0030, D7).
//!
//! Prefix reuse (decision 0036, addendum 2): `DecodeSession::save_prefix` and
//! `DecodeSession::restore_prefix` are the one authority for the reuse rules of every family. An
//! engine owns one `PrefixStore`: pinned roots beside an LRU list of entries, each entry tagged
//! with the numeric path that built it and with its `SaveKind`. The rules decide when to save,
//! which entry a restore takes (the longest stored key that the prompt extends under the same
//! numeric path), and how the token side changes. The closures only copy or share model state:
//! the store never inspects a saved state, and the byte accounting of the entries stays with the
//! executor.

use crate::{
    EngError,
    family::{executor::GenerationState, sampler::Sampler},
    tokenizer::{DecodeBatch, DecodeFinish},
};
use std::{cmp::Reverse, sync::atomic::Ordering};

type Result<T> = std::result::Result<T, EngError>;

/// The model side of one session. An executor builds one value per call from its decoder and its
/// native state. The session never holds it, so no native value leaves the executor.
pub(crate) trait TokenStepper {
    /// The `operation` of the `Backend` error for logits that are narrower than the sampler vocabulary.
    const OPERATION: &'static str;
    /// Feeds one prompt token. No head runs and nothing is read back.
    fn feed(&mut self, token: u32) -> Result<()>;
    /// Feeds `tokens` in order as prompt rows. The default feeds them one by one. A model that
    /// feeds a prompt in chunks (the Flash-Next executor, packet H8b) overrides it.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        tokens.iter().try_for_each(|&token| self.feed(token))
    }
    /// Feeds `token`, runs the head, and returns the full logits row of the next token.
    fn logits(&mut self, token: u32) -> Result<Vec<f32>>;
    /// Feeds `token` and returns the next token. The default reads the full logits row through
    /// `logits`, hands the first `sampler_vocab` values to `sampler`, and returns its draw. A
    /// model that draws on its own device overrides it. A row narrower than `sampler_vocab` is a
    /// `Backend` error of `OPERATION`.
    ///
    /// The fed rule: `DecodeSession::decode` clears the pending token after an `Err` exactly when
    /// `fed` changed during this call. So `fed` must advance when, and only when, the model state
    /// consumed the token. An override owns the same contract. Two paths differ from the code
    /// before this hook existed:
    /// - A CUDA tiered-mode failure in `end_tiered_token` after the step committed
    ///   (`qwen36/cuda/step.rs`, `run_token_inner`). `fed` advanced, so the pending token is now
    ///   cleared where it stayed before. The state is poisoned. The worker resets the session on
    ///   any non-`Backend` error and fails the engine on a `Backend` error, so the difference
    ///   is harmless.
    /// - The unreachable error "the logits step returned no logits" after a commit. It follows the
    ///   same rule.
    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32> {
        let logits = self.logits(token)?;
        sample_logits(Self::OPERATION, &logits, sampler, sampler_vocab)
    }
    /// The number of positions fed so far.
    fn fed(&self) -> usize;
    /// Clears the model state of the session.
    fn reset(&mut self) -> Result<()>;
}

/// Samples only the tokenizer-visible vocabulary, with the same width check for solo and rows.
pub(crate) fn sample_logits(
    operation: &'static str,
    logits: &[f32],
    sampler: &mut Sampler,
    sampler_vocab: usize,
) -> Result<u32> {
    let visible = logits
        .get(..sampler_vocab)
        .ok_or_else(|| EngError::Backend {
            operation,
            message: "the logits are narrower than the sampler vocabulary".to_string(),
        })?;
    sampler.sample(visible)
}

/// The limits of one engine that every `decode` call applies.
#[derive(Clone, Copy)]
pub(crate) struct Limits<'a> {
    /// Every id that ends generation (`TokenizerBundle::eos_token_ids`).
    pub(crate) eos: &'a [u32],
    /// The number of logits that the sampler may choose from (`TokenizerBundle::vocab_size`). The
    /// model head can be wider than the tokenizer, and the extra ids have no text.
    pub(crate) sampler_vocab: usize,
    /// The number of positions that the model accepts.
    pub(crate) context_limit: usize,
}

/// The token side of one session.
#[derive(Debug, Default, Clone)]
pub(crate) struct DecodeSession {
    /// Every committed token: the prompt and the sampled tokens, including a final EOS token.
    history: Vec<u32>,
    /// The number of leading `history` tokens that came from `prefill` (the held prompt).
    prompt_len: usize,
    /// The token that the next step feeds. `None` before a prompt and after EOS.
    pending: Option<u32>,
    /// True after a finish. A new job starts with `reset`.
    finished: bool,
    /// The store entry this session restored. Saves protect and replace only this lineage.
    restored_prefix: Option<u64>,
}

/// The shared policy decision before a solo or row-window model step.
pub(crate) enum DecodePreparation {
    /// Feed this pending token once.
    Token(u32),
    /// No model step is needed. The finish, text tail, and cancellation reset are complete.
    Finished(DecodeBatch),
}

impl DecodeSession {
    /// The committed tokens.
    pub(crate) fn history(&self) -> &[u32] {
        &self.history
    }

    /// The number of leading history tokens that the prompt supplied.
    pub(crate) fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// The token that the next step feeds, or `None`.
    pub(crate) fn pending(&self) -> Option<u32> {
        self.pending
    }

    /// True after a finish and before the next `reset`.
    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    /// Clears the model state, then the tokens. If the model reset fails, the tokens stay.
    pub(crate) fn reset(&mut self, step: &mut impl TokenStepper) -> Result<()> {
        step.reset()?;
        self.history.clear();
        self.prompt_len = 0;
        self.pending = None;
        self.finished = false;
        self.restored_prefix = None;
        Ok(())
    }

    /// The entry this session restored. The eviction callers pass it as the denied job's
    /// protected lineage (review round 4, MF2): the store-global marker names whichever
    /// session restored last, which is another conversation in an interleaved batch.
    pub(crate) fn restored_anchor(&self) -> Option<u64> {
        self.restored_prefix
    }

    /// Consumes one non-empty prompt chunk. It rejects a prompt of `context_limit` tokens or more
    /// before it feeds a token. The last token of the chunk stays pending.
    pub(crate) fn prefill(
        &mut self,
        step: &mut impl TokenStepper,
        toks: &[u32],
        context_limit: usize,
    ) -> Result<()> {
        let Some((&last, head)) = toks.split_last() else {
            return Err(EngError::BadRequest("empty prefill chunk".to_string()));
        };
        if self.finished {
            return Err(EngError::BadRequest("prefill after finish".to_string()));
        }
        let total = self
            .history
            .len()
            .checked_add(toks.len())
            .ok_or_else(|| EngError::BadRequest("prompt length overflows".to_string()))?;
        check_prompt(total, context_limit)?;
        // The token that was last in the earlier chunk is not last any more.
        let mut rows = Vec::with_capacity(toks.len());
        rows.extend(self.pending.take());
        rows.extend_from_slice(head);
        if !rows.is_empty() {
            step.feed_rows(&rows)?;
        }
        self.history.extend_from_slice(toks);
        self.prompt_len = self.history.len();
        self.pending = Some(last);
        Ok(())
    }

    /// Commits one batch of tokens: feeds the pending token through `TokenStepper::next_token`,
    /// which draws the next token from the job's sampler, and keeps it pending. One token leaves
    /// per call. A job that was cancelled, has no tokens left, or hits a limit ends with a finish
    /// reason. If `next_token` fails, the pending token stays only when the model did not feed it.
    pub(crate) fn decode<S: TokenStepper>(
        &mut self,
        step: &mut S,
        generation: &mut GenerationState,
        limits: &Limits<'_>,
    ) -> Result<DecodeBatch> {
        let pending = match self.prepare_decode(step, generation)? {
            DecodePreparation::Token(pending) => pending,
            DecodePreparation::Finished(batch) => return Ok(batch),
        };
        // The token is consumed exactly when the model fed it. A failure before the feed (a failing
        // `logits`) keeps the token pending; a failure after it (a narrow row, an invalid row)
        // leaves no pending token, because a retry would feed the token twice.
        let fed_before = step.fed();
        let token = match step.next_token(pending, &mut generation.sampler, limits.sampler_vocab) {
            Ok(token) => token,
            Err(error) => {
                self.record_feed(fed_before, step.fed());
                return Err(error);
            }
        };
        self.finish_decode_token(step, generation, limits, token)
    }

    /// Validates the token-side decode state without mutation or model calls.
    pub(crate) fn checked_pending(&self) -> Result<u32> {
        if self.finished {
            return Err(EngError::BadRequest("decode after finish".to_string()));
        }
        self.pending
            .ok_or_else(|| EngError::BadRequest("decode before prefill".to_string()))
    }

    /// Applies the same pre-step policy to solo and window decoding, including cancellation
    /// precedence and decoder tails. A ready row retains its pending token until the feed.
    pub(crate) fn prepare_decode<S: TokenStepper>(
        &mut self,
        step: &mut S,
        generation: &mut GenerationState,
    ) -> Result<DecodePreparation> {
        let pending = self.checked_pending()?;
        if generation.cancel.load(Ordering::SeqCst) {
            return self
                .end(
                    step,
                    generation,
                    Vec::new(),
                    String::new(),
                    DecodeFinish::Cancelled,
                )
                .map(DecodePreparation::Finished);
        }
        if let Some(reason) = before_step(generation.remaining_tokens()) {
            return self
                .end(step, generation, Vec::new(), String::new(), reason)
                .map(DecodePreparation::Finished);
        }
        Ok(DecodePreparation::Token(pending))
    }

    /// Records model consumption before sampling can fail. An unfed pending token stays retryable.
    pub(crate) fn record_feed(&mut self, before: usize, after: usize) {
        if before != after {
            self.pending = None;
        }
    }

    /// Publishes one sampled token through the same history, EOS, limit, and text policy as solo
    /// decoding. The caller has already consumed the model input and validated the sampled token.
    pub(crate) fn finish_decode_token<S: TokenStepper>(
        &mut self,
        step: &mut S,
        generation: &mut GenerationState,
        limits: &Limits<'_>,
        token: u32,
    ) -> Result<DecodeBatch> {
        self.pending = None;
        self.history.push(token);
        let mut text = String::new();
        if !limits.eos.contains(&token) {
            match generation.decoder.push(token) {
                Ok(piece) => text = piece,
                Err(error) => {
                    self.reset(step)?;
                    return Err(error);
                }
            }
            generation.record_generated(1);
            self.pending = Some(token);
        }
        match after_token(
            limits.eos,
            token,
            generation.remaining_tokens(),
            step.fed(),
            limits.context_limit,
        ) {
            Some(reason) => self.end(step, generation, vec![token], text, reason),
            None => Ok(DecodeBatch {
                tokens: vec![token],
                text,
                finish: None,
            }),
        }
    }

    /// Ends the job: adds the decoder tail, and either resets a cancelled session or marks the
    /// session finished. A failed tail resets the session and returns the error (decision D8).
    fn end(
        &mut self,
        step: &mut impl TokenStepper,
        generation: &mut GenerationState,
        tokens: Vec<u32>,
        mut text: String,
        reason: DecodeFinish,
    ) -> Result<DecodeBatch> {
        match generation.decoder.finish() {
            Ok(tail) => text.push_str(&tail),
            Err(error) => {
                self.reset(step)?;
                return Err(error);
            }
        }
        if reason == DecodeFinish::Cancelled {
            self.reset(step)?;
        } else {
            self.finished = true;
        }
        Ok(DecodeBatch {
            tokens,
            text,
            finish: Some(reason),
        })
    }

    /// Saves the model state in the store as `kind`. The state must be the state after the held
    /// prompt, which only `prefill` produces. The call needs a pending token, an unfinished
    /// session, a `key` that equals the held prompt exactly, and a model that has fed every key
    /// token except the pending one (`fed + 1 == key.len()`, where `fed` is `TokenStepper::fed`).
    /// Any other session, including one that decoded a token or one whose model is not at the
    /// position of the key, returns `BadRequest` and keeps the store unchanged.
    ///
    /// A stored entry with the same key and the same path is a use: it becomes the most recently
    /// used entry of its list and `snapshot` does not run. A `Root` save whose key equals a
    /// stored LRU entry promotes that entry to a pinned root in the same way (clarification 1,
    /// item 3). An entry with the same key and another path is dropped. A `Root` save fills the
    /// pinned root list, the other kinds the LRU list, each bounded by `StoreLimits`. A
    /// `MemoryPlan` error from `snapshot` drops one eligible victim (the least recently used LRU
    /// entry that this session did not restore and the retention policy does not protect;
    /// the least recently used root other than this
    /// session's restored root only for a `Root` save) before the next `snapshot` call, until
    /// the save succeeds or no victim is left, which returns `Ok(false)`. Any other error propagates. The
    /// entries dropped before either end stay dropped. A tip whose key extends the key of the
    /// Tip entry that this session restored drops that entry after the save succeeds, also
    /// when the tip key repeats a stored key. In a protected store, a full Tip save drops its
    /// restored ancestor Tip before any other room victim, preserving a newly saved Block.
    /// Legacy stores drop that ancestor before the snapshot only when no other entry can go
    /// (clarification 1, item 1). The session never changes.
    pub(crate) fn save_prefix<S, P: PartialEq>(
        &self,
        store: &mut PrefixStore<S, P>,
        key: &[u32],
        path: P,
        fed: usize,
        kind: SaveKind,
        mut snapshot: impl FnMut() -> Result<S>,
    ) -> Result<bool> {
        let pending = self.prefix_save_pending(key, fed)?;
        // The store keeps the last restore for reports. Only this save uses our own lineage,
        // including when a denied snapshot evicts entries or returns an error.
        let observed = std::mem::replace(&mut store.restored, self.restored_prefix);
        let result = store.save(key, pending, path, kind, &mut snapshot);
        store.restored = observed;
        result
    }

    /// Freezes the validated token-side save point while device state is staged separately.
    pub(crate) fn prepare_prefix_save(&self, key: &[u32], fed: usize) -> Result<Self> {
        self.prefix_save_pending(key, fed)?;
        Ok(self.clone())
    }

    fn prefix_save_pending(&self, key: &[u32], fed: usize) -> Result<u32> {
        let Some(pending) = self.pending else {
            return Err(EngError::BadRequest(
                "a prefix save needs a pending token".to_string(),
            ));
        };
        if self.finished || self.history != key || self.prompt_len != key.len() {
            return Err(EngError::BadRequest(
                "the prefix key differs from the held prompt".to_string(),
            ));
        }
        if fed.checked_add(1) != Some(key.len()) {
            return Err(EngError::BadRequest(format!(
                "the model has fed {fed} positions; a prefix of {} tokens needs {}",
                key.len(),
                key.len().saturating_sub(1)
            )));
        }
        Ok(pending)
    }

    /// Replaces the model state and the tokens of this session with the best stored entry. The
    /// candidates are every stored entry, root or LRU entry, whose `path` equals `path`, whose
    /// key is shorter than `prompt`, and whose key is a prefix of `prompt`; the longest key is
    /// tried first. When no candidate exists, nothing changes and the call returns 0. The caller
    /// resets the session first and prefills `prompt[covered..]` afterwards.
    ///
    /// `restore` shares or copies the saved state into the model state. A `MemoryPlan` error from
    /// it evicts ONE other entry in the radix order (the failed candidate protected - unless it
    /// is the only entry, when the last resort spends it and it counts as a drop) and retries
    /// the SAME candidate once; a second `MemoryPlan` error drops that entry and the next
    /// candidate is tried (`restore` must leave the model state unchanged when it returns an
    /// error); any other error propagates and keeps the entries. Per restore call this spends at
    /// most one victim per candidate tried - the candidates list is fixed on entry, so a
    /// persistent denial can spend up to K entries in radix order, and shallower candidates of
    /// the same prompt are NOT protected. On
    /// success the entry becomes the most recently used entry of its list and this session's
    /// protected lineage for saves. The store records the restore for reports. The history is the
    /// key, the pending token is the saved one, and the session is not finished. The call returns
    /// the number of prompt tokens that the state covers.
    pub(crate) fn restore_prefix<S, P: PartialEq>(
        &mut self,
        store: &mut PrefixStore<S, P>,
        prompt: &[u32],
        path: P,
        mut restore: impl FnMut(&S) -> Result<()>,
    ) -> Result<usize> {
        store.begin_request();
        store.restored = None;
        store.restored_kind = None;
        // The shadow observation of this lookup (KV-v1 slice 1), taken over the store exactly
        // as the lookup sees it, before the restore moves an entry or drops a candidate. It
        // changes nothing: the walk only reads keys and paths.
        if store.shadow_enabled {
            store.shadow = Some(store.block_lcp_shadow(prompt, &path));
        }
        for (id, is_root, _len) in store.candidates(prompt, &path) {
            if let Some((key, pending)) = store.restore_entry(id, is_root, &mut restore)? {
                if store.protect_recent_longer {
                    for stored in store.roots.iter_mut().chain(&mut store.lru) {
                        if stored.saved.path == path && key.starts_with(&stored.saved.key) {
                            stored.requests_since_use = 0;
                        }
                    }
                }
                self.history.clone_from(&key);
                self.prompt_len = key.len();
                self.pending = Some(pending);
                self.finished = false; // a restored session is not finished
                self.restored_prefix = Some(id);
                return Ok(key.len());
            }
        }
        Ok(0)
    }
}

impl<S, P: PartialEq> PrefixStore<S, P> {
    /// The shadow block-LCP of one lookup (KV-v1 slice 1, shadow only): the longest token
    /// run that any same-path entry shares with `prompt`, floored to the 128- and 256-token
    /// block granularities, beside the greatest checkpoint the exact-key contract could use
    /// (the longest same-path key that is a full prefix of `prompt` and strictly shorter than
    /// it, the `q < prompt_len` rule). The unfloored shared length is the divergence offset.
    ///
    /// Cost: one early-exit walk per held entry, the sum of `min(key length, prompt length)`
    /// token comparisons over the entries — never a re-walk per block. The report carries
    /// the exact spend in `token_comparisons`, so a report can verify the bound.
    pub(crate) fn block_lcp_shadow(&self, prompt: &[u32], path: &P) -> ShadowLcp {
        let mut same_path = false;
        let mut best_lcp = 0usize;
        let mut usable = 0usize;
        let mut not_shorter = false;
        let mut comparisons = 0u64;
        for stored in self.roots.iter().chain(&self.lru) {
            if &stored.saved.path != path {
                continue;
            }
            same_path = true;
            let key = &stored.saved.key;
            let mut shared = 0usize;
            for (key_token, prompt_token) in key.iter().zip(prompt) {
                comparisons += 1;
                if key_token != prompt_token {
                    break;
                }
                shared += 1;
            }
            best_lcp = best_lcp.max(shared);
            // A full-prefix key is a checkpoint the exact-key contract could take, but only
            // a strictly shorter one passes the q < prompt_len rule of the restore.
            if shared == key.len() {
                if key.len() < prompt.len() {
                    usable = usable.max(key.len());
                } else {
                    not_shorter = true;
                }
            }
        }
        let reason = if self.roots.is_empty() && self.lru.is_empty() {
            ShadowReason::NoTokenMatch
        } else if !same_path {
            ShadowReason::PathMismatch
        } else if usable > 0 {
            ShadowReason::SameAsActual
        } else if not_shorter {
            ShadowReason::QNotBelowPrompt
        } else if best_lcp > 0 {
            ShadowReason::NoUsableCheckpoint
        } else {
            ShadowReason::NoTokenMatch
        };
        ShadowLcp {
            block_lcp_128: u32::try_from(best_lcp / 128 * 128).unwrap_or(u32::MAX),
            block_lcp_256: u32::try_from(best_lcp / 256 * 256).unwrap_or(u32::MAX),
            usable_checkpoint_tokens: u32::try_from(usable).unwrap_or(u32::MAX),
            divergence_offset: u32::try_from(best_lcp).unwrap_or(u32::MAX),
            reason,
            token_comparisons: comparisons,
        }
    }
}

/// What kind of save produced a stored entry (decision 0036, addendum 2): a pinned root, a
/// periodic checkpoint of a long prompt, or the tip of the last prompt. The block cut (KV-v1
/// slice 2) is a fourth kind: the one rolling intermediate entry per lineage that the block
/// prefix-match flag captures at the last block boundary below the tip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SaveKind {
    /// The state at the end of the first system message of a prompt.
    Root,
    /// A periodic checkpoint of a long prompt.
    Periodic,
    /// The state at the last message boundary of a prompt.
    Tip,
    /// The rolling block cut of a lineage (KV-v1 slice 2): an LRU entry like a periodic one,
    /// except that a no-shorter `Block` save on the same path replaces the previous `Block`
    /// entry, so one lineage holds at most one cut at a time.
    Block,
}

/// The limits of a `PrefixStore`: how many pinned roots and how many LRU entries it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoreLimits {
    /// The number of pinned roots (`SaveKind::Root` entries).
    pub(crate) roots: usize,
    /// The number of LRU entries (`SaveKind::Periodic` and `SaveKind::Tip`).
    pub(crate) entries: usize,
}

impl Default for StoreLimits {
    /// The addendum 1 limits: zero roots and one entry.
    fn default() -> Self {
        Self {
            roots: 0,
            entries: 1,
        }
    }
}

/// Whether the shadow block-LCP observation runs (KV-v1 slice 1): off unless
/// `ENGINESRV_SHADOW_BLOCK_LCP` is `1`, `true`, or `on`. The shadow is a read-only
/// observation for the request log; the flag never changes restore or save behavior.
pub(crate) fn shadow_block_lcp_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var_os("ENGINESRV_SHADOW_BLOCK_LCP");
        matches!(
            enabled.as_deref().and_then(std::ffi::OsStr::to_str),
            Some("1" | "true" | "on")
        )
    })
}

/// The block prefix-match spacing (KV-v1 slice 2; default ON since slice 3): `None`
/// only on the explicit rollback — a set `ENGINESRV_PREFIX_BLOCK_MATCH` keeps the
/// feature on solely for `1`/`true`/`on` (trimmed, case-folded), while every other
/// value, including empty, garbage, and non-UTF-8, disables it (the `env_allows`
/// fail-safe rule). An unset variable keeps the default ON.
///
/// The save planner targets
/// the last spacing boundary strictly before the tip. `save_points` moves that
/// target to the next chunk end relative to its cursor. The result must stay before
/// the tip, and a Periodic save takes precedence at a shared position.
///
/// The store keeps at most one Block per stored path. Q36's engine-wide path thus
/// permits at most one Block for the engine, not one for every lineage.
/// Recent-longer protection can retain a longer Block through one unrelated request
/// instead of replacing it with a shorter cut. Other stores keep rolling replacement.
///
/// `ENGINESRV_PREFIX_BLOCK_TOKENS` sets the spacing
/// (default 4,096, the simulated hybrid sweet spot; a value below 1 falls back to
/// the default). The restore contract is unchanged.
pub(crate) fn prefix_block_spacing() -> Option<usize> {
    static SPACING: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *SPACING.get_or_init(|| {
        let enabled = match std::env::var_os("ENGINESRV_PREFIX_BLOCK_MATCH") {
            // Only a genuinely absent variable keeps the new default; a set
            // value that is not UTF-8 is a mis-set rollback and disables.
            None => true,
            Some(value) => value.to_str().is_some_and(|text| {
                let folded = text.trim().to_ascii_lowercase();
                folded == "1" || folded == "true" || folded == "on"
            }),
        };
        if !enabled {
            return None;
        }
        let tokens = std::env::var_os("ENGINESRV_PREFIX_BLOCK_TOKENS")
            .and_then(|value| value.to_str().and_then(|text| text.parse::<usize>().ok()))
            .filter(|&tokens| tokens >= 1)
            .unwrap_or(4_096);
        Some(tokens)
    })
}

/// Why a shadow block-LCP observation stands where it does (KV-v1 slice 1): the class of
/// the gap between what a block-granular restore could use and what the exact-key contract
/// holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShadowReason {
    /// No same-path entry shares even one token with the prompt; an empty store included.
    NoTokenMatch,
    /// Same-path entries share tokens, but none of their keys is a full prefix of the
    /// prompt: the exact-key contract restores nothing, and only a block-granular restore
    /// could use the shared run.
    NoUsableCheckpoint,
    /// The store holds entries, but none of them was built on the numeric path of this
    /// lookup.
    PathMismatch,
    /// A same-path key is a full prefix of the prompt, but none is strictly shorter, so the
    /// `q < prompt_len` rule of the restore contract keeps every one of them out.
    QNotBelowPrompt,
    /// A strictly shorter full-prefix key exists, so the exact-key contract could take the
    /// same checkpoint the shadow reports; `reused_tokens` beside the report shows whether
    /// the restore realized it.
    SameAsActual,
}

impl ShadowReason {
    /// The name of the reason in a request-log line.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NoTokenMatch => "no-token-match",
            Self::NoUsableCheckpoint => "no-usable-checkpoint",
            Self::PathMismatch => "path-mismatch",
            Self::QNotBelowPrompt => "q-not-below-prompt",
            Self::SameAsActual => "same-as-actual",
        }
    }
}

/// The shadow block-LCP observation of one lookup (KV-v1 slice 1, shadow only): what a
/// block-granular restore could have used, beside what the exact-key contract holds. The
/// observation changes no behavior; it rides the prefix report to the request log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShadowLcp {
    /// The longest token run any same-path entry shares with the prompt, floored to a whole
    /// number of 128-token blocks.
    pub(crate) block_lcp_128: u32,
    /// The same run floored to a whole number of 256-token blocks.
    pub(crate) block_lcp_256: u32,
    /// The greatest checkpoint the exact-key contract could use at or before the shared run:
    /// the longest same-path key that is a full prefix of the prompt and strictly shorter
    /// than it (the `q < prompt_len` rule, with `q = key.len()`).
    pub(crate) usable_checkpoint_tokens: u32,
    /// Where the best entry's key diverges from the prompt: the shared run before flooring.
    pub(crate) divergence_offset: u32,
    /// The class of the observation.
    pub(crate) reason: ShadowReason,
    /// The token comparisons the walk spent, so a report can verify the cost bound: one
    /// early-exit walk per held entry, the sum of `min(key length, prompt length)` over the
    /// entries, never a re-walk per block.
    pub(crate) token_comparisons: u64,
}

impl Default for ShadowLcp {
    /// The observation of a lookup that shared nothing: all zero with `NoTokenMatch`.
    fn default() -> Self {
        Self {
            block_lcp_128: 0,
            block_lcp_256: 0,
            usable_checkpoint_tokens: 0,
            divergence_offset: 0,
            reason: ShadowReason::NoTokenMatch,
            token_comparisons: 0,
        }
    }
}

/// The prefix store of an engine (decision 0036, addendum 2): up to `limits.roots` pinned roots
/// and up to `limits.entries` entries in an LRU list. `S` is the model state that the executor
/// saves, and `P` names the numeric path that built it. A state restores only under the path that
/// built it, because a batched prefill matches one-token steps only within a tolerance (decision
/// 0036, item 7). The store holds each entry as its own value: it never clones, copies, inspects,
/// or sizes an `S`, and the byte accounting of the entries stays with the executor. The executor
/// owns the store and declares it before its decoder, so the saved arrays drop before the decoder
/// starts its teardown.
pub(crate) struct PrefixStore<S, P> {
    /// The pinned roots, from the least to the most recently used.
    roots: Vec<Stored<S, P>>,
    /// The LRU entries, from the least to the most recently used.
    lru: Vec<Stored<S, P>>,
    /// The limits of the store.
    limits: StoreLimits,
    /// CUDA pinned stores opt in; other executors keep the legacy eviction rules.
    protect_recent_longer: bool,
    /// The id of the next stored entry. Ids count up and are never reused.
    next_id: u64,
    /// The last restore observation, temporarily replaced by the saving session's lineage.
    restored: Option<u64>,
    /// The kind of the entry that the last `restore_prefix` call restored. It is kept beside
    /// `restored`, because the lineage rule may drop the restored entry before a report reads
    /// the id back (packet SERVE-2-c item 3).
    restored_kind: Option<SaveKind>,
    /// Whether the shadow block-LCP observation runs (KV-v1 slice 1). Off by default; the
    /// flag never changes restore or save behavior.
    shadow_enabled: bool,
    /// The shadow observation of the last `restore_prefix` call on an enabled store, stashed
    /// over the store exactly as the lookup saw it, before any restore moved an entry.
    shadow: Option<ShadowLcp>,
    /// SF1 attribution counters: candidates dropped after an adoption `MemoryPlan`
    /// denial, and entries spent by `evict_lru_one` (the EV-1 order). LIFETIME store totals,
    /// never reset. Each request line carries ONE sample, taken right after the request's own
    /// (last) restore on the batch worker and after its prefill on the single worker; lines are
    /// written at completion, so attribute a request's own restore drops against the sample of
    /// the restore that ran just before it (`restore_drops` rises only at restores, so ordering
    /// samples by value recovers restore order) - never against the previous line. `evictions`
    /// counts only `evict_lru_one` spends and also rises between samples through other slots'
    /// denial evictions; a self-spend counts BOTH counters once, so drops + evictions over-counts
    /// entries lost by the number of self-evictions.
    /// (Opus r1 MF-1, r2 MF-1: the earlier delta-across-two-lines rule was false - a
    /// request writes exactly one line.) They ride the store's `PrefixReport`.
    restore_drops: u64,
    evictions: u64,
    /// B2 attribution counters (the save side). `room_removals` counts entries silently
    /// removed to admit OTHER saves (the room loop and the denial retry inside `save`):
    /// these removals are NOT `evictions` - that counter only counts `evict_lru_one`
    /// spends - so a store that churns entries under save pressure reports `evictions` 0.
    /// `save_refusals` counts saves that returned without storing (room full with every
    /// entry protected, a zero limit, no victim left for a denial retry, or a root
    /// promotion that found no victim). `duplicate_uses` counts saves that were a use,
    /// not a snapshot (rule a). `block_rolls` counts Block cuts that rolled the lineage's
    /// previous cut away (rule b2). `restored_replaces` counts tips that replaced the
    /// restored entry in place (rules a/f). LIFETIME totals with the same one-sample-per-
    /// request-line attribution rule as the SF1 pair.
    room_removals: u64,
    save_refusals: u64,
    duplicate_uses: u64,
    block_rolls: u64,
    restored_replaces: u64,
}

/// The old name of `PrefixStore` (decision 0036, addendum 1), kept for the executors' field
/// declarations.
pub(crate) type PrefixSlot<S, P> = PrefixStore<S, P>;

/// The B2 save-side counter snapshot carried by `PrefixReport`: flat lifetime totals of
/// the store's save decisions, so a request line can attribute a lineage's lost marks to
/// a mechanism (room churn, refusal, duplicate use, block roll, or an in-place replace)
/// instead of guessing.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub(crate) struct SaveSideTotals {
    pub(crate) room_removals: u64,
    pub(crate) save_refusals: u64,
    pub(crate) duplicate_uses: u64,
    pub(crate) block_rolls: u64,
    pub(crate) restored_replaces: u64,
}

impl<S, P> Default for PrefixStore<S, P> {
    fn default() -> Self {
        Self::with_limits(StoreLimits::default())
    }
}

impl<S, P> PrefixStore<S, P> {
    /// A store with the given limits.
    pub(crate) fn with_limits(limits: StoreLimits) -> Self {
        Self {
            roots: Vec::new(),
            lru: Vec::new(),
            limits,
            protect_recent_longer: false,
            next_id: 0,
            restored: None,
            restored_kind: None,
            shadow_enabled: false,
            shadow: None,
            restore_drops: 0,
            evictions: 0,
            room_removals: 0,
            save_refusals: 0,
            duplicate_uses: 0,
            block_rolls: 0,
            restored_replaces: 0,
        }
    }

    /// Preserves a longer entry through one subsequent request.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn with_recent_longer_protection(mut self) -> Self {
        self.protect_recent_longer = true;
        self
    }

    fn begin_request(&mut self) {
        if !self.protect_recent_longer {
            return;
        }
        for stored in self.roots.iter_mut().chain(&mut self.lru) {
            stored.requests_since_use = stored.requests_since_use.saturating_add(1);
        }
    }

    /// The number of key tokens of the most recently used entry, or `None` when the store is
    /// empty. The LRU list wins over the roots; with the `Default` limits this is the one entry.
    pub(crate) fn key_len(&self) -> Option<usize> {
        self.lru
            .last()
            .or_else(|| self.roots.last())
            .map(|stored| stored.saved.key.len())
    }

    /// The number of pinned roots.
    pub(crate) fn roots_len(&self) -> usize {
        self.roots.len()
    }

    /// The number of LRU entries.
    pub(crate) fn entries_len(&self) -> usize {
        self.lru.len()
    }

    /// The limits this store was built with.
    pub(crate) fn limits(&self) -> StoreLimits {
        self.limits
    }

    /// SF1 attribution: candidates dropped after an adoption denial, and entries spent by
    /// the radix-order eviction. LIFETIME totals - the attribution rule is the field docs'
    /// (one sample per request line; compare against the PREVIOUS RESTORE's sample, not
    /// the previous line).
    pub(crate) fn restore_drop_count(&self) -> u32 {
        u32::try_from(self.restore_drops).unwrap_or(u32::MAX)
    }

    /// SF1 attribution twin of `restore_drop_count`.
    pub(crate) fn eviction_count(&self) -> u32 {
        u32::try_from(self.evictions).unwrap_or(u32::MAX)
    }

    /// B2 attribution: the save-side counter snapshot for `PrefixReport` (flat lifetime
    /// totals; the attribution rule is the field docs' - one sample per request line).
    pub(crate) fn save_side_totals(&self) -> SaveSideTotals {
        // u64 end-to-end: these are lifetime totals, and a clamped u32 readout would
        // silently stop reflecting the store's history once it saturates.
        SaveSideTotals {
            room_removals: self.room_removals,
            save_refusals: self.save_refusals,
            duplicate_uses: self.duplicate_uses,
            block_rolls: self.block_rolls,
            restored_replaces: self.restored_replaces,
        }
    }

    /// The kind of the entry that the last `restore_prefix` call restored, or `None` when it
    /// restored nothing. The kind survives the lineage drop of the restored entry, so a report
    /// taken after the job's saves still names the entry the job restored.
    pub(crate) fn restored_kind(&self) -> Option<SaveKind> {
        self.restored_kind
    }

    /// Turns the shadow block-LCP observation on (KV-v1 slice 1). It runs before a
    /// conversation holds entries, beside the limits.
    pub(crate) fn set_shadow_enabled(&mut self, enabled: bool) {
        self.shadow_enabled = enabled;
    }

    /// The shadow observation of the last `restore_prefix` call, or `None` when the store
    /// never ran an enabled lookup.
    pub(crate) fn shadow_report(&self) -> Option<ShadowLcp> {
        self.shadow
    }

    /// Every stored state, roots first, each list from the least to the most recently used.
    /// The store never sizes a state: the executor sums the charge of its own snapshots
    /// (packet SERVE-2-c item 3).
    pub(crate) fn snapshots(&self) -> impl Iterator<Item = &S> {
        self.roots
            .iter()
            .chain(&self.lru)
            .map(|stored| &stored.saved.state)
    }

    /// The prompt tokens shared with a stored key beyond the `restored` tokens (packet
    /// SERVE-2-c item 3): the longest common prefix length between `prompt` and ANY stored
    /// key, of any numeric path, minus `restored`, saturating at 0. The literal common
    /// prefix counts, not a full stored key: a conversation that diverges from a stored tip
    /// at token 30,000 still shares 30,000 tokens with it, and the tokens between the restore
    /// and the divergence are exactly the ones no checkpoint covers (lesson L2 of the
    /// vllm-kvcache study). A restore that took the longest common prefix reports 0.
    pub(crate) fn uncached_shared(&self, prompt: &[u32], restored: usize) -> u32 {
        // `restored` is the floor: the count saturates at 0 when every key shares less.
        let mut longest = restored;
        for stored in self.roots.iter().chain(&self.lru) {
            let shared = stored
                .saved
                .key
                .iter()
                .zip(prompt)
                .take_while(|(key, token)| key == token)
                .count();
            longest = longest.max(shared);
        }
        u32::try_from(longest - restored).unwrap_or(u32::MAX)
    }

    /// The key length and `is_root` of every stored entry, roots first, each list from the least
    /// to the most recently used. For tests and logs.
    pub(crate) fn keys(&self) -> Vec<(usize, bool)> {
        self.roots
            .iter()
            .chain(&self.lru)
            .map(|stored| (stored.saved.key.len(), stored.kind == SaveKind::Root))
            .collect()
    }
}

impl<S, P: PartialEq> PrefixStore<S, P> {
    /// Transfers a replaceable Tip only after the executor proves exclusive live ownership.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn take_tip_if(
        &mut self,
        key: &[u32],
        path: &P,
        eligible: impl Fn(&S) -> bool,
    ) -> Option<S> {
        if !self.protect_recent_longer {
            return None;
        }
        let index = self.lru.iter().position(|stored| {
            stored.kind == SaveKind::Tip
                && stored.saved.path == *path
                && stored.saved.key.len() < key.len()
                && key.starts_with(&stored.saved.key)
                && eligible(&stored.saved.state)
        })?;
        let stored = self.lru.remove(index);
        if self.restored == Some(stored.id) {
            self.restored = None;
            self.restored_replaces += 1;
        }
        Some(stored.saved.state)
    }

    /// Steps (a) to (f) of a save, after `DecodeSession::save_prefix` validated the session. The
    /// order is the order of decision 0036, addendum 2: use a duplicate (or promote it to a
    /// root), drop a same-key entry of another path, make room, take the snapshot with eviction
    /// retries on a denied permit, and apply the lineage rule.
    fn save(
        &mut self,
        key: &[u32],
        pending: u32,
        path: P,
        kind: SaveKind,
        snapshot: &mut dyn FnMut() -> Result<S>,
    ) -> Result<bool> {
        // (a) A stored entry with the same key and the same path is a use, not a new snapshot.
        // A repeated tip that extends the restored entry still replaces that entry (rule 4).
        if let Some(done) = self.use_duplicate(key, &path, kind) {
            if kind == SaveKind::Tip && self.drop_restored_ancestor(key) {
                self.restored_replaces += 1;
            }
            if done {
                self.duplicate_uses += 1;
            } else {
                self.save_refusals += 1; // a root promotion that found no victim
            }
            return Ok(done);
        }
        // (b) An entry with the same key and another path is replaced.
        let other_path =
            |stored: &Stored<S, P>| stored.saved.key == key && stored.saved.path != path;
        self.roots.retain(|stored| !other_path(stored));
        self.lru.retain(|stored| !other_path(stored));
        // A pinned store may preserve a longer cut across an unrelated request.
        if kind == SaveKind::Block {
            if self.lru.iter().any(|stored| {
                stored.kind == SaveKind::Block
                    && stored.saved.path == path
                    && self.protects_longer(stored, key.len())
            }) {
                self.save_refusals += 1;
                return Ok(false);
            }
            let before = self.lru.len();
            self.lru
                .retain(|stored| stored.kind != SaveKind::Block || stored.saved.path != path);
            self.block_rolls += (before - self.lru.len()) as u64;
        }
        // (c) Room. Only a Root save can evict a root here (rule e).
        // While making room, recent-longer protection tries to retire a restored Tip ancestor
        // before other LRU victims. Otherwise, that Tip ancestor is the last resort for an extending Tip.
        match kind {
            SaveKind::Root => {
                if self.limits.roots == 0 {
                    self.save_refusals += 1;
                    return Ok(false);
                }
                if self.roots.len() >= self.limits.roots {
                    match self.first_unprotected_root(key.len()) {
                        Some(pos) => {
                            self.roots.remove(pos);
                            self.room_removals += 1;
                        }
                        None => {
                            self.save_refusals += 1; // the only root is the restored one
                            return Ok(false);
                        }
                    }
                }
            }
            SaveKind::Periodic | SaveKind::Tip | SaveKind::Block => {
                if self.limits.entries == 0 {
                    self.save_refusals += 1;
                    return Ok(false);
                }
                while self.lru.len() >= self.limits.entries {
                    // Keep a new boundary Block when its restored Tip is superseded.
                    if self.protect_recent_longer
                        && kind == SaveKind::Tip
                        && self.drop_restored_ancestor(key)
                    {
                        self.restored_replaces += 1;
                        continue;
                    }
                    match self.first_unprotected(key.len()) {
                        Some(pos) => {
                            self.lru.remove(pos);
                            self.room_removals += 1;
                        }
                        None => {
                            // Last resort (clarification 1, item 1): the restored entry goes,
                            // before the snapshot, when the tip extends it.
                            if kind == SaveKind::Tip && self.drop_restored_ancestor(key) {
                                self.restored_replaces += 1;
                                continue;
                            }
                            self.save_refusals += 1; // every entry is protected
                            return Ok(false);
                        }
                    }
                }
            }
        }
        // (d) The snapshot, with one eviction before each retry of a denied permit. The victim is
        // dropped before the next `snapshot` call: dropping releases its charge.
        loop {
            match snapshot() {
                Ok(state) => {
                    let stored = Stored {
                        id: self.next_id,
                        kind,
                        requests_since_use: 0,
                        saved: SavedPrefix {
                            key: key.to_vec(),
                            pending,
                            path,
                            state,
                        },
                    };
                    self.next_id += 1;
                    match kind {
                        SaveKind::Root => self.roots.push(stored),
                        SaveKind::Periodic | SaveKind::Block => self.lru.push(stored),
                        SaveKind::Tip => {
                            self.lru.push(stored);
                            // (f) A tip that extends the restored entry replaces it.
                            if self.drop_restored_ancestor(key) {
                                self.restored_replaces += 1;
                            }
                        }
                    }
                    return Ok(true);
                }
                Err(EngError::MemoryPlan(_)) => match self.first_unprotected(key.len()) {
                    Some(pos) => {
                        self.lru.remove(pos);
                        self.room_removals += 1;
                    }
                    None if kind == SaveKind::Root => {
                        match self.first_unprotected_root(key.len()) {
                            Some(pos) => {
                                self.roots.remove(pos);
                                self.room_removals += 1;
                            }
                            None => {
                                self.save_refusals += 1; // no victim is left
                                return Ok(false);
                            }
                        }
                    }
                    None => {
                        self.save_refusals += 1; // no victim is left
                        return Ok(false);
                    }
                },
                Err(error) => return Err(error),
            }
        }
    }

    /// Step (a): when a stored entry has the same key and the same path, it is a use. It moves to
    /// the most recently used place of its own list. A `Root` save that finds the key in the LRU
    /// list promotes that entry to a pinned root instead, without a new snapshot: it takes a root
    /// place and drops a no-longer root other than the restored one when the roots are full,
    /// in LRU order. Returns `None` when no entry matches, else whether the save counts as done.
    /// A promotion that finds no root to drop leaves the entry in the LRU list as a use and
    /// returns `Some(false)`.
    fn use_duplicate(&mut self, key: &[u32], path: &P, kind: SaveKind) -> Option<bool> {
        let same = |stored: &Stored<S, P>| stored.saved.key == key && &stored.saved.path == path;
        if let Some(pos) = self.roots.iter().position(same) {
            let mut stored = self.roots.remove(pos);
            stored.requests_since_use = 0;
            self.roots.push(stored);
            return Some(true);
        }
        let pos = self.lru.iter().position(same)?;
        let mut stored = self.lru.remove(pos);
        stored.requests_since_use = 0;
        if kind != SaveKind::Root || self.limits.roots == 0 {
            self.lru.push(stored);
            return Some(true);
        }
        if self.roots.len() >= self.limits.roots {
            let Some(victim) = self.first_unprotected_root(key.len()) else {
                self.lru.push(stored);
                return Some(false);
            };
            self.roots.remove(victim);
            // An unrelated root spent to admit this promotion is the same churn the room
            // loop counts: without this, a promotion that discards a root reads as a plain
            // duplicate use and the lost root is invisible in the totals.
            self.room_removals += 1;
        }
        stored.kind = SaveKind::Root;
        self.roots.push(stored);
        Some(true)
    }

    fn protects_longer(&self, stored: &Stored<S, P>, key_len: usize) -> bool {
        self.protect_recent_longer
            && stored.saved.key.len() > key_len
            && stored.requests_since_use <= 1
    }

    /// The oldest entry protected by neither the saving session nor the optional policy.
    fn first_unprotected(&self, max_key_len: usize) -> Option<usize> {
        self.lru.iter().position(|stored| {
            self.restored != Some(stored.id) && !self.protects_longer(stored, max_key_len)
        })
    }

    /// Drops one LRU entry per call - never more than one - in the radix-compression
    /// order (round 4d SF-1, over review rounds 4/4c): (0) dead TIP leaves - they never
    /// match the next turn of any conversation; (1) single-chain interior anchors -
    /// entries with exactly one maximal same-path extender (per32 under per49), fully
    /// redundant; (2) anchor leaves - each lineage's deepest entry, its best restore;
    /// (3) branch points - entries two or more incomparable branches extend (a shared
    /// header); oldest first within each tier; the DENIED job's own lineage is the
    /// explicit `protect` and spends last (review round 4 MF2), never the store-global
    /// last-restore marker, which names another conversation in an interleaved batch.
    /// Dropping the restored entry is safe because the session that restored it holds
    /// its own permit for the shared arrays; the cost is one cache miss for that
    /// conversation, never a correctness change. Returns whether an entry was removed.
    pub(crate) fn evict_lru_one(&mut self, protect: Option<u64>) -> bool {
        // Radix-compression order (round 4d, SF-1, ranking review rounds 4c R4-MF3):
        // spend what the radix tree can regrow before what it cannot.
        //   tier 0: dead TIP leaves - they never match the next turn of any conversation;
        //   tier 1: single-chain interior anchors - entries with exactly ONE maximal
        //           same-path extender (per32 under per49): fully redundant, the chain's
        //           deeper entry covers every prompt the interior one would;
        //   tier 2: anchor leaves - each lineage's deepest entry, its best restore;
        //   tier 3: branch points - extended by two or more incomparable branches (a
        //           shared header): losing one costs EVERY branch under it;
        //   tier 4: the protected entry, the denied lineage, last resort.
        // Leaf status recomputed between evictions alone would strip a lineage entirely
        // (per49 gone -> per32 becomes a leaf -> next) while another lineage still holds
        // a redundant interior, and the dead-tip tier frees only ~0.9 GB of ~3 GB the
        // replay's t6 needs, so tiers 1-3 ARE reached there: a covered per@32,768 frees
        // ~420 MB at zero t6 reuse cost while branch headers stay for last.
        //
        // Cost: one O(n^2) key-extension pass plus O(ext^2) maximal-extender counting
        // per entry - trivial at the default 6 entries, and the override cap (4096)
        // only makes a denial-path eviction measurably dear, never wrong.
        let branches_of = |index: usize| -> usize {
            let key = &self.lru[index].saved.key;
            let path = &self.lru[index].saved.path;
            let extenders: Vec<usize> = self
                .lru
                .iter()
                .enumerate()
                .filter(|(other, stored)| {
                    *other != index
                        && &stored.saved.path == path
                        && stored.saved.key.len() > key.len()
                        && stored.saved.key.starts_with(key)
                })
                .map(|(other, _)| other)
                .collect();
            extenders
                .iter()
                .filter(|&&extender| {
                    let extender_key = &self.lru[extender].saved.key;
                    !extenders.iter().any(|&deeper| {
                        deeper != extender
                            && self.lru[deeper].saved.key.len() > extender_key.len()
                            && self.lru[deeper].saved.key.starts_with(extender_key)
                    })
                })
                .count()
        };
        let rank_of = |index: usize| {
            let stored = &self.lru[index];
            let branches = branches_of(index);
            match (branches, stored.kind == SaveKind::Tip) {
                (0, true) => 0,
                (1, _) => 1,
                (0, false) => 2,
                (_, _) => 3,
            }
        };
        let victim = self
            .lru
            .iter()
            .enumerate()
            .filter(|(_, stored)| protect != Some(stored.id))
            .map(|(index, _)| index)
            .min_by_key(|&index| (rank_of(index), index))
            .or_else(|| {
                self.lru
                    .iter()
                    .position(|stored| protect == Some(stored.id))
            });
        match victim {
            Some(pos) => {
                let dropped = self.lru.remove(pos);
                if self.restored == Some(dropped.id) {
                    self.restored = None;
                    self.restored_kind = None;
                }
                self.evictions += 1;
                true
            }
            None => false,
        }
    }

    /// The oldest root protected by neither the saving session nor the optional policy.
    fn first_unprotected_root(&self, max_key_len: usize) -> Option<usize> {
        self.roots.iter().position(|stored| {
            self.restored != Some(stored.id) && !self.protects_longer(stored, max_key_len)
        })
    }

    /// Step (f) and the last-resort drop of clarification 1: when the saving session restored an
    /// LRU entry of kind `Tip` that is still stored and the new tip extends its key, that entry
    /// is dropped and the call reports `true`.
    ///
    /// Rule 4 of addendum 2 says that periodic entries and roots are never dropped this way, so a
    /// restored `Periodic` entry stays, and a restored root is not in the LRU list at all. The
    /// last resort of clarification 1 uses the same filter. It matters only for a store with one
    /// LRU entry, whose default limits (0 roots, 1 entry) never hold a periodic entry because the
    /// worker emits no periodic point below two entries. With limits (2, 1) the worker emits no
    /// periodic point either, so no periodic entry can share the list with the tip. A direct
    /// `save_prefix` call that stores a periodic entry in a one-entry list is outside the rules:
    /// the restored periodic entry then stays and the tip save is skipped.
    fn drop_restored_ancestor(&mut self, key: &[u32]) -> bool {
        let Some(id) = self.restored else {
            return false;
        };
        let Some(pos) = self.lru.iter().position(|stored| stored.id == id) else {
            return false;
        };
        let stored = &self.lru[pos];
        if stored.kind == SaveKind::Tip
            && key.len() > stored.saved.key.len()
            && key.starts_with(&stored.saved.key)
        {
            self.lru.remove(pos);
            return true;
        }
        false
    }

    /// The restore candidates for `prompt` under `path`: every stored entry whose path equals
    /// `path` and whose key is a strict prefix of `prompt`, longest key first.
    fn candidates(&self, prompt: &[u32], path: &P) -> Vec<(u64, bool, usize)> {
        let mut found: Vec<(u64, bool, usize)> = Vec::new();
        for (is_root, list) in [(true, &self.roots), (false, &self.lru)] {
            for stored in list {
                if &stored.saved.path == path
                    && stored.saved.key.len() < prompt.len()
                    && prompt.starts_with(&stored.saved.key)
                {
                    found.push((stored.id, is_root, stored.saved.key.len()));
                }
            }
        }
        found.sort_by_key(|&(_, _, len)| Reverse(len));
        found
    }

    /// The first candidate a `restore_prefix` for `prompt` under `path` would try (review
    /// round 4, MF2): an admission-rebuild eviction protects it while no session exists,
    /// so the rebuild does not free the very anchor its next attempt restores.
    pub(crate) fn restore_candidate(&self, prompt: &[u32], path: &P) -> Option<u64> {
        self.candidates(prompt, path).first().map(|&(id, _, _)| id)
    }

    /// One restore attempt on the entry `id`: run `restore` on its state. On success the entry
    /// becomes the most recently used entry of its list, it becomes the protected entry (root or
    /// not), and its key and pending token are returned. A first `MemoryPlan` error evicts one
    /// other entry (this entry protected; alone in the list, it spends itself and counts as a
    /// drop) and retries this entry once; a second `MemoryPlan` error drops the entry and
    /// returns `None`, so the caller tries the next candidate; any other error propagates and
    /// keeps the entries.
    fn restore_entry(
        &mut self,
        id: u64,
        is_root: bool,
        restore: &mut dyn FnMut(&S) -> Result<()>,
    ) -> Result<Option<(Vec<u32>, u32)>> {
        // Two attempts at most. The first adoption `MemoryPlan` denial does NOT drop the
        // candidate (SF1, from the override-16 trace: request 0x18's periodic anchor was
        // the candidate its 90K turn restores, and dropping it made t6 a 410 s re-prefill
        // instead of a 235 s restore): mirror the save path's step (d) - evict ONE victim
        // in the radix order with the failed candidate itself protected, then RETRY THE
        // SAME candidate. Only a second denial (no victim existed, or the pool freed by
        // one victim still could not adopt) runs the drop-and-next behavior. Re-locate
        // the entry after the eviction: the victim may share this list.
        for attempt in 0..2 {
            let list = if is_root {
                &mut self.roots
            } else {
                &mut self.lru
            };
            let Some(pos) = list.iter().position(|stored| stored.id == id) else {
                return Ok(None);
            };
            match restore(&list[pos].saved.state) {
                Ok(()) => {
                    let mut stored = list.remove(pos);
                    stored.requests_since_use = 0;
                    let given = (stored.saved.key.clone(), stored.saved.pending);
                    let id = stored.id;
                    let kind = stored.kind;
                    list.push(stored);
                    self.restored = Some(id);
                    self.restored_kind = Some(kind);
                    return Ok(Some(given));
                }
                Err(EngError::MemoryPlan(_)) => {
                    if attempt == 0 && self.evict_lru_one(Some(id)) {
                        // The freed victim lets the SAME candidate retry - unless
                        // EV-1's tier-4 last resort spent the candidate ITSELF
                        // (nothing else existed), which IS the legacy drop: the
                        // anchor died to its own denial, so count it and stop (Sol
                        // SF1-r1 MF1: a singleton's report must not say zero drops).
                        let still_present = if is_root {
                            self.roots.iter().any(|stored| stored.id == id)
                        } else {
                            self.lru.iter().any(|stored| stored.id == id)
                        };
                        if still_present {
                            continue;
                        }
                        self.restore_drops += 1;
                        return Ok(None);
                    }
                    self.restore_drops += 1;
                    let list = if is_root {
                        &mut self.roots
                    } else {
                        &mut self.lru
                    };
                    if let Some(pos) = list.iter().position(|stored| stored.id == id) {
                        list.remove(pos);
                    }
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(None)
    }
}

/// One stored entry of a `PrefixStore`: the saved prefix plus its store id and its save kind.
struct Stored<S, P> {
    /// The id of this entry in its store. Ids are never reused.
    id: u64,
    /// What kind of save produced this entry.
    kind: SaveKind,
    /// Requests since the last successful use, saturated rather than wrapped.
    requests_since_use: u8,
    /// The key, the pending token, the numeric path, and the state.
    saved: SavedPrefix<S, P>,
}

/// The saved prefix of an entry: the state after the key tokens and the token that the next step
/// feeds.
struct SavedPrefix<S, P> {
    /// The committed prompt tokens that the state covers.
    key: Vec<u32>,
    /// The token that the next step feeds after a restore.
    pending: u32,
    /// The numeric path that built the state.
    path: P,
    state: S,
}

/// Rejects a prompt that leaves no room to generate. A prompt of `total` tokens needs `total`
/// fed positions, and the first sampled token needs one more, so `total` must stay below
/// `context_limit`. For example, 32,768 positions allow 32,767 prompt tokens. This is the one
/// authority for the rule and its message: the session applies it in `prefill`, and the
/// executors apply it through `ExecutorOps::check_prompt` before a job starts.
pub(crate) fn check_prompt(total: usize, context_limit: usize) -> Result<()> {
    if total < context_limit {
        return Ok(());
    }
    Err(EngError::PromptTooLong {
        prompt_tokens: total as u64,
        limit_tokens: context_limit.saturating_sub(1) as u64,
    })
}

/// The finish reason before a step, or `None`. A job with no tokens left ends without a step.
/// `remaining` is `GenerationState::remaining_tokens`.
fn before_step(remaining: Option<u32>) -> Option<DecodeFinish> {
    (remaining == Some(0)).then_some(DecodeFinish::Length)
}

/// The finish reason after the sampler chose `token`, or `None`.
///
/// `eos` holds the end-of-sequence ids. `remaining` is `GenerationState::remaining_tokens` after
/// the caller counted `token` (an EOS token is not counted). `fed` is the number of positions
/// fed to the model so far. The sampled token needs position `fed` in the next step, so `fed` at
/// or above `context_limit` ends the job with `Length`. The order is EOS, output limit, context.
fn after_token(
    eos: &[u32],
    token: u32,
    remaining: Option<u32>,
    fed: usize,
    context_limit: usize,
) -> Option<DecodeFinish> {
    if eos.contains(&token) {
        return Some(DecodeFinish::Stop);
    }
    if remaining == Some(0) || fed >= context_limit {
        return Some(DecodeFinish::Length);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        DecodePreparation, DecodeSession, Limits, PrefixStore, Result, SaveKind, ShadowReason,
        StoreLimits, TokenStepper, after_token, before_step, check_prompt,
    };
    use crate::{
        EngError,
        family::{
            executor::GenerationState,
            sampler::{Sampler, SamplingParams},
        },
        tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer},
    };
    use std::sync::atomic::Ordering;

    /// The synthetic tokenizer: ids 0 to 7 are the letters a to h.
    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

    /// The decoder of the rewriting fixture of `tokenizer/tests.rs`: pushing ids 0 and 1 and then
    /// calling `finish` fails, because the final text no longer extends the text already sent.
    const REWRITING_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Replace","pattern":{"String":"ab"},"content":"X"}]},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;

    /// A tokenizer whose id 0 is the first byte of a two-byte character (see `tokenizer/tests.rs`).
    const SPLIT_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":4,"content":"<end>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"Ã":0,"©":1,"Ã©":2,"a":3},"merges":[]}}"#;

    /// The vocabulary of the fake model.
    const VOCAB: usize = 8;

    /// A scripted model: it records every call and has no native state. A logits step on token `t`
    /// returns a row whose largest value is at id `(t + 1) % VOCAB`, so a greedy job counts up.
    #[derive(Default)]
    struct Fake {
        /// The tokens fed so far, by `feed` or `logits`, in call order.
        fed: Vec<u32>,
        /// The number of `reset` calls that succeeded.
        resets: usize,
        /// `feed` fails for this token.
        fail_feed_on: Option<u32>,
        /// `reset` fails.
        fail_reset: bool,
        /// The batches that `feed_rows` received, in call order.
        batches: Vec<Vec<u32>>,
        /// `logits` fails.
        fail_logits: bool,
        /// The width of a logits row. A row wider than `VOCAB` ends with the value 100.0.
        width: usize,
    }

    impl Fake {
        /// A model whose logits rows are `VOCAB` wide.
        fn new() -> Self {
            Self {
                width: VOCAB,
                ..Self::default()
            }
        }
    }

    impl TokenStepper for Fake {
        const OPERATION: &'static str = "fake";

        fn feed(&mut self, token: u32) -> Result<()> {
            if self.fail_feed_on == Some(token) {
                return Err(EngError::Backend {
                    operation: "fake",
                    message: format!("feed {token} failed"),
                });
            }
            self.fed.push(token);
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
            self.batches.push(tokens.to_vec());
            tokens.iter().try_for_each(|&token| self.feed(token))
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.fail_logits {
                return Err(EngError::Backend {
                    operation: "fake",
                    message: "logits failed".to_string(),
                });
            }
            self.fed.push(token);
            let mut row = vec![0.0; self.width];
            if let Some(slot) = row.get_mut((token as usize + 1) % VOCAB) {
                *slot = 1.0;
            }
            if self.width > VOCAB
                && let Some(last) = row.last_mut()
            {
                *last = 100.0;
            }
            Ok(row)
        }

        fn fed(&self) -> usize {
            self.fed.len()
        }

        fn reset(&mut self) -> Result<()> {
            if self.fail_reset {
                return Err(EngError::Backend {
                    operation: "fake",
                    message: "reset failed".to_string(),
                });
            }
            self.fed.clear();
            self.resets += 1;
            Ok(())
        }
    }

    const LIMIT: usize = 32_768;
    const EOS_NONE: [u32; 0] = [];

    fn limits(eos: &[u32]) -> Limits<'_> {
        Limits {
            eos,
            sampler_vocab: VOCAB,
            context_limit: LIMIT,
        }
    }

    fn generation() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap();
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    /// The synthetic tokenizer cut to its first `size` ids.
    fn tokenizer_with_vocab(size: usize) -> Tokenizer {
        let full = r#""vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}"#;
        let ids: Vec<String> = "abcdefgh"
            .chars()
            .take(size)
            .enumerate()
            .map(|(id, c)| format!("\"{c}\":{id}"))
            .collect();
        let json = TOKENIZER_JSON.replace(full, &format!("\"vocab\":{{{}}}", ids.join(",")));
        assert_ne!(json, TOKENIZER_JSON, "the vocabulary was not replaced");
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    /// What `decode_all` saw: the batches in call order, the error that stopped the loop, and
    /// whether the last batch carried a finish reason. The helper never panics, so a wrong
    /// behavior fails in the assertions of the calling test.
    struct Run {
        batches: Vec<DecodeBatch>,
        error: Option<EngError>,
        finished: bool,
    }

    impl Run {
        /// The batches of a job that finished without an error. Both conditions are assertions.
        fn finished_batches(self) -> Vec<DecodeBatch> {
            assert_eq!(self.error, None, "a decode call failed");
            assert!(self.finished, "the job did not finish in 100 decode calls");
            self.batches
        }
    }

    /// Calls `decode` until a batch carries a finish reason, an error, or 100 calls.
    fn decode_all(
        session: &mut DecodeSession,
        step: &mut Fake,
        generation: &mut GenerationState,
        limits: &Limits<'_>,
    ) -> Run {
        let mut run = Run {
            batches: Vec::new(),
            error: None,
            finished: false,
        };
        for _ in 0..100 {
            match session.decode(step, generation, limits) {
                Ok(batch) => {
                    run.finished = batch.finish.is_some();
                    run.batches.push(batch);
                }
                Err(error) => {
                    run.error = Some(error);
                    return run;
                }
            }
            if run.finished {
                return run;
            }
        }
        run
    }

    fn tokens_of(batches: &[DecodeBatch]) -> Vec<u32> {
        batches.iter().flat_map(|b| b.tokens.clone()).collect()
    }

    #[test]
    fn a_prompt_below_the_limit_is_accepted() {
        assert_eq!(check_prompt(32_767, 32_768), Ok(()));
        assert_eq!(check_prompt(1, 32_768), Ok(()));
    }

    #[test]
    fn a_prompt_at_the_limit_names_both_numbers() {
        assert_eq!(
            check_prompt(32_768, 32_768),
            Err(EngError::PromptTooLong {
                prompt_tokens: 32_768,
                limit_tokens: 32_767
            })
        );
        assert!(matches!(
            check_prompt(40_000, 32_768),
            Err(EngError::PromptTooLong { .. })
        ));
    }

    #[test]
    fn a_new_session_is_empty() {
        let session = DecodeSession::default();
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.prompt_len(), 0, "prompt length");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "finished");
    }

    #[test]
    fn a_prompt_feeds_all_but_its_last_token_and_keeps_the_last_pending() {
        let mut step = Fake::default();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        assert_eq!(step.fed, vec![1, 2], "fed tokens");
        assert_eq!(session.pending(), Some(3), "pending token");
        assert_eq!(session.history(), [1, 2, 3], "history");
        assert_eq!(session.prompt_len(), 3, "prompt length");
    }

    #[test]
    fn a_prompt_in_chunks_feeds_the_same_tokens_as_one_chunk() {
        let mut whole = Fake::default();
        let mut one = DecodeSession::default();
        one.prefill(&mut whole, &[1, 2, 3, 4, 5], LIMIT).unwrap();
        let mut parts = Fake::default();
        let mut two = DecodeSession::default();
        two.prefill(&mut parts, &[1, 2], LIMIT).unwrap();
        assert_eq!(parts.fed, vec![1], "after the first chunk");
        two.prefill(&mut parts, &[3, 4, 5], LIMIT).unwrap();
        assert_eq!(parts.fed, whole.fed, "fed tokens");
        assert_eq!(
            parts.fed,
            vec![1, 2, 3, 4],
            "the pending token of chunk 1 is fed first"
        );
        assert_eq!(two.history(), one.history(), "history");
        assert_eq!(two.pending(), Some(5), "pending token");
        assert_eq!(two.prompt_len(), 5, "prompt length");
    }

    #[test]
    fn a_prompt_reaches_the_model_as_one_batch_per_chunk_with_the_pending_token_first() {
        let mut step = Fake::default();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[7], LIMIT).unwrap();
        assert!(step.batches.is_empty(), "a single token has no rows");
        session.prefill(&mut step, &[1, 2], LIMIT).unwrap();
        session.prefill(&mut step, &[3, 4, 5], LIMIT).unwrap();
        assert_eq!(
            step.batches,
            vec![vec![7, 1], vec![2, 3, 4]],
            "each chunk feeds the pending token of the earlier chunk and its own head"
        );
        assert_eq!(step.fed, vec![7, 1, 2, 3, 4], "fed tokens");
        assert_eq!(session.pending(), Some(5), "pending token");
    }

    /// A stepper that keeps the default `feed_rows`.
    struct Plain(Vec<u32>);

    impl TokenStepper for Plain {
        const OPERATION: &'static str = "plain";

        fn feed(&mut self, token: u32) -> Result<()> {
            self.0.push(token);
            Ok(())
        }

        // Like the real steppers, `logits` advances the position that `fed` reads.
        fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
            self.0.push(token);
            Ok(Vec::new())
        }

        fn fed(&self) -> usize {
            self.0.len()
        }

        fn reset(&mut self) -> Result<()> {
            self.0.clear();
            Ok(())
        }
    }

    #[test]
    fn the_default_feed_rows_feeds_one_token_at_a_time_in_order() {
        let mut step = Plain(Vec::new());
        step.feed_rows(&[4, 5, 6]).unwrap();
        assert_eq!(step.0, vec![4, 5, 6]);
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        assert_eq!(step.0, vec![4, 5, 6, 1, 2], "a prompt through the default");
    }

    #[test]
    fn an_empty_chunk_is_rejected_and_changes_nothing() {
        let mut step = Fake::default();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2], LIMIT).unwrap();
        let result = session.prefill(&mut step, &[], LIMIT);
        assert_eq!(
            result,
            Err(EngError::BadRequest("empty prefill chunk".to_string()))
        );
        assert_eq!(step.fed, vec![1], "fed tokens");
        assert_eq!(session.pending(), Some(2), "pending token");
        assert_eq!(session.history(), [1, 2], "history");
    }

    #[test]
    fn a_chunk_that_crosses_the_limit_is_rejected_before_any_feed() {
        let mut step = Fake::default();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2], 4).unwrap();
        let result = session.prefill(&mut step, &[3, 4], 4);
        assert!(
            matches!(result, Err(EngError::PromptTooLong { .. })),
            "{result:?}"
        );
        assert_eq!(step.fed, vec![1], "no token of the rejected chunk was fed");
        assert_eq!(session.pending(), Some(2), "pending token");
        assert_eq!(session.history(), [1, 2], "history");
        session.prefill(&mut step, &[3], 4).unwrap();
        assert_eq!(
            session.history(),
            [1, 2, 3],
            "a chunk below the limit is accepted"
        );
    }

    #[test]
    fn a_failed_feed_leaves_the_history_unchanged() {
        let mut step = Fake {
            fail_feed_on: Some(2),
            ..Fake::default()
        };
        let mut session = DecodeSession::default();
        let result = session.prefill(&mut step, &[1, 2, 3], LIMIT);
        assert!(
            matches!(result, Err(EngError::Backend { .. })),
            "{result:?}"
        );
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.prompt_len(), 0, "prompt length");
        assert_eq!(session.pending(), None, "pending token");
    }

    #[test]
    fn a_reset_clears_the_model_and_the_tokens() {
        let mut step = Fake::default();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        session.reset(&mut step).unwrap();
        assert_eq!(step.resets, 1, "model resets");
        assert!(step.fed.is_empty(), "model state");
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.prompt_len(), 0, "prompt length");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "finished");
    }

    #[test]
    fn a_failed_model_reset_keeps_the_tokens() {
        let mut step = Fake::default();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        step.fail_reset = true;
        let result = session.reset(&mut step);
        assert!(
            matches!(result, Err(EngError::Backend { .. })),
            "{result:?}"
        );
        assert_eq!(session.history(), [1, 2, 3], "history");
        assert_eq!(session.pending(), Some(3), "pending token");
    }

    #[test]
    fn a_job_with_no_tokens_left_ends_before_the_step() {
        assert_eq!(before_step(Some(0)), Some(DecodeFinish::Length));
        assert_eq!(before_step(Some(1)), None);
        assert_eq!(before_step(None), None);
    }

    #[test]
    fn an_eos_token_stops_even_when_the_other_limits_are_reached() {
        let eos = [248_046, 248_044];
        for id in eos {
            assert_eq!(
                after_token(&eos, id, Some(0), 9_999, 32_768),
                Some(DecodeFinish::Stop),
                "eos {id}"
            );
        }
    }

    #[test]
    fn the_last_allowed_token_ends_with_length() {
        let eos = [248_046, 248_044];
        assert_eq!(
            after_token(&eos, 7, Some(0), 10, 32_768),
            Some(DecodeFinish::Length)
        );
        assert_eq!(after_token(&eos, 7, Some(1), 10, 32_768), None);
        assert_eq!(after_token(&eos, 7, None, 10, 32_768), None);
    }

    #[test]
    fn a_sampled_token_that_has_no_position_ends_with_length() {
        let eos = [248_046, 248_044];
        assert_eq!(after_token(&eos, 7, None, 32_767, 32_768), None);
        assert_eq!(
            after_token(&eos, 7, None, 32_768, 32_768),
            Some(DecodeFinish::Length)
        );
        assert_eq!(
            after_token(&eos, 7, Some(5), 32_769, 32_768),
            Some(DecodeFinish::Length)
        );
    }

    /// Runs the window-facing policy hooks around an independently supplied model step.
    fn split_decode(
        session: &mut DecodeSession,
        step: &mut Fake,
        job: &mut GenerationState,
        limits: &Limits<'_>,
    ) -> Result<DecodeBatch> {
        let pending = match session.prepare_decode(step, job)? {
            DecodePreparation::Finished(batch) => return Ok(batch),
            DecodePreparation::Token(pending) => pending,
        };
        let before = step.fed();
        let sampled = step.next_token(pending, &mut job.sampler, limits.sampler_vocab);
        session.record_feed(before, step.fed());
        session.finish_decode_token(step, job, limits, sampled?)
    }

    #[test]
    fn row_policy_hooks_match_solo_sampling_eos_limits_and_cancellation() {
        for seeded in [false, true] {
            for (eos, max_tokens, cancelled, context) in [
                (vec![], 3, false, LIMIT),
                (vec![4, 6], 10, false, LIMIT),
                (vec![], 0, false, LIMIT),
                (vec![], 0, true, LIMIT),
                (vec![], 10, false, 3),
            ] {
                let make_job = || {
                    let mut job = generation().with_max_tokens(max_tokens);
                    if seeded {
                        job.sampler = Sampler::new(SamplingParams::new(1.0, 1.0, 0).unwrap(), 41);
                    }
                    job.cancel.store(cancelled, Ordering::SeqCst);
                    job
                };
                let (mut solo, mut row) = (DecodeSession::default(), DecodeSession::default());
                let (mut solo_step, mut row_step) = (Fake::new(), Fake::new());
                let (mut solo_job, mut row_job) = (make_job(), make_job());
                solo.prefill(&mut solo_step, &[1, 2, 3], LIMIT).unwrap();
                row.prefill(&mut row_step, &[1, 2, 3], LIMIT).unwrap();
                let limits = Limits {
                    context_limit: context,
                    ..limits(&eos)
                };
                for _ in 0..12 {
                    let expected = solo.decode(&mut solo_step, &mut solo_job, &limits).unwrap();
                    let actual =
                        split_decode(&mut row, &mut row_step, &mut row_job, &limits).unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(row.history(), solo.history());
                    assert_eq!(row.pending(), solo.pending());
                    assert_eq!(row.is_finished(), solo.is_finished());
                    assert_eq!(row_step.fed, solo_step.fed);
                    assert_eq!(row_step.resets, solo_step.resets);
                    assert_eq!(row_job.remaining_tokens(), solo_job.remaining_tokens());
                    if actual.finish.is_some() {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn row_policy_hooks_preserve_pending_only_when_the_input_was_not_fed() {
        for (fail_logits, width, expected_pending) in [(true, VOCAB, Some(3)), (false, 1, None)] {
            let mut session = DecodeSession::default();
            let mut step = Fake::new();
            let mut job = generation();
            session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
            step.fail_logits = fail_logits;
            step.width = width;
            assert!(matches!(
                split_decode(&mut session, &mut step, &mut job, &limits(&EOS_NONE)),
                Err(EngError::Backend { .. })
            ));
            assert_eq!(session.pending(), expected_pending);
            assert_eq!(session.history(), [1, 2, 3]);
        }
        let mut session = DecodeSession::default();
        assert!(matches!(
            session.prepare_decode(&mut Fake::new(), &mut generation()),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn greedy_decode_follows_the_model_and_ends_at_max_tokens() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(3);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let batches =
            decode_all(&mut session, &mut step, &mut job, &limits(&EOS_NONE)).finished_batches();
        assert_eq!(tokens_of(&batches), vec![4, 5, 6], "tokens");
        assert_eq!(batches.len(), 3, "one token per call");
        assert!(
            batches[..2].iter().all(|b| b.finish.is_none()),
            "early finish"
        );
        assert_eq!(batches[2].finish, Some(DecodeFinish::Length), "last finish");
        let text: String = batches.iter().map(|b| b.text.as_str()).collect();
        assert_eq!(text, "efg", "text");
        assert_eq!(session.history(), [1, 2, 3, 4, 5, 6], "history");
        // Two prompt tokens and three pending tokens are fed. The sixth token is pending.
        assert_eq!(step.fed, vec![1, 2, 3, 4, 5], "fed tokens");
        assert_eq!(session.pending(), Some(6), "pending token");
        assert!(session.is_finished(), "finished");
    }

    #[test]
    fn an_eos_token_ends_the_job_with_stop_and_adds_no_text() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(10);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let eos = [6];
        let batches =
            decode_all(&mut session, &mut step, &mut job, &limits(&eos)).finished_batches();
        assert_eq!(tokens_of(&batches), vec![4, 5, 6], "tokens up to EOS");
        let last = batches.last().unwrap();
        assert_eq!(last.finish, Some(DecodeFinish::Stop), "finish");
        assert_eq!(last.tokens, vec![6], "the EOS token is in the batch");
        assert_eq!(last.text, "", "EOS adds no text");
        assert_eq!(session.history(), [1, 2, 3, 4, 5, 6], "EOS is committed");
        assert_eq!(session.pending(), None, "EOS is not fed");
        assert_eq!(job.remaining_tokens(), Some(8), "EOS does not count");
    }

    #[test]
    fn a_limit_of_zero_ends_before_any_step() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(0);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let result = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(
            result,
            Ok(DecodeBatch {
                tokens: Vec::new(),
                text: String::new(),
                finish: Some(DecodeFinish::Length),
            })
        );
        assert_eq!(step.fed, vec![1, 2], "the pending token was not fed");
        assert!(session.is_finished(), "finished");
    }

    #[test]
    fn a_cancelled_job_resets_the_session_and_the_next_job_is_clean() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut cancelled = generation();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        cancelled.cancel.store(true, Ordering::SeqCst);
        let result = session.decode(&mut step, &mut cancelled, &limits(&EOS_NONE));
        assert_eq!(
            result,
            Ok(DecodeBatch {
                tokens: Vec::new(),
                text: String::new(),
                finish: Some(DecodeFinish::Cancelled),
            })
        );
        assert_eq!(step.resets, 1, "model resets");
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "the session is reset, not finished");
        let mut next = generation().with_max_tokens(2);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let batches =
            decode_all(&mut session, &mut step, &mut next, &limits(&EOS_NONE)).finished_batches();
        assert_eq!(tokens_of(&batches), vec![4, 5], "tokens after the cancel");
    }

    #[test]
    fn a_cancel_wins_over_a_limit_of_zero() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(0);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        job.cancel.store(true, Ordering::SeqCst);
        let result = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(
            result,
            Ok(DecodeBatch {
                tokens: Vec::new(),
                text: String::new(),
                finish: Some(DecodeFinish::Cancelled),
            })
        );
        assert_eq!(step.resets, 1, "model resets");
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "the session is reset, not finished");
    }

    #[test]
    fn checked_pending_validates_without_changing_the_session() {
        let mut session = DecodeSession::default();
        assert_eq!(
            session.checked_pending(),
            Err(EngError::BadRequest("decode before prefill".to_string()))
        );
        let mut step = Fake::new();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        assert_eq!(session.checked_pending(), Ok(3));
        assert_eq!(session.history(), [1, 2, 3]);
        assert_eq!(session.prompt_len(), 3);
        assert_eq!(session.pending(), Some(3));
        assert!(!session.is_finished());
        assert_eq!(step.fed, [1, 2]);
        assert_eq!(step.resets, 0);
        session.finished = true;
        session.pending = None;
        assert_eq!(
            session.checked_pending(),
            Err(EngError::BadRequest("decode after finish".to_string()))
        );
        assert!(session.is_finished());
        assert_eq!(session.pending(), None);
    }

    #[test]
    fn a_decode_before_prefill_is_rejected() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let result = session.decode(&mut step, &mut generation(), &limits(&EOS_NONE));
        assert_eq!(
            result,
            Err(EngError::BadRequest("decode before prefill".to_string()))
        );
        assert!(step.fed.is_empty(), "no step ran");
    }

    #[test]
    fn a_decode_after_a_finish_needs_a_reset() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(1);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let first = decode_all(&mut session, &mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(
            first.finished_batches().len(),
            1,
            "the first job ends after one token"
        );
        let fed = step.fed.clone();
        let result = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(
            result,
            Err(EngError::BadRequest("decode after finish".to_string()))
        );
        assert_eq!(step.fed, fed, "no step ran");
        session.reset(&mut step).unwrap();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let mut next = generation().with_max_tokens(1);
        let batches =
            decode_all(&mut session, &mut step, &mut next, &limits(&EOS_NONE)).finished_batches();
        assert_eq!(tokens_of(&batches), vec![4], "tokens after the reset");
    }

    #[test]
    fn a_prefill_after_a_finish_needs_a_reset() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(1);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let batches =
            decode_all(&mut session, &mut step, &mut job, &limits(&EOS_NONE)).finished_batches();
        assert_eq!(tokens_of(&batches), vec![4], "tokens of the first job");
        let held = session.history().to_vec();
        let fed = step.fed.clone();
        let result = session.prefill(&mut step, &[5, 6], LIMIT);
        assert_eq!(
            result,
            Err(EngError::BadRequest("prefill after finish".to_string()))
        );
        assert_eq!(step.fed, fed, "no token was fed");
        assert_eq!(session.history(), held.as_slice(), "history kept");
        assert_eq!(session.pending(), Some(4), "pending kept");
        assert!(session.is_finished(), "still finished");
        session.reset(&mut step).unwrap();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        assert_eq!(session.history(), [1, 2, 3], "after the reset");
    }

    #[test]
    fn a_job_ends_with_length_when_the_next_feed_has_no_position() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let mut job = generation();
        session.prefill(&mut step, &[1, 2, 3], 5).unwrap();
        let eos = EOS_NONE;
        let small = Limits {
            eos: &eos,
            sampler_vocab: VOCAB,
            context_limit: 5,
        };
        let batches = decode_all(&mut session, &mut step, &mut job, &small).finished_batches();
        assert_eq!(tokens_of(&batches), vec![4, 5, 6], "tokens");
        assert_eq!(batches.len(), 3, "the third feed reaches position 5");
        assert_eq!(batches[2].finish, Some(DecodeFinish::Length), "finish");
        assert_eq!(step.fed.len(), 5, "fed positions");
    }

    #[test]
    fn the_sampler_sees_only_the_sampler_vocabulary() {
        let mut step = Fake {
            width: VOCAB + 2,
            ..Fake::default()
        };
        let mut session = DecodeSession::default();
        let mut job = generation().with_max_tokens(3);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let run = decode_all(&mut session, &mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(
            tokens_of(&run.batches),
            vec![4, 5, 6],
            "the value 100.0 beyond the vocabulary must not be drawn"
        );
        assert_eq!(run.error, None, "a decode call failed");
        assert!(run.finished, "the job did not finish");
    }

    /// A job on a model whose logits rows are `width` wide, against a sampler vocabulary of `VOCAB`.
    fn narrow_logits_result(width: usize) -> Result<DecodeBatch> {
        let mut step = Fake {
            width,
            ..Fake::default()
        };
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        session.decode(&mut step, &mut generation(), &limits(&EOS_NONE))
    }

    fn narrow_logits_error() -> EngError {
        EngError::Backend {
            operation: "fake",
            message: "the logits are narrower than the sampler vocabulary".to_string(),
        }
    }

    #[test]
    fn logits_narrower_than_the_sampler_vocabulary_give_a_backend_error() {
        assert_eq!(narrow_logits_result(VOCAB - 4), Err(narrow_logits_error()));
    }

    #[test]
    fn logits_one_short_of_the_sampler_vocabulary_give_a_backend_error() {
        assert_eq!(narrow_logits_result(VOCAB - 1), Err(narrow_logits_error()));
        // A row of exactly the sampler width is accepted.
        let result = narrow_logits_result(VOCAB);
        assert_eq!(
            result.map(|b| b.tokens),
            Ok(vec![4]),
            "tokens at the exact width"
        );
    }

    #[test]
    fn a_failed_step_leaves_the_tokens_and_the_pending_token_unchanged() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        step.fail_logits = true;
        let result = session.decode(&mut step, &mut generation(), &limits(&EOS_NONE));
        assert!(
            matches!(result, Err(EngError::Backend { .. })),
            "{result:?}"
        );
        assert_eq!(session.history(), [1, 2, 3], "history");
        assert_eq!(session.pending(), Some(3), "pending token");
        assert!(!session.is_finished(), "finished");
    }

    /// When an `Override` fails.
    #[derive(Clone, Copy)]
    enum Failure {
        /// Never: `next_token` returns the scripted token.
        Never,
        /// `next_token` fails before it feeds the token.
        BeforeFeed,
        /// `next_token` feeds the token and then fails.
        AfterFeed,
    }

    /// A stepper that overrides `next_token`. Its `logits` counts the call and fails, so a decode
    /// that reaches `logits` is visible in `logits_calls` and in the error.
    struct Override {
        /// The token that `next_token` returns.
        token: u32,
        failure: Failure,
        /// The positions fed so far.
        fed: usize,
        /// The `logits` calls.
        logits_calls: usize,
        /// The arguments of every `next_token` call: the token and the sampler vocabulary.
        calls: Vec<(u32, usize)>,
    }

    impl TokenStepper for Override {
        const OPERATION: &'static str = "override";

        fn feed(&mut self, _token: u32) -> Result<()> {
            self.fed += 1;
            Ok(())
        }

        fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
            self.logits_calls += 1;
            Err(EngError::Backend {
                operation: "override",
                message: "logits must not run".to_string(),
            })
        }

        fn next_token(
            &mut self,
            token: u32,
            _sampler: &mut Sampler,
            sampler_vocab: usize,
        ) -> Result<u32> {
            self.calls.push((token, sampler_vocab));
            if matches!(self.failure, Failure::BeforeFeed) {
                return Err(EngError::Backend {
                    operation: "override",
                    message: "failed before the feed".to_string(),
                });
            }
            self.fed += 1;
            if matches!(self.failure, Failure::AfterFeed) {
                return Err(EngError::Backend {
                    operation: "override",
                    message: "failed after the feed".to_string(),
                });
            }
            Ok(self.token)
        }

        fn fed(&self) -> usize {
            self.fed
        }

        fn reset(&mut self) -> Result<()> {
            self.fed = 0;
            Ok(())
        }
    }

    /// An `Override` that returns `token` or fails as `failure` says, and a session that holds the
    /// prompt `[1, 2, 3]`: two positions are fed and the token 3 is pending.
    fn prefilled_override(token: u32, failure: Failure) -> (Override, DecodeSession) {
        let mut step = Override {
            token,
            failure,
            fed: 0,
            logits_calls: 0,
            calls: Vec::new(),
        };
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        (step, session)
    }

    // The override returns the token 6, where the counting `Fake` would give 4. The session records
    // 6 and keeps it pending; the hook gets the pending token 3 and the sampler vocabulary; the
    // default `logits` is not called; and the job's sampler is not touched (its next 16 draws
    // equal those of a fresh sampler with the same seed).
    //
    // Mutation check: in `decode`, replace the `step.next_token(...)` call by `step.logits(pending)`
    // followed by `generation.sampler.sample(..)`. The override is then bypassed, `logits` runs
    // and fails, and the test fails.
    #[test]
    fn an_overridden_next_token_supplies_the_token_without_logits_or_the_sampler() {
        let (mut step, mut session) = prefilled_override(6, Failure::Never);
        let settings = SamplingParams::new(1.0, 1.0, 0).unwrap();
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap();
        let mut job = GenerationState::with_sampler(
            IncrementalDecoder::new(&tokenizer),
            Sampler::new(settings, 9),
        );
        let batch = session
            .decode(&mut step, &mut job, &limits(&EOS_NONE))
            .unwrap();
        assert_eq!(batch.tokens, vec![6], "batch tokens");
        assert_eq!(session.history(), [1, 2, 3, 6], "history");
        assert_eq!(session.pending(), Some(6), "pending token");
        assert_eq!(step.calls, vec![(3, VOCAB)], "next_token calls");
        assert_eq!(step.logits_calls, 0, "logits calls");
        let row = [0.0_f32; 8];
        let mut fresh = Sampler::new(settings, 9);
        for draw in 0..16 {
            assert_eq!(
                job.sampler.sample(&row),
                fresh.sample(&row),
                "draw {draw}: the sampler was used"
            );
        }
    }

    // The default hook with a stepper whose `OPERATION` is "plain": its `logits` returns an empty
    // row, which is narrower than the sampler vocabulary 8, so the error carries the label of the
    // stepper, not a fixed text of the session. The `Fake` tests above pin the same error for the
    // label "fake".
    //
    // Mutation check: in the default body of `next_token`, replace `Self::OPERATION` by "fake".
    // The label is then "fake" instead of "plain" and the test fails.
    #[test]
    fn the_default_next_token_labels_a_narrow_row_with_the_operation_of_the_stepper() {
        let mut step = Plain(Vec::new());
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let result = session.decode(&mut step, &mut generation(), &limits(&EOS_NONE));
        assert_eq!(
            result,
            Err(EngError::Backend {
                operation: "plain",
                message: "the logits are narrower than the sampler vocabulary".to_string(),
            })
        );
        assert_eq!(session.pending(), None, "pending token");
    }

    // After the model fed the token and the row was narrower than the sampler vocabulary, the
    // token is gone: today's code clears `pending` after `logits` and before the narrowing. The
    // history keeps the prompt, and the model fed the token (three positions).
    //
    // Mutation check: in `decode`, delete the `self.pending = None;` inside the `Err` arm. The
    // pending token 3 then survives the error and the test fails.
    #[test]
    fn a_narrow_row_consumes_the_pending_token() {
        let mut step = Fake {
            width: VOCAB - 1,
            ..Fake::default()
        };
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let result = session.decode(&mut step, &mut generation(), &limits(&EOS_NONE));
        assert_eq!(result, Err(narrow_logits_error()));
        assert_eq!(step.fed, vec![1, 2, 3], "the model fed the token");
        assert_eq!(session.history(), [1, 2, 3], "history");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "finished");
    }

    // A sampler vocabulary of 0 hands the sampler an empty row, which the sampler refuses with a
    // `Backend` error of the operation "sample". The model fed the token before, so the token is
    // gone, exactly as after a narrow row.
    //
    // Mutation check: in `decode`, delete the `self.pending = None;` inside the `Err` arm. The
    // pending token 3 then survives the error and the test fails.
    #[test]
    fn a_sampler_failure_consumes_the_pending_token() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let empty = Limits {
            eos: &EOS_NONE,
            sampler_vocab: 0,
            context_limit: LIMIT,
        };
        let result = session.decode(&mut step, &mut generation(), &empty);
        assert!(
            matches!(
                &result,
                Err(EngError::Backend {
                    operation: "sample",
                    ..
                })
            ),
            "{result:?}"
        );
        assert_eq!(step.fed, vec![1, 2, 3], "the model fed the token");
        assert_eq!(session.history(), [1, 2, 3], "history");
        assert_eq!(session.pending(), None, "pending token");
    }

    // An override that fails before it feeds the token leaves the token pending, as a failing
    // `logits` does (`a_failed_step_leaves_the_tokens_and_the_pending_token_unchanged`). One that
    // fails after the feed consumes it. Both leave the history at the prompt.
    //
    // Mutation check: in `decode`, replace the `if step.fed() != fed_before` condition of the `Err`
    // arm by `true`. The before-feed case then loses its pending token and the test fails.
    #[test]
    fn an_override_that_fails_keeps_the_token_pending_only_when_it_did_not_feed_it() {
        let (mut step, mut session) = prefilled_override(6, Failure::BeforeFeed);
        let result = session.decode(&mut step, &mut generation(), &limits(&EOS_NONE));
        assert!(
            matches!(result, Err(EngError::Backend { .. })),
            "{result:?}"
        );
        assert_eq!(session.pending(), Some(3), "pending token before the feed");
        assert_eq!(session.history(), [1, 2, 3], "history before the feed");
        let (mut step, mut session) = prefilled_override(6, Failure::AfterFeed);
        let result = session.decode(&mut step, &mut generation(), &limits(&EOS_NONE));
        assert!(
            matches!(result, Err(EngError::Backend { .. })),
            "{result:?}"
        );
        assert_eq!(session.pending(), None, "pending token after the feed");
        assert_eq!(session.history(), [1, 2, 3], "history after the feed");
        assert_eq!(step.logits_calls, 0, "logits calls");
    }

    #[test]
    fn a_token_that_the_text_decoder_rejects_resets_the_session() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        // The cut tokenizer knows the ids 0 to 4. The model draws 4 and then 5.
        let mut job = GenerationState::new(IncrementalDecoder::new(&tokenizer_with_vocab(5)));
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let first = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(first.map(|b| b.tokens), Ok(vec![4]), "token 4 decodes");
        let failed = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(_))),
            "{failed:?}"
        );
        assert_eq!(step.resets, 1, "model resets");
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "the session is reset, not finished");
    }

    #[test]
    fn a_failed_decoder_tail_resets_the_session() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        let rewriting = Tokenizer::from_bytes(REWRITING_TOKENIZER_JSON.as_bytes()).unwrap();
        let mut decoder = IncrementalDecoder::new(&rewriting);
        assert_eq!(decoder.push(0).unwrap(), "a");
        assert_eq!(decoder.push(1).unwrap(), "");
        // A limit of zero ends the job in `end` before any step, and `finish` then fails.
        let mut job = GenerationState::new(decoder).with_max_tokens(0);
        session.prefill(&mut step, &[1, 2, 3], LIMIT).unwrap();
        let failed = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(m)) if m.contains("prefix")),
            "{failed:?}"
        );
        assert_eq!(step.resets, 1, "model resets");
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.pending(), None, "pending token");
        assert!(!session.is_finished(), "the session is reset, not finished");
    }

    #[test]
    fn the_decoder_tail_joins_the_text_of_the_last_batch() {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        // The split tokenizer holds the id 0 as half of a UTF-8 sequence. The model draws 0 after 7.
        let split = Tokenizer::from_bytes(SPLIT_TOKENIZER_JSON.as_bytes()).unwrap();
        let mut job = GenerationState::new(IncrementalDecoder::new(&split)).with_max_tokens(1);
        session.prefill(&mut step, &[1, 2, 7], LIMIT).unwrap();
        let result = session.decode(&mut step, &mut job, &limits(&EOS_NONE));
        assert_eq!(
            result,
            Ok(DecodeBatch {
                tokens: vec![0],
                text: "\u{fffd}".to_string(),
                finish: Some(DecodeFinish::Length),
            })
        );
        assert!(session.is_finished(), "finished");
    }

    /// The prefix store of the tests: the snapshot is the list of tokens that the fake has fed,
    /// and the numeric path is a small number.
    type Slot = PrefixStore<Vec<u32>, u8>;

    /// A store with room for `roots` roots and `entries` LRU entries.
    fn store(roots: usize, entries: usize) -> Slot {
        PrefixStore::with_limits(StoreLimits { roots, entries })
    }

    /// Saves the tokens that `step` has fed as a tip.
    fn save(
        session: &DecodeSession,
        slot: &mut Slot,
        step: &Fake,
        key: &[u32],
        path: u8,
    ) -> Result<bool> {
        save_as(session, slot, step, key, path, SaveKind::Tip)
    }

    /// Saves the tokens that `step` has fed as `kind`.
    fn save_as(
        session: &DecodeSession,
        slot: &mut Slot,
        step: &Fake,
        key: &[u32],
        path: u8,
        kind: SaveKind,
    ) -> Result<bool> {
        session.save_prefix(slot, key, path, step.fed(), kind, || Ok(step.fed.clone()))
    }

    /// Gives the saved tokens back to `step`.
    fn restore(
        session: &mut DecodeSession,
        slot: &mut Slot,
        step: &mut Fake,
        prompt: &[u32],
        path: u8,
    ) -> Result<usize> {
        session.restore_prefix(slot, prompt, path, |fed| {
            step.fed.clone_from(fed);
            Ok(())
        })
    }

    /// A session whose prompt `tokens` is prefilled in one chunk.
    fn held(tokens: &[u32]) -> (Fake, DecodeSession) {
        let mut step = Fake::new();
        let mut session = DecodeSession::default();
        session.prefill(&mut step, tokens, LIMIT).unwrap();
        (step, session)
    }

    #[test]
    fn a_crossing_tip_preserves_its_new_block_only_when_protection_is_enabled() {
        let tokens: Vec<u32> = (0..9_500).collect();
        for protected in [false, true] {
            let mut slot = store(0, 2);
            if protected {
                slot = slot.with_recent_longer_protection();
            }
            for (len, kind) in [(4_096, SaveKind::Block), (5_000, SaveKind::Tip)] {
                let key = &tokens[..len];
                let (step, session) = held(key);
                assert_eq!(save_as(&session, &mut slot, &step, key, 1, kind), Ok(true));
            }
            let (mut step, mut session) = held(&tokens[..5_000]);
            session.reset(&mut step).unwrap();
            assert_eq!(
                restore(&mut session, &mut slot, &mut step, &tokens, 1),
                Ok(5_000)
            );
            session
                .prefill(&mut step, &tokens[5_000..8_192], LIMIT)
                .unwrap();
            assert_eq!(
                save_as(
                    &session,
                    &mut slot,
                    &step,
                    &tokens[..8_192],
                    1,
                    SaveKind::Block
                ),
                Ok(true)
            );
            session.prefill(&mut step, &tokens[8_192..], LIMIT).unwrap();
            assert_eq!(save(&session, &mut slot, &step, &tokens, 1), Ok(true));
            let expected = if protected {
                vec![(8_192, false), (9_500, false)]
            } else {
                vec![(9_500, false)]
            };
            assert_eq!(slot.keys(), expected, "protected={protected}");
            assert_eq!(
                slot.save_side_totals(),
                super::SaveSideTotals {
                    room_removals: if protected { 0 } else { 1 },
                    block_rolls: 1,
                    restored_replaces: 1,
                    ..super::SaveSideTotals::default()
                }
            );
            let mut rewrite = tokens.clone();
            rewrite[8_192] = 42_000;
            session.reset(&mut step).unwrap();
            assert_eq!(
                restore(&mut session, &mut slot, &mut step, &rewrite, 1),
                Ok(if protected { 8_192 } else { 0 }),
                "protected={protected}"
            );
        }
    }

    #[test]
    fn restored_replaces_counts_the_c2_take_tip_of_the_restored_entry_once() {
        let tip = [1, 2, 3];
        let extension = [1, 2, 3, 4];
        let next = [1, 2, 3, 4, 5];
        let mut slot = store(1, 1).with_recent_longer_protection();
        let (mut step, mut session) = held(&tip);
        assert_eq!(save(&session, &mut slot, &step, &tip, 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &extension, 1),
            Ok(3)
        );
        session.prefill(&mut step, &extension[3..], LIMIT).unwrap();
        let expected = super::SaveSideTotals {
            restored_replaces: 1,
            ..super::SaveSideTotals::default()
        };
        assert!(slot.take_tip_if(&extension, &1, |_| true).is_some());
        assert_eq!(slot.save_side_totals(), expected);
        assert_eq!(save(&session, &mut slot, &step, &extension, 1), Ok(true));
        assert_eq!(slot.save_side_totals(), expected);
        // A different restored id distinguishes identity from mere restored-state presence.
        let root = [9, 10];
        let (root_step, root_session) = held(&root);
        assert_eq!(
            save_as(
                &root_session,
                &mut slot,
                &root_step,
                &root,
                1,
                SaveKind::Root
            ),
            Ok(true)
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[9, 10, 11], 1),
            Ok(2)
        );
        assert!(slot.take_tip_if(&next, &1, |_| true).is_some());
        assert_eq!(slot.save_side_totals(), expected);
    }

    #[test]
    fn taking_a_recycled_tip_requires_policy_kind_exact_prefix_path_and_ownership() {
        let key = [1, 2, 3];
        let extension = [1, 2, 3, 4];
        let divergent = [1, 2, 9, 4];
        for (enabled, kind, next, path, eligible, expected) in [
            (true, SaveKind::Tip, &extension[..], 1, true, true),
            (false, SaveKind::Tip, &extension[..], 1, true, false),
            (true, SaveKind::Block, &extension[..], 1, true, false),
            (true, SaveKind::Root, &extension[..], 1, true, false),
            (true, SaveKind::Tip, &key[..], 1, true, false),
            (true, SaveKind::Tip, &divergent[..], 1, true, false),
            (true, SaveKind::Tip, &extension[..], 2, true, false),
            (true, SaveKind::Tip, &extension[..], 1, false, false),
        ] {
            let (step, session) = held(&key);
            let mut slot = store(1, 1);
            if enabled {
                slot = slot.with_recent_longer_protection();
            }
            assert_eq!(save_as(&session, &mut slot, &step, &key, 1, kind), Ok(true));
            let keys = slot.keys();
            let taken = slot.take_tip_if(next, &path, |_| eligible);
            assert_eq!(taken.is_some(), expected);
            if expected {
                assert_eq!(taken, Some(step.fed));
                assert!(slot.keys().is_empty());
            } else {
                assert_eq!(slot.keys(), keys, "a rejected source must stay restorable");
            }
        }
    }

    #[test]
    fn a_prepared_prefix_keeps_the_original_point_after_the_live_session_resets() {
        let prompt = [1, 2, 3, 4, 5, 6];
        let key = &prompt[..4];
        let (mut step, mut session) = held(key);
        let point = step.fed.clone();
        let frozen = session.prepare_prefix_save(key, point.len()).unwrap();
        session.reset(&mut step).unwrap();
        let mut slot = store(0, 1);
        assert!(
            frozen
                .save_prefix(&mut slot, key, 7, point.len(), SaveKind::Tip, || Ok(
                    point.clone()
                ))
                .unwrap()
        );
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &prompt, 7).unwrap(),
            key.len()
        );
        assert_eq!(session.history(), key);
        assert_eq!(step.fed, point);
    }

    #[test]
    fn evictions_spend_dead_tips_before_periodic_anchors() {
        // Two lineages (paths 1 and 2), each with a periodic anchor and a longer tip of
        // the same path. Every TIP here is its branch's only deeper entry, so the tips
        // are tier 0 and go first, oldest first, and the periodic anchors - covered
        // by their tips, tier 1 - wait: a later turn restores exactly a periodic anchor
        // while a tip never matches again.
        //
        // Mutation check (round 4e SF-A, honestly scoped): a PURE LRU order drops
        // lineage 1's PERIODIC anchor first - the oldest entry - and the sequence
        // below fails. The tier discriminations this fixture CANNOT make (its
        // periodics are all tier 1, its tips all tier 0, and no key crosses paths)
        // are pinned by a_covered_interior_anchor_spends_before_an_older_anchor_leaf
        // (tier 1 vs tier 2) and a_foreign_deeper_key_of_another_path_does_not_demote_
        // the_anchor (path scoping).
        // One held session per save: a prefix save requires the session's held prompt to
        // BE the key (the pending token is the entry's).
        let (step_a, session_a) = held(&[1, 2]);
        let (step_b, session_b) = held(&[1, 2, 3, 4]);
        let (step_c, session_c) = held(&[2, 2, 2]);
        let (step_d, session_d) = held(&[2, 2, 2, 3, 3]);
        let mut slot = store(0, 4);
        save_as(
            &session_a,
            &mut slot,
            &step_a,
            &[1, 2],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        save(&session_b, &mut slot, &step_b, &[1, 2, 3, 4], 1).unwrap();
        save_as(
            &session_c,
            &mut slot,
            &step_c,
            &[2, 2, 2],
            2,
            SaveKind::Periodic,
        )
        .unwrap();
        save(&session_d, &mut slot, &step_d, &[2, 2, 2, 3, 3], 2).unwrap();
        let lens = |slot: &Slot| slot.keys().iter().map(|key| key.0).collect::<Vec<_>>();
        // No protection: the oldest leaf, lineage 1's tip, goes first. With the tip gone,
        // Dead tips first, oldest first: lineage 1's tip, then lineage 2's tip. Kind is
        // what ranks them (round 4c, R4-MF3): a lineage spends its dead tip before ANY
        // lineage's anchor; a leaf-age order would drop lineage 1's anchor second.
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens(&slot), vec![2, 3, 5]);
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens(&slot), vec![2, 3]);
        // Only now the periodic anchors, oldest first.
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens(&slot), vec![3]);
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens(&slot), Vec::<usize>::new());
        assert!(!slot.evict_lru_one(None));
        drop((
            step_a, session_a, step_b, session_b, step_c, session_c, step_d, session_d,
        ));
    }

    #[test]
    fn a_foreign_deeper_key_of_another_path_does_not_demote_the_anchor() {
        // Round 4d SF-4: the two-path discriminator. Path 1 holds a dead tip and - the
        // YOUNGEST entry - an anchor [1,2]; path 2 holds an OLDER anchor [1,2,3] that
        // extends [1,2] across paths. Extension counting is PATH-SCOPED, so [1,2] has
        // no same-path extender: it is its lineage's anchor leaf (tier 2) and - being
        // the youngest of the tier - survives while the older path-2 anchor spends.
        // A foreign deeper key must not change what a lineage's own tree owes it.
        //
        // Mutation check: dropping the path comparison in `branches_of` counts the
        // foreign [1,2,3] as an extender, demotes [1,2] to a redundant interior
        // (tier 1), and spends it SECOND: the keys assertion [3] fails - the withdrawn
        // ancestor-first order's exact trap.
        let (step_b, session_b) = held(&[1, 2, 3]);
        let (tip_step, tip) = held(&[5, 5, 9]);
        let (step_a, session_a) = held(&[1, 2]);
        let mut slot = store(0, 3);
        save_as(
            &session_b,
            &mut slot,
            &step_b,
            &[1, 2, 3],
            2,
            SaveKind::Periodic,
        )
        .unwrap();
        save(&tip, &mut slot, &tip_step, &[5, 5, 9], 1).unwrap();
        save_as(
            &session_a,
            &mut slot,
            &step_a,
            &[1, 2],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        // The dead tip first (tier 0).
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens_of(&slot), vec![3, 2]);
        // The older ANCHOR LEAF, not the cross-path-"covered" younger one.
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens_of(&slot), vec![2]);
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens_of(&slot), Vec::<usize>::new());
        drop((step_b, session_b, tip_step, tip, step_a, session_a));
    }

    #[test]
    fn a_covered_interior_anchor_spends_before_an_older_anchor_leaf() {
        // Round 4e (Opus SF-A/B-i, Sol r4 SF1): the tier discriminator. Lineage B's
        // anchor [9,9] is saved FIRST - the OLDEST entry - and is its lineage's only
        // entry: an anchor LEAF (tier 2). Lineage A holds a CHAIN {per@3, per@4}, so
        // per@3 is a redundant interior (tier 1: one maximal extender, per@4) and
        // per@4 - with its cover gone after per@3 spends - is A's anchor leaf (tier 2).
        // The real order: per@3 FIRST (tier 1 beats tier 2 even though B is OLDER),
        // then B, then per@4. Distinct lengths keep every step unambiguous.
        //
        // Mutation checks: collapsing tiers 1 and 2 to one age order spends B first
        // (it is the oldest) and the first assertion fails; a pure-LRU order does the
        // same; and after per@3 is gone the recomputed order still ranks B (older)
        // ahead of per@4, so the middle assertion pins tier-2 AGE order too.
        let (b_step, b) = held(&[9, 9]);
        let (per3_step, per3) = held(&[1, 2, 3]);
        let (per4_step, per4) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 3);
        save_as(&b, &mut slot, &b_step, &[9, 9], 1, SaveKind::Periodic).unwrap();
        save_as(
            &per3,
            &mut slot,
            &per3_step,
            &[1, 2, 3],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        save_as(
            &per4,
            &mut slot,
            &per4_step,
            &[1, 2, 3, 4],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        // Tier 1 beats tier 2 regardless of age: A's redundant per@3 goes FIRST,
        // before the OLDER lineage-B anchor.
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens_of(&slot), vec![2, 4], "the interior went first");
        // Tier 2, oldest first: B before A's now-uncovered per@4.
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens_of(&slot), vec![4], "the older anchor leaf went second");
        assert!(slot.evict_lru_one(None));
        assert_eq!(lens_of(&slot), Vec::<usize>::new());
        drop((b_step, b, per3_step, per3, per4_step, per4));
    }

    #[test]
    fn an_eviction_protects_the_denied_lineage_not_the_last_restore() {
        // The interleave of review round 4, MF2: lineage A saves its anchor, lineage B
        // saves its own, and B restores - so the store-global `restored` marker names B.
        // A's denial must protect A's anchor, not the marker's: the eviction with
        // `protect` = A's entry drops B's and keeps A's.
        //
        // Mutation check: filtering on the store-global `restored` marker instead of the
        // passed lineage protects B and drops A, so the keys assertion fails; deleting
        // the protect filter entirely spends whichever entry is oldest, which is A's.
        // Distinct lengths (round 4c, SF-b): lens==[2] cannot tell A from B.
        let (step, session) = held(&[1, 2]);
        let (mut other_step, mut other) = held(&[2, 2, 2]);
        let mut slot = store(0, 2);
        save_as(&session, &mut slot, &step, &[1, 2], 1, SaveKind::Periodic).unwrap();
        save_as(
            &other,
            &mut slot,
            &other_step,
            &[2, 2, 2],
            2,
            SaveKind::Periodic,
        )
        .unwrap();
        let a_id = slot.lru[0].id;
        let b_id = slot.lru[1].id;
        // B restores: the entry moves to the MRU end and the store marker names B.
        assert_eq!(
            restore(&mut other, &mut slot, &mut other_step, &[2, 2, 2, 9], 2).unwrap(),
            3
        );
        assert_eq!(slot.restored, Some(b_id));
        assert_ne!(slot.restored, Some(a_id));
        // A's denial, protecting its own anchor: B's entry goes, and the marker it held
        // is cleared rather than left dangling.
        assert!(slot.evict_lru_one(Some(a_id)));
        let lens = |slot: &Slot| slot.keys().iter().map(|key| key.0).collect::<Vec<_>>();
        // B's 3-token anchor went: the surviving length names A unambiguously.
        assert_eq!(lens(&slot), vec![2]);
        let survivor = slot.lru[0].id;
        assert_eq!(survivor, a_id);
        assert_ne!(survivor, b_id);
        assert_eq!(slot.restored, None);
        drop((step, session, other_step, other));
    }

    #[test]
    fn a_protected_entry_is_the_last_resort_victim() {
        // One entry, and it is the denied job's own anchor: the eviction takes it - the
        // pre-incremental semantics at the limit - and clears the restored marker.
        let (mut step, mut session) = held(&[1, 2]);
        let mut slot = store(0, 1);
        save_as(&session, &mut slot, &step, &[1, 2], 1, SaveKind::Periodic).unwrap();
        let id = slot.lru[0].id;
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 9], 1).unwrap(),
            2
        );
        assert!(slot.evict_lru_one(Some(id)));
        assert_eq!(lens_of(&slot), Vec::<usize>::new());
        assert_eq!(slot.restored, None);
        assert!(!slot.evict_lru_one(Some(id)));
    }

    fn lens_of(slot: &Slot) -> Vec<usize> {
        slot.keys().iter().map(|key| key.0).collect()
    }

    #[test]
    fn save_side_totals_count_duplicate_uses_and_block_rolls() {
        use super::SaveSideTotals;

        // The two save-side counters the sched pins leave at zero: a repeated same-key
        // save of the same kind and path counts one duplicate use (a use, not a second
        // snapshot), and a Block cut that a later Block of the same path extends rolls
        // the stored cut away (the KV-v1 rolling lineage cut).
        let mut slot = store(0, 6);
        let path = 7u8;
        let a = vec![2u32; 8];
        let mut b = a.clone();
        b.extend_from_slice(&[9, 9]);

        let (step, session) = held(&a);
        assert!(save_as(&session, &mut slot, &step, &a, path, SaveKind::Block).unwrap());
        assert!(save_as(&session, &mut slot, &step, &a, path, SaveKind::Block).unwrap());
        assert_eq!(
            slot.save_side_totals(),
            SaveSideTotals {
                duplicate_uses: 1,
                ..SaveSideTotals::default()
            },
            "the repeated key is a use, not a second snapshot"
        );

        let (step_b, session_b) = held(&b);
        assert!(save_as(&session_b, &mut slot, &step_b, &b, path, SaveKind::Block).unwrap());
        assert_eq!(
            slot.save_side_totals(),
            SaveSideTotals {
                duplicate_uses: 1,
                block_rolls: 1,
                ..SaveSideTotals::default()
            },
            "the extending Block rolls the stored cut"
        );
        assert_eq!(
            slot.keys(),
            vec![(b.len(), false)],
            "only the rolled-in cut stays"
        );
    }

    #[test]
    fn a_root_promotion_that_spends_an_unrelated_root_counts_a_room_removal() {
        use super::SaveSideTotals;

        // Sol round-2 hole: promoting an LRU entry to a full root set drops the least
        // recently used other root. That lost root is room churn exactly like the room
        // loop's spends, so it must count - not hide behind the duplicate-use counter.
        let mut slot = store(1, 1);
        let (step_a, session_a) = held(&[2, 2, 2, 2]);
        assert!(
            save_as(
                &session_a,
                &mut slot,
                &step_a,
                &[2, 2, 2, 2],
                1,
                SaveKind::Root
            )
            .unwrap()
        );
        let (step_b, session_b) = held(&[3, 3, 3, 3]);
        assert!(
            save_as(
                &session_b,
                &mut slot,
                &step_b,
                &[3, 3, 3, 3],
                1,
                SaveKind::Tip
            )
            .unwrap()
        );
        // The same key saved as a Root promotes the tip; the root set is full, so the
        // promotion spends A's root.
        assert!(
            save_as(
                &session_b,
                &mut slot,
                &step_b,
                &[3, 3, 3, 3],
                1,
                SaveKind::Root
            )
            .unwrap()
        );
        assert_eq!(
            slot.save_side_totals(),
            SaveSideTotals {
                duplicate_uses: 1,
                room_removals: 1,
                ..SaveSideTotals::default()
            },
            "the promotion's victim root is a room removal, not an invisible loss"
        );
    }

    #[test]
    fn a_restored_prefix_continues_like_a_full_prefill() {
        let prompt = [1, 2, 3, 4, 5, 6];
        let (mut whole, mut full) = held(&prompt);
        let (mut step, mut session) = held(&prompt[..4]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &prompt[..4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        let covered = restore(&mut session, &mut slot, &mut step, &prompt, 1).unwrap();
        assert_eq!(covered, 4, "covered tokens");
        assert_eq!(session.history(), &prompt[..4], "restored history");
        assert_eq!(session.prompt_len(), 4, "restored prompt length");
        assert_eq!(session.pending(), Some(4), "restored pending token");
        assert!(!session.is_finished(), "finished");
        // Three of the four key tokens are fed. The fourth is pending.
        assert_eq!(step.fed, vec![1, 2, 3], "fed tokens after the restore");
        session.prefill(&mut step, &prompt[4..], LIMIT).unwrap();
        assert_eq!(
            step.fed, whole.fed,
            "fed tokens after the rest of the prompt"
        );
        assert_eq!(session.history(), full.history(), "history");
        assert_eq!(session.prompt_len(), 6, "prompt length");
        assert_eq!(session.pending(), Some(6), "pending token");
        let reused = decode_all(
            &mut session,
            &mut step,
            &mut generation(),
            &limits(&EOS_NONE),
        );
        let fresh = decode_all(&mut full, &mut whole, &mut generation(), &limits(&EOS_NONE));
        assert_eq!(
            tokens_of(&reused.batches),
            tokens_of(&fresh.batches),
            "tokens"
        );
    }

    #[test]
    fn a_restore_replaces_a_finished_session() {
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        let mut ended = generation().with_max_tokens(0);
        let batch = session
            .decode(&mut step, &mut ended, &limits(&EOS_NONE))
            .unwrap();
        assert_eq!(batch.finish, Some(DecodeFinish::Length), "finish");
        assert!(session.is_finished(), "the job ended");
        let covered = restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 9], 1).unwrap();
        assert_eq!(covered, 3, "covered tokens");
        assert!(!session.is_finished(), "a restored session is not finished");
        assert_eq!(session.pending(), Some(3), "pending token");
    }

    /// Asserts that a save is rejected as `BadRequest`, that the snapshot closure did not run, and
    /// that the entry of key length 3 is still stored.
    fn assert_save_rejected(
        session: &DecodeSession,
        step: &Fake,
        slot: &mut Slot,
        key: &[u32],
        what: &str,
    ) {
        let ran = std::cell::Cell::new(false);
        let result = session.save_prefix(slot, key, 1, step.fed(), SaveKind::Tip, || {
            ran.set(true);
            Ok(Vec::new())
        });
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "{what}: {result:?}"
        );
        assert!(!ran.get(), "{what}: the snapshot ran");
        assert_eq!(slot.key_len(), Some(3), "{what}: the earlier entry");
    }

    #[test]
    fn a_save_needs_a_pending_token_and_the_held_prompt() {
        let empty = DecodeSession::default();
        let mut none = Slot::default();
        let early = empty.save_prefix(&mut none, &[], 1, 0, SaveKind::Tip, || Ok(Vec::new()));
        assert!(matches!(early, Err(EngError::BadRequest(_))), "{early:?}");
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        assert_save_rejected(&session, &step, &mut slot, &[1, 2], "a shorter key");
        assert_save_rejected(&session, &step, &mut slot, &[1, 2, 9], "a divergent key");
        assert_save_rejected(&session, &step, &mut slot, &[1, 2, 3, 4], "a longer key");
        // The session now holds four tokens. The old key of three tokens is not the held prompt.
        session.prefill(&mut step, &[4], LIMIT).unwrap();
        assert_save_rejected(&session, &step, &mut slot, &[1, 2, 3], "a stale key");
        // A decoded token joins the history but not the held prompt, so even the full history
        // is rejected: only a prefill produces an entry.
        session
            .decode(&mut step, &mut generation(), &limits(&EOS_NONE))
            .unwrap();
        let history = session.history().to_vec();
        assert_eq!(history.len(), 5, "history after one decoded token");
        assert_save_rejected(
            &session,
            &step,
            &mut slot,
            &history,
            "a key with a decoded token",
        );
    }

    #[test]
    fn a_finished_session_cannot_be_saved() {
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        let mut ended = generation().with_max_tokens(0);
        session
            .decode(&mut step, &mut ended, &limits(&EOS_NONE))
            .unwrap();
        assert!(session.is_finished(), "the job ended");
        // The history still equals the key, so only the finish flag rejects the save.
        let result = save(&session, &mut slot, &step, &[1, 2, 3], 1);
        assert!(matches!(result, Err(EngError::BadRequest(_))), "{result:?}");
        assert_eq!(slot.key_len(), None, "no entry");
    }

    #[test]
    fn a_model_that_is_not_at_the_position_of_the_key_cannot_be_saved() {
        // The session holds the key [1, 2, 3], so the model must have fed 2 positions.
        for (name, fed) in [("a model ahead of the key", 3), ("a model behind it", 1)] {
            let (mut step, session) = held(&[1, 2, 3]);
            step.fed.resize(fed, 9);
            let mut slot = Slot::default();
            let ran = std::cell::Cell::new(false);
            let result =
                session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), SaveKind::Tip, || {
                    ran.set(true);
                    Ok(Vec::new())
                });
            assert!(
                matches!(result, Err(EngError::BadRequest(_))),
                "{name}: {result:?}"
            );
            assert!(!ran.get(), "{name}: the snapshot ran");
            assert_eq!(slot.key_len(), None, "{name}: no entry");
        }
    }

    #[test]
    fn a_save_after_a_failed_prefill_is_refused_and_creates_no_entry() {
        // `fail_feed_on` 3 fails on the first row, the pending token, so the model stays at 2
        // positions. 5 fails on the third row, after two rows reached the model.
        for (name, fail_on) in [("the first row fails", 3), ("a later row fails", 5)] {
            let (mut step, mut session) = held(&[1, 2, 3]);
            step.fail_feed_on = Some(fail_on);
            let failed = session.prefill(&mut step, &[4, 5, 6], LIMIT);
            assert!(failed.is_err(), "{name}: the prefill must fail");
            assert_eq!(session.history(), [1, 2, 3], "{name}: history");
            let mut slot = Slot::default();
            let result = save(&session, &mut slot, &step, &[1, 2, 3], 1);
            assert!(
                matches!(result, Err(EngError::BadRequest(_))),
                "{name}: {result:?}"
            );
            assert_eq!(slot.key_len(), None, "{name}: no entry");
        }
    }

    #[test]
    fn a_saved_prefix_restores_again_for_every_job_that_extends_it() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        for job in 1..=3 {
            session.reset(&mut step).unwrap();
            let covered = restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1);
            assert_eq!(covered, Ok(4), "job {job}: covered tokens");
            assert_eq!(session.history(), [1, 2, 3, 4], "job {job}: history");
            assert_eq!(session.pending(), Some(4), "job {job}: pending token");
            assert_eq!(slot.key_len(), Some(4), "job {job}: the entry stays");
        }
    }

    fn backend_error(message: &str) -> EngError {
        EngError::Backend {
            operation: "fake",
            message: message.to_string(),
        }
    }

    /// A snapshot that counts the copies alive.
    struct Live(std::rc::Rc<std::cell::Cell<usize>>);

    impl Live {
        fn new(count: &std::rc::Rc<std::cell::Cell<usize>>) -> Self {
            count.set(count.get() + 1);
            Self(std::rc::Rc::clone(count))
        }
    }

    impl Drop for Live {
        fn drop(&mut self) {
            self.0.set(self.0.get() - 1);
        }
    }

    #[test]
    fn a_save_drops_the_earlier_entry_before_it_takes_the_snapshot() {
        let count = std::rc::Rc::new(std::cell::Cell::new(0));
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot: PrefixStore<Live, u8> = PrefixStore::default();
        let first =
            session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), SaveKind::Tip, || {
                Ok(Live::new(&count))
            });
        assert_eq!(first, Ok(true), "first save");
        assert_eq!(count.get(), 1, "one snapshot alive");
        session.prefill(&mut step, &[4, 5], LIMIT).unwrap();
        let alive_during = std::cell::Cell::new(usize::MAX);
        let second = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || {
                alive_during.set(count.get());
                Ok(Live::new(&count))
            },
        );
        assert_eq!(second, Ok(true), "second save");
        assert_eq!(
            alive_during.get(),
            0,
            "snapshots alive when the new one is taken"
        );
        assert_eq!(count.get(), 1, "snapshots alive after the second save");
        assert_eq!(slot.key_len(), Some(5), "the entry holds the new key");
    }

    #[test]
    fn a_denied_snapshot_returns_false_and_leaves_the_slot_empty() {
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        // A save of a new key drops the earlier entry before the snapshot runs, so a denial
        // leaves the store empty (a save of the same key is a touch and takes no snapshot).
        session.prefill(&mut step, &[4, 5], LIMIT).unwrap();
        let denied = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || Err(EngError::MemoryPlan("no room".to_string())),
        );
        assert_eq!(denied, Ok(false), "a denied charge reports false");
        assert_eq!(slot.key_len(), None, "no entry after a denial");
        assert_eq!(
            session.history(),
            [1, 2, 3, 4, 5],
            "the session is unchanged"
        );
    }

    #[test]
    fn a_snapshot_error_other_than_memory_plan_propagates_and_leaves_the_slot_empty() {
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        // A save of a new key drops the earlier entry before the snapshot runs, so the failure
        // leaves the store empty.
        session.prefill(&mut step, &[4, 5], LIMIT).unwrap();
        let failed = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || Err(backend_error("snapshot failed")),
        );
        assert_eq!(failed, Err(backend_error("snapshot failed")), "the error");
        assert_eq!(slot.key_len(), None, "no entry after a failure");
    }

    #[test]
    fn a_prompt_that_does_not_extend_the_key_restores_nothing() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        let cases: [(&str, &[u32]); 3] = [
            ("equal prompt", &[1, 2, 3, 4]),
            ("shorter prompt", &[1, 2, 3]),
            ("divergent prompt", &[1, 2, 9, 4, 5]),
        ];
        for (name, prompt) in cases {
            let ran = std::cell::Cell::new(false);
            let covered = session.restore_prefix(&mut slot, prompt, 1, |_| {
                ran.set(true);
                Ok(())
            });
            assert_eq!(covered, Ok(0), "{name}: covered tokens");
            assert!(!ran.get(), "{name}: the model state was restored");
            assert!(session.history().is_empty(), "{name}: history");
            assert_eq!(session.pending(), None, "{name}: pending token");
            assert_eq!(slot.key_len(), Some(4), "{name}: a miss keeps the entry");
        }
    }

    #[test]
    fn an_empty_slot_restores_nothing() {
        let mut session = DecodeSession::default();
        let mut slot = Slot::default();
        let covered = session.restore_prefix(&mut slot, &[1, 2], 1, |_| {
            panic!("the model state must not be restored")
        });
        assert_eq!(covered, Ok(0));
    }

    #[test]
    fn a_saved_prefix_restores_only_under_the_numeric_path_that_built_it() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        let other = restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 2);
        assert_eq!(other, Ok(0), "covered tokens under another path");
        assert!(session.history().is_empty(), "history");
        assert_eq!(slot.key_len(), Some(4), "the entry stays for its own path");
        let same = restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1);
        assert_eq!(same, Ok(4), "covered tokens under the same path");
    }

    #[test]
    fn a_restore_denied_by_the_memory_plan_drops_the_entry_and_changes_nothing() {
        let (step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        // The worker resets first. This test skips the reset to show that a denial leaves the
        // tokens of the session as they were.
        let denied = session.restore_prefix(&mut slot, &[1, 2, 3, 4, 5], 1, |_| {
            Err(EngError::MemoryPlan("no room".to_string()))
        });
        assert_eq!(denied, Ok(0), "covered tokens");
        assert_eq!(slot.key_len(), None, "the denied restore drops the entry");
        assert_eq!(session.history(), [1, 2, 3, 4], "history");
        assert_eq!(session.pending(), Some(4), "pending token");
    }

    #[test]
    fn an_adoption_denial_evicts_one_victim_and_retries_the_same_candidate() {
        // SF1 (the next slice), RED from the override-16 trace: request 0x18's periodic
        // anchor at 49,152 was the candidate its 90K turn restores, and the adoption
        // MemoryPlan denial dropped it - the entry that would have made t6 a 235 s
        // restore instead of a 410 s re-prefill. The fix mirrors the save path's step
        // (d): deny -> evict ONE victim (the candidate itself protected, leaf-first) ->
        // retry the SAME candidate; only a second denial drops it. Here the closure is
        // the budget: it denies ONCE (the pool was one victim short) and succeeds after,
        // and the older spendable leaf of the same path is the victim.
        //
        // RED on the base: the drop happens without any eviction, the retry never runs,
        // and the restore covers 0 while the anchor is GONE.
        let (old_step, old) = held(&[1, 2]);
        let (tip_step, tip) = held(&[1, 2, 3, 4]);
        let (anchor_step, mut anchor) = held(&[1, 2, 9]);
        let mut slot = store(0, 3);
        save_as(&old, &mut slot, &old_step, &[1, 2], 1, SaveKind::Periodic).unwrap();
        save(&tip, &mut slot, &tip_step, &[1, 2, 3, 4], 1).unwrap();
        save_as(
            &anchor,
            &mut slot,
            &anchor_step,
            &[1, 2, 9],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        let mut denials = 0;
        let covered = anchor.restore_prefix(&mut slot, &[1, 2, 9, 9], 1, |_| {
            denials += 1;
            if denials == 1 {
                Err(EngError::MemoryPlan("one victim short".to_string()))
            } else {
                Ok(())
            }
        });
        // The retried adoption covers the anchor's three tokens.
        assert_eq!(covered, Ok(3), "the retried candidate restores");
        // The eviction spent the kind-order victim - the dead TIP [1,2,3,4], tier 0 -
        // while the retried candidate survives: one victim, exactly.
        assert_eq!(lens_of(&slot), vec![2, 3], "one victim, the dead tip");
        drop((old_step, old, tip_step, tip, anchor_step, anchor));
    }

    #[test]
    fn a_singleton_candidate_denied_adoption_is_dropped_and_counted() {
        // SF1-r1 MF1: with no other victim, EV-1's tier-4 last resort spends the
        // candidate ITSELF. That is the legacy drop-and-next outcome - the anchor
        // died to its own denial - so `restore_drops` must count it; a silent
        // disappearance would report zero drops and misattribute the replay miss.
        let (step, mut session) = held(&[1, 2]);
        let mut slot = store(0, 3);
        save_as(&session, &mut slot, &step, &[1, 2], 1, SaveKind::Periodic).unwrap();
        let mut denials = 0;
        let covered = session.restore_prefix(&mut slot, &[1, 2, 3], 1, |_| {
            denials += 1;
            Err(EngError::MemoryPlan("nothing to free".to_string()))
        });
        assert_eq!(covered, Ok(0), "no candidate survives");
        assert!(slot.key_len().is_none(), "the singleton is gone");
        assert_eq!(slot.restore_drop_count(), 1, "the drop is counted");
        assert_eq!(
            slot.eviction_count(),
            1,
            "spent through the last-resort eviction"
        );
        drop((step, session));
    }

    #[test]
    fn a_persistent_denial_spends_one_victim_then_drops_the_candidate() {
        // SF1-r1 SF2: denial -> one victim -> SAME candidate retried -> denied AGAIN
        // -> the legacy drop-and-next: the candidate leaves, the next (shorter)
        // candidate restores, and both counters tell the story. Exactly two adoption
        // attempts on the first candidate, never more.
        let (victim_step, victim) = held(&[5, 6]);
        let (anchor_step, mut anchor) = held(&[1, 2, 9]);
        let (short_step, short) = held(&[1, 2]);
        let mut slot = store(0, 6);
        save(&victim, &mut slot, &victim_step, &[5, 6], 1).unwrap();
        save_as(
            &anchor,
            &mut slot,
            &anchor_step,
            &[1, 2, 9],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        save_as(
            &short,
            &mut slot,
            &short_step,
            &[1, 2],
            1,
            SaveKind::Periodic,
        )
        .unwrap();
        let mut denials = 0;
        let covered = anchor.restore_prefix(&mut slot, &[1, 2, 9, 9], 1, |_| {
            denials += 1;
            if denials <= 2 {
                Err(EngError::MemoryPlan("persistent".to_string()))
            } else {
                Ok(())
            }
        });
        assert_eq!(covered, Ok(2), "the fallback candidate restores");
        assert_eq!(
            denials, 3,
            "two attempts on the candidate, then the fallback"
        );
        assert_eq!(
            lens_of(&slot),
            vec![2],
            "the victim and the dropped candidate are both gone"
        );
        assert_eq!(slot.restore_drop_count(), 1, "the candidate's drop");
        assert_eq!(slot.eviction_count(), 1, "exactly one victim was spent");
        drop((victim_step, victim, anchor_step, anchor, short_step, short));
    }

    #[test]
    fn a_restore_error_other_than_memory_plan_propagates_and_changes_nothing() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        let failed = session.restore_prefix(&mut slot, &[1, 2, 3, 4, 5], 1, |_| {
            Err(backend_error("restore failed"))
        });
        assert_eq!(failed, Err(backend_error("restore failed")), "the error");
        assert!(session.history().is_empty(), "history");
        assert_eq!(session.pending(), None, "pending token");
        assert_eq!(slot.key_len(), Some(4), "the entry stays");
    }

    #[test]
    fn the_longest_stored_key_wins() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(2, 6);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the root of 4 tokens"
        );
        session.prefill(&mut step, &[5, 6, 7, 8], LIMIT).unwrap();
        assert_eq!(
            save_as(
                &session,
                &mut slot,
                &step,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1,
                SaveKind::Tip
            ),
            Ok(true),
            "the tip of 8 tokens"
        );
        // A prompt that extends the tip restores the tip: 8 of its 9 tokens are covered.
        session.reset(&mut step).unwrap();
        let extends_tip = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &extends_tip, 1).unwrap(),
            8,
            "the tip covers 8"
        );
        // A prompt that diverges after the root restores the root: 4 of its 7 tokens.
        session.reset(&mut step).unwrap();
        let extends_root = [1, 2, 3, 4, 9, 10, 11];
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &extends_root, 1).unwrap(),
            4,
            "the root covers 4"
        );
        assert_eq!(session.history(), [1, 2, 3, 4], "the root history");
    }

    #[test]
    fn a_second_conversation_with_the_same_system_prompt_restores_the_root() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 6);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true)
        );
        session.prefill(&mut step, &[5, 6, 7, 8], LIMIT).unwrap();
        assert_eq!(
            save_as(
                &session,
                &mut slot,
                &step,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1,
                SaveKind::Tip
            ),
            Ok(true)
        );
        // A second conversation: the same 4 system tokens, a different continuation. The tip of
        // the first conversation is no prefix of it, so the root restores.
        session.reset(&mut step).unwrap();
        let second = [1, 2, 3, 4, 9, 9, 9, 9, 9];
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &second, 1).unwrap(),
            4,
            "the root covers the system prompt"
        );
        assert_eq!(
            session.pending(),
            Some(4),
            "the pending token is the root's"
        );
    }

    #[test]
    fn the_least_recently_used_entry_is_dropped_first() {
        let mut slot = store(0, 3);
        let (mut step_a, mut a) = held(&[1, 2, 3]);
        let (step_b, b) = held(&[4, 5, 6, 7]);
        let (step_c, c) = held(&[8, 9]);
        assert_eq!(save(&a, &mut slot, &step_a, &[1, 2, 3], 1), Ok(true));
        assert_eq!(save(&b, &mut slot, &step_b, &[4, 5, 6, 7], 1), Ok(true));
        assert_eq!(save(&c, &mut slot, &step_c, &[8, 9], 1), Ok(true));
        assert_eq!(
            slot.keys(),
            vec![(3, false), (4, false), (2, false)],
            "least to most recently used"
        );
        // A restore is a use: the first entry becomes the most recently used one.
        assert_eq!(
            restore(&mut a, &mut slot, &mut step_a, &[1, 2, 3, 4], 1).unwrap(),
            3,
            "the first entry restores"
        );
        assert_eq!(
            slot.keys(),
            vec![(4, false), (2, false), (3, false)],
            "the restore made the first entry the most recently used"
        );
        // The fourth save drops the least recently used entry: the second one.
        let (step_d, d) = held(&[3, 3, 3, 3, 3]);
        assert_eq!(save(&d, &mut slot, &step_d, &[3, 3, 3, 3, 3], 1), Ok(true));
        assert_eq!(
            slot.keys(),
            vec![(2, false), (3, false), (5, false)],
            "the second is gone; the first, third, and fourth remain"
        );
    }

    #[test]
    fn a_pinned_root_survives_every_lru_drop() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 2);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the root"
        );
        for last in 5..=9u32 {
            session.prefill(&mut step, &[last], LIMIT).unwrap();
            let key: Vec<u32> = (1..=last).collect();
            assert_eq!(
                save_as(&session, &mut slot, &step, &key, 1, SaveKind::Tip),
                Ok(true),
                "the tip that ends in {last}"
            );
        }
        assert_eq!(slot.roots_len(), 1, "the root survived five tip saves");
        assert_eq!(slot.entries_len(), 2, "only the last two tips remain");
        // A prompt that shares the root but no tip restores the root.
        session.reset(&mut step).unwrap();
        let other = [1, 2, 3, 4, 9, 9, 9];
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &other, 1).unwrap(),
            4,
            "the root still restores"
        );
        assert_eq!(session.history(), [1, 2, 3, 4], "the root history");
    }

    #[test]
    fn a_tip_that_extends_the_restored_entry_replaces_it() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 6);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "the entry restores"
        );
        // The next turn extends the restored entry, so the tip replaces it.
        session.prefill(&mut step, &[5], LIMIT).unwrap();
        assert_eq!(
            save(&session, &mut slot, &step, &[1, 2, 3, 4, 5], 1),
            Ok(true)
        );
        assert_eq!(
            slot.keys(),
            vec![(5, false)],
            "the restored entry is gone; the new key is present"
        );
    }

    #[test]
    fn a_denied_tip_keeps_the_restored_entry() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 6);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "the entry restores"
        );
        session.prefill(&mut step, &[5], LIMIT).unwrap();
        let denied = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || Err(EngError::MemoryPlan("no room".to_string())),
        );
        assert_eq!(denied, Ok(false), "the denied save reports false");
        assert_eq!(
            slot.keys(),
            vec![(4, false)],
            "the restored entry is protected from the denial"
        );
    }

    /// A store with limits `(0, 1)` after a restore: the tip of 4 tokens was saved, a job
    /// restored it for the prompt `[1, 2, 3, 4, 5]`, and the job prefilled the fifth token.
    /// Returns the session and the fake that hold the state of the key `[1, 2, 3, 4, 5]`.
    fn restored_one_entry_store() -> (Fake, DecodeSession, Slot) {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 1);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "the entry restores"
        );
        session.prefill(&mut step, &[5], LIMIT).unwrap();
        (step, session, slot)
    }

    /// With the default limits (no root, one entry) a tip that extends the restored entry
    /// replaces it (clarification 1, item 1). This is the normal flow of every worker job.
    ///
    /// Mutation check: removing the last-resort drop of the restored entry in the room step makes
    /// the save return `Ok(false)` and keeps the old key, so both assertions fail.
    #[test]
    fn a_tip_replaces_the_restored_entry_when_it_is_the_only_one() {
        let (step, session, mut slot) = restored_one_entry_store();
        let calls = std::cell::Cell::new(0usize);
        let saved = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || {
                calls.set(calls.get() + 1);
                Ok(step.fed.clone())
            },
        );
        assert_eq!(saved, Ok(true), "the tip is saved");
        assert_eq!(calls.get(), 1, "one snapshot");
        assert_eq!(
            slot.keys(),
            vec![(5, false)],
            "the entry of 4 tokens is gone; the entry of 5 tokens is stored"
        );
    }

    /// The restored entry goes before the snapshot, so a denied snapshot leaves the store empty
    /// (the behavior of decision 0036, addendum 1).
    ///
    /// Mutation check: keeping the restored entry in the room step makes the save return
    /// `Ok(false)` without a snapshot call and leaves one entry, so the call count and the key
    /// list assertions fail.
    #[test]
    fn a_denied_tip_leaves_the_store_empty_when_the_restored_entry_was_the_only_one() {
        let (step, session, mut slot) = restored_one_entry_store();
        let calls = std::cell::Cell::new(0usize);
        let denied = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || {
                calls.set(calls.get() + 1);
                Err(EngError::MemoryPlan("no room".to_string()))
            },
        );
        assert_eq!(denied, Ok(false), "the denied save reports false");
        assert_eq!(calls.get(), 1, "the snapshot ran once");
        assert!(
            slot.keys().is_empty(),
            "the store is empty: {:?}",
            slot.keys()
        );
    }

    /// A store with `(roots, entries)` limits after a restore of an entry of `kind`: a job saved
    /// the key `[1, 2, 3, 4]` as `kind`, a second job restored it for the prompt `[1, 2, 3, 4, 5]`
    /// and prefilled the fifth token, and the job then saves the tip `[1, 2, 3, 4, 5]`. Returns
    /// the result of the tip save, the number of snapshots that the tip save took, and the store.
    fn tip_after_a_restore_of(
        roots: usize,
        entries: usize,
        kind: SaveKind,
    ) -> (Result<bool>, usize, Slot) {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(roots, entries);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, kind),
            Ok(true),
            "the entry of 4 tokens is stored"
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "the entry restores"
        );
        session.prefill(&mut step, &[5], LIMIT).unwrap();
        let calls = std::cell::Cell::new(0usize);
        let saved = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Tip,
            || {
                calls.set(calls.get() + 1);
                Ok(step.fed.clone())
            },
        );
        (saved, calls.get(), slot)
    }

    /// Rule 4 of addendum 2: a periodic entry is never dropped by the lineage rule. The tip of 5
    /// tokens extends the restored periodic entry of 4 tokens, and the list has room for both.
    ///
    /// Mutation check: removing the `Tip` kind test from `drop_restored_ancestor` drops the
    /// periodic entry, so the key list is `[(5, false)]`.
    #[test]
    fn a_restored_periodic_entry_survives_a_tip_that_extends_it() {
        let (saved, calls, slot) = tip_after_a_restore_of(0, 2, SaveKind::Periodic);
        assert_eq!(saved, Ok(true), "the tip is saved");
        assert_eq!(calls, 1, "one snapshot");
        // The restore made the periodic entry the most recently used one; the tip is newer.
        assert_eq!(
            slot.keys(),
            vec![(4, false), (5, false)],
            "the periodic entry of 4 tokens and the tip of 5 tokens"
        );
    }

    /// A full list drops the other entry for the room, the restored periodic entry stays, and the
    /// lineage rule does not take it afterwards.
    ///
    /// Mutation check: removing the `Tip` kind test from `drop_restored_ancestor` drops the
    /// periodic entry after the save, so the key list is `[(4, false)]` without the periodic
    /// entry of 3 tokens.
    #[test]
    fn a_full_list_keeps_a_restored_periodic_entry_and_drops_the_other_entry() {
        let (mut step_a, mut a) = held(&[1, 2, 3]);
        let (step_b, b) = held(&[5, 6, 7, 8, 9]);
        let mut slot = store(0, 2);
        assert_eq!(
            save_as(&a, &mut slot, &step_a, &[1, 2, 3], 1, SaveKind::Periodic),
            Ok(true),
            "the periodic entry A"
        );
        assert_eq!(save(&b, &mut slot, &step_b, &[5, 6, 7, 8, 9], 1), Ok(true));
        a.reset(&mut step_a).unwrap();
        assert_eq!(
            restore(&mut a, &mut slot, &mut step_a, &[1, 2, 3, 4], 1).unwrap(),
            3,
            "A restores"
        );
        a.prefill(&mut step_a, &[4], LIMIT).unwrap();
        assert_eq!(
            save(&a, &mut slot, &step_a, &[1, 2, 3, 4], 1),
            Ok(true),
            "the tip is saved"
        );
        assert_eq!(
            slot.keys(),
            vec![(3, false), (4, false)],
            "B went in the room step; A (3 tokens) stays beside the tip (4 tokens)"
        );
    }

    /// The last resort of clarification 1 drops a restored entry of kind `Tip` only. A restored
    /// periodic entry in a one-entry list is outside the rules of the worker (it emits no
    /// periodic point below two entries); the store keeps the entry and skips the tip save.
    ///
    /// Mutation check: removing the `Tip` kind test from `drop_restored_ancestor` replaces the
    /// periodic entry, so the call count is 1 and the key list is `[(5, false)]`.
    #[test]
    fn the_last_resort_does_not_drop_a_restored_periodic_entry() {
        let (saved, calls, slot) = tip_after_a_restore_of(0, 1, SaveKind::Periodic);
        assert_eq!(saved, Ok(false), "the tip save is skipped");
        assert_eq!(calls, 0, "no snapshot");
        assert_eq!(slot.keys(), vec![(4, false)], "the periodic entry stays");
    }

    /// A restored root survives a tip that extends it: roots are not in the LRU list, so the
    /// lineage rule cannot reach them. The tip takes the one LRU place.
    #[test]
    fn a_restored_root_survives_a_tip_that_extends_it() {
        let (saved, calls, slot) = tip_after_a_restore_of(1, 1, SaveKind::Root);
        assert_eq!(saved, Ok(true), "the tip is saved");
        assert_eq!(calls, 1, "one snapshot");
        assert_eq!(
            slot.keys(),
            vec![(4, true), (5, false)],
            "the root of 4 tokens (roots first) and the tip of 5 tokens"
        );
    }

    /// A restored entry of kind `Tip` is still replaced by a tip that extends it (rule 4), both
    /// with room in the list and as the last resort in a one-entry list.
    ///
    /// Mutation check: making `drop_restored_ancestor` always return `false` leaves the entry of
    /// 4 tokens beside the tip in the first case and skips the save in the second.
    #[test]
    fn a_restored_tip_entry_is_still_replaced_by_a_tip_that_extends_it() {
        let (saved, calls, slot) = tip_after_a_restore_of(0, 2, SaveKind::Tip);
        assert_eq!(saved, Ok(true), "with room: the tip is saved");
        assert_eq!(calls, 1, "with room: one snapshot");
        assert_eq!(
            slot.keys(),
            vec![(5, false)],
            "with room: the old tip is gone"
        );
        let (saved, calls, slot) = tip_after_a_restore_of(0, 1, SaveKind::Tip);
        assert_eq!(saved, Ok(true), "one entry: the tip is saved");
        assert_eq!(calls, 1, "one entry: one snapshot");
        assert_eq!(
            slot.keys(),
            vec![(5, false)],
            "one entry: the old tip is gone"
        );
    }

    /// A store `(0, 2)` whose list holds the restored entry A of 3 tokens at the front (least
    /// recently used) and a second entry B of 5 tokens behind it. The returned session holds the
    /// key `[1, 2, 3, 4]`, which extends A.
    fn restored_entry_at_the_front() -> (Fake, DecodeSession, Slot) {
        let (mut step_a, mut a) = held(&[1, 2, 3]);
        let (step_b, b) = held(&[5, 6, 7, 8, 9]);
        let mut slot = store(0, 2);
        assert_eq!(save(&a, &mut slot, &step_a, &[1, 2, 3], 1), Ok(true));
        assert_eq!(save(&b, &mut slot, &step_b, &[5, 6, 7, 8, 9], 1), Ok(true));
        a.reset(&mut step_a).unwrap();
        assert_eq!(
            restore(&mut a, &mut slot, &mut step_a, &[1, 2, 3, 4], 1).unwrap(),
            3,
            "A restores"
        );
        // The restore made A the most recently used entry. A repeated save of B is a use and
        // moves A to the front.
        assert_eq!(
            save(&b, &mut slot, &step_b, &[5, 6, 7, 8, 9], 1),
            Ok(true),
            "the use of B"
        );
        assert_eq!(
            slot.keys(),
            vec![(3, false), (5, false)],
            "A is at the front"
        );
        a.prefill(&mut step_a, &[4], LIMIT).unwrap();
        (step_a, a, slot)
    }

    /// With a full list the room step drops the entry other than the restored one, the restored
    /// entry stays until the tip is stored, and the lineage rule then drops it.
    ///
    /// Mutation check: replacing the victim lookup of the room step by the first entry drops A
    /// first, so B stays and the final key list is `[(5, false), (4, false)]`.
    #[test]
    fn a_full_list_drops_the_other_entry_and_the_lineage_drops_the_restored_one() {
        let (step, session, mut slot) = restored_entry_at_the_front();
        assert_eq!(
            save(&session, &mut slot, &step, &[1, 2, 3, 4], 1),
            Ok(true),
            "the tip is saved"
        );
        assert_eq!(
            slot.keys(),
            vec![(4, false)],
            "B went in the room step, A went in the lineage step"
        );
    }

    /// With a full list and a denied snapshot the other entry is gone and the restored entry
    /// stays.
    ///
    /// Mutation check: replacing the victim lookup of the room step by the first entry drops A in
    /// the room step and B in the retry, so the final key list is empty.
    #[test]
    fn a_denied_tip_with_a_full_list_keeps_the_restored_entry_and_drops_the_other() {
        let (step, session, mut slot) = restored_entry_at_the_front();
        let denied = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4],
            1,
            step.fed(),
            SaveKind::Tip,
            || Err(EngError::MemoryPlan("no room".to_string())),
        );
        assert_eq!(denied, Ok(false), "the denied save reports false");
        assert_eq!(
            slot.keys(),
            vec![(3, false)],
            "B was the victim; A stays because it was restored"
        );
    }

    /// A root save never drops the root that the job restored. With one root place and that root
    /// restored, the save has no victim and is skipped before the snapshot.
    ///
    /// Mutation check: letting the room step of a root save drop the first root (the restored
    /// one) reaches the snapshot closure, which panics, so the test fails.
    #[test]
    fn a_root_save_never_drops_the_restored_root() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 0);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the root R"
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "R restores"
        );
        session.prefill(&mut step, &[5], LIMIT).unwrap();
        let skipped = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4, 5],
            1,
            step.fed(),
            SaveKind::Root,
            || panic!("a root save without a victim must not run the snapshot"),
        );
        assert_eq!(skipped, Ok(false), "the save is skipped");
        assert_eq!(slot.keys(), vec![(4, true)], "R is still stored");
    }

    /// With two root places the least recently used root other than the restored one is the
    /// victim, even when the restored root is the least recently used one.
    ///
    /// Mutation check: letting the room step of a root save drop the first root drops the
    /// restored root R2 and leaves `[(4, true), (4, true)]`.
    #[test]
    fn a_root_save_drops_the_other_root_when_the_restored_root_is_the_oldest() {
        let (mut step2, mut second) = held(&[5, 6, 7]);
        let (step1, first) = held(&[1, 2, 3, 4]);
        let mut slot = store(2, 0);
        assert_eq!(
            save_as(&second, &mut slot, &step2, &[5, 6, 7], 1, SaveKind::Root),
            Ok(true),
            "R2"
        );
        assert_eq!(
            save_as(&first, &mut slot, &step1, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "R1"
        );
        // The restore of R2 makes it the most recently used root. A repeated save of R1 is a use
        // and puts R2 at the front again.
        second.reset(&mut step2).unwrap();
        assert_eq!(
            restore(&mut second, &mut slot, &mut step2, &[5, 6, 7, 8], 1).unwrap(),
            3,
            "R2 restores"
        );
        assert_eq!(
            save_as(&first, &mut slot, &step1, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the use of R1"
        );
        assert_eq!(
            slot.keys(),
            vec![(3, true), (4, true)],
            "R2 is at the front"
        );
        second.prefill(&mut step2, &[8], LIMIT).unwrap();
        assert_eq!(
            save_as(&second, &mut slot, &step2, &[5, 6, 7, 8], 1, SaveKind::Root),
            Ok(true),
            "the third root"
        );
        assert_eq!(
            slot.keys(),
            vec![(3, true), (4, true)],
            "R1 was the victim; the restored R2 stays"
        );
    }

    /// A root save whose key equals a stored tip promotes that entry to a pinned root without a
    /// new snapshot. The system prompt then survives the LRU drops.
    ///
    /// Mutation check: treating the promotion as a plain use leaves the entry in the LRU list, so
    /// the first key list assertion fails (`[(4, false)]`).
    #[test]
    fn a_root_save_promotes_a_stored_tip_with_the_same_key() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 2);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        let promoted = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4],
            1,
            step.fed(),
            SaveKind::Root,
            || panic!("a promotion must not run the snapshot"),
        );
        assert_eq!(promoted, Ok(true), "the promotion is a use");
        assert_eq!(slot.keys(), vec![(4, true)], "the entry is a root now");
        assert_eq!(slot.entries_len(), 0, "the LRU list is empty");
        // Three further tips: the list holds two, so the third save drops the first tip.
        for key in [&[5u32, 6][..], &[7, 8, 9][..], &[2, 2, 2, 2, 2][..]] {
            let (other_step, other) = held(key);
            assert_eq!(
                save(&other, &mut slot, &other_step, key, 1),
                Ok(true),
                "the tip {key:?}"
            );
        }
        assert_eq!(
            slot.keys(),
            vec![(4, true), (3, false), (5, false)],
            "the root stays; the two newest tips stay"
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 9], 1).unwrap(),
            4,
            "the promoted root restores"
        );
    }

    /// A tip that repeats a stored key is a use, and when it extends the restored entry the
    /// lineage rule still drops that entry (clarification 1, item 4).
    ///
    /// Mutation check: returning at the duplicate branch without the lineage drop leaves the
    /// restored entry of 3 tokens in the list, so the key list assertion fails.
    #[test]
    fn a_repeated_tip_that_extends_the_restored_entry_still_drops_it() {
        let (mut step_a, mut a) = held(&[1, 2, 3]);
        let (step_t, tip) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 4);
        assert_eq!(save(&a, &mut slot, &step_a, &[1, 2, 3], 1), Ok(true));
        assert_eq!(save(&tip, &mut slot, &step_t, &[1, 2, 3, 4], 1), Ok(true));
        a.reset(&mut step_a).unwrap();
        assert_eq!(
            restore(&mut a, &mut slot, &mut step_a, &[1, 2, 3, 9], 1).unwrap(),
            3,
            "A restores"
        );
        a.prefill(&mut step_a, &[4], LIMIT).unwrap();
        let repeated = a.save_prefix(
            &mut slot,
            &[1, 2, 3, 4],
            1,
            step_a.fed(),
            SaveKind::Tip,
            || panic!("a repeated tip must not run the snapshot"),
        );
        assert_eq!(repeated, Ok(true), "the repeated tip is a use");
        assert_eq!(
            slot.keys(),
            vec![(4, false)],
            "the restored entry of 3 tokens is gone; the tip stays once"
        );
    }

    /// A prompt equal to a stored tip key restores the longest key that is strictly shorter,
    /// here the root.
    ///
    /// Mutation check: accepting a key of the same length as the prompt restores the tip and
    /// returns 8 instead of 4.
    #[test]
    fn a_prompt_equal_to_the_tip_key_restores_the_root() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 2);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the root"
        );
        session.prefill(&mut step, &[5, 6, 7, 8], LIMIT).unwrap();
        assert_eq!(
            save(&session, &mut slot, &step, &[1, 2, 3, 4, 5, 6, 7, 8], 1),
            Ok(true),
            "the tip"
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(
                &mut session,
                &mut slot,
                &mut step,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1
            )
            .unwrap(),
            4,
            "the root covers 4; the equal tip does not restore"
        );
    }

    #[test]
    fn a_denied_permit_evicts_and_retries() {
        let mut slot = store(0, 4);
        let (step_a, a) = held(&[1, 2, 3]);
        let (step_b, b) = held(&[4, 5, 6, 7]);
        let (step_c, c) = held(&[8, 9]);
        assert_eq!(save(&a, &mut slot, &step_a, &[1, 2, 3], 1), Ok(true));
        assert_eq!(save(&b, &mut slot, &step_b, &[4, 5, 6, 7], 1), Ok(true));
        // The third entry is a periodic checkpoint: periodic entries share the LRU list and its
        // eviction rules with tips.
        assert_eq!(
            save_as(&c, &mut slot, &step_c, &[8, 9], 1, SaveKind::Periodic),
            Ok(true)
        );
        // The list holds 3 of 4 entries, so the room step drops none. Each denial drops the
        // least recently used entry before the next snapshot: the first denial drops the entry
        // of 3 tokens, the second the entry of 4 tokens. The entry of 2 tokens stays.
        let (step_d, d) = held(&[3, 3, 3, 3, 3]);
        let calls = std::cell::Cell::new(0usize);
        let saved = d.save_prefix(
            &mut slot,
            &[3, 3, 3, 3, 3],
            1,
            step_d.fed(),
            SaveKind::Tip,
            || {
                calls.set(calls.get() + 1);
                if calls.get() <= 2 {
                    Err(EngError::MemoryPlan("no room".to_string()))
                } else {
                    Ok(step_d.fed.clone())
                }
            },
        );
        assert_eq!(saved, Ok(true), "the third snapshot is stored");
        assert_eq!(calls.get(), 3, "the snapshot ran three times");
        assert_eq!(
            slot.keys(),
            vec![(2, false), (5, false)],
            "the two least recently used entries are gone; the newest one and the new one stay"
        );
    }

    #[test]
    fn a_denied_permit_with_nothing_to_evict_skips_the_save() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 1);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true)
        );
        // A tip save never drops a root, so a denial with an empty LRU list skips the save.
        let (step_t, tip) = held(&[5, 6, 7, 8]);
        let denied = tip.save_prefix(
            &mut slot,
            &[5, 6, 7, 8],
            1,
            step_t.fed(),
            SaveKind::Tip,
            || Err(EngError::MemoryPlan("no room".to_string())),
        );
        assert_eq!(denied, Ok(false), "a tip save never drops the root");
        assert_eq!(slot.roots_len(), 1, "the root is intact");
        assert_eq!(slot.keys(), vec![(4, true)], "only the root is stored");
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 9], 1).unwrap(),
            4,
            "the root still restores"
        );
    }

    #[test]
    fn a_third_root_replaces_the_least_recently_used_root() {
        let mut slot = store(2, 0);
        let (mut step1, mut first) = held(&[1, 2, 3, 4]);
        let (step2, second) = held(&[1, 2, 3, 4, 5, 6]);
        let (step3, third) = held(&[7, 8]);
        assert_eq!(
            save_as(&first, &mut slot, &step1, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true)
        );
        assert_eq!(
            save_as(
                &second,
                &mut slot,
                &step2,
                &[1, 2, 3, 4, 5, 6],
                1,
                SaveKind::Root
            ),
            Ok(true)
        );
        // A restore of the first root is a use: the second root stays the least recently used.
        first.reset(&mut step1).unwrap();
        assert_eq!(
            restore(&mut first, &mut slot, &mut step1, &[1, 2, 3, 4, 9], 1).unwrap(),
            4,
            "the first root restores"
        );
        assert_eq!(
            slot.keys(),
            vec![(6, true), (4, true)],
            "least to most recently used"
        );
        assert_eq!(
            save_as(&third, &mut slot, &step3, &[7, 8], 1, SaveKind::Root),
            Ok(true)
        );
        assert_eq!(
            slot.keys(),
            vec![(4, true), (2, true)],
            "the second root is the victim"
        );
        assert_eq!(slot.entries_len(), 0, "the LRU list stays empty");
    }

    #[test]
    fn a_repeated_root_is_a_touch_not_a_second_snapshot() {
        let (step, session) = held(&[1, 2, 3, 4]);
        let mut slot = store(2, 0);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the first save takes the snapshot"
        );
        let again = session.save_prefix(
            &mut slot,
            &[1, 2, 3, 4],
            1,
            step.fed(),
            SaveKind::Root,
            || panic!("a repeated root must not run the snapshot again"),
        );
        assert_eq!(again, Ok(true), "the repeated save reports a use");
        assert_eq!(slot.roots_len(), 1, "one root, not two");
    }

    #[test]
    fn a_denied_restore_drops_that_entry_and_falls_back_to_the_root() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(2, 6);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true)
        );
        session.prefill(&mut step, &[5, 6, 7, 8], LIMIT).unwrap();
        assert_eq!(
            save_as(
                &session,
                &mut slot,
                &step,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1,
                SaveKind::Tip
            ),
            Ok(true)
        );
        session.reset(&mut step).unwrap();
        // The tip snapshot holds 7 fed tokens, the root snapshot 3: the restore is denied for
        // the tip and succeeds for the root.
        let covered = session.restore_prefix(&mut slot, &[1, 2, 3, 4, 5, 6, 7, 8, 9], 1, |fed| {
            if fed.len() > 3 {
                Err(EngError::MemoryPlan("no room".to_string()))
            } else {
                step.fed.clone_from(fed);
                Ok(())
            }
        });
        assert_eq!(covered, Ok(4), "the root covers 4 after the tip is denied");
        assert_eq!(session.history(), [1, 2, 3, 4], "the root history");
        assert_eq!(session.pending(), Some(4), "the root's pending token");
        assert_eq!(slot.keys(), vec![(4, true)], "the denied tip is dropped");
    }

    #[test]
    fn an_entry_of_another_numeric_path_never_restores_and_is_replaced_on_save() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 2);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 2).unwrap(),
            0,
            "another path restores nothing"
        );
        assert_eq!(slot.entries_len(), 1, "the entry stays for its own path");
        // A save of the same key under the other path replaces the entry.
        let (mut step2, mut other) = held(&[1, 2, 3, 4]);
        assert_eq!(
            save_as(&other, &mut slot, &step2, &[1, 2, 3, 4], 2, SaveKind::Tip),
            Ok(true)
        );
        assert_eq!(slot.keys(), vec![(4, false)], "one entry, the new path's");
        other.reset(&mut step2).unwrap();
        assert_eq!(
            restore(&mut other, &mut slot, &mut step2, &[1, 2, 3, 4, 5], 2).unwrap(),
            4,
            "the new entry restores under its own path"
        );
    }

    #[test]
    fn the_default_limits_keep_exactly_one_entry() {
        let (mut step, mut session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        session.prefill(&mut step, &[4, 5], LIMIT).unwrap();
        assert_eq!(
            save(&session, &mut slot, &step, &[1, 2, 3, 4, 5], 1),
            Ok(true)
        );
        assert_eq!(slot.roots_len(), 0, "no roots by default");
        assert_eq!(slot.entries_len(), 1, "exactly one entry");
        assert_eq!(slot.keys(), vec![(5, false)], "the second key");
    }

    #[test]
    fn a_root_save_is_refused_when_the_limits_have_no_roots() {
        let (step, session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        let refused =
            session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), SaveKind::Root, || {
                Ok(step.fed.clone())
            });
        assert_eq!(
            refused,
            Ok(false),
            "a root save without root slots is refused"
        );
        assert!(slot.keys().is_empty(), "nothing is stored");
        assert_eq!(slot.entries_len(), 0, "no LRU entry either");
    }

    /// The report accessors of the store (packet SERVE-2-c item 3): `restored_kind` names the
    /// entry the last restore took, of either list, and `None` after a restore that took
    /// nothing. The kind survives the lineage drop of the restored entry: the tip that replaces
    /// the restored entry erases the id but not the kind.
    ///
    /// Mutation check: clearing `restored_kind` in the lineage drop makes the kind row fail.
    #[test]
    fn the_store_names_the_kind_of_the_entry_the_last_restore_took() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(1, 6);
        assert_eq!(slot.restored_kind(), None, "a fresh store restored nothing");
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Root),
            Ok(true),
            "the root"
        );
        session.prefill(&mut step, &[5, 6, 7, 8], LIMIT).unwrap();
        assert_eq!(
            save_as(
                &session,
                &mut slot,
                &step,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1,
                SaveKind::Tip
            ),
            Ok(true),
            "the tip"
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 9], 1).unwrap(),
            4,
            "the root restores for a divergent continuation"
        );
        assert_eq!(slot.restored_kind(), Some(SaveKind::Root), "the root kind");
        // A restore that matches nothing clears the kind.
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[9, 9, 9], 1).unwrap(),
            0,
            "nothing matches"
        );
        assert_eq!(slot.restored_kind(), None, "the kind is cleared");
        // A tip restore names the tip, and the kind survives the lineage drop of that entry.
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(
                &mut session,
                &mut slot,
                &mut step,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9],
                1
            )
            .unwrap(),
            8,
            "the tip restores"
        );
        assert_eq!(slot.restored_kind(), Some(SaveKind::Tip), "the tip kind");
        session.prefill(&mut step, &[9], LIMIT).unwrap();
        assert_eq!(
            save_as(
                &session,
                &mut slot,
                &step,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9],
                1,
                SaveKind::Tip
            ),
            Ok(true),
            "the extending tip replaces the restored entry"
        );
        assert_eq!(
            slot.restored_kind(),
            Some(SaveKind::Tip),
            "the kind survives the lineage drop"
        );
    }

    /// `uncached_shared` takes the literal longest common prefix with any stored key (packet
    /// SERVE-2-c item 3, fix M2): a prompt that diverges from the stored tip after the first
    /// token still shares that token with it, a prompt that diverges after three tokens shares
    /// three, and a restore that took the longest common prefix reports 0. A key of another
    /// numeric path counts too: its tokens are shared but no checkpoint could serve them.
    ///
    /// Mutation check: restoring the old `prompt.starts_with(key)` filter makes the divergent
    /// rows report 0.
    #[test]
    fn uncached_shared_takes_the_longest_common_prefix_with_any_stored_key() {
        let (step, session) = held(&[1, 2, 3, 4]);
        let (step_root, root_session) = held(&[1, 2]);
        let mut slot = store(1, 6);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3, 4], 1, SaveKind::Tip),
            Ok(true),
            "the tip"
        );
        assert_eq!(
            save_as(
                &root_session,
                &mut slot,
                &step_root,
                &[1, 2],
                2,
                SaveKind::Root
            ),
            Ok(true),
            "a root of another path"
        );
        // A prompt that diverges from both keys after the first token still shares it: the tip
        // [1, 2, 3, 4] and the root [1, 2] each hold a first token of 1.
        assert_eq!(
            slot.uncached_shared(&[1, 9, 3, 4, 5], 0),
            1,
            "the longest common prefix with the tip is the single token 1"
        );
        // The prompt diverges from the tip after three tokens, so three tokens are shared with
        // the stored tip that no checkpoint of this restore covered (the root of the other path
        // shares only two).
        assert_eq!(
            slot.uncached_shared(&[1, 2, 3, 9, 9], 0),
            3,
            "the tip shares its first three tokens"
        );
        // The tip key covers the first four tokens of the prompt; a restore under another path
        // took none of them, so all four are shared but uncached.
        assert_eq!(
            slot.uncached_shared(&[1, 2, 3, 4, 5], 0),
            4,
            "the full tip key counts beyond a zero restore"
        );
        // A restore that took the longest common prefix reports 0.
        assert_eq!(
            slot.uncached_shared(&[1, 2, 3, 4, 5], 4),
            0,
            "the restored key"
        );
        // A shorter restore leaves the tip tokens uncached.
        assert_eq!(
            slot.uncached_shared(&[1, 2, 3, 4, 5], 2),
            2,
            "the tip covers two tokens beyond the shorter restore"
        );
        // A key of equal length covers the whole prompt.
        assert_eq!(
            slot.uncached_shared(&[1, 2, 3, 4], 0),
            4,
            "a key equal to the prompt covers it"
        );
        // A store whose every key shares less than the restore saturates at 0.
        let empty = store(0, 1);
        assert_eq!(
            empty.uncached_shared(&[7, 7], 2),
            0,
            "no stored key, a restore of two: nothing beyond it"
        );
    }

    /// `uncached_shared` counts a pinned root, too: a store whose only long shared key is a
    /// `SaveKind::Root` entry, beside an LRU tip that shares nothing with the prompt, reports
    /// the root's shared tokens against a zero restore, and 0 for a prompt that shares
    /// nothing.
    ///
    /// Mutation check: skipping `self.roots` in the scan (the LRU list only) makes the root
    /// row report 0.
    #[test]
    fn uncached_shared_counts_a_root_when_no_lru_entry_shares_more() {
        let (step, session) = held(&[5, 6, 7, 8]);
        let (tip_step, tip_session) = held(&[9]);
        let mut slot = store(1, 6);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[5, 6, 7, 8], 1, SaveKind::Root),
            Ok(true),
            "the root"
        );
        assert_eq!(
            save_as(&tip_session, &mut slot, &tip_step, &[9], 2, SaveKind::Tip),
            Ok(true),
            "an LRU tip that shares nothing with the prompt"
        );
        assert_eq!(
            slot.keys(),
            vec![(4, true), (1, false)],
            "the root, then the tip"
        );
        // The prompt [5, 6, 7, 9] shares its first three tokens with the root [5, 6, 7, 8]
        // and nothing with the tip, so a zero restore leaves three shared tokens uncached.
        assert_eq!(
            slot.uncached_shared(&[5, 6, 7, 9], 0),
            3,
            "the root shares its first three tokens beyond the zero restore"
        );
        assert_eq!(
            slot.uncached_shared(&[1, 2], 0),
            0,
            "a prompt that shares nothing with either key reports 0"
        );
    }

    // --- KV-v1 slice 1: the shadow block-LCP of one lookup (shadow only, no behavior change) ---

    /// The shadow of a full hit: the stored key is a strict prefix of the prompt, the real
    /// restore takes it, and the shadow reports the same usable checkpoint with the raw
    /// shared length as the divergence offset.
    #[test]
    fn the_shadow_of_a_full_hit_matches_the_restore() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 6);
        slot.set_shadow_enabled(true);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "the entry restores"
        );
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 4, "the raw shared length");
        assert_eq!(shadow.usable_checkpoint_tokens, 4, "the full-prefix key");
        assert_eq!(shadow.block_lcp_128, 0, "four tokens floor to no 128 block");
        assert_eq!(shadow.block_lcp_256, 0, "four tokens floor to no 256 block");
        assert_eq!(shadow.reason, ShadowReason::SameAsActual);
    }

    /// The core case of KV-v1: the prompt diverges from every stored key mid-conversation, so
    /// the exact-key restore finds nothing, but a block-granular restore could use the shared
    /// blocks. 200 shared tokens floor to one 128 block and no 256 block.
    #[test]
    fn the_shadow_floors_a_mid_conversation_divergence_to_the_block_granularity() {
        let key: Vec<u32> = (1..=300u32).map(|i| i * 2).collect();
        let (mut step, mut session) = held(&key);
        let mut slot = store(0, 6);
        slot.set_shadow_enabled(true);
        assert_eq!(save(&session, &mut slot, &step, &key, 1), Ok(true));
        session.reset(&mut step).unwrap();
        let mut prompt: Vec<u32> = (1..=200u32).map(|i| i * 2).collect();
        prompt.extend([7, 7, 7, 7]);
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &prompt, 1).unwrap(),
            0,
            "the key is not a prefix, so nothing restores"
        );
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 200, "the shared run");
        assert_eq!(shadow.block_lcp_128, 128, "one 128 block");
        assert_eq!(shadow.block_lcp_256, 0, "no 256 block");
        assert_eq!(shadow.usable_checkpoint_tokens, 0);
        assert_eq!(shadow.reason, ShadowReason::NoUsableCheckpoint);
    }

    /// Entries on another numeric path never count: the shadow reports the path mismatch, not
    /// the tokens those entries share.
    #[test]
    fn the_shadow_ignores_entries_of_another_path() {
        let (step, session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 6);
        slot.set_shadow_enabled(true);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 2), Ok(true));
        let (mut other_step, mut other_session) = held(&[9]);
        assert_eq!(
            restore(
                &mut other_session,
                &mut slot,
                &mut other_step,
                &[1, 2, 3, 4, 5],
                1
            )
            .unwrap(),
            0,
            "the other-path entry is not a candidate"
        );
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 0);
        assert_eq!(shadow.usable_checkpoint_tokens, 0);
        assert_eq!(shadow.reason, ShadowReason::PathMismatch);
    }

    /// The q rule of the restore contract, preserved: a stored key that is not strictly
    /// shorter than the prompt (q < prompt_len fails) is no usable checkpoint, even though
    /// every token is shared.
    #[test]
    fn the_shadow_excludes_a_key_that_is_not_strictly_shorter() {
        let key: Vec<u32> = (1..=300u32).collect();
        let (mut step, mut session) = held(&key);
        let mut slot = store(0, 6);
        slot.set_shadow_enabled(true);
        assert_eq!(save(&session, &mut slot, &step, &key, 1), Ok(true));
        session.reset(&mut step).unwrap();
        let prompt = key.clone();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &prompt, 1).unwrap(),
            0,
            "q == prompt_len, so the entry is not a candidate"
        );
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 300, "every token is shared");
        assert_eq!(shadow.block_lcp_128, 256, "two 128 blocks");
        assert_eq!(shadow.block_lcp_256, 256, "one 256 block");
        assert_eq!(shadow.usable_checkpoint_tokens, 0);
        assert_eq!(shadow.reason, ShadowReason::QNotBelowPrompt);
    }

    /// A same-path entry that shares no token with the prompt: the no-token-match class of a
    /// conversation whose system area changed entirely.
    #[test]
    fn the_shadow_reports_no_token_match_when_nothing_is_shared() {
        let (mut step, mut session) = held(&[5, 6, 7, 8]);
        let mut slot = store(0, 6);
        slot.set_shadow_enabled(true);
        assert_eq!(save(&session, &mut slot, &step, &[5, 6, 7, 8], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 9], 1).unwrap(),
            0,
            "nothing restores"
        );
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 0);
        assert_eq!(shadow.reason, ShadowReason::NoTokenMatch);
    }

    /// The cost bound: one early-exit walk per held entry, never a re-walk per block. Eight
    /// entries of 1001 tokens over a 1000-token shared run spend exactly 8 x 1001 comparisons;
    /// an anchor-style per-block re-walk would spend blocks x tokens per entry.
    #[test]
    fn the_shadow_walk_spends_one_pass_per_entry() {
        let base: Vec<u32> = (1..=1000u32).collect();
        let mut slot = store(2, 6);
        slot.set_shadow_enabled(true);
        for (k, kind) in (1..=8u32).zip(
            [SaveKind::Root, SaveKind::Root]
                .into_iter()
                .chain(std::iter::repeat(SaveKind::Tip)),
        ) {
            let mut key = base.clone();
            key.push(k);
            let (step, session) = held(&key);
            assert_eq!(
                save_as(&session, &mut slot, &step, &key, 1, kind),
                Ok(true),
                "entry {k}"
            );
        }
        let mut prompt = base;
        prompt.push(99);
        let shadow = slot.block_lcp_shadow(&prompt, &1);
        assert_eq!(shadow.token_comparisons, 8 * 1001, "one walk per entry");
        assert_eq!(shadow.divergence_offset, 1000);
        assert_eq!(
            shadow.usable_checkpoint_tokens, 0,
            "no key is a full prefix"
        );
    }

    /// Heterogeneous entries pin both maxima against last-entry-wins bugs: a root that
    /// diverges mid-run (200 shared, no checkpoint), the longest full-prefix tip (500, the
    /// usable maximum), an early-mismatch tip (one comparison), and a shorter full-prefix tip
    /// saved LAST (300 - assignment instead of max would report its numbers), plus a long
    /// other-path entry whose 2000 shared tokens must not be compared at all.
    #[test]
    fn the_shadow_picks_the_maxima_across_heterogeneous_entries() {
        let base: Vec<u32> = (1..=500u32).collect();
        let mut slot = store(2, 6);
        slot.set_shadow_enabled(true);
        // A: diverges after 200 tokens - 201 comparisons, no checkpoint candidate.
        let mut key_a = base[..200].to_vec();
        key_a.push(7001);
        let (step_a, session_a) = held(&key_a);
        assert_eq!(
            save_as(&session_a, &mut slot, &step_a, &key_a, 1, SaveKind::Root),
            Ok(true)
        );
        // B: the longest full prefix of the prompt - the usable maximum (500).
        let key_b = base.clone();
        let (step_b, session_b) = held(&key_b);
        assert_eq!(
            save_as(&session_b, &mut slot, &step_b, &key_b, 1, SaveKind::Tip),
            Ok(true)
        );
        // C: an early mismatch - exactly one comparison.
        let key_c: Vec<u32> = (9000..=9100u32).collect();
        let (step_c, session_c) = held(&key_c);
        assert_eq!(
            save_as(&session_c, &mut slot, &step_c, &key_c, 1, SaveKind::Tip),
            Ok(true)
        );
        // E: a shorter full prefix saved last - max must keep B's numbers.
        let key_e = base[..300].to_vec();
        let (step_e, session_e) = held(&key_e);
        assert_eq!(
            save_as(&session_e, &mut slot, &step_e, &key_e, 1, SaveKind::Tip),
            Ok(true)
        );
        // D: a long other-path entry - excluded from the walk entirely.
        let key_d: Vec<u32> = (1..=2000u32).collect();
        let (step_d, session_d) = held(&key_d);
        assert_eq!(
            save_as(&session_d, &mut slot, &step_d, &key_d, 2, SaveKind::Tip),
            Ok(true)
        );
        let mut prompt = base;
        prompt.extend([6001, 6002]);
        let (mut step, mut session) = held(&prompt);
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &prompt, 1).unwrap(),
            500,
            "the longest full prefix restores"
        );
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 500, "the longest shared run");
        assert_eq!(shadow.block_lcp_128, 384, "three 128 blocks of 500");
        assert_eq!(shadow.block_lcp_256, 256, "one 256 block of 500");
        assert_eq!(
            shadow.usable_checkpoint_tokens, 500,
            "the greatest usable checkpoint, not the last entry's 300"
        );
        assert_eq!(shadow.reason, ShadowReason::SameAsActual);
        assert_eq!(
            shadow.token_comparisons, 1002,
            "201 (A diverges) + 500 (B prefix) + 1 (C mismatch) + 300 (E prefix); D other path"
        );
    }

    /// The default is off: no shadow is computed or stashed until the store is enabled.
    #[test]
    fn the_shadow_is_silent_until_enabled() {
        let (mut step, mut session) = held(&[1, 2, 3, 4]);
        let mut slot = store(0, 6);
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3, 4], 1), Ok(true));
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "the entry still restores"
        );
        assert_eq!(slot.shadow_report(), None, "no shadow ran");
    }

    #[test]
    fn a_short_block_preserves_longer_checkpoints_before_snapshot() {
        let mut slot = store(0, 3).with_recent_longer_protection();
        let long = [1, 2, 3, 4, 5, 6];
        let (step, session) = held(&long);
        assert_eq!(
            save_as(&session, &mut slot, &step, &long, 1, SaveKind::Block),
            Ok(true)
        );
        let short = [9, 8, 7];
        let (step, session) = held(&short);
        assert_eq!(
            session.save_prefix(
                &mut slot,
                &short,
                1,
                step.fed(),
                SaveKind::Block,
                || panic!("a declined short block must not take a snapshot")
            ),
            Ok(false)
        );
        assert_eq!(slot.keys(), vec![(long.len(), false)]);
        assert_eq!(
            slot.save_side_totals(),
            super::SaveSideTotals {
                save_refusals: 1,
                ..super::SaveSideTotals::default()
            }
        );
    }

    #[test]
    fn legacy_store_replaces_a_long_tip_with_a_new_shorter_conversation() {
        let mut slot = Slot::default();
        let long = [1, 2, 3, 4, 5, 6];
        let (step, session) = held(&long);
        assert_eq!(save(&session, &mut slot, &step, &long, 1), Ok(true));

        let short = [9, 8, 7];
        let (mut step, mut session) = held(&short);
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &short, 1),
            Ok(0)
        );
        session.prefill(&mut step, &short, LIMIT).unwrap();
        assert_eq!(save(&session, &mut slot, &step, &short, 1), Ok(true));
        assert_eq!(slot.keys(), vec![(3, false)]);

        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[9, 8, 7, 6], 1),
            Ok(3)
        );
        assert_eq!(step.fed, vec![9, 8]);
    }

    #[test]
    fn a_long_conversation_survives_one_side_request_but_not_two() {
        for side_count in [1, 2] {
            let mut slot = store(0, 2).with_recent_longer_protection();
            let long = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            for (key, kind) in [(&long[..6], SaveKind::Block), (&long[..8], SaveKind::Tip)] {
                let (step, session) = held(key);
                assert_eq!(save_as(&session, &mut slot, &step, key, 1, kind), Ok(true));
            }
            for side in 0..side_count {
                let short = [9 + side, 8, 7];
                let (mut step, mut session) = held(&short);
                session.reset(&mut step).unwrap();
                assert_eq!(
                    restore(&mut session, &mut slot, &mut step, &short, 1),
                    Ok(0)
                );
                for (key, kind) in [(&short[..2], SaveKind::Block), (&short[..], SaveKind::Tip)] {
                    let (step, session) = held(key);
                    assert_eq!(
                        save_as(&session, &mut slot, &step, key, 1, kind),
                        Ok(side == 1)
                    );
                }
            }
            let (mut step, mut session) = held(&long);
            session.reset(&mut step).unwrap();
            assert_eq!(
                restore(&mut session, &mut slot, &mut step, &long, 1),
                Ok(if side_count == 1 { 8 } else { 0 })
            );
        }
    }

    #[test]
    fn a_restored_tip_keeps_only_its_same_path_block_recent() {
        let main = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let unrelated = [31, 32, 33, 34];
        for (block, block_path, retained) in [
            (&main[..4], 1, true),
            (&main[..4], 2, false),
            (&unrelated[..], 1, false),
        ] {
            let mut slot = store(0, 2).with_recent_longer_protection();
            for (key, path, kind) in [
                (block, block_path, SaveKind::Block),
                (&main[..6], 1, SaveKind::Tip),
            ] {
                let (step, session) = held(key);
                assert_eq!(
                    save_as(&session, &mut slot, &step, key, path, kind),
                    Ok(true)
                );
            }
            let mut side_saved = None;
            for (key, hit) in [(&main[..8], 6), (&main[..10], 8)] {
                let (mut step, mut session) = held(key);
                session.reset(&mut step).unwrap();
                assert_eq!(restore(&mut session, &mut slot, &mut step, key, 1), Ok(hit));
                session.prefill(&mut step, &key[hit..], LIMIT).unwrap();
                // The CUDA append retires its exclusive Tip before publishing its replacement.
                assert!(slot.take_tip_if(key, &1, |_| true).is_some());
                assert_eq!(save(&session, &mut slot, &step, key, 1), Ok(true));
                if key.len() == 8 {
                    let side = [91, 92];
                    let (mut step, mut session) = held(&side);
                    session.reset(&mut step).unwrap();
                    assert_eq!(restore(&mut session, &mut slot, &mut step, &side, 1), Ok(0));
                    session.prefill(&mut step, &side, LIMIT).unwrap();
                    side_saved = Some(save(&session, &mut slot, &step, &side, 1).unwrap());
                }
            }
            let rewrite = [1, 2, 3, 4, 21, 22, 23, 24, 25, 26, 27];
            let (mut step, mut session) = held(&rewrite);
            session.reset(&mut step).unwrap();
            assert_eq!(
                restore(&mut session, &mut slot, &mut step, &rewrite, 1),
                Ok(if retained { 4 } else { 0 }),
                "append then SIDE must not evict the active lineage's Block"
            );
            assert_eq!(side_saved, Some(!retained));
        }
    }

    #[test]
    fn a_failed_restore_does_not_refresh_its_ancestors() {
        let main = [1, 2, 3, 4, 5, 6, 7];
        let mut slot = store(0, 2).with_recent_longer_protection();
        for (key, kind) in [(&main[..4], SaveKind::Block), (&main[..6], SaveKind::Tip)] {
            let (step, session) = held(key);
            assert_eq!(save_as(&session, &mut slot, &step, key, 1, kind), Ok(true));
        }
        let mut session = DecodeSession::default();
        assert!(matches!(
            session.restore_prefix(&mut slot, &main, 1, |_| {
                Err(EngError::BadRequest("restore failed".into()))
            }),
            Err(EngError::BadRequest(_))
        ));
        assert!(slot.lru.iter().all(|stored| stored.requests_since_use == 1));
    }

    #[test]
    fn a_shorter_conversation_is_admitted_on_turn_two_and_hits_on_turn_three() {
        let mut slot = store(0, 2).with_recent_longer_protection();
        let long = [1, 2, 3, 4, 5, 6, 7, 8];
        for (key, kind) in [(&long[..6], SaveKind::Block), (&long[..], SaveKind::Tip)] {
            let (step, session) = held(key);
            assert_eq!(save_as(&session, &mut slot, &step, key, 1, kind), Ok(true));
        }
        for (prompt, expected_hit, expected_save) in [
            (&[9, 8, 7][..], 0, false),
            (&[9, 8, 7, 6][..], 0, true),
            (&[9, 8, 7, 6, 5][..], 4, true),
        ] {
            let (mut step, mut session) = held(prompt);
            session.reset(&mut step).unwrap();
            assert_eq!(
                restore(&mut session, &mut slot, &mut step, prompt, 1),
                Ok(expected_hit)
            );
            session
                .prefill(&mut step, &prompt[expected_hit..], LIMIT)
                .unwrap();
            // Two saves in one request must not age the old conversation twice.
            for _ in 0..2 {
                assert_eq!(
                    save(&session, &mut slot, &step, prompt, 1),
                    Ok(expected_save)
                );
            }
        }
    }

    #[test]
    fn a_short_save_preserves_longer_checkpoints() {
        for (roots, entries, previous, incoming) in [
            (0, 3, SaveKind::Block, SaveKind::Block),
            (0, 1, SaveKind::Tip, SaveKind::Tip),
            (1, 0, SaveKind::Root, SaveKind::Root),
            (0, 3, SaveKind::Tip, SaveKind::Periodic),
            (2, 0, SaveKind::Root, SaveKind::Root),
        ] {
            let mut slot = store(roots, entries).with_recent_longer_protection();
            let long = [1, 2, 3, 4, 5, 6];
            let (step, session) = held(&long);
            assert_eq!(
                save_as(&session, &mut slot, &step, &long, 1, previous),
                Ok(true)
            );
            let short = [9, 8, 7];
            let (step, session) = held(&short);
            assert_eq!(
                session.save_prefix(&mut slot, &short, 1, step.fed(), incoming, || {
                    Err(EngError::MemoryPlan("denied short checkpoint".into()))
                }),
                Ok(false)
            );
            assert_eq!(
                slot.keys(),
                vec![(long.len(), previous == SaveKind::Root)],
                "{previous:?} -> {incoming:?}, roots={roots}, entries={entries}"
            );
        }
    }

    #[test]
    fn root_promotion_preserves_longer_checkpoints() {
        let mut slot = store(1, 1).with_recent_longer_protection();
        let long = [1, 2, 3, 4, 5, 6];
        let (step, session) = held(&long);
        assert_eq!(
            save_as(&session, &mut slot, &step, &long, 1, SaveKind::Root),
            Ok(true)
        );
        let short = [9, 8, 7];
        let (step, session) = held(&short);
        assert_eq!(
            save_as(&session, &mut slot, &step, &short, 1, SaveKind::Tip),
            Ok(true)
        );
        assert_eq!(
            save_as(&session, &mut slot, &step, &short, 1, SaveKind::Root),
            Ok(false)
        );
        assert_eq!(slot.keys(), vec![(long.len(), true), (short.len(), false)]);
    }

    #[test]
    fn a_save_skips_longer_checkpoints_when_choosing_a_victim() {
        let mut slot = store(0, 2).with_recent_longer_protection();
        for key in [&[1, 2, 3, 4, 5, 6][..], &[8, 9][..], &[9, 8, 7][..]] {
            let (step, session) = held(key);
            assert_eq!(
                save_as(&session, &mut slot, &step, key, 1, SaveKind::Tip),
                Ok(true)
            );
        }
        assert_eq!(slot.keys(), vec![(6, false), (3, false)]);
    }

    // --- KV-v1 slice 2: the block prefix-match cut (flag-gated capture, one rolling per lineage) ---

    /// The ruled K=1: one rolling Block cut per lineage. A new `Block` save on a path drops
    /// the `Block` entry that path held before, in the same save, so a lineage's cut tracks
    /// its conversation end without a third entry.
    #[test]
    fn a_new_block_cut_replaces_the_lineages_previous_one() {
        let mut slot = store(0, 4);
        let (step_a, session_a) = held(&[1, 2, 3, 4]);
        assert_eq!(
            save_as(
                &session_a,
                &mut slot,
                &step_a,
                &[1, 2, 3, 4],
                1,
                SaveKind::Block
            ),
            Ok(true)
        );
        let (step_b, session_b) = held(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            save_as(
                &session_b,
                &mut slot,
                &step_b,
                &[1, 2, 3, 4, 5, 6],
                1,
                SaveKind::Block
            ),
            Ok(true)
        );
        assert_eq!(
            slot.keys(),
            vec![(6, false)],
            "the previous cut of the lineage is gone: one rolling Block per path"
        );
    }

    /// K=1 is per lineage (path), not global: the cuts of two conversations coexist.
    #[test]
    fn block_cuts_of_different_paths_coexist() {
        let mut slot = store(0, 4);
        let (step_a, session_a) = held(&[1, 2, 3, 4]);
        assert_eq!(
            save_as(
                &session_a,
                &mut slot,
                &step_a,
                &[1, 2, 3, 4],
                1,
                SaveKind::Block
            ),
            Ok(true)
        );
        let (step_b, session_b) = held(&[5, 6, 7, 8]);
        assert_eq!(
            save_as(
                &session_b,
                &mut slot,
                &step_b,
                &[5, 6, 7, 8],
                2,
                SaveKind::Block
            ),
            Ok(true)
        );
        assert_eq!(slot.keys(), vec![(4, false), (4, false)]);
    }

    /// A Block cut is an LRU entry like a Periodic one: it fills `entries`, never a root
    /// place, and the oldest entry evicts when the list is full.
    #[test]
    fn a_block_cut_takes_an_lru_place() {
        let mut slot = store(0, 1);
        let (step_a, session_a) = held(&[1, 2, 3, 4]);
        assert_eq!(
            save_as(
                &session_a,
                &mut slot,
                &step_a,
                &[1, 2, 3, 4],
                1,
                SaveKind::Block
            ),
            Ok(true)
        );
        let (step_b, session_b) = held(&[1, 2, 3, 4, 5]);
        assert_eq!(
            save_as(
                &session_b,
                &mut slot,
                &step_b,
                &[1, 2, 3, 4, 5],
                1,
                SaveKind::Tip
            ),
            Ok(true)
        );
        assert_eq!(
            slot.keys(),
            vec![(5, false)],
            "the tip evicts the older block cut: both are LRU entries"
        );
    }

    /// The restore contract is unchanged for a Block cut: it is a candidate like any entry,
    /// and the longest full-prefix key wins, so a rewritten prompt restores the cut below its
    /// divergence (the turn-boundary win the shadow measured as an opportunity).
    #[test]
    fn a_block_cut_restores_a_rewritten_prompt() {
        let mut slot = store(0, 4);
        let (step_a, session_a) = held(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            save_as(
                &session_a,
                &mut slot,
                &step_a,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1,
                SaveKind::Block
            ),
            Ok(true)
        );
        let (step_b, session_b) = held(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(
            save(
                &session_b,
                &mut slot,
                &step_b,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1
            ),
            Ok(true)
        );
        let (mut step_c, mut session_c) = held(&[9]);
        let rewritten: Vec<u32> = [1, 2, 3, 4, 5, 6, 7, 8, 9].to_vec();
        let covered = restore(&mut session_c, &mut slot, &mut step_c, &rewritten, 1).unwrap();
        assert_eq!(covered, 8, "the cut is the longest full-prefix candidate");
        assert_eq!(slot.restored_kind(), Some(SaveKind::Block));
    }

    /// R2 of the RED list: when a block cut makes the rewrite restorable, the shadow agrees
    /// with the actual restore - the same-as-actual class on the very case (a mid-key
    /// divergence) that without the cut would stand as the no-usable-checkpoint gap the
    /// shadow was built to measure.
    #[test]
    fn the_shadow_agrees_when_a_block_cut_restores() {
        let mut slot = store(0, 4);
        slot.set_shadow_enabled(true);
        let (step_a, session_a) = held(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            save_as(
                &session_a,
                &mut slot,
                &step_a,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                1,
                SaveKind::Block
            ),
            Ok(true)
        );
        let (step_b, session_b) = held(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(
            save(
                &session_b,
                &mut slot,
                &step_b,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1
            ),
            Ok(true)
        );
        let (mut step_c, mut session_c) = held(&[9]);
        let rewritten: Vec<u32> = [1, 2, 3, 4, 5, 6, 7, 8, 9].to_vec();
        let covered = restore(&mut session_c, &mut slot, &mut step_c, &rewritten, 1).unwrap();
        assert_eq!(covered, 8, "the cut restores below the divergence");
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.divergence_offset, 9, "the raw shared run");
        assert_eq!(
            shadow.usable_checkpoint_tokens, 8,
            "the shadow's usable checkpoint is the cut the restore took"
        );
        assert_eq!(shadow.reason, ShadowReason::SameAsActual);
    }

    /// R2's flag-off column: without the cut, the same rewrite stands as the gap
    /// (`NoUsableCheckpoint`) the shadow exists to measure - the pin that the shadow does
    /// not silently start agreeing once the entries exist but the flag is off.
    #[test]
    fn the_shadow_keeps_reporting_the_gap_when_the_cut_is_absent() {
        let mut slot = store(0, 4);
        slot.set_shadow_enabled(true);
        let (step_b, session_b) = held(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(
            save(
                &session_b,
                &mut slot,
                &step_b,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1
            ),
            Ok(true)
        );
        let (mut step_c, mut session_c) = held(&[9]);
        let rewritten: Vec<u32> = [1, 2, 3, 4, 5, 6, 7, 8, 9].to_vec();
        let covered = restore(&mut session_c, &mut slot, &mut step_c, &rewritten, 1).unwrap();
        assert_eq!(covered, 0, "no full-prefix entry: the tip is not a prefix");
        let shadow = slot.shadow_report().expect("the shadow ran");
        assert_eq!(shadow.usable_checkpoint_tokens, 0);
        assert_eq!(shadow.reason, ShadowReason::NoUsableCheckpoint);
    }

    /// The environment parser behind `prefix_block_spacing` (Sol SF3): the `OnceLock`
    /// caches the first read process-wide, so every case runs in a fresh child process
    /// with its own uninitialized lock. The parent spawns the test binary per case with
    /// the case's variables (both removed first, so the parent's environment cannot leak
    /// in) and asserts the spacing the child prints for both calls - parsing plus
    /// repeated-read coverage; both reads see unchanged variables, so this does not
    /// isolate the OnceLock caching itself (the expected values hold with the lock
    /// removed). (Sol round-2 SF2.) The flip (KV-v1 slice 3): block prefix-match is ON
    /// by default, fail-safe: a set value keeps it on only for `1`/`true`/`on`
    /// (trimmed, case-folded); every other value — `0`/`false`/`off`, empty, garbage,
    /// non-UTF-8 — rolls it back (Opus MustFix, the env_allows convention).
    #[test]
    fn the_prefix_block_environment_parses_in_child_processes() {
        let cases: &[(&[(&str, &str)], &str)] = &[
            (&[], "Some(4096) Some(4096)"),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", "0")], "None None"),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", "false")], "None None"),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", "off")], "None None"),
            (
                &[("ENGINESRV_PREFIX_BLOCK_MATCH", "1")],
                "Some(4096) Some(4096)",
            ),
            (
                &[
                    ("ENGINESRV_PREFIX_BLOCK_MATCH", "true"),
                    ("ENGINESRV_PREFIX_BLOCK_TOKENS", "512"),
                ],
                "Some(512) Some(512)",
            ),
            (
                &[
                    ("ENGINESRV_PREFIX_BLOCK_MATCH", "on"),
                    ("ENGINESRV_PREFIX_BLOCK_TOKENS", "abc"),
                ],
                "Some(4096) Some(4096)",
            ),
            (
                &[
                    ("ENGINESRV_PREFIX_BLOCK_MATCH", "1"),
                    ("ENGINESRV_PREFIX_BLOCK_TOKENS", "0"),
                ],
                "Some(4096) Some(4096)",
            ),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", "maybe")], "None None"),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", "OFF")], "None None"),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", " 0 ")], "None None"),
            (&[("ENGINESRV_PREFIX_BLOCK_MATCH", "")], "None None"),
            (
                &[("ENGINESRV_PREFIX_BLOCK_MATCH", " ON ")],
                "Some(4096) Some(4096)",
            ),
            (
                &[("ENGINESRV_PREFIX_BLOCK_TOKENS", "512")],
                "Some(512) Some(512)",
            ),
        ];
        if std::env::var_os("PREFIX_BLOCK_CHILD").is_none() {
            for (vars, expected) in cases {
                let mut command =
                    std::process::Command::new(std::env::current_exe().expect("test binary"));
                command
                    .args([
                        "--exact",
                        "family::session::tests::the_prefix_block_environment_parses_in_child_processes",
                        "--test-threads=1",
                        "--nocapture",
                    ])
                    .env("PREFIX_BLOCK_CHILD", "1")
                    .env_remove("ENGINESRV_PREFIX_BLOCK_MATCH")
                    .env_remove("ENGINESRV_PREFIX_BLOCK_TOKENS");
                for (name, value) in *vars {
                    command.env(name, value);
                }
                let output = command.output().expect("spawn the prefix block child");
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    output.status.success(),
                    "prefix block child exited with {}\nstdout:\n{stdout}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr),
                );
                assert!(
                    stdout.contains(&format!("prefix block child: {expected}")),
                    "child stdout missing 'prefix block child: {expected}':\n{stdout}"
                );
            }
            // A non-UTF-8 value cannot ride the &str case table; spawn it directly:
            // a mis-set rollback value disables the feature (the fail-safe rule).
            {
                use std::os::unix::ffi::OsStrExt;
                let mut command =
                    std::process::Command::new(std::env::current_exe().expect("test binary"));
                command
                    .args([
                        "--exact",
                        "family::session::tests::the_prefix_block_environment_parses_in_child_processes",
                        "--test-threads=1",
                        "--nocapture",
                    ])
                    .env("PREFIX_BLOCK_CHILD", "1")
                    .env_remove("ENGINESRV_PREFIX_BLOCK_TOKENS")
                    .env(
                        "ENGINESRV_PREFIX_BLOCK_MATCH",
                        std::ffi::OsStr::from_bytes(b"\xff"),
                    );
                let output = command
                    .output()
                    .expect("spawn the non-UTF-8 prefix block child");
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    output.status.success(),
                    "non-UTF-8 prefix block child exited with {}\nstdout:\n{stdout}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr),
                );
                assert!(
                    stdout.contains("prefix block child: None None"),
                    "non-UTF-8 child stdout missing the disabled spacing:\n{stdout}"
                );
            }
            return;
        }
        // The child: print the parsed spacing twice - the second call must return the
        // cached first read.
        let first = super::prefix_block_spacing();
        let second = super::prefix_block_spacing();
        println!("prefix block child: {first:?} {second:?}");
    }

    #[test]
    fn interleaved_prefix_saves_keep_each_sessions_lineage_and_prepared_point() {
        let mut slot = store(0, 4);
        let (mut step_a, mut a) = held(&[1, 2]);
        let (mut step_b, mut b) = held(&[5, 6, 7]);
        assert_eq!(save(&a, &mut slot, &step_a, &[1, 2], 1), Ok(true));
        assert_eq!(
            save_as(&b, &mut slot, &step_b, &[5, 6, 7], 1, SaveKind::Periodic),
            Ok(true)
        );
        a.reset(&mut step_a).unwrap();
        b.reset(&mut step_b).unwrap();
        assert_eq!(
            restore(&mut a, &mut slot, &mut step_a, &[1, 2, 3, 4], 1),
            Ok(2)
        );
        assert_eq!(
            restore(&mut b, &mut slot, &mut step_b, &[5, 6, 7, 8, 9], 1),
            Ok(3)
        );
        let observed = slot.restored;
        a.prefill(&mut step_a, &[3, 4], LIMIT).unwrap();
        let point = a.prepare_prefix_save(&[1, 2, 3, 4], step_a.fed()).unwrap();
        let fed = step_a.fed.clone();
        a.reset(&mut step_a).unwrap();
        assert_eq!(a.restored_prefix, None, "reset clears the live lineage");
        assert_eq!(
            point.save_prefix(
                &mut slot,
                &[1, 2, 3, 4],
                1,
                fed.len(),
                SaveKind::Tip,
                || Ok(fed.clone())
            ),
            Ok(true)
        );
        assert_eq!(
            slot.keys(),
            vec![(3, false), (4, false)],
            "only A's ancestor is replaced"
        );
        assert_eq!(
            slot.restored, observed,
            "A's save preserves B's restore observation"
        );
        assert_eq!(slot.restored_kind(), Some(SaveKind::Periodic));
        assert_eq!(
            point.save_prefix(
                &mut slot,
                &[1, 2, 3, 4],
                1,
                fed.len(),
                SaveKind::Tip,
                || panic!("a duplicate takes no snapshot")
            ),
            Ok(true)
        );
        b.prefill(&mut step_b, &[8, 9], LIMIT).unwrap();
        assert_eq!(save(&b, &mut slot, &step_b, &[5, 6, 7, 8, 9], 1), Ok(true));
        assert_eq!(
            slot.keys(),
            vec![(3, false), (4, false), (5, false)],
            "B's periodic ancestor stays"
        );
        assert_eq!(slot.restored, observed);
    }

    #[test]
    fn interleaved_denied_prefix_saves_protect_their_own_charged_snapshot() {
        let count = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut slot = PrefixStore::with_limits(StoreLimits {
            roots: 0,
            entries: 3,
        });
        let (mut step_a, mut a) = held(&[1, 2]);
        let (mut step_b, mut b) = held(&[5, 6, 7]);
        for (session, step, key) in [(&a, &step_a, &[1, 2][..]), (&b, &step_b, &[5, 6, 7][..])] {
            assert_eq!(
                session.save_prefix(&mut slot, key, 1u8, step.fed(), SaveKind::Tip, || Ok((
                    step.fed.clone(),
                    Live::new(&count)
                ))),
                Ok(true)
            );
        }
        a.reset(&mut step_a).unwrap();
        b.reset(&mut step_b).unwrap();
        assert_eq!(
            a.restore_prefix(&mut slot, &[1, 2, 3], 1, |state| {
                step_a.fed.clone_from(&state.0);
                Ok(())
            }),
            Ok(2)
        );
        assert_eq!(
            b.restore_prefix(&mut slot, &[5, 6, 7, 8], 1, |state| {
                step_b.fed.clone_from(&state.0);
                Ok(())
            }),
            Ok(3)
        );
        let observed = slot.restored;
        a.prefill(&mut step_a, &[3], LIMIT).unwrap();
        let mut charges = Vec::new();
        assert_eq!(
            a.save_prefix(
                &mut slot,
                &[1, 2, 3],
                1,
                step_a.fed(),
                SaveKind::Tip,
                || {
                    charges.push(count.get());
                    Err(EngError::MemoryPlan("no room".to_string()))
                }
            ),
            Ok(false)
        );
        assert_eq!(charges, [2, 1], "B's charge drops before the retry");
        assert_eq!(
            slot.keys(),
            vec![(2, false)],
            "A's restored snapshot stays protected"
        );
        assert_eq!(count.get(), 1);
        assert_eq!(slot.restored, observed);
        assert_eq!(
            a.save_prefix(
                &mut slot,
                &[1, 2, 3],
                1,
                step_a.fed(),
                SaveKind::Tip,
                || Err(backend_error("snapshot failed"))
            ),
            Err(backend_error("snapshot failed"))
        );
        assert_eq!(
            slot.restored, observed,
            "an error also preserves the observation"
        );
        assert_eq!(
            a.save_prefix(
                &mut slot,
                &[1, 2, 3],
                1,
                step_a.fed(),
                SaveKind::Tip,
                || {
                    assert_eq!(count.get(), 1, "A stays charged until the tip succeeds");
                    Ok((step_a.fed.clone(), Live::new(&count)))
                }
            ),
            Ok(true)
        );
        assert_eq!(
            slot.keys(),
            vec![(3, false)],
            "the retry replaces A's ancestor"
        );
        assert_eq!(count.get(), 1);
        assert_eq!(slot.restored, observed);
        drop(slot);
        assert_eq!(count.get(), 0);
    }

    #[test]
    fn a_session_without_a_restore_does_not_protect_another_sessions_root() {
        let mut slot = store(1, 0);
        let (mut step, mut session) = held(&[1, 2, 3]);
        assert_eq!(
            save_as(&session, &mut slot, &step, &[1, 2, 3], 1, SaveKind::Root),
            Ok(true)
        );
        session.reset(&mut step).unwrap();
        assert_eq!(
            restore(&mut session, &mut slot, &mut step, &[1, 2, 3, 4], 1),
            Ok(3)
        );
        let observed = slot.restored;
        let (fresh_step, fresh) = held(&[5, 6]);
        assert_eq!(
            save_as(&fresh, &mut slot, &fresh_step, &[5, 6], 1, SaveKind::Root),
            Ok(true)
        );
        assert_eq!(slot.keys(), vec![(2, true)]);
        assert_eq!(slot.restored, observed);
    }

    #[test]
    fn two_sessions_restore_one_entry_and_keep_their_own_tokens() {
        let mut slot = store(0, 2);
        let (mut step_a, mut a) = held(&[1, 2, 3, 4]);
        assert_eq!(save(&a, &mut slot, &step_a, &[1, 2, 3, 4], 1), Ok(true));
        // Session B, a fresh session, restores the entry and advances on its own.
        let mut step_b = Fake::new();
        let mut b = DecodeSession::default();
        assert_eq!(
            restore(&mut b, &mut slot, &mut step_b, &[1, 2, 3, 4, 5], 1).unwrap(),
            4,
            "B restores the entry A saved"
        );
        assert_eq!(b.history(), [1, 2, 3, 4], "B holds the key");
        b.prefill(&mut step_b, &[5, 6], LIMIT).unwrap();
        assert_eq!(b.pending(), Some(6), "B advances past the key");
        assert_eq!(
            step_b.fed,
            vec![1, 2, 3, 4, 5],
            "B's model state is its own"
        );
        // A restores the same entry again; B's work never reached A.
        a.reset(&mut step_a).unwrap();
        assert_eq!(
            restore(&mut a, &mut slot, &mut step_a, &[1, 2, 3, 4, 7], 1).unwrap(),
            4,
            "A restores the entry"
        );
        assert_eq!(a.history(), [1, 2, 3, 4], "A holds the key");
        assert_eq!(a.pending(), Some(4), "A's pending token is the key's");
        assert_eq!(
            step_a.fed,
            vec![1, 2, 3],
            "A's model state is the saved one"
        );
    }

    /// The 90K eviction cliff (root cause: the fixed six-entry LRU of the `prefix_reuse`
    /// store), at the store level with the limits the engine really ships. Four paths at
    /// 90K-equivalent depths stagger their saves (deepest anchor, then tip) as a
    /// concurrent turn-group does. At the DEFAULT addendum 2 cap of six the first path's
    /// entries evict wholesale - it restores zero tokens and re-prefills everything (the
    /// arm-1 replay: 391 s TTFT at t6; three of the four paths survive in this shape).
    /// With the operator's exact `prefix_store_entries` override every path keeps its
    /// TIP - its deepest restorable state - and no agent re-prefills. The default stays
    /// six (an entry's real cost grows with its depth), so the override is the opt-in.
    #[test]
    fn four_paths_at_ninety_k_depths_the_default_wipes_the_first_and_the_override_keeps_all() {
        let history = |path: u8, len: u32| -> Vec<u32> {
            std::iter::once(u32::from(path))
                .chain((0..len).map(|i| i % 251 + 1))
                .collect()
        };
        let anchor: u32 = 3 * 16_384; // the deepest periodic multiple below the 61.5K tips
        // The fake session's own prefill limit is 32,768; these depths need a bigger one.
        let held_deep = |tokens: &[u32]| {
            let mut step = Fake::new();
            let mut session = DecodeSession::default();
            session.prefill(&mut step, tokens, 1 << 17).unwrap();
            (step, session)
        };
        let run_group = |limits: StoreLimits| -> Vec<usize> {
            let mut slot = PrefixStore::with_limits(limits);
            // Staggered as concurrent prefills complete: each path's anchor and tip land
            // together, so at a small cap the later paths' saves evict the earlier paths'
            // entries wholesale - the arm-1 pattern.
            for path in 1..=4u8 {
                let key = history(path, anchor);
                let (step, session) = held_deep(&key);
                assert!(
                    save_as(&session, &mut slot, &step, &key, path, SaveKind::Periodic).unwrap(),
                    "path {path} saves its deepest anchor"
                );
                let tip = history(path, anchor + 1);
                let (step, session) = held_deep(&tip);
                assert!(
                    save(&session, &mut slot, &step, &tip, path).unwrap(),
                    "path {path} saves its tip"
                );
            }
            // The next turn of every path restores its deepest restorable state - the tip
            // entry, whose key is one path token plus anchor+1 history tokens.
            (1..=4u8)
                .map(|path| {
                    let key = history(path, anchor + 2);
                    let (mut step, mut session) = held_deep(&key);
                    restore(&mut session, &mut slot, &mut step, &key, path).unwrap()
                })
                .collect()
        };
        // The DEFAULT: the first path is wiped - the cliff, in the store itself.
        assert_eq!(
            run_group(crate::family::qwen36::settings::prefix_store_limits_of(
                true, None
            )),
            vec![
                0,
                anchor as usize + 2,
                anchor as usize + 2,
                anchor as usize + 2
            ],
            "at the default cap path 1 restores nothing; the other three keep their tips"
        );
        // The operator's exact override: every path keeps its tip.
        assert_eq!(
            run_group(crate::family::qwen36::settings::prefix_store_limits_of(
                true,
                Some(16)
            )),
            vec![anchor as usize + 2; 4],
            "with the override every path keeps its tip: no agent re-prefills"
        );
    }
}
