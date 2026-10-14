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
        let visible = logits
            .get(..sampler_vocab)
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits are narrower than the sampler vocabulary".to_string(),
            })?;
        sampler.sample(visible)
    }
    /// The number of positions fed so far.
    fn fed(&self) -> usize;
    /// Clears the model state of the session.
    fn reset(&mut self) -> Result<()>;
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
#[derive(Debug, Default)]
pub(crate) struct DecodeSession {
    /// Every committed token: the prompt and the sampled tokens, including a final EOS token.
    history: Vec<u32>,
    /// The number of leading `history` tokens that came from `prefill` (the held prompt).
    prompt_len: usize,
    /// The token that the next step feeds. `None` before a prompt and after EOS.
    pending: Option<u32>,
    /// True after a finish. A new job starts with `reset`.
    finished: bool,
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
        Ok(())
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
        if self.finished {
            return Err(EngError::BadRequest("decode after finish".to_string()));
        }
        let Some(pending) = self.pending else {
            return Err(EngError::BadRequest("decode before prefill".to_string()));
        };
        if generation.cancel.load(Ordering::SeqCst) {
            return self.end(
                step,
                generation,
                Vec::new(),
                String::new(),
                DecodeFinish::Cancelled,
            );
        }
        if let Some(reason) = before_step(generation.remaining_tokens()) {
            return self.end(step, generation, Vec::new(), String::new(), reason);
        }
        // The token is consumed exactly when the model fed it. A failure before the feed (a failing
        // `logits`) keeps the token pending; a failure after it (a narrow row, an invalid row)
        // leaves no pending token, because a retry would feed the token twice.
        let fed_before = step.fed();
        let token = match step.next_token(pending, &mut generation.sampler, limits.sampler_vocab) {
            Ok(token) => token,
            Err(error) => {
                if step.fed() != fed_before {
                    self.pending = None;
                }
                return Err(error);
            }
        };
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
    /// `MemoryPlan` error from `snapshot` drops one victim (the least recently used LRU entry
    /// that the last restore did not restore; the least recently used root other than the
    /// restored root only for a `Root` save) before the next `snapshot` call, until the save
    /// succeeds or no victim is left, which returns `Ok(false)`. Any other error propagates. The
    /// entries dropped before either end stay dropped. A tip whose key extends the key of the
    /// LRU entry that the last restore restored drops that entry after the save succeeds, also
    /// when the tip key repeats a stored key. When the list is full and no other entry can go,
    /// that restored entry is dropped before the snapshot (clarification 1, item 1). The
    /// session never changes.
    pub(crate) fn save_prefix<S, P: PartialEq>(
        &self,
        store: &mut PrefixStore<S, P>,
        key: &[u32],
        path: P,
        fed: usize,
        kind: SaveKind,
        mut snapshot: impl FnMut() -> Result<S>,
    ) -> Result<bool> {
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
        store.save(key, pending, path, kind, &mut snapshot)
    }

    /// Replaces the model state and the tokens of this session with the best stored entry. The
    /// candidates are every stored entry, root or LRU entry, whose `path` equals `path`, whose
    /// key is shorter than `prompt`, and whose key is a prefix of `prompt`; the longest key is
    /// tried first. When no candidate exists, nothing changes and the call returns 0. The caller
    /// resets the session first and prefills `prompt[covered..]` afterwards.
    ///
    /// `restore` shares or copies the saved state into the model state. A `MemoryPlan` error from
    /// it drops that entry and the next candidate is tried (`restore` must leave the model state
    /// unchanged when it returns an error); any other error propagates and keeps the entries. On
    /// success the entry becomes the most recently used entry of its list, the restored entry
    /// (root or not) becomes the protected entry of the store, the history is the key, the pending token
    /// is the saved one, and the session is not finished. The call returns the number of prompt
    /// tokens that the state covers.
    pub(crate) fn restore_prefix<S, P: PartialEq>(
        &mut self,
        store: &mut PrefixStore<S, P>,
        prompt: &[u32],
        path: P,
        mut restore: impl FnMut(&S) -> Result<()>,
    ) -> Result<usize> {
        store.restored = None;
        for (id, is_root, _len) in store.candidates(prompt, &path) {
            if let Some((key, pending)) = store.restore_entry(id, is_root, &mut restore)? {
                self.history.clone_from(&key);
                self.prompt_len = key.len();
                self.pending = Some(pending);
                self.finished = false; // a restored session is not finished
                return Ok(key.len());
            }
        }
        Ok(0)
    }
}

/// What kind of save produced a stored entry (decision 0036, addendum 2): a pinned root, a
/// periodic checkpoint of a long prompt, or the tip of the last prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SaveKind {
    /// The state at the end of the first system message of a prompt.
    Root,
    /// A periodic checkpoint of a long prompt.
    Periodic,
    /// The state at the last message boundary of a prompt.
    Tip,
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
    /// The id of the next stored entry. Ids count up and are never reused.
    next_id: u64,
    /// The id of the entry that the last `restore_prefix` call restored, root or LRU entry.
    restored: Option<u64>,
}

/// The old name of `PrefixStore` (decision 0036, addendum 1), kept for the executors' field
/// declarations.
pub(crate) type PrefixSlot<S, P> = PrefixStore<S, P>;

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
            next_id: 0,
            restored: None,
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
            if kind == SaveKind::Tip {
                self.drop_restored_ancestor(key);
            }
            return Ok(done);
        }
        // (b) An entry with the same key and another path is replaced.
        let other_path =
            |stored: &Stored<S, P>| stored.saved.key == key && stored.saved.path != path;
        self.roots.retain(|stored| !other_path(stored));
        self.lru.retain(|stored| !other_path(stored));
        // (c) Room. A Periodic or Tip save never drops a root here (rule e). No save drops the
        // entry that this job restored, except the last resort for a tip that extends it.
        match kind {
            SaveKind::Root => {
                if self.limits.roots == 0 {
                    return Ok(false);
                }
                if self.roots.len() >= self.limits.roots {
                    match self.first_unprotected_root() {
                        Some(pos) => {
                            self.roots.remove(pos);
                        }
                        None => return Ok(false), // the only root is the restored one
                    }
                }
            }
            SaveKind::Periodic | SaveKind::Tip => {
                if self.limits.entries == 0 {
                    return Ok(false);
                }
                while self.lru.len() >= self.limits.entries {
                    match self.first_unprotected() {
                        Some(pos) => {
                            self.lru.remove(pos);
                        }
                        None => {
                            // Last resort (clarification 1, item 1): the restored entry goes,
                            // before the snapshot, when the tip extends it.
                            if kind == SaveKind::Tip && self.drop_restored_ancestor(key) {
                                continue;
                            }
                            return Ok(false); // every entry is protected
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
                        SaveKind::Periodic => self.lru.push(stored),
                        SaveKind::Tip => {
                            self.lru.push(stored);
                            // (f) A tip that extends the restored entry replaces it.
                            self.drop_restored_ancestor(key);
                        }
                    }
                    return Ok(true);
                }
                Err(EngError::MemoryPlan(_)) => match self.first_unprotected() {
                    Some(pos) => {
                        self.lru.remove(pos);
                    }
                    None if kind == SaveKind::Root => match self.first_unprotected_root() {
                        Some(pos) => {
                            self.roots.remove(pos);
                        }
                        None => return Ok(false), // no victim is left
                    },
                    None => return Ok(false), // no victim is left
                },
                Err(error) => return Err(error),
            }
        }
    }

    /// Step (a): when a stored entry has the same key and the same path, it is a use. It moves to
    /// the most recently used place of its own list. A `Root` save that finds the key in the LRU
    /// list promotes that entry to a pinned root instead, without a new snapshot: it takes a root
    /// place and drops the least recently used root other than the restored one when the roots
    /// are full. Returns `None` when no entry matches, else whether the save counts as done.
    /// A promotion that finds no root to drop leaves the entry in the LRU list as a use and
    /// returns `Some(false)`.
    fn use_duplicate(&mut self, key: &[u32], path: &P, kind: SaveKind) -> Option<bool> {
        let same = |stored: &Stored<S, P>| stored.saved.key == key && &stored.saved.path == path;
        if let Some(pos) = self.roots.iter().position(same) {
            let stored = self.roots.remove(pos);
            self.roots.push(stored);
            return Some(true);
        }
        let pos = self.lru.iter().position(same)?;
        let mut stored = self.lru.remove(pos);
        if kind != SaveKind::Root || self.limits.roots == 0 {
            self.lru.push(stored);
            return Some(true);
        }
        if self.roots.len() >= self.limits.roots {
            let Some(victim) = self.first_unprotected_root() else {
                self.lru.push(stored);
                return Some(false);
            };
            self.roots.remove(victim);
        }
        stored.kind = SaveKind::Root;
        self.roots.push(stored);
        Some(true)
    }

    /// The position of the least recently used LRU entry that the last restore did not restore
    /// (rule 5: the restored entry is never an eviction victim), or `None`.
    fn first_unprotected(&self) -> Option<usize> {
        self.lru
            .iter()
            .position(|stored| self.restored != Some(stored.id))
    }

    /// The position of the least recently used root that the last restore did not restore, or
    /// `None`.
    fn first_unprotected_root(&self) -> Option<usize> {
        self.roots
            .iter()
            .position(|stored| self.restored != Some(stored.id))
    }

    /// Step (f): when the last restore restored an LRU entry that is still stored and the new tip
    /// extends its key, that entry is dropped and the call reports `true`.
    fn drop_restored_ancestor(&mut self, key: &[u32]) -> bool {
        let Some(id) = self.restored else {
            return false;
        };
        let Some(pos) = self.lru.iter().position(|stored| stored.id == id) else {
            return false;
        };
        if key.len() > self.lru[pos].saved.key.len() && key.starts_with(&self.lru[pos].saved.key) {
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

    /// One restore attempt on the entry `id`: run `restore` on its state. On success the entry
    /// becomes the most recently used entry of its list, it becomes the protected entry (root or
    /// not), and its key and pending token are returned. A `MemoryPlan` error drops that entry
    /// and returns `None`, so the caller tries the next candidate; any other error propagates and
    /// keeps the entries.
    fn restore_entry(
        &mut self,
        id: u64,
        is_root: bool,
        restore: &mut dyn FnMut(&S) -> Result<()>,
    ) -> Result<Option<(Vec<u32>, u32)>> {
        let list = if is_root {
            &mut self.roots
        } else {
            &mut self.lru
        };
        let Some(pos) = list.iter().position(|stored| stored.id == id) else {
            return Ok(None);
        };
        let outcome = restore(&list[pos].saved.state);
        match outcome {
            Ok(()) => {
                let stored = list.remove(pos);
                let given = (stored.saved.key.clone(), stored.saved.pending);
                let id = stored.id;
                list.push(stored);
                self.restored = Some(id);
                Ok(Some(given))
            }
            Err(EngError::MemoryPlan(_)) => {
                list.remove(pos);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

/// One stored entry of a `PrefixStore`: the saved prefix plus its store id and its save kind.
struct Stored<S, P> {
    /// The id of this entry in its store. Ids are never reused.
    id: u64,
    /// What kind of save produced this entry.
    kind: SaveKind,
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
    Err(EngError::ContextLimit(format!(
        "prompt has {total} tokens; this engine allows {} (context limit {context_limit} positions)",
        context_limit.saturating_sub(1)
    )))
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
        DecodeSession, Limits, PrefixStore, Result, SaveKind, StoreLimits, TokenStepper,
        after_token, before_step, check_prompt,
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
            Err(EngError::ContextLimit(
                "prompt has 32768 tokens; this engine allows 32767 (context limit 32768 positions)"
                    .to_string()
            ))
        );
        assert!(matches!(
            check_prompt(40_000, 32_768),
            Err(EngError::ContextLimit(_))
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
            matches!(result, Err(EngError::ContextLimit(_))),
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
        let (other_step, other) = held(&[7, 8, 9]);
        let skipped = other.save_prefix(
            &mut slot,
            &[7, 8, 9],
            1,
            other_step.fed(),
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
    /// restored root R2 and leaves `[(4, true), (2, true)]`.
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
        let (step3, third) = held(&[9, 9]);
        assert_eq!(
            save_as(&third, &mut slot, &step3, &[9, 9], 1, SaveKind::Root),
            Ok(true),
            "the third root"
        );
        assert_eq!(
            slot.keys(),
            vec![(3, true), (2, true)],
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
}
