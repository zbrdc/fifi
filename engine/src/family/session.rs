//! The decode session shared by the resident executors (decision 0033; packet Q36-F1).
//!
//! The Flash-Next executor and the Q36 executor use one `DecodeSession`. It owns the token side of
//! a session: the committed tokens, the pending token, and the finish flag. The model side stays
//! with each executor behind `TokenStepper`.
//!
//! Call order: `reset`, `prefill` for each chunk, then `decode` until a finish. A prompt feeds
//! every token except the last one through `TokenStepper::feed`. The last token stays pending.
//! Each `decode` feeds the pending token through `TokenStepper::logits`, draws the next token from
//! the job's sampler, and keeps that token pending. One token leaves per call.
//!
//! The finish rules are pure functions in this file. The session owns EOS, the output limit, the
//! context limit, and cancellation (decision 0030, D7).
//!
//! Prefix reuse (decision 0036, addendum 1): `DecodeSession::save_prefix` and
//! `DecodeSession::restore_prefix` are the one authority for the reuse rules of every family. An
//! executor owns one `PrefixSlot` and passes two closures that save and restore its native state.
//! The rules decide when to save, when to restore, and how the token side changes. The closures
//! only copy or share model state.

use crate::{
    EngError,
    family::executor::GenerationState,
    tokenizer::{DecodeBatch, DecodeFinish},
};
use std::sync::atomic::Ordering;

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

    /// Commits one batch of tokens: feeds the pending token through a logits step, draws the next
    /// token from the job's sampler, and keeps it pending. One token leaves per call. A job that
    /// was cancelled, has no tokens left, or hits a limit ends with a finish reason.
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
        let logits = step.logits(pending)?;
        self.pending = None;
        let visible = logits
            .get(..limits.sampler_vocab)
            .ok_or_else(|| EngError::Backend {
                operation: S::OPERATION,
                message: "the logits are narrower than the sampler vocabulary".to_string(),
            })?;
        let token = generation.sampler.sample(visible)?;
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

    /// Saves the model state as the one prefix of the slot. The state must be the state after the
    /// held prompt, which only `prefill` produces. The call needs a pending token, an unfinished
    /// session, a `key` that equals the held prompt exactly, and a model that has fed every key
    /// token except the pending one (`fed + 1 == key.len()`, where `fed` is `TokenStepper::fed`).
    /// Any other session, including one that decoded a token or one whose model is not at the
    /// position of the key, returns `BadRequest` and keeps the earlier entry.
    ///
    /// The earlier entry drops before `snapshot` runs, so its charge is free for the new one. A
    /// `MemoryPlan` error from `snapshot` gives `Ok(false)` and leaves the slot empty. Another
    /// error leaves the slot empty and propagates. The session never changes.
    pub(crate) fn save_prefix<S, P>(
        &self,
        slot: &mut PrefixSlot<S, P>,
        key: &[u32],
        path: P,
        fed: usize,
        snapshot: impl FnOnce() -> Result<S>,
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
        slot.entry = None; // the new key replaces the entry
        match snapshot() {
            Ok(state) => {
                slot.entry = Some(SavedPrefix {
                    key: key.to_vec(),
                    pending,
                    path,
                    state,
                });
                Ok(true)
            }
            Err(EngError::MemoryPlan(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Replaces the model state and the tokens of this session with the saved prefix. It does so
    /// only when the entry comes from `path`, its key is shorter than `prompt`, and `prompt` starts
    /// with the key. In every other case it returns 0 and changes nothing. The caller resets the
    /// session first and prefills `prompt[covered..]` afterwards.
    ///
    /// `restore` shares or copies the saved state into the model state. A `MemoryPlan` error from
    /// it drops the entry and returns 0 with the session unchanged (addendum item 7); `restore`
    /// must leave the model state unchanged when it returns an error. Another error propagates and
    /// keeps the entry. On success the history is the key, the pending token is the saved one, and
    /// the session is not finished. The call returns the number of prompt tokens that the state covers.
    pub(crate) fn restore_prefix<S, P: PartialEq>(
        &mut self,
        slot: &mut PrefixSlot<S, P>,
        prompt: &[u32],
        path: P,
        restore: impl FnOnce(&S) -> Result<()>,
    ) -> Result<usize> {
        let Some(saved) = slot.entry.as_ref() else {
            return Ok(0);
        };
        if saved.path != path || saved.key.len() >= prompt.len() || !prompt.starts_with(&saved.key)
        {
            return Ok(0);
        }
        match restore(&saved.state) {
            Ok(()) => {}
            Err(EngError::MemoryPlan(_)) => {
                slot.entry = None; // a denied restore drops the entry
                return Ok(0);
            }
            Err(error) => return Err(error),
        }
        self.history.clone_from(&saved.key);
        self.prompt_len = saved.key.len();
        self.pending = Some(saved.pending);
        self.finished = false; // a restored session is not finished
        Ok(saved.key.len())
    }
}

/// The one saved prefix of an engine. `S` is the model state that the executor saves, and `P`
/// names the numeric path that built it. A state restores only under the path that built it,
/// because a batched prefill matches one-token steps only within a tolerance (decision 0036,
/// item 7). The executor owns the slot and declares it before its decoder, so the saved arrays
/// drop before the decoder starts its teardown.
pub(crate) struct PrefixSlot<S, P> {
    entry: Option<SavedPrefix<S, P>>,
}

impl<S, P> Default for PrefixSlot<S, P> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<S, P> PrefixSlot<S, P> {
    /// The number of key tokens of the entry, or `None` when the slot is empty.
    pub(crate) fn key_len(&self) -> Option<usize> {
        self.entry.as_ref().map(|saved| saved.key.len())
    }
}

/// The entry of a `PrefixSlot`: the state after the key tokens and the token that the next step feeds.
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
/// `context_limit`. For example, 32,768 positions allow 32,767 prompt tokens.
fn check_prompt(total: usize, context_limit: usize) -> Result<()> {
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
        DecodeSession, Limits, PrefixSlot, Result, TokenStepper, after_token, before_step,
        check_prompt,
    };
    use crate::{
        EngError,
        family::executor::GenerationState,
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

        fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
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

    /// The prefix slot of the tests: the snapshot is the list of tokens that the fake has fed, and
    /// the numeric path is a small number.
    type Slot = PrefixSlot<Vec<u32>, u8>;

    /// Saves the tokens that `step` has fed.
    fn save(
        session: &DecodeSession,
        slot: &mut Slot,
        step: &Fake,
        key: &[u32],
        path: u8,
    ) -> Result<bool> {
        session.save_prefix(slot, key, path, step.fed(), || Ok(step.fed.clone()))
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
    /// that the entry of key length 3 is still in the slot.
    fn assert_save_rejected(
        session: &DecodeSession,
        step: &Fake,
        slot: &mut Slot,
        key: &[u32],
        what: &str,
    ) {
        let ran = std::cell::Cell::new(false);
        let result = session.save_prefix(slot, key, 1, step.fed(), || {
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
        let early = empty.save_prefix(&mut none, &[], 1, 0, || Ok(Vec::new()));
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
            let result = session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), || {
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
        let mut slot: PrefixSlot<Live, u8> = PrefixSlot::default();
        let first = session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), || {
            Ok(Live::new(&count))
        });
        assert_eq!(first, Ok(true), "first save");
        assert_eq!(count.get(), 1, "one snapshot alive");
        session.prefill(&mut step, &[4, 5], LIMIT).unwrap();
        let alive_during = std::cell::Cell::new(usize::MAX);
        let second = session.save_prefix(&mut slot, &[1, 2, 3, 4, 5], 1, step.fed(), || {
            alive_during.set(count.get());
            Ok(Live::new(&count))
        });
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
        let (step, session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        let denied = session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), || {
            Err(EngError::MemoryPlan("no room".to_string()))
        });
        assert_eq!(denied, Ok(false), "a denied charge reports false");
        assert_eq!(slot.key_len(), None, "no entry after a denial");
        assert_eq!(session.history(), [1, 2, 3], "the session is unchanged");
    }

    #[test]
    fn a_snapshot_error_other_than_memory_plan_propagates_and_leaves_the_slot_empty() {
        let (step, session) = held(&[1, 2, 3]);
        let mut slot = Slot::default();
        assert_eq!(save(&session, &mut slot, &step, &[1, 2, 3], 1), Ok(true));
        let failed = session.save_prefix(&mut slot, &[1, 2, 3], 1, step.fed(), || {
            Err(backend_error("snapshot failed"))
        });
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
}
