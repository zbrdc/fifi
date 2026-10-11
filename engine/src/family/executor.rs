//! Crate-internal executor seam: the `ExecutorOps` trait and the per-job `GenerationState`.
//!
//! Design: the family descriptor design ("Executor trait"), decision 0033.
//! The scripted test fake lives in `crate::exec_fake`.

use crate::EngError;
use crate::family::sampler::Sampler;
use crate::tokenizer::{DecodeBatch, IncrementalDecoder};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

type Result<T> = std::result::Result<T, EngError>;

/// Per-job generation state. The executor feeds the decoder inside `decode`.
pub(crate) struct GenerationState {
    pub(crate) decoder: IncrementalDecoder,
    pub(crate) cancel: Arc<AtomicBool>,
    /// The sampler of this job. The executor draws from it inside `decode`.
    pub(crate) sampler: Sampler,
    /// The output limit of this job. `None` means the job has no limit of its own.
    max_tokens: Option<u32>,
    /// The tokens that count toward `max_tokens`. An EOS token does not count.
    generated: u32,
}

impl GenerationState {
    /// Creates the state of one job with an unset cancellation flag and a greedy sampler.
    pub(crate) fn new(decoder: IncrementalDecoder) -> Self {
        Self::with_sampler(decoder, Sampler::greedy())
    }

    /// Creates the state of one job with an unset cancellation flag and the given sampler.
    pub(crate) fn with_sampler(decoder: IncrementalDecoder, sampler: Sampler) -> Self {
        Self {
            decoder,
            cancel: Arc::new(AtomicBool::new(false)),
            sampler,
            max_tokens: None,
            generated: 0,
        }
    }

    /// Sets the output limit of this job. A limit of zero ends the job on its first `decode` call.
    pub(crate) fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Returns the tokens the job can still generate, or `None` when the job has no limit.
    pub(crate) fn remaining_tokens(&self) -> Option<u32> {
        self.max_tokens
            .map(|limit| limit.saturating_sub(self.generated))
    }

    /// Counts `count` generated tokens against the limit. Never count an EOS token.
    pub(crate) fn record_generated(&mut self, count: u32) {
        self.generated = self.generated.saturating_add(count);
    }
}

/// The calls the engine worker makes on an executor, in the worker's call order.
pub(crate) trait ExecutorOps {
    type Session;
    /// The committed tokens of the session.
    fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32];
    /// Clears all mutable session state together.
    fn reset(&mut self, st: &mut Self::Session) -> Result<()>;
    /// Consumes one non-empty prompt chunk.
    fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<()>;
    /// Commits one batch of tokens, or reports `Cancelled`.
    fn decode(
        &mut self,
        st: &mut Self::Session,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch>;
    /// Clears transient routing and draft state at a turn boundary.
    fn boundary(&mut self, st: &mut Self::Session);
    /// The number of prompt tokens that one worker piece feeds when this executor plans its own
    /// chunks. `None` means the caller's fallback. The value is never `Some(0)`: the worker treats
    /// it as an engine load failure.
    fn prefill_piece(&self) -> Option<usize> {
        None
    }
    /// Saves the session as the engine's one prefix state and drops any earlier one (decision 0036,
    /// addendum 1). `key` holds the committed prompt tokens of the session. Returns `Ok(false)`
    /// when this executor cannot save or when the memory plan denies the charge. It never changes
    /// the session. Executors that keep no prefix state use this default.
    fn save_prefix(&mut self, _st: &Self::Session, _key: &[u32]) -> Result<bool> {
        Ok(false)
    }
    /// Replaces the reset session with the saved prefix state. It does so only when the saved key is
    /// shorter than `prompt`, equals its first tokens, and comes from the current numeric path.
    /// Returns the number of prompt tokens that the state covers, or 0 with the session unchanged.
    fn restore_prefix(&mut self, _st: &mut Self::Session, _prompt: &[u32]) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutorOps, GenerationState};
    use crate::EngError;
    use crate::family::sampler::{Sampler, SamplingParams};
    use crate::tokenizer::{DecodeBatch, IncrementalDecoder, Tokenizer};
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

    /// An executor that implements only the required calls.
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
    fn an_executor_without_prefix_support_saves_nothing_and_restores_nothing() {
        let mut executor = Bare;
        let mut session = vec![1, 2, 3];
        assert!(!executor.save_prefix(&session, &[1, 2, 3]).expect("save"));
        assert_eq!(
            executor
                .restore_prefix(&mut session, &[1, 2, 3, 4])
                .expect("restore"),
            0
        );
        assert_eq!(session, vec![1, 2, 3], "the session is unchanged");
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
}
