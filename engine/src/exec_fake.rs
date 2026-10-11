//! The scripted test fake of the executor seam.
//!
//! The trait and `GenerationState` live in `crate::family::executor`.
//! Design: `docs/design/family-descriptor.md` ("Executor trait"), decision 0033.

use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::tokenizer::{DecodeBatch, DecodeFinish};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;

type Result<T> = std::result::Result<T, EngError>;

#[derive(Default)]
pub(crate) struct FakeSession {
    history: Vec<u32>,
    finished: bool,
}

/// The one saved prefix of a fake: its key and the numeric path revision that built it.
struct SavedPrefix {
    key: Vec<u32>,
    revision: u32,
}

/// Returns scripted token batches and applies the `decode` finish rules:
/// EOS, output limit, and cancellation. It never matches stop strings.
/// The output limit is the lower of the constructor cap and the job limit in `GenerationState`.
/// It also keeps one saved prefix with the rules of decision 0036, addendum 1.
pub(crate) struct FakeExecutor {
    script: VecDeque<Vec<u32>>,
    eos: Vec<u32>,
    remaining: u32,
    saved: Option<SavedPrefix>,
    revision: u32,
    deny_saves: bool,
}

impl FakeExecutor {
    pub(crate) fn new(script: Vec<Vec<u32>>, eos: &[u32], max_tokens: u32) -> Self {
        Self {
            script: script.into(),
            eos: eos.to_vec(),
            remaining: max_tokens,
            saved: None,
            revision: 1,
            deny_saves: false,
        }
    }

    /// Sets the numeric path revision. A saved prefix restores only under the revision that built it.
    pub(crate) fn set_numeric_revision(&mut self, revision: u32) {
        self.revision = revision;
    }

    /// Makes every later `save_prefix` report a denied memory charge.
    pub(crate) fn deny_saves(&mut self) {
        self.deny_saves = true;
    }
}

impl ExecutorOps for FakeExecutor {
    type Session = FakeSession;

    fn history<'s>(&self, st: &'s FakeSession) -> &'s [u32] {
        &st.history
    }

    fn reset(&mut self, st: &mut FakeSession) -> Result<()> {
        *st = FakeSession::default();
        Ok(())
    }

    fn prefill(&mut self, st: &mut FakeSession, toks: &[u32]) -> Result<()> {
        if toks.is_empty() {
            return Err(EngError::BadRequest("empty prefill chunk".to_string()));
        }
        st.history.extend_from_slice(toks);
        Ok(())
    }

    fn decode(
        &mut self,
        st: &mut FakeSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch> {
        if st.finished {
            return Err(EngError::BadRequest("decode after finish".to_string()));
        }
        if st.history.is_empty() {
            return Err(EngError::BadRequest("decode before prefill".to_string()));
        }
        let mut tokens = Vec::new();
        let mut text = String::new();
        let mut finish = None;
        if generation.cancel.load(Ordering::SeqCst) {
            finish = Some(DecodeFinish::Cancelled);
        } else if self.remaining == 0 || generation.remaining_tokens() == Some(0) {
            finish = Some(DecodeFinish::Length);
        } else {
            let ids = self
                .script
                .pop_front()
                .ok_or_else(|| EngError::BadRequest("script exhausted".to_string()))?;
            for id in ids {
                tokens.push(id);
                st.history.push(id);
                if self.eos.contains(&id) {
                    // The EOS id joins the consumed prefix and adds no text.
                    finish = Some(DecodeFinish::Stop);
                    break;
                }
                match generation.decoder.push(id) {
                    Ok(piece) => text.push_str(&piece),
                    Err(error) => {
                        // Decision D8: an error resets the session.
                        *st = FakeSession::default();
                        return Err(error);
                    }
                }
                self.remaining = self.remaining.saturating_sub(1);
                generation.record_generated(1);
                if self.remaining == 0 || generation.remaining_tokens() == Some(0) {
                    finish = Some(DecodeFinish::Length);
                    break;
                }
            }
        }
        if finish.is_some() {
            match generation.decoder.finish() {
                Ok(tail) => text.push_str(&tail),
                Err(error) => {
                    *st = FakeSession::default();
                    return Err(error);
                }
            }
            if finish == Some(DecodeFinish::Cancelled) {
                // Decision D8: a cancelled job resets the session.
                *st = FakeSession::default();
            } else {
                st.finished = true;
            }
        }
        Ok(DecodeBatch {
            tokens,
            text,
            finish,
        })
    }

    fn boundary(&mut self, _st: &mut FakeSession) {}

    fn save_prefix(&mut self, st: &FakeSession, key: &[u32]) -> Result<bool> {
        if st.history.as_slice() != key {
            return Err(EngError::BadRequest(
                "prefix key differs from the committed tokens".to_string(),
            ));
        }
        self.saved = None;
        if self.deny_saves {
            return Ok(false);
        }
        self.saved = Some(SavedPrefix {
            key: key.to_vec(),
            revision: self.revision,
        });
        Ok(true)
    }

    fn restore_prefix(&mut self, st: &mut FakeSession, prompt: &[u32]) -> Result<usize> {
        let Some(saved) = &self.saved else {
            return Ok(0);
        };
        if saved.revision != self.revision
            || saved.key.len() >= prompt.len()
            || !prompt.starts_with(&saved.key)
        {
            return Ok(0);
        }
        *st = FakeSession {
            history: saved.key.clone(),
            finished: false,
        };
        Ok(saved.key.len())
    }
}

#[cfg(test)]
mod tests {
    use super::FakeExecutor;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0},"merges":[]}}"#;

    /// The EOS id of these tests. The vocabulary holds only token 0, the text "a".
    const EOS: u32 = 7;

    fn state() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    /// A fake that holds `script` and a session that has prefilled one prompt token.
    fn prefilled(
        script: Vec<Vec<u32>>,
        cap: u32,
    ) -> (FakeExecutor, <FakeExecutor as ExecutorOps>::Session) {
        let mut executor = FakeExecutor::new(script, &[EOS], cap);
        let mut session = Default::default();
        executor.prefill(&mut session, &[1]).expect("prefill");
        (executor, session)
    }

    fn decode(
        executor: &mut FakeExecutor,
        session: &mut <FakeExecutor as ExecutorOps>::Session,
        generation: &mut GenerationState,
    ) -> DecodeBatch {
        executor.decode(session, generation).expect("decode")
    }

    #[test]
    fn a_job_limit_below_the_executor_cap_ends_with_length() {
        let (mut executor, mut session) = prefilled(vec![vec![0], vec![0], vec![0]], 100);
        let mut generation = state().with_max_tokens(2);
        let first = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(first.tokens, [0], "first tokens");
        assert_eq!(first.finish, None, "first finish");
        let second = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(second.tokens, [0], "second tokens");
        assert_eq!(second.text, "a", "second text");
        assert_eq!(second.finish, Some(DecodeFinish::Length), "second finish");
        assert_eq!(generation.remaining_tokens(), Some(0), "limit used up");
    }

    #[test]
    fn a_job_without_a_limit_follows_the_executor_cap() {
        let (mut executor, mut session) = prefilled(vec![vec![0], vec![0], vec![0]], 2);
        let mut generation = state();
        let first = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(first.finish, None, "first finish");
        let second = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(second.tokens, [0], "second tokens");
        assert_eq!(second.finish, Some(DecodeFinish::Length), "second finish");
        assert_eq!(generation.remaining_tokens(), None, "still no job limit");
    }

    #[test]
    fn a_job_limit_of_zero_ends_before_any_token() {
        let (mut executor, mut session) = prefilled(vec![vec![0]], 100);
        let mut generation = state().with_max_tokens(0);
        let batch = decode(&mut executor, &mut session, &mut generation);
        assert!(batch.tokens.is_empty(), "no token: {:?}", batch.tokens);
        assert_eq!(batch.text, "", "text");
        assert_eq!(batch.finish, Some(DecodeFinish::Length), "finish");
        assert_eq!(
            executor.history(&session),
            &[1],
            "history keeps the prompt only"
        );
    }

    #[test]
    fn an_eos_token_does_not_count_toward_the_job_limit() {
        let (mut executor, mut session) = prefilled(vec![vec![0], vec![EOS]], 100);
        let mut generation = state().with_max_tokens(3);
        let first = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(first.finish, None, "first finish");
        assert_eq!(generation.remaining_tokens(), Some(2), "after one token");
        let second = decode(&mut executor, &mut session, &mut generation);
        assert_eq!(second.tokens, [EOS], "EOS joins the batch");
        assert_eq!(second.finish, Some(DecodeFinish::Stop), "finish");
        assert_eq!(generation.remaining_tokens(), Some(2), "EOS is not counted");
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::{FakeExecutor, FakeSession};
    use crate::EngError;
    use crate::family::executor::ExecutorOps;

    /// A fake with an empty script, and a session that holds `tokens` as its prefilled prompt.
    fn fake_with_prefilled(tokens: &[u32]) -> (FakeExecutor, FakeSession) {
        let mut executor = FakeExecutor::new(Vec::new(), &[9], 10);
        let mut session = FakeSession::default();
        executor.prefill(&mut session, tokens).expect("prefill");
        (executor, session)
    }

    #[test]
    fn a_saved_prefix_restores_into_a_reset_session_and_reports_its_length() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3, 4]);
        assert!(executor.save_prefix(&session, &[1, 2, 3, 4]).expect("save"));
        let mut fresh = FakeSession::default();
        let covered = executor
            .restore_prefix(&mut fresh, &[1, 2, 3, 4, 5, 6])
            .expect("restore");
        assert_eq!(covered, 4, "covered tokens");
        assert_eq!(executor.history(&fresh), &[1, 2, 3, 4], "restored history");
    }

    #[test]
    fn a_saved_prefix_restores_only_for_a_longer_prompt_that_starts_with_the_key() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3, 4]);
        executor.save_prefix(&session, &[1, 2, 3, 4]).expect("save");
        let cases: [(&str, &[u32]); 3] = [
            ("equal prompt", &[1, 2, 3, 4]),
            ("shorter prompt", &[1, 2, 3]),
            ("divergent prompt", &[1, 2, 9, 4, 5]),
        ];
        for (name, prompt) in cases {
            let mut fresh = FakeSession::default();
            let covered = executor.restore_prefix(&mut fresh, prompt).expect(name);
            assert_eq!(covered, 0, "{name}: covered tokens");
            assert!(
                executor.history(&fresh).is_empty(),
                "{name}: session changed"
            );
        }
    }

    #[test]
    fn a_changed_numeric_revision_hides_the_saved_prefix() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2]);
        executor.save_prefix(&session, &[1, 2]).expect("save");
        executor.set_numeric_revision(2);
        let mut fresh = FakeSession::default();
        let covered = executor
            .restore_prefix(&mut fresh, &[1, 2, 3])
            .expect("restore");
        assert_eq!(covered, 0, "covered tokens under another revision");
    }

    #[test]
    fn a_new_save_replaces_the_old_one_and_a_denied_save_leaves_none() {
        let (mut executor, first) = fake_with_prefilled(&[1, 2]);
        executor.save_prefix(&first, &[1, 2]).expect("save first");
        let mut second = FakeSession::default();
        executor.prefill(&mut second, &[3, 4]).expect("prefill");
        executor.save_prefix(&second, &[3, 4]).expect("save second");
        let mut fresh = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut fresh, &[1, 2, 5])
                .expect("restore"),
            0,
            "the first key survived the second save"
        );
        assert_eq!(
            executor
                .restore_prefix(&mut fresh, &[3, 4, 5])
                .expect("restore"),
            2,
            "the second key is missing"
        );
        executor.deny_saves();
        let saved = executor.save_prefix(&second, &[3, 4]).expect("denied save");
        assert!(!saved, "a denied save reports false");
        let mut other = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut other, &[3, 4, 5])
                .expect("restore"),
            0,
            "a denied save left an entry that restores"
        );
    }

    #[test]
    fn a_save_rejects_a_key_that_differs_from_the_committed_tokens() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3]);
        let rejected = executor.save_prefix(&session, &[1, 2]);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "got {rejected:?}"
        );
    }
}
