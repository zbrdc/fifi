//! The scripted test fake of the executor seam.
//!
//! The trait and `GenerationState` live in `crate::family::executor`.
//! Design: the family descriptor design ("Executor trait"), decision 0033.
//!
//! The fake keeps the real prefix rules (decision 0036, addendum 2): its session holds a real
//! `DecodeSession` for the token side, and `save_prefix` / `restore_prefix` run through
//! `DecodeSession::save_prefix` / `restore_prefix` and a real `PrefixStore`, the same code path
//! the resident executors use. The saved state is `()`: the fake has no model, and the store
//! never inspects a state.

use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::session::{DecodeSession, PrefixStore, SaveKind, StoreLimits, TokenStepper};
use crate::tokenizer::{DecodeBatch, DecodeFinish};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

type Result<T> = std::result::Result<T, EngError>;

/// Locks the shared store. A poisoned lock means a test thread panicked while saving; the
/// store is still readable, so the poison is cleared like every other lock in this crate.
fn lock_store(store: &Mutex<PrefixStore<(), u32>>) -> MutexGuard<'_, PrefixStore<(), u32>> {
    store.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The fake's model side: it counts the fed positions and has no logits. It lives in the
/// session, like the model state of the resident executors, so its positions reset with the
/// session instead of surviving it.
#[derive(Default)]
struct CountingStepper {
    fed: usize,
}

impl TokenStepper for CountingStepper {
    const OPERATION: &'static str = "fake";

    fn feed(&mut self, _token: u32) -> Result<()> {
        self.fed += 1;
        Ok(())
    }

    fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
        Err(EngError::BadRequest("the fake has no logits".to_string()))
    }

    fn fed(&self) -> usize {
        self.fed
    }

    fn reset(&mut self) -> Result<()> {
        self.fed = 0;
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct FakeSession {
    /// Every committed token: the prompt and the scripted tokens, like the history of a real
    /// session after decode.
    history: Vec<u32>,
    /// The token side of the prompt, driven through the real `DecodeSession` prefill and
    /// prefix rules. The fake's scripted `decode` cannot enter it, because the real `decode`
    /// samples one token per call from logits, so the prompt tokens stay here and in `history`.
    core: DecodeSession,
    /// The model side of the session: the count of fed positions.
    step: CountingStepper,
    finished: bool,
}

/// Returns scripted token batches and applies the `decode` finish rules:
/// EOS, output limit, and cancellation. It never matches stop strings.
/// The output limit is the lower of the constructor cap and the job limit in `GenerationState`.
/// Its prefix store is a real `PrefixStore` with the addendum 1 limits by default (zero roots,
/// one entry) and the limits that `with_prefix_limits` sets.
pub(crate) struct FakeExecutor {
    script: VecDeque<Vec<u32>>,
    eos: Vec<u32>,
    remaining: u32,
    /// The limits of the store, reported by `prefix_limits`.
    limits: StoreLimits,
    /// The real prefix store behind a lock, so a test can read it while the fake runs on the
    /// engine thread.
    prefix: Arc<Mutex<PrefixStore<(), u32>>>,
    revision: u32,
    deny_saves: bool,
    /// Fails the next snapshot with this error, before the store sees its result.
    fail_snapshot: Option<EngError>,
}

impl FakeExecutor {
    pub(crate) fn new(script: Vec<Vec<u32>>, eos: &[u32], max_tokens: u32) -> Self {
        Self {
            script: script.into(),
            eos: eos.to_vec(),
            remaining: max_tokens,
            limits: StoreLimits::default(),
            prefix: Arc::new(Mutex::new(PrefixStore::default())),
            revision: 1,
            deny_saves: false,
            fail_snapshot: None,
        }
    }

    /// Sets the limits of the prefix store (decision 0036, addendum 2). The default limits are
    /// the addendum 1 limits: zero roots, one entry.
    pub(crate) fn with_prefix_limits(&mut self, roots: usize, entries: usize) {
        self.limits = StoreLimits { roots, entries };
        *lock_store(&self.prefix) = PrefixStore::with_limits(self.limits);
    }

    /// The handle through which a test reads the store while the fake runs on its engine
    /// thread.
    pub(crate) fn prefix_store(&self) -> Arc<Mutex<PrefixStore<(), u32>>> {
        Arc::clone(&self.prefix)
    }

    /// The stored prefix keys as `(length, is_root)`, roots first, for test assertions.
    pub(crate) fn stored_prefixes(&self) -> Vec<(usize, bool)> {
        lock_store(&self.prefix).keys()
    }

    /// Sets the numeric path revision. A saved prefix restores only under the revision that built it.
    pub(crate) fn set_numeric_revision(&mut self, revision: u32) {
        self.revision = revision;
    }

    /// Makes every later snapshot report a denied memory charge (`MemoryPlan`). The real store
    /// applies its own rules to the denial: it drops victims before each retry and skips the
    /// save when none is left, as it does for the resident executors.
    pub(crate) fn deny_saves(&mut self) {
        self.deny_saves = true;
    }

    /// Fails the next snapshot with `error`, so the save fails like a real snapshot failure.
    /// The failure waits for a snapshot call: a save that the store refuses or serves as a use
    /// without a snapshot leaves it armed.
    pub(crate) fn fail_snapshots(&mut self, error: EngError) {
        self.fail_snapshot = Some(error);
    }
}

impl ExecutorOps for FakeExecutor {
    type Session = FakeSession;

    fn history<'s>(&self, st: &'s FakeSession) -> &'s [u32] {
        &st.history
    }

    fn reset(&mut self, st: &mut FakeSession) -> Result<()> {
        st.core.reset(&mut st.step)?;
        st.history.clear();
        st.finished = false;
        Ok(())
    }

    fn prefill(&mut self, st: &mut FakeSession, toks: &[u32]) -> Result<()> {
        // The fake models no context limit, so every prompt fits.
        st.core.prefill(&mut st.step, toks, usize::MAX)?;
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
                        let _ = self.reset(st);
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
                    let _ = self.reset(st);
                    return Err(error);
                }
            }
            if finish == Some(DecodeFinish::Cancelled) {
                // Decision D8: a cancelled job resets the session.
                let _ = self.reset(st);
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

    fn prefix_limits(&self) -> StoreLimits {
        self.limits
    }

    fn save_prefix(&mut self, st: &FakeSession, key: &[u32], kind: SaveKind) -> Result<bool> {
        // The committed tokens include every scripted token, so a save after a decode is
        // refused before the session rules run.
        if st.history != key {
            return Err(EngError::BadRequest(
                "prefix key differs from the committed tokens".to_string(),
            ));
        }
        let fed = st.step.fed();
        let path = self.revision;
        let deny = self.deny_saves;
        let failing = &mut self.fail_snapshot;
        let mut store = lock_store(&self.prefix);
        st.core.save_prefix(&mut *store, key, path, fed, kind, || {
            if let Some(error) = failing.take() {
                return Err(error);
            }
            if deny {
                return Err(EngError::MemoryPlan(
                    "the fake denies the prefix charge".to_string(),
                ));
            }
            Ok(())
        })
    }

    fn restore_prefix(&mut self, st: &mut FakeSession, prompt: &[u32]) -> Result<usize> {
        let path = self.revision;
        let mut store = lock_store(&self.prefix);
        let covered = st
            .core
            .restore_prefix(&mut *store, prompt, path, |_| Ok(()))?;
        if covered > 0 {
            // The restored state fed every key token but the pending one, and the scripted
            // tokens of an earlier decode never survive a restore.
            st.step.fed = covered - 1;
            st.history = st.core.history().to_vec();
            st.finished = false;
        }
        Ok(covered)
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
    use crate::family::session::SaveKind;

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
        assert!(
            executor
                .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Tip)
                .expect("save")
        );
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
        executor
            .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Tip)
            .expect("save");
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
        executor
            .save_prefix(&session, &[1, 2], SaveKind::Tip)
            .expect("save");
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
        executor
            .save_prefix(&first, &[1, 2], SaveKind::Tip)
            .expect("save first");
        let mut second = FakeSession::default();
        executor.prefill(&mut second, &[3, 4]).expect("prefill");
        executor
            .save_prefix(&second, &[3, 4], SaveKind::Tip)
            .expect("save second");
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
        // A job start restores before it saves. This restore matches nothing, so the entry
        // [3, 4] that the last restore protected is a plain entry again.
        let mut third = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut third, &[5, 6, 7])
                .expect("restore"),
            0,
            "nothing stored matches the third prompt"
        );
        executor.prefill(&mut third, &[5, 6]).expect("prefill");
        executor.deny_saves();
        // The default store holds one entry, so the save drops [3, 4] to make room. The
        // snapshot is then denied and no victim is left: the save is skipped.
        let saved = executor
            .save_prefix(&third, &[5, 6], SaveKind::Tip)
            .expect("denied save");
        assert!(!saved, "a denied save reports false");
        assert!(
            executor.stored_prefixes().is_empty(),
            "the store rules dropped the old entry and the denied save added none"
        );
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
    fn a_denied_save_keeps_the_entry_that_the_last_restore_protected() {
        // The rule the old wipe hid: the restored entry is never a victim, so a denied save of
        // an unrelated key keeps it and the store still restores it.
        let (mut executor, first) = fake_with_prefilled(&[1, 2]);
        executor
            .save_prefix(&first, &[1, 2], SaveKind::Tip)
            .expect("save first");
        let mut restored = FakeSession::default();
        assert_eq!(
            executor
                .restore_prefix(&mut restored, &[1, 2, 5])
                .expect("restore"),
            2,
            "the first key restores"
        );
        let mut unrelated = FakeSession::default();
        executor.prefill(&mut unrelated, &[7, 8]).expect("prefill");
        executor.deny_saves();
        let saved = executor
            .save_prefix(&unrelated, &[7, 8], SaveKind::Tip)
            .expect("denied save");
        assert!(!saved, "a denied save reports false");
        assert_eq!(
            executor.stored_prefixes(),
            vec![(2, false)],
            "the protected entry stays"
        );
    }

    #[test]
    fn an_armed_snapshot_failure_waits_for_a_snapshot_call() {
        // The second save repeats a stored key, so the store serves it as a use and never calls
        // the snapshot. The armed failure must survive it and fail the next real snapshot.
        let (mut executor, first) = fake_with_prefilled(&[1, 2]);
        executor
            .save_prefix(&first, &[1, 2], SaveKind::Tip)
            .expect("save first");
        executor.fail_snapshots(EngError::Checkpoint("snapshot boom".to_string()));
        assert!(
            executor
                .save_prefix(&first, &[1, 2], SaveKind::Tip)
                .expect("a repeated key is a use"),
            "the repeat counts as done"
        );
        let mut second = FakeSession::default();
        executor.prefill(&mut second, &[3, 4]).expect("prefill");
        let failed = executor.save_prefix(&second, &[3, 4], SaveKind::Tip);
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(_))),
            "got {failed:?}"
        );
    }

    #[test]
    fn a_save_rejects_a_key_that_differs_from_the_committed_tokens() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3]);
        let rejected = executor.save_prefix(&session, &[1, 2], SaveKind::Tip);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "got {rejected:?}"
        );
    }

    #[test]
    fn the_fake_saves_a_root_and_a_tip_through_the_real_store() {
        // The addendum 2 limits hold two roots and six entries. A Root save of the key of a
        // stored tip promotes that entry to a pinned root (clarification 1, item 3), so the
        // store ends with one pinned root and no entry.
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3, 4]);
        executor.with_prefix_limits(2, 6);
        executor
            .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Tip)
            .expect("save tip");
        assert_eq!(
            executor.stored_prefixes(),
            vec![(4, false)],
            "the tip alone"
        );
        assert!(
            executor
                .save_prefix(&session, &[1, 2, 3, 4], SaveKind::Root)
                .expect("save root")
        );
        assert_eq!(
            executor.stored_prefixes(),
            vec![(4, true)],
            "the root replaced the tip of the same key"
        );
    }

    #[test]
    fn a_failing_snapshot_fails_the_save_and_leaves_the_store_unchanged() {
        let (mut executor, session) = fake_with_prefilled(&[1, 2, 3]);
        executor.fail_snapshots(EngError::Checkpoint("snapshot boom".to_string()));
        let failed = executor.save_prefix(&session, &[1, 2, 3], SaveKind::Tip);
        assert!(
            matches!(&failed, Err(EngError::Checkpoint(_))),
            "got {failed:?}"
        );
        assert!(executor.stored_prefixes().is_empty(), "the store changed");
    }
}
