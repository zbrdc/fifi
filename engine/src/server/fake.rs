//! The development engine: it replays a fixed reply for every job.
//!
//! The family name `fake` selects it. It lets `enginesrv` run on any machine, and it serves the
//! client fixtures. It runs no model and makes no claim about any real family.
//! The tokenizer files of the engine are trusted as found, because the files carry no pin.

use super::config::EngineConfig;
use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::sched::worker::{Engine, spawn_engine};
use crate::tokenizer::{DecodeBatch, DecodeFinish, Tokenizer, TokenizerBundle, TokenizerPin};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The reply of a fake engine whose configuration sets none.
pub(crate) const DEFAULT_REPLY: &str = "Hello from the fake engine.";

/// The context limit of a fake engine whose configuration sets none.
pub(crate) const DEFAULT_CONTEXT_LIMIT: u32 = 4_096;

/// The tokens that one session of the fake executor has produced so far.
#[derive(Default)]
pub(crate) struct ReplySession {
    history: Vec<u32>,
    sent: usize,
    finished: bool,
}

/// Replays `reply` one token per `decode` call and then the end-of-sequence token.
/// It ignores the sampler. It ends the job with `Length` at the output limit of the job in
/// `GenerationState`, as every executor must.
pub(crate) struct ReplyExecutor {
    reply: Vec<u32>,
    eos: u32,
    delay: Duration,
}

impl ExecutorOps for ReplyExecutor {
    type Session = ReplySession;

    fn history<'s>(&self, st: &'s ReplySession) -> &'s [u32] {
        &st.history
    }

    fn reset(&mut self, st: &mut ReplySession) -> Result<(), EngError> {
        *st = ReplySession::default();
        Ok(())
    }

    fn prefill(&mut self, st: &mut ReplySession, toks: &[u32]) -> Result<(), EngError> {
        if toks.is_empty() {
            return Err(EngError::BadRequest("empty prefill chunk".to_string()));
        }
        st.history.extend_from_slice(toks);
        Ok(())
    }

    fn decode(
        &mut self,
        st: &mut ReplySession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        std::thread::sleep(self.delay);
        if st.finished {
            return Err(EngError::BadRequest("decode after finish".to_string()));
        }
        if st.history.is_empty() {
            return Err(EngError::BadRequest("decode before prefill".to_string()));
        }
        if generation.cancel.load(Ordering::SeqCst) {
            *st = ReplySession::default();
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text: generation.decoder.finish()?,
                finish: Some(DecodeFinish::Cancelled),
            });
        }
        if generation.remaining_tokens() == Some(0) {
            st.finished = true;
            return Ok(DecodeBatch {
                tokens: Vec::new(),
                text: generation.decoder.finish()?,
                finish: Some(DecodeFinish::Length),
            });
        }
        if let Some(&id) = self.reply.get(st.sent) {
            st.sent += 1;
            st.history.push(id);
            generation.record_generated(1);
            let mut text = generation.decoder.push(id)?;
            let finish = (generation.remaining_tokens() == Some(0)).then_some(DecodeFinish::Length);
            if finish.is_some() {
                st.finished = true;
                text.push_str(&generation.decoder.finish()?);
            }
            return Ok(DecodeBatch {
                tokens: vec![id],
                text,
                finish,
            });
        }
        st.finished = true;
        st.history.push(self.eos);
        Ok(DecodeBatch {
            tokens: vec![self.eos],
            text: generation.decoder.finish()?,
            finish: Some(DecodeFinish::Stop),
        })
    }

    fn boundary(&mut self, _st: &mut ReplySession) {}
}

fn checkpoint(message: String) -> EngError {
    EngError::Checkpoint(message)
}

/// Loads the four tokenizer files of `dir`. The pin comes from the files themselves.
pub(crate) fn load_bundle(dir: &Path) -> Result<TokenizerBundle, EngError> {
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .map_err(|error| checkpoint(format!("{name} could not be read: {error}")))
    };
    let tokenizer = read("tokenizer.json")?;
    let config = read("tokenizer_config.json")?;
    let template = read("chat_template.jinja")?;
    let generation = read("generation_config.json")?;
    let digest = |bytes: &[u8]| -> [u8; 32] { Sha256::digest(bytes).into() };
    let pin = TokenizerPin {
        tokenizer_json: digest(&tokenizer),
        tokenizer_config_json: digest(&config),
        chat_template_jinja: digest(&template),
        generation_config_json: digest(&generation),
        vocab_size: Tokenizer::from_bytes(&tokenizer)?.vocab_size(),
    };
    TokenizerBundle::load(dir, &pin)
}

/// A started fake engine, its tokenizer, and its context limit. Every family reports its own
/// context limit this way, and the router uses it for the 409 check and for the output limit.
pub(crate) struct FakeEngine {
    pub(crate) engine: Engine,
    pub(crate) bundle: Arc<TokenizerBundle>,
    pub(crate) context_limit: u32,
}

/// Starts the fake engine of a configuration entry.
pub(crate) fn start(config: &EngineConfig) -> Result<FakeEngine, EngError> {
    let bundle = Arc::new(load_bundle(&config.tokenizer_dir)?);
    let options = config.fake.clone().unwrap_or_default();
    let text = options.reply.as_deref().unwrap_or(DEFAULT_REPLY);
    let executor = ReplyExecutor {
        reply: bundle.tokenizer().encode(text)?,
        eos: bundle.eos_token_id(),
        delay: Duration::from_millis(options.token_delay_ms),
    };
    let engine = spawn_engine(512, move || Ok((executor, ReplySession::default())))?;
    Ok(FakeEngine {
        engine,
        bundle,
        context_limit: config.context_limit.unwrap_or(DEFAULT_CONTEXT_LIMIT),
    })
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_REPLY, FakeEngine, ReplyExecutor, ReplySession, load_bundle, start};
    use crate::EngError;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::sched::job::{Job, JobEvent};
    use crate::sched::worker::EngineState;
    use crate::server::config::{EngineConfig, FakeOptions};
    use crate::tokenizer::{DecodeFinish, IncrementalDecoder};
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    fn tokenizer_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer")
    }

    fn config(options: Option<FakeOptions>) -> EngineConfig {
        EngineConfig {
            id: "e1".to_string(),
            model_dir: PathBuf::from("unused"),
            tokenizer_dir: tokenizer_dir(),
            family: "fake".to_string(),
            model: None,
            context_limit: None,
            fake: options,
            flash_next: None,
            qwen36: None,
        }
    }

    /// Runs one job with a two-token prompt and returns its text and its events.
    fn run(fake: &FakeEngine) -> (String, Vec<JobEvent>) {
        let decoder = IncrementalDecoder::new(fake.bundle.tokenizer());
        run_with(fake, GenerationState::new(decoder))
    }

    fn run_with(fake: &FakeEngine, generation: GenerationState) -> (String, Vec<JobEvent>) {
        let (job, mut handle) = Job::new(vec![104, 105], generation).expect("job");
        fake.engine.handle().try_submit(job).expect("submit");
        let mut events = Vec::new();
        while let Some(event) = handle.events.blocking_recv() {
            events.push(event);
        }
        let text = events
            .iter()
            .filter_map(|event| match event {
                JobEvent::Batch(batch) => Some(batch.text.as_str()),
                _ => None,
            })
            .collect();
        (text, events)
    }

    #[test]
    fn the_default_engine_replays_the_default_reply_for_every_job() {
        let fake = start(&config(None)).expect("start");
        let (first, events) = run(&fake);
        assert_eq!(first, DEFAULT_REPLY);
        let finish = events.iter().find_map(|event| match event {
            JobEvent::Batch(batch) => batch.finish,
            _ => None,
        });
        assert_eq!(finish, Some(DecodeFinish::Stop));
        let (second, _) = run(&fake);
        assert_eq!(
            second, DEFAULT_REPLY,
            "the second job must start from a reset session"
        );
    }

    #[test]
    fn the_engine_reports_its_context_limit() {
        let default = start(&config(None)).expect("start");
        assert_eq!(default.context_limit, 4_096);
        let mut limited = config(None);
        limited.context_limit = Some(99);
        assert_eq!(start(&limited).expect("start").context_limit, 99);
    }

    #[test]
    fn a_configured_reply_replaces_the_default() {
        let options = FakeOptions {
            reply: Some("Fixed.".to_string()),
            token_delay_ms: 0,
        };
        let fake = start(&config(Some(options))).expect("start");
        assert_eq!(run(&fake).0, "Fixed.");
    }

    #[test]
    fn the_output_limit_of_the_job_ends_the_reply_with_length() {
        let fake = start(&config(None)).expect("start");
        let decoder = IncrementalDecoder::new(fake.bundle.tokenizer());
        let (text, events) = run_with(&fake, GenerationState::new(decoder).with_max_tokens(3));
        assert_eq!(text, "Hel");
        let batches: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                JobEvent::Batch(batch) => Some((batch.tokens.len(), batch.finish)),
                _ => None,
            })
            .collect();
        assert_eq!(
            batches,
            vec![(1, None), (1, None), (1, Some(DecodeFinish::Length))]
        );
    }

    #[test]
    fn an_output_limit_of_zero_ends_the_reply_before_any_token() {
        let fake = start(&config(None)).expect("start");
        let decoder = IncrementalDecoder::new(fake.bundle.tokenizer());
        let (text, events) = run_with(&fake, GenerationState::new(decoder).with_max_tokens(0));
        assert_eq!(text, "");
        let first = events.iter().find_map(|event| match event {
            JobEvent::Batch(batch) => Some((batch.tokens.len(), batch.finish)),
            _ => None,
        });
        assert_eq!(first, Some((0, Some(DecodeFinish::Length))));
    }

    #[test]
    fn the_token_delay_applies_to_every_decode() {
        let options = FakeOptions {
            reply: Some("ab".to_string()),
            token_delay_ms: 20,
        };
        let fake = start(&config(Some(options))).expect("start");
        let started = std::time::Instant::now();
        run(&fake);
        let elapsed = started.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(55),
            "three decodes of 20 ms took only {elapsed:?}"
        );
    }

    #[test]
    fn the_last_batch_holds_the_end_of_sequence_token() {
        let fake = start(&config(None)).expect("start");
        let (_, events) = run(&fake);
        let last = events
            .iter()
            .rev()
            .find_map(|event| match event {
                JobEvent::Batch(batch) => Some(batch),
                _ => None,
            })
            .expect("a batch");
        assert_eq!(last.tokens, vec![fake.bundle.eos_token_id()]);
        assert_eq!(fake.engine.handle().state(), EngineState::Ready);
    }

    /// An executor over the fixture tokenizer that replies "ab", and a fresh session.
    fn executor_and_state() -> (ReplyExecutor, ReplySession, GenerationState) {
        let bundle = load_bundle(&tokenizer_dir()).expect("load");
        let executor = ReplyExecutor {
            reply: bundle.tokenizer().encode("ab").expect("encode"),
            eos: bundle.eos_token_id(),
            delay: std::time::Duration::ZERO,
        };
        let generation = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()));
        (executor, ReplySession::default(), generation)
    }

    #[test]
    fn a_cancelled_decode_returns_a_cancelled_batch_and_resets_the_session() {
        let (mut executor, mut session, mut generation) = executor_and_state();
        executor
            .prefill(&mut session, &[104, 105])
            .expect("prefill");
        let first = executor
            .decode(&mut session, &mut generation)
            .expect("decode");
        assert_eq!(first.finish, None);
        assert_eq!(executor.history(&session), &[104, 105, first.tokens[0]]);
        generation.cancel.store(true, Ordering::SeqCst);
        let batch = executor
            .decode(&mut session, &mut generation)
            .expect("decode");
        assert_eq!(batch.finish, Some(DecodeFinish::Cancelled));
        assert!(batch.tokens.is_empty(), "{:?}", batch.tokens);
        assert!(
            executor.history(&session).is_empty(),
            "a cancelled job must reset the session"
        );
        assert_eq!(session.sent, 0);
        assert!(!session.finished);
    }

    #[test]
    fn a_job_cancelled_while_it_runs_ends_cancelled_and_the_next_job_replays_the_reply() {
        let options = FakeOptions {
            reply: Some("abcd".to_string()),
            token_delay_ms: 30,
        };
        let fake = start(&config(Some(options))).expect("start");
        let decoder = IncrementalDecoder::new(fake.bundle.tokenizer());
        let (job, mut handle) =
            Job::new(vec![104, 105], GenerationState::new(decoder)).expect("job");
        fake.engine.handle().try_submit(job).expect("submit");
        let first = handle.events.blocking_recv().expect("first batch");
        assert!(matches!(&first, JobEvent::Batch(batch) if batch.finish.is_none()));
        handle.cancel();
        let mut finish = None;
        while let Some(event) = handle.events.blocking_recv() {
            if let JobEvent::Batch(batch) = event {
                finish = batch.finish.or(finish);
            }
        }
        assert_eq!(finish, Some(DecodeFinish::Cancelled));
        let (text, _) = run(&fake);
        assert_eq!(text, "abcd", "the next job must replay the full reply");
    }

    #[test]
    fn decode_after_the_job_finished_is_an_error() {
        let (mut executor, mut session, mut generation) = executor_and_state();
        executor.prefill(&mut session, &[104]).expect("prefill");
        let mut finish = None;
        while finish.is_none() {
            finish = executor
                .decode(&mut session, &mut generation)
                .expect("decode")
                .finish;
        }
        assert_eq!(finish, Some(DecodeFinish::Stop));
        match executor.decode(&mut session, &mut generation) {
            Err(EngError::BadRequest(message)) => assert_eq!(message, "decode after finish"),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_finished_session_is_an_error_even_when_the_job_is_cancelled() {
        let (mut executor, mut session, mut generation) = executor_and_state();
        executor.prefill(&mut session, &[104]).expect("prefill");
        session.finished = true;
        generation.cancel.store(true, Ordering::SeqCst);
        match executor.decode(&mut session, &mut generation) {
            Err(EngError::BadRequest(message)) => assert_eq!(message, "decode after finish"),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn prefill_rejects_an_empty_chunk() {
        let (mut executor, mut session, _) = executor_and_state();
        match executor.prefill(&mut session, &[]) {
            Err(EngError::BadRequest(message)) => assert_eq!(message, "empty prefill chunk"),
            other => panic!("expected BadRequest, got {other:?}"),
        }
        assert!(executor.history(&session).is_empty());
    }

    #[test]
    fn decode_rejects_an_empty_history() {
        let (mut executor, mut session, mut generation) = executor_and_state();
        match executor.decode(&mut session, &mut generation) {
            Err(EngError::BadRequest(message)) => assert_eq!(message, "decode before prefill"),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_tokenizer_file_is_a_checkpoint_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let error = load_bundle(dir.path()).err().expect("error");
        match error {
            EngError::Checkpoint(message) => {
                assert!(message.contains("tokenizer.json"), "{message}")
            }
            other => panic!("expected Checkpoint, got {other:?}"),
        }
    }

    #[test]
    fn the_bundle_loader_reads_the_fixture_vocabulary() {
        let bundle = load_bundle(&tokenizer_dir()).expect("load");
        assert_eq!(bundle.vocab_size(), 258);
        assert_eq!(bundle.eos_token_id(), 256);
    }
}
