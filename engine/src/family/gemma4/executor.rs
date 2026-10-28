// Staged until the server factory constructs the Gemma engine; remove this allow then.
#![allow(dead_code)]

//! The Humanizer executor: the engine worker drives the resident decode step
//! through the ExecutorOps seam.
//!
//! The session holds the committed tokens and the next position. The model keeps
//! the KV8 planes: a cell is written when a token is fed and read only below the
//! position word, so a reset that clears the history and the position is exact.
//!
//! Sampling happens on the device (sample_gemma); this adapter never reads the
//! logits row back. It reads a token id, its log probability and a status word per step.
//!
//! Limits of this slice:
//! - Prefill feeds one token per step, so a long prompt costs a step per token.
//! - No prefix store: save_prefix and restore_prefix keep their defaults.
//! - The sampler does not yet match the canonical bit-exact contract.

use super::resident::{CELLS, Gemma, VOCAB};
use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::kernels::cuda::resident::sample_gemma::{SampleGemmaCall, SampleParams};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};
use crate::tokenizer::{DecodeBatch, DecodeFinish};

/// Sampling settings of the Humanizer request contract.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sampling {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
    pub(crate) seed: u64,
}

impl Default for Sampling {
    /// The pinned README settings: temperature 1.0, top-p 0.95, top-k 0, min-p 0.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
            seed: 0,
        }
    }
}

/// The committed tokens and the next position of one session.
#[derive(Default)]
pub(crate) struct GemmaSession {
    history: Vec<u32>,
    pos: usize,
}

/// The loaded model, its sampling settings and the per-step sampler buffers.
pub(crate) struct GemmaExecutor {
    model: Gemma,
    sampling: Sampling,
    eos: u32,
    sampler: DeviceSampler,
}

impl GemmaExecutor {
    /// Loads the pinned checkpoint and allocates the sampler buffers.
    pub(crate) fn load(
        path: &std::path::Path,
        header_sha256: Option<&str>,
        sampling: Sampling,
        eos: u32,
    ) -> Result<Self, EngError> {
        let model = super::resident::load(path, header_sha256)?;
        let ctx = &model.ctx;
        Ok(Self {
            sampler: DeviceSampler::new(ctx, VOCAB)?,
            model,
            sampling,
            eos,
        })
    }

    /// Feeds one token and advances the position.
    fn feed(&mut self, st: &mut GemmaSession, token: u32) -> Result<(), EngError> {
        self.model.decode(token, st.pos)?;
        st.pos += 1;
        st.history.push(token);
        Ok(())
    }

    /// Feeds a prompt token whose logits are never read, so the output norm and the
    /// tied-embedding projection are skipped and nothing is synchronized.
    fn feed_hidden(&mut self, st: &mut GemmaSession, token: u32) -> Result<(), EngError> {
        self.model.decode_hidden(token, st.pos)?;
        st.pos += 1;
        st.history.push(token);
        Ok(())
    }

    /// Draws one token from the capped logits of the last fed token.
    fn draw(&self, st: &GemmaSession) -> Result<(u32, f32), EngError> {
        self.sampler.draw(
            &self.model.ctx,
            &self.model.stream,
            self.model.logits_capped.buf.range(0, VOCAB * 4)?,
            SampleParams {
                temperature: self.sampling.temperature,
                top_k: self.sampling.top_k,
                top_p: self.sampling.top_p,
                min_p: self.sampling.min_p,
                seed: self.sampling.seed,
                position: st.pos as u64,
            },
        )
    }
}

impl ExecutorOps for GemmaExecutor {
    type Session = GemmaSession;

    fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
        check_prompt(prompt_tokens)
    }

    fn history<'s>(&self, st: &'s GemmaSession) -> &'s [u32] {
        &st.history
    }

    fn reset(&mut self, st: &mut GemmaSession) -> Result<(), EngError> {
        st.history.clear();
        st.pos = 0;
        Ok(())
    }

    fn prefill(&mut self, st: &mut GemmaSession, toks: &[u32]) -> Result<(), EngError> {
        if toks.is_empty() {
            return Ok(());
        }
        if st.pos + toks.len() > CELLS {
            return Err(EngError::ContextLimit(format!(
                "feeding {} tokens at position {} exceeds the {} cell context",
                toks.len(),
                st.pos,
                CELLS
            )));
        }
        let mut batches = toks.chunks(Gemma::PREFILL_BATCH).peekable();
        while let Some(batch) = batches.next() {
            self.model
                .prefill_batch(batch, st.pos, batches.peek().is_none())?;
            st.pos += batch.len();
            st.history.extend_from_slice(batch);
        }
        Ok(())
    }

    fn decode(
        &mut self,
        st: &mut GemmaSession,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        decode_step(self.eos, self, st, generation, Self::draw, Self::feed)
    }

    fn boundary(&mut self, _st: &mut GemmaSession) {}
}

fn check_prompt(prompt_tokens: usize) -> Result<(), EngError> {
    if prompt_tokens == 0 {
        return Err(EngError::BadRequest(
            "the prompt holds no tokens".to_owned(),
        ));
    }
    if prompt_tokens >= CELLS {
        return Err(EngError::ContextLimit(format!(
            "the prompt holds {prompt_tokens} tokens, leaving no generation cell in the {CELLS} cell context"
        )));
    }
    Ok(())
}

// These private seams let tests exercise control flow without model weights.
struct DeviceSampler {
    token: DeviceBuffer,
    logprob: DeviceBuffer,
    status: DeviceBuffer,
    probs: DeviceBuffer,
    n: usize,
}

impl DeviceSampler {
    fn new(ctx: &CudaContext, n: usize) -> Result<Self, EngError> {
        Ok(Self {
            token: ctx.upload(&[0u8; 4])?,
            logprob: ctx.upload(&[0u8; 4])?,
            status: ctx.upload(&[0u8; 4])?,
            probs: ctx.upload(&vec![0u8; n * 4])?,
            n,
        })
    }

    fn draw(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        logits: DeviceRange<'_>,
        params: SampleParams,
    ) -> Result<(u32, f32), EngError> {
        ctx.sample_gemma(
            stream,
            SampleGemmaCall {
                token: self.token.range(0, 4)?,
                logprob: self.logprob.range(0, 4)?,
                status: self.status.range(0, 4)?,
                probs: self.probs.range(0, self.n * 4)?,
                logits,
                n: self.n,
                params,
            },
        )?;
        stream.synchronize()?;
        let mut status = [0u8; 4];
        ctx.download(&self.status, &mut status)?;
        let status = u32::from_le_bytes(status);
        if status != 0 {
            return Err(EngError::Backend {
                operation: "gemma sampler",
                message: format!("the sampler returned status {status}"),
            });
        }
        let mut bytes = [0u8; 4];
        ctx.download(&self.token, &mut bytes)?;
        let mut lp = [0u8; 4];
        ctx.download(&self.logprob, &mut lp)?;
        Ok((u32::from_le_bytes(bytes), f32::from_le_bytes(lp)))
    }
}

fn decode_step<T>(
    eos: u32,
    executor: &mut T,
    st: &mut GemmaSession,
    generation: &mut GenerationState,
    draw: impl FnOnce(&T, &GemmaSession) -> Result<(u32, f32), EngError>,
    feed: impl FnOnce(&mut T, &mut GemmaSession, u32) -> Result<(), EngError>,
) -> Result<DecodeBatch, EngError> {
    if generation.cancel.load(std::sync::atomic::Ordering::SeqCst) {
        return finish_decode(
            st,
            generation,
            Vec::new(),
            String::new(),
            DecodeFinish::Cancelled,
        );
    }
    if generation.remaining_tokens() == Some(0) {
        return finish_decode(
            st,
            generation,
            Vec::new(),
            String::new(),
            DecodeFinish::Length,
        );
    }
    if st.pos >= CELLS {
        return Err(EngError::ContextLimit(format!(
            "position {} reaches the {CELLS} cell context",
            st.pos
        )));
    }
    let (token, _logprob) = draw(executor, st)?;
    if token >= VOCAB as u32 {
        return Err(EngError::Backend {
            operation: "gemma sampler",
            message: format!("the sampler returned id {token}, above the vocabulary"),
        });
    }
    if token == eos {
        return finish_decode(
            st,
            generation,
            Vec::new(),
            String::new(),
            DecodeFinish::Stop,
        );
    }
    let text = generation.decoder.push(token)?;
    generation.record_generated(1);
    feed(executor, st, token)?;
    if generation.remaining_tokens() == Some(0) {
        return finish_decode(st, generation, vec![token], text, DecodeFinish::Length);
    }
    Ok(DecodeBatch {
        tokens: vec![token],
        text,
        finish: None,
    })
}

fn finish_decode(
    st: &mut GemmaSession,
    generation: &mut GenerationState,
    tokens: Vec<u32>,
    mut text: String,
    reason: DecodeFinish,
) -> Result<DecodeBatch, EngError> {
    match generation.decoder.finish() {
        Ok(tail) => text.push_str(&tail),
        Err(error) => {
            *st = GemmaSession::default();
            return Err(error);
        }
    }
    if reason == DecodeFinish::Cancelled {
        *st = GemmaSession::default();
    }
    Ok(DecodeBatch {
        tokens,
        text,
        finish: Some(reason),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::family::gemma4::resident::CELLS;
    use crate::tokenizer::{IncrementalDecoder, Tokenizer};
    use std::cell::Cell;

    /// Builds a generation state over the model's own tokenizer, so the decoder
    /// accepts the ids the sampler returns. Set GEMMA4_TOKENIZER_JSON to the path
    /// of tokenizer.json; the test returns early without it.
    fn state() -> Option<GenerationState> {
        let path = std::env::var("GEMMA4_TOKENIZER_JSON").ok()?;
        let bytes = std::fs::read(path).expect("tokenizer.json");
        let tokenizer = Tokenizer::from_bytes(&bytes).expect("tokenizer");
        Some(GenerationState::new(IncrementalDecoder::new(&tokenizer)))
    }

    /// Fills the pinned 8,192-cell context, then decodes four steps, so the
    /// memory plan and the thermals are measured at the stated limit.
    /// Set GEMMA4_CONTEXT_TOKENS to the cell count.
    #[test]
    #[ignore = "requires the pinned file, a device and GEMMA4_CONTEXT_TOKENS"]
    fn fills_the_stated_context() {
        let Ok(path) = std::env::var("GEMMA4_Q8_FILE") else {
            return;
        };
        let Ok(cells) = std::env::var("GEMMA4_CONTEXT_TOKENS") else {
            return;
        };
        let cells: usize = cells.parse().expect("context tokens");
        assert!(cells > 8 && cells <= CELLS, "context above the plan");
        let header = std::env::var("GEMMA4_Q8_HEADER_SHA256").ok();
        let load_started = std::time::Instant::now();
        let mut engine = GemmaExecutor::load(
            std::path::Path::new(&path),
            header.as_deref(),
            Sampling::default(),
            1,
        )
        .expect("load");
        eprintln!("timing load {:?}", load_started.elapsed());
        let body = cells - 4;
        let prompt: Vec<u32> = (0..body).map(|i| 2 + (i as u32 % 1000)).collect();
        let mut session = GemmaSession::default();
        engine.check_prompt(body).expect("body fits");
        let prefill_started = std::time::Instant::now();
        for chunk in prompt.chunks(64) {
            engine.prefill(&mut session, chunk).expect("prefill");
        }
        eprintln!(
            "timing prefill {} tokens {:?}",
            body,
            prefill_started.elapsed()
        );
        let Some(mut generation) = state() else {
            return;
        };
        let mut produced = 0usize;
        for _ in 0..4 {
            let step_started = std::time::Instant::now();
            let batch = engine
                .decode(&mut session, &mut generation)
                .expect("decode");
            eprintln!(
                "timing decode at {} {:?} produced {}",
                body + produced,
                step_started.elapsed(),
                batch.tokens.len()
            );
            produced += batch.tokens.len();
            if batch.finish.is_some() {
                break;
            }
        }
        assert_eq!(engine.history(&session).len(), body + produced);
        assert!(produced >= 1, "no token was produced");
    }

    /// Prefill then decode three steps against the pinned checkpoint on a device.
    #[test]
    #[ignore = "requires the pinned Humanizer file and a device"]
    fn prefills_and_decodes_three_tokens() {
        let Ok(path) = std::env::var("GEMMA4_Q8_FILE") else {
            return;
        };
        let header = std::env::var("GEMMA4_Q8_HEADER_SHA256").ok();
        let load_started = std::time::Instant::now();
        let mut engine = GemmaExecutor::load(
            std::path::Path::new(&path),
            header.as_deref(),
            Sampling::default(),
            1,
        )
        .expect("load");
        eprintln!("timing load {:?}", load_started.elapsed());
        let mut session = GemmaSession::default();
        engine.check_prompt(2).expect("two prompt tokens fit");
        let prefill_started = std::time::Instant::now();
        engine.prefill(&mut session, &[2, 17]).expect("prefill");
        eprintln!("timing prefill 2 tokens {:?}", prefill_started.elapsed());
        assert_eq!(engine.history(&session), &[2, 17]);
        let Some(mut generation) = state() else {
            return;
        };
        let mut produced = 0usize;
        for _ in 0..3 {
            let step_started = std::time::Instant::now();
            let batch = engine
                .decode(&mut session, &mut generation)
                .expect("decode");
            eprintln!("timing decode {:?}", step_started.elapsed());
            produced += batch.tokens.len();
            if batch.finish.is_some() {
                break;
            }
        }
        assert!(produced >= 1, "no token was produced");
        assert_eq!(engine.history(&session).len(), 2 + produced);
        engine.reset(&mut session).expect("reset");
        assert!(engine.history(&session).is_empty());
    }

    struct DeviceFixture {
        ctx: CudaContext,
        stream: DeviceStream,
        logits: DeviceBuffer,
        sampler: DeviceSampler,
        temperature: f32,
        draws: Cell<usize>,
        feeds: usize,
    }

    impl DeviceFixture {
        fn new(token: u32) -> Option<Self> {
            let count = crate::kernels::cuda::device_count().expect("CUDA device count");
            eprintln!("CUDA devices visible: {count}");
            if count == 0 {
                assert!(
                    !std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
                        .is_ok_and(|value| !value.is_empty() && value != "0"),
                    "a CUDA device is required"
                );
                return None;
            }
            let ctx = CudaContext::new().expect("context");
            let stream = ctx.stream().expect("stream");
            let mut logits = [-30.0_f32; 32];
            logits[token as usize] = 30.0;
            let logits = logits
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            Some(Self {
                stream,
                logits: ctx.upload(&logits).expect("logits"),
                sampler: DeviceSampler::new(&ctx, 32).expect("sampler"),
                ctx,
                temperature: 0.0,
                draws: Cell::new(0),
                feeds: 0,
            })
        }

        fn draw(&self, st: &GemmaSession) -> Result<(u32, f32), crate::EngError> {
            self.draws.set(self.draws.get() + 1);
            self.sampler.draw(
                &self.ctx,
                &self.stream,
                self.logits.range(0, 32 * 4)?,
                SampleParams {
                    temperature: self.temperature,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed: 0,
                    position: st.pos as u64,
                },
            )
        }

        fn feed(&mut self, st: &mut GemmaSession, token: u32) -> Result<(), crate::EngError> {
            self.feeds += 1;
            st.pos += 1;
            st.history.push(token);
            Ok(())
        }

        fn decode(
            &mut self,
            st: &mut GemmaSession,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, crate::EngError> {
            decode_step(7, self, st, generation, Self::draw, Self::feed)
        }
    }

    #[test]
    fn sampler_status_overflow_is_a_backend_error() {
        let Some(mut device) = DeviceFixture::new(0) else {
            return;
        };
        device.temperature = 1e-40;
        let result = device.draw(&GemmaSession::default());
        assert!(
            matches!(result, Err(crate::EngError::Backend { operation: "gemma sampler", ref message }) if message.contains("2")),
            "expected sampler status 2, got {result:?}"
        );
        device.temperature = 0.0;
        assert_eq!(
            device
                .draw(&GemmaSession::default())
                .expect("next valid draw"),
            (0, 0.0)
        );
    }

    fn generation() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(br#"{
            "version":"1.0",
            "added_tokens":[{"id":7,"content":"<end>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"BPE","vocab":{"a":0,"st":1,"o":2,"pX":3,"\u00c3":4,"\u00a9":5,"!":6},"merges":[]}
        }"#).expect("fixture tokenizer");
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    #[test]
    fn cancellation_precedes_sampling_and_resets_the_session() {
        let Some(mut device) = DeviceFixture::new(0) else {
            return;
        };
        let mut session = GemmaSession {
            history: vec![2, 3],
            pos: 2,
        };
        let mut generation = generation().with_max_tokens(0);
        generation
            .cancel
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let batch = device
            .decode(&mut session, &mut generation)
            .expect("cancel");
        assert_eq!(device.feeds, 0, "a cancelled job must not feed");
        assert_eq!(device.draws.get(), 0, "a cancelled job must not sample");
        assert_eq!(
            batch.finish,
            Some(crate::tokenizer::DecodeFinish::Cancelled)
        );
        assert!(batch.tokens.is_empty());
        assert!(session.history.is_empty());
        assert_eq!(session.pos, 0);
        assert_eq!(generation.remaining_tokens(), Some(0));
    }

    #[test]
    fn a_zero_token_budget_finishes_without_sampling_or_feeding() {
        let Some(mut device) = DeviceFixture::new(0) else {
            return;
        };
        let mut session = GemmaSession {
            history: vec![2, 3],
            pos: 2,
        };
        let mut generation = generation().with_max_tokens(0);
        let batch = device
            .decode(&mut session, &mut generation)
            .expect("length");
        assert_eq!(device.feeds, 0, "a zero-budget job must not feed");
        assert_eq!(device.draws.get(), 0, "a zero-budget job must not sample");
        assert_eq!(batch.finish, Some(DecodeFinish::Length));
        assert!(batch.tokens.is_empty());
        assert_eq!(session.history, [2, 3]);
        assert_eq!(session.pos, 2);
        assert_eq!(generation.remaining_tokens(), Some(0));
    }

    #[test]
    fn a_full_context_fails_before_sampling_or_mutating_generation() {
        let Some(mut device) = DeviceFixture::new(0) else {
            return;
        };
        let mut session = GemmaSession {
            history: vec![2; CELLS],
            pos: CELLS,
        };
        let mut generation = generation().with_max_tokens(1);
        let result = device.decode(&mut session, &mut generation);
        assert_eq!(device.feeds, 0, "a full context must not feed");
        assert_eq!(device.draws.get(), 0, "a full context must not sample");
        assert!(
            matches!(result, Err(crate::EngError::ContextLimit(_))),
            "{result:?}"
        );
        assert_eq!(session.history, vec![2; CELLS]);
        assert_eq!(session.pos, CELLS);
        assert_eq!(generation.remaining_tokens(), Some(1));
        assert_eq!(generation.decoder.finish().expect("no sampled text"), "");
    }

    #[test]
    fn eos_flushes_buffered_utf8_text() {
        let Some(mut device) = DeviceFixture::new(7) else {
            return;
        };
        let mut session = GemmaSession {
            history: vec![2],
            pos: 1,
        };
        let mut generation = generation().with_max_tokens(1);
        assert_eq!(generation.decoder.push(4).expect("partial UTF-8"), "");
        let batch = device.decode(&mut session, &mut generation).expect("EOS");
        assert_eq!(batch.text, "\u{fffd}");
        assert_eq!(batch.finish, Some(DecodeFinish::Stop));
        assert!(batch.tokens.is_empty());
        assert_eq!(device.feeds, 0);
        assert_eq!(session.history, [2]);
        assert_eq!(generation.remaining_tokens(), Some(1));
        assert_eq!(generation.decoder.finish().expect("already flushed"), "");
    }

    #[test]
    fn length_flushes_buffered_utf8_text() {
        let Some(mut device) = DeviceFixture::new(4) else {
            return;
        };
        let mut session = GemmaSession {
            history: vec![2],
            pos: 1,
        };
        let mut generation = generation().with_max_tokens(1);
        let batch = device
            .decode(&mut session, &mut generation)
            .expect("length");
        assert_eq!(batch.text, "\u{fffd}");
        assert_eq!(batch.finish, Some(DecodeFinish::Length));
        assert_eq!(batch.tokens, [4]);
        assert_eq!(device.feeds, 1);
        assert_eq!(session.history, [2, 4]);
        assert_eq!(generation.remaining_tokens(), Some(0));
        assert_eq!(generation.decoder.finish().expect("already flushed"), "");
    }

    #[test]
    fn a_full_context_prompt_is_refused_at_admission() {
        let Some(_device) = DeviceFixture::new(0) else {
            return;
        };
        let result = check_prompt(CELLS);
        assert!(
            matches!(result, Err(crate::EngError::ContextLimit(_))),
            "{result:?}"
        );
        assert!(check_prompt(CELLS - 1).is_ok());
        assert!(matches!(
            check_prompt(CELLS + 1),
            Err(crate::EngError::ContextLimit(_))
        ));
        assert!(matches!(
            check_prompt(0),
            Err(crate::EngError::BadRequest(_))
        ));
    }
}
