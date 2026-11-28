//! The serving entry of the Raon-1 family: the engine start that `enginesrv` calls for a
//! `raon1` entry. The weights are one GGUF file; the tokenizer bundle comes from the same file.
//!
//! One session per engine. The token side is the shared `DecodeSession` (`family::session`):
//! it owns the committed tokens, the pending token and the finish rules. The model side is a
//! `TokenStepper` over the device forward pass: prompt rows run as chunks without the head, and
//! the pending token's step returns the logits that the host sampler draws from.

use std::path::Path;

use crate::{
    EngError, family::qwen::flash_next::factory::StartedEngine, sched::worker::ReadyNote,
    tokenizer::TokenizerBundle,
};

/// The server-family key a config names to select this family.
pub(crate) const FAMILY: &str = "raon1";

/// Starts the engine thread for the GGUF at `model_path` (the file, or a directory with one
/// Raon-1 GGUF). `context_limit` is the configured limit; `None` derives it from device memory.
#[cfg(not(feature = "native-cuda"))]
pub(crate) fn start_raon1_engine(
    model_path: &Path,
    context_limit: Option<u32>,
    bundle: TokenizerBundle,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let _ = (model_path, context_limit, bundle, ready_note);
    Err(EngError::BadRequest(
        "engine config: Raon-1 serves on CUDA; build with --features native-cuda on Linux".into(),
    ))
}

#[cfg(feature = "native-cuda")]
pub(crate) use device::start_raon1_engine;

#[cfg(feature = "native-cuda")]
mod device {
    use super::*;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::family::raon::raon1::cuda::{CHUNK, HeadRows, Raon1Device};
    use crate::family::sampler::Sampler;
    use crate::family::session::{
        DecodeSession, Limits, TokenStepper, check_prompt, sample_logits,
    };
    use crate::sched::worker::spawn_engine_noted;
    use crate::tokenizer::DecodeBatch;

    pub(crate) struct Raon1Session {
        core: DecodeSession,
        pos: usize,
    }

    struct Raon1Serve {
        model: Raon1Device,
        eos: Vec<u32>,
        vocab: usize,
    }

    /// The model side of one session: the device and the number of positions fed.
    struct Step<'a> {
        model: &'a mut Raon1Device,
        pos: &'a mut usize,
    }

    impl TokenStepper for Step<'_> {
        const OPERATION: &'static str = "raon1";

        fn feed(&mut self, token: u32) -> Result<(), EngError> {
            self.feed_rows(&[token])
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
            self.model.feed(tokens, *self.pos, HeadRows::None)?;
            *self.pos += tokens.len();
            Ok(())
        }

        fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
            let mut rows = self.model.feed(&[token], *self.pos, HeadRows::Last)?;
            *self.pos += 1;
            rows.pop().ok_or_else(|| EngError::Backend {
                operation: "raon1",
                message: "the head returned no logits row".into(),
            })
        }

        fn next_token(
            &mut self,
            token: u32,
            sampler: &mut Sampler,
            sampler_vocab: usize,
        ) -> Result<u32, EngError> {
            let on_device = self.model.takes_tokens()
                && sampler.is_greedy()
                && !sampler.capture_armed()
                && sampler.first_top2_margin().is_some();
            if !on_device {
                let logits = self.logits(token)?;
                return sample_logits(Self::OPERATION, &logits, sampler, sampler_vocab);
            }
            let picked = self.model.decode_token(token, *self.pos, sampler_vocab)?;
            *self.pos += 1;
            sampler.record_device_greedy(picked);
            Ok(picked)
        }

        fn fed(&self) -> usize {
            *self.pos
        }

        fn reset(&mut self) -> Result<(), EngError> {
            self.model.reset()?;
            *self.pos = 0;
            Ok(())
        }
    }

    impl ExecutorOps for Raon1Serve {
        type Session = Raon1Session;

        fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
            st.core.history()
        }

        fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
            st.core.reset(&mut Step {
                model: &mut self.model,
                pos: &mut st.pos,
            })
        }

        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            check_prompt(prompt_tokens, self.model.cells())
        }

        fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
            let context = self.model.cells();
            st.core.prefill(
                &mut Step {
                    model: &mut self.model,
                    pos: &mut st.pos,
                },
                toks,
                context,
            )
        }

        fn decode(
            &mut self,
            st: &mut Self::Session,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            let limits = Limits {
                eos: &self.eos,
                sampler_vocab: self.vocab,
                context_limit: self.model.cells(),
            };
            st.core.decode(
                &mut Step {
                    model: &mut self.model,
                    pos: &mut st.pos,
                },
                generation,
                &limits,
            )
        }

        fn boundary(&mut self, _st: &mut Self::Session) {}

        fn prefill_piece(&self) -> Option<usize> {
            Some(CHUNK)
        }
    }

    /// The engine thread loads the model and announces the context it planned; the caller
    /// needs it before the weights finish loading, to bound requests.
    pub(crate) fn start_raon1_engine(
        model_path: &Path,
        context_limit: Option<u32>,
        bundle: TokenizerBundle,
        ready_note: ReadyNote,
    ) -> Result<StartedEngine, EngError> {
        let (_, files) = crate::family::gguf_select::candidates(model_path)?;
        let path = files
            .first()
            .map(|file| file.path.clone())
            .ok_or_else(|| EngError::Checkpoint("raon1 serve: no GGUF file found".into()))?;
        let eos = bundle.eos_token_ids().to_vec();
        let vocab = usize::try_from(bundle.vocab_size()).map_err(|_| {
            EngError::Checkpoint("raon1 serve: the vocabulary exceeds usize".into())
        })?;
        let wanted = context_limit.map(|limit| limit as usize);
        let (announce, resolved) = std::sync::mpsc::sync_channel::<Result<u32, String>>(1);
        let engine = spawn_engine_noted(CHUNK, ready_note, move |_| {
            let model = match Raon1Device::load(&path, wanted, &mut |cells| {
                let _ = announce.send(Ok(u32::try_from(cells).unwrap_or(u32::MAX)));
            }) {
                Ok(model) => model,
                Err(error) => {
                    let _ = announce.send(Err(error.describe()));
                    return Err(error);
                }
            };
            if eos.is_empty() || eos.iter().any(|&id| id as usize >= model.vocab()) {
                let message = "EOS ids must be non-empty and within the vocabulary".to_owned();
                let _ = announce.send(Err(message.clone()));
                return Err(EngError::Checkpoint(format!("raon1 serve: {message}")));
            }
            if vocab > model.vocab() {
                let message = "the tokenizer vocabulary exceeds the checkpoint's".to_owned();
                let _ = announce.send(Err(message.clone()));
                return Err(EngError::Checkpoint(format!("raon1 serve: {message}")));
            }
            let serve = Raon1Serve { model, eos, vocab };
            Ok((
                serve,
                Raon1Session {
                    core: DecodeSession::default(),
                    pos: 0,
                },
            ))
        })?;
        let context = match resolved.recv() {
            Ok(Ok(context)) => context,
            Ok(Err(cause)) => {
                return Err(EngError::Checkpoint(format!(
                    "raon1 start failed before the model was ready: {cause}"
                )));
            }
            Err(_) => {
                return Err(EngError::Checkpoint(format!(
                    "raon1 start failed before the model was ready: {}",
                    engine
                        .handle()
                        .failure()
                        .unwrap_or_else(|| "the engine thread ended without a cause".into())
                )));
            }
        };
        Ok(StartedEngine {
            engine,
            bundle,
            context_limit: context,
        })
    }
}
