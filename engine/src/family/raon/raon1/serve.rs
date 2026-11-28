//! The serving entry of the Raon-1 family: the engine start that `enginesrv` calls for a
//! `raon1` entry. The weights are one GGUF file; the tokenizer bundle comes from the same file.
//!
//! One session per engine, driven by the shared [`SerialDecode`] executor with prompt reuse over
//! [`DecodeSession`]. The model side is [`RaonStep`], the shared [`DeviceStepper`] over
//! [`Raon1Device`]: prompt rows run as chunks without the head, greedy decode picks tokens on the
//! device when the embedding table is resident, and other requests read the logits row back to the
//! host sampler.

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

#[cfg(all(test, feature = "native-cuda"))]
pub(crate) use device::RaonStep;
#[cfg(feature = "native-cuda")]
pub(crate) use device::start_raon1_engine;

#[cfg(feature = "native-cuda")]
mod device {
    use super::*;
    use crate::family::executor::SerialDecode;
    use crate::family::raon::raon1::cuda::{CHUNK, HeadRows, Raon1Device};
    use crate::family::session::{DecodeSession, Kept};
    use crate::family::step::stepper::{DeviceStep, DeviceStepper, ShortlistIo};
    use crate::sched::worker::spawn_engine_noted;

    /// The served stepper of Raon-1: shared position/generation/failure tracking over
    /// [`Raon1Device`].
    pub(crate) type RaonStep = DeviceStepper<Raon1Device>;

    impl DeviceStep for Raon1Device {
        const OPERATION: &'static str = "raon1";

        fn feed_rows(&mut self, tokens: &[u32], pos: usize) -> Result<(), EngError> {
            self.feed(tokens, pos, HeadRows::None)?;
            Ok(())
        }

        fn decode_to_logits(&mut self, token: u32, pos: usize) -> Result<(), EngError> {
            self.step_decode(token, pos)
        }

        fn shortlist_io(&self) -> Option<ShortlistIo<'_>> {
            Some(self.shortlist_io())
        }

        fn takes_tokens(&self) -> bool {
            self.takes_tokens()
        }

        fn decode_token(
            &mut self,
            token: u32,
            pos: usize,
            sampler_vocab: usize,
        ) -> Result<u32, EngError> {
            self.decode_token(token, pos, sampler_vocab)
        }

        fn reset(&mut self) -> Result<(), EngError> {
            self.reset()
        }

        fn reset_running(&mut self) -> Result<(), EngError> {
            self.reset_running()
        }

        fn kept_bytes(&self) -> u64 {
            self.kept_bytes()
        }

        fn keep(&self, rows: usize) -> Result<Kept, EngError> {
            self.keep(rows)
        }

        fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
            self.put_back(kept)
        }
    }

    /// The engine thread loads the model and announces the context it planned once validated.
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
            let opened = (|| -> Result<Raon1Device, EngError> {
                let model = Raon1Device::load(&path, wanted, &mut |_| {})?;
                if eos.is_empty() || eos.iter().any(|&id| id as usize >= model.vocab()) {
                    return Err(EngError::Checkpoint(
                        "raon1 serve: EOS ids must be non-empty and within the vocabulary".into(),
                    ));
                }
                if vocab > model.vocab() {
                    return Err(EngError::Checkpoint(
                        "raon1 serve: the tokenizer vocabulary exceeds the checkpoint's".into(),
                    ));
                }
                Ok(model)
            })();
            let model = match opened {
                Ok(model) => model,
                Err(error) => {
                    let _ = announce.send(Err(error.describe()));
                    return Err(error);
                }
            };
            let cells = model.cells();
            let _ = announce.send(Ok(u32::try_from(cells).unwrap_or(u32::MAX)));
            let budget =
                crate::family::mem::host_store_budget(crate::family::mem::process_resident_bytes());
            let executor = SerialDecode::new(RaonStep::over(model), eos, vocab, cells).with_reuse(
                CHUNK,
                budget,
                crate::family::step::keep::admit,
            );
            Ok((executor, DecodeSession::default()))
        })?;
        let context = match resolved.recv() {
            Ok(Ok(context)) => context,
            Ok(Err(cause)) => {
                let _ = engine.shutdown();
                return Err(EngError::Checkpoint(format!(
                    "raon1 start failed before the model was ready: {cause}"
                )));
            }
            Err(_) => {
                let report = engine.shutdown();
                return Err(EngError::Checkpoint(format!(
                    "raon1 start failed before the model was ready: {}",
                    report
                        .failure
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
