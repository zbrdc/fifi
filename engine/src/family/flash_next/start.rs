//! Starts the Flash-Next engine of one `enginesrv` config entry.
//!
//! The tokenizer bundle loads on the calling thread. It carries only portable data. The engine
//! thread builds everything native inside the `make` closure (`FlashExecutor::open`), so no native
//! value crosses a thread boundary. The tokenizer files come from `tokenizer_dir` and are compared
//! with the artifact manifest; a file that differs logs a warning and is served as found.

use super::artifact::tokenizer_pin;
use super::factory::CONTEXT_LIMIT;
use super::resident::executor::{FlashExecutor, FlashRules};
use super::settings::FlashSettings;
use crate::{
    EngError,
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::TokenizerBundle,
};

/// Loads and pins the tokenizer, then starts the engine thread. The thread loads the weights, so
/// the engine state stays `Loading` until the weights are resident (minutes on the target Mac).
/// Returns the engine and the pinned tokenizer bundle that renders and encodes its prompts.
pub(crate) fn start(
    settings: &FlashSettings,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    let manifest = std::fs::read(&settings.manifest).map_err(|error| {
        EngError::Checkpoint(format!("{}: {error}", settings.manifest.display()))
    })?;
    let pin = tokenizer_pin(&manifest, settings.tokenizer_vocab_size)?;
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &pin)?;
    let eos = bundle.eos_token_ids().to_vec();
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".to_string()))?;
    let prefill_chunk = usize::try_from(settings.prefill_chunk)
        .map_err(|_| EngError::BadRequest("engine config: prefill_chunk exceeds usize".into()))?;
    let settings = settings.clone();
    let rules = FlashRules {
        eos,
        sampler_vocab,
        expert_prefetch: settings.expert_prefetch,
    };
    let engine = spawn_engine_noted(prefill_chunk, ready_note, move |_| {
        let device = crate::family::mlx_budget::DeviceBudget::detect()?;
        let (mut executor, session) = FlashExecutor::open_with_admission(
            &settings.artifact,
            settings.cache_limit,
            rules,
            device,
            prefix_reuse,
        )?;
        if usize::try_from(CONTEXT_LIMIT).ok() != Some(executor.context_limit()) {
            return Err(EngError::Checkpoint(format!(
                "the executor accepts {} positions, but the server reports {CONTEXT_LIMIT}",
                executor.context_limit()
            )));
        }
        if let Some(line) = executor.configure_prefill(&session, &settings, prefix_reuse, device)? {
            eprintln!("{line}");
        }
        Ok((executor, session))
    })?;
    Ok((engine, bundle))
}
