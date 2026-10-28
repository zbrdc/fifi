//! The engine factory for `family: "qwen3_8_flash_next"` entries of the `enginesrv` config
//! (decision 0026; packet H8c).
//!
//! The server start-up code (`server/run.rs`) calls it for the entries of this family. Order of
//! checks: the settings (`settings.rs`), then the backend. A build without `native-mlx` on Apple
//! Silicon stops at the backend check with a clear error and never touches a file.

use super::settings::{FlashEnv, FlashOptions, FlashSettings};
use crate::{
    EngError,
    family::QWEN3_8_FLASH_NEXT,
    sched::worker::{Engine, ReadyNote},
    tokenizer::TokenizerBundle,
};
use std::path::Path;

/// The family name of a Flash-Next engine entry. The family descriptor owns it.
pub(crate) const FAMILY: &str = QWEN3_8_FLASH_NEXT.name;

/// The most tokens of prompt plus completion that the server accepts for this family
/// (`qsa::MAX_POSITIONS` of the resident decoder). `start.rs` compares it with
/// `FlashExecutor::context_limit` when the engine thread opens the executor, so a changed
/// decoder limit stops the load instead of misleading the 409 check.
pub(crate) const CONTEXT_LIMIT: u32 = 32_768;

/// A started engine, the tokenizer bundle that renders and encodes its prompts, and its limit.
pub(crate) struct StartedEngine {
    /// The engine thread. The caller stops it with `Engine::shutdown`.
    pub(crate) engine: Engine,
    /// The pinned tokenizer, template, and EOS ids.
    pub(crate) bundle: TokenizerBundle,
    /// The context limit that the server reports for this engine (`CONTEXT_LIMIT`).
    pub(crate) context_limit: u32,
}

/// Starts the engine of one config entry. Reads `FLASH_NEXT_ARTIFACT_MANIFEST` and
/// `FLASH_NEXT_CACHE_LIMIT` as fallbacks. The engine state is `Loading` until the weights are
/// resident.
pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlashOptions>,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let env = FlashEnv {
        manifest: std::env::var("FLASH_NEXT_ARTIFACT_MANIFEST").ok(),
        cache_limit: std::env::var("FLASH_NEXT_CACHE_LIMIT").ok(),
    };
    start_engine_with(model_dir, tokenizer_dir, options, &env, ready_note)
}

fn start_engine_with(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlashOptions>,
    env: &FlashEnv,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let settings = FlashSettings::new(model_dir, tokenizer_dir, options, env)?;
    let (engine, bundle) = launch(&settings, ready_note)?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit: CONTEXT_LIMIT,
    })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(
    settings: &FlashSettings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    super::start::start(settings, ready_note)
}

#[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
fn launch(
    _settings: &FlashSettings,
    _ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    Err(EngError::BadRequest(
        "engine config: this build has no Flash-Next backend; build with --features native-mlx on Apple Silicon"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_LIMIT, FAMILY, FlashEnv, start_engine_with};
    use crate::{EngError, family::flash_next::settings::FlashOptions};
    use std::path::Path;

    fn message<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn start(options: Option<&FlashOptions>, env: &FlashEnv) -> Result<(), EngError> {
        start_engine_with(Path::new("/a"), Path::new("/a"), options, env, None).map(|_| ())
    }

    #[test]
    fn the_family_name_and_the_context_limit_are_the_documented_values() {
        assert_eq!(FAMILY, "qwen3_8_flash_next");
        assert_eq!(CONTEXT_LIMIT, 32_768);
    }

    #[test]
    fn an_entry_with_a_missing_setting_is_refused_before_the_backend() {
        assert_eq!(
            message(start(None, &FlashEnv::default())),
            "engine config: manifest is missing and FLASH_NEXT_ARTIFACT_MANIFEST is not set"
        );
    }

    #[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn a_complete_entry_in_a_build_without_the_backend_names_the_feature() {
        use std::path::PathBuf;
        let options = FlashOptions {
            manifest: Some(PathBuf::from("/does/not/exist/download-manifest.json")),
            cache_limit_bytes: Some(1024),
            ..FlashOptions::default()
        };
        let text = message(start(Some(&options), &FlashEnv::default()));
        assert!(text.contains("native-mlx"), "{text}");
    }
}
