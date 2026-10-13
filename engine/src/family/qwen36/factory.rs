//! The engine factory for `family: "qwen3_6_35b_a3b"` entries of the `enginesrv` config
//! (decision 0046; packet Q36-F3b).
//!
//! It follows the interface of the Flash-Next factory (`flash_next/factory.rs`) and returns the
//! same `StartedEngine`. The server start-up code (`server/run.rs`) calls it for the entries of
//! this family with the process-wide `prefix_reuse` flag of the configuration file. Order of
//! checks: the settings (`settings.rs`), the `prefix_reuse` session-cap rule when the flag is on
//! (`Q36Settings::check_prefix_reuse_cap`), then the backend. A build with neither `native-mlx`
//! on Apple Silicon nor `native-cuda` on Linux stops at the backend check with a clear error
//! naming both features and never touches a file. Q36 results never satisfy Phase 0B or Phase
//! 1B.

use super::settings::{Backend, Q36Env, Q36Options, Q36Settings};
use crate::{
    EngError,
    family::{QWEN3_6_35B_A3B, flash_next::factory::StartedEngine},
    sched::worker::Engine,
    tokenizer::TokenizerBundle,
};
use std::path::Path;

/// The family name of a Q36 engine entry. The family descriptor owns it.
pub(crate) const FAMILY: &str = QWEN3_6_35B_A3B.name;

/// Starts the engine of one config entry. Reads `Q36_CACHE_LIMIT` as a fallback. The engine state
/// is `Loading` until the weights are resident. `prefix_reuse` is the process-wide flag of the
/// configuration file: when it is on, the session cap must cover the entry and the live session
/// together (`Q36Settings::check_prefix_reuse_cap`).
pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    prefix_reuse: bool,
) -> Result<StartedEngine, EngError> {
    let env = Q36Env {
        cache_limit: std::env::var("Q36_CACHE_LIMIT").ok(),
    };
    start_engine_with(model_dir, tokenizer_dir, options, &env, prefix_reuse)
}

/// Whether this build's Q36 arm supports `prefix_reuse`: the MLX arm does; the CUDA arm gets
/// it in CUDA-SERVE-2, so the server refuses a Q36 entry with the flag on there. A build with
/// NO Q36 backend answers `true` on purpose (KUBDAN-MSG-71 item 3): its entries cannot start
/// at all, and the factory's own refusal ("this build has no Q36 backend") names the real
/// fault, so the server's CUDA-specific text must not shadow it.
pub(crate) fn prefix_reuse_supported() -> bool {
    !cfg!(all(feature = "native-cuda", target_os = "linux"))
}

/// The backend the settings read; a build with neither backend keeps the MLX reading.
fn backend_of_build() -> Backend {
    if cfg!(all(
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    )) {
        Backend::Mlx
    } else if cfg!(all(feature = "native-cuda", target_os = "linux")) {
        Backend::Cuda
    } else {
        Backend::Mlx
    }
}

fn start_engine_with(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    env: &Q36Env,
    prefix_reuse: bool,
) -> Result<StartedEngine, EngError> {
    let settings = Q36Settings::new(model_dir, tokenizer_dir, options, env, backend_of_build())?;
    settings.check_prefix_reuse_cap(prefix_reuse)?;
    let (engine, bundle) = launch(&settings)?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit: settings.context_limit,
    })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(settings: &Q36Settings) -> Result<(Engine, TokenizerBundle), EngError> {
    super::start::start(settings)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn launch(settings: &Q36Settings) -> Result<(Engine, TokenizerBundle), EngError> {
    super::cuda::start::start(settings)
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
fn launch(_settings: &Q36Settings) -> Result<(Engine, TokenizerBundle), EngError> {
    Err(EngError::BadRequest(
        "engine config: this build has no Q36 backend; build with --features native-mlx on Apple Silicon or --features native-cuda on Linux"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{FAMILY, Q36Env, start_engine_with};
    use crate::{EngError, family::qwen36::settings::Q36Options};
    use std::path::Path;

    fn message<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn start(options: Option<&Q36Options>, env: &Q36Env) -> Result<(), EngError> {
        start_engine_with(Path::new("/a"), Path::new("/a"), options, env, false).map(|_| ())
    }

    /// Starts an entry with the process-wide `prefix_reuse` flag set to `flag`.
    fn start_with_reuse(
        options: Option<&Q36Options>,
        env: &Q36Env,
        flag: bool,
    ) -> Result<(), EngError> {
        start_engine_with(Path::new("/a"), Path::new("/a"), options, env, flag).map(|_| ())
    }

    /// The text of a refused start, whatever the check was.
    fn error_text(result: Result<(), EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text))
            | Err(EngError::Checkpoint(text))
            | Err(EngError::MemoryPlan(text))
            | Err(EngError::ContextLimit(text)) => text,
            Err(EngError::Backend { operation, message }) => format!("{operation}: {message}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    // The CUDA arm ignores `session_cap_bytes` (an MLX-only key) and refuses `prefix_reuse` in the
    // server pre-pass, so the cap rule has no CUDA premise.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_prefix_reuse_entry_below_the_required_session_cap_is_refused_with_the_numbers_and_the_hint()
     {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        // The default cap of 2 GiB covers one live session at the default limit (1,342,177,280
        // bytes), but not the entry beside it (2,750,218,240 bytes together).
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), true));
        assert!(
            text.contains("at least 2750218240 bytes")
                && text.contains("context limit 32768")
                && text.contains("the cap is 2147483648 bytes")
                && text.contains("raise session_cap_bytes"),
            "{text}"
        );
    }

    // The CUDA arm ignores `session_cap_bytes` (an MLX-only key) and refuses `prefix_reuse` in the
    // server pre-pass, so the cap rule has no CUDA premise.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_prefix_reuse_entry_at_the_required_session_cap_passes_the_rule() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(2_750_218_240),
            ..Q36Options::default()
        };
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), true));
        assert!(!text.contains("raise session_cap_bytes"), "{text}");
    }

    #[test]
    fn the_session_cap_rule_is_off_without_the_prefix_reuse_flag() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(1),
            ..Q36Options::default()
        };
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), false));
        assert!(!text.contains("raise session_cap_bytes"), "{text}");
    }

    #[test]
    fn the_family_name_is_the_documented_value() {
        assert_eq!(FAMILY, "qwen3_6_35b_a3b");
    }

    // The three tests below pin the MLX reading and the fallback refusal of the builds
    // WITHOUT the CUDA arm; on a `native-cuda` Linux build the arm exists and the entry
    // reaches the CUDA start, so the pinned premises are false there by design.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn an_entry_with_a_missing_setting_is_refused_before_the_backend() {
        assert_eq!(
            message(start(None, &Q36Env::default())),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
    }

    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_bad_member_is_refused_by_name_before_the_backend() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            prefill_chunk: Some(0),
            ..Q36Options::default()
        };
        assert_eq!(
            message(start(Some(&options), &Q36Env::default())),
            "engine config: prefill_chunk must be between 1 and 512, found 0"
        );
    }

    #[test]
    fn a_bad_activation_dtype_is_refused_by_name_before_the_backend() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            activation_dtype: Some("f16".to_string()),
            ..Q36Options::default()
        };
        assert_eq!(
            message(start(Some(&options), &Q36Env::default())),
            "engine config: activation_dtype must be f32 or bf16, got \"f16\""
        );
    }

    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    #[test]
    fn a_complete_entry_in_a_build_without_the_backend_names_both_features() {
        // The FULL fallback text, by equality (KUBDAN-MSG-71 item 2): an include_str! pin of
        // this file's own literals cannot fail, so the no-backend build — where the text is
        // the error — pins it outright. The old single-feature spelling fails here.
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        assert_eq!(
            message(start(Some(&options), &Q36Env::default())),
            "engine config: this build has no Q36 backend; build with --features native-mlx \
             on Apple Silicon or --features native-cuda on Linux"
        );
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn the_cuda_arm_is_selected_and_the_fallback_names_both_features() {
        // A complete CUDA entry needs no cache limit: an EMPTY qwen36 object leaves the
        // settings checks and fails in the CUDA start's tokenizer load, never at a refusal.
        let text = match start(None, &Q36Env::default()) {
            Err(EngError::Checkpoint(text)) => text,
            other => panic!("expected the CUDA start's tokenizer refusal, got {other:?}"),
        };
        assert!(text.contains("tokenizer.json"), "{text}");
        assert!(text.contains("is missing"), "{text}");
        assert!(!text.contains("no Q36 backend"), "{text}");
        // The fallback arm is not compiled in this build: its full text is pinned by equality
        // in `a_complete_entry_in_a_build_without_the_backend_names_both_features`, which runs
        // on the no-backend build where the text IS the error (KUBDAN-MSG-71 item 2 — an
        // include_str! pin of this file's own literals cannot fail).
    }
}
