//! The engine factory for Q36 and dense Qwen3.8 entries of the `enginesrv` config
//! (decision 0046; packet Q36-F3b).
//!
//! It follows the interface of the Flash-Next factory (`flash_next/factory.rs`) and returns the
//! same `StartedEngine`. The server start-up code (`server/run.rs`) calls it for the entries of
//! this family with the process-wide `prefix_reuse` flag of the configuration file. Order of
//! checks: the settings (`settings.rs`), the `prefix_reuse` session-cap rule when the flag is on
//! and the arm is MLX (`Q36Settings::check_prefix_reuse_cap`), then the backend. A build with neither `native-mlx`
//! on Apple Silicon nor `native-cuda` on Linux stops at the backend check with a clear error
//! naming both features and never touches a file. Q36 results never satisfy Phase 0B or Phase
//! 1B.

use super::settings::{Backend, Q36Env, Q36Options, Q36Settings};
use crate::{
    EngError,
    family::{FamilySpec, QWEN3_6_35B_A3B, QWEN3_8_27B_DENSE, flash_next::factory::StartedEngine},
    sched::worker::{Engine, ReadyNote},
    tokenizer::TokenizerBundle,
};
use std::path::Path;

/// The family name of a Q36 engine entry. The family descriptor owns it.
pub(crate) const FAMILY: &str = QWEN3_6_35B_A3B.name;
/// The dense family shares the factory, but not its dimensions or quantization defaults.
pub(crate) const DENSE_FAMILY: &str = QWEN3_8_27B_DENSE.name;

/// The descriptor selected by the server configuration.
pub(crate) fn descriptor(family: &str) -> Option<FamilySpec> {
    match family {
        FAMILY => Some(QWEN3_6_35B_A3B),
        DENSE_FAMILY => Some(QWEN3_8_27B_DENSE),
        _ => None,
    }
}

/// Starts the engine of one config entry. Reads `Q36_CACHE_LIMIT` as a fallback. The engine state
/// is `Loading` until the weights are resident. `prefix_reuse` is the process-wide flag of the
/// configuration file: when it is on, the MLX arm needs a session cap that covers the entry and
/// the live session together (`Q36Settings::check_prefix_reuse_cap`). The CUDA arm keeps its
/// entry in host RAM and has no such rule.
pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    start_engine_for_family(
        FAMILY,
        model_dir,
        tokenizer_dir,
        options,
        prefix_reuse,
        ready_note,
    )
}

/// Starts a registered family with its selected descriptor.
pub(crate) fn start_engine_for_family(
    family: &str,
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let spec = descriptor(family)
        .ok_or_else(|| EngError::BadRequest(format!("unsupported Q36-style family {family}")))?;
    let env = Q36Env {
        cache_limit: std::env::var("Q36_CACHE_LIMIT").ok(),
    };
    start_engine_with(
        model_dir,
        tokenizer_dir,
        options,
        &env,
        prefix_reuse,
        spec,
        ready_note,
    )
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

/// The settings of an entry with the process-wide `prefix_reuse` flag applied (packet
/// SERVE-2-c item 1, fix M1): `Q36Settings::new` builds the checked settings and leaves the
/// flag false, because no config member or environment key carries the process-wide flag, so
/// the flag travels in the settings from here — the MLX `start` reads it into the
/// `prefix_store` limits of the rules. The builder is pure: the factory and its test share it.
fn settings_with_prefix_reuse(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    env: &Q36Env,
    backend: Backend,
    prefix_reuse: bool,
    spec: FamilySpec,
) -> Result<Q36Settings, EngError> {
    let mut settings =
        Q36Settings::new_for_family(model_dir, tokenizer_dir, options, env, backend, spec)?;
    settings.prefix_reuse = prefix_reuse;
    Ok(settings)
}

/// The Linux tests reach `settings_with_prefix_reuse` and the rules; the call from this
/// function is covered by the Mac served smoke (prefix-reuse row).
fn start_engine_with(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    env: &Q36Env,
    prefix_reuse: bool,
    spec: FamilySpec,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let settings = settings_with_prefix_reuse(
        model_dir,
        tokenizer_dir,
        options,
        env,
        backend_of_build(),
        prefix_reuse,
        spec,
    )?;
    settings.check_prefix_reuse_cap(prefix_reuse)?;
    let (engine, bundle) = launch(&settings, ready_note)?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit: settings.context_limit,
    })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    super::start::start(settings, ready_note)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn launch(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    super::cuda::start::start(settings, ready_note)
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
fn launch(
    settings: &Q36Settings,
    _ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    if settings.spec.feed_forward.is_some() {
        return Err(EngError::BadRequest(format!(
            "engine config: this build has no {} backend; build with --features native-mlx on Apple Silicon",
            settings.spec.name
        )));
    }
    Err(EngError::BadRequest(
        "engine config: this build has no Q36 backend; build with --features native-mlx on Apple Silicon or --features native-cuda on Linux"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{FAMILY, Q36Env, start_engine_with};
    use crate::EngError;
    use crate::family::qwen36::settings::Q36Options;
    use std::path::Path;

    fn message<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn start(options: Option<&Q36Options>, env: &Q36Env) -> Result<(), EngError> {
        start_engine_with(
            Path::new("/a"),
            Path::new("/a"),
            options,
            env,
            false,
            crate::family::QWEN3_6_35B_A3B,
            None,
        )
        .map(|_| ())
    }

    /// Starts an entry with the process-wide `prefix_reuse` flag set to `flag`.
    fn start_with_reuse(
        options: Option<&Q36Options>,
        env: &Q36Env,
        flag: bool,
    ) -> Result<(), EngError> {
        start_engine_with(
            Path::new("/a"),
            Path::new("/a"),
            options,
            env,
            flag,
            crate::family::QWEN3_6_35B_A3B,
            None,
        )
        .map(|_| ())
    }

    /// The text of a refused start, whatever the check was.
    fn error_text(result: Result<(), EngError>) -> String {
        match result {
            Err(EngError::ModelOutput(text)) => text,
            Err(EngError::BadRequest(text))
            | Err(EngError::Checkpoint(text))
            | Err(EngError::MemoryPlan(text))
            | Err(EngError::ContextLimit(text)) => text,
            Err(EngError::PromptTooLong {
                prompt_tokens,
                limit_tokens,
            }) => EngError::prompt_too_long_message(prompt_tokens, limit_tokens),
            Err(EngError::Backend { operation, message }) => format!("{operation}: {message}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    // The CUDA arm ignores `session_cap_bytes` (an MLX-only key), so the cap rule has no CUDA
    // premise (decision 0036, addendum 2, CUDA arm).
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_prefix_reuse_entry_below_the_required_session_cap_is_refused_with_the_numbers_and_the_hint()
     {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        // The default cap of 4 GiB covers one live session of the default 8-bit cache at the
        // default limit of 262,144 positions (2,917,990,400 bytes with the DeltaNet state), but
        // not the entry beside it (5,770,117,120 bytes together).
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), true));
        assert!(
            text.contains("at least 5770117120 bytes")
                && text.contains("context limit 262144")
                && text.contains("the cap is 4294967296 bytes")
                && text.contains("raise session_cap_bytes"),
            "{text}"
        );
    }

    // The CUDA arm ignores `session_cap_bytes` (an MLX-only key), so the cap rule has no CUDA
    // premise (decision 0036, addendum 2, CUDA arm).
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_prefix_reuse_entry_at_the_required_session_cap_passes_the_rule() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(5_770_117_120),
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

    /// The pure settings builder carries the flag for both values (fix M1): deleting the line
    /// that sets `settings.prefix_reuse` kept every other test green, so the flag is asserted
    /// here, on the builder the factory itself uses.
    ///
    /// Mutation check: removing `settings.prefix_reuse = prefix_reuse;` fails the `true` row.
    #[test]
    fn the_settings_builder_carries_the_prefix_reuse_flag_for_both_values() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        for flag in [false, true] {
            let settings = super::settings_with_prefix_reuse(
                Path::new("/a"),
                Path::new("/a"),
                Some(&options),
                &Q36Env::default(),
                super::backend_of_build(),
                flag,
                crate::family::QWEN3_6_35B_A3B,
            )
            .unwrap_or_else(|_| panic!("flag {flag}: the settings must build"));
            assert_eq!(settings.prefix_reuse, flag, "flag {flag}");
        }
    }

    #[test]
    fn the_family_name_is_the_documented_value() {
        assert_eq!(FAMILY, "qwen3_6_35b_a3b");
    }

    #[test]
    fn the_factory_selects_the_descriptor_named_by_the_config() {
        use crate::family::{QWEN3_6_35B_A3B, QWEN3_8_27B_DENSE};

        for (name, spec, cap) in [
            ("qwen3_6_35b_a3b", QWEN3_6_35B_A3B, 4 << 30),
            ("qwen3_8_27b_dense", QWEN3_8_27B_DENSE, 9_283_698_688),
        ] {
            let options = Q36Options {
                cache_limit_bytes: Some(1024),
                session_cap_bytes: Some(cap),
                ..Q36Options::default()
            };
            let selected = super::descriptor(name).expect("known family");
            let settings = super::settings_with_prefix_reuse(
                Path::new("/a"),
                Path::new("/a"),
                Some(&options),
                &Q36Env::default(),
                super::Backend::Mlx,
                false,
                selected,
            )
            .unwrap();
            assert_eq!(settings.spec, spec);
            assert_eq!(settings.context_limit, 262_144);
            assert_eq!(settings.session_cap, cap);
        }
        assert!(super::descriptor("unknown").is_none());
    }

    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    #[test]
    fn a_dense_entry_without_a_backend_names_only_the_supported_backend() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(9_283_698_688),
            ..Q36Options::default()
        };
        let text = message(start_engine_with(
            Path::new("/nonexistent"),
            Path::new("/nonexistent"),
            Some(&options),
            &Q36Env::default(),
            false,
            crate::family::QWEN3_8_27B_DENSE,
            None,
        ));
        assert_eq!(
            text,
            "engine config: this build has no qwen3_8_27b_dense backend; build with --features native-mlx on Apple Silicon"
        );
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
            "engine config: prefill_chunk must be between 1 and 2048, found 0"
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
