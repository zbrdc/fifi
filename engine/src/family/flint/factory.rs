//! Flint startup. Native values are constructed only on the engine thread.

use super::settings::FlintOptions;
use std::path::Path;

use crate::{
    EngError,
    sched::worker::{Engine, ReadyNote},
    tokenizer::TokenizerBundle,
};

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
/// Mac MLX serving context limit. Default 32768 since 2026-10-27 (full-attention
/// KV planes grow on demand and reach +0.656 GiB BF16 over 8192 only at the
/// full 32K depth: 14 full layers keep the whole prefix, the
/// ring layers clamp to WINDOW=513). Rollback: FIFI_FLINT_MLX_CONTEXT_LIMIT=8192
/// (read once per process at first use; changes require a restart);
/// an unrecognized SET value fails safe to 8192, not 32768.
pub(crate) fn context_limit() -> u32 {
    use std::sync::OnceLock;
    static LIMIT: OnceLock<u32> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        context_limit_with(os_setting(
            std::env::var_os("FIFI_FLINT_MLX_CONTEXT_LIMIT").as_deref(),
        ))
    })
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(super) fn context_limit_with(setting: Option<&str>) -> u32 {
    match setting.map(str::trim) {
        None => 32768,
        Some("32768") => 32768,
        Some("8192") => 8192,
        // Fail CLOSED to the old default: a mistyped limit must not silently
        // grant 32K capacity (memory) the operator did not ask for.
        Some(_) => 8192,
    }
}

/// One env-setting reader for the piece-size knob: SET-but-non-UTF-8 maps to
/// "" (an unrecognized value), so the fail-safe below applies uniformly.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(super) fn os_setting(value: Option<&std::ffi::OsStr>) -> Option<&str> {
    value.map(|v| v.to_str().unwrap_or(""))
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(super) fn prefill_chunk_rows(setting: Option<&str>) -> usize {
    // w37/w38 evidence: chunk 1024 is the ratified default (+9.31% pp4096,
    // KL <=1.3e-4 vs 512 (log-sum-exp, 2026-10-27 probe fix) with top-1
    // equal, +0.14 GiB peak; w39 ratchet re-baseline per the ratified bars).
    // Rollback = FIFI_FLINT_MLX_BATCH_PREFILL=512. An explicitly SET but
    // unrecognized value (non-UTF-8, empty, mistyped) fails safe to the old
    // 512 default.
    match setting.map(str::trim) {
        None => 1024,
        Some("0") => 1,
        Some("512") => 512,
        Some("1024") => 1024,
        Some(_) => 512,
    }
}

pub(crate) struct StartedEngine {
    pub(crate) engine: Engine,
    pub(crate) bundle: TokenizerBundle,
    pub(crate) context_limit: u32,
}

/// Unsupported builds refuse before sampling defaults, tokenizer, or weight reads.
pub(crate) fn check_backend() -> Result<(), EngError> {
    if cfg!(any(
        all(
            feature = "native-mlx",
            target_os = "macos",
            target_arch = "aarch64"
        ),
        all(feature = "native-cuda", target_os = "linux")
    )) {
        Ok(())
    } else {
        Err(EngError::BadRequest(
            "engine config: Flint requires --features native-mlx on Apple Silicon or --features native-cuda on Linux".into(),
        ))
    }
}

pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlintOptions>,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    check_backend()?;
    launch(model_dir, tokenizer_dir, options, ready_note)
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlintOptions>,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    use crate::sched::worker::spawn_engine_noted;

    if options.is_some() {
        return Err(EngError::BadRequest(
            "flint options select the CUDA pilot, not MLX".into(),
        ));
    }
    let bundle = load_tokenizer(tokenizer_dir)?;
    let eos = bundle.eos_token_ids().to_vec();
    let vocab = bundle.vocab_size() as usize;
    let root = model_dir.to_owned();
    let engine = spawn_engine_noted(1, ready_note, move || {
        super::mlx_executor::FlintExecutor::open(&root, eos, vocab)
    })?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit: context_limit(),
    })
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn launch(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlintOptions>,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    use crate::sched::worker::spawn_engine_noted;

    let options = options
        .ok_or_else(|| {
            EngError::BadRequest(
                "Flint CUDA requires flint context_limit and explicit kv_profile f16 or q8".into(),
            )
        })?
        .clone();
    options.plan()?;
    let context_limit = options.context_limit;
    let bundle = load_tokenizer(tokenizer_dir)?;
    let eos = bundle.eos_token_ids().to_vec();
    let vocab = bundle.vocab_size() as usize;
    let root = model_dir.to_owned();
    let engine = spawn_engine_noted(1, ready_note, move || {
        super::cuda_executor::FlintExecutor::open(&root, &options, eos, vocab)
    })?;
    Ok(StartedEngine {
        engine,
        bundle,
        context_limit,
    })
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
fn launch(
    _model_dir: &Path,
    _tokenizer_dir: &Path,
    _options: Option<&FlintOptions>,
    _ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    Err(EngError::BadRequest(
        "engine config: Flint requires --features native-mlx on Apple Silicon or --features native-cuda on Linux".into(),
    ))
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
))]
pub(super) fn load_tokenizer(dir: &Path) -> Result<TokenizerBundle, EngError> {
    use crate::tokenizer::TokenizerPin;
    // RecursiveSI/flint-int4 revision 3b93f905, publisher manifest flint-int4.json.
    TokenizerBundle::load(
        dir,
        &TokenizerPin {
            tokenizer_json: digest(
                b"a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68",
            ),
            tokenizer_config_json: digest(
                b"159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf",
            ),
            chat_template_jinja: digest(
                b"7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466",
            ),
            generation_config_json: digest(
                b"a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3",
            ),
            vocab_size: 150_144,
        },
    )
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
))]
const fn digest(hex: &[u8; 64]) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("invalid pinned digest"),
        }
    }
    let mut bytes = [0; 32];
    let mut i = 0;
    while i < 32 {
        bytes[i] = nibble(hex[2 * i]) * 16 + nibble(hex[2 * i + 1]);
        i += 1;
    }
    bytes
}

#[cfg(test)]
mod tests {
    #[test]
    fn context_limit_env_fail_safe_is_8192() {
        assert_eq!(super::context_limit_with(None), 32768);
        assert_eq!(super::context_limit_with(Some("32768")), 32768);
        assert_eq!(super::context_limit_with(Some(" 8192 ")), 8192);
        assert_eq!(super::context_limit_with(Some("16384")), 8192);
        assert_eq!(super::context_limit_with(Some("invalid")), 8192);
        assert_eq!(super::context_limit_with(Some("")), 8192);
    }

    #[test]
    fn context_limit_default_is_32k() {
        assert_eq!(super::context_limit_with(None), 32768);
        assert_eq!(super::context_limit_with(Some("8192")), 8192);
        assert_eq!(super::context_limit_with(Some("32768")), 32768);
        // Fail-safe: an unrecognized SET value rolls back to 8192 (SF-2).
        assert_eq!(super::context_limit_with(Some("invalid")), 8192);
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn cuda_build_accepts_flint_backend_before_checkpoint_io() {
        assert!(
            super::check_backend().is_ok(),
            "native-cuda/Linux must admit the Flint factory"
        );
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn cuda_pilot_requires_explicit_profile_before_checkpoint_io() {
        let result = super::start_engine(
            std::path::Path::new("missing-flint-model"),
            std::path::Path::new("missing-flint-tokenizer"),
            None,
            None,
        );
        assert!(matches!(result, Err(crate::EngError::BadRequest(message))
            if message.contains("requires flint context_limit and explicit kv_profile f16 or q8")));
    }

    #[test]
    fn prefill_chunk_default_1024_with_fail_safe_rollback() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(super::prefill_chunk_rows(None), 1024);
        assert_eq!(super::prefill_chunk_rows(Some("1024")), 1024);
        assert_eq!(super::prefill_chunk_rows(Some(" 1024 ")), 1024);
        assert_eq!(super::prefill_chunk_rows(Some("512")), 512);
        assert_eq!(super::prefill_chunk_rows(Some("0")), 1);
        assert_eq!(super::prefill_chunk_rows(Some("0\n")), 1);
        // SET but unrecognized (empty, mistyped, non-UTF-8): fail safe to
        // the rollback size 512, never the 1024 default.
        assert_eq!(super::prefill_chunk_rows(Some("")), 512);
        assert_eq!(super::prefill_chunk_rows(Some("2048")), 512);
        assert_eq!(super::prefill_chunk_rows(Some("invalid")), 512);
        // SET but non-UTF-8: "" is unrecognized => fail-safe 512.
        let invalid = OsStr::from_bytes(b"\xff512");
        assert_eq!(super::os_setting(Some(invalid)), Some(""));
        assert_eq!(
            super::prefill_chunk_rows(super::os_setting(Some(invalid))),
            512
        );
    }

    #[test]
    #[ignore = "requires FLINT_MODEL_DIR with the pinned publisher tokenizer"]
    fn factory_uses_the_complete_pinned_tokenizer_and_flint_grammar() {
        let dir = crate::tokenizer::flint_tests::fixture_dir();
        let bundle = super::load_tokenizer(&dir).unwrap();
        assert_eq!(bundle.vocab_size(), 150_144);
        assert_eq!(bundle.eos_token_ids(), [150001, 150003, 150005, 150018]);
        assert_eq!(
            bundle.tool_grammar(),
            crate::tokenizer::toolcall::ToolGrammar::Flint
        );
    }
}
