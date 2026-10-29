//! Startup and shutdown of `enginesrv`.
//!
//! Order: read the command line, the configuration, and the key; bind the listener; start the
//! engines; serve. `SIGTERM` or `SIGINT` stops the listener, lets the running requests finish,
//! stops the engine threads, and returns. The caller then exits with code 0.

use std::io::Write as _;

use super::config::{Config, EngineConfig, StartupError, config_path};
use super::fake;
use super::router::router;
use super::state::{AppState, EngineEntry};
use super::wire::state_name;
use crate::EngError;
use crate::family::flash_next::factory;
use crate::family::flint;
use crate::family::nemotron35::factory as nemotron_factory;
use crate::family::qwen36::factory as q36_factory;
use crate::family::sampler::SamplingDefaults;
use crate::sched::worker::{Engine, EngineNote, ReadyNote, ShutdownReport};
use crate::tokenizer::TokenizerBundle;
use axum::serve::ListenerExt;
use std::future::Future;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Runs the server. `args` are the command-line arguments after the program name.
/// The call returns after a shutdown signal and the end of the running requests.
pub fn run_cli(args: &[String]) -> Result<(), StartupError> {
    let path = config_path(args)?;
    let config = Config::from_file(&path)?;
    let key = config.api_key(|name| std::env::var(name).ok())?;
    let address = config.listen_address()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| StartupError(format!("cannot start the runtime: {error}")))?;
    let listener = runtime
        .block_on(TcpListener::bind(address))
        .map_err(|error| StartupError(format!("cannot listen on {address}: {error}")))?;
    let bound = listener
        .local_addr()
        .map_err(|error| StartupError(format!("cannot read the listen address: {error}")))?;
    // The handlers exist before the banner, so a signal after the banner always drains the server.
    let shutdown = {
        let _inside_runtime = runtime.enter();
        shutdown_signal()
    };
    let (engines, entries) = start_engines(&config)?;
    let state = Arc::new(AppState::new(&key, entries));
    // One boot line for the warm-turn flip: the effective state and how to
    // roll it back. The polarity comes from the SAME function the request
    // path uses, so the banner can never disagree with the requests.
    let warm_on = crate::server::chat::warm_flag_on();
    let warm_state = if warm_on { "ON" } else { "OFF" };
    match std::env::var("ENGINESRV_WARM_TURNS") {
        Ok(value) => eprintln!("warm turns: {warm_state} (ENGINESRV_WARM_TURNS={value})"),
        Err(std::env::VarError::NotPresent) => {
            eprintln!("warm turns: {warm_state} (default; ENGINESRV_WARM_TURNS=0 disables)");
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!("warm turns: {warm_state} (ENGINESRV_WARM_TURNS is not valid UTF-8)");
        }
    }
    println!("enginesrv listening on {bound}");
    let served = runtime.block_on(serve(listener, state, shutdown));
    let mut failures = Vec::new();
    for (config, engine) in config.engines.iter().zip(engines) {
        failures.extend(shutdown_failure(&config.id, &engine.shutdown()));
    }
    outcome(served, failures)
}

/// The result of a run: an error of the server comes first, then the failures that the engines
/// reported at shutdown. A run without either succeeds.
fn outcome(served: Result<(), StartupError>, failures: Vec<String>) -> Result<(), StartupError> {
    served?;
    if failures.is_empty() {
        Ok(())
    } else {
        Err(StartupError(failures.join("; ")))
    }
}

/// Describes the failure that an engine reported at shutdown, such as a panic or a failed load.
/// An engine without a failure gives `None`.
fn shutdown_failure(id: &str, report: &ShutdownReport) -> Option<String> {
    report.failure.as_ref().map(|cause| {
        format!(
            "engine {id} ended in state {}: {cause}",
            state_name(report.state)
        )
    })
}

/// Starts every engine of the configuration. A failure names the engine. Every entry carries the
/// `prefix_reuse` flag of the configuration file to the chat handler and to the Q36 engine
/// start, whose session-cap rule reads it (`family::qwen36::settings::check_prefix_reuse_cap`),
/// and the sampling defaults that its `generation_config.json` sets together with the flag of
/// whether that file supplied them (the start line quotes both).
/// Applies the per-entry replay opt-outs to a freshly built bundle and
/// reports the effective state on stderr (the template's own load line
/// shows the default resolution, before these overrides). Called while the
/// bundle is still uniquely held: an opt-out that cannot be applied is a
/// startup error, never a silent skip.
fn apply_replay_overrides(
    id: &str,
    bundle: &mut Arc<TokenizerBundle>,
    config: &EngineConfig,
) -> Result<(), StartupError> {
    if config.think_replay != Some(false) && config.notice_toolwrap != Some(false) {
        return Ok(());
    }
    match Arc::get_mut(bundle) {
        Some(bundle) => {
            if config.think_replay == Some(false) {
                bundle.disable_replay();
                if !cfg!(test) {
                    eprintln!("engine {id}: think replay off (config)");
                }
            } else {
                bundle.disable_notice_wrap();
                if !cfg!(test) {
                    eprintln!("engine {id}: notice wrap off (config)");
                }
            }
            Ok(())
        }
        None => Err(StartupError(format!(
            "engine {id} failed to start: the replay opt-out could not be applied (the bundle is already shared)"
        ))),
    }
}

/// Starts every engine of the configuration. A failure names the engine. Every entry carries the
/// `prefix_reuse` flag of the configuration file to the chat handler and to the Q36 engine
/// start, whose session-cap rule reads it (`family::qwen36::settings::check_prefix_reuse_cap`),
/// and the sampling defaults that its `generation_config.json` sets together with the flag of
/// whether that file supplied them (the start line quotes both).
fn start_engines(file: &Config) -> Result<(Vec<Engine>, Vec<EngineEntry>), StartupError> {
    let mut engines = Vec::new();
    let mut entries = Vec::new();
    for config in &file.engines {
        check_staged_family(&config.family).map_err(|error| {
            StartupError(format!(
                "engine {} failed to start: {}",
                config.id,
                describe(&error)
            ))
        })?;
        // The registry lookup sits next to the staged check and before any
        // file read: an unstartable family fails fast without touching the
        // model directory.
        let starter = family_starter(&config.family).ok_or_else(|| {
            StartupError(format!(
                "engine {} failed to start: family {} has no starter in this build",
                config.id, config.family
            ))
        })?;
        let loaded = SamplingDefaults::from_model_dir(&config.model_dir).map_err(|error| {
            StartupError(format!(
                "engine {} failed to start: {}",
                config.id,
                describe(&error)
            ))
        })?;
        // The block prefix-match state is process-wide (the env parse) but only
        // matters for engines whose store can cut a rolling Block entry. The
        // engine thread fires this note with the BUILT executor's own
        // capability - cuts and spacing both - when the engine becomes
        // ready, so the line reports the real store - never a factory flag
        // or the process-wide parse - and a wrong default or a stray
        // rollback stays visible before the engine serves its first request.
        let ready_note: ReadyNote = if cfg!(test) {
            None
        } else {
            block_match_note(
                &config.id,
                file.prefix_reuse,
                // A broken stderr pipe must not turn a Ready engine into a
                // failed one over a log line.
                |line| {
                    let _ = writeln!(std::io::stderr().lock(), "{line}");
                },
            )
        };
        let StartedEngine {
            engine,
            mut bundle,
            context_limit,
        } = starter(config, file.prefix_reuse, ready_note).map_err(|error| {
            StartupError(format!(
                "engine {} failed to start: {}",
                config.id,
                describe(&error)
            ))
        })?;
        // Per-entry opt-outs, applied while the bundle is still uniquely
        // held. The env rollback (ENGINESRV_THINK_REPLAY=0) resolved inside
        // the template already wins over anything set here.
        apply_replay_overrides(&config.id, &mut bundle, config)?;
        let generation = config.model_dir.join("generation_config.json");
        // The label comes from the loader result, not from a second `exists` call, so the file
        // cannot change between the read and the log line.
        eprintln!(
            "{}",
            defaults_line(&config.id, &generation, &loaded.values, loaded.from_file)
        );
        // Flint snapshots are not qualified; the config flag alone never
        // arms them. The env opt-in lifts the family exclusion for the
        // qualification A/B so both arms run the same ELF. The entry flag
        // resolves against the engine's kv profile
        // (a q8 engine logs and arms OFF, not "on").
        let flint_opt_in = config.family == flint::FAMILY
            && flint::prefix_reuse_entry_enabled(
                config
                    .flint
                    .as_ref()
                    .map(|options| options.kv_profile.as_str()),
            );
        if config.family == flint::FAMILY {
            eprintln!(
                "flint prefix reuse: {}",
                if file.prefix_reuse && flint_opt_in {
                    "on"
                } else {
                    "off"
                }
            );
        }
        entries.push(EngineEntry {
            id: config.id.clone(),
            model: config.model_name(),
            handle: engine.handle(),
            bundle,
            context_limit,
            prefix_reuse: entry_prefix_reuse(&config.family, file.prefix_reuse, flint_opt_in),
            warm: std::sync::Arc::new(std::sync::Mutex::new(None)),
            defaults: loaded.values,
        });
        engines.push(engine);
    }
    Ok((engines, entries))
}

/// The start line of one engine: the effective sampling defaults and the file they came from.
/// The line quotes the path of `generation_config.json` and the values in force, so a wrong
/// default is visible before the first request. `from_file` comes from the loader result, so
/// the label always agrees with the read.
fn defaults_line(
    id: &str,
    file: &std::path::Path,
    defaults: &SamplingDefaults,
    from_file: bool,
) -> String {
    let values = format!(
        "temperature {}, top_p {}, top_k {}",
        defaults.temperature, defaults.top_p, defaults.top_k
    );
    if from_file {
        format!(
            "engine {id}: sampling defaults {values} from {}",
            file.display()
        )
    } else {
        format!(
            "engine {id}: sampling defaults {values} ({} absent; contract defaults)",
            file.display()
        )
    }
}

fn check_staged_family(family: &str) -> Result<(), EngError> {
    if family == flint::FAMILY {
        flint::factory::check_backend()?;
    }
    Ok(())
}

/// One family's start result: the uniform shape every factory already
/// produces de facto (engine handle, tokenizer bundle, context limit).
pub(crate) struct StartedEngine {
    pub(crate) engine: Engine,
    pub(crate) bundle: Arc<TokenizerBundle>,
    pub(crate) context_limit: u32,
}

/// The uniform starter signature: one config entry, the process-wide
/// prefix_reuse flag, the ready note. A future family (Gemma4) is one new
/// table entry with this same signature and no other registry edit;
/// adding its name to the config's FAMILIES list is config surface, not
/// registry.
pub(crate) type Starter = fn(&EngineConfig, bool, ReadyNote) -> Result<StartedEngine, EngError>;

fn start_flint(
    config: &EngineConfig,
    _prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let started = flint::factory::start_engine(
        &config.model_dir,
        &config.tokenizer_dir,
        config.flint.as_ref(),
        ready_note,
    )?;
    Ok(StartedEngine {
        engine: started.engine,
        bundle: Arc::new(started.bundle),
        context_limit: started.context_limit,
    })
}

fn start_flash(
    config: &EngineConfig,
    _prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let started = factory::start_engine(
        &config.model_dir,
        &config.tokenizer_dir,
        config.flash_next.as_ref(),
        ready_note,
    )?;
    Ok(StartedEngine {
        engine: started.engine,
        bundle: Arc::new(started.bundle),
        context_limit: started.context_limit,
    })
}

fn start_q36(
    config: &EngineConfig,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let started = q36_factory::start_engine_for_family(
        &config.family,
        &config.model_dir,
        &config.tokenizer_dir,
        config.qwen36.as_ref(),
        prefix_reuse,
        ready_note,
    )?;
    Ok(StartedEngine {
        engine: started.engine,
        bundle: Arc::new(started.bundle),
        context_limit: started.context_limit,
    })
}

fn start_nemotron(
    config: &EngineConfig,
    _prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let started = nemotron_factory::start_engine(
        &config.model_dir,
        &config.tokenizer_dir,
        config.nemotron.as_ref(),
        ready_note,
    )?;
    Ok(StartedEngine {
        engine: started.engine,
        bundle: Arc::new(started.bundle),
        context_limit: started.context_limit,
    })
}

fn start_fake(
    config: &EngineConfig,
    _prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let started = fake::start(config, ready_note)?;
    Ok(StartedEngine {
        engine: started.engine,
        bundle: started.bundle,
        context_limit: started.context_limit,
    })
}

/// The closed registry of families this build starts. Staged backends
/// refuse inside their starter before model or tokenizer reads; the
/// config gate owns the unknown-family rejection text.
const FAMILY_STARTERS: &[(&str, Starter)] = &[
    (flint::FAMILY, start_flint),
    (factory::FAMILY, start_flash),
    (q36_factory::FAMILY, start_q36),
    (q36_factory::DENSE_FAMILY, start_q36),
    (nemotron_factory::FAMILY, start_nemotron),
    (fake::FAMILY, start_fake),
];

/// The per-entry prefix-reuse decision, pure in its inputs so the family
/// exclusion is testable without an engine: other families follow the
/// config flag; Flint additionally needs the env opt-in
/// (FIFI_FLINT_PREFIX_REUSE, read once at load) because its snapshots are
/// not qualified yet.
fn entry_prefix_reuse(family: &str, file_flag: bool, flint_opt_in: bool) -> bool {
    file_flag && (family != flint::FAMILY || flint_opt_in)
}

/// The registry's family names, for the uniqueness pin in tests.
#[cfg(test)]
pub(crate) fn family_names() -> Vec<&'static str> {
    FAMILY_STARTERS.iter().map(|(name, _)| *name).collect()
}

pub(crate) fn family_starter(family: &str) -> Option<Starter> {
    FAMILY_STARTERS
        .iter()
        .find(|(name, _)| *name == family)
        .map(|(_, starter)| *starter)
}

/// The startup line for the block prefix-match. The helper only formats:
/// `None` when the config opted out of `prefix_reuse` or when the engine's
/// ready note reported that the built executor's store cannot cut a rolling
/// Block entry (the worker plans intermediate save points only for a store
/// with at least two LRU entries; today that is the MLX Q36-style start with
/// `prefix_reuse` - Qwen3.6 and dense Qwen3.8). Test suppression is the
/// caller's job: the startup site passes no note under `cfg!(test)`. A
/// rolled-back spacing prints the off line so a stray rollback stays
/// visible.
fn block_match_line(
    id: &str,
    prefix_reuse: bool,
    cuts_block: bool,
    spacing: Option<usize>,
) -> Option<String> {
    if !prefix_reuse || !cuts_block {
        return None;
    }
    Some(match spacing {
        Some(spacing) => format!("engine {id}: block prefix-match on (spacing {spacing} tokens)"),
        None => format!("engine {id}: block prefix-match off (rollback)"),
    })
}

/// The one-shot engine note that prints to `sink`: on [`EngineNote::Ready`]
/// it prints [`block_match_line`] - the cuts flag AND the executor's own
/// spacing, so the line can never disagree with the executor that will do
/// the cutting - and on [`EngineNote::Failed`] it prints the failure line
/// `/ready`'s log builds (`engine {id} failed: {cause}`), so a load that
/// fails before anyone polls the server is no longer silent. The sink is a
/// parameter so a test can capture the line (the production sink is
/// stderr, where a broken pipe must not fail the engine).
fn block_match_note<S>(id: &str, prefix_reuse: bool, sink: S) -> ReadyNote
where
    S: FnOnce(&str) + Send + 'static,
{
    let id = id.to_string();
    Some(Box::new(move |note| match note {
        EngineNote::Ready(capability) => {
            if let Some(line) =
                block_match_line(&id, prefix_reuse, capability.cuts_block, capability.spacing)
            {
                sink(&line);
            }
        }
        EngineNote::Failed(cause) => sink(&format!("engine {id} failed: {cause}")),
    }))
}

fn describe(error: &EngError) -> String {
    match error {
        EngError::ModelOutput(message) => message.clone(),
        EngError::Checkpoint(message)
        | EngError::BadRequest(message)
        | EngError::MemoryPlan(message)
        | EngError::ContextLimit(message) => message.clone(),
        EngError::PromptTooLong {
            prompt_tokens,
            limit_tokens,
        } => EngError::prompt_too_long_message(*prompt_tokens, *limit_tokens),
        EngError::Backend { operation, message } => format!("{operation}: {message}"),
    }
}

/// Serves requests until `shutdown` completes. Then it stops the listener and waits for the
/// running requests.
pub(crate) async fn serve(
    listener: TcpListener,
    state: Arc<AppState>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), StartupError> {
    let listener = listener.tap_io(|stream| {
        if let Err(error) = stream.set_nodelay(true) {
            eprintln!("enginesrv: cannot set TCP_NODELAY on an accepted connection: {error}");
        }
    });
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|error| StartupError(format!("server failure: {error}")))
}

/// Installs the handlers for `SIGTERM` and `SIGINT` at once and returns the future that completes
/// when one of them arrives. The call needs a running Tokio runtime. A signal that arrives before
/// the first poll is not lost. When the process cannot install the handlers, the future waits for
/// the interrupt signal alone.
fn shutdown_signal() -> impl Future<Output = ()> + Send + 'static {
    #[cfg(unix)]
    let streams = {
        use tokio::signal::unix::{SignalKind, signal};
        signal(SignalKind::terminate())
            .and_then(|terminate| signal(SignalKind::interrupt()).map(|int| (terminate, int)))
    };
    async move {
        #[cfg(unix)]
        if let Ok((mut terminate, mut interrupt)) = streams {
            tokio::select! {
                _ = terminate.recv() => {}
                _ = interrupt.recv() => {}
            }
            return;
        }
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_replay_overrides, block_match_line, block_match_note, defaults_line,
        entry_prefix_reuse, outcome, serve, shutdown_failure, start_engines,
    };
    use crate::family::flash_next::settings::FlashOptions;
    use crate::family::qwen36::settings::Q36Options;
    use crate::family::sampler::SamplingDefaults;
    use crate::sched::worker::PrefixCapability;
    use crate::sched::worker::{EngineNote, EngineState, ShutdownReport};
    use crate::server::config::{Config, EngineConfig, FakeOptions, StartupError};
    use crate::server::state::AppState;
    use crate::server::testkit;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::net::TcpListener;

    /// A configuration file with the given engines. The listen address and the key variable are
    /// not used by `start_engines`.
    fn file_with(engines: Vec<EngineConfig>, prefix_reuse: bool) -> Config {
        Config {
            listen: "127.0.0.1:0".to_string(),
            api_key_env: "UNUSED".to_string(),
            prefix_reuse,
            engines,
        }
    }

    fn engine_config(id: &str, options: Option<FakeOptions>) -> EngineConfig {
        EngineConfig {
            id: id.to_string(),
            model_dir: PathBuf::from("unused"),
            tokenizer_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/server/testdata/tokenizer"),
            family: "fake".to_string(),
            think_replay: None,
            notice_toolwrap: None,
            model: Some("demo".to_string()),
            context_limit: Some(512),
            fake: options,
            flash_next: None,
            qwen36: None,
            nemotron: None,
            flint: None,
        }
    }

    /// A tokenizer dir carrying the PINNED qwen template, so an entry's
    /// bundle resolves replay by the template default and the config
    /// opt-out is distinguishable from the default.
    fn pinned_tokenizer_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let source =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ] {
            std::fs::copy(source.join(name), dir.path().join(name)).expect("copy fixture");
        }
        std::fs::copy(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/chat_template.jinja"),
            dir.path().join("chat_template.jinja"),
        )
        .expect("copy the pinned template");
        dir
    }

    /// The Flint family exclusion: the config flag alone never arms Flint
    /// prefix reuse; only the env opt-in lifts it, and other families are
    /// unaffected. Pure in (family, file flag, env result) - no engine.
    /// The full 2x2x2 matrix.
    #[test]
    fn flint_prefix_reuse_needs_the_env_opt_in() {
        let flint = crate::family::flint::FAMILY;
        let other = crate::family::qwen36::factory::FAMILY;
        // The full 2x2x2 matrix (family x file flag x env opt-in).
        assert!(
            entry_prefix_reuse(other, true, false),
            "other families follow the flag"
        );
        assert!(
            entry_prefix_reuse(other, true, true),
            "the opt-in is a no-op for other families"
        );
        assert!(
            !entry_prefix_reuse(other, false, false),
            "flag off is off everywhere"
        );
        assert!(
            !entry_prefix_reuse(other, false, true),
            "file off wins over the env everywhere"
        );
        assert!(
            !entry_prefix_reuse(flint, true, false),
            "Flint stays off without the env"
        );
        assert!(
            entry_prefix_reuse(flint, true, true),
            "the opt-in arms Flint"
        );
        assert!(
            !entry_prefix_reuse(flint, false, true),
            "file off wins over the env"
        );
        assert!(!entry_prefix_reuse(flint, false, false), "both off is off");
    }

    /// The per-entry replay opt-out reaches the entry's bundle through
    /// start_engines: the opt-out entry ends with both flags off, and the
    /// wrap-only entry ends with the wrap off at replay's default. The
    /// assertions hold whatever the shell exported (config and env can
    /// only disable); the default-on resolution itself is pinned by the
    /// `resolve_replay_flags` and eligibility pins in
    /// `tokenizer/tests.rs`, and the apply logic by the helper test below
    /// on a forced-on bundle. Deleting the apply call in start_engines
    /// fails the first assertion in any environment.
    #[test]
    fn per_entry_replay_opt_out_reaches_the_bundle() {
        let dir = pinned_tokenizer_dir();
        let mut plain = engine_config("e1", None);
        plain.tokenizer_dir = dir.path().to_path_buf();
        let mut opt_out = engine_config("e2", None);
        opt_out.tokenizer_dir = dir.path().to_path_buf();
        opt_out.think_replay = Some(false);
        let mut wrap_out = engine_config("e3", None);
        wrap_out.tokenizer_dir = dir.path().to_path_buf();
        wrap_out.notice_toolwrap = Some(false);
        let file = file_with(vec![plain, opt_out, wrap_out], false);
        let (_engines, entries) = start_engines(&file).expect("start");
        let by_id = |id: &str| {
            entries
                .iter()
                .find(|entry| entry.id == id)
                .unwrap_or_else(|| panic!("{id} entry"))
        };
        // Env-independent pins: the opt-out entry is off in both flags
        // whatever the shell exported (the config opt-out always disables;
        // the env rollback only ever disables). Which resolution the
        // default entry ended at is not asserted here — see the doc above.
        assert!(
            !by_id("e2").bundle.think_replay(),
            "the entry config disables replay"
        );
        assert!(
            !by_id("e2").bundle.notice_toolwrap(),
            "disabling replay disables the wrap with it"
        );
        assert!(
            !by_id("e3").bundle.notice_toolwrap(),
            "the wrap opt-out disables the wrap"
        );
        assert_eq!(
            by_id("e3").bundle.think_replay(),
            by_id("e1").bundle.think_replay(),
            "the wrap opt-out leaves replay at the default"
        );
    }

    /// The override logic itself, pinned on a bundle whose flags are forced
    /// on: no shell state can flip these asserts.
    #[test]
    fn replay_overrides_apply_and_fail_loud_when_shared() {
        let mut engine = engine_config("e1", None);
        engine.think_replay = Some(false);
        // A uniquely held bundle: the opt-out applies.
        let mut unique = testkit::bundle_with_pinned_template();
        Arc::get_mut(&mut unique)
            .expect("fresh testkit bundle")
            .force_flags_on();
        apply_replay_overrides("e1", &mut unique, &engine).expect("apply");
        assert!(!unique.think_replay(), "replay off");
        assert!(!unique.notice_toolwrap(), "the wrap rides replay off");
        // The wrap opt-out alone: replay stays on.
        let mut engine = engine_config("e1", None);
        engine.notice_toolwrap = Some(false);
        let mut unique = testkit::bundle_with_pinned_template();
        Arc::get_mut(&mut unique)
            .expect("fresh testkit bundle")
            .force_flags_on();
        apply_replay_overrides("e1", &mut unique, &engine).expect("apply");
        assert!(unique.think_replay(), "replay untouched");
        assert!(!unique.notice_toolwrap(), "wrap off");
        // No opt-out: nothing applied.
        let engine = engine_config("e1", None);
        let mut unique = testkit::bundle_with_pinned_template();
        Arc::get_mut(&mut unique)
            .expect("fresh testkit bundle")
            .force_flags_on();
        apply_replay_overrides("e1", &mut unique, &engine).expect("apply");
        assert!(
            unique.think_replay() && unique.notice_toolwrap(),
            "untouched"
        );
        // A shared bundle with an opt-out set: loud startup error, no skip.
        let shared_origin = testkit::bundle_with_pinned_template();
        let mut shared = Arc::clone(&shared_origin);
        let mut engine = engine_config("e1", None);
        engine.think_replay = Some(false);
        let error = apply_replay_overrides("e1", &mut shared, &engine)
            .expect_err("shared bundle must fail loudly");
        assert!(
            error.0.contains("could not be applied"),
            "the error names the failure: {}",
            error.0
        );
    }

    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    #[test]
    fn unsupported_flint_is_recognized_and_refused_before_file_reads() {
        let dir = tempfile::tempdir().expect("temporary config directory");
        let model_dir = dir.path().join("missing-model");
        let tokenizer_dir = dir.path().join("missing-tokenizer");
        let path = dir.path().join("flint.json");
        let text = serde_json::json!({
            "listen": "127.0.0.1:18080",
            "api_key_env": "UNUSED_FLINT_TEST_KEY",
            "engines": [{
                "id": "flint",
                "family": "flint",
                "model_dir": model_dir,
                "tokenizer_dir": tokenizer_dir
            }]
        });
        std::fs::write(&path, text.to_string()).expect("write config");
        let config = Config::from_file(&path).expect("Flint family is recognized");
        assert!(!model_dir.exists());
        assert!(!tokenizer_dir.exists());
        match super::family_starter(&config.engines[0].family).expect("flint starter")(
            &config.engines[0],
            false,
            None,
        ) {
            Err(crate::EngError::BadRequest(message)) => assert_eq!(
                message,
                "engine config: Flint requires --features native-mlx on Apple Silicon or --features native-cuda on Linux"
            ),
            Err(other) => panic!("expected explicit backend refusal, got {other:?}"),
            Ok(_) => panic!("staged Flint must not start a fake or native engine"),
        }

        std::fs::create_dir(&model_dir).expect("model directory");
        std::fs::write(model_dir.join("generation_config.json"), "not JSON")
            .expect("malformed generation config");
        match super::start_engines(&config) {
            Err(error) => assert_eq!(
                error.0,
                "engine flint failed to start: engine config: Flint requires --features native-mlx on Apple Silicon or --features native-cuda on Linux"
            ),
            Ok(_) => panic!("staged Flint must refuse before sampling-default reads"),
        }
    }

    #[test]
    fn the_dense_family_dispatches_to_the_selected_factory_before_any_load() {
        let mut config = engine_config("dense", None);
        config.family = "qwen3_8_27b_dense".into();
        config.context_limit = None;
        config.qwen36 = Some(Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        });
        let error =
            super::family_starter(&config.family).expect("dense starter")(&config, false, None)
                .err()
                .expect("refused");
        let text = super::describe(&error);
        if cfg!(all(feature = "native-cuda", target_os = "linux")) {
            assert!(text.contains("qwen3_8_27b_dense requires the native-mlx backend"));
        } else {
            assert!(text.contains("at least 9283698688 bytes"), "{text}");
        }
    }

    /// Sends one raw HTTP request and reads the whole answer until the server closes the socket.
    fn exchange(address: std::net::SocketAddr, request: String) -> String {
        let mut socket = std::net::TcpStream::connect(address).expect("connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        socket.write_all(request.as_bytes()).expect("write");
        let mut answer = Vec::new();
        socket.read_to_end(&mut answer).expect("read");
        String::from_utf8_lossy(&answer).to_string()
    }

    fn get(path: &str) -> String {
        format!("GET {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n")
    }

    fn post_stream() -> String {
        let body = r#"{"model":"demo","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            testkit::API_KEY,
            body.len()
        )
    }

    /// Starts a server on a free port with one fake engine that replies "abcdef".
    async fn start(
        delay_ms: u64,
    ) -> (
        std::net::SocketAddr,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<Result<(), crate::server::config::StartupError>>,
        Vec<crate::sched::worker::Engine>,
    ) {
        let options = FakeOptions {
            reply: Some("abcdef".to_string()),
            token_delay_ms: delay_ms,
        };
        let (engines, entries) =
            start_engines(&file_with(vec![engine_config("e1", Some(options))], false))
                .expect("engines");
        let state = Arc::new(AppState::new(testkit::API_KEY, entries));
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let (stop, wait) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(serve(listener, state, async move {
            let _ = wait.await;
        }));
        (address, stop, task, engines)
    }

    #[test]
    fn a_shutdown_report_with_a_failure_gives_a_line_that_names_the_engine_and_the_state() {
        let report = ShutdownReport {
            state: EngineState::NativeFailed,
            failure: Some("engine thread panicked: boom".to_string()),
        };
        assert_eq!(
            shutdown_failure("e7", &report),
            Some(
                "engine e7 ended in state native-failed: engine thread panicked: boom".to_string()
            )
        );
    }

    #[test]
    fn a_shutdown_report_without_a_failure_gives_no_line() {
        let report = ShutdownReport {
            state: EngineState::Stopped,
            failure: None,
        };
        assert_eq!(shutdown_failure("e7", &report), None);
    }

    #[test]
    fn a_clean_run_succeeds() {
        assert_eq!(outcome(Ok(()), Vec::new()), Ok(()));
    }

    #[test]
    fn engine_failures_make_the_run_fail_and_are_all_reported() {
        let failures = vec!["engine e1 broke".to_string(), "engine e2 broke".to_string()];
        let error = outcome(Ok(()), failures).expect_err("failures");
        assert_eq!(error.to_string(), "engine e1 broke; engine e2 broke");
    }

    #[test]
    fn a_server_error_comes_before_the_engine_failures() {
        let served = Err(StartupError("server failure: closed".to_string()));
        let error = outcome(served, vec!["engine e1 broke".to_string()]).expect_err("error");
        assert_eq!(error.to_string(), "server failure: closed");
    }

    #[test]
    fn the_engines_of_the_file_become_entries_with_their_model_and_limit() {
        let (engines, entries) = start_engines(&file_with(
            vec![engine_config("e1", None), engine_config("e2", None)],
            false,
        ))
        .expect("engines");
        assert_eq!(engines.len(), 2);
        let ids: Vec<&str> = entries.iter().map(|entry| entry.id.as_str()).collect();
        assert_eq!(ids, ["e1", "e2"]);
        assert_eq!(entries[0].model, "demo");
        assert_eq!(entries[0].context_limit, 512);
    }

    #[test]
    fn the_prefix_reuse_flag_of_the_file_reaches_every_entry() {
        for flag in [false, true] {
            let file = file_with(
                vec![engine_config("e1", None), engine_config("e2", None)],
                flag,
            );
            let (engines, entries) = start_engines(&file).expect("engines");
            let flags: Vec<bool> = entries.iter().map(|entry| entry.prefix_reuse).collect();
            assert_eq!(flags, [flag, flag], "flag {flag}");
            for engine in engines {
                engine.shutdown();
            }
        }
    }

    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    #[test]
    fn a_q36_entry_with_prefix_reuse_keeps_the_no_backend_text() {
        // Without a Q36 backend the factory's own refusal names the missing backend. The session
        // cap covers the prefix-reuse cap rule (5,770,117,120 bytes for the default 8-bit cache at
        // the default context limit of 262,144 positions), which the factory checks before the
        // backend.
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(6_000_000_000),
            ..Q36Options::default()
        };
        let error = start_engines(&file_with(vec![q36_config(Some(options))], true))
            .err()
            .expect("refused");
        let text = error.to_string();
        assert!(
            text.contains("engine e9 failed to start")
                && text.contains("this build has no Q36 backend"),
            "{text}"
        );
    }

    #[test]
    fn prefix_reuse_is_accepted_for_other_families() {
        // The flag reaches every entry of the fake family.
        let file = file_with(vec![engine_config("e1", None)], true);
        let (engines, entries) = start_engines(&file).expect("engines");
        assert!(entries[0].prefix_reuse);
        for engine in engines {
            engine.shutdown();
        }
    }

    #[test]
    fn a_flash_next_entry_goes_to_the_flash_next_factory_and_its_failure_names_the_engine() {
        let mut flash = engine_config("e8", None);
        flash.family = "qwen3_8_flash_next".to_string();
        flash.context_limit = None;
        flash.flash_next = Some(FlashOptions {
            manifest: Some(PathBuf::from("/does/not/exist/download-manifest.json")),
            cache_limit_bytes: Some(1024),
            ..FlashOptions::default()
        });
        let error = start_engines(&file_with(vec![flash], false))
            .err()
            .expect("error");
        let text = error.to_string();
        // A build without the backend stops at the feature check; the Mac build stops at the
        // missing manifest file. The fake family would have started.
        assert!(
            text.contains("engine e8 failed to start")
                && (text.contains("native-mlx") || text.contains("download-manifest.json")),
            "{text}"
        );
    }

    fn q36_config(options: Option<Q36Options>) -> EngineConfig {
        let mut q36 = engine_config("e9", None);
        q36.family = "qwen3_6_35b_a3b".to_string();
        q36.context_limit = None;
        q36.tokenizer_dir = PathBuf::from("/does/not/exist");
        q36.qwen36 = options;
        q36
    }

    // The pinned dispatch of the builds whose Q36 settings read the MLX keys: the CUDA arm
    // ignores `prefill_chunk` (until chunked CUDA prefill uses it as the chunk cap), so this
    // error cannot occur on a native-cuda build.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn the_prefix_reuse_flag_of_the_file_reaches_the_q36_session_cap_rule() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        // With the flag on, the Q36 engine start applies the session-cap rule and refuses the
        // default cap of 4 GiB before it reaches the backend.
        let error = start_engines(&file_with(vec![q36_config(Some(options))], true))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(
            text.contains("engine e9 failed to start")
                && text.contains("at least 5770117120 bytes")
                && text.contains("the cap is 4294967296 bytes")
                && text.contains("raise session_cap_bytes"),
            "{text}"
        );
        // With the flag off, the same entry passes the rule and stops at the backend check.
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        let error = start_engines(&file_with(vec![q36_config(Some(options))], false))
            .err()
            .expect("error");
        assert!(
            !error.to_string().contains("raise session_cap_bytes"),
            "{error}"
        );
    }

    #[test]
    fn a_q36_entry_with_a_bad_member_fails_in_the_q36_settings_and_names_the_engine() {
        // The development engine would not read the Q36 members, so this error proves the dispatch.
        // The entry sets its own cache limit, which wins over `Q36_CACHE_LIMIT`, so an inherited
        // value cannot change the error. The missing limit is tested with an injected environment
        // in `family::qwen36::factory` and with a clean process environment in `server_q36_cli`.
        // `context_limit` is a member that both Q36 arms read and refuse below its minimum of 2,
        // so one test serves every build.
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            context_limit: Some(1),
            ..Q36Options::default()
        };
        let error = start_engines(&file_with(vec![q36_config(Some(options))], false))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(
            text.contains("engine e9 failed to start")
                && text.contains("context_limit must be between 2 and 4294967295, found 1"),
            "{text}"
        );
    }

    #[test]
    fn a_q36_entry_with_options_reaches_the_q36_backend_check() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        let error = start_engines(&file_with(vec![q36_config(Some(options))], false))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(text.contains("engine e9 failed to start"), "{text}");
        // The Mac build reports the missing tokenizer file with the words "is missing", so the
        // check names the cache-limit settings instead.
        assert!(!text.contains("cache_limit_bytes"), "{text}");
        assert!(!text.contains("Q36_CACHE_LIMIT"), "{text}");
        // A build without the backend stops at the feature check. The Mac build and the CUDA
        // build stop at the pinned loader of their Q36 start path, which words a missing file
        // "is missing from the checkpoint directory". The development engine, which an entry
        // reaches when the Q36 arm is skipped, words it "could not be read".
        #[cfg(not(any(
            all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
            all(feature = "native-cuda", target_os = "linux")
        )))]
        assert!(text.contains("native-mlx"), "{text}");
        #[cfg(any(
            all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
            all(feature = "native-cuda", target_os = "linux")
        ))]
        {
            assert!(
                text.contains("tokenizer.json is missing from the checkpoint directory"),
                "{text}"
            );
            assert!(!text.contains("could not be read"), "{text}");
        }
    }

    #[test]
    fn an_engine_that_cannot_start_stops_startup_and_names_the_engine() {
        let mut broken = engine_config("e7", None);
        broken.tokenizer_dir = PathBuf::from("/nonexistent/tokenizer");
        let error = start_engines(&file_with(vec![broken], false))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(
            text.contains("e7") && text.contains("tokenizer.json"),
            "{text}"
        );
    }

    /// A config entry of the fake family whose `model_dir` is a fresh temp directory.
    fn entry_with_model_dir(text: Option<&str>) -> (tempfile::TempDir, EngineConfig) {
        let dir = tempfile::tempdir().expect("temp dir");
        if let Some(text) = text {
            std::fs::write(dir.path().join("generation_config.json"), text)
                .expect("write generation_config.json");
        }
        let mut config = engine_config("e1", None);
        config.model_dir = dir.path().to_path_buf();
        (dir, config)
    }

    #[test]
    fn a_model_dir_with_generation_config_sets_the_defaults_of_the_entry() {
        let (dir, config) =
            entry_with_model_dir(Some(r#"{"temperature": 1.0, "top_p": 0.95, "top_k": 20}"#));
        let (engines, entries) = start_engines(&file_with(vec![config], false)).expect("engines");
        assert_eq!(
            entries[0].defaults,
            SamplingDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20
            }
        );
        drop(dir);
        for engine in engines {
            engine.shutdown();
        }
    }

    #[test]
    fn a_malformed_generation_config_fails_the_engine_start() {
        let (_dir, config) = entry_with_model_dir(Some("{"));
        let error = start_engines(&file_with(vec![config], false))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(
            text.contains("engine e1 failed to start")
                && text.contains("generation_config.json parse failed"),
            "{text}"
        );
    }

    #[test]
    fn an_out_of_range_generation_config_value_fails_the_engine_start() {
        let (_dir, config) = entry_with_model_dir(Some(r#"{"temperature": 2.5}"#));
        let error = start_engines(&file_with(vec![config], false))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(
            text.contains("engine e1 failed to start")
                && text.contains("temperature must be between 0 and 2, found 2.5"),
            "{text}"
        );
    }

    #[test]
    fn an_oversized_generation_config_fails_the_engine_start_and_names_the_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        // One byte over the 1 MiB cap; the engine start must refuse it and name the path.
        std::fs::write(
            dir.path().join("generation_config.json"),
            vec![b'{'; (1 << 20) + 1],
        )
        .expect("write generation_config.json");
        let mut config = engine_config("e1", None);
        config.model_dir = dir.path().to_path_buf();
        let error = start_engines(&file_with(vec![config], false))
            .err()
            .expect("error");
        let text = error.to_string();
        assert!(
            text.contains("engine e1 failed to start") && text.contains("larger than 1 MiB"),
            "{text}"
        );
        assert!(
            text.contains(dir.path().to_str().expect("utf-8 path")),
            "{text}"
        );
    }

    #[test]
    fn the_start_line_quotes_the_file_and_the_effective_defaults() {
        let from_file = defaults_line(
            "q36",
            std::path::Path::new("/m/model/generation_config.json"),
            &SamplingDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            },
            true,
        );
        assert_eq!(
            from_file,
            "engine q36: sampling defaults temperature 1, top_p 0.95, top_k 20 \
             from /m/model/generation_config.json"
        );
        let absent = defaults_line(
            "e1",
            std::path::Path::new("unused/generation_config.json"),
            &SamplingDefaults::CONTRACT,
            false,
        );
        assert_eq!(
            absent,
            "engine e1: sampling defaults temperature 0.7, top_p 1, top_k 0 \
             (unused/generation_config.json absent; contract defaults)"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_server_answers_over_a_socket_and_stops_on_shutdown() {
        let (address, stop, task, engines) = start(0).await;
        let answer = tokio::task::spawn_blocking(move || exchange(address, get("/health")))
            .await
            .expect("client");
        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        assert!(answer.contains(r#"{"status":"ok"}"#), "{answer}");
        stop.send(()).expect("signal");
        let finished = tokio::time::timeout(Duration::from_secs(5), task).await;
        assert!(finished.is_ok(), "the server did not stop");
        for engine in engines {
            engine.shutdown();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_lets_a_running_stream_finish() {
        let (address, stop, task, engines) = start(30).await;
        let (seen, first_event) = std::sync::mpsc::channel::<()>();
        let client = tokio::task::spawn_blocking(move || {
            let mut socket = std::net::TcpStream::connect(address).expect("connect");
            socket
                .set_read_timeout(Some(Duration::from_secs(10)))
                .expect("timeout");
            socket.write_all(post_stream().as_bytes()).expect("write");
            let mut answer = Vec::new();
            let mut chunk = [0_u8; 256];
            let mut signalled = false;
            loop {
                let count = socket.read(&mut chunk).expect("read");
                if count == 0 {
                    break;
                }
                answer.extend_from_slice(&chunk[..count]);
                if !signalled && String::from_utf8_lossy(&answer).contains("data: {") {
                    signalled = true;
                    seen.send(()).expect("signal the test");
                }
            }
            String::from_utf8_lossy(&answer).to_string()
        });
        tokio::task::spawn_blocking(move || first_event.recv().expect("first event"))
            .await
            .expect("wait");
        stop.send(()).expect("signal");
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(
            !task.is_finished(),
            "the server stopped while a stream was still running"
        );
        let answer = client.await.expect("client");
        assert!(answer.contains("data: [DONE]"), "{answer}");
        let finished = tokio::time::timeout(Duration::from_secs(5), task).await;
        assert!(finished.is_ok(), "the server did not stop");
        for engine in engines {
            engine.shutdown();
        }
    }

    /// The block prefix-match startup line follows the backend capability, not
    /// the family name: only an entry whose engine's ready note reported a
    /// store with two or more entries and whose config kept `prefix_reuse` may
    /// claim the match; a rolled-back spacing prints the off line.
    #[test]
    fn the_block_match_line_follows_the_backend_capability() {
        assert_eq!(
            block_match_line("e", true, true, Some(4_096)).as_deref(),
            Some("engine e: block prefix-match on (spacing 4096 tokens)")
        );
        assert_eq!(
            block_match_line("e", true, true, None).as_deref(),
            Some("engine e: block prefix-match off (rollback)")
        );
        assert_eq!(
            block_match_line("e", true, false, Some(4_096)),
            None,
            "an engine whose store holds fewer than two entries must not claim the match"
        );
        assert_eq!(
            block_match_line("e", false, true, Some(4_096)),
            None,
            "an entry opted out of prefix_reuse prints nothing"
        );
    }

    /// The ready note must go through the same formatter with the executor's
    /// own spacing: a mutant that hard-codes or inverts the capability, or
    /// prints the process-wide spacing instead of the payload's, fails
    /// here, because the capturing sink records exactly what the engine
    /// thread would print.
    #[test]
    fn the_ready_note_prints_the_line_only_for_a_capable_store() {
        let (tx, rx) = std::sync::mpsc::channel();
        let capable = block_match_note("e", true, move |line| {
            let _ = tx.send(line.to_string());
        })
        .expect("the production note exists");
        capable(EngineNote::Ready(PrefixCapability {
            cuts_block: true,
            spacing: Some(4_096),
        }));
        assert_eq!(
            rx.recv().expect("the capable engine prints its line"),
            "engine e: block prefix-match on (spacing 4096 tokens)"
        );
        let (tx, rx) = std::sync::mpsc::channel();
        let incapable = block_match_note("e", true, move |line| {
            let _ = tx.send(line.to_string());
        })
        .expect("the production note exists");
        incapable(EngineNote::Ready(PrefixCapability {
            cuts_block: false,
            spacing: Some(4_096),
        }));
        assert!(
            rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "a store that cannot cut prints nothing"
        );
        // The failure arm prints the exact line /ready's log builds, so a
        // load that fails before anyone polls is no longer silent. The
        // duplicate at the first /ready call is accepted (log_new_failures'
        // `reported` set cannot see this spawn-time print).
        let (tx, rx) = std::sync::mpsc::channel();
        let failed = block_match_note("e", true, move |line| {
            let _ = tx.send(line.to_string());
        })
        .expect("the production note exists");
        failed(EngineNote::Failed("engine load failed: no executor".into()));
        assert_eq!(
            rx.recv().expect("a failed load prints its line"),
            "engine e failed: engine load failed: no executor"
        );
    }
}
