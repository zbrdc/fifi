//! Startup and shutdown of `enginesrv`.
//!
//! Order: read the command line, the configuration, and the key; bind the listener; start the
//! engines; serve. `SIGTERM` or `SIGINT` stops the listener, lets the running requests finish,
//! stops the engine threads, and returns. The caller then exits with code 0.

use super::config::{Config, EngineConfig, StartupError, config_path};
use super::fake;
use super::router::router;
use super::state::{AppState, EngineEntry};
use super::wire::state_name;
use crate::EngError;
use crate::family::flash_next::factory;
use crate::family::qwen36::factory as q36_factory;
use crate::family::sampler::SamplingDefaults;
use crate::sched::worker::{Engine, ShutdownReport};
use crate::tokenizer::TokenizerBundle;
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
fn start_engines(file: &Config) -> Result<(Vec<Engine>, Vec<EngineEntry>), StartupError> {
    let mut engines = Vec::new();
    let mut entries = Vec::new();
    for config in &file.engines {
        let loaded = SamplingDefaults::from_model_dir(&config.model_dir).map_err(|error| {
            StartupError(format!(
                "engine {} failed to start: {}",
                config.id,
                describe(&error)
            ))
        })?;
        let (engine, bundle, context_limit) =
            start_family(config, file.prefix_reuse).map_err(|error| {
                StartupError(format!(
                    "engine {} failed to start: {}",
                    config.id,
                    describe(&error)
                ))
            })?;
        let generation = config.model_dir.join("generation_config.json");
        // The label comes from the loader result, not from a second `exists` call, so the file
        // cannot change between the read and the log line.
        eprintln!(
            "{}",
            defaults_line(&config.id, &generation, &loaded.values, loaded.from_file)
        );
        entries.push(EngineEntry {
            id: config.id.clone(),
            model: config.model_name(),
            handle: engine.handle(),
            bundle,
            context_limit,
            prefix_reuse: file.prefix_reuse,
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

/// Starts the engine of one entry. The config check has accepted only the family names of this
/// build, so every family other than Flash-Next and Q36 is the development engine. Each family
/// reports its own context limit. `prefix_reuse` is the process-wide flag of the configuration
/// file; the Q36 engine start enforces its session-cap rule when it is on.
fn start_family(
    config: &EngineConfig,
    prefix_reuse: bool,
) -> Result<(Engine, Arc<TokenizerBundle>, u32), EngError> {
    if config.family == factory::FAMILY {
        let started = factory::start_engine(
            &config.model_dir,
            &config.tokenizer_dir,
            config.flash_next.as_ref(),
        )?;
        return Ok((
            started.engine,
            Arc::new(started.bundle),
            started.context_limit,
        ));
    }
    if config.family == q36_factory::FAMILY {
        let started = q36_factory::start_engine(
            &config.model_dir,
            &config.tokenizer_dir,
            config.qwen36.as_ref(),
            prefix_reuse,
        )?;
        return Ok((
            started.engine,
            Arc::new(started.bundle),
            started.context_limit,
        ));
    }
    let started = fake::start(config)?;
    Ok((started.engine, started.bundle, started.context_limit))
}

fn describe(error: &EngError) -> String {
    match error {
        EngError::Checkpoint(message)
        | EngError::BadRequest(message)
        | EngError::MemoryPlan(message)
        | EngError::ContextLimit(message) => message.clone(),
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
    use super::{defaults_line, outcome, serve, shutdown_failure, start_engines};
    use crate::family::flash_next::settings::FlashOptions;
    use crate::family::qwen36::settings::Q36Options;
    use crate::family::sampler::SamplingDefaults;
    use crate::sched::worker::{EngineState, ShutdownReport};
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
            model: Some("demo".to_string()),
            context_limit: Some(512),
            fake: options,
            flash_next: None,
            qwen36: None,
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
    // ignores `prefill_chunk` by design, so this error cannot occur on a native-cuda build.
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
}
