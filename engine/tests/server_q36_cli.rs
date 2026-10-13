//! Command-line tests of the `qwen3_6_35b_a3b` entry of `enginesrv` (packet Q36-F3c).
//! No test loads weights. Every run stops at a configuration or backend error before an engine
//! thread starts. Q36 results never satisfy Phase 0B or Phase 1B.
//! The tests whose premise is the MLX reading (the cache-limit requirement, the chunked-prefill
//! keys, the `native-mlx` fallback) are cfg'd off on the `native-cuda` Linux build, where the
//! premise is false by design, and each has its CUDA twin there (KUBDAN-MSG-71 item 1): no
//! cache limit needed, the allocator keys ignored with a start line, the budget keys refused
//! by name, the CUDA start's own tokenizer pin.

use std::process::{Command, Output};

const KEY: &str = "k-secret-9921-do-not-print";
const KEY_VARIABLE: &str = "ENGINESRV_Q36_CLI_KEY";

fn config(listen: &str, qwen36: Option<serde_json::Value>) -> serde_json::Value {
    let mut engine = serde_json::json!({
        "id": "q36",
        "family": "qwen3_6_35b_a3b",
        "model_dir": "/does/not/exist",
        "tokenizer_dir": "/does/not/exist"
    });
    if let Some(options) = qwen36 {
        engine["qwen36"] = options;
    }
    serde_json::json!({
        "listen": listen,
        "api_key_env": KEY_VARIABLE,
        "engines": [engine]
    })
}

/// Runs `enginesrv --config` on `value` with a clean Q36 environment. `env` adds variables.
fn run(value: &serde_json::Value, env: &[(&str, &str)]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_enginesrv"));
    command
        .arg("--config")
        .arg(&path)
        .env_remove("Q36_CACHE_LIMIT")
        .env(KEY_VARIABLE, KEY);
    for (name, value) in env {
        command.env(name, value);
    }
    command.output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
#[test]
fn an_entry_without_a_cache_limit_stops_startup_and_names_the_engine_and_the_key() {
    let output = run(&config("127.0.0.1:0", None), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("engine q36 failed to start"), "{text}");
    assert!(
        text.contains("cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"),
        "{text}"
    );
}

/// The CUDA twin: the arm needs no cache limit, so the entry passes the settings and stops at
/// the CUDA start's own tokenizer pin (KUBDAN-MSG-71 item 1).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn an_entry_without_a_cache_limit_needs_none_on_the_cuda_arm() {
    let output = run(&config("127.0.0.1:0", None), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("engine q36 failed to start"), "{text}");
    assert!(!text.contains("cache_limit_bytes"), "{text}");
    assert!(!text.contains("Q36_CACHE_LIMIT"), "{text}");
    assert!(
        text.contains("tokenizer.json is missing from the checkpoint directory"),
        "{text}"
    );
}

#[test]
fn an_unknown_member_of_the_qwen36_object_stops_startup_and_names_it() {
    let options = serde_json::json!({"manifest": "/m.json"});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("unknown field `manifest`"),
        "{}",
        stderr(&output)
    );
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
#[test]
fn a_bad_prefill_chunk_stops_startup_and_names_the_key() {
    let options = serde_json::json!({"cache_limit_bytes": 1024, "prefill_chunk": 0});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("prefill_chunk must be between 1 and 512, found 0"),
        "{}",
        stderr(&output)
    );
}

/// The CUDA twin: the chunk key is IGNORED with one start line (the allocator key too), and
/// the entry stops at the tokenizer pin, never at the MLX range check.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn a_bad_prefill_chunk_is_ignored_on_the_cuda_arm_with_a_start_line() {
    let options = serde_json::json!({"cache_limit_bytes": 1024, "prefill_chunk": 0});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(!text.contains("prefill_chunk must be between"), "{text}");
    assert!(
        text.contains("q36 cuda: prefill_chunk is an MLX chunked-prefill key and is ignored"),
        "{text}"
    );
    assert!(
        text.contains("q36 cuda: cache_limit_bytes is an MLX allocator key and is ignored"),
        "{text}"
    );
    assert!(
        text.contains("tokenizer.json is missing from the checkpoint directory"),
        "{text}"
    );
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
#[test]
fn a_zero_prefill_budget_stops_startup_and_names_the_key() {
    let options = serde_json::json!({"cache_limit_bytes": 1024, "prefill_budget_bytes": 0});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output)
            .contains("prefill_budget_bytes must be between 1 and 18446744073709551615, found 0"),
        "{}",
        stderr(&output)
    );
}

/// The CUDA twin: the budget key is REFUSED by name whatever its value — 0 included —
/// because the arm prefills one token per step until CPF.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn a_zero_prefill_budget_is_refused_by_name_on_the_cuda_arm() {
    let options = serde_json::json!({"cache_limit_bytes": 1024, "prefill_budget_bytes": 0});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains(
            "prefill_budget_bytes is an MLX chunked-prefill key; the CUDA arm prefills one \
             token per step until CPF"
        ),
        "{}",
        stderr(&output)
    );
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
#[test]
fn a_prefill_reserve_without_a_budget_stops_startup_and_names_both_keys() {
    let options =
        serde_json::json!({"cache_limit_bytes": 1024, "prefill_session_reserve_bytes": 5});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("prefill_session_reserve_bytes needs prefill_budget_bytes"),
        "{}",
        stderr(&output)
    );
}

/// The CUDA twin: the reserve key is refused by name too (KUBDAN-MSG-71 item 1).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn a_prefill_reserve_without_a_budget_is_refused_by_name_on_the_cuda_arm() {
    let options =
        serde_json::json!({"cache_limit_bytes": 1024, "prefill_session_reserve_bytes": 5});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains(
            "prefill_session_reserve_bytes is an MLX chunked-prefill key; the CUDA arm \
             prefills one token per step until CPF"
        ),
        "{}",
        stderr(&output)
    );
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
#[test]
fn a_prefill_budget_without_a_reserve_stops_startup_and_names_both_keys() {
    let options = serde_json::json!({"cache_limit_bytes": 1024, "prefill_budget_bytes": 5});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("prefill_budget_bytes needs prefill_session_reserve_bytes"),
        "{}",
        stderr(&output)
    );
}

/// The CUDA twin: a budget alone is refused by the same named rule (KUBDAN-MSG-71 item 1).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn a_prefill_budget_without_a_reserve_is_refused_by_name_on_the_cuda_arm() {
    let options = serde_json::json!({"cache_limit_bytes": 1024, "prefill_budget_bytes": 5});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains(
            "prefill_budget_bytes is an MLX chunked-prefill key; the CUDA arm prefills one \
             token per step until CPF"
        ),
        "{}",
        stderr(&output)
    );
}

/// The environment supplies the cache limit, so startup passes the cache-limit check and stops at
/// the next step. The next step depends on the build: it is the backend check without the native
/// backend, and the tokenizer file on the Mac build and the CUDA build with it (the CUDA arm
/// needs no cache limit at all, so the environment value changes nothing there). The Mac
/// message says "is missing"
/// about `tokenizer.json`, so the test names the cache-limit settings and not that phrase. The
/// Mac and CUDA messages come from the pinned loader of the Q36 start path. The development
/// engine, which an entry reaches when the Q36 arm is skipped, words the same fault "could not
/// be read".
#[test]
fn the_environment_supplies_the_cache_limit() {
    let output = run(&config("127.0.0.1:0", None), &[("Q36_CACHE_LIMIT", "1024")]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("engine q36 failed to start"), "{text}");
    assert!(!text.contains("cache_limit_bytes"), "{text}");
    assert!(!text.contains("Q36_CACHE_LIMIT"), "{text}");
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
fn the_flash_next_object_on_a_q36_entry_stops_startup_and_names_the_family() {
    let mut value = config("127.0.0.1:0", None);
    value["engines"][0]["flash_next"] = serde_json::json!({});
    let output = run(&value, &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("only the qwen3_8_flash_next family"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_listen_address_on_port_8788_is_refused_for_a_q36_entry() {
    let output = run(&config("127.0.0.1:8788", None), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(stderr(&output).contains("8788"), "{}", stderr(&output));
}

#[test]
fn the_key_value_never_reaches_stdout_or_stderr() {
    let output = run(&config("127.0.0.1:0", None), &[]);
    assert_eq!(
        output.status.code(),
        Some(1),
        "the entry has no cache limit"
    );
    assert!(!stderr(&output).contains(KEY), "key in stderr");
    assert!(
        !String::from_utf8_lossy(&output.stdout).contains(KEY),
        "key in stdout"
    );
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
#[test]
fn a_complete_entry_in_a_build_without_the_backend_names_the_feature() {
    let options = serde_json::json!({"cache_limit_bytes": 1024});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("native-mlx"),
        "{}",
        stderr(&output)
    );
}

/// The CUDA twin: a complete entry STARTS on the CUDA arm (the cache limit is ignored with one
/// line) and stops at the CUDA start's own tokenizer pin — never at a backend refusal.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn a_complete_entry_starts_on_the_cuda_arm_and_stops_at_the_tokenizer_pin() {
    let options = serde_json::json!({"cache_limit_bytes": 1024});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(!text.contains("native-mlx"), "{text}");
    assert!(!text.contains("no Q36 backend"), "{text}");
    assert!(
        text.contains("q36 cuda: cache_limit_bytes is an MLX allocator key and is ignored"),
        "{text}"
    );
    assert!(
        text.contains("tokenizer.json is missing from the checkpoint directory"),
        "{text}"
    );
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn a_complete_entry_with_a_missing_tokenizer_directory_stops_startup_and_names_the_file() {
    let options = serde_json::json!({"cache_limit_bytes": 1024});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    // The wording comes from the pinned loader of the Q36 start path. The development engine
    // words the same fault "could not be read".
    assert!(
        text.contains("tokenizer.json is missing from the checkpoint directory"),
        "{text}"
    );
    assert!(!text.contains("could not be read"), "{text}");
}

/// The shipped 36 GB config goes through the real start path: `Config` deserialization,
/// `Config::check`, and `Q36Settings`. Only `listen` and the two directories change. Startup then
/// stops at the next step. That step is the backend check without the native backend
/// (`native-mlx`), the tokenizer file on the Mac build with it, and on the CUDA build the named
/// refusal of `prefill_budget_bytes` — the first chunked-prefill key the shipped Mac config
/// carries that the CUDA arm refuses (KUBDAN-MSG-71 item 1, line-217 case).
#[test]
fn the_shipped_mac_studio_config_passes_every_check_before_the_backend() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/qwen36-mac-studio-36gb.json");
    let text = std::fs::read_to_string(&path).expect("read the shipped config");
    let mut value: serde_json::Value = serde_json::from_str(&text).expect("parse the config");
    let original: std::net::SocketAddr = value["listen"]
        .as_str()
        .expect("listen")
        .parse()
        .expect("listen address");
    assert!(original.ip().is_loopback(), "{original}");
    assert_ne!(original.port(), 8788);
    let variable = value["api_key_env"]
        .as_str()
        .expect("api_key_env")
        .to_owned();
    assert_eq!(variable, "ENGINESRV_API_KEY");
    // The numbers of the README: 30 GiB budget, and the reserve as the plan sum.
    let options = &value["engines"][0]["qwen36"];
    assert_eq!(options["prefill_budget_bytes"], 30u64 << 30);
    assert_eq!(options["cache_limit_bytes"], 4u64 << 30);
    assert_eq!(
        options["prefill_session_reserve_bytes"],
        1_342_177_280u64 + 65_863_680 + (4u64 << 30) + (1 << 30)
    );
    assert!(options.get("prefill_chunk").is_none());
    value["listen"] = serde_json::json!("127.0.0.1:0");
    value["engines"][0]["model_dir"] = serde_json::json!("/does/not/exist");
    value["engines"][0]["tokenizer_dir"] = serde_json::json!("/does/not/exist");
    let output = run(&value, &[(variable.as_str(), KEY)]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("engine q36 failed to start"), "{text}");
    // The expected stop is the next step of the build; its own text may carry the prefix
    // "engine config:".
    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    let expected =
        "this build has no Q36 backend; build with --features native-mlx on Apple Silicon";
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    let expected = "tokenizer.json is missing from the checkpoint directory";
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    let expected = "prefill_budget_bytes is an MLX chunked-prefill key; the CUDA arm prefills one token \
         per step until CPF";
    assert!(text.contains(expected), "{text}");
    let rest = text
        .replace(&format!("engine config: {expected}"), "")
        .replace(expected, "");
    for refused in [
        "cannot parse",
        "unknown field",
        "engine config:",
        "api_key_env",
        "usage:",
    ] {
        assert!(!rest.contains(refused), "{refused}: {text}");
    }
}
