//! Command-line tests of the `qwen3_8_flash_next` entry of `enginesrv`.
//! No test loads weights. Every run stops at a configuration or backend error before an engine
//! thread starts.

use std::process::{Command, Output};

const KEY: &str = "k-secret-4417-do-not-print";
const KEY_VARIABLE: &str = "ENGINESRV_FLASH_CLI_KEY";

fn config(listen: &str, flash_next: Option<serde_json::Value>) -> serde_json::Value {
    let mut engine = serde_json::json!({
        "id": "flash",
        "family": "qwen3_8_flash_next",
        "model_dir": "/does/not/exist",
        "tokenizer_dir": "/does/not/exist"
    });
    if let Some(options) = flash_next {
        engine["flash_next"] = options;
    }
    serde_json::json!({
        "listen": listen,
        "api_key_env": KEY_VARIABLE,
        "engines": [engine]
    })
}

/// Runs `enginesrv --config` on `value` with a clean Flash-Next environment. `env` adds variables.
fn run(value: &serde_json::Value, env: &[(&str, &str)]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_enginesrv"));
    command
        .arg("--config")
        .arg(&path)
        .env_remove("FLASH_NEXT_ARTIFACT_MANIFEST")
        .env_remove("FLASH_NEXT_CACHE_LIMIT")
        .env(KEY_VARIABLE, KEY);
    for (name, value) in env {
        command.env(name, value);
    }
    command.output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn an_entry_without_a_manifest_stops_startup_and_names_the_engine_and_the_key() {
    let output = run(&config("127.0.0.1:0", None), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("engine flash failed to start"), "{text}");
    assert!(
        text.contains("manifest is missing and FLASH_NEXT_ARTIFACT_MANIFEST is not set"),
        "{text}"
    );
}

#[test]
fn an_entry_without_a_cache_limit_stops_startup_and_names_the_key() {
    let options = serde_json::json!({"manifest": "/does/not/exist/download-manifest.json"});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output)
            .contains("cache_limit_bytes is missing and FLASH_NEXT_CACHE_LIMIT is not set"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn an_unknown_member_of_the_flash_next_object_stops_startup_and_names_it() {
    let options = serde_json::json!({"context_limit": 99});
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("unknown field `context_limit`"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn the_environment_supplies_the_manifest_and_the_limit() {
    let env = [
        (
            "FLASH_NEXT_ARTIFACT_MANIFEST",
            "/does/not/exist/download-manifest.json",
        ),
        ("FLASH_NEXT_CACHE_LIMIT", "1024"),
    ];
    let output = run(&config("127.0.0.1:0", None), &env);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(!text.contains("is missing"), "{text}");
    assert!(!text.contains("is not set"), "{text}");
}

#[test]
fn a_listen_address_on_port_8788_is_refused_for_a_flash_entry() {
    let output = run(&config("127.0.0.1:8788", None), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(stderr(&output).contains("8788"), "{}", stderr(&output));
}

#[test]
fn a_listen_address_off_the_loopback_is_refused_for_a_flash_entry() {
    let output = run(&config("0.0.0.0:0", None), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(stderr(&output).contains("loopback"), "{}", stderr(&output));
}

#[test]
fn the_key_value_never_reaches_stdout_or_stderr() {
    let output = run(&config("127.0.0.1:0", None), &[]);
    assert_eq!(output.status.code(), Some(1), "the entry has no manifest");
    assert!(!stderr(&output).contains(KEY), "key in stderr");
    assert!(
        !String::from_utf8_lossy(&output.stdout).contains(KEY),
        "key in stdout"
    );
}

#[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
#[test]
fn a_complete_entry_in_a_build_without_the_backend_names_the_feature() {
    let options = serde_json::json!({
        "manifest": "/does/not/exist/download-manifest.json",
        "cache_limit_bytes": 1024
    });
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("native-mlx"),
        "{}",
        stderr(&output)
    );
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn a_complete_entry_with_a_missing_manifest_file_stops_startup_and_names_the_path() {
    let options = serde_json::json!({
        "manifest": "/does/not/exist/download-manifest.json",
        "cache_limit_bytes": 1024
    });
    let output = run(&config("127.0.0.1:0", Some(options)), &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("/does/not/exist/download-manifest.json"),
        "{}",
        stderr(&output)
    );
}
