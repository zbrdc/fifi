use serde_json::Value;
use std::{
    path::Path,
    process::{Command, Output},
};

const INPUT_HASH: &str = "14de09435b7daf651b6bfbc7271ad443de5b4b4623f6e393e8fd476f7b34161b";

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/tests/fixtures/scalar.safetensors")
}

fn run(input: &Path, hash: &str, budget: &str, key: &str, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_storage-spike"))
        .arg("--input")
        .arg(input)
        .args([
            "--sha256",
            hash,
            "--max-input-bytes",
            budget,
            "--tensor",
            key,
        ])
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

fn report(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn loads_and_hashes_a_retained_scalar_with_explicit_limits() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("report.json");
    let result = run(&fixture(), INPUT_HASH, "1024", "answer", &output);
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(report["status"], "passed");
    assert_eq!(report["input_sha256"], INPUT_HASH);
    assert_eq!(report["input_bytes"], 73);
    assert_eq!(report["max_input_bytes"], 1024);
    assert_eq!(report["storage_mode"], "owned_anonymous_map");
    assert_eq!(report["initial_read_copy"], true);
    assert_eq!(report["payload_bytes"], 1);
    assert_eq!(report["tensor"]["dtype"], "U8");
    assert_eq!(report["tensor"]["shape"], serde_json::json!([]));
    assert_eq!(report["tensor"]["byte_len"], 1);
    assert_eq!(
        report["tensor"]["sha256"],
        "684888c0ebb17f374298b65ee2807526c066094c701bcc7ebbe1c1095f494fc1"
    );
    assert_eq!(report["source_commit"].as_str().unwrap().len(), 40);
    assert!(report["source_dirty"].is_boolean());
    assert!(report["command"].as_array().unwrap().len() >= 10);
    assert!(report["elapsed_seconds"].as_f64().unwrap() >= 0.0);
    for gate in [
        "direct_file_mapping",
        "backend_import",
        "thread_stress",
        "os_reserve",
        "full_model",
    ] {
        assert_eq!(report["gates"][gate], "not_run");
    }
}

#[test]
fn a_wrong_digest_fails_with_a_structured_report() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("report.json");
    let result = run(&fixture(), &"0".repeat(64), "1024", "answer", &output);
    assert_eq!(result.status.code(), Some(1));
    let report = report(&output);
    assert_eq!(report["status"], "failed");
    assert_eq!(report["input_sha256"], INPUT_HASH);
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("digest mismatch")
    );
    assert!(report["tensor"].is_null());
}

#[test]
fn load_key_and_file_failures_remain_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let malformed = dir.path().join("bad.safetensors");
    std::fs::write(&malformed, b"invalid checkpoint").unwrap();
    for (index, (input, budget, key, cause)) in [
        (fixture(), "1024", "absent", "tensor key is missing"),
        (fixture(), "72", "answer", "MemoryPlan"),
        (dir.path().join("missing"), "1024", "answer", "input open"),
        (dir.path().to_owned(), "1024", "answer", "regular file"),
        (malformed, "1024", "answer", "Checkpoint"),
    ]
    .into_iter()
    .enumerate()
    {
        let output = dir.path().join(format!("{index}.json"));
        let result = run(&input, INPUT_HASH, budget, key, &output);
        assert_eq!(result.status.code(), Some(1));
        let report = report(&output);
        assert_eq!(report["status"], "failed");
        assert!(
            report["error"].as_str().unwrap().contains(cause),
            "{report}"
        );
        assert!(report["tensor"].is_null());
    }
}

#[test]
fn reports_never_replace_existing_reports_or_input_files() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("report.json");
    std::fs::write(&output, b"existing evidence").unwrap();
    let input = dir.path().join("input.safetensors");
    std::fs::copy(fixture(), &input).unwrap();
    let original = std::fs::read(&input).unwrap();
    assert_eq!(
        run(&input, INPUT_HASH, "1024", "answer", &output)
            .status
            .code(),
        Some(1)
    );
    assert_eq!(std::fs::read(&output).unwrap(), b"existing evidence");
    assert_eq!(
        run(&input, INPUT_HASH, "1024", "answer", &input)
            .status
            .code(),
        Some(1)
    );
    assert_eq!(std::fs::read(&input).unwrap(), original);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn report_publication_failures_exit_one() {
    let dir = tempfile::tempdir().unwrap();
    for output in [
        dir.path().join("missing/report.json"),
        dir.path().to_owned(),
    ] {
        let result = run(&fixture(), INPUT_HASH, "1024", "answer", &output);
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&result.stderr).contains("report:"));
    }
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn invalid_arguments_exit_two_without_a_report() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("report.json");
    for args in [
        vec![],
        vec!["--unknown", "x"],
        vec!["--input"],
        vec!["--input", "first", "--input", "second"],
        vec!["--input", ""],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_storage-spike"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(!result.stderr.is_empty());
    }
    for (hash, budget) in [
        ("xyz", "1024"),
        (INPUT_HASH, "0"),
        (INPUT_HASH, "-1"),
        (INPUT_HASH, "18446744073709551616"),
    ] {
        assert_eq!(
            run(&fixture(), hash, budget, "answer", &output)
                .status
                .code(),
            Some(2)
        );
        assert!(!output.exists());
    }
}

#[cfg(unix)]
#[test]
fn report_links_cannot_replace_an_input_or_create_their_target() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input");
    std::fs::copy(fixture(), &input).unwrap();
    let original = std::fs::read(&input).unwrap();
    let hard = dir.path().join("hard");
    std::fs::hard_link(&input, &hard).unwrap();
    let link = dir.path().join("link");
    symlink(&input, &link).unwrap();
    let missing = dir.path().join("missing");
    let dangling = dir.path().join("dangling");
    symlink(&missing, &dangling).unwrap();
    for output in [hard, link, dangling] {
        assert_eq!(
            run(&input, INPUT_HASH, "1024", "answer", &output)
                .status
                .code(),
            Some(1)
        );
        assert_eq!(std::fs::read(&input).unwrap(), original);
    }
    assert!(!missing.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 4);
}

#[cfg(target_os = "linux")]
#[test]
fn non_utf8_input_paths_keep_lossless_report_identity() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let dir = tempfile::tempdir().unwrap();
    let input = dir
        .path()
        .join(std::ffi::OsString::from_vec(vec![b'm', 255]));
    std::fs::copy(fixture(), &input).unwrap();
    let output = dir.path().join("report.json");
    let result = run(&input, INPUT_HASH, "1024", "answer", &output);
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(report["input_sha256"], INPUT_HASH);
    assert_eq!(
        report["command_os_bytes"][2],
        serde_json::json!(input.as_os_str().as_bytes())
    );
    assert_eq!(report["runner_os"], std::env::consts::OS);
}

#[cfg(unix)]
#[test]
fn non_utf8_missing_paths_keep_structured_errors_and_argument_bytes() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let dir = tempfile::tempdir().unwrap();
    let input = dir
        .path()
        .join(std::ffi::OsString::from_vec(vec![b'm', 255]));
    let output = dir.path().join("report.json");
    let result = run(&input, INPUT_HASH, "1024", "answer", &output);
    assert_eq!(
        result.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(report["status"], "failed");
    assert!(report["error"].as_str().unwrap().contains("input open"));
    assert_eq!(
        report["command_os_bytes"][2],
        serde_json::json!(input.as_os_str().as_bytes())
    );
    assert!(report["input_sha256"].is_null());
}
