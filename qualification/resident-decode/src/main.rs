//! Runs the resident Flash-Next parity and decode tests on the Mac and writes one record.
//!
//! Usage:
//!   resident-decode run --engine <engine dir> --out <dir> --build q|p
//!   resident-decode check --out <dir> --build q|p
//!
//! `run` records identity, runs each ignored test by its exact name with `/usr/bin/time -l`, and then
//! does what `check` does. `check` reads the saved reports and exit codes and rewrites the record.
//! The exit status is 0 only when every gate that the build requires passes.

use resident_decode::gates::{Build, Inputs, record};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

/// Full test path prefix of the ignored engine tests.
const PREFIX: &str = "family::flash_next::resident::parity::";
/// Short name, test name, and report variable of the three tests.
const TESTS: [(&str, &str, &str); 3] = [
    (
        "blocks",
        "resident_blocks_match_the_oracle_teacher_forced",
        "FLASH_NEXT_RESIDENT_BLOCKS_REPORT",
    ),
    (
        "prompt",
        "resident_prompt_matches_the_oracle_end_to_end",
        "FLASH_NEXT_RESIDENT_PROMPT_REPORT",
    ),
    (
        "decode",
        "resident_decode_reports_speed_memory_and_host_tokens",
        "FLASH_NEXT_RESIDENT_DECODE_REPORT",
    ),
];

/// The short names of the tests that `build` runs, in order.
fn selected(build: Build) -> &'static [&'static str] {
    match build {
        Build::Q => &["blocks", "prompt", "decode"],
        Build::P => &["decode"],
    }
}

/// Runs a command and returns its trimmed stdout on one line, or `unavailable` with the reason.
fn capture(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
        Ok(out) => format!("unavailable(exit {:?})", out.status.code()),
        Err(error) => format!("unavailable({error})"),
    }
}

/// SHA-256 of a file through `/usr/bin/shasum -a 256`.
fn sha256(path: &Path) -> String {
    let text = capture("/usr/bin/shasum", &["-a", "256", &path.to_string_lossy()]);
    text.split(' ').next().unwrap_or("unavailable").to_string()
}

/// The `Pageins` count of `vm_stat`.
fn pageins() -> String {
    let text = capture("vm_stat", &[]);
    text.split("Pageins:")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map_or_else(
            || "unavailable".to_string(),
            |v| v.trim_end_matches('.').to_string(),
        )
}

/// The listener on TCP port 8000 (the Splash service), or `none`.
fn port8000() -> String {
    let text = capture("lsof", &["-nP", "-iTCP:8000", "-sTCP:LISTEN"]);
    if text.starts_with("unavailable") {
        "none".to_string()
    } else {
        text
    }
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".to_string())
}

/// Identity lines written before the tests.
fn identity_before(engine: &Path, build: Build) -> String {
    let artifact = PathBuf::from(env("FLASH_NEXT_ARTIFACT"));
    let metallib = PathBuf::from(env("MLX_RS_METAL_PATH")).join("mlx.metallib");
    let engine_text = engine.to_string_lossy();
    let dirty = capture("git", &["-C", &engine_text, "status", "--porcelain"]);
    let lines = [
        ("build", build.name().to_string()),
        ("features", build.features().to_string()),
        ("profile", "release".to_string()),
        (
            "engine_rev",
            capture("git", &["-C", &engine_text, "rev-parse", "HEAD"]),
        ),
        ("engine_dirty", (!dirty.is_empty()).to_string()),
        ("rustc", capture("rustc", &["-V"])),
        ("metallib_sha256", sha256(&metallib)),
        ("artifact_dir", artifact.to_string_lossy().into_owned()),
        (
            "artifact_index_sha256",
            sha256(&artifact.join("model.safetensors.index.json")),
        ),
        (
            "artifact_config_sha256",
            sha256(&artifact.join("config.json")),
        ),
        ("oracle_dir", env("FLASH_NEXT_ORACLE")),
        ("cache_limit_env", env("FLASH_NEXT_CACHE_LIMIT")),
        ("hw_model", capture("sysctl", &["-n", "hw.model"])),
        ("hw_memsize", capture("sysctl", &["-n", "hw.memsize"])),
        ("macos", capture("sw_vers", &["-productVersion"])),
        ("swap_before", capture("sysctl", &["-n", "vm.swapusage"])),
        ("pageins_before", pageins()),
        ("port8000_listener", port8000()),
    ];
    lines.iter().map(|(k, v)| format!("{k}={v}\n")).collect()
}

/// The test binary path from a cargo log line `Running unittests src/lib.rs (<path>)`.
fn test_binary(log: &str) -> Option<String> {
    let line = log
        .lines()
        .find(|l| l.contains("Running unittests src/lib.rs ("))?;
    let start = line.find('(')? + 1;
    let end = line.rfind(')')?;
    line.get(start..end).map(str::to_string)
}

/// Runs one test with `/usr/bin/time -l` and returns its exit code (`None` for a signal).
fn run_test(engine: &Path, out: &Path, build: Build, short: &str) -> Result<Option<i32>, String> {
    let Some((_, name, variable)) = TESTS.iter().find(|(s, _, _)| *s == short) else {
        return Err(format!("unknown test {short}"));
    };
    let report = out.join(format!("{}-{short}-report.txt", build.name()));
    let log_path = out.join(format!("{}-{short}.log", build.name()));
    // A report from an earlier run must not count for this run.
    if report.exists() {
        fs::remove_file(&report).map_err(|e| format!("{}: {e}", report.display()))?;
    }
    let log = File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let log_err = log.try_clone().map_err(|e| e.to_string())?;
    let full = format!("{PREFIX}{name}");
    let args = [
        "-l",
        "cargo",
        "test",
        "--offline",
        "--release",
        "--features",
        build.features(),
        "--lib",
        &full,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ];
    println!("running {full} ({})", build.features());
    let status = Command::new("/usr/bin/time")
        .args(args)
        .current_dir(engine)
        .env(variable, &report)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(log_err)
        .status()
        .map_err(|e| format!("cannot start /usr/bin/time: {e}"))?;
    Ok(status.code())
}

fn read(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// Reads the saved reports and exit codes, writes `<build>-record.txt`, and returns the verdict.
fn check(out: &Path, build: Build) -> Result<bool, String> {
    let b = build.name();
    let exits_text = read(&out.join(format!("{b}-exits.txt"))).unwrap_or_default();
    let exits: Vec<(String, Option<i32>)> = exits_text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.parse().ok()))
        .collect();
    let blocks = read(&out.join(format!("{b}-blocks-report.txt")));
    let prompt = read(&out.join(format!("{b}-prompt-report.txt")));
    let decode = read(&out.join(format!("{b}-decode-report.txt")));
    let inputs = Inputs {
        build,
        blocks: blocks.as_deref(),
        prompt: prompt.as_deref(),
        decode: decode.as_deref(),
        exits: &exits,
    };
    let (text, pass) = record(&inputs);
    let path = out.join(format!("{b}-record.txt"));
    fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
    print!("{text}");
    Ok(pass)
}

fn run(engine: &Path, out: &Path, build: Build) -> Result<bool, String> {
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let out = fs::canonicalize(out).map_err(|e| e.to_string())?;
    let engine = fs::canonicalize(engine).map_err(|e| e.to_string())?;
    let b = build.name();
    let before = identity_before(&engine, build);
    fs::write(out.join(format!("{b}-identity-before.txt")), before).map_err(|e| e.to_string())?;
    let mut exits = String::new();
    for short in selected(build) {
        let code = run_test(&engine, &out, build, short)?;
        let code = code.map_or_else(|| "signal".to_string(), |c| c.to_string());
        exits.push_str(&format!("{short}={code}\n"));
    }
    fs::write(out.join(format!("{b}-exits.txt")), exits).map_err(|e| e.to_string())?;
    let first_log = read(&out.join(format!("{b}-{}.log", selected(build)[0]))).unwrap_or_default();
    let binary = test_binary(&first_log);
    let after = [
        (
            "test_binary",
            binary.clone().unwrap_or_else(|| "unavailable".into()),
        ),
        (
            "test_binary_sha256",
            binary.map_or_else(|| "unavailable".into(), |p| sha256(Path::new(&p))),
        ),
        ("swap_after", capture("sysctl", &["-n", "vm.swapusage"])),
        ("pageins_after", pageins()),
        ("port8000_listener_after", port8000()),
    ];
    let after: String = after.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    fs::write(out.join(format!("{b}-identity-after.txt")), after).map_err(|e| e.to_string())?;
    check(&out, build)
}

/// Returns the value after `flag` in `args`.
fn flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let at = args.iter().position(|a| a == flag)?;
    args.get(at + 1).map(String::as_str)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str);
    let build = flag(&args, "--build").and_then(Build::parse);
    let out = flag(&args, "--out").map(PathBuf::from);
    let result = match (command, build, out) {
        (Some("run"), Some(build), Some(out)) => match flag(&args, "--engine") {
            Some(engine) => run(Path::new(engine), &out, build),
            None => Err("run needs --engine".to_string()),
        },
        (Some("check"), Some(build), Some(out)) => check(&out, build),
        _ => Err("usage: resident-decode run --engine <dir> --out <dir> --build q|p, or check --out <dir> --build q|p".to_string()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("resident-decode: {error}");
            ExitCode::from(2)
        }
    }
}
