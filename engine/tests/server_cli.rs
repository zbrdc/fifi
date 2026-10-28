//! Command-line tests of `enginesrv`: configuration errors, startup, a served request, and shutdown.
//! The server runs the `fake` family, which replays a fixed reply and needs no checkpoint.
//! The tests use `kill`, so they run on Unix only.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const KEY: &str = "sk-cli-test-key-4711";
const KEY_VARIABLE: &str = "ENGINESRV_CLI_TEST_KEY";

fn tokenizer_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer")
}

fn config_text(listen: &str, family: &str) -> String {
    // The `fake` member belongs to the fake family alone; any other family carries neither
    // `fake` nor `context_limit`.
    let fake = if family == "fake" {
        r#", "fake": {"reply": "Pong."}"#
    } else {
        ""
    };
    format!(
        r#"{{
            "listen": "{listen}",
            "api_key_env": "{KEY_VARIABLE}",
            "engines": [{{
                "id": "e1", "model": "demo", "family": "{family}",
                "model_dir": "unused", "tokenizer_dir": "{}"{fake}
            }}]
        }}"#,
        tokenizer_dir().display()
    )
}

fn write_config(dir: &Path, text: &str) -> PathBuf {
    let path = dir.join("enginesrv.json");
    std::fs::write(&path, text).expect("write config");
    path
}

fn command(args: &[&str], key: Option<&str>) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_enginesrv"));
    command.args(args).env_remove(KEY_VARIABLE);
    if let Some(key) = key {
        command.env(KEY_VARIABLE, key);
    }
    command
}

/// Waits up to ten seconds for the child to exit. When the time is up, it kills the child.
fn wait_for_exit(child: &mut Child) -> Option<ExitStatus> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().expect("poll the child") {
            return Some(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Runs the binary to its end and returns the exit code and the standard error text.
/// The exit code is `None` when the process did not exit in time or died by a signal.
fn fail(args: &[&str], key: Option<&str>) -> (Option<i32>, String) {
    let mut child = command(args, key)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn enginesrv");
    let status = wait_for_exit(&mut child);
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("stderr")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    (status.and_then(|status| status.code()), stderr)
}

#[test]
fn a_missing_config_argument_prints_the_usage_and_fails() {
    let (code, stderr) = fail(&[], Some(KEY));
    assert_eq!(code, Some(1));
    assert!(
        stderr.contains("usage: enginesrv --config <path>"),
        "{stderr}"
    );
}

#[test]
fn a_missing_config_file_names_the_path() {
    let (code, stderr) = fail(&["--config", "/nonexistent/enginesrv.json"], Some(KEY));
    assert_eq!(code, Some(1));
    assert!(stderr.contains("/nonexistent/enginesrv.json"), "{stderr}");
}

#[test]
fn a_missing_key_variable_stops_startup_and_names_the_variable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_config(dir.path(), &config_text("127.0.0.1:0", "fake"));
    let (code, stderr) = fail(&["--config", path.to_str().expect("utf-8")], None);
    assert_eq!(code, Some(1));
    assert!(stderr.contains(KEY_VARIABLE), "{stderr}");
}

#[test]
fn an_unsupported_family_stops_startup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_config(dir.path(), &config_text("127.0.0.1:0", "flash-next"));
    let (code, stderr) = fail(&["--config", path.to_str().expect("utf-8")], Some(KEY));
    assert_eq!(code, Some(1));
    assert!(stderr.contains("flash-next"), "{stderr}");
    assert!(stderr.contains("does not support"), "{stderr}");
}

#[test]
fn an_address_in_use_stops_startup() {
    let holder = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let listen = holder.local_addr().expect("address").to_string();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_config(dir.path(), &config_text(&listen, "fake"));
    let (code, stderr) = fail(&["--config", path.to_str().expect("utf-8")], Some(KEY));
    assert_eq!(code, Some(1));
    assert!(stderr.contains(&listen), "{stderr}");
}

/// A running `enginesrv` and the address that it printed. Dropping it kills the process.
struct Running {
    child: Child,
    address: String,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts the server and waits up to ten seconds for its banner. The guard owns the child from
/// the spawn on, so a wrong banner or a silent child never leaves a process behind.
fn start(dir: &Path) -> Running {
    let path = write_config(dir, &config_text("127.0.0.1:0", "fake"));
    let mut running = Running {
        child: command(&["--config", path.to_str().expect("utf-8")], Some(KEY))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn enginesrv"),
        address: String::new(),
    };
    let stdout = running.child.stdout.take().expect("stdout");
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let read = BufReader::new(stdout).read_line(&mut line);
        let _ = sender.send(read.map(|_| line));
    });
    let line = receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("the server printed no banner within ten seconds")
        .expect("read the banner");
    running.address = line
        .trim()
        .strip_prefix("enginesrv listening on ")
        .unwrap_or_else(|| panic!("unexpected banner {line:?}"))
        .to_string();
    running
}

fn exchange(address: &str, request: &str) -> String {
    let mut socket = std::net::TcpStream::connect(address).expect("connect");
    socket
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    socket.write_all(request.as_bytes()).expect("write");
    let mut answer = String::new();
    socket.read_to_string(&mut answer).expect("read");
    answer
}

/// Sends the named signal and returns the exit code and the standard error text of the server.
/// The exit code is `None` when the server did not exit in time or died by a signal.
fn stop_with(mut running: Running, signal: &str) -> (Option<i32>, String) {
    let status = Command::new("kill")
        .args([signal, &running.child.id().to_string()])
        .status()
        .expect("run kill");
    assert!(status.success());
    let exit = wait_for_exit(&mut running.child);
    let mut stderr = String::new();
    running
        .child
        .stderr
        .take()
        .expect("stderr")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    (exit.and_then(|status| status.code()), stderr)
}

#[test]
fn the_server_serves_a_chat_request_and_exits_with_code_zero_on_sigterm() {
    let dir = tempfile::tempdir().expect("tempdir");
    let running = start(dir.path());
    let health = exchange(
        &running.address,
        "GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n",
    );
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    let body = r#"{"model":"demo","messages":[{"role":"user","content":"ping"}]}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let reply = exchange(&running.address, &request);
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.contains(r#""content":"Pong.""#), "{reply}");
    let rejected = exchange(
        &running.address,
        "GET /v1/models HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer wrong\r\nConnection: close\r\n\r\n",
    );
    assert!(rejected.starts_with("HTTP/1.1 401"), "{rejected}");
    let (code, stderr) = stop_with(running, "-TERM");
    assert_eq!(code, Some(0), "SIGTERM must end with code 0: {stderr}");
    assert!(!stderr.contains(KEY), "the key reached the log");
}

#[test]
fn a_sigint_right_after_the_banner_ends_the_server_with_code_zero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let running = start(dir.path());
    let (code, stderr) = stop_with(running, "-INT");
    assert_eq!(code, Some(0), "SIGINT must end with code 0: {stderr}");
    assert!(!stderr.contains(KEY), "the key reached the log");
}

#[test]
fn a_sigterm_right_after_the_banner_ends_the_server_with_code_zero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let running = start(dir.path());
    let (code, stderr) = stop_with(running, "-TERM");
    assert_eq!(code, Some(0), "SIGTERM must end with code 0: {stderr}");
}
