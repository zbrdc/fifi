//! The request log of `enginesrv`: a served chat request writes one JSON line to the real
//! standard error of the process. The server runs the `fake` family and needs no checkpoint.
//! The test uses `kill`, so it runs on Unix only.

#![cfg(unix)]

use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const KEY: &str = "sk-log-test-key-4711";
const KEY_VARIABLE: &str = "ENGINESRV_LOG_TEST_KEY";

/// A running `enginesrv`. Dropping it kills the process, so that a failed assertion leaves none.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn config_text() -> String {
    let tokenizer = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
    format!(
        r#"{{
            "listen": "127.0.0.1:0",
            "api_key_env": "{KEY_VARIABLE}",
            "engines": [{{
                "id": "e1", "model": "demo", "family": "fake",
                "model_dir": "unused", "tokenizer_dir": "{}",
                "fake": {{"reply": "Pong."}}
            }}]
        }}"#,
        tokenizer.display()
    )
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

fn chat(address: &str, body: &str) -> String {
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    exchange(address, &request)
}

#[test]
fn a_chat_request_writes_one_json_line_to_standard_error_without_text_or_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("enginesrv.json");
    std::fs::write(&path, config_text()).expect("write config");
    let mut server = Running(
        Command::new(env!("CARGO_BIN_EXE_enginesrv"))
            .args(["--config", path.to_str().expect("utf-8")])
            .env(KEY_VARIABLE, KEY)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn enginesrv"),
    );
    let stdout = server.0.stdout.take().expect("stdout");
    let mut banner = String::new();
    BufReader::new(stdout)
        .read_line(&mut banner)
        .expect("banner");
    let address = banner
        .trim()
        .strip_prefix("enginesrv listening on ")
        .unwrap_or_else(|| panic!("unexpected banner {banner:?}"))
        .to_string();
    let served = chat(
        &address,
        r#"{"model":"demo","messages":[{"role":"user","content":"zebra-prompt"}]}"#,
    );
    assert!(served.starts_with("HTTP/1.1 200"), "{served}");
    let reply: Value =
        serde_json::from_str(served.split("\r\n\r\n").nth(1).expect("body")).expect("reply json");
    // A request that fails before it has a prompt writes no line.
    let unknown = chat(
        &address,
        r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert!(unknown.starts_with("HTTP/1.1 404"), "{unknown}");
    let status = Command::new("kill")
        .args(["-TERM", &server.0.id().to_string()])
        .status()
        .expect("run kill");
    assert!(status.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while server.0.try_wait().expect("poll the server").is_none() {
        assert!(Instant::now() < deadline, "the server did not exit");
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut stderr = String::new();
    server
        .0
        .stderr
        .take()
        .expect("stderr")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    let lines: Vec<Value> = stderr
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|line| line.get("request_id").is_some())
        .collect();
    assert_eq!(lines.len(), 1, "{stderr}");
    let line = &lines[0];
    assert_eq!(line["request_id"], reply["id"]);
    assert_eq!(line["model"], "demo");
    assert_eq!(line["stream"], false);
    assert_eq!(line["status"], 200);
    assert_eq!(line["outcome"], "ok");
    assert_eq!(line["finish_reason"], "stop");
    assert_eq!(line["prompt_tokens"], reply["usage"]["prompt_tokens"]);
    assert_eq!(
        line["completion_tokens"],
        reply["usage"]["completion_tokens"]
    );
    assert!(line["prompt_tokens"].as_u64().expect("count") > 0);
    for secret in ["zebra-prompt", "Pong", KEY, "Bearer"] {
        assert!(!stderr.contains(secret), "{secret} reached the log");
    }
}
