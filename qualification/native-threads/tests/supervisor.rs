use std::process::Command;
use std::time::{Duration, Instant};

use native_threads_qualification::supervisor::{ChildOutcome, supervise};

fn child(mode: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_native-threads-qualification"));
    command.args(["--test-child", mode]);
    command
}

#[test]
fn a_quick_child_passes_without_deadline_expiry() {
    let deadline = Instant::now() + Duration::from_secs(10);
    let outcome = supervise(&mut child("quick"), deadline, Duration::from_millis(25)).unwrap();
    assert_eq!(
        outcome,
        ChildOutcome {
            exit_ok: true,
            deadline_expired: false
        }
    );
}

#[test]
fn a_parked_child_is_stopped_at_the_deadline() {
    let deadline = Instant::now() + Duration::from_millis(300);
    let outcome = supervise(&mut child("park"), deadline, Duration::from_millis(25)).unwrap();
    assert!(outcome.deadline_expired);
    assert!(!outcome.exit_ok);
}

#[test]
fn the_supervisor_signals_no_process_beyond_its_own_child() {
    let bystander = child("park").spawn().unwrap();
    let deadline = Instant::now() + Duration::from_millis(300);
    let outcome = supervise(&mut child("park"), deadline, Duration::from_millis(25)).unwrap();
    assert!(outcome.deadline_expired);
    let mut bystander = bystander;
    assert!(
        bystander.try_wait().unwrap().is_none(),
        "the supervisor must not stop a process it does not own"
    );
    bystander.kill().unwrap();
    bystander.wait().unwrap();
}

#[cfg(not(target_os = "macos"))]
#[test]
fn gate_mode_uses_a_supervised_native_gate_child() {
    let output = Command::new(env!("CARGO_BIN_EXE_native-threads-qualification"))
        .arg("gate")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stdout.contains("\"stage\":\"supervisor\""));
    assert!(stdout.contains("\"deadline_expired\":false"));
    assert!(stderr.contains("the native two-thread gate requires macOS"));
}
