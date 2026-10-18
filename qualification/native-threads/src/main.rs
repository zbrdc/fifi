use std::process::{Command, exit};
use std::time::{Duration, Instant};

use native_threads_qualification::supervisor::supervise;

const SUPERVISOR_DEADLINE: Duration = Duration::from_secs(120);
const SUPERVISOR_POLL: Duration = Duration::from_millis(250);

fn main() {
    let mode = std::env::args().nth(1);
    match mode.as_deref() {
        Some("--test-child") => test_child(),
        Some("run") => exit(run_native(false)),
        Some("run-gate") => exit(run_native(true)),
        Some("gate") => exit(supervised_run("run-gate")),
        _ => exit(supervised_run("run")),
    }
}

fn run_native(two_threads: bool) -> i32 {
    #[cfg(target_os = "macos")]
    {
        native_threads_qualification::native::run(two_threads)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mode = if two_threads {
            "two-thread gate"
        } else {
            "concurrent-caller run"
        };
        eprintln!("the native {mode} requires macOS");
        2
    }
}

fn supervised_run(mode: &str) -> i32 {
    let mut child = Command::new(std::env::current_exe().unwrap());
    child.arg(mode);
    println!(
        "{{\"stage\":\"supervisor\",\"deadline_s\":{}}}",
        SUPERVISOR_DEADLINE.as_secs()
    );
    let deadline = Instant::now() + SUPERVISOR_DEADLINE;
    match supervise(&mut child, deadline, SUPERVISOR_POLL) {
        Ok(outcome) => {
            println!(
                "{{\"stage\":\"supervisor_result\",\"exit_ok\":{},\"deadline_expired\":{}}}",
                outcome.exit_ok, outcome.deadline_expired
            );
            if outcome.exit_ok && !outcome.deadline_expired {
                0
            } else {
                1
            }
        }
        Err(error) => {
            eprintln!("{{\"stage\":\"supervisor_error\",\"error\":{error:?}}}");
            1
        }
    }
}

fn test_child() {
    let behavior = std::env::args().nth(2).unwrap_or_default();
    match behavior.as_str() {
        "quick" => exit(0),
        "park" => {
            std::thread::sleep(Duration::from_secs(3600));
            exit(0);
        }
        _ => {
            eprintln!("unknown test-child behavior: {behavior}");
            exit(2);
        }
    }
}
