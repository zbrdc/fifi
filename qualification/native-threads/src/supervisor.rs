use std::{
    process::{Child, Command},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildOutcome {
    pub exit_ok: bool,
    pub deadline_expired: bool,
}

#[derive(Debug)]
pub enum SupervisorError {
    Spawn(std::io::Error),
    Wait(std::io::Error),
    Kill(std::io::Error),
}

pub fn supervise(
    command: &mut Command,
    deadline: Instant,
    poll_interval: Duration,
) -> Result<ChildOutcome, SupervisorError> {
    let mut child = command.spawn().map_err(SupervisorError::Spawn)?;
    loop {
        if let Some(status) = child.try_wait().map_err(SupervisorError::Wait)? {
            return Ok(ChildOutcome {
                exit_ok: status.success(),
                deadline_expired: false,
            });
        }
        if Instant::now() >= deadline {
            return stop_at_deadline(&mut child);
        }
        std::thread::sleep(poll_interval);
    }
}

fn stop_at_deadline(child: &mut Child) -> Result<ChildOutcome, SupervisorError> {
    child.kill().map_err(SupervisorError::Kill)?;
    let status = child.wait().map_err(SupervisorError::Wait)?;
    Ok(ChildOutcome {
        exit_ok: status.success(),
        deadline_expired: true,
    })
}
