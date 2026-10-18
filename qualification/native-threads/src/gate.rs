use std::{
    sync::Arc,
    thread::{self, JoinHandle, ThreadId},
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded};

use crate::fixtures::{
    CALLER_COUNT, JOBS_PER_CALLER, Output, Request, Response, TOTAL_JOBS, fixture_for,
};
use crate::ledger::Ledger;
use crate::protocol::{ComputeFailure, RunError};

#[derive(Debug)]
pub struct GateStats {
    pub total_jobs: u32,
    pub owner_threads: Vec<ThreadId>,
}

struct Worker {
    ready: Receiver<Result<(), RunError>>,
    start: Sender<()>,
    handle: JoinHandle<Result<Vec<Response>, RunError>>,
}

pub fn run_gate<F, C, B>(
    deadline: Instant,
    new_compute: F,
    check_bound: B,
) -> Result<GateStats, RunError>
where
    F: Fn(u8) -> Result<C, ComputeFailure> + Send + Sync + 'static,
    C: FnMut(Request) -> Result<Output, ComputeFailure> + 'static,
    B: FnOnce() -> Result<(), ComputeFailure>,
{
    let factory = Arc::new(new_compute);
    let mut workers = Vec::new();
    let startup = (|| {
        for caller in 0..CALLER_COUNT {
            check_deadline(deadline, "spawn gate owner")?;
            workers.push(spawn_worker(caller, Arc::clone(&factory), deadline)?);
            wait_ready(workers.last().unwrap(), deadline)?;
        }
        check_deadline(deadline, "check owner bound")?;
        check_bound().map_err(compute_error)?;
        release_workers(&workers, deadline)
    })();
    let owner_threads = workers
        .iter()
        .map(|worker| worker.handle.thread().id())
        .collect();
    let joined = join_workers(workers, deadline);
    startup?;
    validate_results(joined?, deadline)?;
    Ok(GateStats {
        total_jobs: TOTAL_JOBS,
        owner_threads,
    })
}

fn check_deadline(deadline: Instant, stage: &'static str) -> Result<(), RunError> {
    if Instant::now() >= deadline {
        return Err(RunError::Deadline { stage });
    }
    Ok(())
}

fn compute_error(failure: ComputeFailure) -> RunError {
    RunError::Compute {
        operation: failure.operation,
        message: failure.message,
    }
}

fn spawn_worker<F, C>(caller: u8, factory: Arc<F>, deadline: Instant) -> Result<Worker, RunError>
where
    F: Fn(u8) -> Result<C, ComputeFailure> + Send + Sync + 'static,
    C: FnMut(Request) -> Result<Output, ComputeFailure> + 'static,
{
    let (ready_tx, ready) = bounded(1);
    let (start, start_rx) = bounded(1);
    let handle = thread::Builder::new()
        .name(format!("gate-owner-{caller}"))
        .spawn(move || {
            let compute = factory(caller).map_err(compute_error);
            ready_tx
                .send(compute.as_ref().map(|_| ()).map_err(Clone::clone))
                .map_err(|_| RunError::ResponseChannelClosed)?;
            let mut compute = compute?;
            start_rx
                .recv_deadline(deadline)
                .map_err(|error| match error {
                    RecvTimeoutError::Timeout => RunError::Deadline {
                        stage: "owner start",
                    },
                    RecvTimeoutError::Disconnected => RunError::RequestChannelClosed,
                })?;
            run_jobs(caller, &mut compute, deadline)
        })
        .map_err(|error| RunError::Compute {
            operation: "spawn gate owner",
            message: error.to_string(),
        })?;
    Ok(Worker {
        ready,
        start,
        handle,
    })
}

fn wait_ready(worker: &Worker, deadline: Instant) -> Result<(), RunError> {
    worker
        .ready
        .recv_deadline(deadline)
        .map_err(|error| match error {
            RecvTimeoutError::Timeout => RunError::Deadline {
                stage: "owner initialization",
            },
            RecvTimeoutError::Disconnected => RunError::ThreadPanic {
                stage: "owner initialization",
            },
        })?
}

fn release_workers(workers: &[Worker], deadline: Instant) -> Result<(), RunError> {
    check_deadline(deadline, "release gate owners")?;
    for worker in workers {
        worker
            .start
            .send_deadline((), deadline)
            .map_err(|_| RunError::RequestChannelClosed)?;
    }
    Ok(())
}

fn run_jobs(
    caller: u8,
    compute: &mut impl FnMut(Request) -> Result<Output, ComputeFailure>,
    deadline: Instant,
) -> Result<Vec<Response>, RunError> {
    let mut responses = Vec::with_capacity(JOBS_PER_CALLER as usize);
    for job in 0..JOBS_PER_CALLER {
        check_deadline(deadline, "gate job")?;
        let request = Request {
            caller,
            job,
            fixture: fixture_for(caller, job),
        };
        let output = compute(request).map_err(compute_error)?;
        responses.push(Response { request, output });
    }
    Ok(responses)
}

fn join_worker(worker: Worker, deadline: Instant) -> Result<Vec<Response>, RunError> {
    drop(worker.start);
    while !worker.handle.is_finished() {
        check_deadline(deadline, "join gate owner")?;
        thread::sleep(
            Duration::from_millis(1).min(deadline.saturating_duration_since(Instant::now())),
        );
    }
    let result = worker.handle.join().map_err(|_| RunError::ThreadPanic {
        stage: "gate owner",
    })?;
    check_deadline(deadline, "join gate owner")?;
    result
}

fn join_workers(workers: Vec<Worker>, deadline: Instant) -> Result<Vec<Vec<Response>>, RunError> {
    let mut outcome = Ok(Vec::new());
    for worker in workers {
        let joined = join_worker(worker, deadline);
        match (&mut outcome, joined) {
            (Ok(batches), Ok(responses)) => batches.push(responses),
            (Ok(_), Err(error)) => outcome = Err(error),
            (Err(_), _) => {}
        }
    }
    outcome
}

fn validate_results(batches: Vec<Vec<Response>>, deadline: Instant) -> Result<(), RunError> {
    let mut ledger = Ledger::new(CALLER_COUNT, JOBS_PER_CALLER);
    for response in batches.into_iter().flatten() {
        check_deadline(deadline, "gate ledger")?;
        ledger
            .record_request(response.request)
            .map_err(RunError::Ledger)?;
        ledger
            .record_response(response.request, response.output)
            .map_err(RunError::Ledger)?;
    }
    ledger.finish().map_err(RunError::Ledger)?;
    check_deadline(deadline, "gate completion")
}

#[cfg(test)]
mod tests;
