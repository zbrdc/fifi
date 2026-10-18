use std::{
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use crossbeam_channel::{RecvTimeoutError, SendTimeoutError, bounded};

use crate::fixtures::{
    CALLER_COUNT, JOBS_PER_CALLER, Output, Request, Response, TOTAL_JOBS, fixture_for,
};
use crate::ledger::{Ledger, LedgerError};

const REQUEST_QUEUE_CAPACITY: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeFailure {
    pub operation: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunError {
    Deadline {
        stage: &'static str,
    },
    Compute {
        operation: &'static str,
        message: String,
    },
    MisroutedResponse {
        expected: Request,
        observed: Request,
    },
    RequestChannelClosed,
    ResponseChannelClosed,
    Ledger(LedgerError),
    ThreadPanic {
        stage: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    CallerSubmitted {
        caller: u8,
        job: u32,
    },
    CallerReceived {
        caller: u8,
        job: u32,
    },
    OwnerDequeued {
        caller: u8,
        job: u32,
        post_dequeue_queue_len: usize,
    },
    OwnerResponded {
        caller: u8,
        job: u32,
    },
}

#[derive(Debug)]
pub struct RunStats {
    pub total_jobs: u32,
    pub max_post_dequeue_queue_len: usize,
    pub events: Vec<Event>,
}

pub fn validate_routing(expected: Request, observed: Request) -> Result<(), RunError> {
    if expected != observed {
        return Err(RunError::MisroutedResponse { expected, observed });
    }
    Ok(())
}

struct RunState {
    ledger: Mutex<Ledger>,
    events: Mutex<Vec<Event>>,
    error: Mutex<Option<RunError>>,
}

impl RunState {
    fn shared() -> Self {
        Self {
            ledger: Mutex::new(Ledger::new(CALLER_COUNT, JOBS_PER_CALLER)),
            events: Mutex::new(Vec::new()),
            error: Mutex::new(None),
        }
    }

    fn record(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }

    fn with_ledger<T>(&self, action: impl FnOnce(&mut Ledger) -> T) -> T {
        action(&mut self.ledger.lock().unwrap())
    }

    fn fail(&self, error: RunError) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn take_error(&self) -> Option<RunError> {
        self.error.lock().unwrap().take()
    }

    fn settle(self) -> Result<RunStats, RunError> {
        self.with_ledger(|ledger| ledger.finish())
            .map_err(RunError::Ledger)?;
        let events = self.events.into_inner().unwrap();
        let max_post_dequeue_queue_len = events
            .iter()
            .filter_map(|event| match event {
                Event::OwnerDequeued {
                    post_dequeue_queue_len,
                    ..
                } => Some(*post_dequeue_queue_len),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        Ok(RunStats {
            total_jobs: TOTAL_JOBS,
            max_post_dequeue_queue_len,
            events,
        })
    }
}

struct OwnerSide {
    request_rx: crossbeam_channel::Receiver<Request>,
    response_tx: Vec<crossbeam_channel::Sender<Response>>,
    deadline: Instant,
}

fn map_recv_failure(failure: RecvTimeoutError) -> RunError {
    match failure {
        RecvTimeoutError::Timeout => RunError::Deadline { stage: "receive" },
        RecvTimeoutError::Disconnected => RunError::ResponseChannelClosed,
    }
}

fn map_send_failure<T>(failure: SendTimeoutError<T>) -> RunError {
    match failure {
        SendTimeoutError::Timeout(_) => RunError::Deadline { stage: "send" },
        SendTimeoutError::Disconnected(_) => RunError::RequestChannelClosed,
    }
}

fn owner_loop(
    state: &RunState,
    side: &OwnerSide,
    mut compute: impl FnMut(Request) -> Result<Output, ComputeFailure>,
) -> Result<(), RunError> {
    loop {
        let request = match side.request_rx.recv_deadline(side.deadline) {
            Ok(request) => request,
            Err(RecvTimeoutError::Timeout) => {
                return Err(RunError::Deadline {
                    stage: "owner dequeue",
                });
            }
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        };
        state.record(Event::OwnerDequeued {
            caller: request.caller,
            job: request.job,
            post_dequeue_queue_len: side.request_rx.len(),
        });
        let output = compute(request).map_err(|failure| RunError::Compute {
            operation: failure.operation,
            message: failure.message,
        })?;
        let response = Response { request, output };
        let sender = &side.response_tx[response.request.caller as usize];
        match sender.send_deadline(response, side.deadline) {
            Ok(()) => {}
            Err(SendTimeoutError::Timeout(_)) => {
                return Err(RunError::Deadline {
                    stage: "owner response send",
                });
            }
            Err(SendTimeoutError::Disconnected(_)) => {
                return Err(RunError::ResponseChannelClosed);
            }
        }
        state.record(Event::OwnerResponded {
            caller: request.caller,
            job: request.job,
        });
    }
}

struct CallerSide {
    caller: u8,
    request_tx: crossbeam_channel::Sender<Request>,
    response_rx: crossbeam_channel::Receiver<Response>,
    deadline: Instant,
}

fn caller_loop(state: &RunState, side: &CallerSide) -> Result<(), RunError> {
    for job in 0..JOBS_PER_CALLER {
        let request = Request {
            caller: side.caller,
            job,
            fixture: fixture_for(side.caller, job),
        };
        state
            .with_ledger(|ledger| ledger.record_request(request))
            .map_err(RunError::Ledger)?;
        side.request_tx
            .send_deadline(request, side.deadline)
            .map_err(map_send_failure)?;
        state.record(Event::CallerSubmitted {
            caller: side.caller,
            job,
        });
        let response = side
            .response_rx
            .recv_deadline(side.deadline)
            .map_err(map_recv_failure)?;
        validate_routing(request, response.request)?;
        state
            .with_ledger(|ledger| ledger.record_response(response.request, response.output))
            .map_err(RunError::Ledger)?;
        state.record(Event::CallerReceived {
            caller: side.caller,
            job,
        });
    }
    Ok(())
}

pub fn run_protocol<F, C>(deadline: Instant, new_compute: C) -> Result<RunStats, RunError>
where
    C: FnOnce() -> F + Send + 'static,
    F: FnMut(Request) -> Result<Output, ComputeFailure> + 'static,
{
    let state = Arc::new(RunState::shared());
    let (request_tx, request_rx) = bounded::<Request>(REQUEST_QUEUE_CAPACITY);
    let queues = (0..CALLER_COUNT)
        .map(|_| bounded::<Response>(1))
        .collect::<Vec<_>>();
    let owner = thread::Builder::new()
        .name("owner".into())
        .spawn({
            let state = Arc::clone(&state);
            let side = OwnerSide {
                request_rx,
                response_tx: queues.iter().map(|(tx, _)| tx.clone()).collect::<Vec<_>>(),
                deadline,
            };
            move || {
                let compute = new_compute();
                if let Err(error) = owner_loop(&state, &side, compute) {
                    state.fail(error);
                }
            }
        })
        .map_err(|_| RunError::ThreadPanic {
            stage: "spawn owner",
        })?;
    let callers = (0..CALLER_COUNT)
        .map(|caller| {
            thread::Builder::new()
                .name(format!("caller-{caller}"))
                .spawn({
                    let state = Arc::clone(&state);
                    let side = CallerSide {
                        caller,
                        request_tx: request_tx.clone(),
                        response_rx: queues[caller as usize].1.clone(),
                        deadline,
                    };
                    move || {
                        if let Err(error) = caller_loop(&state, &side) {
                            state.fail(error);
                        }
                    }
                })
                .map_err(|_| RunError::ThreadPanic {
                    stage: "spawn caller",
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(request_tx);
    drop(queues);
    let joined = owner
        .join()
        .map_err(|_| RunError::ThreadPanic { stage: "owner" });
    let joined_callers = callers
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .map_err(|_| RunError::ThreadPanic { stage: "caller" })
        })
        .collect::<Result<Vec<_>, _>>();
    joined?;
    joined_callers?;
    if let Some(error) = state.take_error() {
        return Err(error);
    }
    match Arc::try_unwrap(state) {
        Ok(state) => state.settle(),
        Err(_) => Err(RunError::ThreadPanic { stage: "settle" }),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::fixtures::{expected_output, fixture_for};

    fn stand_in() -> impl FnMut(Request) -> Result<Output, ComputeFailure> {
        move |request: Request| Ok(expected_output(request.fixture))
    }

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    #[test]
    fn stand_in_run_completes_every_job() {
        let stats = run_protocol(far_deadline(), stand_in).unwrap();
        assert_eq!(stats.total_jobs, TOTAL_JOBS);
        assert!(stats.max_post_dequeue_queue_len < REQUEST_QUEUE_CAPACITY);
        assert_eq!(stats.events.len(), 4 * TOTAL_JOBS as usize);
        for caller in 0..CALLER_COUNT {
            let submitted = stats
                .events
                .iter()
                .filter(|event| {
                    matches!(event, Event::CallerSubmitted { caller: c, .. } if *c == caller)
                })
                .count();
            assert_eq!(submitted as u32, JOBS_PER_CALLER);
        }
    }

    #[test]
    fn owner_sends_each_response_before_dequeuing_the_next_job() {
        let stats = run_protocol(far_deadline(), stand_in).unwrap();
        let mut responded_since_last_dequeue = 0usize;
        let mut seen_first_dequeue = false;
        let mut violations = 0usize;
        for event in &stats.events {
            match event {
                Event::OwnerDequeued { .. } => {
                    if seen_first_dequeue && responded_since_last_dequeue != 1 {
                        violations += 1;
                    }
                    seen_first_dequeue = true;
                    responded_since_last_dequeue = 0;
                }
                Event::OwnerResponded { .. } => {
                    if !seen_first_dequeue {
                        violations += 1;
                    }
                    responded_since_last_dequeue += 1;
                }
                _ => {}
            }
        }
        assert_eq!(violations, 0);
        assert_eq!(responded_since_last_dequeue, 1);
    }

    #[test]
    fn callers_receive_a_response_before_submitting_the_next_job() {
        let stats = run_protocol(far_deadline(), stand_in).unwrap();
        let position = |caller: u8, job: u32, received: bool| {
            stats
                .events
                .iter()
                .position(|event| {
                    matches!(
                        event,
                        Event::CallerReceived { caller: c, job: j, .. }
                            if *c == caller && *j == job && received
                    ) || matches!(
                        event,
                        Event::CallerSubmitted { caller: c, job: j, .. }
                            if *c == caller && *j == job && !received
                    )
                })
                .unwrap()
        };
        for caller in 0..CALLER_COUNT {
            for job in 0..JOBS_PER_CALLER.saturating_sub(1) {
                assert!(
                    position(caller, job, true) < position(caller, job + 1, false),
                    "caller {caller} submitted job {} before receiving job {job}",
                    job + 1
                );
            }
        }
    }

    #[test]
    fn misrouted_responses_are_rejected() {
        let expected = Request {
            caller: 0,
            job: 3,
            fixture: fixture_for(0, 3),
        };
        let wrong_caller = Request {
            caller: 1,
            ..expected
        };
        let wrong_job = Request { job: 4, ..expected };
        let wrong_fixture = Request {
            fixture: 1 - expected.fixture,
            ..expected
        };
        for observed in [wrong_caller, wrong_job, wrong_fixture] {
            assert!(matches!(
                validate_routing(expected, observed),
                Err(RunError::MisroutedResponse { .. })
            ));
        }
        assert_eq!(validate_routing(expected, expected), Ok(()));
        let _ = Response {
            request: expected,
            output: expected_output(expected.fixture),
        };
    }

    #[test]
    fn expired_deadlines_fail_the_run() {
        let stalled = move || {
            move |_request: Request| -> Result<Output, ComputeFailure> {
                std::thread::sleep(Duration::from_secs(2));
                Ok(expected_output(0))
            }
        };
        let deadline = Instant::now() + Duration::from_millis(200);
        let started = Instant::now();
        let result = run_protocol(deadline, stalled);
        assert!(
            matches!(result, Err(RunError::Deadline { .. })),
            "{result:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn compute_failures_fail_the_run() {
        let failing = move || {
            move |request: Request| -> Result<Output, ComputeFailure> {
                if request.caller == 1 && request.job == 7 {
                    return Err(ComputeFailure {
                        operation: "stand-in",
                        message: "injected".into(),
                    });
                }
                Ok(expected_output(request.fixture))
            }
        };
        let result = run_protocol(far_deadline(), failing);
        assert!(
            matches!(&result, Err(RunError::Compute { operation, message }) if *operation == "stand-in" && message == "injected"),
            "{result:?}"
        );
    }

    #[test]
    fn request_and_response_queues_use_the_contract_capacities() {
        let (request_tx, _request_rx) = bounded::<Request>(REQUEST_QUEUE_CAPACITY);
        let (response_tx, _response_rx) = bounded::<Response>(1);
        drop(request_tx);
        drop(response_tx);
        assert_eq!(REQUEST_QUEUE_CAPACITY, 32);
    }
}
