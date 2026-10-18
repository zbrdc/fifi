use super::*;
use crate::ledger::LedgerError;
use crate::protocol::RunError;

fn failure(operation: &'static str) -> ComputeFailure {
    ComputeFailure {
        operation,
        message: "planned failure".into(),
    }
}

#[test]
fn initialization_failure_releases_every_initialized_owner() {
    for fail_at in [0, 1] {
        let observation = Arc::new(Mutex::new(Observation::default()));
        let created = Arc::clone(&observation);
        let result = run_gate(
            deadline(),
            move |caller| {
                if caller == fail_at {
                    return Err(failure("initialize"));
                }
                Ok(new_compute(caller, Arc::clone(&created)))
            },
            || panic!("the bound check must not follow failed initialization"),
        );
        assert!(matches!(
            result,
            Err(RunError::Compute {
                operation: "initialize",
                ..
            })
        ));
        let state = observation.lock().unwrap();
        assert_eq!(state.created.len(), fail_at as usize);
        assert_eq!(state.dropped.len(), state.created.len());
        assert_eq!(state.used, [0, 0]);
    }
}

#[test]
fn a_failed_bound_check_releases_both_owners_without_dispatch() {
    let observation = Arc::new(Mutex::new(Observation::default()));
    let created = Arc::clone(&observation);
    let result = run_gate(
        deadline(),
        move |caller| Ok(new_compute(caller, Arc::clone(&created))),
        || Err(failure("bound")),
    );
    assert!(matches!(
        result,
        Err(RunError::Compute {
            operation: "bound",
            ..
        })
    ));
    let state = observation.lock().unwrap();
    assert_eq!(state.created.len(), 2);
    assert_eq!(state.dropped.len(), 2);
    assert_eq!(state.used, [0, 0]);
}

#[test]
fn a_compute_failure_propagates_after_both_owners_drop() {
    let observation = Arc::new(Mutex::new(Observation::default()));
    let created = Arc::clone(&observation);
    let result = run_gate(
        deadline(),
        move |caller| {
            let mut compute = new_compute(caller, Arc::clone(&created));
            Ok(move |request: Request| {
                if request.caller == 0 && request.job == 3 {
                    return Err(failure("compute"));
                }
                compute(request)
            })
        },
        || Ok(()),
    );
    assert!(matches!(
        result,
        Err(RunError::Compute {
            operation: "compute",
            ..
        })
    ));
    let state = observation.lock().unwrap();
    assert_eq!(state.dropped.len(), 2);
    assert_eq!(state.used, [3, JOBS_PER_CALLER as usize]);
}

fn corrupt_output(kind: u8, request: Request, mut output: Output) -> Output {
    if request.caller != 0 || request.job != 2 {
        return output;
    }
    match kind {
        0 => output.dtype = "U8",
        1 => output.shape = [4, 1],
        _ => output.values = [123.0; 4],
    }
    output
}

#[test]
fn the_ledger_rejects_wrong_dtype_shape_and_bytes() {
    let failures = [
        LedgerError::UnexpectedDtype {
            caller: 0,
            job: 2,
            observed: "U8",
        },
        LedgerError::UnexpectedShape {
            caller: 0,
            job: 2,
            observed: [4, 1],
        },
        LedgerError::OutputMismatch { caller: 0, job: 2 },
    ];
    for (kind, expected) in failures.into_iter().enumerate() {
        let result = run_gate(
            deadline(),
            move |_| {
                Ok(move |request: Request| {
                    Ok(corrupt_output(
                        kind as u8,
                        request,
                        expected_output(request.fixture),
                    ))
                })
            },
            || Ok(()),
        );
        assert!(matches!(result, Err(RunError::Ledger(error)) if error == expected));
    }
}

#[test]
fn an_expired_deadline_prevents_factory_calls() {
    let observation = Arc::new(Mutex::new(Observation::default()));
    let created = Arc::clone(&observation);
    let result = run_gate(
        Instant::now(),
        move |caller| Ok(new_compute(caller, Arc::clone(&created))),
        || panic!("the bound check must not run after deadline expiry"),
    );
    assert!(matches!(
        result,
        Err(RunError::Deadline {
            stage: "spawn gate owner"
        })
    ));
    assert!(observation.lock().unwrap().created.is_empty());
}

#[test]
fn an_initialization_panic_releases_other_initialized_owners() {
    for fail_at in [0, 1] {
        let observation = Arc::new(Mutex::new(Observation::default()));
        let created = Arc::clone(&observation);
        let result = run_gate(
            deadline(),
            move |caller| {
                assert_ne!(caller, fail_at, "planned initialization panic");
                Ok(new_compute(caller, Arc::clone(&created)))
            },
            || panic!("the bound check must not follow an initialization panic"),
        );
        assert!(matches!(
            result,
            Err(RunError::ThreadPanic {
                stage: "owner initialization"
            })
        ));
        let state = observation.lock().unwrap();
        assert_eq!(state.created.len(), fail_at as usize);
        assert_eq!(state.dropped.len(), state.created.len());
        assert_eq!(state.used, [0, 0]);
    }
}

#[test]
fn a_compute_panic_propagates_after_both_owners_drop() {
    let observation = Arc::new(Mutex::new(Observation::default()));
    let created = Arc::clone(&observation);
    let result = run_gate(
        deadline(),
        move |caller| {
            let mut compute = new_compute(caller, Arc::clone(&created));
            Ok(move |request: Request| {
                assert!(
                    request.caller != 0 || request.job != 2,
                    "planned compute panic"
                );
                compute(request)
            })
        },
        || Ok(()),
    );
    assert!(matches!(
        result,
        Err(RunError::ThreadPanic {
            stage: "gate owner"
        })
    ));
    assert_eq!(observation.lock().unwrap().dropped.len(), 2);
}
