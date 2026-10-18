use crate::fixtures::{
    CALLER_COUNT, JOBS_PER_CALLER, Output, Request, expected_output, fixture_for,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    UnknownCaller {
        caller: u8,
    },
    JobOutOfRange {
        caller: u8,
        job: u32,
    },
    IncorrectFixture {
        caller: u8,
        job: u32,
        observed: u8,
        expected: u8,
    },
    DuplicateRequest {
        caller: u8,
        job: u32,
    },
    DuplicateResponse {
        caller: u8,
        job: u32,
    },
    MissingRequest {
        caller: u8,
        job: u32,
    },
    MissingResponse {
        caller: u8,
        job: u32,
    },
    OutputMismatch {
        caller: u8,
        job: u32,
    },
    UnexpectedDtype {
        caller: u8,
        job: u32,
        observed: &'static str,
    },
    UnexpectedShape {
        caller: u8,
        job: u32,
        observed: [u32; 2],
    },
}

pub struct Ledger {
    requests: Vec<bool>,
    responses: Vec<bool>,
    caller_count: u8,
    jobs_per_caller: u32,
}

impl Ledger {
    pub fn new(caller_count: u8, jobs_per_caller: u32) -> Self {
        let len = caller_count as usize * jobs_per_caller as usize;
        Self {
            requests: vec![false; len],
            responses: vec![false; len],
            caller_count,
            jobs_per_caller,
        }
    }

    pub fn record_request(&mut self, request: Request) -> Result<(), LedgerError> {
        let index = self.expected_index(request)?;
        let slot = &mut self.requests[index];
        if *slot {
            return Err(LedgerError::DuplicateRequest {
                caller: request.caller,
                job: request.job,
            });
        }
        *slot = true;
        Ok(())
    }

    pub fn record_response(&mut self, request: Request, output: Output) -> Result<(), LedgerError> {
        let index = self.expected_index(request)?;
        let expected = expected_output(request.fixture);
        if output.dtype != expected.dtype {
            return Err(LedgerError::UnexpectedDtype {
                caller: request.caller,
                job: request.job,
                observed: output.dtype,
            });
        }
        if output.shape != expected.shape {
            return Err(LedgerError::UnexpectedShape {
                caller: request.caller,
                job: request.job,
                observed: output.shape,
            });
        }
        if !output.same_bytes(&expected) {
            return Err(LedgerError::OutputMismatch {
                caller: request.caller,
                job: request.job,
            });
        }
        let slot = &mut self.responses[index];
        if *slot {
            return Err(LedgerError::DuplicateResponse {
                caller: request.caller,
                job: request.job,
            });
        }
        *slot = true;
        Ok(())
    }

    pub fn finish(&self) -> Result<(), LedgerError> {
        for caller in 0..self.caller_count {
            for job in 0..self.jobs_per_caller {
                self.slot(&self.requests, caller, job)
                    .then_some(())
                    .ok_or(LedgerError::MissingRequest { caller, job })?;
                self.slot(&self.responses, caller, job)
                    .then_some(())
                    .ok_or(LedgerError::MissingResponse { caller, job })?;
            }
        }
        Ok(())
    }

    pub fn recorded_requests(&self) -> usize {
        self.requests.iter().filter(|seen| **seen).count()
    }

    pub fn recorded_responses(&self) -> usize {
        self.responses.iter().filter(|seen| **seen).count()
    }

    fn expected_index(&self, request: Request) -> Result<usize, LedgerError> {
        if request.caller >= self.caller_count {
            return Err(LedgerError::UnknownCaller {
                caller: request.caller,
            });
        }
        if request.job >= self.jobs_per_caller {
            return Err(LedgerError::JobOutOfRange {
                caller: request.caller,
                job: request.job,
            });
        }
        let expected_fixture = fixture_for(request.caller, request.job);
        if request.fixture != expected_fixture {
            return Err(LedgerError::IncorrectFixture {
                caller: request.caller,
                job: request.job,
                observed: request.fixture,
                expected: expected_fixture,
            });
        }
        Ok(self.index(request.caller, request.job))
    }

    fn index(&self, caller: u8, job: u32) -> usize {
        caller as usize * self.jobs_per_caller as usize + job as usize
    }

    fn slot(&self, seen: &[bool], caller: u8, job: u32) -> bool {
        seen[self.index(caller, job)]
    }
}

const _: () = {
    assert!(CALLER_COUNT as u32 * JOBS_PER_CALLER == 10_000);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        CALLER_COUNT, FIXTURE_A_OUTPUT, JOBS_PER_CALLER, Request, expected_output, fixture_for,
    };

    fn request(caller: u8, job: u32) -> Request {
        Request {
            caller,
            job,
            fixture: fixture_for(caller, job),
        }
    }

    #[test]
    fn accepts_each_expected_identifier_exactly_once() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        for caller in 0..CALLER_COUNT {
            for job in 0..2 {
                let req = request(caller, job);
                ledger.record_request(req).unwrap();
                ledger
                    .record_response(req, expected_output(req.fixture))
                    .unwrap();
            }
        }
        ledger.finish().unwrap();
    }

    #[test]
    fn rejects_a_duplicate_request() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        let req = request(0, 0);
        ledger.record_request(req).unwrap();
        assert_eq!(
            ledger.record_request(req),
            Err(LedgerError::DuplicateRequest { caller: 0, job: 0 })
        );
    }

    #[test]
    fn rejects_a_duplicate_response() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        let req = request(1, 1);
        ledger.record_request(req).unwrap();
        ledger
            .record_response(req, expected_output(req.fixture))
            .unwrap();
        assert_eq!(
            ledger.record_response(req, expected_output(req.fixture)),
            Err(LedgerError::DuplicateResponse { caller: 1, job: 1 })
        );
    }

    #[test]
    fn rejects_unknown_callers_and_out_of_range_jobs() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        assert_eq!(
            ledger.record_request(request(2, 0)),
            Err(LedgerError::UnknownCaller { caller: 2 })
        );
        assert_eq!(
            ledger.record_request(request(0, 2)),
            Err(LedgerError::JobOutOfRange { caller: 0, job: 2 })
        );
    }

    #[test]
    fn rejects_an_incorrect_fixture_identifier() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        let wrong = Request {
            caller: 0,
            job: 0,
            fixture: 1,
        };
        assert_eq!(
            ledger.record_request(wrong),
            Err(LedgerError::IncorrectFixture {
                caller: 0,
                job: 0,
                observed: 1,
                expected: 0
            })
        );
    }

    #[test]
    fn rejects_byte_different_output_with_equal_numeric_value() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        let req = request(1, 0);
        ledger.record_request(req).unwrap();
        let neg_zero = Output {
            values: [0.0, -0.0, 0.0, 0.0],
            ..expected_output(req.fixture)
        };
        assert_eq!(
            ledger.record_response(req, neg_zero),
            Err(LedgerError::OutputMismatch { caller: 1, job: 0 })
        );
    }

    #[test]
    fn rejects_mismatched_output_dtype_shape_and_bytes() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        let req = request(0, 0);
        ledger.record_request(req).unwrap();
        let expected = expected_output(req.fixture);
        assert_eq!(
            ledger.record_response(
                req,
                Output {
                    dtype: "F16",
                    ..expected
                }
            ),
            Err(LedgerError::UnexpectedDtype {
                caller: 0,
                job: 0,
                observed: "F16"
            })
        );
        assert_eq!(
            ledger.record_response(
                req,
                Output {
                    shape: [2, 3],
                    ..expected
                }
            ),
            Err(LedgerError::UnexpectedShape {
                caller: 0,
                job: 0,
                observed: [2, 3]
            })
        );
        assert_eq!(
            ledger.record_response(
                req,
                Output {
                    values: [1.0, 2.0, 3.0, 4.0],
                    ..expected
                }
            ),
            Err(LedgerError::OutputMismatch { caller: 0, job: 0 })
        );
    }

    #[test]
    fn a_smaller_ledger_rejects_callers_outside_its_construction_bound() {
        let mut ledger = Ledger::new(1, 2);
        for job in 0..2 {
            let req = request(0, job);
            ledger.record_request(req).unwrap();
            ledger
                .record_response(req, expected_output(req.fixture))
                .unwrap();
        }
        assert_eq!(
            ledger.record_request(request(1, 0)),
            Err(LedgerError::UnknownCaller { caller: 1 })
        );
        ledger.finish().unwrap();
    }

    #[test]
    fn finish_reports_missing_requests_and_responses() {
        let mut ledger = Ledger::new(CALLER_COUNT, 2);
        assert_eq!(
            ledger.finish(),
            Err(LedgerError::MissingRequest { caller: 0, job: 0 })
        );
        let req = request(0, 0);
        ledger.record_request(req).unwrap();
        ledger
            .record_response(req, expected_output(req.fixture))
            .unwrap();
        assert_eq!(
            ledger.finish(),
            Err(LedgerError::MissingRequest { caller: 0, job: 1 })
        );
    }

    #[test]
    fn preallocates_the_full_identifier_space() {
        let mut ledger = Ledger::new(CALLER_COUNT, JOBS_PER_CALLER);
        for caller in 0..CALLER_COUNT {
            for job in 0..JOBS_PER_CALLER {
                let req = request(caller, job);
                ledger.record_request(req).unwrap();
                ledger
                    .record_response(req, expected_output(req.fixture))
                    .unwrap();
            }
        }
        ledger.finish().unwrap();
        assert_eq!(ledger.recorded_requests(), 10_000);
        assert_eq!(ledger.recorded_responses(), 10_000);
        let _ = FIXTURE_A_OUTPUT;
    }
}
