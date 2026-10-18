use std::{
    rc::Rc,
    sync::{Arc, Mutex},
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

use super::{GateStats, run_gate};
use crate::fixtures::{JOBS_PER_CALLER, Output, Request, TOTAL_JOBS, expected_output};
use crate::protocol::ComputeFailure;

mod failures;

#[derive(Default)]
struct Observation {
    created: Vec<(u8, ThreadId)>,
    used: [usize; 2],
    dropped: Vec<ThreadId>,
}

struct Owner {
    caller: u8,
    thread: ThreadId,
    observation: Arc<Mutex<Observation>>,
}

impl Owner {
    fn compute(&self, request: Request) -> Result<Output, ComputeFailure> {
        assert_eq!(thread::current().id(), self.thread);
        assert_eq!(request.caller, self.caller);
        self.observation.lock().unwrap().used[self.caller as usize] += 1;
        Ok(expected_output(request.fixture))
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        assert_eq!(thread::current().id(), self.thread);
        self.observation.lock().unwrap().dropped.push(self.thread);
    }
}

fn new_compute(
    caller: u8,
    observation: Arc<Mutex<Observation>>,
) -> impl FnMut(Request) -> Result<Output, ComputeFailure> {
    let thread = thread::current().id();
    observation.lock().unwrap().created.push((caller, thread));
    let owner = Rc::new(Owner {
        caller,
        thread,
        observation,
    });
    move |request| owner.compute(request)
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn assert_complete(observation: &Observation, stats: &GateStats) {
    assert_eq!(stats.total_jobs, TOTAL_JOBS);
    assert_eq!(observation.created.len(), 2);
    assert_eq!(observation.created[0].0, 0);
    assert_eq!(observation.created[1].0, 1);
    assert_ne!(observation.created[0].1, observation.created[1].1);
    assert_eq!(observation.used, [JOBS_PER_CALLER as usize; 2]);
    assert_eq!(observation.dropped.len(), 2);
    for (index, (_, thread)) in observation.created.iter().enumerate() {
        assert_ne!(*thread, thread::current().id());
        assert_eq!(stats.owner_threads[index], *thread);
        assert!(observation.dropped.contains(thread));
    }
}

#[test]
fn two_threads_keep_their_own_compute_values_for_every_job() {
    let observation = Arc::new(Mutex::new(Observation::default()));
    let factory_observation = Arc::clone(&observation);
    let bound_observation = Arc::clone(&observation);
    let stats = run_gate(
        deadline(),
        move |caller| Ok(new_compute(caller, Arc::clone(&factory_observation))),
        move || {
            let state = bound_observation.lock().unwrap();
            assert_eq!(state.created.len(), 2);
            assert!(state.dropped.is_empty());
            assert_eq!(state.used, [0, 0]);
            Ok(())
        },
    )
    .unwrap();
    assert_complete(&observation.lock().unwrap(), &stats);
}
