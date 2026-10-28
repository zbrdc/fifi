//! Test-only bank transfer timing, queried after the lease's existing completion wait.
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceStream, TimingEvent},
};
use serde::Serialize;
use std::{cell::RefCell, rc::Rc, time::Instant};

#[derive(Default, Serialize)]
pub(crate) struct Report {
    pub(crate) calls: u64,
    pub(crate) dma_bytes: u64,
    pub(crate) gpu_ms: f64,
    pub(crate) event_api_host_ns: u64,
}

impl Report {
    fn add(&mut self, bytes: u64, ms: f32, ns: u64) {
        assert!(
            ms.is_finite() && ms >= 0.0,
            "invalid CUDA bank timing interval"
        );
        self.calls += 1;
        self.dma_bytes += bytes;
        self.gpu_ms += f64::from(ms);
        self.event_api_host_ns += ns;
    }
}

#[derive(Default)]
struct State {
    inflight: u64,
    completed: Vec<Events>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

pub(crate) struct Capture(Rc<RefCell<State>>);

impl Capture {
    pub(crate) fn start() -> Self {
        ACTIVE.with_borrow_mut(|active| {
            assert!(active.is_none(), "nested bank capture");
            let state = Rc::new(RefCell::new(State::default()));
            *active = Some(Rc::clone(&state));
            Self(state)
        })
    }

    pub(crate) fn finish(&self) -> Result<Report, EngError> {
        let completed = {
            let mut state = self.0.borrow_mut();
            assert_eq!(state.inflight, 0, "bank lease must finish before capture");
            std::mem::take(&mut state.completed)
        };
        let mut report = Report::default();
        for events in completed {
            let host = Instant::now();
            let ms = events.begin.elapsed_ms_since(&events.end)?;
            let (bytes, previous) = (events.bytes, events.host_ns);
            drop(events);
            report.add(bytes, ms, previous + host.elapsed().as_nanos() as u64);
        }
        Ok(report)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with_borrow_mut(|active| *active = None);
    }
}

struct Events {
    begin: TimingEvent,
    end: TimingEvent,
    bytes: u64,
    host_ns: u64,
}

pub(crate) struct Pair {
    events: Option<Events>,
    state: Rc<RefCell<State>>,
}

pub(crate) fn begin(
    ctx: &CudaContext,
    stream: &DeviceStream,
    bytes: usize,
) -> Result<Option<Pair>, EngError> {
    let Some(state) = ACTIVE.with_borrow(Clone::clone) else {
        return Ok(None);
    };
    let host = Instant::now();
    let begin = ctx.event_timing()?;
    let end = ctx.event_timing()?;
    stream.record_timing(&begin)?;
    state.borrow_mut().inflight += 1;
    Ok(Some(Pair {
        events: Some(Events {
            begin,
            end,
            bytes: bytes as u64,
            host_ns: host.elapsed().as_nanos() as u64,
        }),
        state,
    }))
}

pub(crate) fn end(stream: &DeviceStream, pair: &mut Option<Pair>) -> Result<(), EngError> {
    if let Some(pair) = pair {
        let events = pair.events.as_mut().expect("uncompleted bank timing pair");
        let host = Instant::now();
        stream.record_timing(&events.end)?;
        events.host_ns += host.elapsed().as_nanos() as u64;
    }
    Ok(())
}

// Called only after the existing bank completion wait; no additional synchronization.
pub(crate) fn complete(pair: Option<Pair>) {
    if let Some(mut pair) = pair {
        let events = pair.events.take().expect("uncompleted bank timing pair");
        let mut state = pair.state.borrow_mut();
        state.inflight -= 1;
        state.completed.push(events);
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        if self.events.is_some() {
            self.state.borrow_mut().inflight -= 1;
        }
    }
}

#[test]
fn counters_reject_invalid_intervals_and_count_aligned_dma_once() {
    let mut report = Report::default();
    report.add(257, 1.5, 7);
    report.add(513, 0.0, 11);
    assert_eq!(
        (
            report.calls,
            report.dma_bytes,
            report.gpu_ms,
            report.event_api_host_ns
        ),
        (2, 770, 1.5, 18)
    );
    for ms in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(std::panic::catch_unwind(|| Report::default().add(1, ms, 0)).is_err());
    }
}

#[test]
fn capture_resets_and_rejects_unfinished_leases_without_querying_events() {
    let capture = Capture::start();
    assert!(std::panic::catch_unwind(Capture::start).is_err());
    capture.0.borrow_mut().inflight = 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| capture.finish())).is_err());
    capture.0.borrow_mut().inflight = 0;
    assert_eq!(capture.finish().unwrap().calls, 0);
    drop(capture);
    assert!(ACTIVE.with_borrow(Option::is_none));
    assert_eq!(Capture::start().finish().unwrap().dma_bytes, 0);
}
