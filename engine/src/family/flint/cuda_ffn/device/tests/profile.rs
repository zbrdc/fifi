use super::super::*;
use crate::kernels::cuda::TimingEvent;
use serde::Serialize;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    time::Instant,
};

#[derive(Clone, Copy)]
pub(crate) enum Phase {
    Other,
    Dense,
    Shared,
    Routed,
}

#[derive(Default, Serialize)]
pub(crate) struct Report {
    pub(crate) gemm_ms: [f64; 4],
    pub(crate) gemm_calls: [u64; 4],
    pub(crate) event_api_host_ns: u64,
    pub(crate) wall_ns: BTreeMap<&'static str, u64>,
    pub(crate) routed_bank_h2d: crate::expert_tier::stream_cuda_profile::Report,
}

impl Report {
    fn add_gemm(&mut self, phase: Phase, ms: f32, host_ns: u64) {
        assert!(ms.is_finite() && ms >= 0.0, "invalid CUDA timing interval");
        self.gemm_ms[phase as usize] += f64::from(ms);
        self.gemm_calls[phase as usize] += 1;
        self.event_api_host_ns += host_ns;
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Report>> = const { RefCell::new(None) };
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Other) };
}

pub(crate) struct Capture(crate::expert_tier::stream_cuda_profile::Capture);

impl Capture {
    pub(crate) fn start() -> Self {
        ACTIVE.with_borrow_mut(|value| {
            assert!(value.is_none(), "nested diagnostic capture");
            *value = Some(Report::default());
        });
        Self(crate::expert_tier::stream_cuda_profile::Capture::start())
    }

    pub(crate) fn finish(self) -> Report {
        let bank = self
            .0
            .finish()
            .expect("completed diagnostic bank transfers");
        let mut report = ACTIVE
            .with_borrow_mut(Option::take)
            .expect("active diagnostic capture");
        report.routed_bank_h2d = bank;
        report
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with_borrow_mut(|value| *value = None);
        PHASE.set(Phase::Other);
    }
}

pub(crate) struct PhaseGuard(Phase);

pub(crate) fn phase(value: Phase) -> PhaseGuard {
    PhaseGuard(PHASE.replace(value))
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        PHASE.set(self.0);
    }
}

pub(crate) struct Wall {
    name: &'static str,
    start: Option<Instant>,
}

pub(crate) fn wall(name: &'static str) -> Wall {
    Wall {
        name,
        start: ACTIVE
            .with_borrow(|value| value.is_some())
            .then(Instant::now),
    }
}

impl Drop for Wall {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let ns =
                u64::try_from(start.elapsed().as_nanos()).expect("diagnostic duration fits u64");
            ACTIVE.with_borrow_mut(|value| {
                if let Some(report) = value {
                    *report.wall_ns.entry(self.name).or_default() += ns;
                }
            });
        }
    }
}

pub(crate) struct Pair {
    begin: TimingEvent,
    end: TimingEvent,
    phase: Phase,
    host_ns: u64,
}

pub(crate) fn project_begin(
    ctx: &CudaContext,
    stream: &DeviceStream,
) -> Result<Option<Pair>, EngError> {
    if !ACTIVE.with_borrow(|value| value.is_some()) {
        return Ok(None);
    }
    let host = Instant::now();
    // Allocate both events before recording the first: no event-creation gap in the GPU interval.
    let begin = ctx.event_timing()?;
    let end = ctx.event_timing()?;
    stream.record_timing(&begin)?;
    Ok(Some(Pair {
        begin,
        end,
        phase: PHASE.get(),
        host_ns: host.elapsed().as_nanos() as u64,
    }))
}

pub(crate) fn project_end(stream: &DeviceStream, pair: &mut Option<Pair>) -> Result<(), EngError> {
    if let Some(pair) = pair {
        let host = Instant::now();
        stream.record_timing(&pair.end)?;
        pair.host_ns += host.elapsed().as_nanos() as u64;
    }
    Ok(())
}

// The caller's existing Nvfp4LaunchStatus::check synchronized an event AFTER this pair.
pub(crate) fn project_complete(pair: Option<Pair>) -> Result<(), EngError> {
    if let Some(pair) = pair {
        let host = Instant::now();
        let ms = pair.begin.elapsed_ms_since(&pair.end)?;
        let phase = pair.phase;
        let previous = pair.host_ns;
        drop(pair);
        let ns = previous + host.elapsed().as_nanos() as u64;
        ACTIVE.with_borrow_mut(|value| {
            value
                .as_mut()
                .expect("active capture")
                .add_gemm(phase, ms, ns)
        });
    }
    Ok(())
}

#[test]
fn phases_restore_and_counters_do_not_leak_between_captures() {
    let capture = Capture::start();
    {
        let _outer = phase(Phase::Routed);
        {
            let _inner = phase(Phase::Shared);
            assert!(matches!(PHASE.get(), Phase::Shared));
        }
        assert!(matches!(PHASE.get(), Phase::Routed));
        let _wall = wall("test");
        ACTIVE.with_borrow_mut(|r| r.as_mut().unwrap().add_gemm(PHASE.get(), 2.5, 10));
    }
    assert!(matches!(PHASE.get(), Phase::Other));
    let report = capture.finish();
    assert_eq!(report.gemm_calls, [0, 0, 0, 1]);
    assert_eq!(report.gemm_ms, [0.0, 0.0, 0.0, 2.5]);
    assert_eq!(report.event_api_host_ns, 10);
    assert!(report.wall_ns.contains_key("test"));
    assert_eq!(Capture::start().finish().gemm_calls, [0; 4]);
    assert!(wall("disabled").start.is_none());
}

#[test]
fn capture_reports_zero_bank_transfers_without_staging() {
    let report = serde_json::to_value(Capture::start().finish()).unwrap();
    assert_eq!(
        report["routed_bank_h2d"],
        serde_json::json!({
            "calls": 0,
            "dma_bytes": 0,
            "gpu_ms": 0.0,
            "event_api_host_ns": 0,
        })
    );
}

#[test]
fn invalid_intervals_and_nested_captures_are_rejected() {
    for ms in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(
            std::panic::catch_unwind(|| Report::default().add_gemm(Phase::Other, ms, 0)).is_err()
        );
    }
    let capture = Capture::start();
    assert!(std::panic::catch_unwind(Capture::start).is_err());
    assert_eq!(capture.finish().gemm_calls, [0; 4]);
    drop(Capture::start());
    assert_eq!(Capture::start().finish().gemm_calls, [0; 4]);
}
