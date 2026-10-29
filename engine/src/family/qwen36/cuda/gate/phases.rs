//! Bookkeeping of the timed region: the 128 decode steps and the three measured phases inside them.
//!
//! The clock is injected so the tests need no device and no wall time. Each phase is timed by its
//! own begin/end pair, so time spent outside every phase stays unaccounted and the evaluator's 3
//! percent check against `decode_ns` means something. Each step is also timed on its own, so the
//! first step can be reported beside the median and never dropped. The host analysis (top-32, hash,
//! lse) is NOT part of the timed region: it runs after `finish` and is reported as `analysis_ns`.
#![cfg_attr(not(test), expect(dead_code, reason = "staged for the gate loop"))]

use std::time::Instant;

pub(super) trait Clock {
    fn now_ns(&mut self) -> u64;
}

/// CLOCK_MONOTONIC through `Instant`.
pub(super) struct MonotonicClock(Instant);

impl MonotonicClock {
    pub(super) fn new() -> Self {
        Self(Instant::now())
    }
}

impl Clock for MonotonicClock {
    fn now_ns(&mut self) -> u64 {
        self.0.elapsed().as_nanos() as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Phase {
    Replay,
    Readback,
    Host,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Timing {
    pub decode_ns: u64,
    pub replay_ns: u64,
    pub readback_ns: u64,
    pub host_ns: u64,
    /// Duration of each decode step, first to last.
    pub step_ns: Vec<u64>,
}

pub(super) struct PhaseTimer<C: Clock> {
    clock: C,
    start: u64,
    spans: [u64; 3],
    open: Option<(Phase, u64)>,
    step_open: Option<u64>,
    steps: Vec<u64>,
}

impl<C: Clock> PhaseTimer<C> {
    pub(super) fn start(mut clock: C) -> Self {
        let start = clock.now_ns();
        Self {
            clock,
            start,
            spans: [0; 3],
            open: None,
            step_open: None,
            steps: Vec::new(),
        }
    }

    pub(super) fn begin(&mut self, phase: Phase) -> Result<(), &'static str> {
        if self.open.is_some() {
            return Err("a phase is already open");
        }
        self.open = Some((phase, self.clock.now_ns()));
        Ok(())
    }

    pub(super) fn end(&mut self, phase: Phase) -> Result<(), &'static str> {
        match self.open.take() {
            Some((open, at)) if open == phase => {
                self.spans[phase as usize] += self.clock.now_ns().saturating_sub(at);
                Ok(())
            }
            _ => Err("end without a matching begin"),
        }
    }

    pub(super) fn step_begin(&mut self) -> Result<(), &'static str> {
        if self.step_open.is_some() {
            return Err("a step is already open");
        }
        self.step_open = Some(self.clock.now_ns());
        Ok(())
    }

    pub(super) fn step_end(&mut self) -> Result<(), &'static str> {
        let at = self.step_open.take().ok_or("step_end without step_begin")?;
        self.steps.push(self.clock.now_ns().saturating_sub(at));
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Timing, &'static str> {
        if self.open.is_some() || self.step_open.is_some() {
            return Err("finish with an open phase or step");
        }
        let decode_ns = self.clock.now_ns().saturating_sub(self.start);
        let [replay_ns, readback_ns, host_ns] = self.spans;
        Ok(Timing {
            decode_ns,
            replay_ns,
            readback_ns,
            host_ns,
            step_ns: self.steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the scripted times in order.
    struct Script(Vec<u64>);

    impl Clock for Script {
        fn now_ns(&mut self) -> u64 {
            self.0.remove(0)
        }
    }

    #[test]
    fn spans_add_up_per_phase_and_the_gap_stays_unaccounted() {
        // start 100; replay 110..150 and 200..230; readback 150..160; host 160..170; finish at 300.
        let mut t = PhaseTimer::start(Script(vec![
            100, 110, 150, 150, 160, 160, 170, 200, 230, 300,
        ]));
        t.begin(Phase::Replay).unwrap();
        t.end(Phase::Replay).unwrap();
        t.begin(Phase::Readback).unwrap();
        t.end(Phase::Readback).unwrap();
        t.begin(Phase::Host).unwrap();
        t.end(Phase::Host).unwrap();
        t.begin(Phase::Replay).unwrap();
        t.end(Phase::Replay).unwrap();
        let got = t.finish().unwrap();
        assert_eq!(
            got,
            Timing {
                decode_ns: 200,
                replay_ns: 70,
                readback_ns: 10,
                host_ns: 10,
                step_ns: vec![]
            }
        );
        assert!(got.replay_ns + got.readback_ns + got.host_ns < got.decode_ns);
    }

    #[test]
    fn misuse_is_an_error_not_a_wrong_number() {
        let mut t = PhaseTimer::start(Script(vec![0, 1, 2, 3, 4]));
        assert!(t.end(Phase::Host).is_err());
        t.begin(Phase::Host).unwrap();
        assert!(t.begin(Phase::Replay).is_err());
        assert!(t.end(Phase::Replay).is_err());
        let mut t = PhaseTimer::start(Script(vec![0, 1, 9]));
        t.begin(Phase::Replay).unwrap();
        assert!(t.finish().is_err());
    }

    #[test]
    fn each_step_is_timed_on_its_own_and_the_first_is_kept() {
        let mut t = PhaseTimer::start(Script(vec![0, 10, 90, 100, 110, 400]));
        t.step_begin().unwrap();
        t.step_end().unwrap();
        t.step_begin().unwrap();
        t.step_end().unwrap();
        let got = t.finish().unwrap();
        assert_eq!(got.step_ns, vec![80, 10]);
        assert_eq!(got.decode_ns, 400);
    }

    #[test]
    fn step_misuse_is_an_error() {
        let mut t = PhaseTimer::start(Script(vec![0, 1, 2]));
        assert!(t.step_end().is_err());
        t.step_begin().unwrap();
        assert!(t.step_begin().is_err());
        assert!(t.finish().is_err());
    }

    #[test]
    fn the_monotonic_clock_never_goes_backwards() {
        let mut c = MonotonicClock::new();
        let (a, b) = (c.now_ns(), c.now_ns());
        assert!(b >= a);
    }
}
