//! M1 row-batched decode scheduling (the mistralrs-study design): decode-only
//! steps over N independent per-sequence states, never mixing prefill into the
//! decode forward. `max_seqs` defaults to 1 - a lone request must run the solo
//! path. The serving scheduler is not yet wired to the staged batch decoder.

use crate::EngError;
use std::collections::BTreeMap;

/// A slot's identity carried AS DATA with every batched row (the basalt
/// pattern): the batch forward stays one code path for any slot subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowHeader {
    pub slot: u32,
    pub pos: u32,
    /// True when the slot finished and its row must not be scheduled again.
    pub finished: bool,
}

/// Admission/cancel bookkeeping for decode slots. The scheduler owns which
/// slots join the next row window (<= 8, the basalt bound); it never touches
/// the model states themselves.
#[derive(Debug)]
pub struct BatchScheduler {
    max_seqs: usize,
    /// The decode window bound (basalt: <= 8 rows shared by all slots).
    window: usize,
    next_slot: u32,
    slots: BTreeMap<u32, SlotState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// Admitted, awaiting its next decode row.
    Active { pos: u32 },
    /// Finished; the row is not scheduled until reset.
    Finished,
}

impl BatchScheduler {
    /// `max_seqs` DEFAULTS TO 1 (the M1 gate: B=1 bit-identical to the solo
    /// path); the batch arms only when the caller raises it.
    pub fn new() -> Self {
        Self::with_max_seqs(1)
    }

    pub fn with_max_seqs(max_seqs: usize) -> Self {
        let max_seqs = max_seqs.clamp(1, 8);
        Self {
            max_seqs,
            window: max_seqs.min(8),
            next_slot: 0,
            slots: BTreeMap::new(),
        }
    }

    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// Admits a new sequence at position 0. Fails cleanly (no state change)
    /// when the slot budget is full.
    pub fn admit(&mut self) -> Result<u32, EngError> {
        if self
            .slots
            .values()
            .filter(|s| matches!(s, SlotState::Active { .. }))
            .count()
            >= self.max_seqs
        {
            return Err(EngError::BadRequest(
                "batch scheduler: the max_seqs budget is full".into(),
            ));
        }
        let slot = self.next_slot;
        self.next_slot = self.next_slot.wrapping_add(1);
        self.slots.insert(slot, SlotState::Active { pos: 0 });
        Ok(slot)
    }

    /// Cancels a slot outright (admission symmetry; the 1,000-cycle leak gate).
    pub fn cancel(&mut self, slot: u32) {
        self.slots.remove(&slot);
    }

    /// The next decode window: the row headers for the active slots, at most
    /// `window` of them, in slot order. An empty window means no work.
    pub fn next_window(&mut self) -> Vec<RowHeader> {
        self.slots
            .iter()
            .filter(|(_, s)| matches!(s, SlotState::Active { .. }))
            .take(self.window)
            .map(|(&slot, s)| RowHeader {
                slot,
                pos: match s {
                    SlotState::Active { pos } => *pos,
                    SlotState::Finished => 0,
                },
                finished: false,
            })
            .collect()
    }

    /// Commits one stepped row: the slot's position advances (decode-only - one
    /// position per step, prefill never rides this scheduler).
    pub fn commit_step(&mut self, slot: u32, finished: bool) {
        let Some(state) = self.slots.get_mut(&slot) else {
            return;
        };
        match state {
            SlotState::Active { pos } => {
                if finished {
                    *state = SlotState::Finished;
                } else {
                    *pos += 1;
                }
            }
            SlotState::Finished => {}
        }
    }

    /// Resets a finished slot for reuse (position back to 0).
    pub fn reset(&mut self, slot: u32) -> Result<(), EngError> {
        match self.slots.get(&slot) {
            Some(SlotState::Finished) => {}
            Some(SlotState::Active { .. }) => {
                return Err(EngError::BadRequest(
                    "batch scheduler: slot is still active".into(),
                ));
            }
            None => return Err(EngError::BadRequest("batch scheduler: unknown slot".into())),
        }
        if self.active() >= self.max_seqs {
            return Err(EngError::BadRequest(
                "batch scheduler: the max_seqs budget is full".into(),
            ));
        }
        self.slots.insert(slot, SlotState::Active { pos: 0 });
        Ok(())
    }

    /// The active (schedulable) slot count - the leak gate's probe.
    pub fn active(&self) -> usize {
        self.slots
            .values()
            .filter(|s| matches!(s, SlotState::Active { .. }))
            .count()
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Binds the scheduler's decode window to a per-row step callback: the
/// window's headers drive the calls in slot order, B=1 is exactly one call,
/// and finished rows are never stepped. Portable - the mock proves the
/// contract without any kernel.
pub struct BatchRunner<'a, F> {
    pub sched: &'a mut BatchScheduler,
    pub step: F,
}

impl<'a, F> BatchRunner<'a, F>
where
    F: FnMut(u32, u32) -> bool,
{
    /// Runs one decode step over the current window, calling `step(slot, pos)`
    /// per row (true = finished). Returns the stepped slots in order.
    pub fn run(&mut self) -> Vec<u32> {
        let headers = self.sched.next_window();
        let mut stepped = Vec::with_capacity(headers.len());
        for h in headers {
            let finished = (self.step)(h.slot, h.pos);
            self.sched.commit_step(h.slot, finished);
            stepped.push(h.slot);
        }
        stepped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_solo_and_bit_identical_by_construction() {
        let mut sched = BatchScheduler::new();
        assert_eq!(sched.max_seqs(), 1, "max_seqs defaults to 1 (the M1 gate)");
        let slot = sched.admit().unwrap();
        // the second admit refuses: a lone request never shares a window
        assert!(sched.admit().is_err());
        let window = sched.next_window();
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].slot, slot);
        assert_eq!(window[0].pos, 0);
    }

    #[test]
    fn the_window_is_capped_at_eight_and_ordered() {
        let mut sched = BatchScheduler::with_max_seqs(8);
        for _ in 0..8 {
            sched.admit().unwrap();
        }
        assert!(sched.admit().is_err(), "the slot budget is full at 8");
        let window = sched.next_window();
        assert_eq!(window.len(), 8);
        let slots: Vec<u32> = window.iter().map(|h| h.slot).collect();
        let mut sorted = slots.clone();
        sorted.sort();
        assert_eq!(slots, sorted, "slot order is stable");
    }

    #[test]
    fn decode_only_steps_advance_one_position() {
        let mut sched = BatchScheduler::with_max_seqs(2);
        let a = sched.admit().unwrap();
        sched.commit_step(a, false);
        let window = sched.next_window();
        assert_eq!(window[0].pos, 1, "one decode step = one position");
    }

    #[test]
    fn finish_removes_the_row_and_reset_restores_it() {
        let mut sched = BatchScheduler::new();
        let slot = sched.admit().unwrap();
        sched.commit_step(slot, true);
        assert!(sched.next_window().is_empty(), "finished rows do not batch");
        sched.reset(slot).unwrap();
        assert_eq!(sched.next_window()[0].pos, 0);
    }

    #[test]
    fn the_batch_runner_contract_holds() {
        let mut sched = BatchScheduler::with_max_seqs(4);
        let slots: Vec<u32> = (0..4).map(|_| sched.admit().unwrap()).collect();
        let mut calls = Vec::new();
        {
            let mut runner = BatchRunner {
                sched: &mut sched,
                step: |slot, pos| {
                    calls.push((slot, pos));
                    false
                },
            };
            let stepped = runner.run();
            assert_eq!(stepped, slots, "slot order is stable");
        }
        assert_eq!(calls.len(), 4, "one call per window row");
        assert_eq!(calls[0], (slots[0], 0), "positions come from the header");
        calls.clear();
        {
            let mut runner = BatchRunner {
                sched: &mut sched,
                step: |slot, pos| {
                    calls.push((slot, pos));
                    false
                },
            };
            runner.run();
        }
        assert_eq!(calls[0].1, 1, "decode steps advance one position");
        sched.commit_step(slots[3], true);
        let window = sched.next_window();
        assert_eq!(window.len(), 3, "the finished slot no longer batches");
        assert!(!window.iter().any(|h| h.slot == slots[3]));
        // B=1 (the default scheduler): exactly one call, the solo path shape
        let mut solo = BatchScheduler::new();
        let slot = solo.admit().unwrap();
        let mut n = 0;
        {
            let mut runner = BatchRunner {
                sched: &mut solo,
                step: |s, p| {
                    n += 1;
                    assert_eq!(s, slot);
                    assert_eq!(p, 0);
                    false
                },
            };
            runner.run();
        }
        assert_eq!(n, 1, "B=1 is exactly one step call - the solo shape");
    }

    #[test]
    fn a_thousand_admit_cancel_cycles_leave_no_leak() {
        let mut sched = BatchScheduler::with_max_seqs(8);
        for i in 0..1_000u32 {
            let slot = sched
                .admit()
                .expect("admit must succeed (canceled each cycle)");
            assert!(sched.active() >= 1);
            if i % 3 == 0 {
                sched.commit_step(slot, false);
            }
            sched.cancel(slot);
        }
        assert_eq!(sched.active(), 0, "no slot survives the cycles");
        assert!(sched.next_window().is_empty());
        // the scheduler remains fully usable afterwards
        let fresh = sched.admit().unwrap();
        assert_eq!(sched.next_window().len(), 1);
        assert_eq!(fresh, 1_000, "slot ids keep advancing without reuse");
    }
}

#[cfg(test)]
mod reset_capacity_tests {
    use super::*;

    /// Reactivating a finished slot is an admission (review MustFix 3): at
    /// capacity it must refuse and change nothing - the admitted slot keeps
    /// running; with capacity free the reset succeeds.
    #[test]
    fn reset_respects_the_admission_capacity() {
        let mut sched = BatchScheduler::with_max_seqs(1);
        let a = sched.admit().unwrap();
        sched.commit_step(a, true);
        let b = sched.admit().unwrap();
        // Capacity 1, slot b active: resetting slot a must refuse.
        let err = sched.reset(a).unwrap_err();
        assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
        assert_eq!(sched.active(), 1);
        assert_eq!(sched.next_window()[0].slot, b);
        sched.commit_step(b, false);
        assert!(sched.reset(b).is_err());
        assert_eq!(sched.next_window()[0].pos, 1);
        // With the capacity free again the reset succeeds.
        sched.commit_step(b, true);
        sched.reset(a).unwrap();
        assert_eq!(sched.active(), 1);
    }
}
