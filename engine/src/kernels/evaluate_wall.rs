//! Bounded, thread-local qualification samples; never a GPU execution timer.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

const MAX_SAMPLES: usize = 32_768;
thread_local! {
    static SAMPLES: RefCell<Option<Samples>> = const { RefCell::new(None) };
}

pub(crate) struct Samples {
    pub(crate) ns: Vec<u64>,
    pub(crate) dropped: u64,
}

impl Samples {
    pub(crate) fn histogram(&self) -> [(Option<u64>, usize); 6] {
        let upper = [
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            u64::MAX,
        ];
        let mut counts = [0; 6];
        for &ns in &self.ns {
            counts[upper.partition_point(|&bound| ns > bound)] += 1;
        }
        std::array::from_fn(|i| ((i < 5).then_some(upper[i]), counts[i]))
    }

    pub(crate) fn percentiles(&self) -> [u64; 3] {
        if self.ns.is_empty() {
            return [0; 3];
        }
        let mut sorted = self.ns.clone();
        sorted.sort_unstable();
        [50, 90, 99].map(|p| sorted[(sorted.len() * p).div_ceil(100) - 1])
    }
}

pub(crate) struct Capture {
    // Finishing or dropping must affect the thread that started the capture.
    _thread: PhantomData<Rc<()>>,
}

impl Capture {
    pub(crate) fn start() -> Result<Self, &'static str> {
        SAMPLES.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err("evaluate wall capture already active on this thread");
            }
            *slot = Some(Samples {
                ns: Vec::with_capacity(MAX_SAMPLES),
                dropped: 0,
            });
            Ok(Self {
                _thread: PhantomData,
            })
        })
    }

    pub(crate) fn finish(self) -> Samples {
        SAMPLES.with(|slot| {
            slot.borrow_mut()
                .take()
                .expect("active evaluate wall capture")
        })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        SAMPLES.with(|slot| slot.borrow_mut().take());
    }
}

pub(super) fn record(ns: u64) {
    SAMPLES.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(samples) = slot.as_mut() {
            if samples.ns.len() == MAX_SAMPLES {
                samples.dropped = samples.dropped.saturating_add(1);
            } else {
                samples.ns.push(ns);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_records_only_its_scope_and_preserves_order() {
        record(7);
        let capture = Capture::start().unwrap();
        record(10);
        record(20);
        let sample = capture.finish();
        assert_eq!(sample.ns, [10, 20]);
        assert_eq!(sample.dropped, 0);
        record(30);
        assert!(Capture::start().unwrap().finish().ns.is_empty());
    }

    #[test]
    fn nested_capture_does_not_replace_the_outer_samples() {
        let capture = Capture::start().unwrap();
        record(10);
        assert!(Capture::start().is_err());
        record(20);
        assert_eq!(capture.finish().ns, [10, 20]);
    }

    #[test]
    fn early_drop_disables_capture() {
        let capture = Capture::start().unwrap();
        record(10);
        drop(capture);
        record(20);
        assert!(Capture::start().unwrap().finish().ns.is_empty());
    }

    #[test]
    fn samples_do_not_mix_engine_threads() {
        let capture = Capture::start().unwrap();
        record(10);
        let other = std::thread::spawn(|| {
            let capture = Capture::start().unwrap();
            record(20);
            capture.finish().ns
        });
        assert_eq!(other.join().unwrap(), [20]);
        assert_eq!(capture.finish().ns, [10]);
    }

    #[test]
    fn histogram_edges_are_inclusive_and_the_tail_is_open() {
        let sample = Samples {
            ns: vec![
                0,
                10_000,
                10_001,
                100_000,
                100_001,
                1_000_000,
                1_000_001,
                10_000_000,
                10_000_001,
                100_000_000,
                u64::MAX,
            ],
            dropped: 0,
        };
        assert_eq!(
            sample.histogram(),
            [
                (Some(10_000), 2),
                (Some(100_000), 2),
                (Some(1_000_000), 2),
                (Some(10_000_000), 2),
                (Some(100_000_000), 2),
                (None, 1)
            ]
        );
    }

    #[test]
    fn percentiles_use_nearest_rank_without_reordering_raw_samples() {
        let sample = Samples {
            ns: vec![40, 0, 10, 30, 20],
            dropped: 0,
        };
        assert_eq!(sample.percentiles(), [20, 40, 40]);
        assert_eq!(sample.ns, [40, 0, 10, 30, 20]);
        let empty = Samples {
            ns: vec![],
            dropped: 0,
        };
        assert_eq!(empty.percentiles(), [0; 3]);
        assert!(empty.histogram().iter().all(|(_, count)| *count == 0));
    }

    #[test]
    fn capacity_overflow_is_visible_without_growing_the_buffer() {
        let capture = Capture::start().unwrap();
        for ns in 0..=MAX_SAMPLES {
            record(ns as u64);
        }
        let sample = capture.finish();
        assert_eq!(sample.ns.len(), MAX_SAMPLES);
        assert_eq!(sample.ns[0], 0);
        assert_eq!(sample.ns[MAX_SAMPLES - 1], (MAX_SAMPLES - 1) as u64);
        assert_eq!(sample.dropped, 1);
    }
}
