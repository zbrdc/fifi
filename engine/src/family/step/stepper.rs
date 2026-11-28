//! Shared [`TokenStepper`] adapter over a resident device step.
//!
//! Tracks the consumed sequence position, the cache generation (bumped when a new sequence writes
//! at position 0 or after a full failure reset), and whether a step failed since the last full
//! clear. Greedy single-token steps pick on the device when the model has a resident embedding
//! table and the sampler does not need host logits; all other steps read logits back to the host
//! sampler. Host-memory denials (`EngError::MemoryPlan`) during `keep` or `put_back` leave device
//! state intact and do not latch failure.

use crate::EngError;
use crate::family::sampler::Sampler;
use crate::family::session::{Kept, TokenStepper, sample_logits};

/// Model-specific device operations driven by [`DeviceStepper`].
pub(crate) trait DeviceStep {
    /// Operation label attached to backend and sampling errors.
    const OPERATION: &'static str;

    /// Feeds `tokens` starting at `pos` without running the output head.
    fn feed_rows(&mut self, tokens: &[u32], pos: usize) -> Result<(), EngError>;

    /// Feeds one `token` at `pos` and returns the host logits row.
    fn logits(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngError>;

    /// Whether [`Self::decode_token`] is available for on-device greedy picks.
    fn takes_tokens(&self) -> bool;

    /// Runs one greedy decode step on the device at `pos` over `sampler_vocab` ids.
    fn decode_token(
        &mut self,
        token: u32,
        pos: usize,
        sampler_vocab: usize,
    ) -> Result<u32, EngError>;

    /// Full clear after a failed step: zeroes both running state and position-addressed cache.
    fn reset(&mut self) -> Result<(), EngError>;

    /// Clears only the running state so position-addressed cache rows stay for prefix reuse.
    fn reset_running(&mut self) -> Result<(), EngError>;

    /// Byte size of one kept running-state snapshot.
    fn kept_bytes(&self) -> u64;

    /// Snapshots the running state after `rows` positions.
    fn keep(&self, rows: usize) -> Result<Kept, EngError>;

    /// Restores a previously kept running-state snapshot.
    fn put_back(&mut self, kept: &Kept) -> Result<(), EngError>;
}

/// Shared [`TokenStepper`] over any [`DeviceStep`].
pub(crate) struct DeviceStepper<D> {
    device: D,
    pos: usize,
    /// Counts sequences that started at position 0 and full clears after a failure.
    generation: u64,
    /// A step failed since the last full clear: what the cache planes hold is unknown.
    failed: bool,
}

impl<D> DeviceStepper<D> {
    pub(crate) fn new(device: D) -> Self {
        Self {
            device,
            pos: 0,
            generation: 0,
            failed: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn device(&self) -> &D {
        &self.device
    }

    /// Before a step at the stepper's position: a step at position 0 starts a sequence that
    /// writes over the rows of the one before.
    fn begin(&mut self) {
        if self.pos == 0 {
            self.generation += 1;
        }
    }

    /// Notes a failed step, so the next reset clears everything the device carries.
    fn checked<T>(&mut self, result: Result<T, EngError>) -> Result<T, EngError> {
        self.failed |= result.is_err();
        result
    }
}

impl<D: DeviceStep> TokenStepper for DeviceStepper<D> {
    const OPERATION: &'static str = D::OPERATION;

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        self.feed_rows(&[token])
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        self.begin();
        let fed = self.device.feed_rows(tokens, self.pos);
        self.checked(fed)?;
        self.pos += tokens.len();
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.begin();
        let row = self.device.logits(token, self.pos);
        let row = self.checked(row)?;
        self.pos += 1;
        Ok(row)
    }

    /// A greedy pick is made on the device when the step holds the embedding table: the token id
    /// goes in, the id of the largest logit comes out, and no row crosses the bus. Any other
    /// sampling, and a request that captures its candidates, takes the logits row to the host; so
    /// does the first token of a request, whose margin over the runner-up the sampler keeps for
    /// the request's report.
    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let on_device = self.device.takes_tokens()
            && sampler.is_greedy()
            && !sampler.capture_armed()
            && sampler.first_top2_margin().is_some();
        if !on_device {
            let logits = self.logits(token)?;
            let drawn = sample_logits(Self::OPERATION, &logits, sampler, sampler_vocab);
            return self.checked(drawn);
        }
        self.begin();
        let picked = self.device.decode_token(token, self.pos, sampler_vocab);
        let picked = self.checked(picked)?;
        self.pos += 1;
        sampler.record_device_greedy(picked);
        Ok(picked)
    }

    fn fed(&self) -> usize {
        self.pos
    }

    /// A new sequence: the running state returns to zeros and the cache rows stay, for a later
    /// request that returns to a kept state. After a failed step everything is cleared and no kept
    /// state matches any more.
    fn reset(&mut self) -> Result<(), EngError> {
        if self.failed {
            self.device.reset()?;
            self.generation += 1;
            self.failed = false;
        } else {
            self.device.reset_running()?;
        }
        self.pos = 0;
        Ok(())
    }

    fn kept_bytes(&self) -> Option<u64> {
        Some(self.device.kept_bytes())
    }

    fn keep(&mut self) -> Result<Kept, EngError> {
        let kept = self.device.keep(self.pos);
        // Host-memory denial during a D2H snapshot leaves device state untouched.
        if !matches!(kept, Err(EngError::MemoryPlan(_))) {
            return self.checked(kept);
        }
        kept
    }

    fn put_back(&mut self, kept: &Kept) -> Result<(), EngError> {
        let put = self.device.put_back(kept);
        if !matches!(put, Err(EngError::MemoryPlan(_))) {
            self.checked(put)?;
        } else {
            put?;
        }
        self.pos = kept.rows;
        Ok(())
    }

    fn kept_generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::sampler::{Sampler, SamplingParams};

    #[derive(Default)]
    struct FakeDevice {
        takes_tokens: bool,
        full_resets: usize,
        running_resets: usize,
        device_picks: usize,
        host_logits: usize,
        fail_keep: Option<EngError>,
        fail_put_back: Option<EngError>,
        fail_feed: bool,
    }

    impl DeviceStep for FakeDevice {
        const OPERATION: &'static str = "fake_device";

        fn feed_rows(&mut self, _tokens: &[u32], _pos: usize) -> Result<(), EngError> {
            if self.fail_feed {
                return Err(EngError::Backend {
                    operation: Self::OPERATION,
                    message: "injected feed error".into(),
                });
            }
            Ok(())
        }

        fn logits(&mut self, _token: u32, _pos: usize) -> Result<Vec<f32>, EngError> {
            self.host_logits += 1;
            Ok(vec![0.0, 2.0, 1.0])
        }

        fn takes_tokens(&self) -> bool {
            self.takes_tokens
        }

        fn decode_token(
            &mut self,
            _token: u32,
            _pos: usize,
            _sampler_vocab: usize,
        ) -> Result<u32, EngError> {
            self.device_picks += 1;
            Ok(1)
        }

        fn reset(&mut self) -> Result<(), EngError> {
            self.full_resets += 1;
            Ok(())
        }

        fn reset_running(&mut self) -> Result<(), EngError> {
            self.running_resets += 1;
            Ok(())
        }

        fn kept_bytes(&self) -> u64 {
            16
        }

        fn keep(&self, rows: usize) -> Result<Kept, EngError> {
            if let Some(err) = self.fail_keep.take() {
                return Err(err);
            }
            Ok(Kept {
                rows,
                parts: vec![vec![0u8; 16]],
            })
        }

        fn put_back(&mut self, _kept: &Kept) -> Result<(), EngError> {
            if let Some(err) = self.fail_put_back.take() {
                return Err(err);
            }
            Ok(())
        }
    }

    #[test]
    fn memory_plan_on_keep_or_put_back_does_not_latch_failure() {
        let mut step = DeviceStepper::new(FakeDevice::default());
        step.feed_rows(&[1, 2, 3]).unwrap();
        assert_eq!(step.kept_generation(), 1);
        assert_eq!(step.fed(), 3);

        step.device.fail_keep = Some(EngError::MemoryPlan("oom".into()));
        assert!(matches!(step.keep(), Err(EngError::MemoryPlan(_))));
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 0);
        assert_eq!(step.device().running_resets, 1);
        assert_eq!(step.kept_generation(), 1);

        let kept = Kept {
            rows: 2,
            parts: vec![vec![0u8; 16]],
        };
        step.device.fail_put_back = Some(EngError::MemoryPlan("oom".into()));
        assert!(matches!(step.put_back(&kept), Err(EngError::MemoryPlan(_))));
        assert_eq!(step.fed(), 0);
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 0);
        assert_eq!(step.device().running_resets, 2);

        step.put_back(&kept).unwrap();
        assert_eq!(step.fed(), 2);
    }

    #[test]
    fn backend_failure_latches_full_reset_and_bumps_generation() {
        let mut step = DeviceStepper::new(FakeDevice {
            fail_feed: true,
            ..FakeDevice::default()
        });
        assert!(step.feed_rows(&[1]).is_err());
        let gen_before = step.kept_generation();
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 1);
        assert_eq!(step.kept_generation(), gen_before + 1);
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 1);
        assert_eq!(step.device().running_resets, 1);
    }

    #[test]
    fn next_token_uses_device_pick_only_after_first_margin_on_greedy() {
        let mut step = DeviceStepper::new(FakeDevice {
            takes_tokens: true,
            ..FakeDevice::default()
        });
        let mut greedy = Sampler::greedy();
        assert_eq!(step.next_token(0, &mut greedy, 3).unwrap(), 1);
        assert_eq!(step.device().host_logits, 1);
        assert_eq!(step.device().device_picks, 0);

        assert_eq!(step.next_token(1, &mut greedy, 3).unwrap(), 1);
        assert_eq!(step.device().host_logits, 1);
        assert_eq!(step.device().device_picks, 1);

        let mut sampled = Sampler::new(SamplingParams::new(0.8, 1.0, 0).unwrap(), 7);
        step.next_token(1, &mut sampled, 3).unwrap();
        step.next_token(1, &mut sampled, 3).unwrap();
        assert_eq!(step.device().device_picks, 1);
        assert_eq!(step.device().host_logits, 3);
    }
}
