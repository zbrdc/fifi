//! Shared [`TokenStepper`] adapter over a resident device step.
//!
//! Tracks the consumed sequence position, the cache generation (bumped when a new sequence writes
//! at position 0 or after a full failure reset), and whether a step failed since the last full
//! clear. Greedy single-token steps pick on the device when the model has a resident embedding
//! table and the sampler does not need host logits; non-greedy requests with `top_k <= 20` or open
//! vocabulary draw via the host-equal device shortlist and softmax-mass certificate; other steps
//! read logits back to the host sampler. Host-memory denials (`EngError::MemoryPlan`) during
//! `keep` or `put_back` leave device state intact and do not latch failure.

use crate::EngError;
use crate::family::sampler::Sampler;
use crate::family::session::{Kept, TokenStepper, sample_logits};
use crate::family::step::tail::{can_shortlist_sample, read_logits_row, sample_shortlist};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// Device buffers of a single-token step for logits readback and host-equal shortlist sampling.
pub(crate) struct ShortlistIo<'a> {
    pub(crate) ctx: &'a CudaContext,
    pub(crate) stream: &'a DeviceStream,
    pub(crate) logits: &'a DeviceBuffer,
    pub(crate) scratch: &'a DeviceBuffer,
    pub(crate) vocab: usize,
}

/// Model-specific device operations driven by [`DeviceStepper`].
pub(crate) trait DeviceStep {
    /// Operation label attached to backend and sampling errors.
    const OPERATION: &'static str;

    /// Feeds `tokens` starting at `pos` without running the output head.
    fn feed_rows(&mut self, tokens: &[u32], pos: usize) -> Result<(), EngError>;

    /// Runs one decode step at `pos`, leaving the logits row on the device.
    fn decode_to_logits(&mut self, _token: u32, _pos: usize) -> Result<(), EngError> {
        Err(EngError::BadRequest(
            "this device step does not implement decode_to_logits".into(),
        ))
    }

    /// Device context, stream, logits row and shortlist scratch of this step, when resident.
    fn shortlist_io(&self) -> Option<ShortlistIo<'_>> {
        None
    }

    /// Feeds one `token` at `pos` and returns the host logits row.
    fn logits(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, EngError> {
        self.decode_to_logits(token, pos)?;
        let io = self
            .shortlist_io()
            .ok_or_else(|| EngError::BadRequest("this device step has no logits buffer".into()))?;
        read_logits_row(io.stream, io.logits, io.vocab)
    }

    /// Whether [`Self::decode_token`] is available for on-device greedy picks.
    fn takes_tokens(&self) -> bool;

    /// Runs one greedy decode step on the device at `pos` over `sampler_vocab` ids.
    fn decode_token(
        &mut self,
        token: u32,
        pos: usize,
        sampler_vocab: usize,
    ) -> Result<u32, EngError>;

    /// Whether [`Self::sample_from_logits`] can draw non-greedy requests on the device via the
    /// host-equal shortlist and softmax-mass certificate.
    fn takes_shortlist(&self) -> bool {
        self.shortlist_io().is_some()
    }

    /// Draws one token from the logits row left on the device by [`Self::decode_to_logits`] via
    /// `sampler`'s host-equal shortlist or open-vocabulary certificate path.
    fn sample_from_logits(
        &self,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let io = self.shortlist_io().ok_or_else(|| {
            EngError::BadRequest("this device step has no shortlist scratch".into())
        })?;
        if sampler_vocab == 0 || sampler_vocab > io.vocab {
            return Err(EngError::Backend {
                operation: Self::OPERATION,
                message: format!(
                    "logits row has {} entries, need at least {sampler_vocab}",
                    io.vocab
                ),
            });
        }
        sample_shortlist(
            io.ctx,
            io.stream,
            io.logits,
            io.scratch,
            sampler,
            sampler_vocab,
        )
    }

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
    pub(crate) fn over(device: D) -> Self {
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

    /// A greedy pick is made on the device when the step holds the embedding table (after the
    /// first token records its top-2 margin). Non-greedy requests with `top_k` in `1..=20` or
    /// open-vocabulary `top_k == 0` draw via the host-equal device shortlist (+ softmax-mass
    /// certificate), reading only 176–240 bytes unless an open draw is uncertified. Requests with
    /// capture armed or `top_k > 20` take the full logits row to the host sampler.
    fn next_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<u32, EngError> {
        let greedy_on_device = self.device.takes_tokens()
            && sampler.is_greedy()
            && !sampler.capture_armed()
            && sampler.first_top2_margin().is_some();
        if greedy_on_device {
            self.begin();
            let picked = self.device.decode_token(token, self.pos, sampler_vocab);
            let picked = self.checked(picked)?;
            self.pos += 1;
            sampler.record_device_greedy(picked);
            return Ok(picked);
        }
        if self.device.takes_shortlist() && can_shortlist_sample(sampler) {
            self.begin();
            let decoded = self.device.decode_to_logits(token, self.pos);
            self.checked(decoded)?;
            self.pos += 1;
            let picked = self.device.sample_from_logits(sampler, sampler_vocab);
            return self.checked(picked);
        }
        let logits = self.logits(token)?;
        let drawn = sample_logits(Self::OPERATION, &logits, sampler, sampler_vocab);
        self.checked(drawn)
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
            let cleared = self.device.reset_running();
            self.checked(cleared)?;
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
    use std::cell::Cell;

    use super::*;
    use crate::family::sampler::{Sampler, SamplingParams};

    #[derive(Default)]
    struct FakeDevice {
        takes_tokens: bool,
        takes_shortlist: bool,
        full_resets: usize,
        running_resets: usize,
        device_picks: usize,
        host_logits: usize,
        fail_keep: Cell<Option<EngError>>,
        fail_put_back: Option<EngError>,
        fail_feed: bool,
        fail_decode: bool,
        fail_reset_running: bool,
        fail_sample_token: bool,
        shortlist_decodes: usize,
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

        fn decode_to_logits(&mut self, _token: u32, _pos: usize) -> Result<(), EngError> {
            if self.fail_decode {
                return Err(EngError::Backend {
                    operation: Self::OPERATION,
                    message: "injected decode error".into(),
                });
            }
            self.shortlist_decodes += 1;
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

        fn takes_shortlist(&self) -> bool {
            self.takes_shortlist
        }

        fn sample_from_logits(
            &self,
            sampler: &mut Sampler,
            sampler_vocab: usize,
        ) -> Result<u32, EngError> {
            if self.fail_sample_token {
                return Err(EngError::Backend {
                    operation: Self::OPERATION,
                    message: "injected sample_from_logits error".into(),
                });
            }
            let ranked = [(1u32, 2.0f32), (2, 1.0), (0, 0.0)];
            if sampler.open_draw_on_device() {
                let top = 2.0_f64;
                let t = sampler.params().temperature();
                let mass: f64 = ranked
                    .iter()
                    .map(|&(_, l)| ((f64::from(l) - top) / t).exp())
                    .sum();
                sampler
                    .sample_open_shortlist(&ranked, sampler_vocab, mass, || Ok(vec![0.0, 2.0, 1.0]))
            } else {
                sampler.sample_shortlist(&ranked, sampler_vocab)
            }
        }

        fn reset(&mut self) -> Result<(), EngError> {
            self.full_resets += 1;
            Ok(())
        }

        fn reset_running(&mut self) -> Result<(), EngError> {
            if self.fail_reset_running {
                self.fail_reset_running = false;
                return Err(EngError::Backend {
                    operation: Self::OPERATION,
                    message: "injected reset_running error".into(),
                });
            }
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
        let mut step = DeviceStepper::over(FakeDevice::default());
        step.feed_rows(&[1, 2, 3]).unwrap();
        assert_eq!(step.kept_generation(), 1);
        assert_eq!(step.fed(), 3);

        step.device
            .fail_keep
            .set(Some(EngError::MemoryPlan("oom".into())));
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
    fn backend_errors_on_keep_put_back_and_reset_running_latch_full_reset() {
        let mut step = DeviceStepper::over(FakeDevice::default());
        step.feed_rows(&[1, 2]).unwrap();
        let gen0 = step.kept_generation();

        step.device.fail_keep.set(Some(EngError::Backend {
            operation: "fake_device",
            message: "d2h failed".into(),
        }));
        assert!(step.keep().is_err());
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 1);
        assert_eq!(step.kept_generation(), gen0 + 1);

        let kept = Kept {
            rows: 2,
            parts: vec![vec![0u8; 16]],
        };
        step.device.fail_put_back = Some(EngError::Backend {
            operation: "fake_device",
            message: "h2d failed".into(),
        });
        assert!(step.put_back(&kept).is_err());
        assert_eq!(step.fed(), 0);
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 2);
        assert_eq!(step.kept_generation(), gen0 + 2);

        step.device.fail_reset_running = true;
        assert!(step.reset().is_err());
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 3);
        assert_eq!(step.kept_generation(), gen0 + 3);
    }

    #[test]
    fn backend_failure_latches_full_reset_and_bumps_generation() {
        let mut step = DeviceStepper::over(FakeDevice {
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
    fn next_token_uses_device_pick_only_after_first_margin_on_greedy_without_capture() {
        let mut step = DeviceStepper::over(FakeDevice {
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

        let _cap = greedy.arm_capture(4);
        assert_eq!(step.next_token(1, &mut greedy, 3).unwrap(), 1);
        assert_eq!(step.device().host_logits, 2);
        assert_eq!(step.device().device_picks, 1);

        let mut sampled = Sampler::new(SamplingParams::new(0.8, 1.0, 0).unwrap(), 7);
        step.next_token(1, &mut sampled, 3).unwrap();
        step.next_token(1, &mut sampled, 3).unwrap();
        assert_eq!(step.device().device_picks, 1);
        assert_eq!(step.device().host_logits, 4);
    }

    #[test]
    fn shortlist_routes_bounded_and_open_sampling_and_falls_back_otherwise() {
        let mut step = DeviceStepper::over(FakeDevice {
            takes_shortlist: true,
            ..FakeDevice::default()
        });
        // Bounded top_k in 1..=20 takes shortlist even on token 0 (and latches margin).
        let mut bounded = Sampler::new(SamplingParams::new(0.8, 0.9, 20).unwrap(), 42);
        let mut bounded_ref = Sampler::new(SamplingParams::new(0.8, 0.9, 20).unwrap(), 42);
        let got = step.next_token(0, &mut bounded, 3).unwrap();
        let want = bounded_ref.sample(&[0.0, 2.0, 1.0]).unwrap();
        assert_eq!(got, want);
        assert_eq!(bounded.rng_state(), bounded_ref.rng_state());
        assert_eq!(bounded.first_top2_margin(), bounded_ref.first_top2_margin());
        assert_eq!(step.device().shortlist_decodes, 1);
        assert_eq!(step.device().host_logits, 0);

        // Open-vocabulary top_k == 0 takes shortlist + mass certificate even on token 0.
        let mut open = Sampler::new(SamplingParams::new(0.8, 0.95, 0).unwrap(), 7);
        let mut open_ref = Sampler::new(SamplingParams::new(0.8, 0.95, 0).unwrap(), 7);
        let got = step.next_token(1, &mut open, 3).unwrap();
        let want = open_ref.sample(&[0.0, 2.0, 1.0]).unwrap();
        assert_eq!(got, want);
        assert_eq!(open.rng_state(), open_ref.rng_state());
        assert_eq!(open.first_top2_margin(), open_ref.first_top2_margin());
        assert_eq!(step.device().shortlist_decodes, 2);
        assert_eq!(step.device().host_logits, 0);

        // top_k > 20 falls back to host logits.
        let mut wide = Sampler::new(SamplingParams::new(0.8, 0.95, 21).unwrap(), 7);
        step.next_token(1, &mut wide, 3).unwrap();
        assert_eq!(step.device().shortlist_decodes, 2);
        assert_eq!(step.device().host_logits, 1);

        // Armed capture falls back to host logits even when top_k <= 20.
        let mut captured = Sampler::new(SamplingParams::new(0.8, 0.9, 10).unwrap(), 7);
        let _cap = captured.arm_capture(4);
        step.next_token(1, &mut captured, 3).unwrap();
        assert_eq!(step.device().shortlist_decodes, 2);
        assert_eq!(step.device().host_logits, 2);

        // Error in decode_to_logits leaves fed unchanged and latches full reset.
        let pos_before = step.fed();
        step.device.fail_decode = true;
        assert!(step.next_token(1, &mut bounded, 3).is_err());
        assert_eq!(step.fed(), pos_before);
        step.device.fail_decode = false;
        let gen0 = step.kept_generation();
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 1);
        assert_eq!(step.kept_generation(), gen0 + 1);

        // Error in sample_from_logits advances fed (token was consumed) and latches full reset.
        step.device.fail_sample_token = true;
        assert!(step.next_token(1, &mut bounded, 3).is_err());
        assert_eq!(step.fed(), 1);
        let gen1 = step.kept_generation();
        step.reset().unwrap();
        assert_eq!(step.device().full_resets, 2);
        assert_eq!(step.kept_generation(), gen1 + 1);
    }
}
