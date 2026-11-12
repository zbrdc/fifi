//! The Flash-Next token driver: one token end to end, behind the shared `TokenStepper` seam.
//!
//! WHAT: [`TokenDriver`] runs the sequence of the host reference (`family/flash_next/step.rs`) over
//! two collaborators and owns the position, the poison flag and the fed count:
//!
//! 1. [`TokenFeeds::begin`]: the embedding row repeated over the branches (the initial stream) and
//!    the PLE embedding of the token, both host work (`feeds`, `embedding`, `ple_feed`).
//! 2. [`write_start`] to the device: the step parameters (token and position) and the stream.
//! 3. The layers before the PLE layer.
//! 4. The PLE block: the stream comes back to the host, [`TokenFeeds::ple_update`] returns the
//!    update, and the sum goes back to the device (the host reference adds it to the stream before
//!    the PLE layer's attention read).
//! 5. The remaining layers.
//! 6. With logits wanted: the final mixer, `lm_head` and the read-back of the logits row.
//! 7. The health check (status word and the sticky arithmetic flag) and the position advance.
//!
//! The device half is [`TokenDevice`] (the decode step and its MoE block on a card), the host half is
//! [`TokenFeeds`] (the pack's embedding and the PLE feed). The traits exist so that the sequence,
//! the poison rule and the executor on top of it run in host tests against fakes; the card
//! implementations are `device.rs` and `pack_feeds.rs`.
//!
//! The fed rule of `TokenStepper`: `fed` advances when, and only when, the model state consumed the
//! token. A failure anywhere after the first write to the card poisons the driver: every later
//! feed is refused until `reset`, as the host reference does after a failed step.
//!
//! Sampling is the shared one: [`TokenStepper::next_token`] reads the logits row and
//! `sample_logits` draws from the first `sampler_vocab` values, so the padded ids of the head
//! (above the tokenizer's vocabulary) are never drawn and a greedy request is argmax.

use std::{
    ops::Range,
    time::{Duration, Instant},
};

use super::feeds::{StartShape, StartSink, write_start};
use super::ple_feed::PleKeyValue;
use crate::{EngError, family::session::TokenStepper};

type Result<T> = std::result::Result<T, EngError>;

/// One persistent tensor of the state arena. The layer index is the model layer, not an ordinal
/// among its kind. Defined here (not in `state`) so the resume seam below can name it in every
/// cfg the driver builds under; `state` imports it for its plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) enum StateSlot {
    /// Recurrent state of a DeltaNet layer.
    DeltaRec(usize),
    /// Conv ring of a DeltaNet layer.
    DeltaConv(usize),
    /// fp16 K cache of a QSA layer.
    QsaK(usize),
    /// fp16 V cache of a QSA layer.
    QsaV(usize),
    /// Pooled indexer keys of a QSA layer.
    QsaIndexKeys(usize),
    /// Raw-key ring of the open indexer block of a QSA layer.
    QsaRawRing(usize),
    PleRing,
    Words,
}

/// The sizes the driver checks and splits the layers by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DriverShape {
    /// Decoder layers the device runs.
    pub(super) layers: usize,
    /// The layer whose attention read follows the PLE update.
    pub(super) ple_layer: usize,
    /// Positions the K/V caches hold: the context limit of the driver.
    pub(super) positions: usize,
    /// The sizes of one token start.
    pub(super) start: StartShape,
}

/// Where the wall time of the last token went. The layer launches are queued, but each MoE block
/// waits on the host for its classify log, so a phase holds the device work it had to wait for; the
/// work still queued when a phase ends is paid by the next call that waits (the stream read-back
/// or the logits read-back).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TokenTimes {
    /// The embedding row, the n-gram plan and the n-gram row reads.
    pub(super) rows: Duration,
    /// The start write: step parameters and stream.
    pub(super) start: Duration,
    /// The layers before the PLE layer.
    pub(super) layers_before: Duration,
    /// The stream read-back at the PLE layer: the wait for everything queued before it.
    pub(super) ple_sync: Duration,
    /// The host PLE step (convolution, projections, norms).
    pub(super) ple_host: Duration,
    /// The stream write-back.
    pub(super) ple_write: Duration,
    /// The layers from the PLE layer on.
    pub(super) layers_after: Duration,
    /// The final mixer, the head and the logits read-back (zero for a fed token).
    pub(super) head: Duration,
    /// The health read.
    pub(super) health: Duration,
}

/// The start vectors of a token.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StartVectors {
    /// `branches * hidden`: the embedding row once per branch.
    pub(super) stream: Vec<f32>,
    /// `ple_embed`: the PLE embedding of the token (its n-gram rows).
    pub(super) ple_embedding: Vec<f32>,
}

/// The host side of a token: the embedding table and the PLE feed.
pub(super) trait TokenFeeds {
    /// The start vectors of `token`. Reads the pack and the n-gram table; changes no state.
    fn begin(&mut self, token: u32) -> Result<StartVectors>;
    /// The start vectors of a whole chunk, in order. The default reads one token at a time,
    /// which serves a stateless feed; a feed whose `ple_update` consumes per-token rows must
    /// override it, because the plain `begin` holds one pending row set at a time.
    fn begin_chunk(&mut self, tokens: &[u32]) -> Result<Vec<StartVectors>> {
        tokens.iter().map(|&token| self.begin(token)).collect()
    }
    /// The PLE update for `token` given the stream at the PLE layer's input (`branches * hidden`).
    /// Advances the PLE history and ring: called once per fed token, after [`Self::begin`].
    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>>;
    /// Installs the key/value pair a device-projected row returns, for the next `ple_update`.
    /// The default refuses: only the device PLE route takes pairs.
    fn set_ple_pair(&mut self, _pair: PleKeyValue) -> Result<()> {
        Err(EngError::BadRequest(
            "this feed does not take device PLE pairs".to_string(),
        ))
    }
    /// Forgets the history and the ring: a new sequence.
    fn reset(&mut self);
    /// The host half of a resume snapshot (token history, host ring mirror), or `None` for a
    /// feed with no PLE state. The default is `None`.
    fn ple_state(&self) -> Option<(&[u32], &[f32])> {
        None
    }
    /// Installs a saved host half, as a prefix restore brings the feed back to a boundary.
    /// The default refuses: only a feed with PLE state resumes.
    fn ple_resume(&mut self, _history: &[u32], _ring: &[f32]) -> Result<()> {
        Err(EngError::BadRequest(
            "this feed has no PLE state to resume".to_string(),
        ))
    }
}

/// The fixed device state a resume saves: one raw byte image per copied slot (the GDN recurrent
/// states, their conv rings, the PLE device ring AND the QSA raw-key ring — its slots hold the
/// open block's accumulated raw keys, which the saving job's own continuation overwrites), the
/// position the model had consumed, and the PLE feed's host half. The QSA K/V planes and the
/// indexer keys are not here: they stay in place on the card and a restore rewinds the valid
/// length onto them.
#[derive(Clone)]
pub(in crate::family::flash_next) struct ResumeSnapshot {
    /// `(slot, bytes)` for every copied slot (the GDN recurrent states, their conv rings, the
    /// PLE device ring, the QSA raw-key ring).
    pub(in crate::family::flash_next) parts: Vec<(StateSlot, Vec<u8>)>,
    /// Tokens the model had consumed when the boundary was saved (the pending token excluded).
    pub(in crate::family::flash_next) pos: usize,
    /// The PLE feed's token history at the boundary.
    pub(in crate::family::flash_next) ple_history: Vec<u32>,
    /// The PLE feed's host ring mirror at the boundary.
    pub(in crate::family::flash_next) ple_ring: Vec<f32>,
}

impl ResumeSnapshot {
    /// The host bytes this snapshot holds: its fixed parts, the PLE token history and the
    /// PLE ring mirror.
    pub(in crate::family::flash_next) fn host_bytes(&self) -> u64 {
        let parts: u64 = self.parts.iter().map(|(_, bytes)| bytes.len() as u64).sum();
        let history = (self.ple_history.len() as u64) * std::mem::size_of::<u32>() as u64;
        let ring = (self.ple_ring.len() as u64) * std::mem::size_of::<f32>() as u64;
        parts + history + ring
    }
}

/// Fixed-state device IO for a driver's device: the seam a [`ResumesState`] driver reads and
/// installs its snapshot through.
pub(super) trait StateIo {
    /// Reads every copied slot, behind the queued work.
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>>;
    /// The bytes [`Self::read_fixed_state`] will produce, without reading them: the host
    /// charge of one snapshot's fixed part.
    fn fixed_state_bytes(&self) -> u64;
    /// Installs every copied slot, ordered before the next launch.
    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<()>;
}

/// The resume seam of the drivers: what the executor's prefix store saves and restores. The
/// position count comes from [`TokenStepper::fed`].
pub(crate) trait ResumesState: StepperPositions {
    fn save_resume(&self) -> Result<ResumeSnapshot>;
    fn restore_resume(&mut self, snapshot: &ResumeSnapshot) -> Result<()>;
    /// The arena generation the driver is in: one advance per fresh sequence. The executor
    /// stamps it into the prefix store's path, so a restore at any other generation never
    /// matches the entry — the rows it left in place may have been overwritten meanwhile.
    fn resume_generation(&self) -> u64;
    /// The host bytes [`Self::save_resume`] would produce now, without reading any device
    /// state: the budget charge of one more entry.
    fn resume_host_bytes(&self) -> u64;
}

/// The device side of a token. It is the sink of the token start.
pub(super) trait TokenDevice: StartSink {
    /// Clears the per-token flags (the EXL3 flag and the status word). Called once per token, after
    /// the start is written and before the first layer.
    fn begin_token(&mut self) -> Result<()>;
    /// Queues layers `range` for the token at `pos`.
    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<()>;
    /// The stream (`branches * hidden`), read back behind the queued work.
    fn read_stream(&mut self) -> Result<Vec<f32>>;
    /// Replaces the stream.
    fn write_stream_back(&mut self, stream: &[f32]) -> Result<()>;
    /// Queues the final mixer and the head.
    fn head(&mut self) -> Result<()>;
    /// The logits row of the last head, behind the queued work.
    fn read_logits(&mut self) -> Result<Vec<f32>>;
    /// Reads the status word and the arithmetic flag; refuses a token that set either.
    fn check_health(&mut self) -> Result<()>;
    /// Zeroes the persistent state: a new sequence.
    fn reset(&mut self) -> Result<()>;
    /// Expert bank misses so far, for a device with a bank.
    fn bank_misses(&self) -> Option<u64> {
        None
    }
}

/// The token driver: a [`TokenStepper`] over a device and its feeds.
pub(super) struct TokenDriver<D: TokenDevice, F: TokenFeeds> {
    device: D,
    feeds: F,
    shape: DriverShape,
    /// Tokens the state has consumed: the next position.
    pos: usize,
    poisoned: bool,
    /// Advances once per fresh sequence (at the ATTEMPT of its first row, so a failed or
    /// partial first write also advances it; the row never needs to commit): the
    /// in-place rows a resume leaves on the card belong to the generation that wrote them, so a
    /// snapshot saved at another generation is structurally unusable and never matches.
    generation: u64,
    times: TokenTimes,
}

impl<D: TokenDevice, F: TokenFeeds> TokenDriver<D, F> {
    pub(super) fn new(device: D, feeds: F, shape: DriverShape) -> Result<Self> {
        super::driver::chunk::prof::check_switch_spelling()?;
        if shape.layers == 0 || shape.ple_layer >= shape.layers || shape.positions == 0 {
            return Err(EngError::BadRequest(format!(
                "Flash-Next driver: {} layers with the PLE layer at {} and {} positions",
                shape.layers, shape.ple_layer, shape.positions
            )));
        }
        Ok(Self {
            device,
            feeds,
            shape,
            pos: 0,
            generation: 0,
            poisoned: false,
            times: TokenTimes::default(),
        })
    }

    /// Where the last token's time went.
    pub(super) fn last_times(&self) -> TokenTimes {
        self.times
    }

    /// Expert bank misses so far.
    pub(super) fn bank_misses(&self) -> Option<u64> {
        self.device.bank_misses()
    }

    /// Positions the caches hold.
    pub(super) fn positions(&self) -> usize {
        self.shape.positions
    }

    /// The device, shared with the routing-diagnostics reads; tests also poke it directly.
    pub(in crate::family::flash_next::cuda) fn device(&self) -> &D {
        &self.device
    }

    /// The device mutably, for the executor tests' failure injection and fixture writes.
    #[cfg(test)]
    pub(in crate::family::flash_next::cuda) fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    #[cfg(test)]
    pub(super) fn feeds(&self) -> &F {
        &self.feeds
    }

    /// One token. `want_logits` runs the head and returns the row.
    fn run_token(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>> {
        if self.poisoned {
            return Err(EngError::BadRequest(
                "Flash-Next driver: the state needs a reset after a failed step".into(),
            ));
        }
        if self.pos >= self.shape.positions {
            return Err(EngError::ContextLimit(format!(
                "Flash-Next driver: the caches hold {} positions",
                self.shape.positions
            )));
        }
        // A refused id changes nothing; `write_start` repeats the rule, but inside the queued part,
        // where a failure poisons the driver.
        if usize::try_from(token).is_ok_and(|id| id >= self.shape.start.sampler_vocab) {
            return Err(EngError::BadRequest(format!(
                "token id {token} is outside the sampler vocabulary 0..{}",
                self.shape.start.sampler_vocab
            )));
        }
        // Host reads only: a failure here leaves the state as it was.
        let opened = Instant::now();
        let start = self.feeds.begin(token)?;
        // The generation advances at the ATTEMPT of a fresh sequence's first row, not only on
        // success: a first chunk that fails after queueing device work may still have written
        // the rows, and a stale snapshot must never survive that.
        if self.pos == 0 {
            self.generation += 1;
        }
        self.times = TokenTimes {
            rows: opened.elapsed(),
            ..TokenTimes::default()
        };
        let outcome = self.queue(token, &start, want_logits);
        match outcome {
            Ok(logits) => {
                self.pos += 1;
                Ok(logits)
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    /// Everything after the first write to the card.
    fn queue(
        &mut self,
        token: u32,
        start: &StartVectors,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let pos = self.pos;
        let shape = self.shape;
        let mut clock = Instant::now();
        let mut lap = || {
            let spent = clock.elapsed();
            clock = Instant::now();
            spent
        };
        write_start(
            &mut self.device,
            &shape.start,
            token,
            pos,
            &start.stream,
            &start.ple_embedding,
        )?;
        self.device.begin_token()?;
        self.times.start = lap();
        self.device.layers(0..shape.ple_layer, pos)?;
        self.times.layers_before = lap();
        let mut stream = self.device.read_stream()?;
        self.times.ple_sync = lap();
        let update = self.feeds.ple_update(token, &stream)?;
        self.times.ple_host = lap();
        if update.len() != stream.len() {
            return Err(EngError::BadRequest(format!(
                "Flash-Next driver: the PLE update has {} values for a stream of {}",
                update.len(),
                stream.len()
            )));
        }
        for (value, delta) in stream.iter_mut().zip(&update) {
            *value += delta;
        }
        self.device.write_stream_back(&stream)?;
        self.times.ple_write = lap();
        self.device.layers(shape.ple_layer..shape.layers, pos)?;
        self.times.layers_after = lap();
        let logits = if want_logits {
            self.device.head()?;
            let row = self.device.read_logits()?;
            self.times.head = lap();
            Some(row)
        } else {
            None
        };
        self.device.check_health()?;
        self.times.health = lap();
        Ok(logits)
    }
}

/// The one [`TokenStepper`] method beyond the trait that an executor calls: the number of
/// positions the caches hold.
pub(crate) trait StepperPositions: TokenStepper {
    fn positions(&self) -> usize;
}

impl<D: TokenDevice, F: TokenFeeds> StepperPositions for TokenDriver<D, F> {
    fn positions(&self) -> usize {
        self.positions()
    }
}

impl<D: TokenDevice, F: TokenFeeds> TokenStepper for TokenDriver<D, F> {
    const OPERATION: &'static str = "flash_next_cuda";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.run_token(token, false).map(|_| ())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.run_token(token, true)?
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".to_string(),
            })
    }

    fn fed(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<()> {
        self.device.reset()?;
        self.feeds.reset();
        self.pos = 0;
        self.poisoned = false;
        Ok(())
    }
}

impl<D: TokenDevice + StateIo, F: TokenFeeds> ResumesState for TokenDriver<D, F> {
    fn resume_generation(&self) -> u64 {
        self.generation
    }

    fn save_resume(&self) -> Result<ResumeSnapshot> {
        let parts = self.device.read_fixed_state()?;
        let (ple_history, ple_ring) = self.feeds.ple_state().unwrap_or((&[], &[]));
        Ok(ResumeSnapshot {
            parts,
            pos: self.pos,
            ple_history: ple_history.to_vec(),
            ple_ring: ple_ring.to_vec(),
        })
    }

    fn restore_resume(&mut self, snapshot: &ResumeSnapshot) -> Result<()> {
        self.device.write_fixed_state(&snapshot.parts)?;
        if !snapshot.ple_ring.is_empty() {
            self.feeds
                .ple_resume(&snapshot.ple_history, &snapshot.ple_ring)?;
        }
        self.pos = snapshot.pos;
        self.poisoned = false;
        Ok(())
    }

    fn resume_host_bytes(&self) -> u64 {
        let (ple_history, ple_ring) = self.feeds.ple_state().unwrap_or((&[], &[]));
        let history = (ple_history.len() as u64) * std::mem::size_of::<u32>() as u64;
        let ring = (ple_ring.len() as u64) * std::mem::size_of::<f32>() as u64;
        self.device.fixed_state_bytes() + history + ring
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for the chunked prefill step")
)]
pub(in crate::family::flash_next) mod chunk;
#[cfg(test)]
pub(super) mod fakes;
#[cfg(test)]
mod tests;
