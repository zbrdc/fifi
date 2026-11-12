//! The card implementation of [`TokenDevice`]: the decode step (`step.rs`) with the split MoE block
//! (`moe_block.rs`) over the expert bank.
//!
//! WHAT: [`CardDevice`] owns a [`FlashStep`] and a [`SplitMoeAt`] and maps the driver's calls onto the
//! step's launches. The token start is written with synchronous copies on the step's stream (the
//! step parameters in the words block, the stream in the step arena), so they are ordered before the
//! first layer. The stream read-back at the PLE layer and the logits read-back wait for the queued
//! work.
//!
//! The split block is the step's MoE seam: `SplitMoeAt` (in `step.rs`) maps a model layer to the bank
//! row and delegates to the block's own `run`.

use std::ops::Range;

use super::{
    driver::TokenDevice,
    feeds::StartSink,
    ple_feed::PleKeyValue,
    state::StateSlot,
    step::{FlashStep, SplitMoeAt},
    step_plan::Slot,
};
use crate::family::flash_next::cuda::feeds::device::DevicePle;
use crate::{
    EngError, family::qwen36::exl3_tensors::f32_le_bytes,
    kernels::cuda::resident::step_params::StepParams,
};

pub(super) struct CardDevice {
    step: FlashStep<'static>,
    moe: SplitMoeAt<'static>,
    /// `branches * hidden`.
    stream_width: usize,
    /// The width of the head's output (the padded vocabulary).
    logits_width: usize,
}

impl CardDevice {
    pub(super) fn new(step: FlashStep<'static>, moe: SplitMoeAt<'static>) -> Self {
        let stream_width = step.geometry.branches * step.geometry.hidden;
        let logits_width = step.model.vocab;
        Self {
            step,
            moe,
            stream_width,
            logits_width,
        }
    }

    /// The kernel the single-row EXL3 projections of this context run on (`legacy`, `wide1`, ...),
    /// from `FIFI_EXL3_SINGLE_GATHER` or the shuffle kernel; the Flash-Next load line does not print it.
    #[cfg(test)]
    pub(super) fn single_row_kernel_name(&self) -> Result<&'static str, EngError> {
        self.step.ctx.single_row_kernel_name()
    }

    /// The single-row launch counters of this context, since it was made: the receipt run's proof
    /// that the grouped launches of the served step (the mixer inputs and the shared expert) ran
    /// on the wide kernel, one launch per group.
    #[cfg(test)]
    pub(super) fn kernel_launches(&self) -> crate::kernels::single_row::SingleRowLaunches {
        self.step.ctx.exl3_launch_counts()
    }

    /// The bank's statistics line for the load log and the per-job report.
    pub(super) fn bank_misses(&self) -> u64 {
        self.moe.bank().stats().misses
    }

    /// The bank's placement digest ([`bank_placement_digest`]): what a run record's step-0
    /// logits hash compares at.
    #[cfg(test)]
    pub(super) fn placement_digest(&self) -> (u32, String) {
        bank_placement_digest(&self.moe)
    }
}

/// The expert bank's placement state: slots per layer and a SHA-256 over every layer's
/// `(place, expert)` residents, by place. The step-0 logits of a run follow this state (the
/// bank size and the free VRAM at load pick the residents), so two run records compare only
/// at the same digest.
#[cfg(test)]
fn bank_placement_digest(moe: &SplitMoeAt<'static>) -> (u32, String) {
    use sha2::{Digest, Sha256};
    let state = moe.bank().state();
    let geometry = *state.geometry();
    let mut hasher = Sha256::new();
    for layer in 0..geometry.layers {
        hasher.update((layer as u32).to_le_bytes());
        for (place, expert) in state.places().occupied(layer) {
            hasher.update(place.to_le_bytes());
            hasher.update(expert.to_le_bytes());
        }
    }
    (geometry.slots as u32, format!("{:x}", hasher.finalize()))
}

impl StartSink for CardDevice {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        let range = self.step.params()?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        let range = self.step.slot(Slot::Stream)?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &f32_le_bytes(stream))
    }

    /// The PLE projections stage their own embedding; the start needs no copy of it.
    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        Ok(())
    }
}

impl TokenDevice for CardDevice {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.step.begin_token()
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        if range.is_empty() {
            return Ok(());
        }
        self.step.run_layers(range, pos, 1, &mut self.moe)
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Stream)?, self.stream_width)
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        StartSink::write_stream(self, stream)
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step.head(1)
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Logits)?, self.logits_width)
    }

    /// Drains the token's work and refuses a nonzero status word or EXL3 flag (the step poisons
    /// itself, and the driver poisons with it).
    fn check_health(&mut self) -> Result<(), EngError> {
        self.step.finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.reset()
    }

    fn bank_misses(&self) -> Option<u64> {
        Some(CardDevice::bank_misses(self))
    }
}

impl super::driver::StateIo for CardDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        self.step.read_fixed_state()
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.step.fixed_state_bytes()
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.step.write_fixed_state(parts)
    }
}

/// The card implementation of [`ChunkDevice`]: the same [`FlashStep`] and [`SplitMoeAt`] as
/// [`CardDevice`], one step whose plan holds `rows` rows, every launch at the chunk's row count.
/// The chunk's row count is remembered from `write_chunk_params`, so the read-backs know their
/// width.
pub(super) struct CardChunkDevice {
    step: FlashStep<'static>,
    moe: SplitMoeAt<'static>,
    /// The PLE projections on the card, when the device route is on: `write_chunk_ple` stages the
    /// chunk's embeddings here and `ple_project_chunk` runs them.
    ple: Option<DevicePle>,
    stream_width: usize,
    logits_width: usize,
    /// The rows of the chunk in flight, set by `write_chunk_params`.
    m: usize,
}

impl CardChunkDevice {
    pub(super) fn new(step: FlashStep<'static>, moe: SplitMoeAt<'static>) -> Self {
        Self::with_ple(step, moe, None)
    }

    pub(super) fn with_ple(
        step: FlashStep<'static>,
        moe: SplitMoeAt<'static>,
        ple: Option<DevicePle>,
    ) -> Self {
        let stream_width = step.geometry.branches * step.geometry.hidden;
        let logits_width = step.model.vocab;
        Self {
            step,
            moe,
            ple,
            stream_width,
            logits_width,
            m: 1,
        }
    }
}

impl CardChunkDevice {
    #[cfg(test)]
    pub(super) fn single_row_kernel_name(&self) -> Result<&'static str, EngError> {
        self.step.ctx.single_row_kernel_name()
    }

    #[cfg(test)]
    pub(super) fn kernel_launches(&self) -> crate::kernels::single_row::SingleRowLaunches {
        self.step.ctx.exl3_launch_counts()
    }

    /// The bank's statistics line, as 's.
    pub(super) fn bank_misses(&self) -> u64 {
        self.moe.bank().stats().misses
    }

    /// The bank's placement digest (slots per layer, SHA-256 of the residents at load):
    /// what a run record's step-0 logits hash compares at.
    #[cfg(test)]
    pub(super) fn placement_digest(&self) -> (u32, String) {
        bank_placement_digest(&self.moe)
    }
}

impl super::driver::chunk::ChunkDevice for CardChunkDevice {
    fn drain(&mut self) -> Result<(), EngError> {
        self.step.stream.synchronize()
    }

    fn write_chunk_params(&mut self, token0: i32, pos: i32, m: i32) -> Result<(), EngError> {
        self.m = usize::try_from(m).map_err(|_| {
            EngError::BadRequest(format!("Flash-Next chunk driver: a negative row count {m}"))
        })?;
        // The kernels read the position word; the token word is unused at m > 1 (the projections
        // carry the rows).
        let params = StepParams::new(token0, pos)?;
        let range = self.step.params()?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &params.to_bytes())
    }

    fn write_chunk_stream(&mut self, rows: &[f32]) -> Result<(), EngError> {
        let range = self.step.slot(Slot::Stream)?;
        range
            .buffer()
            .write_at(self.step.stream, range.offset(), &f32_le_bytes(rows))
    }

    /// The device route stages the chunk's embeddings and queues both projections right away:
    /// they depend only on the embeddings, so the two launches ride ahead of the layers and are
    /// done when `ple_project_chunk` reads them back. Without the route there is nothing to
    /// stage; the projections run on the host inside `ple_update`.
    fn write_chunk_ple(&mut self, rows: &[f32]) -> Result<(), EngError> {
        let Some(ple) = self.ple.as_mut() else {
            return Ok(());
        };
        let m = self.m;
        ple.enqueue_chunk(self.step.ctx, self.step.stream, rows, m)
    }

    fn ple_project_chunk(&mut self) -> Result<Vec<PleKeyValue>, EngError> {
        let m = self.m;
        let Some(ple) = self.ple.as_ref() else {
            return Err(EngError::Backend {
                operation: "flash_next_cuda_chunk",
                message: "the PLE device route was armed but this device has no projections"
                    .to_string(),
            });
        };
        ple.collect_chunk(self.step.stream, m)
    }

    fn begin_chunk(&mut self) -> Result<(), EngError> {
        self.moe.begin_chunk();
        self.step.begin_chunk(self.m)
    }

    fn begin_prefill(&mut self) -> Result<(), EngError> {
        self.moe.begin_prefill();
        Ok(())
    }

    fn end_prefill(&mut self) -> Result<(), EngError> {
        self.moe.end_prefill()
    }

    fn layers(
        &mut self,
        range: std::ops::Range<usize>,
        pos: usize,
        m: usize,
    ) -> Result<(), EngError> {
        if range.is_empty() {
            return Ok(());
        }
        self.step.run_layers(range, pos, m, &mut self.moe)
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step
            .read_f32(self.step.slot(Slot::Stream)?, self.m * self.stream_width)
    }

    fn write_stream_back(&mut self, rows: &[f32]) -> Result<(), EngError> {
        <Self as super::driver::chunk::ChunkDevice>::write_chunk_stream(self, rows)
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step.head(self.m)
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        // The last row of the plan's `[rows][vocab]` logits slot.
        let range = self.step.slot(Slot::Logits)?;
        let last = range.buffer().range(
            range.offset() + (self.m - 1) * self.logits_width * super::step::F32,
            self.logits_width * super::step::F32,
        )?;
        self.step.read_f32(last, self.logits_width)
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.step.finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.m = 1;
        // A reset can land inside an open prefill window; close it so the next single-row
        // step does not ride a stale plan. Best-effort: a failure here means the card is
        // going down, and the step reset below still clears what it can.
        if let Err(error) = self.moe.end_prefill() {
            eprintln!("flash-next cuda: ending the prefill window on reset failed: {error:?}");
        }
        self.step.reset()
    }

    fn bank_misses(&self) -> Option<u64> {
        Some(self.moe.bank().stats().misses)
    }
}

impl super::driver::StateIo for CardChunkDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        self.step.read_fixed_state()
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.step.fixed_state_bytes()
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.step.write_fixed_state(parts)
    }
}
