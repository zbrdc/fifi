//! The card implementation of [`TokenDevice`] for a Flash-Next GGUF: the GGUF decode step
//! (`staged_moe.rs`) over the routed-expert owner, for the opt-in serving path
//! (`INFERENCE_FLASH_NEXT_GGUF_SERVE`, `load/gguf/start.rs`).
//!
//! WHAT: [`GgufCardDevice`] owns a [`GgufStep`] and the [`GgufMoeOwner`] bound to it and maps the
//! driver's calls onto the step's launches, method for method as the EXL3 pack's `CardDevice`
//! (`device.rs`) does. The token start is written with synchronous copies on the step's stream
//! (the step parameters in the words block, the stream in the step arena), ordered before the
//! first layer; the stream read-back at the PLE layer and the logits read-back wait for the
//! queued work.
//!
//! Each side of the PLE update is a separate layer walk with one MoE binding. The step refuses
//! ranges outside its layer window or across the PLE boundary.
//!
//! A GGUF trunk has no expert bank and no disk rung, so the bank and rung reports keep the
//! trait's `None` defaults.

use std::ops::Range;

use super::{
    driver::{StateIo, StateSlot, TokenDevice},
    feeds::StartSink,
    staged_moe::{GgufMoeOwner, GgufStep},
    step_plan::Slot,
};
use crate::{
    EngError, family::exl3_tensors::f32_le_bytes, kernels::cuda::resident::step_params::StepParams,
};

/// The GGUF step and its MoE owner behind the token driver.
pub(super) struct GgufCardDevice<'a> {
    step: GgufStep<'a>,
    moe: GgufMoeOwner<'a>,
    /// `branches * hidden`.
    stream_width: usize,
    /// The width of the head's output (the padded vocabulary).
    logits_width: usize,
}

impl<'a> GgufCardDevice<'a> {
    /// `moe` must have been built from `step` (`GgufMoeOwner::new`); every bind below checks it.
    pub(super) fn new(step: GgufStep<'a>, moe: GgufMoeOwner<'a>) -> Self {
        let inner = step.step();
        let stream_width = inner.geometry.branches * inner.geometry.hidden;
        let logits_width = inner.model.vocab;
        Self {
            step,
            moe,
            stream_width,
            logits_width,
        }
    }
}

impl StartSink for GgufCardDevice<'_> {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        let params = StepParams::new(token, pos)?;
        let step = self.step.step();
        let range = step.params()?;
        range
            .buffer()
            .write_at(&step.stream, range.offset(), &params.to_bytes())
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        let step = self.step.step();
        let range = step.slot(Slot::Stream)?;
        range
            .buffer()
            .write_at(&step.stream, range.offset(), &f32_le_bytes(stream))
    }

    /// The GGUF feed stages its own PLE rows on the host (`GgufTokenFeeds`); the start needs no
    /// copy of the embedding, as for the pack.
    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        Ok(())
    }
}

impl TokenDevice for GgufCardDevice<'_> {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.step.step().begin_token()
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        self.step.run_layers(&mut self.moe, range, pos)
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        let step = self.step.step();
        step.read_f32(step.slot(Slot::Stream)?, self.stream_width)
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        StartSink::write_stream(self, stream)
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step.step().head(1)
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        let step = self.step.step();
        step.read_f32(step.slot(Slot::Logits)?, self.logits_width)
    }

    /// Drains the token's work and refuses a nonzero status word or arithmetic flag (the step
    /// poisons itself, and the driver poisons with it).
    fn check_health(&mut self) -> Result<(), EngError> {
        self.step.step().finish_token()
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.step.step().reset()
    }
}

impl StateIo for GgufCardDevice<'_> {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        self.step.step().read_fixed_state()
    }

    fn fixed_state_bytes(&self) -> u64 {
        self.step.step().fixed_state_bytes()
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.step.step_mut().write_fixed_state(parts)
    }
}
