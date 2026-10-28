//! Diagnostic wrapper around the same transactional decoder used by serving.

use super::projection_checks::CpuChecks;
use super::*;
pub(super) use crate::family::flint::cuda_ffn::device::runtime::values;
use crate::family::flint::{
    cuda_ffn::device::runtime::{Decoder, State},
    exl3_resident::FlintExl3Resident,
    kv_plan::KvPlan,
};

pub(super) struct ResidentDecode<'a> {
    decoder: Decoder<'a>,
    state: State,
    stream: &'a DeviceStream,
}

impl<'a> ResidentDecode<'a> {
    pub(super) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        model: &'a FlintExl3Resident,
        capacity: usize,
    ) -> Result<Self, EngError> {
        Ok(Self {
            decoder: Decoder::new(ctx, stream, model)?,
            state: State::new(ctx, KvPlan::new(capacity, 1, false)?)?,
            stream,
        })
    }

    pub(super) fn reset(&mut self) -> Result<(), EngError> {
        self.state.reset(self.stream)
    }

    pub(super) fn step(
        &mut self,
        token: u32,
        checks: Option<&mut CpuChecks>,
    ) -> Result<Vec<f32>, EngError> {
        match checks {
            Some(checks) => {
                let mut observe = |stream: &DeviceStream,
                                   name: &str,
                                   input: &DeviceBuffer,
                                   raw: &DeviceBuffer| {
                    checks.check(stream, name, input, raw)
                };
                self.decoder
                    .step_observed(&mut self.state, token, &mut observe, |_, logits| {
                        Ok(logits.to_vec())
                    })
            }
            None => self
                .decoder
                .step(&mut self.state, token, |_, logits| Ok(logits.to_vec())),
        }
    }
}
