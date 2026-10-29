//! The GELU-tanh gated activation of the Gemma 4 Humanizer feed-forward block:
//! one launch computing out = 0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715 g^3)))*u.
//!
//! [CudaContext::gelu_ffn] queues the kernel on a [DeviceStream] and returns when it
//! is queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable, m is in 1..=65535, n is a multiple of 32 in 32..=65536, the three ranges
//! hold m*n floats at 4-byte alignment, and out overlaps neither gate nor up.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::gelu_ffn::{GeluFfnRaw, gelu_ffn_launch};

/// One gelu_ffn call: gate and up rows in, the activated product in out.
pub(crate) struct GeluFfnCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) gate: DeviceRange<'b>,
    pub(crate) up: DeviceRange<'b>,
    pub(crate) m: usize,
    pub(crate) n: usize,
}

impl CudaContext {
    /// Queues the gelu_ffn of call on stream, writing call.out.
    pub(crate) fn gelu_ffn(
        &self,
        stream: &DeviceStream,
        call: GeluFfnCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gelu_ffn(&call)?;
        self.launch_on(stream, |stream_raw| gelu_ffn_launch(stream_raw, &raw))
    }

    fn plan_gelu_ffn<'b>(&self, call: &GeluFfnCall<'b>) -> Result<GeluFfnRaw<'b>, EngError> {
        let out_raw = self.raw_range(&call.out)?;
        let gate_raw = self.raw_range(&call.gate)?;
        let up_raw = self.raw_range(&call.up)?;
        let bytes = call
            .m
            .checked_mul(call.n)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the gelu_ffn size overflows".to_owned()))?;
        if !(1..=65_535).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the gelu_ffn takes 1 to 65535 rows, got {}",
                call.m
            )));
        }
        if !(32..=65_536).contains(&call.n) || !call.n.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the gelu_ffn needs n a multiple of 32 in 32..=65536, got {}",
                call.n
            )));
        }
        require_fit("out", &call.out, bytes, 4)?;
        require_fit("gate", &call.gate, bytes, 4)?;
        require_fit("up", &call.up, bytes, 4)?;
        require_apart(
            &Claim::of("out", &call.out, bytes),
            &Claim::of("gate", &call.gate, bytes),
        )?;
        require_apart(
            &Claim::of("out", &call.out, bytes),
            &Claim::of("up", &call.up, bytes),
        )?;
        Ok(GeluFfnRaw {
            out: out_raw,
            gate: gate_raw,
            up: up_raw,
            m: call.m,
            n: call.n,
        })
    }
}

#[cfg(test)]
mod tests;
