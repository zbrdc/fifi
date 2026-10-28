//! The logit softcap of the Gemma 4 Humanizer head: one launch computing
//! out = cap * tanh(x / cap) (the pinned model uses cap 30).
//!
//! [CudaContext::softcap] queues the kernel on a [DeviceStream] and returns when it
//! is queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable, the cap is finite and above zero, n is a multiple of 32 in 32..=4194304,
//! both ranges hold n floats at 4-byte alignment, and out overlaps no byte of x.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::softcap::{SoftcapRaw, softcap_launch};

/// One softcap call: n values in x, the capped values in out.
pub(crate) struct SoftcapCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) cap: f32,
    pub(crate) n: usize,
}

impl CudaContext {
    /// Queues the softcap of call on stream, writing call.out.
    pub(crate) fn softcap(
        &self,
        stream: &DeviceStream,
        call: SoftcapCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_softcap(&call)?;
        self.launch_on(stream, |stream_raw| softcap_launch(stream_raw, &raw))
    }

    fn plan_softcap<'b>(&self, call: &SoftcapCall<'b>) -> Result<SoftcapRaw<'b>, EngError> {
        let (out_raw, x_raw) = (self.raw_range(&call.out)?, self.raw_range(&call.x)?);
        let bytes = call
            .n
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("the softcap size overflows".to_owned()))?;
        if !call.cap.is_finite() || call.cap <= 0.0 {
            return Err(EngError::BadRequest(format!(
                "the softcap needs a finite cap above zero, got {}",
                call.cap
            )));
        }
        if !(32..=4_194_304).contains(&call.n) || !call.n.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the softcap needs n as a multiple of 32 up to 4194304, got {}",
                call.n
            )));
        }
        require_fit("out", &call.out, bytes, 4)?;
        require_fit("x", &call.x, bytes, 4)?;
        require_apart(
            &Claim::of("out", &call.out, bytes),
            &Claim::of("x", &call.x, bytes),
        )?;
        Ok(SoftcapRaw {
            out: out_raw,
            x: x_raw,
            cap: call.cap,
            n: call.n,
        })
    }
}

#[cfg(test)]
mod tests;
