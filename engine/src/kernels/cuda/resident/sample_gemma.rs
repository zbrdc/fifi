//! The device sampler of the Gemma 4 Humanizer decode step: one launch that reads
//! the logits row on the device and writes the chosen token id, its log
//! probability and a status word. The logits row never leaves the device.
//!
//! [CudaContext::sample_gemma] queues the kernel on a [DeviceStream] and returns
//! when it is queued.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable; n is a multiple of 32 in 32..=4194304; the temperature is finite; top_k is
//! 0 or up to n; top_p is in (0, 1]; min_p is in [0, 1); the logits and the scratch
//! row hold n floats at 4-byte alignment and do not overlap; the three outputs hold
//! four bytes each and overlap neither the logits nor the scratch row.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged until the worker calls the sampler")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::sample_gemma::{SampleGemmaRaw, sample_gemma_launch};

/// Sampling settings of one draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SampleParams {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
    pub(crate) seed: u64,
    pub(crate) position: u64,
}

/// The row, the scratch row, the outputs and the settings of one draw.
pub(crate) struct SampleGemmaCall<'b> {
    pub(crate) token: DeviceRange<'b>,
    pub(crate) logprob: DeviceRange<'b>,
    pub(crate) status: DeviceRange<'b>,
    pub(crate) probs: DeviceRange<'b>,
    pub(crate) logits: DeviceRange<'b>,
    pub(crate) n: usize,
    pub(crate) params: SampleParams,
}

impl CudaContext {
    /// Queues one draw of call on stream.
    pub(crate) fn sample_gemma(
        &self,
        stream: &DeviceStream,
        call: SampleGemmaCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_sample(&call)?;
        self.launch_on(stream, |stream_raw| sample_gemma_launch(stream_raw, &raw))
    }

    fn plan_sample<'b>(&self, call: &SampleGemmaCall<'b>) -> Result<SampleGemmaRaw<'b>, EngError> {
        let (token, logprob, status, probs, logits) = (
            self.raw_range(&call.token)?,
            self.raw_range(&call.logprob)?,
            self.raw_range(&call.status)?,
            self.raw_range(&call.probs)?,
            self.raw_range(&call.logits)?,
        );
        if !(32..=4_194_304).contains(&call.n) || !call.n.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the sampler needs n a multiple of 32 up to 4194304, got {}",
                call.n
            )));
        }
        let p = call.params;
        if !p.temperature.is_finite() {
            return Err(EngError::BadRequest(
                "the sampler needs a finite temperature".to_owned(),
            ));
        }
        if p.top_k > call.n {
            return Err(EngError::BadRequest(format!(
                "the sampler takes top_k 0 to {}, got {}",
                call.n, p.top_k
            )));
        }
        if !(p.top_p > 0.0 && p.top_p <= 1.0) {
            return Err(EngError::BadRequest(format!(
                "the sampler needs top_p in (0, 1], got {}",
                p.top_p
            )));
        }
        if !(0.0..1.0).contains(&p.min_p) {
            return Err(EngError::BadRequest(format!(
                "the sampler needs min_p in [0, 1), got {}",
                p.min_p
            )));
        }
        let bytes = call
            .n
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("the sampler size overflows".to_owned()))?;
        require_fit("logits", &call.logits, bytes, 4)?;
        require_fit("probs", &call.probs, bytes, 4)?;
        require_fit("token", &call.token, 4, 4)?;
        require_fit("logprob", &call.logprob, 4, 4)?;
        require_fit("status", &call.status, 4, 4)?;
        let logits_claim = Claim::of("logits", &call.logits, bytes);
        let probs_claim = Claim::of("probs", &call.probs, bytes);
        require_apart(&probs_claim, &logits_claim)?;
        let token_claim = Claim::of("token", &call.token, 4);
        let logprob_claim = Claim::of("logprob", &call.logprob, 4);
        let status_claim = Claim::of("status", &call.status, 4);
        for out in [&token_claim, &logprob_claim, &status_claim] {
            require_apart(out, &logits_claim)?;
            require_apart(out, &probs_claim)?;
        }
        // The three scalar outputs must also be apart from EACH OTHER: the native entry
        // rejects those aliases, and letting that rejection through would latch the
        // process-wide terminal flag instead of returning a BadRequest here.
        require_apart(&token_claim, &logprob_claim)?;
        require_apart(&token_claim, &status_claim)?;
        require_apart(&logprob_claim, &status_claim)?;
        Ok(SampleGemmaRaw {
            token,
            logprob,
            status,
            probs,
            logits,
            n: call.n,
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
            seed: p.seed,
            position: p.position,
        })
    }
}

#[cfg(test)]
mod tests;
