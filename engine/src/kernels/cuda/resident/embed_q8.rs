//! The Q8_0 token-embedding gather of the Gemma 4 Humanizer resident step: one
//! launch reading m token ids on the device and writing m rows of k f32 values
//! from the raw Q8_0 table blocks.
//!
//! [CudaContext::embed_q8] queues the kernel on a [DeviceStream] and returns when
//! it is queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable; m is in 1..=4096; k is a multiple of 32 in 32..=32768; n_rows is in
//! 1..=2^28; every host-known id is inside the table; the ranges hold their bytes at
//! 16-byte (out) and 4-byte (w, ids) alignment; out overlaps neither w nor ids.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::embed_q8::{EmbedQ8Raw, embed_q8_launch};

/// The most tokens one gather takes.
pub(crate) const EMBED_Q8_MAX_TOKENS: usize = 4096;

/// One gather call: the table blocks, the ids and the f32 output rows.
pub(crate) struct EmbedQ8Call<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) ids: DeviceRange<'b>,
    pub(crate) m: usize,
    pub(crate) n_rows: usize,
    pub(crate) k: usize,
}

impl CudaContext {
    /// Queues the gather of call on stream, writing call.out.
    pub(crate) fn embed_q8(
        &self,
        stream: &DeviceStream,
        call: EmbedQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_embed_q8(&call)?;
        self.launch_on(stream, |stream_raw| embed_q8_launch(stream_raw, &raw))
    }

    fn plan_embed_q8<'b>(&self, call: &EmbedQ8Call<'b>) -> Result<EmbedQ8Raw<'b>, EngError> {
        let (out_raw, w_raw, ids_raw) = (
            self.raw_range(&call.out)?,
            self.raw_range(&call.w)?,
            self.raw_range(&call.ids)?,
        );
        if !(1..=EMBED_Q8_MAX_TOKENS).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the embed_q8 takes 1 to {EMBED_Q8_MAX_TOKENS} tokens, got {}",
                call.m
            )));
        }
        if !(32..=32_768).contains(&call.k) || !call.k.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the embed_q8 needs k a multiple of 32 in 32..=32768, got {}",
                call.k
            )));
        }
        if !(1..=(1 << 28)).contains(&call.n_rows) {
            return Err(EngError::BadRequest(format!(
                "the embed_q8 table has {} rows, above 2^28",
                call.n_rows
            )));
        }
        let out_bytes = call
            .m
            .checked_mul(call.k)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the embed_q8 size overflows".to_owned()))?;
        let w_bytes = call
            .n_rows
            .checked_mul(call.k / 32)
            .and_then(|v| v.checked_mul(34))
            .ok_or_else(|| EngError::BadRequest("the embed_q8 table size overflows".to_owned()))?;
        let ids_bytes = call
            .m
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("the embed_q8 id size overflows".to_owned()))?;
        require_fit("out", &call.out, out_bytes, 16)?;
        require_fit("w", &call.w, w_bytes, 4)?;
        require_fit("ids", &call.ids, ids_bytes, 4)?;
        let out = Claim::of("out", &call.out, out_bytes);
        require_apart(&out, &Claim::of("w", &call.w, w_bytes))?;
        require_apart(&out, &Claim::of("ids", &call.ids, ids_bytes))?;
        Ok(EmbedQ8Raw {
            out: out_raw,
            w: w_raw,
            ids: ids_raw,
            m: call.m,
            n_rows: call.n_rows,
            k: call.k,
        })
    }
}

#[cfg(test)]
mod tests;
