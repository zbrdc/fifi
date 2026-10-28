//! Stable device top-20 selection without a full logits readback. Launch-only; no allocation,
//! copy or synchronization. The result is a compact candidate list,not a device RNG/draw.
use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::shortlist::{self as raw, ShortlistRaw};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};
pub(crate) const K: usize = raw::K;
pub(crate) const RESULT_BYTES: usize = raw::RESULT_BYTES;
pub(crate) const RESULT_OFFSET: usize = raw::RESULT_OFFSET;
pub(crate) const SCRATCH_BYTES: usize = raw::SCRATCH_BYTES;
pub(crate) struct ShortlistCall<'a> {
    pub(crate) logits: DeviceRange<'a>,
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) n: usize,
}
impl CudaContext {
    /// Checks the full vocabulary and disjoint reusable scratch;queues stable top20.
    /// The caller orders the finite-input certificate/result read after this stream work.
    pub(crate) fn shortlist(
        &self,
        stream: &DeviceStream,
        call: ShortlistCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let logits = self.raw_range(&call.logits)?;
        let scratch = self.raw_range(&call.scratch)?;
        if !(1..=raw::MAX_LOGITS).contains(&call.n) {
            return Err(EngError::BadRequest(
                "device shortlist vocabulary must be in 1..=2^30".to_owned(),
            ));
        }
        let bytes = call
            .n
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("device shortlist size overflows".to_owned()))?;
        require_fit("shortlist logits", &call.logits, bytes, 4)?;
        require_fit("shortlist scratch", &call.scratch, SCRATCH_BYTES, 16)?;
        require_apart(
            &Claim::of("shortlist scratch", &call.scratch, SCRATCH_BYTES),
            &Claim::of("shortlist logits", &call.logits, bytes),
        )?;
        self.launch_on(stream, move |s| {
            raw::launch(
                s,
                ShortlistRaw {
                    logits,
                    scratch,
                    n: call.n,
                },
            )
        })
    }
}
#[cfg(test)]
mod tests;
