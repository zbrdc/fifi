//! The device facts of the MMQ glue (its C entry, wrapped): the built architecture the
//! cubins run under, the SM count and the opt-in shared memory per block. The dense and
//! grouped GEMM wrappers size the fixup range from `nsm` through this query, never from a
//! literal, and the J replica selects its config table from `cc`.
//!
//! Not a launch: no stream, no allocation, no synchronization — the method only needs a
//! usable backend and the dispatch guard, the `device_is_integrated` shape.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no caller yet: the chunk driver does not call this kernel; drop the expect when it does"
    )
)]

use std::cell::OnceCell;

use crate::EngError;
use crate::kernels::cuda::context::CudaContext;
use crate::kernels::cuda::guard::DispatchGuard;
use crate::kernels::cuda::native::prefill::mmq_facts::mmq_facts;
use crate::kernels::cuda::prefill::mmq_size::fixup_floats_max;
use crate::kernels::cuda::{ADAPTER_OPERATION, TERMINAL, native_error};

/// The facts one launch needs, in Rust widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MmqFacts {
    /// The BUILT architecture (860 on the 3090, which runs the sm_86 cubins), not the
    /// device's own capability.
    pub(crate) cc: i32,
    /// Streaming multiprocessors.
    pub(crate) nsm: usize,
    /// Opt-in shared memory per block, in bytes.
    pub(crate) smpbo: usize,
}

/// The per-context cache of [`CudaContext::mmq_device_facts`]. The device and the built
/// architecture cannot change while the context holds its lease, so the first successful
/// query is kept for the life of the context (the context is thread-confined, hence the
/// non-atomic cell). Only a success is stored: a failed CUDA query latches the backend, so a
/// retry returns the terminal error, and the one other failure (a negative `nsm`) queries
/// again.
#[derive(Default)]
pub(in crate::kernels::cuda) struct MmqFactsCache {
    facts: OnceCell<MmqFacts>,
    /// Native queries made, failures included; tests compare it.
    #[cfg(test)]
    queries: std::cell::Cell<usize>,
}

impl CudaContext {
    /// Reads the MMQ device facts of this context's device through the C entry: the built
    /// architecture, the SM count and the opt-in shared memory per block. The GEMM wrappers
    /// size the fixup range and pick the J config from these. The device is queried once per
    /// context and every later call returns the cached facts (a launch reads them, so they
    /// must not cost five attribute calls each); a failed CUDA query is the backend's own
    /// error and latches there exactly as a launch failure would.
    pub(crate) fn mmq_device_facts(&self) -> Result<MmqFacts, EngError> {
        self.ensure_usable()?;
        if let Some(facts) = self.mmq_facts.facts.get() {
            return Ok(*facts);
        }
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        self.ensure_usable()?;
        #[cfg(test)]
        self.mmq_facts.queries.set(self.mmq_facts.queries.get() + 1);
        let facts = mmq_facts()
            .map_err(|failure| native_error(&TERMINAL, ADAPTER_OPERATION, failure.status))?;
        let facts = MmqFacts {
            cc: facts.cc,
            nsm: usize::try_from(facts.nsm).map_err(|_| {
                EngError::BadRequest("mmq_facts: the device facts hold a negative nsm".to_owned())
            })?,
            smpbo: facts.smpbo,
        };
        Ok(*self.mmq_facts.facts.get_or_init(|| facts))
    }
}

impl CudaContext {
    /// Bytes of the stream-k fixup range every MMQ launch on this device may need,
    /// `nsm * 128 * 128` floats (mmq.cuh:1334-1338, `mmq_size::fixup_floats_max`): the size an
    /// arena allocates once, from the runtime SM count and never from a literal. It reads the
    /// facts through [`Self::mmq_device_facts`], so after the first call it is a cache hit.
    pub(crate) fn mmq_fixup_bytes(&self) -> Result<usize, EngError> {
        fixup_floats_max(self.mmq_device_facts()?.nsm)?
            .checked_mul(4)
            .ok_or_else(|| {
                EngError::BadRequest("mmq_facts: the fixup byte size overflows".to_owned())
            })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
