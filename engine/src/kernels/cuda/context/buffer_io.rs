//! The allocate entry and the stream-ordered ranged reads and writes of a device
//! buffer (U3 and U4 of the resident step; decision 0047).
//!
//! The resident step keeps its state in buffers it allocates itself and moves
//! bytes in and out of them between stream work. Invariants:
//!
//! - [`CudaContext::allocate`] makes an uninitialized allocation that shares the
//!   lease of the context and counts in the live bytes like every buffer.
//! - [`DeviceBuffer::read_at`] and [`DeviceBuffer::write_at`] take the stream that
//!   owns the work on the buffer. The copy is queued on that stream, so it runs
//!   after everything queued there earlier, and the call synchronizes that stream
//!   before it returns. The order is part of the call, not a rule for the caller:
//!   a kernel queued on the stream that writes the buffer is finished before a
//!   `read_at` on the same stream returns its bytes, and a `write_at` lands after
//!   the queued work that precedes it.
//! - Work queued on another stream is not waited for. A caller that moves bytes
//!   between streams joins them with events first ([`super::stream`]).
//! - Both calls are blocking, run inside the dispatch guard, and need no context:
//!   a buffer owns its lease, and the stream must hold the same one.
//! - Every range is proved in bounds here, before any native call, so a caller
//!   mistake is a `BadRequest` and never a terminal adapter failure;
//!   `native::resident` checks it again.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use super::graph::ensure_closed;
use super::stream::{DeviceStream, same_adapter};
use super::{CudaContext, DeviceBuffer, device_error, enter_dispatch};
use crate::EngError;
use crate::kernels::cuda::native::DeviceFailure;
use crate::kernels::cuda::native::resident::{self, RawRange, RawStream};
use crate::kernels::cuda::{TERMINAL, ensure_usable};

impl CudaContext {
    /// Allocates `bytes` of uninitialized device memory that shares this context's lease.
    ///
    /// Returns `BadRequest` for zero bytes (the adapter would reject it with a
    /// status that latches the backend) and `MemoryPlan` when the device cannot
    /// hold the bytes.
    pub(crate) fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        self.ensure_usable()?;
        if bytes == 0 {
            return Err(EngError::BadRequest(
                "cuda allocate of zero bytes".to_owned(),
            ));
        }
        self.alloc(bytes)
    }
}

impl DeviceBuffer {
    /// Copies `out.len()` bytes from byte `offset` of the buffer into `out`,
    /// behind the work queued on `stream`, and returns when `out` holds them.
    ///
    /// Returns `BadRequest` for an empty `out`, a range past the buffer, a
    /// stream of another adapter or a buffer that was already freed.
    pub(crate) fn read_at(
        &self,
        stream: &DeviceStream,
        offset: usize,
        out: &mut [u8],
    ) -> Result<(), EngError> {
        let range = self.io_range(stream, "read", offset, out.len())?;
        run_ordered(stream, |raw| resident::copy_d2h_pageable(range, out, raw))
    }

    /// Copies `bytes` into the buffer at byte `offset`, behind the work queued
    /// on `stream`, and returns when the copy finished.
    ///
    /// Returns `BadRequest` for the conditions of [`Self::read_at`].
    pub(crate) fn write_at(
        &self,
        stream: &DeviceStream,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), EngError> {
        let range = self.io_range(stream, "write", offset, bytes.len())?;
        run_ordered(stream, |raw| resident::copy_h2d_async(range, bytes, raw))
    }

    /// The checks both calls share, then the native view of the range.
    fn io_range(
        &self,
        stream: &DeviceStream,
        what: &str,
        offset: usize,
        len: usize,
    ) -> Result<RawRange<'_>, EngError> {
        ensure_usable(&TERMINAL)?;
        ensure_closed("a ranged copy")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        let device = self
            .raw
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the CUDA buffer was already freed".to_owned()))?;
        if len == 0 {
            return Err(EngError::BadRequest(format!(
                "the CUDA {what} of an empty slice"
            )));
        }
        // `RawRange::new` refuses a range past the end, including an offset sum
        // that overflows; `len` is not 0 here, so `None` means "outside".
        RawRange::new(device, offset, len).ok_or_else(|| {
            EngError::BadRequest(format!(
                "the CUDA {what} of {len} bytes at offset {offset} lies outside a buffer of {} bytes",
                device.bytes()
            ))
        })
    }
}

/// Queues one copy on `stream`, counts it there, then waits for the stream.
///
/// The dispatch guard is released before the wait, because the synchronize
/// enters it again.
fn run_ordered(
    stream: &DeviceStream,
    copy: impl FnOnce(&RawStream) -> Result<(), DeviceFailure>,
) -> Result<(), EngError> {
    let raw = stream.raw()?;
    {
        let _dispatch = enter_dispatch()?;
        copy(raw).map_err(device_error)?;
        stream.note_enqueue();
    }
    stream.synchronize()
}

#[cfg(test)]
mod tests;
