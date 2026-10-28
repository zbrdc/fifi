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
//! - Pageable copies use owned chunks of at most 64 MiB. A failed fence retains
//!   that storage until process exit; a failed D2H chunk never reaches the caller.

use super::graph::ensure_closed;
use super::stream::{DeviceStream, same_adapter};
use super::{CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceRange, device_error, enter_dispatch};
use crate::EngError;
use crate::kernels::cuda::native::DeviceFailure;
use crate::kernels::cuda::native::resident::{self, RawRange, RawStream};
use crate::kernels::cuda::{TERMINAL, ensure_usable};

/// Successful public allocation and ranged copy requests, not physical CUDA calls.
/// Uploads and other copy paths are outside this test-only observer.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct BufferRequests {
    pub(crate) allocations: std::collections::BTreeMap<usize, usize>,
    pub(crate) ranged_h2d: std::collections::BTreeMap<usize, usize>,
    pub(crate) ranged_d2h: std::collections::BTreeMap<usize, usize>,
}

#[cfg(test)]
thread_local! {
    static TEST_HEALTH_FAULT: std::cell::RefCell<std::collections::BTreeMap<usize, (usize, u32)>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
    static SCALAR_READ_REQUESTS: std::cell::RefCell<std::collections::BTreeMap<usize, usize>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
    static BUFFER_REQUESTS: std::cell::RefCell<std::collections::BTreeMap<usize, BufferRequests>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

#[cfg(test)]
fn record_allocation(lease: usize, bytes: usize) {
    BUFFER_REQUESTS.with_borrow_mut(|counts| {
        *counts
            .entry(lease)
            .or_default()
            .allocations
            .entry(bytes)
            .or_default() += 1;
    });
}

#[cfg(test)]
fn record_ranged_h2d(lease: usize, bytes: usize) {
    BUFFER_REQUESTS.with_borrow_mut(|counts| {
        *counts
            .entry(lease)
            .or_default()
            .ranged_h2d
            .entry(bytes)
            .or_default() += 1;
    });
}

#[cfg(test)]
fn take_health_fault(lease: usize, reads: usize) -> Option<(usize, u32)> {
    let current =
        SCALAR_READ_REQUESTS.with_borrow(|counts| counts.get(&lease).copied().unwrap_or(0));
    TEST_HEALTH_FAULT.with_borrow_mut(|faults| {
        let &(target, status) = faults.get(&lease)?;
        let distance = target.checked_sub(current)?;
        if distance == 0 || distance > reads {
            return None;
        }
        faults.remove(&lease);
        Some((distance - 1, status))
    })
}

impl CudaContext {
    #[cfg(test)]
    pub(crate) fn test_poison_scalar_read_after(&self, skip: usize, status: u32) {
        let lease = std::rc::Rc::as_ptr(&self.lease) as usize;
        let target = self
            .scalar_read_requests()
            .checked_add(skip)
            .unwrap()
            .checked_add(1)
            .unwrap();
        TEST_HEALTH_FAULT.with_borrow_mut(|faults| {
            assert!(
                faults.insert(lease, (target, status)).is_none(),
                "unconsumed health fault"
            );
        });
    }

    #[cfg(test)]
    pub(crate) fn test_take_health_fault(&self, reads: usize) -> Option<(usize, u32)> {
        take_health_fault(std::rc::Rc::as_ptr(&self.lease) as usize, reads)
    }

    #[cfg(test)]
    pub(crate) fn test_health_fault_pending(&self) -> bool {
        TEST_HEALTH_FAULT
            .with_borrow(|faults| faults.contains_key(&(std::rc::Rc::as_ptr(&self.lease) as usize)))
    }

    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn buffer_requests(&self) -> BufferRequests {
        BUFFER_REQUESTS.with_borrow(|counts| {
            counts
                .get(&(std::rc::Rc::as_ptr(&self.lease) as usize))
                .cloned()
                .unwrap_or_default()
        })
    }

    #[cfg(test)]
    pub(crate) fn scalar_read_requests(&self) -> usize {
        SCALAR_READ_REQUESTS.with_borrow(|counts| {
            counts
                .get(&(std::rc::Rc::as_ptr(&self.lease) as usize))
                .copied()
                .unwrap_or(0)
        })
    }

    /// Queues equal byte ranges between distinct allocations, without waiting or capturing.
    pub(crate) fn copy_device(
        &self,
        stream: &DeviceStream,
        source: DeviceRange<'_>,
        destination: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        ensure_closed("a device copy")?;
        if source.bytes() != destination.bytes() {
            return Err(EngError::BadRequest(
                "device copies need equal byte ranges".into(),
            ));
        }
        if std::ptr::eq(source.buffer(), destination.buffer()) {
            return Err(EngError::BadRequest(
                "device copies need distinct allocations".into(),
            ));
        }
        let source = self.raw_range(&source)?;
        let destination = self.raw_range(&destination)?;
        self.launch_on(stream, |raw| {
            resident::copy_d2d_async(source, destination, raw)
        })
    }

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
        let buffer = self.alloc(bytes)?;
        #[cfg(test)]
        record_allocation(std::rc::Rc::as_ptr(&self.lease) as usize, bytes);
        Ok(buffer)
    }
}

impl DeviceBuffer {
    /// Copies `out.len()` bytes from byte `offset` of the buffer into `out`,
    /// behind the work queued on `stream`, and returns when `out` holds them.
    /// On failure, only fully completed chunks may have been written to `out`.
    ///
    /// Returns `BadRequest` for an empty `out`, a range past the buffer, a
    /// stream of another adapter or a buffer that was already freed.
    pub(crate) fn read_at(
        &self,
        stream: &DeviceStream,
        offset: usize,
        out: &mut [u8],
    ) -> Result<(), EngError> {
        self.io_range(stream, "read", offset, out.len())?;
        #[cfg(test)]
        if out.len() == 4
            && let Some((_, status)) =
                take_health_fault(std::rc::Rc::as_ptr(&self.lease) as usize, 1)
        {
            self.write_at(stream, offset, &status.to_ne_bytes())?;
        }
        let mut staging = pageable_staging(out.len())?;
        for (index, chunk) in out.chunks_mut(CHUNK_LIMIT).enumerate() {
            staging.resize(chunk.len(), 0);
            let range = self.io_range(stream, "read", offset + index * CHUNK_LIMIT, chunk.len())?;
            staging = run_ordered(stream, |raw| {
                resident::copy_d2h_pageable(range, staging, raw)
            })?;
            chunk.copy_from_slice(&staging);
        }
        #[cfg(test)]
        BUFFER_REQUESTS.with_borrow_mut(|counts| {
            *counts
                .entry(std::rc::Rc::as_ptr(&self.lease) as usize)
                .or_default()
                .ranged_d2h
                .entry(out.len())
                .or_default() += 1;
        });
        #[cfg(test)]
        if out.len() == 4 {
            SCALAR_READ_REQUESTS.with_borrow_mut(|counts| {
                *counts
                    .entry(std::rc::Rc::as_ptr(&self.lease) as usize)
                    .or_default() += 1;
            });
        }
        Ok(())
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
        self.io_range(stream, "write", offset, bytes.len())?;
        let mut staging = pageable_staging(bytes.len())?;
        for (index, chunk) in bytes.chunks(CHUNK_LIMIT).enumerate() {
            staging.clear();
            staging.extend_from_slice(chunk);
            let range =
                self.io_range(stream, "write", offset + index * CHUNK_LIMIT, chunk.len())?;
            staging = run_ordered(stream, |raw| {
                resident::copy_h2d_pageable(range, staging, raw)
            })?;
        }
        #[cfg(test)]
        record_ranged_h2d(std::rc::Rc::as_ptr(&self.lease) as usize, bytes.len());
        Ok(())
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

/// Two adapter leases need at most 128 MiB of staging, including retained storage.
/// Allocation is fallible, and caller-borrowed host pointers never reach CUDA.
fn pageable_staging(bytes: usize) -> Result<Vec<u8>, EngError> {
    let bytes = bytes.min(CHUNK_LIMIT);
    let mut staging = Vec::new();
    staging.try_reserve_exact(bytes).map_err(|error| {
        EngError::MemoryPlan(format!("cuda allocate: pageable copy staging: {error}"))
    })?;
    staging.resize(bytes, 0);
    Ok(staging)
}

/// The native Rust wrapper owns the bytes through copy and fence, even on error.
fn run_ordered(
    stream: &DeviceStream,
    copy: impl FnOnce(&RawStream) -> Result<Vec<u8>, DeviceFailure>,
) -> Result<Vec<u8>, EngError> {
    let raw = stream.raw()?;
    let _dispatch = enter_dispatch()?;
    let completed = copy(raw).map_err(device_error)?;
    stream.note_enqueue();
    Ok(completed)
}

#[cfg(test)]
mod tests;
