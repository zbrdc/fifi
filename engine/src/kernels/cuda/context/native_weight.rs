//! Weights in the GGUF file's own block layout, resident on the device:
//! [`CudaContext::upload_native`] reads a tensor of any type with a native kernel
//! (`QuantType::from_gguf`) through a [`BlockReader`] in chunks of WHOLE ROWS and uploads each chunk at its byte offset into ONE
//! device allocation made up front. No repack runs: the bytes land as the file holds them,
//! row-major blocks, and the MMVQ kernels (`matvec_q`, `matvec_expert`) and the native
//! embedding gather read them there.
//!
//! A chunk holds as many rows as fit `chunk_limit` bytes and at least one, so a row wider than
//! the limit is its own chunk and the last chunk may be short; host memory stays at one chunk.
//! Rows may be ANY count from one up: the 8-row group rule is a tile rule, not a property of
//! the file. The fp16 scale scans of the tiled path do not apply (a native block keeps its
//! exact scales; the scans exist for tile dequantisation), so `upload_native` runs
//! none. A failed read or upload drops the device owner, as a failed tiled upload does.
//! Cleanup is skipped after terminal failure. Admission behaviour is the tiled path's.

use crate::EngError;
use crate::kernels::cuda::native;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use std::{marker::PhantomData, rc::Rc, sync::atomic::Ordering};

use super::{
    BlockReader, CudaContext, DeviceBuffer, DeviceEvent, DeviceStream, PinnedBuf, QuantTensor,
    host_source,
};

/// Two page-locked staging buffers, a stream and one event per buffer: the file read of the
/// next chunk runs while the DMA copy of the previous one is in flight, and the copy reads
/// page-locked memory (about twice the rate of a copy from pageable memory).
pub(super) struct UploadRing {
    stream: DeviceStream,
    slots: [RingSlot; 2],
    next: usize,
    capacity: usize,
}

struct RingSlot {
    buffer: PinnedBuf,
    copied: DeviceEvent,
    in_flight: bool,
}

impl UploadRing {
    fn new(context: &CudaContext, capacity: usize) -> Result<Self, EngError> {
        let slot = || -> Result<RingSlot, EngError> {
            Ok(RingSlot {
                buffer: context.pinned_buf(capacity)?,
                copied: context.event()?,
                in_flight: false,
            })
        };
        Ok(Self {
            stream: context.stream()?,
            slots: [slot()?, slot()?],
            next: 0,
            capacity,
        })
    }

    /// Waits for every copy still in flight; the buffers are free for the host afterwards.
    fn drain(&mut self) -> Result<(), EngError> {
        let mut result = Ok(());
        for slot in &mut self.slots {
            if slot.in_flight {
                slot.in_flight = false;
                if let Err(error) = slot.buffer.wait(&slot.copied) {
                    result = result.and(Err(error));
                }
            }
        }
        result
    }
}

/// A quantized weight in the GGUF file's own block layout, resident on the device: one
/// allocation of `n_rows` rows of `k / block_size` blocks each, `experts` the stack depth of a
/// rank-3 tensor (`None` for a plain matrix). The `matvec_q`/`matvec_expert` kernels, the f32
/// decode matvec and the native `embed` read it; the tiled readers refuse it with a typed error.
pub(crate) struct NativeWeight {
    buffer: DeviceBuffer,
    quant: QuantType,
    k: usize,
    n_rows: usize,
    experts: Option<usize>,
}

/// How a native upload splits into chunks of whole rows: `per_chunk` rows, at least one and as
/// many as fit `chunk_limit` bytes of source. No chunk splits a row; the last chunk may be
/// short.
#[derive(Clone, Copy)]
struct RowChunkPlan {
    rows: usize,
    per_chunk: usize,
}

impl RowChunkPlan {
    fn new(rows: usize, row_bytes: usize, chunk_limit: usize) -> Self {
        Self {
            rows,
            per_chunk: (chunk_limit / row_bytes).clamp(1, rows),
        }
    }

    /// Rows of the chunk that starts at row `first` (at least one).
    fn count_of(&self, first: usize) -> usize {
        self.per_chunk.min(self.rows - first)
    }
}

impl CudaContext {
    /// Upload immutable native blocks into host pages mapped to this device.
    /// The mapping remains private to the initializer until all reads succeed.
    pub(crate) fn upload_native_mapped(
        &self,
        tensor: QuantTensor<'_>,
        chunk_limit: usize,
        read: BlockReader<'_>,
    ) -> Result<NativeWeight, EngError> {
        self.ensure_usable()?;
        super::graph::ensure_closed("a mapped native upload")?;
        let quant = checked_native(tensor)?;
        let row_bytes = quant.weight_bytes(1, tensor.k).ok_or_else(|| {
            EngError::Checkpoint(format!("tensor {} has an invalid row", tensor.name))
        })?;
        let plan = RowChunkPlan::new(tensor.n_rows, row_bytes, chunk_limit);
        let raw = {
            let _dispatch = self.enter()?;
            native::resident::mailbox::UnpublishedMapped::allocate(tensor.source_len)
                .map_err(super::device_error)?
        };
        super::LIVE_PINNED.fetch_add(tensor.source_len, Ordering::AcqRel);
        let mut staging = MappedStaging {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            bytes: tensor.source_len,
        };
        staging.raw.as_mut().expect("mapping").initialize(|host| {
            for first in (0..plan.rows).step_by(plan.per_chunk) {
                let at = first * row_bytes;
                let len = plan.count_of(first) * row_bytes;
                read(at, &mut host[at..at + len])?;
            }
            Ok::<(), EngError>(())
        })?;
        let buffer = DeviceBuffer {
            raw: None,
            mapped: Some(staging.raw.take().expect("mapping").publish()),
            lease: Rc::clone(&self.lease),
            _thread_confined: PhantomData,
            alloc_cache: None,
        };
        Ok(NativeWeight {
            buffer,
            quant,
            k: tensor.k,
            n_rows: tensor.n_rows,
            experts: None,
        })
    }

    /// Runs `run` on the page-locked ring (made, or widened, to hold `capacity` bytes) and
    /// drains it afterwards, so that no copy still reads a staging buffer or writes the device
    /// allocation when this returns. `Ok(None)` when page-locked memory is not available, with
    /// `run` not called.
    pub(super) fn with_upload_ring<R>(
        &self,
        capacity: usize,
        run: impl FnOnce(&mut UploadRing) -> Result<R, EngError>,
    ) -> Result<Option<R>, EngError> {
        let mut held = self.upload_ring.borrow_mut();
        if held.as_ref().is_some_and(|ring| ring.capacity < capacity) {
            // A wider chunk than the ring was made for: the old ring goes first.
            if let Some(mut old) = held.take() {
                old.drain()?;
            }
        }
        if held.is_none() {
            match UploadRing::new(self, capacity) {
                Ok(ring) => *held = Some(ring),
                Err(EngError::MemoryPlan(_)) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        let Some(ring) = held.as_mut() else {
            return Ok(None);
        };
        let result = run(ring);
        let drained = ring.drain();
        result.and_then(|value| drained.map(|()| Some(value)))
    }

    /// Uploads the chunks of `plan` into `buffer` through the page-locked ring, reading chunk
    /// n + 1 from the file while chunk n is copied. `Ok(false)` when the ring cannot be had
    /// (page-locked memory is not available), with nothing written.
    fn upload_through_ring(
        &self,
        buffer: &DeviceBuffer,
        plan: &RowChunkPlan,
        row_bytes: usize,
        chunk_bytes: usize,
        read: BlockReader<'_>,
    ) -> Result<bool, EngError> {
        let uploaded = self.with_upload_ring(chunk_bytes, |ring| {
            for first in (0..plan.rows).step_by(plan.per_chunk) {
                let bytes = plan.count_of(first) * row_bytes;
                let offset = first * row_bytes;
                self.stage_chunk(ring, buffer, offset, bytes, |part| read(offset, part))?;
            }
            Ok(())
        })?;
        Ok(uploaded.is_some())
    }

    /// Fills the next staging buffer of `ring` through `fill` (once its previous copy is done)
    /// and queues its copy to `buffer` at `offset`.
    pub(super) fn stage_chunk(
        &self,
        ring: &mut UploadRing,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let slot = &mut ring.slots[ring.next];
        ring.next ^= 1;
        if slot.in_flight {
            slot.in_flight = false;
            slot.buffer.wait(&slot.copied)?;
        }
        slot.buffer.fill(bytes, fill)?;
        if let Some(cache) = &buffer.alloc_cache {
            cache.settle()?;
        }
        self.copy_from_pinned(
            &ring.stream,
            &mut slot.buffer,
            0,
            buffer.range(offset, bytes)?,
        )?;
        ring.stream.record(&slot.copied)?;
        slot.in_flight = true;
        Ok(())
    }

    /// Reads a tensor of any type with a native kernel (`QuantType::from_gguf`) through `read` in
    /// chunks of whole rows and
    /// uploads the file's own blocks into one device allocation made up front.
    ///
    /// A chunk holds as many whole rows as fit `chunk_limit` bytes and at least one, however
    /// wide the row; the last chunk may be short. Returns `Checkpoint` naming the tensor for
    /// another type, a `k` off the type's block grid or above the matvec limit, a row count of
    /// zero, or a wrong source length; `MemoryPlan` when the device cannot hold the blocks or
    /// the host the chunk. An error from `read` is returned unchanged. A failed read or
    /// upload drops the device owner. Cleanup is skipped after terminal failure.
    pub(crate) fn upload_native(
        &self,
        tensor: QuantTensor<'_>,
        chunk_limit: usize,
        read: BlockReader<'_>,
    ) -> Result<NativeWeight, EngError> {
        self.ensure_usable()?;
        super::graph::ensure_closed("a native upload")?;
        let quant = checked_native(tensor)?;
        let name = tensor.name;
        // Unreachable in practice (`checked_native` proved k on the block grid), still a
        // typed error: the load path never panics.
        let row_bytes = quant.weight_bytes(1, tensor.k).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "tensor {name} has {} values, which the native block grid cannot hold",
                tensor.k
            ))
        })?;
        let plan = RowChunkPlan::new(tensor.n_rows, row_bytes, chunk_limit);
        let buffer = self.alloc(tensor.source_len)?;
        let raw = self.owned(&buffer)?;
        let chunk_bytes = plan.per_chunk * row_bytes;
        if !self.upload_through_ring(&buffer, &plan, row_bytes, chunk_bytes, read)? {
            // No page-locked memory to be had: the chunks go through one pageable buffer.
            let mut source = host_source(name, chunk_bytes)?;
            for first in (0..plan.rows).step_by(plan.per_chunk) {
                let part = &mut source[..plan.count_of(first) * row_bytes];
                read(first * row_bytes, part)?;
                let _dispatch = self.enter()?;
                super::host_write(&buffer, raw, first * row_bytes, part)?;
            }
        }
        Ok(NativeWeight {
            buffer,
            quant,
            k: tensor.k,
            n_rows: tensor.n_rows,
            experts: None,
        })
    }
}

/// Frees a mapping on read failure without ever exposing its device pointer.
struct MappedStaging {
    raw: Option<native::resident::mailbox::UnpublishedMapped>,
    lease: Rc<super::Lease>,
    bytes: usize,
}

impl Drop for MappedStaging {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            let lease = Rc::clone(&self.lease);
            let bytes = self.bytes;
            super::graph::run_or_defer(move || {
                let device = lease.device();
                let _lease = lease;
                super::stream::release(
                    raw.publish(),
                    native::resident::mailbox::mapped_free,
                    &super::LIVE_PINNED,
                    bytes,
                    device,
                );
            });
        }
    }
}

impl NativeWeight {
    /// A weight over a device buffer the caller fills with the file's native blocks, for a
    /// stack that is staged into the buffer after the weight exists. The buffer must hold
    /// exactly the bytes of `n_rows` rows of `k` values.
    pub(crate) fn from_buffer(
        buffer: DeviceBuffer,
        quant: QuantType,
        k: usize,
        n_rows: usize,
        experts: Option<usize>,
    ) -> Result<Self, EngError> {
        let want = quant.weight_bytes(n_rows, k);
        if want != Some(buffer.bytes()) {
            return Err(EngError::BadRequest(format!(
                "a native weight of {n_rows} rows of {k} values holds {want:?} bytes, the buffer has {}",
                buffer.bytes()
            )));
        }
        Ok(Self {
            buffer,
            quant,
            k,
            n_rows,
            experts,
        })
    }

    /// Rows (output values per token) of the weight: the TOTAL row count, which for a
    /// rank-3 stack is `experts` consecutive matrices of `n_rows / experts` rows each
    /// (the depth rides `experts()`).
    pub(crate) fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// The stack depth of a rank-3 tensor, `None` for a plain matrix. Read by the resident
    /// step's native stack lookups.
    pub(crate) fn experts(&self) -> Option<usize> {
        self.experts
    }

    pub(crate) fn quant(&self) -> QuantType {
        self.quant
    }

    /// The device blocks, for launches and parity checks.
    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    /// The stored bytes of the weight.
    #[cfg(feature = "native-cuda")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read by the SenseNova decode trace rig")
    )]
    pub(crate) fn bytes(&self) -> usize {
        self.buffer.bytes()
    }

    /// Records the stack depth of a rank-3 tensor (`None` for a plain matrix). The upload
    /// reads only the tensor's flattened shape, so the loader writes the depth it read from
    /// the GGUF dims here.
    pub(crate) fn with_experts(mut self, experts: Option<usize>) -> Self {
        self.experts = experts;
        self
    }
}

/// A weight held as the file's native blocks: what the native GEMMs read of it.
pub(crate) trait NativeBlocks {
    fn quant(&self) -> QuantType;
    /// Values per row.
    fn k(&self) -> usize;
    /// Rows: output values per token.
    fn n_rows(&self) -> usize;
    /// The device blocks, row after row from the buffer start.
    fn buffer(&self) -> &DeviceBuffer;
}

impl NativeBlocks for NativeWeight {
    fn quant(&self) -> QuantType {
        self.quant
    }
    fn k(&self) -> usize {
        self.k
    }
    fn n_rows(&self) -> usize {
        self.n_rows
    }
    fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }
}

/// Native blocks in a buffer another owner keeps, for a family whose weights rest in its own
/// device buffers.
#[derive(Clone, Copy)]
pub(crate) struct BorrowedNativeWeight<'a> {
    buffer: &'a DeviceBuffer,
    quant: QuantType,
    k: usize,
    n_rows: usize,
}

impl<'a> BorrowedNativeWeight<'a> {
    /// The first `n_rows` rows of `k` values of `quant` at the start of `buffer`, which must
    /// hold at least their bytes.
    pub(crate) fn new(
        buffer: &'a DeviceBuffer,
        quant: QuantType,
        k: usize,
        n_rows: usize,
    ) -> Result<Self, EngError> {
        match quant.weight_bytes(n_rows, k) {
            Some(want) if want <= buffer.bytes() => Ok(Self {
                buffer,
                quant,
                k,
                n_rows,
            }),
            want => Err(EngError::BadRequest(format!(
                "a native weight of {n_rows} rows of {k} values holds {want:?} bytes, the buffer has {}",
                buffer.bytes()
            ))),
        }
    }
}

impl NativeBlocks for BorrowedNativeWeight<'_> {
    fn quant(&self) -> QuantType {
        self.quant
    }
    fn k(&self) -> usize {
        self.k
    }
    fn n_rows(&self) -> usize {
        self.n_rows
    }
    fn buffer(&self) -> &DeviceBuffer {
        self.buffer
    }
}

/// Checks the tensor of a native upload: the type must be one with a native kernel
/// (`QuantType::from_gguf`), `k` a positive multiple of the type's block size up to `native::MAX_DIM`, `n_rows` at
/// least one (any count: no 8-row rule), and the source length exactly the native byte count.
fn checked_native(tensor: QuantTensor<'_>) -> Result<QuantType, EngError> {
    let Some(quant) = QuantType::from_gguf(tensor.ty) else {
        return Err(EngError::Checkpoint(format!(
            "tensor {} has type {:?}, which the CUDA native weights do not take",
            tensor.name, tensor.ty
        )));
    };
    let shape = |why: &str| {
        EngError::Checkpoint(format!(
            "tensor {} has {} rows of {} values, which the native layout cannot hold: {why}",
            tensor.name, tensor.n_rows, tensor.k
        ))
    };
    if tensor.n_rows == 0 {
        return Err(shape("the row count is zero"));
    }
    if tensor.k == 0 || !tensor.k.is_multiple_of(quant.block_size()) {
        return Err(shape("k is off the type's block grid"));
    }
    if tensor.k > native::MAX_DIM {
        return Err(shape("k is above the matvec limit"));
    }
    if quant.weight_bytes(tensor.n_rows, tensor.k) != Some(tensor.source_len) {
        return Err(EngError::Checkpoint(format!(
            "tensor {} has {} bytes of blocks for {} rows of {} values",
            tensor.name, tensor.source_len, tensor.n_rows, tensor.k
        )));
    }
    Ok(quant)
}

#[cfg(test)]
mod tests;
