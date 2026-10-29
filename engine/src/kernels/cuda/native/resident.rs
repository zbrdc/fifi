//! Raw FFI to the resident-decode entries of `native/stream.cu` and `native/projector.cu`
//! (declared in `native/abi_resident.h`). Child module of `native`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants every wrapper here relies on:
//! - A handle (`RawStream`, `RawEvent`, `RawPinned`) is created once and consumed once by its
//!   destroy function. It has no `Drop`: the owner in `context` calls destroy so that a failed
//!   destroy is reported and latches the backend.
//! - A device range is proved in bounds before the native call (`RawRange::new`). The launch
//!   wrappers (`pack_launch`, `matmul_launch`) also prove sizes, alignment and, between the ranges
//!   one call writes and the ranges it reads, disjointness; the native entries repeat these checks
//!   on the addresses. The allocation stays valid until queued work finishes because
//!   `inference_cuda_free` synchronizes the whole device before `cudaFree`.
//! - Launch-only wrappers return the launch status only. A fault in queued work surfaces at
//!   the next synchronize.
//! - Every call runs under the process dispatch guard and on the thread that owns the context.
//!   Pinned memory is read only by that thread and only after an event recorded after the copy
//!   completed.
//! - Stream capture and executable graphs (`RawGraphExec`) live in the child `graph`, which
//!   states its own invariants: a capture begins and ends on one thread, and a graph is consumed
//!   once.
//! - No raw pointer and no unsafe function leaves this module; handles are opaque.
//!
//! Per-kernel entries live in child modules `native/resident/<n>.rs`, one per kernel, declared
//! by one line after the imports below. A child starts with its own safety-invariants doc and may
//! use this module's private items (`RawStream`, `RawRange`, `checked`, `Span`). The recipe is in
//! the doc of `kernels::cuda::resident`.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};
use std::ptr::NonNull;

#[cfg(all(test, feature = "cuda-fault-qualification"))]
static RETAINED_PAGEABLE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn retained_pageable_bytes() -> usize {
    RETAINED_PAGEABLE_BYTES.load(std::sync::atomic::Ordering::Acquire)
}

use super::{
    DeviceFailure, DeviceView, MAX_DIM, RawDevice, STAGE_NONE, TILE_COLS, Tiled, activation_bytes,
    checked, invalid_argument, pack_bytes,
};

// One `pub(in crate::kernels::cuda) mod <n>;` line per kernel child module, alphabetical.
pub(in crate::kernels::cuda) mod add_norm_pack;
pub(in crate::kernels::cuda) mod add_rmsnorm_q8;
pub(in crate::kernels::cuda) mod attn_decode;
pub(in crate::kernels::cuda) mod attn_decode_gemma;
pub(in crate::kernels::cuda) mod attn_decode_q8;
pub(in crate::kernels::cuda) mod attn_prep;
pub(in crate::kernels::cuda) mod embed;
pub(in crate::kernels::cuda) mod embed_q8;
pub(in crate::kernels::cuda) mod exl3;
pub(in crate::kernels::cuda) mod expert_activation;
pub(in crate::kernels::cuda) mod gdn_conv;
pub(in crate::kernels::cuda) mod gdn_delta;
pub(in crate::kernels::cuda) mod gelu_ffn;
pub(in crate::kernels::cuda) mod gemma_matvec;
pub(in crate::kernels::cuda) mod graph;
pub(in crate::kernels::cuda) mod kv8_append;
pub(in crate::kernels::cuda) mod mailbox;
pub(in crate::kernels::cuda) mod matmul_expert;
pub(in crate::kernels::cuda) mod matmul_q5_0;
pub(in crate::kernels::cuda) mod matvec_expert;
pub(in crate::kernels::cuda) mod matvec_q;
pub(in crate::kernels::cuda) mod moe_combine;
pub(in crate::kernels::cuda) mod paged_position;
pub(in crate::kernels::cuda) mod quantize_q8_1;
pub(in crate::kernels::cuda) mod rope_gemma;
pub(in crate::kernels::cuda) mod router;
pub(in crate::kernels::cuda) mod row_add;
pub(in crate::kernels::cuda) mod row_scale;
pub(in crate::kernels::cuda) mod sample_gemma;
pub(in crate::kernels::cuda) mod shortlist;
pub(in crate::kernels::cuda) mod softcap;
pub(in crate::kernels::cuda) mod step_params;
pub(in crate::kernels::cuda) mod swiglu_pack;
pub(in crate::kernels::cuda) mod swiglu_q8;
pub(in crate::kernels::cuda) mod tiered_classify;
pub(in crate::kernels::cuda) mod tiered_wait_add;

/// A non-blocking CUDA stream. Consumed by [`stream_destroy`].
#[derive(Debug)]
#[must_use = "a stream must be passed to stream_destroy"]
pub(in crate::kernels::cuda) struct RawStream(NonNull<c_void>);

impl RawStream {
    /// The raw handle, for the launch entries of the kernel child modules.
    pub(in crate::kernels::cuda) fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

/// A timing-free CUDA event. Consumed by [`event_destroy`].
#[derive(Debug)]
#[must_use = "an event must be passed to event_destroy"]
pub(in crate::kernels::cuda) struct RawEvent(NonNull<c_void>);

/// Page-locked host memory of `bytes` bytes. Consumed by [`pinned_free`].
#[derive(Debug)]
#[must_use = "pinned memory must be passed to pinned_free"]
pub(in crate::kernels::cuda) struct RawPinned {
    ptr: NonNull<c_void>,
    bytes: usize,
}

/// `bytes` bytes of a live device allocation from `offset` on.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct RawRange<'a> {
    device: &'a RawDevice,
    offset: usize,
    bytes: usize,
}

impl<'a> RawRange<'a> {
    /// The range, or `None` when it is empty or ends past the allocation.
    pub(in crate::kernels::cuda) fn new(
        device: &'a RawDevice,
        offset: usize,
        bytes: usize,
    ) -> Option<Self> {
        let end = offset.checked_add(bytes)?;
        (bytes > 0 && end <= device.bytes).then_some(Self {
            device,
            offset,
            bytes,
        })
    }
}

impl RawRange<'_> {
    /// The byte offset of the range inside its allocation. Read-only: `address` relies on
    /// `RawRange::new`'s bound proof, so the fields stay private.
    pub(in crate::kernels::cuda) fn offset(&self) -> usize {
        self.offset
    }

    /// The length of the range in bytes.
    pub(in crate::kernels::cuda) fn bytes(&self) -> usize {
        self.bytes
    }

    /// The device address of the first byte of the range.
    pub(in crate::kernels::cuda) fn address(&self) -> *mut c_void {
        // SAFETY: `RawRange::new` proved `offset + bytes <= device.bytes` with
        // `bytes > 0`, so the offset is inside the live allocation.
        unsafe {
            self.device
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(self.offset)
                .cast()
        }
    }
}

unsafe extern "C" {
    #[link_name = "inference_cuda_stream_create"]
    fn stream_create_raw(out_stream: *mut *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_stream_destroy"]
    fn stream_destroy_raw(stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_stream_synchronize"]
    fn stream_synchronize_raw(stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_event_create"]
    fn event_create_raw(out_event: *mut *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_event_destroy"]
    fn event_destroy_raw(event: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_event_record"]
    fn event_record_raw(event: *mut c_void, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_timing_event_create"]
    fn event_create_timing_raw(out_event: *mut *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_timing_event_record_external"]
    fn event_record_external_raw(
        event: *mut c_void,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_timing_event_elapsed_ms"]
    fn event_elapsed_ms_raw(
        start: *mut c_void,
        end: *mut c_void,
        out_ms: *mut f32,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_stream_wait_event"]
    fn stream_wait_event_raw(
        stream: *mut c_void,
        event: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_event_synchronize"]
    fn event_synchronize_raw(event: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_pinned_alloc"]
    fn pinned_alloc_raw(bytes: usize, out_host: *mut *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_pinned_free"]
    fn pinned_free_raw(host: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_upload_pinned"]
    fn upload_pinned_raw(
        device: *mut c_void,
        device_bytes: usize,
        pinned: *mut c_void,
        pinned_bytes: usize,
        host: *const c_void,
        bytes: usize,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_copy_d2h_async"]
    fn copy_d2h_async_raw(
        host_pinned: *mut c_void,
        device: *const c_void,
        bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_copy_h2d_async"]
    fn copy_h2d_async_raw(
        device: *mut c_void,
        host: *const c_void,
        bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_copy_d2d_async"]
    fn copy_d2d_async_raw(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_pack_launch"]
    fn pack_launch_raw(
        x: *const f32,
        m: c_int,
        k: c_int,
        xf: *mut c_void,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_matmul_launch"]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the C declaration in abi_resident.h"
    )]
    fn matmul_launch_raw(
        ggml_type: c_int,
        m: c_int,
        weight: *const c_void,
        xf: *const c_void,
        y: *mut f32,
        ldy: c_int,
        n_rows: c_int,
        k: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Creates a non-blocking stream.
pub(in crate::kernels::cuda) fn stream_create() -> Result<RawStream, DeviceFailure> {
    let mut handle: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `handle` and `stage` are live, writable values for the call. The
    // adapter writes a handle only on success.
    let status = unsafe { stream_create_raw(&mut handle, &mut stage) };
    checked(status, stage)?;
    NonNull::new(handle)
        .map(RawStream)
        .ok_or_else(invalid_argument)
}

/// Synchronizes, then destroys the stream. When the synchronize fails nothing
/// is destroyed and the failure is returned.
pub(in crate::kernels::cuda) fn stream_destroy(stream: RawStream) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` came from `stream_create` and is consumed here, so it is
    // destroyed at most once and no other handle refers to it afterwards.
    let status = unsafe { stream_destroy_raw(stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Blocks until all work queued on the stream has finished.
pub(in crate::kernels::cuda) fn stream_synchronize(
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` is a live handle for the whole call; `stage` is a live,
    // writable c_int.
    let status = unsafe { stream_synchronize_raw(stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Creates an event with timing disabled.
pub(in crate::kernels::cuda) fn event_create() -> Result<RawEvent, DeviceFailure> {
    let mut handle: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `handle` and `stage` are live, writable values for the call. The
    // adapter writes a handle only on success.
    let status = unsafe { event_create_raw(&mut handle, &mut stage) };
    checked(status, stage)?;
    NonNull::new(handle)
        .map(RawEvent)
        .ok_or_else(invalid_argument)
}

/// Destroys the event without waiting for it.
pub(in crate::kernels::cuda) fn event_destroy(event: RawEvent) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `event` came from `event_create` and is consumed here, so it is
    // destroyed at most once. CUDA releases an event that is still recorded
    // when its work completes.
    let status = unsafe { event_destroy_raw(event.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Records the event on the stream after the work queued so far.
pub(in crate::kernels::cuda) fn event_record(
    event: &RawEvent,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: both handles are live for the whole call and `stage` is a live,
    // writable c_int. The call only queues the record.
    let status = unsafe { event_record_raw(event.0.as_ptr(), stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Creates a timing-enabled event (the graph timer).
pub(in crate::kernels::cuda) fn event_create_timing() -> Result<RawEvent, DeviceFailure> {
    let mut handle: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `handle` and `stage` are live, writable values for the call. The
    // adapter writes a handle only on success.
    let status = unsafe { event_create_timing_raw(&mut handle, &mut stage) };
    checked(status, stage)?;
    NonNull::new(handle)
        .map(RawEvent)
        .ok_or_else(invalid_argument)
}

/// Records the timing event with the external-record flag: inside a stream capture the record
/// becomes an event-record node of the captured graph, so replays re-record it.
pub(in crate::kernels::cuda) fn event_record_external(
    event: &RawEvent,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: both handles are live for the whole call and `stage` is a live,
    // writable c_int. The call only queues the record.
    let status =
        unsafe { event_record_external_raw(event.0.as_ptr(), stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// The milliseconds between two completed timing events.
pub(in crate::kernels::cuda) fn event_elapsed_ms(
    start: &RawEvent,
    end: &RawEvent,
) -> Result<f32, DeviceFailure> {
    let mut ms = 0.0_f32;
    let mut stage = STAGE_NONE;
    // SAFETY: both handles are live for the whole call; `ms` and `stage` are
    // live, writable values. Both records must have completed first.
    let status =
        unsafe { event_elapsed_ms_raw(start.0.as_ptr(), end.0.as_ptr(), &mut ms, &mut stage) };
    checked(status, stage)?;
    Ok(ms)
}

/// Makes the work queued later on `stream` wait for the last record of `event`.
pub(in crate::kernels::cuda) fn stream_wait_event(
    stream: &RawStream,
    event: &RawEvent,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: both handles are live for the whole call and `stage` is a live,
    // writable c_int. The call only queues the wait.
    let status = unsafe { stream_wait_event_raw(stream.0.as_ptr(), event.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Blocks the host until the last record of `event` has completed.
pub(in crate::kernels::cuda) fn event_synchronize(event: &RawEvent) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `event` is a live handle for the whole call; `stage` is a live,
    // writable c_int.
    let status = unsafe { event_synchronize_raw(event.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Allocates `bytes` of page-locked host memory. The adapter rejects 0 bytes.
pub(in crate::kernels::cuda) fn pinned_alloc(bytes: usize) -> Result<RawPinned, DeviceFailure> {
    if bytes == 0 {
        return Err(invalid_argument());
    }
    let mut host: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `host` and `stage` are live, writable values for the call. The
    // adapter writes a pointer only on success.
    let status = unsafe { pinned_alloc_raw(bytes, &mut host, &mut stage) };
    checked(status, stage)?;
    let ptr = NonNull::new(host).ok_or_else(invalid_argument)?;
    Ok(RawPinned { ptr, bytes })
}

/// Waits for the device, then frees the memory. When the wait fails nothing is
/// freed and the failure is returned.
pub(in crate::kernels::cuda) fn pinned_free(pinned: RawPinned) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `pinned` came from `pinned_alloc` and is consumed here, so it is
    // freed at most once and no other handle refers to it afterwards. The
    // adapter waits for the whole device first, so no queued copy still
    // writes the memory.
    let status = unsafe { pinned_free_raw(pinned.ptr.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues a copy of `src` into the pinned memory on `stream`. The copy has not
/// finished when the call returns; read the memory only after an event
/// recorded behind it has completed.
pub(in crate::kernels::cuda) fn copy_d2h_async(
    host: &RawPinned,
    src: RawRange<'_>,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    if src.bytes > host.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: `RawRange::new` proved `offset + bytes <= device.bytes` for the live
    // allocation, so the offset pointer and the `src.bytes` bytes behind it lie
    // inside it; the check above keeps `src.bytes` inside the pinned memory.
    // Both stay valid until the copy finishes: `inference_cuda_free` and
    // `pinned_free` synchronize the device first. `stream` is live for the call.
    let status = unsafe {
        copy_d2h_async_raw(
            host.ptr.as_ptr(),
            src.address(),
            src.bytes,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Copies owned pageable storage behind earlier work on `stream` and waits for completion.
pub(in crate::kernels::cuda) fn copy_h2d_pageable(
    dst: RawRange<'_>,
    host: Vec<u8>,
    stream: &RawStream,
) -> Result<Vec<u8>, DeviceFailure> {
    if host.len() != dst.bytes || host.capacity() > crate::kernels::cuda::CHUNK_LIMIT {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the range is live and in bounds. The owned initialized storage is retained
    // through the stream fence, including a failed copy or completion attempt.
    let status = unsafe {
        copy_h2d_async_raw(
            dst.address(),
            host.as_ptr().cast(),
            host.len(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    complete_pageable_copy(host, stream, checked(status, stage))
}

/// Queues a pinned-host range into a device range. Both owners synchronize before free.
pub(in crate::kernels::cuda) fn copy_h2d_async_at(
    dst: RawRange<'_>,
    host: &RawPinned,
    host_offset: usize,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let end = host_offset
        .checked_add(dst.bytes)
        .ok_or_else(invalid_argument)?;
    if end > host.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: both checked ranges stay alive until their synchronizing frees; the safe pinned
    // owner orders all DMA on one stream and refuses host reads until a covering event completes.
    let status = unsafe {
        let src = host
            .ptr
            .as_ptr()
            .cast::<u8>()
            .add(host_offset)
            .cast_const()
            .cast::<c_void>();
        copy_h2d_async_raw(dst.address(), src, dst.bytes, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

/// Queues equal ranges from distinct device allocations on the explicit stream.
pub(in crate::kernels::cuda) fn copy_d2d_async(
    source: RawRange<'_>,
    destination: RawRange<'_>,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    if source.bytes != destination.bytes || std::ptr::eq(source.device, destination.device) {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: checked ranges stay alive until their synchronizing frees. The context wrapper
    // proves adapter ownership and the ranges belong to distinct live allocations.
    let status = unsafe {
        copy_d2d_async_raw(
            destination.address(),
            source.address().cast_const(),
            source.bytes,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Returns owned pageable bytes only after the stream proves their copy complete.
pub(in crate::kernels::cuda) fn copy_d2h_pageable(
    src: RawRange<'_>,
    mut host: Vec<u8>,
    stream: &RawStream,
) -> Result<Vec<u8>, DeviceFailure> {
    if host.len() != src.bytes || host.capacity() > crate::kernels::cuda::CHUNK_LIMIT {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the range is live and in bounds; the exclusively owned writable storage
    // remains allocated even if the stream cannot prove completion.
    let status = unsafe {
        copy_d2h_async_raw(
            host.as_mut_ptr().cast(),
            src.address(),
            src.bytes,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    complete_pageable_copy(host, stream, checked(status, stage))
}

fn complete_pageable_copy(
    host: Vec<u8>,
    stream: &RawStream,
    copy: Result<(), DeviceFailure>,
) -> Result<Vec<u8>, DeviceFailure> {
    match stream_synchronize(stream) {
        Ok(()) => {
            copy?;
            Ok(host)
        }
        Err(completion) => {
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            RETAINED_PAGEABLE_BYTES.fetch_add(host.capacity(), std::sync::atomic::Ordering::AcqRel);
            // Unknown completion cannot release host storage, even when copy failed first.
            std::mem::forget(host);
            Err(copy.err().unwrap_or(completion))
        }
    }
}

/// The first `min(bytes, 8)` bytes of the pinned memory, zero padded.
///
/// The read is volatile because the device writes the memory behind the
/// compiler's back. The caller must have waited for the copy (see the module
/// doc); this function cannot check it.
pub(in crate::kernels::cuda) fn pinned_read(pinned: &RawPinned) -> [u8; 8] {
    let mut out = [0u8; 8];
    let base = pinned.ptr.as_ptr().cast::<u8>().cast_const();
    for (index, slot) in out.iter_mut().enumerate().take(pinned.bytes) {
        // SAFETY: `index < pinned.bytes`, and the allocation is `pinned.bytes`
        // bytes long until `pinned_free` consumes the handle, which the borrow
        // of `pinned` rules out.
        *slot = unsafe { base.add(index).read_volatile() };
    }
    out
}

/// Queues a copy of `src` into the pinned memory at byte `host_offset` on `stream`, behind the
/// work queued earlier on it. The copy has not finished when the call returns; the safe owner
/// (`PinnedBuf`) tracks it and refuses every read until an event recorded behind it completed.
pub(in crate::kernels::cuda) fn copy_d2h_async_at(
    host: &RawPinned,
    host_offset: usize,
    src: RawRange<'_>,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let end = host_offset
        .checked_add(src.bytes)
        .ok_or_else(invalid_argument)?;
    if end > host.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: `RawRange::new` proved `offset + bytes <= device.bytes` for the live device
    // allocation; the check above proved `host_offset + src.bytes <= host.bytes`, so the offset
    // pointer and the `src.bytes` bytes behind it lie inside the pinned allocation. Both stay
    // valid until the copy finishes: `inference_cuda_free` and `pinned_free` synchronize the
    // device first. `stream` is live for the call.
    let status = unsafe {
        let dst = host
            .ptr
            .as_ptr()
            .cast::<u8>()
            .add(host_offset)
            .cast::<c_void>();
        copy_d2h_async_raw(dst, src.address(), src.bytes, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

/// Zeroes the pinned memory. Only for freshly allocated memory that no copy has been queued
/// into yet, so the host never reads indeterminate bytes.
pub(in crate::kernels::cuda) fn pinned_zero(pinned: &RawPinned) {
    // SAFETY: the allocation is `pinned.bytes` bytes long and live (borrowed); the safe owner calls
    // this before it hands out the buffer, so no DMA is queued into it and nothing else reads
    // or writes it concurrently.
    unsafe { std::ptr::write_bytes(pinned.ptr.as_ptr().cast::<u8>(), 0, pinned.bytes) };
}

/// Copies `out.len()` bytes at `offset` of the pinned memory into `out`.
///
/// Precondition the caller must discharge (this function cannot check it): no device-to-host
/// copy into `[offset, offset + out.len())` is in flight. The only caller, `PinnedBuf::read`,
/// refuses to run while a copy is pending and clears the pending state only after it verified
/// that an event recorded behind the last copy completed. Returns an error for a range outside
/// the allocation.
pub(in crate::kernels::cuda) fn pinned_read_into(
    pinned: &RawPinned,
    offset: usize,
    out: &mut [u8],
) -> Result<(), DeviceFailure> {
    let end = offset.checked_add(out.len()).ok_or_else(invalid_argument)?;
    if end > pinned.bytes {
        return Err(invalid_argument());
    }
    let base = pinned.ptr.as_ptr().cast::<u8>().cast_const();
    for (index, slot) in out.iter_mut().enumerate() {
        // SAFETY: `offset + index < end <= pinned.bytes`, and the allocation stays live and
        // unfreed while `pinned` is borrowed. The volatile read keeps the compiler from caching
        // bytes a device copy wrote; that no copy is writing them right now is the caller's
        // proven precondition above.
        *slot = unsafe { base.add(offset + index).read_volatile() };
    }
    Ok(())
}

/// Writes a checked pinned range. The owner proves no transfer can access it.
pub(in crate::kernels::cuda) fn pinned_write_from(
    pinned: &RawPinned,
    offset: usize,
    bytes: &[u8],
) -> Result<(), DeviceFailure> {
    let end = offset
        .checked_add(bytes.len())
        .ok_or_else(invalid_argument)?;
    if end > pinned.bytes {
        return Err(invalid_argument());
    }
    // SAFETY: the checked destination is live, and the safe owner refuses outstanding
    // transfers. Its API exposes no slice into pinned storage, so the input cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            pinned.ptr.as_ptr().cast::<u8>().add(offset),
            bytes.len(),
        );
    }
    Ok(())
}

/// Copies a host slice through pinned staging into a device allocation.
/// The pinned buffer is intermediate scratch; the device ends up with `host`.
pub(in crate::kernels::cuda) fn upload_pinned(
    device: &RawDevice,
    pinned: &RawPinned,
    host: &[u8],
) -> Result<(), DeviceFailure> {
    if host.is_empty() || host.len() > pinned.bytes || host.len() > device.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: device and pinned are live allocations of at least host.len()
    // bytes, host addresses host.len() initialized bytes for the whole call,
    // and the adapter keeps no host pointer after it returns.
    let status = unsafe {
        upload_pinned_raw(
            device.ptr.as_ptr(),
            device.bytes,
            pinned.ptr.as_ptr(),
            pinned.bytes,
            host.as_ptr().cast(),
            host.len(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// What one pack launch reads and writes: `x` is `m` rows of `k` f32 values, `xf`
/// receives their fp16 form.
pub(in crate::kernels::cuda) struct PackRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) xf: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) k: usize,
}

/// What one matmul launch reads and writes: `tiled` is the shape of the rows
/// that start at `weight`, `xf` is the packed input of `m` tokens and `y`
/// receives `m` rows of `tiled.n_rows` values.
pub(in crate::kernels::cuda) struct MatmulRaw<'a> {
    pub(in crate::kernels::cuda) tiled: Tiled,
    pub(in crate::kernels::cuda) weight: DeviceView<'a>,
    pub(in crate::kernels::cuda) xf: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
}

/// `len` bytes of one allocation from `start` on. The allocation is named by its
/// handle: two live handles are two allocations, which never overlap.
pub(in crate::kernels::cuda) struct Span<'a> {
    device: &'a RawDevice,
    start: usize,
    len: usize,
}

impl<'a> Span<'a> {
    /// The span of the first `len` bytes of `range`.
    pub(in crate::kernels::cuda) fn of(range: &RawRange<'a>, len: usize) -> Self {
        Self {
            device: range.device,
            start: range.offset,
            len,
        }
    }

    /// True when the two spans share no byte. Every span lies inside its
    /// allocation, so the sums cannot overflow; a saturated sum would only refuse.
    pub(in crate::kernels::cuda) fn apart(&self, other: &Self) -> bool {
        !std::ptr::eq(self.device, other.device)
            || self.start.saturating_add(self.len) <= other.start
            || other.start.saturating_add(other.len) <= self.start
    }
}

/// Proves a pack launch before the native call: shape, sizes, alignment (x 4,
/// xf 16) and that `xf`, which the launch writes, does not overlap `x`. Returns
/// `(m, k)` for the call.
fn check_pack(call: &PackRaw<'_>) -> Result<(c_int, c_int), DeviceFailure> {
    let shape = (1..=MAX_DIM).contains(&call.m)
        && (1..=MAX_DIM).contains(&call.k)
        && call.k.is_multiple_of(TILE_COLS);
    let (x_need, xf_need) = pack_bytes(call.m, call.k)
        .filter(|_| shape)
        .ok_or_else(invalid_argument)?;
    let (m, k) = (
        c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        c_int::try_from(call.k).map_err(|_| invalid_argument())?,
    );
    let sized = call.x.bytes >= x_need && call.xf.bytes >= xf_need;
    let aligned = call.x.offset.is_multiple_of(4) && call.xf.offset.is_multiple_of(16);
    let apart = Span::of(&call.x, x_need).apart(&Span::of(&call.xf, xf_need));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok((m, k))
}

/// Proves a matmul launch before the native call: shape, sizes, the weight tiles
/// inside their allocation, alignment (weight 16, xf 16, y 4) and that `y`, which
/// the launch writes, overlaps neither the weight nor `xf`. Returns
/// `(m, n_rows, k)` for the call; `ldy` is `n_rows`.
fn check_matmul(call: &MatmulRaw<'_>) -> Result<(c_int, c_int, c_int), DeviceFailure> {
    let (n_rows, k) = call.tiled.native_dims().ok_or_else(invalid_argument)?;
    if !(1..=MAX_DIM).contains(&call.m) {
        return Err(invalid_argument());
    }
    let m = c_int::try_from(call.m).map_err(|_| invalid_argument())?;
    let need = activation_bytes(call.tiled.n_rows, call.tiled.k, call.m)
        .map_err(|_| invalid_argument())?;
    let tile_bytes = call
        .tiled
        .tile_vecs()
        .and_then(|vecs| vecs.checked_mul(16))
        .ok_or_else(invalid_argument)?;
    let weight_inside = tile_bytes
        .checked_add(call.weight.offset)
        .is_some_and(|end| end <= call.weight.device.bytes);
    let sized = call.xf.bytes >= need.xf && call.y.bytes >= need.y;
    let aligned = call.weight.offset.is_multiple_of(16)
        && call.xf.offset.is_multiple_of(16)
        && call.y.offset.is_multiple_of(4);
    let weight = Span {
        device: call.weight.device,
        start: call.weight.offset,
        len: tile_bytes,
    };
    let (xf, y) = (Span::of(&call.xf, need.xf), Span::of(&call.y, need.y));
    let apart = y.apart(&xf) && y.apart(&weight) && xf.apart(&weight);
    if !(weight_inside && sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok((m, n_rows, k))
}

/// Queues the pack of `x` into `xf` on `stream`: the zeroing of the whole `xf`
/// when `m` is at most 16, then the pack kernel. Returns when both are queued.
pub(in crate::kernels::cuda) fn pack_launch(
    stream: &RawStream,
    call: &PackRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (m, k) = check_pack(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_pack` proved both ranges inside their live allocations, large
    // enough for the call, aligned, and disjoint; the adapter repeats these checks
    // on the addresses. The allocations stay valid until the queued work finishes
    // because `inference_cuda_free` synchronizes the device first. `stream` is live
    // for the call.
    let status = unsafe {
        pack_launch_raw(
            call.x.address().cast::<f32>().cast_const(),
            m,
            k,
            call.xf.address(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Queues the tiled matmul of `call` on `stream`: the split-K zeroing of `y` when
/// the shape splits, then the kernel. Returns when both are queued.
pub(in crate::kernels::cuda) fn matmul_launch(
    stream: &RawStream,
    call: &MatmulRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (m, n_rows, k) = check_matmul(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_matmul` proved the weight tiles and both ranges inside their
    // live allocations, large enough, aligned, and `y` disjoint from the weight and
    // `xf`, so the weight offset stays inside its allocation; the adapter repeats
    // the checks on the addresses. The allocations stay valid until the queued work
    // finishes because `inference_cuda_free` synchronizes the device first.
    // `stream` is live for the call.
    let status = unsafe {
        let weight = call
            .weight
            .device
            .ptr
            .as_ptr()
            .cast::<u8>()
            .add(call.weight.offset);
        matmul_launch_raw(
            call.tiled.ty.ggml(),
            m,
            weight.cast::<c_void>().cast_const(),
            call.xf.address().cast_const(),
            call.y.address().cast::<f32>(),
            n_rows,
            n_rows,
            k,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

// ---- The f32 rows entry of attn_prep (the Int8G32 KV handoff) --------------------------------
//
// Raw FFI to `inference_cuda_attn_prep_rows_launch` of `native/resident/attn_prep.cu` (declared in
// `native/resident/attn_prep.h`): the same kernel's f32 rows instance, which indexes the k and v
// rows by the token and bounds the pos word by the plane cell count. The safety invariants and the
// proofs are the child `attn_prep`'s, mirrored here on the rows extents; the safe wrapper of
// `kernels::cuda::resident::attn_prep` is its only caller.

/// The head size of the rows entry (the vendored kernel's only one).
const ROWS_HEAD_DIM: usize = 256;

/// Mirror of `inference_cuda_attn_prep_rows_args` in `attn_prep.h`, field for field.
#[repr(C)]
struct AttnPrepRowsArgsRaw {
    proj: *const c_void,
    q_norm: *const c_void,
    k_norm: *const c_void,
    k_rows: *mut c_void,
    v_rows: *mut c_void,
    q: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    ld: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    n_rot: c_int,
    cells: c_int,
    has_gate: c_int,
    slot_mode: c_int,
    slot_ring: c_int,
    eps: f32,
    rope_base: f32,
}

const _: () = assert!(
    size_of::<AttnPrepRowsArgsRaw>() == 112
        && offset_of!(AttnPrepRowsArgsRaw, k_rows) == 24
        && offset_of!(AttnPrepRowsArgsRaw, cells) == 88
);

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prep_rows_launch"]
    fn attn_prep_rows_launch_raw(
        args: *const AttnPrepRowsArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one rows attention preparation reads and writes; `ints.rows` is the plane cell count.
pub(in crate::kernels::cuda) struct AttnPrepRowsRaw<'a> {
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) q_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_rows: RawRange<'a>,
    pub(in crate::kernels::cuda) v_rows: RawRange<'a>,
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: attn_prep::AttnInts,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) rope_base: f32,
}

/// Proves a rows launch before the native call: sizes and alignment of the eight ranges (`q` and
/// both rows buffers 16, the rest 4) and that the written extents (`q`, both rows buffers,
/// `status`) share no byte with each other or with the reads. `cells` sizes nothing: the rows
/// buffers hold `m` rows, not `cells`.
fn check_attn_prep_rows(io: &AttnPrepRowsRaw<'_>) -> Result<(), DeviceFailure> {
    let cvt = |n: c_int| usize::try_from(n).map_err(|_| invalid_argument());
    let (m, ld) = (cvt(io.ints.m)?, cvt(io.ints.ld)?);
    let (heads, kv) = (cvt(io.ints.n_head)?, cvt(io.ints.n_kv_head)?);
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(invalid_argument);
    let row_w = mul(heads * 2 + kv * 2, ROWS_HEAD_DIM)?;
    let proj = mul(
        m.saturating_sub(1).saturating_mul(ld).saturating_add(row_w),
        4,
    )?;
    let norm = ROWS_HEAD_DIM * 4;
    let q = mul(mul(mul(m, heads)?, ROWS_HEAD_DIM)?, 4)?;
    let rows = mul(mul(mul(m, kv)?, ROWS_HEAD_DIM)?, 4)?;
    let fits = |range: &RawRange<'_>, bytes: usize, align: usize| {
        range.bytes >= bytes && range.offset.is_multiple_of(align)
    };
    if !(fits(&io.proj, proj, 4)
        && fits(&io.q_norm, norm, 4)
        && fits(&io.k_norm, norm, 4)
        && fits(&io.q, q, 16)
        && fits(&io.k_rows, rows, 16)
        && fits(&io.v_rows, rows, 16)
        && fits(&io.pos, 4, 4)
        && fits(&io.status, 4, 4))
    {
        return Err(invalid_argument());
    }
    let written = [
        Span::of(&io.q, q),
        Span::of(&io.k_rows, rows),
        Span::of(&io.v_rows, rows),
        Span::of(&io.status, 4),
    ];
    let read = [
        Span::of(&io.proj, proj),
        Span::of(&io.q_norm, norm),
        Span::of(&io.k_norm, norm),
        Span::of(&io.pos, 4),
    ];
    for (i, a) in written.iter().enumerate() {
        if !(written[i + 1..].iter().all(|b| a.apart(b)) && read.iter().all(|b| a.apart(b))) {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Queues the f32 rows attention preparation kernel on `stream`: per-head RMSNorm of q and k,
/// rope at the device position, the `q` output and the k/v f32 rows at `[t]`. Returns when queued.
pub(in crate::kernels::cuda) fn attn_prep_rows_launch(
    stream: &RawStream,
    call: AttnPrepRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn_prep_rows(&call)?;
    let ints = call.ints;
    let args = AttnPrepRowsArgsRaw {
        proj: call.proj.address().cast_const(),
        q_norm: call.q_norm.address().cast_const(),
        k_norm: call.k_norm.address().cast_const(),
        k_rows: call.k_rows.address(),
        v_rows: call.v_rows.address(),
        q: call.q.address(),
        pos: call.pos.address().cast_const(),
        status: call.status.address(),
        ld: ints.ld,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        head_dim: 256,
        n_rot: ints.n_rot,
        cells: ints.rows,
        has_gate: 1,
        slot_mode: 0, // INFERENCE_CUDA_SLOT_POSITION
        slot_ring: 0,
        eps: call.eps,
        rope_base: call.rope_base,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as the child `attn_prep`: `check_attn_prep_rows` proved every range inside its live
    // allocation, large enough, aligned and disjoint on the touched extents, and the allocations
    // outlive the queued work; `stream` is live for the call, `args` is a local the entry keeps no
    // pointer to, and the dispatch guard is held by `launch_on`, the only caller.
    let status =
        unsafe { attn_prep_rows_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::{STATUS_INVALID_ARGUMENT, TiledType};
    use super::*;

    /// Host only: a range of real host memory stands in for a device allocation,
    /// because only the pointer arithmetic is under test and nothing is read.
    #[test]
    fn a_range_address_is_the_allocation_address_plus_its_offset() {
        let mut memory = [0u8; 64];
        let base = memory.as_mut_ptr();
        let device = RawDevice {
            ptr: NonNull::new(base.cast::<c_void>()).expect("a stack address"),
            bytes: memory.len(),
        };
        for offset in [0, 1, 16, 63] {
            let range = RawRange::new(&device, offset, 1).expect("a range inside the memory");
            assert_eq!(
                range.address().addr(),
                base.addr() + offset,
                "offset {offset}"
            );
        }
    }

    /// Host only: `RawDevice::unbacked` has no allocation behind it, and
    /// `RawRange::new` only compares numbers.
    #[test]
    fn a_range_is_not_empty_and_lies_inside_the_allocation() {
        let device = RawDevice::unbacked(16);
        assert!(
            RawRange::new(&device, 0, 16).is_some(),
            "the whole allocation"
        );
        assert!(RawRange::new(&device, 15, 1).is_some(), "the last byte");
        assert!(RawRange::new(&device, 0, 0).is_none(), "an empty range");
        assert!(
            RawRange::new(&device, 1, 16).is_none(),
            "one byte past the end"
        );
        assert!(RawRange::new(&device, 16, 1).is_none(), "starts at the end");
        assert!(
            RawRange::new(&device, usize::MAX, 2).is_none(),
            "offset + bytes overflows"
        );
    }

    /// Host only: both host copies refuse a host slice of another length before any native call,
    /// so a dangling stream is never used.
    #[test]
    fn device_copies_reject_a_host_slice_of_the_wrong_length_without_native_call() {
        let device = RawDevice::unbacked(16);
        let stream = RawStream(NonNull::dangling());
        let range = range(&device, 4, 8);
        assert_eq!(
            copy_h2d_pageable(range, vec![0; 7], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_h2d_pageable(range, vec![0; 9], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_h2d_pageable(range, vec![], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_d2h_pageable(range, vec![0; 7], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_d2h_pageable(range, vec![0; 9], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_d2h_pageable(range, vec![], &stream),
            Err(invalid_argument())
        );
    }

    /// Host only: oversized owned storage is refused before the dangling stream is used.
    #[test]
    fn pageable_copies_refuse_storage_capacity_above_the_chunk_limit() {
        let device = RawDevice::unbacked(16);
        let stream = RawStream(NonNull::dangling());
        let range = range(&device, 4, 8);
        for copy in [copy_h2d_pageable, copy_d2h_pageable] {
            let mut host = Vec::with_capacity(crate::kernels::cuda::CHUNK_LIMIT + 1);
            host.extend_from_slice(&[0; 8]);
            assert_eq!(copy(range, host, &stream), Err(invalid_argument()));
        }
    }

    /// Allocation size of the host-only fixtures; no memory backs it.
    const FIXTURE_BYTES: usize = 4 << 20;
    const XF_FRAGMENT: usize = 1_048_592;

    fn range(device: &RawDevice, offset: usize, bytes: usize) -> RawRange<'_> {
        RawRange::new(device, offset, bytes).expect("a range inside the fixture")
    }

    struct PackCase {
        name: &'static str,
        x: (usize, usize),
        xf: (usize, usize),
        shared: bool,
        m: usize,
        k: usize,
        accepted: bool,
    }

    /// One token of 256 values: x is 1024 bytes, xf the fixed 1_048_592.
    const PACK_BASE: PackCase = PackCase {
        name: "two allocations",
        x: (0, 1024),
        xf: (0, XF_FRAGMENT),
        shared: false,
        m: 1,
        k: 256,
        accepted: true,
    };

    #[test]
    fn a_pack_launch_is_checked_for_shape_size_alignment_and_overlap() {
        let cases = [
            PACK_BASE,
            PackCase {
                name: "xf right after x in one allocation",
                shared: true,
                xf: (1024, XF_FRAGMENT),
                ..PACK_BASE
            },
            PackCase {
                name: "x range reaches into xf, but the bytes the call uses do not",
                shared: true,
                x: (0, 2048),
                xf: (1024, XF_FRAGMENT),
                ..PACK_BASE
            },
            PackCase {
                name: "xf overlaps the end of x",
                shared: true,
                xf: (1008, XF_FRAGMENT),
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "x one float short",
                x: (0, 1020),
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "xf one byte short",
                xf: (0, XF_FRAGMENT - 1),
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "x offset not a multiple of 4",
                x: (2, 1024),
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "xf offset not a multiple of 16",
                xf: (4, XF_FRAGMENT),
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "no tokens",
                m: 0,
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "more tokens than MAX_DIM",
                m: MAX_DIM + 1,
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "k = 100",
                k: 100,
                accepted: false,
                ..PACK_BASE
            },
            PackCase {
                name: "k = 0",
                k: 0,
                accepted: false,
                ..PACK_BASE
            },
        ];
        let (a, b) = (
            RawDevice::unbacked(FIXTURE_BYTES),
            RawDevice::unbacked(FIXTURE_BYTES),
        );
        for case in cases {
            let call = PackRaw {
                x: range(&a, case.x.0, case.x.1),
                xf: range(if case.shared { &a } else { &b }, case.xf.0, case.xf.1),
                m: case.m,
                k: case.k,
            };
            let checked = check_pack(&call);
            assert_eq!(checked.is_ok(), case.accepted, "{}", case.name);
            if let Ok(shape) = checked {
                assert_eq!(shape, (1, 256), "{}", case.name);
            }
        }
    }

    /// Where the weight, xf and y ranges of a matmul fixture live.
    #[derive(Clone, Copy)]
    enum Layout {
        Apart,
        XfWithY,
        YWithWeight,
        XfWithWeight,
    }

    struct MatmulCase {
        name: &'static str,
        layout: Layout,
        weight_offset: usize,
        xf: (usize, usize),
        y: (usize, usize),
        m: usize,
        n_rows: usize,
        k: usize,
        accepted: bool,
    }

    /// Eight rows of 256 values, one token: the tiles take 1152 bytes, xf the
    /// fixed 1_048_592 and y 32.
    const MATMUL_BASE: MatmulCase = MatmulCase {
        name: "three allocations",
        layout: Layout::Apart,
        weight_offset: 0,
        xf: (0, XF_FRAGMENT),
        y: (0, 32),
        m: 1,
        n_rows: 8,
        k: 256,
        accepted: true,
    };

    #[test]
    fn a_matmul_launch_is_checked_for_shape_size_alignment_and_overlap() {
        let reject = |name, edit: fn(MatmulCase) -> MatmulCase| {
            edit(MatmulCase {
                name,
                accepted: false,
                ..MATMUL_BASE
            })
        };
        let cases = [
            MATMUL_BASE,
            MatmulCase {
                name: "y right after xf in one allocation",
                layout: Layout::XfWithY,
                y: (XF_FRAGMENT, 32),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "xf range reaches into y, but the bytes the call uses do not",
                layout: Layout::XfWithY,
                xf: (0, XF_FRAGMENT + 4096),
                y: (XF_FRAGMENT, 32),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "y range reaches into xf, but the bytes the call uses do not",
                layout: Layout::XfWithY,
                xf: (32, XF_FRAGMENT),
                y: (0, 4096),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "y right before xf in one allocation",
                layout: Layout::XfWithY,
                xf: (32, XF_FRAGMENT),
                y: (0, 32),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "y right before the weight in one allocation",
                layout: Layout::YWithWeight,
                weight_offset: 32,
                y: (0, 32),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "xf right before the weight in one allocation",
                layout: Layout::XfWithWeight,
                weight_offset: XF_FRAGMENT,
                xf: (0, XF_FRAGMENT),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "y right after the weight in one allocation",
                layout: Layout::YWithWeight,
                y: (1152, 32),
                ..MATMUL_BASE
            },
            MatmulCase {
                name: "xf right after the weight in one allocation",
                layout: Layout::XfWithWeight,
                xf: (1152, XF_FRAGMENT),
                ..MATMUL_BASE
            },
            reject("y inside xf", |c| MatmulCase {
                layout: Layout::XfWithY,
                y: (XF_FRAGMENT - 32, 32),
                ..c
            }),
            reject("y inside the weight tiles", |c| MatmulCase {
                layout: Layout::YWithWeight,
                y: (1024, 32),
                ..c
            }),
            reject("xf over the weight tiles", |c| MatmulCase {
                layout: Layout::XfWithWeight,
                xf: (1136, XF_FRAGMENT),
                ..c
            }),
            reject("xf one byte short", |c| MatmulCase {
                xf: (0, XF_FRAGMENT - 1),
                ..c
            }),
            reject("y one float short", |c| MatmulCase { y: (0, 28), ..c }),
            reject("weight offset not a multiple of 16", |c| MatmulCase {
                weight_offset: 8,
                ..c
            }),
            reject("xf offset not a multiple of 16", |c| MatmulCase {
                xf: (4, XF_FRAGMENT),
                ..c
            }),
            reject("y offset not a multiple of 4", |c| MatmulCase {
                y: (2, 32),
                ..c
            }),
            reject("tiles end past the allocation", |c| MatmulCase {
                weight_offset: FIXTURE_BYTES - 1024,
                ..c
            }),
            reject("no tokens", |c| MatmulCase { m: 0, ..c }),
            reject("more tokens than MAX_DIM", |c| MatmulCase {
                m: MAX_DIM + 1,
                ..c
            }),
            reject("12 rows", |c| MatmulCase { n_rows: 12, ..c }),
            reject("k = 100", |c| MatmulCase { k: 100, ..c }),
        ];
        let devices = [
            RawDevice::unbacked(FIXTURE_BYTES),
            RawDevice::unbacked(FIXTURE_BYTES),
            RawDevice::unbacked(FIXTURE_BYTES),
        ];
        for case in cases {
            let [w, x, y] = match case.layout {
                Layout::Apart => [0, 1, 2],
                Layout::XfWithY => [0, 1, 1],
                Layout::YWithWeight => [0, 1, 0],
                Layout::XfWithWeight => [0, 0, 2],
            };
            let call = MatmulRaw {
                tiled: Tiled {
                    ty: TiledType::Q4K,
                    n_rows: case.n_rows,
                    k: case.k,
                },
                weight: devices[w].at(case.weight_offset),
                xf: range(&devices[x], case.xf.0, case.xf.1),
                y: range(&devices[y], case.y.0, case.y.1),
                m: case.m,
            };
            let checked = check_matmul(&call);
            assert_eq!(checked.is_ok(), case.accepted, "{}", case.name);
            if let Ok(shape) = checked {
                assert_eq!(shape, (1, 8, 256), "{}", case.name);
            }
        }
    }

    /// The arguments of one `inference_cuda_pack_launch` call as plain numbers.
    #[derive(Clone, Copy)]
    struct PackArgs {
        x: usize,
        m: c_int,
        k: c_int,
        xf: usize,
        stream: usize,
    }

    /// A call that would launch: aligned, disjoint, in range. The addresses are
    /// made up; the cases below change one thing so the entry refuses before it
    /// reaches the device, and a valid call is never made.
    const PACK_ARGS: PackArgs = PackArgs {
        x: 0x1_0000,
        m: 1,
        k: 256,
        xf: 0x10_0000,
        stream: 0x1,
    };

    /// A named edit of a valid argument set.
    type Case<T> = (&'static str, fn(&mut T));

    const fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    /// `(status, stage)` of the entry for `args`.
    fn run_pack(args: PackArgs) -> (c_int, c_int) {
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any
        // pointer (see `PACK_ARGS`); `stage` is a live, writable c_int.
        let status = unsafe {
            pack_launch_raw(
                pointer::<f32>(args.x).cast_const(),
                args.m,
                args.k,
                pointer(args.xf),
                pointer(args.stream),
                &mut stage,
            )
        };
        (status, stage)
    }

    /// The arguments of one `inference_cuda_matmul_launch` call as plain numbers.
    #[derive(Clone, Copy)]
    struct MatmulArgs {
        ty: c_int,
        m: c_int,
        weight: usize,
        xf: usize,
        y: usize,
        ldy: c_int,
        n_rows: c_int,
        k: c_int,
        stream: usize,
    }

    /// Eight Q4_K rows of 256 values, one token; see `PACK_ARGS`.
    const MATMUL_ARGS: MatmulArgs = MatmulArgs {
        ty: 12,
        m: 1,
        weight: 0x1_0000,
        xf: 0x10_0000,
        y: 0x30_0000,
        ldy: 8,
        n_rows: 8,
        k: 256,
        stream: 0x1,
    };

    fn run_matmul(args: MatmulArgs) -> (c_int, c_int) {
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any
        // pointer (see `MATMUL_ARGS`); `stage` is a live, writable c_int.
        let status = unsafe {
            matmul_launch_raw(
                args.ty,
                args.m,
                pointer::<c_void>(args.weight).cast_const(),
                pointer::<c_void>(args.xf).cast_const(),
                pointer(args.y),
                args.ldy,
                args.n_rows,
                args.k,
                pointer(args.stream),
                &mut stage,
            )
        };
        (status, stage)
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA
    /// call, so the test needs no device. It covers the C half of the defence in
    /// depth: a NULL pointer or stream, a bad shape, a misaligned pointer, an
    /// overlap that the Rust wrappers already refuse.
    #[test]
    fn the_native_launch_entries_refuse_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        let pack: [Case<PackArgs>; 11] = [
            ("NULL x", |a| a.x = 0),
            ("NULL xf", |a| a.xf = 0),
            ("NULL stream", |a| a.stream = 0),
            ("no tokens", |a| a.m = 0),
            ("more tokens than MAX_DIM", |a| a.m = 32769),
            ("k = 0", |a| a.k = 0),
            ("k = 100", |a| a.k = 100),
            ("k above MAX_DIM", |a| a.k = 32768 + 256),
            ("x offset not a multiple of 4", |a| a.x += 2),
            ("xf offset not a multiple of 16", |a| a.xf += 4),
            ("xf over the end of x", |a| a.xf = a.x + 1008),
        ];
        for (name, edit) in pack {
            let mut args = PACK_ARGS;
            edit(&mut args);
            assert_eq!(run_pack(args), refused, "pack: {name}");
        }
        let matmul: [Case<MatmulArgs>; 17] = [
            ("NULL weight", |a| a.weight = 0),
            ("NULL xf", |a| a.xf = 0),
            ("NULL y", |a| a.y = 0),
            ("NULL stream", |a| a.stream = 0),
            ("a type that is not tiled", |a| a.ty = 99),
            ("12 rows", |a| a.n_rows = 12),
            ("k = 100", |a| a.k = 100),
            ("no tokens", |a| a.m = 0),
            ("more tokens than MAX_DIM", |a| a.m = 32769),
            ("ldy below n_rows", |a| a.ldy = 7),
            ("weight offset not a multiple of 16", |a| a.weight += 8),
            ("xf offset not a multiple of 16", |a| a.xf += 4),
            ("y offset not a multiple of 4", |a| a.y += 2),
            ("y inside xf", |a| a.y = a.xf + 1024),
            ("y inside the weight tiles", |a| a.y = a.weight + 1024),
            ("xf over the weight tiles", |a| a.xf = a.weight + 1136),
            ("y over the end of xf", |a| a.y = a.xf + 1_048_592 - 16),
        ];
        for (name, edit) in matmul {
            let mut args = MATMUL_ARGS;
            edit(&mut args);
            assert_eq!(run_matmul(args), refused, "matmul: {name}");
        }
        let rows: [Case<AttnPrepRowsArgsRaw>; 7] = [
            ("NULL k_rows", |a| a.k_rows = pointer(0)),
            ("NULL v_rows", |a| a.v_rows = pointer(0)),
            ("k_rows not 16-byte aligned", |a| {
                a.k_rows = pointer(0x2_0008)
            }),
            ("v_rows not 16-byte aligned", |a| {
                a.v_rows = pointer(0x3_0008)
            }),
            ("no tokens", |a| a.m = 0),
            ("m 17 (kv8_append's decode bound)", |a| a.m = 17),
            ("cells below m", |a| a.cells = 0),
        ];
        for (name, edit) in rows {
            assert_eq!(run_rows(edit), refused, "rows: {name}");
        }
        // The null argument itself: the entry tests its pointer before any field (the B1F fix),
        // so a NULL args must be refused, not dereferenced.
        let mut stage = STAGE_NONE;
        // SAFETY: the entry refuses a NULL args pointer before it uses any pointer, as the
        // cases above pin for every argument the entry accepts; `stage` is a live, writable
        // c_int.
        let status = unsafe {
            attn_prep_rows_launch_raw(
                pointer::<AttnPrepRowsArgsRaw>(0).cast_const(),
                pointer(0x1),
                &mut stage,
            )
        };
        assert_eq!((status, stage), refused, "rows: NULL args");
    }

    /// A rows call that would launch (see `PACK_ARGS`): the addresses are made up, and every case
    /// below changes one thing so the entry refuses before it uses any pointer.
    const ROWS_BASE: AttnPrepRowsArgsRaw = AttnPrepRowsArgsRaw {
        proj: pointer::<c_void>(0x1_0000).cast_const(),
        q_norm: pointer::<c_void>(0x1_0010).cast_const(),
        k_norm: pointer::<c_void>(0x1_0020).cast_const(),
        k_rows: pointer(0x2_0000),
        v_rows: pointer(0x3_0000),
        q: pointer(0x4_0000),
        pos: pointer::<c_void>(0x5_0000).cast_const(),
        status: pointer(0x5_0010),
        ld: 9216,
        m: 1,
        n_head: 16,
        n_kv_head: 2,
        head_dim: 256,
        n_rot: 64,
        cells: 4096,
        has_gate: 1,
        slot_mode: 0,
        slot_ring: 0,
        eps: 1e-6,
        rope_base: 1e7,
    };

    /// `(status, stage)` of the rows entry after `edit` of the base arguments.
    fn run_rows(edit: fn(&mut AttnPrepRowsArgsRaw)) -> (c_int, c_int) {
        let mut a = ROWS_BASE;
        edit(&mut a);
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any
        // pointer (see `ROWS_BASE`); `stage` is a live, writable c_int.
        let status = unsafe { attn_prep_rows_launch_raw(&a, pointer(0x1), &mut stage) };
        (status, stage)
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod top_logits;
