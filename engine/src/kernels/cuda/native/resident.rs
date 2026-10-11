//! Raw FFI to the resident-decode entries of `native/stream.cu` and `native/projector.cu`
//! (declared in `native/abi_resident.h`). Child module of `native`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants every wrapper here relies on:
//! - A handle (`RawStream`, `RawEvent`, `RawPinned`) is created once and consumed once by its
//!   destroy function. It has no `Drop`: the owner in `context` calls destroy so that a failed
//!   destroy is reported and latches the backend (decision 0022).
//! - A device range is proved in bounds before the native call (`RawRange::new`). The launch
//!   wrappers (`pack_launch`, `matmul_launch`) also prove sizes, alignment and, between the ranges
//!   one call writes and the ranges it reads, disjointness; the native entries repeat these checks
//!   on the addresses. The allocation stays valid until queued work finishes because
//!   `inference_cuda_free` synchronizes the whole device before `cudaFree`.
//! - Launch-only wrappers return the launch status only. A fault in queued work surfaces at
//!   the next synchronize.
//! - Every call runs under the process dispatch guard and on the thread that owns the context
//!   (decisions 0023 and 0024). Pinned memory is read only by that thread and only after an
//!   event recorded after the copy completed.
//! - Stream capture and executable graphs (`RawGraphExec`) live in the child `graph` (RD-2d), which
//!   states its own invariants: a capture begins and ends on one thread, and a graph is consumed once.
//! - No raw pointer and no unsafe function leaves this module; handles are opaque.
//!
//! Per-kernel entries live in child modules `native/resident/<n>.rs`, one per kernel, declared
//! by one line after the imports below. A child starts with its own safety-invariants doc and may
//! use this module's private items (`RawStream`, `RawRange`, `checked`, `Span`). The recipe is in
//! the doc of `kernels::cuda::resident`.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use super::{
    DeviceFailure, DeviceView, MAX_DIM, RawDevice, STAGE_NONE, TILE_COLS, Tiled, activation_bytes,
    checked, invalid_argument, pack_bytes,
};

// One `pub(in crate::kernels::cuda) mod <n>;` line per kernel child module, alphabetical.
pub(in crate::kernels::cuda) mod add_norm_pack;
pub(in crate::kernels::cuda) mod add_rmsnorm_q8;
pub(in crate::kernels::cuda) mod attn_decode;
pub(in crate::kernels::cuda) mod attn_prep;
pub(in crate::kernels::cuda) mod embed;
pub(in crate::kernels::cuda) mod gdn_conv;
pub(in crate::kernels::cuda) mod gdn_delta;
pub(in crate::kernels::cuda) mod graph;
pub(in crate::kernels::cuda) mod matmul_expert;
pub(in crate::kernels::cuda) mod matmul_q5_0;
pub(in crate::kernels::cuda) mod matvec_expert;
pub(in crate::kernels::cuda) mod matvec_q;
pub(in crate::kernels::cuda) mod moe_combine;
pub(in crate::kernels::cuda) mod quantize_q8_1;
pub(in crate::kernels::cuda) mod router;
pub(in crate::kernels::cuda) mod step_params;
pub(in crate::kernels::cuda) mod swiglu_pack;
pub(in crate::kernels::cuda) mod swiglu_q8;

/// A non-blocking CUDA stream. Consumed by [`stream_destroy`].
#[derive(Debug)]
#[must_use = "a stream must be passed to stream_destroy"]
pub(in crate::kernels::cuda) struct RawStream(NonNull<c_void>);

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
    /// The device address of the first byte of the range.
    fn address(&self) -> *mut c_void {
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

/// Queues a copy of `host` into `dst` on `stream`; `host` must be exactly as long as `dst`.
///
/// The copy runs behind the work queued earlier on `stream`. From pageable memory the driver has
/// staged `host` when the call returns, so the slice need not outlive the call. The caller
/// synchronizes `stream` to learn that the copy finished.
pub(in crate::kernels::cuda) fn copy_h2d_async(
    dst: RawRange<'_>,
    host: &[u8],
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    if host.len() != dst.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: `RawRange::new` proved `offset + bytes <= device.bytes` with `bytes > 0` for the
    // live allocation, and the check above makes `host` exactly `dst.bytes` initialized bytes,
    // valid for the call. `stream` is live for the call.
    let status = unsafe {
        copy_h2d_async_raw(
            dst.address(),
            host.as_ptr().cast(),
            host.len(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Queues a copy of `src` into `host` on `stream`; `host` must be exactly as long as `src`.
///
/// The copy runs behind the work queued earlier on `stream`. Into pageable memory the driver
/// returns only after the copy completed, so `host` holds the bytes when the call returns
/// successfully.
pub(in crate::kernels::cuda) fn copy_d2h_pageable(
    src: RawRange<'_>,
    host: &mut [u8],
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    if host.len() != src.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: `RawRange::new` proved the source range inside the live allocation, and the check
    // above makes `host` an exclusive, writable slice of exactly `src.bytes` bytes. For pageable
    // memory the driver does not return before it finished writing them. `stream` is live for
    // the call.
    let status = unsafe {
        copy_d2h_async_raw(
            host.as_mut_ptr().cast(),
            src.address(),
            src.bytes,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
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
struct Span<'a> {
    device: &'a RawDevice,
    start: usize,
    len: usize,
}

impl<'a> Span<'a> {
    fn of(range: &RawRange<'a>, len: usize) -> Self {
        Self {
            device: range.device,
            start: range.offset,
            len,
        }
    }

    /// True when the two spans share no byte. Every span lies inside its
    /// allocation, so the sums cannot overflow; a saturated sum would only refuse.
    fn apart(&self, other: &Self) -> bool {
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
            copy_h2d_async(range, &[0; 7], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_h2d_async(range, &[0; 9], &stream),
            Err(invalid_argument())
        );
        assert_eq!(copy_h2d_async(range, &[], &stream), Err(invalid_argument()));
        assert_eq!(
            copy_d2h_pageable(range, &mut [0; 7], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_d2h_pageable(range, &mut [0; 9], &stream),
            Err(invalid_argument())
        );
        assert_eq!(
            copy_d2h_pageable(range, &mut [], &stream),
            Err(invalid_argument())
        );
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

    fn pointer<T>(address: usize) -> *mut T {
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
    }
}
