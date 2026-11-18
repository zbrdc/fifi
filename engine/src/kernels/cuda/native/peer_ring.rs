//! Raw FFI to the peer mailbox exchange entries of `native/peer_ring.cu`
//! (declared in `native/abi_ring.h`). Child module of `native`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants every wrapper here relies on:
//! - Local ranges are proved in bounds by `RawRange::new` in the safe layer
//!   and re-checked here for the byte counts the kernels read and write.
//! - A peer slot is a bare device address (`u64`) of another device's live
//!   mailbox allocation. It cannot be a `RawRange`: no local handle describes
//!   it. The safe layer records these addresses once, at ring construction,
//!   from the owning contexts' live allocations, and the ring owns every
//!   mailbox allocation for the ring's whole life, so the addresses stay
//!   valid until queued work finishes (the free path synchronizes the whole
//!   device first). Peer access between every pair was enabled before the
//!   ring was built (`peer::enable_peer_access`).
//! - Every call queues work on the caller's stream under the dispatch guard
//!   and returns the launch status only; a timeout or overflow in queued
//!   work is reported in the caller's status word, not here.
//! - No raw pointer and no unsafe function leaves this module.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device decode step, its first production caller"
    )
)]

use std::ffi::{c_int, c_uint, c_ulonglong, c_void};

use super::resident::{RawRange, RawStream};
use super::{DeviceFailure, checked, invalid_argument};

/// Matches `kMaxValues` in `native/peer_ring.cu`.
pub(in crate::kernels::cuda) const MAX_VALUES: usize = 1 << 21;
/// The single-block FP16 publisher retains its previously supported bound.
pub(in crate::kernels::cuda) const MAX_F16_VALUES: usize = 1 << 20;
/// Matches `kMaxPeers` in `native/peer_ring.cu`.
pub(in crate::kernels::cuda) const MAX_PEERS: usize = 3;
/// The wait-sum status bit for a peer that never published.
pub(in crate::kernels::cuda) const STATUS_TIMEOUT: u32 = 1;
/// The wait-sum status bit for a received overflow flag.
pub(in crate::kernels::cuda) const STATUS_OVERFLOW: u32 = 2;

unsafe extern "C" {
    #[link_name = "inference_cuda_ring_publish_f16"]
    fn publish_f16_raw(
        stream: *mut c_void,
        src: *const f32,
        dst_slot: *mut c_void,
        count: c_int,
        seq: c_uint,
    ) -> c_int;
    #[link_name = "inference_cuda_ring_publish_f32"]
    fn publish_f32_raw(
        stream: *mut c_void,
        src: *const f32,
        dst_slot: *mut c_void,
        count: c_int,
        seq: c_uint,
    ) -> c_int;
    #[link_name = "inference_cuda_ring_wait_copy_f32"]
    fn wait_copy_f32_raw(
        stream: *mut c_void,
        out: *mut f32,
        peer_slot: *const c_void,
        count: c_int,
        seq: c_uint,
        timeout_cycles: c_ulonglong,
        status: *mut c_uint,
    ) -> c_int;
    #[link_name = "inference_cuda_ring_wait_sum_f16"]
    fn wait_sum_f16_raw(
        stream: *mut c_void,
        out: *mut f32,
        own: *const f32,
        peer_slot0: *const c_void,
        peer_slot1: *const c_void,
        peer_slot2: *const c_void,
        peer_count: c_int,
        count: c_int,
        seq: c_uint,
        timeout_cycles: c_ulonglong,
        status: *mut c_uint,
    ) -> c_int;
    #[link_name = "inference_cuda_ring_wait_sum_f32"]
    fn wait_sum_f32_raw(
        stream: *mut c_void,
        out: *mut f32,
        own: *const f32,
        peer_slot0: *const c_void,
        peer_slot1: *const c_void,
        peer_slot2: *const c_void,
        peer_count: c_int,
        count: c_int,
        seq: c_uint,
        timeout_cycles: c_ulonglong,
        status: *mut c_uint,
    ) -> c_int;
}

fn slot_ptr(address: u64) -> *mut c_void {
    address as usize as *mut c_void
}

fn checked_count(count: usize, ranges: &[RawRange<'_>]) -> Result<c_int, DeviceFailure> {
    let bytes = count.checked_mul(4).ok_or_else(invalid_argument)?;
    if count == 0 || count > MAX_VALUES || ranges.iter().any(|range| range.bytes() < bytes) {
        return Err(invalid_argument());
    }
    c_int::try_from(count).map_err(|_| invalid_argument())
}

/// Queues the publish of `count` f32 values of `src` into the peer slot at
/// device address `dst_slot`, converted to fp16 unless `f32_payload`.
#[allow(clippy::too_many_arguments)]
pub(in crate::kernels::cuda) fn publish(
    stream: &RawStream,
    src: RawRange<'_>,
    dst_slot: u64,
    count: usize,
    seq: u32,
    f32_payload: bool,
) -> Result<(), DeviceFailure> {
    if !f32_payload && count > MAX_F16_VALUES {
        return Err(invalid_argument());
    }
    let count = checked_count(count, &[src])?;
    if seq == 0 || !dst_slot.is_multiple_of(16) {
        return Err(invalid_argument());
    }
    let src_ptr = src.address().cast::<f32>();
    // SAFETY: `src` covers `count` f32 values of a live allocation (proved
    // above); `dst_slot` is a slot of a live peer mailbox this process may
    // access directly, recorded by the safe layer at ring construction. The
    // entry only queues the kernel on `stream`.
    let status = unsafe {
        if f32_payload {
            publish_f32_raw(stream.as_ptr(), src_ptr, slot_ptr(dst_slot), count, seq)
        } else {
            publish_f16_raw(stream.as_ptr(), src_ptr, slot_ptr(dst_slot), count, seq)
        }
    };
    checked(status, super::STAGE_NONE)
}

/// Queues the wait-sum: spin until the `peer_slots` (device addresses in this
/// device's own mailbox) carry `seq`, then write `own + payloads` to `out`.
#[allow(clippy::too_many_arguments)]
pub(in crate::kernels::cuda) fn wait_sum(
    stream: &RawStream,
    out: RawRange<'_>,
    own: RawRange<'_>,
    peer_slots: &[u64],
    count: usize,
    seq: u32,
    f32_payload: bool,
    timeout_cycles: u64,
    status_word: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    if !f32_payload && count > MAX_F16_VALUES {
        return Err(invalid_argument());
    }
    let count = checked_count(count, &[out, own])?;
    if seq == 0
        || peer_slots.len() > MAX_PEERS
        || peer_slots.iter().any(|&slot| !slot.is_multiple_of(16))
        || status_word.bytes() < 4
    {
        return Err(invalid_argument());
    }
    let mut slots = [std::ptr::null::<c_void>(); MAX_PEERS];
    for (index, &slot) in peer_slots.iter().enumerate() {
        slots[index] = slot_ptr(slot).cast_const();
    }
    let peer_count = c_int::try_from(peer_slots.len()).map_err(|_| invalid_argument())?;
    // SAFETY: `out` and `own` cover `count` f32 values and `status_word` one
    // u32 of live allocations (proved above); the slots are in this device's
    // live mailbox allocation, recorded by the safe layer. The entry only
    // queues the kernel on `stream`.
    let status = unsafe {
        let (out_ptr, own_ptr, word_ptr) = (
            out.address().cast::<f32>(),
            own.address().cast::<f32>(),
            status_word.address().cast::<c_uint>(),
        );
        if f32_payload {
            wait_sum_f32_raw(
                stream.as_ptr(),
                out_ptr,
                own_ptr,
                slots[0],
                slots[1],
                slots[2],
                peer_count,
                count,
                seq,
                timeout_cycles,
                word_ptr,
            )
        } else {
            wait_sum_f16_raw(
                stream.as_ptr(),
                out_ptr,
                own_ptr,
                slots[0],
                slots[1],
                slots[2],
                peer_count,
                count,
                seq,
                timeout_cycles,
                word_ptr,
            )
        }
    };
    checked(status, super::STAGE_NONE)
}

/// Queues the wait-copy: spin until the slot at device address `peer_slot`
/// carries `seq`, then copy its `count` f32 payload values into `out`.
#[allow(clippy::too_many_arguments)]
pub(in crate::kernels::cuda) fn wait_copy(
    stream: &RawStream,
    out: RawRange<'_>,
    peer_slot: u64,
    count: usize,
    seq: u32,
    timeout_cycles: u64,
    status_word: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    let count = checked_count(count, &[out])?;
    if seq == 0 || !peer_slot.is_multiple_of(16) || status_word.bytes() < 4 {
        return Err(invalid_argument());
    }
    // SAFETY: `out` covers `count` f32 values and `status_word` one u32 of
    // live allocations (proved above); the slot is in a live mailbox
    // allocation recorded by the safe layer. The entry only queues.
    let status = unsafe {
        wait_copy_f32_raw(
            stream.as_ptr(),
            out.address().cast::<f32>(),
            slot_ptr(peer_slot).cast_const(),
            count,
            seq,
            timeout_cycles,
            status_word.address().cast::<c_uint>(),
        )
    };
    checked(status, super::STAGE_NONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_retains_its_old_native_limit_before_launch() {
        // SAFETY: aligned, non-null sentinel addresses are never dereferenced;
        // the payload limit rejects both calls before any CUDA launch.
        let status = unsafe {
            publish_f16_raw(
                std::ptr::null_mut(),
                16usize as *const f32,
                16usize as *mut c_void,
                (MAX_F16_VALUES + 1) as c_int,
                1,
            )
        };
        assert_eq!(status, super::super::STATUS_INVALID_ARGUMENT);
        // SAFETY: as above, the adapter rejects the count before a launch.
        let status = unsafe {
            wait_sum_f16_raw(
                std::ptr::null_mut(),
                16usize as *mut f32,
                16usize as *const f32,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                (MAX_F16_VALUES + 1) as c_int,
                1,
                0,
                16usize as *mut c_uint,
            )
        };
        assert_eq!(status, super::super::STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn fp32_refuses_payloads_above_the_wide_limit_before_launch() {
        // SAFETY: the aligned sentinels are never dereferenced; the count
        // is above the native limit and is rejected before any CUDA launch.
        let status = unsafe {
            publish_f32_raw(
                std::ptr::null_mut(),
                16usize as *const f32,
                16usize as *mut c_void,
                (MAX_VALUES + 1) as c_int,
                1,
            )
        };
        assert_eq!(status, super::super::STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn the_native_entry_rejects_bad_arguments_before_any_cuda_call() {
        // SAFETY: every argument is invalid in a way the adapter checks before
        // any CUDA call, so nothing here touches a device.
        let status = unsafe {
            publish_f16_raw(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                0,
            )
        };
        assert_eq!(status, super::super::STATUS_INVALID_ARGUMENT);
        // SAFETY: same adapter-side argument checks, on the wait-sum entry.
        let status = unsafe {
            wait_sum_f32_raw(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                4,
                1,
                1,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, super::super::STATUS_INVALID_ARGUMENT);
        // SAFETY: same adapter-side argument checks, on the wait-copy entry.
        let status = unsafe {
            wait_copy_f32_raw(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                1,
                0,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, super::super::STATUS_INVALID_ARGUMENT);
    }
}
