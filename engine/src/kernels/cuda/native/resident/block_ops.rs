//! Raw FFI to inference_cuda_block_ops_launch of native/resident/block_ops.cu (declared in
//! native/resident/block_ops.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves that each slot the call
//!   uses holds the bytes the wrapper named and that no range the kernel writes shares a byte
//!   with another slot. The C entry repeats the per-operation checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid until then
//!   because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread. No raw pointer
//!   leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

pub(in crate::kernels::cuda) const PTRS: usize = 12;
pub(in crate::kernels::cuda) const INTS: usize = 12;
pub(in crate::kernels::cuda) const FLOATS: usize = 4;

unsafe extern "C" {
    #[link_name = "inference_cuda_block_ops_launch_mode"]
    fn block_ops_launch_mode_raw(
        args: *const BlockOpsArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
        rounded_q8: c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_block_ops_args, field for field.
#[repr(C)]
struct BlockOpsArgsRepr {
    op: c_int,
    ints: [c_int; INTS],
    floats: [f32; FLOATS],
    ptr: [*mut c_void; PTRS],
    bytes: [usize; PTRS],
}

const _: () = assert!(
    size_of::<BlockOpsArgsRepr>() == 264
        && offset_of!(BlockOpsArgsRepr, ints) == 4
        && offset_of!(BlockOpsArgsRepr, floats) == 52
        && offset_of!(BlockOpsArgsRepr, ptr) == 72
        && offset_of!(BlockOpsArgsRepr, bytes) == 168
);

/// One used slot: the range, the bytes the operation touches in it, and whether it writes.
pub(in crate::kernels::cuda) struct SlotRaw<'a> {
    pub(in crate::kernels::cuda) range: RawRange<'a>,
    pub(in crate::kernels::cuda) need: usize,
    pub(in crate::kernels::cuda) written: bool,
}

pub(in crate::kernels::cuda) struct BlockOpsRaw<'a> {
    pub(in crate::kernels::cuda) op: i32,
    pub(in crate::kernels::cuda) ints: [i32; INTS],
    pub(in crate::kernels::cuda) floats: [f32; FLOATS],
    pub(in crate::kernels::cuda) slots: [Option<SlotRaw<'a>>; PTRS],
}

fn check_block_ops(call: &BlockOpsRaw<'_>) -> Result<(), DeviceFailure> {
    for (index, slot) in call.slots.iter().enumerate() {
        let Some(slot) = slot else { continue };
        if slot.need == 0
            || slot.range.bytes() < slot.need
            || !slot.range.offset().is_multiple_of(4)
        {
            return Err(invalid_argument());
        }
        if !slot.written {
            continue;
        }
        let mine = Span::of(&slot.range, slot.need);
        for (other_index, other) in call.slots.iter().enumerate() {
            let Some(other) = other else { continue };
            if other_index != index && !mine.apart(&Span::of(&other.range, other.need)) {
                return Err(invalid_argument());
            }
        }
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn block_ops_launch(
    stream: &RawStream,
    call: &BlockOpsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_block_ops(call)?;
    let mut args = BlockOpsArgsRepr {
        op: call.op,
        ints: call.ints,
        floats: call.floats,
        ptr: [std::ptr::null_mut(); PTRS],
        bytes: [0; PTRS],
    };
    for (index, slot) in call.slots.iter().enumerate() {
        if let Some(slot) = slot {
            args.ptr[index] = slot.range.address();
            args.bytes[index] = slot.range.bytes();
        }
    }
    let mut stage = STAGE_NONE;
    // SAFETY: check_block_ops proved every used slot inside its live allocation, large enough,
    // aligned, and disjoint in the write direction; the adapter repeats the per-operation checks
    // on the addresses. The allocations stay valid until the queued launch finishes because
    // inference_cuda_free synchronizes the device first.
    let rounded_q8 = c_int::from(super::rounded_q8_1());
    let status =
        unsafe { block_ops_launch_mode_raw(&args, stream.as_ptr(), &mut stage, rounded_q8) };
    checked(status, stage)
}
