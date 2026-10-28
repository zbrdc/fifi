//! Native page-local position control. Only the dispatch-guarded composite calls this entry.
//! RawRange proves live allocation bounds; sizes, alignment, geometry and overlap are rechecked
//! here and in CUDA. Device frees synchronize; no raw pointer leaves this module.
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::{
    ffi::{c_int, c_void},
    mem::offset_of,
};

#[repr(C)]
struct Args {
    params: *mut c_void,
    local: *mut c_void,
    expected: c_int,
    base: c_int,
    cells: c_int,
    capacity: c_int,
}
const _: () = {
    assert!(size_of::<Args>() == 32);
    assert!(offset_of!(Args, params) == 0);
    assert!(offset_of!(Args, local) == 8);
    assert!(offset_of!(Args, expected) == 16);
    assert!(offset_of!(Args, base) == 20);
    assert!(offset_of!(Args, cells) == 24);
    assert!(offset_of!(Args, capacity) == 28);
};

unsafe extern "C" {
    fn inference_cuda_paged_position_launch(
        args: *const Args,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct PositionRaw<'a> {
    pub(in crate::kernels::cuda) params: RawRange<'a>,
    pub(in crate::kernels::cuda) local: RawRange<'a>,
    pub(in crate::kernels::cuda) expected: c_int,
    pub(in crate::kernels::cuda) base: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
    pub(in crate::kernels::cuda) capacity: c_int,
}

pub(in crate::kernels::cuda) fn geometry_ok(
    expected: i32,
    base: i32,
    cells: i32,
    capacity: i32,
) -> bool {
    const MAX: i32 = (1 << 30) - 1;
    capacity > 0
        && capacity <= MAX
        && expected >= 0
        && expected < capacity
        && base >= 0
        && base <= expected
        && cells > 0
        && cells <= MAX
        && base % cells == 0
        && expected - base < cells
}

pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    call: &PositionRaw<'_>,
) -> Result<(), DeviceFailure> {
    if !geometry_ok(call.expected, call.base, call.cells, call.capacity)
        || call.params.bytes() < 32
        || !call.params.offset().is_multiple_of(4)
        || call.local.bytes() < 4
        || !call.local.offset().is_multiple_of(4)
        || !Span::of(&call.params, 32).apart(&Span::of(&call.local, 4))
    {
        return Err(invalid_argument());
    }
    // SAFETY: live, sized, aligned and disjoint RawRanges and checked geometry above;
    // the caller holds the context dispatch guard and supplies the explicit live stream.
    unsafe {
        let args = Args {
            params: call.params.address(),
            local: call.local.address(),
            expected: call.expected,
            base: call.base,
            cells: call.cells,
            capacity: call.capacity,
        };
        let mut stage = STAGE_NONE;
        let status = inference_cuda_paged_position_launch(&args, stream.as_ptr(), &mut stage);
        checked(status, stage)
    }
}
