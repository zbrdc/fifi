//! Native device shortlist. RawRange proves live allocation ownership; sizes,alignment and
//! input/scratch disjointness are checked again here. The entry only enqueues on a live stream;
//! buffers outlive queued work because native free synchronizes. No pointer escapes this module.
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

pub(in crate::kernels::cuda) const K: usize = 20;
pub(in crate::kernels::cuda) const RESULT_BYTES: usize = 176;
pub(in crate::kernels::cuda) const RESULT_OFFSET: usize = 2816;
pub(in crate::kernels::cuda) const SCRATCH_BYTES: usize = 2992;
pub(in crate::kernels::cuda) const MAX_LOGITS: usize = 1 << 30;

#[repr(C)]
struct Args {
    logits: *const c_void,
    scratch: *mut c_void,
    n: c_int,
}
const _: () = assert!(size_of::<Args>() == 24 && offset_of!(Args, n) == 16);
unsafe extern "C" {
    #[link_name = "inference_cuda_shortlist_launch"]
    fn launch_raw(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
}
pub(in crate::kernels::cuda) struct ShortlistRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) scratch: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
}
pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    io: ShortlistRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = io.n.checked_mul(4).ok_or_else(invalid_argument)?;
    if !(1..=MAX_LOGITS).contains(&io.n)
        || io.logits.bytes < bytes
        || !io.logits.offset.is_multiple_of(4)
        || io.scratch.bytes < SCRATCH_BYTES
        || !io.scratch.offset.is_multiple_of(16)
        || !Span::of(&io.logits, bytes).apart(&Span::of(&io.scratch, SCRATCH_BYTES))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        logits: io.logits.address().cast_const(),
        scratch: io.scratch.address(),
        n: io.n as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live ranges above hold the checked extents,alignments and disjoint spans.
    // The live stream/argument struct outlive this launch-only call;native free drains work.
    let status = unsafe { launch_raw(&raw const args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
