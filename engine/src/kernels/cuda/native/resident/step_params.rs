//! Raw entries of `native/resident/step_params.cu`: `step_begin`, `step_advance` and the greedy
//! `argmax` (RD-4d, decision 0047). Design guidance only, no code copied: the per-launch data table
//! (C3) and the device-written next token (C1) of the MegaCapybara lessons are ideas, not code.
//!
//! Safety invariants this file relies on:
//! - Each entry is called only through `CudaContext::launch_on`, on the owning thread and under the
//!   dispatch guard (decisions 0023 and 0024). The entries only enqueue and return.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, before the call, on the
//!   extents the kernels touch; the C entries repeat the alignment checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The C argument structs are live locals that the entries read only during the call. Their
//!   `repr(C)` mirrors are private here and pinned by `const` asserts to the `static_assert`s of
//!   `step_params.h`. No raw pointer and no unsafe function leaves this module.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Bytes of the `StepParams` block (`inference_cuda_step_params`).
pub(in crate::kernels::cuda) const STEP_PARAMS_BYTES: usize = 32;
/// Bytes of the argmax scratch: 256 values, 256 indices and 256 flags of 4 bytes.
pub(in crate::kernels::cuda) const ARGMAX_PART_BYTES: usize = 3072;
/// Bytes of the result pair: the token, then the status.
pub(in crate::kernels::cuda) const RESULT_PAIR_BYTES: usize = 8;
/// Largest `m` of one `step_advance`.
pub(in crate::kernels::cuda) const MAX_ADVANCE: usize = 4096;
/// Largest number of logits of one argmax.
pub(in crate::kernels::cuda) const MAX_LOGITS: usize = 1 << 30;

/// Alignment of every `StepParams`, logits and scratch pointer.
const WORD_ALIGN: usize = 4;
/// Alignment of the result pair (condition R-C2a of the packet).
const PAIR_ALIGN: usize = 8;

/// Mirror of `inference_cuda_step_advance_args`.
#[repr(C)]
struct AdvanceArgsRaw {
    params: *mut c_void,
    m: c_int,
}

/// Mirror of `inference_cuda_argmax_args`.
#[repr(C)]
struct ArgmaxArgsRaw {
    logits: *const c_void,
    part: *mut c_void,
    params: *mut c_void,
    result: *mut c_void,
    n: c_int,
}

const _: () = assert!(
    size_of::<AdvanceArgsRaw>() == 16
        && offset_of!(AdvanceArgsRaw, params) == 0
        && offset_of!(AdvanceArgsRaw, m) == 8
);
const _: () = assert!(
    size_of::<ArgmaxArgsRaw>() == 40
        && offset_of!(ArgmaxArgsRaw, logits) == 0
        && offset_of!(ArgmaxArgsRaw, part) == 8
        && offset_of!(ArgmaxArgsRaw, params) == 16
        && offset_of!(ArgmaxArgsRaw, result) == 24
        && offset_of!(ArgmaxArgsRaw, n) == 32
);

unsafe extern "C" {
    #[link_name = "inference_cuda_step_begin_launch"]
    fn step_begin_raw(params: *mut c_void, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_step_advance_launch"]
    fn step_advance_raw(
        args: *const AdvanceArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_argmax_launch"]
    fn argmax_raw(args: *const ArgmaxArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
}

/// What one argmax reads and writes. `n` was bounded by the wrapper.
pub(in crate::kernels::cuda) struct ArgmaxRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) params: RawRange<'a>,
    pub(in crate::kernels::cuda) result: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) n: c_int,
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// Proves the `StepParams` range: 32 bytes, 4-byte aligned.
fn check_params(params: &RawRange<'_>) -> Result<(), DeviceFailure> {
    if fits(params, STEP_PARAMS_BYTES, WORD_ALIGN) {
        return Ok(());
    }
    Err(invalid_argument())
}

/// Proves an argmax before the native call: sizes, alignment (`result` 8, the rest 4) and that
/// `part`, `params` and `result`, which the kernels write, share no byte with each other or with
/// `logits`. Returns `n` as a length.
fn check_argmax(call: &ArgmaxRaw<'_>) -> Result<usize, DeviceFailure> {
    let n = usize::try_from(call.n)
        .ok()
        .filter(|n| (1..=MAX_LOGITS).contains(n))
        .ok_or_else(invalid_argument)?;
    let logits_bytes = n.checked_mul(4).ok_or_else(invalid_argument)?;
    check_params(&call.params)?;
    let sized = fits(&call.logits, logits_bytes, WORD_ALIGN)
        && fits(&call.part, ARGMAX_PART_BYTES, WORD_ALIGN)
        && call
            .result
            .as_ref()
            .is_none_or(|result| fits(result, RESULT_PAIR_BYTES, PAIR_ALIGN));
    if !sized {
        return Err(invalid_argument());
    }
    let mut spans = vec![
        Span::of(&call.logits, logits_bytes),
        Span::of(&call.part, ARGMAX_PART_BYTES),
        Span::of(&call.params, STEP_PARAMS_BYTES),
    ];
    if let Some(result) = &call.result {
        spans.push(Span::of(result, RESULT_PAIR_BYTES));
    }
    for (i, a) in spans.iter().enumerate() {
        if !spans[i + 1..].iter().all(|b| a.apart(b)) {
            return Err(invalid_argument());
        }
    }
    Ok(n)
}

/// Queues `step_begin`: clears the status of the block and flags an invalid state.
pub(in crate::kernels::cuda) fn step_begin_launch(
    stream: &RawStream,
    params: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    check_params(&params)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_params` proved `params` inside its live allocation, 32 bytes long and
    // 4-byte aligned; the entry repeats the alignment check on the address. The allocation stays
    // valid until the queued work finishes because `inference_cuda_free` synchronizes the device
    // first. `stream` is a live handle for the call, and `stage` is a live, writable c_int.
    let status = unsafe { step_begin_raw(params.address(), stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues `step_advance` by `m` positions (`1..=4096`, bounded by the wrapper).
pub(in crate::kernels::cuda) fn step_advance_launch(
    stream: &RawStream,
    params: RawRange<'_>,
    m: c_int,
) -> Result<(), DeviceFailure> {
    check_params(&params)?;
    if !usize::try_from(m).is_ok_and(|m| (1..=MAX_ADVANCE).contains(&m)) {
        return Err(invalid_argument());
    }
    let args = AdvanceArgsRaw {
        params: params.address(),
        m,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_params` proved `params` inside its live allocation, 32 bytes long and
    // 4-byte aligned, and `m` is in range; the entry repeats the checks on the address and the
    // allocation outlives the queued work (`inference_cuda_free` synchronizes first). `args` is a
    // live local that the entry reads only during the call; it keeps no pointer to it. `stream`
    // is a live handle for the call.
    let status = unsafe { step_advance_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the two argmax kernels. An absent `result` passes NULL.
pub(in crate::kernels::cuda) fn argmax_launch(
    stream: &RawStream,
    call: &ArgmaxRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_argmax(call)?;
    let args = ArgmaxArgsRaw {
        logits: call.logits.address().cast_const(),
        part: call.part.address(),
        params: call.params.address(),
        result: call
            .result
            .as_ref()
            .map_or(std::ptr::null_mut(), RawRange::address),
        n: call.n,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_argmax` proved every range inside its live allocation, large enough for the
    // call and aligned as the entry requires (4, and 8 for `result`), and that the touched extents
    // of `logits`, `part`, `params` and `result` share no byte, so the writes to `part`, `params`
    // and `result` cannot alias the reads of `logits`. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads only
    // during the call; an absent `result` is the NULL the entry allows. `stream` is a live handle
    // for the call.
    let status = unsafe { argmax_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
