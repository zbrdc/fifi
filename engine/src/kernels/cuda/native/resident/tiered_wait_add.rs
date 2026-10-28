//! Raw FFI to `inference_cuda_tiered_wait_add_launch` of `native/resident/tiered_wait_add.cu` (declared
//! in `native/resident/tiered_wait_add.h`). Child module of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Original code; design from `ledger/plans/TCAP-DESIGN.md` section 3.2 (TCAP C3).
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Shape, sizes, alignment and disjointness are proved again here (`super::Span` for the
//!   overlaps), so a bad call is refused with `invalid_argument` and never reaches the device. The
//!   ranges the kernel writes (`resid`, `word`) share no byte with any other range; the read ranges
//!   may alias one another.
//! - The C struct is built in one `unsafe` block that cites those checks.
//! - Every allocation stays valid until queued work finishes, because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`; the launch is asynchronous.
//! - The extern declaration, the `repr(C)` mirror and every raw pointer stay private to this file;
//!   nothing is re-exported.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for TCAP C6 (ledger/plans/TCAP-DESIGN.md)")
)]

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

const MAX_HIDDEN: usize = 65536;
const MAX_TIMEOUT_NS: u64 = 120_000_000_000;
const CTRL_BYTES: usize = 64;
const ANSWER_BYTES: usize = 128;

/// Field for field the `inference_cuda_tiered_wait_add_args` of `tiered_wait_add.h`.
#[repr(C)]
struct WaitAddArgsRaw {
    ctrl: *const c_void,
    word: *mut c_void,
    mbox_ans: *const c_void,
    mbox_partial: *const c_void,
    resid: *mut c_void,
    hidden: c_int,
    timeout_ns: u64,
}

// The `static_assert` of the header, repeated: a drift of either side stops the build.
const _: () = {
    assert!(size_of::<WaitAddArgsRaw>() == 56);
    assert!(offset_of!(WaitAddArgsRaw, ctrl) == 0);
    assert!(offset_of!(WaitAddArgsRaw, word) == 8);
    assert!(offset_of!(WaitAddArgsRaw, mbox_ans) == 16);
    assert!(offset_of!(WaitAddArgsRaw, mbox_partial) == 24);
    assert!(offset_of!(WaitAddArgsRaw, resid) == 32);
    assert!(offset_of!(WaitAddArgsRaw, hidden) == 40);
    assert!(offset_of!(WaitAddArgsRaw, timeout_ns) == 48);
};

unsafe extern "C" {
    #[link_name = "inference_cuda_tiered_wait_add_launch"]
    fn tiered_wait_add_launch_raw(
        args: *const WaitAddArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one wait-add launch reads and writes; the layout of every range is in `tiered_wait_add.h`.
pub(in crate::kernels::cuda) struct WaitAddRaw<'a> {
    pub(in crate::kernels::cuda) ctrl: RawRange<'a>,
    pub(in crate::kernels::cuda) word: RawRange<'a>,
    pub(in crate::kernels::cuda) mbox_ans: RawRange<'a>,
    pub(in crate::kernels::cuda) mbox_partial: RawRange<'a>,
    pub(in crate::kernels::cuda) resid: RawRange<'a>,
    pub(in crate::kernels::cuda) hidden: c_int,
    pub(in crate::kernels::cuda) timeout_ns: u64,
}

/// The integers of a call inside the limits of `valid()` in the `.cu` file, as `usize`.
struct Shape {
    hidden: usize,
}

/// The shape of the integers of `call`, or `None` outside the limits (the same test as `shapes_ok`
/// of the `.cu` file).
fn shape(call: &WaitAddRaw<'_>) -> Option<Shape> {
    let hidden = usize::try_from(call.hidden).ok()?;
    let inside = (4..=MAX_HIDDEN).contains(&hidden)
        && hidden.is_multiple_of(4)
        && (1..=MAX_TIMEOUT_NS).contains(&call.timeout_ns);
    inside.then_some(Shape { hidden })
}

/// One range of a call: the range, the bytes it must hold, its alignment, and whether the kernel
/// writes it.
type Region<'c, 'a> = (&'c RawRange<'a>, usize, usize, bool);

/// The five ranges of a call.
fn regions<'c, 'a>(call: &'c WaitAddRaw<'a>, s: &Shape) -> [Region<'c, 'a>; 5] {
    let row = s.hidden * 4;
    [
        (&call.ctrl, CTRL_BYTES, 8, false),
        (&call.word, 4, 4, true),
        (&call.mbox_ans, ANSWER_BYTES, 8, false),
        (&call.mbox_partial, row, 16, false),
        (&call.resid, row, 16, true),
    ]
}

/// Proves a wait-add launch before the native call: the limits, the size and alignment of every
/// range, and that every range the kernel writes (`resid`, `word`) overlaps no other range.
fn check_wait_add(call: &WaitAddRaw<'_>) -> Result<(), DeviceFailure> {
    let s = shape(call).ok_or_else(invalid_argument)?;
    let regions = regions(call, &s);
    let sized = regions
        .iter()
        .all(|&(range, need, align, _)| range.bytes >= need && range.offset.is_multiple_of(align));
    let spans: Vec<(Span<'_>, bool)> = regions
        .iter()
        .map(|&(range, need, _, written)| (Span::of(range, need), written))
        .collect();
    let disjoint = spans.iter().enumerate().all(|(i, (a, a_written))| {
        spans[i + 1..]
            .iter()
            .all(|(b, b_written)| !(*a_written || *b_written) || a.apart(b))
    });
    if !(sized && disjoint) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the wait-add of `call` on `stream`: one kernel. Returns when it is queued.
pub(in crate::kernels::cuda) fn tiered_wait_add_launch(
    stream: &RawStream,
    call: &WaitAddRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_wait_add(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_wait_add` proved every range inside its live allocation (`RawRange::new`),
    // large enough for the call, aligned as the entry requires, and every range the kernel writes
    // disjoint from every other range. The entry checks the addresses again. The allocations stay
    // valid until the queued work finishes because `inference_cuda_free` synchronizes the device
    // first, and `args` lives until the entry returns (it copies the struct into the launch
    // parameters). `stream` is live for the call.
    let status = unsafe {
        let args = WaitAddArgsRaw {
            ctrl: call.ctrl.address().cast_const(),
            word: call.word.address(),
            mbox_ans: call.mbox_ans.address().cast_const(),
            mbox_partial: call.mbox_partial.address().cast_const(),
            resid: call.resid.address(),
            hidden: call.hidden,
            timeout_ns: call.timeout_ns,
        };
        tiered_wait_add_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

#[cfg(test)]
mod tests;
