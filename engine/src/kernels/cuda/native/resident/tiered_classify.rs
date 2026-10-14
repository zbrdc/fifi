//! Raw FFI to `inference_cuda_tiered_classify_launch` of `native/resident/tiered_classify.cu` (declared in
//! `native/resident/tiered_classify.h`). Child module of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Original code; design from `ledger/plans/TCAP-DESIGN.md` section 3.2 (TCAP C2).
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Shape, sizes, alignment and disjointness are proved again here (`super::Span` for the
//!   overlaps), so a bad call is refused with `invalid_argument` and never reaches the device. The
//!   ranges the kernel writes (`slot_ids_out`, `weights_out`, `log_out`, `ctrl`, `word`, `mbox_req`,
//!   `mbox_row`) share no byte with any other range; the four inputs may alias one another.
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

const MAX_EXPERT: usize = 4096;
const MAX_TOP_K: usize = 8;
const MAX_HIDDEN: usize = 65536;
const MAX_LAYER: usize = 1023;
const LOG_BYTES_PER_PICK: usize = 8;
const CTRL_BYTES: usize = 64;
const REQUEST_BYTES: usize = 128;

/// Field for field the `inference_cuda_tiered_classify_args` of `tiered_classify.h`.
#[repr(C)]
struct ClassifyArgsRaw {
    ids: *const c_void,
    weights: *const c_void,
    slot_row: *const c_void,
    norm_row: *const c_void,
    slot_ids_out: *mut c_void,
    weights_out: *mut c_void,
    log_out: *mut c_void,
    ctrl: *mut c_void,
    word: *mut c_void,
    mbox_req: *mut c_void,
    mbox_row: *mut c_void,
    n_expert: c_int,
    slots: c_int,
    top_k: c_int,
    hidden: c_int,
    layer: c_int,
}

// The `static_assert` of the header, repeated: a drift of either side stops the build.
const _: () = {
    assert!(size_of::<ClassifyArgsRaw>() == 112);
    assert!(offset_of!(ClassifyArgsRaw, ids) == 0);
    assert!(offset_of!(ClassifyArgsRaw, weights) == 8);
    assert!(offset_of!(ClassifyArgsRaw, slot_row) == 16);
    assert!(offset_of!(ClassifyArgsRaw, norm_row) == 24);
    assert!(offset_of!(ClassifyArgsRaw, slot_ids_out) == 32);
    assert!(offset_of!(ClassifyArgsRaw, weights_out) == 40);
    assert!(offset_of!(ClassifyArgsRaw, log_out) == 48);
    assert!(offset_of!(ClassifyArgsRaw, ctrl) == 56);
    assert!(offset_of!(ClassifyArgsRaw, word) == 64);
    assert!(offset_of!(ClassifyArgsRaw, mbox_req) == 72);
    assert!(offset_of!(ClassifyArgsRaw, mbox_row) == 80);
    assert!(offset_of!(ClassifyArgsRaw, n_expert) == 88);
    assert!(offset_of!(ClassifyArgsRaw, slots) == 92);
    assert!(offset_of!(ClassifyArgsRaw, top_k) == 96);
    assert!(offset_of!(ClassifyArgsRaw, hidden) == 100);
    assert!(offset_of!(ClassifyArgsRaw, layer) == 104);
};

unsafe extern "C" {
    #[link_name = "inference_cuda_tiered_classify_launch"]
    fn tiered_classify_launch_raw(
        args: *const ClassifyArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one classification launch reads and writes; the layout of every range is in
/// `tiered_classify.h`.
pub(in crate::kernels::cuda) struct ClassifyRaw<'a> {
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) weights: RawRange<'a>,
    pub(in crate::kernels::cuda) slot_row: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_row: RawRange<'a>,
    pub(in crate::kernels::cuda) slot_ids_out: RawRange<'a>,
    pub(in crate::kernels::cuda) weights_out: RawRange<'a>,
    pub(in crate::kernels::cuda) log_out: RawRange<'a>,
    pub(in crate::kernels::cuda) ctrl: RawRange<'a>,
    pub(in crate::kernels::cuda) word: RawRange<'a>,
    pub(in crate::kernels::cuda) mbox_req: RawRange<'a>,
    pub(in crate::kernels::cuda) mbox_row: RawRange<'a>,
    pub(in crate::kernels::cuda) n_expert: c_int,
    pub(in crate::kernels::cuda) slots: c_int,
    pub(in crate::kernels::cuda) top_k: c_int,
    pub(in crate::kernels::cuda) hidden: c_int,
    pub(in crate::kernels::cuda) layer: c_int,
}

/// The five integers of a call inside the limits of `valid()` in the `.cu` file, as `usize`.
struct Shape {
    n_expert: usize,
    top_k: usize,
    hidden: usize,
}

/// The shape of the integers of `call`, or `None` outside the limits (the same test as `shapes_ok`
/// of the `.cu` file).
fn shape(call: &ClassifyRaw<'_>) -> Option<Shape> {
    let [n_expert, slots, top_k, hidden, layer] = [
        call.n_expert,
        call.slots,
        call.top_k,
        call.hidden,
        call.layer,
    ]
    .map(|v| usize::try_from(v).ok());
    let (n_expert, slots, top_k, hidden, layer) = (n_expert?, slots?, top_k?, hidden?, layer?);
    let inside = (1..=MAX_EXPERT).contains(&n_expert)
        && (1..=n_expert).contains(&slots)
        && (1..=MAX_TOP_K).contains(&top_k)
        && (4..=MAX_HIDDEN).contains(&hidden)
        && hidden.is_multiple_of(4)
        && layer <= MAX_LAYER;
    inside.then_some(Shape {
        n_expert,
        top_k,
        hidden,
    })
}

/// One range of a call: the range, the bytes it must hold, its alignment, and whether the kernel
/// writes it.
type Region<'c, 'a> = (&'c RawRange<'a>, usize, usize, bool);

/// The eleven ranges of a call.
fn regions<'c, 'a>(call: &'c ClassifyRaw<'a>, s: &Shape) -> [Region<'c, 'a>; 11] {
    let (top, row) = (s.top_k * 4, s.hidden * 4);
    [
        (&call.ids, top, 4, false),
        (&call.weights, top, 4, false),
        (&call.slot_row, s.n_expert * 4, 4, false),
        (&call.norm_row, row, 16, false),
        (&call.slot_ids_out, top, 4, true),
        (&call.weights_out, top, 4, true),
        (&call.log_out, s.top_k * LOG_BYTES_PER_PICK, 8, true),
        (&call.ctrl, CTRL_BYTES, 8, true),
        (&call.word, 4, 4, true),
        (&call.mbox_req, REQUEST_BYTES, 8, true),
        (&call.mbox_row, row, 16, true),
    ]
}

/// Proves a classification launch before the native call: the limits, the size and alignment of
/// every range, and that every range the kernel writes overlaps no other range.
fn check_classify(call: &ClassifyRaw<'_>) -> Result<(), DeviceFailure> {
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

/// Queues the classification of `call` on `stream`: one kernel. Returns when it is queued.
pub(in crate::kernels::cuda) fn tiered_classify_launch(
    stream: &RawStream,
    call: &ClassifyRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_classify(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_classify` proved every range inside its live allocation (`RawRange::new`),
    // large enough for the call, aligned as the entry requires, and every range the kernel writes
    // disjoint from every other range. The entry checks the addresses again. The allocations stay
    // valid until the queued work finishes because `inference_cuda_free` synchronizes the device
    // first, and `args` lives until the entry returns (it copies the struct into the launch
    // parameters). `stream` is live for the call.
    let status = unsafe {
        let args = ClassifyArgsRaw {
            ids: call.ids.address().cast_const(),
            weights: call.weights.address().cast_const(),
            slot_row: call.slot_row.address().cast_const(),
            norm_row: call.norm_row.address().cast_const(),
            slot_ids_out: call.slot_ids_out.address(),
            weights_out: call.weights_out.address(),
            log_out: call.log_out.address(),
            ctrl: call.ctrl.address(),
            word: call.word.address(),
            mbox_req: call.mbox_req.address(),
            mbox_row: call.mbox_row.address(),
            n_expert: call.n_expert,
            slots: call.slots,
            top_k: call.top_k,
            hidden: call.hidden,
            layer: call.layer,
        };
        tiered_classify_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

#[cfg(test)]
mod tests;
