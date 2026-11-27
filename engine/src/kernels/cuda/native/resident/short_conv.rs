//! Raw FFI to `inference_cuda_short_conv` of `native/resident/short_conv.cu` (declared in
//! `native/resident/short_conv.h`). Child module of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on:
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Shape, sizes, alignment and disjointness are proved again here (`super::Span` for the
//!   overlaps), so a bad call is refused with `invalid_argument` and never reaches the entry.
//!   `history`, `kernel` and `health` share no byte with any other range; `out` is exactly
//!   `input` or apart from it, since a partial overlap would race.
//! - The entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard; it queues one launch and never synchronizes.
//! - Every allocation stays valid until queued work finishes, because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The extern declaration and every raw pointer stay inside this module; no unsafe function
//!   leaves it.

use std::ffi::{c_int, c_void};
use std::mem::size_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest `taps` of a convolution: the host's `sconv_kernel_size` bound.
pub(crate) const MAX_TAPS: usize = 8;
const MAX_WIDTH: usize = 8192;

unsafe extern "C" {
    fn inference_cuda_short_conv(
        input: *const c_void,
        kernel: *const c_void,
        history: *mut c_void,
        out: *mut c_void,
        state: *mut c_void,
        health: *mut c_void,
        width: c_int,
        taps: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The ranges of one launch; the extents come from `[width, taps]`.
pub(in crate::kernels::cuda) struct ConvRaw<'a> {
    pub(in crate::kernels::cuda) input: RawRange<'a>,
    pub(in crate::kernels::cuda) kernel: RawRange<'a>,
    pub(in crate::kernels::cuda) history: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    /// The current row of a depth stream, which takes `state[c] += out[c]`, or absent.
    pub(in crate::kernels::cuda) state: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) health: RawRange<'a>,
}

fn shape(width: usize, taps: usize) -> bool {
    (1..=MAX_WIDTH).contains(&width) && (2..=MAX_TAPS).contains(&taps)
}

fn sized(range: &RawRange<'_>, bytes: usize) -> bool {
    range.bytes >= bytes && range.offset.is_multiple_of(4)
}

/// `out` is exactly `input` (the same allocation at the same offset, hence the same bytes) or
/// shares no byte with it.
fn out_is_input_or_apart(out: &Span<'_>, input: &Span<'_>) -> bool {
    (std::ptr::eq(out.device, input.device) && out.start == input.start) || out.apart(input)
}

/// `out[c] = (acc + input[c]) * (1 / sqrt(taps + 1))` with `acc` the host's ordered tap sum of
/// `input[c]` and the `taps - 1` history rows (the host causal convolution); with a `state`, then
/// `state[c] += out[c]`.
pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    call: &ConvRaw<'_>,
    [width, taps]: [usize; 2],
) -> Result<(), DeviceFailure> {
    let row = width * size_of::<f32>();
    let kernel = taps * row;
    let history = (taps - 1) * row;
    let health = size_of::<f32>();
    if !shape(width, taps)
        || !sized(&call.input, row)
        || !sized(&call.kernel, kernel)
        || !sized(&call.history, history)
        || !sized(&call.out, row)
        || !sized(&call.health, health)
        || call.state.is_some_and(|state| !sized(&state, row))
    {
        return Err(invalid_argument());
    }
    let (input, kernel, history, out, health) = (
        Span::of(&call.input, row),
        Span::of(&call.kernel, kernel),
        Span::of(&call.history, history),
        Span::of(&call.out, row),
        Span::of(&call.health, health),
    );
    let apart = out_is_input_or_apart(&out, &input)
        && input.apart(&kernel)
        && input.apart(&history)
        && input.apart(&health)
        && out.apart(&kernel)
        && out.apart(&history)
        && out.apart(&health)
        && kernel.apart(&history)
        && kernel.apart(&health)
        && history.apart(&health);
    if let Some(state) = &call.state {
        let state = Span::of(state, row);
        if !input.apart(&state)
            || !out.apart(&state)
            || !kernel.apart(&state)
            || !history.apart(&state)
            || !health.apart(&state)
        {
            return Err(invalid_argument());
        }
    }
    if !apart {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte except that
    // `out` may be exactly `input`; the entry repeats the checks on the addresses.
    let status = unsafe {
        inference_cuda_short_conv(
            call.input.address() as *const c_void,
            call.kernel.address() as *const c_void,
            call.history.address(),
            call.out.address(),
            call.state
                .map_or(std::ptr::null_mut(), |state| state.address()),
            call.health.address(),
            width as c_int,
            taps as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
