//! Launch-only FFI to the `inference_cuda_inkling_pass_*` entries of
//! `native/resident/inkling_pass.cu` (contract in `inkling_pass.h`). Child module of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on:
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Shape, sizes, alignment and disjointness are proved again here (`super::Span` for the
//!   overlaps), so a bad call is refused with `invalid_argument` and never reaches the entry.
//! - The entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard; it queues one launch and never synchronizes.
//! - Every allocation stays valid until queued work finishes, because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The extern declarations and every raw pointer stay inside this module; no unsafe function
//!   leaves it.

use std::ffi::{c_int, c_void};
use std::mem::size_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest `blocks` of one stream state.
pub(crate) const MAX_BLOCKS: usize = 8;
pub(crate) const TAPS: usize = 4;
pub(crate) const HEAD: usize = 128;
const MAX_WIDTH: usize = 8192;
const MAX_STRIDE: usize = 65536;
const WORD: usize = size_of::<f32>();

unsafe extern "C" {
    fn inference_cuda_inkling_pass_mix(
        sealed: *const c_void,
        current: *const c_void,
        query: *const c_void,
        out: *mut c_void,
        health: *mut c_void,
        rows: c_int,
        blocks: c_int,
        first: c_int,
        width: c_int,
        out_stride: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_seal(
        sealed: *mut c_void,
        current: *mut c_void,
        next: *const c_void,
        health: *mut c_void,
        rows: c_int,
        blocks: c_int,
        width: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_add(
        current: *mut c_void,
        row: *const c_void,
        health: *mut c_void,
        rows: c_int,
        width: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_prep(
        ranges: *const *mut c_void,
        rows: c_int,
        raw_stride: c_int,
        side_stride: c_int,
        key_at: c_int,
        value_at: c_int,
        dynamic_stride: c_int,
        kv_heads: c_int,
        qk_limit: f32,
        eps: f32,
        dynamic_taps: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_inkling_pass_gate(
        input: *const c_void,
        message: *const c_void,
        gate: *mut c_void,
        health: *mut c_void,
        rows: c_int,
        width: c_int,
        input_stride: c_int,
        message_stride: c_int,
        gate_stride: c_int,
        eps: f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_apply(
        input: *const c_void,
        message: *const c_void,
        gain: *const c_void,
        forget: *const c_void,
        gamma: *const c_void,
        out: *mut c_void,
        health: *mut c_void,
        rows: c_int,
        width: c_int,
        stride: c_int,
        eps: f32,
        control: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_tails(
        source: *const c_void,
        rows: c_int,
        stride: c_int,
        width: c_int,
        past: c_int,
        old: *const c_void,
        out: *mut c_void,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_positions(
        side: *mut c_void,
        rows: c_int,
        stride: c_int,
        pos_at: c_int,
        first: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_inkling_pass_zero(
        words: *mut c_void,
        count: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// One range, aligned to `f32` and holding at least `words` words.
fn sized(range: &RawRange<'_>, words: usize) -> bool {
    range.bytes >= words * WORD && range.offset.is_multiple_of(WORD)
}

/// One range of exactly `bytes` bytes, apart from every earlier range.
fn exact<const N: usize>(
    ranges: &[RawRange<'_>; N],
    index: usize,
    bytes: usize,
    status_words: bool,
) -> bool {
    let range = &ranges[index];
    let aligned = if status_words {
        range.offset.is_multiple_of(size_of::<u32>())
    } else {
        range.offset.is_multiple_of(WORD)
    };
    range.bytes == bytes
        && aligned
        && ranges[..index]
            .iter()
            .all(|other| Span::of(range, bytes).apart(&Span::of(other, other.bytes)))
}

fn shape(blocks: usize, width: usize, eps: f32) -> bool {
    (1..=MAX_BLOCKS).contains(&blocks)
        && (1..=MAX_WIDTH).contains(&width)
        && eps.is_finite()
        && eps > 0.0
}

/// `Stream::mix(first, query)` of every row; `state` is `[sealed (blocks rows per prompt row),
/// current]` and `out` row `r` starts at `r * out_stride` words.
pub(in crate::kernels::cuda) fn mix(
    stream: &RawStream,
    [sealed, current, query, out, health]: [RawRange<'_>; 5],
    [rows, blocks, first, width, out_stride]: [usize; 5],
    eps: f32,
) -> Result<(), DeviceFailure> {
    let row = width * WORD;
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && first <= blocks
        && blocks < MAX_BLOCKS
        && (width..=MAX_STRIDE).contains(&out_stride)
        && shape(1, width, eps)
        && sized(&sealed, blocks * rows * width)
        && sized(&current, rows * width)
        && sized(&query, width)
        && sized(&out, (rows - 1) * out_stride + width)
        && health.bytes == rows * WORD
        && health.offset.is_multiple_of(WORD);
    if !valid {
        return Err(invalid_argument());
    }
    let spans = [
        Span::of(&sealed, blocks * rows * row),
        Span::of(&current, rows * row),
        Span::of(&query, row),
        Span::of(&out, ((rows - 1) * out_stride + width) * WORD),
        Span::of(&health, rows * WORD),
    ];
    if (0..spans.len()).any(|i| (0..i).any(|j| !spans[i].apart(&spans[j]))) {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_pass_mix(
            sealed.address() as *const c_void,
            current.address() as *const c_void,
            query.address() as *const c_void,
            out.address(),
            health.address(),
            rows as c_int,
            blocks as c_int,
            first as c_int,
            width as c_int,
            out_stride as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `Stream::seal(next)` of every row; `state` must hold the new sealed row: `blocks + 1`
/// sealed rows above `current`.
pub(in crate::kernels::cuda) fn seal(
    stream: &RawStream,
    [sealed, current, next, health]: [RawRange<'_>; 4],
    [rows, blocks, width]: [usize; 3],
    eps: f32,
) -> Result<(), DeviceFailure> {
    let row = width * WORD;
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && blocks < MAX_BLOCKS
        && shape(1, width, eps)
        && sized(&sealed, (blocks + 1) * rows * width)
        && sized(&current, rows * width)
        && next.bytes == rows * row
        && next.offset.is_multiple_of(WORD)
        && health.bytes == rows * WORD
        && health.offset.is_multiple_of(WORD);
    if !valid {
        return Err(invalid_argument());
    }
    let spans = [
        Span::of(&sealed, (blocks + 1) * rows * row),
        Span::of(&current, rows * row),
        Span::of(&next, rows * row),
        Span::of(&health, rows * WORD),
    ];
    if (0..spans.len()).any(|i| (0..i).any(|j| !spans[i].apart(&spans[j]))) {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_pass_seal(
            sealed.address(),
            current.address(),
            next.address() as *const c_void,
            health.address(),
            rows as c_int,
            blocks as c_int,
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `Stream::add_current(row)` of every row.
pub(in crate::kernels::cuda) fn add(
    stream: &RawStream,
    [current, row, health]: [RawRange<'_>; 3],
    [rows, width]: [usize; 2],
) -> Result<(), DeviceFailure> {
    let bytes = rows * width * WORD;
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && (1..=MAX_WIDTH).contains(&width)
        && current.bytes == bytes
        && row.bytes == bytes
        && health.bytes == rows * WORD
        && [current.offset, row.offset, health.offset]
            .into_iter()
            .all(|offset| offset.is_multiple_of(WORD));
    if !valid {
        return Err(invalid_argument());
    }
    let spans = [
        Span::of(&current, bytes),
        Span::of(&row, bytes),
        Span::of(&health, rows * WORD),
    ];
    if (0..spans.len()).any(|i| (0..i).any(|j| !spans[i].apart(&spans[j]))) {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and share no byte.
    let status = unsafe {
        inference_cuda_inkling_pass_add(
            current.address(),
            row.address() as *const c_void,
            health.address(),
            rows as c_int,
            width as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// K/V preparation of every row. `ranges` is `[raw, side, history_k, history_v, kernel_k,
/// kernel_v, dynamic, status]`; `dynamic` is any non-empty range when `dynamic` is false.
#[allow(clippy::too_many_arguments)]
pub(in crate::kernels::cuda) fn prep(
    stream: &RawStream,
    ranges: [RawRange<'_>; 8],
    [
        rows,
        raw_stride,
        side_stride,
        key_at,
        value_at,
        dynamic_stride,
        kv_heads,
    ]: [usize; 7],
    [qk_limit, eps]: [f32; 2],
    dynamic: bool,
) -> Result<(), DeviceFailure> {
    let channels = kv_heads * HEAD;
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && (1..=MAX_WIDTH / HEAD).contains(&kv_heads)
        && (2 * channels..=MAX_STRIDE).contains(&raw_stride)
        && (channels..=MAX_STRIDE).contains(&side_stride)
        && key_at + channels <= side_stride
        && value_at + channels <= side_stride
        && (!dynamic || dynamic_stride >= kv_heads * TAPS)
        && [qk_limit, eps].iter().all(|&v| v.is_finite() && v > 0.0)
        && exact(
            &ranges,
            0,
            ((rows - 1) * raw_stride + 2 * channels) * WORD,
            false,
        )
        && exact(
            &ranges,
            1,
            ((rows - 1) * side_stride + key_at.max(value_at) + channels) * WORD,
            false,
        )
        && exact(&ranges, 2, (TAPS - 1) * channels * WORD, false)
        && exact(&ranges, 3, (TAPS - 1) * channels * WORD, false)
        && exact(&ranges, 4, channels * TAPS * WORD, false)
        && exact(&ranges, 5, channels * TAPS * WORD, false)
        && (!dynamic
            || exact(
                &ranges,
                6,
                // The last row carries the K taps and the V taps of its dynamic convolution.
                ((rows - 1) * dynamic_stride + 2 * kv_heads * TAPS) * WORD,
                false,
            ))
        && exact(&ranges, 7, rows * size_of::<u32>(), true);
    if !valid {
        return Err(invalid_argument());
    }
    let pointers = ranges.map(|range| range.address());
    let mut stage = STAGE_NONE;
    // SAFETY: the pointer list is consumed by the entry; the disjoint, aligned extents above
    // cover every device access of the launch.
    let status = unsafe {
        inference_cuda_inkling_pass_prep(
            pointers.as_ptr(),
            rows as c_int,
            raw_stride as c_int,
            side_stride as c_int,
            key_at as c_int,
            value_at as c_int,
            dynamic_stride as c_int,
            kv_heads as c_int,
            qk_limit,
            eps,
            dynamic.into(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// One `f32` range of at least `words` words; a u32 range when `status_words` holds.
fn plane(range: &RawRange<'_>, words: usize, status_words: bool) -> bool {
    range.bytes >= words * if status_words { size_of::<u32>() } else { WORD }
        && range.offset.is_multiple_of(WORD)
}

/// The writers of a launch must not cover any other range it uses.
fn writers_apart(ranges: &[RawRange<'_>], writers: &[usize]) -> bool {
    writers.iter().all(|&w| {
        (0..ranges.len())
            .filter(|&o| o != w && !writers.contains(&o))
            .all(|o| {
                Span::of(&ranges[w], ranges[w].bytes).apart(&Span::of(&ranges[o], ranges[o].bytes))
            })
    })
}

/// `control_input_with_scales` of every row: `gate[r] = [norm(in[r]) | norm(msg[r])]`.
pub(in crate::kernels::cuda) fn gate(
    stream: &RawStream,
    [input, message, gate, health]: [RawRange<'_>; 4],
    [rows, width, input_stride, message_stride, gate_stride]: [usize; 5],
    eps: f32,
) -> Result<(), DeviceFailure> {
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && (1..=MAX_WIDTH / 2).contains(&width)
        && (width..=MAX_STRIDE).contains(&input_stride)
        && (width..=MAX_STRIDE).contains(&message_stride)
        && (2 * width..=MAX_STRIDE).contains(&gate_stride)
        && eps.is_finite()
        && eps > 0.0
        && plane(&input, (rows - 1) * input_stride + width, false)
        && plane(&message, (rows - 1) * message_stride + width, false)
        && plane(&gate, (rows - 1) * gate_stride + 2 * width, false)
        && health.bytes == rows * WORD
        && health.offset.is_multiple_of(WORD);
    if !valid || !writers_apart(&[input, message, gate, health], &[2, 3]) {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and the writers share no byte
    // with any reader.
    let status = unsafe {
        inference_cuda_inkling_pass_gate(
            input.address() as *const c_void,
            message.address() as *const c_void,
            gate.address(),
            health.address(),
            rows as c_int,
            width as c_int,
            input_stride as c_int,
            message_stride as c_int,
            gate_stride as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `apply_control` then `norm` of every row; `out` may equal `input`.
#[allow(clippy::too_many_arguments)]
pub(in crate::kernels::cuda) fn apply(
    stream: &RawStream,
    [input, message, gain, forget, gamma, out, health]: [RawRange<'_>; 7],
    [rows, width, stride]: [usize; 3],
    eps: f32,
    control: bool,
) -> Result<(), DeviceFailure> {
    let row = (rows - 1) * stride + width;
    let same = |a: &RawRange<'_>, b: &RawRange<'_>| a.offset == b.offset;
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && (1..=MAX_WIDTH).contains(&width)
        && (width..=MAX_STRIDE).contains(&stride)
        && eps.is_finite()
        && eps > 0.0
        && plane(&input, row, false)
        && plane(&out, row, false)
        && plane(&gamma, width, false)
        && health.bytes == rows * WORD
        && health.offset.is_multiple_of(WORD)
        && (!control
            || (plane(&message, row, false)
                && plane(&gain, row, false)
                && plane(&forget, row, false)));
    if !valid {
        return Err(invalid_argument());
    }
    let ranges = [input, message, gain, forget, gamma, out, health];
    // The output covers the input only when they are the same range.
    let writers = [5, 6];
    let apart = writers.iter().all(|&w| {
        (0..ranges.len())
            .filter(|&o| o != w && !writers.contains(&o) && !same(&ranges[w], &ranges[o]))
            .all(|o| {
                Span::of(&ranges[w], ranges[w].bytes).apart(&Span::of(&ranges[o], ranges[o].bytes))
            })
    });
    if !apart {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and the writers share no byte
    // with any reader they does not replace.
    let status = unsafe {
        inference_cuda_inkling_pass_apply(
            input.address() as *const c_void,
            message.address() as *const c_void,
            gain.address() as *const c_void,
            forget.address() as *const c_void,
            gamma.address() as *const c_void,
            out.address(),
            health.address(),
            rows as c_int,
            width as c_int,
            stride as c_int,
            eps,
            control.into(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// The last three rows of `old` (its `past` valid rows, oldest first) then `source`, into `out`.
pub(in crate::kernels::cuda) fn tails(
    stream: &RawStream,
    [source, old, out]: [RawRange<'_>; 3],
    [rows, stride, width, past]: [usize; 4],
) -> Result<(), DeviceFailure> {
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && (1..=MAX_WIDTH).contains(&width)
        && (width..=MAX_STRIDE).contains(&stride)
        && past < TAPS
        && plane(&source, (rows - 1) * stride + width, false)
        && plane(&old, (TAPS - 1) * width, false)
        && out.bytes == (TAPS - 1) * width * WORD
        && out.offset.is_multiple_of(WORD);
    if !valid || !writers_apart(&[source, old, out], &[2]) {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are aligned, hold the extents above and the writer shares no byte.
    let status = unsafe {
        inference_cuda_inkling_pass_tails(
            source.address() as *const c_void,
            rows as c_int,
            stride as c_int,
            width as c_int,
            past as c_int,
            old.address() as *const c_void,
            out.address(),
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// `side[r * stride + pos_at] = u32(first + r)` for every row of one sidecar.
pub(in crate::kernels::cuda) fn positions(
    stream: &RawStream,
    side: RawRange<'_>,
    [rows, stride, pos_at, first]: [usize; 4],
) -> Result<(), DeviceFailure> {
    let valid = (1..=c_int::MAX as usize).contains(&rows)
        && (1..=MAX_STRIDE).contains(&stride)
        && pos_at < stride
        && first + rows <= i32::MAX as usize
        && plane(&side, (rows - 1) * stride + 1, true);
    if !valid {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the range is aligned and holds every word the launch writes.
    let status = unsafe {
        inference_cuda_inkling_pass_positions(
            side.address(),
            rows as c_int,
            stride as c_int,
            pos_at as c_int,
            first as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Sets `count` status words to zero.
pub(in crate::kernels::cuda) fn zero(
    stream: &RawStream,
    words: RawRange<'_>,
    count: usize,
) -> Result<(), DeviceFailure> {
    if (1..=1 << 20).contains(&count) && plane(&words, count, true) {
        let mut stage = STAGE_NONE;
        // SAFETY: the range is aligned and holds every word the launch writes.
        let status = unsafe {
            inference_cuda_inkling_pass_zero(
                words.address(),
                count as c_int,
                stream.0.as_ptr(),
                &mut stage,
            )
        };
        checked(status, stage)
    } else {
        Err(invalid_argument())
    }
}
