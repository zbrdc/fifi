//! Raw FFI to the KV8 append entries of `native/resident/kv8_append.cu` (declared in
//! `native/resident/kv8_append.h`). Child module of `native/resident.rs`, so it inherits
//! the `allow(unsafe_code)` of `native.rs`.
//!
//! The layout follows Strata's k34-kv8 `laguna_kv8_view.hpp` (int8 variant only).
//!
//! Safety invariants this file relies on:
//! - The launch entries are called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard. They only enqueue.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here, on the extents the kernel touches,
//!   before the call; the C entries repeat the alignment and shape checks on the addresses.
//! - The C argument structs are built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declarations, the `repr(C)` mirrors and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::{align_of, offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_kv8_append_args` in `kv8_append.h`, field for field.
#[repr(C)]
struct Kv8AppendArgsRaw {
    k: *const c_void,
    v: *const c_void,
    k_codes: *mut c_void,
    k_scales: *mut c_void,
    v_codes: *mut c_void,
    v_scales: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    m: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    code_scale: c_int,
}

/// Mirror of `inference_cuda_kv8_chunk_args` in `kv8_append.h`, field for field.
#[repr(C)]
struct Kv8ChunkArgsRaw {
    k: *const c_void,
    v: *const c_void,
    k_codes: *mut c_void,
    k_scales: *mut c_void,
    v_codes: *mut c_void,
    v_scales: *mut c_void,
    pos0: c_int,
    t: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    code_scale: c_int,
}

const _: () = assert!(
    size_of::<Kv8AppendArgsRaw>() == 88
        && align_of::<Kv8AppendArgsRaw>() == 8
        && offset_of!(Kv8AppendArgsRaw, code_scale) == 80
        && offset_of!(Kv8AppendArgsRaw, k_codes) == 16
        && offset_of!(Kv8AppendArgsRaw, pos) == 48
        && offset_of!(Kv8AppendArgsRaw, status) == 56
        && offset_of!(Kv8AppendArgsRaw, m) == 64
);

const _: () = assert!(
    size_of::<Kv8ChunkArgsRaw>() == 72
        && align_of::<Kv8ChunkArgsRaw>() == 8
        && offset_of!(Kv8ChunkArgsRaw, code_scale) == 68
        && offset_of!(Kv8ChunkArgsRaw, k_codes) == 16
        && offset_of!(Kv8ChunkArgsRaw, pos0) == 48
        && offset_of!(Kv8ChunkArgsRaw, t) == 52
);

unsafe extern "C" {
    #[link_name = "inference_cuda_kv8_append_launch"]
    fn kv8_append_launch_raw(
        args: *const Kv8AppendArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_kv8_chunk_launch"]
    fn kv8_chunk_launch_raw(
        args: *const Kv8ChunkArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launches, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct Kv8Ints {
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
    pub(in crate::kernels::cuda) code_scale: c_int,
}

/// The ranges every entry shares: the f32 k and v rows in, the four planes out.
pub(in crate::kernels::cuda) struct Kv8DataRaw<'a> {
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) v: RawRange<'a>,
    pub(in crate::kernels::cuda) k_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) k_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) v_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) v_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: Kv8Ints,
}

/// The decode entry's device words: the position of token 0 (read) and the status word (mask 4
/// is OR-ed in on a refused launch).
pub(in crate::kernels::cuda) struct Kv8WordsRaw<'a> {
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
}

/// The bytes each range must hold and the alignment of every start (4: the kernel reads f32
/// words and 4-byte packets of int8 codes, and the fp16 scales start at a multiple of 64 bytes).
struct Needs {
    rows: usize,
    codes: usize,
    scales: usize,
}

/// `n` as `usize`, or an invalid-argument failure for a negative value.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// Product of the factors, or an invalid-argument failure on overflow.
fn product(factors: &[usize]) -> Result<usize, DeviceFailure> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(invalid_argument)
}

fn needs(ints: &Kv8Ints) -> Result<Needs, DeviceFailure> {
    let (rows, kv_heads, dim) = (
        size(ints.rows)?,
        size(ints.n_kv_head)?,
        size(ints.head_dim)?,
    );
    let cells = size(ints.cells)?;
    let values = product(&[rows, kv_heads, dim])?;
    Ok(Needs {
        rows: product(&[values, 4])?,
        codes: product(&[kv_heads, cells, dim])?,
        scales: product(&[kv_heads, cells, dim / 32, 2])?,
    })
}

/// True when `range` holds `need` bytes and starts at a multiple of 4.
fn fits(range: &RawRange<'_>, need: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(4)
}

/// Proves the six data ranges of both entries: sizes, alignment, and that the four written
/// ranges (both code arrays, both scale arrays) share no byte with each other or with the two
/// read ranges.
fn check_data(io: &Kv8DataRaw<'_>) -> Result<(), DeviceFailure> {
    if !matches!(io.ints.code_scale, 1 | 2) {
        return Err(invalid_argument());
    }
    let need = needs(&io.ints)?;
    if !(fits(&io.k, need.rows)
        && fits(&io.v, need.rows)
        && fits(&io.k_codes, need.codes)
        && fits(&io.k_scales, need.scales)
        && fits(&io.v_codes, need.codes)
        && fits(&io.v_scales, need.scales))
    {
        return Err(invalid_argument());
    }
    let written = [
        Span::of(&io.k_codes, need.codes),
        Span::of(&io.k_scales, need.scales),
        Span::of(&io.v_codes, need.codes),
        Span::of(&io.v_scales, need.scales),
    ];
    let read = [Span::of(&io.k, need.rows), Span::of(&io.v, need.rows)];
    for (i, a) in written.iter().enumerate() {
        let apart_from_written = written[i + 1..].iter().all(|b| a.apart(b));
        if !(apart_from_written && read.iter().all(|b| a.apart(b))) {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Proves the two words of the decode entry: 4 bytes each, 4-aligned, and apart from every range
/// the kernel writes.
fn check_words(io: &Kv8DataRaw<'_>, words: &Kv8WordsRaw<'_>) -> Result<(), DeviceFailure> {
    if !(fits(&words.pos, 4) && fits(&words.status, 4)) {
        return Err(invalid_argument());
    }
    let need = needs(&io.ints)?;
    let status = Span::of(&words.status, 4);
    let apart = [
        Span::of(&io.k_codes, need.codes),
        Span::of(&io.k_scales, need.scales),
        Span::of(&io.v_codes, need.codes),
        Span::of(&io.v_scales, need.scales),
        Span::of(&words.pos, 4),
    ]
    .iter()
    .all(|a| status.apart(a));
    if !apart {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the decode append (m rows at the device position word) on `stream`. Returns when it
/// is queued.
pub(in crate::kernels::cuda) fn kv8_append_launch(
    stream: &RawStream,
    io: Kv8DataRaw<'_>,
    words: Kv8WordsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_data(&io)?;
    check_words(&io, &words)?;
    let ints = io.ints;
    let args = Kv8AppendArgsRaw {
        k: io.k.address().cast_const(),
        v: io.v.address().cast_const(),
        k_codes: io.k_codes.address(),
        k_scales: io.k_scales.address(),
        v_codes: io.v_codes.address(),
        v_scales: io.v_scales.address(),
        pos: words.pos.address().cast_const(),
        status: words.status.address(),
        m: ints.rows,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cells: ints.cells,
        code_scale: ints.code_scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) `check_data` and `check_words` proved every range inside its live allocation,
    // large enough, 4-aligned, and the four written planes and `status` disjoint from every
    // other range, so each pointer in `args` addresses bytes the kernel may touch; the C entry
    // repeats the checks on the addresses. (b) The allocations outlive the queued work because
    // `inference_cuda_free` synchronizes the device before `cudaFree`. (c) `stream` is borrowed
    // from a live `RawStream`. (d) `args` is a live local that the entry reads only during the
    // call. (e) The dispatch guard is held by `launch_on`, which is the only caller.
    let status = unsafe { kv8_append_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the chunk append (t rows at the host base position `pos0`) on `stream`. Returns when
/// it is queued.
pub(in crate::kernels::cuda) fn kv8_chunk_launch(
    stream: &RawStream,
    io: Kv8DataRaw<'_>,
    pos0: c_int,
) -> Result<(), DeviceFailure> {
    check_data(&io)?;
    let ints = io.ints;
    let args = Kv8ChunkArgsRaw {
        k: io.k.address().cast_const(),
        v: io.v.address().cast_const(),
        k_codes: io.k_codes.address(),
        k_scales: io.k_scales.address(),
        v_codes: io.v_codes.address(),
        v_scales: io.v_scales.address(),
        pos0,
        t: ints.rows,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cells: ints.cells,
        code_scale: ints.code_scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `kv8_append_launch`; `check_data` proved the six data ranges, and the C entry
    // re-checks the addresses plus `pos0 + t <= cells` before any launch.
    let status = unsafe { kv8_chunk_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::{Kv8AppendArgsRaw, Kv8ChunkArgsRaw};
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn kv8_argument_layouts_pin_the_required_code_scale() {
        assert_eq!(size_of::<Kv8AppendArgsRaw>(), 88);
        assert_eq!(offset_of!(Kv8AppendArgsRaw, code_scale), 80);
        assert_eq!(align_of::<Kv8AppendArgsRaw>(), 8);
        assert_eq!(size_of::<Kv8ChunkArgsRaw>(), 72);
        assert_eq!(offset_of!(Kv8ChunkArgsRaw, code_scale), 68);
        assert_eq!(align_of::<Kv8ChunkArgsRaw>(), 8);
    }
}
