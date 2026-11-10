//! EXL3 stream-only C ABI. RawRange proves live allocation bounds; this layer
//! repeats geometry, extent, alignment and overlap checks before C repeats them.
//! Dispatch/stream ownership is enforced by CudaContext; frees drain device use.
//! No raw pointer or unsafe function leaves this module. Faults surface at sync.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "EXL3 M1, family routing stays off")
)]
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::weights::exl3::Codebook;
use std::ffi::{c_int, c_void};
use std::mem::offset_of;

#[repr(C)]
#[derive(Clone, Copy)]
struct PrepArgs {
    trellis: *const u16,
    trellis_bytes: usize,
    out: *mut u16,
    out_bytes: usize,
    k: usize,
    n: usize,
    bits_twice: c_int,
    book: c_int,
}
const _: () = assert!(
    size_of::<PrepArgs>() == 56
        && offset_of!(PrepArgs, trellis) == 0
        && offset_of!(PrepArgs, trellis_bytes) == 8
        && offset_of!(PrepArgs, out) == 16
        && offset_of!(PrepArgs, out_bytes) == 24
        && offset_of!(PrepArgs, k) == 32
        && offset_of!(PrepArgs, n) == 40
        && offset_of!(PrepArgs, bits_twice) == 48
        && offset_of!(PrepArgs, book) == 52
);
unsafe extern "C" {
    fn inference_cuda_exl3_prep(
        args: *const PrepArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn geometry(
    k: usize,
    n: usize,
    bits: u8,
    book: Codebook,
) -> Option<(usize, usize)> {
    if !(128..=(1 << 22)).contains(&k)
        || !(128..=(1 << 22)).contains(&n)
        || !k.is_multiple_of(128)
        || !n.is_multiple_of(128)
    {
        return None;
    }
    let supported = (2..=16).contains(&bits) && bits.is_multiple_of(2)
        || book == Codebook::Mul1 && matches!(bits, 3 | 5 | 7);
    let count = k.checked_mul(n)?;
    if !supported || count > (1usize << 32) {
        return None;
    }
    Some((count / 256 * usize::from(bits) * 16, count * 2))
}
fn book_tag(book: Codebook) -> c_int {
    match book {
        Codebook::ThreeInst => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    }
}

pub(in crate::kernels::cuda) struct PrepRaw<'a> {
    pub(in crate::kernels::cuda) trellis: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) bits_twice: u8,
    pub(in crate::kernels::cuda) book: Codebook,
}
pub(in crate::kernels::cuda) fn prep(
    stream: &RawStream,
    call: &PrepRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (packed, dense) =
        geometry(call.k, call.n, call.bits_twice, call.book).ok_or_else(invalid_argument)?;
    if call.trellis.bytes() < packed
        || call.out.bytes() < dense
        || !call.trellis.offset().is_multiple_of(2)
        || !call.out.offset().is_multiple_of(2)
        || !Span::of(&call.trellis, packed).apart(&Span::of(&call.out, dense))
    {
        return Err(invalid_argument());
    }
    let args = PrepArgs {
        trellis: call.trellis.address().cast(),
        trellis_bytes: call.trellis.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        k: call.k,
        n: call.n,
        bits_twice: c_int::from(call.bits_twice),
        book: book_tag(call.book),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live bounded ranges, complete extents, alignment and disjointness proved above;
    // C repeats these checks. Explicit live stream is under the dispatch guard.
    let status = unsafe { inference_cuda_exl3_prep(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Working byte extents in ABI order: trellis, suh, svh, bias, x, xh, z, out, flag.
pub(in crate::kernels::cuda) fn linear_extents(
    rows: usize,
    k: usize,
    n: usize,
    bits: u8,
    book: Codebook,
    split_k: u32,
) -> Option<[usize; 9]> {
    let (packed, _) = geometry(k, n, bits, book)?;
    if !(1..=65536).contains(&rows) || !matches!(split_k, 2 | 4 | 8) {
        return None;
    }
    let input = rows.checked_mul(k)?;
    let output = rows.checked_mul(n)?;
    if input > (1usize << 32) || output > (1usize << 32) {
        return None;
    }
    let banks = if rows <= 8 { split_k as usize - 1 } else { 1 };
    let z = output.checked_mul(4)?.checked_mul(banks)?;
    Some([
        packed,
        k * 4,
        n * 4,
        n * 4,
        input * 4,
        input * 4,
        z,
        output * 4,
        4,
    ])
}

pub(in crate::kernels::cuda) fn midrows_extents(
    rows: usize,
    k: usize,
    n: usize,
    bits: u8,
    book: Codebook,
) -> Option<[usize; 9]> {
    if !(9..=144).contains(&rows) {
        return None;
    }
    linear_extents(rows, k, n, bits, book, 2)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LinearArgs {
    trellis: *const u16,
    trellis_bytes: usize,
    suh: *const f32,
    suh_bytes: usize,
    svh: *const f32,
    svh_bytes: usize,
    bias: *const f32,
    bias_bytes: usize,
    x: *const f32,
    x_bytes: usize,
    xh: *mut f32,
    xh_bytes: usize,
    z: *mut f32,
    z_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    k: usize,
    n: usize,
    bits_twice: c_int,
    book: c_int,
    fused_publish: u32,
}
const _: () = assert!(
    size_of::<LinearArgs>() == 184
        && offset_of!(LinearArgs, trellis) == 0
        && offset_of!(LinearArgs, trellis_bytes) == 8
        && offset_of!(LinearArgs, suh) == 16
        && offset_of!(LinearArgs, suh_bytes) == 24
        && offset_of!(LinearArgs, svh) == 32
        && offset_of!(LinearArgs, svh_bytes) == 40
        && offset_of!(LinearArgs, bias) == 48
        && offset_of!(LinearArgs, bias_bytes) == 56
        && offset_of!(LinearArgs, x) == 64
        && offset_of!(LinearArgs, x_bytes) == 72
        && offset_of!(LinearArgs, xh) == 80
        && offset_of!(LinearArgs, xh_bytes) == 88
        && offset_of!(LinearArgs, z) == 96
        && offset_of!(LinearArgs, z_bytes) == 104
        && offset_of!(LinearArgs, out) == 112
        && offset_of!(LinearArgs, out_bytes) == 120
        && offset_of!(LinearArgs, flag) == 128
        && offset_of!(LinearArgs, flag_bytes) == 136
        && offset_of!(LinearArgs, rows) == 144
        && offset_of!(LinearArgs, k) == 152
        && offset_of!(LinearArgs, n) == 160
        && offset_of!(LinearArgs, bits_twice) == 168
        && offset_of!(LinearArgs, book) == 172
        && offset_of!(LinearArgs, fused_publish) == 176
);
unsafe extern "C" {
    #[cfg(test)]
    fn inference_cuda_exl3_linear(
        args: *const LinearArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_linear_midrows(
        args: *const LinearArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_linear_midrows_hilo(
        args: *const LinearArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_linear_split(
        args: *const LinearArgs,
        split_k: u32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct LinearRaw<'a> {
    pub(in crate::kernels::cuda) trellis: RawRange<'a>,
    pub(in crate::kernels::cuda) suh: RawRange<'a>,
    pub(in crate::kernels::cuda) svh: RawRange<'a>,
    pub(in crate::kernels::cuda) bias: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) xh: RawRange<'a>,
    pub(in crate::kernels::cuda) z: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) split_k: u32,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) bits_twice: u8,
    pub(in crate::kernels::cuda) book: Codebook,
    /// Producer-fusion arm: round at output_rotate/grouped_output_rotate.
    pub(in crate::kernels::cuda) fused: bool,
}
pub(in crate::kernels::cuda) fn linear(
    stream: &RawStream,
    call: &LinearRaw<'_>,
) -> Result<(), DeviceFailure> {
    linear_mode(stream, call, false)
}

pub(in crate::kernels::cuda) fn linear_midrows(
    stream: &RawStream,
    call: &LinearRaw<'_>,
) -> Result<(), DeviceFailure> {
    linear_mode(stream, call, true)
}

/// [`linear_midrows`] with the activations staged as an F16 high part and a scaled F16 residual
/// (`FIFI_EXL3_TALL_HILO`): same arguments, extents and launch grid, one more entry in the library.
pub(in crate::kernels::cuda) fn linear_midrows_hilo(
    stream: &RawStream,
    call: &LinearRaw<'_>,
) -> Result<(), DeviceFailure> {
    let args = linear_args(call, true)?;
    let mut stage = STAGE_NONE;
    // The hi/lo entry launches the same three emissions as the rows entry, so a recording test sees the same schedule.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    if launch_observer::recording() {
        return checked(
            launch_observer::linear(&args, call.split_k, true, stream, &mut stage),
            stage,
        );
    }
    // SAFETY: as in `linear_mode`: extents, alignment and disjoint writes are checked above and
    // repeated in C; the explicit live stream is protected by the dispatch guard.
    let status =
        unsafe { inference_cuda_exl3_linear_midrows_hilo(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

fn linear_args(call: &LinearRaw<'_>, midrows: bool) -> Result<LinearArgs, DeviceFailure> {
    if !matches!(call.split_k, 2 | 4 | 8) {
        return Err(invalid_argument());
    }
    let needs = if midrows {
        midrows_extents(call.rows, call.k, call.n, call.bits_twice, call.book)
    } else {
        linear_extents(
            call.rows,
            call.k,
            call.n,
            call.bits_twice,
            call.book,
            call.split_k,
        )
    }
    .ok_or_else(invalid_argument)?;
    let ranges = [
        Some(&call.trellis),
        Some(&call.suh),
        Some(&call.svh),
        call.bias.as_ref(),
        Some(&call.x),
        Some(&call.xh),
        Some(&call.z),
        Some(&call.out),
        Some(&call.flag),
    ];
    for (i, range) in ranges.iter().enumerate() {
        let Some(range) = range else { continue };
        let alignment = if i == 0 { 2 } else { 4 };
        if range.bytes() < needs[i] || !range.offset().is_multiple_of(alignment) {
            return Err(invalid_argument());
        }
        for (j, other) in ranges[..i].iter().enumerate() {
            if let Some(other) = other
                && (i >= 5 || j >= 5)
                && !Span::of(range, needs[i]).apart(&Span::of(other, needs[j]))
            {
                return Err(invalid_argument());
            }
        }
    }
    let args = LinearArgs {
        trellis: call.trellis.address().cast(),
        trellis_bytes: call.trellis.bytes(),
        suh: call.suh.address().cast(),
        suh_bytes: call.suh.bytes(),
        svh: call.svh.address().cast(),
        svh_bytes: call.svh.bytes(),
        bias: call
            .bias
            .as_ref()
            .map_or(std::ptr::null(), |r| r.address().cast()),
        bias_bytes: call.bias.as_ref().map_or(0, RawRange::bytes),
        x: call.x.address().cast(),
        x_bytes: call.x.bytes(),
        xh: call.xh.address().cast(),
        xh_bytes: call.xh.bytes(),
        z: call.z.address().cast(),
        z_bytes: call.z.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        k: call.k,
        n: call.n,
        bits_twice: c_int::from(call.bits_twice),
        book: book_tag(call.book),
        fused_publish: u32::from(call.fused),
    };
    Ok(args)
}

fn linear_mode(
    stream: &RawStream,
    call: &LinearRaw<'_>,
    midrows: bool,
) -> Result<(), DeviceFailure> {
    let args = linear_args(call, midrows)?;
    let mut stage = STAGE_NONE;
    // SAFETY: all live owners, extents, alignment and disjoint writes checked above,
    // repeated in C; the explicit live stream is protected by the dispatch guard.
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    if launch_observer::recording() {
        return checked(
            launch_observer::linear(&args, call.split_k, midrows, stream, &mut stage),
            stage,
        );
    }
    let status = unsafe {
        if midrows {
            inference_cuda_exl3_linear_midrows(&args, stream.as_ptr(), &mut stage)
        } else {
            inference_cuda_exl3_linear_split(&args, call.split_k, stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn grouped_split(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
) -> Result<(), DeviceFailure> {
    let split = calls.first().map(|call| call.split_k).unwrap_or(0);
    if calls.is_empty()
        || calls.len() > 48
        || !matches!(split, 4 | 8)
        || calls
            .iter()
            .any(|call| call.rows != 1 || call.split_k != split)
    {
        return Err(invalid_argument());
    }
    let mut args = Vec::with_capacity(calls.len());
    for call in calls {
        args.push(linear_args(call, false)?);
    }
    let mut stage = STAGE_NONE;
    // SAFETY: every record has live bounded owners; C validates the group, the widened
    // z strides and cross-group aliases before ANY enqueue, exactly as the split-2 entry.
    let status = unsafe {
        inference_cuda_exl3_grouped_split(
            args.as_ptr(),
            args.len(),
            split,
            stream.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

unsafe extern "C" {
    fn inference_cuda_exl3_grouped(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_split(
        args: *const LinearArgs,
        count: usize,
        split_k: u32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_gathered(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_indexed(
        args: *const LinearArgs,
        index: *const IndexArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
        cols: u32,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_indexed_option(
        args: *const LinearArgs,
        index: *const IndexArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
        options: u32,
    ) -> c_int;
    #[cfg(test)]
    fn inference_cuda_exl3_book2_probe(
        words: *const c_void,
        count: usize,
        sums: *mut c_void,
        decoded: *mut c_void,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_wide(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
        cols: u32,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_wide_odd(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
        dp4a: u32,
    ) -> c_int;
    fn inference_cuda_exl3_grouped_wide_fin(
        args: *const LinearArgs,
        count: usize,
        stream: *mut c_void,
        stage: *mut c_int,
        dp4a: u32,
    ) -> c_int;
}

/// Mirror of `inference_cuda_exl3_index_args`.
#[repr(C)]
#[derive(Clone, Copy)]
struct IndexArgs {
    ids: *const c_void,
    ids_bytes: usize,
    slot: u32,
    members: u32,
    trellis_stride: usize,
    scale_stride: usize,
    offsets: *const c_void,
}
const _: () = assert!(
    size_of::<IndexArgs>() == 48
        && offset_of!(IndexArgs, slot) == 16
        && offset_of!(IndexArgs, trellis_stride) == 24
        && offset_of!(IndexArgs, scale_stride) == 32
        && offset_of!(IndexArgs, offsets) == 40
);

/// The device-side member choice of one call of an indexed launch.
pub(in crate::kernels::cuda) struct IndexRaw<'a> {
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) slot: u32,
    pub(in crate::kernels::cuda) members: u32,
    /// Bytes between the trellises of consecutive members.
    pub(in crate::kernels::cuda) trellis_stride: usize,
    /// Floats between the suh (and the svh) of consecutive members.
    pub(in crate::kernels::cuda) scale_stride: usize,
    /// Optional u32 byte offset per member (0xFFFFFFFF = absent), replacing both strides.
    pub(in crate::kernels::cuda) offsets: Option<RawRange<'a>>,
}

/// The id-indexed grouped launch: `calls[i]` is one row of member 0 of a bank, `index[i]` names
/// the member from device memory when the kernels run. 1 to 16 calls; `cols` as for
/// [`grouped_wide`].
pub(in crate::kernels::cuda) fn grouped_indexed(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
    index: &[IndexRaw<'_>],
    cols: u32,
) -> Result<(), DeviceFailure> {
    if !matches!(cols, 1 | 2 | 4 | 8) {
        return Err(invalid_argument());
    }
    let (args, indexes) = indexed_records(calls, index, 16)?;
    let mut stage = STAGE_NONE;
    // SAFETY: every call record has live bounded owners and validated local extents (member 0 of
    // its bank), every id range is a live allocation of at least `slot + 1` aligned words, and C
    // validates the group shapes, the contiguity of the working spans and their aliasing before
    // any enqueue. The strides, the member counts and the offset tables (a live allocation of at least
    // `members` aligned words) were bounded against the bank extents by the caller
    // (`CudaContext::exl3_grouped_indexed`); a wrong id or an absent member is skipped by the
    // kernels and sets the sticky flag. `cols` is 1, 2, 4 or 8.
    let status = unsafe {
        inference_cuda_exl3_grouped_indexed(
            args.as_ptr(),
            indexes.as_ptr(),
            args.len(),
            stream.as_ptr(),
            &mut stage,
            cols,
        )
    };
    checked(status, stage)
}

/// [`grouped_indexed`] with opt-in options, one tile per block: `options` bit 0 decodes book 2
/// with DP4A, bit 1 admits up to 20 calls (else up to 16). Same validation and bytes.
pub(in crate::kernels::cuda) fn grouped_indexed_option(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
    index: &[IndexRaw<'_>],
    options: u32,
) -> Result<(), DeviceFailure> {
    let limit = match options {
        1 | 4 | 5 => 16,
        2 | 3 | 6 | 7 => 20,
        _ => return Err(invalid_argument()),
    };
    let (args, indexes) = indexed_records(calls, index, limit)?;
    let mut stage = STAGE_NONE;
    // SAFETY: as for `grouped_indexed`; the option kernels read and write exactly what the
    // default kernels do, for up to 20 calls, which C bounds by the parameter block it fills.
    let status = unsafe {
        inference_cuda_exl3_grouped_indexed_option(
            args.as_ptr(),
            indexes.as_ptr(),
            args.len(),
            stream.as_ptr(),
            &mut stage,
            options,
        )
    };
    checked(status, stage)
}

/// The DP4A decode probe: for each of the `words.bytes() / 4` words, two result words in `sums`
/// (the byte sum, the DP4A result); for each of the 65536 states, two halves in `decoded` (the
/// book-2 decode, its DP4A form).
#[cfg(test)]
pub(in crate::kernels::cuda) fn book2_probe(
    stream: &RawStream,
    words: RawRange<'_>,
    sums: RawRange<'_>,
    decoded: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    let count = words.bytes() / 4;
    if count == 0
        || !words.bytes().is_multiple_of(4)
        || sums.bytes() < count * 8
        || decoded.bytes() < 65536 * 4
        || [&words, &sums, &decoded]
            .iter()
            .any(|r| !r.offset().is_multiple_of(4))
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: the three ranges are live, 4-byte aligned and large enough for what the probe
    // kernels read (count words) and write (2 count words, 2 x 65536 halves).
    let status = unsafe {
        inference_cuda_exl3_book2_probe(
            words.address().cast_const(),
            count,
            sums.address(),
            decoded.address(),
            stream.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// The native records of an id-indexed group of 1 to `limit` calls, after the per-call checks C
/// repeats.
type IndexedRecords = (Vec<LinearArgs>, Vec<IndexArgs>);
fn indexed_records(
    calls: &[LinearRaw<'_>],
    index: &[IndexRaw<'_>],
    limit: usize,
) -> Result<IndexedRecords, DeviceFailure> {
    if calls.is_empty() || calls.len() > limit || calls.len() != index.len() {
        return Err(invalid_argument());
    }
    let mut args = Vec::with_capacity(calls.len());
    let mut indexes = Vec::with_capacity(calls.len());
    for (call, at) in calls.iter().zip(index) {
        let ids_need = (at.slot as usize)
            .checked_add(1)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(invalid_argument)?;
        if call.rows != 1
            || call.split_k != 2
            || at.members == 0
            || at.ids.bytes() < ids_need
            || !at.ids.offset().is_multiple_of(4)
        {
            return Err(invalid_argument());
        }
        match &at.offsets {
            Some(table) => {
                let need = (at.members as usize)
                    .checked_mul(4)
                    .ok_or_else(invalid_argument)?;
                if table.bytes() < need || !table.offset().is_multiple_of(4) {
                    return Err(invalid_argument());
                }
            }
            None if at.trellis_stride == 0 || at.scale_stride == 0 => {
                return Err(invalid_argument());
            }
            None => {}
        }
        args.push(linear_args(call, false)?);
        indexes.push(IndexArgs {
            ids: at.ids.address().cast_const(),
            ids_bytes: at.ids.bytes(),
            slot: at.slot,
            members: at.members,
            trellis_stride: at.trellis_stride,
            scale_stride: at.scale_stride,
            offsets: at
                .offsets
                .as_ref()
                .map_or(std::ptr::null(), |table| table.address().cast_const()),
        });
    }
    Ok((args, indexes))
}

/// The grouped launch with the wide projection: each block covers `cols` (1, 2, 4 or 8) adjacent
/// 16-column tiles. Same validation and results as [`grouped`] with `gather`; `n` of every call
/// must be a multiple of `16 * cols`. With `odd` and one tile per block the launch may run the
/// compile-time rate 10 and rate 6 body of book 2 (`Some(dp4a)`: with the DP4A decode when `dp4a`);
/// the bytes written are the same.
pub(in crate::kernels::cuda) fn grouped_wide(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
    cols: u32,
    odd: Option<bool>,
) -> Result<(), DeviceFailure> {
    if calls.is_empty() || calls.len() > 48 || !matches!(cols, 1 | 2 | 4 | 8) {
        return Err(invalid_argument());
    }
    let mut pairs = 0usize;
    let mut args = Vec::with_capacity(calls.len());
    for call in calls {
        pairs = pairs.checked_add(call.rows).ok_or_else(invalid_argument)?;
        if !(1..=8).contains(&call.rows) || call.split_k != 2 || pairs > 48 {
            return Err(invalid_argument());
        }
        args.push(linear_args(call, false)?);
    }
    let mut stage = STAGE_NONE;
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    if let (Some(dp4a), 1) = (odd, cols) {
        // The same launch through the observed entry, so tests see which kernel ran.
        return checked(
            launch_observer::wide_odd(&args, stream, &mut stage, dp4a),
            stage,
        );
    }
    // SAFETY: as `grouped`: every record has live bounded owners and validated local extents, C
    // validates all group shapes, the column multiple and aliases before any enqueue, and `cols`
    // is 1, 2, 4 or 8.
    let status = unsafe {
        if let (Some(dp4a), 1) = (odd, cols) {
            inference_cuda_exl3_grouped_wide_odd(
                args.as_ptr(),
                args.len(),
                stream.as_ptr(),
                &mut stage,
                u32::from(dp4a),
            )
        } else {
            inference_cuda_exl3_grouped_wide(
                args.as_ptr(),
                args.len(),
                stream.as_ptr(),
                &mut stage,
                cols,
            )
        }
    };
    checked(status, stage)
}

/// The wide odd-rate launch with the input rotation fused into the projection kernel
/// (`exl3_decode_fused.h`): two launches per group set instead of three. Requires one row per
/// call, book 2 at rate 10 or 6, and `k` a multiple of 512 (C validates and refuses otherwise);
/// every byte written equals [`grouped_wide`] at `cols` 1 with `odd`.
pub(in crate::kernels::cuda) fn grouped_fin(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
    dp4a: bool,
) -> Result<(), DeviceFailure> {
    if calls.is_empty() || calls.len() > 48 {
        return Err(invalid_argument());
    }
    let mut pairs = 0usize;
    let mut args = Vec::with_capacity(calls.len());
    for call in calls {
        pairs = pairs.checked_add(call.rows).ok_or_else(invalid_argument)?;
        if call.rows != 1 || call.split_k != 2 || pairs > 48 {
            return Err(invalid_argument());
        }
        args.push(linear_args(call, false)?);
    }
    let mut stage = STAGE_NONE;
    // SAFETY: as `grouped_wide`: every record has live bounded owners and validated local
    // extents; C validates the group shapes, the rates and the half rounding before any enqueue.
    let status = unsafe {
        inference_cuda_exl3_grouped_wide_fin(
            args.as_ptr(),
            args.len(),
            stream.as_ptr(),
            &mut stage,
            u32::from(dp4a),
        )
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn grouped(
    stream: &RawStream,
    calls: &[LinearRaw<'_>],
    gather: bool,
) -> Result<(), DeviceFailure> {
    if calls.is_empty() || calls.len() > 48 {
        return Err(invalid_argument());
    }
    let mut pairs = 0usize;
    let mut args = Vec::with_capacity(calls.len());
    for call in calls {
        pairs = pairs.checked_add(call.rows).ok_or_else(invalid_argument)?;
        if !(1..=8).contains(&call.rows) || call.split_k != 2 || pairs > 48 {
            return Err(invalid_argument());
        }
        args.push(linear_args(call, false)?);
    }
    let mut stage = STAGE_NONE;
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    if launch_observer::recording() {
        // Identical launch plan either way; the observed entry records which
        // variant actually ran so tests cannot claim gather without running it.
        return checked(
            launch_observer::grouped(&args, stream, &mut stage, gather),
            stage,
        );
    }
    // SAFETY: every record has live bounded owners and validated local extents.
    // C validates all group shapes, contiguous writes and cross-group aliases before ANY enqueue.
    let status = unsafe {
        if gather {
            inference_cuda_exl3_grouped_gathered(
                args.as_ptr(),
                args.len(),
                stream.as_ptr(),
                &mut stage,
            )
        } else {
            inference_cuda_exl3_grouped(args.as_ptr(), args.len(), stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

/// Most jobs of one launch, and most rows of one job: the limits of `exl3.h`.
pub(in crate::kernels::cuda) const JOBS_MAX: usize = 4096;
pub(in crate::kernels::cuda) const JOB_ROWS_MAX: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct JobsArgs {
    trellis: *const u16,
    trellis_bytes: usize,
    suh: *const f32,
    suh_bytes: usize,
    svh: *const f32,
    svh_bytes: usize,
    x: *const f32,
    x_bytes: usize,
    xh: *mut u16,
    xh_bytes: usize,
    z: *mut f32,
    z_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    jobs: *const i32,
    jobs_bytes: usize,
    trellis_stride: usize,
    scale_stride: usize,
    members: usize,
    k: usize,
    n: usize,
    total_rows: usize,
    out_ld: usize,
    out_col: usize,
    jobs_count: u32,
    max_rows: u32,
    bits_twice: c_int,
    book: c_int,
    in_rows: *const u32,
    in_rows_bytes: usize,
    out_rows: *const u32,
    out_rows_bytes: usize,
    x_rows: usize,
    out_total_rows: usize,
}
const _: () = assert!(
    size_of::<JobsArgs>() == 272
        && offset_of!(JobsArgs, in_rows) == 224
        && offset_of!(JobsArgs, x_rows) == 256
        && offset_of!(JobsArgs, flag) == 112
        && offset_of!(JobsArgs, jobs) == 128
        && offset_of!(JobsArgs, trellis_stride) == 144
        && offset_of!(JobsArgs, k) == 168
        && offset_of!(JobsArgs, total_rows) == 184
        && offset_of!(JobsArgs, jobs_count) == 208
        && offset_of!(JobsArgs, bits_twice) == 216
);
unsafe extern "C" {
    fn inference_cuda_exl3_jobs(
        args: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_jobs_fused_out(
        args: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
        fused_out: c_int,
    ) -> c_int;
    fn inference_cuda_exl3_jobs_modes(
        args: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
        fused_out: c_int,
        wide_input: c_int,
    ) -> c_int;
    fn inference_cuda_exl3_jobs_wide_input_kernel(args: *const JobsArgs, enabled: c_int) -> c_int;
    fn inference_cuda_exl3_jobs_rotated(
        args: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
        fused_out: c_int,
    ) -> c_int;
    fn inference_cuda_exl3_jobs_rotate_pair(
        first: *const JobsArgs,
        second: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_exl3_jobs_pair_usable(
        first: *const JobsArgs,
        second: *const JobsArgs,
    ) -> c_int;
    #[cfg(feature = "cuda-exl3-rotate-mutants")]
    fn inference_cuda_exl3_jobs_rotate_pair_mutant(
        first: *const JobsArgs,
        second: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
        mutant: c_int,
    ) -> c_int;
    #[cfg(test)]
    fn inference_cuda_exl3_jobs_wide_input_setting(text: *const std::ffi::c_char) -> c_int;
    #[cfg(feature = "cuda-exl3-rotate-mutants")]
    fn inference_cuda_exl3_jobs_rotate_mutant(
        args: *const JobsArgs,
        stream: *mut c_void,
        stage: *mut c_int,
        mutant: c_int,
    ) -> c_int;
    fn inference_cuda_exl3_jobs_kernel(
        args: *const JobsArgs,
        capability_major: c_int,
        ptx_version: c_int,
        enabled: c_int,
    ) -> c_int;
}

/// One job-table projection (`exl3.h`): the ranges, the bank strides in elements and the shape.
pub(in crate::kernels::cuda) struct JobsRaw<'a> {
    pub(in crate::kernels::cuda) trellis: RawRange<'a>,
    pub(in crate::kernels::cuda) suh: RawRange<'a>,
    pub(in crate::kernels::cuda) svh: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) xh: RawRange<'a>,
    pub(in crate::kernels::cuda) z: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) jobs: RawRange<'a>,
    /// u16 elements between the trellises of two members.
    pub(in crate::kernels::cuda) trellis_stride: usize,
    /// F32 values between the `suh` (and between the `svh`) of two members.
    pub(in crate::kernels::cuda) scale_stride: usize,
    pub(in crate::kernels::cuda) members: usize,
    pub(in crate::kernels::cuda) jobs_count: usize,
    pub(in crate::kernels::cuda) max_rows: usize,
    pub(in crate::kernels::cuda) total_rows: usize,
    pub(in crate::kernels::cuda) out_ld: usize,
    pub(in crate::kernels::cuda) out_col: usize,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) bits_twice: u8,
    pub(in crate::kernels::cuda) book: Codebook,
    /// Row lists (`exl3.h`): the row of `x` each table row reads and the row of `out` it lands in;
    /// `None` is the identity, and then `x_rows` / `out_total_rows` equal `total_rows`.
    pub(in crate::kernels::cuda) in_rows: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) out_rows: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) x_rows: usize,
    pub(in crate::kernels::cuda) out_total_rows: usize,
}

/// The numbers of a job-table call that fix its extents, apart from the ranges.
#[derive(Clone, Copy)]
pub(in crate::kernels::cuda) struct JobsShape {
    pub(in crate::kernels::cuda) trellis_stride: usize,
    pub(in crate::kernels::cuda) scale_stride: usize,
    pub(in crate::kernels::cuda) members: usize,
    pub(in crate::kernels::cuda) jobs_count: usize,
    pub(in crate::kernels::cuda) max_rows: usize,
    pub(in crate::kernels::cuda) total_rows: usize,
    pub(in crate::kernels::cuda) out_ld: usize,
    pub(in crate::kernels::cuda) out_col: usize,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) bits_twice: u8,
    pub(in crate::kernels::cuda) book: Codebook,
    pub(in crate::kernels::cuda) x_rows: usize,
    pub(in crate::kernels::cuda) out_total_rows: usize,
    pub(in crate::kernels::cuda) has_in_rows: bool,
    pub(in crate::kernels::cuda) has_out_rows: bool,
}

impl JobsRaw<'_> {
    pub(in crate::kernels::cuda) fn shape(&self) -> JobsShape {
        JobsShape {
            trellis_stride: self.trellis_stride,
            scale_stride: self.scale_stride,
            members: self.members,
            jobs_count: self.jobs_count,
            max_rows: self.max_rows,
            total_rows: self.total_rows,
            out_ld: self.out_ld,
            out_col: self.out_col,
            k: self.k,
            n: self.n,
            bits_twice: self.bits_twice,
            book: self.book,
            x_rows: self.x_rows,
            out_total_rows: self.out_total_rows,
            has_in_rows: self.in_rows.is_some(),
            has_out_rows: self.out_rows.is_some(),
        }
    }
}

/// The byte extents of a job-table call in ABI order: trellis, suh, svh, jobs, x, xh, z, out,
/// flag. `None` for a shape or stride outside the limits of `exl3.h`. The rotated input `xh` is
/// F16 (two bytes a value), where `x` is F32.
pub(in crate::kernels::cuda) fn jobs_extents(call: &JobsShape) -> Option<[usize; 9]> {
    let (packed_bytes, _) = geometry(call.k, call.n, call.bits_twice, call.book)?;
    let single = call.members == 1;
    let in_limits = (1..=JOBS_MAX).contains(&call.jobs_count)
        && (1..=JOB_ROWS_MAX).contains(&call.max_rows)
        && (1..=(1 << 24)).contains(&call.members)
        && (1..=65536).contains(&call.total_rows)
        && call.out_ld >= call.out_col.checked_add(call.n)?
        && call.out_ld <= (1 << 24)
        && call.out_col <= (1 << 24)
        && (1..=65536).contains(&call.x_rows)
        && (1..=65536).contains(&call.out_total_rows)
        && (call.has_in_rows || call.x_rows == call.total_rows)
        && (call.has_out_rows || call.out_total_rows == call.total_rows)
        && (single
            || (call.trellis_stride >= packed_bytes / 2
                && call.scale_stride >= call.k + call.n
                && call.trellis_stride <= (1 << 36)
                && call.scale_stride <= (1 << 36)));
    if !in_limits {
        return None;
    }
    let input = call.total_rows.checked_mul(call.k)?;
    let read = call.x_rows.checked_mul(call.k)?;
    let projected = call.total_rows.checked_mul(call.n)?;
    if input > (1usize << 32) || read > (1usize << 32) || projected > (1usize << 32) {
        return None;
    }
    let last = call.members - 1;
    let (trellis_step, scale_step) = if single {
        (0, 0)
    } else {
        (call.trellis_stride, call.scale_stride)
    };
    let output = (call.out_total_rows - 1)
        .checked_mul(call.out_ld)?
        .checked_add(call.out_col)?
        .checked_add(call.n)?;
    // Members and strides are bounded above, so these products fit; checked anyway.
    let member_end = |step: usize, tail: usize, width: usize| {
        last.checked_mul(step)?
            .checked_add(tail)?
            .checked_mul(width)
    };
    Some([
        member_end(trellis_step, packed_bytes / 2, 2)?,
        member_end(scale_step, call.k, 4)?,
        member_end(scale_step, call.n, 4)?,
        call.jobs_count * 16,
        read * 4,
        input * 2,
        projected * 4,
        output * 4,
        4,
    ])
}

/// The checked arguments of a job-table call: extents, alignment and disjoint writes proved.
fn jobs_args(call: &JobsRaw<'_>) -> Result<JobsArgs, DeviceFailure> {
    let needs = jobs_extents(&call.shape()).ok_or_else(invalid_argument)?;
    let ranges = [
        &call.trellis,
        &call.suh,
        &call.svh,
        &call.jobs,
        &call.x,
        &call.xh,
        &call.z,
        &call.out,
        &call.flag,
    ];
    for (i, range) in ranges.iter().enumerate() {
        let alignment = if i == 0 { 2 } else { 4 };
        if range.bytes() < needs[i] || !range.offset().is_multiple_of(alignment) {
            return Err(invalid_argument());
        }
        // Writes: xh, z, out and flag (indices 5 to 8) are disjoint from every other range.
        for (j, other) in ranges[..i].iter().enumerate() {
            if (i >= 5 || j >= 5) && !Span::of(range, needs[i]).apart(&Span::of(other, needs[j])) {
                return Err(invalid_argument());
            }
        }
    }
    // The row lists are read only: 4-byte aligned, one entry a table row, apart from every range
    // the call writes (xh, z, out and flag, indices 5 to 8 of the extents).
    for list in [&call.in_rows, &call.out_rows].into_iter().flatten() {
        let need = call.total_rows * 4;
        if list.bytes() < need || !list.offset().is_multiple_of(4) {
            return Err(invalid_argument());
        }
        for (j, written) in ranges[5..].iter().enumerate() {
            if !Span::of(list, need).apart(&Span::of(written, needs[5 + j])) {
                return Err(invalid_argument());
            }
        }
    }
    Ok(JobsArgs {
        trellis: call.trellis.address().cast(),
        trellis_bytes: call.trellis.bytes(),
        suh: call.suh.address().cast(),
        suh_bytes: call.suh.bytes(),
        svh: call.svh.address().cast(),
        svh_bytes: call.svh.bytes(),
        x: call.x.address().cast(),
        x_bytes: call.x.bytes(),
        xh: call.xh.address().cast(),
        xh_bytes: call.xh.bytes(),
        z: call.z.address().cast(),
        z_bytes: call.z.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        jobs: call.jobs.address().cast(),
        jobs_bytes: call.jobs.bytes(),
        trellis_stride: call.trellis_stride,
        scale_stride: call.scale_stride,
        members: call.members,
        k: call.k,
        n: call.n,
        total_rows: call.total_rows,
        out_ld: call.out_ld,
        out_col: call.out_col,
        jobs_count: u32::try_from(call.jobs_count).map_err(|_| invalid_argument())?,
        max_rows: u32::try_from(call.max_rows).map_err(|_| invalid_argument())?,
        bits_twice: c_int::from(call.bits_twice),
        book: book_tag(call.book),
        in_rows: call
            .in_rows
            .as_ref()
            .map_or(std::ptr::null(), |list| list.address().cast()),
        in_rows_bytes: call.in_rows.as_ref().map_or(0, |list| list.bytes()),
        out_rows: call
            .out_rows
            .as_ref()
            .map_or(std::ptr::null(), |list| list.address().cast()),
        out_rows_bytes: call.out_rows.as_ref().map_or(0, |list| list.bytes()),
        x_rows: call.x_rows,
        out_total_rows: call.out_total_rows,
    })
}

pub(in crate::kernels::cuda) fn jobs(
    stream: &RawStream,
    call: &JobsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let args = jobs_args(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: live bounded owners, complete extents, alignment and disjoint writes proved above and
    // repeated in C; the table is read on the device, and the C entry guards every job against the
    // bank and the row bounds. The explicit live stream is under the dispatch guard.
    let status = unsafe { inference_cuda_exl3_jobs(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// The same launches with an explicit fused-output-rotation request (the env default is off):
/// for the device tests and the GPU qualification, where an env flip would race the process.
/// Production builds never call it — the env decides — so the dead code is allowed there.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::kernels::cuda) fn jobs_fused(
    stream: &RawStream,
    call: &JobsRaw<'_>,
    fused_out: bool,
) -> Result<(), DeviceFailure> {
    let args = jobs_args(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: the same checked argument block as jobs; the request only selects the epilogue of
    // the register-decode kernel.
    let status = unsafe {
        inference_cuda_exl3_jobs_fused_out(&args, stream.as_ptr(), &mut stage, fused_out as c_int)
    };
    checked(status, stage)
}

/// The same launches with the fused output rotation and the warp-wide input rotation both
/// explicit, for the device tests.
#[cfg(test)]
pub(in crate::kernels::cuda) fn jobs_modes(
    stream: &RawStream,
    call: &JobsRaw<'_>,
    fused_out: bool,
    wide_input: bool,
) -> Result<(), DeviceFailure> {
    let args = jobs_args(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: the same checked argument block as jobs; the requests only select kernels that give
    // the same bytes.
    let status = unsafe {
        inference_cuda_exl3_jobs_modes(
            &args,
            stream.as_ptr(),
            &mut stage,
            fused_out as c_int,
            wide_input as c_int,
        )
    };
    checked(status, stage)
}

/// Whether the call would take the warp-wide input rotation under the request `enabled` (`exl3.h`).
#[cfg(test)]
pub(in crate::kernels::cuda) fn jobs_wide_input_usable(
    call: &JobsRaw<'_>,
    enabled: bool,
) -> Result<bool, DeviceFailure> {
    let args = jobs_args(call)?;
    // SAFETY: a checked argument block; the entry only reads its pointers' values and integers.
    match unsafe { inference_cuda_exl3_jobs_wide_input_kernel(&args, enabled as c_int) } {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid_argument()),
    }
}

/// The projection and output rotation of a call whose `xh` already holds the rotated rows;
/// `fused_out` is the fused output rotation request, `None` for the environment's.
pub(in crate::kernels::cuda) fn jobs_rotated(
    stream: &RawStream,
    call: &JobsRaw<'_>,
    fused_out: Option<bool>,
) -> Result<(), DeviceFailure> {
    let args = jobs_args(call)?;
    let mut stage = STAGE_NONE;
    let request = fused_out.map_or(-1, c_int::from);
    // SAFETY: the same checked argument block as jobs; the entry skips the input rotation and
    // reads the rows of xh that the pair rotation of the same stream wrote.
    let status =
        unsafe { inference_cuda_exl3_jobs_rotated(&args, stream.as_ptr(), &mut stage, request) };
    checked(status, stage)
}

/// The input rotation of two calls that read the same rows, in one pass (`exl3.h`).
pub(in crate::kernels::cuda) fn jobs_rotate_pair(
    stream: &RawStream,
    (first, second): (&JobsRaw<'_>, &JobsRaw<'_>),
) -> Result<(), DeviceFailure> {
    let (first, second) = (jobs_args(first)?, jobs_args(second)?);
    let mut stage = STAGE_NONE;
    // SAFETY: two checked argument blocks; the entry refuses calls that are not consistent or
    // aligned, and writes only the two xh ranges and the shared flag.
    let status = unsafe {
        inference_cuda_exl3_jobs_rotate_pair(&first, &second, stream.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

/// Whether the pair rotation accepts the two calls (`exl3.h`).
pub(in crate::kernels::cuda) fn jobs_pair_usable(
    first: &JobsRaw<'_>,
    second: &JobsRaw<'_>,
) -> Result<bool, DeviceFailure> {
    let (first, second) = (jobs_args(first)?, jobs_args(second)?);
    // SAFETY: two checked argument blocks; the entry only reads their integers and pointer values.
    match unsafe { inference_cuda_exl3_jobs_pair_usable(&first, &second) } {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid_argument()),
    }
}

/// The pair rotation by the deliberately wrong variant `mutant` (1 to 3).
#[cfg(feature = "cuda-exl3-rotate-mutants")]
pub(in crate::kernels::cuda) fn jobs_rotate_pair_mutant(
    stream: &RawStream,
    (first, second): (&JobsRaw<'_>, &JobsRaw<'_>),
    mutant: i32,
) -> Result<(), DeviceFailure> {
    let (first, second) = (jobs_args(first)?, jobs_args(second)?);
    let mut stage = STAGE_NONE;
    // SAFETY: as jobs_rotate_pair; the variant only changes which values land where.
    let status = unsafe {
        inference_cuda_exl3_jobs_rotate_pair_mutant(
            &first,
            &second,
            stream.as_ptr(),
            &mut stage,
            mutant,
        )
    };
    checked(status, stage)
}

/// The input rotation alone, by the deliberately wrong variant `mutant` (1 to 7).
#[cfg(feature = "cuda-exl3-rotate-mutants")]
pub(in crate::kernels::cuda) fn jobs_rotate_mutant(
    stream: &RawStream,
    call: &JobsRaw<'_>,
    mutant: i32,
) -> Result<(), DeviceFailure> {
    let args = jobs_args(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: the same checked argument block as jobs; the entry launches one kernel over the rows
    // of the table and writes only xh and the flag.
    let status = unsafe {
        inference_cuda_exl3_jobs_rotate_mutant(&args, stream.as_ptr(), &mut stage, mutant)
    };
    checked(status, stage)
}

/// Whether the call would run the register-decode kernel on the current device (`exl3.h`).
#[cfg(test)]
pub(in crate::kernels::cuda) fn jobs_use_registers(
    call: &JobsRaw<'_>,
) -> Result<bool, DeviceFailure> {
    let args = jobs_args(call)?;
    // SAFETY: a checked argument block; the entry only reads its integers and the device's
    // compute capability.
    match unsafe { inference_cuda_exl3_jobs_kernel(&args, -1, 0, 0) } {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid_argument()),
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
pub(in crate::kernels::cuda) mod launch_observer {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static COUNT: Cell<Option<u64>> = const { Cell::new(None) };
        static ROUTED: Cell<bool> = const { Cell::new(false) };
        /// 0 none, 1 shuffle, 2 gather, 3 mixed across a leg's dispatches.
        static GATHERED: Cell<u8> = const { Cell::new(0) };
        /// The projection kernel of the last odd-rate wide launch: 3 and 4 the rate 10 and rate 6
        /// body without and with DP4A, 2 the unchanged kernels, 0 none yet.
        static ODD_VARIANT: Cell<u32> = const { Cell::new(0) };
    }

    /// The projection kernel the last odd-rate wide launch of this thread enqueued, as the native
    /// launch reported it (see [`wide_odd`]).
    pub(in crate::kernels::cuda) fn last_odd_variant() -> u32 {
        ODD_VARIANT.with(Cell::get)
    }

    /// The odd-rate wide launch through the observed entry: the same launch plan and bytes, and the
    /// kernel that actually ran is recorded for [`last_odd_variant`].
    pub(super) fn wide_odd(
        args: &[LinearArgs],
        stream: &RawStream,
        stage: &mut c_int,
        dp4a: bool,
    ) -> c_int {
        let mut launches = 0;
        let mut variant = 0u32;
        // SAFETY: as `grouped`: live validated records and owners, C validates the group shapes
        // and aliases before any enqueue, and both outputs stay live through the call.
        let status = unsafe {
            inference_cuda_exl3_grouped_wide_odd_observed(
                args.as_ptr(),
                args.len(),
                stream.as_ptr(),
                stage,
                u32::from(dp4a),
                &mut launches,
                &mut variant,
            )
        };
        ODD_VARIANT.with(|last| last.set(if status == 0 { variant } else { 0 }));
        status
    }

    #[repr(C)]
    struct ObserveArgs {
        rows: usize,
        k: usize,
        n: usize,
        split_k: u32,
        midrows: u32,
        fail_at: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct Observation {
        emissions: [[u32; 7]; 3],
        count: u32,
        complete: u32,
    }

    const _: () = assert!(
        size_of::<ObserveArgs>() == 40
            && offset_of!(ObserveArgs, split_k) == 24
            && offset_of!(ObserveArgs, fail_at) == 32
            && size_of::<Observation>() == 92
            && offset_of!(Observation, count) == 84
            && offset_of!(Observation, complete) == 88
    );

    unsafe extern "C" {
        fn inference_cuda_exl3_observe_linear(
            args: *const ObserveArgs,
            observation: *mut Observation,
        ) -> c_int;
        fn inference_cuda_exl3_observe_grouped(
            args: *const ObserveArgs,
            observation: *mut Observation,
        ) -> c_int;
        fn inference_cuda_exl3_grouped_observed(
            args: *const LinearArgs,
            count: usize,
            stream: *mut c_void,
            stage: *mut c_int,
            launches: *mut u64,
            variant: *mut u32,
        ) -> c_int;
        fn inference_cuda_exl3_grouped_observed_gathered(
            args: *const LinearArgs,
            count: usize,
            stream: *mut c_void,
            stage: *mut c_int,
            launches: *mut u64,
            variant: *mut u32,
        ) -> c_int;
        fn inference_cuda_exl3_grouped_wide_odd_observed(
            args: *const LinearArgs,
            count: usize,
            stream: *mut c_void,
            stage: *mut c_int,
            dp4a: u32,
            launches: *mut u64,
            variant: *mut u32,
        ) -> c_int;
        fn inference_cuda_exl3_rows_kernel(max_rows: u32) -> c_int;
        fn inference_cuda_exl3_linear_observed(
            args: *const LinearArgs,
            split_k: u32,
            midrows: u32,
            stream: *mut c_void,
            stage: *mut c_int,
            launches: *mut u64,
        ) -> c_int;
    }

    /// The projection kernel a wide launch whose largest group has `max_rows` rows runs, as the
    /// native dispatch decides it (`rows_kernel_for` in `exl3.cu`): 1 the one-row body, 2, 4 and 8
    /// the rows kernels, 0 for a count the launch refuses.
    pub(in crate::kernels::cuda) fn rows_kernel(max_rows: u32) -> i32 {
        // SAFETY: a pure function of its argument; it reads no pointer and touches no device.
        unsafe { inference_cuda_exl3_rows_kernel(max_rows) }
    }

    pub(in crate::kernels::cuda) fn grouped_emissions(
        [rows, k, n]: [usize; 3],
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), DeviceFailure> {
        let args = ObserveArgs {
            rows,
            k,
            n,
            split_k: 2,
            midrows: 0,
            fail_at,
        };
        let mut out = Observation::default();
        // SAFETY: live C-layout host records; this calls only the shared host emitter.
        let status = unsafe { inference_cuda_exl3_observe_grouped(&args, &mut out) };
        checked(status, STAGE_NONE)?;
        if out.count > 3 || out.complete > 1 {
            return Err(invalid_argument());
        }
        Ok((
            out.emissions[..out.count as usize].to_vec(),
            out.complete != 0,
        ))
    }

    pub(super) fn grouped(
        args: &[LinearArgs],
        stream: &RawStream,
        stage: &mut c_int,
        gather: bool,
    ) -> c_int {
        let mut launches = 0;
        let mut variant = 0u32;
        // SAFETY: live validated records and owners; C checks cross-group aliases
        // before launch. All three writable host outputs remain live through this
        // call; C writes the variant only after a successful projection enqueue.
        let status = unsafe {
            if gather {
                inference_cuda_exl3_grouped_observed_gathered(
                    args.as_ptr(),
                    args.len(),
                    stream.as_ptr(),
                    stage,
                    &mut launches,
                    &mut variant,
                )
            } else {
                inference_cuda_exl3_grouped_observed(
                    args.as_ptr(),
                    args.len(),
                    stream.as_ptr(),
                    stage,
                    &mut launches,
                    &mut variant,
                )
            }
        };
        COUNT.with(|count| count.set(Some(count.get().unwrap().checked_add(launches).unwrap())));
        // The variant comes from the device launch, never the request; a refusal
        // or zero launches records nothing and fails the leg's assert.
        if status == 0 && variant != 0 {
            GATHERED.with(|used| {
                let code = variant as u8;
                let seen = used.get();
                if seen == 0 {
                    used.set(code);
                } else if seen != code {
                    used.set(3);
                }
            });
        }
        status
    }

    /// Host metadata only: the C binding uses the production emitter, not CUDA.
    pub(in crate::kernels::cuda) fn emissions(
        [rows, k, n]: [usize; 3],
        split_k: u32,
        midrows: bool,
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), DeviceFailure> {
        let args = ObserveArgs {
            rows,
            k,
            n,
            split_k,
            midrows: u32::from(midrows),
            fail_at,
        };
        let mut out = Observation::default();
        // SAFETY: both C-layout host records are live and writable for this call;
        // the entry validates geometry and never interprets a device pointer.
        let status = unsafe { inference_cuda_exl3_observe_linear(&args, &mut out) };
        checked(status, STAGE_NONE)?;
        if out.count > 3 || out.complete > 1 {
            return Err(invalid_argument());
        }
        Ok((
            out.emissions[..out.count as usize].to_vec(),
            out.complete != 0,
        ))
    }

    struct Capture;
    impl Drop for Capture {
        fn drop(&mut self) {
            COUNT.with(|count| count.set(None));
        }
    }

    pub(in crate::kernels::cuda) fn capture<R>(run: impl FnOnce() -> R) -> (R, u64) {
        COUNT.with(|count| {
            assert!(count.get().is_none(), "nested EXL3 launch capture");
            count.set(Some(0));
        });
        // Reset here, not in Drop: callers read the variant after capture returns.
        GATHERED.with(|used| used.set(0));
        let guard = Capture;
        let result = run();
        let count = COUNT.with(|count| count.get().unwrap());
        drop(guard);
        (result, count)
    }

    pub(in crate::kernels::cuda) struct RoutedScope(bool);
    impl Drop for RoutedScope {
        fn drop(&mut self) {
            ROUTED.with(|routed| routed.set(self.0));
        }
    }

    pub(in crate::kernels::cuda) fn routed_scope() -> RoutedScope {
        RoutedScope(ROUTED.with(|routed| routed.replace(true)))
    }

    pub(super) fn recording() -> bool {
        ROUTED.with(Cell::get) && COUNT.with(|count| count.get().is_some())
    }

    /// Which grouped variant a captured leg actually ran: none, all shuffle,
    /// all gather, or mixed. Written by C at the successful projection enqueue.
    pub(in crate::kernels::cuda) fn grouped_variant() -> Option<&'static str> {
        match GATHERED.with(Cell::get) {
            0 => None,
            1 => Some("shuffle"),
            2 => Some("gather"),
            _ => Some("mixed"),
        }
    }

    pub(super) fn linear(
        args: &LinearArgs,
        split_k: u32,
        midrows: bool,
        stream: &RawStream,
        stage: &mut c_int,
    ) -> c_int {
        let mut launches = 0;
        // SAFETY: the caller validated all live device spans and owns the dispatch
        // guard; the two additional writable host outputs outlive this call.
        let status = unsafe {
            inference_cuda_exl3_linear_observed(
                args,
                split_k,
                u32::from(midrows),
                stream.as_ptr(),
                stage,
                &mut launches,
            )
        };
        COUNT.with(|count| count.set(Some(count.get().unwrap().checked_add(launches).unwrap())));
        status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An argument block with the given alignment-relevant fields; its pointers are never read.
    fn selector_args(
        xh: usize,
        trellis: usize,
        stride: usize,
        members: usize,
        k: usize,
    ) -> JobsArgs {
        JobsArgs {
            trellis: trellis as *const u16,
            trellis_bytes: 0,
            suh: std::ptr::null(),
            suh_bytes: 0,
            svh: std::ptr::null(),
            svh_bytes: 0,
            x: std::ptr::null(),
            x_bytes: 0,
            xh: xh as *mut u16,
            xh_bytes: 0,
            z: std::ptr::null_mut(),
            z_bytes: 0,
            out: std::ptr::null_mut(),
            out_bytes: 0,
            flag: std::ptr::null_mut(),
            flag_bytes: 0,
            jobs: std::ptr::null(),
            jobs_bytes: 0,
            trellis_stride: stride,
            scale_stride: 0,
            members,
            k,
            n: 0,
            total_rows: 0,
            out_ld: 0,
            out_col: 0,
            jobs_count: 0,
            max_rows: 0,
            bits_twice: 8,
            book: 0,
            in_rows: std::ptr::null(),
            in_rows_bytes: 0,
            out_rows: std::ptr::null(),
            out_rows_bytes: 0,
            x_rows: 0,
            out_total_rows: 0,
        }
    }

    #[test]
    fn the_register_kernel_is_chosen_by_capability_ptx_alignment_and_k() {
        // The device facts and the switch are arguments, so the test needs no device and does not
        // depend on FIFI_EXL3_JOBS_REG in the environment.
        let select = |args: &JobsArgs, major: c_int, ptx: c_int, enabled: c_int| {
            // SAFETY: the entry reads the integers of the block; a capability of 0 or more is
            // taken as given and never queries a device.
            unsafe { inference_cuda_exl3_jobs_kernel(args, major, ptx, enabled) }
        };
        let aligned = selector_args(0x10_0000, 0x20_0000, 4096, 5, 2048);
        // Compute capability 8.6 (the 3090) on an sm_86 binary takes it; 7.2 does not.
        assert_eq!(select(&aligned, 8, 86, 1), 1);
        assert_eq!(select(&aligned, 9, 90, 1), 1);
        assert_eq!(select(&aligned, 7, 72, 1), 0);
        assert_eq!(select(&aligned, 0, 0, 1), 0);
        // An sm_72-only build run on a compute capability 8 device (PTX JIT) has the trap stub, and
        // reports the SASS arch of the device: only the ptx version of the code shows it.
        assert_eq!(select(&aligned, 8, 72, 1), 0);
        assert_eq!(select(&aligned, 8, 80, 1), 1);
        // The rollback switch.
        assert_eq!(select(&aligned, 8, 86, 0), 0);
        // An input row start or a trellis start that is only 4- or 2-byte aligned, a member stride
        // that is not a multiple of 8 elements, or a K that is not a multiple of 32: old kernel.
        for args in [
            selector_args(0x10_0004, 0x20_0000, 4096, 5, 2048),
            selector_args(0x10_0000, 0x20_0002, 4096, 5, 2048),
            selector_args(0x10_0000, 0x20_0000, 4099, 5, 2048),
            selector_args(0x10_0000, 0x20_0000, 4096, 5, 2064),
        ] {
            assert_eq!(select(&args, 8, 86, 1), 0);
        }
        // One member ignores the stride.
        assert_eq!(
            select(
                &selector_args(0x10_0000, 0x20_0000, 4099, 1, 2048),
                8,
                86,
                1
            ),
            1
        );
        // A null block is refused.
        assert_eq!(
            // SAFETY: a null block is refused before any read.
            unsafe { inference_cuda_exl3_jobs_kernel(std::ptr::null(), 8, 86, 1) },
            -1
        );
    }

    #[test]
    fn the_warp_rotation_is_chosen_by_request_and_alignment() {
        // The decision reads pointer values and integers only, so the block needs no device.
        let usable =
            |x: usize, suh: usize, xh: usize, members: usize, stride: usize, enabled: c_int| {
                let args = JobsArgs {
                    x: x as *const f32,
                    suh: suh as *const f32,
                    scale_stride: stride,
                    ..selector_args(xh, 0x20_0000, 4096, members, 2048)
                };
                // SAFETY: the entry reads the integers of the block; no pointer is dereferenced.
                unsafe { inference_cuda_exl3_jobs_wide_input_kernel(&args, enabled) }
            };
        assert_eq!(usable(0x10_0000, 0x30_0000, 0x40_0000, 5, 2560, 1), 1);
        // Not requested.
        assert_eq!(usable(0x10_0000, 0x30_0000, 0x40_0000, 5, 2560, 0), 0);
        // x or suh off a 16-byte boundary, xh off an 8-byte boundary, a member stride of the
        // scales that is not a multiple of 4 floats.
        assert_eq!(usable(0x10_0008, 0x30_0000, 0x40_0000, 5, 2560, 1), 0);
        assert_eq!(usable(0x10_0000, 0x30_0004, 0x40_0000, 5, 2560, 1), 0);
        assert_eq!(usable(0x10_0000, 0x30_0000, 0x40_0004, 5, 2560, 1), 0);
        assert_eq!(usable(0x10_0000, 0x30_0000, 0x40_0000, 5, 2562, 1), 0);
        // xh on an 8-byte boundary is enough; one member ignores the stride.
        assert_eq!(usable(0x10_0000, 0x30_0000, 0x40_0008, 5, 2560, 1), 1);
        assert_eq!(usable(0x10_0000, 0x30_0000, 0x40_0000, 1, 2562, 1), 1);
        // A null block is refused.
        assert_eq!(
            // SAFETY: a null block is refused before any read.
            unsafe { inference_cuda_exl3_jobs_wide_input_kernel(std::ptr::null(), 1) },
            -1
        );
    }

    #[test]
    fn the_pair_rotation_needs_calls_that_share_their_rows_and_keep_their_outputs_apart() {
        let block = |x: usize, xh: usize, rows: usize, k: usize, flag: usize| JobsArgs {
            x: x as *const f32,
            suh: 0x30_0000 as *const f32,
            xh: xh as *mut u16,
            xh_bytes: rows * k * 2,
            flag: flag as *mut u32,
            total_rows: rows,
            max_rows: 64,
            jobs_count: 3,
            jobs: 0x50_0000 as *const _,
            scale_stride: 2560,
            x_rows: rows,
            ..selector_args(xh, 0x20_0000, 4096, 5, k)
        };
        let usable = |a: &JobsArgs, b: &JobsArgs| {
            // SAFETY: the entry reads the integers and pointer values of the blocks only.
            unsafe { inference_cuda_exl3_jobs_pair_usable(a, b) }
        };
        let first = block(0x10_0000, 0x40_0000, 64, 2048, 0x60_0000);
        let second = block(0x10_0000, 0x48_0000, 64, 2048, 0x60_0000);
        assert_eq!(usable(&first, &second), 1);
        // Each shared fact flipped once on the second call, on its own.
        let flips = [
            (
                "x_rows",
                JobsArgs {
                    x_rows: 63,
                    ..second
                },
            ),
            (
                "in_rows",
                JobsArgs {
                    in_rows: 0x70_0000 as *const u32,
                    ..second
                },
            ),
            (
                "jobs",
                JobsArgs {
                    jobs: 0x50_0010 as *const _,
                    ..second
                },
            ),
            (
                "jobs_count",
                JobsArgs {
                    jobs_count: 4,
                    ..second
                },
            ),
            (
                "max_rows",
                JobsArgs {
                    max_rows: 65,
                    ..second
                },
            ),
            (
                "total_rows",
                JobsArgs {
                    total_rows: 63,
                    ..second
                },
            ),
            (
                "members",
                JobsArgs {
                    members: 4,
                    ..second
                },
            ),
        ];
        for (name, bad) in &flips {
            assert_eq!(usable(&first, bad), 0, "{name}");
        }
        // An output that overlaps something the rotation reads: x, either call's scales, the
        // table or the row list.
        let both = |f: fn(JobsArgs) -> JobsArgs| usable(&f(first), &f(second));
        assert_eq!(
            both(|a| JobsArgs {
                x_bytes: 0x40_0000,
                ..a
            }),
            0,
            "x"
        );
        assert_eq!(
            both(|a| JobsArgs {
                jobs: 0x41_0000 as *const _,
                jobs_bytes: 48,
                ..a
            }),
            0,
            "jobs"
        );
        assert_eq!(
            both(|a| JobsArgs {
                in_rows: 0x41_0000 as *const u32,
                in_rows_bytes: 64,
                ..a
            }),
            0,
            "in_rows"
        );
        assert_eq!(
            usable(
                &JobsArgs {
                    suh_bytes: 0x20_0000,
                    ..first
                },
                &second
            ),
            0,
            "first scales"
        );
        assert_eq!(
            usable(
                &first,
                &JobsArgs {
                    suh_bytes: 0x20_0000,
                    ..second
                }
            ),
            0,
            "second scales"
        );
        // A different input buffer, row count, width, flag, or the same output buffer.
        assert_eq!(
            usable(&first, &block(0x11_0000, 0x48_0000, 64, 2048, 0x60_0000)),
            0
        );
        assert_eq!(
            usable(&first, &block(0x10_0000, 0x48_0000, 63, 2048, 0x60_0000)),
            0
        );
        assert_eq!(
            usable(&first, &block(0x10_0000, 0x48_0000, 64, 1024, 0x60_0000)),
            0
        );
        assert_eq!(
            usable(&first, &block(0x10_0000, 0x48_0000, 64, 2048, 0x60_0010)),
            0
        );
        assert_eq!(usable(&first, &first), 0);
        // Output ranges that overlap (the second starts inside the first's 256 KiB).
        assert_eq!(
            usable(&first, &block(0x10_0000, 0x42_0000, 64, 2048, 0x60_0000)),
            0
        );
        // An output off an 8-byte boundary falls back to the separate calls.
        assert_eq!(
            usable(&first, &block(0x10_0000, 0x48_0004, 64, 2048, 0x60_0000)),
            0
        );
        // SAFETY: a null block is refused before any read.
        assert_eq!(
            unsafe { inference_cuda_exl3_jobs_pair_usable(std::ptr::null(), &first) },
            -1
        );
    }

    #[test]
    fn the_warp_rotation_switch_text_is_read_strictly() {
        let read = |text: Option<&str>| {
            let owned = text.map(|t| std::ffi::CString::new(t).expect("no NUL"));
            let pointer = owned.as_ref().map_or(std::ptr::null(), |t| t.as_ptr());
            // SAFETY: a null pointer or a NUL-terminated string that outlives the call.
            unsafe { inference_cuda_exl3_jobs_wide_input_setting(pointer) }
        };
        assert_eq!(read(None), 0);
        assert_eq!(read(Some("0")), 0);
        assert_eq!(read(Some("1")), 1);
        for bad in ["", "2", "01", "10", "on", "true", " 1", "1 ", "-1", "warp"] {
            assert_eq!(read(Some(bad)), -1, "{bad:?}");
        }
    }

    #[test]
    fn the_job_table_extents_size_the_rotated_input_as_f16() {
        // k = 256, n = 384 at 4 bits (bits_twice 8): 16 row tiles and 24 column tiles of 16 values.
        let shape = JobsShape {
            trellis_stride: 0,
            scale_stride: 0,
            members: 1,
            jobs_count: 2,
            max_rows: 16,
            total_rows: 32,
            out_ld: 384,
            out_col: 0,
            k: 256,
            n: 384,
            bits_twice: 8,
            book: Codebook::Mul1,
            x_rows: 32,
            out_total_rows: 32,
            has_in_rows: false,
            has_out_rows: false,
        };
        let needs = jobs_extents(&shape).expect("a valid shape");
        assert_eq!(needs[3], 2 * 16, "the table: 16 bytes a job");
        assert_eq!(needs[4], 32 * 256 * 4, "x holds F32 values");
        assert_eq!(needs[5], 32 * 256 * 2, "xh holds F16 values");
        assert_eq!(needs[6], 32 * 384 * 4, "z holds F32 values");
        assert_eq!(needs[7], 32 * 384 * 4, "out holds F32 values at ld = n");
        assert_eq!(needs[8], 4, "the flag is one word");
        // A window inside a wider row moves the end of the output only.
        let wide = JobsShape {
            out_ld: 400,
            out_col: 16,
            ..shape
        };
        let needs = jobs_extents(&wide).expect("a window");
        assert_eq!(needs[7], (31 * 400 + 16 + 384) * 4);
        assert_eq!(needs[5], 32 * 256 * 2);
        // Row lists: x and out follow their own row counts, xh stays one row a table row.
        let listed = JobsShape {
            x_rows: 10,
            out_total_rows: 40,
            has_in_rows: true,
            has_out_rows: true,
            ..shape
        };
        let needs = jobs_extents(&listed).expect("lists");
        assert_eq!(needs[4], 10 * 256 * 4, "x holds x_rows rows");
        assert_eq!(needs[5], 32 * 256 * 2, "xh holds a row for every table row");
        assert_eq!(
            needs[7],
            (39 * 384 + 384) * 4,
            "out holds out_total_rows rows"
        );
        // Without a list the row counts must be those of the table.
        for bad in [
            JobsShape {
                x_rows: 10,
                ..shape
            },
            JobsShape {
                out_total_rows: 40,
                ..shape
            },
            JobsShape {
                x_rows: 0,
                has_in_rows: true,
                ..shape
            },
            JobsShape {
                out_total_rows: 65537,
                has_out_rows: true,
                ..shape
            },
        ] {
            assert!(jobs_extents(&bad).is_none());
        }
        // Over the row cap, no jobs, and a window past the row refuse.
        for bad in [
            JobsShape {
                max_rows: 257,
                ..shape
            },
            JobsShape {
                jobs_count: 0,
                ..shape
            },
            JobsShape {
                out_ld: 383,
                ..shape
            },
        ] {
            assert!(jobs_extents(&bad).is_none());
        }
    }

    #[test]
    fn midrows_geometry_requires_9_to_144_rows_and_f32_working_spans() {
        for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
            for rows in [
                0usize,
                1,
                8,
                9,
                15,
                16,
                31,
                32,
                63,
                64,
                127,
                128,
                144,
                145,
                65536,
                usize::MAX,
            ] {
                let expected = (9..=144).contains(&rows).then(|| {
                    [
                        8192,
                        1024,
                        512,
                        512,
                        rows * 1024,
                        rows * 1024,
                        rows * 512,
                        rows * 512,
                        4,
                    ]
                });
                assert_eq!(
                    midrows_extents(rows, 256, 128, 4, book),
                    expected,
                    "{book:?}/rows{rows}"
                );
            }
        }
        for (k, n, bits) in [
            (0, 128, 4),
            (127, 128, 4),
            (128, 129, 4),
            (256, 128, 9),
            (usize::MAX, 128, 4),
        ] {
            assert!(midrows_extents(9, k, n, bits, Codebook::Mul1).is_none());
        }
    }

    #[test]
    fn batched_geometry_checks_both_input_and_output_products() {
        let book = Codebook::Mul1;
        assert_eq!(
            linear_extents(3, 256, 128, 3, book, 2),
            Some([6144, 1024, 512, 512, 3072, 3072, 1536, 1536, 4])
        );
        for rows in [0, 65537, usize::MAX] {
            assert!(linear_extents(rows, 256, 128, 3, book, 2).is_none());
        }
        assert!(linear_extents(65536, 1 << 22, 128, 2, book, 2).is_none());
        assert!(linear_extents(65536, 128, 1 << 22, 2, book, 2).is_none());
        assert!(linear_extents(65536, 65536, 65536, 16, book, 2).is_some());
    }

    #[test]
    fn split_extents_reserve_all_partial_banks_and_reject_other_counts() {
        for splits in [2u32, 4, 8] {
            for rows in [1, 7, 8, 9, 2048] {
                let needs = linear_extents(rows, 384, 128, 7, Codebook::Mul1, splits).unwrap();
                let banks = if rows <= 8 { splits as usize - 1 } else { 1 };
                assert_eq!(needs[6], rows * 128 * 4 * banks);
                assert_eq!(needs[7], rows * 128 * 4);
            }
        }
        for splits in [0, 1, 3, 16, u32::MAX] {
            assert!(linear_extents(1, 128, 128, 2, Codebook::Mul1, splits).is_none());
            assert!(linear_extents(9, 128, 128, 2, Codebook::Mul1, splits).is_none());
        }
    }

    fn grouped_c_fixture(rows: usize, first: usize) -> LinearArgs {
        use std::ptr::{null, without_provenance, without_provenance_mut};
        LinearArgs {
            trellis: without_provenance(0x10000),
            trellis_bytes: 6144,
            suh: without_provenance(0x20000),
            suh_bytes: 1024,
            svh: without_provenance(0x30000),
            svh_bytes: 512,
            bias: null(),
            bias_bytes: 0,
            x: without_provenance(0x50000),
            x_bytes: rows * 1024,
            xh: without_provenance_mut(0x60000 + first * 1024),
            xh_bytes: rows * 1024,
            z: without_provenance_mut(0x70000 + first * 512),
            z_bytes: rows * 512,
            out: without_provenance_mut(0x80000 + first * 512),
            out_bytes: rows * 512,
            flag: without_provenance_mut(0x90000),
            flag_bytes: 4,
            rows,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: 2,
            fused_publish: 0,
        }
    }

    #[test]
    fn malformed_grouped_c_arguments_refuse_before_touching_cuda() {
        use std::ptr::{without_provenance, without_provenance_mut};
        let base = [grouped_c_fixture(1, 0), grouped_c_fixture(1, 1)];
        let mut cases = vec![
            ("no groups", vec![]),
            ("49 groups", vec![base[0]; 49]),
            ("zero rows", vec![grouped_c_fixture(0, 0)]),
            ("nine rows", vec![grouped_c_fixture(9, 0)]),
            ("reordered groups", vec![base[1], base[0]]),
        ];
        // Group 8 keeps the overwritten flag aligned if the pair cap regresses.
        let mut pairs: Vec<_> = (0..8).map(|i| grouped_c_fixture(6, i * 6)).collect();
        pairs.push(grouped_c_fixture(1, 48));
        cases.push(("49 pairs", pairs));
        // xh and z must stay contiguous across calls; out rows may scatter since the
        // scattered-outs merge, but must stay disjoint (overlaps would double-write).
        for (name, field) in [("xh gap", 0), ("z gap", 1), ("out overlap", 2)] {
            let mut calls = base;
            match field {
                0 => calls[1].xh = without_provenance_mut(0x60404),
                1 => calls[1].z = without_provenance_mut(0x70204),
                _ => calls[1].out = without_provenance_mut(0x80100),
            }
            cases.push((name, calls.to_vec()));
        }
        let mut calls = base;
        calls[1].k = 512;
        calls[1].trellis_bytes = 12288;
        calls[1].suh_bytes = 2048;
        calls[1].x_bytes = 2048;
        calls[1].xh_bytes = 2048;
        cases.push(("different k with valid local extents", calls.to_vec()));
        let mut calls = base;
        calls[1].n = 256;
        calls[1].trellis_bytes = 12288;
        calls[1].svh_bytes = 1024;
        calls[1].z_bytes = 1024;
        calls[1].out_bytes = 1024;
        cases.push(("different n with valid local extents", calls.to_vec()));
        let mut calls = base;
        calls[1].flag = without_provenance_mut(0x91000);
        cases.push(("different flag", calls.to_vec()));
        let mut calls = base;
        // Each call is disjoint locally; group-0 Z overlaps only group-1 XH.
        calls[0].z = without_provenance_mut(0x60600);
        calls[1].z = without_provenance_mut(0x60800);
        cases.push(("Z union overlaps following XH", calls.to_vec()));
        let mut calls = base;
        // This read ends at group-1 out, so only the cross-group union overlaps.
        calls[1].x = without_provenance(0x7fe00);
        cases.push(("input overlaps preceding output", calls.to_vec()));
        let mut calls = base;
        // The 6144-byte read ends at group-1 Z; only group-0 Z overlaps it.
        calls[1].trellis = without_provenance(0x6ea00);
        cases.push(("trellis overlaps preceding Z", calls.to_vec()));
        let mut calls = base;
        calls[1].z_bytes -= 1;
        cases.push(("short split-2 Z bank", calls.to_vec()));
        for (name, calls) in cases {
            let mut stage = -1;
            // SAFETY: malformed metadata must be refused before dereferencing the
            // fabricated spans or using the sentinel stream, as for the batched fixture.
            let status = unsafe {
                inference_cuda_exl3_grouped(
                    calls.as_ptr(),
                    calls.len(),
                    without_provenance_mut(16),
                    &mut stage,
                )
            };
            assert_eq!(
                status,
                super::super::super::STATUS_INVALID_ARGUMENT,
                "{name}"
            );
            assert_eq!(stage, STAGE_NONE, "{name}");
        }
    }

    #[test]
    fn malformed_grouped_rust_arguments_refuse_before_touching_cuda() {
        use super::super::RawDevice;
        use crate::kernels::cuda as adapter;
        use std::ptr::NonNull;
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut memory = vec![0u32; 9000];
        let device = RawDevice {
            ptr: NonNull::new(memory.as_mut_ptr().cast()).unwrap(),
            bytes: memory.len() * 4,
        };
        let range = |offset, bytes| RawRange::new(&device, offset, bytes).unwrap();
        let call = |rows, split_k| LinearRaw {
            trellis: range(0, 6144),
            suh: range(8192, 1024),
            svh: range(9216, 512),
            bias: None,
            x: range(10240, 8192),
            xh: range(18432, 8192),
            z: range(26624, 4096),
            out: range(30720, 4096),
            flag: range(34816, 4),
            rows,
            split_k,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: Codebook::Mul1,
            fused: false,
        };
        // Existing raw-wrapper seam: invalid calls must never consume this stream.
        let stream = RawStream(NonNull::dangling());
        for (rows, split, count) in [(1, 2, 0), (1, 2, 49), (0, 2, 1), (9, 2, 1), (1, 4, 1)] {
            let calls: Vec<_> = (0..count).map(|_| call(rows, split)).collect();
            assert_eq!(grouped(&stream, &calls, false), Err(invalid_argument()));
        }
        let mut calls: Vec<_> = (0..6).map(|_| call(8, 2)).collect();
        calls.push(call(1, 2));
        assert_eq!(grouped(&stream, &calls, false), Err(invalid_argument()));
        let mut short = call(1, 2);
        short.z = range(26624, 511);
        assert_eq!(grouped(&stream, &[short], true), Err(invalid_argument()));

        let mut calls = [call(1, 2), call(1, 2)];
        calls[0].z = range(19968, 512);
        calls[1].xh = range(19456, 1024);
        calls[1].z = range(20480, 512);
        calls[1].out = range(31232, 512);
        for call in &calls {
            assert!(linear_args(call, false).is_ok());
        }
        let failure = grouped(&stream, &calls, false).expect_err("cross-group XH/Z alias");
        assert_eq!(failure, invalid_argument());

        // launch_on maps this native refusal to a terminal error, not BadRequest.
        // A local latch exercises that mapper without poisoning the shared backend.
        let terminal = AtomicBool::new(false);
        assert!(adapter::ensure_usable(&terminal).is_ok());
        let error = adapter::native_error(
            &terminal,
            adapter::stage_operation(failure.stage),
            failure.status,
        );
        assert!(matches!(
            error,
            crate::EngError::Backend {
                operation: adapter::ADAPTER_OPERATION,
                ..
            }
        ));
        assert!(terminal.load(Ordering::Acquire));
        assert!(matches!(
            adapter::ensure_usable(&terminal),
            Err(crate::EngError::Backend {
                operation: adapter::EXECUTE_OPERATION,
                ..
            })
        ));
    }

    #[test]
    fn malformed_batched_c_arguments_refuse_before_touching_cuda() {
        let base = LinearArgs {
            trellis: std::ptr::without_provenance(0x10000),
            trellis_bytes: 6144,
            suh: std::ptr::without_provenance(0x20000),
            suh_bytes: 1024,
            svh: std::ptr::without_provenance(0x30000),
            svh_bytes: 512,
            bias: std::ptr::null(),
            bias_bytes: 0,
            x: std::ptr::without_provenance(0x50000),
            x_bytes: 1024,
            xh: std::ptr::without_provenance_mut(0x60000),
            xh_bytes: 1024,
            z: std::ptr::without_provenance_mut(0x70000),
            z_bytes: 512,
            out: std::ptr::without_provenance_mut(0x80000),
            out_bytes: 512,
            flag: std::ptr::without_provenance_mut(0x90000),
            flag_bytes: 4,
            rows: 1,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: 2,
            fused_publish: 0,
        };
        let mut cases = [base; 16];
        cases[0].rows = 0;
        cases[1].rows = 65537;
        cases[2].k = usize::MAX;
        cases[3].book = 3;
        cases[4].bits_twice = 9;
        cases[5].trellis_bytes -= 1;
        cases[6].suh_bytes -= 1;
        cases[7].out_bytes -= 1;
        cases[8].flag_bytes = 2;
        cases[9].out = base.x.cast_mut();
        cases[10].xh = base.z;
        cases[11].flag = std::ptr::without_provenance_mut(0x90001);
        cases[12].bias_bytes = 4;
        cases[13].out = std::ptr::without_provenance_mut(usize::MAX & !3);
        cases[14].svh = std::ptr::null();
        cases[15].rows = usize::MAX;
        for (index, args) in cases.iter().enumerate() {
            let mut stage = -99;
            // SAFETY: malformed scalar/region arguments are host-rejected before CUDA;
            // the opaque stream and fabricated device addresses must remain unused.
            let status = unsafe {
                inference_cuda_exl3_linear(args, std::ptr::without_provenance_mut(16), &mut stage)
            };
            assert_eq!(status, invalid_argument().status, "case {index}");
            assert_eq!(stage, STAGE_NONE, "case {index}");
        }
        for split_k in [2u32, 4, 8] {
            let need = 512 * (split_k as usize - 1);
            let mut bad = [base; 7];
            for args in &mut bad {
                args.z_bytes = need;
            }
            bad[0].z_bytes -= 1;
            bad[1].z = std::ptr::without_provenance_mut(0x70001);
            let last_bank = 0x70000 + need - 512;
            bad[2].flag = std::ptr::without_provenance_mut(last_bank);
            bad[3].out = std::ptr::without_provenance_mut(last_bank);
            bad[4].xh = std::ptr::without_provenance_mut(last_bank);
            bad[5].suh = std::ptr::without_provenance(last_bank);
            bad[6].z = std::ptr::without_provenance_mut(usize::MAX & !3);
            for (index, args) in bad.iter().enumerate() {
                let mut stage = -99;
                // SAFETY: every full-bank span defect must refuse before CUDA.
                let status = unsafe {
                    inference_cuda_exl3_linear_split(
                        args,
                        split_k,
                        std::ptr::without_provenance_mut(16),
                        &mut stage,
                    )
                };
                assert_eq!(
                    status,
                    invalid_argument().status,
                    "split{split_k}/case{index}"
                );
                assert_eq!(stage, STAGE_NONE, "split{split_k}/case{index}");
            }
        }
        for rows in [0usize, 1, 8, 145, 65536] {
            let args = LinearArgs {
                x: std::ptr::without_provenance(0x10000000),
                x_bytes: rows * 1024,
                xh: std::ptr::without_provenance_mut(0x20000000),
                xh_bytes: rows * 1024,
                z: std::ptr::without_provenance_mut(0x30000000),
                z_bytes: rows * 512,
                out: std::ptr::without_provenance_mut(0x40000000),
                out_bytes: rows * 512,
                rows,
                ..base
            };
            let mut stage = -99;
            // SAFETY: bad midrows counts must refuse before CUDA. For nonzero rows,
            // the regions fit legacy geometry, so no other span defect masks this guard.
            let status = unsafe {
                inference_cuda_exl3_linear_midrows(
                    &args,
                    std::ptr::without_provenance_mut(16),
                    &mut stage,
                )
            };
            assert_eq!(status, invalid_argument().status, "midrows/rows{rows}");
            assert_eq!(stage, STAGE_NONE, "midrows/rows{rows}");
        }
        for split_k in [0, 1, 3, 16, u32::MAX] {
            let mut stage = -99;
            // SAFETY: invalid split counts refuse before the sentinel stream is used.
            let status = unsafe {
                inference_cuda_exl3_linear_split(
                    &base,
                    split_k,
                    std::ptr::without_provenance_mut(16),
                    &mut stage,
                )
            };
            assert_eq!(status, invalid_argument().status, "split{split_k}");
            assert_eq!(stage, STAGE_NONE, "split{split_k}");
        }
    }

    #[test]
    fn malformed_c_arguments_refuse_before_touching_cuda() {
        let base = PrepArgs {
            trellis: std::ptr::without_provenance(0x10000),
            trellis_bytes: 4096,
            out: std::ptr::without_provenance_mut(0x20000),
            out_bytes: 32768,
            k: 128,
            n: 128,
            bits_twice: 4,
            book: 0,
        };
        let mut cases = [base; 8];
        cases[0].k = 0;
        cases[1].n = usize::MAX;
        cases[2].book = 3;
        cases[3].bits_twice = 3;
        cases[4].trellis_bytes -= 1;
        cases[5].out_bytes -= 1;
        cases[6].out = base.trellis.cast_mut();
        cases[7].out = std::ptr::without_provenance_mut(0x20001);
        for (index, args) in cases.iter().enumerate() {
            let mut stage = -99;
            // SAFETY: every case is rejected in host validation before any pointer is
            // dereferenced or CUDA called. The opaque sentinel stream must stay unused.
            let status = unsafe {
                inference_cuda_exl3_prep(args, std::ptr::without_provenance_mut(16), &mut stage)
            };
            assert_eq!(
                status,
                super::super::invalid_argument().status,
                "case {index}"
            );
            assert_eq!(stage, STAGE_NONE, "case {index}");
        }
    }

    #[test]
    fn geometry_admits_only_the_format_rates_and_checked_tile_extents() {
        for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
            for bits in 0..=255 {
                let valid = matches!(bits, 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16)
                    || book == Codebook::Mul1 && matches!(bits, 3 | 5 | 7);
                let expected =
                    valid.then_some((128 * 256 / 256 * usize::from(bits) * 16, 128 * 256 * 2));
                assert_eq!(geometry(128, 256, bits, book), expected, "{book:?}/{bits}");
            }
        }
        for (k, n) in [
            (0, 128),
            (128, 0),
            (127, 128),
            (128, 129),
            (usize::MAX, 128),
            (128, usize::MAX),
            (1 << 22, 1 << 22),
        ] {
            assert!(geometry(k, n, 6, Codebook::Mul1).is_none(), "{k}/{n}");
        }
        assert!(geometry(1 << 22, 1024, 16, Codebook::Mul1).is_some());
    }
}
