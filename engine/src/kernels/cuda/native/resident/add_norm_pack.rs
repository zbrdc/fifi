//! Raw FFI to the residual add, RMSNorm and fp16 pack entry of
//! `native/resident/add_norm_pack.cu` (declared in `native/resident/add_norm_pack.h`).
//! Child module of `native/resident.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard. It only enqueues and returns.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes (through
//!   `super::pack_bytes`), alignment and disjointness (`super::Span`) are proved again here, on
//!   the extents the kernel touches, before the call; the C entry repeats the alignment and shape
//!   checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};
use std::ptr::{null, null_mut};

use super::{
    DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument, pack_bytes,
};

/// Mirror of `inference_cuda_add_norm_pack_args` in `add_norm_pack.h`, field for field.
#[repr(C)]
struct AddNormPackArgsRaw {
    h: *mut c_void,
    delta: *const c_void,
    w: *const c_void,
    xf: *mut c_void,
    norm_out: *mut c_void,
    m: c_int,
    n: c_int,
    eps: f32,
}

const _: () = assert!(
    size_of::<AddNormPackArgsRaw>() == 56
        && offset_of!(AddNormPackArgsRaw, h) == 0
        && offset_of!(AddNormPackArgsRaw, delta) == 8
        && offset_of!(AddNormPackArgsRaw, w) == 16
        && offset_of!(AddNormPackArgsRaw, xf) == 24
        && offset_of!(AddNormPackArgsRaw, norm_out) == 32
        && offset_of!(AddNormPackArgsRaw, m) == 40
        && offset_of!(AddNormPackArgsRaw, n) == 44
        && offset_of!(AddNormPackArgsRaw, eps) == 48
);

unsafe extern "C" {
    #[link_name = "inference_cuda_add_norm_pack_launch"]
    fn add_norm_pack_launch_raw(
        args: *const AddNormPackArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The two integer arguments of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AddNormInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n: c_int,
}

/// What one fused add, norm and pack reads and writes: the converted ranges, the integers and
/// `eps`. `delta` and `norm_out` are optional.
pub(in crate::kernels::cuda) struct AddNormRaw<'a> {
    pub(in crate::kernels::cuda) h: RawRange<'a>,
    pub(in crate::kernels::cuda) weight: RawRange<'a>,
    pub(in crate::kernels::cuda) xf: RawRange<'a>,
    pub(in crate::kernels::cuda) delta: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) norm_out: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) ints: AddNormInts,
    pub(in crate::kernels::cuda) eps: f32,
}

/// The bytes each range must hold.
struct Needs {
    row: usize,
    weight: usize,
    xf: usize,
}

/// `n` as `usize`, or an invalid-argument failure for a negative value.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

fn needs(ints: &AddNormInts) -> Result<Needs, DeviceFailure> {
    let (m, n) = (size(ints.m)?, size(ints.n)?);
    let (row, xf) = pack_bytes(m, n).ok_or_else(invalid_argument)?;
    Ok(Needs {
        row,
        weight: n * size_of::<f32>(),
        xf,
    })
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// Sizes and alignment of the ranges: `xf` and `norm_out` 16, the rest 4.
fn check_fit(io: &AddNormRaw<'_>, need: &Needs) -> bool {
    fits(&io.h, need.row, 4)
        && fits(&io.weight, need.weight, 4)
        && fits(&io.xf, need.xf, 16)
        && io.delta.is_none_or(|delta| fits(&delta, need.row, 4))
        && io.norm_out.is_none_or(|out| fits(&out, need.row, 16))
}

/// Proves a launch before the native call: the shape integers, sizes and alignment of every range
/// and that the extents the kernel writes (`h`, `xf`, `norm_out`) share no byte with each other
/// or with the ones it only reads (`delta`, `weight`). `delta` and `weight` may overlap.
fn check_add_norm(io: &AddNormRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints)?;
    if !check_fit(io, &need) {
        return Err(invalid_argument());
    }
    let mut written = vec![Span::of(&io.h, need.row), Span::of(&io.xf, need.xf)];
    written.extend(io.norm_out.map(|out| Span::of(&out, need.row)));
    let mut read = vec![Span::of(&io.weight, need.weight)];
    read.extend(io.delta.map(|delta| Span::of(&delta, need.row)));
    for (i, a) in written.iter().enumerate() {
        let apart_from_written = written[i + 1..].iter().all(|b| a.apart(b));
        if !(apart_from_written && read.iter().all(|b| a.apart(b))) {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Queues the fused add, RMSNorm and fp16 pack on `stream`: `h += delta` in place when `delta` is
/// given, the f32 norm row into `norm_out` when given, and the packed fp16 row into `xf`. Returns
/// when it is queued.
pub(in crate::kernels::cuda) fn add_norm_pack_launch(
    stream: &RawStream,
    call: AddNormRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_add_norm(&call)?;
    let args = AddNormPackArgsRaw {
        h: call.h.address(),
        delta: call.delta.map_or(null(), |r| r.address().cast_const()),
        w: call.weight.address().cast_const(),
        xf: call.xf.address(),
        norm_out: call.norm_out.map_or(null_mut(), |r| r.address()),
        m: call.ints.m,
        n: call.ints.n,
        eps: call.eps,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) `check_add_norm` proved every range inside its live allocation and large enough
    // for the call; the allocations outlive the queued work because `inference_cuda_free`
    // synchronizes the device before `cudaFree`. (b) It also proved the alignment (16 for `xf` and
    // `norm_out`, 4 for the rest) and that the extents of `h`, `xf` and `norm_out` share no byte
    // with any other range, so the kernel's writes cannot alias its reads; the C entry repeats the
    // checks on the addresses. (c) `stream` is borrowed from a live `RawStream`. (d) `args` is a
    // live local that the entry reads only during the call and keeps no pointer to. (e) The
    // dispatch guard is held by `launch_on`, which is the only caller.
    let status =
        unsafe { add_norm_pack_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use std::ptr::without_provenance_mut;

    use super::*;
    use crate::kernels::cuda::native::STATUS_INVALID_ARGUMENT;

    /// A made-up device address `gib` GiB up; every address in these tests is 16-byte aligned and
    /// far from the others.
    fn at(gib: usize) -> *mut c_void {
        without_provenance_mut(gib << 30)
    }

    /// A valid argument set. The cases below change one field of it. The entry refuses before it
    /// reaches a CUDA call, so no address is read; the base itself is never launched.
    fn base() -> AddNormPackArgsRaw {
        AddNormPackArgsRaw {
            h: at(1),
            delta: at(2).cast_const(),
            w: at(3).cast_const(),
            xf: at(4),
            norm_out: at(5),
            m: 1,
            n: 256,
            eps: 1e-6,
        }
    }

    /// The status and stage that the raw entry reports for `args` and the stream value `stream`.
    fn answer(args: &AddNormPackArgsRaw, stream: *mut c_void) -> (c_int, c_int) {
        let mut stage = -1;
        // SAFETY: `args` is a live local. Every case is refused before a CUDA call, so neither
        // the made-up addresses nor the made-up stream value are dereferenced.
        let status = unsafe { add_norm_pack_launch_raw(&raw const *args, stream, &mut stage) };
        (status, stage)
    }

    /// One edit of the valid base.
    type Edit = fn(&mut AddNormPackArgsRaw);

    /// Host only: each argument set is one edit away from a valid one and the entry answers
    /// `INVALID_ARGUMENT` with the stage `NONE`, before any CUDA call.
    #[test]
    fn the_native_add_norm_entry_refuses_bad_arguments_before_any_device_call() {
        let cases: [(&str, Edit); 15] = [
            ("m 0", |a| a.m = 0),
            ("m 32769", |a| a.m = 32_769),
            ("n 255", |a| a.n = 255),
            ("n 257", |a| a.n = 257),
            ("n 33024", |a| a.n = 33_024),
            ("eps 0", |a| a.eps = 0.0),
            ("eps NaN", |a| a.eps = f32::NAN),
            ("eps infinity", |a| a.eps = f32::INFINITY),
            ("eps negative", |a| a.eps = -1e-6),
            ("norm_out at +4", |a| {
                a.norm_out = a.norm_out.wrapping_byte_add(4)
            }),
            ("xf at +8", |a| a.xf = a.xf.wrapping_byte_add(8)),
            ("h at +2", |a| a.h = a.h.wrapping_byte_add(2)),
            ("xf equal to h", |a| a.xf = a.h),
            ("delta equal to h", |a| a.delta = a.h.cast_const()),
            ("w inside xf", |a| {
                a.w = a.xf.wrapping_byte_add(64).cast_const()
            }),
        ];
        let stream = without_provenance_mut::<c_void>(16);
        let want = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        for (name, edit) in cases {
            let mut args = base();
            edit(&mut args);
            assert_eq!(answer(&args, stream), want, "{name}");
        }
        assert_eq!(answer(&base(), null_mut()), want, "a NULL stream");
    }
}
