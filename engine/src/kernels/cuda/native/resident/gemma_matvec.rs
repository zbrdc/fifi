use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_gemma_matvec_launch"]
    fn gemma_matvec_launch_raw(
        args: *const GemmaMatvecArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_gemma_matvec_args (gemma_matvec.h), field for field.
#[repr(C)]
struct GemmaMatvecArgsRepr {
    w_qs: *const c_void,
    w_scales: *const c_void,
    x_q8: *const c_void,
    y: *mut c_void,
    rows: c_int,
    k: c_int,
    ncols: c_int,
}

const _: () = assert!(
    size_of::<GemmaMatvecArgsRepr>() == 48
        && offset_of!(GemmaMatvecArgsRepr, w_qs) == 0
        && offset_of!(GemmaMatvecArgsRepr, w_scales) == 8
        && offset_of!(GemmaMatvecArgsRepr, x_q8) == 16
        && offset_of!(GemmaMatvecArgsRepr, y) == 24
        && offset_of!(GemmaMatvecArgsRepr, rows) == 32
        && offset_of!(GemmaMatvecArgsRepr, k) == 36
        && offset_of!(GemmaMatvecArgsRepr, ncols) == 40
);

/// What one gemma matvec launch reads and writes.
pub(in crate::kernels::cuda) struct GemmaMatvecRaw<'a> {
    pub(in crate::kernels::cuda) w_qs: RawRange<'a>,
    pub(in crate::kernels::cuda) w_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

/// Proves a gemma matvec launch before the native call.
fn check_gemma_matvec(call: &GemmaMatvecRaw<'_>) -> Result<(), DeviceFailure> {
    if !(1..=(1usize << 20)).contains(&call.rows)
        || !(32..=65_536).contains(&call.k)
        || !call.k.is_multiple_of(32)
        || !(1..=8).contains(&call.ncols)
    {
        return Err(invalid_argument());
    }
    let blocks = call.k / 32;
    let qs = call.rows.checked_mul(call.k).ok_or_else(invalid_argument)?;
    let scales = call
        .rows
        .checked_mul(blocks)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(invalid_argument)?;
    let y_bytes = call
        .rows
        .checked_mul(call.ncols)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let stride = call.k.next_multiple_of(512) / 32;
    let act = ((call.ncols - 1) * stride + blocks) * 36;
    let sized = call.w_qs.bytes >= qs
        && call.w_scales.bytes >= scales
        && call.x_q8.bytes >= act
        && call.y.bytes >= y_bytes;
    let aligned = call.w_qs.offset.is_multiple_of(16)
        && call.w_scales.offset.is_multiple_of(2)
        && call.x_q8.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let written = Span::of(&call.y, y_bytes);
    let apart = written.apart(&Span::of(&call.w_qs, qs))
        && written.apart(&Span::of(&call.w_scales, scales))
        && written.apart(&Span::of(&call.x_q8, act));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the gemma matvec of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn gemma_matvec_launch(
    stream: &RawStream,
    call: &GemmaMatvecRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gemma_matvec(call)?;
    let args = GemmaMatvecArgsRepr {
        w_qs: call.w_qs.address().cast_const(),
        w_scales: call.w_scales.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        y: call.y.address(),
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_gemma_matvec proved every plane inside its live allocation, large enough,
    // aligned, and apart from y; the adapter repeats the checks on the addresses. The
    // allocations stay valid until the queued launch finishes because inference_cuda_free
    // synchronizes the device first. stream is live for the call and args is a plain value.
    let status = unsafe { gemma_matvec_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::RawDevice;
    use super::*;

    #[test]
    fn checks_batched_extents_and_column_limits() {
        let device = RawDevice::unbacked(1 << 20);
        let range = |offset, bytes| RawRange::new(&device, offset, bytes).expect("range");
        let mut call = GemmaMatvecRaw {
            w_qs: range(0, 4 * 64),
            w_scales: range(4096, 4 * 2 * 2),
            x_q8: range(8192, (7 * 16 + 2) * 36),
            y: range(16384, 8 * 4 * 4),
            rows: 4,
            k: 64,
            ncols: 8,
        };
        assert_eq!(check_gemma_matvec(&call), Ok(()));
        for ncols in [0, 9, usize::MAX] {
            call.ncols = ncols;
            assert!(check_gemma_matvec(&call).is_err(), "ncols {ncols}");
        }
        call.ncols = 8;
        call.x_q8 = range(8192, 8 * 2 * 36);
        assert!(
            check_gemma_matvec(&call).is_err(),
            "unpadded activation rows"
        );
        call.x_q8 = range(8192, (7 * 16 + 2) * 36);
        call.y = range(16384, 8 * 4 * 4 - 4);
        assert!(check_gemma_matvec(&call).is_err(), "last output missing");
        call.y = range(8192 + 7 * 16 * 36, 8 * 4 * 4);
        assert!(
            check_gemma_matvec(&call).is_err(),
            "last activation overlaps output"
        );
    }
}
