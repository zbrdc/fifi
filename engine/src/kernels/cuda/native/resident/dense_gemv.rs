//! Raw FFI to `inference_cuda_dense_gemv_launch` of `native/resident/dense_gemv.cu` (declared in
//! `native/resident/dense_gemv.h`): shared dense matvec for half-precision and packed GGUF weights.
//! Child of `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entry is called only through `CudaContext::launch_on`, on the owning thread and under
//!   the dispatch guard. It only enqueues and returns the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here on the extents the kernel
//!   touches; the C entry repeats the checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The argument struct is a live local that the entry reads only during the call. Its `repr(C)`
//!   mirror is private here and pinned by `const` asserts to the field order of `dense_gemv.h`.
//!   No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use crate::weights::GgufType;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Limits of `dense_gemv.h`.
pub(in crate::kernels::cuda) const GEMV_MAX_M: usize = 32768;
pub(in crate::kernels::cuda) const GEMV_MAX_ROWS: usize = 1 << 20;
pub(in crate::kernels::cuda) const GEMV_MAX_COLS: usize = 12288;
pub(in crate::kernels::cuda) const GEMV_MAX_LD: usize = 1 << 20;

pub(in crate::kernels::cuda) mod moe;

const FLOAT: usize = 4;
const HALF: usize = 2;
const GGUF_DTYPE_BASE: c_int = 256;

/// Element type of the weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseDtype {
    F16,
    Bf16,
    GgufNvfp4,
    PackedGguf(GgufType),
}

impl DenseDtype {
    pub(crate) fn from_gguf(kind: GgufType) -> Option<Self> {
        use GgufType::*;
        match kind {
            F16 => Some(Self::F16),
            Bf16 => Some(Self::Bf16),
            Nvfp4 => Some(Self::GgufNvfp4),
            Q40 | Q50 | Q80 | Q20 | Q2K | Q3K | Q4K | Q5K | Q6K | Iq1S | Iq1M | Iq2Xxs | Iq2Xs
            | Iq2S | Iq3Xxs | Iq3S | Iq4Nl | Iq4Xs | Mxfp4 => Some(Self::PackedGguf(kind)),
            _ => None,
        }
    }

    pub(crate) fn is_packed_gguf(self) -> bool {
        self.gguf_kind().is_some()
    }

    fn gguf_kind(self) -> Option<GgufType> {
        match self {
            Self::GgufNvfp4 => Some(GgufType::Nvfp4),
            Self::PackedGguf(kind) => Some(kind),
            _ => None,
        }
    }

    fn code(self) -> c_int {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
            Self::GgufNvfp4 => 2,
            Self::PackedGguf(kind) => GGUF_DTYPE_BASE + kind.id() as c_int,
        }
    }
}

/// Mirror of `inference_cuda_dense_gemv_args`.
#[repr(C)]
struct DenseGemvArgsRaw {
    w: *const c_void,
    x: *const c_void,
    y: *mut c_void,
    m: c_int,
    rows: c_int,
    cols: c_int,
    ld_x: c_int,
    ld_y: c_int,
    weight_dtype: c_int,
}

const _: () = assert!(
    size_of::<DenseGemvArgsRaw>() == 48
        && offset_of!(DenseGemvArgsRaw, w) == 0
        && offset_of!(DenseGemvArgsRaw, x) == 8
        && offset_of!(DenseGemvArgsRaw, y) == 16
        && offset_of!(DenseGemvArgsRaw, m) == 24
        && offset_of!(DenseGemvArgsRaw, rows) == 28
        && offset_of!(DenseGemvArgsRaw, cols) == 32
        && offset_of!(DenseGemvArgsRaw, ld_x) == 36
        && offset_of!(DenseGemvArgsRaw, ld_y) == 40
        && offset_of!(DenseGemvArgsRaw, weight_dtype) == 44
);

unsafe extern "C" {
    #[link_name = "inference_cuda_dense_gemv_launch"]
    fn dense_gemv_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The geometry of one launch: `m` activation rows of `cols` values at stride `ld_x`, a weight
/// of `rows` rows of `cols` values, and `rows` results per activation row at stride `ld_y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct GemvDims {
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) cols: usize,
    pub(in crate::kernels::cuda) ld_x: usize,
    pub(in crate::kernels::cuda) ld_y: usize,
}

/// What one launch touches; see `dense_gemv.h` for the layouts.
pub(in crate::kernels::cuda) struct DenseGemvRaw<'a> {
    pub(in crate::kernels::cuda) dims: GemvDims,
    pub(in crate::kernels::cuda) dtype: DenseDtype,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
}

/// Bytes each range of a launch must hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseGemvBytes {
    pub(crate) w: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
}

/// `None` outside the limits of `dense_gemv.h` or when a size overflows. The last activation row
/// and the last result row hold only `cols` and `rows` values: the strides pad between rows.
pub(in crate::kernels::cuda) fn gemv_need(dims: GemvDims) -> Option<DenseGemvBytes> {
    typed_gemv_need(dims, DenseDtype::F16)
}

pub(in crate::kernels::cuda) fn typed_gemv_need(
    dims: GemvDims,
    dtype: DenseDtype,
) -> Option<DenseGemvBytes> {
    let inside = (1..=GEMV_MAX_M).contains(&dims.m)
        && (1..=GEMV_MAX_ROWS).contains(&dims.rows)
        && (1..=GEMV_MAX_COLS).contains(&dims.cols)
        && (dims.cols..=GEMV_MAX_LD).contains(&dims.ld_x)
        && (dims.rows..=GEMV_MAX_LD).contains(&dims.ld_y);
    if !inside {
        return None;
    }
    let strided = |last: usize, stride: usize, tail: usize| {
        (last - 1)
            .checked_mul(stride)?
            .checked_add(tail)?
            .checked_mul(FLOAT)
    };
    let row_bytes = match dtype.gguf_kind() {
        Some(kind) => {
            let block = kind.block_elems() as usize;
            if DenseDtype::from_gguf(kind) != Some(dtype) || !dims.cols.is_multiple_of(block) {
                return None;
            }
            (dims.cols / block).checked_mul(kind.block_bytes() as usize)?
        }
        None => dims.cols.checked_mul(HALF)?,
    };
    Some(DenseGemvBytes {
        w: dims.rows.checked_mul(row_bytes)?,
        x: strided(dims.m, dims.ld_x, dims.cols)?,
        y: strided(dims.m, dims.ld_y, dims.rows)?,
    })
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes() >= need && range.offset().is_multiple_of(align)
}

/// The result is apart from both inputs; the inputs may share bytes.
fn check(call: &DenseGemvRaw<'_>) -> Result<DenseGemvBytes, DeviceFailure> {
    let need = typed_gemv_need(call.dims, call.dtype).ok_or_else(invalid_argument)?;
    let sized =
        fits(&call.w, need.w, HALF) && fits(&call.x, need.x, FLOAT) && fits(&call.y, need.y, FLOAT);
    let out = Span::of(&call.y, need.y);
    let apart = out.apart(&Span::of(&call.w, need.w)) && out.apart(&Span::of(&call.x, need.x));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    Ok(need)
}

/// Queues the kernel on `stream`: `y[t * ld_y + r] = sum_c w[r][c] * x[t * ld_x + c]`.
pub(in crate::kernels::cuda) fn dense_gemv_launch(
    stream: &RawStream,
    call: DenseGemvRaw<'_>,
) -> Result<(), DeviceFailure> {
    check(&call)?;
    let dims = call.dims;
    let as_int = |v: usize| c_int::try_from(v).map_err(|_| invalid_argument());
    let args = DenseGemvArgsRaw {
        w: call.w.address().cast_const(),
        x: call.x.address().cast_const(),
        y: call.y.address(),
        m: as_int(dims.m)?,
        rows: as_int(dims.rows)?,
        cols: as_int(dims.cols)?,
        ld_x: as_int(dims.ld_x)?,
        ld_y: as_int(dims.ld_y)?,
        weight_dtype: call.dtype.code(),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check` proved every range inside its live allocation (`RawRange::new`), large
    // enough for the call, aligned (the weight to 2, x and y to 4), and the written `y` apart from
    // `w` and `x`; the entry checks the addresses again. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads only
    // during the call, `stream` is live, and `stage` is a writable c_int.
    let status = unsafe { dense_gemv_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(m: usize, rows: usize, cols: usize, ld_x: usize, ld_y: usize) -> GemvDims {
        GemvDims {
            m,
            rows,
            cols,
            ld_x,
            ld_y,
        }
    }

    #[test]
    fn the_last_row_of_a_strided_operand_holds_only_its_values() {
        // A destination row wider than the results: 21,632 values at stride 21,728, results at its tail.
        let need = gemv_need(dims(3, 48, 2560, 2560, 21_728)).unwrap();
        assert_eq!(need.w, 48 * 2560 * 2);
        assert_eq!(need.x, 3 * 2560 * 4);
        assert_eq!(need.y, (2 * 21_728 + 48) * 4);
        let single = gemv_need(dims(1, 48, 2560, 2560, 48)).unwrap();
        assert_eq!((single.x, single.y), (2560 * 4, 48 * 4));
    }

    #[test]
    fn dimensions_outside_the_header_limits_are_refused() {
        assert!(gemv_need(dims(0, 48, 2560, 2560, 48)).is_none());
        assert!(gemv_need(dims(GEMV_MAX_M + 1, 48, 2560, 2560, 48)).is_none());
        assert!(gemv_need(dims(1, 0, 2560, 2560, 48)).is_none());
        assert!(gemv_need(dims(1, GEMV_MAX_ROWS + 1, 8, 8, GEMV_MAX_ROWS + 1)).is_none());
        assert!(gemv_need(dims(1, 48, GEMV_MAX_COLS + 1, GEMV_MAX_COLS + 1, 48)).is_none());
        // A stride below the row width would overlap rows.
        assert!(gemv_need(dims(2, 48, 2560, 2559, 48)).is_none());
        assert!(gemv_need(dims(2, 48, 2560, 2560, 47)).is_none());
        assert!(gemv_need(dims(2, 48, 2560, GEMV_MAX_LD + 1, 48)).is_none());
        assert!(gemv_need(dims(1, 48, GEMV_MAX_COLS, GEMV_MAX_COLS, 48)).is_some());
    }
}
