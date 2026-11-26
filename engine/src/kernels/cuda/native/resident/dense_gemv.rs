//! Raw FFI to `inference_cuda_dense_gemv_launch` of `native/resident/dense_gemv.cu` (declared in
//! `native/resident/dense_gemv.h`): shared dense matvec for floating-point and packed GGUF weights.
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

mod decode_attention_group;
mod decode_control_pair;
pub(in crate::kernels::cuda) use decode_attention_group::try_attention_group;
pub(in crate::kernels::cuda) mod moe;
pub(in crate::kernels::cuda) use decode_control_pair::try_control_pair;

const FLOAT: usize = 4;
const HALF: usize = 2;
const GGUF_DTYPE_BASE: c_int = 256;

/// Element type of the weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseDtype {
    F16,
    Bf16,
    F32,
    GgufNvfp4,
    PackedGguf(GgufType),
}

impl DenseDtype {
    /// Types supported by the packed GGUF loader, including its half formats.
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

    pub(in crate::kernels::cuda) fn weight_alignment(self) -> usize {
        match self {
            Self::F32 => FLOAT,
            _ => HALF,
        }
    }

    fn code(self) -> c_int {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
            Self::GgufNvfp4 => 2,
            Self::F32 => 3,
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
    flags: c_int,
}

/// `INFERENCE_CUDA_DENSE_GEMV_DECODE_SPLIT` in dense_gemv.h.
const DECODE_SPLIT: c_int = 1;
const PREFILL_TENSOR_GEMM: c_int = 2;

/// The native flag bits of a call.
fn flags(call: &DenseGemvRaw<'_>) -> c_int {
    (if call.decode_split { DECODE_SPLIT } else { 0 })
        | (if call.prefill_tensor_gemm {
            PREFILL_TENSOR_GEMM
        } else {
            0
        })
}

const _: () = assert!(
    size_of::<DenseGemvArgsRaw>() == 56
        && offset_of!(DenseGemvArgsRaw, w) == 0
        && offset_of!(DenseGemvArgsRaw, x) == 8
        && offset_of!(DenseGemvArgsRaw, y) == 16
        && offset_of!(DenseGemvArgsRaw, m) == 24
        && offset_of!(DenseGemvArgsRaw, rows) == 28
        && offset_of!(DenseGemvArgsRaw, cols) == 32
        && offset_of!(DenseGemvArgsRaw, ld_x) == 36
        && offset_of!(DenseGemvArgsRaw, ld_y) == 40
        && offset_of!(DenseGemvArgsRaw, weight_dtype) == 44
        && offset_of!(DenseGemvArgsRaw, flags) == 48
);

unsafe extern "C" {
    #[link_name = "inference_cuda_dense_gemv_launch"]
    fn dense_gemv_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_gguf_q8_row_reuse_width_launch"]
    fn gguf_q8_row_reuse_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
        out_width: *mut c_int,
        prefer_four: c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_gguf_exact_expert_tile_launch"]
    fn gguf_exact_expert_tile_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
        out_tile: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_ragged_pair_launch"]
    fn inkling_ragged_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
        used: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_down_launch"]
    fn inkling_down_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_panel_launch"]
    fn inkling_panel_raw(
        args: *const DenseGemvArgsRaw,
        tmp: *mut c_void,
        bytes: usize,
        stream: *mut c_void,
        stage: *mut c_int,
        used: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_qkv_visit_launch"]
    fn inkling_qkv_raw(
        args: *const DenseGemvArgsRaw,
        count: c_int,
        stream: *mut c_void,
        stage: *mut c_int,
        used: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_packed_visit_launch"]
    fn inkling_packed_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
        variant: c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_controller_pair_launch"]
    fn inkling_heads_raw(
        a: *const DenseGemvArgsRaw,
        b: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
        used: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_inkling_iq2_visit_launch"]
    fn inkling_iq2_raw(
        args: *const DenseGemvArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_gguf_wmma_launch"]
    fn gguf_wmma_raw(
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
    /// The context's weight-visit choice (`DenseGroupPolicy::weight_visit`).
    pub(in crate::kernels::cuda) weight_visit: bool,
    pub(in crate::kernels::cuda) prefill_tensor_gemm: bool,
    /// The context's split-kernel choice (`DenseGroupPolicy::decode_split`).
    pub(in crate::kernels::cuda) decode_split: bool,
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
#[cfg(test)]
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
    projection_need(dims, dtype)
}

pub(in crate::kernels::cuda) fn typed_wmma_need(
    dims: GemvDims,
    dtype: DenseDtype,
) -> Option<DenseGemvBytes> {
    use GgufType::{Iq1M, Iq1S, Iq2S, Iq2Xs, Iq2Xxs, Q20};
    let quant = matches!(
        dtype,
        DenseDtype::GgufNvfp4 | DenseDtype::PackedGguf(Iq1S | Iq1M | Iq2Xxs | Iq2Xs | Iq2S | Q20)
    );
    let geometries = [dims.m, dims.rows, dims.cols, dims.ld_x, dims.ld_y];
    if !quant
        || geometries
            .into_iter()
            .any(|n| n == 0 || c_int::try_from(n).is_err())
        || dims.ld_x < dims.cols
        || dims.ld_y < dims.rows
    {
        return None;
    }
    projection_need(dims, dtype)
}

fn projection_need(dims: GemvDims, dtype: DenseDtype) -> Option<DenseGemvBytes> {
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
        None => dims.cols.checked_mul(dtype.weight_alignment())?,
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
fn check(call: &DenseGemvRaw<'_>, need: DenseGemvBytes) -> Result<(), DeviceFailure> {
    let sized = fits(&call.w, need.w, call.dtype.weight_alignment())
        && fits(&call.x, need.x, FLOAT)
        && fits(&call.y, need.y, FLOAT);
    let out = Span::of(&call.y, need.y);
    let apart = out.apart(&Span::of(&call.w, need.w)) && out.apart(&Span::of(&call.x, need.x));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the kernel on `stream`: `y[t * ld_y + r] = sum_c w[r][c] * x[t * ld_x + c]`.
pub(in crate::kernels::cuda) fn dense_gemv_launch(
    stream: &RawStream,
    call: DenseGemvRaw<'_>,
) -> Result<(), DeviceFailure> {
    let need = typed_gemv_need(call.dims, call.dtype).ok_or_else(invalid_argument)?;
    use GgufType::{Iq1S, Iq2S, Iq2Xs, Iq3Xxs, Q4K, Q5K, Q6K, Q80};
    let (m, n, k) = (call.dims.m, call.dims.rows, call.dims.cols);
    let attn = matches!(
        call.dtype,
        DenseDtype::PackedGguf(Q80 | Q4K | Q5K | Q6K | Iq3Xxs)
    ) && matches!(m, 6 | 7 | 8 | 63 | 64 | 127 | 128 | 255 | 256)
        && matches!((n, k), (384 | 1024 | 3072, 2304) | (2304, 3072))
        && !(n == 384 && m <= 8 && call.dtype != DenseDtype::PackedGguf(Q80));
    let expert = matches!(call.dtype, DenseDtype::PackedGguf(Iq2Xs | Iq1S | Iq2S))
        && (5..=15).contains(&m)
        && (n, k) == (2304, 2304);
    let down = call.dtype == DenseDtype::F32
        && (n, k) == (64, 4608)
        && matches!(m, 7 | 8 | 21 | 63 | 64 | 127 | 128 | 255 | 256);
    let entry = if down && call.weight_visit {
        ProjectionEntry::ControllerDown
    } else if (attn || expert) && call.weight_visit {
        ProjectionEntry::PackedVisit(3)
    } else {
        ProjectionEntry::Gemv
    };
    projection_launch(stream, call, need, entry).map(|_| ())
}

/// Same strided F32 extents and checks as ordinary GEMV. Native dispatch checks the active device.
/// Returns the row-reuse width, or zero for ordinary GEMV. Never retries a launch failure.
pub(in crate::kernels::cuda) fn gguf_q8_row_reuse_launch(
    stream: &RawStream,
    call: DenseGemvRaw<'_>,
    prefer_four: bool,
) -> Result<c_int, DeviceFailure> {
    let need = typed_gemv_need(call.dims, call.dtype).ok_or_else(invalid_argument)?;
    projection_launch(stream, call, need, ProjectionEntry::Q8RowReuse(prefer_four))
}

/// Same operands as dense GEMV. The native entry selects the exact tile or ordinary GEMV
/// before enqueueing, and reports whether it tiled. Do not retry a launch failure.
pub(in crate::kernels::cuda) fn gguf_exact_expert_tile_launch(
    stream: &RawStream,
    call: DenseGemvRaw<'_>,
) -> Result<bool, DeviceFailure> {
    let need = typed_gemv_need(call.dims, call.dtype).ok_or_else(invalid_argument)?;
    let dims = call.dims;
    let visit = match (call.dtype, dims.rows, dims.cols, dims.m) {
        (DenseDtype::PackedGguf(GgufType::Q6K), 3072, 2304, 8 | 63 | 64)
        | (DenseDtype::PackedGguf(GgufType::Iq2Xxs), 2304, 2304, 4..=16)
        | (DenseDtype::PackedGguf(GgufType::Q20), 2304, 1152, 4..=32) => call.weight_visit,
        _ => false,
    };
    let entry = if visit {
        ProjectionEntry::Q8RowReuse(false)
    } else {
        ProjectionEntry::ExactExpertTile
    };
    projection_launch(stream, call, need, entry).map(|tile| tile != 0)
}

pub(in crate::kernels::cuda) fn gguf_wmma_launch(
    stream: &RawStream,
    call: DenseGemvRaw<'_>,
) -> Result<(), DeviceFailure> {
    let need = typed_wmma_need(call.dims, call.dtype).ok_or_else(invalid_argument)?;
    let eligible = matches!(
        (call.dtype, call.dims.rows, call.dims.cols),
        (
            DenseDtype::PackedGguf(GgufType::Iq2Xs | GgufType::Iq1S | GgufType::Iq2S),
            2304,
            2304
        ) | (DenseDtype::GgufNvfp4, 2304, 1152)
    );
    let entry = if eligible
        && call.dims.m <= 512
        && call.dims.ld_x <= GEMV_MAX_LD
        && call.dims.ld_y <= GEMV_MAX_LD
        && call.weight_visit
    {
        ProjectionEntry::Iq2Visit
    } else {
        ProjectionEntry::Wmma
    };
    projection_launch(stream, call, need, entry).map(|_| ())
}

enum ProjectionEntry {
    Gemv,
    Wmma,
    Iq2Visit,
    PackedVisit(c_int),
    ControllerDown,
    ExactExpertTile,
    Q8RowReuse(bool),
}

fn projection_args(
    call: &DenseGemvRaw<'_>,
    need: DenseGemvBytes,
) -> Result<DenseGemvArgsRaw, DeviceFailure> {
    check(call, need)?;
    let dims = call.dims;
    let as_int = |v: usize| c_int::try_from(v).map_err(|_| invalid_argument());
    Ok(DenseGemvArgsRaw {
        w: call.w.address().cast_const(),
        x: call.x.address().cast_const(),
        y: call.y.address(),
        m: as_int(dims.m)?,
        rows: as_int(dims.rows)?,
        cols: as_int(dims.cols)?,
        ld_x: as_int(dims.ld_x)?,
        ld_y: as_int(dims.ld_y)?,
        weight_dtype: call.dtype.code(),
        flags: flags(call),
    })
}

pub(in crate::kernels::cuda) fn try_expert_panel(
    stream: &RawStream,
    call: &DenseGemvRaw<'_>,
    tmp: RawRange<'_>,
) -> Result<bool, DeviceFailure> {
    if !(33..=512).contains(&call.dims.m)
        || call.dims.rows != 2304
        || !matches!(
            (call.dtype, call.dims.cols),
            (DenseDtype::GgufNvfp4, 1152)
                | (
                    // IQ1_S direct WMMA with a CTA-local codebook beats two stages.
                    DenseDtype::PackedGguf(GgufType::Iq2Xs | GgufType::Iq2S),
                    2304
                )
        )
    {
        return Ok(false);
    }
    let bytes = call.dims.rows * call.dims.cols * 2;
    if !fits(&tmp, bytes, 16) || call.dims.ld_x > GEMV_MAX_LD || call.dims.ld_y > GEMV_MAX_LD {
        return Ok(false);
    }
    let need = typed_wmma_need(call.dims, call.dtype).ok_or_else(invalid_argument)?;
    let scratch = Span::of(&tmp, bytes);
    if !scratch.apart(&Span::of(&call.w, need.w))
        || !scratch.apart(&Span::of(&call.x, need.x))
        || !scratch.apart(&Span::of(&call.y, need.y))
    {
        return Ok(false);
    }
    let args = projection_args(call, need)?;
    let (mut stage, mut used) = (STAGE_NONE, 0);
    // SAFETY: operands and scratch are validated live ranges and disjoint. The
    // caller reserved this span exclusively until the stream-ordered stage ends.
    let status = unsafe {
        inkling_panel_raw(
            &raw const args,
            tmp.address(),
            bytes,
            stream.0.as_ptr(),
            &mut stage,
            &mut used,
        )
    };
    checked(status, stage)?;
    Ok(used != 0)
}

pub(in crate::kernels::cuda) fn try_ragged_pair(
    stream: &RawStream,
    a: &DenseGemvRaw<'_>,
    b: &DenseGemvRaw<'_>,
) -> Result<bool, DeviceFailure> {
    if a.dtype != b.dtype
        || a.dims.rows != 2304
        || a.dims.rows != b.dims.rows
        || a.dims.cols != b.dims.cols
    {
        return Ok(false);
    }
    let max = match (a.dtype, a.dims.cols) {
        (DenseDtype::PackedGguf(GgufType::Iq2Xxs), 2304) => 16,
        (DenseDtype::PackedGguf(GgufType::Q20), 1152) => 32,
        _ => return Ok(false),
    };
    let bin = |m| match m {
        4 => Some(0),
        5..=8 => Some(1),
        9.. if m <= max => Some(2),
        _ => None,
    };
    if bin(a.dims.m).is_none() || bin(a.dims.m) != bin(b.dims.m) {
        return Ok(false);
    }
    let raw = |c: &DenseGemvRaw<'_>| {
        projection_args(
            c,
            typed_gemv_need(c.dims, c.dtype).ok_or_else(invalid_argument)?,
        )
    };
    let args = [raw(a)?, raw(b)?];
    let (mut stage, mut used) = (STAGE_NONE, 0);
    // SAFETY: two validated, live single-call operand sets. The native entry
    // rechecks all five cross-call write hazards before enqueueing a joint grid.
    let status =
        unsafe { inkling_ragged_raw(args.as_ptr(), stream.0.as_ptr(), &mut stage, &mut used) };
    checked(status, stage)?;
    Ok(used != 0)
}

pub(in crate::kernels::cuda) fn try_qkv_pair(
    stream: &RawStream,
    a: &DenseGemvRaw<'_>,
    b: &DenseGemvRaw<'_>,
) -> Result<bool, DeviceFailure> {
    if a.dims.rows != 1024
        || b.dims.rows != 1024
        || a.dims.cols != 2304
        || b.dims.cols != 2304
        || !matches!(a.dims.m, 7 | 8)
        || b.dims.m != a.dims.m
        || a.dtype != DenseDtype::PackedGguf(GgufType::Q80)
        || a.dtype != b.dtype
    {
        return Ok(false);
    }
    let args = [
        projection_args(
            a,
            typed_gemv_need(a.dims, a.dtype).ok_or_else(invalid_argument)?,
        )?,
        projection_args(
            b,
            typed_gemv_need(b.dims, b.dtype).ok_or_else(invalid_argument)?,
        )?,
    ];
    let (mut stage, mut used) = (STAGE_NONE, 0);
    // SAFETY: projection_args proved both per-call ranges; the native admission
    // checks every cross-call output conflict before submitting a single kernel.
    let status =
        unsafe { inkling_qkv_raw(args.as_ptr(), 2, stream.0.as_ptr(), &mut stage, &mut used) };
    checked(status, stage)?;
    Ok(used != 0)
}

pub(in crate::kernels::cuda) fn try_controller_pair(
    stream: &RawStream,
    a: &DenseGemvRaw<'_>,
    b: &DenseGemvRaw<'_>,
) -> Result<bool, DeviceFailure> {
    if a.dtype != DenseDtype::F32
        || b.dtype != a.dtype
        || a.dims.cols != 64
        || a.dims.rows != 2304
        || b.dims != a.dims
        || a.x.address() != b.x.address()
    {
        return Ok(false);
    }
    let need = || typed_gemv_need(a.dims, a.dtype).ok_or_else(invalid_argument);
    let (a, b) = (projection_args(a, need()?)?, projection_args(b, need()?)?);
    let (mut stage, mut used) = (STAGE_NONE, 0);
    // SAFETY: projection_args validated both live ranges. Native checks cross-call
    // overlap, strides and alignment before treating the two calls as independent.
    let status = unsafe {
        inkling_heads_raw(
            &raw const a,
            &raw const b,
            stream.0.as_ptr(),
            &mut stage,
            &mut used,
        )
    };
    checked(status, stage)?;
    Ok(used != 0)
}

fn projection_launch(
    stream: &RawStream,
    call: DenseGemvRaw<'_>,
    need: DenseGemvBytes,
    entry: ProjectionEntry,
) -> Result<c_int, DeviceFailure> {
    let args = projection_args(&call, need)?;
    let mut stage = STAGE_NONE;
    let mut selected = 0;
    let stream = stream.0.as_ptr();
    // SAFETY: `check` proved every range inside its live allocation (`RawRange::new`), large
    // enough for the call, aligned (the weight to 2, x and y to 4), and the written `y` apart from
    // `w` and `x`; the entry checks the addresses again. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads only
    // during the call, `stream` is live, and `stage` and `selected` are writable c_ints.
    let status = unsafe {
        match entry {
            ProjectionEntry::Gemv => dense_gemv_raw(&raw const args, stream, &mut stage),
            ProjectionEntry::ControllerDown => {
                inkling_down_raw(&raw const args, stream, &mut stage)
            }
            ProjectionEntry::PackedVisit(variant) => {
                inkling_packed_raw(&raw const args, stream, &mut stage, variant)
            }
            ProjectionEntry::Iq2Visit => inkling_iq2_raw(&raw const args, stream, &mut stage),
            ProjectionEntry::Wmma => gguf_wmma_raw(&raw const args, stream, &mut stage),
            ProjectionEntry::ExactExpertTile => {
                gguf_exact_expert_tile_raw(&raw const args, stream, &mut stage, &mut selected)
            }
            ProjectionEntry::Q8RowReuse(four) => gguf_q8_row_reuse_raw(
                &raw const args,
                stream,
                &mut stage,
                &mut selected,
                if call.weight_visit { -1 } else { four.into() },
            ),
        }
    };
    checked(status, stage)?;
    Ok(selected)
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
    fn wmma_admits_wider_batches_and_checks_packed_geometry() {
        let m = GEMV_MAX_M + 1;
        let q2 = DenseDtype::PackedGguf(GgufType::Q20);
        let allowed = typed_wmma_need(dims(m, 7, 128, 135, 11), q2).unwrap();
        assert_eq!(allowed.w, 7 * 2 * 18);
        assert_eq!(allowed.x, ((m - 1) * 135 + 128) * 4);
        assert_eq!(allowed.y, ((m - 1) * 11 + 7) * 4);
        assert!(typed_wmma_need(dims(32, 7, 129, 135, 11), q2).is_none());
        assert!(typed_wmma_need(dims(32, 7, 128, 127, 11), q2).is_none());
        assert!(typed_wmma_need(dims(32, 7, 128, 135, 11), DenseDtype::F32).is_none());
        assert!(typed_wmma_need(dims(i32::MAX as usize + 1, 7, 128, 135, 11), q2).is_none());
    }

    #[test]
    fn exact_expert_tile_uses_baseline_gemv_extents_including_fallbacks() {
        let iq2 = DenseDtype::PackedGguf(GgufType::Iq2Xxs);
        let q2 = DenseDtype::PackedGguf(GgufType::Q20);
        let gate = typed_gemv_need(dims(5, 2304, 2304, 2312, 2316), iq2).unwrap();
        assert_eq!(gate.w, 2304 * 9 * 66);
        assert_eq!(gate.x, (4 * 2312 + 2304) * 4);
        assert_eq!(gate.y, (4 * 2316 + 2304) * 4);
        let down = typed_gemv_need(dims(4, 2304, 1152, 1152, 2304), q2).unwrap();
        assert_eq!(down.w, 2304 * 18 * 18);
        assert_eq!(down.x, 4 * 1152 * 4);
        assert_eq!(down.y, 4 * 2304 * 4);
        // Singleton, oversize tile and other codec are valid ordinary GEMVs.
        assert!(typed_gemv_need(dims(1, 2304, 2304, 2304, 2304), iq2).is_some());
        assert!(typed_gemv_need(dims(513, 2305, 2560, 2560, 2305), iq2).is_some());
        assert!(
            typed_gemv_need(
                dims(2, 9, 256, 256, 9),
                DenseDtype::PackedGguf(GgufType::Q80)
            )
            .is_some()
        );
        assert!(typed_gemv_need(dims(2, 2304, 2303, 2304, 2304), iq2).is_none());
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
