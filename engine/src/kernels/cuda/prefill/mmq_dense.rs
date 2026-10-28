//! The dense MMQ GEMM wrapper (CPF-V4 G2, decision 0049 item 11): [`CudaContext::mmq_dense`]
//! queues one launch of `native/prefill/mmq_dense.cu`, which calls the vendored
//! `mul_mat_q_case<Q8_0|Q4_K|Q6_K>` (mmq.cuh) on a Native weight and the q8_1 columns of
//! `mmq_quant`, and writes `dst[j * ld + col_off + i]` for `i < n`, `j < m`. It allocates
//! nothing, copies nothing and never synchronizes. The shape rules below are mirrored by the
//! C entry so a native call can never be refused as an invalid argument, which would latch
//! the backend; the byte sizes are the wrapper's alone, because the C ABI carries no
//! capacities (it adds the alignment, overlap and fixup-size checks that need none, or read
//! the device). The stream-k fixup range is scratch
//! that a launch writes and reads back: two launches on different streams must not share
//! one, and the wrapper requires the device maximum ([`CudaContext::mmq_fixup_bytes`]).
//!
//! Every refusal is a `BadRequest` tagged `mmq_dense: <rule>`, in this order: the weight type
//! (Q5_0 has no dense case in this packet), the K grid (`K % 256`, plan section B.3 rule M2,
//! before any other k rule so K = 384 reaches this tag for every type; it also implies
//! `K % qk` for the three cases, so no separate qk rule exists), the k extent, the column
//! count `M` (the 2,048 chunk cap, G1's `NCOLS_LIMIT`), the row count, the destination
//! stride, the `2^30` tile-grid limit of the vendored launch (mmq.cuh:1402), bounded with the
//! smallest tile width J = 8 because the launch picks J, the conversions to the C ABI's
//! `int`, and the destination reach (`dst reach`: `(m - 1) * ld + n` must not exceed
//! `i32::MAX`, because the kernels index the destination in 32-bit `int`). The byte sizes are
//! G1's: the q8_1 range `q8_1_bytes(K, M)` carries the one constant pad.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no caller yet: the chunk driver does not call this kernel; drop the expect when it does"
    )
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::mmq_dense::{MmqDenseRaw, mmq_dense_launch};
use crate::kernels::cuda::prefill::mmq_quant::MmqQuantizeCall;
use crate::kernels::cuda::prefill::mmq_size::{MmqType, NCOLS_LIMIT, q8_1_bytes};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// The MMQ k grid: the k loop reads 256 values per iteration with no k clamp (mmq.cuh:856).
const K_GRID: usize = 256;
/// The k extent bound (the tree's `MAX_DIM`).
const K_MAX: usize = 32_768;
/// The smallest tile width J of any table row, and the `I` tile of every kept row.
const TILE_J_MIN: usize = 8;
const TILE_I: usize = 128;
/// The vendored `variable kbc must not overflow` bound on `ntiles_dst * blocks_per_ne00`.
const GRID_LIMIT: usize = 1 << 30;

/// The converted integers of one dense call, `c_int` for the FFI mirror; the rule checks
/// build them, so no conversion can fail inside a launch closure.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseInts {
    pub(crate) ty: c_int,
    pub(crate) k: c_int,
    pub(crate) n: c_int,
    pub(crate) m: c_int,
    pub(crate) ld: c_int,
    pub(crate) col_off: c_int,
}

/// The bytes of the three ranges a dense call uses: the weight blocks, the q8_1 columns
/// (with the pad) and the destination reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseNeeds {
    pub(crate) w: usize,
    pub(crate) y: usize,
    pub(crate) dst: usize,
}

/// A `BadRequest` that names the rule that fired (the `mmq_dense` tag).
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("mmq_dense: {tag}: {detail}"))
}

/// `value` as `c_int` for the rule `tag`, or the "exceeds i32" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the scalar rules of a dense call in the module doc's order and converts the
/// integers once. `ty` is the weight's own recorded type, `n` its row count.
pub(crate) fn check_dense_rules(
    ty: QuantType,
    k: usize,
    n: usize,
    m: usize,
    ld: usize,
    col_off: usize,
) -> Result<DenseInts, EngError> {
    check_dense_mmq_rules(mmq_type_of(ty)?, k, n, m, ld, col_off)
}

pub(super) fn check_dense_mmq_rules(
    ty: MmqType,
    k: usize,
    n: usize,
    m: usize,
    ld: usize,
    col_off: usize,
) -> Result<DenseInts, EngError> {
    if ty == MmqType::Q50 {
        return Err(rule("type", "Q5_0 has no dense case in this packet"));
    }
    if !k.is_multiple_of(K_GRID) {
        return Err(rule(
            "K-grid",
            &format!("k is {k}, not a multiple of {K_GRID} (the MMQ k grid)"),
        ));
    }
    if k == 0 || k > K_MAX {
        return Err(rule(
            "K extent",
            &format!("k is {k}, not {K_GRID}..={K_MAX}"),
        ));
    }
    if m == 0 || m > NCOLS_LIMIT {
        return Err(rule(
            "M",
            &format!("m is {m}, outside 1..={NCOLS_LIMIT} (the chunk cap)"),
        ));
    }
    if n == 0 {
        return Err(rule("N", "n is 0"));
    }
    if col_off.checked_add(n).is_none_or(|reach| ld < reach) {
        return Err(rule(
            "ld",
            &format!("ld is {ld}, below col_off {col_off} + n {n}"),
        ));
    }
    let qk = match ty {
        MmqType::Q80 | MmqType::Q50 => 32,
        MmqType::Q4K | MmqType::Q6K => 256,
        MmqType::Nvfp4 => 64,
    };
    check_grid(qk, k, n, m)?;
    let ints = DenseInts {
        ty: super::mmq_quant::ggml_of(ty),
        k: to_int("k", k)?,
        n: to_int("n", n)?,
        m: to_int("m", m)?,
        ld: to_int("ld", ld)?,
        col_off: to_int("col_off", col_off)?,
    };
    check_dst_reach(n, m, ld)?;
    Ok(ints)
}

/// The vendored kernels index the destination in 32-bit `int` (`offset_dst` at mmq.cuh:953,
/// :1041, :1130 and :1273, `stride_col_dst` at :907), so the last element, at
/// `(m - 1) * ld + n - 1`, must stay below `2^31`: a larger reach wraps the offset and
/// stores before the destination. Qwen's largest is 2,047 * 8,192 = 16.8 M.
fn check_dst_reach(n: usize, m: usize, ld: usize) -> Result<(), EngError> {
    let reach = m
        .checked_sub(1)
        .and_then(|last| last.checked_mul(ld))
        .and_then(|lead| lead.checked_add(n));
    match reach {
        Some(reach) if reach <= i32::MAX as usize => Ok(()),
        _ => Err(rule(
            "dst reach",
            &format!("(m - 1) * ld + n with m {m}, ld {ld}, n {n} exceeds i32::MAX"),
        )),
    }
}

/// The `2^30` grid limit, mirrored: `ceil(m/8) * ceil(n/128) * (k/qk)` must stay below it.
fn check_grid(qk: usize, k: usize, n: usize, m: usize) -> Result<(), EngError> {
    let grid = m
        .div_ceil(TILE_J_MIN)
        .checked_mul(n.div_ceil(TILE_I))
        .and_then(|tiles| tiles.checked_mul(k / qk));
    match grid {
        Some(grid) if grid < GRID_LIMIT => Ok(()),
        _ => Err(rule(
            "grid limit",
            &format!("the tile grid of m {m}, n {n}, k {k} reaches {GRID_LIMIT} or more"),
        )),
    }
}

/// Bytes of a dense call's ranges: `n` rows of `k` values of native blocks, `q8_1_bytes(k,
/// m)` with the constant pad, and the destination reach `(m - 1) * ld + col_off + n` floats.
pub(crate) fn dense_needs(
    ty: QuantType,
    k: usize,
    n: usize,
    m: usize,
    ld: usize,
    col_off: usize,
) -> Result<DenseNeeds, EngError> {
    let overflow = || rule("size overflow", "a range size overflows");
    let w = ty.weight_bytes(n, k).ok_or_else(overflow)?;
    let dst = m
        .checked_sub(1)
        .and_then(|last| last.checked_mul(ld))
        .and_then(|lead| lead.checked_add(col_off))
        .and_then(|reach| reach.checked_add(n))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(overflow)?;
    Ok(DenseNeeds {
        w,
        y: q8_1_bytes(k, m)?,
        dst,
    })
}

/// The input of a dense GEMM: `m` activation rows of the weight's `k` values, read from `x`
/// (f32, `row_stride` floats apart), quantized into the q8_1 columns `y` that
/// [`MmqDenseCall::q8_1`] takes.
pub(crate) struct MmqDenseInputCall<'b> {
    pub(crate) packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking,
    pub(crate) weight: &'b NativeWeight,
    pub(crate) m: usize,
    pub(crate) row_stride: usize,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

/// The q8_1 scale layout (D4 or DS4) follows the weight type of the GEMM that reads the
/// buffer; a mismatch gives wrong values, not a fault. This is the one place the dense path
/// derives it, from the weight itself.
fn mmq_type_of(ty: QuantType) -> Result<MmqType, EngError> {
    match ty {
        QuantType::Q80 => Ok(MmqType::Q80),
        QuantType::Q4K => Ok(MmqType::Q4K),
        QuantType::Q6K => Ok(MmqType::Q6K),
        QuantType::Q50 => Err(rule("type", "Q5_0 has no dense case in this packet")),
    }
}

/// One dense GEMM: `dst[j * ld + col_off + i] = sum_k w[i][k] * x[j][k]` for `i < n`,
/// `j < m`. The weight carries its own type, `k` and row count, so a call can never
/// disagree with the blocks it reads; `q8_1` holds the `mmq_quantize` columns of the same
/// type's scale layout, `fixup` the stream-k fixup range.
pub(crate) struct MmqDenseCall<'b> {
    pub(crate) weight: &'b NativeWeight,
    /// An optional `(row_offset, rows)` SEGMENT of `weight` instead of its whole row
    /// extent. Segmented calls exist so a wide fused upload can back several launches of
    /// the exact per-segment shapes: the vendored MMQ picks its stream-k schedule from
    /// `(n, m, k)`, so one wide launch is NOT bitwise the four narrow ones it replaces
    /// (the GDN fusion divergence, 2026-10-27); same-shape launches over the same bytes
    /// are.
    pub(crate) w_seg: Option<(usize, usize)>,
    pub(crate) m: usize,
    pub(crate) q8_1: DeviceRange<'b>,
    pub(crate) dst: DeviceRange<'b>,
    pub(crate) ld: usize,
    pub(crate) col_off: usize,
    pub(crate) fixup: DeviceRange<'b>,
}

impl CudaContext {
    /// Queues the plain q8_1 quantize of a dense GEMM's input on `stream`: the family-facing
    /// path to the buffer [`Self::mmq_dense`] reads, since the raw-range quantize wrappers
    /// are private to the prefill tree. The scale layout comes from `call.weight`, and `k`
    /// is its own, so the input cannot disagree with the GEMM that consumes it. Every
    /// refusal of `mmq_quantize` applies (the K grid, the chunk cap, the capacity of `y` with
    /// G1's pad, alignment, overlap).
    pub(crate) fn mmq_dense_input(
        &self,
        stream: &DeviceStream,
        call: MmqDenseInputCall<'_>,
    ) -> Result<(), EngError> {
        self.mmq_quantize(
            stream,
            MmqQuantizeCall {
                packing: call.packing,
                ty: mmq_type_of(call.weight.quant())?,
                k: call.weight.k(),
                row_stride: call.row_stride,
                ncols: call.m,
                x: call.x,
                gather: None,
                y: call.y,
            },
        )
    }

    /// Queues the dense GEMM of `call` on `stream`, writing the `n` floats of each of the
    /// `m` destination columns from `col_off`. The destination holds valid values after the
    /// stream drained. Every refusal of the module doc (shape, fit, alignment, overlap,
    /// fixup floor, foreign buffer or stream) happens here before any native call.
    pub(crate) fn mmq_dense(
        &self,
        stream: &DeviceStream,
        call: MmqDenseCall<'_>,
    ) -> Result<(), EngError> {
        let fixup_floor = self.mmq_fixup_bytes()?;
        self.launch_dense(stream, call, fixup_floor)
    }

    /// The checked launch, with the fixup range's floor as an argument: production passes
    /// the device maximum; the coupling row of plan section B.0 (c) passes a smaller one to
    /// get past the wrapper. A range below the launch's exact need then meets the C entry's
    /// own refusal (an invalid argument, status -1), not the vendored require of plan section
    /// B.0 (c), which stays as the backstop behind it (lane C's ruling on MSG-85 B2).
    fn launch_dense(
        &self,
        stream: &DeviceStream,
        call: MmqDenseCall<'_>,
        fixup_floor: usize,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (ty, k) = (call.weight.quant(), call.weight.k());
        let (w_off, n) = match call.w_seg {
            // A segment must be whole rows inside the weight's own extent.
            Some((off, rows)) => {
                let end = off
                    .checked_add(rows)
                    .ok_or_else(|| rule("weight segment", "the segment's row extent overflows"))?;
                if rows == 0 || end > call.weight.n_rows() {
                    return Err(rule(
                        "weight segment",
                        &format!(
                            "rows {rows} at offset {off} leave the weight's {} rows",
                            call.weight.n_rows()
                        ),
                    ));
                }
                (off, rows)
            }
            None => (0, call.weight.n_rows()),
        };
        let ints = check_dense_rules(ty, k, n, call.m, call.ld, call.col_off)?;
        let need = dense_needs(ty, k, n, call.m, call.ld, call.col_off)?;
        let row_bytes = ty
            .weight_bytes(1, k)
            .ok_or_else(|| rule("size overflow", "a range size overflows"))?;
        let w_start = w_off
            .checked_mul(row_bytes)
            .ok_or_else(|| rule("size overflow", "a range size overflows"))?;
        let w = call.weight.buffer().range(w_start, need.w)?;
        require_fit("weight", &w, need.w, 16)?;
        require_fit("q8_1", &call.q8_1, need.y, 16)?;
        require_fit("dst", &call.dst, need.dst, 4)?;
        require_fit("fixup", &call.fixup, fixup_floor, 16)?;
        let claims = [
            Claim::of("weight", &w, need.w),
            Claim::of("q8_1", &call.q8_1, need.y),
            Claim::of("dst", &call.dst, need.dst),
        ];
        let fixup = Claim::of("fixup", &call.fixup, call.fixup.bytes());
        require_apart(&claims[2], &claims[0])?;
        require_apart(&claims[2], &claims[1])?;
        require_apart(&fixup, &claims[0])?;
        require_apart(&fixup, &claims[1])?;
        require_apart(&fixup, &claims[2])?;
        let raw = MmqDenseRaw {
            w: self.raw_range(&w)?,
            y: self.raw_range(&call.q8_1)?,
            dst: self.raw_range(&call.dst)?,
            fixup: self.raw_range(&call.fixup)?,
            ints,
        };
        self.launch_on(stream, move |stream_raw| mmq_dense_launch(stream_raw, raw))
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
