//! Safe wrappers of the MMQ q8_1 quantize entries: the plain and gathered
//! `CudaContext::mmq_quantize` and `CudaContext::mmq_quantize_scatter` of
//! `native/prefill/mmq_quant.cu`, which dispatch the explicit [`MmqPacking`] policy. Every MMQ
//! GEMM input is requantized from f32. The weight type selects the D4/DS4 layout and the
//! policy-specific quantizer. Q8_0/Q6_K/Q5_0/NVFP4 use D4, while Q4_K uses DS4
//! (mmq.cuh:61-101).
//!
//! The wrappers refuse every bad argument as a `BadRequest` before any native call (the house
//! pattern; the C entry mirrors the rules so it can never latch the backend), with the error
//! tags `mmq_quant: <rule>`. The q8_1 range is sized by [`mmq_size::q8_1_bytes`] — the one
//! constant pad is part of the capacity: a range one byte short is refused, and the quantize
//! never writes the pad (the GEMM's y-tile copies may read it; the determinism test pins that).
//!
//! Visibility: these raw-range wrappers are `pub(in crate::kernels::cuda::prefill)` — only
//! the prefill tree may call them, and the scatter and gather maps reach the entries from
//! outside this module only through the typed `MmidIds` wrappers of `mmq_mmid`. The map
//! VALUES are trusted device indices (the guarded ids builder makes them permutations); this
//! wrapper checks their capacity only. The gathered call's source spans `ncols` rows (the
//! SwiGLU `act` layout, one row per compact column); a smaller source is refused by the fit
//! rule, and a `src_rows` field would be the extension if a caller ever needs a wider
//! source.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no caller yet: the chunk driver does not call this kernel; drop the expect when it does"
    )
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::mmq_quant::{
    MmqQuantizeRaw, MmqQuantizeScatterRaw, QuantInts, ScatterInts, mmq_quantize_launch,
    mmq_quantize_scatter_launch,
};
use crate::kernels::cuda::prefill::mmq_size::{MmqType, NCOLS_LIMIT, q8_1_bytes};

/// The k grid of the MMQ GEMM: the k loop reads 256-value strides (`MMQ_ITER_K`) with no k
/// clamp, so a buffer the GEMM cannot read safely is never built. Checked before every other k
/// rule, so K = 384 (a multiple of 128 and of every flat qk) reaches this tag for every type.
///
/// The one definition of the K rules of the MMQ glue: the mmid wrapper imports this and `K_MAX`
/// instead of keeping a copy.
pub(crate) const K_GRID: usize = 256;
/// The k extent bound (the tree's `MAX_DIM`).
pub(crate) const K_MAX: usize = 32_768;
/// The row-stride bound: `(ncols - 1) * row_stride + k` floats must stay well inside usize.
const ROW_STRIDE_MAX: usize = 1 << 30;
/// The gathered/scattered column bound: `ne_get_rows = M * top_k` at top_k 8 and the 2,048
/// chunk cap.
const ROWS_LIMIT: usize = 16_384;
/// The top_k bound (the ids guard's own rule).
const TOP_K_LIMIT: usize = 32;

/// A `BadRequest` that names the rule that fired (the `mmq_quant` tag).
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("mmq_quant: {tag}: {detail}"))
}

/// The `ggml_type` discriminant of `ty` (the shim values, `QuantType::ggml`'s; pinned by
/// test against the decode side).
pub(super) fn ggml_of(ty: MmqType) -> i32 {
    match ty {
        MmqType::Q80 => 8,
        MmqType::Q50 => 6,
        MmqType::Q4K => 12,
        MmqType::Q6K => 14,
        MmqType::Nvfp4 => 40,
    }
}

/// `value` as `c_int` for the rule `tag`, or the "exceeds i32" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the scalar rules of a quantize call and converts the integers once. `gathered`
/// switches the column bound between the plain chunk cap and the gathered `ne_get_rows`
/// bound. Every [`MmqType`] has a q8_1 ds layout, so the "no layout" refusal is the C
/// entry's alone (a C caller passing an unknown discriminant).
fn check_quantize_rules(
    ty: MmqType,
    k: usize,
    row_stride: usize,
    ncols: usize,
    gathered: bool,
) -> Result<QuantInts, EngError> {
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
    if row_stride < k || !row_stride.is_multiple_of(4) || row_stride > ROW_STRIDE_MAX {
        return Err(rule(
            "row_stride",
            &format!(
                "row_stride is {row_stride}, not a multiple of 4 with k..={ROW_STRIDE_MAX} \
                 (k is {k})"
            ),
        ));
    }
    let cap = if gathered { ROWS_LIMIT } else { NCOLS_LIMIT };
    if ncols == 0 || ncols > cap {
        return Err(rule(
            "ncols",
            &format!(
                "ncols is {ncols}, outside 1..={cap} ({})",
                if gathered {
                    "the gathered ne_get_rows bound"
                } else {
                    "the chunk cap"
                }
            ),
        ));
    }
    Ok(QuantInts {
        packing: MmqPacking::Vendor as c_int,
        ty: ggml_of(ty),
        k: to_int("k", k)?,
        row_stride: to_int("row_stride", row_stride)?,
        ncols: to_int("ncols", ncols)?,
    })
}

/// Checks the scalar rules of a scatter call and converts the integers once.
fn check_scatter_rules(
    ty: MmqType,
    k: usize,
    row_stride: usize,
    m: usize,
    top_k: usize,
) -> Result<ScatterInts, EngError> {
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
    if row_stride < k || !row_stride.is_multiple_of(4) || row_stride > ROW_STRIDE_MAX {
        return Err(rule(
            "row_stride",
            &format!(
                "row_stride is {row_stride}, not a multiple of 4 with k..={ROW_STRIDE_MAX} \
                 (k is {k})"
            ),
        ));
    }
    if m == 0 || m > NCOLS_LIMIT {
        return Err(rule(
            "M",
            &format!("m is {m}, outside 1..={NCOLS_LIMIT} (the chunk cap)"),
        ));
    }
    if top_k == 0 || top_k > TOP_K_LIMIT {
        return Err(rule(
            "top_k",
            &format!("top_k is {top_k}, outside 1..={TOP_K_LIMIT}"),
        ));
    }
    let nrows = m
        .checked_mul(top_k)
        .ok_or_else(|| rule("ne_get_rows", "m * top_k overflows"))?;
    if nrows > ROWS_LIMIT {
        return Err(rule(
            "ne_get_rows",
            &format!("m * top_k is {nrows}, above {ROWS_LIMIT}"),
        ));
    }
    Ok(ScatterInts {
        packing: MmqPacking::Vendor as c_int,
        ty: ggml_of(ty),
        k: to_int("k", k)?,
        row_stride: to_int("row_stride", row_stride)?,
        m: to_int("m", m)?,
        top_k: to_int("top_k", top_k)?,
    })
}

/// Bytes of a quantize call's ranges: the source holds `(ncols - 1) * row_stride + k`
/// floats (`ncols = M * top_k` at the gathered down call), the output
/// `q8_1_bytes(k, ncols)` with the constant pad.
fn quantize_needs(k: usize, row_stride: usize, ncols: usize) -> Result<(usize, usize), EngError> {
    let x = (ncols - 1)
        .checked_mul(row_stride)
        .and_then(|lead| lead.checked_add(k))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(|| rule("size overflow", "the source size overflows"))?;
    Ok((x, q8_1_bytes(k, ncols)?))
}

/// Bytes of a scatter call's ranges: source, inverse map and output.
fn scatter_needs(
    k: usize,
    row_stride: usize,
    m: usize,
    top_k: usize,
) -> Result<(usize, usize, usize), EngError> {
    let x = (m - 1)
        .checked_mul(row_stride)
        .and_then(|lead| lead.checked_add(k))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(|| rule("size overflow", "the source size overflows"))?;
    let inv = m
        .checked_mul(top_k)
        .and_then(|ints| ints.checked_mul(4))
        .ok_or_else(|| rule("size overflow", "the inverse map size overflows"))?;
    Ok((
        x,
        inv,
        q8_1_bytes(
            k,
            m.checked_mul(top_k)
                .ok_or_else(|| rule("size overflow", "m * top_k overflows"))?,
        )?,
    ))
}

/// Activation packing is a caller policy, separate from the consuming weight layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum MmqPacking {
    /// Preserve the vendored 32-value scale groups and original rounding.
    Vendor = 0,
    /// Reference-compatible Q8_0 and Q4_K packing. Other types retain vendor packing.
    CpuReference = 1,
}

/// One (optionally gathered) quantize call: `ncols` output columns over `ncols` source
/// rows; `gather` some reads row `gather[j]` into column j (the gathered down input).
pub(crate) struct MmqQuantizeCall<'b> {
    pub(crate) packing: MmqPacking,
    pub(crate) ty: MmqType,
    pub(crate) k: usize,
    pub(crate) row_stride: usize,
    pub(crate) ncols: usize,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) gather: Option<DeviceRange<'b>>,
    pub(crate) y: DeviceRange<'b>,
}

/// One scatter call: `m` rows, each quantized once into its `top_k` compact columns.
pub(crate) struct MmqQuantizeScatterCall<'b> {
    pub(crate) packing: MmqPacking,
    pub(crate) ty: MmqType,
    pub(crate) k: usize,
    pub(crate) row_stride: usize,
    pub(crate) m: usize,
    pub(crate) top_k: usize,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) inv: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

impl CudaContext {
    /// Queues the (optionally gathered) q8_1 quantize of `call` on `stream`: `ncols`
    /// columns, one `block_q8_1_mmq` per 128 values per column, k-block-major, in the ds
    /// layout of `call.ty`. The buffer is valid after the stream drained; the pad tail is
    /// never written. Every refusal of the module doc (shape, fit, alignment, overlap,
    /// foreign buffer or stream) happens here before any native call.
    pub(in crate::kernels::cuda::prefill) fn mmq_quantize(
        &self,
        stream: &DeviceStream,
        call: MmqQuantizeCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let gathered = call.gather.is_some();
        let mut ints =
            check_quantize_rules(call.ty, call.k, call.row_stride, call.ncols, gathered)?;
        ints.packing = call.packing as c_int;
        let (x_need, y_need) = quantize_needs(call.k, call.row_stride, call.ncols)?;
        require_fit("x", &call.x, x_need, 16)?;
        require_fit("y", &call.y, y_need, 16)?;
        if let Some(map) = &call.gather {
            require_fit("gather", map, call.ncols * 4, 4)?;
        }
        let out = Claim::of("y", &call.y, y_need);
        require_apart(&out, &Claim::of("x", &call.x, x_need))?;
        if let Some(map) = &call.gather {
            require_apart(&out, &Claim::of("gather", map, call.ncols * 4))?;
        }
        let x = self.raw_range(&call.x)?;
        let y = self.raw_range(&call.y)?;
        let gather = call
            .gather
            .as_ref()
            .map(|map| self.raw_range(map))
            .transpose()?;
        self.launch_on(stream, move |raw| {
            mmq_quantize_launch(raw, MmqQuantizeRaw { x, gather, y, ints })
        })
    }

    /// Queues the scatter quantize of `call` on `stream`: each of the `m` source rows is
    /// quantized once and its block written to the `top_k` compact columns the inverse map
    /// names, `ne_get_rows = m * top_k` columns in total. The same status contract as
    /// [`Self::mmq_quantize`], with the `m`, `top_k` and `ne_get_rows` domain rules.
    pub(in crate::kernels::cuda::prefill) fn mmq_quantize_scatter(
        &self,
        stream: &DeviceStream,
        call: MmqQuantizeScatterCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let mut ints = check_scatter_rules(call.ty, call.k, call.row_stride, call.m, call.top_k)?;
        ints.packing = call.packing as c_int;
        let (x_need, inv_need, y_need) =
            scatter_needs(call.k, call.row_stride, call.m, call.top_k)?;
        require_fit("x", &call.x, x_need, 16)?;
        require_fit("ids_src1_inv", &call.inv, inv_need, 4)?;
        require_fit("y", &call.y, y_need, 16)?;
        let out = Claim::of("y", &call.y, y_need);
        require_apart(&out, &Claim::of("x", &call.x, x_need))?;
        require_apart(&out, &Claim::of("ids_src1_inv", &call.inv, inv_need))?;
        let x = self.raw_range(&call.x)?;
        let inv = self.raw_range(&call.inv)?;
        let y = self.raw_range(&call.y)?;
        self.launch_on(stream, move |raw| {
            mmq_quantize_scatter_launch(raw, MmqQuantizeScatterRaw { x, inv, y, ints })
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
