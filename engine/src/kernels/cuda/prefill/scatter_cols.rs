//! Safe launch wrapper of `scatter_cols`: copy a contiguous f32 block of `rows` rows of `n`
//! values into the columns `col_off..col_off + n` of the rows of a wider f32 matrix,
//! `dst[r * ld + col_off + c] = src[r * n + c]`. An EXL3 projection writes its output contiguous,
//! while the chunk rows that several projections share (the DeltaNet and attention input rows,
//! the gate/up rows of the SwiGLU) hold them side by side.
//!
//! [`CudaContext::scatter_cols`] queues one kernel on a [`DeviceStream`] and returns. The call
//! allocates nothing, copies nothing and never synchronizes. The copy is exact.
//!
//! Every argument is checked here, before any native call, and refused as `BadRequest`: the
//! rules mirror `valid_shape` of `scatter_cols.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend).

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::scatter_cols::{
    ScatterColsInts, ScatterColsRaw, scatter_cols_launch,
};

pub(crate) const MAX_ROWS: usize = 32_768;
pub(crate) const MAX_WIDTH: usize = 65_536;
/// `n`, `col_off` and `ld` are multiples of this: the kernel moves `float4`.
pub(crate) const QUAD: usize = 4;

/// Geometry of one call: `rows` rows of `n` values go to columns `col_off..col_off + n` of a
/// destination whose rows are `ld` floats apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScatterColsShape {
    pub(crate) rows: usize,
    pub(crate) n: usize,
    pub(crate) ld: usize,
    pub(crate) col_off: usize,
}

/// One call. Both ranges are device ranges of this context.
///
/// - `src`: f32 `[rows][n]`, 16-byte aligned, read only.
/// - `dst`: f32 `[rows][ld]`, 16-byte aligned: columns `col_off..col_off + n` of every row are
///   written, no other byte. Its extent shares no byte with the `src` extent.
pub(crate) struct ScatterColsCall<'b> {
    pub(crate) shape: ScatterColsShape,
    pub(crate) src: DeviceRange<'b>,
    pub(crate) dst: DeviceRange<'b>,
}

/// Bytes of the two extents a call touches: `rows * n` floats of `src` and
/// `(rows - 1) * ld + col_off + n` floats of `dst`. `None` for zero rows or on overflow.
pub(crate) fn scatter_cols_needs(shape: &ScatterColsShape) -> Option<(usize, usize)> {
    let before = shape.rows.checked_sub(1)?;
    let src = shape.rows.checked_mul(shape.n)?.checked_mul(4)?;
    let dst = before
        .checked_mul(shape.ld)?
        .checked_add(shape.col_off)?
        .checked_add(shape.n)?
        .checked_mul(4)?;
    Some((src, dst))
}

fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("scatter_cols: {tag}: {detail}"))
}

fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

fn require_quad(tag: &str, value: usize) -> Result<(), EngError> {
    if value.is_multiple_of(QUAD) {
        return Ok(());
    }
    Err(rule(
        tag,
        &format!("{tag} is {value}, not a multiple of {QUAD}"),
    ))
}

/// Checks every shape rule of `valid_shape` in `scatter_cols.cu` and converts the integers once,
/// so that no conversion can fail inside the launch closure.
pub(in crate::kernels::cuda) fn check_shape(
    shape: &ScatterColsShape,
) -> Result<ScatterColsInts, EngError> {
    if !(1..=MAX_ROWS).contains(&shape.rows) {
        let detail = format!("rows is {}, not 1 to {MAX_ROWS}", shape.rows);
        return Err(rule("rows", &detail));
    }
    if !(QUAD..=MAX_WIDTH).contains(&shape.n) {
        let detail = format!("n is {}, not {QUAD} to {MAX_WIDTH}", shape.n);
        return Err(rule("n", &detail));
    }
    require_quad("n", shape.n)?;
    require_quad("col_off", shape.col_off)?;
    require_quad("ld", shape.ld)?;
    let end = shape
        .col_off
        .checked_add(shape.n)
        .ok_or_else(|| rule("col_off", "col_off + n overflows"))?;
    if shape.ld < end {
        let detail = format!("ld is {}, below col_off + n {end}", shape.ld);
        return Err(rule("ld", &detail));
    }
    Ok(ScatterColsInts {
        rows: to_int("rows", shape.rows)?,
        n: to_int("n", shape.n)?,
        ld: to_int("ld", shape.ld)?,
        col_off: to_int("col_off", shape.col_off)?,
    })
}

impl CudaContext {
    /// Queues the scatter of `call` on `stream`. Returns when the kernel is queued; `dst` is
    /// valid after the stream drained.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a buffer
    /// or stream of another context, a shape the kernel refuses, a range that is too short or
    /// not 16-byte aligned, and a `dst` extent that overlaps the `src` extent.
    pub(crate) fn scatter_cols(
        &self,
        stream: &DeviceStream,
        call: ScatterColsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let src = self.raw_range(&call.src)?;
        let dst = self.raw_range(&call.dst)?;
        let ints = check_shape(&call.shape)?;
        let (src_bytes, dst_bytes) = scatter_cols_needs(&call.shape)
            .ok_or_else(|| rule("size", "the extents overflow usize"))?;
        require_fit("src", &call.src, src_bytes, 16)?;
        require_fit("dst", &call.dst, dst_bytes, 16)?;
        require_apart(
            &Claim::of("dst", &call.dst, dst_bytes),
            &Claim::of("src", &call.src, src_bytes),
        )?;
        self.launch_on(stream, move |raw| {
            scatter_cols_launch(raw, ScatterColsRaw { src, dst, ints })
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
