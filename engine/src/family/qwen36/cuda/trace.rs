//! The CUDA readback glue of the layer trace: `resident/trace.rs` owns the files and the
//! counters, this module reads the rows back from the device. Everything here runs ONLY under
//! `Q36_LAYER_TRACE_DIR` (the callers hold an `Option`), so the default path pays nothing.

use super::moe::Launch;
use crate::{EngError, kernels::cuda::DeviceRange};

/// How many rows of a prefill piece the trace keeps: the last five, enough to see the seam
/// rows next to the far rows without dumping the whole piece.
const PREFILL_TRACE_ROWS: usize = 5;

/// Reads one F32 range back to host values (little-endian). `read_at` orders the copy on the
/// stream, so the values are final when it returns.
pub(super) fn f32_range(
    launch: &Launch<'_>,
    range: &DeviceRange<'_>,
) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0u8; range.bytes()];
    range
        .buffer()
        .read_at(launch.stream, range.offset(), &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// The trace rows of one prefill piece: the last `PREFILL_TRACE_ROWS` rows of `x2`
/// (the f32 `[m][hidden]` post-mixer residual), row-major, still in piece order.
/// `hidden` comes from the model dims, NOT from the range's size: the chunk slot
/// is allocated for the piece CAP, so `x2.bytes() / m` is not the row width.
pub(super) fn prefill_tail(
    launch: &Launch<'_>,
    x2: &DeviceRange<'_>,
    m: usize,
    hidden: usize,
) -> Result<Vec<f32>, EngError> {
    let rows = m.min(PREFILL_TRACE_ROWS);
    let offset = (m - rows) * hidden * 4;
    let mut bytes = vec![0u8; rows * hidden * 4];
    x2.buffer()
        .read_at(launch.stream, x2.offset() + offset, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}
