//! The element-wise residual sum of a step, the contract of the CUDA adapter's row add:
//! `out = x + y` in f32 over `n` floats.
//!
//! [`Frame::row_add`] encodes one kernel into the open command buffer of a frame. The call
//! allocates nothing and copies nothing. With `fused`, `out` holds the bf16 round trip of each
//! sum, the publish contract: a sum whose round trip is not finite is written 0 with bit 0 of
//! the flag set.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from
//! the native layer other than a missing device latches the backend terminal. A refused call
//! changes nothing.

use crate::EngError;

use super::super::context::{Grid, Shared};
use super::super::frame::Frame;
use super::super::native::Binding;
use super::super::{DeviceRange, MetalContext};
use super::{RANGE_ALIGN, overlaps, require_range};

#[cfg(test)]
mod tests;

const SOURCE: &str = include_str!("../native/row_add.metal");
const KERNEL: &str = "row_add";
const FUSED_KERNEL: &str = "row_add_fused";

/// The most floats of one call.
pub(crate) const ROW_ADD_MAX_N: usize = 4_194_304;
/// `n` is a multiple of this, the rule of the CUDA adapter, so that a call is valid on both.
const N_MULTIPLE: usize = 32;
/// Threads a threadgroup; one thread a float.
const THREADS: usize = 256;
const VALUE_BYTES: usize = 4;

/// One row add, `out = x + y` over `n` floats. With `fused`, `out` holds the bf16 round trip
/// of each sum instead, a sum whose round trip is not finite is written 0 with bit 0 of `flag`
/// set, and `flag` is required. `out` is not `x` or `y`.
#[derive(Clone, Copy)]
pub(crate) struct RowAddCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) n: usize,
    pub(crate) flag: Option<DeviceRange<'b>>,
    pub(crate) fused: bool,
}

/// A call after its checks: its bindings in the kernel's order, the shader and its parameter,
/// the float count as the shader takes it.
#[derive(Clone, Copy)]
struct RowAddPlan {
    out: Binding,
    x: Binding,
    y: Binding,
    flag: Option<Binding>,
    kernel: &'static str,
    n: u32,
}

impl MetalContext {
    /// One row add: a frame of one dispatch, drained.
    pub(crate) fn row_add(&self, call: RowAddCall<'_>) -> Result<(), EngError> {
        let mut frame = self.frame();
        frame.row_add(call)?;
        frame.drain().map(|_| ())
    }

    /// The same dispatch `repeats` times in one command buffer of a frame, for a timing test:
    /// the GPU time of the command buffer in nanoseconds.
    #[cfg(test)]
    pub(crate) fn row_add_timed(
        &self,
        call: RowAddCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        let mut frame = self.frame_of(repeats);
        for _ in 0..repeats {
            frame.row_add(call)?;
        }
        frame.drain()
    }
}

impl Frame {
    /// Encodes one row add into the frame.
    pub(super) fn row_add(&mut self, call: RowAddCall<'_>) -> Result<(), EngError> {
        let shared = self.shared();
        let plan = plan_row_add(shared, &call)?;
        let kernel = shared.kernel(plan.kernel, SOURCE)?;
        let RowAddPlan {
            out, x, y, flag, n, ..
        } = plan;
        let bound: Vec<Binding> = [out, x, y].into_iter().chain(flag).collect();
        // One thread a float, whole threadgroups.
        let threads = n as usize;
        let grid: Grid = [threads.div_ceil(THREADS) * THREADS, 1, 1];
        self.encode(&kernel, &bound, &n.to_le_bytes(), grid, [THREADS, 1, 1], 1)
    }
}

/// The rules of the CUDA adapter, in its order: the count, then the three ranges, then the
/// flag of a fused call.
fn plan_row_add(shared: &Shared, call: &RowAddCall<'_>) -> Result<RowAddPlan, EngError> {
    let n = call.n;
    if !(N_MULTIPLE..=ROW_ADD_MAX_N).contains(&n) || !n.is_multiple_of(N_MULTIPLE) {
        return Err(EngError::BadRequest(format!(
            "the row_add needs n as a multiple of {N_MULTIPLE} up to {ROW_ADD_MAX_N}, got {n}"
        )));
    }
    let bytes = n * VALUE_BYTES;
    require_range(shared, "out", &call.out, bytes, RANGE_ALIGN)?;
    require_range(shared, "x", &call.x, bytes, RANGE_ALIGN)?;
    require_range(shared, "y", &call.y, bytes, RANGE_ALIGN)?;
    if overlaps(&call.out, &call.x) || overlaps(&call.out, &call.y) {
        return Err(EngError::BadRequest(
            "the out range overlaps the x or the y range".to_owned(),
        ));
    }
    let mut flag = None;
    if call.fused {
        let range = call
            .flag
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("fused row_add needs a flag".to_owned()))?;
        require_range(shared, "flag", range, VALUE_BYTES, RANGE_ALIGN)?;
        let apart = [&call.out, &call.x, &call.y]
            .iter()
            .all(|other| !overlaps(range, other));
        if !apart {
            return Err(EngError::BadRequest(
                "the flag range overlaps the out, x or y range".to_owned(),
            ));
        }
        flag = Some(range.binding());
    }
    Ok(RowAddPlan {
        out: call.out.binding(),
        x: call.x.binding(),
        y: call.y.binding(),
        flag,
        kernel: if call.fused { FUSED_KERNEL } else { KERNEL },
        n: n as u32,
    })
}
