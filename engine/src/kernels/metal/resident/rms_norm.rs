//! The RMS norm of a step with its optional residual add, the contract of the CUDA adapter's
//! `add_rmsnorm_q8` without its q8_1 row: over `rows` rows of `n` floats,
//! `out = x * rsqrt(sum(x * x) / n + eps) * weight`, and with `delta`, `x` takes `x + delta`
//! first, the new residual. The Metal projections read f32 rows, so no q8_1 row is written.
//!
//! [`Frame::rms_norm`] encodes one kernel into the open command buffer of a frame. The call
//! allocates nothing and copies nothing.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from
//! the native layer other than a missing device latches the backend terminal. A refused call
//! changes nothing.

use crate::EngError;

use super::super::context::{Grid, Shared};
use super::super::frame::Frame;
use super::super::native::Binding;
use super::super::{DeviceRange, MetalContext};
use super::{overlaps, require_range};

#[cfg(test)]
mod tests;

const SOURCE: &str = include_str!("../native/rms_norm.metal");
const KERNEL: &str = "rms_norm";
const ADD_KERNEL: &str = "rms_norm_add";

/// The most floats of one row: the CUDA adapter's `kMaxWidth`.
pub(crate) const RMS_NORM_MAX_N: usize = 32_768;
/// The most rows of one call: the CUDA adapter's `kMaxRows`, one block a row.
pub(crate) const RMS_NORM_MAX_ROWS: usize = 262_144;
/// `n` is a multiple of this, the rule of the CUDA adapter, so that a call is valid on both.
const N_MULTIPLE: usize = 32;
/// The alignment of the kernel's 16-byte loads.
const LOAD_ALIGN: usize = 16;
/// Threads a threadgroup: one threadgroup a row, the CUDA adapter's block.
const THREADS: usize = 256;
const VALUE_BYTES: usize = 4;

/// One RMS norm over `rows` rows of `n` floats, consecutive rows and one weight. `x`: f32
/// `[rows][n]`, read, and written in place with `x + delta` when `delta` is given. `delta`:
/// `None`, or f32 `[rows][n]`, read only. `weight`: f32 `[n]`, read only. `out`: f32
/// `[rows][n]`, the normalized rows; it shares no byte with another range. Every range starts
/// on a 16-byte grid.
#[derive(Clone, Copy)]
pub(crate) struct RmsNormCall<'b> {
    pub(crate) rows: usize,
    pub(crate) n: usize,
    /// Finite and above 0 (GGUF `layer_norm_rms_epsilon`).
    pub(crate) eps: f32,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) delta: Option<DeviceRange<'b>>,
    pub(crate) weight: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

/// The kernel's parameters, the layout of `RmsNormParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormParams {
    n: u32,
    eps: f32,
}

const _: () = assert!(std::mem::size_of::<RmsNormParams>() == 8);

/// A call after its checks: its bindings in the kernel's order, the shader and its parameter.
#[derive(Clone, Copy)]
struct RmsNormPlan {
    bound: [Option<Binding>; 4],
    params: RmsNormParams,
}

impl MetalContext {
    /// One RMS norm: a frame of one dispatch, drained.
    pub(crate) fn rms_norm(&self, call: RmsNormCall<'_>) -> Result<(), EngError> {
        let mut frame = self.frame();
        frame.rms_norm(call)?;
        frame.drain().map(|_| ())
    }

    /// The same dispatch `repeats` times in one command buffer of a frame, for a timing test:
    /// the GPU time of the command buffer in nanoseconds.
    #[cfg(test)]
    pub(crate) fn rms_norm_timed(
        &self,
        call: RmsNormCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        let mut frame = self.frame_of(repeats);
        for _ in 0..repeats {
            frame.rms_norm(call)?;
        }
        frame.drain()
    }
}

impl Frame {
    /// Encodes one RMS norm into the frame.
    pub(super) fn rms_norm(&mut self, call: RmsNormCall<'_>) -> Result<(), EngError> {
        let shared = self.shared();
        let plan = plan_rms_norm(shared, &call)?;
        let kernel = shared.kernel(
            if call.delta.is_some() {
                ADD_KERNEL
            } else {
                KERNEL
            },
            SOURCE,
        )?;
        let bound: Vec<Binding> = plan.bound.into_iter().flatten().collect();
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&plan.params.n.to_le_bytes());
        bytes[4..].copy_from_slice(&plan.params.eps.to_le_bytes());
        // One threadgroup of 256 threads a row.
        let grid: Grid = [THREADS, call.rows, 1];
        self.encode(&kernel, &bound, &bytes, grid, [THREADS, 1, 1], 1)
    }
}

/// The rules of the CUDA adapter, in its order: the counts, then `eps`, then the four ranges
/// as far as the call gives them, then the overlaps of the ranges it writes.
fn plan_rms_norm(shared: &Shared, call: &RmsNormCall<'_>) -> Result<RmsNormPlan, EngError> {
    let RmsNormCall {
        rows,
        n,
        eps,
        x,
        delta,
        weight,
        out,
    } = *call;
    if !(N_MULTIPLE..=RMS_NORM_MAX_N).contains(&n) || !n.is_multiple_of(N_MULTIPLE) {
        return Err(EngError::BadRequest(format!(
            "the rms_norm needs n as a multiple of {N_MULTIPLE} up to {RMS_NORM_MAX_N}, got {n}"
        )));
    }
    if !(1..=RMS_NORM_MAX_ROWS).contains(&rows) {
        return Err(EngError::BadRequest(format!(
            "the rms_norm needs rows from 1 to {RMS_NORM_MAX_ROWS}, got {rows}"
        )));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(EngError::BadRequest(format!(
            "the rms_norm needs eps finite and above 0, got {eps}"
        )));
    }
    let row_bytes = n * VALUE_BYTES;
    let rows_bytes = rows * row_bytes;
    require_range(shared, "x", &x, rows_bytes, LOAD_ALIGN)?;
    if let Some(delta) = delta.as_ref() {
        require_range(shared, "delta", delta, rows_bytes, LOAD_ALIGN)?;
    }
    require_range(shared, "weight", &weight, row_bytes, LOAD_ALIGN)?;
    require_range(shared, "out", &out, rows_bytes, LOAD_ALIGN)?;
    let inputs = [Some(&x), delta.as_ref(), Some(&weight)];
    if inputs.iter().flatten().any(|other| overlaps(&out, other)) {
        return Err(EngError::BadRequest(
            "the out range overlaps the x, delta or weight range".to_owned(),
        ));
    }
    // With the add, x takes the sums: it may share no byte with delta or weight.
    let sums_into = [delta.as_ref(), Some(&weight)];
    if delta.is_some() && sums_into.iter().flatten().any(|other| overlaps(&x, other)) {
        return Err(EngError::BadRequest(
            "the x range takes the sum and overlaps the delta or the weight range".to_owned(),
        ));
    }
    Ok(RmsNormPlan {
        // x, delta, weight, out: the kernels' order of buffers, delta only in the add.
        bound: [
            Some(x.binding()),
            delta.map(|d| d.binding()),
            Some(weight.binding()),
            Some(out.binding()),
        ],
        params: RmsNormParams { n: n as u32, eps },
    })
}
