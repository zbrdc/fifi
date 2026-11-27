//! The SwiGLU of a step's feed-forward block in f32, the contract of the CUDA adapter's
//! swiglu rows: `act[r][c] = silu(gu[r][c]) * gu[r][up_off + c]`.
//!
//! [`Frame::swiglu_rows`] encodes one kernel into the open command buffer of a frame. The
//! call allocates nothing and copies nothing.
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

const SOURCE: &str = include_str!("../native/swiglu_rows.metal");
const KERNEL: &str = "swiglu_rows";

/// The most rows of one call.
pub(crate) const SWIGLU_MAX_ROWS: usize = 32_768;
/// The most outputs of one row.
pub(crate) const SWIGLU_MAX_FF: usize = 32_768;
/// `n_ff`, `up_off`, `ld_in` and `ld_out` are multiples of this: the kernel moves four floats.
pub(crate) const SWIGLU_QUAD: usize = 4;
/// The alignment of the kernel's 16-byte loads.
const LOAD_ALIGN: usize = 16;
/// Threads a threadgroup, four columns each.
const THREADS: usize = 256;
const VALUE_BYTES: usize = 4;

/// Geometry of one call: `rows` rows of `n_ff` outputs; the up half starts at column `up_off`
/// of a `gu` row of `ld_in` floats, and an `act` row holds `ld_out` floats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SwigluRowsShape {
    pub(crate) rows: usize,
    pub(crate) n_ff: usize,
    pub(crate) up_off: usize,
    pub(crate) ld_in: usize,
    pub(crate) ld_out: usize,
}

/// One call. `gu`: f32 `[rows][ld_in]`, read only, the gate at columns `0..n_ff` and the up
/// values at `up_off..up_off + n_ff`; other columns are not read. `act`: f32 `[rows][ld_out]`,
/// written at columns `0..n_ff` and no other byte. Both start on a 16-byte grid and share no
/// byte.
#[derive(Clone, Copy)]
pub(crate) struct SwigluRowsCall<'b> {
    pub(crate) shape: SwigluRowsShape,
    pub(crate) gu: DeviceRange<'b>,
    pub(crate) act: DeviceRange<'b>,
}

/// The bytes the two ranges of a call need at least.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SwigluRowsBytes {
    pub(crate) gu: usize,
    pub(crate) act: usize,
}

/// The bytes of the two ranges of a call of this shape: `(rows - 1) * ld_in + up_off + n_ff`
/// floats of `gu` and `(rows - 1) * ld_out + n_ff` floats of `act`. `None` when the shape
/// breaks a rule of the CUDA adapter or on overflow.
pub(crate) fn swiglu_rows_bytes(shape: SwigluRowsShape) -> Option<SwigluRowsBytes> {
    let SwigluRowsShape {
        rows,
        n_ff,
        up_off,
        ld_in,
        ld_out,
    } = shape;
    let quad = |value: usize| value.is_multiple_of(SWIGLU_QUAD);
    let up_end = up_off.checked_add(n_ff)?;
    let fits = (1..=SWIGLU_MAX_ROWS).contains(&rows)
        && (SWIGLU_QUAD..=SWIGLU_MAX_FF).contains(&n_ff)
        && quad(n_ff)
        && quad(up_off)
        && up_off >= n_ff
        && quad(ld_in)
        && ld_in >= up_end
        && quad(ld_out)
        && ld_out >= n_ff;
    if !fits {
        return None;
    }
    let extent = |ld: usize, tail: usize| {
        (rows - 1)
            .checked_mul(ld)?
            .checked_add(tail)?
            .checked_mul(VALUE_BYTES)
    };
    Some(SwigluRowsBytes {
        gu: extent(ld_in, up_end)?,
        act: extent(ld_out, n_ff)?,
    })
}

fn invalid_shape(shape: SwigluRowsShape) -> EngError {
    EngError::BadRequest(format!(
        "the Metal swiglu takes 1 to {SWIGLU_MAX_ROWS} rows, {SWIGLU_QUAD} to {SWIGLU_MAX_FF} outputs a row, and n_ff, up_off, ld_in and ld_out multiples of {SWIGLU_QUAD} with up_off at least n_ff, ld_in at least up_off + n_ff and ld_out at least n_ff, got {shape:?}"
    ))
}

/// The kernel's parameters, the layout of `SwigluParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct SwigluParams {
    rows: u32,
    n_ff: u32,
    up_off: u32,
    ld_in: u32,
    ld_out: u32,
}

const _: () = assert!(std::mem::size_of::<SwigluParams>() == 20);

impl SwigluParams {
    fn bytes(self) -> [u8; 20] {
        let mut out = [0u8; 20];
        let fields = [self.rows, self.n_ff, self.up_off, self.ld_in, self.ld_out];
        for (slot, value) in fields.into_iter().enumerate() {
            out[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}

/// A call after its checks: its bindings, the shader and its parameters.
#[derive(Clone, Copy)]
struct SwigluRowsPlan {
    gu: Binding,
    act: Binding,
    params: SwigluParams,
}

impl MetalContext {
    /// One SwiGLU: a frame of one dispatch, drained.
    pub(crate) fn swiglu_rows(&self, call: SwigluRowsCall<'_>) -> Result<(), EngError> {
        let mut frame = self.frame();
        frame.swiglu_rows(call)?;
        frame.drain().map(|_| ())
    }

    /// The same dispatch `repeats` times in one command buffer of a frame, for a timing test:
    /// the GPU time of the command buffer in nanoseconds.
    #[cfg(test)]
    pub(crate) fn swiglu_rows_timed(
        &self,
        call: SwigluRowsCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        let mut frame = self.frame_of(repeats);
        for _ in 0..repeats {
            frame.swiglu_rows(call)?;
        }
        frame.drain()
    }
}

impl Frame {
    /// Encodes one SwiGLU into the frame.
    pub(super) fn swiglu_rows(&mut self, call: SwigluRowsCall<'_>) -> Result<(), EngError> {
        let shared = self.shared();
        let plan = plan_swiglu_rows(shared, &call)?;
        let kernel = shared.kernel(KERNEL, SOURCE)?;
        let SwigluRowsShape { rows, n_ff, .. } = call.shape;
        // Four columns a thread, 256 threads a threadgroup, one row a grid row.
        let quads = n_ff / SWIGLU_QUAD;
        let grid: Grid = [quads.div_ceil(THREADS / SWIGLU_QUAD) * THREADS, rows, 1];
        self.encode(
            &kernel,
            &[plan.gu, plan.act],
            &plan.params.bytes(),
            grid,
            [THREADS, 1, 1],
            1,
        )
    }
}

fn plan_swiglu_rows(
    shared: &Shared,
    call: &SwigluRowsCall<'_>,
) -> Result<SwigluRowsPlan, EngError> {
    let shape = call.shape;
    let need = swiglu_rows_bytes(shape).ok_or_else(|| invalid_shape(shape))?;
    require_range(shared, "gate and up rows", &call.gu, need.gu, LOAD_ALIGN)?;
    require_range(shared, "activation rows", &call.act, need.act, LOAD_ALIGN)?;
    if overlaps(&call.act, &call.gu) {
        return Err(EngError::BadRequest(
            "the act range overlaps the gate and up range".to_owned(),
        ));
    }
    let SwigluRowsShape {
        rows,
        n_ff,
        up_off,
        ld_in,
        ld_out,
    } = shape;
    let params = SwigluParams {
        rows: rows as u32,
        n_ff: n_ff as u32,
        up_off: up_off as u32,
        ld_in: ld_in as u32,
        ld_out: ld_out as u32,
    };
    Ok(SwigluRowsPlan {
        gu: call.gu.binding(),
        act: call.act.binding(),
        params,
    })
}
