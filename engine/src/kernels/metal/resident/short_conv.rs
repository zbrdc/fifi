//! The short convolution of a step, the contract of the CUDA adapter's resident short conv
//! (`kernels/cuda/resident/short_conv.rs`), which is the arithmetic of `family/step/conv.rs`'s
//! `Variant::Residual`: for every channel `c`, `out[c] = (acc + input[c]) * (1 / sqrt(taps + 1))`
//! where `acc` is `0 + k[taps - 1] * input[c] + k[taps - 2] * history[0][c] + .. + k[0] *
//! history[taps - 2][c]` in that order, every operation rounded to nearest even. `history` holds
//! `taps - 1` rows of `width` floats, the newest past input first; a launch advances it in place
//! to the raw input, the old row 0, and so on. `health` is left alone, or set to a quiet NaN when
//! an input, a tap, a history value or an output was not finite. With `state`, the current row of
//! a depth stream takes `state[c] += out[c]`, one rounded add.
//!
//! [`Frame::short_conv`] encodes one kernel into the open command buffer of a frame. The call
//! allocates nothing and copies nothing. The norm is computed here in f32, as the CUDA kernel
//! does on the device: `taps + 1`, the square root, the reciprocal, each rounded.
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

const SOURCE: &str = include_str!("../native/short_conv.metal");
const KERNEL: &str = "short_conv";
const STATE_KERNEL: &str = "short_conv_state";
/// The most columns of one call, the CUDA adapter's maximum.
pub(crate) const SHORT_CONV_MAX_WIDTH: usize = 8192;
/// The most taps of one call, the CUDA adapter's maximum.
pub(crate) const SHORT_CONV_MAX_TAPS: usize = 8;
/// Threads a threadgroup; one thread a channel.
const THREADS: usize = 256;
const VALUE_BYTES: usize = 4;

/// The kernel's parameters, the layout of `ShortConvParams` in the shader.
#[derive(Clone, Copy)]
struct ShortConvParams {
    width: u32,
    taps: u32,
    norm: f32,
}
const _: () = assert!(std::mem::size_of::<ShortConvParams>() == 12);

/// One short convolution. `out` is either exactly `input` or shares no byte with it; `kernel`,
/// `history`, `health` and `state` share no byte with any other range.
#[derive(Clone, Copy)]
pub(crate) struct ShortConvCall<'b> {
    pub(crate) input: DeviceRange<'b>,
    pub(crate) kernel: DeviceRange<'b>,
    pub(crate) history: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    /// The current row of a depth stream, which takes `state[c] += out[c]` in the same launch;
    /// or absent.
    pub(crate) state: Option<DeviceRange<'b>>,
    pub(crate) health: DeviceRange<'b>,
    pub(crate) width: usize,
    pub(crate) taps: usize,
}

/// A call after its checks: its bindings in the kernel's order, the shader and its parameters.
#[derive(Clone, Copy)]
struct ShortConvPlan {
    input: Binding,
    taps: Binding,
    history: Binding,
    out: Binding,
    state: Option<Binding>,
    health: Binding,
    name: &'static str,
    params: ShortConvParams,
}

impl MetalContext {
    /// One short convolution: a frame of one dispatch, drained.
    pub(crate) fn short_conv(&self, call: ShortConvCall<'_>) -> Result<(), EngError> {
        let mut frame = self.frame();
        frame.short_conv(call)?;
        frame.drain().map(|_| ())
    }
}

impl Frame {
    /// Encodes one short convolution into the frame.
    pub(super) fn short_conv(&mut self, call: ShortConvCall<'_>) -> Result<(), EngError> {
        let shared = self.shared();
        let plan = plan_short_conv(shared, &call)?;
        let function = shared.kernel(plan.name, SOURCE)?;
        let ShortConvPlan {
            input,
            taps,
            history,
            out,
            state,
            health,
            params,
            ..
        } = plan;
        let bound: Vec<Binding> = [input, taps, history, out]
            .into_iter()
            .chain(state)
            .chain([health])
            .collect();
        let mut bytes = [0u8; 12];
        bytes[..4].copy_from_slice(&params.width.to_le_bytes());
        bytes[4..8].copy_from_slice(&params.taps.to_le_bytes());
        bytes[8..].copy_from_slice(&params.norm.to_le_bytes());
        // One thread a channel, whole threadgroups.
        let threads = params.width as usize;
        let grid: Grid = [threads.div_ceil(THREADS) * THREADS, 1, 1];
        self.encode(&function, &bound, &bytes, grid, [THREADS, 1, 1], 1)
    }
}

/// The rules of the CUDA adapter, in its order: the shape, then the six ranges, then that `out`
/// is `input` or apart from it, and that every other pair of ranges shares no byte.
fn plan_short_conv(shared: &Shared, call: &ShortConvCall<'_>) -> Result<ShortConvPlan, EngError> {
    let ShortConvCall {
        input,
        kernel,
        history,
        out,
        state,
        health,
        width,
        taps,
    } = *call;
    if !(1..=SHORT_CONV_MAX_WIDTH).contains(&width) || !(2..=SHORT_CONV_MAX_TAPS).contains(&taps) {
        return Err(EngError::BadRequest(format!(
            "the short conv takes 1 to {SHORT_CONV_MAX_WIDTH} columns and 2 to {SHORT_CONV_MAX_TAPS} taps, got width {width}, taps {taps}"
        )));
    }
    let row = width * VALUE_BYTES;
    require_range(shared, "input", &input, row, RANGE_ALIGN)?;
    require_range(shared, "taps", &kernel, taps * row, RANGE_ALIGN)?;
    require_range(shared, "history", &history, (taps - 1) * row, RANGE_ALIGN)?;
    require_range(shared, "output", &out, row, RANGE_ALIGN)?;
    require_range(shared, "health", &health, VALUE_BYTES, RANGE_ALIGN)?;
    if let Some(state) = &state {
        require_range(shared, "state", state, row, RANGE_ALIGN)?;
    }
    // `out` is exactly `input`, the same address, or shares no byte with it.
    if out.address() != input.address() && overlaps(&out, &input) {
        return Err(EngError::BadRequest(
            "the output range overlaps the input range without being it".to_owned(),
        ));
    }
    let others = [&kernel, &history, &health];
    if others
        .iter()
        .any(|other| overlaps(&out, other) || overlaps(&input, other))
    {
        return Err(EngError::BadRequest(
            "the taps, history or health range overlaps the input or the output range".to_owned(),
        ));
    }
    if overlaps(&kernel, &history) || overlaps(&kernel, &health) || overlaps(&history, &health) {
        return Err(EngError::BadRequest(
            "two of the taps, history and health ranges overlap".to_owned(),
        ));
    }
    if let Some(state) = &state
        && [&input, &out, &kernel, &history, &health]
            .iter()
            .any(|other| overlaps(state, other))
    {
        return Err(EngError::BadRequest(
            "the state range overlaps another range".to_owned(),
        ));
    }
    Ok(ShortConvPlan {
        input: input.binding(),
        taps: kernel.binding(),
        history: history.binding(),
        out: out.binding(),
        state: state.map(|range| range.binding()),
        health: health.binding(),
        name: if state.is_some() {
            STATE_KERNEL
        } else {
            KERNEL
        },
        params: ShortConvParams {
            width: width as u32,
            taps: taps as u32,
            norm: norm(taps),
        },
    })
}

/// The norm, as the CUDA kernel computes it on the device: `taps + 1` as f32, the square root,
/// the reciprocal, each rounded to nearest even.
fn norm(taps: usize) -> f32 {
    (taps as f32 + 1.0).sqrt().recip()
}
