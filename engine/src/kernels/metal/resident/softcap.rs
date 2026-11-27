//! The soft cap of a head's logits, the contract of the CUDA adapter's cap in place:
//! `x = cap * tanh(x / cap)` in f32 over the first `n` floats of a row. A NaN, an infinity or a
//! zero of the row is left as it is: a caller that validates the raw projection reads the first
//! two back, and a zero keeps its sign, as on CUDA.
//!
//! The shader computes `tanh` itself: the library's is not within the CUDA adapter's frozen
//! bound on this device (`native/softcap.metal`).
//!
//! [`Frame::softcap_in_place`] encodes one kernel into the open command buffer of a frame. The
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
use super::{RANGE_ALIGN, require_range};

#[cfg(test)]
mod tests;

const SOURCE: &str = include_str!("../native/softcap.metal");
const KERNEL: &str = "softcap_in_place";

/// The most floats of one call, the CUDA adapter's maximum.
pub(crate) const SOFTCAP_MAX_N: usize = 4_194_304;
/// Threads a threadgroup; one thread a float.
const THREADS: usize = 256;
const VALUE_BYTES: usize = 4;

/// The kernel's parameters, the layout of `SoftcapParams` in the shader.
#[derive(Clone, Copy)]
struct SoftcapParams {
    n: u32,
    cap: f32,
}
const _: () = assert!(std::mem::size_of::<SoftcapParams>() == 8);

/// One cap in place over the first `n` floats of `x`.
#[derive(Clone, Copy)]
pub(crate) struct SoftcapCall<'b> {
    pub(crate) x: DeviceRange<'b>,
    pub(crate) n: usize,
    pub(crate) cap: f32,
}

/// A call after its checks: its binding, the shader's parameters.
#[derive(Clone, Copy)]
struct SoftcapPlan {
    x: Binding,
    params: SoftcapParams,
}

impl MetalContext {
    /// One cap in place: a frame of one dispatch, drained.
    pub(crate) fn softcap_in_place(&self, call: SoftcapCall<'_>) -> Result<(), EngError> {
        let mut frame = self.frame();
        frame.softcap_in_place(call)?;
        frame.drain().map(|_| ())
    }

    /// The same dispatch `repeats` times in one command buffer of a frame, for a timing test:
    /// the GPU time of the command buffer in nanoseconds.
    #[cfg(test)]
    pub(crate) fn softcap_in_place_timed(
        &self,
        call: SoftcapCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        let mut frame = self.frame_of(repeats);
        for _ in 0..repeats {
            frame.softcap_in_place(call)?;
        }
        frame.drain()
    }
}

impl Frame {
    /// Encodes one cap in place into the frame.
    pub(super) fn softcap_in_place(&mut self, call: SoftcapCall<'_>) -> Result<(), EngError> {
        let shared = self.shared();
        let plan = plan_softcap(shared, &call)?;
        let kernel = shared.kernel(KERNEL, SOURCE)?;
        let SoftcapPlan { x, params } = plan;
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&params.n.to_le_bytes());
        bytes[4..].copy_from_slice(&params.cap.to_le_bytes());
        // One thread a float, whole threadgroups.
        let threads = params.n as usize;
        let grid: Grid = [threads.div_ceil(THREADS) * THREADS, 1, 1];
        self.encode(&kernel, &[x], &bytes, grid, [THREADS, 1, 1], 1)
    }
}

/// The rules of the CUDA adapter, in its order: the cap, then the count, then the range.
fn plan_softcap(shared: &Shared, call: &SoftcapCall<'_>) -> Result<SoftcapPlan, EngError> {
    let SoftcapCall { x, n, cap } = *call;
    if !cap.is_finite() || cap <= 0.0 {
        return Err(EngError::BadRequest(format!(
            "the softcap needs a finite cap above zero, got {cap}"
        )));
    }
    if !(1..=SOFTCAP_MAX_N).contains(&n) {
        return Err(EngError::BadRequest(format!(
            "the softcap needs n in 1..={SOFTCAP_MAX_N}, got {n}"
        )));
    }
    require_range(shared, "logits", &x, n * VALUE_BYTES, RANGE_ALIGN)?;
    Ok(SoftcapPlan {
        x: x.binding(),
        params: SoftcapParams { n: n as u32, cap },
    })
}
