//! The resident Inkling causal short convolution, launch-only: `CausalConv::apply` of the host
//! (`attn_sconv` and `mlp_sconv`) over a history that stays on the device.
//!
//! `history` packs the `taps - 1` newest past inputs, row 0 the newest, `width` floats each
//! (`native/resident/inkling_conv.h`). The launch advances that history in place to
//! `[raw input, old row 0, ..]`. `health` is left alone, or set to quiet NaN if any input,
//! weight, history value or output was not finite.

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

pub(crate) use super::super::native::resident::inkling_conv::MAX_TAPS;
use super::super::native::resident::inkling_conv::{self as native, ConvRaw};

const MAX_WIDTH: usize = 8192;

/// The ranges of one launch. `out` is either exactly `input` or disjoint from it.
pub(crate) struct ConvCall<'a> {
    pub(crate) input: DeviceRange<'a>,
    pub(crate) kernel: DeviceRange<'a>,
    pub(crate) history: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    /// The current row of a depth stream, which takes `state[c] += out[c]` (the host's
    /// add_current of the convolved row) instead of a following add launch; or absent.
    pub(crate) state: Option<DeviceRange<'a>>,
    pub(crate) health: DeviceRange<'a>,
}

fn check_shape(width: usize, taps: usize) -> Result<(), EngError> {
    if (1..=MAX_WIDTH).contains(&width) && (2..=MAX_TAPS).contains(&taps) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA Inkling conv takes 1 to {MAX_WIDTH} columns and 2 to {MAX_TAPS} taps, got \
         width {width}, taps {taps}"
    )))
}

/// `require_fit`, then the exact byte count: the extents of a launch are fully determined by
/// `width` and `taps`, so a range that holds more than the call uses is refused too.
fn check_exact(name: &'static str, range: &DeviceRange<'_>, bytes: usize) -> Result<(), EngError> {
    require_fit(name, range, bytes, align_of::<f32>())?;
    if range.bytes() == bytes {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA {name} range holds {} bytes but the call needs exactly {bytes}",
        range.bytes()
    )))
}

impl CudaContext {
    /// `CausalConv::apply` of the host, without dynamic taps: for every channel `c`,
    /// `out[c] = (acc + input[c]) * (1 / sqrt(taps + 1))` where `acc` is `0 + k[taps - 1] *
    /// input[c] + k[taps - 2] * history[0][c] + .. + k[0] * history[taps - 2][c]` in that
    /// order, every operation rounded to nearest even. `history` holds `taps - 1` rows of
    /// `width`, newest first; zero rows stand for the inputs a fresh sequence lacks.
    /// `out` may be `input` itself; the history still captures the raw input.
    pub(crate) fn inkling_conv(
        &self,
        stream: &DeviceStream,
        call: ConvCall<'_>,
        (width, taps): (usize, usize),
    ) -> Result<(), EngError> {
        check_shape(width, taps)?;
        let row = width * size_of::<f32>();
        let kernel = taps * row;
        let history = (taps - 1) * row;
        let health = size_of::<f32>();
        check_exact("Inkling conv input", &call.input, row)?;
        check_exact("Inkling conv kernel", &call.kernel, kernel)?;
        check_exact("Inkling conv history", &call.history, history)?;
        check_exact("Inkling conv output", &call.out, row)?;
        check_exact("Inkling conv health", &call.health, health)?;
        if let Some(state) = &call.state {
            check_exact("Inkling conv state", state, row)?;
        }
        let input = Claim::of("Inkling conv input", &call.input, row);
        let out = Claim::of("Inkling conv output", &call.out, row);
        let kernel = Claim::of("Inkling conv kernel", &call.kernel, kernel);
        let history = Claim::of("Inkling conv history", &call.history, history);
        let health = Claim::of("Inkling conv health", &call.health, health);
        // `out` is either exactly `input` (the same allocation at the same offset, both `row`
        // bytes by the checks above) or shares no byte with it.
        let in_place = std::ptr::eq(call.out.buffer(), call.input.buffer())
            && call.out.offset() == call.input.offset();
        if !in_place {
            require_apart(&out, &input)?;
        }
        // `history`, `kernel` and `health` share no byte with any other range.
        for other in [&kernel, &history, &health] {
            require_apart(&out, other)?;
            require_apart(&input, other)?;
        }
        require_apart(&kernel, &history)?;
        require_apart(&kernel, &health)?;
        require_apart(&history, &health)?;
        // `state` shares no byte with any other range.
        if let Some(state) = &call.state {
            let state = Claim::of("Inkling conv state", state, row);
            for other in [&input, &out, &kernel, &history, &health] {
                require_apart(&state, other)?;
            }
        }
        self.ensure_usable()?;
        let native_call = ConvRaw {
            input: self.raw_range(&call.input)?,
            kernel: self.raw_range(&call.kernel)?,
            history: self.raw_range(&call.history)?,
            out: self.raw_range(&call.out)?,
            state: call
                .state
                .as_ref()
                .map(|state| self.raw_range(state))
                .transpose()?,
            health: self.raw_range(&call.health)?,
        };
        self.launch_on(stream, |raw| {
            native::launch(raw, &native_call, [width, taps])
        })
    }
}
