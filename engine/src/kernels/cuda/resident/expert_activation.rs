//! The chunk expert stream's device-side activation: the relu-squared
//! elementwise op between the grouped up and down projections (the row
//! guard/undo is the shared resident::row_scale). Launch-only: nothing
//! allocates, copies or synchronizes. Argument errors return BadRequest
//! before any native launch.
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::expert_activation::{
    ReluSquaredRaw, relu_squared_launch,
};

/// One elementwise relu-squared: `values` f32, out disjoint from x.
pub(crate) struct ReluSquaredCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) values: usize,
}

impl CudaContext {
    /// Queues the elementwise relu-squared of call on stream, writing call.out.
    pub(crate) fn relu_squared(
        &self,
        stream: &DeviceStream,
        call: ReluSquaredCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bytes = call
            .values
            .checked_mul(4)
            .filter(|bytes| *bytes <= 1usize << 34)
            .ok_or_else(|| EngError::BadRequest("the relu-squared size overflows".to_owned()))?;
        if call.values == 0 {
            return Err(EngError::BadRequest(
                "the relu-squared needs at least one value".to_owned(),
            ));
        }
        let (out_raw, x_raw) = (self.raw_range(&call.out)?, self.raw_range(&call.x)?);
        require_fit("out", &call.out, bytes, 4)?;
        require_fit("x", &call.x, bytes, 4)?;
        require_apart(
            &Claim::of("out", &call.out, bytes),
            &Claim::of("x", &call.x, bytes),
        )?;
        let raw = ReluSquaredRaw {
            out: out_raw,
            x: x_raw,
            values: call.values,
        };
        self.launch_on(stream, |stream_raw| relu_squared_launch(stream_raw, &raw))
    }
}
