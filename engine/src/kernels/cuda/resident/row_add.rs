//! The element-wise residual sum of the Gemma 4 Humanizer step: one launch computing
//! out = x + y in f32.
//!
//! [CudaContext::row_add] queues the kernel on a [DeviceStream] and returns when it is
//! queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Every refusal is a BadRequest and happens before any native call: the backend is
//! usable, n is a multiple of 32 in 32..=4194304, the three ranges hold n floats at
//! 4-byte alignment, and out overlaps no byte of x or y.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::row_add::{RowAddRaw, row_add_launch};

/// One row_add call: n values in x and y, their sum in out.
pub(crate) struct RowAddCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) n: usize,
}

impl CudaContext {
    /// Queues the sum of call on stream, writing call.out.
    pub(crate) fn row_add(
        &self,
        stream: &DeviceStream,
        call: RowAddCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_row_add(&call)?;
        self.launch_on(stream, |stream_raw| row_add_launch(stream_raw, &raw))
    }

    fn plan_row_add<'b>(&self, call: &RowAddCall<'b>) -> Result<RowAddRaw<'b>, EngError> {
        let (out_raw, x_raw, y_raw) = (
            self.raw_range(&call.out)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.y)?,
        );
        let bytes = call
            .n
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("the row_add size overflows".to_owned()))?;
        if !(32..=4_194_304).contains(&call.n) || !call.n.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the row_add needs n as a multiple of 32 up to 4194304, got {}",
                call.n
            )));
        }
        require_fit("out", &call.out, bytes, 4)?;
        require_fit("x", &call.x, bytes, 4)?;
        require_fit("y", &call.y, bytes, 4)?;
        let out = Claim::of("out", &call.out, bytes);
        require_apart(&out, &Claim::of("x", &call.x, bytes))?;
        require_apart(&out, &Claim::of("y", &call.y, bytes))?;
        Ok(RowAddRaw {
            out: out_raw,
            x: x_raw,
            y: y_raw,
            n: call.n,
        })
    }
}

#[cfg(test)]
mod tests;
