use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::gemma_matvec::{GemmaMatvecRaw, gemma_matvec_launch};

/// A repacked weight plane and 1 to 8 activation rows, padded to 512 values.
/// The output is contiguous f32 [ncols][rows].
pub(crate) struct GemmaMatvecCall<'b> {
    pub(crate) w_qs: DeviceRange<'b>,
    pub(crate) w_scales: DeviceRange<'b>,
    pub(crate) x_q8: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) rows: usize,
    pub(crate) k: usize,
    pub(crate) ncols: usize,
}

impl CudaContext {
    /// Queues the gemma matvec of call on stream, writing call.y.
    pub(crate) fn gemma_matvec(
        &self,
        stream: &DeviceStream,
        call: GemmaMatvecCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gemma_matvec(&call)?;
        self.launch_on(stream, |stream_raw| gemma_matvec_launch(stream_raw, &raw))
    }

    /// Every check of a call, then the native view of its ranges.
    fn plan_gemma_matvec<'b>(
        &self,
        call: &GemmaMatvecCall<'b>,
    ) -> Result<GemmaMatvecRaw<'b>, EngError> {
        let w_qs = self.raw_range(&call.w_qs)?;
        let w_scales = self.raw_range(&call.w_scales)?;
        let x_q8 = self.raw_range(&call.x_q8)?;
        let y = self.raw_range(&call.y)?;
        if !(1..=(1usize << 20)).contains(&call.rows) {
            return Err(EngError::BadRequest(format!(
                "the gemma matvec takes rows in 1 to {}, got {}",
                1usize << 20,
                call.rows
            )));
        }
        if call.k < 32 || call.k > 65_536 || !call.k.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the gemma matvec needs k a multiple of 32 in 32 to 65536, got {}",
                call.k
            )));
        }
        if !(1..=8).contains(&call.ncols) {
            return Err(EngError::BadRequest(format!(
                "the gemma matvec takes 1 to 8 columns, got {}",
                call.ncols
            )));
        }
        let blocks = call.k / 32;
        let qs = call
            .rows
            .checked_mul(call.k)
            .ok_or_else(|| EngError::BadRequest("the gemma matvec size overflows".to_owned()))?;
        let scales = call
            .rows
            .checked_mul(blocks)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| EngError::BadRequest("the gemma matvec size overflows".to_owned()))?;
        let y_bytes = call
            .rows
            .checked_mul(call.ncols)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the gemma matvec size overflows".to_owned()))?;
        let stride = call.k.next_multiple_of(512) / 32;
        let act = ((call.ncols - 1) * stride + blocks) * 36;
        require_fit("w_qs", &call.w_qs, qs, 16)?;
        require_fit("w_scales", &call.w_scales, scales, 2)?;
        require_fit("x_q8", &call.x_q8, act, 4)?;
        require_fit("y", &call.y, y_bytes, 4)?;
        let written = Claim::of("y", &call.y, y_bytes);
        for (name, range, bytes) in [
            ("w_qs", &call.w_qs, qs),
            ("w_scales", &call.w_scales, scales),
            ("x_q8", &call.x_q8, act),
        ] {
            require_apart(&written, &Claim::of(name, range, bytes))?;
        }
        Ok(GemmaMatvecRaw {
            w_qs,
            w_scales,
            x_q8,
            y,
            rows: call.rows,
            k: call.k,
            ncols: call.ncols,
        })
    }
}

#[cfg(test)]
mod tests;
