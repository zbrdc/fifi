//! F32 row operations: constant scaling, per-row power-of-two normalization and
//! restoration, and BF16-precision publication in F32 working storage.
//!
//! [CudaContext::row_scale] queues the kernel on a [DeviceStream] and returns when it is
//! queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! Argument errors return BadRequest before a native launch. Validation checks backend
//! usability, buffer ownership, and checked size arithmetic first. It then checks a finite
//! scale, m in 1..=65536, n in 1..=2^22, and m*n <= 2^32. Both ranges must hold m*n
//! floats at 4-byte alignment, and out must be disjoint from x. Stream ownership is checked
//! after range planning. A rejected call queues no work. Existing terminal errors keep
//! their error mapping.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::row_scale::{RowScaleRaw, row_scale_launch};

/// One row-scale call: m rows of n f32 values in x, the scaled copy in out.
pub(crate) struct RowScaleCall<'b> {
    pub(crate) out: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) scale: f32,
    pub(crate) m: usize,
    pub(crate) n: usize,
}

impl CudaContext {
    /// Queues the row scale of call on stream, writing call.out.
    ///
    /// Returns BadRequest for a stream or buffer of another adapter, a non-finite
    /// scale, m or n outside their ranges, a short or misaligned range, and an out
    /// that overlaps x.
    pub(crate) fn row_scale(
        &self,
        stream: &DeviceStream,
        call: RowScaleCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_row_scale(&call)?;
        self.launch_on(stream, |stream_raw| row_scale_launch(stream_raw, &raw))
    }

    /// Every check of a row scale, then the native view of its ranges.
    fn plan_row_scale<'b>(&self, call: &RowScaleCall<'b>) -> Result<RowScaleRaw<'b>, EngError> {
        let (out_raw, x_raw) = (self.raw_range(&call.out)?, self.raw_range(&call.x)?);
        let bytes = call
            .m
            .checked_mul(call.n)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("the row scale size overflows".to_owned()))?;
        if !call.scale.is_finite() {
            return Err(EngError::BadRequest(format!(
                "the row scale needs a finite scale, got {}",
                call.scale
            )));
        }
        if !(1..=65_536).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the row scale takes 1 to 65536 rows, got {}",
                call.m
            )));
        }
        if !(1..=(1 << 22)).contains(&call.n) {
            return Err(EngError::BadRequest(format!(
                "the row scale takes 1 to 4194304 values per row, got {}",
                call.n
            )));
        }
        if bytes / 4 > (1usize << 32) {
            return Err(EngError::BadRequest(
                "the row scale count exceeds 2^32".to_owned(),
            ));
        }
        require_fit("out", &call.out, bytes, 4)?;
        require_fit("x", &call.x, bytes, 4)?;
        require_apart(
            &Claim::of("out", &call.out, bytes),
            &Claim::of("x", &call.x, bytes),
        )?;
        Ok(RowScaleRaw {
            out: out_raw,
            x: x_raw,
            scale: call.scale,
            m: call.m,
            n: call.n,
        })
    }
}

/// Per-row power-of-two range guard and undo; publish only after stream completion
/// and checking `flag & 1 == 0` in the caller-owned flag. This call never clears that flag.
pub(crate) struct RowPow2Call<'b> {
    pub(crate) x: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) scales: DeviceRange<'b>,
    pub(crate) flag: DeviceRange<'b>,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) mode: RowPow2Mode,
}

#[derive(Clone, Copy)]
pub(crate) enum RowPow2Mode {
    Normalize,
    Restore,
}

impl CudaContext {
    pub(crate) fn row_pow2(
        &self,
        stream: &DeviceStream,
        call: RowPow2Call<'_>,
    ) -> Result<(), EngError> {
        use crate::kernels::cuda::native::resident::row_scale as ffi;
        self.ensure_usable()?;
        let (values, scales) = ffi::row_pow2_geometry(call.rows, call.columns)
            .ok_or_else(|| EngError::BadRequest("row pow2: invalid geometry or overflow".into()))?;
        let ranges = [&call.x, &call.out, &call.scales, &call.flag];
        let needs = [values, values, scales, 4];
        for i in 0..ranges.len() {
            self.raw_range(ranges[i])?;
            require_fit("row pow2 region", ranges[i], needs[i], 4)?;
            for j in 0..i {
                require_apart(
                    &Claim::of("row pow2 region", ranges[i], needs[i]),
                    &Claim::of("row pow2 region", ranges[j], needs[j]),
                )?;
            }
        }
        let raw = ffi::RowPow2Raw {
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.out)?,
            scales: self.raw_range(&call.scales)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.rows,
            columns: call.columns,
            restore: matches!(call.mode, RowPow2Mode::Restore),
        };
        self.launch_on(stream, move |stream_raw| {
            ffi::row_pow2_launch(stream_raw, &raw)
        })
    }
}

/// RN-even BF16 publication in F32 working storage. A finite input
/// can overflow BF16: consume only after completion and caller flag & 1 == 0.
pub(crate) struct Bf16PublishCall<'b> {
    pub(crate) x: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) flag: DeviceRange<'b>,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl CudaContext {
    pub(crate) fn publish_bf16_rows(
        &self,
        stream: &DeviceStream,
        call: Bf16PublishCall<'_>,
    ) -> Result<(), EngError> {
        use crate::kernels::cuda::native::resident::row_scale as ffi;
        self.ensure_usable()?;
        let (values, _) = ffi::row_pow2_geometry(call.rows, call.columns).ok_or_else(|| {
            EngError::BadRequest("BF16 publication: invalid geometry or overflow".into())
        })?;
        let ranges = [&call.x, &call.out, &call.flag];
        let needs = [values, values, 4];
        for i in 0..ranges.len() {
            self.raw_range(ranges[i])?;
            require_fit("BF16 publication region", ranges[i], needs[i], 4)?;
            for j in 0..i {
                require_apart(
                    &Claim::of("BF16 publication region", ranges[i], needs[i]),
                    &Claim::of("BF16 publication region", ranges[j], needs[j]),
                )?;
            }
        }
        let raw = ffi::Bf16PublishRaw {
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.rows,
            columns: call.columns,
        };
        self.launch_on(stream, move |raw_stream| {
            ffi::bf16_publish_launch(raw_stream, &raw)
        })
    }
}

#[cfg(test)]
mod bf16_tests;
#[cfg(test)]
mod guard_tests;
#[cfg(test)]
mod tests;
