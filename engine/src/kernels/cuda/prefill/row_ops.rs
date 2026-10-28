//! Shared F32 row math and gather operations. These calls allocate, copy and
//! synchronize nothing; callers retain owners and check the flag after completion.
//! BF16 publication remains a separate shared row_scale operation.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::row_ops::{
    self as ffi, RowGatherRaw, RowMulRaw, RowSumRaw, RowUnaryRaw,
};
use crate::kernels::cuda::native::resident::row_scale::row_pow2_geometry;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RowShape {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl RowShape {
    pub(crate) fn bytes(self) -> Option<usize> {
        row_pow2_geometry(self.rows, self.columns).map(|(data, _)| data)
    }
}

pub(crate) struct RowUnaryCall<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) shape: RowShape,
}

pub(crate) struct RowMulCall<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) rhs: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) shape: RowShape,
    pub(crate) rhs_columns: usize,
}

pub(crate) struct RowGatherCall<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) indices: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) shape: RowShape,
    pub(crate) source_rows: usize,
}

pub(crate) struct RowSumCall<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) shape: RowShape,
    pub(crate) rank: usize,
}

fn geometry_error() -> EngError {
    EngError::BadRequest("row operation shape is outside the shared row geometry".into())
}

fn check_regions(
    context: &CudaContext,
    regions: &[(&'static str, &DeviceRange<'_>, usize, bool)],
) -> Result<(), EngError> {
    for (index, &(name, range, bytes, written)) in regions.iter().enumerate() {
        context.raw_range(range)?;
        require_fit(name, range, bytes, 4)?;
        for &(other_name, other_range, other_bytes, other_written) in &regions[..index] {
            if written || other_written {
                require_apart(
                    &Claim::of(name, range, bytes),
                    &Claim::of(other_name, other_range, other_bytes),
                )?;
            }
        }
    }
    Ok(())
}

impl CudaContext {
    /// F32 sigmoid; nonfinite raw inputs set flag bit0 even if sigmoid would hide them.
    pub(crate) fn row_sigmoid(
        &self,
        stream: &DeviceStream,
        call: RowUnaryCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bytes = call.shape.bytes().ok_or_else(geometry_error)?;
        check_regions(
            self,
            &[
                ("x", &call.x, bytes, false),
                ("out", &call.out, bytes, true),
                ("flag", &call.flag, 4, true),
            ],
        )?;
        let raw = RowUnaryRaw {
            x: self.raw_range(&call.x)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.shape.rows,
            columns: call.shape.columns,
        };
        self.launch_on(stream, |stream| ffi::row_sigmoid_launch(stream, &raw))
    }

    /// Elementwise F32 product, with either a full RHS or one RHS value per row.
    pub(crate) fn row_multiply(
        &self,
        stream: &DeviceStream,
        call: RowMulCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (bytes, rhs) =
            ffi::multiply_bytes(call.shape.rows, call.shape.columns, call.rhs_columns)
                .ok_or_else(geometry_error)?;
        check_regions(
            self,
            &[
                ("x", &call.x, bytes, false),
                ("rhs", &call.rhs, rhs, false),
                ("out", &call.out, bytes, true),
                ("flag", &call.flag, 4, true),
            ],
        )?;
        let raw = RowMulRaw {
            x: self.raw_range(&call.x)?,
            rhs: self.raw_range(&call.rhs)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.shape.rows,
            columns: call.shape.columns,
            rhs_columns: call.rhs_columns,
        };
        self.launch_on(stream, |stream| ffi::row_mul_launch(stream, &raw))
    }

    /// Gathers contiguous rows by device i32 IDs; bad IDs set the shared bound mask.
    pub(crate) fn row_gather(
        &self,
        stream: &DeviceStream,
        call: RowGatherCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (source, ids, target) =
            ffi::gather_bytes(call.shape.rows, call.shape.columns, call.source_rows)
                .ok_or_else(geometry_error)?;
        check_regions(
            self,
            &[
                ("x", &call.x, source, false),
                ("indices", &call.indices, ids, false),
                ("out", &call.out, target, true),
                ("flag", &call.flag, 4, true),
            ],
        )?;
        let raw = RowGatherRaw {
            x: self.raw_range(&call.x)?,
            indices: self.raw_range(&call.indices)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.shape.rows,
            columns: call.shape.columns,
            source_rows: call.source_rows,
        };
        self.launch_on(stream, |stream| ffi::row_gather_launch(stream, &raw))
    }

    /// Sums original selected-rank order in F32, without per-add BF16 publication.
    pub(crate) fn row_rank_sum(
        &self,
        stream: &DeviceStream,
        call: RowSumCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (source, target) = ffi::rank_sum_bytes(call.shape.rows, call.shape.columns, call.rank)
            .ok_or_else(geometry_error)?;
        check_regions(
            self,
            &[
                ("x", &call.x, source, false),
                ("out", &call.out, target, true),
                ("flag", &call.flag, 4, true),
            ],
        )?;
        let raw = RowSumRaw {
            x: self.raw_range(&call.x)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.shape.rows,
            columns: call.shape.columns,
            rank: call.rank,
        };
        self.launch_on(stream, |stream| ffi::row_rank_sum_launch(stream, &raw))
    }
}

#[cfg(test)]
mod tests;
