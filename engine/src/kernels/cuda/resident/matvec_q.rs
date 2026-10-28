//! The dense MMVQ matvec against native GGUF-layout weights (MATVEC-PAR S1 c3b, decision 0047,
//! stage 1): [`CudaContext::matvec_q`] queues one launch of the vendored `mul_mat_vec_q` kernel
//! through the V-MV1 host entry and returns when it is queued. It allocates nothing, copies
//! nothing and never synchronizes.
//!
//! `w` holds the file's native blocks (`[rows][k / block_size]`, one [`QuantType`] for the whole
//! stack), `x_q8` the padded `quantize_q8_1` output of `ncols` activation rows, `y` the f32
//! result with column `j` at `y + j * rows`. The kernel computes column `j` from `x_q8` row `j`
//! alone; the row-invariance contract (decision 0037: column `j` of an `ncols` launch is
//! bit-identical to the 1-column launch) is pinned by test 4.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; `k` is a multiple of the type's block size; `rows` is a positive multiple of 4
//! ([`ROWS_MULTIPLE`]); `ncols` is in `1..=8`; `w`, `x_q8` and `y` hold the bytes the call uses and
//! are 4-byte aligned; `y` overlaps neither `w` nor `x_q8`. A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
#[cfg(test)]
use crate::kernels::cuda::native;
use crate::kernels::cuda::native::resident::matvec_q::{MatvecRaw, matvec_q_launch};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
#[cfg(test)]
use crate::kernels::cuda::{ADAPTER_OPERATION, TERMINAL, native_error};

/// The four weight formats the V-MV1-trimmed MMVQ instantiates (Qwen3.6 and Mellum). The
/// discriminants are the `ggml_type` values of the shim `ggml.h` (`GGML_TYPE_Q4_K = 12`,
/// `Q5_0 = 6`, `Q6_K = 14`, `Q8_0 = 8`), not numbers from memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantType {
    Q4K,
    Q50,
    Q6K,
    Q80,
}

impl QuantType {
    /// The `ggml_type` value the vendored code switches on.
    pub(crate) fn ggml(self) -> i32 {
        match self {
            Self::Q4K => 12,
            Self::Q50 => 6,
            Self::Q6K => 14,
            Self::Q80 => 8,
        }
    }

    /// Values per block: 256 for the K-quants, 32 for the flat formats.
    pub(crate) fn block_size(self) -> usize {
        match self {
            Self::Q4K | Self::Q6K => 256,
            Self::Q50 | Self::Q80 => 32,
        }
    }

    /// Bytes per block: Q4_K 144, Q5_0 22, Q6_K 210, Q8_0 34 (ggml-common.h static_asserts).
    pub(crate) fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q50 => 22,
            Self::Q6K => 210,
            Self::Q80 => 34,
        }
    }

    /// Bytes of the native blocks of `rows` rows of `k` values. `None` on overflow or when `k`
    /// is not a multiple of the block size.
    pub(crate) fn weight_bytes(self, rows: usize, k: usize) -> Option<usize> {
        if !k.is_multiple_of(self.block_size()) {
            return None;
        }
        rows.checked_mul(k.checked_div(self.block_size())?)?
            .checked_mul(self.block_bytes())
    }
}

/// Rows must be a multiple of this: the MMVQ kernel loads the weight rows of a block as a group of
/// up to 4 and never receives the row count, so a matrix of other height would be read past its
/// end. A limitation until a guarded kernel exists (KUBDAN-MSG-64 item 4, note for S3); every
/// Qwen3.6 shape is a multiple of 4.
pub(crate) const ROWS_MULTIPLE: usize = 4;

/// One dense matvec: `y[j * rows + r] = sum_k x_q8[j][k] * w[r][k]`.
pub(crate) struct MatvecCall<'b> {
    pub(crate) ty: QuantType,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x_q8: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
}

/// Bytes of the three ranges a dense matvec uses.
struct MatvecNeeds {
    w: usize,
    x_q8: usize,
    y: usize,
}

/// What a dense matvec of `rows` x `k` against `ncols` activation rows reads and writes: the native
/// blocks of `w`, the padded q8_1 rows of `x_q8` and the `rows * ncols` floats of `y`. `None` on
/// overflow or when `k` is off the type's block grid.
fn matvec_needs(ty: QuantType, k: usize, rows: usize, ncols: usize) -> Option<MatvecNeeds> {
    Some(MatvecNeeds {
        w: ty.weight_bytes(rows, k)?,
        x_q8: q8_1_bytes(k, ncols)?,
        y: rows.checked_mul(ncols)?.checked_mul(4)?,
    })
}

impl CudaContext {
    /// Queues the dense matvec of `call` on `stream`, writing `call.y`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for `k` that is not a
    /// multiple of the type's block size, for `rows` 0 or not a multiple of 4, for `ncols` outside
    /// `1..=8`, for a `w`, `x_q8` or `y` range that is short or misaligned, and for a `y` that
    /// overlaps `w` or `x_q8`.
    pub(crate) fn matvec_q(
        &self,
        stream: &DeviceStream,
        call: MatvecCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_matvec(&call)?;
        self.launch_on(stream, |stream_raw| matvec_q_launch(stream_raw, &raw))
    }

    /// Every check of a matvec, then the native view of its ranges.
    fn plan_matvec<'b>(&self, call: &MatvecCall<'b>) -> Result<MatvecRaw<'b>, EngError> {
        let (w_raw, x_raw, y_raw) = (
            self.raw_range(&call.w)?,
            self.raw_range(&call.x_q8)?,
            self.raw_range(&call.y)?,
        );
        if !call.k.is_multiple_of(call.ty.block_size()) || call.k == 0 {
            return Err(EngError::BadRequest(format!(
                "the CUDA matvec needs k as a multiple of {}, got {}",
                call.ty.block_size(),
                call.k
            )));
        }
        if call.rows == 0 || !call.rows.is_multiple_of(ROWS_MULTIPLE) {
            return Err(EngError::BadRequest(format!(
                "the CUDA matvec needs rows as a positive multiple of {ROWS_MULTIPLE}, got {}",
                call.rows
            )));
        }
        if !(1..=8).contains(&call.ncols) {
            return Err(EngError::BadRequest(format!(
                "the CUDA matvec takes 1 to 8 columns, got {}",
                call.ncols
            )));
        }
        let need = matvec_needs(call.ty, call.k, call.rows, call.ncols)
            .ok_or_else(|| EngError::BadRequest("the CUDA matvec size overflows".to_owned()))?;
        require_fit("w", &call.w, need.w, 4)?;
        require_fit("x_q8", &call.x_q8, need.x_q8, 4)?;
        require_fit("y", &call.y, need.y, 4)?;
        let written = Claim::of("y", &call.y, need.y);
        require_apart(&written, &Claim::of("w", &call.w, need.w))?;
        require_apart(&written, &Claim::of("x_q8", &call.x_q8, need.x_q8))?;
        Ok(MatvecRaw {
            ty: call.ty,
            w: w_raw,
            x_q8: x_raw,
            y: y_raw,
            k: call.k,
            rows: call.rows,
            ncols: call.ncols,
        })
    }

    /// Test support of the seam-1 microbench (test 11): records a raw timing event behind the
    /// work queued on `stream` and returns its opaque handle, for
    /// [`native::resident::matvec_q::event_elapsed_ms_raw`]. The handle is destroyed by
    /// [`Self::bench_destroy`]. This rides `launch_on` so the raw stream never leaves the native
    /// tree; it queues nothing but the event record.
    #[cfg(test)]
    pub(crate) fn bench_mark(&self, stream: &DeviceStream) -> Result<usize, EngError> {
        self.launch_on(stream, |raw| {
            native::resident::matvec_q::event_record_raw(raw).map(|event| event as usize)
        })
    }

    /// Test support: the milliseconds between two `bench_mark` handles.
    #[cfg(test)]
    pub(crate) fn bench_elapsed_ms(&self, start: usize, end: usize) -> Result<f32, EngError> {
        native::resident::matvec_q::event_elapsed_ms_raw(start as *mut _, end as *mut _)
            .map_err(|failure| native_error(&TERMINAL, ADAPTER_OPERATION, failure.status))
    }

    /// Test support: destroys one `bench_mark` handle.
    #[cfg(test)]
    pub(crate) fn bench_destroy(&self, event: usize) -> Result<(), EngError> {
        native::resident::matvec_q::event_destroy_raw(event as *mut _)
            .map_err(|failure| native_error(&TERMINAL, ADAPTER_OPERATION, failure.status))
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
