//! The fused residual add, RMSNorm and q8_1 quantize of the resident step:
//! [`CudaContext::add_rmsnorm_q8`] queues ONE kernel on a [`DeviceStream`] for the step's single
//! row (m = 1) and returns when it is queued. It allocates nothing, copies nothing and never
//! synchronizes.
//!
//! Outputs: `norm_out` holds the f32 normalized row, byte-identical to what `add_norm_pack` writes
//! on the same input (the native kernel restates `add_rmsnorm_pack_k`'s strided sum of squares, its
//! block reduction order, its `rsqrtf` and its expression `row[i] * rstd * w[i]`; the router keeps
//! reading it). `q8` holds the `block_q8_1` quantization of the same values, one warp per 32-value
//! block over the row padded to [`MATRIX_ROW_PADDING`] (512) values, the tail past `n` written as
//! zeros. The quantizer rule is the vendored `quantize_q8_1`'s (`d = amax/127`,
//! `q = amax == 0 ? 0 : roundf(x/d)`, `ds = half2(d, sum x)` with the warp butterfly sum) computed
//! in IEEE arithmetic: the native TU is in the warnings-into-errors group WITHOUT `-use_fast_math`
//! (only the vendored objects use it), so `d` is the IEEE division and the `q8` bytes are
//! bit-checkable against a host reference.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; `n` is a multiple of 32 (one q8_1 block) in `32..=32768`; `eps`
//! is finite and above 0; `h`, `delta`, `weight` and `norm_out` hold `n` floats and `q8` holds
//! [`q8_1_bytes`]`(n, 1)` bytes, with `norm_out` 16-byte aligned and the rest 4; the written
//! ranges (`h`, `norm_out`, `q8`) overlap no other range of the call (`delta` and `weight` may
//! overlap each other). A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::add_rmsnorm_q8::{
    AddNormQ8Raw, add_rmsnorm_f32_launch, add_rmsnorm_q8_launch,
};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

/// The unit of the row width: one q8_1 block of 32 values.
pub(crate) const NORM_WIDTH_UNIT: usize = 32;
/// The vendored `kern::kMaxK`.
pub(crate) const MAX_WIDTH: usize = 32_768;

/// One fused add, norm and q8_1 quantize call of the resident step's row. All ranges are device
/// ranges of this context.
///
/// - `h`: f32 `[n]`, 4-byte aligned; read, and written in place with `h + delta` when `delta` is
///   given. It is the new residual.
/// - `delta`: `None`, or f32 `[n]`, 4-byte aligned, read only. It may share bytes with `weight`.
/// - `weight`: f32 `[n]`, 4-byte aligned, read only (the RMSNorm weight).
/// - `norm_out`: f32 `[n]`, 16-byte aligned; the normalized row (the router input).
/// - `q8`: [`q8_1_bytes`]`(n, 1)` bytes, 4-byte aligned; the padded `block_q8_1` row.
/// - `eps`: finite and above 0 (GGUF `layer_norm_rms_epsilon`).
pub(crate) struct AddNormQ8Call<'b> {
    pub(crate) n: usize,
    pub(crate) eps: f32,
    pub(crate) h: DeviceRange<'b>,
    pub(crate) delta: Option<DeviceRange<'b>>,
    pub(crate) weight: DeviceRange<'b>,
    pub(crate) norm_out: DeviceRange<'b>,
    pub(crate) q8: DeviceRange<'b>,
    /// Row count: one block per row, consecutive rows, one shared weight. 1 restates the
    /// single-row call of the resident step.
    pub(crate) rows: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AddNormQ8Needs {
    pub(crate) row: usize,
    pub(crate) weight: usize,
    pub(crate) q8: usize,
}

pub(crate) fn add_rmsnorm_q8_needs(n: usize, rows: usize) -> Option<AddNormQ8Needs> {
    Some(AddNormQ8Needs {
        row: n.checked_mul(4)?.checked_mul(rows)?,
        weight: n.checked_mul(4)?,
        q8: q8_1_bytes(n, rows)?,
    })
}

impl CudaContext {
    /// Queues the fused add, RMSNorm and q8_1 quantize of `call` on `stream`, writing `norm_out`,
    /// `q8` and (when `delta` is given) `h` in place.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for `n` that is not a
    /// multiple of 32 in `32..=32768`, for `eps` not finite and above 0, for a range that is
    /// short or misaligned, and for a written range that overlaps another range of the call.
    pub(crate) fn add_rmsnorm_q8(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.add_rmsnorm_q8_sized(stream, call, 0, 0, true)
    }

    /// [`Self::add_rmsnorm_q8`] that skips the `q8` quantization and writes only `h` (when `delta`
    /// is given) and `norm_out` when `skip_q8` is true.
    pub(crate) fn add_rmsnorm_maybe_q8(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
        skip_q8: bool,
    ) -> Result<(), EngError> {
        self.add_rmsnorm_q8_sized(stream, call, 0, 0, !skip_q8)
    }

    /// [`Self::add_rmsnorm_q8`] without q8_1 quantization (writes `h` and `norm_out` only).
    pub(crate) fn add_rmsnorm_f32(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.add_rmsnorm_q8_sized(stream, call, 0, 0, false)
    }

    /// [`Self::add_rmsnorm_q8`] over rows of `h` that sit `h_stride` f32 elements apart (0 is `n`,
    /// dense); `delta`, `norm_out` and `q8` stay dense. The rows of a strided `h` are the slices
    /// one layer reads out of a wider per-row plane.
    pub(crate) fn add_rmsnorm_q8_strided(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
        h_stride: usize,
    ) -> Result<(), EngError> {
        self.add_rmsnorm_q8_sized(stream, call, h_stride, 0, true)
    }

    /// [`Self::add_rmsnorm_q8`] with 1024 threads in each row's block: a quarter of the
    /// strided rounds of the 256-thread form, several times faster per launch on a 4096-wide
    /// row, at the price of another float summation order (the last bits of the sum of
    /// squares, so of `norm_out`). For lanes gated by a tolerance, not by a byte-identical row.
    pub(crate) fn add_rmsnorm_q8_wide(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.add_rmsnorm_q8_sized(stream, call, 0, 1024, true)
    }

    fn add_rmsnorm_q8_sized(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
        h_stride: usize,
        threads: usize,
        write_q8: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_add_rmsnorm_q8(&call, h_stride, threads)?;
        self.launch_on(stream, |stream_raw| {
            if write_q8 {
                add_rmsnorm_q8_launch(stream_raw, &raw)
            } else {
                add_rmsnorm_f32_launch(stream_raw, &raw)
            }
        })
    }

    fn plan_add_rmsnorm_q8<'b>(
        &self,
        call: &AddNormQ8Call<'b>,
        h_stride: usize,
        threads: usize,
    ) -> Result<AddNormQ8Raw<'b>, EngError> {
        let h = self.raw_range(&call.h)?;
        let delta = call.delta.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let weight = self.raw_range(&call.weight)?;
        let norm_out = self.raw_range(&call.norm_out)?;
        let q8 = self.raw_range(&call.q8)?;
        if call.n < NORM_WIDTH_UNIT || call.n > MAX_WIDTH {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 takes n in {NORM_WIDTH_UNIT} to {MAX_WIDTH}, got {}",
                call.n
            )));
        }
        if !call.n.is_multiple_of(NORM_WIDTH_UNIT) {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 needs n as a multiple of {NORM_WIDTH_UNIT}, got {}",
                call.n
            )));
        }
        if !call.eps.is_finite() || call.eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the CUDA add_rmsnorm_q8 needs eps finite and above 0".to_owned(),
            ));
        }
        if call.rows == 0 || call.rows > 262_144 {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 takes rows in 1 to 262144, got {}",
                call.rows
            )));
        }
        if !matches!(threads, 0 | 256 | 1024) {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 takes 0, 256 or 1024 threads per block, got {threads}"
            )));
        }
        let need = add_rmsnorm_q8_needs(call.n, call.rows).ok_or_else(|| {
            EngError::BadRequest("the CUDA add_rmsnorm_q8 size overflows".to_owned())
        })?;
        if h_stride != 0 && h_stride < call.n {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 needs h_stride 0 or at least n, got {h_stride}"
            )));
        }
        let h_need = match h_stride {
            0 => need.row,
            stride => stride
                .checked_mul(4)
                .and_then(|row| row.checked_mul(call.rows - 1))
                .and_then(|lead| lead.checked_add(call.n * 4))
                .ok_or_else(|| {
                    EngError::BadRequest("the CUDA add_rmsnorm_q8 size overflows".to_owned())
                })?,
        };
        require_fit("h", &call.h, h_need, 4)?;
        if let Some(delta) = &call.delta {
            require_fit("delta", delta, need.row, 4)?;
        }
        require_fit("weight", &call.weight, need.weight, 4)?;
        require_fit("norm_out", &call.norm_out, need.row, 16)?;
        require_fit("q8", &call.q8, need.q8, 4)?;
        let written = [
            Claim::of("h", &call.h, h_need),
            Claim::of("norm_out", &call.norm_out, need.row),
            Claim::of("q8", &call.q8, need.q8),
        ];
        let read = [
            call.delta
                .as_ref()
                .map(|delta| Claim::of("delta", delta, need.row)),
            Some(Claim::of("weight", &call.weight, need.weight)),
        ];
        for (i, claim) in written.iter().enumerate() {
            for other in written[i + 1..].iter().chain(read.iter().flatten()) {
                require_apart(claim, other)?;
            }
        }
        Ok(AddNormQ8Raw {
            h,
            delta,
            weight,
            norm_out,
            q8,
            n: call.n,
            eps: call.eps,
            rows: call.rows,
            h_stride,
            threads,
        })
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
