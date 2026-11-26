//! The dense MMVQ matvec against native GGUF-layout weights: [`CudaContext::matvec_q`] queues one
//! launch of the vendored `mul_mat_vec_q` kernel through the vendored host entry and returns when
//! it is queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! `w` holds the file's native blocks (`[rows][k / block_size]`, one [`QuantType`] for the whole
//! stack), `x_q8` the padded `quantize_q8_1` output of `ncols` activation rows, `y` the f32 result
//! with column `j` at `y + j * rows`. The kernel computes column `j` from `x_q8` row `j` alone; the
//! row-invariance contract (column `j` of an `ncols` launch is bit-identical to the 1-column
//! launch) is pinned by test 4.
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
use crate::kernels::cuda::native::resident::matvec_gguf::{MatvecGgufRaw, matvec_gguf_launch};
use crate::kernels::cuda::native::resident::matvec_q::{MatvecRaw, matvec_q_launch};
use crate::kernels::cuda::native::resident::matvec_q3k::matvec_q3k_launch;
use crate::kernels::cuda::native::resident::matvec_q5k::matvec_q5k_launch;
use crate::kernels::cuda::native::resident::matvec_q40::matvec_q40_launch;
use crate::kernels::cuda::native::resident::matvec_q80::matvec_q80_launch;
use crate::weights::GgufType;

use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
#[cfg(test)]
use crate::kernels::cuda::{ADAPTER_OPERATION, TERMINAL, native_error};

/// The weight formats the matvec serves; [`QuantType::ggml`] is each one's `ggml_type`, the GGUF
/// tensor type id. The dense matvec runs Q4_K, Q5_0, Q6_K and Q8_0 on the vendored MMVQ (Qwen3.6
/// and Mellum), Q5_K and Q3_K on the in-house kernels (`native/resident/matvec_q5k.cu`,
/// `native/resident/matvec_q3k.cu`), and `Iq4Xs`, `Iq4Nl`, `Iq2S`, `Iq3Xxs`, `Iq2Xxs`, `Q20`,
/// `Iq2Xs`, `Iq3S`, `Q2K`, `Iq1S` and `Iq1M` on the in-house codebook matvec
/// (`native/resident/matvec_gguf.cu`), which also has the codebook arms of the dequantizing
/// GEMM. The routed-expert matvec has a vendored case for every format (`matvec_expert`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantType {
    Q4K,
    Q50,
    Q6K,
    Q80,
    Q5K,
    Q40,
    Q3K,
    Iq4Xs,
    Iq4Nl,
    Iq2S,
    Iq3Xxs,
    Iq2Xxs,
    Q20,
    Iq2Xs,
    Iq3S,
    Q2K,
    Iq1S,
    Iq1M,
}

impl QuantType {
    /// The GGUF tensor type of the blocks this type reads.
    pub(crate) fn gguf(self) -> GgufType {
        match self {
            Self::Q4K => GgufType::Q4K,
            Self::Q50 => GgufType::Q50,
            Self::Q6K => GgufType::Q6K,
            Self::Q80 => GgufType::Q80,
            Self::Q5K => GgufType::Q5K,
            Self::Q40 => GgufType::Q40,
            Self::Q3K => GgufType::Q3K,
            Self::Iq4Xs => GgufType::Iq4Xs,
            Self::Iq4Nl => GgufType::Iq4Nl,
            Self::Iq2S => GgufType::Iq2S,
            Self::Iq3Xxs => GgufType::Iq3Xxs,
            Self::Iq2Xxs => GgufType::Iq2Xxs,
            Self::Q20 => GgufType::Q20,
            Self::Iq2Xs => GgufType::Iq2Xs,
            Self::Iq3S => GgufType::Iq3S,
            Self::Q2K => GgufType::Q2K,
            Self::Iq1S => GgufType::Iq1S,
            Self::Iq1M => GgufType::Iq1M,
        }
    }

    /// The `ggml_type` value the vendored code switches on: the GGUF tensor type id, which the
    /// vendored ggml.h shim's enum values are. A native entry refuses a type it has no case for.
    pub(crate) fn ggml(self) -> i32 {
        i32::try_from(self.gguf().id()).expect("GGUF type ids are small")
    }

    /// True for the types the in-house codebook kernels (`matvec_gguf.cu` and the codebook arms
    /// of `gemm_dq.cu`) serve: every IQ type, Q2_0 and Q2_K.
    pub(crate) fn is_codebook(self) -> bool {
        matches!(
            self,
            Self::Iq4Xs
                | Self::Iq4Nl
                | Self::Iq2S
                | Self::Iq3Xxs
                | Self::Iq2Xxs
                | Self::Q20
                | Self::Iq2Xs
                | Self::Iq3S
                | Self::Q2K
                | Self::Iq1S
                | Self::Iq1M
        )
    }

    /// True for the types whose prefill GEMM is the in-house dequantizing GEMM, dense and routed:
    /// it reads the f32 activation rows directly, so no q8_1 quantize pass feeds the role and it
    /// imposes no q8_1 layout on a buffer it shares with vendored siblings.
    pub(crate) fn uses_dequantizing_gemm(self) -> bool {
        matches!(self, Self::Q5K | Self::Q40 | Self::Q3K) || self.is_codebook()
    }

    /// True for the types whose routed-expert matvec runs the in-house window kernel
    /// (`matvec_gguf.cu`). The vendored MMVQ has a case for every type and takes the four it
    /// served first (Q4_K, Q5_0, Q6_K, Q8_0) and the six of the Inkling 2 Mini GGUFs' experts
    /// (IQ4_NL, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, Q2_0), where it measured faster on the 3090:
    /// a whole decode token 6% faster on the IQ4_NL model and 5% on the IQ1_M one. The other
    /// types stay on the window kernel until a measurement moves them; [`ExpertKernel::of`]
    /// applies the route.
    ///
    /// [`ExpertKernel::of`]: crate::kernels::cuda::resident::matvec_expert::ExpertKernel::of
    pub(crate) fn expert_uses_window(self) -> bool {
        !matches!(
            self,
            Self::Q4K
                | Self::Q50
                | Self::Q6K
                | Self::Q80
                | Self::Iq4Nl
                | Self::Iq1S
                | Self::Iq2Xxs
                | Self::Iq2Xs
                | Self::Iq2S
                | Self::Q20
        )
    }

    pub(crate) fn block_size(self) -> usize {
        match self {
            Self::Q4K
            | Self::Q6K
            | Self::Q5K
            | Self::Q3K
            | Self::Iq4Xs
            | Self::Iq2S
            | Self::Iq3Xxs
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3S
            | Self::Q2K
            | Self::Iq1S
            | Self::Iq1M => 256,
            Self::Q20 => 64,
            Self::Q50 | Self::Q80 | Self::Q40 | Self::Iq4Nl => 32,
        }
    }

    /// Bytes per block: Q4_K 144, Q5_0 22, Q6_K 210, Q8_0 34 (ggml-common.h static_asserts);
    /// Q5_K 176 (d, dmin, 12 scale bytes, 32 qh bytes, 128 qs bytes); Q4_0 18 (f16 scale and
    /// 16 code bytes); Q3_K 110 (32 hmask, 64 qs, 12 packed 6-bit scales, f16 d);
    /// IQ4_XS 136 (f16 d, u16 scales_h, scales_l[4], 128 qs); IQ4_NL 18 and Q2_0 18 (f16 d and
    /// 16 code bytes); IQ2_XXS 66, IQ2_XS 74, IQ2_S 82, IQ3_XXS 98, IQ3_S 110, Q2_K 84, IQ1_S 50, IQ1_M 56 (the block
    /// layouts of `weights/gguf.rs`, the CPU oracle's sizes).
    pub(crate) fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q50 => 22,
            Self::Q6K => 210,
            Self::Q80 => 34,
            Self::Q5K => 176,
            Self::Q40 => 18,
            Self::Q3K => 110,
            Self::Iq4Xs => 136,
            Self::Iq4Nl | Self::Q20 => 18,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq3S => 110,
            Self::Q2K => 84,
            Self::Iq1S => 50,
            Self::Iq1M => 56,
        }
    }

    /// The kernel type of a file's tensor type, `None` for a type without a native kernel.
    pub(crate) fn from_gguf(ty: GgufType) -> Option<Self> {
        Some(match ty {
            GgufType::Q4K => Self::Q4K,
            GgufType::Q50 => Self::Q50,
            GgufType::Q6K => Self::Q6K,
            GgufType::Q80 => Self::Q80,
            GgufType::Q5K => Self::Q5K,
            GgufType::Q40 => Self::Q40,
            GgufType::Q3K => Self::Q3K,
            GgufType::Iq4Xs => Self::Iq4Xs,
            GgufType::Iq4Nl => Self::Iq4Nl,
            GgufType::Iq2S => Self::Iq2S,
            GgufType::Iq3Xxs => Self::Iq3Xxs,
            GgufType::Iq2Xxs => Self::Iq2Xxs,
            GgufType::Q20 => Self::Q20,
            GgufType::Iq2Xs => Self::Iq2Xs,
            GgufType::Iq3S => Self::Iq3S,
            GgufType::Q2K => Self::Q2K,
            GgufType::Iq1S => Self::Iq1S,
            GgufType::Iq1M => Self::Iq1M,
            _ => return None,
        })
    }

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
/// end. A limitation until a guarded kernel exists; every Qwen3.6 shape is a multiple of 4.
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

struct MatvecNeeds {
    w: usize,
    x_q8: usize,
    y: usize,
}

fn matvec_needs(ty: QuantType, k: usize, rows: usize, ncols: usize) -> Option<MatvecNeeds> {
    Some(MatvecNeeds {
        w: ty.weight_bytes(rows, k)?,
        x_q8: q8_1_bytes(k, ncols)?,
        y: rows.checked_mul(ncols)?.checked_mul(4)?,
    })
}

impl CudaContext {
    pub(crate) fn matvec_q(
        &self,
        stream: &DeviceStream,
        call: MatvecCall<'_>,
    ) -> Result<(), EngError> {
        self.matvec_q_impl(stream, call, false)
    }

    /// The column-invariant contract: column `j` of any launch is bit-identical to a
    /// one-column launch over the same bytes, whatever `ncols` is. The vendored MMVQ
    /// launcher does not promise that past four columns (its launch geometry changes with
    /// the column count), so the arms it still serves — Q4_K, Q5_0 and Q6_K — launch per
    /// column when the batch is wider than four, which is the documented mitigation of
    /// this module. That costs those arms one launch per column past four (an
    /// eight-column batch pays eight sequential launches instead of one, with no shared
    /// read of the weight rows across columns); the in-house arms batch at every width.
    /// Q8_0 under the contract takes the in-house warp-per-row kernel; Q5_K,
    /// Q4_0 and Q3_K are in-house and column-invariant at every width. Callers whose
    /// reuse promise needs a token to compute the same thing whatever batch it rode in
    /// (a prefill batch vs a decode step) select this entry; the default entry keeps the vendored
    /// numerics, so served defaults do not move.
    pub(crate) fn matvec_q_invariant(
        &self,
        stream: &DeviceStream,
        call: MatvecCall<'_>,
    ) -> Result<(), EngError> {
        if matches!(call.ty, QuantType::Q4K | QuantType::Q50 | QuantType::Q6K) && call.ncols > 4 {
            let row_bytes = q8_1_bytes(call.k, 1).ok_or_else(|| {
                EngError::BadRequest("the CUDA matvec activation size overflows".to_owned())
            })?;
            let y_bytes = call.rows.checked_mul(4).ok_or_else(|| {
                EngError::BadRequest("the CUDA matvec out size overflows".to_owned())
            })?;
            for col in 0..call.ncols {
                let single = MatvecCall {
                    ty: call.ty,
                    w: call.w,
                    x_q8: call
                        .x_q8
                        .buffer()
                        .range(call.x_q8.offset() + col * row_bytes, row_bytes)?,
                    y: call
                        .y
                        .buffer()
                        .range(call.y.offset() + col * y_bytes, y_bytes)?,
                    k: call.k,
                    rows: call.rows,
                    ncols: 1,
                };
                self.matvec_q_impl(stream, single, true)?;
            }
            return Ok(());
        }
        self.matvec_q_impl(stream, call, true)
    }

    fn matvec_q_impl(
        &self,
        stream: &DeviceStream,
        call: MatvecCall<'_>,
        invariant: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_matvec(&call, invariant)?;
        let row_bytes = q8_1_bytes(call.k, 1).expect("checked in plan_matvec");
        self.launch_on(stream, |stream_raw| match call.ty {
            // The in-house Q5_K kernel: its own ABI, native blocks, whole rows.
            QuantType::Q5K => matvec_q5k_launch(
                stream_raw,
                &crate::kernels::cuda::native::resident::matvec_q5k::MatvecQ5kRaw {
                    w: raw.w,
                    x_q8: raw.x_q8,
                    y: raw.y,
                    k: raw.k,
                    rows: raw.rows,
                    ncols: raw.ncols,
                },
                row_bytes,
            ),
            QuantType::Q40 => matvec_q40_launch(
                stream_raw,
                &crate::kernels::cuda::native::resident::matvec_q40::MatvecQ40Raw {
                    w: raw.w,
                    x_q8: raw.x_q8,
                    y: raw.y,
                    k: raw.k,
                    rows: raw.rows,
                    ncols: raw.ncols,
                },
                row_bytes,
            ),
            // The in-house Q3_K kernel: its own ABI, native 110-byte blocks, whole rows.
            QuantType::Q3K => matvec_q3k_launch(
                stream_raw,
                &crate::kernels::cuda::native::resident::matvec_q3k::MatvecQ3kRaw {
                    w: raw.w,
                    x_q8: raw.x_q8,
                    y: raw.y,
                    k: raw.k,
                    rows: raw.rows,
                    ncols: raw.ncols,
                },
                row_bytes,
            ),
            // The in-house codebook matvec: its own ABI, native blocks, whole rows, the
            // type's lookup table passed to the kernel by value.
            ty if ty.is_codebook() => matvec_gguf_launch(
                stream_raw,
                &MatvecGgufRaw {
                    ty,
                    w: raw.w,
                    x_q8: raw.x_q8,
                    y: raw.y,
                    k: raw.k,
                    rows: raw.rows,
                    ncols: raw.ncols,
                },
                row_bytes,
            ),
            // The in-house Q8_0 kernel under the column-invariant contract: its own ABI,
            // native 34-byte blocks, whole rows, lane arithmetics independent of ncols.
            // Without the contract the vendored launcher keeps serving Q8_0, so served
            // defaults do not move.
            QuantType::Q80 if invariant => matvec_q80_launch(
                stream_raw,
                &crate::kernels::cuda::native::resident::matvec_q80::MatvecQ80Raw {
                    w: raw.w,
                    x_q8: raw.x_q8,
                    y: raw.y,
                    k: raw.k,
                    rows: raw.rows,
                    ncols: raw.ncols,
                },
                row_bytes,
            ),
            _ => matvec_q_launch(stream_raw, &raw),
        })
    }

    pub(super) fn plan_matvec<'b>(
        &self,
        call: &MatvecCall<'b>,
        invariant: bool,
    ) -> Result<MatvecRaw<'b>, EngError> {
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
        if call.rows == 0
            || (!(matches!(call.ty, QuantType::Q5K | QuantType::Q40 | QuantType::Q3K)
                || call.ty.is_codebook()
                || (call.ty == QuantType::Q80 && invariant))
                && !call.rows.is_multiple_of(ROWS_MULTIPLE))
        {
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

    /// Test support of the matvec microbench (test 11): records a raw timing event behind the
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

    #[cfg(test)]
    pub(crate) fn bench_elapsed_ms(&self, start: usize, end: usize) -> Result<f32, EngError> {
        native::resident::matvec_q::event_elapsed_ms_raw(start as *mut _, end as *mut _)
            .map_err(|failure| native_error(&TERMINAL, ADAPTER_OPERATION, failure.status))
    }

    #[cfg(test)]
    pub(crate) fn bench_destroy(&self, event: usize) -> Result<(), EngError> {
        native::resident::matvec_q::event_destroy_raw(event as *mut _)
            .map_err(|failure| native_error(&TERMINAL, ADAPTER_OPERATION, failure.status))
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
