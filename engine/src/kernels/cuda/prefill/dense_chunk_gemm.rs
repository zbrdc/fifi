//! The dense prompt-chunk GEMM of a family whose weights rest as the file's native blocks: the
//! q8_1 input buffer and the stream-k fixup range the shared MMQ GEMM needs, and the two calls
//! that use them. A family keeps one of these beside its weights and runs a chunk's projections
//! through [`Self::input`] (once per activation) and [`Self::project`] (once per weight).

use crate::EngError;
use crate::kernels::cuda::context::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};
use crate::kernels::cuda::prefill::mmq_dense::{
    MmqDenseCall, MmqDenseInputCall, check_dense_rules, takes,
};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::{BorrowedNativeWeight, NativeBlocks, NativeWeight};
use crate::weights::GgufType;

/// The most bytes the stream-k fixup range takes on any device (132 SMs of 128 x 128 floats).
const FIXUP_CAP_BYTES: usize = 16 << 20;

pub(crate) struct DenseChunkGemm {
    input: DeviceBuffer,
    fixup: DeviceBuffer,
}

impl DenseChunkGemm {
    /// The device bytes of a gemm over activations of up to `k_max` values and `max_cols` rows,
    /// for the admission to charge before anything is allocated.
    pub(crate) fn bytes(k_max: usize, max_cols: usize) -> Option<u64> {
        Some((q8_1_bytes(k_max, max_cols).ok()? + FIXUP_CAP_BYTES) as u64)
    }

    pub(crate) fn new(ctx: &CudaContext, k_max: usize, max_cols: usize) -> Result<Self, EngError> {
        Ok(Self {
            input: ctx.allocate(q8_1_bytes(k_max, max_cols)?)?,
            fixup: ctx.allocate(ctx.mmq_fixup_bytes()?)?,
        })
    }

    /// Quantizes the `m` f32 rows `x` of `weight.k()` values into the input of the
    /// [`Self::project`] calls that follow; the weight names the layout, so any weight of the
    /// activation's width does.
    pub(crate) fn input(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        weight: &NativeWeight,
        x: DeviceRange<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        ctx.mmq_dense_input(
            stream,
            MmqDenseInputCall {
                packing: MmqPacking::CpuReference,
                weight,
                m,
                row_stride: weight.k(),
                x,
                y: self.input.range(0, self.input.bytes())?,
            },
        )
    }

    /// Writes the `m` rows of `weight.n_rows()` f32 outputs of the last [`Self::input`] into `y`,
    /// row after row.
    pub(crate) fn project(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        weight: &NativeWeight,
        m: usize,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        ctx.mmq_dense(
            stream,
            MmqDenseCall {
                weight,
                w_seg: None,
                m,
                q8_1: self.input.range(0, self.input.bytes())?,
                dst: y,
                ld: weight.n_rows(),
                col_off: 0,
                fixup: self.fixup.range(0, self.fixup.bytes())?,
            },
        )
    }
}

/// The prompt-chunk GEMM of a family whose chunk rows pack several activation planes and whose
/// weights rest as native blocks in its own buffers: one q8_1 input per plane and weight type,
/// and the stream-k fixup range, made on first use and kept for the next chunk.
#[derive(Default)]
pub(crate) struct PlaneGemm {
    fixup: Option<DeviceBuffer>,
    inputs: Vec<((usize, QuantType), DeviceBuffer)>,
}

/// One projection of [`PlaneGemm::project`]: `weight` over the rows of activation plane `plane`
/// at `x` (`ld_x` floats apart), written to `y` (`ld_y` floats apart).
pub(crate) struct PlaneProjection<'a> {
    pub(crate) weight: BorrowedNativeWeight<'a>,
    pub(crate) plane: usize,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) y: DeviceRange<'a>,
    pub(crate) ld_x: usize,
    pub(crate) ld_y: usize,
}

impl PlaneGemm {
    /// The weight of `rows` rows of `cols` values of `ty` at the start of `codes`, when the GEMM
    /// has a case for its type and shape over `m` rows and the rows at `x` are whole 16-byte
    /// copies.
    pub(crate) fn weight<'a>(
        codes: DeviceRange<'a>,
        ty: GgufType,
        [rows, cols]: [usize; 2],
        (m, x, ld_x): (usize, &DeviceRange<'_>, usize),
        ld_y: usize,
    ) -> Option<BorrowedNativeWeight<'a>> {
        let quant = QuantType::from_gguf(ty).filter(|&quant| takes(quant))?;
        let aligned = codes.offset().is_multiple_of(16)
            && x.offset().is_multiple_of(16)
            && ld_x.is_multiple_of(4);
        if !aligned || check_dense_rules(quant, cols, rows, m, ld_y, 0).is_err() {
            return None;
        }
        BorrowedNativeWeight::new(codes, quant, cols, rows).ok()
    }

    /// Queues `calls` over `m` rows on `stream`: each activation plane is quantized to q8_1 once
    /// per weight type, then each weight's GEMM writes its rows.
    pub(crate) fn project(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        calls: &[PlaneProjection<'_>],
        m: usize,
    ) -> Result<(), EngError> {
        if self.fixup.is_none() {
            self.fixup = Some(ctx.allocate(ctx.mmq_fixup_bytes()?)?);
        }
        let mut quantized = Vec::new();
        for call in calls {
            let key = (call.plane, call.weight.quant());
            let need = q8_1_bytes(call.weight.k(), m)?;
            let index = match self.inputs.iter().position(|(held, _)| *held == key) {
                Some(index) if self.inputs[index].1.bytes() >= need => index,
                Some(index) => {
                    self.inputs[index].1 = ctx.allocate(need)?;
                    index
                }
                None => {
                    self.inputs.push((key, ctx.allocate(need)?));
                    self.inputs.len() - 1
                }
            };
            let input = &self.inputs[index].1;
            let q8_1 = input.range(0, input.bytes())?;
            if !quantized.contains(&key) {
                ctx.mmq_dense_input(
                    stream,
                    MmqDenseInputCall {
                        packing: MmqPacking::CpuReference,
                        weight: &call.weight,
                        m,
                        row_stride: call.ld_x,
                        x: call.x,
                        y: q8_1,
                    },
                )?;
                quantized.push(key);
            }
            let fixup = self.fixup.as_ref().ok_or_else(|| {
                EngError::BadRequest("the plane GEMM lost its fixup range".to_owned())
            })?;
            ctx.mmq_dense(
                stream,
                MmqDenseCall {
                    weight: &call.weight,
                    w_seg: None,
                    m,
                    q8_1,
                    dst: call.y,
                    ld: call.ld_y,
                    col_off: 0,
                    fixup: fixup.range(0, fixup.bytes())?,
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
