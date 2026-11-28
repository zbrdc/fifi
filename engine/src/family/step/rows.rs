//! The activation rows of a chunk and the projections over them.
//!
//! A projection reads its input as q8_1 rows (the quantized matvec) and, when the producer wrote
//! them, as f32 rows (the batched matrix product). [`StepIo`] carries what every projection of a
//! step needs: the context, the stream, the resident weights by name and the scratch of the
//! batched product. The kernel follows the type of the weight tensor.

use crate::EngError;
use crate::family::project::cuda::{CudaWeight, CudaWeights};
use crate::kernels::cuda::prefill::gemm_dq::GemmDqTypedCall;
use crate::kernels::cuda::prefill::mmq_dense::{self, MmqDenseCall, MmqDenseInputCall};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as mmq_q8_bytes;
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::block_ops::{BlockOp, RowOutputs};
use crate::kernels::cuda::resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape};
use crate::kernels::cuda::resident::matvec_gguf_f32::DenseF32Call;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::resident::pack_rows::{PackRowsCall, PackSegment};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight};

/// Bytes of `rows` q8_1 rows of `width` values.
pub(crate) fn q8(width: usize, rows: usize) -> usize {
    q8_1_bytes(width, rows).expect("a q8_1 row size fits usize")
}

/// The tensor `name` of a device load, resident in the blocks of its file type. `family` names
/// the model in the refusal.
pub(crate) fn native<'a>(
    weights: &'a CudaWeights,
    family: &str,
    name: &str,
) -> Result<&'a NativeWeight, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Native { weight, .. }) => Ok(weight),
        _ => Err(EngError::Checkpoint(format!(
            "{family}: tensor {name} is not resident as native blocks"
        ))),
    }
}

/// The tensor `name` of a device load, resident as its raw bytes.
pub(crate) fn raw<'a>(
    weights: &'a CudaWeights,
    family: &str,
    name: &str,
) -> Result<&'a DeviceBuffer, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
        _ => Err(EngError::Checkpoint(format!(
            "{family}: tensor {name} is not resident as a raw tensor"
        ))),
    }
}

/// The weight map of a step: the tensors of the device load and the tensors a model derived
/// from them at load, each under its name. `family` names the model in refusals.
#[derive(Clone, Copy)]
pub(crate) struct Map<'a> {
    pub(crate) family: &'static str,
    pub(crate) weights: &'a CudaWeights,
    pub(crate) derived: Option<&'a CudaWeights>,
}

impl<'a> Map<'a> {
    /// The tensor `name`, resident in the blocks of its file type.
    pub(crate) fn native(&self, name: &str) -> Result<&'a NativeWeight, EngError> {
        native(self.weights, self.family, name)
    }

    /// The tensor `name`, resident as its raw bytes: a derived tensor when the model made one
    /// under that name.
    pub(crate) fn raw(&self, name: &str) -> Result<DeviceRange<'a>, EngError> {
        let buffer = match self.derived {
            Some(derived) if derived.get(name).is_some() => raw(derived, self.family, name)?,
            _ => raw(self.weights, self.family, name)?,
        };
        buffer.range(0, buffer.bytes())
    }
}

/// The most rows of a chunk whose every projection goes through the matvec: up to four
/// activation columns it reads the weights once and gives every column the bits of a
/// one-column launch. On an RTX 3090 its weight sweep took 1.06 times one column at two columns
/// and 1.45 at four. Against an exact route (every weight decoded, f32 activations) on
/// SenseNova-Loop, rows of such chunks then sit as close as a one-row step (mean KL 0.0015),
/// where the batched product left them at 0.0038.
pub(crate) const FEW_ROWS: usize = 4;
/// The most rows the matvec takes for rows that land back to back. From 5 to 8 rows the rows
/// that land apart take the batched product: the matvec for those too measured slower.
const MATVEC_ROWS: usize = 8;

/// Width from which a row's norm takes the wide block, and the most rows of a chunk that do: a
/// few rows then norm with the bits of a one-row step, and faster than in the narrow block.
const WIDE_NORM: usize = 4096;
const WIDE_ROWS: usize = 16;

/// One RMS norm: rows of `d` values at `x`, plus the rows at `delta` when given, scaled by
/// `weight`. With `delta`, the sum also lands back in `x`.
pub(crate) struct Norm<'a> {
    pub(crate) d: usize,
    pub(crate) eps: f32,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) delta: Option<DeviceRange<'a>>,
    pub(crate) weight: DeviceRange<'a>,
}

/// The activation rows a projection reads: q8_1 rows for the quantized matvec and, when the
/// producer wrote them, the f32 rows for the batched matrix product.
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub(crate) q8: &'a DeviceBuffer,
    pub(crate) f32: Option<&'a DeviceBuffer>,
}

impl<'a> Rows<'a> {
    pub(crate) fn q8(q8: &'a DeviceBuffer) -> Self {
        Self { q8, f32: None }
    }

    pub(crate) fn both(q8: &'a DeviceBuffer, f32: &'a DeviceBuffer) -> Self {
        Self { q8, f32: Some(f32) }
    }
}

/// What the projections of one step share. `family` names the model in refusals.
#[derive(Clone, Copy)]
pub(crate) struct StepIo<'a> {
    pub(crate) family: &'static str,
    pub(crate) ctx: &'a CudaContext,
    pub(crate) stream: &'a DeviceStream,
    pub(crate) weights: &'a CudaWeights,
    /// Tensors the model derived at load, looked up by name before the load's.
    pub(crate) derived: Option<&'a CudaWeights>,
    /// Scratch of the batched product: the q8_1 rows of its input.
    pub(crate) mmq_q8: &'a DeviceBuffer,
    /// Scratch of the batched product: its fix-up words.
    pub(crate) fixup: &'a DeviceBuffer,
    /// Projections of Q4_K and Q6_K weights read the f32 rows and decode every weight value,
    /// with no q8_1 step: slower, and the accuracy reference of the quantized routes.
    pub(crate) exact: bool,
}

/// Where a projection writes: row `t` lands at `col_off` of row `t` of `dst`, whose rows are
/// `ld` floats apart.
#[derive(Clone, Copy)]
pub(crate) struct Out<'b> {
    pub(crate) dst: DeviceRange<'b>,
    pub(crate) ld: usize,
    pub(crate) col_off: usize,
}

impl<'b> Out<'b> {
    /// `n` rows of `width` floats back to back at the start of `buffer`.
    pub(crate) fn packed(
        buffer: &'b DeviceBuffer,
        n: usize,
        width: usize,
    ) -> Result<Self, EngError> {
        Ok(Self {
            dst: buffer.range(0, n * width * 4)?,
            ld: width,
            col_off: 0,
        })
    }
}

impl<'a> StepIo<'a> {
    pub(crate) fn run(&self, op: BlockOp<'_>) -> Result<(), EngError> {
        self.ctx.block_op(self.stream, op)
    }

    pub(crate) fn map(&self) -> Map<'a> {
        Map {
            family: self.family,
            weights: self.weights,
            derived: self.derived,
        }
    }

    /// The tensor `name`, resident in the blocks of its file type.
    pub(crate) fn native(&self, name: &str) -> Result<&'a NativeWeight, EngError> {
        self.map().native(name)
    }

    /// The tensor `name`, resident as its raw bytes.
    pub(crate) fn raw(&self, name: &str) -> Result<DeviceRange<'a>, EngError> {
        self.map().raw(name)
    }

    pub(crate) fn whole<'b>(buffer: &'b DeviceBuffer) -> Result<DeviceRange<'b>, EngError> {
        buffer.range(0, buffer.bytes())
    }

    /// The norm of `n` rows: the f32 rows into `y` and their q8_1 rows into `q8_out`, through the
    /// fused add, norm and quantize kernel. A few wide rows take the kernel's 1024-thread block,
    /// which sums the squares in another order than the 256-thread block of a larger chunk.
    pub(crate) fn norm(
        &self,
        n: usize,
        norm: Norm<'_>,
        y: DeviceRange<'_>,
        q8_out: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let wide = n <= WIDE_ROWS && norm.d >= WIDE_NORM;
        let call = AddNormQ8Call {
            n: norm.d,
            eps: norm.eps,
            h: norm.x,
            delta: norm.delta,
            weight: norm.weight,
            norm_out: y,
            q8: q8_out,
            rows: n,
        };
        if wide {
            return self.ctx.add_rmsnorm_q8_wide(self.stream, call);
        }
        self.ctx.add_rmsnorm_q8(self.stream, call)
    }

    /// [`Self::norm`] of whole buffers with no added rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn norm_rows(
        &self,
        n: usize,
        d: usize,
        eps: f32,
        x: &DeviceBuffer,
        weight: DeviceRange<'_>,
        y: &DeviceBuffer,
        q8_out: &DeviceBuffer,
    ) -> Result<(), EngError> {
        let x = x.range(0, n * d * 4)?;
        let norm = Norm {
            d,
            eps,
            x,
            delta: None,
            weight,
        };
        self.norm(n, norm, y.range(0, n * d * 4)?, q8_out.range(0, q8(d, n))?)
    }

    /// `y[t] = W x[t]` for `n` rows of the tensor `name`, tagged with a role for
    /// `FIFI_CUDA_DECODE_DQ_ROLES` routing.
    pub(crate) fn mv_role(
        &self,
        role: &str,
        name: &str,
        x: Rows<'_>,
        y: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        let weight = self.native(name)?;
        self.project_role_into(role, weight, x, n, Out::packed(y, n, weight.n_rows())?)
    }

    /// `y[t] = W x[t]` for `n` rows, written back to back into `y`.
    pub(crate) fn project(
        &self,
        weight: &NativeWeight,
        x: Rows<'_>,
        y: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        self.project_into(weight, x, n, Out::packed(y, n, weight.n_rows())?)
    }

    /// Whether `n` rows through `weight` take the batched product: a chunk of more than 8 rows,
    /// or one of more than a few rows whose output rows are not back to back, when the weight
    /// type and the row length allow it and the rows exist as f32.
    fn batched(&self, weight: &NativeWeight, x: Rows<'_>, n: usize, packed: bool) -> bool {
        (n > MATVEC_ROWS || (n > FEW_ROWS && !packed))
            && mmq_dense::serves(weight)
            && x.f32.is_some()
    }

    /// `y[t] = W x[t]` for `n` rows into `out`. One row, and up to 8 rows written back to back,
    /// use the quantized matvec on the q8_1 rows; other chunks use the batched product on the
    /// f32 rows. With [`Self::exact`] every weight takes a decoding route over the f32 rows.
    /// Projection with an explicit role tag for opt-in `FIFI_CUDA_DECODE_DQ_ROLES` routing.
    pub(crate) fn project_role_into(
        &self,
        role: &str,
        weight: &NativeWeight,
        x: Rows<'_>,
        n: usize,
        out: Out<'_>,
    ) -> Result<(), EngError> {
        let (k, rows) = (weight.k(), weight.n_rows());
        let span = ((n - 1) * out.ld + rows) * 4;
        let y = out
            .dst
            .buffer()
            .range(out.dst.offset() + out.col_off * 4, span)?;
        if !self.exact
            && n == 1
            && crate::kernels::cuda::resident::matvec_dq::decode_dq_role_enabled(role)
            && crate::kernels::cuda::resident::matvec_dq::serves(weight.quant())
            && k.is_multiple_of(256)
            && let Some(f32_rows) = x.f32
        {
            return self
                .ctx
                .matvec_dq_native(self.stream, weight, f32_rows.range(0, k * 4)?, y);
        }
        self.project_into(weight, x, n, out)
    }

    pub(crate) fn project_into(
        &self,
        weight: &NativeWeight,
        x: Rows<'_>,
        n: usize,
        out: Out<'_>,
    ) -> Result<(), EngError> {
        let (k, rows) = (weight.k(), weight.n_rows());
        let span = ((n - 1) * out.ld + rows) * 4;
        let y = out
            .dst
            .buffer()
            .range(out.dst.offset() + out.col_off * 4, span)?;
        // A product that would read rounded rows under the exact switch is no reference.
        if self.exact && x.f32.is_none() {
            return Err(EngError::BadRequest(format!(
                "{}: the exact route has no f32 rows for a product of {k} inputs",
                self.family
            )));
        }
        #[cfg(test)]
        let exact = self.exact || exact_class::holds(weight);
        #[cfg(not(test))]
        let exact = self.exact;
        if exact
            && crate::kernels::cuda::resident::matvec_dq::serves(weight.quant())
            && k.is_multiple_of(256)
            && let Some(f32_rows) = x.f32
        {
            if n == 1 {
                return self.ctx.matvec_dq_native(
                    self.stream,
                    weight,
                    f32_rows.range(0, k * 4)?,
                    y,
                );
            }
            if matches!(weight.quant(), QuantType::Q4K | QuantType::Q6K) {
                let row_bytes = weight.quant().weight_bytes(1, k).ok_or_else(|| {
                    EngError::MemoryPlan("the dense projection size overflows".to_owned())
                })?;
                return self.ctx.gemm_dq(
                    self.stream,
                    weight.quant(),
                    GemmDqTypedCall {
                        w: weight.buffer().range(0, rows * row_bytes)?,
                        x: f32_rows.range(0, n * k * 4)?,
                        y,
                        k,
                        rows,
                        ncols: n,
                        x_col_stride: k,
                        y_col_stride: out.ld,
                        x_f32: true,
                    },
                );
            }
        }
        // Every other type on the exact route: its blocks decoded per value over the f32 row.
        if exact && let Some(f32_rows) = x.f32 {
            let row_bytes = weight.quant().weight_bytes(1, k).ok_or_else(|| {
                EngError::MemoryPlan("the dense projection size overflows".to_owned())
            })?;
            for t in 0..n {
                self.ctx.matvec_gguf_f32(
                    self.stream,
                    DenseF32Call {
                        ty: weight.quant(),
                        w: weight.buffer().range(0, rows * row_bytes)?,
                        x: f32_rows.range(t * k * 4, k * 4)?,
                        y: out.dst.slice((t * out.ld + out.col_off) * 4, rows * 4)?,
                        k,
                        rows,
                        ncols: 1,
                    },
                )?;
            }
            return Ok(());
        }
        let packed = out.ld == rows && out.col_off == 0;
        if self.batched(weight, x, n, packed)
            && let Some(f32_rows) = x.f32
        {
            let q8_range = self.mmq_q8.range(0, mmq_q8_bytes(k, n)?)?;
            self.ctx.mmq_dense_input(
                self.stream,
                MmqDenseInputCall {
                    // A scale per 32 values and the sum of the rounded codes: the product
                    // then reads what the one-row matvec reads.
                    packing: MmqPacking::Rounded,
                    weight,
                    m: n,
                    row_stride: k,
                    x: f32_rows.range(0, n * k * 4)?,
                    y: q8_range,
                },
            )?;
            return self.ctx.mmq_dense(
                self.stream,
                MmqDenseCall {
                    weight,
                    w_seg: None,
                    m: n,
                    q8_1: q8_range,
                    dst: out.dst,
                    ld: out.ld,
                    col_off: out.col_off,
                    fixup: Self::whole(self.fixup)?,
                },
            );
        }
        if n > 1 && !packed {
            return self.project_apart(weight, x, n, out);
        }
        self.ctx
            .matvec_native(self.stream, weight, x.q8.range(0, q8(k, n))?, y, n)
    }

    /// [`Self::project_into`] through the matvec for rows that land apart. The matvec writes its
    /// rows back to back: a few rows go there in one launch, into the scratch of the batched
    /// product, and one copy spreads them; more rows go one at a time.
    fn project_apart(
        &self,
        weight: &NativeWeight,
        x: Rows<'_>,
        n: usize,
        out: Out<'_>,
    ) -> Result<(), EngError> {
        let (k, rows) = (weight.k(), weight.n_rows());
        let together = n * rows * 4;
        if n <= FEW_ROWS && together <= self.mmq_q8.bytes() {
            let spill = self.mmq_q8.range(0, together)?;
            self.ctx
                .matvec_native(self.stream, weight, x.q8.range(0, q8(k, n))?, spill, n)?;
            let segment = PackSegment {
                src: spill,
                n: rows,
                dst_off: out.col_off,
            };
            let span = ((n - 1) * out.ld + out.col_off + rows) * 4;
            return self.ctx.pack_rows(
                self.stream,
                PackRowsCall {
                    m: n,
                    ld: out.ld,
                    dst: out.dst.slice(0, span.min(out.dst.bytes()))?,
                    segments: vec![segment],
                },
            );
        }
        let row = q8(k, 1);
        for t in 0..n {
            let y = out.dst.slice((t * out.ld + out.col_off) * 4, rows * 4)?;
            self.ctx
                .matvec_native(self.stream, weight, x.q8.range(t * row, row)?, y, 1)?;
        }
        Ok(())
    }

    /// Whether the projection of `n` rows through `weight`, written back to back, reads the
    /// q8_1 rows of its input.
    pub(crate) fn reads_q8(&self, weight: &NativeWeight, x: Rows<'_>, n: usize) -> bool {
        let exact = self.exact && x.f32.is_some();
        !exact && !self.batched(weight, x, n, true)
    }

    /// The projection `y[t] = W x[t]` of `n` f32 rows through the F16 matrix `w`, `rows` by
    /// `cols`. Larger non-exact chunks run on f16 tensor cores with f32 accumulation; few-row
    /// steps and the exact route keep the scalar f32 matvec.
    pub(crate) fn f16_project(
        &self,
        w: DeviceRange<'_>,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        n: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), EngError> {
        if !self.exact
            && n > FEW_ROWS
            && n <= 65536
            && rows <= 1_048_576
            && cols <= 65536
            && w.offset().is_multiple_of(4)
        {
            return self.run(BlockOp::f16_gemm(
                n,
                rows,
                cols,
                x.range(0, n * cols * 4)?,
                w,
                y.range(0, n * rows * 4)?,
            )?);
        }
        self.ctx.dense_gemv(
            self.stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(n, rows, cols),
                dtype: DenseDtype::F16,
                w,
                x: x.range(0, n * cols * 4)?,
                y: y.range(0, n * rows * 4)?,
            },
        )
    }

    /// Quantizes `n` rows of `width` f32 values (x * sigmoid(gate) when a gate is given) into
    /// q8_1 rows, and also writes the f32 product when `out` has f32 rows.
    pub(crate) fn quantize(
        &self,
        n: usize,
        width: usize,
        x: &DeviceBuffer,
        gate: Option<&DeviceBuffer>,
        out: Rows<'_>,
    ) -> Result<(), EngError> {
        self.run(BlockOp::gated_quant(
            n,
            width,
            x.range(0, n * width * 4)?,
            gate.map(|g| g.range(0, n * width * 4)).transpose()?,
            RowOutputs {
                q8: Some(out.q8.range(0, q8(width, n))?),
                f32: out.f32.map(|b| b.range(0, n * width * 4)).transpose()?,
            },
        )?)
    }
}

/// A test's switch: the weights of one shape and type take the decoding routes while every
/// other weight stays on its own, so an anatomy test can find which products carry an error.
#[cfg(test)]
pub(crate) mod exact_class {
    use super::{NativeWeight, QuantType};
    use std::sync::Mutex;

    static CLASS: Mutex<Option<(usize, usize, QuantType)>> = Mutex::new(None);

    /// Sets the class as `(rows, k, type)`, or clears it.
    pub(crate) fn set(class: Option<(usize, usize, QuantType)>) {
        *CLASS.lock().unwrap() = class;
    }

    pub(super) fn holds(weight: &NativeWeight) -> bool {
        *CLASS.lock().unwrap() == Some((weight.n_rows(), weight.k(), weight.quant()))
    }
}
