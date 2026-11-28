//! One routed (grouped, per-expert) GEMM of a MoE stage, for any family: the stack's type picks the
//! in-house dequantizing GEMM (it reads the f32 activation rows) or the vendored grouped MMQ (it
//! reads q8_1 columns built from them), so a stage quantizes only for the stacks that need it.

use super::{DqMmidCall, MmidIds, MmidLaunchCall, MmidQuantCall, MmidWeight, mmq_type_of};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_quant::{K_GRID, K_MAX};
use crate::kernels::cuda::prefill::mmq_size::{Q8_1Layout, weight_tail};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream, NativeWeight};

/// The `MmidWeight` of one routed stack (stages 13a, 13b, 16): the whole stack buffer, its
/// type, `k` values per expert row, the expert's `n` rows, and the per-expert byte stride
/// derived from the type.
pub(crate) fn routed_weight<'a>(
    stack: &'a NativeWeight,
    k: usize,
    n: usize,
) -> Result<MmidWeight<'a, QuantType>, EngError> {
    let quant = stack.quant();
    let stride_bytes = quant.weight_bytes(n, k).ok_or_else(|| {
        EngError::BadRequest(format!(
            "routed stack: the {quant:?} stride of {n} rows of {k} values overflows"
        ))
    })?;
    Ok(MmidWeight {
        base: stack.buffer().range(0, stack.buffer().bytes())?,
        ty: quant,
        k,
        n,
        stride_bytes,
    })
}

/// The q8_1 layout a shared quantize buffer holds right now. Vendored GEMMs of one group may
/// need different layouts (Q8_0 and Q6_K read the D4 form, Q4_K the DS4 form), so a role whose
/// layout differs quantizes the buffer again before it launches; a role of the layout the buffer
/// already holds reads it as it is.
#[derive(Default)]
pub(crate) struct Q8Cursor(std::cell::Cell<Option<Q8_1Layout>>);

impl Q8Cursor {
    /// Runs `quantize` unless the buffer already holds the layout of a `quant` weight.
    pub(crate) fn ensure(
        &self,
        quant: QuantType,
        quantize: impl FnOnce() -> Result<(), EngError>,
    ) -> Result<(), EngError> {
        let layout = super::mmq_type_of(quant)?.q8_1_layout();
        if self.0.get() != Some(layout) {
            quantize()?;
            self.0.set(Some(layout));
        }
        Ok(())
    }

    /// Records that the buffer was quantized for a `quant` weight.
    pub(crate) fn note(&self, quant: QuantType) {
        if let Ok(ty) = super::mmq_type_of(quant) {
            self.0.set(Some(ty.q8_1_layout()));
        }
    }
}

/// How the q8_1 columns of a routed stage are built for a vendored stack: from the f32 rows
/// through the ids (`gather` for the down input, else the scatter of the gate and up input).
pub(crate) struct RoutedQuant<'q> {
    pub(crate) ids: &'q MmidIds<'q>,
    pub(crate) gather: bool,
    pub(crate) packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking,
    pub(crate) cursor: Q8Cursor,
}

/// What one routed GEMM of a stage reads: the vendored grouped MMQ reads the q8_1 columns `q8`
/// (built on demand, at the layout of the stack that reads them), the in-house dequantizing GEMM
/// the f32 `rows` (the token rows of the gate and up input, or the slot rows of the down input
/// when `per_slot`), so a stage quantizes only for the stacks that need it.
pub(crate) struct RoutedX<'a, 'q> {
    pub(crate) q8: DeviceRange<'a>,
    pub(crate) rows: DeviceRange<'a>,
    pub(crate) row_stride: usize,
    pub(crate) per_slot: bool,
    pub(crate) quant: RoutedQuant<'q>,
}

/// Whether the vendored grouped MMQ serves `weight`: its type has an MMQ case, `k` is a whole
/// number of weight blocks in `K_GRID..=K_MAX`, and the stack holds the trailing zero pad the
/// last k tile reads when `k` is off `K_GRID`.
fn mmq_serves_weight(weight: &NativeWeight) -> bool {
    let (quant, k) = (weight.quant(), weight.k());
    let Ok(mmq) = mmq_type_of(quant) else {
        return false;
    };
    if !(K_GRID..=K_MAX).contains(&k) || !k.is_multiple_of(mmq.block_values()) {
        return false;
    }
    let Some(base) = quant.weight_bytes(weight.n_rows(), k) else {
        return false;
    };
    let Some(tail) = weight_tail(mmq, k) else {
        return false;
    };
    base.checked_add(tail)
        .is_some_and(|need| weight.buffer().bytes() >= need)
}

/// Whether `weight` takes the dequantizing GEMM: the in-house types, and a stack whose shape or
/// padding the vendored grouped MMQ does not accept.
fn takes_dq(weight: &NativeWeight) -> bool {
    let quant = weight.quant();
    quant.uses_dequantizing_gemm()
        || (!mmq_serves_weight(weight) && super::grouped_dq_serves(quant))
}

/// Whether a routed GEMM serves the stack `weight`.
pub(crate) fn routed_serves(weight: &NativeWeight) -> bool {
    takes_dq(weight) || mmq_serves_weight(weight)
}

/// One routed GEMM of `weight` over `ids`: the dequantizing GEMM for the in-house types and for
/// row lengths off the MMQ grid, the vendored grouped MMQ for the rest. `dst` receives the f32
/// slot rows at stride `ld`.
#[allow(clippy::too_many_arguments)] // one launch description: weight, ids, input, output, scratch
pub(crate) fn routed_gemm(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &NativeWeight,
    ids: &MmidIds<'_>,
    x: &RoutedX<'_, '_>,
    ld: usize,
    dst: DeviceRange<'_>,
    fixup: DeviceRange<'_>,
) -> Result<(), EngError> {
    let n = weight.n_rows() / weight.experts().unwrap_or(1);
    let routed = routed_weight(weight, weight.k(), n)?;
    if takes_dq(weight) {
        ctx.gemm_dq_mmid(
            stream,
            DqMmidCall {
                shape: ids.shape(),
                weight: routed,
                ids,
                x: x.rows,
                x_row_stride: x.row_stride,
                x_per_slot: x.per_slot,
                y: dst,
                ld,
                col_tile: 0,
            },
        )
    } else {
        x.quant.cursor.ensure(weight.quant(), || {
            let call = MmidQuantCall {
                packing: x.quant.packing,
                ty: mmq_type_of(weight.quant())?,
                k: weight.k(),
                row_stride: x.row_stride,
                x: x.rows,
                y: x.q8,
            };
            if x.quant.gather {
                ctx.mmq_quantize_gather_ids(stream, x.quant.ids, call)
            } else {
                ctx.mmq_quantize_scatter_ids(stream, x.quant.ids, call)
            }
        })?;
        ctx.mmq_mmid_launch(
            stream,
            MmidLaunchCall {
                shape: ids.shape(),
                weight: MmidWeight {
                    base: routed.base,
                    ty: mmq_type_of(routed.ty)?,
                    k: routed.k,
                    n: routed.n,
                    stride_bytes: routed.stride_bytes,
                },
                ids,
                y: x.q8,
                dst,
                ld,
                fixup,
            },
        )
    }
}
