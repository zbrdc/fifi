//! The grouped (per-expert) dequantizing GEMM behind the ids of an [`MmidIds`]: the weight is the
//! file's own blocks of one of the in-house dequantizing types (Q3_K, Q4_0, Q5_K, IQ4_XS, every
//! codebook type and Q2_K), the activation is f32 rows, and no q8_1 quantize pass feeds it.
//!
//! Dense launches use a validated [`MmidIds`] permutation. Sparse launches use a validated
//! [`GroupedDqRows`] selection that leaves other rows available to a separate kernel.

use super::{GroupedDqRows, MmidIds, MmidShape, MmidWeight, rule};
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::gemm_dq::{
    GemmDqMmidRaw, gemm_dq_mmid_launch, gemm_dq_mmid_type,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// One grouped dequantizing GEMM. All ranges are device ranges of this context.
///
/// - `weight`: the expert stack, `weight.n` rows of `weight.k` values per expert, experts
///   `weight.stride_bytes` apart; its type is one [`gemm_dq_mmid_type`] accepts.
/// - `x`: f32 rows of `x_row_stride >= k` floats (a multiple of 4), 16-byte aligned, read only. With `x_per_slot`
///   false they are the token rows (slot `s` reads row `s / top_k`: the gate and up input);
///   with it true row `s` itself (the down input, in token-slot order).
/// - `y`: f32 slot rows of `ld >= n` floats, `(m * top_k - 1) * ld + n` floats, 4-byte aligned:
///   row `t * top_k + r` receives pick `r` of token `t`, like the vendored launch's `dst`.
pub(crate) struct DqMmidCall<'b, 'i> {
    pub(crate) shape: MmidShape,
    pub(crate) weight: MmidWeight<'b, QuantType>,
    pub(crate) ids: &'i MmidIds<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) x_row_stride: usize,
    pub(crate) x_per_slot: bool,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) ld: usize,
    /// The column tile of the launch: 0 chooses from the expected columns per expert, else 64
    /// or 128.
    pub(crate) col_tile: usize,
}

/// Replace only selected expert-major rows. `metadata` holds the plan's IDs followed by
/// its bounds, enqueued earlier on the same stream as part of a checked arena upload.
pub(crate) struct SparseDqCall<'b, 'p> {
    pub(crate) weight: MmidWeight<'b, QuantType>,
    pub(crate) plan: &'p GroupedDqRows,
    pub(crate) metadata: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

struct DqLaunch<'b> {
    weight: MmidWeight<'b, QuantType>,
    ids_dst: DeviceRange<'b>,
    expert_bounds: DeviceRange<'b>,
    x: DeviceRange<'b>,
    y: DeviceRange<'b>,
    experts: usize,
    total_cols: usize,
    x_div: usize,
    x_row_stride: usize,
    ld: usize,
    col_tile: usize,
}

/// Whether the grouped dequantizing GEMM decodes `quant` (the family-side check before it
/// plans a grouped expert stack).
pub(crate) fn grouped_dq_serves(quant: QuantType) -> bool {
    gemm_dq_mmid_type(quant)
}

impl CudaContext {
    /// Queues one grouped dequantizing GEMM of `call` on `stream` and returns when it is
    /// queued; `y` holds the products after the stream drained. The ids are read in stream
    /// order, so their constructor's work must precede this launch on `stream`.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a type
    /// the GEMM does not serve, ids made for another shape, a `k` that is not a positive
    /// multiple of the type's block size and of 64, strides below the row length, a range that
    /// is too short or misaligned, and `y` overlapping another range of the call.
    pub(crate) fn gemm_dq_mmid(
        &self,
        stream: &DeviceStream,
        call: DqMmidCall<'_, '_>,
    ) -> Result<(), EngError> {
        if call.ids.shape != call.shape {
            let detail = format!("ids of {:?} for a {:?} launch", call.ids.shape, call.shape);
            return Err(rule("ids shape", &detail));
        }
        super::check_ids_shape(&call.shape)?;
        self.launch_grouped_dq(
            stream,
            DqLaunch {
                weight: call.weight,
                ids_dst: call.ids.out.ids_dst,
                expert_bounds: call.ids.out.expert_bounds,
                x: call.x,
                y: call.y,
                experts: call.shape.n_experts,
                total_cols: call.shape.ne_get_rows(),
                x_div: if call.x_per_slot { 1 } else { call.shape.top_k },
                x_row_stride: call.x_row_stride,
                ld: call.ld,
                col_tile: call.col_tile,
            },
        )
    }

    pub(crate) fn gemm_dq_sparse_rows(
        &self,
        stream: &DeviceStream,
        call: SparseDqCall<'_, '_>,
    ) -> Result<(), EngError> {
        let id_bytes = call.plan.ids().len() * 4;
        let bounds_bytes = call.plan.bounds().len() * 4;
        let ids_dst = call.metadata.slice(0, id_bytes)?;
        let expert_bounds = call.metadata.slice(id_bytes, bounds_bytes)?;
        let (x_row_stride, ld) = (call.weight.k, call.weight.n);
        self.launch_grouped_dq(
            stream,
            DqLaunch {
                weight: call.weight,
                ids_dst,
                expert_bounds,
                x: call.x,
                y: call.y,
                experts: call.plan.bounds().len() - 1,
                total_cols: call.plan.ids().len(),
                x_div: 1,
                x_row_stride,
                ld,
                col_tile: 64,
            },
        )
    }

    fn launch_grouped_dq(&self, stream: &DeviceStream, call: DqLaunch<'_>) -> Result<(), EngError> {
        self.ensure_usable()?;
        let weight = &call.weight;
        let ty = weight.ty;
        if !gemm_dq_mmid_type(ty) {
            return Err(rule(
                "type",
                &format!("the grouped dequantizing GEMM has no {ty:?} decoder"),
            ));
        }
        let (k, n) = (weight.k, weight.n);
        if k == 0 || !k.is_multiple_of(ty.block_size().max(64)) {
            return Err(rule(
                "K",
                &format!(
                    "K is {k}, not a positive multiple of {}",
                    ty.block_size().max(64)
                ),
            ));
        }
        let expert_bytes = ty
            .weight_bytes(n, k)
            .ok_or_else(|| rule("stride", "the expert size overflows"))?;
        if n == 0 || weight.stride_bytes < expert_bytes {
            return Err(rule(
                "stride",
                &format!(
                    "{} bytes between experts, below the {expert_bytes} of {n} rows",
                    weight.stride_bytes
                ),
            ));
        }
        if call.x_row_stride < k || call.ld < n {
            return Err(rule(
                "stride",
                &format!(
                    "x row stride {} for K {k}, y row stride {} for N {n}",
                    call.x_row_stride, call.ld
                ),
            ));
        }
        let total_cols = call.total_cols;
        let x_rows = total_cols.div_ceil(call.x_div);
        let need_w = (call.experts - 1)
            .checked_mul(weight.stride_bytes)
            .and_then(|head| head.checked_add(expert_bytes))
            .ok_or_else(|| rule("size", "the weight stack size overflows"))?;
        let need_x = (x_rows - 1)
            .checked_mul(call.x_row_stride)
            .and_then(|head| head.checked_add(k))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(|| rule("size", "the activation size overflows"))?;
        let need_y = (total_cols - 1)
            .checked_mul(call.ld)
            .and_then(|head| head.checked_add(n))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(|| rule("size", "the output size overflows"))?;
        let w_align = if ty == QuantType::Q5K { 16 } else { 2 };
        require_fit("w", &weight.base, need_w, w_align)?;
        require_fit("x", &call.x, need_x, 16)?;
        if !call.x_row_stride.is_multiple_of(4) || !matches!(call.col_tile, 0 | 64 | 128) {
            return Err(rule(
                "stride",
                &format!(
                    "x row stride {} is not a multiple of 4 or column tile {} is not 0, 64 or 128",
                    call.x_row_stride, call.col_tile
                ),
            ));
        }
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &weight.base, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        let ids_dst_bytes = total_cols * 4;
        let bounds_bytes = (call.experts + 1) * 4;
        require_fit("ids_dst", &call.ids_dst, ids_dst_bytes, 4)?;
        require_fit("expert_bounds", &call.expert_bounds, bounds_bytes, 4)?;
        require_apart(
            &written,
            &Claim::of("ids_dst", &call.ids_dst, ids_dst_bytes),
        )?;
        require_apart(
            &written,
            &Claim::of("expert_bounds", &call.expert_bounds, bounds_bytes),
        )?;
        let raw = GemmDqMmidRaw {
            ty,
            w: self.raw_range(&weight.base)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            ids_dst: self.raw_range(&call.ids_dst)?,
            expert_bounds: self.raw_range(&call.expert_bounds)?,
            k,
            rows: n,
            n_experts: call.experts,
            total_cols,
            x_div: call.x_div,
            x_row_stride: call.x_row_stride,
            y_row_stride: call.ld,
            expert_stride: weight.stride_bytes,
            col_tile: call.col_tile,
        };
        self.launch_on(stream, move |s| gemm_dq_mmid_launch(s, &raw))
    }
}
