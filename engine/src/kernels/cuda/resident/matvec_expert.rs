//! The MMVQ expert matvec against a native-layout expert stack: [`CudaContext::matvec_expert`]
//! queues ONE launch for the used slots, with the per-slot expert ids read on the DEVICE when
//! the kernel runs (the router's `ids` buffer), and returns when it is queued. It allocates
//! nothing, copies nothing and never synchronizes.
//!
//! `w_stack` holds `n_expert` experts of `rows` x `k` native blocks back to back; `x_q8` holds one
//! shared padded activation row (`x_per_slot` false, the gate/up view: every slot reads row 0) or
//! one row per used slot (`x_per_slot` true, the down view); `y` is f32 `[n_used][y_slot_stride]`
//! with `y_slot_stride >= rows` (the step's ExpertGu is `[top_k][gate|up]`, so gate and up are two
//! launches into one `[slot][2*rows]` buffer at offsets 0 and `rows`).
//!
//! Ids contract: the kernel reads `ids[s]` when it runs, so a captured graph replay follows a
//! rewritten ids word. An id outside `[0, n_expert)` — the router writes -1 only on a faulted row —
//! makes the kernel write the quiet NaN `0x7fc00000` to that slot's rows and read no weight (the id
//! range guard); the resident step keeps the router status-word check.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; `k` is a multiple of the type's block size; `rows` is a positive multiple of 4
//! ([`ROWS_MULTIPLE`]); `n_used` is in `1..=8` and at most `n_expert`; `y_slot_stride >= rows`;
//! every range holds the bytes the call uses and is 4-byte aligned; `y` overlaps none of
//! `w_stack`, `x_q8`, `ids`. A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::matvec_expert::{
    ExpertMatvecRaw, matvec_expert_launch,
};
use crate::kernels::cuda::resident::matvec_q::{QuantType, ROWS_MULTIPLE};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

pub(crate) struct ExpertMatvecCall<'b> {
    pub(crate) ty: QuantType,
    pub(crate) w_stack: DeviceRange<'b>,
    pub(crate) ids: DeviceRange<'b>,
    pub(crate) x_q8: DeviceRange<'b>,
    pub(crate) x_per_slot: bool,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) y_slot_stride: usize,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) n_expert: usize,
    pub(crate) n_used: usize,
    /// The fused GLU gate stack (`None` = plain dot). When `Some` the launch writes
    /// silu(gate)*up per y element; it must carry the same type, k, rows and n_expert as
    /// `w_stack`, and only the shared-x view (`x_per_slot == false`) may take it.
    pub(crate) glu_gate: Option<DeviceRange<'b>>,
}

#[derive(Clone, Copy)]
struct ExpertShape {
    ty: QuantType,
    k: usize,
    rows: usize,
    n_expert: usize,
    n_used: usize,
    x_per_slot: bool,
    y_slot_stride: usize,
}

struct ExpertNeeds {
    w_stack: usize,
    ids: usize,
    x_q8: usize,
    y: usize,
}

fn expert_needs(shape: &ExpertShape) -> Option<ExpertNeeds> {
    let x_rows = if shape.x_per_slot { shape.n_used } else { 1 };
    let y_floats = shape
        .y_slot_stride
        .checked_mul(shape.n_used.checked_sub(1)?)?
        .checked_add(shape.rows)?;
    Some(ExpertNeeds {
        w_stack: shape
            .ty
            .weight_bytes(shape.rows.checked_mul(shape.n_expert)?, shape.k)?,
        ids: shape.n_used.checked_mul(4)?,
        x_q8: q8_1_bytes(shape.k, x_rows)?,
        y: y_floats.checked_mul(4)?,
    })
}

impl ExpertMatvecCall<'_> {
    fn shape(&self) -> ExpertShape {
        ExpertShape {
            ty: self.ty,
            k: self.k,
            rows: self.rows,
            n_expert: self.n_expert,
            n_used: self.n_used,
            x_per_slot: self.x_per_slot,
            y_slot_stride: self.y_slot_stride,
        }
    }
}

impl CudaContext {
    pub(crate) fn matvec_expert(
        &self,
        stream: &DeviceStream,
        call: ExpertMatvecCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_matvec_expert(&call)?;
        self.launch_on(stream, |stream_raw| matvec_expert_launch(stream_raw, &raw))
    }

    fn plan_matvec_expert<'b>(
        &self,
        call: &ExpertMatvecCall<'b>,
    ) -> Result<ExpertMatvecRaw<'b>, EngError> {
        let (w_raw, ids_raw, x_raw, y_raw) = (
            self.raw_range(&call.w_stack)?,
            self.raw_range(&call.ids)?,
            self.raw_range(&call.x_q8)?,
            self.raw_range(&call.y)?,
        );
        if !call.k.is_multiple_of(call.ty.block_size()) || call.k == 0 {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert matvec needs k as a multiple of {}, got {}",
                call.ty.block_size(),
                call.k
            )));
        }
        if call.rows == 0 || call.n_expert == 0 {
            return Err(EngError::BadRequest(
                "the CUDA expert matvec needs rows and experts above zero".to_owned(),
            ));
        }
        if !call.rows.is_multiple_of(ROWS_MULTIPLE) {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert matvec needs rows as a multiple of {ROWS_MULTIPLE}, got {}",
                call.rows
            )));
        }
        if call.n_used < 1 || call.n_used > 8 || call.n_used > call.n_expert {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert launch takes 1 to {} used slots, got {}",
                call.n_expert.min(8),
                call.n_used
            )));
        }
        if call.y_slot_stride < call.rows {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert y stride {} is below the {} rows of a slot",
                call.y_slot_stride, call.rows
            )));
        }
        let need = expert_needs(&call.shape())
            .ok_or_else(|| EngError::BadRequest("the CUDA expert size overflows".to_owned()))?;
        require_fit("w_stack", &call.w_stack, need.w_stack, 4)?;
        require_fit("ids", &call.ids, need.ids, 4)?;
        require_fit("x_q8", &call.x_q8, need.x_q8, 4)?;
        require_fit("y", &call.y, need.y, 4)?;
        let written = Claim::of("y", &call.y, need.y);
        require_apart(&written, &Claim::of("w_stack", &call.w_stack, need.w_stack))?;
        require_apart(&written, &Claim::of("ids", &call.ids, need.ids))?;
        require_apart(&written, &Claim::of("x_q8", &call.x_q8, need.x_q8))?;
        if let Some(gate) = &call.glu_gate {
            if call.x_per_slot {
                return Err(EngError::BadRequest(
                    "the CUDA expert GLU fusion takes the shared-x view only".to_owned(),
                ));
            }
            require_fit("glu_gate", gate, need.w_stack, 4)?;
            require_apart(&written, &Claim::of("glu_gate", gate, need.w_stack))?;
        }
        Ok(ExpertMatvecRaw {
            ty: call.ty,
            w_stack: w_raw,
            ids: ids_raw,
            x_q8: x_raw,
            x_per_slot: call.x_per_slot,
            y: y_raw,
            y_slot_stride: call.y_slot_stride,
            k: call.k,
            rows: call.rows,
            n_expert: call.n_expert,
            n_used: call.n_used,
            glu_gate: call
                .glu_gate
                .as_ref()
                .map(|g| self.raw_range(g))
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod tests;
