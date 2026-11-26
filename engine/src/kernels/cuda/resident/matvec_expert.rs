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
//!
//! The vendored MMVQ has a case for every [`QuantType`]. [`ExpertKernel::of`] routes a type
//! ([`QuantType::expert_uses_window`]): the types measured faster there run there, and every
//! other type runs the in-house window kernel (`native/resident/matvec_gguf.cu`) until a
//! measurement moves it: same layout and ids contract, but `rows` is any positive count and
//! `n_used` is in `1..=32` (Flash-Next routes ten experts per token).
//! [`CudaContext::matvec_expert_on`] names the kernel, for the comparisons.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::matvec_expert::{
    ExpertMatvecRaw, matvec_expert_launch,
};
use crate::kernels::cuda::native::resident::matvec_gguf::{
    ExpertGgufRaw, MAX_USED, matvec_gguf_expert_launch,
};
use crate::kernels::cuda::resident::matvec_q::{QuantType, ROWS_MULTIPLE};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

/// Per-launch limit of the vendored expert kernel, matching its native guard.
const VENDORED_MAX_USED: usize = 8;

/// The kernel a routed-expert matvec runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertKernel {
    /// The vendored MMVQ: `rows` a multiple of 4, up to 8 used slots per launch.
    Vendored,
    /// The in-house window kernel: any row count, up to 32 used slots per launch.
    Window,
}

impl ExpertKernel {
    /// The route of `ty`: [`QuantType::expert_uses_window`].
    pub(crate) fn of(ty: QuantType) -> Self {
        if ty.expert_uses_window() {
            Self::Window
        } else {
            Self::Vendored
        }
    }
}

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
        let kernel = ExpertKernel::of(call.ty);
        self.matvec_expert_on(stream, call, kernel)
    }

    /// [`Self::matvec_expert`] on a named kernel instead of the type's route.
    pub(crate) fn matvec_expert_on(
        &self,
        stream: &DeviceStream,
        call: ExpertMatvecCall<'_>,
        kernel: ExpertKernel,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if kernel == ExpertKernel::Window {
            let raw = self.plan_matvec_expert_codebook(&call)?;
            let row_bytes = q8_1_bytes(call.k, 1).expect("checked in plan");
            return self.launch_on(stream, |stream_raw| {
                matvec_gguf_expert_launch(stream_raw, &raw, row_bytes)
            });
        }
        let raw = self.plan_matvec_expert(&call)?;
        self.launch_on(stream, |stream_raw| matvec_expert_launch(stream_raw, &raw))
    }

    /// Whether [`Self::matvec_expert`] takes `call`: the checks it makes before queuing.
    pub(crate) fn matvec_expert_takes(&self, call: &ExpertMatvecCall<'_>) -> bool {
        if ExpertKernel::of(call.ty) == ExpertKernel::Window {
            self.plan_matvec_expert_codebook(call).is_ok()
        } else {
            self.plan_matvec_expert(call).is_ok()
        }
    }

    /// Queue a routed projection in supported per-format chunks. The original
    /// `matvec_expert` keeps its per-launch limits. Full-buffer claims are checked before
    /// any chunk is queued, so an earlier output cannot overwrite a later chunk's input.
    pub(crate) fn matvec_expert_chunks(
        &self,
        stream: &DeviceStream,
        call: ExpertMatvecCall<'_>,
    ) -> Result<(), EngError> {
        enum Plan<'a> {
            Window(ExpertGgufRaw<'a>),
            Vendored(ExpertMatvecRaw<'a>),
        }
        self.ensure_usable()?;
        if call.n_used == 0 || call.n_used > call.n_expert || call.y_slot_stride < call.rows {
            return Err(EngError::BadRequest(
                "invalid CUDA expert chunk geometry".into(),
            ));
        }
        let needs = expert_needs(&call.shape())
            .ok_or_else(|| EngError::BadRequest("the CUDA expert size overflows".into()))?;
        for range in [&call.w_stack, &call.ids, &call.x_q8, &call.y] {
            self.raw_range(range)?;
        }
        require_fit("w_stack", &call.w_stack, needs.w_stack, 4)?;
        require_fit("ids", &call.ids, needs.ids, 4)?;
        require_fit("x_q8", &call.x_q8, needs.x_q8, 4)?;
        require_fit("y", &call.y, needs.y, 4)?;
        let written = Claim::of("y", &call.y, needs.y);
        require_apart(
            &written,
            &Claim::of("w_stack", &call.w_stack, needs.w_stack),
        )?;
        require_apart(&written, &Claim::of("ids", &call.ids, needs.ids))?;
        require_apart(&written, &Claim::of("x_q8", &call.x_q8, needs.x_q8))?;
        if let Some(gate) = &call.glu_gate {
            self.raw_range(gate)?;
            require_fit("glu_gate", gate, needs.w_stack, 4)?;
            require_apart(&written, &Claim::of("glu_gate", gate, needs.w_stack))?;
        }
        let qrow = q8_1_bytes(call.k, 1)
            .ok_or_else(|| EngError::BadRequest("the CUDA expert size overflows".into()))?;
        let window = ExpertKernel::of(call.ty) == ExpertKernel::Window;
        let max = if window { MAX_USED } else { VENDORED_MAX_USED };
        let mut plans = Vec::new();
        for first in (0..call.n_used).step_by(max) {
            let used = (call.n_used - first).min(max);
            let y_bytes = ((used - 1) * call.y_slot_stride + call.rows) * 4;
            let (x_offset, x_bytes) = if call.x_per_slot {
                (first * qrow, used * qrow)
            } else {
                (0, qrow)
            };
            let chunk = ExpertMatvecCall {
                ty: call.ty,
                w_stack: call.w_stack,
                ids: call
                    .ids
                    .buffer()
                    .range(call.ids.offset() + first * 4, used * 4)?,
                x_q8: call
                    .x_q8
                    .buffer()
                    .range(call.x_q8.offset() + x_offset, x_bytes)?,
                x_per_slot: call.x_per_slot,
                y: call
                    .y
                    .buffer()
                    .range(call.y.offset() + first * call.y_slot_stride * 4, y_bytes)?,
                y_slot_stride: call.y_slot_stride,
                k: call.k,
                rows: call.rows,
                n_expert: call.n_expert,
                n_used: used,
                glu_gate: call.glu_gate,
            };
            plans.push(if window {
                Plan::Window(self.plan_matvec_expert_codebook(&chunk)?)
            } else {
                Plan::Vendored(self.plan_matvec_expert(&chunk)?)
            });
        }
        for plan in &plans {
            self.launch_on(stream, |stream_raw| match plan {
                Plan::Window(raw) => matvec_gguf_expert_launch(stream_raw, raw, qrow),
                Plan::Vendored(raw) => matvec_expert_launch(stream_raw, raw),
            })?;
        }
        Ok(())
    }

    fn plan_matvec_expert_codebook<'b>(
        &self,
        call: &ExpertMatvecCall<'b>,
    ) -> Result<ExpertGgufRaw<'b>, EngError> {
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
        if call.n_used < 1 || call.n_used > MAX_USED || call.n_used > call.n_expert {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert launch takes 1 to {} used slots, got {}",
                call.n_expert.min(MAX_USED),
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
        Ok(ExpertGgufRaw {
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
        if call.n_used < 1 || call.n_used > VENDORED_MAX_USED || call.n_used > call.n_expert {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert launch takes 1 to {} used slots, got {}",
                call.n_expert.min(VENDORED_MAX_USED),
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
