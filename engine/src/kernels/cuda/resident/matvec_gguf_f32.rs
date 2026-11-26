//! The exact f32 twins of the in-house GGUF matvec (`native/resident/matvec_gguf.cu`, the
//! `inference_cuda_matvec_gguf_f32_*` entries): [`CudaContext::matvec_gguf_f32`] queues one
//! dense launch and [`CudaContext::matvec_gguf_f32_expert`] one launch for the used slots of an
//! expert stack; both return when the work is queued. The weights are the file's native blocks
//! of any type the window decoders serve; the activation is the f32 row itself, never
//! quantized, and every product is the decoded weight times the activation in one fused
//! multiply-add. They allocate nothing, copy nothing and never synchronize.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend
//! is usable; `k` is a positive multiple of the type's block size; the counts are in range
//! (`rows` positive and `ncols` 1..=8 for the dense call; `rows` and `n_expert` positive,
//! `n_used` in `1..=MAX_USED` and at most `n_expert`, and `y_slot_stride >= rows` for the
//! expert call) and each fits the kernel's `int`; every range holds the bytes the call uses and
//! starts 4-byte aligned, `x` 16-byte aligned (a lane reads its activation window as 16-byte
//! words); `y` overlaps none of what the kernel reads. A gate stack needs the shared activation
//! row. A refused call changes nothing. The ids contract is the one of [`super::matvec_expert`]:
//! an id outside `[0, n_expert)` writes NaN rows and reads no weight.

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::matvec_gguf::{
    ExpertGgufF32Raw, MAX_USED, MatvecGgufF32Raw, matvec_gguf_f32_expert_launch,
    matvec_gguf_f32_launch,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;

const MAX_COLS: usize = 8;

/// One exact dense projection: `y[j * rows + r] = sum_i x[j][i] * w[r][i]`.
pub(crate) struct DenseF32Call<'a> {
    pub(crate) ty: QuantType,
    /// Native blocks `[rows][k / block_size]`.
    pub(crate) w: DeviceRange<'a>,
    /// The f32 activation `[ncols][k]`, packed.
    pub(crate) x: DeviceRange<'a>,
    /// The f32 output `[ncols][rows]`.
    pub(crate) y: DeviceRange<'a>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
}

/// One exact expert projection. Slot `s` computes `y[s * y_slot_stride .. + rows)` from the
/// weights of expert `ids[s]` (read on the device when the kernel runs) and activation row `s`
/// (`x_per_slot`) or row 0; with `glu_gate`, each element is `up * silu(gate)` of the gate stack.
/// The stack layout and the rest of the fields are those of
/// [`super::matvec_expert::ExpertMatvecCall`], with `x` f32.
pub(crate) struct ExpertF32Call<'a> {
    pub(crate) ty: QuantType,
    pub(crate) w_stack: DeviceRange<'a>,
    pub(crate) ids: DeviceRange<'a>,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) x_per_slot: bool,
    pub(crate) y: DeviceRange<'a>,
    pub(crate) y_slot_stride: usize,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) n_expert: usize,
    pub(crate) n_used: usize,
    pub(crate) glu_gate: Option<DeviceRange<'a>>,
}

struct DenseNeeds {
    w: usize,
    x: usize,
    y: usize,
}

/// The geometry of an expert call, apart from its ranges.
#[derive(Clone, Copy)]
struct ExpertShape {
    ty: QuantType,
    k: usize,
    rows: usize,
    n_expert: usize,
    n_used: usize,
    x_per_slot: bool,
    y_slot_stride: usize,
    gate: bool,
}

struct ExpertNeeds {
    w_stack: usize,
    ids: usize,
    x: usize,
    y: usize,
}

fn bad(detail: impl AsRef<str>) -> EngError {
    EngError::BadRequest(format!(
        "the CUDA exact GGUF projection {}",
        detail.as_ref()
    ))
}

/// `k` a positive multiple of the type's block size.
fn check_k(ty: QuantType, k: usize) -> Result<(), EngError> {
    if k == 0 || !k.is_multiple_of(ty.block_size()) {
        return Err(bad(format!(
            "needs k as a multiple of {}, got {k}",
            ty.block_size()
        )));
    }
    Ok(())
}

/// Every named count fits the kernel's `int`.
fn check_counts(counts: &[(&str, usize)]) -> Result<(), EngError> {
    for &(name, count) in counts {
        if c_int::try_from(count).is_err() {
            return Err(bad(format!("needs {name} to fit an int, got {count}")));
        }
    }
    Ok(())
}

/// The dense call's geometry and the bytes it reads and writes, before any range is looked at.
fn dense_needs(ty: QuantType, k: usize, rows: usize, ncols: usize) -> Result<DenseNeeds, EngError> {
    check_k(ty, k)?;
    if rows == 0 {
        return Err(bad("needs a positive row count"));
    }
    if !(1..=MAX_COLS).contains(&ncols) {
        return Err(bad(format!("takes 1 to {MAX_COLS} columns, got {ncols}")));
    }
    check_counts(&[("k", k), ("rows", rows)])?;
    let (Some(w), Some(x), Some(y)) = (
        ty.weight_bytes(rows, k),
        4usize.checked_mul(k).and_then(|n| n.checked_mul(ncols)),
        rows.checked_mul(ncols).and_then(|n| n.checked_mul(4)),
    ) else {
        return Err(bad("size overflows"));
    };
    Ok(DenseNeeds { w, x, y })
}

impl ExpertF32Call<'_> {
    fn shape(&self) -> ExpertShape {
        ExpertShape {
            ty: self.ty,
            k: self.k,
            rows: self.rows,
            n_expert: self.n_expert,
            n_used: self.n_used,
            x_per_slot: self.x_per_slot,
            y_slot_stride: self.y_slot_stride,
            gate: self.glu_gate.is_some(),
        }
    }
}

/// The expert call's geometry and the bytes it reads and writes, before any range is looked
/// at.
fn expert_needs(call: &ExpertShape) -> Result<ExpertNeeds, EngError> {
    check_k(call.ty, call.k)?;
    if call.rows < 1 || call.n_expert < 1 {
        return Err(bad("needs positive rows and expert counts"));
    }
    if !(1..=MAX_USED).contains(&call.n_used) || call.n_used > call.n_expert {
        return Err(bad(format!(
            "takes 1 to {MAX_USED} used slots inside the stack, got {} of {}",
            call.n_used, call.n_expert
        )));
    }
    if call.y_slot_stride < call.rows {
        return Err(bad(format!(
            "needs y_slot_stride of at least {}, got {}",
            call.rows, call.y_slot_stride
        )));
    }
    if call.x_per_slot && call.gate {
        return Err(bad(
            "takes a gate stack only with the shared activation row",
        ));
    }
    check_counts(&[
        ("k", call.k),
        ("rows", call.rows),
        ("n_expert", call.n_expert),
        ("y_slot_stride", call.y_slot_stride),
    ])?;
    let x_rows = if call.x_per_slot { call.n_used } else { 1 };
    let y_floats = call
        .y_slot_stride
        .checked_mul(call.n_used - 1)
        .and_then(|n| n.checked_add(call.rows));
    let Some(expert_rows) = call.rows.checked_mul(call.n_expert) else {
        return Err(bad("size overflows"));
    };
    let (Some(w_stack), Some(x), Some(y_floats)) = (
        call.ty.weight_bytes(expert_rows, call.k),
        4usize
            .checked_mul(call.k)
            .and_then(|n| n.checked_mul(x_rows)),
        y_floats,
    ) else {
        return Err(bad("size overflows"));
    };
    Ok(ExpertNeeds {
        w_stack,
        ids: call.n_used * 4,
        x,
        y: y_floats * 4,
    })
}

impl CudaContext {
    /// The exact dense projection of `call`: one launch, every column in it.
    pub(crate) fn matvec_gguf_f32(
        &self,
        stream: &DeviceStream,
        call: DenseF32Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let need = dense_needs(call.ty, call.k, call.rows, call.ncols)?;
        require_fit("w", &call.w, need.w, 4)?;
        require_fit("x", &call.x, need.x, 16)?;
        require_fit("y", &call.y, need.y, 4)?;
        let written = Claim::of("y", &call.y, need.y);
        require_apart(&written, &Claim::of("w", &call.w, need.w))?;
        require_apart(&written, &Claim::of("x", &call.x, need.x))?;
        let raw = MatvecGgufF32Raw {
            ty: call.ty,
            w: self.raw_range(&call.w)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            k: call.k,
            rows: call.rows,
            ncols: call.ncols,
        };
        self.launch_on(stream, |stream_raw| {
            matvec_gguf_f32_launch(stream_raw, &raw)
        })
    }

    /// The exact expert projection of `call`: one launch for all the used slots.
    pub(crate) fn matvec_gguf_f32_expert(
        &self,
        stream: &DeviceStream,
        call: ExpertF32Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let need = expert_needs(&call.shape())?;
        require_fit("w_stack", &call.w_stack, need.w_stack, 4)?;
        require_fit("ids", &call.ids, need.ids, 4)?;
        require_fit("x", &call.x, need.x, 16)?;
        require_fit("y", &call.y, need.y, 4)?;
        let written = Claim::of("y", &call.y, need.y);
        require_apart(&written, &Claim::of("w_stack", &call.w_stack, need.w_stack))?;
        require_apart(&written, &Claim::of("x", &call.x, need.x))?;
        require_apart(&written, &Claim::of("ids", &call.ids, need.ids))?;
        if let Some(gate) = &call.glu_gate {
            require_fit("glu_gate", gate, need.w_stack, 4)?;
            require_apart(&written, &Claim::of("glu_gate", gate, need.w_stack))?;
        }
        let raw = ExpertGgufF32Raw {
            ty: call.ty,
            w_stack: self.raw_range(&call.w_stack)?,
            ids: self.raw_range(&call.ids)?,
            x: self.raw_range(&call.x)?,
            x_per_slot: call.x_per_slot,
            y: self.raw_range(&call.y)?,
            y_slot_stride: call.y_slot_stride,
            k: call.k,
            rows: call.rows,
            n_expert: call.n_expert,
            n_used: call.n_used,
            glu_gate: match &call.glu_gate {
                Some(gate) => Some(self.raw_range(gate)?),
                None => None,
            },
        };
        self.launch_on(stream, |stream_raw| {
            matvec_gguf_f32_expert_launch(stream_raw, &raw)
        })
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
