//! The tiled matmul against one expert of a stack, with the expert id read from device memory
//! when the kernel runs (`kern::matmul_expert`).
//!
//! [`CudaContext::matmul_expert`] queues the vendored kernel on a [`DeviceStream`] and returns
//! when it is queued. It allocates nothing, copies nothing and never synchronizes; a fault in the
//! queued work surfaces at the next synchronize of the stream. `ids[slot]` is read by the kernel,
//! so a captured graph needs no host-known id and the ids buffer may be rewritten between two
//! launches. The host never reads an id: one outside `0..count` makes the kernel write NaN to
//! every output and read no weight.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. In order: the
//! backend is usable; `ids`, `xf`, `y` and the stack belong to this context; `experts` divides
//! the weight into experts of whole 8-row groups; `1 <= count <= experts`; `ids` holds
//! `slot + 1` ints at a 4-byte offset; `m` is in `1..=8` and the shape does not split along k
//! (the vendored `split_k` decides, through the adapter); `xf` and `y` hold what the call uses;
//! `y`, which the kernel writes, overlaps neither `xf`, the experts' tiles nor the ids. A refused
//! call changes nothing.
//!
//! The weight type picks the kernel: a Q5_0 stack runs the own kernel of
//! `native/resident/matmul_q5_0.cu`, every other type the vendored `matmul_expert`. Q5_0 is a
//! weight format, not another operation, so the resident step calls this one method whatever the
//! type. The checks above are shared; only the shape rule differs (`q5_0_shape_ok`: k a multiple
//! of 128, no split along k) and `xf` holds a row of `native::activation_k` values (K = 896 gives
//! a 1024-wide fragment-order buffer with a zero tail the kernel never reads).

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::matmul_expert::{
    ExpertRaw, expert_shape_ok, matmul_expert_launch,
};
use crate::kernels::cuda::native::resident::matmul_q5_0::{
    matmul_expert_q5_0_launch, q5_0_shape_ok,
};
use crate::kernels::cuda::native::{self, TileVec, TiledType};
use crate::kernels::cuda::{
    ADAPTER_OPERATION, CudaContext, DeviceRange, DeviceStream, MatmulIo, TERMINAL, TiledWeight,
    native_error,
};

/// The most tokens an expert launch takes (`kernels.h`).
const MAX_EXPERT_TOKENS: usize = 8;

/// Which expert of a stack: expert `ids[slot]`, read on the device when the kernel runs, valid in
/// `0..count`.
pub(crate) struct ExpertSelect<'b> {
    pub(crate) ids: DeviceRange<'b>,
    pub(crate) slot: usize,
    pub(crate) count: usize,
}

/// A stack of `experts` experts of equal shape, back to back in one tiled weight.
pub(crate) struct ExpertStack<'a> {
    pub(crate) weight: &'a TiledWeight,
    pub(crate) experts: usize,
}

/// The stack, the expert selection, and the packed input and output of `io.m` tokens. `io.y`
/// holds `io.m` rows of one expert's rows.
pub(crate) struct ExpertCall<'b> {
    pub(crate) stack: ExpertStack<'b>,
    pub(crate) select: ExpertSelect<'b>,
    pub(crate) io: MatmulIo<'b>,
}

impl CudaContext {
    /// Queues the matmul of `call.io.xf` against expert `ids[slot]` of the stack on `stream`,
    /// writing `call.io.y`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a stack that does not
    /// split into `experts` experts of whole 8-row groups, for `count` outside `1..=experts`, for
    /// an ids range that is short or misaligned, for `m` outside `1..=8` or a shape that splits
    /// along k, for an `xf` or `y` range that is short or misaligned, and for a `y` that
    /// overlaps `xf`, the experts' tiles or the ids.
    pub(crate) fn matmul_expert(
        &self,
        stream: &DeviceStream,
        call: ExpertCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_expert(&call)?;
        self.launch_on(stream, |stream_raw| match raw.ty {
            TiledType::Q50 => matmul_expert_q5_0_launch(stream_raw, &raw),
            TiledType::Q4K | TiledType::Q6K | TiledType::Q80 => {
                matmul_expert_launch(stream_raw, &raw)
            }
        })
    }

    /// Every check of an expert matmul, then the native view of its ranges.
    fn plan_expert<'b>(&self, call: &ExpertCall<'b>) -> Result<ExpertRaw<'b>, EngError> {
        let (ids, xf, y) = (&call.select.ids, &call.io.xf, &call.io.y);
        let (ids_raw, xf_raw, y_raw) = (
            self.raw_range(ids)?,
            self.raw_range(xf)?,
            self.raw_range(y)?,
        );
        let weight: &'b TiledWeight = call.stack.weight;
        let rows = expert_rows(weight, call.stack.experts)?;
        let (slot, count, m) = (call.select.slot, call.select.count, call.io.m);
        if count < 1 || count > call.stack.experts {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert launch needs 1 to {} experts, got {count}",
                call.stack.experts
            )));
        }
        check_c_ints(&[
            ("slot", slot),
            ("count", count),
            ("rows", rows),
            ("k", weight.k()),
            ("m", m),
        ])?;
        let ids_need = slot
            .checked_add(1)
            .and_then(|slots| slots.checked_mul(size_of::<i32>()))
            .ok_or_else(|| EngError::BadRequest("the CUDA expert slot overflows".to_owned()))?;
        require_fit("ids", ids, ids_need, size_of::<i32>())?;
        let k = weight.k();
        let ty = weight.layout().ty;
        check_shape(ty, m, rows, k)?;
        let need = native::activation_bytes(rows, native::activation_k(ty, k), m)
            .map_err(|status| native_error(&TERMINAL, ADAPTER_OPERATION, status))?;
        require_fit("xf", xf, need.xf, 16)?;
        require_fit("y", y, need.y, 4)?;
        let stack_bytes = expert_bytes(weight, rows)?
            .checked_mul(count)
            .ok_or_else(|| {
                EngError::BadRequest("the CUDA expert stack size overflows".to_owned())
            })?;
        let stack = weight.buffer().range(0, stack_bytes)?;
        let stack_raw = self.raw_range(&stack)?;
        let tiles = Claim::of("stack", &stack, stack_bytes);
        let written = Claim::of("y", y, need.y);
        require_apart(&written, &Claim::of("xf", xf, need.xf))?;
        require_apart(&written, &Claim::of("ids", ids, ids_need))?;
        require_apart(&written, &tiles)?;
        Ok(ExpertRaw {
            ty,
            stack: stack_raw,
            ids: ids_raw,
            slot,
            count,
            xf: xf_raw,
            y: y_raw,
            m,
            rows,
            k,
        })
    }
}

/// Rows of one expert: the weight split into `experts` equal experts of whole 8-row groups.
fn expert_rows(weight: &TiledWeight, experts: usize) -> Result<usize, EngError> {
    let total = weight.n_rows();
    if experts < 1 || !total.is_multiple_of(experts) || !(total / experts).is_multiple_of(8) {
        return Err(EngError::BadRequest(format!(
            "the CUDA weight of {total} rows does not split into {experts} experts of whole 8-row groups"
        )));
    }
    Ok(total / experts)
}

/// `BadRequest` unless `m` tokens against an expert of `rows` x `k` values of type `ty` are a
/// decode shape.
fn check_shape(ty: TiledType, m: usize, rows: usize, k: usize) -> Result<(), EngError> {
    if !(1..=MAX_EXPERT_TOKENS).contains(&m) {
        return Err(EngError::BadRequest(format!(
            "the CUDA expert launch takes 1 to {MAX_EXPERT_TOKENS} tokens, got {m}"
        )));
    }
    let shape_ok = match ty {
        TiledType::Q50 => q5_0_shape_ok(m, rows, k, true),
        TiledType::Q4K | TiledType::Q6K | TiledType::Q80 => expert_shape_ok(m, rows, k),
    };
    if !shape_ok {
        return Err(EngError::BadRequest(format!(
            "the CUDA expert launch needs a decode shape; {m} tokens against {rows} rows of {k} \
             values split along k or are not a tiled shape"
        )));
    }
    Ok(())
}

/// `BadRequest` unless every value the native layer converts to a C int fits one. A failed
/// conversion there is an adapter failure and latches the backend, so a caller mistake such as
/// a huge `slot` (which `require_fit` alone only bounds by the size of the ids range) must stop
/// here. Checked: slot, count, rows, k and m (the five conversions of `call_args`).
fn check_c_ints(values: &[(&str, usize)]) -> Result<(), EngError> {
    for &(name, value) in values {
        if i32::try_from(value).is_err() {
            return Err(EngError::BadRequest(format!(
                "the CUDA expert {name} {value} does not fit a C int"
            )));
        }
    }
    Ok(())
}

/// Bytes of the tiles of one expert of `rows` rows.
fn expert_bytes(weight: &TiledWeight, rows: usize) -> Result<usize, EngError> {
    let layout = weight.layout();
    let one = native::Tiled {
        n_rows: rows,
        ..layout
    };
    one.tile_vecs()
        .and_then(|vecs| vecs.checked_mul(size_of::<TileVec>()))
        .ok_or_else(|| EngError::BadRequest("the CUDA expert size overflows".to_owned()))
}

#[cfg(test)]
mod tests;
