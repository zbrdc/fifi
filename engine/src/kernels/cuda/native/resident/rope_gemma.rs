//! Raw FFI to inference_cuda_rope_gemma_launch of native/resident/rope_gemma.cu
//! (declared in native/resident/rope_gemma.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves the sizes,
//!   alignment, and the read/write split: q and k are rotated in place, pos and the
//!   optional factor table are read. The C entry repeats the checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid
//!   until then because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread
//!   (decisions 0023 and 0024). No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_rope_gemma_launch"]
    fn rope_gemma_launch_raw(
        args: *const RopeGemmaArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_rope_gemma_args, field for field.
#[repr(C)]
struct RopeGemmaArgsRepr {
    q: *mut c_void,
    k: *mut c_void,
    pos: *const c_void,
    factors: *const c_void,
    rows_q: c_int,
    rows_k: c_int,
    n_q_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    global_mode: c_int,
}

const _: () = assert!(
    size_of::<RopeGemmaArgsRepr>() == 56
        && offset_of!(RopeGemmaArgsRepr, q) == 0
        && offset_of!(RopeGemmaArgsRepr, k) == 8
        && offset_of!(RopeGemmaArgsRepr, pos) == 16
        && offset_of!(RopeGemmaArgsRepr, factors) == 24
        && offset_of!(RopeGemmaArgsRepr, rows_q) == 32
        && offset_of!(RopeGemmaArgsRepr, rows_k) == 36
        && offset_of!(RopeGemmaArgsRepr, n_q_head) == 40
        && offset_of!(RopeGemmaArgsRepr, n_kv_head) == 44
        && offset_of!(RopeGemmaArgsRepr, head_dim) == 48
        && offset_of!(RopeGemmaArgsRepr, global_mode) == 52
);

/// One rope_gemma launch: q and k rows in place, the position word, and the factor table.
pub(in crate::kernels::cuda) struct RopeGemmaRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) factors: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) rows_q: usize,
    pub(in crate::kernels::cuda) rows_k: usize,
    pub(in crate::kernels::cuda) n_q_head: usize,
    pub(in crate::kernels::cuda) n_kv_head: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
    pub(in crate::kernels::cuda) global_mode: bool,
}

fn check_rope_gemma(call: &RopeGemmaRaw<'_>) -> Result<(), DeviceFailure> {
    let q_bytes = call
        .rows_q
        .checked_mul(call.n_q_head)
        .and_then(|v| v.checked_mul(call.head_dim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let k_bytes = call
        .rows_k
        .checked_mul(call.n_kv_head)
        .and_then(|v| v.checked_mul(call.head_dim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let f_bytes = call.head_dim.checked_div(2).and_then(|v| v.checked_mul(4));
    let dims = (1..=16).contains(&call.rows_q)
        && (1..=16).contains(&call.rows_k)
        && call.n_q_head == 16
        && (call.n_kv_head == 1 || call.n_kv_head == 8)
        && matches!(
            (call.head_dim, call.global_mode),
            (256, false) | (512, true)
        );
    let sized = call.q.bytes >= q_bytes
        && call.k.bytes >= k_bytes
        && call.pos.bytes >= 4
        && f_bytes.is_none_or(|need| call.factors.as_ref().is_none_or(|f| f.bytes >= need));
    let aligned = call.q.offset.is_multiple_of(4)
        && call.k.offset.is_multiple_of(4)
        && call.pos.offset.is_multiple_of(4)
        && call
            .factors
            .as_ref()
            .is_none_or(|f| f.offset.is_multiple_of(4));
    let q = Span::of(&call.q, q_bytes);
    let k = Span::of(&call.k, k_bytes);
    let pos = Span::of(&call.pos, 4);
    let apart = q.apart(&k)
        && q.apart(&pos)
        && k.apart(&pos)
        && call.factors.as_ref().is_none_or(|f| {
            let factors = Span::of(f, call.head_dim / 2 * 4);
            q.apart(&factors) && k.apart(&factors)
        });
    if !(dims && sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the rope_gemma of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn rope_gemma_launch(
    stream: &RawStream,
    call: &RopeGemmaRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_rope_gemma(call)?;
    let args = RopeGemmaArgsRepr {
        q: call.q.address(),
        k: call.k.address(),
        pos: call.pos.address().cast_const(),
        factors: call
            .factors
            .as_ref()
            .map(|f| f.address().cast_const())
            .unwrap_or(std::ptr::null()),
        rows_q: c_int::try_from(call.rows_q).map_err(|_| invalid_argument())?,
        rows_k: c_int::try_from(call.rows_k).map_err(|_| invalid_argument())?,
        n_q_head: c_int::try_from(call.n_q_head).map_err(|_| invalid_argument())?,
        n_kv_head: c_int::try_from(call.n_kv_head).map_err(|_| invalid_argument())?,
        head_dim: c_int::try_from(call.head_dim).map_err(|_| invalid_argument())?,
        global_mode: c_int::from(call.global_mode),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_rope_gemma proved the ranges inside their live allocations, large
    // enough, aligned, and q apart from k; the adapter repeats the checks on the
    // addresses. The allocations stay valid until the queued launch finishes because
    // inference_cuda_free synchronizes the device first.
    let status = unsafe { rope_gemma_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
