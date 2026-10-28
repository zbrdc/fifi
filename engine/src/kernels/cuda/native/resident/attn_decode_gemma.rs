//! Raw FFI to inference_cuda_attn_decode_gemma_launch of
//! native/resident/attn_decode_gemma.cu (declared in native/resident/
//! attn_decode_gemma.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves the code, scale,
//!   q, out and kv_len sizes and alignment, and that out, which the kernel writes,
//!   shares no byte with q, the code planes or the scale planes. The C entry repeats
//!   the checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid
//!   until then because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread
//!   (decisions 0023 and 0024). No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_decode_gemma_launch"]
    fn attn_decode_gemma_launch_raw(
        args: *const AttnDecodeGemmaArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_attn_decode_gemma_args, field for field.
#[repr(C)]
struct AttnDecodeGemmaArgsRepr {
    out: *mut c_void,
    status: *mut c_void,
    q: *const c_void,
    k_codes: *const c_void,
    k_scales: *const c_void,
    v_codes: *const c_void,
    v_scales: *const c_void,
    kv_len: *const c_void,
    m: c_int,
    n_q_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    window: c_int,
    scale: f32,
}

const _: () = assert!(
    size_of::<AttnDecodeGemmaArgsRepr>() == 96
        && offset_of!(AttnDecodeGemmaArgsRepr, out) == 0
        && offset_of!(AttnDecodeGemmaArgsRepr, status) == 8
        && offset_of!(AttnDecodeGemmaArgsRepr, q) == 16
        && offset_of!(AttnDecodeGemmaArgsRepr, k_codes) == 24
        && offset_of!(AttnDecodeGemmaArgsRepr, k_scales) == 32
        && offset_of!(AttnDecodeGemmaArgsRepr, v_codes) == 40
        && offset_of!(AttnDecodeGemmaArgsRepr, v_scales) == 48
        && offset_of!(AttnDecodeGemmaArgsRepr, kv_len) == 56
        && offset_of!(AttnDecodeGemmaArgsRepr, m) == 64
        && offset_of!(AttnDecodeGemmaArgsRepr, n_q_head) == 68
        && offset_of!(AttnDecodeGemmaArgsRepr, n_kv_head) == 72
        && offset_of!(AttnDecodeGemmaArgsRepr, head_dim) == 76
        && offset_of!(AttnDecodeGemmaArgsRepr, cells) == 80
        && offset_of!(AttnDecodeGemmaArgsRepr, window) == 84
        && offset_of!(AttnDecodeGemmaArgsRepr, scale) == 88
);

/// One attention launch over the KV8 planes.
pub(in crate::kernels::cuda) struct AttnDecodeGemmaRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) k_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) k_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) v_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) v_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) kv_len: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n_q_head: usize,
    pub(in crate::kernels::cuda) n_kv_head: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
    pub(in crate::kernels::cuda) cells: usize,
    pub(in crate::kernels::cuda) window: usize,
    pub(in crate::kernels::cuda) scale: f32,
}

fn plane_bytes(kv_heads: usize, cells: usize, head_dim: usize) -> Option<(usize, usize)> {
    let codes = kv_heads.checked_mul(cells)?.checked_mul(head_dim)?;
    let scales = kv_heads
        .checked_mul(cells)?
        .checked_mul(head_dim.checked_div(32)?)?
        .checked_mul(2)?;
    Some((codes, scales))
}

fn check_attn_decode_gemma(call: &AttnDecodeGemmaRaw<'_>) -> Result<(), DeviceFailure> {
    let (codes, scales) =
        plane_bytes(call.n_kv_head, call.cells, call.head_dim).ok_or_else(invalid_argument)?;
    let q_bytes = call
        .m
        .checked_mul(call.n_q_head)
        .and_then(|v| v.checked_mul(call.head_dim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let dims = (1..=16).contains(&call.m)
        && call.n_q_head == 16
        && matches!(call.n_kv_head, 1 | 8)
        && matches!(call.head_dim, 256 | 512)
        && (1..=i32::MAX as usize).contains(&call.cells)
        && call.window <= call.cells
        && call.scale.is_finite()
        && call.scale.abs() < 1e30;
    let reads = [
        (&call.q, q_bytes, 4),
        (&call.k_codes, codes, 1),
        (&call.k_scales, scales, 2),
        (&call.v_codes, codes, 1),
        (&call.v_scales, scales, 2),
        (&call.kv_len, 4, 4),
    ];
    if !dims
        || call.out.bytes < q_bytes
        || !call.out.offset.is_multiple_of(4)
        || reads.iter().any(|(range, bytes, alignment)| {
            range.bytes < *bytes || !range.offset.is_multiple_of(*alignment)
        })
    {
        return Err(invalid_argument());
    }
    let output = Span::of(&call.out, q_bytes);
    if reads
        .iter()
        .any(|(range, bytes, _)| !output.apart(&Span::of(range, *bytes)))
    {
        return Err(invalid_argument());
    }
    if let Some(status) = &call.status {
        let status_span = Span::of(status, 4);
        if status.bytes < 4
            || !status.offset.is_multiple_of(4)
            || !status_span.apart(&output)
            || reads
                .iter()
                .any(|(range, bytes, _)| !status_span.apart(&Span::of(range, *bytes)))
        {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Queues the attention of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn attn_decode_gemma_launch(
    stream: &RawStream,
    call: &AttnDecodeGemmaRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn_decode_gemma(call)?;
    let args = AttnDecodeGemmaArgsRepr {
        out: call.out.address(),
        status: call
            .status
            .as_ref()
            .map(|s| s.address())
            .unwrap_or(std::ptr::null_mut()),
        q: call.q.address().cast_const(),
        k_codes: call.k_codes.address().cast_const(),
        k_scales: call.k_scales.address().cast_const(),
        v_codes: call.v_codes.address().cast_const(),
        v_scales: call.v_scales.address().cast_const(),
        kv_len: call.kv_len.address().cast_const(),
        m: c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        n_q_head: c_int::try_from(call.n_q_head).map_err(|_| invalid_argument())?,
        n_kv_head: c_int::try_from(call.n_kv_head).map_err(|_| invalid_argument())?,
        head_dim: c_int::try_from(call.head_dim).map_err(|_| invalid_argument())?,
        cells: c_int::try_from(call.cells).map_err(|_| invalid_argument())?,
        window: c_int::try_from(call.window).map_err(|_| invalid_argument())?,
        scale: call.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_attn_decode_gemma proved the ranges inside their live allocations,
    // large enough, aligned, and out disjoint from q and the code planes; the adapter
    // repeats the checks on the addresses. The allocations stay valid until the queued
    // launch finishes because inference_cuda_free synchronizes the device first.
    let status = unsafe { attn_decode_gemma_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
