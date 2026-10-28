//! Raw FFI to inference_cuda_embed_q8_launch of native/resident/embed_q8.cu.
//! Child of native::resident.
//!
//! Safety invariants: RawRange::new proved each range in bounds; this module
//! re-proves sizes and alignment and that out is disjoint from the table and the
//! ids; the C entry repeats the checks. Launch-only; the dispatch guard holds
//! (decisions 0023 and 0024).

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_embed_q8_launch"]
    fn embed_q8_launch_raw(
        args: *const EmbedQ8ArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_embed_q8_args, field for field.
#[repr(C)]
struct EmbedQ8ArgsRepr {
    out: *mut c_void,
    w: *const c_void,
    ids: *const c_void,
    m: c_int,
    n_rows: c_int,
    k: c_int,
}

const _: () = assert!(
    size_of::<EmbedQ8ArgsRepr>() == 40
        && offset_of!(EmbedQ8ArgsRepr, out) == 0
        && offset_of!(EmbedQ8ArgsRepr, w) == 8
        && offset_of!(EmbedQ8ArgsRepr, ids) == 16
        && offset_of!(EmbedQ8ArgsRepr, m) == 24
        && offset_of!(EmbedQ8ArgsRepr, n_rows) == 28
        && offset_of!(EmbedQ8ArgsRepr, k) == 32
);

/// One embed_q8 launch.
pub(in crate::kernels::cuda) struct EmbedQ8Raw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n_rows: usize,
    pub(in crate::kernels::cuda) k: usize,
}

fn check_embed_q8(call: &EmbedQ8Raw<'_>) -> Result<(), DeviceFailure> {
    let out_bytes = call
        .m
        .checked_mul(call.k)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let w_bytes = call
        .n_rows
        .checked_mul(call.k / 32)
        .and_then(|v| v.checked_mul(34))
        .ok_or_else(invalid_argument)?;
    let ids_bytes = call.m.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized =
        call.out.bytes >= out_bytes && call.w.bytes >= w_bytes && call.ids.bytes >= ids_bytes;
    let aligned = call.out.offset.is_multiple_of(16)
        && call.w.offset.is_multiple_of(4)
        && call.ids.offset.is_multiple_of(4);
    let out = Span::of(&call.out, out_bytes);
    let apart =
        out.apart(&Span::of(&call.w, w_bytes)) && out.apart(&Span::of(&call.ids, ids_bytes));
    let dims = (1..=4096).contains(&call.m)
        && (1..=(1 << 28)).contains(&call.n_rows)
        && (32..=32_768).contains(&call.k)
        && call.k.is_multiple_of(32);
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the gather of call on stream.
pub(in crate::kernels::cuda) fn embed_q8_launch(
    stream: &RawStream,
    call: &EmbedQ8Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_embed_q8(call)?;
    let args = EmbedQ8ArgsRepr {
        out: call.out.address(),
        w: call.w.address().cast_const(),
        ids: call.ids.address().cast_const(),
        m: c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        n_rows: c_int::try_from(call.n_rows).map_err(|_| invalid_argument())?,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_embed_q8 proved sizes, alignment and the write/read separation;
    // the adapter repeats them on the addresses.
    let status = unsafe { embed_q8_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
