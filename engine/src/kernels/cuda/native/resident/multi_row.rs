//! Raw FFI to the many-rows entries of the resident decode step: `native/resident/multi_row.cu`
//! (the conv, delta and attention-prep entries), `kv8_append.cu` and `attn_decode_q8.cu` (the
//! append and decode entries), all declared in `native/resident/multi_row.h`. One launch advances
//! `n` rows, each the single token of a sequence of its own; the recurrent state or the KV planes
//! of a row come from a device table. Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entries are called only through `CudaContext::launch_on`, on the owning thread and under
//!   the dispatch guard. They only enqueue and return the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) of the ranges a call names are proved again here, on the extents
//!   the kernels touch; the C entries repeat the alignment and shape checks on the addresses.
//! - A table range holds the addresses or records the safe layer wrote from live, proved ranges of
//!   the same context (`resident::multi_row`); that layer owns the proof that those addresses stay
//!   inside live allocations, apart where a kernel writes, and sized for the geometry. This file
//!   checks only that the table range itself holds `n` entries at the alignment the kernel reads.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The C argument structs are live locals that an entry reads only during the call (the kernel
//!   arguments are copied at launch). Their `repr(C)` mirrors are private here and pinned by
//!   `const` asserts to the field order of `multi_row.h`. No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Most rows of one launch (`INFERENCE_CUDA_MULTI_ROW_MAX`).
pub(in crate::kernels::cuda) const MULTI_ROW_MAX: usize = 256;
/// Bytes of one entry of a state table (an address) and of a KV row table (`inference_cuda_kv_row`).
pub(in crate::kernels::cuda) const STATE_ENTRY_BYTES: usize = 8;
pub(in crate::kernels::cuda) const KV_ROW_BYTES: usize = 64;
/// Bytes of the `StepParams` block a row names (position at byte 4, `kv_len` at 8, status at 16).
pub(in crate::kernels::cuda) const STEP_BYTES: usize = 32;

/// Mirror of `inference_cuda_kv_row`; the safe layer writes this layout into a KV row table.
#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::kernels::cuda) struct KvRowRaw {
    pub(in crate::kernels::cuda) k_codes: u64,
    pub(in crate::kernels::cuda) k_scales: u64,
    pub(in crate::kernels::cuda) v_codes: u64,
    pub(in crate::kernels::cuda) v_scales: u64,
    pub(in crate::kernels::cuda) step: u64,
    pub(in crate::kernels::cuda) cells: u64,
    pub(in crate::kernels::cuda) reserved: [u64; 2],
}

const _: () = assert!(
    size_of::<KvRowRaw>() == KV_ROW_BYTES
        && offset_of!(KvRowRaw, step) == 32
        && offset_of!(KvRowRaw, cells) == 40
);

#[repr(C)]
struct ConvArgsRaw {
    proj: *const c_void,
    w: *const c_void,
    states: *const *mut c_void,
    out: *mut c_void,
    n: c_int,
    ld: c_int,
    channels: c_int,
}

#[repr(C)]
struct DeltaArgsRaw {
    conv: *const c_void,
    proj: *const c_void,
    a: *const c_void,
    dt_bias: *const c_void,
    norm_w: *const c_void,
    states: *const *mut c_void,
    out: *mut c_void,
    n: c_int,
    conv_ld: c_int,
    ld: c_int,
    z_off: c_int,
    beta_off: c_int,
    alpha_off: c_int,
    n_k_heads: c_int,
    n_v_heads: c_int,
    eps: f32,
}

#[repr(C)]
struct PrepArgsRaw {
    proj: *const c_void,
    q_norm: *const c_void,
    k_norm: *const c_void,
    k_rows: *mut c_void,
    v_rows: *mut c_void,
    q: *mut c_void,
    rows: *const c_void,
    n: c_int,
    ld: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    n_rot: c_int,
    cells: c_int,
    eps: f32,
    rope_base: f32,
}

#[repr(C)]
struct AppendArgsRaw {
    k: *const c_void,
    v: *const c_void,
    rows: *const c_void,
    n: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    code_scale: c_int,
    ring: c_int,
}

#[repr(C)]
struct GemmaArgsRaw {
    q: *const c_void,
    rows: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    n: c_int,
    n_q_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    window: c_int,
    split: c_int,
    max_splits: c_int,
    min_keys: c_int,
    ring: c_int,
    scale: f32,
}

#[repr(C)]
struct DecodeArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    rows: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    n: c_int,
    ld: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    max_splits: c_int,
    min_keys: c_int,
    window: c_int,
    ring: c_int,
    scale: f32,
}

#[repr(C)]
struct Append16ArgsRaw {
    k: *const c_void,
    v: *const c_void,
    rows: *const c_void,
    n: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
}

const _: () = assert!(
    size_of::<Append16ArgsRaw>() == 40
        && offset_of!(Append16ArgsRaw, n) == 24
        && size_of::<ConvArgsRaw>() == 48
        && offset_of!(ConvArgsRaw, n) == 32
        && size_of::<DeltaArgsRaw>() == 96
        && offset_of!(DeltaArgsRaw, n) == 56
        && offset_of!(DeltaArgsRaw, eps) == 88
        && size_of::<PrepArgsRaw>() == 96
        && offset_of!(PrepArgsRaw, n) == 56
        && offset_of!(PrepArgsRaw, rope_base) == 88
        && size_of::<AppendArgsRaw>() == 48
        && offset_of!(AppendArgsRaw, n) == 24
        && size_of::<GemmaArgsRaw>() == 80
        && offset_of!(GemmaArgsRaw, n) == 32
        && offset_of!(GemmaArgsRaw, scale) == 72
        && offset_of!(AppendArgsRaw, ring) == 44
        && size_of::<DecodeArgsRaw>() == 88
        && offset_of!(DecodeArgsRaw, n) == 40
        && offset_of!(DecodeArgsRaw, scale) == 80
);

unsafe extern "C" {
    #[link_name = "inference_cuda_gdn_conv_multi_launch"]
    fn conv_raw(args: *const ConvArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_gdn_conv_multi_ieee_launch"]
    fn conv_ieee_raw(args: *const ConvArgsRaw, stream: *mut c_void, out_stage: *mut c_int)
    -> c_int;
    #[link_name = "inference_cuda_gdn_delta_multi_launch"]
    fn delta_raw(args: *const DeltaArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_gdn_delta_multi_ieee_launch"]
    fn delta_ieee_raw(
        args: *const DeltaArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_attn_prep_multi_launch"]
    fn prep_raw(args: *const PrepArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_kv8_append_multi_launch"]
    fn append_raw(args: *const AppendArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_attn_decode_gemma_multi_launch"]
    fn gemma_raw(args: *const GemmaArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_attn_decode_q8_multi_launch"]
    fn decode_raw(args: *const DecodeArgsRaw, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_attn_decode_multi_launch"]
    fn decode16_raw(
        args: *const DecodeArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_kv16_append_multi_launch"]
    fn append16_raw(
        args: *const Append16ArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes() >= need && range.offset().is_multiple_of(align)
}

fn int(value: usize) -> Result<c_int, DeviceFailure> {
    c_int::try_from(value).map_err(|_| invalid_argument())
}

/// The table range of `n` entries of `entry` bytes, 8-aligned.
fn table_ok(table: &RawRange<'_>, n: usize, entry: usize) -> bool {
    (1..=MULTI_ROW_MAX).contains(&n)
        && n.checked_mul(entry)
            .is_some_and(|need| fits(table, need, 8))
}

pub(in crate::kernels::cuda) struct ConvMultiRaw<'a> {
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) states: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) channels: usize,
}

pub(in crate::kernels::cuda) fn gdn_conv_multi_launch(
    stream: &RawStream,
    call: ConvMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    let float = size_of::<f32>();
    let proj_need = call
        .n
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(call.ld))
        .and_then(|floats| floats.checked_add(call.channels))
        .and_then(|floats| floats.checked_mul(float))
        .ok_or_else(invalid_argument)?;
    let out_need = call
        .n
        .checked_mul(call.channels)
        .and_then(|floats| floats.checked_mul(float))
        .ok_or_else(invalid_argument)?;
    let w_need = call
        .channels
        .checked_mul(4 * float)
        .ok_or_else(invalid_argument)?;
    let sized = table_ok(&call.states, call.n, STATE_ENTRY_BYTES)
        && fits(&call.proj, proj_need, 4)
        && fits(&call.w, w_need, 16)
        && fits(&call.out, out_need, 4)
        && call.ld >= call.channels;
    let out = Span::of(&call.out, out_need);
    let apart = out.apart(&Span::of(&call.proj, proj_need))
        && out.apart(&Span::of(&call.w, w_need))
        && out.apart(&Span::of(&call.states, call.n * STATE_ENTRY_BYTES));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    let args = ConvArgsRaw {
        proj: call.proj.address().cast_const(),
        w: call.w.address().cast_const(),
        states: call.states.address().cast_const().cast(),
        out: call.out.address(),
        n: int(call.n)?,
        ld: int(call.ld)?,
        channels: int(call.channels)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are proved inside live allocations above, sized for the call and aligned
    // (`w` to 16, the table to 8, the rest to 4), and `out` shares no byte with another range; the
    // entry repeats the alignment and shape checks. The state addresses in the table were proved
    // by the safe layer. The allocations outlive the queued work (`inference_cuda_free`
    // synchronizes first). `args` is a live local the entry reads only during the call; `stream`
    // is a live handle and `stage` a live, writable c_int.
    let status = unsafe {
        if super::exact_transcendentals() {
            conv_ieee_raw(&args, stream.as_ptr(), &mut stage)
        } else {
            conv_raw(&args, stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) struct DeltaMultiRaw<'a> {
    pub(in crate::kernels::cuda) conv: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) dt_bias: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_w: RawRange<'a>,
    pub(in crate::kernels::cuda) states: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) conv_ld: usize,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) offsets: [usize; 3],
    pub(in crate::kernels::cuda) n_k_heads: usize,
    pub(in crate::kernels::cuda) n_v_heads: usize,
    pub(in crate::kernels::cuda) eps: f32,
}

pub(in crate::kernels::cuda) fn gdn_delta_multi_launch(
    stream: &RawStream,
    call: DeltaMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    const D: usize = 128;
    let f = size_of::<f32>();
    let rows = call.n.checked_sub(1).ok_or_else(invalid_argument)?;
    let conv_cols = (2 * call.n_k_heads + call.n_v_heads) * D;
    let z_end = call.offsets[0] + call.n_v_heads * D;
    let proj_cols = z_end
        .max(call.offsets[1] + call.n_v_heads)
        .max(call.offsets[2] + call.n_v_heads);
    let conv_need = (rows * call.conv_ld + conv_cols) * f;
    let proj_need = (rows * call.ld + proj_cols) * f;
    let head_bytes = call.n_v_heads * f;
    let out_need = call.n * call.n_v_heads * D * f;
    let sized = table_ok(&call.states, call.n, STATE_ENTRY_BYTES)
        && fits(&call.conv, conv_need, 4)
        && fits(&call.proj, proj_need, 4)
        && fits(&call.a, head_bytes, 4)
        && fits(&call.dt_bias, head_bytes, 4)
        && fits(&call.norm_w, D * f, 4)
        && fits(&call.out, out_need, 4)
        && call.conv_ld >= conv_cols
        && call.ld >= proj_cols;
    let out = Span::of(&call.out, out_need);
    let apart = [
        Span::of(&call.conv, conv_need),
        Span::of(&call.proj, proj_need),
        Span::of(&call.a, head_bytes),
        Span::of(&call.dt_bias, head_bytes),
        Span::of(&call.norm_w, D * f),
        Span::of(&call.states, call.n * STATE_ENTRY_BYTES),
    ]
    .iter()
    .all(|read| out.apart(read));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    let args = DeltaArgsRaw {
        conv: call.conv.address().cast_const(),
        proj: call.proj.address().cast_const(),
        a: call.a.address().cast_const(),
        dt_bias: call.dt_bias.address().cast_const(),
        norm_w: call.norm_w.address().cast_const(),
        states: call.states.address().cast_const().cast(),
        out: call.out.address(),
        n: int(call.n)?,
        conv_ld: int(call.conv_ld)?,
        ld: int(call.ld)?,
        z_off: int(call.offsets[0])?,
        beta_off: int(call.offsets[1])?,
        alpha_off: int(call.offsets[2])?,
        n_k_heads: int(call.n_k_heads)?,
        n_v_heads: int(call.n_v_heads)?,
        eps: call.eps,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `gdn_conv_multi_launch`: every range proved live, sized and aligned above, `out`
    // apart from every other range, the entry repeating the shape checks (including `eps`), the
    // state addresses proved by the safe layer, `args` and `stage` live for the call.
    let status = unsafe {
        if super::exact_transcendentals() {
            delta_ieee_raw(&args, stream.as_ptr(), &mut stage)
        } else {
            delta_raw(&args, stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) struct PrepMultiRaw<'a> {
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) q_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_rows: RawRange<'a>,
    pub(in crate::kernels::cuda) v_rows: RawRange<'a>,
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) table: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) heads: [usize; 2],
    pub(in crate::kernels::cuda) n_rot: usize,
    pub(in crate::kernels::cuda) cells: usize,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) rope_base: f32,
}

pub(in crate::kernels::cuda) fn attn_prep_multi_launch(
    stream: &RawStream,
    call: PrepMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    const D: usize = 256;
    let f = size_of::<f32>();
    let [n_head, n_kv_head] = call.heads;
    let row_width = (n_head * 2 + n_kv_head * 2) * D;
    let rows = call.n.checked_sub(1).ok_or_else(invalid_argument)?;
    let proj_need = (rows * call.ld + row_width) * f;
    let q_need = call.n * n_head * D * f;
    let kv_need = call.n * n_kv_head * D * f;
    let sized = table_ok(&call.table, call.n, KV_ROW_BYTES)
        && fits(&call.proj, proj_need, 4)
        && fits(&call.q_norm, D * f, 4)
        && fits(&call.k_norm, D * f, 4)
        && fits(&call.q, q_need, 16)
        && fits(&call.k_rows, kv_need, 16)
        && fits(&call.v_rows, kv_need, 16)
        && call.ld >= row_width;
    let writes = [
        Span::of(&call.q, q_need),
        Span::of(&call.k_rows, kv_need),
        Span::of(&call.v_rows, kv_need),
    ];
    let reads = [
        Span::of(&call.proj, proj_need),
        Span::of(&call.q_norm, D * f),
        Span::of(&call.k_norm, D * f),
        Span::of(&call.table, call.n * KV_ROW_BYTES),
    ];
    let apart = writes.iter().enumerate().all(|(i, w)| {
        reads.iter().all(|r| w.apart(r)) && writes[i + 1..].iter().all(|other| w.apart(other))
    });
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    let args = PrepArgsRaw {
        proj: call.proj.address().cast_const(),
        q_norm: call.q_norm.address().cast_const(),
        k_norm: call.k_norm.address().cast_const(),
        k_rows: call.k_rows.address(),
        v_rows: call.v_rows.address(),
        q: call.q.address(),
        rows: call.table.address().cast_const(),
        n: int(call.n)?,
        ld: int(call.ld)?,
        n_head: int(n_head)?,
        n_kv_head: int(n_kv_head)?,
        head_dim: int(D)?,
        n_rot: int(call.n_rot)?,
        cells: int(call.cells)?,
        eps: call.eps,
        rope_base: call.rope_base,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as `gdn_conv_multi_launch`: the ranges proved live, sized and aligned (`q`, `k_rows`
    // and `v_rows` to 16, the table to 8) with the three written ranges apart from each other and
    // from every read range; the entry repeats the shape and scalar checks; the table's step and
    // plane addresses were proved by the safe layer; `args` and `stage` are live for the call.
    let status = unsafe { prep_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) struct AppendMultiRaw<'a> {
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) v: RawRange<'a>,
    pub(in crate::kernels::cuda) table: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) n_kv_head: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
    pub(in crate::kernels::cuda) cells: usize,
    pub(in crate::kernels::cuda) code_scale: u32,
    pub(in crate::kernels::cuda) ring: usize,
}

pub(in crate::kernels::cuda) fn kv8_append_multi_launch(
    stream: &RawStream,
    call: AppendMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    let need = call.n * call.n_kv_head * call.head_dim * size_of::<f32>();
    let sized = table_ok(&call.table, call.n, KV_ROW_BYTES)
        && fits(&call.k, need, 4)
        && fits(&call.v, need, 4);
    if !sized {
        return Err(invalid_argument());
    }
    let args = AppendArgsRaw {
        k: call.k.address().cast_const(),
        v: call.v.address().cast_const(),
        rows: call.table.address().cast_const(),
        n: int(call.n)?,
        n_kv_head: int(call.n_kv_head)?,
        head_dim: int(call.head_dim)?,
        cells: int(call.cells)?,
        code_scale: c_int::try_from(call.code_scale).map_err(|_| invalid_argument())?,
        ring: int(call.ring)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the `k` and `v` rows and the table are proved live, sized and aligned above (rows to
    // 4, the table to 8); the entry repeats the shape checks; the planes and step blocks the table
    // names were proved, and proved apart, by the safe layer, which is the only writer of the
    // table. `args` and `stage` are live for the call.
    let status = unsafe { append_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) struct Append16MultiRaw<'a> {
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) v: RawRange<'a>,
    pub(in crate::kernels::cuda) table: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) n_kv_head: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
    pub(in crate::kernels::cuda) cells: usize,
}

pub(in crate::kernels::cuda) fn kv16_append_multi_launch(
    stream: &RawStream,
    call: Append16MultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    let need = call.n * call.n_kv_head * call.head_dim * size_of::<f32>();
    let sized = table_ok(&call.table, call.n, KV_ROW_BYTES)
        && fits(&call.k, need, 4)
        && fits(&call.v, need, 4);
    if !sized {
        return Err(invalid_argument());
    }
    let args = Append16ArgsRaw {
        k: call.k.address().cast_const(),
        v: call.v.address().cast_const(),
        rows: call.table.address().cast_const(),
        n: int(call.n)?,
        n_kv_head: int(call.n_kv_head)?,
        head_dim: int(call.head_dim)?,
        cells: int(call.cells)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the `k` and `v` rows and the table are proved live, sized and aligned above (rows to
    // 4, the table to 8); the entry repeats the shape checks; the planes and step blocks the table
    // names were proved, and proved apart, by the safe layer, which is the only writer of the
    // table. `args` and `stage` are live for the call.
    let status = unsafe { append16_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) struct DecodeMultiRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) table: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) ld: usize,
    pub(in crate::kernels::cuda) heads: [usize; 2],
    pub(in crate::kernels::cuda) head_dim: usize,
    pub(in crate::kernels::cuda) cells: usize,
    pub(in crate::kernels::cuda) split: [usize; 2],
    pub(in crate::kernels::cuda) span: [usize; 2],
    pub(in crate::kernels::cuda) scale: f32,
}

/// The entry of one cache form: the KV8 planes (`inference_cuda_attn_decode_q8_multi_launch`) or the
/// fp16 planes (`inference_cuda_attn_decode_multi_launch`); both take the same arguments.
type DecodeEntry = unsafe extern "C" fn(*const DecodeArgsRaw, *mut c_void, *mut c_int) -> c_int;

pub(in crate::kernels::cuda) fn attn_decode_q8_multi_launch(
    stream: &RawStream,
    call: DecodeMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    decode_multi_launch(stream, call, decode_raw)
}

pub(in crate::kernels::cuda) fn attn_decode_multi_launch(
    stream: &RawStream,
    call: DecodeMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    decode_multi_launch(stream, call, decode16_raw)
}

fn decode_multi_launch(
    stream: &RawStream,
    call: DecodeMultiRaw<'_>,
    entry: DecodeEntry,
) -> Result<(), DeviceFailure> {
    let f = size_of::<f32>();
    let [n_head, n_kv_head] = call.heads;
    let q_need = call.n * n_head * call.head_dim * f;
    let out_need = q_need;
    let part_need = call
        .n
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(call.split[0]))
        .and_then(|v| v.checked_mul(8 * (2 + call.head_dim) * f))
        .ok_or_else(invalid_argument)?;
    let proj_need = call
        .n
        .checked_sub(1)
        .map(|rows| (rows * call.ld + n_head * 2 * call.head_dim) * f)
        .ok_or_else(invalid_argument)?;
    let sized = table_ok(&call.table, call.n, KV_ROW_BYTES)
        && fits(&call.q, q_need, 16)
        && fits(&call.part, part_need, 16)
        && fits(&call.out, out_need, 4)
        && call
            .proj
            .as_ref()
            .is_none_or(|proj| fits(proj, proj_need, 4));
    let out = Span::of(&call.out, out_need);
    let part = Span::of(&call.part, part_need);
    let mut reads = vec![
        Span::of(&call.q, q_need),
        Span::of(&call.table, call.n * KV_ROW_BYTES),
    ];
    if let Some(proj) = &call.proj {
        reads.push(Span::of(proj, proj_need));
    }
    let apart = out.apart(&part) && reads.iter().all(|read| out.apart(read) && part.apart(read));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    let args = DecodeArgsRaw {
        q: call.q.address().cast_const(),
        proj: call
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |proj| proj.address().cast_const()),
        rows: call.table.address().cast_const(),
        part: call.part.address(),
        out: call.out.address(),
        n: int(call.n)?,
        ld: if call.proj.is_some() {
            int(call.ld)?
        } else {
            0
        },
        n_head: int(n_head)?,
        n_kv_head: int(n_kv_head)?,
        head_dim: int(call.head_dim)?,
        cells: int(call.cells)?,
        max_splits: int(call.split[0])?,
        min_keys: int(call.split[1])?,
        window: int(call.span[0])?,
        ring: int(call.span[1])?,
        scale: call.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are proved live, sized and aligned above (`q` and `part` to 16, the table
    // to 8, the rest to 4) with `out` and `part` apart from each other and from every read range;
    // the entry repeats the shape checks of the single-row entry; the plane and step addresses in
    // the table were proved by the safe layer; `args` and `stage` are live for the call.
    let status = unsafe { entry(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) struct GemmaMultiRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) table: RawRange<'a>,
    pub(in crate::kernels::cuda) part: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) heads: [usize; 2],
    pub(in crate::kernels::cuda) head_dim: usize,
    pub(in crate::kernels::cuda) cells: usize,
    pub(in crate::kernels::cuda) window: usize,
    pub(in crate::kernels::cuda) ring: usize,
    pub(in crate::kernels::cuda) split: [usize; 2],
    pub(in crate::kernels::cuda) scale: f32,
}

pub(in crate::kernels::cuda) fn attn_decode_gemma_multi_launch(
    stream: &RawStream,
    call: GemmaMultiRaw<'_>,
) -> Result<(), DeviceFailure> {
    let f = size_of::<f32>();
    let [n_q_head, _] = call.heads;
    let q_need = call.n * n_q_head * call.head_dim * f;
    let part_need = match &call.part {
        Some(_) => call
            .n
            .checked_mul(n_q_head)
            .and_then(|v| v.checked_mul(call.split[0]))
            .and_then(|v| v.checked_mul((2 + call.head_dim) * f))
            .ok_or_else(invalid_argument)?,
        None => 0,
    };
    let sized = table_ok(&call.table, call.n, KV_ROW_BYTES)
        && fits(&call.q, q_need, 4)
        && fits(&call.out, q_need, 4)
        && call.part.as_ref().is_none_or(|p| fits(p, part_need, 16));
    let out = Span::of(&call.out, q_need);
    let mut reads = vec![
        Span::of(&call.q, q_need),
        Span::of(&call.table, call.n * KV_ROW_BYTES),
    ];
    let mut apart = true;
    if let Some(part) = &call.part {
        let span = Span::of(part, part_need);
        apart = out.apart(&span);
        reads.push(span);
    }
    apart = apart && reads.iter().all(|read| out.apart(read));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    let args = GemmaArgsRaw {
        q: call.q.address().cast_const(),
        rows: call.table.address().cast_const(),
        part: call
            .part
            .as_ref()
            .map_or(std::ptr::null_mut(), |p| p.address()),
        out: call.out.address(),
        n: int(call.n)?,
        n_q_head: int(n_q_head)?,
        n_kv_head: int(call.heads[1])?,
        head_dim: int(call.head_dim)?,
        cells: int(call.cells)?,
        window: int(call.window)?,
        split: int(usize::from(call.part.is_some()))?,
        max_splits: int(call.split[0])?,
        min_keys: int(call.split[1])?,
        ring: int(call.ring)?,
        scale: call.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are proved live, sized and aligned above (`part` to 16, the table to 8,
    // the rest to 4) with `out` apart from every other range. The entry checks the shape, the
    // alignment and the split rules (a `part` pointer, `max_splits` 1 to 128, `min_keys` at
    // least 8) of the single-row entry on the arguments, but not extents: the extents of every
    // row's planes and step block were proved, and proved apart from the outputs, by the safe layer,
    // which is the only writer of the table. `args` and `stage` are live for the call.
    let status = unsafe { gemma_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
