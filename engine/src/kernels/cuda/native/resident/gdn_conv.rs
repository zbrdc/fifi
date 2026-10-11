//! Raw FFI to `inference_cuda_gdn_conv_launch` of `native/resident/gdn_conv.cu` (declared in
//! `native/resident/gdn_conv.h`): the launch shim of the vendored `conv_silu_k` (RD-7a, decision
//! 0047, stage 1). Child of `native::resident`, so it inherits the `allow(unsafe_code)` of
//! `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Design guidance only, no code copied: nothing here comes from MegaCapybara (no
//! MegaCapybara-derived design); the kernel is the user's own xavier-qwen code restored in RD-4V.
//!
//! Safety invariants this file relies on:
//! - The entry is called only through `CudaContext::launch_on`, on the owning thread and under the
//!   dispatch guard (decisions 0023 and 0024). It only enqueues and returns the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, before the call, on the
//!   extents the kernel touches; the C entry repeats the checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The C argument struct is a live local that the entry reads only during the call (the kernel
//!   arguments are copied at launch, so nothing keeps the pointer). Its `repr(C)` mirror is private
//!   here and pinned by a `const` assert to the field order of `gdn_conv.h`. No raw pointer and no
//!   unsafe function leaves this module.
//! - The conv state is planar `[3][channels]` on the device; ggml keeps `[channel][tap]`. Import
//!   and export of a state (owner RD-9a2) must transpose it.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Largest `m` of one launch (`gdn_conv.h`).
pub(in crate::kernels::cuda) const GDN_CONV_MAX_M: usize = 32768;
/// Largest `channels` of one launch.
pub(in crate::kernels::cuda) const GDN_CONV_MAX_CHANNELS: usize = 65536;
/// Largest `ld` (the row stride of `proj`, in floats).
pub(in crate::kernels::cuda) const GDN_CONV_MAX_LD: usize = 1 << 20;

/// Alignment of `proj`, `state` and `out`: one f32.
const WORD_ALIGN: usize = 4;
/// Alignment of `w`: the kernel loads one float4 per channel.
const TAPS_ALIGN: usize = 16;

/// Mirror of `inference_cuda_gdn_conv_args`.
#[repr(C)]
struct GdnConvArgsRaw {
    proj: *const c_void,
    w: *const c_void,
    state: *mut c_void,
    out: *mut c_void,
    m: c_int,
    ld: c_int,
    channels: c_int,
}

const _: () = assert!(
    size_of::<GdnConvArgsRaw>() == 48
        && offset_of!(GdnConvArgsRaw, proj) == 0
        && offset_of!(GdnConvArgsRaw, w) == 8
        && offset_of!(GdnConvArgsRaw, state) == 16
        && offset_of!(GdnConvArgsRaw, out) == 24
        && offset_of!(GdnConvArgsRaw, m) == 32
        && offset_of!(GdnConvArgsRaw, ld) == 36
        && offset_of!(GdnConvArgsRaw, channels) == 40
);

unsafe extern "C" {
    #[link_name = "inference_cuda_gdn_conv_launch"]
    fn gdn_conv_raw(
        args: *const GdnConvArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one conv launch reads and writes: `proj` holds `m` rows of `ld` floats (the first
/// `channels` of each are read), `w` the `[channels][4]` taps, `state` the planar `[3][channels]`
/// window (read and written) and `out` receives `m` rows of `channels` values.
pub(in crate::kernels::cuda) struct GdnConvRaw<'a> {
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) state: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) channels: usize,
    pub(in crate::kernels::cuda) ld: usize,
}

/// Bytes each range of a call must hold.
struct Sizes {
    proj: usize,
    w: usize,
    state: usize,
    out: usize,
}

/// The shape as C ints, `None` outside the limits of `gdn_conv.h`.
fn call_ints(call: &GdnConvRaw<'_>) -> Option<(c_int, c_int, c_int)> {
    let shape_ok = (1..=GDN_CONV_MAX_M).contains(&call.m)
        && (1..=GDN_CONV_MAX_CHANNELS).contains(&call.channels)
        && (call.channels..=GDN_CONV_MAX_LD).contains(&call.ld);
    if !shape_ok {
        return None;
    }
    Some((
        c_int::try_from(call.m).ok()?,
        c_int::try_from(call.ld).ok()?,
        c_int::try_from(call.channels).ok()?,
    ))
}

fn call_sizes(call: &GdnConvRaw<'_>) -> Option<Sizes> {
    let float = size_of::<f32>();
    let reach = (call.m - 1)
        .checked_mul(call.ld)?
        .checked_add(call.channels)?;
    Some(Sizes {
        proj: reach.checked_mul(float)?,
        w: call.channels.checked_mul(4 * float)?,
        state: call.channels.checked_mul(3 * float)?,
        out: call.m.checked_mul(call.channels)?.checked_mul(float)?,
    })
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// Proves a conv launch before the native call: limits, sizes, alignment (`w` 16, the rest 4) and
/// that `state` and `out`, which the kernel writes, share no byte with any other range. `proj`
/// and `w` are read only and may overlap each other.
fn check_gdn_conv(call: &GdnConvRaw<'_>) -> Result<(c_int, c_int, c_int), DeviceFailure> {
    let ints = call_ints(call).ok_or_else(invalid_argument)?;
    let need = call_sizes(call).ok_or_else(invalid_argument)?;
    let sized = fits(&call.proj, need.proj, WORD_ALIGN)
        && fits(&call.w, need.w, TAPS_ALIGN)
        && fits(&call.state, need.state, WORD_ALIGN)
        && fits(&call.out, need.out, WORD_ALIGN);
    let state = Span::of(&call.state, need.state);
    let out = Span::of(&call.out, need.out);
    let apart = state.apart(&out)
        && state.apart(&Span::of(&call.proj, need.proj))
        && state.apart(&Span::of(&call.w, need.w))
        && out.apart(&Span::of(&call.proj, need.proj))
        && out.apart(&Span::of(&call.w, need.w));
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    Ok(ints)
}

/// Queues the conv step of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn gdn_conv_launch(
    stream: &RawStream,
    call: GdnConvRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (m, ld, channels) = check_gdn_conv(&call)?;
    let args = GdnConvArgsRaw {
        proj: call.proj.address().cast_const(),
        w: call.w.address().cast_const(),
        state: call.state.address(),
        out: call.out.address(),
        m,
        ld,
        channels,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_gdn_conv` proved the four ranges inside their live allocations, large enough
    // for the call (`proj` for `(m - 1) * ld + channels` floats), aligned (`w` to 16, the rest to
    // 4), and that `state` and `out`, which the kernel writes, share no byte with another range;
    // the entry repeats these checks on the addresses. The allocations outlive the queued work
    // (`inference_cuda_free` synchronizes first). `args` is a live local that the entry reads
    // only during the call: the kernel arguments are copied at launch, so nothing keeps the
    // pointer. `stream` is a live handle for the call, and `stage` is a live, writable c_int.
    let status = unsafe { gdn_conv_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
