//! Raw FFI to the device router entry `inference_cuda_router_launch` of
//! `native/resident/router.cu` (declared in `native/resident/router.h`). Child module of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs` (decision 0047).
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Sizes, alignment and disjointness are proved again here with `super::Span` before the call,
//!   so a bad call is refused with `invalid_argument` and never reaches the device.
//! - The C struct is built in one `unsafe` block that cites those checks.
//! - Every allocation stays valid until queued work finishes, because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`; the launch is asynchronous.
//! - The extern declaration, the `repr(C)` mirror and every raw pointer stay private to this file;
//!   nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::ptr;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Field for field the `inference_cuda_router_args` of `router.h`.
#[repr(C)]
struct RouterArgsRaw {
    x: *const c_void,
    w: *const c_void,
    w_shared: *const c_void,
    logits: *mut c_void,
    ids: *mut c_void,
    weights: *mut c_void,
    shared_gate: *mut c_void,
    status: *mut c_void,
    m: c_int,
    hidden: c_int,
    n_expert: c_int,
    top_k: c_int,
}

unsafe extern "C" {
    #[link_name = "inference_cuda_router_launch"]
    fn router_launch_raw(
        args: *const RouterArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The integers of one router call, converted by the wrapper before the launch.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct RouterInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) hidden: c_int,
    pub(in crate::kernels::cuda) n_expert: c_int,
    pub(in crate::kernels::cuda) top_k: c_int,
}

/// What one router launch reads and writes. `w_shared` and `shared_gate` are both given or both
/// absent.
pub(in crate::kernels::cuda) struct RouterRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) w_shared: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) weights: RawRange<'a>,
    pub(in crate::kernels::cuda) shared_gate: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: RouterInts,
}

/// The bytes each range must hold for the ints of a call.
struct Need {
    x: usize,
    w: usize,
    w_shared: usize,
    logits: usize,
    picks: usize,
    gate: usize,
}

/// The byte needs of `ints`, or `None` when an int is not positive or a product overflows.
fn needs(ints: &RouterInts) -> Option<Need> {
    let dim = |value: c_int| usize::try_from(value).ok().filter(|v| *v > 0);
    let (m, hidden) = (dim(ints.m)?, dim(ints.hidden)?);
    let (n_expert, top_k) = (dim(ints.n_expert)?, dim(ints.top_k)?);
    let row = hidden.checked_mul(4)?;
    Some(Need {
        x: m.checked_mul(row)?,
        w: n_expert.checked_mul(row)?,
        w_shared: row,
        logits: m.checked_mul(n_expert.checked_add(1)?)?.checked_mul(4)?,
        picks: m.checked_mul(top_k)?.checked_mul(4)?,
        gate: m.checked_mul(4)?,
    })
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// True when the optional range is absent, or present and fitting.
fn fits_if_given(range: Option<&RawRange<'_>>, need: usize, align: usize) -> bool {
    range.is_none_or(|range| fits(range, need, align))
}

/// True when no range the launch writes shares a byte with another range of the call.
fn outputs_apart(io: &RouterRaw<'_>, need: &Need) -> bool {
    let mut outputs = vec![
        Span::of(&io.logits, need.logits),
        Span::of(&io.ids, need.picks),
        Span::of(&io.weights, need.picks),
        Span::of(&io.status, 4),
    ];
    if let Some(gate) = &io.shared_gate {
        outputs.push(Span::of(gate, need.gate));
    }
    let mut inputs = vec![Span::of(&io.x, need.x), Span::of(&io.w, need.w)];
    if let Some(w_shared) = &io.w_shared {
        inputs.push(Span::of(w_shared, need.w_shared));
    }
    outputs.iter().enumerate().all(|(index, out)| {
        inputs.iter().all(|input| out.apart(input))
            && outputs[index + 1..].iter().all(|other| out.apart(other))
    })
}

/// Proves a router launch before the native call: the optional pair, sizes, alignment (x, w and
/// w_shared 16; the rest 4) and that nothing the launch writes overlaps another range.
fn check_router(io: &RouterRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints).ok_or_else(invalid_argument)?;
    let paired = io.w_shared.is_some() == io.shared_gate.is_some();
    let sized = fits(&io.x, need.x, 16)
        && fits(&io.w, need.w, 16)
        && fits_if_given(io.w_shared.as_ref(), need.w_shared, 16)
        && fits(&io.logits, need.logits, 4)
        && fits(&io.ids, need.picks, 4)
        && fits(&io.weights, need.picks, 4)
        && fits_if_given(io.shared_gate.as_ref(), need.gate, 4)
        && fits(&io.status, 4, 4);
    if !(paired && sized && outputs_apart(io, &need)) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the two router kernels on `stream`: the logits against `w` (and `w_shared`), then the
/// softmax, the top-k selection and the renormalization. Returns when both are queued.
pub(in crate::kernels::cuda) fn router_launch(
    stream: &RawStream,
    io: RouterRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_router(&io)?;
    let optional = |range: &Option<RawRange<'_>>| {
        range
            .as_ref()
            .map_or(ptr::null_mut(), |range| range.address())
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_router` proved every range inside its live allocation (`RawRange::new`),
    // large enough for the call, aligned as the entry requires, and every written range disjoint
    // from the others; `w_shared` and `shared_gate` are both given or both null. The entry checks
    // the addresses again. The allocations stay valid until the queued work finishes because
    // `inference_cuda_free` synchronizes the device first, and `args` lives until the entry
    // returns (it copies the struct into the launch parameters). `stream` is live for the call.
    let status = unsafe {
        let args = RouterArgsRaw {
            x: io.x.address().cast_const(),
            w: io.w.address().cast_const(),
            w_shared: optional(&io.w_shared).cast_const(),
            logits: io.logits.address(),
            ids: io.ids.address(),
            weights: io.weights.address(),
            shared_gate: optional(&io.shared_gate),
            status: io.status.address(),
            m: io.ints.m,
            hidden: io.ints.hidden,
            n_expert: io.ints.n_expert,
            top_k: io.ints.top_k,
        };
        router_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

#[repr(C)]
struct RouterLogitsArgsRaw {
    x: *const f32,
    x_bytes: usize,
    w: *const f32,
    w_bytes: usize,
    logits: *mut f32,
    logits_bytes: usize,
    m: c_int,
    hidden: c_int,
    n_expert: c_int,
}

#[repr(C)]
struct RouterSigmoidArgsRaw {
    logits: *const f32,
    logits_bytes: usize,
    ids: *mut c_int,
    ids_bytes: usize,
    weights: *mut f32,
    weights_bytes: usize,
    status: *mut c_int,
    status_bytes: usize,
    m: c_int,
    n_expert: c_int,
    top_k: c_int,
}

const _: () = assert!(
    std::mem::size_of::<RouterLogitsArgsRaw>() == 64
        && std::mem::offset_of!(RouterLogitsArgsRaw, x) == 0
        && std::mem::offset_of!(RouterLogitsArgsRaw, x_bytes) == 8
        && std::mem::offset_of!(RouterLogitsArgsRaw, w) == 16
        && std::mem::offset_of!(RouterLogitsArgsRaw, w_bytes) == 24
        && std::mem::offset_of!(RouterLogitsArgsRaw, logits) == 32
        && std::mem::offset_of!(RouterLogitsArgsRaw, logits_bytes) == 40
        && std::mem::offset_of!(RouterLogitsArgsRaw, m) == 48
        && std::mem::offset_of!(RouterLogitsArgsRaw, hidden) == 52
        && std::mem::offset_of!(RouterLogitsArgsRaw, n_expert) == 56
);
const _: () = assert!(
    std::mem::size_of::<RouterSigmoidArgsRaw>() == 80
        && std::mem::offset_of!(RouterSigmoidArgsRaw, logits) == 0
        && std::mem::offset_of!(RouterSigmoidArgsRaw, logits_bytes) == 8
        && std::mem::offset_of!(RouterSigmoidArgsRaw, ids) == 16
        && std::mem::offset_of!(RouterSigmoidArgsRaw, ids_bytes) == 24
        && std::mem::offset_of!(RouterSigmoidArgsRaw, weights) == 32
        && std::mem::offset_of!(RouterSigmoidArgsRaw, weights_bytes) == 40
        && std::mem::offset_of!(RouterSigmoidArgsRaw, status) == 48
        && std::mem::offset_of!(RouterSigmoidArgsRaw, status_bytes) == 56
        && std::mem::offset_of!(RouterSigmoidArgsRaw, m) == 64
        && std::mem::offset_of!(RouterSigmoidArgsRaw, n_expert) == 68
        && std::mem::offset_of!(RouterSigmoidArgsRaw, top_k) == 72
);

unsafe extern "C" {
    #[link_name = "inference_cuda_router_logits_launch"]
    fn router_logits_launch_raw(
        args: *const RouterLogitsArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_router_sigmoid_select_launch"]
    fn router_sigmoid_launch_raw(
        args: *const RouterSigmoidArgsRaw,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct RouterLogitsRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: RouterInts,
}

pub(in crate::kernels::cuda) struct RouterSigmoidRaw<'a> {
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) weights: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: RouterInts,
}

fn split_needs(ints: &RouterInts) -> Option<Need> {
    if !(1..=4096).contains(&ints.m)
        || !(4..=65536).contains(&ints.hidden)
        || ints.hidden % 4 != 0
        || !(1..=256).contains(&ints.n_expert)
        || !(1..=ints.n_expert.min(32)).contains(&ints.top_k)
    {
        return None;
    }
    let mut need = needs(ints)?;
    need.logits = need
        .logits
        .checked_sub(usize::try_from(ints.m).ok()?.checked_mul(4)?)?;
    Some(need)
}

fn check_split_ranges(
    regions: &[(&RawRange<'_>, usize, usize, bool)],
) -> Result<(), DeviceFailure> {
    for (index, &(range, bytes, align, written)) in regions.iter().enumerate() {
        if !fits(range, bytes, align) {
            return Err(invalid_argument());
        }
        for &(other, other_bytes, _, other_written) in &regions[..index] {
            if (written || other_written)
                && !Span::of(range, bytes).apart(&Span::of(other, other_bytes))
            {
                return Err(invalid_argument());
            }
        }
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn router_logits_launch(
    stream: &RawStream,
    io: RouterLogitsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let need = split_needs(&io.ints).ok_or_else(invalid_argument)?;
    check_split_ranges(&[
        (&io.x, need.x, 16, false),
        (&io.w, need.w, 16, false),
        (&io.logits, need.logits, 4, true),
    ])?;
    let args = RouterLogitsArgsRaw {
        x: io.x.address().cast::<f32>().cast_const(),
        x_bytes: io.x.bytes,
        w: io.w.address().cast::<f32>().cast_const(),
        w_bytes: io.w.bytes,
        logits: io.logits.address().cast(),
        logits_bytes: io.logits.bytes,
        m: io.ints.m,
        hidden: io.ints.hidden,
        n_expert: io.ints.n_expert,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live owned extents, alignment and disjoint writes were proved above.
    // launch_on leases this stream; allocation drops drain queued use before free.
    let status = unsafe { router_logits_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn router_sigmoid_launch(
    stream: &RawStream,
    io: RouterSigmoidRaw<'_>,
) -> Result<(), DeviceFailure> {
    let need = split_needs(&io.ints).ok_or_else(invalid_argument)?;
    check_split_ranges(&[
        (&io.logits, need.logits, 4, false),
        (&io.ids, need.picks, 4, true),
        (&io.weights, need.picks, 4, true),
        (&io.status, 4, 4, true),
    ])?;
    let args = RouterSigmoidArgsRaw {
        logits: io.logits.address().cast::<f32>().cast_const(),
        logits_bytes: io.logits.bytes,
        ids: io.ids.address().cast(),
        ids_bytes: io.ids.bytes,
        weights: io.weights.address().cast(),
        weights_bytes: io.weights.bytes,
        status: io.status.address().cast(),
        status_bytes: io.status.bytes,
        m: io.ints.m,
        n_expert: io.ints.n_expert,
        top_k: io.ints.top_k,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: checked live ranges have disjoint writes; C repeats shape/span checks.
    // Stream lease and synchronizing allocation drops keep all queued accesses live.
    let status = unsafe { router_sigmoid_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod split_tests;
