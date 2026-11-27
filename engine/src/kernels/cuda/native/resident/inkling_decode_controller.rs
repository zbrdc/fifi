//! Raw FFI to `inference_cuda_decode_controller_gate_down` and
//! `inference_cuda_decode_controller_pair_norm` of
//! `native/resident/inkling_decode_controller.cu` (declared in
//! `native/resident/inkling_decode_controller.h`). Child module of `native::resident`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on:
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Shapes, sizes, alignment and disjointness are proved again here (`super::Span`), so a bad
//!   call is refused with `invalid_argument` and never reaches the entry; the entry repeats the
//!   checks on the addresses.
//! - The entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard; it queues one launch and never synchronizes.
//! - Every allocation stays valid until queued work finishes, because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The extern declarations and every raw pointer stay inside this module; no unsafe function
//!   leaves it.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

pub(crate) const CONTROLLER_MAX_ROLES: usize = 4;
const MAX_WIDTH: usize = 4096;
/// The rank of the two projections of the pair launch: sixteen lanes of four samples a row.
pub(crate) const PAIR_RANK: usize = 64;
const WORD: usize = size_of::<f32>();

#[repr(C)]
#[derive(Clone, Copy)]
struct GateRoleRaw {
    input: *const c_void,
    message: *const c_void,
    gate: *mut c_void,
    down: *const c_void,
    reduced: *mut c_void,
    health: [*mut c_void; 2],
}

#[repr(C)]
struct GateArgsRaw {
    role: [GateRoleRaw; CONTROLLER_MAX_ROLES],
    tickets: *mut c_void,
    roles: c_int,
    width: c_int,
    rank: c_int,
    gate_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ApplyRoleRaw {
    input: *const c_void,
    message: *const c_void,
    gamma: *const c_void,
    reduced: *const c_void,
    gain_w: *const c_void,
    forget_w: *const c_void,
    gain: *mut c_void,
    forget: *mut c_void,
    out: *mut c_void,
    health: *mut c_void,
}

#[repr(C)]
struct ApplyArgsRaw {
    role: [ApplyRoleRaw; CONTROLLER_MAX_ROLES],
    tickets: *mut c_void,
    roles: c_int,
    rank: c_int,
    width: c_int,
    eps: f32,
}

const _: () = {
    assert!(
        offset_of!(GateArgsRaw, tickets) == CONTROLLER_MAX_ROLES * 7 * size_of::<*mut c_void>()
    );
    assert!(
        offset_of!(GateArgsRaw, roles)
            == offset_of!(GateArgsRaw, tickets) + size_of::<*mut c_void>()
    );
    assert!(
        offset_of!(ApplyArgsRaw, tickets) == CONTROLLER_MAX_ROLES * 10 * size_of::<*mut c_void>()
    );
    assert!(
        offset_of!(ApplyArgsRaw, roles)
            == offset_of!(ApplyArgsRaw, tickets) + size_of::<*mut c_void>()
    );
    assert!(size_of::<GateArgsRaw>() == offset_of!(GateArgsRaw, gate_eps) + 4);
    assert!(size_of::<ApplyArgsRaw>() == offset_of!(ApplyArgsRaw, eps) + 4);
};

unsafe extern "C" {
    fn inference_cuda_decode_controller_gate_down(
        args: *const GateArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_decode_controller_pair_norm(
        args: *const ApplyArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The gate launch of one role: the rows `inference_cuda_controller_gate_role` names, in its
/// order; `health` holds the input half's word, then the message half's (one word may be both).
pub(in crate::kernels::cuda) struct GateRole<'a> {
    pub(in crate::kernels::cuda) input: RawRange<'a>,
    pub(in crate::kernels::cuda) message: RawRange<'a>,
    pub(in crate::kernels::cuda) gate: RawRange<'a>,
    pub(in crate::kernels::cuda) down: RawRange<'a>,
    pub(in crate::kernels::cuda) reduced: RawRange<'a>,
    pub(in crate::kernels::cuda) health: [RawRange<'a>; 2],
}

/// The pair launch of one role: the rows `inference_cuda_controller_apply_role` names.
pub(in crate::kernels::cuda) struct ApplyRole<'a> {
    pub(in crate::kernels::cuda) input: RawRange<'a>,
    pub(in crate::kernels::cuda) message: RawRange<'a>,
    pub(in crate::kernels::cuda) gamma: RawRange<'a>,
    pub(in crate::kernels::cuda) reduced: RawRange<'a>,
    pub(in crate::kernels::cuda) gain_w: RawRange<'a>,
    pub(in crate::kernels::cuda) forget_w: RawRange<'a>,
    pub(in crate::kernels::cuda) gain: RawRange<'a>,
    pub(in crate::kernels::cuda) forget: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) health: RawRange<'a>,
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

fn finite_positive(eps: f32) -> bool {
    eps.is_finite() && eps > 0.0
}

fn roles_fit(roles: usize, width: usize, tickets: &RawRange<'_>, eps: f32) -> bool {
    (1..=CONTROLLER_MAX_ROLES).contains(&roles)
        && (16..=MAX_WIDTH).contains(&width)
        && width.is_multiple_of(16)
        && fits(tickets, roles * 4, 4)
        && finite_positive(eps)
}

/// Every written span shares no byte with `tickets`, with any read span, with any span of
/// `others` and with any other written span, except the pairs `allow_equal` names.
fn writes_apart(
    writes: &[Span<'_>],
    reads: &[Span<'_>],
    tickets: &Span<'_>,
    others: &[Span<'_>],
    allow_equal: impl Fn(usize, usize) -> bool,
) -> bool {
    writes.iter().enumerate().all(|(i, w)| {
        w.apart(tickets)
            && reads.iter().all(|r| w.apart(r))
            && others.iter().all(|o| w.apart(o))
            && writes
                .iter()
                .enumerate()
                .all(|(j, v)| j >= i || allow_equal(i, j) || w.apart(v))
    })
}

/// The extents of one gate role: its reads, its writes and every span a later role must miss.
fn gate_spans<'a>(role: &GateRole<'a>, [width, rank]: [usize; 2]) -> Option<GateSpans<'a>> {
    let GateRole {
        input,
        message,
        gate,
        down,
        reduced,
        health,
    } = role;
    let [row, gate_bytes, reduced_bytes] = [width * WORD, 2 * width * WORD, rank * WORD];
    let fits_all = fits(input, row, 4)
        && fits(message, row, 4)
        && fits(gate, gate_bytes, 16)
        && fits(down, gate_bytes * rank, 16)
        && fits(reduced, reduced_bytes, 4)
        && health.iter().all(|h| fits(h, WORD, 4));
    if !fits_all {
        return None;
    }
    Some(GateSpans {
        reads: [
            Span::of(input, row),
            Span::of(message, row),
            Span::of(down, gate_bytes * rank),
        ],
        writes: [
            Span::of(gate, gate_bytes),
            Span::of(reduced, reduced_bytes),
            Span::of(&health[0], WORD),
            Span::of(&health[1], WORD),
        ],
        all: [
            Span::of(input, row),
            Span::of(message, row),
            Span::of(gate, gate_bytes),
            Span::of(reduced, reduced_bytes),
            Span::of(&health[0], WORD),
            Span::of(&health[1], WORD),
        ],
    })
}

struct GateSpans<'a> {
    reads: [Span<'a>; 3],
    writes: [Span<'a>; 4],
    all: [Span<'a>; 6],
}

/// Gate norms and down reduction of `roles` roles of `width` floats and `rank` rows: `gate =
/// [rms(input) | rms(message)]` with `gate_eps`, `reduced = down * gate`; the tickets of the
/// pair launch are cleared. One u32 ticket per role in `tickets`.
pub(in crate::kernels::cuda) fn gate_down(
    stream: &RawStream,
    roles: &[GateRole<'_>],
    tickets: RawRange<'_>,
    [width, rank]: [usize; 2],
    gate_eps: f32,
) -> Result<(), DeviceFailure> {
    if !roles_fit(roles.len(), width, &tickets, gate_eps)
        || !(8..=1024).contains(&rank)
        || !rank.is_multiple_of(8)
    {
        return Err(invalid_argument());
    }
    let ticket_span = Span::of(&tickets, roles.len() * 4);
    let mut others: Vec<Span<'_>> = Vec::new();
    for role in roles {
        let Some(spans) = gate_spans(role, [width, rank]) else {
            return Err(invalid_argument());
        };
        // The two health words may be one word: each writer stores the same quiet NaN.
        if !writes_apart(
            &spans.writes,
            &spans.reads,
            &ticket_span,
            &others,
            |i, j| i == 3 && j == 2,
        ) {
            return Err(invalid_argument());
        }
        others.extend(spans.all);
    }
    let mut args = GateArgsRaw {
        role: [GateRoleRaw::EMPTY; CONTROLLER_MAX_ROLES],
        tickets: tickets.address(),
        roles: roles.len() as c_int,
        width: width as c_int,
        rank: rank as c_int,
        gate_eps,
    };
    for (slot, role) in args.role.iter_mut().zip(roles) {
        *slot = GateRoleRaw {
            input: role.input.address(),
            message: role.message.address(),
            gate: role.gate.address(),
            down: role.down.address(),
            reduced: role.reduced.address(),
            health: [role.health[0].address(), role.health[1].address()],
        };
    }
    let mut stage = STAGE_NONE;
    // SAFETY: every range is aligned, holds its extent and shares no byte it may not; the entry
    // repeats the checks on the addresses. The argument struct lives through the call and every
    // buffer outlives the queued work.
    let status =
        unsafe { inference_cuda_decode_controller_gate_down(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

impl GateRoleRaw {
    const EMPTY: Self = Self {
        input: std::ptr::null(),
        message: std::ptr::null(),
        gate: std::ptr::null_mut(),
        down: std::ptr::null(),
        reduced: std::ptr::null_mut(),
        health: [std::ptr::null_mut(); 2],
    };
}

/// The extents of one apply role: its reads, its writes and every span a later role must miss.
fn apply_spans<'a>(role: &ApplyRole<'a>, row: usize) -> Option<ApplySpans<'a>> {
    let ApplyRole {
        input,
        message,
        gamma,
        reduced,
        gain_w,
        forget_w,
        gain,
        forget,
        out,
        health,
    } = role;
    let [reduced_bytes, projection] = [PAIR_RANK * WORD, row * PAIR_RANK];
    let fits_all = fits(input, row, 4)
        && fits(message, row, 4)
        && fits(gamma, row, 4)
        && fits(reduced, reduced_bytes, 4)
        && fits(gain_w, projection, 16)
        && fits(forget_w, projection, 16)
        && fits(gain, row, 4)
        && fits(forget, row, 4)
        && fits(out, row, 4)
        && fits(health, WORD, 4);
    if !fits_all {
        return None;
    }
    Some(ApplySpans {
        reads: [
            Span::of(input, row),
            Span::of(message, row),
            Span::of(gamma, row),
            Span::of(reduced, reduced_bytes),
            Span::of(gain_w, projection),
            Span::of(forget_w, projection),
        ],
        writes: [
            Span::of(gain, row),
            Span::of(forget, row),
            Span::of(out, row),
            Span::of(health, WORD),
        ],
        all: [
            Span::of(input, row),
            Span::of(message, row),
            Span::of(gamma, row),
            Span::of(reduced, reduced_bytes),
            Span::of(gain, row),
            Span::of(forget, row),
            Span::of(out, row),
            Span::of(health, WORD),
        ],
    })
}

struct ApplySpans<'a> {
    reads: [Span<'a>; 6],
    writes: [Span<'a>; 4],
    all: [Span<'a>; 8],
}

/// Both projections and the controller norm of `roles` roles: `gain = gain_w * reduced`,
/// `forget = forget_w * reduced` (`rank` is [`PAIR_RANK`]), then `out = rmsnorm((input - tanh(
/// forget) * input) + tanh(gain) * message) * gamma` with `eps` in the last block of each role.
pub(in crate::kernels::cuda) fn pair_norm(
    stream: &RawStream,
    roles: &[ApplyRole<'_>],
    tickets: RawRange<'_>,
    width: usize,
    eps: f32,
) -> Result<(), DeviceFailure> {
    if !roles_fit(roles.len(), width, &tickets, eps) {
        return Err(invalid_argument());
    }
    let row = width * WORD;
    let ticket_span = Span::of(&tickets, roles.len() * 4);
    let mut others: Vec<Span<'_>> = Vec::new();
    for role in roles {
        let Some(spans) = apply_spans(role, row) else {
            return Err(invalid_argument());
        };
        if !writes_apart(
            &spans.writes,
            &spans.reads,
            &ticket_span,
            &others,
            |_, _| false,
        ) {
            return Err(invalid_argument());
        }
        others.extend(spans.all);
    }
    let mut args = ApplyArgsRaw {
        role: [ApplyRoleRaw::EMPTY; CONTROLLER_MAX_ROLES],
        tickets: tickets.address(),
        roles: roles.len() as c_int,
        rank: PAIR_RANK as c_int,
        width: width as c_int,
        eps,
    };
    for (slot, role) in args.role.iter_mut().zip(roles) {
        *slot = ApplyRoleRaw {
            input: role.input.address(),
            message: role.message.address(),
            gamma: role.gamma.address(),
            reduced: role.reduced.address(),
            gain_w: role.gain_w.address(),
            forget_w: role.forget_w.address(),
            gain: role.gain.address(),
            forget: role.forget.address(),
            out: role.out.address(),
            health: role.health.address(),
        };
    }
    let mut stage = STAGE_NONE;
    // SAFETY: as `gate_down`; the entry repeats the checks on the addresses.
    let status =
        unsafe { inference_cuda_decode_controller_pair_norm(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

impl ApplyRoleRaw {
    const EMPTY: Self = Self {
        input: std::ptr::null(),
        message: std::ptr::null(),
        gamma: std::ptr::null(),
        reduced: std::ptr::null(),
        gain_w: std::ptr::null(),
        forget_w: std::ptr::null(),
        gain: std::ptr::null_mut(),
        forget: std::ptr::null_mut(),
        out: std::ptr::null_mut(),
        health: std::ptr::null_mut(),
    };
}
