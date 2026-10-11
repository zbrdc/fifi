//! Safe launch wrapper of the attention preparation of one full-attention layer (RD-8a, decision
//! 0047, seam 1): per-head RMSNorm of q and k, NEOX rope over the first `n_rot` dimensions at the
//! position `*pos + t` read from the device, the f32 `q` output, and the append of the normed and
//! rotated k and the unchanged v to the fp16 KV cache at the same rows.
//!
//! Design guidance only, no code copied (idea from MegaCapybara v1.35 static RE, section 6.1 C10,
//! layout only).
//!
//! [`CudaContext::attn_prep`] queues one kernel on a [`DeviceStream`] and returns. The call
//! allocates nothing, copies nothing and never synchronizes; a fault in the queued work surfaces
//! at the next synchronize of the stream and latches the backend there.
//!
//! The device `pos` word (`StepParams.pos`) is authoritative. The kernel reads only that word:
//! token `t` goes to row `*pos + t` of both caches. The host value `admit_pos` is advisory: it is
//! used for the admission check [`ensure_rows`] and for nothing else, and it never reaches the
//! kernel. RD-9d owns the host mirror of the word and runs the admission check per replay window;
//! in a captured graph `admit_pos` is checked once, at capture. If the host value and the word
//! disagree in range, the kernel writes the row the word names. A word outside
//! `0..=rows - m` is refused on the device: no KV row is written, every `q` element is NaN and
//! mask 4 is OR-ed into the status word (`StepParams.status`), so the host may see status bits 1,
//! 2 and 4 in any combination (precedence 4, 1, 2).
//!
//! Layouts: `proj` is f32 `[m][ld]`; per q head `[q 256 | gate 256]`, then `n_kv_head` k heads of
//! 256, then `n_kv_head` v heads of 256. The gate half is not read here: the decode attention
//! reads head `h`'s gate from the same row at column `h * 512 + 256 + d`. `q` is f32
//! `[m][n_head][256]`, normed times weight, then rotated, not scaled. The caches are fp16
//! `[rows][n_kv_head][256]`, rounded to nearest even; fp16 is the only path of stage 1.
//!
//! Only head size 256 with an output gate and a linear cache (`SlotMode::Position`) run. Mellum
//! (head 128, no gate, a ring on 21 layers) needs MEL-K1, a new kernel behind this API, and is
//! refused as `BadRequest` before any native call.
//!
//! Every argument is checked here, before any native call: the rules mirror `valid_attn_prep` of
//! `attn_prep.cu`, so the native entry cannot return `INVALID_ARGUMENT` (which would latch the
//! backend). The ranges are converted with `raw_range` first, then the shape rules (tag
//! `attn_prep: <rule>`), then the admission check, then `require_fit` and `require_apart`; on a
//! latched context a call with a bad range or argument still answers `BadRequest`, and a call
//! with good arguments answers the latch error (`launch_on` checks the terminal flag itself).
//!
//! RoPE accuracy: the kernel computes the angle in f32. The tests hold it to the class RP1 (lane
//! M REPLY-53): f32 RoPE angle; not checked against an f64 reference above position 1,150; A5
//! covers positions up to about 1,150 only.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::attn_prep::{AttnInts, AttnPrepRaw, attn_prep_launch};
use crate::kernels::cuda::resident::step_params::ensure_rows;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The only head size of the vendored kernel.
pub(crate) const HEAD_DIM: usize = 256;
/// The most tokens one call takes (the grid's y extent).
pub(crate) const MAX_TOKENS: usize = 65_535;
/// The most query heads.
pub(crate) const MAX_N_HEAD: usize = 256;
/// The most kv heads.
pub(crate) const MAX_N_KV_HEAD: usize = 64;
/// The most cache rows: `2^30 - 1`.
pub(crate) const MAX_ROWS: usize = (1 << 30) - 1;
/// Bytes of one norm weight vector: 256 f32.
const NORM_BYTES: usize = HEAD_DIM * 4;

/// Where row `*pos + t` of the cache lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotMode {
    /// The row is the position (Qwen3.6, a linear cache).
    Position,
    /// The row is the position modulo the ring length (Mellum, MEL-K1). Refused today.
    Ring(u32),
}

/// Geometry of one call: `m` tokens, `n_head` query heads over `n_kv_head` kv heads, the head
/// size, the rotated dimensions `n_rot` (a power of two in `2..=256`), whether the projection
/// row carries an output gate, the slot mode, the cache length `rows` and the row length `ld` of
/// the projection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnPrepShape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) n_rot: usize,
    pub(crate) has_gate: bool,
    pub(crate) slot: SlotMode,
    pub(crate) rows: usize,
    pub(crate) ld: usize,
}

/// The scalars of the norm and the rope: `eps` finite and above 0 (GGUF
/// `layer_norm_rms_epsilon`), `rope_base` finite and above 1 (GGUF `rope.freq_base`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RopeParams {
    pub(crate) eps: f32,
    pub(crate) rope_base: f32,
}

/// One attention preparation call. All ranges are device ranges of this context.
///
/// - `admit_pos`: the host's advisory position of token 0, used only for the admission check
///   `ensure_rows(admit_pos, m, rows)`; the kernel never sees it.
/// - `proj`: f32 `[m][ld]`, 4-byte aligned, see the module doc for the row layout.
/// - `q_norm`, `k_norm`: f32 256 each, 4-byte aligned; they may alias each other.
/// - `k_cache`, `v_cache`: fp16 `[rows][n_kv_head][256]`, 16-byte aligned; row `*pos + t` is
///   written.
/// - `q`: f32 `[m][n_head][256]`, 16-byte aligned; NaN on a refused launch.
/// - `pos`: one device `i32` (`StepParams.pos`), 4-byte aligned; read only.
/// - `status`: one device `i32` (`StepParams.status`), 4-byte aligned; receives mask 4 on a
///   refused launch and is untouched by a valid one.
pub(crate) struct AttnPrepCall<'b> {
    pub(crate) shape: AttnPrepShape,
    pub(crate) rope: RopeParams,
    pub(crate) admit_pos: usize,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) q_norm: DeviceRange<'b>,
    pub(crate) k_norm: DeviceRange<'b>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) pos: DeviceRange<'b>,
    pub(crate) status: DeviceRange<'b>,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("attn_prep: {tag}: {detail}"))
}

/// A `BadRequest` for a configuration that needs the Mellum kernel.
fn needs_mel_k1(tag: &str, detail: &str) -> EngError {
    rule(
        tag,
        &format!("{detail}; this configuration needs the Mellum kernel MEL-K1"),
    )
}

/// `value` as `c_int` for the rule `tag`, or the "exceeds i32" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// `value` as `c_int` when it lies in `low..=high`, else the refusal for `tag`.
fn ranged(tag: &str, value: usize, low: usize, high: usize) -> Result<c_int, EngError> {
    if !(low..=high).contains(&value) {
        return Err(rule(tag, &format!("{tag} is {value}, not {low} to {high}")));
    }
    to_int(tag, value)
}

/// Floats in one projection row: per q head `[q 256 | gate 256]`, then the k heads, then the v
/// heads. Saturates instead of overflowing, so it is total.
pub(crate) fn row_width(n_head: usize, n_kv_head: usize) -> usize {
    n_head
        .saturating_mul(2 * HEAD_DIM)
        .saturating_add(n_kv_head.saturating_mul(2 * HEAD_DIM))
}

/// Bytes `proj` must hold: `m` rows of stride `ld`, the last one `row_w` long.
pub(crate) fn proj_bytes(m: usize, ld: usize, row_w: usize) -> usize {
    m.saturating_sub(1)
        .saturating_mul(ld)
        .saturating_add(row_w)
        .saturating_mul(4)
}

/// Bytes of each cache: `rows` rows of `n_kv_head` fp16 heads of 256.
pub(crate) fn cache_bytes(rows: usize, n_kv_head: usize) -> usize {
    rows.saturating_mul(n_kv_head)
        .saturating_mul(HEAD_DIM)
        .saturating_mul(2)
}

/// Bytes of `q`: `m` rows of `n_head` f32 heads of 256.
pub(crate) fn q_bytes(m: usize, n_head: usize) -> usize {
    m.saturating_mul(n_head)
        .saturating_mul(HEAD_DIM)
        .saturating_mul(4)
}

/// Checks the configuration rules (head size, gate, slot mode).
fn check_configuration(shape: &AttnPrepShape) -> Result<(), EngError> {
    if shape.head_dim != HEAD_DIM {
        return Err(needs_mel_k1(
            "head_dim",
            &format!("head_dim is {}, only 256 runs", shape.head_dim),
        ));
    }
    if !shape.has_gate {
        return Err(needs_mel_k1(
            "gate",
            "only a projection with an output gate runs",
        ));
    }
    if let SlotMode::Ring(ring) = shape.slot {
        return Err(needs_mel_k1(
            "slot_ring",
            &format!("a ring of {ring} rows is not supported, only SlotMode::Position runs"),
        ));
    }
    Ok(())
}

/// Checks every shape rule and converts the integers once, so that no conversion can fail inside
/// the launch closure. Each refusal carries the tag `attn_prep: <rule>`.
pub(in crate::kernels::cuda) fn check_shape(shape: &AttnPrepShape) -> Result<AttnInts, EngError> {
    check_configuration(shape)?;
    let n_head = ranged("n_head", shape.n_head, 1, MAX_N_HEAD)?;
    let n_kv_head = ranged("n_kv_head", shape.n_kv_head, 1, MAX_N_KV_HEAD)?;
    let m = ranged("m", shape.m, 1, MAX_TOKENS)?;
    let n_rot = ranged("n_rot", shape.n_rot, 2, HEAD_DIM)?;
    if !shape.n_rot.is_power_of_two() {
        return Err(rule(
            "n_rot",
            &format!("n_rot is {}, not a power of two", shape.n_rot),
        ));
    }
    let rows = ranged("rows", shape.rows, shape.m, MAX_ROWS)?;
    let row_w = row_width(shape.n_head, shape.n_kv_head);
    if shape.ld < row_w {
        return Err(rule(
            "ld",
            &format!("ld is {}, below the row width {row_w}", shape.ld),
        ));
    }
    let ld = to_int("ld", shape.ld)?;
    Ok(AttnInts {
        m,
        n_head,
        n_kv_head,
        n_rot,
        rows,
        ld,
    })
}

/// Checks `eps` (finite, above 0) and `rope_base` (finite, above 1).
pub(crate) fn check_rope(rope: &RopeParams) -> Result<(), EngError> {
    if !(rope.eps.is_finite() && rope.eps > 0.0) {
        return Err(rule("eps", "eps is not finite and above 0"));
    }
    if !(rope.rope_base.is_finite() && rope.rope_base > 1.0) {
        return Err(rule("rope_base", "rope_base is not finite and above 1"));
    }
    Ok(())
}

/// Size and alignment of each range, one `require_fit` per range. Only valid after
/// [`check_shape`], so the sizes cannot overflow.
fn check_fit(call: &AttnPrepCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    let row_w = row_width(shape.n_head, shape.n_kv_head);
    let cache = cache_bytes(shape.rows, shape.n_kv_head);
    require_fit("proj", &call.proj, proj_bytes(shape.m, shape.ld, row_w), 4)?;
    require_fit("q_norm", &call.q_norm, NORM_BYTES, 4)?;
    require_fit("k_norm", &call.k_norm, NORM_BYTES, 4)?;
    require_fit("q", &call.q, q_bytes(shape.m, shape.n_head), 16)?;
    require_fit("k_cache", &call.k_cache, cache, 16)?;
    require_fit("v_cache", &call.v_cache, cache, 16)?;
    require_fit("pos", &call.pos, 4, 4)?;
    require_fit("status", &call.status, 4, 4)
}

/// Disjointness of the touched extents: the four written ranges (`q`, both caches, `status`)
/// pairwise (6 pairs) and each against the four read ranges (16 pairs). The read ranges may
/// overlap each other. The claims on `pos` and `status` are their 4 bytes, not the whole
/// `StepParams` block.
fn check_apart(call: &AttnPrepCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    let row_w = row_width(shape.n_head, shape.n_kv_head);
    let cache = cache_bytes(shape.rows, shape.n_kv_head);
    let written = [
        Claim::of("q", &call.q, q_bytes(shape.m, shape.n_head)),
        Claim::of("k_cache", &call.k_cache, cache),
        Claim::of("v_cache", &call.v_cache, cache),
        Claim::of("status", &call.status, 4),
    ];
    let read = [
        Claim::of("proj", &call.proj, proj_bytes(shape.m, shape.ld, row_w)),
        Claim::of("q_norm", &call.q_norm, NORM_BYTES),
        Claim::of("k_norm", &call.k_norm, NORM_BYTES),
        Claim::of("pos", &call.pos, 4),
    ];
    for (i, claim) in written.iter().enumerate() {
        for other in written[i + 1..].iter().chain(read.iter()) {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Queues the attention preparation of `call` on `stream`: RMSNorm and rope of every q and k
    /// head at position `*pos + t`, the `q` output, and the append of the k and v rows to the
    /// caches. Returns when the kernel is queued; the outputs are valid after the stream drained.
    ///
    /// The device word `*pos` is authoritative and the kernel reads only it; `admit_pos` is the
    /// host's advisory value for the admission check. A word outside `0..=rows - m` makes the
    /// launch write NaN to `q`, no KV row, and OR mask 4 into the status word. Mellum
    /// configurations (head size 128, no gate, a ring) are refused.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a buffer
    /// or stream of another context, a freed buffer, a shape, rope scalar or configuration the
    /// kernel refuses, an `admit_pos` that does not leave `m` rows, a range that is too short or
    /// misaligned, and a written range that overlaps another range of the call.
    pub(crate) fn attn_prep(
        &self,
        stream: &DeviceStream,
        call: AttnPrepCall<'_>,
    ) -> Result<(), EngError> {
        let proj = self.raw_range(&call.proj)?;
        let q_norm = self.raw_range(&call.q_norm)?;
        let k_norm = self.raw_range(&call.k_norm)?;
        let k_cache = self.raw_range(&call.k_cache)?;
        let v_cache = self.raw_range(&call.v_cache)?;
        let q = self.raw_range(&call.q)?;
        let pos = self.raw_range(&call.pos)?;
        let status = self.raw_range(&call.status)?;
        let ints = check_shape(&call.shape)?;
        check_rope(&call.rope)?;
        ensure_rows(call.admit_pos, call.shape.m, call.shape.rows)?;
        check_fit(&call)?;
        check_apart(&call)?;
        let RopeParams { eps, rope_base } = call.rope;
        self.launch_on(stream, move |raw| {
            attn_prep_launch(
                raw,
                AttnPrepRaw {
                    proj,
                    q_norm,
                    k_norm,
                    k_cache,
                    v_cache,
                    q,
                    pos,
                    status,
                    ints,
                    eps,
                    rope_base,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
