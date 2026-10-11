//! Safe launch wrapper of the split-KV decode attention kernel (RD-8b, decision 0047, seam 1).
//!
//! Design guidance only, no code copied: the idea is split-KV decode attention with a fixed-order
//! combine (MegaCapybara v1.35 static RE, `research/megacapybara-re-lessons.md` section 6.1 C10).
//!
//! [`CudaContext::attn_decode`] queues two kernels on a [`DeviceStream`] and returns: the part
//! kernel gives every (split, kv head, token) block one partial softmax of its key range, and the
//! combine kernel merges the partials in split-index order (no float atomic), divides, applies
//! `sigmoid(gate)` when `proj` is given and writes `out`. The call allocates nothing, copies
//! nothing and never synchronizes; a fault in the queued work surfaces at the next synchronize of
//! the stream and latches the backend there.
//!
//! What the caller guarantees:
//! - `kv_len` is a device `i32` word read when the kernels run, so one captured graph serves every
//!   position. Token 0 sees the keys `0..kv_len`, token `t` sees `0..kv_len + t`, cut to the last
//!   `window` positions when `window > 0`. For a decode step `kv_len = pos + 1` after `attn_prep`
//!   appended row `pos`; `attn_prep`'s `pos_dev` and this word must come from one `StepParams`,
//!   because an out-of-range `*pos_dev` is a silent out-of-bounds write in `attn_prep` that this
//!   kernel cannot repair.
//! - The cache rows `0..kv_len + m - 1` (or the ring cells) hold the appended K and V before the
//!   call is queued, ordered on the same stream or by an event.
//! - `part` is scratch that is dead after the call. Any use of its bytes on another stream (the
//!   arena may share them, RD-9a) must be ordered by an event; the overlap check of this call
//!   covers this call only.
//!
//! A `kv_len` outside `1..=cache_rows - (m - 1)` (a linear cache; any value in `1..2^30` for a
//! ring) makes `out` NaN. The host cannot check a device word, so the non-finite logit check ends
//! the job instead: a non-finite `out` reaches the logits and is status bit 2 (Backend `sample`,
//! REPLY-45), and an invalid `pos`/`kv_len` pair is bit 4 (Backend `step_params`, REPLY-47).
//!
//! Every other argument is checked here, before any native call, and refused as `BadRequest`:
//! the rules below mirror `valid()` of `attn_decode.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). Local rules carry the tag
//! `attn_decode: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::attn_decode::{
    AttnDecodeRaw, AttnInts, attn_decode_launch,
};

/// Query heads per kv head; the lane layout of the kernel needs exactly 8.
const GROUP: usize = 8;
/// The most tokens one call takes.
const MAX_TOKENS: usize = 16;
/// The most kv heads (the kernel's grid and the part layout are sized for it).
const MAX_KV_HEADS: usize = 64;
/// The most splits (one thread per split in the combine statistics step).
const MAX_SPLITS: usize = 128;
/// The fewest keys per split is a multiple of this tile of keys.
const TILE_KEYS: usize = 32;
/// The largest `min_keys`.
const MAX_MIN_KEYS: usize = 1 << 20;
/// Exclusive upper bound of the score scale.
const SCALE_LIMIT: f32 = 1e30;

/// Geometry of one call: `m` tokens, `n_head` query heads over `n_kv_head` kv heads (exactly 8
/// per kv head), `head_dim` 128 or 256, a cache of `cache_rows` rows and the row length `ld` of
/// the projection (0 or unused without `proj`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnShape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cache_rows: usize,
    pub(crate) ld: usize,
}

/// Which positions a token sees: `window` 0 is all of them, else the last `window` (the token's
/// own position counts as one); `ring` 0 is a linear cache, else position `p` lives in row
/// `p % ring` (needs `window > 0` and `window + m - 1 <= ring <= cache_rows`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnSpan {
    pub(crate) window: i64,
    pub(crate) ring: i64,
}

impl AttnSpan {
    /// All positions of a linear cache (Qwen3.6).
    pub(crate) const FULL: Self = Self { window: 0, ring: 0 };
}

/// The split plan: a grid of `max_splits` blocks per kv head and token, and the fewest keys
/// (a multiple of 32) worth one split. They are constants of a step: the same input gives the
/// same bytes only for one plan.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnSplit {
    pub(crate) max_splits: usize,
    pub(crate) min_keys: usize,
}

impl AttnSplit {
    /// The stage 1 plan.
    pub(crate) const STAGE1: Self = Self {
        max_splits: 64,
        min_keys: 32,
    };
}

/// One decode attention call. All ranges are device ranges of this context.
///
/// - `q`: f32 `[m][n_head][head_dim]`, 16-byte aligned, after RMSNorm and rope, not scaled.
/// - `proj`: none (no output gate) or f32 `[m][ld]` with `ld >= n_head * 2 * head_dim`; head
///   `h`'s gate is at column `h * 2 * head_dim + head_dim + d`.
/// - `k_cache`, `v_cache`: fp16 `[cache_rows][n_kv_head][head_dim]`, 16-byte aligned.
/// - `kv_len`: one device `i32`, 4-byte aligned.
/// - `part`: f32 scratch of [`attn_part_bytes`] bytes, 16-byte aligned, needs no initialization.
/// - `out`: f32 `[m][n_head * head_dim]`.
/// - `scale`: multiplies the scores, finite and in `(0, 1e30)`; `1/sqrt(256) = 0.0625` for Qwen3.6.
pub(crate) struct AttnDecodeCall<'b> {
    pub(crate) shape: AttnShape,
    pub(crate) span: AttnSpan,
    pub(crate) split: AttnSplit,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    pub(crate) kv_len: DeviceRange<'b>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("attn_decode: {tag}: {detail}"))
}

/// Bytes of the `part` scratch: `m * n_kv_head * max_splits * 8 * (2 + head_dim) * 4`.
///
/// Returns `BadRequest` ("size overflow") when the product does not fit `usize`.
pub(crate) fn attn_part_bytes(
    m: usize,
    n_kv_head: usize,
    head_dim: usize,
    max_splits: usize,
) -> Result<usize, EngError> {
    head_dim
        .checked_add(2)
        .and_then(|row| row.checked_mul(GROUP * 4))
        .and_then(|row| row.checked_mul(max_splits))
        .and_then(|row| row.checked_mul(n_kv_head))
        .and_then(|row| row.checked_mul(m))
        .ok_or_else(|| rule("size overflow", "the part scratch size overflows"))
}

/// `value` as `c_int`, or the "exceeds i32" refusal naming `name`.
fn to_int(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule("exceeds i32", &format!("{name} is {value}")))
}

/// `value` as `c_int` for the signed window and ring.
fn signed_int(name: &str, value: i64) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule("exceeds i32", &format!("{name} is {value}")))
}

/// Rule (2): every value that goes to C as `c_int` fits `i32`, converted once so that no
/// conversion can fail inside the launch closure. `ld` is 0 without `proj`.
fn convert_ints(call: &AttnDecodeCall<'_>) -> Result<AttnInts, EngError> {
    let (shape, span, split) = (&call.shape, &call.span, &call.split);
    let ld = if call.proj.is_some() { shape.ld } else { 0 };
    Ok(AttnInts {
        m: to_int("m", shape.m)?,
        n_head: to_int("n_head", shape.n_head)?,
        n_kv_head: to_int("n_kv_head", shape.n_kv_head)?,
        head_dim: to_int("head_dim", shape.head_dim)?,
        ld: to_int("ld", ld)?,
        cache_rows: to_int("cache_rows", shape.cache_rows)?,
        max_splits: to_int("max_splits", split.max_splits)?,
        min_keys: to_int("min_keys", split.min_keys)?,
        window: signed_int("window", span.window)?,
        ring: signed_int("ring", span.ring)?,
    })
}

/// Rule (3), geometry: tokens, heads, head size, row counts, the projection row.
fn check_geometry(call: &AttnDecodeCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    if !(1..=MAX_TOKENS).contains(&shape.m) {
        return Err(rule("tokens", &format!("m is {}, not 1 to 16", shape.m)));
    }
    let kv_heads = (1..=MAX_KV_HEADS).contains(&shape.n_kv_head);
    if !kv_heads || shape.n_head != GROUP * shape.n_kv_head {
        return Err(rule(
            "heads",
            &format!(
                "n_head {} and n_kv_head {} are not 8 query heads per kv head, 1 to 64 kv heads",
                shape.n_head, shape.n_kv_head
            ),
        ));
    }
    if shape.head_dim != 128 && shape.head_dim != 256 {
        return Err(rule("head_dim", "head_dim is not 128 or 256"));
    }
    if shape.cache_rows < 1 {
        return Err(rule("cache_rows", "cache_rows is 0"));
    }
    if call.proj.is_some() && shape.ld < shape.n_head * 2 * shape.head_dim {
        return Err(rule("ld", "ld is below n_head * 2 * head_dim"));
    }
    Ok(())
}

/// Rule (3), window and ring: a ring needs a window, holds the window plus the later tokens of
/// the batch and fits the cache. The sum is in `i64`: `window` may be `i32::MAX`.
fn check_span(call: &AttnDecodeCall<'_>) -> Result<(), EngError> {
    let (span, shape) = (&call.span, &call.shape);
    if span.window < 0 {
        return Err(rule("window", "window is negative"));
    }
    if span.ring < 0 {
        return Err(rule("ring", "ring is negative"));
    }
    if span.ring == 0 {
        return Ok(());
    }
    let tokens_after = i64::try_from(shape.m - 1).map_err(|_| rule("tokens", "m is too large"))?;
    let rows = i64::try_from(shape.cache_rows).map_err(|_| rule("cache_rows", "too large"))?;
    if span.window == 0 {
        return Err(rule("ring", "a ring needs a window"));
    }
    if span.ring > rows {
        return Err(rule("ring", "ring is above cache_rows"));
    }
    if span.ring < span.window + tokens_after {
        return Err(rule("ring", "ring is below window + m - 1"));
    }
    Ok(())
}

/// Rule (3), split plan and scale.
fn check_plan(call: &AttnDecodeCall<'_>) -> Result<(), EngError> {
    let split = &call.split;
    if !(1..=MAX_SPLITS).contains(&split.max_splits) {
        return Err(rule("max_splits", "max_splits is not 1 to 128"));
    }
    let keys_ok = (TILE_KEYS..=MAX_MIN_KEYS).contains(&split.min_keys)
        && split.min_keys.is_multiple_of(TILE_KEYS);
    if !keys_ok {
        return Err(rule(
            "min_keys",
            "min_keys is not a multiple of 32 up to 2^20",
        ));
    }
    if !(call.scale.is_finite() && call.scale > 0.0 && call.scale < SCALE_LIMIT) {
        return Err(rule("scale", "scale is not finite and in (0, 1e30)"));
    }
    Ok(())
}

/// Bytes each range must hold, from checked arithmetic.
struct Needs {
    q: usize,
    proj: usize,
    cache: usize,
    part: usize,
    out: usize,
}

/// Product of the factors, or the "size overflow" refusal.
fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(|| rule("size overflow", "a range size overflows"))
}

/// Rules (4) and (5), the sizes: every product is checked.
fn needs(call: &AttnDecodeCall<'_>) -> Result<Needs, EngError> {
    let shape = &call.shape;
    let heads = product(&[shape.n_head, shape.head_dim])?;
    let proj = if call.proj.is_some() {
        let last_rows = product(&[shape.m - 1, shape.ld, 4])?;
        last_rows
            .checked_add(product(&[heads, 2, 4])?)
            .ok_or_else(|| rule("size overflow", "the proj size overflows"))?
    } else {
        0
    };
    let split = &call.split;
    Ok(Needs {
        q: product(&[shape.m, heads, 4])?,
        proj,
        cache: product(&[shape.cache_rows, shape.n_kv_head, shape.head_dim, 2])?,
        part: attn_part_bytes(shape.m, shape.n_kv_head, shape.head_dim, split.max_splits)?,
        out: product(&[shape.m, heads, 4])?,
    })
}

/// Rules (4) and (5), size and alignment, one `require_fit` per range.
fn check_fit(call: &AttnDecodeCall<'_>, need: &Needs) -> Result<(), EngError> {
    require_fit("q", &call.q, need.q, 16)?;
    if let Some(proj) = &call.proj {
        require_fit("proj", proj, need.proj, 4)?;
    }
    require_fit("k_cache", &call.k_cache, need.cache, 16)?;
    require_fit("v_cache", &call.v_cache, need.cache, 16)?;
    require_fit("kv_len", &call.kv_len, 4, 4)?;
    require_fit("part", &call.part, need.part, 16)?;
    require_fit("out", &call.out, need.out, 4)
}

/// Rule (6), disjointness: `out` and `part` are written, so each is apart from every other range
/// (11 pairs with `proj`); the inputs may overlap each other.
fn check_apart(call: &AttnDecodeCall<'_>, need: &Needs) -> Result<(), EngError> {
    let out = Claim::of("out", &call.out, need.out);
    let part = Claim::of("part", &call.part, need.part);
    let mut reads = vec![
        Claim::of("q", &call.q, need.q),
        Claim::of("k_cache", &call.k_cache, need.cache),
        Claim::of("v_cache", &call.v_cache, need.cache),
        Claim::of("kv_len", &call.kv_len, 4),
    ];
    if let Some(proj) = &call.proj {
        reads.push(Claim::of("proj", proj, need.proj));
    }
    for read in &reads {
        require_apart(&out, read)?;
    }
    require_apart(&out, &part)?;
    for read in &reads {
        require_apart(&part, read)?;
    }
    Ok(())
}

impl CudaContext {
    /// Queues the decode attention of `call` on `stream`: the part kernel, then the combine
    /// kernel. Returns when both are queued; the output is valid after the stream drained.
    ///
    /// The module doc lists what the caller guarantees (the `kv_len` word, the appended cache
    /// rows, the ordering of any other-stream use of `part`) and the NaN rule for a `kv_len`
    /// outside its range.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// stream or buffer of another context, an integer that does not fit `i32`, a geometry,
    /// window, ring, split or scale the kernel refuses, a range that is too short or misaligned,
    /// and an `out` or `part` range that overlaps another range of the call.
    pub(crate) fn attn_decode(
        &self,
        stream: &DeviceStream,
        call: AttnDecodeCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = convert_ints(&call)?;
        check_geometry(&call)?;
        check_span(&call)?;
        check_plan(&call)?;
        let need = needs(&call)?;
        check_fit(&call, &need)?;
        check_apart(&call, &need)?;
        let q = self.raw_range(&call.q)?;
        let proj = call.proj.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let k_cache = self.raw_range(&call.k_cache)?;
        let v_cache = self.raw_range(&call.v_cache)?;
        let kv_len = self.raw_range(&call.kv_len)?;
        let part = self.raw_range(&call.part)?;
        let out = self.raw_range(&call.out)?;
        let scale = call.scale;
        self.launch_on(stream, move |raw| {
            attn_decode_launch(
                raw,
                AttnDecodeRaw {
                    q,
                    proj,
                    k_cache,
                    v_cache,
                    kv_len,
                    part,
                    out,
                    ints,
                    scale,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
