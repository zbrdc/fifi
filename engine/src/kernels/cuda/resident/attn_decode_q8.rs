//! Safe launch wrapper of the KV8 split-KV decode attention kernel: the split-KV decode
//! attention of `attn_decode` over the int8 code and fp16 scale planes of the KV8 format, read
//! in contiguous 64-cell tiles and dequantized in registers.
//!
//! The packed-int8 layout follows Strata's laguna_kv8 flash decode.
//!
//! [`CudaContext::attn_decode_q8`] queues two kernels on a [`DeviceStream`] and returns: the
//! part kernel gives every (split, kv head, token) block one partial softmax of its key range,
//! dequantizing `code * f32(stored scale)` in registers, and the combine kernel merges the
//! partials in split-index order (no float atomic), divides and applies `sigmoid(gate)` when
//! `proj` is given. There is no scratch beyond `part` and no arena term. A fault in the queued
//! work surfaces at the next synchronize of the stream and latches the backend there.
//!
//! The contract is that of [`super::attn_decode`] (module doc there): the device `kv_len` word
//! is authoritative, token `t` sees the keys `0..kv_len + t` cut to the last `window`
//! positions, a linear cache (`ring` 0) needs `kv_len <= cells - (m - 1)` and a ring needs a
//! window that fits it, `out` is NaN for a `kv_len` outside its range, and the appended cells
//! must hold K and V before the call is queued. One difference: `min_keys` is a multiple of the
//! KV8 tile 64 (the fp16 kernel takes 32), so the split plans of the two kernels are not
//! interchangeable.
//!
//! Every argument is checked here, before any native call, and refused as `BadRequest`: the
//! rules mirror `valid()` of `attn_decode_q8.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). Local rules carry the tag
//! `attn_decode_q8: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::attn_decode_q8::{
    self, AttnDecodeQ8Raw, AttnQ8Ints, attn_decode_q8_launch, attn_decode_q8_sinks_launch,
};
use crate::kernels::cuda::resident::attn_decode::{AttnSpan, AttnSplit};

/// Query heads per kv head; the lane layout of the kernel needs exactly 8.
const GROUP: usize = 8;
/// The most tokens one call takes.
const MAX_TOKENS: usize = 16;
/// The most kv heads (the kernel's grid and the part layout are sized for it).
const MAX_KV_HEADS: usize = 64;
/// The most splits (one thread per split in the combine statistics step).
const MAX_SPLITS: usize = 128;
/// The fewest keys per split is a multiple of this tile of keys (the contiguous tile the kernel
/// reads; the fp16 kernel's tile is 32).
const TILE_KEYS: usize = 64;
/// The largest `min_keys`.
pub(crate) const MAX_MIN_KEYS: usize = 1 << 20;
/// Exclusive upper bound of the score scale.
const SCALE_LIMIT: f32 = 1e30;

/// Geometry of one call: `m` tokens, `n_head` query heads over `n_kv_head` kv heads (exactly 8
/// per kv head), `head_dim` 128 or 256, `cells` plane cells per kv head (the KV capacity) and
/// the row length `ld` of the projection (0 or unused without `proj`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnQ8Shape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cells: usize,
    pub(crate) ld: usize,
}

/// One KV8 decode attention call. All ranges are device ranges of this context.
///
/// - `q`: f32 `[m][n_head][head_dim]`, 16-byte aligned, after RMSNorm and rope, not scaled.
/// - `proj`: none (no output gate) or f32 `[m][ld]` with `ld >= n_head * 2 * head_dim`; head
///   `h`'s gate is at column `h * 2 * head_dim + head_dim + d`.
/// - `k_codes`, `v_codes`: int8 `[n_kv_head][cells][head_dim]`, 4-byte aligned.
/// - `k_scales`, `v_scales`: fp16 `[n_kv_head][cells][head_dim / 32]`, 4-byte aligned.
/// - `kv_len`: one device `i32`, 4-byte aligned.
/// - `part`: f32 scratch of [`attn_q8_part_bytes`] bytes, 16-byte aligned, needs no
///   initialization.
/// - `out`: f32 `[m][n_head * head_dim]`.
/// - `scale`: multiplies the scores, finite and in `(0, 1e30)`; `1/16` for head 256.
pub(crate) struct AttnDecodeQ8Call<'b> {
    pub(crate) shape: AttnQ8Shape,
    pub(crate) span: AttnSpan,
    pub(crate) split: AttnSplit,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) k_codes: DeviceRange<'b>,
    pub(crate) k_scales: DeviceRange<'b>,
    pub(crate) v_codes: DeviceRange<'b>,
    pub(crate) v_scales: DeviceRange<'b>,
    pub(crate) kv_len: DeviceRange<'b>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("attn_decode_q8: {tag}: {detail}"))
}

/// Bytes of the `part` scratch: `m * n_kv_head * max_splits * 8 * (2 + head_dim) * 4` (the
/// layout of the fp16 kernel).
///
/// Returns `BadRequest` ("size overflow") when the product does not fit `usize`.
pub(crate) fn attn_q8_part_bytes(
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

/// Every value that goes to C as `c_int` fits `i32`, converted once so that no conversion can
/// fail inside the launch closure. `ld` is 0 without `proj`.
fn convert_ints(call: &AttnDecodeQ8Call<'_>) -> Result<AttnQ8Ints, EngError> {
    let (shape, span, split) = (&call.shape, &call.span, &call.split);
    let ld = if call.proj.is_some() { shape.ld } else { 0 };
    Ok(AttnQ8Ints {
        m: to_int("m", shape.m)?,
        n_head: to_int("n_head", shape.n_head)?,
        n_kv_head: to_int("n_kv_head", shape.n_kv_head)?,
        head_dim: to_int("head_dim", shape.head_dim)?,
        ld: to_int("ld", ld)?,
        cells: to_int("cells", shape.cells)?,
        max_splits: to_int("max_splits", split.max_splits)?,
        min_keys: to_int("min_keys", split.min_keys)?,
        window: signed_int("window", span.window)?,
        ring: signed_int("ring", span.ring)?,
    })
}

/// Geometry: tokens, heads, head size, cells, the projection row.
fn check_geometry(call: &AttnDecodeQ8Call<'_>) -> Result<(), EngError> {
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
    if shape.cells < 1 {
        return Err(rule("cells", "cells is 0"));
    }
    if call.proj.is_some() && shape.ld < shape.n_head * 2 * shape.head_dim {
        return Err(rule("ld", "ld is below n_head * 2 * head_dim"));
    }
    Ok(())
}

/// Window and ring, as `attn_decode`: a ring needs a window, holds the window plus the later
/// tokens of the batch and fits the cells. The sum is in `i64`: `window` may be `i32::MAX`.
fn check_span(call: &AttnDecodeQ8Call<'_>) -> Result<(), EngError> {
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
    let cells = i64::try_from(shape.cells).map_err(|_| rule("cells", "too large"))?;
    if span.window == 0 {
        return Err(rule("ring", "a ring needs a window"));
    }
    if span.ring > cells {
        return Err(rule("ring", "ring is above cells"));
    }
    if span.ring < span.window + tokens_after {
        return Err(rule("ring", "ring is below window + m - 1"));
    }
    Ok(())
}

/// The split plan and the scale. `min_keys` is a multiple of the KV8 tile 64.
fn check_plan(call: &AttnDecodeQ8Call<'_>) -> Result<(), EngError> {
    let split = &call.split;
    if !(1..=MAX_SPLITS).contains(&split.max_splits) {
        return Err(rule("max_splits", "max_splits is not 1 to 128"));
    }
    let keys_ok = (TILE_KEYS..=MAX_MIN_KEYS).contains(&split.min_keys)
        && split.min_keys.is_multiple_of(TILE_KEYS);
    if !keys_ok {
        return Err(rule(
            "min_keys",
            "min_keys is not a multiple of 64 up to 2^20",
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
    codes: usize,
    scales: usize,
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

/// The sizes of every range.
fn needs(call: &AttnDecodeQ8Call<'_>) -> Result<Needs, EngError> {
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
    let values = product(&[shape.n_kv_head, shape.cells, shape.head_dim])?;
    let split = &call.split;
    let groups = values / 32;
    Ok(Needs {
        q: product(&[shape.m, heads, 4])?,
        proj,
        codes: values,
        scales: product(&[groups, 2])?,
        part: attn_q8_part_bytes(shape.m, shape.n_kv_head, shape.head_dim, split.max_splits)?,
        out: product(&[shape.m, heads, 4])?,
    })
}

/// Size and alignment, one `require_fit` per range.
fn check_fit(call: &AttnDecodeQ8Call<'_>, need: &Needs) -> Result<(), EngError> {
    require_fit("q", &call.q, need.q, 16)?;
    if let Some(proj) = &call.proj {
        require_fit("proj", proj, need.proj, 4)?;
    }
    require_fit("k_codes", &call.k_codes, need.codes, 4)?;
    require_fit("k_scales", &call.k_scales, need.scales, 4)?;
    require_fit("v_codes", &call.v_codes, need.codes, 4)?;
    require_fit("v_scales", &call.v_scales, need.scales, 4)?;
    require_fit("kv_len", &call.kv_len, 4, 4)?;
    require_fit("part", &call.part, need.part, 16)?;
    require_fit("out", &call.out, need.out, 4)
}

/// Disjointness: `out` and `part` are written, so each is apart from every other range; the
/// inputs may overlap each other.
fn check_apart(call: &AttnDecodeQ8Call<'_>, need: &Needs) -> Result<(), EngError> {
    let out = Claim::of("out", &call.out, need.out);
    let part = Claim::of("part", &call.part, need.part);
    let mut reads = vec![
        Claim::of("q", &call.q, need.q),
        Claim::of("k_codes", &call.k_codes, need.codes),
        Claim::of("k_scales", &call.k_scales, need.scales),
        Claim::of("v_codes", &call.v_codes, need.codes),
        Claim::of("v_scales", &call.v_scales, need.scales),
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
    /// Queues the KV8 decode attention of `call` on `stream`: the part kernel, then the
    /// fixed-order combine kernel. Returns when both are queued; the output is valid after the
    /// stream drained.
    ///
    /// The module doc lists what the caller guarantees (the `kv_len` word, the appended cells,
    /// the ordering of any other-stream use of `part`) and the NaN rule for a `kv_len` outside
    /// its range.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// stream or buffer of another context, an integer that does not fit `i32`, a geometry,
    /// window, ring, split or scale the kernel refuses, a range that is too short or misaligned,
    /// and an `out` or `part` range that overlaps another range of the call.
    pub(crate) fn attn_decode_q8(
        &self,
        stream: &DeviceStream,
        call: AttnDecodeQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.attn_decode_q8_inner(stream, call, None)
    }

    /// Adds one raw F32 zero-value softmax logit per query head, without QK scaling.
    /// Sinks must be disjoint from out/part; the device kv_len and capture contract is unchanged.
    #[allow(dead_code)] // Staged until a model caller is qualified.
    pub(crate) fn attn_decode_q8_sinks(
        &self,
        stream: &DeviceStream,
        call: AttnDecodeQ8Call<'_>,
        sinks: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.attn_decode_q8_inner(stream, call, Some(sinks))
    }

    fn attn_decode_q8_inner(
        &self,
        stream: &DeviceStream,
        call: AttnDecodeQ8Call<'_>,
        sinks: Option<DeviceRange<'_>>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = convert_ints(&call)?;
        check_geometry(&call)?;
        check_span(&call)?;
        check_plan(&call)?;
        let need = needs(&call)?;
        check_fit(&call, &need)?;
        check_apart(&call, &need)?;
        if let Some(sinks) = &sinks {
            let bytes = product(&[call.shape.n_head, 4])?;
            require_fit("decode sinks", sinks, bytes, 4)?;
            let read = Claim::of("decode sinks", sinks, bytes);
            require_apart(&Claim::of("out", &call.out, need.out), &read)?;
            require_apart(&Claim::of("part", &call.part, need.part), &read)?;
        }
        let q = self.raw_range(&call.q)?;
        let proj = call.proj.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let k_codes = self.raw_range(&call.k_codes)?;
        let k_scales = self.raw_range(&call.k_scales)?;
        let v_codes = self.raw_range(&call.v_codes)?;
        let v_scales = self.raw_range(&call.v_scales)?;
        let kv_len = self.raw_range(&call.kv_len)?;
        let part = self.raw_range(&call.part)?;
        let out = self.raw_range(&call.out)?;
        let sinks = sinks.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let scale = call.scale;
        self.launch_on(stream, move |raw| {
            let io = AttnDecodeQ8Raw {
                q,
                proj,
                k_codes,
                k_scales,
                v_codes,
                v_scales,
                kv_len,
                part,
                out,
                ints,
                scale,
            };
            match sinks {
                Some(sinks) => attn_decode_q8_sinks_launch(raw, io, sinks),
                None => attn_decode_q8_launch(raw, io),
            }
        })
    }
}

/// One local, nonempty page contributes to one group of the full stride-64 scratch.
/// Only one query row and all-visible linear attention are supported.
/// Low-level producers must use the same query, scale and geometry for every group;
/// preceding pages are full and the last has the remaining visible keys. Use the checked pass.
pub(crate) struct AttnQ8PartialCall<'b> {
    pub(crate) shape: AttnQ8Shape,
    pub(crate) visible_keys: usize,
    pub(crate) group: usize,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) k_codes: DeviceRange<'b>,
    pub(crate) k_scales: DeviceRange<'b>,
    pub(crate) v_codes: DeviceRange<'b>,
    pub(crate) v_scales: DeviceRange<'b>,
    pub(crate) part: DeviceRange<'b>,
}

/// Combines the populated page groups once; cells is the logical KV capacity.
/// visible_keys is the positive global visible count, not an append position.
/// Low-level callers must populate every active group in the same scratch generation,
/// with matching producer geometry and no concurrent writes, ordered before this merge.
/// Production uses the checked pass instead of composing these primitives directly.
pub(crate) struct AttnQ8CombineCall<'b> {
    pub(crate) shape: AttnQ8Shape,
    pub(crate) page_cells: usize,
    pub(crate) visible_keys: usize,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) out: DeviceRange<'b>,
}

const PAGED_SPLITS: usize = 64;

fn paged_ints(
    shape: &AttnQ8Shape,
    page_cells: usize,
    visible: usize,
    has_proj: bool,
) -> Result<AttnQ8Ints, EngError> {
    if shape.m != 1
        || !(1..=MAX_KV_HEADS).contains(&shape.n_kv_head)
        || shape.n_head != GROUP * shape.n_kv_head
        || !matches!(shape.head_dim, 128 | 256)
    {
        return Err(rule("paged geometry", "requires m1, 8:1 heads, D128/256"));
    }
    if !(TILE_KEYS..=MAX_MIN_KEYS).contains(&page_cells)
        || !page_cells.is_multiple_of(TILE_KEYS)
        || shape.cells == 0
        || shape.cells > product(&[page_cells, PAGED_SPLITS])?
        || visible == 0
        || visible > shape.cells
    {
        return Err(rule(
            "paged length",
            "invalid page, logical capacity or visible count",
        ));
    }
    if has_proj && shape.ld < shape.n_head * 2 * shape.head_dim {
        return Err(rule("ld", "ld is below n_head * 2 * head_dim"));
    }
    Ok(AttnQ8Ints {
        m: 1,
        n_head: to_int("n_head", shape.n_head)?,
        n_kv_head: to_int("n_kv_head", shape.n_kv_head)?,
        head_dim: to_int("head_dim", shape.head_dim)?,
        ld: if has_proj { to_int("ld", shape.ld)? } else { 0 },
        cells: to_int("cells", shape.cells)?,
        max_splits: PAGED_SPLITS as c_int,
        min_keys: to_int("page_cells", page_cells)?,
        window: 0,
        ring: 0,
    })
}

fn partial_ranges(call: &AttnQ8PartialCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    let codes = product(&[shape.n_kv_head, shape.cells, shape.head_dim])?;
    let scales = product(&[shape.n_kv_head, shape.cells, shape.head_dim / 32, 2])?;
    let q = product(&[shape.n_head, shape.head_dim, 4])?;
    let part = attn_q8_part_bytes(1, shape.n_kv_head, shape.head_dim, PAGED_SPLITS)?;
    require_fit("part", &call.part, part, 16)?;
    let written = Claim::of("part", &call.part, part);
    for (name, range, bytes, alignment) in [
        ("q", &call.q, q, 16),
        ("k_codes", &call.k_codes, codes, 4),
        ("k_scales", &call.k_scales, scales, 4),
        ("v_codes", &call.v_codes, codes, 4),
        ("v_scales", &call.v_scales, scales, 4),
    ] {
        require_fit(name, range, bytes, alignment)?;
        require_apart(&written, &Claim::of(name, range, bytes))?;
    }
    Ok(())
}

fn combine_ranges(call: &AttnQ8CombineCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    let out = product(&[shape.n_head, shape.head_dim, 4])?;
    let part = attn_q8_part_bytes(1, shape.n_kv_head, shape.head_dim, PAGED_SPLITS)?;
    require_fit("part", &call.part, part, 16)?;
    require_fit("out", &call.out, out, 4)?;
    let output = Claim::of("out", &call.out, out);
    let scratch = Claim::of("part", &call.part, part);
    require_apart(&output, &scratch)?;
    if let Some(proj) = &call.proj {
        let bytes = product(&[shape.n_head, shape.head_dim, 2, 4])?;
        require_fit("proj", proj, bytes, 4)?;
        let gate = Claim::of("proj", proj, bytes);
        require_apart(&output, &gate)?;
        require_apart(&scratch, &gate)?;
    }
    Ok(())
}

impl CudaContext {
    /// Queues one local page into one group of the full stride-64 scratch.
    /// Capture is refused; other-stream uses must be joined before this call.
    pub(super) fn attn_decode_q8_partial(
        &self,
        stream: &DeviceStream,
        call: AttnQ8PartialCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        self.ensure_uncaptured("paged attention")?;
        let ints = paged_ints(&call.shape, call.shape.cells, call.visible_keys, false)?;
        if call.group >= PAGED_SPLITS
            || !(call.scale.is_finite() && call.scale > 0.0 && call.scale < SCALE_LIMIT)
        {
            return Err(rule("paged partial", "invalid output group or score scale"));
        }
        partial_ranges(&call)?;
        let io = attn_decode_q8::AttnQ8PartialRaw {
            q: self.raw_range(&call.q)?,
            k_codes: self.raw_range(&call.k_codes)?,
            k_scales: self.raw_range(&call.k_scales)?,
            v_codes: self.raw_range(&call.v_codes)?,
            v_scales: self.raw_range(&call.v_scales)?,
            part: self.raw_range(&call.part)?,
            ints,
            visible: to_int("visible_keys", call.visible_keys)?,
            group: to_int("group", call.group)?,
            scale: call.scale,
        };
        self.launch_on(stream, |stream| {
            attn_decode_q8::attn_q8_partial_launch(stream, io)
        })
    }

    /// Combines populated groups in index order and applies the optional gate once.
    pub(super) fn attn_decode_q8_combine(
        &self,
        stream: &DeviceStream,
        call: AttnQ8CombineCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        self.ensure_uncaptured("paged attention")?;
        let ints = paged_ints(
            &call.shape,
            call.page_cells,
            call.visible_keys,
            call.proj.is_some(),
        )?;
        combine_ranges(&call)?;
        let io = attn_decode_q8::AttnQ8CombineRaw {
            part: self.raw_range(&call.part)?,
            proj: call
                .proj
                .as_ref()
                .map(|range| self.raw_range(range))
                .transpose()?,
            out: self.raw_range(&call.out)?,
            ints,
            visible: to_int("visible_keys", call.visible_keys)?,
        };
        self.launch_on(stream, |stream| {
            attn_decode_q8::attn_q8_combine_launch(stream, io)
        })
    }
}

pub(crate) mod paged;

#[cfg(all(test, feature = "native-cuda"))]
mod variant_tests;

#[cfg(all(test, feature = "native-cuda"))]
mod composition_tests;

#[cfg(all(test, feature = "native-cuda"))]
mod paged_tests;

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
