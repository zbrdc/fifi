//! Pure host-side plan and sizing of the chunked-prefill attention (CPF-ATTN-A, decision
//! 0049 items 9 and 11). No device code and no allocation: these functions pick the KV-split
//! plan of `attn_prefill` and size its partial buffer, so the driver (CPF-7a) can admit a
//! chunk inside the measured budget before any launch.
//!
//! The partial layout is the decode layout of `attn_decode` generalised from `G` heads times
//! `m` tokens to `G` heads times `m` rows (a query tile of 8 rows holds 8 times `G`
//! row-cells): every query row keeps its own (max, sum) pair and its own `head_dim`-float
//! accumulator per query head of its kv head's group, so the bytes are
//! `ceil(m/q_tile) * q_tile * n_kv_head * max_splits * G * (2 + head_dim) * 4` with the rows
//! rounded up to the tile.
//!
//! The plan is independent of `kv_len`: the buffer serves the whole launch grid, and the
//! kernels write only the splits the key count activates (the device split rule reduces
//! `max_splits`, it never raises it). That is the no-N-squared property of decision 0049 —
//! a deeper context never grows this workspace — and the sizing tests pin it.
//!
//! Arithmetic is checked throughout: an overflowing product is a `BadRequest` (the pattern of
//! `resident::attn_decode::attn_part_bytes`), and a plan whose partials do not fit
//! [`CPF_ATTN_PART_CAP`] at one split is a `MemoryPlan` refusal: the chunk is too large for
//! the budget and must shrink (decision 0049 item 9; the driver owns admission).
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the prefill driver (decision 0049)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::attn_prefill::{AttnPrefillRaw, attn_prefill_launch};
use crate::kernels::cuda::resident::attn_prep::{HEAD_DIM, MAX_ROWS, MAX_TOKENS};

/// Query heads per kv head: the compiled group (lane C condition 8).
const GROUP: usize = 8;
/// The most kv heads (the kernel's grid and the part layout are sized for it).
const MAX_KV_HEADS: usize = 64;
/// The largest `min_keys` (attn_decode's bound).
const MAX_MIN_KEYS: usize = 1 << 20;
/// Exclusive upper bound of the score scale.
const SCALE_LIMIT: f32 = 1e30;

/// Query rows per tile of the prefill attention kernels: the compiled tile of design section
/// 3.7 (eight query heads of a GQA group packed with eight tokens into 64 MMA rows).
pub(crate) const CPF_ATTN_TILE_ROWS: usize = 8;
/// Keys per tile of the prefill attention kernels: the granule of a key range (the tile of
/// `attn_decode`).
pub(crate) const CPF_ATTN_TILE_KEYS: usize = 32;
/// The fewest keys worth one split at chunk scale (a multiple of the key tile).
pub(crate) const CPF_ATTN_MIN_KEYS: usize = 256;
/// The most splits one plan takes (one combine thread per split, the `attn_decode` bound).
pub(crate) const CPF_ATTN_MAX_SPLITS: usize = 128;
/// The SM count the five-argument sizing form binds: the 3090's 82 SMs.
pub(crate) const CPF_ATTN_SM_COUNT: usize = 82;
/// The block target of the plan: two blocks per SM (`2 x 82` on the 3090, the "below 2 x 82"
/// rule of design section 3.7).
pub(crate) const CPF_ATTN_SM_TARGET: usize = 2 * CPF_ATTN_SM_COUNT;
/// The KV-split partial budget of decision 0049 item 9: 64 MiB, charged flat in every
/// admission row. The plan-sized bytes never exceed it (the plan reduces its splits to fit).
pub(crate) const CPF_ATTN_PART_CAP: usize = 64 * 1024 * 1024;

/// The geometry of one prefill attention call: a chunk of `m` rows, `n_head` query heads over
/// `n_kv_head` kv heads, and the head dimension. Every value is a run-time argument (lane C
/// condition 8); the sizing is generic in all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpfAttnGeometry {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
}

/// The split plan of one prefill attention call: the query tile, the launch grid's split
/// bound and the fewest keys (a multiple of [`CPF_ATTN_TILE_KEYS`]) worth one split. The
/// part kernel and the combine recompute the same deterministic plan from the launch
/// arguments, so no device word carries it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpfAttnPlan {
    pub(crate) q_tile: usize,
    pub(crate) max_splits: usize,
    pub(crate) min_keys: usize,
}

/// A `BadRequest` that names the sizing rule that fired (the `attn_prefill` tag; the wrapper
/// mirrors these texts).
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("attn_prefill: {tag}: {detail}"))
}

/// Query heads per kv head of `geometry` (`n_head / n_kv_head`), or the `group` refusal when
/// the query heads do not divide into equal groups: a row-cell is `G` heads wide.
fn group_of(geometry: CpfAttnGeometry) -> Result<usize, EngError> {
    if geometry.n_kv_head == 0 || !geometry.n_head.is_multiple_of(geometry.n_kv_head) {
        return Err(rule(
            "group",
            &format!(
                "n_head {} is not a whole multiple of n_kv_head {}",
                geometry.n_head, geometry.n_kv_head
            ),
        ));
    }
    Ok(geometry.n_head / geometry.n_kv_head)
}

/// Tiles of `q_tile` rows that cover `m` rows, rounded up.
fn tiles(m: usize, q_tile: usize) -> Result<usize, EngError> {
    if q_tile == 0 {
        return Err(rule("q_tile", "q_tile is 0"));
    }
    Ok(m.div_ceil(q_tile))
}

/// The split plan of one call. `max_splits` fills the block target ([`CPF_ATTN_SM_TARGET`]
/// from `sm_count`) after the `q_tiles * n_kv_head` base grid, then shrinks until the
/// partial bytes fit [`CPF_ATTN_PART_CAP`]; the floor is one split, and a chunk whose
/// partials do not fit at one split is a `MemoryPlan` refusal.
///
/// `kv_len` (the keys visible to the chunk's last row, `p0 + m`) fixes no part of the plan:
/// the buffer serves every split of the grid and the kernels keep only the splits the key
/// count activates, so the plan — and the workspace it sizes — is independent of `kv_len`.
/// That is the no-N-squared rule the tests pin; the argument stays (named `_kv_len`, since
/// no line of this function reads it) so the signature names the launch geometry the plan
/// belongs to.
pub(crate) fn cpf_attn_split_plan(
    geometry: CpfAttnGeometry,
    _kv_len: usize,
    q_tile: usize,
    sm_count: usize,
) -> Result<CpfAttnPlan, EngError> {
    let q_tiles = tiles(geometry.m, q_tile)?;
    let base = q_tiles
        .checked_mul(geometry.n_kv_head)
        .ok_or_else(|| rule("size overflow", "the base grid size overflows"))?;
    if base == 0 {
        return Err(rule("m", "a chunk has at least one row"));
    }
    let target = sm_count
        .checked_mul(2)
        .ok_or_else(|| rule("size overflow", "twice the SM count overflows"))?;
    let mut max_splits = target.div_ceil(base).clamp(1, CPF_ATTN_MAX_SPLITS);
    loop {
        let plan = CpfAttnPlan {
            q_tile,
            max_splits,
            min_keys: CPF_ATTN_MIN_KEYS,
        };
        if cpf_attn_part_bytes(geometry, plan)? <= CPF_ATTN_PART_CAP {
            return Ok(plan);
        }
        if max_splits == 1 {
            return Err(EngError::MemoryPlan(format!(
                "attn_prefill: part cap: the partials of a chunk of {} rows over {} kv heads \
                 need more than the {} byte budget at one split; shrink the chunk",
                geometry.m, geometry.n_kv_head, CPF_ATTN_PART_CAP
            )));
        }
        max_splits -= 1;
    }
}

/// Bytes of the `part` scratch of one plan: `ceil(m/q_tile) * q_tile * n_kv_head *
/// max_splits * G * (2 + head_dim) * 4`, the decode partial layout with the rows rounded up
/// to the tile (every query row of a tile keeps its own statistics pair and its own
/// accumulator, so a q_tile cell holds `q_tile * G` stat pairs then `q_tile * G * head_dim`
/// accumulator floats).
pub(crate) fn cpf_attn_part_bytes(
    geometry: CpfAttnGeometry,
    plan: CpfAttnPlan,
) -> Result<usize, EngError> {
    let group = group_of(geometry)?;
    let rows = tiles(geometry.m, plan.q_tile)?
        .checked_mul(plan.q_tile)
        .ok_or_else(|| rule("size overflow", "the rounded row count overflows"))?;
    geometry
        .head_dim
        .checked_add(2)
        .and_then(|per_head| per_head.checked_mul(group))
        .and_then(|per_cell| per_cell.checked_mul(plan.max_splits))
        .and_then(|per_row| per_row.checked_mul(geometry.n_kv_head))
        .and_then(|cells| cells.checked_mul(rows))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(|| rule("size overflow", "the part scratch size overflows"))
}

/// Bytes of the whole attention workspace of one chunk on the default device (the 3090's
/// [`CPF_ATTN_SM_COUNT`] SMs): the composition of [`cpf_attn_split_plan`] and
/// [`cpf_attn_part_bytes`]. `heads` is the geometry pair `(n_head, n_kv_head)` — the partial
/// cell is `G` heads wide, so both head counts enter — and `kv_len` is the keys the chunk
/// attends; the bytes do not depend on it.
pub(crate) fn cpf_attn_workspace_bytes(
    chunk: usize,
    heads: (usize, usize),
    head_dim: usize,
    kv_len: usize,
    tile: usize,
) -> Result<usize, EngError> {
    cpf_attn_workspace_bytes_for_sm(chunk, heads, head_dim, kv_len, tile, CPF_ATTN_SM_COUNT)
}

/// The same composition with the SM count explicit, so a smaller device plans against its own
/// block target.
pub(crate) fn cpf_attn_workspace_bytes_for_sm(
    chunk: usize,
    heads: (usize, usize),
    head_dim: usize,
    kv_len: usize,
    tile: usize,
    sm_count: usize,
) -> Result<usize, EngError> {
    let geometry = CpfAttnGeometry {
        m: chunk,
        n_head: heads.0,
        n_kv_head: heads.1,
        head_dim,
    };
    let plan = cpf_attn_split_plan(geometry, kv_len, tile, sm_count)?;
    cpf_attn_part_bytes(geometry, plan)
}

// ---- the safe wrapper (CPF-ATTN-A C7) -----------------------------------------------------

/// The span rules of a call, `attn_decode`'s [`AttnSpan`] (window 0 is all positions, else the
/// last `window`; ring 0 is a linear cache, else position p lives in row `p % ring`).
pub(crate) type CpfAttnSpan = crate::kernels::cuda::resident::attn_decode::AttnSpan;

/// The split plan of a call, `attn_decode`'s [`AttnSplit`] type with the prefill plan's
/// values: see [`CPF_ATTN_MIN_KEYS`] and the plan above.
pub(crate) type CpfAttnSplit = crate::kernels::cuda::resident::attn_decode::AttnSplit;

/// The compiled query tile (lane C ruling 5: fixed at 8 and refused otherwise).
pub(crate) const CPF_ATTN_Q_TILE: usize = CPF_ATTN_TILE_ROWS;

/// Geometry of one prefill attention call: `m` tokens at the positions `p0 .. p0 + m`,
/// `n_head` query heads over `n_kv_head` kv heads (exactly 8 per kv head), the head size, the
/// cache length and the projection row length (0 or unused without `proj`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnPrefillShape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cache_rows: usize,
    pub(crate) ld: usize,
    /// The position of token 0: a launch argument (decision 0049 item 1), never a device word.
    pub(crate) p0: i64,
}

/// One prefill attention call. All ranges are device ranges of this context.
///
/// - `q`: f32 `[m][n_head][head_dim]`, 16-byte aligned, after RMSNorm and rope, not scaled.
/// - `proj`: none (no output gate) or f32 `[m][ld]` with `ld >= n_head * 2 * head_dim`; head
///   `h`'s gate is at column `h * 2 * head_dim + head_dim + d`.
/// - `k_cache`, `v_cache`: fp16 `[cache_rows][n_kv_head][head_dim]`, 16-byte aligned.
/// - `part`: f32 scratch of [`cpf_attn_part_bytes`] bytes (the plan's), 16-byte aligned, no
///   initialization needed; only the active splits are written.
/// - `out`: f32 `[m][n_head * head_dim]`, 16-byte aligned; NaN on a refused launch.
/// - `status`: `None`, or one device `i32` — the prefill status word of decision 0049 item 10
///   — that receives bit 8 on a refused launch and is untouched by a valid one.
/// - `q_tile`: the compiled tile; only 8 runs (lane C ruling 5).
/// - `scale`: multiplies the scores, finite and in `(0, 1e30)`; `1/sqrt(256) = 0.0625`.
pub(crate) struct AttnPrefillCall<'b> {
    pub(crate) shape: AttnPrefillShape,
    pub(crate) span: CpfAttnSpan,
    pub(crate) split: CpfAttnSplit,
    pub(crate) q_tile: usize,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

/// The geometry view of a call for the sizing functions.
fn geometry(shape: &AttnPrefillShape) -> CpfAttnGeometry {
    CpfAttnGeometry {
        m: shape.m,
        n_head: shape.n_head,
        n_kv_head: shape.n_kv_head,
        head_dim: shape.head_dim,
    }
}

/// The plan view of a call's split values.
fn plan_of(split: &CpfAttnSplit) -> CpfAttnPlan {
    CpfAttnPlan {
        q_tile: CPF_ATTN_Q_TILE,
        max_splits: split.max_splits,
        min_keys: split.min_keys,
    }
}

/// `value` as `c_int` for the rule `tag`, or the "exceeds i32" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks every shape rule and converts the integers once, so that no conversion can fail
/// inside the launch closure. Each refusal carries the tag `attn_prefill: <rule>`.
pub(in crate::kernels::cuda) fn check_shape(
    shape: &AttnPrefillShape,
    q_tile: usize,
) -> Result<crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillInts, EngError> {
    use crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillInts;
    if q_tile != CPF_ATTN_Q_TILE {
        return Err(rule(
            "q_tile",
            &format!("q_tile is {q_tile}, not the compiled tile {CPF_ATTN_Q_TILE}"),
        ));
    }
    if shape.head_dim != HEAD_DIM {
        return Err(rule(
            "head_dim",
            &format!("head_dim is {}, only 256 runs", shape.head_dim),
        ));
    }
    let n_kv_head = shape.n_kv_head;
    if !(1..=MAX_KV_HEADS).contains(&n_kv_head) || shape.n_head != GROUP * n_kv_head {
        return Err(rule(
            "group",
            &format!(
                "n_head {} over n_kv_head {} is not 8 query heads per kv head, 1 to 64 kv heads",
                shape.n_head, n_kv_head
            ),
        ));
    }
    if !(1..=MAX_TOKENS).contains(&shape.m) {
        return Err(rule("m", &format!("m is {}, not 1 to 65535", shape.m)));
    }
    if !(1..=MAX_ROWS).contains(&shape.cache_rows) {
        return Err(rule(
            "cache_rows",
            &format!("cache_rows is {}, not 1 to 2^30 - 1", shape.cache_rows),
        ));
    }
    if shape.p0 < 0 {
        return Err(rule("p0", &format!("p0 is {}, below 0", shape.p0)));
    }
    let span = shape
        .p0
        .checked_add(i64::try_from(shape.m).map_err(|_| rule("m", "m is too large"))?)
        .ok_or_else(|| rule("p0", "p0 + m overflows"))?;
    if span > i64::try_from(shape.cache_rows).map_err(|_| rule("cache_rows", "too large"))? {
        return Err(rule(
            "p0",
            &format!(
                "p0 + m is {span}, above the {} cache rows",
                shape.cache_rows
            ),
        ));
    }
    Ok(AttnPrefillInts {
        ld: to_int("ld", shape.ld)?,
        p0: c_int::try_from(shape.p0).map_err(|_| rule("p0", "p0 is above i32::MAX"))?,
        m: to_int("m", shape.m)?,
        n_head: to_int("n_head", shape.n_head)?,
        n_kv_head: to_int("n_kv_head", n_kv_head)?,
        head_dim: to_int("head_dim", shape.head_dim)?,
        cache_rows: to_int("cache_rows", shape.cache_rows)?,
        max_splits: 0,
        min_keys: 0,
        window: 0,
        ring: 0,
    })
}

/// Checks the projection row, the span rules and the split plan.
fn check_span_split(call: &AttnPrefillCall<'_>) -> Result<(), EngError> {
    let (shape, span, split) = (&call.shape, &call.span, &call.split);
    if call.proj.is_some() && shape.ld < shape.n_head * 2 * shape.head_dim {
        return Err(rule("ld", "ld is below n_head * 2 * head_dim"));
    }
    if span.window < 0 {
        return Err(rule("window", "window is negative"));
    }
    if span.ring < 0 {
        return Err(rule("ring", "ring is negative"));
    }
    if span.ring > 0 {
        let tokens_after = i64::try_from(shape.m - 1).map_err(|_| rule("m", "m is too large"))?;
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
    }
    if !(1..=CPF_ATTN_MAX_SPLITS).contains(&split.max_splits) {
        return Err(rule(
            "max_splits",
            &format!("max_splits is {}, not 1 to 128", split.max_splits),
        ));
    }
    let keys_ok = (CPF_ATTN_TILE_KEYS..=MAX_MIN_KEYS).contains(&split.min_keys)
        && split.min_keys.is_multiple_of(CPF_ATTN_TILE_KEYS);
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

/// Bytes each range must hold, from the checked sizing functions (a product that cannot fit
/// `usize` was already refused as a shape rule; the saturating helpers stay total).
struct Needs {
    q: usize,
    proj: usize,
    cache: usize,
    part: usize,
    out: usize,
}

fn needs(call: &AttnPrefillCall<'_>) -> Result<Needs, EngError> {
    let shape = &call.shape;
    let heads = shape.n_head * shape.head_dim;
    let proj = if call.proj.is_some() {
        (shape.m - 1)
            .saturating_mul(shape.ld)
            .saturating_add(heads * 2)
            .saturating_mul(4)
    } else {
        0
    };
    Ok(Needs {
        q: shape.m.saturating_mul(heads).saturating_mul(4),
        proj,
        cache: shape
            .cache_rows
            .saturating_mul(shape.n_kv_head)
            .saturating_mul(shape.head_dim)
            .saturating_mul(2),
        part: cpf_attn_part_bytes(geometry(shape), plan_of(&call.split))?,
        out: shape.m.saturating_mul(heads).saturating_mul(4),
    })
}

/// Size and alignment of each range, one `require_fit` per range: the `part` guard compares
/// against the plan's bytes (`cpf_attn_part_bytes`).
fn check_fit(call: &AttnPrefillCall<'_>, need: &Needs) -> Result<(), EngError> {
    require_fit("q", &call.q, need.q, 16)?;
    if let Some(proj) = &call.proj {
        require_fit("proj", proj, need.proj, 4)?;
    }
    require_fit("k_cache", &call.k_cache, need.cache, 16)?;
    require_fit("v_cache", &call.v_cache, need.cache, 16)?;
    require_fit("part", &call.part, need.part, 16)?;
    require_fit("out", &call.out, need.out, 16)?;
    if let Some(status) = &call.status {
        require_fit("status", status, 4, 4)?;
    }
    Ok(())
}

/// Disjointness: `out` and `part` are written, so each is apart from every other range of the
/// call (14 pairs with `proj` and `status`); the inputs may overlap each other.
fn check_apart(call: &AttnPrefillCall<'_>, need: &Needs) -> Result<(), EngError> {
    let out = Claim::of("out", &call.out, need.out);
    let part = Claim::of("part", &call.part, need.part);
    let mut reads = vec![
        Claim::of("q", &call.q, need.q),
        Claim::of("k_cache", &call.k_cache, need.cache),
        Claim::of("v_cache", &call.v_cache, need.cache),
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
    /// Queues the chunked-prefill attention of `call` on `stream`: the part kernel (one
    /// partial per query row per active split), then the combine (fixed order, the sigmoid
    /// output gate when `proj` is given). Returns when both are queued; the output is valid
    /// after the stream drained.
    ///
    /// A `p0` outside `0..=cache_rows - m` is refused here (and by the C entry, and
    /// re-checked on the device, where it writes no partial, NaN `out` and bit 8 of the
    /// optional status word). A chunk whose partials exceed the 64 MiB budget at the call's
    /// own split count is a `MemoryPlan` refusal: shrink the chunk.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a tile, shape, span, split or scale the kernel
    /// refuses, a `p0 + m` outside the cache, a range that is too short or misaligned, and
    /// an `out` or `part` range that overlaps another range of the call.
    pub(crate) fn attn_prefill(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let mut ints = check_shape(&call.shape, call.q_tile)?;
        ints.ld = if call.proj.is_some() { ints.ld } else { 0 };
        check_span_split(&call)?;
        let bytes = cpf_attn_part_bytes(geometry(&call.shape), plan_of(&call.split))?;
        if bytes > CPF_ATTN_PART_CAP {
            return Err(EngError::MemoryPlan(format!(
                "attn_prefill: part cap: the partials of a chunk of {} rows over {} kv heads \
                 at {} splits need {bytes} bytes, above the {CPF_ATTN_PART_CAP} byte budget; \
                 shrink the chunk",
                call.shape.m, call.shape.n_kv_head, call.split.max_splits
            )));
        }
        let need = needs(&call)?;
        check_fit(&call, &need)?;
        check_apart(&call, &need)?;
        ints.max_splits = to_int("max_splits", call.split.max_splits)?;
        ints.min_keys = to_int("min_keys", call.split.min_keys)?;
        ints.window = i64_int("window", call.span.window)?;
        ints.ring = i64_int("ring", call.span.ring)?;
        let AttnPrefillCall {
            q,
            proj,
            k_cache,
            v_cache,
            part,
            out,
            status,
            scale,
            ..
        } = &call;
        let (q, k_cache, v_cache, part, out) = (
            self.raw_range(q)?,
            self.raw_range(k_cache)?,
            self.raw_range(v_cache)?,
            self.raw_range(part)?,
            self.raw_range(out)?,
        );
        let proj = proj.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let status = status.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let scale = *scale;
        self.launch_on(stream, move |raw| {
            attn_prefill_launch(
                raw,
                AttnPrefillRaw {
                    q,
                    proj,
                    k_cache,
                    v_cache,
                    part,
                    out,
                    status,
                    ints,
                    scale,
                },
            )
        })
    }
}

/// `value` as `c_int` for the signed window and ring.
fn i64_int(name: &str, value: i64) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule("exceeds i32", &format!("{name} is {value}")))
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
