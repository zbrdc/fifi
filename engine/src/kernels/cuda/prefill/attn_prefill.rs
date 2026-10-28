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
//! Two plans. [`cpf_attn_split_plan`] sizes `s_A`, the minimum split count, and is
//! independent of `kv_len` (the kernels write only the splits the key count activates; the
//! device split rule reduces `max_splits`, it never raises it). [`cpf_attn_launch_plan`] is
//! the only plan the driver calls, for the grid and the launched bytes: it starts from `s_A`
//! and, at depth, raises the split count to fill whole waves of the SMs, never past the part
//! cap. So the launched bytes grow with `kv_len`, up to [`CPF_ATTN_PART_CAP`], while the
//! allocation does not: the driver allocates `part` at the flat cap, which admission already
//! charges in every row (decision 0049 item 9). That keeps the no-N-squared property of
//! decision 0049 — a deeper context never grows the workspace — and the sizing tests pin it.
//!
//! Arithmetic is checked throughout: an overflowing product is a `BadRequest` (the pattern of
//! `resident::attn_decode::attn_part_bytes`), and a plan whose partials do not fit
//! [`CPF_ATTN_PART_CAP`] at one split is a `MemoryPlan` refusal: the chunk is too large for
//! the budget and must shrink (decision 0049 item 9; the driver owns admission).
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for CPF-7, the prefill driver (decision 0049)"
    )
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::attn_prefill::{
    AttnKvRaw, AttnPrefillLaunch, AttnPrefillRaw, attn_prefill_launch,
};
use crate::kernels::cuda::resident::attn_prep::{HEAD_DIM, MAX_ROWS, MAX_TOKENS};
use crate::tensor::Dtype;

/// The head-lane width of the packed MMA tile; GQA1 uses its first lane.
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
/// The block target of the split plan: a 164-block target of design section 3.7 (`2 x 82` on
/// the 3090). It is a block count, not an occupancy: the part kernel runs one block per SM
/// ([`CPF_ATTN_BLOCKS_PER_SM`]).
pub(crate) const CPF_ATTN_SM_TARGET: usize = 2 * CPF_ATTN_SM_COUNT;
/// Blocks of the part kernel resident on one SM at once: its 96 KB of shared memory and its
/// 512 threads at 128 registers (the whole register file) each allow one. The launch plan
/// counts waves in these slots.
pub(crate) const CPF_ATTN_BLOCKS_PER_SM: usize = 1;
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
    if geometry.n_head == 0
        || geometry.n_kv_head == 0
        || !geometry.n_head.is_multiple_of(geometry.n_kv_head)
    {
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

/// The split plan of one call: it sizes `s_A`, the minimum split count, not the launch
/// ([`cpf_attn_launch_plan`] starts from it). `max_splits` fills the block target
/// ([`CPF_ATTN_SM_TARGET`] from `sm_count`) after the `q_tiles * n_kv_head` base grid, then
/// shrinks until the partial bytes fit [`CPF_ATTN_PART_CAP`]; the floor is one split, and a
/// chunk whose partials do not fit at one split is a `MemoryPlan` refusal.
///
/// `kv_len` (the keys visible to the chunk's last row, `p0 + m`) fixes no part of this plan,
/// so `s_A` — and the workspace [`cpf_attn_workspace_bytes`] sizes from it — is independent
/// of `kv_len`. That is the no-N-squared rule the tests pin; the argument stays (named
/// `_kv_len`, since no line of this function reads it) so the signature names the launch
/// geometry the plan belongs to.
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

/// Bytes of the split plan's workspace of one chunk on the default device (the 3090's
/// [`CPF_ATTN_SM_COUNT`] SMs): the composition of [`cpf_attn_split_plan`] and
/// [`cpf_attn_part_bytes`], so the bytes of `s_A` splits. `heads` is the geometry pair
/// `(n_head, n_kv_head)` — the partial cell is `G` heads wide, so both head counts enter —
/// and `kv_len` is the keys the chunk attends; the bytes do not depend on it. The launch
/// plan may launch more splits at depth; the driver allocates `part` at the flat
/// [`CPF_ATTN_PART_CAP`], which bounds both.
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

/// The launch plan of one call: the only plan function the driver calls, for the grid and
/// the launched bytes (`cpf_attn_part_bytes` of this plan, at most [`CPF_ATTN_PART_CAP`];
/// the driver allocates `part` at the cap). `kv_len` is `p0 + m`; `window` is the span's
/// window (0: all keys).
///
/// It starts from `s_A` ([`cpf_attn_split_plan`]). At depth, where every query tile has keys
/// for more splits, it takes the split count in `s_A + 1 ..= s_cap` whose `q_tiles * n_kv_head
/// * s` blocks fill whole waves of `sm_count * CPF_ATTN_BLOCKS_PER_SM` slots best (ties go to
/// the smaller count). `s_cap` is 128 walked down to the most splits whose partials fit the
/// cap; the `ceil(c_min / 256)` bound — `c_min` the fewest keys of any tile, from the
/// device's union-of-windows rule — is implied by full activation and debug-asserted, not
/// enforced. A count above `s_A` is taken only when every tile activates
/// all its splits under the device split rule, so the blocks carry equal work. Equality of
/// this plan and the launch's `max_splits` is the caller's contract.
///
/// Refusals: those of the split plan; `kv_len < m` (`attn_prefill: kv_len`); an `sm_count`
/// of 0 (`attn_prefill: sm_count`).
pub(crate) fn cpf_attn_launch_plan(
    geometry: CpfAttnGeometry,
    kv_len: usize,
    window: usize,
    sm_count: usize,
) -> Result<CpfAttnPlan, EngError> {
    let least = cpf_attn_split_plan(geometry, kv_len, CPF_ATTN_Q_TILE, sm_count)?;
    let p0 = kv_len.checked_sub(geometry.m).ok_or_else(|| {
        rule(
            "kv_len",
            &format!("kv_len {kv_len} is below the chunk's {} rows", geometry.m),
        )
    })?;
    if sm_count == 0 {
        return Err(rule("sm_count", "sm_count is 0"));
    }
    let c_min = tile_key_counts(geometry.m, p0, window).min().unwrap_or(0);
    let mut s_cap = CPF_ATTN_MAX_SPLITS;
    while s_cap > least.max_splits
        && cpf_attn_part_bytes(
            geometry,
            CpfAttnPlan {
                max_splits: s_cap,
                ..least
            },
        )? > CPF_ATTN_PART_CAP
    {
        s_cap -= 1;
    }
    let base = geometry
        .m
        .div_ceil(CPF_ATTN_Q_TILE)
        .saturating_mul(geometry.n_kv_head);
    let slots = sm_count.saturating_mul(CPF_ATTN_BLOCKS_PER_SM);
    let waves = |s: usize| base.saturating_mul(s).div_ceil(slots);
    let mut best = least.max_splits;
    for s in least.max_splits + 1..=s_cap {
        let all_active = tile_key_counts(geometry.m, p0, window)
            .all(|count| active_splits(count, s, CPF_ATTN_MIN_KEYS) == s);
        // Efficiency is s / waves(s) up to the common factor base / slots: compare the
        // fractions exactly, and keep the smaller count on a tie.
        if all_active && s.saturating_mul(waves(best)) > best.saturating_mul(waves(s)) {
            best = s;
        }
    }
    // Full activation of every tile at min_keys CPF_ATTN_MIN_KEYS implies s <= ceil(c_min /
    // min_keys), so no explicit bound on s is needed (pinned over the whole domain by a test).
    debug_assert!(best == least.max_splits || best <= c_min.div_ceil(CPF_ATTN_MIN_KEYS));
    Ok(CpfAttnPlan {
        max_splits: best,
        ..least
    })
}

/// Keys of each query tile of a chunk of `m` rows at `p0`: the union of its rows' windows,
/// the device's `visible` rule (from the first row's oldest visible position to the last
/// row's own).
fn tile_key_counts(m: usize, p0: usize, window: usize) -> impl Iterator<Item = usize> {
    (0..m.div_ceil(CPF_ATTN_Q_TILE)).map(move |tile| {
        let t0 = tile * CPF_ATTN_Q_TILE;
        let n_keys = p0 + m.min(t0 + CPF_ATTN_Q_TILE);
        let first = if window > 0 {
            (p0 + t0 + 1).saturating_sub(window)
        } else {
            0
        };
        n_keys - first
    })
}

/// Splits that hold keys for a tile of `count` keys under the device split rule
/// (`split_plan` of attn_prefill.cu, attn_decode's verbatim).
fn active_splits(count: usize, max_splits: usize, min_keys: usize) -> usize {
    let n_split = count.div_ceil(min_keys).clamp(1, max_splits);
    let chunk = count
        .div_ceil(n_split)
        .div_ceil(CPF_ATTN_TILE_KEYS)
        .saturating_mul(CPF_ATTN_TILE_KEYS);
    count.div_ceil(chunk)
}

// ---- the safe wrapper (CPF-ATTN-A C7) -----------------------------------------------------

/// The span rules of a call, `attn_decode`'s [`AttnSpan`] (window 0 is all positions, else the
/// last `window`; ring 0 is a linear cache, else position p lives in row `p % ring`). A ring
/// uses physical `cache_rows >= ring` rows and absolute `p0 + m <= MAX_ROWS`. Stage the
/// immutable union `ring >= window + m - 1` before launch. Linear calls need `p0 + m <= cache_rows`.
pub(crate) type CpfAttnSpan = crate::kernels::cuda::resident::attn_decode::AttnSpan;

/// The split plan of a call, `attn_decode`'s [`AttnSplit`] type with the prefill plan's
/// values: see [`CPF_ATTN_MIN_KEYS`] and the plan above.
pub(crate) type CpfAttnSplit = crate::kernels::cuda::resident::attn_decode::AttnSplit;

/// The compiled query tile (lane C ruling 5: fixed at 8 and refused otherwise).
pub(crate) const CPF_ATTN_Q_TILE: usize = CPF_ATTN_TILE_ROWS;

/// Span sizing permits read-only broadcasts; output admission must reject self-overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnStrides {
    pub(crate) time_stride_bytes: usize,
    pub(crate) head_stride_bytes: usize,
}

impl AttnStrides {
    fn aligned(self, rows: usize, heads: usize, align: usize) -> bool {
        rows == 0
            || heads == 0
            || ((rows <= 1 || self.time_stride_bytes.is_multiple_of(align))
                && (heads <= 1 || self.head_stride_bytes.is_multiple_of(align)))
    }

    pub(crate) fn span_bytes(self, rows: usize, heads: usize, cell_bytes: usize) -> Option<usize> {
        if rows == 0 || heads == 0 {
            return Some(0);
        }
        if cell_bytes == 0 {
            return None;
        }
        (rows - 1)
            .checked_mul(self.time_stride_bytes)?
            .checked_add((heads - 1).checked_mul(self.head_stride_bytes)?)?
            .checked_add(cell_bytes)
    }
}

/// Byte layout of one read-only affine scale or bias plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnScalarLayout {
    pub(crate) dtype: Dtype,
    pub(crate) strides: AttnStrides,
    pub(crate) group_stride_bytes: usize,
}

impl AttnScalarLayout {
    pub(crate) fn span_bytes(self, rows: usize, heads: usize, groups: usize) -> Option<usize> {
        let scalar_bytes = match self.dtype {
            Dtype::F16 | Dtype::Bf16 | Dtype::F32 => {
                usize::try_from(self.dtype.byte_width()).ok()?
            }
            _ => return None,
        };
        if rows == 0 || heads == 0 || groups == 0 {
            return Some(0);
        }
        let cell_bytes = (groups - 1)
            .checked_mul(self.group_stride_bytes)?
            .checked_add(scalar_bytes)?;
        self.strides.span_bytes(rows, heads, cell_bytes)
    }
}

/// KV payload encoding, independent of query/output backing and affine scalar planes.
/// F16 and Q8 affine are implemented; other tags are refused before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttnKvStorage {
    F16,
    Bf16,
    F32,
    Q8Affine {
        group_size: usize,
        signed: bool,
    },
    Q4Affine {
        group_size: usize,
        low_nibble_first: bool,
    },
}

impl AttnKvStorage {
    fn payload_cell_bytes(self, dim: usize) -> Option<usize> {
        if dim == 0 {
            return None;
        }
        match self {
            Self::F16 => dim.checked_mul(usize::try_from(Dtype::F16.byte_width()).ok()?),
            Self::Bf16 => dim.checked_mul(usize::try_from(Dtype::Bf16.byte_width()).ok()?),
            Self::F32 => dim.checked_mul(usize::try_from(Dtype::F32.byte_width()).ok()?),
            Self::Q8Affine { group_size, .. } => {
                (group_size > 0 && dim.is_multiple_of(group_size)).then_some(dim)
            }
            Self::Q4Affine { group_size, .. } => {
                (group_size > 0 && dim.is_multiple_of(group_size)).then_some(dim.div_ceil(2))
            }
        }
    }
}

/// Read-only KV payload and independently laid out affine parameter planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnKvLayout {
    pub(crate) storage: AttnKvStorage,
    pub(crate) payload_strides: AttnStrides,
    pub(crate) scales: Option<AttnScalarLayout>,
    pub(crate) bias: Option<AttnScalarLayout>,
}

/// Separate extents preserve absent planes versus present, empty planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnKvExtents {
    pub(crate) payload_bytes: usize,
    pub(crate) scale_bytes: Option<usize>,
    pub(crate) bias_bytes: Option<usize>,
}

impl AttnKvLayout {
    /// Sizes only: launch admission must still prove origin alignment, ownership, range fit
    /// and writable non-overlap. Read-only plane broadcasts are permitted here.
    pub(crate) fn extents(
        self,
        rows: usize,
        heads: usize,
        dim: usize,
    ) -> Result<AttnKvExtents, EngError> {
        let cell_bytes = self
            .storage
            .payload_cell_bytes(dim)
            .ok_or_else(|| rule("layout", "invalid payload dimension or affine group size"))?;
        let payload_bytes = self
            .payload_strides
            .span_bytes(rows, heads, cell_bytes)
            .ok_or_else(|| rule("layout", "KV payload extent overflows"))?;
        let (scale_bytes, bias_bytes) = match self.storage {
            AttnKvStorage::Q8Affine { group_size, .. }
            | AttnKvStorage::Q4Affine { group_size, .. } => {
                // Payload sizing proved a positive, dividing group size.
                let groups = dim / group_size;
                let scalar_span = |plane: AttnScalarLayout| {
                    plane.span_bytes(rows, heads, groups).ok_or_else(|| {
                        rule(
                            "layout",
                            "invalid affine scalar dtype or overflowing extent",
                        )
                    })
                };
                let scales = self
                    .scales
                    .ok_or_else(|| rule("layout", "affine KV requires a scale plane layout"))?;
                (
                    Some(scalar_span(scales)?),
                    self.bias.map(scalar_span).transpose()?,
                )
            }
            AttnKvStorage::F16 | AttnKvStorage::Bf16 | AttnKvStorage::F32 => {
                if self.scales.is_some() || self.bias.is_some() {
                    return Err(rule(
                        "layout",
                        "float KV does not take affine parameter planes",
                    ));
                }
                (None, None)
            }
        };
        Ok(AttnKvExtents {
            payload_bytes,
            scale_bytes,
            bias_bytes,
        })
    }
}

/// Borrowed read-only KV planes; check again at submission before forming native arguments.
#[derive(Clone, Copy)]
pub(crate) struct AttnKvView<'b> {
    pub(crate) layout: AttnKvLayout,
    pub(crate) payload: DeviceRange<'b>,
    pub(crate) scales: Option<DeviceRange<'b>>,
    pub(crate) bias: Option<DeviceRange<'b>>,
}

impl AttnKvView<'_> {
    /// Physical reads only; the caller supplies admitted, nonempty write claims.
    /// Backend support and stream/completion admission stay with the caller.
    pub(in crate::kernels::cuda) fn check(
        &self,
        ctx: &CudaContext,
        shape: [usize; 3],
        writes: &[Claim<'_>],
    ) -> Result<AttnKvExtents, EngError> {
        let [rows, heads, dim] = shape;
        let extents = self.layout.extents(rows, heads, dim)?;
        let payload_align = match self.layout.storage {
            AttnKvStorage::F16 | AttnKvStorage::Bf16 => 2,
            AttnKvStorage::F32 => 4,
            AttnKvStorage::Q8Affine { .. } | AttnKvStorage::Q4Affine { .. } => 1,
        };
        let check_plane = |name: &'static str, range: &DeviceRange<'_>, need, align| {
            require_fit(name, range, need, if need == 0 { 1 } else { align })?;
            ctx.owned(range.buffer())?;
            if need != 0 {
                for write in writes {
                    require_apart(&Claim::of(name, range, need), write)?;
                }
            }
            Ok::<_, EngError>(())
        };
        if !self
            .layout
            .payload_strides
            .aligned(rows, heads, payload_align)
        {
            return Err(rule("kv_payload", "unaligned active payload stride"));
        }
        check_plane(
            "kv_payload",
            &self.payload,
            extents.payload_bytes,
            payload_align,
        )?;
        let groups = match self.layout.storage {
            AttnKvStorage::Q8Affine { group_size, .. }
            | AttnKvStorage::Q4Affine { group_size, .. } => dim / group_size,
            _ => 1,
        };
        let parameters = [
            (
                "kv_scales",
                self.layout.scales,
                self.scales,
                extents.scale_bytes,
            ),
            ("kv_bias", self.layout.bias, self.bias, extents.bias_bytes),
        ];
        for (name, layout, range, need) in parameters {
            match (layout, range, need) {
                (None, None, None) => {}
                (Some(layout), Some(range), Some(need)) => {
                    let align = usize::try_from(layout.dtype.byte_width())
                        .map_err(|_| rule(name, "invalid scalar alignment"))?;
                    if !layout.strides.aligned(rows, heads, align)
                        || (rows != 0
                            && heads != 0
                            && groups > 1
                            && !layout.group_stride_bytes.is_multiple_of(align))
                    {
                        return Err(rule(name, "unaligned active scalar stride"));
                    }
                    check_plane(name, &range, need, align)?;
                }
                _ => return Err(rule(name, "KV plane layout and range presence disagree")),
            }
        }
        Ok(extents)
    }
}

fn check_storage(storage: AttnKvStorage, dim: usize) -> Result<(), EngError> {
    if storage == AttnKvStorage::F16 {
        return Ok(());
    }
    if storage.payload_cell_bytes(dim).is_none() {
        return Err(rule(
            "storage",
            "invalid KV payload extent or affine group size",
        ));
    }
    if matches!(storage, AttnKvStorage::Q8Affine { .. }) && [64, 128, 256].contains(&dim) {
        return Ok(());
    }
    Err(rule(
        "storage",
        &format!("{storage:?} at D{dim} is not implemented by this CUDA entry"),
    ))
}

/// Geometry of one prefill attention call: `m` tokens at the positions `p0 .. p0 + m`,
/// `n_head` query heads over `n_kv_head` kv heads (GQA1/8 initially), the head size, the
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

/// One absolute key segment occupying one slot of the shared partial buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnPrefillSegment {
    pub(crate) key_origin: usize,
    pub(crate) key_rows: usize,
    pub(crate) part_index: usize,
}

/// One prefill attention call. All ranges are device ranges of this context.
///
/// - `q`: f32 `[m][n_head][head_dim]`, 16-byte aligned, after RMSNorm and rope, not scaled.
/// - `proj`: none (no output gate) or f32 `[m][ld]` with `ld >= n_head * 2 * head_dim`; head
///   `h`'s gate is at column `h * 2 * head_dim + head_dim + d`.
/// - `sinks`: optional published f32 `[n_head]`, 4-byte aligned, raw and unscaled;
///   contributes denominator mass once after partition merge, with no value vector.
/// - `k_cache`, `v_cache`: legacy F16 planes, or the corresponding Q8 view payload ranges.
/// - `kv_views`: both Q8 planes and affine layouts; absent for the legacy F16 path.
/// - `part`: f32 scratch of [`cpf_attn_part_bytes`] bytes of the call's split (the launch
///   plan's, [`cpf_attn_launch_plan`]; the driver allocates the flat [`CPF_ATTN_PART_CAP`]),
///   16-byte aligned, no initialization needed; only the active splits are written.
/// - `out`: f32 `[m][n_head * head_dim]`, 16-byte aligned; NaN on device bounds failure.
/// - `status`: `None`, or one device `i32`; device bounds failure ORs mask 8.
///   Host-side refusals change no device bytes; valid launches leave status untouched.
/// - `q_tile`: the compiled tile; only 8 runs (lane C ruling 5).
/// - `scale`: multiplies the scores, finite and in `(0, 1e30)`; `1/sqrt(256) = 0.0625`.
pub(crate) struct AttnPrefillCall<'b> {
    pub(crate) storage: AttnKvStorage,
    pub(crate) shape: AttnPrefillShape,
    pub(crate) span: CpfAttnSpan,
    pub(crate) split: CpfAttnSplit,
    pub(crate) q_tile: usize,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    pub(crate) kv_views: Option<[AttnKvView<'b>; 2]>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
    pub(crate) sinks: Option<DeviceRange<'b>>,
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
    check_shape_mode(shape, q_tile, false, AttnKvStorage::F16)
}

fn check_full_shape(
    shape: &AttnPrefillShape,
    q_tile: usize,
    span: &CpfAttnSpan,
    storage: AttnKvStorage,
) -> Result<crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillInts, EngError> {
    check_shape_mode(shape, q_tile, span.ring > 0, storage)
}

fn check_shape_mode(
    shape: &AttnPrefillShape,
    q_tile: usize,
    absolute_positions: bool,
    storage: AttnKvStorage,
) -> Result<crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillInts, EngError> {
    use crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillInts;
    if q_tile != CPF_ATTN_Q_TILE {
        return Err(rule(
            "q_tile",
            &format!("q_tile is {q_tile}, not the compiled tile {CPF_ATTN_Q_TILE}"),
        ));
    }
    let q8 = matches!(storage, AttnKvStorage::Q8Affine { .. });
    if ![64, 128, HEAD_DIM].contains(&shape.head_dim) || (!q8 && shape.head_dim == 64) {
        return Err(rule(
            "head_dim",
            &format!("head_dim {} is unsupported for {storage:?}", shape.head_dim),
        ));
    }
    let n_kv_head = shape.n_kv_head;
    if !(1..=MAX_KV_HEADS).contains(&n_kv_head)
        || !(1..=MAX_TOKENS).contains(&shape.n_head)
        || !shape.n_head.is_multiple_of(n_kv_head)
        || (!q8 && !(shape.n_head == n_kv_head || shape.n_head == GROUP * n_kv_head))
    {
        return Err(rule(
            "group",
            &format!(
                "n_head {} over n_kv_head {} is unsupported for the selected storage (1 to 64 kv heads)",
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
    let limit = if absolute_positions {
        MAX_ROWS
    } else {
        shape.cache_rows
    };
    if span > i64::try_from(limit).map_err(|_| rule("cache_rows", "too large"))? {
        return Err(rule(
            "p0",
            &format!("p0 + m is {span}, above the {} cache rows", limit),
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

fn check_segment_shape(
    shape: &AttnPrefillShape,
    q_tile: usize,
    split: &CpfAttnSplit,
    storage: AttnKvStorage,
) -> Result<crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillInts, EngError> {
    let ints = check_shape_mode(shape, q_tile, true, storage)?;
    let bytes = cpf_attn_part_bytes(geometry(shape), plan_of(split))?;
    if bytes > CPF_ATTN_PART_CAP {
        return Err(EngError::MemoryPlan(format!(
            "attn_prefill: segment partials need {bytes} bytes, above the {CPF_ATTN_PART_CAP} byte budget"
        )));
    }
    Ok(ints)
}

fn check_segment(
    shape: &AttnPrefillShape,
    split: &CpfAttnSplit,
    segment: AttnPrefillSegment,
    storage: AttnKvStorage,
) -> Result<AttnPrefillLaunch, EngError> {
    let ints = check_segment_shape(shape, CPF_ATTN_Q_TILE, split, storage)?;
    let end = ints.p0 as usize + ints.m as usize;
    if segment.key_rows == 0
        || segment.key_rows > shape.cache_rows
        || segment
            .key_origin
            .checked_add(segment.key_rows)
            .is_none_or(|hi| hi > end)
        || segment.part_index >= split.max_splits
    {
        return Err(rule(
            "segment",
            "key extent or partial slot is out of bounds",
        ));
    }
    Ok(AttnPrefillLaunch::Part {
        key_origin: to_int("key_origin", segment.key_origin)?,
        key_rows: to_int("key_rows", segment.key_rows)?,
        part_index: to_int("part_index", segment.part_index)?,
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
        if span
            .window
            .checked_add(tokens_after)
            .is_none_or(|need| span.ring < need)
        {
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
    sinks: usize,
    cache: [usize; 2],
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
    let cache = match &call.kv_views {
        Some(views) => {
            let extent = |view: &AttnKvView<'_>| {
                view.layout
                    .extents(shape.cache_rows, shape.n_kv_head, shape.head_dim)
                    .map(|e| e.payload_bytes)
            };
            [extent(&views[0])?, extent(&views[1])?]
        }
        None => {
            [shape
                .cache_rows
                .saturating_mul(shape.n_kv_head)
                .saturating_mul(shape.head_dim)
                .saturating_mul(2); 2]
        }
    };
    Ok(Needs {
        q: shape.m.saturating_mul(heads).saturating_mul(4),
        proj,
        sinks: shape.n_head.saturating_mul(4),
        cache,
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
    if let Some(sinks) = &call.sinks {
        require_fit("sinks", sinks, need.sinks, 4)?;
    }
    let cache_align = if call.kv_views.is_some() { 1 } else { 16 };
    require_fit("k_cache", &call.k_cache, need.cache[0], cache_align)?;
    require_fit("v_cache", &call.v_cache, need.cache[1], cache_align)?;
    require_fit("part", &call.part, need.part, 16)?;
    require_fit("out", &call.out, need.out, 16)?;
    if let Some(status) = &call.status {
        require_fit("status", status, 4, 4)?;
    }
    Ok(())
}

/// Disjointness: `out` and `part` are written, so each is apart from every other range of the
/// call; the inputs may overlap each other, but a status write must not corrupt sinks.
fn check_apart(call: &AttnPrefillCall<'_>, need: &Needs) -> Result<(), EngError> {
    let out = Claim::of("out", &call.out, need.out);
    let part = Claim::of("part", &call.part, need.part);
    let reads = [
        Some(Claim::of("q", &call.q, need.q)),
        Some(Claim::of("k_cache", &call.k_cache, need.cache[0])),
        Some(Claim::of("v_cache", &call.v_cache, need.cache[1])),
        call.proj
            .as_ref()
            .map(|proj| Claim::of("proj", proj, need.proj)),
        call.sinks
            .as_ref()
            .map(|sinks| Claim::of("sinks", sinks, need.sinks)),
    ];
    for read in reads.iter().flatten() {
        require_apart(&out, read)?;
    }
    require_apart(&out, &part)?;
    for read in reads.iter().flatten() {
        require_apart(&part, read)?;
    }
    if let Some(status) = &call.status {
        let status = Claim::of("status", status, 4);
        require_apart(&status, &out)?;
        require_apart(&status, &part)?;
        for read in reads.iter().flatten() {
            require_apart(&status, read)?;
        }
    }
    Ok(())
}

fn check_kv_views(
    ctx: &CudaContext,
    call: &AttnPrefillCall<'_>,
    need: &Needs,
) -> Result<(), EngError> {
    let Some(views) = &call.kv_views else {
        return if call.storage == AttnKvStorage::F16 {
            Ok(())
        } else {
            Err(rule("kv_views", "Q8 storage requires both plane views"))
        };
    };
    if !matches!(call.storage, AttnKvStorage::Q8Affine { .. }) {
        return Err(rule("kv_views", "tagged views require Q8 storage"));
    }
    let mut writes = vec![
        Claim::of("out", &call.out, need.out),
        Claim::of("part", &call.part, need.part),
    ];
    if let Some(status) = &call.status {
        writes.push(Claim::of("status", status, 4));
    }
    for (view, cache) in views.iter().zip([&call.k_cache, &call.v_cache]) {
        if view.layout.storage != call.storage
            || !std::ptr::eq(view.payload.buffer(), cache.buffer())
            || view.payload.offset() != cache.offset()
            || view.payload.bytes() != cache.bytes()
        {
            return Err(rule(
                "kv_views",
                "selected storage and cache payload range must match each view",
            ));
        }
        view.check(
            ctx,
            [
                call.shape.cache_rows,
                call.shape.n_kv_head,
                call.shape.head_dim,
            ],
            &writes,
        )?;
    }
    Ok(())
}

fn raw_kv<'b>(ctx: &CudaContext, view: &AttnKvView<'b>) -> Result<AttnKvRaw<'b>, EngError> {
    Ok(AttnKvRaw {
        layout: view.layout,
        payload: ctx.raw_range(&view.payload)?,
        scales: view.scales.as_ref().map(|r| ctx.raw_range(r)).transpose()?,
        bias: view.bias.as_ref().map(|r| ctx.raw_range(r)).transpose()?,
    })
}

impl CudaContext {
    /// Queues the chunked-prefill attention of `call` on `stream`: the part kernel (one
    /// partial per query row per active split; on sm_80 and later the head_dim-quarter
    /// kernel, 512 threads, one block per SM), then the combine (fixed order, the sigmoid
    /// output gate when `proj` is given). Returns when both are queued; the output is valid
    /// after the stream drained. The split is the caller's: the driver passes the launch
    /// plan's ([`cpf_attn_launch_plan`]).
    ///
    /// Positions require `p0 >= 0` and `p0 + m <= MAX_ROWS`; linear calls additionally need
    /// `p0 + m <= cache_rows`. Rust and C refuse bad bounds, also re-checked on the device
    /// (no partial, NaN `out`, mask 8 of the optional status word). A chunk whose partials
    /// exceed 64 MiB at its own split count is a `MemoryPlan` refusal: shrink the chunk.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a tile, shape, span, split or scale the kernel
    /// refuses, a `p0 + m` beyond its applicable bound, a short or misaligned range, and
    /// an `out` or `part` range that overlaps another range of the call.
    pub(crate) fn attn_prefill(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
    ) -> Result<(), EngError> {
        self.attn_prefill_enqueue(stream, call, AttnPrefillLaunch::Full)
    }

    /// Explicit sink-aware decode correctness path; default resident dispatch is unchanged.
    /// Uses host p0, not the resident device kv_len. The raw per-head sink is added once.
    pub(crate) fn attn_decode_sinks_via_prefill(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
    ) -> Result<(), EngError> {
        if call.shape.m != 1 || call.sinks.is_none() {
            return Err(rule("decode sinks", "requires m=1 and raw per-head sinks"));
        }
        self.attn_prefill(stream, call)
    }

    /// Queues one staged key segment into its slot; p0 remains absolute.
    /// All slots, including neutral rows, must be produced before the one combine.
    pub(crate) fn attn_prefill_part(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
        segment: AttnPrefillSegment,
    ) -> Result<(), EngError> {
        let launch = check_segment(&call.shape, &call.split, segment, call.storage)?;
        self.attn_prefill_enqueue(stream, call, launch)
    }

    /// Combines every initialized segment slot on the same stream and gates once.
    pub(crate) fn attn_prefill_combine(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
    ) -> Result<(), EngError> {
        self.attn_prefill_enqueue(stream, call, AttnPrefillLaunch::Combine)
    }

    /// Same-ELF fixture path (A ruling 01a18101): identical admission and args, but the
    /// native entry launches the q8 part kernel with the pre-hoist per-element math so
    /// tests can compare it with the hoisted kernel byte for byte. Fixture builds only.
    #[cfg(all(test, feature = "cuda-q8-reference"))]
    pub(crate) fn attn_prefill_q8_reference(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
    ) -> Result<(), EngError> {
        self.attn_prefill_enqueue(stream, call, AttnPrefillLaunch::Q8Reference)
    }

    /// Same-ELF fixture path (MF1, Opus bee8b4539): identical admission and args, but the
    /// native entry passes the tiled opt-in — the only way to reach the step-2 kernel.
    /// The default launch keeps the old kernel in every build. Fixture builds only.
    #[cfg(all(test, feature = "cuda-q8-reference"))]
    pub(crate) fn attn_prefill_q8_tiled(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
    ) -> Result<(), EngError> {
        self.attn_prefill_enqueue(stream, call, AttnPrefillLaunch::Q8Tiled)
    }

    fn attn_prefill_enqueue(
        &self,
        stream: &DeviceStream,
        call: AttnPrefillCall<'_>,
        launch: AttnPrefillLaunch,
    ) -> Result<(), EngError> {
        check_storage(call.storage, call.shape.head_dim)?;
        match (call.storage, call.kv_views.is_some()) {
            (AttnKvStorage::F16, false) | (AttnKvStorage::Q8Affine { .. }, true) => {}
            _ => {
                return Err(rule(
                    "kv_views",
                    "F16 requires no views; Q8 requires both views",
                ));
            }
        }
        self.ensure_usable()?;
        #[cfg(all(test, feature = "cuda-q8-reference"))]
        let full = matches!(
            launch,
            AttnPrefillLaunch::Full | AttnPrefillLaunch::Q8Reference | AttnPrefillLaunch::Q8Tiled
        );
        #[cfg(not(all(test, feature = "cuda-q8-reference")))]
        let full = matches!(launch, AttnPrefillLaunch::Full);
        let mut ints = if full {
            check_full_shape(&call.shape, call.q_tile, &call.span, call.storage)?
        } else {
            if call.span.window != 0 || call.span.ring != 0 {
                return Err(rule("segment", "staging requires zero window and ring"));
            }
            check_segment_shape(&call.shape, call.q_tile, &call.split, call.storage)?
        };
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
        check_kv_views(self, &call, &need)?;
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
            sinks,
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
        let sinks = sinks.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let kv_views = match call.kv_views.as_ref() {
            Some([k, v]) => Some([raw_kv(self, k)?, raw_kv(self, v)?]),
            None => None,
        };
        let scale = *scale;
        self.launch_on(stream, move |raw| {
            attn_prefill_launch(
                raw,
                AttnPrefillRaw {
                    q,
                    proj,
                    k_cache,
                    v_cache,
                    kv_views,
                    part,
                    out,
                    status,
                    sinks,
                    ints,
                    scale,
                },
                launch,
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

#[cfg(test)]
#[path = "attn_prefill/view_tests.rs"]
mod view_tests;

#[cfg(test)]
#[path = "attn_prefill/affine_view_tests.rs"]
mod affine_view_tests;

#[cfg(test)]
#[path = "attn_prefill/storage_tests.rs"]
mod storage_tests;

#[cfg(all(test, feature = "native-cuda"))]
#[path = "attn_prefill/kv_view_tests.rs"]
mod kv_view_tests;
