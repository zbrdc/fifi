//! Safe wrappers of the mmid entries of the MMQ glue: the grouped (per-expert) GEMM and the
//! id tensors it reads. Each call queues work on a [`DeviceStream`] and returns; none
//! allocates, copies or synchronizes:
//!
//! - [`CudaContext::mmq_ids_guard`], the ids guard as its own entry: the router ids into a
//!   guarded copy, every flagged row (an id outside `0..n_experts`, or two equal ids)
//!   rewritten to `0..top_k`, mask 1 of the optional prefill status word for a row whose
//!   every slot is -1 (the device router's fault form, `route_experts`) and mask 8 for any
//!   other flagged row (`prefill_chunk`). The tiered arm runs it on the router ids before its
//!   D2H copy, so the tiered planner's check sees the flags.
//! - [`CudaContext::mmq_build_ids`]: the same guard, always, then the vendored `mm_ids_helper`
//!   on the guarded copy. It returns the [`MmidIds`] of the call.
//! - [`CudaContext::mmq_mmid_launch`]: one grouped GEMM over a weight stack, a `NativeWeight`
//!   or a bank of slots, through the vendored `mul_mat_q_case<Q4_K|Q6_K>`, with its ids taken
//!   only as an [`MmidIds`]. The `mmq_args` fill is the pure [`fill_mmid_args`], the pin's
//!   grouped assignment (mmq.cu:255-261), and the launch validates the padding itself (the
//!   q8_1 pad of `y` and the fixup-gather pad of `ids_dst`, both sized by `mmq_size`), the K
//!   grid and the stream-k grid limit, so ids over a slot bank get the same guards.
//!
//! - [`CudaContext::mmq_quantize_scatter_ids`] and [`CudaContext::mmq_quantize_gather_ids`]: the
//!   thin typed wrappers over the quantize entries of `mmq_quant`. They take the scatter map
//!   (`ids_src1`) and the gather map (`ids_dst`) from an [`MmidIds`] and from nowhere else, so
//!   a store index of the quantize kernels is never caller-supplied; the raw-range quantize
//!   wrappers are visible to the prefill tree only.
//!
//! [`MmidIds`] is constructor-only: its fields are private, and the builder and the checked
//! host constructor are the only ways to make one, so a grouped launch never trusts
//! caller-built id values. It carries `ids_src1`, the inverse map the scatter quantizer stores
//! through. The type proves which code wrote the ids; it does not prove that the bytes are
//! intact when a launch reads them, which rests on stream order and on the arena plan.
//!
//! Every argument is checked here, before any native call, and refused as `BadRequest`: the
//! rules mirror the C entries' (`mmq_mmid.h`), so a native entry cannot return
//! `INVALID_ARGUMENT`, which would latch the backend. Local rules carry the tag
//! `mmq_mmid: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the chunked-prefill driver")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::mmq_mmid::{
    BuildIdsRaw, IdsGuardRaw, IdsShapeInts, MmidGeometry, MmidLaunchRaw, mmq_build_ids_launch,
    mmq_ids_guard_launch, mmq_mmid_launch,
};
use crate::kernels::cuda::prefill::mmq_quant::{
    K_GRID, K_MAX, MmqQuantizeCall, MmqQuantizeScatterCall,
};
use crate::kernels::cuda::prefill::mmq_size::{
    MmqType, NCOLS_LIMIT, expert_bounds_ints, fixup_floats_max, ids_dst_ints, ids_src1_ints,
    q8_1_bytes,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// The most experts: the z extent of the MMQ tiling grid, where the expert is the channel
/// (`block_nums_xy_tiling`, mmq.cuh:1368).
pub(crate) const MAX_EXPERTS: usize = 65_535;
/// The most ids per token: the widest compiled case of `mm_ids_helper` (mmid.cu:171).
pub(crate) const MAX_TOP_K: usize = 32;
/// The stream-k limit, `ntiles * blocks_per_ne00 < 2^30` (mmq.cuh:1402).
const GRID_LIMIT: usize = 1 << 30;
/// The tile count at the smallest J the launcher picks: mmq.cuh:1399 computes `100 * ntiles_dst` in
/// `int` for the tiling decision, so the tile count must stay at or below `i32::MAX / 100`; J = 8
/// gives the largest count, which the mirror below bounds for every J.
const TILES_LIMIT: usize = 0x7fff_ffff / 100;
/// The smallest tile width the launch can pick (it picks J itself) and the row tile of every
/// kept config row.
const J_MIN: usize = 8;
const I_TILE: usize = 128;
/// `i32::MAX`: the kernels compute the weight and dst indexes in `int`.
const INT_MAX: usize = 0x7fff_ffff;

/// The shape every id tensor of one mmid call shares: `m` tokens of `top_k` ids over
/// `n_experts` experts. The rules: `1..=65_535` experts, `1..=2_048` tokens (the chunk cap,
/// `mmq_size::NCOLS_LIMIT`), `1..=min(n_experts, 32)` ids per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MmidShape {
    pub(crate) n_experts: usize,
    pub(crate) m: usize,
    pub(crate) top_k: usize,
}

impl MmidShape {
    /// The compact columns, `m * top_k` (upstream `ne_get_rows`); at most 65,536 under the
    /// shape rules.
    pub(crate) fn ne_get_rows(&self) -> usize {
        self.m.saturating_mul(self.top_k)
    }
}

/// One guard call. All ranges are device ranges of this context.
///
/// - `ids`: the router ids, i32 `[m][top_k]` contiguous, 4-byte aligned; read only.
/// - `ids_g`: the guarded copy, i32 `[m][top_k]`, 4-byte aligned.
/// - `status`: `None`, or one device `i32`, the prefill status word (never
///   `StepParams.status`): ORed with mask 1 or 8 as above, untouched when no row is flagged.
pub(crate) struct IdsGuardCall<'b> {
    pub(crate) shape: MmidShape,
    pub(crate) ids: DeviceRange<'b>,
    pub(crate) ids_g: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

/// The ranges an [`MmidIds`] holds, written by its constructor; i32, 4-byte aligned, apart.
///
/// - `ids_src1`: `m * top_k` ints, the inverse map: token slot `t * top_k + r` to its compact
///   column.
/// - `ids_dst`: at least `m * top_k + m + 127` ints (the stream-k fixup gather pad,
///   mmq.cuh:1295-1302); the first `m * top_k` map compact column to token slot.
/// - `expert_bounds`: `n_experts + 1` ints; expert `e` owns the compact columns
///   `expert_bounds[e]..expert_bounds[e + 1]`.
pub(crate) struct MmidIdsOut<'b> {
    pub(crate) ids_src1: DeviceRange<'b>,
    pub(crate) ids_dst: DeviceRange<'b>,
    pub(crate) expert_bounds: DeviceRange<'b>,
}

/// The ids of one grouped launch, made only by [`CudaContext::mmq_build_ids`] or the checked
/// host constructor. Its values meet the id invariants: `ids_dst` is a permutation of
/// `0..m * top_k`, `ids_src1` its inverse, the bounds rise from 0 to `m * top_k`, and no expert
/// holds more than `m` columns.
pub(crate) struct MmidIds<'b> {
    out: MmidIdsOut<'b>,
    shape: MmidShape,
}

/// Caller-built ids on the host (the tiered planner's): `ids_dst`, compact column to token
/// slot, and `expert_bounds`, in the layout of [`MmidIdsOut`].
pub(crate) struct HostIds<'h> {
    pub(crate) ids_dst: &'h [i32],
    pub(crate) expert_bounds: &'h [i32],
}

impl<'b> MmidIds<'b> {
    /// The checked host constructor: checks every value invariant of `host` ([`check_host_ids`])
    /// and the `out` ranges before any device write, derives the inverse map
    /// `ids_src1[ids_dst[c]] = c` itself (a caller never supplies it: the scatter quantizer stores
    /// through it), then uploads `ids_dst`, `ids_src1` and `expert_bounds` with ordered `write_at`
    /// on `stream`, each returning when its bytes landed. Only the first `m * top_k` ints of
    /// `ids_dst` are written.
    ///
    /// Returns `BadRequest`, before any device write, for a shape outside the rules of
    /// [`MmidShape`], a value that breaks them (tag `mmq_mmid: ids values`), and `out` ranges that
    /// are too short, misaligned or overlapping; `write_at` refuses a stream or buffer of
    /// another context, also before its copy.
    pub(crate) fn from_host(
        stream: &DeviceStream,
        shape: MmidShape,
        host: HostIds<'_>,
        out: MmidIdsOut<'b>,
    ) -> Result<Self, EngError> {
        check_ids_shape(&shape)?;
        let inverse = check_host_ids(&shape, &host)?;
        require_written_apart(&out_claims(&out, &shape)?, &[])?;
        write_ints(stream, &out.ids_dst, host.ids_dst)?;
        write_ints(stream, &out.ids_src1, &inverse)?;
        write_ints(stream, &out.expert_bounds, host.expert_bounds)?;
        Ok(Self { out, shape })
    }
}

impl MmidIds<'_> {
    /// The shape the ids were made for.
    pub(crate) fn shape(&self) -> MmidShape {
        self.shape
    }
}

/// The `MmqType` of the `mmq_quant` entries for a weight `QuantType` (the one mapping between
/// the launch-side weight type and the quantize-side type; the typed scatter and gather
/// wrappers call the quantize entries through it). The four glued types map one for one, so
/// the D4/DS4 scale layout of every q8_1 buffer follows its weight (mmq.cuh:61-101).
pub(crate) fn mmq_type_of(quant: QuantType) -> MmqType {
    match quant {
        QuantType::Q80 => MmqType::Q80,
        QuantType::Q50 => MmqType::Q50,
        QuantType::Q4K => MmqType::Q4K,
        QuantType::Q6K => MmqType::Q6K,
    }
}

/// The activations of one typed quantize call. `x` holds f32 rows of `row_stride >= k` floats
/// (the token rows of the gate/up input for the scatter, the `m * top_k` rows of the gate/up
/// output in token-slot order for the gather), `y` the q8_1 columns the quantize entry writes
/// (`q8_1_bytes(k, m * top_k)` bytes, the constant q8_1 pad included). The scale layout
/// follows `quant`, the type of the weight that will read `y`.
pub(crate) struct MmidQuantCall<'b> {
    pub(crate) packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking,
    pub(crate) quant: QuantType,
    pub(crate) k: usize,
    pub(crate) row_stride: usize,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

/// The `ids values` refusal of [`check_host_ids`].
fn values(detail: &str) -> EngError {
    rule("ids values", detail)
}

/// The value invariants of caller-built ids, checked on the host: `expert_bounds` holds
/// `n_experts + 1` values rising from 0 to `m * top_k`, with at most `m` columns per expert (the
/// grouped launch's grid computes `ceil(m / J)` tiles of each expert, so a column past them would
/// never be written); `ids_dst` holds `m * top_k` entries below `m * top_k`, each exactly once.
/// Returns the inverse map those entries imply, after [`check_ids_shape`].
pub(crate) fn check_host_ids(shape: &MmidShape, host: &HostIds<'_>) -> Result<Vec<i32>, EngError> {
    let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
    let (n_dst, n_bounds) = (host.ids_dst.len(), host.expert_bounds.len());
    if n_dst != ne || n_bounds != e + 1 {
        return Err(values(&format!(
            "ids_dst holds {n_dst} entries and expert_bounds {n_bounds}, not {ne} and {}",
            e + 1
        )));
    }
    check_host_bounds(shape, host.expert_bounds)?;
    let mut inverse = vec![-1; ne];
    for (c, (&slot, column)) in host.ids_dst.iter().zip(0..).enumerate() {
        let index = usize::try_from(slot)
            .ok()
            .filter(|&index| index < ne)
            .ok_or_else(|| values(&format!("ids_dst[{c}] is {slot}, outside 0..{ne}")))?;
        if inverse[index] != -1 {
            return Err(values(&format!(
                "ids_dst[{c}] repeats the token slot {slot}"
            )));
        }
        inverse[index] = column;
    }
    Ok(inverse)
}

/// The bounds half of [`check_host_ids`]: from 0 to `m * top_k`, never falling, at most `m`
/// columns per expert.
fn check_host_bounds(shape: &MmidShape, bounds: &[i32]) -> Result<(), EngError> {
    let ne = shape.ne_get_rows();
    let (first, last) = (bounds[0], bounds[shape.n_experts]);
    if first != 0 || usize::try_from(last).ok() != Some(ne) {
        return Err(values(&format!(
            "expert_bounds runs from {first} to {last}, not from 0 to {ne}"
        )));
    }
    for (expert, pair) in bounds.windows(2).enumerate() {
        let count = i64::from(pair[1]) - i64::from(pair[0]);
        if count < 0 {
            let (from, to) = (pair[0], pair[1]);
            return Err(values(&format!(
                "expert_bounds falls from {from} to {to} at expert {expert}"
            )));
        }
        if usize::try_from(count).is_ok_and(|count| count > shape.m) {
            return Err(values(&format!(
                "expert {expert} holds {count} columns, above m = {}",
                shape.m
            )));
        }
    }
    Ok(())
}

/// `values` as little-endian bytes at the start of `range`, ordered on `stream`; the caller
/// proved that `range` holds them.
fn write_ints(
    stream: &DeviceStream,
    range: &DeviceRange<'_>,
    values: &[i32],
) -> Result<(), EngError> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    range.buffer().write_at(stream, range.offset(), &bytes)
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("mmq_mmid: {tag}: {detail}"))
}

/// `value` as `c_int` when it lies in `low..=high`, else the refusal for `tag`.
fn ranged(tag: &str, value: usize, low: usize, high: usize) -> Result<c_int, EngError> {
    if !(low..=high).contains(&value) {
        return Err(rule(tag, &format!("{tag} is {value}, not {low} to {high}")));
    }
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the shape rules every mmid entry shares and converts them once, so no conversion
/// can fail inside a launch closure.
pub(in crate::kernels::cuda) fn check_ids_shape(
    shape: &MmidShape,
) -> Result<IdsShapeInts, EngError> {
    let n_experts = ranged("n_experts", shape.n_experts, 1, MAX_EXPERTS)?;
    let m = ranged("m", shape.m, 1, NCOLS_LIMIT)?;
    let top_k = ranged("top_k", shape.top_k, 1, MAX_TOP_K.min(shape.n_experts))?;
    Ok(IdsShapeInts {
        n_experts,
        m,
        top_k,
    })
}

/// Bytes of the `[m][top_k]` i32 rows, after [`check_ids_shape`].
fn rows_bytes(shape: &MmidShape) -> usize {
    shape.ne_get_rows() * 4
}

/// `BadRequest` unless every written claim is apart from every other claim of the call.
fn require_written_apart(written: &[Claim<'_>], read: &[Claim<'_>]) -> Result<(), EngError> {
    for (i, claim) in written.iter().enumerate() {
        for other in written[i + 1..].iter().chain(read) {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

/// The sizes and alignment of a guard call, then its written claims (`ids_g`, the status
/// word) and its read claim (`ids`).
fn guard_claims<'a>(call: &IdsGuardCall<'a>) -> Result<(Vec<Claim<'a>>, Claim<'a>), EngError> {
    let rows = rows_bytes(&call.shape);
    require_fit("ids", &call.ids, rows, 4)?;
    require_fit("ids_g", &call.ids_g, rows, 4)?;
    let mut written = vec![Claim::of("ids_g", &call.ids_g, rows)];
    if let Some(status) = &call.status {
        require_fit("status", status, 4, 4)?;
        written.push(Claim::of("status", status, 4));
    }
    Ok((written, Claim::of("ids", &call.ids, rows)))
}

/// The sizes and alignment of the ranges of an [`MmidIds`], then their claims.
fn out_claims<'a>(out: &MmidIdsOut<'a>, shape: &MmidShape) -> Result<[Claim<'a>; 3], EngError> {
    let ne = shape.ne_get_rows();
    let src1 = ids_src1_ints(ne) * 4;
    let dst = ids_dst_ints(ne, shape.m)? * 4;
    let bounds = expert_bounds_ints(shape.n_experts)? * 4;
    require_fit("ids_src1", &out.ids_src1, src1, 4)?;
    require_fit("ids_dst", &out.ids_dst, dst, 4)?;
    require_fit("expert_bounds", &out.expert_bounds, bounds, 4)?;
    Ok([
        Claim::of("ids_src1", &out.ids_src1, src1),
        Claim::of("ids_dst", &out.ids_dst, dst),
        Claim::of("expert_bounds", &out.expert_bounds, bounds),
    ])
}

/// A weight stack: `n_experts` matrices of `n` rows by `k` values in the native GGUF blocks of
/// `quant`, the first at the start of `base`, each `stride_bytes` after the one before (a
/// `NativeWeight`'s buffer, or a bank of slots with `n_experts = slots`). `base` is 16-byte
/// aligned.
pub(crate) struct MmidWeight<'b> {
    pub(crate) base: DeviceRange<'b>,
    pub(crate) quant: QuantType,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) stride_bytes: usize,
}

/// One grouped launch. All ranges are device ranges of this context.
///
/// - `ids`: an [`MmidIds`] of exactly `shape`.
/// - `y`: the q8_1 compact columns of the gather quantize, `q8_1_bytes(k, m * top_k)` bytes
///   (the constant pad included), 16-byte aligned; read only.
/// - `dst`: f32 rows of `ld >= n` floats, `(m * top_k - 1) * ld + n` floats, 4-byte aligned:
///   row `t * top_k + r` receives pick `r` of token `t` (mmq.cuh:470, mmid.cu:110).
/// - `fixup`: f32 scratch of the stream-k fixup, `fixup_floats_max(nsm)` floats with `nsm` from
///   the device facts, 16-byte aligned.
pub(crate) struct MmidLaunchCall<'b, 'i> {
    pub(crate) shape: MmidShape,
    pub(crate) weight: MmidWeight<'b>,
    pub(crate) ids: &'i MmidIds<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) dst: DeviceRange<'b>,
    pub(crate) ld: usize,
    pub(crate) fixup: DeviceRange<'b>,
}

/// The numbers a grouped launch is checked and filled from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MmidLaunchShape {
    pub(crate) ids: MmidShape,
    pub(crate) quant: QuantType,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) stride_bytes: usize,
    pub(crate) ld: usize,
}

impl MmidLaunchShape {
    /// The weight blocks of one row, `k / 256` for both glued types.
    fn row_blocks(&self) -> usize {
        self.k / self.quant.block_size()
    }

    /// The per-expert stride in weight blocks (upstream `s02`).
    fn stride_blocks(&self) -> usize {
        self.stride_bytes / self.quant.block_bytes()
    }

    /// The weight blocks the launch reads, after [`check_launch_shape`].
    fn weight_blocks(&self) -> usize {
        (self.ids.n_experts - 1) * self.stride_blocks() + self.n * self.row_blocks()
    }

    /// The dst floats the launch writes, after [`check_launch_shape`].
    fn dst_floats(&self) -> usize {
        (self.ids.ne_get_rows() - 1) * self.ld + self.n
    }
}

/// Checks every number rule of a grouped launch (`mmq_mmid.h`) in this order: the id shape, the
/// type (Q4_K and Q6_K; a grouped Q8_0 or Q5_0 has no arm), the K grid (before any per-type rule),
/// K's range, N, ld, the stream-k grid limit bounded with J = 8, the stride (whole blocks, at least
/// one matrix, an `int`), and the `int` weight and dst indexes. Each bound keeps every product
/// below 2^56, so the sums cannot overflow.
pub(crate) fn check_launch_shape(s: &MmidLaunchShape) -> Result<(), EngError> {
    check_ids_shape(&s.ids)?;
    if !matches!(s.quant, QuantType::Q4K | QuantType::Q6K) {
        let detail = format!("a grouped {:?} launch has no glued arm", s.quant);
        return Err(rule("type", &detail));
    }
    if !s.k.is_multiple_of(K_GRID) {
        let detail = format!("K is {}, not a multiple of {K_GRID}", s.k);
        return Err(rule("K-grid", &detail));
    }
    ranged("K", s.k, K_GRID, K_MAX)?;
    ranged("N", s.n, 1, INT_MAX)?;
    ranged("ld", s.ld, s.n, INT_MAX)?;
    let tiles = s.ids.m.div_ceil(J_MIN) * s.n.div_ceil(I_TILE) * s.ids.n_experts;
    if tiles > TILES_LIMIT {
        let detail = format!(
            "{tiles} tiles at the smallest J exceed i32::MAX / 100 when the launcher scales \
             them for its tiling decision (mmq.cuh:1399)"
        );
        return Err(rule("grid limit", &detail));
    }
    if tiles * s.row_blocks() >= GRID_LIMIT {
        let detail = format!("{tiles} tiles of {} blocks reach 2^30", s.row_blocks());
        return Err(rule("grid limit", &detail));
    }
    let whole = s.stride_bytes.is_multiple_of(s.quant.block_bytes());
    if !whole || s.stride_blocks() < s.n * s.row_blocks() || s.stride_blocks() > INT_MAX {
        let detail = format!("{} bytes is not a stride of whole matrices", s.stride_bytes);
        return Err(rule("stride", &detail));
    }
    if s.weight_blocks() > INT_MAX {
        let detail = format!("{} weight blocks exceed i32::MAX", s.weight_blocks());
        return Err(rule("weight index", &detail));
    }
    if s.dst_floats() > INT_MAX {
        let detail = format!("{} dst floats exceed i32::MAX", s.dst_floats());
        return Err(rule("dst index", &detail));
    }
    Ok(())
}

/// The `mmq_args` geometry of a grouped launch: the pin's assignment (mmq.cu:255-261), after
/// [`check_launch_shape`], which bounds every field far below `i64::MAX`. The token count is the
/// grid width (`ncols_max = ncols_opt = m`, upstream `ne12`), the expert is the channel
/// (`nchannels_x = nchannels_y = n_experts`, stride `s02` in weight blocks), `y` and `dst` carry
/// the `m * top_k` compact columns, and the channel and sample strides of `y` and `dst` are 0
/// (every grouped path zeroes those offsets, mmq.cuh:966-967, :1054-1055, :1143-1144).
/// The widest tile the grouped launch may ask for. Above it the vendored Q4_K table has no tiling
/// config, and under stream-k a column's reduction order follows its index inside the tile, which
/// is the property pg1b pins.
const GROUPED_TILE_MAX: i64 = 64;

/// The columns one expert is expected to own: the average of the grouped launch.
fn hint_cols(ne: usize, experts: usize) -> i64 {
    let per = ne.div_ceil(experts.max(1)).max(1);
    i64::try_from(per).unwrap_or(i64::MAX)
}

/// The tile hint of the grouped launch (see `fill_mmid_args` and the vendored chooser).
///
/// The vendored `mul_mat_q_switch_J` picks J, the destination tile width, from `ncols_opt`, while
/// the launch grid still spans every token (`ncols_max`). With the whole chunk as the hint it picks
/// the widest tile, so every expert computes a full tile although it owns a few columns; a block
/// whose tile starts past its expert returns before it loads anything (`mmq.cuh`, `jt*J >= col_diff`).
/// Aiming the hint at the average expert count rounds J down instead.
///
/// On by default: `INFERENCE_CUDA_MMID_COLS_HINT` unset or `auto` follows `hint_cols`; `off` or
/// `0` disables it, and a positive number is used as given. The switch exists because the tile
/// width changes the k tiling and therefore the accumulation grouping.
fn mmid_geometry(s: &MmidLaunchShape) -> MmidGeometry {
    let mut g = fill_mmid_args(s);
    // The tiling rows of the vendored Q4_K tiling patch exist for Q4_K only. Every other type keeps
    // its stream-k configs, and under stream-k a narrower tile changes the per-column reduction
    // order without the tiling guarantee, so the hint must not reach those types.
    if s.quant == QuantType::Q4K
        && let Some(hint) = cols_hint(s)
    {
        g.ncols_opt = clamp_hint(hint, g.ncols_max);
    }
    g
}

/// The tile hint a grouped launch may use: the operator value, clamped to a width the vendored
/// Q4_K table implements as tiling, and never past the token count.
fn clamp_hint(hint: i64, ncols_max: i64) -> i64 {
    hint.clamp(1, GROUPED_TILE_MAX.min(ncols_max.max(1)))
}

// A test-only override of the hint, so a device test can execute a chosen tile width. The
// environment is read once per process, so a sweep cannot use it.
#[cfg(test)]
thread_local! {
    static HINT_OVERRIDE: std::cell::Cell<Option<Option<i64>>> =
        const { std::cell::Cell::new(None) };
}

/// Sets the test override: `Some(Some(j))` forces tile width j, `Some(None)` forces the hint off,
/// `None` restores the environment.
#[cfg(test)]
pub(in crate::kernels::cuda) fn set_hint_override(hint: Option<Option<i64>>) {
    HINT_OVERRIDE.with(|cell| cell.set(hint));
}

/// The operator hint, read once per process.
fn cols_hint(s: &MmidLaunchShape) -> Option<i64> {
    static HINT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    #[cfg(test)]
    if let Some(forced) = HINT_OVERRIDE.with(|cell| cell.get()) {
        return forced;
    }
    let raw = HINT.get_or_init(|| std::env::var("INFERENCE_CUDA_MMID_COLS_HINT").ok());
    let text = raw.as_deref().unwrap_or("auto").trim();
    if text == "off" || text == "0" {
        return None;
    }
    if text.is_empty() || text == "auto" {
        return Some(hint_cols(s.ids.ne_get_rows(), s.ids.n_experts));
    }
    text.parse::<i64>().ok().filter(|n| *n > 0)
}

pub(in crate::kernels::cuda) fn fill_mmid_args(s: &MmidLaunchShape) -> MmidGeometry {
    let wide = |value: usize| i64::try_from(value).unwrap_or(i64::MAX);
    let (ne, experts, tokens) = (
        wide(s.ids.ne_get_rows()),
        wide(s.ids.n_experts),
        wide(s.ids.m),
    );
    MmidGeometry {
        type_x: i64::from(s.quant.ggml()),
        ncols_x: wide(s.k),
        nrows_x: wide(s.n),
        ncols_dst: ne,
        stride_row_x: wide(s.row_blocks()),
        ncols_y: ne,
        nrows_dst: wide(s.ld),
        nchannels_x: experts,
        nchannels_y: experts,
        stride_channel_x: wide(s.stride_blocks()),
        stride_channel_y: 0,
        stride_channel_dst: 0,
        nsamples_x: 1,
        nsamples_y: 1,
        stride_sample_x: 0,
        stride_sample_y: 0,
        stride_sample_dst: 0,
        ncols_max: tokens,
        ncols_opt: tokens,
    }
}

/// The sizes and alignment of the ranges of a grouped launch (the q8_1 pad for `y`, the
/// fixup-gather pad for `ids_dst`, the fixup maximum of `nsm`), then their claims in the order
/// weight, y, ids_dst, expert_bounds, dst, fixup.
fn launch_claims<'a>(
    call: &MmidLaunchCall<'a, '_>,
    s: &MmidLaunchShape,
    nsm: usize,
) -> Result<[Claim<'a>; 6], EngError> {
    let (ne, ids) = (s.ids.ne_get_rows(), &call.ids.out);
    let needs = [
        (
            "weight",
            &call.weight.base,
            s.weight_blocks() * s.quant.block_bytes(),
            16,
        ),
        ("y", &call.y, q8_1_bytes(s.k, ne)?, 16),
        ("ids_dst", &ids.ids_dst, ids_dst_ints(ne, s.ids.m)? * 4, 4),
        (
            "expert_bounds",
            &ids.expert_bounds,
            expert_bounds_ints(s.ids.n_experts)? * 4,
            4,
        ),
        ("dst", &call.dst, s.dst_floats() * 4, 4),
        ("fixup", &call.fixup, fixup_floats_max(nsm)? * 4, 16),
    ];
    for (name, range, need, align) in &needs {
        require_fit(name, range, *need, *align)?;
    }
    Ok(needs.map(|(name, range, need, _)| Claim::of(name, range, need)))
}

impl CudaContext {
    /// Queues the ids guard of `call` on `stream` and returns; `ids_g` and the status word
    /// are valid after the stream drained.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a shape outside the rules of [`MmidShape`], a
    /// range that is too short or misaligned, and a written range (`ids_g`, `status`) that
    /// overlaps another range of the call.
    pub(crate) fn mmq_ids_guard(
        &self,
        stream: &DeviceStream,
        call: IdsGuardCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_ids_guard(&call)?;
        self.launch_on(stream, move |s| mmq_ids_guard_launch(s, raw))
    }

    /// Queues the id builder on `stream`: the ids guard of `guard`, then `mm_ids_helper` on the
    /// guarded copy into `out`. Returns the [`MmidIds`] when both are queued; its ranges hold
    /// the ids after the stream drained, and a launch queued behind them on the same stream
    /// reads them in order.
    ///
    /// Returns `BadRequest` before any native call for the conditions of
    /// [`Self::mmq_ids_guard`], for `out` ranges that are too short (the fixup-gather pad of
    /// `ids_dst`), misaligned or overlapping any range of the call, and for an `m` whose
    /// `4 * m` bytes of helper shared memory exceed the device's opt-in limit (mmid.cu:147).
    pub(crate) fn mmq_build_ids<'b>(
        &self,
        stream: &DeviceStream,
        guard: IdsGuardCall<'b>,
        out: MmidIdsOut<'b>,
    ) -> Result<MmidIds<'b>, EngError> {
        self.ensure_usable()?;
        let raw = self.plan_build_ids(&guard, &out)?;
        self.launch_on(stream, move |s| mmq_build_ids_launch(s, raw))?;
        Ok(MmidIds {
            out,
            shape: guard.shape,
        })
    }

    /// Queues the scatter quantize of `call` on `stream` (the gate/up input of a grouped launch):
    /// each of the `m` token rows is quantized once into its `top_k` compact columns, through
    /// the inverse map `ids_src1` of `ids`. Refusals and status are those of
    /// [`Self::mmq_quantize_scatter`]; the shape comes from `ids`.
    pub(crate) fn mmq_quantize_scatter_ids(
        &self,
        stream: &DeviceStream,
        ids: &MmidIds<'_>,
        call: MmidQuantCall<'_>,
    ) -> Result<(), EngError> {
        let MmidShape { m, top_k, .. } = ids.shape;
        self.mmq_quantize_scatter(
            stream,
            MmqQuantizeScatterCall {
                packing: call.packing,
                ty: mmq_type_of(call.quant),
                k: call.k,
                row_stride: call.row_stride,
                m,
                top_k,
                x: call.x,
                inv: ids.out.ids_src1,
                y: call.y,
            },
        )
    }

    /// Queues the gathered quantize of `call` on `stream` (the down input): column `j` of `y` is
    /// the quantize of row `ids_dst[j]` of `x`, the forward map of `ids`, so rows in token-slot
    /// order come out in the compact order the down launch reads.
    pub(crate) fn mmq_quantize_gather_ids(
        &self,
        stream: &DeviceStream,
        ids: &MmidIds<'_>,
        call: MmidQuantCall<'_>,
    ) -> Result<(), EngError> {
        self.mmq_quantize(
            stream,
            MmqQuantizeCall {
                packing: call.packing,
                ty: mmq_type_of(call.quant),
                k: call.k,
                row_stride: call.row_stride,
                ncols: ids.shape.ne_get_rows(),
                x: call.x,
                gather: Some(ids.out.ids_dst),
                y: call.y,
            },
        )
    }

    /// Every check of a guard call, then the native view of its ranges.
    fn plan_ids_guard<'b>(&self, call: &IdsGuardCall<'b>) -> Result<IdsGuardRaw<'b>, EngError> {
        let raw = IdsGuardRaw {
            ids: self.raw_range(&call.ids)?,
            ids_g: self.raw_range(&call.ids_g)?,
            status: call
                .status
                .as_ref()
                .map(|s| self.raw_range(s))
                .transpose()?,
            shape: check_ids_shape(&call.shape)?,
        };
        let (written, read) = guard_claims(call)?;
        require_written_apart(&written, &[read])?;
        Ok(raw)
    }

    /// Every check of a builder call, then the native view of its ranges.
    fn plan_build_ids<'b>(
        &self,
        guard: &IdsGuardCall<'b>,
        out: &MmidIdsOut<'b>,
    ) -> Result<BuildIdsRaw<'b>, EngError> {
        let raw = BuildIdsRaw {
            guard: self.plan_ids_guard(guard)?,
            ids_src1: self.raw_range(&out.ids_src1)?,
            ids_dst: self.raw_range(&out.ids_dst)?,
            expert_bounds: self.raw_range(&out.expert_bounds)?,
        };
        let (mut written, read) = guard_claims(guard)?;
        written.extend(out_claims(out, &guard.shape)?);
        require_written_apart(&written, &[read])?;
        let shared = guard.shape.m * 4;
        let smpbo = self.mmq_device_facts()?.smpbo;
        if shared > smpbo {
            return Err(rule(
                "shared memory",
                &format!("mm_ids_helper needs {shared} bytes, above the {smpbo} of the device"),
            ));
        }
        Ok(raw)
    }
}

impl CudaContext {
    /// Queues one grouped GEMM of `call` on `stream`: the vendored `mul_mat_q_case` of the
    /// weight type, with the stream-k fixup when the launch needs one. Returns when the kernels
    /// are queued; `dst` holds the products after the stream drained. The ids are read in
    /// stream order, so their constructor's work must precede this launch on `stream`.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a number outside the rules of
    /// [`check_launch_shape`], ids made for another shape, a range that is too short or
    /// misaligned, and a written range (`dst`, `fixup`) that overlaps another range of the call.
    pub(crate) fn mmq_mmid_launch(
        &self,
        stream: &DeviceStream,
        call: MmidLaunchCall<'_, '_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_mmid_launch(&call)?;
        self.launch_on(stream, move |s| mmq_mmid_launch(s, raw))
    }

    /// Every check of a grouped launch, then the native view of its ranges and its geometry.
    fn plan_mmid_launch<'b>(
        &self,
        call: &MmidLaunchCall<'b, '_>,
    ) -> Result<MmidLaunchRaw<'b>, EngError> {
        let ids = &call.ids.out;
        let shape = MmidLaunchShape {
            ids: call.shape,
            quant: call.weight.quant,
            k: call.weight.k,
            n: call.weight.n,
            stride_bytes: call.weight.stride_bytes,
            ld: call.ld,
        };
        let (x, y) = (self.raw_range(&call.weight.base)?, self.raw_range(&call.y)?);
        let (ids_dst, expert_bounds) = (
            self.raw_range(&ids.ids_dst)?,
            self.raw_range(&ids.expert_bounds)?,
        );
        let (dst, fixup) = (self.raw_range(&call.dst)?, self.raw_range(&call.fixup)?);
        check_launch_shape(&shape)?;
        if call.ids.shape != call.shape {
            let detail = format!("ids of {:?} for a {:?} launch", call.ids.shape, call.shape);
            return Err(rule("ids shape", &detail));
        }
        let nsm = self.mmq_device_facts()?.nsm;
        let claims = launch_claims(call, &shape, nsm)?;
        require_written_apart(&claims[4..], &claims[..4])?;
        Ok(MmidLaunchRaw {
            x,
            y,
            ids_dst,
            expert_bounds,
            dst,
            fixup,
            geometry: mmid_geometry(&shape),
            // The one `usize` -> `c_int` edge of the facts (the Rust binding reads `nsm` as
            // `usize`; the C entry takes it as `int`), checked so a driver reporting more
            // than 2^31 SMs is refused instead of wrapped.
            nsm: c_int::try_from(nsm)
                .map_err(|_| rule("facts", &format!("the device reports {nsm} SMs")))?,
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
pub(in crate::kernels::cuda) mod tests;
