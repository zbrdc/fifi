//! The low-rank attention mixer of a visit: every head reads one cached latent row per position.
//!
//! The query takes two projections with a norm between them and ends as a content part and a
//! rotary part per head. One projection writes the key and value of a row as a single latent
//! row with its rotary part; the prep norms it, turns the rotary part at the row's position and
//! stores it in the layer's cache. A per-head factor carries the query's content part into the
//! latent space, so the score of a head is one dot product over a cached row; a second factor
//! carries the attended latent back to the head width. A projected gate scales the result
//! before the output projection.
//!
//! The score has two forms over the same cache and the same partials. A step of one row, a
//! chunk of up to [`FEW_ROWS`] rows and every chunk under the exact switch score a row at a time
//! with f32 queries, so those rows keep the bits of a one-row step. A larger chunk scores all
//! heads of 8 rows against a key tile as one matrix product on tensor cores, its queries
//! rounded to f16 like the cached rows; a prompt at depth then costs a fraction of the row form.

use super::super::rows::{FEW_ROWS, Rows, StepIo};
use crate::EngError;
use crate::kernels::cuda::resident::block_ops::{BlockOp, MLA_GROUP, MlaShape};
use crate::kernels::cuda::{DeviceBuffer, DeviceRange};

/// The partial rows a visit may launch at most; the scratch is sized for them.
pub(crate) const PART_ROWS: usize = 320;

/// Token rows of one query tile of the chunk score.
const TILE_ROWS: usize = 8;
/// The fewest keys a split of a score holds.
const SPLIT_KEYS: usize = 1024;
/// The blocks a chunk score launches at least when its keys allow: a short chunk at depth
/// splits its keys so that every multiprocessor of a large card has a block.
const SCORE_BLOCKS: usize = 160;
/// The most key splits a score takes.
const MAX_SPLITS: usize = 64;

/// The tensors of one layer.
pub(crate) struct Names {
    /// The two query projections and the norm between them.
    pub(crate) query_low: String,
    pub(crate) query_norm: String,
    pub(crate) query: String,
    /// The projection of the shared key and value row, and the norm of its latent part.
    pub(crate) latent: String,
    pub(crate) latent_norm: String,
    pub(crate) gate: String,
    pub(crate) out: String,
}

impl Names {
    /// Every tensor of the layer, in the order a load takes them.
    pub(crate) fn tensors(&self) -> [&str; 7] {
        [
            &self.query_low,
            &self.query_norm,
            &self.query,
            &self.latent,
            &self.latent_norm,
            &self.gate,
            &self.out,
        ]
    }

    /// Resolves every tensor, so a missing one stops a load and not a step.
    pub(crate) fn check(&self, map: &super::super::rows::Map<'_>) -> Result<(), EngError> {
        for name in [&self.query_norm, &self.latent_norm] {
            map.raw(name)?;
        }
        for name in [
            &self.query_low,
            &self.query,
            &self.latent,
            &self.gate,
            &self.out,
        ] {
            map.native(name)?;
        }
        Ok(())
    }
}

/// The widths of the layers.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    /// The content part of a head's query.
    pub(crate) key_dim: usize,
    /// The latent part and the rotary part of a cached row.
    pub(crate) rank: usize,
    pub(crate) rot_dims: usize,
    /// The width between the two query projections.
    pub(crate) query_rank: usize,
    pub(crate) scale: f32,
    pub(crate) eps: f32,
}

/// The rows a visit works in, the layer's two factors and its cache.
pub(crate) struct LatentRows<'a> {
    /// The normed input rows.
    pub(crate) input: Rows<'a>,
    pub(crate) query_low: &'a DeviceBuffer,
    /// The q8_1 rows of the normed low query; its f32 rows go to [`Self::out`], which the
    /// visit writes last.
    pub(crate) query_low_q8: &'a DeviceBuffer,
    pub(crate) query: &'a DeviceBuffer,
    pub(crate) latent: &'a DeviceBuffer,
    /// The queries carried into the latent space.
    pub(crate) absorbed: &'a DeviceBuffer,
    pub(crate) part: &'a DeviceBuffer,
    pub(crate) mixed: &'a DeviceBuffer,
    /// The attended rows at the head width, their gate, and the gated rows.
    pub(crate) attended: &'a DeviceBuffer,
    pub(crate) gate: &'a DeviceBuffer,
    pub(crate) gated: Rows<'a>,
    /// The mixer output rows, hidden wide.
    pub(crate) out: &'a DeviceBuffer,
    /// The factors that carry a query into the latent space and a latent to the head width.
    pub(crate) key_factor: DeviceRange<'a>,
    pub(crate) value_factor: DeviceRange<'a>,
    /// The layer's cache and the positions it holds.
    pub(crate) cache: DeviceRange<'a>,
    pub(crate) cells: usize,
    /// Optional device `i32` position word (`StepParams.pos`) for graph-captured 1-row decode.
    pub(crate) step_pos: Option<DeviceRange<'a>>,
}

/// One low-rank attention visit over `n` rows whose first row sits at `p0`.
pub(crate) fn mix(
    io: &StepIo<'_>,
    names: &Names,
    d: Dims,
    rows: &LatentRows<'_>,
    n: usize,
    p0: usize,
) -> Result<(), EngError> {
    let (row, query_row, width) = (
        d.rank + d.rot_dims,
        d.key_dim + d.rot_dims,
        d.heads * d.head_dim,
    );
    io.mv(&names.query_low, rows.input, rows.query_low, n)?;
    io.norm_rows(
        n,
        d.query_rank,
        d.eps,
        rows.query_low,
        io.raw(&names.query_norm)?,
        rows.out,
        rows.query_low_q8,
    )?;
    io.mv(
        &names.query,
        Rows::both(rows.query_low_q8, rows.out),
        rows.query,
        n,
    )?;
    io.mv(&names.latent, rows.input, rows.latent, n)?;
    if let Some(pos_word) = rows.step_pos {
        if n != 1 || io.exact || !(MLA_GROUP..=64).contains(&d.heads) {
            return Err(EngError::BadRequest(format!(
                "{}: step_pos requires a single non-exact row with 8..=64 heads",
                io.family
            )));
        }
        io.run(BlockOp::mla_kv_prep_step(
            d.rank,
            d.rot_dims,
            rows.cells,
            d.eps,
            rows.latent.range(0, row * 4)?,
            io.raw(&names.latent_norm)?,
            rows.cache,
            pos_word,
        )?)?;
    } else {
        io.run(BlockOp::mla_kv_prep(
            n,
            d.rank,
            d.rot_dims,
            p0,
            rows.cells,
            d.eps,
            rows.latent.range(0, n * row * 4)?,
            io.raw(&names.latent_norm)?,
            rows.cache,
        )?)?;
    }
    let shape = MlaShape {
        n,
        heads: d.heads,
        nope: d.key_dim,
        rope: d.rot_dims,
        rank: d.rank,
    };
    io.run(BlockOp::mla_q_absorb(
        shape,
        rows.query.range(0, n * d.heads * query_row * 4)?,
        rows.key_factor,
        rows.absorbed.range(0, n * d.heads * row * 4)?,
    )?)?;
    let splits = score(io, d, rows, n, p0)?;
    io.run(BlockOp::mla_reduce(
        n,
        d.heads,
        d.rank,
        splits,
        rows.part.range(0, part_bytes(d, n, splits))?,
        rows.mixed.range(0, n * d.heads * d.rank * 4)?,
    )?)?;
    io.run(BlockOp::mla_project(
        n,
        d.heads,
        d.rank,
        d.head_dim,
        rows.mixed.range(0, n * d.heads * d.rank * 4)?,
        rows.value_factor,
        rows.attended.range(0, n * width * 4)?,
    )?)?;
    io.mv(&names.gate, rows.input, rows.gate, n)?;
    io.quantize(n, width, rows.attended, Some(rows.gate), rows.gated)?;
    io.mv(&names.out, rows.gated, rows.out, n)
}

/// Bytes of the partials of `n` rows over `splits` key splits.
fn part_bytes(d: Dims, n: usize, splits: usize) -> usize {
    n * (d.heads / MLA_GROUP) * splits * MLA_GROUP * (d.rank + 4) * 4
}

/// Scores the absorbed queries of `n` rows at `p0..` against the cached rows and leaves one
/// partial softmax per row, head and key split in [`LatentRows::part`]. Returns the splits.
pub(crate) fn score(
    io: &StepIo<'_>,
    d: Dims,
    rows: &LatentRows<'_>,
    n: usize,
    p0: usize,
) -> Result<usize, EngError> {
    let row = d.rank + d.rot_dims;
    let shape = MlaShape {
        n,
        heads: d.heads,
        nope: d.key_dim,
        rope: d.rot_dims,
        rank: d.rank,
    };
    let keys = p0 + n;
    let groups = d.heads / MLA_GROUP;
    // A single non-exact row uses the shared-KV decode kernel (all heads share one KV load per
    // split; fixed split count when reading pos from StepParams for graph capture).
    if rows.step_pos.is_some() && (n != 1 || io.exact || !(MLA_GROUP..=64).contains(&d.heads)) {
        return Err(EngError::BadRequest(format!(
            "{}: step_pos requires a single non-exact row with 8..=64 heads",
            io.family
        )));
    }
    if n == 1 && !io.exact && (MLA_GROUP..=64).contains(&d.heads) {
        let cap = MAX_SPLITS
            .min(rows.part.bytes() / part_bytes(d, 1, 1))
            .max(1);
        let splits = if rows.step_pos.is_some() {
            cap
        } else {
            keys.div_ceil(16).clamp(1, cap)
        };
        io.run(BlockOp::mla_attn_decode(
            shape,
            splits,
            p0,
            rows.cells,
            d.scale,
            rows.absorbed.range(0, d.heads * row * 4)?,
            rows.cache,
            rows.part.range(0, part_bytes(d, 1, splits))?,
            rows.step_pos,
        )?)?;
        return Ok(splits);
    }
    // A few rows keep the bits of a one-row step, and the exact route keeps its queries in f32:
    // both score a row at a time. A larger chunk scores in tiles on tensor cores.
    let tiled = !io.exact && n > FEW_ROWS;
    let splits = if tiled {
        keys.div_ceil(SPLIT_KEYS)
            .min(SCORE_BLOCKS.div_ceil(n.div_ceil(TILE_ROWS) * groups))
            .min(MAX_SPLITS)
            .min(rows.part.bytes() / part_bytes(d, n, 1))
            .max(1)
    } else if n >= 64 {
        1
    } else {
        keys.div_ceil(SPLIT_KEYS)
            .clamp(1, 32)
            .min(PART_ROWS / (n * groups))
            .max(1)
    };
    let per_split = keys.div_ceil(splits).div_ceil(32) * 32;
    let op = if tiled {
        BlockOp::mla_attn_chunk
    } else {
        BlockOp::mla_attn
    };
    io.run(op(
        shape,
        p0,
        splits,
        per_split,
        d.scale,
        rows.absorbed.range(0, n * d.heads * row * 4)?,
        rows.cache,
        rows.part.range(0, part_bytes(d, n, splits))?,
    )?)?;
    Ok(splits)
}
