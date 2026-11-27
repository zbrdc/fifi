//! The routed expert projections of a chunk as grouped products.
//!
//! A router picks `used` of `experts` stacked weight matrices for every token row. Above a few
//! rows the projections through the picked experts run grouped by expert: every expert's
//! weights are read once for all the rows that picked it, and the products land at the slot
//! rows (`used` per token) a per-token launch writes. [`Picks::ids`] regroups the router ids of
//! a chunk, [`token_rows`] and [`slot_rows`] describe the input of a product, and [`grouped`]
//! runs one stack.

use super::rows::StepIo;
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_mmid::{
    IdsGuardCall, MmidIds, MmidIdsOut, MmidShape, Q8Cursor, RoutedQuant, RoutedX, routed_gemm,
    routed_serves,
};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::{expert_bounds_ints, ids_dst_ints};
use crate::kernels::cuda::{DeviceBuffer, DeviceRange};

/// The router picks of a model and the scratch their regrouping works in.
pub(crate) struct Picks<'a> {
    pub(crate) experts: usize,
    pub(crate) used: usize,
    /// The router ids, i32 `[rows][used]`.
    pub(crate) ids: &'a DeviceBuffer,
    pub(crate) ids_g: &'a DeviceBuffer,
    pub(crate) ids_src1: &'a DeviceBuffer,
    pub(crate) ids_dst: &'a DeviceBuffer,
    pub(crate) expert_bounds: &'a DeviceBuffer,
}

impl<'a> Picks<'a> {
    /// The ints `ids_dst` holds for a chunk of `rows` token rows.
    pub(crate) fn dst_ints(used: usize, rows: usize) -> Result<usize, EngError> {
        ids_dst_ints(rows * used, rows)
    }

    /// The id tables of `n` token rows, regrouped by expert.
    pub(crate) fn ids(&self, io: &StepIo<'_>, n: usize) -> Result<MmidIds<'a>, EngError> {
        let picks = n * self.used * 4;
        io.ctx.mmq_build_ids(
            io.stream,
            IdsGuardCall {
                shape: MmidShape {
                    n_experts: self.experts,
                    m: n,
                    top_k: self.used,
                },
                ids: self.ids.range(0, picks)?,
                ids_g: self.ids_g.range(0, picks)?,
                status: None,
            },
            MmidIdsOut {
                ids_src1: self.ids_src1.range(0, picks)?,
                ids_dst: self.ids_dst.range(0, Self::dst_ints(self.used, n)? * 4)?,
                expert_bounds: self
                    .expert_bounds
                    .range(0, expert_bounds_ints(self.experts)? * 4)?,
            },
        )
    }
}

/// Whether the expert stack `name` runs as a grouped product.
pub(crate) fn serves(io: &StepIo<'_>, name: &str) -> bool {
    io.native(name).is_ok_and(routed_serves)
}

/// The input of a product every picked expert of a token reads: the f32 token `rows` of `k`
/// values, and `q8` as the scratch of their quantized columns.
pub(crate) fn token_rows<'a, 'q>(
    ids: &'q MmidIds<'q>,
    rows: DeviceRange<'a>,
    k: usize,
    q8: DeviceRange<'a>,
) -> RoutedX<'a, 'q> {
    input(ids, rows, k, q8, false)
}

/// The input of a product that reads one row per pick: the f32 slot `rows` of `k` values.
pub(crate) fn slot_rows<'a, 'q>(
    ids: &'q MmidIds<'q>,
    rows: DeviceRange<'a>,
    k: usize,
    q8: DeviceRange<'a>,
) -> RoutedX<'a, 'q> {
    input(ids, rows, k, q8, true)
}

fn input<'a, 'q>(
    ids: &'q MmidIds<'q>,
    rows: DeviceRange<'a>,
    k: usize,
    q8: DeviceRange<'a>,
    per_slot: bool,
) -> RoutedX<'a, 'q> {
    RoutedX {
        q8,
        rows,
        row_stride: k,
        per_slot,
        quant: RoutedQuant {
            ids,
            gather: per_slot,
            packing: MmqPacking::Rounded,
            cursor: Q8Cursor::default(),
        },
    }
}

/// One grouped product of the expert stack `name` over the picks `ids`: the slot rows land in
/// `dst` at stride `ld`.
pub(crate) fn grouped(
    io: &StepIo<'_>,
    name: &str,
    ids: &MmidIds<'_>,
    x: &RoutedX<'_, '_>,
    ld: usize,
    dst: DeviceRange<'_>,
) -> Result<(), EngError> {
    routed_gemm(
        io.ctx,
        io.stream,
        io.native(name)?,
        ids,
        x,
        ld,
        dst,
        StepIo::whole(io.fixup)?,
    )
}
