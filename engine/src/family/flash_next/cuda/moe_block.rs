//! The MoE block of one Flash-Next layer on a device hidden row, with the routed experts in the
//! expert bank (`bank.rs`) instead of all resident. One call per layer and token.
//!
//! WHAT, in launch order on the one stream:
//!
//! 1. The router (shared kernel) writes the picks, their renormalized weights and the shared gate.
//! 2. The layer's slot row goes to the device and `tiered_classify` maps every pick to its bank
//!    slot: a resident pick to its slot, a missed pick to slot 0 with weight +0.0 (the filler
//!    rule). The classify log (ten ids and weights, 80 bytes) comes back to the host.
//! 3. The resident pass is queued: gate and up of the ten slots (id-indexed launches over the bank
//!    buffer), SwiGLU, down, the shared expert, and the combine with the classified weights.
//! 4. The bank plans and runs the copies for the picks (`DeviceBank::step`, all ten picks, hits
//!    included, so the policy sees every use). The copies run on the bank's own stream, which is
//!    not ordered with the block's stream: they overlap the resident pass, and the host waits for
//!    them before it queues the miss pass.
//! 5. With at least one miss, the miss pass is queued: the same launches over the slot row after
//!    the copies, with the missed picks on their new slots at their true weights and every other
//!    pick on the filler at +0.0 ([`miss_pass_inputs`]); the combine adds it to the resident
//!    result.
//!
//! WHY this shape: the resident pass needs nothing a miss brings, so it runs while the host
//! plans and waits for the copies. The overlap is safe because of one rule of the bank's policy:
//! the slot of every pick and slot 0 are never victims in a pass, so a copy only writes a slot
//! that neither the resident pass (the picks' slots and the filler slot 0) nor anything earlier
//! reads. The slot-row upload at the start of the call waits for the block's stream, so no
//! launch of an earlier layer or token is still reading this layer's slots when a copy starts.
//! The shared expert rides in the first pass because it never misses. A layer without a miss is one pass whose result goes straight to the output. The two
//! passes together touch every pick exactly once: the weights of the two passes add to the
//! router's weights bit for bit.
//!
//! The split result is not bit-identical to one launch over an all-resident bank: the resident
//! picks and the missed picks are summed in two groups, so the floating-point association
//! differs. Every product is the same. The tests state the tolerance.
//!
//! The filler rule multiplies slot 0's expert output by +0.0. If that output held an infinity or
//! a NaN the product would be NaN and would poison the sum. With the check off (the default) a
//! non-finite slot 0 output turns the whole block output, and so the token, into NaN with no
//! error; [`SplitMoe::verify_finite_filler`] reads the filler rows back and refuses it, at the
//! price of a wait on the stream before the copies start, which removes the overlap of step 4.
//!
//! The pure parts (the classify oracle, the log parser, the spans of the bank buffer and the
//! private scratch layout) are host code with host tests; the launches are native-cuda only.

use std::ops::Range;

use super::bank::miss_pass_inputs;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::flash_next::cuda::driver::chunk::prof;
use crate::{
    EngError,
    family::{
        flash_next::moe::slot_bank::{PARTS, SlotBankLayout},
        project::exl3_jobs_plan::EXL3_JOB_WORDS,
        qwen36::tiered::classify::{Classified, ROUTE_INVALID, classify_reference},
    },
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();
/// Bytes of one entry of the classify log: the id and the bits of the weight.
const LOG_ENTRY: usize = 8;
/// Control block, request line: the sizes the classify kernel writes whether or not it is read.
const CTRL_BYTES: usize = 64;
const REQUEST_BYTES: usize = 128;
/// Every scratch region starts at a multiple of this (the arena alignment).
const REGION_ALIGN: usize = 256;

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next split MoE block: {detail}"))
}

fn align_up(value: usize) -> usize {
    value.next_multiple_of(REGION_ALIGN)
}

/// The block's private device scratch: the slot row, the classify outputs and the second pass's
/// inputs. One allocation, regions at aligned offsets: 24,832 bytes at the real sizes and one
/// row (pinned by the host tests), and the row-indexed regions grow with the planned chunk
/// width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScratchLayout {
    /// The chunk row count this layout is sized for; run_rows admits at most this many rows.
    pub(super) rows: usize,
    /// i32 `[n_expert]`: the layer's slot row, uploaded before each classification.
    pub(super) slot_row: usize,
    /// i32 `[top_k]`: the bank slot of each pick for the resident pass.
    pub(super) slot_ids: usize,
    /// f32 `[top_k]`: the resident pass's combine weights.
    pub(super) weights: usize,
    /// `8 * top_k` bytes: the raw picks (id, weight bits).
    pub(super) log: usize,
    pub(super) ctrl: usize,
    pub(super) word: usize,
    pub(super) mbox_req: usize,
    /// f32 `[hidden]`: the classify kernel copies the row here when a pick misses.
    pub(super) mbox_row: usize,
    /// i32 `[top_k]` and f32 `[top_k]`: the miss pass's slots and weights.
    pub(super) miss_ids: usize,
    pub(super) miss_weights: usize,
    /// f32 `[hidden]`: the resident pass's result while a miss pass is still to come.
    pub(super) partial: usize,
    /// i32 `[(2 + EXL3_JOB_WORDS) * routed]`: the sort lists and the job tables of the chunk's
    /// two routed job-table passes (2 list words plus one job record per pair; a source's
    /// jobs are at most its pairs).
    pub(super) perm: usize,
    pub(super) total: usize,
}

impl ScratchLayout {
    /// `rows` is the chunk row count the multi-row block plans for (`1` for the decode block);
    /// only the regions the block indexes per row scale with it.
    pub(super) fn new(
        n_expert: usize,
        top_k: usize,
        hidden: usize,
        rows: usize,
    ) -> Result<Self, EngError> {
        let overflow = || refuse("the scratch size overflows".into());
        let mul = |a: Option<usize>, b: usize| a.and_then(|at| at.checked_mul(b));
        let routed = rows.checked_mul(top_k).ok_or_else(overflow)?;
        let regions = [
            n_expert.checked_mul(I32),
            mul(rows.checked_mul(top_k), I32),
            mul(rows.checked_mul(top_k), F32),
            top_k.checked_mul(LOG_ENTRY),
            Some(CTRL_BYTES),
            Some(I32),
            Some(REQUEST_BYTES),
            hidden.checked_mul(F32),
            mul(rows.checked_mul(top_k), I32),
            mul(rows.checked_mul(top_k), F32),
            mul(rows.checked_mul(hidden), F32),
            // Two sources (the bank's and the ring's slots) plan separately: their lists sum
            // to `routed` pairs and their job tables to at most one job per pair (an expert
            // with rows owns at least one pair of its source), so 2 list words plus one
            // 4-word job record per pair cover both sources' lists and tables.
            (2 + EXL3_JOB_WORDS)
                .checked_mul(routed)
                .and_then(|words| words.checked_mul(I32)),
        ];
        let mut offsets = [0usize; 12];
        let mut at = 0usize;
        for (offset, size) in offsets.iter_mut().zip(regions) {
            *offset = at;
            at = size
                .ok_or_else(overflow)?
                .checked_next_multiple_of(REGION_ALIGN)
                .and_then(|size| at.checked_add(size))
                .ok_or_else(overflow)?;
        }
        let [
            slot_row,
            slot_ids,
            weights,
            log,
            ctrl,
            word,
            mbox_req,
            mbox_row,
            miss_ids,
            miss_weights,
            partial,
            perm,
        ] = offsets;
        Ok(Self {
            rows,
            slot_row,
            slot_ids,
            weights,
            log,
            ctrl,
            word,
            mbox_req,
            mbox_row,
            miss_ids,
            miss_weights,
            partial,
            perm,
            total: at,
        })
    }
}

/// The picks of the classify log: `(ids, weights)` in rank order.
pub(super) fn parse_log(raw: &[u8], top_k: usize) -> Result<(Vec<i32>, Vec<f32>), EngError> {
    if raw.len() != top_k * LOG_ENTRY {
        return Err(refuse(format!(
            "the classify log has {} bytes, {top_k} picks need {}",
            raw.len(),
            top_k * LOG_ENTRY
        )));
    }
    let (mut ids, mut weights) = (Vec::with_capacity(top_k), Vec::with_capacity(top_k));
    for entry in raw.as_chunks::<LOG_ENTRY>().0 {
        let (id, weight) = entry.split_at(I32);
        ids.push(i32::from_le_bytes(id.try_into().expect("four bytes")));
        weights.push(f32::from_le_bytes(weight.try_into().expect("four bytes")));
    }
    Ok((ids, weights))
}

/// The host's classification of the router's picks against `slot_row` before the copies: what the
/// device kernel computed (the kernel equals this oracle bit for bit). Refuses a route the
/// kernel flagged and a router fault, which have no second pass.
pub(super) fn classify_picks(
    slot_row: &[i32],
    slots: usize,
    ids: &[i32],
    weights: &[f32],
) -> Result<Classified, EngError> {
    let classified = classify_reference(slot_row, slots, ids, weights)
        .map_err(|error| refuse(format!("pick lists of different lengths: {error:?}")))?;
    if classified.word & ROUTE_INVALID != 0 {
        return Err(refuse(format!("the router's picks are invalid: {ids:?}")));
    }
    if let Some(at) = ids.iter().position(|&id| id < 0) {
        return Err(refuse(format!(
            "the router faulted on pick {at} (id {}); non-finite logits",
            ids[at]
        )));
    }
    Ok(classified)
}

/// Where the three parts of an expert sit in a bank buffer of `slots` slots, as the id-indexed
/// launch addresses them: member `i` of a part is the component at its start plus `i` strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SlotSpans {
    pub(super) slots: usize,
    /// Bytes between the trellises of two slots.
    pub(super) trellis_stride: usize,
    /// F32 values between the `suh` (and the `svh`) of two slots.
    pub(super) scale_stride: usize,
    pub(super) parts: [PartSpan; PARTS],
}

/// One part's ranges of the bank buffer, from the start of its component in slot 0 to its end in
/// the last slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PartSpan {
    pub(super) trellis: Range<usize>,
    pub(super) suh: Range<usize>,
    pub(super) svh: Range<usize>,
    /// `[k, n]` of the matrix.
    pub(super) shape: [usize; 2],
    pub(super) bits_twice: u8,
}

impl SlotSpans {
    pub(super) fn new(layout: &SlotBankLayout, slots: usize) -> Result<Self, EngError> {
        let mut parts = Vec::with_capacity(PARTS);
        for index in 0..PARTS {
            let part = layout.part(index);
            let [trellis, suh, svh] = layout.bank_ranges(index, slots)?;
            let bits_twice = u8::try_from(part.words / 8)
                .ok()
                .filter(|&bits| usize::from(bits) * 8 == part.words)
                .ok_or_else(|| {
                    refuse(format!("{} words per tile is not a whole rate", part.words))
                })?;
            parts.push(PartSpan {
                trellis,
                suh,
                svh,
                shape: part.shape,
                bits_twice,
            });
        }
        let record = layout.record_bytes();
        if layout.trellis_stride_bytes() != record || layout.scale_stride_floats() * F32 != record {
            return Err(refuse(
                "the trellis and scale strides of a slot differ from the record size".into(),
            ));
        }
        Ok(Self {
            slots,
            trellis_stride: layout.trellis_stride_bytes(),
            scale_stride: layout.scale_stride_floats(),
            parts: parts
                .try_into()
                .map_err(|_| refuse("three parts expected".into()))?,
        })
    }
}

/// The first pick whose weight is +0.0 (a filler pick) and whose down row holds an infinity or a NaN:
/// `rows` holds `weights.len()` rows of `hidden` f32 values. The product of such a row with +0.0 is
/// NaN, which would reach every column of the sum.
pub(super) fn first_poisoned_filler(rows: &[u8], hidden: usize, weights: &[f32]) -> Option<usize> {
    let width = hidden * F32;
    if width == 0 {
        return None;
    }
    rows.chunks_exact(width)
        .zip(weights)
        .position(|(bytes, &weight)| {
            weight == 0.0
                && !bytes
                    .as_chunks::<F32>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
        })
}

/// The two passes of a layer, as the host decided them: how the router's picks split between the
/// resident pass and the miss pass.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PassInputs {
    /// Slots and weights of the miss pass (`None` without a miss).
    pub(super) miss: Option<(Vec<i32>, Vec<f32>)>,
    pub(super) misses: usize,
}

/// The miss pass's inputs after the copies, from the classification made before them.
pub(super) fn plan_second_pass(
    ids: &[i32],
    weights: &[f32],
    classified: &Classified,
    slot_row_after: &[i32],
) -> Result<PassInputs, EngError> {
    let misses = classified.misses.len();
    if misses == 0 {
        return Ok(PassInputs { miss: None, misses });
    }
    let inputs = miss_pass_inputs(ids, weights, classified, slot_row_after)?;
    Ok(PassInputs {
        miss: Some(inputs),
        misses,
    })
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[allow(unused_imports, reason = "the decode step is the first caller")]
pub(in crate::family::flash_next) use device::{SplitMoe, SplitMoeTrunk};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[allow(
    unused_imports,
    reason = "the host tests import these from the sibling test module"
)]
pub(super) use device::{source_offsets, source_plan};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod device {
    /// The router's picks per row: each row's expert ids and their weights.
    type RoutedPicks = Vec<(Vec<i32>, Vec<f32>)>;
    use super::super::step_plan::MAX_STEP_ROWS;
    use super::{
        CTRL_BYTES, F32, I32, LOG_ENTRY, REQUEST_BYTES, ScratchLayout, SlotSpans, classify_picks,
        first_poisoned_filler, parse_log, pick_union, plan_second_pass, prof, refuse,
    };
    use crate::{
        EngError,
        family::{
            exl3::moe::{
                Exl3MoeBufs, IndexedPair, Scratch, SharedRowsIo, SharedWeights, grouped_indexed,
                shared_of, shared_rows,
            },
            flash_next::{cuda::bank::device::DeviceBank, moe::slot_bank::SlotBankLayout},
            project::{
                cuda::{
                    exl3::{Exl3BankSpan, Exl3Members, Exl3Weight},
                    exl3_jobs::{
                        Lists, PairPlan, Scratch as JobScratch, Table, jobs_call, pair_layout,
                        plan_jobs, plan_pairs, project_jobs, write_lists,
                    },
                },
                exl3_jobs_plan::EXL3_JOB_WORDS,
            },
            qwen36::cuda::moe::{Launch, MoeDims, RouteBufs},
        },
        kernels::cuda::{
            CudaContext, DeviceBuffer, DeviceRange,
            prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
            resident::{
                moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
                router::{RouterCall, RouterOut, RouterShape, RouterWeights},
                tiered_classify::{ClassifyCall, ClassifyShape},
            },
        },
        weights::exl3::Codebook,
    };

    /// The `rows` rows of `range` at row `index` of `row_bytes` each.
    fn rows_of<'b>(
        range: &DeviceRange<'b>,
        row_bytes: usize,
        index: usize,
        rows: usize,
    ) -> Result<DeviceRange<'b>, EngError> {
        range
            .buffer()
            .range(range.offset() + index * row_bytes, rows * row_bytes)
    }

    /// The counting sort of one source's pairs: `sub` lists the chunk's pair positions of the
    /// source's picks in chunk order; the shared sort runs over their slot ids and the lists
    /// are translated back to the chunk's pair positions and rows.
    pub(crate) fn source_plan(
        ids: &[i32],
        sub: &[usize],
        slots: usize,
        top_k: usize,
    ) -> Result<PairPlan, EngError> {
        let picks: Vec<i32> = sub.iter().map(|&at| ids[at]).collect();
        let mut plan = plan_pairs(&picks, slots, top_k)?;
        let overflow = || refuse("the chunk's pair count exceeds i32".into());
        for (order, scatter) in plan.order.iter_mut().zip(&mut plan.scatter) {
            let pair = sub[*scatter as usize];
            *scatter = i32::try_from(pair).map_err(|_| overflow())?;
            *order = i32::try_from(pair / top_k).map_err(|_| overflow())?;
        }
        Ok(plan)
    }

    /// The perm-region byte offset of each source's lists and job table, and the offset of its
    /// gate/up segment of the pair faces: source 0 (the bank) at the region's base, source 1
    /// (the ring) after source 0's lists (2 words a pair) and its job table (a 4-word record
    /// a job).
    pub(crate) fn source_offsets(
        perm: usize,
        plans: [&PairPlan; 2],
        sub: [&[usize]; 2],
    ) -> ([usize; 2], [usize; 2]) {
        let mut perm_at = perm;
        let mut gu_at = 0usize;
        let mut at_of = [0usize; 2];
        let mut seg_of = [0usize; 2];
        for source in 0..2 {
            at_of[source] = perm_at;
            seg_of[source] = gu_at;
            let jobs = plan_jobs(&plans[source].counts).len();
            perm_at += (2 * sub[source].len() + EXL3_JOB_WORDS * jobs) * I32;
            gu_at += sub[source].len();
        }
        (at_of, seg_of)
    }

    /// The rotated-row and partial-sum ranges of a fused gate and up launch over `n` pairs,
    /// when the second rotation fits behind the partial sums of the scratch's `z` range;
    /// `None` otherwise, and the caller runs the two projections separately.
    fn pair_scratch_of<'b>(
        scratch: &Scratch<'b>,
        n: usize,
        inter: usize,
        hidden: usize,
    ) -> Result<Option<(JobScratch<'b>, JobScratch<'b>)>, EngError> {
        let Some(layout) = pair_layout((n, inter, hidden), scratch.z.bytes()) else {
            return Ok(None);
        };
        let z = scratch
            .z
            .buffer()
            .range(scratch.z.offset(), layout.z_bytes)?;
        let second_xh = scratch
            .z
            .buffer()
            .range(scratch.z.offset() + layout.tail, layout.xh_bytes)?;
        Ok(Some((
            JobScratch { xh: scratch.xh, z },
            JobScratch { xh: second_xh, z },
        )))
    }

    /// The trunk weights of one layer's MoE block: the router, the shared expert's gate row, and
    /// the shared expert's three matrices. The caller borrows them from the loaded weights.
    pub(in crate::family::flash_next) struct SplitMoeTrunk<'w> {
        /// Raw F32 `[n_expert][hidden]`.
        pub(in crate::family::flash_next) router: &'w DeviceBuffer,
        /// Raw F32 `[hidden]`: the shared expert's gate row.
        pub(in crate::family::flash_next) router_shared: &'w DeviceBuffer,
        pub(in crate::family::flash_next) shared_gate: &'w Exl3Weight,
        pub(in crate::family::flash_next) shared_up: &'w Exl3Weight,
        pub(in crate::family::flash_next) shared_down: &'w Exl3Weight,
    }

    /// The block of every layer: one trunk per layer, the bank and the private scratch.
    pub(in crate::family::flash_next) struct SplitMoe<'w> {
        trunks: Vec<SplitMoeTrunk<'w>>,
        bank: DeviceBank,
        spans: SlotSpans,
        /// The record layout the bank spans were built from; the ring's spans are rebuilt
        /// from it at the ring's own record count, which may be smaller than the bank's.
        bank_layout: SlotBankLayout,
        layout: ScratchLayout,
        scratch: DeviceBuffer,
        book: Codebook,
        verify_finite_filler: bool,
        /// The prefill expert ring: `None` until [`SplitMoe::plan_ring`] funds one; the
        /// multi-row path streams through it instead of admitting into the bank.
        ring: Option<crate::family::flash_next::cuda::prefill_ring::device::PrefillRing>,
    }

    /// One pass over the bank.
    struct Pass<'a> {
        layer: usize,
        /// i32 `[top_k]`: the bank slot of each pick.
        ids: DeviceRange<'a>,
        /// f32 `[top_k]`: the combine weight of each pick.
        weights: DeviceRange<'a>,
        /// Whether the shared expert rides in this pass.
        shared: bool,
        /// f32 `[hidden]`: the residual the combine adds to.
        res: DeviceRange<'a>,
        out: DeviceRange<'a>,
    }

    /// One pair of an id-indexed launch: the part, the pick whose slot id chooses the member, the
    /// input row and the output row.
    struct Item<'a> {
        part: usize,
        pick: u32,
        x: DeviceRange<'a>,
        out: DeviceRange<'a>,
    }

    fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
        buffer.range(0, buffer.bytes())
    }

    fn row<'b>(
        range: &DeviceRange<'b>,
        width: usize,
        index: usize,
    ) -> Result<DeviceRange<'b>, EngError> {
        range.buffer().range(range.offset() + index * width, width)
    }

    fn le_i32(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn le_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    impl<'w> SplitMoe<'w> {
        /// `trunks[layer]` is layer `layer`'s trunk; `bank` has one row per layer. `layout` is the
        /// bank's record layout, the one its records were packed with. Allocates the private
        /// scratch (24,832 bytes at the decode shape, growing with the planned width), zeroed.
        pub(in crate::family::flash_next) fn new(
            ctx: &CudaContext,
            trunks: Vec<SplitMoeTrunk<'w>>,
            bank: DeviceBank,
            layout: &SlotBankLayout,
            dims: &MoeDims,
        ) -> Result<Self, EngError> {
            let geometry = *bank.geometry();
            if trunks.len() != geometry.layers {
                return Err(refuse(format!(
                    "{} trunks for a bank of {} layers",
                    trunks.len(),
                    geometry.layers
                )));
            }
            if geometry.experts != dims.n_expert
                || geometry.top_k != dims.top_k
                || geometry.record_bytes != layout.record_bytes()
            {
                return Err(refuse(format!(
                    "the bank ({} experts, {} picks, {} byte records) does not match the block \
                     ({} experts, {} picks, {} byte records)",
                    geometry.experts,
                    geometry.top_k,
                    geometry.record_bytes,
                    dims.n_expert,
                    dims.top_k,
                    layout.record_bytes()
                )));
            }
            let (gate, down) = (layout.part(0).shape, layout.part(2).shape);
            if gate != [dims.hidden, dims.inter] || down != [dims.inter, dims.hidden] {
                return Err(refuse(format!(
                    "the record holds {gate:?} and {down:?} matrices; the block is {} wide with \
                     an inner width of {}",
                    dims.hidden, dims.inter
                )));
            }
            let spans = SlotSpans::new(layout, geometry.slots)?;
            let book = trunks
                .first()
                .ok_or_else(|| refuse("a block without layers".into()))?
                .shared_gate
                .view()?
                .book;
            let scratch_layout =
                ScratchLayout::new(dims.n_expert, dims.top_k, dims.hidden, dims.rows)?;
            let scratch = ctx.upload(&vec![0u8; scratch_layout.total])?;
            Ok(Self {
                trunks,
                bank,
                spans,
                bank_layout: layout.clone(),
                layout: scratch_layout,
                scratch,
                book,
                verify_finite_filler: false,
                ring: None,
            })
        }

        /// Tries to fund the prefill ring: `budget` record slots of device memory, clamped to
        /// twice the streamed experts per layer (the ideal horizon) and never above it. A ring
        /// smaller than one full layer is not usable (a layer's grouped pass reads every
        /// streamed expert at once) and is not built. Returns the slot count funded, `0` when
        /// none. Call once, after the constructor.
        pub(in crate::family::flash_next) fn plan_ring(
            &mut self,
            ctx: &CudaContext,
            budget: usize,
        ) -> Result<usize, EngError> {
            let geometry = *self.bank.geometry();
            let streamed = geometry
                .experts
                .checked_sub(geometry.slots)
                .unwrap_or(geometry.experts);
            let want = budget.min(2 * streamed).max(1);
            let mut ring =
                crate::family::flash_next::cuda::prefill_ring::device::PrefillRing::build(
                    ctx, geometry, want,
                )?;
            ring.replan(&self.bank);
            if ring.ring_slots() < ring.per_layer() {
                return Ok(0);
            }
            let slots = ring.ring_slots();
            self.ring = Some(ring);
            Ok(slots)
        }

        /// Re-plans the ring against the bank as decode left it, at the start of every prompt's
        /// chunked prefill.
        pub(in crate::family::flash_next) fn begin_prefill(&mut self) {
            if let Some(ring) = &mut self.ring {
                ring.replan(&self.bank);
                ring.set_active(true);
                ring.begin_chunk();
            }
        }

        /// Closes the prefill window: waits the ring's copies out and hands the pinned stores
        /// back to the decode path, whose miss copies run on the bank's own stream.
        pub(in crate::family::flash_next) fn end_prefill(&mut self) -> Result<(), EngError> {
            let Some(ring) = self.ring.as_mut() else {
                return Ok(());
            };
            if !ring.active() {
                return Ok(());
            }
            let layers = self.bank.geometry().layers;
            for layer in 0..layers {
                self.bank.synchronize_store(layer, ring.copy_stream())?;
            }
            ring.set_active(false);
            Ok(())
        }

        /// Resets the ring's per-chunk cursors (the chunk driver's `begin_chunk`).
        pub(in crate::family::flash_next) fn begin_chunk(&mut self) {
            if let Some(ring) = &mut self.ring {
                ring.begin_chunk();
            }
        }

        pub(in crate::family::flash_next) fn bank(&self) -> &DeviceBank {
            &self.bank
        }

        /// When set, every pass reads the down rows back and refuses a filler pick (weight +0.0)
        /// whose expert output is not finite. Off by default: it adds a read of
        /// `top_k * hidden` floats to every pass.
        pub(in crate::family::flash_next) fn verify_finite_filler(&mut self, on: bool) {
            self.verify_finite_filler = on;
        }

        /// Queues layer `layer`'s MoE block on `launch.stream`. Reads `bufs.route.norm_row` (f32
        /// `[hidden]`, the hyper-mixed row) and `bufs.route.resid_in` (a zero row); writes the
        /// block output, the weighted picks plus the gated shared expert and no residual, to
        /// `bufs.route.resid_out`. A `resid_in` that is not zero is added into the result. Returns
        /// after every launch of the block is queued. It waits on the stream for the slot row, for
        /// the classify log and, on a miss, for the bank copies. With the filler check off, a
        /// non-finite output of slot 0 makes the block output NaN without an error.
        pub(in crate::family::flash_next) fn run(
            &mut self,
            launch: &Launch<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
        ) -> Result<(), EngError> {
            if dims.rows != 1 {
                return Err(refuse(format!(
                    "the block takes one row, got {}",
                    dims.rows
                )));
            }
            let geometry = *self.bank.geometry();
            if dims.n_expert != geometry.experts
                || dims.top_k != geometry.top_k
                || layer >= geometry.layers
            {
                return Err(refuse(format!(
                    "layer {layer} with {} experts and {} picks does not fit a bank of {} layers, \
                     {} experts and {} picks",
                    dims.n_expert, dims.top_k, geometry.layers, geometry.experts, geometry.top_k
                )));
            }
            // The single row of a chunked prefill's sampling step rides the ring like the
            // chunk rows: its picks are inside the prompt's plan, and the decode miss path
            // it would otherwise take copies the same pinned stores the ring streams on
            // another stream.
            if self.ring.as_ref().is_some_and(|ring| ring.active()) {
                return self.run_rows_ring(launch, layer, bufs, dims);
            }
            let (ctx, stream) = (launch.ctx, launch.stream);
            let sc = self.layout;
            let (h, k) = (dims.hidden, dims.top_k);
            let profile = prof::on();
            let slot_row_before: Vec<i32> = self.bank.slot_row(layer)?.to_vec();
            let scratch = &self.scratch;
            let slot_ids = scratch.range(sc.slot_ids, k * I32)?;
            let slot_weights = scratch.range(sc.weights, k * F32)?;
            let partial = scratch.range(sc.partial, h * F32)?;

            prof::blocked(34, profile, || {
                scratch.write_at(stream, sc.slot_row, &le_i32(&slot_row_before))
            })?;
            let trunk = &self.trunks[layer];
            ctx.router(
                stream,
                RouterCall {
                    shape: RouterShape {
                        m: 1,
                        hidden: h,
                        n_expert: dims.n_expert,
                        top_k: k,
                    },
                    x: bufs.route.norm_row,
                    weights: RouterWeights {
                        w: whole(trunk.router)?,
                        w_shared: Some(whole(trunk.router_shared)?),
                    },
                    logits: bufs.route.logits,
                    out: RouterOut {
                        ids: bufs.route.ids,
                        weights: bufs.route.weights,
                        shared_gate: Some(bufs.route.shared_gate),
                        status: bufs.route.status,
                    },
                },
            )?;
            ctx.tiered_classify(
                stream,
                ClassifyCall {
                    shape: ClassifyShape {
                        n_expert: dims.n_expert,
                        slots: geometry.slots,
                        top_k: k,
                        hidden: h,
                        layer,
                    },
                    ids: bufs.route.ids,
                    weights: bufs.route.weights,
                    slot_row: scratch.range(sc.slot_row, dims.n_expert * I32)?,
                    norm_row: bufs.route.norm_row,
                    slot_ids_out: slot_ids,
                    weights_out: slot_weights,
                    log_out: scratch.range(sc.log, k * LOG_ENTRY)?,
                    ctrl: scratch.range(sc.ctrl, CTRL_BYTES)?,
                    word: scratch.range(sc.word, I32)?,
                    mbox_req: scratch.range(sc.mbox_req, REQUEST_BYTES)?,
                    mbox_row: scratch.range(sc.mbox_row, h * F32)?,
                },
            )?;
            let mut raw = vec![0u8; k * LOG_ENTRY];
            prof::blocked(35, profile, || scratch.read_at(stream, sc.log, &mut raw))?;
            let (ids, weights) = parse_log(&raw, k)?;
            let classified = classify_picks(&slot_row_before, geometry.slots, &ids, &weights)?;

            // The resident pass. With a miss its result waits in the partial buffer; without one
            // it is the block's output.
            let first_out = if classified.misses.is_empty() {
                bufs.route.resid_out
            } else {
                partial
            };
            self.pass(
                launch,
                bufs,
                dims,
                Pass {
                    layer,
                    ids: slot_ids,
                    weights: slot_weights,
                    shared: true,
                    res: bufs.route.resid_in,
                    out: first_out,
                },
            )?;
            if self.verify_finite_filler {
                self.check_filler(launch, bufs, dims, 0, &classified.weights)?;
            }

            // The copies for every pick, hits included; they run behind the resident pass.
            self.bank.step(ctx, layer, &ids)?;
            let slot_row_after = self.bank.slot_row(layer)?.to_vec();
            let second = plan_second_pass(&ids, &weights, &classified, &slot_row_after)?;
            if let Some((miss_slots, miss_weights)) = second.miss {
                let scratch = &self.scratch;
                prof::blocked(36, profile, || {
                    scratch.write_at(stream, sc.miss_ids, &le_i32(&miss_slots))?;
                    scratch.write_at(stream, sc.miss_weights, &le_f32(&miss_weights))
                })?;
                self.pass(
                    launch,
                    bufs,
                    dims,
                    Pass {
                        layer,
                        ids: scratch.range(sc.miss_ids, k * I32)?,
                        weights: scratch.range(sc.miss_weights, k * F32)?,
                        shared: false,
                        res: scratch.range(sc.partial, h * F32)?,
                        out: bufs.route.resid_out,
                    },
                )?;
                if self.verify_finite_filler {
                    self.check_filler(launch, bufs, dims, 0, &miss_weights)?;
                }
            }
            Ok(())
        }

        /// The buffers of row `r` of a multi-row chunk, as the one-row [`Self::run`] sees them:
        /// the miss pass of `run_rows` reuses `pass` over them.
        fn row_of<'b>(
            bufs: &Exl3MoeBufs<'b>,
            dims: &MoeDims,
            r: usize,
        ) -> Result<Exl3MoeBufs<'b>, EngError> {
            let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
            Ok(Exl3MoeBufs {
                route: RouteBufs {
                    resid_in: row(&bufs.route.resid_in, h * F32, r)?,
                    resid_out: row(&bufs.route.resid_out, h * F32, r)?,
                    norm_row: row(&bufs.route.norm_row, h * F32, r)?,
                    logits: row(&bufs.route.logits, (dims.n_expert + 1) * F32, r)?,
                    ids: row(&bufs.route.ids, k * F32, r)?,
                    weights: row(&bufs.route.weights, k * F32, r)?,
                    shared_gate: row(&bufs.route.shared_gate, F32, r)?,
                    status: bufs.route.status,
                },
                expert_gu: row(&bufs.expert_gu, k * 2 * i * F32, r)?,
                expert_act: row(&bufs.expert_act, k * i * F32, r)?,
                expert_down: row(&bufs.expert_down, k * h * F32, r)?,
                shared_gu: row(&bufs.shared_gu, 2 * dims.shared_inter * F32, r)?,
                shared_act: row(&bufs.shared_act, dims.shared_inter * F32, r)?,
                shared_out: row(&bufs.shared_out, h * F32, r)?,
                scratch: bufs.scratch,
            })
        }

        /// One router launch over every row of the chunk, then one read-back: `(ids, weights)`
        /// per row. Shared by the admit path and the ring path of `run_rows`.
        fn route_rows(
            &self,
            launch: &Launch<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
        ) -> Result<RoutedPicks, EngError> {
            let m = dims.rows;
            let (h, k, e) = (dims.hidden, dims.top_k, dims.n_expert);
            let (ctx, stream) = (launch.ctx, launch.stream);
            let trunk = &self.trunks[layer];
            ctx.router_rows(
                stream,
                RouterCall {
                    shape: RouterShape {
                        m,
                        hidden: h,
                        n_expert: e,
                        top_k: k,
                    },
                    x: bufs.route.norm_row,
                    weights: RouterWeights {
                        w: whole(trunk.router)?,
                        w_shared: Some(whole(trunk.router_shared)?),
                    },
                    logits: bufs.route.logits,
                    out: RouterOut {
                        ids: bufs.route.ids,
                        weights: bufs.route.weights,
                        shared_gate: Some(bufs.route.shared_gate),
                        status: bufs.route.status,
                    },
                },
            )?;
            let mut raw_ids = vec![0u8; m * k * I32];
            let profile = prof::on();
            let raw_weights = prof::blocked(37, profile, || {
                bufs.route
                    .ids
                    .buffer()
                    .read_at(stream, bufs.route.ids.offset(), &mut raw_ids)?;
                let mut raw_weights = vec![0u8; m * k * F32];
                bufs.route.weights.buffer().read_at(
                    stream,
                    bufs.route.weights.offset(),
                    &mut raw_weights,
                )?;
                Ok::<_, EngError>(raw_weights)
            })?;
            let mut picks: Vec<(Vec<i32>, Vec<f32>)> = Vec::with_capacity(m);
            for r in 0..m {
                let ids: Vec<i32> = raw_ids[r * k * I32..][..k * I32]
                    .as_chunks::<I32>()
                    .0
                    .iter()
                    .map(|bytes| i32::from_le_bytes(*bytes))
                    .collect();
                if ids.iter().any(|&id| !(0..e as i32).contains(&id)) {
                    return Err(refuse(
                        "the router picked no valid expert (non-finite logits?)".into(),
                    ));
                }
                let weights: Vec<f32> = raw_weights[r * k * F32..][..k * F32]
                    .as_chunks::<I32>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect();
                picks.push((ids, weights));
            }
            Ok(picks)
        }

        /// The MoE block of a chunk of `dims.rows` rows (`2..=MAX_STEP_ROWS`), the batched shape of
        /// [`Self::run`]: one router launch over every row, then — when the chunk's distinct
        /// picks fit the bank and any of them is missing — one `admit` of their union, so
        /// every row's picks ride the flat resident pass in grouped launches with the shared
        /// expert and the first combine over all rows, and no per-row tail runs (the classify's
        /// chunk-initial slot row is the post-admit row; an already-resident union skips the
        /// admit too). When the union overflows the bank (small-slot banks), the union admit
        /// is skipped and, one row at a time, that row's bank copies and miss pass run against
        /// the chunk-initial slot row instead.
        ///
        /// Ordering: the bank's copies run on the bank's own stream, unordered against this
        /// stream. The union `admit` copies behind same-layer passes of earlier chunks that
        /// may still be queued here, and, on the fallback path, each row's `step` picks victim
        /// slots this chunk's own passes read; each is preceded by a full drain of this stream,
        /// which serializes execution (one drain per chunk, or per dirty row).
        /// On the fallback a pick that was a chunk-initial hit but was evicted by an earlier
        /// row's step is re-admitted by this row's step (one wasted copy); its contribution
        /// stays in the first pass, so the chunked sums can associate differently than the
        /// one-row-at-a-time block — within the two-pass tolerance, like a miss.
        pub(in crate::family::flash_next) fn run_rows(
            &mut self,
            launch: &Launch<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
        ) -> Result<(), EngError> {
            let m = dims.rows;
            if !(2..=MAX_STEP_ROWS).contains(&m) {
                return Err(refuse(format!(
                    "the block takes 2 to {MAX_STEP_ROWS} rows, got {m}"
                )));
            }
            if m > self.layout.rows {
                return Err(refuse(format!(
                    "the block plans {} rows, got {m}",
                    self.layout.rows
                )));
            }
            let geometry = *self.bank.geometry();
            if dims.n_expert != geometry.experts
                || dims.top_k != geometry.top_k
                || layer >= geometry.layers
            {
                return Err(refuse(format!(
                    "layer {layer} with {} experts and {} picks does not fit a bank of {} layers, \
                     {} experts and {} picks",
                    dims.n_expert, dims.top_k, geometry.layers, geometry.experts, geometry.top_k
                )));
            }
            if self.ring.is_some() {
                return self.run_rows_ring(launch, layer, bufs, dims);
            }
            let (ctx, stream) = (launch.ctx, launch.stream);
            let sc = self.layout;
            let (h, i, k, e) = (dims.hidden, dims.inter, dims.top_k, dims.n_expert);
            let scratch = &self.scratch;
            let slot_ids = scratch.range(sc.slot_ids, m * k * I32)?;
            let slot_weights = scratch.range(sc.weights, m * k * F32)?;
            let partial = scratch.range(sc.partial, m * h * F32)?;

            // One router over every row; its picks drive the host bank, so they come back once.
            let profile = prof::on();
            let mut t = std::time::Instant::now();
            let mut tick = |slot: usize| {
                if profile {
                    prof::add(slot, t.elapsed());
                    t = std::time::Instant::now();
                }
            };
            let picks = self.route_rows(launch, layer, bufs, dims)?;
            tick(11);

            // Chunk-admit: when the chunk's distinct picks fit the bank, admit their union in
            // one pass before the classify — every row's picks then ride the flat resident
            // pass and no per-row tail runs. Nothing before this point reads the bank's
            // slots, but the drain keeps the ordering rule (the copies are unordered against
            // this stream) uniform with `step`.
            let picks_rows: Vec<&[i32]> = picks.iter().map(|(ids, _)| ids.as_slice()).collect();
            let union = pick_union(&picks_rows, e);
            let fits = union.len() <= geometry.slots;
            let missing = {
                let slot_row_now = self.bank.slot_row(layer)?;
                union
                    .iter()
                    .filter(|&&id| slot_row_now[usize::from(id)] < 0)
                    .count()
            };
            if fits && missing > 0 {
                launch.stream.synchronize()?;
                self.bank.admit(ctx, layer, &union)?;
            }
            tick(12);

            // One classify per row against the slot row after the admit above: a pick another
            // row of the chunk missed still rides this row's own second pass.
            let slot_row: Vec<i32> = self.bank.slot_row(layer)?.to_vec();

            prof::blocked(38, profile, || {
                scratch.write_at(stream, sc.slot_row, &le_i32(&slot_row))
            })?;
            let mut classified = Vec::with_capacity(m);
            for (r, pick) in picks.iter().enumerate() {
                ctx.tiered_classify(
                    stream,
                    ClassifyCall {
                        shape: ClassifyShape {
                            n_expert: e,
                            slots: geometry.slots,
                            top_k: k,
                            hidden: h,
                            layer,
                        },
                        ids: row(&bufs.route.ids, k * I32, r)?,
                        weights: row(&bufs.route.weights, k * F32, r)?,
                        slot_row: scratch.range(sc.slot_row, e * I32)?,
                        norm_row: row(&bufs.route.norm_row, h * F32, r)?,
                        slot_ids_out: scratch.range(sc.slot_ids + r * k * I32, k * I32)?,
                        weights_out: scratch.range(sc.weights + r * k * F32, k * F32)?,
                        log_out: scratch.range(sc.log, k * LOG_ENTRY)?,
                        ctrl: scratch.range(sc.ctrl, CTRL_BYTES)?,
                        word: scratch.range(sc.word, I32)?,
                        mbox_req: scratch.range(sc.mbox_req, REQUEST_BYTES)?,
                        mbox_row: scratch.range(sc.mbox_row, h * F32)?,
                    },
                )?;
                classified.push(classify_picks(&slot_row, geometry.slots, &pick.0, &pick.1)?);
            }
            // Every pick resident against the chunk-initial row: the union admit made the
            // whole chunk clean, the common case after `a688f3af9`.
            let all_clean = classified.iter().all(|c| c.misses.is_empty());
            tick(28);

            // The resident pass of every row in flat launches: pick `r * k + p` indexes the
            // flat slot ids the classifies wrote.
            let mut gu = Vec::with_capacity(2 * m * k);
            let mut down = Vec::with_capacity(m * k);
            for r in 0..m {
                for pick in 0..k {
                    for part in 0..2usize {
                        gu.push(Item {
                            part,
                            pick: (r * k + pick) as u32,
                            x: row(&bufs.route.norm_row, h * F32, r)?,
                            out: row(&bufs.expert_gu, i * F32, 2 * (r * k + pick) + part)?,
                        });
                    }
                    down.push(Item {
                        part: 2,
                        pick: (r * k + pick) as u32,
                        x: row(&bufs.expert_act, i * F32, r * k + pick)?,
                        out: row(&bufs.expert_down, h * F32, r * k + pick)?,
                    });
                }
            }
            self.grouped(launch, layer, &gu, (slot_ids, &bufs.scratch))?;
            tick(29);
            ctx.swiglu_rows(
                stream,
                SwigluRowsCall {
                    shape: SwigluRowsShape {
                        rows: m * k,
                        n_ff: i,
                        up_off: i,
                        ld_in: 2 * i,
                        ld_out: i,
                    },
                    gu: bufs.expert_gu,
                    act: bufs.expert_act,
                },
            )?;
            self.grouped(launch, layer, &down, (slot_ids, &bufs.scratch))?;
            tick(30);
            if self.verify_finite_filler {
                for (r, class) in classified.iter().enumerate() {
                    self.check_filler(launch, bufs, dims, r, &class.weights)?;
                }
            }

            // The shared expert of every row, then the first combine over the resident pass.
            shared_rows(
                launch,
                &SharedWeights {
                    gate: self.trunks[layer].shared_gate,
                    up: self.trunks[layer].shared_up,
                    down: self.trunks[layer].shared_down,
                },
                &SharedRowsIo {
                    norm_row: bufs.route.norm_row,
                    gu: bufs.shared_gu,
                    act: bufs.shared_act,
                    out: bufs.shared_out,
                    scratch: bufs.scratch,
                },
                (h, dims.shared_inter, m),
            )?;
            ctx.moe_combine(
                stream,
                CombineCall {
                    shape: CombineShape {
                        m,
                        n: h,
                        top_k: k,
                        ld_down: h,
                    },
                    res: bufs.route.resid_in,
                    down: bufs.expert_down,
                    weights: slot_weights,
                    shared: Some(SharedTerm {
                        gate: bufs.route.shared_gate,
                        row: bufs.shared_out,
                    }),
                    out: if all_clean {
                        // An all-clean chunk has no second pass, so the resident sum is the
                        // block output: write it straight to `resid_out` and skip the per-row
                        // tail (8 zero-weight writes plus 8 m=1 combines per layer-chunk).
                        CombineOut::To(bufs.route.resid_out)
                    } else {
                        CombineOut::To(partial)
                    },
                },
            )?;
            tick(31);

            // Per row: the copies for its picks, then its miss pass — or, when every pick was
            // resident, the one combine that moves the resident result to the block output.
            // An all-clean chunk skips the loop whole: every tail combine would only copy
            // `partial` to `resid_out` at zero miss weights.
            for r in 0..m {
                if all_clean {
                    break;
                }
                if classified[r].misses.is_empty() {
                    prof::blocked(39, profile, || {
                        scratch.write_at(
                            stream,
                            sc.miss_weights + r * k * F32,
                            &le_f32(&vec![0.0; k]),
                        )
                    })?;
                    ctx.moe_combine(
                        stream,
                        CombineCall {
                            shape: CombineShape {
                                m: 1,
                                n: h,
                                top_k: k,
                                ld_down: h,
                            },
                            res: row(&partial, h * F32, r)?,
                            down: row(&bufs.expert_down, k * h * F32, r)?,
                            weights: scratch.range(sc.miss_weights + r * k * F32, k * F32)?,
                            shared: None,
                            out: CombineOut::To(row(&bufs.route.resid_out, h * F32, r)?),
                        },
                    )?;
                    continue;
                }
                // The copies overwrite victim slots this chunk's flat pass (or an earlier
                // row's miss pass) may still be reading, and they run on the bank's own
                // stream: drain this stream first. `step` itself returns once its copies
                // are done, so execution is serialized row by row.
                launch.stream.synchronize()?;
                self.bank.step(ctx, layer, &picks[r].0)?;
                let slot_row_after = self.bank.slot_row(layer)?.to_vec();
                let second =
                    plan_second_pass(&picks[r].0, &picks[r].1, &classified[r], &slot_row_after)?;
                tick(32);
                if let Some((miss_slots, miss_weights)) = second.miss {
                    prof::blocked(39, profile, || {
                        scratch.write_at(
                            stream,
                            sc.miss_ids + r * k * I32,
                            &le_i32(&miss_slots),
                        )?;
                        scratch.write_at(
                            stream,
                            sc.miss_weights + r * k * F32,
                            &le_f32(&miss_weights),
                        )
                    })?;
                    let row_bufs = Self::row_of(bufs, dims, r)?;
                    self.pass(
                        launch,
                        &row_bufs,
                        dims,
                        Pass {
                            layer,
                            ids: scratch.range(sc.miss_ids + r * k * I32, k * I32)?,
                            weights: scratch.range(sc.miss_weights + r * k * F32, k * F32)?,
                            shared: false,
                            res: row(&partial, h * F32, r)?,
                            out: row(&bufs.route.resid_out, h * F32, r)?,
                        },
                    )?;
                    if self.verify_finite_filler {
                        self.check_filler(launch, &row_bufs, dims, 0, &miss_weights)?;
                    }
                }
            }
            tick(33);
            tick(13);
            Ok(())
        }

        /// Reads the down rows back and refuses a non-finite row whose pick weight is zero.
        /// `r` indexes the rows of a multi-row chunk (`0` for the single-row block).
        fn check_filler(
            &self,
            launch: &Launch<'_>,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
            r: usize,
            weights: &[f32],
        ) -> Result<(), EngError> {
            let (h, k) = (dims.hidden, dims.top_k);
            let profile = prof::on();
            let mut raw = vec![0u8; k * h * F32];
            prof::blocked(42, profile, || {
                bufs.expert_down.buffer().read_at(
                    launch.stream,
                    bufs.expert_down.offset() + r * k * h * F32,
                    &mut raw,
                )
            })?;
            match first_poisoned_filler(&raw, h, weights) {
                Some(pick) => Err(refuse(format!(
                    "the filler expert (slot 0) gave a non-finite output on pick {pick}; \
                     0.0 times it would poison the sum"
                ))),
                None => Ok(()),
            }
        }

        /// The MoE block of a chunk over the prefill ring: every pick reads either its bank
        /// slot (the bank is frozen read-only for the whole prefill) or the ring slot the copy
        /// stream filled, so there is no admit, no classify and no miss pass — one job-table
        /// launch per source per part, the shared expert and one combine straight to the block
        /// output. At the start of layer `l`'s half the ring issues layer `l + 1`'s experts, so
        /// the copies run behind this half's compute and the next layer's attention.
        ///
        /// Ordering: the ring's copies run on its own copy stream; the compute stream waits
        /// the `copied` events of this layer's entries before the grouped launches and records
        /// their `used` events after, the same discipline as the bank's tail. The router's
        /// read-back drains the compute stream once per layer; the copies keep streaming
        /// through it.
        ///
        /// Accuracy: a streamed record's bits equal its banked bits (the same pinned-store
        /// bytes), and the combine runs once over every pick, so the sum's association is the
        /// resident reference's. The launch family differs, though: this branch's job-table
        /// tile kernels stage the activation as F16 operands (2^-11 operand error), the flat
        /// launches the resident reference runs do not — the ring device test bounds that
        /// difference, which no source split or pick ordering can move.
        fn run_rows_ring(
            &mut self,
            launch: &Launch<'_>,
            layer: usize,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
        ) -> Result<(), EngError> {
            let m = dims.rows;
            let (ctx, stream) = (launch.ctx, launch.stream);
            let sc = self.layout;
            if m > sc.rows {
                return Err(refuse(format!("the block plans {} rows, got {m}", sc.rows)));
            }
            let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
            let geometry = *self.bank.geometry();
            let scratch = &self.scratch;

            // The profiler's ring taps walk this body in call order; the route read-back is
            // the one drain of the layer, so it also absorbs the prior layer's queued compute.
            let profile = prof::on();
            let mut t = std::time::Instant::now();
            let mut tick = |slot: usize| {
                if profile {
                    prof::add(slot, t.elapsed());
                    t = std::time::Instant::now();
                }
            };

            // The next layer's experts stream now, behind this half's compute.
            let ring_slots = self
                .ring
                .as_ref()
                .expect("the ring branch is taken only with a ring")
                .ring_slots();
            self.ring
                .as_mut()
                .expect("the ring branch is taken only with a ring")
                .issue_through(ctx, &mut self.bank, layer + 1)?;
            tick(22);

            let picks = self.route_rows(launch, layer, bufs, dims)?;
            let slot_row: Vec<i32> = self.bank.slot_row(layer)?.to_vec();
            tick(23);

            // Host classification against the frozen row: every pick keeps a slot id in its
            // own source — the bank's slot for a resident pick, the ring's slot otherwise.
            // One ids array serves both: the indexed launch reads `ids[pick]` against the
            // members of the item's own source (the kernel skips values out of range).
            let mut ids = vec![0i32; m * k];
            let mut weights = vec![0f32; m * k];
            let mut is_ring = vec![false; m * k];
            for (r, (row_ids, row_weights)) in picks.iter().enumerate() {
                classify_picks(&slot_row, geometry.slots, row_ids, row_weights)?;
                for (p, &id) in row_ids.iter().enumerate() {
                    let at = r * k + p;
                    weights[at] = row_weights[p];
                    match slot_row[id as usize] {
                        slot if slot >= 0 => ids[at] = slot,
                        _ => {
                            let slot = self
                                .ring
                                .as_ref()
                                .and_then(|ring| ring.slot_of(layer, id as u16))
                                .ok_or_else(|| {
                                    refuse(format!(
                                        "expert {id} of layer {layer} has neither a bank slot nor a ring slot"
                                    ))
                                })?;
                            ids[at] = i32::from(slot);
                            is_ring[at] = true;
                        }
                    }
                }
            }
            prof::blocked(40, profile, || {
                scratch.write_at(stream, sc.weights, &le_f32(&weights))
            })?;
            tick(24);

            // The compute waits this layer's copies before any job reads a ring slot.
            prof::blocked(41, profile, || {
                self.ring
                    .as_ref()
                    .expect("the ring branch is taken only with a ring")
                    .wait_layer(stream, layer)
            })?;
            // The routed job table of both passes: the chunk's pairs are sorted per source —
            // the bank's slots and the ring's slots are two member banks. Gate and up read
            // the chunk rows through the first list and write disjoint segments of the
            // gate/up face, the SwiGLU covers each segment, down writes each pair row to
            // its place through the second list. Every part is one launch per source
            // whatever the number of experts, in place of one indexed launch per sixteen
            // picks.
            let mut sub = [Vec::new(), Vec::new()];
            for (at, &to_ring) in is_ring.iter().enumerate() {
                sub[usize::from(to_ring)].push(at);
            }
            let slots_of = [geometry.slots, ring_slots];
            let plans: Vec<_> = (0..2)
                .map(|source| source_plan(&ids, &sub[source], slots_of[source], k))
                .collect::<Result<Vec<_>, EngError>>()?;
            // Each source's lists and job table sit at their own offset of the perm region,
            // and its gate/up and SwiGLU rows form their own segment of the pair faces.
            let (at_of, seg_of) =
                source_offsets(sc.perm, [&plans[0], &plans[1]], [&sub[0], &sub[1]]);
            let ring_of = self
                .ring
                .as_ref()
                .expect("the ring branch is taken only with a ring");
            let ring_spans = SlotSpans::new(&self.bank_layout, ring_slots)?;
            let sources = [
                (self.bank.slots(layer)?, &self.spans),
                (ring_of.slots(), &ring_spans),
            ];
            let pair = ctx.exl3_jobs_pair_enabled()?;
            // Gate and up of every source, each followed by the SwiGLU of its segment.
            for (source, (buffer, spans)) in sources.iter().enumerate() {
                let n = sub[source].len();
                if n == 0 {
                    continue;
                }
                let plan = &plans[source];
                let jobs = plan_jobs(&plan.counts);
                let perm =
                    scratch.range(at_of[source], (2 * n + EXL3_JOB_WORDS * jobs.len()) * I32)?;
                write_lists(launch, perm, (plan, &jobs))?;
                let listed = Lists {
                    input: Some(scratch.range(perm.offset(), n * I32)?),
                    output: None,
                    x_rows: m,
                    out_rows: n,
                };
                let table = Table {
                    jobs: &jobs,
                    range: scratch.range(perm.offset() + 2 * n * I32, jobs.len() * 16)?,
                    rows: n,
                };
                let seg = rows_of(&bufs.expert_gu, 2 * i * F32, seg_of[source], n)?;
                let act = rows_of(&bufs.expert_act, i * F32, seg_of[source], n)?;
                let rotated = JobScratch {
                    xh: rows_of(&bufs.scratch.xh, h * F32, 0, n)?,
                    z: rows_of(&bufs.scratch.z, i * F32, 0, n)?,
                };
                let members = [
                    self.members_of(buffer, spans, 0)?,
                    self.members_of(buffer, spans, 1)?,
                ];
                let fused = if pair {
                    pair_scratch_of(&bufs.scratch, n, i, h)?
                } else {
                    None
                };
                match fused {
                    Some((first, second)) => {
                        let calls = (
                            jobs_call(
                                bufs.scratch.flag,
                                first,
                                table,
                                &members[0],
                                bufs.route.norm_row,
                                (seg, 2 * i, 0),
                                listed,
                            )?,
                            jobs_call(
                                bufs.scratch.flag,
                                second,
                                table,
                                &members[1],
                                bufs.route.norm_row,
                                (seg, 2 * i, i),
                                listed,
                            )?,
                        );
                        ctx.exl3_jobs_pair_mode(stream, calls, (true, None))?;
                    }
                    None => {
                        project_jobs(
                            launch,
                            bufs.scratch.flag,
                            rotated,
                            table,
                            &members[0],
                            bufs.route.norm_row,
                            (seg, 2 * i, 0),
                            listed,
                        )?;
                        project_jobs(
                            launch,
                            bufs.scratch.flag,
                            rotated,
                            table,
                            &members[1],
                            bufs.route.norm_row,
                            (seg, 2 * i, i),
                            listed,
                        )?;
                    }
                }
                ctx.swiglu_rows(
                    stream,
                    SwigluRowsCall {
                        shape: SwigluRowsShape {
                            rows: n,
                            n_ff: i,
                            up_off: i,
                            ld_in: 2 * i,
                            ld_out: i,
                        },
                        gu: seg,
                        act,
                    },
                )?;
            }
            tick(25);
            // Down of every source, each pair row written to its place through the scatter
            // list the combine then reads in pick order.
            for (source, (buffer, spans)) in sources.iter().enumerate() {
                let n = sub[source].len();
                if n == 0 {
                    continue;
                }
                let plan = &plans[source];
                let jobs = plan_jobs(&plan.counts);
                let perm =
                    scratch.range(at_of[source], (2 * n + EXL3_JOB_WORDS * jobs.len()) * I32)?;
                let scatter = scratch.range(perm.offset() + n * I32, n * I32)?;
                let table = Table {
                    jobs: &jobs,
                    range: scratch.range(perm.offset() + 2 * n * I32, jobs.len() * 16)?,
                    rows: n,
                };
                let act = rows_of(&bufs.expert_act, i * F32, seg_of[source], n)?;
                let listed = Lists {
                    input: None,
                    output: Some(scatter),
                    x_rows: n,
                    out_rows: m * k,
                };
                project_jobs(
                    launch,
                    bufs.scratch.flag,
                    JobScratch {
                        xh: rows_of(&bufs.scratch.xh, h * F32, 0, n)?,
                        z: rows_of(&bufs.scratch.z, h * F32, 0, n)?,
                    },
                    table,
                    &self.members_of(buffer, spans, 2)?,
                    act,
                    (bufs.expert_down, h, 0),
                    listed,
                )?;
            }
            tick(26);

            // The shared expert of every row, then the one combine over every pick.
            let shared_weights = SharedWeights {
                gate: self.trunks[layer].shared_gate,
                up: self.trunks[layer].shared_up,
                down: self.trunks[layer].shared_down,
            };
            if m > 1 {
                shared_rows(
                    launch,
                    &shared_weights,
                    &SharedRowsIo {
                        norm_row: bufs.route.norm_row,
                        gu: bufs.shared_gu,
                        act: bufs.shared_act,
                        out: bufs.shared_out,
                        scratch: bufs.scratch,
                    },
                    (h, dims.shared_inter, m),
                )?;
            } else {
                // The K-row shared expert takes at least 2 rows; the sampling row runs the
                // one-row variant over its single-row buffers.
                shared_of(launch, &shared_weights, bufs, dims)?;
            }
            ctx.moe_combine(
                stream,
                CombineCall {
                    shape: CombineShape {
                        m,
                        n: h,
                        top_k: k,
                        ld_down: h,
                    },
                    res: bufs.route.resid_in,
                    down: bufs.expert_down,
                    weights: scratch.range(sc.weights, m * k * F32)?,
                    shared: Some(SharedTerm {
                        gate: bufs.route.shared_gate,
                        row: bufs.shared_out,
                    }),
                    out: CombineOut::To(bufs.route.resid_out),
                },
            )?;

            // The slots of this layer free for the copies behind us.
            self.ring
                .as_mut()
                .expect("the ring branch is taken only with a ring")
                .free_layer(stream, layer)?;
            tick(27);
            Ok(())
        }

        /// One id-indexed launch group over the bank buffer of `layer`, chunked at the launch
        /// capacity; every chunk starts again at row 0 of the scratch.
        fn grouped_at<'a>(
            &'a self,
            launch: &Launch<'_>,
            items: &[Item<'a>],
            bank: (&DeviceBuffer, u32),
            ids: DeviceRange<'_>,
            scratch: &Scratch<'a>,
        ) -> Result<(), EngError> {
            let first = items
                .first()
                .ok_or_else(|| refuse("an indexed launch without pairs".into()))?;
            let [k, n] = self.spans.parts[first.part].shape;
            let mut pairs = Vec::with_capacity(items.len());
            for item in items {
                let (weights, members, spans) = (bank.0, bank.1, &self.spans);
                let span = &spans.parts[item.part];
                if span.shape != [k, n] {
                    return Err(refuse("a launch group mixes matrix shapes".into()));
                }
                pairs.push(IndexedPair {
                    span: Exl3BankSpan::from_parts(
                        weights.range(span.trellis.start, span.trellis.len())?,
                        weights.range(span.suh.start, span.suh.len())?,
                        weights.range(span.svh.start, span.svh.len())?,
                        members,
                        self.spans.trellis_stride,
                        self.spans.scale_stride,
                        span.shape,
                        span.bits_twice,
                        self.book,
                    ),
                    slot: item.pick,
                    x: item.x,
                    out: item.out,
                });
            }
            grouped_indexed(launch, &pairs, ids, scratch)
        }

        /// One id-indexed launch group over the bank buffer of `layer`, chunked at the launch
        /// capacity; every chunk starts again at row 0 of the scratch.
        fn grouped<'a>(
            &'a self,
            launch: &Launch<'_>,
            layer: usize,
            items: &[Item<'a>],
            (ids, scratch): (DeviceRange<'a>, &Scratch<'a>),
        ) -> Result<(), EngError> {
            let members = u32::try_from(self.spans.slots)
                .map_err(|_| refuse("the bank has too many slots".into()))?;
            self.grouped_at(
                launch,
                items,
                (self.bank.slots(layer)?, members),
                ids,
                scratch,
            )
        }

        /// The members of part `part` over one source of expert records — the bank's layer
        /// buffer or the ring's staging — as the job table addresses them: member `i`'s part
        /// matrices sit at the part's offsets of the record plus `i` record strides. The
        /// strides of the shared struct count u16 elements (trellis) and f32 values (scales).
        fn members_of<'b>(
            &self,
            buffer: &'b DeviceBuffer,
            spans: &SlotSpans,
            part: usize,
        ) -> Result<Exl3Members<'b>, EngError> {
            let span = &spans.parts[part];
            if !spans.trellis_stride.is_multiple_of(2) {
                return Err(refuse(
                    "the bank's record stride is not a whole u16 count".into(),
                ));
            }
            Ok(Exl3Members {
                trellis: buffer.range(span.trellis.start, span.trellis.len())?,
                suh: buffer.range(span.suh.start, span.suh.len())?,
                svh: buffer.range(span.svh.start, span.svh.len())?,
                trellis_stride: spans.trellis_stride / 2,
                scale_stride: spans.scale_stride,
                members: spans.slots,
                shape: span.shape,
                bits_twice: span.bits_twice,
                book: self.book,
            })
        }

        /// One pass over the bank of `pass.layer`: gate and up of the picks, SwiGLU, down, the
        /// shared expert when asked, and the combine of `pass.weights` on top of `pass.res`.
        fn pass(
            &self,
            launch: &Launch<'_>,
            bufs: &Exl3MoeBufs<'_>,
            dims: &MoeDims,
            pass: Pass<'_>,
        ) -> Result<(), EngError> {
            let (h, i, k) = (dims.hidden, dims.inter, dims.top_k);
            let mut items = Vec::with_capacity(2 * k);
            for pick in 0..k {
                for part in 0..2usize {
                    items.push(Item {
                        part,
                        pick: pick as u32,
                        x: bufs.route.norm_row,
                        out: row(&bufs.expert_gu, i * F32, 2 * pick + part)?,
                    });
                }
            }
            self.grouped(launch, pass.layer, &items, (pass.ids, &bufs.scratch))?;
            launch.ctx.swiglu_rows(
                launch.stream,
                SwigluRowsCall {
                    shape: SwigluRowsShape {
                        rows: k,
                        n_ff: i,
                        up_off: i,
                        ld_in: 2 * i,
                        ld_out: i,
                    },
                    gu: bufs.expert_gu,
                    act: bufs.expert_act,
                },
            )?;
            let mut items = Vec::with_capacity(k);
            for pick in 0..k {
                items.push(Item {
                    part: 2,
                    pick: pick as u32,
                    x: row(&bufs.expert_act, i * F32, pick)?,
                    out: row(&bufs.expert_down, h * F32, pick)?,
                });
            }
            self.grouped(launch, pass.layer, &items, (pass.ids, &bufs.scratch))?;
            let shared = if pass.shared {
                let trunk = &self.trunks[pass.layer];
                shared_of(
                    launch,
                    &SharedWeights {
                        gate: trunk.shared_gate,
                        up: trunk.shared_up,
                        down: trunk.shared_down,
                    },
                    bufs,
                    dims,
                )?;
                Some(SharedTerm {
                    gate: bufs.route.shared_gate,
                    row: bufs.shared_out,
                })
            } else {
                None
            };
            launch.ctx.moe_combine(
                launch.stream,
                CombineCall {
                    shape: CombineShape {
                        m: 1,
                        n: h,
                        top_k: k,
                        ld_down: h,
                    },
                    res: pass.res,
                    down: bufs.expert_down,
                    weights: pass.weights,
                    shared,
                    out: CombineOut::To(pass.out),
                },
            )
        }
    }
}

/// The distinct expert ids of a chunk's picks, first use first. The order only matters when
/// the union overflows the bank (the oldest uses stay); on the admit path it always fits.
pub(in crate::family::flash_next) fn pick_union(rows: &[&[i32]], n_expert: usize) -> Vec<u16> {
    let mut seen = vec![false; n_expert];
    let mut union = Vec::with_capacity(rows.first().map_or(0, |row| row.len()));
    for row in rows {
        for &id in *row {
            // run_rows refuses ids outside 0..n_expert before the first pick_union call.
            if !seen[id as usize] {
                seen[id as usize] = true;
                union.push(id as u16);
            }
        }
    }
    union
}
