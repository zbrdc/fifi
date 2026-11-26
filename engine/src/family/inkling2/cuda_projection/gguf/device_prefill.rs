//! Keep resident GGUF expert intermediates on the device during grouped prefill.
use crate::family::inkling2::switches::Switch;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use super::{
    DEVICE_HEADROOM, GgufCuda, bad,
    streaming::{Matrix, PrefillRoute, group_route_ids},
};
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayloadSource},
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, PinnedBuf,
        prefill::{
            mmq_mmid::{GroupedDqRows, MmidWeight, SparseDqCall, grouped_dq_serves},
            row_ops::{ClippedExpertCall, IndexedRankSumCall, RowGatherCall, RowShape},
        },
        resident::{
            dense_gemv::{
                DenseDtype, DenseGemvCall, DenseGemvGroupCounts, DenseGemvShape, dense_gemv_bytes,
            },
            matvec_q::QuantType,
        },
    },
};

// The direct packed projection measurements favor GEMV for narrow expert groups.
const WMMA_MIN_ROWS: [usize; 2] = [16, 32];

#[derive(Clone, Copy, PartialEq, Eq)]
struct Geometry {
    hidden: usize,
    middle: usize,
    rank: usize,
    grouped_experts: usize,
}

impl Geometry {
    // Each window recomputes its layout. For the same tensor shape, bounding the
    // grouped metadata also bounds the padded prefix of the device and host arenas.
    fn can_serve(self, shape: Self) -> bool {
        self.hidden == shape.hidden
            && self.middle == shape.middle
            && self.rank == shape.rank
            && self.grouped_experts >= shape.grouped_experts
    }
}

struct Layout {
    input: Range<usize>,
    row_ids: Range<usize>,
    inverse_ids: Range<usize>,
    gate_scales: Range<usize>,
    down_scales: Range<usize>,
    weights: Range<usize>,
    grouped: [Range<usize>; 2],
    flag: Range<usize>,
    output: Range<usize>,
    gathered: Range<usize>,
    gate: Range<usize>,
    middle: Range<usize>,
    down: Range<usize>,
}

impl Layout {
    fn new(rows: usize, shape: Geometry) -> Option<Self> {
        Self::for_input(rows, shape, false)
    }

    fn for_input(rows: usize, shape: Geometry, retained: bool) -> Option<Self> {
        let slots = rows.checked_mul(shape.rank)?;
        let mut offset = 0usize;
        let mut span = |width: usize, height: usize| {
            let start = offset;
            offset = start.checked_add(width.checked_mul(height)?.checked_mul(4)?)?;
            Some(start..offset)
        };
        let input = span(if retained { 0 } else { rows }, shape.hidden)?;
        let row_ids = span(slots, 1)?;
        let inverse_ids = span(slots, 1)?;
        let gate_scales = span(slots, 1)?;
        let down_scales = span(slots, 1)?;
        let weights = span(slots, 1)?;
        let grouped = if shape.grouped_experts > 0 {
            let words = slots.checked_add(shape.grouped_experts)?.checked_add(1)?;
            let spans = [span(words, 1)?, span(words, 1)?];
            // F32 vector loads in GEMM require packed operands at 16-byte alignment.
            let align_words = (3 - (spans[1].end / 4 % 4)) % 4;
            span(align_words, 1)?;
            spans
        } else {
            [0..0, 0..0]
        };
        Some(Self {
            input,
            row_ids,
            inverse_ids,
            gate_scales,
            down_scales,
            weights,
            grouped,
            flag: span(1, 1)?,
            output: span(rows, shape.hidden)?,
            gathered: span(slots, shape.hidden)?,
            gate: span(slots, shape.middle.checked_mul(2)?)?,
            middle: span(slots, shape.middle)?,
            down: span(slots, shape.hidden)?,
        })
    }

    fn fit(requested: usize, shape: Geometry, capacity: usize) -> Option<usize> {
        let mut hi = requested.min(DenseGemvShape::MAX_ACTIVATION_ROWS.checked_div(shape.rank)?);
        let mut lo = 0;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if Self::new(mid, shape).is_some_and(|layout| layout.down.end <= capacity) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        (lo >= 2).then_some(lo)
    }
}

/// Maximum device arena useful at this model's context, before driver rounding.
pub(super) fn device_workspace_bytes(spec: &super::super::TextSpec, context: usize) -> Option<u64> {
    let shape = Geometry {
        hidden: usize::try_from(spec.hidden_size).ok()?,
        middle: usize::try_from(spec.intermediate_size).ok()?,
        rank: usize::try_from(spec.num_experts_per_tok).ok()?,
        grouped_experts: 0,
    };
    let rows = context.min(DenseGemvShape::MAX_ACTIVATION_ROWS.checked_div(shape.rank)?);
    u64::try_from(Layout::new(rows, shape)?.down.end).ok()
}

#[derive(Default)]
pub(super) struct Workspace {
    scratch: Option<Scratch>,
}

struct Scratch {
    buffer: DeviceBuffer,
    stream: DeviceStream,
    host: PinnedBuf,
    /// Recorded on the caller's stream right behind the last unfenced upload of `host`; the next
    /// window waits for it before writing `host` again (nothing else is synchronized).
    uploaded: Option<DeviceEvent>,
    rows: usize,
    geometry: Geometry,
}

impl Workspace {
    fn cached(&mut self, shape: Geometry) -> Option<&mut Scratch> {
        self.scratch
            .as_mut()
            .filter(|buffer| buffer.geometry.can_serve(shape))
    }

    fn fit(
        &mut self,
        cuda: &CudaContext,
        requested: usize,
        shape: Geometry,
    ) -> Result<Option<&mut Scratch>, EngError> {
        if self
            .cached(shape)
            .is_some_and(|buffer| buffer.rows >= requested)
        {
            return Ok(self.cached(shape));
        }
        // This temporary work can consume the existing reserve above the family floor.
        // Check only after expert and optional projection placement has completed.
        let free = cuda
            .memory_info()?
            .free
            .saturating_sub(DEVICE_HEADROOM as usize);
        let Some(rows) = Layout::fit(requested, shape, free) else {
            return Ok(self.cached(shape));
        };
        if self.cached(shape).is_some_and(|buffer| buffer.rows >= rows) {
            return Ok(self.cached(shape));
        }
        let layout =
            Layout::new(rows, shape).ok_or_else(|| bad("expert prefill arena overflows"))?;
        let Some(scratch) = Scratch::allocate(cuda, rows, shape, &layout)? else {
            return Ok(self.cached(shape));
        };
        eprintln!(
            "Inkling CUDA device expert workspace {}",
            serde_json::json!({"requested_rows": requested, "admitted_rows": rows,
                "device_bytes": layout.down.end, "pinned_bytes": layout.output.end})
        );
        self.scratch = Some(scratch);
        Ok(self.scratch.as_mut())
    }
}

/// Where an unfenced window stages its metadata and publishes its flag word.
struct Unfenced<'a> {
    stream: &'a DeviceStream,
    flag_out: DeviceRange<'a>,
    destination: DeviceRange<'a>,
}

struct Projections<'a> {
    geometry: Geometry,
    caps: [f32; 2],
    dtypes: [DenseDtype; 2],
    min_wmma_rows: [Option<usize>; 2],
    exact_tile: bool,
    grouped_weights: [Option<MmidWeight<'a>>; 2],
    matrices: BTreeMap<ExpertId, [Matrix<'a>; 2]>,
}

#[derive(Clone, Copy)]
pub(in crate::family::inkling2) enum ExpertPrefillRows<'a> {
    Host(&'a [PrefillRoute<'a>]),
    Resident {
        ids: &'a [&'a [usize]],
        hidden: usize,
        activations: DeviceRange<'a>,
    },
}

#[derive(Clone, Copy)]
enum RowRoutes<'a> {
    Host(&'a [PrefillRoute<'a>]),
    Resident(&'a [&'a [usize]]),
}

impl<'a> RowRoutes<'a> {
    fn len(self) -> usize {
        match self {
            Self::Host(r) => r.len(),
            Self::Resident(ids) => ids.len(),
        }
    }

    fn ids(self) -> impl Iterator<Item = &'a [usize]> {
        (0..self.len()).map(move |i| match self {
            Self::Host(r) => r[i].ids,
            Self::Resident(ids) => ids[i],
        })
    }

    fn window(self, span: Range<usize>) -> Self {
        match self {
            Self::Host(r) => Self::Host(&r[span]),
            Self::Resident(ids) => Self::Resident(&ids[span]),
        }
    }
}

impl<'a> ExpertPrefillRows<'a> {
    fn prepare(self) -> Option<(RowRoutes<'a>, usize, Option<DeviceRange<'a>>)> {
        match self {
            Self::Host(routes) => {
                let hidden = routes.first()?.input.len();
                if hidden == 0 || routes.iter().any(|r| r.input.len() != hidden) {
                    return None;
                }
                Some((RowRoutes::Host(routes), hidden, None))
            }
            Self::Resident {
                ids,
                hidden,
                activations,
            } => {
                // There are no host rows to fall back to on a mismatched range.
                admits_router_input(ids.len(), hidden, activations.bytes()).then_some((
                    RowRoutes::Resident(ids),
                    hidden,
                    Some(activations),
                ))
            }
        }
    }
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn supports_grouped_verify(&self, layers: usize) -> bool {
        self.device_prefill.is_some()
            && self.moe_scratch.is_some()
            && !self
                .step
                .uses_q8_activations(DenseDtype::PackedGguf(crate::weights::GgufType::Q80))
            && self.resident.placement().full_layers().len() == layers
    }

    pub(in crate::family::inkling2) fn supports_grouped_prefill(&self) -> bool {
        self.has_stream_banks()
            || (self.device_prefill.is_some()
                && !self.resident.placement().full_layers().is_empty())
    }

    pub(in crate::family::inkling2) fn prefills_device_layer(&self, layer: u32) -> bool {
        self.device_prefill.is_some() && !self.resident.placement().has_cold(layer)
    }

    /// Validate every routable expert and reserve scratch before a device-only input is produced.
    pub(in crate::family::inkling2) fn admit_device_prefill(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        [rows, rank, hidden, experts]: [usize; 4],
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
    ) -> Result<bool, EngError> {
        let Some(workspace) = &self.device_prefill else {
            return Ok(false);
        };
        if !self.resident.placement().full_layers().contains(&layer)
            || experts == 0
            || !self.valid_device_prefill(rows, rank, dtypes, caps)
        {
            return Ok(false);
        }
        let Some(projections) =
            self.prepare_device_experts(source, layer, 0..experts, [rank, hidden], dtypes, caps)?
        else {
            return Ok(false);
        };
        Ok(workspace
            .borrow_mut()
            .fit(&self.context, rows, projections.geometry)?
            .is_some())
    }

    fn valid_device_prefill(
        &self,
        rows: usize,
        rank: usize,
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
    ) -> bool {
        rows >= 2
            && (1..=32).contains(&rank)
            && caps.iter().all(|cap| cap.is_finite() && *cap > 0.0)
            && dtypes
                .iter()
                .all(|&dtype| dtype.is_packed_gguf() && !self.step.uses_q8_activations(dtype))
    }

    pub(in crate::family::inkling2) fn prefill_device_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        (batch, destination): (ExpertPrefillRows<'_>, Option<DeviceRange<'_>>),
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
        weight: impl Fn(usize, usize) -> f32,
    ) -> Result<Option<Vec<Vec<f32>>>, EngError> {
        self.prefill_device_experts_impl(
            source,
            layer,
            (batch, destination),
            dtypes,
            caps,
            weight,
            None,
        )
    }

    /// The one call site of the routed-expert stage for the device-resident prefill pass
    /// (`FIFI_INKLING_CUDA_RESIDENT_PREFILL_PASS`): the same kernels as
    /// [`Self::prefill_device_experts`], enqueued on `stream` behind the caller's work, with
    /// the window metadata staged through `host` and the window's flag word copied into
    /// `flag_out` for the caller's fence. One window, a device destination, nothing read.
    pub(in crate::family::inkling2) fn prefill_device_experts_unfenced<'a>(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        (batch, destination): (ExpertPrefillRows<'_>, Option<DeviceRange<'_>>),
        (dtypes, caps): ([DenseDtype; 2], [f32; 2]),
        weight: impl Fn(usize, usize) -> f32,
        (stream, flag_out): (&'a DeviceStream, DeviceRange<'a>),
    ) -> Result<Option<()>, EngError> {
        let (batch, Some(destination)) = (batch, destination) else {
            return Ok(None);
        };
        let unfenced = Unfenced {
            stream,
            flag_out,
            destination,
        };
        self.prefill_device_experts_impl(
            source,
            layer,
            (batch, None),
            dtypes,
            caps,
            weight,
            Some(unfenced),
        )
        .map(|option| option.map(|_| ()))
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_device_experts_impl(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        (batch, destination): (ExpertPrefillRows<'_>, Option<DeviceRange<'_>>),
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
        weight: impl Fn(usize, usize) -> f32,
        unfenced: Option<Unfenced<'_>>,
    ) -> Result<Option<Vec<Vec<f32>>>, EngError> {
        let Some(workspace) = &self.device_prefill else {
            return Ok(None);
        };
        let Some((routes, hidden, retained)) = batch.prepare() else {
            return Ok(None);
        };
        if let Some(out) = destination
            && (!admits_router_input(routes.len(), hidden, out.bytes())
                || !out.offset().is_multiple_of(4))
        {
            return Err(bad(
                "incorrect CUDA expert output destination extent or alignment",
            ));
        }
        if let Some(out) = unfenced.as_ref().map(|u| u.destination)
            && (!admits_router_input(routes.len(), hidden, out.bytes())
                || !out.offset().is_multiple_of(4))
        {
            return Err(bad(
                "incorrect CUDA expert output destination extent or alignment",
            ));
        }
        let rank = routes.ids().next().map_or(0, <[usize]>::len);
        if !self.valid_device_prefill(routes.len(), rank, dtypes, caps)
            || routes.ids().any(|ids| ids.len() != rank)
        {
            return Ok(None);
        }
        let Some(projections) = self.prepare_device_experts(
            source,
            layer,
            routes.ids().flatten().copied().collect::<BTreeSet<_>>(),
            [rank, hidden],
            dtypes,
            caps,
        )?
        else {
            return Ok(None);
        };
        let cuda = &self.context;
        let mut workspace = workspace.borrow_mut();
        if unfenced.is_some()
            && workspace
                .cached(projections.geometry)
                .is_none_or(|scratch| scratch.rows < routes.len())
        {
            return Ok(None);
        }
        let Some(scratch) = workspace.fit(cuda, routes.len(), projections.geometry)? else {
            return Ok(None);
        };
        let mut output = Vec::with_capacity(if destination.is_some() {
            0
        } else {
            routes.len()
        });
        let mut counts = [DenseGemvGroupCounts::default(); 2];
        let mut grouped_slots = [0; 2];
        for first in (0..routes.len()).step_by(scratch.rows) {
            let routes = routes.window(first..(first + scratch.rows).min(routes.len()));
            // The checked full-batch extent bounds both products and every window.
            let input = retained
                .map(|batch| batch.slice(first * hidden * 4, routes.len() * hidden * 4))
                .transpose()?;
            let plan = RoutedWindow::prepare(
                &projections,
                layer,
                routes,
                retained.is_some(),
                |row, id| weight(first + row, id),
            )?;
            let target = destination
                .or(unfenced.as_ref().map(|u| u.destination))
                .map(|out| out.slice(first * hidden * 4, routes.len() * hidden * 4))
                .transpose()?;
            let (rows, launches) = match &unfenced {
                Some(u) => (
                    Vec::new(),
                    scratch.enqueue_unfenced(
                        cuda,
                        u.stream,
                        u.flag_out,
                        &projections,
                        &plan,
                        (input, target),
                    )?,
                ),
                None => scratch.execute(cuda, &projections, &plan, (input, target))?,
            };
            output.extend(rows);
            for (total, count) in counts.iter_mut().zip(launches) {
                total.wmma += count.wmma;
                total.exact_tiles += count.exact_tiles;
            }
            for (total, selection) in grouped_slots.iter_mut().zip(&plan.grouped) {
                *total += selection.as_ref().map_or(0, GroupedDqRows::selected);
            }
        }
        if self.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA device expert prefill {}",
                serde_json::json!({"layer": layer, "rows": routes.len(),
                    "windows": routes.len().div_ceil(scratch.rows),
                    "router_input_h2d_bytes_saved": retained.map_or(0, |input| input.bytes()),
                    "expert_output_d2h_bytes_avoided": destination.map_or(0, |out| out.bytes()),
                    "wmma_gate_calls": counts[0].wmma, "wmma_down_calls": counts[1].wmma,
                    "grouped_dq_gate_slots": grouped_slots[0], "grouped_dq_down_slots": grouped_slots[1],
                    "exact_tile_gate_calls": counts[0].exact_tiles,
                    "exact_tile_down_calls": counts[1].exact_tiles})
            );
        }
        Ok(Some(output))
    }

    fn prepare_device_experts<'a>(
        &'a self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        experts: impl IntoIterator<Item = usize>,
        [rank, hidden]: [usize; 2],
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
    ) -> Result<Option<Projections<'a>>, EngError> {
        let mut matrices = BTreeMap::new();
        let mut middle = 0;
        for expert in experts {
            let id = ExpertId {
                layer,
                expert: expert as u32,
            };
            let [gate, down] = self.resident_prefill_matrices(source, id)?;
            let width = down.shape[1];
            if width == 0
                || gate.scale == 0.0
                || down.scale == 0.0
                || [hidden, width] != down.shape
                || [2 * width, hidden] != gate.shape
                || (middle != 0 && width != middle)
            {
                return Ok(None);
            }
            middle = width;
            matrices.insert(id, [gate, down]);
        }
        if dense_gemv_bytes(DenseGemvShape::packed(1, 2 * middle, hidden), dtypes[0]).is_err()
            || dense_gemv_bytes(DenseGemvShape::packed(1, hidden, middle), dtypes[1]).is_err()
        {
            return Ok(None);
        }
        let shapes = [[2 * middle, hidden], [hidden, middle]];
        let (grouped_weights, grouped_experts) =
            if self.device_prefill_grouped_dq && self.device_prefill_wmma {
                let stacks = self.resident.original_layer(layer)?;
                let weights = std::array::from_fn(|stage| {
                    grouped_weight(dtypes[stage], shapes[stage], stacks[stage])
                });
                let experts = weights
                    .iter()
                    .flatten()
                    .next()
                    .map_or(0, |w| w.base.bytes() / w.stride_bytes);
                (weights, experts)
            } else {
                ([None, None], 0)
            };
        Ok(Some(Projections {
            geometry: Geometry {
                hidden,
                middle,
                rank,
                grouped_experts,
            },
            dtypes,
            min_wmma_rows: WMMA_MIN_ROWS.map(|rows| self.device_prefill_wmma.then_some(rows)),
            exact_tile: self.device_prefill_exact_tile,
            grouped_weights,
            caps,
            matrices,
        }))
    }
}

fn grouped_weight(
    dtype: DenseDtype,
    [n, k]: [usize; 2],
    base: DeviceRange<'_>,
) -> Option<MmidWeight<'_>> {
    let DenseDtype::PackedGguf(kind) = dtype else {
        return None;
    };
    let quant = QuantType::from_gguf(kind).filter(|&quant| grouped_dq_serves(quant))?;
    if !k.is_multiple_of(quant.block_size().max(64)) || !n.is_multiple_of(4) {
        return None;
    }
    let stride_bytes = quant.weight_bytes(n, k)?;
    if stride_bytes == 0
        || !base.bytes().is_multiple_of(stride_bytes)
        || !stride_bytes.is_multiple_of(2)
    {
        return None;
    }
    Some(MmidWeight {
        base,
        quant,
        k,
        n,
        stride_bytes,
    })
}

// The expert workspace is independent of the dense step arena. Consume only an
// exact whole-batch range after the producer's synchronous logits readback.
fn admits_router_input(rows: usize, hidden: usize, bytes: usize) -> bool {
    rows >= 2
        && hidden != 0
        && rows
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            == Some(bytes)
}

struct RoutedWindow {
    upload: Vec<[u8; 4]>,
    groups: Vec<(ExpertId, Range<usize>)>,
    grouped: [Option<GroupedDqRows>; 2],
    layout: Layout,
    rows: usize,
}

impl RoutedWindow {
    fn prepare(
        projections: &Projections<'_>,
        layer: u32,
        routes: RowRoutes<'_>,
        retained: bool,
        weight: impl Fn(usize, usize) -> f32,
    ) -> Result<Self, EngError> {
        let shape = projections.geometry;
        let layout = Layout::for_input(routes.len(), shape, retained)
            .ok_or_else(|| bad("expert prefill arena overflows"))?;
        let mut upload = vec![[0; 4]; layout.flag.end / 4];
        if !retained {
            let RowRoutes::Host(host) = routes else {
                return Err(bad("missing resident expert prefill input"));
            };
            for (dst, value) in upload[..layout.input.end / 4]
                .iter_mut()
                .zip(host.iter().flat_map(|route| route.input))
            {
                *dst = value.to_ne_bytes();
            }
        }
        let groups = pack_routes(
            &mut upload,
            &layout,
            shape.rank,
            group_route_ids(layer, routes.ids()),
            |id| {
                projections.matrices[&id]
                    .each_ref()
                    .map(|matrix| matrix.scale)
            },
            weight,
        )?;
        let grouped = pack_grouped_rows(projections, &layout, &groups, &mut upload);
        Ok(Self {
            upload,
            groups,
            grouped,
            layout,
            rows: routes.len(),
        })
    }
}

fn pack_grouped_rows(
    projections: &Projections<'_>,
    layout: &Layout,
    groups: &[(ExpertId, Range<usize>)],
    upload: &mut [[u8; 4]],
) -> [Option<GroupedDqRows>; 2] {
    std::array::from_fn(|stage| {
        projections.grouped_weights[stage]?;
        let minimum = projections.min_wmma_rows[stage]?;
        let experts = projections.geometry.grouped_experts;
        let plan = GroupedDqRows::new(
            experts,
            layout.row_ids.len() / 4,
            minimum,
            groups
                .iter()
                .map(|(id, span)| (id.expert as usize, span.clone())),
        )?;
        let area = &layout.grouped[stage];
        for (dst, &word) in upload[area.start / 4..area.end / 4]
            .iter_mut()
            .zip(plan.ids().iter().chain(plan.bounds()))
        {
            *dst = word.to_ne_bytes();
        }
        Some(plan)
    })
}

fn pack_routes(
    upload: &mut [[u8; 4]],
    layout: &Layout,
    rank: usize,
    groups: BTreeMap<ExpertId, Vec<(usize, usize)>>,
    scales: impl Fn(ExpertId) -> [f32; 2],
    weight: impl Fn(usize, usize) -> f32,
) -> Result<Vec<(ExpertId, Range<usize>)>, EngError> {
    let mut next = 0;
    let mut batches = Vec::with_capacity(groups.len());
    for (id, lanes) in groups {
        let first = next;
        let [gate, down] = scales(id);
        for (row, lane) in lanes {
            let original = row * rank + lane;
            let weight = weight(row, id.expert as usize);
            if !weight.is_finite() {
                return Err(bad("CUDA expert routing requires finite weights"));
            }
            upload[layout.row_ids.start / 4 + next] = (row as i32).to_ne_bytes();
            upload[layout.inverse_ids.start / 4 + original] = (next as i32).to_ne_bytes();
            upload[layout.weights.start / 4 + original] = weight.to_ne_bytes();
            upload[layout.gate_scales.start / 4 + next] = gate.to_ne_bytes();
            upload[layout.down_scales.start / 4 + next] = down.to_ne_bytes();
            next += 1;
        }
        batches.push((id, first..next));
    }
    Ok(batches)
}

impl Scratch {
    fn allocate(
        cuda: &CudaContext,
        rows: usize,
        geometry: Geometry,
        layout: &Layout,
    ) -> Result<Option<Self>, EngError> {
        let allocate = || -> Result<_, EngError> {
            let buffer = cuda.allocate(layout.down.end)?;
            if (cuda.memory_info()?.free as u64) < DEVICE_HEADROOM {
                // Driver allocation granularity can exceed the logical extent.
                return Ok(None);
            }
            Ok(Some(Self {
                buffer,
                stream: cuda.stream()?,
                host: cuda.pinned_buf(layout.output.end)?,
                uploaded: None,
                rows,
                geometry,
            }))
        };
        match allocate() {
            Err(EngError::MemoryPlan(_)) => Ok(None),
            other => other,
        }
    }

    fn execute(
        &mut self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        operands: (Option<DeviceRange<'_>>, Option<DeviceRange<'_>>),
    ) -> Result<(Vec<Vec<f32>>, [DenseGemvGroupCounts; 2]), EngError> {
        let l = &plan.layout;
        let upload = self.buffer.range(0, l.flag.end)?;
        self.host.write(0, plan.upload.as_flattened())?;
        cuda.copy_from_pinned(&self.stream, &mut self.host, 0, upload)?;
        let result = self
            .enqueue(cuda, &self.stream, projections, plan, operands)
            .and_then(|counts| {
                // With a device destination, only status crosses the host boundary.
                let end = if operands.1.is_some() {
                    l.flag.end
                } else {
                    l.output.end
                };
                let readback = self.buffer.range(l.flag.start, end - l.flag.start)?;
                cuda.copy_to_pinned(&self.stream, readback, &mut self.host, l.flag.start)?;
                Ok(counts)
            });
        // A launch can be refused after the upload has been queued.
        let waited = self.host.synchronize(&self.stream);
        self.uploaded = None;
        let counts = result?;
        waited?;
        let mut flag = [0; 4];
        self.host.read(l.flag.start, &mut flag)?;
        if u32::from_ne_bytes(flag) != 0 {
            return Err(bad(
                "CUDA expert prefill produced a nonfinite output or invalid route",
            ));
        }
        if operands.1.is_some() {
            return Ok((Vec::new(), counts));
        }
        let mut out = vec![[0; 4]; l.output.len() / 4];
        self.host.read(l.output.start, out.as_flattened_mut())?;
        let rows = out
            .chunks_exact(projections.geometry.hidden)
            .map(|row| row.iter().copied().map(f32::from_ne_bytes).collect())
            .collect();
        Ok((rows, counts))
    }

    /// The launch-only form for a device-resident caller: the window's metadata goes through
    /// `host` (it stays unread until the caller's fence) on `stream`, the kernels follow on the
    /// same stream, and the window's flag word is copied into `flag_out` for that fence. One
    /// window only; nothing is synchronized or read.
    fn enqueue_unfenced(
        &mut self,
        cuda: &CudaContext,
        stream: &DeviceStream,
        flag_out: DeviceRange<'_>,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        operands: (Option<DeviceRange<'_>>, Option<DeviceRange<'_>>),
    ) -> Result<[DenseGemvGroupCounts; 2], EngError> {
        let l = &plan.layout;
        if plan.rows > self.rows || flag_out.bytes() < 4 {
            return Err(bad("the unfenced expert window does not fit its admission"));
        }
        let upload = self.buffer.range(0, l.flag.end)?;
        // The pinned page of the previous window is released by the event recorded behind its
        // upload: a tiny copy long finished by the time the next visit reaches its experts.
        if let Some(event) = self.uploaded.take() {
            self.host.wait(&event)?;
        }
        self.host.write(0, plan.upload.as_flattened())?;
        cuda.copy_from_pinned(stream, &mut self.host, 0, upload)?;
        let event = cuda.event()?;
        stream.record(&event)?;
        self.uploaded = Some(event);
        let counts = self.enqueue(cuda, stream, projections, plan, operands)?;
        cuda.copy_device(stream, self.buffer.range(l.flag.start, 4)?, flag_out)?;
        Ok(counts)
    }

    fn enqueue(
        &self,
        cuda: &CudaContext,
        stream: &DeviceStream,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        (retained, output): (Option<DeviceRange<'_>>, Option<DeviceRange<'_>>),
    ) -> Result<[DenseGemvGroupCounts; 2], EngError> {
        let l = &plan.layout;
        let shape = projections.geometry;
        let count = plan.rows * shape.rank;
        let range = |span: &Range<usize>| self.buffer.range(span.start, span.len());
        let flag = range(&l.flag)?;
        cuda.row_gather(
            stream,
            RowGatherCall {
                x: match retained {
                    Some(input) => input,
                    None => range(&l.input)?,
                },
                indices: range(&l.row_ids)?,
                out: range(&l.gathered)?,
                flag,
                shape: RowShape {
                    rows: count,
                    columns: shape.hidden,
                },
                source_rows: plan.rows,
            },
        )?;
        let gate = self.project(cuda, projections, plan, stream, 0)?;
        cuda.row_clipped_expert(
            stream,
            ClippedExpertCall {
                x: range(&l.gate)?,
                scales: range(&l.gate_scales)?,
                out: range(&l.middle)?,
                flag,
                shape: RowShape {
                    rows: count,
                    columns: shape.middle,
                },
                caps: projections.caps,
            },
        )?;
        let down = self.project(cuda, projections, plan, stream, 1)?;
        cuda.row_indexed_rank_sum(
            stream,
            IndexedRankSumCall {
                x: range(&l.down)?,
                inverse_ids: range(&l.inverse_ids)?,
                scales: range(&l.down_scales)?,
                weights: range(&l.weights)?,
                out: match output {
                    Some(range) => range,
                    None => range(&l.output)?,
                },
                flag,
                shape: RowShape {
                    rows: plan.rows,
                    columns: shape.hidden,
                },
                rank: shape.rank,
            },
        )?;
        Ok([gate, down])
    }

    fn project(
        &self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        stream: &DeviceStream,
        stage: usize,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        let l = &plan.layout;
        let pair = [(&l.gathered, &l.gate), (&l.middle, &l.down)][stage];
        let (x_span, y_span) = pair;
        let minimum = projections.min_wmma_rows[stage];
        let selection = plan.grouped[stage].as_ref();
        if let (Some(selection), Some(weight)) = (selection, projections.grouped_weights[stage]) {
            let range = |span: &Range<usize>| self.buffer.range(span.start, span.len());
            cuda.gemm_dq_sparse_rows(
                stream,
                SparseDqCall {
                    weight,
                    plan: selection,
                    metadata: range(&l.grouped[stage])?,
                    x: range(x_span)?,
                    y: range(y_span)?,
                },
            )?;
        }
        let calls = plan
            .groups
            .iter()
            .filter(|(id, _)| selection.is_none_or(|plan| !plan.includes(id.expert as usize)))
            .map(|(id, slots)| {
                let matrix = &projections.matrices[id][stage];
                let [rows, cols] = matrix.shape;
                let span = |range: &Range<usize>, width| -> Result<DeviceRange<'_>, EngError> {
                    self.buffer.range(
                        range.start + slots.start * width * 4,
                        slots.len() * width * 4,
                    )
                };
                Ok(DenseGemvCall {
                    w: matrix.data,
                    x: span(x_span, cols)?,
                    y: span(y_span, rows)?,
                    shape: DenseGemvShape::packed(slots.len(), rows, cols),
                    dtype: projections.dtypes[stage],
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        if Switch::PrefillWeightPanels.on() {
            // Down is unwritten during gate; gate is dead after row_clipped_expert.
            // Both stages and all users of these spans are ordered on `stream`.
            let dead = if stage == 0 { &l.down } else { &l.gate };
            let pad = (16 - dead.start % 16) % 16;
            if dead.len() > pad {
                let tmp = self.buffer.range(dead.start + pad, dead.len() - pad)?;
                return cuda.dense_gemv_group_with_panels(
                    stream,
                    &calls,
                    minimum,
                    projections.exact_tile,
                    tmp,
                );
            }
        }
        cuda.dense_gemv_group_with_tiles(stream, &calls, minimum, projections.exact_tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_router_rows_remove_only_input_from_expert_upload() {
        let shape = Geometry {
            hidden: 2304,
            middle: 512,
            rank: 8,
            grouped_experts: 0,
        };
        for rows in [2, 8, 31, 512] {
            let original = Layout::new(rows, shape).unwrap();
            let retained = Layout::for_input(rows, shape, true).unwrap();
            let saved = rows * shape.hidden * size_of::<f32>();
            assert!(admits_router_input(rows, shape.hidden, saved));
            assert!(!admits_router_input(rows, shape.hidden, saved - 1));
            assert!(retained.input.is_empty());
            assert_eq!(retained.row_ids.start, 0);
            assert_eq!(retained.flag.end + saved, original.flag.end);
            assert_eq!(retained.down.end + saved, original.down.end);
        }
        assert!(!admits_router_input(usize::MAX, 2, 0));
    }

    #[test]
    fn device_prompt_arena_fits_width_by_workspace_and_launch_bounds() {
        let shape = Geometry {
            hidden: 2304,
            middle: 512,
            rank: 8,
            grouped_experts: 0,
        };
        let single = Layout::new(1, shape).unwrap();
        let plan = Layout::new(512, shape).unwrap();
        assert_eq!(plan.input.len(), 512 * 2304 * 4);
        assert_eq!(plan.gate.len(), 512 * 8 * 2 * 512 * 4);
        assert_eq!(plan.down.end, 512 * (single.down.end - 4) + 4);
        assert_eq!(Layout::fit(512, shape, plan.down.end), Some(512));
        assert_eq!(Layout::fit(512, shape, plan.down.end - 1), Some(511));
        assert_eq!(Layout::fit(512, shape, single.down.end), None);
        assert_eq!(
            Layout::fit(usize::MAX, shape, usize::MAX),
            Some(DenseGemvShape::MAX_ACTIVATION_ROWS / 8)
        );
    }

    #[test]
    fn grouped_expert_workspace_serves_plain_and_smaller_grouped_layouts() {
        let largest = Geometry {
            hidden: 2304,
            middle: 1152,
            rank: 8,
            grouped_experts: 128,
        };
        let plain = Geometry {
            grouped_experts: 0,
            ..largest
        };
        assert!(!plain.can_serve(largest));
        assert!(!largest.can_serve(Geometry { rank: 4, ..plain }));
        assert!(!largest.can_serve(Geometry {
            hidden: 1152,
            ..plain
        }));
        assert!(!largest.can_serve(Geometry {
            middle: 512,
            ..plain
        }));
        let capacity = Layout::new(511, largest).unwrap();
        for experts in [0, 1, 31, 127, 128] {
            let shape = Geometry {
                grouped_experts: experts,
                ..largest
            };
            assert!(largest.can_serve(shape));
            for n in [2, 29, 511] {
                for retained in [false, true] {
                    let layout = Layout::for_input(n, shape, retained).unwrap();
                    assert!(layout.output.end <= capacity.output.end);
                    assert!(layout.down.end <= capacity.down.end);
                }
            }
        }
    }

    #[test]
    fn grouped_expert_layout_accounts_for_metadata_and_vector_alignment() {
        let shape = Geometry {
            hidden: 2304,
            middle: 1152,
            rank: 8,
            grouped_experts: 128,
        };
        for rows in [2, 31, 511, 512] {
            let full = Layout::new(rows, shape).unwrap();
            let retained = Layout::for_input(rows, shape, true).unwrap();
            for l in [&full, &retained] {
                assert_eq!(l.grouped[0].start, l.weights.end);
                assert_eq!(l.grouped[0].end, l.grouped[1].start);
                assert!(l.grouped[1].end <= l.flag.start);
                assert_eq!(l.grouped[1].len(), (rows * 8 + 129) * 4);
                assert_eq!(l.gathered.start % 16, 0);
                assert_eq!(l.middle.start % 16, 0);
            }
            assert_eq!(Layout::fit(rows, shape, full.down.end), Some(rows));
            assert_ne!(Layout::fit(rows, shape, full.down.end - 1), Some(rows));
        }
    }

    #[test]
    fn device_prompt_routes_keep_original_rank_and_expert_scale() {
        let shape = Geometry {
            hidden: 2,
            middle: 2,
            rank: 2,
            grouped_experts: 0,
        };
        let input = [0.0; 2];
        let routes = [
            PrefillRoute {
                ids: &[2, 0],
                input: &input,
            },
            PrefillRoute {
                ids: &[1, 2],
                input: &input,
            },
        ];
        let layout = Layout::new(routes.len(), shape).unwrap();
        let groups = group_route_ids(5, RowRoutes::Host(&routes).ids());
        let ids: [&[usize]; 2] = [&[2, 0], &[1, 2]];
        let resident = RowRoutes::Resident(&ids);
        assert_eq!(group_route_ids(5, resident.ids()), groups);
        assert_eq!(resident.window(1..2).ids().next(), Some(ids[1]));
        let mut upload = vec![[0; 4]; layout.flag.end / 4];
        let batches = pack_routes(
            &mut upload,
            &layout,
            shape.rank,
            groups,
            |id| {
                let expert = id.expert as f32;
                [expert + 2.0, expert + 3.0]
            },
            |row, id| (10 * (row + 1) + id) as f32,
        )
        .unwrap();
        let slot = |bytes: &Range<usize>, i| upload[bytes.start / 4 + i];
        let ints = |span: &Range<usize>| {
            std::array::from_fn::<_, 4, _>(|i| i32::from_ne_bytes(slot(span, i)))
        };
        let floats = |span: &Range<usize>| {
            std::array::from_fn::<_, 4, _>(|i| f32::from_ne_bytes(slot(span, i)))
        };
        assert_eq!(
            batches
                .iter()
                .map(|(id, slots)| (id.expert, slots.clone()))
                .collect::<Vec<_>>(),
            [(0, 0..1), (1, 1..2), (2, 2..4)]
        );
        assert_eq!(ints(&layout.row_ids), [0, 1, 0, 1]);
        assert_eq!(ints(&layout.inverse_ids), [2, 0, 1, 3]);
        assert_eq!(floats(&layout.gate_scales), [2.0, 3.0, 4.0, 4.0]);
        assert_eq!(floats(&layout.down_scales), [3.0, 4.0, 5.0, 5.0]);
        assert_eq!(floats(&layout.weights), [12.0, 10.0, 21.0, 22.0]);
    }
}
