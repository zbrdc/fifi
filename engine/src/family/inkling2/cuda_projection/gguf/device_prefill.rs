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
    family::inkling2::cpu_forward::ROUTE_NORMALIZATION_FLOOR,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        prefill::{
            mmq_mmid::{
                GroupedDqRows, HostIds, IdsGuardCall, MmidIds, MmidIdsOut, MmidLaunchCall,
                MmidQuantCall, MmidShape, MmidWeight, SparseDqCall, grouped_dq_serves,
            },
            mmq_quant::MmqPacking,
            mmq_size::{
                MmqType, WEIGHT_TAIL_BYTES, fixup_floats_max, ids_dst_ints, q8_1_bytes, weight_tail,
            },
            row_ops::{ClippedExpertCall, IndexedRankSumCall, RowGatherCall, RowShape},
        },
        resident::{
            dense_gemv::{
                DenseDtype, DenseGemvCall, DenseGemvGroupCounts, DenseGemvShape, dense_gemv_bytes,
            },
            matvec_q::QuantType,
            router::{
                MAX_ROWS as ROUTER_MAX_ROWS, RouterLogitsCall, RouterShape, RouterSigmoidBias,
                RouterSigmoidCall, router_packed_logits_bytes,
            },
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
    /// The router's experts when the rows route on the device, else 0.
    routed_experts: usize,
    /// The stack's experts when a stage runs on the vendored grouped MMQ, else 0.
    mmq_experts: usize,
}

impl Geometry {
    // Each window recomputes its layout. For the same tensor shape, bounding the
    // grouped metadata also bounds the padded prefix of the device and host arenas.
    fn can_serve(self, shape: Self) -> bool {
        self.hidden == shape.hidden
            && self.middle == shape.middle
            && self.rank == shape.rank
            && self.grouped_experts >= shape.grouped_experts
            && self.routed_experts >= shape.routed_experts
            && self.mmq_experts >= shape.mmq_experts
    }

    /// The smallest geometry that serves both, for one tensor shape.
    fn union(self, shape: Self) -> Self {
        if (self.hidden, self.middle, self.rank) != (shape.hidden, shape.middle, shape.rank) {
            return shape;
        }
        Self {
            grouped_experts: self.grouped_experts.max(shape.grouped_experts),
            routed_experts: self.routed_experts.max(shape.routed_experts),
            mmq_experts: self.mmq_experts.max(shape.mmq_experts),
            ..shape
        }
    }
}

/// The host plan fills everything before `flag` and zeroes `flag`; on the host route it also
/// writes the inverse ids and weights, which a device route writes itself.
struct Layout {
    input: Range<usize>,
    row_ids: Range<usize>,
    gate_scales: Range<usize>,
    down_scales: Range<usize>,
    grouped: [Range<usize>; 2],
    flag: Range<usize>,
    inverse_ids: Range<usize>,
    weights: Range<usize>,
    route: Option<RouteSpans>,
    mmid: Option<MmidSpans>,
    output: Range<usize>,
    gathered: Range<usize>,
    gate: Range<usize>,
    middle: Range<usize>,
    down: Range<usize>,
}

/// The ids of a grouped MMQ stage, which reads and writes the slots in their compact order: the
/// identity maps (`ids_dst` with the stream-k fixup gather pad) and the expert bounds.
struct MmidSpans {
    ids_dst: Range<usize>,
    ids_src1: Range<usize>,
    bounds: Range<usize>,
}

/// Device routing scratch: the bias and the status word are uploaded together, and the status,
/// the expert bounds and the compact slot map come back together.
struct RouteSpans {
    bias: Range<usize>,
    status: Range<usize>,
    bounds: Range<usize>,
    ids_dst: Range<usize>,
    picks: Range<usize>,
    guarded: Range<usize>,
    logits: Range<usize>,
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
        let gate_scales = span(slots, 1)?;
        let down_scales = span(slots, 1)?;
        let grouped = if shape.grouped_experts > 0 {
            let words = slots.checked_add(shape.grouped_experts)?.checked_add(1)?;
            [span(words, 1)?, span(words, 1)?]
        } else {
            [0..0, 0..0]
        };
        let mmid = if shape.mmq_experts > 0 {
            Some(MmidSpans {
                ids_dst: span(ids_dst_ints(slots, rows).ok()?, 1)?,
                ids_src1: span(slots, 1)?,
                bounds: span(shape.mmq_experts.checked_add(1)?, 1)?,
            })
        } else {
            None
        };
        let flag = span(1, 1)?;
        let inverse_ids = span(slots, 1)?;
        let weights = span(slots, 1)?;
        let experts = shape.routed_experts;
        let route = if experts > 0 {
            Some(RouteSpans {
                bias: span(experts, 1)?,
                status: span(1, 1)?,
                bounds: span(experts.checked_add(1)?, 1)?,
                // The grouped launch's fixup gather reads up to `rows + 127` past the slots.
                ids_dst: span(slots.checked_add(rows)?.checked_add(127)?, 1)?,
                picks: span(slots, 1)?,
                guarded: span(slots, 1)?,
                logits: span(rows, experts)?,
            })
        } else {
            None
        };
        if shape.grouped_experts > 0 || shape.mmq_experts > 0 {
            // F32 vector loads in GEMM and the q8_1 gather of a tensor-core stage require packed
            // operands at 16-byte alignment.
            let end = route.as_ref().map_or(weights.end, |spans| spans.logits.end);
            span((4 - end / 4 % 4) % 4, 1)?;
        }
        Some(Self {
            input,
            row_ids,
            gate_scales,
            down_scales,
            grouped,
            flag,
            inverse_ids,
            weights,
            route,
            mmid,
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
        routed_experts: usize::try_from(spec.n_routed_experts).ok()?,
        mmq_experts: if Switch::PrefillTensorCoreExperts.on() {
            usize::try_from(spec.n_routed_experts).ok()?
        } else {
            0
        },
    };
    let rows = context.min(DenseGemvShape::MAX_ACTIVATION_ROWS.checked_div(shape.rank)?);
    u64::try_from(Layout::new(rows, shape)?.down.end).ok()
}

#[derive(Default)]
pub(super) struct Workspace {
    scratch: Option<Scratch>,
    /// The rows of the server's prompt chunk ([`GgufCuda::plan_prompt_chunk`]).
    chunk_rows: usize,
}

struct Scratch {
    buffer: DeviceBuffer,
    /// The q8_1 columns and the stream-k fixup of a grouped MMQ stage, when the geometry has one.
    mmq: Option<MmqScratch>,
    stream: DeviceStream,
    host: PinnedBuf,
    rows: usize,
    geometry: Geometry,
}

struct MmqScratch {
    buffer: DeviceBuffer,
    fixup_at: usize,
    fixup_bytes: usize,
}

impl MmqScratch {
    /// The q8_1 columns of `slots` slots of the widest stage input and the fixup of the device.
    fn allocate(cuda: &CudaContext, slots: usize, k: usize) -> Result<Self, EngError> {
        let fixup_at = q8_1_bytes(k, slots)?.next_multiple_of(256);
        let fixup_bytes = fixup_floats_max(cuda.mmq_device_facts()?.nsm)? * size_of::<f32>();
        let bytes = fixup_at
            .checked_add(fixup_bytes)
            .ok_or_else(|| bad("expert prefill MMQ scratch overflows"))?;
        Ok(Self {
            buffer: cuda.allocate(bytes)?,
            fixup_at,
            fixup_bytes,
        })
    }
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
        // A growing scratch takes the whole prompt chunk and every geometry it has served, so
        // the served start's warmup sizes it and no later prompt waits on an allocation.
        let grown = self
            .scratch
            .as_ref()
            .map_or(shape, |buffer| buffer.geometry.union(shape));
        let Some(rows) = Layout::fit(requested.max(self.chunk_rows), grown, free) else {
            return Ok(self.cached(shape));
        };
        if self.cached(shape).is_some_and(|buffer| buffer.rows >= rows) {
            return Ok(self.cached(shape));
        }
        let layout =
            Layout::new(rows, grown).ok_or_else(|| bad("expert prefill arena overflows"))?;
        let Some(scratch) = Scratch::allocate(cuda, rows, grown, &layout)? else {
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

struct Projections<'a> {
    geometry: Geometry,
    caps: [f32; 2],
    dtypes: [DenseDtype; 2],
    min_wmma_rows: [Option<usize>; 2],
    exact_tile: bool,
    grouped_weights: [Option<MmidWeight<'a>>; 2],
    /// The stacks of the stages that run on the vendored grouped MMQ, over every slot.
    mmq_weights: [Option<MmidWeight<'a>>; 2],
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

/// Prompt rows that route on the device: `activations` holds the normalized rows, `router`
/// names the layer's F32 router, and `bias` and `scale` are its expert bias and route scale.
pub(in crate::family::inkling2) struct DeviceRoute<'a> {
    pub(in crate::family::inkling2) router: &'a str,
    pub(in crate::family::inkling2) bias: &'a [f32],
    pub(in crate::family::inkling2) scale: f32,
    pub(in crate::family::inkling2) rank: usize,
    pub(in crate::family::inkling2) hidden: usize,
    pub(in crate::family::inkling2) activations: DeviceRange<'a>,
}

struct Router<'a> {
    w: DeviceRange<'a>,
    bias: &'a [f32],
    scale: f32,
}

/// The routed experts of a device route: the outputs, unless they went to a destination, and
/// each row's experts in rank order, `rank` per row.
pub(in crate::family::inkling2) struct DeviceRouted {
    pub(in crate::family::inkling2) outputs: Vec<Vec<f32>>,
    pub(in crate::family::inkling2) picks: Vec<usize>,
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

    /// Sizes the device expert prefill scratch for prompt chunks of `rows` rows.
    pub(in crate::family::inkling2) fn plan_prompt_chunk(&self, rows: usize) {
        if let Some(workspace) = &self.device_prefill {
            workspace.borrow_mut().chunk_rows = rows;
        }
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
                .map(|out| out.slice(first * hidden * 4, routes.len() * hidden * 4))
                .transpose()?;
            let (rows, launches) = scratch.execute(cuda, &projections, &plan, (input, target))?;
            output.extend(rows);
            for (total, count) in counts.iter_mut().zip(launches) {
                total.wmma += count.wmma;
                total.exact_tiles += count.exact_tiles;
            }
            for (total, selection) in grouped_slots.iter_mut().zip(&plan.grouped) {
                *total += selection.as_ref().map_or(0, GroupedDqRows::selected);
            }
        }
        self.report_expert_prefill(
            layer,
            [routes.len(), scratch.rows],
            [retained, destination],
            (counts, grouped_slots),
        );
        Ok(Some(output))
    }

    /// Routes the prompt rows of `route` on the device and runs their experts. The router
    /// logits stay on the device; only the status, the expert bounds and the compact slot map
    /// come back, and the host plans the expert launches from them. Returns `None`, before any
    /// device work, when the device does not route these rows.
    pub(in crate::family::inkling2) fn route_device_experts(
        &self,
        source: &mut impl ExpertPayloadSource,
        layer: u32,
        (route, destination): (DeviceRoute<'_>, Option<DeviceRange<'_>>),
        dtypes: [DenseDtype; 2],
        caps: [f32; 2],
    ) -> Result<Option<DeviceRouted>, EngError> {
        let Some(workspace) = self
            .device_prefill
            .as_ref()
            .filter(|_| self.device_prompt_routing)
        else {
            return Ok(None);
        };
        let Some((weight, DenseDtype::F32)) = self.dense.get(route.router) else {
            return Ok(None);
        };
        let DeviceRoute {
            bias,
            scale,
            rank,
            hidden,
            activations,
            ..
        } = route;
        let [experts, width] = weight.shape;
        let stride = hidden * size_of::<f32>();
        let rows = activations.bytes().checked_div(stride).unwrap_or(0);
        // Windows never exceed the router's row limit, which the grouped ids share.
        let shape = RouterShape {
            m: rows.clamp(1, ROUTER_MAX_ROWS),
            hidden,
            n_expert: experts,
            top_k: rank,
        };
        if width != hidden
            || bias.len() != experts
            || !scale.is_finite()
            || !admits_router_input(rows, hidden, activations.bytes())
            || !self.valid_device_prefill(rows, rank, dtypes, caps)
            || router_packed_logits_bytes(shape).is_err()
        {
            return Ok(None);
        }
        if let Some(out) = destination
            && (!admits_router_input(rows, hidden, out.bytes()) || !out.offset().is_multiple_of(4))
        {
            return Err(bad(
                "incorrect CUDA expert output destination extent or alignment",
            ));
        }
        let Some(mut projections) =
            self.prepare_device_experts(source, layer, 0..experts, [rank, hidden], dtypes, caps)?
        else {
            return Ok(None);
        };
        projections.geometry.routed_experts = experts;
        let cuda = &self.context;
        let mut workspace = workspace.borrow_mut();
        let Some(scratch) = workspace.fit(cuda, rows, projections.geometry)? else {
            return Ok(None);
        };
        let router = Router {
            w: weight.codes.range(0, weight.codes.bytes())?,
            bias,
            scale,
        };
        let mut routed = DeviceRouted {
            outputs: Vec::with_capacity(if destination.is_some() { 0 } else { rows }),
            picks: Vec::with_capacity(rows * rank),
        };
        let mut counts = [DenseGemvGroupCounts::default(); 2];
        let mut grouped_slots = [0; 2];
        let window = scratch.rows.min(ROUTER_MAX_ROWS);
        for first in (0..rows).step_by(window) {
            let count = (rows - first).min(window);
            let input = activations.slice(first * stride, count * stride)?;
            let target = destination
                .map(|out| out.slice(first * stride, count * stride))
                .transpose()?;
            let (plan, picks) = scratch.route(cuda, &projections, layer, &router, input)?;
            let (rows, launches) =
                scratch.execute(cuda, &projections, &plan, (Some(input), target))?;
            routed.outputs.extend(rows);
            routed.picks.extend(picks);
            for (total, count) in counts.iter_mut().zip(launches) {
                total.wmma += count.wmma;
                total.exact_tiles += count.exact_tiles;
            }
            for (total, selection) in grouped_slots.iter_mut().zip(&plan.grouped) {
                *total += selection.as_ref().map_or(0, GroupedDqRows::selected);
            }
        }
        self.report_expert_prefill(
            layer,
            [rows, window],
            [Some(activations), destination],
            (counts, grouped_slots),
        );
        Ok(Some(routed))
    }

    fn report_expert_prefill(
        &self,
        layer: u32,
        [rows, window]: [usize; 2],
        [retained, destination]: [Option<DeviceRange<'_>>; 2],
        (counts, grouped_slots): ([DenseGemvGroupCounts; 2], [usize; 2]),
    ) {
        if self.prefill_report.is_none() {
            return;
        }
        eprintln!(
            "Inkling CUDA device expert prefill {}",
            serde_json::json!({"layer": layer, "rows": rows,
                "windows": rows.div_ceil(window),
                "router_input_h2d_bytes_saved": retained.map_or(0, |input| input.bytes()),
                "expert_output_d2h_bytes_avoided": destination.map_or(0, |out| out.bytes()),
                "wmma_gate_calls": counts[0].wmma, "wmma_down_calls": counts[1].wmma,
                "grouped_dq_gate_slots": grouped_slots[0], "grouped_dq_down_slots": grouped_slots[1],
                "exact_tile_gate_calls": counts[0].exact_tiles,
                "exact_tile_down_calls": counts[1].exact_tiles})
        );
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
        let (grouped_weights, grouped_experts, mmq_weights, mmq_experts) =
            if self.device_prefill_grouped_dq && self.device_prefill_wmma {
                let stacks = self.resident.original_layer(layer)?;
                let weights = std::array::from_fn(|stage| {
                    grouped_weight(dtypes[stage], shapes[stage], stacks[stage])
                });
                let whole = Switch::PrefillTensorCoreExperts
                    .on()
                    .then(|| self.resident.mmq_layer(layer))
                    .flatten();
                let mmq = std::array::from_fn(|stage| {
                    whole.and_then(|stacks| mmq_weight(dtypes[stage], shapes[stage], stacks[stage]))
                });
                // A tensor-core stack also holds its tail, which is shorter than one expert.
                let experts = |weights: &[Option<MmidWeight<'_>>; 2]| {
                    weights
                        .iter()
                        .flatten()
                        .next()
                        .map_or(0, |w| w.base.bytes() / w.stride_bytes)
                };
                (weights, experts(&weights), mmq, experts(&mmq))
            } else {
                ([None, None], 0, [None, None], 0)
            };
        Ok(Some(Projections {
            geometry: Geometry {
                hidden,
                middle,
                rank,
                grouped_experts,
                routed_experts: 0,
                mmq_experts,
            },
            dtypes,
            min_wmma_rows: WMMA_MIN_ROWS.map(|rows| self.device_prefill_wmma.then_some(rows)),
            exact_tile: self.device_prefill_exact_tile,
            grouped_weights,
            mmq_weights,
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

/// The stack of one stage on the vendored grouped MMQ: a type with a case, a row of whole
/// blocks, and whole expert matrices followed by the zeroed tail (`base` is a
/// `ResidentExperts::mmq_layer` stack) that a row off the k grid reads into.
fn mmq_weight(
    dtype: DenseDtype,
    [n, k]: [usize; 2],
    base: DeviceRange<'_>,
) -> Option<MmidWeight<'_>> {
    let DenseDtype::PackedGguf(kind) = dtype else {
        return None;
    };
    let quant = QuantType::from_gguf(kind).filter(|&quant| MmqType::of(quant).is_some())?;
    if !k.is_multiple_of(quant.block_size())
        || weight_tail(quant, k).is_none_or(|tail| tail > WEIGHT_TAIL_BYTES)
    {
        return None;
    }
    let stride_bytes = quant.weight_bytes(n, k)?;
    let matrices = base.bytes().checked_sub(WEIGHT_TAIL_BYTES)?;
    if stride_bytes <= WEIGHT_TAIL_BYTES || !matrices.is_multiple_of(stride_bytes) {
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

/// Each expert's slots in the expert-major order, experts ascending.
type ExpertGroups = Vec<(ExpertId, Range<usize>)>;

struct RoutedWindow {
    upload: Vec<[u8; 4]>,
    groups: ExpertGroups,
    grouped: [Option<GroupedDqRows>; 2],
    /// The host copy of the grouped MMQ ids the upload carries, when a stage runs there.
    mmid: Option<MmidHost>,
    layout: Layout,
    rows: usize,
}

/// The compact slots map to themselves; `bounds[e]` is the first slot of expert `e`.
struct MmidHost {
    ids_dst: Vec<i32>,
    bounds: Vec<i32>,
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
        let mut upload = vec![[0; 4]; layout.weights.end / 4];
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
        let mmid = pack_mmid(
            projections.geometry.mmq_experts,
            &layout,
            &groups,
            &mut upload,
        )?;
        Ok(Self {
            upload,
            groups,
            grouped,
            mmid,
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
        if projections.mmq_weights[stage].is_some() {
            return None;
        }
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

/// Writes the grouped MMQ ids of `groups` into `upload`: every slot is its own compact column, so
/// the stage reads its input rows and writes its output rows in the order the other stages use.
fn pack_mmid(
    experts: usize,
    layout: &Layout,
    groups: &[(ExpertId, Range<usize>)],
    upload: &mut [[u8; 4]],
) -> Result<Option<MmidHost>, EngError> {
    let Some(spans) = &layout.mmid else {
        return Ok(None);
    };
    let slots = spans.ids_src1.len() / 4;
    let mut bounds = vec![0i32; experts + 1];
    for (id, span) in groups {
        let expert = id.expert as usize;
        if expert >= experts {
            return Err(bad(
                "CUDA expert prefill routed an expert outside the MMQ stack",
            ));
        }
        bounds[expert + 1] = i32::try_from(span.len())
            .map_err(|_| bad("CUDA expert prefill slot count overflows"))?;
    }
    for e in 0..experts {
        bounds[e + 1] += bounds[e];
    }
    if usize::try_from(bounds[experts]).ok() != Some(slots) {
        return Err(bad("CUDA expert prefill groups do not cover every slot"));
    }
    let ids_dst: Vec<i32> = (0..slots)
        .map(|slot| {
            i32::try_from(slot).map_err(|_| bad("CUDA expert prefill slot count overflows"))
        })
        .collect::<Result<_, _>>()?;
    for (span, values) in [
        (&spans.ids_dst, &ids_dst),
        (&spans.ids_src1, &ids_dst),
        (&spans.bounds, &bounds),
    ] {
        for (dst, value) in upload[span.start / 4..].iter_mut().zip(values.iter()) {
            *dst = value.to_ne_bytes();
        }
    }
    Ok(Some(MmidHost { ids_dst, bounds }))
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

/// The host part of a device-grouped plan. `bounds` and `columns` are the device's expert
/// bounds and compact slot map (column to slot `row * rank + lane`), in expert order and row
/// order within an expert, as [`pack_routes`] groups them. Fills each column's row and expert
/// scales and returns the expert groups and each slot's expert.
fn pack_device_grouping(
    upload: &mut [[u8; 4]],
    layout: &Layout,
    (layer, rank): (u32, usize),
    (bounds, columns): (&[i32], &[i32]),
    scales: impl Fn(ExpertId) -> Option<[f32; 2]>,
) -> Result<(ExpertGroups, Vec<usize>), EngError> {
    let invalid = || bad("CUDA prompt routing returned an invalid expert grouping");
    let slots = columns.len();
    if bounds.first() != Some(&0) || bounds.last().copied() != i32::try_from(slots).ok() {
        return Err(invalid());
    }
    let mut groups = Vec::new();
    let mut picks = vec![0; slots];
    for (expert, pair) in bounds.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(invalid());
        }
        // The first bound is zero and the bounds rise to `slots`.
        let [start, end] = [pair[0], pair[1]].map(|bound| bound as usize);
        if start == end {
            continue;
        }
        let id = ExpertId {
            layer,
            expert: expert as u32,
        };
        let [gate, down] = scales(id).ok_or_else(invalid)?.map(f32::to_ne_bytes);
        for (column, &slot) in columns.iter().enumerate().take(end).skip(start) {
            let slot = usize::try_from(slot)
                .ok()
                .filter(|&slot| slot < slots)
                .ok_or_else(invalid)?;
            upload[layout.row_ids.start / 4 + column] = ((slot / rank) as i32).to_ne_bytes();
            upload[layout.gate_scales.start / 4 + column] = gate;
            upload[layout.down_scales.start / 4 + column] = down;
            picks[slot] = expert;
        }
        groups.push((id, start..end));
    }
    Ok((groups, picks))
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
            let mmq = if geometry.mmq_experts > 0 {
                let k = geometry.hidden.max(geometry.middle);
                Some(MmqScratch::allocate(cuda, rows * geometry.rank, k)?)
            } else {
                None
            };
            if (cuda.memory_info()?.free as u64) < DEVICE_HEADROOM {
                // Driver allocation granularity can exceed the logical extent.
                return Ok(None);
            }
            Ok(Some(Self {
                buffer,
                mmq,
                stream: cuda.stream()?,
                host: cuda.pinned_buf(layout.output.end)?,
                rows,
                geometry,
            }))
        };
        match allocate() {
            Err(EngError::MemoryPlan(_)) => Ok(None),
            other => other,
        }
    }

    /// Routes the rows of `input` on the device and groups their slots by expert there, then
    /// plans the window from the status, expert bounds and compact slot map that come back.
    /// The device holds the inverse ids and the route weights; the plan uploads the rest.
    /// Returns the plan and each row's experts in rank order.
    fn route(
        &mut self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        layer: u32,
        router: &Router<'_>,
        input: DeviceRange<'_>,
    ) -> Result<(RoutedWindow, Vec<usize>), EngError> {
        let shape = projections.geometry;
        let rows = input.bytes() / (shape.hidden * size_of::<f32>());
        let layout = Layout::for_input(rows, shape, true)
            .ok_or_else(|| bad("expert prefill arena overflows"))?;
        let spans = layout
            .route
            .as_ref()
            .ok_or_else(|| bad("expert prefill arena has no routing scratch"))?;
        let (experts, rank) = (shape.routed_experts, shape.rank);
        let slots = rows * rank;
        let mut sent: Vec<[u8; 4]> = router.bias.iter().map(|b| b.to_ne_bytes()).collect();
        sent.push([0; 4]);
        self.host.write(spans.bias.start, sent.as_flattened())?;
        let queued = self.enqueue_route(
            cuda,
            (&layout, spans),
            router,
            input,
            [rows, shape.hidden, rank],
        );
        // A launch can be refused after the bias upload has been queued.
        let waited = self.host.synchronize(&self.stream);
        queued?;
        waited?;
        let read_ints = |span: Range<usize>| -> Result<Vec<i32>, EngError> {
            let mut words = vec![[0; 4]; span.len() / 4];
            self.host.read(span.start, words.as_flattened_mut())?;
            Ok(words.into_iter().map(i32::from_ne_bytes).collect())
        };
        if read_ints(spans.status.clone())?[0] != 0 {
            return Err(bad(
                "CUDA prompt routing met a nonfinite router logit or an invalid expert",
            ));
        }
        let bounds = read_ints(spans.bounds.clone())?;
        let columns = read_ints(spans.ids_dst.start..spans.ids_dst.start + slots * 4)?;
        if bounds.len() != experts + 1 {
            return Err(bad(
                "CUDA prompt routing returned an invalid expert grouping",
            ));
        }
        let mut upload = vec![[0; 4]; layout.flag.end / 4];
        let (groups, picks) = pack_device_grouping(
            &mut upload,
            &layout,
            (layer, rank),
            (&bounds, &columns),
            |id| {
                projections
                    .matrices
                    .get(&id)
                    .map(|pair| pair.each_ref().map(|matrix| matrix.scale))
            },
        )?;
        let grouped = pack_grouped_rows(projections, &layout, &groups, &mut upload);
        let mmid = pack_mmid(
            projections.geometry.mmq_experts,
            &layout,
            &groups,
            &mut upload,
        )?;
        Ok((
            RoutedWindow {
                upload,
                groups,
                grouped,
                mmid,
                layout,
                rows,
            },
            picks,
        ))
    }

    /// Queues the bias upload, the router logits and selection, the grouped ids and the
    /// readback of the status, bounds and compact slot map.
    fn enqueue_route(
        &mut self,
        cuda: &CudaContext,
        (layout, spans): (&Layout, &RouteSpans),
        router: &Router<'_>,
        x: DeviceRange<'_>,
        [rows, hidden, rank]: [usize; 3],
    ) -> Result<(), EngError> {
        let experts = router.bias.len();
        let sent = self
            .buffer
            .range(spans.bias.start, spans.status.end - spans.bias.start)?;
        cuda.copy_from_pinned(&self.stream, &mut self.host, spans.bias.start, sent)?;
        let range = |span: &Range<usize>| self.buffer.range(span.start, span.len());
        let shape = RouterShape {
            m: rows,
            hidden,
            n_expert: experts,
            top_k: rank,
        };
        let logits = range(&spans.logits)?;
        let status = range(&spans.status)?;
        cuda.router_logits_rows(
            &self.stream,
            RouterLogitsCall {
                shape,
                x,
                w: router.w,
                logits,
            },
        )?;
        cuda.router_sigmoid_select_rows(
            &self.stream,
            RouterSigmoidCall {
                shape,
                bias: Some(RouterSigmoidBias::HostRounded {
                    values: range(&spans.bias)?,
                    floor: ROUTE_NORMALIZATION_FLOOR,
                    scale: router.scale,
                }),
                logits,
                ids: range(&spans.picks)?,
                weights: range(&layout.weights)?,
                status,
            },
        )?;
        cuda.mmq_build_ids(
            &self.stream,
            IdsGuardCall {
                shape: MmidShape {
                    n_experts: experts,
                    m: rows,
                    top_k: rank,
                },
                ids: range(&spans.picks)?,
                ids_g: range(&spans.guarded)?,
                status: Some(status),
            },
            MmidIdsOut {
                ids_src1: range(&layout.inverse_ids)?,
                ids_dst: range(&spans.ids_dst)?,
                expert_bounds: range(&spans.bounds)?,
            },
        )?;
        let back = spans.status.start..spans.ids_dst.start + rows * rank * 4;
        let returned = self.buffer.range(back.start, back.len())?;
        cuda.copy_to_pinned(&self.stream, returned, &mut self.host, back.start)
    }

    fn execute(
        &mut self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        operands: (Option<DeviceRange<'_>>, Option<DeviceRange<'_>>),
    ) -> Result<(Vec<Vec<f32>>, [DenseGemvGroupCounts; 2]), EngError> {
        let l = &plan.layout;
        let upload = self.buffer.range(0, size_of_val(plan.upload.as_slice()))?;
        self.host.write(0, plan.upload.as_flattened())?;
        cuda.copy_from_pinned(&self.stream, &mut self.host, 0, upload)?;
        let result = self
            .enqueue(cuda, projections, plan, operands)
            .and_then(|counts| {
                // With a device destination, only status crosses the host boundary.
                let flag = self.buffer.range(l.flag.start, l.flag.len())?;
                cuda.copy_to_pinned(&self.stream, flag, &mut self.host, l.flag.start)?;
                if operands.1.is_none() {
                    let output = self.buffer.range(l.output.start, l.output.len())?;
                    cuda.copy_to_pinned(&self.stream, output, &mut self.host, l.output.start)?;
                }
                Ok(counts)
            });
        // A launch can be refused after the upload has been queued.
        let waited = self.host.synchronize(&self.stream);
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

    fn enqueue(
        &self,
        cuda: &CudaContext,
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
            &self.stream,
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
        let gate = self.project(cuda, projections, plan, 0)?;
        cuda.row_clipped_expert(
            &self.stream,
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
        let down = self.project(cuda, projections, plan, 1)?;
        cuda.row_indexed_rank_sum(
            &self.stream,
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

    /// One stage over every slot on the vendored grouped MMQ: the compact input rows are quantized
    /// to q8_1 columns at the layout of the stack, then each expert's columns run through its
    /// matrix into the compact output rows.
    fn project_mmq(
        &self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        weight: MmidWeight<'_>,
        (x_span, y_span): (&Range<usize>, &Range<usize>),
    ) -> Result<(), EngError> {
        let missing = || bad("CUDA expert prefill MMQ stage has no ids or scratch");
        let (host, spans, scratch) = (
            plan.mmid.as_ref().ok_or_else(missing)?,
            plan.layout.mmid.as_ref().ok_or_else(missing)?,
            self.mmq.as_ref().ok_or_else(missing)?,
        );
        let range = |span: &Range<usize>| self.buffer.range(span.start, span.len());
        let shape = MmidShape {
            n_experts: projections.geometry.mmq_experts,
            m: plan.rows,
            top_k: projections.geometry.rank,
        };
        let ids = MmidIds::wrap(
            shape,
            HostIds {
                ids_dst: &host.ids_dst,
                expert_bounds: &host.bounds,
            },
            (0, shape.ne_get_rows()),
            MmidIdsOut {
                ids_src1: range(&spans.ids_src1)?,
                ids_dst: range(&spans.ids_dst)?,
                expert_bounds: range(&spans.bounds)?,
            },
        )?;
        let q8 = scratch
            .buffer
            .range(0, q8_1_bytes(weight.k, shape.ne_get_rows())?)?;
        cuda.mmq_quantize_gather_ids(
            &self.stream,
            &ids,
            MmidQuantCall {
                packing: MmqPacking::Vendor,
                quant: weight.quant,
                k: weight.k,
                row_stride: weight.k,
                x: range(x_span)?,
                y: q8,
            },
        )?;
        cuda.mmq_mmid_launch(
            &self.stream,
            MmidLaunchCall {
                shape,
                weight,
                ids: &ids,
                y: q8,
                dst: range(y_span)?,
                ld: weight.n,
                fixup: scratch
                    .buffer
                    .range(scratch.fixup_at, scratch.fixup_bytes)?,
            },
        )
    }

    fn project(
        &self,
        cuda: &CudaContext,
        projections: &Projections<'_>,
        plan: &RoutedWindow,
        stage: usize,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        let l = &plan.layout;
        let pair = [(&l.gathered, &l.gate), (&l.middle, &l.down)][stage];
        let (x_span, y_span) = pair;
        if let Some(weight) = projections.mmq_weights[stage] {
            self.project_mmq(cuda, projections, plan, weight, pair)?;
            return Ok(DenseGemvGroupCounts::default());
        }
        let minimum = projections.min_wmma_rows[stage];
        let selection = plan.grouped[stage].as_ref();
        if let (Some(selection), Some(weight)) = (selection, projections.grouped_weights[stage]) {
            let range = |span: &Range<usize>| self.buffer.range(span.start, span.len());
            cuda.gemm_dq_sparse_rows(
                &self.stream,
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
            // Both stages and all users of these spans are ordered on self.stream.
            let dead = if stage == 0 { &l.down } else { &l.gate };
            let pad = (16 - dead.start % 16) % 16;
            if dead.len() > pad {
                let tmp = self.buffer.range(dead.start + pad, dead.len() - pad)?;
                return cuda.dense_gemv_group_with_panels(
                    &self.stream,
                    &calls,
                    minimum,
                    projections.exact_tile,
                    tmp,
                );
            }
        }
        cuda.dense_gemv_group_with_tiles(&self.stream, &calls, minimum, projections.exact_tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_mmq_ids_map_every_compact_slot_to_itself_within_expert_bounds() {
        let shape = Geometry {
            hidden: 256,
            middle: 128,
            rank: 2,
            grouped_experts: 4,
            routed_experts: 0,
            mmq_experts: 4,
        };
        let layout = Layout::new(3, shape).unwrap();
        let spans = layout.mmid.as_ref().unwrap();
        assert_eq!(spans.ids_dst.len() / 4, 6 + 3 + 127);
        assert!(spans.bounds.end <= layout.flag.start);
        let id = |expert| ExpertId { layer: 7, expert };
        let groups = [(id(0), 0..2), (id(2), 2..5), (id(3), 5..6)];
        let mut upload = vec![[0xff; 4]; layout.flag.end / 4];
        let host = pack_mmid(4, &layout, &groups, &mut upload)
            .unwrap()
            .unwrap();
        assert_eq!(host.bounds, [0, 2, 2, 5, 6]);
        assert_eq!(host.ids_dst, [0, 1, 2, 3, 4, 5]);
        let words = |span: &Range<usize>, n: usize| -> Vec<i32> {
            upload[span.start / 4..span.start / 4 + n]
                .iter()
                .map(|w| i32::from_ne_bytes(*w))
                .collect()
        };
        assert_eq!(words(&spans.ids_dst, 6), host.ids_dst);
        assert_eq!(words(&spans.ids_src1, 6), host.ids_dst);
        assert_eq!(words(&spans.bounds, 5), host.bounds);
        let short = [(id(0), 0..2), (id(2), 2..5)];
        assert!(pack_mmid(4, &layout, &short, &mut upload).is_err());
        let outside = [(id(0), 0..2), (id(4), 2..6)];
        assert!(pack_mmid(4, &layout, &outside, &mut upload).is_err());
        let plain = Layout::new(
            3,
            Geometry {
                mmq_experts: 0,
                ..shape
            },
        )
        .unwrap();
        assert!(plain.mmid.is_none());
        assert!(
            pack_mmid(0, &plain, &groups, &mut upload)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn retained_router_rows_remove_only_input_from_expert_upload() {
        let shape = Geometry {
            hidden: 2304,
            middle: 512,
            rank: 8,
            grouped_experts: 0,
            routed_experts: 0,
            mmq_experts: 0,
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
    fn tensor_core_only_expert_stacks_align_their_activation_planes() {
        // Q4_K and Q6_K stacks take the tensor-core stage without the dequantizing grouped one.
        for routed_experts in [0, 128] {
            let shape = Geometry {
                hidden: 2304,
                middle: 1152,
                rank: 8,
                grouped_experts: 0,
                routed_experts,
                mmq_experts: 128,
            };
            for rows in [1, 2, 3, 7, 31, 255, 256, 511, 512] {
                let layout = Layout::new(rows, shape).unwrap();
                for plane in [
                    &layout.output,
                    &layout.gathered,
                    &layout.gate,
                    &layout.middle,
                ] {
                    assert_eq!(plane.start % 16, 0, "rows {rows}, routed {routed_experts}");
                }
            }
        }
    }

    #[test]
    fn device_prompt_arena_fits_width_by_workspace_and_launch_bounds() {
        let shape = Geometry {
            hidden: 2304,
            middle: 512,
            rank: 8,
            grouped_experts: 0,
            routed_experts: 0,
            mmq_experts: 0,
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
            routed_experts: 0,
            mmq_experts: 0,
        };
        let plain = Geometry {
            grouped_experts: 0,
            ..largest
        };
        assert!(!plain.can_serve(largest));
        let routed = Geometry {
            routed_experts: 128,
            mmq_experts: 128,
            ..plain
        };
        let both = largest.union(routed);
        assert!(both.can_serve(largest) && both.can_serve(routed) && both.can_serve(plain));
        assert_eq!(largest.union(Geometry { rank: 4, ..plain }).rank, 4);
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
            routed_experts: 0,
            mmq_experts: 0,
        };
        for rows in [2, 31, 511, 512] {
            let full = Layout::new(rows, shape).unwrap();
            let retained = Layout::for_input(rows, shape, true).unwrap();
            for l in [&full, &retained] {
                assert_eq!(l.grouped[0].start, l.down_scales.end);
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
            routed_experts: 0,
            mmq_experts: 0,
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
        let mut upload = vec![[0; 4]; layout.weights.end / 4];
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

    #[test]
    fn device_grouping_plans_the_rows_and_scales_of_the_host_route() {
        let rank = 3;
        let experts = 7;
        let shape = Geometry {
            hidden: 4,
            middle: 4,
            rank,
            grouped_experts: 0,
            routed_experts: experts,
            mmq_experts: 0,
        };
        let mut seed = 12345u64;
        let mut next = |bound: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % bound
        };
        for rows in [2, 5, 17] {
            // Distinct experts per row, as the router picks them.
            let picks: Vec<Vec<usize>> = (0..rows)
                .map(|_| {
                    let mut row = Vec::new();
                    while row.len() < rank {
                        let expert = next(experts);
                        if !row.contains(&expert) {
                            row.push(expert);
                        }
                    }
                    row
                })
                .collect();
            let scales = |id: ExpertId| {
                let expert = id.expert as f32;
                Some([expert + 0.5, expert + 1.5])
            };
            let layout = Layout::for_input(rows, shape, true).unwrap();
            let mut host = vec![[0; 4]; layout.weights.end / 4];
            let ids: Vec<&[usize]> = picks.iter().map(Vec::as_slice).collect();
            let host_groups = pack_routes(
                &mut host,
                &layout,
                rank,
                group_route_ids(5, RowRoutes::Resident(&ids).ids()),
                |id| scales(id).unwrap(),
                |_, _| 1.0,
            )
            .unwrap();
            // The device's grouping: experts in order, rows in order within an expert.
            let mut bounds = vec![0];
            let mut columns = Vec::new();
            for expert in 0..experts {
                for (row, picked) in picks.iter().enumerate() {
                    if let Some(lane) = picked.iter().position(|&e| e == expert) {
                        columns.push((row * rank + lane) as i32);
                    }
                }
                bounds.push(columns.len() as i32);
            }
            let mut device = vec![[0; 4]; layout.flag.end / 4];
            let (groups, slots) =
                pack_device_grouping(&mut device, &layout, (5, rank), (&bounds, &columns), scales)
                    .unwrap();
            assert_eq!(groups, host_groups);
            assert_eq!(slots, picks.concat());
            for span in [&layout.row_ids, &layout.gate_scales, &layout.down_scales] {
                assert_eq!(
                    device[span.start / 4..span.end / 4],
                    host[span.start / 4..span.end / 4]
                );
            }
            // The device's inverse map is the host's inverse ids.
            let mut inverse = vec![0; rows * rank];
            for (column, &slot) in columns.iter().enumerate() {
                inverse[slot as usize] = column as i32;
            }
            let span = &layout.inverse_ids;
            let host_inverse: Vec<i32> = host[span.start / 4..span.end / 4]
                .iter()
                .map(|word| i32::from_ne_bytes(*word))
                .collect();
            assert_eq!(inverse, host_inverse);
        }
        let layout = Layout::for_input(2, shape, true).unwrap();
        let mut upload = vec![[0; 4]; layout.flag.end / 4];
        let scales = |_| Some([1.0, 1.0]);
        for (bounds, columns) in [
            (vec![0, 3, 2, 6, 6, 6, 6, 6], vec![0, 1, 2, 3, 4, 5]),
            (vec![0, 3, 3, 3, 3, 3, 3, 5], vec![0, 1, 2, 3, 4, 5]),
            (vec![0, 3, 3, 3, 3, 3, 3, 6], vec![0, 1, 2, 3, 4, 6]),
        ] {
            assert!(
                pack_device_grouping(&mut upload, &layout, (5, rank), (&bounds, &columns), scales)
                    .is_err()
            );
        }
    }
}
