//! What the resident step owns for a session and the weights it resolves beyond the mixers: the
//! arena of one row, the layer table, the shapes of the MoE, embedding and head weights, and the
//! lookups that turn them into typed weights.
//!
//! The shapes are pure host arithmetic from `StepDims` (no model number is written here) and are
//! tested on the CPU. The lookups read `CudaWeights` and are covered by the scratch device run.
//!
//! One row only: the MoE block takes `m = 1`, so the plan, the arena and every launch of the step
//! are made for [`STEP_ROWS`] row.

use super::sparse_exec::{SparseConfig, SparseExec, SparseGeometry};
use super::{
    dims::{KvMode, StepDims, StepOptions},
    layer_table::LayerTable,
    mixer_geometry::{DenseSpec, MixerGeometry, check_dense, check_native_dense},
    mixer_weights::dense_weight,
    moe::{MoeDims, MoeLayerWeights, StackWeights},
    paging_plan::PagingPlan,
    paging_store::device_first::DeviceCachePlan,
    shard::ShardTable,
    state::PagedStatePlan,
    state_layout::StateLayout,
    step_plan::{KernelScratch, MAX_STEP_ROWS, StepPlan},
    tiered_step::LayerParts,
};
use crate::{
    EngError,
    family::spec::AttentionKind,
    family::{
        gguf_model::{Inventory, Role},
        project::cuda::{CudaWeight, CudaWeights, WeightLayout, exl3::Exl3Weight},
    },
    kernels::cuda::{
        CudaContext, DeviceArena, DeviceBuffer, NativeWeight, TiledWeight,
        resident::{
            attn_decode_q8::{AttnQ8Shape, paged::PagedQ8Scratch},
            matmul_expert::ExpertStack,
            matvec_q::QuantType,
        },
    },
    weights::GgufType,
};

use std::cell::RefCell;

pub(super) const STEP_ROWS: usize = 1;
const F32: usize = size_of::<f32>();

/// An unchecked `values * F32` would wrap and silently match a wrapped size.
pub(super) fn f32_bytes(values: usize) -> Result<usize, EngError> {
    values
        .checked_mul(F32)
        .ok_or_else(|| EngError::BadRequest(format!("{values} f32 values do not fit a byte count")))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StackSpec {
    pub(super) experts: usize,
    pub(super) rows: usize,
    pub(super) k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MoeSpecs {
    /// f32 values of `ffn_gate_inp` and of `ffn_gate_inp_shexp`.
    pub(super) router_values: usize,
    pub(super) shexp_values: usize,
    pub(super) gate: StackSpec,
    pub(super) up: StackSpec,
    pub(super) down: StackSpec,
    pub(super) shared_gate: DenseSpec,
    pub(super) shared_up: DenseSpec,
    pub(super) shared_down: DenseSpec,
}

impl MoeSpecs {
    pub(super) fn new(dims: &StepDims) -> Result<Self, EngError> {
        let (h, m) = (dims.shape.hidden, &dims.shape.moe);
        let router_values = m.experts.checked_mul(h).ok_or_else(|| {
            EngError::Checkpoint("moe specs: the router size overflows".to_owned())
        })?;
        let stack = |rows, k| StackSpec {
            experts: m.experts,
            rows,
            k,
        };
        let dense = |role, rows, k| DenseSpec { role, rows, k };
        Ok(Self {
            router_values,
            shexp_values: h,
            gate: stack(m.inter, h),
            up: stack(m.inter, h),
            down: stack(h, m.inter),
            shared_gate: dense(Role::FfnGateShexp, m.shared_inter, h),
            shared_up: dense(Role::FfnUpShexp, m.shared_inter, h),
            shared_down: dense(Role::FfnDownShexp, h, m.shared_inter),
        })
    }

    /// The specs of `card` under `table`: every expert's and the shared
    /// expert's intermediate width becomes the card's slice (gate/up rows,
    /// down columns); the router, the residual width and the expert count
    /// stay full, because routing and the residual row are replicated. The
    /// checks then validate the CARD's loaded (sliced) shapes by name.
    pub(super) fn for_card(
        dims: &StepDims,
        table: &ShardTable,
        card: usize,
    ) -> Result<Self, EngError> {
        let (h, m) = (dims.shape.hidden, &dims.shape.moe);
        let inter = table.expert_inter(card)?.len();
        let shared = table.shared_inter(card)?.len();
        let stack = |rows, k| StackSpec {
            experts: m.experts,
            rows,
            k,
        };
        let dense = |role, rows, k| DenseSpec { role, rows, k };
        Ok(Self {
            router_values: m.experts.checked_mul(h).ok_or_else(|| {
                EngError::Checkpoint("moe specs: the router size overflows".to_owned())
            })?,
            shexp_values: h,
            gate: stack(inter, h),
            up: stack(inter, h),
            down: stack(h, inter),
            shared_gate: dense(Role::FfnGateShexp, shared, h),
            shared_up: dense(Role::FfnUpShexp, shared, h),
            shared_down: dense(Role::FfnDownShexp, h, shared),
        })
    }
}

pub(super) fn moe_dims(dims: &StepDims) -> MoeDims {
    MoeDims::new(dims, STEP_ROWS)
}

/// Refuses a NATIVE expert stack that is not the launch's shape: the rule of [`check_stack`] plus
/// the MMVQ row-group rule on the PER-EXPERT rows (`rows / experts % 4`).
pub(super) fn check_native_stack(
    name: &str,
    have: (usize, usize, Option<usize>),
    want: &StackSpec,
) -> Result<(), EngError> {
    check_stack(name, have, want)?;
    let experts = have.2.filter(|&experts| experts > 0).ok_or_else(|| {
        EngError::Checkpoint(format!("moe weights: {name} is not an expert stack"))
    })?;
    if !have.0.is_multiple_of(experts) || !(have.0 / experts).is_multiple_of(4) {
        return Err(EngError::Checkpoint(format!(
            "moe weights: {name} stacks experts of {} rows, the native matvec needs a multiple of 4",
            have.0 / experts.max(1)
        )));
    }
    Ok(())
}

/// `have` is the tile's total rows, its `k` and its stack depth.
pub(super) fn check_stack(
    name: &str,
    have: (usize, usize, Option<usize>),
    want: &StackSpec,
) -> Result<(), EngError> {
    let (rows, k, experts) = have;
    let total = want.experts.checked_mul(want.rows);
    if experts != Some(want.experts) || Some(rows) != total || k != want.k {
        return Err(EngError::Checkpoint(format!(
            "moe weights: {name} is {rows} rows of {k} (stack {experts:?}), the step needs {} experts of {} rows of {}",
            want.experts, want.rows, want.k
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GlobalSpecs {
    pub(super) embed: DenseSpec,
    pub(super) head: DenseSpec,
    /// f32 values of `output_norm`.
    pub(super) norm_values: usize,
}

impl GlobalSpecs {
    pub(super) fn new(dims: &StepDims) -> Self {
        let (h, v) = (dims.shape.hidden, dims.shape.vocab);
        Self {
            embed: DenseSpec {
                role: Role::TokenEmbd,
                rows: v,
                k: h,
            },
            head: DenseSpec {
                role: Role::Output,
                rows: v,
                k: h,
            },
            norm_values: h,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GlobalNames {
    pub(super) embed: String,
    pub(super) head: String,
    pub(super) norm: String,
}

impl GlobalNames {
    pub(super) fn new(inventory: &Inventory) -> Result<Self, EngError> {
        let find = |role: Role| {
            inventory
                .tensors()
                .iter()
                .find(|t| t.role == role)
                .map(|t| t.name.clone())
                .ok_or_else(|| EngError::Checkpoint(format!("the inventory has no {role:?}")))
        };
        Ok(Self {
            embed: find(Role::TokenEmbd)?,
            head: find(Role::Output)?,
            norm: find(Role::OutputNorm)?,
        })
    }
}

pub(super) fn check_paged_modes(paged: bool, capture: bool, chunked: bool) -> Result<(), EngError> {
    if paged && (capture || chunked) {
        return Err(EngError::BadRequest(
            "paged KV requires eager per-token prefill; disable capture and chunked prefill"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Refuses the session options stage 1 cannot run before anything is planned or allocated.
/// `forked_experts` makes the plan allocate one `XfDown` slot per routed expert
/// (`step_plan.rs`), while the step launches the single `Slot::XfDown(0)` pack of the unforked
/// plan: accepting it would silently compute a wrong result. Per-expert `XfDown` slots are a
/// stage-2 item (stage 1b, branch streams), so the option is refused with a typed error.
pub(super) fn check_stage1_options(options: &StepOptions) -> Result<(), EngError> {
    if options.forked_experts {
        return Err(EngError::BadRequest(
            "forked_experts is refused in stage 1: the step launches the single XfDown pack, \
             and per-expert XfDown slots are a stage-2 item"
                .to_owned(),
        ));
    }
    if let Some(plan) = options.tiered {
        plan.check()?;
        if options.layout != WeightLayout::Native {
            return Err(EngError::Checkpoint(String::from(
                "tiered needs the native layout: the bank stacks are native blocks and the expert \
                 launches read them",
            )));
        }
    }
    Ok(())
}

fn paged_attention_shape(dims: &StepDims, capacity: usize) -> Result<AttnQ8Shape, EngError> {
    let attn = &dims.shape.attn;
    let ld = attn
        .heads_q
        .checked_mul(2)
        .and_then(|n| n.checked_mul(attn.head_dim))
        .ok_or_else(|| EngError::MemoryPlan("paged projection stride overflows".into()))?;
    Ok(AttnQ8Shape {
        m: STEP_ROWS,
        n_head: attn.heads_q,
        n_kv_head: attn.heads_kv,
        head_dim: attn.head_dim,
        cells: capacity,
        ld,
    })
}

/// CPU-only resource plan, retained through load admission and device allocation.
pub(super) struct PagedResourcesPlan {
    dims: StepDims,
    plan: StepPlan,
    state: PagedStatePlan,
    attention: AttnQ8Shape,
    attention_bytes: usize,
    table: LayerTable,
    geometry: MixerGeometry,
    names: GlobalNames,
    options: StepOptions,
}

impl PagedResourcesPlan {
    pub(super) fn new(
        dims: StepDims,
        inventory: &Inventory,
        options: &StepOptions,
        host_limit: usize,
    ) -> Result<Self, EngError> {
        check_stage1_options(options)?;
        let state = PagedStatePlan::new(&dims, options, host_limit)?;
        Self::with_state(dims, inventory, options, state)
    }

    pub(super) fn new_cached(
        dims: StepDims,
        inventory: &Inventory,
        options: &StepOptions,
        cache: DeviceCachePlan,
    ) -> Result<Self, EngError> {
        check_stage1_options(options)?;
        let state = PagedStatePlan::new_cached(&dims, options, cache)?;
        Self::with_state(dims, inventory, options, state)
    }

    pub(super) fn cache_rows(&self) -> Option<usize> {
        self.state.cache_rows()
    }

    pub(super) fn fixed_device_bytes(
        dims: &StepDims,
        options: &StepOptions,
    ) -> Result<usize, EngError> {
        check_stage1_options(options)?;
        PagedStatePlan::check_options(options)?;
        let state = StateLayout::new(dims, options)?;
        let paging = PagingPlan::new(dims, options.context_tokens)?;
        let scratch = KernelScratch::for_dims(dims, STEP_ROWS, options.kv_mode)?;
        let plan = StepPlan::new_paged(dims, options, &scratch)?;
        let shape = paged_attention_shape(dims, options.context_tokens)?;
        let attention = PagedQ8Scratch::allocation_bytes(&shape, paging.group_cells)?;
        [
            state.delta.total,
            state.words.total,
            paging.device_bytes,
            plan.arena_bytes(),
            attention,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or_else(|| EngError::MemoryPlan("fixed cache resources overflow".into()))
        })
    }

    fn with_state(
        dims: StepDims,
        inventory: &Inventory,
        options: &StepOptions,
        state: PagedStatePlan,
    ) -> Result<Self, EngError> {
        let scratch = KernelScratch::for_dims(&dims, STEP_ROWS, options.kv_mode)?;
        let plan = StepPlan::new_paged(&dims, options, &scratch)?;
        let attention = paged_attention_shape(&dims, options.context_tokens)?;
        let attention_bytes = PagedQ8Scratch::allocation_bytes(&attention, state.page_cells())?;
        let table = LayerTable::new(&dims, inventory)?;
        let geometry = MixerGeometry::new(&dims)?;
        let names = GlobalNames::new(inventory)?;
        let planned = Self {
            dims,
            plan,
            state,
            attention,
            attention_bytes,
            table,
            geometry,
            names,
            options: *options,
        };
        planned.device_bytes()?;
        Ok(planned)
    }

    pub(super) fn device_bytes(&self) -> Result<usize, EngError> {
        self.state
            .device_bytes()?
            .checked_add(self.plan.arena_bytes())
            .and_then(|n| n.checked_add(self.attention_bytes))
            .ok_or_else(|| EngError::MemoryPlan("paged resource size overflows".to_owned()))
    }
}

pub(super) struct PagedResources {
    pub(super) state: PagedStatePlan,
    pub(super) scratch: RefCell<PagedQ8Scratch>,
    pub(super) options: StepOptions,
}

/// The activation arena of the K-row verify window (`qwen36.draft_verify = krow`): the plan of a
/// window of the longest length, over its own arena. A window of `k` rows uses the first `k` rows
/// of every slot (all slots are row-major), so one plan serves every window length. The one-row
/// plan and arena are not touched.
pub(super) struct KrowResources {
    pub(super) plan: StepPlan,
    pub(super) arena: DeviceArena,
}

/// Everything a session builds once and the step borrows.
pub(in crate::family) struct StepResources {
    pub(super) dims: StepDims,
    pub(super) plan: StepPlan,
    pub(super) arena: DeviceArena,
    pub(super) table: LayerTable,
    pub(super) geometry: MixerGeometry,
    pub(super) names: GlobalNames,
    pub(super) paged: Option<PagedResources>,
    /// The query-aware sparse decode resources, present only when the env flag armed them,
    /// the KV mode is the q8 resident path, and VRAM admission passed (R3, fail-open).
    pub(super) sparse: Option<SparseExec>,
    /// The K-row arena, present only when [`Self::with_krow`] built it.
    pub(super) krow: Option<KrowResources>,
}

impl StepResources {
    /// Adds the arena of the K-row verify window. Refused for the paged and the sparse
    /// resources, which have no K-row body. Allocation happens here, before the first token.
    pub(super) fn with_krow(
        mut self,
        context: &CudaContext,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        if self.paged.is_some() || self.sparse.is_some() {
            return Err(EngError::BadRequest(
                "step resources: the K-row window needs the resident KV without the sparse chain"
                    .to_owned(),
            ));
        }
        let scratch = KernelScratch::for_verify(&self.dims, MAX_STEP_ROWS, options.kv_mode)?;
        let plan = StepPlan::new_verify(&self.dims, MAX_STEP_ROWS, options, &scratch)?;
        let arena = context.arena(&plan.arena_tensors())?;
        self.krow = Some(KrowResources { plan, arena });
        Ok(self)
    }

    /// Device bytes of the K-row arena: zero when there is none.
    pub(super) fn krow_arena_bytes(&self) -> usize {
        self.krow.as_ref().map_or(0, |krow| krow.arena.bytes())
    }

    /// Allocation happens here and never inside a token.
    pub(super) fn new(
        context: &CudaContext,
        dims: StepDims,
        inventory: &Inventory,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        check_stage1_options(options)?;
        let scratch = KernelScratch::for_dims(&dims, STEP_ROWS, options.kv_mode)?;
        let plan = StepPlan::new(&dims, STEP_ROWS, options, &scratch)?;
        let arena = context.arena(&plan.arena_tensors())?;
        let table = LayerTable::new(&dims, inventory)?;
        let geometry = MixerGeometry::new(&dims)?;
        let names = GlobalNames::new(inventory)?;
        let sparse = admit_sparse(context, &dims, options);
        Ok(Self {
            dims,
            plan,
            arena,
            table,
            geometry,
            names,
            paged: None,
            sparse,
            krow: None,
        })
    }

    pub(super) fn allocate_paged(
        context: &CudaContext,
        planned: PagedResourcesPlan,
    ) -> Result<Self, EngError> {
        context.ensure_uncaptured("paged step resources")?;
        let arena = context.arena(&planned.plan.arena_tensors())?;
        let scratch = context.paged_q8_scratch(planned.attention, planned.state.page_cells())?;
        Ok(Self {
            dims: planned.dims,
            plan: planned.plan,
            arena,
            table: planned.table,
            geometry: planned.geometry,
            names: planned.names,
            paged: Some(PagedResources {
                state: planned.state,
                scratch: RefCell::new(scratch),
                options: planned.options,
            }),
            sparse: None,
            krow: None,
        })
    }
}

/// VRAM kept free beyond the sparse planes when admitting them (probe value; 4e-3 measures it).
const DEFAULT_SPARSE_HEADROOM: u64 = 2 << 30;

/// R3 admission: exact geometry from the session shape, `context_tokens` as the cell capacity
/// (never a reduced one), and a fail-open outcome — any refusal or error leaves dense Q8
/// authoritative and the sparse path dark, loudly.
fn admit_sparse(
    context: &CudaContext,
    dims: &StepDims,
    options: &StepOptions,
) -> Option<SparseExec> {
    if !super::sparse_exec::armed() || options.kv_mode != KvMode::Int8G32 {
        return None;
    }
    let attn = &dims.shape.attn;
    let geom = SparseGeometry {
        n_q_head: attn.heads_q,
        n_kv_head: attn.heads_kv,
        head_dim: attn.head_dim,
        layers: dims.layers_of(AttentionKind::Full),
        capacity: options.context_tokens,
    };
    match SparseExec::new(
        context,
        geom,
        SparseConfig::default(),
        DEFAULT_SPARSE_HEADROOM,
    ) {
        Ok(exec) => Some(exec),
        Err(e) => {
            eprintln!("qwen36: sparse attention not admitted, dense Q8 stays authoritative: {e:?}");
            None
        }
    }
}

fn tiled<'w>(
    weights: &'w CudaWeights,
    name: &str,
) -> Result<(&'w TiledWeight, Option<usize>), EngError> {
    match weights.get(name) {
        Some(CudaWeight::Tiled {
            weight, experts, ..
        }) => Ok((weight, *experts)),
        _ => Err(EngError::Checkpoint(format!(
            "step weights: {name} is not a tiled tensor"
        ))),
    }
}

pub(super) fn raw_buffer<'w>(
    weights: &'w CudaWeights,
    name: &str,
    values: usize,
) -> Result<&'w DeviceBuffer, EngError> {
    let bytes = f32_bytes(values)?;
    match weights.get(name) {
        Some(CudaWeight::Raw { ty, buffer, .. })
            if *ty == GgufType::F32 && buffer.bytes() == bytes =>
        {
            Ok(buffer)
        }
        _ => Err(EngError::Checkpoint(format!(
            "step weights: {name} is not a raw f32 tensor of {values} values"
        ))),
    }
}

fn dense<'w>(
    weights: &'w CudaWeights,
    name: &str,
    want: &DenseSpec,
) -> Result<&'w TiledWeight, EngError> {
    let (weight, experts) = tiled(weights, name)?;
    check_dense(name, (weight.n_rows(), weight.k(), experts), want)?;
    Ok(weight)
}

fn stack_of<'w>(
    weights: &'w CudaWeights,
    name: &str,
    want: &StackSpec,
    layout: WeightLayout,
) -> Result<StackWeights<'w>, EngError> {
    match (weights.get(name), layout) {
        (
            Some(CudaWeight::Tiled {
                weight, experts, ..
            }),
            WeightLayout::Tiled,
        ) => {
            check_stack(name, (weight.n_rows(), weight.k(), *experts), want)?;
            Ok(StackWeights::Tiled(ExpertStack {
                weight,
                experts: want.experts,
            }))
        }
        (Some(CudaWeight::Native { weight, .. }), WeightLayout::Native) => {
            let experts = weight.experts().ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "moe weights: {name} is not an expert stack, the step needs {} experts",
                    want.experts
                ))
            })?;
            check_native_stack(name, (weight.n_rows(), weight.k(), Some(experts)), want)?;
            Ok(StackWeights::Native(weight))
        }
        // The tiled-head hybrid's sibling: an expert stack the load overrode to tiles beside
        // the native session, so the block runs the Tiled per-slot form for this layer.
        (
            Some(CudaWeight::Tiled {
                weight, experts, ..
            }),
            WeightLayout::Native,
        ) => {
            check_stack(name, (weight.n_rows(), weight.k(), *experts), want)?;
            Ok(StackWeights::Tiled(ExpertStack {
                weight,
                experts: want.experts,
            }))
        }
        (Some(CudaWeight::Exl3 { .. } | CudaWeight::Exl3Bank { .. }), _)
        | (_, WeightLayout::Exl3) => Err(EngError::Checkpoint(format!(
            "moe weights: {name} needs EXL3 per-expert dispatch"
        ))),
        // (Tiled, Native) never reaches here: the first arm already accepted it (the hybrid
        // seam serves Tiled expert stacks in a Native session).
        (Some(CudaWeight::Native { .. }), WeightLayout::Tiled) => Err(EngError::Checkpoint(
            format!("moe weights: {name} is loaded in the other layout than the session"),
        )),
        (Some(CudaWeight::Raw { .. }), _) | (None, _) => Err(EngError::Checkpoint(format!(
            "moe weights: {name} is not a quantized stack"
        ))),
    }
}

pub(super) fn moe_layer_weights<'w>(
    table: &LayerTable,
    weights: &'w CudaWeights,
    specs: &MoeSpecs,
    layer: usize,
    layout: WeightLayout,
) -> Result<MoeLayerWeights<'w>, EngError> {
    let name = |role| table.name(layer, role);
    Ok(MoeLayerWeights {
        router: raw_buffer(weights, name(Role::FfnGateInp)?, specs.router_values)?,
        router_shared: raw_buffer(weights, name(Role::FfnGateInpShexp)?, specs.shexp_values)?,
        gate: stack_of(weights, name(Role::FfnGateExps)?, &specs.gate, layout)?,
        up: stack_of(weights, name(Role::FfnUpExps)?, &specs.up, layout)?,
        down: stack_of(weights, name(Role::FfnDownExps)?, &specs.down, layout)?,
        shared_gate: dense_weight(
            weights,
            name(Role::FfnGateShexp)?,
            &specs.shared_gate,
            layout,
        )?,
        shared_up: dense_weight(weights, name(Role::FfnUpShexp)?, &specs.shared_up, layout)?,
        shared_down: dense_weight(
            weights,
            name(Role::FfnDownShexp)?,
            &specs.shared_down,
            layout,
        )?,
    })
}

/// What the tiered mode takes from the loaded model (the routed expert stacks are not loaded then).
pub(super) fn moe_layer_parts<'w>(
    table: &LayerTable,
    weights: &'w CudaWeights,
    specs: &MoeSpecs,
    layer: usize,
    layout: WeightLayout,
) -> Result<LayerParts<'w>, EngError> {
    let tensor = |role| table.name(layer, role);
    Ok(LayerParts {
        router: raw_buffer(weights, tensor(Role::FfnGateInp)?, specs.router_values)?,
        router_shared: raw_buffer(weights, tensor(Role::FfnGateInpShexp)?, specs.shexp_values)?,
        shared_gate: dense_weight(
            weights,
            tensor(Role::FfnGateShexp)?,
            &specs.shared_gate,
            layout,
        )?,
        shared_up: dense_weight(weights, tensor(Role::FfnUpShexp)?, &specs.shared_up, layout)?,
        shared_down: dense_weight(
            weights,
            tensor(Role::FfnDownShexp)?,
            &specs.shared_down,
            layout,
        )?,
    })
}

/// The native embedding gather reads Q4_K blocks; the native head is one `matvec_q` over the
/// Q6_K blocks.
#[derive(Clone, Copy)]
pub(super) enum GlobalWeight<'w> {
    Tiled(&'w TiledWeight),
    Native(&'w NativeWeight),
    Bf16(&'w DeviceBuffer),
    Exl3(&'w Exl3Weight),
}

pub(super) type EmbedWeight<'w> = GlobalWeight<'w>;
pub(super) type HeadWeight<'w> = GlobalWeight<'w>;

impl GlobalWeight<'_> {
    pub(super) fn tiled(&self) -> Result<&TiledWeight, EngError> {
        match self {
            Self::Tiled(weight) => Ok(weight),
            Self::Native(_) => Err(EngError::Checkpoint(
                "step weights: the global tensor is loaded native; the tiled gather reads tiles"
                    .to_owned(),
            )),
            Self::Bf16(_) | Self::Exl3(_) => Err(EngError::Checkpoint(
                "step weights: the global tensor is loaded EXL3; the tiled gather reads tiles"
                    .to_owned(),
            )),
        }
    }

    pub(super) fn native(&self) -> Result<&NativeWeight, EngError> {
        match self {
            Self::Native(weight) => Ok(weight),
            Self::Tiled(_) => Err(EngError::Checkpoint(
                "step weights: the global tensor is loaded tiled; the native gather reads blocks"
                    .to_owned(),
            )),
            Self::Bf16(_) | Self::Exl3(_) => Err(EngError::Checkpoint(
                "step weights: the global tensor is loaded EXL3; the native gather reads blocks"
                    .to_owned(),
            )),
        }
    }
}

pub(super) fn check_native_embed(name: &str, quant: QuantType) -> Result<(), EngError> {
    if quant == QuantType::Q4K {
        return Ok(());
    }
    Err(EngError::Checkpoint(format!(
        "step weights: {name} is {quant:?}, the native embedding gather takes Q4_K blocks"
    )))
}

fn exl3_global<'w>(
    weights: &'w CudaWeights,
    name: &str,
    spec: &DenseSpec,
) -> Result<GlobalWeight<'w>, EngError> {
    match (weights.get(name), spec.role) {
        (Some(CudaWeight::Exl3 { weight, .. }), Role::Output) => {
            check_dense(name, (weight.shape[1], weight.shape[0], None), spec)?;
            Ok(GlobalWeight::Exl3(weight))
        }
        (Some(CudaWeight::Raw { ty, buffer, .. }), Role::TokenEmbd)
            if *ty == GgufType::Bf16 && buffer.bytes() == spec.rows * spec.k * 2 =>
        {
            Ok(GlobalWeight::Bf16(buffer))
        }
        _ => Err(EngError::Checkpoint(format!(
            "step weights: {name} is not the EXL3 {:?} tensor of {} rows of {}",
            spec.role, spec.rows, spec.k
        ))),
    }
}

pub(super) struct GlobalWeights<'w> {
    pub(super) embed: GlobalWeight<'w>,
    pub(super) head: GlobalWeight<'w>,
    pub(super) output_norm: &'w DeviceBuffer,
}

impl<'w> GlobalWeights<'w> {
    pub(super) fn new(
        names: &GlobalNames,
        specs: &GlobalSpecs,
        weights: &'w CudaWeights,
        layout: WeightLayout,
    ) -> Result<Self, EngError> {
        let global = |name: &str, spec: &DenseSpec| -> Result<GlobalWeight<'w>, EngError> {
            match layout {
                WeightLayout::Exl3 => exl3_global(weights, name, spec),
                WeightLayout::Tiled => Ok(GlobalWeight::Tiled(dense(weights, name, spec)?)),
                WeightLayout::Native => match weights.get(name) {
                    Some(CudaWeight::Native { weight, .. }) => {
                        check_native_dense(name, (weight.n_rows(), weight.k()), spec)?;
                        Ok(GlobalWeight::Native(weight))
                    }
                    // Decode-only hybrid: the head may ride as tiles beside a native load (the
                    // load-time override), so the step's Tiled head arm serves it.
                    Some(CudaWeight::Tiled { .. }) if name == names.head => {
                        Ok(GlobalWeight::Tiled(dense(weights, name, spec)?))
                    }
                    _ => Err(EngError::Checkpoint(format!(
                        "step weights: {name} is not a native tensor"
                    ))),
                },
            }
        };
        let embed = global(&names.embed, &specs.embed)?;
        if let GlobalWeight::Native(weight) = embed {
            check_native_embed(&names.embed, weight.quant())?;
        }
        Ok(Self {
            embed,
            head: global(&names.head, &specs.head)?,
            output_norm: raw_buffer(weights, &names.norm, specs.norm_values)?,
        })
    }
}
