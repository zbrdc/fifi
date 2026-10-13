//! What the resident step owns for a session and the weights it resolves beyond the mixers
//! (decision 0047, packet RD-9d): the arena of one row, the layer table, the shapes of the MoE,
//! embedding and head weights, and the lookups that turn them into typed weights.
//!
//! The shapes are pure host arithmetic from `StepDims` (no model number is written here) and are
//! tested on the CPU. The lookups read `CudaWeights` and are covered by the scratch device run.
//!
//! One row only: the MoE block takes `m = 1` (lane C, RD-9c C-L7), so the plan, the arena and every
//! launch of the step are made for [`STEP_ROWS`] row.
//!
//! Design guidance only, no code copied.

use super::{
    dims::{StepDims, StepOptions},
    layer_table::LayerTable,
    mixer_geometry::{DenseSpec, MixerGeometry, check_dense, check_native_dense},
    mixer_weights::dense_weight,
    moe::{MoeDims, MoeLayerWeights, StackWeights},
    step_plan::{KernelScratch, StepPlan},
    tiered_step::LayerParts,
};
use crate::{
    EngError,
    family::{
        gguf_model::{Inventory, Role},
        project::cuda::{CudaWeight, CudaWeights, WeightLayout},
    },
    kernels::cuda::{
        CudaContext, DeviceArena, DeviceBuffer, NativeWeight, TiledWeight,
        resident::{matmul_expert::ExpertStack, matvec_q::QuantType},
    },
    weights::GgufType,
};

/// Rows of one decode step: the MoE block takes one row.
pub(super) const STEP_ROWS: usize = 1;
const F32: usize = size_of::<f32>();

/// The bytes of `values` f32 values, refused (not wrapped) when the byte count overflows a
/// `usize`: an unchecked `values * F32` would wrap and silently match a wrapped size.
pub(super) fn f32_bytes(values: usize) -> Result<usize, EngError> {
    values
        .checked_mul(F32)
        .ok_or_else(|| EngError::BadRequest(format!("{values} f32 values do not fit a byte count")))
}

/// The shape of a stacked expert weight: `experts` matrices of `rows` rows of `k` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StackSpec {
    pub(super) experts: usize,
    pub(super) rows: usize,
    pub(super) k: usize,
}

/// The shapes of the weights the MoE block reads.
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
}

/// The dimensions RD-9c's block takes, from the layout record, for the one row of the step.
pub(super) fn moe_dims(dims: &StepDims) -> MoeDims {
    MoeDims::new(dims, STEP_ROWS)
}

/// Refuses a NATIVE expert stack that is not the launch's shape (MATVEC-PAR S3b): the rule of
/// [`check_stack`] plus the MMVQ row-group rule on the PER-EXPERT rows (`rows / experts % 4`).
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

/// Refuses a stacked weight whose rows, length or stack depth differ from `want`. `have` is the
/// tile's total rows, its `k` and its stack depth.
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

/// The shapes of the embedding table, the output head and the output norm.
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

/// The GGUF names of the three tensors that belong to no layer.
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

/// Everything a session builds once and the step borrows.
pub(super) struct StepResources {
    pub(super) dims: StepDims,
    pub(super) plan: StepPlan,
    pub(super) arena: DeviceArena,
    pub(super) table: LayerTable,
    pub(super) geometry: MixerGeometry,
    pub(super) names: GlobalNames,
}

impl StepResources {
    /// Plans one row, allocates the arena and builds the lookups. Allocation happens here and
    /// never inside a token.
    pub(super) fn new(
        context: &CudaContext,
        dims: StepDims,
        inventory: &Inventory,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        check_stage1_options(options)?;
        let scratch = KernelScratch::for_dims(&dims, STEP_ROWS)?;
        let plan = StepPlan::new(&dims, STEP_ROWS, options, &scratch)?;
        let arena = context.arena(&plan.arena_tensors())?;
        let table = LayerTable::new(&dims, inventory)?;
        let geometry = MixerGeometry::new(&dims)?;
        let names = GlobalNames::new(inventory)?;
        Ok(Self {
            dims,
            plan,
            arena,
            table,
            geometry,
            names,
        })
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

fn raw_buffer<'w>(
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

/// Resolves one expert stack in `layout` (MATVEC-PAR S3b): the tiles (shape rule of
/// `check_stack`) or the file's own blocks (the same shape rule plus the per-expert `rows % 4`
/// of `check_native_stack`), and a tensor of the other layout is a `Checkpoint` naming it.
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
        (Some(CudaWeight::Tiled { .. }), WeightLayout::Native)
        | (Some(CudaWeight::Native { .. }), WeightLayout::Tiled) => Err(EngError::Checkpoint(
            format!("moe weights: {name} is loaded in the other layout than the session"),
        )),
        (Some(CudaWeight::Raw { .. }), _) | (None, _) => Err(EngError::Checkpoint(format!(
            "moe weights: {name} is not a quantized stack"
        ))),
    }
}

/// The MoE weights of `layer`, shapes checked against `specs`, in the session's `layout`.
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

/// The router and shared-expert weights of `layer`, shapes checked against `specs`: what the
/// tiered mode takes from the loaded model (the routed expert stacks are not loaded then).
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

/// One of the two global gathers of the step (MATVEC-PAR S3b): the tiles of the Tiled load or
/// the file's own blocks of the Native load (the native embedding gather reads Q4_K blocks, the
/// native head is one `matvec_q` over the Q6_K blocks).
#[derive(Clone, Copy)]
pub(super) enum GlobalWeight<'w> {
    Tiled(&'w TiledWeight),
    Native(&'w NativeWeight),
}

/// The embedding table of the step, either layout.
pub(super) type EmbedWeight<'w> = GlobalWeight<'w>;
/// The output head of the step, either layout.
pub(super) type HeadWeight<'w> = GlobalWeight<'w>;

impl GlobalWeight<'_> {
    /// The tiles for the tiled gather (a native weight refuses by text).
    pub(super) fn tiled(&self) -> Result<&TiledWeight, EngError> {
        match self {
            Self::Tiled(weight) => Ok(weight),
            Self::Native(_) => Err(EngError::Checkpoint(
                "step weights: the global tensor is loaded native; the tiled gather reads tiles"
                    .to_owned(),
            )),
        }
    }

    /// The native blocks for the native gather.
    pub(super) fn native(&self) -> Result<&NativeWeight, EngError> {
        match self {
            Self::Native(weight) => Ok(weight),
            Self::Tiled(_) => Err(EngError::Checkpoint(
                "step weights: the global tensor is loaded tiled; the native gather reads blocks"
                    .to_owned(),
            )),
        }
    }
}

/// Refuses a NATIVE embedding table of a type the gather does not read (MATVEC-PAR S3b): the
/// native gather takes Q4_K blocks exactly.
pub(super) fn check_native_embed(name: &str, quant: QuantType) -> Result<(), EngError> {
    if quant == QuantType::Q4K {
        return Ok(());
    }
    Err(EngError::Checkpoint(format!(
        "step weights: {name} is {quant:?}, the native embedding gather takes Q4_K blocks"
    )))
}

/// The embedding table, the output head and the output norm, in the session's `layout`.
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
                WeightLayout::Tiled => Ok(GlobalWeight::Tiled(dense(weights, name, spec)?)),
                WeightLayout::Native => {
                    let Some(CudaWeight::Native { weight, .. }) = weights.get(name) else {
                        return Err(EngError::Checkpoint(format!(
                            "step weights: {name} is not a native tensor"
                        )));
                    };
                    check_native_dense(name, (weight.n_rows(), weight.k()), spec)?;
                    Ok(GlobalWeight::Native(weight))
                }
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
