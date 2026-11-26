//! Project directly from mapped GGUF matrices. Cold experts reuse fixed staging buffers.

use crate::family::inkling2::switches::Switch;
use std::{cell::RefCell, collections::BTreeMap};

use super::{
    DEVICE_HEADROOM, DensePlan, DenseStep, DenseUpload, Resident, bad,
    dense::{ControllerRow, ControllerWeights, DraftHead, ResidentProjection},
    upload_tensor,
};
use crate::family::inkling2::cpu_weights::Projection;
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayloadSource, Placement, VRAM_HEADROOM_BYTES},
    family::mapped_file::MappedFile,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream,
        plan::{Demand, admit},
        resident::dense_gemv::{
            DenseDtype, GgufMoeCall, GgufMoePicks, GgufMoeShape, GgufMoeWeights,
        },
    },
    sched::job::PrefillBankReport,
    weights::{GgufType, repack_nvfp4_into},
};

pub(in crate::family::inkling2) mod attention;
mod cache;
mod cold_controller;
mod controller_residency;
mod controller_workspace;
mod device_prefill;
mod grouped_controller_arithmetic;
mod residency;
mod resident_local;
pub(in crate::family::inkling2) mod resident_mlp;
mod resident_prefill;
pub(in crate::family::inkling2) mod router;
pub(in crate::family::inkling2) mod streaming;
mod token_controllers;

pub(in crate::family::inkling2) use resident_local::{
    ConvRows, LocalDecodeWidths, PrepWeights, ResidentLocalDecode,
};

pub(in crate::family::inkling2) use device_prefill::{
    DeviceRoute, DeviceRouted, ExpertPrefillRows,
};
pub(in crate::family::inkling2) use residency::{ExpertLayer, ExpertStack};
use residency::{PackedBudget, ResidentExperts};

pub(in crate::family::inkling2) struct GgufCuda {
    step: DenseStep,
    controlled_attention_step: Option<DenseStep>,
    draft_head: Option<DraftHead>,
    dense: Projections,
    token_controllers: Projections,
    packed_gguf: bool,
    group_projections: bool,
    group_attention: bool,
    batch_router: bool,
    reuse_router_input: bool,
    resident_attention: Option<attention::ResidentAttention>,
    hidden: usize,
    gate_rank: usize,
    moe_scratch: Option<ResidentMoe>,
    route_scratch: Option<router::RouteScratch>,
    decode_exact_q2_down: bool,
    batch_streamed: bool,
    prefill_resident: bool,
    device_prefill: Option<RefCell<device_prefill::Workspace>>,
    device_prefill_wmma: bool,
    device_prefill_grouped_dq: bool,
    device_prefill_exact_tile: bool,
    /// Prompt rows route on the device and only their expert grouping comes back.
    device_prompt_routing: bool,
    cold_controller: Option<cold_controller::Staging>,
    resident: ResidentExperts,
    cache: RefCell<BTreeMap<u32, cache::LayerCache>>,
    prefill_report: Option<RefCell<PrefillBankReport>>,
    stream_banks: Option<RefCell<crate::expert_tier::CudaStreamBanks>>,
    expert_codes: DeviceBuffer,
    expert_scales: DeviceBuffer,
    packed: RefCell<Packed>,
    context: CudaContext,
}

pub(in crate::family::inkling2) struct GgufDenseUpload<'a> {
    pub matrix: DenseUpload<'a>,
    pub dtype: DenseDtype,
}

type Projections = BTreeMap<String, (Resident, DenseDtype)>;

type ControllerOutput = Option<[Vec<f32>; 2]>;

pub(in crate::family::inkling2) use super::dense::{
    ControlledAttentionInput, ControlledRows, PrefillProjection, ResidentController,
};
use super::dense::{LocalInputs, LocalProjection};

pub(in crate::family::inkling2) struct LocalDecode<'a> {
    pub cursor: (usize, usize),
    pub prep: attention::LocalPrep<'a>,
    pub output: Option<attention::OutputProjection<'a>>,
}

type ControllerPayload<'a> = [&'a [u8]; 3];

enum BatchWeights<'a> {
    Resident(ControllerWeights<'a>),
    Cold(&'a cold_controller::Staging, ControllerPayload<'a>),
}

pub(in crate::family::inkling2) struct BatchController<'a> {
    cuda: &'a GgufCuda,
    step: &'a DenseStep,
    weights: BatchWeights<'a>,
    shape: [usize; 2],
    capacity: usize,
}

impl BatchController<'_> {
    pub(in crate::family::inkling2) fn project(
        &self,
        inputs: &[&[f32]],
        role: &str,
    ) -> Result<Vec<[Vec<f32>; 2]>, EngError> {
        let (weights, staged) = match self.weights {
            BatchWeights::Resident(weights) => (weights, false),
            BatchWeights::Cold(buffers, payload) => {
                (buffers.upload(&self.cuda.context, payload)?, true)
            }
        };
        let values = self
            .step
            .project_controller_batch_bounded(
                &self.cuda.context,
                inputs,
                weights,
                self.shape,
                self.capacity,
            )?
            .ok_or_else(|| bad("CUDA controller batch was not admitted"))?;
        if staged {
            static USED: std::sync::Once = std::sync::Once::new();
            USED.call_once(|| eprintln!("Inkling CUDA used staged prefill controller"));
            if self.cuda.prefill_report.is_some() {
                eprintln!(
                    "Inkling CUDA staged prefill controller {}",
                    serde_json::json!({"role": role, "rows": inputs.len()})
                );
            }
        }
        let receipt = || {
            serde_json::json!({
                "role": role, "rows": inputs.len(), "tiles": inputs.len().div_ceil(self.capacity),
            })
        };
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used multirow controller {}", receipt()));
        if self.cuda.prefill_report.is_some() {
            eprintln!("Inkling CUDA multirow controller tile report {}", receipt());
        }
        Ok(values)
    }
}

pub(in crate::family::inkling2) struct ResidentMlpController<'a> {
    context: &'a CudaContext,
    weights: ControllerWeights<'a>,
}

impl ResidentMlpController<'_> {
    pub(in crate::family::inkling2) fn run<'a>(
        &self,
        scratch: &'a mut ResidentController,
        rows: (&[Vec<f32>], &[Vec<f32>], &[f32]),
        eps: (f32, f32),
    ) -> Result<DeviceRange<'a>, EngError> {
        scratch.run(self.context, self.weights, rows, eps)
    }

    /// [`Self::run`] of `rows` pairs already written into the scratch's inputs.
    pub(in crate::family::inkling2) fn run_written<'a>(
        &self,
        scratch: &'a mut ResidentController,
        rows: usize,
        eps: (f32, f32),
    ) -> Result<DeviceRange<'a>, EngError> {
        scratch.run_written(self.context, self.weights, rows, eps)
    }
}

pub(in crate::family::inkling2) struct AttentionControllers<'a> {
    cuda: &'a GgufCuda,
    weights: [Option<ControllerWeights<'a>>; 3],
    shape: [usize; 2],
}

impl AttentionControllers<'_> {
    pub(in crate::family::inkling2) fn grouped_chained_projections<const N: usize>(
        &self,
        inputs: &[[ControlledAttentionInput<'_>; 3]],
        requests: [Option<Projection<'_>>; N],
        eps: f32,
    ) -> Result<Option<Vec<[Vec<f32>; N]>>, EngError> {
        let mut result = Vec::with_capacity(inputs.len());
        let admitted =
            self.grouped_chained_tiles(inputs, requests, (eps, &[], None), |host, _| {
                result.extend(host);
                Ok(())
            })?;
        Ok(admitted.then_some(result))
    }

    /// The controllers, projections, step and tile rows of a chained tile run over `rows` rows,
    /// or none when the run does not fit.
    #[allow(clippy::type_complexity)]
    fn chained_admission<'s, const N: usize>(
        &'s self,
        requests: [Option<Projection<'_>>; N],
        rows: usize,
        keep: &[usize],
    ) -> Result<
        Option<(
            [ControllerWeights<'s>; 3],
            [Option<ResidentProjection<'s>>; N],
            &'s DenseStep,
            usize,
        )>,
        EngError,
    > {
        let [Some(q), Some(k), Some(v)] = self.weights else {
            return Ok(None);
        };
        let Some(weights) = self
            .cuda
            .attention_group([self.shape[1]; 3], requests)
            .transpose()?
        else {
            return Ok(None);
        };
        if keep.iter().any(|&i| weights[i].is_none()) {
            return Ok(None);
        }
        let (step, capacity) = self.chained_workspace(&weights, rows, !keep.is_empty())?;
        Ok((capacity >= 2).then_some(([q, k, v], weights, step, capacity)))
    }

    /// The chained tiles of controlled rows already on the device; false, with nothing
    /// launched, when the run does not fit.
    pub(in crate::family::inkling2) fn grouped_chained_device_tiles<const N: usize>(
        &self,
        rows: &ControlledRows<'_>,
        requests: [Option<Projection<'_>>; N],
        (eps, keep): (f32, &[usize]),
        mut consume: impl FnMut(Vec<[Vec<f32>; N]>, Option<DeviceRange<'_>>) -> Result<(), EngError>,
    ) -> Result<bool, EngError> {
        if rows.hidden != self.shape[1] {
            return Ok(false);
        }
        let Some((controllers, weights, step, capacity)) =
            self.chained_admission(requests, rows.rows, keep)?
        else {
            return Ok(false);
        };
        let arithmetic = grouped_controller_arithmetic::select();
        for first in (0..rows.rows).step_by(capacity) {
            let count = (rows.rows - first).min(capacity);
            let (host, device) = step
                .controlled_attention_device_batch(
                    &self.cuda.context,
                    (rows, (first, count)),
                    controllers,
                    weights,
                    (eps, arithmetic, keep),
                )?
                .ok_or_else(|| bad("admitted CUDA controlled attention batch declined"))?;
            if host.len() != count {
                return Err(bad(
                    "CUDA controlled attention batch returned wrong row count",
                ));
            }
            consume(host, device)?;
        }
        Ok(true)
    }

    #[allow(clippy::type_complexity)]
    pub(in crate::family::inkling2) fn grouped_chained_tiles<const N: usize>(
        &self,
        inputs: &[[ControlledAttentionInput<'_>; 3]],
        requests: [Option<Projection<'_>>; N],
        (eps, keep, scales): (f32, &[usize], Option<&[[[f32; 2]; 3]]>),
        mut consume: impl FnMut(Vec<[Vec<f32>; N]>, Option<DeviceRange<'_>>) -> Result<(), EngError>,
    ) -> Result<bool, EngError> {
        let Some(([q, k, v], weights, step, capacity)) =
            self.chained_admission(requests, inputs.len(), keep)?
        else {
            return Ok(false);
        };
        if scales.is_some_and(|s| s.len() != inputs.len()) {
            return Err(bad("invalid Inkling controller gate scale row count"));
        }
        let arithmetic = grouped_controller_arithmetic::select();
        for (index, chunk) in inputs.chunks(capacity).enumerate() {
            let first = index * capacity;
            let tile_scales = scales.map(|s| &s[first..first + chunk.len()]);
            let (host, device) = step
                .controlled_attention_batch(
                    &self.cuda.context,
                    chunk,
                    [q, k, v],
                    weights,
                    (eps, arithmetic, keep, tile_scales),
                )?
                .ok_or_else(|| bad("admitted CUDA controlled attention batch declined"))?;
            if host.len() != chunk.len() {
                return Err(bad(
                    "CUDA controlled attention batch returned wrong row count",
                ));
            }
            consume(host, device)?;
        }
        if !keep.is_empty() && self.cuda.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA grouped query tiles {}",
                serde_json::json!({"rows": inputs.len(), "tile_rows": capacity,
                    "tiles": inputs.len().div_ceil(capacity),
                    "wide": !std::ptr::eq(step, &self.cuda.step),
                    "gate_upload_bytes_saved": scales.map_or(0, |s| s.len() * 6 * (self.shape[1] - 1) * 4)})
            );
        }
        Ok(true)
    }

    fn chained_workspace<const N: usize>(
        &self,
        weights: &[Option<ResidentProjection<'_>>; N],
        rows: usize,
        resident_query: bool,
    ) -> Result<(&DenseStep, usize), EngError> {
        let step = &self.cuda.step;
        let baseline = step
            .controlled_attention_batch_capacity(self.shape, *weights)?
            .unwrap_or(0);
        if resident_query
            && rows > baseline
            && let Some(wide) = &self.cuda.controlled_attention_step
        {
            let capacity = wide
                .controlled_attention_batch_capacity(self.shape, *weights)?
                .unwrap_or(0);
            let desired = controller_workspace::target_rows();
            let target = [256, 128, 64, controller_workspace::TILE_ROWS]
                .into_iter()
                .find(|&r| r <= capacity && r <= desired)
                .unwrap_or(controller_workspace::TILE_ROWS);
            if capacity >= target && target > baseline {
                return Ok((wide, target));
            }
        }
        Ok((step, baseline))
    }

    pub(in crate::family::inkling2) fn chained_projections<const N: usize>(
        &self,
        inputs: [ControlledAttentionInput<'_>; 3],
        requests: [Option<Projection<'_>>; N],
        eps: f32,
    ) -> Result<Option<[Vec<f32>; N]>, EngError> {
        let [Some(q), Some(k), Some(v)] = self.weights else {
            return Ok(None);
        };
        let Some(weights) = self.cuda.attention_group([self.shape[1]; 3], requests) else {
            return Ok(None);
        };
        self.cuda
            .step
            .controlled_attention(&self.cuda.context, inputs, [q, k, v], weights?, eps)
    }

    pub(in crate::family::inkling2) fn local_decode(
        &self,
        inputs: [ControlledAttentionInput<'_>; 3],
        requests: [Option<Projection<'_>>; 8],
        decode: LocalDecode<'_>,
    ) -> Result<Option<attention::LocalResult>, EngError> {
        let [Some(q), Some(k), Some(v)] = self.weights else {
            return Ok(None);
        };
        let eps = decode.prep.math.eps;
        self.cuda.local_decode(
            LocalInputs::Controlled(inputs, [q, k, v], eps),
            self.shape[1],
            requests,
            decode,
        )
    }

    pub(in crate::family::inkling2) fn project(
        &self,
        inputs: &[[&[f32]; 3]],
        multirow: bool,
    ) -> Result<Vec<[ControllerOutput; 3]>, EngError> {
        if multirow && inputs.len() > 1 {
            let capacity = self.cuda.step.controller_batch_capacity(self.shape)?;
            if capacity >= 2 {
                let controllers = self.weights.map(|weights| {
                    weights.map(|weights| BatchController {
                        cuda: self.cuda,
                        step: &self.cuda.step,
                        weights: BatchWeights::Resident(weights),
                        shape: self.shape,
                        capacity,
                    })
                });
                return project_controller_roles(controllers, inputs);
            }
        }
        let rows: Vec<_> = inputs
            .iter()
            .flat_map(|row| self.weights.into_iter().zip(row))
            .filter_map(|(weights, &input)| weights.map(|weights| ControllerRow { weights, input }))
            .collect();
        let values = self
            .cuda
            .step
            .controller_rows(&self.cuda.context, &rows, self.shape)?;
        if self.cuda.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA grouped controllers {}",
                serde_json::json!({"attention_rows": inputs.len(), "controller_rows": rows.len()})
            );
        }
        let mut values = values.into_iter();
        Ok(inputs
            .iter()
            .map(|_| {
                self.weights.map(|weights| {
                    weights.map(|_| {
                        values
                            .next()
                            .expect("each resident controller produced a row")
                    })
                })
            })
            .collect())
    }
}

pub(in crate::family::inkling2) fn project_controller_roles(
    controllers: [Option<BatchController<'_>>; 3],
    inputs: &[[&[f32]; 3]],
) -> Result<Vec<[ControllerOutput; 3]>, EngError> {
    let mut result = vec![[None, None, None]; inputs.len()];
    for (index, controller) in controllers.into_iter().enumerate() {
        let Some(controller) = controller else {
            continue;
        };
        let role = ["q", "k", "v"][index];
        let gates: Vec<_> = inputs.iter().map(|row| row[index]).collect();
        for (row, projection) in result.iter_mut().zip(controller.project(&gates, role)?) {
            row[index] = Some(projection);
        }
    }
    Ok(result)
}

fn projection_bytes(matrices: &[GgufDenseUpload<'_>]) -> u64 {
    matrices
        .iter()
        .map(|item| item.matrix.codes.len() as u64)
        .sum()
}

fn upload_projections(
    context: &CudaContext,
    matrices: &[GgufDenseUpload<'_>],
) -> Result<Projections, EngError> {
    matrices
        .iter()
        .map(|GgufDenseUpload { matrix, dtype }| {
            let codes = upload_tensor(context, matrix.codes)?;
            let weight = Resident {
                codes,
                shape: matrix.shape,
            };
            Ok((matrix.name.to_owned(), (weight, *dtype)))
        })
        .collect()
}

fn projection_group(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(stem, _)| stem)
}

fn upload_optional_projections(
    context: &CudaContext,
    matrices: &[GgufDenseUpload<'_>],
    dense: &mut Projections,
    headroom: u64,
) -> Result<(), EngError> {
    // A controller needs all three matrices resident to use one upload and readback.
    for group in
        matrices.chunk_by(|a, b| projection_group(a.matrix.name) == projection_group(b.matrix.name))
    {
        let free = (context.memory_info()?.free as u64).saturating_sub(headroom);
        if projection_bytes(group) > free {
            continue;
        }
        let uploaded = upload_projections(context, group)?;
        if context.memory_info()?.free as u64 >= headroom {
            dense.extend(uploaded);
        }
    }
    Ok(())
}

pub(in crate::family::inkling2) struct Options {
    pub direct: bool,
    pub packed: bool,
    pub group_projections: bool,
    pub stream: bool,
    pub budget: u64,
    pub cache_reserve: Option<u64>,
    pub resident_attention: Option<attention::Plan>,
    pub allow_device_routing: bool,
    pub context_limit: usize,
}

struct ResidentMoe {
    shape: GgufMoeShape,
    scratch: DeviceBuffer,
}

struct Packed {
    codes: Vec<u8>,
    scales: Vec<u8>,
}

pub(in crate::family::inkling2) struct ExpertView<'a> {
    pub id: ExpertId,
    pub part: usize,
    pub kind: GgufType,
    pub shape: [usize; 2],
    pub bytes: &'a [u8],
    pub factor: f32,
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn context(&self) -> &CudaContext {
        &self.context
    }

    pub(in crate::family::inkling2) fn resident_attention(
        &self,
    ) -> Option<(&CudaContext, &attention::ResidentAttention)> {
        self.resident_attention
            .as_ref()
            .map(|attn| (&self.context, attn))
    }

    pub(in crate::family::inkling2) fn relative_attention_projection(
        &self,
        name: &str,
        shape: [usize; 2],
    ) -> Result<Option<attention::OutputProjection<'_>>, EngError> {
        let Some(attention) = &self.resident_attention else {
            return Ok(None);
        };
        if attention.projected_rows() != Some(shape[0]) {
            return Ok(None);
        }
        let Some((weight, dtype)) = self.dense.get(name) else {
            return Ok(None);
        };
        if weight.shape != shape || self.step.uses_q8_activations(*dtype) {
            return Ok(None);
        }
        Ok(Some(attention::OutputProjection {
            weight: weight.codes.range(0, weight.codes.bytes())?,
            dtype: *dtype,
            rows: shape[0],
        }))
    }

    pub(in crate::family::inkling2) fn load(
        dense: Vec<GgufDenseUpload<'_>>,
        spec: &super::TextSpec,
        layers: Vec<ExpertLayer<'_>>,
        file: &MappedFile,
        Options {
            direct,
            packed,
            group_projections,
            stream,
            budget,
            cache_reserve,
            resident_attention,
            allow_device_routing,
            context_limit,
        }: Options,
    ) -> Result<Self, EngError> {
        let hidden = spec.hidden_size as usize;
        let ff = spec.intermediate_size as usize;
        let rows = hidden.max(2 * ff);
        let shape = dense.iter().fold([rows, hidden.max(ff)], |[r, c], item| {
            [r.max(item.matrix.shape[0]), c.max(item.matrix.shape[1])]
        });
        let elements = rows * hidden.max(ff);
        let stage = [2 * elements, elements / 16];
        let group_attention = Switch::GroupAttention.on();
        let [rows, columns] = shape;
        let attention_rows = dense
            .iter()
            .filter(|item| item.matrix.name.contains(".attn."))
            .map(|item| item.matrix.shape[0])
            .max()
            .unwrap_or(0);
        let packed_outputs = if group_attention {
            rows.max(3 * hidden + 8 * attention_rows)
        } else {
            rows
        };
        let batch_experts = packed && Switch::BatchExperts.on();
        let device_route = allow_device_routing
            && group_projections
            && batch_experts
            && Switch::DeviceRouting.on();
        let route_shape = device_route.then(|| router::route_shape(spec));
        let route_plan = route_shape
            .map(|shape| router::RoutePlan::new(shape, &layers))
            .transpose()?;
        let route_bytes = route_plan.as_ref().map_or(0, router::RoutePlan::bytes);
        let route_input = route_shape.map_or(0, |shape| hidden + shape.n_expert + 1);
        let plan = DensePlan::new(
            [packed_outputs, columns.max(route_input)],
            hidden,
            spec.core.gate_rank as usize,
        )?;
        let (required, mut optional): (Vec<_>, Vec<_>) = dense
            .into_iter()
            .partition(|item| item.dtype != DenseDtype::F32);
        let tensors = projection_bytes(&required);
        let (admission, _) = CudaContext::fixed_projection_plan(shape)?;
        let context = CudaContext::with_admission_cache(&admission, Switch::AllocCache.on())?;
        context.set_dense_group_policy(super::dense_group_policy());
        let prefill_resident = packed && Switch::ResidentPrefill.on();
        let device_prefill = packed && Switch::DeviceExpertPrefill.on();
        let stream_plan = stream.then(|| streaming::plan(hidden, ff)).transpose()?;
        let mut stream_bytes = stream_plan.map_or(0, |plan| plan.device_bytes());
        let moe_shape = GgufMoeShape {
            hidden,
            middle: ff,
            experts: spec.n_routed_experts as usize,
        };
        let moe_bytes = if batch_experts {
            moe_shape
                .scratch_bytes(spec.num_experts_per_tok as usize)
                .ok_or_else(|| bad("CUDA GGUF expert group exceeds its limits"))?
        } else {
            0
        };
        let demand = Demand {
            tensors,
            work: (stage[0] + stage[1] + plan.device_bytes + moe_bytes + route_bytes) as u64
                + stream_bytes,
            kv_cache: 0,
            headroom: DEVICE_HEADROOM,
        };
        admit(&demand, context.memory_info()?.free as u64)?;
        let q8_activations = Switch::Q8Activations.on();
        let step = DenseStep::new(&context, &plan)?.with_q8_activations(q8_activations);
        let expert_codes = context.allocate(stage[0])?;
        let expert_scales = context.allocate(stage[1])?;
        let mut dense = upload_projections(&context, &required)?;
        let mut stream_banks = stream_plan
            .map(|plan| crate::expert_tier::CudaStreamBanks::new(&context, plan))
            .transpose()?;
        let moe_scratch = (moe_bytes > 0)
            .then(|| {
                context.allocate(moe_bytes).map(|scratch| ResidentMoe {
                    shape: moe_shape,
                    scratch,
                })
            })
            .transpose()?;
        let route_scratch = route_plan.map(|plan| plan.upload(&context)).transpose()?;
        let experts = spec.n_routed_experts as u32;
        let full_expert_fit = packed && residency::packed_model_fits(&context, &layers, budget)?;
        let mut batch_streamed = false;
        // Keep the exact resident placement when the model fits with the original banks.
        // A streamed group needs one larger slot; the second still serves serial prefill.
        if batch_experts
            && !full_expert_fit
            && Switch::BatchStreamedExperts.on()
            && let (Some(banks), Some(original)) = (&mut stream_banks, stream_plan)
        {
            let plan = streaming::group_plan(
                &layers,
                experts as usize,
                spec.num_experts_per_tok as usize,
                original,
            )?;
            let free = (context.memory_info()?.free as u64).saturating_sub(VRAM_HEADROOM_BYTES);
            if plan.bank_bytes() <= free {
                banks.replace_first(&context, plan)?;
                stream_bytes = original.bank_bytes() + plan.bank_bytes();
                batch_streamed = true;
            }
        }
        optional.sort_by_key(|item| !item.matrix.name.ends_with(".mlp.gate.weight"));
        if !full_expert_fit {
            upload_optional_projections(&context, &optional, &mut dense, VRAM_HEADROOM_BYTES)?;
        }
        let resident = if packed {
            ResidentExperts::load_packed(
                &context,
                experts,
                layers,
                file,
                PackedBudget {
                    total: budget,
                    cache: cache_reserve,
                },
            )?
        } else {
            ResidentExperts::load_nvfp4(&context, experts, layers, file, direct)?
        };
        let resident_attention = match resident_attention {
            Some(plan) if resident.placement().full_layers().len() == spec.stored_layers() => {
                let charge_reserve = Switch::ResidentAttentionReserve.on();
                plan.load(&context, charge_reserve)?
            }
            Some(_) => {
                eprintln!("Inkling CUDA relative attention requires resident expert weights");
                None
            }
            None => None,
        };
        // Admit optional dense projections after persistent state and resident experts.
        if full_expert_fit {
            let headroom = resident_attention.as_ref().map_or(
                VRAM_HEADROOM_BYTES,
                attention::ResidentAttention::projection_headroom,
            );
            upload_optional_projections(&context, &optional, &mut dense, headroom)?;
            if allow_device_routing
                && group_projections
                && controller_residency::enabled()
                && resident_attention.is_some()
                && let Some(workspace) =
                    controller_residency::expert_workspace(spec, context_limit, device_prefill)
            {
                controller_residency::admit(
                    &context,
                    &optional,
                    &mut dense,
                    [headroom, workspace],
                )?;
            }
        }
        let sum_bytes = |kind: Option<DenseDtype>| {
            dense
                .values()
                .filter(|(_, dtype)| kind.is_none_or(|kind| kind == *dtype))
                .map(|(weight, _)| weight.codes.bytes() as u64)
                .sum::<u64>()
        };
        if let Some(scratch) = &route_scratch {
            let layers: Vec<_> = resident
                .placement()
                .full_layers()
                .into_iter()
                .filter(|&layer| {
                    scratch.has_layer(layer)
                        && dense
                            .get(&format!("model.llm.layers.{layer}.mlp.gate.weight"))
                            .is_some_and(|(_, dtype)| *dtype == DenseDtype::F32)
                })
                .collect();
            let visits = spec
                .visits()
                .iter()
                .filter(|visit| layers.contains(&(visit.weight_layer as u32)))
                .count();
            eprintln!(
                "Inkling GGUF CUDA device routing {}",
                serde_json::json!({"layers": layers, "visits": visits})
            );
        }
        eprintln!(
            "Inkling GGUF CUDA projections {}",
            serde_json::json!({"resident_projection_count": dense.len(), "resident_bf16_bytes": sum_bytes(Some(DenseDtype::Bf16)), "resident_f32_bytes": sum_bytes(Some(DenseDtype::F32)), "resident_dense_bytes": sum_bytes(None), "packed_gguf": packed, "q8_1_projection_activations": q8_activations, "batched_resident_experts": moe_scratch.is_some(), "batched_streamed_experts": batch_streamed, "batched_resident_prefill": prefill_resident,
                "direct_resident_nvfp4": direct || packed, "cold_stream_device_bytes": stream_bytes, "packed_expert_staging_bytes": stage[0] + stage[1], "dense_stream_arena_bytes": plan.device_bytes,
                "resident_expert_bytes": resident.bytes(),
                "resident_expert_layers": resident.placement().full_layers(),
                "resident_expert_count": resident.placement().resident_experts()})
        );
        let profile = stream && Switch::Profile.on();
        if profile {
            let admitted: Vec<_> = optional
                .iter()
                .map(|item| item.matrix.name)
                .filter(|name| dense.contains_key(*name))
                .collect();
            eprintln!(
                "Inkling GGUF CUDA optional projections {}",
                serde_json::json!(admitted)
            );
        }
        let cold_controller = if full_expert_fit
            && group_projections
            && spec.core.gate_rank != 0
            && Switch::StagePrefillControllers.on()
            && optional.iter().any(|item| {
                item.matrix.name.contains(".controllers.") && !dense.contains_key(item.matrix.name)
            }) {
            cold_controller::Staging::admit(&context, [spec.core.gate_rank as usize, hidden])?
        } else {
            None
        };
        // Allocate only after base placement and cold controller staging are final.
        let controlled_attention_step = if full_expert_fit
            && group_projections
            && device_prefill
            && resident_attention.is_some()
            && controller_workspace::enabled()
            && let Some(max_output) = attention_rows.checked_mul(8)
            && let Some(expert_bytes) =
                controller_residency::expert_workspace(spec, context_limit, true)
        {
            controller_workspace::admit(
                &context,
                [spec.core.gate_rank as usize, hidden, max_output],
                expert_bytes,
            )?
        } else {
            None
        };
        // Do not change prefill/host placement to qualify decode. The extra
        // matrices, if admitted, belong exclusively to the token chain.
        let token_controllers = if full_expert_fit
            && group_projections
            && resident_attention.is_some()
            && Switch::ResidentToken.on()
        {
            token_controllers::load(
                &context,
                &optional,
                &dense,
                controller_residency::expert_workspace(spec, context_limit, device_prefill),
            )?
        } else {
            Projections::new()
        };
        Ok(Self {
            step,
            token_controllers,
            controlled_attention_step,
            draft_head: None,
            dense,
            packed_gguf: packed,
            group_projections,
            group_attention,
            batch_router: Switch::BatchRouter.on(),
            resident_attention,
            hidden,
            gate_rank: spec.core.gate_rank as usize,
            moe_scratch,
            decode_exact_q2_down: route_scratch.is_some()
                && Switch::DecodeExactQ2Down.on()
                && context
                    .device_attributes()
                    .is_ok_and(|attrs| attrs.compute_capability >= (7, 5)),
            route_scratch,
            batch_streamed,
            prefill_resident,
            reuse_router_input: device_prefill && Switch::PrefillRouterInputReuse.on(),
            device_prefill: device_prefill.then(RefCell::default),
            device_prefill_wmma: device_prefill && Switch::DeviceExpertPrefillWmma.on(),
            device_prefill_grouped_dq: device_prefill
                && Switch::DeviceExpertPrefillGroupedDq.on()
                && context
                    .device_attributes()
                    .is_ok_and(|attrs| attrs.compute_capability >= (7, 5)),
            device_prompt_routing: device_prefill && Switch::DeviceRouting.on(),
            device_prefill_exact_tile: device_prefill
                && Switch::DeviceExpertPrefillExactTile.on()
                // The kernels use SM75 tensor-core instructions. A refused diagnostic keeps
                // the baseline.
                && context
                    .device_attributes()
                    .is_ok_and(|attrs| attrs.compute_capability >= (7, 5)),
            cold_controller,
            resident,
            cache: RefCell::default(),
            prefill_report: profile.then(RefCell::default),
            stream_banks: stream_banks.map(RefCell::new),
            expert_codes,
            expert_scales,
            packed: RefCell::new(Packed {
                codes: vec![0; elements / 2],
                scales: vec![0; elements / 16],
            }),
            context,
        })
    }

    pub(in crate::family::inkling2) fn enable_draft_head(
        &mut self,
        cap: f32,
    ) -> Result<(), EngError> {
        let (weight, dtype) = self.draft_head_weight()?;
        let head = DraftHead::new(&self.context, weight.shape, *dtype, cap)?;
        eprintln!("Inkling CUDA DSpark corrected head enabled");
        self.draft_head = Some(head);
        Ok(())
    }

    pub(in crate::family::inkling2) fn draft_greedy(
        &self,
        rows: [&[f32]; 2],
        vocab: usize,
        cap: f32,
    ) -> Result<Option<u32>, EngError> {
        let Some(head) = &self.draft_head else {
            return Ok(None);
        };
        let (weight, dtype) = self.draft_head_weight()?;
        if weight.shape[0] != vocab {
            return Err(bad("CUDA draft head vocabulary changed after admission"));
        }
        head.choose(&self.step, &self.context, (weight, *dtype), rows, cap)
            .map(Some)
    }

    fn draft_head_weight(&self) -> Result<&(Resident, DenseDtype), EngError> {
        self.dense
            .get(crate::family::inkling2::OUTPUT_HEAD)
            .ok_or_else(|| bad("CUDA draft head requires a resident GGUF vocabulary projection"))
    }

    pub(in crate::family::inkling2) fn project(
        &self,
        name: &str,
        rows: usize,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        let (resident, dtype) = self.dense.get(name)?;
        if resident.shape != [rows, x.len()] {
            return Some(Err(bad(format!(
                "bad GGUF CUDA projection shape for {name}"
            ))));
        }
        Some(
            if *dtype == DenseDtype::F32 && name.ends_with(".mlp.gate.weight") {
                self.step.project_router(&self.context, resident, x)
            } else if *dtype == DenseDtype::Bf16 {
                self.step.project(&self.context, resident, x)
            } else {
                resident
                    .codes
                    .range(0, resident.codes.bytes())
                    .and_then(|raw| {
                        self.step.project_gguf(
                            &self.context,
                            None,
                            (*dtype, raw, resident.shape, 1.0),
                            x,
                        )
                    })
            },
        )
    }

    pub(in crate::family::inkling2) fn resident_head_supported(&self, shape: [usize; 2]) -> bool {
        self.dense
            .get(crate::family::inkling2::OUTPUT_HEAD)
            .is_some_and(|(w, d)| {
                w.shape == shape && d.is_packed_gguf() && !self.step.uses_q8_activations(*d)
            })
    }

    pub(in crate::family::inkling2) fn enqueue_resident_head(
        &self,
        stream: &crate::kernels::cuda::DeviceStream,
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
        cap: f32,
    ) -> Result<(), EngError> {
        use crate::kernels::cuda::resident::dense_gemv::{DenseGemvCall, DenseGemvShape};
        let (w, d) = self
            .dense
            .get(crate::family::inkling2::OUTPUT_HEAD)
            .ok_or_else(|| bad("missing resident head"))?;
        let [rows, cols] = w.shape;
        self.context.dense_gemv(
            stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(1, rows, cols),
                dtype: *d,
                w: w.codes.range(0, w.codes.bytes())?,
                x,
                y,
            },
        )?;
        self.context.softcap_in_place(stream, y, rows, cap)
    }

    pub(in crate::family::inkling2) fn capped_head(
        &self,
        x: &[f32],
        rows: usize,
        cap: f32,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some((weight, dtype)) = self.dense.get(crate::family::inkling2::OUTPUT_HEAD) else {
            return Ok(None);
        };
        if weight.shape != [rows, x.len()]
            || !dtype.is_packed_gguf()
            || !cap.is_finite()
            || cap <= 0.0
        {
            return Ok(None);
        }
        let output = self
            .step
            .capped_gguf_head(&self.context, (weight, *dtype), x, cap)?;
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used resident vocabulary softcap"));
        Ok(Some(output))
    }

    pub(in crate::family::inkling2) fn batch_prefill_router(&self, layer: u32, name: &str) -> bool {
        // The cold path interleaves route observation with prediction for the
        // next token. Keep its existing ordering.
        self.batch_router
            && !self.resident.placement().has_cold(layer)
            && self
                .dense
                .get(name)
                .is_some_and(|(_, dtype)| *dtype == DenseDtype::F32)
    }

    pub(in crate::family::inkling2) fn prefill_router_logits(
        &self,
        name: &str,
        inputs: &[&[f32]],
        layer: u32,
        prompt: bool,
    ) -> Result<(Vec<f32>, Option<DeviceRange<'_>>), EngError> {
        let (weight, _) = self
            .dense
            .get(name)
            .ok_or_else(|| bad("missing resident prefill router projection"))?;
        let reuse = prompt && self.reuse_router_input && self.prefills_device_layer(layer);
        self.step.project_router_rows(
            &self.context,
            weight,
            inputs,
            reuse.then_some(&self.expert_codes),
        )
    }

    pub(in crate::family::inkling2) fn resident_prefill_router_shape(
        &self,
        layer: u32,
        name: &str,
        rows: usize,
    ) -> Option<[usize; 2]> {
        let (weight, _) = self.dense.get(name)?;
        (self.batch_prefill_router(layer, name) && self.step.admits_resident_router(weight, rows))
            .then_some(weight.shape)
    }

    pub(in crate::family::inkling2) fn resident_prefill_router_logits(
        &self,
        name: &str,
        input: DeviceRange<'_>,
    ) -> Result<Vec<f32>, EngError> {
        let (weight, _) = self
            .dense
            .get(name)
            .filter(|(_, dtype)| *dtype == DenseDtype::F32)
            .ok_or_else(|| bad("missing F32 CUDA prefill router"))?;
        self.step
            .project_resident_router_rows(&self.context, weight, input)
    }

    pub(in crate::family::inkling2) fn attention_projections<const N: usize>(
        &self,
        inputs: [&[f32]; 3],
        requests: [Option<Projection<'_>>; N],
    ) -> Option<Result<[Vec<f32>; N], EngError>> {
        let weights = self.attention_group(inputs.map(<[f32]>::len), requests)?;
        Some(weights.and_then(|weights| self.step.project_group(&self.context, inputs, weights)))
    }

    pub(in crate::family::inkling2) fn attention_projections_rows<const N: usize>(
        &self,
        inputs: &[[&[f32]; 3]],
        requests: [Option<Projection<'_>>; N],
        allow_q8_row_reuse: bool,
    ) -> Result<Option<Vec<[Vec<f32>; N]>>, EngError> {
        let Some(first) = inputs.first() else {
            return Ok(None);
        };
        let Some(weights) = self.attention_group(first.map(<[f32]>::len), requests) else {
            return Ok(None);
        };
        let projected =
            self.step
                .project_group_rows(&self.context, inputs, weights?, allow_q8_row_reuse)?;
        if projected.is_some() && self.prefill_report.is_some() {
            eprintln!(
                "Inkling CUDA batched attention projections: {} rows",
                inputs.len()
            );
        }
        Ok(projected)
    }

    pub(in crate::family::inkling2) fn prefill_projection(
        &self,
        widths: [usize; 3],
        requests: [Option<Projection<'_>>; 8],
    ) -> Result<Option<PrefillProjection<'_>>, EngError> {
        if !self.packed_gguf {
            return Ok(None);
        }
        let Some(weights) = self.attention_group(widths, requests).transpose()? else {
            return Ok(None);
        };
        if weights[..6].iter().any(Option::is_none) {
            return Ok(None);
        }
        let reuse = Switch::PrefillReuseExpertStage.on();
        if reuse {
            static USED: std::sync::Once = std::sync::Once::new();
            USED.call_once(|| {
                eprintln!("Inkling CUDA reused expert staging for attention projection")
            });
        }
        // Cold-expert calls upload each weight afresh and fence before returning;
        // attention finishes consuming its sidecar before this stage is reused.
        Ok(Some(PrefillProjection {
            context: &self.context,
            step: &self.step,
            weights,
            scratch: reuse.then_some(&self.expert_codes),
        }))
    }

    pub(in crate::family::inkling2) fn plain_local_decode(
        &self,
        inputs: [&[f32]; 3],
        requests: [Option<Projection<'_>>; 8],
        decode: LocalDecode<'_>,
    ) -> Result<Option<attention::LocalResult>, EngError> {
        self.local_decode(
            LocalInputs::Plain(inputs),
            inputs[0].len(),
            requests,
            decode,
        )
    }

    fn local_decode(
        &self,
        inputs: LocalInputs<'_>,
        hidden: usize,
        requests: [Option<Projection<'_>>; 8],
        decode: LocalDecode<'_>,
    ) -> Result<Option<attention::LocalResult>, EngError> {
        let Some(attention) = &self.resident_attention else {
            return Ok(None);
        };
        if !self
            .step
            .local_inputs_fit(hidden, matches!(inputs, LocalInputs::Controlled(..)))
        {
            return Ok(None);
        }
        let Some(weights) = self.attention_group([hidden; 3], requests).transpose()? else {
            return Ok(None);
        };
        let [
            Some(q),
            Some(k),
            Some(v),
            Some(rel),
            Some(sinks),
            Some(gates),
            None,
            None,
        ] = weights
        else {
            return Ok(None);
        };
        let projections = LocalProjection {
            dense: &self.step,
            inputs,
            weights: [q, k, v, rel, sinks, gates],
        };
        attention.step_local(
            &self.context,
            decode.cursor,
            decode.prep,
            projections,
            decode.output,
        )
    }

    fn attention_group<const N: usize>(
        &self,
        widths: [usize; 3],
        requests: [Option<Projection<'_>>; N],
    ) -> Option<Result<[Option<ResidentProjection<'_>>; N], EngError>> {
        if !self.group_attention {
            return None;
        }
        let mut weights = [None; N];
        for (request, out) in requests.into_iter().zip(&mut weights) {
            let Some(p) = request else { continue };
            let (weight, dtype) = self.dense.get(p.name)?;
            if weight.shape != [p.rows, widths[p.input]] {
                return Some(Err(bad(format!(
                    "bad GGUF CUDA projection shape for {}",
                    p.name
                ))));
            }
            *out = Some(ResidentProjection {
                weight,
                dtype: *dtype,
                input: p.input,
            });
        }
        Some(Ok(weights))
    }

    pub(in crate::family::inkling2) fn controller(
        &self,
        names: [&str; 3],
        x: &[f32],
        shape: [usize; 2],
    ) -> Option<Result<[Vec<f32>; 2], EngError>> {
        Some(
            self.controller_weights(names, x.len(), shape)?
                .and_then(|weights| self.step.controller(&self.context, weights, x)),
        )
    }

    fn controller_weights(
        &self,
        names: [&str; 3],
        input_width: usize,
        shape: [usize; 2],
    ) -> Option<Result<ControllerWeights<'_>, EngError>> {
        self.resolve_controller_weights(names, input_width, shape, false)
    }

    fn resolve_controller_weights(
        &self,
        names: [&str; 3],
        input_width: usize,
        [rank, hidden]: [usize; 2],
        token: bool,
    ) -> Option<Result<ControllerWeights<'_>, EngError>> {
        if !self.group_projections {
            return None;
        }
        let shapes = [[rank, input_width], [hidden, rank], [hidden, rank]];
        let [a, b, c] = names.map(|name| {
            self.dense.get(name).or_else(|| {
                if token {
                    self.token_controllers.get(name)
                } else {
                    None
                }
            })
        });
        let weights = [a?, b?, c?];
        if weights
            .iter()
            .zip(shapes)
            .any(|((w, _), shape)| w.shape != shape)
        {
            return Some(Err(bad("bad GGUF CUDA controller projection shape")));
        }
        Some(Ok(weights.map(|(w, dtype)| (w, *dtype))))
    }

    pub(in crate::family::inkling2) fn resident_mlp_controller(
        &self,
        names: [&str; 3],
        shape @ [_, hidden, rank]: [usize; 3],
    ) -> Result<Option<ResidentMlpController<'_>>, EngError> {
        Ok(self
            .controller_weights(names, 2 * hidden, [rank, hidden])
            .transpose()?
            .filter(|&weights| ResidentController::admits(shape, weights))
            .map(|weights| ResidentMlpController {
                context: &self.context,
                weights,
            }))
    }

    pub(in crate::family::inkling2) fn stages_prefill_controllers(&self) -> bool {
        self.cold_controller.is_some()
    }

    pub(in crate::family::inkling2) fn batch_controller<'a>(
        &'a self,
        names: [&str; 3],
        shape: [usize; 2],
        rows: usize,
        cold_payload: impl FnOnce() -> Result<Option<ControllerPayload<'a>>, EngError>,
    ) -> Result<Option<BatchController<'a>>, EngError> {
        let capacity = self.step.controller_batch_capacity(shape)?;
        if rows < 2 || capacity < 2 {
            return Ok(None);
        }
        let weights = match self
            .controller_weights(names, 2 * shape[1], shape)
            .transpose()?
        {
            Some(weights) => BatchWeights::Resident(weights),
            None => {
                let Some(staging) = &self.cold_controller else {
                    return Ok(None);
                };
                let Some(payload) = cold_payload()? else {
                    return Ok(None);
                };
                BatchWeights::Cold(staging, payload)
            }
        };
        // The chained attention tile was synchronously consumed before the
        // MLP controller. Borrow its workspace, not any resident weights.
        let target = controller_workspace::target_rows();
        let mut selected = (&self.step, capacity);
        if target > capacity
            && Switch::PrefillWeightMlp.on()
            && let Some(wide) = &self.controlled_attention_step
        {
            let cap = wide.controller_batch_capacity(shape)?.min(target);
            if cap > capacity {
                selected = (wide, cap);
            }
        }
        Ok(Some(BatchController {
            cuda: self,
            step: selected.0,
            weights,
            shape,
            capacity: selected.1,
        }))
    }

    pub(in crate::family::inkling2) fn attention_controllers(
        &self,
        names: [[&str; 3]; 3],
        shape: [usize; 2],
        rows: usize,
    ) -> Result<Option<AttentionControllers<'_>>, EngError> {
        let mut weights = [None; 3];
        for (weight, names) in weights.iter_mut().zip(names) {
            *weight = self
                .controller_weights(names, 2 * shape[1], shape)
                .transpose()?;
        }
        if weights.iter().flatten().count().saturating_mul(rows) < 2
            || self.step.controller_capacity(shape[1])? < 2
        {
            return Ok(None);
        }
        Ok(Some(AttentionControllers {
            cuda: self,
            weights,
            shape,
        }))
    }

    pub(in crate::family::inkling2) fn expert_dtype(&self, kind: GgufType) -> Option<DenseDtype> {
        if self.packed_gguf || matches!(kind, GgufType::Bf16 | GgufType::Nvfp4) {
            DenseDtype::from_gguf(kind)
        } else {
            None
        }
    }

    pub(in crate::family::inkling2) fn expert(
        &self,
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        if self.packed_gguf {
            let dtype = self.expert_dtype(expert.kind)?;
            return (self.resident.placement().placement(expert.id) == Placement::Gpu)
                .then(|| self.resident_packed_expert(expert, dtype, x));
        }
        match expert.kind {
            GgufType::Bf16 => Some(self.bf16_expert(expert, x)),
            GgufType::Nvfp4 => {
                if self.resident.placement().placement(expert.id) != Placement::Gpu
                    && !can_repack(expert.bytes)
                {
                    return None;
                }
                Some(self.nvfp4_expert(expert, x))
            }
            _ => None,
        }
    }

    pub(in crate::family::inkling2) fn batches_expert_layer(
        &self,
        layer: u32,
        dtypes: [DenseDtype; 2],
    ) -> bool {
        self.moe_scratch.is_some()
            && (!self.resident.placement().has_cold(layer)
                || (self.batch_streamed
                    && self.streams_layer(layer)
                    && !dtypes
                        .into_iter()
                        .any(|dtype| self.step.uses_q8_activations(dtype))))
    }

    pub(in crate::family::inkling2) fn expert_group(
        &self,
        source: &mut impl ExpertPayloadSource,
        group: streaming::ExpertGroup<'_>,
    ) -> Result<Vec<f32>, EngError> {
        if self.streams_layer(group.route.layer) {
            return self.stream_expert_group(source, &group);
        }
        let stacks = self.resident.original_layer(group.route.layer)?;
        self.run_expert_group(None, &group, GgufMoeWeights::Stacks(stacks))
    }

    fn run_expert_group(
        &self,
        stream: Option<&DeviceStream>,
        group: &streaming::ExpertGroup<'_>,
        weights: GgufMoeWeights<'_>,
    ) -> Result<Vec<f32>, EngError> {
        let moe = self
            .moe_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA GGUF expert scratch"))?;
        if group.route.input.len() != moe.shape.hidden {
            return Err(bad("CUDA GGUF expert group input width differs"));
        }
        let scratch = moe.scratch.range(0, moe.scratch.bytes())?;
        self.step
            .run_layer(&self.context, stream, group.route.input, |stream, x, y| {
                self.context.gguf_moe(
                    stream,
                    GgufMoeCall {
                        shape: moe.shape,
                        dtypes: group.dtypes,
                        caps: group.route.caps,
                        picks: GgufMoePicks::Host(group.picks),
                        weights,
                        x,
                        scratch,
                        y,
                    },
                )
            })
    }

    fn resident_packed_expert(
        &self,
        expert: &ExpertView<'_>,
        dtype: DenseDtype,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let Some(residency::Weight::Original(raw)) =
            self.resident
                .ranges(expert.id, expert.part, [expert.bytes.len(), 0])?
        else {
            return Err(bad("resident GGUF expert has no original packed weight"));
        };
        self.step.project_gguf(
            &self.context,
            None,
            (dtype, raw, expert.shape, expert.factor),
            x,
        )
    }

    fn bf16_expert(&self, expert: &ExpertView<'_>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.context
            .upload_at(&self.expert_codes, 0, expert.bytes)?;
        let mut output =
            self.step
                .project_buffer(&self.context, &self.expert_codes, expert.shape, x)?;
        if expert.factor != 1.0 {
            for value in &mut output {
                *value *= expert.factor;
            }
        }
        Ok(output)
    }

    fn nvfp4_expert(&self, expert: &ExpertView<'_>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if expert.factor == 0.0 {
            return Ok(vec![0.0; expert.shape[0]]);
        }
        let n = expert.shape[0] * expert.shape[1];
        if let Some(buffers) = self
            .resident
            .ranges(expert.id, expert.part, [n / 2, n / 16])?
        {
            return match buffers {
                residency::Weight::Original(raw) => self.step.project_gguf(
                    &self.context,
                    None,
                    (DenseDtype::GgufNvfp4, raw, expert.shape, expert.factor),
                    x,
                ),
                residency::Weight::Converted(pair) => self.project_nvfp4(pair, expert, x),
            };
        }
        let Packed { codes, scales } = &mut *self.packed.borrow_mut();
        let (codes, scales) = (&mut codes[..n / 2], &mut scales[..n / 16]);
        repack_nvfp4_into(expert.bytes, codes, scales)?;
        self.context.upload_at(&self.expert_codes, 0, codes)?;
        self.context.upload_at(&self.expert_scales, 0, scales)?;
        self.project_nvfp4(
            [
                self.expert_codes.range(0, codes.len())?,
                self.expert_scales.range(0, scales.len())?,
            ],
            expert,
            x,
        )
    }

    fn project_nvfp4(
        &self,
        [codes, scales]: [DeviceRange<'_>; 2],
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.context
            .nvfp4_matvec_range(codes, scales, expert.factor, expert.shape, x)
    }
}

pub(in crate::family::inkling2) fn can_repack(bytes: &[u8]) -> bool {
    let (blocks, tail) = bytes.as_chunks::<36>();
    // The GGUF unsigned scale 0xff has no E4M3FN representation.
    tail.is_empty() && blocks.iter().all(|b| !b[..4].contains(&0xff))
}
