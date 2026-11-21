//! Header-only context admission for Qwen loads that opt into disk staging.

use std::path::Path;

use super::{
    chunk_plan::ChunkCost, dense_stage::DenseStage, executor::device_reserve, ops::Bytes,
    weights::ExpertSlices,
};
use crate::{
    EngError,
    family::{
        admission_sketch::largest_fitting_context,
        keys::{KeyClass, KeyRole, Projection, classify},
        mlx_budget::DeviceBudget,
        prefill::mlx::admitted_chunk,
        quantization::{ModuleQuantization, QuantizationMap},
        qwen36::{
            host::HostShape,
            host_plan::MIN_DERIVED_CONTEXT,
            settings::{Activation, DEFAULT_SESSION_CAP, KvDtype, MlxPlacement, shape_of},
        },
        spec::FamilySpec,
        stored::{SlicedWeight, sliced, sliced_matrix},
    },
    tensor::Dtype,
    weights::StreamedCheckpoint,
};

pub(in crate::family::qwen36) struct ContextRequest<'a> {
    pub artifact: &'a Path,
    pub spec: &'a FamilySpec,
    pub placement: MlxPlacement,
    pub activation: Activation,
    pub kv: KvDtype,
    pub host_budget: u64,
    pub reserve_override: u64,
    pub cache_limit: u64,
    pub max_context: u32,
    pub max_rows: u32,
    pub session_cap: Option<u64>,
    pub prefix_reuse: bool,
}

pub(in crate::family::qwen36) struct ContextPlan {
    pub context: u32,
    pub session_cap: u64,
}

struct Weights {
    base: u64,
    stages: Stages,
}

enum Stages {
    Dense(u64),
    Experts {
        full: u64,
        layers: Vec<ExpertSlices>,
    },
}

fn checked_total(terms: &[u64]) -> Result<u64, EngError> {
    terms.iter().try_fold(0u64, |total, &bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| EngError::MemoryPlan("Qwen MLX context charges overflow".into()))
    })
}

fn read_weight_plan(
    ckpt: &StreamedCheckpoint,
    config: &[u8],
    shape: &HostShape,
    request: &ContextRequest<'_>,
) -> Result<Weights, EngError> {
    let quant = QuantizationMap::from_json(config)?;
    let f32_scales = request.activation == Activation::F32;
    let mut base = Bytes::default();
    // Consts imports three F32 scalars and two vectors of key_dim.
    base.add(4 * (3 + 2 * shape.delta.key_dim as u64))?;
    let dense = request.placement == MlxPlacement::DenseFfnStaging;
    let mut stages: Vec<[Option<SlicedWeight>; 3]> = vec![[None, None, None]; shape.kinds.len()];
    for key in ckpt.names() {
        // The text loader ignores keys outside the text model it opens.
        let Ok((class, role)) = classify(key, request.spec) else {
            continue;
        };
        let payload = key.ends_with(".weight");
        let hidden = request.spec.hidden;
        match class {
            KeyClass::EmbedTokens => {}
            KeyClass::Router { .. } | KeyClass::SharedExpertGate { .. } => {
                // The resident importer replaces these packed inputs with F32 tensors.
                if payload {
                    let rows = if matches!(class, KeyClass::Router { .. }) {
                        request.spec.experts
                    } else {
                        1
                    };
                    base.add(4 * u64::from(hidden) * u64::from(rows))?;
                }
            }
            KeyClass::DenseMlp { layer, projection }
            | KeyClass::ExpertStack { layer, projection } => {
                let Some(stem) = key.strip_suffix(".weight") else {
                    continue;
                };
                let ModuleQuantization::Quantized(q) = quant.resolve(stem)? else {
                    return Err(EngError::Checkpoint(format!(
                        "module {stem} is not quantized"
                    )));
                };
                let inter = request
                    .spec
                    .feed_forward
                    .map_or(request.spec.moe_inter, |ff| ff.intermediate);
                let (rows, cols, slot) = match projection {
                    Projection::Gate => (inter, hidden, 0),
                    Projection::Up => (inter, hidden, 1),
                    Projection::Down => (hidden, inter, 2),
                };
                let weight = if dense {
                    sliced_matrix(ckpt, stem, q, rows, cols)?
                } else {
                    sliced(ckpt, stem, q, rows, cols)?
                };
                stages[layer as usize][slot] = Some(weight);
            }
            _ => {
                let (dtype, shape) = ckpt
                    .stored(key)
                    .ok_or_else(|| EngError::Checkpoint(format!("missing Qwen tensor {key}")))?;
                // Plain tensors load as F32. Norm conversion to BF16 comes after weight admission.
                let width = match role {
                    KeyRole::Dense => Dtype::F32.byte_width(),
                    KeyRole::Packed if !payload && f32_scales => Dtype::F32.byte_width(),
                    KeyRole::Packed => dtype.byte_width(),
                };
                let bytes = shape
                    .iter()
                    .try_fold(width, |total, &dim| total.checked_mul(u64::from(dim)))
                    .ok_or_else(|| {
                        EngError::MemoryPlan(format!("Qwen tensor {key} byte count overflows"))
                    })?;
                base.add(bytes)?;
            }
        }
    }

    let layers = stages
        .into_iter()
        .enumerate()
        .map(|(layer, projections)| match projections {
            [Some(gate), Some(up), Some(down)] => Ok([gate, up, down]),
            _ => Err(EngError::Checkpoint(format!(
                "Qwen layer {layer} has an incomplete staged weight set"
            ))),
        });
    let stages = if dense {
        let peak = layers.into_iter().try_fold(0u64, |peak, weights| {
            Ok::<_, EngError>(peak.max(DenseStage::new(ckpt, weights?, f32_scales)?.charge()?))
        })?;
        Stages::Dense(peak)
    } else {
        let layers = layers
            .map(|weights| Ok(ExpertSlices::new(weights?, f32_scales)))
            .collect::<Result<Vec<_>, EngError>>()?;
        let mut full = Bytes::default();
        for layer in &layers {
            full.add(layer.bytes(ckpt, request.spec.experts)?)?;
        }
        Stages::Experts {
            full: full.total(),
            layers,
        }
    };
    Ok(Weights {
        base: base.total(),
        stages,
    })
}

impl ContextRequest<'_> {
    fn session_terms(
        &self,
        shape: &HostShape,
        context: u32,
    ) -> Result<Option<(u64, u64)>, EngError> {
        use crate::family::qwen36::settings::{cache_growth_per_position, delta_state_bytes};
        let live_kv = cache_growth_per_position(self.spec, self.kv, self.activation)?
            .checked_mul(u64::from(context))
            .ok_or_else(|| EngError::MemoryPlan("Qwen MLX context cache bytes overflow".into()))?;
        let minimum_cap = if self.prefix_reuse {
            checked_total(&[live_kv, live_kv, delta_state_bytes(self.spec)?])?
        } else if self.spec.feed_forward.is_some() {
            checked_total(&[live_kv, delta_state_bytes(self.spec)?])?
        } else {
            live_kv
        };
        let cap = self
            .session_cap
            .unwrap_or(if self.spec.feed_forward.is_some() {
                minimum_cap
            } else {
                DEFAULT_SESSION_CAP
            });
        if minimum_cap > cap {
            return Ok(None);
        }
        let reserve = device_reserve(
            shape,
            self.activation,
            self.kv,
            context as usize,
            self.cache_limit,
            self.prefix_reuse.then_some(cap),
        )?
        .max(self.reserve_override);
        Ok(Some((cap, reserve)))
    }

    fn choose(
        &self,
        ckpt: &StreamedCheckpoint,
        shape: &HostShape,
        weights: &Weights,
        budget: u64,
    ) -> Result<ContextPlan, EngError> {
        let cannot_fit = || {
            EngError::MemoryPlan(format!(
                "Qwen MLX cannot fit {MIN_DERIVED_CONTEXT} context tokens and staged weights within {budget} bytes; set a smaller context_limit or increase the budget"
            ))
        };
        let (_, floor_reserve) = self
            .session_terms(shape, MIN_DERIVED_CONTEXT)?
            .ok_or_else(cannot_fit)?;
        let remaining = budget
            .checked_sub(checked_total(&[weights.base, floor_reserve])?)
            .ok_or_else(cannot_fit)?;
        // If the full MoE bank fits at the floor, retain it in the search. Every admitted
        // context will then make the same resident-first decision as the loader.
        let resident_experts = match &weights.stages {
            Stages::Experts { full, .. } if *full <= remaining => *full,
            _ => 0,
        };
        let stage_for = |rows: u32| -> Result<u64, EngError> {
            match &weights.stages {
                Stages::Dense(bytes) => Ok(*bytes),
                Stages::Experts { .. } if resident_experts != 0 => Ok(resident_experts),
                Stages::Experts { layers, .. } => {
                    let count = rows.saturating_mul(self.spec.top_k).min(self.spec.experts);
                    layers
                        .iter()
                        .try_fold(0u64, |peak, layer| Ok(peak.max(layer.bytes(ckpt, count)?)))
                }
            }
        };
        if self.max_context < MIN_DERIVED_CONTEXT || stage_for(1)? > remaining {
            return Err(cannot_fit());
        }
        let heads = u64::from(self.spec.heads_q);
        let floor_chunk = ChunkCost::new(heads, MIN_DERIVED_CONTEXT)?;
        let rows = admitted_chunk(self.max_rows, remaining, |rows| {
            checked_total(&[stage_for(rows)?, floor_chunk.workspace(rows)?])
        })?
        .unwrap_or(1);
        let placed = checked_total(&[weights.base, stage_for(rows)?])?;
        let fits = |context| -> Result<Option<u64>, EngError> {
            let Some((cap, reserve)) = self.session_terms(shape, context)? else {
                return Ok(None);
            };
            let workspace = if rows > 1 {
                ChunkCost::new(heads, context)?.workspace(rows)?
            } else {
                0
            };
            Ok((checked_total(&[placed, reserve, workspace])? <= budget).then_some(cap))
        };
        let context = if fits(self.max_context)?.is_some() {
            self.max_context
        } else {
            largest_fitting_context(self.max_context, &|ctx| {
                fits(ctx).is_ok_and(|cap| cap.is_some())
            })
            .filter(|&ctx| ctx >= MIN_DERIVED_CONTEXT)
            .ok_or_else(cannot_fit)?
        };
        Ok(ContextPlan {
            context,
            session_cap: fits(context)?.ok_or_else(cannot_fit)?,
        })
    }
}

pub(in crate::family::qwen36) fn derive(
    request: ContextRequest<'_>,
) -> Result<ContextPlan, EngError> {
    let device = DeviceBudget::detect()?.ok_or_else(|| {
        EngError::MemoryPlan(
            "Qwen staged context sizing requires a Metal working-set budget".into(),
        )
    })?;
    let config = std::fs::read(request.artifact.join("config.json"))
        .map_err(|error| EngError::Checkpoint(format!("config.json: {error}")))?;
    let shape = shape_of(&config, request.spec)?;
    let ckpt = super::executor::open_checkpoint(request.artifact)?;
    let weights = read_weight_plan(&ckpt, &config, &shape, &request)?;
    let budget = device.prefill(Some(request.host_budget));
    request.choose(&ckpt, &shape, &weights, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        family::qwen36::{
            host_load::HostWeights,
            resident::weights::{ExpertDevicePlan, ResidentWeights},
            tiny,
        },
        kernels::MlxKernels,
        weights::acquire_read,
    };

    fn request(spec: &FamilySpec, placement: MlxPlacement) -> ContextRequest<'_> {
        ContextRequest {
            artifact: Path::new("/unused"),
            spec,
            placement,
            activation: Activation::Bf16,
            kv: KvDtype::Q8,
            host_budget: 4 << 30,
            reserve_override: 0,
            cache_limit: 64 << 20,
            max_context: 16_384,
            max_rows: 128,
            session_cap: None,
            prefix_reuse: false,
        }
    }

    #[test]
    fn header_weights_match_the_loader_before_norm_conversion() {
        let _heat = crate::native_test_lock();
        for (spec, placement) in [
            (&tiny::TINY_DENSE, MlxPlacement::DenseFfnStaging),
            (&tiny::TINY_GQA, MlxPlacement::ExpertFallback),
        ] {
            for activation in [Activation::F32, Activation::Bf16] {
                let config = tiny::config(spec);
                let ckpt = tiny::checkpoint(spec);
                let shape = shape_of(&config, spec).unwrap();
                let mut request = request(spec, placement);
                request.activation = activation;
                acquire_read::reset();
                acquire_read::set(true);
                let plan = read_weight_plan(&ckpt, &config, &shape, &request);
                acquire_read::set(false);
                assert_eq!(acquire_read::snapshot(), (0, 0, 0));
                let plan = plan.unwrap();
                let dense = placement == MlxPlacement::DenseFfnStaging;
                let device_plan = match &plan.stages {
                    Stages::Dense(_) => None,
                    Stages::Experts { layers, full } => {
                        let one = layers
                            .iter()
                            .map(|layer| layer.bytes(&ckpt, spec.top_k).unwrap())
                            .max()
                            .unwrap();
                        assert!(one < *full);
                        Some(ExpertDevicePlan {
                            device: DeviceBudget::new(plan.base + one).unwrap(),
                            reserve: 0,
                            max_seqs: 1,
                        })
                    }
                };
                let host =
                    HostWeights::load_with_dense_staging(&ckpt, spec, &config, dense).unwrap();
                let mut kernels = MlxKernels::try_new().unwrap();
                let actual = ResidentWeights::import_with(
                    &mut kernels,
                    &ckpt,
                    host,
                    activation == Activation::F32,
                    device_plan,
                )
                .unwrap();
                assert_eq!(actual.byte_len().unwrap(), plan.base);
                if let Stages::Dense(bytes) = plan.stages {
                    assert_eq!(actual.dense_stage_bytes().unwrap(), bytes);
                } else {
                    assert!(actual.streams_experts());
                }
            }
        }
    }

    #[test]
    fn search_keeps_prefill_width_and_charges_a_retained_prefix() {
        let spec = &tiny::TINY_DENSE;
        let config = tiny::config(spec);
        let ckpt = tiny::checkpoint(spec);
        let shape = shape_of(&config, spec).unwrap();
        let mut request = request(spec, MlxPlacement::DenseFfnStaging);
        request.kv = KvDtype::F32;
        request.prefix_reuse = true;
        let weights = read_weight_plan(&ckpt, &config, &shape, &request).unwrap();
        let Stages::Dense(stage) = weights.stages else {
            panic!("dense stage")
        };
        let context = 8192;
        let (cap, reserve) = request.session_terms(&shape, context).unwrap().unwrap();
        let chunk = ChunkCost::new(u64::from(spec.heads_q), context).unwrap();
        let budget = weights.base + stage + reserve + chunk.workspace(request.max_rows).unwrap();
        let plan = request.choose(&ckpt, &shape, &weights, budget).unwrap();
        assert_eq!(plan.context, context);
        assert_eq!(plan.session_cap, cap);
        // A host budget too low for even one forward refuses at the supported floor.
        let (minimum_cap, minimum_reserve) = request
            .session_terms(&shape, MIN_DERIVED_CONTEXT)
            .unwrap()
            .unwrap();
        assert!(matches!(
            request.choose(
                &ckpt,
                &shape,
                &weights,
                weights.base + stage + minimum_reserve - 1
            ),
            Err(EngError::MemoryPlan(_))
        ));
        request.session_cap = Some(minimum_cap - 1);
        assert!(request.choose(&ckpt, &shape, &weights, budget).is_err());
    }
}
