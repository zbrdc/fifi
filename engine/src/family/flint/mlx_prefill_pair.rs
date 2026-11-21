//! Two-piece disk prefill with shared expert staging.

use super::{super::weights::FlintCheckpoint, *};
use crate::family::prefill::mlx::{local_expert_ids, sorted_expert_ids, submit_shared_expert};
use crate::kernels::HostPages;
use std::{
    ops::{ControlFlow, Range},
    sync::atomic::{AtomicBool, Ordering},
};

const EXPERTS: usize = 256;
const TOP_K: u32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family::flint) enum PairStaging {
    Shared,
    Waves64,
}

impl PairStaging {
    pub(in crate::family::flint) const fn slots(self) -> usize {
        match self {
            Self::Shared => EXPERTS,
            Self::Waves64 => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family::flint) struct PairPlan {
    pub(in crate::family::flint) split: usize,
    pub(in crate::family::flint) staging: PairStaging,
}

#[derive(Default)]
pub(super) struct StagePeak {
    pub(super) mapped_bytes: usize,
    pub(super) experts: usize,
}

impl Model {
    pub(in crate::family::flint) fn prefill_pair(
        &mut self,
        state: &mut State,
        tokens: &[u32],
        plan: PairPlan,
        cancel: Option<&AtomicBool>,
    ) -> Result<ControlFlow<()>> {
        let Some((first, second)) = tokens.split_at_checked(plan.split) else {
            return Err(EngError::BadRequest(
                "Flint prefill pair split is outside the input".into(),
            ));
        };
        if state.pos != 0
            || !state.valid
            || !(2..=1024).contains(&first.len())
            || !(2..=1024).contains(&second.len())
            || tokens.iter().any(|&id| id >= 150_144)
        {
            return Err(EngError::BadRequest(
                "Flint prefill pair has invalid state or rows".into(),
            ));
        }
        let rows = [first.len() as u32, second.len() as u32];
        ChunkCachePlan::new(0, rows[0], rows[1], context_limit())?;
        #[cfg(feature = "native-fault-qualification")]
        let timing = super::super::mlx_expert_timing::ForwardTiming::begin(&self.kernels, 0);
        // Once writes begin, an execution failure requires the normal session reset.
        state.valid = false;
        let (_stage_peak, forward) =
            self.forward_pair(state, [first, second], rows, plan.staging, cancel)?;
        if forward.is_continue() {
            state.pos = rows.iter().sum();
            state.valid = true;
        }
        #[cfg(feature = "native-fault-qualification")]
        if let Some(timing) = timing {
            timing.finish_group(
                &self.kernels,
                rows.iter().sum(),
                state.full_capacity,
                Some(super::super::mlx_expert_timing::PairTiming {
                    piece_rows: rows,
                    stage_slots: plan.staging.slots(),
                    peak_experts: _stage_peak.experts,
                    peak_mapped_bytes: _stage_peak.mapped_bytes,
                    completed_layers: match forward {
                        ControlFlow::Break(layers) => layers,
                        ControlFlow::Continue(()) => self.layers.len(),
                    },
                    cancelled: forward.is_break(),
                }),
            );
        }
        Ok(forward.map_break(|_| ()))
    }

    fn forward_pair(
        &mut self,
        state: &mut State,
        tokens: [&[u32]; 2],
        rows: [u32; 2],
        strategy: PairStaging,
        cancel: Option<&AtomicBool>,
    ) -> Result<(StagePeak, ControlFlow<usize>)> {
        let k = &mut self.kernels;
        state.ensure_full_capacity(k, rows.iter().sum())?;
        let [first, second] = tokens;
        let mut first = k.import_tensor(&super::super::precision::embedding(
            &self.source.embedding_rows(first)?,
        )?)?;
        let mut second = k.import_tensor(&super::super::precision::embedding(
            &self.source.embedding_rows(second)?,
        )?)?;
        let mut first_health = Vec::new();
        let mut second_health = Vec::new();
        let mut stage_peak = StagePeak::default();
        for (index, (layer, plane)) in self.layers.iter().zip(&mut state.planes).enumerate() {
            // The prior routed stage has evaluated both outputs and retired its pages.
            if cancel.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
                return Ok((stage_peak, ControlFlow::Break(index)));
            }
            let plan = ChunkCachePlan::new(index, 0, rows[0], state.full_capacity)?;
            let first_input = FfnInput::prepare(k, layer, &first, plane, &plan)?;
            let plan = ChunkCachePlan::new(index, rows[0], rows[1], state.full_capacity)?;
            let second_input = FfnInput::prepare(k, layer, &second, plane, &plan)?;
            if let Ffn::Moe {
                router,
                routed: Routed::Disk { source, stem, .. },
                shared,
            } = &layer.ffn
            {
                let first_routed = RoutedInput::prepare(k, first_input, router)?;
                let second_routed = RoutedInput::prepare(k, second_input, router)?;
                let mut loader = StageLoader {
                    source,
                    stem,
                    coalesce: expert_read_runs().coalesce(rows[0] + rows[1]),
                    peak: &mut stage_peak,
                };
                [first, second] = shared_stage(
                    k,
                    &mut loader,
                    shared,
                    [&first_routed, &second_routed],
                    strategy,
                )?;
                first_health.push(first_routed.route.health);
                second_health.push(second_routed.route.health);
            } else {
                first = first_input.finish(k, &layer.ffn, &mut first_health)?;
                second = second_input.finish(k, &layer.ffn, &mut second_health)?;
            }
        }
        k.evaluate(&[&first, &second])?;
        mlx_route::read_step(k, None, &first_health)?;
        mlx_route::read_step(k, None, &second_health)?;
        let forward = if cancel.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            ControlFlow::Break(self.layers.len())
        } else {
            ControlFlow::Continue(())
        };
        Ok((stage_peak, forward))
    }
}

struct FfnInput {
    normalized: ExecutionTensor,
    residual: ExecutionTensor,
}

impl FfnInput {
    fn prepare(
        k: &mut MlxKernels,
        layer: &Layer,
        x: &ExecutionTensor,
        plane: &mut Option<Plane>,
        plan: &ChunkCachePlan,
    ) -> Result<Self> {
        let normalized = k.rms_norm(x, &layer.input_norm, 1e-6)?;
        let attended = attention(k, layer, &normalized, plane, plan)?;
        let residual = k.binary(BinaryOp::Add, x, &attended)?;
        let normalized = k.rms_norm(&residual, &layer.post_norm, 1e-6)?;
        Ok(Self {
            normalized,
            residual,
        })
    }

    fn finish(
        self,
        k: &mut MlxKernels,
        weights: &Ffn,
        health: &mut Vec<ExecutionTensor>,
    ) -> Result<ExecutionTensor> {
        let (output, route_health) = ffn(k, weights, &self.normalized)?;
        health.extend(route_health);
        k.binary(BinaryOp::Add, &self.residual, &output)
    }
}

struct RoutedInput {
    ffn: FfnInput,
    route: mlx_route::Route,
    expert_ids: Vec<u32>,
}

impl RoutedInput {
    fn prepare(k: &mut MlxKernels, ffn: FfnInput, router: &ExecutionTensor) -> Result<Self> {
        let logits = k.mm(&ffn.normalized, router)?;
        let route = mlx_route::select(k, &logits)?;
        let expert_ids = host_expert_ids(&k.readback(route.ids.ids())?)?;
        Ok(Self {
            ffn,
            route,
            expert_ids,
        })
    }

    fn finish(
        &self,
        k: &mut MlxKernels,
        routed: &ExecutionTensor,
        shared: &ExecutionTensor,
    ) -> Result<ExecutionTensor> {
        let output = precision_native::combine(k, routed, &self.route.weights, shared)?;
        let output = k.binary(BinaryOp::Add, &self.ffn.residual, &output)?;
        k.evaluate(&[&output])?;
        Ok(output)
    }
}

fn shared_stage(
    k: &mut MlxKernels,
    loader: &mut StageLoader<'_>,
    shared: &SwiGlu,
    [first, second]: [&RoutedInput; 2],
    strategy: PairStaging,
) -> Result<[ExecutionTensor; 2]> {
    let selected = [first.expert_ids.as_slice(), second.expert_ids.as_slice()];
    let joined = selected.concat();
    let experts = sorted_expert_ids(&joined);
    let split = first.expert_ids.len() as u32;
    let inputs = [&first.ffn.normalized, &second.ffn.normalized];
    let first_shared = shared.apply(k, inputs[0])?;
    let second_shared = shared.apply(k, inputs[1])?;
    let first_shared = submit_shared_expert(k, first_shared)?;
    let second_shared = submit_shared_expert(k, second_shared)?;
    let output = match strategy {
        PairStaging::Shared => {
            let stage = loader.load(k, &experts, &joined)?;
            let first_ids = stage.ids(k, 0..split)?;
            let routed = grouped_experts(k, &stage.weights, inputs[0], &first_ids)?;
            let first = first.finish(k, &routed, &first_shared)?;
            let second_ids = stage.ids(k, split..joined.len() as u32)?;
            let second_routed = grouped_experts(k, &stage.weights, inputs[1], &second_ids)?;
            let second = second.finish(k, &second_routed, &second_shared)?;
            drop((first_ids, second_ids, routed, second_routed));
            stage.retire(k, &second)?;
            [first, second]
        }
        PairStaging::Waves64 => {
            let [first_routed, second_routed] =
                wave_experts(k, loader, inputs, selected, &experts)?;
            [
                first.finish(k, &first_routed, &first_shared)?,
                second.finish(k, &second_routed, &second_shared)?,
            ]
        }
    };
    #[cfg(feature = "native-fault-qualification")]
    trace_expert_accesses(
        loader.stem,
        inputs[0].shape()[0] + inputs[1].shape()[0],
        &experts,
    );
    Ok(output)
}

pub(super) struct StageLoader<'a> {
    pub(super) source: &'a FlintCheckpoint,
    pub(super) stem: &'a str,
    pub(super) coalesce: bool,
    pub(super) peak: &'a mut StagePeak,
}

impl StageLoader<'_> {
    fn load(
        &mut self,
        k: &mut MlxKernels,
        experts: &[u32],
        selected: &[u32],
    ) -> Result<ExpertStage> {
        let ids = id_tensor(&local_expert_ids(selected, experts)?)?;
        let (weights, local, pages) =
            SwiGlu::load_expert_pages(k, self.source, self.stem, experts, &ids, self.coalesce)?;
        self.peak.mapped_bytes = self.peak.mapped_bytes.max(pages.len());
        self.peak.experts = self.peak.experts.max(experts.len());
        Ok(ExpertStage {
            weights,
            local,
            pages,
            slots: experts.len() as u32,
        })
    }
}

struct ExpertStage {
    weights: SwiGlu,
    local: ExecutionTensor,
    pages: HostPages,
    slots: u32,
}

impl ExpertStage {
    fn ids(&self, k: &mut MlxKernels, rows: Range<u32>) -> Result<BoundedIndices> {
        let ids = k.slice(&self.local, &[rows.start], &[rows.end])?;
        k.bound_indices(&ids, self.slots)
    }

    fn retire(self, k: &mut MlxKernels, anchor: &ExecutionTensor) -> Result<()> {
        let Self {
            weights,
            local,
            mut pages,
            ..
        } = self;
        drop((weights, local));
        pages.release();
        k.evaluate(&[anchor])?;
        if pages.in_use() {
            return Err(EngError::Backend {
                operation: "flint prefill pair",
                message: "shared expert pages did not retire".into(),
            });
        }
        Ok(())
    }
}

fn id_tensor(ids: &[u32]) -> Result<Tensor> {
    Tensor::try_from_host(
        Dtype::U32,
        vec![ids.len() as u32],
        ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
    )
}

struct WavePiece<'a> {
    input: &'a ExecutionTensor,
    positions_by_expert: Vec<Vec<usize>>,
    outputs: Vec<ExecutionTensor>,
    emitted: Vec<usize>,
}

impl<'a> WavePiece<'a> {
    fn new(input: &'a ExecutionTensor, selected: &[u32]) -> Result<Self> {
        // MLX argsort is stable. Keep the original order within each expert.
        let mut positions_by_expert = vec![Vec::new(); EXPERTS];
        for (pair, &expert) in selected.iter().enumerate() {
            let positions = positions_by_expert
                .get_mut(expert as usize)
                .ok_or_else(|| {
                    EngError::BadRequest("Flint wave route is outside the expert bank".into())
                })?;
            positions.push(pair);
        }
        Ok(Self {
            input,
            positions_by_expert,
            outputs: Vec::new(),
            emitted: Vec::new(),
        })
    }

    fn positions(&self, wave: &[u32]) -> Vec<usize> {
        wave.iter()
            .flat_map(|&expert| self.positions_by_expert[expert as usize].iter().copied())
            .collect()
    }

    fn project(
        &mut self,
        k: &mut MlxKernels,
        stage: &ExpertStage,
        start: u32,
        positions: Vec<usize>,
    ) -> Result<()> {
        if positions.is_empty() {
            return Ok(());
        }
        let count = positions.len() as u32;
        let ids = stage.ids(k, start..start + count)?;
        let token_rows = positions
            .iter()
            .map(|&pair| pair as u32 / TOP_K)
            .collect::<Vec<_>>();
        let token_rows = k.import_tensor(&id_tensor(&token_rows)?)?;
        let sorted = k.take(self.input, &token_rows, 0)?;
        let sorted = k.reshape(&sorted, &[count, 1, self.input.shape()[1]])?;
        let output = sorted_expert_pairs(k, &stage.weights, &sorted, &ids)?;
        k.evaluate(&[&output])?;
        self.outputs.push(output);
        self.emitted.extend(positions);
        Ok(())
    }

    fn restore(self, k: &mut MlxKernels) -> Result<ExecutionTensor> {
        let rows = self.input.shape()[0];
        let hidden = self.input.shape()[1];
        let pairs = (rows * TOP_K) as usize;
        if self.emitted.len() != pairs {
            return Err(EngError::BadRequest(
                "Flint waves missed routed pairs".into(),
            ));
        }
        let mut inverse = vec![u32::MAX; pairs];
        for (encoded, original) in self.emitted.into_iter().enumerate() {
            let entry = inverse
                .get_mut(original)
                .filter(|entry| **entry == u32::MAX)
                .ok_or_else(|| {
                    EngError::BadRequest("Flint waves repeated or exceeded a routed pair".into())
                })?;
            *entry = encoded as u32;
        }
        let outputs = self.outputs.iter().collect::<Vec<_>>();
        let joined = k.concatenate(&outputs, 0)?;
        let inverse = k.import_tensor(&id_tensor(&inverse)?)?;
        let ordered = k.take(&joined, &inverse, 0)?;
        k.reshape(&ordered, &[rows, TOP_K, hidden])
    }
}

pub(super) fn wave_experts(
    k: &mut MlxKernels,
    loader: &mut StageLoader<'_>,
    inputs: [&ExecutionTensor; 2],
    selected: [&[u32]; 2],
    union: &[u32],
) -> Result<[ExecutionTensor; 2]> {
    let mut first = WavePiece::new(inputs[0], selected[0])?;
    let mut second = WavePiece::new(inputs[1], selected[1])?;
    for wave in union.chunks(PairStaging::Waves64.slots()) {
        let first_positions = first.positions(wave);
        let second_positions = second.positions(wave);
        let ids = first_positions
            .iter()
            .map(|&pair| selected[0][pair])
            .chain(second_positions.iter().map(|&pair| selected[1][pair]))
            .collect::<Vec<_>>();
        let stage = loader.load(k, wave, &ids)?;
        let split = first_positions.len() as u32;
        first.project(k, &stage, 0, first_positions)?;
        second.project(k, &stage, split, second_positions)?;
        stage.retire(k, inputs[0])?;
    }
    Ok([first.restore(k)?, second.restore(k)?])
}
