//! Visit-major text execution. Each visit advances its own causal state in token
//! order, then computes the routed expert outputs using the shared CUDA banks.

use super::{
    Attention, AttentionInput, CpuForward, LayerVisit, MessageBank, MoeRoute, PROMPT_ROW_SPLIT,
    Role, Stream, TapRow, VisitState, Weights, add, apply_control, attention_norm_name,
    attention_residual, bad, control_names,
    controller_batch::{ChainedAttention, controller_batch_enabled, multirow_controllers_enabled},
    rms_norm,
    window::PendingRow,
};
use crate::family::inkling2::switches::Switch;
use crate::kernels::cuda::DeviceRange;
use crate::{
    EngError,
    family::inkling2::{
        cpu_attention::{AttentionUndo, VisitUndo},
        cuda_projection::gguf::{BatchController, ExpertPrefillRows, streaming::PrefillRoute},
        gguf_weights::GgufTextWeights,
    },
};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod resident_first;
mod resident_mlp;

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod weight_visit_tests;

struct PromptRow<'a> {
    stream: Stream,
    prelude: Option<Stream>,
    bank: Option<MessageBank>,
    taps: TapRow<'a>,
    journal: Option<VerifyRow>,
}

struct VerifyRow {
    attention: Option<AttentionUndo>,
    undo: Vec<VisitUndo>,
}

impl PromptRow<'_> {
    fn begin_pass(&mut self, pass: usize) -> Result<(), EngError> {
        if pass == 0 {
            self.prelude = Some(self.stream.clone());
        } else {
            let prelude = self
                .prelude
                .as_ref()
                .ok_or_else(|| bad("Inkling prefill lost its core prelude"))?;
            self.bank = Some(MessageBank {
                first_block: prelude.block_count() + 1,
                stream: std::mem::replace(&mut self.stream, prelude.clone()),
            });
        }
        Ok(())
    }
}

struct MlpRow {
    route: MoeRoute,
    input: Vec<f32>,
}

impl CpuForward {
    pub(in crate::family::inkling2) fn supports_grouped_prefill(&self) -> bool {
        if self.draft_window_enabled() {
            self.grouped_draft_prefill
        } else {
            matches!(&self.weights, Weights::Gguf(weights) if weights.supports_grouped_prefill())
        }
    }

    pub(in crate::family::inkling2) fn enable_grouped_draft_prefill(&mut self) {
        self.grouped_draft_prefill = self.supports_exact_draft_grouping();
    }

    pub(in crate::family::inkling2) fn enable_grouped_verify(&mut self) -> bool {
        self.grouped_verify = self.supports_exact_draft_grouping();
        self.grouped_verify
    }

    fn supports_exact_draft_grouping(&self) -> bool {
        self.draft_window_enabled()
            && matches!(&self.weights, Weights::Gguf(weights)
                if weights.supports_grouped_verify(self.spec.stored_layers()))
    }

    pub(in crate::family::inkling2) fn grouped_verify_workspace_bytes(
        &self,
    ) -> Result<u64, EngError> {
        self.prefill_workspace_bytes_per_row()?
            .checked_mul(self.window_limit.min(self.context) as u64)
            .ok_or_else(|| {
                EngError::MemoryPlan("Inkling grouped verify workspace overflows".into())
            })
    }

    pub(in crate::family::inkling2) fn prefill_workspace_bytes_per_row(
        &self,
    ) -> Result<u64, EngError> {
        let hidden = u128::from(self.spec.hidden_size);
        let middle = u128::from(self.spec.intermediate_size);
        let experts = u128::from(self.spec.n_routed_experts);
        let top_k = u128::from(self.spec.num_experts_per_tok);
        let blocks = self
            .visits
            .iter()
            .filter(|visit| {
                visit
                    .weight_layer
                    .is_multiple_of(self.spec.n_layers_per_attn_res_block)
            })
            .count() as u128;

        // Current stream, prelude, prior pass bank, and a transient stream clone,
        // each with its cached normalized current row, even though prelude clones
        // and message banks share immutable normalized rows.
        let stream_values = (4 * (blocks + 1) + 4) * hidden;
        // Routed contributions, MLP input, accumulated output, and staging
        // copies of the input, output, and gate/up and down intermediates.
        let mut projection_values = (top_k + 6) * hidden + 6 * middle;
        let mut projection_metadata = 0;
        if prefill_projection_batch_enabled() {
            let query = u128::from(
                self.spec
                    .num_attention_heads
                    .max(self.spec.swa_num_attention_heads),
            );
            let kv = u128::from(
                self.spec
                    .num_key_value_heads
                    .max(self.spec.swa_num_key_value_heads),
            );
            let width = u128::from(self.spec.head_dim.max(self.spec.swa_head_dim));
            projection_values += 3 * hidden
                + 2 * (query + kv) * width
                + query * (self.math.d_rel as u128 + 1)
                + 2 * kv * self.math.sconv_kernel_size as u128;
            projection_metadata = (size_of::<[Vec<f32>; 3]>()
                + size_of::<super::super::cpu_attention::AttentionProjection>()
                + size_of::<[&[f32]; 3]>()) as u128;
            if controller_batch_enabled() {
                // Three gates, messages, gain/forget outputs and temporary transfer copies.
                projection_values += 33 * hidden;
                projection_metadata += (15 * size_of::<Vec<f32>>()
                    + 3 * size_of::<[&[f32]; 3]>()
                    + 18 * size_of::<crate::kernels::cuda::resident::dense_gemv::DenseGemvCall<'_>>(
                    )) as u128;
            }
        }
        if multirow_controllers_enabled() {
            // MLP residuals/messages/gates/projections plus the tile's host transfer copies.
            projection_values += 12 * hidden;
            projection_metadata += (8 * size_of::<Vec<f32>>()) as u128;
        }
        if resident_mlp::enabled() && !self.draft_window_enabled() {
            // Pinned input/message slab, shared gamma, and the mixed messages.
            projection_values += 4 * hidden + 3;
            projection_metadata += size_of::<Vec<f32>>() as u128;
        }
        if crate::family::inkling2::cuda_projection::gguf::attention::prefill::enabled() {
            // Finished attention rows coexist until the residual update phase.
            projection_values += hidden;
            projection_metadata += size_of::<Vec<f32>>() as u128;
        }
        // Batched router logits coexist with the selected rows' scores.
        let numeric_bytes =
            (stream_values + projection_values + 2 * experts) * size_of::<f32>() as u128;
        // Include row handles and reserve room for geometrically grown lane lists.
        let row_metadata = (size_of::<PromptRow<'_>>()
            + size_of::<MlpRow>()
            + size_of::<PrefillRoute<'_>>()
            + size_of::<Vec<f32>>()) as u128;
        let stream_metadata = 8 * blocks * size_of::<Vec<f32>>() as u128;
        let lanes = size_of::<(usize, usize)>();
        let resident_slots = size_of::<(crate::expert_tier::ExpertId, (usize, usize))>();
        let route_metadata = top_k
            * (2 * (lanes + resident_slots) + size_of::<usize>() + size_of::<Vec<f32>>()) as u128;
        u64::try_from(
            numeric_bytes + row_metadata + stream_metadata + route_metadata + projection_metadata,
        )
        .map_err(|_| EngError::MemoryPlan("Inkling prefill workspace size overflows".into()))
    }

    pub(in crate::family::inkling2) fn prefill_grouped(
        &mut self,
        tokens: &[u32],
    ) -> Result<(), EngError> {
        if self.draft_window_enabled() && !self.grouped_draft_prefill {
            return Err(bad("Inkling speculation requires token-major prefill"));
        }
        if self.failed {
            return Err(bad(
                "a previous Inkling forward invalidated the sequence state",
            ));
        }
        // Prefill reads host rows; a chain token left them on the device.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        self.materialize_resident()?;
        if tokens.is_empty()
            || tokens.len() > self.context.saturating_sub(self.position)
            || tokens
                .iter()
                .any(|&id| u64::from(id) >= self.spec.unpadded_vocab_size)
        {
            return Err(bad(
                "prefill rows or tokens are outside the Inkling text limits",
            ));
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if self.resident_pass(tokens)? {
            return Ok(());
        }
        if tokens.len() < 2 {
            return Err(bad(
                "prefill rows or tokens are outside the Inkling text limits",
            ));
        }
        self.window = None;
        self.failed = true;
        let mut state = std::mem::take(&mut self.state);
        let rows = self.run_grouped_rows(&mut state, tokens, false)?;
        if !self.taps.is_empty() {
            // Complete every terminal tap before publishing the committed piece.
            let features = rows
                .into_iter()
                .map(|row| {
                    self.finish_row(&row.stream, row.taps, false)
                        .map(|row| row.features)
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.features.extend(features.into_iter().flatten());
        }
        self.state = state;
        self.position += tokens.len();
        self.failed = false;
        Ok(())
    }

    pub(super) fn grouped_verify_rows(
        &self,
        state: &mut [VisitState],
        tokens: &[u32],
    ) -> Result<Vec<PendingRow>, EngError> {
        self.run_grouped_rows(state, tokens, true)?
            .into_iter()
            .map(|row| {
                let journal = row.journal.expect("grouped verify row has a journal");
                let forward = self.finish_row(&row.stream, row.taps, true)?;
                PendingRow::new(journal.undo, forward)
            })
            .collect()
    }

    fn run_grouped_rows(
        &self,
        state: &mut [VisitState],
        tokens: &[u32],
        verify: bool,
    ) -> Result<Vec<PromptRow<'_>>, EngError> {
        let Weights::Gguf(weights) = &self.weights else {
            return Err(bad("grouped execution needs CUDA GGUF weights"));
        };
        let mut rows = tokens
            .iter()
            .map(|&id| {
                Ok(PromptRow {
                    stream: self.embed(id)?,
                    prelude: None,
                    bank: None,
                    taps: TapRow::new(&self.taps, self.hidden()),
                    journal: verify.then(|| VerifyRow {
                        attention: None,
                        undo: Vec::with_capacity(self.visits.len()),
                    }),
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let core = &self.spec.core;
        let width = core.core_end - core.core_start;
        let end = core.core_start + width * core.num_passes;
        let mut resident_mlp = (!verify && !self.draft_window_enabled() && resident_mlp::enabled())
            .then(resident_mlp::Workspace::default);
        for &visit in &self.visits {
            let slot = visit.state_slot;
            if (core.core_start..end).contains(&slot)
                && (slot - core.core_start).is_multiple_of(width)
            {
                for row in &mut rows {
                    row.begin_pass((slot - core.core_start) / width)?;
                }
            } else if slot == end {
                for row in &mut rows {
                    row.bank = None;
                    row.prelude = None;
                }
            }
            self.prompt_visit(
                visit,
                &mut state[slot],
                &mut rows,
                weights,
                (verify, resident_mlp.as_mut()),
            )?;
        }
        Ok(rows)
    }

    fn prompt_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        rows: &mut [PromptRow<'_>],
        weights: &GgufTextWeights,
        (verify, resident_mlp): (bool, Option<&mut resident_mlp::Workspace>),
    ) -> Result<(), EngError> {
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let prefix = format!("{layer}.mlp");
        let mut projected = None;
        let router = format!("{prefix}.gate.weight");
        let batch_router = weights.batch_prefill_router(physical as u32, &router);
        let banked = rows.first().is_some_and(|row| row.bank.is_some());
        // Choose a device-only consumer before attention can advance causal state.
        let resident = if batch_router
            && banked
            && let Some(scratch) = resident_mlp
        {
            scratch.admit(self, weights, (physical, &layer, &prefix), rows.len())?
        } else {
            None
        };
        let batched_attention = batch_router
            && !self.draft_window_enabled()
            && rows.len() > 1
            && !verify
            && crate::family::inkling2::cuda_projection::gguf::attention::prefill::enabled()
            && weights.resident_attention().is_some();
        let chained = if batched_attention {
            self.grouped_chained_attention(
                visit,
                &layer,
                rows.iter().map(|row| (&row.stream, row.bank.as_ref())),
                state,
            )?
        } else {
            None
        };
        if let Some(ChainedAttention::Host(values)) = chained.as_ref() {
            self.diagnose_chained_projections(visit, &layer, rows, weights, values)?;
        }
        let mut attention_inputs = if chained.is_some() {
            None
        } else {
            self.prompt_attention_inputs(visit, &layer, rows)?
        };
        // Without the resident batched router, preserve the interleave between
        // per-row prefetch and route observation on cold expert layers.
        let controller =
            if batch_router && banked && resident.is_none() && multirow_controllers_enabled() {
                let names = control_names(&layer, Role::Mlp);
                weights.batch_controller(
                    names.each_ref().map(String::as_str),
                    [self.spec.core.gate_rank as usize, self.hidden()],
                    rows.len(),
                    !self.draft_window_enabled(),
                )?
            } else {
                None
            };
        let control_batched = controller.is_some() || resident.is_some();
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if batched_attention
            && let Some((inputs, _)) = &attention_inputs
            && self.try_resident_first_visit(
                visit,
                state,
                rows,
                weights,
                (&layer, &prefix),
                inputs,
            )?
        {
            return Ok(());
        }
        let batched = if let Some(chained) = chained {
            let outputs = match chained {
                ChainedAttention::Host(p) => {
                    self.attention_prefill_batch(&layer, visit, state, p)?
                }
                ChainedAttention::Resident(outputs) => outputs,
            };
            Some(self.prompt_attention_residuals(visit, &layer, rows, outputs, control_batched)?)
        } else if batched_attention {
            if let Some((inputs, batched_controllers)) = attention_inputs.take() {
                let outputs =
                    match self.attention_prefill_bridge(&layer, visit, state, &inputs, None)? {
                        Some(outputs) => Some(outputs),
                        None => self
                            .project_prompt_attention(
                                weights,
                                &layer,
                                visit,
                                &inputs,
                                batched_controllers,
                                true,
                            )?
                            .map(|p| self.attention_prefill_batch(&layer, visit, state, p))
                            .transpose()?,
                    };
                outputs
                    .map(|outputs| {
                        self.prompt_attention_residuals(
                            visit,
                            &layer,
                            rows,
                            outputs,
                            control_batched,
                        )
                    })
                    .transpose()?
            } else {
                None
            }
        } else {
            None
        };
        if let Some((inputs, batched_controllers)) = attention_inputs {
            projected = self.project_prompt_attention(
                weights,
                &layer,
                visit,
                &inputs,
                batched_controllers,
                !verify,
            )?;
        }
        // Verification journals keep the per-row MLP history.
        let mlp_conv =
            if rows.iter().all(|row| row.journal.is_none()) && !self.draft_window_enabled() {
                self.admit_prompt_mlp_conv(state, &layer, rows.len())?
            } else {
                None
            };
        let destination = mlp_conv
            .as_ref()
            .map(|conv| conv.expert_destination())
            .transpose()?;
        let mut projected = projected.into_iter().flatten();
        let inputs = rows.iter_mut().map(|row| {
            row.taps.capture(visit.state_slot, row.stream.current());
            if let Some(journal) = &mut row.journal {
                journal.attention =
                    Some(state.checkpoint_attention(self.kv_capacity(visit.attention)));
            }
            let input = self.mlp_residual(
                visit,
                &layer,
                state,
                &mut row.stream,
                row.bank.as_ref(),
                AttentionInput::Grouped(projected.next()),
            )?;
            if control_batched {
                Ok(input)
            } else {
                self.finish_mlp_input(input, row.bank.as_ref(), physical, &layer)
            }
        });
        let (outputs, written) = if batch_router {
            let mut inputs = match batched {
                Some(inputs) => inputs,
                None => inputs.collect::<Result<Vec<_>, _>>()?,
            };
            if let Some(resident) = resident {
                (
                    resident.execute(self, rows, &inputs, destination)?,
                    destination.is_some(),
                )
            } else {
                if let Some(controller) = controller {
                    self.prompt_mlp_control(visit, &layer, rows, &mut inputs, controller)?;
                }
                let (mlp, retained) = self.prompt_routes(weights, physical, &prefix, inputs)?;
                self.prompt_experts(weights, physical, &prefix, &mlp, (retained, destination))?
            }
        } else {
            let mlp = inputs
                .map(|input| {
                    let input = input?;
                    let route = self.moe_route(physical, &prefix, &input)?;
                    Ok(MlpRow { route, input })
                })
                .collect::<Result<Vec<_>, EngError>>()?;
            self.prompt_experts(weights, physical, &prefix, &mlp, (None, destination))?
        };
        if let Some(conv) = mlp_conv.filter(|_| written) {
            for (row, output) in rows
                .iter_mut()
                .zip(self.finish_prompt_mlp_conv(state, conv)?)
            {
                row.stream.add_current(&output);
            }
            static USED: std::sync::Once = std::sync::Once::new();
            USED.call_once(|| eprintln!("Inkling CUDA used prompt expert output convolution"));
            return Ok(());
        }
        for (row, output) in rows.iter_mut().zip(outputs) {
            if let Some(journal) = &mut row.journal {
                let attention = journal
                    .attention
                    .take()
                    .expect("grouped attention already ran");
                journal.undo.push(attention.with_mlp(state));
            }
            self.finish_layer(&layer, state, &mut row.stream, output)?;
        }
        Ok(())
    }

    fn prompt_attention_residuals(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: &mut [PromptRow<'_>],
        outputs: Vec<Vec<f32>>,
        control_batched: bool,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let physical = visit.weight_layer;
        let query = self.weights.vec(
            &format!("model.llm.attn_res.mlp_queries.{physical}"),
            self.hidden(),
        )?;
        for row in rows.iter_mut() {
            row.taps.capture(visit.state_slot, row.stream.current());
        }
        if outputs.len() != rows.len() {
            return Err(bad("Inkling prompt attention returned the wrong row count"));
        }
        let seal = physical.is_multiple_of(self.spec.n_layers_per_attn_res_block);
        let mut residuals: Vec<_> = rows
            .iter_mut()
            .map(|row| &mut row.stream)
            .zip(outputs)
            .collect();
        let inputs = crate::family::host_rows::map_rows(
            &mut residuals,
            PROMPT_ROW_SPLIT,
            |(stream, attn)| attention_residual(stream, seal, &query, std::mem::take(attn)),
        );
        if control_batched {
            return Ok(inputs);
        }
        rows.iter()
            .zip(inputs)
            .map(|(row, input)| self.finish_mlp_input(input, row.bank.as_ref(), physical, layer))
            .collect()
    }

    fn prompt_mlp_control(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: &[PromptRow<'_>],
        inputs: &mut [Vec<f32>],
        controller: BatchController<'_>,
    ) -> Result<(), EngError> {
        let _span = self.phase_span(super::Work::Controller);
        let controls = {
            let _subspan = self.phase_span(super::Work::MlpControlPrepare);
            inputs
                .iter()
                .zip(rows)
                .map(|(input, row)| {
                    let bank = row
                        .bank
                        .as_ref()
                        .ok_or_else(|| bad("Inkling grouped MLP lost a message bank"))?;
                    self.control_input(input, bank, visit.weight_layer, Role::Mlp)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let gates: Vec<_> = controls
            .iter()
            .map(|control| control.gate.as_slice())
            .collect();
        let projections = {
            let _subspan = self.phase_span(super::Work::MlpControlProject);
            controller.project(&gates, "mlp")?
        };
        let _subspan = self.phase_span(super::Work::MlpControlApply);
        let norm = format!("{layer}.mlp_norm.weight");
        for ((input, control), projection) in inputs.iter_mut().zip(controls).zip(projections) {
            apply_control(input, control.message, projection);
            self.norm(input, &norm)?;
        }
        Ok(())
    }

    fn prompt_routes<'a>(
        &self,
        weights: &'a GgufTextWeights,
        layer: usize,
        prefix: &str,
        inputs: Vec<Vec<f32>>,
    ) -> Result<(Vec<MlpRow>, Option<DeviceRange<'a>>), EngError> {
        let _span = self.phase_span(super::Work::Router);
        let input_rows: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
        let (logits, retained) = weights.prefill_router_logits(
            &format!("{prefix}.gate.weight"),
            &input_rows,
            layer as u32,
            !self.draft_window_enabled(),
        )?;
        let experts = self.spec.n_routed_experts as usize;
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        let routes = self.route_prefill_logits(layer, &logits, &bias, inputs.len())?;
        let rows = inputs
            .into_iter()
            .zip(routes)
            .map(|(input, route)| MlpRow { input, route })
            .collect();
        Ok((rows, retained))
    }

    pub(super) fn route_prefill_logits(
        &self,
        layer: usize,
        logits: &[f32],
        bias: &[f32],
        rows: usize,
    ) -> Result<Vec<MoeRoute>, EngError> {
        if bias.is_empty() || rows.checked_mul(bias.len()) != Some(logits.len()) {
            return Err(bad("unexpected Inkling grouped router logit extent"));
        }
        let routes = logits
            .chunks_exact(bias.len())
            .map(|scores| self.moe_route_from_logits(layer, scores.to_vec(), bias))
            .collect::<Result<Vec<_>, _>>()?;
        if std::env::var("FIFI_INKLING_DEBUG_PREFILL_ROUTES").as_deref() == Ok("1") {
            let picked: Vec<_> = routes.iter().map(|route| &route.selected).collect();
            // The selection margin of every row: the weakest selected score above the
            // strongest unselected one (sigmoid score plus bias, the order `route` sorts by).
            let margins: Vec<f32> = routes
                .iter()
                .map(|selected| {
                    let key = |id: usize| selected.scores[id] + bias[id];
                    let low = selected
                        .selected
                        .iter()
                        .map(|&id| key(id))
                        .fold(f32::INFINITY, f32::min);
                    let high = (0..bias.len())
                        .filter(|id| !selected.selected.contains(id))
                        .map(key)
                        .fold(f32::NEG_INFINITY, f32::max);
                    low - high
                })
                .collect();
            eprintln!(
                "Inkling grouped route diagnostic {}",
                serde_json::json!({"layer":layer,"routes":picked,"margins":margins})
            );
        }
        Ok(routes)
    }

    fn diagnose_chained_projections(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: &[PromptRow<'_>],
        weights: &GgufTextWeights,
        fused: &[super::super::cpu_attention::AttentionProjection],
    ) -> Result<(), EngError> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static DONE: AtomicBool = AtomicBool::new(false);
        if std::env::var("FIFI_INKLING_DEBUG_CHAIN_COMPARE").as_deref() != Ok("1")
            || DONE.swap(true, Ordering::Relaxed)
        {
            return Ok(());
        }
        let (inputs, batched) = self
            .prompt_attention_inputs(visit, layer, rows)?
            .ok_or_else(|| bad("diagnostic requires baseline grouped inputs"))?;
        let baseline = self
            .project_prompt_attention(weights, layer, visit, &inputs, batched, false)?
            .ok_or_else(|| bad("diagnostic requires baseline attention projections"))?;
        if baseline.len() != fused.len() {
            return Err(bad("diagnostic row count differs"));
        }
        let mut stats = [(0.0f32, 0usize, 0usize, 0usize); 8];
        for (row, (a, b)) in baseline.iter().zip(fused).enumerate() {
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                if a.len() != b.len() {
                    return Err(bad("diagnostic projection width differs"));
                }
                for (column, (&x, &y)) in a.iter().zip(b).enumerate() {
                    if !x.is_finite() || !y.is_finite() {
                        return Err(bad("nonfinite diagnostic projection"));
                    }
                    let delta = (x - y).abs();
                    stats[index].3 += usize::from(x.to_bits() != y.to_bits());
                    if delta > stats[index].0 {
                        stats[index].0 = delta;
                        stats[index].1 = row;
                        stats[index].2 = column;
                    }
                }
            }
        }
        eprintln!(
            "Inkling grouped chain diagnostic {}",
            serde_json::json!({
                "layer":layer, "state_slot":visit.state_slot, "rows":rows.len(), "baseline_batched":batched,
                "projections":stats.map(|(max_abs,row,column,unequal)| serde_json::json!({"max_abs":max_abs,"row":row,"column":column,"unequal":unequal})),
            })
        );
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn prompt_attention_inputs(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: &[PromptRow<'_>],
    ) -> Result<Option<(Vec<[Vec<f32>; 3]>, bool)>, EngError> {
        if !prefill_projection_batch_enabled() {
            return Ok(None);
        }
        let grouped = self.grouped_attention_inputs(
            visit,
            layer,
            rows.iter().map(|row| (&row.stream, row.bank.as_ref())),
        )?;
        let batched_controllers = grouped.is_some();
        let inputs = match grouped {
            Some(inputs) => inputs,
            None if rows.iter().all(|row| row.bank.is_none()) => {
                self.unbanked_attention_inputs(visit, layer, rows)?
            }
            None => rows
                .iter()
                .map(|row| {
                    self.attention_inputs_scalar(visit, layer, &row.stream, row.bank.as_ref())
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        Ok(Some((inputs, batched_controllers)))
    }

    /// The attention inputs of rows without a message bank, as `attention_inputs_scalar` makes
    /// them: each role's depth mix (or the current row on the first visit), RMS-normalized and
    /// scaled. The weights are read once and the rows are split over host threads.
    fn unbanked_attention_inputs(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: &[PromptRow<'_>],
    ) -> Result<Vec<[Vec<f32>; 3]>, EngError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let local = visit.attention == Attention::Local;
        let roles = if local { 1 } else { 3 };
        let hidden = self.hidden();
        let queries = match visit.weight_layer.checked_sub(1) {
            None => Vec::new(),
            Some(query) => (0..roles)
                .map(|row| {
                    self.weights.row(
                        &format!("model.llm.attn_res.attn_queries.{query}"),
                        row,
                        hidden,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let gammas = (0..roles)
            .map(|i| {
                self.weights
                    .vec(&attention_norm_name(layer, local, i), hidden)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let eps = self.math.rms_norm_eps;
        let streams: Vec<&Stream> = rows.iter().map(|row| &row.stream).collect();
        Ok(crate::family::host_rows::map_shared(
            &streams,
            PROMPT_ROW_SPLIT,
            |stream| {
                let input = |role: usize| {
                    let mut x = match queries.get(role) {
                        Some(query) => stream.mix(0, query),
                        None => stream.current().to_vec(),
                    };
                    rms_norm(&mut x, eps);
                    for (value, gamma) in x.iter_mut().zip(gammas[role].iter()) {
                        *value *= gamma;
                    }
                    x
                };
                let first = input(0);
                if local {
                    [first.clone(), first.clone(), first]
                } else {
                    [first, input(1), input(2)]
                }
            },
        ))
    }

    fn project_prompt_attention(
        &self,
        weights: &GgufTextWeights,
        layer: &str,
        visit: LayerVisit,
        inputs: &[[Vec<f32>; 3]],
        batched_controllers: bool,
        allow_q8_row_reuse: bool,
    ) -> Result<Option<Vec<super::super::cpu_attention::AttentionProjection>>, EngError> {
        // Grouped verification shares this path but keeps the existing projection dispatch.
        let projected = self.prefill_attention_projections(
            weights,
            layer,
            visit.attention,
            inputs,
            allow_q8_row_reuse,
        )?;
        if projected.is_none() && batched_controllers {
            // Keep the completed controller rows when Q/K/V projection itself falls back.
            return inputs
                .iter()
                .map(|input| self.project_attention(layer, visit.attention, input))
                .collect::<Result<Vec<_>, _>>()
                .map(Some);
        }
        Ok(projected)
    }

    fn prompt_experts(
        &self,
        weights: &GgufTextWeights,
        layer: usize,
        prefix: &str,
        mlp: &[MlpRow],
        (retained, destination): (Option<DeviceRange<'_>>, Option<DeviceRange<'_>>),
    ) -> Result<(Vec<Vec<f32>>, bool), EngError> {
        let span = self.phase_span(super::Work::Experts);
        let routes: Vec<_> = mlp
            .iter()
            .map(|row| PrefillRoute {
                ids: &row.route.selected,
                input: &row.input,
            })
            .collect();
        let names = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let names = names.each_ref().map(String::as_str);
        let mut ids = Vec::new();
        let batch = if let Some(activations) = retained {
            ids.extend(routes.iter().map(|row| row.ids));
            ExpertPrefillRows::Resident {
                ids: &ids,
                hidden: self.hidden(),
                activations,
            }
        } else {
            ExpertPrefillRows::Host(&routes)
        };
        if let Some(output) = weights.prefill_device_experts(
            names,
            layer as u32,
            (batch, destination),
            [self.math.mlp_gate_softcap, self.math.mlp_up_softcap],
            |row, id| mlp[row].route.weight(id, self.math.route_scale),
        )? {
            return Ok((output, destination.is_some()));
        }
        // Draft conditioning and verification preserve the scalar expert group's
        // device activation and rank reduction, including on fallback.
        let batch = if self.draft_window_enabled() {
            None
        } else {
            weights.prefill_experts(names, layer as u32, &routes, |gate, up| {
                self.math.expert_activation(gate, up)
            })?
        };
        let Some(batch) = batch else {
            drop(span);
            return Ok((
                mlp.iter()
                    .map(|row| self.moe_with_route(layer, prefix, &row.input, &row.route))
                    .collect::<Result<_, _>>()?,
                false,
            ));
        };
        Ok((
            mlp.iter()
                .zip(batch)
                .map(|(input, results)| {
                    let mut output = vec![0.; self.hidden()];
                    for (&id, contribution) in input.route.selected.iter().zip(results) {
                        add(
                            &mut output,
                            &contribution,
                            input.route.weight(id, self.math.route_scale),
                        );
                    }
                    output
                })
                .collect(),
            false,
        ))
    }
}

fn prefill_projection_batch_enabled() -> bool {
    Switch::BatchAttentionProjections.on()
}

#[test]
#[ignore = "requires a leased CUDA device and a fully resident packed Inkling 2 Mini GGUF"]
fn grouped_draft_prefill_matches_serial_taps_and_restored_state() {
    use crate::family::{session::TokenStepper, speculate::driver::WindowModel};
    use std::{env, path::Path, sync::Arc};

    let directory = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF");
    let mut model = CpuForward::open(Path::new(&directory), 11).unwrap();
    model.spec.sliding_window_size = 2;
    model.prepare_draft(vec![5, 23, 41, 48], 3).unwrap();
    model.enable_cuda(0).unwrap();
    assert!(!model.supports_grouped_prefill());

    model.feed_rows(&[1, 2, 3]).unwrap();
    let prefix = model.snapshot().unwrap();
    let suffix = [4, 5, 6, 7, 8, 9];
    model.feed_rows(&suffix).unwrap();
    let serial = model.snapshot().unwrap();
    let logits = model.logits(10).unwrap();
    assert_eq!(serial.features.len(), serial.position);
    assert!(
        serial
            .features
            .iter()
            .all(|row| row.len() == 4 * model.hidden())
    );
    model.enable_grouped_draft_prefill();
    assert!(model.supports_grouped_prefill());

    for width in [2, 4] {
        let old_lineage = Arc::clone(&model.draft_lineage);
        model.restore(&prefix);
        assert!(!Arc::ptr_eq(&old_lineage, &model.draft_lineage));
        assert_eq!(model.features, prefix.features);
        assert_eq!(model.position(), prefix.position);
        assert!(model.feed_rows(&[suffix[0], model.vocab_size()]).is_err());
        assert_eq!(model.features, prefix.features);
        assert_eq!(model.position(), prefix.position);

        model.run_window(&suffix[..1]).unwrap();
        let resumed_lineage = Arc::clone(&model.draft_lineage);
        for chunk in suffix.chunks(width) {
            model.feed_rows(chunk).unwrap();
            assert_eq!(model.features.len(), model.position());
        }
        assert!(model.window.is_none());
        assert!(Arc::ptr_eq(&resumed_lineage, &model.draft_lineage));
        assert_eq!(model.position(), serial.position);
        for (visit, (actual, expected)) in model.state.iter().zip(&serial.visits).enumerate() {
            assert!(
                actual == expected,
                "width {width}, visit {visit} state differed"
            );
        }
        for (row, (actual, expected)) in model.features.iter().zip(&serial.features).enumerate() {
            assert!(
                actual
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(expected.iter().map(|x| x.to_bits())),
                "width {width}, row {row} tap bits differed"
            );
        }
        assert!(
            model
                .logits(10)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .eq(logits.iter().map(|x| x.to_bits())),
            "width {width} continuation logits differed"
        );
    }
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and serial reference logits"]
fn grouped_prefill_matches_serial_logits_and_restored_state() {
    use std::{env, fs, path::Path};

    let directory = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF directory");
    let reference =
        env::var_os("FIFI_INKLING_REFERENCE_LOGITS").expect("set the serial logits path");
    let tokens: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("set comma-separated token ids")
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    let expected = fs::read(reference).unwrap();
    let mut model = CpuForward::open(Path::new(&directory), tokens.len()).unwrap();
    model.enable_cuda(0).unwrap();
    assert!(model.supports_grouped_prefill());
    let vocab = model.vocab_size() as usize;
    let row_bytes = vocab * size_of::<f32>();
    assert_eq!(expected.len(), tokens.len() * row_bytes);

    for width in [2, 8, 16] {
        model.reset();
        // The same held prefix is used for two independent continuations.
        for &token in &tokens[..3] {
            model.token(token, false).unwrap();
        }
        let prefix = model.snapshot().unwrap();
        for _ in 0..2 {
            model.restore(&prefix);
            for chunk in tokens[3..].chunks(width + 1) {
                let (&last, head) = chunk.split_last().unwrap();
                if head.len() > 1 {
                    model.prefill_grouped(head).unwrap();
                } else if let Some(&token) = head.first() {
                    model.token(token, false).unwrap();
                }
                let position = model.position();
                let logits = model.token(last, true).unwrap().unwrap();
                let reference = &expected[position * row_bytes..][..row_bytes];
                assert_eq!(logits.len(), vocab);
                assert!(
                    reference
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(logits)
                        .all(|(&bytes, logit)| bytes == logit.to_le_bytes()),
                    "logit mismatch at position {position}, chunk {width}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and fixed prompt token IDs"]
fn device_grouped_prefill_last_logits() {
    use std::{env, fs, path::Path};

    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF path");
    let output = env::var_os("FIFI_INKLING_LOGITS").expect("set the output logits path");
    let tokens: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("set comma-separated token IDs")
        .split(',')
        .map(|value| value.parse().expect("valid token ID"))
        .collect();
    let prefix_rows: usize = env::var("FIFI_INKLING_GROUPED_PREFIX_ROWS")
        .expect("set the number of grouped input rows before the logits positions")
        .parse()
        .expect("valid grouped input size");
    assert!((2..tokens.len()).contains(&prefix_rows));

    let mut model = CpuForward::open(Path::new(&checkpoint), tokens.len()).unwrap();
    model.enable_cuda(0).unwrap();
    assert!(model.supports_grouped_prefill());
    let (prefix, tail) = tokens.split_at(prefix_rows);
    model.prefill_grouped(prefix).unwrap();
    let saved = model.snapshot().unwrap();
    let trace = env::var("FIFI_INKLING_TEST_VISIT_FINGERPRINT").as_deref() == Ok("1");
    if trace {
        eprintln!(
            "Inkling first visit prefill state {:016x}",
            model.state[0].parity_fingerprint()
        );
    }
    let continuation = |model: &mut CpuForward| -> Vec<u8> {
        tail.iter()
            .enumerate()
            .flat_map(|(i, &token)| {
                let logits = model.token(token, true).unwrap().unwrap();
                if trace && i == 0 {
                    eprintln!(
                        "Inkling first visit first-decode state {:016x}",
                        model.state[0].parity_fingerprint()
                    );
                }
                logits
            })
            .flat_map(f32::to_le_bytes)
            .collect()
    };
    let logits = continuation(&mut model);
    model.restore(&saved);
    let restored = continuation(&mut model);
    assert_eq!(logits, restored, "restored continuation logits differ");
    fs::write(output, logits).unwrap();
}
