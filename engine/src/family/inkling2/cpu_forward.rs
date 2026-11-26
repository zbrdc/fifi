//! Text execution for Inkling 2 Mini, including CUDA-backed projections.

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::inkling2::switches::Switch;
use std::{borrow::Cow, fs, path::Path, sync::Arc};

use serde::Deserialize;

use super::{
    Attention, LayerVisit, TextSpec, bad,
    cpu_attention::VisitState,
    cpu_weights::{NativeWeights, Weights},
    gguf_weights::{GgufPlan, Metadata, gguf_path},
    phase_profile::{PhaseProfile, Span, Work},
};
use crate::{EngError, expert_tier::ExpertId};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod controller_batch;
mod depth_mix;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod prefill;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod prompt_streams;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod resident_visit;
mod window;

use depth_mix::Stream;

pub(super) const MIX_EPS: f32 = 1e-6;
/// How a prompt's independent per-row host work is split over threads.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) const PROMPT_ROW_SPLIT: crate::family::host_rows::RowSplit =
    crate::family::host_rows::RowSplit {
        max_threads: 8,
        min_rows: 16,
    };
pub(super) const ROUTE_NORMALIZATION_FLOOR: f32 = 1.0 / 16384.0;

/// The route of one row's router logits, without recording it.
fn route_scores(mut scores: Vec<f32>, bias: &[f32], top_k: usize) -> Result<MoeRoute, EngError> {
    for score in &mut scores {
        *score = sigmoid(*score);
    }
    let selected = route(&scores, bias, top_k)?;
    let total = selected
        .iter()
        .map(|&id| scores[id])
        .sum::<f32>()
        .max(ROUTE_NORMALIZATION_FLOOR);
    Ok(MoeRoute {
        selected,
        scores,
        total,
    })
}

#[derive(Deserialize)]
struct CpuConfig {
    eos_token_id: u32,
    text_config: TextMath,
}

#[derive(Deserialize)]
pub(super) struct TextMath {
    pub(super) d_rel: usize,
    pub(super) rel_extent: usize,
    pub(super) sconv_kernel_size: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) qk_l2_clip_threshold: f32,
    pub(super) rel_l2_clip_threshold: f32,
    log_scaling_n_floor: usize,
    final_logit_softcapping: f32,
    mlp_gate_softcap: f32,
    mlp_up_softcap: f32,
    route_scale: f32,
    q_bias: bool,
    o_bias: bool,
    attn_res_normalize_values: bool,
    use_value_norm_no_gamma: bool,
    use_gate_bias: bool,
}

impl TextMath {
    fn expert_activation(&self, gate: f32, up: f32) -> f32 {
        softcap(gate, self.mlp_gate_softcap) * sigmoid(gate) * softcap(up, self.mlp_up_softcap)
    }
}

pub(super) struct CpuForward {
    pub(super) spec: TextSpec,
    pub(super) math: TextMath,
    pub(super) weights: Weights,
    pub(super) context: usize,
    pub(super) phase_profile: Option<Box<PhaseProfile>>,
    visits: Vec<LayerVisit>,
    state: Vec<VisitState>,
    position: usize,
    failed: bool,
    eos_token_id: u32,
    taps: Vec<usize>,
    features: Vec<Arc<[f32]>>,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    draft_lineage: Arc<()>,
    window: Option<window::PendingWindow>,
    window_limit: usize,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    grouped_verify: bool,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    grouped_draft_prefill: bool,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    resident_visit: std::cell::RefCell<Option<resident_visit::Workspace>>,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    resident_token: Option<resident_visit::TokenWorkspace>,
    #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
    resident_pair: [std::cell::RefCell<Option<resident_visit::Pair>>; 2],
}

pub(super) struct PrefixState {
    position: usize,
    visits: Vec<VisitState>,
    features: Vec<Arc<[f32]>>,
}

struct ForwardRow {
    head_input: Option<Vec<f32>>,
    features: Option<Arc<[f32]>>,
}

/// Capture the residual entering each selected execution visit, followed by the
/// normalized head input when the final position is selected.
struct TapRow<'a> {
    pending: &'a [usize],
    values: Vec<f32>,
}

impl<'a> TapRow<'a> {
    fn new(pending: &'a [usize], width: usize) -> Self {
        Self {
            pending,
            values: Vec::with_capacity(pending.len() * width),
        }
    }

    fn wants(&self, visit: usize) -> bool {
        self.pending.first() == Some(&visit)
    }

    fn capture(&mut self, visit: usize, hidden: &[f32]) {
        if self.wants(visit) {
            self.values.extend_from_slice(hidden);
            self.pending = &self.pending[1..];
        }
    }

    fn finish(self) -> Option<Arc<[f32]>> {
        debug_assert!(self.pending.is_empty());
        (!self.values.is_empty()).then(|| self.values.into())
    }
}

struct MessageBank {
    first_block: usize,
    stream: Stream,
}

#[derive(Clone, Copy)]
enum Role {
    Query,
    Key,
    Value,
    Mlp,
}

const ATTENTION_ROLES: [(Role, &str); 3] = [
    (Role::Query, "q_input_norm"),
    (Role::Key, "k_input_norm"),
    (Role::Value, "v_input_norm"),
];

struct ControlInput {
    gate: Vec<f32>,
    message: Vec<f32>,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::Query => "attn",
            Self::Key => "k",
            Self::Value => "v",
            Self::Mlp => "mlp",
        }
    }
}

impl CpuForward {
    pub(super) fn open(directory: &Path, context: usize) -> Result<Self, EngError> {
        let (spec, config, weights) = if let Some(path) = gguf_path(directory)? {
            let plan = GgufPlan::open(&path)?;
            let config = CpuConfig::from_gguf(&plan, context)?;
            let (spec, weights) = plan.map(&path)?;
            (spec, config, Weights::Gguf(Box::new(weights)))
        } else {
            let bytes = fs::read(directory.join("config.json"))
                .map_err(|error| bad(format!("read text configuration: {error}")))?;
            let spec = TextSpec::from_config(&bytes)?;
            let config = CpuConfig::parse(&bytes, &spec, context)?;
            let weights = NativeWeights::open(directory, &spec)?;
            (spec, config, Weights::Native(Box::new(weights)))
        };
        let gguf = matches!(&weights, Weights::Gguf(_));
        let profile =
            gguf && std::env::var("FIFI_INKLING_GGUF_PHASE_PROFILE").as_deref() == Ok("1");
        let visits = spec.visits();
        let state = (0..visits.len()).map(|_| VisitState::default()).collect();
        Ok(Self {
            spec,
            math: config.text_config,
            weights,
            context,
            phase_profile: profile.then(Box::default),
            visits,
            state,
            position: 0,
            failed: false,
            eos_token_id: config.eos_token_id,
            taps: Vec::new(),
            features: Vec::new(),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            draft_lineage: Arc::new(()),
            window: None,
            window_limit: 0,
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            grouped_verify: false,
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            grouped_draft_prefill: false,
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            resident_visit: Default::default(),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            resident_token: None,
            #[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
            resident_pair: Default::default(),
        })
    }

    pub(super) fn phase_span(&self, work: Work) -> Option<Span<'_>> {
        self.phase_profile.as_ref()?.span(work)
    }

    pub(super) fn hidden(&self) -> usize {
        self.spec.hidden_size as usize
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn enable_cuda(&mut self, host_reserve: u64) -> Result<(), EngError> {
        let attention = super::cuda_projection::gguf::attention::Config {
            context: self.context,
            relative_dims: self.math.d_rel,
            relative_extent: self.math.rel_extent,
        };
        let config = super::gguf_weights::CudaConfig {
            attention,
            draft_target: self.draft_window_enabled(),
        };
        self.weights
            .enable_cuda(&self.spec, self.math.rms_norm_eps, host_reserve, config)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn draft_cuda_context(&self) -> Option<&crate::kernels::cuda::CudaContext> {
        match &self.weights {
            Weights::Gguf(weights) => weights.cuda_context(),
            Weights::Native(_) => None,
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn draft_target_reserve_credit(&self) -> u64 {
        match &self.weights {
            Weights::Gguf(weights) => weights
                .resident_attention()
                .map_or(0, |(_, attention)| attention.reserve_credit()),
            Weights::Native(_) => 0,
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn enable_cuda_draft_head(&mut self, cap: f32) -> Result<(), EngError> {
        let Weights::Gguf(weights) = &mut self.weights else {
            return Err(bad("CUDA draft head requires an Inkling GGUF target"));
        };
        weights.enable_cuda_draft_head(cap)
    }

    pub(super) fn vocab_size(&self) -> u32 {
        self.spec.unpadded_vocab_size as u32
    }

    pub(super) fn validate_tokenizer(&self, vocab: u32, eos: &[u32]) -> Result<(), EngError> {
        validate_tokenizer(self.vocab_size(), self.eos_token_id, vocab, eos)
    }

    /// Conservative RAM accounting for F32 KV, row ownership and four causal convolutions.
    pub(super) fn host_state_bytes(&self) -> Result<u64, EngError> {
        let payload = self.spec.kv_plan(self.context as u32)?.windowed_bytes * 2;
        let row_overhead = payload / 64;
        let convolution_and_deques = self.visits.len() as u64 * 256 * 1024;
        let feature_bytes = self
            .feature_bytes_per_row()
            .checked_mul(self.context as u64)
            .ok_or_else(|| {
                EngError::MemoryPlan("Inkling target feature cache size overflows".into())
            })?;
        (payload + row_overhead + convolution_and_deques)
            .checked_add(feature_bytes)
            .ok_or_else(|| EngError::MemoryPlan("Inkling host state size overflows".into()))
    }

    pub(super) fn snapshot(&mut self) -> Result<PrefixState, EngError> {
        // The method exists only in CUDA builds; without the feature no row is ever device-only.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        self.materialize_resident()?;
        Ok(PrefixState {
            position: self.position,
            visits: self.state.clone(),
            features: self.features.clone(),
        })
    }

    pub(super) fn restore(&mut self, saved: &PrefixState) {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        self.invalidate_device_attention();
        // The saved host rows replace the sequence wholesale; device-only rows are dropped with
        // their cursors, so no visit stays resident.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        debug_assert!(saved.visits.iter().all(|state| !state.is_resident()));
        self.window = None;
        self.features.clone_from(&saved.features);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if !self.taps.is_empty() {
            self.draft_lineage = Arc::new(());
        }
        self.state.clone_from(&saved.visits);
        self.position = saved.position;
        self.failed = false;
    }

    pub(super) fn position(&self) -> usize {
        self.position
    }

    pub(super) fn reset(&mut self) {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        self.invalidate_device_attention();
        self.window = None;
        self.features.clear();
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if !self.taps.is_empty() {
            self.draft_lineage = Arc::new(());
        }
        for slot in &mut self.state {
            *slot = VisitState::default();
        }
        // A failed forward moved the invalidated state out of the sequence.
        self.state
            .resize_with(self.visits.len(), VisitState::default);
        self.position = 0;
        self.failed = false;
    }

    pub(super) fn token(
        &mut self,
        id: u32,
        collect_logits: bool,
    ) -> Result<Option<Vec<f32>>, EngError> {
        if self.failed {
            return Err(bad(
                "a previous CPU forward failed and invalidated the sequence state",
            ));
        }
        if self.position >= self.context || u64::from(id) >= self.spec.unpadded_vocab_size {
            return Err(bad("token or sequence is outside the Inkling text limits"));
        }

        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if collect_logits && self.admit_resident_token()? {
            return self.commit_resident_token(id).map(Some);
        }
        // The host-orchestrated path below reads host rows; a chain token left them on the device.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        self.materialize_resident()?;
        self.window = None;
        self.failed = true;
        let mut state = std::mem::take(&mut self.state);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        let profile = self
            .weights
            .cuda_projection()
            .and_then(|cuda| cuda.begin_token().map(|start| (cuda, start)));
        let row = self.forward_row(&mut state, id, collect_logits)?;
        let logits = row
            .head_input
            .as_deref()
            .map(|input| self.logits_for_head(input))
            .transpose()?;
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some((cuda, start)) = profile {
            cuda.end_token(start, self.position, collect_logits);
        }
        self.position += 1;
        self.state = state;
        self.features.extend(row.features);
        self.failed = false;
        Ok(logits)
    }

    fn forward_row(
        &self,
        state: &mut [VisitState],
        id: u32,
        collect_head_input: bool,
    ) -> Result<ForwardRow, EngError> {
        let mut taps = TapRow::new(&self.taps, self.hidden());
        let stream = self.forward(state, id, &mut taps, collect_head_input)?;
        self.finish_row(&stream, taps, collect_head_input)
    }

    fn finish_row(
        &self,
        stream: &Stream,
        mut taps: TapRow<'_>,
        collect_head_input: bool,
    ) -> Result<ForwardRow, EngError> {
        let end = self.visits.len();
        let head_input = if collect_head_input || taps.wants(end) {
            let hidden = self.final_hidden(stream)?;
            taps.capture(end, &hidden);
            collect_head_input.then_some(hidden)
        } else {
            None
        };
        Ok(ForwardRow {
            head_input,
            features: taps.finish(),
        })
    }

    fn embed(&self, id: u32) -> Result<Stream, EngError> {
        let mut input = self
            .weights
            .row("model.llm.embed.weight", id as usize, self.hidden())?
            .into_owned();
        self.norm(&mut input, "model.llm.embed_norm.weight")?;
        Ok(Stream::new(input))
    }

    fn forward(
        &self,
        state: &mut [VisitState],
        id: u32,
        taps: &mut TapRow<'_>,
        decode: bool,
    ) -> Result<Stream, EngError> {
        let mut stream = self.embed(id)?;
        let core = &self.spec.core;
        let width = core.core_end - core.core_start;

        self.run_visits(
            &self.visits[..core.core_start],
            state,
            &mut stream,
            None,
            taps,
            decode,
        )?;
        let prelude = stream.clone();
        let mut bank = None;
        for pass in 0..core.num_passes {
            let start = core.core_start + pass * width;
            if pass > 0 {
                bank = Some(MessageBank {
                    first_block: prelude.block_count() + 1,
                    stream: std::mem::replace(&mut stream, prelude.clone()),
                });
            }
            self.run_visits(
                &self.visits[start..start + width],
                state,
                &mut stream,
                bank.as_ref(),
                taps,
                decode,
            )?;
        }
        let suffix = core.core_start + core.num_passes * width;
        self.run_visits(
            &self.visits[suffix..],
            state,
            &mut stream,
            None,
            taps,
            decode,
        )?;
        Ok(stream)
    }

    fn final_hidden(&self, stream: &Stream) -> Result<Vec<f32>, EngError> {
        let query = self
            .weights
            .vec("model.llm.attn_res.output_query", self.hidden())?;
        let mut last = stream.mix(0, &query);
        self.norm(&mut last, "model.llm.norm.weight")?;
        Ok(last)
    }

    fn project_logits(&self, hidden: &[f32]) -> Result<Vec<f32>, EngError> {
        self.weights.project(
            super::OUTPUT_HEAD,
            hidden,
            self.spec.unpadded_vocab_size as usize,
        )
    }

    fn logits_for_head(&self, hidden: &[f32]) -> Result<Vec<f32>, EngError> {
        let _span = self.phase_span(Work::TargetVocab);
        let cap = self.math.final_logit_softcapping;
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if !self.draft_window_enabled()
            && let Weights::Gguf(weights) = &self.weights
            && let Some(cuda) = weights.cuda()
            && vocab_softcap_enabled()
            && let Some(logits) =
                cuda.capped_head(hidden, self.spec.unpadded_vocab_size as usize, cap)?
        {
            return Ok(logits);
        }
        let mut logits = self.project_logits(hidden)?;
        for value in &mut logits {
            *value = softcap(*value, cap);
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(bad("non-finite native CPU logits"));
        }
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_visits(
        &self,
        visits: &[LayerVisit],
        state: &mut [VisitState],
        stream: &mut Stream,
        bank: Option<&MessageBank>,
        taps: &mut TapRow<'_>,
        decode: bool,
    ) -> Result<(), EngError> {
        for &visit in visits {
            taps.capture(visit.state_slot, stream.current());
            self.layer(visit, &mut state[visit.state_slot], stream, bank, decode)?;
        }
        Ok(())
    }

    fn layer(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        stream: &mut Stream,
        bank: Option<&MessageBank>,
        decode: bool,
    ) -> Result<(), EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if decode
            && let Some(bank) = bank
            && self.try_resident_visit(visit, state, stream, bank)?
        {
            return Ok(());
        }
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let residual = self.mlp_residual(
            visit,
            &layer,
            state,
            stream,
            bank,
            AttentionInput::Token { decode },
        )?;
        let prefix = format!("{layer}.mlp");
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if decode
            && !self.draft_window_enabled()
            && let Some(bank) = bank
            && let Some(mlp) =
                self.chained_mlp_experts(physical, &layer, &prefix, &residual, bank)?
        {
            return self.finish_layer(&layer, state, stream, mlp);
        }
        let input = self.finish_mlp_input(residual, bank, physical, &layer)?;
        let mlp = self.moe(physical, &prefix, &input, decode)?;
        self.finish_layer(&layer, state, stream, mlp)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn chained_mlp_experts(
        &self,
        physical: usize,
        layer: &str,
        prefix: &str,
        input: &[f32],
        bank: &MessageBank,
    ) -> Result<Option<Vec<f32>>, EngError> {
        use super::cuda_projection::gguf::{
            ControlledAttentionInput,
            router::{MlpController, RouteInput},
        };
        if !Switch::ChainMlpExperts.on() {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        if !weights
            .cuda()
            .is_some_and(|cuda| cuda.has_resident_route(physical as u32))
        {
            return Ok(None);
        }
        let (control, gamma) = {
            let _span = self.phase_span(Work::Controller);
            (
                self.control_input(input, bank, physical, Role::Mlp)?,
                self.weights
                    .vec(&format!("{layer}.mlp_norm.weight"), self.hidden())?,
            )
        };
        let names = control_names(layer, Role::Mlp);
        weights.device_routed_experts(
            prefix,
            RouteInput {
                layer: physical as u32,
                decode: true,
                input,
                caps: [self.math.mlp_gate_softcap, self.math.mlp_up_softcap],
                scale: self.math.route_scale,
            },
            self.phase_profile.as_deref(),
            Some(MlpController {
                names: names.each_ref().map(String::as_str),
                rank: self.spec.core.gate_rank as usize,
                eps: self.math.rms_norm_eps,
                input: ControlledAttentionInput {
                    gate: &control.gate,
                    input,
                    message: &control.message,
                    gamma: &gamma,
                },
            }),
        )
    }

    fn mlp_residual(
        &self,
        visit: LayerVisit,
        layer: &str,
        state: &mut VisitState,
        stream: &mut Stream,
        bank: Option<&MessageBank>,
        attention: AttentionInput,
    ) -> Result<Vec<f32>, EngError> {
        let physical = visit.weight_layer;
        let query = self.weights.vec(
            &format!("model.llm.attn_res.mlp_queries.{physical}"),
            self.hidden(),
        )?;
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Weights::Gguf(weights) = &self.weights {
            weights.predict_experts(physical, || stream.mix(0, &query));
        }
        let attn = self.layer_attention(visit, layer, state, stream, bank, attention)?;
        Ok(self.mlp_residual_after_attention(physical, stream, &query, attn))
    }

    fn mlp_residual_after_attention(
        &self,
        physical: usize,
        stream: &mut Stream,
        query: &[f32],
        attn: Vec<f32>,
    ) -> Vec<f32> {
        let seal = physical.is_multiple_of(self.spec.n_layers_per_attn_res_block);
        attention_residual(stream, seal, query, attn)
    }

    fn finish_mlp_input(
        &self,
        mut input: Vec<f32>,
        bank: Option<&MessageBank>,
        physical: usize,
        layer: &str,
    ) -> Result<Vec<f32>, EngError> {
        self.control(&mut input, bank, physical, layer, Role::Mlp)?;
        self.norm(&mut input, &format!("{layer}.mlp_norm.weight"))?;
        Ok(input)
    }

    fn finish_layer(
        &self,
        layer: &str,
        state: &mut VisitState,
        stream: &mut Stream,
        mlp: Vec<f32>,
    ) -> Result<(), EngError> {
        stream.add_current(&self.mlp_conv(mlp, layer, state)?);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_attention(
        &self,
        visit: LayerVisit,
        layer: &str,
        state: &mut VisitState,
        stream: &Stream,
        bank: Option<&MessageBank>,
        attention: AttentionInput,
    ) -> Result<Vec<f32>, EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        {
            if let AttentionInput::Grouped(Some(projected)) = attention {
                return self.attention(layer, visit, state, projected);
            }
            if let AttentionInput::Token { decode } = attention {
                if decode
                    && let Some(output) =
                        self.chained_local_decode(visit, layer, state, stream, bank)?
                {
                    return Ok(output);
                }
                if let Some(bank) = bank
                    && !self.draft_window_enabled()
                    && let Some(projected) =
                        self.chained_attention_inputs(visit, layer, stream, bank)?
                {
                    return self.attention(layer, visit, state, projected);
                }
            }
        }
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let AttentionInput::Token { decode: _decode } = attention;
        let inputs = self.attention_inputs(visit, layer, stream, bank)?;
        let projected = self.project_attention(layer, visit.attention, &inputs)?;
        self.attention(layer, visit, state, projected)
    }

    fn attention_inputs(
        &self,
        visit: LayerVisit,
        layer: &str,
        stream: &Stream,
        bank: Option<&MessageBank>,
    ) -> Result<[Vec<f32>; 3], EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(mut rows) =
            self.grouped_attention_inputs(visit, layer, std::iter::once((stream, bank)))?
        {
            return Ok(rows.pop().expect("one attention input row was requested"));
        }
        self.attention_inputs_scalar(visit, layer, stream, bank)
    }

    fn attention_inputs_scalar(
        &self,
        visit: LayerVisit,
        layer: &str,
        stream: &Stream,
        bank: Option<&MessageBank>,
    ) -> Result<[Vec<f32>; 3], EngError> {
        let local = visit.attention == Attention::Local;
        let mut inputs = [Vec::new(), Vec::new(), Vec::new()];
        for (i, (role, _)) in ATTENTION_ROLES.into_iter().enumerate() {
            if local && i > 0 && bank.is_none() {
                inputs[i] = inputs[0].clone();
                continue;
            }
            let mut input = self.attention_input(visit, stream, i)?;
            self.control(&mut input, bank, visit.weight_layer, layer, role)?;
            self.norm_attention_input(&mut input, layer, local, i)?;
            inputs[i] = input;
        }
        Ok(inputs)
    }

    fn attention_input(
        &self,
        visit: LayerVisit,
        stream: &Stream,
        role_index: usize,
    ) -> Result<Vec<f32>, EngError> {
        let Some(query_index) = visit.weight_layer.checked_sub(1) else {
            return Ok(stream.current().to_vec());
        };
        let row = if visit.attention == Attention::Local {
            0
        } else {
            role_index
        };
        let query = self.weights.row(
            &format!("model.llm.attn_res.attn_queries.{query_index}"),
            row,
            self.hidden(),
        )?;
        Ok(stream.mix(0, &query))
    }

    fn norm_attention_input(
        &self,
        input: &mut [f32],
        layer: &str,
        local: bool,
        role_index: usize,
    ) -> Result<(), EngError> {
        self.norm(input, &attention_norm_name(layer, local, role_index))
    }

    fn norm(&self, values: &mut [f32], weight: &str) -> Result<(), EngError> {
        let weights = self.weights.vec(weight, values.len())?;
        rms_norm(values, self.math.rms_norm_eps);
        for (value, weight) in values.iter_mut().zip(weights.iter()) {
            *value *= weight;
        }
        Ok(())
    }

    fn control(
        &self,
        input: &mut [f32],
        bank: Option<&MessageBank>,
        physical: usize,
        layer: &str,
        role: Role,
    ) -> Result<(), EngError> {
        let Some(bank) = bank else {
            return Ok(());
        };
        let _span = self.phase_span(Work::Controller);
        let ControlInput { gate, message } = self.control_input(input, bank, physical, role)?;
        let names = control_names(layer, role);
        let projection = self.weights.controller(
            names.each_ref().map(String::as_str),
            &gate,
            [self.spec.core.gate_rank as usize, self.hidden()],
        )?;
        apply_control(input, message, projection);
        Ok(())
    }

    fn control_input(
        &self,
        input: &[f32],
        bank: &MessageBank,
        physical: usize,
        role: Role,
    ) -> Result<ControlInput, EngError> {
        self.control_input_with_scales(input, bank, physical, role)
            .map(|(control, _)| control)
    }

    fn control_input_with_scales(
        &self,
        input: &[f32],
        bank: &MessageBank,
        physical: usize,
        role: Role,
    ) -> Result<(ControlInput, [f32; 2]), EngError> {
        let query = self.control_query(physical, role)?;
        Ok(control_input_from(input, bank, &query))
    }

    fn control_query(&self, physical: usize, role: Role) -> Result<Cow<'_, [f32]>, EngError> {
        let (kind, offset) = match role {
            Role::Query => ("cross_pass_queries", 0),
            Role::Key => ("cross_pass_kv_queries", 0),
            Role::Value => ("cross_pass_kv_queries", 1),
            Role::Mlp => ("cross_pass_queries", 1),
        };
        let row = 2 * (physical - self.spec.core.core_start) + offset;
        self.weights.row(
            &format!("model.llm.middle_cycle_{kind}"),
            row,
            self.hidden(),
        )
    }

    fn moe_route(&self, layer: usize, prefix: &str, input: &[f32]) -> Result<MoeRoute, EngError> {
        let _span = self.phase_span(Work::Router);
        let experts = self.spec.n_routed_experts as usize;
        let logits = self
            .weights
            .project(&format!("{prefix}.gate.weight"), input, experts)?;
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        self.moe_route_from_logits(layer, logits, &bias)
    }

    fn moe_route_from_logits(
        &self,
        layer: usize,
        scores: Vec<f32>,
        bias: &[f32],
    ) -> Result<MoeRoute, EngError> {
        let route = route_scores(scores, bias, self.spec.num_experts_per_tok as usize)?;
        self.observe_route(layer, &route.selected);
        Ok(route)
    }

    /// Records a route's experts for the streamed expert cache.
    fn observe_route(&self, layer: usize, selected: &[usize]) {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Weights::Gguf(weights) = &self.weights {
            weights.observe_experts(layer, selected);
        }
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let _ = (layer, selected);
    }

    fn moe(
        &self,
        layer: usize,
        prefix: &str,
        input: &[f32],
        decode: bool,
    ) -> Result<Vec<f32>, EngError> {
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let _ = decode;
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if !self.draft_window_enabled()
            && let Weights::Gguf(weights) = &self.weights
            && let Some(out) = weights.device_routed_experts(
                prefix,
                super::cuda_projection::gguf::router::RouteInput {
                    layer: layer as u32,
                    decode,
                    input,
                    caps: [self.math.mlp_gate_softcap, self.math.mlp_up_softcap],
                    scale: self.math.route_scale,
                },
                self.phase_profile.as_deref(),
                None,
            )?
        {
            return Ok(out);
        }
        let route = self.moe_route(layer, prefix, input)?;
        self.moe_with_route(layer, prefix, input, &route)
    }

    fn moe_with_route(
        &self,
        layer: usize,
        prefix: &str,
        input: &[f32],
        route: &MoeRoute,
    ) -> Result<Vec<f32>, EngError> {
        let _span = self.phase_span(Work::Experts);
        let gate_up = format!("{prefix}.experts.w13_weight");
        let down = format!("{prefix}.experts.w2_weight");
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Weights::Gguf(weights) = &self.weights
            && let Some(output) = weights.batched_experts(
                [&gate_up, &down],
                super::cuda_projection::gguf::streaming::ResidentRoute {
                    layer: layer as u32,
                    ids: &route.selected,
                    input,
                    caps: [self.math.mlp_gate_softcap, self.math.mlp_up_softcap],
                },
                |id| route.weight(id, self.math.route_scale),
            )?
        {
            return Ok(output);
        }
        let activation = self.weights.activate(input)?;
        let mut output = vec![0.0; self.hidden()];
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Weights::Gguf(weights) = &self.weights
            && weights.stream_experts(
                [&gate_up, &down],
                layer as u32,
                &route.selected,
                input,
                |gate, up| self.math.expert_activation(gate, up),
                |id, row| add(&mut output, row, route.weight(id, self.math.route_scale)),
            )?
        {
            return Ok(output);
        }
        for &id in &route.selected {
            let key = ExpertId {
                layer: layer as u32,
                expert: u32::try_from(id).map_err(|_| bad("Inkling expert id overflows"))?,
            };
            let pairs = self
                .weights
                .expert(&gate_up, key, activation.as_ref(), input)?;
            let middle: Vec<f32> = pairs
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[gate, up]| self.math.expert_activation(gate, up))
                .collect();
            let activation = self.weights.activate(&middle)?;
            let contribution = self
                .weights
                .expert(&down, key, activation.as_ref(), &middle)?;
            add(
                &mut output,
                &contribution,
                route.weight(id, self.math.route_scale),
            );
        }
        Ok(output)
    }
}

fn control_names(layer: &str, role: Role) -> [String; 3] {
    let ctrl = format!("{layer}.controllers.{}", role.name());
    ["down", "gain", "forget"].map(|part| format!("{ctrl}.{part}_weight"))
}

fn attention_norm_name(layer: &str, local: bool, role_index: usize) -> String {
    let norm = if local {
        "attn_norm"
    } else {
        ATTENTION_ROLES[role_index].1
    };
    format!("{layer}.{norm}.weight")
}

fn apply_control(input: &mut [f32], message: Vec<f32>, [gain, forget]: [Vec<f32>; 2]) {
    for (((value, message), gain), forget) in input.iter_mut().zip(message).zip(gain).zip(forget) {
        *value = (*value - forget.tanh() * *value) + gain.tanh() * message;
    }
}

enum AttentionInput {
    Token {
        decode: bool,
    },
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    Grouped(Option<super::cpu_attention::AttentionProjection>),
}

struct MoeRoute {
    selected: Vec<usize>,
    scores: Vec<f32>,
    total: f32,
}

impl MoeRoute {
    fn weight(&self, expert: usize, scale: f32) -> f32 {
        scale * (self.scores[expert] / self.total)
    }
}

pub(crate) fn inspect_cpu_config(
    bytes: &[u8],
    spec: &TextSpec,
    context: u32,
    tokenizer_dir: &Path,
) -> Result<(), EngError> {
    let config = CpuConfig::parse(bytes, spec, context as usize)?;
    let bundle = crate::tokenizer::inkling2::load(tokenizer_dir)?;
    validate_tokenizer(
        spec.unpadded_vocab_size as u32,
        config.eos_token_id,
        bundle.vocab_size(),
        bundle.eos_token_ids(),
    )
}

pub(crate) fn inspect_gguf_config(
    plan: &GgufPlan,
    context: u32,
    tokenizer_dir: &Path,
) -> Result<(), EngError> {
    let config = CpuConfig::from_gguf(plan, context as usize)?;
    let bundle = crate::tokenizer::inkling2::load(tokenizer_dir)?;
    validate_tokenizer(
        plan.spec.unpadded_vocab_size as u32,
        config.eos_token_id,
        bundle.vocab_size(),
        bundle.eos_token_ids(),
    )
}

fn validate_tokenizer(
    model_vocab: u32,
    model_eos: u32,
    vocab: u32,
    eos: &[u32],
) -> Result<(), EngError> {
    if model_vocab != vocab || !eos.contains(&model_eos) {
        return Err(bad(
            "tokenizer vocabulary or EOS differs from the checkpoint",
        ));
    }
    Ok(())
}

impl CpuConfig {
    fn from_gguf(plan: &GgufPlan, context: usize) -> Result<Self, EngError> {
        let meta = Metadata::new(&plan.header);
        let arch = plan.layout.architecture();
        let int = |key| {
            meta.integer(&format!("{arch}.{key}"))
                .map(|value| value as usize)
        };
        let float = |key| meta.float(&format!("{arch}.{key}"));
        let eos = meta.integer("tokenizer.ggml.eos_token_id")?;
        if eos >= plan.spec.unpadded_vocab_size {
            return Err(bad("GGUF EOS is outside the unpadded vocabulary"));
        }
        let math = TextMath {
            d_rel: int("attention.relative_length")?,
            rel_extent: int("attention.relative_extent")?,
            sconv_kernel_size: int("shortconv.l_cache")?,
            rms_norm_eps: float("attention.layer_norm_rms_epsilon")?,
            qk_l2_clip_threshold: float("attention.qk_clip")?,
            rel_l2_clip_threshold: float("attention.relative_clip")?,
            log_scaling_n_floor: int("attention.log_scale_floor")?,
            final_logit_softcapping: float("final_logit_softcapping")?,
            mlp_gate_softcap: float("activation.situ_beta")?,
            mlp_up_softcap: float("activation.situ_linear_beta")?,
            route_scale: float("expert_weights_scale")?,
            q_bias: false,
            o_bias: false,
            attn_res_normalize_values: true,
            use_value_norm_no_gamma: true,
            use_gate_bias: true,
        };
        math.validate(context, &plan.spec)?;
        Ok(Self {
            eos_token_id: eos as u32,
            text_config: math,
        })
    }

    fn parse(bytes: &[u8], spec: &TextSpec, context: usize) -> Result<Self, EngError> {
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|error| bad(format!("read text forward parameters: {error}")))?;
        config.text_config.validate(context, spec)?;
        Ok(config)
    }
}

impl TextMath {
    fn validate(&self, context: usize, spec: &TextSpec) -> Result<(), EngError> {
        let max = self.log_scaling_n_floor.min(spec.model_max_length as usize);
        if !(1..=max).contains(&context) {
            return Err(bad(format!(
                "Inkling text context must be in 1..={max} with the current attention implementation"
            )));
        }
        // The CPU pilot accepts the released Mini attention and convolution sizes.
        let attention_dims = [
            spec.num_attention_heads,
            spec.num_key_value_heads,
            spec.head_dim,
            spec.swa_num_attention_heads,
            spec.swa_num_key_value_heads,
            spec.swa_head_dim,
        ];
        if attention_dims != [24, 8, 128, 24, 8, 128]
            || spec.hidden_size != 2304
            || spec.vocab_size != 201_024
            || spec.unpadded_vocab_size != 200_058
            || [self.d_rel, self.rel_extent, self.sconv_kernel_size] != [16, 1024, 4]
            || self.q_bias
            || self.o_bias
            || !self.use_gate_bias
            || !self.attn_res_normalize_values
            || !self.use_value_norm_no_gamma
            || ![
                self.rms_norm_eps,
                self.qk_l2_clip_threshold,
                self.rel_l2_clip_threshold,
                self.final_logit_softcapping,
                self.mlp_gate_softcap,
                self.mlp_up_softcap,
                self.route_scale,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0)
        {
            return Err(bad("unsupported native CPU forward parameters"));
        }
        Ok(())
    }
}

pub(super) fn route(scores: &[f32], bias: &[f32], count: usize) -> Result<Vec<usize>, EngError> {
    let mut order: Vec<(usize, f32)> = scores
        .iter()
        .zip(bias)
        .enumerate()
        .map(|(id, (score, bias))| (id, score + bias))
        .collect();
    if order.iter().any(|&(_, score)| !score.is_finite()) {
        return Err(bad("non-finite expert routing score"));
    }
    order.sort_unstable_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    Ok(order.into_iter().take(count).map(|(id, _)| id).collect())
}

pub(super) fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

pub(super) fn rms_norm(x: &mut [f32], eps: f32) {
    rms_norm_with_scale(x, eps);
}

/// Adds the attention output to the stream (sealing a block at a block boundary) and returns the
/// stream's MLP depth mix for `query`.
fn attention_residual(stream: &mut Stream, seal: bool, query: &[f32], attn: Vec<f32>) -> Vec<f32> {
    if seal {
        stream.seal(attn);
    } else {
        stream.add_current(&attn);
    }
    stream.mix(0, query)
}

/// The controller input of one role: the row's input followed by the message the bank mixes for
/// `query`, each half RMS-normalized, and the two norm scales.
fn control_input_from(
    input: &[f32],
    bank: &MessageBank,
    query: &[f32],
) -> (ControlInput, [f32; 2]) {
    let message = bank.stream.mix(bank.first_block, query);
    let mut gate = input.to_vec();
    gate.extend_from_slice(&message);
    let (left, right) = gate.split_at_mut(input.len());
    let scales = [
        rms_norm_with_scale(left, MIX_EPS),
        rms_norm_with_scale(right, MIX_EPS),
    ];
    (ControlInput { gate, message }, scales)
}

fn rms_norm_with_scale(x: &mut [f32], eps: f32) -> f32 {
    let scale = (dot(x, x) / x.len() as f32 + eps).sqrt().recip();
    for value in x {
        *value *= scale;
    }
    scale
}

pub(super) fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).recip()
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn vocab_softcap_enabled() -> bool {
    Switch::VocabSoftcap.on()
}

fn softcap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

fn add(dst: &mut [f32], src: &[f32], scale: f32) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += scale * src;
    }
}

#[test]
fn routing_uses_bias_to_rank_and_orders_ties_by_expert() {
    assert_eq!(
        route(&[0.9, 0.2, 0.5, 0.5], &[0.0, 1.0, 0.0, 0.0], 3).unwrap(),
        [1, 0, 2]
    );
}

#[test]
#[ignore = "requires the native Inkling 2 Mini safetensors checkpoint"]
fn native_cpu_logits() {
    logits_probe(|_| ());
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires a leased CUDA device and the native Mini checkpoint"]
fn native_cuda_logits() {
    logits_probe(|model| model.enable_cuda(0).unwrap());
}

#[cfg(test)]
fn logits_probe(prepare: impl FnOnce(&mut CpuForward)) {
    use std::{
        env,
        io::{BufWriter, Write},
        time::Instant,
    };

    let directory = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set FIFI_INKLING_MODEL_DIR");
    let output = env::var_os("FIFI_INKLING_LOGITS").expect("set FIFI_INKLING_LOGITS");
    let tokens =
        env::var("FIFI_INKLING_TOKEN_IDS").expect("set comma-separated FIFI_INKLING_TOKEN_IDS");
    let tokens = tokens
        .split(',')
        .map(|token| token.parse::<u32>().expect("invalid token ID"))
        .collect::<Vec<_>>();
    let last_only = env::var_os("FIFI_INKLING_LAST_LOGITS_ONLY").is_some();
    let mut model = CpuForward::open(Path::new(&directory), tokens.len()).unwrap();
    prepare(&mut model);
    let mut output = BufWriter::new(fs::File::create(output).unwrap());
    let last_index = tokens.len() - 1;
    for (index, token) in tokens.into_iter().enumerate() {
        let start = Instant::now();
        let Some(logits) = model
            .token(token, !last_only || index == last_index)
            .unwrap()
        else {
            eprintln!(
                "token={token} prefill seconds={:.3}",
                start.elapsed().as_secs_f64()
            );
            continue;
        };
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        eprintln!(
            "token={token} top_id={} top_logit={} seconds={:.3}",
            top.0,
            top.1,
            start.elapsed().as_secs_f64()
        );
        for value in logits {
            output.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    output.flush().unwrap();
}
