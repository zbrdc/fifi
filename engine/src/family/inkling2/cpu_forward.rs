//! Text execution for Inkling 2 Mini, including CUDA-backed projections.

use std::{fs, path::Path};

use serde::Deserialize;

use super::{
    Attention, LayerVisit, TextSpec, bad,
    cpu_attention::VisitState,
    cpu_weights::{NativeWeights, Weights},
    gguf_weights::{GgufPlan, Metadata, gguf_path},
};
use crate::{EngError, expert_tier::ExpertId};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod prefill;

const MIX_EPS: f32 = 1e-6;

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
    visits: Vec<LayerVisit>,
    state: Vec<VisitState>,
    position: usize,
    failed: bool,
    eos_token_id: u32,
}

pub(super) struct PrefixState {
    position: usize,
    visits: Vec<VisitState>,
}

#[derive(Clone)]
struct Stream {
    blocks: Vec<Vec<f32>>,
    current: Vec<f32>,
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
        let visits = spec.visits();
        let state = (0..visits.len()).map(|_| VisitState::default()).collect();
        Ok(Self {
            spec,
            math: config.text_config,
            weights,
            context,
            visits,
            state,
            position: 0,
            failed: false,
            eos_token_id: config.eos_token_id,
        })
    }

    pub(super) fn hidden(&self) -> usize {
        self.spec.hidden_size as usize
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn enable_cuda(&mut self, host_reserve: u64) -> Result<(), EngError> {
        self.weights
            .enable_cuda(&self.spec, self.math.rms_norm_eps, host_reserve)
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
        Ok(payload + row_overhead + convolution_and_deques)
    }

    pub(super) fn snapshot(&self) -> PrefixState {
        PrefixState {
            position: self.position,
            visits: self.state.clone(),
        }
    }

    pub(super) fn restore(&mut self, saved: &PrefixState) {
        self.state.clone_from(&saved.visits);
        self.position = saved.position;
        self.failed = false;
    }

    pub(super) fn position(&self) -> usize {
        self.position
    }

    pub(super) fn reset(&mut self) {
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

        self.failed = true;
        let mut state = std::mem::take(&mut self.state);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        let profile = self
            .weights
            .cuda_projection()
            .and_then(|cuda| cuda.begin_token().map(|start| (cuda, start)));
        let stream = self.forward(&mut state, id)?;
        let logits = collect_logits.then(|| self.logits(&stream)).transpose()?;
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some((cuda, start)) = profile {
            cuda.end_token(start, self.position, collect_logits);
        }
        self.position += 1;
        self.state = state;
        self.failed = false;
        Ok(logits)
    }

    fn embed(&self, id: u32) -> Result<Stream, EngError> {
        let mut input = self
            .weights
            .row("model.llm.embed.weight", id as usize, self.hidden())?;
        self.norm(&mut input, "model.llm.embed_norm.weight")?;
        Ok(Stream {
            blocks: Vec::new(),
            current: input,
        })
    }

    fn forward(&self, state: &mut [VisitState], id: u32) -> Result<Stream, EngError> {
        let mut stream = self.embed(id)?;
        let core = &self.spec.core;
        let width = core.core_end - core.core_start;

        self.run_visits(&self.visits[..core.core_start], state, &mut stream, None)?;
        let prelude = stream.clone();
        let mut bank = None;
        for pass in 0..core.num_passes {
            let start = core.core_start + pass * width;
            if pass > 0 {
                bank = Some(MessageBank {
                    first_block: prelude.blocks.len() + 1,
                    stream: std::mem::replace(&mut stream, prelude.clone()),
                });
            }
            self.run_visits(
                &self.visits[start..start + width],
                state,
                &mut stream,
                bank.as_ref(),
            )?;
        }
        let suffix = core.core_start + core.num_passes * width;
        self.run_visits(&self.visits[suffix..], state, &mut stream, None)?;
        Ok(stream)
    }

    fn logits(&self, stream: &Stream) -> Result<Vec<f32>, EngError> {
        let query = self
            .weights
            .vec("model.llm.attn_res.output_query", self.hidden())?;
        let mut last = stream.mix(0, &query);
        self.norm(&mut last, "model.llm.norm.weight")?;
        let mut logits = self.weights.project(
            "model.llm.unembed.weight",
            &last,
            self.spec.unpadded_vocab_size as usize,
        )?;
        for value in &mut logits {
            *value = softcap(*value, self.math.final_logit_softcapping);
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(bad("non-finite native CPU logits"));
        }
        Ok(logits)
    }

    fn run_visits(
        &self,
        visits: &[LayerVisit],
        state: &mut [VisitState],
        stream: &mut Stream,
        bank: Option<&MessageBank>,
    ) -> Result<(), EngError> {
        for &visit in visits {
            self.layer(visit, &mut state[visit.state_slot], stream, bank)?;
        }
        Ok(())
    }

    fn layer(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        stream: &mut Stream,
        bank: Option<&MessageBank>,
    ) -> Result<(), EngError> {
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let input = self.mlp_input(visit, &layer, state, stream, bank)?;
        let mlp = self.moe(physical, &format!("{layer}.mlp"), &input)?;
        self.finish_layer(&layer, state, stream, mlp)
    }

    fn mlp_input(
        &self,
        visit: LayerVisit,
        layer: &str,
        state: &mut VisitState,
        stream: &mut Stream,
        bank: Option<&MessageBank>,
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
        let inputs = self.attention_inputs(visit, layer, stream, bank)?;
        let attn = self.attention(layer, visit.attention, &inputs, state)?;
        if physical.is_multiple_of(self.spec.n_layers_per_attn_res_block) {
            stream
                .blocks
                .push(std::mem::replace(&mut stream.current, attn));
        } else {
            add(&mut stream.current, &attn, 1.0);
        }

        let mut mlp_input = stream.mix(0, &query);
        self.control(&mut mlp_input, bank, physical, layer, Role::Mlp)?;
        self.norm(&mut mlp_input, &format!("{layer}.mlp_norm.weight"))?;
        Ok(mlp_input)
    }

    fn finish_layer(
        &self,
        layer: &str,
        state: &mut VisitState,
        stream: &mut Stream,
        mlp: Vec<f32>,
    ) -> Result<(), EngError> {
        add(&mut stream.current, &self.mlp_conv(mlp, layer, state)?, 1.0);
        Ok(())
    }

    fn attention_inputs(
        &self,
        visit: LayerVisit,
        layer: &str,
        stream: &Stream,
        bank: Option<&MessageBank>,
    ) -> Result<[Vec<f32>; 3], EngError> {
        let physical = visit.weight_layer;
        let local = visit.attention == Attention::Local;
        let mut inputs = [Vec::new(), Vec::new(), Vec::new()];
        for (i, (role, norm)) in [
            (Role::Query, "q_input_norm"),
            (Role::Key, "k_input_norm"),
            (Role::Value, "v_input_norm"),
        ]
        .into_iter()
        .enumerate()
        {
            if local && i > 0 && bank.is_none() {
                inputs[i] = inputs[0].clone();
                continue;
            }
            let mut input = if let Some(query_index) = physical.checked_sub(1) {
                let row = if local { 0 } else { i };
                let query = self.weights.row(
                    &format!("model.llm.attn_res.attn_queries.{query_index}"),
                    row,
                    self.hidden(),
                )?;
                stream.mix(0, &query)
            } else {
                stream.current.clone()
            };
            self.control(&mut input, bank, physical, layer, role)?;
            let norm = if local { "attn_norm" } else { norm };
            self.norm(&mut input, &format!("{layer}.{norm}.weight"))?;
            inputs[i] = input;
        }
        Ok(inputs)
    }

    fn norm(&self, values: &mut [f32], weight: &str) -> Result<(), EngError> {
        let weights = self.weights.vec(weight, values.len())?;
        rms_norm(values, self.math.rms_norm_eps);
        for (value, weight) in values.iter_mut().zip(weights) {
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
        let (kind, offset) = match role {
            Role::Query => ("cross_pass_queries", 0),
            Role::Key => ("cross_pass_kv_queries", 0),
            Role::Value => ("cross_pass_kv_queries", 1),
            Role::Mlp => ("cross_pass_queries", 1),
        };
        let row = 2 * (physical - self.spec.core.core_start) + offset;
        let query = self.weights.row(
            &format!("model.llm.middle_cycle_{kind}"),
            row,
            self.hidden(),
        )?;
        let message = bank.stream.mix(bank.first_block, &query);
        let ctrl = format!("{layer}.controllers.{}", role.name());

        let mut gate = input.to_vec();
        gate.extend_from_slice(&message);
        for half in gate.chunks_exact_mut(self.hidden()) {
            rms_norm(half, MIX_EPS);
        }
        let [gain, forget] = self.weights.controller(
            [
                &format!("{ctrl}.down_weight"),
                &format!("{ctrl}.gain_weight"),
                &format!("{ctrl}.forget_weight"),
            ],
            &gate,
            [self.spec.core.gate_rank as usize, self.hidden()],
        )?;
        for (((value, message), gain), forget) in
            input.iter_mut().zip(message).zip(gain).zip(forget)
        {
            *value = (*value - forget.tanh() * *value) + gain.tanh() * message;
        }
        Ok(())
    }

    fn moe_route(&self, layer: usize, prefix: &str, input: &[f32]) -> Result<MoeRoute, EngError> {
        let experts = self.spec.n_routed_experts as usize;
        let scores = self
            .weights
            .project(&format!("{prefix}.gate.weight"), input, experts)?
            .into_iter()
            .map(sigmoid)
            .collect::<Vec<_>>();
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        let selected = route(&scores, &bias, self.spec.num_experts_per_tok as usize)?;
        let total = selected
            .iter()
            .map(|&id| scores[id])
            .sum::<f32>()
            .max(1.0 / 16384.0);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Weights::Gguf(weights) = &self.weights {
            weights.observe_experts(layer, &selected);
        }
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let _ = layer;
        Ok(MoeRoute {
            selected,
            scores,
            total,
        })
    }

    fn moe(&self, layer: usize, prefix: &str, input: &[f32]) -> Result<Vec<f32>, EngError> {
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
        let gate_up = format!("{prefix}.experts.w13_weight");
        let down = format!("{prefix}.experts.w2_weight");
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Weights::Gguf(weights) = &self.weights
            && let Some(output) = weights.resident_experts(
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
        let int = |key| {
            meta.integer(&format!("inkling2.{key}"))
                .map(|value| value as usize)
        };
        let float = |key| meta.float(&format!("inkling2.{key}"));
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

impl Stream {
    fn mix(&self, first_block: usize, query: &[f32]) -> Vec<f32> {
        let states = self.blocks[first_block..]
            .iter()
            .rev()
            .chain(std::iter::once(&self.current));
        let scale = (query.len() as f32).sqrt().recip();
        let rows = states
            .map(|state| {
                let mut norm = state.clone();
                rms_norm(&mut norm, MIX_EPS);
                (dot(&norm, query) * scale, norm)
            })
            .collect::<Vec<_>>();
        let peak = rows
            .iter()
            .map(|(score, _)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut mass = 0.0;
        let mut result = vec![0.0; query.len()];
        for (score, state) in rows {
            let weight = (score - peak).exp();
            mass += weight;
            add(&mut result, &state, weight);
        }
        for value in &mut result {
            *value /= mass;
        }
        result
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
    let scale = (dot(x, x) / x.len() as f32 + eps).sqrt().recip();
    for value in x {
        *value *= scale;
    }
}

pub(super) fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).recip()
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
fn depth_mix_can_exclude_the_first_core_boundary() {
    let stream = Stream {
        blocks: vec![vec![0.0, 2.0]],
        current: vec![2.0, 0.0],
    };
    let both = stream.mix(0, &[0.0; 2]);
    let current = stream.mix(1, &[0.0; 2]);
    assert!((both[0] - both[1]).abs() < 1e-6);
    assert!((current[0] - std::f32::consts::SQRT_2).abs() < 1e-6);
    assert_eq!(current[1], 0.0);
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
