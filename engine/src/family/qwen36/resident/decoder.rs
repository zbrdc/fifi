//! The resident Q36 decoder (packet Q36-D): weights, session budget, and the step entry point.
//!
//! `ResidentDecoder` runs one token per `step` in a `StepMode`. It owns the adapter and the weights.
//! It holds no session: the caller owns each `ResidentState` and passes it to `step`.

use super::super::settings::{Activation, KvDtype, Q8_GROUP};
use super::moe::Routing;
use super::ops::{Ctx, cast_to, f32_tensor, short_import};
use super::prefill;
use super::state::{ResidentSnapshot, ResidentState};
use super::step;
use super::weights::{RMixer, ResidentWeights};
#[cfg(test)]
use crate::tensor::ExecutionTensor;
use crate::{
    EngError,
    family::qwen36::host_load::HostWeights,
    kernels::MlxKernels,
    sched::SessionBudget,
    tensor::{Dtype, Tensor},
    weights::StreamedCheckpoint,
};

/// The test-only graph probe of packet GPU-GAP-COUNT: it runs after the token graph is built
/// and before the step's single `evaluate`, with the kernels handle and the lazy outputs.
#[cfg(test)]
type GraphProbe = Box<dyn FnMut(&mut MlxKernels, &[&ExecutionTensor]) -> Result<(), EngError>>;

/// What one step computes and reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StepMode {
    /// Greedy decoding: the head runs and the token id is read back.
    Generate,
    /// A prompt token whose logits are unused: no head, no readback after `evaluate`.
    Prompt,
    /// `Generate` plus the logits, for a sampler.
    Logits,
    /// Parity runs only: `Logits` plus the residual stream after every layer.
    Record,
}

/// Host results of one step.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StepResult {
    /// The greedy next token, lowest index on ties. `None` in `Prompt`.
    pub(super) token: Option<u32>,
    /// Next-token logits `[vocab]`. `Logits` and `Record` only.
    pub(super) logits: Option<Vec<f32>>,
    /// The `[hidden]` residual stream after each layer, as the host reference records it. `Record` only.
    pub(super) layer_out: Option<Vec<Vec<f32>>>,
    /// The experts of each layer in rank order, `top_k` per layer. `Record` only.
    pub(super) routes: Option<Vec<Vec<u32>>>,
    /// The router probability row `[experts]` behind each recorded host route, the oracle of the
    /// near-tie rule of the route-parity gate (fix3 of packet Q36-BF16-a; decision 0046 A6).
    /// `Record` with `Routing::Host` only: the row is already on the host there, so recording it
    /// adds no synchronization.
    pub(super) probs: Option<Vec<Vec<f32>>>,
}

/// The default context limit of decision 0046 item 6.
pub(super) const DEFAULT_CONTEXT_LIMIT: u32 = 32_768;

/// The retire cap of the decoder's adapter: 4 GiB. With device routing a token builds one graph
/// and every dropped temporary stays retired until the token's `evaluate`. The default cap of
/// 64 MiB would force a flush synchronization inside the token (decision 0039 retire list). The
/// retired arrays are inputs of the pending graph, so the flush would free nothing.
pub(super) const DECODE_RETIRE_CAP: u64 = 4 << 30;

/// The default rows of one prefill chunk (plan item Q36-E: 256 to 512 first).
pub(super) const DEFAULT_CHUNK_ROWS: u32 = 512;

/// The most rows that `set_chunk_rows` accepts.
pub(super) const MAX_CHUNK_ROWS: u32 = 2_048;

/// One resident Q36 model on its engine thread.
///
/// Field order is drop order: the adapter drops first, then the weights and the pages under them.
pub(super) struct ResidentDecoder<'c> {
    pub(super) kernels: MlxKernels,
    pub(super) ckpt: &'c StreamedCheckpoint,
    pub(super) weights: ResidentWeights,
    /// The engine's session budget. Each step charges its cache growth here. The cap is `u64::MAX`
    /// until the executor sets the admitted value.
    pub(super) budget: SessionBudget,
    /// The first position that a step refuses with `ContextLimit`.
    pub(super) context_limit: u32,
    /// The most rows of one prefill chunk. `prefill` splits a longer prompt, and a chunk at
    /// a great depth is split again into smaller chunks (`kv_plan::depth_ranges`).
    pub(super) chunk_rows: u32,
    /// How the layers pick their experts. `Device` unless a parity run asks for `Host`.
    pub(super) routing: Routing,
    /// `mlx_compile` on the DeltaNet and device-routed MoE decode blocks (packet MX-COMPILE-1a):
    /// the `qwen36.compile` setting, false until `set_compile` turns it on. `step` copies it
    /// into its `Ctx`, `delta::step` and `moe::step_device` read it, and no other path compiles.
    pub(super) compile: bool,
    /// The seconds of this decoder's load-time compile warm-up, per block kind: empty until
    /// `set_compile(true)` traces the programs. `WarmupSeconds` documents the shape.
    pub(super) compile_warmup: WarmupSeconds,
    /// A test-only probe that runs after the token graph is built and before the step's single
    /// `evaluate` (packet GPU-GAP-COUNT): the graph dump of `decode_graph_primitives` reads the
    /// lazy outputs there without evaluating them. `None` unless a test sets it, so every other
    /// build and test runs the same step as before; the probe makes no native call of its own
    /// except through the kernels handle it receives.
    #[cfg(test)]
    pub(super) graph_probe: Option<GraphProbe>,
}

/// The seconds of the load-time compile warm-up, per block kind (packet MX-COMPILE-1a, fix
/// round 1): `delta` holds one entry per traced DeltaNet program and `moe` one entry per traced
/// device-routed MoE program, each covering the whole warm step (the trace call and its
/// evaluation). They are measured inside THIS decoder, so the start line never reports another
/// engine's seconds, and the two lists are labeled separately.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct WarmupSeconds {
    /// One entry per traced DeltaNet program, in plan order.
    pub(super) delta: Vec<f64>,
    /// One entry per traced device-routed MoE program, in plan order.
    pub(super) moe: Vec<f64>,
}

impl<'c> ResidentDecoder<'c> {
    /// Sets the retire cap to `DECODE_RETIRE_CAP`, imports the weights (see
    /// `ResidentWeights::import`, with F32 scales and biases), and consumes `host`. The routing is
    /// `Routing::Device`.
    pub(super) fn new(
        mut kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
    ) -> Result<Self, EngError> {
        kernels.set_retire_cap(DECODE_RETIRE_CAP)?;
        let weights = ResidentWeights::import(&mut kernels, ckpt, host)?;
        Ok(Self::with_weights(kernels, ckpt, weights, context_limit))
    }

    /// Builds the decoder for `activation` (packet Q36-BF16-a), the one setting that pairs the
    /// scale storage with the activation dtype. `Activation::F32` is the qualification
    /// control and equals `new`. `Activation::Bf16` keeps the stored BF16 scales and biases, then
    /// lets the adapter take BF16 operands, converts the 100 layer norms to BF16 with one
    /// synchronization, and sets `Consts::act`; the key-value cache, the DeltaNet state and ring,
    /// the router, the final norm, and the logits stay F32. `ensure_scale_pairing` refuses a
    /// mismatched scale pair before any of that. The chunked prefill runs the same activation
    /// dtype (packet Q36-BF16-b: the chunk blocks cast at their boundaries and guard their dtypes).
    ///
    /// The head keeps an F32 activation over BF16 scales (packet Q36-BF16-a, documented, not
    /// changed): the final norm promotes the BF16 hidden state against the F32 final-norm weight,
    /// so the `lm_head` qmm runs F32 over the stored BF16 scales, and MLX casts the scales per
    /// token (about 64 MB of temporaries). A speed-table row for the Mac run.
    pub(super) fn new_with_activation(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
        activation: Activation,
    ) -> Result<Self, EngError> {
        // The product path: the scale import follows the activation, so the census always matches.
        // The cache form is the F32 control (packet KV8-INTEGRATE: the shipped default stays F32
        // until the A7 rule-4 gate).
        Self::new_paired(
            kernels,
            ckpt,
            host,
            context_limit,
            activation,
            activation == Activation::F32,
            KvDtype::F32,
        )
    }

    /// Builds the decoder for the cache form `kv` (packet KV8-INTEGRATE; decision 0046 A7):
    /// `KvDtype::F32` is the control and equals `new_with_activation`, and `KvDtype::Q8` keeps
    /// the attention cache as 8-bit affine codes beside scales and biases of the activation
    /// dtype - the new K and V rows stay in the activation dtype and `quantize_rows` packs them,
    /// so BF16 activations give BF16 scales (1,088 bytes per layer and position on the real
    /// model) and F32 activations give F32 scales (1,152). A Q8 engine whose `head_dim` is not a
    /// multiple of the group (64) is refused at load with `BadRequest` naming the setting, before
    /// any weight imports.
    pub(super) fn new_with_kv(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
        activation: Activation,
        kv: KvDtype,
    ) -> Result<Self, EngError> {
        Self::new_paired(
            kernels,
            ckpt,
            host,
            context_limit,
            activation,
            activation == Activation::F32,
            kv,
        )
    }

    /// The test form of `new_with_activation` (packet Q36-BF16-a fix): the scale import follows the
    /// `Q36_SCALE_DTYPE` knob instead of `activation`, which is the only way to drive the
    /// mismatched pair that `ensure_scale_pairing` refuses, because the product constructor
    /// cannot build it.
    #[cfg(test)]
    pub(super) fn new_with_scale_override(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
        activation: Activation,
    ) -> Result<Self, EngError> {
        Self::new_paired(
            kernels,
            ckpt,
            host,
            context_limit,
            activation,
            super::scale_knob::scales_f32_requested(),
            KvDtype::F32,
        )
    }

    /// The one constructor body: it imports the scales in the `f32_scales` form, then refuses a
    /// pair that does not match `activation` before anything else changes. The cache form `kv`
    /// is checked against the head dimension BEFORE any import (a Q8 cache needs a multiple of
    /// the group of 64) and lands on the weights, where `zero_with` reads it for every cache.
    fn new_paired(
        mut kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
        activation: Activation,
        f32_scales: bool,
        kv: KvDtype,
    ) -> Result<Self, EngError> {
        let act = activation.dtype();
        if kv == KvDtype::Q8
            && !host
                .shape()
                .attn
                .head_dim
                .is_multiple_of(usize::try_from(Q8_GROUP).unwrap_or(0))
        {
            return Err(EngError::BadRequest(format!(
                "kv_dtype q8 needs a head_dim that is a multiple of {Q8_GROUP}, got {}",
                host.shape().attn.head_dim
            )));
        }
        kernels.set_retire_cap(DECODE_RETIRE_CAP)?;
        let mut weights = if f32_scales {
            ResidentWeights::import(&mut kernels, ckpt, host)?
        } else {
            ResidentWeights::import_bf16_scales(&mut kernels, ckpt, host)?
        };
        weights.ensure_scale_pairing(activation)?;
        if act == Dtype::Bf16 {
            kernels.allow_bf16(true);
            weights.convert_norms(&mut kernels, Dtype::Bf16)?;
            weights.consts.act = Dtype::Bf16;
        }
        weights.kv = kv;
        Ok(Self::with_weights(kernels, ckpt, weights, context_limit))
    }

    fn with_weights(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        weights: ResidentWeights,
        context_limit: u32,
    ) -> Self {
        Self {
            kernels,
            ckpt,
            weights,
            budget: SessionBudget::new(u64::MAX),
            context_limit,
            chunk_rows: DEFAULT_CHUNK_ROWS,
            routing: Routing::default(),
            compile: false,
            compile_warmup: WarmupSeconds::default(),
            #[cfg(test)]
            graph_probe: None,
        }
    }

    /// Sets the test-only graph probe (packet GPU-GAP-COUNT), the hook that sees the lazy
    /// outputs of one step before its `evaluate`. Test builds only; production code has no
    /// setter, so no product path can install a probe.
    #[cfg(test)]
    pub(super) fn set_graph_probe(&mut self, probe: GraphProbe) {
        self.graph_probe = Some(probe);
    }

    /// Selects how the next steps route. The parity gate runs `Host` as the reference. The choice
    /// applies to every state of this decoder.
    pub(super) fn set_routing(&mut self, routing: Routing) {
        self.routing = routing;
    }

    /// Turns the compiled decode blocks on or off (packet MX-COMPILE-1a item 6, fix round 1).
    /// Turning them on admits the WHOLE model first — every DeltaNet layer and every
    /// device-routed MoE layer's weight facts over the warm-up plan — and refuses the load with
    /// `EngError::Backend { operation: "compile_warm_up", .. }` when the plan refuses (a packed
    /// weight without biases, more than four distinct programs), so no compiled call can fail in
    /// the middle of a request. The warm-up then traces ONE layer per distinct cache key on zero
    /// tensors, evaluates the traced results, and records the per-block seconds in
    /// `compile_warmup`, so the first token of the first request replays a compiled program
    /// instead of tracing one. A trace failure is a native `Backend` HERE, at load, and the
    /// adapter is terminal afterwards. Turning the setting off runs the warm-up never and
    /// changes nothing else; the executor calls this once, right after construction, before the
    /// first state exists. A Host-routed decoder warms its DeltaNet blocks and not its MoE
    /// blocks (the host-routed MoE never compiles).
    pub(super) fn set_compile(&mut self, on: bool) -> Result<(), EngError> {
        self.compile = on;
        if on {
            self.warm_compile()?;
        }
        Ok(())
    }

    /// The load-time compile warm-up: the plan over ALL layers decides which programs the model
    /// needs (nothing when the setting is off), and each needed program is traced once on zero
    /// tensors — the layer input, the recurrent state, and the convolution ring of the first
    /// DeltaNet layer of that program — through the compiled seam, then evaluated so the trace
    /// and its first evaluation happen at load. The outputs are dropped; no state of any request
    /// sees them.
    fn warm_compile(&mut self) -> Result<(), EngError> {
        // The facts of every layer, then the admission: a model the seam cannot compile refuses
        // the load HERE, before any trace.
        let act = self.weights.consts.act;
        let shape = self.weights.shape.clone();
        let eps = shape.eps;
        let delta_facts: Vec<super::compile_plan::DeltaLayerFacts> = self
            .weights
            .layers
            .iter()
            .enumerate()
            .filter_map(|(layer, current)| match &current.mixer {
                RMixer::Delta(w) => Some(super::compiled::delta_facts(layer, &shape, w, act, eps)),
                RMixer::Attn(_) => None,
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let moe_facts: Vec<super::compile_plan::MoeLayerFacts> = self
            .weights
            .layers
            .iter()
            .enumerate()
            .map(|(layer, current)| super::compiled::moe_facts(layer, &shape, &current.moe, act))
            .collect::<Result<Vec<_>, EngError>>()?;
        let plan = super::compile_plan::model_warmup(
            self.compile,
            self.routing == Routing::Device,
            &delta_facts,
            &moe_facts,
        )
        .map_err(|error| match error {
            EngError::BadRequest(message) => EngError::Backend {
                operation: "compile_warm_up",
                message,
            },
            other => other,
        })?;
        if plan.delta.is_empty() && plan.moe.is_empty() {
            return Ok(());
        }
        let mut ctx = Ctx {
            kernels: &mut self.kernels,
            consts: &self.weights.consts,
            compile: true,
        };
        // The zero input of the layer, imported with one synchronization; it converts to the
        // activation dtype of the decoder. The zero DeltaNet state and ring follow the checked
        // dimensions of the plan (LOW-2: no unchecked subtraction or product).
        let u32_of = |n: usize| {
            u32::try_from(n).map_err(|_| {
                EngError::BadRequest("the compile warm-up dimension exceeds u32".into())
            })
        };
        let count_of = |name: &'static str, dims: &[u32]| {
            let product = dims
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)));
            let product = product.ok_or_else(|| {
                EngError::BadRequest(format!("the compile warm-up {name} exceeds u64"))
            })?;
            usize::try_from(product).map_err(|_| {
                EngError::BadRequest(format!("the compile warm-up {name} exceeds usize"))
            })
        };
        let hosts_x = f32_tensor(vec![1, u32_of(shape.hidden)?], &vec![0.0; shape.hidden])?;
        let mut hosts = vec![hosts_x];
        if !plan.delta.is_empty() {
            let dims = delta_facts
                .first()
                .map(|facts| facts.dims)
                .ok_or_else(|| EngError::BadRequest("no DeltaNet layer".into()))?;
            let [_, kd, vh, vd, cd, taps, _, _] = dims;
            hosts.push(f32_tensor(
                vec![vh, vd, kd],
                &vec![0.0; count_of("state", &[vh, vd, kd])?],
            )?);
            hosts.push(f32_tensor(
                vec![cd, taps],
                &vec![0.0; count_of("ring", &[cd, taps])?],
            )?);
        }
        let refs: Vec<&Tensor> = hosts.iter().collect();
        let mut imported = ctx.kernels.import_many(&refs)?.into_iter();
        let mut next = || imported.next().ok_or_else(short_import);
        let x_zero = next()?;
        let act = ctx.consts.act;
        let x = cast_to(&mut ctx, &x_zero, act)?;
        let (mut delta_seconds, mut moe_seconds) = (Vec::new(), Vec::new());
        if !plan.delta.is_empty() {
            let rec = next()?;
            let ring = next()?;
            for (layer, _id) in &plan.delta {
                let RMixer::Delta(w) = &self
                    .weights
                    .layers
                    .get(*layer)
                    .ok_or_else(|| {
                        EngError::BadRequest("the compile warm-up layer is missing".into())
                    })?
                    .mixer
                else {
                    return Err(EngError::BadRequest(
                        "the compile warm-up found no DeltaNet layer".into(),
                    ));
                };
                let started = std::time::Instant::now();
                let got = super::compiled::delta_step(&mut ctx, &shape, w, &rec, &ring, &x)?;
                ctx.kernels.evaluate(&[&got.out, &got.rec, &got.ring])?;
                delta_seconds.push(started.elapsed().as_secs_f64());
            }
        }
        for (layer, _id) in &plan.moe {
            let moe = &self
                .weights
                .layers
                .get(*layer)
                .ok_or_else(|| EngError::BadRequest("the compile warm-up layer is missing".into()))?
                .moe;
            let started = std::time::Instant::now();
            let got = super::compiled::moe_step_device(&mut ctx, &shape, moe, &x)?;
            ctx.kernels
                .evaluate(&[&got.0, got.1.ids.ids(), &got.1.gates, &got.1.health])?;
            moe_seconds.push(started.elapsed().as_secs_f64());
        }
        self.compile_warmup = WarmupSeconds {
            delta: delta_seconds,
            moe: moe_seconds,
        };
        Ok(())
    }

    /// Replaces the session budget. Tests use small caps so that a step can be denied. Call it before
    /// any state exists: permits of earlier steps charge the old budget.
    #[cfg(test)]
    pub(super) fn set_session_budget(&mut self, budget: SessionBudget) {
        self.budget = budget;
    }

    /// Bytes charged to the session budget by the committed steps.
    #[cfg(test)]
    pub(super) fn session_used(&self) -> u64 {
        self.budget.used()
    }

    /// Sets the most rows of one prefill chunk. A value of zero or above `MAX_CHUNK_ROWS` returns
    /// `BadRequest` and changes nothing.
    pub(super) fn set_chunk_rows(&mut self, rows: u32) -> Result<(), EngError> {
        if rows == 0 || rows > MAX_CHUNK_ROWS {
            return Err(EngError::BadRequest(format!(
                "a prefill chunk holds between 1 and {MAX_CHUNK_ROWS} rows"
            )));
        }
        self.chunk_rows = rows;
        Ok(())
    }

    /// Imports a zero state with one synchronization.
    pub(super) fn zero_state(&mut self) -> Result<ResidentState, EngError> {
        ResidentState::zero(&mut self.kernels, &self.weights)
    }

    /// Replaces `state` with a zero state. On error `state` is unchanged.
    pub(super) fn reset(&mut self, state: &mut ResidentState) -> Result<(), EngError> {
        state.reset(&mut self.kernels, &self.weights)
    }

    /// Saves `state` as a snapshot that shares its arrays and charges the session budget. See
    /// `ResidentState::snapshot`.
    pub(super) fn snapshot(&mut self, state: &ResidentState) -> Result<ResidentSnapshot, EngError> {
        state.snapshot(&mut self.budget)
    }

    /// Replaces `state` with `snapshot`. See `ResidentState::restore`.
    pub(super) fn restore(
        &mut self,
        state: &mut ResidentState,
        snapshot: &ResidentSnapshot,
    ) -> Result<(), EngError> {
        state.restore(snapshot, &self.weights, &mut self.budget)
    }

    /// Runs one token. See `step::step`.
    pub(super) fn step(
        &mut self,
        state: &mut ResidentState,
        token: u32,
        mode: StepMode,
    ) -> Result<StepResult, EngError> {
        step::step(self, state, token, mode)
    }

    /// Runs a prompt in chunks. See `prefill::step::prefill`.
    ///
    /// The chunks run in `Prompt` mode except the last, which runs in `mode`. The result is that of
    /// the last chunk: the head reads the last row of the prompt.
    ///
    /// A prompt of one token runs as a chunk of one row, not as `step`. A chunk routes as this
    /// decoder holds (`Routing::Host` or `Routing::Device`, packet Q36-PF-DEVROUTE), so the result
    /// has `routes: None` in every mode. `step` returns `Some(routes)` in `Record` mode.
    ///
    /// The host checks of the whole prompt (a poisoned state, an empty prompt, and the context
    /// limit) run first and change nothing on error. Each chunk repeats the host checks and takes
    /// its growth permit before it marks the state poisoned. A failed check or a denied permit
    /// (`MemoryPlan`) in a later chunk leaves the state clean: the chunks before it stay
    /// published, `state.pos()` is the position after the last of them, and the session budget
    /// holds their charge. A caller that retries must continue from `state.pos()` or `reset` the
    /// state. An error after the state is marked poisoned (a native failure, or a token outside
    /// the vocabulary, which the read of the embedding rows finds) poisons the state, drops its
    /// layer states, and releases the permit of that chunk. Only `reset` recovers a poisoned state.
    pub(super) fn prefill(
        &mut self,
        state: &mut ResidentState,
        tokens: &[u32],
        mode: StepMode,
    ) -> Result<StepResult, EngError> {
        prefill::step::prefill(self, state, tokens, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;

    /// The host load of the four-layer tiny model.
    fn host(ckpt: &StreamedCheckpoint) -> HostWeights {
        HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap()
    }

    /// The BF16 constructor of packet Q36-BF16-a keeps the stored BF16 scales and biases, converts
    /// the 100-shaped norm set (here 8 layer norms and 4 head norms) to BF16, and holds `Bf16` as
    /// the activation dtype. The F32 constructor widens every scale and bias and keeps the norms
    /// F32. The two censuses have the same shape (43 packed weights of 2 parts each), so a wiring
    /// that imports F32 scales for BF16 activations shows here as an F32 part or a construction
    /// refusal from `ensure_scale_pairing`.
    #[test]
    fn the_activation_dtype_pairs_the_scales_and_converts_the_norms() {
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let f32 =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, host(&ckpt), 1024).unwrap();
        let bf16 = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host(&ckpt),
            1024,
            Activation::Bf16,
        )
        .unwrap();
        assert_eq!(f32.weights.consts.act, Dtype::F32, "the F32 control");
        assert_eq!(
            bf16.weights.consts.act,
            Dtype::Bf16,
            "the activation dtype is set"
        );
        let (f32, bf16) = (f32.weights.scale_census(), bf16.weights.scale_census());
        assert_eq!(
            (f32.weights, f32.f32_parts, f32.bf16_parts, f32.other_parts),
            (43, 86, 0, 0),
            "the F32 import widens every scale and bias"
        );
        assert_eq!(
            (
                bf16.weights,
                bf16.f32_parts,
                bf16.bf16_parts,
                bf16.other_parts
            ),
            (43, 0, 86, 0),
            "the BF16 import keeps every scale and bias as stored"
        );
    }

    /// The norms of the BF16 decoder are BF16 (they are stored BF16 in the checkpoint, so the
    /// conversion loses nothing) and the final norm stays F32. A missing `convert_norms` leaves
    /// the input norms F32 and the dtype guard of the step fires.
    #[test]
    fn the_bf16_constructor_converts_the_norms_and_keeps_the_final_norm_f32() {
        let _serial = crate::native_test_lock();
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host(&ckpt),
            1024,
            Activation::Bf16,
        )
        .unwrap();
        assert_eq!(
            decoder.weights.final_norm.dtype(),
            Dtype::F32,
            "the final norm stays F32"
        );
        let (mut layer_norms, mut head_norms) = (0usize, 0usize);
        for layer in &decoder.weights.layers {
            assert_eq!(layer.input_norm.dtype(), Dtype::Bf16, "an input norm");
            assert_eq!(layer.post_norm.dtype(), Dtype::Bf16, "a post norm");
            layer_norms += 2;
            if let super::super::weights::RMixer::Attn(a) = &layer.mixer {
                assert_eq!(a.q_norm.dtype(), Dtype::Bf16, "a query norm");
                assert_eq!(a.k_norm.dtype(), Dtype::Bf16, "a key norm");
                head_norms += 2;
            }
        }
        assert_eq!(
            (layer_norms, head_norms),
            (8, 4),
            "every norm of the fixture"
        );
    }

    /// The pairing refusal of `ensure_scale_pairing`, driven through the `Q36_SCALE_DTYPE` test
    /// knob (packet Q36-BF16-a fix): the knob picks the scale import independently of the
    /// activation, which is the only way to build the mismatched pair, because the product
    /// constructor cannot. `std::env::set_var` is denied outside the kernels modules, so each
    /// direction runs in a child process that carries the knob, as the rebind tests of the
    /// kernels adapter do; the child asserts the refusal text and the parent requires its pass.
    #[test]
    fn the_constructor_refuses_the_mismatched_scale_pair_from_the_knob() {
        const CHILD: &str = "INFERENCE_Q36_PAIRING_CHILD";
        // One mismatch direction: the knob value and the activation it must refuse.
        struct Mismatch {
            knob: (&'static str, &'static str),
            activation: Activation,
            want: &'static str,
        }
        let rows = [
            Mismatch {
                // BF16 scales under an `f32` activation: the 31.6 tok/s trap that Q36-G0 closed.
                knob: ("Q36_SCALE_DTYPE", "bf16"),
                activation: Activation::F32,
                want: "activation_dtype f32 needs F32 scales and biases; the census holds \
                       43 packed weights with 0 F32, 86 BF16, and 0 other parts",
            },
            Mismatch {
                // F32 scales under a `bf16` activation: every qmm would promote back to F32.
                knob: ("Q36_SCALE_DTYPE", "f32"),
                activation: Activation::Bf16,
                want: "activation_dtype bf16 needs BF16 scales and biases; the census holds \
                       43 packed weights with 86 F32, 0 BF16, and 0 other parts",
            },
        ];
        if let Some(which) = std::env::var_os(CHILD) {
            // The child: its knob is already in the environment; build and refuse.
            let row = match which.to_str() {
                Some("f32") => &rows[1],
                _ => &rows[0],
            };
            let ckpt = tiny::checkpoint(&TINY_GQA);
            let refused = ResidentDecoder::new_with_scale_override(
                MlxKernels::try_new().unwrap(),
                &ckpt,
                host(&ckpt),
                1024,
                row.activation,
            );
            assert!(
                matches!(&refused, Err(EngError::BadRequest(m)) if m == row.want),
                "the mismatched pair must refuse with the census, got {:?}",
                refused.err()
            );
            return;
        }
        let _serial = crate::native_test_lock();
        let path = module_path!().split_once("::").unwrap().1;
        for row in &rows {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    &format!(
                        "{path}::the_constructor_refuses_the_mismatched_scale_pair_from_the_knob"
                    ),
                    "--nocapture",
                ])
                .env(CHILD, row.knob.1)
                .env(row.knob.0, row.knob.1)
                .output()
                .unwrap();
            let captured = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                output.status.success() && captured.contains("1 passed"),
                "the {} direction must refuse in the child: {captured}",
                row.knob.1
            );
        }
    }
}
