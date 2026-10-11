//! Flash-Next decode step through all layers (design section "Decode step").

use super::deltanet::{self, DeltaShape, DeltaState, DeltaWeights};
use super::hyper::{self, HyperShape, HyperWeights};
use super::math::expect_len;
use super::moe::{self, MoeShape, MoeWeights};
use super::ple::{self, PleShape, PleState, PleWeights};
use super::qsa::{self, QsaCache, QsaShape, QsaWeights};
use crate::EngError;
use crate::family::project::Projector;

/// Checked shapes of every block and the model-level layout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelShape {
    /// Hyper-connection shape shared by every read and apply.
    pub(crate) hyper: HyperShape,
    /// DeltaNet layer shape.
    pub(crate) delta: DeltaShape,
    /// QSA layer shape.
    pub(crate) qsa: QsaShape,
    /// MoE shape.
    pub(crate) moe: MoeShape,
    /// PLE shape.
    pub(crate) ple: PleShape,
    /// Layer index whose attention read follows the PLE update.
    pub(crate) ple_layer: usize,
    /// Vocabulary size, the `lm_head` output width.
    pub(crate) vocab: usize,
}

/// Attention weights of one layer.
pub(crate) enum AttnWeights<'a, W> {
    /// A DeltaNet layer.
    Delta(DeltaWeights<'a, W>),
    /// A QSA layer.
    Qsa(QsaWeights<'a, W>),
}

/// Weights of one decoder layer.
pub(crate) struct FlashLayerWeights<'a, W> {
    /// Hyper-connection before the attention block.
    pub(crate) attn_hyper: HyperWeights<'a, W>,
    /// Attention block.
    pub(crate) attn: AttnWeights<'a, W>,
    /// Hyper-connection before the MoE block.
    pub(crate) mlp_hyper: HyperWeights<'a, W>,
    /// MoE block.
    pub(crate) moe: MoeWeights<'a, W>,
}

/// Weights of the whole model.
pub(crate) struct ModelWeights<'a, W> {
    /// Embedding table `[vocab, hidden]`.
    pub(crate) embed: &'a W,
    /// Decoder layers in order.
    pub(crate) layers: Vec<FlashLayerWeights<'a, W>>,
    /// PLE block.
    pub(crate) ple: PleWeights<'a, W>,
    /// Root mixer, without inject.
    pub(crate) mixer: HyperWeights<'a, W>,
    /// Output head `[vocab, hidden]`.
    pub(crate) lm_head: &'a W,
}

/// Mutable state of one layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayerState {
    /// DeltaNet recurrent state.
    Delta(DeltaState),
    /// QSA cache.
    Qsa(QsaCache),
}

/// Mutable state of one sequence.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelState {
    layers: Vec<LayerState>,
    ple: PleState,
    pos: u32,
    poisoned: bool,
}

impl ModelState {
    /// Returns a fresh state whose layer kinds follow `weights`.
    pub(crate) fn new<W>(shape: &ModelShape, weights: &ModelWeights<'_, W>) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|layer| match &layer.attn {
                AttnWeights::Delta(_) => LayerState::Delta(DeltaState::new(&shape.delta)),
                AttnWeights::Qsa(_) => LayerState::Qsa(QsaCache::new()),
            })
            .collect();
        Self {
            layers,
            ple: PleState::new(&shape.ple),
            pos: 0,
            poisoned: false,
        }
    }

    /// Position of the next token.
    pub(crate) fn pos(&self) -> u32 {
        self.pos
    }

    /// Replaces this state with a fresh state.
    pub(crate) fn reset<W>(&mut self, shape: &ModelShape, weights: &ModelWeights<'_, W>) {
        *self = Self::new(shape, weights);
    }

    /// Host bytes held by all layer states and the PLE state, or an error when the sum overflows `usize`.
    pub(crate) fn byte_len(&self) -> Result<usize, EngError> {
        let overflow = || EngError::MemoryPlan("model state byte count overflows usize".into());
        let mut total = 0_usize;
        for layer in &self.layers {
            let bytes = match layer {
                LayerState::Delta(state) => state.byte_len(),
                LayerState::Qsa(cache) => cache.byte_len(),
            };
            total = total.checked_add(bytes).ok_or_else(overflow)?;
        }
        total.checked_add(self.ple.byte_len()).ok_or_else(overflow)
    }
}

#[cfg(test)]
impl ModelState {
    /// Layer states in model order (test observer).
    pub(crate) fn layers(&self) -> &[LayerState] {
        &self.layers
    }
}

/// Result of one decode step.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StepOutput {
    /// Next-token logits, length `vocab`.
    pub(crate) logits: Vec<f32>,
    /// Post-MoE residual of every layer, when recording was requested.
    pub(crate) layer_out: Option<Vec<Vec<f32>>>,
}

/// Runs one token through the model and returns its logits.
pub(crate) fn step<P: Projector>(
    projector: &mut P,
    shape: &ModelShape,
    weights: &ModelWeights<'_, P::Weight>,
    state: &mut ModelState,
    token: u32,
    record: bool,
) -> Result<StepOutput, EngError> {
    if state.poisoned {
        return Err(EngError::BadRequest(
            "model state needs a reset after a failed step".into(),
        ));
    }
    if state.pos == u32::MAX {
        return Err(EngError::ContextLimit("position limit reached".into()));
    }
    if weights.layers.len() != state.layers.len() || shape.ple_layer >= weights.layers.len() {
        return Err(EngError::BadRequest(
            "model layer count does not match the state or the PLE layer".into(),
        ));
    }
    match run(projector, shape, weights, state, token, record) {
        Ok(output) => Ok(output),
        Err(err) => {
            state.poisoned = true;
            Err(err)
        }
    }
}

/// Runs the layers for one token. The caller poisons the state when this fails.
fn run<P: Projector>(
    projector: &mut P,
    shape: &ModelShape,
    weights: &ModelWeights<'_, P::Weight>,
    state: &mut ModelState,
    token: u32,
    record: bool,
) -> Result<StepOutput, EngError> {
    let pos = state.pos;
    let hidden = shape.hyper.hidden();
    let branches = shape.hyper.branches();

    let row = projector.rows(weights.embed, &[token])?;
    expect_len(&row, hidden, "embedding row")?;
    let mut hyper_state: Vec<f32> = Vec::with_capacity(shape.hyper.width());
    for _ in 0..branches {
        hyper_state.extend_from_slice(&row);
    }

    let mut recorded = Vec::new();
    for index in 0..weights.layers.len() {
        let layer = weights
            .layers
            .get(index)
            .ok_or_else(|| EngError::BadRequest("layer weights are missing".into()))?;
        let layer_state = state
            .layers
            .get_mut(index)
            .ok_or_else(|| EngError::BadRequest("layer state is missing".into()))?;

        if index == shape.ple_layer {
            let update = ple::step(
                projector,
                &shape.ple,
                &weights.ple,
                &mut state.ple,
                token,
                &hyper_state,
            )?;
            expect_len(&update, hyper_state.len(), "ple update")?;
            for (value, delta) in hyper_state.iter_mut().zip(&update) {
                *value += delta;
            }
        }

        let read = hyper::read(projector, &shape.hyper, &layer.attn_hyper, &hyper_state)?;
        let gate = read.gate.ok_or_else(|| {
            EngError::BadRequest("attention hyper-connection needs inject".into())
        })?;
        let block = match (&layer.attn, layer_state) {
            (AttnWeights::Delta(w), LayerState::Delta(s)) => {
                deltanet::step(projector, &shape.delta, w, s, &read.mixed)?
            }
            (AttnWeights::Qsa(w), LayerState::Qsa(c)) => {
                qsa::step(projector, &shape.qsa, w, c, pos, &read.mixed)?
            }
            _ => {
                return Err(EngError::BadRequest(
                    "layer state kind does not match its weights".into(),
                ));
            }
        };
        hyper::apply(&shape.hyper, &mut hyper_state, &gate, &block)?;

        let read = hyper::read(projector, &shape.hyper, &layer.mlp_hyper, &hyper_state)?;
        let gate = read
            .gate
            .ok_or_else(|| EngError::BadRequest("mlp hyper-connection needs inject".into()))?;
        let out = moe::step(projector, &shape.moe, &layer.moe, &read.mixed)?;
        hyper::apply(&shape.hyper, &mut hyper_state, &gate, &out)?;

        if record {
            recorded.push(hyper_state.clone());
        }
    }

    let mixed = hyper::read(projector, &shape.hyper, &weights.mixer, &hyper_state)?.mixed;
    let logits = projector.linear(weights.lm_head, &mixed)?;
    expect_len(&logits, shape.vocab, "logits")?;
    state.pos = pos
        .checked_add(1)
        .ok_or_else(|| EngError::ContextLimit("position overflows u32".into()))?;
    Ok(StepOutput {
        logits,
        layer_out: record.then_some(recorded),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::flash_next::ple::PleDims;
    use crate::family::flash_next::qsa::QsaDims;
    use crate::family::project::{DenseProjector, DenseWeight};

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length of {actual:?}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(close(*a, *e), "element {i}: actual {a}, expected {e}");
        }
    }

    fn z(r: usize, c: usize) -> DenseWeight {
        DenseWeight::matrix(r, c, vec![0.0; r * c])
    }

    fn zs(d: usize, r: usize, c: usize) -> DenseWeight {
        DenseWeight::stack(d, r, c, vec![0.0; d * r * c])
    }

    fn good_head() -> DenseWeight {
        DenseWeight::matrix(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    }

    struct Fixture {
        embed: DenseWeight,
        hyper_norm: [f32; 4],
        hyper_down: DenseWeight,
        hyper_up: DenseWeight,
        hyper_inject: DenseWeight,
        delta_qkv: DenseWeight,
        delta_z: DenseWeight,
        delta_b: DenseWeight,
        delta_a: DenseWeight,
        delta_out: DenseWeight,
        delta_conv: [f32; 6],
        delta_a_log: [f32; 1],
        delta_dt_bias: [f32; 1],
        delta_norm: [f32; 1],
        qsa_q: DenseWeight,
        qsa_k: DenseWeight,
        qsa_v: DenseWeight,
        qsa_o: DenseWeight,
        qsa_idx: DenseWeight,
        qsa_q_norm: [f32; 2],
        qsa_k_norm: [f32; 2],
        qsa_idx_q_norm: [f32; 2],
        qsa_idx_k_norm: [f32; 2],
        moe_router: [f32; 4],
        moe_gate: DenseWeight,
        moe_up: DenseWeight,
        moe_down: DenseWeight,
        moe_shared_gate: DenseWeight,
        moe_shared_up: DenseWeight,
        moe_shared_down: DenseWeight,
        moe_shared_scalar: DenseWeight,
        ple_multipliers: [i64; 3],
        ple_vocab: [i64; 2],
        ple_offsets: [i64; 2],
        ple_shards: Vec<DenseWeight>,
        ple_shard_rows: [u64; 1],
        ple_key: DenseWeight,
        ple_value: DenseWeight,
        ple_norm_key: [f32; 4],
        ple_norm_query: [f32; 4],
        ple_norm_conv: [f32; 4],
        ple_conv: [f32; 8],
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                embed: DenseWeight::matrix(3, 2, vec![3.0, 4.0, 1.0, 0.0, 0.0, 2.0]),
                hyper_norm: [0.0; 4],
                hyper_down: z(1, 4),
                hyper_up: z(4, 1),
                hyper_inject: z(2, 4),
                delta_qkv: z(3, 2),
                delta_z: z(1, 2),
                delta_b: z(1, 2),
                delta_a: z(1, 2),
                delta_out: z(2, 1),
                delta_conv: [0.0; 6],
                delta_a_log: [0.0],
                delta_dt_bias: [0.0],
                delta_norm: [1.0],
                qsa_q: z(4, 2),
                qsa_k: z(2, 2),
                qsa_v: z(2, 2),
                qsa_o: z(2, 2),
                qsa_idx: z(4, 2),
                qsa_q_norm: [0.0; 2],
                qsa_k_norm: [0.0; 2],
                qsa_idx_q_norm: [0.0; 2],
                qsa_idx_k_norm: [0.0; 2],
                moe_router: [0.0; 4],
                moe_gate: zs(2, 1, 2),
                moe_up: zs(2, 1, 2),
                moe_down: zs(2, 2, 1),
                moe_shared_gate: z(1, 2),
                moe_shared_up: z(1, 2),
                moe_shared_down: z(2, 1),
                moe_shared_scalar: z(1, 2),
                ple_multipliers: [1, 1, 1],
                ple_vocab: [5, 5],
                ple_offsets: [0, 5],
                ple_shards: vec![DenseWeight::matrix(
                    10,
                    1,
                    (0..10).map(|v| v as f32).collect(),
                )],
                ple_shard_rows: [10],
                ple_key: z(4, 2),
                ple_value: z(2, 2),
                ple_norm_key: [0.0; 4],
                ple_norm_query: [0.0; 4],
                ple_norm_conv: [0.0; 4],
                ple_conv: [0.0; 8],
            }
        }

        fn shape(ple_layer: usize) -> ModelShape {
            ModelShape {
                hyper: HyperShape::new(2, 2, 1, 1e-6).unwrap(),
                delta: DeltaShape::new(1, 1, 1, 1, 2, 2, 1e-6).unwrap(),
                qsa: QsaShape::new(QsaDims {
                    heads_q: 1,
                    heads_kv: 1,
                    head_dim: 2,
                    idx_heads: 1,
                    idx_dim: 2,
                    ratio: 1,
                    budget: 1,
                    rotary_dim: 2,
                    theta: 10_000.0,
                    hidden: 2,
                    eps: 1e-6,
                })
                .unwrap(),
                moe: MoeShape::new(2, 1, 1, 1, 2, true).unwrap(),
                ple: PleShape::new(PleDims {
                    branches: 2,
                    hidden: 2,
                    ngram: 3,
                    heads_per_ngram: 1,
                    head_dim: 1,
                    conv_kernel: 2,
                    eos: 0,
                    eps: 1e-6,
                })
                .unwrap(),
                ple_layer,
                vocab: 3,
            }
        }

        fn hyper(&self, inject: bool) -> HyperWeights<'_, DenseWeight> {
            HyperWeights {
                norm: &self.hyper_norm,
                down: &self.hyper_down,
                up: &self.hyper_up,
                inject: inject.then_some(&self.hyper_inject),
            }
        }

        fn moe(&self) -> MoeWeights<'_, DenseWeight> {
            MoeWeights {
                router: &self.moe_router,
                gate: &self.moe_gate,
                up: &self.moe_up,
                down: &self.moe_down,
                shared_gate: &self.moe_shared_gate,
                shared_up: &self.moe_shared_up,
                shared_down: &self.moe_shared_down,
                shared_scalar: &self.moe_shared_scalar,
            }
        }

        fn weights<'a>(&'a self, lm_head: &'a DenseWeight) -> ModelWeights<'a, DenseWeight> {
            ModelWeights {
                embed: &self.embed,
                layers: vec![
                    FlashLayerWeights {
                        attn_hyper: self.hyper(true),
                        attn: AttnWeights::Delta(DeltaWeights {
                            qkv: &self.delta_qkv,
                            z: &self.delta_z,
                            b: &self.delta_b,
                            a: &self.delta_a,
                            out: &self.delta_out,
                            conv: &self.delta_conv,
                            a_log: &self.delta_a_log,
                            dt_bias: &self.delta_dt_bias,
                            norm: &self.delta_norm,
                        }),
                        mlp_hyper: self.hyper(true),
                        moe: self.moe(),
                    },
                    FlashLayerWeights {
                        attn_hyper: self.hyper(true),
                        attn: AttnWeights::Qsa(QsaWeights {
                            q: &self.qsa_q,
                            k: &self.qsa_k,
                            v: &self.qsa_v,
                            o: &self.qsa_o,
                            idx: &self.qsa_idx,
                            q_norm: &self.qsa_q_norm,
                            k_norm: &self.qsa_k_norm,
                            idx_q_norm: &self.qsa_idx_q_norm,
                            idx_k_norm: &self.qsa_idx_k_norm,
                        }),
                        mlp_hyper: self.hyper(true),
                        moe: self.moe(),
                    },
                ],
                ple: PleWeights {
                    multipliers: &self.ple_multipliers,
                    vocab: &self.ple_vocab,
                    offsets: &self.ple_offsets,
                    shards: &self.ple_shards,
                    shard_rows: &self.ple_shard_rows,
                    weight_scale: 1.0,
                    key: &self.ple_key,
                    value: &self.ple_value,
                    norm_key: &self.ple_norm_key,
                    norm_query: &self.ple_norm_query,
                    norm_conv: &self.ple_norm_conv,
                    conv: &self.ple_conv,
                },
                mixer: self.hyper(false),
                lm_head,
            }
        }
    }

    #[test]
    fn two_steps_through_a_zero_block_model() {
        let fixture = Fixture::new();
        let head = good_head();
        let weights = fixture.weights(&head);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &weights);
        let mut projector = DenseProjector::default();

        let first = step(&mut projector, &shape, &weights, &mut state, 0, true).unwrap();
        assert_close(&first.logits, &[0.424_264_1, 0.565_685_4, 0.989_949_5]);
        let layers = first.layer_out.as_ref().expect("recorded layers");
        assert_eq!(layers.len(), 2);
        for layer in layers {
            assert_close(layer, &[3.0, 4.0, 3.0, 4.0]);
        }
        assert_eq!(projector.calls, 44);
        assert_eq!(state.pos(), 1);

        let second = step(&mut projector, &shape, &weights, &mut state, 2, false).unwrap();
        assert_close(
            &second.logits,
            &[
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
        );
        assert_eq!(second.layer_out, None);
        assert_eq!(projector.calls, 88);
        assert_eq!(state.pos(), 2);
    }

    #[test]
    fn state_kinds_follow_the_layer_weights() {
        let fixture = Fixture::new();
        let head = good_head();
        let weights = fixture.weights(&head);
        let shape = Fixture::shape(1);
        let state = ModelState::new(&shape, &weights);

        assert!(matches!(state.layers[0], LayerState::Delta(_)));
        assert!(matches!(state.layers[1], LayerState::Qsa(_)));
        assert_eq!(state.byte_len().unwrap(), 64);
    }

    #[test]
    fn byte_len_after_one_step() {
        let fixture = Fixture::new();
        let head = good_head();
        let weights = fixture.weights(&head);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &weights);
        let mut projector = DenseProjector::default();

        let out = step(&mut projector, &shape, &weights, &mut state, 0, false);
        assert!(out.is_ok());
        assert_eq!(state.byte_len().unwrap(), 88);
    }

    #[test]
    fn byte_len_counts_a_non_eos_history_token() {
        let fixture = Fixture::new();
        let head = good_head();
        let weights = fixture.weights(&head);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &weights);
        let mut projector = DenseProjector::default();

        let out = step(&mut projector, &shape, &weights, &mut state, 1, false);
        assert!(out.is_ok());
        assert_eq!(state.byte_len().unwrap(), 92);
    }

    #[test]
    fn position_limit_is_a_context_error_without_change() {
        let fixture = Fixture::new();
        let head = good_head();
        let weights = fixture.weights(&head);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &weights);
        state.pos = u32::MAX;
        let before = state.clone();
        let mut projector = DenseProjector::default();

        let result = step(&mut projector, &shape, &weights, &mut state, 0, false);
        assert!(matches!(result, Err(EngError::ContextLimit(_))));
        assert_eq!(projector.calls, 0);
        assert_eq!(state, before);
    }

    struct OrderProjector {
        inner: DenseProjector,
        log: Vec<usize>,
    }

    impl Projector for OrderProjector {
        type Weight = DenseWeight;
        fn linear(&mut self, w: &DenseWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
            self.log.push(std::ptr::from_ref(w) as usize);
            self.inner.linear(w, x)
        }
        fn expert_linear(
            &mut self,
            w: &DenseWeight,
            e: u32,
            x: &[f32],
        ) -> Result<Vec<f32>, EngError> {
            self.log.push(std::ptr::from_ref(w) as usize);
            self.inner.expert_linear(w, e, x)
        }
        fn rows(&mut self, w: &DenseWeight, r: &[u32]) -> Result<Vec<f32>, EngError> {
            self.log.push(std::ptr::from_ref(w) as usize);
            self.inner.rows(w, r)
        }
    }

    #[test]
    fn call_order_follows_the_oracle_step() {
        let fx = Fixture::new();
        let head = good_head();
        let weights = fx.weights(&head);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &weights);
        let mut projector = OrderProjector {
            inner: DenseProjector::default(),
            log: Vec::new(),
        };
        let a = |w: &DenseWeight| std::ptr::from_ref(w) as usize;
        let attn_read = [a(&fx.hyper_down), a(&fx.hyper_up), a(&fx.hyper_inject)];
        let moe_calls = [
            a(&fx.moe_gate),
            a(&fx.moe_up),
            a(&fx.moe_down),
            a(&fx.moe_shared_gate),
            a(&fx.moe_shared_up),
            a(&fx.moe_shared_down),
            a(&fx.moe_shared_scalar),
        ];
        let mut expected = vec![a(&fx.embed)];
        expected.extend(attn_read);
        expected.extend([
            a(&fx.delta_qkv),
            a(&fx.delta_z),
            a(&fx.delta_b),
            a(&fx.delta_a),
            a(&fx.delta_out),
        ]);
        expected.extend(attn_read);
        expected.extend(moe_calls);
        expected.extend([
            a(&fx.ple_shards[0]),
            a(&fx.ple_shards[0]),
            a(&fx.ple_key),
            a(&fx.ple_value),
        ]);
        expected.extend(attn_read);
        expected.extend([
            a(&fx.qsa_q),
            a(&fx.qsa_k),
            a(&fx.qsa_v),
            a(&fx.qsa_idx),
            a(&fx.qsa_o),
        ]);
        expected.extend(attn_read);
        expected.extend(moe_calls);
        expected.extend([a(&fx.hyper_down), a(&fx.hyper_up)]);
        expected.push(a(&head));

        step(&mut projector, &shape, &weights, &mut state, 1, false).unwrap();
        assert_eq!(projector.log.len(), 44);
        assert_eq!(projector.log, expected);
    }

    #[test]
    fn failed_step_poisons_until_reset() {
        let fixture = Fixture::new();
        let good = good_head();
        let bad = DenseWeight::matrix(3, 3, vec![0.0; 9]);
        let good_weights = fixture.weights(&good);
        let bad_weights = fixture.weights(&bad);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &good_weights);
        let mut projector = DenseProjector::default();

        let failed = step(&mut projector, &shape, &bad_weights, &mut state, 0, false);
        assert!(matches!(failed, Err(EngError::BadRequest(_))));

        let poisoned = step(&mut projector, &shape, &good_weights, &mut state, 0, false);
        assert!(matches!(poisoned, Err(EngError::BadRequest(_))));

        state.reset(&shape, &good_weights);
        let recovered = step(&mut projector, &shape, &good_weights, &mut state, 0, false);
        assert!(recovered.is_ok());
        assert_eq!(state.pos(), 1);
    }

    #[test]
    fn ple_layer_outside_the_model_is_rejected_without_poisoning() {
        let fixture = Fixture::new();
        let head = good_head();
        let weights = fixture.weights(&head);
        let bad_shape = Fixture::shape(2);
        let shape = Fixture::shape(1);
        let mut state = ModelState::new(&shape, &weights);
        let mut projector = DenseProjector::default();

        let rejected = step(&mut projector, &bad_shape, &weights, &mut state, 0, false);
        assert!(matches!(rejected, Err(EngError::BadRequest(_))));
        assert_eq!(projector.calls, 0);

        let recovered = step(&mut projector, &shape, &weights, &mut state, 0, false);
        assert!(recovered.is_ok());
    }
}
