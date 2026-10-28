use std::{cell::RefCell, collections::HashMap, rc::Rc};

use inference_engine::{
    EngError,
    family::nemotron35::{
        NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
        NemotronTextSpec,
        exec::{
            ExpertIndex, ExpertProjection, NemotronExecSpec, NemotronStack, NemotronState,
            WeightSource,
        },
    },
    tensor::{Dtype, Tensor},
};

const ROOT: &str = "language_model.backbone.";
type Counts = Rc<RefCell<HashMap<String, usize>>>;

fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn values(t: &Tensor) -> Vec<f32> {
    t.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 3e-6, "{a} != {e}");
    }
}

fn spec(blocks: Vec<NemotronBlockKind>) -> NemotronExecSpec {
    NemotronExecSpec::new(
        NemotronGeometry {
            hidden_size: 8,
            vocab_size: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            layer_norm_epsilon: 2.0,
        },
        NemotronMambaGeometry {
            groups: 1,
            state_size: 1,
            conv_kernel: 2,
            heads: 2,
            head_dim: 4,
            expand: 1,
            inner_size: 8,
            chunk_size: 1,
        },
        NemotronMoeGeometry {
            intermediate_size: 1,
            moe_intermediate_size: 1,
            latent_size: 2,
            shared_expert_intermediate_size: 1,
            routed_experts: 4,
            shared_experts: 1,
            experts_per_token: 1,
            n_group: 1,
            n_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        },
        blocks,
    )
    .unwrap()
}

struct FakeSource {
    dense: HashMap<String, Tensor>,
    counts: Counts,
}

impl FakeSource {
    fn new(attention: bool) -> Self {
        let mut source = Self {
            dense: HashMap::new(),
            counts: Rc::default(),
        };
        let mut embeddings = vec![0.; 24];
        for row in 0..3 {
            embeddings[row * 8 + usize::from(attention && row == 1)] = 4.;
        }
        source.put("embeddings.weight", &[3, 8], &embeddings);
        source.put("norm_f.weight", &[8], &[1.; 8]);
        let mut head = vec![0.; 24];
        head[0] = 1.;
        head[9] = 2.;
        head[17] = 2.;
        source.dense.insert(
            "language_model.lm_head.weight".into(),
            tensor(&[3, 8], &head),
        );
        source.put("layers.0.norm.weight", &[8], &[1.; 8]);
        if attention {
            let mut kv = vec![0.; 32];
            kv[0] = 1.;
            kv[9] = 1.;
            let mut out = vec![0.; 64];
            out[0] = 1.;
            source.put("layers.0.mixer.q_proj.weight", &[8, 8], &[0.; 64]);
            source.put("layers.0.mixer.k_proj.weight", &[4, 8], &kv);
            source.put("layers.0.mixer.v_proj.weight", &[4, 8], &kv);
            source.put("layers.0.mixer.o_proj.weight", &[8, 8], &out);
        } else {
            let mut input = vec![0.; 20 * 8];
            // Gate rows 0..8 stay zero. x,B,C rows 8..18 read normalized x[0]=2.
            for row in 8..18 {
                input[row * 8] = 1.;
            }
            source.put("layers.0.mixer.in_proj.weight", &[20, 8], &input);
            source.put("layers.0.mixer.out_proj.weight", &[8, 8], &[0.; 64]);
            source.put("layers.0.mixer.norm.weight", &[8], &[1.; 8]);
            source.put(
                "layers.0.mixer.conv1d.weight",
                &[10, 1, 2],
                &[0., 1.].repeat(10),
            );
            source.put("layers.0.mixer.conv1d.bias", &[10], &[0.; 10]);
            for name in ["dt_bias", "A_log", "D"] {
                source.put(&format!("layers.0.mixer.{name}"), &[2], &[0.; 2]);
            }
            source.put("layers.1.norm.weight", &[8], &[1.; 8]);
            source.put("layers.1.mixer.gate.weight", &[4, 8], &[0.; 32]);
            source.put(
                "layers.1.mixer.gate.e_score_correction_bias",
                &[4],
                &[0., 0., 1., 0.],
            );
            let mut up = vec![0.; 16];
            up[0] = 0.5;
            source.put("layers.1.mixer.fc1_latent_proj.weight", &[2, 8], &up);
            let mut down = vec![0.; 16];
            down[2] = 3.;
            source.put("layers.1.mixer.fc2_latent_proj.weight", &[8, 2], &down);
        }
        source
    }

    fn put(&mut self, name: &str, shape: &[u32], values: &[f32]) {
        self.dense
            .insert(format!("{ROOT}{name}"), tensor(shape, values));
    }
}

impl WeightSource for FakeSource {
    fn dense(&mut self, name: &str) -> Result<Tensor, EngError> {
        *self.counts.borrow_mut().entry(name.into()).or_default() += 1;
        self.dense
            .get(name)
            .cloned()
            .ok_or_else(|| EngError::Checkpoint(format!("unknown text weight: {name}")))
    }

    fn expert(
        &mut self,
        layer: u32,
        expert: ExpertIndex,
        projection: ExpertProjection,
    ) -> Result<Tensor, EngError> {
        assert_eq!(layer, 1);
        let key = format!("expert:{layer}:{expert:?}:{projection:?}");
        *self.counts.borrow_mut().entry(key).or_default() += 1;
        match (expert, projection) {
            (ExpertIndex::Routed(2), ExpertProjection::Up) => Ok(tensor(&[1, 2], &[1., 0.])),
            (ExpertIndex::Routed(2), ExpertProjection::Down) => Ok(tensor(&[2, 1], &[1., 0.])),
            (ExpertIndex::Shared, ExpertProjection::Up) => {
                Ok(tensor(&[1, 8], &[0.5, 0., 0., 0., 0., 0., 0., 0.]))
            }
            (ExpertIndex::Shared, ExpertProjection::Down) => {
                Ok(tensor(&[8, 1], &[0., 0., 4., 0., 0., 0., 0., 0.]))
            }
            _ => Err(EngError::Checkpoint("unselected expert requested".into())),
        }
    }
}

fn stack(capacity: u32) -> NemotronStack<FakeSource> {
    let spec = spec(vec![NemotronBlockKind::Mamba, NemotronBlockKind::Moe]);
    let state = NemotronState::new(&spec, capacity).unwrap();
    NemotronStack::new(spec, FakeSource::new(false), state).unwrap()
}

#[test]
fn tiny_stack_prefill_decode_and_literal_logits() {
    let mut stack = stack(8);
    stack.prefill(&[0, 1, 2]).unwrap();
    // Embedding=[4,0,..], RMS=sqrt(16/8+2)=2; normalized=[2,0,..].
    // Mamba gate and out_proj are zero, so its residual remains [4,0,..].
    // Router logits=0, correction selects expert 2, normalized top-1 weight=1.
    // Latent=[1,0], expert relu2(1)=1, latent down adds [0,3,0,..].
    // Shared relu2(.5*2)=1 adds [0,0,4,..]; residual gives [4,3,4,..].
    // Final RMS=sqrt((16+9+16)/8+2)=sqrt(57/8); head gives [4,6,6]/RMS.
    let rms = (57_f32 / 8.).sqrt();
    close(stack.logits().unwrap(), &[4. / rms, 6. / rms, 6. / rms]);
    assert_eq!(stack.state().position(), 3);
    assert_eq!(stack.next_token().unwrap(), 1); // Tied ids 1 and 2 choose 1.
    assert_eq!(stack.state().position(), 4);
}

#[test]
fn next_token_runs_one_step_using_the_previous_sample() {
    let make_stack = || {
        let spec = spec(vec![NemotronBlockKind::Attention]);
        let state = NemotronState::new(&spec, 8).unwrap();
        NemotronStack::new(spec, FakeSource::new(true), state).unwrap()
    };
    let mut decoded = make_stack();
    let mut explicit = make_stack();
    // Token 1 predicts 1; consuming token 0 instead would predict 0.
    decoded.prefill(&[1]).unwrap();
    assert_eq!(explicit.step(1).unwrap(), 1);
    let first = decoded.next_token().unwrap();
    assert_eq!(first, explicit.step(1).unwrap());
    assert_eq!(decoded.next_token().unwrap(), explicit.step(first).unwrap());
    assert_eq!(decoded.state().position(), 3);
    close(decoded.logits().unwrap(), explicit.logits().unwrap());
}

#[test]
fn mamba_state_evolves_through_prefill_and_decode() {
    let mut stack = stack(8);
    stack.prefill(&[0, 0, 0]).unwrap();
    let (conv, ssm) = stack.state().mamba_state(0).unwrap();
    close(&values(conv), &[2., 2.].repeat(10));
    // x=B=SiLU(2); dt=ln(2); A=-1; decay=1/2.
    // s_n=ln(2)*SiLU(2)^2*(1+1/2+...+1/2^(n-1)).
    let increment = 2_f32.ln() * (2. / (1. + (-2_f32).exp())).powi(2);
    close(&values(ssm), &[increment * 1.75; 8]);
    stack.next_token().unwrap();
    close(
        &values(stack.state().mamba_state(0).unwrap().1),
        &[increment * 1.875; 8],
    );
    // Conv:10*2, SSM:2*4*1, all F32. MoE has no mutable state.
    assert_eq!(stack.state().total_state_bytes(), (20 + 8) * 4);
}

#[test]
fn attention_appends_kv_rows_and_retains_them() {
    let spec = spec(vec![NemotronBlockKind::Attention]);
    let state = NemotronState::new(&spec, 4).unwrap();
    // Two caches (keys and values) of [capacity, kv_width=1x4].
    assert_eq!(state.total_state_bytes(), (4 * 4) * 4 * 2);
    let mut stack = NemotronStack::new(spec, FakeSource::new(true), state).unwrap();
    stack.step(0).unwrap();
    stack.step(1).unwrap();
    // The cache keeps the projected rows of consumed positions: rows 0 and
    // 1 are written (and differ — different inputs), rows >= position stay
    // zero until consumed.
    let keys = stack.attention_keys(0).unwrap();
    assert_eq!(keys.len(), 4 * 4);
    let (row0, rest) = keys.split_at(4);
    let (row1, tail) = rest.split_at(4);
    assert!(row0.iter().any(|&v| v != 0.));
    assert!(row1.iter().any(|&v| v != 0.));
    assert!(row0 != row1);
    assert!(tail.iter().all(|&v| v == 0.));
    // Q=0 gives uniform attention. V history=[2,0], current context=1.
    // The second residual is [0,4,..], so attention output=[1,4,..].
    let rms = (17_f32 / 8. + 2.).sqrt();
    close(stack.logits().unwrap(), &[1. / rms, 8. / rms, 8. / rms]);
    assert_eq!(stack.state().position(), 2);
}

#[test]
fn dense_weights_fetch_once_and_experts_refetch_each_step() {
    let spec = spec(vec![NemotronBlockKind::Mamba, NemotronBlockKind::Moe]);
    let source = FakeSource::new(false);
    let counts = Rc::clone(&source.counts);
    let state = NemotronState::new(&spec, 4).unwrap();
    let mut stack = NemotronStack::new(spec, source, state).unwrap();
    stack.prefill(&[0, 1]).unwrap();
    assert_eq!(counts.borrow().len(), 21);
    for (name, count) in counts.borrow().iter() {
        // Dense and latent weights are fetched once. The stack caches only
        // the bounded shared pair (layers x two projections), so shared
        // expert weights also fetch once. Routed experts stream from the
        // source every step — the source owns their residency.
        let routed = name.starts_with("expert:") && name.contains("Routed(");
        let expected = 1 + usize::from(routed);
        assert_eq!(*count, expected, "{name}");
    }
}

#[test]
fn mtp_and_unknown_names_are_not_text_weights() {
    let mut source = FakeSource::new(false);
    for name in ["language_model.mtp.layers.0.norm.weight", "unknown"] {
        assert!(matches!(source.dense(name), Err(EngError::Checkpoint(_))));
    }
    let descriptor =
        NemotronTextSpec::from_config(include_bytes!("fixtures/nemotron35/config.json")).unwrap();
    assert!(
        descriptor
            .key_plan("language_model.mtp.layers.0.norm.weight")
            .is_err()
    );
    let spec = NemotronExecSpec::from_descriptor(&descriptor).unwrap();
    assert_eq!(spec.block_order(), descriptor.block_order());
}

#[test]
fn capacity_overflow_and_invalid_inputs_do_not_advance_state() {
    let mut stack = stack(1);
    assert!(stack.next_token().is_err());
    assert!(stack.step(3).is_err());
    assert_eq!(stack.state().position(), 0);
    stack.step(0).unwrap();
    assert!(stack.step(0).is_err());
    assert!(stack.prefill(&[0, 0]).is_err());
    assert_eq!(stack.state().position(), 1);
}

#[test]
fn malformed_weight_is_a_checkpoint_error_without_state_publication() {
    let spec = spec(vec![NemotronBlockKind::Mamba, NemotronBlockKind::Moe]);
    let state = NemotronState::new(&spec, 4).unwrap();
    let mut source = FakeSource::new(false);
    source.put("layers.1.mixer.fc2_latent_proj.weight", &[1], &[0.]);
    let mut stack = NemotronStack::new(spec, source, state).unwrap();
    assert!(matches!(stack.step(0), Err(EngError::Checkpoint(_))));
    assert_eq!(stack.state().position(), 0);
    assert!(
        values(stack.state().mamba_state(0).unwrap().1)
            .iter()
            .all(|&v| v == 0.)
    );
}

#[test]
fn state_rejects_invalid_capacity_and_stack_rejects_a_different_layout() {
    let spec = spec(vec![NemotronBlockKind::Mamba]);
    assert!(NemotronState::new(&spec, 0).is_err());
    assert!(NemotronState::new(&spec, 17).is_err());
    let state = NemotronState::new(&spec, 4).unwrap();
    let other = self::spec(vec![NemotronBlockKind::Moe]);
    assert!(NemotronStack::new(other, FakeSource::new(false), state).is_err());
}

#[test]
fn execution_spec_rejects_inconsistent_or_unrepresentable_geometry() {
    let descriptor =
        NemotronTextSpec::from_config(include_bytes!("fixtures/nemotron35/config.json")).unwrap();
    let g = descriptor.geometry();
    let m = descriptor.mamba_geometry();
    let e = descriptor.moe_geometry();
    let blocks = vec![NemotronBlockKind::Mamba];
    let mut bad_g = g;
    bad_g.layer_norm_epsilon = f64::NAN;
    assert!(NemotronExecSpec::new(bad_g, m, e, blocks.clone()).is_err());
    bad_g = g;
    bad_g.hidden_size += 1;
    assert!(NemotronExecSpec::new(bad_g, m, e, blocks.clone()).is_err());
    let mut bad_m = m;
    bad_m.state_size = u32::MAX;
    assert!(NemotronExecSpec::new(g, bad_m, e, blocks.clone()).is_err());
    let mut bad_e = e;
    bad_e.experts_per_token = e.routed_experts + 1;
    assert!(NemotronExecSpec::new(g, m, bad_e, blocks).is_err());
    assert!(NemotronExecSpec::new(g, m, e, vec![]).is_err());
}

#[test]
fn state_checks_allocation_extents_before_allocating() {
    let descriptor =
        NemotronTextSpec::from_config(include_bytes!("fixtures/nemotron35/config.json")).unwrap();
    let mut g = descriptor.geometry();
    let mut m = descriptor.mamba_geometry();
    let mut e = descriptor.moe_geometry();
    g.hidden_size = 1 << 29;
    g.num_attention_heads = 1;
    g.num_key_value_heads = 1;
    g.head_dim = g.hidden_size;
    g.max_position_embeddings = u32::MAX;
    m.heads = 1;
    m.head_dim = g.hidden_size;
    m.inner_size = g.hidden_size;
    m.expand = 1;
    m.groups = 1;
    m.state_size = 1;
    e.n_groups = 1;
    let spec = NemotronExecSpec::new(g, m, e, vec![NemotronBlockKind::Attention; 2]).unwrap();
    assert!(matches!(
        NemotronState::new(&spec, u32::MAX),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn nonfinite_and_wrong_dtype_weights_are_rejected() {
    for weight in [
        tensor(&[8], &[f32::NAN; 8]),
        Tensor::try_from_host(Dtype::U32, vec![8], vec![0; 32]).unwrap(),
    ] {
        let attention_spec = spec(vec![NemotronBlockKind::Attention]);
        let state = NemotronState::new(&attention_spec, 4).unwrap();
        let mut source = FakeSource::new(true);
        source.dense.insert(format!("{ROOT}norm_f.weight"), weight);
        let mut stack = NemotronStack::new(attention_spec, source, state).unwrap();
        assert!(matches!(stack.step(0), Err(EngError::Checkpoint(_))));
        assert_eq!(stack.state().position(), 0);
        // The stack-owned K/V caches may hold scratch rows at or above the
        // live position after a failed step (the poisoned source can never
        // retry successfully here); rows below it only ever come from
        // successful steps, and a failed session is discarded wholesale by
        // the executor, so scratch never escapes.
    }
}

#[test]
fn prefill_exposes_its_prediction_before_next_token_advances() {
    // Contract for callers (runner): the FIRST generated token is last_prediction()
    // after prefill; next_token() returns the token AFTER that. Emitting the stored
    // prediction first, then advancing, must equal the stack's own sequence.
    let mut a = stack(16);
    a.prefill(&[0, 1, 2]).unwrap();
    let first = a.last_prediction().expect("prediction after prefill");
    let second = a.next_token().unwrap();
    let mut b = stack(16);
    b.prefill(&[0, 1, 2]).unwrap();
    assert_eq!(b.last_prediction(), Some(first));
    // Feeding 'first' explicitly must produce 'second': emit-first is equivalent.
    assert_eq!(b.step(first).unwrap(), second);
}

#[test]
fn reset_restores_pristine_state_and_reuses_dense_weights() {
    for blocks in [
        vec![NemotronBlockKind::Attention],
        vec![NemotronBlockKind::Mamba, NemotronBlockKind::Moe],
    ] {
        let geometry = spec(blocks);
        let source = FakeSource::new(geometry.block_order() == [NemotronBlockKind::Attention]);
        let counts = source.counts.clone();
        let mut stack = NemotronStack::new(
            geometry.clone(),
            source,
            NemotronState::new(&geometry, 4).unwrap(),
        )
        .unwrap();
        let pristine = stack.state().clone();
        stack.prefill(&[1, 0]).unwrap();
        let consumed = stack.state().clone();
        let logits = stack.logits().unwrap().to_vec();
        let fetches = counts.borrow().clone();
        stack.reset();
        assert_eq!(stack.state().position(), 0);
        assert_state_equal(stack.state(), &pristine);
        assert_eq!(stack.logits(), None);
        assert_eq!(stack.last_prediction(), None);
        stack.prefill(&[1, 0]).unwrap();
        assert_eq!(stack.state().position(), consumed.position());
        assert_state_equal(stack.state(), &consumed);
        assert_eq!(stack.logits().unwrap(), logits);
        for (name, count) in fetches
            .iter()
            .filter(|(name, _)| !name.starts_with("expert:"))
        {
            assert_eq!(counts.borrow().get(name), Some(count), "{name}");
        }
    }
}

fn assert_state_equal(actual: &NemotronState, expected: &NemotronState) {
    for block in 0..2 {
        match (actual.mamba_state(block), expected.mamba_state(block)) {
            (Some((actual_conv, actual_ssm)), Some((expected_conv, expected_ssm))) => {
                assert_eq!(actual_conv.host_bytes(), expected_conv.host_bytes());
                assert_eq!(actual_ssm.host_bytes(), expected_ssm.host_bytes());
            }
            (None, None) => {}
            _ => panic!("different block states"),
        }
        // Attention blocks keep no state beyond the shared position counter;
        // their K/V caches live on the stack.
    }
}
