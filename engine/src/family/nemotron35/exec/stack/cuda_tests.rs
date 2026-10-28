use std::collections::HashMap;

use crate::{
    EngError,
    family::nemotron35::{
        NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
        admission::{
            self, ACTIVATION_WORKSPACE_BYTES,
            cuda::{CudaAdmission, CudaPlanInput, admit_cuda_plan},
        },
        exec::{
            ExpertIndex, ExpertProjection, NemotronExecSpec, NemotronStack, NemotronState,
            WeightSource,
        },
    },
    memory::MemoryPolicy,
    tensor::{Dtype, Tensor},
};

const ROOT: &str = "language_model.backbone.";

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

fn fixture_admission(source: &mut NonzeroMixedSource, spec: &NemotronExecSpec) -> CudaAdmission {
    let mut tensors: Vec<_> = source.dense.values().cloned().collect();
    for expert in [ExpertIndex::Routed(2), ExpertIndex::Shared] {
        for projection in [ExpertProjection::Up, ExpertProjection::Down] {
            tensors.push(source.expert(2, expert, projection).unwrap());
        }
    }
    let mut source_bytes = 0_u64;
    let mut largest = 0_u64;
    let mut max_shape = [0_u64; 2];
    for tensor in &tensors {
        let bytes = u64::try_from(tensor.host_bytes().unwrap().len()).unwrap();
        source_bytes = source_bytes.checked_add(bytes).unwrap();
        largest = largest.max(bytes);
        if let &[rows, columns] = tensor.shape() {
            max_shape[0] = max_shape[0].max(u64::from(rows));
            max_shape[1] = max_shape[1].max(u64::from(columns));
        }
    }
    let states = NemotronState::planned_bytes(spec, 12)
        .unwrap()
        .checked_mul(3)
        .unwrap();
    let copies = largest.checked_mul(2).unwrap();
    let host_base_bytes = ACTIVATION_WORKSPACE_BYTES
        .checked_add(source_bytes)
        .and_then(|bytes| bytes.checked_add(states))
        .and_then(|bytes| bytes.checked_add(copies))
        .unwrap();
    admit_cuda_plan(
        CudaPlanInput {
            host_base_bytes,
            largest_tensor_bytes: largest,
            packed_pair_bytes: 0,
            max_shape,
            device_weight_cache_bytes: 0,
        },
        MemoryPolicy::default_share(),
        admission::host_available_bytes().expect("measured host memory"),
    )
    .unwrap()
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
struct NonzeroMixedSource {
    dense: HashMap<String, Tensor>,
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl NonzeroMixedSource {
    fn new() -> Self {
        let mut source = Self {
            dense: HashMap::new(),
        };
        source.put_matrix(
            "language_model.backbone.embeddings.weight",
            3,
            8,
            &[
                (0, 0, 1.0),
                (0, 1, 0.25),
                (1, 1, 1.0),
                (1, 2, 0.25),
                (2, 2, 1.0),
            ],
        );
        source.put_vector("language_model.backbone.norm_f.weight", &[1.0; 8]);
        source.put_matrix(
            "language_model.lm_head.weight",
            3,
            8,
            &[(0, 0, 0.5), (1, 1, 1.0), (2, 2, 0.75)],
        );

        for block in 0..3 {
            source.put_vector(&format!("{ROOT}layers.{block}.norm.weight"), &[1.0; 8]);
        }
        source.put_vector(&format!("{ROOT}layers.0.mixer.norm.weight"), &[1.0; 8]);
        let mut mamba_input = Vec::new();
        for index in 0..8 {
            mamba_input.push((index, index, 0.2));
            mamba_input.push((8 + index, index, 0.2));
        }
        mamba_input.extend([(16, 0, 0.2), (17, 1, 0.2), (18, 0, 0.1), (19, 1, 0.1)]);
        source.put_matrix(
            &format!("{ROOT}layers.0.mixer.in_proj.weight"),
            20,
            8,
            &mamba_input,
        );
        source.put_matrix(
            &format!("{ROOT}layers.0.mixer.out_proj.weight"),
            8,
            8,
            &(0..8).map(|index| (index, index, 0.1)).collect::<Vec<_>>(),
        );
        source.put(
            &format!("{ROOT}layers.0.mixer.conv1d.weight"),
            &[10, 1, 2],
            &(0..10).flat_map(|_| [0.25, 0.5]).collect::<Vec<_>>(),
        );
        source.put_vector(&format!("{ROOT}layers.0.mixer.conv1d.bias"), &[0.0; 10]);
        source.put_vector(&format!("{ROOT}layers.0.mixer.dt_bias"), &[0.1; 2]);
        source.put_vector(&format!("{ROOT}layers.0.mixer.A_log"), &[0.0; 2]);
        source.put_vector(&format!("{ROOT}layers.0.mixer.D"), &[0.1; 2]);

        source.put_matrix(
            &format!("{ROOT}layers.1.mixer.q_proj.weight"),
            8,
            8,
            &[(0, 0, 0.25), (4, 1, 0.25)],
        );
        source.put_matrix(
            &format!("{ROOT}layers.1.mixer.k_proj.weight"),
            4,
            8,
            &[(0, 0, 0.25), (1, 1, 0.25)],
        );
        source.put_matrix(
            &format!("{ROOT}layers.1.mixer.v_proj.weight"),
            4,
            8,
            &[(0, 0, 0.5), (1, 1, 0.5)],
        );
        source.put_matrix(
            &format!("{ROOT}layers.1.mixer.o_proj.weight"),
            8,
            8,
            &[(0, 0, 0.25), (1, 1, 0.25), (4, 4, 0.25)],
        );

        source.put_matrix(
            &format!("{ROOT}layers.2.mixer.gate.weight"),
            4,
            8,
            &[(0, 1, 0.1), (2, 0, 0.2)],
        );
        source.put_vector(
            &format!("{ROOT}layers.2.mixer.gate.e_score_correction_bias"),
            &[0.0, 0.0, 1.0, 0.0],
        );
        source.put_matrix(
            &format!("{ROOT}layers.2.mixer.fc1_latent_proj.weight"),
            2,
            8,
            &[(0, 0, 0.25), (1, 1, 0.25)],
        );
        source.put_matrix(
            &format!("{ROOT}layers.2.mixer.fc2_latent_proj.weight"),
            8,
            2,
            &[(0, 0, 0.25), (1, 1, 0.25), (2, 0, 0.1)],
        );
        source
    }

    fn put_matrix(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        entries: &[(usize, usize, f32)],
    ) {
        let mut data = vec![0.0; rows * columns];
        for &(row, column, value) in entries {
            data[row * columns + column] = value;
        }
        self.put(name, &[rows as u32, columns as u32], &data);
    }

    fn put_vector(&mut self, name: &str, values: &[f32]) {
        self.put(name, &[values.len() as u32], values);
    }

    fn put(&mut self, name: &str, shape: &[u32], values: &[f32]) {
        self.dense.insert(name.into(), tensor(shape, values));
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
impl WeightSource for NonzeroMixedSource {
    fn dense(&mut self, name: &str) -> Result<Tensor, EngError> {
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
        if layer != 2 {
            return Err(EngError::Checkpoint("unexpected expert layer".into()));
        }
        match (expert, projection) {
            (ExpertIndex::Routed(2), ExpertProjection::Up) => Ok(tensor(&[1, 2], &[0.25, 0.0])),
            (ExpertIndex::Routed(2), ExpertProjection::Down) => Ok(tensor(&[2, 1], &[0.25, 0.125])),
            (ExpertIndex::Shared, ExpertProjection::Up) => {
                Ok(tensor(&[1, 8], &[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            }
            (ExpertIndex::Shared, ExpertProjection::Down) => Ok(tensor(
                &[8, 1],
                &[0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )),
            _ => Err(EngError::Checkpoint("unselected expert requested".into())),
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_fixture_spec() -> NemotronExecSpec {
    spec(vec![
        NemotronBlockKind::Mamba,
        NemotronBlockKind::Attention,
        NemotronBlockKind::Moe,
    ])
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn cuda_device_present() -> bool {
    let count = crate::kernels::cuda::device_count().expect("CUDA device count query");
    println!("CUDA devices visible: {count}");
    if count > 0 {
        return true;
    }
    let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
        .is_ok_and(|value| !value.is_empty() && value != "0");
    assert!(
        !required,
        "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible"
    );
    println!("SKIPPED cuda_stack_matches_cpu_across_prefill_and_decode: no CUDA device");
    false
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn assert_cuda_head(
    cpu: &NemotronStack<NonzeroMixedSource>,
    cuda: &NemotronStack<NonzeroMixedSource>,
) {
    let expected = cpu.logits().expect("the host arm keeps its row");
    assert!(
        cuda.logits().is_none(),
        "the CUDA arm returned a host vocabulary row"
    );
    let mut best = 0;
    for id in 1..expected.len() {
        if expected[id] > expected[best] {
            best = id;
        }
    }
    assert_eq!(cuda.last_prediction(), Some(best as u32));
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn assert_cuda_state(
    cpu: &NemotronStack<NonzeroMixedSource>,
    cuda: &NemotronStack<NonzeroMixedSource>,
) {
    assert_eq!(cuda.state().position(), cpu.state().position());
    assert_eq!(
        cuda.state().total_state_bytes(),
        cpu.state().total_state_bytes()
    );
    for block in [0, 1, 2] {
        match block {
            0 => {
                let (cpu_conv, cpu_ssm) = cpu.state().mamba_state(block).unwrap();
                let (cuda_conv, cuda_ssm) = cuda.state().mamba_state(block).unwrap();
                assert_eq!(cuda_conv.host_bytes(), cpu_conv.host_bytes());
                assert_eq!(cuda_ssm.host_bytes(), cpu_ssm.host_bytes());
            }
            1 => {
                let to_bytes = |row: Option<&[f32]>| {
                    row.unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>()
                };
                assert_eq!(
                    to_bytes(cuda.attention_keys(block)),
                    to_bytes(cpu.attention_keys(block))
                );
            }
            2 => {}
            _ => unreachable!(),
        }
    }
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
fn cuda_stack_matches_cpu_across_prefill_and_decode() {
    let _gpu = super::gpu_test_lock();
    if !cuda_device_present() {
        return;
    }
    let spec = cuda_fixture_spec();
    let mut cpu = NemotronStack::new(
        spec.clone(),
        NonzeroMixedSource::new(),
        NemotronState::new(&spec, 12).unwrap(),
    )
    .unwrap();
    let mut source = NonzeroMixedSource::new();
    let state = NemotronState::new(&spec, 12).unwrap();
    let admission = fixture_admission(&mut source, &spec);
    let mut cuda = NemotronStack::new_cuda(spec.clone(), source, state, admission).unwrap();

    cpu.prefill(&[0, 1]).unwrap();
    cuda.prefill(&[0, 1]).unwrap();
    assert_cuda_head(&cpu, &cuda);
    assert_cuda_state(&cpu, &cuda);
    assert!(
        values(cpu.state().mamba_state(0).unwrap().1)
            .iter()
            .any(|&value| value != 0.0)
    );
    assert!(
        cpu.attention_keys(1)
            .unwrap()
            .iter()
            .any(|&value| value != 0.0)
    );

    for _ in 0..3 {
        assert_eq!(cuda.next_token().unwrap(), cpu.next_token().unwrap());
        assert_cuda_head(&cpu, &cuda);
        assert_cuda_state(&cpu, &cuda);
    }
}
