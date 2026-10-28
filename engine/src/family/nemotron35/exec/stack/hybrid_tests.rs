//! Portable hybrid-decode tests: a synthetic Mamba/MoE/attention model with real NVFP4 expert
//! bytes, decoded once by the sequential packed path and once with the routed experts on the CPU
//! kernel. The two differ only in how the expert dots are summed.

use std::collections::HashMap;

use super::super::{
    ExpertIndex, ExpertProjection, GpuLayerPlan, HybridConfig, NemotronExecSpec, NemotronStack,
    NemotronState, PackedExpert, WeightSource,
};
use crate::{
    EngError,
    family::nemotron35::{
        NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
    },
    kernels::nvfp4_cpu::{ExpertView, LayerExperts, Nvfp4Matrix},
    tensor::{Dtype, Tensor},
};

#[test]
fn grouped_opt_in_selects_batched_routed_gpu() {
    let cases = [
        (false, None, false, false),
        (false, None, true, false),
        (false, Some(false), false, true),
        (false, Some(false), true, false),
        (false, Some(true), false, false),
        (false, Some(true), true, false),
        (true, None, false, true),
        (true, None, true, true),
        (true, Some(false), false, true),
        (true, Some(false), true, true),
        (true, Some(true), false, true),
        (true, Some(true), true, true),
    ];
    for (no_batch, hybrid_batch, grouped, expected) in cases {
        assert_eq!(
            super::routed_per_pair(no_batch, hybrid_batch, grouped),
            expected,
            "NO_BATCH={no_batch}, hybrid={hybrid_batch:?}, grouped={grouped}"
        );
    }
}

const ROOT: &str = "language_model.backbone.";
const HIDDEN: u32 = 16;
const VOCAB: u32 = 7;
const LATENT: u32 = 32;
const INTER: u32 = 32;
const SHARED: u32 = 16;
const EXPERTS: u32 = 8;
const BLOCKS: [NemotronBlockKind; 5] = [
    NemotronBlockKind::Mamba,
    NemotronBlockKind::Moe,
    NemotronBlockKind::Attention,
    NemotronBlockKind::Moe,
    NemotronBlockKind::Moe,
];

fn spec() -> NemotronExecSpec {
    NemotronExecSpec::new(
        NemotronGeometry {
            hidden_size: HIDDEN,
            vocab_size: VOCAB,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 32,
            layer_norm_epsilon: 1e-5,
        },
        NemotronMambaGeometry {
            groups: 1,
            state_size: 2,
            conv_kernel: 2,
            heads: 2,
            head_dim: 8,
            expand: 1,
            inner_size: 16,
            chunk_size: 1,
        },
        NemotronMoeGeometry {
            intermediate_size: SHARED,
            moe_intermediate_size: INTER,
            latent_size: LATENT,
            shared_expert_intermediate_size: SHARED,
            routed_experts: EXPERTS,
            shared_experts: 1,
            experts_per_token: 3,
            n_group: 1,
            n_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
        },
        BLOCKS.to_vec(),
    )
    .unwrap()
}

/// Deterministic noise, so the fixture needs no random-number crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

type Projection = (&'static [u8], &'static [u8], f32, [u32; 2]);

struct Fixture {
    dense: HashMap<String, Tensor>,
    /// Decoded shared experts, kept so a device cache sees the same bytes on every call.
    shared: HashMap<(u32, bool), Tensor>,
    experts: HashMap<(u32, u32, bool), Projection>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_intermediate(INTER)
    }

    fn with_intermediate(intermediate: u32) -> Self {
        let mut rng = Lcg(0x5eed);
        let mut dense = HashMap::new();
        let mut put = |name: String, shape: &[u32], scale: f32, offset: f32| {
            let count: u32 = shape.iter().product();
            let values: Vec<f32> = (0..count).map(|_| offset + scale * rng.unit()).collect();
            dense.insert(name, tensor(shape, &values));
        };
        let both = |tail: &str| format!("{ROOT}{tail}");
        put(both("embeddings.weight"), &[VOCAB, HIDDEN], 1.0, 0.0);
        put(both("norm_f.weight"), &[HIDDEN], 0.2, 1.0);
        put(
            "language_model.lm_head.weight".into(),
            &[VOCAB, HIDDEN],
            0.6,
            0.0,
        );
        for (block, kind) in BLOCKS.iter().enumerate() {
            let at = |tail: &str| format!("{ROOT}layers.{block}.{tail}");
            put(at("norm.weight"), &[HIDDEN], 0.2, 1.0);
            match kind {
                NemotronBlockKind::Mamba => {
                    put(at("mixer.norm.weight"), &[16], 0.2, 1.0);
                    // gate 16 + conv (x 16 + B 2 + C 2) + dt 2.
                    put(at("mixer.in_proj.weight"), &[38, HIDDEN], 0.4, 0.0);
                    put(at("mixer.out_proj.weight"), &[HIDDEN, 16], 0.4, 0.0);
                    put(at("mixer.conv1d.weight"), &[20, 1, 2], 0.5, 0.0);
                    put(at("mixer.conv1d.bias"), &[20], 0.1, 0.0);
                    put(at("mixer.dt_bias"), &[2], 0.1, 0.0);
                    put(at("mixer.A_log"), &[2], 0.1, 0.5);
                    put(at("mixer.D"), &[2], 0.1, 1.0);
                }
                NemotronBlockKind::Attention => {
                    put(at("mixer.q_proj.weight"), &[HIDDEN, HIDDEN], 0.5, 0.0);
                    put(at("mixer.k_proj.weight"), &[8, HIDDEN], 0.5, 0.0);
                    put(at("mixer.v_proj.weight"), &[8, HIDDEN], 0.5, 0.0);
                    put(at("mixer.o_proj.weight"), &[HIDDEN, HIDDEN], 0.5, 0.0);
                }
                NemotronBlockKind::Moe => {
                    put(at("mixer.gate.weight"), &[EXPERTS, HIDDEN], 0.6, 0.0);
                    put(
                        at("mixer.gate.e_score_correction_bias"),
                        &[EXPERTS],
                        0.05,
                        0.0,
                    );
                    put(
                        at("mixer.fc1_latent_proj.weight"),
                        &[LATENT, HIDDEN],
                        0.4,
                        0.0,
                    );
                    put(
                        at("mixer.fc2_latent_proj.weight"),
                        &[HIDDEN, LATENT],
                        0.4,
                        0.0,
                    );
                }
            }
        }
        let mut experts = HashMap::new();
        for (block, kind) in BLOCKS.iter().enumerate() {
            if *kind != NemotronBlockKind::Moe {
                continue;
            }
            for expert in 0..EXPERTS {
                for (down, rows, columns) in
                    [(false, intermediate, LATENT), (true, LATENT, intermediate)]
                {
                    let packed: Vec<u8> =
                        (0..rows * columns / 2).map(|_| rng.next() as u8).collect();
                    // E4M3 codes 0x28..=0x47: finite positive group scales of moderate size.
                    let scale: Vec<u8> = (0..rows * columns / 16)
                        .map(|_| 0x28 + (rng.next() % 32) as u8)
                        .collect();
                    experts.insert(
                        (block as u32, expert, down),
                        (
                            &*Box::leak(packed.into_boxed_slice()),
                            &*Box::leak(scale.into_boxed_slice()),
                            0.03125,
                            [rows, columns],
                        ),
                    );
                }
            }
        }
        Self {
            dense,
            shared: HashMap::new(),
            experts,
        }
    }
}

impl WeightSource for Fixture {
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
        // Only the shared expert is decoded; routed experts travel as packed bytes.
        let ExpertIndex::Shared = expert else {
            return Err(EngError::BadRequest("routed experts are packed".into()));
        };
        let down = projection == ExpertProjection::Down;
        let weight = self.shared.entry((layer, down)).or_insert_with(|| {
            let mut rng = Lcg(u64::from(layer) * 2 + u64::from(down));
            let shape = if down {
                [HIDDEN, SHARED]
            } else {
                [SHARED, HIDDEN]
            };
            let values: Vec<f32> = (0..shape[0] * shape[1]).map(|_| 0.4 * rng.unit()).collect();
            tensor(&shape, &values)
        });
        Ok(weight.clone())
    }

    fn has_packed_experts(&self) -> bool {
        true
    }

    fn expert_packed(
        &mut self,
        layer: u32,
        expert: ExpertIndex,
        projection: ExpertProjection,
    ) -> Result<PackedExpert, EngError> {
        let ExpertIndex::Routed(index) = expert else {
            return Err(EngError::BadRequest("shared experts are decoded".into()));
        };
        let &(packed, scale, weight_scale_2, shape) = self
            .experts
            .get(&(layer, index, projection == ExpertProjection::Down))
            .ok_or_else(|| EngError::Checkpoint("unknown expert".into()))?;
        Ok(PackedExpert {
            packed: packed.to_vec(),
            scale: scale.to_vec(),
            weight_scale_2,
            input_scale: 1.0,
            shape,
        })
    }

    fn cpu_expert_layer(&mut self, layer: u32) -> Result<LayerExperts<'static>, EngError> {
        let view = |index: u32, down: bool| {
            let &(packed, scale, weight_scale_2, [rows, columns]) =
                &self.experts[&(layer, index, down)];
            Nvfp4Matrix::new(
                packed,
                scale,
                weight_scale_2,
                rows as usize,
                columns as usize,
            )
            .map_err(|error| EngError::Checkpoint(error.to_string()))
        };
        let views = (0..EXPERTS)
            .map(|index| {
                Ok(ExpertView {
                    up: view(index, false)?,
                    down: view(index, true)?,
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        LayerExperts::new(views).map_err(|error| EngError::Checkpoint(error.to_string()))
    }
}

fn stack(config: Option<HybridConfig>) -> NemotronStack<Fixture> {
    let spec = spec();
    let state = NemotronState::new(&spec, 16).unwrap();
    match config {
        Some(config) => NemotronStack::new_hybrid(spec, Fixture::new(), state, config).unwrap(),
        None => NemotronStack::new(spec, Fixture::new(), state).unwrap(),
    }
}

const PROMPT: [u32; 3] = [1, 4, 2];

fn decode(stack: &mut NemotronStack<Fixture>, tokens: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    stack.prefill(&PROMPT).unwrap();
    let mut produced = vec![stack.last_prediction().unwrap()];
    let mut logits = vec![stack.logits().unwrap().to_vec()];
    while produced.len() < tokens {
        produced.push(stack.next_token().unwrap());
        logits.push(stack.logits().unwrap().to_vec());
    }
    (produced, logits)
}

fn cpu_only(threads: usize) -> HybridConfig {
    HybridConfig {
        cpu_threads: threads,
        gpu_layers: GpuLayerPlan::Fixed(0),
        gpu_batch: false,
    }
}

#[test]
fn cpu_experts_decode_like_the_sequential_packed_path() {
    let (want_tokens, want_logits) = decode(&mut stack(None), 6);
    let mut hybrid = stack(Some(cpu_only(3)));
    let (tokens, logits) = decode(&mut hybrid, 6);
    assert_eq!(tokens, want_tokens);
    let mut worst = 0.0_f32;
    for (row, want) in logits.iter().zip(&want_logits) {
        let peak = want.iter().fold(0.0_f32, |peak, v| peak.max(v.abs()));
        for (got, want) in row.iter().zip(want) {
            worst = worst.max((got - want).abs() / peak);
        }
    }
    // The kernels sum the same products in a different order; 1e-5 of the logit peak is loose
    // enough for that and far tighter than a missed expert or a wrong weight could hide in.
    assert!(worst < 1e-5, "logits differ by {worst} of peak");
}

#[test]
fn the_report_names_the_placement_and_splits_every_token() {
    let mut hybrid = stack(Some(cpu_only(2)));
    let steps = PROMPT.len() + 3;
    decode(&mut hybrid, 4);
    let report = hybrid.hybrid_report().expect("hybrid report");
    assert_eq!(report.cpu_threads, 2);
    assert_eq!(report.cpu_layers, vec![1, 3, 4]);
    assert!(report.gpu_layers.is_empty());
    assert_eq!(report.tokens.len(), steps);
    for split in &report.tokens {
        assert!(split.wall >= split.cpu_exposed);
        // Three CPU-served MoE layers per token, each timed on both threads.
        assert!(!split.cpu_routed.is_zero() && !split.shared.is_zero());
    }
    assert!(stack(None).hybrid_report().is_none());
}

#[test]
fn a_host_projector_cannot_hold_gpu_layers_but_auto_means_none_here() {
    let spec = spec();
    let state = NemotronState::new(&spec, 16).unwrap();
    let refused = NemotronStack::new_hybrid(
        spec.clone(),
        Fixture::new(),
        state.clone(),
        HybridConfig {
            cpu_threads: 1,
            gpu_layers: GpuLayerPlan::Fixed(1),
            gpu_batch: false,
        },
    );
    assert!(matches!(refused, Err(EngError::MemoryPlan(_))));
    let auto = NemotronStack::new_hybrid(
        spec,
        Fixture::new(),
        state,
        HybridConfig {
            cpu_threads: 1,
            gpu_layers: GpuLayerPlan::Auto,
            gpu_batch: false,
        },
    )
    .unwrap();
    assert!(auto.hybrid_report().unwrap().gpu_layers.is_empty());
}

#[test]
fn hybrid_decode_is_repeatable_and_independent_of_the_thread_count() {
    let (a, logits_a) = decode(&mut stack(Some(cpu_only(1))), 5);
    let (b, logits_b) = decode(&mut stack(Some(cpu_only(4))), 5);
    assert_eq!(a, b);
    // The kernel's results are bit-identical across thread counts, so the logits must be too.
    assert_eq!(logits_a, logits_b);
}

#[test]
fn hybrid_chunked_prefill_matches_sequential_bit_for_bit() {
    // The chunked path (per-row KV attention + one batched pool call per
    // MoE layer) must reproduce the sequential hybrid path exactly. The
    // 12-token prompt at rows 16 exceeds the pool's MAX_BATCH_ROWS (8),
    // covering the multi-group path.
    let long_prompt = [1, 4, 2, 0, 1, 2, 4, 1, 0, 2, 1, 4];
    for (rows, prompt) in [
        (2, &PROMPT[..]),
        (3, &PROMPT[..]),
        (16, &PROMPT[..]),
        (16, &long_prompt[..]),
    ] {
        let mut sequential = stack(Some(cpu_only(3)));
        let mut chunked = stack(Some(cpu_only(3)));
        sequential.prefill(prompt).unwrap();
        chunked.prefill_chunked(prompt, rows).unwrap();
        assert_eq!(chunked.state().position(), sequential.state().position());
        assert_eq!(chunked.last_prediction(), sequential.last_prediction());
        assert_eq!(chunked.logits(), sequential.logits());
        for _ in 0..3 {
            assert_eq!(
                chunked.next_token().unwrap(),
                sequential.next_token().unwrap(),
                "rows {rows}"
            );
        }
        assert_eq!(chunked.logits(), sequential.logits(), "rows {rows}");
    }
}

fn guard_inputs(rows: usize) -> Vec<f32> {
    (0..rows * LATENT as usize)
        .map(|index| ((index % 17) as f32 - 8.0) * 1024.0)
        .collect()
}

fn relu_squared_peak(
    fixture: &Fixture,
    xs: &[f32],
    picks: &[Vec<crate::expert_tier::ExpertPick>],
) -> f32 {
    let mut peak = 0.0_f32;
    for (x, picks) in xs.as_chunks::<{ LATENT as usize }>().0.iter().zip(picks) {
        for pick in picks {
            let &(packed, scales, factor, shape) = &fixture.experts[&(1, pick.expert, false)];
            let matrix =
                Nvfp4Matrix::new(packed, scales, factor, shape[0] as usize, shape[1] as usize)
                    .unwrap();
            let mut up = vec![0.0; shape[0] as usize];
            matrix
                .matvec(crate::kernels::nvfp4_cpu::Isa::Scalar, x, &mut up)
                .unwrap();
            for value in up {
                peak = peak.max(value.max(0.0).powi(2));
            }
        }
    }
    peak
}

#[test]
fn guard_fixture_exceeds_f16_before_the_row_guard() {
    let xs = guard_inputs(12);
    let picks = vec![
        (0..EXPERTS)
            .map(|expert| crate::expert_tier::ExpertPick {
                expert,
                weight: 1.0
            })
            .collect();
        12
    ];
    assert!(xs.iter().all(|v| v.is_finite() && v.abs() <= 65_504.0));
    let peak = relu_squared_peak(&Fixture::new(), &xs, &picks);
    assert!(
        peak.is_finite() && peak > 65_504.0,
        "pre-guard relu2 peak={peak}"
    );
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod cuda {
    use super::*;
    use crate::family::nemotron35::admission::{
        self, ACTIVATION_WORKSPACE_BYTES,
        cuda::{CudaAdmission, CudaPlanInput, admit_cuda_plan},
    };
    use crate::memory::MemoryPolicy;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// Serializes the bodies of the two hybrid CUDA tests against each other when the lib tests
    /// run on several threads. Each test holds one stack, and so one adapter lease, at a time,
    /// so together they hold at most one of the two process slots; that leaves the other slot
    /// for the remaining CUDA tests of the binary and lowers the peak device memory. A panic
    /// while a stack is live latches the backend for the process, and the lock does not
    /// prevent that. The lock does not cover those other tests either; the documented GPU run
    /// uses `--test-threads=1`. A test that panics while holding the lock must not fail the
    /// other with a poisoned lock.
    fn gpu() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[test]
    fn configured_stream_admission_is_bounded_and_default_off() {
        const CASE: &str = "FIFI_NEM35_STREAM_ADMISSION_CASE";
        const NAME: &str = "family::nemotron35::exec::stack::hybrid_tests::cuda::configured_stream_admission_is_bounded_and_default_off";
        if let Ok(case) = std::env::var(CASE) {
            let mut fixture = Fixture::new();
            if case == "malformed" {
                fixture.experts.get_mut(&(3, 0, false)).unwrap().3[0] += 16;
            }
            let mut ticket = admission();
            let before = ticket.device_weight_cache_bytes();
            let capacity = if case == "large" { 30_000 } else { 16 };
            let result = super::super::streaming::reserve_streaming(
                &spec(),
                &mut fixture,
                capacity,
                &mut ticket,
            );
            if case == "malformed" {
                assert!(result.is_err());
                assert!(ticket.stream_plan().is_none());
                assert_eq!(ticket.device_weight_cache_bytes(), before);
            } else {
                result.unwrap();
                let plan = ticket.stream_plan();
                if matches!(case.as_str(), "valid" | "capacity") {
                    let plan = plan.expect("stream admission");
                    assert_eq!(plan.max_pairs, 16 * 3);
                    assert_eq!(plan.up_shape, [INTER as usize, LATENT as usize]);
                } else {
                    assert!(plan.is_none(), "{case} unexpectedly reserved streaming");
                    assert_eq!(ticket.device_weight_cache_bytes(), before);
                }
            }
            return;
        }
        for (case, flag, rows) in [
            ("unset", "unset", "16"),
            ("cpu", "cpu", "16"),
            ("no-chunk", "gpu", "0"),
            ("small", "gpu", "8"),
            ("large", "gpu", "30000"),
            ("valid", "gpu", "16"),
            ("capacity", "gpu", "1024"),
            ("malformed", "gpu", "16"),
        ] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", NAME])
                .env(CASE, case)
                .env("NEM35_CUDA_CHUNK_PREFILL", rows)
                .env_remove("NEM35_CHUNK_EXPERTS");
            if flag != "unset" {
                command.env("NEM35_CHUNK_EXPERTS", flag);
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{case}: status={} stdout={} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn chunk_dispatch_reaches_device_only_when_opted_in() {
        const CASE: &str = "FIFI_NEM35_CHUNK_DISPATCH_CASE";
        const NAME: &str = "family::nemotron35::exec::stack::hybrid_tests::cuda::chunk_dispatch_reaches_device_only_when_opted_in";
        let _gpu = super::super::gpu_test_lock();
        if !device_present() {
            println!("SKIPPED: no CUDA device");
            return;
        }
        if let Ok(case) = std::env::var(CASE) {
            let state = NemotronState::new(&spec(), 16).unwrap();
            let mut stack = NemotronStack::new_cuda_hybrid(
                spec(),
                Fixture::new(),
                state,
                admission(),
                cpu_only(2),
            )
            .unwrap();
            stack.prefill_chunked(&[1; 12], 16).unwrap();
            assert_eq!(stack.state().position(), 12);
            assert!(stack.last_prediction().is_some());
            assert_eq!(
                stack.hybrid.as_ref().unwrap().stream_banks.is_some(),
                case == "gpu"
            );
            return;
        }
        for case in ["unset", "cpu", "gpu"] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", NAME, "--nocapture"])
                .env(CASE, case)
                .env("NEM35_CUDA_CHUNK_PREFILL", "16")
                .env_remove("NEM35_CHUNK_EXPERTS");
            if case != "unset" {
                command.env("NEM35_CHUNK_EXPERTS", case);
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{case}: status={} stdout={} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(!String::from_utf8_lossy(&output.stdout).contains("SKIPPED"));
        }
    }

    fn streaming_admission(fixture: &mut Fixture, rows: usize) -> CudaAdmission {
        let mut ticket = admission();
        let plan = super::super::streaming::stream_plan(fixture, 1, EXPERTS, rows * 3).unwrap();
        ticket.reserve_streaming(plan).unwrap();
        ticket
    }

    fn admission() -> CudaAdmission {
        let mut fixture = Fixture::new();
        let mut total = 0_u64;
        let mut largest = 0_u64;
        for tensor in fixture.dense.values() {
            let bytes = tensor.host_bytes().unwrap().len() as u64;
            total += bytes;
            largest = largest.max(bytes);
        }
        let shared = fixture
            .expert(1, ExpertIndex::Shared, ExpertProjection::Up)
            .unwrap();
        largest = largest.max(shared.host_bytes().unwrap().len() as u64);
        let state_bytes = NemotronState::planned_bytes(&spec(), 16).unwrap() * 3;
        let host_base_bytes = ACTIVATION_WORKSPACE_BYTES + total * 2 + state_bytes + largest * 4;
        admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes,
                largest_tensor_bytes: largest,
                // 32 x 32 nibbles plus their group scales.
                packed_pair_bytes: 32 * 32 / 2 + 32 * 32 / 16,
                max_shape: [38, 32],
                device_weight_cache_bytes: 1 << 20,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap()
    }

    fn cuda_stack(config: Option<HybridConfig>) -> NemotronStack<Fixture> {
        let spec = spec();
        let state = NemotronState::new(&spec, 16).unwrap();
        match config {
            Some(config) => {
                NemotronStack::new_cuda_hybrid(spec, Fixture::new(), state, admission(), config)
                    .unwrap()
            }
            None => NemotronStack::new_cuda(spec, Fixture::new(), state, admission()).unwrap(),
        }
    }

    fn tokens(stack: &mut NemotronStack<Fixture>, count: usize) -> Vec<u32> {
        stack.prefill(&PROMPT).unwrap();
        let mut produced = vec![stack.last_prediction().unwrap()];
        while produced.len() < count {
            produced.push(stack.next_token().unwrap());
        }
        produced
    }

    fn device_present() -> bool {
        crate::kernels::cuda::device_count().expect("CUDA device count query") > 0
    }

    #[test]
    fn gpu_resident_layers_decode_like_cpu_layers_and_upload_nothing_per_token() {
        let _gpu = gpu();
        if !device_present() {
            println!("SKIPPED: no CUDA device");
            return;
        }
        let want = tokens(&mut cuda_stack(None), 10);
        assert!(
            want.iter().any(|&token| token != want[0]),
            "the fixture must not decode one repeated token: {want:?}"
        );
        // The adapter allows two live leases per process, so each stack goes before the next.
        let mut all_cpu = cuda_stack(Some(cpu_only(2)));
        assert_eq!(tokens(&mut all_cpu, 10), want);
        drop(all_cpu);

        let mut mixed = cuda_stack(Some(HybridConfig {
            cpu_threads: 2,
            gpu_layers: GpuLayerPlan::Fixed(2),
            gpu_batch: false,
        }));
        assert_eq!(tokens(&mut mixed, 10), want);
        let report = mixed.hybrid_report().unwrap();
        // The last two MoE layers are resident, the first one is CPU-served.
        assert_eq!(report.gpu_layers, vec![3, 4]);
        assert_eq!(report.cpu_layers, vec![1]);
        // 2 layers x 8 experts x 2 projections x (512 packed + 64 scale bytes).
        assert_eq!(report.gpu_resident_bytes, 2 * 8 * 2 * 576);
        for split in &report.tokens {
            assert!(
                !split.gpu_routed.is_zero(),
                "resident layers ran on the device"
            );
        }
        // After the first token every dense weight is resident, and no routed expert is ever
        // uploaded again: the steady state moves no weight bytes at all.
        let steady = &report.tokens[PROMPT.len()..];
        assert!(
            steady.iter().all(|split| split.upload_bytes == 0),
            "{steady:?}"
        );
        drop(mixed);

        // The opt-in batched kernel agrees at this size (it does not at production shapes).
        let mut batched = cuda_stack(Some(HybridConfig {
            cpu_threads: 2,
            gpu_layers: GpuLayerPlan::Fixed(2),
            gpu_batch: true,
        }));
        assert_eq!(tokens(&mut batched, 10), want);
        drop(batched);

        // All MoE layers resident is also a valid placement.
        let mut resident = cuda_stack(Some(HybridConfig {
            cpu_threads: 1,
            gpu_layers: GpuLayerPlan::Fixed(3),
            gpu_batch: false,
        }));
        assert_eq!(tokens(&mut resident, 10), want);
        assert!(resident.hybrid_report().unwrap().cpu_layers.is_empty());
    }

    #[test]
    fn more_gpu_layers_than_moe_layers_is_refused() {
        let _gpu = gpu();
        if !device_present() {
            println!("SKIPPED: no CUDA device");
            return;
        }
        let spec = spec();
        let state = NemotronState::new(&spec, 16).unwrap();
        let refused = NemotronStack::new_cuda_hybrid(
            spec,
            Fixture::new(),
            state,
            admission(),
            HybridConfig {
                cpu_threads: 1,
                gpu_layers: GpuLayerPlan::Fixed(4),
                gpu_batch: false,
            },
        );
        assert!(matches!(refused, Err(EngError::BadRequest(_))));
    }

    #[test]
    fn chunk_device_stream_matches_the_cpu_pool_when_demand_grows() {
        check_chunk_device_stream(INTER);
    }

    #[test]
    fn chunk_device_stream_admits_payload_above_one_mib() {
        let intermediate = 1 << 16;
        let part_bytes = intermediate as usize * LATENT as usize * 9 / 16;
        assert!(part_bytes > 1 << 20);
        check_chunk_device_stream(intermediate);
    }

    #[test]
    fn rectangular_chunk_device_stream_matches_the_cpu_pool_within_tolerance() {
        for intermediate in [LATENT * 2, LATENT / 2] {
            check_chunk_device_stream(intermediate);
        }
    }

    fn check_chunk_device_stream(intermediate: u32) {
        let _gpu = super::super::gpu_test_lock();
        if !device_present() {
            println!("SKIPPED: no CUDA device");
            return;
        }
        // 16 rows x 3 picks exercises the streaming arm (> STREAM_ROWS);
        // layer 1 is the first MoE layer of the 5-block fixture.
        let rows = 16_usize;
        let mut xs = Vec::with_capacity(rows * LATENT as usize);
        for index in 0..rows * LATENT as usize {
            xs.push(((index % 17) as f32 - 8.0) * 0.25);
        }
        let picks: Vec<Vec<crate::expert_tier::ExpertPick>> = (0..rows)
            .map(|row| {
                (0..3)
                    .map(|slot| crate::expert_tier::ExpertPick {
                        expert: ((row * 3 + slot) as u32) % EXPERTS,
                        weight: 0.1 + 0.1 * slot as f32,
                    })
                    .collect()
            })
            .collect();
        let union: std::collections::BTreeSet<_> =
            picks.iter().flatten().map(|pick| pick.expert).collect();
        assert_eq!(union.len(), EXPERTS as usize);
        let mut spec = spec();
        spec.moe.moe_intermediate_size = intermediate;
        let state = NemotronState::new(&spec, 16).unwrap();
        let mut fixture = Fixture::with_intermediate(intermediate);
        let ticket = streaming_admission(&mut fixture, rows);
        let mut stack =
            NemotronStack::new_cuda_hybrid(spec, fixture, state, ticket, cpu_only(2)).unwrap();
        let super::super::NemotronStack {
            hybrid,
            projector,
            source,
            ..
        } = &mut stack;
        let hybrid = hybrid.as_mut().expect("hybrid runtime");
        let got = match projector {
            super::super::StackProjector::Cuda(cuda) => {
                let first = vec![
                    vec![crate::expert_tier::ExpertPick {
                        expert: 0,
                        weight: 1.0,
                    }];
                    rows
                ];
                super::super::streaming::gpu_stream_routed(
                    cuda, hybrid, source, 1, EXPERTS, &xs, &first,
                )
                .expect("the first chunk streams one expert");
                super::super::streaming::gpu_stream_routed(
                    cuda, hybrid, source, 1, EXPERTS, &xs, &picks,
                )
                .expect("the next chunk streams all experts");
                for down in [false, true] {
                    let key = (1, EXPERTS - 1, down);
                    let shape = source.experts[&key].3;
                    source.experts.get_mut(&key).unwrap().3[0] += 16;
                    let mismatched = super::super::streaming::gpu_stream_routed(
                        cuda, hybrid, source, 1, EXPERTS, &xs, &picks,
                    );
                    source.experts.get_mut(&key).unwrap().3 = shape;
                    assert!(mismatched.is_err(), "mismatched part shape was accepted");
                }
                let mut invalid_picks = picks.clone();
                invalid_picks[0][0].weight = f32::NAN;
                assert!(
                    super::super::streaming::gpu_stream_routed(
                        cuda,
                        hybrid,
                        source,
                        1,
                        EXPERTS,
                        &xs,
                        &invalid_picks,
                    )
                    .is_err()
                );
                let mut invalid = xs.clone();
                invalid[0] = f32::NAN;
                assert!(
                    super::super::streaming::gpu_stream_routed(
                        cuda, hybrid, source, 1, EXPERTS, &invalid, &picks,
                    )
                    .is_err()
                );
                let factor = source.experts[&(1, 0, false)].2;
                source.experts.get_mut(&(1, 0, false)).unwrap().2 = 1e22;
                assert!(!relu_squared_peak(source, &xs, &first).is_finite());
                let overflow = super::super::streaming::gpu_stream_routed(
                    cuda, hybrid, source, 1, EXPERTS, &xs, &first,
                );
                source.experts.get_mut(&(1, 0, false)).unwrap().2 = factor;
                assert!(
                    overflow.is_err(),
                    "activation overflow must not be published"
                );
                super::super::streaming::gpu_stream_routed(
                    cuda, hybrid, source, 1, EXPERTS, &xs, &picks,
                )
                .expect("bank/context reuse after numeric rejection")
            }
            _ => panic!("the cuda fixture must hold a CUDA projector"),
        };
        let mut want = vec![0.0; xs.len()];
        let pick_refs: Vec<&[crate::expert_tier::ExpertPick]> =
            picks.iter().map(|row| row.as_slice()).collect();
        stack
            .hybrid
            .as_ref()
            .unwrap()
            .cpu
            .routed_batch(1, &xs, &pick_refs, &mut want)
            .expect("the CPU pool serves the same request");
        assert_eq!(got.len(), want.len());
        assert!(got.iter().all(|value| value.is_finite()));
        // The grouped kernel is an F16-operand approximation of the CPU
        // pool's F32 path; a scaled worst-case bound, not bit equality.
        let scale = want.iter().fold(1e-6_f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst / scale <= 2e-2,
            "stream disagrees with the pool: worst {worst} on scale {scale}"
        );
    }

    #[test]
    fn chunk_device_stream_guards_relu_squared_intermediates_beyond_f16() {
        let _gpu = super::super::gpu_test_lock();
        if !device_present() {
            println!("SKIPPED: no CUDA device");
            return;
        }
        // Up operands fit F16, but relu2 does not; prove the guard is required.
        let rows = 12_usize;
        let xs = guard_inputs(rows);
        let picks: Vec<Vec<crate::expert_tier::ExpertPick>> = (0..rows)
            .map(|row| {
                (0..3)
                    .map(|slot| crate::expert_tier::ExpertPick {
                        expert: ((row * 3 + slot) as u32) % EXPERTS,
                        weight: 0.1 + 0.1 * slot as f32,
                    })
                    .collect()
            })
            .collect();
        let mut fixture = Fixture::new();
        let peak = relu_squared_peak(&fixture, &xs, &picks);
        assert!(
            peak.is_finite() && peak > 65_504.0,
            "pre-guard relu2 peak={peak}"
        );
        let ticket = streaming_admission(&mut fixture, rows);
        let state = NemotronState::new(&spec(), rows as u32).unwrap();
        let mut stack =
            NemotronStack::new_cuda_hybrid(spec(), fixture, state, ticket, cpu_only(2)).unwrap();
        let super::super::NemotronStack {
            hybrid,
            projector,
            source,
            ..
        } = &mut stack;
        let hybrid = hybrid.as_mut().expect("hybrid runtime");
        let got = match projector {
            super::super::StackProjector::Cuda(cuda) => super::super::streaming::gpu_stream_routed(
                cuda, hybrid, source, 1, EXPERTS, &xs, &picks,
            )
            .expect("the guarded device arm streams the layer"),
            _ => panic!("the cuda fixture must hold a CUDA projector"),
        };
        let mut want = vec![0.0; xs.len()];
        let pick_refs: Vec<&[crate::expert_tier::ExpertPick]> =
            picks.iter().map(|row| row.as_slice()).collect();
        stack
            .hybrid
            .as_ref()
            .unwrap()
            .cpu
            .routed_batch(1, &xs, &pick_refs, &mut want)
            .expect("the CPU pool serves the same request");
        let scale = want.iter().fold(1e-6_f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst / scale <= 2e-2,
            "the guarded stream disagrees with the pool: worst {worst} on scale {scale}"
        );
    }
}
