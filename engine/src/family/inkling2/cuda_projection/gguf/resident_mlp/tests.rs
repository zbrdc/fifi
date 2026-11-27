//! Device tests of the resident MLP chain on a synthetic `GgufCuda` built through `load`:
//! Q8_0 expert stacks, an F32 router and F32 controllers of one layer, with the host entry
//! `routed_expert_group` as the oracle. `load` chains the layer only under
//! `FIFI_INKLING_CUDA_BATCH_EXPERTS=1` and `FIFI_INKLING_CUDA_DEVICE_ROUTING=1`, which the
//! test run must set (edition 2024 forbids `set_var` and the crate denies unsafe code).

use std::{
    borrow::Cow,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use super::ResidentMlpRequest;
use crate::{
    EngError,
    family::{
        inkling2::{
            TextSpec,
            cpu_forward::MIX_EPS,
            cuda_projection::{
                DenseUpload,
                gguf::{
                    ControlledAttentionInput, ExpertLayer, ExpertStack, GgufCuda, GgufDenseUpload,
                    Options,
                    router::{MlpController, RouteInput},
                },
            },
        },
        mapped_file::MappedFile,
    },
    kernels::cuda::{DeviceBuffer, DeviceStream, device_present, resident::dense_gemv::DenseDtype},
    weights::GgufType,
};

const HIDDEN: usize = 32;
const EXPERTS: usize = 8;
const TOP_K: usize = 2;
const RANK: usize = 16;
const LAYER: u32 = 0;
const CAPS: [f32; 2] = [1.0, 2.0];
const SCALE: f32 = 1.0;
const EPS: f32 = 1e-5;

fn controller() -> [&'static str; 3] {
    ["down", "gain", "forget"].map(|part| match part {
        "down" => "model.llm.layers.0.controllers.mlp.down_weight",
        "gain" => "model.llm.layers.0.controllers.mlp.gain_weight",
        _ => "model.llm.layers.0.controllers.mlp.forget_weight",
    })
}

fn prefix() -> String {
    format!("model.llm.layers.{LAYER}.mlp")
}

fn routing_flags() -> bool {
    [
        "FIFI_INKLING_CUDA_BATCH_EXPERTS",
        "FIFI_INKLING_CUDA_DEVICE_ROUTING",
    ]
    .iter()
    .all(|name| std::env::var(name).as_deref() == Ok("1"))
}

fn spec() -> TextSpec {
    let config = serde_json::json!({
        "model_type": "inkling2_loop",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "eos_token_id": 200006,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        "text_config": {
            "model_type": "inkling2_loop_text", "model_max_length": 1048576,
            "hidden_size": HIDDEN, "num_hidden_layers": 3, "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 8,
            "sliding_window_size": 1024, "local_layer_ids": [0],
            "n_layers_per_attn_res_block": 1,
            "n_routed_experts": EXPERTS, "num_experts_per_tok": TOP_K, "n_shared_experts": 0,
            "intermediate_size": HIDDEN,
            "loop_config": {"core_start": 1, "core_end": 2, "num_passes": 2,
                            "gate_rank": RANK, "is_varloop": false}
        }
    });
    TextSpec::from_config(&serde_json::to_vec(&config).unwrap()).unwrap()
}

/// Q8_0 rows of one 32-sample block: an FP16 scale, then 32 signed samples.
fn q8_rows(rows: usize, phase: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|row| {
            0x2000u16.to_le_bytes().into_iter().chain(
                (0..32).map(move |col| (((row * 19 + col * 13 + phase) % 31) as i8 - 15) as u8),
            )
        })
        .collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn f32_matrix([rows, columns]: [usize; 2], period: usize) -> Vec<u8> {
    floats(
        &(0..rows * columns)
            .map(|i| (((i * 7 + 1) % period * 2) as i32 - period as i32) as f32 / 256.0)
            .collect::<Vec<_>>(),
    )
}

/// The host's `rms_norm_with_scale`: the normalized copy of `row`.
fn normalized(row: &[f32], eps: f32) -> Vec<f32> {
    let scale = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + eps)
        .sqrt()
        .recip();
    row.iter().map(|v| v * scale).collect()
}

/// One layer with Q8_0 experts over a zero file of the stack size, an F32 router and
/// controllers, and a dummy wide F32 projection that gives the arena room for the chain.
fn load() -> GgufCuda {
    static FILES: AtomicU32 = AtomicU32::new(0);
    assert!(
        routing_flags(),
        "the resident MLP test needs FIFI_INKLING_CUDA_BATCH_EXPERTS=1 and FIFI_INKLING_CUDA_DEVICE_ROUTING=1"
    );
    let gate = q8_rows(EXPERTS * 2 * HIDDEN, 5);
    let down = q8_rows(EXPERTS * HIDDEN, 9);
    let path: PathBuf = std::env::temp_dir().join(format!(
        "inkling-resident-mlp-{}-{}.bin",
        std::process::id(),
        FILES.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, vec![0u8; gate.len() + down.len()]).unwrap();
    let file = MappedFile::open(&path, (gate.len() + down.len()) as u64).unwrap();
    fs::remove_file(&path).unwrap();
    let scale_row = |base: f32| -> Vec<u8> {
        floats(
            &(0..EXPERTS)
                .map(|expert| base + (expert % 5) as f32 / 8.0)
                .collect::<Vec<_>>(),
        )
    };
    let gate_scales = scale_row(0.5);
    let down_scales = scale_row(1.5);
    let parts = [
        (gate.as_slice(), 0..gate.len(), gate_scales.as_slice()),
        (
            down.as_slice(),
            gate.len()..gate.len() + down.len(),
            down_scales.as_slice(),
        ),
    ]
    .map(|(bytes, file_range, tensor_scales)| ExpertStack {
        bytes,
        file_range,
        tensor_scales: Some(tensor_scales),
        shape: [EXPERTS as u32 * 2, HIDDEN as u32],
    });
    let layers = vec![ExpertLayer {
        layer: LAYER,
        parts,
    }];
    let [down_name, gain_name, forget_name] = controller();
    let [router, down, gain, forget, dummy] = [
        f32_matrix([EXPERTS, HIDDEN], 11),
        f32_matrix([RANK, 2 * HIDDEN], 9),
        f32_matrix([HIDDEN, RANK], 7),
        f32_matrix([HIDDEN, RANK], 5),
        f32_matrix([512, HIDDEN], 13),
    ];
    let dense = vec![
        GgufDenseUpload {
            matrix: DenseUpload {
                name: "model.llm.layers.0.mlp.gate.weight",
                codes: &router,
                shape: [EXPERTS, HIDDEN],
            },
            dtype: DenseDtype::F32,
        },
        GgufDenseUpload {
            matrix: DenseUpload {
                name: down_name,
                codes: &down,
                shape: [RANK, 2 * HIDDEN],
            },
            dtype: DenseDtype::F32,
        },
        GgufDenseUpload {
            matrix: DenseUpload {
                name: gain_name,
                codes: &gain,
                shape: [HIDDEN, RANK],
            },
            dtype: DenseDtype::F32,
        },
        GgufDenseUpload {
            matrix: DenseUpload {
                name: forget_name,
                codes: &forget,
                shape: [HIDDEN, RANK],
            },
            dtype: DenseDtype::F32,
        },
        GgufDenseUpload {
            matrix: DenseUpload {
                name: "model.llm.layers.0.dummy.weight",
                codes: &dummy,
                shape: [512, HIDDEN],
            },
            dtype: DenseDtype::F32,
        },
    ];
    GgufCuda::load(
        dense,
        &spec(),
        layers,
        &file,
        Options {
            direct: false,
            packed: true,
            group_projections: true,
            stream: false,
            budget: 1 << 30,
            cache_reserve: None,
            resident_attention: None,
            allow_device_routing: true,
            context_limit: 0,
        },
    )
    .unwrap()
}

/// The caller's rows: `[input | message | gamma | out: 7 | health: 7 | gate health: 7]`
/// and `[bias | status: 0]`.
struct Rows {
    stream: DeviceStream,
    rows: DeviceBuffer,
    words: DeviceBuffer,
}

impl Rows {
    fn new(context: &crate::kernels::cuda::CudaContext, input: &[f32], message: &[f32]) -> Self {
        let stream = context.stream().unwrap();
        let gamma: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let mut values = [input, message, &gamma].concat();
        values.extend([7.0; 2 * HIDDEN]);
        let rows = context.upload(&floats(&values)).unwrap();
        let bias: Vec<f32> = (0..EXPERTS).map(|i| i as f32 * 0.25 - 1.0).collect();
        let mut words = floats(&bias);
        words.extend_from_slice(&0i32.to_ne_bytes());
        let words = context.upload(&words).unwrap();
        Self {
            stream,
            rows,
            words,
        }
    }

    fn bias(&self) -> Vec<f32> {
        (0..EXPERTS).map(|i| i as f32 * 0.25 - 1.0).collect()
    }

    fn range(&self, row: usize, floats: usize) -> crate::kernels::cuda::DeviceRange<'_> {
        self.rows.range(row * HIDDEN * 4, floats * 4).unwrap()
    }

    fn request(&self) -> ResidentMlpRequest<'_> {
        ResidentMlpRequest {
            layer: LAYER,
            caps: CAPS,
            scale: SCALE,
            dtypes: [DenseDtype::from_gguf(GgufType::Q80).unwrap(); 2],
            controller: controller(),
            rank: RANK,
            eps: EPS,
            input: self.range(0, HIDDEN),
            pre_normed: None,
            message: self.range(1, HIDDEN),
            gamma: self.range(2, HIDDEN),
            bias: self.words.range(0, EXPERTS * 4).unwrap(),
            out: self.range(3, HIDDEN),
            health: self.range(4, 1),
            gate_health: self.range(4, 2).slice(4, 4).unwrap(),
            status: self.words.range(EXPERTS * 4, 4).unwrap(),
        }
    }

    /// Output row, the two health words and router status, read back behind the stream.
    fn read_out(&self) -> (Vec<f32>, [f32; 2], i32) {
        let mut out = vec![0u8; (HIDDEN + 2) * 4];
        self.rows
            .read_at(&self.stream, 3 * HIDDEN * 4, &mut out)
            .unwrap();
        let health = out.split_off(HIDDEN * 4);
        let mut status = [0u8; 4];
        self.words
            .read_at(&self.stream, EXPERTS * 4, &mut status)
            .unwrap();
        let out: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        (
            out,
            std::array::from_fn(|i| {
                f32::from_ne_bytes(health[4 * i..4 * i + 4].try_into().unwrap())
            }),
            i32::from_ne_bytes(status),
        )
    }
}

fn input() -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| ((i * 3) % 17) as f32 * 0.1 - 0.8)
        .collect()
}

fn message() -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| ((i * 5) % 23) as f32 * 0.07 - 0.5)
        .collect()
}

/// The oracle: the host entry on the same host rows, with the host's gate halves.
fn host_entry(cuda: &GgufCuda, input: &[f32], message: &[f32], bias: &[f32]) -> Vec<f32> {
    let gate = [normalized(input, MIX_EPS), normalized(message, MIX_EPS)].concat();
    let gamma: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let output = cuda
        .routed_expert_group(
            &prefix(),
            RouteInput {
                layer: LAYER,
                decode: true,
                input,
                caps: CAPS,
                scale: SCALE,
            },
            [DenseDtype::from_gguf(GgufType::Q80).unwrap(); 2],
            Some(MlpController {
                names: controller(),
                rank: RANK,
                eps: EPS,
                input: ControlledAttentionInput {
                    gate: &gate,
                    input,
                    message,
                    gamma: &gamma,
                },
            }),
            |_| Ok(Cow::Owned(bias.to_vec())),
        )
        .unwrap()
        .expect("the host entry chains the layer");
    assert_eq!(output.len(), HIDDEN);
    output
}

#[test]
#[ignore = "requires a CUDA device, FIFI_INKLING_CUDA_BATCH_EXPERTS=1 and FIFI_INKLING_CUDA_DEVICE_ROUTING=1"]
fn resident_mlp_chain_matches_the_host_entry_and_lands_in_the_callers_row() {
    if !device_present("Inkling resident MLP chain") {
        return;
    }
    let cuda = load();
    assert!(cuda.resident_mlp_supported(LAYER, controller(), [RANK, HIDDEN]));
    let (input, message) = (input(), message());
    let rows = Rows::new(cuda.context(), &input, &message);
    let expected = host_entry(&cuda, &input, &message, &rows.bias());
    cuda.enqueue_controlled_mlp(&rows.stream, rows.request())
        .unwrap_or_else(|e| panic!("{e:?}"));
    let (out, health, status) = rows.read_out();
    assert_eq!(
        (health, status),
        ([0.0, 7.0], 0),
        "health {health:?}, status {status}"
    );
    assert_eq!(out.len(), HIDDEN);
    for (i, (a, b)) in out.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() <= 1e-4, "out[{i}] = {a}, host entry {b}");
    }
    assert_ne!(out.first(), Some(&7.0), "the sentinel row is still there");
}

#[test]
#[ignore = "requires a CUDA device, FIFI_INKLING_CUDA_BATCH_EXPERTS=1 and FIFI_INKLING_CUDA_DEVICE_ROUTING=1"]
fn resident_mlp_chain_reports_a_nan_input_through_health_output_and_status() {
    if !device_present("Inkling resident MLP chain NaN") {
        return;
    }
    let cuda = load();
    let rows = Rows::new(cuda.context(), &[f32::NAN; HIDDEN], &message());
    cuda.enqueue_controlled_mlp(&rows.stream, rows.request())
        .unwrap();
    let (out, health, status) = rows.read_out();
    assert!(out.iter().all(|v| v.is_nan()), "out {out:?}");
    assert!(health.iter().all(|v| v.is_nan()), "health {health:?}");
    assert_ne!(status & 1, 0, "the router status {status}");
}

#[test]
#[ignore = "requires a CUDA device, FIFI_INKLING_CUDA_BATCH_EXPERTS=1 and FIFI_INKLING_CUDA_DEVICE_ROUTING=1"]
fn resident_mlp_refuses_before_any_launch_and_reports_support() {
    if !device_present("Inkling resident MLP chain rejections") {
        return;
    }
    let cuda = load();
    for (layer, names, [rank, hidden]) in [
        (LAYER, controller(), [RANK, HIDDEN]),
        (LAYER, controller(), [RANK, 2 * HIDDEN]),
        (LAYER, controller(), [2 * RANK, HIDDEN]),
        (LAYER, ["a", "b", "c"], [RANK, HIDDEN]),
        (LAYER + 1, controller(), [RANK, HIDDEN]),
    ] {
        assert_eq!(
            cuda.resident_mlp_supported(layer, names, [rank, hidden]),
            (layer, names, rank, hidden) == (LAYER, controller(), RANK, HIDDEN),
            "{layer} {names:?} [{rank}, {hidden}]"
        );
    }
    let rows = Rows::new(cuda.context(), &input(), &message());
    let refused = |request: ResidentMlpRequest<'_>, checkpoint: bool, expected: &str| {
        let enqueued = rows.stream.enqueued();
        let error = cuda
            .enqueue_controlled_mlp(&rows.stream, request)
            .unwrap_err();
        match &error {
            EngError::Checkpoint(message) if checkpoint => {
                assert!(message.contains(expected), "{error:?}")
            }
            EngError::BadRequest(message) if !checkpoint => {
                assert!(message.contains(expected), "{error:?}")
            }
            _ => panic!("{error:?}"),
        }
        assert_eq!(
            rows.stream.enqueued(),
            enqueued,
            "a rejected call enqueued work"
        );
        let (out, health, status) = rows.read_out();
        assert!(
            out.iter().all(|&v| v == 7.0) && health == [7.0; 2] && status == 0,
            "a rejected call wrote {out:?} / {health:?} / {status}"
        );
    };
    let mut request = rows.request();
    request.layer = LAYER + 1;
    refused(request, true, "no resident F32 CUDA router");
    let mut request = rows.request();
    request.dtypes = [DenseDtype::F32; 2];
    refused(request, true, "experts are not packed GGUF");
    let mut request = rows.request();
    request.bias = request
        .bias
        .slice(0, (EXPERTS - 1) * size_of::<f32>())
        .unwrap();
    refused(request, false, "controlled MLP router bias range");
    let mut request = rows.request();
    request.eps = f32::NAN;
    refused(request, false, "invalid CUDA controlled MLP epsilon");
    let mut request = rows.request();
    request.out = rows.range(3, HIDDEN - 1);
    refused(request, false, "controlled MLP output range");
    let mut request = rows.request();
    request.health = rows.range(3, 1);
    refused(request, false, "ranges overlap");
    let mut request = rows.request();
    request.controller = ["a", "b", "c"];
    refused(request, true, "no resident CUDA MLP controller");
}
