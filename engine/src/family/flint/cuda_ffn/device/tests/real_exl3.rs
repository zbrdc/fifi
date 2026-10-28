//! Opt-in resident diagnostic. Every observation is NOT SAME-PACK ORACLE.

use super::*;
use crate::family::{
    flint::{
        cuda_ffn::{device::runtime::State, real_model, reference::real::gguf},
        exl3_resident::{FlintResidentLimits, load_flint_resident},
        kv_plan::KvPlan,
    },
    project::cuda::Reserve,
};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

mod comparison;
mod decode;
mod projection_checks;
mod tests;

const CAPACITY: usize = 32;
const DECODE_STEPS: usize = 16;
const VOCAB: usize = 150144;
const CASES: [(&str, &[u32]); 2] = [
    ("p01", &[630, 6963, 297, 9865, 332]),
    ("code01", &[1182, 836, 3444, 11, 286, 1080, 279, 597]),
];
const P01_GGUF_LOGITS: &str = "/media/dan/nvme/fifi/scrch/d5cba1dd/flint-full49-5/teacher-native-full49-01d7df368e1f/actual/p01-logits.f32le";
const P01_GGUF_SHA256: &str = "c6360d1a1a66ee7e01e4308f177ee4bc97243632b793049ce0828aea9b0a9964";
const HOST_BRIDGE: &str = "device Q/K normalization and RoPE; GPU CPF; host F16 K/V conversion and write; NOT an all-device graph";

fn cached_head(bytes: &[u8], expected_sha256: &str) -> Result<Vec<f32>, EngError> {
    if bytes.len() != VOCAB * 4 || format!("{:x}", Sha256::digest(bytes)) != expected_sha256 {
        return Err(bad("cached GGUF head length or SHA-256 mismatch"));
    }
    let logits = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect::<Vec<_>>();
    comparison::greedy(&logits)?;
    Ok(logits)
}

#[test]
#[ignore = "requires admitted GPU, full EXL3 pack and cached p01 head; NOT SAME-PACK ORACLE"]
fn resident_full49_first_token_and_sixteen_decode() {
    let started = Instant::now();
    let pack = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in EXL3 pack"));
    let output = real_model::fresh_directory(
        "FIFI_FLINT_CUDA_RESULT_DIR",
        &[pack.clone(), PathBuf::from(P01_GGUF_LOGITS)],
    );
    let cuda_started = Instant::now();
    let ctx = CudaContext::new().expect("EXL3 full49 requires CUDA");
    let stream = ctx.stream().unwrap();
    let cuda_init_ms = cuda_started.elapsed().as_secs_f64() * 1000.0;
    let before = ctx.memory_info().unwrap();
    let plan = KvPlan::new(CAPACITY, 1, false).unwrap();
    println!(
        "{}",
        serde_json::json!({"stage":"EXL3 load start","label":comparison::LABEL})
    );
    let load_started = Instant::now();
    let model = load_flint_resident(
        &ctx,
        &stream,
        &pack,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: before.free as u64,
            reserve: Reserve {
                kv_cache: plan.bytes(4 * 128 * 2 * 2).unwrap() as u64,
                work: (256_u64 << 20)
                    .checked_add(State::scratch_bytes(&plan).unwrap())
                    .unwrap(),
                headroom: 1 << 30,
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let weights_free = ctx.memory_info().unwrap().free;
    let load_report = serde_json::json!({
        "stage":"EXL3 load complete","label":comparison::LABEL,
        "cuda_init_ms":cuda_init_ms,"load_ms":load_ms,"layers":model.layer_count(),
        "resident_weight_bytes":model.device_weight_bytes(),"free_before":before.free,
        "free_after_weights":weights_free,"free_samples_are_not_peak":true,
        "execution_bridge":HOST_BRIDGE,"not_a_benchmark":true,
    });
    real_model::write_new(&output.join("load.json"), &load_report);
    println!("{load_report}");
    let kv_started = Instant::now();
    let mut decode = decode::ResidentDecode::new(&ctx, &stream, &model, CAPACITY).unwrap();
    let kv_init_ms = kv_started.elapsed().as_secs_f64() * 1000.0;
    let kv_free = ctx.memory_info().unwrap().free;
    let device_started = Instant::now();
    let mut heads = Vec::with_capacity(CASES.len());
    for (name, prompt) in CASES {
        decode.reset().unwrap();
        let prefill_started = Instant::now();
        let mut logits = Vec::new();
        for &token in prompt {
            logits = decode.step(token, None).unwrap();
        }
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        let first_token_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let head_bytes = logits
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let head_sha256 = format!("{:x}", Sha256::digest(&head_bytes));
        let first_id = comparison::greedy(&logits).unwrap();
        real_model::write_new(
            &output.join(format!("{name}-exl3-head.json")),
            &serde_json::json!({
                "label":comparison::LABEL,"case":name,"input_ids":prompt,"head_logits":logits,
                "head_sha256_f32le":head_sha256,"first_id":first_id,
                "prefill_ms_including_host_bridge_not_benchmark":prefill_ms,
                "first_token_wall_ms_since_start":first_token_wall_ms,
                "execution_bridge":HOST_BRIDGE,"cpu_fp64_checks_during_device_stage":false,
            }),
        );
        heads.push(logits.clone());
        let decode_started = Instant::now();
        let mut ids = Vec::with_capacity(DECODE_STEPS + 1);
        for step in 0..=DECODE_STEPS {
            let token = comparison::greedy(&logits).unwrap();
            ids.push(token);
            if step < DECODE_STEPS {
                logits = decode.step(token, None).unwrap();
            }
        }
        let report = serde_json::json!({
            "label":comparison::LABEL,"stage":"EXL3 device case complete","case":name,
            "input_ids":prompt,"greedy_ids":ids,"head_sha256_f32le":head_sha256,
            "prefill_ms_including_host_bridge_not_benchmark":prefill_ms,
            "decode_16_ms_including_host_bridge_not_benchmark":decode_started.elapsed().as_secs_f64()*1000.0,
            "capacity":CAPACITY,"physical_kv":"F16","layers":model.layer_count(),
            "resident_weight_bytes":model.device_weight_bytes(),"free_after_kv":kv_free,
            "free_after_decode":ctx.memory_info().unwrap().free,"free_samples_are_not_peak":true,
            "execution_bridge":HOST_BRIDGE,"cpu_fp64_checks_during_device_stage":false,
            "gguf_comparison":if name=="p01" {"cached first head only; later stage"} else {"unavailable; no code01 artifact"},
            "qualification":"diagnostic only; no full-model oracle or quality bound",
        });
        real_model::write_new(&output.join(format!("{name}-device.json")), &report);
        println!("{report}");
    }
    let device_ms = device_started.elapsed().as_secs_f64() * 1000.0;

    // Replay the sample positions only after both device trajectories are durable.
    println!(
        "{}",
        serde_json::json!({"stage":"CPU ideal-FP64 replay start","label":comparison::LABEL})
    );
    let cpu_started = Instant::now();
    let mut checks = projection_checks::CpuChecks::open(&pack).unwrap();
    for ((name, prompt), head) in CASES.into_iter().zip(&heads) {
        decode.reset().unwrap();
        checks.case = name;
        for (position, &token) in prompt.iter().enumerate() {
            checks.position = position;
            let sample = (position + 1 == prompt.len()).then_some(&mut checks);
            decode.step(token, sample).unwrap();
        }
        checks.position = prompt.len();
        decode
            .step(comparison::greedy(head).unwrap(), Some(&mut checks))
            .unwrap();
    }
    let checks_completed = checks.count;
    let cpu_fp64_replay_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
    real_model::write_new(
        &output.join("cpu-checks.json"),
        &serde_json::json!({"label":comparison::LABEL,"checks_completed":checks_completed,
            "cpu_fp64_replay_ms":cpu_fp64_replay_ms,"reference":"ideal FP64, NOT FP16-operand-rounded","threshold":null}),
    );
    let drop_started = Instant::now();
    drop(checks);
    drop(decode);
    drop(model);
    stream.synchronize().unwrap();
    let resident_drop_ms = drop_started.elapsed().as_secs_f64() * 1000.0;

    // This fixed, unqualified cached head never starts a live GGUF decoder.
    let comparison_started = Instant::now();
    let cached = cached_head(&std::fs::read(P01_GGUF_LOGITS).unwrap(), P01_GGUF_SHA256).unwrap();
    let comparison = comparison::compare(&cached, &heads[0]).unwrap();
    let report = serde_json::json!({
        "label":comparison::LABEL,"case":"p01","input_ids":CASES[0].1,
        "first_id_agrees":comparison.first_id_agrees,"kl_gguf_to_exl3":comparison.kl_gguf_to_exl3,
        "gguf_first_id":comparison::greedy(&cached).unwrap(),"exl3_first_id":comparison::greedy(&heads[0]).unwrap(),
        "first_divergence":null,"gguf_decode_ids_available":false,
        "comparison_scope":"cached first head only; no GGUF continuation artifact",
        "cached_path":P01_GGUF_LOGITS,"cached_sha256":P01_GGUF_SHA256,"cached_bytes":VOCAB*4,
        "cached_capture_head":"ff0279cdf4a45163eda9325b4283162bfff65240",
        "cached_capture_exit_code":101,"cached_capture_passes":6,"cached_capture_cases":8,
        "cached_capture_qualification":false,"gguf_model_sha256":gguf::MODEL_SHA256,"physical_kv":"F16",
        "execution_bridge":HOST_BRIDGE,"qualification":"diagnostic reference ONLY; NOT same-pack or full quality",
    });
    real_model::write_new(&output.join("p01-comparison.json"), &report);
    println!("{report}");
    let times = serde_json::json!({
        "label":comparison::LABEL,"cuda_init_ms":cuda_init_ms,"load_ms":load_ms,
        "kv_init_ms":kv_init_ms,"device_cases_ms":device_ms,"cpu_fp64_replay_ms":cpu_fp64_replay_ms,
        "resident_drop_ms":resident_drop_ms,"cached_comparison_ms":comparison_started.elapsed().as_secs_f64()*1000.0,
        "total_wall_ms":started.elapsed().as_secs_f64()*1000.0,"not_a_benchmark":true,"execution_bridge":HOST_BRIDGE,
    });
    real_model::write_new(&output.join("stage-times.json"), &times);
    println!("{times}");
    assert_eq!(checks_completed, CASES.len() * 2 * (49 * 4 + 1));
    assert!(
        comparison.first_id_agrees,
        "NOT SAME-PACK ORACLE: first ID differs; diagnostic data retained"
    );
}
