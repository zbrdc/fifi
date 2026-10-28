//! Full49 is opt-in. Attention states stay separate; only final head rows are batched.

use super::*;
use crate::family::flint::{cuda_ffn::real_model, gguf_weights};
use crate::weights::GgufType;

pub(in crate::family::flint::cuda_ffn::device::tests) fn head(
    ctx: &CudaContext,
    stream: &DeviceStream,
    source: (&File, &GgufHeader),
    states: &[DeviceBuffer],
    last_row_only: bool,
) -> Result<Vec<f32>, EngError> {
    let (file, header) = source;
    let gamma = gguf_weights::required(header, "output_norm.weight", GgufType::F32, &[2048])?;
    let gamma = gguf::plain(file, gamma).map_err(|e| bad(&e))?;
    let mut head_rows = Vec::with_capacity(states.len() * WIDTH);
    for state in states {
        let norm = normalized(ctx, stream, state, &gamma)?;
        let values = values(stream, &norm)?;
        let start = if last_row_only {
            values.len() - WIDTH
        } else {
            0
        };
        head_rows.extend_from_slice(&values[start..]);
    }
    let rows = head_rows.len() / WIDTH;
    // This disclosed host bridge copies actual CUDA outputs, never reference hidden states.
    let input = ctx.upload(&f32_bytes(&head_rows))?;
    let source = gguf_weights::matrix_source(header, "output", [150144, WIDTH])?;
    let weight = weights::Weight::load(ctx, file, &source)?;
    let descriptors = vec![weight.view(ctx)?; rows];
    let plan = ctx.nvfp4_grouped_plan(&descriptors)?;
    let mut frame = Frame::new(ctx, stream)?;
    let slot = frame.input(
        input.range(0, input.bytes())?,
        RowShape {
            rows,
            columns: WIDTH,
        },
    )?;
    let output = frame.project_f32(&plan, slot, rows, weight.shape)?;
    values(stream, &frame.finish(output)?)
}

#[test]
#[ignore = "requires fifi-gpu-run admission, pinned full49 oracle and GGUF"]
fn full49_eight_prefixes_match_shared_layers_and_llama_first_token() {
    run(true);
}

#[test]
#[ignore = "requires admitted GPU; cumulative/head diagnostic, never shared-layer qualification"]
fn full49_eight_prefixes_cumulative_diagnostic_only() {
    run(false);
}

fn run(check_shared: bool) {
    let prefixes = real_model::prefixes();
    let path = std::env::var("FIFI_FLINT_FULL_ORACLE").expect("prepared full49 oracle");
    let sha = std::env::var("FIFI_FLINT_FULL_ORACLE_SHA256").expect("frozen full49 oracle SHA256");
    let oracle = real_model::Oracle::parse(&std::fs::read(&path).unwrap(), &sha).unwrap();
    let (file, header, identity) = real_model::model();
    let root = real_model::fresh_directory("FIFI_FLINT_CUDA_RESULT_DIR", &[PathBuf::from(&path)]);
    let ctx = CudaContext::new().expect("full49 requires CUDA");
    let stream = ctx.stream().unwrap();
    let mut previous = prefixes
        .iter()
        .map(|p| real_model::embedding(&file, &header, &p.input_ids))
        .collect::<Vec<_>>();
    for (case, input) in previous.iter().enumerate() {
        assert_eq!(
            input.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            oracle.layers[0][case].input_bits
        );
    }
    let mut states = previous
        .iter()
        .map(|input| ctx.upload(&f32_bytes(input)).unwrap())
        .collect::<Vec<_>>();
    println!("FULL49_BEGIN cases=8 layers=49 physical_kv=F16 host_qk_bridge=true post_unlock=true");
    for (index, reference) in oracle.layers.iter().enumerate() {
        let layer =
            weights::Layer::load(&ctx, &file, &header, index as u32, 5 + index % 2).unwrap();
        for (case, prefix) in prefixes.iter().enumerate() {
            let expected = reference[case]
                .output_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>();
            if check_shared {
                let shared_input = reference[case]
                    .input_bits
                    .iter()
                    .map(|&v| f32::from_bits(v))
                    .collect::<Vec<_>>();
                let shared_input = ctx.upload(&f32_bytes(&shared_input)).unwrap();
                let shared_output = device_layer(&ctx, &stream, &shared_input, &layer).unwrap();
                let shared = values(&stream, &shared_output).unwrap();
                assert_eq!(shared.len(), prefix.input_ids.len() * WIDTH);
                assert!(
                    shared
                        .iter()
                        .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
                );
                real_model::write_new(
                    &root.join(format!("shared-{index}-{}.json", prefix.case)),
                    &serde_json::json!({
                        "layer":index,"case":prefix.case,"rows":prefix.input_ids.len(),"shared_input":true,
                        "physical_kv":"F16","cache_rows":layer.cache_rows,"sliding":layer.sliding,"ring":0,"p0":0,
                        "model_sha256":gguf::MODEL_SHA256,"oracle_sha256":sha,"host_qk_bridge":true,
                        "input_bits":reference[case].input_bits,"output_bits":shared.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    }),
                );
                println!("FULL49_SHARED layer={index} case={}", prefix.case);
                real_quality::compare(&expected, &shared, WIDTH).unwrap();
            }

            assert_eq!(
                values(&stream, &states[case])
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                previous[case]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            let output = device_layer(&ctx, &stream, &states[case], &layer).unwrap();
            let actual = values(&stream, &output).unwrap();
            assert_eq!(actual.len(), prefix.input_ids.len() * WIDTH);
            assert!(
                actual
                    .iter()
                    .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
            );
            real_model::write_new(
                &root.join(format!("chain-{index}-{}.json", prefix.case)),
                &serde_json::json!({
                    "layer":index,"case":prefix.case,"rows":prefix.input_ids.len(),"shared_input":false,"diagnostic_only":true,
                    "physical_kv":"F16","cache_rows":layer.cache_rows,"sliding":layer.sliding,"ring":0,"p0":0,
                    "model_sha256":gguf::MODEL_SHA256,"oracle_sha256":sha,"host_qk_bridge":true,
                    "input_bits":previous[case].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "output_bits":actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                }),
            );
            let diagnostic = real_quality::compare(&expected, &actual, WIDTH);
            println!(
                "FULL49_CUMULATIVE layer={index} case={} quality={diagnostic:?}",
                prefix.case
            );
            previous[case] = actual;
            states[case] = output;
        }
        gguf::verify_prepared_identity(&file, &identity).unwrap();
    }
    let logits = head(&ctx, &stream, (&file, &header), &states, true).unwrap();
    assert_eq!(logits.len(), prefixes.len() * 150144);
    assert!(logits.iter().all(|v| v.is_finite()));
    let mut first_tokens = Vec::with_capacity(prefixes.len());
    for (prefix, row) in prefixes.iter().zip(logits.as_chunks::<150144>().0) {
        // Llama argmax keeps the first equal maximum.
        let token = row
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .unwrap()
            .0;
        first_tokens.push(token);
        let bytes = row.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let mut out = File::options()
            .write(true)
            .create_new(true)
            .open(root.join(format!("{}-logits.f32le", prefix.case)))
            .unwrap();
        out.write_all(&bytes).unwrap();
        out.sync_all().unwrap();
    }
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    real_model::write_new(
        &root.join("first-tokens.json"),
        &serde_json::json!({
            "model_sha256":gguf::MODEL_SHA256,"oracle_sha256":sha,"prefixes_sha256":real_model::PREFIX_SHA256,
            "llama_golden_sha256":real_model::GOLDEN_SHA256,"raw_argmax":first_tokens,"expected_raw_argmax":real_model::FIRST_TOKENS,
            "head_dtype":"F32","head_weight_dtype":"NVFP4","host_qk_bridge":true,"host_head_row_bridge":true,
            "shared_checks_executed":check_shared,"qualification":false,
            "full_serving_quality":"NOT_EVALUATED: eight prefixes are not the frozen 2048 teacher-forced rows",
        }),
    );
    assert_eq!(first_tokens, real_model::FIRST_TOKENS);
    println!("FULL49_END first_tokens=8/8 post_unlock=true");
}

#[test]
#[ignore = "requires pinned real GGUF, CUDA and frozen 2049 teacher IDs"]
fn teacher_forced_prefill_2048_diagnostic() {
    use sha2::{Digest, Sha256};
    let _total = super::super::profile::wall("teacher_total");
    let setup = super::super::profile::wall("setup");

    let input_path = PathBuf::from(std::env::var("FIFI_FLINT_TEACHER_IDS").unwrap());
    let input_bytes = std::fs::read(&input_path).unwrap();
    let input_sha = "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0";
    assert_eq!(format!("{:x}", Sha256::digest(&input_bytes)), input_sha);
    let ids: Vec<u32> = serde_json::from_slice(&input_bytes).unwrap();
    assert_eq!(ids.len(), 2049);
    assert!(ids.iter().all(|&id| id < 150144));
    let root = real_model::fresh_directory(
        "FIFI_FLINT_CUDA_RESULT_DIR",
        std::slice::from_ref(&input_path),
    );
    let (file, header, identity) = real_model::model();
    let ctx = CudaContext::new().expect("teacher prefill requires CUDA");
    let stream = ctx.stream().unwrap();
    let embedding = real_model::embedding(&file, &header, &ids[..2048]);
    assert_eq!(embedding.len(), 2048 * WIDTH);
    let mut state = ctx.upload(&f32_bytes(&embedding)).unwrap();
    println!("TEACHER_BEGIN rows=2048 layers=49 physical_kv=F16 host_qk_bridge=true");
    drop(setup);
    let layers = super::super::profile::wall("layers_total");
    for index in 0..49 {
        let layer = {
            let _load = super::super::profile::wall("layer_load_inclusive");
            weights::Layer::load(&ctx, &file, &header, index, 2048).unwrap()
        };
        state = device_layer(&ctx, &stream, &state, &layer).unwrap();
        assert_eq!(state.bytes(), 2048 * WIDTH * 4);
        gguf::verify_prepared_identity(&file, &identity).unwrap();
        println!("TEACHER_LAYER_END layer={index} rows=2048");
    }
    drop(layers);
    let head_wall = super::super::profile::wall("head_inclusive");
    let logits = head(&ctx, &stream, (&file, &header), &[state], false).unwrap();
    drop(head_wall);
    assert_eq!(logits.len(), 2048 * 150144);
    assert!(logits.iter().all(|v| v.is_finite()));
    let logit_io = super::super::profile::wall("logit_serialize_write_fsync");
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(root.join("teacher-logits.f32le"))
        .unwrap();
    for row in logits.as_chunks::<150144>().0 {
        let bytes = row.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        output.write_all(&bytes).unwrap();
    }
    output.sync_all().unwrap();
    assert_eq!(output.metadata().unwrap().len(), 1_229_979_648);
    drop(logit_io);
    assert_eq!(std::fs::read(&input_path).unwrap(), input_bytes);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    real_model::write_new(
        &root.join("TEACHER-END.json"),
        &serde_json::json!({
            "schema_version":1,"capture_complete":true,"qualification":false,
            "rows":2048,"vocabulary":150144,"dtype":"F32LE","layers":49,
            "model_sha256":gguf::MODEL_SHA256,"token_ids_sha256":input_sha,
            "labels":"ids[1..2049]; row i consumes ids[0..=i]",
            "physical_kv":"F16","cache_rows":2048,"p0":0,"ring":0,
            "host_qk_bridge":true,"host_head_row_bridge":true,
            "shared_layer_gates_executed":false,
            "dense_misses":"five remain open; explained upstream-product difference; end-to-end teacher gate pending",
            "notes":"Full-vocabulary raw logits only; not serving qualification or a performance result."
        }),
    );
    println!("TEACHER_END rows=2048 bytes=1229979648");
}

#[test]
#[ignore = "one admitted instrumented teacher trial; exclusive coordinator window required"]
fn teacher_forced_prefill_2048_phase_profile() {
    let capture = super::super::profile::Capture::start();
    teacher_forced_prefill_2048_diagnostic();
    let report = capture.finish();
    assert!(report.gemm_calls.iter().sum::<u64>() > 0);
    let wall = report.wall_ns["teacher_total"];
    assert!(wall > 0);
    let mode = std::env::var("FLINT_CUDA_FFN_MMQ").unwrap_or_else(|_| "0".into());
    if mode == "0" {
        assert!(report.gemm_calls[3] > 0);
    }
    let output = PathBuf::from(std::env::var("FIFI_FLINT_CUDA_RESULT_DIR").unwrap());
    real_model::write_new(
        &output.join("PHASE-PROFILE.json"),
        &serde_json::json!({
            "schema_version":1, "mmq_mode":mode, "report":report,
            "gemm_phase_order":["other_attention_head", "dense_ffn", "shared_ffn", "routed_experts"],
            "routed_gemm_share_of_rust_teacher_wall":report.gemm_ms[3] * 1e6 / wall as f64,
            "event_scope":"CUDA events around grouped_enqueue, read after existing status completion; no new syncs",
            "overhead":"event_api_host_ns counts create/record/query/destroy; empty-pair device calibration in Q1 receipt; not subtracted",
            "wall_scope":"nested inclusive spans: do not sum parents and children; outer process wall measured by controller",
            "qualification":false, "default_flip":false,
            "mmq_control_note":"Frame mode0 phase tags only; mode1 is output-hash control, not a phase-share conclusion"
        }),
    );
}
