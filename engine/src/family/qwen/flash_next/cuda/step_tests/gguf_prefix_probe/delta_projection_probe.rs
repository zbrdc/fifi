//! Test-only matched-input DeltaNet projections; no recurrent state or full model execution.
use super::*;
use crate::{
    family::{
        qwen::flash_next::cuda::step::{project_q6_f16, project_q6_f32},
        step::rows::NativeRow,
    },
    kernels::cuda::{
        DeviceBuffer,
        resident::{
            quantize_q8_1::q8_1_bytes, router::RouterShape, router_rows::RouterLogitsStridedCall,
        },
    },
};

fn capture_input(path: &Path) -> Vec<f32> {
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut reference: Option<Vec<f32>> = None;
    for name in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
    ] {
        let mut matches = rows.iter().filter(|r| r["name"] == name);
        let row = matches.next().unwrap_or_else(|| panic!("missing {name}"));
        assert!(matches.next().is_none(), "duplicate {name}");
        assert!(row["expert"].is_null(), "unexpected expert tag for {name}");
        let input: Vec<f32> = serde_json::from_value(row["input"].clone()).unwrap();
        assert!(!input.is_empty() && input.iter().all(|v| v.is_finite()));
        if let Some(prior) = &reference {
            assert_eq!(input, *prior, "captured inputs differ for {name}");
        } else {
            reference = Some(input);
        }
    }
    reference.unwrap()
}

#[test]
fn delta_projection_capture_requires_all_unique_matching_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("capture");
    let row = |name: &str, input: f32| format!("{{\"name\":\"{name}\",\"input\":[{input}]}}\n");
    let names = [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
    ];
    let valid: String = names.iter().map(|n| row(n, 1.0)).collect();
    std::fs::write(&p, &valid).unwrap();
    assert_eq!(capture_input(&p), vec![1.0]);
    std::fs::write(&p, valid.clone() + &row(names[0], 1.0)).unwrap();
    assert!(std::panic::catch_unwind(|| capture_input(&p)).is_err());
    std::fs::write(&p, valid.replace(&row(names[3], 1.0), &row(names[3], 2.0))).unwrap();
    assert!(std::panic::catch_unwind(|| capture_input(&p)).is_err());
    std::fs::write(&p, valid.replace(&row(names[2], 1.0), "")).unwrap();
    assert!(std::panic::catch_unwind(|| capture_input(&p)).is_err());
    let tagged = row(names[0], 1.0).replace("\"input\"", "\"expert\":4,\"input\"");
    std::fs::write(&p, valid.replace(&row(names[0], 1.0), &tagged)).unwrap();
    assert!(std::panic::catch_unwind(|| capture_input(&p)).is_err());
}

fn q6_results(
    step: &FlashStep<'_>,
    input: &DeviceBuffer,
    name: &str,
    rows: usize,
) -> serde_json::Value {
    let Some(CudaWeight::Native { weight, .. }) = step.weights.get(name) else {
        panic!("missing native {name}")
    };
    assert_eq!(weight.k() * 4, input.bytes());
    assert_eq!(weight.n_rows(), rows);
    let out = step.ctx.allocate(rows * 4).unwrap();
    let scratch = step.slot(Slot::GgufDecodeQ8).unwrap();
    let x = input.range(0, input.bytes()).unwrap();
    let y = out.range(0, out.bytes()).unwrap();
    let poison = f32::NAN.to_ne_bytes().repeat(rows);
    out.write_at(&step.stream, 0, &poison).unwrap();
    step.project(name, x, y, 1).unwrap();
    let q8_output = step.read_f32(y, rows).unwrap();
    let q8_len = q8_1_bytes(weight.k(), 1).unwrap();
    let mut q8 = vec![0; q8_len];
    scratch
        .buffer()
        .read_at(&step.stream, scratch.offset(), &mut q8)
        .unwrap();
    let mut alternatives = Vec::new();
    for f16 in [true, false] {
        out.write_at(&step.stream, 0, &poison).unwrap();
        let call = NativeRow {
            weight,
            input: x,
            output: y,
        };
        if f16 {
            project_q6_f16(&step.ctx, &step.stream, call).unwrap();
        } else {
            project_q6_f32(&step.ctx, &step.stream, call).unwrap();
        }
        alternatives.push(step.read_f32(y, rows).unwrap());
    }
    assert!(
        q8_output
            .iter()
            .chain(alternatives.iter().flatten())
            .all(|v| v.is_finite())
    );
    assert!(q8_output.iter().any(|v| *v != 0.0));
    let run = |mode| {
        let call = NativeRow {
            weight,
            input: x,
            output: y,
        };
        match mode {
            "q8" => step.project(name, x, y, 1).unwrap(),
            "f16" => project_q6_f16(&step.ctx, &step.stream, call).unwrap(),
            "f32" => project_q6_f32(&step.ctx, &step.stream, call).unwrap(),
            _ => unreachable!(),
        }
    };
    let mut timing = Vec::new();
    for mode in ["q8", "f16", "f32", "f32", "f16", "q8"] {
        for _ in 0..5 {
            run(mode);
        }
        step.stream.synchronize().unwrap();
        let start = step.ctx.event_timing().unwrap();
        let end = step.ctx.event_timing().unwrap();
        step.stream.record_timing(&start).unwrap();
        for _ in 0..40 {
            run(mode);
        }
        step.stream.record_timing(&end).unwrap();
        step.stream.synchronize().unwrap();
        let ms = f64::from(start.elapsed_ms_since(&end).unwrap()) / 40.0;
        assert!(ms.is_finite() && ms > 0.0);
        eprintln!(
            "matched Delta projection timing: {name} mode={mode} ms_per_call={ms:.6} repeat=40 (resident weights; excludes upload and readback)"
        );
        timing.push((mode, ms));
    }
    let mut last_q8 = vec![0; q8_len];
    scratch
        .buffer()
        .read_at(&step.stream, scratch.offset(), &mut last_q8)
        .unwrap();
    assert_eq!(last_q8, q8);
    serde_json::json!({"name": name, "rows": rows, "q8_bytes": q8,
        "q8": q8_output, "f16": alternatives[0], "f32": alternatives[1], "timing_ms":timing})
}

fn float_result(
    step: &FlashStep<'_>,
    input: &DeviceBuffer,
    name: &str,
    rows: usize,
) -> serde_json::Value {
    let out = step.ctx.allocate(rows * 4).unwrap();
    out.write_at(&step.stream, 0, &f32::NAN.to_ne_bytes().repeat(rows))
        .unwrap();
    step.ctx
        .router_logits_rows_strided(
            &step.stream,
            RouterLogitsStridedCall {
                shape: RouterShape {
                    m: 1,
                    hidden: step.geometry.hidden,
                    n_expert: rows,
                    top_k: 1,
                },
                x: input.range(0, input.bytes()).unwrap(),
                w: step.float(name).unwrap(),
                logits: out.range(0, out.bytes()).unwrap(),
                logit_stride: rows,
            },
        )
        .unwrap();
    let actual = step
        .read_f32(out.range(0, out.bytes()).unwrap(), rows)
        .unwrap();
    assert!(actual.iter().all(|v| v.is_finite()) && actual.iter().any(|v| *v != 0.0));
    serde_json::json!({"name":name,"rows":rows,"f32":actual})
}

#[test]
#[ignore = "requires verified pinned GGUF and admitted CUDA device"]
fn pinned_layer_zero_delta_projections_match_host_input() {
    assert!(
        std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_DELTA_IN").is_none(),
        "disable live input ablation in matched control"
    );
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("verified first shard");
    let capture = std::env::var_os("FIFI_FLASH_GGUF_HOST_CAPTURE").expect("verified host capture");
    let output = std::env::var_os("FIFI_FLASH_GGUF_DELTA_PROJECTION_OUT").expect("new output path");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    assert_pinned_file(
        &capture,
        "b0873cedfb9733049335904d64635d20b155e68614de0166a287333f5b5640e6",
    );
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let values = capture_input(Path::new(&capture));
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights =
        CudaWeights::from_native(model.upload_layer_trunk(0, &ctx, &stream).unwrap()).unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        model.config().model_shape().unwrap(),
        model.config().kinds(),
        1,
        model.map().max_quantized_input_width().unwrap(),
        0..1,
    )
    .unwrap();
    step.begin_token().unwrap();
    assert_eq!(values.len(), step.geometry.hidden);
    let input_bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let input = ctx.upload(&input_bytes).unwrap();
    let mut readback = vec![0; input_bytes.len()];
    input.read_at(&stream, 0, &mut readback).unwrap();
    assert_eq!(
        readback, input_bytes,
        "GPU input must exactly match host capture"
    );
    let d = step.geometry.delta;
    let qkv = q6_results(
        &step,
        &input,
        &layer_name(0, "linear_attn.in_proj_qkv"),
        d.conv_dim,
    );
    let z = q6_results(
        &step,
        &input,
        &layer_name(0, "linear_attn.in_proj_z"),
        d.value_width(),
    );
    assert_eq!(
        qkv["q8_bytes"], z["q8_bytes"],
        "same input must yield same Q8 codes"
    );
    let beta = float_result(
        &step,
        &input,
        &layer_name(0, "linear_attn.in_proj_b.weight"),
        d.value_heads,
    );
    let alpha = float_result(
        &step,
        &input,
        &layer_name(0, "linear_attn.in_proj_a.weight"),
        d.value_heads,
    );
    let receipt = serde_json::json!({"hidden":step.geometry.hidden, "key_heads":d.key_heads,
        "key_dim":d.key_dim, "value_heads":d.value_heads, "value_dim":d.value_dim,
        "qkv":qkv,"z":z,"beta":beta,"alpha":alpha});
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer(file, &receipt).unwrap();
    eprintln!(
        "matched DeltaNet projections captured qkv={} z={} beta={} alpha={}; not a propagated run",
        d.conv_dim,
        d.value_width(),
        d.value_heads,
        d.value_heads
    );
}
