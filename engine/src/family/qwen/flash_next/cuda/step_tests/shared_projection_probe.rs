//! Matched-input diagnostics for native GGUF shared-expert projections.
use super::*;
use crate::{
    family::{qwen::flash_next::cuda::step::project_q6_f32, step::rows::NativeRow},
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, NativeWeight,
        prefill::gemm_dq::GemmDqTypedCall,
        resident::{matvec_q::QuantType, quantize_q8_1::q8_1_bytes},
    },
};

fn all(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

fn read_f32(stream: &DeviceStream, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_ne_bytes(*x))
        .collect()
}

fn event_ms(ctx: &CudaContext, stream: &DeviceStream, run: impl Fn()) -> f64 {
    for _ in 0..5 {
        run();
    }
    stream.synchronize().unwrap();
    let start = ctx.event_timing().unwrap();
    let end = ctx.event_timing().unwrap();
    stream.record_timing(&start).unwrap();
    for _ in 0..40 {
        run();
    }
    stream.record_timing(&end).unwrap();
    stream.synchronize().unwrap();
    f64::from(start.elapsed_ms_since(&end).unwrap()) / 40.0
}

pub(super) fn projection(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &NativeWeight,
    input: &[f32],
    expected: &[f32],
) -> serde_json::Value {
    let (k, rows) = (weight.k(), weight.n_rows());
    assert_eq!((input.len(), expected.len()), (k, rows));
    assert!(weight.experts().is_none());
    let x = ctx.allocate(k * 4).unwrap();
    let y = ctx.allocate(rows * 4).unwrap();
    let q8 = ctx.allocate(q8_1_bytes(k, 1).unwrap()).unwrap();
    x.write_at(
        stream,
        0,
        &input
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let row = NativeRow {
        weight,
        input: all(&x),
        output: all(&y),
    };
    let native = || row.quantized(ctx, stream, all(&q8)).unwrap();
    let f16 = || {
        ctx.gemm_dq_codebook(
            stream,
            weight.quant(),
            GemmDqTypedCall {
                w: all(weight.buffer()),
                x: all(&x),
                y: all(&y),
                k,
                rows,
                ncols: 1,
                x_col_stride: 0,
                y_col_stride: 0,
                x_f32: true,
            },
        )
        .unwrap()
    };
    let f32 = || project_q6_f32(ctx, stream, row).unwrap();
    native();
    let native_output = read_f32(stream, &y);
    let mut q8_bytes = vec![0; q8.bytes()];
    q8.read_at(stream, 0, &mut q8_bytes).unwrap();
    f16();
    let f16_output = read_f32(stream, &y);
    let f32_output = if weight.quant() == QuantType::Q6K {
        f32();
        Some(read_f32(stream, &y))
    } else {
        assert_eq!(weight.quant(), QuantType::Q80);
        None
    };
    let native_error = relative_l2(&native_output, expected);
    let f16_error = relative_l2(&f16_output, expected);
    assert!(native_error < 0.03 && f16_error < 0.003);
    if let Some(ref output) = f32_output {
        assert!(relative_l2(output, expected) < 0.00001);
    }
    let native_ms = event_ms(ctx, stream, native);
    let f16_ms = event_ms(ctx, stream, f16);
    let f32_ms = if f32_output.is_some() {
        Some(event_ms(ctx, stream, f32))
    } else {
        None
    };
    assert!(native_ms > 0.0 && f16_ms > 0.0 && f32_ms.is_none_or(|x| x > 0.0));
    serde_json::json!({"k":k,"rows":rows,"native_output":native_output,"q8_bytes":q8_bytes,
        "f16_output":f16_output,"f32_output":f32_output,"native_error":native_error,
        "f16_error":f16_error,"native_ms":native_ms,"f16_ms":f16_ms,"f32_ms":f32_ms})
}

#[test]
#[ignore = "requires an admitted CUDA device"]
fn gguf_shared_projection_synthetic_q8_pilot() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let uploaded = crate::family::qwen::flash_next::load::gguf::tests::synthetic_native_head(&ctx);
    let CudaWeight::Native { weight, .. } = uploaded else {
        panic!("synthetic Q8_0 head is not native")
    };
    let result = projection(&ctx, &stream, &weight, &vec![1.0; 256], &[256.0; 4]);
    assert!(result["f16_error"].as_f64().unwrap() < 1e-6);
    eprintln!("synthetic Q8_0 shared projection pilot: {result}");
}

#[test]
#[ignore = "requires verified pinned GGUF, layer-0 host capture and an admitted CUDA device"]
fn gguf_shared_projection_matched_inputs() {
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (uploaded, input, want) =
        crate::family::qwen::flash_next::load::gguf::tests::pinned_shared_host_reference(&ctx);
    let weights = CudaWeights::from_native(uploaded).unwrap();
    let mut results = serde_json::Map::new();
    for (part, x, y) in [
        ("gate", &input, &want[0]),
        ("up", &input, &want[1]),
        ("down", &want[2], &want[3]),
    ] {
        let name = layer_name(0, &format!("mlp.shared_expert.{part}_proj"));
        let Some(CudaWeight::Native { weight, .. }) = weights.get(&name) else {
            panic!("missing {name}")
        };
        let value = projection(&ctx, &stream, weight, x, y);
        eprintln!(
            "shared matched {part}: native={} f16={} native_ms={} f16_ms={} f32_ms={}",
            value["native_error"],
            value["f16_error"],
            value["native_ms"],
            value["f16_ms"],
            value["f32_ms"]
        );
        results.insert(part.into(), value);
    }
    let path =
        std::env::var_os("FIFI_FLASH_GGUF_SHARED_MATCHED_OUT").expect("set fresh receipt path");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(file, &serde_json::json!({"layer":0,"results":results})).unwrap();
}
