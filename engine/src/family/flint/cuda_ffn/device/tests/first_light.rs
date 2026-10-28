//! Private canonical-layer fixture; not a production executor or a timing benchmark.
//! Q/K are zero, so head norm and rope are fixed points. The KV bridge materializes
//! projected BF16 values as physical F16; replace it with the future shared D128 producer.

use super::*;
use crate::kernels::cuda::prefill::attn_prefill::{
    AttnKvStorage, AttnPrefillCall, AttnPrefillShape, CPF_ATTN_SM_COUNT, CpfAttnGeometry,
    CpfAttnSpan, CpfAttnSplit, cpf_attn_launch_plan, cpf_attn_part_bytes,
};
use crate::kernels::cuda::resident::add_norm_pack::AddNormCall;
use crate::kernels::cuda::xf_bytes;
use reference::layer::{self, INTERMEDIATE, WIDTH};

const ROWS: usize = 4;
const HEAD_DIM: usize = 128;
const HEADS: usize = WIDTH / HEAD_DIM;
const SHAPE: RowShape = RowShape {
    rows: ROWS,
    columns: WIDTH,
};

fn layer_input() -> Vec<f32> {
    let mut input = vec![0.0; ROWS * WIDTH];
    for token in 0..ROWS {
        input[token * WIDTH + token * (WIDTH / ROWS)] = 1.0;
    }
    input
}

fn layer_expert(ctx: &CudaContext, down_factor: f32) -> OwnedExpert {
    OwnedExpert {
        gate: Weight::new(ctx, [INTERMEDIATE, WIDTH], UNIT),
        up: Weight::new(
            ctx,
            [INTERMEDIATE, WIDTH],
            Diagonal {
                factor: 0.5,
                ..UNIT
            },
        ),
        down: Weight::new(
            ctx,
            [WIDTH, INTERMEDIATE],
            Diagonal {
                factor: down_factor,
                ..UNIT
            },
        ),
    }
}

fn read_rows(stream: &DeviceStream, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_ne_bytes(*value))
        .collect())
}

fn normalized(
    ctx: &CudaContext,
    stream: &DeviceStream,
    x: &DeviceBuffer,
) -> Result<DeviceBuffer, EngError> {
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(x.range(0, x.bytes())?, SHAPE)?;
    let raw = frame.allocate(SHAPE)?;
    let weight = ctx.upload(&f32_bytes(&[1.0; WIDTH]))?;
    let packed =
        ctx.allocate(xf_bytes(ROWS, WIDTH).ok_or_else(|| bad("first-light norm scratch"))?)?;
    ctx.add_norm_pack(
        stream,
        AddNormCall {
            m: ROWS,
            n: WIDTH,
            eps: 1e-6,
            h: frame.range(x)?,
            delta: None,
            weight: weight.range(0, weight.bytes())?,
            xf: packed.range(0, packed.bytes())?,
            norm_out: Some(frame.range(raw)?),
        },
    )?;
    let published = frame.publish(raw, SHAPE)?;
    frame.finish(published)
}

fn projection(
    ctx: &CudaContext,
    stream: &DeviceStream,
    x: &DeviceBuffer,
    weight: &Weight,
) -> Result<DeviceBuffer, EngError> {
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(x.range(0, x.bytes())?, SHAPE)?;
    let weights = vec![weight.view(ctx); ROWS];
    let plan = ctx.nvfp4_grouped_plan(&weights)?;
    let output = frame.project(&plan, x, ROWS, [WIDTH, WIDTH])?;
    frame.finish(output)
}

fn residual(
    ctx: &CudaContext,
    stream: &DeviceStream,
    x: &DeviceBuffer,
    delta: &DeviceBuffer,
) -> Result<DeviceBuffer, EngError> {
    let mut frame = Frame::new(ctx, stream)?;
    let out = frame.allocate(SHAPE)?;
    ctx.row_add(
        stream,
        RowAddCall {
            x: x.range(0, x.bytes())?,
            y: delta.range(0, delta.bytes())?,
            out: frame.range(out)?,
            n: ROWS * WIDTH,
        },
    )?;
    let published = frame.publish(out, SHAPE)?;
    frame.finish(published)
}

fn physical_f16(
    ctx: &CudaContext,
    stream: &DeviceStream,
    source: &DeviceBuffer,
) -> Result<DeviceBuffer, EngError> {
    let values = read_rows(stream, source)?;
    let mut payload = Vec::with_capacity(values.len() * 2);
    for value in values {
        let stored = half::f16::from_f32(value);
        if !value.is_finite() || value.to_bits() & 0xffff != 0 || stored.to_f32() != value {
            return Err(bad("first-light KV bridge requires FP16-exact BF16 values"));
        }
        payload.extend_from_slice(&stored.to_bits().to_ne_bytes());
    }
    ctx.upload(&payload)
}

fn attention(
    ctx: &CudaContext,
    stream: &DeviceStream,
    qkv: [&DeviceBuffer; 3],
) -> Result<DeviceBuffer, EngError> {
    let [q, k, v] = qkv;
    for projected in [q, k] {
        if read_rows(stream, projected)?
            .iter()
            .any(|&value| value != 0.0)
        {
            return Err(bad("first-light head norm/rope requires zero Q/K"));
        }
    }
    let k = physical_f16(ctx, stream, k)?;
    let v = physical_f16(ctx, stream, v)?;
    let geometry = CpfAttnGeometry {
        m: ROWS,
        n_head: HEADS,
        n_kv_head: HEADS,
        head_dim: HEAD_DIM,
    };
    let plan = cpf_attn_launch_plan(geometry, ROWS, 0, CPF_ATTN_SM_COUNT)?;
    let part = ctx.allocate(cpf_attn_part_bytes(geometry, plan)?)?;
    let sinks = ctx.upload(&f32_bytes(&[0.0; HEADS]))?;
    let mut frame = Frame::new(ctx, stream)?;
    let out = frame.allocate(SHAPE)?;
    ctx.attn_prefill(
        stream,
        AttnPrefillCall {
            storage: AttnKvStorage::F16,
            kv_views: None,
            shape: AttnPrefillShape {
                m: ROWS,
                n_head: HEADS,
                n_kv_head: HEADS,
                head_dim: HEAD_DIM,
                cache_rows: ROWS,
                ld: 0,
                p0: 0,
            },
            span: CpfAttnSpan::FULL,
            split: CpfAttnSplit {
                max_splits: plan.max_splits,
                min_keys: plan.min_keys,
            },
            q_tile: plan.q_tile,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            q: q.range(0, q.bytes())?,
            proj: None,
            k_cache: k.range(0, k.bytes())?,
            v_cache: v.range(0, v.bytes())?,
            part: part.range(0, part.bytes())?,
            out: frame.range(out)?,
            status: Some(frame.flag()?),
            sinks: Some(sinks.range(0, sinks.bytes())?),
        },
    )?;
    let published = frame.publish(out, SHAPE)?;
    frame.finish(published)
}

fn device_layer(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    weights: &FfnWeights<'_>,
) -> Result<DeviceBuffer, EngError> {
    let norm = normalized(ctx, stream, input)?;
    let zero = Weight::new(
        ctx,
        [WIDTH, WIDTH],
        Diagonal {
            code: 0,
            decoded: 0.0,
            ..UNIT
        },
    );
    let identity = Weight::new(ctx, [WIDTH, WIDTH], UNIT);
    let q = projection(ctx, stream, &norm, &zero)?;
    let k = projection(ctx, stream, &norm, &zero)?;
    let v = projection(ctx, stream, &norm, &identity)?;
    let attended = attention(ctx, stream, [&q, &k, &v])?;
    let projected = projection(ctx, stream, &attended, &identity)?;
    let h = residual(ctx, stream, input, &projected)?;
    let norm = normalized(ctx, stream, &h)?;
    let ffn = forward(
        ctx,
        stream,
        FfnCall {
            weights,
            x: norm.range(0, norm.bytes())?,
            rows: ROWS,
        },
    )?;
    residual(ctx, stream, &h, &ffn)
}

#[test]
#[ignore = "requires an explicit coordinated GPU window"]
fn dense_whole_layer_matches_independent_bf16_oracle() {
    let expected = layer::run(&layer::weights(), &layer_input(), ROWS, Kind::Dense).unwrap();
    assert_eq!(expected[0], 137.0);
    let actual = {
        let ctx = CudaContext::new().expect("first light requires a real CUDA device");
        let stream = ctx.stream().unwrap();
        let dense = layer_expert(&ctx, 1.0);
        let input = ctx.upload(&f32_bytes(&layer_input())).unwrap();
        let output =
            device_layer(&ctx, &stream, &input, &FfnWeights::Dense(dense.view(&ctx))).unwrap();
        read_rows(&stream, &output)
    }
    .unwrap();
    assert!(
        layer::same_bf16(&actual, &expected),
        "{actual:?} != {expected:?}"
    );
    println!(
        "FIRST_LIGHT dense device=required T=4 H=256 D=128 Hq=KVh=2 KV=physicalF16 exact_bf16_match=true"
    );
}

#[test]
#[ignore = "requires an explicit coordinated GPU window"]
fn moe_whole_layer_matches_independent_bf16_oracle() {
    let expected = layer::run(&layer::weights(), &layer_input(), ROWS, Kind::Moe).unwrap();
    assert_eq!(expected[0], 209.0);
    let actual = {
        let ctx = CudaContext::new().expect("first light requires a real CUDA device");
        let stream = ctx.stream().unwrap();
        let experts: Vec<_> = (0..8)
            .map(|id| layer_expert(&ctx, 1.0 + id as f32 / 8.0))
            .collect();
        let views: Vec<_> = experts.iter().map(|expert| expert.view(&ctx)).collect();
        let shared = layer_expert(&ctx, 0.25);
        let router = ctx.upload(&f32_bytes(&[0.0; 8 * WIDTH])).unwrap();
        let input = ctx.upload(&f32_bytes(&layer_input())).unwrap();
        let output = device_layer(
            &ctx,
            &stream,
            &input,
            &FfnWeights::Moe {
                router: router.range(0, router.bytes()).unwrap(),
                experts: &views,
                shared: shared.view(&ctx),
            },
        )
        .unwrap();
        read_rows(&stream, &output)
    }
    .unwrap();
    assert!(
        layer::same_bf16(&actual, &expected),
        "{actual:?} != {expected:?}"
    );
    println!(
        "FIRST_LIGHT moe device=required T=4 H=256 D=128 Hq=KVh=2 KV=physicalF16 exact_bf16_match=true"
    );
}

#[test]
#[ignore = "requires an explicit coordinated GPU window"]
fn dense_then_moe_matches_each_independent_bf16_boundary() {
    let expected = layer::dense_then_moe(&layer::weights(), &layer_input(), ROWS).unwrap();
    assert_eq!(expected[0][0], 137.0);
    assert_eq!(expected[1][0], 344.0);
    let ctx = CudaContext::new().expect("layer loop requires a real CUDA device");
    let stream = ctx.stream().unwrap();
    let dense = layer_expert(&ctx, 1.0);
    let experts: Vec<_> = (0..8)
        .map(|id| layer_expert(&ctx, 1.0 + id as f32 / 8.0))
        .collect();
    let views: Vec<_> = experts.iter().map(|expert| expert.view(&ctx)).collect();
    let shared = layer_expert(&ctx, 0.25);
    let router = ctx.upload(&f32_bytes(&[0.0; 8 * WIDTH])).unwrap();
    let weights = [
        FfnWeights::Dense(dense.view(&ctx)),
        FfnWeights::Moe {
            router: router.range(0, router.bytes()).unwrap(),
            experts: &views,
            shared: shared.view(&ctx),
        },
    ];
    let mut activation = ctx.upload(&f32_bytes(&layer_input())).unwrap();
    for (index, (weights, expected)) in weights.iter().zip(expected).enumerate() {
        activation = device_layer(&ctx, &stream, &activation, weights).unwrap();
        let actual = read_rows(&stream, &activation).unwrap();
        assert!(
            layer::same_bf16(&actual, &expected),
            "layer {index} BF16 boundary: {actual:?} != {expected:?}"
        );
    }
    println!(
        "LAYER_LOOP dense_moe device=required layers=2 T=4 H=256 D=128 Hq=KVh=2 KV=physicalF16 every_boundary_exact_bf16_match=true"
    );
}
