//! Opt-in real layers and sequential qualification. D128 norm/RoPE and KV storage use HOST.
//! Projection, learned RMS, attention, dense/routed FFN and residuals execute on CUDA.

mod chain;
pub(in crate::family::flint::cuda_ffn) mod decode;
pub(in crate::family::flint::cuda_ffn::device::tests) mod model;
mod rca;
mod routes;
pub(in crate::family::flint::cuda_ffn::device::tests) mod weights;

use super::*;
use crate::family::flint::cuda_ffn::{
    real_quality,
    reference::real::{self, gguf},
};
use crate::kernels::cuda::prefill::attn_prefill::{
    AttnKvStorage, AttnPrefillCall, AttnPrefillShape, CPF_ATTN_SM_COUNT, CpfAttnGeometry,
    CpfAttnSpan, CpfAttnSplit, cpf_attn_launch_plan, cpf_attn_part_bytes,
};
use crate::kernels::cuda::resident::add_norm_pack::AddNormCall;
use crate::kernels::cuda::xf_bytes;
use crate::weights::GgufHeader;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
};

const ROWS: usize = 4;
const WIDTH: usize = 2048;

fn row_shape(input: &DeviceBuffer, columns: usize) -> Result<RowShape, EngError> {
    Ok(RowShape {
        rows: crate::family::flint::cuda_ffn::real_chain::rows_from_bytes(input.bytes(), columns)
            .map_err(|e| bad(&e))?,
        columns,
    })
}

fn values(stream: &DeviceStream, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(bad("real-layer F32 backing length"));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect())
}

fn normalized(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    gamma: &[f32],
) -> Result<DeviceBuffer, EngError> {
    let shape = row_shape(input, WIDTH)?;
    let mut frame = Frame::new(ctx, stream)?;
    let input = frame.input(input.range(0, input.bytes())?, shape)?;
    let raw = frame.allocate(shape)?;
    let gamma = ctx.upload(&f32_bytes(gamma))?;
    let packed =
        ctx.allocate(xf_bytes(shape.rows, WIDTH).ok_or_else(|| bad("real-layer norm scratch"))?)?;
    ctx.add_norm_pack(
        stream,
        AddNormCall {
            m: shape.rows,
            n: WIDTH,
            eps: 1e-6,
            h: frame.range(input)?,
            delta: None,
            weight: gamma.range(0, gamma.bytes())?,
            xf: packed.range(0, packed.bytes())?,
            norm_out: Some(frame.range(raw)?),
        },
    )?;
    let published = frame.publish(raw, shape)?;
    frame.finish(published)
}

fn projection(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    weight: &weights::Weight,
) -> Result<DeviceBuffer, EngError> {
    let shape = row_shape(input, weight.shape[1])?;
    let mut frame = Frame::new(ctx, stream)?;
    let input = frame.input(input.range(0, input.bytes())?, shape)?;
    let descriptors = vec![weight.view(ctx)?; shape.rows];
    let plan = ctx.nvfp4_grouped_plan(&descriptors)?;
    let projected = frame.project(&plan, input, shape.rows, weight.shape)?;
    frame.finish(projected)
}

fn residual(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    delta: &DeviceBuffer,
) -> Result<DeviceBuffer, EngError> {
    let shape = row_shape(input, WIDTH)?;
    if delta.bytes() != input.bytes() {
        return Err(bad("real-layer residual dimensions disagree"));
    }
    let mut frame = Frame::new(ctx, stream)?;
    let output = frame.allocate(shape)?;
    ctx.row_add(
        stream,
        RowAddCall {
            x: input.range(0, input.bytes())?,
            y: delta.range(0, delta.bytes())?,
            out: frame.range(output)?,
            n: shape.rows * WIDTH,
        },
    )?;
    let published = frame.publish(output, shape)?;
    frame.finish(published)
}

fn bridge(
    ctx: &CudaContext,
    stream: &DeviceStream,
    qkv: [&DeviceBuffer; 3],
    layer: &weights::Layer,
) -> Result<[DeviceBuffer; 3], EngError> {
    let _wall = super::profile::wall("host_qk_bridge_and_upload");
    let q = values(stream, qkv[0])?;
    let k = values(stream, qkv[1])?;
    let v = values(stream, qkv[2])?;
    let [q, k] = real::qk_producer_oracle(
        &q,
        &k,
        [&layer.qk_norm[0], &layer.qk_norm[1]],
        layer.sliding.then_some(0),
    )
    .map_err(|e| bad(&e))?;
    let cache = |input: &[f32]| -> Result<DeviceBuffer, EngError> {
        let rounded = real::half_cache(input).map_err(|e| bad(&e))?;
        let mut bytes = rounded
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        let capacity = layer
            .cache_rows
            .checked_mul(4 * 128 * 2)
            .filter(|&capacity| capacity >= bytes.len())
            .ok_or_else(|| bad("real-layer physical KV capacity overflow"))?;
        bytes
            .try_reserve_exact(capacity - bytes.len())
            .map_err(|_| bad("real-layer physical KV allocation"))?;
        bytes.resize(capacity, 0);
        ctx.upload(&bytes)
    };
    Ok([ctx.upload(&f32_bytes(&q))?, cache(&k)?, cache(&v)?])
}

fn attention(
    ctx: &CudaContext,
    stream: &DeviceStream,
    qkv: [&DeviceBuffer; 3],
    layer: &weights::Layer,
) -> Result<DeviceBuffer, EngError> {
    let rows = row_shape(qkv[0], 4096)?.rows;
    if layer.cache_rows < rows {
        return Err(bad(
            "real-layer physical KV capacity is smaller than the prefix",
        ));
    }
    let [q, k, v] = bridge(ctx, stream, qkv, layer)?;
    let geometry = CpfAttnGeometry {
        m: rows,
        n_head: 32,
        n_kv_head: 4,
        head_dim: 128,
    };
    let span = CpfAttnSpan {
        window: if layer.sliding { 513 } else { 0 },
        ring: 0,
    };
    let plan = cpf_attn_launch_plan(geometry, rows, span.window as usize, CPF_ATTN_SM_COUNT)?;
    let part = ctx.allocate(cpf_attn_part_bytes(geometry, plan)?)?;
    let sinks = ctx.upload(&f32_bytes(&layer.sinks))?;
    let shape = RowShape {
        rows,
        columns: 4096,
    };
    let mut frame = Frame::new(ctx, stream)?;
    let output = frame.allocate(shape)?;
    ctx.attn_prefill(
        stream,
        AttnPrefillCall {
            storage: AttnKvStorage::F16,
            kv_views: None,
            shape: AttnPrefillShape {
                m: rows,
                n_head: 32,
                n_kv_head: 4,
                head_dim: 128,
                cache_rows: layer.cache_rows,
                ld: 0,
                p0: 0,
            },
            span,
            split: CpfAttnSplit {
                max_splits: plan.max_splits,
                min_keys: plan.min_keys,
            },
            q_tile: plan.q_tile,
            scale: (128.0_f32).sqrt().recip(),
            q: q.range(0, q.bytes())?,
            proj: None,
            k_cache: k.range(0, k.bytes())?,
            v_cache: v.range(0, v.bytes())?,
            part: part.range(0, part.bytes())?,
            out: frame.range(output)?,
            status: Some(frame.flag()?),
            sinks: Some(sinks.range(0, sinks.bytes())?),
        },
    )?;
    let published = frame.publish(output, shape)?;
    frame.finish(published)
}

fn device_layer_prefix(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    layer: &weights::Layer,
) -> Result<(DeviceBuffer, DeviceBuffer), EngError> {
    let _wall = super::profile::wall("layer_prefix_inclusive");
    let norm = normalized(ctx, stream, input, &layer.pre_norm)?;
    let q = projection(ctx, stream, &norm, &layer.qkv[0])?;
    let k = projection(ctx, stream, &norm, &layer.qkv[1])?;
    let v = projection(ctx, stream, &norm, &layer.qkv[2])?;
    let attended = attention(ctx, stream, [&q, &k, &v], layer)?;
    let projected = projection(ctx, stream, &attended, &layer.output)?;
    let residual_input = residual(ctx, stream, input, &projected)?;
    let norm = normalized(ctx, stream, &residual_input, &layer.post_norm)?;
    Ok((residual_input, norm))
}

pub(in crate::family::flint::cuda_ffn::device::tests) fn device_layer(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    layer: &weights::Layer,
) -> Result<DeviceBuffer, EngError> {
    let _wall = super::profile::wall("layer_execute_inclusive");
    let (residual_input, norm) = device_layer_prefix(ctx, stream, input, layer)?;
    let ffn = {
        let _ffn_wall = super::profile::wall("ffn_inclusive");
        layer.ffn(ctx, stream, &norm)?
    };
    residual(ctx, stream, &residual_input, &ffn)
}

#[derive(Deserialize)]
struct Oracle {
    schema_version: u32,
    model_sha256: String,
    layer: u32,
    rows: usize,
    physical_kv: String,
    sliding: bool,
    token_ids: Vec<u32>,
    input_bits: Vec<u32>,
    output_bits: Vec<u32>,
}

fn oracle(layer: u32) -> (Oracle, String) {
    let (name, digest) = match layer {
        0 => (
            "layer-0-expected.json",
            "20c3526525f011ad5c97ddfd3c26facd3ad6cc3f53cbd2b10559709e9be4566e",
        ),
        2 => (
            "layer-2-expected.json",
            "915e32c8f15552895b147b07035c2f75ec16bb0f997a8a337e80ead2eb0ce090",
        ),
        _ => panic!("only frozen standalone layers 0 and 2 are qualified here"),
    };
    let root =
        PathBuf::from(std::env::var("FIFI_FLINT_ORACLE_DIR").expect("frozen CPU oracle directory"));
    let bytes = std::fs::read(root.join(name)).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), digest);
    let oracle: Oracle = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.model_sha256, gguf::MODEL_SHA256);
    assert_eq!((oracle.layer, oracle.rows), (layer, ROWS));
    assert_eq!(oracle.physical_kv, "F16");
    assert_eq!(oracle.sliding, layer == 2);
    assert_eq!(oracle.token_ids, [0, 1, 2, 3]);
    for values in [&oracle.input_bits, &oracle.output_bits] {
        assert_eq!(values.len(), ROWS * WIDTH);
        assert!(
            values
                .iter()
                .all(|&bits| bits & 0xffff == 0 && f32::from_bits(bits).is_finite())
        );
    }
    (oracle, digest.to_string())
}

fn qualify(layer: u32) {
    let (oracle, oracle_sha256) = oracle(layer);
    let identity_path =
        std::env::var("FIFI_FLINT_MODEL_IDENTITY").expect("host model identity receipt");
    let identity = serde_json::from_reader(File::open(identity_path).unwrap()).unwrap();
    let model = std::env::var("FIFI_FLINT_GGUF").expect("pinned real GGUF");
    let file = File::open(model).unwrap();
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let input = oracle
        .input_bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect::<Vec<_>>();
    let expected = oracle
        .output_bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect::<Vec<_>>();
    let actual = {
        let ctx = CudaContext::new().expect("real CUDA qualification requires a GPU");
        let stream = ctx.stream().unwrap();
        let weights = weights::Layer::load(&ctx, &file, &header, layer, ROWS).unwrap();
        let input = ctx.upload(&f32_bytes(&input)).unwrap();
        let output = device_layer(&ctx, &stream, &input, &weights).unwrap();
        values(&stream, &output).unwrap()
    };
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    assert!(
        actual
            .iter()
            .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
    );
    let output = PathBuf::from(
        std::env::var("FIFI_FLINT_CUDA_RESULT_DIR").expect("fresh CUDA result directory"),
    );
    let mut artifact = File::options()
        .write(true)
        .create_new(true)
        .open(output.join(format!("layer-{layer}-actual.json")))
        .unwrap();
    serde_json::to_writer(
        &mut artifact,
        &serde_json::json!({
            "layer": layer, "rows": ROWS, "model_sha256": gguf::MODEL_SHA256,
            "oracle_sha256": oracle_sha256, "physical_kv": "F16",
            "bridge": "HOST D128 norm/RoPE; CUDA projections, attention and FFN",
            "output_bits": actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        }),
    )
    .unwrap();
    artifact.flush().unwrap();
    artifact.sync_all().unwrap();
    println!("REAL_CUDA layer={layer} M=4 physical_kv=F16 host_qk_bridge=true post_unlock=true");
    real_quality::compare(&expected, &actual, WIDTH).unwrap();
}

#[test]
#[ignore = "requires fifi-gpu-run admission and frozen real artifacts"]
fn dense_layer_0_m4_matches_real_fp64_oracle() {
    qualify(0);
}

#[test]
#[ignore = "requires fifi-gpu-run admission and frozen real artifacts"]
fn moe_layer_2_m4_matches_real_fp64_oracle() {
    qualify(2);
}

#[test]
#[ignore = "requires fifi-gpu-run admission; no model"]
fn prefix_norm_and_residual_keep_three_four_and_five_rows() {
    let ctx = CudaContext::new().expect("requires CUDA");
    let stream = ctx.stream().unwrap();
    for rows in [3, 4, 5] {
        let input = (0..rows)
            .flat_map(|row| vec![(row + 1) as f32; WIDTH])
            .collect::<Vec<_>>();
        let device = ctx.upload(&f32_bytes(&input)).unwrap();
        let norm = normalized(&ctx, &stream, &device, &[1.0; WIDTH]).unwrap();
        assert_eq!(values(&stream, &norm).unwrap(), vec![1.0; rows * WIDTH]);
        let output = residual(&ctx, &stream, &device, &norm).unwrap();
        assert_eq!(
            values(&stream, &output).unwrap(),
            input.iter().map(|x| x + 1.0).collect::<Vec<_>>()
        );
    }
}

#[test]
#[ignore = "requires fifi-gpu-run admission; no model"]
fn raw_head_projection_keeps_f32_without_changing_hidden_publication() {
    let ctx = CudaContext::new().expect("requires CUDA");
    let stream = ctx.stream().unwrap();
    let factor = 1.0 + 1.0 / 256.0;
    let weight = super::Weight::new(
        &ctx,
        [super::H; 2],
        super::Diagonal {
            factor,
            ..super::UNIT
        },
    );
    let shape = RowShape {
        rows: 3,
        columns: super::H,
    };
    let input = ctx
        .upload(&f32_bytes(&vec![1.0; shape.rows * shape.columns]))
        .unwrap();
    let descriptors = vec![weight.view(&ctx); shape.rows];
    let plan = ctx.nvfp4_grouped_plan(&descriptors).unwrap();
    for raw in [true, false] {
        let mut frame = Frame::new(&ctx, &stream).unwrap();
        let x = frame
            .input(input.range(0, input.bytes()).unwrap(), shape)
            .unwrap();
        let projected = if raw {
            frame.project_f32(&plan, x, shape.rows, [super::H; 2])
        } else {
            frame.project(&plan, x, shape.rows, [super::H; 2])
        }
        .unwrap();
        let output = frame.finish(projected).unwrap();
        assert_eq!(
            values(&stream, &output).unwrap(),
            vec![if raw { factor } else { 1.0 }; shape.rows * shape.columns]
        );
    }
}
