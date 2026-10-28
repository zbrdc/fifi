//! Saved-boundary RCA only. Continuing after numeric misses never qualifies the model.

use super::*;
use crate::family::flint::cuda_ffn::real_model;

pub(super) fn oracle() -> (real_model::Oracle, String) {
    let path = std::env::var("FIFI_FLINT_FULL_ORACLE").expect("frozen oracle");
    let sha = std::env::var("FIFI_FLINT_FULL_ORACLE_SHA256").expect("frozen oracle SHA256");
    let oracle = real_model::Oracle::parse(&std::fs::read(path).unwrap(), &sha).unwrap();
    (oracle, sha)
}

fn root() -> PathBuf {
    PathBuf::from(std::env::var("FIFI_FLINT_RCA_RESULT_DIR").expect("fresh RCA directory"))
}

fn boundary(name: &str, shape: RowShape, data: Vec<f32>, source: &str) -> serde_json::Value {
    assert_eq!(data.len(), shape.rows * shape.columns);
    assert!(
        data.iter()
            .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
    );
    serde_json::json!({"name":name,"columns":shape.columns,"source":source,
        "bits":data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()})
}

fn dense_ffn(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    layer: &weights::Layer,
) -> Result<(DeviceBuffer, Vec<serde_json::Value>), EngError> {
    let rows = row_shape(input, WIDTH)?.rows;
    let weight = layer.dense_weights(ctx)?;
    let plan = ExpertPlan::new(ctx, &vec![weight; rows])?;
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(
        input.range(0, input.bytes())?,
        RowShape {
            rows,
            columns: WIDTH,
        },
    )?;
    let shape = RowShape {
        rows,
        columns: plan.intermediate,
    };
    let gate = frame.project(&plan.gate, x, rows, [plan.intermediate, plan.hidden])?;
    let up = frame.project(&plan.up, x, rows, [plan.intermediate, plan.hidden])?;
    let sig = frame.sigmoid(gate, shape)?;
    let silu = frame.multiply(gate, sig, shape, shape.columns)?;
    let product = frame.multiply(silu, up, shape, shape.columns)?;
    let down = frame.project(&plan.down, product, rows, [plan.hidden, plan.intermediate])?;
    let mut boundaries = Vec::new();
    for (name, slot, columns) in [
        ("gate", gate, shape.columns),
        ("up", up, shape.columns),
        ("sigmoid", sig, shape.columns),
        ("silu", silu, shape.columns),
        ("product", product, shape.columns),
        ("down", down, WIDTH),
    ] {
        boundaries.push(boundary(
            name,
            RowShape { rows, columns },
            values(stream, &frame.buffers[slot].0)?,
            "CUDA",
        ));
    }
    // Late health and descriptor lifetime checks remain identical to forward().
    Ok((frame.finish(down)?, boundaries))
}

fn trace(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    layer: &weights::Layer,
) -> Result<Vec<serde_json::Value>, EngError> {
    let rows = row_shape(input, WIDTH)?.rows;
    let norm = normalized(ctx, stream, input, &layer.pre_norm)?;
    let q = projection(ctx, stream, &norm, &layer.qkv[0])?;
    let k = projection(ctx, stream, &norm, &layer.qkv[1])?;
    let v = projection(ctx, stream, &norm, &layer.qkv[2])?;
    let attended = attention(ctx, stream, [&q, &k, &v], layer)?;
    let projected = projection(ctx, stream, &attended, &layer.output)?;
    let residual_input = residual(ctx, stream, input, &projected)?;
    let post_norm = normalized(ctx, stream, &residual_input, &layer.post_norm)?;
    let (down, ffn_boundaries) = dense_ffn(ctx, stream, &post_norm, layer)?;
    let output = residual(ctx, stream, &residual_input, &down)?;
    let mut boundaries = Vec::new();
    for (name, buffer, columns) in [
        ("input", input, WIDTH),
        ("pre_norm", &norm, WIDTH),
        ("q_proj", &q, 4096),
        ("k_proj", &k, 512),
        ("v_proj", &v, 512),
        ("attention", &attended, 4096),
        ("out_proj", &projected, WIDTH),
        ("residual", &residual_input, WIDTH),
        ("post_norm", &post_norm, WIDTH),
    ] {
        boundaries.push(boundary(
            name,
            RowShape { rows, columns },
            values(stream, buffer)?,
            "CUDA",
        ));
    }
    // This is explicitly a deterministic HOST replay, not a tap of physical device-cache bytes.
    let [host_q, host_k] = real::qk_producer_oracle(
        &values(stream, &q)?,
        &values(stream, &k)?,
        [&layer.qk_norm[0], &layer.qk_norm[1]],
        None,
    )
    .map_err(|e| bad(&e))?;
    for (name, data, columns, source) in [
        ("q_norm", host_q, 4096, "HOST_QK_PRODUCER_REPLAY"),
        ("k_norm", host_k.clone(), 512, "HOST_QK_PRODUCER_REPLAY"),
        (
            "k_cache",
            real::half_cache(&host_k).map_err(|e| bad(&e))?,
            512,
            "HOST_F16_CACHE_REPLAY",
        ),
        (
            "v_cache",
            real::half_cache(&values(stream, &v)?).map_err(|e| bad(&e))?,
            512,
            "HOST_F16_CACHE_REPLAY",
        ),
    ] {
        boundaries.push(boundary(name, RowShape { rows, columns }, data, source));
    }
    boundaries.extend(ffn_boundaries);
    boundaries.push(boundary(
        "output",
        RowShape {
            rows,
            columns: WIDTH,
        },
        values(stream, &output)?,
        "CUDA",
    ));
    Ok(boundaries)
}

#[test]
#[ignore = "requires fifi-gpu-run, pinned GGUF/oracle/saved miss; RCA, not qualification"]
fn capture_l0_p01_m5_and_m4_boundaries() {
    let (oracle, sha) = oracle();
    let saved = std::fs::read(
        std::env::var("FIFI_FLINT_RCA_SAVED_M5").expect("saved failing actual artifact"),
    )
    .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&saved)),
        "c4ae06d75a4721b56a351f006da5fcd3854ce13eca7aef6a042ab920099b774b"
    );
    let saved: serde_json::Value = serde_json::from_slice(&saved).unwrap();
    assert_eq!(
        saved["input_bits"],
        serde_json::json!(oracle.layers[0][0].input_bits)
    );
    let (file, header, identity) = real_model::model();
    let ctx = CudaContext::new().expect("RCA requires CUDA");
    let stream = ctx.stream().unwrap();
    let layer = weights::Layer::load(&ctx, &file, &header, 0, 5).unwrap();
    for rows in [5, 4] {
        let bits = &oracle.layers[0][0].input_bits[..rows * WIDTH];
        let input = bits.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>();
        let input = ctx.upload(&f32_bytes(&input)).unwrap();
        assert_eq!(
            values(&stream, &input)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        let baseline = device_layer(&ctx, &stream, &input, &layer).unwrap();
        let baseline = values(&stream, &baseline)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        let boundaries = trace(&ctx, &stream, &input, &layer).unwrap();
        assert_eq!(
            boundaries.last().unwrap()["bits"],
            serde_json::json!(baseline),
            "instrumentation changed same-group result"
        );
        if rows == 5 {
            assert_eq!(
                saved["output_bits"],
                serde_json::json!(baseline),
                "saved M5 was not reproduced"
            );
        }
        gguf::verify_prepared_identity(&file, &identity).unwrap();
        real_model::write_new(
            &root().join(format!("device-l0-p01-m{rows}.json")),
            &serde_json::json!({
                "schema_version":1,"diagnostic_only":true,"layer":0,"case":"p01","rows":rows,"cache_rows":5,
                "model_sha256":gguf::MODEL_SHA256,"prefixes_sha256":real_model::PREFIX_SHA256,"oracle_sha256":sha,
                "input_bits":bits,"baseline_output_bits":baseline,"trace_matches_untraced":true,
                "saved_m5_bit_exact":rows == 5,"boundaries":boundaries,
                "notes":"M4 versus M5 is a grouping sensitivity probe, not a bit-equality requirement. Q/K norm and cache records are labeled host replays. No cumulative propagation or qualification."
            }),
        );
        println!(
            "RCA_L0_BOUNDARIES rows={rows} cache_rows=5 trace_matches_untraced=true diagnostic_only=true"
        );
    }
}

#[test]
#[ignore = "requires fifi-gpu-run, pinned full49 oracle/GGUF; collect shared misses, never qualify"]
fn collect_all49_shared_diagnostics() {
    collect_shared_diagnostics(0);
}

#[test]
#[ignore = "requires fifi-gpu-run; recover only layers30..49 after the sealed partial run"]
fn collect_remaining19_shared_diagnostics() {
    collect_shared_diagnostics(30);
}

fn collect_shared_diagnostics(start: usize) {
    let (oracle, sha) = oracle();
    let (file, header, identity) = real_model::model();
    let ctx = CudaContext::new().expect("shared diagnostics require CUDA");
    let stream = ctx.stream().unwrap();
    for (index, reference) in oracle.layers.iter().enumerate().skip(start) {
        let layer =
            weights::Layer::load(&ctx, &file, &header, index as u32, 5 + index % 2).unwrap();
        for (case, state) in reference.iter().enumerate() {
            let rows = oracle.token_ids[case].len();
            let host = state
                .input_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>();
            let input = ctx.upload(&f32_bytes(&host)).unwrap();
            assert_eq!(
                values(&stream, &input)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                state.input_bits
            );
            let output = device_layer(&ctx, &stream, &input, &layer).unwrap();
            let actual = values(&stream, &output).unwrap();
            assert_eq!(actual.len(), rows * WIDTH);
            assert!(
                actual
                    .iter()
                    .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
            );
            let expected = state
                .output_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>();
            let shared_limit_pass = real_quality::compare(&expected, &actual, WIDTH).is_ok();
            real_model::write_new(
                &root().join(format!("shared-{index}-p{:02}.json", case + 1)),
                &serde_json::json!({
                    "schema_version":1,"diagnostic_only":true,"mode":"shared_diagnostic","layer":index,
                    "case":format!("p{:02}",case+1),"rows":rows,"cache_rows":layer.cache_rows,
                    "model_sha256":gguf::MODEL_SHA256,"prefixes_sha256":real_model::PREFIX_SHA256,
                    "oracle_sha256":sha,"physical_kv":"F16","input_bits":state.input_bits,
                    "output_bits":actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "shared_limit_pass":shared_limit_pass,"qualification":false,"cumulative_executed":false
                }),
            );
        }
        gguf::verify_prepared_identity(&file, &identity).unwrap();
        println!("RCA_SHARED_LAYER layer={index} cases=8 diagnostic_only=true");
    }
}
