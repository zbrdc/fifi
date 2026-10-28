//! Native six-case routing attribution. Forced decisions are diagnostic inputs, not serving.

use super::*;
use crate::family::flint::cuda_ffn::real_model;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn directory(name: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(name).expect("routing evidence directory"));
    assert!(path.is_dir());
    path
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn probe_ffn(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    weights: &FfnWeights<'_>,
) -> Result<(DeviceBuffer, Value), EngError> {
    let shape = row_shape(input, WIDTH)?;
    let block = weights.block(shape.rows)?;
    let FfnWeights::Moe { router, .. } = weights else {
        return Err(bad("routing probe requires MoE"));
    };
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(input.range(0, input.bytes())?, shape)?;
    let routed = route(&mut frame, *router, x, block)?;
    let (raw, published) = routed.logits.expect("normal route has physical logit taps");
    let raw = values(stream, &frame.buffers[raw].0)?;
    let published = values(stream, &frame.buffers[published].0)?;
    let gates = values(stream, &frame.buffers[routed.gates].0)?;
    assert_eq!(raw.len(), shape.rows * block.experts);
    assert_eq!(published.len(), raw.len());
    assert_eq!(gates.len(), block.pairs);
    assert!(
        raw.iter()
            .chain(&published)
            .chain(&gates)
            .all(|v| v.is_finite())
    );
    let record = serde_json::json!({"ids":routed.ids,"gates":gates,
        "logits_raw_bits":bits(&raw),"logits_bits":bits(&published)});
    let output = moe_routed(frame, x, block, weights, routed)?;
    Ok((output, record))
}

fn forced_ffn(
    ctx: &CudaContext,
    stream: &DeviceStream,
    input: &DeviceBuffer,
    weights: &FfnWeights<'_>,
    reference: &Value,
) -> Result<DeviceBuffer, EngError> {
    let shape = row_shape(input, WIDTH)?;
    let block = weights.block(shape.rows)?;
    let (ids, gates) = gguf::routes::decisions(reference, shape.rows).map_err(|e| bad(&e))?;
    assert_eq!(gates.len(), block.pairs);
    let grouping = Grouping::new(&ids, shape.rows, block.experts)?;
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(input.range(0, input.bytes())?, shape)?;
    let bytes = f32_bytes(&gates);
    let gates = frame.keep(ctx.upload(&bytes)?, bytes.len());
    let routed = Routed {
        ids,
        grouping,
        gates,
        logits: None,
    };
    moe_routed(frame, x, block, weights, routed)
}

#[test]
#[ignore = "requires fifi-gpu-run, sealed partial and new six-case FP64 routing records"]
fn capture_and_force_moe_miss_routes() {
    capture(&gguf::routes::PAIRS);
}

#[test]
#[ignore = "requires fifi-gpu-run and new eight-case tail routing records"]
fn capture_and_force_tail_moe_miss_routes() {
    capture(&gguf::routes::TAIL_PAIRS);
}

fn capture(pairs: &[(usize, usize)]) {
    let (oracle, sha) = rca::oracle();
    let (file, header, identity) = real_model::model();
    let references = directory("FIFI_FLINT_ROUTE_REFERENCE");
    let partial = directory("FIFI_FLINT_ROUTE_PARTIAL");
    let root = real_model::fresh_directory(
        "FIFI_FLINT_RCA_RESULT_DIR",
        &[
            references.clone(),
            partial.clone(),
            PathBuf::from(std::env::var("FIFI_FLINT_FULL_ORACLE").unwrap()),
        ],
    );
    let ctx = CudaContext::new().expect("native routing RCA requires CUDA");
    let stream = ctx.stream().unwrap();
    for &(index, case) in pairs {
        let rows = oracle.token_ids[case].len();
        let state = &oracle.layers[index][case];
        let reference_bytes = std::fs::read(
            references.join(format!("reference-routing-{index}-p{:02}.json", case + 1)),
        )
        .unwrap();
        let reference: Value = serde_json::from_slice(&reference_bytes).unwrap();
        let saved_bytes =
            std::fs::read(partial.join(format!("shared-{index}-p{:02}.json", case + 1))).unwrap();
        let saved: Value = serde_json::from_slice(&saved_bytes).unwrap();
        for data in [&reference, &saved] {
            assert_eq!(data["layer"], index);
            assert_eq!(data["case"], format!("p{:02}", case + 1));
            assert_eq!(data["rows"], rows);
            assert_eq!(data["model_sha256"], gguf::MODEL_SHA256);
            assert_eq!(data["oracle_sha256"], sha);
            assert_eq!(data["input_bits"], serde_json::json!(state.input_bits));
        }
        assert_eq!(reference["prefixes_sha256"], real_model::PREFIX_SHA256);
        assert_eq!(reference["experts"], 256);
        assert_eq!(reference["original_output_bit_exact"], true);
        assert_eq!(reference["diagnostic_only"], true);
        assert_eq!(reference["qualification"], false);
        gguf::routes::decisions(&reference, rows).unwrap();
        let layer =
            weights::Layer::load(&ctx, &file, &header, index as u32, 5 + index % 2).unwrap();
        let weights::Ffn::Moe {
            router,
            experts,
            shared,
        } = &layer.ffn
        else {
            panic!("MoE layer expected");
        };
        let bank = experts
            .iter()
            .map(|expert| expert.view(&ctx).unwrap())
            .collect::<Vec<_>>();
        let weights = FfnWeights::Moe {
            router: router.range(0, router.bytes()).unwrap(),
            experts: &bank,
            shared: shared.view(&ctx).unwrap(),
        };
        let host = state
            .input_bits
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>();
        let input = ctx.upload(&f32_bytes(&host)).unwrap();
        let baseline = device_layer(&ctx, &stream, &input, &layer).unwrap();
        let baseline = values(&stream, &baseline).unwrap();
        assert_eq!(
            saved["output_bits"],
            serde_json::json!(bits(&baseline)),
            "sealed partial was not reproduced"
        );
        let (residual_input, normalized) =
            device_layer_prefix(&ctx, &stream, &input, &layer).unwrap();
        let (down, routing) = probe_ffn(&ctx, &stream, &normalized, &weights).unwrap();
        let instrumented = residual(&ctx, &stream, &residual_input, &down).unwrap();
        assert_eq!(
            bits(&values(&stream, &instrumented).unwrap()),
            bits(&baseline),
            "routing taps changed default output"
        );
        let own = forced_ffn(&ctx, &stream, &normalized, &weights, &routing).unwrap();
        let own = residual(&ctx, &stream, &residual_input, &own).unwrap();
        let own = values(&stream, &own).unwrap();
        let own_identity = bits(&own) == bits(&baseline);
        let forced = forced_ffn(&ctx, &stream, &normalized, &weights, &reference).unwrap();
        let forced = residual(&ctx, &stream, &residual_input, &forced).unwrap();
        let forced = values(&stream, &forced).unwrap();
        let normalized = values(&stream, &normalized).unwrap();
        let residual_input = values(&stream, &residual_input).unwrap();
        assert!(
            baseline
                .iter()
                .chain(&forced)
                .chain(&own)
                .chain(&normalized)
                .chain(&residual_input)
                .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
        );
        gguf::verify_prepared_identity(&file, &identity).unwrap();
        assert_eq!(
            std::fs::read(
                references.join(format!("reference-routing-{index}-p{:02}.json", case + 1))
            )
            .unwrap(),
            reference_bytes
        );
        assert_eq!(
            std::fs::read(partial.join(format!("shared-{index}-p{:02}.json", case + 1))).unwrap(),
            saved_bytes
        );
        real_model::write_new(
            &root.join(format!("device-routing-{index}-p{:02}.json", case + 1)),
            &serde_json::json!({
                "schema_version":1,"diagnostic_only":true,"qualification":false,
                "layer":index,"case":format!("p{:02}",case+1),"rows":rows,"experts":256,
                "model_sha256":gguf::MODEL_SHA256,"oracle_sha256":sha,
                "prefixes_sha256":real_model::PREFIX_SHA256,"input_bits":state.input_bits,
                "reference_sha256":format!("{:x}",Sha256::digest(&reference_bytes)),
                "partial_case_sha256":format!("{:x}",Sha256::digest(&saved_bytes)),
                "residual_bits":bits(&residual_input),"post_norm_bits":bits(&normalized),
                "ids":routing["ids"],"gates":routing["gates"],
                "logits_raw_bits":routing["logits_raw_bits"],"logits_bits":routing["logits_bits"],
                "baseline_output_bits":bits(&baseline),"forced_reference_output_bits":bits(&forced),
                "forced_own_output_bits":bits(&own),"own_route_identity_bit_exact":own_identity,
                "instrumented_matches_untraced":true,"saved_partial_bit_exact":true,"bias":null,
                "notes":"CUDA physical logits and selected gates. No physical sigmoid score tap; any host score replay must be labeled. Per-layer counts cover this case only, not all31 rows."
            }),
        );
        assert!(
            own_identity,
            "forcing native decisions changed default output"
        );
        println!(
            "RCA_DEVICE_ROUTING layer={index} case={} rows={rows} default_bit_exact=true",
            case + 1
        );
    }
    gguf::routes::manifest(&root, "device-routing", pairs, (&oracle, &sha));
}
