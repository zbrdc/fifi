//! Shared-input qualification is separate from cumulative CUDA-state diagnostics.

use super::*;
use crate::family::flint::cuda_ffn::real_chain;

fn oracle() -> (real_chain::Oracle, String) {
    let root = std::env::var("FIFI_FLINT_CHAIN_ORACLE_DIR").expect("prepared chain oracle");
    let sha = std::env::var("FIFI_FLINT_CHAIN_ORACLE_SHA256").expect("frozen chain oracle SHA256");
    let bytes = std::fs::read(PathBuf::from(root).join("chain-expected.json")).unwrap();
    (real_chain::Oracle::parse(&bytes, &sha, 6).unwrap(), sha)
}

fn save_layer(
    root: &std::path::Path,
    oracle_sha256: &str,
    state: (&real_chain::Layer, usize, &[u32], &[f32]),
    shared_input: bool,
) {
    let (layer, capacity, input, output) = state;
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(root.join(format!("chain-layer-{}-actual.json", layer.layer)))
        .unwrap();
    serde_json::to_writer(
        &mut file,
        &serde_json::json!({
            "layer":layer.layer,"rows":ROWS,"physical_kv":"F16","cache_rows":capacity,
            "sliding":layer.sliding,"ring":0,"p0":0,"model_sha256":gguf::MODEL_SHA256,
            "oracle_sha256":oracle_sha256,"sequential":!shared_input,"host_qk_bridge":true,
            "shared_input":shared_input,"diagnostic_only":!shared_input,
            "input_bits":input,"output_bits":output.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        }),
    )
    .unwrap();
    file.flush().unwrap();
    file.sync_all().unwrap();
}

fn run(shared_input: bool) {
    let (oracle, sha) = oracle();
    let model = std::env::var("FIFI_FLINT_GGUF").expect("pinned GGUF");
    let identity_path = std::env::var("FIFI_FLINT_MODEL_IDENTITY").expect("host identity receipt");
    let identity = serde_json::from_reader(File::open(identity_path).unwrap()).unwrap();
    let file = File::open(model).unwrap();
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let output_root =
        PathBuf::from(std::env::var("FIFI_FLINT_CUDA_RESULT_DIR").expect("fresh actual directory"));
    let ctx = CudaContext::new().expect("real CUDA chain requires GPU");
    let stream = ctx.stream().unwrap();
    let initial = oracle
        .input_bits
        .iter()
        .map(|&v| f32::from_bits(v))
        .collect::<Vec<_>>();
    let mut input = ctx.upload(&f32_bytes(&initial)).unwrap();
    let mut previous_bits = oracle.input_bits.clone();
    let first = usize::from(shared_input);
    let count = oracle.layers.len() - first;
    println!(
        "REAL_CHAIN_BEGIN layers={count} shared_input={shared_input} M=4 physical_kv=F16 host_qk_bridge=true post_unlock=true"
    );
    for layer in oracle.layers.iter().skip(first) {
        if shared_input {
            previous_bits.clone_from(&layer.input_bits);
            let reference_input = previous_bits
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect::<Vec<_>>();
            input = ctx.upload(&f32_bytes(&reference_input)).unwrap();
        }
        let observed_input = values(&stream, &input)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(observed_input, previous_bits);
        // Different physical capacities must preserve the same logical four-row prefix.
        let capacity = ROWS + layer.layer as usize % 2;
        let weights = weights::Layer::load(&ctx, &file, &header, layer.layer, capacity).unwrap();
        assert_eq!(weights.sliding, layer.sliding);
        let output = device_layer(&ctx, &stream, &input, &weights).unwrap();
        let actual = values(&stream, &output).unwrap();
        assert_eq!(actual.len(), ROWS * WIDTH);
        assert!(
            actual
                .iter()
                .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
        );
        save_layer(
            &output_root,
            &sha,
            (layer, capacity, &previous_bits, &actual),
            shared_input,
        );
        let expected = layer
            .output_bits
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>();
        println!(
            "REAL_CHAIN_LAYER layer={} M=4 physical_kv=F16 cache_rows={} post_unlock=true",
            layer.layer, capacity
        );
        let quality = real_quality::compare(&expected, &actual, WIDTH);
        if shared_input {
            quality.unwrap();
        } else {
            // Only shared-input error has a layerwise limit; final logits judge the chain.
            println!(
                "REAL_CHAIN_DIAGNOSTIC layer={} quality={quality:?}",
                layer.layer
            );
        }
        previous_bits = actual.iter().map(|v| v.to_bits()).collect();
        input = output;
        gguf::verify_prepared_identity(&file, &identity).unwrap();
    }
    println!("REAL_CHAIN_END layers={count} shared_input={shared_input} post_unlock=true");
}

#[test]
#[ignore = "requires fifi-gpu-run admission and the pinned sequential FP64 oracle"]
fn layers_1_through_5_match_shared_input_fp64_oracle() {
    run(true);
}

#[test]
#[ignore = "requires fifi-gpu-run admission; cumulative errors are diagnostic, not qualification"]
fn layers_0_through_5_propagate_cuda_state() {
    run(false);
}
