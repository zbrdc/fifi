//! Independent CPU trajectory for the eight frozen prefixes; no CUDA outputs enter it.

use super::*;
use crate::family::flint::cuda_ffn::{real_model, reference::real::run};

#[test]
#[ignore = "CPU model preparation requires fifi-build and the pinned GGUF"]
fn freeze_full49_eight_prefixes() {
    let (file, header, identity) = real_model::model();
    let prefixes = real_model::prefixes();
    let mut inputs = prefixes
        .iter()
        .map(|p| real_model::embedding(&file, &header, &p.input_ids))
        .collect::<Vec<_>>();
    let mut layers = Vec::with_capacity(49);
    for index in 0..49 {
        let layer = load(&file, &header, index).unwrap();
        let mut states = Vec::with_capacity(prefixes.len());
        for (input, prefix) in inputs.iter_mut().zip(&prefixes) {
            let expected = run(&layer.view(), input, prefix.input_ids.len()).unwrap();
            states.push(real_model::State {
                input_bits: input.iter().map(|v| v.to_bits()).collect(),
                output_bits: expected.output.iter().map(|v| v.to_bits()).collect(),
            });
            *input = expected.output;
        }
        layers.push(states);
        verify_prepared_identity(&file, &identity).unwrap();
        println!("FULL49_CPU_LAYER layer={index} cases=8 physical_kv=F16");
    }
    let oracle = real_model::Oracle {
        schema_version: 1,
        model_sha256: MODEL_SHA256.into(),
        prefixes_sha256: real_model::PREFIX_SHA256.into(),
        physical_kv: "F16".into(),
        token_ids: prefixes.into_iter().map(|p| p.input_ids).collect(),
        layers,
    };
    oracle.validate().unwrap();
    let output =
        PathBuf::from(std::env::var("FIFI_FLINT_FULL_ORACLE_OUT").expect("fresh oracle file"));
    real_model::write_new(&output, &oracle);
}
