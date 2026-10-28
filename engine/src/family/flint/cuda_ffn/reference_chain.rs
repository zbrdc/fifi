//! Sequential pinned-GGUF oracle preparation; no backend state enters this trajectory.

use super::super::{Expected, HIDDEN};
use super::layer_topology;
use serde_json::Value;

const TOKEN_IDS: [u32; 4] = [0, 1, 2, 3];

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn validate_state(values: &[f32]) -> Result<(), String> {
    if values.len() != TOKEN_IDS.len() * HIDDEN
        || values
            .iter()
            .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err("chain state must contain four finite BF16 rows".into());
    }
    Ok(())
}

fn collect_layers(
    mut input: Vec<f32>,
    mut step: impl FnMut(u32, &[f32]) -> Result<Expected, String>,
) -> Result<Vec<Value>, String> {
    validate_state(&input)?;
    let mut layers = Vec::with_capacity(6);
    for index in 0..6 {
        let expected = step(index, &input)?;
        validate_state(&expected.output)?;
        let (_, sliding) = layer_topology(index)?;
        layers.push(serde_json::json!({
            "layer": index,
            "sliding": sliding,
            "input_bits": bits(&input),
            "output_bits": bits(&expected.output),
            "ids": expected.ids,
            "gates_bits": bits(&expected.gates),
            "sorted_pairs": expected.sorted_pairs,
            "inverse": expected.inverse,
            "max_intermediate_bits": expected.max_intermediate.to_bits(),
        }));
        input = expected.output;
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::super::super::run;
    use super::super::{
        Kind, MODEL_SHA256, dequantize, gguf_weights, load, raw, verify_prepared_identity,
    };
    use super::*;
    use crate::weights::{GgufHeader, GgufType};
    use std::{
        fs::File,
        io::{BufReader, Write},
        path::PathBuf,
    };

    #[test]
    fn pinned_topology_covers_all_dense_and_full_layers() {
        let mut dense = Vec::new();
        let mut full = Vec::new();
        for index in 0..49 {
            let (kind, sliding) = layer_topology(index).unwrap();
            if matches!(kind, Kind::Dense) {
                dense.push(index);
            }
            if !sliding {
                full.push(index);
            }
        }
        assert_eq!(dense, [0, 1]);
        assert_eq!(full, [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]);
        assert!(layer_topology(49).is_err());
        assert!(layer_topology(u32::MAX).is_err());
    }

    #[test]
    fn chain_uses_previous_output_for_every_layer() {
        let words = TOKEN_IDS.len() * HIDDEN;
        let mut input = vec![1.0; words];
        input[0] = -0.0;
        let mut calls = Vec::new();
        let layers = collect_layers(input.clone(), |index, state| {
            calls.push(index);
            if index == 0 {
                assert_eq!(bits(state), bits(&input));
            } else {
                assert_eq!(state, vec![index as f32 + 1.0; words]);
            }
            Ok(Expected {
                output: vec![index as f32 + 2.0; words],
                ..Expected::default()
            })
        })
        .unwrap();
        assert_eq!(calls, [0, 1, 2, 3, 4, 5]);
        assert_eq!(layers.len(), 6);
        assert_eq!(layers[0]["input_bits"][0], (-0.0_f32).to_bits());
        for (index, layer) in layers.iter().enumerate() {
            assert_eq!(layer["layer"], index as u32);
            assert_eq!(layer["sliding"], ![0, 1, 5].contains(&index));
        }
        for pair in layers.windows(2) {
            assert_eq!(pair[0]["output_bits"], pair[1]["input_bits"]);
        }
    }

    #[test]
    fn chain_rejects_invalid_state_and_stops_after_failure() {
        let words = TOKEN_IDS.len() * HIDDEN;
        for invalid in [vec![], vec![f32::NAN; words], vec![1.001; words]] {
            assert!(
                collect_layers(invalid, |_, _| panic!("invalid input reached a layer")).is_err()
            );
        }
        let mut calls = Vec::new();
        let result = collect_layers(vec![1.0; words], |index, state| {
            calls.push(index);
            Ok(Expected {
                output: if index == 2 { vec![] } else { state.to_vec() },
                ..Expected::default()
            })
        });
        assert!(result.is_err());
        assert_eq!(calls, [0, 1, 2]);
        assert!(collect_layers(vec![1.0; words], |_, _| Err("layer failed".into())).is_err());
    }

    #[test]
    #[ignore = "requires explicit CPU model-preparation authority and the pinned Flint GGUF"]
    fn freeze_pinned_gguf_chain_0_through_5() {
        let path = PathBuf::from(std::env::var_os("FIFI_FLINT_GGUF").expect("pinned GGUF"));
        let output = PathBuf::from(
            std::env::var_os("FIFI_FLINT_CHAIN_ORACLE_DIR").expect("fresh chain output directory"),
        );
        let target = output.join("chain-expected.json");
        assert!(output.is_dir());
        assert!(!target.try_exists().unwrap(), "chain output already exists");
        let identity_path = PathBuf::from(
            std::env::var_os("FIFI_FLINT_MODEL_IDENTITY").expect("host model identity receipt"),
        );
        let identity: Value =
            serde_json::from_reader(BufReader::new(File::open(identity_path).unwrap())).unwrap();
        let file = File::open(path).unwrap();
        verify_prepared_identity(&file, &identity).unwrap();
        let header = GgufHeader::parse(
            &mut BufReader::new(&file),
            file.metadata().unwrap().len(),
            64 << 20,
        )
        .unwrap();
        let mut input = Vec::with_capacity(TOKEN_IDS.len() * HIDDEN);
        for token in TOKEN_IDS {
            let range = gguf_weights::embedding_row(&header, token).unwrap();
            let data = raw(&file, range).unwrap();
            input.extend(dequantize(GgufType::Bf16, &data).unwrap());
        }
        let layers = collect_layers(input.clone(), |index, state| {
            let layer = load(&file, &header, index)?;
            run(&layer.view(), state, TOKEN_IDS.len())
        })
        .unwrap();
        verify_prepared_identity(&file, &identity).unwrap();
        let artifact = serde_json::json!({
            "schema_version": 1,
            "model_sha256": MODEL_SHA256,
            "physical_kv": "F16",
            "rows": TOKEN_IDS.len(),
            "token_ids": TOKEN_IDS,
            "input_bits": bits(&input),
            "layers": layers,
        });
        let mut out = File::options()
            .write(true)
            .create_new(true)
            .open(target)
            .unwrap();
        serde_json::to_writer(&mut out, &artifact).unwrap();
        out.flush().unwrap();
        out.sync_all().unwrap();
    }
}
