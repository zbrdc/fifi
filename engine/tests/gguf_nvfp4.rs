use inference_engine::weights::{
    GgufType, dequantize, dequantize_into, dequantize_nvfp4_expert_into, repack_nvfp4_into,
    sha256_hex,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf-nvfp4")
}

fn manifest() -> Value {
    serde_json::from_slice(&std::fs::read(dir().join("manifest.json")).unwrap()).unwrap()
}

fn fixture(entry: &Value, key: &str) -> Vec<u8> {
    std::fs::read(dir().join(entry[key]["file"].as_str().unwrap())).unwrap()
}

fn same_bits(actual: &[f32], expected: &[u8]) {
    let (words, rest) = expected.as_chunks::<4>();
    assert!(rest.is_empty());
    assert_eq!(actual.len(), words.len());
    for (i, (value, bits)) in actual.iter().zip(words).enumerate() {
        assert!(value.is_finite(), "element {i}");
        assert_eq!(value.to_bits(), u32::from_le_bytes(*bits), "element {i}");
    }
}

#[test]
fn nvfp4_type40_reports_the_gguf_block_geometry() {
    let ty = GgufType::from_id(40).expect("NVFP4 is a supported GGUF storage type");
    assert_eq!(ty.id(), 40);
    assert_eq!(ty.name(), "NVFP4");
    assert_eq!(ty.block_elems(), 64);
    assert_eq!(ty.block_bytes(), 36);
}

#[test]
fn nvfp4_decodes_split_nibbles_and_each_unsigned_group_scale() {
    let mut block = vec![0x38, 0x40, 0x30, 0x7f];
    for _ in 0..4 {
        for code in 0u8..8 {
            block.push(((code + 8) << 4) | code);
        }
    }
    let want: [f32; 64] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
        1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0, 0.0,
        0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
    ];
    let got = dequantize(GgufType::Nvfp4, &block).unwrap();
    assert_eq!(got.len(), want.len());
    for (i, (value, expected)) in got.iter().zip(want).enumerate() {
        assert!(value.is_finite(), "column {i}");
        assert_eq!(value.to_bits(), expected.to_bits(), "column {i}");
    }
}

#[test]
fn oracle_fixtures_pin_source_and_verify_every_file_digest() {
    let m = manifest();
    assert_eq!(
        m["model_sha256"],
        "97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde"
    );
    assert_eq!(
        m["oracle"]["target_commit"],
        "a37cc46104eb50900d5ab4a9ba02a0abdac039db"
    );
    let entries = m["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 4);
    for entry in entries {
        for key in ["blocks", "scales", "raw_f32", "scaled_f32"] {
            let bytes = fixture(entry, key);
            assert_eq!(bytes.len() as u64, entry[key]["bytes"].as_u64().unwrap());
            assert_eq!(
                sha256_hex(&Sha256::digest(&bytes).into()),
                entry[key]["sha256"]
            );
        }
    }
}

#[test]
fn all_scale_bytes_and_three_real_tensors_decode_bit_exactly() {
    for entry in manifest()["entries"].as_array().unwrap() {
        let blocks = fixture(entry, "blocks");
        let want = fixture(entry, "raw_f32");
        let got = dequantize(GgufType::Nvfp4, &blocks).unwrap();
        same_bits(&got, &want);
        let mut into = vec![0.0; got.len()];
        dequantize_into(GgufType::Nvfp4, &blocks, &mut into).unwrap();
        same_bits(&into, &want);
    }
}

#[test]
fn real_expert_rows_use_their_own_f32_tensor_multiplier() {
    let m = manifest();
    let mut routed = false;
    for entry in m["entries"].as_array().unwrap() {
        let source = fixture(entry, "blocks");
        let scales = fixture(entry, "scales");
        let want = fixture(entry, "scaled_f32");
        let count = entry["experts"].as_u64().unwrap() as usize;
        if entry["name"] == "real-routed-gate" {
            routed = true;
            assert_eq!(count, 256);
            assert!(scales.as_chunks::<4>().0.iter().any(|s| s != &scales[..4]));
        }
        for expert in 0..count {
            let mut row = [0.0; 64];
            dequantize_nvfp4_expert_into(&source, &scales, expert, &mut row)
                .unwrap_or_else(|e| panic!("{} expert{expert}: {e:?}", entry["name"]));
            same_bits(&row, &want[expert * 256..(expert + 1) * 256]);
        }
    }
    assert!(routed);
}

#[test]
fn expert_decode_checks_ranges_scales_and_finite_results() {
    let mut source = [0x77; 36];
    source[..4].fill(0x38);
    let mut out = [0.0; 64];
    let one = 1.0f32.to_le_bytes();
    for scales in [&[][..], &[0, 1, 2][..]] {
        assert!(dequantize_nvfp4_expert_into(&source, scales, 0, &mut out).is_err());
    }
    assert!(dequantize_nvfp4_expert_into(&source, &one, 1, &mut out).is_err());
    assert!(dequantize_nvfp4_expert_into(&source[..35], &one, 0, &mut out).is_err());
    assert!(dequantize_nvfp4_expert_into(&source, &one, 0, &mut out[..63]).is_err());
    for scale in [f32::NAN, f32::INFINITY, f32::MAX] {
        assert!(dequantize_nvfp4_expert_into(&source, &scale.to_le_bytes(), 0, &mut out).is_err());
    }
    assert!(dequantize_into(GgufType::Nvfp4, &source[..35], &mut out).is_err());
    assert!(dequantize_into(GgufType::Nvfp4, &source, &mut out[..63]).is_err());
}

#[test]
fn repacking_into_modelopt_preserves_every_representable_weight_bit() {
    use inference_engine::{kernels::nvfp4::WeightBlock, tensor::Dtype};
    for entry in manifest()["entries"].as_array().unwrap() {
        let source = fixture(entry, "blocks");
        let want = fixture(entry, "raw_f32");
        for (i, block) in source.as_chunks::<36>().0.iter().enumerate() {
            let mut data = [0xa5; 32];
            let mut scales = [0xa5; 4];
            let result = repack_nvfp4_into(block, &mut data, &mut scales);
            if block[..4].contains(&0xff) {
                assert!(result.is_err(), "finite480 is not ModelOpt E4M3FN");
                assert_eq!(data, [0xa5; 32]);
                assert_eq!(scales, [0xa5; 4]);
                continue;
            }
            result.unwrap();
            let tensor = WeightBlock {
                packed: &data,
                weight_scale: &scales,
                weight_scale_2: 1.0,
                input_scale: None,
            }
            .decode(&[1, 64], Dtype::F32)
            .unwrap();
            let values: Vec<f32> = tensor
                .host_bytes()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect();
            same_bits(&values, &want[i * 256..(i + 1) * 256]);
        }
    }
}

#[test]
fn repacking_checks_all_lengths_before_touching_outputs() {
    let mut source = [0x38; 72];
    let mut data = [0xa5; 64];
    let mut scales = [0xa5; 8];
    assert!(repack_nvfp4_into(&source[..71], &mut data, &mut scales).is_err());
    assert_eq!(data, [0xa5; 64]);
    assert!(repack_nvfp4_into(&source, &mut data[..63], &mut scales).is_err());
    assert!(repack_nvfp4_into(&source, &mut data, &mut scales[..7]).is_err());
    assert_eq!(data, [0xa5; 64]);
    assert_eq!(scales, [0xa5; 8]);
    source[36] = 0xff;
    assert!(repack_nvfp4_into(&source, &mut data, &mut scales).is_err());
    assert_eq!(data, [0xa5; 64]);
    assert_eq!(scales, [0xa5; 8]);
    source[36] = 0x38;
    repack_nvfp4_into(&source, &mut data, &mut scales).unwrap();
    let tensor = inference_engine::kernels::nvfp4::WeightBlock {
        packed: &data,
        weight_scale: &scales,
        weight_scale_2: 1.0,
        input_scale: None,
    }
    .decode(&[2, 64], inference_engine::tensor::Dtype::F32)
    .unwrap();
    let got: Vec<f32> = tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let expected: Vec<u8> = dequantize(GgufType::Nvfp4, &source)
        .unwrap()
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    same_bits(&got, &expected);
}
