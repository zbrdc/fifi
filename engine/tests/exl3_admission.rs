use std::collections::HashMap;

use inference_engine::{
    EngError,
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{Codebook, Config, Matrix},
    },
};
use serde_json::{Value, json};

struct Components(HashMap<String, Tensor>);
impl WeightSource for Components {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.0.get(key).cloned()
    }
    fn resident_bytes(&self) -> u64 {
        self.0.values().map(Tensor::byte_len).sum()
    }
    fn tier(&self) -> Tier {
        Tier::Resident
    }
}
impl Components {
    fn put(&mut self, suffix: &str, tensor: Tensor) {
        self.0.insert(format!("matrix.{suffix}"), tensor);
    }
}
fn tensor(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Tensor {
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}
fn f16_vector(length: u32, word: u16) -> Tensor {
    tensor(
        Dtype::F16,
        vec![length],
        word.to_le_bytes().repeat(length as usize),
    )
}
fn source() -> Components {
    Components(
        [
            (
                "matrix.trellis".into(),
                tensor(Dtype::I16, vec![8, 8, 48], vec![0; 6144]),
            ),
            ("matrix.suh".into(), f16_vector(128, 0x3c00)),
            ("matrix.svh".into(), f16_vector(128, 0x3c00)),
            (
                "matrix.mul1".into(),
                tensor(Dtype::I32, vec![], 0x83dc_d12du32.to_le_bytes().to_vec()),
            ),
        ]
        .into(),
    )
}
fn metadata() -> Value {
    json!({"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1","bits":3.08,"out_scales":"always"}})
}
fn config() -> Config {
    Config::from_json(&serde_json::to_vec(&metadata()).unwrap()).unwrap()
}

#[test]
fn admitted_matrix_retains_its_backing() {
    let matrix = {
        let s = source();
        Matrix::load(&s, "matrix", &config()).unwrap()
    };
    assert_eq!(matrix.trellis().host_bytes().unwrap().len(), 6144);
    assert_eq!(matrix.suh().shape(), [128]);
    assert_eq!(matrix.svh().shape(), [128]);
    assert!(matrix.bias().is_none());
}
#[test]
fn marker_selection_covers_three_inst_and_mcg() {
    for (name, book, word) in [
        ("3inst", Codebook::ThreeInst, None),
        ("mcg", Codebook::Mcg, Some(0xcbac_1fedu32)),
    ] {
        let mut s = source();
        s.0.remove("matrix.mul1");
        if let Some(word) = word {
            s.put(
                "mcg",
                tensor(Dtype::I32, vec![], word.to_le_bytes().to_vec()),
            );
        }
        let mut value = metadata();
        value["quantization_config"]["codebook"] = json!(name);
        let cfg = Config::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(Matrix::load(&s, "matrix", &cfg).unwrap().codebook(), book);
    }
}
#[test]
fn absent_config_codebook_defers_to_marker() {
    let mut value = metadata();
    value["quantization_config"]
        .as_object_mut()
        .unwrap()
        .remove("codebook");
    let cfg = Config::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(
        Matrix::load(&source(), "matrix", &cfg).unwrap().codebook(),
        Codebook::Mul1
    );
}
#[test]
fn mul1_half_integer_rates_come_from_trellis() {
    for words in [24, 40, 56] {
        let mut s = source();
        s.put(
            "trellis",
            tensor(
                Dtype::I16,
                vec![8, 8, words],
                vec![0; 8 * 8 * words as usize * 2],
            ),
        );
        assert_eq!(
            Matrix::load(&s, "matrix", &config())
                .unwrap()
                .bitrate_twice(),
            (words / 8) as u8
        );
    }
}
#[test]
fn expert_division_metadata_is_retained() {
    let mut value = metadata();
    value["quantization_config"]["version"] = json!("1.6.2");
    value["quantization_config"]["expert_interm_div"] = json!(128);
    let cfg = Config::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(cfg.version(), "1.6.2");
    assert_eq!(cfg.expert_interm_div(), 128);
    assert!(Matrix::load(&source(), "matrix", &cfg).is_ok());
}
#[test]
fn optional_float_bias_is_retained() {
    for dtype in [Dtype::F16, Dtype::F32] {
        let mut s = source();
        let width = if dtype == Dtype::F16 { 2 } else { 4 };
        s.put("bias", tensor(dtype, vec![128], vec![0; 128 * width]));
        assert_eq!(
            Matrix::load(&s, "matrix", &config())
                .unwrap()
                .bias()
                .unwrap()
                .dtype(),
            dtype
        );
    }
}

#[test]
fn missing_trellis_is_refused() {
    let mut s = source();
    s.0.remove("matrix.trellis");
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn missing_suh_is_refused() {
    let mut s = source();
    s.0.remove("matrix.suh");
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn missing_svh_is_refused() {
    let mut s = source();
    s.0.remove("matrix.svh");
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn float_trellis_is_refused() {
    let mut s = source();
    s.put("trellis", tensor(Dtype::F16, vec![8, 8, 48], vec![0; 6144]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn rank_two_trellis_is_refused() {
    let mut s = source();
    s.put("trellis", tensor(Dtype::I16, vec![64, 48], vec![0; 6144]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn zero_dimension_is_refused() {
    let mut s = source();
    s.put("trellis", tensor(Dtype::I16, vec![0, 8, 48], vec![]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn non_128_dimension_is_refused() {
    let mut s = source();
    s.put(
        "trellis",
        tensor(Dtype::I16, vec![7, 8, 48], vec![0; 7 * 8 * 48 * 2]),
    );
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn unsupported_tile_extent_is_refused() {
    let mut s = source();
    s.put(
        "trellis",
        tensor(Dtype::I16, vec![8, 8, 17], vec![0; 8 * 8 * 17 * 2]),
    );
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn wrong_scale_dtype_is_refused() {
    let mut s = source();
    s.put("suh", tensor(Dtype::F32, vec![128], vec![0; 512]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn short_suh_is_refused() {
    let mut s = source();
    s.put("suh", f16_vector(127, 0x3c00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn short_svh_is_refused() {
    let mut s = source();
    s.put("svh", f16_vector(127, 0x3c00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn rank_two_scale_is_refused() {
    let mut s = source();
    s.put("svh", tensor(Dtype::F16, vec![1, 128], vec![0; 256]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn nan_suh_is_refused() {
    let mut s = source();
    s.put("suh", f16_vector(128, 0x7e00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn infinite_svh_is_refused() {
    let mut s = source();
    s.put("svh", f16_vector(128, 0x7c00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn unsigned_marker_is_refused() {
    let mut s = source();
    s.put(
        "mul1",
        tensor(Dtype::U32, vec![], 0x83dc_d12du32.to_le_bytes().to_vec()),
    );
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn vector_marker_is_refused() {
    let mut s = source();
    s.put(
        "mul1",
        tensor(Dtype::I32, vec![1], 0x83dc_d12du32.to_le_bytes().to_vec()),
    );
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn unknown_marker_word_is_refused() {
    let mut s = source();
    s.put("mul1", tensor(Dtype::I32, vec![], vec![0; 4]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn multiple_markers_are_refused() {
    let mut s = source();
    s.put(
        "mcg",
        tensor(Dtype::I32, vec![], 0xcbac_1fedu32.to_le_bytes().to_vec()),
    );
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn marker_config_mismatch_is_refused() {
    let mut value = metadata();
    value["quantization_config"]["codebook"] = json!("mcg");
    let cfg = Config::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(Matrix::load(&source(), "matrix", &cfg).is_err());
}
#[test]
fn half_integer_three_inst_is_refused() {
    let mut s = source();
    s.0.remove("matrix.mul1");
    s.put("trellis", tensor(Dtype::I16, vec![8, 8, 24], vec![0; 3072]));
    let mut value = metadata();
    value["quantization_config"]
        .as_object_mut()
        .unwrap()
        .remove("codebook");
    let cfg = Config::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(Matrix::load(&s, "matrix", &cfg).is_err());
}
#[test]
fn packed_signs_and_old_scale_are_refused() {
    for suffix in ["su", "sv", "scale"] {
        let mut s = source();
        s.put(suffix, tensor(Dtype::I16, vec![8], vec![0; 16]));
        assert!(Matrix::load(&s, "matrix", &config()).is_err(), "{suffix}");
    }
}
#[test]
fn integer_bias_is_refused() {
    let mut s = source();
    s.put("bias", tensor(Dtype::I32, vec![128], vec![0; 512]));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn short_bias_is_refused() {
    let mut s = source();
    s.put("bias", f16_vector(127, 0));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn nonfinite_bias_is_refused() {
    let mut s = source();
    s.put("bias", f16_vector(128, 0x7c00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}
#[test]
fn invalid_config_fields_are_refused() {
    for (key, value) in [
        ("quant_method", json!("gptq")),
        ("codebook", json!("unknown")),
        ("version", json!("0.0.3")),
        ("version", json!("1.7.0")),
        ("version", json!("1.6.2.dev")),
        ("expert_interm_div", json!(0)),
        ("expert_interm_div", json!(2)),
    ] {
        let mut meta = metadata();
        meta["quantization_config"][key] = value;
        assert!(
            Config::from_json(&serde_json::to_vec(&meta).unwrap()).is_err(),
            "{key}"
        );
    }
    assert!(Config::from_json(b"{}").is_err());
}
#[test]
fn duplicate_config_members_are_refused() {
    assert!(Config::from_json(br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1","codebook":"mcg"}}"#).is_err());
}
#[test]
fn oversized_config_is_refused_before_parse() {
    assert!(matches!(
        Config::from_json(&vec![b' '; 1_048_577]),
        Err(EngError::MemoryPlan(_))
    ));
}

fn nonsquare_source() -> Components {
    let mut s = source();
    s.put(
        "trellis",
        tensor(Dtype::I16, vec![8, 16, 48], vec![0; 12288]),
    );
    s.put("svh", f16_vector(256, 0x3c00));
    s.put("bias", f16_vector(256, 0));
    s
}

#[test]
fn nonsquare_matrix_preserves_input_and_output_axes() {
    let matrix = Matrix::load(&nonsquare_source(), "matrix", &config()).unwrap();
    assert_eq!(matrix.shape(), [128, 256]);
    assert_eq!(matrix.suh().shape(), [128]);
    assert_eq!(matrix.svh().shape(), [256]);
    assert_eq!(matrix.bias().unwrap().shape(), [256]);
}

#[test]
fn suh_cannot_use_the_output_extent() {
    let mut s = nonsquare_source();
    s.put("suh", f16_vector(256, 0x3c00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}

#[test]
fn svh_cannot_use_the_input_extent() {
    let mut s = nonsquare_source();
    s.put("svh", f16_vector(128, 0x3c00));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}

#[test]
fn bias_cannot_use_the_input_extent() {
    let mut s = nonsquare_source();
    s.put("bias", f16_vector(128, 0));
    assert!(Matrix::load(&s, "matrix", &config()).is_err());
}

#[test]
fn nonfinite_f32_bias_is_refused() {
    for value in [f32::INFINITY, f32::NAN] {
        let mut s = source();
        s.put(
            "bias",
            tensor(Dtype::F32, vec![128], value.to_le_bytes().repeat(128)),
        );
        assert!(Matrix::load(&s, "matrix", &config()).is_err());
    }
}

#[test]
fn conversion_options_do_not_override_modern_layout_admission() {
    for version in ["1.4.0", "1.5.0", "1.6.4"] {
        for out_scales in ["auto", "always"] {
            let mut value = metadata();
            value["quantization_config"]["version"] = json!(version);
            value["quantization_config"]["out_scales"] = json!(out_scales);
            let cfg = Config::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(Matrix::load(&source(), "matrix", &cfg).is_ok());
        }
    }
}

#[test]
fn physical_rate_boundaries_are_checked_at_admission() {
    for words in [16, 128, 8, 136, 72] {
        let mut s = source();
        s.put(
            "trellis",
            tensor(Dtype::I16, vec![8, 8, words], vec![0; 128 * words as usize]),
        );
        let result = Matrix::load(&s, "matrix", &config());
        if matches!(words, 16 | 128) {
            assert_eq!(result.unwrap().bitrate_twice(), (words / 8) as u8);
        } else {
            assert!(result.is_err(), "{words}");
        }
    }
}
