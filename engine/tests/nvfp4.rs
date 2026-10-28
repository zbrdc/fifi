use inference_engine::{
    EngError,
    kernels::nvfp4::{ScaleRole, WeightBlock},
    tensor::{Dtype, Tensor},
};

const UNIT: &[u8] = include_bytes!("fixtures/nvfp4/unit.bin");
const SIGNED: &[u8] = include_bytes!("fixtures/nvfp4/signed.bin");
const SCALES: &[u8] = include_bytes!("fixtures/nvfp4/scales.bin");
const MAGNITUDES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
];

fn block() -> WeightBlock<'static> {
    WeightBlock {
        packed: UNIT,
        weight_scale: &SCALES[..1],
        weight_scale_2: 1.0,
        input_scale: None,
    }
}

fn floats(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::F32);
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn invalid(block: &WeightBlock<'_>, shape: &[u32], dtype: Dtype, field: &str) {
    match block.decode(shape, dtype) {
        Err(EngError::BadRequest(message)) => assert!(message.contains(field), "{message}"),
        other => panic!("expected invalid {field}, got {other:?}"),
    }
}

#[test]
fn nvfp4_full_group_with_unit_scales_uses_low_nibble_first() {
    let tensor = block().decode(&[1, 16], Dtype::F32).unwrap();
    assert_eq!(tensor.shape(), &[1, 16]);
    assert_eq!(floats(&tensor), MAGNITUDES);
}

#[test]
fn nvfp4_e4m3_two_times_second_level_half_is_unit_scale() {
    let block = WeightBlock {
        weight_scale: &SCALES[1..],
        weight_scale_2: 0.5,
        ..block()
    };
    assert_eq!(
        floats(&block.decode(&[16], Dtype::F32).unwrap()),
        MAGNITUDES
    );
}

#[test]
fn nvfp4_sign_bit_codes_preserve_negative_zero() {
    let block = WeightBlock {
        packed: SIGNED,
        ..block()
    };
    let actual = floats(&block.decode(&[16], Dtype::F32).unwrap());
    let expected = [
        -0.0f32, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
        -4.0, -6.0,
    ];
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn nvfp4_rejects_malformed_lengths_shapes_and_dtype() {
    for packed in [&UNIT[..7], &[0u8; 9][..]] {
        invalid(
            &WeightBlock { packed, ..block() },
            &[16],
            Dtype::F32,
            "packed",
        );
    }
    for weight_scale in [&[][..], SCALES] {
        invalid(
            &WeightBlock {
                weight_scale,
                ..block()
            },
            &[16],
            Dtype::F32,
            "weight_scale",
        );
    }
    for shape in [&[][..], &[15], &[0], &[2, 8], &[u32::MAX, u32::MAX, 16]] {
        invalid(&block(), shape, Dtype::F32, "shape");
    }
    invalid(&block(), &[16], Dtype::U8, "dtype");
}

#[test]
fn nvfp4_scale_roles_round_trip_the_pinned_config_member_names() {
    let hf: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/nvfp4/hf_quant_config.json")).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/nvfp4/config.json")).unwrap();
    let index: serde_json::Value = serde_json::from_slice(include_bytes!(
        "fixtures/nvfp4/model.safetensors.index.json"
    ))
    .unwrap();
    assert_eq!(
        hf["quantization"]["quantized_layers"],
        config["quantization_config"]["quantized_layers"]
    );
    let group = &config["quantization_config"]["config_groups"]["group_1"];
    for member in ["weights", "input_activations"] {
        assert_eq!(group[member]["num_bits"], 4);
        assert_eq!(group[member]["group_size"], 16);
    }
    let mut roles = std::collections::BTreeMap::new();
    for name in index["weight_map"].as_object().unwrap().keys() {
        let (module, suffix) = name.rsplit_once('.').unwrap();
        if suffix == "weight" {
            continue;
        }
        let role = ScaleRole::parse(suffix).unwrap();
        assert_eq!(role.name(), suffix);
        *roles.entry(role.name()).or_insert(0) += 1;
        if suffix == "weight_scale_2" {
            assert_eq!(
                hf["quantization"]["quantized_layers"][module]["quant_algo"],
                "NVFP4"
            );
        }
    }
    assert_eq!(
        roles,
        std::collections::BTreeMap::from([
            ("input_scale", 2),
            ("weight_scale", 2),
            ("weight_scale_2", 1),
        ])
    );
    for unknown in [
        "k_scale",
        "v_scale",
        "input_scale_2",
        "weight_scale_3",
        "Weight_scale",
        "",
    ] {
        assert!(
            matches!(ScaleRole::parse(unknown), Err(EngError::BadRequest(message)) if message.contains(unknown))
        );
    }
}

#[test]
fn nvfp4_e4m3_table_covers_subnormal_normal_and_finite_only_boundaries() {
    use inference_engine::kernels::nvfp4::E4M3_VALUES;
    // E4M3FN has bias 7, subnormal steps 1/512, and a maximum finite value of 448.
    for (code, expected) in [
        (0x00, 0.0f32),
        (0x01, 0.001953125),
        (0x07, 0.013671875),
        (0x08, 0.015625),
        (0x38, 1.0),
        (0x40, 2.0),
        (0x77, 240.0),
        (0x78, 256.0),
        (0x7e, 448.0),
        (0x80, -0.0),
        (0x81, -0.001953125),
        (0xb8, -1.0),
        (0xfe, -448.0),
    ] {
        assert_eq!(E4M3_VALUES[code].to_bits(), expected.to_bits());
    }
    assert!(E4M3_VALUES[0x7f].is_nan());
    assert!(E4M3_VALUES[0xff].is_nan());
    assert!(!E4M3_VALUES.iter().any(|v| v.is_infinite()));
}

#[test]
fn nvfp4_multiple_rows_use_distinct_group_scales_and_preserve_activation_scale() {
    let packed = [0x22; 16]; // Every nibble is E2M1 1.0.
    let block = WeightBlock {
        packed: &packed,
        weight_scale: SCALES,
        weight_scale_2: 0.5,
        input_scale: Some(7.0),
    };
    let tensor = block.decode(&[2, 16], Dtype::F32).unwrap();
    assert_eq!(floats(&tensor), [vec![0.5; 16], vec![1.0; 16]].concat());
    assert_eq!(block.input_scale, Some(7.0));
}

#[test]
fn nvfp4_zero_and_subnormal_group_scales_decode_without_flush_to_zero() {
    for (scale, expected) in [(0x00, 0.0), (0x01, 0.001953125), (0x7e, 448.0)] {
        let block = WeightBlock {
            packed: &[0x22; 8],
            weight_scale: &[scale],
            ..block()
        };
        assert_eq!(
            floats(&block.decode(&[16], Dtype::F32).unwrap()),
            vec![expected; 16]
        );
    }
}

#[test]
fn nvfp4_rejects_invalid_scales_and_nonfinite_decoded_weights() {
    for code in [0x7f, 0xff, 0x80, 0xb8, 0xfe] {
        invalid(
            &WeightBlock {
                weight_scale: &[code],
                ..block()
            },
            &[16],
            Dtype::F32,
            "weight_scale",
        );
    }
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        invalid(
            &WeightBlock {
                weight_scale_2: scale,
                ..block()
            },
            &[16],
            Dtype::F32,
            "weight_scale_2",
        );
        invalid(
            &WeightBlock {
                input_scale: Some(scale),
                ..block()
            },
            &[16],
            Dtype::F32,
            "input_scale",
        );
    }
    invalid(
        &WeightBlock {
            weight_scale: &[0x7e],
            weight_scale_2: f32::MAX,
            ..block()
        },
        &[16],
        Dtype::F32,
        "weight_scale",
    );
    invalid(
        &WeightBlock {
            weight_scale_2: f32::MAX,
            ..block()
        },
        &[16],
        Dtype::F32,
        "decoded weight",
    );
}

#[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
#[test]
fn nvfp4_f16_requires_the_existing_half_support() {
    invalid(&block(), &[16], Dtype::F16, "dtype");
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn nvfp4_f16_view_preserves_hand_computed_values_and_rejects_overflow() {
    let block = WeightBlock {
        packed: SIGNED,
        ..block()
    };
    let tensor = block.decode(&[16], Dtype::F16).unwrap();
    assert_eq!(tensor.dtype(), Dtype::F16);
    let bits: Vec<u16> = tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    assert_eq!(
        bits,
        [
            0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000, 0xc200, 0xc400, 0xc600, 0x8000, 0xb800, 0xbc00,
            0xbe00, 0xc000, 0xc200, 0xc400, 0xc600
        ]
    );
    invalid(
        &WeightBlock {
            weight_scale_2: 65504.0,
            ..block
        },
        &[16],
        Dtype::F16,
        "F16",
    );
}
