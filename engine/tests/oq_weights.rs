use std::sync::Arc;

use inference_engine::{
    kernels::{PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, WeightSource},
};

fn fixture(name: &str) -> (Tensor, Tensor, Option<Tensor>) {
    let source = ResidentWeights::from_bytes(Arc::from(
        include_bytes!("fixtures/oq-packed.safetensors").as_slice(),
    ))
    .unwrap();
    (
        source.tensor(&format!("{name}.payload")).unwrap(),
        source.tensor(&format!("{name}.scales")).unwrap(),
        source.tensor(&format!("{name}.biases")),
    )
}

#[test]
fn affine_three_bit_weight_retains_the_reference_storage() {
    let (payload, scales, biases) = fixture("affine_3_g32");
    let owners = [
        payload.host_bytes().unwrap().as_ptr(),
        scales.host_bytes().unwrap().as_ptr(),
        biases.as_ref().unwrap().host_bytes().unwrap().as_ptr(),
    ];
    let spec = QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits: 3,
        group_size: 32,
    };
    let weight = PackedWeight::try_new(payload, scales, biases, vec![2, 64], spec).unwrap();
    assert_eq!(weight.quantization(), spec);
    assert_eq!(weight.logical_shape(), [2, 64]);
    assert_eq!(weight.payload().shape(), [2, 6]);
    assert_eq!(weight.payload().dtype(), Dtype::U32);
    assert_eq!(weight.scales().shape(), [2, 2]);
    assert_eq!(weight.scales().dtype(), Dtype::F32);
    assert_eq!(weight.biases().unwrap().shape(), [2, 2]);
    assert_eq!(weight.biases().unwrap().dtype(), Dtype::F32);
    assert_eq!(
        &weight.payload().host_bytes().unwrap()[..4],
        2610060215u32.to_le_bytes()
    );
    assert_eq!(
        &weight.scales().host_bytes().unwrap()[..4],
        (-2.25f32).to_le_bytes()
    );
    assert_eq!(
        &weight.biases().unwrap().host_bytes().unwrap()[..4],
        9.0f32.to_le_bytes()
    );
    assert_eq!(
        [
            weight.payload().host_bytes().unwrap().as_ptr(),
            weight.scales().host_bytes().unwrap().as_ptr(),
            weight.biases().unwrap().host_bytes().unwrap().as_ptr()
        ],
        owners
    );
}

fn affine(bits: u32, group_size: u32) -> QuantizationSpec {
    QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits,
        group_size,
    }
}

fn assert_checkpoint(result: Result<PackedWeight, inference_engine::EngError>, field: &str) {
    let Err(inference_engine::EngError::Checkpoint(message)) = result else {
        panic!("invalid packed storage must fail for {field}");
    };
    assert!(message.contains(field), "{message}");
}

#[test]
fn affine_widths_accept_the_declared_modes_and_reject_unsupported_bits() {
    for (bits, width, dtype) in [
        (2, 4, Dtype::F32),
        (3, 6, Dtype::F32),
        (4, 8, Dtype::Bf16),
        (5, 10, Dtype::F16),
        (6, 12, Dtype::Bf16),
        (8, 16, Dtype::F32),
    ] {
        let (payload, scales, biases) = fixture(&format!("affine_{bits}_g32"));
        let weight =
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(bits, 32)).unwrap();
        assert_eq!(weight.quantization(), affine(bits, 32));
        assert_eq!(weight.payload().shape(), [2, width]);
        assert_eq!(weight.scales().dtype(), dtype);
    }
    for bits in [0, 1, 7, 16, u32::MAX] {
        let (payload, scales, biases) = fixture("affine_3_g32");
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(bits, 32)),
            "bits",
        );
    }
}

#[test]
fn affine_groups_preserve_the_reference_layout_and_reject_other_sizes() {
    for (group_size, logical_width, packed_width) in [(32, 64, 8), (64, 64, 8), (128, 128, 16)] {
        let (payload, scales, biases) = fixture(&format!("affine_4_g{group_size}"));
        let weight = PackedWeight::try_new(
            payload,
            scales,
            biases,
            vec![2, logical_width],
            affine(4, group_size),
        )
        .unwrap();
        assert_eq!(weight.quantization().group_size, group_size);
        assert_eq!(weight.payload().shape(), [2, packed_width]);
    }
    for group_size in [0, 16, 48, 256, u32::MAX] {
        let (payload, scales, biases) = fixture("affine_3_g32");
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(3, group_size)),
            "group_size",
        );
    }
}

#[test]
fn packed_weights_require_matrix_rank_and_positive_logical_dimensions() {
    for shape in [vec![], vec![64], vec![0, 64], vec![2, 0], vec![0, 2, 64]] {
        let (payload, scales, biases) = fixture("affine_3_g32");
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, shape, affine(3, 32)),
            "logical shape",
        );
    }
}

#[test]
fn logical_element_count_cannot_overflow() {
    let (payload, scales, biases) = fixture("affine_3_g32");
    assert_checkpoint(
        PackedWeight::try_new(
            payload,
            scales,
            biases,
            vec![u32::MAX, u32::MAX, 64],
            affine(3, 32),
        ),
        "overflow",
    );
}

#[test]
fn logical_width_requires_complete_quantization_groups() {
    for width in [1, 31, 33, 63, 65] {
        let (payload, scales, biases) = fixture("affine_3_g32");
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, width], affine(3, 32)),
            "logical width",
        );
    }
}

#[test]
fn packed_payload_requires_u32_storage() {
    for (dtype, bytes) in [
        (Dtype::U8, 12),
        (Dtype::F16, 24),
        (Dtype::Bf16, 24),
        (Dtype::F32, 48),
        (Dtype::I64, 96),
    ] {
        let (_, scales, biases) = fixture("affine_3_g32");
        let payload = Tensor::try_from_host(dtype, vec![2, 6], vec![0; bytes]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(3, 32)),
            "payload dtype",
        );
    }
}

#[test]
fn payload_shape_must_match_the_full_logical_shape_and_bit_width() {
    for (shape, bytes) in [
        (vec![], 4),
        (vec![12], 48),
        (vec![2, 5], 40),
        (vec![2, 7], 56),
        (vec![1, 2, 6], 48),
        (vec![1, 6], 24),
    ] {
        let (_, scales, biases) = fixture("affine_3_g32");
        let payload = Tensor::try_from_host(Dtype::U32, shape, vec![0; bytes]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(3, 32)),
            "payload shape",
        );
    }
    for bits in [3, 5, 6] {
        let (payload, scales, biases) = fixture("affine_4_g32");
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(bits, 32)),
            "payload shape",
        );
    }
}

#[test]
fn affine_scales_require_floating_point_storage() {
    for (dtype, bytes) in [
        (Dtype::U8, 4),
        (Dtype::U32, 16),
        (Dtype::I64, 32),
        (Dtype::I16, 8),
        (Dtype::I32, 16),
    ] {
        let (payload, _, biases) = fixture("affine_3_g32");
        let scales = Tensor::try_from_host(dtype, vec![2, 2], vec![0; bytes]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(3, 32)),
            "scales dtype",
        );
    }
}

#[test]
fn affine_weights_require_additive_biases_with_matching_dtype() {
    let (payload, scales, _) = fixture("affine_3_g32");
    assert_checkpoint(
        PackedWeight::try_new(payload, scales, None, vec![2, 64], affine(3, 32)),
        "biases",
    );
    for (dtype, bytes) in [
        (Dtype::F16, 8),
        (Dtype::Bf16, 8),
        (Dtype::U8, 4),
        (Dtype::U32, 16),
        (Dtype::I64, 32),
    ] {
        let (payload, scales, _) = fixture("affine_3_g32");
        let biases = Tensor::try_from_host(dtype, vec![2, 2], vec![0; bytes]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, Some(biases), vec![2, 64], affine(3, 32)),
            "biases dtype",
        );
    }
}

#[test]
fn scale_and_bias_shapes_cannot_broadcast_or_omit_dimensions() {
    for (shape, bytes) in [
        (vec![], 4),
        (vec![4], 16),
        (vec![2, 1], 8),
        (vec![1, 2], 8),
        (vec![1, 2, 2], 16),
        (vec![2, 3], 24),
    ] {
        let (payload, _, biases) = fixture("affine_3_g32");
        let scales = Tensor::try_from_host(Dtype::F32, shape.clone(), vec![0; bytes]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, biases, vec![2, 64], affine(3, 32)),
            "scales shape",
        );
        let (payload, scales, _) = fixture("affine_3_g32");
        let biases = Tensor::try_from_host(Dtype::F32, shape, vec![0; bytes]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(payload, scales, Some(biases), vec![2, 64], affine(3, 32)),
            "biases shape",
        );
    }
}

#[test]
fn mxfp_settings_require_the_mode_bit_width_and_group_32() {
    for (mode, name, bits) in [
        (QuantizationMode::Mxfp4, "mxfp4_4_g32", 4),
        (QuantizationMode::Mxfp8, "mxfp8_8_g32", 8),
    ] {
        let (payload, scales, biases) = fixture(name);
        let spec = QuantizationSpec {
            mode,
            bits,
            group_size: 32,
        };
        let weight = PackedWeight::try_new(payload, scales, biases, vec![2, 64], spec).unwrap();
        assert_eq!(weight.quantization(), spec);
        for invalid_bits in [0, 3, 4, 5, 6, 8, u32::MAX] {
            if invalid_bits == bits {
                continue;
            }
            let (payload, scales, biases) = fixture(name);
            assert_checkpoint(
                PackedWeight::try_new(
                    payload,
                    scales,
                    biases,
                    vec![2, 64],
                    QuantizationSpec {
                        bits: invalid_bits,
                        ..spec
                    },
                ),
                "bits",
            );
        }
        for group_size in [0, 16, 64, 128, u32::MAX] {
            let (payload, scales, biases) = fixture(name);
            assert_checkpoint(
                PackedWeight::try_new(
                    payload,
                    scales,
                    biases,
                    vec![2, 64],
                    QuantizationSpec { group_size, ..spec },
                ),
                "group_size",
            );
        }
    }
}

#[test]
fn mxfp_scales_require_encoded_u8_storage() {
    for (mode, name, bits) in [
        (QuantizationMode::Mxfp4, "mxfp4_4_g32", 4),
        (QuantizationMode::Mxfp8, "mxfp8_8_g32", 8),
    ] {
        for (dtype, bytes) in [
            (Dtype::F16, 8),
            (Dtype::Bf16, 8),
            (Dtype::F32, 16),
            (Dtype::U32, 16),
            (Dtype::I64, 32),
        ] {
            let (payload, _, biases) = fixture(name);
            let scales = Tensor::try_from_host(dtype, vec![2, 2], vec![0; bytes]).unwrap();
            assert_checkpoint(
                PackedWeight::try_new(
                    payload,
                    scales,
                    biases,
                    vec![2, 64],
                    QuantizationSpec {
                        mode,
                        bits,
                        group_size: 32,
                    },
                ),
                "scales dtype",
            );
        }
    }
}

#[test]
fn mxfp_weights_reject_any_additive_bias() {
    for (mode, name, bits) in [
        (QuantizationMode::Mxfp4, "mxfp4_4_g32", 4),
        (QuantizationMode::Mxfp8, "mxfp8_8_g32", 8),
    ] {
        let (payload, scales, _) = fixture(name);
        let biases = Tensor::try_from_host(Dtype::U8, vec![2, 2], vec![0; 4]).unwrap();
        assert_checkpoint(
            PackedWeight::try_new(
                payload,
                scales,
                Some(biases),
                vec![2, 64],
                QuantizationSpec {
                    mode,
                    bits,
                    group_size: 32,
                },
            ),
            "biases",
        );
    }
}

#[test]
fn mxfp_reference_bytes_and_owners_survive_without_biases() {
    for (mode, name, bits, encoded_scale, first_word) in [
        (QuantizationMode::Mxfp4, "mxfp4_4_g32", 4, 128, 278576333u32),
        (
            QuantizationMode::Mxfp8,
            "mxfp8_8_g32",
            8,
            121,
            4109957884u32,
        ),
    ] {
        let (payload, scales, biases) = fixture(name);
        let owners = [
            payload.host_bytes().unwrap().as_ptr(),
            scales.host_bytes().unwrap().as_ptr(),
        ];
        let weight = PackedWeight::try_new(
            payload,
            scales,
            biases,
            vec![2, 64],
            QuantizationSpec {
                mode,
                bits,
                group_size: 32,
            },
        )
        .unwrap();
        assert!(weight.biases().is_none());
        assert_eq!(weight.scales().dtype(), Dtype::U8);
        assert_eq!(weight.scales().host_bytes().unwrap(), [encoded_scale; 4]);
        assert_eq!(
            &weight.payload().host_bytes().unwrap()[..4],
            first_word.to_le_bytes()
        );
        assert_eq!(
            [
                weight.payload().host_bytes().unwrap().as_ptr(),
                weight.scales().host_bytes().unwrap().as_ptr()
            ],
            owners
        );
    }
}

#[test]
fn batched_storage_requires_matching_leading_dimensions() {
    let (payload, scales, biases) = fixture("affine_3_g32");
    let p = Tensor::try_from_host(
        Dtype::U32,
        vec![2, 2, 6],
        payload.host_bytes().unwrap().repeat(2),
    )
    .unwrap();
    let s = Tensor::try_from_host(
        Dtype::F32,
        vec![2, 2, 2],
        scales.host_bytes().unwrap().repeat(2),
    )
    .unwrap();
    let b = Tensor::try_from_host(
        Dtype::F32,
        vec![2, 2, 2],
        biases.as_ref().unwrap().host_bytes().unwrap().repeat(2),
    )
    .unwrap();
    let weight = PackedWeight::try_new(
        p.clone(),
        s.clone(),
        Some(b.clone()),
        vec![2, 2, 64],
        affine(3, 32),
    )
    .unwrap();
    assert_eq!(weight.logical_shape(), [2, 2, 64]);
    assert_eq!(weight.payload().shape(), [2, 2, 6]);
    assert_eq!(weight.scales().shape(), [2, 2, 2]);
    assert_checkpoint(
        PackedWeight::try_new(
            payload,
            s.clone(),
            Some(b.clone()),
            vec![2, 2, 64],
            affine(3, 32),
        ),
        "payload shape",
    );
    assert_checkpoint(
        PackedWeight::try_new(p.clone(), scales, Some(b), vec![2, 2, 64], affine(3, 32)),
        "scales shape",
    );
    assert_checkpoint(
        PackedWeight::try_new(p, s, biases, vec![2, 2, 64], affine(3, 32)),
        "biases shape",
    );
}
