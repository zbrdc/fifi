use crate::kernels::cuda::native::{
    RawDevice,
    resident::{RawRange, Span},
};
use crate::kernels::{cuda::testkit::silu_f64, nvfp4::WeightBlock};
use crate::weights::{GgufType, dequantize_into, repack_nvfp4_into};
use half::bf16;

// ggml-common.h's doubled E2M1 values require the UE4M3 decoder's compensating /2.
const INTEGER_E2M1: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[test]
fn dense_ffn_shape_uses_actual_nvfp4_blocks_without_legacy_metadata() {
    let shape = super::Nvfp4DenseShape::new(2048, 6144, 5).expect("Flint L0 gate, p01");
    let ints = shape.ints;
    assert_eq!(
        (ints.ty, ints.k, ints.n, ints.m, ints.ld, ints.col_off),
        (40, 2048, 6144, 5, 6144, 0)
    );
    assert_eq!(
        (shape.needs.w, shape.needs.y, shape.needs.dst),
        (7_077_888, 29_952, 122_880)
    );
    assert_eq!(shape.factor_bytes, 528);
}

#[test]
fn dense_shape_and_projection_factors_refuse_invalid_inputs() {
    for (k, n, m) in [
        (0, 6144, 5),
        (384, 6144, 5),
        (33024, 6144, 5),
        (2048, 0, 5),
        (2048, 6144, 0),
        (2048, 6144, 2049),
        (2048, usize::MAX, 5),
        (2048, i32::MAX as usize, 2),
    ] {
        assert!(super::Nvfp4DenseShape::new(k, n, m).is_err());
    }
    let shape = super::Nvfp4DenseShape::new(2048, 6144, 5).unwrap();
    for factor in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(shape.factors(factor).is_err());
    }
    for factor in [f32::from_bits(1), 0.375, 1.0, f32::MAX] {
        let bytes = shape.factors(factor).unwrap();
        assert_eq!(bytes.len(), shape.factor_bytes);
        for word in bytes.as_chunks::<4>().0 {
            assert_eq!(f32::from_ne_bytes(*word).to_bits(), factor.to_bits());
        }
    }
}

#[test]
fn review_dense_factors_cover_every_tail_tile() {
    for m in [1, 5, 64, 128, 144, 2048] {
        let shape = super::Nvfp4DenseShape::new(2048, 6144, m).unwrap();
        let bytes = shape.factors(0.375).unwrap();
        for j in (8..=128).step_by(8) {
            let reach = m.div_ceil(j) * j * 4;
            assert!(
                bytes.len() >= reach,
                "M={m}, J={j}, capacity={}",
                bytes.len()
            );
        }
    }
}

#[test]
fn review_dense_weight_blocks_stay_in_signed_native_indices() {
    let n = i32::MAX as usize / (32768 / 64);
    assert!(super::Nvfp4DenseShape::new(32768, n, 1).is_ok());
    assert!(super::Nvfp4DenseShape::new(32768, n + 1, 1).is_err());
    assert!(super::Nvfp4DenseShape::new(32768, 4_194_432, 1).is_err());
}

fn gguf_matrix() -> Vec<u8> {
    let mut blocks = Vec::new();
    for scale in [0x38, 0x40, 0x48, 0x50] {
        for _ in 0..4 {
            blocks.extend_from_slice(&[scale; 4]);
            blocks.extend_from_slice(&[0x21; 32]);
        }
    }
    blocks
}

#[test]
fn original_gguf_and_canonical_fixture_have_known_projection_dots() {
    let source = gguf_matrix();
    let x: Vec<f32> = (0..256)
        .map(|column| if column % 16 < 8 { 1.0 } else { 2.0 })
        .collect();
    let mut original = [0.0; 1024];
    dequantize_into(GgufType::Nvfp4, &source, &mut original).unwrap();
    let mut packed = [0; 512];
    let mut scales = [0; 64];
    repack_nvfp4_into(&source, &mut packed, &mut scales).unwrap();
    let mut canonical = [0.0; 4];
    WeightBlock {
        packed: &packed,
        weight_scale: &scales,
        weight_scale_2: 1.0,
        input_scale: None,
    }
    .matvec(4, 256, &x, &mut canonical)
    .unwrap();
    let expected = [320.0, 640.0, 1280.0, 2560.0];
    assert_eq!(canonical, expected);
    for (row, expected) in original.as_chunks::<256>().0.iter().zip(expected) {
        let dot: f64 = row
            .iter()
            .zip(&x)
            .map(|(&weight, &activation)| f64::from(weight) * f64::from(activation))
            .sum();
        assert_eq!(dot, f64::from(expected));
    }
}

#[test]
fn integer_reference_and_half_scale_match_both_weight_layouts() {
    let mut source = vec![0x00, 0x01, 0x38, 0x7e];
    for group in 0..4u8 {
        for lane in 0..8u8 {
            let low = (lane + group * 4) & 15;
            let high = (low + 8) & 15;
            source.push(low | (high << 4));
        }
    }
    let mut original = [0.0; 64];
    dequantize_into(GgufType::Nvfp4, &source, &mut original).unwrap();
    let mut packed = [0; 32];
    let mut scales = [0; 4];
    repack_nvfp4_into(&source, &mut packed, &mut scales).unwrap();
    let weight = WeightBlock {
        packed: &packed,
        weight_scale: &scales,
        weight_scale_2: 1.0,
        input_scale: None,
    };
    let half_scales = [0.0, 1.0 / 1024.0, 0.5, 224.0];
    let mut basis = [0.0; 64];
    for column in 0..64 {
        let group = column / 16;
        let code = (column % 16 + group * 4) % 16;
        let expected = f32::from(INTEGER_E2M1[code]) * half_scales[group];
        basis[column] = 1.0;
        let mut canonical = [0.0];
        weight.matvec(1, 64, &basis, &mut canonical).unwrap();
        basis[column] = 0.0;
        assert_eq!(original[column], expected, "GGUF column {column}");
        assert_eq!(canonical[0], expected, "canonical column {column}");
        if expected != 0.0 {
            assert_ne!(canonical[0], expected * 2.0, "missing /2 column {column}");
        }
    }
}

#[test]
fn compact_projection_factors_preserve_bf16_publication_before_down_scaling() {
    let weights = WeightBlock {
        packed: &[0x21; 128],
        weight_scale: &[0x38; 16],
        weight_scale_2: 1.0,
        input_scale: None,
    };
    let x = [1.0, 2.0].repeat(128);
    let mut plain = [0.0];
    weights.matvec(1, 256, &x, &mut plain).unwrap();
    assert_eq!(plain, [320.0]);
    let gate_scales = [1.0 / 160.0, 1.0 / 320.0, 1.0 / 640.0];
    let up_scales = [1.0 / 320.0, 1.0 / 160.0, 1.0 / 80.0];
    let down_scales = [3.0 / 97.0, 7.0 / 97.0, 11.0 / 97.0];
    let expected = [
        -0.1708984375,
        -0.01470947265625,
        -0.1708984375,
        -0.07763671875,
    ];
    for (expert, expected) in [2, 0, 2, 1].into_iter().zip(expected) {
        let gate = -plain[0] * gate_scales[expert];
        let up = (plain[0] * 2.0) * up_scales[expert];
        let silu = silu_f64(f64::from(gate)) as f32;
        let hidden = bf16::from_f32(silu * up).to_f32();
        let published = bf16::from_f32(hidden * down_scales[expert]).to_f32();
        assert_eq!(f64::from(published), expected, "compact expert {expert}");
        if expert == 0 {
            assert_eq!(hidden, -0.4765625);
            let folded = bf16::from_f32(silu * (up * down_scales[expert])).to_f32();
            assert_eq!(f64::from(folded), -0.0147705078125);
            assert_ne!(published, folded, "down cannot move before hidden BF16");
            let late_gate_scale = (silu_f64(f64::from(-plain[0])) as f32) * gate_scales[expert];
            assert_ne!(silu, late_gate_scale, "gate scale must precede SiLU");
        }
    }
}

#[test]
fn host_range_fixture_distinguishes_output_aliases_from_adjacent_sidebands() {
    let weight_bytes = gguf_matrix().len();
    let arena = RawDevice::unbacked(weight_bytes + 16 + 64);
    let weight = RawRange::new(&arena, 0, weight_bytes).unwrap();
    let sideband = RawRange::new(&arena, weight_bytes, 16).unwrap();
    let output = RawRange::new(&arena, weight_bytes + 16, 64).unwrap();
    let weight_span = Span::of(&weight, weight.bytes());
    let sideband_span = Span::of(&sideband, sideband.bytes());
    let output_span = Span::of(&output, output.bytes());
    assert!(weight_span.apart(&sideband_span));
    assert!(weight_span.apart(&output_span));
    assert!(sideband_span.apart(&output_span));
    for offset in [0, weight_bytes - 4] {
        let aliased = RawRange::new(&arena, offset, 64).unwrap();
        let aliased_span = Span::of(&aliased, aliased.bytes());
        assert!(!weight_span.apart(&aliased_span));
        assert!(!aliased_span.apart(&weight_span));
    }
    let sideband_alias = RawRange::new(&arena, weight_bytes, 64).unwrap();
    let sideband_alias_span = Span::of(&sideband_alias, sideband_alias.bytes());
    assert!(weight_span.apart(&sideband_alias_span));
    assert!(!sideband_span.apart(&sideband_alias_span));
    let other = RawDevice::unbacked(64);
    let separate = RawRange::new(&other, 0, 64).unwrap();
    assert!(weight_span.apart(&Span::of(&separate, separate.bytes())));
    let short = RawDevice::unbacked(weight_bytes - 1);
    assert!(RawRange::new(&short, 0, weight_bytes).is_none());
    assert!(RawRange::new(&arena, 0, 0).is_none());
    assert!(RawRange::new(&arena, usize::MAX, 2).is_none());
}

#[test]
fn short_or_canonically_unrepresentable_gguf_fixture_does_not_publish_repack() {
    let source = gguf_matrix();
    let mut unrepresentable = source.clone();
    unrepresentable[source.len() - 36] = 0xff;
    let mut original = [0.0; 1024];
    dequantize_into(GgufType::Nvfp4, &unrepresentable, &mut original).unwrap();
    assert_eq!(original[960], 240.0, "original GGUF 0xff is finite");
    for bytes in [&source[..source.len() - 1], &unrepresentable] {
        let mut packed = [0xa5; 512];
        let mut scales = [0xa5; 64];
        assert!(repack_nvfp4_into(bytes, &mut packed, &mut scales).is_err());
        assert_eq!(packed, [0xa5; 512]);
        assert_eq!(scales, [0xa5; 64]);
    }
    let mut signed_source = source;
    signed_source[0] = 0xb8;
    let mut packed = [0; 512];
    let mut scales = [0; 64];
    repack_nvfp4_into(&signed_source, &mut packed, &mut scales).unwrap();
    assert_eq!(scales[0], 0x38, "original GGUF ignores bit7");
}

#[test]
fn canonical_descriptor_and_first_group_refusals_leave_output_untouched() {
    let source = gguf_matrix();
    let mut packed = [0; 512];
    let mut scales = [0; 64];
    repack_nvfp4_into(&source, &mut packed, &mut scales).unwrap();
    let reject = |packed: &[u8], scales: &[u8], factor| {
        let weight = WeightBlock {
            packed,
            weight_scale: scales,
            weight_scale_2: factor,
            input_scale: None,
        };
        let mut output = [12345.0; 4];
        assert!(weight.matvec(4, 256, &[1.0; 256], &mut output).is_err());
        assert_eq!(output, [12345.0; 4]);
    };
    reject(&packed[..511], &scales, 1.0);
    reject(&packed, &scales[..63], 1.0);
    for factor in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        reject(&packed, &scales, factor);
    }
    for code in [0x80, 0xb8, 0x7f, 0xff] {
        let mut invalid = scales;
        invalid[0] = code;
        reject(&packed, &invalid, 1.0);
    }
}
