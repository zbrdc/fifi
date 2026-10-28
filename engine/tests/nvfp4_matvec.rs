//! Fused NVFP4 matvec parity and rejection tests.

use inference_engine::kernels::nvfp4::WeightBlock;
use inference_engine::tensor::Dtype;

#[test]
#[allow(clippy::chunks_exact_to_as_chunks)]
fn matvec_matches_a_decoded_row_dotted_with_the_same_activation() {
    let packed = [
        0x10u8, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32,
        0x10,
    ];
    let scales = [0x40u8, 0x40];
    let block = WeightBlock {
        packed: &packed,
        weight_scale: &scales,
        weight_scale_2: 0.5,
        input_scale: None,
    };
    let decoded = block.decode(&[2, 16], Dtype::F32).expect("decode");
    let bytes = decoded.host_bytes().expect("bytes");
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let x: Vec<f32> = (0..16).map(|index| index as f32 * 0.25 - 1.0).collect();
    let expected: Vec<f32> = values
        .chunks_exact(16)
        .map(|row| row.iter().zip(&x).map(|(w, v)| w * v).sum())
        .collect();
    let mut actual = vec![0.0f32; 2];
    block.matvec(2, 16, &x, &mut actual).expect("matvec");
    assert_eq!(actual, expected);
}

#[test]
fn matvec_rejects_mismatched_extents_and_non_finite_activations() {
    let packed = [
        0x10u8, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32,
        0x10,
    ];
    let scales = [0x40u8, 0x40];
    let block = WeightBlock {
        packed: &packed,
        weight_scale: &scales,
        weight_scale_2: 0.5,
        input_scale: None,
    };
    let good = vec![1.0f32; 16];
    let mut out = vec![0.0f32; 2];
    block.matvec(2, 16, &good, &mut out).expect("valid");
    assert!(block.matvec(0, 16, &good, &mut out).is_err());
    assert!(block.matvec(2, 15, &good, &mut out).is_err());
    assert!(block.matvec(1, 16, &good, &mut out).is_err());
    assert!(block.matvec(2, 16, &good[..8], &mut out).is_err());
    let mut short: Vec<f32> = Vec::new();
    assert!(block.matvec(2, 16, &good, &mut short).is_err());
    let mut bad = good.clone();
    bad[3] = f32::NAN;
    assert!(block.matvec(2, 16, &bad, &mut out).is_err());
}
