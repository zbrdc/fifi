use inference_engine::{
    EngError,
    kernels::f8::WeightBlock,
    tensor::{Dtype, Tensor},
};

const CODES: &[u8] = include_bytes!("fixtures/f8/codes.bin");
const SCALE_2: &[u8] = include_bytes!("fixtures/f8/weight-scale-2.bin");
const SCALE_QUARTER: &[u8] = include_bytes!("fixtures/f8/weight-scale-quarter.bin");
const SCALE_448: &[u8] = include_bytes!("fixtures/f8/weight-scale-448.bin");
const INPUT_SCALE: &[u8] = include_bytes!("fixtures/f8/input-scale.bin");

fn block<'a>(
    codes: &'a [u8],
    weight_scale: &'a [u8],
    input_scale: Option<&'a [u8]>,
) -> WeightBlock<'a> {
    WeightBlock {
        codes,
        weight_scale,
        input_scale,
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
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn invalid(block: &WeightBlock<'_>, shape: &[u32], field: &str) {
    match block.decode(shape) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains(field), "expected {field}, got {message}")
        }
        other => panic!("expected invalid {field}, got {other:?}"),
    }
}

#[test]
fn f8_hand_vectors_decode_and_keep_input_scale_separate() {
    let (tensor, input_scale) = block(&CODES[..1], SCALE_2, Some(INPUT_SCALE))
        .decode(&[1])
        .unwrap();
    assert_eq!(floats(&tensor), [1.0]); // E4M3FN 0x30 is 0.5; 0.5 * 2 = 1.
    assert_eq!(input_scale, Some(7.0));

    let (tensor, input_scale) = block(&CODES[1..2], SCALE_QUARTER, None)
        .decode(&[1])
        .unwrap();
    assert_eq!(floats(&tensor), [0.5]); // E4M3FN 0x40 is 2.0; 2 * 0.25 = 0.5.
    assert_eq!(input_scale, None);

    let (tensor, _) = block(&CODES[2..3], SCALE_2, None).decode(&[1]).unwrap();
    let negative_zero = floats(&tensor)[0];
    assert_eq!(negative_zero, 0.0);
    assert!(negative_zero.is_sign_negative());

    let (tensor, _) = block(&CODES[3..4], SCALE_448, None).decode(&[1]).unwrap();
    assert_eq!(floats(&tensor), [0.875]); // E4M3FN subnormal 2^-9 times 448.
}

#[test]
fn f8_rejects_nan_codes_wrong_lengths_and_invalid_scales() {
    for code in [0x7f, 0xff] {
        invalid(&block(&[code], SCALE_2, None), &[1], "weight code");
    }
    invalid(&block(&CODES[..3], SCALE_2, None), &[4], "codes");
    invalid(
        &block(&CODES[..1], &SCALE_2[..3], None),
        &[1],
        "weight_scale",
    );
    invalid(&block(&CODES[..1], &[0; 5], None), &[1], "weight_scale");
    invalid(
        &block(&CODES[..1], SCALE_2, Some(&INPUT_SCALE[..3])),
        &[1],
        "input_scale",
    );

    invalid(&block(&[], SCALE_2, None), &[u32::MAX; 3], "shape product");
    invalid(
        &block(&[], SCALE_2, None),
        &[u32::MAX, u32::MAX],
        "shape byte length",
    );
    let maximum_scale = f32::MAX.to_le_bytes();
    invalid(
        &block(&[0x7e], &maximum_scale, None),
        &[1],
        "decoded weight",
    );

    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bytes = scale.to_le_bytes();
        invalid(&block(&CODES[..1], &bytes, None), &[1], "weight_scale");
        invalid(
            &block(&CODES[..1], SCALE_2, Some(&bytes)),
            &[1],
            "input_scale",
        );
    }
}
