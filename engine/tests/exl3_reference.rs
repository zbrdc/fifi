use inference_engine::{
    EngError,
    tensor::{Dtype, Tensor},
    weights::exl3::Codebook,
};

#[test]
fn codebook_states_follow_the_published_half_rules() {
    for (book, expected) in [
        (Codebook::ThreeInst, [0x3a25, 0xbb5b, 0xb110]),
        (Codebook::Mcg, [0x3f60, 0x304e, 0x3acd]),
        (Codebook::Mul1, [0xc2e8, 0x3921, 0xb936]),
    ] {
        for (state, expected) in [0, 1, u16::MAX].into_iter().zip(expected) {
            assert_eq!(
                book.decode_state(state),
                expected,
                "{book:?}, state={state}"
            );
        }
    }
}

#[test]
fn tile_uses_msb_stream_order_and_the_format_permutation() {
    let mut words = [0; 16];
    words[1] = 0x8000;
    let positions = [
        0, 16, 128, 144, 8, 24, 136, 152, 32, 48, 160, 176, 40, 56, 168, 184,
    ];
    for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
        let mut actual = [0; 256];
        book.decode_tile(&words, &mut actual).unwrap();
        let mut expected = [book.decode_state(0); 256];
        for (shift, position) in positions.into_iter().enumerate() {
            expected[position] = book.decode_state(1 << shift);
        }
        assert_eq!(actual, expected, "{book:?}");
    }
}

#[test]
fn tile_wraps_the_last_stream_bit_into_the_first_state() {
    let mut words = [0; 16];
    words[14] = 1;
    let mut actual = [0; 256];
    Codebook::Mul1.decode_tile(&words, &mut actual).unwrap();
    assert_eq!(actual[0], Codebook::Mul1.decode_state(2));
    assert_eq!(actual[255], Codebook::Mul1.decode_state(1));
}

#[test]
fn tile_accepts_only_supported_rates_without_partial_writes_on_refusal() {
    for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
        for count in 0..=136 {
            let integer = matches!(count, 16 | 32 | 48 | 64 | 80 | 96 | 112 | 128);
            let half = book == Codebook::Mul1 && matches!(count, 24 | 40 | 56);
            let mut actual = [0x7e00; 256];
            let result = book.decode_tile(&vec![0; count], &mut actual);
            if integer || half {
                result.unwrap();
                assert_eq!(
                    actual,
                    [book.decode_state(0); 256],
                    "{book:?}, words={count}"
                );
            } else {
                assert!(
                    matches!(result, Err(EngError::Checkpoint(_))),
                    "{book:?}, words={count}"
                );
                assert_eq!(actual, [0x7e00; 256], "{book:?}, words={count}");
            }
        }
    }
}

fn bit_walk_tile(book: Codebook, words: &[u16]) -> [u16; 256] {
    let stream: Vec<_> = words
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|pair| {
            let word = u32::from(pair[0]) | (u32::from(pair[1]) << 16);
            (0..32).rev().map(move |bit| ((word >> bit) & 1) as u16)
        })
        .collect();
    let mut output = [0; 256];
    for row in 0..16 {
        for col in 0..16 {
            let p = 32 * (col % 8) + 8 * ((row % 8) / 2) + row % 2 + 2 * (row / 8) + 4 * (col / 8);
            let end = (p + 1) * words.len() / 16;
            let begin = (end + stream.len() - 16) % stream.len();
            let state = (0..16).fold(0u16, |s, bit| {
                (s << 1) | stream[(begin + bit) % stream.len()]
            });
            output[row * 16 + col] = book.decode_state(state);
        }
    }
    output
}

#[test]
fn all_rates_match_an_independent_bit_by_bit_format_walk() {
    let mut seed = 0x1234_5678u32;
    for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
        for count in [16usize, 32, 48, 64, 80, 96, 112, 128, 24, 40, 56] {
            if !count.is_multiple_of(16) && book != Codebook::Mul1 {
                continue;
            }
            let words: Vec<_> = (0..count)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 16) as u16
                })
                .collect();
            let mut actual = [0; 256];
            book.decode_tile(&words, &mut actual).unwrap();
            assert_eq!(
                actual,
                bit_walk_tile(book, &words),
                "{book:?}, words={count}"
            );
        }
    }
}

#[test]
fn matrix_places_input_and_output_tiles_on_their_declared_axes() {
    let mut words = vec![0u16; 8 * 16 * 16];
    words[(16 + 2) * 16 + 1] = 0x8000;
    let bytes = words.into_iter().flat_map(u16::to_le_bytes).collect();
    let trellis = Tensor::try_from_host(Dtype::I16, vec![8, 16, 16], bytes).unwrap();
    let book = Codebook::Mcg;
    let mut actual = vec![0; 128 * 256];
    book.reconstruct_into(&trellis, &mut actual).unwrap();
    let mut expected = vec![book.decode_state(0); actual.len()];
    let positions = [
        0, 16, 128, 144, 8, 24, 136, 152, 32, 48, 160, 176, 40, 56, 168, 184,
    ];
    for (shift, position) in positions.into_iter().enumerate() {
        expected[(16 + position / 16) * 256 + 32 + position % 16] = book.decode_state(1 << shift);
    }
    assert_eq!(actual, expected);
}

#[test]
fn matrix_refuses_invalid_metadata_before_touching_output() {
    for (dtype, shape) in [
        (Dtype::I32, vec![8, 8, 16]),
        (Dtype::F16, vec![8, 8, 16]),
        (Dtype::I16, vec![8, 8]),
        (Dtype::I16, vec![0, 8, 16]),
        (Dtype::I16, vec![1, 8, 16]),
        (Dtype::I16, vec![8, 1, 16]),
        (Dtype::I16, vec![8, 8, 8]),
        (Dtype::I16, vec![8, 8, 24]),
    ] {
        let width = if dtype == Dtype::I32 { 4 } else { 2 };
        let len = shape.iter().map(|&d| d as usize).product::<usize>() * width;
        let trellis = Tensor::try_from_host(dtype, shape.clone(), vec![0; len]).unwrap();
        let mut output = [0x7e00];
        assert!(
            matches!(
                Codebook::Mcg.reconstruct_into(&trellis, &mut output),
                Err(EngError::Checkpoint(_))
            ),
            "{dtype:?}, shape={shape:?}"
        );
        assert_eq!(output, [0x7e00]);
    }
    let trellis = Tensor::try_from_host(Dtype::I16, vec![8, 8, 16], vec![0; 2048]).unwrap();
    let mut output = [0x7e00];
    assert!(matches!(
        Codebook::Mcg.reconstruct_into(&trellis, &mut output),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(output, [0x7e00]);
}
