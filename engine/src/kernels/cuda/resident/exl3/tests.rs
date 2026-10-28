//! EXL3 device checks: run only in the canonical three-lock GPU window.
use super::Exl3PrepCall;
use crate::kernels::cuda::{CudaContext, testkit};
use crate::tensor::{Dtype, Tensor};
use crate::weights::exl3::Codebook;

#[test]
fn midrows_policy_keeps_small_rows_legacy_and_honors_rollback() {
    use std::ffi::OsStr;
    for (rows, enabled) in [
        (0, false),
        (8, false),
        (9, false),
        (32, false),
        (63, false),
        (64, true),
        (128, true),
        (144, true),
        (145, false),
        (usize::MAX, false),
    ] {
        assert_eq!(super::midrows_enabled(rows, None), enabled, "rows{rows}");
        assert!(!super::midrows_enabled(rows, Some(OsStr::new("1"))));
    }
    for value in ["", "0", "01", "true"] {
        assert!(super::midrows_enabled(64, Some(OsStr::new(value))));
    }
}

#[test]
fn all_codebooks_and_rates_prepare_bit_exact_nonsquare_wprime() {
    if !testkit::device_present("EXL3 W-prime all codebooks/rates") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x4558_4c33);
    let (k, n) = (128usize, 256usize);
    for book in [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1] {
        for bits_twice in 2u8..=16 {
            if bits_twice % 2 != 0 && !(book == Codebook::Mul1 && matches!(bits_twice, 3 | 5 | 7)) {
                continue;
            }
            let words = usize::from(bits_twice) * 8;
            let bytes: Vec<u8> = (0..k * n / 256 * words)
                .flat_map(|_| (rng.next_u64() as u16).to_le_bytes())
                .collect();
            let tensor = Tensor::try_from_host(
                Dtype::I16,
                vec![(k / 16) as u32, (n / 16) as u32, words as u32],
                bytes.clone(),
            )
            .expect("trellis");
            let mut expected = vec![0u16; k * n];
            book.reconstruct_into(&tensor, &mut expected)
                .expect("CPU format oracle");
            let trellis = testkit::upload_bytes(&ctx, &bytes);
            let output = testkit::upload_filled(&ctx, 0xff, k * n * 2);
            ctx.exl3_prep(
                &stream,
                Exl3PrepCall {
                    trellis: trellis.range(0, bytes.len()).unwrap(),
                    out: output.range(0, k * n * 2).unwrap(),
                    k,
                    n,
                    bits_twice,
                    book,
                },
            )
            .expect("GPU prep");
            stream.synchronize().expect("drain");
            let actual = testkit::read_bytes(&ctx, &output, k * n * 2);
            let expected: Vec<u8> = expected
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect();
            assert_eq!(actual, expected, "{book:?}, bits_twice={bits_twice}");
        }
    }
}
