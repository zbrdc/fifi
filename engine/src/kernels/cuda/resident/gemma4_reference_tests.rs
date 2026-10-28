//! Pinned Humanizer Q8_0 row slices and independently packed Q8_1 activations.
use super::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::{CudaContext, testkit};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const CASES: &[(usize, &str, &str)] = &[
    (
        3840,
        "hidden-3840.bin",
        "a0b2c922c9587effb66f742c2cb991c06458fc21c38fb154c0cd85bf4840e2e4",
    ),
    (
        4096,
        "local-out-4096.bin",
        "dbb352c1acd0327b3008b14d07965fbc6a433e1615ac5e2b682a0a92e7cc8fa5",
    ),
    (
        8192,
        "global-out-8192.bin",
        "b7123070417ae717da5dcbbacb038d68cc1deadfcb2d00f6992369f0d34a0100",
    ),
    (
        15360,
        "ffn-down-15360.bin",
        "10341f2fd0f52b875cb8b98322fe5381ee2bcb41248ceed8343f6502035fafb7",
    ),
];

#[test]
#[ignore = "requires externally generated pinned Humanizer fixtures"]
fn real_q80_rows_match_frozen_quantized_activation_dots() {
    if !testkit::device_present("Humanizer real Q80 matvec") {
        return;
    }
    let root = PathBuf::from(
        std::env::var_os("INFERENCE_GEMMA_MATVEC_FIXTURES")
            .expect("the explicit real-fixture run needs INFERENCE_GEMMA_MATVEC_FIXTURES"),
    );
    for &(k, name, sha) in CASES {
        let fixture = std::fs::read(root.join(name)).expect("frozen fixture");
        assert_eq!(
            format!("{:x}", Sha256::digest(&fixture)),
            sha,
            "fixture identity"
        );
        let w_bytes = 4 * (k / 32) * 34;
        let x_bytes = k.div_ceil(512) * 512 / 32 * 36;
        assert_eq!(fixture.len(), w_bytes + x_bytes + 64);
        let w_input = &fixture[..w_bytes];
        let x_input = &fixture[w_bytes..w_bytes + x_bytes];
        let expected = fixture[w_bytes + x_bytes..].as_chunks::<8>().0;
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let w = testkit::upload_bytes(&ctx, w_input);
        let x = testkit::upload_bytes(&ctx, x_input);
        let y = testkit::upload_bytes(&ctx, &[0xA5; 80]);
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty: QuantType::Q80,
                w: w.range(0, w_bytes).expect("weights"),
                x_q8: x.range(0, x_bytes).expect("packed activations"),
                y: y.range(32, 16).expect("guarded output"),
                k,
                rows: 4,
                ncols: 1,
            },
        )
        .expect("projection");
        stream.synchronize().expect("completion");
        let output = testkit::read_bytes(&ctx, &y, 80);
        assert_eq!(&output[..32], &[0xA5; 32], "output prefix k={k}");
        assert_eq!(&output[48..], &[0xA5; 32], "output suffix k={k}");
        assert_eq!(
            testkit::read_bytes(&ctx, &w, w_bytes),
            w_input,
            "weights changed"
        );
        assert_eq!(
            testkit::read_bytes(&ctx, &x, x_bytes),
            x_input,
            "activations changed"
        );
        for (row, actual) in testkit::floats(&output[32..48]).into_iter().enumerate() {
            let value = f64::from_le_bytes(expected[row]);
            let sum_abs = f64::from_le_bytes(expected[4 + row]);
            let bound = 64.0 * f64::from(f32::EPSILON) / 2.0 * sum_abs;
            assert!(
                (f64::from(actual) - value).abs() <= bound,
                "k={k} row={row}: {actual} versus {value}, bound={bound}"
            );
        }
    }
}

const RMS_CASES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/rmsnorm-frozen.bin"
));

#[test]
fn rmsnorm_rows_match_the_eighteen_frozen_cases() {
    use super::add_rmsnorm_q8::AddNormQ8Call;
    if !testkit::device_present("Gemma weighted and unweighted RMSNorm") {
        return;
    }
    assert_eq!(
        u32::from_le_bytes(RMS_CASES[..4].try_into().expect("count")),
        18
    );
    let mut cursor = 4;
    for case in 0..18 {
        let n =
            u32::from_le_bytes(RMS_CASES[cursor..cursor + 4].try_into().expect("width")) as usize;
        let epsilon = f32::from_le_bytes(
            RMS_CASES[cursor + 4..cursor + 8]
                .try_into()
                .expect("epsilon"),
        );
        cursor += 8;
        let input = &RMS_CASES[cursor..cursor + 4 * n];
        let weights = &RMS_CASES[cursor + 4 * n..cursor + 8 * n];
        let expected = RMS_CASES[cursor + 8 * n..cursor + 16 * n]
            .as_chunks::<8>()
            .0;
        cursor += 16 * n;
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let h = testkit::upload_bytes(&ctx, input);
        let w = testkit::upload_bytes(&ctx, weights);
        let output = testkit::upload_bytes(&ctx, &vec![0xA5; n * 4 + 64]);
        let packed_bytes = super::quantize_q8_1::q8_1_bytes(n, 1).expect("packed size");
        let packed = testkit::upload_bytes(&ctx, &vec![0xA5; packed_bytes + 64]);
        ctx.add_rmsnorm_q8(
            &stream,
            AddNormQ8Call {
                n,
                eps: epsilon,
                h: h.range(0, n * 4).expect("input"),
                delta: None,
                weight: w.range(0, n * 4).expect("weight"),
                norm_out: output.range(32, n * 4).expect("guarded output"),
                q8: packed.range(32, packed_bytes).expect("guarded pack"),
            },
        )
        .expect("normalization");
        stream.synchronize().expect("completion");
        let actual = testkit::read_bytes(&ctx, &output, n * 4 + 64);
        let codes = testkit::read_bytes(&ctx, &packed, packed_bytes + 64);
        assert_eq!(&actual[..32], &[0xA5; 32], "output prefix");
        assert_eq!(&actual[n * 4 + 32..], &[0xA5; 32], "output suffix");
        assert_eq!(&codes[..32], &[0xA5; 32], "pack prefix");
        assert_eq!(&codes[packed_bytes + 32..], &[0xA5; 32], "pack suffix");
        assert_eq!(testkit::read_bytes(&ctx, &h, n * 4), input, "input changed");
        assert_eq!(
            testkit::read_bytes(&ctx, &w, n * 4),
            weights,
            "weight changed"
        );
        for (index, value) in testkit::floats(&actual[32..n * 4 + 32]).iter().enumerate() {
            let reference = f64::from_le_bytes(expected[index]);
            assert!(
                (f64::from(*value) - reference).abs() <= 2e-6 + 2e-5 * reference.abs(),
                "case={case} n={n} element={index}: {value} versus {reference}"
            );
        }
    }
    assert_eq!(cursor, RMS_CASES.len());
}
