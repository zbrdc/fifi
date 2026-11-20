//! Tests of the scatter_cols wrapper: (1) the scattered block equals the source in its columns
//! and every other byte of the destination keeps its canary (rows x widths x offsets, among them
//! the DeltaNet input row of the model); (2) the pure shape and extent rules; (3) the refusal
//! table, each rule reached and asserted by its own text.
//!
//! Device-test structure: every device phase returns host `Vec`s, the owners drop, then the
//! asserts run. A host without a device prints `SKIPPED` through `testkit::device_present`.

use super::{
    MAX_ROWS, MAX_WIDTH, ScatterColsCall, ScatterColsShape, check_shape, scatter_cols_needs,
};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer};

/// The byte that fills `dst` before a launch: a stray write shows against it.
const CANARY: u8 = 0xa5;
/// Floats before and after the destination extent that must stay untouched.
const PAD: usize = 64;

fn shape(rows: usize, n: usize, ld: usize, col_off: usize) -> ScatterColsShape {
    ScatterColsShape {
        rows,
        n,
        ld,
        col_off,
    }
}

/// Runs one scatter and returns the whole destination buffer (padding included).
fn run(shape: ScatterColsShape, src: &[f32]) -> Result<Vec<f32>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (_, dst_bytes) = scatter_cols_needs(&shape).expect("needs");
    let total = dst_bytes + 2 * PAD * 4;
    let src_buf = testkit::upload_bytes(&ctx, &to_bytes(src));
    let dst_buf: DeviceBuffer = testkit::upload_filled(&ctx, CANARY, total);
    ctx.scatter_cols(
        &stream,
        ScatterColsCall {
            shape,
            src: src_buf.range(0, src.len() * 4)?,
            dst: dst_buf.range(PAD * 4, dst_bytes)?,
        },
    )?;
    stream.synchronize()?;
    Ok(testkit::floats(&testkit::read_bytes(&ctx, &dst_buf, total)))
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn canary_value() -> f32 {
    f32::from_bits(u32::from_le_bytes([CANARY; 4]))
}

#[test]
fn the_block_lands_in_its_columns_and_nothing_else_moves() {
    if !testkit::device_present("scatter_cols columns") {
        return;
    }
    let mut rng = Rng::new(7);
    // The model's rows: the DeltaNet input row is 8192 + 4096 + 32 + 32 wide, the attention
    // input row 2 * 4096 + 2 * 512; the gate/up row of the experts is 2 * 512.
    let cases = [
        shape(1, 4, 4, 0),
        shape(7, 32, 12_352, 12_288),
        shape(144, 512, 1_024, 512),
        shape(144, 8_192, 12_352, 0),
        shape(33, 4_096, 12_352, 8_192),
        shape(2_048, 32, 12_352, 12_320),
        shape(4_096, 32, 12_352, 12_320),
        shape(32_768, 32, 64, 32),
    ];
    for case in cases {
        let src: Vec<f32> = (0..case.rows * case.n)
            .map(|_| rng.normal() as f32)
            .collect();
        let got = run(case, &src).expect("scatter");
        let canary = canary_value().to_bits();
        let (_, dst_bytes) = scatter_cols_needs(&case).expect("needs");
        let live = dst_bytes / 4;
        for i in 0..PAD {
            assert_eq!(got[i].to_bits(), canary, "{case:?}: before the extent");
            assert_eq!(got[PAD + live + i].to_bits(), canary, "{case:?}: after");
        }
        for r in 0..case.rows {
            for c in 0..case.ld.min(live - r * case.ld) {
                let value = got[PAD + r * case.ld + c].to_bits();
                if (case.col_off..case.col_off + case.n).contains(&c) {
                    let want = src[r * case.n + c - case.col_off].to_bits();
                    assert_eq!(value, want, "{case:?}: row {r} column {c}");
                } else {
                    assert_eq!(value, canary, "{case:?}: row {r} gap column {c}");
                }
            }
        }
    }
}

#[test]
fn the_pure_rules_pin_the_extents_and_the_bounds() {
    assert_eq!(
        scatter_cols_needs(&shape(3, 8, 20, 4)),
        Some((3 * 8 * 4, (2 * 20 + 4 + 8) * 4))
    );
    assert_eq!(scatter_cols_needs(&shape(0, 8, 20, 4)), None);
    assert_eq!(scatter_cols_needs(&shape(2, usize::MAX, 8, 0)), None);
    assert!(check_shape(&shape(MAX_ROWS, MAX_WIDTH, MAX_WIDTH, 0)).is_ok());
    for (bad, tag) in [
        (shape(0, 8, 8, 0), "rows"),
        (shape(MAX_ROWS + 1, 8, 8, 0), "rows"),
        (shape(1, 0, 8, 0), "n"),
        (shape(1, MAX_WIDTH + 4, MAX_WIDTH + 4, 0), "n"),
        (shape(1, 6, 8, 0), "n"),
        (shape(1, 8, 16, 2), "col_off"),
        (shape(1, 8, 18, 0), "ld"),
        (shape(1, 8, 12, 8), "ld"),
    ] {
        let refused = check_shape(&bad).expect_err("a refusal");
        let EngError::BadRequest(text) = refused else {
            panic!("{bad:?}: not a BadRequest");
        };
        assert!(
            text.starts_with(&format!("scatter_cols: {tag}:")),
            "{bad:?}: {text}"
        );
    }
}

#[test]
fn the_wrapper_refuses_short_unaligned_and_overlapping_ranges_before_any_launch() {
    if !testkit::device_present("scatter_cols refusals") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let case = shape(4, 8, 16, 8);
    let (src_bytes, dst_bytes) = scatter_cols_needs(&case).expect("needs");
    let src = testkit::upload_filled(&ctx, 0, 2 * src_bytes);
    let dst = testkit::upload_filled(&ctx, 0, 2 * dst_bytes);
    let call = |src_at: usize, src_len: usize, dst_at: usize, dst_len: usize| ScatterColsCall {
        shape: case,
        src: src.range(src_at, src_len).expect("src"),
        dst: dst.range(dst_at, dst_len).expect("dst"),
    };
    assert!(
        ctx.scatter_cols(&stream, call(0, src_bytes, 0, dst_bytes))
            .is_ok()
    );
    for refused in [
        call(0, src_bytes - 4, 0, dst_bytes),
        call(0, src_bytes, 0, dst_bytes - 4),
        call(4, src_bytes, 0, dst_bytes),
        call(0, src_bytes, 4, dst_bytes),
    ] {
        assert!(matches!(
            ctx.scatter_cols(&stream, refused),
            Err(EngError::BadRequest(_))
        ));
    }
    // The destination inside the source extent.
    let both = testkit::upload_filled(&ctx, 0, src_bytes + dst_bytes);
    let overlap = ScatterColsCall {
        shape: case,
        src: both.range(0, src_bytes).expect("src"),
        dst: both.range(src_bytes - 16, dst_bytes).expect("dst"),
    };
    assert!(matches!(
        ctx.scatter_cols(&stream, overlap),
        Err(EngError::BadRequest(_))
    ));
    stream.synchronize().expect("drain");
}
