//! Tests (3090) of `quantize_q8_1` (seam 1: the Rust wrapper against the C entry against the
//! vendored `quantize_q8_1` kernel). Device tests assume `--test-threads=1` (decision 0040).
//!
//! The vendored objects are built with `-use_fast_math` (decision 0049 item 5, KUBDAN-MSG-62
//! A4): the f32 scale `d = amax/127` is computed with the approximate division (up to 3 ulp from
//! the IEEE value) and codes can differ at rounding boundaries, and the half rounding happens on
//! that value. The device test therefore asserts INVARIANTS, not bytes:
//! - `|q*d - x| <= d/2 + eps` with `eps = 0.07*|d| + 1e-30`: `d/2` is the roundf bound; `eps`
//!   covers 3 f32 ulp of approximate division scaled by |q| <= 127 and the one half-ulp of
//!   storing `d` as a half (127 * 2^-11 ~ 0.062 |d|);
//! - every `q` in [-127, 127];
//! - the stored half `d` within 1 half-ulp of the IEEE `amax/127` rounded to a half;
//! - the stored half `s` within 1 half-ulp of the f32 butterfly reference (xor offsets
//!   16, 8, 4, 2, 1; the kernel's fixed warp_reduce_sum order);
//! - the padded tail of every row is exactly zero (the kernel writes it: xi = 0 gives d = 0,
//!   q = 0, s = 0), all-zero blocks are exactly zero, and nothing beyond `q8_1_bytes` is
//!   written (canaries);
//!
//! and it REPORTS the share of bit-equal blocks against the IEEE host reference (precedent:
//! the 0049 quantizer sub-check measured 67 percent bit equal, at most 3 ulp). A denormal-only
//! column may be flushed to zero by `-ftz` (a legal fast-math outcome): the test accepts the
//! flushed result or the invariants, and reports which it saw.

use super::{QuantizeCall, q8_1_bytes, quantize_needs};
use crate::EngError;
use crate::kernels::cuda::resident::matvec_q::tests::{Owner, assert_bad_request};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// Values per row of the device test; 2048 needs no padding, so the tail blocks probe the
/// padding rule with k % 512 != 0 as well (a second run below uses k 2080).
const K: usize = 2080;

/// The kernel's f32 warp reduction order for the stored s (xor offsets 16, 8, 4, 2, 1).
fn butterfly_sum(values: &[f32]) -> f32 {
    let mut lanes: Vec<f32> = values.to_vec();
    let mut offset = 16;
    while offset > 0 {
        let before = lanes.clone();
        for lane in 0..32 {
            lanes[lane] += before[lane ^ offset];
        }
        offset /= 2;
    }
    lanes[0]
}

/// The IEEE host reference of one 32-value block: (d half bits, s half bits, qs bytes).
fn reference_block(values: &[f32]) -> (u16, u16, [i8; 32]) {
    let amax = values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let d = amax / 127.0f32;
    let mut qs = [0i8; 32];
    for (i, value) in values.iter().enumerate() {
        qs[i] = if amax == 0.0 {
            0
        } else {
            (value / d).round() as i8
        };
    }
    (
        testkit::f64_to_f16_bits(d as f64),
        testkit::f64_to_f16_bits(butterfly_sum(values) as f64),
        qs,
    )
}

/// Parses one 36-byte `block_q8_1`: (d, s, qs).
fn parse_block(bytes: &[u8]) -> (f32, f32, [i8; 32]) {
    let d = testkit::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    let s = testkit::f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
    let mut qs = [0i8; 32];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = bytes[4 + i] as i8;
    }
    (d, s, qs)
}

/// Half-code distance of a stored half from its reference value.
fn half_codes_away(stored: f32, reference: f32) -> u16 {
    testkit::f64_to_f16_bits(stored as f64).abs_diff(testkit::f64_to_f16_bits(reference as f64))
}

#[test]
fn q8_1_bytes_is_pure_and_exact() {
    assert_eq!(q8_1_bytes(512, 1), Some(16 * 36));
    assert_eq!(q8_1_bytes(2048, 1), Some(64 * 36));
    // k 32 pads to 512: 16 blocks, not 1.
    assert_eq!(q8_1_bytes(32, 1), Some(16 * 36));
    assert_eq!(q8_1_bytes(513, 1), Some((1024 / 32) * 36));
    assert_eq!(q8_1_bytes(512, 8), Some(16 * 36 * 8));
    assert_eq!(q8_1_bytes(0, 1), Some(0));
    assert_eq!(q8_1_bytes(usize::MAX, 8), None);
}

/// The byte counts `plan_quantize` checks a call against (KUBDAN-MSG-64 item 6, mutant S4).
#[test]
fn quantize_needs_are_pure_and_exact() {
    // x: ncols rows of k f32 values; y: the padded block_q8_1 rows.
    assert_eq!(quantize_needs(2048, 2), Some((2048 * 2 * 4, 64 * 36 * 2)));
    // k 2304 pads to 2560: 80 blocks per row, not 72, on the y side only.
    assert_eq!(quantize_needs(2304, 3), Some((2304 * 3 * 4, 80 * 36 * 3)));
    assert_eq!(quantize_needs(usize::MAX / 2, 8), None);
    assert_eq!(quantize_needs(usize::MAX, 1), None);
}

#[test]
fn quantize_q8_1_matches_the_host_reference() {
    if !testkit::device_present("quantize_q8_1_matches_the_host_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream: DeviceStream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5eed_8137);
    let ncols = 8;
    let mut columns: Vec<Vec<f32>> = Vec::with_capacity(ncols);
    for _ in 0..5 {
        columns.push((0..K).map(|_| (rng.normal() as f32) * 0.05).collect());
    }
    columns.push(vec![0.0; K]);
    let outlier = 3.7e5f32;
    columns.push(
        (0..K)
            .map(|i| if i == 1000 { outlier } else { 0.0 })
            .collect(),
    );
    let denormal = 4.2e-40f32;
    columns.push((0..K).map(|_| denormal).collect());
    let flat: Vec<f32> = columns.iter().flat_map(|c| c.iter().copied()).collect();
    let x = testkit::upload_f32(&ctx, &flat);
    let y_bytes = q8_1_bytes(K, ncols).expect("size");
    let y = testkit::upload_filled(&ctx, 0xAA, y_bytes + 64);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: x.range(0, flat.len() * 4).expect("x range"),
            y: y.range(0, y_bytes).expect("y range"),
            k: K,
            ncols,
        },
    )
    .expect("queue the quantize");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(&ctx, &y, y_bytes + 64);
    for (i, byte) in bytes[y_bytes..].iter().enumerate() {
        assert_eq!(*byte, 0xAA, "canary {i} beyond the output was written");
    }
    let blocks_per_row = q8_1_bytes(K, 1).expect("size") / 36;
    let mut bit_equal = 0usize;
    let mut total = 0usize;
    let mut flushed_columns = 0usize;
    for (col, column) in columns.iter().enumerate() {
        let denormal_column =
            !column.iter().all(|v| *v == 0.0) && column.iter().all(|v| v.abs() < 1e-38);
        for b in 0..blocks_per_row {
            let live = K.min((b + 1) * 32).saturating_sub(b * 32);
            let block_values: Vec<f32> = (0..32)
                .map(|i| if i < live { column[b * 32 + i] } else { 0.0 })
                .collect();
            let (ref_d, ref_s, ref_qs) = reference_block(&block_values);
            let (d, s, qs) = parse_block(&bytes[(col * blocks_per_row + b) * 36..]);
            let (ref_df, ref_sf) = (testkit::f16_to_f32(ref_d), testkit::f16_to_f32(ref_s));
            total += 1;
            if testkit::f64_to_f16_bits(d as f64) == ref_d
                && testkit::f64_to_f16_bits(s as f64) == ref_s
                && qs == ref_qs
            {
                bit_equal += 1;
            }
            if block_values.iter().all(|v| *v == 0.0) {
                assert_eq!(
                    (d, s, qs),
                    (0.0, 0.0, [0i8; 32]),
                    "all-zero block {b} of column {col}"
                );
                continue;
            }
            if denormal_column {
                let flushed = d == 0.0 && s == 0.0 && qs == [0i8; 32];
                assert!(
                    flushed || (half_codes_away(d, ref_df) <= 1 && half_codes_away(s, ref_sf) <= 1),
                    "denormal block {b}: d={d} s={s} against {ref_df} {ref_sf}"
                );
                if flushed {
                    flushed_columns += 1;
                }
                continue;
            }
            assert!(
                half_codes_away(d, ref_df) <= 1,
                "d {d} vs {ref_df} in block {b}"
            );
            assert!(
                half_codes_away(s, ref_sf) <= 1,
                "s {s} vs {ref_sf} in block {b}"
            );
            let eps = 0.07 * d.abs() + 1e-30;
            for i in 0..32 {
                assert!((-127..=127).contains(&qs[i]), "q {} out of range", qs[i]);
                let input = block_values[i];
                assert!(
                    (qs[i] as f32 * d - input).abs() <= d / 2.0 + eps,
                    "col {col} block {b} i {i}: |{} - {input}| > {}/2 + {eps}",
                    qs[i] as f32 * d,
                    d
                );
            }
        }
    }
    println!(
        "quantize bit-equal blocks: {bit_equal}/{total} ({:.1} percent); denormal flushed blocks: {flushed_columns}",
        100.0 * bit_equal as f64 / total as f64
    );
}

/// Where `y` of a quantize refusal case lies.
#[derive(Clone, Copy)]
enum YIn {
    Own,
    X,
}

/// One refusal case of the quantize wrapper: the numbers of the call and the (offset, bytes) of its
/// ranges inside the rig's buffers.
#[derive(Clone, Copy)]
struct Spec {
    k: usize,
    ncols: usize,
    x: (usize, usize),
    y: (usize, usize),
    y_in: YIn,
    x_of: Owner,
    y_of: Owner,
    stream_of: Owner,
}

/// k 2080 pads to 2560: two columns read 2 * 2080 floats and write 2 * 80 blocks of 36 bytes.
const X_NEED: usize = 2 * 2080 * 4;
const Y_NEED: usize = 2 * 80 * 36;
const VALID: Spec = Spec {
    k: 2080,
    ncols: 2,
    x: (0, X_NEED),
    y: (0, Y_NEED),
    y_in: YIn::Own,
    x_of: Owner::Own,
    y_of: Owner::Own,
    stream_of: Owner::Own,
};

/// The rig buffers are larger than the call needs, so a case can offset, shorten or overlap a range.
const RIG_BYTES: usize = 32_768;

struct RefusalRig {
    stream: DeviceStream,
    x: DeviceBuffer,
    y: DeviceBuffer,
    ctx: CudaContext,
}

/// The second adapter: a stream and two buffers that belong to another context.
struct OtherContext {
    stream: DeviceStream,
    x: DeviceBuffer,
    y: DeviceBuffer,
    _ctx: CudaContext,
}

fn attempt(rig: &RefusalRig, other: &OtherContext, spec: Spec) -> Result<(), EngError> {
    let x_buffer = match spec.x_of {
        Owner::Own => &rig.x,
        Owner::Other => &other.x,
    };
    let y_buffer = match (spec.y_of, spec.y_in) {
        (Owner::Other, _) => &other.y,
        (Owner::Own, YIn::Own) => &rig.y,
        (Owner::Own, YIn::X) => &rig.x,
    };
    let stream = match spec.stream_of {
        Owner::Own => &rig.stream,
        Owner::Other => &other.stream,
    };
    rig.ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: x_buffer.range(spec.x.0, spec.x.1)?,
            y: y_buffer.range(spec.y.0, spec.y.1)?,
            k: spec.k,
            ncols: spec.ncols,
        },
    )
}

/// KUBDAN-MSG-64 item 6 for `plan_quantize`: every refusal is a `BadRequest` with the text of its
/// own check, nothing is queued, the backend stays usable, and the valid call of the same rig runs.
#[test]
fn quantize_rejects_unsound_calls_before_launching() {
    if !testkit::device_present("quantize_rejects_unsound_calls_before_launching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let rig = RefusalRig {
        stream: ctx.stream().expect("stream"),
        x: testkit::upload_filled(&ctx, 0, RIG_BYTES),
        y: testkit::upload_filled(&ctx, 0xAB, 16_384),
        ctx,
    };
    let other_ctx = CudaContext::new().expect("second context");
    let other = OtherContext {
        stream: other_ctx.stream().expect("stream of the other context"),
        x: testkit::upload_filled(&other_ctx, 0, RIG_BYTES),
        y: testkit::upload_filled(&other_ctx, 0xAB, 16_384),
        _ctx: other_ctx,
    };
    let cases: [(&str, &str, Spec); 15] = [
        ("k 0", "needs k as a multiple of 32", Spec { k: 0, ..VALID }),
        (
            "k 2090, off the 32 grid",
            "needs k as a multiple of 32",
            Spec { k: 2090, ..VALID },
        ),
        (
            "no columns",
            "takes 1 to 8 rows",
            Spec { ncols: 0, ..VALID },
        ),
        (
            "nine columns",
            "takes 1 to 8 rows",
            Spec { ncols: 9, ..VALID },
        ),
        (
            "k that overflows the byte count",
            "size overflows",
            Spec {
                k: usize::MAX - 31,
                ..VALID
            },
        ),
        (
            "x one float short",
            "x range holds",
            Spec {
                x: (0, X_NEED - 4),
                ..VALID
            },
        ),
        (
            "x misaligned by 2",
            "x range starts at offset",
            Spec {
                x: (2, X_NEED),
                ..VALID
            },
        ),
        (
            "y one block short",
            "y range holds",
            Spec {
                y: (0, Y_NEED - 36),
                ..VALID
            },
        ),
        (
            "y unpadded: 65 blocks per column",
            "y range holds",
            Spec {
                y: (0, 2 * 65 * 36),
                ..VALID
            },
        ),
        (
            "y misaligned by 2",
            "y range starts at offset",
            Spec {
                y: (2, Y_NEED),
                ..VALID
            },
        ),
        (
            "y over x",
            "y and x ranges overlap",
            Spec {
                y: (8192, Y_NEED),
                y_in: YIn::X,
                ..VALID
            },
        ),
        (
            "y over the end of x",
            "y and x ranges overlap",
            Spec {
                y: (X_NEED - 4, Y_NEED),
                y_in: YIn::X,
                ..VALID
            },
        ),
        (
            "x of another context",
            "buffer belongs",
            Spec {
                x_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "y of another context",
            "buffer belongs",
            Spec {
                y_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "a stream of another context",
            "stream belongs",
            Spec {
                stream_of: Owner::Other,
                ..VALID
            },
        ),
    ];
    for (case, needle, spec) in cases {
        assert_bad_request(attempt(&rig, &other, spec), case, needle);
    }
    // No refusal queued anything: y is untouched and the backend works; then the valid call runs.
    rig.stream.synchronize().expect("drain");
    let untouched = testkit::read_bytes(&rig.ctx, &rig.y, 16_384);
    assert!(untouched.iter().all(|&byte| byte == 0xAB), "y untouched");
    rig.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
    attempt(&rig, &other, VALID).expect("the valid call launches");
    rig.stream.synchronize().expect("drain");
    let written = testkit::read_bytes(&rig.ctx, &rig.y, Y_NEED);
    assert!(
        written.iter().all(|&byte| byte != 0xAB),
        "the valid call wrote y"
    );
}
