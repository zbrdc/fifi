use std::mem::size_of;

use super::{PrepMath, PrepRow, PrepShape, PrepWeights, Scored, prep_row};
use crate::kernels::cuda::resident::relative_attention::LogScale;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present};

const SHAPE: PrepShape = PrepShape {
    queries: 2,
    kv: 1,
    dims: 16,
};

const FLOOR: LogScale = LogScale {
    floor: 8192,
    alpha: 0.1,
};

/// Deterministic values in [-1, 1) from a 64-bit linear congruential sequence.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

/// The prepared query, relative coefficients, key and value of one row at `pos`, from the same
/// projections, weights and histories every call.
fn prepare(
    ctx: &CudaContext,
    stream: &DeviceStream,
    log: Option<LogScale>,
    pos: i32,
) -> [Vec<f32>; 4] {
    let words = SHAPE.direct_words();
    let mut rng = Lcg(7);
    let buffers: Vec<DeviceBuffer> = words
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let values: Vec<f32> = match i {
                // The clip scales are positive; the status words start clear.
                8 | 9 => (0..n).map(|_| 1.0 + rng.next().abs()).collect(),
                14 => vec![0.0; n],
                _ => (0..n).map(|_| 4.0 * rng.next()).collect(),
            };
            let buffer = ctx.allocate(n * size_of::<f32>()).unwrap();
            buffer.write_at(stream, 0, &bytes(&values)).unwrap();
            buffer
        })
        .collect();
    let range = |i: usize| buffers[i].range(0, words[i] * size_of::<f32>()).unwrap();
    let position = ctx.allocate(size_of::<i32>()).unwrap();
    position.write_at(stream, 0, &pos.to_ne_bytes()).unwrap();
    prep_row(
        ctx,
        stream,
        PrepRow {
            shape: SHAPE,
            math: PrepMath {
                qk_limit: 2.0,
                rel_limit: 1.5,
                eps: 1e-6,
                log,
            },
            rows: Scored {
                q: range(0),
                relative: range(1),
                sinks: range(6),
                gates: range(7),
                k: range(4),
                v: range(5),
            },
            raw_k: range(2),
            raw_v: range(3),
            weights: PrepWeights {
                query_scale: range(8),
                relative_scale: range(9),
                key_taps: range(10),
                value_taps: range(11),
            },
            key_history: range(12),
            value_history: range(13),
            status: range(14),
            pos: position.range(0, size_of::<i32>()).unwrap(),
        },
    )
    .unwrap();
    [0, 1, 4, 5].map(|i| {
        let mut out = vec![0u8; words[i] * size_of::<f32>()];
        buffers[i].read_at(stream, 0, &mut out).unwrap();
        floats(&out)
    })
}

fn bits(rows: &[Vec<f32>; 4]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// Through the floor the scale is exactly 1, so a prep that reads it keeps the bits of one that
/// does not; past it the clipped query and relative coefficients are those rows times
/// `1 + alpha * ln((pos + 1) / floor)`, and the key and value are untouched.
#[test]
fn the_log_query_scale_is_one_through_the_floor_and_logarithmic_past_it() {
    if !device_present("relative prep log query scale") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for pos in [0, 100, 8191] {
        assert_eq!(
            bits(&prepare(&ctx, &stream, Some(FLOOR), pos)),
            bits(&prepare(&ctx, &stream, None, pos)),
            "position {pos}"
        );
    }
    for pos in [8192, 20_000, 262_143] {
        let scaled = prepare(&ctx, &stream, Some(FLOOR), pos);
        let plain = prepare(&ctx, &stream, None, pos);
        let tau = 1.0 + FLOOR.alpha * ((pos + 1) as f32 / FLOOR.floor as f32).ln();
        assert!(tau > 1.0, "position {pos}: tau {tau}");
        for (row, (scaled, plain)) in scaled[..2].iter().zip(&plain[..2]).enumerate() {
            for (a, b) in scaled.iter().zip(plain) {
                let want = b * tau;
                assert!(
                    (a - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "position {pos}, plane {row}: {a} against {want}"
                );
            }
        }
        assert_eq!(
            bits(&scaled)[2..],
            bits(&plain)[2..],
            "position {pos}: key and value"
        );
    }
}
