//! Device tests (Mac) of the RMS norm: the Rust wrapper against the bridge against the shaders
//! of `native/rms_norm.metal`, compared with an f64 host reference.
//!
//! Every buffer carries sentinel bytes around the used bytes and `out` is pre-filled with a NaN
//! pattern: a refused call must leave it untouched, `x` must hold its sums with `delta` and
//! nothing else, and `delta` and `weight` must never change. The shaders take the correctly
//! rounded division and reciprocal square root, so a result differs from the f64 reference by
//! the order of the f32 sum of squares only: a thread adds at most `n / 256` rounded squares
//! on a partial sum that never decreases, `2 · (n / 256) · 2^-24` at most, and thirty-two more
//! roundings cover the reduction, the mean, the norm and the two products. A square is an f32 product on
//! the device, an overflow to infinity and a subnormal flush included, so the reference squares
//! in f32 too. Each refusal is checked by its error text. The suite runs with
//! `--test-threads=1`, because two tests of it hold adapter leases.

use super::{RMS_NORM_MAX_N, RMS_NORM_MAX_ROWS, RmsNormCall};
use crate::EngError;
use crate::kernels::metal::MetalContext;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, f32_bytes, floats, sentinel};

/// The f32 unit roundoff, 2^-24.
const U: f64 = 1.0 / 16_777_216.0;
/// A floor for a result the device flushes to zero.
const ABS: f64 = 1e-37;
/// The epsilon of the models the first serve reads (GGUF `layer_norm_rms_epsilon`).
const EPS: f32 = 1e-6;

struct Case {
    rows: usize,
    n: usize,
    /// The rows before the call.
    x: Vec<f32>,
    /// `None` for a norm without the add.
    delta: Option<Vec<f32>>,
    weight: Vec<f32>,
}

impl Case {
    fn new(rng: &mut Rng, rows: usize, n: usize, with_delta: bool) -> Self {
        let draw = |rng: &mut Rng, len: usize| (0..len).map(|_| rng.normal() as f32).collect();
        Self {
            rows,
            n,
            x: draw(rng, rows * n),
            delta: with_delta.then(|| draw(rng, rows * n)),
            weight: draw(rng, n),
        }
    }

    fn bytes(&self, rows: usize) -> usize {
        rows * self.n * 4
    }

    /// The f32 value the device squares: `x + delta` under the device's rule, or `x`.
    fn value(&self, r: usize, i: usize) -> f32 {
        let at = r * self.n + i;
        match &self.delta {
            Some(delta) => sum(self.x[at], delta[at]),
            None => self.x[at],
        }
    }

    /// The f64 reference of row `r`: the f32 squares, summed in f64.
    fn reference(&self, r: usize) -> Vec<f64> {
        let squares: f64 = (0..self.n)
            .map(|i| f64::from(square(self.value(r, i))))
            .sum();
        let rstd = 1.0 / (squares / self.n as f64 + f64::from(EPS)).sqrt();
        (0..self.n)
            .map(|i| f64::from(self.value(r, i)) * rstd * f64::from(self.weight[i]))
            .collect()
    }
}

/// The plain sum under the device's rule: a subnormal operand and a subnormal result are each
/// a zero of their sign (row_add.metal has the measurement).
fn sum(x: f32, y: f32) -> f32 {
    flush(flush(x) + flush(y))
}

/// A square as the device gives it: an f32 product, a subnormal one a zero of its sign.
fn square(v: f32) -> f32 {
    flush(v * v)
}

/// A value as the device's arithmetic takes it and gives it: below the least normal, a zero of
/// its sign.
fn flush(v: f32) -> f32 {
    if v.is_subnormal() {
        0.0f32.copysign(v)
    } else {
        v
    }
}

/// `out` and `x` after a call that must run; `extra` shifts the four ranges past the 16-byte
/// grid they need. `delta` and `weight` are read back unchanged.
fn run(ctx: &MetalContext, c: &Case, extra: [usize; 4]) -> (Vec<f32>, Vec<f32>) {
    let bytes = c.bytes(c.rows);
    let x = Padded::new(ctx, &f32_bytes(&c.x), extra[0]);
    let delta = c
        .delta
        .as_ref()
        .map(|d| Padded::new(ctx, &f32_bytes(d), extra[1]));
    let weight = Padded::new(ctx, &f32_bytes(&c.weight), extra[2]);
    let out = Padded::new(ctx, &sentinel(bytes), extra[3]);
    ctx.rms_norm(RmsNormCall {
        rows: c.rows,
        n: c.n,
        eps: EPS,
        x: x.range(0, bytes),
        delta: delta.as_ref().map(|d| d.range(0, bytes)),
        weight: weight.all(),
        out: out.all(),
    })
    .expect("a valid call");
    let x_after = floats(&x.read("x"));
    if let (Some(d), Some(want)) = (&delta, &c.delta) {
        assert!(d.read("delta") == f32_bytes(want), "the call changed delta");
    }
    assert!(
        weight.read("weight") == f32_bytes(&c.weight),
        "the call changed the weight"
    );
    (floats(&out.read("out")), x_after)
}

/// Every result lies in its bound, or is a NaN where the reference is one, and `x` holds the
/// values the device squares: its sums with `delta`, or its own values.
fn check(c: &Case, got: &[f32], x_after: &[f32]) {
    let bound = 2.0 * (c.n as f64 / 256.0) * U + 32.0 * U;
    for r in 0..c.rows {
        let want = c.reference(r);
        for (i, want_i) in want.iter().enumerate() {
            let at = r * c.n + i;
            let error = (f64::from(got[at]) - want_i).abs();
            let limit = bound * want_i.abs() + ABS;
            let close = if want_i.is_nan() {
                got[at].is_nan()
            } else if want_i.is_infinite() {
                got[at].to_bits() == (*want_i as f32).to_bits()
            } else {
                error <= limit
            };
            assert!(
                close,
                "row {r}, float {i}: value {}, f64 reference {want_i}, error {error} over the bound {limit}",
                got[at]
            );
            let stored = c.value(r, i);
            assert!(
                stored.is_nan() && x_after[at].is_nan()
                    || x_after[at].to_bits() == stored.to_bits(),
                "row {r}, float {i}: x holds {}, not {stored}",
                x_after[at]
            );
        }
    }
}

/// The call must be refused with `text` in its message and leave `out` and `x` untouched.
fn refuses(
    ctx: &MetalContext,
    call: RmsNormCall<'_>,
    x: &Padded,
    x_bytes: &[u8],
    out: &Padded,
    out_bytes: usize,
    text: &str,
) {
    match ctx.rms_norm(call) {
        Err(EngError::BadRequest(message)) => assert!(message.contains(text), "{message}"),
        Err(other) => panic!("another error: {other:?}"),
        Ok(()) => panic!("the call ran"),
    }
    assert!(
        out.read("out") == sentinel(out_bytes),
        "a refused call wrote the output rows"
    );
    assert!(x.read("x") == x_bytes, "a refused call changed x");
}

#[test]
fn the_rms_norm_matches_the_f64_reference() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x524D_534E_0001);
    // One hidden row, four rows of a prefill chunk, the narrowest row, a row of 1024.
    for (rows, n) in [(1usize, 2304usize), (4, 2304), (1, 32), (2, 1024)] {
        for with_delta in [false, true] {
            let case = Case::new(&mut rng, rows, n, with_delta);
            let (got, x_after) = run(&rig.ctx, &case, [0; 4]);
            check(&case, &got, &x_after);
            // 16, 32, 48 and 64 bytes past the 16-byte grid.
            let (got, x_after) = run(&rig.ctx, &case, [16, 32, 48, 64]);
            check(&case, &got, &x_after);
        }
    }
}

#[test]
fn the_rms_norm_keeps_the_infinities_the_nans_and_the_tiny_values() {
    let rig = Rig::new();
    let mut weight = vec![0.0f32; 32];
    for (i, w) in weight.iter_mut().enumerate() {
        *w = if i % 2 == 0 { 1.0 } else { -0.5 };
    }
    for with_delta in [false, true] {
        // A row whose squares overflow: rstd is 0, so a finite value gives a zero and an
        // infinity a NaN. A row of zeros, subnormals and tiny values, where the epsilon rules.
        let x = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e20,
            -1e20,
            1e-30,
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ]
        .into_iter()
        .chain(std::iter::repeat_n(0.75, 32 - 13))
        .chain(
            [
                0.0f32, -0.0, 1e-40, -1e-40, 1e-20, -1e-20, 0.5, -0.5, 1.0, -1.0, 3.0, 7.0,
            ]
            .into_iter()
            .chain(std::iter::repeat_n(0.25, 32 - 12)),
        )
        .collect::<Vec<f32>>();
        let mut delta = vec![0.0f32; 64];
        delta[3] = 0.5;
        delta[10] = -2.0;
        delta[34] = 1e-40;
        delta[38] = -0.25;
        let case = Case {
            rows: 2,
            n: 32,
            x,
            delta: with_delta.then_some(delta),
            weight: weight.clone(),
        };
        let (got, x_after) = run(&rig.ctx, &case, [0; 4]);
        check(&case, &got, &x_after);
    }
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x524D_534E_0002);
    let c = Case::new(&mut rng, 2, 2304, true);
    let bytes = c.bytes(2);
    let x_bytes = f32_bytes(&c.x);
    let x = Padded::new(&rig.ctx, &x_bytes, 0);
    let delta = Padded::new(&rig.ctx, &f32_bytes(c.delta.as_ref().expect("the add")), 0);
    let weight = Padded::new(&rig.ctx, &f32_bytes(&c.weight), 0);
    let out = Padded::new(&rig.ctx, &sentinel(bytes), 0);
    let other = Rig::new();
    let alien = Padded::new(&other.ctx, &f32_bytes(&[0.0; 2304]), 0);
    // The count, then eps, then the four ranges as far as the call gives them, then the
    // overlaps of the ranges it writes.
    let cases = [
        (
            RmsNormCall {
                n: 0,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "n as a multiple of 32",
        ),
        (
            RmsNormCall {
                n: 33,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "n as a multiple of 32",
        ),
        (
            RmsNormCall {
                n: RMS_NORM_MAX_N + 32,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "up to 32768",
        ),
        (
            RmsNormCall {
                rows: 0,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "rows from 1 to",
        ),
        (
            RmsNormCall {
                rows: RMS_NORM_MAX_ROWS + 1,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "rows from 1 to",
        ),
        (
            RmsNormCall {
                eps: 0.0,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "eps finite and above 0",
        ),
        (
            RmsNormCall {
                eps: f32::NAN,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "eps finite and above 0",
        ),
        (
            RmsNormCall {
                eps: f32::INFINITY,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "eps finite and above 0",
        ),
        (
            RmsNormCall {
                x: x.range(0, 2304 * 4),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "needs 18432 bytes of x",
        ),
        (
            RmsNormCall {
                delta: Some(delta.range(0, 2304 * 4)),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "needs 18432 bytes of delta",
        ),
        (
            RmsNormCall {
                weight: weight.range(0, 2304 * 4 - 4),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "needs 9216 bytes of weight",
        ),
        (
            RmsNormCall {
                out: out.range(0, bytes - 4),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "needs 18432 bytes of out",
        ),
        (
            RmsNormCall {
                x: x.range(4, bytes),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "at a multiple of 16",
        ),
        (
            RmsNormCall {
                weight: alien.all(),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "of another adapter",
        ),
        (
            RmsNormCall {
                out: x.range(0, bytes),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "out range overlaps",
        ),
        (
            RmsNormCall {
                out: delta.range(0, bytes),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "out range overlaps",
        ),
        (
            RmsNormCall {
                rows: 1,
                out: weight.range(0, 2304 * 4),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "out range overlaps",
        ),
        (
            RmsNormCall {
                delta: Some(x.range(0, bytes)),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "takes the sum and overlaps",
        ),
        (
            RmsNormCall {
                x: delta.range(0, bytes),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "takes the sum and overlaps",
        ),
        (
            RmsNormCall {
                rows: 1,
                x: weight.range(0, 2304 * 4),
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "takes the sum and overlaps",
        ),
        // A negative epsilon, the one count-like rule not yet hit.
        (
            RmsNormCall {
                eps: -1e-6,
                ..valid(&x, &delta, &weight, &out, bytes)
            },
            "eps finite and above 0",
        ),
    ];
    for (call, text) in cases {
        refuses(&rig.ctx, call, &x, &x_bytes, &out, bytes, text);
    }
}

/// A valid call of two rows over the four buffers.
fn valid<'b>(
    x: &'b Padded,
    delta: &'b Padded,
    weight: &'b Padded,
    out: &'b Padded,
    bytes: usize,
) -> RmsNormCall<'b> {
    RmsNormCall {
        rows: 2,
        n: 2304,
        eps: EPS,
        x: x.range(0, bytes),
        delta: Some(delta.range(0, bytes)),
        weight: weight.all(),
        out: out.all(),
    }
}

/// The median of 21 single launches and the mean of 128 dispatches in a frame, in
/// nanoseconds.
fn timed(ctx: &MetalContext, c: &Case) -> (u64, u64) {
    let bytes = c.bytes(c.rows);
    let x = Padded::new(ctx, &f32_bytes(&c.x), 0);
    let delta = c.delta.as_ref().map(|d| Padded::new(ctx, &f32_bytes(d), 0));
    let weight = Padded::new(ctx, &f32_bytes(&c.weight), 0);
    let out = Padded::new(ctx, &sentinel(bytes), 0);
    let call = RmsNormCall {
        rows: c.rows,
        n: c.n,
        eps: EPS,
        x: x.range(0, bytes),
        delta: delta.as_ref().map(|d| d.range(0, bytes)),
        weight: weight.all(),
        out: out.all(),
    };
    let one = || ctx.rms_norm_timed(call, 1).expect("one timed launch");
    for _ in 0..5 {
        one();
    }
    let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
    singles.sort_unstable();
    let framed = ctx.rms_norm_timed(call, 128).expect("128 timed dispatches") / 128;
    (singles[singles.len() / 2], framed)
}

#[test]
#[ignore = "prints the RMS norm's time on this device"]
fn rms_norm_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x524D_534E_0003);
    println!(
        "device: {} live bytes before the timing",
        rig.ctx.live_bytes()
    );
    // One hidden row, four rows of it, the widest row; each without and with the add.
    for (rows, n) in [(1usize, 2304usize), (4, 2304), (1, 32_768)] {
        for with_delta in [false, true] {
            let c = Case::new(&mut rng, rows, n, with_delta);
            let (single, framed) = timed(&rig.ctx, &c);
            println!(
                "NORM rows {rows} n {n}, add {with_delta}: one launch {} us, 128 in a frame {} us a dispatch",
                single as f64 / 1e3,
                framed as f64 / 1e3,
            );
        }
    }
}
