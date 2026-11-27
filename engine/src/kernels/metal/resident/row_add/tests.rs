//! Device tests (Mac) of the row add: the Rust wrapper against the bridge against the shader
//! of `native/row_add.metal`, compared with an f64 host reference.
//!
//! Every buffer carries sentinel bytes around the used bytes and `out` is pre-filled with a
//! NaN pattern: a refused call must leave it untouched, and the inputs must never change. The
//! plain sum is one f32 add, so every result must have the bits of the host's sum under the
//! device's rule, overflow, cancellation and subnormals included: the device makes a subnormal
//! operand and a subnormal result a zero of its sign, where the CUDA adapter's add keeps them.
//! The fused form is checked against the `half`
//! crate's bf16 round trip and its flag. Each refusal is checked by its error text. The suite
//! runs with `--test-threads=1`, because two tests of it hold adapter leases.

use std::time::Instant;

use half::bf16;

use super::{ROW_ADD_MAX_N, RowAddCall};
use crate::EngError;
use crate::kernels::metal::MetalContext;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, f32_bytes, floats, sentinel};

/// The flag before a fused call: every bit but bit 0, so a set bit 0 is seen and no other bit
/// may change.
const FLAG_BEFORE: u32 = 0xa5a5_a5a4;

struct Case {
    /// The inputs, for the reference and the unchanged check.
    x: Vec<f32>,
    y: Vec<f32>,
    out: Padded,
    input_x: Padded,
    input_y: Padded,
    flag: Padded,
}

impl Case {
    fn new(ctx: &MetalContext, x: Vec<f32>, y: Vec<f32>, extra: (usize, usize, usize)) -> Self {
        Self {
            out: Padded::new(ctx, &sentinel(x.len() * 4), extra.0),
            input_x: Padded::new(ctx, &f32_bytes(&x), extra.1),
            input_y: Padded::new(ctx, &f32_bytes(&y), extra.2),
            flag: Padded::new(ctx, &FLAG_BEFORE.to_le_bytes(), 0),
            x,
            y,
        }
    }

    /// The call over the first `n` floats of the buffers.
    fn call(&self, n: usize, fused: bool) -> RowAddCall<'_> {
        RowAddCall {
            out: self.out.range(0, n * 4),
            x: self.input_x.range(0, n * 4),
            y: self.input_y.range(0, n * 4),
            n,
            flag: fused.then(|| self.flag.all()),
            fused,
        }
    }
}

/// `out` after a call that must run; the inputs are read back unchanged.
fn run(ctx: &MetalContext, c: &Case, n: usize, fused: bool) -> Vec<f32> {
    ctx.row_add(c.call(n, fused)).expect("a valid call");
    assert!(c.input_x.read("x") == f32_bytes(&c.x), "the call changed x");
    assert!(c.input_y.read("y") == f32_bytes(&c.y), "the call changed y");
    floats(&c.out.read("out"))
}

/// The plain sum under the device's rule: a subnormal operand and a subnormal result are each
/// a zero of its sign (M4 Max, 2026-11-27, under every math option the SDK names).
fn sum(x: f32, y: f32) -> f32 {
    flush(flush(x) + flush(y))
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

/// The fused result of one f32 sum: its bf16 round trip, or 0 with the flag's bit 0 set when
/// that is not finite.
fn published(sum: f32, flag: &mut u32) -> f32 {
    let round = bf16::from_f32(sum).to_f32();
    if round.is_finite() {
        round
    } else {
        *flag |= 1;
        0.0
    }
}

/// Two values are the same f32, or both NaN: the pattern of a NaN is the device's.
fn same_or_nan(got: f32, want: f32) -> bool {
    want.is_nan() && got.is_nan() || got.to_bits() == want.to_bits()
}

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

/// Pads a list of special pairs to the 32 floats a call takes at least, with zero and
/// negative zero, whose sum is a positive zero.
fn pad_to_32(mut x: Vec<f32>, mut y: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    x.extend(std::iter::repeat_n(0.0, 32 - x.len()));
    y.extend(std::iter::repeat_n(-0.0, 32 - y.len()));
    (x, y)
}

fn check_plain(rig: &Rig, x: &[f32], y: &[f32], n: usize, extra: (usize, usize, usize)) {
    let c = Case::new(&rig.ctx, x.to_vec(), y.to_vec(), extra);
    let got = run(&rig.ctx, &c, n, false);
    assert_eq!(got.len(), n, "the result's length");
    for i in 0..n {
        assert!(
            same_or_nan(got[i], sum(x[i], y[i])),
            "float {i}: value {}, f32 sum {}",
            got[i],
            sum(x[i], y[i])
        );
    }
}

fn check_fused(rig: &Rig, x: &[f32], y: &[f32]) {
    let n = x.len();
    let c = Case::new(&rig.ctx, x.to_vec(), y.to_vec(), (0, 0, 0));
    let mut flag = FLAG_BEFORE;
    let want: Vec<f32> = (0..n)
        .map(|i| published(sum(x[i], y[i]), &mut flag))
        .collect();
    let got = run(&rig.ctx, &c, n, true);
    for i in 0..n {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "float {i}");
    }
    let after = u32::from_le_bytes(c.flag.read("flag")[..4].try_into().expect("four bytes"));
    assert_eq!(after, flag, "the flag");
}

/// The call must be refused with `text` in its message and leave `out` untouched.
fn refuses(ctx: &MetalContext, call: RowAddCall<'_>, out: &Padded, text: &str) {
    match ctx.row_add(call) {
        Err(EngError::BadRequest(message)) => assert!(message.contains(text), "{message}"),
        Err(other) => panic!("another error: {other:?}"),
        Ok(()) => panic!("the call ran"),
    }
    assert!(
        out.read("out") == sentinel(2304 * 4),
        "a refused call wrote the output"
    );
}

#[test]
fn the_row_add_is_the_f32_sum_of_its_inputs() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xA11D_A11D_0001);
    // One hidden row, one row of four streams, a chunk of 32 rows, a long call.
    let start = Instant::now();
    for n in [32usize, 2304, 4 * 2304, 32 * 2304, 96 * 1024] {
        let (x, y) = (draw(&mut rng, n, 1.0), draw(&mut rng, n, 1.0));
        // On the 16-byte grid, then off it by 4, 8 and 12 bytes: the 4-byte grid the call needs.
        check_plain(&rig, &x, &y, n, (0, 0, 0));
        check_plain(&rig, &x, &y, n, (4, 8, 12));
    }
    println!("the plain sums took {:?}", start.elapsed());
}

#[test]
fn the_row_add_keeps_the_special_values_of_the_f32_sum() {
    let rig = Rig::new();
    // Sums that overflow, cancel to zero or are not a number; a subnormal operand and a
    // subnormal result, each a zero of its sign.
    let (tiny, big) = (1e-40_f32, 3.0e38_f32);
    let least = f32::MIN_POSITIVE;
    let near = least * (1.0 + 2.0f32.powi(-9));
    let x = vec![
        tiny,
        big,
        f32::MAX,
        1.0,
        f32::INFINITY,
        least,
        near,
        -near,
        -tiny,
    ];
    let y = vec![
        tiny,
        big,
        -f32::MAX,
        -1.0,
        f32::NEG_INFINITY,
        tiny,
        -least,
        least,
        -tiny,
    ];
    let (x, y) = pad_to_32(x, y);
    check_plain(&rig, &x, &y, 32, (0, 0, 0));
}

#[test]
fn the_fused_row_add_publishes_the_bf16_round_trip() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xB16B_16B1_6002);
    // Sums of every magnitude, up to the overflow of the f32 sum.
    for sd in [1.0, 1e37] {
        let (x, y) = (draw(&mut rng, 2304, sd), draw(&mut rng, 2304, sd));
        check_fused(&rig, &x, &y);
    }
    // Sums with a subnormal operand, a zero of its sign; that overflow or are not a number; or
    // that lie exactly between two bf16 values: the first rounds down to 1.0, an even mantissa,
    // the second up.
    let (tiny, big) = (1e-40_f32, 3.0e38_f32);
    let x = vec![
        tiny,
        big,
        f32::INFINITY,
        1.0 + 2.0f32.powi(-8),
        1.0 + 2.0f32.powi(-7) + 2.0f32.powi(-8),
    ];
    let y = vec![tiny, big, f32::NEG_INFINITY, 0.0, 0.0];
    let (x, y) = pad_to_32(x, y);
    check_fused(&rig, &x, &y);
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    let c = Case::new(&rig.ctx, vec![0.0; 2304], vec![0.0; 2304], (0, 0, 0));
    let (out, x, y) = (c.out.all(), c.input_x.all(), c.input_y.all());
    let plain = |n| RowAddCall {
        out,
        x,
        y,
        n,
        flag: None,
        fused: false,
    };
    // The count.
    for n in [0, 31, 33, ROW_ADD_MAX_N + 32] {
        refuses(
            &rig.ctx,
            plain(n),
            &c.out,
            "needs n as a multiple of 32 up to 4194304",
        );
    }
    // The ranges: short, misaligned, of another adapter, overlapping; then the fused form.
    let other = Rig::new();
    let alien = Padded::new(&other.ctx, &f32_bytes(&[0.0; 2304]), 0);
    let cases = [
        (
            RowAddCall {
                out: c.out.range(0, 2300 * 4),
                ..plain(2304)
            },
            "needs 9216 bytes of out",
        ),
        (
            RowAddCall {
                x: c.input_x.range(0, 2300 * 4),
                ..plain(2304)
            },
            "needs 9216 bytes of x",
        ),
        (
            RowAddCall {
                y: c.input_y.range(0, 2300 * 4),
                ..plain(2304)
            },
            "needs 9216 bytes of y",
        ),
        (
            RowAddCall {
                out: c.out.range(2, 2304 * 4),
                ..plain(2304)
            },
            "at a multiple of 4",
        ),
        (
            RowAddCall {
                x: alien.all(),
                ..plain(2304)
            },
            "of another adapter",
        ),
        (
            RowAddCall {
                out: x,
                ..plain(2304)
            },
            "overlaps the x or the y range",
        ),
        (
            RowAddCall {
                fused: true,
                ..plain(2304)
            },
            "fused row_add needs a flag",
        ),
        (
            RowAddCall {
                flag: Some(c.flag.range(0, 2)),
                fused: true,
                ..plain(2304)
            },
            "needs 4 bytes of flag",
        ),
        (
            RowAddCall {
                flag: Some(c.out.range(0, 4)),
                fused: true,
                ..plain(2304)
            },
            "the flag range overlaps",
        ),
    ];
    for (call, text) in cases {
        refuses(&rig.ctx, call, &c.out, text);
    }
}

/// The median of 21 single launches and the mean of 128 dispatches in a frame, in
/// nanoseconds.
fn timed(ctx: &MetalContext, call: RowAddCall<'_>) -> (u64, u64) {
    let one = || ctx.row_add_timed(call, 1).expect("one timed launch");
    for _ in 0..5 {
        one();
    }
    let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
    singles.sort_unstable();
    let framed = ctx.row_add_timed(call, 128).expect("128 timed dispatches") / 128;
    (singles[singles.len() / 2], framed)
}

#[test]
#[ignore = "prints the row add's time on this device"]
fn row_add_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5A5A_1234_0001);
    println!(
        "device: {} live bytes before the timing",
        rig.ctx.live_bytes()
    );
    // One hidden row, one row of four streams, a chunk of 32 rows.
    for n in [2304usize, 4 * 2304, 32 * 2304] {
        let c = Case::new(
            &rig.ctx,
            draw(&mut rng, n, 1.0),
            draw(&mut rng, n, 1.0),
            (0, 0, 0),
        );
        check_plain(&rig, &c.x, &c.y, n, (0, 0, 0));
        let (single, framed) = timed(&rig.ctx, c.call(n, false));
        println!(
            "F32 n {n}: one launch {} us, 128 in a frame {} us a dispatch",
            single as f64 / 1e3,
            framed as f64 / 1e3,
        );
    }
}
