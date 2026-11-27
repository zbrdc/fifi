//! Device tests (Mac) of the soft cap: the Rust wrapper against the bridge against the shader
//! of `native/softcap.metal`, against the fixture the CUDA adapter's cap was frozen with and an
//! f64 host reference.
//!
//! Every buffer carries sentinel bytes around the used bytes, and the floats after the first
//! `n` must come back with their bits: an in-place call writes the first `n` floats and nothing
//! else. A zero keeps the sign of its bits, and a NaN or an infinity stays as it was. Each
//! refusal is checked by its error text and leaves the row with its bits. The suite runs with
//! `--test-threads=1`, because tests of it hold adapter leases.

use super::{SOFTCAP_MAX_N, SoftcapCall};
use crate::EngError;
use crate::kernels::metal::MetalContext;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, f32_bytes, floats};

/// The bound of CUDA's frozen fixture: an absolute part and a relative part of the expectation.
const ABS_BOUND: f64 = 2e-6;
const REL_BOUND: f64 = 2e-5;
/// The bound of CUDA's in-place test.
const SPOT_BOUND: f64 = 4e-6;
/// Wolfram|Alpha's values of `15 * tanh(x / 15)` at x 17 and -37, the inputs of CUDA's
/// in-place test.
const SPOT_17: f64 = 12.182_355_119_421_865;
const SPOT_NEG_37: f64 = -14.785_471_480_820_894;

const CASE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/softcap-frozen.bin"
));

struct Case {
    /// The row before the call, for the reference and the tail check.
    x: Vec<f32>,
    row: Padded,
}

impl Case {
    fn new(ctx: &MetalContext, x: Vec<f32>, extra: usize) -> Self {
        Self {
            row: Padded::new(ctx, &f32_bytes(&x), extra),
            x,
        }
    }

    /// The call over the first `n` floats of the row.
    fn call(&self, n: usize, cap: f32) -> SoftcapCall<'_> {
        SoftcapCall {
            x: self.row.range(0, n * 4),
            n,
            cap,
        }
    }
}

/// The fixture: a u32 case count, then per case an f32 cap, a u32 count, the f32 inputs and the
/// f64 expectations.
fn cases() -> Vec<(f32, Vec<f32>, Vec<f64>)> {
    let u32_at = |bytes: &[u8]| u32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
    let mut rest = CASE;
    let count = u32_at(rest) as usize;
    rest = &rest[4..];
    let mut cases = Vec::with_capacity(count);
    for _ in 0..count {
        let cap = f32::from_bits(u32_at(rest));
        rest = &rest[4..];
        let n = u32_at(rest) as usize;
        rest = &rest[4..];
        let input: Vec<f32> = rest[..n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        rest = &rest[n * 4..];
        let expected: Vec<f64> = rest[..n * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        rest = &rest[n * 8..];
        cases.push((cap, input, expected));
    }
    assert!(rest.is_empty(), "trailing fixture bytes");
    cases
}

/// The cap of one float in f64: the reference of the device's f32 computation.
fn reference(x: f32, cap: f32) -> f64 {
    f64::from(cap) * (f64::from(x) / f64::from(cap)).tanh()
}

/// Whether two rows hold the same bits.
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.iter()
        .map(|v| v.to_bits())
        .eq(b.iter().map(|v| v.to_bits()))
}

/// A result against its expectation, under the fixture's bound. A NaN or an infinity the row
/// held stays what it was, whatever the fixture's reference says of it.
fn within(got: f32, input: f32, want: f64) -> bool {
    if input.is_nan() {
        return got.is_nan();
    }
    if input.is_infinite() {
        return got.to_bits() == input.to_bits();
    }
    (f64::from(got) - want).abs() <= ABS_BOUND + REL_BOUND * want.abs()
}

/// The row after a call that must run: the first `n` floats capped, the rest with their bits.
fn run(ctx: &MetalContext, c: &Case, n: usize, cap: f32) -> Vec<f32> {
    ctx.softcap_in_place(c.call(n, cap)).expect("a valid call");
    let got = floats(&c.row.read("row"));
    assert!(same_bits(&got[n..], &c.x[n..]), "the call changed the tail");
    got
}

/// The call must be refused with `text` in its message and leave the row with its bits.
fn refuses(ctx: &MetalContext, call: SoftcapCall<'_>, c: &Case, text: &str) {
    match ctx.softcap_in_place(call) {
        Err(EngError::BadRequest(message)) => assert!(message.contains(text), "{message}"),
        Err(other) => panic!("another error: {other:?}"),
        Ok(()) => panic!("the call ran"),
    }
    assert!(
        same_bits(&floats(&c.row.read("row")), &c.x),
        "a refused call changed the row"
    );
}

/// The whole row capped against the reference, `extra` bytes past the sentinel: on the 16-byte
/// grid, then off it by 4, 8 and 12 bytes, the 4-byte grid the call needs.
fn check(rig: &Rig, x: &[f32], cap: f32, extra: usize) {
    let c = Case::new(&rig.ctx, x.to_vec(), extra);
    let got = run(&rig.ctx, &c, x.len(), cap);
    for (i, (&value, &input)) in got.iter().zip(x).enumerate() {
        let want = reference(input, cap);
        assert!(
            within(value, input, want),
            "float {i}: value {value}, reference {want}"
        );
    }
}

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

#[test]
fn the_softcap_matches_the_frozen_reference_of_cuda() {
    let rig = Rig::new();
    let mut checked = 0;
    for (cap, input, expected) in cases() {
        if input.is_empty() {
            continue;
        }
        let c = Case::new(&rig.ctx, input.clone(), 0);
        let got = run(&rig.ctx, &c, input.len(), cap);
        for (i, (&value, &want)) in got.iter().zip(&expected).enumerate() {
            assert!(
                within(value, input[i], want),
                "cap {cap}, float {i}: {value} versus {want}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "the fixture held no value");
}

#[test]
fn the_softcap_is_the_f64_reference_on_random_rows() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x50F7_CAE0_0001);
    // A cap about the row's spread, one far above it and one far below: small values, where the
    // absolute part of the bound rules, and saturation, where the relative part rules.
    for cap in [0.5f32, 15.0, 30.0, 1000.0] {
        for (n, scale) in [(1, 8.0), (7, 0.125), (2304, 4.0), (4 * 2304, 64.0)] {
            let x = draw(&mut rng, n, f64::from(cap) * scale);
            for extra in [0, 4, 8, 12] {
                check(&rig, &x, cap, extra);
            }
        }
    }
}

#[test]
fn the_softcap_in_place_caps_the_prefix_and_keeps_the_special_values() {
    let rig = Rig::new();
    let cap = 15.0f32;
    // Two zeros of both signs, the inputs of CUDA's in-place test, three values that are not
    // finite, one far beyond the cap, and a tail the call must not touch.
    let mut x = vec![
        0.0,
        -0.0,
        17.0,
        -37.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1234.0,
    ];
    x.extend(draw(&mut Rng::new(0x7A11_0002), 56, f64::from(cap)));
    let c = Case::new(&rig.ctx, x, 0);
    let got = run(&rig.ctx, &c, 8, cap);
    assert_eq!(got[0].to_bits(), 0.0f32.to_bits(), "a zero");
    assert_eq!(got[1].to_bits(), (-0.0f32).to_bits(), "a negative zero");
    for (value, spot) in [(got[2], SPOT_17), (got[3], SPOT_NEG_37)] {
        assert!(
            (f64::from(value) - spot).abs() < SPOT_BOUND,
            "{value} against Wolfram|Alpha's {spot}"
        );
    }
    for (i, input) in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN]
        .into_iter()
        .enumerate()
    {
        let value = got[i + 4];
        if input.is_nan() {
            assert!(value.is_nan(), "a NaN became {value}");
        } else {
            assert_eq!(value.to_bits(), input.to_bits(), "an infinity changed");
        }
    }
    let want = f64::from(cap) * (1234.0f64 / f64::from(cap)).tanh();
    assert!(
        within(got[7], 1234.0, want),
        "the saturated value {}",
        got[7]
    );
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    // Ones: a call that ran would cap them to about 15.
    let c = Case::new(&rig.ctx, vec![1.0; 2304], 0);
    // The cap: zero, negative, not a number, infinite.
    for cap in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        refuses(
            &rig.ctx,
            SoftcapCall {
                cap,
                ..c.call(8, cap)
            },
            &c,
            "a finite cap above zero",
        );
    }
    // The count: empty, over the maximum; and the maximum's bytes asked of a short range. Each
    // range is a real one of 8 floats: the kit refuses a range that does not fit its buffer, and
    // the count is checked before the range.
    let short = |n: usize| SoftcapCall {
        n,
        ..c.call(8, 15.0)
    };
    for n in [0, SOFTCAP_MAX_N + 1] {
        refuses(&rig.ctx, short(n), &c, "n in 1..=4194304");
    }
    refuses(
        &rig.ctx,
        short(SOFTCAP_MAX_N),
        &c,
        "needs 16777216 bytes of logits",
    );
    // The range: short, misaligned, of another adapter.
    let other = Rig::new();
    let alien = Padded::new(&other.ctx, &f32_bytes(&[0.0; 2304]), 0);
    let cases = [
        (
            SoftcapCall {
                x: c.row.range(0, 7 * 4),
                ..c.call(8, 15.0)
            },
            "needs 32 bytes of logits",
        ),
        (
            SoftcapCall {
                x: c.row.range(2, 8 * 4),
                ..c.call(8, 15.0)
            },
            "at a multiple of 4",
        ),
        (
            SoftcapCall {
                x: alien.range(0, 8 * 4),
                ..c.call(8, 15.0)
            },
            "of another adapter",
        ),
    ];
    for (call, text) in cases {
        refuses(&rig.ctx, call, &c, text);
    }
}

/// The median of 21 single launches and the mean of 128 dispatches in a frame, in nanoseconds.
fn timed(ctx: &MetalContext, call: SoftcapCall<'_>) -> (u64, u64) {
    let one = || {
        ctx.softcap_in_place_timed(call, 1)
            .expect("one timed launch")
    };
    for _ in 0..5 {
        one();
    }
    let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
    singles.sort_unstable();
    let framed = ctx
        .softcap_in_place_timed(call, 128)
        .expect("128 timed dispatches")
        / 128;
    (singles[singles.len() / 2], framed)
}

#[test]
#[ignore = "prints the soft cap's time on this device"]
fn softcap_in_place_is_timed() {
    let rig = Rig::new();
    let cap = 30.0f32;
    println!(
        "device: {} live bytes before the timing",
        rig.ctx.live_bytes()
    );
    // A hidden row, the first Metal model's vocabulary, and a power of two above it.
    for n in [2304usize, 200_058, 262_144] {
        let x = draw(&mut Rng::new(0x5A5A_1234_0008), n, 1.0);
        let c = Case::new(&rig.ctx, x, 0);
        let (single, framed) = timed(&rig.ctx, c.call(n, cap));
        println!(
            "SOFTCAP n {n}: one launch {} us, 128 in a frame {} us a dispatch",
            single as f64 / 1e3,
            framed as f64 / 1e3,
        );
    }
}
