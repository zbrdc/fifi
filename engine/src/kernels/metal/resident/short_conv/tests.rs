//! Device tests (Mac) of the short convolution: the Rust wrapper against the bridge against the
//! shader of `native/short_conv.metal`, against an f32 host twin in the device's order.
//!
//! Every buffer carries sentinel bytes around the used bytes. The convolution is a fixed chain
//! of f32 multiplies and adds with no contraction, so every result must have the bits of the
//! twin under the device's rule, subnormals included: the device makes a subnormal operand and
//! a subnormal result a zero of its sign, where the CUDA adapter keeps them. A sequence checks
//! the history each launch leaves: the raw inputs, newest first, even when the output overwrote
//! the input. A value that is not finite puts a quiet NaN of no payload into the health word and
//! nothing else; a finite launch leaves the word alone. Each refusal is checked by its error
//! text and changes nothing. The suite runs with `--test-threads=1`, because tests of it hold
//! adapter leases.

use super::{SHORT_CONV_MAX_TAPS, SHORT_CONV_MAX_WIDTH, ShortConvCall};
use crate::EngError;
use crate::kernels::metal::MetalContext;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, f32_bytes, floats};

/// The quiet NaN of the health word, with no payload.
const NOT_FINITE: u32 = 0x7fc0_0000;
/// The health word before a launch: a finite value that a finite launch must leave alone.
const HEALTH_BEFORE: u32 = 0x3f80_0000;

/// A row of the kit's normal distribution.
fn normal_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.normal() as f32).collect()
}

/// A value as the device's arithmetic takes it and gives it: below the least normal, a zero of
/// its sign.
fn flush(value: f32) -> f32 {
    if value.is_subnormal() {
        0.0f32.copysign(value)
    } else {
        value
    }
}

/// One f32 multiply under the device's rule.
fn product(a: f32, b: f32) -> f32 {
    flush(flush(a) * flush(b))
}

/// One f32 add under the device's rule.
fn sum(a: f32, b: f32) -> f32 {
    flush(flush(a) + flush(b))
}

/// The bits of a row.
fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|value| value.to_bits()).collect()
}

/// Channel `c` of the convolution in the device's order: `x` the input, the channel's taps
/// oldest first, the history's rows newest first.
fn convolved(
    x: f32,
    weights: &[f32],
    history: &[f32],
    c: usize,
    width: usize,
    taps: usize,
    norm: f32,
) -> f32 {
    let k = &weights[c * taps..c * taps + taps];
    let mut acc = sum(0.0, product(k[taps - 1], x));
    for lag in 1..taps {
        acc = sum(
            acc,
            product(k[taps - 1 - lag], history[(lag - 1) * width + c]),
        );
    }
    product(sum(acc, x), norm)
}

/// One step of the twin: the output row, and the history after it, the input then the old rows.
fn step(
    input: &[f32],
    weights: &[f32],
    history: &mut Vec<f32>,
    width: usize,
    taps: usize,
) -> Vec<f32> {
    let norm = (taps as f32 + 1.0).sqrt().recip();
    let out = (0..width)
        .map(|c| convolved(input[c], weights, history, c, width, taps, norm))
        .collect();
    let mut next = input.to_vec();
    next.extend_from_slice(history.as_slice());
    next.truncate((taps - 1) * width);
    *history = next;
    out
}

/// One whole sequence against the twin, the output in place or apart, with or without a state
/// row, every buffer `extra` bytes off the four-byte grid.
fn check_sequence(
    rig: &Rig,
    width: usize,
    taps: usize,
    steps: usize,
    in_place: bool,
    with_state: bool,
    extra: usize,
) {
    let mut rng = Rng::new(0x5c0c_0000 ^ (width as u64) ^ ((taps as u64) << 20));
    let weights = normal_row(&mut rng, width * taps);
    let weights_pad = Padded::new(&rig.ctx, &f32_bytes(&weights), extra);
    let history_pad = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; (taps - 1) * width]), extra);
    let out_pad = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width]), extra);
    let mut state = normal_row(&mut rng, width);
    let state_pad = Padded::new(&rig.ctx, &f32_bytes(&state), extra);
    let mut history = vec![0.0; (taps - 1) * width];
    for _ in 0..steps {
        let input = normal_row(&mut rng, width);
        let input_pad = Padded::new(&rig.ctx, &f32_bytes(&input), extra);
        let want = step(&input, &weights, &mut history, width, taps);
        let health = Padded::new(&rig.ctx, &HEALTH_BEFORE.to_le_bytes(), extra);
        rig.ctx
            .short_conv(ShortConvCall {
                input: input_pad.all(),
                kernel: weights_pad.all(),
                history: history_pad.all(),
                out: if in_place {
                    input_pad.all()
                } else {
                    out_pad.all()
                },
                state: with_state.then(|| state_pad.all()),
                health: health.all(),
                width,
                taps,
            })
            .expect("a valid call");
        let got = if in_place {
            input_pad.read("input")
        } else {
            out_pad.read("out")
        };
        assert_eq!(
            bits(&floats(&got)),
            bits(&want),
            "width {width}, taps {taps}: the output"
        );
        if !in_place {
            assert_eq!(
                bits(&floats(&input_pad.read("input"))),
                bits(&input),
                "the input changed"
            );
        }
        assert_eq!(
            bits(&floats(&history_pad.read("history"))),
            bits(&history),
            "the history"
        );
        if with_state {
            state = state
                .iter()
                .zip(&want)
                .map(|(&before, &value)| sum(before, value))
                .collect();
            assert_eq!(
                bits(&floats(&state_pad.read("state"))),
                bits(&state),
                "the state row"
            );
        }
        assert_eq!(
            floats(&health.read("health"))[0].to_bits(),
            HEALTH_BEFORE,
            "a finite launch changed the health word"
        );
    }
}

#[test]
fn the_short_conv_is_the_f32_convolution_across_a_sequence() {
    let rig = Rig::new();
    for taps in [2usize, 4, 8] {
        for width in [1usize, 33, 2304] {
            check_sequence(&rig, width, taps, 12, false, false, 0);
        }
    }
    check_sequence(&rig, 2304, 4, 12, false, false, 4);
}

#[test]
fn an_output_in_place_equals_one_apart() {
    let rig = Rig::new();
    check_sequence(&rig, 2304, 4, 12, true, false, 0);
    check_sequence(&rig, 129, 8, 6, true, false, 4);
}

#[test]
fn the_state_row_takes_the_convolved_row() {
    let rig = Rig::new();
    check_sequence(&rig, 2304, 4, 12, false, true, 0);
    check_sequence(&rig, 1, 2, 4, true, true, 0);
}

/// One launch over one input row, one history and an optional state row, the output apart from
/// the input: the output row and the health word after it.
fn launch(
    rig: &Rig,
    input: &[f32],
    weights: &[f32],
    history: &[f32],
    state: Option<&[f32]>,
    width: usize,
    taps: usize,
) -> (Vec<f32>, f32) {
    let input_pad = Padded::new(&rig.ctx, &f32_bytes(input), 0);
    let weights_pad = Padded::new(&rig.ctx, &f32_bytes(weights), 0);
    let history_pad = Padded::new(&rig.ctx, &f32_bytes(history), 0);
    let out_pad = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width]), 0);
    let state_pad = state.map(|row| Padded::new(&rig.ctx, &f32_bytes(row), 0));
    let health = Padded::new(&rig.ctx, &HEALTH_BEFORE.to_le_bytes(), 0);
    rig.ctx
        .short_conv(ShortConvCall {
            input: input_pad.all(),
            kernel: weights_pad.all(),
            history: history_pad.all(),
            out: out_pad.all(),
            state: state_pad.as_ref().map(Padded::all),
            health: health.all(),
            width,
            taps,
        })
        .expect("a valid call");
    let health = floats(&health.read("health"))[0];
    (floats(&out_pad.read("out")), health)
}

#[test]
fn a_value_that_is_not_finite_is_reported_in_the_health_word() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5c0c_0001);
    let (width, taps) = (64usize, 4usize);
    let weights = normal_row(&mut rng, width * taps);
    let zeros = vec![0.0; (taps - 1) * width];
    let state = vec![0.0; width];
    let finite = normal_row(&mut rng, width);
    let is_quiet_nan = |health: f32| health.to_bits() == NOT_FINITE;

    // A NaN and an infinity in the input.
    let mut input = finite.clone();
    input[17] = f32::NAN;
    assert!(
        is_quiet_nan(launch(&rig, &input, &weights, &zeros, None, width, taps).1),
        "a NaN input"
    );
    input[17] = f32::INFINITY;
    assert!(
        is_quiet_nan(launch(&rig, &input, &weights, &zeros, None, width, taps).1),
        "an infinite input"
    );

    // An infinite tap, multiplied by a zero of the history.
    let mut broken = weights.clone();
    broken[5 * taps] = f32::INFINITY;
    assert!(
        is_quiet_nan(launch(&rig, &finite, &broken, &zeros, None, width, taps).1),
        "an infinite tap"
    );

    // An infinite history value.
    let mut history = zeros.clone();
    history[width + 3] = f32::NEG_INFINITY;
    assert!(
        is_quiet_nan(launch(&rig, &finite, &weights, &history, None, width, taps).1),
        "an infinite history value"
    );

    // An output that overflows.
    let huge = vec![1.0e20; width];
    let huge_weights = vec![1.0e20; width * taps];
    assert!(
        is_quiet_nan(launch(&rig, &huge, &huge_weights, &zeros, None, width, taps).1),
        "an output that overflows"
    );

    // A state sum that overflows, the output finite.
    let full = vec![f32::MAX; width];
    let big = vec![1.0e38; width];
    assert!(
        is_quiet_nan(launch(&rig, &big, &weights, &zeros, Some(&full), width, taps).1),
        "a state sum that overflows"
    );

    // A finite launch leaves the word alone.
    let (_, health) = launch(&rig, &finite, &weights, &zeros, Some(&state), width, taps);
    assert_eq!(health.to_bits(), HEALTH_BEFORE, "a finite launch");
}

#[test]
fn a_subnormal_value_is_a_zero_of_its_sign_to_the_arithmetic() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5c0c_0002);
    let (width, taps) = (64usize, 4usize);
    let weights = normal_row(&mut rng, width * taps);
    let mut input = normal_row(&mut rng, width);
    for (i, value) in [
        1.0e-40f32,
        -1.0e-40,
        f32::from_bits(1),
        f32::MIN_POSITIVE * 0.25,
    ]
    .into_iter()
    .enumerate()
    {
        input[i] = value;
    }
    let mut history = vec![0.0; (taps - 1) * width];
    history[width + 3] = -1.0e-40;
    let before = history.clone();
    let want = step(&input, &weights, &mut history, width, taps);
    let (out, health) = launch(&rig, &input, &weights, &before, None, width, taps);
    assert_eq!(bits(&out), bits(&want), "the output");
    assert_eq!(health.to_bits(), HEALTH_BEFORE, "a subnormal is finite");
}

/// The call must be refused with `text` in its message.
fn refuses(ctx: &MetalContext, call: ShortConvCall<'_>, text: &str) {
    match ctx.short_conv(call) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains(text), "{text}: {message}")
        }
        Err(other) => panic!("another error: {other:?}"),
        Ok(()) => panic!("the call ran: {text}"),
    }
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    let (width, taps) = (2304usize, 4usize);
    let row = width * 4;
    let input = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width]), 0);
    let weights = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width * taps]), 0);
    let history = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width * (taps - 1)]), 0);
    let out = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width]), 0);
    let state = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width]), 0);
    let health = Padded::new(&rig.ctx, &HEALTH_BEFORE.to_le_bytes(), 0);
    let (input_r, weights_r, history_r, out_r, state_r, health_r) = (
        input.all(),
        weights.all(),
        history.all(),
        out.all(),
        state.all(),
        health.all(),
    );
    let valid = |width: usize, taps: usize| ShortConvCall {
        input: input_r,
        kernel: weights_r,
        history: history_r,
        out: out_r,
        state: Some(state_r),
        health: health_r,
        width,
        taps,
    };

    // The shape.
    for (bad_width, bad_taps) in [
        (0, 4),
        (SHORT_CONV_MAX_WIDTH + 1, 4),
        (2304, 1),
        (2304, SHORT_CONV_MAX_TAPS + 1),
    ] {
        refuses(
            &rig.ctx,
            valid(bad_width, bad_taps),
            "columns and 2 to 8 taps",
        );
    }

    // The ranges: short, misaligned, of another adapter.
    let alien_rig = Rig::new();
    let alien = Padded::new(&alien_rig.ctx, &f32_bytes(&vec![0.0; width]), 0);
    let alien_r = alien.all();
    // A buffer of four rows, for ranges that overlap the input at its start.
    let pool = Padded::new(&rig.ctx, &f32_bytes(&vec![0.0; width * taps]), 0);
    let pool_r = pool.range(0, row);
    let cases: Vec<(ShortConvCall<'_>, &str)> = vec![
        (
            ShortConvCall {
                input: input.range(0, row - 4),
                ..valid(width, taps)
            },
            "needs 9216 bytes of input",
        ),
        (
            ShortConvCall {
                kernel: weights.range(0, row * taps - 4),
                ..valid(width, taps)
            },
            "needs 36864 bytes of taps",
        ),
        (
            ShortConvCall {
                history: history.range(0, row * (taps - 1) - 4),
                ..valid(width, taps)
            },
            "needs 27648 bytes of history",
        ),
        (
            ShortConvCall {
                out: out.range(0, row - 4),
                ..valid(width, taps)
            },
            "needs 9216 bytes of output",
        ),
        (
            ShortConvCall {
                state: Some(state.range(0, row - 4)),
                ..valid(width, taps)
            },
            "needs 9216 bytes of state",
        ),
        (
            ShortConvCall {
                health: health.range(2, 4),
                ..valid(width, taps)
            },
            "needs 4 bytes of health at a multiple of 4",
        ),
        (
            ShortConvCall {
                input: input.range(2, row),
                ..valid(width, taps)
            },
            "needs 9216 bytes of input at a multiple of 4",
        ),
        (
            ShortConvCall {
                input: alien_r,
                ..valid(width, taps)
            },
            "of another adapter",
        ),
        (
            ShortConvCall {
                input: pool_r,
                out: pool.range(4, row),
                ..valid(width, taps)
            },
            "overlaps the input range without being it",
        ),
        (
            ShortConvCall {
                out: weights_r,
                ..valid(width, taps)
            },
            "the taps, history or health range overlaps the input or the output range",
        ),
        (
            ShortConvCall {
                input: pool_r,
                history: pool.range(0, row * (taps - 1)),
                ..valid(width, taps)
            },
            "the taps, history or health range overlaps the input or the output range",
        ),
        (
            ShortConvCall {
                history: weights.range(0, row * (taps - 1)),
                ..valid(width, taps)
            },
            "two of the taps, history and health ranges overlap",
        ),
        (
            ShortConvCall {
                state: Some(out_r),
                ..valid(width, taps)
            },
            "the state range overlaps another range",
        ),
    ];
    for (call, text) in cases {
        refuses(&rig.ctx, call, text);
    }

    // Nothing changed.
    assert_eq!(
        floats(&history.read("history")),
        vec![0.0; width * (taps - 1)]
    );
    assert_eq!(floats(&out.read("out")), vec![0.0; width]);
    assert_eq!(floats(&pool.read("pool")), vec![0.0; width * taps]);
    assert_eq!(floats(&state.read("state")), vec![0.0; width]);
    assert_eq!(floats(&health.read("health"))[0].to_bits(), HEALTH_BEFORE);
}
