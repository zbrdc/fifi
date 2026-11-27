use std::mem::size_of;

use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present, testkit::Rng,
};

use super::{ConvCall, MAX_TAPS, MAX_WIDTH};

/// Tolerance of the device result against the host mirror: 1e-6 relative, on a scale that never
/// drops below 1 (the outputs are O(1) sums of at most 8 products of standard normal values).
const RELATIVE: f32 = 1e-6;

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

fn row_bytes(width: usize) -> usize {
    width * size_of::<f32>()
}

fn upload(buffer: &DeviceBuffer, stream: &DeviceStream, offset: usize, values: &[f32]) {
    buffer.write_at(stream, offset, &bytes(values)).unwrap();
}

fn download(buffer: &DeviceBuffer, stream: &DeviceStream, offset: usize, words: usize) -> Vec<f32> {
    let mut out = vec![0u8; words * size_of::<f32>()];
    buffer.read_at(stream, offset, &mut out).unwrap();
    floats(&out)
}

fn normal_row(rng: &mut Rng, width: usize) -> Vec<f32> {
    (0..width).map(|_| rng.normal() as f32).collect()
}

/// `CausalConv::apply` of `family/inkling2/cpu_attention.rs` without dynamic taps, in f32 and
/// in its order. `history` is the host's (newest row first) and is updated like the host's.
mod host {
    pub fn apply(
        input: &[f32],
        kernel: &[f32],
        taps: usize,
        history: &mut Vec<Vec<f32>>,
    ) -> Vec<f32> {
        let norm = ((taps + 1) as f32).sqrt().recip();
        let width = input.len();
        let mut output = vec![0.0; width];
        for (lag, row) in std::iter::once(input)
            .chain(history.iter().map(Vec::as_slice))
            .enumerate()
        {
            for (channel, (result, &x)) in output.iter_mut().zip(row).enumerate() {
                *result += kernel[channel * taps + taps - 1 - lag] * x;
            }
        }
        for (result, &x) in output.iter_mut().zip(input) {
            *result = (*result + x) * norm;
        }
        history.insert(0, input.to_vec());
        history.truncate(taps - 1);
        output
    }
}

/// Where the output of a launch goes: a disjoint buffer, or the input row itself.
#[derive(Clone, Copy)]
enum Output {
    Disjoint,
    InPlace,
}

/// The first `steps` rows are inputs; the trailing `taps - 1` rows are one history that
/// the kernel advances in place. Keeping both in the same allocation also exercises
/// range validation for disjoint subranges of one buffer.
struct Sequence {
    rows: DeviceBuffer,
    kernel: DeviceBuffer,
    out: DeviceBuffer,
    health: DeviceBuffer,
    steps: usize,
    width: usize,
    taps: usize,
}

impl Sequence {
    fn new(
        ctx: &CudaContext,
        stream: &DeviceStream,
        steps: usize,
        width: usize,
        taps: usize,
        kernel: &[f32],
    ) -> Self {
        let row = row_bytes(width);
        let sequence = Self {
            rows: ctx.allocate(row * (steps + taps - 1)).unwrap(),
            kernel: ctx.allocate(row * taps).unwrap(),
            out: ctx.allocate(row).unwrap(),
            health: ctx.allocate(size_of::<f32>()).unwrap(),
            steps,
            width,
            taps,
        };
        // A fresh sequence has no past input: every history row is zero.
        upload(
            &sequence.rows,
            stream,
            0,
            &vec![0.0; width * (steps + taps - 1)],
        );
        upload(&sequence.kernel, stream, 0, kernel);
        sequence
    }

    fn row(&self, index: usize) -> usize {
        row_bytes(self.width) * index
    }

    /// The input row of `step`, uploaded before the launch.
    fn input(&self, step: usize) -> DeviceRange<'_> {
        self.rows
            .range(self.row(self.steps - 1 - step), row_bytes(self.width))
            .unwrap()
    }

    fn history(&self) -> DeviceRange<'_> {
        self.rows
            .range(
                self.row(self.steps),
                row_bytes(self.width) * (self.taps - 1),
            )
            .unwrap()
    }

    fn health_range(&self) -> DeviceRange<'_> {
        self.health.range(0, size_of::<f32>()).unwrap()
    }

    fn health(&self, stream: &DeviceStream) -> f32 {
        download(&self.health, stream, 0, 1)[0]
    }

    /// Upload only the current input. The next step must see the raw input in the
    /// newest history row, even when this step overwrites the input with its output.
    fn conv(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        step: usize,
        input: &[f32],
        output: Output,
    ) -> Result<Vec<f32>, String> {
        let row = self.row(self.steps - 1 - step);
        upload(&self.rows, stream, row, input);
        upload(&self.health, stream, 0, &[0.0]);
        let input_range = self.input(step);
        let out_range = match output {
            Output::Disjoint => self.out.range(0, row_bytes(self.width)).unwrap(),
            Output::InPlace => input_range,
        };
        ctx.inkling_conv(
            stream,
            ConvCall {
                input: input_range,
                kernel: self
                    .kernel
                    .range(0, row_bytes(self.width) * self.taps)
                    .unwrap(),
                history: self.history(),
                out: out_range,
                state: None,
                health: self.health_range(),
            },
            (self.width, self.taps),
        )
        .map_err(|error| format!("{error:?}"))?;
        let out = match output {
            Output::Disjoint => download(&self.out, stream, 0, self.width),
            Output::InPlace => download(&self.rows, stream, row, self.width),
        };
        Ok(out)
    }
}

fn assert_within_relative(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            (a - b).abs() <= RELATIVE * (1.0 + b.abs()),
            "{what}: {a} vs {b}"
        );
    }
}

/// The device follows the host's summation order with round-to-nearest operations, so
/// bit-identical results are expected; this test does not rely on them and keeps a frozen
/// tolerance of 1e-6 relative. The history is the device's own, carried across the calls.
#[test]
fn conv_matches_the_host_across_a_sequence_within_rounding() {
    if !device_present("Inkling conv sequence") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_c0a0);
    for taps in [2usize, 4, 8] {
        for width in [1usize, 33, 2304] {
            let kernel = normal_row(&mut rng, width * taps);
            let steps = 12;
            let sequence = Sequence::new(&ctx, &stream, steps, width, taps, &kernel);
            let mut history: Vec<Vec<f32>> = Vec::new();
            for step in 0..steps {
                let input = normal_row(&mut rng, width);
                let out = sequence
                    .conv(&ctx, &stream, step, &input, Output::Disjoint)
                    .unwrap();
                let expected = host::apply(&input, &kernel, taps, &mut history);
                assert_within_relative(
                    &out,
                    &expected,
                    &format!("taps {taps}, width {width}, step {step}"),
                );
                assert_eq!(
                    sequence.health(&stream).to_bits(),
                    0,
                    "taps {taps}, width {width}, step {step}: health"
                );
            }
        }
    }
}

/// A history shorter than `taps - 1` rows is zero rows on the device. `k * 0.0` and `acc + 0.0`
/// are exact, so the warm-up steps of a sequence equal the host's convolution over fewer rows
/// bit for bit.
#[test]
fn a_zero_padded_short_history_is_bit_identical_to_the_host_with_fewer_rows() {
    if !device_present("Inkling conv short history") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_c0b1);
    let width = 2304;
    for taps in [2usize, 4, 8] {
        let kernel = normal_row(&mut rng, width * taps);
        // Steps 0 ..= taps - 2: step `s` has `s` real history rows, fewer than `taps - 1`.
        let steps = taps - 1;
        let sequence = Sequence::new(&ctx, &stream, steps, width, taps, &kernel);
        let mut history: Vec<Vec<f32>> = Vec::new();
        for step in 0..steps {
            assert_eq!(history.len(), step, "taps {taps}, step {step}: host rows");
            let input = normal_row(&mut rng, width);
            let out = sequence
                .conv(&ctx, &stream, step, &input, Output::Disjoint)
                .unwrap();
            let expected = host::apply(&input, &kernel, taps, &mut history);
            assert_eq!(out, expected, "taps {taps}, step {step}");
        }
    }
}

/// `out` may be the input row itself: the kernel reads `x` before it writes `out[c]`, so the
/// result equals the one of a disjoint output bit for bit at every step of a sequence.
#[test]
fn an_output_aliasing_the_input_equals_a_disjoint_output() {
    if !device_present("Inkling conv in place") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_c0c2);
    let (width, taps, steps) = (2304usize, 4usize, 12usize);
    let kernel = normal_row(&mut rng, width * taps);
    let disjoint = Sequence::new(&ctx, &stream, steps, width, taps, &kernel);
    let in_place = Sequence::new(&ctx, &stream, steps, width, taps, &kernel);
    for step in 0..steps {
        let input = normal_row(&mut rng, width);
        let expected = disjoint
            .conv(&ctx, &stream, step, &input, Output::Disjoint)
            .unwrap();
        let out = in_place
            .conv(&ctx, &stream, step, &input, Output::InPlace)
            .unwrap();
        assert_eq!(out, expected, "step {step}");
        let newest = download(&in_place.rows, &stream, in_place.history().offset(), width);
        assert_eq!(newest, input, "step {step}: raw history");
        assert_eq!(in_place.health(&stream).to_bits(), 0, "step {step}: health");
    }
}

/// A NaN or infinite input, kernel value or history value, or a non-finite output, puts a quiet
/// NaN into the health word; the launch itself succeeds.
#[test]
fn non_finite_values_are_reported_in_the_health_word() {
    if !device_present("Inkling conv health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_c0d3);
    let (width, taps, steps) = (64usize, 4usize, 4usize);
    let kernel = normal_row(&mut rng, width * taps);
    // The kernel writes 0x7fc00000: the quiet bit is set and the payload is not signalling.
    let is_quiet_nan = |health: f32| health.is_nan() && (health.to_bits() & 0x0040_0000) != 0;

    // A NaN and an infinity in the input.
    let sequence = Sequence::new(&ctx, &stream, steps, width, taps, &kernel);
    let mut input = normal_row(&mut rng, width);
    input[17] = f32::NAN;
    sequence
        .conv(&ctx, &stream, 0, &input, Output::Disjoint)
        .unwrap();
    assert!(is_quiet_nan(sequence.health(&stream)), "NaN input");
    let mut input = normal_row(&mut rng, width);
    input[3] = f32::NEG_INFINITY;
    sequence
        .conv(&ctx, &stream, 1, &input, Output::Disjoint)
        .unwrap();
    assert!(is_quiet_nan(sequence.health(&stream)), "infinite input");

    // An infinite kernel value: the last lag of one channel.
    let mut broken = kernel.clone();
    broken[5 * taps] = f32::INFINITY;
    let sequence = Sequence::new(&ctx, &stream, steps, width, taps, &broken);
    sequence
        .conv(
            &ctx,
            &stream,
            0,
            &normal_row(&mut rng, width),
            Output::Disjoint,
        )
        .unwrap();
    assert!(
        is_quiet_nan(sequence.health(&stream)),
        "infinite kernel value"
    );

    // An infinite history value: the row step 1 reads as its newest history row is overwritten
    // after a healthy step 0.
    let sequence = Sequence::new(&ctx, &stream, steps, width, taps, &kernel);
    sequence
        .conv(
            &ctx,
            &stream,
            0,
            &normal_row(&mut rng, width),
            Output::Disjoint,
        )
        .unwrap();
    assert_eq!(sequence.health(&stream).to_bits(), 0, "finite launch");
    let mut row = normal_row(&mut rng, width);
    row[9] = f32::INFINITY;
    upload(&sequence.rows, &stream, sequence.history().offset(), &row);
    sequence
        .conv(
            &ctx,
            &stream,
            1,
            &normal_row(&mut rng, width),
            Output::Disjoint,
        )
        .unwrap();
    assert!(
        is_quiet_nan(sequence.health(&stream)),
        "infinite history value"
    );

    // Finite inputs whose output overflows: every input is f32::MAX and the first tap 2.0.
    let mut overflow = kernel.clone();
    overflow.fill(0.0);
    for channel in 0..width {
        overflow[channel * taps + taps - 1] = 2.0;
    }
    let sequence = Sequence::new(&ctx, &stream, steps, width, taps, &overflow);
    sequence
        .conv(&ctx, &stream, 0, &vec![f32::MAX; width], Output::Disjoint)
        .unwrap();
    assert!(is_quiet_nan(sequence.health(&stream)), "infinite output");
}

/// Every rejection, by its error text, before any launch: shape, extents (too small and not
/// exact), alignment, and `out` or `history` sharing bytes with the input.
#[test]
fn wrapper_rejects_bad_shapes_and_shared_bytes_by_error_text() {
    if !device_present("Inkling conv rejections") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (width, taps) = (64usize, 4usize);
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let kernel = normal_row(&mut Rng::new(0x5eed_c0d4), width * taps);
    let sequence = Sequence::new(&ctx, &stream, 8, width, taps, &kernel);
    let kernel_range = sequence.kernel.range(0, row * taps).unwrap();
    let health = sequence.health_range();
    let good_input = sequence.input(0);
    let good_history = sequence.history();
    let good_out = sequence.out.range(0, row).unwrap();
    let call = |input: DeviceRange<'_>,
                history: DeviceRange<'_>,
                out: DeviceRange<'_>,
                shape: (usize, usize)| {
        ctx.inkling_conv(
            &stream,
            ConvCall {
                input,
                kernel: kernel_range,
                history,
                out,
                state: None,
                health,
            },
            shape,
        )
        .map_err(|error| format!("{error:?}"))
    };
    let takes = format!(
        "the CUDA Inkling conv takes 1 to {MAX_WIDTH} columns and 2 to {MAX_TAPS} taps, got"
    );
    for (name, shape) in [
        ("one tap", (width, 1)),
        ("nine taps", (width, 9)),
        ("zero width", (0, taps)),
        ("width past the bound", (MAX_WIDTH + 1, taps)),
    ] {
        let error = call(good_input, good_history, good_out, shape).unwrap_err();
        assert!(error.contains(&takes), "{name}: {error}");
        assert!(
            error.contains(&format!("width {}, taps {}", shape.0, shape.1)),
            "{name}: {error}"
        );
    }

    // Extents: too small, then not exact.
    let error = call(
        good_input,
        sequence.rows.range(sequence.row(4), row * 2).unwrap(),
        good_out,
        (width, taps),
    )
    .unwrap_err();
    assert!(
        error
            .contains("the CUDA Inkling conv history range holds 512 bytes but the call needs 768"),
        "{error}"
    );
    let error = call(
        sequence.rows.range(sequence.row(4), row * 4).unwrap(),
        good_history,
        good_out,
        (width, taps),
    )
    .unwrap_err();
    assert!(
        error.contains(
            "the CUDA Inkling conv input range holds 1024 bytes but the call needs exactly 256"
        ),
        "{error}"
    );
    let error = call(
        good_input,
        good_history,
        sequence.rows.range(sequence.row(2), row * 2).unwrap(),
        (width, taps),
    )
    .unwrap_err();
    assert!(
        error.contains(
            "the CUDA Inkling conv output range holds 512 bytes but the call needs exactly 256"
        ),
        "{error}"
    );

    // Alignment: an input starting two bytes into a row.
    let error = call(
        sequence.rows.range(good_input.offset() + 2, row).unwrap(),
        good_history,
        good_out,
        (width, taps),
    )
    .unwrap_err();
    assert!(
        error.contains("the CUDA Inkling conv input range starts at offset"),
        "{error}"
    );

    // `out` one word into the input row: neither the input itself nor apart from it.
    let error = call(
        good_input,
        good_history,
        sequence
            .rows
            .range(good_input.offset() + word, row)
            .unwrap(),
        (width, taps),
    )
    .unwrap_err();
    assert!(
        error.contains("the CUDA Inkling conv output and Inkling conv input ranges overlap"),
        "{error}"
    );

    // The history starting inside the input row.
    let error = call(
        good_input,
        sequence
            .rows
            .range(good_input.offset() + word, row * (taps - 1))
            .unwrap(),
        good_out,
        (width, taps),
    )
    .unwrap_err();
    assert!(
        error.contains("the CUDA Inkling conv input and Inkling conv history ranges overlap"),
        "{error}"
    );

    // `out` exactly `input` is accepted, and the context still launches afterwards.
    call(good_input, good_history, good_input, (width, taps)).unwrap();
    call(good_input, good_history, good_out, (width, taps)).unwrap();
}

/// A `state` row takes `state += out` as the add launch would (bit-exact, one rounded add) while
/// the output is the plain convolution's; a `state` that overlaps the output or the history is
/// refused before any launch.
#[test]
fn a_state_row_takes_the_convolved_row_as_the_add_launch_would() {
    use crate::kernels::cuda::resident::depth_attention::AddCall;

    if !device_present("Inkling conv state add") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let (width, taps) = (2304usize, 4usize);
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let kernel = normal_row(&mut Rng::new(0x57a7_eadd), width * taps);
    let sequence = Sequence::new(&ctx, &stream, 8, width, taps, &kernel);
    // The host history carries across the steps as the device's does.
    let mut host_history = Vec::new();
    for step in 0..6usize {
        let input = normal_row(&mut Rng::new(100 + step as u64), width);
        let history_before = download(
            &sequence.rows,
            &stream,
            sequence.row(sequence.steps),
            width * (taps - 1),
        );
        let plain = sequence
            .conv(&ctx, &stream, step, &input, Output::Disjoint)
            .unwrap();
        let current = normal_row(&mut Rng::new(500 + step as u64), width);
        let expected = host::apply(&input, &kernel, taps, &mut host_history);
        assert_within_relative(&plain, &expected, "plain output");
        // The state row: the plain output added into it by the add launch ...
        let state = ctx.allocate(row * 2).unwrap();
        upload(&state, &stream, 0, &current);
        upload(
            &state,
            &stream,
            row,
            &normal_row(&mut Rng::new(900 + step as u64), width),
        );
        upload(&sequence.health, &stream, 0, &[0.0]);
        ctx.depth_attention_add(
            &stream,
            AddCall {
                state: state.range(0, row).unwrap(),
                row: sequence.out.range(0, row).unwrap(),
                health: sequence.health_range(),
            },
            width,
        )
        .unwrap();
        let added = download(&state, &stream, 0, width);
        assert_eq!(sequence.health(&stream).to_bits(), 0);
        let host_added: Vec<f32> = current.iter().zip(&plain).map(|(a, b)| a + b).collect();
        assert_eq!(added, host_added, "step {step}: the add launch");
        // ... equals the fused launch: `state[c] += out[c]` inside the convolution.
        upload(&state, &stream, 0, &current);
        upload(&state, &stream, row, &input);
        // The same history the plain convolution saw.
        upload(
            &sequence.rows,
            &stream,
            sequence.row(sequence.steps),
            &history_before,
        );
        upload(&sequence.health, &stream, 0, &[0.0]);
        ctx.inkling_conv(
            &stream,
            ConvCall {
                input: state.range(row, row).unwrap(),
                kernel: sequence.kernel.range(0, row * taps).unwrap(),
                history: sequence.history(),
                out: sequence.out.range(0, row).unwrap(),
                state: Some(state.range(0, row).unwrap()),
                health: sequence.health_range(),
            },
            (width, taps),
        )
        .unwrap();
        let fused = download(&sequence.out, &stream, 0, width);
        let fused_state = download(&state, &stream, 0, width);
        assert_within_relative(&fused, &expected, "fused output");
        assert_eq!(fused_state, added, "step {step}: the state add");
        assert_eq!(sequence.health(&stream).to_bits(), 0);
    }
    // A `state` sharing bytes with the output is refused by error text, before any launch.
    let input = normal_row(&mut Rng::new(7), width);
    upload(&sequence.rows, &stream, 0, &input);
    let error = ctx
        .inkling_conv(
            &stream,
            ConvCall {
                input: sequence.input(0),
                kernel: sequence.kernel.range(0, row * taps).unwrap(),
                history: sequence.history(),
                out: sequence.out.range(0, row).unwrap(),
                state: Some(sequence.out.range(word, row - word).unwrap()),
                health: sequence.health_range(),
            },
            (width, taps),
        )
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("Inkling conv state"),
        "{error:?}"
    );
}
