use std::mem::size_of;

use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceStream, device_present, lane_mix, lane_norm, testkit::Rng,
};

use super::{
    AddCall, MAX_BLOCKS, MAX_MIX_JOBS, MAX_NORM_ROWS, MixCall, MixJob, MixShape, MixStatesCall,
    MixStatesShape, NormCall, SealCall, States, StreamShape,
};

const EPS: f32 = 1e-5;

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

fn upload(buffer: &DeviceBuffer, stream: &DeviceStream, values: &[f32]) {
    buffer.write_at(stream, 0, &bytes(values)).unwrap();
}

fn download(buffer: &DeviceBuffer, stream: &DeviceStream, offset: usize, words: usize) -> Vec<f32> {
    let mut out = vec![0u8; words * size_of::<f32>()];
    buffer.read_at(stream, offset, &mut out).unwrap();
    floats(&out)
}

fn normal_row(rng: &mut Rng, width: usize) -> Vec<f32> {
    (0..width).map(|_| rng.normal() as f32).collect()
}

/// The host algorithm of `cpu_forward/depth_mix.rs`, in f32 and in its order.
mod host {
    fn rms_norm(row: &mut [f32], eps: f32) {
        let dot: f32 = row.iter().map(|v| v * v).sum();
        let scale = (dot / row.len() as f32 + eps).sqrt().recip();
        for value in row {
            *value *= scale;
        }
    }

    /// `Stream::seal(next)`: returns the sealed row, leaves `next` as the current row.
    pub fn seal(current: &mut Vec<f32>, next: Vec<f32>, eps: f32) -> Vec<f32> {
        let mut previous = std::mem::replace(current, next);
        rms_norm(&mut previous, eps);
        previous
    }

    /// `rms_norm_with_scale` of the host: the normalized copy of `src`.
    pub fn norm(src: &[f32], eps: f32) -> Vec<f32> {
        let mut row = src.to_vec();
        rms_norm(&mut row, eps);
        row
    }

    /// `Stream::mix(first_block, query)` of a state with normalized sealed rows.
    pub fn mix(
        sealed: &[Vec<f32>],
        current: &[f32],
        first_block: usize,
        query: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mut normalized = current.to_vec();
        rms_norm(&mut normalized, eps);
        let rows: Vec<&[f32]> = sealed[first_block..]
            .iter()
            .rev()
            .map(|row| row.as_slice())
            .chain([normalized.as_slice()])
            .collect();
        let scale = (query.len() as f32).sqrt().recip();
        let scores: Vec<f32> = rows
            .iter()
            .map(|row| row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>() * scale)
            .collect();
        let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut mass = 0.0f32;
        let mut output = vec![0.0f32; query.len()];
        for (score, row) in scores.into_iter().zip(&rows) {
            let weight = crate::kernels::portable_exp::portable_exp(score - peak);
            mass += weight;
            for (out, value) in output.iter_mut().zip(row.iter()) {
                *out += weight * value;
            }
        }
        for value in &mut output {
            *value /= mass;
        }
        output
    }
}

/// One stream state of `MAX_BLOCKS + 2` rows, a spare row, a query, an output and a health word.
struct Buffers {
    state: DeviceBuffer,
    row: DeviceBuffer,
    query: DeviceBuffer,
    out: DeviceBuffer,
    health: DeviceBuffer,
}

impl Buffers {
    fn new(ctx: &CudaContext, width: usize) -> Self {
        let row = row_bytes(width);
        Self {
            state: ctx.allocate(row * (MAX_BLOCKS + 2)).unwrap(),
            row: ctx.allocate(row).unwrap(),
            query: ctx.allocate(row).unwrap(),
            out: ctx.allocate(row).unwrap(),
            health: ctx.allocate(size_of::<f32>()).unwrap(),
        }
    }

    fn clear_health(&self, stream: &DeviceStream) {
        upload(&self.health, stream, &[0.0]);
    }

    fn health(&self, stream: &DeviceStream) -> f32 {
        download(&self.health, stream, 0, 1)[0]
    }

    /// Uploads `state` (the current row, then the sealed rows) and `query`, mixes, reads `out`.
    #[allow(clippy::too_many_arguments, reason = "one test fixture")]
    fn mix(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &[f32],
        query: &[f32],
        first_block: usize,
        blocks: usize,
        width: usize,
    ) -> Result<Vec<f32>, String> {
        upload(&self.state, stream, state);
        upload(&self.query, stream, query);
        self.clear_health(stream);
        ctx.depth_attention_mix(
            stream,
            MixCall {
                state: self
                    .state
                    .range(0, row_bytes(width) * (blocks + 1))
                    .unwrap(),
                query: self.query.range(0, row_bytes(width)).unwrap(),
                out: self.out.range(0, row_bytes(width)).unwrap(),
                health: self.health.range(0, size_of::<f32>()).unwrap(),
            },
            MixShape {
                first_block,
                blocks,
                width,
                eps: EPS,
            },
        )
        .map_err(|error| format!("{error:?}"))?;
        Ok(download(&self.out, stream, 0, width))
    }
}

fn max_error(out: &[f32], expected: &[f32]) -> f32 {
    out.iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Within rounding of the host's `Stream::mix`, and bit for bit the lane-order model of the
/// device mix ([`lane_mix`]).
#[test]
fn mix_matches_the_host_algorithm_within_rounding_and_the_lane_order_bit_for_bit() {
    if !device_present("depth stream mix") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_572e);
    for width in [1usize, 33, 2304] {
        let buffers = Buffers::new(&ctx, width);
        let query = normal_row(&mut rng, width);
        let mut current = normal_row(&mut rng, width);
        let mut sealed: Vec<Vec<f32>> = Vec::new();
        for blocks in 0..=12usize {
            for first_block in [0, blocks / 2, blocks] {
                let mut state = current.clone();
                for row in &sealed {
                    state.extend_from_slice(row);
                }
                let out = buffers
                    .mix(&ctx, &stream, &state, &query, first_block, blocks, width)
                    .unwrap();
                let expected = host::mix(&sealed, &current, first_block, &query, EPS);
                let error = max_error(&out, &expected);
                assert!(
                    error <= 1e-4,
                    "width {width} blocks {blocks} first_block {first_block}: max error {error}"
                );
                let rows: Vec<&[f32]> = sealed.iter().map(Vec::as_slice).collect();
                let lane = lane_mix(&current, &rows, first_block, &query, EPS);
                assert!(
                    out.iter()
                        .map(|v| v.to_bits())
                        .eq(lane.iter().map(|v| v.to_bits())),
                    "width {width} blocks {blocks} first_block {first_block}: lane order bits"
                );
                let health = buffers.health(&stream);
                assert_eq!(health.to_bits(), 0, "width {width} blocks {blocks}");
            }
            let next = normal_row(&mut rng, width);
            sealed.push(host::seal(&mut current, next, EPS));
        }
    }
}

#[test]
fn seal_stores_the_normalized_row_and_add_is_bit_exact() {
    if !device_present("depth stream seal and add") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_572f);
    let width = 2304;
    let row = row_bytes(width);
    let buffers = Buffers::new(&ctx, width);
    let mut current = normal_row(&mut rng, width);
    let mut sealed: Vec<Vec<f32>> = Vec::new();
    for blocks in 0..12usize {
        let next = normal_row(&mut rng, width);
        let mut state = current.clone();
        for row in &sealed {
            state.extend_from_slice(row);
        }
        upload(&buffers.state, &stream, &state);
        upload(&buffers.row, &stream, &next);
        buffers.clear_health(&stream);
        ctx.depth_attention_seal(
            &stream,
            SealCall {
                state: buffers.state.range(0, row * (blocks + 2)).unwrap(),
                next: buffers.row.range(0, row).unwrap(),
                health: buffers.health.range(0, size_of::<f32>()).unwrap(),
            },
            StreamShape {
                blocks,
                width,
                eps: EPS,
            },
        )
        .unwrap();
        let expected = host::seal(&mut current, next.clone(), EPS);
        let stored = download(&buffers.state, &stream, row * (blocks + 1), width);
        for (a, b) in stored.iter().zip(&expected) {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "blocks {blocks}: {a} vs {b}"
            );
        }
        assert_eq!(download(&buffers.state, &stream, 0, width), next);
        assert_eq!(buffers.health(&stream).to_bits(), 0, "blocks {blocks}");
        sealed.push(expected);
    }

    let addend = normal_row(&mut rng, width);
    upload(&buffers.state, &stream, &current);
    upload(&buffers.row, &stream, &addend);
    buffers.clear_health(&stream);
    ctx.depth_attention_add(
        &stream,
        AddCall {
            state: buffers.state.range(0, row).unwrap(),
            row: buffers.row.range(0, row).unwrap(),
            health: buffers.health.range(0, size_of::<f32>()).unwrap(),
        },
        width,
    )
    .unwrap();
    let expected: Vec<f32> = current.iter().zip(&addend).map(|(a, b)| a + b).collect();
    assert_eq!(download(&buffers.state, &stream, 0, width), expected);
    assert_eq!(buffers.health(&stream).to_bits(), 0);
}

#[test]
fn non_finite_arithmetic_is_reported_in_the_health_word() {
    if !device_present("depth stream health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 64;
    let row = row_bytes(width);
    let buffers = Buffers::new(&ctx, width);
    let query = normal_row(&mut Rng::new(7), width);
    let mut broken = normal_row(&mut Rng::new(8), width);
    broken[17] = f32::INFINITY;
    let out = buffers
        .mix(&ctx, &stream, &broken, &query, 0, 0, width)
        .unwrap();
    assert!(buffers.health(&stream).is_nan());
    assert!(out.iter().any(|value| value.is_nan()));

    let mut current = normal_row(&mut Rng::new(9), width);
    let mut next = normal_row(&mut Rng::new(10), width);
    next[3] = f32::NAN;
    upload(&buffers.state, &stream, &current);
    upload(&buffers.row, &stream, &next);
    buffers.clear_health(&stream);
    ctx.depth_attention_seal(
        &stream,
        SealCall {
            state: buffers.state.range(0, row * 2).unwrap(),
            next: buffers.row.range(0, row).unwrap(),
            health: buffers.health.range(0, size_of::<f32>()).unwrap(),
        },
        StreamShape {
            blocks: 0,
            width,
            eps: EPS,
        },
    )
    .unwrap();
    assert!(buffers.health(&stream).is_nan());

    current[5] = f32::NEG_INFINITY;
    let addend = vec![f32::INFINITY; width];
    upload(&buffers.state, &stream, &current);
    upload(&buffers.row, &stream, &addend);
    buffers.clear_health(&stream);
    ctx.depth_attention_add(
        &stream,
        AddCall {
            state: buffers.state.range(0, row).unwrap(),
            row: buffers.row.range(0, row).unwrap(),
            health: buffers.health.range(0, size_of::<f32>()).unwrap(),
        },
        width,
    )
    .unwrap();
    assert!(buffers.health(&stream).is_nan());
}

#[test]
fn wrapper_rejects_bad_shapes_and_shared_bytes_before_launching() {
    if !device_present("depth stream rejections") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 64;
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let buffers = Buffers::new(&ctx, width);
    let shape = |first_block, blocks, eps| MixShape {
        first_block,
        blocks,
        width,
        eps,
    };
    // `mix` names the error of one launch; `state` is an offset and a byte count.
    let mix = |state: (usize, usize), out: usize, health: usize, shape| {
        ctx.depth_attention_mix(
            &stream,
            MixCall {
                state: buffers.state.range(state.0, state.1).unwrap(),
                query: buffers.query.range(0, row).unwrap(),
                out: buffers.out.range(out, row).unwrap(),
                health: buffers.health.range(health, word).unwrap(),
            },
            shape,
        )
        .map_err(|error| format!("{error:?}"))
    };
    let takes = "the CUDA depth stream mix takes";
    for bad in [
        shape(1, 0, EPS),
        shape(0, MAX_BLOCKS + 1, EPS),
        shape(0, 0, 0.0),
        shape(0, 0, f32::NAN),
        MixShape {
            first_block: 0,
            blocks: 0,
            width: 0,
            eps: EPS,
        },
    ] {
        let error = mix((0, row * 2), 0, 0, bad).unwrap_err();
        assert!(error.contains(takes), "{bad:?}: {error}");
    }
    let error = mix((0, row), 0, 0, shape(0, 1, EPS)).unwrap_err();
    assert!(
        error.contains("the CUDA depth stream state range holds 256 bytes but the call needs 512"),
        "{error}"
    );
    let error = mix((0, row * 2), 0, 0, shape(0, 0, EPS)).unwrap_err();
    assert!(
        error.contains("holds 512 bytes but the call needs exactly 256"),
        "{error}"
    );

    // The output inside the two state rows the launch reads.
    let error = ctx
        .depth_attention_mix(
            &stream,
            MixCall {
                state: buffers.state.range(0, row * 2).unwrap(),
                query: buffers.query.range(0, row).unwrap(),
                out: buffers.state.range(row, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            shape(0, 1, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains("the CUDA depth stream mix output and depth stream state ranges overlap"),
        "{error}"
    );
    // The health word inside the output.
    let error = ctx
        .depth_attention_mix(
            &stream,
            MixCall {
                state: buffers.state.range(0, row * 2).unwrap(),
                query: buffers.query.range(0, row).unwrap(),
                out: buffers.out.range(0, row).unwrap(),
                health: buffers.out.range(row - word, word).unwrap(),
            },
            shape(0, 1, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains("the CUDA depth stream mix output and depth stream health ranges overlap"),
        "{error}"
    );
    // The same allocation, different bytes: accepted, and the context still launches.
    ctx.depth_attention_mix(
        &stream,
        MixCall {
            state: buffers.state.range(0, row * 2).unwrap(),
            query: buffers.query.range(0, row).unwrap(),
            out: buffers.state.range(row * 2, row).unwrap(),
            health: buffers.health.range(0, word).unwrap(),
        },
        shape(0, 1, EPS),
    )
    .unwrap();

    let error = ctx
        .depth_attention_seal(
            &stream,
            SealCall {
                state: buffers.state.range(0, row * 2).unwrap(),
                next: buffers.state.range(row, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            StreamShape {
                blocks: 0,
                width,
                eps: EPS,
            },
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains("the CUDA depth stream state and depth stream next row ranges overlap"),
        "{error}"
    );
    let error = ctx
        .depth_attention_add(
            &stream,
            AddCall {
                state: buffers.state.range(0, row).unwrap(),
                row: buffers.state.range(0, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            width,
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains("the CUDA depth stream state and depth stream addend row ranges overlap"),
        "{error}"
    );
    let error = ctx
        .depth_attention_add(
            &stream,
            AddCall {
                state: buffers.state.range(0, row).unwrap(),
                row: buffers.row.range(0, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            },
            0,
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(error.contains("the CUDA depth stream add takes"), "{error}");
}

#[test]
fn norm_is_the_lane_order_scale_and_refuses_to_normalize_in_place() {
    if !device_present("depth stream norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 2304;
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let buffers = Buffers::new(&ctx, width);
    let src = normal_row(&mut Rng::new(0x5eed_5730), width);
    upload(&buffers.row, &stream, &src);
    buffers.clear_health(&stream);
    ctx.depth_attention_norm_rows(
        &stream,
        &[NormCall {
            src: buffers.row.range(0, row).unwrap(),
            dst: buffers.out.range(0, row).unwrap(),
            health: buffers.health.range(0, word).unwrap(),
        }],
        (width, EPS),
    )
    .unwrap();
    let expected = lane_norm(&src, None, EPS);
    let actual = download(&buffers.out, &stream, 0, width);
    assert!(
        actual
            .iter()
            .map(|v| v.to_bits())
            .eq(expected.iter().map(|v| v.to_bits()))
    );
    assert_eq!(buffers.health(&stream).to_bits(), 0);

    let error = ctx
        .depth_attention_norm_rows(
            &stream,
            &[NormCall {
                src: buffers.out.range(0, row).unwrap(),
                dst: buffers.out.range(0, row).unwrap(),
                health: buffers.health.range(0, word).unwrap(),
            }],
            (width, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains(
            "the CUDA depth stream norm output and depth stream norm source ranges overlap"
        ),
        "{error}"
    );
}

/// One launch over several states, `stride` rows apart with gaps between them, gives each state
/// the bits of its own single-state launch: mix, then seal, then add.
#[test]
fn states_in_one_launch_match_one_launch_each_bit_for_bit() {
    if !device_present("depth stream states") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x0057_47e5);
    let (width, blocks, rows) = (2304usize, 4usize, 5usize);
    let stride = blocks + 3;
    let row = row_bytes(width);
    let states: Vec<f32> = (0..rows * stride)
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let query = normal_row(&mut rng, width);
    let addends: Vec<f32> = (0..rows)
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let next: Vec<f32> = (0..rows)
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let alloc = |words: usize| ctx.allocate(words * size_of::<f32>()).unwrap();
    let (state, single) = (alloc(states.len()), alloc(stride * width));
    let (q, out, one_out) = (alloc(width), alloc(rows * width), alloc(width));
    let (adds, nexts, health) = (alloc(addends.len()), alloc(next.len()), alloc(1));
    upload(&state, &stream, &states);
    upload(&q, &stream, &query);
    upload(&adds, &stream, &addends);
    upload(&nexts, &stream, &next);
    upload(&health, &stream, &[0.0]);
    let many = States { rows, stride };
    let span = |used: usize| ((rows - 1) * stride + used) * row;
    let mix = MixShape {
        first_block: 1,
        blocks,
        width,
        eps: EPS,
    };
    ctx.depth_attention_mix_states(
        &stream,
        MixStatesCall {
            state: state.range(0, span(blocks + 1)).unwrap(),
            query: Some(q.range(0, row).unwrap()),
            gamma: None,
            out: out.range(0, rows * row).unwrap(),
            health: health.range(0, 4).unwrap(),
        },
        MixStatesShape {
            first_block: mix.first_block,
            blocks,
            width,
            out_stride: width,
            mix_eps: EPS,
            norm_eps: 1.0,
        },
        many,
    )
    .unwrap();
    let seal = StreamShape {
        blocks,
        width,
        eps: EPS,
    };
    ctx.depth_attention_seal_states(
        &stream,
        SealCall {
            state: state.range(0, span(blocks + 2)).unwrap(),
            next: nexts.range(0, rows * row).unwrap(),
            health: health.range(0, 4).unwrap(),
        },
        seal,
        many,
    )
    .unwrap();
    ctx.depth_attention_add_states(
        &stream,
        AddCall {
            state: state.range(0, span(1)).unwrap(),
            row: adds.range(0, rows * row).unwrap(),
            health: health.range(0, 4).unwrap(),
        },
        width,
        many,
    )
    .unwrap();
    let mixed = download(&out, &stream, 0, rows * width);
    let after = download(&state, &stream, 0, rows * stride * width);
    for r in 0..rows {
        let own = &states[r * stride * width..(r + 1) * stride * width];
        upload(&single, &stream, own);
        ctx.depth_attention_mix(
            &stream,
            MixCall {
                state: single.range(0, (blocks + 1) * row).unwrap(),
                query: q.range(0, row).unwrap(),
                out: one_out.range(0, row).unwrap(),
                health: health.range(0, 4).unwrap(),
            },
            mix,
        )
        .unwrap();
        let expected = download(&one_out, &stream, 0, width);
        assert!(
            mixed[r * width..(r + 1) * width]
                .iter()
                .map(|v| v.to_bits())
                .eq(expected.iter().map(|v| v.to_bits())),
            "mix of state {r}"
        );
        ctx.depth_attention_seal(
            &stream,
            SealCall {
                state: single.range(0, (blocks + 2) * row).unwrap(),
                next: nexts.range(r * row, row).unwrap(),
                health: health.range(0, 4).unwrap(),
            },
            seal,
        )
        .unwrap();
        ctx.depth_attention_add(
            &stream,
            AddCall {
                state: single.range(0, row).unwrap(),
                row: adds.range(r * row, row).unwrap(),
                health: health.range(0, 4).unwrap(),
            },
            width,
        )
        .unwrap();
        let expected = download(&single, &stream, 0, stride * width);
        let got = &after[r * stride * width..(r + 1) * stride * width];
        assert!(
            got.iter()
                .map(|v| v.to_bits())
                .eq(expected.iter().map(|v| v.to_bits())),
            "seal and add of state {r}"
        );
    }
    assert_eq!(download(&health, &stream, 0, 1)[0], 0.0);
}

/// Six gate halves in one launch, each with its own health word, give each row the bits of its
/// own one-row launch; then the refusals.
#[test]
fn norm_rows_match_one_row_each_bit_for_bit() {
    if !device_present("depth stream norm rows") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x0d0e_5ca1);
    let (width, count) = (2304usize, 6usize);
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let alloc = |words: usize| ctx.allocate(words * size_of::<f32>()).unwrap();
    let sources: Vec<f32> = (0..count)
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let (src, dst, one, health) = (
        alloc(count * width),
        alloc(count * width),
        alloc(width),
        alloc(count),
    );
    upload(&src, &stream, &sources);
    upload(&health, &stream, &vec![0.0; count]);
    // Row `r` of `src` normed into `out`, with its own health word. `out` is left to inference
    // so that every call shares the buffers' lifetime.
    let call = |r: usize, out| NormCall {
        src: src.range(r * row, row).unwrap(),
        dst: out,
        health: health.range(r * word, word).unwrap(),
    };
    let rows: Vec<NormCall<'_>> = (0..count)
        .map(|r| call(r, dst.range(r * row, row).unwrap()))
        .collect();
    ctx.depth_attention_norm_rows(&stream, &rows, (width, EPS))
        .unwrap();
    let together = download(&dst, &stream, 0, count * width);
    for r in 0..count {
        ctx.depth_attention_norm_rows(
            &stream,
            &[call(r, one.range(0, row).unwrap())],
            (width, EPS),
        )
        .unwrap();
        let alone = download(&one, &stream, 0, width);
        let same = together[r * width..][..width]
            .iter()
            .zip(&alone)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "row {r} differs from its own launch");
        let expected = host::norm(&sources[r * width..][..width], EPS);
        for (a, b) in alone.iter().zip(&expected) {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "row {r}: {a} vs {b}"
            );
        }
    }
    assert!(
        download(&health, &stream, 0, count)
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    let refused = |rows: &[NormCall<'_>]| {
        ctx.depth_attention_norm_rows(&stream, rows, (width, EPS))
            .is_err()
    };
    assert!(refused(&[]));
    let many: Vec<NormCall<'_>> = (0..=MAX_NORM_ROWS)
        .map(|r| call(r % count, dst.range((r % count) * row, row).unwrap()))
        .collect();
    assert!(refused(&many));
    // Two rows writing one output, an output on another row's source, health on a source.
    assert!(refused(&[
        call(0, one.range(0, row).unwrap()),
        call(1, one.range(0, row).unwrap()),
    ]));
    assert!(refused(&[
        call(0, src.range(row, row).unwrap()),
        call(1, dst.range(row, row).unwrap()),
    ]));
    assert!(refused(&[NormCall {
        src: src.range(0, row).unwrap(),
        dst: dst.range(0, row).unwrap(),
        health: src.range(0, word).unwrap(),
    }]));
}

/// The mixes of a visit in one launch: four queries on one state (the bank) and one on another
/// (the depth), each output the bits of its own single launch; then the refusals.
#[test]
fn mix_jobs_match_one_launch_each_bit_for_bit() {
    if !device_present("depth stream mix jobs") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x0b10_c4ed);
    let width = 2304usize;
    let row = row_bytes(width);
    let alloc = |words: usize| ctx.allocate(words * size_of::<f32>()).unwrap();
    // (state blocks, first block) of each job: four on the bank state, one on the depth state.
    let (bank_blocks, depth_blocks) = (5usize, 7usize);
    let extents = [
        (bank_blocks, 2),
        (bank_blocks, 2),
        (bank_blocks, 2),
        (bank_blocks, 2),
    ]
    .into_iter()
    .chain([(depth_blocks, 0)])
    .collect::<Vec<_>>();
    let bank: Vec<f32> = (0..=bank_blocks)
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let depth: Vec<f32> = (0..=depth_blocks)
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let queries: Vec<f32> = (0..extents.len())
        .flat_map(|_| normal_row(&mut rng, width))
        .collect();
    let (bank_state, depth_state) = (alloc(bank.len()), alloc(depth.len()));
    let (query, out, one, health) = (
        alloc(queries.len()),
        alloc(extents.len() * width),
        alloc(width),
        alloc(1),
    );
    upload(&bank_state, &stream, &bank);
    upload(&depth_state, &stream, &depth);
    upload(&query, &stream, &queries);
    upload(&health, &stream, &[0.0]);
    let state_of = |j: usize| {
        let (blocks, _) = extents[j];
        let buffer = if j < 4 { &bank_state } else { &depth_state };
        buffer.range(0, (blocks + 1) * row).unwrap()
    };
    let jobs: Vec<MixJob<'_>> = extents
        .iter()
        .enumerate()
        .map(|(j, &(blocks, first_block))| MixJob {
            state: state_of(j),
            query: query.range(j * row, row).unwrap(),
            out: out.range(j * row, row).unwrap(),
            first_block,
            blocks,
        })
        .collect();
    let health_range = health.range(0, 4).unwrap();
    ctx.depth_attention_mix_jobs(&stream, &jobs, &health_range, (width, EPS))
        .unwrap();
    let together = download(&out, &stream, 0, extents.len() * width);
    for (j, &(blocks, first_block)) in extents.iter().enumerate() {
        ctx.depth_attention_mix(
            &stream,
            MixCall {
                state: state_of(j),
                query: query.range(j * row, row).unwrap(),
                out: one.range(0, row).unwrap(),
                health: health.range(0, 4).unwrap(),
            },
            MixShape {
                first_block,
                blocks,
                width,
                eps: EPS,
            },
        )
        .unwrap();
        let alone = download(&one, &stream, 0, width);
        let same = together[j * width..][..width]
            .iter()
            .zip(&alone)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "job {j} differs from its own launch");
    }
    assert_eq!(download(&health, &stream, 0, 1)[0].to_bits(), 0);
    // Refusals: no jobs, too many jobs, one output twice, an output on a read state.
    let refused = |jobs: &[MixJob<'_>]| {
        ctx.depth_attention_mix_jobs(&stream, jobs, &health_range, (width, EPS))
            .is_err()
    };
    assert!(refused(&[]));
    let many: Vec<MixJob<'_>> = (0..=MAX_MIX_JOBS)
        .map(|_| MixJob {
            state: state_of(0),
            query: query.range(0, row).unwrap(),
            out: one.range(0, row).unwrap(),
            first_block: 0,
            blocks: bank_blocks,
        })
        .collect();
    assert!(refused(&many));
    assert!(refused(&many[..2]));
    let onto_state = [
        MixJob {
            state: state_of(4),
            query: query.range(0, row).unwrap(),
            out: bank_state.range(0, row).unwrap(),
            first_block: 0,
            blocks: depth_blocks,
        },
        MixJob {
            state: state_of(0),
            query: query.range(row, row).unwrap(),
            out: one.range(0, row).unwrap(),
            first_block: 0,
            blocks: bank_blocks,
        },
    ];
    assert!(refused(&onto_state));
    let wide = [MixJob {
        state: state_of(4),
        query: query.range(0, row).unwrap(),
        out: one.range(0, row).unwrap(),
        first_block: 0,
        blocks: MAX_BLOCKS + 1,
    }];
    assert!(refused(&wide));
}
