use std::mem::size_of;

use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, device_present, testkit::Rng};

use super::{
    AddCall, MAX_BLOCKS, MAX_MIX_JOBS, MAX_WORD_ROWS, MixCall, MixJob, MixNorm, MixShape, NormCall,
    SealCall, StreamShape,
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
            let weight = (score - peak).exp();
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
        ctx.inkling_stream_mix(
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

#[test]
fn mix_matches_the_host_algorithm_within_rounding() {
    if !device_present("Inkling stream mix") {
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
    if !device_present("Inkling stream seal and add") {
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
        ctx.inkling_stream_seal(
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
    ctx.inkling_stream_add(
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
    if !device_present("Inkling stream health") {
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
    ctx.inkling_stream_seal(
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
    ctx.inkling_stream_add(
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
    if !device_present("Inkling stream rejections") {
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
        ctx.inkling_stream_mix(
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
    let takes = "the CUDA Inkling stream mix takes";
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
        error
            .contains("the CUDA Inkling stream state range holds 256 bytes but the call needs 512"),
        "{error}"
    );
    let error = mix((0, row * 2), 0, 0, shape(0, 0, EPS)).unwrap_err();
    assert!(
        error.contains("holds 512 bytes but the call needs exactly 256"),
        "{error}"
    );

    // The output inside the two state rows the launch reads.
    let error = ctx
        .inkling_stream_mix(
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
        error
            .contains("the CUDA Inkling stream mix output and Inkling stream state ranges overlap"),
        "{error}"
    );
    // The health word inside the output.
    let error = ctx
        .inkling_stream_mix(
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
        error.contains(
            "the CUDA Inkling stream mix output and Inkling stream health ranges overlap"
        ),
        "{error}"
    );
    // The same allocation, different bytes: accepted, and the context still launches.
    ctx.inkling_stream_mix(
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
        .inkling_stream_seal(
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
        error.contains("the CUDA Inkling stream state and Inkling stream next row ranges overlap"),
        "{error}"
    );
    let error = ctx
        .inkling_stream_add(
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
        error
            .contains("the CUDA Inkling stream state and Inkling stream addend row ranges overlap"),
        "{error}"
    );
    let error = ctx
        .inkling_stream_add(
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
    assert!(
        error.contains("the CUDA Inkling stream add takes"),
        "{error}"
    );
}

#[test]
fn norm_rows_match_the_host_scale_and_refuse_shared_bytes() {
    if !device_present("Inkling stream norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 2304;
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let buffers = Buffers::new(&ctx, width);
    let mut rng = Rng::new(0x5eed_5730);
    let rows = [normal_row(&mut rng, width), normal_row(&mut rng, width)];
    // Two sources: the test row and its negation (the same scale, a different row).
    upload(&buffers.row, &stream, &rows[0]);
    upload(&buffers.out, &stream, &rows[1]);
    buffers.clear_health(&stream);
    let call = |src, dst, health| NormCall { src, dst, health };
    ctx.inkling_stream_norm_rows(
        &stream,
        &[
            call(
                buffers.row.range(0, row).unwrap(),
                buffers.state.range(0, row).unwrap(),
                buffers.health.range(0, word).unwrap(),
            ),
            call(
                buffers.out.range(0, row).unwrap(),
                buffers.query.range(0, row).unwrap(),
                buffers.health.range(0, word).unwrap(),
            ),
        ],
        (width, EPS),
    )
    .unwrap();
    for (buffer, source) in [(&buffers.state, &rows[0]), (&buffers.query, &rows[1])] {
        let expected = host::norm(source, EPS);
        for (a, b) in download(buffer, &stream, 0, width).iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
    // Both rows named one health word: each block found every value finite and left it zero.
    assert_eq!(buffers.health(&stream).to_bits(), 0);

    let error = ctx
        .inkling_stream_norm_rows(
            &stream,
            &[
                call(
                    buffers.row.range(0, row).unwrap(),
                    buffers.state.range(0, row).unwrap(),
                    buffers.health.range(0, word).unwrap(),
                ),
                call(
                    buffers.out.range(0, row).unwrap(),
                    buffers.state.range(0, row).unwrap(),
                    buffers.health.range(0, word).unwrap(),
                ),
            ],
            (width, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains(
            "the CUDA Inkling stream norm output and Inkling stream norm output ranges overlap"
        ),
        "{error}"
    );
    let error = ctx
        .inkling_stream_norm_rows(
            &stream,
            &[call(
                buffers.state.range(0, row).unwrap(),
                buffers.state.range(0, row).unwrap(),
                buffers.health.range(0, word).unwrap(),
            )],
            (width, EPS),
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(
        error.contains(
            "the CUDA Inkling stream norm output and Inkling stream norm source ranges overlap"
        ),
        "{error}"
    );
}

/// The host algorithm's stream state for `blocks` sealed rows and a new current row.
fn job_state(rng: &mut Rng, width: usize, blocks: usize) -> Vec<f32> {
    let mut current = normal_row(rng, width);
    let mut state = Vec::new();
    for _ in 0..blocks {
        let sealed = host::seal(&mut current, normal_row(rng, width), EPS);
        state.push(sealed);
    }
    current.extend(state.into_iter().flatten());
    current
}

#[test]
fn mix_jobs_equal_the_single_launches_bit_for_bit() {
    if !device_present("Inkling stream mix jobs") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    ctx.set_dense_group_policy(
        crate::kernels::cuda::resident::dense_gemv::DenseGroupPolicy {
            decode_glue: true,
            ..Default::default()
        },
    );
    let stream = ctx.stream().unwrap();
    let width = 2304;
    let row = row_bytes(width);
    let word = size_of::<f32>();
    let mut rng = Rng::new(0x5eed_5731);
    // Bank mixes start inside the sealed rows; the attention mix reads all of them.
    let shapes = [
        [0usize, 0usize],
        [3, 7],
        [0, 12],
        [12, 12],
        [5, 9],
        [1, 2],
        [0, 1],
        [7, 8],
    ];
    let count = shapes.len();
    assert_eq!(count, MAX_MIX_JOBS);
    let state = ctx
        .allocate(row * shapes.iter().map(|[_, b]| b + 1).sum::<usize>())
        .unwrap();
    let queries = ctx.allocate(row * count).unwrap();
    let single = ctx.allocate(row * count).unwrap();
    let together = ctx.allocate(row * count).unwrap();
    let health = ctx.allocate(word).unwrap();
    let mut at = 0;
    let mut jobs = Vec::new();
    for (j, [first_block, blocks]) in shapes.into_iter().enumerate() {
        let rows = blocks + 1;
        state
            .write_at(
                &stream,
                at * row,
                &bytes(&job_state(&mut rng, width, blocks)),
            )
            .unwrap();
        queries
            .write_at(&stream, j * row, &bytes(&normal_row(&mut rng, width)))
            .unwrap();
        upload(&health, &stream, &[0.0]);
        let job = MixJob {
            state: state.range(at * row, rows * row).unwrap(),
            query: queries.range(j * row, row).unwrap(),
            out: single.range(j * row, row).unwrap(),
            health: health.range(0, word).unwrap(),
            addend: None,
            norm: None,
            first_block,
            blocks,
        };
        ctx.inkling_stream_mix(
            &stream,
            MixCall {
                state: job.state,
                query: job.query,
                out: job.out,
                health: job.health,
            },
            MixShape {
                first_block,
                blocks,
                width,
                eps: EPS,
            },
        )
        .unwrap();
        assert_eq!(download(&health, &stream, 0, 1)[0].to_bits(), 0, "job {j}");
        jobs.push(MixJob {
            out: together.range(j * row, row).unwrap(),
            ..job
        });
        at += rows;
    }
    ctx.inkling_stream_mix_jobs(&stream, &jobs, (width, EPS))
        .unwrap();
    assert_eq!(
        download(&single, &stream, 0, count * width),
        download(&together, &stream, 0, count * width),
        "one launch with a block per job differs from the single launches"
    );
    assert_eq!(download(&health, &stream, 0, 1)[0].to_bits(), 0);
    // Two jobs may not write the same output row; a job may not read another job's output.
    let mut clash = Vec::new();
    for job in &jobs {
        clash.push(MixJob {
            state: job.state,
            query: job.query,
            out: job.out,
            health: job.health,
            addend: None,
            norm: None,
            first_block: job.first_block,
            blocks: job.blocks,
        });
    }
    clash[1].out = clash[0].out;
    let error = ctx
        .inkling_stream_mix_jobs(&stream, &clash, (width, EPS))
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(error.contains("overlap"), "{error}");
    clash[1].out = together.range(row, row).unwrap();
    clash[0].query = clash[1].out;
    let error = ctx
        .inkling_stream_mix_jobs(&stream, &clash, (width, EPS))
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(error.contains("overlap"), "{error}");
}

#[test]
fn word_rows_zero_and_copy_whole_words_in_one_launch() {
    if !device_present("Inkling word rows") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let words = [1usize, 70, 3, 256];
    let total: usize = words.iter().sum();
    let source = ctx.allocate(total * 4).unwrap();
    let target = ctx.allocate(total * 4).unwrap();
    let mut at = 0;
    let mut zeros = Vec::new();
    let mut pairs = Vec::new();
    for &n in &words {
        let s = source.range(at * 4, n * 4).unwrap();
        let t = target.range(at * 4, n * 4).unwrap();
        zeros.push(t);
        pairs.push((s, t));
        at += n;
    }
    upload(&source, &stream, &vec![7.0; total]);
    upload(&target, &stream, &vec![9.0; total]);
    ctx.inkling_zero_words(&stream, &zeros).unwrap();
    assert!(
        download(&target, &stream, 0, total)
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    ctx.inkling_copy_words(&stream, &pairs).unwrap();
    assert!(
        download(&target, &stream, 0, total)
            .iter()
            .all(|v| *v == 7.0)
    );
    // A row that is not a whole number of words, a pair of different sizes, a pair that shares
    // bytes, an empty list and too many rows are refused before any launch.
    let error = ctx
        .inkling_zero_words(&stream, &[target.range(0, 6).unwrap()])
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(error.contains("whole number of words"), "{error}");
    let error = ctx
        .inkling_copy_words(
            &stream,
            &[(source.range(0, 8).unwrap(), target.range(0, 12).unwrap())],
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(error.contains("same whole number of words"), "{error}");
    let error = ctx
        .inkling_copy_words(
            &stream,
            &[(source.range(0, 8).unwrap(), source.range(4, 8).unwrap())],
        )
        .map_err(|error| format!("{error:?}"))
        .unwrap_err();
    assert!(error.contains("overlap"), "{error}");
    assert!(ctx.inkling_zero_words(&stream, &[]).is_err());
    let many = (0..=MAX_WORD_ROWS).map(|_| zeros[0]).collect::<Vec<_>>();
    assert!(ctx.inkling_zero_words(&stream, &many).is_err());
}

/// A job with an addend and a norm epilogue is the host's add, then mix, then weighted norm of
/// the mix output: the current row is bit-exact, the output and the normed row are within the
/// rounding of the single launches, and the health word stays clear. Two jobs sharing a health
/// word pass; a job whose normed row is its output, or one whose addend is a row another job of
/// the launch writes, is refused before any launch.
#[test]
fn mix_jobs_add_and_norm_match_the_host_within_rounding() {
    use crate::EngError;

    if !device_present("Inkling stream mix jobs with addend and norm") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    ctx.set_dense_group_policy(
        crate::kernels::cuda::resident::dense_gemv::DenseGroupPolicy {
            decode_glue: true,
            ..Default::default()
        },
    );
    let mut rng = Rng::new(0xadd0_57a7);
    let width = 2304;
    let row = row_bytes(width);
    let blocks = 6;
    let buffers = Buffers::new(&ctx, width);
    let twin = Buffers::new(&ctx, width);
    let extra = ctx.allocate(row * 3).unwrap();
    let [add, gamma, normed] = [0, 1, 2].map(|i| extra.range(i * row, row).unwrap());
    let addend: Vec<f32> = (0..width).map(|_| 0.25 * rng.normal() as f32).collect();
    let gains: Vec<f32> = (0..width)
        .map(|_| (0.75 + 0.25 * rng.normal().tanh()) as f32)
        .collect();
    let query = normal_row(&mut rng, width);
    let twin_query = normal_row(&mut rng, width);
    let state = job_state(&mut rng, width, blocks);
    let twin_state = job_state(&mut rng, width, blocks);
    let sealed: Vec<Vec<f32>> = state[width..]
        .chunks_exact(width)
        .map(<[f32]>::to_vec)
        .collect();
    let host_current: Vec<f32> = state[..width]
        .iter()
        .zip(&addend)
        .map(|(a, b)| a + b)
        .collect();
    let host_twin_current = twin_state[..width].to_vec();
    upload(&buffers.state, &stream, &state);
    upload(&twin.state, &stream, &twin_state);
    upload(&buffers.query, &stream, &query);
    upload(&twin.query, &stream, &twin_query);
    extra.write_at(&stream, 0, &bytes(&addend)).unwrap();
    extra.write_at(&stream, row, &bytes(&gains)).unwrap();
    buffers.clear_health(&stream);
    let health = buffers.health.range(0, size_of::<f32>()).unwrap();
    let job = MixJob {
        state: buffers.state.range(0, row * (blocks + 1)).unwrap(),
        query: buffers.query.range(0, row).unwrap(),
        out: buffers.out.range(0, row).unwrap(),
        health,
        addend: Some(add),
        norm: Some(MixNorm {
            gamma,
            normed,
            eps: EPS,
        }),
        first_block: 2,
        blocks,
    };
    let twin_job = MixJob {
        state: twin.state.range(0, row * (blocks + 1)).unwrap(),
        query: twin.query.range(0, row).unwrap(),
        out: twin.out.range(0, row).unwrap(),
        health,
        addend: None,
        norm: None,
        first_block: 0,
        blocks,
    };
    // A job whose normed row is its own output writes the row twice: refused.
    let mut clash = copy_job(&job);
    clash.norm = Some(MixNorm {
        gamma,
        normed: job.out,
        eps: EPS,
    });
    let error = ctx.inkling_stream_mix_jobs(&stream, &[clash], (width, EPS));
    assert!(matches!(error, Err(EngError::BadRequest(_))), "{error:?}");
    // A twin whose addend is a row the first job writes: refused before any launch.
    let mut clash = copy_job(&twin_job);
    clash.addend = Some(job.out);
    let error = ctx.inkling_stream_mix_jobs(&stream, &[copy_job(&job), clash], (width, EPS));
    assert!(matches!(error, Err(EngError::BadRequest(_))), "{error:?}");
    let refused = download(&buffers.out, &stream, 0, width);
    assert!(refused.iter().all(|v| v.to_bits() == 0), "nothing ran");
    // The first job with its addend and norm, next to the twin that shares the health word.
    ctx.inkling_stream_mix_jobs(&stream, &[job, twin_job], (width, EPS))
        .unwrap();
    let out = download(&buffers.out, &stream, 0, width);
    let current = download(&buffers.state, &stream, 0, width);
    let normed_row = download(&extra, &stream, 2 * row, width);
    let twin_out = download(&twin.out, &stream, 0, width);
    let expected = host::mix(&sealed, &host_current, 2, &query, EPS);
    let twin_sealed: Vec<Vec<f32>> = twin_state[width..]
        .chunks_exact(width)
        .map(<[f32]>::to_vec)
        .collect();
    let twin_expected = host::mix(&twin_sealed, &host_twin_current, 0, &twin_query, EPS);
    let mut standalone_state = state.clone();
    standalone_state[..width].copy_from_slice(&host_current);
    let separate = Buffers::new(&ctx, width)
        .mix(&ctx, &stream, &standalone_state, &query, 2, blocks, width)
        .unwrap();
    assert_eq!(out, separate, "fused output versus standalone mix");
    // Same sequential-host tolerance as mix_matches_the_host_algorithm_within_rounding:
    // device dots use a warp tree instead of left-to-right summation. The separate
    // launch above is the bit-exact oracle for changes introduced by fusion.
    let error = max_error(&out, &expected);
    assert!(error <= 1e-4, "mix max absolute error {error}");
    assert!(
        max_error(&twin_out, &twin_expected) <= 1e-4,
        "twin mix output"
    );
    let expected_norm: Vec<f32> = host::norm(&expected, EPS)
        .iter()
        .zip(&gains)
        .map(|(a, g)| a * g)
        .collect();
    for (i, (a, b)) in normed_row.iter().zip(&expected_norm).enumerate() {
        assert!(
            (a - b).abs() <= 4e-6 * (1.0 + b.abs()),
            "normed element {i}: {a} vs {b}"
        );
    }
    assert_eq!(current, host_current, "the add is bit-exact");
    assert_eq!(buffers.health(&stream).to_bits(), 0);
}

/// A copy of `job` over the same ranges.
fn copy_job<'a>(job: &MixJob<'a>) -> MixJob<'a> {
    MixJob {
        state: job.state,
        query: job.query,
        out: job.out,
        health: job.health,
        addend: job.addend,
        norm: job.norm,
        first_block: job.first_block,
        blocks: job.blocks,
    }
}
