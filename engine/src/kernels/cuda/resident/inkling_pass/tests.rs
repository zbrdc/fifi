use std::mem::size_of;

use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present, testkit::Rng,
};

use super::{HEAD, MAX_BLOCKS, PassAddCall, PassMixCall, PassPrepCall, PassSealCall, TAPS};

/// The device sums are tree reductions where the host sums in storage order, and the device
/// `expf` is not the host libm's, so the frozen tolerance is relative with an absolute floor.
const TOLERANCE: f32 = 2e-5;
const WIDTH: usize = 2304;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn upload(buffer: &DeviceBuffer, stream: &DeviceStream, offset: usize, values: &[f32]) {
    buffer.write_at(stream, offset, &bytes(values)).unwrap();
}

fn download(buffer: &DeviceBuffer, stream: &DeviceStream, range: DeviceRange<'_>) -> Vec<f32> {
    let mut out = vec![0u8; range.bytes()];
    buffer.read_at(stream, range.offset(), &mut out).unwrap();
    out.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

fn normal_rows(rng: &mut Rng, words: usize) -> Vec<f32> {
    (0..words).map(|_| rng.normal() as f32).collect()
}

fn row(values: &[f32], index: usize, width: usize) -> &[f32] {
    &values[index * width..][..width]
}

/// `rms_norm_with_scale` of `family/inkling2/cpu_forward.rs`.
fn normalize(values: &[f32], eps: f32) -> Vec<f32> {
    let dot: f32 = values.iter().map(|v| v * v).sum();
    let scale = (dot / values.len() as f32 + eps).sqrt().recip();
    values.iter().map(|v| v * scale).collect()
}

/// `Stream::mix(first, query)` over sealed rows (oldest first) and the current row.
fn host_mix(
    sealed: &[Vec<f32>],
    current: &[f32],
    first: usize,
    query: &[f32],
    eps: f32,
) -> Vec<f32> {
    let scale = (query.len() as f32).sqrt().recip();
    let last = normalize(current, eps);
    let rows: Vec<&[f32]> = sealed[first..]
        .iter()
        .rev()
        .map(Vec::as_slice)
        .chain(std::iter::once(last.as_slice()))
        .collect();
    let scores: Vec<f32> = rows.iter().map(|r| dot(r, query) * scale).collect();
    let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
    let mass: f32 = weights.iter().sum();
    (0..query.len())
        .map(|c| {
            rows.iter()
                .zip(&weights)
                .map(|(r, w)| w * r[c])
                .sum::<f32>()
                / mass
        })
        .collect()
}

fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

fn assert_within(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            (a - b).abs() <= TOLERANCE * (1.0 + b.abs()),
            "{what}: {a} vs {b}"
        );
    }
}

struct State {
    sealed: DeviceBuffer,
    current: DeviceBuffer,
    next: DeviceBuffer,
    out: DeviceBuffer,
    query: DeviceBuffer,
    health: DeviceBuffer,
    rows: usize,
}

impl State {
    fn new(ctx: &CudaContext, stream: &DeviceStream, rows: usize) -> Self {
        let plane = WIDTH * size_of::<f32>() * rows;
        let state = Self {
            sealed: ctx.allocate(plane * MAX_BLOCKS).unwrap(),
            current: ctx.allocate(plane).unwrap(),
            // Two planes: the second half of each `next` row is the addend of the add step.
            next: ctx.allocate(2 * plane).unwrap(),
            out: ctx.allocate(2 * plane).unwrap(),
            query: ctx.allocate(WIDTH * size_of::<f32>()).unwrap(),
            health: ctx.allocate(rows * size_of::<f32>()).unwrap(),
            rows,
        };
        upload(&state.health, stream, 0, &vec![0.0; rows]);
        state
    }

    fn sealed(&self, blocks: usize) -> DeviceRange<'_> {
        self.sealed
            .range(0, blocks * self.rows * WIDTH * size_of::<f32>())
            .unwrap()
    }

    fn plane<'a>(&self, buffer: &'a DeviceBuffer) -> DeviceRange<'a> {
        buffer
            .range(0, self.rows * WIDTH * size_of::<f32>())
            .unwrap()
    }

    fn mix(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        blocks: usize,
        first: usize,
        query: &[f32],
        out_stride: usize,
    ) -> Vec<f32> {
        upload(&self.query, stream, 0, query);
        ctx.inkling_pass_mix(
            stream,
            PassMixCall {
                sealed: self.sealed(blocks),
                current: self.plane(&self.current),
                query: self.query.range(0, WIDTH * size_of::<f32>()).unwrap(),
                out: self.out.range(0, self.out.bytes()).unwrap(),
                health: self.health.range(0, self.rows * size_of::<f32>()).unwrap(),
            },
            [self.rows, blocks, first, WIDTH, out_stride],
            1e-6,
        )
        .unwrap();
        download(
            &self.out,
            stream,
            self.out.range(0, self.out.bytes()).unwrap(),
        )
    }
}

fn assert_healthy(state: &State, stream: &DeviceStream, what: &str) {
    let health = download(
        &state.health,
        stream,
        state
            .health
            .range(0, state.rows * size_of::<f32>())
            .unwrap(),
    );
    for (r, value) in health.iter().enumerate() {
        assert!(value.is_finite(), "{what}: health of row {r}");
    }
}

/// Mix, seal and add follow the host stream: the mixed rows, the sealed row and the current
/// row after each step agree with the host within the frozen tolerance, for every block count
/// the pass uses and every `first_block`.
#[test]
fn mix_seal_and_add_match_the_host_stream() {
    if !device_present("Inkling pass stream") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_9a55);
    for rows in [1usize, 5] {
        let state = State::new(&ctx, &stream, rows);
        let mut current: Vec<Vec<f32>> = (0..rows).map(|_| normal_rows(&mut rng, WIDTH)).collect();
        upload(&state.current, &stream, 0, &current.concat());
        // The prelude of a core pass: one sealed row (the first block) and the current row.
        let mut sealed: Vec<Vec<Vec<f32>>> = current
            .iter()
            .map(|current| vec![normalize(current, 1e-6)])
            .collect();
        upload(&state.sealed, &stream, 0, &sealed.concat().concat());
        for blocks in 1..4usize {
            assert_healthy(&state, &stream, "start");
            for first in 0..=blocks {
                let query = normal_rows(&mut rng, WIDTH);
                // Second half of each output row stays untouched: out_stride is 2 * WIDTH.
                let out = state.mix(&ctx, &stream, blocks, first, &query, 2 * WIDTH);
                for r in 0..rows {
                    let expected = host_mix(&sealed[r], &current[r], first, &query, 1e-6);
                    assert_within(
                        row(&out, 2 * r, WIDTH),
                        &expected,
                        &format!("rows {rows}, blocks {blocks}, first {first}, row {r}"),
                    );
                }
                assert_healthy(&state, &stream, "mix");
            }
            let next: Vec<Vec<f32>> = (0..rows)
                .map(|_| normal_rows(&mut rng, 2 * WIDTH))
                .collect();
            // The seal takes the packed first half of each `next` row.
            let first: Vec<f32> = next
                .iter()
                .flat_map(|row| row[..WIDTH].iter().copied())
                .collect();
            upload(&state.next, &stream, 0, &first);
            ctx.inkling_pass_seal(
                &stream,
                PassSealCall {
                    sealed: state.sealed(blocks + 1),
                    current: state.plane(&state.current),
                    next: state.plane(&state.next),
                    health: state.health.range(0, rows * size_of::<f32>()).unwrap(),
                },
                [rows, blocks, WIDTH],
                1e-6,
            )
            .unwrap();
            for r in 0..rows {
                sealed[r].push(normalize(&current[r], 1e-6));
                current[r].copy_from_slice(&next[r][..WIDTH]);
            }
            let device = download(&state.current, &stream, state.plane(&state.current));
            let sealed_device = download(&state.sealed, &stream, state.sealed(blocks + 1));
            for r in 0..rows {
                assert_within(
                    row(&device, r, WIDTH),
                    &current[r],
                    &format!("seal blocks {blocks}, row {r}: current"),
                );
                assert_eq!(sealed[r].len(), blocks + 1);
                for (b, sealed_row) in sealed[r].iter().enumerate() {
                    assert_within(
                        row(&sealed_device, b * rows + r, WIDTH),
                        sealed_row,
                        &format!("seal blocks {blocks}, row {r}: sealed {b}"),
                    );
                }
            }
            assert_healthy(&state, &stream, "seal");
            let addend: Vec<f32> = (0..rows)
                .flat_map(|r| row(&next[r], 1, WIDTH))
                .copied()
                .collect();
            upload(&state.next, &stream, 0, &addend);
            ctx.inkling_pass_add(
                &stream,
                PassAddCall {
                    current: state.plane(&state.current),
                    row: state.plane(&state.next),
                    health: state.health.range(0, rows * size_of::<f32>()).unwrap(),
                },
                [rows, WIDTH],
            )
            .unwrap();
            let device = download(&state.current, &stream, state.plane(&state.current));
            for r in 0..rows {
                let expected: Vec<f32> = current[r]
                    .iter()
                    .zip(&next[r][WIDTH..])
                    .map(|(a, b)| a + b)
                    .collect();
                assert_within(
                    row(&device, r, WIDTH),
                    &expected,
                    &format!("add blocks {blocks}, row {r}"),
                );
                current[r] = expected;
            }
            assert_healthy(&state, &stream, "add");
        }
    }
}

/// `CausalConv::apply`, then the host's per-head clip of K and RMS norm of V
/// (`prepare_attention_row`). `history` is oldest first; zero rows stand for missing rows.
struct PrepHost {
    history: [Vec<Vec<f32>>; 2],
}

impl PrepHost {
    #[allow(clippy::too_many_arguments)]
    fn convolve(
        &mut self,
        plane: usize,
        raw: &[Vec<f32>],
        kernel: &[f32],
        dynamic: Option<&[Vec<f32>]>,
        channels: usize,
        qk_limit: f32,
        eps: f32,
    ) -> Vec<Vec<f32>> {
        let taps = TAPS;
        // `CausalConv::apply`: dynamic taps count as a second kernel.
        let kernels = if dynamic.is_some() { 2 } else { 1 };
        let norm = ((kernels * taps + 1) as f32).sqrt().recip();
        let mut rows: Vec<(Vec<f32>, bool)> = Vec::new();
        for (r, input) in raw.iter().enumerate() {
            let mut output = vec![0.0; channels];
            let mut sources: Vec<&[f32]> = vec![input.as_slice()];
            for lag in 1..taps {
                if r >= lag {
                    sources.push(&raw[r - lag]);
                } else if self.history[plane].len() >= lag {
                    // Rows inserted so far sit in front of the pre-sequence history: the row
                    // `lag` steps back keeps its index while `r` grows.
                    sources.push(&self.history[plane][lag - 1]);
                }
            }
            for (lag, source) in sources.iter().enumerate() {
                for channel in 0..channels {
                    let weight = match dynamic {
                        None => kernel[channel * taps + taps - 1 - lag],
                        Some(coefficients) => {
                            kernel[lag * channels + channel]
                                + coefficients[r][(channel / HEAD) * taps + lag]
                        }
                    };
                    output[channel] += weight * source[channel];
                }
            }
            for (result, x) in output.iter_mut().zip(input) {
                *result = (*result + x) * norm;
            }
            let value_plane = plane == 1;
            for head in output.as_chunks_mut::<HEAD>().0 {
                let squares: f32 = head.iter().map(|v| v * v).sum();
                let scale = if value_plane {
                    (squares / HEAD as f32 + eps).sqrt().recip()
                } else {
                    (qk_limit / (squares.sqrt() + 1e-6)).min(1.0)
                };
                for value in head {
                    *value *= scale;
                }
            }
            rows.push((output, false));
            self.history[plane].insert(0, input.clone());
            self.history[plane].truncate(taps - 1);
        }
        rows.into_iter().map(|(row, _)| row).collect()
    }
}

/// The prepared K and V rows equal the host's for both attention kinds, with a history shorter
/// than three rows at the start of the sequence, read from the strided sidecar layout the
/// attention tile uses.
#[test]
fn prep_matches_the_host_for_static_and_dynamic_taps() {
    if !device_present("Inkling pass prep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng::new(0x5eed_9a77);
    let (rows, kv_heads, relative) = (37usize, 8usize, 24usize * 16);
    let channels = kv_heads * HEAD;
    let q_side = 24 * HEAD;
    let side_stride = q_side + 2 * channels + relative + 24 + 24 * HEAD;
    let (key_at, value_at) = (q_side, q_side + channels);
    for dynamic in [false, true] {
        for history_rows in [0usize, 2, 3] {
            let raw: Vec<Vec<f32>> = (0..rows)
                .map(|_| normal_rows(&mut rng, 2 * channels))
                .collect();
            let mut host = PrepHost {
                history: std::array::from_fn(|_| {
                    (0..history_rows)
                        .map(|_| normal_rows(&mut rng, channels))
                        .collect()
                }),
            };
            let kernels: [Vec<f32>; 2] =
                std::array::from_fn(|_| normal_rows(&mut rng, channels * TAPS));
            let coefficients: Vec<Vec<f32>> = (0..rows)
                .map(|_| normal_rows(&mut rng, 2 * kv_heads * TAPS))
                .collect();
            let side = vec![0.0; rows * side_stride];
            let word = size_of::<f32>();
            let raw_buffer = ctx.upload(&bytes(&raw.concat())).unwrap();
            let side_buffer = ctx.upload(&bytes(&side)).unwrap();
            let history: Vec<DeviceBuffer> = host
                .history
                .iter()
                .map(|rows| {
                    // Right-aligned, oldest first: the device keeps zero rows where a fresh
                    // sequence lacks history.
                    let mut plane = vec![0.0; (TAPS - 1 - rows.len()) * channels];
                    for row in rows.iter().rev() {
                        plane.extend_from_slice(row);
                    }
                    ctx.upload(&bytes(&plane)).unwrap()
                })
                .collect();
            let kernel: Vec<DeviceBuffer> = kernels
                .iter()
                .map(|k| ctx.upload(&bytes(k)).unwrap())
                .collect();
            let dynamic_buffer = ctx.upload(&bytes(&coefficients.concat())).unwrap();
            let status = ctx.upload(&vec![0u8; rows * size_of::<u32>()]).unwrap();
            ctx.inkling_pass_prep(
                &stream,
                PassPrepCall {
                    raw: raw_buffer.range(0, raw_buffer.bytes()).unwrap(),
                    // The exact extent the entry reads: the last row stops after its V words.
                    side: side_buffer
                        .range(0, ((rows - 1) * side_stride + value_at + channels) * word)
                        .unwrap(),
                    history: [
                        history[0].range(0, history[0].bytes()).unwrap(),
                        history[1].range(0, history[1].bytes()).unwrap(),
                    ],
                    kernel: [
                        kernel[0].range(0, kernel[0].bytes()).unwrap(),
                        kernel[1].range(0, kernel[1].bytes()).unwrap(),
                    ],
                    dynamic: dynamic_buffer
                        .range(0, rows * 2 * kv_heads * TAPS * word)
                        .unwrap(),
                    status: status.range(0, status.bytes()).unwrap(),
                },
                rows,
                kv_heads,
                [
                    2 * channels,
                    side_stride,
                    key_at,
                    value_at,
                    2 * kv_heads * TAPS,
                ],
                (4.0, 1e-5),
                dynamic,
            )
            .unwrap();
            let device = download(
                &side_buffer,
                &stream,
                side_buffer.range(0, side_buffer.bytes()).unwrap(),
            );
            let mut status_bytes = vec![0u8; rows * size_of::<u32>()];
            status.read_at(&stream, 0, &mut status_bytes).unwrap();
            assert!(
                status_bytes.iter().all(|&b| b == 0),
                "dynamic {dynamic}, history {history_rows}: status"
            );
            // Each row of the dynamic plane holds the K coefficients, then the V coefficients.
            let dynamic_k = dynamic.then(|| {
                coefficients
                    .iter()
                    .map(|row| row[..kv_heads * TAPS].to_vec())
                    .collect::<Vec<_>>()
            });
            let expected_k = host.convolve(
                0,
                &raw[..],
                &kernels[0],
                dynamic_k.as_deref(),
                channels,
                4.0,
                1e-5,
            );
            let dynamic_arg = dynamic.then(|| {
                coefficients
                    .iter()
                    .map(|row| row[kv_heads * TAPS..].to_vec())
                    .collect::<Vec<_>>()
            });
            let expected_v = host.convolve(
                1,
                raw.iter()
                    .map(|row| row[channels..].to_vec())
                    .collect::<Vec<_>>()
                    .as_slice(),
                &kernels[1],
                dynamic_arg.as_deref(),
                channels,
                4.0,
                1e-5,
            );
            for r in 0..rows {
                let mut full = vec![0.0; side_stride];
                full[key_at..key_at + channels].copy_from_slice(&expected_k[r]);
                full[value_at..value_at + channels].copy_from_slice(&expected_v[r]);
                assert_within(
                    row(&device, r, side_stride),
                    &full,
                    &format!("dynamic {dynamic}, history {history_rows}, row {r}"),
                );
            }
            drop((
                raw_buffer,
                side_buffer,
                dynamic_buffer,
                status,
                history,
                kernel,
            ));
        }
    }
}

/// Every refusal names the bad range before any launch: an overlapping output, a short sealed
/// plane, a block count outside the contract and a misaligned status plane.
#[test]
fn the_pass_kernels_refuse_bad_calls_before_launching() {
    if !device_present("Inkling pass rejections") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let rows = 3;
    let state = State::new(&ctx, &stream, rows);
    let mix = |sealed: DeviceRange<'_>, out: DeviceRange<'_>, blocks: usize| {
        ctx.inkling_pass_mix(
            &stream,
            PassMixCall {
                sealed,
                current: state.plane(&state.current),
                query: state.query.range(0, WIDTH * size_of::<f32>()).unwrap(),
                out,
                health: state.health.range(0, rows * size_of::<f32>()).unwrap(),
            },
            [rows, blocks, 0, WIDTH, WIDTH],
            1e-6,
        )
    };
    let error = mix(state.sealed(2), state.plane(&state.current), 2)
        .unwrap_err()
        .describe();
    assert!(error.contains("overlap"), "{error}");
    let error = mix(state.sealed(1), state.plane(&state.out), 2)
        .unwrap_err()
        .describe();
    assert!(error.contains("Inkling pass mix sealed"), "{error}");
    let error = mix(state.sealed(1), state.plane(&state.out), MAX_BLOCKS + 1)
        .unwrap_err()
        .describe();
    assert!(error.contains("sealed blocks"), "{error}");
    let status = ctx.allocate(rows * size_of::<u32>() + 4).unwrap();
    let bad_status = status.range(2, rows * size_of::<u32>()).unwrap();
    let buffer = ctx.allocate(4 * size_of::<f32>()).unwrap();
    let range = buffer.range(0, 4 * size_of::<f32>()).unwrap();
    let error = ctx
        .inkling_pass_prep(
            &stream,
            PassPrepCall {
                raw: range,
                side: state.plane(&state.out),
                history: [range, range],
                kernel: [range, range],
                dynamic: range,
                status: bad_status,
            },
            rows,
            1,
            [2 * HEAD, 6 * HEAD, 0, HEAD, 2 * TAPS],
            (4.0, 1e-5),
            false,
        )
        .unwrap_err()
        .describe();
    assert!(error.contains("Inkling pass prep raw"), "{error}");
}

/// The tails of a strided source (the raw K/V plane: row r holds K then V) and of a source
/// shorter than the taps, where the older rows come from the previous block's plane.
#[test]
fn tails_take_the_last_rows_of_a_strided_source_and_the_older_plane() {
    if !device_present("Inkling pass tails") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let width = 64;
    let word = size_of::<f32>();
    let key = |r: usize| vec![r as f32; width];
    let value = |r: usize| vec![(r + 100) as f32; width];
    let strided = |rows: usize| {
        (0..rows)
            .flat_map(|r| [key(r), value(r)])
            .flatten()
            .collect::<Vec<f32>>()
    };
    for (rows, past) in [(8, 0), (5, 2), (2, 2)] {
        // Source row r is [r | r + 100]; the previous plane holds rows 90, 91, 92 oldest first.
        let source = ctx.upload(&bytes(&strided(rows))).unwrap();
        // The previous plane is packed: one row of `width` words per history row.
        let old = ctx
            .upload(&bytes(&(90..93).flat_map(key).collect::<Vec<f32>>()))
            .unwrap();
        let out = ctx.allocate((TAPS - 1) * width * word).unwrap();
        ctx.inkling_pass_tails(
            &stream,
            [
                plane(&source, rows * 2 * width),
                plane(&old, (TAPS - 1) * width),
                plane(&out, (TAPS - 1) * width),
            ],
            [rows, 2 * width, width, past],
        )
        .unwrap();
        let got = download(&out, &stream, plane(&out, (TAPS - 1) * width));
        // Host `CausalConv` rows newest first are 92 (or the last row), then the row before, ...
        let mut expected = Vec::new();
        for i in 0..TAPS - 1 {
            let j = rows + past - TAPS + 1 + i;
            expected.extend(if j < past {
                key(90 + TAPS - 1 - past + j)
            } else {
                key(j - past)
            });
        }
        assert_eq!(got, expected, "{rows} rows, {past} past");
    }
}

fn plane(buffer: &DeviceBuffer, words: usize) -> DeviceRange<'_> {
    buffer.range(0, words * size_of::<f32>()).unwrap()
}
