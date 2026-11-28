use std::mem::size_of;

use super::super::super::conv;
use super::{
    AttendRow, AttendRows, KvPrepRows, Output, PrepMath, PrepRow, PrepRows, PrepShape, PrepWeights,
    QueryClipRows, Rings, ScoreRow, Scored, ScoredRows, attend_row, attend_rows, prep_row,
    prep_rows,
};
use crate::kernels::cuda::resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape};
use crate::kernels::cuda::resident::relative_attention::{
    HEAD_DIM, LogScale, RelativeAttentionShape, RelativeAttentionTile, RelativeKvFormat,
};
use crate::kernels::cuda::resident::relative_prep::{KvPrepShape, OutputConvShape};
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

struct VisitFixture {
    shape: RelativeAttentionShape,
    hidden: usize,
    math: PrepMath,
    table: Vec<f32>,
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
    k_taps: Vec<f32>,
    v_taps: Vec<f32>,
    o_weight: Vec<f32>,
    o_taps: Vec<f32>,
    // Raw projections per token: (q, raw_k, raw_v, relative, sinks, gates).
    tokens: Vec<[Vec<f32>; 6]>,
}

impl VisitFixture {
    fn new(capacity: usize, tokens: usize, log: Option<LogScale>, seed: u64) -> Self {
        let shape = RelativeAttentionShape {
            query_heads: SHAPE.queries,
            kv_heads: SHAPE.kv,
            relative_dims: SHAPE.dims,
            relative_extent: capacity,
            capacity,
        };
        let hidden = 64;
        let q = SHAPE.queries * HEAD_DIM;
        let kv = SHAPE.kv * HEAD_DIM;
        let rel = SHAPE.queries * SHAPE.dims;
        let taps = PrepShape::TAPS;
        let mut rng = Lcg(seed);
        let mut take = |n: usize, scale: f32, pos: bool| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let v = rng.next();
                    if pos { 0.8 + v.abs() } else { scale * v }
                })
                .collect()
        };
        let tokens = (0..tokens)
            .map(|_| {
                [
                    take(q, 2.5, false),
                    take(kv, 2.0, false),
                    take(kv, 2.0, false),
                    take(rel, 2.0, false),
                    take(SHAPE.queries, 1.0, false),
                    take(q, 1.5, false),
                ]
            })
            .collect();
        Self {
            shape,
            hidden,
            math: PrepMath {
                qk_limit: 2.0,
                rel_limit: 1.5,
                eps: 1e-6,
                log,
            },
            table: take(SHAPE.dims * capacity, 0.5, false),
            q_scale: take(q, 1.0, true),
            rel_scale: take(rel, 1.0, true),
            k_taps: take(kv * taps, 0.5, false),
            v_taps: take(kv * taps, 0.5, false),
            o_weight: take(hidden * q, 0.25, false),
            o_taps: take(hidden * taps, 0.5, false),
            tokens,
        }
    }
}

struct StageTrace {
    prep_q: Vec<Vec<f32>>,
    prep_k: Vec<Vec<f32>>,
    prep_v: Vec<Vec<f32>>,
    attended: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    k_ring: Vec<u8>,
    v_ring: Vec<u8>,
}

fn upload(ctx: &CudaContext, stream: &DeviceStream, values: &[f32]) -> DeviceBuffer {
    let buf = ctx.allocate(std::mem::size_of_val(values)).unwrap();
    buf.write_at(stream, 0, &bytes(values)).unwrap();
    buf
}

fn read_floats(stream: &DeviceStream, buf: &DeviceBuffer) -> Vec<f32> {
    let mut raw = vec![0u8; buf.bytes()];
    buf.read_at(stream, 0, &mut raw).unwrap();
    floats(&raw)
}

fn read_raw(stream: &DeviceStream, buf: &DeviceBuffer) -> Vec<u8> {
    let mut raw = vec![0u8; buf.bytes()];
    buf.read_at(stream, 0, &mut raw).unwrap();
    raw
}

/// Runs `prefix` one-row steps, then `count` steps either one row at a time (`chunk == None`) or
/// through `prep_rows` + `attend_rows` + `dense_gemv` + `conv::rows` (`Some(tiled)`).
fn run_visit(
    ctx: &CudaContext,
    stream: &DeviceStream,
    fix: &VisitFixture,
    prefix: usize,
    count: usize,
    chunk: Option<bool>,
) -> StageTrace {
    let q = SHAPE.queries * HEAD_DIM;
    let kv = SHAPE.kv * HEAD_DIM;
    let rel = SHAPE.queries * SHAPE.dims;
    let taps = PrepShape::TAPS;
    let past_rows = taps - 1;
    let h = fix.hidden;
    let word = size_of::<f32>();

    let table = upload(ctx, stream, &fix.table);
    let q_scale = upload(ctx, stream, &fix.q_scale);
    let rel_scale = upload(ctx, stream, &fix.rel_scale);
    let k_taps = upload(ctx, stream, &fix.k_taps);
    let v_taps = upload(ctx, stream, &fix.v_taps);
    let o_weight = upload(ctx, stream, &fix.o_weight);
    let o_taps = upload(ctx, stream, &fix.o_taps);

    let ring_bytes = fix.shape.capacity * RelativeKvFormat::F32.row_bytes(SHAPE.kv);
    let k_ring = upload(ctx, stream, &vec![0.0; ring_bytes / word]);
    let v_ring = upload(ctx, stream, &vec![0.0; ring_bytes / word]);
    let k_hist = upload(ctx, stream, &vec![0.0; past_rows * kv]);
    let v_hist = upload(ctx, stream, &vec![0.0; past_rows * kv]);
    let o_hist = upload(ctx, stream, &vec![0.0; past_rows * h]);

    let rings = Rings {
        shape: fix.shape,
        kv_format: RelativeKvFormat::F32,
        table: table.range(0, table.bytes()).unwrap(),
        k: k_ring.range(0, ring_bytes).unwrap(),
        v: v_ring.range(0, ring_bytes).unwrap(),
    };

    let q_buf = ctx.allocate(q * word).unwrap();
    let rel_buf = ctx.allocate(rel * word).unwrap();
    let raw_k_buf = ctx.allocate(kv * word).unwrap();
    let raw_v_buf = ctx.allocate(kv * word).unwrap();
    let k_buf = ctx.allocate(kv * word).unwrap();
    let v_buf = ctx.allocate(kv * word).unwrap();
    let sinks_buf = ctx.allocate(SHAPE.queries * word).unwrap();
    let gates_buf = ctx.allocate(q * word).unwrap();
    let attended_buf = ctx.allocate(q * word).unwrap();
    let projected_buf = ctx.allocate(h * word).unwrap();
    let out_buf = ctx.allocate(h * word).unwrap();
    let pos_buf = ctx.allocate(word).unwrap();
    let prep_status = ctx.allocate(SHAPE.direct_words()[14] * word).unwrap();
    let conv_health = ctx.allocate(word).unwrap();

    // Keep raw projected outputs oldest-first so the chunk arm can seed its oldest-first past.
    let mut o_raw_hist: Vec<Vec<f32>> = Vec::new();
    let mut trace = StageTrace {
        prep_q: Vec::new(),
        prep_k: Vec::new(),
        prep_v: Vec::new(),
        attended: Vec::new(),
        conv: Vec::new(),
        k_ring: Vec::new(),
        v_ring: Vec::new(),
    };

    let single_limit = if chunk.is_none() {
        prefix + count
    } else {
        prefix
    };
    for pos in 0..single_limit {
        let tok = &fix.tokens[pos];
        q_buf.write_at(stream, 0, &bytes(&tok[0])).unwrap();
        raw_k_buf.write_at(stream, 0, &bytes(&tok[1])).unwrap();
        raw_v_buf.write_at(stream, 0, &bytes(&tok[2])).unwrap();
        rel_buf.write_at(stream, 0, &bytes(&tok[3])).unwrap();
        sinks_buf.write_at(stream, 0, &bytes(&tok[4])).unwrap();
        gates_buf.write_at(stream, 0, &bytes(&tok[5])).unwrap();
        pos_buf
            .write_at(stream, 0, &(pos as i32).to_ne_bytes())
            .unwrap();
        prep_status
            .write_at(stream, 0, &vec![0u8; prep_status.bytes()])
            .unwrap();
        conv_health.write_at(stream, 0, &[0u8; 4]).unwrap();

        let scored = Scored {
            q: q_buf.range(0, q_buf.bytes()).unwrap(),
            relative: rel_buf.range(0, rel_buf.bytes()).unwrap(),
            sinks: sinks_buf.range(0, sinks_buf.bytes()).unwrap(),
            gates: gates_buf.range(0, gates_buf.bytes()).unwrap(),
            k: k_buf.range(0, k_buf.bytes()).unwrap(),
            v: v_buf.range(0, v_buf.bytes()).unwrap(),
        };
        prep_row(
            ctx,
            stream,
            PrepRow {
                shape: SHAPE,
                math: fix.math,
                rows: scored,
                raw_k: raw_k_buf.range(0, raw_k_buf.bytes()).unwrap(),
                raw_v: raw_v_buf.range(0, raw_v_buf.bytes()).unwrap(),
                weights: PrepWeights {
                    query_scale: q_scale.range(0, q_scale.bytes()).unwrap(),
                    relative_scale: rel_scale.range(0, rel_scale.bytes()).unwrap(),
                    key_taps: k_taps.range(0, k_taps.bytes()).unwrap(),
                    value_taps: v_taps.range(0, v_taps.bytes()).unwrap(),
                },
                key_history: k_hist.range(0, k_hist.bytes()).unwrap(),
                value_history: v_hist.range(0, v_hist.bytes()).unwrap(),
                status: prep_status.range(0, prep_status.bytes()).unwrap(),
                pos: pos_buf.range(0, word).unwrap(),
            },
        )
        .unwrap();
        attend_row(
            ctx,
            stream,
            AttendRow {
                rows: scored,
                rings,
                output: Output {
                    weight: o_weight.range(0, o_weight.bytes()).unwrap(),
                    dtype: DenseDtype::F32,
                    rows: h,
                    conv_taps: o_taps.range(0, o_taps.bytes()).unwrap(),
                    conv_history: o_hist.range(0, o_hist.bytes()).unwrap(),
                },
                pos: pos_buf.range(0, word).unwrap(),
                status: prep_status.range(0, word).unwrap(),
                conv_health: conv_health.range(0, word).unwrap(),
                attended: attended_buf.range(0, attended_buf.bytes()).unwrap(),
                projected: projected_buf.range(0, projected_buf.bytes()).unwrap(),
                out: out_buf.range(0, out_buf.bytes()).unwrap(),
            },
        )
        .unwrap();
        assert!(read_raw(stream, &prep_status).iter().all(|&b| b == 0));
        assert!(read_raw(stream, &conv_health).iter().all(|&b| b == 0));
        if pos < prefix {
            o_raw_hist.push(read_floats(stream, &projected_buf));
        } else {
            trace.prep_q.push(read_floats(stream, &q_buf));
            trace.prep_k.push(read_floats(stream, &k_buf));
            trace.prep_v.push(read_floats(stream, &v_buf));
            trace.attended.push(read_floats(stream, &attended_buf));
            trace.conv.push(read_floats(stream, &out_buf));
        }
    }

    if let Some(tiled) = chunk {
        // Pack sidecar rows: [q, raw_k, raw_v, relative, sinks, gates].
        let side_words = q + 2 * kv + rel + SHAPE.queries + q;
        let side_stride = side_words * word;
        let mut side_host = Vec::with_capacity(count * side_words);
        for tok in &fix.tokens[prefix..prefix + count] {
            for plane in tok {
                side_host.extend_from_slice(plane);
            }
        }
        let side_buf = upload(ctx, stream, &side_host);

        let mut scales_host = Vec::with_capacity(q + rel);
        scales_host.extend_from_slice(&fix.q_scale);
        scales_host.extend_from_slice(&fix.rel_scale);
        let scales_buf = upload(ctx, stream, &scales_host);

        // Right-aligned oldest-first raw K/V history (from prefix rows).
        let kv_hist_count = prefix.min(past_rows);
        let mut kv_past_host = Vec::with_capacity(2 * past_rows * kv);
        for plane in [1usize, 2] {
            kv_past_host.resize(kv_past_host.len() + (past_rows - kv_hist_count) * kv, 0.0);
            for tok in &fix.tokens[prefix - kv_hist_count..prefix] {
                kv_past_host.extend_from_slice(&tok[plane]);
            }
        }
        let kv_past_buf = upload(ctx, stream, &kv_past_host);

        let mut kv_taps_host = Vec::with_capacity(2 * taps * kv);
        kv_taps_host.extend_from_slice(&fix.k_taps);
        kv_taps_host.extend_from_slice(&fix.v_taps);
        let kv_taps_buf = upload(ctx, stream, &kv_taps_host);

        let out_stride_words = 2 * kv + 1;
        let out_stride = out_stride_words * word;
        let kv_out_buf = upload(ctx, stream, &vec![0.0; count * out_stride_words]);
        let status_buf = upload(ctx, stream, &vec![0.0; count]);

        prep_rows(
            ctx,
            stream,
            PrepRows {
                query: Some(QueryClipRows {
                    shape: fix.shape,
                    rows: count,
                    raw_kv_bytes: 2 * kv * word,
                    limits: [fix.math.qk_limit, fix.math.rel_limit],
                    log: fix.math.log.map(|l| (l, prefix)),
                    side: side_buf.range(0, side_buf.bytes()).unwrap(),
                    scales: scales_buf.range(0, scales_buf.bytes()).unwrap(),
                    status: status_buf.range(0, status_buf.bytes()).unwrap(),
                }),
                kv: Some(KvPrepRows {
                    shape: KvPrepShape {
                        rows: count,
                        kv_heads: SHAPE.kv,
                        history: kv_hist_count,
                        dynamic: false,
                        raw_stride: side_words,
                        raw: [q, q + kv],
                        out_stride: out_stride_words,
                        out: [0, kv],
                        positions: Some((2 * kv, prefix)),
                        limit: fix.math.qk_limit,
                        eps: fix.math.eps,
                    },
                    raw: side_buf.range(0, side_buf.bytes()).unwrap(),
                    history: kv_past_buf.range(0, kv_past_buf.bytes()).unwrap(),
                    taps: kv_taps_buf.range(0, kv_taps_buf.bytes()).unwrap(),
                    out: kv_out_buf.range(0, kv_out_buf.bytes()).unwrap(),
                    status: status_buf.range(0, status_buf.bytes()).unwrap(),
                }),
            },
        )
        .unwrap();

        let att_chunk = ctx.allocate(count * q * word).unwrap();
        let rel_off = (q + 2 * kv) * word;
        let sink_off = rel_off + rel * word;
        let gate_off = sink_off + SHAPE.queries * word;
        let scored = if tiled {
            let tile = RelativeAttentionTile {
                first: prefix,
                count,
                input_stride: out_stride,
                query_stride: side_stride,
                side_stride,
            };
            let b = tile
                .region_bytes_for(fix.shape, RelativeKvFormat::F32)
                .unwrap();
            ScoredRows::Tile {
                tile,
                row: Box::new(ScoreRow {
                    rows: Scored {
                        q: side_buf.range(0, b[4]).unwrap(),
                        relative: side_buf.range(rel_off, b[7]).unwrap(),
                        sinks: side_buf.range(sink_off, b[9]).unwrap(),
                        gates: side_buf.range(gate_off, b[10]).unwrap(),
                        k: kv_out_buf.range(0, b[5]).unwrap(),
                        v: kv_out_buf.range(kv * word, b[6]).unwrap(),
                    },
                    pos: kv_out_buf.range(2 * kv * word, b[11]).unwrap(),
                    status: status_buf.range(0, b[3]).unwrap(),
                    out: att_chunk.range(0, b[0]).unwrap(),
                }),
            }
        } else {
            let calls = (0..count)
                .map(|i| {
                    let s = i * side_stride;
                    let o = i * out_stride;
                    ScoreRow {
                        rows: Scored {
                            q: side_buf.range(s, q * word).unwrap(),
                            relative: side_buf.range(s + rel_off, rel * word).unwrap(),
                            sinks: side_buf.range(s + sink_off, SHAPE.queries * word).unwrap(),
                            gates: side_buf.range(s + gate_off, q * word).unwrap(),
                            k: kv_out_buf.range(o, kv * word).unwrap(),
                            v: kv_out_buf.range(o + kv * word, kv * word).unwrap(),
                        },
                        pos: kv_out_buf.range(o + 2 * kv * word, word).unwrap(),
                        status: status_buf.range(i * word, word).unwrap(),
                        out: att_chunk.range(i * q * word, q * word).unwrap(),
                    }
                })
                .collect();
            ScoredRows::PerRow(calls)
        };
        attend_rows(ctx, stream, AttendRows { rings, scored }).unwrap();

        let proj_chunk = ctx.allocate(count * h * word).unwrap();
        ctx.dense_gemv(
            stream,
            DenseGemvCall {
                shape: DenseGemvShape::packed(count, h, q),
                dtype: DenseDtype::F32,
                w: o_weight.range(0, o_weight.bytes()).unwrap(),
                x: att_chunk.range(0, att_chunk.bytes()).unwrap(),
                y: proj_chunk.range(0, proj_chunk.bytes()).unwrap(),
            },
        )
        .unwrap();

        let o_hist_count = o_raw_hist.len().min(past_rows);
        let mut o_past_host = vec![0.0f32; (past_rows - o_hist_count) * h];
        for row in &o_raw_hist[o_raw_hist.len() - o_hist_count..] {
            o_past_host.extend_from_slice(row);
        }
        let o_past_buf = upload(ctx, stream, &o_past_host);
        let conv_chunk = ctx.allocate(count * h * word).unwrap();
        conv::rows(
            ctx,
            stream,
            conv::Variant::Residual,
            OutputConvShape {
                rows: count,
                width: h,
                history: o_hist_count,
            },
            conv::Rows {
                input: proj_chunk.range(0, proj_chunk.bytes()).unwrap(),
                history: o_past_buf.range(0, o_past_buf.bytes()).unwrap(),
                taps: o_taps.range(0, o_taps.bytes()).unwrap(),
                out: conv_chunk.range(0, conv_chunk.bytes()).unwrap(),
                status: status_buf.range(0, status_buf.bytes()).unwrap(),
            },
        )
        .unwrap();
        assert!(read_raw(stream, &status_buf).iter().all(|&b| b == 0));

        let side_out = read_floats(stream, &side_buf);
        let kv_out = read_floats(stream, &kv_out_buf);
        let att_out = read_floats(stream, &att_chunk);
        let conv_out = read_floats(stream, &conv_chunk);
        for i in 0..count {
            trace.prep_q.push(side_out[i * side_words..][..q].to_vec());
            trace
                .prep_k
                .push(kv_out[i * out_stride_words..][..kv].to_vec());
            trace
                .prep_v
                .push(kv_out[i * out_stride_words + kv..][..kv].to_vec());
            trace.attended.push(att_out[i * q..][..q].to_vec());
            trace.conv.push(conv_out[i * h..][..h].to_vec());
        }
    }

    trace.k_ring = read_raw(stream, &k_ring);
    trace.v_ring = read_raw(stream, &v_ring);
    trace
}

/// 2 to 4 rows through the chunk prep, per-row attention and output convolution keep the exact
/// bits of repeated one-row steps across a local-window ring wrap.
#[test]
fn chunk_rows_match_one_row_steps_bit_for_bit_across_ring_wrap() {
    if !device_present("relative chunk ring wrap parity") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let fix = VisitFixture::new(8, 10, None, 19);
    for count in [2usize, 3, 4] {
        let single = run_visit(&ctx, &stream, &fix, 6, count, None);
        let chunk = run_visit(&ctx, &stream, &fix, 6, count, Some(false));
        for row in 0..count {
            let eq = |a: &[f32], b: &[f32]| {
                a.iter()
                    .map(|v| v.to_bits())
                    .eq(b.iter().map(|v| v.to_bits()))
            };
            assert!(
                eq(&chunk.prep_q[row], &single.prep_q[row]),
                "count {count} row {row}: prep_q"
            );
            assert!(
                eq(&chunk.prep_k[row], &single.prep_k[row]),
                "count {count} row {row}: prep_k"
            );
            assert!(
                eq(&chunk.prep_v[row], &single.prep_v[row]),
                "count {count} row {row}: prep_v"
            );
            assert!(
                eq(&chunk.attended[row], &single.attended[row]),
                "count {count} row {row}: attended"
            );
            assert!(
                eq(&chunk.conv[row], &single.conv[row]),
                "count {count} row {row}: conv"
            );
        }
        assert_eq!(chunk.k_ring, single.k_ring, "count {count}: k_ring");
        assert_eq!(chunk.v_ring, single.v_ring, "count {count}: v_ring");
    }
}

/// Reports where each relative-attention chunk form departs from the one-row step, stage by
/// stage, past the log-scale floor: per-row chunks stay bit-identical through every stage; the
/// query tile keeps prep bit-identical and departs only in the tiled score and the convolution
/// after it.
#[test]
fn card_reports_where_a_relative_chunk_leaves_the_exact_row() {
    if !device_present("relative chunk anatomy") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let log = Some(LogScale {
        floor: 4,
        alpha: 0.1,
    });
    let fix = VisitFixture::new(32, 22, log, 31);
    let prefix = 6;
    let reference = run_visit(&ctx, &stream, &fix, prefix, 16, None);
    let again = run_visit(&ctx, &stream, &fix, prefix, 1, None);
    let per2 = run_visit(&ctx, &stream, &fix, prefix, 2, Some(false));
    let per4 = run_visit(&ctx, &stream, &fix, prefix, 4, Some(false));
    let tile2 = run_visit(&ctx, &stream, &fix, prefix, 2, Some(true));
    let tile4 = run_visit(&ctx, &stream, &fix, prefix, 4, Some(true));
    let tile16 = run_visit(&ctx, &stream, &fix, prefix, 16, Some(true));

    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let rel = |a: &[f32], b: &[f32]| {
        let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        norm(&diff) / norm(b).max(1e-30)
    };
    type StagePick = fn(&StageTrace) -> &[Vec<f32>];
    let stages: [(&str, StagePick); 5] = [
        ("prep_q", |t| &t.prep_q),
        ("prep_k", |t| &t.prep_k),
        ("prep_v", |t| &t.prep_v),
        ("attend", |t| &t.attended),
        ("conv", |t| &t.conv),
    ];
    let arms: [(&str, &StageTrace); 6] = [
        ("1-again", &again),
        ("2-row", &per2),
        ("4-row", &per4),
        ("2-tile", &tile2),
        ("4-tile", &tile4),
        ("16-tile", &tile16),
    ];
    eprintln!(
        "stage     1-again      2-row      4-row     2-tile     4-tile    16-tile (max rel L2 over rows)"
    );
    for (stage, pick) in stages {
        let mut errs = [0.0f64; 6];
        for (idx, (_, arm)) in arms.iter().enumerate() {
            let got = pick(arm);
            let want = pick(&reference);
            errs[idx] = got
                .iter()
                .zip(want)
                .map(|(a, b)| {
                    let r = rel(a, b);
                    assert!(r.is_finite(), "{stage} arm {idx}: non-finite relative L2");
                    r
                })
                .fold(0.0f64, f64::max);
        }
        eprintln!(
            "{stage:<8} {:>9.2e}  {:>9.2e}  {:>9.2e}  {:>9.2e}  {:>9.2e}  {:>9.2e}",
            errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]
        );
        // 1-again, 2-row and 4-row must match one-row steps bit for bit at every stage.
        for (name, arm) in &arms[..3] {
            for (r, (a, b)) in pick(arm).iter().zip(pick(&reference)).enumerate() {
                assert!(
                    a.iter()
                        .map(|v| v.to_bits())
                        .eq(b.iter().map(|v| v.to_bits())),
                    "{stage} {name} row {r}"
                );
            }
        }
        // Query tiles must match bit for bit at every prep stage, and stay close after scoring.
        for (idx, (name, arm)) in arms[3..].iter().enumerate() {
            if stage.starts_with("prep_") {
                for (r, (a, b)) in pick(arm).iter().zip(pick(&reference)).enumerate() {
                    assert!(
                        a.iter()
                            .map(|v| v.to_bits())
                            .eq(b.iter().map(|v| v.to_bits())),
                        "{stage} {name} row {r}"
                    );
                }
            } else {
                assert!(
                    errs[3 + idx] < 5e-3,
                    "{stage} {name} relative L2 {} exceeded 5e-3",
                    errs[3 + idx]
                );
            }
        }
    }
}
