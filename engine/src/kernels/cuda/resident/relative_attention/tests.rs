use super::*;
use crate::kernels::cuda::resident::step_params::{POS_MAX, STATUS_INVALID_STEP_STATE};
use crate::kernels::cuda::testkit::{
    Rng, device_present, floats, read_bytes, upload_f32, upload_filled,
};
use crate::kernels::cuda::{CapturePlan, DeviceBuffer};

const SHAPE: RelativeAttentionShape = RelativeAttentionShape {
    query_heads: 24,
    kv_heads: 8,
    relative_dims: 16,
    // A ring past the block's bias-table threshold (64 rows) and an extent short of it, so the
    // positions cover both bias paths and distances with and without a bias.
    relative_extent: 70,
    capacity: 83,
};
const QUERY: usize = SHAPE.query_heads * HEAD_DIM;
const KV: usize = SHAPE.kv_heads * HEAD_DIM;

fn pattern(len: usize, seed: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed as u64);
    (0..len)
        .map(|_| (rng.below(1031) as f32 - 515.0) / 431.0)
        .collect()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().copied().flat_map(f32::to_le_bytes).collect()
}

struct Fixture {
    q: Vec<f32>,
    relative: Vec<f32>,
    table: Vec<f32>,
    sinks: Vec<f32>,
    gates: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl Fixture {
    fn new(steps: usize) -> Self {
        Self {
            q: pattern(QUERY, 1),
            relative: pattern(SHAPE.query_heads * SHAPE.relative_dims, 2),
            table: pattern(SHAPE.relative_dims * SHAPE.relative_extent, 3),
            sinks: pattern(SHAPE.query_heads, 4)
                .into_iter()
                .map(|s| s * 8.0)
                .collect(),
            gates: pattern(QUERY, 5).into_iter().map(|x| x * 4.0).collect(),
            keys: (0..steps).map(|p| pattern(KV, p * 2 + 6)).collect(),
            values: (0..steps).map(|p| pattern(KV, p * 2 + 7)).collect(),
        }
    }

    fn reference(&self, pos: usize) -> Vec<f32> {
        let row = Row {
            q: &self.q,
            relative: &self.relative,
            sinks: &self.sinks,
            gates: &self.gates,
        };
        reference(row, &self.keys, &self.values, &self.table, pos)
    }
}

/// The per-row inputs of one query: Q, the relative coefficients, the sinks and the gates.
#[derive(Clone, Copy)]
struct Row<'a> {
    q: &'a [f32],
    relative: &'a [f32],
    sinks: &'a [f32],
    gates: &'a [f32],
}

/// Newest-first FP64 reference with the unscaled relative logits and zero-value sink in the
/// same positions as the CPU family; `keys` and `values` are indexed by position.
fn reference(
    row: Row<'_>,
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    table: &[f32],
    pos: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; QUERY];
    let visible = pos.saturating_sub(SHAPE.capacity - 1)..=pos;
    let (rows, _) = out.as_chunks_mut::<HEAD_DIM>();
    for (head, slot) in rows.iter_mut().enumerate() {
        let q = head * HEAD_DIM;
        let kv = head / (SHAPE.query_heads / SHAPE.kv_heads) * HEAD_DIM;
        let scores = scores(row, keys, table, pos, head);
        let sink = f64::from(row.sinks[head]);
        let peak = scores.iter().copied().fold(sink, f64::max);
        let masses: Vec<_> = scores.iter().map(|score| (score - peak).exp()).collect();
        let normalizer = (sink - peak).exp() + masses.iter().sum::<f64>();
        for (d, result) in slot.iter_mut().enumerate() {
            let value: f64 = visible
                .clone()
                .rev()
                .zip(&masses)
                .map(|(token, mass)| mass * f64::from(values[token][kv + d]))
                .sum();
            let gate = 1.0 / (1.0 + (-f64::from(row.gates[q + d])).exp());
            *result = (value / normalizer * gate) as f32;
        }
    }
    out
}

fn scores(row: Row<'_>, keys: &[Vec<f32>], table: &[f32], pos: usize, head: usize) -> Vec<f64> {
    let q = head * HEAD_DIM;
    let rel = head * SHAPE.relative_dims;
    let kv = head / (SHAPE.query_heads / SHAPE.kv_heads) * HEAD_DIM;
    (0..=(pos.min(SHAPE.capacity - 1)))
        .map(|distance| {
            let qk: f64 = (0..HEAD_DIM)
                .map(|d| f64::from(row.q[q + d]) * f64::from(keys[pos - distance][kv + d]))
                .sum();
            let mut bias = 0.0;
            if distance < SHAPE.relative_extent {
                for c in 0..SHAPE.relative_dims {
                    bias += f64::from(row.relative[rel + c])
                        * f64::from(table[c * SHAPE.relative_extent + distance]);
                }
            }
            qk / (HEAD_DIM as f64).sqrt() + bias
        })
        .collect()
}

struct Buffers {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    relative: DeviceBuffer,
    table: DeviceBuffer,
    sinks: DeviceBuffer,
    gates: DeviceBuffer,
    keys: DeviceBuffer,
    values: DeviceBuffer,
    pos: DeviceBuffer,
    status: DeviceBuffer,
    out: DeviceBuffer,
}

impl Buffers {
    fn new(ctx: &CudaContext, fixture: &Fixture) -> Self {
        let allocate = |bytes| upload_filled(ctx, 0xff, bytes);
        Self {
            q: upload_f32(ctx, &fixture.q),
            k: allocate(KV * 4),
            v: allocate(KV * 4),
            relative: upload_f32(ctx, &fixture.relative),
            table: upload_f32(ctx, &fixture.table),
            sinks: upload_f32(ctx, &fixture.sinks),
            gates: upload_f32(ctx, &fixture.gates),
            keys: allocate(SHAPE.capacity * KV * 4),
            values: allocate(SHAPE.capacity * KV * 4),
            pos: allocate(4),
            status: upload_filled(ctx, 0, 4),
            out: allocate(QUERY * 4),
        }
    }

    fn pins(&self) -> [&DeviceBuffer; RANGES] {
        [
            &self.q,
            &self.k,
            &self.v,
            &self.relative,
            &self.table,
            &self.sinks,
            &self.gates,
            &self.keys,
            &self.values,
            &self.pos,
            &self.status,
            &self.out,
        ]
    }

    fn call(&self) -> RelativeAttentionCall<'_> {
        RelativeAttentionCall {
            shape: SHAPE,
            kv_format: RelativeKvFormat::F32,
            q: whole(&self.q),
            k: whole(&self.k),
            v: whole(&self.v),
            relative: whole(&self.relative),
            table: whole(&self.table),
            sinks: whole(&self.sinks),
            gates: whole(&self.gates),
            k_cache: whole(&self.keys),
            v_cache: whole(&self.values),
            pos: whole(&self.pos),
            status: whole(&self.status),
            out: whole(&self.out),
        }
    }

    fn set_position(&self, stream: &DeviceStream, fixture: &Fixture, pos: usize) {
        self.k
            .write_at(stream, 0, &bytes(&fixture.keys[pos]))
            .unwrap();
        self.v
            .write_at(stream, 0, &bytes(&fixture.values[pos]))
            .unwrap();
        self.pos
            .write_at(stream, 0, &(pos as i32).to_le_bytes())
            .unwrap();
    }
}

fn whole(buf: &DeviceBuffer) -> DeviceRange<'_> {
    buf.range(0, buf.bytes()).expect("whole device buffer")
}

fn compare(actual: &[u8], expected: &[f32], pos: usize) {
    let actual = floats(actual);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 2e-5,
            "position {pos}, element {i}: expected {e}, got {a}"
        );
    }
}

fn assert_refuses(ctx: &CudaContext, stream: &DeviceStream, buf: &Buffers) {
    let mut call = buf.call();
    call.k = buf.keys.range(0, KV * 4).unwrap();
    let Err(EngError::BadRequest(message)) = ctx.relative_attention(stream, call) else {
        panic!("key/cache overlap must be refused");
    };
    assert!(message.contains("overlap"), "{message}");

    let mut call = buf.call();
    call.relative = buf.relative.range(0, buf.relative.bytes() - 4).unwrap();
    let Err(EngError::BadRequest(message)) = ctx.relative_attention(stream, call) else {
        panic!("short relative coefficients must be refused");
    };
    assert!(message.contains("needs"), "{message}");
}

#[test]
fn geometry_rejects_invalid_shapes() {
    assert!(SHAPE.region_bytes().is_some());
    for shape in [
        RelativeAttentionShape {
            kv_heads: 0,
            ..SHAPE
        },
        RelativeAttentionShape {
            query_heads: 23,
            ..SHAPE
        },
        RelativeAttentionShape {
            relative_dims: 33,
            ..SHAPE
        },
        RelativeAttentionShape {
            relative_extent: 0,
            ..SHAPE
        },
        RelativeAttentionShape {
            capacity: usize::MAX,
            ..SHAPE
        },
    ] {
        assert!(shape.region_bytes().is_none());
    }
}

#[test]
fn query_tile_validates_strided_bounds_and_positions() {
    let tile = RelativeAttentionTile {
        first: POS_MAX as usize - 1,
        count: 2,
        input_stride: (2 * KV + 1) * 4,
        query_stride: (2 * QUERY + SHAPE.query_heads * (SHAPE.relative_dims + 1)) * 4,
        side_stride: (2 * QUERY + SHAPE.query_heads * (SHAPE.relative_dims + 1)) * 4,
    };
    let row = SHAPE.region_bytes().unwrap();
    let extents = tile.region_bytes(SHAPE).unwrap();
    assert_eq!(extents[0], 2 * row[0]);
    assert_eq!(extents[1..3], row[1..3]);
    assert_eq!(extents[3], 8);
    assert!(tile.input_stride < row[4]); // host rows carry only K/V/position
    assert_eq!(extents[4], row[4] + tile.query_stride);
    assert_eq!(extents[5], row[5] + tile.input_stride);
    assert_eq!(extents[6], row[6] + tile.input_stride);
    assert_eq!(extents[7], row[7] + tile.query_stride);
    assert_eq!(extents[8], row[8]);
    assert_eq!(extents[9], row[9] + tile.side_stride);
    assert_eq!(extents[10], row[10] + tile.side_stride);
    assert_eq!(extents[11], 4 + tile.input_stride);
    for invalid in [
        RelativeAttentionTile { count: 1, ..tile },
        RelativeAttentionTile {
            count: QUERY_TILE_LIMIT + 1,
            ..tile
        },
        RelativeAttentionTile {
            first: POS_MAX as usize,
            ..tile
        },
        RelativeAttentionTile {
            first: usize::MAX,
            ..tile
        },
        RelativeAttentionTile {
            input_stride: 4,
            ..tile
        },
        RelativeAttentionTile {
            query_stride: row[4] - 4,
            ..tile
        },
        RelativeAttentionTile {
            side_stride: tile.side_stride + 1,
            ..tile
        },
        RelativeAttentionTile {
            input_stride: MAX_STRIDE + 4,
            ..tile
        },
    ] {
        assert!(invalid.region_bytes(SHAPE).is_none());
    }
}

#[test]
fn rolling_relative_attention_capture_matches_cpu() {
    if !device_present("rolling relative attention") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let steps = SHAPE.capacity * 2 + 3;
    let fixture = Fixture::new(steps);
    let eager = Buffers::new(&ctx, &fixture);
    let replay = Buffers::new(&ctx, &fixture);
    assert_refuses(&ctx, &stream, &eager);

    eager.set_position(&stream, &fixture, 0);
    ctx.relative_attention(&stream, eager.call()).unwrap();
    stream.synchronize().unwrap();
    let pins = replay.pins();
    let (graph, ()) = ctx
        .capture(
            &stream,
            CapturePlan {
                pins: &pins,
                holds: Vec::new(),
            },
            || ctx.relative_attention(&stream, replay.call()),
        )
        .unwrap();
    assert_eq!(graph.nodes(), 1);

    for pos in 0..steps {
        eager.set_position(&stream, &fixture, pos);
        replay.set_position(&stream, &fixture, pos);
        ctx.relative_attention(&stream, eager.call()).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let captured = read_bytes(&ctx, &replay.out, QUERY * 4);
        let direct = read_bytes(&ctx, &eager.out, QUERY * 4);
        assert_eq!(captured, direct, "position {pos}");
        compare(&captured, &fixture.reference(pos), pos);
    }

    assert_ring(&ctx, &replay.keys, &fixture.keys);
    assert_ring(&ctx, &replay.values, &fixture.values);
    assert_invalid_positions(&ctx, &stream, &replay, || graph.launch(&stream).unwrap());
}

fn assert_ring(ctx: &CudaContext, ring: &DeviceBuffer, rows: &[Vec<f32>]) {
    let all = read_bytes(ctx, ring, ring.bytes());
    for (pos, row) in rows.iter().enumerate().skip(rows.len() - SHAPE.capacity) {
        let start = pos % SHAPE.capacity * KV * 4;
        assert_eq!(&all[start..start + KV * 4], bytes(row), "ring row {pos}");
    }
}

fn assert_invalid_positions(
    ctx: &CudaContext,
    stream: &DeviceStream,
    buf: &Buffers,
    launch: impl Fn(),
) {
    let snapshots = [&buf.keys, &buf.values, &buf.out].map(|b| read_bytes(ctx, b, b.bytes()));
    for invalid in [-1, POS_MAX + 1] {
        buf.status
            .write_at(stream, 0, &1_i32.to_le_bytes())
            .unwrap();
        buf.pos.write_at(stream, 0, &invalid.to_le_bytes()).unwrap();
        launch();
        stream.synchronize().unwrap();
        let status = read_bytes(ctx, &buf.status, 4).try_into().unwrap();
        assert_eq!(i32::from_le_bytes(status), 1 | STATUS_INVALID_STEP_STATE);
        for (b, snapshot) in [&buf.keys, &buf.values, &buf.out]
            .into_iter()
            .zip(&snapshots)
        {
            assert_eq!(&read_bytes(ctx, b, b.bytes()), snapshot);
        }
    }
}

/// A query tile against the FP64 reference: rows `first .. first + count` with their own
/// queries, coefficients, sinks and gates, the older keys in the pre-tile ring (its other rows
/// NaN, so a wrong row shows) and the tile's own keys staged. The tensor-core kernel rounds Q,
/// K, V, the probabilities, the coefficients and the bias table to f16; an emulation of that
/// rounding on these values peaks at 1.1e-3. The commit then publishes the tile's keys and
/// values to the ring bit for bit.
#[test]
fn query_tile_matches_the_fp64_reference() {
    if !device_present("relative attention query tile") {
        return;
    }
    const TOLERANCE: f32 = 4e-3;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    // A tile wrapping the ring with three query blocks, a first tile, and the smallest tile.
    for (first, count) in [(150, 130), (0, 70), (7, 2)] {
        let positions = first + count;
        let keys: Vec<Vec<f32>> = (0..positions).map(|p| pattern(KV, p * 2 + 6)).collect();
        let values: Vec<Vec<f32>> = (0..positions).map(|p| pattern(KV, p * 2 + 7)).collect();
        let rows = |width: usize, seed: usize, scale: f32| -> Vec<f32> {
            pattern(count * width, seed)
                .into_iter()
                .map(|x| x * scale)
                .collect()
        };
        let q = rows(QUERY, 1, 1.0);
        let relative = rows(SHAPE.query_heads * SHAPE.relative_dims, 2, 1.0);
        let sinks = rows(SHAPE.query_heads, 4, 8.0);
        let gates = rows(QUERY, 5, 4.0);
        let table = pattern(SHAPE.relative_dims * SHAPE.relative_extent, 3);
        let mut ring = [
            vec![f32::NAN; SHAPE.capacity * KV],
            vec![f32::NAN; SHAPE.capacity * KV],
        ];
        for p in first.saturating_sub(SHAPE.capacity)..first {
            let at = p % SHAPE.capacity * KV;
            ring[0][at..at + KV].copy_from_slice(&keys[p]);
            ring[1][at..at + KV].copy_from_slice(&values[p]);
        }
        let positions_bytes: Vec<u8> = (first..positions)
            .flat_map(|p| i32::try_from(p).unwrap().to_le_bytes())
            .collect();
        let q_buf = upload_f32(&ctx, &q);
        let k_buf = upload_f32(&ctx, &keys[first..].concat());
        let v_buf = upload_f32(&ctx, &values[first..].concat());
        let relative_buf = upload_f32(&ctx, &relative);
        let table_buf = upload_f32(&ctx, &table);
        let sinks_buf = upload_f32(&ctx, &sinks);
        let gates_buf = upload_f32(&ctx, &gates);
        let ring_buf = ring.each_ref().map(|plane| upload_f32(&ctx, plane));
        let pos_buf = ctx.upload(&positions_bytes).unwrap();
        let status = upload_filled(&ctx, 0, count * 4);
        let out = upload_filled(&ctx, 0xff, count * QUERY * 4);
        let call = RelativeAttentionCall {
            shape: SHAPE,
            kv_format: RelativeKvFormat::F32,
            q: whole(&q_buf),
            k: whole(&k_buf),
            v: whole(&v_buf),
            relative: whole(&relative_buf),
            table: whole(&table_buf),
            sinks: whole(&sinks_buf),
            gates: whole(&gates_buf),
            k_cache: whole(&ring_buf[0]),
            v_cache: whole(&ring_buf[1]),
            out: whole(&out),
            pos: whole(&pos_buf),
            status: whole(&status),
        };
        let tile = RelativeAttentionTile {
            first,
            count,
            input_stride: 0,
            query_stride: 0,
            side_stride: 0,
        };
        ctx.relative_attention_tile(&stream, call, tile).unwrap();
        stream.synchronize().unwrap();
        assert!(
            read_bytes(&ctx, &status, count * 4).iter().all(|&b| b == 0),
            "first {first}: a row reported an invalid state"
        );
        let got = floats(&read_bytes(&ctx, &out, count * QUERY * 4));
        let mut worst = 0.0f32;
        for (r, actual) in got.as_chunks::<QUERY>().0.iter().enumerate() {
            let row = Row {
                q: &q[r * QUERY..][..QUERY],
                relative: &relative[r * SHAPE.query_heads * SHAPE.relative_dims..]
                    [..SHAPE.query_heads * SHAPE.relative_dims],
                sinks: &sinks[r * SHAPE.query_heads..][..SHAPE.query_heads],
                gates: &gates[r * QUERY..][..QUERY],
            };
            let expected = reference(row, &keys, &values, &table, first + r);
            for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                let error = (a - e).abs();
                assert!(
                    error <= TOLERANCE,
                    "first {first}, row {r}, element {i}: expected {e}, got {a}"
                );
                worst = worst.max(error);
            }
        }
        println!("query tile at {first} of {count} rows: worst error {worst:e}");
        for (plane, rows) in ring_buf.iter().zip([&keys, &values]) {
            let all = read_bytes(&ctx, plane, plane.bytes());
            let kept = positions.saturating_sub(SHAPE.capacity).max(first);
            for (p, row) in rows.iter().enumerate().skip(kept) {
                let at = p % SHAPE.capacity * KV * 4;
                assert_eq!(&all[at..at + KV * 4], bytes(row), "ring row {p}");
            }
        }
    }
}

#[test]
fn compact_kv_formats_match_shared_kv_format_and_round_trip() {
    for kv_heads in [1, 8, 16] {
        for fmt in [
            RelativeKvFormat::F32,
            RelativeKvFormat::F16,
            RelativeKvFormat::Int8,
        ] {
            assert!(fmt.plane_bytes_matches_shared(kv_heads), "{fmt:?}");
        }
    }
    let row = pattern(KV, 42);
    for (fmt, tol) in [
        (RelativeKvFormat::F16, 2e-3),
        (RelativeKvFormat::Int8, 2e-2),
    ] {
        let mut encoded = Vec::new();
        fmt.encode_row(&row, &mut encoded);
        assert_eq!(encoded.len(), fmt.row_bytes(SHAPE.kv_heads));
        let decoded = fmt.decode_row(&encoded);
        assert_eq!(decoded.len(), row.len());
        for (i, (&a, &b)) in decoded.iter().zip(&row).enumerate() {
            assert!((a - b).abs() <= tol, "{fmt:?} [{i}]: {a} vs {b}");
        }
    }
    let mut poisoned = row.clone();
    poisoned[37] = f32::NAN;
    let mut encoded = Vec::new();
    RelativeKvFormat::Int8.encode_row(&poisoned, &mut encoded);
    let decoded = RelativeKvFormat::Int8.decode_row(&encoded);
    assert!(decoded[32..64].iter().all(|v| v.is_nan()));
    assert!(decoded[..32].iter().all(|v| v.is_finite()));
}

#[test]
fn compact_kv_rings_match_fp64_reference_in_decode_and_tile() {
    if !device_present("compact relative attention rings") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let steps = SHAPE.capacity + 11;
    let fixture = Fixture::new(steps);
    let table_buf = upload_f32(&ctx, &fixture.table);
    let q_buf = upload_f32(&ctx, &fixture.q);
    let rel_buf = upload_f32(&ctx, &fixture.relative);
    let sink_buf = upload_f32(&ctx, &fixture.sinks);
    let gate_buf = upload_f32(&ctx, &fixture.gates);

    for (fmt, decode_tol, tile_tol) in [
        (RelativeKvFormat::F16, 3e-3f32, 4e-3f32),
        (RelativeKvFormat::Int8, 3e-2f32, 3e-2f32),
    ] {
        let row_bytes = fmt.row_bytes(SHAPE.kv_heads);
        let cache_bytes = SHAPE.capacity * row_bytes;
        let k_ring = upload_filled(&ctx, 0, cache_bytes);
        let v_ring = upload_filled(&ctx, 0, cache_bytes);
        let k_buf = ctx.allocate(KV * 4).unwrap();
        let v_buf = ctx.allocate(KV * 4).unwrap();
        let pos_buf = ctx.allocate(4).unwrap();
        let status = upload_filled(&ctx, 0, 4);
        let out = ctx.allocate(QUERY * 4).unwrap();
        for pos in 0..steps {
            k_buf
                .write_at(&stream, 0, &bytes(&fixture.keys[pos]))
                .unwrap();
            v_buf
                .write_at(&stream, 0, &bytes(&fixture.values[pos]))
                .unwrap();
            pos_buf
                .write_at(&stream, 0, &(pos as i32).to_le_bytes())
                .unwrap();
            ctx.relative_attention(
                &stream,
                RelativeAttentionCall {
                    shape: SHAPE,
                    kv_format: fmt,
                    q: whole(&q_buf),
                    k: whole(&k_buf),
                    v: whole(&v_buf),
                    relative: whole(&rel_buf),
                    table: whole(&table_buf),
                    sinks: whole(&sink_buf),
                    gates: whole(&gate_buf),
                    k_cache: whole(&k_ring),
                    v_cache: whole(&v_ring),
                    pos: whole(&pos_buf),
                    status: whole(&status),
                    out: whole(&out),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            if pos == 0 || pos == SHAPE.capacity - 1 || pos + 1 == steps {
                let got = floats(&read_bytes(&ctx, &out, QUERY * 4));
                let want = fixture.reference(pos);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= decode_tol,
                    "{fmt:?} decode pos {pos}: worst {worst:e} > {decode_tol:e}"
                );
            }
        }
        // Device-encoded ring cells match host encode_row bit-for-bit.
        for (ring, rows) in [(&k_ring, &fixture.keys), (&v_ring, &fixture.values)] {
            let all = read_bytes(&ctx, ring, cache_bytes);
            for (pos, row) in rows.iter().enumerate().skip(steps - SHAPE.capacity) {
                let mut expected = Vec::with_capacity(row_bytes);
                fmt.encode_row(row, &mut expected);
                let start = (pos % SHAPE.capacity) * row_bytes;
                assert_eq!(
                    &all[start..start + row_bytes],
                    expected,
                    "{fmt:?} row {pos}"
                );
            }
        }
        // Tile path past a ring wrap reading pre-tile compact cells and committing compact cells.
        let (first, count) = (100usize, 70usize);
        let total = first + count;
        let keys: Vec<Vec<f32>> = (0..total).map(|p| pattern(KV, p * 2 + 6)).collect();
        let values: Vec<Vec<f32>> = (0..total).map(|p| pattern(KV, p * 2 + 7)).collect();
        let mut init = [vec![0u8; cache_bytes], vec![0u8; cache_bytes]];
        for p in first.saturating_sub(SHAPE.capacity)..first {
            let at = (p % SHAPE.capacity) * row_bytes;
            let mut enc_k = Vec::with_capacity(row_bytes);
            let mut enc_v = Vec::with_capacity(row_bytes);
            fmt.encode_row(&keys[p], &mut enc_k);
            fmt.encode_row(&values[p], &mut enc_v);
            init[0][at..at + row_bytes].copy_from_slice(&enc_k);
            init[1][at..at + row_bytes].copy_from_slice(&enc_v);
        }
        let tile_rings = init.each_ref().map(|b| ctx.upload(b).unwrap());
        let rows_f = |width: usize, seed: usize, scale: f32| -> Vec<f32> {
            pattern(count * width, seed)
                .into_iter()
                .map(|x| x * scale)
                .collect()
        };
        let q = rows_f(QUERY, 1, 1.0);
        let rel = rows_f(SHAPE.query_heads * SHAPE.relative_dims, 2, 1.0);
        let sinks = rows_f(SHAPE.query_heads, 4, 8.0);
        let gates = rows_f(QUERY, 5, 4.0);
        let pos_bytes: Vec<u8> = (first..total)
            .flat_map(|p| (p as i32).to_le_bytes())
            .collect();
        let q_t = upload_f32(&ctx, &q);
        let k_t = upload_f32(&ctx, &keys[first..].concat());
        let v_t = upload_f32(&ctx, &values[first..].concat());
        let rel_t = upload_f32(&ctx, &rel);
        let sink_t = upload_f32(&ctx, &sinks);
        let gate_t = upload_f32(&ctx, &gates);
        let pos_t = ctx.upload(&pos_bytes).unwrap();
        let status_t = upload_filled(&ctx, 0, count * 4);
        let out_t = upload_filled(&ctx, 0xff, count * QUERY * 4);
        ctx.relative_attention_tile(
            &stream,
            RelativeAttentionCall {
                shape: SHAPE,
                kv_format: fmt,
                q: whole(&q_t),
                k: whole(&k_t),
                v: whole(&v_t),
                relative: whole(&rel_t),
                table: whole(&table_buf),
                sinks: whole(&sink_t),
                gates: whole(&gate_t),
                k_cache: whole(&tile_rings[0]),
                v_cache: whole(&tile_rings[1]),
                out: whole(&out_t),
                pos: whole(&pos_t),
                status: whole(&status_t),
            },
            RelativeAttentionTile {
                first,
                count,
                input_stride: 0,
                query_stride: 0,
                side_stride: 0,
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert!(
            read_bytes(&ctx, &status_t, count * 4)
                .iter()
                .all(|&b| b == 0)
        );
        let got = floats(&read_bytes(&ctx, &out_t, count * QUERY * 4));
        let mut worst = 0.0f32;
        for (r, actual) in got.as_chunks::<QUERY>().0.iter().enumerate() {
            let row = Row {
                q: &q[r * QUERY..][..QUERY],
                relative: &rel[r * SHAPE.query_heads * SHAPE.relative_dims..]
                    [..SHAPE.query_heads * SHAPE.relative_dims],
                sinks: &sinks[r * SHAPE.query_heads..][..SHAPE.query_heads],
                gates: &gates[r * QUERY..][..QUERY],
            };
            let want = reference(row, &keys, &values, &fixture.table, first + r);
            for (a, e) in actual.iter().zip(&want) {
                worst = worst.max((a - e).abs());
            }
        }
        assert!(
            worst <= tile_tol,
            "{fmt:?} tile worst {worst:e} > {tile_tol:e}"
        );
        for (plane, rows) in tile_rings.iter().zip([&keys, &values]) {
            let all = read_bytes(&ctx, plane, cache_bytes);
            for (p, row) in rows.iter().enumerate().skip(first) {
                let mut expected = Vec::with_capacity(row_bytes);
                fmt.encode_row(row, &mut expected);
                let at = (p % SHAPE.capacity) * row_bytes;
                assert_eq!(&all[at..at + row_bytes], expected, "{fmt:?} tile row {p}");
            }
        }
    }
}
