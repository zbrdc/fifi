//! Device tests (3090) of the `pack` and `matmul` entries on a stream: the Rust
//! wrapper against the C entry against the vendored kernels.
//!
//! "Bytes equal" means `to_bits()` equality of every f32. The reference is the
//! blocking path, `CudaContext::linear_rows`, which runs the same kernels on the
//! default stream. A download or a blocking linear call waits for the whole
//! device, so the equality tests prove the data, not the ordering; the two
//! synchronize tests read the pinned word without any other wait for that.
//!
//! The owner counts and the queued work are process-wide, so these tests assume
//! `--test-threads=1`, as the device suite is always run.
//!
//! Each refusal is checked by the text of its error as well as its variant, so a
//! test fails when another check refuses the call instead of the one under test.

use std::time::{Duration, Instant};

use crate::EngError;
use crate::kernels::cuda::native::{self, MAX_DIM};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceStream, MatmulIo, PackCall, Projection,
    QuantTensor, TiledWeight, WordValue,
};
use crate::weights::{GgufType, dequantize};

/// Bytes of the fixed fragment-order `xf` buffer.
const XF_FRAGMENT: usize = 1_048_592;
/// The 8 bytes a test copies into the pinned word.
const WORD_BYTES: [u8; 8] = [0x01, 0x02, 0x03, 0x04, 0x0D, 0x0C, 0x0B, 0x0A];
const EXPECTED: WordValue = WordValue {
    token: 0x0403_0201,
    status: 0x0A0B_0C0D,
};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &str) {
    let Err(error) = result else {
        panic!("{case}: expected a BadRequest, got Ok");
    };
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected BadRequest containing {needle:?}, got {error:?}"
    );
}

/// Bytes of sentinel around the buffers of a slack run.
const SLACK_TAIL: usize = 64;
/// Where a slack run puts x, xf and y inside their buffers (aligned for each).
const SLACK_OFFSETS: (usize, usize, usize) = (4, 256, 64);

/// How `on_stream` sets up one run.
struct Run<'a> {
    projection: Projection<'a>,
    x: &'a [f32],
    /// Put x, xf and y at nonzero offsets inside larger buffers, with sentinel
    /// bytes before and after each range, and check that none of them changed.
    slack: bool,
    /// Queue this work on the stream first, so the launches sit behind it.
    backlog: Option<&'a PackBacklog>,
}

/// Asserts that every byte of `bytes` outside `inside` still equals `sentinel`.
fn assert_untouched(bytes: &[u8], inside: std::ops::Range<usize>, sentinel: u8, what: &str) {
    assert!(
        bytes[..inside.start].iter().all(|&byte| byte == sentinel),
        "{what}: bytes before the range changed"
    );
    assert!(
        bytes[inside.end..].iter().all(|&byte| byte == sentinel),
        "{what}: bytes after the range changed"
    );
}

/// Packs `run.x` and multiplies the rows of `run.projection` on `stream`, waits
/// for the stream and returns `y`. `xf` starts as 0xAA and `y` as 0xFF, so a
/// byte the launches did not write shows.
fn on_stream(ctx: &CudaContext, stream: &DeviceStream, run: Run<'_>) -> Vec<f32> {
    let Projection {
        weight,
        first_row,
        rows,
    } = run.projection;
    let (k, m) = (weight.k(), run.x.len() / weight.k());
    let need = native::activation_bytes(rows, k, m).expect("sizes of the call");
    let (x_at, xf_at, y_at) = if run.slack { SLACK_OFFSETS } else { (0, 0, 0) };
    let tail = if run.slack { SLACK_TAIL } else { 0 };
    let mut x_bytes = vec![0xCC; x_at];
    x_bytes.extend(run.x.iter().flat_map(|value| value.to_le_bytes()));
    x_bytes.extend(vec![0xCC; tail]);
    let x_buffer = testkit::upload_bytes(ctx, &x_bytes);
    let xf_buffer = testkit::upload_filled(ctx, 0xAA, xf_at + need.xf + tail);
    let y_buffer = testkit::upload_filled(ctx, 0xFF, y_at + need.y + tail);
    // Every upload is done: an upload waits for the device and would drain the backlog.
    let queued = run
        .backlog
        .map(|backlog| (backlog.queue(ctx, stream), Instant::now()));
    let xf = xf_buffer.range(xf_at, need.xf).expect("xf range");
    let pack = PackCall {
        x: x_buffer.range(x_at, need.x).expect("x range"),
        xf,
        m,
        k,
    };
    ctx.pack(stream, pack).expect("queue the pack");
    let projection = Projection {
        weight,
        first_row,
        rows,
    };
    let io = MatmulIo {
        xf,
        y: y_buffer.range(y_at, need.y).expect("y range"),
        m,
    };
    ctx.matmul(stream, projection, io)
        .expect("queue the matmul");
    stream.synchronize().expect("drain the stream");
    if let Some((host, queued_at)) = queued {
        report_margin("pack backlog", host, queued_at.elapsed());
    }
    let y_all = testkit::read_bytes(ctx, &y_buffer, y_at + need.y + tail);
    if run.slack {
        assert_untouched(&y_all, y_at..y_at + need.y, 0xFF, "y");
        let xf_all = testkit::read_bytes(ctx, &xf_buffer, xf_at + need.xf + tail);
        assert_untouched(&xf_all, xf_at..xf_at + need.xf, 0xAA, "xf");
    }
    testkit::floats(&y_all[y_at..y_at + need.y])
}

/// One weight shape of the equality tests: type, rows of the whole weight, k.
struct Shape {
    ty: GgufType,
    rows: usize,
    k: usize,
}

/// For every shape, token count and first row: the stream path and the blocking
/// path give the same bytes for 32 rows of the weight.
fn assert_stream_equals_blocking(shapes: &[Shape], tokens: &[usize]) {
    let ctx = CudaContext::new().expect("context");
    for shape in shapes {
        let weight = testkit::upload_weight(&ctx, shape.ty, shape.rows, shape.k);
        for &m in tokens {
            for first_row in [0, 8] {
                let case = format!("{:?} k {} m {m} first_row {first_row}", shape.ty, shape.k);
                let x = testkit::input(m, shape.k);
                let stream = ctx.stream().expect("fresh stream");
                let run = Run {
                    projection: Projection {
                        weight: &weight,
                        first_row,
                        rows: 32,
                    },
                    x: &x,
                    // The first_row = 8 half also runs at nonzero offsets in its buffers.
                    slack: first_row == 8,
                    backlog: None,
                };
                let streamed = on_stream(&ctx, &stream, run);
                let reference = testkit::blocking(&ctx, &weight, (first_row, 32), &x);
                assert!(
                    reference.iter().any(|&value| value != 0.0),
                    "{case}: the reference is all zero, so the comparison proves nothing"
                );
                assert_eq!(bits(&streamed), bits(&reference), "{case}");
            }
        }
    }
}

#[test]
fn decode_launches_on_a_stream_equal_the_blocking_path() {
    if !testkit::device_present("decode_launches_on_a_stream_equal_the_blocking_path") {
        return;
    }
    // The last shape splits along k: the matmul also queues a split-K memset, and
    // two partial sums onto zero add in either order to the same bits.
    let shapes = [
        Shape {
            ty: GgufType::Q4K,
            rows: 64,
            k: 512,
        },
        Shape {
            ty: GgufType::Q6K,
            rows: 64,
            k: 512,
        },
        Shape {
            ty: GgufType::Q80,
            rows: 64,
            k: 256,
        },
        Shape {
            ty: GgufType::Q4K,
            rows: 64,
            k: 4096,
        },
    ];
    assert_stream_equals_blocking(&shapes, &[1, 3, 8, 9, 16]);
}

#[test]
fn wide_launches_on_a_stream_equal_the_blocking_path() {
    if !testkit::device_present("wide_launches_on_a_stream_equal_the_blocking_path") {
        return;
    }
    // 17 to 32 tokens, 33 to 64 and above are the three gemm tile instances.
    let shapes = [
        Shape {
            ty: GgufType::Q4K,
            rows: 64,
            k: 512,
        },
        Shape {
            ty: GgufType::Q6K,
            rows: 64,
            k: 512,
        },
        Shape {
            ty: GgufType::Q80,
            rows: 64,
            k: 512,
        },
        Shape {
            ty: GgufType::Q80,
            rows: 64,
            k: 256,
        },
    ];
    assert_stream_equals_blocking(&shapes, &[17, 32, 33, 64, 65]);
}

#[test]
fn pack_on_a_stream_leaves_no_stale_fragment_bytes() {
    if !testkit::device_present("pack_on_a_stream_leaves_no_stale_fragment_bytes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let k = 512;

    // Fragment order (one token): the pack writes only the positions of its token,
    // so the zeroing queued before it is what clears the rest.
    let x = testkit::upload_f32(&ctx, &testkit::input(1, k));
    let xf = testkit::upload_filled(&ctx, 0xAA, XF_FRAGMENT);
    let call = PackCall {
        x: x.range(0, k * 4).expect("x range"),
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        m: 1,
        k,
    };
    ctx.pack(&stream, call).expect("queue the pack");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(&ctx, &xf, XF_FRAGMENT);
    assert!(
        bytes[XF_FRAGMENT - 16..].iter().all(|&byte| byte == 0),
        "the last 16 bytes are zero"
    );
    let (vectors, _) = bytes.as_chunks::<16>();
    let used = vectors
        .iter()
        .filter(|vector| vector.iter().any(|&byte| byte != 0))
        .count();
    assert!(used <= k / 8, "{used} non-zero vectors, at most {}", k / 8);

    // Row-major (17 tokens): fully written for the call, never zeroed beyond it.
    let m = 17;
    let used_bytes = m * k * 2;
    let x = testkit::upload_f32(&ctx, &testkit::input(m, k));
    let xf = testkit::upload_filled(&ctx, 0xAA, used_bytes + 64);
    let call = PackCall {
        x: x.range(0, m * k * 4).expect("x range"),
        xf: xf.range(0, used_bytes + 64).expect("xf range"),
        m,
        k,
    };
    ctx.pack(&stream, call).expect("queue the wide pack");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(&ctx, &xf, used_bytes + 64);
    assert!(
        bytes[used_bytes..].iter().all(|&byte| byte == 0xAA),
        "bytes beyond m * k halves are untouched"
    );
    assert!(
        bytes[..used_bytes].iter().any(|&byte| byte != 0xAA),
        "the pack wrote its halves"
    );
}

/// Buffers for the reject table. The weight is large enough that its buffer can
/// hold a whole fragment-order `xf`, so a case can put `xf` over its tiles.
struct Fixture {
    ctx: CudaContext,
    stream: DeviceStream,
    weight: TiledWeight,
    x: DeviceBuffer,
    xf: DeviceBuffer,
    y: DeviceBuffer,
    both: DeviceBuffer,
}

const SENTINEL_XF: u8 = 0xAA;
const SENTINEL_Y: u8 = 0xFF;

impl Fixture {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        // 8192 rows of 256 values: 1_179_648 bytes of tiles.
        let weight = testkit::upload_weight(&ctx, GgufType::Q4K, 8192, 256);
        let x = testkit::upload_f32(&ctx, &testkit::input(8, 256));
        let xf = testkit::upload_filled(&ctx, SENTINEL_XF, XF_FRAGMENT + 64);
        let y = testkit::upload_filled(&ctx, SENTINEL_Y, 256);
        let both = testkit::upload_filled(&ctx, SENTINEL_XF, XF_FRAGMENT + 4096);
        Self {
            ctx,
            stream,
            weight,
            x,
            xf,
            y,
            both,
        }
    }

    /// A pack on the fixture's own stream and `x`/`xf` buffers: byte ranges, then `(m, k)`.
    fn pack(
        &self,
        x: (usize, usize),
        xf: (usize, usize),
        shape: (usize, usize),
    ) -> Result<(), EngError> {
        self.pack_on(&self.stream, (&self.x, &self.xf), (x, xf), shape)
    }

    fn pack_on(
        &self,
        stream: &DeviceStream,
        buffers: (&DeviceBuffer, &DeviceBuffer),
        ranges: ((usize, usize), (usize, usize)),
        shape: (usize, usize),
    ) -> Result<(), EngError> {
        let call = PackCall {
            x: buffers.0.range(ranges.0.0, ranges.0.1)?,
            xf: buffers.1.range(ranges.1.0, ranges.1.1)?,
            m: shape.0,
            k: shape.1,
        };
        self.ctx.pack(stream, call)
    }

    /// Rows `first_row..first_row + rows` of the weight, one token.
    fn matmul(
        &self,
        buffers: (&DeviceBuffer, &DeviceBuffer),
        ranges: ((usize, usize), (usize, usize)),
        rows: (usize, usize, usize),
    ) -> Result<(), EngError> {
        let projection = Projection {
            weight: &self.weight,
            first_row: rows.0,
            rows: rows.1,
        };
        let io = MatmulIo {
            xf: buffers.0.range(ranges.0.0, ranges.0.1)?,
            y: buffers.1.range(ranges.1.0, ranges.1.1)?,
            m: rows.2,
        };
        self.ctx.matmul(&self.stream, projection, io)
    }
}

#[test]
fn launch_wrappers_reject_bad_ranges_before_any_native_call() {
    if !testkit::device_present("launch_wrappers_reject_bad_ranges_before_any_native_call") {
        return;
    }
    let fx = Fixture::new();
    let other = CudaContext::new().expect("second context");
    let foreign_stream = other.stream().expect("stream of the other context");
    let foreign_x = testkit::upload_f32(&other, &testkit::input(1, 256));
    let (x_ok, xf_ok) = ((0, 1024), (0, XF_FRAGMENT));
    let xf_buffers = (&fx.x, &fx.xf);
    let both = (&fx.both, &fx.both);
    let xf_y = (&fx.xf, &fx.y);

    let pack: Vec<(&str, &str, Result<(), EngError>)> = vec![
        (
            "x one float short",
            "x range holds",
            fx.pack((0, 1020), xf_ok, (1, 256)),
        ),
        (
            "xf one byte short",
            "xf range holds",
            fx.pack(x_ok, (0, XF_FRAGMENT - 1), (1, 256)),
        ),
        (
            "x misaligned by 2",
            "x range starts at offset",
            fx.pack((2, 1024), xf_ok, (1, 256)),
        ),
        (
            "xf misaligned by 4",
            "xf range starts at offset",
            fx.pack(x_ok, (4, XF_FRAGMENT), (1, 256)),
        ),
        ("no tokens", "tokens", fx.pack(x_ok, xf_ok, (0, 256))),
        (
            "more tokens than MAX_DIM",
            "tokens",
            fx.pack(x_ok, xf_ok, (MAX_DIM + 1, 256)),
        ),
        ("k = 100", "multiple of 256", fx.pack(x_ok, xf_ok, (1, 100))),
        (
            "x and xf overlap",
            "x and xf ranges overlap",
            fx.pack_on(&fx.stream, both, ((0, 1024), (1008, XF_FRAGMENT)), (1, 256)),
        ),
        (
            "a stream of another context",
            "stream belongs",
            fx.pack_on(&foreign_stream, xf_buffers, (x_ok, xf_ok), (1, 256)),
        ),
        (
            "a buffer of another context",
            "buffer belongs",
            fx.pack_on(&fx.stream, (&foreign_x, &fx.xf), (x_ok, xf_ok), (1, 256)),
        ),
    ];
    let matmul: Vec<(&str, &str, Result<(), EngError>)> = vec![
        (
            "y one float short",
            "y range holds",
            fx.matmul(xf_y, (xf_ok, (0, 60)), (0, 16, 1)),
        ),
        (
            "xf one byte short",
            "xf range holds",
            fx.matmul(xf_y, ((0, XF_FRAGMENT - 1), (0, 64)), (0, 16, 1)),
        ),
        (
            "xf misaligned by 4",
            "xf range starts at offset",
            fx.matmul(xf_y, ((4, XF_FRAGMENT), (0, 64)), (0, 16, 1)),
        ),
        (
            "y misaligned by 2",
            "y range starts at offset",
            fx.matmul(xf_y, (xf_ok, (2, 64)), (0, 16, 1)),
        ),
        (
            "xf and y overlap",
            "y and xf ranges overlap",
            fx.matmul(both, (xf_ok, (XF_FRAGMENT - 32, 64)), (0, 16, 1)),
        ),
        (
            "y over the weight tiles",
            "y and weight ranges overlap",
            fx.matmul((&fx.xf, &fx.weight.buffer), (xf_ok, (0, 64)), (0, 16, 1)),
        ),
        (
            "xf over the weight tiles",
            "xf and weight ranges overlap",
            fx.matmul(
                (&fx.weight.buffer, &fx.y),
                ((16, XF_FRAGMENT), (0, 64)),
                (0, 16, 1),
            ),
        ),
        (
            "12 rows",
            "whole 8-row groups",
            fx.matmul(xf_y, (xf_ok, (0, 64)), (0, 12, 1)),
        ),
        (
            "rows past the weight",
            "whole 8-row groups",
            fx.matmul(xf_y, (xf_ok, (0, 64)), (8184, 16, 1)),
        ),
        (
            "no tokens",
            "tokens",
            fx.matmul(xf_y, (xf_ok, (0, 64)), (0, 16, 0)),
        ),
        (
            "more tokens than MAX_DIM",
            "tokens",
            fx.matmul(xf_y, (xf_ok, (0, 64)), (0, 16, MAX_DIM + 1)),
        ),
    ];
    for (case, needle, result) in pack.into_iter().chain(matmul) {
        assert_bad_request(result, case, needle);
    }

    // No refusal queued anything: the sentinels are untouched and the backend works.
    fx.stream.synchronize().expect("drain");
    let xf = testkit::read_bytes(&fx.ctx, &fx.xf, XF_FRAGMENT + 64);
    assert!(xf.iter().all(|&byte| byte == SENTINEL_XF), "xf untouched");
    let y = testkit::read_bytes(&fx.ctx, &fx.y, 256);
    assert!(y.iter().all(|&byte| byte == SENTINEL_Y), "y untouched");
    fx.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
}

/// A large matmul to queue on a stream many times. `prepare` does every upload
/// (an upload waits for the device); `queue` only launches.
struct Backlog {
    weight: TiledWeight,
    x: DeviceBuffer,
    xf: DeviceBuffer,
    y: DeviceBuffer,
}

/// Launches in the backlog. At about 50 microseconds of device time each, the
/// stream is still busy for tens of milliseconds after the host queued the last one.
const BACKLOG_LAUNCHES: usize = 1000;
const BACKLOG_ROWS: usize = 8192;
const BACKLOG_K: usize = 8192;
const BACKLOG_TOKENS: usize = 16;

impl Backlog {
    /// An 8192 x 8192 Q4_K weight and the buffers of a 16-token call on it.
    fn prepare(ctx: &CudaContext) -> Self {
        let weight = testkit::upload_weight(ctx, GgufType::Q4K, BACKLOG_ROWS, BACKLOG_K);
        let need = native::activation_bytes(BACKLOG_ROWS, BACKLOG_K, BACKLOG_TOKENS)
            .expect("sizes of the call");
        Self {
            weight,
            x: testkit::upload_f32(ctx, &testkit::input(BACKLOG_TOKENS, BACKLOG_K)),
            xf: testkit::upload_filled(ctx, 0, need.xf),
            y: testkit::upload_filled(ctx, 0, need.y),
        }
    }

    /// Packs 16 tokens, then queues `BACKLOG_LAUNCHES` matmuls on `stream` and
    /// returns without waiting. Returns how long the host took to queue them.
    fn queue(&self, ctx: &CudaContext, stream: &DeviceStream) -> Duration {
        let started = Instant::now();
        let (rows, k, m) = (BACKLOG_ROWS, BACKLOG_K, BACKLOG_TOKENS);
        let need = native::activation_bytes(rows, k, m).expect("sizes of the call");
        let xf_range = self.xf.range(0, need.xf).expect("xf range");
        let y_range = self.y.range(0, need.y).expect("y range");
        let pack = PackCall {
            x: self.x.range(0, need.x).expect("x range"),
            xf: xf_range,
            m,
            k,
        };
        ctx.pack(stream, pack).expect("queue the pack");
        for _ in 0..BACKLOG_LAUNCHES {
            let projection = Projection {
                weight: &self.weight,
                first_row: 0,
                rows,
            };
            let io = MatmulIo {
                xf: xf_range,
                y: y_range,
                m,
            };
            ctx.matmul(stream, projection, io)
                .expect("queue a backlog matmul");
        }
        started.elapsed()
    }
}

/// Prints how long the host took to queue a backlog and how long the device was
/// still busy afterwards. The backlog proves an ordering only while the second
/// number is far above the time the test needs to queue its own launches, so the
/// device session reads this line.
fn report_margin(kind: &str, host: Duration, device_after: Duration) {
    println!(
        "backlog margin ({kind}): host queued in {:.2} ms, device busy {:.2} ms after the last launch",
        host.as_secs_f64() * 1e3,
        device_after.as_secs_f64() * 1e3
    );
    if device_after < Duration::from_millis(2) {
        println!("WARNING backlog margin ({kind}) is below 2 ms: the ordering tests prove little");
    }
}

/// Packs queued on a stream, for the wrong-stream test. A pack of 256 tokens of
/// 32768 values moves about 50 MB, so each one keeps the device busy longer than
/// the host takes to launch it, and its 256 blocks fill only part of the device.
/// That second property is what the test needs: when a backlog fills every
/// multiprocessor, the device starts kernels in the order they were submitted, so
/// a launch on the wrong stream still starts after the backlog (the first version
/// of this test used 8192-token packs and both stream mutants survived it).
struct PackBacklog {
    x: DeviceBuffer,
    xf: DeviceBuffer,
}

const PACK_BACKLOG_LAUNCHES: usize = 2000;
const PACK_BACKLOG_TOKENS: usize = 256;
const PACK_BACKLOG_K: usize = 32768;

impl PackBacklog {
    /// The buffers of the packs. The values do not matter, only the memory traffic.
    fn prepare(ctx: &CudaContext) -> Self {
        let (m, k) = (PACK_BACKLOG_TOKENS, PACK_BACKLOG_K);
        let (x_bytes, xf_bytes) = native::pack_bytes(m, k).expect("sizes of the pack");
        Self {
            x: testkit::upload_filled(ctx, 0, x_bytes),
            xf: testkit::upload_filled(ctx, 0, xf_bytes),
        }
    }

    /// Queues `PACK_BACKLOG_LAUNCHES` packs on `stream` and returns without
    /// waiting. Returns how long the host took to queue them.
    fn queue(&self, ctx: &CudaContext, stream: &DeviceStream) -> Duration {
        let started = Instant::now();
        let (m, k) = (PACK_BACKLOG_TOKENS, PACK_BACKLOG_K);
        let (x_bytes, xf_bytes) = native::pack_bytes(m, k).expect("sizes of the pack");
        for _ in 0..PACK_BACKLOG_LAUNCHES {
            let pack = PackCall {
                x: self.x.range(0, x_bytes).expect("x range"),
                xf: self.xf.range(0, xf_bytes).expect("xf range"),
                m,
                k,
            };
            ctx.pack(stream, pack).expect("queue a backlog pack");
        }
        started.elapsed()
    }
}

#[test]
fn a_stream_synchronize_waits_for_the_work_queued_before_it() {
    if !testkit::device_present("a_stream_synchronize_waits_for_the_work_queued_before_it") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let source = testkit::upload_bytes(&ctx, &WORD_BYTES);
    let word = ctx.pinned_word().expect("pinned word");
    let backlog = Backlog::prepare(&ctx);
    let host = backlog.queue(&ctx, &stream);
    let queued_at = Instant::now();
    // The copy is behind the whole backlog, so only a real wait lets the host see it.
    ctx.copy_word_to_host(&stream, source.range(0, 8).expect("word source"), &word)
        .expect("queue the copy");
    stream.synchronize().expect("drain the stream");
    report_margin(
        "matmul backlog, stream synchronize",
        host,
        queued_at.elapsed(),
    );
    assert_eq!(word.peek().expect("peek"), WORD_BYTES);
    drop(backlog);
}

#[test]
fn an_event_synchronize_waits_for_the_copy_behind_queued_work() {
    if !testkit::device_present("an_event_synchronize_waits_for_the_copy_behind_queued_work") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let event = ctx.event().expect("event");
    let source = testkit::upload_bytes(&ctx, &WORD_BYTES);
    let word = ctx.pinned_word().expect("pinned word");
    let backlog = Backlog::prepare(&ctx);
    let host = backlog.queue(&ctx, &stream);
    let queued_at = Instant::now();
    ctx.copy_word_to_host(&stream, source.range(0, 8).expect("word source"), &word)
        .expect("queue the copy");
    stream.record(&event).expect("record behind the copy");
    let value = word.read_after(&event).expect("read the word");
    report_margin(
        "matmul backlog, event synchronize",
        host,
        queued_at.elapsed(),
    );
    assert_eq!(value, EXPECTED);
    stream.synchronize().expect("drain the stream");
    drop(backlog);
}

#[test]
fn every_successful_launch_counts_one_enqueue_and_a_foreign_stream_counts_none() {
    if !testkit::device_present(
        "every_successful_launch_counts_one_enqueue_and_a_foreign_stream_counts_none",
    ) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let other = CudaContext::new().expect("second context");
    let (stream, foreign) = (
        ctx.stream().expect("stream"),
        other.stream().expect("stream of the other context"),
    );
    assert_eq!(stream.enqueued(), 0, "a fresh stream");

    // The helper itself: it hands the closure's value back and counts one launch.
    let value = ctx.launch_on(&stream, |_| Ok(7)).expect("a launch");
    assert_eq!((value, stream.enqueued()), (7, 1));
    let mut ran = false;
    let refused = ctx.launch_on(&foreign, |_| {
        ran = true;
        Ok(())
    });
    assert_bad_request(refused, "a stream of another context", "stream belongs");
    assert!(!ran, "the launch did not run");
    assert_eq!(
        foreign.enqueued(),
        0,
        "nothing counted on the foreign stream"
    );

    // The two entries go through it: one launch each, however many operations they queue.
    let weight = testkit::upload_weight(&ctx, GgufType::Q4K, 8, 256);
    let x = testkit::upload_f32(&ctx, &testkit::input(1, 256));
    let xf = testkit::upload_filled(&ctx, 0, XF_FRAGMENT);
    let y = testkit::upload_filled(&ctx, 0, 32);
    let xf_range = xf.range(0, XF_FRAGMENT).expect("xf range");
    let pack = PackCall {
        x: x.range(0, 1024).expect("x range"),
        xf: xf_range,
        m: 1,
        k: 256,
    };
    ctx.pack(&stream, pack).expect("queue the pack");
    assert_eq!(stream.enqueued(), 2, "the pack");
    let projection = Projection {
        weight: &weight,
        first_row: 0,
        rows: 8,
    };
    let io = MatmulIo {
        xf: xf_range,
        y: y.range(0, 32).expect("y range"),
        m: 1,
    };
    ctx.matmul(&stream, projection, io)
        .expect("queue the matmul");
    assert_eq!(stream.enqueued(), 3, "the matmul");

    // The pinned word's sequence follows the same count: an event recorded after
    // the matmul but before the copy cannot release the copy, one behind it can.
    let (early, late) = (ctx.event().expect("early"), ctx.event().expect("late"));
    let source = testkit::upload_bytes(&ctx, &WORD_BYTES);
    let word = ctx.pinned_word().expect("pinned word");
    stream.record(&early).expect("record early");
    ctx.copy_word_to_host(&stream, source.range(0, 8).expect("word source"), &word)
        .expect("queue the copy");
    stream.record(&late).expect("record late");
    assert_bad_request(
        word.read_after(&early).map(|_| ()),
        "an event recorded before the copy",
        "before the word copy",
    );
    assert_eq!(word.read_after(&late).expect("read the word"), EXPECTED);
    assert_eq!(stream.enqueued(), 6, "record, copy, record");
}

#[test]
fn launches_run_behind_earlier_work_on_their_stream() {
    if !testkit::device_present("launches_run_behind_earlier_work_on_their_stream") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let backlog = PackBacklog::prepare(&ctx);
    // A fragment-order call, a row-major call and a split-K call, each behind a
    // backlog of packs on the same stream. The backlog is made of packs because a
    // wrong stream moves a whole kind of launch: a matmul backlog would move with
    // the matmul under test and prove nothing.
    // - A matmul that ran on the default stream would start at once and read an xf
    //   that the pack behind the backlog has not written yet.
    // - A pack that ran on the default stream would take the backlog's packs with
    //   it (they use the same entry), so the matmul left on the stream would start
    //   at once and read an xf that is not packed.
    // The warm-up below matters as much as the backlog: with the first backlog of
    // 8192-token packs and no warm-up, both stream mutants survived on the 3090.
    // The limits: the ordering is proved only while the backlog outlasts the time
    // the test needs to queue its own launches (the margin line printed by each
    // case shows it; the test does not assert it because device timing varies), and
    // only while the backlog leaves multiprocessors free for a launch on another
    // stream (see `PackBacklog`).
    let cases = [(512, 1), (512, 33), (4096, 8)];
    for (k, m) in cases {
        let case = format!("k {k} m {m}");
        let weight = testkit::upload_weight(&ctx, GgufType::Q4K, 64, k);
        let x = testkit::input(m, k);
        // Warm-up: the first launch of a kernel loads its module (lazy loading), and
        // that load waits for the whole device. Unwarmed, the first launch of a kernel
        // that went to the wrong stream would wait for the backlog and hide the mistake.
        let warm = Run {
            projection: Projection {
                weight: &weight,
                first_row: 0,
                rows: 32,
            },
            x: &x,
            slack: false,
            backlog: None,
        };
        on_stream(&ctx, &ctx.stream().expect("warm-up stream"), warm);
        let stream = ctx.stream().expect("stream");
        let run = Run {
            projection: Projection {
                weight: &weight,
                first_row: 0,
                rows: 32,
            },
            x: &x,
            slack: false,
            backlog: Some(&backlog),
        };
        let streamed = on_stream(&ctx, &stream, run);
        let reference = testkit::blocking(&ctx, &weight, (0, 32), &x);
        assert!(
            reference.iter().any(|&value| value != 0.0),
            "{case}: the reference is all zero, so the comparison proves nothing"
        );
        assert_eq!(bits(&streamed), bits(&reference), "{case}");
    }
}

#[test]
fn adjacent_ranges_larger_than_the_call_needs_are_accepted() {
    if !testkit::device_present("adjacent_ranges_larger_than_the_call_needs_are_accepted") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let weight = testkit::upload_weight(&ctx, GgufType::Q4K, 64, 256);
    let x = testkit::input(1, 256);
    // One buffer: x (1024 bytes), then xf, then y right after the xf bytes the call
    // uses. The x range and the xf range are larger than the call needs, so the
    // ranges overlap while the bytes the call uses do not.
    let y_at = 1024 + XF_FRAGMENT;
    let mut bytes: Vec<u8> = x.iter().flat_map(|value| value.to_le_bytes()).collect();
    bytes.extend(vec![0xAA; XF_FRAGMENT]);
    bytes.extend(vec![0xFF; 64 + 64]);
    let all = testkit::upload_bytes(&ctx, &bytes);
    let pack = PackCall {
        x: all
            .range(0, 2048)
            .expect("x range, larger than the call needs"),
        xf: all.range(1024, XF_FRAGMENT).expect("xf range"),
        m: 1,
        k: 256,
    };
    ctx.pack(&stream, pack)
        .expect("a pack whose x range reaches into xf");
    let projection = Projection {
        weight: &weight,
        first_row: 0,
        rows: 32,
    };
    let io = MatmulIo {
        xf: all
            .range(1024, XF_FRAGMENT + 64)
            .expect("xf range, larger than the call needs"),
        y: all.range(y_at, 128).expect("y range"),
        m: 1,
    };
    ctx.matmul(&stream, projection, io)
        .expect("a matmul whose xf range reaches into y");
    stream.synchronize().expect("drain the stream");
    let all_bytes = testkit::read_bytes(&ctx, &all, bytes.len());
    let streamed = testkit::floats(&all_bytes[y_at..y_at + 128]);
    let reference = testkit::blocking(&ctx, &weight, (0, 32), &x);
    assert!(reference.iter().any(|&value| value != 0.0));
    assert_eq!(bits(&streamed), bits(&reference));
}

// ---- Q5_0: the repository's own kernel behind `matmul` ----------------------------------------

/// Tolerance class T3 (products with f16 inputs): the NMSE bound per output vector.
const NMSE_BOUND: f64 = 1e-5;
/// Output lengths from which the argmax of a token is compared (the T3 rule: 64 and up).
const ARGMAX_MIN_ROWS: usize = 64;
/// The T3 report line threshold ("report at 1e-6").
const REPORT_NMSE: f64 = 1e-6;
const Q5_0_BLOCK: usize = 22;

/// Random Q5_0 blocks of `rows` x `k`: a finite fp16 `d` of magnitude 2^-5 to 2^4 (so `16 |d|`
/// stays far inside the fp16 range), random `qh` and `qs`.
fn q5_0_source(seed: u64, rows: usize, k: usize) -> Vec<u8> {
    let mut rng = testkit::Rng::new(seed);
    let blocks = rows * (k / 32);
    let mut bytes = Vec::with_capacity(blocks * Q5_0_BLOCK);
    for _ in 0..blocks {
        let exponent = 10 + u16::try_from(rng.below(9)).expect("below 9");
        let mantissa = u16::try_from(rng.below(1024)).expect("below 1024");
        let sign = u16::try_from(rng.below(2)).expect("below 2") << 15;
        bytes.extend_from_slice(&(sign | exponent << 10 | mantissa).to_le_bytes());
        bytes.extend((0..Q5_0_BLOCK - 2).map(|_| (rng.next_u64() >> 32) as u8));
    }
    bytes
}

fn upload_q5_0(ctx: &CudaContext, source: &[u8], rows: usize, k: usize) -> TiledWeight {
    let tensor = QuantTensor {
        name: "q5_0 test weight",
        ty: GgufType::Q50,
        n_rows: rows,
        k,
        source_len: source.len(),
    };
    ctx.upload_tiled(tensor, CHUNK_LIMIT, &mut |offset, buffer| {
        buffer.copy_from_slice(&source[offset..offset + buffer.len()]);
        Ok(())
    })
    .expect("upload of the Q5_0 test weight")
}

/// `x · Wᵀ` per token in f64: the exactly dequantized weights (the CPU Q5_0 decode) times the
/// fp16-rounded activations the pack writes, which is the T3 reference.
fn q5_0_reference(source: &[u8], rows: usize, k: usize, x: &[f32]) -> Vec<f64> {
    let weights = dequantize(GgufType::Q50, source).expect("dequantize the source");
    let mut out = Vec::with_capacity(x.len() / k * rows);
    for token in x.chunks_exact(k) {
        let half: Vec<f64> = token
            .iter()
            .map(|&value| f64::from(testkit::f16_to_f32(testkit::f16_bits(value))))
            .collect();
        for row in weights.chunks_exact(k).take(rows) {
            out.push(row.iter().zip(&half).map(|(w, x)| f64::from(*w) * x).sum());
        }
    }
    out
}

fn normalized_mse(device: &[f32], reference: &[f64]) -> f64 {
    let error: f64 = device
        .iter()
        .zip(reference)
        .map(|(y, r)| (f64::from(*y) - r).powi(2))
        .sum();
    let energy: f64 = reference.iter().map(|r| r.powi(2)).sum();
    assert!(energy > 0.0, "the reference token has no energy");
    error / energy
}

fn argmax_of<T: Copy + PartialOrd>(values: &[T]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// The T3 gate on every token of `device` (`rows` values each): NMSE within [`NMSE_BOUND`] and, from
/// [`ARGMAX_MIN_ROWS`] rows, equal argmax. Prints the maximum NMSE next to the bound and returns it.
fn assert_t3(label: &str, device: &[f32], reference: &[f64], rows: usize) -> f64 {
    assert_eq!(device.len(), reference.len(), "{label}: output length");
    let mut worst = 0.0f64;
    for (token, (got, want)) in device
        .chunks_exact(rows)
        .zip(reference.chunks_exact(rows))
        .enumerate()
    {
        let nmse = normalized_mse(got, want);
        assert!(
            nmse <= NMSE_BOUND,
            "{label}: token {token} NMSE {nmse:e} exceeds the bound {NMSE_BOUND:e}"
        );
        if rows >= ARGMAX_MIN_ROWS {
            assert_eq!(
                argmax_of(got),
                argmax_of(want),
                "{label}: token {token} argmax"
            );
        }
        worst = worst.max(nmse);
    }
    println!("T3 {label}: max NMSE {worst:.3e} (bound {NMSE_BOUND:e})");
    if worst > REPORT_NMSE {
        println!("OVER_1E-6 {label}: max NMSE {worst:.3e}");
    }
    worst
}

/// Packs `x` (m rows of the weight's `k` values) and multiplies rows `rows.0..rows.0 + rows.1` of a
/// Q5_0 `weight` on `stream`, as the kernel contract needs: every staged row is `k` rounded up to
/// the 256 grid, and its tail repeats `tail` (f32 bit patterns). Waits and returns `y`.
fn run_q5_0(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: &TiledWeight,
    rows: (usize, usize),
    x: &[f32],
    tail: &[u32],
) -> Vec<f32> {
    let (k, m) = (weight.k(), x.len() / weight.k());
    let k_x = k.next_multiple_of(256);
    let mut staged = Vec::with_capacity(m * k_x);
    for token in x.chunks_exact(k) {
        staged.extend_from_slice(token);
        staged.extend((0..k_x - k).map(|i| f32::from_bits(tail[i % tail.len()])));
    }
    let need = native::activation_bytes(rows.1, k_x, m).expect("sizes of the call");
    let x_buffer = testkit::upload_f32(ctx, &staged);
    let xf = testkit::upload_filled(ctx, 0xAA, need.xf);
    let y = testkit::upload_filled(ctx, 0xFF, need.y);
    let pack = PackCall {
        x: x_buffer.range(0, need.x).expect("x range"),
        xf: xf.range(0, need.xf).expect("xf range"),
        m,
        k: k_x,
    };
    ctx.pack(stream, pack).expect("queue the pack");
    let projection = Projection {
        weight,
        first_row: rows.0,
        rows: rows.1,
    };
    let io = MatmulIo {
        xf: xf.range(0, need.xf).expect("xf range"),
        y: y.range(0, need.y).expect("y range"),
        m,
    };
    ctx.matmul(stream, projection, io)
        .expect("queue the Q5_0 matmul");
    stream.synchronize().expect("drain the stream");
    testkit::floats(&testkit::read_bytes(ctx, &y, need.y))
}

#[test]
fn q5_0_dense_matmul_matches_the_cpu_reference() {
    const NAME: &str = "q5_0_dense_matmul_matches_the_cpu_reference";
    if !testkit::device_present(NAME) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5e15_0004);
    let mut worst = 0.0f64;
    // K = 896 has the tile-less lanes, 384 the odd single tail, 128 a lone half tile, 1024 no tail.
    for (rows, k) in [(64, 128), (64, 384), (64, 896), (64, 1024), (2304, 896)] {
        let source = q5_0_source(rng.next_u64(), rows, k);
        let weight = upload_q5_0(&ctx, &source, rows, k);
        for m in [1, 8, 9, 16] {
            let x: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
            let device = run_q5_0(&ctx, &stream, &weight, (0, rows), &x, &[0]);
            let reference = q5_0_reference(&source, rows, k, &x);
            let label = format!("q5_0 dense rows={rows} k={k} m={m}");
            worst = worst.max(assert_t3(&label, &device, &reference, rows));
        }
    }
    println!("T3 q5_0 dense: maximum NMSE {worst:.3e} (bound {NMSE_BOUND:e})");
}

/// The fp16 value of weight `(row, column)` that the kernel must dequantize: one rounding of
/// `d * (q5 - 16)`, with `d` an fp16 widened to f32 (the product of an 11-bit and a 5-bit integer is
/// exact in f32, so rounding it to fp16 here is the single rounding of the device).
fn q5_0_fp16_weight(source: &[u8], k: usize, row: usize, column: usize) -> f32 {
    let block = &source[(row * (k / 32) + column / 32) * Q5_0_BLOCK..][..Q5_0_BLOCK];
    let d = testkit::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let e = column % 32;
    let low = if e < 16 {
        block[6 + e] & 0x0F
    } else {
        block[6 + e - 16] >> 4
    };
    let q5 = i32::from(low) | i32::from((qh >> e) & 1 != 0) << 4;
    let product = d * (q5 - 16) as f32;
    testkit::f16_to_f32(testkit::f16_bits(product))
}

#[test]
fn q5_0_one_hot_probe_reads_back_the_fp16_dequantized_weights() {
    const NAME: &str = "q5_0_one_hot_probe_reads_back_the_fp16_dequantized_weights";
    if !testkit::device_present(NAME) {
        return;
    }
    let (rows, k) = (16, 896);
    let mut source = q5_0_source(0x5e15_0005, rows, k);
    // Block 0 has an all-ones qh and block 1 an all-zero qs and qh: the extremes of q5.
    source[2..6].fill(0xFF);
    source[Q5_0_BLOCK + 2..2 * Q5_0_BLOCK].fill(0);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let weight = upload_q5_0(&ctx, &source, rows, k);
    for first in (0..k).step_by(8) {
        let mut x = vec![0.0f32; 8 * k];
        for token in 0..8 {
            x[token * k + first + token] = 1.0;
        }
        let y = run_q5_0(&ctx, &stream, &weight, (0, rows), &x, &[0]);
        for token in 0..8 {
            for row in 0..rows {
                // The accumulator starts at +0 and `-0 + +0` is `+0`, so the device never returns
                // a negative zero: the model's `-0` (a zero weight times a negative d) reads as `+0`.
                let want = q5_0_fp16_weight(&source, k, row, first + token) + 0.0;
                let got = y[token * rows + row];
                assert!(
                    got.to_bits() == want.to_bits(),
                    "weight ({row}, {}) reads back {got:?}, the fp16 model says {want:?}",
                    first + token
                );
            }
        }
    }
}

#[test]
fn q5_0_xf_beyond_k_is_never_read() {
    const NAME: &str = "q5_0_xf_beyond_k_is_never_read";
    if !testkit::device_present(NAME) {
        return;
    }
    let (rows, k) = (64, 896);
    let source = q5_0_source(0x5e15_0008, rows, k);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let weight = upload_q5_0(&ctx, &source, rows, k);
    let mut rng = testkit::Rng::new(0x5e15_0009);
    let finite: Vec<u32> = (0..128)
        .map(|_| (rng.normal() as f32 * 1.0e3).to_bits())
        .collect();
    let tails: [(&str, &[u32]); 4] = [
        ("NaN", &[0x7fc0_0000]),
        ("0xFF bytes", &[0xffff_ffff]),
        ("infinity", &[0x7f80_0000]),
        ("random finite values", &finite),
    ];
    for m in [1, 8, 16] {
        let x: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
        let clean = run_q5_0(&ctx, &stream, &weight, (0, rows), &x, &[0]);
        for (name, tail) in tails {
            let poisoned = run_q5_0(&ctx, &stream, &weight, (0, rows), &x, tail);
            assert_eq!(bits(&poisoned), bits(&clean), "m={m} tail {name}");
        }
    }
}

#[test]
fn q5_0_matmul_wrapper_rejects_unsound_calls_before_launching() {
    const NAME: &str = "q5_0_matmul_wrapper_rejects_unsound_calls_before_launching";
    if !testkit::device_present(NAME) {
        return;
    }
    let (rows, k) = (16, 896);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let weight = upload_q5_0(&ctx, &q5_0_source(0x5e15_000a, rows, k), rows, k);
    let xf = testkit::upload_filled(&ctx, 0xAA, XF_FRAGMENT);
    let y = testkit::upload_filled(&ctx, 0xFF, 1024);
    // Every case refuses before it reads a byte of `y`, so one small range serves all of them.
    let io = |m: usize| MatmulIo {
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        y: y.range(0, 64).expect("y range"),
        m,
    };
    let projection = |first_row, rows| Projection {
        weight: &weight,
        first_row,
        rows,
    };
    assert_bad_request(
        ctx.matmul(&stream, projection(0, 8), io(0)),
        "no tokens",
        "takes 1 to 32768 tokens",
    );
    assert_bad_request(
        ctx.matmul(&stream, projection(0, 8), io(17)),
        "seventeen tokens",
        "Q5_0 decode matmul takes 1 to 16 tokens",
    );
    assert_bad_request(
        ctx.matmul(&stream, projection(4, 8), io(1)),
        "rows that start inside a group",
        "not whole 8-row groups",
    );
    assert_bad_request(
        ctx.matmul(&stream, projection(8, 16), io(1)),
        "rows past the weight",
        "not whole 8-row groups",
    );
    let short_xf = MatmulIo {
        xf: xf.range(0, XF_FRAGMENT - 1).expect("xf range"),
        y: y.range(0, 32).expect("y range"),
        m: 1,
    };
    assert_bad_request(
        ctx.matmul(&stream, projection(0, 8), short_xf),
        "xf one byte short",
        "xf range holds",
    );
    let short_y = MatmulIo {
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        y: y.range(0, 28).expect("y range"),
        m: 1,
    };
    assert_bad_request(
        ctx.matmul(&stream, projection(0, 8), short_y),
        "y one float short",
        "y range holds",
    );
    let y_over_xf = MatmulIo {
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        y: xf.range(XF_FRAGMENT - 32, 32).expect("y range"),
        m: 1,
    };
    assert_bad_request(
        ctx.matmul(&stream, projection(0, 8), y_over_xf),
        "y over the end of xf",
        "y and xf ranges overlap",
    );
    let y_in_weight = MatmulIo {
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        y: weight.buffer().range(0, 32).expect("y range"),
        m: 1,
    };
    assert_bad_request(
        ctx.matmul(&stream, projection(0, 8), y_in_weight),
        "y inside the weight tiles",
        "y and weight ranges overlap",
    );
    assert_eq!(stream.enqueued(), 0, "no refusal queued anything");
    assert!(
        testkit::read_bytes(&ctx, &y, 1024)
            .iter()
            .all(|&b| b == 0xFF),
        "y untouched"
    );
    let other = CudaContext::new().expect("second context");
    let foreign = other.stream().expect("stream of the other context");
    assert_bad_request(
        ctx.matmul(&foreign, projection(0, 8), io(1)),
        "a stream of another context",
        "stream belongs",
    );
    ctx.ensure_usable()
        .expect("the refusals leave the backend usable");
}
mod repeat;
