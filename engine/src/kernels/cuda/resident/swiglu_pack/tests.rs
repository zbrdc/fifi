//! Seam-1 device tests (3090) of the SwiGLU + fp16 pack wrapper: the Rust wrapper against the C
//! entry against the vendored kernel, compared with a float64 host reference.
//!
//! Design guidance only, no code copied (no MegaCapybara-derived design).
//!
//! The reference is `silu_f64(gate) * up` in f64 from the exact f32 inputs, and the bound is the
//! lane M class E2 (REPLY-44, REPLY-73): the fp16 result equals the round to nearest even of the
//! reference, or the adjacent value with the reference within 2^-20 (relative) of the rounding
//! midpoint (`testkit::e2_holds`). It is a measured domain, not an analytic bound: any finite `up`
//! for `|gate| <= 12`, and `up` of sd up to 100 for `12 < |gate| <= 219`. The 15 E2 misses at `up`
//! sd 10^4 in `|gate|` [-30, -12] (expf probe, rev 3 evidence) are recorded as the measured edge
//! of the domain, not tuned away. The kernel assumption is any finite up for |gate| <= 12, and up
//! sd <= 100 for 12 < |gate| <= 219 (lane M, 2026-10-10).
//!
//! The device suite runs with `--test-threads=1` (decision 0040). A NaN is tested with
//! `f32::is_nan`, never bitwise; everything else with `to_bits`; whole byte buffers through
//! `testkit::fnv1a64`.
//!
//! Every printed maximum is labelled with the build profile: the cargo debug profile gives nvcc
//! `-G -O0`, so the release run is the one that speaks for the shipped kernel.

use super::{MAX_TOKENS, MAX_WIDTH, SwigluPackCall, check_shape, gu_bytes};
use crate::EngError;
use crate::kernels::cuda::native::xf_bytes;
use crate::kernels::cuda::testkit::{
    self, Rng, blocking, e2_holds, f16_to_f32, f64_to_f16_bits, floats, fnv1a64, read_bytes,
    silu_f64, unpack_xf, upload_bytes, upload_f32, upload_filled, upload_weight,
};
use crate::kernels::cuda::{
    CapturePlan, CudaContext, DeviceBuffer, DeviceStream, MatmulIo, PackCall, Projection,
    TiledWeight,
};
use crate::weights::GgufType;

const PROFILE: &str = if cfg!(debug_assertions) {
    "debug (-G -O0)"
} else {
    "release"
};

/// A context and its stream.
struct Dev {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Dev {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }
}

/// Rows, live columns and padded width of one call.
#[derive(Clone, Copy, Debug)]
struct Shape {
    m: usize,
    n_ff: usize,
    k_pad: usize,
}

/// Standard deviations of the gate and the up values.
#[derive(Clone, Copy)]
struct Scale {
    gate: f64,
    up: f64,
}

/// What the device holds after a call: the bytes of `gu` and of `xf`.
struct After {
    gu: Vec<u8>,
    xf: Vec<u8>,
}

/// `gu` of `shape`: live columns from `rng`, the tails NaN, so a missed zero-fill puts NaN into `xf`.
fn make_gu(rng: &mut Rng, shape: Shape, scale: Scale) -> Vec<f32> {
    let mut gu = vec![f32::NAN; shape.m * 2 * shape.k_pad];
    for t in 0..shape.m {
        let row = t * 2 * shape.k_pad;
        for i in 0..shape.n_ff {
            gu[row + i] = (rng.normal() * scale.gate) as f32;
            gu[row + shape.k_pad + i] = (rng.normal() * scale.up) as f32;
        }
    }
    gu
}

/// Uploads `gu`, fills `xf` with `xf_fill`, runs the call (padded form when `n_ff < k_pad`) and
/// returns what the device holds afterwards.
fn run_case(dev: &Dev, shape: Shape, gu: &[f32], xf_fill: u8) -> After {
    let gu_len = gu.len() * 4;
    let xf_len = xf_bytes(shape.m, shape.k_pad).expect("xf size");
    let gu_buffer = upload_f32(&dev.ctx, gu);
    let xf_buffer = upload_filled(&dev.ctx, xf_fill, xf_len);
    let call = SwigluPackCall {
        gu: gu_buffer.range(0, gu_len).expect("gu range"),
        xf: xf_buffer.range(0, xf_len).expect("xf range"),
        m: shape.m,
        n_ff: shape.n_ff,
    };
    if shape.n_ff == shape.k_pad {
        dev.ctx.swiglu_pack(&dev.stream, call)
    } else {
        dev.ctx.swiglu_pack_padded(&dev.stream, call, shape.k_pad)
    }
    .expect("swiglu_pack");
    dev.stream.synchronize().expect("drain the stream");
    After {
        gu: read_bytes(&dev.ctx, &gu_buffer, gu_len),
        xf: read_bytes(&dev.ctx, &xf_buffer, xf_len),
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The 16 bytes of the shared zero tail entry `xf[kXfEntries]` (65,536 data entries, then the
/// tail): every padding lane of `mma884::load_x` reads it. MEMSET-TRIM: the pack kernel writes
/// it itself, so no whole-buffer memset runs.
fn tail(xf: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&xf[65_536 * 16..65_536 * 16 + 16]);
    out
}

/// The float64 reference of one element.
fn ref_swiglu(gate: f32, up: f32) -> f64 {
    silu_f64(f64::from(gate)) * f64::from(up)
}

/// Counts of the E2 check of the live columns.
#[derive(Default)]
struct Stats {
    checked: usize,
    misses: usize,
    one_step: usize,
}

/// E2 check of every live element of `xf` against the reference of `gu`.
fn check_live(shape: Shape, gu: &[f32], xf: &[u8], stats: &mut Stats) {
    let table = unpack_xf(xf, shape.m, shape.k_pad);
    for t in 0..shape.m {
        let row = t * 2 * shape.k_pad;
        for i in 0..shape.n_ff {
            let reference = ref_swiglu(gu[row + i], gu[row + shape.k_pad + i]);
            let bits = table[t * shape.k_pad + i];
            stats.checked += 1;
            if !e2_holds(bits, reference) {
                stats.misses += 1;
            } else if bits != f64_to_f16_bits(reference) {
                stats.one_step += 1;
            }
        }
    }
}

/// Runs one group of calls (`seeds` seeds) of one unpadded shape and returns the E2 counts. Also
/// asserts that `gu` came back unchanged.
fn run_group(dev: &Dev, shape: Shape, scale: Scale, seeds: std::ops::Range<u64>) -> Stats {
    let mut stats = Stats::default();
    for seed in seeds {
        let mut rng = Rng::new(0x5a1a_0000 + seed);
        let gu = make_gu(&mut rng, shape, scale);
        let after = run_case(dev, shape, &gu, 0xFF);
        assert_eq!(
            fnv1a64(&after.gu),
            fnv1a64(&to_bytes(&gu)),
            "gu changed by an unpadded call, m {} n_ff {} seed {seed}",
            shape.m,
            shape.n_ff
        );
        check_live(shape, &gu, &after.xf, &mut stats);
    }
    stats
}

fn unpadded(m: usize, n_ff: usize) -> Shape {
    Shape {
        m,
        n_ff,
        k_pad: n_ff,
    }
}

/// Prints the counts of a group and returns its misses.
fn report(label: &str, stats: &Stats) -> usize {
    println!(
        "[{PROFILE}] swiglu_pack {label}: {} elements, {} E2 misses, {} one-step",
        stats.checked, stats.misses, stats.one_step
    );
    stats.misses
}

#[test]
fn swiglu_pack_matches_the_float64_reference() {
    if !testkit::device_present("swiglu_pack_matches_the_float64_reference") {
        return;
    }
    let dev = Dev::new();
    let model = Scale { gate: 1.5, up: 1.0 };
    let mut misses = 0;
    let mut main = Stats::default();
    for m in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
        for n_ff in [256, 512, 1024] {
            let part = run_group(&dev, unpadded(m, n_ff), model, 0..1);
            main.checked += part.checked;
            main.misses += part.misses;
            main.one_step += part.one_step;
        }
    }
    misses += report("main shapes", &main);
    let seeds = run_group(&dev, unpadded(1, 512), model, 100..160);
    misses += report("60 seeds at m 1, 512", &seeds);
    let wide_up = run_group(
        &dev,
        unpadded(8, 512),
        Scale {
            gate: 3.0,
            up: 100.0,
        },
        0..4,
    );
    misses += report("gate sd 3, up sd 100", &wide_up);
    let zero_gate = run_group(&dev, unpadded(8, 512), Scale { gate: 0.0, up: 1.0 }, 0..2);
    misses += report("gate 0", &zero_gate);
    for m in [1, 17] {
        let widest = run_group(&dev, unpadded(m, 32_768), model, 0..1);
        misses += report(&format!("n_ff 32768, m {m}"), &widest);
    }
    for sd in [6.0, 12.0, 60.0] {
        let stress = run_group(&dev, unpadded(5, 1024), Scale { gate: sd, up: 1.0 }, 0..1);
        misses += report(&format!("stress gate sd {sd}"), &stress);
    }
    assert_eq!(misses, 0, "E2 misses over all groups");
}

#[test]
fn swiglu_pack_bounds_are_pinned() {
    for (m, n_ff, k_pad) in [
        (1, 256, 256),
        (1, 1, 256),
        (5, 255, 256),
        (32_768, 32_768, 32_768),
        (1, 896, 1024),
    ] {
        check_shape(m, n_ff, k_pad).expect("an edge that is accepted");
    }
    let wrapped = (1usize << 32) + 256;
    let refused: [(usize, usize, usize, &str); 18] = [
        (0, 256, 256, "m"),
        (32_769, 256, 256, "m"),
        (usize::MAX, 256, 256, "m"),
        ((1 << 32) + 1, 256, 256, "m"),
        (1, 256, 0, "k_pad"),
        (1, 255, 255, "k_pad"),
        (1, 256, 257, "k_pad"),
        (1, 256, 300, "k_pad"),
        (1, 256, 33_024, "k_pad"),
        (1, 256, usize::MAX, "k_pad"),
        (1, 256, wrapped, "k_pad"),
        (1, 0, 256, "n_ff"),
        (1, 257, 256, "n_ff"),
        (1, usize::MAX, 256, "n_ff"),
        (1, (1 << 32) + 1, 256, "n_ff"),
        (1, 1025, 1024, "n_ff"),
        (1, 1, 128, "k_pad"),
        (1, 1, 32_768 + 256, "k_pad"),
    ];
    for (m, n_ff, k_pad, tag) in refused {
        let needle = format!("swiglu_pack: {tag}:");
        match check_shape(m, n_ff, k_pad) {
            Err(EngError::BadRequest(message)) if message.contains(&needle) => {}
            other => panic!("m {m} n_ff {n_ff} k_pad {k_pad}: expected {needle}, got {other:?}"),
        }
    }
    assert_eq!((MAX_TOKENS, MAX_WIDTH), (32_768, 32_768));
    assert_eq!(gu_bytes(1, 512), Some(4096));
    assert_eq!(gu_bytes(17, 1024), Some(139_264));
    assert_eq!(gu_bytes(usize::MAX, 512), None);
    assert_eq!(xf_bytes(1, 512), Some(1_048_592));
    assert_eq!(xf_bytes(17, 512), Some(17_408));
    pin_testkit_helpers();
}

/// The shared helpers this packet relies on, pinned at their edges.
fn pin_testkit_helpers() {
    let tie_up = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
    assert_eq!(f64_to_f16_bits(tie_up), 0x3c01);
    assert_eq!(f64_to_f16_bits(1.0 + 2f64.powi(-11)), 0x3c00);
    assert_eq!(f64_to_f16_bits(65_520.0), 0x7c00);
    assert_eq!(f64_to_f16_bits(2f64.powi(-25)), 0);
    assert_eq!(f64_to_f16_bits(-0.0), 0x8000);
    assert_eq!(f64_to_f16_bits(f64::NAN), 0x7e00);
    assert_eq!(f64_to_f16_bits(-1e9), 0xfc00);
    let midpoint = 1.0 + 2f64.powi(-11);
    assert!(
        e2_holds(0x3c01, midpoint * (1.0 - 1e-7)),
        "inside the window"
    );
    assert!(
        !e2_holds(0x3c01, midpoint * (1.0 - 1e-5)),
        "outside the window"
    );
    assert!(!e2_holds(0x3c02, 1.0), "two steps away");
    assert!(!e2_holds(0x7bff, 70_000.0), "70000 against 65504");
    assert!(e2_holds(0x7c00, 65_519.999_9), "the overflow midpoint");
    assert!(!e2_holds(0x7c00, 65_500.0), "below the overflow window");
    assert!(
        e2_holds(0x8000, 0.0) && e2_holds(0x0000, -0.0),
        "the two zeros"
    );
    assert!(!e2_holds(0x3c00, f64::NAN));
    assert!((silu_f64(1.0) - 0.731_058_578_630_004_9).abs() < 1e-15);
    let tiny = silu_f64(-100.0);
    assert!(
        tiny < 0.0 && tiny > -1e-40,
        "a signed tiny value, got {tiny}"
    );
    assert_eq!(silu_f64(0.0), 0.0);
    pin_unpack_xf();
}

/// `unpack_xf` on a hand-built fragment-order buffer (token 2, lane 6, q 1, wi 3) and a
/// row-major one.
fn pin_unpack_xf() {
    let mut fragment = vec![0u8; 1_048_592];
    let vec_index = 3 * 32 + 6;
    for j in 0..4u16 {
        for (half, value) in [(j, 0x1000 + 76 + j), (4 + j, 0x1000 + 108 + j)] {
            let at = vec_index * 16 + 2 * usize::from(half);
            fragment[at..at + 2].copy_from_slice(&value.to_le_bytes());
        }
    }
    let table = unpack_xf(&fragment, 3, 256);
    for j in 0..4 {
        assert_eq!(table[2 * 256 + 76 + j], 0x1000 + 76 + j as u16);
        assert_eq!(table[2 * 256 + 108 + j], 0x1000 + 108 + j as u16);
    }
    assert_eq!(table.iter().filter(|&&v| v != 0).count(), 8);
    let row_major: Vec<u8> = (0..17 * 256u16)
        .flat_map(|i| i.wrapping_mul(3).to_le_bytes())
        .collect();
    let table = unpack_xf(&row_major, 17, 256);
    for (i, &value) in table.iter().enumerate() {
        assert_eq!(value, (i as u16).wrapping_mul(3), "row-major element {i}");
    }
}

/// Rows of the weight in the consumer check of the padded width.
const WEIGHT_ROWS: usize = 64;

/// The `xf` a pack makes of `x` (`[m][k]` f32), read back.
fn pack_reference(dev: &Dev, x: &[f32], shape: Shape) -> Vec<u8> {
    let x_buffer = upload_f32(&dev.ctx, x);
    let xf_len = xf_bytes(shape.m, shape.k_pad).expect("xf size");
    let xf_buffer = upload_filled(&dev.ctx, 0xFF, xf_len);
    let call = PackCall {
        x: x_buffer.range(0, x.len() * 4).expect("x range"),
        xf: xf_buffer.range(0, xf_len).expect("xf range"),
        m: shape.m,
        k: shape.k_pad,
    };
    dev.ctx.pack(&dev.stream, call).expect("pack");
    dev.stream.synchronize().expect("drain the pack");
    read_bytes(&dev.ctx, &xf_buffer, xf_len)
}

/// `y` of the matmul of a Q4_K weight over the fragment-order or row-major `xf` bytes.
fn consumer_y(dev: &Dev, weight: &TiledWeight, xf: &[u8], m: usize) -> Vec<u8> {
    let xf_buffer: DeviceBuffer = upload_bytes(&dev.ctx, xf);
    let y_len = m * WEIGHT_ROWS * 4;
    let y_buffer = upload_filled(&dev.ctx, 0xFF, y_len);
    let projection = Projection {
        weight,
        first_row: 0,
        rows: WEIGHT_ROWS,
    };
    let io = MatmulIo {
        xf: xf_buffer.range(0, xf.len()).expect("xf range"),
        y: y_buffer.range(0, y_len).expect("y range"),
        m,
    };
    dev.ctx.matmul(&dev.stream, projection, io).expect("matmul");
    dev.stream.synchronize().expect("drain the matmul");
    read_bytes(&dev.ctx, &y_buffer, y_len)
}

/// One shape of the padded-width contract (MEL-C1b): tails of `gu` zeroed, tail of `xf` exactly
/// `0x0000`, bytes equal to the pack of `[h | 0]`, the same result onto two garbage patterns and
/// the same consumer output.
fn check_padded(dev: &Dev, shape: Shape, stats: &mut Stats) {
    let mut rng = Rng::new(0x7ad0_0000 + (shape.n_ff * 7 + shape.m) as u64);
    let gu = make_gu(&mut rng, shape, Scale { gate: 1.5, up: 1.0 });
    let tail_columns = |row: usize, half: usize| {
        (shape.n_ff..shape.k_pad).map(move |i| row * 2 * shape.k_pad + half * shape.k_pad + i)
    };
    for t in 0..shape.m {
        assert!(
            (0..2).all(|half| tail_columns(t, half).all(|at| gu[at].is_nan())),
            "the tails start as NaN"
        );
    }
    let first = run_case(dev, shape, &gu, 0xFF);
    let second = run_case(dev, shape, &gu, 0x5A);
    // MEMSET-TRIM: only the data lanes and the tail are written, so the two fills are compared on
    // what a reader consumes (the unpacked rows and the tail), not on the unread fill bytes.
    assert_eq!(
        unpack_xf(&first.xf, shape.m, shape.k_pad),
        unpack_xf(&second.xf, shape.m, shape.k_pad),
        "{shape:?}: the 0xFF and 0x5A data lanes"
    );
    // MEMSET-TRIM: the tail entry exists only in the fragment-order buffer (m <= 16); above that
    // the buffer is row-major and fully written.
    if shape.m <= 16 {
        assert_eq!(
            tail(&first.xf),
            [0u8; 16],
            "{shape:?}: the tail entry kept garbage"
        );
    }
    let after = floats(&first.gu);
    for (at, (&got, &sent)) in after.iter().zip(&gu).enumerate() {
        // Only a tail starts as NaN, and it must come back as +0.0; a live word is unchanged.
        let want = if sent.is_nan() { 0 } else { sent.to_bits() };
        assert_eq!(got.to_bits(), want, "{shape:?}: gu word {at}");
    }
    let table = unpack_xf(&first.xf, shape.m, shape.k_pad);
    let mut x = vec![0f32; shape.m * shape.k_pad];
    for t in 0..shape.m {
        for i in 0..shape.k_pad {
            let bits = table[t * shape.k_pad + i];
            assert!(
                i < shape.n_ff || bits == 0,
                "{shape:?}: tail column {i} is {bits:#06x}"
            );
            x[t * shape.k_pad + i] = f16_to_f32(bits);
        }
    }
    check_live(shape, &gu, &first.xf, stats);
    // MEMSET-TRIM: `pack` still zeroes its whole output buffer, so the comparison with the pack
    // of `[h | 0]` covers the unpacked rows and the tail, the bytes a reader consumes.
    let reference = pack_reference(dev, &x, shape);
    assert_eq!(
        unpack_xf(&first.xf, shape.m, shape.k_pad),
        unpack_xf(&reference, shape.m, shape.k_pad),
        "{shape:?}: xf rows against the pack of [h | 0]"
    );
    if shape.m <= 16 {
        assert_eq!(
            tail(&first.xf),
            tail(&reference),
            "{shape:?}: the tail against the pack of [h | 0]"
        );
    }
    let weight = upload_weight(&dev.ctx, GgufType::Q4K, WEIGHT_ROWS, shape.k_pad);
    assert_eq!(
        consumer_y(dev, &weight, &first.xf, shape.m),
        consumer_y(dev, &weight, &second.xf, shape.m),
        "{shape:?}: the matmul over the two results"
    );
}

#[test]
fn swiglu_pack_pads_with_exact_zeros_for_mellum_shapes() {
    if !testkit::device_present("swiglu_pack_pads_with_exact_zeros_for_mellum_shapes") {
        return;
    }
    let dev = Dev::new();
    let mut stats = Stats::default();
    let shapes = [
        (896, 1024),
        (512, 768),
        (256, 512),
        (8, 256),
        (1, 256),
        (255, 256),
        (1000, 1024),
        (1536, 2048),
        (512, 512),
    ];
    for (n_ff, k_pad) in shapes {
        for m in [1, 17] {
            check_padded(&dev, Shape { m, n_ff, k_pad }, &mut stats);
        }
    }
    assert_eq!(
        report("padded shapes", &stats),
        0,
        "E2 misses of the live columns"
    );
}

/// Packs `gu` of `shape` into the given `xf` buffer and returns its bytes: the buffer keeps its
/// old bytes wherever the kernel writes none (MEMSET-TRIM: the lanes of an earlier call stay
/// stale, exactly as in an aliased arena slot).
fn pack_into(dev: &Dev, shape: Shape, gu: &[f32], xf: &DeviceBuffer) -> Vec<u8> {
    let gu_buffer = upload_f32(&dev.ctx, gu);
    let xf_len = xf_bytes(shape.m, shape.k_pad).expect("xf size");
    let call = SwigluPackCall {
        gu: gu_buffer.range(0, gu.len() * 4).expect("gu range"),
        xf: xf.range(0, xf_len).expect("xf range"),
        m: shape.m,
        n_ff: shape.n_ff,
    };
    if shape.n_ff == shape.k_pad {
        dev.ctx.swiglu_pack(&dev.stream, call)
    } else {
        dev.ctx.swiglu_pack_padded(&dev.stream, call, shape.k_pad)
    }
    .expect("swiglu_pack");
    dev.stream.synchronize().expect("drain the stream");
    read_bytes(&dev.ctx, xf, xf_len)
}

/// MEMSET-TRIM test 1: with the whole `xf` buffer pre-filled with garbage (0xFF, the tail
/// included) the pack itself must leave (a) the shared zero tail entry zero, (b) the live columns
/// at the float64 reference, and (c) a following `matmul` bitwise equal to the same matmul over a
/// buffer prepared the old way (zeroed, then packed). `m = 16` at `k_pad = 32768` covers the top
/// data entry `kXfEntries - 1` (token 15's last tile), which an off-by-one tail write would
/// clobber; the padded shape covers the 2-D tail memsets path.
#[test]
fn swiglu_pack_writes_the_zero_tail_over_garbage() {
    if !testkit::device_present("swiglu_pack_writes_the_zero_tail_over_garbage") {
        return;
    }
    let dev = Dev::new();
    let mut stats = Stats::default();
    let mut shapes: Vec<Shape> = [1, 3, 8, 9, 16].map(|m| unpadded(m, 512)).to_vec();
    shapes.push(unpadded(16, 32_768));
    shapes.push(Shape {
        m: 3,
        n_ff: 896,
        k_pad: 1024,
    });
    for shape in shapes {
        let mut rng = Rng::new(0x7bd0_0000 + (shape.m * 97 + shape.n_ff) as u64);
        let gu = make_gu(&mut rng, shape, Scale { gate: 1.5, up: 1.0 });
        let first = run_case(&dev, shape, &gu, 0xFF);
        let second = run_case(&dev, shape, &gu, 0x00);
        assert_eq!(
            tail(&first.xf),
            [0u8; 16],
            "{shape:?}: the tail entry kept garbage"
        );
        check_live(shape, &gu, &first.xf, &mut stats);
        assert_eq!(
            unpack_xf(&first.xf, shape.m, shape.k_pad),
            unpack_xf(&second.xf, shape.m, shape.k_pad),
            "{shape:?}: the data lanes depend on the fill"
        );
        let weight = upload_weight(&dev.ctx, GgufType::Q4K, WEIGHT_ROWS, shape.k_pad);
        let ya = consumer_y(&dev, &weight, &first.xf, shape.m);
        let yb = consumer_y(&dev, &weight, &second.xf, shape.m);
        if shape.k_pad == 32_768 {
            // At k = 32768 the matmul splits K 16 ways with atomicAdd accumulation, so two runs
            // over identical buffers differ in the low bits (see the add_norm_pack twin); a
            // wrong lane moves a partial sum wholesale, orders above this tolerance.
            let (dev, ref_max) = (floats(&ya), floats(&yb));
            let scale = ref_max.iter().fold(1.0f32, |top, v| top.max(v.abs()));
            let worst = dev
                .iter()
                .zip(&ref_max)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= 1e-3 * scale,
                "{shape:?}: the matmul over garbage differs by {worst:e} (scale {scale:e})"
            );
            println!(
                "[{PROFILE}] MEMSET-TRIM swiglu_pack {shape:?}: split-K matmul, max |d| {worst:e} of scale {scale:e}"
            );
        } else {
            assert_eq!(
                ya, yb,
                "{shape:?}: the matmul over garbage differs from the old way"
            );
        }
    }
    assert_eq!(report("memset-trim garbage", &stats), 0, "E2 misses");
}

/// MEMSET-TRIM test 2: a pack of `m = 3` straight into a pack of `m = 1` on the same buffer, with
/// no re-zeroing between the calls; the `m = 1` live columns, the tail and the matmul must not
/// depend on the stale lanes of tokens 1 and 2.
#[test]
fn swiglu_pack_ignores_stale_padding_between_launches() {
    if !testkit::device_present("swiglu_pack_ignores_stale_padding_between_launches") {
        return;
    }
    let dev = Dev::new();
    let shape = unpadded(1, 512);
    let xf_len = xf_bytes(shape.m, shape.k_pad).expect("xf size");
    let xf = upload_filled(&dev.ctx, 0xFF, xf_len);
    let mut rng = Rng::new(0x7bd0_1000);
    let scale = Scale { gate: 1.5, up: 1.0 };
    let stale = make_gu(&mut rng, unpadded(3, shape.n_ff), scale);
    let _ = pack_into(&dev, unpadded(3, shape.n_ff), &stale, &xf);
    let gu = make_gu(&mut rng, shape, scale);
    let bytes = pack_into(&dev, shape, &gu, &xf);
    assert_eq!(
        tail(&bytes),
        [0u8; 16],
        "the tail entry is not zero after m=3 then m=1"
    );
    let mut stats = Stats::default();
    check_live(shape, &gu, &bytes, &mut stats);
    assert_eq!(stats.misses, 0, "E2 misses after m=3 then m=1");
    let old = run_case(&dev, shape, &gu, 0x00);
    assert_eq!(
        unpack_xf(&bytes, shape.m, shape.k_pad),
        unpack_xf(&old.xf, shape.m, shape.k_pad),
        "the m=1 data lanes depend on the stale bytes"
    );
    let weight = upload_weight(&dev.ctx, GgufType::Q4K, WEIGHT_ROWS, shape.k_pad);
    assert_eq!(
        consumer_y(&dev, &weight, &bytes, shape.m),
        consumer_y(&dev, &weight, &old.xf, shape.m),
        "the matmul over the stale buffer differs from the old way"
    );
}

/// Offset and bytes of a range inside its buffer.
type Geo = (usize, usize);

/// One call that must be refused: the call arguments (`m`, `n_ff`, `k_pad`), the two ranges and
/// whether both lie in one allocation. The buffers are sized for the geometry of the row.
struct Row {
    name: &'static str,
    needle: &'static str,
    call: (usize, usize, usize),
    gu: Geo,
    xf: Geo,
    shared: bool,
}

/// The ranges that fit `m` rows of width `k_pad`; geometry too large to allocate falls back to the
/// smallest valid one, since the shape check refuses it before any size is looked at.
fn fitting(m: usize, k_pad: usize) -> (Geo, Geo) {
    let (m, k_pad) = if m > 40_000 || k_pad > 40_000 {
        (1, 256)
    } else {
        (m.max(1), k_pad.max(256))
    };
    let gu = gu_bytes(m, k_pad).expect("gu size");
    ((0, gu), (0, xf_bytes(m, k_pad).expect("xf size")))
}

/// A refusal of the shape of the call, buffers sized for it.
fn shape_row(name: &'static str, tag: &'static str, call: (usize, usize, usize)) -> Row {
    let (gu, xf) = fitting(call.0, call.2);
    Row {
        name,
        needle: tag,
        call,
        gu,
        xf,
        shared: false,
    }
}

/// A refusal of a range of the valid base call `m` 1, `n_ff` = `k_pad` 256.
fn range_row(name: &'static str, needle: &'static str, gu: Geo, xf: Geo, shared: bool) -> Row {
    Row {
        name,
        needle,
        call: (1, 256, 256),
        gu,
        xf,
        shared,
    }
}

const XF_FRAGMENT: usize = 1_048_592;

fn refusal_rows() -> Vec<Row> {
    let (gu, xf) = fitting(1, 256);
    let (gu17, xf17) = fitting(17, 256);
    vec![
        shape_row("m 0", "swiglu_pack: m:", (0, 256, 256)),
        shape_row("m 32769, sized", "swiglu_pack: m:", (32_769, 256, 256)),
        shape_row("m usize::MAX", "swiglu_pack: m:", (usize::MAX, 256, 256)),
        shape_row(
            "m that wraps a C int",
            "swiglu_pack: m:",
            ((1 << 32) + 1, 256, 256),
        ),
        shape_row("k_pad 0", "swiglu_pack: k_pad:", (1, 256, 0)),
        shape_row("k_pad 255", "swiglu_pack: k_pad:", (1, 255, 255)),
        shape_row("k_pad 257", "swiglu_pack: k_pad:", (1, 256, 257)),
        shape_row("k_pad 300", "swiglu_pack: k_pad:", (1, 256, 300)),
        shape_row(
            "k_pad 33024, sized",
            "swiglu_pack: k_pad:",
            (1, 256, 33_024),
        ),
        shape_row(
            "k_pad usize::MAX",
            "swiglu_pack: k_pad:",
            (1, 256, usize::MAX),
        ),
        shape_row(
            "k_pad that wraps a C int",
            "swiglu_pack: k_pad:",
            (1, 256, (1 << 32) + 256),
        ),
        shape_row("n_ff 0", "swiglu_pack: n_ff:", (1, 0, 256)),
        shape_row("n_ff above k_pad", "swiglu_pack: n_ff:", (1, 1025, 1024)),
        shape_row(
            "n_ff usize::MAX",
            "swiglu_pack: n_ff:",
            (1, usize::MAX, 256),
        ),
        shape_row(
            "n_ff that wraps a C int",
            "swiglu_pack: n_ff:",
            (1, (1 << 32) + 1, 256),
        ),
        range_row(
            "gu one float short",
            "gu range holds",
            (0, gu.1 - 4),
            xf,
            false,
        ),
        range_row(
            "xf one uint4 short",
            "xf range holds",
            gu,
            (0, xf.1 - 16),
            false,
        ),
        range_row(
            "xf of m * k_pad * 2 bytes at m 1",
            "xf range holds",
            gu,
            (0, 512),
            false,
        ),
        Row {
            name: "xf one uint4 short at m 17",
            needle: "xf range holds",
            call: (17, 256, 256),
            gu: gu17,
            xf: (0, xf17.1 - 16),
            shared: false,
        },
        range_row(
            "gu at offset 2",
            "gu range starts at offset 2",
            (2, gu.1),
            xf,
            false,
        ),
        range_row(
            "xf at offset 4",
            "xf range starts at offset 4",
            gu,
            (4, xf.1),
            false,
        ),
        range_row(
            "xf at offset 8",
            "xf range starts at offset 8",
            gu,
            (8, xf.1),
            false,
        ),
        range_row(
            "xf equal to gu",
            "ranges overlap",
            (0, XF_FRAGMENT),
            (0, XF_FRAGMENT),
            true,
        ),
        range_row(
            "xf inside gu",
            "ranges overlap",
            (0, 2 * XF_FRAGMENT),
            (16, XF_FRAGMENT),
            true,
        ),
        range_row(
            "gu inside xf",
            "ranges overlap",
            (64, gu.1),
            (0, XF_FRAGMENT),
            true,
        ),
    ]
}

/// Runs the padded form of the call on fresh buffers. Returns the result and whether the contents
/// of both buffers are byte for byte what they were before.
fn attempt(dev: &Dev, row: &Row) -> (Result<(), EngError>, bool) {
    let gu_end = row.gu.0 + row.gu.1;
    let xf_end = row.xf.0 + row.xf.1;
    let first_len = if row.shared {
        gu_end.max(xf_end)
    } else {
        gu_end
    };
    let first = upload_filled(&dev.ctx, 0xA5, first_len);
    let second = (!row.shared).then(|| upload_filled(&dev.ctx, 0x5A, xf_end));
    let xf_owner = second.as_ref().unwrap_or(&first);
    let hashes = || {
        (
            fnv1a64(&read_bytes(&dev.ctx, &first, first_len)),
            fnv1a64(&read_bytes(
                &dev.ctx,
                xf_owner,
                if row.shared { first_len } else { xf_end },
            )),
        )
    };
    let before = hashes();
    let (m, n_ff, k_pad) = row.call;
    let call = SwigluPackCall {
        gu: first.range(row.gu.0, row.gu.1).expect("gu range"),
        xf: xf_owner.range(row.xf.0, row.xf.1).expect("xf range"),
        m,
        n_ff,
    };
    let result = dev.ctx.swiglu_pack_padded(&dev.stream, call, k_pad);
    dev.stream.synchronize().expect("drain the stream");
    (result, before == hashes())
}

/// The ranges of a call whose arguments are valid: each must be accepted.
fn accepted_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    let mut add = |name, call: (usize, usize, usize), gu: Geo, xf: Geo, shared| {
        rows.push(Row {
            name,
            needle: "",
            call,
            gu,
            xf,
            shared,
        });
    };
    let (gu, xf) = fitting(1, 512);
    add("n_ff equal to k_pad", (1, 512, 512), gu, xf, false);
    let (gu, xf) = fitting(1, 256);
    add("n_ff 1, padded", (1, 1, 256), gu, xf, false);
    add("m 1, k_pad 256", (1, 256, 256), gu, xf, false);
    add(
        "xf directly behind gu",
        (1, 256, 256),
        (0, gu.1),
        (gu.1, xf.1),
        true,
    );
    add(
        "larger ranges",
        (1, 256, 256),
        (0, gu.1 + 64),
        (0, xf.1 + 64),
        false,
    );
    let (gu, xf) = fitting(32_768, 256);
    add("m 32768", (32_768, 256, 256), gu, xf, false);
    let (gu, xf) = fitting(1, 32_768);
    add("k_pad 32768", (1, 32_768, 32_768), gu, xf, false);
    rows
}

#[test]
fn swiglu_pack_rejects_bad_arguments_before_any_launch() {
    if !testkit::device_present("swiglu_pack_rejects_bad_arguments_before_any_launch") {
        return;
    }
    let dev = Dev::new();
    for row in refusal_rows() {
        let (result, unchanged) = attempt(&dev, &row);
        match result {
            Err(EngError::BadRequest(message)) if message.contains(row.needle) => {}
            other => panic!(
                "{}: expected BadRequest containing {:?}, got {other:?}",
                row.name, row.needle
            ),
        }
        assert!(unchanged, "{}: a refused call changed a buffer", row.name);
    }
    refuse_foreign_buffers(&dev);
    dev.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
    for row in accepted_rows() {
        let (result, _) = attempt(&dev, &row);
        result.unwrap_or_else(|error| panic!("{}: expected Ok, got {error:?}", row.name));
    }
}

/// A buffer of another context, as `gu` and as `xf`, is refused by the ownership check that
/// converts the ranges. (A stream of another context is checked by `launch_on`, which lane C's
/// tests cover: context/launch/tests.rs:690-717, :429 and :1236, and context/stream/tests.rs:166.
/// The buffer case is also covered at context/launch/tests.rs:434. A zero-byte range is not
/// representable from safe code, covered by context/stream/tests.rs:226; neither is a freed
/// buffer: `DeviceBuffer::range` refuses it.)
fn refuse_foreign_buffers(dev: &Dev) {
    let other = CudaContext::new().expect("second context");
    let (gu_len, xf_len) = (2048, XF_FRAGMENT);
    let own = (
        upload_filled(&dev.ctx, 1, gu_len),
        upload_filled(&dev.ctx, 2, xf_len),
    );
    let foreign = (
        upload_filled(&other, 3, gu_len),
        upload_filled(&other, 4, xf_len),
    );
    for (name, gu, xf) in [("gu", &foreign.0, &own.1), ("xf", &own.0, &foreign.1)] {
        let call = SwigluPackCall {
            gu: gu.range(0, gu_len).expect("gu range"),
            xf: xf.range(0, xf_len).expect("xf range"),
            m: 1,
            n_ff: 256,
        };
        match dev.ctx.swiglu_pack(&dev.stream, call) {
            Err(EngError::BadRequest(message)) if message.contains("buffer belongs") => {}
            other => panic!("foreign {name}: expected a refusal, got {other:?}"),
        }
    }
}

/// Width of the activation, rows of the gate and up projections, and rows of the down projection
/// of the chain test.
const CHAIN_K: usize = 2048;
const CHAIN_FF: usize = 512;
const CHAIN_OUT: usize = 256;

/// What the chain test reads back from the device.
struct Chain {
    gu: Vec<f32>,
    xf: Vec<u8>,
    y: Vec<f32>,
}

/// pack -> gate matmul and up matmul into the two halves of one `gu` -> `swiglu_pack` -> down
/// matmul, all on one stream: the RD-9c shape at seam 1.
fn run_chain(dev: &Dev, weights: (&TiledWeight, &TiledWeight), x: &[f32]) -> Chain {
    let (ctx, stream) = (&dev.ctx, &dev.stream);
    let x_buffer = upload_f32(ctx, x);
    let xf_in = upload_filled(ctx, 0xFF, XF_FRAGMENT);
    let gu = upload_filled(ctx, 0xFF, CHAIN_FF * 8);
    let xf_mid = upload_filled(ctx, 0xFF, XF_FRAGMENT);
    let y = upload_filled(ctx, 0xFF, CHAIN_OUT * 4);
    let pack = PackCall {
        x: x_buffer.range(0, CHAIN_K * 4).expect("x range"),
        xf: xf_in.range(0, XF_FRAGMENT).expect("xf range"),
        m: 1,
        k: CHAIN_K,
    };
    ctx.pack(stream, pack).expect("pack");
    for (half, first_row) in [(0, 0), (1, CHAIN_FF)] {
        let projection = Projection {
            weight: weights.0,
            first_row,
            rows: CHAIN_FF,
        };
        let io = MatmulIo {
            xf: xf_in.range(0, XF_FRAGMENT).expect("xf range"),
            y: gu
                .range(half * CHAIN_FF * 4, CHAIN_FF * 4)
                .expect("half of gu"),
            m: 1,
        };
        ctx.matmul(stream, projection, io)
            .expect("gate or up matmul");
    }
    let call = SwigluPackCall {
        gu: gu.range(0, CHAIN_FF * 8).expect("gu range"),
        xf: xf_mid.range(0, XF_FRAGMENT).expect("xf range"),
        m: 1,
        n_ff: CHAIN_FF,
    };
    ctx.swiglu_pack(stream, call).expect("swiglu_pack");
    let projection = Projection {
        weight: weights.1,
        first_row: 0,
        rows: CHAIN_OUT,
    };
    let io = MatmulIo {
        xf: xf_mid.range(0, XF_FRAGMENT).expect("xf range"),
        y: y.range(0, CHAIN_OUT * 4).expect("y range"),
        m: 1,
    };
    ctx.matmul(stream, projection, io).expect("down matmul");
    stream.synchronize().expect("drain the chain");
    Chain {
        gu: floats(&read_bytes(ctx, &gu, CHAIN_FF * 8)),
        xf: read_bytes(ctx, &xf_mid, XF_FRAGMENT),
        y: floats(&read_bytes(ctx, &y, CHAIN_OUT * 4)),
    }
}

#[test]
fn swiglu_pack_expert_chain_equals_the_host_reference() {
    if !testkit::device_present("swiglu_pack_expert_chain_equals_the_host_reference") {
        return;
    }
    let dev = Dev::new();
    // One 1,024-row weight: rows 0..512 are the gate, rows 512..1024 the up projection, so the
    // two halves differ and a swap is visible. Small inputs keep gate and up inside the domain.
    let gate_up = upload_weight(&dev.ctx, GgufType::Q4K, 2 * CHAIN_FF, CHAIN_K);
    let down = upload_weight(&dev.ctx, GgufType::Q4K, CHAIN_OUT, CHAIN_FF);
    let mut rng = Rng::new(0x6a11_0006);
    let x: Vec<f32> = (0..CHAIN_K)
        .map(|_| (rng.normal() * 0.002) as f32)
        .collect();
    let chain = run_chain(&dev, (&gate_up, &down), &x);
    assert!(
        chain.gu.iter().all(|v| v.is_finite()),
        "gate and up are finite"
    );
    assert!(chain.y.iter().all(|v| v.is_finite()), "y is finite");
    let mut stats = Stats::default();
    check_live(unpadded(1, CHAIN_FF), &chain.gu, &chain.xf, &mut stats);
    assert_eq!(
        report("expert chain", &stats),
        0,
        "E2 misses on the chain inputs"
    );
    // Printed, not asserted: no lane M rule exists for the chain error.
    let gate = blocking(&dev.ctx, &gate_up, (0, CHAIN_FF), &x);
    let up = blocking(&dev.ctx, &gate_up, (CHAIN_FF, CHAIN_FF), &x);
    // Both paths run the same matmul kernel, so each half of gu equals its projection bit for bit;
    // this pins which half holds the gate and which the up values.
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    assert_eq!(
        bits(&chain.gu[..CHAIN_FF]),
        bits(&gate),
        "the gate half of gu"
    );
    assert_eq!(bits(&chain.gu[CHAIN_FF..]), bits(&up), "the up half of gu");
    let h: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| f16_to_f32(f64_to_f16_bits(ref_swiglu(g, u))))
        .collect();
    let host = blocking(&dev.ctx, &down, (0, CHAIN_OUT), &h);
    let scale = host.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = (chain.y.iter().zip(&host)).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let gate_max = chain.gu[..CHAIN_FF]
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()));
    println!(
        "[{PROFILE}] swiglu_pack chain: max |gate| {gate_max}, max |y| {scale}, max |y - host| {worst} ({:.3e} relative)",
        worst / scale
    );
}

/// The padded and the unpadded call, captured into a graph, replay to the bytes of the eager call.
#[test]
fn swiglu_pack_inside_a_capture_equals_eager() {
    if !testkit::device_present("swiglu_pack_inside_a_capture_equals_eager") {
        return;
    }
    let dev = Dev::new();
    // The nodes of a call: the kernel (the whole-xf memset is gone, MEMSET-TRIM), plus two more
    // 2-D memsets for the padded width.
    let padded = Shape {
        m: 1,
        n_ff: 896,
        k_pad: 1024,
    };
    for (shape, nodes) in [(padded, 3), (unpadded(1, 512), 1)] {
        let mut rng = Rng::new(0xca97_0000 + shape.n_ff as u64);
        let scale = Scale { gate: 1.5, up: 1.0 };
        let gu = to_bytes(&make_gu(&mut rng, shape, scale));
        let xf_len = xf_bytes(shape.m, shape.k_pad).expect("xf size");
        let gu_buffer = upload_bytes(&dev.ctx, &gu);
        let xf_buffer = upload_filled(&dev.ctx, 0xFF, xf_len);
        let launch = || {
            let call = SwigluPackCall {
                gu: gu_buffer.range(0, gu.len()).expect("gu range"),
                xf: xf_buffer.range(0, xf_len).expect("xf range"),
                m: shape.m,
                n_ff: shape.n_ff,
            };
            dev.ctx.swiglu_pack_padded(&dev.stream, call, shape.k_pad)
        };
        launch().expect("the eager call, which also loads the module");
        dev.stream.synchronize().expect("drain the eager call");
        let read = || {
            (
                read_bytes(&dev.ctx, &gu_buffer, gu.len()),
                read_bytes(&dev.ctx, &xf_buffer, xf_len),
            )
        };
        let eager = read();
        let plan = CapturePlan {
            pins: &[&gu_buffer, &xf_buffer],
            holds: Vec::new(),
        };
        let (graph, ()) = dev.ctx.capture(&dev.stream, plan, launch).expect("capture");
        assert_eq!(
            graph.nodes(),
            nodes,
            "{shape:?}: nodes of the captured call"
        );
        for replay in 0..2 {
            gu_buffer.write_at(&dev.stream, 0, &gu).expect("restore gu");
            xf_buffer
                .write_at(&dev.stream, 0, &vec![0xFF; xf_len])
                .expect("garbage xf");
            graph.launch(&dev.stream).expect("replay");
            dev.stream.synchronize().expect("drain the replay");
            assert_eq!(read(), eager, "{shape:?}: replay {replay} against eager");
        }
    }
}

/// After a sticky device fault a bad range is still reported as a `BadRequest` and a valid call
/// gets the latch error. The fault is made in a child process, as `fault.rs` does.
#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn swiglu_pack_reports_a_bad_range_before_a_latched_context() {
    use crate::kernels::cuda::guard::DispatchGuard;
    use crate::kernels::cuda::{TERMINAL, native, native_error, stage_operation};
    const CHILD_ENV: &str = "INFERENCE_CUDA_SWIGLU_CHILD";
    const NAME: &str = "kernels::cuda::resident::swiglu_pack::tests::swiglu_pack_reports_a_bad_range_before_a_latched_context";
    if !testkit::device_present(NAME) {
        return;
    }
    if std::env::var(CHILD_ENV).is_err() {
        let exe = std::env::current_exe().expect("test binary path");
        let output = std::process::Command::new(exe)
            .args(["--exact", NAME, "--test-threads=1", "--nocapture"])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn the fault child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("swiglu fault child: done"),
            "fault child: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        return;
    }
    let dev = Dev::new();
    let gu = upload_filled(&dev.ctx, 0, 2048);
    let xf = upload_filled(&dev.ctx, 0, XF_FRAGMENT);
    let short = upload_filled(&dev.ctx, 0, 1024);
    fn call<'a>(gu: &'a DeviceBuffer, xf: &'a DeviceBuffer, xf_bytes: usize) -> SwigluPackCall<'a> {
        SwigluPackCall {
            gu: gu.range(0, 2048).expect("gu range"),
            xf: xf.range(0, xf_bytes).expect("xf range"),
            m: 1,
            n_ff: 256,
        }
    }
    {
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        let failure = native::fault_illegal_address().expect_err("illegal address must fault");
        drop(native_error(
            &TERMINAL,
            stage_operation(failure.stage),
            failure.status,
        ));
    }
    let bad = dev.ctx.swiglu_pack(&dev.stream, call(&gu, &short, 1024));
    assert!(
        matches!(&bad, Err(EngError::BadRequest(message)) if message.contains("xf range holds")),
        "a bad range on a latched context: {bad:?}"
    );
    let good = dev
        .ctx
        .swiglu_pack(&dev.stream, call(&gu, &xf, XF_FRAGMENT));
    assert!(
        matches!(
            &good,
            Err(EngError::Backend {
                operation: "execute",
                ..
            })
        ),
        "a good call on a latched context: {good:?}"
    );
    println!("swiglu fault child: done");
}
