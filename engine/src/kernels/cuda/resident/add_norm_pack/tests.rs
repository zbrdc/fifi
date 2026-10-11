//! Seam-1 device tests (3090) of the fused add, RMSNorm and fp16 pack wrapper against a float64
//! host reference written from the contract (`norm = r * (1 / sqrt(sum(r^2) / n + eps)) * w`,
//! `r` the stored f32 residual). Lane M REPLY-73: `norm_out` is class E3 at 2^-17 per vector,
//! `xf` class E2, the residual add bit equal; the E1 form is printed and never asserted. The
//! suite runs with `--test-threads=1`; every statistic line names the build profile (the cargo
//! debug profile compiles the kernels with `-G -O0`).

use std::collections::BTreeMap;

use super::{AddNormCall, check_eps, check_shape, row_bytes};
use crate::EngError;
use crate::kernels::cuda::native::xf_bytes;
use crate::kernels::cuda::testkit::{
    self, Rng, e2_holds, f64_to_f16_bits, fnv1a64, read_bytes, unpack_xf, upload_f32,
    upload_filled, upload_weight,
};
use crate::kernels::cuda::{
    CapturePlan, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, MatmulIo, PackCall,
    Projection, TiledWeight,
};
use crate::weights::GgufType;

/// E3, lane M REPLY-73: 2^-17 per vector for an f32 reduction.
const NORM_OUT_E3: f64 = 1.0 / 131_072.0;
/// The byte that fills `xf` and `norm_out` before a launch: a NaN pattern, never a result.
const FILL: u8 = 0xff;

/// One data family: the scale of `h`, `eps`, the top of the weight range, and whether four
/// dimensions of every row are multiplied by 100.
#[derive(Clone, Copy)]
struct Group {
    name: &'static str,
    scale: f64,
    eps: f32,
    weight_top: f64,
    outliers: bool,
}

#[rustfmt::skip]
const fn group(name: &'static str, scale: f64) -> Group {
    Group { name, scale, eps: 1e-6, weight_top: 1.5, outliers: false }
}

/// The Qwen3.6 scale.
const MODEL: Group = group("model", 1.0);
/// The stress families: outliers, large, small, tiny, a big weight, zero, two other `eps`.
#[rustfmt::skip]
const STRESS: [Group; 8] = [
    Group { outliers: true, ..group("outliers", 1.0) }, group("large", 1e4), group("small", 1e-3),
    group("tiny", 1e-20), Group { weight_top: 50.0, ..group("bigw", 1.0) }, group("zero", 0.0),
    Group { eps: 1e-5, ..group("eps1e-5", 1.0) }, Group { eps: 1e-12, ..group("eps1e-12", 1.0) },
];

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }
}

/// Host inputs of one launch.
struct Inputs {
    m: usize,
    n: usize,
    eps: f32,
    h: Vec<f32>,
    delta: Vec<f32>,
    weight: Vec<f32>,
}

#[rustfmt::skip]
fn inputs(group: Group, m: usize, n: usize, seed: u64) -> Inputs {
    let mut rng = Rng::new(seed);
    let mut h: Vec<f32> = (0..m * n).map(|_| (rng.normal() * group.scale) as f32).collect();
    if group.outliers {
        for row in 0..m {
            for _ in 0..4 {
                h[row * n + rng.below(n as u64) as usize] *= 100.0;
            }
        }
    }
    let delta = (0..m * n).map(|_| (rng.normal() * 0.5 * group.scale) as f32).collect();
    let weight = (0..n).map(|_| (0.5 + rng.unit() * (group.weight_top - 0.5)) as f32).collect();
    Inputs { m, n, eps: group.eps, h, delta, weight }
}

/// The first `bytes` bytes of `buffer`.
fn range(buffer: &DeviceBuffer, bytes: usize) -> DeviceRange<'_> {
    buffer.range(0, bytes).expect("range")
}

/// What a launch left in `h`, `xf` and `norm_out`, as bytes.
struct Output {
    h: Vec<u8>,
    xf: Vec<u8>,
    out: Vec<u8>,
}

/// One launch on fresh buffers: `xf` and `norm_out` start filled with `FILL`, the call is queued
/// on the rig's stream, the stream is drained and the three outputs are read back.
fn launch(rig: &Rig, inp: &Inputs, with_delta: bool, with_out: bool) -> Output {
    let ctx = &rig.ctx;
    let row = row_bytes(inp.m, inp.n).expect("row bytes");
    let xf_len = xf_bytes(inp.m, inp.n).expect("xf bytes");
    let h = upload_f32(ctx, &inp.h);
    let delta = upload_f32(ctx, &inp.delta);
    let weight = upload_f32(ctx, &inp.weight);
    let xf = upload_filled(ctx, FILL, xf_len);
    let out = upload_filled(ctx, FILL, row);
    let call = AddNormCall {
        m: inp.m,
        n: inp.n,
        eps: inp.eps,
        h: range(&h, row),
        delta: with_delta.then(|| range(&delta, row)),
        weight: range(&weight, inp.n * 4),
        xf: range(&xf, xf_len),
        norm_out: with_out.then(|| range(&out, row)),
    };
    ctx.add_norm_pack(&rig.stream, call).expect("launch");
    rig.stream.synchronize().expect("drain the stream");
    Output {
        h: read_bytes(ctx, &h, row),
        xf: read_bytes(ctx, &xf, xf_len),
        out: read_bytes(ctx, &out, row),
    }
}

/// The residual the kernel stores: `fl32(h + delta)`, or `h` unchanged.
fn residual(inp: &Inputs, with_delta: bool) -> Vec<f32> {
    if !with_delta {
        return inp.h.clone();
    }
    inp.h.iter().zip(&inp.delta).map(|(h, d)| h + d).collect()
}

/// The f64 reference of every row of the stored residual `r`.
fn ref_norm(r: &[f32], inp: &Inputs) -> Vec<f64> {
    let n = inp.n;
    let mut out = Vec::with_capacity(r.len());
    for row in r.chunks(n) {
        let ss: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let rstd = 1.0 / (ss / n as f64 + f64::from(inp.eps)).sqrt();
        out.extend(
            row.iter()
                .zip(&inp.weight)
                .map(|(&v, &w)| f64::from(v) * rstd * f64::from(w)),
        );
    }
    out
}

/// E3 value: the worst over the rows of `max|d| / max(max|ref|, 1)`.
fn e3_value(dev: &[f32], reference: &[f64], n: usize) -> f64 {
    let rows = dev.chunks(n).zip(reference.chunks(n));
    rows.map(|(d, r)| {
        let top = r.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
        let worst = d.iter().zip(r).map(|(&x, &y)| (f64::from(x) - y).abs());
        worst.fold(0.0, f64::max) / top
    })
    .fold(0.0, f64::max)
}

/// The E1 form, printed only: `max |d| / (2^-20 * max(|ref|, 1))` over the elements.
fn e1_ratio(dev: &[f32], reference: &[f64]) -> f64 {
    dev.iter()
        .zip(reference)
        .map(|(&x, &y)| (f64::from(x) - y).abs() / (2f64.powi(-20) * y.abs().max(1.0)))
        .fold(0.0, f64::max)
}

/// The maxima of one group.
#[derive(Default)]
struct Worst {
    e3: f64,
    e1: f64,
    one_step: usize,
    elements: usize,
}

#[derive(Default)]
struct Report(BTreeMap<&'static str, Worst>);

impl Report {
    fn print(&self) {
        let profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        for (group, w) in &self.0 {
            println!(
                "RD4A-T1 profile={profile} group={group} e3={:.3e} (limit {NORM_OUT_E3:.3e}) \
                 e1_ratio={:.4} (printed, not a gate) fp16_one_step={} elements={}",
                w.e3, w.e1, w.one_step, w.elements
            );
        }
    }
}

/// Asserts everything the contract says about one launch and folds its statistics into `worst`.
/// `with_out` says whether `norm_out` was passed, `label` names the case in a failure.
fn check(label: &str, inp: &Inputs, run: (bool, bool), got: &Output, worst: &mut Worst) {
    let (with_delta, with_out) = run;
    let r = residual(inp, with_delta);
    // `h` is fl32(h + delta), bit for bit; without a delta it is unchanged.
    let want_h: Vec<u8> = r.iter().flat_map(|v| v.to_le_bytes()).collect();
    let at = testkit::first_difference(&got.h, &want_h);
    assert_eq!(
        at, None,
        "{label}: h differs from fl32(h + delta) at byte {at:?}"
    );
    let reference = ref_norm(&r, inp);
    let packed = unpack_xf(&got.xf, inp.m, inp.n);
    for (i, (&bits, &want)) in packed.iter().zip(&reference).enumerate() {
        assert!(
            e2_holds(bits, want),
            "{label}: xf[{i}] = {bits:#06x} misses the E2 rule for {want:e}"
        );
        worst.one_step += usize::from(bits != f64_to_f16_bits(want));
    }
    worst.elements += packed.len();
    if with_out {
        check_norm_out(label, inp, (&reference, &packed), got, worst);
    }
}

/// The `norm_out` assertions: finite, E3 against the reference, and `xf` is exactly the fp16
/// rounding of the device `norm_out`.
fn check_norm_out(
    label: &str,
    inp: &Inputs,
    want: (&[f64], &[u16]),
    got: &Output,
    worst: &mut Worst,
) {
    let dev = testkit::floats(&got.out);
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{label}: norm_out is not finite"
    );
    let e3 = e3_value(&dev, want.0, inp.n);
    assert!(e3 <= NORM_OUT_E3, "{label}: E3 value {e3:e} above 2^-17");
    worst.e3 = worst.e3.max(e3);
    worst.e1 = worst.e1.max(e1_ratio(&dev, want.0));
    for (i, (&bits, &value)) in want.1.iter().zip(&dev).enumerate() {
        assert_eq!(
            bits,
            f64_to_f16_bits(f64::from(value)),
            "{label}: xf[{i}] is not the fp16 of norm_out"
        );
    }
}

/// Runs the four (delta, norm_out) combinations of `inp`, checks each, and checks that dropping
/// `norm_out` leaves the `xf` bytes alone.
fn sweep_combinations(rig: &Rig, label: &str, inp: &Inputs, worst: &mut Worst) {
    for with_delta in [true, false] {
        let with = launch(rig, inp, with_delta, true);
        let without = launch(rig, inp, with_delta, false);
        let tag = format!("{label} delta={with_delta}");
        for (with_out, got) in [(true, &with), (false, &without)] {
            check(
                &format!("{tag} out={with_out}"),
                inp,
                (with_delta, with_out),
                got,
                worst,
            );
        }
        assert_eq!(
            fnv1a64(&with.xf),
            fnv1a64(&without.xf),
            "{tag}: xf depends on norm_out"
        );
    }
}

/// Shapes `m` x `n` with the model data, all four combinations of `delta` and `norm_out`.
fn sweep_shapes(rig: &Rig, report: &mut Report) {
    let worst = report.0.entry(MODEL.name).or_default();
    for m in [1, 2, 3, 7, 8, 9, 15, 16, 17, 33] {
        for n in [256, 512, 2048, 4096] {
            let inp = inputs(MODEL, m, n, 0x4a00 + (m * 7919 + n) as u64);
            sweep_combinations(rig, &format!("model m={m} n={n}"), &inp, worst);
        }
    }
}

/// Every stress group at `m = 3`, `n = 2048`; the zero group must give all-zero outputs.
fn sweep_groups(rig: &Rig, report: &mut Report) {
    for (i, group) in STRESS.iter().enumerate() {
        let inp = inputs(*group, 3, 2048, 0x4a10 + i as u64);
        let worst = report.0.entry(group.name).or_default();
        sweep_combinations(rig, group.name, &inp, worst);
        if group.scale == 0.0 {
            // The data are 0 * normal, so some are -0.0: compare values, not bytes.
            let got = launch(rig, &inp, true, true);
            let zero = testkit::floats(&got.out).iter().all(|&v| v == 0.0);
            assert!(zero, "zero: norm_out is not all zero");
            let packed = unpack_xf(&got.xf, inp.m, inp.n);
            assert!(
                packed.iter().all(|&b| b & 0x7fff == 0),
                "zero: xf is not all zero"
            );
        }
    }
}

/// The widest row (`n = 32768`) and 100 seeds at the model shape `m = 1`, `n = 2048`.
fn sweep_wide_and_seeds(rig: &Rig, report: &mut Report) {
    let worst = report.0.entry("wide").or_default();
    for m in [1, 17] {
        let inp = inputs(group("wide", 1.0), m, 32_768, 0x4a20 + m as u64);
        sweep_combinations(rig, &format!("wide m={m}"), &inp, worst);
    }
    let worst = report.0.entry(MODEL.name).or_default();
    for seed in 0..100 {
        let inp = inputs(MODEL, 1, 2048, 0x4a30 + seed);
        let got = launch(rig, &inp, true, true);
        check(&format!("seed {seed}"), &inp, (true, true), &got, worst);
    }
}

/// Pinned, not a claim (RD-4a A-1): a finite row whose sum of squares overflows f32 gives finite
/// zeros, and one `+inf` among large values gives one NaN and zeros elsewhere.
fn pin_overflow(rig: &Rig) {
    let mut inp = inputs(MODEL, 1, 256, 0x4a40);
    inp.h = vec![1e30; 256];
    let got = launch(rig, &inp, false, true);
    assert!(
        got.out.iter().all(|&b| b == 0),
        "overflowing finite row: norm_out is not zero"
    );
    assert!(
        got.xf.iter().all(|&b| b == 0),
        "overflowing finite row: xf is not zero"
    );
    inp.h[5] = f32::INFINITY;
    let got = launch(rig, &inp, false, true);
    let out = testkit::floats(&got.out);
    let nans: Vec<usize> = (0..256).filter(|&i| out[i].is_nan()).collect();
    assert_eq!(nans, [5], "a +inf element must give exactly one NaN");
    assert!(
        out.iter().enumerate().all(|(i, &v)| i == 5 || v == 0.0),
        "the other elements of the +inf row must be zero"
    );
}

#[test]
fn add_norm_pack_matches_the_float64_reference() {
    if !testkit::device_present("add_norm_pack_matches_the_float64_reference") {
        return;
    }
    let rig = Rig::new();
    let mut report = Report::default();
    sweep_shapes(&rig, &mut report);
    sweep_groups(&rig, &mut report);
    sweep_wide_and_seeds(&rig, &mut report);
    pin_overflow(&rig);
    report.print();
}

#[test]
fn add_norm_pack_is_deterministic() {
    if !testkit::device_present("add_norm_pack_is_deterministic") {
        return;
    }
    let rig = Rig::new();
    let digest = |o: &Output| (fnv1a64(&o.h), fnv1a64(&o.xf), fnv1a64(&o.out));
    for m in [1, 16] {
        let inp = inputs(MODEL, m, 2048, 0x4a50 + m as u64);
        let first = digest(&launch(&rig, &inp, true, true));
        for repeat in 1..200 {
            let again = digest(&launch(&rig, &inp, true, true));
            assert_eq!(again, first, "m={m}: launch {repeat} differs from launch 0");
        }
    }
}

/// Rows of the weight that T2 multiplies with.
const Y_ROWS: usize = 64;

/// The `y` bytes of one matmul of `xf` against `weight`, after the stream drained.
fn matmul_y(rig: &Rig, weight: &TiledWeight, xf: (&DeviceBuffer, usize), m: usize) -> Vec<u8> {
    let y_len = m * Y_ROWS * 4;
    let y = upload_filled(&rig.ctx, FILL, y_len);
    let io = MatmulIo {
        xf: xf.0.range(0, xf.1).expect("xf range"),
        y: y.range(0, y_len).expect("y range"),
        m,
    };
    let projection = Projection {
        weight,
        first_row: 0,
        rows: Y_ROWS,
    };
    rig.ctx.matmul(&rig.stream, projection, io).expect("matmul");
    rig.stream.synchronize().expect("drain the matmul");
    read_bytes(&rig.ctx, &y, y_len)
}

/// One shape: the fused `xf`, the `xf` that `pack` makes of the fused `norm_out`, and the matmul
/// results of both.
fn pack_and_matmul(rig: &Rig, m: usize, n: usize) {
    let ctx = &rig.ctx;
    let inp = inputs(MODEL, m, n, 0x4a60 + (m * 31 + n) as u64);
    let row = row_bytes(m, n).expect("row bytes");
    let xf_len = xf_bytes(m, n).expect("xf bytes");
    let h = upload_f32(ctx, &inp.h);
    let delta = upload_f32(ctx, &inp.delta);
    let weight = upload_f32(ctx, &inp.weight);
    let fused = upload_filled(ctx, FILL, xf_len);
    let out = upload_filled(ctx, FILL, row);
    let packed = upload_filled(ctx, 0, xf_len);
    let call = AddNormCall {
        m,
        n,
        eps: inp.eps,
        h: range(&h, row),
        delta: Some(range(&delta, row)),
        weight: range(&weight, n * 4),
        xf: range(&fused, xf_len),
        norm_out: Some(range(&out, row)),
    };
    ctx.add_norm_pack(&rig.stream, call).expect("fused launch");
    let pack = PackCall {
        x: range(&out, row),
        xf: range(&packed, xf_len),
        m,
        k: n,
    };
    ctx.pack(&rig.stream, pack).expect("pack of norm_out");
    rig.stream.synchronize().expect("drain");
    let (a, b) = (
        read_bytes(ctx, &fused, xf_len),
        read_bytes(ctx, &packed, xf_len),
    );
    assert_eq!(
        (a.len(), fnv1a64(&a)),
        (b.len(), fnv1a64(&b)),
        "m={m} n={n}: the fused xf differs from pack"
    );
    let weights = upload_weight(ctx, GgufType::Q80, Y_ROWS, n);
    let ya = matmul_y(rig, &weights, (&fused, xf_len), m);
    let yb = matmul_y(rig, &weights, (&packed, xf_len), m);
    assert_eq!(
        fnv1a64(&ya),
        fnv1a64(&yb),
        "m={m} n={n}: the matmul reads the two xf differently"
    );
}

#[test]
fn add_norm_pack_xf_equals_the_pack_wrapper_and_feeds_the_matmul() {
    if !testkit::device_present("add_norm_pack_xf_equals_the_pack_wrapper_and_feeds_the_matmul") {
        return;
    }
    let rig = Rig::new();
    for m in [1, 8, 16, 17, 40] {
        pack_and_matmul(&rig, m, 2048);
    }
    pack_and_matmul(&rig, 1, 4096);
}

/// The text of a `BadRequest`, or a failure naming the unexpected result.
fn bad_text<T: std::fmt::Debug>(result: Result<T, EngError>) -> String {
    match result {
        Err(EngError::BadRequest(text)) => text,
        other => panic!("expected a BadRequest, got {other:?}"),
    }
}

/// `check_shape`, `check_eps`, `row_bytes` and `xf_bytes` at the edge of every bound. The two
/// wrapping values (`2^32 + 1`, `2^32 + 256`) would pass as `m` 1 and `n` 256 through an `as`
/// conversion.
fn pin_shape_bounds() {
    for (m, n) in [(1, 256), (32_768, 256), (1, 32_768), (7, 768)] {
        let ints = check_shape(m, n).expect("a shape at the limit");
        assert_eq!((ints.m as usize, ints.n as usize), (m, n));
    }
    for m in [0, 32_769, usize::MAX, (1 << 32) + 1] {
        let text = bad_text(check_shape(m, 256));
        assert!(text.starts_with("add_norm_pack: m:"), "m={m}: {text}");
    }
    for n in [0, 255, 257, 33_024, usize::MAX, (1 << 32) + 256] {
        let text = bad_text(check_shape(1, n));
        assert!(text.starts_with("add_norm_pack: n:"), "n={n}: {text}");
    }
    for eps in [1e-6, f32::MIN_POSITIVE, f32::MAX] {
        check_eps(eps).expect("a usable eps");
    }
    for eps in [0.0, -0.0, -1e-6, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let text = bad_text(check_eps(eps));
        assert!(text.starts_with("add_norm_pack: eps:"), "eps={eps}: {text}");
    }
    assert_eq!(row_bytes(1, 2048), Some(8192));
    assert_eq!(row_bytes(32_768, 32_768), Some(1 << 32));
    assert_eq!(row_bytes(usize::MAX / 4 + 1, 1), None);
    assert_eq!(xf_bytes(1, 2048), Some(1_048_592));
    assert_eq!(xf_bytes(17, 2048), Some(69_632));
}

/// The two testkit rules: `f64_to_f16_bits` rounds from the f64 and `e2_holds` has a 2^-20 window.
fn pin_fp16_helpers() {
    // Through f32 the first value would tie to even (0x3c00); from the f64 it rounds up.
    assert_eq!(
        f64_to_f16_bits(1.0 + 2f64.powi(-11) + 2f64.powi(-40)),
        0x3c01
    );
    assert_eq!(f64_to_f16_bits(1.0 + 2f64.powi(-11)), 0x3c00);
    assert_eq!(f64_to_f16_bits(65520.0), 0x7c00);
    assert_eq!(f64_to_f16_bits(2f64.powi(-25)), 0x0000);
    assert_eq!(f64_to_f16_bits(-0.0), 0x8000);
    assert_eq!(f64_to_f16_bits(f64::NAN), 0x7e00);
    assert_eq!(f64_to_f16_bits(2f64.powi(-24)), 0x0001);
    let mid = 1.0 + 2f64.powi(-11);
    let near = mid * (1.0 + 2f64.powi(-21));
    assert!(
        e2_holds(0x3c00, near) && e2_holds(0x3c01, near),
        "2^-21 from the midpoint"
    );
    let far = mid * (1.0 + 2f64.powi(-19));
    assert!(
        e2_holds(0x3c01, far) && !e2_holds(0x3c00, far),
        "2^-19 from the midpoint"
    );
    assert!(
        e2_holds(0x8000, 0.0) && e2_holds(0x0000, -0.0),
        "the two zeros"
    );
    assert!(
        e2_holds(0x7bff, 65520.0) && e2_holds(0x7c00, 65520.0),
        "the overflow midpoint"
    );
    assert!(
        e2_holds(0x7c00, 70_000.0) && !e2_holds(0x7bff, 70_000.0),
        "past the overflow"
    );
}

/// `unpack_xf` on a hand-built fragment-order buffer: token 2 of tile 0 sits in lane 6 (`r` = 2,
/// `q` = 1) and `wi` = 3 gives `base` = 64 + 12 = 76, with the values at `base + j` and
/// `base + 32 + j`.
fn pin_unpack_order() {
    let vec = 3 * 32 + 6;
    let mut xf = vec![0u8; 65_537 * 16];
    for j in 0..8u16 {
        let at = vec * 16 + 2 * usize::from(j);
        xf[at..at + 2].copy_from_slice(&(0x1000 + j).to_le_bytes());
    }
    let table = unpack_xf(&xf, 3, 256);
    for j in 0..4u16 {
        let at = 2 * 256 + 76 + usize::from(j);
        assert_eq!(table[at], 0x1000 + j, "base + {j}");
        assert_eq!(table[at + 32], 0x1004 + j, "base + 32 + {j}");
    }
    assert_eq!(
        table.iter().filter(|&&v| v != 0).count(),
        8,
        "only token 2 holds values"
    );
}

#[test]
fn add_norm_pack_bounds_are_pinned() {
    pin_shape_bounds();
    pin_fp16_helpers();
    pin_unpack_order();
}

const ORACLE_NEEDS: &str = "needs QWEN36_ORACLE_DIR RD4A_FIXTURES";
/// The layers of the fixture: block 0 is `blk.0.attn_norm.weight`, then for layer index `j`
/// block `1 + 2j` is `post_attention_norm` of the layer and `2 + 2j` the next `attn_norm`.
const ORACLE_LAYERS: [usize; 4] = [0, 3, 17, 38];

/// The sorted distinct rows `round(i * (total - 1) / (count - 1))`: the rule of `norm_ref_check.py`.
fn pick(total: usize, count: usize) -> Vec<usize> {
    let step = (total - 1) as f64 / (count - 1) as f64;
    let mut rows: Vec<usize> = (0..count)
        .map(|i| (i as f64 * step).round_ties_even() as usize)
        .collect();
    rows.dedup();
    rows
}

/// The rows `idx` of the dump `key` of `phase` (f32 `[rows][2048]`, the token the slowest axis).
fn dump_rows(phase: &str, key: &str, idx: &[usize]) -> Vec<f32> {
    let dir = std::env::var("QWEN36_ORACLE_DIR").expect("QWEN36_ORACLE_DIR");
    let path = std::path::Path::new(&dir).join(format!("cpu/dump/P4/dumps/{phase}/{key}.bin"));
    let all = testkit::read_f32_file(&path).expect("dump file");
    idx.iter()
        .flat_map(|&t| all[t * 2048..(t + 1) * 2048].iter().copied())
        .collect()
}

/// One call against the dump: `h` must come back bit equal to `want_h` (when the call adds),
/// `norm_out` must be finite; its E3 value and the E2 misses of `xf` against the dump, and the
/// host reference against the dump, are printed, not asserted (no lane M rule exists for a bound
/// against the llama.cpp dump).
fn against_dump(
    rig: &Rig,
    label: &str,
    inp: &Inputs,
    with_delta: bool,
    dump: (Option<&[f32]>, &[f32]),
) {
    let got = launch(rig, inp, with_delta, true);
    let want_h = dump.0.map(<[f32]>::to_vec).unwrap_or_else(|| inp.h.clone());
    let want_bytes: Vec<u8> = want_h.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        testkit::first_difference(&got.h, &want_bytes),
        None,
        "{label}: h differs from the dump"
    );
    let dev = testkit::floats(&got.out);
    assert!(
        dev.len() == inp.m * inp.n && dev.iter().all(|v| v.is_finite()),
        "{label}: norm_out shape or finiteness"
    );
    let reference: Vec<f64> = dump.1.iter().map(|&v| f64::from(v)).collect();
    let packed = unpack_xf(&got.xf, inp.m, inp.n);
    let misses = packed
        .iter()
        .zip(&reference)
        .filter(|&(&b, &r)| !e2_holds(b, r))
        .count();
    let host = ref_norm(&residual(inp, with_delta), inp);
    let e3_dev = e3_value(&dev, &reference, inp.n);
    println!(
        "RD4A-T4 {label}: e3(device vs dump)={:.3e} e3(host reference vs dump)={:.3e} xf_e2_misses_vs_dump={misses}/{}",
        e3_dev,
        e3_value(
            &host.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &reference,
            inp.n
        ),
        packed.len()
    );
    // Gross sanity, not a tolerance claim: lane M still owns the real bound against the dump. It
    // catches a wrong weight, layer or row (E3 about 3), which the report-only numbers hide.
    assert!(
        e3_dev < 0.1,
        "{label}: E3 against the dump is {e3_dev:e}, gross sanity limit 0.1"
    );
}

/// The three calls RD-9 makes for `layer`: the first norm of the embedding row (layer 0), the
/// mid-layer call (to `attn_residual` and `attn_post_norm`) and the boundary call (to `l_out`
/// and the next `attn_norm`).
fn oracle_layer(rig: &Rig, phase: &str, rows: &[usize], fixture: &[f32], j: usize) {
    let layer = ORACLE_LAYERS[j];
    let weight = |block: usize| fixture[block * 2048..(block + 1) * 2048].to_vec();
    let make = |h: Vec<f32>, delta: Option<Vec<f32>>, w: Vec<f32>| Inputs {
        m: rows.len(),
        n: 2048,
        eps: 1e-6,
        delta: delta.unwrap_or_else(|| vec![0.0; h.len()]),
        h,
        weight: w,
    };
    let get = |key: &str| dump_rows(phase, key, rows);
    let prev = if layer == 0 {
        "model.input_embed".to_owned()
    } else {
        format!("l_out-{}", layer - 1)
    };
    if layer == 0 {
        let inp = make(get("model.input_embed"), None, weight(0));
        against_dump(
            rig,
            &format!("{phase} L0 first norm"),
            &inp,
            false,
            (None, &get("attn_norm-0")),
        );
    }
    let mixer = if (layer + 1).is_multiple_of(4) {
        "attn_output"
    } else {
        "linear_attn_out"
    };
    let mid = make(
        get(&prev),
        Some(get(&format!("{mixer}-{layer}"))),
        weight(1 + 2 * j),
    );
    let residual_dump = get(&format!("attn_residual-{layer}"));
    let post = get(&format!("attn_post_norm-{layer}"));
    against_dump(
        rig,
        &format!("{phase} L{layer} mid"),
        &mid,
        true,
        (Some(&residual_dump), &post),
    );
    let boundary = make(
        residual_dump,
        Some(get(&format!("ffn_out-{layer}"))),
        weight(2 + 2 * j),
    );
    let l_out = get(&format!("l_out-{layer}"));
    let next = get(&format!("attn_norm-{}", layer + 1));
    against_dump(
        rig,
        &format!("{phase} L{layer} boundary"),
        &boundary,
        true,
        (Some(&l_out), &next),
    );
}

#[test]
#[ignore = "needs QWEN36_ORACLE_DIR RD4A_FIXTURES"]
fn add_norm_pack_matches_the_rd_m6_dump() {
    if !testkit::device_present("add_norm_pack_matches_the_rd_m6_dump") {
        return;
    }
    let dir = std::env::var("RD4A_FIXTURES").expect(ORACLE_NEEDS);
    let fixture = testkit::read_f32_file(&std::path::Path::new(&dir).join("rd4a_norm_weights.bin"))
        .expect("fixture");
    assert_eq!(fixture.len(), 9 * 2048, "fixture shape");
    let rig = Rig::new();
    for (phase, rows) in [("prefill", pick(69, 16)), ("decode", vec![0, 1, 2, 3])] {
        for j in 0..ORACLE_LAYERS.len() {
            oracle_layer(&rig, phase, &rows, &fixture, j);
        }
    }
}

/// Where one range of a refusal case lives: buffer index, byte offset, bytes.
#[derive(Clone, Copy)]
struct Slot {
    buf: usize,
    offset: usize,
    bytes: usize,
}

/// Slot order: h, delta, weight, xf, norm_out; each in its own buffer by default.
const NAMES: [&str; 5] = ["h", "delta", "weight", "xf", "norm_out"];
const H: usize = 0;
const D: usize = 1;
const W: usize = 2;
const X: usize = 3;
const O: usize = 4;

/// One call: the shape, `eps`, the five ranges and whether `delta` and `norm_out` are given.
#[derive(Clone)]
struct Spec {
    m: usize,
    n: usize,
    eps: f32,
    slots: [Slot; 5],
    with_delta: bool,
    with_out: bool,
}

impl Spec {
    /// A valid call for `m` x `n` with exactly sized ranges.
    fn new(m: usize, n: usize) -> Self {
        let (row, xf) = (m * n * 4, xf_bytes(m, n).expect("xf bytes"));
        let slot = |buf, bytes| Slot {
            buf,
            offset: 0,
            bytes,
        };
        Spec {
            m,
            n,
            eps: EPS,
            slots: [
                slot(H, row),
                slot(D, row),
                slot(W, n * 4),
                slot(X, xf),
                slot(O, row),
            ],
            with_delta: true,
            with_out: true,
        }
    }

    /// The same call with slot `i` changed by `edit`.
    fn with(&self, i: usize, edit: impl FnOnce(&mut Slot)) -> Self {
        let mut spec = self.clone();
        edit(&mut spec.slots[i]);
        spec
    }

    /// The same call with slot `b` at offset 0 of the buffer of slot `a`: the two overlap.
    fn overlapping(&self, a: usize, b: usize) -> Self {
        self.with(b, |slot| (slot.buf, slot.offset) = (self.slots[a].buf, 0))
    }
}

const EPS: f32 = 1e-6;

/// Runs `spec` with every buffer filled with 0x11; slot `foreign` (if any) lives on `other`.
/// Returns the result and whether every buffer is byte for byte unchanged.
fn attempt(
    rig: &Rig,
    other: &CudaContext,
    spec: &Spec,
    foreign: Option<usize>,
) -> (Result<(), EngError>, bool) {
    let mut sizes = [16usize; 5];
    for slot in &spec.slots {
        sizes[slot.buf] = sizes[slot.buf].max(slot.offset + slot.bytes);
    }
    let owner = |buf: usize| {
        foreign
            .filter(|&i| spec.slots[i].buf == buf)
            .map_or(&rig.ctx, |_| other)
    };
    let buffers: Vec<DeviceBuffer> = (0..5)
        .map(|b| upload_filled(owner(b), 0x11, sizes[b]))
        .collect();
    let digest = || -> Vec<u64> {
        (0..5)
            .map(|b| fnv1a64(&read_bytes(owner(b), &buffers[b], sizes[b])))
            .collect()
    };
    let before = digest();
    let at = |i: usize| buffers[spec.slots[i].buf].range(spec.slots[i].offset, spec.slots[i].bytes);
    let ranges: Result<Vec<_>, _> = (0..5).map(at).collect();
    let [h, delta, weight, xf, out]: [DeviceRange<'_>; 5] =
        ranges.expect("ranges").try_into().ok().expect("five");
    let call = AddNormCall {
        m: spec.m,
        n: spec.n,
        eps: spec.eps,
        h,
        delta: spec.with_delta.then_some(delta),
        weight,
        xf,
        norm_out: spec.with_out.then_some(out),
    };
    let result = rig.ctx.add_norm_pack(&rig.stream, call);
    rig.stream.synchronize().expect("drain");
    (result, digest() == before)
}

/// The refusals, each one change from the valid base `(2, 512)`: (name, text it must contain, call).
#[rustfmt::skip]
fn refusal_rows() -> Vec<(String, String, Spec)> {
    let base = Spec::new(2, 512);
    let mut rows: Vec<(String, String, Spec)> = Vec::new();
    let mut add = |name: &str, needle: &str, spec: Spec| rows.push((name.to_owned(), needle.to_owned(), spec));
    for m in [0, usize::MAX] {
        add(&format!("m={m}"), "add_norm_pack: m:", Spec { m, ..base.clone() });
    }
    add("m=32769", "add_norm_pack: m:", Spec::new(32_769, 256));
    for n in [0, 255, 257, usize::MAX] {
        add(&format!("n={n}"), "add_norm_pack: n:", Spec { n, ..base.clone() });
    }
    add("n=33024", "add_norm_pack: n:", Spec::new(1, 33_024));
    for eps in [0.0, -1e-6, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        add(&format!("eps={eps}"), "add_norm_pack: eps:", Spec { eps, ..base.clone() });
    }
    for (i, short) in [(H, 4), (D, 4), (W, 4), (X, 16), (O, 4)] {
        add(&format!("{} short", NAMES[i]), &format!("{} range holds", NAMES[i]), base.with(i, |s| s.bytes -= short));
    }
    add("xf short above 16 tokens", "xf range holds", Spec::new(17, 256).with(X, |s| s.bytes -= 2));
    add("xf of m*n*2 bytes at m <= 16", "xf range holds", base.with(X, |s| s.bytes = 2 * 512 * 2));
    for (i, offset) in [(H, 2), (D, 2), (W, 2), (X, 4), (X, 8), (O, 2), (O, 4), (O, 8)] {
        let needle = format!("{} range starts at offset {offset}", NAMES[i]);
        add(&format!("{} at +{offset}", NAMES[i]), &needle, base.with(i, |s| s.offset = offset));
    }
    // Claim order: the written ranges h, xf, norm_out, each against the later written and the read ones.
    for (a, b) in [(H, X), (H, O), (H, W), (H, D), (X, O), (X, W), (X, D), (O, W), (O, D)] {
        let needle = format!("{} and {} ranges overlap", NAMES[a], NAMES[b]);
        add(&format!("{} over {}", NAMES[b], NAMES[a]), &needle, base.overlapping(a, b));
    }
    add("xf over h without delta", "h and xf ranges overlap", Spec { with_delta: false, ..base.overlapping(H, X) });
    rows
}

/// Calls that must be accepted: shared views, touching extents, larger ranges, the extremes.
#[rustfmt::skip]
fn control_specs() -> Vec<(&'static str, Spec)> {
    let base = Spec::new(2, 512);
    let row = base.slots[H].bytes;
    let bare = Spec { with_delta: false, with_out: false, ..Spec::new(1, 256) };
    vec![
        ("delta and weight share bytes", base.overlapping(D, W)),
        ("h and delta touch, extents disjoint", base.with(D, |s| (s.buf, s.offset) = (H, row))),
        ("xf larger than xf_bytes", base.with(X, |s| s.bytes += 4096)),
        ("norm_out in the tail of a larger buffer", base.with(O, |s| s.offset = 16)),
        ("no delta, no norm_out, m=1 n=256", bare),
        ("m=17 n=256 with an exact xf", Spec::new(17, 256)),
        ("m=32768 n=256", Spec::new(32_768, 256)),
        ("m=1 n=32768", Spec::new(1, 32_768)),
    ]
}

#[test]
fn add_norm_pack_rejects_bad_arguments_before_any_launch() {
    if !testkit::device_present("add_norm_pack_rejects_bad_arguments_before_any_launch") {
        return;
    }
    let (rig, other) = (Rig::new(), CudaContext::new().expect("second context"));
    for (name, needle, spec) in refusal_rows() {
        let (result, unchanged) = attempt(&rig, &other, &spec, None);
        let text = bad_text(result);
        assert!(
            text.contains(&needle),
            "{name}: wanted {needle:?}, got {text:?}"
        );
        assert!(unchanged, "{name}: a refused call changed a buffer");
    }
    for (i, name) in NAMES.iter().enumerate() {
        let (result, unchanged) = attempt(&rig, &other, &Spec::new(2, 512), Some(i));
        assert!(
            bad_text(result).contains("buffer belongs"),
            "foreign {name}"
        );
        assert!(unchanged, "foreign {name}: a refused call changed a buffer");
    }
    for (name, spec) in control_specs() {
        let (result, _) = attempt(&rig, &other, &spec, None);
        result.unwrap_or_else(|e| panic!("control {name}: {e:?}"));
    }
    rig.ctx
        .ensure_usable()
        .expect("refusals leave the backend usable");
}

/// Five buffers for a call of `inp`: h, delta, weight, xf (0xFF) and norm_out (0xFF).
fn buffers_for(ctx: &CudaContext, inp: &Inputs) -> [DeviceBuffer; 5] {
    let (row, xf_len) = (
        row_bytes(inp.m, inp.n).expect("row"),
        xf_bytes(inp.m, inp.n).expect("xf"),
    );
    [
        upload_f32(ctx, &inp.h),
        upload_f32(ctx, &inp.delta),
        upload_f32(ctx, &inp.weight),
        upload_filled(ctx, FILL, xf_len),
        upload_filled(ctx, FILL, row),
    ]
}

/// The call of `inp` on `b`, with `m` replaced when a bad shape is wanted.
fn call_on<'b>(b: &'b [DeviceBuffer; 5], inp: &Inputs, m: usize) -> AddNormCall<'b> {
    let (row, xf_len) = (
        row_bytes(inp.m, inp.n).expect("row"),
        xf_bytes(inp.m, inp.n).expect("xf"),
    );
    AddNormCall {
        m,
        n: inp.n,
        eps: inp.eps,
        h: range(&b[0], row),
        delta: Some(range(&b[1], row)),
        weight: range(&b[2], inp.n * 4),
        xf: range(&b[3], xf_len),
        norm_out: Some(range(&b[4], row)),
    }
}

/// A latched backend answers every call with the latch error, a bad shape included, because the
/// terminal flag is the first check. The latch is process wide, so the test runs itself again in a
/// child process that sets the flag.
#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn add_norm_pack_reports_a_bad_range_before_a_latched_context() {
    const CHILD: &str = "RD4A_LATCH_CHILD";
    const NAME: &str = "kernels::cuda::resident::add_norm_pack::tests::add_norm_pack_reports_a_bad_range_before_a_latched_context";
    if !testkit::device_present(NAME) {
        return;
    }
    if std::env::var(CHILD).is_err() {
        let exe = std::env::current_exe().expect("test binary path");
        let args = ["--exact", NAME, "--test-threads=1", "--nocapture"];
        let out = std::process::Command::new(exe)
            .args(args)
            .env(CHILD, "1")
            .output()
            .expect("spawn the child");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && text.contains("latch child: done"),
            "child failed:\n{text}"
        );
        return;
    }
    let rig = Rig::new();
    let inp = inputs(MODEL, 1, 256, 0x4a70);
    let b = buffers_for(&rig.ctx, &inp);
    crate::kernels::cuda::TERMINAL.store(true, std::sync::atomic::Ordering::Release);
    for (case, m) in [("a good call", 1), ("a call with m = 0", 0)] {
        let result = rig.ctx.add_norm_pack(&rig.stream, call_on(&b, &inp, m));
        assert!(
            matches!(&result, Err(EngError::Backend { message, .. }) if message.contains("terminal")),
            "{case}: expected the latch error, got {result:?}"
        );
    }
    println!("latch child: done");
}

/// One captured call replays to the bytes of the eager call, as two graph nodes (memset, kernel).
#[test]
fn add_norm_pack_inside_a_capture_equals_eager() {
    if !testkit::device_present("add_norm_pack_inside_a_capture_equals_eager") {
        return;
    }
    let rig = Rig::new();
    let inp = inputs(MODEL, 1, 2048, 0x4a80);
    let b = buffers_for(&rig.ctx, &inp);
    let (row, xf_len) = (
        row_bytes(1, 2048).expect("row"),
        xf_bytes(1, 2048).expect("xf"),
    );
    let read =
        || [(0, row), (3, xf_len), (4, row)].map(|(i, len)| read_bytes(&rig.ctx, &b[i], len));
    let reset = || {
        let h: Vec<u8> = inp.h.iter().flat_map(|v| v.to_le_bytes()).collect();
        b[0].write_at(&rig.stream, 0, &h).expect("reset h");
        b[3].write_at(&rig.stream, 0, &vec![FILL; xf_len])
            .expect("reset xf");
        b[4].write_at(&rig.stream, 0, &vec![FILL; row])
            .expect("reset norm_out");
    };
    rig.ctx
        .add_norm_pack(&rig.stream, call_on(&b, &inp, 1))
        .expect("eager call");
    rig.stream.synchronize().expect("drain");
    let eager = read();
    reset();
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let (graph, ()) = rig
        .ctx
        .capture(&rig.stream, plan, || {
            rig.ctx.add_norm_pack(&rig.stream, call_on(&b, &inp, 1))
        })
        .expect("capture");
    assert_eq!(graph.nodes(), 2, "a memset and a kernel");
    assert_eq!(read()[1], vec![FILL; xf_len], "the capture ran nothing");
    graph.launch(&rig.stream).expect("replay");
    rig.stream.synchronize().expect("drain the replay");
    assert_eq!(read(), eager, "the replay differs from the eager call");
}
