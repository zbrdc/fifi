//! Seam-1 device tests (3090) of the KV8 split-KV decode attention wrapper (RD-8l, class Q8b):
//! the Rust wrapper against the C entry against the two kernels, compared with a plain f64 host
//! reference that reads the SAME dequantized values (`code * f32(stored fp16 scale)`), inside
//! the frozen reduction class E3 (lane M REPLY-44; the class of attn_decode, tests.rs).
//!
//! The reference is written from the window and ring convention of the fp16 contract (token `t`
//! is at position `P = kv_len - 1 + t` and sees `p <= P` with `P - p < window`; position `p`
//! lives in cell `p` or `p % ring`), not from the kernel. The int8 codes and the fp16 scales are
//! generated directly, so the reference and the kernel read one set of dequantized values.
//! Cells that no token of a call may read are int8 -128 with a NaN scale (0x7e00), and `part`
//! starts as NaN, so an out-of-range read poisons the result.
//!
//! The device suite runs with `--test-threads=1` (decision 0040). The fixed combine order is a
//! property the E3 comparisons above cannot see (they never read `part`), but it IS pinned: the
//! last test of this file owns the `part` scratch, reads the partials back and requires the
//! device `out` to be bit-equal to an index-order host combine of exactly those partials.
//!
//! Each refusal is checked by the text of its error, so a test fails when another check refuses
//! the call instead of the one under test.

use super::{AttnDecodeQ8Call, AttnQ8Shape, AttnSpan, AttnSplit, attn_q8_part_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// Class E3 (frozen, lane M REPLY-44): `max|d| <= 2^-17 * max(max|ref|, 1)` per head vector.
const E3: f64 = 1.0 / 131_072.0;
/// Two times the largest per-head query spread of the RD-M6 oracle (1.691144648391655): a case
/// with a larger `q` standard deviation is a stress case, held to a sanity bound only.
const IN_RANGE_QSD: f64 = 3.382;
const STRESS_SANITY: f64 = 1e-2;
/// An in-range maximum above 2^-19 is reported to lane M as a headroom warning.
const HEADROOM: f64 = 1.0 / 524_288.0;
/// fp16 quiet NaN: the scale of every cell no token may read (its codes are -128).
const F16_NAN: u16 = 0x7e00;
const GROUP: usize = 8;

/// One call: geometry, split plan, data parameters. `kv_len` is the device word.
#[derive(Clone)]
struct Spec {
    m: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cells: usize,
    window: i64,
    ring: i64,
    gate: bool,
    kv_len: i32,
    split: AttnSplit,
    qsd: f64,
}

impl Spec {
    /// Qwen3.6: head 256, 16 query heads over 2 kv heads, gate, linear cache of 4,200 cells.
    fn qwen(m: usize, kv_len: i32) -> Self {
        let (window, ring) = (AttnSpan::FULL.window, AttnSpan::FULL.ring);
        Self {
            m,
            n_head: 16,
            n_kv_head: 2,
            head_dim: 256,
            cells: 4200,
            window,
            ring,
            gate: true,
            kv_len,
            split: AttnSplit {
                max_splits: 64,
                min_keys: 64,
            },
            qsd: 1.2,
        }
    }
    /// A windowed or ring cache of another head size, 16 query heads over 2 kv heads.
    fn other(head_dim: usize, window: i64, ring: i64, cells: usize, gate: bool) -> Self {
        Self {
            head_dim,
            window,
            ring,
            cells,
            gate,
            ..Self::qwen(1, 1)
        }
    }
    fn ld(&self) -> usize {
        self.n_head * 2 * self.head_dim
    }
    fn scale(&self) -> f32 {
        (1.0 / (self.head_dim as f64).sqrt()) as f32
    }
    fn shape(&self) -> AttnQ8Shape {
        AttnQ8Shape {
            m: self.m,
            n_head: self.n_head,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            cells: self.cells,
            ld: self.ld(),
        }
    }
    fn span(&self) -> AttnSpan {
        AttnSpan {
            window: self.window,
            ring: self.ring,
        }
    }
    fn label(&self) -> String {
        format!(
            "m={} d={} win={} ring={} cells={} gate={} kv_len={} split=({},{}) qsd={}",
            self.m,
            self.head_dim,
            self.window,
            self.ring,
            self.cells,
            self.gate,
            self.kv_len,
            self.split.max_splits,
            self.split.min_keys,
            self.qsd
        )
    }
    /// The positions token `t` sees: `p <= P` and, with a window, `P - p < window`.
    fn visible(&self, t: usize) -> std::ops::Range<i64> {
        let last = i64::from(self.kv_len) - 1 + t as i64;
        let first = if self.window > 0 {
            (last - self.window + 1).max(0)
        } else {
            0
        };
        first..last + 1
    }
    /// The cache cell of position `p`.
    fn cell_of(&self, p: i64) -> usize {
        usize::try_from(if self.ring > 0 { p % self.ring } else { p }).expect("a non-negative cell")
    }
    /// Cells that some token reads.
    fn readable(&self) -> Vec<bool> {
        let mut cells = vec![false; self.cells];
        for t in 0..self.m {
            for p in self.visible(t) {
                cells[self.cell_of(p)] = true;
            }
        }
        cells
    }
}

/// Host data of one call: q, the gate rows, and the four KV8 planes.
struct Problem {
    spec: Spec,
    q: Vec<f32>,
    proj: Vec<f32>,
    k_codes: Vec<i8>,
    k_scales: Vec<u16>,
    v_codes: Vec<i8>,
    v_scales: Vec<u16>,
}

/// Codes of `spec`: pool draws in the cells some token reads, -128 with a NaN scale elsewhere
/// (all finite when `all_finite`, for calls whose `kv_len` is invalid). Every 32-value group of
/// a readable cell is all-zero (scale 0, codes 0) with a fixed probability, so the decode meets
/// the all-zero groups of the format.
fn codes(spec: &Spec, rng: &mut Rng, all_finite: bool) -> (Vec<i8>, Vec<u16>) {
    let readable = if all_finite {
        vec![true; spec.cells]
    } else {
        spec.readable()
    };
    let groups = spec.head_dim / 32;
    let mut out_codes = Vec::with_capacity(spec.n_kv_head * spec.cells * spec.head_dim);
    let mut out_scales = Vec::with_capacity(spec.n_kv_head * spec.cells * groups);
    for _head in 0..spec.n_kv_head {
        for &read in &readable {
            for _ in 0..groups {
                if read && rng.below(4) != 0 {
                    // A realistic scale: amax / 127 with amax about four standard deviations,
                    // so the dequantized values c * s span a standard-normal-like range (the
                    // oracle distribution the E3 class was frozen on) instead of an inflated one.
                    out_scales.push(f16_bits((rng.normal().abs() * (4.0 / 127.0)) as f32));
                    for _ in 0..32 {
                        // `as` binds tighter than `-`: the i8 form `draw as i8 - 127` overflows
                        // for draws of 128 and up. For draws 0..=254 the wrapped release value
                        // equals this exact form, so release data is unchanged.
                        out_codes.push((rng.below(255) as i32 - 127) as i8);
                    }
                } else if read {
                    // An all-zero group of the format: scale 0, codes 0, dequantizing to 0.
                    out_scales.push(0);
                    out_codes.extend(std::iter::repeat_n(0, 32));
                } else {
                    out_scales.push(F16_NAN);
                    out_codes.extend(std::iter::repeat_n(-128, 32));
                }
            }
        }
    }
    (out_codes, out_scales)
}

/// `q` is N(0, 1) times `qsd`; the q half of each projection head is NaN (the kernel must read
/// only the gate half), the gate half is N(0, 1).
fn problem(spec: &Spec, all_finite: bool) -> Problem {
    let mut rng =
        Rng::new(0x5eed ^ u64::from(spec.kv_len.unsigned_abs()) ^ ((spec.m as u64) << 40));
    let d = spec.head_dim;
    let q = (0..spec.m * spec.n_head * d)
        .map(|_| (rng.normal() * spec.qsd) as f32)
        .collect();
    let proj_len = if spec.gate { spec.m * spec.ld() } else { 0 };
    let proj = (0..proj_len)
        .map(|i| {
            if i % (2 * d) < d {
                f32::NAN
            } else {
                rng.normal() as f32
            }
        })
        .collect();
    let (k_codes, k_scales) = codes(spec, &mut rng, all_finite);
    let (v_codes, v_scales) = codes(spec, &mut rng, all_finite);
    Problem {
        spec: spec.clone(),
        q,
        proj,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
    }
}

/// How `part` starts: NaN, or finite values drawn from a seed.
#[derive(Clone, Copy)]
enum PartFill {
    Nan,
    Finite(u64),
}

fn part_content(bytes: usize, fill: PartFill) -> Vec<u8> {
    let mut rng = Rng::new(match fill {
        PartFill::Nan => 1,
        PartFill::Finite(seed) => seed,
    });
    let draw = |rng: &mut Rng| match fill {
        PartFill::Nan => f32::NAN,
        PartFill::Finite(_) => (rng.unit() * 8.0 - 4.0) as f32,
    };
    (0..bytes / 4)
        .flat_map(|_| draw(&mut rng).to_le_bytes())
        .collect()
}

fn le_i8(values: &[i8]) -> Vec<u8> {
    values.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn le16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|c| c.to_le_bytes()).collect()
}

/// The whole buffer as a range.
fn whole(b: &DeviceBuffer) -> DeviceRange<'_> {
    b.range(0, b.bytes()).expect("range")
}

/// Uploads the problem, queues one call, waits and returns `out`. `out` starts as 0xFF bytes.
fn launch(rig: &Rig, p: &Problem, fill: PartFill) -> Vec<f32> {
    let (ctx, s) = (&rig.ctx, &p.spec);
    let part_len =
        attn_q8_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits).expect("part");
    let out_len = s.m * s.n_head * s.head_dim * 4;
    let q = testkit::upload_f32(ctx, &p.q);
    let proj = s.gate.then(|| testkit::upload_f32(ctx, &p.proj));
    let k_codes = testkit::upload_bytes(ctx, &le_i8(&p.k_codes));
    let k_scales = testkit::upload_bytes(ctx, &le16(&p.k_scales));
    let v_codes = testkit::upload_bytes(ctx, &le_i8(&p.v_codes));
    let v_scales = testkit::upload_bytes(ctx, &le16(&p.v_scales));
    let word = testkit::upload_bytes(ctx, &s.kv_len.to_le_bytes());
    let part = testkit::upload_bytes(ctx, &part_content(part_len, fill));
    let out = testkit::upload_filled(ctx, 0xFF, out_len);
    let call = AttnDecodeQ8Call {
        shape: s.shape(),
        span: s.span(),
        split: s.split,
        scale: s.scale(),
        q: whole(&q),
        proj: proj.as_ref().map(whole),
        k_codes: whole(&k_codes),
        k_scales: whole(&k_scales),
        v_codes: whole(&v_codes),
        v_scales: whole(&v_scales),
        kv_len: whole(&word),
        part: whole(&part),
        out: whole(&out),
    };
    rig.ctx
        .attn_decode_q8(&rig.stream, call)
        .expect("queue the attention");
    rig.stream.synchronize().expect("drain");
    testkit::floats(&testkit::read_bytes(ctx, &out, out_len))
}

/// The dequantized value of `codes[cell * d + dim]` against `scales[cell * groups + dim / 32]`.
fn value(codes: &[i8], scales: &[u16], cell: usize, dim: usize, head: usize, spec: &Spec) -> f64 {
    let (per_head_codes, groups) = (spec.cells * spec.head_dim, spec.head_dim / 32);
    let code = codes[head * per_head_codes + cell * spec.head_dim + dim];
    let bits = scales[head * spec.cells * groups + cell * groups + dim / 32];
    f64::from(code) * f64::from(f16_to_f32(bits))
}

/// The f64 reference of head `h` of token `t`.
fn reference(p: &Problem, t: usize, h: usize) -> Vec<f64> {
    let s = &p.spec;
    let (d, kvh) = (s.head_dim, h / GROUP);
    let q = &p.q[(t * s.n_head + h) * d..][..d];
    let cells: Vec<usize> = s.visible(t).map(|pos| s.cell_of(pos)).collect();
    let dot = |cell: usize| -> f64 {
        q.iter()
            .enumerate()
            .map(|(dim, &qi)| f64::from(qi) * value(&p.k_codes, &p.k_scales, cell, dim, kvh, s))
            .sum()
    };
    let scores: Vec<f64> = cells
        .iter()
        .map(|&cell| f64::from(s.scale()) * dot(cell))
        .collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|&x| (x - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut out = vec![0.0f64; d];
    for (&cell, &w) in cells.iter().zip(&weights) {
        for (dim, acc) in out.iter_mut().enumerate() {
            *acc += w * value(&p.v_codes, &p.v_scales, cell, dim, kvh, s);
        }
    }
    for (i, acc) in out.iter_mut().enumerate() {
        let x = if s.gate {
            f64::from(p.proj[t * s.ld() + h * 2 * d + d + i])
        } else {
            f64::INFINITY
        };
        *acc = *acc / total / (1.0 + (-x).exp());
    }
    out
}

/// Worst `max|dev - ref| / max(max|ref|, 1)` over the head vectors of `heads` for every token.
fn worst_error(p: &Problem, out: &[f32], heads: &[usize]) -> f64 {
    let s = &p.spec;
    let mut worst = 0.0f64;
    for (t, &h) in (0..s.m).flat_map(|t| heads.iter().map(move |h| (t, h))) {
        let want = reference(p, t, h);
        let got = &out[(t * s.n_head + h) * s.head_dim..][..s.head_dim];
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite output at token {t} head {h}: {}",
            s.label()
        );
        let top = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        let diff = got
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - w).abs()));
        worst = worst.max(diff / top);
    }
    worst
}

/// A device context and its stream.
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

/// Runs `spec` and asserts E3 per head vector (in range) or the sanity bound (stress). Returns
/// the error and whether the case is a stress case.
fn check(rig: &Rig, spec: &Spec, heads: &[usize]) -> (f64, bool) {
    let p = problem(spec, false);
    let error = worst_error(&p, &launch(rig, &p, PartFill::Nan), heads);
    let stress = spec.qsd > IN_RANGE_QSD;
    println!("max error {error:e} stress={stress} {}", spec.label());
    let bound = if stress { STRESS_SANITY } else { E3 };
    assert!(
        error <= bound,
        "{}: {error:e} is above {bound:e}",
        spec.label()
    );
    (error, stress)
}

fn check_all(rig: &Rig, spec: &Spec) -> (f64, bool) {
    let heads = (0..spec.n_head).collect::<Vec<_>>();
    check(rig, spec, &heads)
}

/// Prints the in-range and the stress maxima, with the lane M warnings.
fn report(test: &str, outcomes: &[(f64, bool)]) {
    let top = |stress: bool| {
        outcomes
            .iter()
            .filter(|o| o.1 == stress)
            .fold(0.0f64, |m, o| m.max(o.0))
    };
    let (in_range, stress) = (top(false), top(true));
    println!("{test}: in-range maximum {in_range:e}, stress maximum {stress:e}");
    if in_range > HEADROOM {
        println!("{test}: HEADROOM WARNING: in-range maximum above 2^-19, report to lane M");
    }
    if stress > 2.0 * E3 {
        println!("{test}: FINDING: a stress maximum is above 2^-16, report to lane M");
    }
}

#[test]
fn attn_decode_q8_matches_the_host_reference() {
    if !testkit::device_present("attn_decode_q8_matches_the_host_reference") {
        return;
    }
    let rig = Rig::new();
    let mut specs: Vec<Spec> = [1, 2, 31, 32, 33, 64, 65, 100, 129, 1000, 1025, 4097]
        .map(|kv| Spec::qwen(1, kv))
        .to_vec();
    for kv in [33, 100, 1025] {
        for (max_splits, min_keys) in [(1, 64), (4, 64), (64, 128)] {
            specs.push(Spec {
                split: AttnSplit {
                    max_splits,
                    min_keys,
                },
                ..Spec::qwen(1, kv)
            });
        }
    }
    for (m, kv) in [2, 3, 5, 16]
        .into_iter()
        .flat_map(|m| [1, 33, 100].map(|kv| (m, kv)))
    {
        specs.push(Spec::qwen(m, kv));
    }
    specs.extend([1, 16].map(|m| Spec::qwen(m, (4200 - (m - 1)) as i32)));
    specs.extend([6.0, 20.0, 0.0].map(|qsd| Spec {
        qsd,
        ..Spec::qwen(1, 1025)
    }));
    let mut outcomes: Vec<_> = specs.iter().map(|spec| check_all(&rig, spec)).collect();
    // The long-context rows, one head per kv head: 32,768 and 131,072 keys (both inside one
    // hold of the gate run).
    for kv_len in [32_768_u32, 131_072] {
        let spec = Spec {
            cells: kv_len as usize + 64,
            ..Spec::qwen(1, kv_len as i32)
        };
        outcomes.push(check(&rig, &spec, &[0, 8]));
    }
    report("T1", &outcomes);
}

#[test]
fn attn_decode_q8_matches_the_reference_for_head_128_window_ring_and_no_gate() {
    if !testkit::device_present(
        "attn_decode_q8_matches_the_reference_for_head_128_window_ring_and_no_gate",
    ) {
        return;
    }
    let rig = Rig::new();
    // (`Spec::other(head_dim, window, ring, cells, gate)`, tokens per call, kv_len values)
    #[rustfmt::skip]
    let groups: [(Spec, &[usize], &[i32]); 5] = [
        (Spec::other(128, 0, 0, 3000, false), &[1, 3], &[1, 33, 1025, 2049]),
        (Spec::other(128, 0, 0, 1200, true), &[1, 2], &[33, 1025]),
        (Spec::other(256, 64, 0, 1100, true), &[1, 3], &[10, 64, 65, 200, 1000]),
        (Spec::other(256, 64, 79, 79, true), &[1, 16], &[10, 64, 65, 300]),
        (Spec::other(128, 1024, 1039, 2200, false), &[1, 5], &[1100, 1500, 3000]),
    ];
    let mut outcomes = Vec::new();
    for (geometry, tokens, lengths) in groups {
        for (&m, &kv_len) in tokens
            .iter()
            .flat_map(|m| lengths.iter().map(move |kv| (m, kv)))
        {
            outcomes.push(check_all(
                &rig,
                &Spec {
                    m,
                    kv_len,
                    ..geometry.clone()
                },
            ));
        }
    }
    report("T1b", &outcomes);
}

#[test]
fn attn_decode_q8_is_deterministic() {
    if !testkit::device_present("attn_decode_q8_is_deterministic") {
        return;
    }
    let rig = Rig::new();
    let ring = Spec {
        m: 2,
        kv_len: 300,
        ..Spec::other(128, 64, 79, 79, false)
    };
    for spec in [Spec::qwen(1, 1025), Spec::qwen(3, 700), ring] {
        let p = problem(&spec, false);
        let first = launch(&rig, &p, PartFill::Finite(11));
        let second = launch(&rig, &p, PartFill::Finite(12));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&first), bits(&second), "{}", spec.label());
        assert!(first.iter().all(|x| x.is_finite()), "{}", spec.label());
    }
}

/// Launches `spec` with finite planes (the length word may be invalid) and returns `out`.
fn launch_finite(rig: &Rig, spec: &Spec) -> Vec<f32> {
    launch(rig, &problem(spec, true), PartFill::Nan)
}

#[test]
fn attn_decode_q8_marks_an_invalid_kv_len_with_nan() {
    if !testkit::device_present("attn_decode_q8_marks_an_invalid_kv_len_with_nan") {
        return;
    }
    let rig = Rig::new();
    let (cells, m) = (600usize, 2usize);
    let linear = Spec {
        cells,
        ..Spec::qwen(m, 1)
    };
    let edge = (cells - (m - 1)) as i32;
    for kv_len in [0, -3, cells as i32 + 1, edge + 1] {
        let out = launch_finite(
            &rig,
            &Spec {
                kv_len,
                ..linear.clone()
            },
        );
        assert!(out.iter().all(|x| x.is_nan()), "linear kv_len {kv_len}");
    }
    check_all(
        &rig,
        &Spec {
            kv_len: edge,
            ..linear
        },
    );
    let ring = Spec::other(256, 64, 79, 79, true);
    for kv_len in [0, 1 << 30] {
        let out = launch_finite(
            &rig,
            &Spec {
                kv_len,
                ..ring.clone()
            },
        );
        assert!(out.iter().all(|x| x.is_nan()), "ring kv_len {kv_len}");
    }
    check_all(&rig, &ring);
}

#[test]
fn attn_q8_part_bytes_pins_the_scratch_rule() {
    assert_eq!(attn_q8_part_bytes(1, 2, 256, 64).expect("size"), 1_056_768);
    assert_eq!(
        attn_q8_part_bytes(16, 2, 256, 64).expect("size"),
        16_908_288
    );
    assert_eq!(attn_q8_part_bytes(1, 2, 256, 1).expect("size"), 16_512);
    assert_eq!(attn_q8_part_bytes(1, 4, 128, 64).expect("size"), 1_064_960);
    let overflow = attn_q8_part_bytes(usize::MAX / 2, 2, 256, 64);
    match overflow {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains("size overflow"), "{message}")
        }
        other => panic!("expected a size overflow refusal, got {other:?}"),
    }
}

// ---- refusals before any launch ----

/// Where a range lives: its buffer holds `offset + bytes` bytes. Slots: q, proj, k_codes,
/// k_scales, v_codes, v_scales, kv_len, part, out.
#[derive(Clone, Copy)]
struct Slot {
    offset: usize,
    bytes: usize,
}

const Q: usize = 0;
const PROJ: usize = 1;
const KC: usize = 2;
const KS: usize = 3;
const VC: usize = 4;
const VS: usize = 5;
const KV: usize = 6;
const PART: usize = 7;
const OUT: usize = 8;

/// A call that one edit makes bad: the geometry, plus the byte layout of every range.
#[derive(Clone)]
struct Plan {
    spec: Spec,
    scale: f32,
    ld: usize,
    slots: [Slot; 9],
    /// Two slots that share one buffer, both from offset 0.
    alias: Option<(usize, usize)>,
    label: &'static str,
    foreign_stream: bool,
    foreign_q: bool,
}

impl Plan {
    /// A valid call: Qwen shape, one token, a 4-cell cache, every buffer exactly sized.
    fn base() -> Self {
        let spec = Spec {
            cells: 4,
            ..Spec::qwen(1, 1)
        };
        let empty = Slot {
            offset: 0,
            bytes: 0,
        };
        let mut plan = Self {
            scale: spec.scale(),
            ld: 0,
            spec,
            slots: [empty; 9],
            alias: None,
            label: "base",
            foreign_stream: false,
            foreign_q: false,
        };
        plan.fit();
        plan
    }

    /// Sizes every range for the current geometry (plain arithmetic, written here on purpose).
    fn fit(&mut self) {
        self.ld = self.spec.ld();
        let s = &self.spec;
        let heads = s.n_head * s.head_dim;
        let part_floats = s.m * s.n_kv_head * s.split.max_splits * 8 * (2 + s.head_dim);
        let proj = if s.gate {
            (s.m - 1) * s.ld() * 4 + heads * 2 * 4
        } else {
            0
        };
        let codes = s.n_kv_head * s.cells * s.head_dim;
        let scales = s.n_kv_head * s.cells * (s.head_dim / 32) * 2;
        let bytes = [
            s.m * heads * 4,
            proj,
            codes,
            scales,
            codes,
            scales,
            4,
            part_floats * 4,
            s.m * heads * 4,
        ];
        for (slot, bytes) in self.slots.iter_mut().zip(bytes) {
            *slot = Slot { offset: 0, bytes };
        }
    }
}

/// Runs the call of `plan` (buffers sized by the slots; the length word is 1) and returns the
/// result and the `out` bytes afterwards. `other` supplies the foreign stream and buffer.
fn attempt(rig: &Rig, other: &Rig, plan: &Plan) -> (Result<(), EngError>, Vec<u8>) {
    let (ctx, s) = (&rig.ctx, &plan.spec);
    let content = |i: usize, slot: &Slot| -> Vec<u8> {
        let mut bytes = vec![if i == OUT { 0xFF } else { 0 }; slot.offset + slot.bytes];
        if i == KV {
            bytes[slot.offset..slot.offset + 2].copy_from_slice(&1u16.to_le_bytes());
        }
        bytes
    };
    let bufs: Vec<DeviceBuffer> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, slot)| testkit::upload_bytes(ctx, &content(i, slot)))
        .collect();
    let shared = plan.alias.map(|(a, b)| {
        let bytes = plan.slots[a].bytes.max(plan.slots[b].bytes);
        (a, b, testkit::upload_filled(ctx, 0xFF, bytes))
    });
    let foreign = testkit::upload_filled(&other.ctx, 0, plan.slots[Q].bytes);
    let range = |i: usize| -> DeviceRange<'_> {
        let buffer = match &shared {
            Some((a, b, buf)) if i == *a || i == *b => buf,
            _ if i == Q && plan.foreign_q => &foreign,
            _ => &bufs[i],
        };
        buffer
            .range(plan.slots[i].offset, plan.slots[i].bytes)
            .expect("range")
    };
    let call = AttnDecodeQ8Call {
        shape: AttnQ8Shape {
            ld: plan.ld,
            ..s.shape()
        },
        span: s.span(),
        split: s.split,
        scale: plan.scale,
        q: range(Q),
        proj: s.gate.then(|| range(PROJ)),
        k_codes: range(KC),
        k_scales: range(KS),
        v_codes: range(VC),
        v_scales: range(VS),
        kv_len: range(KV),
        part: range(PART),
        out: range(OUT),
    };
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let result = ctx.attn_decode_q8(stream, call);
    stream
        .synchronize()
        .unwrap_or_else(|e| panic!("{}: drain: {e:?}", plan.label));
    (
        result,
        testkit::read_bytes(ctx, &bufs[OUT], plan.slots[OUT].bytes),
    )
}

type Edit = fn(&mut Plan);
type Row = (&'static str, &'static str, Edit);

/// Rows of the refusal test: label, text the refusal must contain, edit. Every edit starts from
/// `Plan::base()` and sizes the buffers for the geometry it changes (`fit`); the scalar bound
/// checks come before the size checks, so the `2^31` rows keep the base buffers.
fn refusals() -> Vec<Row> {
    vec![
        ("m 0", "attn_decode_q8: tokens", |p: &mut Plan| p.spec.m = 0),
        ("m 17", "attn_decode_q8: tokens", |p| {
            p.spec.m = 17;
            p.fit();
        }),
        (
            "n_head not 8 times n_kv_head",
            "attn_decode_q8: heads",
            |p| {
                p.spec.n_head = 15;
                p.fit();
            },
        ),
        ("n_kv_head 65", "attn_decode_q8: heads", |p| {
            (p.spec.n_kv_head, p.spec.n_head) = (65, 520);
            p.fit();
        }),
        ("head_dim 64", "attn_decode_q8: head_dim", |p| {
            p.spec.head_dim = 64;
            p.fit();
        }),
        ("ld one short", "attn_decode_q8: ld", |p| p.ld -= 1),
        ("window -1", "attn_decode_q8: window", |p| {
            p.spec.window = -1
        }),
        ("ring without window", "a ring needs a window", |p| {
            p.spec.ring = 4
        }),
        ("cells 0", "attn_decode_q8: cells", |p| p.spec.cells = 0),
        ("min_keys above 2^20", "attn_decode_q8: min_keys", |p| {
            p.spec.split.min_keys = (1 << 20) + 64
        }),
        (
            "min_keys 32 (the fp16 tile)",
            "attn_decode_q8: min_keys",
            |p| p.spec.split.min_keys = 32,
        ),
        ("ring below window + m - 1", "attn_decode_q8: ring", |p| {
            (p.spec.window, p.spec.ring, p.spec.m, p.spec.cells) = (64, 70, 16, 100);
            p.fit();
        }),
        ("ring one above cells", "attn_decode_q8: ring", |p| {
            (p.spec.window, p.spec.ring, p.spec.cells) = (64, 101, 100);
            p.fit();
        }),
        ("max_splits 0", "attn_decode_q8: max_splits", |p| {
            p.spec.split.max_splits = 0
        }),
        ("max_splits 129", "attn_decode_q8: max_splits", |p| {
            p.spec.split.max_splits = 129;
            p.fit();
        }),
        ("min_keys 0", "attn_decode_q8: min_keys", |p| {
            p.spec.split.min_keys = 0
        }),
        ("scale 0", "attn_decode_q8: scale", |p| p.scale = 0.0),
        ("scale NaN", "attn_decode_q8: scale", |p| p.scale = f32::NAN),
        ("scale infinite", "attn_decode_q8: scale", |p| {
            p.scale = f32::INFINITY
        }),
        ("scale 1e31", "attn_decode_q8: scale", |p| p.scale = 1e31),
        ("q one float short", "the CUDA q range holds", |p| {
            p.slots[Q].bytes -= 4
        }),
        ("proj one float short", "the CUDA proj range holds", |p| {
            p.slots[PROJ].bytes -= 4
        }),
        (
            "k_codes one byte short",
            "the CUDA k_codes range holds",
            |p| p.slots[KC].bytes -= 1,
        ),
        (
            "k_scales one byte short",
            "the CUDA k_scales range holds",
            |p| p.slots[KS].bytes -= 1,
        ),
        (
            "v_codes one byte short",
            "the CUDA v_codes range holds",
            |p| p.slots[VC].bytes -= 1,
        ),
        (
            "v_scales one byte short",
            "the CUDA v_scales range holds",
            |p| p.slots[VS].bytes -= 1,
        ),
        (
            "kv_len range of 2 bytes",
            "the CUDA kv_len range holds",
            |p| p.slots[KV].bytes = 2,
        ),
        ("part one float short", "the CUDA part range holds", |p| {
            p.slots[PART].bytes -= 4
        }),
        ("out one float short", "the CUDA out range holds", |p| {
            p.slots[OUT].bytes -= 4
        }),
        (
            "q misaligned by 4",
            "the CUDA q range starts at offset",
            |p| p.slots[Q].offset = 4,
        ),
        (
            "k_codes misaligned by 2",
            "the CUDA k_codes range starts at offset",
            |p| p.slots[KC].offset = 2,
        ),
        (
            "part misaligned by 4",
            "the CUDA part range starts at offset",
            |p| p.slots[PART].offset = 4,
        ),
        (
            "out and part overlap",
            "the CUDA out and part ranges overlap",
            |p| p.alias = Some((OUT, PART)),
        ),
        (
            "out and q overlap",
            "the CUDA out and q ranges overlap",
            |p| p.alias = Some((OUT, Q)),
        ),
        (
            "part and kv_len overlap",
            "the CUDA part and kv_len ranges overlap",
            |p| p.alias = Some((PART, KV)),
        ),
        (
            "out and k_codes overlap",
            "the CUDA out and k_codes ranges overlap",
            |p| p.alias = Some((OUT, KC)),
        ),
        ("stream of another context", "stream belongs", |p| {
            p.foreign_stream = true
        }),
        (
            "buffer of another context",
            "belongs to another adapter",
            |p| p.foreign_q = true,
        ),
        ("ld i32::MAX + 1", "attn_decode_q8: exceeds i32", |p| {
            p.ld = 1 << 31
        }),
        ("cells 2^31", "attn_decode_q8: exceeds i32", |p| {
            p.spec.cells = 1 << 31
        }),
        ("max_splits 2^31", "attn_decode_q8: exceeds i32", |p| {
            p.spec.split.max_splits = 1 << 31
        }),
        ("min_keys 2^31", "attn_decode_q8: exceeds i32", |p| {
            p.spec.split.min_keys = 1 << 31
        }),
        ("window 2^31", "attn_decode_q8: exceeds i32", |p| {
            p.spec.window = 1 << 31
        }),
        ("ring 2^31", "attn_decode_q8: exceeds i32", |p| {
            p.spec.ring = 1 << 31
        }),
    ]
}

/// Calls that are valid at their boundary: each launches and gives finite output.
fn accepted() -> Vec<(&'static str, Edit)> {
    vec![
        ("ring == cells", |p: &mut Plan| {
            (p.spec.window, p.spec.ring, p.spec.cells) = (64, 100, 100);
            p.fit();
        }),
        ("ring == window + m - 1", |p| {
            (p.spec.window, p.spec.ring, p.spec.m, p.spec.cells) = (64, 79, 16, 100);
            p.fit();
        }),
        ("n_kv_head 64", |p| {
            (p.spec.n_kv_head, p.spec.n_head) = (64, 512);
            p.fit();
        }),
        ("scale 1e29", |p| p.scale = 1e29),
        ("kv_len range of 8 bytes", |p| p.slots[KV].bytes = 8),
    ]
}

#[test]
fn attn_decode_q8_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("attn_decode_q8_rejects_bad_calls_before_any_launch") {
        return;
    }
    let (rig, other) = (Rig::new(), Rig::new());
    let (first, first_out) = attempt(&rig, &other, &Plan::base());
    first.expect("the base call is valid");
    for (label, needle, edit) in refusals() {
        let mut plan = Plan::base();
        plan.label = label;
        edit(&mut plan);
        let (result, out) = attempt(&rig, &other, &plan);
        let message = match result {
            Err(EngError::BadRequest(message)) => message,
            other => panic!("{label}: expected a BadRequest, got {other:?}"),
        };
        assert!(
            message.contains(needle),
            "{label}: expected {needle:?}, got {message:?}"
        );
        assert!(
            plan.alias.is_some() || out.iter().all(|&b| b == 0xFF),
            "{label}: out changed"
        );
    }
    for (label, edit) in accepted() {
        let mut plan = Plan::base();
        edit(&mut plan);
        let (result, out) = attempt(&rig, &other, &plan);
        result.unwrap_or_else(|error| panic!("{label}: {error:?}"));
        assert!(
            testkit::floats(&out).iter().all(|x| x.is_finite()),
            "{label}"
        );
    }
    let (again, again_out) = attempt(&rig, &other, &Plan::base());
    again.expect("the backend is usable after the refusals");
    assert_eq!(first_out, again_out, "the base call after the refusals");
}

// ---- the split-combine order, pinned bit-exactly against the kernel's own partials ----

/// The gate value of every gate row of the pin: `expf(-30)` is about 9.4e-14, far below half
/// an ulp of 1, so `1 + expf(-30)` and the sigmoid are exactly 1 on the host and the device.
const PIN_GATE: f32 = 30.0;

/// `split_plan` of `attn_decode_q8.cu` on the host (all integers, so exact): `n_active` splits
/// of `chunk` keys, a multiple of the 64-key tile, cover `count` keys.
fn pin_split_plan(count: usize, split: AttnSplit) -> (usize, usize) {
    let n_split = count.div_ceil(split.min_keys).max(1).min(split.max_splits);
    let chunk = count.div_ceil(n_split).div_ceil(64) * 64;
    (chunk, count.div_ceil(chunk))
}

/// The first visible position of token `t` and its count, as `visible` of the kernel.
fn pin_visible(spec: &Spec, t: usize) -> (i64, usize) {
    let n_keys = i64::from(spec.kv_len) + t as i64;
    let count = if spec.window > 0 {
        n_keys.min(spec.window)
    } else {
        n_keys
    };
    (n_keys - count, count as usize)
}

/// The data of one combine-pin case. Every query head of a kv head shares one q vector, so one
/// hot key per split (K codes `127 * sign(q)`, scale 1.0, or 512.0 for the skewed split) is the
/// strict maximum of its split for all eight query heads, with the same bits in every split
/// (the kernel's dot is a fixed fmaf chain over the same input bits); it sits at the FIRST key
/// of its split. Filler keys have codes up to 127 at scale 1/64, so their scores stay below
/// `hot / 64` and their probabilities underflow to +0: every partial is exactly
/// `m = the hot score, l = 1, a[dim] = the hot V[dim]`. The hot V of split 0 is a ±127 pattern
/// at scale 4.0, split 1 its exact negation, split 2 a fresh pattern at 2^-15 and the later
/// splits fresh patterns at 2^-16: the index-order A sum cancels split 0 against split 1 and
/// keeps an odd multiple of 127 * 2^-16 of the rest, while any other order must round at the
/// ulp of the ±508 terms and lands a 2^-16 off every element.
fn pin_problem(spec: &Spec, skew: Option<usize>, seed: u64) -> Problem {
    use std::collections::HashMap;
    let mut rng = Rng::new(seed);
    let (d, groups) = (spec.head_dim, spec.head_dim / 32);
    let pattern = |rng: &mut Rng| {
        (0..d)
            .map(|_| if rng.below(2) == 0 { 127 } else { -127 })
            .collect::<Vec<i8>>()
    };
    let q0: Vec<Vec<f32>> = (0..spec.n_kv_head)
        .map(|_| (0..d).map(|_| (rng.normal() * spec.qsd) as f32).collect())
        .collect();
    let pat: Vec<Vec<i8>> = (0..spec.n_kv_head).map(|_| pattern(&mut rng)).collect();
    let filler = |rng: &mut Rng| (rng.below(255) as i32 - 127) as i8;
    // The hot cells (the first visible key of every split of every token; one set, because all
    // tokens of a case share the plan), mapped to their split index.
    let mut hot = HashMap::new();
    for t in 0..spec.m {
        let (first, count) = pin_visible(spec, t);
        let (chunk, n_active) = pin_split_plan(count, spec.split);
        for s in 0..n_active {
            hot.insert(spec.cell_of(first + s as i64 * chunk as i64), s);
        }
    }
    let readable = spec.readable();
    let mut q = Vec::with_capacity(spec.m * spec.n_head * d);
    let (mut k_codes, mut k_scales) = (Vec::new(), Vec::new());
    let (mut v_codes, mut v_scales) = (Vec::new(), Vec::new());
    for k in 0..spec.n_kv_head {
        for (cell, &read) in readable.iter().enumerate() {
            let (kc, ks, vc, vs): (Vec<i8>, f32, Vec<i8>, f32) = if let Some(&s) = hot.get(&cell) {
                let hot_k = if Some(s) == skew { 512.0 } else { 1.0 };
                let hot_codes: Vec<i8> = q0[k]
                    .iter()
                    .map(|&x| if x >= 0.0 { 127 } else { -127 })
                    .collect();
                let (vc, vs) = match s {
                    0 => (pat[k].clone(), 4.0),
                    1 => (pat[k].iter().map(|&c| -c).collect(), 4.0),
                    2 => (pattern(&mut rng), 1.0 / 32768.0),
                    _ => (pattern(&mut rng), 1.0 / 65536.0),
                };
                (hot_codes, hot_k, vc, vs)
            } else if read {
                let fill = |rng: &mut Rng| (0..d).map(|_| filler(rng)).collect::<Vec<i8>>();
                (fill(&mut rng), 1.0 / 64.0, fill(&mut rng), 1.0 / 64.0)
            } else {
                // A cell no token may read: poison, as `codes` above.
                (vec![-128; d], f32::NAN, vec![-128; d], f32::NAN)
            };
            k_codes.extend(kc);
            k_scales.extend(std::iter::repeat_n(f16_bits(ks), groups));
            v_codes.extend(vc);
            v_scales.extend(std::iter::repeat_n(f16_bits(vs), groups));
        }
    }
    for h in 0..spec.n_head {
        q.extend_from_slice(&q0[h / GROUP]);
    }
    let proj = if spec.gate {
        (0..spec.m * spec.ld())
            .map(|i| if i % (2 * d) < d { f32::NAN } else { PIN_GATE })
            .collect()
    } else {
        Vec::new()
    };
    Problem {
        spec: spec.clone(),
        q: q.repeat(spec.m),
        proj,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
    }
}

/// The combine of `attn_q8_combine_k` in f32, from the device partials of `part`: the exact
/// block max over the active splits, the weights `expf(m_s - M)`, the L and A fmaf sums in
/// split-index order, the division and the optional sigmoid gate, with every operation and its
/// order read from the kernel. The data of the pin admits only two `expf` inputs: `m_s - M` is
/// 0 (the split maxima have the same bits, and `expf(0)` is exactly 1 in every implementation)
/// or, for the skewed split, deeply negative (`e^x` underflows to +0 on the host and the device
/// alike); the gate value 30 makes `1 + expf(-30)` exactly 1 on both. So `f32::exp` matches the
/// device `expf` bit for bit, `mul_add` is fmaf and `+`, `*`, `/` are IEEE on both: the replica
/// shares every bit with the kernel and needs no ulp tolerance. Reversing the sums (m7) changes
/// the A sum of every element by a 2^-16 step, far beyond any tolerance.
fn pin_host_combine(p: &Problem, t: usize, h: usize, part: &[f32]) -> Vec<f32> {
    let s = &p.spec;
    let (d, kvh, g) = (s.head_dim, h / GROUP, h % GROUP);
    let (_, count) = pin_visible(s, t);
    let (_, n_active) = pin_split_plan(count, s.split);
    let max_splits = s.split.max_splits;
    let cell0 = (t * s.n_kv_head + kvh) * max_splits;
    let mut partial = Vec::with_capacity(n_active);
    for split in 0..n_active {
        let at = (cell0 + split) * GROUP * 2 + g * 2;
        partial.push((part[at], part[at + 1])); // (m, l) of the split
    }
    let top = partial
        .iter()
        .fold(f32::NEG_INFINITY, |m, &(mx, _)| m.max(mx));
    let weights: Vec<f32> = partial.iter().map(|&(mx, _)| (mx - top).exp()).collect();
    let mut total = 0.0f32;
    for (&(_, l), &w) in partial.iter().zip(&weights) {
        total = l.mul_add(w, total);
    }
    let acc_base = s.m * s.n_kv_head * max_splits * GROUP * 2;
    (0..d)
        .map(|dim| {
            let mut acc = 0.0f32;
            for split in 0..n_active {
                let a = part[acc_base + ((cell0 + split) * GROUP + g) * d + dim];
                acc = a.mul_add(weights[split], acc);
            }
            let sig = if s.gate {
                let x = p.proj[t * s.ld() + h * 2 * d + d + dim];
                1.0f32 / (1.0f32 + (-x).exp())
            } else {
                1.0f32
            };
            (acc / total) * sig
        })
        .collect()
}

/// One case of the pin on the device: geometry, the skewed split (none or 1) and the seed.
struct PinCase {
    label: &'static str,
    spec: Spec,
    skew: Option<usize>,
    seed: u64,
}

/// One device result: the case label, its problem, the device `out` and the device `part`.
struct PinOutcome {
    label: &'static str,
    problem: Problem,
    out: Vec<f32>,
    part: Vec<f32>,
}

/// Uploads the problem (`part` starts as NaN, so a read the kernel must not make is visible),
/// queues one call, drains the stream and returns `out` and `part` as plain vectors.
fn pin_launch(
    ctx: &CudaContext,
    stream: &DeviceStream,
    p: &Problem,
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    let s = &p.spec;
    let part_bytes = attn_q8_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits)?;
    let out_bytes = s.m * s.n_head * s.head_dim * 4;
    let q = testkit::upload_f32(ctx, &p.q);
    let proj = s.gate.then(|| testkit::upload_f32(ctx, &p.proj));
    let k_codes = testkit::upload_bytes(ctx, &le_i8(&p.k_codes));
    let k_scales = testkit::upload_bytes(ctx, &le16(&p.k_scales));
    let v_codes = testkit::upload_bytes(ctx, &le_i8(&p.v_codes));
    let v_scales = testkit::upload_bytes(ctx, &le16(&p.v_scales));
    let word = testkit::upload_bytes(ctx, &s.kv_len.to_le_bytes());
    let part = testkit::upload_bytes(ctx, &part_content(part_bytes, PartFill::Nan));
    let out = testkit::upload_filled(ctx, 0xFF, out_bytes);
    ctx.attn_decode_q8(
        stream,
        AttnDecodeQ8Call {
            shape: s.shape(),
            span: s.span(),
            split: s.split,
            scale: s.scale(),
            q: whole(&q),
            proj: proj.as_ref().map(whole),
            k_codes: whole(&k_codes),
            k_scales: whole(&k_scales),
            v_codes: whole(&v_codes),
            v_scales: whole(&v_scales),
            kv_len: whole(&word),
            part: whole(&part),
            out: whole(&out),
        },
    )?;
    stream.synchronize()?;
    Ok((
        testkit::floats(&testkit::read_bytes(ctx, &out, out_bytes)),
        testkit::floats(&testkit::read_bytes(ctx, &part, part_bytes)),
    ))
}

/// The device phase: one launch per case, every device owner (buffers, stream, context) dropped
/// before the plain vectors return, so the assertions below cannot unwind through a drop (S3b).
fn pin_device_phase(cases: &[PinCase]) -> Result<Vec<PinOutcome>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut outcomes = Vec::new();
    for case in cases {
        let problem = pin_problem(&case.spec, case.skew, case.seed);
        let (out, part) = pin_launch(&ctx, &stream, &problem)?;
        outcomes.push(PinOutcome {
            label: case.label,
            problem,
            out,
            part,
        });
    }
    Ok(outcomes)
}

/// The combine order of `attn_decode_q8` (KV8), pinned bit-exactly: the device `out` must equal
/// an index-order host combine of the partials the kernel itself left in `part` (IMPL-KV8-M7).
#[test]
fn attn_decode_q8_combine_is_bit_equal_to_an_index_order_host_combine_of_its_partials() {
    if !testkit::device_present(
        "attn_decode_q8_combine_is_bit_equal_to_an_index_order_host_combine_of_its_partials",
    ) {
        return;
    }
    let qwen = |m: usize, kv_len: i32, cells: usize| Spec {
        cells,
        split: AttnSplit {
            max_splits: 8,
            min_keys: 64,
        },
        ..Spec::qwen(m, kv_len)
    };
    // Every case has n_active >= 3, so the order matters. The Qwen shapes give counts 1024 and
    // 1017..1020: chunk 128, 8 active splits (8 for every token of m = 4). The ring case gives
    // count 512: chunk 64, 8 splits. The skewed case is the Qwen shape with the hot K of split
    // 1 at scale 512, so its partial maximum sits ~5e5 above the others: every other weight
    // underflows to +0 and the split maxima differ by orders of magnitude.
    let cases = vec![
        PinCase {
            label: "qwen m=1 d=256 gate, 8 splits",
            spec: qwen(1, 1024, 1200),
            skew: None,
            seed: 0x51,
        },
        PinCase {
            label: "qwen m=4 d=256 gate, 8 splits",
            spec: qwen(4, 1017, 1024),
            skew: None,
            seed: 0x52,
        },
        PinCase {
            label: "d=128 window 512 ring 520, 8 splits",
            spec: Spec {
                m: 1,
                kv_len: 2000,
                split: AttnSplit {
                    max_splits: 8,
                    min_keys: 64,
                },
                ..Spec::other(128, 512, 520, 520, false)
            },
            skew: None,
            seed: 0x53,
        },
        PinCase {
            label: "qwen m=1, split 1 K scale 512, 8 splits",
            spec: qwen(1, 1024, 1200),
            skew: Some(1),
            seed: 0x54,
        },
    ];
    let outcomes = pin_device_phase(&cases).expect("the device phase of the combine pin");
    for outcome in &outcomes {
        let spec = &outcome.problem.spec;
        for t in 0..spec.m {
            let (_, count) = pin_visible(spec, t);
            let (_, n_active) = pin_split_plan(count, spec.split);
            assert!(
                n_active >= 3,
                "{}: only {n_active} active splits",
                outcome.label
            );
            for h in 0..spec.n_head {
                let want = pin_host_combine(&outcome.problem, t, h, &outcome.part);
                let got = &outcome.out[(t * spec.n_head + h) * spec.head_dim..][..spec.head_dim];
                for (dim, (&device, &host)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        device.to_bits(),
                        host.to_bits(),
                        "{}: token {t} head {h} dim {dim}: device {device:e} (bits {:#010x}) vs \
                         the index-order host combine {host:e} (bits {:#010x})",
                        outcome.label,
                        device.to_bits(),
                        host.to_bits(),
                    );
                }
            }
        }
        println!("{}: the combine order is pinned bit-exactly", outcome.label);
    }
}
