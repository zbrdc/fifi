//! Seam-1 device tests (3090) of the split-KV decode attention wrapper: the Rust wrapper against
//! the C entry against the two kernels, compared with a plain f64 host reference.
//!
//! The reference is written from the window and ring convention of the contract (token `t` is at
//! position `P = kv_len - 1 + t` and sees `p <= P` with `P - p < window`; position `p` lives in row
//! `p` or `p % ring`), not from the kernel. K and V are read from the same fp16 bits as the device.
//! Cells that no token of a call may read are fp16 NaN, and `part` starts as NaN, so an
//! out-of-range read poisons the result.
//!
//! The device suite runs with `--test-threads=1` (decision 0040). The fixed combine order IS
//! pinned bit-exactly by the T8 test at the end of this file: it owns its `part` scratch, reads
//! it back after the launch and recomputes `attn_combine_k` on the host in split-index order.
//!
//! Each refusal is checked by the text of its error, so a test fails when another check refuses
//! the call instead of the one under test.

use super::{AttnDecodeCall, AttnShape, AttnSpan, AttnSplit, attn_part_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// The fallible upload of `bytes` for the Result-only device phases of this file: the shared
/// `testkit` uploads panic, which must not happen while a device owner is alive.
fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer, EngError> {
    ctx.upload(bytes)
}

/// The fallible upload of `values` for the Result-only device phases.
fn upload_f32(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer, EngError> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    upload_bytes(ctx, &bytes)
}

/// The fallible upload of `len` bytes that all equal `byte`.
fn upload_filled(ctx: &CudaContext, byte: u8, len: usize) -> Result<DeviceBuffer, EngError> {
    upload_bytes(ctx, &vec![byte; len])
}

/// The fallible download of the first `len` bytes of `buffer`.
fn read_bytes(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; len];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes)
}

/// Class E3 for RD-8b, final: `max|d| <= 2^-17 * max(max|ref|, 1)` per head vector (lane M
/// REPLY-44, REPLY-48 and REPLY-50; decision 0047 on main `1a547172`, "finalize E3 and R1").
const E3: f64 = 1.0 / 131_072.0;
/// Two times the largest per-head query spread of the RD-M6 oracle (1.691144648391655): a case
/// with a larger `q` standard deviation is a stress case, held to a sanity bound only.
const IN_RANGE_QSD: f64 = 3.382;
const STRESS_SANITY: f64 = 1e-2;
/// An in-range maximum above 2^-19 is reported to lane M as a headroom warning.
const HEADROOM: f64 = 1.0 / 524_288.0;
/// fp16 quiet NaN: the content of every cell no token may read.
const F16_NAN: u16 = 0x7e00;
const GROUP: usize = 8;

/// One call: geometry, split plan, data parameters. `kv_len` is the device word.
#[derive(Clone)]
struct Spec {
    m: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cache_rows: usize,
    window: i64,
    ring: i64,
    gate: bool,
    kv_len: i32,
    split: AttnSplit,
    qsd: f64,
}

#[rustfmt::skip]
impl Spec {
    /// Qwen3.6: head 256, 16 query heads over 2 kv heads, gate, linear cache of 4,200 rows.
    fn qwen(m: usize, kv_len: i32) -> Self {
        let (window, ring) = (AttnSpan::FULL.window, AttnSpan::FULL.ring);
        Self { m, n_head: 16, n_kv_head: 2, head_dim: 256, cache_rows: 4200, window, ring, gate: true, kv_len, split: AttnSplit::STAGE1, qsd: 1.2 }
    }
    /// A windowed or ring cache of another head size, 16 query heads over 2 kv heads.
    fn other(head_dim: usize, window: i64, ring: i64, cache_rows: usize, gate: bool) -> Self {
        Self { head_dim, window, ring, cache_rows, gate, ..Self::qwen(1, 1) }
    }
    fn ld(&self) -> usize { self.n_head * 2 * self.head_dim }
    fn scale(&self) -> f32 { (1.0 / (self.head_dim as f64).sqrt()) as f32 }
    fn shape(&self) -> AttnShape {
        AttnShape { m: self.m, n_head: self.n_head, n_kv_head: self.n_kv_head, head_dim: self.head_dim, cache_rows: self.cache_rows, ld: self.ld() }
    }
    fn span(&self) -> AttnSpan { AttnSpan { window: self.window, ring: self.ring } }
    fn label(&self) -> String {
        format!("m={} d={} win={} ring={} rows={} gate={} kv_len={} split=({},{}) qsd={}", self.m, self.head_dim, self.window, self.ring, self.cache_rows, self.gate, self.kv_len, self.split.max_splits, self.split.min_keys, self.qsd)
    }
    /// The positions token `t` sees: `p <= P` and, with a window, `P - p < window`.
    fn visible(&self, t: usize) -> std::ops::Range<i64> {
        let last = i64::from(self.kv_len) - 1 + t as i64;
        let first = if self.window > 0 { (last - self.window + 1).max(0) } else { 0 };
        first..last + 1
    }
    /// The cache row of position `p`.
    fn row_of(&self, p: i64) -> usize {
        usize::try_from(if self.ring > 0 { p % self.ring } else { p }).expect("a non-negative row")
    }
    /// Rows that some token reads.
    fn readable(&self) -> Vec<bool> {
        let mut rows = vec![false; self.cache_rows];
        for t in 0..self.m {
            for p in self.visible(t) { rows[self.row_of(p)] = true; }
        }
        rows
    }
}

/// fp16 values of N(0, 1) draws and their f64 decodes, shared by the cases of a test.
struct Pool {
    bits: Vec<u16>,
    value: Vec<f64>,
}

impl Pool {
    fn new() -> Self {
        let mut rng = Rng::new(0xa77e_0001);
        let bits = (0..1 << 16)
            .map(|_| f16_bits(rng.normal() as f32))
            .collect();
        let value = (0..=u16::MAX).map(|b| f64::from(f16_to_f32(b))).collect();
        Self { bits, value }
    }
}

/// Host data of one call.
struct Problem {
    spec: Spec,
    q: Vec<f32>,
    proj: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
}

/// K or V of `spec`: pool draws in the rows some token reads, NaN elsewhere (all finite when
/// `all_finite`, for calls whose `kv_len` is invalid).
fn cache(spec: &Spec, pool: &Pool, rng: &mut Rng, all_finite: bool) -> Vec<u16> {
    let readable = if all_finite {
        vec![true; spec.cache_rows]
    } else {
        spec.readable()
    };
    let row_len = spec.n_kv_head * spec.head_dim;
    let mut out = Vec::with_capacity(spec.cache_rows * row_len);
    for &read in &readable {
        let draw = |rng: &mut Rng| {
            if read {
                pool.bits[(rng.next_u64() & 0xffff) as usize]
            } else {
                F16_NAN
            }
        };
        out.extend((0..row_len).map(|_| draw(rng)));
    }
    out
}

/// `q` is N(0, 1) times `qsd`; the q half of each projection head is NaN (the kernel must read
/// only the gate half), the gate half is N(0, 1).
fn problem(spec: &Spec, pool: &Pool, all_finite: bool) -> Problem {
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
    let k = cache(spec, pool, &mut rng, all_finite);
    let v = cache(spec, pool, &mut rng, all_finite);
    Problem {
        spec: spec.clone(),
        q,
        proj,
        k,
        v,
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

fn le_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|b| b.to_le_bytes()).collect()
}

/// The whole buffer as a range.
fn whole(b: &DeviceBuffer) -> DeviceRange<'_> {
    b.range(0, b.bytes()).expect("range")
}

/// The whole buffer as a range through `?`, for the Result-only device phases (every owner
/// drops before they return, so a range failure must not panic).
fn try_whole(b: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    b.range(0, b.bytes())
}

/// Uploads the problem, queues one call, waits and returns `out`. `out` starts as 0xFF bytes.
fn launch(rig: &Rig, p: &Problem, fill: PartFill) -> Vec<f32> {
    let (ctx, s) = (&rig.ctx, &p.spec);
    let part_len = attn_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits).expect("part");
    let out_len = s.m * s.n_head * s.head_dim * 4;
    let q = testkit::upload_f32(ctx, &p.q);
    let proj = s.gate.then(|| testkit::upload_f32(ctx, &p.proj));
    let k = testkit::upload_bytes(ctx, &le_bytes(&p.k));
    let v = testkit::upload_bytes(ctx, &le_bytes(&p.v));
    let word = testkit::upload_bytes(ctx, &s.kv_len.to_le_bytes());
    let part = testkit::upload_bytes(ctx, &part_content(part_len, fill));
    let out = testkit::upload_filled(ctx, 0xFF, out_len);
    let call = AttnDecodeCall {
        shape: s.shape(),
        span: s.span(),
        split: s.split,
        scale: s.scale(),
        q: whole(&q),
        proj: proj.as_ref().map(whole),
        k_cache: whole(&k),
        v_cache: whole(&v),
        kv_len: whole(&word),
        part: whole(&part),
        out: whole(&out),
    };
    ctx.attn_decode(&rig.stream, call)
        .expect("queue the attention");
    rig.stream.synchronize().expect("drain");
    testkit::floats(&testkit::read_bytes(ctx, &out, out_len))
}

/// The f64 reference of head `h` of token `t`.
fn reference(p: &Problem, pool: &Pool, t: usize, h: usize) -> Vec<f64> {
    let s = &p.spec;
    let (d, kvh) = (s.head_dim, h / GROUP);
    let q = &p.q[(t * s.n_head + h) * d..][..d];
    let rows: Vec<usize> = s.visible(t).map(|pos| s.row_of(pos)).collect();
    let at = |row: usize| (row * s.n_kv_head + kvh) * d;
    let dot = |row: usize| -> f64 {
        q.iter()
            .zip(&p.k[at(row)..])
            .map(|(&qi, &ki)| f64::from(qi) * pool.value[usize::from(ki)])
            .sum()
    };
    let scores: Vec<f64> = rows
        .iter()
        .map(|&row| f64::from(s.scale()) * dot(row))
        .collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|&x| (x - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut out = vec![0.0f64; d];
    for (&row, &w) in rows.iter().zip(&weights) {
        for (acc, &vi) in out.iter_mut().zip(&p.v[at(row)..]) {
            *acc += w * pool.value[usize::from(vi)];
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
fn worst_error(p: &Problem, pool: &Pool, out: &[f32], heads: &[usize]) -> f64 {
    let s = &p.spec;
    let mut worst = 0.0f64;
    for (t, &h) in (0..s.m).flat_map(|t| heads.iter().map(move |h| (t, h))) {
        let want = reference(p, pool, t, h);
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

/// A device context and its stream, and the shared fp16 pool.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        Self::try_new().expect("the device rig")
    }

    /// The fallible construction of the rig, for the Result-only device phases: a failure
    /// returns after the context has dropped, so no owner is alive when the phase propagates.
    fn try_new() -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        Ok(Self { ctx, stream })
    }
}

/// Runs `spec` and asserts E3 per head vector (in range) or the sanity bound (stress). Returns
/// the error and whether the case is a stress case.
fn check(rig: &Rig, pool: &Pool, spec: &Spec, heads: &[usize]) -> (f64, bool) {
    let p = problem(spec, pool, false);
    let error = worst_error(&p, pool, &launch(rig, &p, PartFill::Nan), heads);
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

fn check_all(rig: &Rig, pool: &Pool, spec: &Spec) -> (f64, bool) {
    check(rig, pool, spec, &(0..spec.n_head).collect::<Vec<_>>())
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
fn attn_decode_matches_the_host_reference() {
    if !testkit::device_present("attn_decode_matches_the_host_reference") {
        return;
    }
    let (rig, pool) = (Rig::new(), Pool::new());
    let mut specs: Vec<Spec> = [1, 2, 31, 32, 33, 64, 65, 100, 129, 1000, 1025, 4097]
        .map(|kv| Spec::qwen(1, kv))
        .to_vec();
    for kv in [33, 100, 1025] {
        for (max_splits, min_keys) in [(1, 32), (4, 64), (64, 128)] {
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
    let mut outcomes: Vec<_> = specs
        .iter()
        .map(|spec| check_all(&rig, &pool, spec))
        .collect();
    // 65,537 keys in a 66,000-row cache, q sd 1.2 and q sd 20 (stress); heads 0 and 8, one per kv head.
    for qsd in [1.2, 20.0] {
        let spec = Spec {
            cache_rows: 66_000,
            qsd,
            ..Spec::qwen(1, 65_537)
        };
        outcomes.push(check(&rig, &pool, &spec, &[0, 8]));
    }
    report("T1", &outcomes);
}

#[test]
fn attn_decode_matches_the_host_reference_for_head_128_window_ring_and_no_gate() {
    if !testkit::device_present(
        "attn_decode_matches_the_host_reference_for_head_128_window_ring_and_no_gate",
    ) {
        return;
    }
    let (rig, pool) = (Rig::new(), Pool::new());
    // (`Spec::other(head_dim, window, ring, rows, gate)`, tokens per call, kv_len values)
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
                &pool,
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
fn attn_decode_is_deterministic() {
    if !testkit::device_present("attn_decode_is_deterministic") {
        return;
    }
    let (rig, pool) = (Rig::new(), Pool::new());
    let ring = Spec {
        m: 2,
        kv_len: 300,
        ..Spec::other(128, 64, 79, 79, false)
    };
    for spec in [Spec::qwen(1, 1025), Spec::qwen(3, 700), ring] {
        let p = problem(&spec, &pool, false);
        let first = launch(&rig, &p, PartFill::Finite(11));
        let second = launch(&rig, &p, PartFill::Finite(12));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&first), bits(&second), "{}", spec.label());
        assert!(first.iter().all(|x| x.is_finite()), "{}", spec.label());
    }
}

/// Launches `spec` with finite caches (the length word may be invalid) and returns `out`.
fn launch_finite(rig: &Rig, pool: &Pool, spec: &Spec) -> Vec<f32> {
    launch(rig, &problem(spec, pool, true), PartFill::Nan)
}

#[test]
fn attn_decode_marks_an_invalid_kv_len_with_nan() {
    if !testkit::device_present("attn_decode_marks_an_invalid_kv_len_with_nan") {
        return;
    }
    let (rig, pool) = (Rig::new(), Pool::new());
    let (rows, m) = (600usize, 2usize);
    let linear = Spec {
        cache_rows: rows,
        ..Spec::qwen(m, 1)
    };
    let edge = (rows - (m - 1)) as i32;
    for kv_len in [0, -3, rows as i32 + 1, edge + 1] {
        let out = launch_finite(
            &rig,
            &pool,
            &Spec {
                kv_len,
                ..linear.clone()
            },
        );
        assert!(out.iter().all(|x| x.is_nan()), "linear kv_len {kv_len}");
    }
    check_all(
        &rig,
        &pool,
        &Spec {
            kv_len: edge,
            ..linear
        },
    );
    let ring = Spec::other(256, 64, 79, 79, true);
    for kv_len in [0, 1 << 30] {
        let out = launch_finite(
            &rig,
            &pool,
            &Spec {
                kv_len,
                ..ring.clone()
            },
        );
        assert!(out.iter().all(|x| x.is_nan()), "ring kv_len {kv_len}");
    }
    check_all(&rig, &pool, &ring);
}

#[test]
fn attn_part_bytes_pins_the_scratch_rule() {
    assert_eq!(attn_part_bytes(1, 2, 256, 64).expect("size"), 1_056_768);
    assert_eq!(attn_part_bytes(16, 2, 256, 64).expect("size"), 16_908_288);
    assert_eq!(attn_part_bytes(1, 2, 256, 1).expect("size"), 16_512);
    assert_eq!(attn_part_bytes(1, 4, 128, 64).expect("size"), 1_064_960);
    let overflow = attn_part_bytes(usize::MAX / 2, 2, 256, 64);
    assert!(matches!(overflow, Err(EngError::BadRequest(m)) if m.contains("size overflow")));
}

/// A fresh 4-byte word per length (the in-place `write_at` form waits for RD-2c).
#[test]
fn attn_decode_reads_the_length_from_each_word() {
    if !testkit::device_present("attn_decode_reads_the_length_from_each_word") {
        return;
    }
    let (rig, pool) = (Rig::new(), Pool::new());
    for kv_len in [10, 100, 33, 1024, 1] {
        check_all(
            &rig,
            &pool,
            &Spec {
                cache_rows: 1100,
                ..Spec::qwen(1, kv_len)
            },
        );
    }
}

// ---- T4: refusals before any launch ----

/// Where a range lives: its buffer holds `offset + bytes` bytes. Slots: q, proj, k_cache,
/// v_cache, kv_len, part, out.
#[derive(Clone, Copy)]
struct Slot {
    offset: usize,
    bytes: usize,
}

const Q: usize = 0;
const PROJ: usize = 1;
const K: usize = 2;
const V: usize = 3;
const KV: usize = 4;
const PART: usize = 5;
const OUT: usize = 6;

/// A call that one edit makes bad: the geometry, plus the byte layout of every range.
#[derive(Clone)]
struct Plan {
    spec: Spec,
    scale: f32,
    ld: usize,
    slots: [Slot; 7],
    /// Two slots that share one buffer, both from offset 0.
    alias: Option<(usize, usize)>,
    /// The row's label, named by a panic in the helpers.
    label: &'static str,
    foreign_stream: bool,
    foreign_q: bool,
}

impl Plan {
    /// A valid call: Qwen shape, one token, a 4-row cache, every buffer exactly sized.
    fn base() -> Self {
        let spec = Spec {
            cache_rows: 4,
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
            slots: [empty; 7],
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
        let cache = s.cache_rows * s.n_kv_head * s.head_dim * 2;
        let bytes = [
            s.m * heads * 4,
            proj,
            cache,
            cache,
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
    let content = |i: usize, slot: &Slot| {
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
    let call = AttnDecodeCall {
        shape: AttnShape {
            ld: plan.ld,
            ..s.shape()
        },
        span: s.span(),
        split: s.split,
        scale: plan.scale,
        q: range(Q),
        proj: s.gate.then(|| range(PROJ)),
        k_cache: range(K),
        v_cache: range(V),
        kv_len: range(KV),
        part: range(PART),
        out: range(OUT),
    };
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let result = ctx.attn_decode(stream, call);
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

/// Rows of T4: label, text the refusal must contain, edit. Every edit starts from `Plan::base()`
/// and sizes the buffers for the geometry it changes (`fit`); the scalar bound checks come before
/// the size checks, so the `2^31` rows keep the base buffers.
#[rustfmt::skip]
fn refusals() -> Vec<Row> {
    const BIG: usize = 1 << 31;
    vec![
        ("m 0", "attn_decode: tokens", |p| p.spec.m = 0),
        ("m 17", "attn_decode: tokens", |p| { p.spec.m = 17; p.fit(); }),
        ("n_head not 8 times n_kv_head", "attn_decode: heads", |p| { p.spec.n_head = 15; p.fit(); }),
        ("n_kv_head 65", "attn_decode: heads", |p| { (p.spec.n_kv_head, p.spec.n_head) = (65, 520); p.fit(); }),
        ("head_dim 64", "attn_decode: head_dim", |p| { p.spec.head_dim = 64; p.fit(); }),
        ("ld one short", "attn_decode: ld", |p| p.ld -= 1),
        ("window -1", "attn_decode: window", |p| p.spec.window = -1),
        ("ring without window", "a ring needs a window", |p| p.spec.ring = 4),
        ("cache_rows 0", "attn_decode: cache_rows", |p| p.spec.cache_rows = 0),
        ("min_keys above 2^20", "attn_decode: min_keys", |p| p.spec.split.min_keys = (1 << 20) + 32),
        ("ring below window + m - 1", "attn_decode: ring", |p| { (p.spec.window, p.spec.ring, p.spec.m, p.spec.cache_rows) = (64, 70, 16, 100); p.fit(); }),
        ("ring one above cache_rows", "attn_decode: ring", |p| { (p.spec.window, p.spec.ring, p.spec.cache_rows) = (64, 101, 100); p.fit(); }),
        ("window i32::MAX with ring 100 and m 16", "attn_decode: ring", |p| { (p.spec.window, p.spec.ring, p.spec.m, p.spec.cache_rows) = (i64::from(i32::MAX), 100, 16, 100); p.fit(); }),
        ("max_splits 0", "attn_decode: max_splits", |p| p.spec.split.max_splits = 0),
        ("max_splits 129", "attn_decode: max_splits", |p| { p.spec.split.max_splits = 129; p.fit(); }),
        ("min_keys 16", "attn_decode: min_keys", |p| p.spec.split.min_keys = 16),
        ("min_keys 48", "attn_decode: min_keys", |p| p.spec.split.min_keys = 48),
        ("scale 0", "attn_decode: scale", |p| p.scale = 0.0),
        ("scale NaN", "attn_decode: scale", |p| p.scale = f32::NAN),
        ("scale infinite", "attn_decode: scale", |p| p.scale = f32::INFINITY),
        ("scale 1e31", "attn_decode: scale", |p| p.scale = 1e31),
        ("q one float short", "the CUDA q range holds", |p| p.slots[Q].bytes -= 4),
        ("proj one float short", "the CUDA proj range holds", |p| p.slots[PROJ].bytes -= 4),
        ("k_cache one row short", "the CUDA k_cache range holds", |p| p.slots[K].bytes -= 1024),
        ("v_cache one row short", "the CUDA v_cache range holds", |p| p.slots[V].bytes -= 1024),
        ("kv_len range of 2 bytes", "the CUDA kv_len range holds", |p| p.slots[KV].bytes = 2),
        ("part one float short", "the CUDA part range holds", |p| p.slots[PART].bytes -= 4),
        ("out one float short", "the CUDA out range holds", |p| p.slots[OUT].bytes -= 4),
        ("q misaligned by 4", "the CUDA q range starts at offset", |p| p.slots[Q].offset = 4),
        ("k_cache misaligned by 8", "the CUDA k_cache range starts at offset", |p| p.slots[K].offset = 8),
        ("part misaligned by 4", "the CUDA part range starts at offset", |p| p.slots[PART].offset = 4),
        ("out and part overlap", "the CUDA out and part ranges overlap", |p| p.alias = Some((OUT, PART))),
        ("out and q overlap", "the CUDA out and q ranges overlap", |p| p.alias = Some((OUT, Q))),
        ("part and kv_len overlap", "the CUDA part and kv_len ranges overlap", |p| p.alias = Some((PART, KV))),
        ("stream of another context", "stream belongs", |p| p.foreign_stream = true),
        ("buffer of another context", "belongs to another adapter", |p| p.foreign_q = true),
        ("ld i32::MAX + 1", "attn_decode: exceeds i32", |p| p.ld = BIG),
        ("cache_rows 2^31", "attn_decode: exceeds i32", |p| p.spec.cache_rows = BIG),
        ("max_splits 2^31", "attn_decode: exceeds i32", |p| p.spec.split.max_splits = BIG),
        ("min_keys 2^31", "attn_decode: exceeds i32", |p| p.spec.split.min_keys = BIG),
        ("window 2^31", "attn_decode: exceeds i32", |p| p.spec.window = 1 << 31),
        ("ring 2^31", "attn_decode: exceeds i32", |p| p.spec.ring = 1 << 31),
    ]
}

/// Calls that are valid at their boundary: each launches and gives finite output.
#[rustfmt::skip]
fn accepted() -> Vec<(&'static str, Edit)> {
    vec![
        ("ring == cache_rows", |p| { (p.spec.window, p.spec.ring, p.spec.cache_rows) = (64, 100, 100); p.fit(); }),
        ("ring == window + m - 1", |p| { (p.spec.window, p.spec.ring, p.spec.m, p.spec.cache_rows) = (64, 79, 16, 100); p.fit(); }),
        ("n_kv_head 64", |p| { (p.spec.n_kv_head, p.spec.n_head) = (64, 512); p.fit(); }),
        ("scale 1e29", |p| p.scale = 1e29),
        ("kv_len range of 8 bytes", |p| p.slots[KV].bytes = 8),
    ]
}

#[test]
fn attn_decode_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("attn_decode_rejects_bad_calls_before_any_launch") {
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

// ---------------------------------------------------------------- T7 (RD-11a)

/// The capture plan of the owed test: the rig owns every buffer, so no pin and no hold.
fn no_pins() -> CapturePlan<'static> {
    CapturePlan {
        pins: &[],
        holds: Vec::new(),
    }
}

/// The persistent buffers of the owed test: one `attn_decode` whose `kv_len` is a device word
/// that the rig rewrites between replays.
struct WordRig<'r> {
    ctx: &'r CudaContext,
    stream: &'r DeviceStream,
    spec: Spec,
    q: &'r DeviceBuffer,
    proj: Option<&'r DeviceBuffer>,
    k: &'r DeviceBuffer,
    v: &'r DeviceBuffer,
    word: &'r DeviceBuffer,
    part: &'r DeviceBuffer,
    out: &'r DeviceBuffer,
    nan_part: Vec<u8>,
    out_len: usize,
}

impl WordRig<'_> {
    /// Sets the length word and restores the scratch and the output to their start image (host
    /// writes, outside any window).
    fn set(&self, len: i32) {
        let stream = self.stream;
        self.word
            .write_at(stream, 0, &len.to_le_bytes())
            .expect("set the length word");
        self.part
            .write_at(stream, 0, &self.nan_part)
            .expect("reset the partials");
        self.out
            .write_at(stream, 0, &vec![0xFF; self.out_len])
            .expect("poison the output");
    }

    fn queue(&self) -> Result<(), EngError> {
        self.ctx.attn_decode(
            self.stream,
            AttnDecodeCall {
                shape: self.spec.shape(),
                span: self.spec.span(),
                split: self.spec.split,
                scale: self.spec.scale(),
                q: whole(self.q),
                proj: self.proj.map(whole),
                k_cache: whole(self.k),
                v_cache: whole(self.v),
                kv_len: whole(self.word),
                part: whole(self.part),
                out: whole(self.out),
            },
        )
    }

    /// One eager launch at `len`; the caller drains the stream.
    fn eager(&self, len: i32) -> Vec<u8> {
        self.set(len);
        self.queue().expect("queue the attention");
        self.stream.synchronize().expect("drain");
        testkit::read_bytes(self.ctx, self.out, self.out_len)
    }
}

/// T7 (RD-8b, owed; the seven lengths of its graph row): one `attn_decode` captured and replayed
/// at kv_len 10, 100, 33, 1024, 2048, 1 and 777, with the device word rewritten between replays,
/// equals the eager launch at that length byte for byte, on a two-node graph (the split-KV pair).
#[test]
fn attn_decode_inside_a_capture_follows_the_word() {
    if !testkit::device_present("attn_decode_inside_a_capture_follows_the_word") {
        return;
    }
    let rig = Rig::new();
    let pool = Pool::new();
    let lengths = [10, 100, 33, 1024, 2048, 1, 777];
    let spec = Spec::qwen(1, 2048);
    let p = problem(&spec, &pool, true);
    let (ctx, stream) = (&rig.ctx, &rig.stream);
    let part_len = attn_part_bytes(spec.m, spec.n_kv_head, spec.head_dim, spec.split.max_splits)
        .expect("part bytes");
    let out_len = spec.m * spec.n_head * spec.head_dim * 4;
    let (q, proj) = (
        testkit::upload_f32(ctx, &p.q),
        testkit::upload_f32(ctx, &p.proj),
    );
    let (k, v) = (
        testkit::upload_bytes(ctx, &le_bytes(&p.k)),
        testkit::upload_bytes(ctx, &le_bytes(&p.v)),
    );
    let word = testkit::upload_bytes(ctx, &spec.kv_len.to_le_bytes());
    let nan_part = part_content(part_len, PartFill::Nan);
    let (part, out) = (
        testkit::upload_bytes(ctx, &nan_part),
        testkit::upload_filled(ctx, 0xFF, out_len),
    );
    let words = WordRig {
        ctx,
        stream,
        spec: spec.clone(),
        q: &q,
        proj: Some(&proj),
        k: &k,
        v: &v,
        word: &word,
        part: &part,
        out: &out,
        nan_part,
        out_len,
    };
    let eager: Vec<Vec<u8>> = lengths.iter().map(|&len| words.eager(len)).collect();
    words.set(lengths[0]);
    let (graph, ()) = ctx
        .capture(stream, no_pins(), || words.queue())
        .expect("capture of the attention");
    assert_eq!(graph.nodes(), 2, "the split-KV graph holds two kernels");
    for (case, &len) in lengths.iter().enumerate() {
        words.set(len);
        graph.launch(stream).expect("replay");
        stream.synchronize().expect("drain");
        assert_eq!(
            testkit::read_bytes(ctx, &out, out_len),
            eager[case],
            "kv_len {len}: the replay differs from the eager launch"
        );
    }
}

// ------------------------------------------- T8 (IMPL-ATTN-COMBINE-PIN)

/// How the K rows of a T8 case are built: one key vector in every readable row, the rows of
/// the first active split scaled up by a factor, or one tied-hot row and one near-hot row per
/// split at a shared hot factor. See [`combine_case`].
#[derive(Clone, Copy)]
enum Skew {
    Uniform,
    FirstSplit(f32),
    NearHot(f32),
}
/// Host data of one T8 case: `q`, the projection (empty without the gate), the K and V bits and
/// the f64 score gap a skew opens between the first split's max and the others' (0.0 without).
struct CombineCase {
    spec: Spec,
    q: Vec<f32>,
    proj: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
    skew_gap: f64,
}

/// The device `split_plan` (attn_decode.cu) replicated on the host, so the test reads exactly
/// the `n_active` cells per (token, kv head) that `attn_combine_k` merged: `chunk` is a multiple
/// of the 32-key tile, `n_active` splits of `chunk` keys cover `n_keys`.
fn host_split_plan(n_keys: usize, split: &AttnSplit) -> (usize, usize) {
    let want = n_keys.div_ceil(split.min_keys);
    let n_split = want.max(1).min(split.max_splits);
    let chunk = n_keys.div_ceil(n_split).div_ceil(32) * 32;
    (chunk, n_keys.div_ceil(chunk))
}

/// One T8 case. Every readable K row holds one key vector, so every key of every active split
/// scores the same bits per (token, head) and the partial maxima `m_s` are all equal: every
/// combine weight is `expf(0) = 1`, exact on the device and on the host. With
/// `Skew::FirstSplit(f)` (`uniform_q` then repeats one `q` row over every head, so the scaled
/// split is the top split for every head at once) the rows of the first active split hold `f`
/// times the vector instead and every other weight underflows `expf` to `+0` on both sides;
/// `skew_gap` carries the score gap the caller asserts. V stays random per row, so the
/// accumulators `a_{s,d}` are comparable, unordered sums and the sum order decides the bits.
/// The gate half of the projection is 0 (`sigmoid(0) = 1/2` exactly); the q half stays NaN.
///
/// With `Skew::NearHot(f)` the FIRST row of every split holds `f` times the vector, the next
/// `28 >> split` rows `f * 1023/1024` times it (one f16 step of 2^-10 below the hot bits, so
/// their scores sit a small `delta` under the tied split maximum), and every other row the
/// plain vector: the split maxima stay tied at the hot bits (every combine weight exactly
/// `expf(0) = 1`) while the near-hot probabilities make `l_s = 1 + (28 >> split) * expf(-delta)`
/// distinct f32 values a factor of two apart - 1.3 to 10 - so partial sums of the L chain round
/// on different ulp grids in different orders and the L sum alone re-rounds under a reversed
/// order (the uniform cases' `l_s` are the integer key counts of the chunks, order-blind, and
/// near-equal `l_s` can round the same in both orders). The plain rows sit `skew_gap` below the
/// hot rows, far past the `expf` underflow line, so only the hot and near-hot keys shape the
/// partials. Every near-hot case draws `q` positive per query head and its keys positive, so
/// `dot(q, k) > 0` for every head: the hot row leads every head's split at once while each
/// head's own score sets its near-hot delta. The `l_s` set therefore differs per head (the
/// `uniform_q` case below would put every head on one `l_s` set).
fn combine_case(spec: &Spec, pool: &Pool, uniform_q: bool, skew: Skew) -> CombineCase {
    let mut rng =
        Rng::new(0xb1a5 ^ u64::from(spec.kv_len.unsigned_abs()) ^ ((spec.m as u64) << 32));
    let d = spec.head_dim;
    let mut draw = || (rng.normal() * spec.qsd) as f32;
    // The near-hot cases vary the l set per query head: q is drawn positive per head and the
    // keys below are positive too, so `f * dot(q, k)` leads `dot(q, k) > 0` for every head.
    let per_head = !uniform_q && matches!(skew, Skew::NearHot(_));
    let q0: Vec<f32> = (0..d).map(|_| draw()).collect();
    let q = if uniform_q {
        q0.repeat(spec.m * spec.n_head)
    } else if per_head {
        (0..spec.m * spec.n_head * d)
            .map(|_| draw().abs())
            .collect()
    } else {
        (0..spec.m * spec.n_head * d).map(|_| draw()).collect()
    };
    let key: Vec<u16> = if uniform_q {
        q0.iter().map(|&x| f16_bits(x)).collect()
    } else if per_head {
        (0..d).map(|_| f16_bits(0.25 + rng.unit() as f32)).collect()
    } else {
        (0..d)
            .map(|_| pool.bits[usize::from(rng.next_u64() as u16)])
            .collect()
    };
    let positions = spec.visible(0);
    let n_keys =
        usize::try_from(positions.end - positions.start).expect("the visible count fits usize");
    let (chunk, _) = host_split_plan(n_keys, &spec.split);
    // The key factor of every visible row of token 0 (the linear qwen cases give every token
    // the same split layout): 1 everywhere, the first split scaled up, or - for NearHot - the
    // first row of every split at the hot factor and its second row a shade below it.
    let mut row_factor = vec![1.0f32; spec.cache_rows];
    match skew {
        Skew::Uniform => {}
        Skew::FirstSplit(f) => {
            for position in positions.take(chunk) {
                row_factor[spec.row_of(position)] = f;
            }
        }
        Skew::NearHot(f) => {
            for (at, position) in positions.enumerate() {
                let (split, key_at) = (at / chunk, at % chunk);
                let near = 28usize >> split;
                row_factor[spec.row_of(position)] = if key_at == 0 {
                    f
                } else if key_at <= near {
                    f * (1023.0 / 1024.0)
                } else {
                    1.0
                };
            }
        }
    }
    let readable = spec.readable();
    let row_len = spec.n_kv_head * d;
    let mut k = vec![F16_NAN; spec.cache_rows * row_len];
    for (row, cells) in k.chunks_exact_mut(row_len).enumerate() {
        if !readable[row] {
            continue;
        }
        for (i, cell) in cells.iter_mut().enumerate() {
            let bits = key[i % d];
            let factor = row_factor[row];
            *cell = if factor == 1.0 {
                bits
            } else {
                f16_bits(factor * f16_to_f32(bits))
            };
        }
    }
    let hot_factor = match skew {
        Skew::Uniform => 1.0,
        Skew::FirstSplit(f) | Skew::NearHot(f) => f,
    };
    let skew_gap = if hot_factor != 1.0 {
        let value = |bits: u16| f64::from(f16_to_f32(bits));
        // The score gap the hot row opens over the plain rows, per K element: the scaled key
        // bits against the row bits. The minimum over every (token, head) q vector is the one
        // the caller holds above the expf underflow line; the near-hot cases' per-head l sets
        // are why the minimum cannot use the first head's q alone.
        let deltas: Vec<f64> = key
            .iter()
            .map(|&bits| value(f16_bits(hot_factor * f16_to_f32(bits))) - value(bits))
            .collect();
        q.chunks_exact(d)
            .map(|q| {
                f64::from(spec.scale())
                    * q.iter()
                        .zip(&deltas)
                        .map(|(&q, &delta)| f64::from(q) * delta)
                        .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    } else {
        0.0
    };
    let v = cache(spec, pool, &mut rng, false);
    let proj = if spec.gate {
        (0..spec.m * spec.ld())
            .map(|i| if i % (2 * d) < d { f32::NAN } else { 0.0 })
            .collect()
    } else {
        Vec::new()
    };
    CombineCase {
        spec: spec.clone(),
        q,
        proj,
        k,
        v,
        skew_gap,
    }
}

/// Uploads one case, queues one `attn_decode` through the wrapper, drains the stream and reads
/// `out` and the owned `part` scratch back. Returns plain host vectors; every device owner is
/// dropped before the return, so the caller's assertions never panic inside a live backend
/// (the S3b lesson: a drop while panicking latches the backend).
fn launch_combine(rig: &Rig, case: &CombineCase) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    let (ctx, spec) = (&rig.ctx, &case.spec);
    let part_bytes = attn_part_bytes(spec.m, spec.n_kv_head, spec.head_dim, spec.split.max_splits)?;
    let out_bytes = spec.m * spec.n_head * spec.head_dim * 4;
    let q = upload_f32(ctx, &case.q)?;
    let proj = spec.gate.then(|| upload_f32(ctx, &case.proj)).transpose()?;
    let k = upload_bytes(ctx, &le_bytes(&case.k))?;
    let v = upload_bytes(ctx, &le_bytes(&case.v))?;
    let word = upload_bytes(ctx, &spec.kv_len.to_le_bytes())?;
    let part = upload_bytes(ctx, &part_content(part_bytes, PartFill::Nan))?;
    let out = upload_filled(ctx, 0xFF, out_bytes)?;
    ctx.attn_decode(
        &rig.stream,
        AttnDecodeCall {
            shape: spec.shape(),
            span: spec.span(),
            split: spec.split,
            scale: spec.scale(),
            q: try_whole(&q)?,
            proj: proj.as_ref().map(try_whole).transpose()?,
            k_cache: try_whole(&k)?,
            v_cache: try_whole(&v)?,
            kv_len: try_whole(&word)?,
            part: try_whole(&part)?,
            out: try_whole(&out)?,
        },
    )?;
    rig.stream.synchronize()?;
    let out = testkit::floats(&read_bytes(ctx, &out, out_bytes)?);
    let part = testkit::floats(&read_bytes(ctx, &part, part_bytes)?);
    Ok((out, part))
}

/// Host replica of `attn_combine_k` in f32 from the read-back `part`, for every token and head:
/// the block max over the active splits, `w_s = expf(m_s - M)` as `f32::exp` (see the T8 test
/// doc), the L and A sums as `mul_add` chains in split-index order (`fmaf` on the device), then
/// the division and the gate. Returns `out` as `[m][n_head][head_dim]`.
fn host_combine(spec: &Spec, proj: Option<&[f32]>, part: &[f32]) -> Vec<f32> {
    let (m, heads, kv_heads, d) = (spec.m, spec.n_head, spec.n_kv_head, spec.head_dim);
    let splits = spec.split.max_splits;
    let stats_end = m * kv_heads * splits * GROUP * 2;
    let mut out = vec![0.0f32; m * heads * d];
    for t in 0..m {
        let (_, n_active) = host_split_plan(spec.visible(t).count(), &spec.split);
        for h in 0..heads {
            let (kv_head, g) = (h / GROUP, h % GROUP);
            let cell0 = (t * kv_heads + kv_head) * splits;
            let max_of = |s: usize| part[(cell0 + s) * GROUP * 2 + g * 2];
            let sum_of = |s: usize| part[(cell0 + s) * GROUP * 2 + g * 2 + 1];
            let mut top = f32::NEG_INFINITY;
            for s in 0..n_active {
                top = top.max(max_of(s));
            }
            let weight = |s: usize| (max_of(s) - top).exp();
            let mut l = 0.0f32;
            for s in 0..n_active {
                l = sum_of(s).mul_add(weight(s), l);
            }
            for dim in 0..d {
                let acc_of = |s: usize| part[stats_end + (cell0 + s) * GROUP * d + g * d + dim];
                let mut a = 0.0f32;
                for s in 0..n_active {
                    a = acc_of(s).mul_add(weight(s), a);
                }
                let sig = match proj {
                    None => 1.0f32,
                    Some(rows) => {
                        let gate = rows[t * spec.ld() + h * 2 * d + d + dim];
                        1.0f32 / (1.0f32 + (-gate).exp())
                    }
                };
                out[(t * heads + h) * d + dim] = (a / l) * sig;
            }
        }
    }
    out
}

/// One T8 case with its label and whether it is a near-hot L-order case.
struct T8Case {
    name: &'static str,
    case: CombineCase,
    near_hot: bool,
}

/// One T8 device result: the case reference and the device `out` and `part` as plain vectors.
struct CombineOutcome<'a> {
    case: &'a T8Case,
    out: Vec<f32>,
    part: Vec<f32>,
}

/// The T8 device phase: one launch per case, every owner dropped with the `Rig` before the
/// plain outcomes return, so the assertions below never run inside a live backend.
fn combine_device_phase<'a>(
    cases: impl IntoIterator<Item = &'a T8Case>,
) -> Result<Vec<CombineOutcome<'a>>, EngError> {
    let rig = Rig::try_new()?;
    let mut outcomes = Vec::new();
    for case in cases {
        let (out, part) = launch_combine(&rig, &case.case)?;
        outcomes.push(CombineOutcome { case, out, part });
    }
    Ok(outcomes)
}

/// The combine's L sums of one case, in f32, from the device partials of `part`: the exact
/// block max over the active splits, the weights `expf(m_s - M)` and the `fmaf` chain exactly
/// as `attn_combine_k` sums them (`L = fmaf(l_s, w_s, L)`). One `(index, reversed)` pair per
/// (token, query head); the case is blind to the L order when every pair is bit-equal.
fn combine_l_orders(spec: &Spec, part: &[f32]) -> Vec<(f32, f32)> {
    let (m, heads, kv_heads) = (spec.m, spec.n_head, spec.n_kv_head);
    let splits = spec.split.max_splits;
    let mut orders = Vec::new();
    for t in 0..m {
        let (_, n_active) = host_split_plan(spec.visible(t).count(), &spec.split);
        for h in 0..heads {
            let (kv_head, g) = (h / GROUP, h % GROUP);
            let cell0 = (t * kv_heads + kv_head) * splits;
            let maxes: Vec<f32> = (0..n_active)
                .map(|s| part[(cell0 + s) * GROUP * 2 + g * 2])
                .collect();
            let sums: Vec<f32> = (0..n_active)
                .map(|s| part[(cell0 + s) * GROUP * 2 + g * 2 + 1])
                .collect();
            let top = maxes.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
            let weights: Vec<f32> = maxes.iter().map(|&mx| (mx - top).exp()).collect();
            let mut index = 0.0f32;
            for (&sum, &weight) in sums.iter().zip(&weights) {
                index = sum.mul_add(weight, index);
            }
            let mut reversed = 0.0f32;
            for (&sum, &weight) in sums.iter().zip(&weights).rev() {
                reversed = sum.mul_add(weight, reversed);
            }
            orders.push((index, reversed));
        }
    }
    orders
}

/// T8: the split-combine ORDER of `attn_decode`, pinned bit-exactly against the case's own
/// partials (git-db made this pin required for the fp16 KV path).
///
/// The test owns the `part` scratch it passes to the kernel, reads `out` and `part` back after
/// the launch and recomputes the combine on the host in f32 exactly as `attn_combine_k` does
/// (max over the active splits, the L and A `fmaf` sums in split-index order, the division,
/// the gate); the device `out` must be BIT-equal (`to_bits`) element by element.
///
/// Which `expf` the device uses, and why no ulp tolerance is needed: the build compiles
/// `inference_cuda_native` WITHOUT `-use_fast_math` (engine/build.rs; only the vendored MMQ
/// objects take it), so the device `expf` is the accurate CUDA `expf` of libdevice, not `__expf`.
/// CUDA's `expf` does not round like the host `f32::exp` (glibc) on generic inputs, so a generic
/// case could not be bit-pinned at all (the fallback would be a tolerance of at most 2 ulps).
/// The data instead keeps every weight `expf`-exact, so the replica is exact and the assert is
/// bit-equality: the uniform cases give every active split the same max-score bits, so every
/// weight is `expf(0) = 1` on both sides, and the skewed case scales one split's keys up so far
/// (`skew_gap >= 120`, far past the `x <= -104` where `expf` underflows to `+0`) that every
/// other weight is exactly `+0` on both sides; the near-hot cases' plain rows sit the same gap
/// below their tied hot rows. `fmaf` mirrors `mul_add`, the division is IEEE
/// on both, and the gate is 0, so `sigmoid(0) = 1/2` exactly; nothing else in the combine has a
/// device intrinsic. Reversing the two index-order sums (mutant m7-fp16) re-rounds the A sums
/// of the uniform cases and the L sum of the near-hot cases, and fails the bit assert. Each
/// near-hot case is probed first: the reversed L of its own device partials must differ in at
/// least one bit, so a case that cannot see the L order fails the probe instead of passing the
/// bit comparison vacuously.
#[test]
fn attn_decode_combine_is_bit_equal_to_an_index_order_host_combine_of_its_partials() {
    if !testkit::device_present(
        "attn_decode_combine_is_bit_equal_to_an_index_order_host_combine_of_its_partials",
    ) {
        return;
    }
    let pool = Pool::new();
    let qwen_split = AttnSplit {
        max_splits: 8,
        min_keys: 256,
    };
    #[rustfmt::skip]
    let raw = [
        (Spec { split: qwen_split, ..Spec::qwen(1, 1025) }, false, Skew::Uniform, "qwen m=1 kv 1025"),
        (Spec { split: qwen_split, ..Spec::qwen(4, 1025) }, false, Skew::Uniform, "qwen m=4 kv 1025"),
        (Spec { split: AttnSplit { max_splits: 8, min_keys: 32 }, kv_len: 200, ..Spec::other(128, 128, 143, 143, false) }, false, Skew::Uniform, "head 128 window 128 ring 143"),
        (Spec { split: qwen_split, ..Spec::qwen(1, 1025) }, true, Skew::FirstSplit(40.0), "qwen skewed first split x40"),
        (Spec { split: qwen_split, ..Spec::qwen(1, 1025) }, false, Skew::NearHot(50.0), "qwen near-hot tied maxima x50, per-head l sets"),
        (Spec { split: qwen_split, ..Spec::qwen(1, 1025) }, false, Skew::NearHot(45.0), "qwen near-hot tied maxima x45, per-head l sets"),
        (Spec { split: qwen_split, ..Spec::qwen(1, 1025) }, false, Skew::NearHot(37.0), "qwen near-hot tied maxima x37, per-head l sets"),
        (Spec { split: qwen_split, ..Spec::qwen(1, 1025) }, false, Skew::NearHot(21.0), "qwen near-hot tied maxima x21, per-head l sets"),
    ];
    // Build and precheck every case on the host first: no device owner exists at this point.
    let mut cases = Vec::new();
    for (spec, uniform_q, skew, name) in raw {
        let case = combine_case(&spec, &pool, uniform_q, skew);
        if matches!(skew, Skew::FirstSplit(_) | Skew::NearHot(_)) {
            assert!(
                case.skew_gap >= 120.0,
                "{name}: the score gap {} is below the expf underflow line",
                case.skew_gap
            );
        }
        for t in 0..spec.m {
            let (_, n_active) = host_split_plan(spec.visible(t).count(), &spec.split);
            assert!(
                n_active >= 3,
                "{name}: token {t} has {n_active} active splits"
            );
        }
        cases.push(T8Case {
            name,
            near_hot: matches!(skew, Skew::NearHot(_)),
            case,
        });
    }
    // MAJOR F1: every near-hot case must see the reversed L sum in its own device partials BY
    // CONSTRUCTION, not by the luck of the rounding. This probe runs before the pin's device
    // phase; a blind case fails loudly here with its label instead of passing the bit
    // comparison. The probe makes the same launch as the pin (deterministic kernel, same
    // inputs), so the checked l set is the pinned one.
    let probed = combine_device_phase(cases.iter().filter(|c| c.near_hot))
        .expect("the probe phase of the near-hot L preconditions");
    for outcome in &probed {
        let orders = combine_l_orders(&outcome.case.case.spec, &outcome.part);
        let differing: Vec<&(f32, f32)> = orders
            .iter()
            .filter(|(index, reversed)| index.to_bits() != reversed.to_bits())
            .collect();
        println!(
            "precondition {}: {} of {} L sets differ in index and reversed order",
            outcome.case.name,
            differing.len(),
            orders.len()
        );
        for (index, reversed) in differing.iter().take(1) {
            println!(
                "  first: index {:#010x} ({index:e}), reversed {:#010x} ({reversed:e})",
                index.to_bits(),
                reversed.to_bits()
            );
        }
        assert!(
            !differing.is_empty(),
            "{}: the near-hot case cannot see the L order: all {} of its (token, head) L sets \
             sum bit-identically in index and reversed order; reseed or reshape the case",
            outcome.case.name,
            orders.len()
        );
    }
    // The pin's own device phase, result-only like the probe: every owner drops at the return.
    let outcomes = combine_device_phase(cases.iter()).expect("the device phase of the combine pin");
    for outcome in &outcomes {
        let name = outcome.case.name;
        let case = &outcome.case.case;
        assert!(
            outcome.out.iter().all(|x| x.is_finite()),
            "{name}: a non-finite device output element before the bit comparison"
        );
        let gate = (!case.proj.is_empty()).then_some(case.proj.as_slice());
        let want = host_combine(&case.spec, gate, &outcome.part);
        for t in 0..case.spec.m {
            for h in 0..case.spec.n_head {
                for dim in 0..case.spec.head_dim {
                    let at = (t * case.spec.n_head + h) * case.spec.head_dim + dim;
                    let (got, expected) = (outcome.out[at], want[at]);
                    assert_eq!(
                        got.to_bits(),
                        expected.to_bits(),
                        "{name}: t {t} h {h} dim {dim}: device {got:e}, host {expected:e}"
                    );
                }
            }
        }
    }
}
