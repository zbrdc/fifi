//! Seam-1 device tests (3090) of the KV8 split-KV decode attention wrapper: the Rust wrapper
//! against the C entry against the two kernels, compared with a plain f64 host reference that
//! reads the SAME dequantized values (`code * f32(stored fp16 scale)`), inside the frozen
//! reduction class E3 (the class of attn_decode, tests.rs).
//!
//! The reference is written from the window and ring convention of the fp16 contract (token `t`
//! is at position `P = kv_len - 1 + t` and sees `p <= P` with `P - p < window`; position `p`
//! lives in cell `p` or `p % ring`), not from the kernel. The int8 codes and the fp16 scales are
//! generated directly, so the reference and the kernel read one set of dequantized values.
//! Cells that no token of a call may read are int8 -128 with a NaN scale (0x7e00), and `part`
//! starts as NaN, so an out-of-range read poisons the result.
//!
//! The device suite runs with `--test-threads=1`. The fixed combine order is a property the E3
//! comparisons above cannot see (they never read `part`), but it IS pinned: the last test of
//! this file owns the `part` scratch, reads the partials back and requires the device `out` to
//! be bit-equal to an index-order host combine of exactly those partials.
//!
//! Each refusal is checked by the text of its error, so a test fails when another check refuses
//! the call instead of the one under test.

use super::{AttnDecodeQ8Call, AttnQ8Shape, AttnSpan, AttnSplit, attn_q8_part_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

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

/// Adds the phase's row label to a device error, so a drain or upload failure names the row
/// whose call hit it again.
fn labelled(label: &str, error: EngError) -> EngError {
    match error {
        EngError::Backend { operation, message } => EngError::Backend {
            operation,
            message: format!("{label}: {message}"),
        },
        other => other,
    }
}

mod sinks;

/// Class E3 (frozen): `max|d| <= 2^-17 * max(max|ref|, 1)` per head vector.
const E3: f64 = 1.0 / 131_072.0;
/// Two times the largest per-head query spread of the llama.cpp oracle dumps (1.691144648391655):
/// a case with a larger `q` standard deviation is a stress case, held to a sanity bound only.
const IN_RANGE_QSD: f64 = 3.382;
const STRESS_SANITY: f64 = 1e-2;
/// An in-range maximum above 2^-19 is printed as a headroom warning.
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

/// The whole buffer as a range, or its range error (the Result-only phases use it through
/// `?`; every owner drops before they return, so a failure must not panic).
fn whole(b: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    b.range(0, b.bytes())
}

/// Uploads the problem, queues one call, waits and returns `out` through `?`. `out` starts as
/// 0xFF bytes. Called only inside a Result-only device phase.
fn launch(rig: &Rig, p: &Problem, fill: PartFill) -> Result<Vec<f32>, EngError> {
    let (ctx, s) = (&rig.ctx, &p.spec);
    let part_len = attn_q8_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits)?;
    let out_len = s.m * s.n_head * s.head_dim * 4;
    let q = upload_f32(ctx, &p.q)?;
    let proj = s.gate.then(|| upload_f32(ctx, &p.proj)).transpose()?;
    let k_codes = upload_bytes(ctx, &le_i8(&p.k_codes))?;
    let k_scales = upload_bytes(ctx, &le16(&p.k_scales))?;
    let v_codes = upload_bytes(ctx, &le_i8(&p.v_codes))?;
    let v_scales = upload_bytes(ctx, &le16(&p.v_scales))?;
    let word = upload_bytes(ctx, &s.kv_len.to_le_bytes())?;
    let part = upload_bytes(ctx, &part_content(part_len, fill))?;
    let out = upload_filled(ctx, 0xFF, out_len)?;
    let call = AttnDecodeQ8Call {
        shape: s.shape(),
        span: s.span(),
        split: s.split,
        scale: s.scale(),
        q: whole(&q)?,
        proj: proj.as_ref().map(whole).transpose()?,
        k_codes: whole(&k_codes)?,
        k_scales: whole(&k_scales)?,
        v_codes: whole(&v_codes)?,
        v_scales: whole(&v_scales)?,
        kv_len: whole(&word)?,
        part: whole(&part)?,
        out: whole(&out)?,
    };
    rig.ctx.attn_decode_q8(&rig.stream, call)?;
    rig.stream.synchronize()?;
    Ok(testkit::floats(&read_bytes(ctx, &out, out_len)?))
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

/// Worst `max|dev - ref| / max(max|ref|, 1)` over the head vectors of `heads` for every token,
/// with the first non-finite head vector (none expected). The computation asserts nothing: the
/// caller asserts after the device phase has dropped every owner.
fn worst_error(p: &Problem, out: &[f32], heads: &[usize]) -> (f64, Option<(usize, usize)>) {
    let s = &p.spec;
    let mut worst = 0.0f64;
    let mut non_finite = None;
    for (t, &h) in (0..s.m).flat_map(|t| heads.iter().map(move |h| (t, h))) {
        let want = reference(p, t, h);
        let got = &out[(t * s.n_head + h) * s.head_dim..][..s.head_dim];
        if non_finite.is_none() && got.iter().any(|x| !x.is_finite()) {
            non_finite = Some((t, h));
        }
        let top = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        let diff = got
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - w).abs()));
        worst = worst.max(diff / top);
    }
    (worst, non_finite)
}

/// A device context and its stream, owned by one device phase.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        Ok(Self { ctx, stream })
    }
}

/// One measured device run: the spec label, the worst error, whether the case is a stress
/// case, and the first non-finite head vector (none expected).
struct RunOutcome {
    label: String,
    error: f64,
    stress: bool,
    non_finite: Option<(usize, usize)>,
}

/// Runs `spec` on the device and measures it against the host reference. Asserts nothing: the
/// caller asserts after the device phase drops every owner.
fn run_case(rig: &Rig, spec: &Spec, heads: &[usize]) -> Result<RunOutcome, EngError> {
    let p = problem(spec, false);
    let (error, non_finite) = worst_error(&p, &launch(rig, &p, PartFill::Nan)?, heads);
    let stress = spec.qsd > IN_RANGE_QSD;
    println!("max error {error:e} stress={stress} {}", spec.label());
    Ok(RunOutcome {
        label: spec.label(),
        error,
        stress,
        non_finite,
    })
}

fn run_all(rig: &Rig, spec: &Spec) -> Result<RunOutcome, EngError> {
    let heads = (0..spec.n_head).collect::<Vec<_>>();
    run_case(rig, spec, &heads)
}

/// Asserts one measured run: every output element finite, and the error inside the E3 bound
/// (in range) or the stress sanity bound.
fn assert_case(outcome: &RunOutcome) {
    if let Some((t, h)) = outcome.non_finite {
        panic!("non-finite output at token {t} head {h}: {}", outcome.label);
    }
    let bound = if outcome.stress { STRESS_SANITY } else { E3 };
    assert!(
        outcome.error <= bound,
        "{}: {:e} is above {:e}",
        outcome.label,
        outcome.error,
        bound
    );
}

/// The device phase of the reference runs: one Rig for every spec, dropped with its buffers
/// before the plain outcomes return, so the asserts never run inside a live backend (a drop
/// while panicking latches it).
fn reference_phase(specs: &[(Spec, Vec<usize>)]) -> Result<Vec<RunOutcome>, EngError> {
    let rig = Rig::new()?;
    specs
        .iter()
        .map(|(spec, heads)| run_case(&rig, spec, heads))
        .collect()
}

/// The device phase of the all-heads reference runs, for the geometry sweeps.
fn reference_phase_all(specs: &[Spec]) -> Result<Vec<RunOutcome>, EngError> {
    let rig = Rig::new()?;
    specs.iter().map(|spec| run_all(&rig, spec)).collect()
}

/// Prints the in-range and the stress maxima, with the headroom and stress warnings.
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
        println!("{test}: HEADROOM WARNING: in-range maximum above 2^-19");
    }
    if stress > 2.0 * E3 {
        println!("{test}: FINDING: a stress maximum is above 2^-16");
    }
}

#[test]
fn attn_decode_q8_matches_the_host_reference() {
    if !testkit::device_present("attn_decode_q8_matches_the_host_reference") {
        return;
    }
    let mut specs: Vec<(Spec, Vec<usize>)> = [1, 2, 31, 32, 33, 64, 65, 100, 129, 1000, 1025, 4097]
        .map(|kv| (Spec::qwen(1, kv), (0..16).collect()))
        .to_vec();
    for kv in [33, 100, 1025] {
        for (max_splits, min_keys) in [(1, 64), (4, 64), (64, 128)] {
            specs.push((
                Spec {
                    split: AttnSplit {
                        max_splits,
                        min_keys,
                    },
                    ..Spec::qwen(1, kv)
                },
                (0..16).collect(),
            ));
        }
    }
    for (m, kv) in [2, 3, 5, 16]
        .into_iter()
        .flat_map(|m| [1, 33, 100].map(|kv| (m, kv)))
    {
        specs.push((Spec::qwen(m, kv), (0..16).collect()));
    }
    specs.extend([1, 16].map(|m| (Spec::qwen(m, (4200 - (m - 1)) as i32), (0..16).collect())));
    specs.extend([6.0, 20.0, 0.0].map(|qsd| {
        (
            Spec {
                qsd,
                ..Spec::qwen(1, 1025)
            },
            (0..16).collect(),
        )
    }));
    // The long-context rows, one head per kv head: 32,768 and 131,072 keys (both inside one
    // hold of the gate run).
    for kv_len in [32_768_u32, 131_072] {
        let spec = Spec {
            cells: kv_len as usize + 64,
            ..Spec::qwen(1, kv_len as i32)
        };
        specs.push((spec, vec![0, 8]));
    }
    let outcomes = reference_phase(&specs).expect("the device phase of the reference runs");
    for outcome in &outcomes {
        assert_case(outcome);
    }
    report(
        "T1",
        &outcomes
            .iter()
            .map(|o| (o.error, o.stress))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn attn_decode_q8_matches_the_reference_for_head_128_window_ring_and_no_gate() {
    if !testkit::device_present(
        "attn_decode_q8_matches_the_reference_for_head_128_window_ring_and_no_gate",
    ) {
        return;
    }
    // (`Spec::other(head_dim, window, ring, cells, gate)`, tokens per call, kv_len values)
    #[rustfmt::skip]
    let groups: [(Spec, &[usize], &[i32]); 5] = [
        (Spec::other(128, 0, 0, 3000, false), &[1, 3], &[1, 33, 1025, 2049]),
        (Spec::other(128, 0, 0, 1200, true), &[1, 2], &[33, 1025]),
        (Spec::other(256, 64, 0, 1100, true), &[1, 3], &[10, 64, 65, 200, 1000]),
        (Spec::other(256, 64, 79, 79, true), &[1, 16], &[10, 64, 65, 300]),
        (Spec::other(128, 1024, 1039, 2200, false), &[1, 5], &[1100, 1500, 3000]),
    ];
    let mut specs = Vec::new();
    for (geometry, tokens, lengths) in groups {
        for (&m, &kv_len) in tokens
            .iter()
            .flat_map(|m| lengths.iter().map(move |kv| (m, kv)))
        {
            specs.push(Spec {
                m,
                kv_len,
                ..geometry.clone()
            });
        }
    }
    let outcomes = reference_phase_all(&specs).expect("the device phase of the reference runs");
    for outcome in &outcomes {
        assert_case(outcome);
    }
    report(
        "T1b",
        &outcomes
            .iter()
            .map(|o| (o.error, o.stress))
            .collect::<Vec<_>>(),
    );
}

/// One determinism outcome: the spec label and the `out` vectors of its two launches.
struct PairOutcome {
    label: String,
    first: Vec<f32>,
    second: Vec<f32>,
}

/// The device phase of the determinism cases: two launches per spec (the `part` scratch drawn
/// from two different seeds), every owner dropped before the plain vectors return.
fn determinism_phase(specs: &[Spec]) -> Result<Vec<PairOutcome>, EngError> {
    let rig = Rig::new()?;
    let mut outcomes = Vec::new();
    for spec in specs {
        let p = problem(spec, false);
        let first = launch(&rig, &p, PartFill::Finite(11))?;
        let second = launch(&rig, &p, PartFill::Finite(12))?;
        outcomes.push(PairOutcome {
            label: spec.label(),
            first,
            second,
        });
    }
    Ok(outcomes)
}

#[test]
fn attn_decode_q8_is_deterministic() {
    if !testkit::device_present("attn_decode_q8_is_deterministic") {
        return;
    }
    let ring = Spec {
        m: 2,
        kv_len: 300,
        ..Spec::other(128, 64, 79, 79, false)
    };
    let specs = [Spec::qwen(1, 1025), Spec::qwen(3, 700), ring];
    let outcomes = determinism_phase(&specs).expect("the device phase of the determinism cases");
    for outcome in &outcomes {
        assert!(
            outcome.first.iter().all(|x| x.is_finite()),
            "{}: a non-finite first output element",
            outcome.label
        );
        assert!(
            outcome.second.iter().all(|x| x.is_finite()),
            "{}: a non-finite second output element",
            outcome.label
        );
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&outcome.first),
            bits(&outcome.second),
            "{}",
            outcome.label
        );
    }
}

/// Launches `spec` with finite planes (the length word may be invalid) and returns `out`
/// through `?`; called only inside a Result-only device phase.
fn launch_finite(rig: &Rig, spec: &Spec) -> Result<Vec<f32>, EngError> {
    launch(rig, &problem(spec, true), PartFill::Nan)
}

/// One invalid-length outcome: its label and the device `out` of the launch.
struct InvalidOutcome {
    label: String,
    out: Vec<f32>,
}

/// The device phase of the invalid-length cases: the linear and ring words that must mark the
/// whole `out` NaN, then the two valid boundary runs, every owner dropped before the plain
/// vectors return.
fn invalid_length_phase(
    linear: &Spec,
    ring: &Spec,
    edge: i32,
    words: &[i32],
    ring_words: &[i32],
) -> Result<(Vec<InvalidOutcome>, RunOutcome, RunOutcome), EngError> {
    let rig = Rig::new()?;
    let mut invalid = Vec::new();
    for &kv_len in words {
        let out = launch_finite(
            &rig,
            &Spec {
                kv_len,
                ..linear.clone()
            },
        )?;
        invalid.push(InvalidOutcome {
            label: format!("linear kv_len {kv_len}"),
            out,
        });
    }
    let edge_run = run_all(
        &rig,
        &Spec {
            kv_len: edge,
            ..linear.clone()
        },
    )?;
    for &kv_len in ring_words {
        let out = launch_finite(
            &rig,
            &Spec {
                kv_len,
                ..ring.clone()
            },
        )?;
        invalid.push(InvalidOutcome {
            label: format!("ring kv_len {kv_len}"),
            out,
        });
    }
    let ring_run = run_all(&rig, ring)?;
    Ok((invalid, edge_run, ring_run))
}

#[test]
fn attn_decode_q8_marks_an_invalid_kv_len_with_nan() {
    if !testkit::device_present("attn_decode_q8_marks_an_invalid_kv_len_with_nan") {
        return;
    }
    let (cells, m) = (600usize, 2usize);
    let linear = Spec {
        cells,
        ..Spec::qwen(m, 1)
    };
    let edge = (cells - (m - 1)) as i32;
    let ring = Spec::other(256, 64, 79, 79, true);
    let (invalid, edge_run, ring_run) = invalid_length_phase(
        &linear,
        &ring,
        edge,
        &[0, -3, cells as i32 + 1, edge + 1],
        &[0, 1 << 30],
    )
    .expect("the device phase of the invalid-length cases");
    for outcome in &invalid {
        assert!(outcome.out.iter().all(|x| x.is_nan()), "{}", outcome.label);
    }
    assert_case(&edge_run);
    assert_case(&ring_run);
}

/// The Result-only device phase of the poison row: queues one call through `?`, drains, reads
/// `out`, and drops the context, the stream and every buffer as it returns, so no assert,
/// expect or unwrap runs while a CUDA owner is alive. `out` starts as 0xFF
/// bytes and the `part` scratch as NaN, exactly as `launch` prepares a case.
fn poison_device_phase(p: &Problem) -> Result<Vec<f32>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let s = &p.spec;
    let part_len = attn_q8_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits)?;
    let out_len = s.m * s.n_head * s.head_dim * 4;
    let q = testkit::upload_f32(&ctx, &p.q);
    let proj = s.gate.then(|| testkit::upload_f32(&ctx, &p.proj));
    let k_codes = testkit::upload_bytes(&ctx, &le_i8(&p.k_codes));
    let k_scales = testkit::upload_bytes(&ctx, &le16(&p.k_scales));
    let v_codes = testkit::upload_bytes(&ctx, &le_i8(&p.v_codes));
    let v_scales = testkit::upload_bytes(&ctx, &le16(&p.v_scales));
    let word = testkit::upload_bytes(&ctx, &s.kv_len.to_le_bytes());
    let part = testkit::upload_bytes(&ctx, &part_content(part_len, PartFill::Nan));
    let out = testkit::upload_filled(&ctx, 0xFF, out_len);
    let call = AttnDecodeQ8Call {
        shape: s.shape(),
        span: s.span(),
        split: s.split,
        scale: s.scale(),
        q: q.range(0, q.bytes())?,
        proj: proj.as_ref().map(|b| b.range(0, b.bytes())).transpose()?,
        k_codes: k_codes.range(0, k_codes.bytes())?,
        k_scales: k_scales.range(0, k_scales.bytes())?,
        v_codes: v_codes.range(0, v_codes.bytes())?,
        v_scales: v_scales.range(0, v_scales.bytes())?,
        kv_len: word.range(0, word.bytes())?,
        part: part.range(0, part.bytes())?,
        out: out.range(0, out.bytes())?,
    };
    ctx.attn_decode_q8(&stream, call)?;
    stream.synchronize()?;
    Ok(testkit::floats(&testkit::read_bytes(&ctx, &out, out_len)))
}

/// (e) The stored NaN group reaches attention: a visible cache cell whose K and V group 0 hold
/// the fp16 NaN scale the NaN-propagating `kv8_append` kernel stores (the output pin of
/// `kv8_append/tests.rs`) makes every dimension of the 8 query heads of that kv head non-finite;
/// the other kv head's heads stay finite. The reference is not used: this is a pure finiteness
/// pin of the poison rule.
#[test]
fn attn_decode_q8_poison_reaches_the_heads_of_a_nan_stored_group() {
    if !testkit::device_present("attn_decode_q8_poison_reaches_the_heads_of_a_nan_stored_group") {
        return;
    }
    let spec = Spec::qwen(1, 33);
    let kvh = 1usize;
    let mut p = problem(&spec, false);
    let groups = spec.head_dim / 32;
    let cell = spec.cell_of(i64::from(spec.kv_len) - 1);
    let at = kvh * spec.cells * groups + cell * groups;
    p.k_scales[at] = F16_NAN;
    p.v_scales[at] = F16_NAN;
    let out = poison_device_phase(&p).expect("the device phase of the poison row");
    // Host data only from here (the phase dropped every owner before it returned).
    for h in 0..spec.n_head {
        let row = &out[h * spec.head_dim..][..spec.head_dim];
        if h / GROUP == kvh {
            assert!(
                row.iter().all(|x| x.is_nan()),
                "head {h}: a NaN stored group must poison every dimension: {}",
                spec.label()
            );
        } else {
            assert!(
                row.iter().all(|x| x.is_finite()),
                "head {h}: the other kv head must stay finite: {}",
                spec.label()
            );
        }
    }
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
/// result of the call and the `out` bytes afterwards through `?`. `other` supplies the foreign
/// stream and buffer. Called only inside a Result-only device phase.
fn attempt(
    rig: &Rig,
    other: &Rig,
    plan: &Plan,
) -> Result<(Result<(), EngError>, Vec<u8>), EngError> {
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
        .map(|(i, slot)| upload_bytes(ctx, &content(i, slot)))
        .collect::<Result<_, _>>()?;
    let shared = plan
        .alias
        .map(|(a, b)| -> Result<_, EngError> {
            let bytes = plan.slots[a].bytes.max(plan.slots[b].bytes);
            Ok((a, b, upload_filled(ctx, 0xFF, bytes)?))
        })
        .transpose()?;
    let foreign = upload_filled(&other.ctx, 0, plan.slots[Q].bytes)?;
    let range = |i: usize| -> Result<DeviceRange<'_>, EngError> {
        let buffer = match &shared {
            Some((a, b, buf)) if i == *a || i == *b => buf,
            _ if i == Q && plan.foreign_q => &foreign,
            _ => &bufs[i],
        };
        buffer.range(plan.slots[i].offset, plan.slots[i].bytes)
    };
    let call = AttnDecodeQ8Call {
        shape: AttnQ8Shape {
            ld: plan.ld,
            ..s.shape()
        },
        span: s.span(),
        split: s.split,
        scale: plan.scale,
        q: range(Q)?,
        proj: if s.gate { Some(range(PROJ)?) } else { None },
        k_codes: range(KC)?,
        k_scales: range(KS)?,
        v_codes: range(VC)?,
        v_scales: range(VS)?,
        kv_len: range(KV)?,
        part: range(PART)?,
        out: range(OUT)?,
    };
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let result = ctx.attn_decode_q8(stream, call);
    stream
        .synchronize()
        .map_err(|error| labelled(plan.label, error))?;
    let out = read_bytes(ctx, &bufs[OUT], plan.slots[OUT].bytes)?;
    Ok((result, out))
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

/// One attempted call: the result of the call, the `out` bytes afterwards, and whether the
/// plan aliased two ranges (whose launch may legally write `out`).
struct Attempt {
    result: Result<(), EngError>,
    out: Vec<u8>,
    aliased: bool,
}

/// The whole device phase of the refusal test: the base call, every refusal row and every
/// accepted row, then the base call again. Both rigs drop with every buffer before the plain
/// attempts return, so the asserts below never run inside a live backend.
struct RefusalPhase {
    first: Attempt,
    rows: Vec<Attempt>,
    accepted: Vec<Attempt>,
    again: Attempt,
}

fn refusal_phase(
    rows: &[Row],
    accepted_rows: &[(&'static str, Edit)],
) -> Result<RefusalPhase, EngError> {
    let (rig, other) = (Rig::new()?, Rig::new()?);
    let run = |plan: &Plan| -> Result<Attempt, EngError> {
        let (result, out) = attempt(&rig, &other, plan)?;
        Ok(Attempt {
            result,
            out,
            aliased: plan.alias.is_some(),
        })
    };
    let first = run(&Plan::base())?;
    let mut attempts = Vec::new();
    for (label, _needle, edit) in rows {
        let mut plan = Plan::base();
        plan.label = label;
        edit(&mut plan);
        attempts.push(run(&plan)?);
    }
    let mut accepted_attempts = Vec::new();
    for (label, edit) in accepted_rows {
        let mut plan = Plan::base();
        plan.label = label;
        edit(&mut plan);
        accepted_attempts.push(run(&plan)?);
    }
    let again = run(&Plan::base())?;
    Ok(RefusalPhase {
        first,
        rows: attempts,
        accepted: accepted_attempts,
        again,
    })
}

#[test]
fn attn_decode_q8_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("attn_decode_q8_rejects_bad_calls_before_any_launch") {
        return;
    }
    let (rows, accepted_rows) = (refusals(), accepted());
    let phase = refusal_phase(&rows, &accepted_rows).expect("the device phase of the refusal test");
    phase.first.result.expect("the base call is valid");
    for ((label, needle, _edit), attempt) in rows.iter().zip(&phase.rows) {
        let message = match &attempt.result {
            Err(EngError::BadRequest(message)) => message,
            other => panic!("{label}: expected a BadRequest, got {other:?}"),
        };
        assert!(
            message.contains(needle),
            "{label}: expected {needle:?}, got {message:?}"
        );
        assert!(
            attempt.aliased || attempt.out.iter().all(|&b| b == 0xFF),
            "{label}: out changed"
        );
    }
    for ((label, _edit), attempt) in accepted_rows.iter().zip(&phase.accepted) {
        if let Err(error) = &attempt.result {
            panic!("{label}: {error:?}");
        }
        assert!(
            testkit::floats(&attempt.out).iter().all(|x| x.is_finite()),
            "{label}"
        );
    }
    phase
        .again
        .result
        .expect("the backend is usable after the refusals");
    assert_eq!(
        phase.first.out, phase.again.out,
        "the base call after the refusals"
    );
}

// ---- the split-combine order, pinned bit-exactly against the kernel's own partials ----

/// The gate value of every gate row of the pin: 0, so the sigmoid `1 / (1 + expf(-0))` is
/// exactly 0.5 on the host and the device (the former value 30 made it exactly 1, which left
/// the combine's `* sig` stage unpinned: a kernel that drops it changed no bit).
const PIN_GATE: f32 = 0.0;

/// The K scale of a skewed split's hot key, the plain hot key scale, and the filler scale of
/// the pin. The skew and the near-hot filler gaps are both expressed through this pair.
const HOT_SCALE: f32 = 512.0;
const PLAIN_SCALE: f32 = 1.0;
const FILLER_SCALE: f32 = 1.0 / 64.0;

/// `split_plan` of `attn_decode_q8.cu` on the host (all integers, so exact): `n_active` splits
/// of `chunk` keys, a multiple of the 64-key tile, cover `count` keys.
fn pin_split_plan(count: usize, split: AttnSplit) -> (usize, usize) {
    // Degenerate inputs (count 0, cap 0, min_keys 0) must produce a plan, not a division
    // by zero: every divisor is clamped to at least 1 so a zero-count plan is
    // (chunk 64, 0 active).
    let n_split = count
        .div_ceil(split.min_keys.max(1))
        .max(1)
        .min(split.max_splits.max(1));
    let chunk = count.div_ceil(n_split).div_ceil(64).max(1) * 64;
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
///
/// With `near_hot` every split also holds one NEAR-HOT key at its second position: the hot
/// codes at the f16 scale `1 - (s + 1) * 2^-10` (every step an exact f16), so its score sits
/// `delta_s = (1 - scale_s) * hot` strictly below the shared split maximum while every filler
/// still underflows to +0. Its probability `expf(-delta_s)` makes the partials
/// `l_s = 1 + expf(-delta_s)` distinct, non-dyadic f32 values, which the device computes and
/// the pin reads back: the L fmaf sum of the combine then rounds differently in split-index
/// order than in any other order. Without it every `l_s` is exactly 1 (the weight of the one
/// hot key) and L is 8 in every order, so the A-tuned patterns cannot see the L loop at all.
/// The near-hot V is filler V; its contribution to `a[dim]` is whatever the device computed,
/// because the pin compares against the device's own partials: the partials need not be exact,
/// only the shared split maxima do. Returns the problem and the f64 score gap the case opens:
/// between the skewed split's hot key and the others' with a skew, between the hot key and the
/// fillers of every split with `near_hot` (the filler gap the test holds above the expf
/// underflow line), and 0.0 otherwise.
fn pin_problem(spec: &Spec, skew: Option<usize>, near_hot: bool, seed: u64) -> (Problem, f64) {
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
    // The near-hot cells (the second visible key of every split, when the case asks for them;
    // every split of the plan holds at least two keys), mapped to their split index the same
    // way. A cell that is both hot (for one token) and near (for another) stays hot: the hot
    // map wins the classification below.
    let mut near = HashMap::new();
    if near_hot {
        for t in 0..spec.m {
            let (first, count) = pin_visible(spec, t);
            let (chunk, n_active) = pin_split_plan(count, spec.split);
            for s in 0..n_active {
                if s * chunk + 1 < count {
                    near.insert(spec.cell_of(first + (s * chunk + 1) as i64), s);
                }
            }
        }
    }
    let readable = spec.readable();
    let mut q = Vec::with_capacity(spec.m * spec.n_head * d);
    let (mut k_codes, mut k_scales) = (Vec::new(), Vec::new());
    let (mut v_codes, mut v_scales) = (Vec::new(), Vec::new());
    for k in 0..spec.n_kv_head {
        for (cell, &read) in readable.iter().enumerate() {
            let (kc, ks, vc, vs): (Vec<i8>, f32, Vec<i8>, f32) = if let Some(&s) = hot.get(&cell) {
                let hot_k = if Some(s) == skew {
                    HOT_SCALE
                } else {
                    PLAIN_SCALE
                };
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
            } else if let Some(&s) = near.get(&cell) {
                // The near-hot key of split `s`: the hot codes at the f16 scale
                // `1 - (s + 1) / 1024`, one step of 2^-10 further from 1 per split, so its
                // score sits `delta_s` below the split maximum and `l_s` differs per split.
                let near_codes: Vec<i8> = q0[k]
                    .iter()
                    .map(|&x| if x >= 0.0 { 127 } else { -127 })
                    .collect();
                let fill = |rng: &mut Rng| (0..d).map(|_| filler(rng)).collect::<Vec<i8>>();
                (
                    near_codes,
                    PLAIN_SCALE - ((s + 1) as f32) / 1024.0,
                    fill(&mut rng),
                    FILLER_SCALE,
                )
            } else if read {
                let fill = |rng: &mut Rng| (0..d).map(|_| filler(rng)).collect::<Vec<i8>>();
                (fill(&mut rng), FILLER_SCALE, fill(&mut rng), FILLER_SCALE)
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
    // The f64 gap the case opens for one kv head, from the scale difference between its hot
    // key (`127 * sign(q)` codes) and the keys below it: with a skew the hot key of the skewed
    // split is at HOT_SCALE against the others' PLAIN_SCALE; with `near_hot` the plain hot key
    // is at PLAIN_SCALE against fillers whose score is at most `127 * sum |q| / 64`, so the
    // gap is `scale * 127 * (PLAIN_SCALE - FILLER_SCALE) * sum |q|`. The minimum over the kv
    // heads is what the test holds above the expf underflow line.
    let gap = |factor: f32| {
        q0.iter()
            .map(|q| {
                f64::from(spec.scale())
                    * 127.0
                    * f64::from(factor)
                    * q.iter().fold(0.0, |sum, &x| sum + f64::from(x.abs()))
            })
            .fold(f64::INFINITY, f64::min)
    };
    let gap = if skew.is_some() {
        gap(HOT_SCALE - PLAIN_SCALE)
    } else if near_hot {
        gap(PLAIN_SCALE - FILLER_SCALE)
    } else {
        0.0
    };
    (
        Problem {
            spec: spec.clone(),
            q: q.repeat(spec.m),
            proj,
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        },
        gap,
    )
}

/// The combine of `attn_q8_combine_k` in f32, from the device partials of `part`: the exact
/// block max over the active splits, the weights `expf(m_s - M)`, the L and A fmaf sums in
/// split-index order, the division and the optional sigmoid gate, with every operation and its
/// order read from the kernel. The data of the pin admits only two `expf` inputs: `m_s - M` is
/// 0 (the split maxima have the same bits, and `expf(0)` is exactly 1 in every implementation)
/// or, for the skewed split, deeply negative (`e^x` underflows to +0 on the host and the device
/// alike); the gate value 0 makes the sigmoid exactly 0.5 on both. So `f32::exp` matches the
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

/// One case of the pin on the device: geometry, the skewed split (none or 1), whether every
/// split also holds a near-hot key, and the seed.
struct PinCase {
    label: &'static str,
    spec: Spec,
    skew: Option<usize>,
    near_hot: bool,
    seed: u64,
}

/// One device result: the case label, its problem, the device `out`, the device `part`, and
/// the f64 score gap the case opened (0.0 without a skew or near-hot fillers).
struct PinOutcome {
    label: &'static str,
    problem: Problem,
    out: Vec<f32>,
    part: Vec<f32>,
    gap: f64,
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
    let q = upload_f32(ctx, &p.q)?;
    let proj = s.gate.then(|| upload_f32(ctx, &p.proj)).transpose()?;
    let k_codes = upload_bytes(ctx, &le_i8(&p.k_codes))?;
    let k_scales = upload_bytes(ctx, &le16(&p.k_scales))?;
    let v_codes = upload_bytes(ctx, &le_i8(&p.v_codes))?;
    let v_scales = upload_bytes(ctx, &le16(&p.v_scales))?;
    let word = upload_bytes(ctx, &s.kv_len.to_le_bytes())?;
    let part = upload_bytes(ctx, &part_content(part_bytes, PartFill::Nan))?;
    let out = upload_filled(ctx, 0xFF, out_bytes)?;
    ctx.attn_decode_q8(
        stream,
        AttnDecodeQ8Call {
            shape: s.shape(),
            span: s.span(),
            split: s.split,
            scale: s.scale(),
            q: whole(&q)?,
            proj: proj.as_ref().map(whole).transpose()?,
            k_codes: whole(&k_codes)?,
            k_scales: whole(&k_scales)?,
            v_codes: whole(&v_codes)?,
            v_scales: whole(&v_scales)?,
            kv_len: whole(&word)?,
            part: whole(&part)?,
            out: whole(&out)?,
        },
    )?;
    stream.synchronize()?;
    Ok((
        testkit::floats(&read_bytes(ctx, &out, out_bytes)?),
        testkit::floats(&read_bytes(ctx, &part, part_bytes)?),
    ))
}

/// The device phase: one launch per case, every device owner (buffers, stream, context) dropped
/// before the plain vectors return, so the assertions below cannot unwind through a drop.
fn pin_device_phase<'a>(
    cases: impl IntoIterator<Item = &'a PinCase>,
) -> Result<Vec<PinOutcome>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut outcomes = Vec::new();
    for case in cases {
        let (problem, gap) = pin_problem(&case.spec, case.skew, case.near_hot, case.seed);
        let (out, part) = pin_launch(&ctx, &stream, &problem)?;
        outcomes.push(PinOutcome {
            label: case.label,
            problem,
            out,
            part,
            gap,
        });
    }
    Ok(outcomes)
}

/// The combine's L sums of one case, in f32, from the device partials of `part`: the exact
/// block max, the weights `expf(m_s - M)` and the `fmaf` chain exactly as the kernel sums them
/// (`L = fmaf(s_l[s], s_w[s], L)`). One `(index, reversed)` pair per (token, kv head,
/// query-head group); the case is blind to the L order when every pair is bit-equal.
fn pin_l_orders(p: &Problem, part: &[f32]) -> Vec<(f32, f32)> {
    let s = &p.spec;
    let max_splits = s.split.max_splits;
    let mut orders = Vec::new();
    for t in 0..s.m {
        let (_, count) = pin_visible(s, t);
        let (_, n_active) = pin_split_plan(count, s.split);
        for kvh in 0..s.n_kv_head {
            let cell0 = (t * s.n_kv_head + kvh) * max_splits;
            for g in 0..GROUP {
                let at = |split: usize| (cell0 + split) * GROUP * 2 + g * 2;
                let top = (0..n_active).fold(f32::NEG_INFINITY, |m, split| m.max(part[at(split)]));
                let weights: Vec<f32> = (0..n_active)
                    .map(|split| (part[at(split)] - top).exp())
                    .collect();
                let mut index = 0.0f32;
                let mut reversed = 0.0f32;
                for split in 0..n_active {
                    index = part[at(split) + 1].mul_add(weights[split], index);
                }
                for split in (0..n_active).rev() {
                    reversed = part[at(split) + 1].mul_add(weights[split], reversed);
                }
                orders.push((index, reversed));
            }
        }
    }
    orders
}

/// The combine order of `attn_decode_q8` (KV8), pinned bit-exactly: the device `out` must equal
/// an index-order host combine of the partials the kernel itself left in `part`.
/// The A-tuned cases pin the A sum, the near-hot cases pin the L sum (their `l_s` are distinct
/// non-dyadic values, and a precondition on the device's own partials fails the test before the
/// pin if a case cannot see the reversed order), the skewed case pins the max and underflow
/// paths of the weights, and the gate value 0 pins the final `* sigmoid` factor. Every output
/// element is asserted finite before the bit comparison, so no NaN payload can pass it
/// vacuously.
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
    // underflows to +0 and the split maxima differ by orders of magnitude. The near-hot cases
    // (three extra seeds, so one blind draw cannot hide the L order) keep every split maximum
    // tied (every weight exactly `expf(0) = 1`) and give every split a distinct non-dyadic
    // `l_s`, so the L sum alone re-rounds under a reversed order. The four near-hot cases are
    // probed first and each must show a reversed L that differs in at least one bit.
    let cases = vec![
        PinCase {
            label: "qwen m=1 d=256 gate, 8 splits",
            spec: qwen(1, 1024, 1200),
            skew: None,
            near_hot: false,
            seed: 0x51,
        },
        PinCase {
            label: "qwen m=4 d=256 gate, 8 splits",
            spec: qwen(4, 1017, 1024),
            skew: None,
            near_hot: false,
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
            near_hot: false,
            seed: 0x53,
        },
        PinCase {
            label: "qwen m=1, split 1 K scale 512, 8 splits",
            spec: qwen(1, 1024, 1200),
            skew: Some(1),
            near_hot: false,
            seed: 0x54,
        },
        PinCase {
            label: "qwen m=1 d=256 gate, 8 near-hot splits, tied maxima",
            spec: qwen(1, 1024, 1200),
            skew: None,
            near_hot: true,
            seed: 0x55,
        },
        PinCase {
            label: "qwen m=1 d=256 gate, 8 near-hot splits, seed 0x56",
            spec: qwen(1, 1024, 1200),
            skew: None,
            near_hot: true,
            seed: 0x56,
        },
        PinCase {
            label: "qwen m=1 d=256 gate, 8 near-hot splits, seed 0x57",
            spec: qwen(1, 1024, 1200),
            skew: None,
            near_hot: true,
            seed: 0x57,
        },
        PinCase {
            label: "qwen m=1 d=256 gate, 8 near-hot splits, seed 0x58",
            spec: qwen(1, 1024, 1200),
            skew: None,
            near_hot: true,
            seed: 0x58,
        },
    ];
    // MAJOR F1: each near-hot case must see a reversed L sum in the device's own partials BY
    // CONSTRUCTION, not by the luck of the rounding. This probe runs before the pin's device
    // phase; a blind case fails loudly here with its label instead of passing the bit
    // comparison. The probe is one extra launch per near-hot case, the same launch the pin
    // makes (deterministic kernel, same inputs), so the checked l set is the pinned one.
    let probed = pin_device_phase(cases.iter().filter(|case| case.near_hot))
        .expect("the probe phase of the near-hot L preconditions");
    for outcome in &probed {
        let orders = pin_l_orders(&outcome.problem, &outcome.part);
        let differing: Vec<&(f32, f32)> = orders
            .iter()
            .filter(|(index, reversed)| index.to_bits() != reversed.to_bits())
            .collect();
        println!(
            "precondition {}: {} of {} L sets differ in index and reversed order",
            outcome.label,
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
            "{}: the near-hot case cannot see the L order: all {} of its (token, kv head, \
             group) L sets sum bit-identically in index and reversed order; reseed or reshape \
             the case",
            outcome.label,
            orders.len()
        );
    }
    let outcomes = pin_device_phase(&cases).expect("the device phase of the combine pin");
    for (case, outcome) in cases.iter().zip(&outcomes) {
        if case.skew.is_some() || case.near_hot {
            assert!(
                outcome.gap >= 120.0,
                "{}: the score gap {} is below the expf underflow line",
                outcome.label,
                outcome.gap
            );
        }
        assert!(
            outcome.out.iter().all(|x| x.is_finite()),
            "{}: a non-finite device output element before the bit comparison",
            outcome.label
        );
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

/// Prefetch-arm fixture: the prefetch arm must be bitwise-equal to the legacy arm
/// (the multiply-then-fmaf order per head is shared, so any difference is a defect), over
/// several depths, a batch, a ring, and distinct seeds. `part` and `out` are over-allocated
/// with poisoned guard tails past the bytes the call needs, so a read or write the kernel
/// must not make is visible; the per-row maximum of the legacy-vs-prefetch difference is
/// asserted zero exactly, with the row reported on failure. The distinct-query case first
/// proves on the f64 reference that its rows separate the heads, then asserts the DEVICE
/// outputs differ pairwise over every head pair. The whole device leg runs in a child
/// process: forced arms can never leak into another device test of this binary in this binary.
#[cfg(feature = "cuda-arms-qualification")]
#[test]
fn attn_decode_q8_prefetch_arm_is_bit_equal() {
    const NAME: &str = "attn_decode_q8_prefetch_arm_is_bit_equal";
    if !testkit::device_present(NAME) {
        return;
    }
    const CHILD_PATH: &str =
        "kernels::cuda::resident::attn_decode_q8::tests::attn_decode_q8_prefetch_arm_is_bit_equal";
    if !testkit::latching_test_in_child(CHILD_PATH, &["FIXTURE-LEGS-OK", "ARM-RECEIPTS-OK"]) {
        return;
    }
    // The arms and the attrs below target the register-V kernels: production enables the
    // arm only for R1 unset or exactly "1" and otherwise silently runs the original
    // kernel, so a leg that resolved to the original kernel would fail the dispatch-tag
    // assert below. Refuse here unless the arm is actually enabled.
    match std::env::var("INFERENCE_CUDA_KV8_ATTN_R1") {
        Err(std::env::VarError::NotPresent) => {}
        Ok(ref v) if v == "1" => {}
        other => panic!(
            "{NAME}: the fixture needs the register-V arm enabled; R1 must be unset or exactly '1', got {other:?}"
        ),
    }
    use crate::kernels::cuda::native::resident::attn_decode_q8::{
        attn_decode_q8_arm_attrs, attn_decode_q8_dispatched, attn_decode_q8_kernel_identity,
        attn_decode_q8_kernel_identity_reset, attn_decode_q8_set_arms,
    };

    // The two instantiations must be DIFFERENT kernels an arm toggle that
    // silently compares a kernel with itself fails here instead of passing vacuously.
    let (legacy_regs, legacy_smem) =
        attn_decode_q8_arm_attrs(0).unwrap_or_else(|e| panic!("{NAME}: legacy attrs: {e:?}"));
    let (prefetch_regs, prefetch_smem) =
        attn_decode_q8_arm_attrs(1).unwrap_or_else(|e| panic!("{NAME}: prefetch attrs: {e:?}"));
    assert_ne!(
        (legacy_regs, legacy_smem),
        (prefetch_regs, prefetch_smem),
        "{NAME}: the forced arms resolved to the same kernel image",
    );
    println!(
        "{NAME}: arm attrs legacy=({legacy_regs}r {legacy_smem}B) prefetch=({prefetch_regs}r {prefetch_smem}B)"
    );

    struct Case {
        label: &'static str,
        spec: Spec,
        seed: u64,
        distinct_queries: bool,
    }
    let mut ring = Spec::qwen(1, 4100);
    ring.cells = 4096;
    ring.window = 2048;
    ring.ring = 4096;
    let mut deep = Spec::qwen(1, 4224);
    deep.cells = 4232;
    let mut full = Spec::qwen(1, 4224);
    full.cells = 4224; // full linear cache: the visible count equals the cell count
    let mut partial = Spec::qwen(1, 4225); // 66 whole tiles + a one-key partial last tile
    partial.cells = 4232;
    let mut batch = Spec::qwen(2, 4160);
    batch.cells = 4168;
    let mut m16 = Spec::qwen(16, 4160); // the m cap: every warp row a distinct token
    m16.cells = 4175;
    let cases = [
        Case {
            label: "depth-96",
            spec: Spec::qwen(1, 96),
            seed: 0xC3C3,
            distinct_queries: false,
        },
        Case {
            label: "depth-1088-distinct-q",
            // No gate: its random per-head sigmoid would let heads differ even with
            // wrong q-head indexing; gate coverage lives in the pinned cases.
            spec: {
                let mut s = Spec::qwen(1, 1088);
                s.gate = false;
                s
            },
            seed: 0xC4C4,
            distinct_queries: true,
        },
        Case {
            label: "batch-4-distinct-q",
            // m=4: distinct queries must separate heads for every token, not just m=1;
            // no gate (its random per-head sigmoid would mask wrong q-head indexing).
            spec: {
                let mut s = Spec::qwen(4, 1088);
                s.gate = false;
                s
            },
            seed: 0xC6C6,
            distinct_queries: true,
        },
        Case {
            label: "depth-4224",
            spec: deep,
            seed: 0xC5C5,
            distinct_queries: false,
        },
        Case {
            label: "full-linear-4224",
            spec: full,
            seed: 0xCAC5,
            distinct_queries: false,
        },
        Case {
            label: "partial-tile-4225",
            spec: partial,
            seed: 0xCBC5,
            distinct_queries: false,
        },
        Case {
            label: "batch-m2-4160",
            spec: batch,
            seed: 0xC6C6,
            distinct_queries: false,
        },
        Case {
            label: "batch-m16-4160",
            spec: m16,
            seed: 0xC9C6,
            distinct_queries: false,
        },
        Case {
            label: "ring-2048",
            spec: ring,
            seed: 0xC7C7,
            distinct_queries: false,
        },
    ];

    /// One launch with poisoned guard tails: 0xA5 after `part`, 0x5A after `out`. Returns the
    /// raw bytes of both buffers including the guards. The guards prove out-of-range
    /// WRITES; a read past the end is caught only through the byte-exact output compare
    /// (poison read into an output changes those bytes), not by the guards alone.
    fn guarded_launch(
        ctx: &CudaContext,
        stream: &DeviceStream,
        p: &Problem,
        guard: usize,
    ) -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let s = &p.spec;
        let part_bytes = attn_q8_part_bytes(s.m, s.n_kv_head, s.head_dim, s.split.max_splits)?;
        let out_bytes = s.m * s.n_head * s.head_dim * 4;
        let q = upload_f32(ctx, &p.q)?;
        let proj = s.gate.then(|| upload_f32(ctx, &p.proj)).transpose()?;
        let k_codes = upload_bytes(ctx, &le_i8(&p.k_codes))?;
        let k_scales = upload_bytes(ctx, &le16(&p.k_scales))?;
        let v_codes = upload_bytes(ctx, &le_i8(&p.v_codes))?;
        let v_scales = upload_bytes(ctx, &le16(&p.v_scales))?;
        let word = upload_bytes(ctx, &s.kv_len.to_le_bytes())?;
        let mut part = part_content(part_bytes, PartFill::Nan);
        part.extend(std::iter::repeat_n(0xA5, guard));
        let mut out = vec![0xFF; out_bytes];
        out.extend(std::iter::repeat_n(0x5A, guard));
        let part = upload_bytes(ctx, &part)?;
        let out = upload_bytes(ctx, &out)?;
        ctx.attn_decode_q8(
            stream,
            AttnDecodeQ8Call {
                shape: s.shape(),
                span: s.span(),
                split: s.split,
                scale: s.scale(),
                q: whole(&q)?,
                proj: proj.as_ref().map(whole).transpose()?,
                k_codes: whole(&k_codes)?,
                k_scales: whole(&k_scales)?,
                v_codes: whole(&v_codes)?,
                v_scales: whole(&v_scales)?,
                kv_len: whole(&word)?,
                part: whole(&part)?,
                out: whole(&out)?,
            },
        )?;
        stream.synchronize()?;
        Ok((
            read_bytes(ctx, &part, part_bytes + guard)?,
            read_bytes(ctx, &out, out_bytes + guard)?,
        ))
    }

    const GUARD: usize = 64;
    struct ArmOutcome {
        out: Vec<f32>,
        part_raw: Vec<u8>,
        part_guard: Vec<u8>,
        out_guard: Vec<u8>,
    }
    let arm = |ctx: &CudaContext,
               stream: &DeviceStream,
               p: &Problem,
               prefetch: i32|
     -> Result<ArmOutcome, EngError> {
        attn_decode_q8_set_arms(prefetch)
            .unwrap_or_else(|code| panic!("arm force {prefetch} rejected: {code}"));
        let (mut part_raw, out_raw) = guarded_launch(ctx, stream, p, GUARD)?;
        let out_len = out_raw.len() - GUARD;
        let part_guard = part_raw.split_off(part_raw.len() - GUARD);
        Ok(ArmOutcome {
            out: testkit::floats(&out_raw[..out_len]),
            part_raw,
            part_guard,
            out_guard: out_raw[out_len..].to_vec(),
        })
    };

    let ctx = match CudaContext::new() {
        Ok(ctx) => ctx,
        Err(error) => panic!("{NAME}: no CUDA context: {error:?}"),
    };
    let stream = ctx.stream().expect("a stream");
    // A panic on any leg must still restore the arms (forced arms left behind by an
    // early `.expect` poison every later launch in this process).
    struct ArmRestore;
    impl Drop for ArmRestore {
        fn drop(&mut self) {
            let _ = attn_decode_q8_set_arms(-1);
        }
    }
    let _arm_restore = ArmRestore;
    for case in &cases {
        // The distinct case uses problem(): its q rows are drawn independently per
        // (token, head), and its keys carry no planted hot key (pin_problem's first-key
        // sign pattern at +-127 saturates the softmax - one weight of exactly 1.0 in fp32
        // for every head once a split holds enough keys - which made distinct queries
        // return bit-identical outputs; confirmed fixture-side).
        // Every other case keeps pin_problem and its hot-key structure on purpose.
        let problem = if case.distinct_queries {
            problem(&case.spec, true)
        } else {
            pin_problem(&case.spec, None, false, case.seed).0
        };
        if case.distinct_queries {
            // Data precondition on the f64 reference (before any device launch): the distinct
            // rows must actually separate the heads, else the head-distinctness assert below
            // is vacuous and the DATA is at fault (saturation), not the kernel.
            let s = &problem.spec;
            let mut worst_rel = 0.0f64;
            for t in 0..s.m {
                let a = reference(&problem, t, 0);
                let b = reference(&problem, t, 1);
                let denom = a.iter().fold(0.0f64, |x, &v| x.max(v.abs())).max(1.0);
                worst_rel = worst_rel
                    .max(a.iter().zip(&b).map(|(&x, &y)| (x - y).abs()).sum::<f64>() / denom);
            }
            assert!(
                worst_rel > 1e-3,
                "{}: reference heads 0/1 differ only {worst_rel:.2e} - the case data saturates (one hot key takes all the mass); fix the generator, not the kernel",
                case.label
            );
        }
        // Explicit 0 and 1: never env-follow, so the two legs cannot silently be the same
        // kernel or bypass register-V entirely (the two arms exist).
        // Device-written kernel identity: the kernel itself stamps its id into a device
        // word (thread (0,0,0) of the launch), read back after the leg. A launch
        // statement that resolves to the WRONG kernel leaves the other leg's stamp (or
        // zero) here even though the host dispatch tag above still names the intent,
        // so this is the proof that the executed kernel, not the enqueue path, is the
        // one under test.
        attn_decode_q8_kernel_identity_reset()
            .unwrap_or_else(|code| panic!("{}: identity reset failed: {code}", case.label));
        let legacy = arm(&ctx, &stream, &problem, 0).expect("the legacy arm");
        // Host dispatch tag (1 legacy, 2 prefetch), read right after each leg: proves
        // which enqueue path ran. A leg resolved to the original kernel gives 0.
        let legacy_tag = attn_decode_q8_dispatched();
        let legacy_ran = attn_decode_q8_kernel_identity()
            .unwrap_or_else(|code| panic!("{}: identity read failed: {code}", case.label));
        attn_decode_q8_kernel_identity_reset()
            .unwrap_or_else(|code| panic!("{}: identity reset failed: {code}", case.label));
        let prefetch = arm(&ctx, &stream, &problem, 1).expect("the prefetch arm");
        let prefetch_tag = attn_decode_q8_dispatched();
        let prefetch_ran = attn_decode_q8_kernel_identity()
            .unwrap_or_else(|code| panic!("{}: identity read failed: {code}", case.label));
        assert_eq!(
            legacy_tag, 1,
            "{}: the legacy leg did not dispatch the legacy kernel",
            case.label
        );
        assert_eq!(
            prefetch_tag, 2,
            "{}: the prefetch leg did not dispatch the prefetch kernel",
            case.label
        );
        assert_eq!(
            legacy_ran, 1,
            "{}: the DEVICE identity after the legacy leg is not the legacy kernel's",
            case.label
        );
        assert_eq!(
            prefetch_ran, 2,
            "{}: the DEVICE identity after the prefetch leg is not the prefetch kernel's",
            case.label
        );

        // Every asserted element is finite first, so no NaN can pass a comparison vacuously.
        for (label, values) in [("legacy", &legacy.out), ("prefetch", &prefetch.out)] {
            for (index, value) in values.iter().enumerate() {
                assert!(
                    value.is_finite(),
                    "{}: {label} out[{index}] is {value}",
                    case.label
                );
            }
        }
        // Guards: a kernel that read or wrote past its buffers cannot pass this.
        assert_eq!(
            legacy.part_guard,
            vec![0xA5; GUARD],
            "{}: part guard",
            case.label
        );
        assert_eq!(
            legacy.out_guard,
            vec![0x5A; GUARD],
            "{}: out guard",
            case.label
        );
        assert_eq!(
            prefetch.part_guard,
            vec![0xA5; GUARD],
            "{}: part guard",
            case.label
        );
        assert_eq!(
            prefetch.out_guard,
            vec![0x5A; GUARD],
            "{}: out guard",
            case.label
        );
        // Bitwise identity of out and part (raw bytes: the untouched poison cells hold NaN
        // bits that byte comparison keeps exact) and the per-row maximum on failure.
        assert_eq!(
            legacy.part_raw, prefetch.part_raw,
            "{}: part bytes differ between the legacy and prefetch arms",
            case.label
        );
        let spec = &case.spec;
        for t in 0..spec.m {
            for h in 0..spec.n_head {
                let row = &legacy.out[(t * spec.n_head + h) * spec.head_dim..][..spec.head_dim];
                let other = &prefetch.out[(t * spec.n_head + h) * spec.head_dim..][..spec.head_dim];
                let max = row
                    .iter()
                    .zip(other)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert_eq!(
                    max, 0.0,
                    "{}: token {t} head {h} differs between the arms (max {max:e})",
                    case.label
                );
                let row_bits: Vec<u32> = row.iter().map(|x| x.to_bits()).collect();
                let other_bits: Vec<u32> = other.iter().map(|x| x.to_bits()).collect();
                assert_eq!(
                    row_bits, other_bits,
                    "{}: token {t} head {h} bits",
                    case.label
                );
            }
        }
        // Per-head f64 reference check: every head's device row must match the f64
        // reference within fp32 softmax tolerance. Wrong q-head indexing (the failure
        // mode pairwise head-distinctness can miss under gating) produces O(1)
        // differences here.
        {
            let mut worst = 0.0f64;
            for t in 0..spec.m {
                for h in 0..spec.n_head {
                    let r = reference(&problem, t, h);
                    // A NaN reference element would make f64::max skip the diff and the
                    // bound pass vacuously, so finiteness is asserted up front.
                    assert!(
                        r.iter().all(|v| v.is_finite()),
                        "{}: f64 reference not finite at token {t} head {h}",
                        case.label
                    );
                    let row = &legacy.out[(t * spec.n_head + h) * spec.head_dim..][..spec.head_dim];
                    let denom = r.iter().fold(0.0f64, |x, &v| x.max(v.abs())).max(1.0);
                    let diff = row
                        .iter()
                        .zip(&r)
                        .map(|(&a, &b)| (a as f64 - b).abs())
                        .fold(0.0f64, f64::max);
                    worst = worst.max(diff / denom);
                }
            }
            assert!(
                worst <= 1e-3,
                "{}: device vs f64 reference, worst relative head error {worst:e}",
                case.label
            );
        }
        if case.distinct_queries {
            // The perturbed queries must actually reach the outputs: EVERY pair of heads of
            // one token must differ, so the case is not vacuously identical across heads
            // (pairs inside a KV group share V rows, the strongest form of the check).
            let d = spec.head_dim;
            for t in 0..spec.m {
                for h0 in 0..spec.n_head {
                    for h1 in (h0 + 1)..spec.n_head {
                        let a = &legacy.out[(t * spec.n_head + h0) * d..][..d];
                        let b = &legacy.out[(t * spec.n_head + h1) * d..][..d];
                        assert_ne!(
                            a, b,
                            "{}: token {t} heads {h0}/{h1} identical under distinct queries",
                            case.label
                        );
                    }
                }
            }
        }
        println!(
            "{}: the prefetch arm is bitwise-equal to the legacy arm",
            case.label
        );
    }
    println!("ARM-RECEIPTS-OK");
    println!("FIXTURE-LEGS-OK");
}

/// Host-only split-plan pin (moved out of the device test): the
/// production cap 64 at 65,536 keys gives an even 1024-key chunk over 64 active splits.
#[cfg(feature = "cuda-arms-qualification")]
#[test]
fn attn_decode_q8_split_plan_pin_64_cap_at_64k_keys() {
    use crate::kernels::cuda::native::resident::attn_decode_q8::attn_decode_q8_split_plan;
    let (chunk, n_active) = attn_decode_q8_split_plan(65_536, 64, 64)
        .unwrap_or_else(|e| panic!("split plan pin: {e:?}"));
    assert_eq!(
        (chunk, n_active),
        (1024, 64),
        "the 64-cap plan at 65,536 keys"
    );
    let (chunk, n_active) = attn_decode_q8_split_plan(65_536, 61, 64)
        .unwrap_or_else(|e| panic!("split plan pin (61 cap): {e:?}"));
    assert_eq!(
        (chunk, n_active),
        (1088, 61),
        "the 61-cap plan at 65,536 keys"
    );
    let (chunk, n_active) = attn_decode_q8_split_plan(65_536, 41, 64)
        .unwrap_or_else(|e| panic!("split plan pin (41 cap): {e:?}"));
    assert_eq!(
        (chunk, n_active),
        (1600, 41),
        "the 41-cap plan at 65,536 keys"
    );
    assert!(
        attn_decode_q8_split_plan(0, 64, 64).is_err(),
        "degenerate n_keys must be rejected, not divide by zero"
    );
    assert!(
        attn_decode_q8_split_plan(i32::MAX, 64, 64).is_err(),
        "i32::MAX keys overflow want's sum and must be rejected"
    );
    assert!(
        attn_decode_q8_split_plan(2_147_483_600, 1, 1).is_err(),
        "only the tile round-up overflows here: c0+63 crosses INT_MAX"
    );
    // The exact accepted edge at cap 64 under addition bounds: n_keys + chunk is
    // INT_MAX here; one key more and that ADDITION itself overflows on device
    // before the -1 (C evaluates (a + b) - 1).
    let (chunk, n_active) = attn_decode_q8_split_plan(2_114_445_375, 64, 64)
        .unwrap_or_else(|e| panic!("split plan pin (edge): {e:?}"));
    assert_eq!(
        (chunk, n_active),
        (33_038_272, 64),
        "the 64-cap plan at the exact int overflow edge"
    );
    assert!(
        attn_decode_q8_split_plan(2_114_445_376, 64, 64).is_err(),
        "one key past the edge, n_keys + chunk = INT_MAX + 1 overflows the addition"
    );

    // Sweep: the host mirror must equal the pure-Rust model of the device arithmetic
    // for every plausible key count and the four qualification caps.
    for cap in [1, 41, 61, 64, 128] {
        for keys in 1..=70_000 {
            let host = attn_decode_q8_split_plan(keys, cap, 64).expect("sweep host");
            let model = pin_split_plan(
                keys as usize,
                AttnSplit {
                    max_splits: cap as usize,
                    min_keys: 64,
                },
            );
            assert_eq!(
                (host.0 as usize, host.1 as usize),
                model,
                "mirror vs model drift at keys {keys} cap {cap}"
            );
        }
    }
}
#[cfg(feature = "cuda-arms-qualification")]
#[test]
fn attn_decode_q8_split_plan_device_probe_pins_the_host_mirror() {
    use crate::kernels::cuda::native::resident::attn_decode_q8::{
        attn_decode_q8_split_plan, attn_decode_q8_split_plan_device,
    };
    if !testkit::device_present("attn_decode_q8_split_plan_device_probe_pins_the_host_mirror") {
        return;
    }
    // The DEVICE helper must agree with the host mirror on the same tuples: this pins
    // both copies, not only the host mirror. GPU-guarded: the probe allocates and
    // launches, so a GPU-less feature run skips it instead of panicking.
    for &(keys, cap, expect) in &[
        (65_536, 64, (1024, 64)),
        (65_536, 61, (1088, 61)),
        (65_536, 41, (1600, 41)),
        (2_114_445_375, 64, (33_038_272, 64)),
    ] {
        let host = attn_decode_q8_split_plan(keys, cap, 64)
            .unwrap_or_else(|e| panic!("host mirror at cap {cap}: {e:?}"));
        let device = attn_decode_q8_split_plan_device(keys, cap, 64)
            .unwrap_or_else(|e| panic!("device probe at cap {cap}: {e:?}"));
        assert_eq!(host, expect, "host mirror at cap {cap}, keys {keys}");
        assert_eq!(
            device, expect,
            "device split_plan at cap {cap}, keys {keys}"
        );
    }
}
