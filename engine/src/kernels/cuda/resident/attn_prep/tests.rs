//! Seam-1 device tests (3090) of the attention preparation wrapper: the Rust wrapper against the C
//! entry against the patched vendored kernel, compared with a float64 host reference.
//!
//! The reference is written from the contract (RMSNorm times weight, then NEOX rope over the first
//! `n_rot` dimensions at position `*pos + t`, computed in f64 from the exact f32 inputs), not from
//! the kernel. The tolerance is the class RP1 (lane M REPLY-53, REPLY-56, REPLY-58). Every RP1
//! statement carries its limit label: f32 RoPE angle; not checked against an f64 reference above
//! position 1,150; A5 covers positions up to about 1,150 only.
//!
//! The device suite runs with `--test-threads=1` (decision 0040). Each refusal is checked by the
//! text of its error, so a test fails when another check refuses the call instead of the one
//! under test.
//!
//! Argued, not tested: (1) a freed buffer: a `DeviceRange` borrows its buffer, so the compiler
//! forbids freeing it early. (2) T10 (graph capture) is deferred to RD-11a: the capture window
//! (RD-2d) is absent. (3) T11 (a latched context): the fault child of `fault.rs` is lane C's.
//! (4) The size and offsets of the raw argument struct are pinned by a `const` assert in the
//! native child (the struct is private there). (5) The chain "q NaN -> decode -> non-finite
//! logits" is first tested at RD-9 integration.

use super::{
    AttnPrepCall, AttnPrepShape, HEAD_DIM, MAX_N_HEAD, MAX_N_KV_HEAD, MAX_ROWS, MAX_TOKENS,
    RopeParams, SlotMode, cache_bytes, check_rope, check_shape, proj_bytes, q_bytes, row_width,
};
use crate::EngError;
use crate::kernels::cuda::resident::step_params::{
    OFFSET_POS, OFFSET_STATUS, POS_MAX, STATUS_NON_FINITE_LOGITS, STEP_PARAMS_BYTES, StepParams,
    StepStatus,
};
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceStream};

/// RP1, lane M REPLY-53, provisional until the first GPU step; f32 RoPE angle; not checked against
/// an f64 reference above position 1,150; A5 covers positions up to about 1,150 only.
const RP1_ANGLE: f64 = 1.0 / 4_194_304.0;
/// Rows of canary on each side of the cache range inside its buffer.
const CANARY_ROWS: usize = 64;
/// The byte that fills every cache cell no launch may write.
const SENT: u8 = 0x5a;
/// The byte that fills `q` before a launch: not a NaN, so a NaN write is visible.
const Q_FILL: u8 = 0x55;
const EPS: f32 = 1e-6;
const BASE: f32 = 1e7;
const D: usize = HEAD_DIM;

/// The model-scale group has input scale 1.7, the large-amplitude group 20.
#[derive(Clone, Copy, PartialEq)]
enum Group {
    Model,
    Large,
}

fn qwen(m: usize, rows: usize) -> AttnPrepShape {
    shape(16, 2, 64, m, rows)
}

#[rustfmt::skip]
fn shape(n_head: usize, n_kv_head: usize, n_rot: usize, m: usize, rows: usize) -> AttnPrepShape {
    AttnPrepShape { m, n_head, n_kv_head, head_dim: D, n_rot, has_gate: true, slot: SlotMode::Position, rows, ld: row_width(n_head, n_kv_head) }
}

// ---- host models -------------------------------------------------------------------------

/// The f64 reference of one head row: RMSNorm times weight, then NEOX rope at position `p`.
fn ref_head(x: &[f32], w: &[f32], p: usize, n_rot: usize) -> Vec<f64> {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let rstd = 1.0 / (ss / D as f64 + f64::from(EPS)).sqrt();
    let y: Vec<f64> = x
        .iter()
        .zip(w)
        .map(|(&v, &w)| f64::from(v) * rstd * f64::from(w))
        .collect();
    let mut out = y.clone();
    let half = n_rot / 2;
    for i in 0..half {
        let theta = p as f64 * f64::from(BASE).powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = theta.sin_cos();
        out[i] = y[i] * c - y[i + half] * s;
        out[i + half] = y[i] * s + y[i + half] * c;
    }
    out
}

/// The RP1 bound of element `d` at position `p`.
fn rp1(reference: &[f64], d: usize, p: usize, n_rot: usize) -> f64 {
    let mut bound = 2f64.powi(-20) * reference[d].abs().max(1.0);
    if d < n_rot {
        let half = n_rot / 2;
        let i = d % half;
        bound += RP1_ANGLE * (p as f64 + 1.0) * reference[i].hypot(reference[i + half]);
    }
    bound
}

/// Binary16 bits of an f64 value, round to nearest even without an f32 step (no double rounding).
/// Overflow gives the signed infinity, subnormals are kept.
fn f64_to_f16_bits(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let sign: u16 = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = v.abs();
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 2f64.powi(-14) {
        return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
    }
    let mut e = i32::try_from((a.to_bits() >> 52) & 0x7ff).expect("exponent") - 1023;
    let mut frac = ((a / 2f64.powi(e) - 1.0) * 1024.0).round_ties_even();
    if frac >= 1024.0 {
        frac = 0.0;
        e += 1;
    }
    if e + 15 >= 31 {
        return sign | 0x7c00;
    }
    sign | (((e + 15) as u16) << 10) | frac as u16
}

fn f16_value(bits: u16) -> f64 {
    f64::from(f16_to_f32(bits))
}

/// The unit in the last place of a binary16 value.
fn ulp16(x: f64) -> f64 {
    let x = x.abs();
    if x < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(i32::try_from((x.to_bits() >> 52) & 0x7ff).expect("exponent") - 1023 - 10)
    }
}

// ---- report ------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Worst {
    ratio: f64,
    at: (usize, usize, usize, usize),
}

const BANDS: [&str; 5] = ["0-3", "4-1023", "1024-8191", "8192-32767", "32768-131071"];

fn band_of(p: usize) -> usize {
    match p {
        0..=3 => 0,
        4..=1023 => 1,
        1024..=8191 => 2,
        8192..=32767 => 3,
        _ => 4,
    }
}

/// The REPLY-53 report: the worst ratio per position band and group, the q-only ratios of
/// REPLY-58 and the K-tail numbers of REPLY-56.
#[derive(Default)]
struct Report {
    all: [[Worst; 5]; 2],
    q: [[Worst; 5]; 2],
    ktail_fail: usize,
    ktail_ulp: f64,
}

impl Report {
    #[rustfmt::skip]
    fn note(&mut self, group: Group, is_q: bool, ratio: f64, p: usize, at: (usize, usize, usize)) {
        let (g, b) = (usize::from(group == Group::Large), band_of(p));
        let entry = Worst { ratio, at: (p, at.0, at.1, at.2) };
        if is_q && ratio > self.q[g][b].ratio { self.q[g][b] = entry; }
        if ratio > self.all[g][b].ratio { self.all[g][b] = entry; }
    }

    #[rustfmt::skip]
    fn print(&self, test: &str) {
        println!("device: RTX 3090 (see the hold log), nvcc of the build image, -fmad default [{test}]");
        for (name, table) in [("BAND", &self.all), ("QBAND", &self.q)] {
            for (g, group) in ["model", "large"].iter().enumerate() {
                for (b, band) in BANDS.iter().enumerate() {
                    let w = table[g][b];
                    println!("{name} {group} {band} worst ratio {:.4} at pos {} token {} head {} dim {}", w.ratio, w.at.0, w.at.1, w.at.2, w.at.3);
                }
            }
        }
        println!("KTAIL interval failures {} (must be 0); max |v - RN16(ref)| {:.4} fp16 ulp", self.ktail_fail, self.ktail_ulp);
    }

    fn worst(&self) -> f64 {
        self.all.iter().flatten().fold(0.0, |m, w| m.max(w.ratio))
    }
}

// ---- fixture -----------------------------------------------------------------------------

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

/// Host inputs of one call.
struct Data {
    proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

#[rustfmt::skip]
fn data(s: &AttnPrepShape, seed: u64, scale: f64) -> Data {
    let mut rng = Rng::new(seed);
    let proj = (0..s.m * s.ld).map(|_| (rng.normal() * scale) as f32).collect();
    let mut weight = || -> Vec<f32> { (0..D).map(|_| (1.0 + 0.3 * rng.normal()) as f32).collect() };
    let (q_norm, k_norm) = (weight(), weight());
    Data { proj, q_norm, k_norm }
}

/// One range of a call: buffer index, byte offset, bytes.
type Slot = (usize, usize, usize);
/// Slot order: proj, q_norm, k_norm, q, k_cache, v_cache, pos, status.
const NAMES: [&str; 8] = [
    "proj", "q_norm", "k_norm", "q", "k_cache", "v_cache", "pos", "status",
];
const PROJ: usize = 0;
const QN: usize = 1;
const KN: usize = 2;
const Q: usize = 3;
const KC: usize = 4;
const VC: usize = 5;
const POS: usize = 6;
const STATUS: usize = 7;
/// Buffer indexes: 0 proj, 1 q_norm, 2 k_norm, 3 q, 4 k cache, 5 v cache, 6 the StepParams block,
/// 7 and 8 spare word buffers.
const Q_BUF: usize = 3;
const KC_BUF: usize = 4;
const PARAMS_BUF: usize = 6;

/// What one fixture needs: geometry, data, the position word and the status before the launch.
struct Cfg {
    shape: AttnPrepShape,
    scale: f64,
    seed: u64,
    word: i32,
    status: i32,
    q_tail: usize,
    slack: usize,
}

impl Cfg {
    #[rustfmt::skip]
    fn new(shape: AttnPrepShape, word: i32) -> Self {
        let seed = 0xa77e_0000 ^ (word as u64) ^ ((shape.m as u64) << 32);
        Self { shape, scale: 1.7, seed, word, status: 0, q_tail: 0, slack: 0 }
    }
}

struct Fx {
    rig: Rig,
    bufs: Vec<DeviceBuffer>,
    slots: [Slot; 8],
    base: [Slot; 8],
    shape: AttnPrepShape,
    rope: RopeParams,
    admit: usize,
    data: Data,
}

fn le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn padded(mut bytes: Vec<u8>, slack: usize, fill: u8) -> Vec<u8> {
    bytes.resize(bytes.len() + slack, fill);
    bytes
}

#[rustfmt::skip]
fn params_block(word: i32, status: i32) -> [u8; STEP_PARAMS_BYTES] {
    StepParams { token_id: 0, pos: word, kv_len: word.wrapping_add(1), kv_slot: word, status, pad: [0; 3] }.to_bytes()
}

impl Fx {
    #[rustfmt::skip]
    fn new(cfg: &Cfg) -> Self {
        let rig = Rig::new();
        let s = cfg.shape;
        let data = data(&s, cfg.seed, cfg.scale);
        let ctx = &rig.ctx;
        let row_bytes = s.n_kv_head * D * 2;
        let cache = cache_bytes(s.rows, s.n_kv_head);
        let cache_buf = vec![SENT; cache + 2 * CANARY_ROWS * row_bytes + cfg.slack];
        let q_len = q_bytes(s.m, s.n_head);
        let words = vec![0u8; 64 + cfg.slack];
        let params = padded(params_block(cfg.word, cfg.status).to_vec(), 32 + cfg.slack, 0);
        let bufs = vec![
            testkit::upload_bytes(ctx, &padded(le(&data.proj), cfg.slack, 0)),
            testkit::upload_bytes(ctx, &padded(le(&data.q_norm), cfg.slack, 0)),
            testkit::upload_bytes(ctx, &padded(le(&data.k_norm), cfg.slack, 0)),
            testkit::upload_filled(ctx, Q_FILL, q_len + cfg.q_tail * 4 + cfg.slack),
            testkit::upload_bytes(ctx, &cache_buf),
            testkit::upload_bytes(ctx, &cache_buf),
            testkit::upload_bytes(ctx, &params),
            testkit::upload_bytes(ctx, &words),
            testkit::upload_bytes(ctx, &words),
        ];
        let off = CANARY_ROWS * row_bytes;
        let proj = proj_bytes(s.m, s.ld, row_width(s.n_head, s.n_kv_head));
        let slots = [(0, 0, proj), (1, 0, D * 4), (2, 0, D * 4), (3, 0, q_len), (4, off, cache), (5, off, cache), (PARAMS_BUF, OFFSET_POS, 4), (PARAMS_BUF, OFFSET_STATUS, 4)];
        let rope = RopeParams { eps: EPS, rope_base: BASE };
        Self { rig, bufs, slots, base: slots, shape: s, rope, admit: usize::try_from(cfg.word).unwrap_or(0), data }
    }

    /// Queues the call on `stream`; `foreign` replaces the range of one slot by a range of a buffer
    /// of another context.
    fn attempt(
        &self,
        stream: &DeviceStream,
        foreign: Option<(usize, &DeviceBuffer)>,
    ) -> Result<(), EngError> {
        let r = |i: usize| {
            let (index, offset, bytes) = self.slots[i];
            match foreign {
                Some((slot, other)) if slot == i => other.range(offset, bytes),
                _ => self.bufs[index].range(offset, bytes),
            }
        };
        let call = AttnPrepCall {
            shape: self.shape,
            rope: self.rope,
            admit_pos: self.admit,
            proj: r(PROJ)?,
            q_norm: r(QN)?,
            k_norm: r(KN)?,
            k_cache: r(KC)?,
            v_cache: r(VC)?,
            q: r(Q)?,
            pos: r(POS)?,
            status: r(STATUS)?,
        };
        self.rig.ctx.attn_prep(stream, call)
    }

    /// Queues the call and drains the stream when it was accepted.
    fn run(&self) -> Result<(), EngError> {
        self.attempt(&self.rig.stream, None)?;
        self.rig.stream.synchronize().expect("drain");
        Ok(())
    }

    fn bytes(&self, buffer: usize) -> Vec<u8> {
        let b = &self.bufs[buffer];
        testkit::read_bytes(&self.rig.ctx, b, b.bytes())
    }

    #[rustfmt::skip]
    fn status_word(&self) -> i32 {
        let block = self.bytes(PARAMS_BUF);
        i32::from_le_bytes(block[OFFSET_STATUS..OFFSET_STATUS + 4].try_into().expect("word"))
    }

    fn cache_range(&self) -> std::ops::Range<usize> {
        let s = &self.shape;
        let start = CANARY_ROWS * s.n_kv_head * D * 2;
        start..start + cache_bytes(s.rows, s.n_kv_head)
    }

    /// Every byte of both cache buffers, except the append rows `[first, first + m)`, is the
    /// sentinel; with `first` None, no byte was written at all.
    fn caches_untouched(&self, written: Option<(usize, usize)>) -> bool {
        let row = self.shape.n_kv_head * D * 2;
        let skip = written.map(|(first, m)| {
            let start = (CANARY_ROWS + first) * row;
            start..start + m * row
        });
        [KC, VC].iter().all(|&c| {
            let bytes = self.bytes(self.slots[c].0);
            let keep = |range: std::ops::Range<usize>| bytes[range].iter().all(|&b| b == SENT);
            match &skip {
                None => keep(0..bytes.len()),
                Some(w) => keep(0..w.start) && keep(w.end..bytes.len()),
            }
        })
    }

    fn q_values(&self) -> Vec<f32> {
        let bytes = self.bytes(Q);
        let n = q_bytes(self.shape.m, self.shape.n_head);
        testkit::floats(&bytes[..n])
    }
}

/// How strictly `verify` holds each element.
#[derive(Clone, Copy, PartialEq)]
enum Bound {
    Rp1,
    /// `2^-19 * max(|ref|, 1)` for every element: isolates the library `sincosf` (n_rot = 2 rows).
    Tight,
}

#[rustfmt::skip]
fn bound_of(kind: Bound, reference: &[f64], d: usize, p: usize, n_rot: usize) -> f64 {
    match kind {
        Bound::Rp1 => rp1(reference, d, p, n_rot),
        Bound::Tight => 2f64.powi(-19) * reference[d].abs().max(1.0),
    }
}

/// How a valid launch is judged: the report group, the bound and the status word it must leave.
#[derive(Clone, Copy)]
struct Mode {
    group: Group,
    kind: Bound,
    status: i32,
}

/// Checks a valid launch whose token 0 is at position `pos0`: q by the bound, K by the interval
/// rule for every dimension, V bit equal, the rows outside the append and the canaries intact,
/// and the status word as the mode says.
#[rustfmt::skip]
fn verify(fx: &Fx, pos0: usize, mode: Mode, rep: &mut Report) {
    let Mode { group, kind, status } = mode;
    let s = fx.shape;
    let (q, kc, vc) = (fx.q_values(), fx.bytes(KC), fx.bytes(VC));
    let base = fx.cache_range().start;
    let kv_base = s.n_head * 2 * D;
    let half = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
    for t in 0..s.m {
        let p = pos0 + t;
        let row = &fx.data.proj[t * s.ld..];
        for h in 0..s.n_head {
            let reference = ref_head(&row[h * 2 * D..][..D], &fx.data.q_norm, p, s.n_rot);
            for d in 0..D {
                let b = bound_of(kind, &reference, d, p, s.n_rot);
                let ratio = (f64::from(q[(t * s.n_head + h) * D + d]) - reference[d]).abs() / b;
                assert!(ratio <= 1.0, "q pos {p} token {t} head {h} dim {d}: ratio {ratio}");
                rep.note(group, true, ratio, p, (t, h, d));
            }
        }
        for j in 0..s.n_kv_head {
            let reference = ref_head(&row[kv_base + j * D..][..D], &fx.data.k_norm, p, s.n_rot);
            let at = base + (p * s.n_kv_head + j) * D * 2;
            for d in 0..D {
                let b = bound_of(kind, &reference, d, p, s.n_rot);
                let (lo, hi) = (f16_value(f64_to_f16_bits(reference[d] - b)), f16_value(f64_to_f16_bits(reference[d] + b)));
                let v = f16_value(half(&kc, at + 2 * d));
                let inside = v >= lo && v <= hi;
                assert!(inside || d >= s.n_rot, "K pos {p} token {t} head {j} dim {d}: {v} not in [{lo}, {hi}]");
                if d >= s.n_rot {
                    let rr = f16_value(f64_to_f16_bits(reference[d]));
                    rep.ktail_ulp = rep.ktail_ulp.max((v - rr).abs() / ulp16(rr));
                    rep.ktail_fail += usize::from(!inside);
                }
                rep.note(group, false, (v - reference[d]).abs() / (b + 2f64.powi(-11) * reference[d].abs()), p, (t, j, d));
                let want = f16_bits(row[kv_base + (s.n_kv_head + j) * D + d]);
                assert_eq!(half(&vc, at + 2 * d), want, "V pos {p} token {t} head {j} dim {d}");
            }
        }
    }
    assert_eq!(rep.ktail_fail, 0, "K tail interval failures");
    assert_eq!(fx.status_word(), status, "a valid launch leaves the status alone");
    assert!(fx.caches_untouched(Some((pos0, s.m))), "rows outside the append changed");
    assert!(fx.bytes(Q)[q_bytes(s.m, s.n_head)..].iter().all(|&b| b == Q_FILL), "the bytes after q changed");
}

/// Runs a valid case on a fresh fixture and verifies it.
#[rustfmt::skip]
fn case(cfg: &Cfg, group: Group, kind: Bound, rep: &mut Report) -> Fx {
    let fx = Fx::new(cfg);
    fx.run().expect("the launch is accepted");
    let pos0 = usize::try_from(cfg.word).expect("a valid word");
    verify(&fx, pos0, Mode { group, kind, status: cfg.status }, rep);
    fx
}

// ---- T1, T2 ------------------------------------------------------------------------------

#[test]
fn attn_prep_matches_the_float64_reference_on_the_qwen_shape() {
    if !testkit::device_present("attn_prep_matches_the_float64_reference_on_the_qwen_shape") {
        return;
    }
    let mut rep = Report::default();
    let cases = [
        (0, 1, 4),
        (1, 1, 4),
        (63, 1, 64),
        (1023, 1, 1100),
        (3, 16, 19),
        (100, 17, 2000),
        (2047, 257, 2400),
        (8191, 4, 8195),
        (65535, 1, 65536),
        (65536, 2, 65538),
        (131_071, 1, 131_072),
        (131_056, 16, 131_072),
    ];
    for (pos, m, rows) in cases {
        let mut cfg = Cfg::new(qwen(m, rows), pos);
        cfg.q_tail = 1024;
        case(&cfg, Group::Model, Bound::Rp1, &mut rep);
    }
    for pos in [1023, 8191, 32767, 65535, 131_071] {
        let mut cfg = Cfg::new(qwen(1, usize::try_from(pos).expect("pos") + 1), pos);
        cfg.scale = 20.0;
        case(&cfg, Group::Large, Bound::Rp1, &mut rep);
    }
    rep.print("T1");
    assert!(rep.worst() <= 1.0);
}

#[test]
fn attn_prep_handles_other_shapes() {
    if !testkit::device_present("attn_prep_handles_other_shapes") {
        return;
    }
    let mut rep = Report::default();
    for n_rot in [2, 4, 32, 64, 128, 256] {
        let cfg = Cfg::new(shape(4, 1, n_rot, 3, 70_004), 70_001);
        case(&cfg, Group::Model, Bound::Rp1, &mut rep);
    }
    let mut mha = Cfg::new(shape(3, 3, 64, 5, 40), 12);
    mha.scale = 3.0;
    case(&mha, Group::Model, Bound::Rp1, &mut rep);
    let mut small = Cfg::new(shape(32, 4, 64, 2, 9), 5);
    small.scale = 0.3;
    case(&small, Group::Model, Bound::Rp1, &mut rep);
    case(
        &Cfg::new(shape(1, 1, 64, 1, 1), 0),
        Group::Model,
        Bound::Rp1,
        &mut rep,
    );
    let mut wide = qwen(3, 110);
    wide.ld += 64;
    case(&Cfg::new(wide, 100), Group::Model, Bound::Rp1, &mut rep);
    for (pos, m, rows) in [
        (70_001, 1, 70_002),
        (131_071, 1, 131_072),
        (100_000, 2, 100_002),
    ] {
        let cfg = Cfg::new(shape(4, 1, 2, m, rows), pos);
        case(&cfg, Group::Model, Bound::Tight, &mut rep);
    }
    rep.print("T2");
}

// ---- T3 to T6 ----------------------------------------------------------------------------

#[rustfmt::skip]
fn nan_launch(fx: &Fx) {
    assert!(fx.q_values().iter().all(|v| v.is_nan()), "q is NaN on a refused launch");
    assert!(fx.caches_untouched(None), "a refused launch writes no cache byte");
}

#[test]
#[rustfmt::skip]
fn the_position_word_drives_the_rope_angle_and_the_row() {
    if !testkit::device_present("the_position_word_drives_the_rope_angle_and_the_row") {
        return;
    }
    let mut rep = Report::default();
    let mut outputs = Vec::new();
    for word in [5, 100, 7] {
        let mut cfg = Cfg::new(qwen(2, 128), word);
        cfg.seed = 99;
        outputs.push(case(&cfg, Group::Model, Bound::Rp1, &mut rep).q_values());
    }
    assert!(outputs[0] != outputs[1] && outputs[1] != outputs[2] && outputs[0] != outputs[2]);
    let mut fx = Fx::new(&Cfg::new(qwen(2, 200), 100));
    fx.admit = 5;
    fx.run().expect("the word is authoritative");
    verify(&fx, 100, Mode { group: Group::Model, kind: Bound::Rp1, status: 0 }, &mut rep);
    let mut cfg = Cfg::new(qwen(1, 100), 4096);
    cfg.status = 2;
    let mut fx = Fx::new(&cfg);
    fx.admit = 5;
    fx.run().expect("a disagreeing word is refused on the device");
    nan_launch(&fx);
    assert!(StepStatus(fx.status_word()).invalid_step_state());
}

#[test]
#[rustfmt::skip]
fn a_poisoned_position_word_writes_nothing_and_reports_mask_4() {
    if !testkit::device_present("a_poisoned_position_word_writes_nothing_and_reports_mask_4") {
        return;
    }
    let (m, rows) = (16_usize, 256_i32);
    for word in [-1, i32::MIN, rows, rows + 1, rows - 15, i32::MAX, i32::MAX - 7, 1 << 30, -2, 4096] {
        let mut cfg = Cfg::new(qwen(m, rows as usize), word);
        cfg.status = STATUS_NON_FINITE_LOGITS;
        let mut fx = Fx::new(&cfg);
        fx.admit = if word == 4096 { 5 } else { 0 };
        let before = fx.bytes(PARAMS_BUF);
        fx.run().expect("a poisoned word is not an error of the call");
        nan_launch(&fx);
        let status = fx.status_word();
        assert!(StepStatus(status).invalid_step_state(), "word {word}: mask 4");
        assert_eq!(status & !4, STATUS_NON_FINITE_LOGITS, "word {word}: earlier bits survive");
        assert_eq!(before[..OFFSET_STATUS], fx.bytes(PARAMS_BUF)[..OFFSET_STATUS], "word {word}: other words");
    }
    let mut rep = Report::default();
    for (word, m, rows) in [(255, 1, 256), (240, 16, 256), (0, 16, 256)] {
        let mut cfg = Cfg::new(qwen(m, rows), word);
        cfg.status = STATUS_NON_FINITE_LOGITS;
        case(&cfg, Group::Model, Bound::Rp1, &mut rep);
    }
}

#[test]
#[rustfmt::skip]
fn a_saturated_step_advance_feeds_attn_prep_a_poisoned_word() {
    if !testkit::device_present("a_saturated_step_advance_feeds_attn_prep_a_poisoned_word") {
        return;
    }
    let mut fx = Fx::new(&Cfg::new(qwen(1, 100), 0));
    let block = StepParams::new(0, POS_MAX).expect("the largest position");
    fx.bufs[PARAMS_BUF].write_at(&fx.rig.stream, 0, &block.to_bytes()).expect("write the saturated block");
    let params = fx.bufs[PARAMS_BUF].range(0, STEP_PARAMS_BYTES).expect("params");
    fx.rig.ctx.step_advance(&fx.rig.stream, params, 1).expect("advance");
    fx.rig.stream.synchronize().expect("drain");
    let read = StepParams::from_bytes(fx.bytes(PARAMS_BUF)[..STEP_PARAMS_BYTES].try_into().expect("block"));
    assert_eq!(read.pos, -1, "the saturated advance poisons the word");
    fx.admit = 0;
    fx.run().expect("the wrapper admits 0 + 1 <= 100");
    nan_launch(&fx);
    assert!(StepStatus(fx.status_word()).invalid_step_state());
}

#[test]
fn attn_prep_is_deterministic() {
    if !testkit::device_present("attn_prep_is_deterministic") {
        return;
    }
    let fx = Fx::new(&Cfg::new(qwen(4, 64), 20));
    fx.run().expect("first");
    let first = (fx.bytes(Q), fx.bytes(KC), fx.bytes(VC));
    fx.run().expect("second");
    assert!(first == (fx.bytes(Q), fx.bytes(KC), fx.bytes(VC)));
}

// ---- T7: refusals ------------------------------------------------------------------------

type ShapeEdit = fn(&mut AttnPrepShape);
type Edit = Box<dyn Fn(&mut Fx)>;
type Row = (String, String, Edit);

fn row(label: &str, text: &str, edit: impl Fn(&mut Fx) + 'static) -> Row {
    (label.to_owned(), text.to_owned(), Box::new(edit))
}

/// Shape and scalar rows: the tag of the rule that must fire.
#[rustfmt::skip]
fn shape_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    let mut add = |label: &str, tag: &str, f: fn(&mut Fx, usize), v: usize| {
        rows.push(row(&format!("{label} {v}"), &format!("attn_prep: {tag}:"), move |fx| f(fx, v)));
    };
    for v in [128, 0, 512] { add("head_dim", "head_dim", |fx, v| fx.shape.head_dim = v, v); }
    for v in [0, 257, usize::MAX] { add("n_head", "n_head", |fx, v| fx.shape.n_head = v, v); }
    for v in [0, 65] { add("n_kv_head", "n_kv_head", |fx, v| fx.shape.n_kv_head = v, v); }
    for v in [0, 65_536, usize::MAX] { add("m", "m", |fx, v| fx.shape.m = v, v); }
    for v in [0, 1, 63, 96, 126, 258, 512, 1 << 20, usize::MAX] { add("n_rot", "n_rot", |fx, v| fx.shape.n_rot = v, v); }
    for v in [1, 1 << 30, 0, usize::MAX] { add("rows", "rows", |fx, v| fx.shape.rows = v, v); }
    for v in [4607, 0, usize::MAX] { add("ld", "ld", |fx, v| fx.shape.ld = v, v); }
    for v in [0.0, f32::NAN, f32::INFINITY, -1.0] {
        rows.push(row(&format!("eps {v}"), "attn_prep: eps:", move |fx| fx.rope.eps = v));
    }
    for v in [1.0, 0.0, f32::NAN, f32::INFINITY] {
        rows.push(row(&format!("rope_base {v}"), "attn_prep: rope_base:", move |fx| fx.rope.rope_base = v));
    }
    rows.push(row("has_gate false", "attn_prep: gate:", |fx| fx.shape.has_gate = false));
    for r in [0, 1, 64, u32::MAX] {
        rows.push(row(&format!("Ring({r})"), "attn_prep: slot_ring:", move |fx| fx.shape.slot = SlotMode::Ring(r)));
    }
    rows.push(row("admit_pos + m > rows", "the step needs positions", |fx| fx.admit = fx.shape.rows - fx.shape.m + 1));
    rows.push(row("admit_pos far above rows", "the step needs positions", |fx| fx.admit = POS_MAX as usize + 1));
    rows
}

/// Size and alignment rows: the text of `require_fit`.
#[rustfmt::skip]
fn fit_rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for (slot, less) in [(PROJ, 4), (Q, 4), (KC, 2), (VC, 2), (QN, 4), (KN, 4), (POS, 2), (STATUS, 2)] {
        let text = format!("CUDA {} range holds", NAMES[slot]);
        rows.push(row(&format!("{} short", NAMES[slot]), &text, move |fx| fx.slots[slot].2 -= less));
    }
    for (slot, by) in [(PROJ, 2), (QN, 2), (KN, 2), (POS, 2), (STATUS, 2), (Q, 4), (Q, 8), (KC, 4), (KC, 8), (VC, 4), (VC, 8)] {
        let text = format!("CUDA {} range starts at offset", NAMES[slot]);
        rows.push(row(&format!("{} at +{by}", NAMES[slot]), &text, move |fx| fx.slots[slot].1 += by));
    }
    rows
}

/// The 22 pairs `check_apart` compares: the written slots (q, k_cache, v_cache, status) pairwise,
/// then each against the read slots (proj, q_norm, k_norm, pos). The later slot of a pair is moved
/// onto the first one, in separate word buffers for `pos` and `status`.
fn overlap_rows() -> Vec<Row> {
    let written = [Q, KC, VC, STATUS];
    let read = [PROJ, QN, KN, POS];
    let mut rows = Vec::new();
    for (i, &w) in written.iter().enumerate() {
        for &o in written[i + 1..].iter().chain(&read) {
            let text = format!("{} and {} ranges overlap", NAMES[w], NAMES[o]);
            rows.push(row(
                &format!("{} on {}", NAMES[o], NAMES[w]),
                &text,
                move |fx| {
                    fx.slots[POS] = (7, 0, 4);
                    fx.slots[STATUS] = (8, 0, 4);
                    let target = fx.slots[w];
                    let aligned = if matches!(o, Q | KC | VC) {
                        target.1 & !15
                    } else {
                        target.1
                    };
                    fx.slots[o] = (target.0, aligned, fx.base[o].2);
                },
            ));
        }
    }
    rows
}

#[test]
fn attn_prep_rejects_bad_arguments_before_any_launch() {
    if !testkit::device_present("attn_prep_rejects_bad_arguments_before_any_launch") {
        return;
    }
    let mut cfg = Cfg::new(shape(8, 1, 64, 2, 16), 0);
    cfg.slack = 1 << 20;
    let mut fx = Fx::new(&cfg);
    let other = Rig::new();
    let foreign = testkit::upload_filled(&other.ctx, 0, 1 << 20);
    fx.run().expect("the base call is valid");
    let before = (fx.bytes(PARAMS_BUF), fx.bytes(KC_BUF));
    let mut rows = shape_rows();
    rows.extend(fit_rows());
    rows.extend(overlap_rows());
    assert_eq!(rows.len(), 83);
    let saved = (fx.slots, fx.shape, fx.rope, fx.admit);
    for (label, text, edit) in &rows {
        edit(&mut fx);
        let result = fx.run();
        assert!(
            matches!(&result, Err(EngError::BadRequest(m)) if m.contains(text.as_str())),
            "{label}: {result:?}"
        );
        (fx.slots, fx.shape, fx.rope, fx.admit) = saved;
    }
    for slot in 0..8 {
        let result = fx.attempt(&fx.rig.stream, Some((slot, &foreign)));
        assert!(
            matches!(&result, Err(EngError::BadRequest(m)) if m.contains("buffer belongs")),
            "{slot}: {result:?}"
        );
    }
    let result = fx.attempt(&other.stream, None);
    assert!(
        matches!(&result, Err(EngError::BadRequest(m)) if m.contains("stream belongs")),
        "{result:?}"
    );
    let sizes: Vec<usize> = fx.bufs.iter().map(DeviceBuffer::bytes).collect();
    for (buffer, bytes) in sizes.iter().enumerate() {
        assert!(fx.bufs[buffer].range(0, 0).is_err() && fx.bufs[buffer].range(1, *bytes).is_err());
    }
    // Words and caches did not change; the backend is usable afterwards.
    assert!(before == (fx.bytes(PARAMS_BUF), fx.bytes(KC_BUF)));
    fx.run().expect("a good call after the refusals");
    controls(&mut fx);
}

/// Calls that look wrong but are valid: the touched extents are disjoint, the whole ranges are not.
fn controls(fx: &mut Fx) {
    let saved = (fx.slots, fx.shape, fx.rope, fx.admit);
    fx.slots[QN] = fx.slots[KN];
    fx.run().expect("the norm weights may alias");
    fx.slots = saved.0;
    let need = fx.slots[Q].2;
    fx.slots[Q].2 += 64;
    fx.slots[POS] = (Q_BUF, need, 4);
    fx.slots[STATUS] = (Q_BUF, need + 16, 4);
    fx.run().expect("pos and status in the tail of q");
    fx.slots = saved.0;
    fx.slots[KC].2 += 64;
    fx.slots[POS] = (KC_BUF, fx.slots[KC].1 + fx.slots[VC].2, 4);
    fx.run().expect("pos in the tail of a larger k_cache");
    fx.slots = saved.0;
    for n_rot in [2, 256] {
        fx.shape.n_rot = n_rot;
        fx.run().expect("n_rot at its edges");
    }
    fx.shape = saved.1;
}

#[test]
fn attn_prep_refuses_the_configurations_it_cannot_run() {
    if !testkit::device_present("attn_prep_refuses_the_configurations_it_cannot_run") {
        return;
    }
    let fx_for = |edit: ShapeEdit| {
        let mut fx = Fx::new(&Cfg::new(qwen(1, 8), 0));
        edit(&mut fx.shape);
        fx
    };
    let edits: [(ShapeEdit, &str); 3] = [
        (|s| s.head_dim = 128, "attn_prep: head_dim:"),
        (|s| s.has_gate = false, "attn_prep: gate:"),
        (|s| s.slot = SlotMode::Ring(1024), "attn_prep: slot_ring:"),
    ];
    for (edit, tag) in edits {
        let fx = fx_for(edit);
        let result = fx.run();
        assert!(
            matches!(&result, Err(EngError::BadRequest(m)) if m.contains(tag) && m.contains("MEL-K1")),
            "{tag}: {result:?}"
        );
        assert!(fx.caches_untouched(None));
    }
    fx_for(|_| {}).run().expect("a Qwen call succeeds");
}

// ---- T9: pure ----------------------------------------------------------------------------

#[test]
fn attn_prep_bounds_are_pinned() {
    let ok = |f: fn(&mut AttnPrepShape)| {
        let mut s = qwen(1, 1);
        f(&mut s);
        s.ld = s.ld.max(row_width(s.n_head, s.n_kv_head));
        check_shape(&s).is_ok()
    };
    assert!(ok(|_| {}) && ok(|s| s.n_rot = 2) && ok(|s| s.n_rot = 256));
    assert!(!ok(|s| s.n_rot = 96) && !ok(|s| s.n_rot = 126) && !ok(|s| s.n_rot = 258));
    assert!(!ok(|s| s.n_rot = 512) && !ok(|s| s.n_rot = 1 << 20));
    assert!(ok(|s| s.n_head = MAX_N_HEAD) && !ok(|s| s.n_head = MAX_N_HEAD + 1));
    assert!(ok(|s| s.n_kv_head = MAX_N_KV_HEAD) && !ok(|s| s.n_kv_head = MAX_N_KV_HEAD + 1));
    assert!(ok(|s| s.rows = MAX_ROWS) && !ok(|s| s.rows = MAX_ROWS + 1) && !ok(|s| s.rows = 0));
    let mut short = qwen(1, 1);
    short.ld -= 1;
    assert!(check_shape(&short).is_err());
    let mut big = qwen(MAX_TOKENS, MAX_TOKENS);
    assert!(check_shape(&big).is_ok());
    big.m += 1;
    big.rows += 1;
    assert!(check_shape(&big).is_err());
    let rope = |eps, rope_base| check_rope(&RopeParams { eps, rope_base }).is_ok();
    assert!(rope(1e-6, 1e7) && !rope(0.0, 1e7) && !rope(f32::NAN, 1e7) && !rope(1e-6, 1.0));
    assert_eq!(row_width(16, 2), 9216);
    assert_eq!(proj_bytes(3, 10_000, 9216), 4 * (2 * 10_000 + 9216));
    assert_eq!(cache_bytes(131_072, 2), 134_217_728);
    assert_eq!(q_bytes(3, 16), 3 * 16 * 256 * 4);
    golden_rope();
    pinned_rounding();
}

/// The f64 formula against recorded Python values (the pairing and the signs are pinned only through T1): ten recorded (pos, i, cos, sin) rows and (0, i).
#[rustfmt::skip]
fn golden_rope() {
    let golden = [
        (1, 0, 0.540_302_305_868_139_8, 0.841_470_984_807_896_5), (1, 31, 0.999_999_999_999_986_3, 1.654_817_099_943_173_7e-7),
        (1022, 0, -0.555_038_079_315_548_3, -0.831_824_939_821_899_8), (1022, 1, -0.264_759_383_480_313, 0.964_314_507_232_933_9),
        (65_535, 0, 0.192_344_018_605_863_98, 0.981_327_559_231_140_2), (65_535, 1, 0.938_320_122_200_641, -0.345_767_766_388_618_3),
        (65_535, 31, 0.999_941_195_257_119_6, 0.010_844_631_287_559_251), (131_071, 0, -0.817_983_499_387_949_1, -0.575_241_683_754_789_3),
        (131_071, 1, 0.994_821_084_544_182_3, -0.101_641_574_891_070_06), (131_071, 31, 0.999_764_784_355_494_9, 0.021_688_152_586_395_913),
        (0, 0, 1.0, 0.0), (0, 1, 1.0, 0.0), (0, 7, 1.0, 0.0), (0, 15, 1.0, 0.0), (0, 31, 1.0, 0.0),
    ];
    for (p, i, c, s) in golden {
        let theta = f64::from(p) * 1e7_f64.powf(-2.0 * f64::from(i) / 64.0);
        assert!((theta.cos() - c).abs() < 1e-10 && (theta.sin() - s).abs() < 1e-10, "{p} {i}");
    }
}

/// `rp1` on and off the rotated dimensions, `rn16` ties and subnormals, and the local f64 rounding.
fn pinned_rounding() {
    let reference: Vec<f64> = (0..D)
        .map(|d| if d == 1 || d == 3 { 3.0 } else { 0.5 })
        .collect();
    let rotated = 2f64.powi(-20) * 3.0 + RP1_ANGLE * 11.0 * 3f64.hypot(3.0);
    assert!((rp1(&reference, 1, 10, 4) - rotated).abs() < 1e-18);
    assert_eq!(rp1(&reference, 100, 10, 4), 2f64.powi(-20));
    let tie = 1.0 + 2f32.powi(-11);
    assert_eq!(
        (f16_bits(tie), f16_bits(1.0 + 3.0 * 2f32.powi(-11))),
        (0x3c00, 0x3c02)
    );
    assert_eq!(
        (
            f16_bits(2f32.powi(-24)),
            f16_bits(2f32.powi(-25)),
            f16_bits(3.0 * 2f32.powi(-26))
        ),
        (1, 0, 1)
    );
    assert_eq!(
        f64_to_f16_bits(1.0 + 2f64.powi(-11) + 2f64.powi(-40)),
        0x3c01
    );
    assert_eq!(
        (f64_to_f16_bits(65_520.0), f64_to_f16_bits(-65_520.0)),
        (0x7c00, 0xfc00)
    );
    assert_eq!(
        (f64_to_f16_bits(2f64.powi(-25)), f64_to_f16_bits(-0.0)),
        (0, 0x8000)
    );
    assert!(f16_value(f64_to_f16_bits(f64::NAN)).is_nan());
}

// ---------------------------------------------------------------- T10 (RD-11a)

/// The capture plan of the owed test: the fixture owns every buffer, so no pin and no hold.
fn no_pins() -> CapturePlan<'static> {
    CapturePlan {
        pins: &[],
        holds: Vec::new(),
    }
}

/// Restores every mutable buffer of the fixture to its pristine image with the position word
/// set to `word` (the host writes are outside any capture window).
fn reset_to(fx: &Fx, word: i32) {
    let s = fx.shape;
    let row = s.n_kv_head * D * 2;
    let cache = vec![SENT; cache_bytes(s.rows, s.n_kv_head) + 2 * CANARY_ROWS * row];
    let q = vec![Q_FILL; q_bytes(s.m, s.n_head)];
    let block = params_block(word, 0).to_vec();
    let stream = &fx.rig.stream;
    fx.bufs[KC]
        .write_at(stream, 0, &cache)
        .expect("reset the k cache");
    fx.bufs[VC]
        .write_at(stream, 0, &cache)
        .expect("reset the v cache");
    fx.bufs[Q].write_at(stream, 0, &q).expect("reset q");
    fx.bufs[PARAMS_BUF]
        .write_at(stream, 0, &block)
        .expect("reset the step block");
}

/// The full bytes of one eager launch at `word` (fresh fixture, same data: the seed stays the
/// base configuration's).
fn eager_prep_bytes(cfg: &Cfg, word: i32) -> [Vec<u8>; 4] {
    let fx = Fx::new(&Cfg { word, ..*cfg });
    fx.run().expect("the eager launch");
    [
        fx.bytes(Q),
        fx.bytes(KC),
        fx.bytes(VC),
        fx.bytes(PARAMS_BUF),
    ]
}

/// T10 (RD-8a, owed): one `attn_prep` captured and replayed with the position word rewritten
/// between replays equals the eager launch at that position byte for byte (q, both caches, the
/// step block).
#[test]
fn attn_prep_inside_a_capture_equals_eager() {
    if !testkit::device_present("attn_prep_inside_a_capture_equals_eager") {
        return;
    }
    let base = Cfg::new(qwen(1, 4200), 5);
    let words = [5, 77, 4090];
    let eager: Vec<[Vec<u8>; 4]> = words.iter().map(|&w| eager_prep_bytes(&base, w)).collect();
    let fx = Fx::new(&Cfg {
        word: words[0],
        ..base
    });
    fx.run().expect("the warm launch");
    reset_to(&fx, words[0]);
    let (graph, ()) = fx
        .rig
        .ctx
        .capture(&fx.rig.stream, no_pins(), || {
            fx.attempt(&fx.rig.stream, None)
        })
        .expect("capture of attn_prep");
    assert_eq!(graph.nodes(), 1, "the attn_prep graph holds one kernel");
    for (case, &word) in words.iter().enumerate() {
        reset_to(&fx, word);
        graph.launch(&fx.rig.stream).expect("replay");
        for (at, slot, name) in [(0, Q, "q"), (1, KC, "the k cache"), (2, VC, "the v cache")] {
            assert_eq!(
                fx.bytes(fx.slots[slot].0),
                eager[case][at],
                "word {word}: {name} differs from the eager launch"
            );
        }
        assert_eq!(
            fx.bytes(PARAMS_BUF),
            eager[case][3],
            "word {word}: the step block differs"
        );
    }
}
