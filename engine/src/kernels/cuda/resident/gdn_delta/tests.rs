//! Seam-1 device tests (3090) of the gated DeltaNet wrapper (RD-7b): the Rust wrapper against the C shim against
//! `gated_delta_k`, compared with an f64 host reference written for the Qwen3.6 GGUF semantics; the RD-M6 llama.cpp
//! dumps are the oracle data of the six `delta_oracle_*` tests. Run with `--test-threads=1` (decision 0040).
//!
//! E4 is lane M's REPLY-55 (2^-12, E-form, provisional until the first GPU step; domain one DeltaNet layer, 128 to
//! 1,152 steps). Every group prints its UNROUNDED maximum with the head and the step; a failed comparison is reported,
//! never widened. Argued, not tested: T6 (graph capture) is deferred to RD-11a (no `GraphExec` at the base); a freed
//! buffer cannot be written as a row (a `DeviceRange` borrows its buffer); the kernel reads no position word
//! (`check_no_position.py`), so no poisoned `StepParams` test applies.
#![allow(
    non_snake_case,
    reason = "the oracle test names carry the layer and prompt exactly as the packet names them"
)]

use std::path::{Path, PathBuf};

use super::{DeltaBytes, DeltaCall, DeltaShape, DeltaWeights, GDN_EPS, GDN_HEAD_DIM, delta_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, fnv1a64, read_f32_file, silu_f64, softplus_f64};
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// E4, provisional (REPLY-55): `max |d| <= E4 * max(max |ref|, 1)` per step vector and for the end state; lane M
/// tightens it after the first GPU step.
const E4: f64 = 1.0 / 4096.0;
/// x <= X_STOP (2^-14) makes E4 final; above it the packet says STOP and report.
const X_STOP: f64 = 1.0 / 16384.0;
/// The reference proves itself against `attn_output` within 2^-18.
const SELF_CHECK: f64 = 1.0 / 262_144.0;
/// Steps from which a group counts toward x.
const DOMAIN_START: usize = 128;
const D: usize = GDN_HEAD_DIM;
/// Bytes of sentinel (256 floats) before and after every device buffer.
const PAD_BYTES: usize = 256 * 4;
const SENTINEL: [u8; 4] = 0x7fc0_deadu32.to_le_bytes();
const LIMIT: &str = "DeltaNet f32; E4 holds for rollouts of 128 to 1,152 steps only; not checked beyond 1,151 steps";

// ---- host data

/// Row lengths and column offsets of one layout.
#[derive(Clone, Copy)]
struct Layout {
    conv_ld: usize,
    ld: usize,
    z_off: usize,
    beta_off: usize,
    alpha_off: usize,
}

impl Layout {
    /// Qwen3.6: `conv_ld` 8192, `proj` row `[z | beta | alpha]` of 4160.
    fn production() -> Self {
        Self {
            conv_ld: 8192,
            ld: 4160,
            z_off: 0,
            beta_off: 4096,
            alpha_off: 4128,
        }
    }

    /// Odd offsets and padded rows, as the RD-4V harness.
    fn odd(nk: usize, nv: usize) -> Self {
        let (alpha_off, z_off) = (3, 3 + nv + 5);
        let beta_off = z_off + nv * D + 7;
        Self {
            conv_ld: (2 * nk + nv) * D + 16,
            ld: beta_off + nv + 11,
            z_off,
            beta_off,
            alpha_off,
        }
    }
}

/// Host inputs of a launch sequence; `conv` and `proj` hold the exact reach and every column the kernel does not read is NaN.
#[derive(Clone)]
struct Inputs {
    nk: usize,
    nv: usize,
    m: usize,
    lay: Layout,
    conv: Vec<f32>,
    proj: Vec<f32>,
    a: Vec<f32>,
    dt: Vec<f32>,
    nw: Vec<f32>,
    state0: Vec<f32>,
}

impl Inputs {
    fn blank(nk: usize, nv: usize, m: usize, lay: Layout) -> Self {
        let need = exact_bytes(Self::shape_of(nk, nv, lay, m));
        let (conv, proj) = (vec![f32::NAN; need.conv / 4], vec![f32::NAN; need.proj / 4]);
        Self {
            nk,
            nv,
            m,
            lay,
            conv,
            proj,
            a: vec![0.0; nv],
            dt: vec![0.0; nv],
            nw: vec![1.0; D],
            state0: vec![0.0; nv * D * D],
        }
    }

    fn shape_of(nk: usize, nv: usize, l: Layout, m: usize) -> DeltaShape {
        DeltaShape {
            m,
            n_k_heads: nk,
            n_v_heads: nv,
            conv_ld: l.conv_ld,
            ld: l.ld,
            z_off: l.z_off,
            beta_off: l.beta_off,
            alpha_off: l.alpha_off,
        }
    }

    fn shape(&self, m: usize) -> DeltaShape {
        Self::shape_of(self.nk, self.nv, self.lay, m)
    }
    fn conv_row(&self, t: usize) -> &[f32] {
        &self.conv[t * self.lay.conv_ld..]
    }
    fn proj_row(&self, t: usize) -> &[f32] {
        &self.proj[t * self.lay.ld..]
    }
}

/// `ln(expm1(y))`, the inverse of the kernel's softplus (with its cut above 20).
fn softplus_inv(y: f64) -> f64 {
    if y > 20.0 { y } else { y.exp_m1().ln() }
}

fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.unit()
}

/// The T1 generator of the packet: conv q, k ~ N(0,1), v ~ N(0,7^2), z ~ N(0,3^2), beta logit ~ N(0,2.5^2),
/// `a = -exp(U(-4, 4.3))`, `dt_bias ~ U(-5.3, 15.6)`, a target decay exponent `10^U(-5, 1.5)` per head times
/// `exp(0.3 N)` per step, start state N(0, 0.1^2). With `forced` and 32 heads, heads 29 to 31 get 3e-8, 1e-6, 1e-5.
fn generate(nk: usize, nv: usize, m: usize, lay: Layout, seed: u64, forced: bool) -> Inputs {
    let mut rng = Rng::new(seed);
    let mut inp = Inputs::blank(nk, nv, m, lay);
    for h in 0..nv {
        inp.a[h] = -uniform(&mut rng, -4.0, 4.3).exp() as f32;
        inp.dt[h] = uniform(&mut rng, -5.3, 15.6) as f32;
    }
    inp.nw = (0..D).map(|_| uniform(&mut rng, 0.5, 1.5) as f32).collect();
    let base: Vec<f64> = (0..nv)
        .map(|_| 10f64.powf(uniform(&mut rng, -5.0, 1.5)))
        .collect();
    inp.state0
        .iter_mut()
        .for_each(|s| *s = (rng.normal() * 0.1) as f32);
    for t in 0..m {
        let (conv, proj) = (t * lay.conv_ld, t * lay.ld);
        for c in 0..(2 * nk + nv) * D {
            inp.conv[conv + c] = (rng.normal() * if c < 2 * nk * D { 1.0 } else { 7.0 }) as f32;
        }
        for c in 0..nv * D {
            inp.proj[proj + lay.z_off + c] = (rng.normal() * 3.0) as f32;
        }
        for h in 0..nv {
            inp.proj[proj + lay.beta_off + h] = (rng.normal() * 2.5) as f32;
            let e = if forced && nv == 32 && h >= 29 {
                [3e-8, 1e-6, 1e-5][h - 29]
            } else {
                base[h] * (0.3 * rng.normal()).exp()
            };
            inp.proj[proj + lay.alpha_off + h] =
                (softplus_inv(e / f64::from(inp.a[h]).abs()) - f64::from(inp.dt[h])) as f32;
        }
    }
    inp
}

/// Edge rows of the `(16, 32, 4)` shape: a key head with zero q and k, a decay of exactly 1, `a = -72`.
#[derive(Clone, Copy, Debug)]
enum Edge {
    ZeroKeyHead,
    TinyDecay,
    LargeA,
}

fn apply_edge(inp: &mut Inputs, edge: Edge) {
    let (nk, lay) = (inp.nk, inp.lay);
    if let Edge::LargeA = edge {
        inp.a[7] = -72.0;
    }
    for t in 0..inp.m {
        let (conv, proj) = (t * lay.conv_ld, t * lay.ld);
        match edge {
            Edge::ZeroKeyHead => [3 * D, (nk + 3) * D]
                .iter()
                .for_each(|c| inp.conv[conv + c..conv + c + D].fill(0.0)),
            Edge::TinyDecay => inp.proj[proj + lay.alpha_off + 5] = -30.0 - inp.dt[5],
            Edge::LargeA => {
                inp.proj[proj + lay.alpha_off + 7] =
                    (softplus_inv(0.5 / 72.0) - f64::from(inp.dt[7])) as f32
            }
        }
    }
}

// ---- f64 reference

/// Variants of the reference that must differ from the real one (the comparison is not blind to them).
#[derive(Clone, Copy, Default)]
struct Blind {
    grouped_heads: bool,
    sigmoid_gate: bool,
    flip_a: bool,
}

/// Reference output for a list of heads: `out[(t * heads + hi) * 128 + j]`, the delta-rule output before the gated
/// norm (`pre`) and the end state `[hi][i][j]`.
struct RefRun {
    heads: Vec<usize>,
    out: Vec<f64>,
    pre: Vec<f64>,
    state: Vec<f64>,
}

/// One step of the delta rule for one head in f64, in the kernel's state layout `s[i * 128 + j]`.
fn ref_step(s: &mut [f64], q: &[f64], k: &[f64], v: &[f64], (decay, beta): (f64, f64)) -> Vec<f64> {
    let mut sk = vec![0.0; D];
    for i in 0..D {
        for j in 0..D {
            s[i * D + j] *= decay;
            sk[j] += s[i * D + j] * k[i];
        }
    }
    let (mut o, d): (Vec<f64>, Vec<f64>) = (
        vec![0.0; D],
        (0..D).map(|j| (v[j] - sk[j]) * beta).collect(),
    );
    for i in 0..D {
        for j in 0..D {
            s[i * D + j] += k[i] * d[j];
            o[j] += s[i * D + j] * q[i];
        }
    }
    o
}

/// `x * scale / sqrt(sum(x^2) + eps)` in f64.
fn l2(x: &[f32], eps: f64, scale: f64) -> Vec<f64> {
    let factor = scale / (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() + eps).sqrt();
    x.iter().map(|&v| f64::from(v) * factor).collect()
}

/// The kernel's rule in f64: tiled key heads `h % n_k`, `silu(z)` gate, `a` as stored, the exact decay. Written from
/// the GGUF semantics, not from the kernel text.
fn ref_delta(inp: &Inputs, heads: &[usize], blind: Blind) -> RefRun {
    let (eps, lay, nk) = (f64::from(GDN_EPS), inp.lay, inp.nk);
    let n = heads.len();
    let mut run = RefRun {
        heads: heads.to_vec(),
        out: vec![0.0; inp.m * n * D],
        pre: vec![0.0; inp.m * n * D],
        state: Vec::new(),
    };
    for (hi, &h) in heads.iter().enumerate() {
        let kh = if blind.grouped_heads {
            h / (inp.nv / nk)
        } else {
            h % nk
        };
        let mut s: Vec<f64> = inp.state0[h * D * D..(h + 1) * D * D]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        for t in 0..inp.m {
            let (conv, proj) = (inp.conv_row(t), inp.proj_row(t));
            let (q, k) = (
                l2(&conv[kh * D..][..D], eps, 1.0 / (D as f64).sqrt()),
                l2(&conv[(nk + kh) * D..][..D], eps, 1.0),
            );
            let v: Vec<f64> = conv[2 * nk * D + h * D..][..D]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let a = f64::from(inp.a[h]) * if blind.flip_a { -1.0 } else { 1.0 };
            let decay =
                (softplus_f64(f64::from(proj[lay.alpha_off + h]) + f64::from(inp.dt[h])) * a).exp();
            let beta = 1.0 / (1.0 + (-f64::from(proj[lay.beta_off + h])).exp());
            let o = ref_step(&mut s, &q, &k, &v, (decay, beta));
            let rstd = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / D as f64 + eps).sqrt();
            for j in 0..D {
                let z = f64::from(proj[lay.z_off + h * D + j]);
                let gate = if blind.sigmoid_gate {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    silu_f64(z)
                };
                run.pre[(t * n + hi) * D + j] = o[j];
                run.out[(t * n + hi) * D + j] = o[j] * rstd * f64::from(inp.nw[j]) * gate;
            }
        }
        run.state.extend(s);
    }
    run
}

// ---- comparisons

/// The worst of a group: value, step and head.
#[derive(Clone, Copy, Default)]
struct Stat {
    max: f64,
    step: usize,
    head: usize,
}

impl Stat {
    fn note(&mut self, value: f64, step: usize, head: usize) {
        if value > self.max {
            *self = Self {
                max: value,
                step,
                head,
            };
        }
    }
    fn show(&self) -> String {
        format!("{:e} (head {}, step {})", self.max, self.head, self.step)
    }
}

/// `|g - w|`, infinite for a NaN on either side.
fn gap(g: f64, w: f64) -> f64 {
    let d = (g - w).abs();
    if d.is_nan() { f64::INFINITY } else { d }
}

/// E-form error of one step: `max |d| / max(max |want|, 1)` over the heads, and the head of the largest `|d|`.
fn step_error(got: &[&[f32]], want: &[&[f64]], heads: &[usize]) -> (f64, usize) {
    let top = want
        .iter()
        .flat_map(|w| w.iter())
        .fold(1.0f64, |m, x| m.max(x.abs()));
    let mut worst = (0.0f64, heads[0]);
    for ((g, w), &h) in got.iter().zip(want).zip(heads) {
        let diff = g
            .iter()
            .zip(w.iter())
            .fold(0.0f64, |m, (&a, &b)| m.max(gap(f64::from(a), b)));
        if diff > worst.0 {
            worst = (diff, h);
        }
    }
    (worst.0 / top, worst.1)
}

/// Device results: `out[(t * n_v + h) * 128 + j]` and the end state.
struct Run {
    out: Vec<f32>,
    state: Vec<f32>,
}

/// Error of step `t` of `run` against the reference.
fn run_step_error(inp: &Inputs, run: &Run, want: &RefRun, t: usize) -> (f64, usize) {
    let n = want.heads.len();
    let got: Vec<&[f32]> = want
        .heads
        .iter()
        .map(|&h| &run.out[(t * inp.nv + h) * D..][..D])
        .collect();
    let exp: Vec<&[f64]> = (0..n)
        .map(|hi| &want.out[(t * n + hi) * D..][..D])
        .collect();
    step_error(&got, &exp, &want.heads)
}

/// End-state error: `(max |S - S_ref|, max |S_ref|)`; E-form divides by `max(max |S_ref|, 1)`.
fn state_error(run: &Run, want: &RefRun) -> (f64, f64) {
    let max_ref = want.state.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut diff = 0.0f64;
    for (hi, &h) in want.heads.iter().enumerate() {
        let (got, exp) = (
            &run.state[h * D * D..(h + 1) * D * D],
            &want.state[hi * D * D..(hi + 1) * D * D],
        );
        diff = got
            .iter()
            .zip(exp)
            .fold(diff, |m, (&g, &w)| m.max(gap(f64::from(g), w)));
    }
    (diff, max_ref)
}

/// The worst over all steps and over the steps from 128 on.
struct Group {
    all: Stat,
    late: Stat,
}

/// Compares `run` with the f64 reference on `heads`, prints one line and asserts E4 for the steps and the end state.
fn judge(label: &str, inp: &Inputs, run: &Run, heads: &[usize]) -> Group {
    let want = ref_delta(inp, heads, Blind::default());
    let mut group = Group {
        all: Stat::default(),
        late: Stat::default(),
    };
    for t in 0..inp.m {
        let (error, head) = run_step_error(inp, run, &want, t);
        group.all.note(error, t, head);
        if t >= DOMAIN_START {
            group.late.note(error, t, head);
        }
    }
    let (diff, max_ref) = state_error(run, &want);
    let state = diff / max_ref.max(1.0);
    println!(
        "GROUP {label}: all steps {}; steps>=128 {}; end state {state:e}, max|S_ref| {max_ref:.4}, state/max|S_ref| {:e}",
        group.all.show(),
        group.late.show(),
        diff / max_ref
    );
    assert!(
        group.all.max <= E4,
        "{label}: output {} is above E4 {E4:e}",
        group.all.show()
    );
    assert!(
        state <= E4,
        "{label}: end state {state:e} is above E4 {E4:e}"
    );
    group
}

fn stop_rule(label: &str, x: f64) {
    println!(
        "STOP-RULE: x = {x:e} ({label}; <= X_STOP {X_STOP:e}: final; above: STOP and report) holds={}",
        x <= X_STOP
    );
}

// ---- device side

/// A device context and its stream.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        Self {
            stream: ctx.stream().expect("stream"),
            ctx,
        }
    }
}

/// The bytes each range of `shape` must hold, by plain arithmetic written here (not by `delta_bytes`, which is under test).
fn exact_bytes(s: DeltaShape) -> DeltaBytes {
    let proj_cols = [
        s.z_off + s.n_v_heads * D,
        s.beta_off + s.n_v_heads,
        s.alpha_off + s.n_v_heads,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let (nv, conv_cols) = (s.n_v_heads, (2 * s.n_k_heads + s.n_v_heads) * D);
    let row = |ld: usize, cols: usize| ((s.m - 1) * ld + cols) * 4;
    DeltaBytes {
        conv: row(s.conv_ld, conv_cols),
        proj: row(s.ld, proj_cols),
        a: nv * 4,
        dt_bias: nv * 4,
        norm_w: D * 4,
        state: nv * D * D * 4,
        out: s.m * nv * D * 4,
    }
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A device buffer `[sentinel][skew][data][sentinel]` with 256 floats of sentinel on both sides.
struct Padded {
    buf: DeviceBuffer,
    front: usize,
    image: Vec<u8>,
}

impl Padded {
    fn new(ctx: &CudaContext, data: &[u8], skew: usize) -> Self {
        let pad = |n: usize| SENTINEL.iter().copied().cycle().take(n);
        let front = PAD_BYTES + skew;
        let image: Vec<u8> = pad(front)
            .chain(data.iter().copied())
            .chain(pad(PAD_BYTES))
            .collect();
        Self {
            buf: testkit::upload_bytes(ctx, &image),
            front,
            image,
        }
    }

    fn at(&self, offset: usize, bytes: usize) -> DeviceRange<'_> {
        self.buf.range(self.front + offset, bytes).expect("range")
    }

    /// The data bytes, after asserting that both sentinel pads are unchanged; a read-only buffer must keep its data too.
    fn verify(&self, ctx: &CudaContext, read_only: bool, name: &str) -> Vec<u8> {
        let (now, end) = (
            testkit::read_bytes(ctx, &self.buf, self.image.len()),
            self.image.len() - PAD_BYTES,
        );
        assert!(
            now[..self.front] == self.image[..self.front],
            "{name}: the front sentinel changed"
        );
        assert!(
            now[end..] == self.image[end..],
            "{name}: the back sentinel changed"
        );
        assert!(
            !read_only || now == self.image,
            "{name}: a read-only range changed"
        );
        now[self.front..end].to_vec()
    }
}

/// The slots of a call, in this order.
const CONV: usize = 0;
const PROJ: usize = 1;
const A: usize = 2;
const DT: usize = 3;
const NW: usize = 4;
const STATE: usize = 5;
const OUT: usize = 6;
const NAMES: [&str; 7] = ["conv", "proj", "a", "dt_bias", "norm_w", "state", "out"];

/// Device buffers and where each slot sits in them: (buffer index, byte offset).
struct Bufs {
    pads: Vec<Padded>,
    at: [(usize, usize); 7],
    out_floats: usize,
    state_floats: usize,
}

impl Bufs {
    /// Seven separate buffers of the inputs, every data region `skew` bytes past its pad.
    fn separate(ctx: &CudaContext, inp: &Inputs, skew: usize) -> Self {
        let out = vec![0xFFu8; inp.m * inp.nv * D * 4];
        let images = [
            le_bytes(&inp.conv),
            le_bytes(&inp.proj),
            le_bytes(&inp.a),
            le_bytes(&inp.dt),
            le_bytes(&inp.nw),
            le_bytes(&inp.state0),
            out,
        ];
        let pads = images
            .iter()
            .map(|data| Padded::new(ctx, data, skew))
            .collect();
        Self {
            pads,
            at: std::array::from_fn(|i| (i, 0)),
            out_floats: inp.m * inp.nv * D,
            state_floats: inp.nv * D * D,
        }
    }

    fn range(&self, slot: usize, extra: usize, bytes: usize) -> DeviceRange<'_> {
        let (buffer, offset) = self.at[slot];
        self.pads[buffer].at(offset + extra, bytes)
    }

    /// The call for `shape.m` tokens that start at token `row` of the sequence.
    fn call(&self, shape: DeltaShape, row: usize) -> DeltaCall<'_> {
        let need = exact_bytes(shape);
        DeltaCall {
            shape,
            eps: GDN_EPS,
            conv: self.range(CONV, row * shape.conv_ld * 4, need.conv),
            proj: self.range(PROJ, row * shape.ld * 4, need.proj),
            weights: DeltaWeights {
                a: self.range(A, 0, need.a),
                dt_bias: self.range(DT, 0, need.dt_bias),
                norm_w: self.range(NW, 0, need.norm_w),
            },
            state: self.range(STATE, 0, need.state),
            out: self.range(OUT, row * shape.n_v_heads * D * 4, need.out),
        }
    }

    /// Checks every sentinel and every read-only buffer; returns `out` and the end state.
    fn finish(&self, rig: &Rig) -> Run {
        let written = [self.at[STATE].0, self.at[OUT].0];
        let data: Vec<Vec<u8>> = self
            .pads
            .iter()
            .enumerate()
            .map(|(i, pad)| pad.verify(&rig.ctx, !written.contains(&i), NAMES[i.min(OUT)]))
            .collect();
        let fetch = |(buffer, offset): (usize, usize), floats: usize| {
            testkit::floats(&data[buffer][offset..offset + floats * 4])
        };
        Run {
            out: fetch(self.at[OUT], self.out_floats),
            state: fetch(self.at[STATE], self.state_floats),
        }
    }

    fn launch(&self, rig: &Rig, shape: DeltaShape, row: usize) {
        rig.ctx
            .gdn_delta(&rig.stream, self.call(shape, row))
            .expect("queue the delta step");
        rig.stream.synchronize().expect("drain");
    }
}

/// Runs `inp` as consecutive launches of the sizes in `split` (they sum to `inp.m`) on fresh buffers.
fn launch_all(rig: &Rig, inp: &Inputs, split: &[usize]) -> Run {
    assert_eq!(split.iter().sum::<usize>(), inp.m);
    let bufs = Bufs::separate(&rig.ctx, inp, 0);
    let mut row = 0;
    for &m in split {
        bufs.launch(rig, inp.shape(m), row);
        row += m;
    }
    bufs.finish(rig)
}

// ---- T1

fn all_heads(nv: usize) -> Vec<usize> {
    (0..nv).collect()
}

fn seed_of(nk: usize, nv: usize, m: usize) -> u64 {
    0xd17a_0000 ^ ((nk as u64) << 32) ^ ((nv as u64) << 16) ^ m as u64
}

#[test]
fn delta_matches_the_f64_reference_short() {
    if !testkit::device_present("delta_matches_the_f64_reference_short") {
        return;
    }
    let rig = Rig::new();
    let shapes = [
        (16, 32, 1),
        (16, 32, 4),
        (16, 32, 16),
        (16, 32, 69),
        (2, 4, 3),
        (1, 1, 2),
        (1, 8, 2),
        (4, 12, 5),
        (8, 32, 3),
        (64, 128, 1),
    ];
    for (nk, nv, m) in shapes {
        let inp = generate(nk, nv, m, Layout::odd(nk, nv), seed_of(nk, nv, m), false);
        let run = launch_all(&rig, &inp, &[m]);
        judge(&format!("T1a ({nk},{nv},{m})"), &inp, &run, &all_heads(nv));
        if (nk, nv, m) == (16, 32, 1) {
            assert_not_blind(&inp, &run);
        }
    }
    for edge in [Edge::ZeroKeyHead, Edge::TinyDecay, Edge::LargeA] {
        let mut inp = generate(
            16,
            32,
            4,
            Layout::odd(16, 32),
            seed_of(16, 32, 4) ^ 0x77,
            false,
        );
        apply_edge(&mut inp, edge);
        let run = launch_all(&rig, &inp, &[4]);
        assert!(
            run.out.iter().all(|x| x.is_finite()),
            "{edge:?}: non-finite output"
        );
        judge(&format!("T1a edge {edge:?}"), &inp, &run, &all_heads(32));
    }
}

/// The reference with grouped key heads, `sigmoid(z)` or the sign of `a` flipped differs from the real one by more than 1e-2.
fn assert_not_blind(inp: &Inputs, run: &Run) {
    let variants = [
        (
            "grouped key heads",
            Blind {
                grouped_heads: true,
                ..Blind::default()
            },
        ),
        (
            "sigmoid(z)",
            Blind {
                sigmoid_gate: true,
                ..Blind::default()
            },
        ),
        (
            "sign of a",
            Blind {
                flip_a: true,
                ..Blind::default()
            },
        ),
    ];
    for (name, blind) in variants {
        let (error, _) = run_step_error(inp, run, &ref_delta(inp, &all_heads(inp.nv), blind), 0);
        println!("NOT-BLIND {name}: {error:e}");
        assert!(
            error.is_nan() || error > 1e-2,
            "the reference is blind to {name}: {error:e}"
        );
    }
}

/// One launch of `steps` tokens on the production layout; heads: four ordinary ones and the forced-decay heads 29 to 31.
fn long_rollout(rig: &Rig, steps: usize, label: &str) -> Group {
    let inp = generate(
        16,
        32,
        steps,
        Layout::production(),
        0x71b0_0000 ^ steps as u64,
        true,
    );
    let group = judge(
        label,
        &inp,
        &launch_all(rig, &inp, &[steps]),
        &[0, 1, 16, 28, 29, 30, 31],
    );
    println!("LIMIT {label}: {LIMIT}");
    group
}

#[test]
fn delta_matches_the_f64_reference_300() {
    if !testkit::device_present("delta_matches_the_f64_reference_300") {
        return;
    }
    stop_rule(
        "T1b steps 128 to 299",
        long_rollout(&Rig::new(), 300, "T1b (16,32,300)").late.max,
    );
}

#[test]
fn delta_matches_the_f64_reference_1152() {
    if !testkit::device_present("delta_matches_the_f64_reference_1152") {
        return;
    }
    let group = long_rollout(&Rig::new(), 1152, "T1c (16,32,1152) reported only");
    println!(
        "T1c reported only (REPLY-59 Q1): maximum {}",
        group.late.show()
    );
}

#[test]
fn delta_pins_the_qk_norm_eps() {
    if !testkit::device_present("delta_pins_the_qk_norm_eps") {
        return;
    }
    let rig = Rig::new();
    for (nk, nv, m) in [(16, 32, 1), (2, 4, 3)] {
        for sum in [1e-6f64, 1e-4, 1e-2] {
            let mut inp = generate(
                nk,
                nv,
                m,
                Layout::odd(nk, nv),
                seed_of(nk, nv, m) ^ 0x5,
                false,
            );
            inp.state0.fill(0.0);
            for t in 0..m {
                for seg in 0..2 * nk {
                    let row = &mut inp.conv[t * inp.lay.conv_ld + seg * D..][..D];
                    let norm = row
                        .iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    row.iter_mut()
                        .for_each(|x| *x = (f64::from(*x) * sum.sqrt() / norm) as f32);
                }
            }
            judge(
                &format!("T1d ({nk},{nv},{m}) sum(x^2)={sum:e}"),
                &inp,
                &launch_all(&rig, &inp, &[m]),
                &all_heads(nv),
            );
        }
    }
}

// ---- T3, T4, T7, T8

#[test]
fn delta_state_carries_across_calls() {
    if !testkit::device_present("delta_state_carries_across_calls") {
        return;
    }
    let rig = Rig::new();
    let inp = generate(16, 32, 12, Layout::odd(16, 32), 0x7033, false);
    let whole = launch_all(&rig, &inp, &[12]);
    for split in [vec![5, 7], vec![1; 12]] {
        let run = launch_all(&rig, &inp, &split);
        assert_eq!(
            le_bytes(&run.out),
            le_bytes(&whole.out),
            "out, split {split:?}"
        );
        assert_eq!(
            le_bytes(&run.state),
            le_bytes(&whole.state),
            "state, split {split:?}"
        );
    }
}

#[test]
fn delta_is_deterministic() {
    if !testkit::device_present("delta_is_deterministic") {
        return;
    }
    let rig = Rig::new();
    let inp = generate(16, 32, 3, Layout::odd(16, 32), 0x7044, false);
    let bufs = Bufs::separate(&rig.ctx, &inp, 0);
    // A finite start for `out`, so that an output which depends on its previous contents (0.5 * out + ...) differs
    // between the runs instead of staying NaN.
    let out = &bufs.pads[bufs.at[OUT].0];
    out.buf
        .write_at(
            &rig.stream,
            out.front,
            &le_bytes(&vec![1.0; bufs.out_floats]),
        )
        .expect("finite out");
    let mut outs = Vec::new();
    for _ in 0..2 {
        bufs.launch(&rig, inp.shape(3), 0);
        outs.push(le_bytes(&bufs.finish(&rig).out));
        let pad = &bufs.pads[bufs.at[STATE].0];
        pad.buf
            .write_at(&rig.stream, pad.front, &le_bytes(&inp.state0))
            .expect("reset the state");
    }
    assert_eq!(
        outs[0], outs[1],
        "the second run into the same out buffer differs"
    );
}

#[test]
fn delta_state_index_roles() {
    if !testkit::device_present("delta_state_index_roles") {
        return;
    }
    let lay = Layout {
        conv_ld: 3 * D,
        ld: D + 2,
        z_off: 0,
        beta_off: D,
        alpha_off: D + 1,
    };
    let mut inp = Inputs::blank(1, 1, 1, lay);
    (inp.state0[5 * D + 9], inp.state0[9 * D + 5]) = (3.0, -2.0);
    inp.conv[..D].fill(0.0);
    inp.conv[5] = 1.0;
    inp.conv[D..2 * D].fill(1.0);
    inp.conv[2 * D..3 * D].fill(0.0);
    inp.proj[..D].fill(1.0);
    (inp.proj[D], inp.proj[D + 1], inp.a[0]) = (-30.0, -30.0, -1.0);
    let run = launch_all(&Rig::new(), &inp, &[1]);
    let want = ref_delta(&inp, &[0], Blind::default());
    let (error, _) = run_step_error(&inp, &run, &want, 0);
    println!(
        "T7 out error {error:e}, out[9] {} against {}",
        run.out[9], want.out[9]
    );
    assert!(
        error <= E4 && want.out[9].abs() > 1.0,
        "out against the reference: {error:e}"
    );
    assert!(
        (0..D).filter(|&j| j != 9).all(|j| run.out[j].abs() <= 1e-6),
        "out is nonzero away from j = 9"
    );
    let moved = run
        .state
        .iter()
        .zip(&inp.state0)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(moved <= 1e-9, "the state changed by {moved:e}");
}

#[test]
fn delta_bytes_pins_the_size_rules() {
    let shape = DeltaShape {
        m: 1,
        n_k_heads: 16,
        n_v_heads: 32,
        conv_ld: 8192,
        ld: 4160,
        z_off: 0,
        beta_off: 4096,
        alpha_off: 4128,
    };
    let want = DeltaBytes {
        conv: 32_768,
        proj: 16_640,
        a: 128,
        dt_bias: 128,
        norm_w: 512,
        state: 2_097_152,
        out: 16_384,
    };
    assert_eq!(delta_bytes(shape).expect("sizes"), want);
    let three = delta_bytes(DeltaShape { m: 3, ..shape }).expect("sizes");
    assert_eq!(
        (three.conv, three.proj, three.out),
        ((2 * 8192 + 8192) * 4, (2 * 4160 + 4160) * 4, 3 * 16_384)
    );
    // Rows longer than the columns read: the reach is (m - 1) * row length + columns, not m * row length.
    let wide = delta_bytes(DeltaShape {
        m: 3,
        conv_ld: 8200,
        ld: 4167,
        ..shape
    })
    .expect("sizes");
    assert_eq!(
        (wide.conv, wide.proj),
        ((2 * 8200 + 8192) * 4, (2 * 4167 + 4160) * 4)
    );
    for bad in [
        DeltaShape {
            conv_ld: usize::MAX / 2,
            m: 4,
            ..shape
        },
        DeltaShape { m: 0, ..shape },
    ] {
        assert!(matches!(delta_bytes(bad), Err(EngError::BadRequest(_))));
    }
}

// ---- T5, T5b

#[derive(Clone, Copy)]
struct Slot {
    offset: usize,
    bytes: usize,
}

/// One refusal row: the geometry, the byte slots of the seven ranges, and what to break.
#[derive(Clone)]
struct Plan {
    shape: DeltaShape,
    eps: f32,
    slots: [Slot; 7],
    alias: Option<(usize, usize)>,
    foreign_stream: bool,
    foreign_conv: bool,
}

impl Plan {
    /// A valid call: Qwen3.6 geometry, two tokens, every buffer exactly sized.
    fn base() -> Self {
        let shape = Inputs::shape_of(16, 32, Layout::production(), 2);
        let mut plan = Self {
            shape,
            eps: GDN_EPS,
            slots: [Slot {
                offset: 0,
                bytes: 0,
            }; 7],
            alias: None,
            foreign_stream: false,
            foreign_conv: false,
        };
        plan.fit();
        plan
    }

    /// Sizes every range for the current geometry (plain arithmetic, written here on purpose).
    fn fit(&mut self) {
        let s = self.shape;
        let proj_cols = [
            s.z_off + s.n_v_heads * D,
            s.beta_off + s.n_v_heads,
            s.alpha_off + s.n_v_heads,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let bytes = [
            ((s.m - 1) * s.conv_ld + (2 * s.n_k_heads + s.n_v_heads) * D) * 4,
            ((s.m - 1) * s.ld + proj_cols) * 4,
            s.n_v_heads * 4,
            s.n_v_heads * 4,
            D * 4,
            s.n_v_heads * D * D * 4,
            s.m * s.n_v_heads * D * 4,
        ];
        self.slots
            .iter_mut()
            .zip(bytes)
            .for_each(|(slot, bytes)| *slot = Slot { offset: 0, bytes });
    }
}

/// Runs the call of `plan` on zero-filled buffers (`out` is 0xFF); returns the result, whether every buffer is unchanged
/// and the `out` bytes. `other` supplies the foreign stream and buffer.
fn attempt(rig: &Rig, other: &Rig, plan: &Plan) -> (Result<(), EngError>, bool, Vec<u8>) {
    let ctx = &rig.ctx;
    let bufs: Vec<DeviceBuffer> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            testkit::upload_filled(
                ctx,
                if i == OUT { 0xFF } else { 0 },
                s.offset + s.bytes.max(4),
            )
        })
        .collect();
    let shared = plan
        .alias
        .map(|(x, y)| testkit::upload_filled(ctx, 0, plan.slots[x].bytes.max(plan.slots[y].bytes)));
    let foreign = testkit::upload_filled(&other.ctx, 0, plan.slots[CONV].bytes);
    let hashes = || {
        bufs.iter()
            .chain(shared.iter())
            .map(|b| fnv1a64(&testkit::read_bytes(ctx, b, b.bytes())))
            .collect::<Vec<_>>()
    };
    let before = hashes();
    let range = |i: usize| -> DeviceRange<'_> {
        let buffer = match (&shared, plan.alias) {
            (Some(buf), Some((x, y))) if i == x || i == y => buf,
            _ if i == CONV && plan.foreign_conv => &foreign,
            _ => &bufs[i],
        };
        buffer
            .range(plan.slots[i].offset, plan.slots[i].bytes)
            .expect("range")
    };
    let weights = DeltaWeights {
        a: range(A),
        dt_bias: range(DT),
        norm_w: range(NW),
    };
    let call = DeltaCall {
        shape: plan.shape,
        eps: plan.eps,
        conv: range(CONV),
        proj: range(PROJ),
        weights,
        state: range(STATE),
        out: range(OUT),
    };
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let result = ctx.gdn_delta(stream, call);
    // A refused call queued nothing; a call the wrapper wrongly accepted may latch the backend, and the row must
    // then fail on its own result (with its label), not on the drain.
    let drained = stream.synchronize();
    if result.is_ok() {
        drained.expect("drain");
    } else if !matches!(result, Err(EngError::BadRequest(_))) {
        return (result, false, Vec::new());
    }
    (
        result,
        hashes() == before,
        testkit::read_bytes(ctx, &bufs[OUT], bufs[OUT].bytes()),
    )
}

type Edit = fn(&mut Plan);

/// The Rust rows of the refusal table (label, text the refusal must contain, edit); every edit starts from `Plan::base()`.
fn refusals() -> Vec<(&'static str, &'static str, Edit)> {
    vec![
        ("m 0", "gdn_delta: tokens", |p| p.shape.m = 0),
        ("m 32769", "gdn_delta: tokens", |p| p.shape.m = 32769),
        ("n_k 0", "gdn_delta: n_k_heads", |p| p.shape.n_k_heads = 0),
        ("n_k 65", "gdn_delta: n_k_heads", |p| p.shape.n_k_heads = 65),
        ("n_v 0", "gdn_delta: n_v_heads", |p| p.shape.n_v_heads = 0),
        ("n_v 129", "gdn_delta: n_v_heads", |p| {
            p.shape.n_v_heads = 129
        }),
        ("n_v 32 with n_k 3", "gdn_delta: heads", |p| {
            p.shape.n_k_heads = 3;
            p.fit();
        }),
        ("n_v 8 with n_k 16", "gdn_delta: heads", |p| {
            p.shape.n_v_heads = 8;
            p.fit();
        }),
        ("conv_ld 8191", "gdn_delta: conv_ld", |p| {
            p.shape.conv_ld = 8191
        }),
        ("conv_ld 1048577", "gdn_delta: conv_ld", |p| {
            p.shape.conv_ld = 1_048_577
        }),
        ("ld below the z end", "is below the z end", |p| {
            (
                p.shape.z_off,
                p.shape.beta_off,
                p.shape.alpha_off,
                p.shape.ld,
            ) = (100, 0, 40, 4195);
            p.fit();
        }),
        ("ld below the beta end", "is below the beta end", |p| {
            (
                p.shape.z_off,
                p.shape.beta_off,
                p.shape.alpha_off,
                p.shape.ld,
            ) = (0, 4200, 0, 4231);
            p.fit();
        }),
        ("ld below the alpha end", "is below the alpha end", |p| {
            p.shape.ld = 4159;
            p.fit();
        }),
        ("ld 4194305", "gdn_delta: ld", |p| p.shape.ld = 4_194_305),
        ("z_off 2^24+1", "is below the z end", |p| {
            p.shape.z_off = (1 << 24) + 1
        }),
        ("eps 1e-5", "gdn_delta: eps", |p| p.eps = 1e-5),
        ("eps 0", "gdn_delta: eps", |p| p.eps = 0.0),
        ("eps -0", "gdn_delta: eps", |p| p.eps = -0.0),
        ("eps NaN", "gdn_delta: eps", |p| p.eps = f32::NAN),
        ("eps one ulp above", "gdn_delta: eps", |p| {
            p.eps = f32::from_bits(GDN_EPS.to_bits() + 1)
        }),
        ("eps one ulp below", "gdn_delta: eps", |p| {
            p.eps = f32::from_bits(GDN_EPS.to_bits() - 1)
        }),
        ("stream of another context", "stream belongs", |p| {
            p.foreign_stream = true
        }),
        (
            "buffer of another context",
            "belongs to another adapter",
            |p| p.foreign_conv = true,
        ),
    ]
}

/// The size, alignment and overlap rows: for each range one float short and 2 bytes in, and the 11 write/read pairs.
fn slot_plans() -> Vec<(String, String, Plan)> {
    let mut rows = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        let (mut short, mut skew) = (Plan::base(), Plan::base());
        short.slots[i].bytes -= 4;
        skew.slots[i].offset = 2;
        rows.push((
            format!("{name} one float short"),
            format!("the CUDA {name} range holds"),
            short,
        ));
        rows.push((
            format!("{name} +2 bytes"),
            format!("the CUDA {name} range starts at offset"),
            skew,
        ));
    }
    for (w, r) in [STATE, OUT]
        .into_iter()
        .flat_map(|w| {
            [OUT, CONV, PROJ, A, DT, NW]
                .into_iter()
                .map(move |r| (w, r))
        })
        .filter(|&(w, r)| w != r && !(w == OUT && r == OUT))
    {
        let mut plan = Plan::base();
        plan.alias = Some((w, r));
        rows.push((
            format!("{} and {} overlap", NAMES[w], NAMES[r]),
            format!("the CUDA {} and {} ranges overlap", NAMES[w], NAMES[r]),
            plan,
        ));
    }
    rows
}

#[test]
fn delta_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("delta_rejects_bad_calls_before_any_launch") {
        return;
    }
    let (rig, other) = (Rig::new(), Rig::new());
    let (first, _, first_out) = attempt(&rig, &other, &Plan::base());
    first.expect("the base call is valid");
    let mut rows = slot_plans();
    for (label, needle, edit) in refusals() {
        let mut plan = Plan::base();
        edit(&mut plan);
        rows.push((label.to_owned(), needle.to_owned(), plan));
    }
    assert_eq!(rows.len(), 14 + 11 + 23, "the pairs of the table");
    for (label, needle, plan) in rows {
        let (result, unchanged, out) = attempt(&rig, &other, &plan);
        let Err(EngError::BadRequest(message)) = result else {
            panic!("{label}: expected a BadRequest, got {result:?}")
        };
        assert!(
            message.contains(&needle),
            "{label}: expected {needle:?}, got {message:?}"
        );
        assert!(
            unchanged && out.iter().all(|&b| b == 0xFF),
            "{label}: a buffer changed"
        );
    }
    let (again, _, again_out) = attempt(&rig, &other, &Plan::base());
    again.expect("the backend is usable after the refusals");
    assert_eq!(first_out, again_out, "the base call after the refusals");
}

#[test]
fn delta_accepts_the_exact_sizes() {
    if !testkit::device_present("delta_accepts_the_exact_sizes") {
        return;
    }
    let rig = Rig::new();
    let lay = Layout {
        conv_ld: 8192 + 16,
        ld: 4160 + 7,
        ..Layout::production()
    };
    let inp = generate(16, 32, 3, lay, 0x7055, false);
    for skew in [0, 4] {
        let bufs = Bufs::separate(&rig.ctx, &inp, skew);
        bufs.launch(&rig, inp.shape(3), 0);
        judge(
            &format!("T5b exact sizes, skew {skew}"),
            &inp,
            &bufs.finish(&rig),
            &[0, 5, 17, 31],
        );
    }
    // conv and proj in one buffer, a and dt_bias in one buffer, out right after the conv reach.
    let mut rng = Rng::new(0x7056);
    let need = exact_bytes(inp.shape(3));
    let image: Vec<f32> = (0..need.conv / 4).map(|_| rng.normal() as f32).collect();
    let vec_a: Vec<f32> = (0..32)
        .map(|_| -uniform(&mut rng, 0.1, 1.0) as f32)
        .collect();
    let shared = Inputs {
        conv: image.clone(),
        proj: image[..need.proj / 4].to_vec(),
        a: vec_a.clone(),
        dt: vec_a.clone(),
        ..Inputs::blank(16, 32, 3, lay)
    };
    let first = [le_bytes(&image), vec![0xFFu8; need.out]].concat();
    let pads = vec![
        Padded::new(&rig.ctx, &first, 0),
        Padded::new(&rig.ctx, &le_bytes(&vec_a), 0),
        Padded::new(&rig.ctx, &le_bytes(&shared.nw), 0),
        Padded::new(&rig.ctx, &le_bytes(&shared.state0), 0),
    ];
    let bufs = Bufs {
        pads,
        at: [
            (0, 0),
            (0, 0),
            (1, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, need.conv),
        ],
        out_floats: need.out / 4,
        state_floats: 32 * D * D,
    };
    bufs.launch(&rig, shared.shape(3), 0);
    judge(
        "T5b shared reads",
        &shared,
        &bufs.finish(&rig),
        &[0, 5, 17, 31],
    );
}

// ---- T2

/// The oracle and fixture directories. The oracle tests are `#[ignore]`d (REQUIRE_DEVICE governs the device, never
/// data); a deliberate run with `--ignored` and a missing variable panics, so a skip never counts as a pass.
fn oracle_dirs() -> (PathBuf, PathBuf) {
    let var = |name| std::env::var_os(name).filter(|v| !v.is_empty());
    match (var("QWEN36_ORACLE_DIR"), var("RD7B_FIXTURES")) {
        (Some(oracle), Some(fixtures)) => (oracle.into(), fixtures.into()),
        _ => panic!("QWEN36_ORACLE_DIR and RD7B_FIXTURES must be set to run the oracle tests"),
    }
}

/// A dumped q or k row (already L2-normalized, `x1`) scaled by `c = sqrt(eps / (1 - r))` so that the kernel's own
/// norm returns `x1`; `c = 1` when `1 - r < 1e-12`.
fn prescale(row: &[f32]) -> Vec<f32> {
    let r: f64 = row.iter().map(|&x| f64::from(x).powi(2)).sum();
    let c = if 1.0 - r < 1e-12 {
        1.0
    } else {
        (f64::from(GDN_EPS) / (1.0 - r)).sqrt()
    };
    row.iter().map(|&x| (f64::from(x) * c) as f32).collect()
}

/// The dumps of one layer and prompt, prefill rows followed by decode rows.
struct Oracle {
    prefill_rows: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
}

fn load_oracle(root: &Path, prompt: &str, layer: usize) -> Oracle {
    let mut o = Oracle {
        prefill_rows: 0,
        q: vec![],
        k: vec![],
        v: vec![],
        attn: vec![],
    };
    for phase in ["prefill", "decode"] {
        let load = |key: &str| {
            read_f32_file(
                &root
                    .join("cpu/dump")
                    .join(prompt)
                    .join("dumps")
                    .join(phase)
                    .join(format!("{key}-{layer}.bin")),
            )
            .expect("oracle dump")
        };
        let attn = load("attn_output");
        if phase == "prefill" {
            o.prefill_rows = attn.len() / (32 * D);
        }
        o.q.extend(load("q_conv_predelta"));
        o.k.extend(load("k_conv_predelta"));
        o.v.extend(load("v_conv_predelta"));
        o.attn.extend(attn);
    }
    o
}

/// Conv rows `[q | k | v]` with q and k pre-scaled, proj rows `[z | beta | alpha]` with a synthetic `z` in [1.5, 3.0), the
/// fixture scalars (`a`, `dt_bias`, `norm_w`, then alpha and beta logits per step), zero start state.
fn oracle_inputs(fixture: &[f32], o: &Oracle) -> Inputs {
    let steps = (fixture.len() - 192) / 64;
    assert_eq!(
        steps * 32 * D,
        o.attn.len(),
        "fixture and dump steps differ"
    );
    let lay = Layout::production();
    let mut inp = Inputs {
        a: fixture[..32].to_vec(),
        dt: fixture[32..64].to_vec(),
        nw: fixture[64..192].to_vec(),
        ..Inputs::blank(16, 32, steps, lay)
    };
    let (alpha, beta) = fixture[192..].split_at(steps * 32);
    let mut rng = Rng::new(0x7a0 ^ steps as u64);
    for t in 0..steps {
        let conv = &mut inp.conv[t * lay.conv_ld..];
        for kh in 0..16 {
            conv[kh * D..][..D].copy_from_slice(&prescale(&o.q[(t * 16 + kh) * D..][..D]));
            conv[(16 + kh) * D..][..D].copy_from_slice(&prescale(&o.k[(t * 16 + kh) * D..][..D]));
        }
        conv[32 * D..64 * D].copy_from_slice(&o.v[t * 32 * D..][..32 * D]);
        let proj = &mut inp.proj[t * lay.ld..];
        proj[..32 * D].fill_with(|| uniform(&mut rng, 1.5, 3.0) as f32);
        proj[lay.beta_off..lay.beta_off + 32].copy_from_slice(&beta[t * 32..][..32]);
        proj[lay.alpha_off..lay.alpha_off + 32].copy_from_slice(&alpha[t * 32..][..32]);
    }
    inp
}

/// Stats by phase and step range: [prefill, decode] x [steps below 128, steps from 128].
#[derive(Default)]
struct Grid([[Stat; 2]; 2]);

impl Grid {
    fn note(&mut self, prefill_rows: usize, step: usize, (value, head): (f64, usize)) {
        self.0[usize::from(step >= prefill_rows)][usize::from(step >= DOMAIN_START)]
            .note(value, step, head);
    }

    fn late(&self) -> f64 {
        self.0[0][1].max.max(self.0[1][1].max)
    }
    fn worst(&self) -> f64 {
        self.0.iter().flatten().fold(0.0, |m, s| m.max(s.max))
    }

    fn print(&self, label: &str) {
        for (i, (phase, range)) in [
            ("prefill", "steps<128"),
            ("prefill", "steps>=128"),
            ("decode", "steps<128"),
            ("decode", "steps>=128"),
        ]
        .iter()
        .enumerate()
        {
            println!(
                "GROUP {label} {phase} {range}: {}",
                self.0[i / 2][i % 2].show()
            );
        }
    }
}

/// Part (i): all 32 heads against the gated norm of the dumped `attn_output` (llama.cpp's own delta-rule output). Part (ii):
/// four heads against `ref_delta`, the self check of the reference against the dump, and the end state.
fn judge_oracle(inp: &Inputs, o: &Oracle, run: &Run) -> (Grid, Grid) {
    let (mut part_i, mut part_ii, mut own) = (Grid::default(), Grid::default(), Stat::default());
    let want = ref_delta(inp, &[0, 5, 17, 31], Blind::default());
    let heads = all_heads(32);
    for t in 0..inp.m {
        let expected: Vec<Vec<f64>> = (0..32)
            .map(|h| {
                let pre: Vec<f64> = o.attn[(t * 32 + h) * D..][..D]
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect();
                let rstd = 1.0
                    / (pre.iter().map(|x| x * x).sum::<f64>() / D as f64 + f64::from(GDN_EPS))
                        .sqrt();
                (0..D)
                    .map(|j| {
                        pre[j]
                            * rstd
                            * f64::from(inp.nw[j])
                            * silu_f64(f64::from(inp.proj_row(t)[h * D + j]))
                    })
                    .collect()
            })
            .collect();
        let got: Vec<&[f32]> = heads
            .iter()
            .map(|&h| &run.out[(t * 32 + h) * D..][..D])
            .collect();
        part_i.note(
            o.prefill_rows,
            t,
            step_error(
                &got,
                &expected.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &heads,
            ),
        );
        part_ii.note(o.prefill_rows, t, run_step_error(inp, run, &want, t));
        let dumped: Vec<&[f32]> = want
            .heads
            .iter()
            .map(|&h| &o.attn[(t * 32 + h) * D..][..D])
            .collect();
        let pre: Vec<&[f64]> = (0..4)
            .map(|hi| &want.pre[(t * 4 + hi) * D..][..D])
            .collect();
        own.note(step_error(&dumped, &pre, &want.heads).0, t, 0);
    }
    println!("SELF-CHECK ref_delta against attn_output: {}", own.show());
    assert!(
        own.max <= SELF_CHECK,
        "the reference is off the dump by {:e}",
        own.max
    );
    let (diff, max_ref) = state_error(run, &want);
    println!(
        "END-STATE heads {:?}: {:e}, max|S_ref| {max_ref:.4}, state/max|S_ref| {:e}",
        want.heads,
        diff / max_ref.max(1.0),
        diff / max_ref
    );
    assert!(
        diff / max_ref.max(1.0) <= E4,
        "end state {diff:e} is above E4"
    );
    (part_i, part_ii)
}

fn oracle_test(name: &str, layer: usize, prompt: &str) {
    if !testkit::device_present(name) {
        return;
    }
    let (root, fixtures) = oracle_dirs();
    let fixture =
        read_f32_file(&fixtures.join(format!("rd7b_L{layer}_{prompt}.bin"))).expect("fixture");
    let oracle = load_oracle(&root, prompt, layer);
    let inp = oracle_inputs(&fixture, &oracle);
    let split: Vec<usize> = std::iter::once(oracle.prefill_rows)
        .chain(std::iter::repeat_n(1, inp.m - oracle.prefill_rows))
        .collect();
    let (part_i, part_ii) = judge_oracle(&inp, &oracle, &launch_all(&Rig::new(), &inp, &split));
    let tag = format!("L{layer} {prompt}");
    part_i.print(&format!("{tag} part (i), not counted in x"));
    part_ii.print(&format!("{tag} part (ii), counts in x from step 128"));
    println!("LIMIT {tag}: {LIMIT}");
    assert!(
        part_i.worst() <= E4 && part_ii.worst() <= E4,
        "{tag}: above E4 (part i {:e}, part ii {:e})",
        part_i.worst(),
        part_ii.worst()
    );
    stop_rule(&format!("{tag} part (ii) steps >= 128"), part_ii.late());
}

macro_rules! oracle_tests {
    ($($name:ident: $layer:literal, $prompt:literal;)*) => { $(#[test] #[ignore = "needs QWEN36_ORACLE_DIR and RD7B_FIXTURES"] fn $name() { oracle_test(stringify!($name), $layer, $prompt); })* };
}

oracle_tests! {
    delta_oracle_L0_P4: 0, "P4";
    delta_oracle_L1_P4: 1, "P4";
    delta_oracle_L17_P4: 17, "P4";
    delta_oracle_L38_P4: 38, "P4";
    delta_oracle_L1_P2: 1, "P2";
    delta_oracle_L5_P1: 5, "P1";
}

// ---------------------------------------------------------------- T6 (RD-11a)

/// The capture plan of the owed test: the rig owns every buffer, so no pin and no hold.
fn no_pins() -> CapturePlan<'static> {
    CapturePlan {
        pins: &[],
        holds: Vec::new(),
    }
}

/// T6 (RD-7b, owed): one `gdn_delta` call captured and replayed twice equals the eager run bit
/// for bit (out and end state), on a one-node graph. The warm-up launch and the eager reference
/// run on scratch buffers, so the captured buffers still hold their start state.
#[test]
fn delta_inside_a_capture_equals_eager() {
    if !testkit::device_present("delta_inside_a_capture_equals_eager") {
        return;
    }
    let rig = Rig::new();
    let (nk, nv, m) = (4, 8, 1);
    let lay = Layout::odd(nk, nv);
    let inp = generate(nk, nv, m, lay, 0xd17a_1105, false);
    let shape = inp.shape(m);
    let bits = |values: &[f32]| -> Vec<u32> { values.iter().map(|f| f.to_bits()).collect() };
    let same = |got: &Run, want: &Run, case: &str| {
        assert_eq!(bits(&got.out), bits(&want.out), "{case}: out differs");
        assert_eq!(bits(&got.state), bits(&want.state), "{case}: state differs");
    };
    let eager = launch_all(&rig, &inp, &[m]);
    let bufs = Bufs::separate(&rig.ctx, &inp, 0);
    let (graph, ()) = rig
        .ctx
        .capture(&rig.stream, no_pins(), || {
            rig.ctx.gdn_delta(&rig.stream, bufs.call(shape, 0))
        })
        .expect("capture of the delta step");
    assert_eq!(graph.nodes(), 1, "the delta graph holds one kernel");
    graph.launch(&rig.stream).expect("the first replay");
    same(&bufs.finish(&rig), &eager, "replay 1");
    // Replay 2 starts from the same start state: both restores are host writes, outside any
    // window.
    let state0 = le_bytes(&inp.state0);
    bufs.pads[STATE]
        .buf
        .write_at(&rig.stream, PAD_BYTES, &state0)
        .expect("reset the state");
    bufs.pads[OUT]
        .buf
        .write_at(&rig.stream, PAD_BYTES, &vec![0xFF; m * nv * D * 4])
        .expect("poison the output");
    graph.launch(&rig.stream).expect("the second replay");
    same(&bufs.finish(&rig), &eager, "replay 2");
}
