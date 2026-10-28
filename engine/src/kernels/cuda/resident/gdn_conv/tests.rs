//! Seam-1 device tests (3090) of the DeltaNet conv wrapper: the Rust wrapper against the C shim
//! against the vendored `conv_silu_k`, compared with an f64 host reference written for the
//! Qwen3.6 GGUF semantics (window `(s0, s1, s2, x)`, tap 0 on the oldest input, then SiLU) and
//! with the RD-M6 llama.cpp dumps.
//!
//! Every device buffer carries 256 floats of NaN sentinel before and after the used bytes; the
//! sentinels, the unused pad columns of `proj` and the inputs must be unchanged after a launch.
//! Each refusal is checked by the text of its error, so a test fails when another check refuses
//! the call instead of the one under test. The suite runs with `--test-threads=1` (decision
//! 0040).

use std::path::{Path, PathBuf};

use super::{ConvCall, ConvShape, conv_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// Class E1 (frozen, lane M REPLY-44): `max|d| <= 2^-20 * max(max|ref|, 1)` per output vector
/// against an f64 reference; the conv is a 4-term dot plus `x / (1 + __expf(-x))`.
const E1: f64 = 1.0 / 1_048_576.0;
/// Floats of sentinel before and after every buffer's used bytes.
const PAD: usize = 256;
/// The NaN pattern of the sentinels.
const SENT: u32 = 0x7fc0_dead;
/// `l2` epsilon of ggml's `build_gdn_l2_norm`.
const L2_EPS: f64 = 1e-6;
/// Decode rows of the oracle dump that the Q8_0 emulation of the fixture misses (layer, first
/// row, last row): the quantization of one `attn_norm` input there differs from llama.cpp's, so
/// the fixture input is wrong at that step (RD-7a.md "Evidence status"; the propagation fit of
/// `conv_check.py` confirms one wrong conv input). Only the comparison with the dump skips these
/// rows; the kernel is held to `ref_conv` on every row, these included.
const MASKED: &[(usize, usize, usize)] = &[(38, 91, 94)];

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

/// A device buffer of f32 parts between sentinel words. `extra` moves the first data byte past
/// the 256-byte boundary to test alignments.
struct Padded {
    buf: DeviceBuffer,
    lead: usize,
    data_floats: usize,
}

impl Padded {
    fn new(ctx: &CudaContext, parts: &[&[f32]], extra: usize) -> Self {
        let sentinel = f32::from_bits(SENT);
        let lead = PAD * 4 + extra;
        let mut values = vec![sentinel; lead / 4];
        for part in parts {
            values.extend_from_slice(part);
        }
        let data_floats = values.len() - lead / 4;
        values.extend(std::iter::repeat_n(sentinel, PAD));
        Self {
            buf: testkit::upload_f32(ctx, &values),
            lead,
            data_floats,
        }
    }

    fn filled(ctx: &CudaContext, floats: usize) -> Self {
        Self::new(ctx, &[&vec![f32::from_bits(SENT); floats]], 0)
    }

    /// The range of `bytes` bytes that starts `skip` bytes into the data.
    fn range(&self, skip: usize, bytes: usize) -> DeviceRange<'_> {
        self.buf.range(self.lead + skip, bytes).expect("range")
    }

    fn all(&self) -> DeviceRange<'_> {
        self.range(0, self.data_floats * 4)
    }

    /// The data floats; the sentinels on both sides must be intact.
    fn read(&self, ctx: &CudaContext, what: &str) -> Vec<f32> {
        let total = self.lead + self.data_floats * 4 + PAD * 4;
        let all = testkit::floats(&testkit::read_bytes(ctx, &self.buf, total));
        let (lead, rest) = all.split_at(self.lead / 4);
        let (data, trail) = rest.split_at(self.data_floats);
        let intact = |side: &[f32]| side.iter().all(|v| v.to_bits() == SENT);
        assert!(intact(lead), "{what}: the sentinel before the data changed");
        assert!(intact(trail), "{what}: the sentinel after the data changed");
        data.to_vec()
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

/// `m` rows of `ld` floats: N(0, sd) in the first `channels`, NaN in the pad columns.
fn proj_rows(rng: &mut Rng, shape: ConvShape, sd: f64) -> Vec<f32> {
    let mut rows = Vec::with_capacity(shape.m * shape.ld);
    for _ in 0..shape.m {
        rows.extend(draw(rng, shape.channels, sd));
        rows.extend(std::iter::repeat_n(f32::NAN, shape.ld - shape.channels));
    }
    rows
}

/// The f64 reference: `out[t][c] = silu(sum_k w[c][k] * window[k])` over the window
/// `(s0, s1, s2, x)`, oldest first, and the state after the last token (the last three inputs).
fn ref_conv(w: &[f32], state: &[f32], proj: &[f32], shape: ConvShape) -> (Vec<f64>, Vec<f32>) {
    let ConvShape { m, channels, ld } = shape;
    let mut out = vec![0.0; m * channels];
    let mut last = vec![0.0f32; 3 * channels];
    for c in 0..channels {
        let mut win = [state[c], state[channels + c], state[2 * channels + c], 0.0];
        for t in 0..m {
            win[3] = proj[t * ld + c];
            let v: f64 = (0..4)
                .map(|k| f64::from(w[c * 4 + k]) * f64::from(win[k]))
                .sum();
            out[t * channels + c] = testkit::silu_f64(v);
            win = [win[1], win[2], win[3], 0.0];
        }
        for k in 0..3 {
            last[k * channels + c] = win[k];
        }
    }
    (out, last)
}

/// `max|d| / max(max|ref|, 1)` of one vector.
fn e_form(dev: &[f32], reference: &[f64]) -> f64 {
    let scale = reference.iter().fold(1.0f64, |a, r| a.max(r.abs()));
    let worst = dev
        .iter()
        .zip(reference)
        .map(|(d, r)| (f64::from(*d) - r).abs())
        .fold(
            0.0f64,
            |a, d| if d.is_nan() { f64::INFINITY } else { a.max(d) },
        );
    worst / scale
}

/// Worst `e_form` over the `m` rows of `out`.
fn worst_rows(out: &[f32], reference: &[f64], channels: usize) -> f64 {
    out.chunks(channels)
        .zip(reference.chunks(channels))
        .map(|(d, r)| e_form(d, r))
        .fold(0.0, f64::max)
}

/// One launch on explicit ranges, drained.
fn run_ranges(rig: &Rig, shape: ConvShape, ranges: [DeviceRange<'_>; 4]) -> Result<(), EngError> {
    let [proj, weights, state, out] = ranges;
    let call = ConvCall {
        shape,
        proj,
        weights,
        state,
        out,
    };
    let result = rig.ctx.gdn_conv(&rig.stream, call);
    rig.stream.synchronize().expect("drain");
    result
}

/// One launch on whole `Padded` buffers; returns `out`. `proj` is checked unchanged.
fn run_padded(rig: &Rig, shape: ConvShape, bufs: [&Padded; 4], proj_host: &[f32]) -> Vec<f32> {
    let [proj, w, state, out] = bufs;
    let ranges = [proj.all(), w.all(), state.all(), out.all()];
    run_ranges(rig, shape, ranges).expect("a valid call");
    let proj_back = proj.read(&rig.ctx, "proj");
    assert_eq!(bits(&proj_back), bits(proj_host), "proj changed");
    w.read(&rig.ctx, "weights");
    out.read(&rig.ctx, "out")
}

struct Case {
    shape: ConvShape,
    w: Vec<f32>,
    state: Vec<f32>,
    proj: Vec<f32>,
}

fn case(rng: &mut Rng, (channels, ld, m): (usize, usize, usize), sd: f64) -> Case {
    let shape = ConvShape { m, channels, ld };
    Case {
        shape,
        w: draw(rng, channels * 4, 0.5),
        state: draw(rng, channels * 3, 1.0),
        proj: proj_rows(rng, shape, sd),
    }
}

/// Runs `c` once from its start state; returns `(out, state)`.
fn run_case(rig: &Rig, c: &Case) -> (Vec<f32>, Vec<f32>) {
    let ctx = &rig.ctx;
    let (proj, w) = (
        Padded::new(ctx, &[&c.proj], 0),
        Padded::new(ctx, &[&c.w], 0),
    );
    let state = Padded::new(ctx, &[&c.state], 0);
    let out = Padded::filled(ctx, c.shape.m * c.shape.channels);
    let out_host = run_padded(rig, c.shape, [&proj, &w, &state, &out], &c.proj);
    (out_host, state.read(ctx, "state"))
}

#[test]
fn conv_matches_the_f64_reference() {
    if !testkit::device_present("conv_matches_the_f64_reference") {
        return;
    }
    let rig = Rig::new();
    let shapes = [
        (8192, 8192, 1),
        (8192, 8192, 16),
        (8192, 12288, 3),
        (384, 400, 69),
        (1, 1, 1),
        (255, 255, 5),
        (257, 300, 2),
        (4, 4, 7),
        (65536, 65536, 2),
    ];
    let mut rng = Rng::new(0x7a01_0001);
    for dims in shapes {
        let mut worst = 0.0f64;
        for sd in [1.0, 4.0, 12.0, 4.0] {
            let c = case(&mut rng, dims, sd);
            let (out, state) = run_case(&rig, &c);
            let (reference, last) = ref_conv(&c.w, &c.state, &c.proj, c.shape);
            let error = worst_rows(&out, &reference, c.shape.channels);
            assert!(error <= E1, "{dims:?} sd {sd}: {error:e} is above {E1:e}");
            assert_eq!(bits(&state), bits(&last), "{dims:?} sd {sd}: the state");
            worst = worst.max(error);
        }
        println!("shape (channels, ld, m) {dims:?}: max error {worst:e}");
    }
}

fn oracle_dirs() -> Option<(PathBuf, PathBuf)> {
    let oracle = std::env::var_os("QWEN36_ORACLE_DIR")?;
    let fixtures = std::env::var_os("RD7A_FIXTURES")?;
    Some((oracle.into(), fixtures.into()))
}

fn masked(layer: usize, decode_row: usize) -> bool {
    MASKED
        .iter()
        .any(|&(l, first, last)| l == layer && (first..=last).contains(&decode_row))
}

/// The dump rows of `name` for `phase`: `(rows of width)`.
fn dump(dir: &Path, phase: &str, name: &str, layer: usize, width: usize) -> Vec<Vec<f32>> {
    let path = dir
        .join("cpu/dump/P4/dumps")
        .join(phase)
        .join(format!("{name}_conv_predelta-{layer}.bin"));
    let all = testkit::read_f32_file(&path).expect("dump file");
    all.chunks(width).map(<[f32]>::to_vec).collect()
}

/// E-form error of one conv output row (q 0..128, k 128..256, v 256..384) against the dump rows.
fn row_error(row: &[f64], q: &[f32], k: &[f32], v: &[f32]) -> f64 {
    let norm = |part: &[f64]| {
        let sum: f64 = part.iter().map(|x| x * x).sum();
        let scale = 1.0 / (sum + L2_EPS).sqrt();
        part.iter().map(|x| x * scale).collect::<Vec<f64>>()
    };
    let against = |mine: &[f64], expected: &[f32]| {
        let reference: Vec<f64> = expected.iter().map(|x| f64::from(*x)).collect();
        let dev: Vec<f32> = mine.iter().map(|x| *x as f32).collect();
        e_form(&dev, &reference)
    };
    against(&norm(&row[0..128]), &q[0..128])
        .max(against(&norm(&row[128..256]), &k[384..512]))
        .max(against(&row[256..384], &v[640..768]))
}

struct Dumps {
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

fn load_dumps(dir: &Path, phase: &str, layer: usize) -> Dumps {
    Dumps {
        q: dump(dir, phase, "q", layer, 2048),
        k: dump(dir, phase, "k", layer, 2048),
        v: dump(dir, phase, "v", layer, 4096),
    }
}

/// The kernel row widened to f64, then normalized and compared as `row_error` does.
fn device_row_error(out_row: &[f32], d: &Dumps, row: usize) -> f64 {
    let wide: Vec<f64> = out_row.iter().map(|x| f64::from(*x)).collect();
    row_error(&wide, &d.q[row], &d.k[row], &d.v[row])
}

/// Run explicitly with `--ignored` and `QWEN36_ORACLE_DIR` and `RD7A_FIXTURES` set: the
/// standard suite needs no oracle data (`INFERENCE_CUDA_REQUIRE_DEVICE` governs the device only).
#[test]
#[ignore = "needs QWEN36_ORACLE_DIR and RD7A_FIXTURES"]
fn conv_matches_the_rd_m6_dump() {
    if !testkit::device_present("conv_matches_the_rd_m6_dump") {
        return;
    }
    let (oracle, fixtures) = oracle_dirs()
        .expect("QWEN36_ORACLE_DIR and RD7A_FIXTURES must be set for the ignored oracle test");
    let rig = Rig::new();
    for layer in [0usize, 17, 38] {
        check_layer(&rig, &oracle, &fixtures, layer);
    }
}

fn check_layer(rig: &Rig, oracle: &Path, fixtures: &Path, layer: usize) {
    const CH: usize = 384;
    let fixture =
        testkit::read_f32_file(&fixtures.join(format!("rd7a_L{layer}_P4.bin"))).expect("fixture");
    let (w, inputs) = fixture.split_at(CH * 4);
    let (pre, dec) = (
        load_dumps(oracle, "prefill", layer),
        load_dumps(oracle, "decode", layer),
    );
    assert_eq!(
        (pre.q.len(), dec.q.len()),
        (69, 127),
        "layer {layer}: dump rows"
    );
    assert_eq!(inputs.len(), 196 * CH, "layer {layer}: fixture rows");
    let ctx = &rig.ctx;
    let zero = vec![0.0f32; 3 * CH];
    let (wbuf, state) = (Padded::new(ctx, &[w], 0), Padded::new(ctx, &[&zero], 0));
    // The reference alone, in one pass, against the same dumps.
    let whole = ConvShape {
        m: 196,
        channels: CH,
        ld: CH,
    };
    let (reference, _) = ref_conv(w, &zero, inputs, whole);
    let (mut dev_max, mut ref_max) = ([0.0f64; 2], [0.0f64; 2]);
    let mut launch = |from: usize, rows: usize, phase: usize| {
        let shape = ConvShape {
            m: rows,
            channels: CH,
            ld: CH,
        };
        let slice = &inputs[from * CH..(from + rows) * CH];
        let (proj, out) = (
            Padded::new(ctx, &[slice], 0),
            Padded::filled(ctx, rows * CH),
        );
        let got = run_padded(rig, shape, [&proj, &wbuf, &state, &out], slice);
        for r in 0..rows {
            let step = from + r;
            let (d, row) = if phase == 0 {
                (&pre, r)
            } else {
                (&dec, step - 69)
            };
            let raw = e_form(
                &got[r * CH..(r + 1) * CH],
                &reference[step * CH..(step + 1) * CH],
            );
            assert!(
                raw <= E1,
                "layer {layer} step {step}: kernel vs f64 reference {raw:e} is above {E1:e}"
            );
            if phase == 1 && masked(layer, row) {
                continue;
            }
            let dev = device_row_error(&got[r * CH..(r + 1) * CH], d, row);
            let host = row_error(
                &reference[step * CH..(step + 1) * CH],
                &d.q[row],
                &d.k[row],
                &d.v[row],
            );
            assert!(
                dev <= E1,
                "layer {layer} step {step}: kernel {dev:e} is above {E1:e}"
            );
            assert!(
                host <= E1,
                "layer {layer} step {step}: reference {host:e} is above {E1:e}"
            );
            dev_max[phase] = dev_max[phase].max(dev);
            ref_max[phase] = ref_max[phase].max(host);
        }
    };
    launch(0, 69, 0);
    for step in 69..196 {
        launch(step, 1, 1);
    }
    println!(
        "layer {layer}: kernel vs dump prefill {:e} decode {:e}; f64 reference vs dump prefill {:e} decode {:e}",
        dev_max[0], dev_max[1], ref_max[0], ref_max[1]
    );
}

#[test]
fn conv_state_carries_across_calls() {
    if !testkit::device_present("conv_state_carries_across_calls") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x7a03_0003);
    let whole = case(&mut rng, (512, 600, 16), 4.0);
    let ConvShape { channels, ld, .. } = whole.shape;
    let run_split = |splits: &[usize]| -> (Vec<f32>, Vec<f32>) {
        let ctx = &rig.ctx;
        let (w, state) = (
            Padded::new(ctx, &[&whole.w], 0),
            Padded::new(ctx, &[&whole.state], 0),
        );
        let mut out = Vec::new();
        let mut from = 0;
        for &rows in splits {
            let shape = ConvShape {
                m: rows,
                channels,
                ld,
            };
            let slice = &whole.proj[from * ld..(from + rows) * ld];
            let (proj, o) = (
                Padded::new(ctx, &[slice], 0),
                Padded::filled(ctx, rows * channels),
            );
            out.extend(run_padded(&rig, shape, [&proj, &w, &state, &o], slice));
            from += rows;
        }
        (out, state.read(ctx, "state"))
    };
    let (one_out, one_state) = run_split(&[16]);
    for splits in [vec![5, 11], vec![1; 16]] {
        let (out, state) = run_split(&splits);
        assert_eq!(bits(&out), bits(&one_out), "{splits:?}: out");
        assert_eq!(bits(&state), bits(&one_state), "{splits:?}: state");
    }
}

#[test]
fn conv_is_deterministic() {
    if !testkit::device_present("conv_is_deterministic") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let mut rng = Rng::new(0x7a04_0004);
    let c = case(&mut rng, (384, 400, 9), 4.0);
    let (proj, w) = (
        Padded::new(ctx, &[&c.proj], 0),
        Padded::new(ctx, &[&c.w], 0),
    );
    // One out buffer for both runs: the second overwrites the first. It starts finite (a NaN
    // start would hide a kernel whose output depends on the previous contents).
    let out = Padded::new(ctx, &[&vec![1.0f32; c.shape.m * c.shape.channels]], 0);
    let mut runs = Vec::new();
    for _ in 0..2 {
        let state = Padded::new(ctx, &[&c.state], 0);
        let got = run_padded(&rig, c.shape, [&proj, &w, &state, &out], &c.proj);
        runs.push((bits(&got), bits(&state.read(ctx, "state"))));
    }
    assert!(runs[0] == runs[1], "two runs from one start state differ");
}

#[test]
fn conv_bytes_pins_the_size_rules() {
    let bytes = conv_bytes(ConvShape {
        m: 1,
        channels: 8192,
        ld: 8192,
    })
    .expect("sizes");
    assert_eq!(
        (bytes.proj, bytes.weights, bytes.state, bytes.out),
        (32_768, 131_072, 98_304, 32_768)
    );
    let wide = conv_bytes(ConvShape {
        m: 3,
        channels: 8192,
        ld: 12288,
    })
    .expect("sizes");
    assert_eq!(wide.proj, 131_072);
    let huge = ConvShape {
        m: usize::MAX,
        channels: 8192,
        ld: usize::MAX,
    };
    assert!(conv_bytes(huge).is_err(), "an overflowing shape");
    assert!(
        conv_bytes(ConvShape {
            m: 0,
            channels: 1,
            ld: 1
        })
        .is_err(),
        "m 0"
    );
}

const PROJ: usize = 0;
const W: usize = 1;
const STATE: usize = 2;
const OUT: usize = 3;

/// A refusal case: the geometry, the byte size and the extra offset of each range, and what to
/// share or take from the other context. Every case is one change from `Plan::base()`.
struct Plan {
    shape: ConvShape,
    bytes: [usize; 4],
    extra: [usize; 4],
    alias: Option<(usize, usize)>,
    foreign_stream: bool,
    foreign: Option<usize>,
}

impl Plan {
    /// A valid call with `m >= 2` and `ld > channels`, every range exactly sized.
    fn base() -> Self {
        let mut plan = Self {
            shape: ConvShape {
                m: 3,
                channels: 256,
                ld: 300,
            },
            bytes: [0; 4],
            extra: [0; 4],
            alias: None,
            foreign_stream: false,
            foreign: None,
        };
        plan.fit();
        plan
    }

    /// Sizes the ranges for the geometry by plain arithmetic, written here on purpose.
    fn fit(&mut self) {
        let ConvShape { m, channels, ld } = self.shape;
        self.bytes = [
            ((m - 1) * ld + channels) * 4,
            channels * 16,
            channels * 12,
            m * channels * 4,
        ];
    }
}

type Edit = fn(&mut Plan);
type Row = (&'static str, &'static str, Edit);

fn refusals() -> Vec<Row> {
    vec![
        ("m 0", "conv launch takes 1 to", |p| p.shape.m = 0),
        ("m 32769", "conv launch takes 1 to", |p| p.shape.m = 32_769),
        ("channels 0", "conv launch takes 1 to 65536 channels", |p| {
            p.shape.channels = 0
        }),
        (
            "channels 65537",
            "conv launch takes 1 to 65536 channels",
            |p| {
                p.shape.channels = 65_537;
                p.shape.ld = 65_537;
            },
        ),
        ("ld one below channels", "conv launch takes ld", |p| {
            p.shape.ld = p.shape.channels - 1
        }),
        ("ld 1048577", "conv launch takes ld", |p| {
            p.shape.ld = 1_048_577
        }),
        ("proj one float short", "the CUDA proj range holds", |p| {
            p.bytes[PROJ] -= 4
        }),
        (
            "weights one float short",
            "the CUDA weights range holds",
            |p| p.bytes[W] -= 4,
        ),
        ("state one float short", "the CUDA state range holds", |p| {
            p.bytes[STATE] -= 4
        }),
        ("out one float short", "the CUDA out range holds", |p| {
            p.bytes[OUT] -= 4
        }),
        ("proj +2", "the CUDA proj range starts at offset", |p| {
            p.extra[PROJ] = 2
        }),
        ("state +2", "the CUDA state range starts at offset", |p| {
            p.extra[STATE] = 2
        }),
        ("out +2", "the CUDA out range starts at offset", |p| {
            p.extra[OUT] = 2
        }),
        (
            "weights +4",
            "the CUDA weights range starts at offset",
            |p| p.extra[W] = 4,
        ),
        (
            "weights +8",
            "the CUDA weights range starts at offset",
            |p| p.extra[W] = 8,
        ),
        (
            "state and out overlap",
            "the CUDA state and out ranges overlap",
            |p| p.alias = Some((STATE, OUT)),
        ),
        (
            "state and proj overlap",
            "the CUDA state and proj ranges overlap",
            |p| p.alias = Some((STATE, PROJ)),
        ),
        (
            "state and weights overlap",
            "the CUDA state and weights ranges overlap",
            |p| p.alias = Some((STATE, W)),
        ),
        (
            "out and proj overlap",
            "the CUDA out and proj ranges overlap",
            |p| p.alias = Some((OUT, PROJ)),
        ),
        (
            "out and weights overlap",
            "the CUDA out and weights ranges overlap",
            |p| p.alias = Some((OUT, W)),
        ),
        ("stream of another context", "stream belongs", |p| {
            p.foreign_stream = true
        }),
        (
            "buffer of another context",
            "belongs to another adapter",
            |p| p.foreign = Some(PROJ),
        ),
    ]
}

/// Runs a plan on zero-filled buffers; returns the result and the hash of every buffer before
/// and after.
/// What one launch attempt gives back: the call's result, the result of draining the stream
/// (asserted by the caller after the row's own checks, so a mutant dies on its own assertion), and
/// the hash of every buffer before and after.
struct Attempt {
    result: Result<(), EngError>,
    drain: Result<(), EngError>,
    before: Vec<u64>,
    after: Result<Vec<u64>, EngError>,
}

/// Runs a plan on buffers filled with 0xFF (every byte a launch writes or leaves changes the
/// hash).
fn attempt(rig: &Rig, other: &Rig, plan: &Plan) -> Attempt {
    let ctx = &rig.ctx;
    let shared = plan
        .alias
        .map(|(a, b)| testkit::upload_filled(ctx, 0xFF, own_len(plan, a).max(own_len(plan, b))));
    let own: Vec<DeviceBuffer> = (0..4)
        .map(|i| {
            testkit::upload_filled(ctx, 0xFF, PAD * 4 + plan.extra[i] + plan.bytes[i] + PAD * 4)
        })
        .collect();
    let foreign = plan
        .foreign
        .map(|i| testkit::upload_filled(&other.ctx, 0xFF, own_len(plan, i)));
    let buffer = |i: usize| -> &DeviceBuffer {
        match (&shared, plan.alias, &foreign, plan.foreign) {
            (Some(s), Some((a, b)), _, _) if i == a || i == b => s,
            (_, _, Some(f), Some(j)) if i == j => f,
            _ => &own[i],
        }
    };
    let range = |i: usize| {
        buffer(i)
            .range(PAD * 4 + plan.extra[i], plan.bytes[i])
            .expect("range")
    };
    let hash = || -> Result<Vec<u64>, EngError> {
        let mut seen: Vec<&DeviceBuffer> = Vec::new();
        let mut hashes = Vec::new();
        for i in 0..4 {
            let b = buffer(i);
            if seen.iter().any(|s| std::ptr::eq(*s, b)) {
                continue;
            }
            seen.push(b);
            let len = if plan.foreign == Some(i) {
                own_len(plan, i)
            } else {
                own_len_of(plan, i, plan.alias)
            };
            let owner = if plan.foreign == Some(i) {
                &other.ctx
            } else {
                ctx
            };
            let mut bytes = vec![0u8; len];
            owner.download(b, &mut bytes)?;
            hashes.push(testkit::fnv1a64(&bytes));
        }
        Ok(hashes)
    };
    let before = hash().expect("hash before the call");
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let call = ConvCall {
        shape: plan.shape,
        proj: range(PROJ),
        weights: range(W),
        state: range(STATE),
        out: range(OUT),
    };
    let result = ctx.gdn_conv(stream, call);
    let drain = stream.synchronize();
    Attempt {
        result,
        drain,
        before,
        after: hash(),
    }
}

fn own_len(plan: &Plan, i: usize) -> usize {
    PAD * 4 + plan.extra[i] + plan.bytes[i] + PAD * 4
}

/// Allocation length of buffer `i`, which is the shared one for an aliased slot.
fn own_len_of(plan: &Plan, i: usize, alias: Option<(usize, usize)>) -> usize {
    match alias {
        Some((a, b)) if i == a || i == b => own_len(plan, a).max(own_len(plan, b)),
        _ => own_len(plan, i),
    }
}

#[test]
fn conv_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("conv_rejects_bad_calls_before_any_launch") {
        return;
    }
    let (rig, other) = (Rig::new(), Rig::new());
    let first = attempt(&rig, &other, &Plan::base());
    first.result.expect("the base call is valid");
    first.drain.expect("drain after the base call");
    for (label, needle, edit) in refusals() {
        let mut plan = Plan::base();
        edit(&mut plan);
        let run = attempt(&rig, &other, &plan);
        let message = match run.result {
            Err(EngError::BadRequest(message)) => message,
            other => panic!("{label}: expected a BadRequest, got {other:?}"),
        };
        assert!(
            message.contains(needle),
            "{label}: expected {needle:?}, got {message:?}"
        );
        let after = run
            .after
            .unwrap_or_else(|e| panic!("{label}: reading the buffers back: {e:?}"));
        assert_eq!(run.before, after, "{label}: a buffer changed");
        run.drain
            .unwrap_or_else(|e| panic!("{label}: drain after the refusal: {e:?}"));
        let again = attempt(&rig, &other, &Plan::base());
        again
            .result
            .unwrap_or_else(|e| panic!("{label}: the backend is not usable afterwards: {e:?}"));
    }
}

/// Calls at the exact boundary of every rule: all accepted and equal to the reference.
#[test]
fn conv_accepts_the_exact_sizes() {
    if !testkit::device_present("conv_accepts_the_exact_sizes") {
        return;
    }
    let rig = Rig::new();
    let ctx = &rig.ctx;
    let mut rng = Rng::new(0x7a05_0005);
    let c = case(&mut rng, (64, 80, 5), 4.0);
    let check = |label: &str, out: &[f32], state: &[f32], c: &Case| {
        let (reference, last) = ref_conv(&c.w, &c.state, &c.proj, c.shape);
        let error = worst_rows(out, &reference, c.shape.channels);
        assert!(error <= E1, "{label}: {error:e}");
        assert_eq!(bits(state), bits(&last), "{label}: state");
    };
    // Exact sizes (ld > channels, m > 1) and proj, state and out at byte offset 4.
    let (out, state) = run_case(&rig, &c);
    check("exact sizes", &out, &state, &c);
    let (proj, w) = (
        Padded::new(ctx, &[&c.proj], 4),
        Padded::new(ctx, &[&c.w], 0),
    );
    let (st, o) = (
        Padded::new(ctx, &[&c.state], 4),
        Padded::new(ctx, &[&vec![f32::from_bits(SENT); 5 * 64]], 4),
    );
    let got = run_padded(&rig, c.shape, [&proj, &w, &st, &o], &c.proj);
    check("offset 4", &got, &st.read(ctx, "state"), &c);
    // State directly after the last float proj reads.
    let reach = (c.shape.m - 1) * c.shape.ld + c.shape.channels;
    let shared = Padded::new(ctx, &[&c.proj[..reach], &c.state], 0);
    let (w2, o2) = (Padded::new(ctx, &[&c.w], 0), Padded::filled(ctx, 5 * 64));
    let ranges = [
        shared.range(0, reach * 4),
        w2.all(),
        shared.range(reach * 4, 3 * 64 * 4),
        o2.all(),
    ];
    run_ranges(&rig, c.shape, ranges).expect("state after the proj reach");
    let both = shared.read(ctx, "shared");
    check("state after proj", &o2.read(ctx, "out"), &both[reach..], &c);
    // proj and weights share bytes: ld == channels, the taps are the first 256 floats of proj.
    let flat = ConvShape {
        m: 5,
        channels: 64,
        ld: 64,
    };
    let mut shared_c = case(&mut rng, (64, 64, 5), 1.0);
    shared_c.w = shared_c.proj[..256].to_vec();
    let buf = Padded::new(ctx, &[&shared_c.proj], 0);
    let (st3, o3) = (
        Padded::new(ctx, &[&shared_c.state], 0),
        Padded::filled(ctx, 5 * 64),
    );
    run_ranges(
        &rig,
        flat,
        [buf.all(), buf.range(0, 256 * 4), st3.all(), o3.all()],
    )
    .expect("proj and weights overlap");
    check(
        "proj and weights overlap",
        &o3.read(ctx, "out"),
        &st3.read(ctx, "state"),
        &shared_c,
    );
}
