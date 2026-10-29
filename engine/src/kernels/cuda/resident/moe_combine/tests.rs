//! Device tests (3090) of the MoE combine wrapper at seam 1, against `emu` (the sequence in plain
//! f32, one operation per statement, which Rust never contracts) and `ref_f64` (the same f32
//! inputs in f64). Inputs are shaped like the router's output, built here with `testkit::Rng`:
//! no oracle data, no environment variable, no `#[ignore]`; the chain router, `matmul_expert`,
//! combine is tested with the MoE block of the decode step. Bound: class E3 (final), per token
//! row, never widened: `max |got - ref| <= 2^-17 * max(max |ref|, 1)`. Bit equality with `emu`
//! is extra. Run with `--test-threads=1`.

use crate::EngError;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream};
use crate::kernels::cuda::resident::moe_combine::{
    CombineCall, CombineOut, CombineShape, SharedTerm, check_shape, combine_down_bytes,
    combine_row_bytes,
};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CapturePlan, DeviceBuffer};

/// The E3 bound, `2^-17`.
const E3: f64 = 1.0 / 131_072.0;
const POISON: u8 = 0x55;

#[rustfmt::skip]
fn shape(m: usize, n: usize, top_k: usize, ld_down: usize) -> CombineShape { CombineShape { m, n, top_k, ld_down } }

fn dims(s: &CombineShape) -> (usize, usize, usize, usize) {
    (s.m, s.n, s.top_k, s.ld_down)
}

/// Host copy of one call: `down` rows are `ld_down` floats (NaN after `n`); `shared` is (gate, row).
struct Inputs {
    shape: CombineShape,
    res: Vec<f32>,
    down: Vec<f32>,
    weights: Vec<f32>,
    shared: Option<(Vec<f32>, Vec<f32>)>,
}

impl Inputs {
    /// The (factor, value) pairs of token `t`, column `j` in summation order: picks, then shared.
    fn terms(&self, t: usize, j: usize) -> Vec<(f32, f32)> {
        let (_, n, top_k, ld_down) = dims(&self.shape);
        let pick = |p: usize| (t * top_k + p, (t * top_k + p) * ld_down + j);
        let mut terms: Vec<_> = (0..top_k)
            .map(|p| (self.weights[pick(p).0], self.down[pick(p).1]))
            .collect();
        if let Some((gate, row)) = &self.shared {
            terms.push((gate[t], row[t * n + j]));
        }
        terms
    }

    /// The pinned graph's sequence: the first product starts the sum (`-0` stays `-0`), the
    /// residual comes last.
    fn emu(&self, t: usize, j: usize) -> f32 {
        let terms = self.terms(t, j);
        let mut sum = terms[0].0 * terms[0].1;
        for &(factor, value) in &terms[1..] {
            let product = factor * value;
            sum += product;
        }
        self.res[t * self.shape.n + j] + sum
    }

    fn ref_f64(&self, t: usize, j: usize) -> f64 {
        f64::from(self.res[t * self.shape.n + j]) + dot64(&self.terms(t, j))
    }
}

/// The products of `terms` summed in f64.
fn dot64(terms: &[(f32, f32)]) -> f64 {
    terms
        .iter()
        .map(|&(f, v)| f64::from(f) * f64::from(v))
        .sum()
}

fn gaussian(rng: &mut Rng, len: usize, sigma: f64) -> Vec<f32> {
    (0..len).map(|_| (rng.normal() * sigma) as f32).collect()
}

/// Router-shaped weights: softmax of `top_k` normals, descending.
fn softmax_weights(rng: &mut Rng, top_k: usize) -> Vec<f32> {
    let mut logits: Vec<f64> = (0..top_k).map(|_| rng.normal()).collect();
    logits.sort_by(|a, b| b.total_cmp(a));
    let exps: Vec<f64> = logits.iter().map(|l| (l - logits[0]).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| (e / total) as f32).collect()
}

/// Random inputs: residual rms `sigma.0`, down and shared rms `sigma.1`. Every third repetition
/// zeroes one weight per token.
fn random_inputs(
    rng: &mut Rng,
    s: CombineShape,
    shared: bool,
    sigma: (f64, f64),
    rep: usize,
) -> Inputs {
    let (m, n, top_k, ld_down) = dims(&s);
    let mut down = vec![f32::NAN; m * top_k * ld_down];
    for row in 0..m * top_k {
        down[row * ld_down..row * ld_down + n].copy_from_slice(&gaussian(rng, n, sigma.1));
    }
    let mut weights: Vec<f32> = (0..m).flat_map(|_| softmax_weights(rng, top_k)).collect();
    for t in (0..m).filter(|_| rep % 3 == 2) {
        weights[t * top_k + rng.below(top_k as u64) as usize] = 0.0;
    }
    let shared = shared.then(|| {
        let gate = (0..m).map(|_| (1.0 / (1.0 + (-rng.normal()).exp())) as f32);
        (gate.collect(), gaussian(rng, m * n, sigma.1))
    });
    let res = gaussian(rng, m * n, sigma.0);
    Inputs {
        shape: s,
        res,
        down,
        weights,
        shared,
    }
}

/// Makes the last pick undo the sum of the others: the partial sums far exceed the result.
fn cancel_last_pick(inp: &mut Inputs) {
    let (m, n, top_k, ld_down) = dims(&inp.shape);
    for t in (0..m).filter(|t| top_k > 1 && inp.weights[t * top_k + top_k - 1] > 0.0) {
        for j in 0..n {
            let terms = inp.terms(t, j);
            let last = f64::from(terms[top_k - 1].0);
            inp.down[(t * top_k + top_k - 1) * ld_down + j] =
                (-dot64(&terms[..top_k - 1]) / last) as f32;
        }
    }
}

/// The context and stream of a test, or `None` on a host with no device.
fn rig(test: &str) -> Option<(CudaContext, DeviceStream)> {
    testkit::device_present(test).then(|| {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        (ctx, stream)
    })
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Runs one combine (in place or to a poisoned output); the inputs must read back unchanged.
fn run(ctx: &CudaContext, stream: &DeviceStream, inp: &Inputs, in_place: bool) -> Vec<f32> {
    let rows = combine_row_bytes(&inp.shape).expect("row bytes");
    let up = |values: &[f32]| testkit::upload_f32(ctx, values);
    let (res, down, weights) = (up(&inp.res), up(&inp.down), up(&inp.weights));
    let shared = inp.shared.as_ref().map(|(gate, row)| (up(gate), up(row)));
    let out = testkit::upload_filled(ctx, POISON, rows);
    let call = CombineCall {
        shape: inp.shape,
        res: res.range(0, rows).expect("res"),
        down: down.range(0, 4 * inp.down.len()).expect("down"),
        weights: weights.range(0, 4 * inp.weights.len()).expect("weights"),
        shared: shared.as_ref().map(|(gate, row)| SharedTerm {
            gate: gate.range(0, 4 * inp.shape.m).expect("gate"),
            row: row.range(0, rows).expect("row"),
        }),
        out: if in_place {
            CombineOut::InPlace
        } else {
            CombineOut::To(out.range(0, rows).expect("out"))
        },
    };
    ctx.moe_combine(stream, call).expect("combine");
    stream.synchronize().expect("synchronize");
    let read = |buffer, len| testkit::read_bytes(ctx, buffer, len);
    for (buffer, values) in [(&down, &inp.down), (&weights, &inp.weights)] {
        let after = read(buffer, 4 * values.len());
        assert!(after == bytes_of(values), "an input changed");
    }
    testkit::floats(&read(if in_place { &res } else { &out }, rows))
}

/// Asserts E3 per token row and bit equality with `emu`; returns the largest difference and
/// the largest difference over its bound.
fn check(label: &str, inp: &Inputs, got: &[f32]) -> (f64, f64) {
    let (n, mut max) = (inp.shape.n, (0f64, 0f64));
    for t in 0..inp.shape.m {
        let (mut worst, mut peak) = (0f64, 1f64);
        for j in 0..n {
            let want = inp.ref_f64(t, j);
            worst = worst.max((f64::from(got[t * n + j]) - want).abs());
            peak = peak.max(want.abs());
            let exact = inp.emu(t, j);
            assert!(
                got[t * n + j].to_bits() == exact.to_bits(),
                "{label}: ({t},{j}) vs emu"
            );
        }
        let bound = E3 * peak;
        assert!(
            worst <= bound,
            "{label} row {t}: E3 {worst:e} above {bound:e}"
        );
        max = (max.0.max(worst), max.1.max(worst / bound));
    }
    max
}

/// `(m, n, top_k, ld_down, shared, in_place)`.
#[rustfmt::skip]
const CONFIGS: [(usize, usize, usize, usize, bool, bool); 12] = [
    (1, 2048, 8, 2048, true, false), (1, 2048, 8, 2048, true, true), (1, 2048, 8, 2112, true, false),
    (1, 2048, 8, 2048, false, false), (3, 2048, 8, 2048, true, false), (16, 2048, 8, 2048, true, true),
    (1, 2048, 1, 2048, true, false), (1, 2048, 32, 2048, true, false), (2, 4, 8, 4, true, false),
    (5, 260, 3, 264, false, false), (1, 65536, 8, 65536, true, false), (64, 2304, 8, 2304, false, false),
];

/// (residual rms, down rms): four production pairs, four cancelling stress pairs.
#[rustfmt::skip]
const SCALES: [(f64, f64); 8] = [
    (0.1, 0.1), (0.3, 0.3), (1.0, 0.3), (0.05, 0.5), (1.0, 1.0), (4.0, 4.0), (8.0, 8.0), (30.0, 0.05),
];

/// Bit equality with `emu` (and, in the harness, with `l_out`) holds because the order is the
/// pinned graph's: pick order, shared last, residual last. A failure is a defect report, never a
/// reason to widen E3.
#[test]
fn combine_matches_the_host_sequence() {
    let Some((ctx, stream)) = rig("combine_matches_the_host_sequence") else {
        return;
    };
    let mut rng = Rng::new(0x4e01);
    let mut maxima = [(0f64, 0f64); 2];
    for (m, n, top_k, ld_down, with_shared, in_place) in CONFIGS {
        for (profile, sigma) in SCALES.into_iter().enumerate() {
            for rep in 0..8 {
                let s = shape(m, n, top_k, ld_down);
                let mut inp = random_inputs(&mut rng, s, with_shared, sigma, rep);
                if profile >= 4 {
                    cancel_last_pick(&mut inp);
                }
                let got = run(&ctx, &stream, &inp, in_place);
                let label = format!("{m}x{n} k{top_k} ld{ld_down} {sigma:?} rep {rep}");
                let (worst, ratio) = check(&label, &inp, &got);
                let kind = &mut maxima[profile / 4];
                *kind = (kind.0.max(worst), kind.1.max(ratio));
            }
        }
    }
    for (name, (worst, ratio)) in ["production", "stress"].into_iter().zip(maxima) {
        println!("combine T1 {name}: E3 max abs {worst:e}, bound {E3:e}, max d/bound {ratio:.3e}");
    }
    // `res`, every pick and the shared row are `-0`: the output stays `-0`, as in ggml.
    let zero = Inputs {
        shape: shape(2, 8, 4, 8),
        res: vec![-0.0; 16],
        down: vec![-0.0; 64],
        weights: vec![0.25; 8],
        shared: Some((vec![0.5; 2], vec![-0.0; 16])),
    };
    let got = run(&ctx, &stream, &zero, false);
    assert!(got.iter().all(|v| v.to_bits() == 0x8000_0000));
}

/// One token of 4 identical columns: `picks` at weight 1, `res` 0, shared term `0.5 * 1.0`.
fn one_token(ctx: &CudaContext, stream: &DeviceStream, picks: &[f32], shared: bool) -> Vec<f32> {
    let inp = Inputs {
        shape: shape(1, 4, picks.len(), 4),
        res: vec![0.0; 4],
        down: picks.iter().flat_map(|p| [*p; 4]).collect(),
        weights: vec![1.0; picks.len()],
        shared: shared.then(|| (vec![0.5], vec![1.0; 4])),
    };
    run(ctx, stream, &inp, false)
}

/// Hard-coded expectations, independent of `emu`.
#[test]
fn combine_sums_in_pick_order_with_the_shared_term_last() {
    let Some((ctx, stream)) = rig("combine_sums_in_pick_order_with_the_shared_term_last") else {
        return;
    };
    let (cancelling, ties) = ([1e8, 1.0, 1.0, -1e8], [16_777_216.0, 1.0, 1.0, 1.0]);
    // Pick order gives 0 (the order 0,3,2,1 gives 2); shared last gives 0.5 (after pick 0, 0).
    assert_eq!(one_token(&ctx, &stream, &cancelling, false), [0.0; 4]);
    assert_eq!(one_token(&ctx, &stream, &cancelling, true), [0.5; 4]);
    // Each 1 is lost to the tie to even: 2^24 (a pairwise tree gives +2, a full reverse +4).
    assert_eq!(one_token(&ctx, &stream, &ties, false), [16_777_216.0; 4]);
    assert_eq!(one_token(&ctx, &stream, &ties, true), [16_777_216.0; 4]);
}

#[test]
fn combine_byte_functions_pin_the_size_rules() {
    let s = shape(1, 2048, 8, 2112);
    assert_eq!(combine_row_bytes(&s).expect("row bytes"), 8_192);
    // 4 * (7 * 2112 + 2048)
    assert_eq!(combine_down_bytes(&s).expect("down bytes"), 67_328);
    assert!(combine_row_bytes(&shape(usize::MAX, 4, 1, 4)).is_err());
    assert!(combine_down_bytes(&shape(usize::MAX, 4, 2, 4)).is_err());
}

#[test]
fn combine_shape_rules_need_no_device() {
    let accepted = [
        shape(1, 2048, 8, 2048),
        shape(4096, 4, 32, 4),
        shape(1, 65536, 1, 1 << 26),
    ];
    for ok in accepted {
        check_shape(&ok).expect("accepted");
    }
    #[rustfmt::skip]
    let refused = [
        (0, 2048, 8, 2048), (4097, 2048, 8, 2048), (1, 0, 8, 2048), (1, 6, 8, 2048), (1, 65540, 8, 65540),
        (1, 2048, 0, 2048), (1, 2048, 33, 2048), (1, 2048, 8, 2044), (1, 2048, 8, 2050), (1, 2048, 8, (1 << 26) + 4),
    ];
    for (m, n, top_k, ld_down) in refused {
        let refusal = check_shape(&shape(m, n, top_k, ld_down)).expect_err("refused");
        assert!(
            format!("{refusal:?}").contains(&format!("m {m}, n {n}")),
            "{refusal:?}"
        );
    }
}

// ---------------------------------------------------------------- T2, T5, T6

/// Asserts `got` equals `emu` for every element: NaN by `is_nan()`, the rest bit for bit.
fn assert_emu(label: &str, inp: &Inputs, got: &[f32]) {
    let n = inp.shape.n;
    for (i, value) in got.iter().enumerate() {
        let want = inp.emu(i / n, i % n);
        let same = if want.is_nan() {
            value.is_nan()
        } else {
            value.to_bits() == want.to_bits()
        };
        assert!(same, "{label}: element {i} is {value:?}, emu says {want:?}");
    }
}

/// A Qwen3.6-sized call (1 token, 8 picks, the shared pair) with these weights and this gate.
fn given(rng: &mut Rng, weights: [f32; 8], gate: f32) -> Inputs {
    let mut inp = random_inputs(rng, shape(1, 2048, 8, 2048), true, (0.3, 0.3), 0);
    inp.weights = weights.to_vec();
    inp.shared.as_mut().expect("shared").0 = vec![gate];
    inp
}

/// The weights and the gate are used as given: no renormalization, clamp, sigmoid or softmax.
#[test]
fn combine_applies_weights_and_gate_as_given() {
    let Some((ctx, stream)) = rig("combine_applies_weights_and_gate_as_given") else {
        return;
    };
    let mut rng = Rng::new(0x4e02);
    let cases = [
        ("sum 4, gate 2", [0.5; 8], 2.0),
        (
            "3, 1, zeros, gate -1",
            [3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            -1.0,
        ),
    ];
    for (label, weights, gate) in cases {
        let inp = given(&mut rng, weights, gate);
        let got = run(&ctx, &stream, &inp, false);
        check(label, &inp, &got);
    }
}

/// The NaN rules of the module doc, each against `emu`, which propagates NaN like the kernel.
#[test]
fn combine_preserves_nan_from_a_faulty_router_row() {
    let Some((ctx, stream)) = rig("combine_preserves_nan_from_a_faulty_router_row") else {
        return;
    };
    let mut rng = Rng::new(0x4e05);
    // A faulty router row (weights 0, gate 0) with the NaN down rows `matmul_expert` writes.
    let mut inp = random_inputs(&mut rng, shape(3, 64, 8, 64), true, (0.3, 0.3), 0);
    for p in 8..16 {
        inp.down[p * 64..(p + 1) * 64].fill(f32::NAN);
    }
    inp.weights[8..16].fill(0.0);
    inp.shared.as_mut().expect("shared").0[1] = 0.0;
    let got = run(&ctx, &stream, &inp, false);
    assert!(
        got[64..128].iter().all(|v| v.is_nan()),
        "the faulty row is not all NaN"
    );
    assert_emu("faulty row", &inp, &got);
    // A NaN or infinite pick under weight 0, and a NaN shared value under gate 0, hit one column.
    for (label, poison) in [("NaN pick", f32::NAN), ("inf pick", f32::INFINITY)] {
        let mut inp = given(&mut rng, [0.25, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0], 0.5);
        inp.down[3 * 2048 + 7] = poison;
        let got = run(&ctx, &stream, &inp, false);
        assert_eq!(
            (0..2048).filter(|&j| got[j].is_nan()).collect::<Vec<_>>(),
            [7],
            "{label}"
        );
        assert_emu(label, &inp, &got);
    }
    let mut inp = given(&mut rng, [0.125; 8], 0.0);
    inp.shared.as_mut().expect("shared").1[9] = f32::NAN;
    let got = run(&ctx, &stream, &inp, false);
    assert!(got[9].is_nan() && got.iter().filter(|v| v.is_nan()).count() == 1);
    assert_emu("NaN shared", &inp, &got);
    // The limit: zero weights with finite down rows give `res`, as the module doc says.
    let inp = given(&mut rng, [0.0; 8], 0.0);
    let got = run(&ctx, &stream, &inp, false);
    assert!(
        got.iter()
            .zip(&inp.res)
            .all(|(g, r)| g.to_bits() == r.to_bits())
    );
}

#[test]
fn combine_is_deterministic_and_in_place_equals_out_of_place() {
    let Some((ctx, stream)) = rig("combine_is_deterministic_and_in_place_equals_out_of_place")
    else {
        return;
    };
    let mut rng = Rng::new(0x4e06);
    for (m, n, ld_down) in [(1, 2048, 2048), (16, 2048, 2112)] {
        let inp = random_inputs(&mut rng, shape(m, n, 8, ld_down), true, (0.3, 0.3), 0);
        let bits = |values: Vec<f32>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let first = bits(run(&ctx, &stream, &inp, false));
        assert!(
            first == bits(run(&ctx, &stream, &inp, false)),
            "m {m}: two launches differ"
        );
        assert!(
            first == bits(run(&ctx, &stream, &inp, true)),
            "m {m}: in place differs"
        );
    }
}

// ---------------------------------------------------------------- T4

/// A buffer, an offset and the bytes of a range.
type Part<'a> = (&'a DeviceBuffer, usize, usize);

/// The ranges of one call; `out` is `None` for the in-place case. A refusal row changes one field
/// of a valid base.
#[derive(Clone, Copy)]
struct Parts<'a> {
    shape: CombineShape,
    res: Part<'a>,
    down: Part<'a>,
    weights: Part<'a>,
    gate: Part<'a>,
    shared: Part<'a>,
    out: Option<Part<'a>>,
}

/// Two tokens of 64 columns, four picks at a stride of 96: every buffer holds what the call needs
/// plus 64 bytes, so a shifted range still fits its buffer. `pool` lets two ranges overlap.
struct Bufs {
    shape: CombineShape,
    sizes: [usize; 7],
    buffers: [DeviceBuffer; 7],
}

impl Bufs {
    fn new(ctx: &CudaContext) -> Self {
        let shape = shape(2, 64, 4, 96);
        let rows = combine_row_bytes(&shape).expect("rows");
        let sizes = [
            rows,
            combine_down_bytes(&shape).expect("down"),
            4 * 2 * 4,
            4 * 2,
            rows,
            rows,
            8192,
        ];
        let sizes = sizes.map(|bytes| bytes + 64);
        let buffers =
            std::array::from_fn(|i| testkit::upload_filled(ctx, 0x10 + i as u8, sizes[i]));
        Self {
            shape,
            sizes,
            buffers,
        }
    }

    fn base(&self) -> Parts<'_> {
        let rows = combine_row_bytes(&self.shape).expect("rows");
        let part = |i: usize, bytes: usize| (&self.buffers[i], 0, bytes);
        Parts {
            shape: self.shape,
            res: part(0, rows),
            down: part(1, combine_down_bytes(&self.shape).expect("down")),
            weights: part(2, 32),
            gate: part(3, 8),
            shared: part(4, rows),
            out: Some(part(5, rows)),
        }
    }

    /// One digest per buffer: a refused call must change none.
    fn digests(&self, ctx: &CudaContext) -> Vec<u64> {
        let all = self.buffers.iter().zip(self.sizes);
        all.map(|(b, len)| testkit::fnv1a64(&testkit::read_bytes(ctx, b, len)))
            .collect()
    }
}

fn range_of(part: Part<'_>) -> Result<DeviceRange<'_>, EngError> {
    part.0.range(part.1, part.2)
}

fn longer(part: Part<'_>) -> Part<'_> {
    (part.0, part.1, part.2 + 16)
}

fn submit(ctx: &CudaContext, stream: &DeviceStream, p: Parts<'_>) -> Result<(), EngError> {
    let range = range_of;
    let out = match p.out {
        Some(part) => CombineOut::To(range(part)?),
        None => CombineOut::InPlace,
    };
    let shared = SharedTerm {
        gate: range(p.gate)?,
        row: range(p.shared)?,
    };
    let (res, down, weights) = (range(p.res)?, range(p.down)?, range(p.weights)?);
    ctx.moe_combine(
        stream,
        CombineCall {
            shape: p.shape,
            res,
            down,
            weights,
            shared: Some(shared),
            out,
        },
    )
}

fn assert_refused(result: Result<(), EngError>, case: &str, needle: &str) {
    match result {
        Err(EngError::BadRequest(message)) => assert!(
            message.contains(needle),
            "{case}: `{message}` lacks `{needle}`"
        ),
        other => panic!("{case}: expected BadRequest with `{needle}`, got {other:?}"),
    }
}

/// The refusal rows: every limit of `check_shape`, then short, shifted and overlapping ranges.
fn refusals<'a>(b: &'a Bufs) -> Vec<(String, String, Parts<'a>)> {
    let base = b.base();
    let mut rows = Vec::new();
    #[rustfmt::skip]
    let shapes = [
        (0, 64, 4, 96), (4097, 64, 4, 96), (2, 0, 4, 96), (2, 6, 4, 96), (2, 65540, 4, 96), (2, 64, 0, 96),
        (2, 64, 33, 96), (2, 64, 4, 60), (2, 64, 4, 66), (2, 64, 4, (1 << 26) + 4),
    ];
    for (m, n, top_k, ld_down) in shapes {
        let case = format!("m {m} n {n} top_k {top_k} ld_down {ld_down}");
        rows.push((
            case,
            "the CUDA combine takes".to_owned(),
            Parts {
                shape: shape(m, n, top_k, ld_down),
                ..base
            },
        ));
    }
    let short = |part: Part<'a>| (part.0, part.1, part.2 - 4);
    let shifted = |part: Part<'a>, by: usize| (part.0, part.1 + by, part.2);
    let (out, in_place) = (base.out.expect("out"), Parts { out: None, ..base });
    let sized = [
        (
            "res",
            Parts {
                res: short(base.res),
                ..base
            },
        ),
        (
            "down",
            Parts {
                down: short(base.down),
                ..base
            },
        ),
        (
            "weights",
            Parts {
                weights: short(base.weights),
                ..base
            },
        ),
        (
            "shared_gate",
            Parts {
                gate: short(base.gate),
                ..base
            },
        ),
        (
            "shared",
            Parts {
                shared: short(base.shared),
                ..base
            },
        ),
        (
            "out",
            Parts {
                out: Some(short(out)),
                ..base
            },
        ),
    ];
    let aligned = [
        (
            "res +4",
            Parts {
                res: shifted(base.res, 4),
                ..base
            },
        ),
        (
            "down +8",
            Parts {
                down: shifted(base.down, 8),
                ..base
            },
        ),
        (
            "weights +2",
            Parts {
                weights: shifted(base.weights, 2),
                ..base
            },
        ),
        (
            "gate +2",
            Parts {
                gate: shifted(base.gate, 2),
                ..base
            },
        ),
        (
            "shared +4",
            Parts {
                shared: shifted(base.shared, 4),
                ..base
            },
        ),
        (
            "out +4",
            Parts {
                out: Some(shifted(out, 4)),
                ..base
            },
        ),
    ];
    rows.extend(sized.map(|(name, p)| (format!("{name} short"), "range holds".to_owned(), p)));
    rows.extend(aligned.map(|(name, p)| (name.to_owned(), "is not a multiple of".to_owned(), p)));
    // The written range (`out`, or `res` in place) starts on the last 16 bytes of another input.
    let pool = |bytes: usize| (&b.buffers[6], 0, bytes);
    let inputs = [
        ("res", base.res),
        ("down", base.down),
        ("weights", base.weights),
        ("shared_gate", base.gate),
        ("shared", base.shared),
    ];
    for (name, input) in inputs {
        let tail = (&b.buffers[6], base.res.2 - 16, input.2);
        let over = |p: Parts<'a>| match name {
            "res" => Parts { res: tail, ..p },
            "down" => Parts { down: tail, ..p },
            "weights" => Parts { weights: tail, ..p },
            "shared_gate" => Parts { gate: tail, ..p },
            _ => Parts { shared: tail, ..p },
        };
        let needle = format!("the CUDA out and {name} ranges overlap");
        rows.push((
            format!("out over {name}"),
            needle,
            over(Parts {
                out: Some(pool(base.res.2)),
                ..base
            }),
        ));
        if name != "res" {
            let needle = format!("the CUDA res and {name} ranges overlap");
            rows.push((
                format!("in place, {name} over res"),
                needle,
                over(Parts {
                    res: pool(base.res.2),
                    ..in_place
                }),
            ));
        }
    }
    rows
}

#[test]
fn combine_rejects_bad_calls_before_any_launch() {
    let Some((ctx, stream)) = rig("combine_rejects_bad_calls_before_any_launch") else {
        return;
    };
    let (other, bufs) = (CudaContext::new().expect("second context"), Bufs::new(&ctx));
    let before = bufs.digests(&ctx);
    for (case, needle, parts) in refusals(&bufs) {
        assert_refused(submit(&ctx, &stream, parts), &case, &needle);
    }
    let base = bufs.base();
    let foreign_stream = other.stream().expect("stream of the other context");
    assert_refused(
        submit(&ctx, &foreign_stream, base),
        "a stream of another context",
        "belongs to another adapter",
    );
    let foreign = testkit::upload_filled(&other, 0x77, 4096);
    let from_other = Parts {
        res: (&foreign, 0, base.res.2),
        ..base
    };
    assert_refused(
        submit(&ctx, &stream, from_other),
        "a buffer of another context",
        "belongs to another adapter",
    );
    assert_eq!(
        bufs.digests(&ctx),
        before,
        "a refused call changed a buffer"
    );
    // Accepted controls: in place, a longer `down` and a longer `out` (at-least checks).
    submit(&ctx, &stream, Parts { out: None, ..base }).expect("in place");
    submit(
        &ctx,
        &stream,
        Parts {
            down: longer(base.down),
            ..base
        },
    )
    .expect("a longer down");
    submit(
        &ctx,
        &stream,
        Parts {
            out: Some(longer(base.out.expect("out"))),
            ..base
        },
    )
    .expect("a longer out");
    stream
        .synchronize()
        .expect("the backend is usable after the refusals");
}

// ---------------------------------------------------------------- T6: graph capture

/// The capture plan of the capture test: the rig owns every buffer, so no pin and no hold.
fn no_pins() -> CapturePlan<'static> {
    CapturePlan {
        pins: &[],
        holds: Vec::new(),
    }
}

/// T6: one captured `moe_combine` replayed three times with the weights and the shared gate
/// rewritten between replays reads the new values every time: each replay equals the eager
/// launch on those values byte for byte and the host sequence bit for bit, on a one-node graph.
#[test]
fn combine_inside_a_capture_reads_the_new_values_between_replays() {
    let Some((ctx, stream)) = rig("combine_inside_a_capture_reads_the_new_values_between_replays")
    else {
        return;
    };
    let mut rng = Rng::new(0x4e11);
    let s = shape(1, 512, 8, 512);
    let rows = combine_row_bytes(&s).expect("row bytes");
    let base = random_inputs(&mut rng, s, true, (0.3, 0.3), 0);
    let (res, down, shared_row) = (
        base.res.clone(),
        base.down.clone(),
        base.shared.clone().expect("gated").1,
    );
    // Three weight/gate variants over the one fixed problem.
    let variants: Vec<(Vec<f32>, Vec<f32>)> = (0..3)
        .map(|rep| {
            let v = random_inputs(&mut rng, s, true, (0.3, 0.3), rep);
            let (gate, _) = v.shared.expect("gated");
            (v.weights, gate)
        })
        .collect();
    let down_bytes = 4 * down.len();
    let weights_bytes = 4 * s.m * s.top_k;
    let gate_bytes = 4 * s.m;
    let (res_b, down_b, row_b) = (
        testkit::upload_f32(&ctx, &res),
        testkit::upload_f32(&ctx, &down),
        testkit::upload_f32(&ctx, &shared_row),
    );
    let (weights_b, gate_b) = (
        testkit::upload_f32(&ctx, &variants[0].0),
        testkit::upload_f32(&ctx, &variants[0].1),
    );
    let out_b = testkit::upload_filled(&ctx, POISON, rows);
    let set_variant = |case: usize| {
        weights_b
            .write_at(&stream, 0, &bytes_of(&variants[case].0))
            .expect("rewrite the weights");
        gate_b
            .write_at(&stream, 0, &bytes_of(&variants[case].1))
            .expect("rewrite the gate");
        out_b
            .write_at(&stream, 0, &vec![POISON; rows])
            .expect("poison the output");
    };
    let call = || {
        ctx.moe_combine(
            &stream,
            CombineCall {
                shape: s,
                res: res_b.range(0, rows).expect("res"),
                down: down_b.range(0, down_bytes).expect("down"),
                weights: weights_b.range(0, weights_bytes).expect("weights"),
                shared: Some(SharedTerm {
                    gate: gate_b.range(0, gate_bytes).expect("gate"),
                    row: row_b.range(0, rows).expect("row"),
                }),
                out: CombineOut::To(out_b.range(0, rows).expect("out")),
            },
        )
    };
    // The eager references on the same buffers (the first run also warms the kernel).
    let eager: Vec<Vec<u8>> = variants
        .iter()
        .enumerate()
        .map(|(case, _)| {
            set_variant(case);
            call().expect("the eager combine");
            stream.synchronize().expect("drain");
            testkit::read_bytes(&ctx, &out_b, rows)
        })
        .collect();
    set_variant(0);
    let (graph, ()) = ctx
        .capture(&stream, no_pins(), call)
        .expect("capture of the combine");
    assert_eq!(graph.nodes(), 1, "the combine graph holds one kernel");
    for (case, (weights, gate)) in variants.iter().enumerate() {
        set_variant(case);
        graph.launch(&stream).expect("replay");
        stream.synchronize().expect("drain");
        let got = testkit::read_bytes(&ctx, &out_b, rows);
        assert_eq!(
            got, eager[case],
            "replay {case}: the replay differs from the eager launch"
        );
        let inp = Inputs {
            shape: s,
            res: res.clone(),
            down: down.clone(),
            weights: weights.clone(),
            shared: Some((gate.clone(), shared_row.clone())),
        };
        let floats = testkit::floats(&got);
        check(&format!("replay {case}"), &inp, &floats);
    }
}
