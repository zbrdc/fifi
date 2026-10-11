//! Device tests (3090) of the router wrapper at seam 1: the Rust wrapper against the C entry
//! against the kernels, checked against two host references written here. `ref_f64` is the rule in
//! f64 (logits, softmax, rank by logit then lower index, weights, gate). `route_f32` is the f32
//! rule of `route_experts` (libm `exp`, sequential sums) applied to the logits the device wrote.
//! The test does not call `family::route_experts`: `kernels::cuda` never depends on `family`.
//!
//! The bounds are the class R1 of lane M: the margin `2^-10`, the weight and the gate `2^-17`.
//! Rows whose selected set differs from the f64 set inside the margin are counted and reported,
//! they do not fail (S1 and A5 handle real flips). The device `expf` and tree sum differ from libm
//! by rounding, so a rank swap between probabilities within 4 ulp is counted as near-equal.
//!
//! The device tests assume `--test-threads=1`, as the device suite is always run (decision 0040).
//! T6 (a capture) is deferred to RD-11a: this base has no capture window (V9).

use crate::EngError;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream};
use crate::kernels::cuda::resident::router::{
    ROUTER_STATUS_BIT, RouterCall, RouterOut, RouterShape, RouterWeights, router_logits_bytes,
};
use crate::kernels::cuda::testkit::{self, Rng};

/// The R1 margin, `2^-10` (lane M REPLY-44, final by REPLY-50, decision 0047).
const R1_MARGIN: f64 = 1.0 / 1024.0;
/// The R1 bound on a weight, `2^-17` (REPLY-44, final by REPLY-50).
const R1_WEIGHT: f64 = 1.0 / 131_072.0;
/// The R1 bound on the shared gate, `2^-17` (REPLY-44, final by REPLY-50).
const R1_GATE: f64 = 1.0 / 131_072.0;
const POISON: u8 = 0x55;
const SLACK: usize = 64;

fn shape(hidden: usize, n_expert: usize, top_k: usize, m: usize) -> RouterShape {
    RouterShape {
        m,
        hidden,
        n_expert,
        top_k,
    }
}

// ---------------------------------------------------------------- host references

struct HostRow {
    logits: Vec<f64>,
    /// Expert ids by logit descending, then lower index.
    order: Vec<usize>,
    probs: Vec<f64>,
    gate: Option<f64>,
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn ref_f64(x: &[f32], w: &[f32], shared: Option<&[f32]>, n_expert: usize) -> HostRow {
    let hidden = x.len();
    let logits: Vec<f64> = (0..n_expert)
        .map(|e| dot64(x, &w[e * hidden..(e + 1) * hidden]))
        .collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    let max = logits[order[0]];
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    HostRow {
        probs: exps.iter().map(|e| e / sum).collect(),
        gate: shared.map(|s| 1.0 / (1.0 + (-dot64(x, s)).exp())),
        logits,
        order,
    }
}

/// The f32 rule of `route_experts`: ids in rank order, weights, and the f32 probabilities.
fn route_f32(logits: &[f32], top_k: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum = exps.iter().fold(0.0f32, |acc, e| acc + e);
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]).then(a.cmp(b)));
    order.truncate(top_k);
    let kept = order.iter().fold(0.0f32, |acc, e| acc + probs[*e]);
    let weights = order.iter().map(|e| probs[*e] / kept).collect();
    (order, weights, probs)
}

fn ulp_apart(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

// ---------------------------------------------------------------- device rig

type Part<'a> = (&'a DeviceBuffer, usize, usize);

/// The ranges of one call, each as buffer, offset and bytes. A test changes one field of a valid
/// base to reach one refusal.
#[derive(Clone, Copy)]
struct Parts<'a> {
    x: Part<'a>,
    w: Part<'a>,
    w_shared: Option<Part<'a>>,
    logits: Part<'a>,
    ids: Part<'a>,
    weights: Part<'a>,
    shared_gate: Option<Part<'a>>,
    status: Part<'a>,
}

struct Rig<'c> {
    ctx: &'c CudaContext,
    stream: &'c DeviceStream,
    shape: RouterShape,
    x: DeviceBuffer,
    w: DeviceBuffer,
    w_shared: Option<DeviceBuffer>,
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    shared_gate: Option<DeviceBuffer>,
    status: DeviceBuffer,
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn padded(ctx: &CudaContext, mut bytes: Vec<u8>, fill: u8) -> DeviceBuffer {
    bytes.resize(bytes.len() + SLACK, fill);
    testkit::upload_bytes(ctx, &bytes)
}

fn ints(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect()
}

impl<'c> Rig<'c> {
    fn new(
        ctx: &'c CudaContext,
        stream: &'c DeviceStream,
        shape: RouterShape,
        data: (&[f32], &[f32], Option<&[f32]>),
    ) -> Self {
        let picks = shape.m * shape.top_k * 4;
        let gated = data.2.is_some();
        Self {
            ctx,
            stream,
            shape,
            x: padded(ctx, bytes_of(data.0), 0xAB),
            w: padded(ctx, bytes_of(data.1), 0xAB),
            w_shared: data.2.map(|s| padded(ctx, bytes_of(s), 0xAB)),
            logits: padded(
                ctx,
                vec![POISON; router_logits_bytes(shape).expect("bytes")],
                POISON,
            ),
            ids: padded(ctx, vec![POISON; picks], POISON),
            weights: padded(ctx, vec![POISON; picks], POISON),
            shared_gate: gated.then(|| padded(ctx, vec![POISON; shape.m * 4], POISON)),
            status: padded(ctx, vec![0; 4], 0),
        }
    }

    fn parts(&self) -> Parts<'_> {
        let s = self.shape;
        let picks = s.m * s.top_k * 4;
        Parts {
            x: (&self.x, 0, s.m * s.hidden * 4),
            w: (&self.w, 0, s.n_expert * s.hidden * 4),
            w_shared: self.w_shared.as_ref().map(|b| (b, 0, s.hidden * 4)),
            logits: (&self.logits, 0, router_logits_bytes(s).expect("bytes")),
            ids: (&self.ids, 0, picks),
            weights: (&self.weights, 0, picks),
            shared_gate: self.shared_gate.as_ref().map(|b| (b, 0, s.m * 4)),
            status: (&self.status, 0, 4),
        }
    }

    fn call(&self) -> Result<(), EngError> {
        self.call_with(self.shape, self.stream, self.parts())
    }

    fn call_with(
        &self,
        shape: RouterShape,
        stream: &DeviceStream,
        p: Parts<'_>,
    ) -> Result<(), EngError> {
        let range = range_of;
        let call = RouterCall {
            shape,
            x: range(p.x)?,
            weights: RouterWeights {
                w: range(p.w)?,
                w_shared: p.w_shared.map(range).transpose()?,
            },
            logits: range(p.logits)?,
            out: RouterOut {
                ids: range(p.ids)?,
                weights: range(p.weights)?,
                shared_gate: p.shared_gate.map(range).transpose()?,
                status: range(p.status)?,
            },
        };
        self.ctx.router(stream, call)
    }

    fn read(&self, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
        self.stream.synchronize().expect("synchronize");
        testkit::read_bytes(self.ctx, buffer, bytes)
    }

    fn read_logits(&self) -> Vec<f32> {
        let bytes = router_logits_bytes(self.shape).expect("bytes");
        testkit::floats(&self.read(&self.logits, bytes))
    }

    fn read_ids(&self) -> Vec<i32> {
        ints(&self.read(&self.ids, self.shape.m * self.shape.top_k * 4))
    }

    fn read_weights(&self) -> Vec<f32> {
        testkit::floats(&self.read(&self.weights, self.shape.m * self.shape.top_k * 4))
    }

    fn read_gate(&self) -> Option<Vec<f32>> {
        let buffer = self.shared_gate.as_ref()?;
        Some(testkit::floats(&self.read(buffer, self.shape.m * 4)))
    }

    fn read_status(&self) -> i32 {
        ints(&self.read(&self.status, 4))[0]
    }

    /// Every byte the call writes or that a refusal must leave alone.
    fn snapshot(&self) -> Vec<Vec<u8>> {
        let s = self.shape;
        let picks = s.m * s.top_k * 4 + SLACK;
        let mut all = vec![
            self.read(&self.logits, router_logits_bytes(s).expect("bytes") + SLACK),
            self.read(&self.ids, picks),
            self.read(&self.weights, picks),
            self.read(&self.status, 4 + SLACK),
        ];
        if let Some(gate) = &self.shared_gate {
            all.push(self.read(gate, s.m * 4 + SLACK));
        }
        all
    }
}

fn range_of(part: Part<'_>) -> Result<DeviceRange<'_>, EngError> {
    part.0.range(part.1, part.2)
}

fn gaussian(rng: &mut Rng, len: usize, scale: f64) -> Vec<f32> {
    (0..len).map(|_| (rng.normal() * scale) as f32).collect()
}

// ---------------------------------------------------------------- T1

#[derive(Default)]
struct Stats {
    rows: usize,
    ids_f32: usize,
    weights_f32: usize,
    set_f64: usize,
    weight_by_id: usize,
    gate: usize,
    status_ok: usize,
    near_equal: usize,
    flips: usize,
    max_weight_f32: f64,
    max_weight_f64: f64,
    max_gate: f64,
}

struct RowCase<'a> {
    label: &'a str,
    top_k: usize,
    x: &'a [f32],
    w: &'a [f32],
    shared: Option<&'a [f32]>,
}

struct DeviceRow<'a> {
    logits: &'a [f32],
    ids: &'a [i32],
    weights: &'a [f32],
    gate: Option<f32>,
}

/// Check (1) and (2): the ids and weights against the f32 rule on the device logits.
fn check_f32_rule(case: &RowCase<'_>, dev: &DeviceRow<'_>, stats: &mut Stats) {
    let (ids, weights, probs) = route_f32(dev.logits, case.top_k);
    let got: Vec<usize> = dev
        .ids
        .iter()
        .map(|i| usize::try_from(*i).expect("id"))
        .collect();
    if let Some(rank) = (0..case.top_k).find(|r| got[*r] != ids[*r]) {
        let (a, b) = (got[rank], ids[rank]);
        assert!(
            ulp_apart(probs[a], probs[b]) <= 4,
            "{}: rank {rank} holds expert {a}, the f32 rule says {b}",
            case.label
        );
        let (mut left, mut right) = (got[rank..].to_vec(), ids[rank..].to_vec());
        left.sort_unstable();
        right.sort_unstable();
        assert_eq!(left, right, "{}: the tail sets differ", case.label);
        stats.near_equal += 1;
    }
    stats.ids_f32 += 1;
    let worst = (0..case.top_k)
        .map(|r| f64::from((dev.weights[r] - weights[r]).abs()))
        .fold(0.0, f64::max);
    stats.max_weight_f32 = stats.max_weight_f32.max(worst);
    assert!(
        worst <= R1_WEIGHT,
        "{}: weight differs by {worst:e} from the f32 rule",
        case.label
    );
    stats.weights_f32 += 1;
}

/// Checks (3) and (4): the margin-set rule and the weight by id against f64.
fn check_f64_rule(case: &RowCase<'_>, dev: &DeviceRow<'_>, host: &HostRow, stats: &mut Stats) {
    let k = case.top_k;
    let boundary = host.logits[host.order[k - 1]];
    let chosen: Vec<usize> = dev
        .ids
        .iter()
        .map(|i| usize::try_from(*i).expect("id"))
        .collect();
    for e in 0..host.logits.len() {
        let present = chosen.contains(&e);
        assert!(
            present || host.logits[e] <= boundary + R1_MARGIN,
            "{}: expert {e} (logit {}) is missing, the k-th logit is {boundary}",
            case.label,
            host.logits[e]
        );
        assert!(
            !present || host.logits[e] >= boundary - R1_MARGIN,
            "{}: expert {e} (logit {}) is selected, the k-th logit is {boundary}",
            case.label,
            host.logits[e]
        );
    }
    stats.set_f64 += 1;
    let mut want = host.order[..k].to_vec();
    let mut have = chosen.clone();
    want.sort_unstable();
    have.sort_unstable();
    if want != have {
        stats.flips += 1;
        return;
    }
    let kept: f64 = host.order[..k].iter().map(|e| host.probs[*e]).sum();
    for (slot, id) in chosen.iter().enumerate() {
        let diff = (f64::from(dev.weights[slot]) - host.probs[*id] / kept).abs();
        stats.max_weight_f64 = stats.max_weight_f64.max(diff);
        assert!(
            diff <= R1_WEIGHT,
            "{}: weight of expert {id} differs by {diff:e} from f64",
            case.label
        );
    }
    stats.weight_by_id += 1;
}

fn check_row(case: &RowCase<'_>, dev: &DeviceRow<'_>, stats: &mut Stats) {
    stats.rows += 1;
    let host = ref_f64(case.x, case.w, case.shared, case.w.len() / case.x.len());
    check_f32_rule(case, dev, stats);
    check_f64_rule(case, dev, &host, stats);
    if let (Some(want), Some(got)) = (host.gate, dev.gate) {
        let diff = (f64::from(got) - want).abs();
        stats.max_gate = stats.max_gate.max(diff);
        assert!(diff <= R1_GATE, "{}: gate differs by {diff:e}", case.label);
        stats.gate += 1;
    }
}

fn run_shape(
    ctx: &CudaContext,
    stream: &DeviceStream,
    s: RouterShape,
    gated: bool,
    seed: u64,
) -> Stats {
    let mut rng = Rng::new(seed);
    let unit_w = gaussian(&mut rng, s.n_expert * s.hidden, 1.0);
    let unit_s = gaussian(&mut rng, s.hidden, 1.0);
    let mut stats = Stats::default();
    let sentinel = padded(ctx, vec![POISON; s.m * 4], POISON);
    for batch in 0..20 {
        let scale = 0.02 + 0.6 * f64::from(batch) / 19.0;
        let x = gaussian(&mut rng, s.m * s.hidden, 1.0);
        let w: Vec<f32> = unit_w
            .iter()
            .map(|v| (f64::from(*v) * scale) as f32)
            .collect();
        let shared: Vec<f32> = unit_s
            .iter()
            .map(|v| (f64::from(*v) * scale) as f32)
            .collect();
        let shared = gated.then_some(shared);
        let rig = Rig::new(ctx, stream, s, (&x, &w, shared.as_deref()));
        rig.call().expect("router call");
        let (logits, ids, weights) = (rig.read_logits(), rig.read_ids(), rig.read_weights());
        let gate = rig.read_gate();
        assert_eq!(rig.read_status(), 0, "{s:?} batch {batch}: status");
        stats.status_ok += 1;
        for row in 0..s.m {
            let label = format!(
                "{:?} batch {batch} row {row}",
                (s.hidden, s.n_expert, s.top_k, s.m)
            );
            let case = RowCase {
                label: &label,
                top_k: s.top_k,
                x: &x[row * s.hidden..(row + 1) * s.hidden],
                w: &w,
                shared: shared.as_deref(),
            };
            let dev = DeviceRow {
                logits: &logits[row * (s.n_expert + 1)..row * (s.n_expert + 1) + s.n_expert],
                ids: &ids[row * s.top_k..(row + 1) * s.top_k],
                weights: &weights[row * s.top_k..(row + 1) * s.top_k],
                gate: gate.as_ref().map(|g| g[row]),
            };
            check_row(&case, &dev, &mut stats);
        }
        if !gated {
            assert_eq!(
                testkit::read_bytes(ctx, &sentinel, s.m * 4),
                vec![POISON; s.m * 4],
                "the unused shared_gate buffer changed"
            );
        }
    }
    stats
}

#[test]
fn router_matches_the_host_references() {
    if !testkit::device_present("router_matches_the_host_references") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let gated_only = [
        shape(2048, 256, 8, 1),
        shape(2048, 256, 8, 4),
        shape(2048, 256, 8, 16),
        shape(2048, 256, 1, 2),
        shape(2048, 256, 32, 2),
        shape(128, 8, 8, 3),
        shape(2052, 100, 2, 5),
    ];
    let both = [
        shape(2304, 64, 8, 3),
        shape(2048, 63, 8, 3),
        shape(128, 1, 1, 2),
        shape(2052, 7, 7, 3),
        shape(256, 31, 4, 2),
        shape(2048, 255, 8, 2),
    ];
    let mut cases: Vec<(RouterShape, bool)> = gated_only.iter().map(|s| (*s, true)).collect();
    for s in both {
        cases.push((s, false));
        cases.push((s, true));
    }
    let mut total = Stats::default();
    for (index, (s, gated)) in cases.into_iter().enumerate() {
        let stats = run_shape(&ctx, &stream, s, gated, 0x5eed_0001 + index as u64);
        println!(
            "T1 {:?} gate {gated}: rows {} ids_f32 {} weights_f32 {} set_f64 {} weight_by_id {} gate {} status {} near_equal {} flips {} max_w_f32 {:e} max_w_f64 {:e} max_gate {:e}",
            (s.hidden, s.n_expert, s.top_k, s.m),
            stats.rows,
            stats.ids_f32,
            stats.weights_f32,
            stats.set_f64,
            stats.weight_by_id,
            stats.gate,
            stats.status_ok,
            stats.near_equal,
            stats.flips,
            stats.max_weight_f32,
            stats.max_weight_f64,
            stats.max_gate
        );
        total.rows += stats.rows;
        total.flips += stats.flips;
        total.near_equal += stats.near_equal;
        total.max_weight_f32 = total.max_weight_f32.max(stats.max_weight_f32);
        total.max_weight_f64 = total.max_weight_f64.max(stats.max_weight_f64);
        total.max_gate = total.max_gate.max(stats.max_gate);
    }
    println!(
        "T1 total: rows {} margin flips {} ({:.3} percent) near_equal {} max_w_f32 {:e} max_w_f64 {:e} max_gate {:e}",
        total.rows,
        total.flips,
        100.0 * total.flips as f64 / total.rows as f64,
        total.near_equal,
        total.max_weight_f32,
        total.max_weight_f64,
        total.max_gate
    );
}

// ---------------------------------------------------------------- exact-logit helpers

/// A rig whose logit `e` is exactly `values[e]`: `x` is ones and the router row `e` holds
/// `values[e]` in column 0 and zeros elsewhere, so every dot is exact.
fn exact_rig<'c>(
    ctx: &'c CudaContext,
    stream: &'c DeviceStream,
    values: &[f32],
    top_k: usize,
    gated: bool,
) -> Rig<'c> {
    let hidden = 2048;
    let s = shape(hidden, values.len(), top_k, 1);
    let x = vec![1.0f32; hidden];
    let mut w = vec![0.0f32; values.len() * hidden];
    for (e, v) in values.iter().enumerate() {
        w[e * hidden] = *v;
    }
    let shared = vec![0.0f32; hidden];
    Rig::new(ctx, stream, s, (&x, &w, gated.then_some(shared.as_slice())))
}

fn expect_rank(rig: &Rig<'_>, ids: &[i32], label: &str) {
    rig.call().expect("router call");
    assert_eq!(rig.read_ids(), ids, "{label}: ids");
    assert_eq!(rig.read_status(), 0, "{label}: status");
}

#[test]
fn router_breaks_ties_to_the_lower_index() {
    if !testkit::device_present("router_breaks_ties_to_the_lower_index") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let zero = exact_rig(&ctx, &stream, &[0.0; 256], 8, true);
    expect_rank(&zero, &[0, 1, 2, 3, 4, 5, 6, 7], "all zero");
    assert_eq!(zero.read_weights(), vec![0.125f32; 8], "all-zero weights");
    assert_eq!(
        zero.read_gate().expect("gate"),
        vec![0.5f32],
        "all-zero gate"
    );

    let mut tied = [0.0f32; 256];
    for e in [3, 5, 20, 37, 40] {
        tied[e] = 5.0;
    }
    expect_rank(
        &exact_rig(&ctx, &stream, &tied, 8, false),
        &[3, 5, 20, 37, 40, 0, 1, 2],
        "tie across lanes, k 8",
    );
    expect_rank(
        &exact_rig(&ctx, &stream, &tied, 3, false),
        &[3, 5, 20],
        "tie across lanes, k 3",
    );

    let mut rows = [0.0f32; 256];
    for e in [40, 41, 200, 201] {
        rows[e] = 7.0;
    }
    expect_rank(
        &exact_rig(&ctx, &stream, &rows, 8, false),
        &[40, 41, 200, 201, 0, 1, 2, 3],
        "four identical rows",
    );

    let mut under = [-200.0f32; 256];
    under[100] = 0.0;
    let rig = exact_rig(&ctx, &stream, &under, 8, false);
    expect_rank(&rig, &[100, 0, 1, 2, 3, 4, 5, 6], "underflow");
    assert_eq!(rig.read_weights()[0], 1.0, "underflow: weight of the first");
}

// ---------------------------------------------------------------- T3

fn assert_faulty_row(rig: &Rig<'_>, row: usize, label: &str) {
    let k = rig.shape.top_k;
    assert_eq!(
        rig.read_ids()[row * k..(row + 1) * k],
        vec![-1; k],
        "{label}: ids"
    );
    assert_eq!(
        rig.read_weights()[row * k..(row + 1) * k],
        vec![0.0f32; k],
        "{label}: weights"
    );
    assert_eq!(rig.read_gate().expect("gate")[row], 0.0, "{label}: gate");
    assert_eq!(
        rig.read_status() & ROUTER_STATUS_BIT,
        ROUTER_STATUS_BIT,
        "{label}: status bit"
    );
}

#[test]
fn router_flags_faulty_rows() {
    if !testkit::device_present("router_flags_faulty_rows") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = Rng::new(0x7a17);
    let s = shape(2048, 256, 8, 4);
    let mut x = gaussian(&mut rng, s.m * s.hidden, 1.0);
    let w = gaussian(&mut rng, s.n_expert * s.hidden, 0.05);
    let shared = gaussian(&mut rng, s.hidden, 0.05);
    x[s.hidden + 17] = f32::NAN;
    let rig = Rig::new(&ctx, &stream, s, (&x, &w, Some(&shared)));
    rig.call().expect("router call");
    assert_faulty_row(&rig, 1, "NaN in row 1");
    let (ids, weights) = (rig.read_ids(), rig.read_weights());
    for row in [0, 2, 3] {
        let slots = row * 8..(row + 1) * 8;
        assert!(
            ids[slots.clone()].iter().all(|i| (0..256).contains(i)),
            "row {row} ids"
        );
        let sum: f32 = weights[slots].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {row} weights sum {sum}");
    }

    let mut x = gaussian(&mut rng, s.hidden, 1.0);
    let mut big_w = vec![0.0f32; 256 * s.hidden];
    x[0] = 10.0;
    big_w[0] = 3.0e38;
    let one = shape(2048, 256, 8, 1);
    let rig = Rig::new(&ctx, &stream, one, (&x, &big_w, Some(&shared)));
    rig.call().expect("router call");
    assert_faulty_row(&rig, 0, "overflowing logit");

    let mut big_shared = vec![0.0f32; s.hidden];
    big_shared[0] = 3.0e38;
    let rig = Rig::new(
        &ctx,
        &stream,
        one,
        (&x, &w[..256 * s.hidden], Some(&big_shared)),
    );
    rig.call().expect("router call");
    assert_faulty_row(&rig, 0, "overflowing gate dot");

    for (name, bad) in [
        ("NaN", f32::NAN),
        ("+inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
    ] {
        for lane in [10usize, 30, 33] {
            let mut values = [0.0f32; 256];
            values[lane] = bad;
            let rig = exact_rig(&ctx, &stream, &values, 8, true);
            rig.call().expect("router call");
            assert_faulty_row(&rig, 0, &format!("{name} logit at expert {lane}"));
        }
    }

    let mut near = [0.0f32; 256];
    for e in [7, 99, 200] {
        near[e] = 100.0;
    }
    let rig = exact_rig(&ctx, &stream, &near, 8, true);
    expect_rank(&rig, &[7, 99, 200, 0, 1, 2, 3, 4], "logits near 100");
    let sum: f32 = rig.read_weights().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "logits near 100: weights sum {sum}"
    );
}

#[test]
fn router_flags_faulty_rows_for_an_odd_expert_count_without_a_gate() {
    if !testkit::device_present("router_flags_faulty_rows_for_an_odd_expert_count_without_a_gate") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut values = [0.0f32; 63];
    values[10] = f32::NAN;
    let rig = exact_rig(&ctx, &stream, &values, 8, false);
    rig.call().expect("router call");
    assert_eq!(rig.read_ids(), vec![-1; 8], "NaN logit: ids");
    assert_eq!(rig.read_weights(), vec![0.0f32; 8], "NaN logit: weights");
    assert_eq!(rig.read_status(), ROUTER_STATUS_BIT, "NaN logit: status");

    let mut rng = Rng::new(0x0dd5);
    let s = shape(2048, 63, 8, 3);
    let mut x = gaussian(&mut rng, s.m * s.hidden, 1.0);
    let w = gaussian(&mut rng, s.n_expert * s.hidden, 0.05);
    x[s.hidden + 5] = f32::NAN;
    let rig = Rig::new(&ctx, &stream, s, (&x, &w, None));
    rig.call().expect("router call");
    let (ids, weights) = (rig.read_ids(), rig.read_weights());
    assert_eq!(ids[8..16], vec![-1; 8], "NaN row: ids");
    assert_eq!(weights[8..16], vec![0.0f32; 8], "NaN row: weights");
    assert_eq!(rig.read_status() & ROUTER_STATUS_BIT, ROUTER_STATUS_BIT);
    for row in [0, 2] {
        let slots = row * 8..(row + 1) * 8;
        assert!(
            ids[slots.clone()].iter().all(|i| (0..63).contains(i)),
            "row {row} ids"
        );
        let sum: f32 = weights[slots].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {row} weights sum {sum}");
    }
}

// ---------------------------------------------------------------- T4

#[test]
fn router_is_deterministic() {
    if !testkit::device_present("router_is_deterministic") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (s, gated) in [
        (shape(2048, 256, 8, 4), true),
        (shape(2048, 63, 8, 3), false),
    ] {
        let mut rng = Rng::new(0xde7e + s.n_expert as u64);
        let x = gaussian(&mut rng, s.m * s.hidden, 1.0);
        let w = gaussian(&mut rng, s.n_expert * s.hidden, 0.1);
        let shared = gaussian(&mut rng, s.hidden, 0.1);
        let mut rig = Rig::new(
            &ctx,
            &stream,
            s,
            (&x, &w, gated.then_some(shared.as_slice())),
        );
        rig.call().expect("first launch");
        let first = (
            rig.read_logits(),
            rig.read_ids(),
            rig.read_weights(),
            rig.read_gate(),
        );
        rig.logits = padded(
            &ctx,
            vec![0xFF; router_logits_bytes(s).expect("bytes")],
            0xFF,
        );
        rig.call().expect("second launch");
        let second = (
            rig.read_logits(),
            rig.read_ids(),
            rig.read_weights(),
            rig.read_gate(),
        );
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let stride = s.n_expert + 1;
        for row in 0..s.m {
            let range = row * stride..row * stride + s.n_expert;
            assert_eq!(
                bits(&first.0[range.clone()]),
                bits(&second.0[range]),
                "{s:?}: logits differ"
            );
        }
        assert_eq!(first.1, second.1, "{s:?}: ids differ");
        assert_eq!(bits(&first.2), bits(&second.2), "{s:?}: weights differ");
        assert_eq!(
            first.3.as_deref().map(bits),
            second.3.as_deref().map(bits),
            "{s:?}: gate differs"
        );
    }
}

// ---------------------------------------------------------------- T5

fn assert_refused(result: Result<(), EngError>, case: &str, needle: &str) {
    match result {
        Err(EngError::BadRequest(message)) => {
            assert!(
                message.contains(needle),
                "{case}: `{message}` lacks `{needle}`"
            );
        }
        other => panic!("{case}: expected BadRequest with `{needle}`, got {other:?}"),
    }
}

/// The limit and pairing rows for one base rig.
fn refuse_limits(rig: &Rig<'_>) {
    let base = rig.shape;
    let limits = "the CUDA router takes";
    let rows = [
        ("m 0", RouterShape { m: 0, ..base }),
        ("m 4097", RouterShape { m: 4097, ..base }),
        ("hidden 0", RouterShape { hidden: 0, ..base }),
        ("hidden 2", RouterShape { hidden: 2, ..base }),
        ("hidden 6", RouterShape { hidden: 6, ..base }),
        (
            "hidden 65540",
            RouterShape {
                hidden: 65540,
                ..base
            },
        ),
        (
            "n_expert 0",
            RouterShape {
                n_expert: 0,
                ..base
            },
        ),
        (
            "n_expert 257",
            RouterShape {
                n_expert: 257,
                ..base
            },
        ),
        ("top_k 0", RouterShape { top_k: 0, ..base }),
        ("top_k 33", RouterShape { top_k: 33, ..base }),
        (
            "top_k above n_expert",
            RouterShape {
                n_expert: 4,
                top_k: 5,
                ..base
            },
        ),
    ];
    for (case, bad) in rows {
        assert_refused(rig.call_with(bad, rig.stream, rig.parts()), case, limits);
    }
}

fn refuse_pairing(rig: &Rig<'_>) {
    let together = "together or neither";
    let spare = padded(rig.ctx, vec![0; 4 * rig.shape.m], 0);
    let mut parts = rig.parts();
    parts.shared_gate = None;
    assert_refused(
        rig.call_with(rig.shape, rig.stream, parts),
        "w_shared without shared_gate",
        together,
    );
    let mut parts = rig.parts();
    parts.w_shared = None;
    parts.shared_gate = Some((&spare, 0, 4 * rig.shape.m));
    assert_refused(
        rig.call_with(rig.shape, rig.stream, parts),
        "shared_gate without w_shared",
        together,
    );
}

/// A part of `buffer` with `bytes` less than the full range.
fn short(part: Part<'_>, by: usize) -> Part<'_> {
    (part.0, part.1, part.2 - by)
}

fn shifted(part: Part<'_>, by: usize) -> Part<'_> {
    (part.0, part.1 + by, part.2)
}

fn refuse_sizes_and_alignment(rig: &Rig<'_>) {
    let s = rig.shape;
    let call = |parts: Parts<'_>| rig.call_with(s, rig.stream, parts);
    let base = rig.parts();
    let holds = |name: &str| format!("the CUDA {name} range holds");
    let starts = |name: &str| format!("the CUDA {name} range starts at offset");
    assert_refused(
        call(Parts {
            x: short(base.x, 4),
            ..base
        }),
        "x short",
        &holds("x"),
    );
    assert_refused(
        call(Parts {
            w: short(base.w, 4),
            ..base
        }),
        "w short",
        &holds("w"),
    );
    assert_refused(
        call(Parts {
            w_shared: base.w_shared.map(|p| short(p, 4)),
            ..base
        }),
        "w_shared short",
        &holds("w_shared"),
    );
    assert_refused(
        call(Parts {
            logits: short(base.logits, 4),
            ..base
        }),
        "logits short",
        &holds("logits"),
    );
    assert_refused(
        call(Parts {
            ids: short(base.ids, 4),
            ..base
        }),
        "ids short",
        &holds("ids"),
    );
    assert_refused(
        call(Parts {
            weights: short(base.weights, 4),
            ..base
        }),
        "weights short",
        &holds("weights"),
    );
    assert_refused(
        call(Parts {
            shared_gate: base.shared_gate.map(|p| short(p, 4)),
            ..base
        }),
        "shared_gate short",
        &holds("shared_gate"),
    );
    assert_refused(
        call(Parts {
            status: (base.status.0, 0, 2),
            ..base
        }),
        "status of 2 bytes",
        &holds("status"),
    );
    assert_refused(
        call(Parts {
            x: shifted(base.x, 4),
            ..base
        }),
        "x +4",
        &starts("x"),
    );
    assert_refused(
        call(Parts {
            w: shifted(base.w, 8),
            ..base
        }),
        "w +8",
        &starts("w"),
    );
    assert_refused(
        call(Parts {
            w_shared: base.w_shared.map(|p| shifted(p, 4)),
            ..base
        }),
        "w_shared +4",
        &starts("w_shared"),
    );
    assert_refused(
        call(Parts {
            shared_gate: base.shared_gate.map(|p| shifted(p, 2)),
            ..base
        }),
        "shared_gate +2",
        &starts("shared_gate"),
    );
    assert_refused(
        call(Parts {
            logits: shifted(base.logits, 2),
            ..base
        }),
        "logits +2",
        &starts("logits"),
    );
    assert_refused(
        call(Parts {
            ids: shifted(base.ids, 2),
            ..base
        }),
        "ids +2",
        &starts("ids"),
    );
    assert_refused(
        call(Parts {
            weights: shifted(base.weights, 2),
            ..base
        }),
        "weights +2",
        &starts("weights"),
    );
    assert_refused(
        call(Parts {
            status: shifted(base.status, 2),
            ..base
        }),
        "status +2",
        &starts("status"),
    );
}

/// Overlap rows beyond ids/weights: one pair of ranges moved into one buffer per row, the written
/// range named first in the refusal.
fn refuse_more_overlaps(rig: &Rig<'_>, ctx: &CudaContext) {
    let s = rig.shape;
    let picks = s.m * s.top_k * 4;
    let logits = router_logits_bytes(s).expect("bytes");
    let both = padded(ctx, vec![POISON; 20_000], POISON);
    let base = rig.parts();
    let rows = [
        (
            "ids over logits",
            Parts {
                logits: (&both, 0, logits),
                ids: (&both, logits - 4, picks),
                ..base
            },
            "the CUDA ids and logits ranges overlap",
        ),
        (
            "weights over status",
            Parts {
                weights: (&both, 0, picks),
                status: (&both, picks - 4, 4),
                ..base
            },
            "the CUDA weights and status ranges overlap",
        ),
        (
            "weights over shared_gate",
            Parts {
                weights: (&both, 0, picks),
                shared_gate: Some((&both, picks - 4, s.m * 4)),
                ..base
            },
            "the CUDA weights and shared_gate ranges overlap",
        ),
        (
            "ids over x",
            Parts {
                ids: (&both, 12, picks),
                x: (&both, 0, s.m * s.hidden * 4),
                ..base
            },
            "the CUDA ids and x ranges overlap",
        ),
    ];
    for (case, parts, needle) in rows {
        assert_refused(rig.call_with(s, rig.stream, parts), case, needle);
    }
}

#[test]
fn router_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("router_rejects_bad_calls_before_any_launch") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let other = CudaContext::new().expect("second context");
    let foreign_stream = other.stream().expect("stream of the other context");
    let mut rng = Rng::new(0xbad0);
    let s = shape(2048, 256, 8, 2);
    let x = gaussian(&mut rng, s.m * s.hidden, 1.0);
    let w = gaussian(&mut rng, s.n_expert * s.hidden, 0.05);
    let shared = gaussian(&mut rng, s.hidden, 0.05);
    let rig = Rig::new(&ctx, &stream, s, (&x, &w, Some(&shared)));
    let before = rig.snapshot();
    refuse_limits(&rig);
    refuse_pairing(&rig);
    refuse_sizes_and_alignment(&rig);

    let odd = shape(2048, 63, 8, 2);
    let odd_rig = Rig::new(&ctx, &stream, odd, (&x, &w[..63 * 2048], None));
    let odd_before = odd_rig.snapshot();
    refuse_limits(&odd_rig);

    let pair = padded(&ctx, vec![POISON; 2 * s.m * s.top_k * 4], POISON);
    let picks = s.m * s.top_k * 4;
    let base = rig.parts();
    let overlap = Parts {
        ids: (&pair, 0, picks),
        weights: (&pair, picks - 4, picks),
        ..base
    };
    assert_refused(
        rig.call_with(s, &stream, overlap),
        "ids and weights overlap",
        "the CUDA ids and weights ranges overlap",
    );
    refuse_more_overlaps(&rig, &ctx);
    assert_refused(
        rig.call_with(s, &foreign_stream, base),
        "a stream of another context",
        "belongs to another adapter",
    );
    let foreign_x = testkit::upload_f32(&other, &x);
    let foreign = Parts {
        x: (&foreign_x, 0, s.m * s.hidden * 4),
        ..base
    };
    assert_refused(
        rig.call_with(s, &stream, foreign),
        "a buffer of another context",
        "belongs to another adapter",
    );

    assert_eq!(rig.snapshot(), before, "a refused call changed an output");
    assert_eq!(
        odd_rig.snapshot(),
        odd_before,
        "a refused call changed an output"
    );

    // Accepted controls: `require_fit` is an at-least check, so a longer status passes.
    let wide = padded(&ctx, vec![0; 4], 0);
    let longer = Parts {
        status: (&wide, 0, 8),
        ..base
    };
    rig.call_with(s, &stream, longer)
        .expect("status of 8 bytes");
    let inner = Parts {
        status: (&wide, 4, 4),
        ..base
    };
    rig.call_with(s, &stream, inner)
        .expect("status at offset 4");
    stream.synchronize().expect("synchronize");
    rig.call()
        .expect("the backend is usable after the refusals");
    assert_eq!(rig.read_status(), 0, "clean status after the controls");
}

// ---------------------------------------------------------------- T7

#[test]
fn router_logits_bytes_pins_the_scratch_rule() {
    assert_eq!(
        router_logits_bytes(shape(2048, 256, 8, 1)).expect("bytes"),
        1_028
    );
    assert_eq!(
        router_logits_bytes(shape(2048, 256, 8, 16)).expect("bytes"),
        16_448
    );
    assert!(router_logits_bytes(shape(2048, 256, 8, usize::MAX)).is_err());
}
