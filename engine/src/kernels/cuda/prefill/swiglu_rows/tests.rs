//! Seam-1 tests of the swiglu_rows wrapper (CPF-7a part 1; CPF-V4-CPF7-PLAN section C.2, N2;
//! decision 0049 item 15). The plan's list: (1) the f64 reference under SW0, with the
//! finiteness clause to |g| = 88 and the flush-to-zero cases reported; (2) the layout rows with
//! canaries in the gaps; (3) shapes of 1, 7 and 16,384 rows; (4) bit equality of two runs; (5)
//! canaries around `act` (every value test checks every byte of the act buffer); (6) the refusal
//! tables, each guard reached and asserted by its own text. Lane C adds the numerics row against
//! the decode path's `swiglu_q8` and the NaN/inf pass-through row.
//!
//! SW0 (decision 0049 A5 item 3 and A6 item 3, PROVISIONAL until this packet's measured maximum
//! freezes it): relative error at most 2^-18 against the f64 `silu(g) * u` of the f32 inputs for
//! |g| <= 12, normal inputs and a normal exact value; a finite output for |g| <= 88. A case where
//! flush-to-zero fires is reported with its absolute error and not graded. The bound is lane M's,
//! not set here; the tests print the measured maximum on lines that start with SWIGLU-REPORT.
//!
//! Device-test structure (the S3b lesson): every device phase is `Result`-only (only `?` while a
//! device owner lives), every owner is dropped, then the asserts run on host `Vec`s. The tests run
//! with `--test-threads=1`. Lane C's merge gate: after the device phase and before any bound
//! check, every output value is asserted finite ([`assert_finite`]; the NaN/inf row asserts its
//! exact non-finite bits instead). `act` is prefilled with a NaN pattern, so an output the kernel
//! never wrote fails that gate, which a max-of-deltas judge (`f64::max` drops NaN) could not see.
//!
//! Argued, not tested: the `rows` and `n_ff` i32 refusals of `check_shape` (their range rules come
//! first and lie far below i32::MAX); the wrapper's `size` refusal (after `check_shape` every factor
//! is below 2^31, so the extents cannot overflow a 64-bit usize; the pure test pins the overflow
//! answers of `swiglu_rows_needs`); the NULL guards of the C entry (a `RawRange` always addresses
//! a live allocation).

use super::{
    MAX_FF, MAX_ROWS, SwigluRowsCall, SwigluRowsNeeds, SwigluRowsShape, check_shape,
    swiglu_rows_needs,
};
use crate::EngError;
use crate::kernels::cuda::resident::add_rmsnorm_q8::tests::{butterfly32, q8_ref};
use crate::kernels::cuda::resident::swiglu_q8::{SwigluQ8Call, swiglu_q8_needs};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// SW0: 2^-18 relative.
const SW0: f64 = 1.0 / 262_144.0;
/// The gate domain of SW0's relative bound.
const SW0_GATE: f64 = 12.0;
/// The gate domain of SW0's finiteness clause.
const FINITE_GATE: f64 = 88.0;
/// The byte that fills `act` before a launch: f32 0xffffffff is a NaN, and not the device's
/// canonical NaN 0x7fffffff, so an output the kernel did not write fails the finiteness gate.
const ACT_FILL: u8 = 0xff;
/// Fill bytes before each range and after it: a stray write stays visible, and a shifted, short
/// or moved range still lies inside its buffer, so a refusal row reaches its own guard.
const PAD: usize = 256;
const SLACK: usize = 4_096;
/// The buffer and slot indices of the rig.
const GU: usize = 0;
const ACT: usize = 1;

fn shape(rows: usize, n_ff: usize, up_off: usize, ld_in: usize, ld_out: usize) -> SwigluRowsShape {
    SwigluRowsShape {
        rows,
        n_ff,
        up_off,
        ld_in,
        ld_out,
    }
}

/// The Qwen3.6 MoE shape at `rows` rows: n_ff 512, gate and up back to back in a row of 1,024
/// (the gate/up GEMM output), act rows of 512.
fn qwen(rows: usize) -> SwigluRowsShape {
    shape(rows, 512, 512, 1_024, 512)
}

fn needs(shape: &SwigluRowsShape) -> Result<SwigluRowsNeeds, EngError> {
    swiglu_rows_needs(shape)
        .ok_or_else(|| EngError::BadRequest("the test shape overflows".to_owned()))
}

/// A gate value uniform in [-limit, limit).
fn gate(rng: &mut Rng, limit: f64) -> f32 {
    ((rng.unit() * 2.0 - 1.0) * limit) as f32
}

/// An up value: a random sign times 2^e, e uniform in [-4, 4). With |g| <= 12 every exact value
/// is a normal f32, and every 32-value block sum stays inside the fp16 range of swiglu_q8's `s`.
fn up(rng: &mut Rng) -> f32 {
    let magnitude = (rng.unit() * 8.0 - 4.0).exp2();
    (if rng.below(2) == 0 {
        magnitude
    } else {
        -magnitude
    }) as f32
}

/// The gu floats of `shape`, `rows * ld_in` values: the gate at columns `0..n_ff`, the up value at
/// `up_off..up_off + n_ff`, NaN in every other column, so a read of a gap shows as NaN.
fn gu_values(
    shape: &SwigluRowsShape,
    rng: &mut Rng,
    gate_of: &dyn Fn(&mut Rng) -> f32,
    up_of: &dyn Fn(&mut Rng) -> f32,
) -> Vec<f32> {
    let mut values = vec![f32::NAN; shape.rows * shape.ld_in];
    for row in values.chunks_exact_mut(shape.ld_in) {
        let (gates, rest) = row.split_at_mut(shape.up_off);
        for (g, u) in gates[..shape.n_ff].iter_mut().zip(&mut rest[..shape.n_ff]) {
            *g = gate_of(rng);
            *u = up_of(rng);
        }
    }
    values
}

fn le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The bytes of the gu buffer: PAD bytes of NaN, the gu extent of `values`, SLACK bytes of NaN.
fn gu_bytes(values: &[f32], need: &SwigluRowsNeeds) -> Vec<u8> {
    let mut bytes = le(&vec![f32::NAN; PAD / 4]);
    bytes.extend(le(&values[..need.gu / 4]));
    bytes.extend(le(&vec![f32::NAN; SLACK / 4]));
    bytes
}

/// f64 `silu(g) * u` of the f32 inputs: the SW0 reference.
fn reference(g: f32, u: f32) -> f64 {
    testkit::silu_f64(f64::from(g)) * f64::from(u)
}

/// (buffer, offset, bytes) of one range of the call.
type Slot = (usize, usize, usize);

/// One context and stream, the gu and act buffers, the slots the call takes its ranges from, and
/// the shape.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    bufs: [DeviceBuffer; 2],
    slots: [Slot; 2],
    shape: SwigluRowsShape,
}

impl Rig {
    /// The gu buffer of [`gu_bytes`] and an act buffer of PAD + extent + SLACK fill bytes; both
    /// slots are the extents at offset PAD.
    fn new(shape: SwigluRowsShape, values: &[f32]) -> Result<Self, EngError> {
        let need = needs(&shape)?;
        if values.len() * 4 < need.gu {
            return Err(EngError::BadRequest(
                "the test values do not cover the gu extent".to_owned(),
            ));
        }
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let bufs = [
            ctx.upload(&gu_bytes(values, &need))?,
            ctx.upload(&vec![ACT_FILL; PAD + need.act + SLACK])?,
        ];
        Ok(Self {
            ctx,
            stream,
            bufs,
            slots: [(GU, PAD, need.gu), (ACT, PAD, need.act)],
            shape,
        })
    }

    fn call(&self) -> Result<SwigluRowsCall<'_>, EngError> {
        let range = |(buffer, offset, bytes): Slot| self.bufs[buffer].range(offset, bytes);
        Ok(SwigluRowsCall {
            shape: self.shape,
            gu: range(self.slots[GU])?,
            act: range(self.slots[ACT])?,
        })
    }

    /// Queues the call the slots describe.
    fn attempt(&self) -> Result<(), EngError> {
        self.ctx.swiglu_rows(&self.stream, self.call()?)
    }

    /// Queues the call and drains the stream.
    fn run(&self) -> Result<(), EngError> {
        self.attempt()?;
        self.stream.synchronize()
    }

    /// The whole of buffer `index`.
    fn bytes(&self, index: usize) -> Result<Vec<u8>, EngError> {
        let mut out = vec![0u8; self.bufs[index].bytes()];
        self.ctx.download(&self.bufs[index], &mut out)?;
        Ok(out)
    }

    fn reset_act(&self) -> Result<(), EngError> {
        let fill = vec![ACT_FILL; self.bufs[ACT].bytes()];
        self.bufs[ACT].write_at(&self.stream, 0, &fill)
    }
}

/// One good launch of `shape` on `values`; returns the whole act buffer and the whole gu buffer.
fn launch(shape: SwigluRowsShape, values: &[f32]) -> Result<(Vec<u8>, Vec<u8>), EngError> {
    let rig = Rig::new(shape, values)?;
    rig.run()?;
    let act = rig.bytes(ACT)?;
    let gu = rig.bytes(GU)?;
    drop(rig);
    Ok((act, gu))
}

/// The largest graded SW0 ratio of a run, where it fell, and how many values were graded.
#[derive(Debug, Default, Clone, Copy)]
struct Worst {
    rel: f64,
    g: f32,
    u: f32,
    graded: usize,
}

/// Lane C's merge gate, run after the device phase and before any bound check: EVERY output value
/// (row r < rows, column c < n_ff of an act range that starts at PAD) is finite. An output the
/// kernel never wrote keeps the NaN fill and fails here, so no max-of-deltas judge can drop it.
fn assert_finite(shape: &SwigluRowsShape, act: &[u8], what: &str) {
    let floats = testkit::floats(&act[PAD..]);
    for r in 0..shape.rows {
        for c in 0..shape.n_ff {
            let out = floats[r * shape.ld_out + c];
            assert!(
                out.is_finite(),
                "{what}: act row {r} column {c} is {out} (bits {:08x}), not finite",
                out.to_bits()
            );
        }
    }
}

/// Asserts the act buffer of one run of `shape` on `values`: first [`assert_finite`]; then every
/// byte of the buffer outside the outputs (before, between and after the rows) keeps the fill;
/// then an exact zero gives zero, and with |g| <= 12 and a normal exact value the output is
/// inside SW0 of the f64 reference. Returns the worst graded ratio.
fn assert_act(shape: &SwigluRowsShape, values: &[f32], act: &[u8], what: &str) -> Worst {
    assert_finite(shape, act, what);
    assert!(
        act[..PAD].iter().all(|&byte| byte == ACT_FILL),
        "{what}: a byte before act changed"
    );
    let floats = testkit::floats(&act[PAD..]);
    let mut written = vec![false; floats.len()];
    for r in 0..shape.rows {
        for c in 0..shape.n_ff {
            written[r * shape.ld_out + c] = true;
        }
    }
    let fill = u32::from_le_bytes([ACT_FILL; 4]);
    for (at, value) in floats.iter().enumerate() {
        if !written[at] {
            assert_eq!(
                value.to_bits(),
                fill,
                "{what}: act float {at} is not an output"
            );
        }
    }
    let mut worst = Worst::default();
    for r in 0..shape.rows {
        for c in 0..shape.n_ff {
            let g = values[r * shape.ld_in + c];
            let u = values[r * shape.ld_in + shape.up_off + c];
            let out = floats[r * shape.ld_out + c];
            let exact = reference(g, u);
            if exact == 0.0 {
                assert_eq!(out, 0.0, "{what}: act[{r}][{c}], g {g}, u {u}");
                continue;
            }
            if f64::from(g).abs() > SW0_GATE || exact.abs() < f64::from(f32::MIN_POSITIVE) {
                continue;
            }
            let rel = (f64::from(out) - exact).abs() / exact.abs();
            assert!(
                rel <= SW0,
                "{what}: act[{r}][{c}] = {out} vs {exact} (g {g}, u {u}): relative {rel:e} > 2^-18"
            );
            worst.graded += 1;
            if rel > worst.rel {
                (worst.rel, worst.g, worst.u) = (rel, g, u);
            }
        }
    }
    worst
}

fn report(what: &str, worst: &Worst) {
    println!(
        "SWIGLU-REPORT {what}: SW0 max relative error {:.4e} = 2^{:.3} at g {:e}, u {:e}, over {} \
         graded values (bound 2^-18)",
        worst.rel,
        worst.rel.log2(),
        worst.g,
        worst.u,
        worst.graded
    );
}

/// Test (1) and the 16,384-row shape of (3): the Qwen MoE shape at the 8M = 16,384-row cap, gates
/// uniform over SW0's whole domain [-12, 12), up values +-2^[-4, 4). Every value is graded.
#[test]
fn swiglu_rows_matches_the_f64_reference_within_sw0() {
    if !testkit::device_present("swiglu_rows_matches_the_f64_reference_within_sw0") {
        return;
    }
    let shape = qwen(MAX_ROWS);
    let mut rng = Rng::new(0x5a_0001);
    let values = gu_values(&shape, &mut rng, &|r| gate(r, SW0_GATE), &up);
    let (act, _) = launch(shape, &values).expect("the device phase");
    // The draws span the domain (the LCG-scale lesson): the gates reach within 0.1% of both ends.
    let gates = values
        .as_chunks::<1_024>()
        .0
        .iter()
        .flat_map(|row| &row[..512]);
    let (low, high) = gates.fold((0f32, 0f32), |(lo, hi), &g| (lo.min(g), hi.max(g)));
    assert!(low < -11.988 && high > 11.988, "gates span [{low}, {high}]");
    let worst = assert_act(&shape, &values, &act, "qwen 16,384 rows");
    assert_eq!(worst.graded, MAX_ROWS * 512, "every value is graded");
    report("test 1, qwen 16,384 rows, |g| < 12", &worst);
}

/// Normal inputs where flush-to-zero or the approximate division's large-denominator rule can
/// fire: a subnormal gate, a subnormal up value, normal inputs with a subnormal exact value, gates
/// near -88 whose `1 + exp(-g)` lies above 2^126, and a gate of 88 whose `exp(-g)` is subnormal.
const FTZ_CASES: [(f32, f32); 8] = [
    (1.0e-40, 1.5),
    (-1.0e-40, 2.0),
    (1.5, 1.0e-40),
    (-3.0, -1.0e-40),
    (1.0e-20, 1.0e-20),
    (-87.5, 0.0625),
    (-88.0, 4.0),
    (88.0, 1.0),
];

/// The finiteness clause of SW0 and the FTZ report: gates uniform in [-88, 88), with
/// [`FTZ_CASES`] in the first columns of row 0. Every output is finite; the |g| <= 12 normal
/// values are graded; the rest is reported.
#[test]
fn swiglu_rows_is_finite_to_88_and_reports_the_ftz_cases() {
    if !testkit::device_present("swiglu_rows_is_finite_to_88_and_reports_the_ftz_cases") {
        return;
    }
    let shape = qwen(64);
    let mut rng = Rng::new(0x5a_0002);
    let mut values = gu_values(&shape, &mut rng, &|r| gate(r, FINITE_GATE), &up);
    for (c, &(g, u)) in FTZ_CASES.iter().enumerate() {
        (values[c], values[shape.up_off + c]) = (g, u);
    }
    let (act, _) = launch(shape, &values).expect("the device phase");
    let worst = assert_act(&shape, &values, &act, "gates to 88");
    report("test 1b, gates to 88, graded part", &worst);
    let floats = testkit::floats(&act[PAD..]);
    let (mut wide, mut zeroed) = (0f64, 0usize);
    for r in 0..shape.rows {
        for c in 0..shape.n_ff {
            let (g, u) = (values[r * 1_024 + c], values[r * 1_024 + 512 + c]);
            let (out, exact) = (f64::from(floats[r * 512 + c]), reference(g, u));
            let normal = exact.abs() >= f64::from(f32::MIN_POSITIVE);
            if normal && out == 0.0 {
                zeroed += 1;
            } else if normal && f64::from(g).abs() > SW0_GATE {
                wide = wide.max((out - exact).abs() / exact.abs());
            }
        }
    }
    println!(
        "SWIGLU-REPORT test 1b: 12 < |g| <= 88, ungraded max relative error {wide:.4e}; {zeroed} \
         normal exact values came out zero"
    );
    for (c, &(g, u)) in FTZ_CASES.iter().enumerate() {
        let (out, exact) = (floats[c], reference(g, u));
        println!(
            "SWIGLU-REPORT ftz g {g:e} u {u:e}: exact {exact:e}, out {out:e} (bits {:08x}), \
             absolute error {:e}",
            out.to_bits(),
            (f64::from(out) - exact).abs()
        );
    }
}

/// Test (2) with the 7-row shape of (3): a gap of 8 floats between the gate and the up half, a
/// gap of 8 after the up half (ld_in 88 > up_off + n_ff = 80), act rows of 40 for 36 outputs, NaN
/// in every gu gap, and gate and up drawn from different ranges so that swapped halves differ.
/// The gaps of act and the bytes around it keep the fill; gu is unchanged.
#[test]
fn swiglu_rows_reads_its_layout_and_writes_only_act() {
    if !testkit::device_present("swiglu_rows_reads_its_layout_and_writes_only_act") {
        return;
    }
    let shape = shape(7, 36, 44, 88, 40);
    let mut rng = Rng::new(0x5a_0003);
    let values = gu_values(&shape, &mut rng, &|r| gate(r, 6.0), &|r| {
        (2.0 + r.unit()) as f32
    });
    let (act, gu) = launch(shape, &values).expect("the device phase");
    let worst = assert_act(&shape, &values, &act, "layout 7 rows");
    assert_eq!(worst.graded, 7 * 36);
    report("test 2, layout 7 rows", &worst);
    let need = needs(&shape).expect("sizes");
    assert_eq!(gu, gu_bytes(&values, &need), "gu changed");
}

/// Test (3): one row at the narrowest and the widest n_ff, and seven rows at the widest n_ff with
/// gaps after both halves and after the outputs.
#[test]
fn swiglu_rows_takes_one_row_and_the_extreme_widths() {
    if !testkit::device_present("swiglu_rows_takes_one_row_and_the_extreme_widths") {
        return;
    }
    let cases = [
        ("1 row, n_ff 4", shape(1, 4, 4, 8, 4)),
        (
            "1 row, n_ff 32768",
            shape(1, MAX_FF, MAX_FF, 2 * MAX_FF, MAX_FF),
        ),
        (
            "7 rows, n_ff 32768 with gaps",
            shape(7, MAX_FF, MAX_FF + 4, 2 * MAX_FF + 12, MAX_FF + 8),
        ),
    ];
    let mut rng = Rng::new(0x5a_0004);
    let data: Vec<Vec<f32>> = cases
        .iter()
        .map(|(_, s)| gu_values(s, &mut rng, &|r| gate(r, SW0_GATE), &up))
        .collect();
    let phase = || -> Result<Vec<Vec<u8>>, EngError> {
        let mut acts = Vec::new();
        for ((_, s), values) in cases.iter().zip(&data) {
            acts.push(launch(*s, values)?.0);
        }
        Ok(acts)
    };
    let acts = phase().expect("the device phase");
    for (((what, s), values), act) in cases.iter().zip(&data).zip(&acts) {
        let worst = assert_act(s, values, act, what);
        assert_eq!(worst.graded, s.rows * s.n_ff, "{what}");
        report(&format!("test 3, {what}"), &worst);
    }
}

/// Test (4): two runs of the same call, with act reset to the fill between them, are bit-equal.
#[test]
fn swiglu_rows_is_deterministic() {
    if !testkit::device_present("swiglu_rows_is_deterministic") {
        return;
    }
    let shape = qwen(2_048);
    let mut rng = Rng::new(0x5a_0005);
    let values = gu_values(&shape, &mut rng, &|r| gate(r, SW0_GATE), &up);
    let phase = || -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let rig = Rig::new(shape, &values)?;
        rig.run()?;
        let first = rig.bytes(ACT)?;
        rig.reset_act()?;
        rig.run()?;
        let second = rig.bytes(ACT)?;
        drop(rig);
        Ok((first, second))
    };
    let (first, second) = phase().expect("the device phase");
    assert_finite(&shape, &second, "the second run");
    assert_act(&shape, &values, &first, "the first run");
    assert!(first == second, "two identical runs differ");
}

// ---- the refusal tables (test 6) ------------------------------------------------------------

/// The base call of the refusal tables: 8 rows, n_ff 32, gaps after both halves and the outputs.
/// Its extents are 2,528 bytes of gu and 1,136 of act.
fn refusal_shape() -> SwigluRowsShape {
    shape(8, 32, 40, 80, 36)
}

/// One refusal case: a label, the text the error must contain, and the edit that breaks exactly
/// one rule while every earlier check still passes.
struct Case {
    label: String,
    text: String,
    edit: Box<dyn Fn(&mut Rig)>,
}

fn case(label: &str, text: &str, edit: impl Fn(&mut Rig) + 'static) -> Case {
    Case {
        label: label.to_owned(),
        text: text.to_owned(),
        edit: Box::new(edit),
    }
}

/// The outcome of every case of a table, in order.
type Outcomes = Vec<Result<(), EngError>>;

/// Hashes of both buffers after the first good call, after the refusals and after a last good
/// call, the outcome of every case, and the act buffer of the last good call.
type Refusals = ([u64; 2], [u64; 2], [u64; 2], Outcomes, Vec<u8>);

/// The device phase of a refusal table: a good call, every case on the base call, then a good
/// call again. Owners are dropped before any assertion.
fn refusal_phase(cases: &[Case]) -> Result<Refusals, EngError> {
    let shape = refusal_shape();
    let values = gu_values(&shape, &mut Rng::new(0x5a_0006), &|r| gate(r, 6.0), &up);
    let mut rig = Rig::new(shape, &values)?;
    let hashes = |rig: &Rig| -> Result<[u64; 2], EngError> {
        Ok([
            testkit::fnv1a64(&rig.bytes(GU)?),
            testkit::fnv1a64(&rig.bytes(ACT)?),
        ])
    };
    rig.run()?;
    let first = hashes(&rig)?;
    let (base_shape, base_slots) = (rig.shape, rig.slots);
    let mut outcomes = Vec::new();
    for case in cases {
        (case.edit)(&mut rig);
        outcomes.push(rig.attempt());
        (rig.shape, rig.slots) = (base_shape, base_slots);
    }
    rig.stream.synchronize()?;
    let after = hashes(&rig)?;
    rig.run()?;
    let last = hashes(&rig)?;
    let act = rig.bytes(ACT)?;
    drop(rig);
    Ok((first, after, last, outcomes, act))
}

/// Runs a table and asserts it: each case refused with its own text, no buffer byte changed by
/// the refusals, and the good call afterwards still accepted and equal.
fn assert_refusals(cases: Vec<Case>) {
    let (first, after, last, outcomes, act) = refusal_phase(&cases).expect("the device phase");
    assert_finite(&refusal_shape(), &act, "the good call after the refusals");
    assert_eq!(first, after, "a refused call changed a buffer");
    assert_eq!(first, last, "the good call after the refusals differs");
    for (case, outcome) in cases.iter().zip(&outcomes) {
        assert!(
            matches!(outcome, Err(EngError::BadRequest(text)) if text.contains(case.text.as_str())),
            "{}: {outcome:?}",
            case.label
        );
    }
}

#[test]
fn swiglu_rows_refuses_shapes_the_kernel_refuses() {
    if !testkit::device_present("swiglu_rows_refuses_shapes_the_kernel_refuses") {
        return;
    }
    const BIG: usize = 1 << 31;
    let tag = |rule: &str, detail: &str| format!("swiglu_rows: {rule}: {detail}");
    let cases = vec![
        case("rows 0", &tag("rows", "rows is 0, not 1 to 16384"), |r| {
            r.shape.rows = 0
        }),
        case("rows 16385", &tag("rows", "rows is 16385, not 1 to"), |r| {
            r.shape.rows = MAX_ROWS + 1
        }),
        case("n_ff 0", &tag("n_ff", "n_ff is 0, not 4 to 32768"), |r| {
            r.shape.n_ff = 0
        }),
        case("n_ff 32772", &tag("n_ff", "n_ff is 32772, not 4 to"), |r| {
            r.shape.n_ff = MAX_FF + 4
        }),
        case(
            "n_ff 30",
            &tag("n_ff", "n_ff is 30, not a multiple of 4"),
            |r| r.shape.n_ff = 30,
        ),
        case(
            "up_off 42",
            &tag("up_off", "up_off is 42, not a multiple"),
            |r| r.shape.up_off = 42,
        ),
        case(
            "up_off 28",
            &tag("up_off", "up_off is 28, below n_ff 32"),
            |r| r.shape.up_off = 28,
        ),
        case(
            "ld_in 82",
            &tag("ld_in", "ld_in is 82, not a multiple"),
            |r| r.shape.ld_in = 82,
        ),
        case(
            "ld_in 68",
            &tag("ld_in", "ld_in is 68, below up_off + n_ff 72"),
            |r| r.shape.ld_in = 68,
        ),
        case(
            "ld_out 38",
            &tag("ld_out", "ld_out is 38, not a multiple"),
            |r| r.shape.ld_out = 38,
        ),
        case(
            "ld_out 28",
            &tag("ld_out", "ld_out is 28, below n_ff 32"),
            |r| r.shape.ld_out = 28,
        ),
        case(
            "up_off + n_ff wraps",
            &tag("up_off", "up_off + n_ff overflows"),
            |r| r.shape.up_off = usize::MAX - 3,
        ),
        case(
            "up_off 2^31",
            &tag("up_off", "up_off is 2147483648, above i32::MAX"),
            |r| (r.shape.up_off, r.shape.ld_in) = (BIG, BIG + 64),
        ),
        case(
            "ld_in 2^31",
            &tag("ld_in", "ld_in is 2147483648, above i32::MAX"),
            |r| r.shape.ld_in = BIG,
        ),
        case(
            "ld_out 2^31",
            &tag("ld_out", "ld_out is 2147483648, above i32::MAX"),
            |r| r.shape.ld_out = BIG,
        ),
    ];
    assert_refusals(cases);
}

#[test]
fn swiglu_rows_refuses_short_and_misaligned_ranges() {
    if !testkit::device_present("swiglu_rows_refuses_short_and_misaligned_ranges") {
        return;
    }
    let mut cases = Vec::new();
    for (slot, name) in [(GU, "gu"), (ACT, "act")] {
        let short = format!("the CUDA {name} range holds");
        cases.push(case(&format!("{name} short"), &short, move |r| {
            r.slots[slot].2 -= 4
        }));
        // +4 and +8 keep the byte count and pass the size rule; +8 proves the rule is 16 bytes.
        for by in [4, 8] {
            let text = format!("the CUDA {name} range starts at offset {}", PAD + by);
            cases.push(case(&format!("{name} at +{by}"), &text, move |r| {
                r.slots[slot].1 += by
            }));
        }
    }
    assert_refusals(cases);
}

/// The overlap rule on the extents, then the adjacent placement it must accept.
#[test]
fn swiglu_rows_refuses_act_over_gu_and_takes_act_beside_it() {
    if !testkit::device_present("swiglu_rows_refuses_act_over_gu_and_takes_act_beside_it") {
        return;
    }
    let base = refusal_shape();
    let need = needs(&base).expect("sizes");
    let text = "the CUDA act and gu ranges overlap";
    let act = need.act;
    let cases = vec![
        case("act on gu", text, move |r| r.slots[ACT] = (GU, PAD, act)),
        case("act over the last 16 bytes of gu", text, move |r| {
            r.slots[ACT] = (GU, PAD + need.gu - 16, act)
        }),
        // Row 0's gate ends at column 32 and its up half starts at column 40: a gap the kernel
        // never reads is still inside the gu extent the rule claims.
        case("act starting in a gap of gu", text, move |r| {
            r.slots[ACT] = (GU, PAD + 128, act)
        }),
        case("gu on act", text, move |r| {
            r.slots[GU] = (ACT, PAD, need.gu)
        }),
    ];
    assert_refusals(cases);
    // act right after the gu extent, in the same buffer, shares no byte with it: accepted. The
    // slack after the gu extent is first set to the act fill, so the canary check applies.
    let values = gu_values(&base, &mut Rng::new(0x5a_0008), &|r| gate(r, 6.0), &up);
    let phase = || -> Result<Vec<u8>, EngError> {
        let mut rig = Rig::new(base, &values)?;
        let fill = vec![ACT_FILL; SLACK];
        rig.bufs[GU].write_at(&rig.stream, PAD + need.gu, &fill)?;
        rig.slots[ACT] = (GU, PAD + need.gu, need.act);
        rig.run()?;
        let gu = rig.bytes(GU)?;
        drop(rig);
        Ok(gu)
    };
    let gu = phase().expect("the device phase");
    let (extent, after) = gu.split_at(PAD + need.gu);
    let mut view = vec![ACT_FILL; PAD];
    view.extend_from_slice(after);
    let worst = assert_act(&base, &values, &view, "adjacent act");
    assert_eq!(worst.graded, 8 * 32);
    assert_eq!(
        extent,
        &gu_bytes(&values, &need)[..PAD + need.gu],
        "gu changed"
    );
}

#[test]
fn swiglu_rows_refuses_foreign_buffers_and_streams() {
    if !testkit::device_present("swiglu_rows_refuses_foreign_buffers_and_streams") {
        return;
    }
    let shape = refusal_shape();
    let values = gu_values(&shape, &mut Rng::new(0x5a_0009), &|r| gate(r, 6.0), &up);
    let phase = || -> Result<(Outcomes, Vec<u8>), EngError> {
        let rig = Rig::new(shape, &values)?;
        let other = Rig::new(shape, &values)?;
        let mut outcomes = Vec::new();
        for slot in [GU, ACT] {
            let (_, offset, bytes) = other.slots[slot];
            let foreign = other.bufs[slot].range(offset, bytes)?;
            let mut call = rig.call()?;
            if slot == GU {
                call.gu = foreign;
            } else {
                call.act = foreign;
            }
            outcomes.push(rig.ctx.swiglu_rows(&rig.stream, call));
        }
        outcomes.push(rig.ctx.swiglu_rows(&other.stream, rig.call()?));
        outcomes.push(rig.run());
        let act = rig.bytes(ACT)?;
        drop(other);
        drop(rig);
        Ok((outcomes, act))
    };
    let (outcomes, act) = phase().expect("the device phase");
    assert!(outcomes[3].is_ok(), "the good call: {:?}", outcomes[3]);
    assert_finite(&shape, &act, "the good call");
    for (index, text) in ["buffer belongs", "buffer belongs", "stream belongs"]
        .iter()
        .enumerate()
    {
        assert!(
            matches!(&outcomes[index], Err(EngError::BadRequest(m)) if m.contains(text)),
            "case {index}: {:?}",
            outcomes[index]
        );
    }
}

// ---- lane C's rows: swiglu_q8 and NaN/inf ---------------------------------------------------

/// The most two SW0-bounded values of one exact product differ by, relative to either one:
/// 2 * 2^-18 / (1 - 2^-18).
const ETA: f64 = 2.0 * SW0 / (1.0 - SW0);
/// One f32 rounding: 2^-24.
const ROUND: f64 = 1.0 / 16_777_216.0;

/// What the swiglu_q8 comparison saw.
#[derive(Debug, Default)]
struct Tally {
    blocks: usize,
    equal: usize,
    d: usize,
    s: usize,
    codes: usize,
    distance: f64,
}

/// The value of the fp16 rounding (to nearest, ties to even) of `v`.
fn half_of(v: f64) -> f64 {
    f64::from(testkit::f16_to_f32(testkit::f64_to_f16_bits(v)))
}

/// Asserts one stored swiglu_q8 block against the window around swiglu_rows' values `x` (the
/// derivation is in the doc of the test below) and counts how it differs from `q8_ref::block(x)`.
fn assert_block_in_window(x: &[f32; 32], stored: &[u8], block: usize, tally: &mut Tally) {
    let host = q8_ref::block(x);
    tally.blocks += 1;
    if host[..] == stored[..36] {
        tally.equal += 1;
        return;
    }
    let half = |at: usize| {
        f64::from(testkit::f16_to_f32(u16::from_le_bytes([
            stored[at],
            stored[at + 1],
        ])))
    };
    let amax = x.iter().fold(0f64, |m, v| m.max(f64::from(v.abs())));
    let d = half(0);
    let (d_lo, d_hi) = (
        amax / 127.0 * (1.0 - ETA) * (1.0 - ROUND),
        amax / 127.0 * (1.0 + ETA) * (1.0 + ROUND),
    );
    assert!(
        half_of(d_lo) <= d && d <= half_of(d_hi),
        "block {block}: d {d} outside [{d_lo}, {d_hi}]"
    );
    let mut lanes = *x;
    butterfly32(&mut lanes);
    let slack = (ETA + 11.0 * ROUND) * x.iter().map(|v| f64::from(v.abs())).sum::<f64>();
    let (s, s_r) = (half(2), f64::from(lanes[0]));
    assert!(
        half_of(s_r - slack) <= s && s <= half_of(s_r + slack),
        "block {block}: s {s} outside {s_r} +- {slack}"
    );
    let low = (1.0 - ETA) * (1.0 - ROUND) / ((1.0 + ETA) * (1.0 + ROUND));
    let high = (1.0 + ETA) * (1.0 + ROUND) / ((1.0 - ETA) * (1.0 - ROUND));
    for (i, value) in x.iter().enumerate() {
        let q = f64::from(stored[4 + i] as i8);
        let r = f64::from(*value) * 127.0 / amax;
        let (a, b) = (r * low, r * high);
        assert!(
            a.min(b).round() <= q && q <= a.max(b).round(),
            "block {block} value {i}: code {q} outside the codes of [{a}, {b}]"
        );
        tally.codes += usize::from(stored[4 + i] != host[4 + i]);
        tally.distance = tally.distance.max((r - q).abs());
    }
    tally.d += usize::from(stored[..2] != host[..2]);
    tally.s += usize::from(stored[2..4] != host[2..4]);
}

/// Lane C's numerics row against the decode path's SwiGLU, `swiglu_q8` (MATVEC-PAR S3a K2:
/// `x / (1 + __expf(-x))` with IEEE division, no flush to zero, then the q8_1 block rule in IEEE
/// arithmetic). swiglu_q8 stores only q8_1 blocks, never its f32 value, so the f32 outputs cannot
/// be compared directly. The design: both kernels run on the SAME device gu bytes (the Qwen
/// layout, 2,048 rows; swiglu_q8 in launches of 8 slots), and the host passes swiglu_rows' f32
/// outputs x through `q8_ref::block`, the byte-exact host form of swiglu_q8's block rule. Where
/// both kernels compute the same f32 values, the blocks are byte-equal. Where they differ, the
/// difference is bounded: swiglu_rows is inside SW0 (asserted here as well), and swiglu_q8's
/// value is inside the same 2^-18 for |g| <= 12 (`__expf`'s 2 + floor(1.173 * 12) = 16 ulp, plus
/// 0.5 for the add, 0.5 for the IEEE division and 0.5 for the product: 17.5 ulp = 2^-18.87). So
/// each of its values lies within ETA = 2 * 2^-18 / (1 - 2^-18) of x, relative. With one f32
/// rounding ROUND = 2^-24 per operation, every stored field must lie in a window around x:
/// - d = fp16(f32(amax / 127)): between fp16(amax / 127 * (1 - ETA)(1 - ROUND)) and
///   fp16(amax / 127 * (1 + ETA)(1 + ROUND));
/// - each code q = roundf(f32(x8 / d8)): between the roundings of r * (1 - ETA)(1 - ROUND) /
///   ((1 + ETA)(1 + ROUND)) and r * (1 + ETA)(1 + ROUND) / ((1 - ETA)(1 - ROUND)), r =
///   127 * x / amax;
/// - s = fp16 of the butterfly sum: within (ETA + 11 ROUND) * sum|x| of the host butterfly sum
///   of x (two five-level sums plus the value difference), then fp16-rounded.
///
/// A swapped half, a sigmoid, a dropped column or any error well above 2^-18 moves codes outside
/// their windows; an error inside SW0 can only flip a code whose r sits within about
/// 127 * 2 * ETA = 0.002 of a rounding boundary. The test reports how many blocks were
/// byte-equal and how many fields differed.
#[test]
fn swiglu_rows_reproduces_swiglu_q8_through_its_block_rule() {
    if !testkit::device_present("swiglu_rows_reproduces_swiglu_q8_through_its_block_rule") {
        return;
    }
    let rows = 2_048;
    let shape = qwen(rows);
    let values = gu_values(
        &shape,
        &mut Rng::new(0x5a_0007),
        &|r| gate(r, SW0_GATE),
        &up,
    );
    let phase = || -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let rig = Rig::new(shape, &values)?;
        rig.run()?;
        let need8 = swiglu_q8_needs(512, 1_024, 8)
            .ok_or_else(|| EngError::BadRequest("swiglu_q8 sizes".to_owned()))?;
        let q8 = rig.ctx.upload(&vec![ACT_FILL; rows / 8 * need8.q8])?;
        for group in 0..rows / 8 {
            let call = SwigluQ8Call {
                gu: rig.bufs[GU].range(PAD + group * 8 * 1_024 * 4, need8.gu)?,
                q8: q8.range(group * need8.q8, need8.q8)?,
                n_ff: 512,
                y_slot_stride: 1_024,
                slots: 8,
            };
            rig.ctx.swiglu_q8(&rig.stream, call)?;
        }
        rig.stream.synchronize()?;
        let act = rig.bytes(ACT)?;
        let mut stored = vec![0u8; q8.bytes()];
        rig.ctx.download(&q8, &mut stored)?;
        drop(q8);
        drop(rig);
        Ok((act, stored))
    };
    let (act, stored) = phase().expect("the device phase");
    // The finiteness gate on both outputs: swiglu_rows' values, and the d and s halves of every
    // stored block (the 0xff fill is an fp16 NaN, so a block swiglu_q8 skipped fails here).
    assert_finite(&shape, &act, "the swiglu_q8 row");
    for (block, bytes) in stored.as_chunks::<36>().0.iter().enumerate() {
        for (field, at) in [("d", 0), ("s", 2)] {
            let half = testkit::f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
            assert!(
                half.is_finite(),
                "swiglu_q8 row {} block {}: {field} is {half}, not finite",
                block / 16,
                block % 16
            );
        }
    }
    let worst = assert_act(&shape, &values, &act, "the swiglu_q8 row");
    let x = testkit::floats(&act[PAD..PAD + rows * 512 * 4]);
    let mut tally = Tally::default();
    let blocks = x.as_chunks::<32>().0.iter().zip(stored.as_chunks::<36>().0);
    for (block, (chunk, bytes)) in blocks.enumerate() {
        assert_block_in_window(chunk, bytes, block, &mut tally);
    }
    assert_eq!(tally.blocks, rows * 16, "every block compared");
    report("swiglu_q8 row, swiglu_rows' values", &worst);
    println!(
        "SWIGLU-REPORT swiglu_q8 row: {} of {} blocks byte-equal; {} d, {} s and {} code fields \
         differ, every one inside its window; max |r - q| {:.6}",
        tally.equal, tally.blocks, tally.d, tally.s, tally.codes, tally.distance
    );
}

/// The (g, u) inputs of the pass-through row: NaN and infinities in either half.
const SPECIALS: [(f32, f32); 18] = [
    (f32::NAN, 1.5),
    (f32::NAN, 0.0),
    (f32::INFINITY, 1.5),
    (f32::INFINITY, -2.0),
    (f32::INFINITY, 0.0),
    (f32::NEG_INFINITY, 1.5),
    (f32::NEG_INFINITY, 0.0),
    (1.5, f32::NAN),
    (-3.0, f32::NAN),
    (0.0, f32::NAN),
    (1.5, f32::INFINITY),
    (-3.0, f32::INFINITY),
    (0.0, f32::INFINITY),
    (1.5, f32::NEG_INFINITY),
    (-3.0, f32::NEG_INFINITY),
    (0.0, f32::NEG_INFINITY),
    (-100.0, f32::INFINITY),
    (f32::NAN, f32::INFINITY),
];

/// The NaN every NaN result of the device carries: the canonical 0x7fffffff of PTX f32
/// arithmetic, whatever the payload of the input NaN.
const DEVICE_NAN: u32 = 0x7fff_ffff;

/// Lane C's pass-through row: NaN and infinities in g or u come out as IEEE evaluates the
/// expression `g / (1 + exp(-g)) * u` in f32 (exp(+inf) = inf, -inf / inf = NaN, 0 * inf = NaN,
/// inf * u = +-inf), asserted bit for bit: an infinity with its sign, a NaN as the device's
/// canonical NaN.
#[test]
fn swiglu_rows_passes_nan_and_inf_through() {
    if !testkit::device_present("swiglu_rows_passes_nan_and_inf_through") {
        return;
    }
    let shape = shape(1, 20, 20, 40, 20);
    let mut values = vec![1.0f32; 40];
    for (c, &(g, u)) in SPECIALS.iter().enumerate() {
        (values[c], values[20 + c]) = (g, u);
    }
    let (act, _) = launch(shape, &values).expect("the device phase");
    let out = testkit::floats(&act[PAD..PAD + 20 * 4]);
    // The two padding columns (g = u = 1) are ordinary outputs: finite.
    for (c, value) in out.iter().enumerate().skip(SPECIALS.len()) {
        assert!(value.is_finite(), "row 0 column {c} is {value}, not finite");
    }
    for (c, &(g, u)) in SPECIALS.iter().enumerate() {
        let ieee = g / (1.0 + (-g).exp()) * u;
        let want = if ieee.is_nan() {
            DEVICE_NAN
        } else {
            ieee.to_bits()
        };
        println!(
            "SWIGLU-REPORT special g {g} u {u}: out bits {:08x}, IEEE {ieee}",
            out[c].to_bits()
        );
        assert_eq!(out[c].to_bits(), want, "g {g}, u {u}: out {}", out[c]);
    }
    assert!(
        act[..PAD].iter().all(|&b| b == ACT_FILL) && act[PAD + 80..].iter().all(|&b| b == ACT_FILL),
        "the bytes around act changed"
    );
}

// ---- pure: sizes and shape rules ------------------------------------------------------------

#[test]
fn swiglu_rows_needs_are_exact() {
    // The Qwen MoE shape at the 16,384-row cap: the whole [rows][1,024] gu and [rows][512] act.
    assert_eq!(
        swiglu_rows_needs(&qwen(MAX_ROWS)),
        Some(SwigluRowsNeeds {
            gu: 67_108_864,
            act: 33_554_432
        })
    );
    // The layout shape of test 2: the last row ends at its up half (6 * 88 + 80 floats) and at
    // its 36th output (6 * 40 + 36 floats); the trailing gaps are outside the extents.
    assert_eq!(
        swiglu_rows_needs(&shape(7, 36, 44, 88, 40)),
        Some(SwigluRowsNeeds {
            gu: 2_432,
            act: 1_104
        })
    );
    assert_eq!(
        swiglu_rows_needs(&shape(1, 4, 4, 8, 4)),
        Some(SwigluRowsNeeds { gu: 32, act: 16 })
    );
    assert_eq!(swiglu_rows_needs(&shape(0, 4, 4, 8, 4)), None);
    assert_eq!(swiglu_rows_needs(&shape(usize::MAX, 4, 4, 8, 4)), None);
    // Wrap-to-small: two rows of ld_in usize::MAX - 3 make (MAX - 3) + 8 floats, which a
    // wrapping form would answer as 4 floats (16 bytes); only the checked add refuses.
    assert_eq!(swiglu_rows_needs(&shape(2, 4, 4, usize::MAX - 3, 4)), None);
    assert_eq!(swiglu_rows_needs(&shape(1, 4, usize::MAX - 3, 8, 4)), None);
    assert_eq!(swiglu_rows_needs(&shape(1, 4, usize::MAX / 4, 8, 4)), None);
}

#[test]
fn swiglu_rows_shape_rules_are_pinned() {
    let ok = |s: SwigluRowsShape| check_shape(&s).is_ok();
    let big = (1usize << 31) - 4;
    assert!(ok(shape(1, 4, 4, 8, 4)) && ok(shape(MAX_ROWS, MAX_FF, MAX_FF, 2 * MAX_FF, MAX_FF)));
    assert!(ok(shape(1, 4, big - 4, big, big)));
    assert!(!ok(shape(0, 4, 4, 8, 4)) && !ok(shape(MAX_ROWS + 1, 4, 4, 8, 4)));
    assert!(
        !ok(shape(1, 0, 4, 8, 4))
            && !ok(shape(1, MAX_FF + 4, MAX_FF + 4, 2 * MAX_FF + 8, MAX_FF + 4))
    );
    assert!(!ok(shape(1, 6, 8, 16, 8)) && !ok(shape(1, 4, 6, 12, 4)) && !ok(shape(1, 8, 4, 12, 8)));
    assert!(!ok(shape(1, 4, 4, 10, 4)) && !ok(shape(1, 4, 8, 8, 4)));
    assert!(!ok(shape(1, 4, 4, 8, 6)) && !ok(shape(1, 8, 8, 16, 4)));
    assert!(!ok(shape(1, 4, 4, big + 4, 4)) && !ok(shape(1, 4, 4, 8, big + 4)));
    let ints = check_shape(&shape(7, 36, 44, 88, 40)).expect("a valid shape");
    assert_eq!(
        (ints.rows, ints.n_ff, ints.up_off, ints.ld_in, ints.ld_out),
        (7, 36, 44, 88, 40)
    );
}
