//! Device tests (3090) of the step parameter wrapper at seam 1: the Rust wrapper against the C
//! entries against the kernels. Every output is an integer word, so the check is word equality
//! against a host model (class E1x, bit exact). The tests run with `--test-threads=1`.
//!
//! The host models are written here from the rules of the packet, not from the kernel:
//! `cpu_argmax` is a literal copy of `family/sampler.rs::argmax` (`kernels::cuda` must not depend
//! on `family`), `cpu_rejects` is the predicate of its `check_logits`, `ref_row` is the device
//! rule for a row `check_logits` rejects, and `sim_advance` and `sim_begin` are the step words.
//! Argued, not tested: (0) T13, the latched context, per lane C's ruling: `fault.rs` is lane C's,
//! and lane C adds one resident-wrapper call to its fault child in a later packet. (1) T9's freed-buffer row: a `DeviceRange` borrows its buffer, so the
//! compiler forbids freeing the buffer early. (2) T13, the latched context: every resident
//! wrapper enters through `launch_on`, whose `ensure_usable` refusal after a latch is covered by
//! the fault child of `fault.rs` (feature `cuda-fault-qualification`); lane C adds one
//! resident-wrapper call to that child in a later packet. (3) T10 (graph capture) is deferred to
//! RD-11a, which must list it as owed (`write_at` and `GraphExec` are absent at bc7f391).

use std::mem::{offset_of, size_of};

use super::{
    ARGMAX_PART_BYTES, ArgmaxCall, MAX_ADVANCE, MAX_LOGITS, OFFSET_KV_LEN, OFFSET_KV_SLOT,
    OFFSET_POS, OFFSET_STATUS, OFFSET_TOKEN_ID, POS_MAX, RESULT_PAIR_BYTES, ResultPair,
    STATUS_INVALID_STEP_STATE, STATUS_NON_FINITE_LOGITS, STATUS_ROUTER_FAULT, STEP_PARAMS_BYTES,
    StepParams, StepStatus, WORD_MAX, check_n, ensure_rows,
};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const SENTINEL: i32 = 0x1111_1111;
const PAD_SENTINEL: [i32; 3] = [0x2222_2222, 0x3333_3333, 0x4444_4444];
const NON_FINITE: &str = "logits hold a non-finite value";

/// A valid block at `pos` whose padding is a sentinel, so a stray write shows.
fn block(token_id: i32, pos: i32, status: i32) -> StepParams {
    StepParams {
        token_id,
        pos,
        kv_len: pos + 1,
        kv_slot: pos,
        status,
        pad: PAD_SENTINEL,
    }
}

/// Literal copy of `family/sampler.rs::argmax`.
fn cpu_argmax(logits: &[f32]) -> u32 {
    let mut best_index = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    u32::try_from(best_index).expect("row index")
}

/// The predicate of `check_logits` that applies to a non-empty row.
fn cpu_rejects(logits: &[f32]) -> bool {
    logits.is_empty() || logits.iter().any(|logit| !logit.is_finite())
}

/// The device rule: a NaN never wins, the largest value wins, the lowest index on a tie, and a
/// row with no comparable value gives -1.
fn ref_row(logits: &[f32]) -> i32 {
    let mut best: Option<(f32, usize)> = None;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_nan() && best.is_none_or(|(top, _)| value > top) {
            best = Some((value, index));
        }
    }
    best.map_or(-1, |(_, index)| i32::try_from(index).expect("row index"))
}

fn valid_state(p: &StepParams) -> bool {
    let pos = i64::from(p.pos);
    (0..=i64::from(POS_MAX)).contains(&pos)
        && i64::from(p.kv_len) == pos + 1
        && i64::from(p.kv_slot) == pos
}

/// `step_begin`: the status is 0, or 4 for an invalid state; nothing else changes.
fn sim_begin(p: StepParams) -> StepParams {
    StepParams {
        status: if valid_state(&p) { 0 } else { 4 },
        ..p
    }
}

/// `step_advance`: all three words move, or all three become -1; the others never change.
fn sim_advance(p: StepParams, m: i32) -> StepParams {
    let moved = i64::from(p.pos) + i64::from(m);
    let (pos, len, slot) = if valid_state(&p) && moved <= i64::from(POS_MAX) {
        (p.pos + m, p.kv_len + m, p.kv_slot + m)
    } else {
        (-1, -1, -1)
    };
    StepParams {
        pos,
        kv_len: len,
        kv_slot: slot,
        ..p
    }
}

/// A row of finite values that are all below -1.
fn base_row(n: usize) -> Vec<f32> {
    (0..n).map(|i| -1.0 - (i % 7) as f32 * 0.25).collect()
}

fn random_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * 3.0) as f32).collect()
}

fn bits(p: &StepParams) -> [u8; STEP_PARAMS_BYTES] {
    p.to_bytes()
}

/// `None` when the result is a `BadRequest` whose message contains `needle`.
fn bad_request_problem(result: Result<(), EngError>, case: &str, needle: &str) -> Option<String> {
    match result {
        Err(EngError::BadRequest(message)) if message.contains(needle) => None,
        other => Some(format!(
            "{case}: expected BadRequest containing {needle:?}, got {other:?}"
        )),
    }
}

fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &str) {
    if let Some(problem) = bad_request_problem(result, case, needle) {
        panic!("{problem}");
    }
}

fn assert_backend(error: Option<EngError>, operation: &str, message: Option<&str>, case: &str) {
    match error {
        Some(EngError::Backend {
            operation: got,
            message: text,
        }) => {
            assert_eq!(got, operation, "{case}: operation");
            if let Some(want) = message {
                assert_eq!(text, want, "{case}: message");
            }
        }
        other => panic!("{case}: expected Backend {operation}, got {other:?}"),
    }
}

/// One context and one stream.
struct Fx {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Fx {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }

    fn upload_params(&self, p: StepParams) -> DeviceBuffer {
        testkit::upload_bytes(&self.ctx, &bits(&p))
    }

    /// Waits for the stream, then reads the block back.
    fn read_params(&self, buffer: &DeviceBuffer) -> StepParams {
        self.stream.synchronize().expect("synchronize");
        let bytes = testkit::read_bytes(&self.ctx, buffer, STEP_PARAMS_BYTES);
        StepParams::from_bytes(bytes.try_into().expect("32 bytes"))
    }

    fn read_pair(&self, buffer: &DeviceBuffer, at: usize) -> ResultPair {
        self.stream.synchronize().expect("synchronize");
        let bytes = testkit::read_bytes(&self.ctx, buffer, at + RESULT_PAIR_BYTES);
        ResultPair::from_bytes(bytes[at..].try_into().expect("8 bytes"))
    }

    fn begin(&self, p: StepParams) -> StepParams {
        let buffer = self.upload_params(p);
        let range = buffer.range(0, STEP_PARAMS_BYTES).expect("range");
        self.ctx.step_begin(&self.stream, range).expect("begin");
        self.read_params(&buffer)
    }

    fn queue_advance(&self, buffer: &DeviceBuffer, m: u32) {
        let range = buffer.range(0, STEP_PARAMS_BYTES).expect("range");
        self.ctx
            .step_advance(&self.stream, range, m)
            .expect("advance");
    }

    fn advance(&self, p: StepParams, m: u32) -> StepParams {
        let buffer = self.upload_params(p);
        self.queue_advance(&buffer, m);
        self.read_params(&buffer)
    }

    /// Runs one argmax over `row` on a block `start`, with the scratch filled with `scratch`.
    fn argmax(&self, row: &[f32], start: StepParams, scratch: u8) -> (StepParams, ResultPair) {
        let logits = testkit::upload_f32(&self.ctx, row);
        let params = self.upload_params(start);
        let part = testkit::upload_filled(&self.ctx, scratch, ARGMAX_PART_BYTES);
        let result = testkit::upload_filled(&self.ctx, 0xEE, RESULT_PAIR_BYTES);
        let call = ArgmaxCall {
            n: row.len(),
            logits: logits.range(0, 4 * row.len()).expect("logits range"),
            part: part.range(0, ARGMAX_PART_BYTES).expect("part range"),
            params: params.range(0, STEP_PARAMS_BYTES).expect("params range"),
            result: Some(result.range(0, RESULT_PAIR_BYTES).expect("result range")),
        };
        self.ctx.argmax(&self.stream, call).expect("argmax");
        let pair = self.read_pair(&result, 0);
        (self.read_params(&params), pair)
    }

    /// Argmax over `row` with the model checks of T1 to T5: token, both directions of the
    /// `check_logits` predicate, status OR, the chained `token_id`, no other word written, and
    /// the pair equal to the block.
    fn check_row(&self, case: &str, row: &[f32], preset: i32) -> ResultPair {
        let start = block(SENTINEL, 11, preset);
        let (after, pair) = self.argmax(row, start, 0xFF);
        let flagged = cpu_rejects(row);
        assert_eq!(pair.token, ref_row(row), "{case}: token");
        assert_eq!(pair.status.non_finite_logits(), flagged, "{case}: flag");
        assert_eq!(
            pair.status.0,
            preset | if flagged { 2 } else { 0 },
            "{case}: status"
        );
        if !flagged {
            let sampler = i32::try_from(cpu_argmax(row)).expect("row index");
            assert_eq!(pair.token, sampler, "{case}: the sampler rule");
        }
        assert_eq!(after.token_id, pair.token, "{case}: token_id");
        assert_eq!(after.status, pair.status.0, "{case}: params status");
        let untouched = |p: &StepParams| (p.pos, p.kv_len, p.kv_slot, p.pad);
        assert_eq!(untouched(&after), untouched(&start), "{case}: other words");
        assert!(pair.matches_params(&after), "{case}: the pair is a copy");
        pair
    }
}

#[test]
fn argmax_matches_the_sampler_rule_on_finite_rows() {
    if !testkit::device_present("argmax_matches_the_sampler_rule_on_finite_rows") {
        return;
    }
    let fx = Fx::new();
    let mut rng = Rng::new(0x4d_0001);
    for n in [1, 2, 31, 32, 33, 256, 257, 65536, 65537, 248_320] {
        let row = random_row(&mut rng, n);
        fx.check_row(&format!("random n {n}"), &row, 0);
    }
    for n in [65537, 248_320] {
        for p in [
            0, 1, 31, 32, 255, 256, 257, 1023, 4096, 32768, 65535, 65536, 65537, 100_000,
        ]
        .into_iter()
        .chain([n / 2, n - 2, n - 1])
        .filter(|&p| p < n)
        {
            let mut row = base_row(n);
            row[p] = 100.0;
            let pair = fx.check_row(&format!("max at {p} of {n}"), &row, 0);
            assert_eq!(pair.token, i32::try_from(p).expect("index"));
        }
    }
    signed_zeros_and_extremes(&fx);
    one_long_row(&fx);
}

/// `-0.0` and `+0.0` are equal, so the lower index wins whichever sign it has; denormals are
/// not flushed; `FLT_MAX` is finite.
fn signed_zeros_and_extremes(fx: &Fx) {
    for (n, lo, hi) in [(1000, 3, 5), (6000, 100, 5000), (248_320, 100_000, 200_000)] {
        for (low_zero, high_zero) in [(-0.0_f32, 0.0_f32), (0.0, -0.0)] {
            let mut row = base_row(n);
            row[lo] = low_zero;
            row[hi] = high_zero;
            let pair = fx.check_row(&format!("zeros {lo} {hi} of {n}"), &row, 0);
            assert_eq!(pair.token, i32::try_from(lo).expect("index"));
        }
    }
    let mut row = base_row(100);
    row[2] = 0.0;
    row[7] = f32::from_bits(1);
    let pair = fx.check_row("denormal above +0.0", &row, 0);
    assert_eq!(pair.token, 7, "a flush-to-zero build would tie and give 2");
    let mut row = base_row(100);
    (row[9], row[60]) = (f32::MAX, f32::MAX);
    assert!(!fx.check_row("FLT_MAX", &row, 0).status.non_finite_logits());
}

/// One row of 2^24 + 1 values: 256 grid-stride rounds per thread.
fn one_long_row(fx: &Fx) {
    let n = (1 << 24) + 1;
    let mut row = base_row(n);
    for (case, spots) in [
        ("max at 0", vec![0]),
        ("max at n - 1", vec![n - 1]),
        ("max at 65536", vec![65536]),
        ("tie n/2 and n - 1", vec![n / 2, n - 1]),
    ] {
        for &p in &spots {
            row[p] = 50.0;
        }
        let pair = fx.check_row(&format!("long row, {case}"), &row, 0);
        assert_eq!(pair.token, i32::try_from(spots[0]).expect("index"));
        for &p in &spots {
            row[p] = base_row(1)[0];
        }
    }
}

#[test]
fn argmax_breaks_ties_to_the_lowest_index() {
    if !testkit::device_present("argmax_breaks_ties_to_the_lowest_index") {
        return;
    }
    let fx = Fx::new();
    let n = 66_000;
    let sets: [&[usize]; 7] = [
        &[31, 32],
        &[255, 256],
        &[256, 65791],
        &[65536, 65537],
        &[7, n - 1],
        &[900, 40, 65000],
        &[],
    ];
    for tie in sets {
        let mut row = vec![0.5_f32; n];
        for &p in tie {
            row[p] = 9.0;
        }
        let pair = fx.check_row(&format!("tie {tie:?}"), &row, 0);
        assert_eq!(pair.token, tie.iter().min().map_or(0, |&p| p as i32));
    }
}

/// The rows of T3: `(name, row)` with one defect at `p` in a row of `n` values.
fn defective_rows(n: usize, p: usize) -> Vec<(&'static str, Vec<f32>)> {
    let with = |value: f32| {
        let mut row = base_row(n);
        row[p] = value;
        row
    };
    let mut three = base_row(n);
    for at in [p, (p + 5) % n, (p + 9) % n] {
        three[at] = f32::INFINITY;
    }
    let mut nan_neg = vec![f32::NEG_INFINITY; n];
    nan_neg[p] = f32::NAN;
    let mut nan_one = vec![f32::NAN; n];
    nan_one[p] = 1.5;
    vec![
        ("one NaN", with(f32::NAN)),
        ("one +inf", with(f32::INFINITY)),
        ("one -inf", with(f32::NEG_INFINITY)),
        ("three +inf", three),
        ("NaN among -inf", nan_neg),
        ("one finite among NaN", nan_one),
    ]
}

#[test]
fn argmax_flags_every_non_finite_value() {
    if !testkit::device_present("argmax_flags_every_non_finite_value") {
        return;
    }
    let fx = Fx::new();
    let n = 65_537;
    let mut rows = vec![
        ("all -inf", vec![f32::NEG_INFINITY; n]),
        ("all NaN", vec![f32::NAN; n]),
        ("a single NaN", vec![f32::NAN]),
    ];
    for p in [0, 31, 256, 4096, 65_535, 65_536] {
        rows.extend(defective_rows(n, p));
    }
    for (case, row) in rows {
        let pair = fx.check_row(case, &row, 0);
        assert!(cpu_rejects(&row), "{case}: check_logits rejects the row");
        assert_backend(pair.into_token().err(), "sample", Some(NON_FINITE), case);
    }
    let all_neg = fx.check_row("all -inf", &vec![f32::NEG_INFINITY; 300], 0);
    assert_eq!(all_neg.token, 0);
    let all_nan = fx.check_row("all NaN", &vec![f32::NAN; 300], 0);
    assert_eq!(all_nan.token, -1);
}

#[test]
fn argmax_ors_into_status_and_publishes_the_result_pair() {
    if !testkit::device_present("argmax_ors_into_status_and_publishes_the_result_pair") {
        return;
    }
    let fx = Fx::new();
    let clean = base_row(1000);
    let mut nan_row = base_row(1000);
    nan_row[500] = f32::NAN;
    assert_eq!(fx.check_row("clean, router bit", &clean, 1).status.0, 1);
    assert_eq!(fx.check_row("NaN, router bit", &nan_row, 1).status.0, 3);
    for at in [8, 16] {
        result_pair_at(&fx, &clean, at);
    }
    late_writer_is_detected(&fx, &clean, &nan_row);
    null_result_writes_nothing(&fx, &clean);
    for (status, operation) in [
        (1, "route_experts"),
        (3, "route_experts"),
        (4, "step_params"),
        (5, "step_params"),
        (6, "step_params"),
        (7, "step_params"),
    ] {
        let mut bytes = [0_u8; 8];
        bytes[..4].copy_from_slice(&41_i32.to_le_bytes());
        bytes[4..].copy_from_slice(&i32::to_le_bytes(status));
        let pair = ResultPair::from_bytes(bytes);
        assert_eq!((pair.token, pair.status), (41, StepStatus(status)));
        assert_backend(pair.into_token().err(), operation, None, "into_token");
    }
    let ok = ResultPair::from_bytes([41, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(ok.into_token().expect("status 0 gives the token"), 41);
}

fn result_pair_at(fx: &Fx, row: &[f32], at: usize) {
    let logits = testkit::upload_f32(&fx.ctx, row);
    let params = fx.upload_params(block(SENTINEL, 11, 0));
    let part = testkit::upload_filled(&fx.ctx, 0, ARGMAX_PART_BYTES);
    let result = testkit::upload_filled(&fx.ctx, 0xEE, 64);
    let call = ArgmaxCall {
        n: row.len(),
        logits: logits.range(0, 4 * row.len()).expect("logits"),
        part: part.range(0, ARGMAX_PART_BYTES).expect("part"),
        params: params.range(0, STEP_PARAMS_BYTES).expect("params"),
        result: Some(result.range(at, RESULT_PAIR_BYTES).expect("result")),
    };
    fx.ctx.argmax(&fx.stream, call).expect("argmax");
    let pair = fx.read_pair(&result, at);
    assert_eq!(
        (pair.token, pair.status),
        (0, StepStatus(0)),
        "pair at {at}"
    );
    assert!(pair.matches_params(&fx.read_params(&params)));
}

/// An argmax call over `row` with the scratch and block in `scratch`.
fn pair_call<'b>(
    n: usize,
    row: &'b DeviceBuffer,
    scratch: (&'b DeviceBuffer, &'b DeviceBuffer),
    pair: Option<&'b DeviceBuffer>,
) -> ArgmaxCall<'b> {
    ArgmaxCall {
        n,
        logits: row.range(0, 4 * n).expect("logits"),
        part: scratch.0.range(0, ARGMAX_PART_BYTES).expect("part"),
        params: scratch.1.range(0, STEP_PARAMS_BYTES).expect("params"),
        result: pair.map(|b| b.range(0, RESULT_PAIR_BYTES).expect("result")),
    }
}

/// R-C2b: a second argmax that writes `status` after the pair was made leaves the pair stale.
fn late_writer_is_detected(fx: &Fx, clean: &[f32], nan_row: &[f32]) {
    let logits = testkit::upload_f32(&fx.ctx, clean);
    let late = testkit::upload_f32(&fx.ctx, nan_row);
    let params = fx.upload_params(block(SENTINEL, 11, 0));
    let part = testkit::upload_filled(&fx.ctx, 0, ARGMAX_PART_BYTES);
    let result = testkit::upload_filled(&fx.ctx, 0xEE, RESULT_PAIR_BYTES);
    let scratch = (&part, &params);
    let run = |row, pair| pair_call(clean.len(), row, scratch, pair);
    fx.ctx
        .argmax(&fx.stream, run(&logits, Some(&result)))
        .expect("first");
    let pair = fx.read_pair(&result, 0);
    assert!(pair.matches_params(&fx.read_params(&params)), "clean path");
    let same = fx.read_params(&params);
    let other_token = StepParams {
        token_id: same.token_id + 1,
        ..same
    };
    assert!(!pair.matches_params(&other_token), "the token is compared");
    fx.ctx
        .argmax(&fx.stream, run(&late, None))
        .expect("late writer");
    let after = fx.read_params(&params);
    assert_eq!(after.status, STATUS_NON_FINITE_LOGITS);
    assert!(
        !pair.matches_params(&after),
        "a late writer makes the pair stale"
    );
    let mut moved_token = after;
    moved_token.token_id += 1;
    assert!(!fx.read_pair(&result, 0).matches_params(&moved_token));
}

fn null_result_writes_nothing(fx: &Fx, row: &[f32]) {
    let logits = testkit::upload_f32(&fx.ctx, row);
    // The block is followed by sentinel bytes in the same buffer: with no pair, the bytes next to
    // the block must stay as they were.
    let mut image = bits(&block(SENTINEL, 11, 0)).to_vec();
    image.extend([0xEE; 32]);
    let params = testkit::upload_bytes(&fx.ctx, &image);
    let part = testkit::upload_filled(&fx.ctx, 0, ARGMAX_PART_BYTES);
    let call = ArgmaxCall {
        n: row.len(),
        logits: logits.range(0, 4 * row.len()).expect("logits"),
        part: part.range(0, ARGMAX_PART_BYTES).expect("part"),
        params: params.range(0, STEP_PARAMS_BYTES).expect("params"),
        result: None,
    };
    fx.ctx
        .argmax(&fx.stream, call)
        .expect("argmax without a pair");
    fx.stream.synchronize().expect("synchronize");
    let bytes = testkit::read_bytes(&fx.ctx, &params, image.len());
    assert_eq!(
        bytes[STEP_PARAMS_BYTES..],
        [0xEE; 32],
        "an absent pair writes nothing"
    );
}

#[test]
fn argmax_chains_the_token_and_writes_no_other_word() {
    if !testkit::device_present("argmax_chains_the_token_and_writes_no_other_word") {
        return;
    }
    let fx = Fx::new();
    let mut rng = Rng::new(0x4d_0005);
    for n in [1, 300, 70_000] {
        let row = random_row(&mut rng, n);
        let pair = fx.check_row(&format!("chain n {n}"), &row, 0);
        assert_eq!(pair.token, i32::try_from(cpu_argmax(&row)).expect("index"));
    }
}

#[test]
fn step_begin_clears_the_status_and_flags_a_bad_state() {
    if !testkit::device_present("step_begin_clears_the_status_and_flags_a_bad_state") {
        return;
    }
    let fx = Fx::new();
    let valid = block(77, 100, 3);
    assert_eq!(fx.begin(valid), StepParams { status: 0, ..valid });
    let top = POS_MAX + 1;
    let mut bad = vec![
        StepParams {
            status: 3,
            ..StepParams::poisoned()
        },
        StepParams {
            pos: top,
            kv_len: top + 1,
            kv_slot: top,
            ..valid
        },
        StepParams {
            pos: -1,
            kv_len: 0,
            kv_slot: -1,
            ..valid
        },
        StepParams {
            kv_len: 100,
            ..valid
        },
        StepParams {
            kv_slot: 99,
            ..valid
        },
        StepParams {
            pos: i32::MAX,
            kv_len: i32::MIN,
            kv_slot: i32::MAX,
            ..valid
        },
    ];
    bad.push(StepParams {
        kv_len: WORD_MAX,
        ..valid
    });
    for p in bad {
        let after = fx.begin(p);
        assert_eq!(after, sim_begin(p), "begin on {p:?}");
        assert_eq!(after.status, STATUS_INVALID_STEP_STATE, "mask 4 for {p:?}");
    }
}

#[test]
fn step_advance_adds_m_and_keeps_kv_len_one_ahead() {
    if !testkit::device_present("step_advance_adds_m_and_keeps_kv_len_one_ahead") {
        return;
    }
    let fx = Fx::new();
    let start = block(77, 100, 3);
    for m in [1, 2, 3, 16, MAX_ADVANCE as u32] {
        let after = fx.advance(start, m);
        let moved = 100 + m as i32;
        assert_eq!(
            (after.pos, after.kv_len, after.kv_slot),
            (moved, moved + 1, moved)
        );
        assert_eq!(
            (after.token_id, after.status, after.pad),
            (77, 3, PAD_SENTINEL)
        );
    }
    let mut rng = Rng::new(0x4d_0007);
    for (from, span) in [(0, 5), (POS_MAX - 3000, 7)] {
        let buffer = fx.upload_params(block(5, from, 0));
        let mut model = block(5, from, 0);
        for chunk in 0..10 {
            for _ in 0..100 {
                let m = 1 + rng.below(span) as u32;
                fx.queue_advance(&buffer, m);
                model = sim_advance(model, m as i32);
            }
            assert_eq!(fx.read_params(&buffer), model, "chunk {chunk} from {from}");
        }
    }
    advance_at_the_top(&fx);
}

/// The top: `POS_MAX` is reachable, one more position and every corrupt word give -1, never 0.
fn advance_at_the_top(fx: &Fx) {
    let near = block(9, POS_MAX - 5, 0);
    let exact = fx.advance(near, 5);
    assert_eq!(
        (exact.pos, exact.kv_len, exact.kv_slot),
        (POS_MAX, WORD_MAX, POS_MAX)
    );
    let over = fx.advance(near, 6);
    assert_eq!((over.pos, over.kv_len, over.kv_slot), (-1, -1, -1));
    assert_eq!(
        fx.begin(over).status,
        STATUS_INVALID_STEP_STATE,
        "the poison is reported"
    );
    let top = POS_MAX + 1;
    let corrupt = [
        block(9, POS_MAX, 0),
        StepParams::poisoned(),
        StepParams {
            pos: top,
            kv_len: top + 1,
            kv_slot: top,
            ..near
        },
        StepParams {
            kv_len: near.pos,
            ..near
        },
        StepParams {
            kv_slot: near.pos + 1,
            ..near
        },
        StepParams {
            pos: i32::MAX,
            kv_len: i32::MIN,
            kv_slot: i32::MAX,
            ..near
        },
    ];
    for p in corrupt {
        let after = fx.advance(p, 1);
        assert_eq!(
            (after.pos, after.kv_len, after.kv_slot),
            (-1, -1, -1),
            "{p:?}"
        );
        assert_eq!(after, sim_advance(p, 1));
    }
}

#[test]
fn step_kernels_are_deterministic() {
    if !testkit::device_present("step_kernels_are_deterministic") {
        return;
    }
    let fx = Fx::new();
    let mut rng = Rng::new(0x4d_0008);
    let row: Vec<f32> = (0..65_537).map(|_| rng.below(3) as f32).collect();
    let start = block(SENTINEL, 11, 0);
    let first = fx.argmax(&row, start, 0xFF);
    let second = fx.argmax(&row, start, 0x00);
    assert_eq!(first, second, "the scratch contents do not matter");
    assert_eq!(
        first.1.token,
        i32::try_from(cpu_argmax(&row)).expect("index")
    );
}

/// Offsets and sizes of the valid argmax call inside the one 32 KiB buffer of T9.
const ARENA: usize = 32768;
const N: usize = 64;
type Span = (usize, usize);
/// `[logits, part, params, result]`.
const VALID: [Span; 4] = [(4096, 4 * N), (8192, 3072), (0, 32), (16384, 8)];

fn with(index: usize, span: Span) -> [Span; 4] {
    let mut spans = VALID;
    spans[index] = span;
    spans
}

struct Calls<'a> {
    fx: &'a Fx,
    arena: &'a DeviceBuffer,
}

impl Calls<'_> {
    fn begin(&self, params: Span) -> Result<(), EngError> {
        let range = self.arena.range(params.0, params.1)?;
        self.fx.ctx.step_begin(&self.fx.stream, range)
    }

    fn advance(&self, params: Span, m: u32) -> Result<(), EngError> {
        let range = self.arena.range(params.0, params.1)?;
        self.fx.ctx.step_advance(&self.fx.stream, range, m)
    }

    fn argmax(&self, n: usize, spans: [Span; 4]) -> Result<(), EngError> {
        self.argmax_on(&self.fx.stream, [self.arena; 4], n, spans)
    }

    fn argmax_on(
        &self,
        stream: &DeviceStream,
        buffers: [&DeviceBuffer; 4],
        n: usize,
        spans: [Span; 4],
    ) -> Result<(), EngError> {
        let call = ArgmaxCall {
            n,
            logits: buffers[0].range(spans[0].0, spans[0].1)?,
            part: buffers[1].range(spans[1].0, spans[1].1)?,
            params: buffers[2].range(spans[2].0, spans[2].1)?,
            result: Some(buffers[3].range(spans[3].0, spans[3].1)?),
        };
        self.fx.ctx.argmax(stream, call)
    }
}

type Case = (String, &'static str, Result<(), EngError>);

fn params_rows(c: &Calls<'_>) -> Vec<Case> {
    type Run<'r> = Box<dyn Fn(Span) -> Result<(), EngError> + 'r>;
    let runs: [(&str, Run<'_>); 3] = [
        ("begin", Box::new(|p| c.begin(p))),
        ("advance", Box::new(|p| c.advance(p, 1))),
        ("argmax", Box::new(|p| c.argmax(N, with(2, p)))),
    ];
    let mut rows: Vec<Case> = Vec::new();
    for (call, run) in &runs {
        let named = |what: &str| format!("{call}: {what}");
        rows.push((named("params 31 bytes"), "params range holds", run((0, 31))));
        rows.push((
            named("params at offset 2"),
            "params range starts at offset",
            run((2, 32)),
        ));
        rows.push((named("params of 0 bytes"), "does not fit", run((0, 0))));
        rows.push((
            named("params past the end"),
            "does not fit",
            run((ARENA - 16, 32)),
        ));
    }
    let too_many = u32::try_from(MAX_ADVANCE).expect("m") + 1;
    rows.push(("m = 0".into(), "step advance takes", c.advance(VALID[2], 0)));
    rows.push((
        "m = 4097".into(),
        "step advance takes",
        c.advance(VALID[2], too_many),
    ));
    rows
}

fn argmax_rows(c: &Calls<'_>) -> Vec<Case> {
    let cases: Vec<(&str, &'static str, usize, [Span; 4])> = vec![
        ("n = 0", "argmax takes", 0, VALID),
        (
            "logits one float short",
            "logits range holds",
            N,
            with(0, (4096, 4 * N - 4)),
        ),
        (
            "logits at offset 2",
            "logits range starts at offset",
            N,
            with(0, (4098, 4 * N)),
        ),
        ("logits of 0 bytes", "does not fit", N, with(0, (4096, 0))),
        (
            "part 3068 bytes",
            "part range holds",
            N,
            with(1, (8192, 3068)),
        ),
        (
            "part at offset 2",
            "part range starts at offset",
            N,
            with(1, (8194, 3072)),
        ),
        (
            "part past the end",
            "does not fit",
            N,
            with(1, (ARENA - 16, 3072)),
        ),
        (
            "result 4 bytes",
            "result range holds",
            N,
            with(3, (16384, 4)),
        ),
        (
            "result at offset 2",
            "result range starts at offset",
            N,
            with(3, (16386, 8)),
        ),
        (
            "result at offset 4",
            "result range starts at offset",
            N,
            with(3, (16388, 8)),
        ),
        (
            "part inside logits",
            "logits and part ranges overlap",
            N,
            with(1, (4196, 3072)),
        ),
        (
            "result on params",
            "params and result ranges overlap",
            N,
            with(3, (16, 8)),
        ),
        (
            "logits on params",
            "logits and params ranges overlap",
            N,
            with(0, (0, 4 * N)),
        ),
    ];
    cases
        .into_iter()
        .map(|(name, needle, n, spans)| (name.to_owned(), needle, c.argmax(n, spans)))
        .collect()
}

/// A buffer or a stream of another context is refused, for each of the four ranges.
fn foreign_rows(c: &Calls<'_>, other: &Fx) -> Vec<Case> {
    let foreign = testkit::upload_filled(&other.ctx, 0, ARENA);
    let mut rows: Vec<Case> = Vec::new();
    for (index, name) in ["logits", "part", "params", "result"]
        .into_iter()
        .enumerate()
    {
        let mut buffers = [c.arena; 4];
        buffers[index] = &foreign;
        let result = c.argmax_on(&c.fx.stream, buffers, N, VALID);
        rows.push((
            format!("{name} in another context"),
            "buffer belongs",
            result,
        ));
    }
    let result = c.argmax_on(&other.stream, [c.arena; 4], N, VALID);
    rows.push(("stream of another context".into(), "stream belongs", result));
    rows
}

#[test]
fn step_calls_reject_bad_arguments_before_any_launch() {
    if !testkit::device_present("step_calls_reject_bad_arguments_before_any_launch") {
        return;
    }
    let fx = Fx::new();
    let other = Fx::new();
    let arena = testkit::upload_filled(&fx.ctx, 0, ARENA);
    let c = Calls {
        fx: &fx,
        arena: &arena,
    };
    let before = testkit::read_bytes(&fx.ctx, &arena, ARENA);
    let mut rows = params_rows(&c);
    rows.extend(argmax_rows(&c));
    rows.extend(foreign_rows(&c, &other));
    // Every row is checked before the test fails, so one latched backend does not hide which
    // rows were refused for the right reason.
    let problems: Vec<String> = rows
        .into_iter()
        .filter_map(|(case, needle, result)| bad_request_problem(result, &case, needle))
        .collect();
    assert!(
        problems.is_empty(),
        "rows that failed:\n{}",
        problems.join("\n")
    );
    fx.stream.synchronize().expect("synchronize");
    let after = testkit::read_bytes(&fx.ctx, &arena, ARENA);
    assert!(before == after, "a refused call wrote to the device");
    // Accepted controls: the extents the call touches are disjoint, the whole ranges are not,
    // and a range longer than the call needs is fine (`require_fit` means at least).
    let tail = 4096 + 4 * N;
    let big: [Span; 4] = [
        (4096, 4 * N + 4096),
        (8192, 3072 + 64),
        (tail, 32),
        (8192 + 3072, 8),
    ];
    c.argmax(N, big)
        .expect("logits and part ranges that hold params and result in their tails");
    c.argmax(N, with(3, (16384, 64)))
        .expect("a result range longer than 8 bytes");
    c.argmax(N, with(2, (0, 64)))
        .expect("a params range longer than 32 bytes");
    c.begin((0, 64)).expect("begin with a long params range");
    c.advance((0, 64), 1)
        .expect("advance with a long params range");
    fx.stream
        .synchronize()
        .expect("the backend is usable afterwards");
}

#[test]
fn step_params_layout_is_pinned() {
    assert_eq!(size_of::<StepParams>(), 32);
    let offsets = [
        offset_of!(StepParams, token_id),
        offset_of!(StepParams, pos),
        offset_of!(StepParams, kv_len),
        offset_of!(StepParams, kv_slot),
        offset_of!(StepParams, status),
    ];
    assert_eq!(offsets, [0, 4, 8, 12, 16]);
    assert_eq!(
        offsets,
        [
            OFFSET_TOKEN_ID,
            OFFSET_POS,
            OFFSET_KV_LEN,
            OFFSET_KV_SLOT,
            OFFSET_STATUS
        ]
    );
    let distinct = StepParams {
        token_id: 0x0403_0201,
        pos: 0x0807_0605,
        kv_len: 0x0c0b_0a09,
        kv_slot: 0x100f_0e0d,
        status: 0x1413_1211,
        pad: [0x1817_1615, 0x1c1b_1a19, 0x201f_1e1d],
    };
    let want: Vec<u8> = (1..=32).collect();
    assert_eq!(distinct.to_bytes().to_vec(), want, "little endian");
    assert_eq!(StepParams::from_bytes(distinct.to_bytes()), distinct);
    let poisoned = StepParams::poisoned();
    assert_eq!(
        (
            poisoned.pos,
            poisoned.kv_len,
            poisoned.kv_slot,
            poisoned.status
        ),
        (-1, -1, -1, 0)
    );
    assert_eq!(
        (ARGMAX_PART_BYTES, RESULT_PAIR_BYTES, WORD_MAX),
        (3072, 8, (1 << 30) - 1)
    );
    new_and_bounds();
}

fn new_and_bounds() {
    let start = StepParams::new(7, 100).expect("a valid start");
    assert_eq!((start.kv_len, start.kv_slot, start.status), (101, 100, 0));
    StepParams::new(7, 0).expect("pos 0");
    StepParams::new(7, POS_MAX).expect("pos POS_MAX");
    for (token, pos) in [(7, -1), (7, WORD_MAX), (7, 1 << 30), (-1, 0)] {
        let refused = StepParams::new(token, pos).map(|_| ());
        assert_bad_request(refused, &format!("new({token}, {pos})"), "step");
    }
    let top = usize::try_from(POS_MAX).expect("positive");
    let rows = [
        (0, 1, 1, true),
        (0, 2, 1, false),
        (10, 5, 15, true),
        (10, 6, 15, false),
        (5, 0, 100, false),
        (top - 5, 5, usize::MAX, true),
        (top - 5, 6, usize::MAX, false),
        (top + 1, 1, usize::MAX, false),
        (usize::MAX, 1, usize::MAX, false),
        (5, usize::MAX, usize::MAX, false),
        (50, 3 * 16, 100, true),
        (50, 3 * 17, 100, false),
    ];
    for (pos, advance, rows, ok) in rows {
        let result = ensure_rows(pos, advance, rows);
        assert_eq!(result.is_ok(), ok, "ensure_rows({pos}, {advance}, {rows})");
    }
    for n in [1, MAX_LOGITS] {
        assert_eq!(
            check_n(n).expect("in range"),
            i32::try_from(n).expect("fits")
        );
    }
    for n in [0, MAX_LOGITS + 1, 1 << 32, usize::MAX] {
        assert_bad_request(
            check_n(n).map(|_| ()),
            &format!("check_n({n})"),
            "argmax takes",
        );
    }
}

#[test]
fn step_status_maps_as_the_replies_say() {
    assert!(StepStatus(0).error().is_none());
    let table = [
        (1, "route_experts"),
        (2, "sample"),
        (3, "route_experts"),
        (4, "step_params"),
        (5, "step_params"),
        (6, "step_params"),
        (7, "step_params"),
    ];
    for (status, operation) in table {
        assert_backend(
            StepStatus(status).error(),
            operation,
            None,
            &format!("status {status}"),
        );
    }
    assert_backend(
        StepStatus(2).error(),
        "sample",
        Some(NON_FINITE),
        "mask 2 text",
    );
    assert_eq!((STATUS_ROUTER_FAULT, STATUS_NON_FINITE_LOGITS), (1, 2));
    let all = StepStatus(7);
    assert!(all.router_fault() && all.non_finite_logits() && all.invalid_step_state());
    assert!(!StepStatus(3).invalid_step_state());
    assert!(
        StepStatus(8).error().is_none(),
        "an unknown bit alone is no error"
    );
    assert_eq!(
        (StepStatus(8).unknown_bits(), StepStatus(15).unknown_bits()),
        (8, 8)
    );
    assert_eq!(StepStatus(7).unknown_bits(), 0);
    for status in 0..16 {
        let mut bytes = [0_u8; 8];
        bytes[0] = 5;
        bytes[4] = status;
        let pair = ResultPair::from_bytes(bytes);
        let error = StepStatus(i32::from(status)).error();
        assert_eq!(
            pair.into_token().is_err(),
            error.is_some(),
            "status {status}"
        );
    }
}
