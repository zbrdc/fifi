//! Tests of the tiered wait-add wrapper (TCAP C3). The host-only tests pin the shape limits and the
//! source of the kernel; the device tests run the real kernel with a `MailboxPort` on a server
//! thread that plays the CPU server (C5 will replace it): the control block is set by upload, not by
//! classify, and every round gets the sequence it expects. Inputs are deterministic patterns derived
//! from the sequence; no oracle data, no environment variable, no `#[ignore]`. Run with
//! `--test-threads=1`.
//!
//! Every device phase is a function that creates its own context, stream, buffers and mailbox, has
//! only `?` inside, drops every owner when it returns and hands back plain host records; the tests
//! assert on those records afterwards (an assertion that panics while a context is alive would
//! latch the backend for the rest of a one-process run).

use super::{CTRL_BYTES, MAX_TIMEOUT_NS, WaitAddCall, WaitAddShape, check_shape};
use crate::EngError;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::context::{CudaContext, DeviceStream, Mailbox};
use crate::kernels::cuda::resident::moe_combine::{CombineCall, CombineOut, CombineShape};
use crate::kernels::cuda::testkit::{self, assert_in_order, squeezed};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// The sentinel byte of a buffer the kernel must not touch.
const UNTOUCHED: u8 = 0x77;
/// The quiet NaN the kernel poisons a failed row with.
const QUIET_NAN: u32 = 0x7FC0_0000;

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bits_of(bytes: &[u8]) -> Vec<u32> {
    testkit::floats(bytes).iter().map(|v| v.to_bits()).collect()
}

fn word_of(bytes: &[u8]) -> i32 {
    bytes
        .first_chunk::<4>()
        .map_or(0, |word| i32::from_le_bytes(*word))
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; len];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes)
}

/// Element `j` of a row (`salt` 0) or a partial (`salt` 1) of sequence `k`: a finite float in
/// [1, 2) derived from a hash of `(k, j)`, so a wrong or torn partial equals the expected one only
/// by a 2^-23 collision per element.
fn pattern(salt: u64, k: u64, j: usize) -> f32 {
    let mut x = (k << 24) ^ ((j as u64) << 1) ^ salt;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    f32::from_bits(0x3F80_0000 | (x as u32 & 0x007F_FFFF))
}

fn pattern_vec(salt: u64, k: u64, hidden: usize) -> Vec<f32> {
    (0..hidden).map(|j| pattern(salt, k, j)).collect()
}

/// The v1 second-combine arithmetic the kernel must equal bit for bit: `r + (1.0 * p)`, one
/// operation per statement so Rust never contracts them.
fn summed(row: &[f32], partial: &[f32]) -> Vec<u32> {
    row.iter()
        .zip(partial)
        .map(|(&r, &p)| {
            let term = 1.0f32 * p;
            (r + term).to_bits()
        })
        .collect()
}

/// Equal bit for bit, except that any NaN equals any NaN: the payload of a NaN a sum produces is the hardware's
/// choice (the device's add gives the canonical 0x7FFFFFFF where an x86 host gives 0xFFC00000 for an invalid operation and propagates the payload of a NaN operand), and only its being a
/// NaN is the contract.
fn same_modulo_nan(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&x, &y)| x == y || (f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan()))
}

/// The control block, the word and the residual row of one wait-add rig; the mailbox is not here
/// because a phase owns it next to the rig and the server thread holds its port.
struct Rig {
    ctrl: DeviceBuffer,
    word: DeviceBuffer,
    resid: DeviceBuffer,
}

impl Rig {
    fn new(ctx: &CudaContext, hidden: usize) -> Result<Self, EngError> {
        Ok(Self {
            ctrl: ctx.upload(&[0; CTRL_BYTES])?,
            word: ctx.upload(&[0; 4])?,
            resid: ctx.upload(&vec![0; hidden * 4])?,
        })
    }

    fn write(
        &self,
        stream: &DeviceStream,
        buffer: &DeviceBuffer,
        bytes: &[u8],
    ) -> Result<(), EngError> {
        buffer.write_at(stream, 0, bytes)
    }

    /// `ctrl[0] = miss`, `ctrl[1] = expect`: what classify would have written on the stream.
    fn set_ctrl(&self, stream: &DeviceStream, miss: u64, expect: u64) -> Result<(), EngError> {
        let mut bytes = [0u8; CTRL_BYTES];
        bytes[..8].copy_from_slice(&miss.to_le_bytes());
        bytes[8..16].copy_from_slice(&expect.to_le_bytes());
        self.write(stream, &self.ctrl, &bytes)
    }

    fn set_word(&self, stream: &DeviceStream, value: i32) -> Result<(), EngError> {
        self.write(stream, &self.word, &value.to_le_bytes())
    }

    fn set_row(&self, stream: &DeviceStream, row: &[f32]) -> Result<(), EngError> {
        self.write(stream, &self.resid, &bytes_f32(row))
    }

    fn word(&self, ctx: &CudaContext) -> Result<i32, EngError> {
        Ok(word_of(&read(ctx, &self.word, 4)?))
    }

    fn row_bits(&self, ctx: &CudaContext, hidden: usize) -> Result<Vec<u32>, EngError> {
        Ok(bits_of(&read(ctx, &self.resid, hidden * 4)?))
    }

    fn row_bytes(&self, ctx: &CudaContext, hidden: usize) -> Result<Vec<u8>, EngError> {
        read(ctx, &self.resid, hidden * 4)
    }
}

/// Queues one wait-add of `hidden` elements with a 2-second budget (the late-answer test passes
/// its own).
fn launch(
    ctx: &CudaContext,
    stream: &DeviceStream,
    mailbox: &Mailbox,
    rig: &Rig,
    hidden: usize,
    timeout_ns: u64,
) -> Result<(), EngError> {
    let call = WaitAddCall {
        hidden,
        ctrl: rig.ctrl.range(0, CTRL_BYTES)?,
        word: rig.word.range(0, 4)?,
        resid: rig.resid.range(0, hidden * 4)?,
        mailbox,
        timeout_ns,
    };
    ctx.tiered_wait_add(stream, call)
}

/// The interesting pairs (residual element, partial element): the signs of zero, subnormals, large
/// magnitudes, both infinities (`Inf + -Inf` included: its NaN payload is the hardware's, so the host sum is compared modulo
/// the payload) and the quiet NaN; with the weight 1.0 the kernel's sum is exactly `r + p`.
fn special_pairs() -> Vec<(f32, f32)> {
    let sub = f32::from_bits(1);
    vec![
        (0.0, 0.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (1.5, -0.0),
        (-0.0, sub),
        (sub, sub),
        (sub, -sub),
        (-sub, sub),
        (1.0, sub),
        (sub, f32::MAX),
        (f32::MAX, f32::MAX),
        (-f32::MAX, -f32::MAX),
        (f32::MAX, -f32::MAX),
        (f32::MIN_POSITIVE, f32::MAX),
        (f32::INFINITY, 1.0),
        (-f32::INFINITY, 1.0),
        (1.0, f32::INFINITY),
        (1.0, -f32::INFINITY),
        (f32::INFINITY, f32::INFINITY),
        (f32::INFINITY, -f32::INFINITY),
        (-f32::INFINITY, f32::INFINITY),
        (-f32::INFINITY, -f32::INFINITY),
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::NAN, f32::NAN),
        (f32::INFINITY, f32::NAN),
        (f32::NAN, -f32::INFINITY),
        (-1.0, 1.0),
        (1e30, -1e30),
    ]
}

/// One served round's record: the kernel's row, the device combine's row over the same inputs,
/// the host sum, the word and whether the server answered.
struct Served {
    got: Vec<u32>,
    combine: Vec<u32>,
    want: Vec<u32>,
    word: i32,
    answered: bool,
}

/// One served round at `hidden` with the special pairs: sequence 1 is answered with the special
/// partial.
fn special_rounds(hidden: usize) -> Result<Served, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let pairs = special_pairs();
    let row: Vec<f32> = (0..hidden).map(|j| pairs[j % pairs.len()].0).collect();
    let partial: Vec<f32> = (0..hidden).map(|j| pairs[j % pairs.len()].1).collect();
    let want = summed(&row, &partial);
    let rig = Rig::new(&ctx, hidden)?;
    rig.set_ctrl(&stream, 1, 1)?;
    rig.set_word(&stream, 0)?;
    rig.set_row(&stream, &row)?;
    let go = AtomicBool::new(false);
    let (queued, served) = thread::scope(|scope| {
        let server = scope.spawn(|| {
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            port.answer(1, Ok(&partial))
        });
        let queued = launch(&ctx, &stream, &mailbox, &rig, hidden, 2_000_000_000);
        go.store(true, Ordering::Release);
        (queued, server.join().unwrap_or(false))
    });
    queued?;
    stream.synchronize()?;
    let got = rig.row_bits(&ctx, hidden)?;
    // The device oracle: the second moe_combine of the v1 block, m 1, top_k 1, weight 1.0.
    let res = ctx.upload(&bytes_f32(&row))?;
    let down = ctx.upload(&bytes_f32(&partial))?;
    let weights = ctx.upload(&bytes_f32(&[1.0]))?;
    let out = ctx.upload(&vec![0u8; hidden * 4])?;
    let combine = CombineCall {
        shape: CombineShape {
            m: 1,
            n: hidden,
            top_k: 1,
            ld_down: hidden,
        },
        res: res.range(0, hidden * 4)?,
        down: down.range(0, hidden * 4)?,
        weights: weights.range(0, 4)?,
        shared: None,
        out: CombineOut::To(out.range(0, hidden * 4)?),
    };
    ctx.moe_combine(&stream, combine)?;
    stream.synchronize()?;
    let combine_row = bits_of(&read(&ctx, &out, hidden * 4)?);
    let word = rig.word(&ctx)?;
    Ok(Served {
        got,
        combine: combine_row,
        want,
        word,
        answered: served,
    })
}

/// The kernel's sum is bitwise the host IEEE sum `r + (1.0 * p)` and the device combine's, over
/// zero signs, subnormals, large magnitudes, infinities and NaN, at the Qwen3.6 width and a small
/// one; a served round leaves the word alone.
#[test]
fn tiered_wait_add_equals_moe_combine_top_k_one_bitwise() {
    if !testkit::device_present("tiered_wait_add_equals_moe_combine_top_k_one_bitwise") {
        return;
    }
    for hidden in [2048usize, 256] {
        let Served {
            got,
            combine,
            want,
            word,
            answered,
        } = special_rounds(hidden).expect("the device phase");
        assert!(answered, "hidden {hidden}: the server answered");
        assert!(
            want.contains(&QUIET_NAN) && want.contains(&f32::INFINITY.to_bits()),
            "hidden {hidden}: the pairs cover NaN and +Inf"
        );
        assert!(
            same_modulo_nan(&got, &want),
            "hidden {hidden}: the kernel equals the host sum (NaN payloads aside)"
        );
        assert_eq!(
            got, combine,
            "hidden {hidden}: the kernel equals the device combine"
        );
        assert_eq!(word, 0, "hidden {hidden}: a served round leaves the word");
    }
}

/// The phase of the no-miss test: a sentinel row and word, miss count 0, a 1 ms budget and no
/// server at all.
fn untouched_phase(hidden: usize) -> Result<(Vec<u8>, Vec<u8>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let rig = Rig::new(&ctx, hidden)?;
    rig.set_ctrl(&stream, 0, 1)?;
    rig.set_word(&stream, 0xAAAA_AAAAu32 as i32)?;
    rig.write(&stream, &rig.resid, &vec![UNTOUCHED; hidden * 4])?;
    launch(&ctx, &stream, &mailbox, &rig, hidden, 1_000_000)?;
    stream.synchronize()?;
    Ok((rig.row_bytes(&ctx, hidden)?, read(&ctx, &rig.word, 4)?))
}

/// With no miss the kernel writes nothing at all: the sentinel row and word stay byte for byte.
#[test]
fn tiered_wait_add_without_misses_writes_nothing() {
    if !testkit::device_present("tiered_wait_add_without_misses_writes_nothing") {
        return;
    }
    let (row, word) = untouched_phase(256).expect("the device phase");
    assert!(row.iter().all(|&b| b == UNTOUCHED), "the row is untouched");
    assert_eq!(
        word_of(&word),
        0xAAAA_AAAAu32 as i32,
        "the word is untouched"
    );
}

/// One unanswered wait at a 20 ms budget, starting from `start_word`.
fn timeout_phase(hidden: usize, start_word: i32) -> Result<(i32, Vec<u32>, Duration), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let rig = Rig::new(&ctx, hidden)?;
    rig.set_ctrl(&stream, 1, 1)?;
    rig.set_word(&stream, start_word)?;
    rig.set_row(&stream, &vec![0.5; hidden])?;
    let port = mailbox.port()?;
    let finished = AtomicBool::new(false);
    let started = Instant::now();
    // The watchdog: if the wait has not ended after two seconds (a kernel without its own timeout), it answers with
    // an error so the kernel returns and the test fails with a message instead of hanging.
    let (queued, synced) = thread::scope(|scope| {
        scope.spawn(|| {
            while started.elapsed() < Duration::from_secs(2) {
                if finished.load(Ordering::Acquire) {
                    return;
                }
                thread::sleep(Duration::from_millis(5));
            }
            let _ = port.answer(1, Err(99));
        });
        let queued = launch(&ctx, &stream, &mailbox, &rig, hidden, 20_000_000);
        let synced = queued.as_ref().map_or(Ok(()), |()| stream.synchronize());
        finished.store(true, Ordering::Release);
        (queued, synced)
    });
    queued?;
    synced?;
    let elapsed = started.elapsed();
    Ok((rig.word(&ctx)?, rig.row_bits(&ctx, hidden)?, elapsed))
}

/// A wait nobody answers times out after its budget: bit 2 lands in the word (OR, not store: a
/// word that already holds 4 becomes 6), every element of the row is the quiet NaN, and the call
/// returns after at least the budget and well before two seconds.
#[test]
fn tiered_wait_add_times_out_with_its_bit_and_a_nan_row() {
    if !testkit::device_present("tiered_wait_add_times_out_with_its_bit_and_a_nan_row") {
        return;
    }
    let (word, row, elapsed) = timeout_phase(256, 0).expect("the device phase");
    assert_eq!(word, 2, "the timeout bit, starting from 0");
    assert!(
        row.iter().all(|&b| b == QUIET_NAN),
        "the row is the quiet NaN"
    );
    assert!(elapsed >= Duration::from_millis(20), "waited {elapsed:?}");
    assert!(elapsed < Duration::from_secs(2), "waited {elapsed:?}");
    let (word, row, _) = timeout_phase(256, 4).expect("the second device phase");
    assert_eq!(word, 6, "the timeout bit ORs into a word that holds 4");
    assert!(
        row.iter().all(|&b| b == QUIET_NAN),
        "the second row is the quiet NaN"
    );
}

/// One round whose server answers `result` for sequence 1 after `delay`.
fn answered_round(
    hidden: usize,
    result: Result<Vec<f32>, u32>,
    delay: Duration,
    timeout_ns: u64,
) -> Result<(i32, Vec<u32>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let row = pattern_vec(0, 1, hidden);
    let rig = Rig::new(&ctx, hidden)?;
    rig.set_ctrl(&stream, 1, 1)?;
    // The word starts at 4 (another kernel's bit), so an OR is told from a store.
    rig.set_word(&stream, 4)?;
    rig.set_row(&stream, &row)?;
    let go = AtomicBool::new(false);
    let (queued, served) = thread::scope(|scope| {
        let server = scope.spawn(|| {
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            thread::sleep(delay);
            match &result {
                Ok(partial) => port.answer(1, Ok(partial)),
                Err(code) => port.answer(1, Err(*code)),
            }
        });
        let queued = launch(&ctx, &stream, &mailbox, &rig, hidden, timeout_ns);
        go.store(true, Ordering::Release);
        (queued, server.join().unwrap_or(false))
    });
    queued?;
    if !served {
        return Err(EngError::BadRequest("the server never answered".to_owned()));
    }
    stream.synchronize()?;
    Ok((rig.word(&ctx)?, rig.row_bits(&ctx, hidden)?))
}

/// A server error (code 7) ORs bit 1 into the word (4 becomes 5) and nothing else, and poisons the row.
#[test]
fn tiered_wait_add_reports_a_server_error() {
    if !testkit::device_present("tiered_wait_add_reports_a_server_error") {
        return;
    }
    let (word, row) =
        answered_round(256, Err(7), Duration::ZERO, 2_000_000_000).expect("the device phase");
    assert_eq!(word, 5, "the CPU_FAILED bit ORed into 4");
    assert!(
        row.iter().all(|&b| b == QUIET_NAN),
        "the row is the quiet NaN"
    );
}

/// An answer that arrives 30 ms into a 2-second wait is accepted: the row is the sum and the word
/// is untouched (a hoisted weak load of `done` never sees the answer and times out instead).
#[test]
fn tiered_wait_add_accepts_an_answer_that_arrives_late() {
    if !testkit::device_present("tiered_wait_add_accepts_an_answer_that_arrives_late") {
        return;
    }
    let partial = pattern_vec(1, 1, 256);
    let row = pattern_vec(0, 1, 256);
    let want = summed(&row, &partial);
    let (word, got) = answered_round(256, Ok(partial), Duration::from_millis(30), 2_000_000_000)
        .expect("the device phase");
    assert_eq!(got, want, "the late answer is summed");
    assert_eq!(word, 4, "the word is untouched");
}

/// The phase of the stale-answer test: expect 5; the server first answers sequence 4 with a
/// poisoned partial, then, 20 ms later, sequence 5 with the real one.
fn stale_phase(hidden: usize) -> Result<(i32, Vec<u32>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let row = pattern_vec(0, 5, hidden);
    let poison = pattern_vec(3, 5, hidden);
    let real = pattern_vec(1, 5, hidden);
    let rig = Rig::new(&ctx, hidden)?;
    rig.set_ctrl(&stream, 1, 5)?;
    rig.set_word(&stream, 0)?;
    rig.set_row(&stream, &row)?;
    let go = AtomicBool::new(false);
    let (queued, served) = thread::scope(|scope| {
        let server = scope.spawn(|| {
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let stale = port.answer(4, Ok(&poison));
            thread::sleep(Duration::from_millis(20));
            stale && port.answer(5, Ok(&real))
        });
        let queued = launch(&ctx, &stream, &mailbox, &rig, hidden, 2_000_000_000);
        go.store(true, Ordering::Release);
        (queued, server.join().unwrap_or(false))
    });
    queued?;
    if !served {
        return Err(EngError::BadRequest("the server never answered".to_owned()));
    }
    stream.synchronize()?;
    Ok((rig.word(&ctx)?, rig.row_bits(&ctx, hidden)?))
}

/// An answer for an older sequence is stale and never accepted: with expect 5 the kernel ignores a
/// published sequence 4 with a poisoned partial and sums the real answer.
#[test]
fn tiered_wait_add_ignores_a_stale_answer() {
    if !testkit::device_present("tiered_wait_add_ignores_a_stale_answer") {
        return;
    }
    let (word, got) = stale_phase(256).expect("the device phase");
    let row = pattern_vec(0, 5, 256);
    let real = pattern_vec(1, 5, 256);
    assert_eq!(got, summed(&row, &real), "the real partial is summed");
    assert_ne!(
        got,
        summed(&row, &pattern_vec(3, 5, 256)),
        "the poisoned partial is not"
    );
    assert_eq!(word, 0, "the word is untouched");
}

/// The phase of the protocol-fault test: expect 3; the server answers sequence 4, an answer from the future.
fn ahead_phase(hidden: usize) -> Result<(i32, Vec<u32>, Duration), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let partial = pattern_vec(1, 3, hidden);
    let rig = Rig::new(&ctx, hidden)?;
    rig.set_ctrl(&stream, 1, 3)?;
    rig.set_word(&stream, 0)?;
    rig.set_row(&stream, &pattern_vec(0, 3, hidden))?;
    let go = AtomicBool::new(false);
    let started = Instant::now();
    let (queued, served) = thread::scope(|scope| {
        let server = scope.spawn(|| {
            while !go.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            port.answer(4, Ok(&partial))
        });
        let queued = launch(&ctx, &stream, &mailbox, &rig, hidden, 2_000_000_000);
        go.store(true, Ordering::Release);
        (queued, server.join().unwrap_or(false))
    });
    queued?;
    if !served {
        return Err(EngError::BadRequest("the server never answered".to_owned()));
    }
    stream.synchronize()?;
    let elapsed = started.elapsed();
    Ok((rig.word(&ctx)?, rig.row_bits(&ctx, hidden)?, elapsed))
}

/// An answer AHEAD of the expected sequence (the control block was re-zeroed under an old mailbox) is a protocol fault: bit 1
/// of the word, the NaN row, and no wait for the deadline.
#[test]
fn tiered_wait_add_faults_on_an_answer_that_is_ahead() {
    if !testkit::device_present("tiered_wait_add_faults_on_an_answer_that_is_ahead") {
        return;
    }
    let (word, row, elapsed) = ahead_phase(256).expect("the device phase");
    assert_eq!(word, 1, "the protocol fault sets bit 1");
    assert!(
        row.iter().all(|&b| b == QUIET_NAN),
        "the row is the quiet NaN"
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "the fault did not wait for the deadline: {elapsed:?}"
    );
}

/// What the stress phase saw, as plain data.
#[derive(Default)]
struct Rounds {
    done: u64,
    bad: u64,
    first_note: Option<String>,
}

/// 2,000 launches on one mailbox, each with a fresh residual, control block (sequence k) and
/// partial; one server thread answers in order, told each sequence by the host only after the
/// launch is queued, and every result is checked bitwise against the host sum.
fn rounds_phase(hidden: usize, rounds: u64) -> Result<Rounds, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let rig = Rig::new(&ctx, hidden)?;
    let (go, served) = (AtomicU64::new(0), AtomicU64::new(0));
    let stop = AtomicBool::new(false);
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut seen = Rounds::default();
    let (driven, _) = thread::scope(|scope| {
        let server = scope.spawn(|| {
            let mut after = 0u64;
            while after < rounds {
                while go.load(Ordering::Acquire) <= after {
                    if Instant::now() > deadline || stop.load(Ordering::Acquire) {
                        return;
                    }
                    std::hint::spin_loop();
                }
                let k = after + 1;
                if !port.answer(k, Ok(&pattern_vec(1, k, hidden))) {
                    return;
                }
                after = k;
                served.store(k, Ordering::Release);
            }
        });
        let mut driven = Ok(());
        let mut run_round = |k: u64| -> Result<(), EngError> {
            rig.set_ctrl(&stream, 1, k)?;
            rig.set_row(&stream, &pattern_vec(0, k, hidden))?;
            launch(&ctx, &stream, &mailbox, &rig, hidden, 2_000_000_000)?;
            go.store(k, Ordering::Release);
            while served.load(Ordering::Acquire) < k {
                if Instant::now() > deadline {
                    return Err(EngError::BadRequest(format!(
                        "round {k} was never answered"
                    )));
                }
                std::hint::spin_loop();
            }
            stream.synchronize()?;
            let got = rig.row_bits(&ctx, hidden)?;
            let want: Vec<u32> = (0..hidden)
                .map(|j| {
                    let term = 1.0f32 * pattern(1, k, j);
                    (pattern(0, k, j) + term).to_bits()
                })
                .collect();
            seen.done += 1;
            if let Some(j) = (0..hidden).find(|&j| got[j] != want[j]) {
                seen.bad += 1;
                if seen.first_note.is_none() {
                    seen.first_note = Some(format!(
                        "round {k} element {j}: {:#010x}, want {:#010x}",
                        got[j], want[j]
                    ));
                }
            }
            Ok(())
        };
        for k in 1..=rounds {
            driven = run_round(k);
            if driven.is_err() {
                break;
            }
        }
        stop.store(true, Ordering::Release);
        (driven, server.join().ok())
    });
    driven?;
    if served.load(Ordering::Acquire) != rounds {
        return Err(EngError::BadRequest(
            "the server stopped before the last round".to_owned(),
        ));
    }
    Ok(seen)
}

/// Every one of 2,000 waits on one mailbox holds: the sequences rise, the server answers in order
/// and each row equals the host sum bit for bit.
#[test]
fn tiered_wait_add_rounds_hold_over_2000_launches() {
    if !testkit::device_present("tiered_wait_add_rounds_hold_over_2000_launches") {
        return;
    }
    let seen = rounds_phase(256, 2_000).expect("the device phase");
    assert_eq!(seen.done, 2_000, "every round ran");
    assert_eq!(
        seen.bad, 0,
        "every row equalled the host sum: {:?}",
        seen.first_note
    );
}

/// The spare buffers of the refusal phase: three allocations of 16,384 bytes, in the order of the
/// call's ranges.
struct Spare([DeviceBuffer; 3]);

/// One edit of a call: the index of the range (0 ctrl, 1 word, 2 resid), its offset in its
/// allocation and its length (`None`: exact).
type Edit = (usize, usize, Option<usize>);

/// The call of the refusal phase: every range exact and aligned except the `edits`.
fn refusal_call<'b>(
    spare: &'b Spare,
    mailbox: &'b Mailbox,
    hidden: usize,
    edits: &[Edit],
) -> Result<WaitAddCall<'b>, EngError> {
    let sized = [CTRL_BYTES, 4, hidden * 4];
    let r = |i: usize| {
        let edit = edits.iter().find(|e| e.0 == i);
        let (offset, bytes) = edit.map_or((0, sized[i]), |e| (e.1, e.2.unwrap_or(sized[i])));
        spare.0[i].range(offset, bytes)
    };
    Ok(WaitAddCall {
        hidden,
        ctrl: r(0)?,
        word: r(1)?,
        resid: r(2)?,
        mailbox,
        timeout_ns: 2_000_000_000,
    })
}

/// The cases of the refusal test: a name, the edits, and the text the refusal must contain (`None`:
/// accepted).
fn refusal_cases() -> Vec<(&'static str, Vec<Edit>, Option<&'static str>)> {
    vec![
        ("base", vec![], None),
        (
            "short ctrl",
            vec![(0, 0, Some(56))],
            Some("the CUDA ctrl range holds 56 bytes but the call needs 64"),
        ),
        (
            "short word",
            vec![(1, 0, Some(2))],
            Some("the CUDA word range holds 2 bytes but the call needs 4"),
        ),
        (
            "short resid",
            vec![(2, 0, Some(1020))],
            Some("the CUDA resid range holds 1020 bytes but the call needs 8192"),
        ),
        (
            "misaligned ctrl",
            vec![(0, 4, None)],
            Some("the CUDA ctrl range starts at offset 4, which is not a multiple of 8"),
        ),
        (
            "misaligned word",
            vec![(1, 2, None)],
            Some("the CUDA word range starts at offset 2, which is not a multiple of 4"),
        ),
        (
            "misaligned resid",
            vec![(2, 8, None)],
            Some("the CUDA resid range starts at offset 8, which is not a multiple of 16"),
        ),
    ]
}

/// The device phase of the refusal test: each case is one call; the result is the error text, or
/// `None` when the call went through. Then the three extra cases: a written range over another
/// range, and a hidden that differs from the mailbox's.
fn refusal_phase(hidden: usize) -> Result<Vec<Option<String>>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let spare = Spare([
        ctx.upload(&vec![0u8; 16_384])?,
        ctx.upload(&vec![0u8; 16_384])?,
        ctx.upload(&vec![0u8; 16_384])?,
    ]);
    let text = |result: Result<(), EngError>| result.err().map(|e| format!("{e:?}"));
    let mut out = Vec::new();
    for (_, edits, _) in refusal_cases() {
        out.push(text(ctx.tiered_wait_add(
            &stream,
            refusal_call(&spare, &mailbox, hidden, &edits)?,
        )));
    }
    let mut over = refusal_call(&spare, &mailbox, hidden, &[])?;
    over.resid = spare.0[0].range(0, hidden * 4)?;
    out.push(text(ctx.tiered_wait_add(&stream, over)));
    let mut over_word = refusal_call(&spare, &mailbox, hidden, &[])?;
    over_word.word = spare.0[2].range(16, 4)?;
    out.push(text(ctx.tiered_wait_add(&stream, over_word)));
    let mismatch = WaitAddCall {
        hidden: hidden + 4,
        ..refusal_call(&spare, &mailbox, hidden, &[])?
    };
    out.push(text(ctx.tiered_wait_add(&stream, mismatch)));
    stream.synchronize()?;
    Ok(out)
}

/// The wrapper refuses a short range, a misaligned range, a written range over another range and a
/// hidden that differs from the mailbox's, each with its own text before any launch, and accepts
/// the exact call.
#[test]
fn the_wrapper_refuses_bad_ranges_with_their_own_text() {
    if !testkit::device_present("the_wrapper_refuses_bad_ranges_with_their_own_text") {
        return;
    }
    let hidden = 2048usize;
    let got = refusal_phase(hidden).expect("the device phase");
    let mut cases: Vec<(&str, Option<&str>)> = refusal_cases()
        .into_iter()
        .map(|(name, _, text)| (name, text))
        .collect();
    cases.push((
        "resid over ctrl",
        Some("the CUDA resid and ctrl ranges overlap"),
    ));
    cases.push((
        "word over resid",
        Some("the CUDA word and resid ranges overlap"),
    ));
    cases.push((
        "hidden differs from the mailbox",
        Some("the CUDA tiered wait-add takes the hidden width of its mailbox, 2048; got 2052"),
    ));
    assert_eq!(got.len(), cases.len());
    for (text, (name, want)) in got.iter().zip(cases) {
        match (text, want) {
            (None, None) => {}
            (Some(text), Some(want)) => assert!(text.contains(want), "{name}: {text}"),
            (text, want) => panic!("{name}: got {text:?}, wanted {want:?}"),
        }
    }
}

/// Host only: the shape limits, each side of each one, with the text that names the values.
#[test]
fn check_shape_refuses_every_limit_and_names_the_values() {
    let good = WaitAddShape {
        hidden: 2048,
        timeout_ns: 2_000_000_000,
    };
    assert!(check_shape(&good).is_ok());
    let edges = [
        ("hidden 0", WaitAddShape { hidden: 0, ..good }),
        ("hidden 3", WaitAddShape { hidden: 3, ..good }),
        (
            "hidden 65540",
            WaitAddShape {
                hidden: 65540,
                ..good
            },
        ),
        (
            "timeout 0",
            WaitAddShape {
                timeout_ns: 0,
                ..good
            },
        ),
        (
            "timeout past the limit",
            WaitAddShape {
                timeout_ns: MAX_TIMEOUT_NS + 1,
                ..good
            },
        ),
    ];
    for (name, shape) in edges {
        let text = format!("{:?}", check_shape(&shape).expect_err(name));
        assert!(
            text.contains(&format!("hidden {}", shape.hidden)),
            "{name}: {text}"
        );
        assert!(
            text.contains(&format!("timeout {}", shape.timeout_ns)),
            "{name}: {text}"
        );
    }
    let edges_ok = [
        WaitAddShape {
            hidden: 65536,
            timeout_ns: MAX_TIMEOUT_NS,
        },
        WaitAddShape {
            hidden: 4,
            timeout_ns: 1,
        },
    ];
    for shape in edges_ok {
        assert!(check_shape(&shape).is_ok(), "{shape:?}");
    }
}

// ---- source pin ----
//
// The wait's ordering and its failure arithmetic cannot all be seen from the host: a plain load of
// the answer line can look correct on an x86 host whose device writes arrive posted and in order,
// and the poison path needs a device to run at all. So the order the design gives is pinned in the
// source text, whitespace-insensitively, and a mutant that weakens the acquire poll, reads the
// partial with a plain or vector load, drops the 1.0 x p term, swaps the failure bits or drops the
// NaN poison dies here on every host.

#[test]
fn the_wait_add_source_keeps_its_acquire_relaxed_and_nan_semantics() {
    let kernel = squeezed(include_str!("../../native/resident/tiered_wait_add.cu"));
    assert_in_order(
        &kernel,
        &[
            "ld.acquire.sys.u64",
            "__nanosleep",
            "ld.relaxed.sys.u32",
            "__syncthreads()",
            "ld.relaxed.sys.f32",
            "__fadd_rn(",
            "__fmul_rn(1.0f",
            "atomicOr(word, 1)",
            "atomicOr(word, 2)",
            "0x7FC00000",
        ],
        "tiered_wait_add.cu",
    );
    assert!(
        !kernel.contains("a.mbox_partial["),
        "no plain element load of the partial"
    );
    assert!(
        !kernel.contains("*a.mbox_partial"),
        "no plain load of the partial"
    );
    assert!(!kernel.contains("volatile float"), "no volatile float");
    assert!(
        !kernel.contains("ld.relaxed.sys.u64"),
        "the poll is an acquire, never a relaxed load"
    );
    assert!(
        !kernel.contains("partial["),
        "the partial is read only through the relaxed asm load"
    );
    assert!(
        !kernel.contains("#if"),
        "no preprocessor conditional hides a statement"
    );
}

/// The compiled kernel: the PTX of `tiered_wait_add_k` polls `done` with an acquire at system scope and never with a relaxed
/// or volatile load, sleeps between polls, and reads the partial with exactly as many relaxed system loads as it reads
/// residual elements with plain ones (one each per element): a hoisted acquire, a relaxed poll or a plain partial load
/// changes these counts. Skipped when there is no nvcc, unless `INFERENCE_CUDA_REQUIRE_DEVICE` is set.
#[test]
fn the_compiled_wait_add_polls_with_an_acquire_and_reads_the_partial_relaxed() {
    let files = [
        ("abi.h", include_str!("../../native/abi.h")),
        ("device.h", include_str!("../../native/device.h")),
        (
            "resident/tiered_wait_add.h",
            include_str!("../../native/resident/tiered_wait_add.h"),
        ),
        (
            "resident/tiered_wait_add.cu",
            include_str!("../../native/resident/tiered_wait_add.cu"),
        ),
    ];
    let Some(lines) =
        testkit::kernel_ptx("tiered_wait_add_k", "resident/tiered_wait_add.cu", &files)
    else {
        return;
    };
    let count = |prefix: &str| lines.iter().filter(|l| l.starts_with(prefix)).count();
    assert!(
        count("ld.acquire.sys.u64") >= 1,
        "the poll is an acquire at system scope"
    );
    assert_eq!(count("ld.relaxed.sys.u64"), 0, "no relaxed poll");
    assert_eq!(count("ld.volatile"), 0, "no volatile load");
    assert!(count("nanosleep.u32") >= 1, "the poll sleeps");
    assert!(count("ld.relaxed.sys.u32") >= 1, "the code is read relaxed");
    assert_eq!(
        count("ld.relaxed.sys.f32"),
        count("ld.global.f32"),
        "one relaxed partial load per plain residual load"
    );
    assert!(
        count("ld.relaxed.sys.f32") >= 1,
        "the partial is read with relaxed system loads"
    );
}
