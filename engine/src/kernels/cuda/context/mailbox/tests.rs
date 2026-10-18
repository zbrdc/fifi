//! Tests of the context mailbox (TCAP C1b). The host-only tests run the port over ordinary memory; the device
//! tests use the real mapping and the real classify kernel as the publisher: `mailbox_orders_the_row_before_the_
//! sequence_under_stress` is the user-approved 10,000-request stress test (2026-10-13), the one that must kill the
//! ordering mutants of the publication in `tiered_classify.cu`.
//!
//! Every device phase is a function that creates its own context, stream and buffers, has only `?` inside, drops
//! every owner when it returns and hands back plain host records; the tests assert on those records afterwards
//! (an assertion that panics while a context is alive would latch the backend for the rest of a one-process run).
//! Run with `--test-threads=1`.

use super::*;
use crate::kernels::cuda::context::DeviceStream;
use crate::kernels::cuda::native::resident::RawRange;
use crate::kernels::cuda::native::resident::mailbox::tests::over_host_memory;
use crate::kernels::cuda::native::resident::tiered_classify::{
    ClassifyRaw, tiered_classify_launch,
};
use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::testkit::{self, assert_in_order, squeezed};
use crate::kernels::cuda::{CapturePlan, DeviceBuffer};
use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::thread;
use std::time::{Duration, Instant};

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Element `j` of the row (`salt` 0) or the partial (`salt` 1) of request `k`: a finite float in [1, 2) derived from a
/// hash of `(k, j)`, so a stale or torn word equals the expected one only by a 2^-23 collision per element.
fn pattern(salt: u64, k: u64, j: usize) -> f32 {
    f32::from_bits(0x3F80_0000 | (mix((k << 24) ^ ((j as u64) << 1) ^ salt) as u32 & 0x007F_FFFF))
}

fn pattern_vec(salt: u64, k: u64, hidden: usize) -> Vec<f32> {
    (0..hidden).map(|j| pattern(salt, k, j)).collect()
}

fn host_port(backing: &mut [u64], hidden: usize) -> MailboxPort {
    let layout = MailboxLayout::new(hidden).expect("a layout");
    assert!(
        size_of_val(backing) >= layout.total_bytes(),
        "the backing holds the mailbox"
    );
    MailboxPort {
        shared: Arc::new(Shared {
            raw: over_host_memory(backing),
            layout,
        }),
    }
}

/// What the device does for one request, on the host: row and header first, then the release of `seq`.
fn publish(port: &MailboxPort, seq: u64, row: &[f32]) {
    let layout = port.shared.layout;
    let raw = &port.shared.raw;
    assert!(raw.write_f32s(layout.row_offset(), row));
    let header: Vec<u32> = [3, 1, 7, 0, 0, 0, 0, 0, 0, 0]
        .iter()
        .chain(&[0u32; 8])
        .copied()
        .collect();
    assert!(raw.write_words(LAYER_AT, &header));
    assert!(raw.store_release(0, seq));
}

#[test]
fn the_layout_puts_every_line_on_a_boundary_and_refuses_bad_widths() {
    for hidden in [4usize, 12, 256, 2048, 65536] {
        let layout = MailboxLayout::new(hidden).expect("a layout");
        assert_eq!(layout.row_offset(), REQUEST_BYTES);
        assert_eq!(layout.answer_offset() % LINE, 0);
        assert!(
            layout.answer_offset() >= layout.row_offset() + hidden * 4,
            "the row fits before the answer line"
        );
        assert_eq!(
            layout.partial_offset(),
            layout.answer_offset() + ANSWER_BYTES
        );
        assert!(
            layout.total_bytes() >= layout.partial_offset() + hidden * 4,
            "the partial fits"
        );
    }
    assert_eq!(
        MailboxLayout::new(2048).expect("layout").total_bytes(),
        128 + 8192 + 128 + 8192
    );
    for bad in [0usize, 1, 3, 6, 65540, usize::MAX] {
        assert!(
            MailboxLayout::new(bad).is_err(),
            "hidden {bad} was accepted"
        );
    }
}

#[test]
fn the_port_polls_the_header_once_per_sequence_and_answers_with_the_partial_or_a_code() {
    let mut backing = vec![0u64; 600];
    let port = host_port(&mut backing, 256);
    assert_eq!(port.poll(0), None, "nothing published yet");
    let row = pattern_vec(0, 1, 256);
    publish(&port, 1, &row);
    let request = port.poll(0).expect("the request");
    assert_eq!((request.seq, request.layer, request.count), (1, 3, 1));
    assert_eq!(request.ids[0], 7);
    assert_eq!(port.poll(1), None, "an answered sequence is not seen again");
    let mut seen = vec![0f32; 256];
    assert!(port.read_row(&mut seen));
    assert_eq!(
        seen.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        row.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(
        !port.read_row(&mut [0f32; 255]),
        "a row of another width is refused"
    );
    let layout = port.shared.layout;
    let partial = pattern_vec(1, 1, 256);
    assert!(port.answer(1, Ok(&partial)));
    assert_eq!(
        port.shared.raw.load_acquire(layout.answer_offset()),
        Some(1)
    );
    let mut code = [9u32];
    assert!(
        port.shared
            .raw
            .read_words(layout.answer_offset() + CODE_AT, &mut code)
    );
    assert_eq!(code, [0], "a partial answer carries code 0");
    assert!(port.answer(2, Err(5)));
    assert!(
        port.shared
            .raw
            .read_words(layout.answer_offset() + CODE_AT, &mut code)
    );
    assert_eq!(
        (port.shared.raw.load_acquire(layout.answer_offset()), code),
        (Some(2), [5])
    );
    assert!(
        !port.answer(3, Err(0)),
        "an error answer needs a nonzero code"
    );
    assert!(
        !port.answer(3, Ok(&partial[..255])),
        "a partial of another width is refused"
    );
    assert_eq!(
        port.shared.raw.load_acquire(layout.answer_offset()),
        Some(2),
        "a refused answer wrote nothing"
    );
    // A later request brings its own header: the poll is per sequence, not once.
    let second_row = pattern_vec(0, 2, 256);
    publish(&port, 2, &second_row);
    assert!(port.shared.raw.write_words(LAYER_AT, &[9, 2, 11, 12]));
    let later = port.poll(1).expect("the second request");
    assert_eq!(
        (later.seq, later.layer, later.count, later.ids[..2].to_vec()),
        (2, 9, 2, vec![11, 12])
    );
    assert_eq!(port.poll(2), None);
    assert!(port.read_row(&mut vec![0f32; 256]));
}

/// The reader's result when its wait for `done` ran out of time (no word was judged).
const TIMED_OUT: u64 = u64::MAX;

/// The host half of the answer: a reader that sees `done = n` must see the partial and the code of answer `n`. The
/// reader plays the device; the port answers from this thread. A `done` released before the partial is written (or
/// a code written after it) shows as a stale word.
#[test]
fn an_answer_writes_the_partial_and_the_code_before_it_releases_done() {
    const ROUNDS: u64 = 4000;
    const HIDDEN: usize = 2048;
    let mut backing = vec![0u64; 2200];
    let port = host_port(&mut backing, HIDDEN);
    let layout = port.shared.layout;
    let deadline = Instant::now() + Duration::from_secs(20);
    let bad = thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let raw = &port.shared.raw;
            let (mut bad, mut seen) = (0u64, vec![0f32; HIDDEN]);
            for round in 1..=ROUNDS {
                publish(&port, round, &pattern_vec(0, round, 8));
                while raw.load_acquire(layout.answer_offset()).unwrap_or(0) < round {
                    if Instant::now() > deadline {
                        return TIMED_OUT;
                    }
                    std::hint::spin_loop();
                }
                let mut code = [0u32];
                // The code first, within nanoseconds of the acquire, so that a code written after `done` is seen stale.
                assert!(raw.read_words(layout.answer_offset() + CODE_AT, &mut code));
                assert!(raw.read_f32s(layout.partial_offset(), &mut seen));
                let want = round % 3;
                bad += u64::from(code[0] != want as u32);
                if want == 0 {
                    bad += seen
                        .iter()
                        .enumerate()
                        .filter(|&(j, v)| v.to_bits() != pattern(1, round, j).to_bits())
                        .count() as u64;
                }
            }
            bad
        });
        let mut after = 0;
        while after < ROUNDS && Instant::now() < deadline {
            if let Some(request) = port.poll(after) {
                let partial = pattern_vec(1, request.seq, HIDDEN);
                let result = if request.seq.is_multiple_of(3) {
                    Ok(partial.as_slice())
                } else {
                    Err((request.seq % 3) as u32)
                };
                assert!(port.answer(request.seq, result));
                after = request.seq;
            } else {
                std::hint::spin_loop();
            }
        }
        reader.join().expect("the reader")
    });
    assert_ne!(
        bad, TIMED_OUT,
        "an answer never arrived within the deadline"
    );
    assert_eq!(bad, 0, "words of an older answer were read after done");
}

// ---- device tests ----

const TOP_K: usize = 8;
const N_EXPERT: usize = 256;
/// The bank slot a resident pick of the stress maps to (the bank has 8 slots).
const HIT_SLOT: i32 = 3;

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The eight distinct expert ids of request `k` (29 is coprime to 256).
fn ids_of(k: u64) -> [i32; TOP_K] {
    let base = (k * 37 % N_EXPERT as u64) as i32;
    std::array::from_fn(|i| (base + 29 * i as i32) % N_EXPERT as i32)
}

fn weights_of(k: u64) -> [f32; TOP_K] {
    std::array::from_fn(|i| (i as f32 + 1.0) / 64.0 + (k % 7) as f32)
}

/// The picks of request `k` that are resident: one in four requests has one hit, one in 101 is all hits (no
/// publication at all), the rest miss everywhere.
fn hits_of(k: u64) -> Vec<usize> {
    if k.is_multiple_of(101) {
        (0..TOP_K).collect()
    } else if k.is_multiple_of(4) {
        vec![(k / 4 % TOP_K as u64) as usize]
    } else {
        Vec::new()
    }
}

/// The misses `(id, weight)` of request `k` in pick order.
fn misses_of(k: u64) -> Vec<(i32, f32)> {
    let hits = hits_of(k);
    let (ids, weights) = (ids_of(k), weights_of(k));
    (0..TOP_K)
        .filter(|i| !hits.contains(i))
        .map(|i| (ids[i], weights[i]))
        .collect()
}

fn layer_of(k: u64) -> u32 {
    (k % 40) as u32
}

/// What the server thread saw, as plain data.
#[derive(Debug, Default)]
struct Outcome {
    served: u64,
    bad: u64,
    notes: Vec<String>,
    elapsed: Duration,
}

impl Outcome {
    fn flag(&mut self, note: String) {
        self.bad += 1;
        if self.notes.len() < 6 {
            self.notes.push(note);
        }
    }
}

/// The server: polls as fast as it can, checks every request against the plan of its sequence, answers.
fn serve(
    port: &MailboxPort,
    published: &[u64],
    answered: &AtomicU64,
    stop: &AtomicBool,
    deadline: Instant,
) -> Outcome {
    let (mut out, mut after, mut row) = (Outcome::default(), 0u64, vec![0f32; port.hidden()]);
    let mut spins = 0u32;
    while !stop.load(Ordering::Acquire) {
        spins = spins.wrapping_add(1);
        if spins.is_multiple_of(4096) && Instant::now() > deadline {
            out.flag("the server's own deadline passed".to_owned());
            break;
        }
        let Some(request) = port.poll(after) else {
            std::hint::spin_loop();
            continue;
        };
        let seq = after + 1;
        let Some(&k) = published.get((seq - 1) as usize) else {
            out.flag(format!("request {} past the plan", request.seq));
            break;
        };
        let misses = misses_of(k);
        let header_ok = request.seq == seq
            && request.layer == layer_of(k)
            && request.count as usize == misses.len()
            && (0..TOP_K).all(|i| {
                let (id, w) = misses.get(i).copied().unwrap_or((-1, 0.0));
                request.ids[i] == id && request.weights[i].to_bits() == w.to_bits()
            });
        if !header_ok {
            out.flag(format!(
                "seq {seq} (k {k}): header {request:?}, want layer {} misses {misses:?}",
                layer_of(k)
            ));
        }
        if !port.read_row(&mut row) {
            out.flag(format!("seq {seq}: the row read was refused"));
        } else if let Some(j) =
            (0..row.len()).find(|&j| row[j].to_bits() != pattern(0, k, j).to_bits())
        {
            out.flag(format!(
                "seq {seq} (k {k}): row[{j}] = {:#010x}, want {:#010x}",
                row[j].to_bits(),
                pattern(0, k, j).to_bits()
            ));
        }
        if !port.answer(seq, Ok(&pattern_vec(1, k, row.len()))) {
            out.flag(format!("seq {seq}: the answer was refused"));
        }
        after = seq;
        out.served += 1;
        answered.store(seq, Ordering::Release);
    }
    out
}

fn raw_of<'b>(
    ctx: &CudaContext,
    buffer: &'b DeviceBuffer,
    bytes: usize,
) -> Result<RawRange<'b>, EngError> {
    ctx.raw_range(&buffer.range(0, bytes)?)
}

/// The device buffers of the stress and the launch of one classification into the mailbox.
struct Rig<'c> {
    ctx: &'c CudaContext,
    stream: &'c DeviceStream,
    mailbox: &'c Mailbox,
    hidden: usize,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    slot_row: DeviceBuffer,
    norm_row: DeviceBuffer,
    outs: [DeviceBuffer; 5],
}

impl Rig<'_> {
    fn launch(&self, layer: u32) -> Result<(), EngError> {
        let ctx = self.ctx;
        let r = |buffer, bytes| raw_of(ctx, buffer, bytes);
        let [slot_ids_out, weights_out, log_out, ctrl, word] = &self.outs;
        let call = ClassifyRaw {
            ids: r(&self.ids, TOP_K * 4)?,
            weights: r(&self.weights, TOP_K * 4)?,
            slot_row: r(&self.slot_row, N_EXPERT * 4)?,
            norm_row: r(&self.norm_row, self.hidden * 4)?,
            slot_ids_out: r(slot_ids_out, TOP_K * 4)?,
            weights_out: r(weights_out, TOP_K * 4)?,
            log_out: r(log_out, TOP_K * 8)?,
            ctrl: r(ctrl, 64)?,
            word: r(word, 4)?,
            mbox_req: self.mailbox.request_range()?,
            mbox_row: self.mailbox.row_range()?,
            n_expert: N_EXPERT as c_int,
            slots: 8,
            top_k: TOP_K as c_int,
            hidden: self.hidden as c_int,
            layer: layer as c_int,
        };
        ctx.launch_on(self.stream, |raw| tiered_classify_launch(raw, &call))
    }

    fn write(&self, buffer: &DeviceBuffer, offset: usize, bytes: &[u8]) -> Result<(), EngError> {
        buffer.write_at(self.stream, offset, bytes)
    }
}

/// The device phase of the stress: request after request, the next only after the server answered the last.
fn drive(
    rig: &Rig<'_>,
    requests: u64,
    answered: &AtomicU64,
    deadline: Instant,
) -> Result<(), EngError> {
    let mut seq = 0u64;
    for k in 1..=requests {
        let (ids, hits) = (ids_of(k), hits_of(k));
        rig.write(&rig.ids, 0, &bytes_i32(&ids))?;
        rig.write(&rig.weights, 0, &bytes_f32(&weights_of(k)))?;
        rig.write(&rig.norm_row, 0, &bytes_f32(&pattern_vec(0, k, rig.hidden)))?;
        for &i in &hits {
            rig.write(&rig.slot_row, ids[i] as usize * 4, &HIT_SLOT.to_le_bytes())?;
        }
        rig.launch(layer_of(k))?;
        rig.stream.synchronize()?;
        if hits.len() < TOP_K {
            seq += 1;
            let mut spins = 0u32;
            while answered.load(Ordering::Acquire) < seq {
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(1024) && Instant::now() > deadline {
                    return Err(refuse(format!(
                        "request {k} (seq {seq}) was never answered"
                    )));
                }
                std::hint::spin_loop();
            }
        }
        for &i in &hits {
            rig.write(&rig.slot_row, ids[i] as usize * 4, &(-1i32).to_le_bytes())?;
        }
    }
    Ok(())
}

/// The stress: `requests` requests of `hidden`-wide rows through the real classify kernel into the real mapping,
/// a server thread checking each one. Returns what the server saw; a device-side failure is the `Err`.
fn stress(hidden: usize, requests: u64) -> Result<Outcome, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mailbox = ctx.mailbox(hidden)?;
    let port = mailbox.port()?;
    let rig = Rig {
        ctx: &ctx,
        stream: &stream,
        mailbox: &mailbox,
        hidden,
        ids: ctx.upload(&[0; TOP_K * 4])?,
        weights: ctx.upload(&[0; TOP_K * 4])?,
        slot_row: ctx.upload(&bytes_i32(&[-1; N_EXPERT]))?,
        norm_row: ctx.upload(&vec![0; hidden * 4])?,
        outs: [
            ctx.upload(&[0; TOP_K * 4])?,
            ctx.upload(&[0; TOP_K * 4])?,
            ctx.upload(&[0; TOP_K * 8])?,
            ctx.upload(&[0; 64])?,
            ctx.upload(&[0; 4])?,
        ],
    };
    let published: Vec<u64> = (1..=requests)
        .filter(|&k| !misses_of(k).is_empty())
        .collect();
    let (answered, stop) = (AtomicU64::new(0), AtomicBool::new(false));
    let seconds = std::env::var("TCAP_STRESS_DEADLINE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    let began = Instant::now();
    let deadline = began + Duration::from_secs(seconds);
    let (driven, outcome) = thread::scope(|scope| {
        let server = scope.spawn(|| serve(&port, &published, &answered, &stop, deadline));
        let driven = drive(&rig, requests, &answered, deadline);
        stop.store(true, Ordering::Release);
        let outcome = server.join().unwrap_or_else(|_| Outcome {
            bad: 1,
            notes: vec!["the server thread panicked".to_owned()],
            ..Outcome::default()
        });
        (driven, outcome)
    });
    driven?;
    stream.synchronize()?;
    Ok(Outcome {
        bad: outcome.bad + u64::from(outcome.served != published.len() as u64),
        elapsed: began.elapsed(),
        ..outcome
    })
}

#[test]
fn mailbox_orders_the_row_before_the_sequence_under_stress() {
    if !testkit::device_present("mailbox_orders_the_row_before_the_sequence_under_stress") {
        return;
    }
    let outcome = stress(2048, 10_000).expect("the device phase");
    eprintln!(
        "stress hidden 2048: {} requests served in {:?}",
        outcome.served, outcome.elapsed
    );
    assert_eq!(
        outcome.bad, 0,
        "the server saw wrong requests: {:?}",
        outcome.notes
    );
}

#[test]
fn mailbox_stress_holds_at_the_smallest_and_the_widest_rows() {
    if !testkit::device_present("mailbox_stress_holds_at_the_smallest_and_the_widest_rows") {
        return;
    }
    let small = stress(4, 4_000).expect("the device phase at hidden 4");
    let wide = stress(65536, 400).expect("the device phase at hidden 65536");
    assert_eq!(small.bad, 0, "hidden 4: {:?}", small.notes);
    assert_eq!(wide.bad, 0, "hidden 65536: {:?}", wide.notes);
}

/// What the lifecycle phase saw, as plain data.
#[derive(Debug, Default)]
struct Lifecycle {
    counted: usize,
    total: usize,
    zeroed: bool,
    leaked_with_a_port: bool,
    port_still_reads: bool,
    freed_without_a_port: bool,
    native_frees: usize,
    refusals: Vec<String>,
}

/// The device phase of the lifecycle test: allocation count, zeroing, the leak with a live port, the free without one,
/// and the refusals (a bad width, a call inside a capture window).
fn lifecycle() -> Result<Lifecycle, EngError> {
    let q8_len = q8_1_bytes(256, 1).ok_or_else(|| refuse("the q8 size"))?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (before, destroys) = (ctx.live_owners().pinned, ctx.destroy_calls());
    let mut seen = Lifecycle::default();
    let mailbox = ctx.mailbox(2048)?;
    seen.total = mailbox.layout().total_bytes();
    seen.counted = ctx.live_owners().pinned - before;
    let port = mailbox.port()?;
    let mut row = vec![1f32; 2048];
    seen.zeroed =
        port.poll(0).is_none() && port.read_row(&mut row) && row.iter().all(|v| v.to_bits() == 0);
    // A port is alive: dropping the owner must leak the pages, not free memory the server may still read.
    drop(mailbox);
    seen.leaked_with_a_port =
        ctx.live_owners().pinned - before == seen.total && ctx.destroy_calls() == destroys;
    seen.port_still_reads = port.read_row(&mut row) && port.poll(0).is_none();
    drop(port);
    let second = ctx.mailbox(256)?;
    drop(second);
    seen.freed_without_a_port = ctx.live_owners().pinned - before == seen.total;
    seen.native_frees = ctx.destroy_calls() - destroys;
    seen.refusals
        .push(format!("{:?}", ctx.mailbox(3).map(|_| ())));
    let x = ctx.upload(&bytes_f32(&[0.5f32; 256]))?;
    let q8 = ctx.upload(&vec![0u8; q8_len])?;
    let quantize = || {
        let call = QuantizeCall {
            x: x.range(0, 1024)?,
            y: q8.range(0, q8_len)?,
            k: 256,
            ncols: 1,
        };
        ctx.quantize_q8_1(&stream, call)
    };
    quantize()?;
    stream.synchronize()?;
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let inside = std::cell::RefCell::new(Vec::new());
    ctx.capture(&stream, plan, || {
        quantize()?;
        inside
            .borrow_mut()
            .push(format!("{:?}", ctx.mailbox(4).map(|_| ())));
        Ok(())
    })?;
    seen.refusals.extend(inside.into_inner());
    Ok(seen)
}

#[test]
fn a_mailbox_is_counted_zeroed_leaked_while_a_port_lives_and_refused_inside_a_capture_window() {
    if !testkit::device_present(
        "a_mailbox_is_counted_zeroed_leaked_while_a_port_lives_and_refused_inside_a_capture_window",
    ) {
        return;
    }
    let seen = lifecycle().expect("the device phase");
    assert_eq!(
        seen.counted, seen.total,
        "the pinned count covers the whole mapping"
    );
    assert!(seen.zeroed, "the mapping starts zeroed and unpublished");
    assert!(
        seen.leaked_with_a_port,
        "dropping the owner freed memory a port still holds"
    );
    assert!(
        seen.port_still_reads,
        "the port read the mapping after the owner was dropped"
    );
    assert!(
        seen.freed_without_a_port,
        "a mailbox without a port is released on drop"
    );
    assert_eq!(
        seen.native_frees, 1,
        "exactly the mailbox without a port was freed"
    );
    assert!(
        seen.refusals[0].contains("multiple of 4"),
        "{:?}",
        seen.refusals
    );
    assert!(
        seen.refusals[1].contains("not allowed inside a capture window"),
        "{:?}",
        seen.refusals
    );
}

// ---- source pins ----
//
// The ordering of the publication cannot be seen on every host: on the x86 laptop and the 3090 box the device-to-host
// writes of this kernel arrive in order even without the fences (the stress test above survives a kernel with no
// ordering at all), and a host-side acquire/release mutant only shows on the aarch64 Jetsons. So the order the review
// approved is pinned in the source text, whitespace-insensitively, and the protocol steps are pinned in the order the
// design gives them. A mutant that drops a fence, weakens the release store or reorders the host steps dies here on
// every host.

/// The memory order named by the first `Ordering::` after `function` in `text`.
fn first_ordering<'t>(text: &'t str, function: &str) -> &'t str {
    let body = &text[text
        .find(function)
        .unwrap_or_else(|| panic!("`{function}` not found"))..];
    let named = &body[body.find("Ordering::").expect("an ordering") + "Ordering::".len()..];
    named
        .split(|c: char| !c.is_alphanumeric())
        .next()
        .expect("a name")
}

#[test]
fn the_kernel_publishes_the_row_before_the_sequence_with_two_fences_and_a_release_store() {
    let kernel = squeezed(include_str!("../../native/resident/tiered_classify.cu"));
    assert_in_order(
        &kernel,
        &[
            "dst[j] = src[j];",
            "__threadfence_system(); __syncthreads(); if (threadIdx.x == 0) {",
            "a.mbox_req[1] =",
            "__threadfence_system(); const uint64_t seq = a.ctrl[2] + 1;",
            "store_release_system(a.mbox_req, seq);",
        ],
        "tiered_classify.cu",
    );
    assert!(
        kernel.contains("asm volatile(\"st.release.sys.u64 [%0], %1;\""),
        "the sequence is released with st.release.sys"
    );
    assert!(
        !kernel.contains("a.mbox_req[0]"),
        "no plain store to the sequence word"
    );
}

#[test]
fn the_c_entry_maps_and_zeroes_the_allocation() {
    let entry = squeezed(include_str!("../../native/resident/mailbox.cu"));
    assert_in_order(
        &entry,
        &[
            "cudaHostAlloc(&host, bytes, cudaHostAllocMapped)",
            "cudaHostGetDevicePointer(&device, host, 0)",
            "memset(host, 0, bytes);",
        ],
        "mailbox.cu",
    );
}

#[test]
fn the_host_side_keeps_its_acquire_and_release_and_the_steps_in_protocol_order() {
    let accessors = squeezed(include_str!("../../native/resident/mailbox.rs"));
    assert_eq!(first_ordering(&accessors, "fn load_acquire"), "Acquire");
    assert_eq!(first_ordering(&accessors, "fn store_release"), "Release");
    let port = squeezed(include_str!("../mailbox.rs"));
    assert_in_order(
        &port,
        &["fn poll", "raw.load_acquire(0)", "raw.read_words(LAYER_AT"],
        "poll: acquire seq, then the header",
    );
    assert_in_order(
        &port,
        &[
            "fn answer",
            "write_f32s(layout.partial_offset(), partial)",
            "raw.write_words(layout.answer_offset() + CODE_AT",
            "&& raw.store_release(layout.answer_offset(), seq)",
        ],
        "answer: the partial, then the code, then done",
    );
}

/// The compiled kernel: nvcc turns the embedded source into PTX and the LAST store of the kernel, the sequence, must be a
/// system-scope release, or a strong (relaxed or volatile) store behind a system fence. Comments and statements that only
/// look right in the source cannot fool it, and the PTX-equivalent variants (a missing fence behind a release store,
/// `st.relaxed.sys` behind the fences) pass. Skipped when there is no nvcc, unless `INFERENCE_CUDA_REQUIRE_DEVICE` is set.
#[test]
fn the_compiled_kernel_ends_with_a_system_scope_release_of_the_sequence() {
    let files = [
        ("abi.h", include_str!("../../native/abi.h")),
        ("device.h", include_str!("../../native/device.h")),
        (
            "resident/tiered_classify.h",
            include_str!("../../native/resident/tiered_classify.h"),
        ),
        (
            "resident/tiered_classify.cu",
            include_str!("../../native/resident/tiered_classify.cu"),
        ),
    ];
    let Some(lines) =
        testkit::kernel_ptx("tiered_classify_k", "resident/tiered_classify.cu", &files)
    else {
        return;
    };
    let last_store = lines
        .iter()
        .rposition(|l| l.starts_with("st."))
        .expect("the kernel stores");
    let fence = |l: &String| {
        ["membar.sys", "fence.sc.sys", "fence.acq_rel.sys"]
            .iter()
            .any(|f| l.starts_with(f))
    };
    let release = lines[last_store].contains(".release.sys.");
    // A strong store counts only behind the NEAREST system fence, and only when no row or header store (anything
    // wider or narrower than the 64-bit control words) follows that fence: a fence before the header stores
    // orders nothing of what the sequence publishes.
    let nearest_fence = lines[..last_store].iter().rposition(fence);
    let fenced_strong = (lines[last_store].contains(".relaxed.sys.")
        || lines[last_store].contains(".volatile."))
        && nearest_fence.is_some_and(|at| {
            !lines[at..last_store]
                .iter()
                .any(|l| l.starts_with("st.") && !l.contains(".u64") && !l.contains(".b64"))
        });
    assert!(
        release || fenced_strong,
        "the last store of the kernel is `{}`: not a system-scope release",
        lines[last_store]
    );
    assert!(
        lines[last_store].contains(".u64"),
        "the sequence is a 64-bit word: `{}`",
        lines[last_store]
    );
}
