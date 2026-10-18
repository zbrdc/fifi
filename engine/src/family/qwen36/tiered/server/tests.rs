//! Tests of the CPU-expert server (TCAP C5): a mock port scripts the device's
//! published requests and records the answers, while the locator, the shared region
//! and the CPU pool are real, over the synthetic GGUF fixtures of the locator tests —
//! so every answer is checked against the same jobs run through a fresh pool.

use crate::family::qwen36::tiered::cpu_pool::{CpuPool, ExpertJob, MappedRegion, PoolError};
use crate::family::qwen36::tiered::locator::ExpertLocator;
use crate::family::qwen36::tiered::locator::ExpertTensorNames;
use crate::family::qwen36::tiered::locator::LocatorError;
use crate::family::qwen36::tiered::locator::tests::{Lcg, expert_file, parse};
use crate::family::qwen36::tiered::server::{
    ERR_ANSWER, ERR_BAD_REQUEST, ERR_DEAD, ERR_LOCATOR, ERR_MISSING, ERR_NON_FINITE, ERR_POOL,
    ERR_ROW, LayerServer, RequestPort, ServeError, ServerRequest,
};
use crate::kernels::expert_cpu::ExpertShape;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The fixture shape: small enough that a whole token of experts runs in milliseconds.
const SHAPE: ExpertShape = ExpertShape {
    hidden: 256,
    inter: 256,
};
/// The synthetic model: 3 layers of 8 experts.
const LAYERS: usize = 3;
const EXPERTS: usize = 8;
/// Generous pool timeout, reached only if something is broken.
const LONG: Duration = Duration::from_secs(60);

/// One recorded answer: the sequence and the bits of the partial or the error code.
type Recorded = (u64, Result<Vec<u32>, u32>);

/// The scripted device side of the mailbox: `poll` pops the next published request
/// whatever its sequence is (the SERVER owns the staleness rule, so a stale or
/// repeated pop reaches it), `read_row` serves the scripted row after an optional
/// delay (whose entry flag lets a test observe the thread inside a request) or refuses
/// it on demand, `answer` records every `(seq, bits | code)` it was given, and the
/// drop flag observes the server thread's death through the port it owns.
struct MockInner {
    requests: Mutex<VecDeque<ServerRequest>>,
    answers: Mutex<Vec<Recorded>>,
    row: Vec<f32>,
    refuse_row: AtomicBool,
    refuse_answer: AtomicBool,
    panic_on_poll: AtomicBool,
    panic_on_row: AtomicBool,
    block_poll: AtomicBool,
    row_entered: AtomicBool,
    row_delay_ms: AtomicUsize,
    polls: AtomicUsize,
    dropped: AtomicBool,
}

impl MockInner {
    /// Publishes a request, as the device would.
    fn publish(&self, request: ServerRequest) {
        self.requests.lock().unwrap().push_back(request);
    }

    /// The recorded answers, in order.
    fn answers(&self) -> Vec<Recorded> {
        self.answers.lock().unwrap().clone()
    }
}

/// The port handle; the server thread owns the only other one, so its `Drop` marks the
/// thread gone.
#[derive(Clone)]
struct MockPort(Arc<MockInner>);

impl MockPort {
    /// A port whose rows are `row` (its width is the row's length).
    fn new(row: Vec<f32>) -> (MockPort, Arc<MockInner>) {
        let inner = Arc::new(MockInner {
            requests: Mutex::new(VecDeque::new()),
            answers: Mutex::new(Vec::new()),
            row,
            refuse_row: AtomicBool::new(false),
            refuse_answer: AtomicBool::new(false),
            panic_on_poll: AtomicBool::new(false),
            panic_on_row: AtomicBool::new(false),
            block_poll: AtomicBool::new(false),
            row_entered: AtomicBool::new(false),
            row_delay_ms: AtomicUsize::new(0),
            polls: AtomicUsize::new(0),
            dropped: AtomicBool::new(false),
        });
        (MockPort(Arc::clone(&inner)), inner)
    }
}

impl RequestPort for MockPort {
    fn hidden(&self) -> usize {
        self.0.row.len()
    }

    fn poll(&self, _after: u64) -> Option<ServerRequest> {
        self.0.polls.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.0.panic_on_poll.load(Ordering::Relaxed),
            "scripted panic in poll"
        );
        while self.0.block_poll.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.0.requests.lock().unwrap().pop_front()
    }

    fn read_row(&self, out: &mut [f32]) -> bool {
        if self.0.refuse_row.load(Ordering::Relaxed) {
            return false;
        }
        assert!(
            !self.0.panic_on_row.swap(false, Ordering::Relaxed),
            "scripted panic in read_row"
        );
        self.0.row_entered.store(true, Ordering::Relaxed);
        let delay = self.0.row_delay_ms.load(Ordering::Relaxed);
        if delay > 0 {
            std::thread::sleep(Duration::from_millis(delay as u64));
        }
        if out.len() != self.0.row.len() {
            return false;
        }
        out.copy_from_slice(&self.0.row);
        true
    }

    fn answer(&self, seq: u64, result: Result<&[f32], u32>) -> bool {
        let recorded = match result {
            Ok(partial) => Ok(partial.iter().map(|value| value.to_bits()).collect()),
            Err(code) => Err(code),
        };
        self.0.answers.lock().unwrap().push((seq, recorded));
        !self.0.refuse_answer.load(Ordering::Relaxed)
    }
}

impl Drop for MockPort {
    fn drop(&mut self) {
        self.0.dropped.store(true, Ordering::Relaxed);
    }
}

/// A region that counts its `bytes()` calls and panics on the `panic_at`-th one
/// (`panic_at == 0` never panics). The pool calls `bytes()` once per job inside
/// `start_layer` (validation, server thread) and once inside the worker, so a chosen
/// call number kills a worker thread (the `WorkerLost` path) and the count itself
/// proves whether the pool was asked to start anything at all.
struct CountingRegion {
    bytes: Arc<Vec<u8>>,
    calls: Arc<AtomicUsize>,
    panic_at: u64,
}

impl MappedRegion for CountingRegion {
    fn bytes(&self) -> &[u8] {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) as u64 + 1;
        if call == self.panic_at {
            panic!("region poisoned at bytes() call {call}");
        }
        self.bytes.as_slice()
    }
}

/// A synthetic expert file, the shared region over its bytes, and the normed row.
fn fixture(seed: u64) -> (Vec<u8>, Arc<dyn MappedRegion>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let ((bytes, _), _) = expert_file(&mut rng, LAYERS, EXPERTS, SHAPE);
    let row: Vec<f32> = (0..SHAPE.hidden)
        .map(|_| rng.f32_between(-1.0, 1.0))
        .collect();
    let region: Arc<dyn MappedRegion> = Arc::new(bytes.clone());
    (bytes, region, row)
}

/// A locator over the fixture's bytes (tests build as many as they need).
fn locate(bytes: &[u8]) -> ExpertLocator {
    let header = parse(bytes);
    ExpertLocator::new(
        &header,
        &ExpertTensorNames::llama_cpp(),
        LAYERS,
        EXPERTS,
        SHAPE,
    )
    .unwrap()
}

/// One request: `count` picks, the rest of the header padded as the device pads it
/// (ids -1, weights 0.0); more than 8 picks fill the array and keep the count, which
/// is exactly the `count > 8` refusal case.
fn request(seq: u64, layer: u32, ids: &[i32], weights: &[f32]) -> ServerRequest {
    let mut out = ServerRequest {
        seq,
        layer,
        count: ids.len() as u32,
        ids: [-1; 8],
        weights: [0.0; 8],
    };
    let picks = ids.len().min(8);
    out.ids[..picks].copy_from_slice(&ids[..picks]);
    out.weights[..picks].copy_from_slice(&weights[..picks]);
    out
}

/// The reference answer: the same jobs, in the same pick order, through a fresh pool
/// on the calling thread's side, as bits (the pool's contract is exact bits).
fn reference(
    locator: &ExpertLocator,
    region: &Arc<dyn MappedRegion>,
    layer: usize,
    ids: &[i32],
    weights: &[f32],
    row: &[f32],
) -> Vec<u32> {
    let jobs: Vec<ExpertJob> = ids
        .iter()
        .zip(weights)
        .map(|(&id, &weight)| {
            locator
                .job(Arc::clone(region), layer, id as usize, weight)
                .unwrap()
        })
        .collect();
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let sum = pool.run_layer(jobs, row, LONG).unwrap().unwrap();
    sum.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn server_answers_in_pick_order_with_the_pool_sum() {
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0001);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row.clone());
    let server = LayerServer::spawn(port, locate(&bytes), region.clone(), pool, LONG).unwrap();
    // Token 1: three misses in pick order [5, 1, 6]; the answer is the bits of the
    // same jobs in that order through a fresh pool.
    let (a, wa) = ([5i32, 1, 6], [0.5f32, 0.25, -0.125]);
    let want_a = reference(&locate(&bytes), &region, 1, &a, &wa, &row);
    server.arm();
    inner.publish(request(1, 1, &a, &wa));
    let token = server.collect(1);
    assert_eq!((token.served, token.first_error.is_none()), (1, true));
    assert_eq!(inner.answers(), vec![(1, Ok(want_a.clone()))]);
    // Token 2: the same experts in ANOTHER pick order answer the bits of THAT order —
    // a server summing in expert-id order would answer identically for both.
    let (b, wb) = ([1i32, 6, 5], [0.25f32, -0.125, 0.5]);
    let want_b = reference(&locate(&bytes), &region, 1, &b, &wb, &row);
    assert_ne!(want_a, want_b, "the two pick orders must differ in bits");
    server.arm();
    inner.publish(request(2, 1, &b, &wb));
    let token = server.collect(1);
    assert_eq!((token.served, token.first_error.is_none()), (1, true));
    assert_eq!(inner.answers(), vec![(1, Ok(want_a)), (2, Ok(want_b))]);
    server.shutdown().unwrap();
}

#[test]
fn server_turns_a_pool_timeout_into_an_error_answer_and_a_typed_error() {
    // A zero timeout always expires (the deadline is checked first), the same way the
    // host tests provoke PoolError::Timeout.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0002);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, Duration::ZERO).unwrap();
    server.arm();
    inner.publish(request(1, 2, &[3, 4], &[0.5, 0.5]));
    let token = server.collect(1);
    assert_eq!(token.served, 1);
    assert_eq!(inner.answers(), vec![(1, Err(ERR_POOL))]);
    assert!(matches!(
        token.first_error,
        Some(ServeError::Pool(PoolError::Timeout { .. }))
    ));
    server.shutdown().unwrap();
}

#[test]
fn server_ignores_a_stale_sequence_and_answers_each_request_once() {
    // The mock pops 1, then 1 again, then an older 0, then 2: exactly the requests 1
    // and 2 are answered, each once, and served counts two.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0003);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row.clone());
    let server = LayerServer::spawn(port, locate(&bytes), region.clone(), pool, LONG).unwrap();
    let want = reference(&locate(&bytes), &region, 0, &[2, 5], &[0.5, 0.25], &row);
    server.arm();
    for seq in [1u64, 1, 0, 2] {
        inner.publish(request(seq, 0, &[2, 5], &[0.5, 0.25]));
    }
    let token = server.collect(2);
    assert_eq!(token.served, 2);
    assert_eq!(inner.answers(), vec![(1, Ok(want.clone())), (2, Ok(want))]);
    assert!(token.first_error.is_none());
    assert!(token.busy > Duration::ZERO, "two served requests took time");
    assert!(inner.polls.load(Ordering::Relaxed) >= 4);
    server.shutdown().unwrap();
}

#[test]
fn a_refused_answer_is_recorded_as_an_answer_error_and_never_retried() {
    // The port refuses the answer (as the real one does for a bad length or a zero code): the request is consumed
    // (served counts it, `after` moves on), the first error is `Answer`, and the same sequence is not answered again.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0007);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    inner.refuse_answer.store(true, Ordering::Relaxed);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    server.arm();
    inner.publish(request(1, 0, &[2], &[0.5]));
    let token = server.collect(1);
    assert_eq!(token.served, 1);
    assert_eq!(token.first_error, Some(ServeError::Answer));
    assert_eq!(inner.answers().len(), 1, "a refused answer is not retried");
    server.shutdown().unwrap();
}

#[test]
fn dropping_the_handle_stops_and_joins_the_thread() {
    // No shutdown: the Drop sets the stop flag, wakes the thread and joins it, so by the time the drop returns the thread has
    // dropped its port (the mock's flag) - the server is gone before the caller drops the mailbox next.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0008);
    let pool = CpuPool::new(1, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    let timeout = Duration::from_millis(200);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, timeout).unwrap();
    server.arm();
    drop(server);
    assert!(
        inner.dropped.load(Ordering::Acquire),
        "the thread was still running after the drop"
    );
}

/// Lets a port blocked in `poll` go after `after`: the safety net that turns a missing
/// bound (a join that would wait for the stuck thread) into a failed assert instead of a hang.
fn release_poll_after(inner: &Arc<MockInner>, after: Duration) -> std::thread::JoinHandle<()> {
    let inner = Arc::clone(inner);
    std::thread::spawn(move || {
        std::thread::sleep(after);
        inner.block_poll.store(false, Ordering::Relaxed);
    })
}

#[test]
fn dropping_the_handle_waits_for_a_thread_that_is_slow_to_leave() {
    // poll holds the thread for 300 ms after the stop flag is set: the drop must wait that out (its bound is 61 s), so the
    // port is gone when it returns.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_000F);
    let pool = CpuPool::new(1, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    inner.block_poll.store(true, Ordering::Relaxed);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    server.arm();
    std::thread::sleep(Duration::from_millis(30));
    let release = release_poll_after(&inner, Duration::from_millis(300));
    drop(server);
    assert!(
        inner.dropped.load(Ordering::Acquire),
        "the drop returned while the thread was still inside poll"
    );
    release.join().unwrap();
}

#[test]
fn dropping_a_handle_whose_thread_is_stuck_returns_within_its_bound() {
    // poll never returns until 3 s: the drop gives up after timeout + 1 s (about 1.1 s here) and leaves the thread detached.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0010);
    let pool = CpuPool::new(1, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    inner.block_poll.store(true, Ordering::Relaxed);
    let timeout = Duration::from_millis(100);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, timeout).unwrap();
    server.arm();
    std::thread::sleep(Duration::from_millis(30));
    let release = release_poll_after(&inner, Duration::from_secs(3));
    let began = Instant::now();
    drop(server);
    let waited = began.elapsed();
    assert!(
        waited >= timeout && waited < Duration::from_millis(2500),
        "the drop waited {waited:?}, expected about 1.1 s"
    );
    release.join().unwrap();
}

#[test]
fn a_dead_server_thread_is_seen_by_collect_at_once_and_stays_dead() {
    // A panic in a PORT call is outside the caught request body: the thread dies. collect must say Dead without waiting out its
    // bound (timeout 60 s here, so a wait would show), on this token and on every later one.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0009);
    let pool = CpuPool::new(1, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    inner.panic_on_poll.store(true, Ordering::Relaxed);
    server.arm();
    let began = Instant::now();
    let first = server.collect(1);
    assert_eq!(first.first_error, Some(ServeError::Dead));
    assert!(
        began.elapsed() < Duration::from_secs(5),
        "collect waited {:?}",
        began.elapsed()
    );
    server.arm();
    let again = server.collect(1);
    assert_eq!(
        again.first_error,
        Some(ServeError::Dead),
        "a dead server keeps refusing"
    );
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "the second collect waited too"
    );
    assert_eq!(server.shutdown(), Err(ServeError::Dead));
}

#[test]
fn a_panic_while_serving_is_caught_answered_dead_and_the_next_request_is_served() {
    // read_row panics once, inside the caught request body: that request is answered ERR_DEAD and recorded as Dead, the thread
    // lives on and serves the next request normally.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_000A);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row.clone());
    let server = LayerServer::spawn(port, locate(&bytes), region.clone(), pool, LONG).unwrap();
    let want = reference(&locate(&bytes), &region, 0, &[2], &[0.5], &row);
    inner.panic_on_row.store(true, Ordering::Relaxed);
    server.arm();
    inner.publish(request(1, 0, &[2], &[0.5]));
    inner.publish(request(2, 0, &[2], &[0.5]));
    let token = server.collect(2);
    assert_eq!(token.served, 2);
    assert_eq!(token.first_error, Some(ServeError::Dead));
    assert_eq!(inner.answers(), vec![(1, Err(ERR_DEAD)), (2, Ok(want))]);
    server.shutdown().unwrap();
}

#[test]
fn collect_disarms_the_thread() {
    // After collect the thread neither polls nor answers: a request published then stays unanswered until the next arm.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_000B);
    let pool = CpuPool::new(1, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    server.arm();
    assert_eq!(server.collect(0).served, 0);
    std::thread::sleep(Duration::from_millis(20));
    let polls = inner.polls.load(Ordering::Relaxed);
    inner.publish(request(1, 0, &[2], &[0.5]));
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(
        inner.polls.load(Ordering::Relaxed),
        polls,
        "a disarmed thread polled"
    );
    assert!(inner.answers().is_empty(), "a disarmed thread answered");
    server.arm();
    assert_eq!(server.collect(1).served, 1, "the next arm serves it");
    server.shutdown().unwrap();
}

#[test]
fn spawn_refuses_a_port_or_locator_that_does_not_fit_the_pool() {
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_000C);
    let narrow = ExpertShape {
        hidden: SHAPE.hidden / 2,
        inter: SHAPE.inter,
    };
    let pool = CpuPool::new(1, narrow).unwrap();
    let (port, _) = MockPort::new(row.clone());
    let refused = LayerServer::spawn(port, locate(&bytes), region.clone(), pool, LONG).err();
    assert!(
        matches!(refused, Some(ServeError::BadRequest(ref why)) if why.contains("hidden")),
        "{refused:?}"
    );
    // The port and the pool agree on the width, the locator was built for another shape.
    let other = ExpertShape {
        hidden: SHAPE.hidden,
        inter: SHAPE.inter / 2,
    };
    let pool = CpuPool::new(1, other).unwrap();
    let (port, _) = MockPort::new(row);
    let refused = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).err();
    assert!(
        matches!(refused, Some(ServeError::BadRequest(ref why)) if why.contains("shape")),
        "{refused:?}"
    );
}

#[test]
fn arm_zeroes_the_busy_time_and_the_busy_time_accumulates_over_a_token() {
    // The port delays each row read by 30 ms, so a request is busy for at least 30 ms: two requests of one token add up to at
    // least 60 ms (a busy time that is overwritten per request would show one), and the next arm starts again from zero.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_000D);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    inner.row_delay_ms.store(30, Ordering::Relaxed);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    server.arm();
    inner.publish(request(1, 0, &[2], &[0.5]));
    inner.publish(request(2, 0, &[3], &[0.5]));
    let token = server.collect(2);
    assert!(
        token.busy >= Duration::from_millis(60),
        "busy {:?}",
        token.busy
    );
    server.arm();
    let empty = server.collect(0);
    assert_eq!(
        (empty.served, empty.busy),
        (0, Duration::ZERO),
        "arm zeroes the counters"
    );
    server.shutdown().unwrap();
}

#[test]
fn shutdown_gives_up_on_a_thread_stuck_in_a_port_call_within_its_bound() {
    // poll blocks (until a 3 s safety net lets it go), so the thread cannot see the stop flag: shutdown returns Dead after its
    // bound (timeout + 1 s) instead of waiting for it.
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_000E);
    let pool = CpuPool::new(1, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    inner.block_poll.store(true, Ordering::Relaxed);
    let timeout = Duration::from_millis(100);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, timeout).unwrap();
    server.arm();
    std::thread::sleep(Duration::from_millis(30));
    let release = release_poll_after(&inner, Duration::from_secs(3));
    let began = Instant::now();
    assert_eq!(server.shutdown(), Err(ServeError::Dead));
    let waited = began.elapsed();
    assert!(
        waited >= timeout && waited < Duration::from_millis(2500),
        "shutdown waited {waited:?}, expected about 1.1 s"
    );
    release.join().unwrap();
}

#[test]
fn a_worker_panic_answers_worker_lost_and_the_next_request_is_served() {
    // Request 1 (one pick) calls bytes() once in start_layer's validation and once in
    // the worker, so call 4 is request 2's WORKER call: the region panics inside the
    // worker thread (its panic is printed by the default hook — that noise is this
    // test's doing), the worker dies and the pool reports WorkerLost; request 3 is
    // served normally by the surviving worker.
    let (bytes, _, row) = fixture(0x5EED_C0C5_0000_0004);
    let calls = Arc::new(AtomicUsize::new(0));
    let region: Arc<dyn MappedRegion> = Arc::new(CountingRegion {
        bytes: Arc::new(bytes.clone()),
        calls: Arc::clone(&calls),
        panic_at: 4,
    });
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row.clone());
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    server.arm();
    for seq in 1..=3u64 {
        inner.publish(request(seq, 0, &[3], &[0.5]));
    }
    let token = server.collect(3);
    assert_eq!(token.served, 3);
    let plain: Arc<dyn MappedRegion> = Arc::new(bytes.clone());
    let want = reference(&locate(&bytes), &plain, 0, &[3], &[0.5], &row);
    let answers = inner.answers();
    assert_eq!(answers[0], (1, Ok(want.clone())));
    assert_eq!(answers[1], (2, Err(ERR_POOL)));
    assert_eq!(answers[2], (3, Ok(want)));
    assert!(matches!(
        token.first_error,
        Some(ServeError::Pool(PoolError::WorkerLost))
    ));
    server.shutdown().unwrap();
}

#[test]
fn server_shutdown_joins_within_the_timeout_while_busy() {
    // A one-thread pool over a wide shape keeps the request in `finish` for a real
    // stretch, and the port's row delay flags that the thread is inside the request;
    // shutdown is called then and must join within timeout + 1 s plus slack, leaving
    // no thread behind (the port it owns is dropped).
    const WIDE: ExpertShape = ExpertShape {
        hidden: 2048,
        inter: 2048,
    };
    let mut rng = Lcg(0x5EED_C0C5_0000_0005);
    let ((bytes, _), _) = expert_file(&mut rng, 1, 8, WIDE);
    let header = parse(&bytes);
    let locator = ExpertLocator::new(&header, &ExpertTensorNames::llama_cpp(), 1, 8, WIDE).unwrap();
    let row: Vec<f32> = (0..WIDE.hidden)
        .map(|_| rng.f32_between(-1.0, 1.0))
        .collect();
    let region: Arc<dyn MappedRegion> = Arc::new(bytes);
    let pool = CpuPool::new(1, WIDE).unwrap();
    let (port, inner) = MockPort::new(row);
    inner.row_delay_ms.store(40, Ordering::Relaxed);
    // Five seconds: the compute of this request is about 0.8 s in a debug build, which an 800 ms timeout would cut.
    let timeout = Duration::from_secs(5);
    let server = LayerServer::spawn(port, locator, region, pool, timeout).unwrap();
    server.arm();
    inner.publish(request(1, 0, &[0, 1, 2, 3, 4, 5, 6, 7], &[0.125; 8]));
    let gate = Instant::now();
    while !inner.row_entered.load(Ordering::Relaxed) {
        assert!(
            gate.elapsed() < Duration::from_secs(2),
            "the server never entered the request"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let started = Instant::now();
    server.shutdown().unwrap();
    let elapsed = started.elapsed();
    let bound = timeout + Duration::from_secs(1) + Duration::from_secs(2);
    assert!(
        elapsed <= bound,
        "shutdown took {elapsed:?}, bound {bound:?}"
    );
    assert!(
        inner.dropped.load(Ordering::Relaxed),
        "the thread dropped its port on the way out"
    );
}

#[test]
fn server_answers_a_non_finite_partial_with_an_error_code() {
    // HOW the non-finite partial is provoked: a NaN WEIGHT is refused by validation,
    // so the overflow comes from the router weight — f32::MAX times the fixture's
    // O(1)-magnitude expert outputs is Inf well before the sum ends (the reference is
    // asserted non-finite, so the fixture can never silently stop provoking it).
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0006);
    let (ids, weights) = ([1i32, 4, 6], [f32::MAX, f32::MAX, f32::MAX]);
    let check = reference(&locate(&bytes), &region, 1, &ids, &weights, &row);
    assert!(
        check.iter().any(|bits| !f32::from_bits(*bits).is_finite()),
        "the fixture must overflow the sum"
    );
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    server.arm();
    inner.publish(request(1, 1, &ids, &weights));
    let token = server.collect(1);
    assert_eq!(token.served, 1);
    assert_eq!(inner.answers(), vec![(1, Err(ERR_NON_FINITE))]);
    assert_eq!(token.first_error, Some(ServeError::NonFinitePartial));
    server.shutdown().unwrap();
}

#[test]
fn server_refuses_a_bad_request_with_an_error_answer() {
    // Each refusal is its own token, so its own message is observable; the counting
    // region proves no refusal started a pool job; the final token (two refusals and
    // one good request) proves the FIRST error is kept, every refusal is still
    // answered, and the server keeps serving afterwards.
    let (bytes, _, row) = fixture(0x5EED_C0C5_0000_0007);
    let calls = Arc::new(AtomicUsize::new(0));
    let region: Arc<dyn MappedRegion> = Arc::new(CountingRegion {
        bytes: Arc::new(bytes.clone()),
        calls: Arc::clone(&calls),
        panic_at: 0,
    });
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row.clone());
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, LONG).unwrap();
    let cases = [
        ("count 0", 1, Vec::new(), Vec::new(), false),
        (
            "count 9",
            1,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 0],
            vec![0.5; 9],
            false,
        ),
        (
            "id past the layer",
            1,
            vec![EXPERTS as i32, 1],
            vec![0.5, 0.5],
            false,
        ),
        ("repeated id", 1, vec![2, 2], vec![0.5, 0.5], false),
        ("nan weight", 1, vec![2, 3], vec![0.5, f32::NAN], false),
        (
            "layer past the model",
            LAYERS as u32,
            vec![2],
            vec![0.5],
            false,
        ),
        ("refused row", 1, vec![2, 4], vec![0.5, 0.25], true),
    ];
    let mut messages: Vec<String> = Vec::new();
    for (index, (name, layer, ids, weights, refuse_row)) in cases.iter().enumerate() {
        inner.refuse_row.store(*refuse_row, Ordering::Relaxed);
        server.arm();
        inner.publish(request(index as u64 + 1, *layer, ids, weights));
        let token = server.collect(1);
        assert_eq!(token.served, 1, "{name}");
        let answers = inner.answers();
        let (seq, code) = answers.last().unwrap();
        assert_eq!(*seq, index as u64 + 1, "{name}");
        let want = if *refuse_row {
            ERR_ROW
        } else {
            ERR_BAD_REQUEST
        };
        assert_eq!(*code, Err(want), "{name}");
        if *refuse_row {
            assert_eq!(token.first_error, Some(ServeError::Row), "{name}");
        } else {
            match token.first_error {
                Some(ServeError::BadRequest(message)) => messages.push(message),
                other => panic!("{name}: expected BadRequest, got {other:?}"),
            }
        }
    }
    inner.refuse_row.store(false, Ordering::Relaxed);
    assert_eq!(calls.load(Ordering::Relaxed), 0, "no refusal started a job");
    // Every distinct check has its own message; the two count cases share theirs.
    let mut distinct = messages.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 5, "{messages:?}");
    // The final token: count 0, then a repeated id, then a good request.
    server.arm();
    inner.publish(request(8, 1, &[], &[]));
    inner.publish(request(9, 1, &[2, 2], &[0.5, 0.5]));
    inner.publish(request(10, 1, &[2, 4], &[0.5, 0.25]));
    let token = server.collect(3);
    assert_eq!(token.served, 3);
    let plain: Arc<dyn MappedRegion> = Arc::new(bytes.clone());
    let want = reference(&locate(&bytes), &plain, 1, &[2, 4], &[0.5, 0.25], &row);
    let answers = inner.answers();
    assert_eq!(answers[7], (8, Err(ERR_BAD_REQUEST)));
    assert_eq!(answers[8], (9, Err(ERR_BAD_REQUEST)));
    assert_eq!(answers[9], (10, Ok(want)));
    match token.first_error {
        Some(ServeError::BadRequest(ref message)) => {
            assert_eq!(message, &messages[0], "the FIRST error is kept")
        }
        other => panic!("expected the first BadRequest, got {other:?}"),
    }
    // Only the good request reached the pool: two jobs, validated and computed.
    assert_eq!(calls.load(Ordering::Relaxed), 4);
    server.shutdown().unwrap();
}

#[test]
fn an_unarmed_server_serves_nothing_and_collect_reports_the_shortfall() {
    let timeout = Duration::from_millis(50); // short: the shortfall wait stays ~1 s
    let (bytes, region, row) = fixture(0x5EED_C0C5_0000_0008);
    let pool = CpuPool::new(2, SHAPE).unwrap();
    let (port, inner) = MockPort::new(row);
    let server = LayerServer::spawn(port, locate(&bytes), region, pool, timeout).unwrap();
    // Published but unarmed: the thread neither polls nor answers.
    inner.publish(request(1, 1, &[2, 4], &[0.5, 0.25]));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(inner.polls.load(Ordering::Relaxed), 0);
    assert!(inner.answers().is_empty());
    // Armed: the request is answered; collect(2) then reports the bounded shortfall.
    server.arm();
    let started = Instant::now();
    let token = server.collect(2);
    let elapsed = started.elapsed();
    assert_eq!(token.served, 1);
    assert_eq!(inner.answers().len(), 1);
    assert_eq!(
        token.first_error,
        Some(ServeError::Missing {
            expected: 2,
            served: 1
        })
    );
    let bound = timeout + Duration::from_secs(1) + Duration::from_millis(500);
    assert!(
        elapsed >= timeout && elapsed <= bound,
        "collect waited {elapsed:?}, bound {bound:?}"
    );
    server.shutdown().unwrap();
}

#[test]
fn error_codes_are_nonzero_and_distinct() {
    let errors = [
        ServeError::BadRequest(String::from("x")),
        ServeError::Row,
        ServeError::Locator(LocatorError::Overflow),
        ServeError::Pool(PoolError::WorkerLost),
        ServeError::NonFinitePartial,
        ServeError::Answer,
        ServeError::Missing {
            expected: 2,
            served: 1,
        },
        ServeError::Dead,
    ];
    let mut codes: Vec<u32> = errors.iter().map(ServeError::code).collect();
    assert!(codes.iter().all(|code| *code != 0));
    codes.sort();
    codes.dedup();
    assert_eq!(codes.len(), errors.len());
    // The documented consts are exactly the variants' codes.
    assert_eq!(errors[0].code(), ERR_BAD_REQUEST);
    assert_eq!(errors[1].code(), ERR_ROW);
    assert_eq!(errors[2].code(), ERR_LOCATOR);
    assert_eq!(errors[3].code(), ERR_POOL);
    assert_eq!(errors[4].code(), ERR_NON_FINITE);
    assert_eq!(errors[5].code(), ERR_ANSWER);
    assert_eq!(errors[6].code(), ERR_MISSING);
    assert_eq!(errors[7].code(), ERR_DEAD);
}

#[test]
fn server_names_no_cuda_type() {
    // A compile-level check by construction (like the host's): the needles are
    // assembled at run time, so their text cannot appear in the production half
    // except through this test, and any other mention of them fails here.
    let source = include_str!("../server.rs");
    let production = source.split("mod tests").next().unwrap_or_default();
    for needle in [format!("kernels::{}", "cuda"), format!("{}Context", "Cuda")] {
        assert!(
            !production.contains(&needle),
            "server.rs must not contain {needle}"
        );
    }
}
