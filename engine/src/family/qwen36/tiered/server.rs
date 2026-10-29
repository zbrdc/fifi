//! The CPU-expert server of the tiered mailbox, staged for the CUDA adapter: the host thread that
//! answers the device's per-layer expert requests while the device waits inside its graph, by
//! polling the mailbox through the [`RequestPort`] trait (the only bridge to the device; the CUDA
//! adapter implements it) and serving the missed experts on the CPU pool in pick order, answering
//! with the pool's job-order sum bit for bit.
//! Threads and the clock live here because bounded waits and busy time are the point: every condvar wait carries a
//! timeout, every loop a stop flag, and a caught panic never poisons a lock, so nothing here can hang.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, Instant};

use super::cpu_pool::{CpuPool, MappedRegion, PoolError};
use super::locator::{ExpertLocator, LocatorError};

/// One request the device published (mirror of the mailbox header).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ServerRequest {
    /// Sequence the device released; one per published request, from 1.
    pub seq: u64,
    /// The layer that asked, `0..layers`.
    pub layer: u32,
    /// Valid entries of `ids` and `weights`; anything outside `1..=8` is refused.
    pub count: u32,
    /// The missed expert ids in pick order; entries past `count` are ignored.
    pub ids: [i32; 8],
    /// The router weights in pick order; entries past `count` are ignored.
    pub weights: [f32; 8],
}

/// The server's side of the mailbox: the CUDA port's mirror as a trait (the adapter that implements
/// it for the device is the only bridge). The protocol's ordering that an implementation must keep:
/// `poll` ACQUIRES the sequence word, so the header it returns and the row are those the device
/// wrote before releasing it; `read_row` is valid only for the request just polled (the device
/// writes the next one only after it has seen the answer); `answer` writes the partial and the code
/// first and RELEASES `done = seq` last, so a device that acquires `done` sees both. No call may
/// block past `collect`'s bound (`timeout + 1 s`): a request still inside a port call when the next
/// token is armed would count toward that token.
pub trait RequestPort: Send + 'static {
    /// Width of the normed row in floats; must equal the pool's `shape().hidden`.
    fn hidden(&self) -> usize;
    /// The request with a sequence above `after`, or `None` until the device publishes one.
    fn poll(&self, after: u64) -> Option<ServerRequest>;
    /// Copies the normed row of the polled request into `out`; `false` if it cannot.
    fn read_row(&self, out: &mut [f32]) -> bool;
    /// Answers `seq` with the partial (`hidden` floats) or an error code; `false` when refused.
    fn answer(&self, seq: u64, result: Result<&[f32], u32>) -> bool;
}

/// Answer code of [`ServeError::BadRequest`]: the request failed validation.
pub const ERR_BAD_REQUEST: u32 = 1;
/// Answer code of [`ServeError::Row`]: the port refused the normed row.
pub const ERR_ROW: u32 = 2;
/// Answer code of [`ServeError::Locator`]: the locator could not name an expert's bytes.
pub const ERR_LOCATOR: u32 = 3;
/// Answer code of [`ServeError::Pool`]: the pool refused or failed the layer.
pub const ERR_POOL: u32 = 4;
/// Answer code of [`ServeError::NonFinitePartial`]: the partial was not all finite.
pub const ERR_NON_FINITE: u32 = 5;
/// Answer code of [`ServeError::Answer`]: the port could not take the answer.
pub const ERR_ANSWER: u32 = 6;
/// Code of [`ServeError::Missing`]: a collect deadline passed short (caller-side only).
pub const ERR_MISSING: u32 = 7;
/// Answer code of [`ServeError::Dead`]: the request was lost to a caught panic.
pub const ERR_DEAD: u32 = 8;

/// Why a request was answered with an error (the first of a token is kept).
#[derive(Debug, PartialEq)]
pub enum ServeError {
    /// Broken request: count, layer, an id, a repeated id or a weight failed validation.
    BadRequest(String),
    /// The port refused the normed row; nothing was started.
    Row,
    /// The locator refused a `(layer, expert)` after validation (defensive only).
    Locator(LocatorError),
    /// The pool refused the jobs or failed the run (input, timeout, worker lost).
    Pool(PoolError),
    /// The job-order partial held a non-finite value; the device must never add NaN.
    NonFinitePartial,
    /// The port refused the answer; the request was consumed but not delivered.
    Answer,
    /// `collect`'s bounded wait ended with `served` of `expected` requests answered.
    Missing { expected: u64, served: u64 },
    /// A panic was caught serving the request, or the thread ended abnormally.
    Dead,
}

impl ServeError {
    /// The code answered to the device: nonzero, stable, one distinct value per variant.
    pub fn code(&self) -> u32 {
        match self {
            Self::BadRequest(_) => ERR_BAD_REQUEST,
            Self::Row => ERR_ROW,
            Self::Locator(_) => ERR_LOCATOR,
            Self::Pool(_) => ERR_POOL,
            Self::NonFinitePartial => ERR_NON_FINITE,
            Self::Answer => ERR_ANSWER,
            Self::Missing { .. } => ERR_MISSING,
            Self::Dead => ERR_DEAD,
        }
    }
}

/// One token's outcome: requests answered, first error kept, summed serving time.
pub struct TokenServe {
    /// Requests answered this token, successes and errors alike.
    pub served: u64,
    /// The first error of the token, or `None` when every answer was a success.
    pub first_error: Option<ServeError>,
    /// Summed serving time of the token's requests (polling excluded).
    pub busy: Duration,
}

/// How long the armed poll loop spins before yielding; a later measurement tunes it.
const POLL_SPIN: Duration = Duration::from_micros(50);
/// Timeout of every condvar wait while disarmed, and `shutdown`'s join pacing sleep.
const IDLE_WAIT: Duration = Duration::from_millis(1);
/// Slack of the caller-side bounds (`collect`, `shutdown`): no wait is unbounded.
const WAIT_SLACK: Duration = Duration::from_secs(1);

/// What the thread serves with, bundled to travel into the thread.
struct Rig<P: RequestPort> {
    port: P,
    locator: ExpertLocator,
    region: Arc<dyn MappedRegion>,
    pool: CpuPool,
    timeout: Duration,
}

/// The armed/stop flags and token bookkeeping, shared under one lock.
#[derive(Default)]
struct Token {
    armed: bool,
    stop: bool,
    /// The thread has ended (stopped, or died of a panic outside the caught request body).
    dead: bool,
    served: u64,
    first_error: Option<ServeError>,
    busy: Duration,
}

impl<P: RequestPort> Rig<P> {
    /// The partial of one request: validation first, jobs in pick order, the pool's job-order sum bit for bit; `Err`
    /// carries the code the caller answers with.
    fn partial(&self, request: ServerRequest) -> Result<Vec<f32>, ServeError> {
        let bad = |why: &str| Err(ServeError::BadRequest(String::from(why)));
        let locator = &self.locator;
        let count = request.count as usize;
        if request.count == 0 || count > request.ids.len() {
            return bad("count outside 1..=8");
        }
        let layer = usize::try_from(request.layer).unwrap_or(usize::MAX);
        if layer >= locator.layers() {
            return bad("layer outside the model");
        }
        let mut jobs = Vec::with_capacity(count);
        for (i, &id) in request.ids[..count].iter().enumerate() {
            if id < 0 || id as usize >= locator.experts() {
                return bad("expert id outside the layer");
            }
            if request.ids[..i].contains(&id) {
                return bad("expert id picked twice");
            }
            if !request.weights[i].is_finite() {
                return bad("expert weight not finite");
            }
            let region = Arc::clone(&self.region);
            let job = locator.job(region, layer, id as usize, request.weights[i]);
            jobs.push(job.map_err(ServeError::Locator)?);
        }
        let mut row = vec![0.0f32; self.port.hidden()];
        if !self.port.read_row(&mut row) {
            return Err(ServeError::Row);
        }
        let pending = self.pool.start_layer(jobs, &row);
        let pending = pending.map_err(ServeError::Pool)?;
        let partial = match pending {
            None => return Err(ServeError::Pool(PoolError::BadInput(String::from("empty")))),
            Some(pending) => pending.finish(self.timeout).map_err(ServeError::Pool)?,
        };
        if partial.iter().any(|value| !value.is_finite()) {
            return Err(ServeError::NonFinitePartial);
        }
        Ok(partial)
    }
}

/// Marks the thread dead when it ends by any road, including a panic outside the caught request body (in a port
/// call): `collect` then sees it at once instead of waiting out its bound.
struct ExitGuard(Arc<(Mutex<Token>, Condvar)>);

impl Drop for ExitGuard {
    fn drop(&mut self) {
        self.0.0.lock().unwrap_or_else(PoisonError::into_inner).dead = true;
        self.0.1.notify_all();
    }
}

/// The server thread: while armed, poll and serve; while disarmed, wait on the condvar; leave when stopped. `after`
/// (the last answered sequence) persists, so a stale sequence is never answered twice; a panic inside one request is
/// caught, answered `ERR_DEAD`, and the next is served.
fn thread_main<P: RequestPort>(rig: Rig<P>, shared: Arc<(Mutex<Token>, Condvar)>) {
    let _exit = ExitGuard(Arc::clone(&shared));
    let (mutex, signal) = (&shared.0, &shared.1);
    let (mut after, mut spin_until) = (0u64, Instant::now() + POLL_SPIN);
    loop {
        // Disarmed: neither poll nor answer, just wait out the 1 ms slices.
        let mut token = mutex.lock().unwrap_or_else(PoisonError::into_inner);
        while !token.armed && !token.stop {
            let waited = signal.wait_timeout(token, IDLE_WAIT);
            token = waited.unwrap_or_else(PoisonError::into_inner).0;
        }
        if token.stop {
            return;
        }
        drop(token);
        let Some(request) = rig.port.poll(after).filter(|request| request.seq > after) else {
            if Instant::now() >= spin_until {
                std::thread::yield_now();
                spin_until = Instant::now() + POLL_SPIN;
            } else {
                std::hint::spin_loop();
            }
            continue;
        };
        let started = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let port = &rig.port;
            let answered = match rig.partial(request) {
                Ok(partial) => port.answer(request.seq, Ok(&partial)).then_some(None),
                Err(error) => port
                    .answer(request.seq, Err(error.code()))
                    .then_some(Some(error)),
            };
            answered.unwrap_or(Some(ServeError::Answer))
        }));
        let error = outcome.unwrap_or_else(|_panic| {
            let _ = rig.port.answer(request.seq, Err(ERR_DEAD));
            Some(ServeError::Dead)
        });
        let mut token = mutex.lock().unwrap_or_else(PoisonError::into_inner);
        token.busy += started.elapsed();
        token.served += 1;
        after = request.seq;
        token.first_error = token.first_error.take().or(error);
        signal.notify_all();
    }
}

/// The CPU-expert server: one thread over a port, a locator, a region and a pool.
pub struct LayerServer {
    shared: Arc<(Mutex<Token>, Condvar)>,
    handle: Option<JoinHandle<()>>,
    timeout: Duration,
}

impl LayerServer {
    /// Starts the thread; `Err(BadRequest)` when the port's `hidden` disagrees with the pool's, `Err(Dead)` on no thread.
    pub fn spawn(
        port: impl RequestPort,
        locator: ExpertLocator,
        region: Arc<dyn MappedRegion>,
        pool: CpuPool,
        timeout: Duration,
    ) -> Result<LayerServer, ServeError> {
        if port.hidden() != pool.shape().hidden {
            return Err(ServeError::BadRequest("port/pool hidden mismatch".into()));
        }
        if locator.shape() != pool.shape() {
            return Err(ServeError::BadRequest("locator/pool shape mismatch".into()));
        }
        let shared = Arc::new((Mutex::new(Token::default()), Condvar::new()));
        let thread_shared = Arc::clone(&shared);
        let rig = Rig {
            port,
            locator,
            region,
            pool,
            timeout,
        };
        let handle = Builder::new()
            .name(String::from("qwen36-tiered-server"))
            .spawn(move || thread_main(rig, thread_shared))
            .map_err(|_| ServeError::Dead)?;
        Ok(LayerServer {
            shared,
            handle: Some(handle),
            timeout,
        })
    }

    /// Starts a token: zeroes the counters and arms the thread; `after` persists. Invariant: no request of the previous
    /// token is still in flight, because every port call returns well within `collect`'s `timeout + 1 s`.
    pub fn arm(&self) {
        let mut token = self.shared.0.lock().unwrap_or_else(PoisonError::into_inner);
        token.armed = true;
        token.served = 0;
        token.busy = Duration::ZERO;
        token.first_error = None;
        self.shared.1.notify_all();
    }

    /// Ends a token: bounded wait for `expected` answers, then disarm and report; a shortfall is `Missing` unless an
    /// error came first, and the next `arm` zeroes the counters.
    pub fn collect(&self, expected: u64) -> TokenServe {
        let deadline = Instant::now() + self.timeout + WAIT_SLACK;
        let mut token = self.shared.0.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let now = Instant::now();
            if token.served >= expected || token.stop || token.dead || now >= deadline {
                break;
            }
            let waited = self.shared.1.wait_timeout(token, deadline - now);
            token = waited.unwrap_or_else(PoisonError::into_inner).0;
        }
        let served = token.served;
        let first_error = if served < expected && token.first_error.is_none() {
            // A dead thread is reported as such, now and on every later token; a live shortfall is `Missing`.
            Some(if token.dead {
                ServeError::Dead
            } else {
                ServeError::Missing { expected, served }
            })
        } else {
            token.first_error.take()
        };
        token.armed = false;
        self.shared.1.notify_all();
        TokenServe {
            served,
            first_error,
            busy: token.busy,
        }
    }

    /// Stops the thread: stop flag, wake, join bounded by `timeout + 1 s`; else `Dead`.
    pub fn shutdown(mut self) -> Result<(), ServeError> {
        let mut token = self.shared.0.lock().unwrap_or_else(PoisonError::into_inner);
        token.stop = true;
        drop(token); // the join below must not hold the lock the thread wakes on
        self.shared.1.notify_all();
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        let deadline = Instant::now() + self.timeout + WAIT_SLACK;
        while !handle.is_finished() && Instant::now() < deadline {
            std::thread::sleep(IDLE_WAIT);
        }
        if !handle.is_finished() {
            return Err(ServeError::Dead);
        }
        match handle.join() {
            Ok(()) => Ok(()),
            Err(_) => Err(ServeError::Dead),
        }
    }
}

/// Dropping a handle without `shutdown` stops the thread and joins it, bounded by `timeout + 1 s` (a thread stuck in
/// a port call is left detached), so the server is gone before whatever the caller drops next: the mailbox it polls.
impl Drop for LayerServer {
    fn drop(&mut self) {
        self.shared
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stop = true;
        self.shared.1.notify_all();
        let Some(handle) = self.handle.take() else {
            return;
        };
        let deadline = Instant::now() + self.timeout + WAIT_SLACK;
        while !handle.is_finished() && Instant::now() < deadline {
            std::thread::sleep(IDLE_WAIT);
        }
        if handle.is_finished() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests;
