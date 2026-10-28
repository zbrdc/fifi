//! The CPU expert pool of the tiered offload, host only (OFF-2 P6).
//!
//! The types staged here describe WHAT the pool consumes: shared mapped regions
//! ([`MappedRegion`]), one routed expert's slices and router weight ([`ExpertJob`]),
//! and the typed failures of a run ([`PoolError`]). The pool itself ([`CpuPool`])
//! makes no CUDA calls and holds no device context (the CUDA context is
//! thread-confined to the engine thread); its workers read only mapped bytes and the
//! activation row they are given, block at most a caller-injected timeout, sum in
//! fixed job order from +0.0, and return `None` for an empty job list (signed zeros
//! survive). Each pool runs one expert kernel, chosen at construction
//! ([`ExpertKernel`]): the safe f32 reference, the Q8_K integer kernel
//! (`expert_cpu_q8k`, llama.cpp's activation grid) or the per-32 group integer kernel
//! (`expert_cpu_q8g`, the kernel of the tier), dotted against the same mapped Q4_K bytes.
//! Nothing about Qwen3.6 is hard-coded: all geometry comes from the kernel's
//! [`ExpertShape`](crate::kernels::expert_cpu::ExpertShape). The tiered step (OFF-2 D5)
//! drives it through `TieredHost`.

use std::collections::VecDeque;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::kernels::expert_cpu::{
    ExpertCpuError, ExpertScratch, ExpertShape, ExpertSlices, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES,
    expert_forward,
};
use crate::kernels::expert_cpu_q8g::{ExpertScratchQ8g, expert_forward_q8g};
use crate::kernels::expert_cpu_q8k::{DotBackend, ExpertScratchQ8k, expert_forward_q8k};

/// Which expert kernel the workers run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertKernel {
    /// The safe f32 reference kernel (what [`CpuPool::new`] uses).
    F32,
    /// Q8_K activations (llama.cpp's CPU grid, 256 values per scale), integer dot; the backend
    /// decides scalar or AVX2. NOT the tier's kernel: its rounding noise is independent of the
    /// oracle's and failed OFF-2 gate (b) at 80 percent CPU picks.
    Q8k(DotBackend),
    /// Activations on the 32-value group grid of the device path, integer dot; the backend
    /// decides scalar or AVX2. The tier's kernel: it passed gate (b) where Q8_K failed.
    Q8g(DotBackend),
}

/// One worker's whole kernel state: the scratch of exactly the kernel its pool
/// runs and nothing else, so a worker never carries buffers it cannot use.
enum Worker {
    /// The f32 reference kernel with its [`ExpertScratch`].
    F32 {
        /// Reusable f32 kernel scratch, sized for the pool shape.
        scratch: ExpertScratch,
    },
    /// The Q8_K integer kernel with its dot backend and its [`ExpertScratchQ8k`].
    Q8k {
        /// The scalar/AVX2 dispatch, fixed with the pool.
        backend: DotBackend,
        /// Reusable Q8_K kernel scratch, sized for the pool shape.
        scratch: ExpertScratchQ8k,
    },
    /// The per-32 group integer kernel with its dot backend and its [`ExpertScratchQ8g`].
    Q8g {
        /// The scalar/AVX2 dispatch, fixed with the pool.
        backend: DotBackend,
        /// Reusable per-32 kernel scratch, sized for the pool shape.
        scratch: ExpertScratchQ8g,
    },
}

impl Worker {
    /// Allocates the scratch of exactly `kernel` for `shape` (only that one).
    fn new(kernel: ExpertKernel, shape: ExpertShape) -> Self {
        match kernel {
            ExpertKernel::F32 => Self::F32 {
                scratch: ExpertScratch::new(shape),
            },
            ExpertKernel::Q8k(backend) => Self::Q8k {
                backend,
                scratch: ExpertScratchQ8k::new(shape),
            },
            ExpertKernel::Q8g(backend) => Self::Q8g {
                backend,
                scratch: ExpertScratchQ8g::new(shape),
            },
        }
    }

    /// Runs one expert through this worker's kernel; the errors are the kernel's
    /// own, so both kernels' refusals surface per run as [`PoolError::Expert`].
    fn compute(
        &mut self,
        shape: ExpertShape,
        slices: &ExpertSlices<'_>,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), ExpertCpuError> {
        match self {
            Self::F32 { scratch } => expert_forward(shape, slices, x, scratch, out),
            Self::Q8k { backend, scratch } => {
                expert_forward_q8k(*backend, shape, slices, x, scratch, out)
            }
            Self::Q8g { backend, scratch } => {
                expert_forward_q8g(*backend, shape, slices, x, scratch, out)
            }
        }
    }
}

/// Any shared, immutable region holding expert weight bytes (an mmap of the GGUF, a
/// `Vec<u8>`, ...): the pool only ever reads `bytes`, so every `AsRef<[u8]>` that
/// can travel between threads is a region.
pub trait MappedRegion: Send + Sync + 'static {
    /// The expert weight bytes, valid for as long as the region is alive.
    fn bytes(&self) -> &[u8];
}

impl<T: AsRef<[u8]> + Send + Sync + 'static> MappedRegion for T {
    fn bytes(&self) -> &[u8] {
        self.as_ref()
    }
}

/// One routed expert of one layer: where its three Q4_K slices live inside `region`
/// (half-open byte offsets) and its router weight. The pool checks the ranges
/// against its shape before any work starts.
pub struct ExpertJob {
    /// Shared mapped region holding this expert's weight bytes.
    pub region: Arc<dyn MappedRegion>,
    /// Gate projection bytes: `inter` rows over `hidden` inputs, Q4_K blocks.
    pub gate: Range<usize>,
    /// Up projection bytes: `inter` rows over `hidden` inputs, Q4_K blocks.
    pub up: Range<usize>,
    /// Down projection bytes: `hidden` rows over `inter` inputs, Q4_K blocks.
    pub down: Range<usize>,
    /// Router weight; the layer sum scales this expert's output by it.
    pub weight: f32,
}

/// Typed failure of a pool run: every wait is bounded, so no call can hang and every
/// failure of a run is one of these variants.
#[derive(Debug, PartialEq)]
pub enum PoolError {
    /// Broken caller contract; the message names the job (for job-level violations)
    /// and what was wrong. Nothing was computed.
    BadInput(String),
    /// [`expert_forward`](crate::kernels::expert_cpu::expert_forward) refused a job;
    /// `job` is the run's lowest failing index, whatever order failures arrived in.
    Expert {
        /// Index of the failing job in the submitted list.
        job: usize,
        /// The kernel's reason.
        error: ExpertCpuError,
    },
    /// `pending` jobs never answered within the timeout; their late results are
    /// discarded, so the run had no outcome.
    Timeout {
        /// Whole milliseconds actually waited before giving up.
        waited_ms: u64,
        /// Jobs of the run that never delivered a result.
        pending: usize,
    },
    /// A worker died on a panic, so its job can never answer; the pool keeps serving
    /// with the surviving workers.
    WorkerLost,
}

/// One job as it travels to a worker: (index, expert, shared row, run sender).
struct TaskItem(usize, ExpertJob, Arc<[f32]>, Sender<JobOutcome>);

enum JobOutcome {
    /// The unscaled `hidden`-long output; the ordered sum applies the weight.
    Done(usize, Vec<f32>),
    /// The kernel refused the job.
    Failed(usize, ExpertCpuError),
    /// A worker panicked while holding a job of this run.
    Lost,
}

/// Queue state under the lock; the zero value is a valid idle queue.
#[derive(Default)]
struct PoolQueue {
    jobs: VecDeque<TaskItem>,
    shutdown: bool,
}

/// Shared pool state; `gate` is `None` in real pools (gated tests hold workers).
struct PoolInner {
    queue: Mutex<PoolQueue>,
    ready: Condvar,
    gate: Option<Arc<(Mutex<bool>, Condvar)>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait_gate(gate: &Arc<(Mutex<bool>, Condvar)>) {
    let (flag, cv) = &**gate;
    let mut open = lock(flag);
    while !*open {
        open = cv.wait(open).unwrap_or_else(PoisonError::into_inner);
    }
}

/// The worker main loop: take a job (waiting when idle), wait on the test gate if
/// there is one, compute, report on the run's channel. A panic is caught, reported
/// as `Lost`, and ends that worker; the survivors keep serving.
fn worker_loop(shared: Arc<PoolInner>, shape: ExpertShape, mut worker: Worker) {
    loop {
        let item = {
            let mut queue = lock(&shared.queue);
            loop {
                if let Some(item) = queue.jobs.pop_front() {
                    break Some(item);
                }
                if queue.shutdown {
                    break None;
                }
                queue = shared
                    .ready
                    .wait(queue)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let Some(item) = item else { return };
        if let Some(gate) = &shared.gate {
            wait_gate(gate);
        }
        let job = item.1;
        let bytes = job.region.bytes();
        let slices = ExpertSlices {
            gate: &bytes[job.gate],
            up: &bytes[job.up],
            down: &bytes[job.down],
        };
        let mut out = vec![0.0f32; shape.hidden];
        match catch_unwind(AssertUnwindSafe(|| {
            worker.compute(shape, &slices, &item.2, &mut out)
        })) {
            Ok(Ok(())) => {
                let _ = item.3.send(JobOutcome::Done(item.0, out));
            }
            Ok(Err(error)) => {
                let _ = item.3.send(JobOutcome::Failed(item.0, error));
            }
            Err(_panic) => {
                let _ = item.3.send(JobOutcome::Lost);
                return;
            }
        }
    }
}

/// The CPU expert pool: persistent host worker threads over one shared job queue,
/// each owning the scratch of the pool's [`ExpertKernel`] (a run allocates no
/// kernel scratch).
pub struct CpuPool {
    threads: usize,
    shape: ExpertShape,
    kernel: ExpertKernel,
    shared: Arc<PoolInner>,
    handles: Vec<JoinHandle<()>>,
}

impl CpuPool {
    /// Creates a pool of `threads` workers for `shape` (`threads >= 1`) running the
    /// safe f32 kernel; a shape the kernel refuses surfaces per run as
    /// [`PoolError::Expert`].
    pub fn new(threads: usize, shape: ExpertShape) -> Result<CpuPool, PoolError> {
        Self::build(threads, shape, ExpertKernel::F32, None)
    }

    /// Creates a pool of `threads` workers for `shape` (`threads >= 1`) running
    /// `kernel`: the f32 reference or the Q8_K integer kernel with its backend.
    /// Each worker owns exactly that kernel's scratch; the run contract is the
    /// `CpuPool::new` one, with the chosen kernel in place of `expert_forward`.
    pub fn with_kernel(
        threads: usize,
        shape: ExpertShape,
        kernel: ExpertKernel,
    ) -> Result<CpuPool, PoolError> {
        Self::build(threads, shape, kernel, None)
    }

    /// The worker count, fixed at construction.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The expert kernel the workers run, fixed at construction.
    pub fn kernel(&self) -> ExpertKernel {
        self.kernel
    }

    /// The expert geometry the workers were built for, fixed at construction.
    pub fn shape(&self) -> ExpertShape {
        self.shape
    }

    /// Builds the pool around a gate (`None` for real pools): spawns `threads`
    /// workers, each with its own scratch for `kernel`; if a spawn fails, the
    /// already-spawned workers are stopped and joined as [`PoolError::WorkerLost`].
    fn build(
        threads: usize,
        shape: ExpertShape,
        kernel: ExpertKernel,
        gate: Option<Arc<(Mutex<bool>, Condvar)>>,
    ) -> Result<CpuPool, PoolError> {
        if threads == 0 {
            return Err(PoolError::BadInput(format!(
                "cpu pool needs at least one thread, got {threads}"
            )));
        }
        let shared = Arc::new(PoolInner {
            queue: Mutex::new(PoolQueue::default()),
            ready: Condvar::new(),
            gate,
        });
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let worker_shared = Arc::clone(&shared);
            let worker = Worker::new(kernel, shape);
            let spawned = std::thread::Builder::new()
                .name(String::from("qwen36-cpu-pool"))
                .spawn(move || worker_loop(worker_shared, shape, worker));
            match spawned {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    lock(&shared.queue).shutdown = true;
                    shared.ready.notify_all();
                    for handle in handles.drain(..) {
                        let _ = handle.join();
                    }
                    return Err(PoolError::WorkerLost);
                }
            }
        }
        Ok(CpuPool {
            threads,
            shape,
            kernel,
            shared,
            handles,
        })
    }

    /// Computes all jobs of ONE layer: the CPU half of its MoE output, or
    /// `Ok(None)` for an empty job list (the caller adds nothing; signed zeros
    /// survive). Every argument is validated before any work — ranges inside the
    /// region and exactly the shape's slice lengths (gate/up `inter*hidden/256*144`
    /// bytes, down `hidden*inter/256*144`), `hidden` exactly `shape.hidden` long,
    /// finite weights; violations are [`PoolError::BadInput`] naming the job, and
    /// nothing is queued before every job passes. The result is the sum over jobs
    /// IN JOB ORDER of `weight_j * <pool kernel>(job_j)` (f32 `expert_forward`, or
    /// `expert_forward_q8k` through the pool's backend), f32 from +0.0, so the
    /// bits never depend on which thread finished first; the wait is bounded by
    /// `timeout`, and late results die with the run's channel (one per run).
    pub fn run_layer(
        &self,
        jobs: Vec<ExpertJob>,
        hidden: &[f32],
        timeout: Duration,
    ) -> Result<Option<Vec<f32>>, PoolError> {
        match self.start_layer(jobs, hidden)? {
            None => Ok(None),
            Some(pending) => pending.finish(timeout).map(Some),
        }
    }

    /// The first half of [`CpuPool::run_layer`]: validates every argument exactly
    /// as `run_layer` does, queues all jobs and returns at once with a
    /// [`PendingLayer`] (`Ok(None)` for an empty job list), so the caller can
    /// enqueue GPU work while the workers run. [`PendingLayer::finish`] joins and
    /// sums. Dropping a pending layer without finishing discards its results; the
    /// queued jobs still run to completion.
    pub fn start_layer(
        &self,
        jobs: Vec<ExpertJob>,
        hidden: &[f32],
    ) -> Result<Option<PendingLayer>, PoolError> {
        if jobs.is_empty() {
            return Ok(None);
        }
        let weights: Vec<f32> = jobs.iter().map(|job| job.weight).collect();
        if hidden.len() != self.shape.hidden {
            return Err(PoolError::BadInput(format!(
                "hidden row is {} long, the shape wants {}",
                hidden.len(),
                self.shape.hidden
            )));
        }
        // Saturating like the kernel's clamp: a nonsense shape cannot fake a length.
        let side =
            self.shape.inter.saturating_mul(self.shape.hidden) / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        let down =
            self.shape.hidden.saturating_mul(self.shape.inter) / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        for (j, job) in jobs.iter().enumerate() {
            let bytes = job.region.bytes();
            let check = |name: &str, range: &Range<usize>, want: usize| -> Result<(), PoolError> {
                let (start, end) = (range.start, range.end);
                if start > end || end > bytes.len() || end - start != want {
                    Err(PoolError::BadInput(format!(
                        "job {j} {name} range {start}..{end} is not a {want}-byte slice \
                        inside the {} region bytes",
                        bytes.len()
                    )))
                } else {
                    Ok(())
                }
            };
            check("gate", &job.gate, side)?;
            check("up", &job.up, side)?;
            check("down", &job.down, down)?;
            if !job.weight.is_finite() {
                return Err(PoolError::BadInput(format!(
                    "job {j} weight {} is not finite",
                    job.weight
                )));
            }
        }

        let (tx, rx) = channel();
        let x: Arc<[f32]> = hidden.iter().copied().collect();
        {
            let mut queue = lock(&self.shared.queue);
            for (index, job) in jobs.into_iter().enumerate() {
                queue
                    .jobs
                    .push_back(TaskItem(index, job, Arc::clone(&x), tx.clone()));
            }
        }
        self.shared.ready.notify_all();
        Ok(Some(PendingLayer {
            rx,
            weights,
            hidden: self.shape.hidden,
        }))
    }
}

/// The queued jobs of one layer: join them with [`PendingLayer::finish`].
pub struct PendingLayer {
    rx: Receiver<JobOutcome>,
    weights: Vec<f32>,
    hidden: usize,
}

impl PendingLayer {
    /// Waits for every job (at most `timeout` from this call) and returns the
    /// job-order sum of `weight_j * output_j`, f32 from +0.0, exactly the value
    /// `run_layer` returns. Errors are those of `run_layer`: the lowest failing
    /// job as [`PoolError::Expert`], [`PoolError::Timeout`] (late results are
    /// discarded with this layer's channel) or [`PoolError::WorkerLost`].
    pub fn finish(self, timeout: Duration) -> Result<Vec<f32>, PoolError> {
        let PendingLayer {
            rx,
            weights,
            hidden,
        } = self;
        let n = weights.len();
        // Collect: every job answers exactly once; failures are kept and the lowest
        // index wins (reported after every answer, so arrival order cannot change
        // it). `Lost` or a dead channel: a worker is gone.
        let mut outs: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
        let mut failure: Option<(usize, ExpertCpuError)> = None;
        let start = Instant::now();
        let deadline = start + timeout;
        let mut received = 0usize;
        while received < n {
            let now = Instant::now();
            if now >= deadline {
                return Err(PoolError::Timeout {
                    waited_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                    pending: n - received,
                });
            }
            match rx.recv_timeout(deadline - now) {
                Ok(JobOutcome::Done(index, out)) => {
                    if let Some(slot) = outs.get_mut(index) {
                        *slot = Some(out);
                        received += 1;
                    }
                }
                Ok(JobOutcome::Failed(index, error)) => {
                    if failure.as_ref().is_none_or(|(i, _)| index < *i) {
                        failure = Some((index, error));
                    }
                    received += 1;
                }
                Ok(JobOutcome::Lost) => return Err(PoolError::WorkerLost),
                Err(RecvTimeoutError::Timeout) => {} // re-check the deadline
                Err(RecvTimeoutError::Disconnected) => return Err(PoolError::WorkerLost),
            }
        }
        if let Some((job, error)) = failure {
            return Err(PoolError::Expert { job, error });
        }

        // Every job answered: accumulate in job order (weight-scaled, from +0.0).
        let mut sum = vec![0.0f32; hidden];
        for (weight, out) in weights.into_iter().zip(outs) {
            let Some(out) = out else {
                // Unreachable (the loop returns unless every job answered); the
                // guard keeps a lost message from silently producing a wrong sum.
                return Err(PoolError::WorkerLost);
            };
            for (s, v) in sum.iter_mut().zip(out) {
                *s += weight * v;
            }
        }
        Ok(sum)
    }
}

impl Drop for CpuPool {
    /// Closes the queue and joins every worker; a join error only means a worker
    /// was killed by an abort, and dropping it keeps `drop` from ever panicking.
    /// Open a gated pool first, or the join blocks forever.
    fn drop(&mut self) {
        lock(&self.shared.queue).shutdown = true;
        self.shared.ready.notify_all();
        self.handles.drain(..).for_each(|h| {
            let _ = h.join();
        });
    }
}

/// Test seam: workers of a gated pool wait on it before each job (not public API).
#[cfg(test)]
struct Gate {
    inner: Arc<(Mutex<bool>, Condvar)>,
}

#[cfg(test)]
impl Gate {
    /// Opens the gate (one-shot): every waiting worker proceeds.
    fn open(&self) {
        let (flag, cv) = &*self.inner;
        *lock(flag) = true;
        cv.notify_all();
    }
}

#[cfg(test)]
impl CpuPool {
    /// Test seam (not public API); open the gate before dropping the pool.
    fn new_gated(threads: usize, shape: ExpertShape) -> (CpuPool, Gate) {
        Self::new_gated_kernel(threads, shape, ExpertKernel::F32)
    }

    /// Test seam (not public API): a gated pool running `kernel`, so the timeout
    /// and late-result behaviour can be pinned for either kernel; open the gate
    /// before dropping the pool.
    fn new_gated_kernel(
        threads: usize,
        shape: ExpertShape,
        kernel: ExpertKernel,
    ) -> (CpuPool, Gate) {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let pool =
            Self::build(threads, shape, kernel, Some(gate.clone())).expect("gated test pool");
        (pool, Gate { inner: gate })
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuPool, ExpertJob, ExpertKernel, MappedRegion, PoolError};
    use crate::kernels::expert_cpu::{
        ExpertCpuError, ExpertScratch, ExpertShape, ExpertSlices, Q4K_BLOCK_BYTES,
        Q4K_BLOCK_VALUES, expert_forward,
    };
    use crate::kernels::expert_cpu_q8g::{ExpertScratchQ8g, expert_forward_q8g};
    use crate::kernels::expert_cpu_q8k::{DotBackend, ExpertScratchQ8k, expert_forward_q8k};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// The real Qwen3.6 expert shape as a test input (the pool hard-codes nothing).
    const REAL: ExpertShape = ExpertShape {
        hidden: 2048,
        inter: 512,
    };
    /// Generous timeout, reached only if something is broken.
    const LONG: Duration = Duration::from_secs(60);
    /// The gated tests' short budget (a run must give up within ~50 ms).
    const SHORT: Duration = Duration::from_millis(50);

    /// Tiny deterministic LCG (Knuth MMIX constants); tests never touch the OS.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn f32_between(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() as f32 / (1u64 << 53) as f32)
        }
    }

    /// Moderate f16 scale patterns; they keep the expert outputs finite and small.
    const F16_BITS: [u16; 4] = [0x3500, 0x3c00, 0x4000, 0xbc00];

    /// `blocks` random Q4_K blocks with moderate, finite scales.
    fn random_blocks(rng: &mut Lcg, blocks: usize) -> Vec<u8> {
        let mut w = vec![0u8; blocks * Q4K_BLOCK_BYTES];
        for blk in w.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
            blk[4..].iter_mut().for_each(|b| *b = rng.next() as u8);
            let d = F16_BITS[(rng.next() % 4) as usize].to_le_bytes();
            blk[0..2].copy_from_slice(&d);
            blk[2..4].copy_from_slice(&d);
        }
        w
    }

    /// Byte length of a `rows x cols` Q4_K slice: the task's formula.
    fn slice_len(rows: usize, cols: usize) -> usize {
        rows * cols / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES
    }

    /// `n` experts in one region: expert j is [gate | up | down], adjacent in order.
    fn expert_region(rng: &mut Lcg, shape: ExpertShape, n: usize) -> Vec<u8> {
        let side = shape.inter * shape.hidden / Q4K_BLOCK_VALUES;
        let down = shape.hidden * shape.inter / Q4K_BLOCK_VALUES;
        let mut region = Vec::new();
        for _ in 0..n {
            region.extend(random_blocks(rng, side));
            region.extend(random_blocks(rng, side));
            region.extend(random_blocks(rng, down));
        }
        region
    }

    /// The `weights.len()` jobs over `region`, weights attached in expert order.
    fn jobs_over(region: &Arc<Vec<u8>>, shape: ExpertShape, weights: &[f32]) -> Vec<ExpertJob> {
        let region: Arc<dyn MappedRegion> = region.clone();
        let side = slice_len(shape.inter, shape.hidden);
        let stride = 3 * side;
        let mut jobs = Vec::with_capacity(weights.len());
        for (j, &weight) in weights.iter().enumerate() {
            let at = j * stride;
            jobs.push(ExpertJob {
                region: region.clone(),
                gate: at..at + side,
                up: at + side..at + 2 * side,
                down: at + 2 * side..at + stride,
                weight,
            });
        }
        jobs
    }

    /// Serial reference: the pool's contract on the calling thread, visiting the
    /// experts in `order` (position -> expert index), one scratch like a worker.
    fn serial_sum(jobs: &[ExpertJob], order: &[usize], hidden: &[f32]) -> Vec<f32> {
        let mut scratch = ExpertScratch::new(REAL);
        let mut out = vec![0.0; REAL.hidden];
        let mut sum = vec![0.0; REAL.hidden];
        for &e in order {
            let job = &jobs[e];
            let bytes = job.region.bytes();
            let slices = ExpertSlices {
                gate: &bytes[job.gate.clone()],
                up: &bytes[job.up.clone()],
                down: &bytes[job.down.clone()],
            };
            expert_forward(REAL, &slices, hidden, &mut scratch, &mut out).unwrap();
            sum.iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s += job.weight * v);
        }
        sum
    }

    /// Region, activation row and weights for `n` experts of the real shape.
    fn setup(rng: &mut Lcg, n: usize, wlo: f32, whi: f32) -> (Arc<Vec<u8>>, Vec<f32>, Vec<f32>) {
        let region = Arc::new(expert_region(rng, REAL, n));
        let hidden = (0..REAL.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        let weights: Vec<f32> = (0..n).map(|_| rng.f32_between(wlo, whi)).collect();
        (region, hidden, weights)
    }

    /// The bits of a row: the pool's contract is exact bits, not a tolerance.
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn assert_same_bits(got: &[f32], want: &[f32]) {
        assert_eq!(bits(got), bits(want));
    }

    #[test]
    fn a_region_is_any_shared_byte_source() {
        // The blanket impl covers Vec<u8> and Arc<Vec<u8>> (the in-memory stand-in
        // for a mapping); bytes() reads through AsRef without copying.
        let v = vec![7u8, 8, 9];
        assert_eq!(MappedRegion::bytes(&v), &[7u8, 8, 9][..]);
        let arc: Arc<dyn MappedRegion> = Arc::new(vec![7u8, 8, 9]);
        assert_eq!(arc.bytes(), &[7u8, 8, 9][..]);
        // The error type stays comparable, so callers can match on it exactly.
        assert_eq!(
            PoolError::Expert {
                job: 2,
                error: ExpertCpuError::BadShape(String::from("no")),
            },
            PoolError::Expert {
                job: 2,
                error: ExpertCpuError::BadShape(String::from("no")),
            }
        );
        assert_ne!(
            PoolError::WorkerLost,
            PoolError::Timeout {
                waited_ms: 0,
                pending: 0
            }
        );
    }

    #[test]
    fn cpu_pool_returns_none_for_no_jobs() {
        // An empty job list answers None: the caller adds nothing (signed zeros).
        let shape = ExpertShape {
            hidden: 2048,
            inter: 512,
        };
        let pool = CpuPool::new(3, shape).unwrap();
        assert_eq!(
            pool.run_layer(Vec::new(), &[], Duration::from_secs(5)),
            Ok(None)
        );
    }

    #[test]
    fn cpu_pool_takes_no_device_types() {
        // A compile-level check by construction: the needles are assembled at run
        // time, so their text cannot appear in this file except through this test,
        // and any other mention of them fails here.
        let src = include_str!("cpu_pool.rs");
        for needle in [format!("kernels::{}", "cuda"), format!("{}Context", "Cuda")] {
            assert!(
                !src.contains(&needle),
                "cpu_pool.rs must not contain {needle}"
            );
        }
    }

    #[test]
    fn cpu_pool_matches_a_serial_sum_of_expert_forward() {
        // 8 random experts of the real shape; a 4-thread pool equals the serial
        // ordered sum bitwise.
        let (region, hidden, weights) = setup(&mut Lcg(0x1B1E_55AD_EA7D_0001), 8, -0.5, 0.5);
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum(&jobs, &[0, 1, 2, 3, 4, 5, 6, 7], &hidden);
        let pool = CpuPool::new(4, REAL).unwrap();
        let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
        assert_same_bits(&got, &want);
    }

    #[test]
    fn cpu_pool_sum_is_in_job_order_and_independent_of_thread_count() {
        let (region, hidden, weights) = setup(&mut Lcg(0x00C0_FFEE_5EED_0002), 8, -0.6, 0.6);
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum(&jobs, &[0, 1, 2, 3, 4, 5, 6, 7], &hidden);
        // (a) asserted: identical bits for 1, 2, 3 and 8 workers.
        for threads in [1, 2, 3, 8] {
            let pool = CpuPool::new(threads, REAL).unwrap();
            assert_eq!(pool.threads(), threads);
            let jobs = jobs_over(&region, REAL, &weights);
            let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
            assert_same_bits(&got, &want);
        }
        // (b) asserted: the submission order IS the sum order - reversed jobs give
        // the reversed-order reference bitwise, and for this seed the two
        // references differ somewhere (the reversal really re-orders the sum).
        let want_rev = serial_sum(&jobs, &[7, 6, 5, 4, 3, 2, 1, 0], &hidden);
        let mut rev = jobs_over(&region, REAL, &weights);
        rev.reverse();
        let pool = CpuPool::new(8, REAL).unwrap();
        let got_rev = pool.run_layer(rev, &hidden, LONG).unwrap().unwrap();
        assert_same_bits(&got_rev, &want_rev);
        assert!(bits(&want) != bits(&want_rev));
    }

    #[test]
    fn cpu_pool_rejects_bad_input_before_any_work() {
        // Each BadInput case is its own error naming its job; the valid run after
        // them proves the workers were never touched.
        let (region, hidden, weights) = setup(&mut Lcg(0xFEED_F00D_0000_0003), 3, -0.2, 0.2);
        let pool = CpuPool::new(2, REAL).unwrap();
        let cases = [
            ("range outside", hidden.clone()),
            ("wrong slice length", hidden.clone()),
            ("hidden short", hidden[..2047].to_vec()),
            ("hidden long", [hidden.as_slice(), &[0.0f32]].concat()),
            ("nan weight", hidden.clone()),
        ];
        let mut seen: Vec<String> = Vec::new();
        for (i, (name, row)) in cases.into_iter().enumerate() {
            let ws = if i == 4 {
                vec![f32::NAN, 0.3, 0.1]
            } else {
                weights.clone()
            };
            let mut jobs = jobs_over(&region, REAL, &ws);
            match i {
                0 => jobs[1].down.end = region.len() + 8, // job 1 leaves the region
                1 => jobs[2].up.end -= 1,                 // job 2's up one byte short
                _ => {}
            }
            let msg = match pool.run_layer(jobs, &row, LONG) {
                Err(PoolError::BadInput(msg)) => msg,
                other => panic!("{name}: expected BadInput, got {other:?}"),
            };
            assert!(!seen.contains(&msg), "{name}: error not its own: {msg}");
            seen.push(msg);
        }
        assert!(
            seen[0].contains("job 1") && seen[0].contains("down"),
            "{seen:?}"
        );
        assert!(seen[4].contains("job 0"), "{seen:?}");
        let jobs = jobs_over(&region, REAL, &weights);
        let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum(&jobs, &[0, 1, 2], &hidden);
        assert_same_bits(&got, &want);
    }

    #[test]
    fn cpu_pool_join_times_out_with_a_typed_error() {
        // Gate closed: the workers hold their jobs without computing, so a 50 ms
        // timeout fires (~50-250 ms by design; 2 s catches a stuck wait).
        let (region, hidden, weights) = setup(&mut Lcg(0xC10C_5EED_0000_0005), 8, -0.4, 0.4);
        let (pool, gate) = CpuPool::new_gated(4, REAL);
        let jobs = jobs_over(&region, REAL, &weights);
        let start = Instant::now();
        let err = pool.run_layer(jobs, &hidden, SHORT).unwrap_err();
        let wall = start.elapsed();
        assert!(wall >= Duration::from_millis(45) && wall <= Duration::from_secs(2));
        let PoolError::Timeout { waited_ms, pending } = err else {
            panic!("expected Timeout, got {err:?}")
        };
        assert!((50..1500).contains(&waited_ms), "waited {waited_ms} ms");
        assert!(pending > 0 && pending <= 8, "pending {pending}");
        gate.open(); // release the workers so Drop's join cannot block
    }

    #[test]
    fn a_timed_out_run_does_not_corrupt_the_next_run() {
        // Run 1 times out at the gate; opening it lets its late results flow into
        // the dropped channel (discarded). Run 2 must equal the reference bitwise:
        // a leaked late result would double-count an expert.
        let (region, hidden, weights) = setup(&mut Lcg(0xD1CE_C0DE_0000_0006), 8, -0.3, 0.3);
        let (pool, gate) = CpuPool::new_gated(4, REAL);
        let jobs = jobs_over(&region, REAL, &weights);
        let first = pool.run_layer(jobs, &hidden, SHORT);
        assert!(matches!(first, Err(PoolError::Timeout { .. })));
        gate.open();
        let jobs = jobs_over(&region, REAL, &weights);
        let next = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum(&jobs, &[0, 1, 2, 3, 4, 5, 6, 7], &hidden);
        assert_same_bits(&next, &want);
    }

    #[test]
    fn a_worker_error_names_the_lowest_failing_job() {
        // Choice (stated): a shape the kernel refuses - inter = 100 is not a
        // multiple of 256, so expert_forward answers BadShape for every job while
        // the slices keep the task-formula length (the pool's validation passes;
        // the failure surfaces from the kernel). All five jobs fail identically,
        // so job 0 must win whatever order they arrive in.
        let shape = ExpertShape {
            hidden: 2048,
            inter: 100,
        };
        let region = Arc::new(vec![0u8; 5 * 3 * 115_200]);
        let hidden = vec![0.25f32; 2048];
        let weights = [0.5f32, -0.5, 0.25, -0.25, 0.125];
        let pool = CpuPool::new(4, shape).unwrap();
        let jobs = jobs_over(&region, shape, &weights);
        let err = pool.run_layer(jobs, &hidden, LONG).unwrap_err();
        let PoolError::Expert { job, error } = err else {
            panic!("expected Expert, got {err:?}")
        };
        assert_eq!(job, 0);
        assert!(matches!(error, ExpertCpuError::BadShape(_)), "{error:?}");
    }

    #[test]
    fn drop_joins_workers_without_panicking() {
        // Create, run, drop - 50 times; every run stays bitwise correct (the pool
        // value drops at each loop-body end, joining its workers).
        let (region, hidden, weights) = setup(&mut Lcg(0xD0C7_5EED_0000_0004), 2, -0.3, 0.4);
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum(&jobs, &[0, 1], &hidden);
        for _ in 0..50 {
            let pool = CpuPool::new(2, REAL).unwrap();
            let jobs = jobs_over(&region, REAL, &weights);
            let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
            assert_same_bits(&got, &want);
        }
    }

    /// Serial reference for the Q8_K kernel: the pool's contract on the calling
    /// thread, each expert running `expert_forward_q8k` through `backend` (the
    /// f32 twin is `serial_sum`).
    fn serial_sum_q8k(
        backend: DotBackend,
        jobs: &[ExpertJob],
        order: &[usize],
        hidden: &[f32],
    ) -> Vec<f32> {
        let mut scratch = ExpertScratchQ8k::new(REAL);
        let mut out = vec![0.0; REAL.hidden];
        let mut sum = vec![0.0; REAL.hidden];
        for &e in order {
            let job = &jobs[e];
            let bytes = job.region.bytes();
            let slices = ExpertSlices {
                gate: &bytes[job.gate.clone()],
                up: &bytes[job.up.clone()],
                down: &bytes[job.down.clone()],
            };
            expert_forward_q8k(backend, REAL, &slices, hidden, &mut scratch, &mut out).unwrap();
            sum.iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s += job.weight * v);
        }
        sum
    }

    #[test]
    fn with_kernel_reports_its_kernel_and_new_is_f32() {
        // `new` keeps meaning F32; `with_kernel` reports the kernel it was given,
        // either kind, and refuses zero threads exactly like `new`.
        let f32_pool = CpuPool::new(2, REAL).unwrap();
        assert_eq!(f32_pool.kernel(), ExpertKernel::F32);
        let explicit = CpuPool::with_kernel(2, REAL, ExpertKernel::F32).unwrap();
        assert_eq!(explicit.kernel(), ExpertKernel::F32);
        let kernel = ExpertKernel::Q8k(DotBackend::scalar());
        let pool = CpuPool::with_kernel(3, REAL, kernel).unwrap();
        assert_eq!(pool.kernel(), kernel);
        assert_eq!(pool.threads(), 3);
        assert!(matches!(
            CpuPool::with_kernel(0, REAL, kernel),
            Err(PoolError::BadInput(_))
        ));
    }

    #[test]
    fn q8k_pool_matches_a_serial_sum_of_expert_forward_q8k() {
        // 8 random experts of the real shape; a 4-thread Q8k(detect()) pool equals
        // the serial ordered sum computed with expert_forward_q8k directly,
        // bitwise (the pool's order contract does not depend on the kernel).
        let (region, hidden, weights) = setup(&mut Lcg(0x1B1E_55AD_EA7D_0007), 8, -0.5, 0.5);
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum_q8k(
            DotBackend::detect(),
            &jobs,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &hidden,
        );
        let pool = CpuPool::with_kernel(4, REAL, ExpertKernel::Q8k(DotBackend::detect())).unwrap();
        assert_eq!(pool.kernel(), ExpertKernel::Q8k(DotBackend::detect()));
        let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
        assert_same_bits(&got, &want);
    }

    /// Serial reference for the per-32 kernel (the Q8_K twin is `serial_sum_q8k`).
    fn serial_sum_q8g(
        backend: DotBackend,
        jobs: &[ExpertJob],
        order: &[usize],
        hidden: &[f32],
    ) -> Vec<f32> {
        let mut scratch = ExpertScratchQ8g::new(REAL);
        let mut out = vec![0.0; REAL.hidden];
        let mut sum = vec![0.0; REAL.hidden];
        for &e in order {
            let job = &jobs[e];
            let bytes = job.region.bytes();
            let slices = ExpertSlices {
                gate: &bytes[job.gate.clone()],
                up: &bytes[job.up.clone()],
                down: &bytes[job.down.clone()],
            };
            expert_forward_q8g(backend, REAL, &slices, hidden, &mut scratch, &mut out).unwrap();
            sum.iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s += job.weight * v);
        }
        sum
    }

    #[test]
    fn q8g_pool_sum_is_independent_of_thread_count_and_backend() {
        // The per-32 kernels are bitwise equal across backends and the pool sums in job
        // order, so 1, 2, 3 and 8 threads with the scalar and the detected backend all give
        // the bits of the serial scalar reference.
        let (region, hidden, weights) = setup(&mut Lcg(0x6E32_0C0A_1ED0_0009), 8, -0.5, 0.5);
        let jobs = jobs_over(&region, REAL, &weights);
        let order = [0, 1, 2, 3, 4, 5, 6, 7];
        let want = serial_sum_q8g(DotBackend::scalar(), &jobs, &order, &hidden);
        for backend in [DotBackend::scalar(), DotBackend::detect()] {
            for threads in [1usize, 2, 3, 8] {
                let kernel = ExpertKernel::Q8g(backend);
                let pool = CpuPool::with_kernel(threads, REAL, kernel).unwrap();
                assert_eq!(pool.kernel(), kernel);
                let jobs = jobs_over(&region, REAL, &weights);
                let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
                assert_same_bits(&got, &want);
            }
        }
    }

    #[test]
    fn q8k_pool_sum_is_independent_of_thread_count_and_backend() {
        // The Q8_K kernels are bitwise equal across backends and the pool sums in
        // job order, so 1, 2, 3 and 8 threads with the scalar and the detected
        // backend all give the bits of the serial scalar reference.
        let (region, hidden, weights) = setup(&mut Lcg(0x7EAD_0C0A_1ED0_0008), 8, -0.5, 0.5);
        let jobs = jobs_over(&region, REAL, &weights);
        let order = [0, 1, 2, 3, 4, 5, 6, 7];
        let want = serial_sum_q8k(DotBackend::scalar(), &jobs, &order, &hidden);
        for backend in [DotBackend::scalar(), DotBackend::detect()] {
            for threads in [1usize, 2, 3, 8] {
                let kernel = ExpertKernel::Q8k(backend);
                let pool = CpuPool::with_kernel(threads, REAL, kernel).unwrap();
                let jobs = jobs_over(&region, REAL, &weights);
                let got = pool.run_layer(jobs, &hidden, LONG).unwrap().unwrap();
                assert_same_bits(&got, &want);
            }
        }
    }

    #[test]
    fn q8k_pool_is_close_to_the_f32_pool() {
        // Same jobs through both kernels. Metric: the largest element error of the
        // layer sum relative to the largest element of the f32 sum. The Q8_K path
        // quantizes the activation rows (half a step per value), so the sums differ
        // by that cost, not bitwise. Router-like positive weights, zero-mean random
        // weight bytes and activations in [-1, 1].
        let (region, hidden, weights) = setup(&mut Lcg(0x0C1E_5ED0_F32C_0009), 8, 0.05, 0.4);
        let f32_pool = CpuPool::new(4, REAL).unwrap();
        let kernel = ExpertKernel::Q8k(DotBackend::detect());
        let q8k_pool = CpuPool::with_kernel(4, REAL, kernel).unwrap();
        let want = f32_pool
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap()
            .unwrap();
        let got = q8k_pool
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap()
            .unwrap();
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(peak > 0.0);
        // Measured 8.7e-3 on this seeded data; the bound keeps about 3x headroom.
        assert!(worst / peak <= 3e-2, "relative error {:e}", worst / peak);
    }

    #[test]
    fn q8k_pool_reports_kernel_errors_and_timeouts_like_the_f32_pool() {
        // A shape the kernel refuses (inter = 100) names job 0 through the Q8_K
        // kernel, exactly as the f32 pool does.
        let shape = ExpertShape {
            hidden: 2048,
            inter: 100,
        };
        let kernel = ExpertKernel::Q8k(DotBackend::detect());
        let region = Arc::new(vec![0u8; 5 * 3 * 115_200]);
        let weights = [0.5f32, -0.5, 0.25, -0.25, 0.125];
        let pool = CpuPool::with_kernel(4, shape, kernel).unwrap();
        let jobs = jobs_over(&region, shape, &weights);
        let err = pool
            .run_layer(jobs, &vec![0.25f32; 2048], LONG)
            .unwrap_err();
        let PoolError::Expert { job, error } = err else {
            panic!("expected Expert, got {err:?}")
        };
        assert_eq!(job, 0);
        assert!(matches!(error, ExpertCpuError::BadShape(_)), "{error:?}");
        // A gated Q8_K pool times out, and after the gate opens the next run is
        // correct (late results of the first run never leak into it).
        let (region, hidden, weights) = setup(&mut Lcg(0x6A7E_D8C0_0000_000A), 8, -0.3, 0.3);
        let (pool, gate) = CpuPool::new_gated_kernel(4, REAL, kernel);
        let first = pool.run_layer(jobs_over(&region, REAL, &weights), &hidden, SHORT);
        assert!(matches!(first, Err(PoolError::Timeout { .. })), "{first:?}");
        gate.open();
        let next = pool
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap()
            .unwrap();
        let jobs = jobs_over(&region, REAL, &weights);
        let want = serial_sum_q8k(
            DotBackend::detect(),
            &jobs,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &hidden,
        );
        assert_same_bits(&next, &want);
    }

    #[test]
    fn start_layer_then_finish_equals_run_layer_bitwise() {
        // The caller may do other work (here: a second pool's run) between the
        // two halves; the sum is the job-order sum either way.
        let (region, hidden, weights) = setup(&mut Lcg(0x57A2_7F1E_0000_000B), 8, -0.5, 0.5);
        let kernel = ExpertKernel::Q8k(DotBackend::detect());
        let pool = CpuPool::with_kernel(4, REAL, kernel).unwrap();
        let pending = pool
            .start_layer(jobs_over(&region, REAL, &weights), &hidden)
            .unwrap()
            .expect("jobs were given");
        let other = CpuPool::new(2, REAL).unwrap();
        other
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap();
        let split = pending.finish(LONG).unwrap();
        let whole = pool
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap()
            .unwrap();
        assert_same_bits(&split, &whole);
    }

    #[test]
    fn start_layer_validates_like_run_layer_and_returns_none_for_no_jobs() {
        let (region, hidden, weights) = setup(&mut Lcg(0x57A2_7F1E_0000_000C), 2, -0.5, 0.5);
        let pool = CpuPool::new(2, REAL).unwrap();
        assert!(pool.start_layer(Vec::new(), &hidden).unwrap().is_none());
        let short = &hidden[..hidden.len() - 1];
        assert!(matches!(
            pool.start_layer(jobs_over(&region, REAL, &weights), short),
            Err(PoolError::BadInput(_))
        ));
        let mut jobs = jobs_over(&region, REAL, &weights);
        jobs[1].weight = f32::NAN;
        assert!(matches!(
            pool.start_layer(jobs, &hidden),
            Err(PoolError::BadInput(_))
        ));
        // Nothing was queued by the refusals: the next run is still correct.
        let got = pool
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap()
            .unwrap();
        let jobs = jobs_over(&region, REAL, &weights);
        assert_same_bits(&got, &serial_sum(&jobs, &[0, 1], &hidden));
    }

    #[test]
    fn a_pending_layer_times_out_typed_and_a_dropped_one_is_harmless() {
        let (region, hidden, weights) = setup(&mut Lcg(0x57A2_7F1E_0000_000D), 8, -0.3, 0.3);
        let (pool, gate) = CpuPool::new_gated(4, REAL);
        let pending = pool
            .start_layer(jobs_over(&region, REAL, &weights), &hidden)
            .unwrap()
            .expect("jobs were given");
        let err = pending.finish(SHORT).unwrap_err();
        assert!(matches!(err, PoolError::Timeout { .. }), "{err:?}");
        let dropped = pool
            .start_layer(jobs_over(&region, REAL, &weights), &hidden)
            .unwrap();
        drop(dropped);
        gate.open();
        let next = pool
            .run_layer(jobs_over(&region, REAL, &weights), &hidden, LONG)
            .unwrap()
            .unwrap();
        let jobs = jobs_over(&region, REAL, &weights);
        assert_same_bits(
            &next,
            &serial_sum(&jobs, &[0, 1, 2, 3, 4, 5, 6, 7], &hidden),
        );
    }
}
