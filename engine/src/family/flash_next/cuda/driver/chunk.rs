//! The chunk driver: one prompt chunk of `m` tokens end to end, behind the same rules as the
//! token driver.
//!
//! WHAT: [`ChunkDriver`] mirrors [`super::TokenDriver`] at chunk granularity — the prefill
//! counterpart of the decode step. Per chunk:
//!
//! 1. [`super::TokenFeeds::begin`] per token (host reads; a batched pack read can replace the
//!    loop later without touching this contract).
//! 2. The chunk start: step parameters `(token0, pos, m)`, the `m` stream rows, the `m` PLE
//!    embeddings.
//! 3. The layers before the PLE layer, `m` rows at `pos`.
//! 4. ONE host roundtrip: the `m` stream rows come back, each row gets its own
//!    [`super::TokenFeeds::ple_update`] on the host, and the rows return in one write.
//! 5. The remaining layers.
//! 6. With logits wanted: the final mixer and `lm_head` over the LAST row only, and the read-back
//!    of its logits row.
//! 7. The health check and the position advance by `m`.
//!
//! The rules carry over from the token driver unchanged: the fed count advances only when the
//! model state consumed the whole chunk; a failure after the first write to the card poisons the
//! driver (every later chunk is refused until `reset`); a refused id or an over-long chunk
//! changes nothing and does not poison.
//!
//! The device half is [`ChunkDevice`]. It exists so the sequence, the poison rule and the
//! position accounting run in host tests against fakes; the card implementation is staged for the
//! chunked prefill step (`step`/`mixer` at `m` rows).

use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use super::{DriverShape, ResumeSnapshot, TokenFeeds};
use crate::EngError;
use crate::family::flash_next::cuda::ple_feed::{HostProjection, PleKeyValue, PleProjection};
use crate::family::session::TokenStepper;

type Result<T> = std::result::Result<T, EngError>;

/// Coarse per-phase timing of the chunked prefill (INFERENCE_FLASH_NEXT_PROFILE=1): where a
/// chunk's wall time goes, accumulated over a prompt and dumped at its sampling row.
pub(in crate::family::flash_next) mod prof {
    use super::*;

    /// The phases of `queue` plus the inside of one layer, in call order. Slots 22-27 walk the
    /// interior of the ring rows-block, which the one-row block's taps (slots 11-13) never
    /// reach: the ring issues no admit and no miss pass. Slots 34-42 are the host time blocked
    /// inside the page-io waits — the `stream_synchronize` of a pageable copy — per site,
    /// upload and readback separately. They are nested inside the phases above them, so the
    /// phases sum counts them twice.
    pub(super) const NAMES: [&str; 43] = [
        "begin(host feeds)",
        "start writes",
        "layers<PLE",
        "read_stream",
        "ple_update(host)",
        "write_stream_back",
        "layers>=PLE",
        "head+read",
        "  hyper conn",
        "  mixer(attn)",
        "  moe_block",
        "    moe route",
        "    moe admit",
        "    moe launch",
        "    ple ngram gather",
        "    ple step math",
        "      ple plan+emb",
        "      ple key matvec",
        "      ple value matvec",
        "      ple branch norms",
        "      ple conv",
        "      ple ring",
        "  ring issue",
        "  ring route+drain",
        "  ring classify+ids",
        "  ring wait+gu",
        "  ring swiglu+down",
        "  ring shared+combine",
        "  rows classify",
        "  rows grouped gu",
        "  rows swiglu+down",
        "  rows shared+combine",
        "  rows dirty sync+step",
        "  rows miss+clean tails",
        "blk step slot_row up",
        "blk step log readback",
        "blk step miss uploads",
        "blk route picks readback",
        "blk rows admit slot_row up",
        "blk rows tail uploads",
        "blk ring weights up",
        "blk ring wait_layer",
        "blk filler readback",
    ];
    pub(super) static NS: [AtomicU64; 43] = [const { AtomicU64::new(0) }; 43];
    pub(super) static ROWS: AtomicU64 = AtomicU64::new(0);
    /// Chunk products the batch host route ran (m >= threads), and the rows it routed to the
    /// per-row projections instead (m < threads, decode's one-row chunks included). Counted
    /// unconditionally — a run proves its route only from a printed counter — but only printed
    /// under the profile switch.
    pub(super) static PRODUCTS: AtomicU64 = AtomicU64::new(0);
    pub(super) static FALLBACK_ROWS: AtomicU64 = AtomicU64::new(0);
    static ON: OnceLock<bool> = OnceLock::new();

    /// The route counters of this process so far, for the chunk-driver tests.
    pub(in crate::family::flash_next) fn route_counters() -> (u64, u64) {
        (
            PRODUCTS.load(AtomicOrdering::Relaxed),
            FALLBACK_ROWS.load(AtomicOrdering::Relaxed),
        )
    }

    /// Zeroes the route counters; tests call this so their reads are deltas.
    pub(in crate::family::flash_next) fn reset_route_counters() {
        PRODUCTS.store(0, AtomicOrdering::Relaxed);
        FALLBACK_ROWS.store(0, AtomicOrdering::Relaxed);
    }

    pub(in crate::family::flash_next) fn on() -> bool {
        *ON.get_or_init(|| {
            let on = parse_switch(
                std::env::var("INFERENCE_FLASH_NEXT_PROFILE")
                    .ok()
                    .as_deref(),
            )
            .unwrap_or(false);
            crate::kernels::page_io_block::arm(on);
            on
        })
    }

    /// The strict spelling the two diagnostic switches share: `1`, `true` or `on` arm them,
    /// unset, `0`, `false` or `off` leave them off, and anything else is refused at driver
    /// build by `check_switch_spelling`. The fast paths call the same parser, so the value
    /// that would be refused is never the value that arms.
    pub(in crate::family::flash_next) fn parse_switch(
        raw: Option<&str>,
    ) -> std::result::Result<bool, String> {
        match raw {
            None | Some("0") | Some("false") | Some("off") => Ok(false),
            Some("1") | Some("true") | Some("on") => Ok(true),
            Some(other) => Err(other.to_string()),
        }
    }

    /// Refuses an unknown spelling of the Flash-Next switches at driver build, before any
    /// chunk runs; a driver is never constructed with a value the fast paths would misread.
    pub(in crate::family::flash_next) fn check_switch_spelling() -> Result<()> {
        for name in [
            "INFERENCE_FLASH_NEXT_PROFILE",
            "INFERENCE_FLASH_NEXT_DRAIN_TICK",
            "INFERENCE_FLASH_NEXT_PREFIX_RESUME",
        ] {
            let refused = |seen: &str| {
                EngError::BadRequest(format!(
                    "Flash-Next driver: {name} must be 1, true, on, 0, false or off, not {seen}"
                ))
            };
            match std::env::var(name) {
                Err(std::env::VarError::NotPresent) => {}
                Err(std::env::VarError::NotUnicode(_)) => {
                    return Err(refused("a non-unicode value"));
                }
                Ok(value) => {
                    if let Err(bad) = parse_switch(Some(&value)) {
                        return Err(refused(&bad));
                    }
                }
            }
        }
        Ok(())
    }

    /// The host-stage measurement switch (`INFERENCE_FLASH_NEXT_DRAIN_TICK`): with the
    /// profile on, every phase boundary first synchronises the step stream and then closes
    /// the phase's slot, so a slot carries the wait that finishes its own queued work (the
    /// queue is empty at its start; the host-only part is the slot minus that wait). Only
    /// the step stream is drained — the prefill ring's copy stream and the expert bank's
    /// stream keep running. The mode serialises host and device, so walls under it are
    /// longer than without it and comparable only to each other.
    pub(in crate::family::flash_next) fn drain_tick() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            parse_switch(
                std::env::var("INFERENCE_FLASH_NEXT_DRAIN_TICK")
                    .ok()
                    .as_deref(),
            )
            .unwrap_or(false)
        })
    }

    pub(in crate::family::flash_next) fn add(slot: usize, spent: std::time::Duration) {
        NS[slot].fetch_add(spent.as_nanos() as u64, AtomicOrdering::Relaxed);
    }

    /// Runs `f`, charging its wall time to `slot` when the profile is on: the page-io
    /// sites time a call whose internals block on the stream, so the charge is the block.
    pub(in crate::family::flash_next) fn blocked<R>(
        slot: usize,
        on: bool,
        f: impl FnOnce() -> Result<R>,
    ) -> Result<R> {
        let started = on.then(std::time::Instant::now);
        let spent = f();
        if let Some(started) = started {
            add(slot, started.elapsed());
        }
        spent
    }

    /// Prints one line per phase and resets; call once per prompt, after its sampling row.
    pub(super) fn dump_and_reset(total_rows: u64) {
        let spent: Vec<u64> = NS.iter().map(|n| n.load(AtomicOrdering::Relaxed)).collect();
        let sum: u64 = spent.iter().sum();
        if sum == 0 {
            ROWS.store(0, AtomicOrdering::Relaxed);
            return;
        }
        for (name, ns) in NAMES.iter().zip(&spent) {
            eprintln!(
                "flash-next chunk profile: {name:18} {:9.0} ms {:5.1}%",
                *ns as f64 / 1e6,
                100.0 * *ns as f64 / sum as f64
            );
        }
        eprintln!(
            "flash-next chunk profile: {total_rows} rows, phases sum {:.1} s (the sum counts the interior slots 8-33 and the nested page-io slots 34-42 twice)",
            sum as f64 / 1e9
        );
        let products = PRODUCTS.load(AtomicOrdering::Relaxed);
        let fallback = FALLBACK_ROWS.load(AtomicOrdering::Relaxed);
        if products + fallback > 0 {
            eprintln!(
                "flash-next chunk profile: {products} chunk products, {fallback} per-row projection rows (chunks narrower than the projection threads)"
            );
        }
        if let Some((up, down)) = crate::kernels::page_io_block::take() {
            eprintln!(
                "flash-next chunk profile: page-io blocked: uploads {:.0} ms, readbacks {:.0} ms",
                up as f64 / 1e6,
                down as f64 / 1e6
            );
        }
        for n in &NS {
            n.store(0, AtomicOrdering::Relaxed);
        }
        ROWS.store(0, AtomicOrdering::Relaxed);
        PRODUCTS.store(0, AtomicOrdering::Relaxed);
        FALLBACK_ROWS.store(0, AtomicOrdering::Relaxed);
    }
}

/// One phase boundary under the profile: drain first, then charge the elapsed time to
/// the slot being closed. The drain synchronises the stream the phase just queued, so
/// its wait belongs inside that phase's slot; draining after the reset would push it
/// into the next phase. A free function (not the closure) so a host test can pin the
/// order against a fake device.
fn advance_boundary<D: ChunkDevice>(
    dev: &mut D,
    slot: usize,
    t: &mut Instant,
    profile: bool,
    drain_on: bool,
    add: impl FnOnce(usize, std::time::Duration),
) -> Result<()> {
    if drain_on {
        dev.drain()?;
    }
    if profile {
        add(slot, t.elapsed());
        *t = Instant::now();
    }
    Ok(())
}

/// The device side of a chunk. The start writes are the sink of the host feeds at chunk
/// granularity; every buffer holds `m` rows where the token step holds one.
pub(in crate::family::flash_next::cuda) trait ChunkDevice {
    /// Writes the step parameters of the chunk: its first token id, its start position and its
    /// row count.
    fn write_chunk_params(&mut self, token0: i32, pos: i32, m: i32) -> Result<()>;
    /// Writes the `m` stream rows (`m * branches * hidden`), the initial stream of the chunk.
    fn write_chunk_stream(&mut self, rows: &[f32]) -> Result<()>;
    /// Writes the `m` PLE embedding rows (`m * ple_embed`).
    fn write_chunk_ple(&mut self, rows: &[f32]) -> Result<()>;
    /// Runs the chunk's PLE projections on the device and returns each row's keys and value,
    /// behind the queued layers. Only a device with the PLE projections answers; the default
    /// refuses so a driver with the device route on cannot silently fall back to the host
    /// projections.
    fn ple_project_chunk(&mut self) -> Result<Vec<PleKeyValue>> {
        Err(EngError::BadRequest(
            "this device has no PLE projection route".to_string(),
        ))
    }
    /// Clears the per-chunk flags (the EXL3 flag and the status word). Called once per chunk,
    /// after the start is written and before the first layer.
    fn begin_chunk(&mut self) -> Result<()>;
    /// Re-plans the prefill expert ring against the bank as decode left it. Called once at the
    /// start of a chunked prefill (a `feed_rows` call), before the first chunk.
    fn begin_prefill(&mut self) -> Result<()> {
        Ok(())
    }
    /// Closes the prefill window after the sampling row ran. The single-token step of later
    /// decode uses the bank's miss path again; a no-op when no window is open.
    fn end_prefill(&mut self) -> Result<()> {
        Ok(())
    }
    /// Queues layers `range` for the chunk of `m` rows at `pos`.
    fn layers(&mut self, range: Range<usize>, pos: usize, m: usize) -> Result<()>;
    /// The `m` stream rows, read back behind the queued work.
    fn read_stream(&mut self) -> Result<Vec<f32>>;
    /// Replaces the `m` stream rows.
    fn write_stream_back(&mut self, rows: &[f32]) -> Result<()>;
    /// Queues the final mixer and the head over the last row of the chunk.
    fn head(&mut self) -> Result<()>;
    /// The logits row of the last row's head, behind the queued work.
    fn read_logits(&mut self) -> Result<Vec<f32>>;
    /// Reads the status word and the arithmetic flag; refuses a chunk that set either.
    fn check_health(&mut self) -> Result<()>;
    /// Synchronises the step stream of this device — the only stream the host-stage
    /// measurement mode (`INFERENCE_FLASH_NEXT_DRAIN_TICK` with the profile on) drains;
    /// the prefill ring's copy stream and the expert bank's stream keep running. It runs
    /// at a phase boundary before the phase's slot closes, so the slot carries the wait
    /// that finishes its own queued work; the default does nothing.
    fn drain(&mut self) -> Result<()> {
        Ok(())
    }
    /// Zeroes the persistent state: a new sequence.
    fn reset(&mut self) -> Result<()>;
    /// The bank's miss count for the per-job report, if the device has a bank.
    fn bank_misses(&self) -> Option<u64> {
        None
    }
}

/// The chunk driver: a [`ChunkDevice`] and its [`TokenFeeds`] under the token driver's rules.
/// `feed_rows` splits the prompt into device chunks of at most `chunk_cap` rows (1 = the token
/// loop, the decode shape).
pub(in crate::family::flash_next::cuda) struct ChunkDriver<D: ChunkDevice, F: TokenFeeds> {
    device: D,
    feeds: F,
    shape: DriverShape,
    /// Tokens the state has consumed: the next position.
    pos: usize,
    poisoned: bool,
    /// Advances once per fresh sequence (at the ATTEMPT of its first row, so a failed or
    /// partial first write also advances it; the row never needs to commit): the
    /// in-place rows a resume leaves on the card belong to the generation that wrote them, so a
    /// snapshot saved at another generation is structurally unusable and never matches.
    generation: u64,
    /// The most rows one device chunk runs; the caller sets it from the memory plan.
    pub(in crate::family::flash_next::cuda) chunk_cap: usize,
    /// Whether the PLE projections run on the device (`ChunkDevice::ple_project_chunk`).
    pub(in crate::family::flash_next::cuda) ple_device: bool,
    /// The batch host route: the projections of the whole chunk in one blocked product
    /// (`HostProjection::project_chunk`), each row's pair then feeding the unchanged host
    /// step through the fixed projection. `None` keeps the per-row projections of
    /// [`TokenFeeds::ple_update`].
    pub(in crate::family::flash_next::cuda) ple_host_batch: Option<Box<HostProjection>>,
}

impl<D: ChunkDevice, F: TokenFeeds> ChunkDriver<D, F> {
    pub(in crate::family::flash_next::cuda) fn new(
        device: D,
        feeds: F,
        shape: DriverShape,
    ) -> Result<Self> {
        prof::check_switch_spelling()?;
        if shape.layers <= shape.ple_layer {
            return Err(EngError::BadRequest(format!(
                "Flash-Next chunk driver: the PLE layer {} must sit inside the {} layers",
                shape.ple_layer, shape.layers
            )));
        }
        Ok(Self {
            device,
            feeds,
            shape,
            pos: 0,
            poisoned: false,
            generation: 0,
            chunk_cap: 1,
            ple_device: false,
            ple_host_batch: None,
        })
    }

    /// The number of positions the caches hold, for the executor.
    pub(super) fn positions(&self) -> usize {
        self.shape.positions
    }

    pub(in crate::family::flash_next::cuda) fn device(&self) -> &D {
        &self.device
    }

    /// The device for tests that arm its fakes before a run.
    pub(in crate::family::flash_next::cuda) fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    pub(super) fn feeds(&self) -> &F {
        &self.feeds
    }

    /// The chunk of `tokens` (at least one): feeds every token, runs the layers with one PLE
    /// roundtrip, and returns the logits of the LAST row when `want_logits`.
    pub(super) fn run_chunk(
        &mut self,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let m = tokens.len();
        if self.poisoned {
            return Err(EngError::BadRequest(
                "Flash-Next chunk driver: the state needs a reset after a failed step".into(),
            ));
        }
        if m == 0 {
            return Err(EngError::BadRequest(
                "Flash-Next chunk driver: a chunk needs at least one token".into(),
            ));
        }
        if self.pos + m > self.shape.positions {
            return Err(EngError::ContextLimit(format!(
                "Flash-Next chunk driver: the caches hold {} positions, a chunk of {m} at {} does not fit",
                self.shape.positions, self.pos
            )));
        }
        // A refused id changes nothing; the chunk start repeats the rule, but inside the queued
        // part, where a failure poisons the driver.
        if tokens.iter().any(|token| {
            usize::try_from(*token).is_ok_and(|id| id >= self.shape.start.sampler_vocab)
        }) {
            return Err(EngError::BadRequest(format!(
                "Flash-Next chunk driver: a token id is outside the sampler vocabulary 0..{}",
                self.shape.start.sampler_vocab
            )));
        }
        // Host reads only: a failure here leaves the state as it was.
        let t0 = Instant::now();
        let profile = prof::on();
        let starts = self.feeds.begin_chunk(tokens)?;
        if profile {
            prof::add(0, t0.elapsed());
            prof::ROWS.fetch_add(m as u64, AtomicOrdering::Relaxed);
        }
        // The generation advances at the ATTEMPT of a fresh sequence's first chunk, not only
        // on success: a failed chunk that queued device work may still have written the rows,
        // and a stale snapshot must never survive that.
        if self.pos == 0 {
            self.generation += 1;
        }
        let outcome = self.queue(tokens, &starts, want_logits);
        match outcome {
            Ok(logits) => {
                self.pos += m;
                Ok(logits)
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    /// Everything after the first write to the card.
    fn queue(
        &mut self,
        tokens: &[u32],
        starts: &[super::StartVectors],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let m = tokens.len();
        let pos = self.pos;
        let shape = self.shape;

        let profile = prof::on();
        let drain_on = profile && prof::drain_tick();
        let mut t = Instant::now();
        let mut tick = |dev: &mut D, slot: usize| -> Result<()> {
            advance_boundary(dev, slot, &mut t, profile, drain_on, prof::add)
        };
        self.device
            .write_chunk_params(tokens[0] as i32, pos as i32, m as i32)?;
        let mut stream_rows = Vec::with_capacity(m * shape.start.stream);
        let mut ple_rows = Vec::with_capacity(m * shape.start.ple_embed);
        for start in starts {
            stream_rows.extend_from_slice(&start.stream);
            ple_rows.extend_from_slice(&start.ple_embedding);
        }
        self.device.write_chunk_stream(&stream_rows)?;
        self.device.write_chunk_ple(&ple_rows)?;
        self.device.begin_chunk()?;
        tick(&mut self.device, 1)?;
        self.device.layers(0..shape.ple_layer, pos, m)?;
        tick(&mut self.device, 2)?;
        // The batch host route projects the chunk's embeddings after the pre-PLE layers are
        // queued, so the product runs while the card works (its input is the chunk's start
        // vectors, not device output). Chunks narrower than the projection threads keep the
        // per-row projections: a one-row product would sweep the 131 MB of weights on a single
        // thread, while `matvec` splits the rows across all of them.
        let host_pairs = match self.ple_host_batch.as_deref_mut() {
            Some(host) if m >= host.threads() => {
                prof::PRODUCTS.fetch_add(1, AtomicOrdering::Relaxed);
                Some(host.project_chunk(&ple_rows, m)?)
            }
            Some(host) => {
                let embed = shape.start.ple_embed;
                let rows = ple_rows
                    .chunks(embed)
                    .map(|embedding| host.project(embedding))
                    .collect::<Result<Vec<_>>>()?;
                prof::FALLBACK_ROWS.fetch_add(rows.len() as u64, AtomicOrdering::Relaxed);
                Some(rows)
            }
            None => None,
        };
        let mut rows = self.device.read_stream()?;
        tick(&mut self.device, 3)?;
        let width = shape.start.stream;
        if rows.len() != m * width {
            return Err(EngError::Backend {
                operation: "flash_next_cuda_chunk",
                message: format!(
                    "the chunk read back {} stream values for {m} rows of {width}",
                    rows.len()
                ),
            });
        }
        // The device route projects the chunk's embeddings on the card (one m-row launch pair)
        // and hands each row's keys/value to the host step, which then runs unchanged; the host
        // route runs its own projections inside `ple_update`.
        let mut pairs = (if self.ple_device {
            Some(self.device.ple_project_chunk()?)
        } else {
            host_pairs
        })
        .map(IntoIterator::into_iter);
        for (token, row) in tokens.iter().zip(rows.chunks_mut(width)) {
            if let Some(pair) = pairs.as_mut().and_then(std::iter::Iterator::next) {
                self.feeds.set_ple_pair(pair)?;
            }
            let update = self.feeds.ple_update(*token, row)?;
            if update.len() != width {
                return Err(EngError::BadRequest(format!(
                    "Flash-Next chunk driver: the PLE update has {} values for a stream of {width}",
                    update.len()
                )));
            }
            for (value, delta) in row.iter_mut().zip(&update) {
                *value += delta;
            }
        }
        tick(&mut self.device, 4)?;
        self.device.write_stream_back(&rows)?;
        tick(&mut self.device, 5)?;
        self.device.layers(shape.ple_layer..shape.layers, pos, m)?;
        tick(&mut self.device, 6)?;
        let logits = if want_logits {
            self.device.head()?;
            let row = self.device.read_logits()?;
            tick(&mut self.device, 7)?;
            Some(row)
        } else {
            None
        };
        self.device.check_health()?;
        Ok(logits)
    }
}

impl<D: ChunkDevice, F: TokenFeeds> super::StepperPositions for ChunkDriver<D, F> {
    fn positions(&self) -> usize {
        self.positions()
    }
}

impl<D: ChunkDevice, F: TokenFeeds> TokenStepper for ChunkDriver<D, F> {
    const OPERATION: &'static str = "flash_next_cuda_chunk";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.run_chunk(&[token], false).map(|_| ())
    }

    /// The prefill path: the whole prompt chunk lands as device chunks of at most `chunk_cap`
    /// rows, so the executor's `feed_rows` call is the chunked prefill entry.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let cap = self.chunk_cap.max(1);
        self.device.begin_prefill()?;
        for chunk in tokens.chunks(cap) {
            self.run_chunk(chunk, false).map(|_| ())?;
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        let logits = self
            .run_chunk(&[token], true)?
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".to_string(),
            })?;
        // The sampling row closed the prefill: the ring's window ends with it, and every
        // later single-token step takes the decode path. A failure here leaves the prefill
        // window open on an advanced position: poison, so a reset is required.
        if let Err(error) = self.device.end_prefill() {
            self.poisoned = true;
            return Err(error);
        }
        if prof::on() {
            prof::dump_and_reset(prof::ROWS.load(AtomicOrdering::Relaxed));
        }
        Ok(logits)
    }

    fn fed(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<()> {
        self.device.reset()?;
        self.feeds.reset();
        self.pos = 0;
        self.poisoned = false;
        Ok(())
    }
}

impl<D: ChunkDevice + super::StateIo, F: TokenFeeds> super::ResumesState for ChunkDriver<D, F> {
    fn resume_generation(&self) -> u64 {
        self.generation
    }

    fn resume_host_bytes(&self) -> u64 {
        let (ple_history, ple_ring) = self.feeds.ple_state().unwrap_or((&[], &[]));
        let history = (ple_history.len() as u64) * std::mem::size_of::<u32>() as u64;
        let ring = (ple_ring.len() as u64) * std::mem::size_of::<f32>() as u64;
        self.device.fixed_state_bytes() + history + ring
    }

    fn save_resume(&self) -> Result<ResumeSnapshot> {
        let parts = self.device.read_fixed_state()?;
        let (ple_history, ple_ring) = self.feeds.ple_state().unwrap_or((&[], &[]));
        Ok(ResumeSnapshot {
            parts,
            pos: self.pos,
            ple_history: ple_history.to_vec(),
            ple_ring: ple_ring.to_vec(),
        })
    }

    fn restore_resume(&mut self, snapshot: &ResumeSnapshot) -> Result<()> {
        self.device.write_fixed_state(&snapshot.parts)?;
        if !snapshot.ple_ring.is_empty() {
            self.feeds
                .ple_resume(&snapshot.ple_history, &snapshot.ple_ring)?;
        }
        self.pos = snapshot.pos;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
