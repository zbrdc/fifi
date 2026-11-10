//! The host side of the tiered decode step: classify the routed experts of a
//! layer against the residency table, start the CPU jobs of the missed ones, and decide
//! the bank changes of the token boundary.
//!
//! WHAT: [`TieredHost`] owns the tier policy, the residency ledger, the expert locator
//! over the mapped GGUF, the CPU pool and the per-token counters. The device side is a
//! trait ([`BankDevice`]), so everything here runs and is tested without CUDA. WHY: the
//! engine must decide, per layer and between the router and the expert launches, which
//! picks the GPU bank serves and which the CPU serves, without a second place that could
//! disagree with the ledger.
//!
//! The rules of one layer: a missed pick maps to bank slot 0 with device weight 0.0 so the
//! existing expert launches stay unchanged, and the router-fault id `-1` passes through.
//! The rules of the token boundary: every swap goes through the full residency state
//! machine, and all publication happens here, never inside a layer. Driven by the tiered
//! step.

use std::fmt;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use super::cpu_pool::{CpuPool, MappedRegion, PendingLayer, PoolError};
use super::guess::{RecallMeter, RouterGuess};
use super::locator::{ExpertLocator, ExpertRanges, LocatorError};
use super::policy::{ExpertPolicy, Swap};
use super::report::TokenReport;
use super::residency::{EventId, ResidencyError, ResidencyTable};
use super::{PolicyMode, TieredConfig};

/// Why a host operation failed. Every error leaves the ledger consistent; a failed bank
/// fill additionally desynchronises the policy from the ledger (see
/// [`HostError::Poisoned`]).
#[derive(Debug, PartialEq)]
pub enum HostError {
    /// The configuration, locator and pool do not describe the same model.
    Config(String),
    /// A route the host refuses: a layer or expert out of range, a duplicate pick, a
    /// non-finite weight, or lists of different lengths. Nothing was started.
    BadRoute(String),
    Order(String),
    Pool(PoolError),
    Residency(ResidencyError),
    Locator(LocatorError),
    Device(String),
    /// A bank fill failed earlier: the policy and the ledger no longer agree, so
    /// `end_token` refuses until the host is rebuilt. Classification still reads the
    /// ledger, so layers keep being served correctly (every unfilled pick on the CPU).
    Poisoned,
    Inconsistent(String),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Config(m) => write!(f, "tiered host config: {m}"),
            HostError::BadRoute(m) => write!(f, "tiered host route: {m}"),
            HostError::Order(m) => write!(f, "tiered host order: {m}"),
            HostError::Pool(e) => write!(f, "tiered host cpu pool: {e:?}"),
            HostError::Residency(e) => write!(f, "tiered host residency: {e:?}"),
            HostError::Locator(e) => write!(f, "tiered host locator: {e:?}"),
            HostError::Device(m) => write!(f, "tiered host device: {m}"),
            HostError::Poisoned => write!(f, "tiered host poisoned by a failed bank fill"),
            HostError::Inconsistent(m) => write!(f, "tiered host inconsistent: {m}"),
        }
    }
}

impl std::error::Error for HostError {}

/// The device half of a bank change: copies the three slices of one expert into a slot.
/// The real implementation writes them with blocking copies; tests use a mock.
pub trait BankDevice {
    /// Fills `slot` of the bank of `layer` with the expert whose bytes `ranges` locate
    /// inside `region`. Returns when the fill is QUEUED (a blocking implementation
    /// returns when the bytes are on the device); completion is observed through
    /// [`BankDevice::wait_fills`].
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError>;

    /// Waits until every fill queued by [`BankDevice::fill_slot`] has landed, so the
    /// caller can complete the events and publish. The default says every blocking
    /// fill already landed; an asynchronous implementation overrides it. `Err` means
    /// a queued fill failed: the caller poisons exactly as a failed `fill_slot`.
    fn wait_fills(&mut self) -> Result<(), HostError> {
        Ok(())
    }
}

/// What the device launches of one layer need: per pick the bank slot to read and the
/// weight the combine applies.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPlanHost {
    /// Bank slot per pick; `-1` is the router-fault id passed through unchanged; a missed
    /// expert reads slot 0.
    pub slot_ids: Vec<i32>,
    /// Combine weight per pick; `0.0` for the picks the CPU serves.
    pub device_weights: Vec<f32>,
}

struct InFlight {
    layer: usize,
    jobs: Option<PendingLayer>,
}

/// A router weight in millionths for the weight-mass counters; a negative weight counts as zero.
fn weight_ppm(weight: f32) -> u64 {
    (f64::from(weight).max(0.0) * 1e6).round() as u64
}

/// One row of a chunk's route: the token-width inputs of `begin_layer`, one row of
/// the M the chunk feeds at once. The CPU pool reads the chunk's shared normed rows
/// separately (`start_chunk_pool`'s block), so no hidden slice rides per row.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkRouteRow {
    pub ids: Vec<i32>,
    pub weights: Vec<f32>,
}

/// The device plan of one chunk layer: `rows.len() * top_k` entries, row-major (row r's
/// picks at `r * top_k..(r+1) * top_k`), the M-row twin of [`LayerPlanHost`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkLayerPlanHost {
    pub slot_ids: Vec<i32>,
    pub device_weights: Vec<f32>,
    /// The cold staging channel per pick: `-1` when the pick is not cold-served (a hit,
    /// an overflow miss or a router fault), else its channel `0..cold.len()`. A cold
    /// pick's `slot_ids` entry is unused (0) and its `device_weights` entry carries the
    /// REAL router weight — the staging launch computes the pick on the device.
    pub cold_ids: Vec<i32>,
}

/// The CPU work of one chunk layer, grouped by expert: one entry per DISTINCT missed
/// expert, carrying the rows that routed to it with their per-row weights — the shape
/// the grouped pool job (slice 4) consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkMissGroup {
    pub expert: u16,
    pub rows: Vec<usize>,
    pub row_weights: Vec<f32>,
}

/// One expert the cold staging bank serves for a chunk layer: its channel and the byte
/// ranges of its three stacks in the checkpoint mapping (the fill gathers them
/// stack-major into the pinned ring and DMAs them into the staging slots).
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkColdStage {
    pub expert: u16,
    pub channel: u16,
    pub gate: std::ops::Range<usize>,
    pub up: std::ops::Range<usize>,
    pub down: std::ops::Range<usize>,
}

/// The picks of every layer of one token as the device logged them: `(id, weight)` in rank order,
/// exactly `top_k` per layer (`id == -1` is the router-fault value, any other negative id is
/// refused). WHY: the classification moves onto the device, so the host learns a token's
/// routes after it ran and accounts for them at the boundary with [`TieredHost::account_token`].
/// First caller: the mailbox's CUDA adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteLog {
    layers: usize,
    top_k: usize,
    picks: Vec<(i32, f32)>,
}

impl RouteLog {
    pub fn new(layers: usize, top_k: usize, picks: Vec<(i32, f32)>) -> Result<Self, HostError> {
        if layers.checked_mul(top_k) != Some(picks.len()) {
            return Err(HostError::BadRoute(format!(
                "a log of {layers} layers of {top_k} picks holds {} picks",
                picks.len()
            )));
        }
        Ok(Self {
            layers,
            top_k,
            picks,
        })
    }

    fn layer(&self, layer: usize) -> &[(i32, f32)] {
        let at = layer.saturating_mul(self.top_k);
        self.picks.get(at..at + self.top_k).unwrap_or(&[])
    }
}

/// The guess side channel of one host: a worker thread that turns a layer's raw residual
/// into the next layer's ranked expert advice, off the step thread (one guess is a
/// 2 MiB router matvec, ~0.2 ms — 39 of them would cost a measurable slice of a token).
/// The worker owns a cheap clone of the (immutable, `Arc`-backed) [`RouterGuess`]; the
/// host hands it `(layer, resid)` after a layer's route readback and consumes whatever
/// advice has arrived by the next `begin_layer` of the target layer. Stale advice (the
/// worker lost the race) is dropped, never queued: advice exists to be early.
struct GuessWorker {
    k: usize,
    to_worker: mpsc::Sender<(usize, Vec<f32>)>,
    from_worker: mpsc::Receiver<(usize, Vec<u16>)>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for GuessWorker {
    fn drop(&mut self) {
        // Dropping the sender closes the channel; the worker exits on the recv error
        // after at most one in-flight guess (~0.2 ms).
        self.to_worker = mpsc::channel().0;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub struct TieredHost {
    cfg: TieredConfig,
    locator: ExpertLocator,
    region: Arc<dyn MappedRegion>,
    pool: CpuPool,
    timeout: Duration,
    policy: ExpertPolicy,
    table: ResidencyTable,
    /// The valid expert ids routed by this token, per layer, in routing order.
    routes: Vec<Vec<u16>>,
    in_flight: Option<InFlight>,
    report: TokenReport,
    next_event: EventId,
    poisoned: bool,
    /// Fills queued since the last drain: their events complete (and their slots
    /// publish) only at the token boundary or after preload, never mid-token.
    pending_fills: Vec<EventId>,
    /// The router-guess prefetch side channel, set by [`TieredHost::set_guess`].
    guess: Option<GuessWorker>,
    /// The advice consumed by the latest `begin_layer` of its target layer, taken by the
    /// runtime for the prefetch fills: `(target layer, ranked ids)`.
    arrived: Option<(usize, Vec<u16>)>,
    /// Running recall of every consumed advice against the route it met.
    meter: RecallMeter,
    /// The meter counts already reported, so a token report carries only its delta.
    meter_seen: (u64, u64),
}

/// The ledger deadline of a synchronous fill: any later clock value expires it, which is how
/// `abandon_fills` poisons the slot of a fill the device failed.
const FILL_DEADLINE_NS: u64 = 1;

/// The per-call cap of evicting guess promotions: a guess has no routed evidence, so at
/// 77 percent recall a wrong promotion evicts a proven resident and costs the token BOTH
/// experts on the CPU. Free-slot promotions are uncapped (they evict nothing).
const GUESS_EVICT_CAP: usize = 2;

impl TieredHost {
    /// Builds the host for an adaptive policy. `Err(Config)` when the config is invalid,
    /// is not adaptive (a static profile is not wired in v1), or disagrees with the
    /// locator (layers, experts) or the pool (expert shape).
    pub fn new(
        cfg: TieredConfig,
        locator: ExpertLocator,
        region: Arc<dyn MappedRegion>,
        pool: CpuPool,
        timeout: Duration,
    ) -> Result<Self, HostError> {
        cfg.check().map_err(HostError::Config)?;
        if cfg.mode != PolicyMode::Adaptive {
            return Err(HostError::Config(String::from(
                "only the adaptive policy is wired in v1",
            )));
        }
        if locator.layers() != cfg.layers || locator.experts() != cfg.experts {
            return Err(HostError::Config(format!(
                "the locator indexes {} layers of {} experts, the config {} of {}",
                locator.layers(),
                locator.experts(),
                cfg.layers,
                cfg.experts
            )));
        }
        if pool.shape() != locator.shape() {
            return Err(HostError::Config(String::from(
                "the cpu pool and the locator use different expert shapes",
            )));
        }
        let policy = ExpertPolicy::new(&cfg, None).map_err(HostError::Config)?;
        Ok(Self {
            table: ResidencyTable::new(&cfg),
            routes: vec![Vec::new(); cfg.layers],
            policy,
            cfg,
            locator,
            region,
            pool,
            timeout,
            in_flight: None,
            report: TokenReport::default(),
            next_event: 1,
            poisoned: false,
            pending_fills: Vec::new(),
            guess: None,
            arrived: None,
            meter: RecallMeter::default(),
            meter_seen: (0, 0),
        })
    }

    /// Classifies the picks of `layer` and starts the CPU jobs of the missed ones.
    ///
    /// `ids` and `weights` are the router's picks in rank order (`ids[i] == -1` is the
    /// router-fault value; any other negative id is refused); `hidden` is the normed row the
    /// CPU experts read; `resid` is the RAW residual before the block's norm, the input the
    /// router guess of the NEXT layer wants (it applies its own rmsnorm). A resident pick
    /// maps to its bank slot with its weight; a missed pick maps to slot 0 with weight 0.0
    /// and one CPU job; the fault id -1 passes through. Nothing is started when the route
    /// is refused. When a guess worker is set, the advice that arrived for this layer is
    /// metered against the real route, staged for the runtime's prefetch fills
    /// ([`TieredHost::take_arrived_advice`]), and the residual is handed to the worker for
    /// the next layer's advice.
    pub fn begin_layer(
        &mut self,
        layer: usize,
        ids: &[i32],
        weights: &[f32],
        hidden: &[f32],
        resid: &[f32],
    ) -> Result<LayerPlanHost, HostError> {
        if let Some(open) = &self.in_flight {
            return Err(HostError::Order(format!(
                "layer {layer} begun while layer {} is unfinished",
                open.layer
            )));
        }
        self.check_route(layer, ids, weights)?;
        if let Some(worker) = &self.guess {
            let mut arrived = None;
            while let Ok((target, advice)) = worker.from_worker.try_recv() {
                if target == layer {
                    arrived = Some((target, advice));
                }
            }
            if let Some((_, advice)) = &arrived {
                let actual: Vec<u16> = ids
                    .iter()
                    .filter(|&&id| id >= 0)
                    .map(|&id| id as u16)
                    .collect();
                self.meter.add(advice, &actual);
            }
            self.arrived = arrived;
            let _ = worker.to_worker.send((layer, resid.to_vec()));
        }
        let mut plan = LayerPlanHost {
            slot_ids: Vec::with_capacity(ids.len()),
            device_weights: Vec::with_capacity(ids.len()),
        };
        let mut jobs = Vec::new();
        let (mut hits, mut served) = (0u32, 0u32);
        let (mut routed_ppm, mut cpu_ppm) = (0u64, 0u64);
        for (&id, &weight) in ids.iter().zip(weights) {
            if id < 0 {
                plan.slot_ids.push(id);
                plan.device_weights.push(weight);
                continue;
            }
            let expert = id as u16;
            let ppm = weight_ppm(weight);
            routed_ppm += ppm;
            if let Some((slot, _)) = self.table.lookup(layer, expert) {
                plan.slot_ids.push(slot as i32);
                plan.device_weights.push(weight);
                hits += 1;
            } else {
                let job = self
                    .locator
                    .job(Arc::clone(&self.region), layer, usize::from(expert), weight)
                    .map_err(HostError::Locator)?;
                jobs.push(job);
                plan.slot_ids.push(0);
                plan.device_weights.push(0.0);
                served += 1;
                cpu_ppm += ppm;
            }
        }
        let started = self
            .pool
            .start_layer(jobs, hidden)
            .map_err(HostError::Pool)?;
        self.routes[layer] = ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u16)
            .collect();
        self.in_flight = Some(InFlight {
            layer,
            jobs: started,
        });
        self.report.bank_hits += hits;
        self.report.cpu_served += served;
        self.report.routed_weight_ppm += routed_ppm;
        self.report.cpu_weight_ppm += cpu_ppm;
        Ok(plan)
    }

    /// Classifies the picks of `rows.len()` rows of `layer` at once and returns the device
    /// plan, the cold staging list and the grouped OVERFLOW misses. The M-row twin of
    /// [`Self::begin_layer`]: validate EVERY row first (the `account_token` pattern — a
    /// refusal changes nothing), then classify all rows' picks against the table three
    /// ways — a hit takes its bank slot, a miss of one of the `cold_cap` HOTTEST distinct
    /// missed experts (by pick count, ties by expert id) takes a cold staging channel
    /// with its REAL router weight, and every missed expert past `cold_cap` stays a CPU
    /// pool group with the weight-0.0 dummy of today — then stash the routes per row for
    /// the chunk boundary. `cold_cap` 0 keeps today's two-way plan exactly.
    pub fn begin_chunk_layer(
        &mut self,
        layer: usize,
        rows: &[ChunkRouteRow],
        cold_cap: usize,
    ) -> Result<(ChunkLayerPlanHost, Vec<ChunkMissGroup>, Vec<ChunkColdStage>), HostError> {
        if let Some(open) = &self.in_flight {
            return Err(HostError::Order(format!(
                "begin_chunk_layer({layer}) while layer {} is in flight",
                open.layer
            )));
        }
        // Validate every row before classifying any: a bad row changes nothing.
        for (row_at, row) in rows.iter().enumerate() {
            match self.check_route(layer, &row.ids, &row.weights) {
                Ok(()) => {}
                Err(error) => {
                    return Err(HostError::BadRoute(format!("row {row_at}: {error:?}")));
                }
            }
        }
        // Count the missed picks per distinct expert first: the hottest `cold_cap` take
        // the staging channels (the cut that minimizes overflowed picks for a fixed cap).
        let mut miss_picks: Vec<usize> = vec![0; self.cfg.experts];
        for row in rows {
            for &id in &row.ids {
                if id >= 0 && self.table.lookup(layer, id as u16).is_none() {
                    miss_picks[id as usize] += 1;
                }
            }
        }
        let mut hot: Vec<u16> = (0..self.cfg.experts)
            .filter(|&e| miss_picks[e] > 0)
            .map(|e| e as u16)
            .collect();
        hot.sort_by(|&a, &b| {
            miss_picks[usize::from(b)]
                .cmp(&miss_picks[usize::from(a)])
                .then(a.cmp(&b))
        });
        hot.truncate(cold_cap);
        let channel_of = |expert: u16| hot.iter().position(|&e| e == expert);
        let mut plan = ChunkLayerPlanHost {
            slot_ids: Vec::with_capacity(rows.len() * self.cfg.top_k),
            device_weights: Vec::with_capacity(rows.len() * self.cfg.top_k),
            cold_ids: vec![-1; rows.len() * self.cfg.top_k],
        };
        let mut misses: Vec<ChunkMissGroup> = Vec::new();
        let mut row_routes: Vec<Vec<u16>> = Vec::with_capacity(rows.len());
        for row in rows {
            let mut route = Vec::with_capacity(row.ids.len());
            for (&id, &weight) in row.ids.iter().zip(&row.weights) {
                if id < 0 {
                    plan.slot_ids.push(id);
                    plan.device_weights.push(weight);
                    continue;
                }
                let expert = id as u16;
                let flat = plan.slot_ids.len();
                // The same accounting the decode classify makes per pick, so a chunk
                // piece's report separates bank hits, cold-staged serves and pool jobs
                // (an all-zeros row means no traffic, not an unwired counter).
                let ppm = weight_ppm(weight);
                self.report.routed_weight_ppm += ppm;
                if let Some((slot, _)) = self.table.lookup(layer, expert) {
                    plan.slot_ids.push(slot as i32);
                    plan.device_weights.push(weight);
                    self.report.bank_hits += 1;
                } else if let Some(channel) = channel_of(expert) {
                    // Cold-served: the staging launch computes the pick on the device,
                    // so the combine needs the REAL weight and no pool job exists. The
                    // channel is below `cold_cap` (itself below the staging depth), so
                    // the cast cannot wrap.
                    plan.slot_ids.push(0);
                    plan.device_weights.push(weight);
                    plan.cold_ids[flat] = channel as i32;
                    self.report.cold_served += 1;
                    self.report.cold_weight_ppm += ppm;
                } else {
                    plan.slot_ids.push(0);
                    plan.device_weights.push(0.0);
                    self.report.cpu_served += 1;
                    self.report.cpu_weight_ppm += ppm;
                    let row_at = row_routes.len();
                    match misses.iter_mut().find(|g| g.expert == expert) {
                        Some(group) => {
                            group.rows.push(row_at);
                            group.row_weights.push(weight);
                        }
                        None => misses.push(ChunkMissGroup {
                            expert,
                            rows: vec![row_at],
                            row_weights: vec![weight],
                        }),
                    }
                }
                route.push(expert);
            }
            row_routes.push(route);
        }
        self.routes[layer] = row_routes.into_iter().flatten().collect();
        let cold = hot
            .iter()
            .enumerate()
            .map(|(channel, &expert)| {
                let ranges = self
                    .locator
                    .ranges(layer, usize::from(expert))
                    .map_err(HostError::Locator)?;
                Ok(ChunkColdStage {
                    expert,
                    channel: channel as u16,
                    gate: ranges.gate,
                    up: ranges.up,
                    down: ranges.down,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((plan, misses, cold))
    }

    /// The checkpoint mapping the cold fill gathers its slices from (the same region the
    /// CPU pool reads; an `Arc` clone, so the fill borrows nothing of the host).
    pub fn cold_region(&self) -> Arc<dyn MappedRegion> {
        Arc::clone(&self.region)
    }

    /// Joins the grouped CPU jobs of a chunk layer and returns the dense [m][hidden]
    /// partial (the M-row twin of finish_layer's return).
    pub fn finish_chunk_layer(&mut self, layer: usize) -> Result<Option<Vec<f32>>, HostError> {
        self.finish_layer(layer)
    }

    /// Starts the grouped pool serving of a chunk layer's misses: one GroupedExpertJob
    /// per DISTINCT missed expert (the locator names its byte ranges; the region, row
    /// list and per-row weights come from the miss group). The caller runs the GPU
    /// stages while the workers serve, then joins via `PendingChunk::finish`.
    pub fn start_chunk_pool(
        &mut self,
        layer: usize,
        misses: &[ChunkMissGroup],
        block: &[f32],
        m: usize,
    ) -> Result<Option<super::cpu_pool::PendingChunk>, HostError> {
        use super::cpu_pool::GroupedExpertJob;
        if misses.is_empty() {
            return Ok(None);
        }
        let jobs: Vec<GroupedExpertJob> = misses
            .iter()
            .map(|group| {
                let ranges = self
                    .locator
                    .ranges(layer, usize::from(group.expert))
                    .map_err(HostError::Locator)?;
                Ok(GroupedExpertJob {
                    region: Arc::clone(&self.region),
                    gate: ranges.gate,
                    up: ranges.up,
                    down: ranges.down,
                    rows: group.rows.clone(),
                    row_weights: group.row_weights.clone(),
                })
            })
            .collect::<Result<_, _>>()?;
        self.pool
            .start_chunk_layer(jobs, block, m)
            .map_err(HostError::Pool)
    }

    /// Accounts for a whole token from the device's route log: the same counters and the same
    /// `routes` per layer as [`Self::begin_layer`] leaves, without starting any CPU job (the device
    /// classified and a server ran the jobs). Every layer is validated first, so a refused log
    /// changes nothing. Refuses while a v1 layer is in flight. A log must hold exactly the config's
    /// `top_k` picks per layer. First caller: the mailbox's CUDA adapter.
    pub fn account_token(&mut self, log: &RouteLog) -> Result<(), HostError> {
        if let Some(open) = &self.in_flight {
            return Err(HostError::Order(format!(
                "account_token while layer {} is in flight",
                open.layer
            )));
        }
        if log.layers != self.cfg.layers || log.top_k != self.cfg.top_k {
            return Err(HostError::BadRoute(format!(
                "a log of {} layers of {} picks for {} layers of top_k {}",
                log.layers, log.top_k, self.cfg.layers, self.cfg.top_k
            )));
        }
        for layer in 0..log.layers {
            let (ids, weights): (Vec<i32>, Vec<f32>) = log.layer(layer).iter().copied().unzip();
            self.check_route(layer, &ids, &weights)?;
        }
        for layer in 0..log.layers {
            let picks = log.layer(layer);
            self.routes[layer] = picks
                .iter()
                .filter(|&&(id, _)| id >= 0)
                .map(|&(id, _)| id as u16)
                .collect();
            for &(id, weight) in picks.iter().filter(|&&(id, _)| id >= 0) {
                let ppm = weight_ppm(weight);
                self.report.routed_weight_ppm += ppm;
                if self.table.lookup(layer, id as u16).is_some() {
                    self.report.bank_hits += 1;
                } else {
                    self.report.cpu_served += 1;
                    self.report.cpu_weight_ppm += ppm;
                }
            }
        }
        Ok(())
    }

    /// The device slot table row of `layer`: for every expert its bank slot while it is published,
    /// else -1 (a miss; never 0, which is a real slot). `Err(BadRoute)` for a layer outside the
    /// model. First caller: the mailbox's CUDA adapter.
    pub fn slot_row(&self, layer: usize) -> Result<Vec<i32>, HostError> {
        if layer >= self.cfg.layers {
            return Err(HostError::BadRoute(format!(
                "layer {layer} is outside 0..{}",
                self.cfg.layers
            )));
        }
        (0..self.cfg.experts)
            .map(|expert| {
                let Some((slot, _)) = self.table.lookup(layer, expert as u16) else {
                    return Ok(-1);
                };
                i32::try_from(slot)
                    .map_err(|_| HostError::Config(format!("slot {slot} does not fit an i32")))
            })
            .collect()
    }

    /// Validates a route: the layer, the list lengths, every expert id (in range, picked once) and
    /// its weight (finite). The router-fault id -1 passes through unchecked; any other negative
    /// id is outside the layer, as in the device classification (`classify_reference`).
    fn check_route(&self, layer: usize, ids: &[i32], weights: &[f32]) -> Result<(), HostError> {
        let bad = |what: String| Err(HostError::BadRoute(what));
        if layer >= self.cfg.layers {
            return bad(format!("layer {layer} is outside 0..{}", self.cfg.layers));
        }
        if ids.len() != weights.len() || ids.len() > self.cfg.top_k {
            return bad(format!(
                "{} ids and {} weights for top_k {}",
                ids.len(),
                weights.len(),
                self.cfg.top_k
            ));
        }
        let mut picks: Vec<i32> = Vec::with_capacity(ids.len());
        for (&id, &weight) in ids.iter().zip(weights) {
            if id == -1 {
                continue;
            }
            if usize::try_from(id).map_or(true, |e| e >= self.cfg.experts) {
                return bad(format!("expert {id} is outside 0..{}", self.cfg.experts));
            }
            if picks.contains(&id) {
                return bad(format!("expert {id} is picked twice in layer {layer}"));
            }
            if !weight.is_finite() {
                return bad(format!("expert {id} has the non-finite weight {weight}"));
            }
            picks.push(id);
        }
        Ok(())
    }

    /// Joins the CPU jobs of `layer` (at most the configured timeout) and returns the
    /// job-order partial sum, or `None` when no pick was served on the CPU. The layer is
    /// closed whatever the outcome, so a failed layer does not block the next one.
    pub fn finish_layer(&mut self, layer: usize) -> Result<Option<Vec<f32>>, HostError> {
        match self.in_flight.take() {
            Some(open) if open.layer == layer => match open.jobs {
                None => Ok(None),
                Some(jobs) => jobs.finish(self.timeout).map(Some).map_err(HostError::Pool),
            },
            Some(open) => {
                let busy = open.layer;
                self.in_flight = Some(open);
                Err(HostError::Order(format!(
                    "finish_layer({layer}) while layer {busy} is in flight"
                )))
            }
            None => Err(HostError::Order(format!(
                "finish_layer({layer}) with no layer begun"
            ))),
        }
    }

    /// Drops what a token that failed left behind: the layer in flight (nobody joins its CPU
    /// jobs, the late results go with their channel), the routes and the counters of the layers
    /// that ran. WHY: a failure between `begin_layer` and `finish_layer` would leave
    /// `in_flight` set, and every later `begin_layer` would refuse with [`HostError::Order`]
    /// until the engine was rebuilt. The ledger, the policy and the poison flag are untouched
    /// (a failed fill stays poisoned: the policy and the ledger still disagree).
    pub fn abort_token(&mut self) {
        self.in_flight = None;
        self.routes = vec![Vec::new(); self.cfg.layers];
        self.report = TokenReport::default();
    }

    /// Ends the token: feeds this token's routes to policy P, runs the swaps it decides
    /// through the residency state machine (unpublish the victim, begin the fill, copy
    /// through `device`, complete the event) and publishes what landed. Returns the
    /// token's counters and starts the next token. A failed fill publishes what was
    /// already filled, poisons the host and returns the error.
    pub fn end_token(&mut self, device: &mut dyn BankDevice) -> Result<TokenReport, HostError> {
        if self.poisoned {
            return Err(HostError::Poisoned);
        }
        if let Some(open) = &self.in_flight {
            return Err(HostError::Order(format!(
                "end_token while layer {} is in flight",
                open.layer
            )));
        }
        let routes = std::mem::replace(&mut self.routes, vec![Vec::new(); self.cfg.layers]);
        self.policy.observe(&routes);
        let swaps = self.policy.boundary();
        // Publish everything a mid-token prefetch queued BEFORE the boundary picks
        // victims: the policy already counts those promotees as residents, so an
        // unpublish of one that is still Filling would refuse (the invariant the
        // original inline-complete fills never had to state).
        self.drain_fills(device)?;
        for swap in &swaps {
            // The policy can select a promotee from an earlier swap in this same
            // boundary as the next victim. Publish any queued fills before trying
            // to unpublish that victim; skipping individual swaps would leave the
            // policy and ledger out of step for longer chains.
            if let Some(evict) = swap.evict
                && self.table.lookup(usize::from(swap.layer), evict).is_none()
            {
                self.drain_fills(device)?;
            }
            if let Err(error) = self.apply_swap(device, swap) {
                self.poisoned = true;
                self.table.publish_ready();
                return Err(error);
            }
        }
        self.drain_fills(device)?;
        let mut report = std::mem::take(&mut self.report);
        report.swaps = u32::try_from(swaps.len()).unwrap_or(u32::MAX);
        let (hits, total) = self.meter.counts();
        report.guess_hits = hits.saturating_sub(self.meter_seen.0);
        report.guess_samples = total.saturating_sub(self.meter_seen.1);
        self.meter_seen = (hits, total);
        Ok(report)
    }

    /// Fills the banks before the first token: for `e` in `0..min(slots, experts)` expert `e` goes
    /// into a slot of every layer, each fill through the full residency state machine, then
    /// everything is published at once. Returns the number of fills. WHY: a gate that compares
    /// the tiered step with the plain step bit for bit needs every expert resident from the first
    /// token, and a start can skip the cold start. A failed fill poisons the host as in
    /// `end_token`.
    pub fn preload_all(&mut self, device: &mut dyn BankDevice) -> Result<usize, HostError> {
        if self.poisoned {
            return Err(HostError::Poisoned);
        }
        if let Some(open) = &self.in_flight {
            return Err(HostError::Order(format!(
                "preload_all while layer {} is in flight",
                open.layer
            )));
        }
        let per_layer = self.cfg.bank_slots_per_layer.min(self.cfg.experts);
        let mut fills = 0;
        for layer in 0..self.cfg.layers {
            for expert in 0..per_layer {
                let (layer16, expert16) = (u16::try_from(layer), u16::try_from(expert));
                let (Ok(layer16), Ok(expert16)) = (layer16, expert16) else {
                    return Err(HostError::Config(String::from("preload ids exceed u16")));
                };
                self.policy
                    .preload(layer, expert16)
                    .map_err(HostError::Config)?;
                let swap = Swap {
                    layer: layer16,
                    promote: expert16,
                    evict: None,
                };
                if let Err(error) = self.apply_swap(device, &swap) {
                    self.poisoned = true;
                    self.table.publish_ready();
                    return Err(error);
                }
                fills += 1;
            }
        }
        self.drain_fills(device)?;
        Ok(fills)
    }

    /// Arms the router-guess prefetch: spawns the advice worker (a cheap `Arc` clone of
    /// `guess`) that ranks `k` experts of the layer AFTER the one whose residual the
    /// host hands it. Advice is best-effort by design: nothing here can change a
    /// token's result, only which experts sit in the banks of later tokens.
    pub fn set_guess(&mut self, guess: RouterGuess, k: usize) {
        let (tx_in, rx_in) = mpsc::channel::<(usize, Vec<f32>)>();
        let (tx_out, rx_out) = mpsc::channel::<(usize, Vec<u16>)>();
        let worker_guess = guess.clone();
        let handle = std::thread::Builder::new()
            .name(String::from("tiered-router-guess"))
            .spawn(move || {
                while let Ok((layer, resid)) = rx_in.recv() {
                    let target = layer + 1;
                    if target < worker_guess.layers()
                        && let Ok(advice) = worker_guess.top_k(target, &resid, k)
                    {
                        let _ = tx_out.send((target, advice));
                    }
                }
            })
            .expect("spawn the router-guess worker");
        self.guess = Some(GuessWorker {
            k,
            to_worker: tx_in,
            from_worker: rx_out,
            handle: Some(handle),
        });
    }

    /// Whether the router-guess prefetch is armed (an armed worker exists).
    pub fn has_guess(&self) -> bool {
        self.guess.is_some()
    }

    /// The advice the latest `begin_layer` of its target layer consumed, if any: the
    /// runtime takes it to issue the prefetch fills. One-shot.
    pub fn take_arrived_advice(&mut self) -> Option<(usize, Vec<u16>)> {
        self.arrived.take()
    }

    /// Issues the prefetch fills of `advice` for `layer` through the same residency state
    /// machine as the boundary swaps: free-slot promotions are free, full-bank promotions
    /// only when the advised expert outscores the lowest resident of the layer, and at most
    /// [`GUESS_EVICT_CAP`] evicting promotions per call — a guess has no routed evidence,
    /// so it must not be able to thrash a whole bank. Everything publishes at the token
    /// boundary like every other fill; a failed fill poisons the host as in `end_token`.
    pub fn prefetch(
        &mut self,
        device: &mut dyn BankDevice,
        layer: usize,
        advice: &[u16],
    ) -> Result<(), HostError> {
        if self.poisoned {
            return Err(HostError::Poisoned);
        }
        if layer >= self.cfg.layers {
            return Err(HostError::BadRoute(format!(
                "prefetch layer {layer} is outside 0..{}",
                self.cfg.layers
            )));
        }
        let mut evictions = 0;
        for &expert in advice {
            let Some(swap) = self.policy.advise(layer, expert) else {
                continue;
            };
            if let Some(evict) = swap.evict
                && self.table.lookup(layer, evict).is_none()
            {
                // The victim's slot is not Published — its own fill is still in
                // flight (a mid-token promotion of an earlier advice). Skip this
                // promotion entirely: advice is best-effort by contract, and racing
                // the state machine is the one thing it must never do.
                self.policy.unadvise(layer, expert, swap.evict);
                continue;
            }
            if swap.evict.is_some() {
                if evictions >= GUESS_EVICT_CAP {
                    // Undo the policy-side mark: this promotion is not happening.
                    self.policy.unadvise(layer, expert, swap.evict);
                    continue;
                }
                evictions += 1;
                self.report.guess_evicts += 1;
            }
            if let Err(error) = self.apply_swap(device, &swap) {
                self.poisoned = true;
                self.table.publish_ready();
                return Err(error);
            }
            self.report.guess_fills += 1;
        }
        Ok(())
    }

    fn apply_swap(&mut self, device: &mut dyn BankDevice, swap: &Swap) -> Result<(), HostError> {
        let layer = usize::from(swap.layer);
        if let Some(evict) = swap.evict {
            self.table
                .unpublish(layer, evict)
                .map_err(HostError::Residency)?;
        }
        let event = self.next_event;
        self.next_event += 1;
        let slot = self
            .table
            .begin_fill(layer, swap.promote, event, FILL_DEADLINE_NS)
            .map_err(HostError::Residency)?;
        let ranges = self
            .locator
            .ranges(layer, usize::from(swap.promote))
            .map_err(HostError::Locator)?;
        if let Err(error) = device.fill_slot(layer, slot, &ranges, &self.region) {
            // The failed slot poisons, but the fills QUEUED BEFORE it may have landed:
            // complete what a wait can still salvage, so only the failure is poisoned.
            let _ = device.wait_fills();
            for event in std::mem::take(&mut self.pending_fills) {
                let _ = self.table.event_complete(event);
            }
            self.abandon_fills()?;
            return Err(error);
        }
        // The fill is queued, not necessarily landed: its event completes at the next
        // drain (the token boundary), which is also the only publisher.
        self.pending_fills.push(event);
        Ok(())
    }

    /// Waits the queued fills, completes their events and PUBLISHES what landed.
    /// Publication still never happens inside a layer — this runs at the token
    /// boundary (and after preload) — but it must also run BETWEEN the boundary's
    /// two halves: a victim the boundary is about to unpublish may be a mid-token
    /// promotee whose slot is Ready (event complete) but not yet Published, and
    /// `unpublish` refuses anything but a Published slot. A failed wait poisons the
    /// host exactly like a failed fill.
    fn drain_fills(&mut self, device: &mut dyn BankDevice) -> Result<(), HostError> {
        if let Err(error) = device.wait_fills() {
            self.poisoned = true;
            let _ = self.abandon_fills();
            self.table.publish_ready();
            return Err(error);
        }
        for event in std::mem::take(&mut self.pending_fills) {
            if let Err(error) = self.table.event_complete(event) {
                self.poisoned = true;
                self.table.publish_ready();
                return Err(HostError::Residency(error));
            }
        }
        self.table.publish_ready();
        Ok(())
    }

    /// Poisons and frees every fill still in flight (only the one that just failed).
    fn abandon_fills(&mut self) -> Result<(), HostError> {
        for (layer, slot) in self.table.expire(u64::MAX) {
            self.table
                .reclaim_poisoned(layer, slot)
                .map_err(HostError::Residency)?;
        }
        Ok(())
    }

    pub fn is_resident(&self, layer: usize, expert: u16) -> bool {
        self.table.lookup(layer, expert).is_some()
    }

    pub fn check_consistency(&self) -> Result<(), HostError> {
        self.table.check_partition().map_err(HostError::Residency)?;
        for layer in 0..self.cfg.layers {
            for expert in 0..self.cfg.experts {
                let id = u16::try_from(expert).unwrap_or(u16::MAX);
                if self.policy.is_resident(layer, expert) != self.is_resident(layer, id) {
                    return Err(HostError::Inconsistent(format!(
                        "layer {layer} expert {expert}: the policy and the ledger disagree"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::tiered::classify::{ROUTE_INVALID, classify_reference};
    use crate::family::qwen36::tiered::cpu_pool::ExpertKernel;
    use crate::family::qwen36::tiered::locator::ExpertTensorNames;
    use crate::family::qwen36::tiered::locator::tests::{Lcg, expert_file, parse};
    use crate::kernels::expert_cpu::{ExpertShape, ExpertSlices};
    use crate::kernels::expert_cpu_q8k::{DotBackend, ExpertScratchQ8k, expert_forward_q8k};

    const SHAPE: ExpertShape = ExpertShape {
        hidden: 256,
        inter: 256,
    };
    const LAYERS: usize = 3;
    const EXPERTS: usize = 8;
    const TOP_K: usize = 4;
    const SLOTS: usize = 3;

    fn config_with(slots: usize) -> TieredConfig {
        config_with_rate(slots, 0.25)
    }

    fn config_with_rate(slots: usize, rate: f64) -> TieredConfig {
        TieredConfig {
            layers: LAYERS,
            experts: EXPERTS,
            top_k: TOP_K,
            bank_slots_per_layer: slots,
            mode: PolicyMode::Adaptive,
            swaps_per_token: rate,
            decay: 0.999,
        }
    }

    fn config() -> TieredConfig {
        config_with(SLOTS)
    }

    /// A host over a synthetic GGUF, a reference locator over the same file and the row.
    struct Fixture {
        host: TieredHost,
        reference: ExpertLocator,
        region: Arc<dyn MappedRegion>,
        hidden: Vec<f32>,
    }

    fn fixture_with_rate(timeout: Duration, slots: usize, rate: f64) -> Fixture {
        let mut rng = Lcg(0xD4D4_0000_5EED_0001);
        let ((bytes, _), _) = expert_file(&mut rng, LAYERS, EXPERTS, SHAPE);
        let header = parse(&bytes);
        let names = ExpertTensorNames::llama_cpp();
        let make = || ExpertLocator::new(&header, &names, LAYERS, EXPERTS, SHAPE).unwrap();
        let region: Arc<dyn MappedRegion> = Arc::new(bytes);
        let kernel = ExpertKernel::Q8k(DotBackend::scalar());
        let pool = CpuPool::with_kernel(2, SHAPE, kernel).unwrap();
        let host = TieredHost::new(
            config_with_rate(slots, rate),
            make(),
            Arc::clone(&region),
            pool,
            timeout,
        )
        .unwrap();
        let hidden = (0..SHAPE.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        Fixture {
            host,
            reference: make(),
            region,
            hidden,
        }
    }

    fn fixture_with_slots(timeout: Duration, slots: usize) -> Fixture {
        fixture_with_rate(timeout, slots, 0.25)
    }

    fn fixture_with(timeout: Duration) -> Fixture {
        fixture_with_rate(timeout, SLOTS, 0.25)
    }

    fn fixture() -> Fixture {
        fixture_with(Duration::from_secs(60))
    }

    /// Publishes `expert` of `layer` straight through the ledger (the boundary path is
    /// tested with `end_token`).
    fn install(host: &mut TieredHost, layer: usize, expert: u16, event: u64) -> usize {
        let slot = host.table.begin_fill(layer, expert, event, 1).unwrap();
        host.table.event_complete(event).unwrap();
        host.table.publish_ready();
        slot
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn classification_covers_resident_missed_fault_and_mixed_picks() {
        let mut f = fixture();
        let s2 = install(&mut f.host, 1, 2, 1);
        let s5 = install(&mut f.host, 1, 5, 2);
        let plan = f
            .host
            .begin_layer(
                1,
                &[2, 3, -1, 5],
                &[0.4, 0.3, 0.0, 0.2],
                &f.hidden,
                &f.hidden,
            )
            .unwrap();
        assert_eq!(plan.slot_ids, vec![s2 as i32, 0, -1, s5 as i32]);
        assert_eq!(plan.device_weights, vec![0.4, 0.0, 0.0, 0.2]);
        assert!(f.host.finish_layer(1).unwrap().is_some());
        let plan = f
            .host
            .begin_layer(1, &[5, 2], &[0.6, 0.4], &f.hidden, &f.hidden)
            .unwrap();
        assert_eq!(plan.slot_ids, vec![s5 as i32, s2 as i32]);
        assert_eq!(plan.device_weights, vec![0.6, 0.4]);
        assert_eq!(f.host.finish_layer(1), Ok(None));
        let plan = f
            .host
            .begin_layer(0, &[-1, -1, -1, -1], &[0.0; 4], &f.hidden, &f.hidden)
            .unwrap();
        assert_eq!(plan.slot_ids, vec![-1; 4]);
        assert_eq!(f.host.finish_layer(0), Ok(None));
        let plan = f
            .host
            .begin_layer(2, &[0, 1], &[0.5, 0.5], &f.hidden, &f.hidden)
            .unwrap();
        assert_eq!(plan.slot_ids, vec![0, 0]);
        assert_eq!(plan.device_weights, vec![0.0, 0.0]);
        assert!(f.host.finish_layer(2).unwrap().is_some());
        assert_eq!((f.host.report.bank_hits, f.host.report.cpu_served), (4, 3));
        // Weight mass: the mixed layer routed 0.4 + 0.3 + 0.2 (0.3 on the CPU), the resident layer 0.6 + 0.4, the
        // all-CPU layer 0.5 + 0.5, the faulted layer none.
        assert_eq!(f.host.report.routed_weight_ppm, 2_900_000);
        assert_eq!(f.host.report.cpu_weight_ppm, 1_300_000);
    }

    /// The three-way chunk classify: the hottest `cold_cap` distinct missed experts take
    /// staging channels WITH their real router weights (no pool job), the rest stay
    /// weight-0.0 overflow groups, and cold_cap 0 keeps today's two-way plan.
    #[test]
    fn chunk_classify_cuts_the_hottest_misses_into_cold_channels() {
        let mut f = fixture();
        let s2 = install(&mut f.host, 1, 2, 1);
        let row = |ids: &[i32], weights: &[f32]| ChunkRouteRow {
            ids: ids.to_vec(),
            weights: weights.to_vec(),
        };
        let rows = [
            row(&[2, 7, 3, 5], &[0.30, 0.20, 0.10, 0.02]),
            row(&[3, 7, 2, 0], &[0.25, 0.15, 0.05, 0.01]),
            row(&[3, 6, 2, 1], &[0.10, 0.05, 0.04, 0.01]),
        ];
        // Miss counts: expert 3 three picks, expert 7 two, experts 0/1/5/6 one each —
        // the cap of 2 stages 3 and 7, everything else overflows to the pool.
        let (plan, overflow, cold) = f.host.begin_chunk_layer(1, &rows, 2).unwrap();
        assert_eq!(
            plan.slot_ids,
            vec![s2 as i32, 0, 0, 0, 0, 0, s2 as i32, 0, 0, 0, s2 as i32, 0]
        );
        // Cold picks keep their REAL weights (the combine multiplies them); the
        // overflow picks keep the 0.0 dummy of the CPU path; hits keep theirs.
        assert_eq!(
            plan.device_weights,
            vec![
                0.30, 0.20, 0.10, 0.0, 0.25, 0.15, 0.05, 0.0, 0.10, 0.0, 0.04, 0.0
            ]
        );
        assert_eq!(
            plan.cold_ids,
            vec![-1, 1, 0, -1, 0, 1, -1, -1, 0, -1, -1, -1]
        );
        // The overflow groups: one per distinct missed expert, its rows and weights.
        let experts: Vec<u16> = {
            let mut e: Vec<u16> = overflow.iter().map(|g| g.expert).collect();
            e.sort();
            e
        };
        assert_eq!(experts, vec![0, 1, 5, 6]);
        let by_expert = |e: u16| overflow.iter().find(|g| g.expert == e).unwrap();
        assert_eq!(by_expert(5).rows, vec![0]);
        assert_eq!(by_expert(5).row_weights, vec![0.02]);
        assert_eq!(by_expert(6).rows, vec![2]);
        // The staged list: hottest first, channels in that order, and the ranges the
        // locator names for each expert.
        let expect = |expert: usize| f.reference.ranges(1, expert).unwrap();
        assert_eq!(cold.len(), 2);
        assert_eq!(cold[0].expert, 3);
        assert_eq!(cold[0].channel, 0);
        assert_eq!(cold[0].gate, expect(3).gate);
        assert_eq!(cold[1].expert, 7);
        assert_eq!(cold[1].channel, 1);
        assert_eq!(cold[1].down, expect(7).down);

        // The report separates the three outcomes — hits, staged serves, pool jobs —
        // with the mass in millionths per pick, the decode classify's accounting, so a
        // chunk piece's tiered summary can never read as an unwired counter.
        assert_eq!(
            (
                f.host.report.bank_hits,
                f.host.report.cold_served,
                f.host.report.cpu_served
            ),
            (3, 5, 4)
        );
        assert_eq!(f.host.report.routed_weight_ppm, 1_280_000);
        assert_eq!(f.host.report.cold_weight_ppm, 800_000);
        assert_eq!(f.host.report.cpu_weight_ppm, 90_000);

        // cold_cap 0: every miss overflows, the plan matches today's two-way form.
        let (plan, overflow, cold) = f.host.begin_chunk_layer(1, &rows, 0).unwrap();
        assert!(plan.cold_ids.iter().all(|&c| c == -1));
        assert_eq!(overflow.len(), 6);
        assert!(cold.is_empty());
    }

    #[test]
    fn the_cpu_partial_is_the_job_order_sum_over_the_locator_ranges_of_the_missed_experts() {
        let mut f = fixture();
        install(&mut f.host, 1, 6, 1);
        let ids = [3, 6, 1];
        let weights = [0.25f32, 0.5, 0.125];
        f.host
            .begin_layer(1, &ids, &weights, &f.hidden, &f.hidden)
            .unwrap();
        let got = f.host.finish_layer(1).unwrap().unwrap();
        // Reference: expert_forward_q8k over the slices the locator names, experts 3
        // and 1 (6 is resident) in job order, weight-scaled, from +0.0.
        let bytes = f.region.bytes();
        let mut scratch = ExpertScratchQ8k::new(SHAPE);
        let mut out = vec![0.0f32; SHAPE.hidden];
        let mut want = vec![0.0f32; SHAPE.hidden];
        for (expert, weight) in [(3usize, 0.25f32), (1, 0.125)] {
            let r = f.reference.ranges(1, expert).unwrap();
            let slices = ExpertSlices {
                gate: &bytes[r.gate],
                up: &bytes[r.up],
                down: &bytes[r.down],
            };
            let backend = DotBackend::scalar();
            expert_forward_q8k(backend, SHAPE, &slices, &f.hidden, &mut scratch, &mut out).unwrap();
            for (s, v) in want.iter_mut().zip(&out) {
                *s += weight * v;
            }
        }
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn bad_routes_start_nothing() {
        let mut f = fixture();
        let h = f.hidden.clone();
        let cases: [(usize, Vec<i32>, Vec<f32>); 6] = [
            (LAYERS, vec![1], vec![0.5]),
            (0, vec![EXPERTS as i32], vec![0.5]),
            (0, vec![2, 2], vec![0.5, 0.5]),
            (0, vec![1], vec![f32::NAN]),
            (0, vec![1, 2], vec![0.5]),
            (0, vec![0, 1, 2, 3, 4], vec![0.2; 5]),
        ];
        for (layer, ids, weights) in cases {
            let result = f.host.begin_layer(layer, &ids, &weights, &h, &h);
            assert!(matches!(result, Err(HostError::BadRoute(_))), "{ids:?}");
            assert!(matches!(
                f.host.finish_layer(layer.min(LAYERS - 1)),
                Err(HostError::Order(_))
            ));
        }
        assert_eq!(f.host.report, TokenReport::default());
        assert!(f.host.begin_layer(0, &[1], &[0.5], &h, &h).is_ok());
    }

    #[test]
    fn finish_layer_out_of_order_is_refused_and_the_layer_stays_in_flight() {
        let mut f = fixture();
        f.host
            .begin_layer(1, &[3], &[0.5], &f.hidden, &f.hidden)
            .unwrap();
        assert!(matches!(
            f.host.begin_layer(2, &[3], &[0.5], &f.hidden, &f.hidden),
            Err(HostError::Order(_))
        ));
        assert!(matches!(f.host.finish_layer(2), Err(HostError::Order(_))));
        assert!(f.host.finish_layer(1).unwrap().is_some());
        assert!(matches!(f.host.finish_layer(1), Err(HostError::Order(_))));
    }

    #[test]
    fn a_pool_timeout_is_a_typed_error_and_the_host_stays_usable() {
        // A zero budget always times out (the deadline is checked first); the next
        // layer must still begin and finish.
        let mut f = fixture_with(Duration::ZERO);
        f.host
            .begin_layer(1, &[3, 4], &[0.5, 0.5], &f.hidden, &f.hidden)
            .unwrap();
        let err = f.host.finish_layer(1).unwrap_err();
        assert!(
            matches!(err, HostError::Pool(PoolError::Timeout { .. })),
            "{err:?}"
        );
        assert!(f.host.in_flight.is_none());
        f.host
            .begin_layer(2, &[1], &[0.5], &f.hidden, &f.hidden)
            .unwrap();
        assert!(f.host.finish_layer(2).is_err());
        f.host
            .begin_layer(0, &[-1], &[0.0], &f.hidden, &f.hidden)
            .unwrap();
        assert_eq!(f.host.finish_layer(0), Ok(None));
    }

    #[test]
    fn new_refuses_a_config_a_locator_and_a_pool_that_disagree() {
        let mut rng = Lcg(0xD4D4_0000_5EED_0002);
        let ((bytes, _), _) = expert_file(&mut rng, LAYERS, EXPERTS, SHAPE);
        let header = parse(&bytes);
        let names = ExpertTensorNames::llama_cpp();
        let region: Arc<dyn MappedRegion> = Arc::new(bytes);
        let locator = || ExpertLocator::new(&header, &names, LAYERS, EXPERTS, SHAPE).unwrap();
        let pool = |shape| CpuPool::new(1, shape).unwrap();
        let build = |cfg: TieredConfig, shape| {
            TieredHost::new(
                cfg,
                locator(),
                Arc::clone(&region),
                pool(shape),
                Duration::ZERO,
            )
        };
        assert!(build(config(), SHAPE).is_ok());
        let other = ExpertShape {
            hidden: 256,
            inter: 512,
        };
        let refused = [
            build(config(), other),
            build(
                TieredConfig {
                    layers: 2,
                    ..config()
                },
                SHAPE,
            ),
            build(
                TieredConfig {
                    experts: 16,
                    ..config()
                },
                SHAPE,
            ),
            build(
                TieredConfig {
                    top_k: 0,
                    bank_slots_per_layer: 0,
                    ..config()
                },
                SHAPE,
            ),
            build(
                TieredConfig {
                    mode: PolicyMode::Static,
                    ..config()
                },
                SHAPE,
            ),
        ];
        for result in refused {
            assert!(matches!(result.err(), Some(HostError::Config(_))));
        }
    }

    #[test]
    fn host_takes_no_device_types() {
        let source = include_str!("host.rs");
        let production = source.split("mod tests").next().unwrap_or_default();
        for needle in [["kernels", "::cuda"].concat(), ["Cuda", "Context"].concat()] {
            assert!(!production.contains(&needle), "{needle}");
        }
    }

    #[derive(Default)]
    struct MockDevice {
        fills: Vec<(usize, usize, ExpertRanges)>,
        fail_on: Option<usize>,
        fail_wait: bool,
        fail_wait_at: Option<usize>,
        waits: usize,
    }

    impl BankDevice for MockDevice {
        fn fill_slot(
            &mut self,
            layer: usize,
            slot: usize,
            ranges: &ExpertRanges,
            _region: &Arc<dyn MappedRegion>,
        ) -> Result<(), HostError> {
            if self.fail_on == Some(self.fills.len()) {
                return Err(HostError::Device(String::from("mock fill failure")));
            }
            self.fills.push((layer, slot, ranges.clone()));
            Ok(())
        }

        fn wait_fills(&mut self) -> Result<(), HostError> {
            let call = self.waits;
            self.waits += 1;
            if self.fail_wait || self.fail_wait_at == Some(call) {
                return Err(HostError::Device(String::from("mock wait failure")));
            }
            Ok(())
        }
    }

    /// The `(layer, slot, expert)` rows of the fills, the expert found by its ranges.
    fn filled(device: &MockDevice, reference: &ExpertLocator) -> Vec<(usize, usize, usize)> {
        let expert_of = |layer: usize, ranges: &ExpertRanges| {
            (0..EXPERTS)
                .find(|&e| reference.ranges(layer, e).as_ref() == Ok(ranges))
                .expect("the fill names a locator range")
        };
        let row = |(layer, slot, ranges): &(usize, usize, ExpertRanges)| {
            (*layer, *slot, expert_of(*layer, ranges))
        };
        device.fills.iter().map(row).collect()
    }

    fn run_token(
        f: &mut Fixture,
        picks: &[Vec<i32>],
        device: &mut MockDevice,
    ) -> Result<TokenReport, HostError> {
        for (layer, ids) in picks.iter().enumerate() {
            let weights = vec![1.0 / ids.len() as f32; ids.len()];
            f.host
                .begin_layer(layer, ids, &weights, &f.hidden, &f.hidden)?;
            f.host.finish_layer(layer)?;
        }
        f.host.end_token(device)
    }

    #[test]
    fn warm_up_fills_the_first_routed_experts_and_publishes_them_at_the_boundary() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        let picks = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![1, 3, 5, 7]];
        let report = run_token(&mut f, &picks, &mut device).unwrap();
        // Three slots per layer: the first three routed experts of each layer fill, in
        // route order; the fourth stays on the CPU.
        let want = [
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 0, 4),
            (1, 1, 5),
            (1, 2, 6),
        ];
        assert_eq!(filled(&device, &f.reference)[..6], want);
        assert_eq!(device.fills.len(), 9);
        assert_eq!(
            (report.bank_hits, report.cpu_served, report.swaps),
            (0, 12, 9)
        );
        f.host.check_consistency().unwrap();
        assert!(f.host.is_resident(0, 2) && !f.host.is_resident(0, 3));
        let report = run_token(&mut f, &picks, &mut device).unwrap();
        assert_eq!((report.bank_hits, report.cpu_served), (9, 3));
        f.host.check_consistency().unwrap();
    }

    #[test]
    fn end_token_swaps_respect_the_credit_and_keep_the_ledger_and_policy_in_step() {
        // A skewed deterministic stream over 600 tokens: low expert ids are popular.
        let mut f = fixture();
        let mut rng = Lcg(0xD4D4_0000_5EED_0003);
        let mut device = MockDevice::default();
        let (mut swaps, mut served) = (0u32, 0u32);
        for _ in 0..600 {
            let mut picks = Vec::new();
            for _ in 0..LAYERS {
                let mut ids: Vec<i32> = Vec::new();
                while ids.len() < TOP_K {
                    let e = (rng.next() % 8).min(rng.next() % 8) as i32;
                    if !ids.contains(&e) {
                        ids.push(e);
                    }
                }
                picks.push(ids);
            }
            let report = run_token(&mut f, &picks, &mut device).unwrap();
            swaps += report.swaps;
            served += report.cpu_served;
            f.host.check_consistency().unwrap();
        }
        // Warm-up fills (3 slots x 3 layers) plus floor(0.25 x 600) credit swaps.
        assert!(swaps > 9 && swaps <= 9 + 150, "{swaps} swaps");
        assert!(served < 600 * (LAYERS * TOP_K) as u32);
        for layer in 0..LAYERS {
            let residents = (0..EXPERTS as u16).filter(|&e| f.host.is_resident(layer, e));
            assert_eq!(residents.count(), SLOTS, "layer {layer} is full");
        }
        for (layer, slot, _) in filled(&device, &f.reference) {
            assert!(layer < LAYERS && slot < SLOTS);
        }
    }

    #[test]
    fn a_failing_fill_keeps_the_ledger_consistent_and_poisons_only_the_boundary() {
        let mut f = fixture();
        let mut device = MockDevice {
            fail_on: Some(1),
            ..MockDevice::default()
        };
        let picks = vec![vec![0, 1], vec![2], vec![3]];
        let err = run_token(&mut f, &picks, &mut device).unwrap_err();
        assert_eq!(err, HostError::Device(String::from("mock fill failure")));
        f.host.table.check_partition().unwrap();
        assert!(f.host.is_resident(0, 0) && !f.host.is_resident(0, 1));
        assert!(matches!(
            f.host.check_consistency(),
            Err(HostError::Inconsistent(_))
        ));
        assert_eq!(f.host.end_token(&mut device), Err(HostError::Poisoned));
        let plan = f
            .host
            .begin_layer(0, &[0, 1], &[0.5, 0.5], &f.hidden, &f.hidden)
            .unwrap();
        assert_eq!(plan.device_weights, vec![0.5, 0.0]);
        assert!(f.host.finish_layer(0).unwrap().is_some());
    }

    #[test]
    fn end_token_refuses_while_a_layer_is_in_flight() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        f.host
            .begin_layer(0, &[1], &[0.5], &f.hidden, &f.hidden)
            .unwrap();
        assert!(matches!(
            f.host.end_token(&mut device),
            Err(HostError::Order(_))
        ));
        f.host.finish_layer(0).unwrap();
        assert!(f.host.end_token(&mut device).is_ok());
    }

    #[test]
    fn an_aborted_token_frees_the_layer_in_flight_and_drops_its_counters() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        f.host
            .begin_layer(0, &[1], &[0.5], &f.hidden, &f.hidden)
            .unwrap();
        // The failure between begin_layer and finish_layer: without the abort every later
        // layer is refused for good.
        assert!(matches!(
            f.host.begin_layer(1, &[2], &[0.5], &f.hidden, &f.hidden),
            Err(HostError::Order(_))
        ));
        f.host.abort_token();
        f.host
            .begin_layer(1, &[2], &[0.5], &f.hidden, &f.hidden)
            .unwrap();
        f.host.finish_layer(1).unwrap();
        let report = f.host.end_token(&mut device).unwrap();
        assert_eq!(
            (report.cpu_served, report.bank_hits),
            (1, 0),
            "only the layer after the abort is counted"
        );
    }

    #[test]
    fn an_aborted_token_drops_its_routes_before_the_policy_sees_them() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        f.host
            .begin_layer(0, &[1, 2], &[0.5, 0.5], &f.hidden, &f.hidden)
            .unwrap();
        f.host.abort_token();
        // The next token reaches only layers 1 and 2: a stale route of layer 0 would reach the
        // policy at the boundary, and the policy would fill layer 0 for it.
        for layer in 1..LAYERS {
            f.host
                .begin_layer(layer, &[3], &[1.0], &f.hidden, &f.hidden)
                .unwrap();
            f.host.finish_layer(layer).unwrap();
        }
        f.host.end_token(&mut device).unwrap();
        assert!(!device.fills.is_empty(), "the routed layers were filled");
        assert!(
            device.fills.iter().all(|(layer, _, _)| *layer != 0),
            "layer 0 was never routed after the abort"
        );
    }

    #[test]
    fn an_abort_leaves_the_poison_of_a_failed_fill_in_place() {
        let mut f = fixture();
        let mut device = MockDevice {
            fail_on: Some(0),
            ..MockDevice::default()
        };
        run_token(&mut f, &[vec![0], vec![1]], &mut device).unwrap_err();
        f.host.abort_token();
        assert_eq!(f.host.end_token(&mut device), Err(HostError::Poisoned));
    }

    /// The seeded picks of one token, per layer: `TOP_K` distinct experts with fixed weights, a
    /// router fault (every id -1, weight 0.0) in layer 1 of every seventh token, and a mixed list
    /// (a fault id between valid picks) in layer 2 of every fifth token. The design's 2,000-token
    /// replay of the OFFT trace is not in the repo: these are seeded stand-ins.
    fn seeded_routes(rng: &mut Lcg, token: usize) -> Vec<Vec<(i32, f32)>> {
        const WEIGHTS: [f32; TOP_K] = [0.4, 0.3, 0.2, 0.1];
        (0..LAYERS)
            .map(|layer| {
                if layer == 1 && token % 7 == 6 {
                    return vec![(-1, 0.0); TOP_K];
                }
                if layer == 2 && token % 5 == 3 {
                    return vec![(2, 0.4), (-1, 0.0), (5, 0.2), (3, 0.1)];
                }
                let mut ids: Vec<i32> = Vec::new();
                while ids.len() < TOP_K {
                    let id = (rng.next() % EXPERTS as u64) as i32;
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                ids.into_iter().zip(WEIGHTS).collect()
            })
            .collect()
    }

    #[test]
    fn account_token_equals_the_per_layer_counters() {
        let (mut v1, mut logged) = (fixture(), fixture());
        let (mut dev_v1, mut dev_logged) = (MockDevice::default(), MockDevice::default());
        let mut rng = Lcg(0x7CA9_0004_0000_0001);
        for token in 0..120 {
            let routes = seeded_routes(&mut rng, token);
            for (layer, picks) in routes.iter().enumerate() {
                let (ids, weights): (Vec<i32>, Vec<f32>) = picks.iter().copied().unzip();
                v1.host
                    .begin_layer(layer, &ids, &weights, &v1.hidden, &v1.hidden)
                    .unwrap();
                v1.host.finish_layer(layer).unwrap();
            }
            let log = RouteLog::new(LAYERS, TOP_K, routes.concat()).unwrap();
            logged.host.account_token(&log).unwrap();
            let want = v1.host.end_token(&mut dev_v1).unwrap();
            let got = logged.host.end_token(&mut dev_logged).unwrap();
            assert_eq!(got, want, "token {token}");
            for layer in 0..LAYERS {
                assert_eq!(
                    logged.host.slot_row(layer),
                    v1.host.slot_row(layer),
                    "token {token} layer {layer}"
                );
            }
        }
        assert_eq!(
            dev_logged.fills, dev_v1.fills,
            "the same swaps in the same order"
        );
        assert!(!dev_logged.fills.is_empty(), "the run really swapped");
        logged.host.check_consistency().unwrap();
    }

    #[test]
    fn a_refused_log_changes_nothing_and_a_good_one_counts_once() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        // A pick twice in layer 2 is refused as a whole: layers 0 and 1 are not counted either.
        let mut picks: Vec<(i32, f32)> = vec![(1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25)]
            .into_iter()
            .cycle()
            .take(LAYERS * TOP_K)
            .collect();
        picks[2 * TOP_K + 1] = (1, 0.25);
        let bad = RouteLog::new(LAYERS, TOP_K, picks.clone()).unwrap();
        assert!(matches!(
            f.host.account_token(&bad),
            Err(HostError::BadRoute(_))
        ));
        picks[2 * TOP_K + 1] = (2, 0.25);
        let good = RouteLog::new(LAYERS, TOP_K, picks).unwrap();
        f.host.account_token(&good).unwrap();
        let report = f.host.end_token(&mut device).unwrap();
        assert_eq!(
            report.cpu_served,
            (LAYERS * TOP_K) as u32,
            "nothing is resident yet"
        );
        assert_eq!(report.routed_weight_ppm, 250_000 * (LAYERS * TOP_K) as u64);
        assert_eq!(
            report.cpu_weight_ppm, report.routed_weight_ppm,
            "all of it on the CPU"
        );
    }

    #[test]
    fn begin_layer_and_the_log_refuse_the_same_routes() {
        let good = [(1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25)];
        let bad: [(&str, [(i32, f32); TOP_K]); 4] = [
            ("pick twice", [(1, 0.25), (2, 0.25), (1, 0.25), (3, 0.25)]),
            (
                "id below the fault id",
                [(1, 0.25), (2, 0.25), (-2, 0.25), (3, 0.25)],
            ),
            (
                "expert past the layer",
                [(1, 0.25), (2, 0.25), (EXPERTS as i32, 0.25), (3, 0.25)],
            ),
            (
                "non-finite weight",
                [(1, 0.25), (2, f32::NAN), (4, 0.25), (3, 0.25)],
            ),
        ];
        for (what, picks) in bad {
            let mut f = fixture();
            let (ids, weights): (Vec<i32>, Vec<f32>) = picks.iter().copied().unzip();
            assert!(
                matches!(
                    f.host.begin_layer(2, &ids, &weights, &f.hidden, &f.hidden),
                    Err(HostError::BadRoute(_))
                ),
                "begin_layer: {what}"
            );
            let row = f.host.slot_row(2).unwrap();
            let classified = classify_reference(&row, EXPERTS, &ids, &weights).unwrap();
            assert_eq!(classified.word, ROUTE_INVALID, "classify_reference: {what}");
            let all = [good.to_vec(), good.to_vec(), picks.to_vec()].concat();
            let log = RouteLog::new(LAYERS, TOP_K, all).unwrap();
            assert!(
                matches!(f.host.account_token(&log), Err(HostError::BadRoute(_))),
                "log: {what}"
            );
        }
    }

    #[test]
    fn classify_reference_equals_begin_layer_plans() {
        // The oracle of the device classification against the host classification it replaces:
        // seeded single-layer routes over a ledger with 4 residents and 4 missing experts of 8
        // (so a route can be all hits, all misses or mixed), with whole-layer router faults and
        // mixed lists (a fault id among valid picks). The slot ids, the weights (by bits) and the
        // miss list (ids in pick order with their weights) must equal what begin_layer returns.
        // 3,000 routes, not the design's 10,000: every begin_layer miss starts a real CPU expert
        // job, and the counted categories below each occur dozens of times. The design's replay
        // of the OFFT trace is not in the repo.
        const RESIDENT: [u16; 4] = [0, 2, 3, 5];
        let mut f = fixture_with_slots(Duration::from_secs(60), EXPERTS);
        let mut event = 1;
        for layer in 0..LAYERS {
            for expert in RESIDENT {
                install(&mut f.host, layer, expert, event);
                event += 1;
            }
        }
        let mut rng = Lcg(0x7CA9_0001);
        let (mut all_hit, mut all_miss, mut mixed, mut whole_fault, mut mixed_fault) =
            (0, 0, 0, 0, 0);
        let mut total_missed = 0u32;
        for route in 0..3_000 {
            let layer = route % LAYERS;
            let mut ids: Vec<i32> = Vec::new();
            let valid = if route % 13 == 12 {
                0
            } else if route % 11 == 10 {
                TOP_K - 1
            } else {
                TOP_K
            };
            while ids.len() < valid {
                let id = (rng.next() % EXPERTS as u64) as i32;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            while ids.len() < TOP_K {
                ids.insert((rng.next() % (ids.len() as u64 + 1)) as usize, -1);
            }
            let weights: Vec<f32> = ids
                .iter()
                .map(|&id| {
                    if id < 0 {
                        0.0
                    } else {
                        0.1 + (id as f32) * 0.05
                    }
                })
                .collect();
            let want = f
                .host
                .begin_layer(layer, &ids, &weights, &f.hidden, &f.hidden)
                .unwrap();
            f.host.finish_layer(layer).unwrap();
            let row = f.host.slot_row(layer).unwrap();
            let got = classify_reference(&row, EXPERTS, &ids, &weights).unwrap();
            assert_eq!(got.slot_ids, want.slot_ids, "route {route}");
            assert_eq!(
                bits(&got.weights),
                bits(&want.device_weights),
                "route {route}"
            );
            assert_eq!(got.word, 0, "route {route}");
            // begin_layer sends a miss to slot 0 with weight 0.0, and every real weight here is positive.
            let want_misses: Vec<(i32, u32)> = (0..ids.len())
                .filter(|&i| ids[i] >= 0 && want.device_weights[i] == 0.0)
                .map(|i| (ids[i], weights[i].to_bits()))
                .collect();
            let got_misses: Vec<(i32, u32)> = got
                .misses
                .iter()
                .map(|&(id, w)| (id, w.to_bits()))
                .collect();
            assert_eq!(
                got_misses, want_misses,
                "route {route}: the miss list, in pick order"
            );
            let valid_picks = ids.iter().filter(|&&id| id >= 0).count();
            let missed = want_misses.len();
            total_missed += missed as u32;
            match (valid_picks, ids.contains(&-1)) {
                (0, _) => whole_fault += 1,
                (_, true) => mixed_fault += 1,
                _ if missed == 0 => all_hit += 1,
                _ if missed == TOP_K => all_miss += 1,
                _ => mixed += 1,
            }
        }
        assert_eq!(
            total_missed, f.host.report.cpu_served,
            "the misses of the run are what the host counted as served on the CPU"
        );
        for (what, seen) in [
            ("all hits", all_hit),
            ("all misses", all_miss),
            ("mixed hits and misses", mixed),
            ("whole-layer faults", whole_fault),
            ("fault ids among valid picks", mixed_fault),
        ] {
            assert!(seen >= 10, "the run covers {what} only {seen} times");
        }
    }

    #[test]
    fn an_id_below_the_fault_id_is_refused_by_the_route_check_and_the_log() {
        // The router writes -1 and nothing else below zero. The device classification refuses
        // any other negative id (ROUTE_INVALID), so the v1 check and the log path refuse it too.
        let mut f = fixture();
        let (ids, weights) = ([2, -2, 5, 3], [0.4, 0.0, 0.2, 0.1]);
        assert!(matches!(
            f.host.begin_layer(0, &ids, &weights, &f.hidden, &f.hidden),
            Err(HostError::BadRoute(_))
        ));
        let picks: Vec<(i32, f32)> = [(2, 0.4), (-1, 0.0), (5, 0.2), (3, 0.1)].repeat(LAYERS);
        f.host
            .account_token(&RouteLog::new(LAYERS, TOP_K, picks.clone()).unwrap())
            .unwrap();
        let mut bad = picks;
        bad[TOP_K + 1].0 = -2;
        assert!(matches!(
            f.host
                .account_token(&RouteLog::new(LAYERS, TOP_K, bad).unwrap()),
            Err(HostError::BadRoute(_))
        ));
    }

    #[test]
    fn a_log_is_refused_while_a_v1_layer_is_in_flight() {
        let mut f = fixture();
        f.host
            .begin_layer(0, &[1], &[0.5], &f.hidden, &f.hidden)
            .unwrap();
        let log = RouteLog::new(LAYERS, TOP_K, vec![(1, 0.5); LAYERS * TOP_K]).unwrap();
        assert!(matches!(
            f.host.account_token(&log),
            Err(HostError::Order(_))
        ));
    }

    #[test]
    fn a_log_of_the_wrong_shape_is_refused() {
        // Distinct ids in every layer: check_route would accept each layer, so only the shape
        // guard of account_token can refuse these logs.
        let distinct = |layers: usize, top_k: usize| {
            let picks = (0..layers * top_k)
                .map(|i| ((i % top_k) as i32, 0.25))
                .collect();
            RouteLog::new(layers, top_k, picks).unwrap()
        };
        assert!(RouteLog::new(LAYERS, TOP_K, vec![(0, 0.5); LAYERS * TOP_K - 1]).is_err());
        let mut f = fixture();
        for (what, log) in [
            ("one layer short", distinct(LAYERS - 1, TOP_K)),
            ("one layer too many", distinct(LAYERS + 1, TOP_K)),
            ("one pick too wide", distinct(LAYERS, TOP_K + 1)),
            ("one pick too narrow", distinct(LAYERS, TOP_K - 1)),
        ] {
            assert!(
                matches!(f.host.account_token(&log), Err(HostError::BadRoute(_))),
                "{what}"
            );
        }
        f.host.account_token(&distinct(LAYERS, TOP_K)).unwrap();
        assert!(f.host.slot_row(LAYERS).is_err());
    }

    #[test]
    fn slot_rows_equal_the_ledger_after_every_boundary() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        let mut rng = Lcg(0x7CA9_0004_0000_0002);
        for token in 0..60 {
            let routes = seeded_routes(&mut rng, token);
            let log = RouteLog::new(LAYERS, TOP_K, routes.concat()).unwrap();
            f.host.account_token(&log).unwrap();
            f.host.end_token(&mut device).unwrap();
            for layer in 0..LAYERS {
                let row = f.host.slot_row(layer).unwrap();
                assert_eq!(row.len(), EXPERTS);
                let mut slots = Vec::new();
                for (expert, &slot) in row.iter().enumerate() {
                    let ledger = f
                        .host
                        .table
                        .lookup(layer, expert as u16)
                        .map_or(-1, |(slot, _)| slot as i32);
                    assert_eq!(slot, ledger, "token {token} layer {layer} expert {expert}");
                    assert_eq!(slot >= 0, f.host.is_resident(layer, expert as u16));
                    assert!((-1..SLOTS as i32).contains(&slot), "slot {slot}");
                    if slot >= 0 {
                        slots.push(slot);
                    }
                }
                slots.sort_unstable();
                slots.dedup();
                let resident = row.iter().filter(|&&s| s >= 0).count();
                assert_eq!(
                    slots.len(),
                    resident,
                    "token {token} layer {layer}: slots are distinct"
                );
            }
        }
    }

    #[test]
    fn a_preload_fills_every_slot_of_every_layer_and_everything_is_then_a_bank_hit() {
        // As many slots as experts: every pick is a bank hit from the first token.
        let mut f = fixture_with_slots(Duration::from_secs(60), EXPERTS);
        let mut device = MockDevice::default();
        assert_eq!(f.host.preload_all(&mut device).unwrap(), LAYERS * EXPERTS);
        f.host.check_consistency().unwrap();
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS as u16 {
                assert!(
                    f.host.is_resident(layer, expert),
                    "layer {layer} expert {expert}"
                );
            }
        }
        let rows = filled(&device, &f.reference);
        assert_eq!(rows.len(), LAYERS * EXPERTS);
        assert_eq!(
            rows[..3],
            [(0, 0, 0), (0, 1, 1), (0, 2, 2)],
            "slot i holds expert i"
        );
        let plan = f
            .host
            .begin_layer(
                1,
                &[7, 0, 3, 5],
                &[0.4, 0.3, 0.2, 0.1],
                &f.hidden,
                &f.hidden,
            )
            .unwrap();
        assert_eq!(
            plan.slot_ids,
            vec![7, 0, 3, 5],
            "the preload puts expert e into slot e"
        );
        assert_eq!(plan.device_weights, vec![0.4, 0.3, 0.2, 0.1]);
        assert_eq!(f.host.finish_layer(1), Ok(None), "nothing runs on the CPU");
        let mut few = fixture();
        assert_eq!(
            few.host.preload_all(&mut MockDevice::default()).unwrap(),
            LAYERS * SLOTS
        );
        assert!(few.host.is_resident(2, 2) && !few.host.is_resident(2, 3));
        assert!(matches!(
            few.host.preload_all(&mut MockDevice::default()),
            Err(HostError::Config(_))
        ));
    }

    #[test]
    fn a_failing_preload_poisons_the_host_and_keeps_the_ledger_consistent() {
        let mut f = fixture();
        let mut device = MockDevice {
            fail_on: Some(4),
            ..MockDevice::default()
        };
        assert!(matches!(
            f.host.preload_all(&mut device),
            Err(HostError::Device(_))
        ));
        f.host.table.check_partition().unwrap();
        assert!(
            f.host.is_resident(0, 0) && f.host.is_resident(1, 0),
            "what landed is published"
        );
        assert!(matches!(
            f.host.preload_all(&mut device),
            Err(HostError::Poisoned)
        ));
    }

    #[test]
    fn prefetch_fills_through_the_state_machine_and_publishes_at_the_boundary() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        // Token one routes layer 0 to expert 1: a miss (nothing is installed).
        f.host
            .begin_layer(0, &[1], &[1.0], &f.hidden, &f.hidden)
            .unwrap();
        assert!(f.host.finish_layer(0).unwrap().is_some());
        // The arrived advice names the miss (1) and a second expert (2): both promote
        // into free slots, evicting nothing.
        f.host.prefetch(&mut device, 0, &[1, 2]).unwrap();
        assert!(
            !f.host.is_resident(0, 1) && !f.host.is_resident(0, 2),
            "a queued fill is not resident until the boundary publishes it"
        );
        let report = f.host.end_token(&mut device).unwrap();
        assert_eq!((report.guess_fills, report.guess_evicts), (2, 0));
        assert!(f.host.is_resident(0, 1) && f.host.is_resident(0, 2));
        f.host.check_consistency().unwrap();
        // The next token's layer 0 serves both picks from the bank.
        f.host
            .begin_layer(0, &[1, 2], &[0.5, 0.5], &f.hidden, &f.hidden)
            .unwrap();
        assert!(f.host.finish_layer(0).unwrap().is_none(), "both picks hit");
        f.host.end_token(&mut device).unwrap();
        f.host.check_consistency().unwrap();
    }

    #[test]
    fn prefetch_never_evicts_a_victim_whose_fill_is_still_in_flight() {
        // Two slots, both preloaded: experts 0 and 1 sit Published. Scoring: expert 1
        // routed three times, expert 2 twice (the first advice's promotee), expert 3
        // three times (the second advice's candidate, which must beat 2).
        let mut f = fixture_with_rate(Duration::from_secs(60), 2, 0.0);
        let mut device = MockDevice::default();
        f.host.preload_all(&mut device).unwrap();
        let token = |f: &mut Fixture, device: &mut MockDevice, id: u16| {
            f.host
                .begin_layer(0, &[i32::from(id)], &[1.0], &f.hidden, &f.hidden)
                .unwrap();
            f.host.finish_layer(0).unwrap();
            f.host.end_token(device).unwrap();
        };
        for id in [1u16, 1, 1, 2, 2, 3, 3, 3] {
            token(&mut f, &mut device, id);
        }
        let fills_before = filled(&device, &f.reference).len();
        // First advice: promote 2, evict the unscored 0 (Published: applied, pending).
        f.host.prefetch(&mut device, 0, &[2]).unwrap();
        assert_eq!(filled(&device, &f.reference).len(), fills_before + 1);
        // Second advice: promote 3 over the PENDING 2 (now the lowest resident).
        // The host must skip it rather than unpublish a filling slot.
        f.host.prefetch(&mut device, 0, &[3]).unwrap();
        assert_eq!(
            filled(&device, &f.reference).len(),
            fills_before + 1,
            "the promotion racing the pending fill is skipped"
        );
        let report = f.host.end_token(&mut device).unwrap();
        assert_eq!(report.guess_fills, 1);
        assert!(f.host.is_resident(0, 2), "the pending fill publishes");
        assert!(
            !f.host.is_resident(0, 3),
            "the skipped candidate never filled"
        );
        f.host.check_consistency().unwrap();
    }

    #[test]
    fn chained_boundary_swaps_publish_a_pending_victim_before_eviction() {
        let mut f = fixture_with_rate(Duration::from_secs(60), 2, 2.0);
        let mut device = MockDevice::default();
        f.host.preload_all(&mut device).unwrap();
        let routes = [
            vec![1, 2, 6, 5],
            vec![7, 3, 0],
            vec![0, 4],
            vec![2, 7, 0, 4],
            vec![3, 6, 5, 0],
            vec![0, 7, 4],
        ];
        let mut last = None;
        let mut last_waits = 0;
        for ids in routes {
            let before = device.waits;
            last = Some(run_token(&mut f, &[ids], &mut device).unwrap());
            last_waits = device.waits - before;
            f.host.check_consistency().unwrap();
        }
        assert_eq!(last.unwrap().swaps, 2);
        assert_eq!(last_waits, 3, "only the dependent swap needs an extra wait");
        assert!(f.host.is_resident(0, 7));
        assert!(!f.host.is_resident(0, 4));
    }

    #[test]
    fn three_chained_boundary_swaps_keep_policy_and_ledger_in_sync() {
        let mut f = fixture_with_rate(Duration::from_secs(60), 2, 8.0);
        let mut device = MockDevice::default();
        f.host.preload_all(&mut device).unwrap();
        let before = device.waits;
        run_token(&mut f, &[vec![0, 3]], &mut device).unwrap();
        assert_eq!(device.waits - before, 2, "an independent swap adds no wait");
        f.host.check_consistency().unwrap();
        let before = device.waits;
        let report = run_token(&mut f, &[vec![6, 2, 4, 3]], &mut device).unwrap();
        assert_eq!(report.swaps, 3);
        assert_eq!(
            device.waits - before,
            4,
            "both pending victims must publish"
        );
        f.host.check_consistency().unwrap();
        assert!(f.host.is_resident(0, 6));
        assert!(!f.host.is_resident(0, 2));
        assert!(!f.host.is_resident(0, 4));
    }

    #[test]
    fn a_failed_wait_for_a_pending_boundary_victim_poisons_the_host() {
        let mut f = fixture_with_rate(Duration::from_secs(60), 2, 2.0);
        let mut device = MockDevice::default();
        f.host.preload_all(&mut device).unwrap();
        for ids in [
            vec![1, 2, 6, 5],
            vec![7, 3, 0],
            vec![0, 4],
            vec![2, 7, 0, 4],
            vec![3, 6, 5, 0],
        ] {
            run_token(&mut f, &[ids], &mut device).unwrap();
        }
        // The first boundary wait succeeds; the dependent victim's wait fails.
        device.fail_wait_at = Some(device.waits + 1);
        assert!(matches!(
            run_token(&mut f, &[vec![0, 7, 4]], &mut device),
            Err(HostError::Device(_))
        ));
        assert!(matches!(
            f.host.end_token(&mut device),
            Err(HostError::Poisoned)
        ));
    }

    #[test]
    fn a_failed_wait_fills_poisons_like_a_failed_fill() {
        let mut f = fixture();
        let mut device = MockDevice {
            fail_wait: true,
            ..MockDevice::default()
        };
        f.host
            .begin_layer(0, &[1], &[1.0], &f.hidden, &f.hidden)
            .unwrap();
        f.host.finish_layer(0).unwrap();
        f.host.prefetch(&mut device, 0, &[1]).unwrap();
        assert!(
            matches!(f.host.end_token(&mut device), Err(HostError::Device(_))),
            "the wait failure surfaces, and the poison flag stays"
        );
        assert!(matches!(
            f.host.end_token(&mut device),
            Err(HostError::Poisoned)
        ));
        // Poisoned means the policy and the ledger disagree by design; classification
        // still serves every pick from the ledger (all misses, on the CPU).
        assert!(!f.host.is_resident(0, 1));
    }
}
