//! The host side of the tiered decode step (OFF-2 D4): classify the routed experts of a
//! layer against the residency table, start the CPU jobs of the missed ones, and decide
//! the bank changes of the token boundary.
//!
//! WHAT: [`TieredHost`] owns the tier policy, the residency ledger, the expert locator
//! over the mapped GGUF, the CPU pool and the per-token counters. The device side is a
//! trait ([`BankDevice`]), so everything here runs and is tested without CUDA. WHY: the
//! engine must decide, per layer and between the router and the expert launches, which
//! picks the GPU bank serves and which the CPU serves, without a second place that could
//! disagree with the ledger. Design: `ledger/plans/OFF-2-DEVICE-WIRING.md` sections 2-3.
//!
//! The rules of one layer (a missed pick maps to bank slot 0 with device weight 0.0 so the
//! existing expert launches stay unchanged; the router-fault id `-1` passes through) and
//! of the token boundary (every swap goes through the full residency state machine; all
//! publication happens here, never inside a layer) are fixed by the ruling of
//! 2026-10-13. Driven by the tiered step (OFF-2 D5).

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use super::cpu_pool::{CpuPool, MappedRegion, PendingLayer, PoolError};
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
    /// A call out of order, e.g. `finish_layer` for a layer that was not begun.
    Order(String),
    /// The CPU pool refused or failed the layer.
    Pool(PoolError),
    /// The residency ledger refused an operation.
    Residency(ResidencyError),
    /// The locator could not name the bytes of an expert.
    Locator(LocatorError),
    /// The device refused to fill a bank slot.
    Device(String),
    /// A bank fill failed earlier: the policy and the ledger no longer agree, so
    /// `end_token` refuses until the host is rebuilt. Classification still reads the
    /// ledger, so layers keep being served correctly (every unfilled pick on the CPU).
    Poisoned,
    /// `check_consistency` found the policy and the ledger disagreeing.
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
/// The real implementation (D5) writes them with blocking copies; tests use a mock.
pub trait BankDevice {
    /// Fills `slot` of the bank of `layer` with the expert whose bytes `ranges` locate
    /// inside `region`. Returns when the bytes are on the device.
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError>;
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

/// The CPU jobs of the layer in flight, tagged with their layer.
struct InFlight {
    layer: usize,
    jobs: Option<PendingLayer>,
}

/// A router weight in millionths for the weight-mass counters; a negative weight counts as zero.
fn weight_ppm(weight: f32) -> u64 {
    (f64::from(weight).max(0.0) * 1e6).round() as u64
}

/// The picks of every layer of one token as the device logged them: `(id, weight)` in rank order,
/// exactly `top_k` per layer (`id == -1` is the router-fault value, any other negative id is
/// refused). WHY (TCAP): the classification moves onto the device, so the host learns a token's
/// routes after it ran and accounts for them at the boundary with [`TieredHost::account_token`].
/// First caller: TCAP C6.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteLog {
    layers: usize,
    top_k: usize,
    picks: Vec<(i32, f32)>,
}

impl RouteLog {
    /// A log of `layers` layers of `top_k` picks each; `Err(BadRoute)` when `picks` is not exactly
    /// `layers * top_k` long.
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

    /// The picks of `layer`, empty for a layer outside the log.
    fn layer(&self, layer: usize) -> &[(i32, f32)] {
        let at = layer.saturating_mul(self.top_k);
        self.picks.get(at..at + self.top_k).unwrap_or(&[])
    }
}

/// The host side of the tiered step.
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
}

/// The ledger deadline of a synchronous fill: any later clock value expires it, which is how
/// `abandon_fills` poisons the slot of a fill the device failed.
const FILL_DEADLINE_NS: u64 = 1;

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
        })
    }

    /// Classifies the picks of `layer` and starts the CPU jobs of the missed ones.
    ///
    /// `ids` and `weights` are the router's picks in rank order (`ids[i] == -1` is the
    /// router-fault value; any other negative id is refused); `hidden` is the normed row the
    /// CPU experts read. A resident pick maps to its bank slot with its weight; a missed pick
    /// maps to slot 0 with weight 0.0 and one CPU job; the fault id -1 passes through. Nothing
    /// is started when the route is refused.
    pub fn begin_layer(
        &mut self,
        layer: usize,
        ids: &[i32],
        weights: &[f32],
        hidden: &[f32],
    ) -> Result<LayerPlanHost, HostError> {
        if let Some(open) = &self.in_flight {
            return Err(HostError::Order(format!(
                "layer {layer} begun while layer {} is unfinished",
                open.layer
            )));
        }
        self.check_route(layer, ids, weights)?;
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

    /// Accounts for a whole token from the device's route log: the same counters and the same
    /// `routes` per layer as [`Self::begin_layer`] leaves, without starting any CPU job (the device
    /// classified and a server ran the jobs). Every layer is validated first, so a refused log
    /// changes nothing. Refuses while a v1 layer is in flight. A log must hold exactly the config's
    /// `top_k` picks per layer. First caller: TCAP C6.
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
    /// model. First caller: TCAP C6.
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
        for swap in &swaps {
            if let Err(error) = self.apply_swap(device, swap) {
                self.poisoned = true;
                self.table.publish_ready();
                return Err(error);
            }
        }
        self.table.publish_ready();
        let mut report = std::mem::take(&mut self.report);
        report.swaps = u32::try_from(swaps.len()).unwrap_or(u32::MAX);
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
        self.table.publish_ready();
        Ok(fills)
    }

    /// One bank change through the ledger: unpublish the victim, take a free slot for the
    /// promoted expert, copy its bytes and complete the fill event.
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
            self.abandon_fills()?;
            return Err(error);
        }
        self.table
            .event_complete(event)
            .map_err(HostError::Residency)
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

    /// Whether `expert` of `layer` is published in its bank.
    pub fn is_resident(&self, layer: usize, expert: u16) -> bool {
        self.table.lookup(layer, expert).is_some()
    }

    /// Checks the ledger's partition invariants and that the policy and the ledger agree
    /// on every layer's residents.
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
        TieredConfig {
            layers: LAYERS,
            experts: EXPERTS,
            top_k: TOP_K,
            bank_slots_per_layer: slots,
            mode: PolicyMode::Adaptive,
            swaps_per_token: 0.25,
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

    fn fixture_with_slots(timeout: Duration, slots: usize) -> Fixture {
        let mut rng = Lcg(0xD4D4_0000_5EED_0001);
        let ((bytes, _), _) = expert_file(&mut rng, LAYERS, EXPERTS, SHAPE);
        let header = parse(&bytes);
        let names = ExpertTensorNames::llama_cpp();
        let make = || ExpertLocator::new(&header, &names, LAYERS, EXPERTS, SHAPE).unwrap();
        let region: Arc<dyn MappedRegion> = Arc::new(bytes);
        let kernel = ExpertKernel::Q8k(DotBackend::scalar());
        let pool = CpuPool::with_kernel(2, SHAPE, kernel).unwrap();
        let host = TieredHost::new(
            config_with(slots),
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

    fn fixture_with(timeout: Duration) -> Fixture {
        fixture_with_slots(timeout, SLOTS)
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
        // Mixed: resident 2, missed 3, router fault, resident 5.
        let plan = f
            .host
            .begin_layer(1, &[2, 3, -1, 5], &[0.4, 0.3, 0.0, 0.2], &f.hidden)
            .unwrap();
        assert_eq!(plan.slot_ids, vec![s2 as i32, 0, -1, s5 as i32]);
        assert_eq!(plan.device_weights, vec![0.4, 0.0, 0.0, 0.2]);
        assert!(f.host.finish_layer(1).unwrap().is_some());
        // All resident: nothing for the CPU.
        let plan = f
            .host
            .begin_layer(1, &[5, 2], &[0.6, 0.4], &f.hidden)
            .unwrap();
        assert_eq!(plan.slot_ids, vec![s5 as i32, s2 as i32]);
        assert_eq!(plan.device_weights, vec![0.6, 0.4]);
        assert_eq!(f.host.finish_layer(1), Ok(None));
        // All faulted: every id passes through, no job.
        let plan = f
            .host
            .begin_layer(0, &[-1, -1, -1, -1], &[0.0; 4], &f.hidden)
            .unwrap();
        assert_eq!(plan.slot_ids, vec![-1; 4]);
        assert_eq!(f.host.finish_layer(0), Ok(None));
        // Nothing resident in layer 2: everything on the CPU through slot 0.
        let plan = f
            .host
            .begin_layer(2, &[0, 1], &[0.5, 0.5], &f.hidden)
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

    #[test]
    fn the_cpu_partial_is_the_job_order_sum_over_the_locator_ranges_of_the_missed_experts() {
        let mut f = fixture();
        install(&mut f.host, 1, 6, 1);
        let ids = [3, 6, 1];
        let weights = [0.25f32, 0.5, 0.125];
        f.host.begin_layer(1, &ids, &weights, &f.hidden).unwrap();
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
            let result = f.host.begin_layer(layer, &ids, &weights, &h);
            assert!(matches!(result, Err(HostError::BadRoute(_))), "{ids:?}");
            assert!(matches!(
                f.host.finish_layer(layer.min(LAYERS - 1)),
                Err(HostError::Order(_))
            ));
        }
        assert_eq!(f.host.report, TokenReport::default());
        assert!(f.host.begin_layer(0, &[1], &[0.5], &h).is_ok());
    }

    #[test]
    fn finish_layer_out_of_order_is_refused_and_the_layer_stays_in_flight() {
        let mut f = fixture();
        f.host.begin_layer(1, &[3], &[0.5], &f.hidden).unwrap();
        assert!(matches!(
            f.host.begin_layer(2, &[3], &[0.5], &f.hidden),
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
            .begin_layer(1, &[3, 4], &[0.5, 0.5], &f.hidden)
            .unwrap();
        let err = f.host.finish_layer(1).unwrap_err();
        assert!(
            matches!(err, HostError::Pool(PoolError::Timeout { .. })),
            "{err:?}"
        );
        assert!(f.host.in_flight.is_none());
        f.host.begin_layer(2, &[1], &[0.5], &f.hidden).unwrap();
        assert!(f.host.finish_layer(2).is_err());
        f.host.begin_layer(0, &[-1], &[0.0], &f.hidden).unwrap();
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

    /// Records the fills the host asks for; can fail the n-th one.
    #[derive(Default)]
    struct MockDevice {
        fills: Vec<(usize, usize, ExpertRanges)>,
        fail_on: Option<usize>,
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

    /// One token: every layer's picks (equal weights), then the boundary.
    fn run_token(
        f: &mut Fixture,
        picks: &[Vec<i32>],
        device: &mut MockDevice,
    ) -> Result<TokenReport, HostError> {
        for (layer, ids) in picks.iter().enumerate() {
            let weights = vec![1.0 / ids.len() as f32; ids.len()];
            f.host.begin_layer(layer, ids, &weights, &f.hidden)?;
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
        // The next token finds them resident.
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
        // The ledger holds: the first fill landed and is published, the failed one is
        // gone and its slot is free again.
        f.host.table.check_partition().unwrap();
        assert!(f.host.is_resident(0, 0) && !f.host.is_resident(0, 1));
        assert!(matches!(
            f.host.check_consistency(),
            Err(HostError::Inconsistent(_))
        ));
        // The boundary refuses from now on, the layers are still served from the ledger.
        assert_eq!(f.host.end_token(&mut device), Err(HostError::Poisoned));
        let plan = f
            .host
            .begin_layer(0, &[0, 1], &[0.5, 0.5], &f.hidden)
            .unwrap();
        assert_eq!(plan.device_weights, vec![0.5, 0.0]);
        assert!(f.host.finish_layer(0).unwrap().is_some());
    }

    #[test]
    fn end_token_refuses_while_a_layer_is_in_flight() {
        let mut f = fixture();
        let mut device = MockDevice::default();
        f.host.begin_layer(0, &[1], &[0.5], &f.hidden).unwrap();
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
        f.host.begin_layer(0, &[1], &[0.5], &f.hidden).unwrap();
        // The failure between begin_layer and finish_layer: without the abort every later
        // layer is refused for good.
        assert!(matches!(
            f.host.begin_layer(1, &[2], &[0.5], &f.hidden),
            Err(HostError::Order(_))
        ));
        f.host.abort_token();
        f.host.begin_layer(1, &[2], &[0.5], &f.hidden).unwrap();
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
        // The failed token routed experts 1 and 2 in layer 0 and then failed.
        f.host
            .begin_layer(0, &[1, 2], &[0.5, 0.5], &f.hidden)
            .unwrap();
        f.host.abort_token();
        // The next token reaches only layers 1 and 2: a stale route of layer 0 would reach the
        // policy at the boundary, and the policy would fill layer 0 for it.
        for layer in 1..LAYERS {
            f.host.begin_layer(layer, &[3], &[1.0], &f.hidden).unwrap();
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
        // The same routes through the v1 path (begin_layer and finish_layer per layer) and through
        // the log of the device: equal reports, equal swaps, equal slot rows, token after token.
        let (mut v1, mut logged) = (fixture(), fixture());
        let (mut dev_v1, mut dev_logged) = (MockDevice::default(), MockDevice::default());
        let mut rng = Lcg(0x7CA9_0004_0000_0001);
        for token in 0..120 {
            let routes = seeded_routes(&mut rng, token);
            for (layer, picks) in routes.iter().enumerate() {
                let (ids, weights): (Vec<i32>, Vec<f32>) = picks.iter().copied().unzip();
                v1.host
                    .begin_layer(layer, &ids, &weights, &v1.hidden)
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
                    f.host.begin_layer(2, &ids, &weights, &f.hidden),
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
                .begin_layer(layer, &ids, &weights, &f.hidden)
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
            f.host.begin_layer(0, &ids, &weights, &f.hidden),
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
        f.host.begin_layer(0, &[1], &[0.5], &f.hidden).unwrap();
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
            .begin_layer(1, &[7, 0, 3, 5], &[0.4, 0.3, 0.2, 0.1], &f.hidden)
            .unwrap();
        assert_eq!(
            plan.slot_ids,
            vec![7, 0, 3, 5],
            "the preload puts expert e into slot e"
        );
        assert_eq!(plan.device_weights, vec![0.4, 0.3, 0.2, 0.1]);
        assert_eq!(f.host.finish_layer(1), Ok(None), "nothing runs on the CPU");
        // With fewer slots the lowest-numbered experts fill, and a second preload is refused.
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
}
