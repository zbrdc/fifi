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
    /// `ids` and `weights` are the router's picks in rank order (`ids[i] < 0` is the
    /// router-fault value); `hidden` is the normed row the CPU experts read. A resident
    /// pick maps to its bank slot with its weight; a missed pick maps to slot 0 with
    /// weight 0.0 and one CPU job; a negative id passes through. Nothing is started when
    /// the route is refused.
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

    /// Validates a route: the layer, the list lengths, every non-negative expert id (in
    /// range, picked once) and its weight (finite).
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
            if id < 0 {
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
