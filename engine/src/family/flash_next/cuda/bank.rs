//! The expert bank of the Qwen3.8 Flash-Next CUDA step: which routed experts sit in the device slots
//! of every layer, which pinned host place holds every other expert, and the copies a token's picks
//! cause. The policy, the host places and the budget are pure host code; the device owner at the end
//! of the file only allocates, fills and copies.
//!
//! WHAT, in the order of a decode layer:
//!
//! 1. The router's picks go through `tiered_classify` against the layer's slot row
//!    ([`ExpertBankState::slot_row`], the oracle is `SlotTable::classify` / `classify_reference`): a
//!    resident pick maps to its slot, a missed pick to slot 0 with weight +0.0 (the filler rule).
//! 2. The resident pass runs, and the host learns the picks (hits included) so that
//!    [`ExpertBankState::begin_pass`] can plan the copies: every expert of the pass is protected, a
//!    miss takes the least recently used unprotected slot, and the slot's old expert leaves the card.
//! 3. [`PassCopies`] says what to move. The copy is synchronous (the missed picks wait for it).
//! 4. The missed picks run on their new slots with their true weights, every other pick on the
//!    filler ([`miss_pass_inputs`]); the two passes' weights add to the router's weights bit for bit.
//!
//! WHY this shape:
//!
//! - Policy. The T0 replay (`flash-next-routing-locality-t0-trace`) gives per-layer LRU 0.870 / 0.915
//!   hits at 143 / 184 slots against 0.683 / 0.763 for a static popularity core, so the popularity
//!   profile ([`parse_route_counts`], `TierPlan::hot_set`) only fills the slots before the first
//!   prompt. Q36's cap of 0.25 swaps per token came from SSD-backed misses; a Flash-Next miss is one
//!   72 us copy from pinned RAM, so every miss is admitted at once.
//! - Slot 0 is protected while a pass has a miss, because the filler reads it while the copies write
//!   other slots. A torn float there would make `0.0 * x` a NaN. So a bank needs `top_k + 1` slots.
//! - The pinned store is exclusive by default ([`HostMode::Exclusive`]): it holds only the experts
//!   that are not on the card, 25.6 to 31.1 GiB instead of the 43.07 GiB of an inclusive store. When
//!   an expert leaves a slot, its record goes back into the host place its successor just left
//!   (one spare place per layer makes the order work: demote first, then free the incoming place).
//!   The write-back (D2H) leg is not measured yet. [`HostMode::Inclusive`] keeps every expert in the
//!   store and drops the D2H leg; [`HostPinPlan`] prices both.
//! - One record per expert and one device buffer per layer: the record layout is `SlotBankLayout`'s
//!   (equal strides for the id-indexed launch), the policy and the placement are pure and tested on
//!   the host.
//!
//! A shared slot cache in `expert_tier` can replace [`ReplaceOnMissLru`] behind [`SlotPolicy`]
//! and nothing else: the interface is `plan_pass(layer, picks) -> PassPlan`,
//! `admit(layer, experts_by_use) -> PassPlan`, `slot_row(layer)` and `expert_in_slot(layer, slot)`.
//! Host places, copies, budget and the device owner do not change. The policy must keep three rules:
//! the experts of a pass are never victims, slot 0 is not a victim in a pass with a miss, and the
//! slot row it publishes is the classifier's table.
//!
//! Nothing here is called by a model step yet; the decode step wires it in a later change.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use crate::{
    EngError,
    family::{
        flash_next::moe::{slot_bank::SlotBankLayout, tier_plan::TierPlan},
        qwen36::tiered::classify::Classified,
    },
};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
pub(super) const NO_PLACE: u32 = u32::MAX;

fn plan_error(detail: String) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next expert bank: {detail}"))
}

fn bank_error(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next expert bank: {detail}"))
}

/// The sizes of the bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BankGeometry {
    pub(super) layers: usize,
    pub(super) experts: usize,
    pub(super) top_k: usize,
    /// Device slots in every layer.
    pub(super) slots: usize,
    /// Bytes of one expert record (`SlotBankLayout::record_bytes`, 1,881,600 in the 3.05 pack).
    pub(super) record_bytes: usize,
}

impl BankGeometry {
    /// Refuses a bank without `top_k + 1` slots (the filler needs one), more slots than experts,
    /// expert numbers that do not fit u16 and a record that is not whole floats.
    pub(super) fn new(
        layers: usize,
        experts: usize,
        top_k: usize,
        slots: usize,
        record_bytes: usize,
    ) -> Result<Self, EngError> {
        if layers == 0 || top_k == 0 || record_bytes == 0 || !record_bytes.is_multiple_of(4) {
            return Err(plan_error(format!(
                "{layers} layers, top-{top_k} and a {record_bytes}-byte record are not a bank"
            )));
        }
        // `u16::MAX` is the empty-slot sentinel of the policy, so it cannot be an expert number.
        if experts == 0 || experts > usize::from(u16::MAX) {
            return Err(plan_error(format!(
                "{experts} experts do not fit a 16-bit expert number below the sentinel"
            )));
        }
        if slots <= top_k {
            return Err(plan_error(format!(
                "{slots} slots per layer cannot protect {top_k} picks and the filler slot; \
                 the bank needs {} or more",
                top_k + 1
            )));
        }
        if slots > experts {
            return Err(plan_error(format!(
                "{slots} slots exceed the {experts} experts of a layer"
            )));
        }
        Ok(Self {
            layers,
            experts,
            top_k,
            slots,
            record_bytes,
        })
    }

    /// The bank the tier plan settled, for records of `layout`. The record size must be the plan's
    /// expert size, which comes from the header walk.
    pub(super) fn from_plan(plan: &TierPlan, layout: &SlotBankLayout) -> Result<Self, EngError> {
        let shape = plan.shape();
        if layout.record_bytes() as u64 != shape.expert_bytes {
            return Err(plan_error(format!(
                "the record is {} bytes, the tier plan counts {} per expert",
                layout.record_bytes(),
                shape.expert_bytes
            )));
        }
        Self::new(
            shape.layers,
            shape.experts,
            shape.top_k,
            plan.slots_per_layer(),
            layout.record_bytes(),
        )
    }

    /// Byte offset of `slot` (or of a host place) in a layer's buffer.
    pub(super) fn offset(&self, index: usize) -> usize {
        index * self.record_bytes
    }

    pub(super) fn device_bytes_per_layer(&self) -> u64 {
        self.slots as u64 * self.record_bytes as u64
    }
}

/// Which experts the pinned store holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HostMode {
    /// Only the experts that are not in a slot, plus one spare place per layer. An expert that
    /// leaves a slot is written back into the place its successor just left.
    Exclusive,
    /// Every expert of every layer: eviction is free, and the store is 43 GiB.
    Inclusive,
}

impl HostMode {
    pub(super) fn places_per_layer(self, geometry: &BankGeometry) -> usize {
        match self {
            Self::Exclusive => geometry.experts - geometry.slots + 1,
            Self::Inclusive => geometry.experts,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host memory budget.
// ---------------------------------------------------------------------------------------------

/// `MemTotal` and `MemAvailable` of the machine, in bytes. `available` is the caller's minimum of
/// the host figure and every cgroup ancestor's remaining bytes (`flint::exl3_resident::HostBudget`
/// reads that on the CUDA path; see [`HostPinPlan::new`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HostMemory {
    pub(super) total: u64,
    pub(super) available: u64,
}

impl HostMemory {
    /// Parses the text of `/proc/meminfo`. Refuses a missing field, a unit other than kB and an
    /// available figure above the total.
    pub(super) fn from_meminfo(text: &str) -> Result<Self, EngError> {
        let field = |name: &str| -> Result<u64, EngError> {
            let invalid = || plan_error(format!("/proc/meminfo has no valid {name} in kB"));
            let line = text
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|rest| rest.strip_prefix(':'))
                .ok_or_else(invalid)?;
            let mut words = line.split_whitespace();
            let kib: u64 = words
                .next()
                .and_then(|value| value.parse().ok())
                .ok_or_else(invalid)?;
            if words.next() != Some("kB") || words.next().is_some() {
                return Err(invalid());
            }
            kib.checked_mul(1024).ok_or_else(invalid)
        };
        let (total, available) = (field("MemTotal")?, field("MemAvailable")?);
        if total == 0 || available > total {
            return Err(plan_error("/proc/meminfo totals are inconsistent".into()));
        }
        Ok(Self { total, available })
    }

    /// The OS floor: a tenth of the RAM, at least 8 GiB. The same rule as Flint's
    /// `HostBudget::from_meminfo`, which keeps it private.
    pub(super) fn os_floor(&self) -> u64 {
        self.total.div_ceil(10).max(8 * GIB)
    }
}

/// The host terms besides the pinned store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HostTerms {
    /// Process memory outside the pins: the BF16 embedding (1.18 GiB), the CUDA driver and process
    /// (about 1.5 GiB), small host buffers (0.1 GiB).
    pub(super) engine_heap: u64,
    /// Kept free so the plan does not ride the line (exllamav3 keeps 2 GiB as well).
    pub(super) margin: u64,
    /// The least page cache the n-gram table's random reads must keep. A floor, not a target: all
    /// RAM the plan leaves free is page cache. The default is 0, from measured cold n-gram row
    /// reads on the real file: 52 us median per row (read-only bench), 160-200 us per 16-row decode
    /// token (about 1% of a token, and the reads can start before the first layer) and 242 ms per
    /// 4096-token chunk (production reader).
    /// A warm page is a bonus, not a requirement, so the pinned store may take the RAM. At 0 the
    /// floor check never refuses; a nonzero floor needs a measured low hit rate on real traffic.
    /// The n-gram feed's own reserve constant, also 0, replaces this literal when that feed is
    /// merged.
    pub(super) ngram_cache_min: u64,
    /// Rows of n-gram embedding in flight at once in host memory: the staging of the row reader.
    /// A 4096-token chunk reads 65,536 rows and a double-buffered chunk worker holds two blocks.
    pub(super) ngram_staging_rows: u64,
    /// Metadata allowance charged to each of the two stream-bank slots besides the payload.
    pub(super) stream_slot_metadata: u64,
}

/// Host bytes per n-gram row in flight: 160 decoded F32 values plus the row id and its sort key.
/// 656 bytes. The n-gram feed's staging function gives the same figure and replaces this constant
/// when that feed is merged.
pub(super) const NGRAM_STAGING_BYTES_PER_ROW: u64 = 160 * 4 + 8 + 8;

impl HostTerms {
    /// The defaults for the 62 GiB host: 3 GiB heap, 2 GiB margin, no n-gram page-cache floor
    /// (measured, see the field), two 4096-token chunks of n-gram staging (86 MB).
    pub(super) const DEFAULT: Self = Self {
        engine_heap: 3 * GIB,
        margin: 2 * GIB,
        ngram_cache_min: 0,
        ngram_staging_rows: 2 * 65_536,
        stream_slot_metadata: MIB,
    };

    /// Host bytes the n-gram row reader holds in flight.
    pub(super) fn ngram_staging_bytes(&self) -> Option<u64> {
        self.ngram_staging_rows
            .checked_mul(NGRAM_STAGING_BYTES_PER_ROW)
    }
}

/// What the host must hold for a bank, and what that leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HostPinPlan {
    pub(super) mode: HostMode,
    pub(super) places_per_layer: usize,
    /// Pinned bytes of the expert places of all layers.
    pub(super) store_bytes: u64,
    /// The two-slot scratch of the stream bank (prefill belt), charged now so the belt does not
    /// re-plan the host: both slots, each a full record plus metadata
    /// (`stream-bank-scratch-is-two-slot-total`: the argument of `StreamBankPlan` is the total).
    pub(super) scratch_bytes: u64,
    /// The n-gram row reader's host staging (`HostTerms::ngram_staging_rows` rows).
    pub(super) ngram_staging_bytes: u64,
    pub(super) os_floor: u64,
    /// Store, scratch, n-gram staging, heap and margin: what this process takes.
    pub(super) process_bytes: u64,
    /// RAM left for the page cache: `MemAvailable` (other processes' holds already out) minus what
    /// this plan needs now (store, scratch, n-gram staging, heap, margin and the OS floor).
    pub(super) page_cache_left: u64,
}

impl HostPinPlan {
    /// Sizes the pinned store from the machine and refuses (it never shrinks the store, and never
    /// changes the context) when
    /// - the process and the OS floor exceed `available`, or
    /// - the page cache left (`available` minus the need) is below `ngram_cache_min`.
    ///
    /// On the CUDA path the caller also runs `HostBudget::probe()?.admit_for(process_bytes, ..)`,
    /// which applies the cgroup ancestors (and refuses a cgroup-namespace root, so run on the host).
    pub(super) fn new(
        geometry: &BankGeometry,
        mode: HostMode,
        memory: &HostMemory,
        terms: &HostTerms,
    ) -> Result<Self, EngError> {
        let overflow = || plan_error("the host byte counts overflow".into());
        let places_per_layer = mode.places_per_layer(geometry);
        let store_bytes = (geometry.layers as u64)
            .checked_mul(places_per_layer as u64)
            .and_then(|places| places.checked_mul(geometry.record_bytes as u64))
            .ok_or_else(overflow)?;
        let scratch_bytes = 2 * (geometry.record_bytes as u64 + terms.stream_slot_metadata);
        let os_floor = memory.os_floor();
        let ngram_staging_bytes = terms.ngram_staging_bytes().ok_or_else(overflow)?;
        let process_bytes = store_bytes
            .checked_add(scratch_bytes)
            .and_then(|sum| sum.checked_add(ngram_staging_bytes))
            .and_then(|sum| sum.checked_add(terms.engine_heap))
            .and_then(|sum| sum.checked_add(terms.margin))
            .ok_or_else(overflow)?;
        let needed_now = process_bytes.checked_add(os_floor).ok_or_else(overflow)?;
        if needed_now > memory.available {
            return Err(plan_error(format!(
                "{store_bytes} pinned store bytes, {scratch_bytes} stream scratch, \
                 {ngram_staging_bytes} n-gram staging, {} heap and {} margin plus the {os_floor} \
                 byte OS floor need {needed_now} bytes, {} are available",
                terms.engine_heap, terms.margin, memory.available
            )));
        }
        // Cannot underflow: `needed_now <= available` above.
        let page_cache_left = memory.available - needed_now;
        if page_cache_left < terms.ngram_cache_min {
            return Err(plan_error(format!(
                "{page_cache_left} bytes of page cache are left of {} available; the n-gram \
                 page-cache floor is {}",
                memory.available, terms.ngram_cache_min
            )));
        }
        Ok(Self {
            mode,
            places_per_layer,
            store_bytes,
            scratch_bytes,
            ngram_staging_bytes,
            os_floor,
            process_bytes,
            page_cache_left,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The popularity profile.
// ---------------------------------------------------------------------------------------------

/// Parses a route-count profile (`flash-next-route-counts/1`): `#` header lines (the schema, the
/// layer and expert counts and the source), then `layer<TAB>expert<TAB>count` for the nonzero
/// cells. Returns `counts[layer][expert]`, the input of `TierPlan::hot_set`. Refuses another
/// schema or shape, a cell outside the shape, a cell twice and a malformed line.
pub(super) fn parse_route_counts(
    text: &str,
    layers: usize,
    experts: usize,
) -> Result<Vec<Vec<u64>>, EngError> {
    let bad = |what: String| plan_error(format!("route counts: {what}"));
    let (mut schema, mut seen_layers, mut seen_experts) = (false, None, None);
    let mut counts = vec![vec![0u64; experts]; layers];
    let mut seen = vec![vec![false; experts]; layers];
    for (number, line) in text.lines().enumerate() {
        if let Some(header) = line.strip_prefix('#') {
            let header = header.trim();
            if header == "schema=flash-next-route-counts/1" {
                schema = true;
            } else if let Some(value) = header.strip_prefix("layers=") {
                seen_layers = value.parse::<usize>().ok();
            } else if let Some(value) = header.strip_prefix("experts=") {
                seen_experts = value.parse::<usize>().ok();
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        if !schema || seen_layers != Some(layers) || seen_experts != Some(experts) {
            return Err(bad(format!(
                "line {} comes before a header for schema flash-next-route-counts/1 with {layers} \
                 layers and {experts} experts",
                number + 1
            )));
        }
        let mut cell = line.split('\t');
        let mut next = |what: &str| -> Result<u64, EngError> {
            cell.next()
                .and_then(|word| word.parse().ok())
                .ok_or_else(|| bad(format!("line {}: no {what}", number + 1)))
        };
        let (layer, expert, count) = (next("layer")?, next("expert")?, next("count")?);
        if cell.next().is_some() {
            return Err(bad(format!(
                "line {} has more than three fields",
                number + 1
            )));
        }
        let (layer, expert) = (layer as usize, expert as usize);
        let place = seen
            .get_mut(layer)
            .and_then(|row| row.get_mut(expert))
            .ok_or_else(|| {
                bad(format!(
                    "line {}: cell {layer}/{expert} is outside the shape",
                    number + 1
                ))
            })?;
        if std::mem::replace(place, true) {
            return Err(bad(format!(
                "line {}: cell {layer}/{expert} twice",
                number + 1
            )));
        }
        counts[layer][expert] = count;
    }
    if !schema || seen_layers != Some(layers) || seen_experts != Some(experts) {
        return Err(bad(format!(
            "no header for schema flash-next-route-counts/1 with {layers} layers and {experts} experts"
        )));
    }
    Ok(counts)
}

// ---------------------------------------------------------------------------------------------
// The slot policy.
// ---------------------------------------------------------------------------------------------

/// One slot changing hands: `incoming` takes `slot` from `outgoing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SlotMove {
    pub(super) slot: u16,
    pub(super) incoming: u16,
    pub(super) outgoing: u16,
}

/// What a policy decided for one pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct PassPlan {
    /// Distinct experts of the pass.
    pub(super) distinct: usize,
    pub(super) hits: usize,
    /// In pick order.
    pub(super) moves: Vec<SlotMove>,
}

/// The seam a shared policy fills in. Every method keeps the rules in the module doc.
pub(super) trait SlotPolicy {
    /// Plans the copies of one layer for the router's picks. `picks` holds every pick of the token
    /// in this layer, hits included (`-1`, the router fault, is skipped; any other id outside the
    /// layer is refused; repeats count once). Atomic: an `Err` changes nothing.
    fn plan_pass(&mut self, layer: usize, picks: &[i32]) -> Result<PassPlan, EngError>;

    /// Admits experts without a router pass, oldest use first (a prefill chunk's experts after the
    /// chunk). The last `slots` distinct experts are kept; the rest are dropped.
    fn admit(&mut self, layer: usize, experts_by_use: &[u16]) -> Result<PassPlan, EngError>;

    /// The classifier's table of `layer`: the slot of every expert, or -1.
    fn slot_row(&self, layer: usize) -> &[i32];

    /// The expert in `slot` of `layer`.
    fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16>;
}

#[derive(Debug, Clone)]
struct LayerLru {
    /// Slot of every expert or -1.
    slot_of: Vec<i32>,
    /// Expert of every slot.
    resident: Vec<u16>,
    /// Slots, least recently used first.
    recency: Vec<u16>,
}

impl LayerLru {
    /// Runs the experts of `order` through the layer: a hit becomes the most recent, a miss takes
    /// the least recent slot whose expert `demanded` does not protect. Fails only when everything
    /// is protected, which the callers' capacity check rules out.
    fn apply(&mut self, order: &[u16], demanded: &[bool]) -> Result<PassPlan, EngError> {
        let mut plan = PassPlan {
            distinct: order.len(),
            ..PassPlan::default()
        };
        for &expert in order {
            let slot = match u16::try_from(self.slot_of[usize::from(expert)]) {
                Ok(slot) => {
                    plan.hits += 1;
                    slot
                }
                Err(_) => {
                    let slot = self
                        .recency
                        .iter()
                        .copied()
                        .find(|&slot| !demanded[usize::from(self.resident[usize::from(slot)])])
                        .ok_or_else(|| bank_error("no unprotected victim".into()))?;
                    let outgoing = self.resident[usize::from(slot)];
                    self.slot_of[usize::from(outgoing)] = -1;
                    self.slot_of[usize::from(expert)] = i32::from(slot);
                    self.resident[usize::from(slot)] = expert;
                    plan.moves.push(SlotMove {
                        slot,
                        incoming: expert,
                        outgoing,
                    });
                    slot
                }
            };
            let at = self
                .recency
                .iter()
                .position(|&other| other == slot)
                .ok_or_else(|| bank_error("invalid recency order".into()))?;
            self.recency.remove(at);
            self.recency.push(slot);
        }
        Ok(plan)
    }
}

/// Per-layer replace-on-miss LRU with the experts of the pass protected (Flint's cache semantics,
/// written for 512 experts and 10 picks). Every miss is admitted; there is no swap limit.
#[derive(Debug, Clone)]
pub(super) struct ReplaceOnMissLru {
    geometry: BankGeometry,
    layers: Vec<LayerLru>,
    /// Per-layer pick counts, written to `usage_out` on drop so a run can produce the popularity
    /// profile of its own workload (the read side is `FIFI_FLASH_NEXT_EXPERT_STATS`, or the
    /// `expert_stats` engine key). `None` when no write path is set.
    usage: Option<Vec<Vec<u64>>>,
    /// Where the dump goes: the `FIFI_FLASH_NEXT_EXPERT_USAGE_OUT` value, or the engine's
    /// `expert_stats` path (which then also feeds the next start's fill).
    usage_out: Option<std::path::PathBuf>,
    /// Per-layer miss counts of the same dump: where the slot capacity actually binds.
    miss_counts: Vec<u64>,
}

impl ReplaceOnMissLru {
    /// Starts from a filled bank: `rows[layer][expert]` is the slot of the expert or -1, with exactly
    /// `slots` experts held (`TierPlan::hot_set`'s table). The least popular holder (by `counts`,
    /// ties to the higher id) is the first victim.
    pub(super) fn from_fill(
        geometry: BankGeometry,
        rows: &[&[i32]],
        counts: &[Vec<u64>],
        usage_out: Option<&std::path::Path>,
    ) -> Result<Self, EngError> {
        if rows.len() != geometry.layers || counts.len() != geometry.layers {
            return Err(bank_error(format!(
                "{} rows and {} count rows for {} layers",
                rows.len(),
                counts.len(),
                geometry.layers
            )));
        }
        let mut layers = Vec::with_capacity(geometry.layers);
        for (layer, (row, count)) in rows.iter().zip(counts).enumerate() {
            if row.len() != geometry.experts || count.len() != geometry.experts {
                return Err(bank_error(format!(
                    "layer {layer}: a row of {} and counts of {} for {} experts",
                    row.len(),
                    count.len(),
                    geometry.experts
                )));
            }
            let mut resident = vec![u16::MAX; geometry.slots];
            for (expert, &slot) in row.iter().enumerate() {
                if slot < 0 {
                    continue;
                }
                let held = resident.get_mut(slot as usize).ok_or_else(|| {
                    bank_error(format!("layer {layer}: slot {slot} is outside the bank"))
                })?;
                if *held != u16::MAX {
                    return Err(bank_error(format!("layer {layer}: slot {slot} twice")));
                }
                *held = expert as u16;
            }
            if resident.contains(&u16::MAX) {
                return Err(bank_error(format!("layer {layer}: a slot is not filled")));
            }
            let mut recency: Vec<u16> = (0..geometry.slots as u16).collect();
            recency.sort_by_key(|&slot| {
                let expert = resident[usize::from(slot)];
                (count[usize::from(expert)], std::cmp::Reverse(expert))
            });
            layers.push(LayerLru {
                slot_of: row.to_vec(),
                resident,
                recency,
            });
        }
        let usage_out = usage_out.map(std::path::Path::to_path_buf);
        let usage = usage_out
            .as_ref()
            .map(|_| vec![vec![0u64; geometry.experts]; geometry.layers]);
        let miss_counts = vec![0u64; geometry.layers];
        Ok(Self {
            geometry,
            layers,
            usage,
            usage_out,
            miss_counts,
        })
    }

    /// Writes the collected pick counts in the `expert_stats` layer-key format the profile
    /// reader takes, so the file feeds straight back through `FIFI_FLASH_NEXT_EXPERT_STATS`.
    fn dump_usage(&self) {
        let Some(path) = self.usage_out.as_deref() else {
            return;
        };
        let Some(usage) = &self.usage else { return };
        let layers: serde_json::Map<String, serde_json::Value> = usage
            .iter()
            .enumerate()
            .map(|(layer, counts)| {
                (
                    format!("model.language_model.layers.{layer}.mlp"),
                    serde_json::json!(counts),
                )
            })
            .collect();
        let usage_misses = self.miss_counts.clone();
        let body =
            serde_json::json!({"schema": "fifi-usage/1", "layers": layers, "misses": usage_misses});
        // Write-and-rename: a crash mid-dump must not leave a truncated file the next start
        // refuses. The temp file sits next to the target so the rename stays on one filesystem.
        let temp = path.with_extension("json.tmp");
        match serde_json::to_vec(&body) {
            Ok(bytes) => {
                let written =
                    std::fs::write(&temp, bytes).and_then(|()| std::fs::rename(&temp, path));
                if let Err(e) = written {
                    std::fs::remove_file(&temp).ok();
                    eprintln!("flash-next cuda: expert usage dump {}: {e}", path.display());
                }
            }
            // The body is maps of u64 vectors: serialization cannot fail. Skip the write
            // instead of emitting an empty file a next start would refuse.
            Err(e) => eprintln!("flash-next cuda: expert usage dump {}: {e}", path.display()),
        }
    }

    fn layer(&self, layer: usize) -> Result<&LayerLru, EngError> {
        self.layers
            .get(layer)
            .ok_or_else(|| bank_error(format!("layer {layer} is outside the bank")))
    }

    fn check_expert(&self, id: i32) -> Result<u16, EngError> {
        usize::try_from(id)
            .ok()
            .filter(|&expert| expert < self.geometry.experts)
            .map(|expert| expert as u16)
            .ok_or_else(|| bank_error(format!("expert {id} is outside the layer")))
    }
}

impl Drop for ReplaceOnMissLru {
    fn drop(&mut self) {
        self.dump_usage();
    }
}

impl SlotPolicy for ReplaceOnMissLru {
    fn plan_pass(&mut self, layer: usize, picks: &[i32]) -> Result<PassPlan, EngError> {
        let current = self.layer(layer)?;
        let mut demanded = vec![false; self.geometry.experts];
        let mut order = Vec::with_capacity(picks.len());
        // Pick ids deferred to after `current`'s borrow ends, for the usage profile.
        let mut picked = Vec::with_capacity(picks.len());
        for &id in picks {
            if id == -1 {
                continue;
            }
            let expert = self.check_expert(id)?;
            picked.push(usize::from(expert));
            if !std::mem::replace(&mut demanded[usize::from(expert)], true) {
                order.push(expert);
            }
        }
        if order
            .iter()
            .any(|&expert| current.slot_of[usize::from(expert)] < 0)
        {
            // The filler: slot 0 is read by the resident pass while the copies run.
            demanded[usize::from(current.resident[0])] = true;
        }
        let protected = demanded.iter().filter(|&&flag| flag).count();
        if protected > self.geometry.slots {
            return Err(bank_error(format!(
                "{protected} protected experts exceed the {} slots",
                self.geometry.slots
            )));
        }
        let mut next = current.clone();
        let _ = current; // end the borrow: the write-back below takes the layer slot
        if let Some(usage) = &mut self.usage {
            for expert in picked {
                usage[layer][expert] += 1;
            }
        }
        let plan = next.apply(&order, &demanded)?;
        self.miss_counts[layer] += plan.moves.len() as u64;
        self.layers[layer] = next;
        Ok(plan)
    }

    fn admit(&mut self, layer: usize, experts_by_use: &[u16]) -> Result<PassPlan, EngError> {
        let current = self.layer(layer)?;
        let mut demanded = vec![false; self.geometry.experts];
        let mut newest_first = Vec::new();
        for &expert in experts_by_use.iter().rev() {
            self.check_expert(i32::from(expert))?;
            if newest_first.len() < self.geometry.slots
                && !std::mem::replace(&mut demanded[usize::from(expert)], true)
            {
                newest_first.push(expert);
            }
        }
        newest_first.reverse();
        let mut next = current.clone();
        let plan = next.apply(&newest_first, &demanded)?;
        self.layers[layer] = next;
        Ok(plan)
    }

    fn slot_row(&self, layer: usize) -> &[i32] {
        self.layers.get(layer).map_or(&[], |lru| &lru.slot_of)
    }

    fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16> {
        self.layers.get(layer)?.resident.get(slot).copied()
    }
}

// ---------------------------------------------------------------------------------------------
// Host places and the copies of a pass.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LayerPlaces {
    /// Place of every expert in the pinned store or `NO_PLACE`.
    place_of: Vec<u32>,
    free: Vec<u32>,
}

/// The place of every expert that is in the pinned store.
#[derive(Debug, Clone)]
pub(super) struct HostPlaces {
    mode: HostMode,
    layers: Vec<LayerPlaces>,
}

impl HostPlaces {
    /// `rows[layer][expert]` is the slot of the expert or -1. Exclusive: the experts without a slot
    /// take places 0.. in id order and one place stays free. Inclusive: place = expert.
    fn new(geometry: &BankGeometry, mode: HostMode, rows: &[&[i32]]) -> Self {
        let layers = rows
            .iter()
            .map(|row| match mode {
                HostMode::Inclusive => LayerPlaces {
                    place_of: (0..geometry.experts as u32).collect(),
                    free: Vec::new(),
                },
                HostMode::Exclusive => {
                    let mut place_of = vec![NO_PLACE; geometry.experts];
                    let mut next = 0u32;
                    for (expert, &slot) in row.iter().enumerate() {
                        if slot < 0 {
                            place_of[expert] = next;
                            next += 1;
                        }
                    }
                    LayerPlaces {
                        place_of,
                        free: vec![next],
                    }
                }
            })
            .collect();
        Self { mode, layers }
    }

    pub(super) fn place_of(&self, layer: usize, expert: u16) -> Option<u32> {
        let place = *self.layers.get(layer)?.place_of.get(usize::from(expert))?;
        (place != NO_PLACE).then_some(place)
    }

    /// `(place, expert)` of every expert the store holds in `layer`, by place.
    pub(super) fn occupied(&self, layer: usize) -> Vec<(u32, u16)> {
        let mut held: Vec<(u32, u16)> = self.layers.get(layer).map_or_else(Vec::new, |places| {
            places
                .place_of
                .iter()
                .enumerate()
                .filter(|&(_, &place)| place != NO_PLACE)
                .map(|(expert, &place)| (place, expert as u16))
                .collect()
        });
        held.sort_unstable();
        held
    }

    /// An expert leaves a slot: Exclusive gives it a free place (the one to write it back to).
    fn demote(&mut self, layer: usize, expert: u16) -> Result<Option<u32>, EngError> {
        if self.mode == HostMode::Inclusive {
            return Ok(None);
        }
        let places = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| bank_error(format!("layer {layer} is outside the bank")))?;
        let place = places
            .free
            .pop()
            .ok_or_else(|| bank_error("no free host place for the evicted expert".into()))?;
        let slot = &mut places.place_of[usize::from(expert)];
        if *slot != NO_PLACE {
            return Err(bank_error(format!(
                "expert {expert} already has a host place"
            )));
        }
        *slot = place;
        Ok(Some(place))
    }

    /// An expert enters a slot: its place is where the copy reads, and Exclusive frees it.
    fn promote(&mut self, layer: usize, expert: u16) -> Result<u32, EngError> {
        let places = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| bank_error(format!("layer {layer} is outside the bank")))?;
        let place = places.place_of[usize::from(expert)];
        if place == NO_PLACE {
            return Err(bank_error(format!("expert {expert} has no host place")));
        }
        if self.mode == HostMode::Exclusive {
            places.place_of[usize::from(expert)] = NO_PLACE;
            places.free.push(place);
        }
        Ok(place)
    }
}

/// The evicted expert, written back (D2H) from the slot into `dst_place` before the slot is
/// overwritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Demote {
    pub(super) expert: u16,
    pub(super) dst_place: u32,
}

/// One slot refill, in stream order: the optional write-back, then the H2D of `src_place`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CopyOp {
    pub(super) slot: u16,
    pub(super) incoming: u16,
    pub(super) src_place: u32,
    pub(super) demote: Option<Demote>,
}

/// The copies of one layer's pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PassCopies {
    pub(super) layer: usize,
    pub(super) distinct: usize,
    pub(super) hits: usize,
    pub(super) ops: Vec<CopyOp>,
}

/// Counters of the passes so far.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BankStats {
    pub(super) passes: u64,
    pub(super) picks: u64,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) demotions: u64,
    /// Passes with three or more misses in one layer: the first two hide behind the resident pass
    /// (`flash-next-moe-graph-ab-measured`).
    pub(super) layers_with_3_plus_misses: u64,
    /// Passes by their miss count, `0..=top_k`.
    pub(super) miss_histogram: Vec<u64>,
}

impl BankStats {
    fn new(top_k: usize) -> Self {
        Self {
            passes: 0,
            picks: 0,
            hits: 0,
            misses: 0,
            demotions: 0,
            layers_with_3_plus_misses: 0,
            miss_histogram: vec![0; top_k + 1],
        }
    }

    pub(super) fn hit_rate(&self) -> f64 {
        if self.picks == 0 {
            1.0
        } else {
            self.hits as f64 / self.picks as f64
        }
    }
}

/// The host owner of the bank: the policy, the host places, the counters.
pub(super) struct ExpertBankState<P: SlotPolicy> {
    geometry: BankGeometry,
    mode: HostMode,
    policy: P,
    places: HostPlaces,
    stats: BankStats,
    poisoned: bool,
}

impl ExpertBankState<ReplaceOnMissLru> {
    /// The bank of `plan`, filled from the popularity profile `counts[layer][expert]` (the static
    /// fill of `TierPlan::hot_set`), then managed by replace-on-miss LRU.
    pub(super) fn new(
        plan: &TierPlan,
        layout: &SlotBankLayout,
        mode: HostMode,
        counts: &[Vec<u64>],
    ) -> Result<Self, EngError> {
        let geometry = BankGeometry::from_plan(plan, layout)?;
        let hot = plan.hot_set(counts)?;
        if hot.table.slots() != geometry.slots {
            return Err(bank_error(
                "the hot set and the bank disagree on the slot count".into(),
            ));
        }
        let rows: Vec<&[i32]> = (0..geometry.layers)
            .map(|layer| hot.table.row(layer))
            .collect();
        let policy = ReplaceOnMissLru::from_fill(geometry, &rows, counts, None)?;
        Self::with_policy(geometry, mode, policy)
    }
}

impl<P: SlotPolicy> ExpertBankState<P> {
    /// A bank around any policy that starts filled (every slot holds an expert).
    pub(super) fn with_policy(
        geometry: BankGeometry,
        mode: HostMode,
        policy: P,
    ) -> Result<Self, EngError> {
        let rows: Vec<&[i32]> = (0..geometry.layers)
            .map(|layer| policy.slot_row(layer))
            .collect();
        if rows.iter().any(|row| row.len() != geometry.experts) {
            return Err(bank_error("the policy has no slot row for a layer".into()));
        }
        for (layer, row) in rows.iter().enumerate() {
            let held = row.iter().filter(|&&slot| slot >= 0).count();
            if held != geometry.slots {
                return Err(bank_error(format!(
                    "layer {layer}: the policy holds {held} experts, the bank has {} slots",
                    geometry.slots
                )));
            }
        }
        let places = HostPlaces::new(&geometry, mode, &rows);
        Ok(Self {
            geometry,
            mode,
            policy,
            places,
            stats: BankStats::new(geometry.top_k),
            poisoned: false,
        })
    }

    pub(super) fn geometry(&self) -> &BankGeometry {
        &self.geometry
    }

    pub(super) fn mode(&self) -> HostMode {
        self.mode
    }

    pub(super) fn places(&self) -> &HostPlaces {
        &self.places
    }

    pub(super) fn stats(&self) -> &BankStats {
        &self.stats
    }

    /// The classifier's slot row of `layer` (upload it after every pass that moved a slot).
    pub(super) fn slot_row(&self, layer: usize) -> &[i32] {
        self.policy.slot_row(layer)
    }

    pub(super) fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16> {
        self.policy.expert_in_slot(layer, slot)
    }

    /// Plans the copies of one decode layer for the router's picks (all of them, hits included).
    pub(super) fn begin_pass(
        &mut self,
        layer: usize,
        picks: &[i32],
    ) -> Result<PassCopies, EngError> {
        self.check_usable()?;
        let plan = self.policy.plan_pass(layer, picks)?;
        self.translate(layer, plan, true)
    }

    /// Admits the experts a prefill chunk used, oldest use first. Not counted as decode traffic.
    pub(super) fn admit(
        &mut self,
        layer: usize,
        experts_by_use: &[u16],
    ) -> Result<PassCopies, EngError> {
        self.check_usable()?;
        let plan = self.policy.admit(layer, experts_by_use)?;
        self.translate(layer, plan, false)
    }

    /// Marks the host model as ahead of or apart from the device: every later pass is refused. The
    /// device owner calls it when the copies of a planned pass failed.
    pub(super) fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Whether the host model may no longer match the device. Readers of the slot row check it
    /// (`DeviceBank::slot_row` refuses once it is set).
    pub(super) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn check_usable(&self) -> Result<(), EngError> {
        if self.poisoned {
            return Err(bank_error(
                "an earlier pass left the host places inconsistent with the policy".into(),
            ));
        }
        Ok(())
    }

    fn translate(
        &mut self,
        layer: usize,
        plan: PassPlan,
        counted: bool,
    ) -> Result<PassCopies, EngError> {
        let mut ops = Vec::with_capacity(plan.moves.len());
        for mv in &plan.moves {
            // Demote first: the write-back takes a free place, and the incoming expert's place
            // frees only after its copy (stream order makes the reuse safe).
            let step = self
                .places
                .demote(layer, mv.outgoing)
                .and_then(|demote| Ok((demote, self.places.promote(layer, mv.incoming)?)));
            let (demote, src_place) = step.inspect_err(|_| self.poisoned = true)?;
            ops.push(CopyOp {
                slot: mv.slot,
                incoming: mv.incoming,
                src_place,
                demote: demote.map(|dst_place| Demote {
                    expert: mv.outgoing,
                    dst_place,
                }),
            });
        }
        if counted {
            let misses = plan.moves.len();
            self.stats.passes += 1;
            self.stats.picks += plan.distinct as u64;
            self.stats.hits += plan.hits as u64;
            self.stats.misses += misses as u64;
            self.stats.demotions += ops.iter().filter(|op| op.demote.is_some()).count() as u64;
            self.stats.layers_with_3_plus_misses += u64::from(misses >= 3);
            let bin = misses.min(self.geometry.top_k);
            self.stats.miss_histogram[bin] += 1;
        }
        Ok(PassCopies {
            layer,
            distinct: plan.distinct,
            hits: plan.hits,
            ops,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The filler rule for the second pass.
// ---------------------------------------------------------------------------------------------

/// The slot ids and weights of the missed-pick launch, after the copies: a missed pick runs on its
/// new slot with the router's weight, every other pick on the filler (slot 0, weight +0.0). The
/// resident pass (`classified.slot_ids`, `classified.weights`) and this one add to the router's
/// weights bit for bit, so the combine sees each pick exactly once.
///
/// Refuses a row the classifier flagged invalid, a router fault, lists of different lengths and a
/// missed expert that still has no slot.
pub(super) fn miss_pass_inputs(
    ids: &[i32],
    weights: &[f32],
    classified: &Classified,
    slot_row_after: &[i32],
) -> Result<(Vec<i32>, Vec<f32>), EngError> {
    if ids.len() != weights.len() || ids.len() != classified.slot_ids.len() {
        return Err(bank_error("the pick lists differ in length".into()));
    }
    if classified.word != 0 {
        return Err(bank_error(
            "the classifier flagged the route invalid".into(),
        ));
    }
    let mut slots = Vec::with_capacity(ids.len());
    let mut out = Vec::with_capacity(ids.len());
    for (&id, &weight) in ids.iter().zip(weights) {
        if id < 0 {
            return Err(bank_error("a router fault has no second pass".into()));
        }
        if classified.misses.iter().any(|&(missed, _)| missed == id) {
            let slot = slot_row_after
                .get(id as usize)
                .copied()
                .filter(|&slot| slot >= 0)
                .ok_or_else(|| {
                    bank_error(format!("missed expert {id} has no slot after the copy"))
                })?;
            slots.push(slot);
            out.push(weight);
        } else {
            slots.push(0);
            out.push(0.0);
        }
    }
    Ok((slots, out))
}

// ---------------------------------------------------------------------------------------------
// The record source.
// ---------------------------------------------------------------------------------------------

/// Where the bytes of an expert record come from (the pack through `SlotBankLayout`, see
/// `bank_source::PackRecordSource`). The device owner reads every record once at build.
pub(in crate::family::flash_next) trait RecordSource {
    /// Writes the record of `expert` in `layer` into `out` (exactly one record).
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError>;
}

// ---------------------------------------------------------------------------------------------
// The device owner.
// ---------------------------------------------------------------------------------------------

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) mod device {
    //! One device buffer of `slots` records and one pinned buffer of host places per layer. The
    //! pinned buffers come from `CudaContext::pinned_buf` (`cudaHostAlloc`, zeroed, 0.4 s per GiB),
    //! the path Flint's store works on under the 8 MiB `RLIMIT_MEMLOCK`; nothing here calls `mlock`.
    //! They are allocated once and held: dropping one synchronizes the device.

    use super::{
        BankGeometry, BankStats, ExpertBankState, HostPinPlan, PassCopies, RecordSource,
        ReplaceOnMissLru,
    };
    use crate::{
        EngError,
        kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf},
    };

    pub(in crate::family::flash_next) struct DeviceBank {
        state: ExpertBankState<ReplaceOnMissLru>,
        slots: Vec<DeviceBuffer>,
        store: Vec<PinnedBuf>,
        stream: DeviceStream,
    }

    impl DeviceBank {
        /// Allocates the buffers, writes the host places and fills every slot from `source`.
        /// Returns when the device holds the fill. `plan` is the host budget the caller admitted
        /// (`HostPinPlan::new`, plus Flint's cgroup-aware `HostBudget::admit_for` on the CUDA path):
        /// it must be for this bank's mode and place count, so a store is never pinned unchecked.
        pub(in crate::family::flash_next) fn build(
            ctx: &CudaContext,
            state: ExpertBankState<ReplaceOnMissLru>,
            source: &dyn RecordSource,
            plan: &HostPinPlan,
        ) -> Result<Self, EngError> {
            let geometry: BankGeometry = *state.geometry();
            let places = state.mode().places_per_layer(&geometry);
            let store_bytes = (geometry.layers as u64)
                .checked_mul(places as u64)
                .and_then(|places| places.checked_mul(geometry.record_bytes as u64));
            if plan.mode != state.mode()
                || plan.places_per_layer != places
                || Some(plan.store_bytes) != store_bytes
            {
                return Err(EngError::BadRequest(format!(
                    "Flash-Next expert bank: the host plan is for {:?} with {} places per layer and \
                     {} store bytes, the bank is {:?} with {places} places and {store_bytes:?} bytes",
                    plan.mode,
                    plan.places_per_layer,
                    plan.store_bytes,
                    state.mode()
                )));
            }
            let stream = ctx.stream()?;
            let mut slots = Vec::with_capacity(geometry.layers);
            let mut store = Vec::with_capacity(geometry.layers);
            let mut record = vec![0u8; geometry.record_bytes];
            for layer in 0..geometry.layers {
                slots.push(ctx.allocate(geometry.slots * geometry.record_bytes)?);
                let mut pinned = ctx.pinned_buf(places * geometry.record_bytes)?;
                for (place, expert) in state.places().occupied(layer) {
                    source.record(layer, expert, &mut record)?;
                    pinned.write(geometry.offset(place as usize), &record)?;
                }
                store.push(pinned);
                for slot in 0..geometry.slots {
                    let expert = state.expert_in_slot(layer, slot).ok_or_else(|| {
                        EngError::BadRequest("a slot of the filled bank is empty".into())
                    })?;
                    source.record(layer, expert, &mut record)?;
                    slots[layer].write_at(&stream, geometry.offset(slot), &record)?;
                }
            }
            stream.synchronize()?;
            Ok(Self {
                state,
                slots,
                store,
                stream,
            })
        }

        /// The host model, for tests only: it still answers after a failed copy, when it is ahead
        /// of the device. Production code reads [`Self::slot_row`], [`Self::stats`] and
        /// [`Self::geometry`].
        #[cfg(test)]
        pub(in crate::family::flash_next) fn state(&self) -> &ExpertBankState<ReplaceOnMissLru> {
            &self.state
        }

        /// The counters of the passes so far (hits, misses, write-backs, miss histogram).
        pub(in crate::family::flash_next) fn stats(&self) -> &BankStats {
            self.state.stats()
        }

        /// The sizes of the bank: layers, experts, top-k, slots per layer and the record size.
        pub(in crate::family::flash_next) fn geometry(&self) -> &BankGeometry {
            self.state.geometry()
        }

        /// Whether a failed copy left the device out of step with the host model. Every later
        /// `step`, `admit`, `slot_row`, `slots` and `read_place` is refused.
        pub(in crate::family::flash_next) fn is_poisoned(&self) -> bool {
            self.state.is_poisoned()
        }

        fn check_usable(&self) -> Result<(), EngError> {
            if self.state.is_poisoned() {
                return Err(EngError::BadRequest(
                    "Flash-Next expert bank: a failed copy left the device out of step with the \
                     host model; drop the bank and build a new one"
                        .into(),
                ));
            }
            Ok(())
        }

        /// The classifier's slot row of `layer` (upload it after every pass that moved a slot).
        /// Refused once the bank is poisoned, so a launch cannot classify against a table the
        /// device does not hold.
        pub(in crate::family::flash_next) fn slot_row(
            &self,
            layer: usize,
        ) -> Result<&[i32], EngError> {
            self.check_usable()?;
            match self.state.slot_row(layer) {
                [] => Err(EngError::BadRequest(format!(
                    "layer {layer} is outside the bank"
                ))),
                row => Ok(row),
            }
        }

        /// The bank buffer of `layer`: slot `s` starts at `s * record_bytes`. Refused once the
        /// bank is poisoned.
        pub(in crate::family::flash_next) fn slots(
            &self,
            layer: usize,
        ) -> Result<&DeviceBuffer, EngError> {
            self.check_usable()?;
            self.slots
                .get(layer)
                .ok_or_else(|| EngError::BadRequest(format!("layer {layer} is outside the bank")))
        }

        /// Queues the bytes of one expert place from the pinned store into `dst` on `stream` (the
        /// prefill ring's copy stream). The host model is not changed.
        /// The pinned host place of an expert's record, for the prefill ring's plan (a
        /// non-resident expert without a place cannot stream and rides the miss pass).
        pub(in crate::family::flash_next) fn place_of(
            &self,
            layer: usize,
            expert: u16,
        ) -> Option<u32> {
            self.state.places().place_of(layer, expert)
        }

        pub(in crate::family::flash_next) fn stream_place(
            &mut self,
            ctx: &CudaContext,
            stream: &DeviceStream,
            layer: usize,
            place: u32,
            dst: DeviceRange<'_>,
        ) -> Result<(), EngError> {
            self.check_usable()?;
            let geometry = self.state.geometry();
            let store = self.store.get_mut(layer).ok_or_else(|| {
                EngError::BadRequest(format!("layer {layer} is outside the bank"))
            })?;
            ctx.copy_from_pinned(stream, store, geometry.offset(place as usize), dst)
        }

        pub(in crate::family::flash_next) fn stream(&self) -> &DeviceStream {
            &self.stream
        }

        /// One decode layer: plans the copies for the router's picks (all of them, hits included),
        /// runs them and returns when they have finished. The only way to move a slot, so the
        /// device cannot lag the host model; if a copy fails the state is poisoned and every later
        /// step is refused. A `ctx` of another adapter than the one that built the bank fails in
        /// the first copy, after the host model has moved, so it poisons the bank too: pass the
        /// context of the build.
        ///
        /// The copies run on the bank's own stream, unordered against the caller's stream, so a
        /// caller with reads of the banked bytes still pending on its own stream must drain it
        /// before every `step` (`SplitMoe::run_rows` does; `run` relies on its caller's drains).
        pub(in crate::family::flash_next) fn step(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            picks: &[i32],
        ) -> Result<PassCopies, EngError> {
            self.check_usable()?;
            let copies = self.state.begin_pass(layer, picks)?;
            self.finish(ctx, copies)
        }

        /// Admits the experts a prefill chunk used (oldest use first) and runs the copies.
        pub(in crate::family::flash_next) fn admit(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            experts_by_use: &[u16],
        ) -> Result<PassCopies, EngError> {
            self.check_usable()?;
            let copies = self.state.admit(layer, experts_by_use)?;
            self.finish(ctx, copies)
        }

        fn finish(
            &mut self,
            ctx: &CudaContext,
            copies: PassCopies,
        ) -> Result<PassCopies, EngError> {
            match self.run_copies(ctx, &copies) {
                Ok(()) => Ok(copies),
                Err(error) => {
                    self.state.poison();
                    Err(error)
                }
            }
        }

        /// Runs the copies of `copies` on the bank's stream, in order, and returns when they have
        /// finished: per op the write-back (D2H) into the host place, then the H2D into the slot.
        fn run_copies(&mut self, ctx: &CudaContext, copies: &PassCopies) -> Result<(), EngError> {
            if copies.ops.is_empty() {
                return Ok(());
            }
            let geometry: BankGeometry = *self.state.geometry();
            let layer = copies.layer;
            let (Some(bank), Some(store)) = (self.slots.get(layer), self.store.get_mut(layer))
            else {
                return Err(EngError::BadRequest(format!(
                    "layer {layer} is outside the bank"
                )));
            };
            let record = geometry.record_bytes;
            for op in &copies.ops {
                let slot = bank.range(geometry.offset(usize::from(op.slot)), record)?;
                if let Some(demote) = op.demote {
                    ctx.copy_to_pinned(
                        &self.stream,
                        bank.range(geometry.offset(usize::from(op.slot)), record)?,
                        store,
                        geometry.offset(demote.dst_place as usize),
                    )?;
                }
                ctx.copy_from_pinned(
                    &self.stream,
                    store,
                    geometry.offset(op.src_place as usize),
                    slot,
                )?;
            }
            store.synchronize(&self.stream)
        }

        /// Reads one host place back (tests): the pinned buffer must have no copy outstanding.
        pub(in crate::family::flash_next) fn read_place(
            &self,
            layer: usize,
            place: u32,
            out: &mut [u8],
        ) -> Result<(), EngError> {
            self.check_usable()?;
            let geometry = self.state.geometry();
            self.store
                .get(layer)
                .ok_or_else(|| EngError::BadRequest(format!("layer {layer} is outside the bank")))?
                .read(geometry.offset(place as usize), out)
        }
    }
}
