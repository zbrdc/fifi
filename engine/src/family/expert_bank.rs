//! The shared host half of the expert-bank seam: the slot-policy contract, the host places,
//! the copy plan of a pass and the counters. Ungated on purpose — plain host gates compile and
//! run the seam's tests without the card feature (knowledge:
//! `native-cuda-gated-tests-invisible-to-featureless-lib-gates`). The Flash-Next bank
//! (`family/flash_next/cuda/bank.rs`) keeps the device owner, the pinned-store budget and its
//! `ReplaceOnMissLru` policy; this module is what a second family's bank can share without
//! adopting Flash-Next's pass protocol: the state types are policy-agnostic, the trait is the
//! four-method contract whose rules are stated there.

use crate::EngError;

pub(crate) const NO_PLACE: u32 = u32::MAX;

pub(crate) fn plan_error(detail: String) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next expert bank: {detail}"))
}

pub(crate) fn bank_error(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next expert bank: {detail}"))
}
/// The sizes of the bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BankGeometry {
    pub(crate) layers: usize,
    pub(crate) experts: usize,
    pub(crate) top_k: usize,
    /// Device slots in every layer.
    pub(crate) slots: usize,
    /// Bytes of one expert record (`SlotBankLayout::record_bytes`, 1,881,600 in the 3.05 pack).
    pub(crate) record_bytes: usize,
}

impl BankGeometry {
    /// Refuses a bank without `top_k + 1` slots (the filler needs one), more slots than experts,
    /// expert numbers that do not fit u16 and a record that is not whole floats.
    pub(crate) fn new(
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

    // The bank the tier plan settled is Flash-Next's construction: `from_plan` lives in
    // `family/flash_next/cuda/bank.rs` (an inherent impl of this struct) so the seam does not
    // depend on the Flash-Next moe modules.

    /// Byte offset of `slot` (or of a host place) in a layer's buffer.
    pub(crate) fn offset(&self, index: usize) -> usize {
        index * self.record_bytes
    }

    pub(crate) fn device_bytes_per_layer(&self) -> u64 {
        self.slots as u64 * self.record_bytes as u64
    }
}

/// Which experts the pinned store holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostMode {
    /// Only the experts that are not in a slot, plus one spare place per layer. An expert that
    /// leaves a slot is written back into the place its successor just left.
    Exclusive,
    /// Every expert of every layer: eviction is free, and the store is 43 GiB.
    Inclusive,
}

impl HostMode {
    pub(crate) fn places_per_layer(self, geometry: &BankGeometry) -> usize {
        match self {
            Self::Exclusive => geometry.experts - geometry.slots + 1,
            Self::Inclusive => geometry.experts,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The slot policy.
// ---------------------------------------------------------------------------------------------

/// One slot changing hands: `incoming` takes `slot` from `outgoing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotMove {
    pub(crate) slot: u16,
    pub(crate) incoming: u16,
    pub(crate) outgoing: u16,
}

/// What a policy decided for one pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct PassPlan {
    /// Distinct experts of the pass.
    pub(crate) distinct: usize,
    pub(crate) hits: usize,
    /// In pick order.
    pub(crate) moves: Vec<SlotMove>,
}

/// The classifier-facing table half of the seam: the slot row a tiered classifier reads and
/// the slot -> expert inverse. A family whose pass protocol differs from Flash-Next's
/// pass copies implements just this half (Q36's `LayerTables` does) without adopting
/// `SlotPolicy`'s copy planning.
pub(crate) trait SlotTable {
    /// The classifier's table of `layer`: the slot of every expert, or -1.
    fn slot_row(&self, layer: usize) -> &[i32];

    /// The expert in `slot` of `layer`.
    fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16>;
}

/// The seam a shared policy fills in. Every method keeps the rules in the module doc.
pub(crate) trait SlotPolicy: SlotTable {
    /// Plans the copies of one layer for the router's picks. `picks` holds every pick of the token
    /// in this layer, hits included (`-1`, the router fault, is skipped; any other id outside the
    /// layer is refused; repeats count once). Atomic: an `Err` changes nothing.
    fn plan_pass(&mut self, layer: usize, picks: &[i32]) -> Result<PassPlan, EngError>;

    /// Admits experts without a router pass, oldest use first (a prefill chunk's experts after the
    /// chunk). The last `slots` distinct experts are kept; the rest are dropped.
    fn admit(&mut self, layer: usize, experts_by_use: &[u16]) -> Result<PassPlan, EngError>;
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
pub(crate) struct HostPlaces {
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

    pub(crate) fn place_of(&self, layer: usize, expert: u16) -> Option<u32> {
        let place = *self.layers.get(layer)?.place_of.get(usize::from(expert))?;
        (place != NO_PLACE).then_some(place)
    }

    /// `(place, expert)` of every expert the store holds in `layer`, by place.
    pub(crate) fn occupied(&self, layer: usize) -> Vec<(u32, u16)> {
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
pub(crate) struct Demote {
    pub(crate) expert: u16,
    pub(crate) dst_place: u32,
}

/// One slot refill, in stream order: the optional write-back, then the H2D of `src_place`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyOp {
    pub(crate) slot: u16,
    pub(crate) incoming: u16,
    pub(crate) src_place: u32,
    pub(crate) demote: Option<Demote>,
}

/// The copies of one layer's pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PassCopies {
    pub(crate) layer: usize,
    pub(crate) distinct: usize,
    pub(crate) hits: usize,
    pub(crate) ops: Vec<CopyOp>,
}

/// Counters of the passes so far.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BankStats {
    pub(crate) passes: u64,
    pub(crate) picks: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) demotions: u64,
    /// Passes with three or more misses in one layer: the first two hide behind the resident pass
    /// (`flash-next-moe-graph-ab-measured`).
    pub(crate) layers_with_3_plus_misses: u64,
    /// Passes by their miss count, `0..=top_k`.
    pub(crate) miss_histogram: Vec<u64>,
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

    pub(crate) fn hit_rate(&self) -> f64 {
        if self.picks == 0 {
            1.0
        } else {
            self.hits as f64 / self.picks as f64
        }
    }
}

/// The host owner of the bank: the policy, the host places, the counters.
pub(crate) struct ExpertBankState<P: SlotPolicy> {
    geometry: BankGeometry,
    mode: HostMode,
    policy: P,
    places: HostPlaces,
    stats: BankStats,
    poisoned: bool,
}
impl<P: SlotPolicy> ExpertBankState<P> {
    /// A bank around any policy that starts filled (every slot holds an expert).
    pub(crate) fn with_policy(
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

    pub(crate) fn geometry(&self) -> &BankGeometry {
        &self.geometry
    }

    pub(crate) fn mode(&self) -> HostMode {
        self.mode
    }

    pub(crate) fn places(&self) -> &HostPlaces {
        &self.places
    }

    pub(crate) fn stats(&self) -> &BankStats {
        &self.stats
    }

    /// The classifier's slot row of `layer` (upload it after every pass that moved a slot).
    pub(crate) fn slot_row(&self, layer: usize) -> &[i32] {
        self.policy.slot_row(layer)
    }

    pub(crate) fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16> {
        self.policy.expert_in_slot(layer, slot)
    }

    /// Plans the copies of one decode layer for the router's picks (all of them, hits included).
    pub(crate) fn begin_pass(
        &mut self,
        layer: usize,
        picks: &[i32],
    ) -> Result<PassCopies, EngError> {
        self.check_usable()?;
        let plan = self.policy.plan_pass(layer, picks)?;
        self.translate(layer, plan, true)
    }

    /// Admits the experts a prefill chunk used, oldest use first. Not counted as decode traffic.
    pub(crate) fn admit(
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
    pub(crate) fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Whether the host model may no longer match the device. Readers of the slot row check it
    /// (`DeviceBank::slot_row` refuses once it is set).
    pub(crate) fn is_poisoned(&self) -> bool {
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
