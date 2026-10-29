//! The residency tables of the expert bank slots.
//!
//! WHAT: the generation/ledger protocol of the per-layer VRAM banks. A slot walks
//! `Free -> Filling -> Ready -> Published -> Draining -> Free`, with `Poisoned` as
//! the dead end of a fill that missed its deadline. WHY: the GPU fill and the CPU
//! serve run asynchronously, and a stale read of a swapped expert corrupts the
//! decode; the per-slot generations make every lookup self-verifying. Events are
//! plain ids and time arrives as a parameter. Driven by `TieredHost`.

use super::TieredConfig;

/// The id of one asynchronous fill event; the caller assigns them (a device
/// stream id or a plain counter) and completes them in `event_complete`.
pub type EventId = u64;

/// Why a reader holds a slot; the kinds count separately, so a drain can wait
/// for exactly the readers it must.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderKind {
    /// A device-side copy reading the slot as its source; short-lived.
    CopySource,
    /// A CPU expert job computing on its bytes; lives to the end of the step.
    CpuJob,
}

/// The lifecycle state of one bank slot. `Free` is the declared default so a
/// zeroed `Slot` is a valid empty slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlotState {
    /// Empty; `begin_fill` may take it.
    #[default]
    Free,
    /// A fill is in flight for its expert; not readable yet.
    Filling,
    /// The fill landed; publication happens only at the token boundary.
    Ready,
    /// Live: readers may acquire it and lookups find it.
    Published,
    /// Unpublished but still read by leftover readers; frees when they finish.
    Draining,
    /// A fill that missed its deadline; never published, only reclaimed.
    Poisoned,
}

/// One bank slot: state, owner, generation and readers. `generation` survives
/// `Free`, so every publication of a slot raises it by one, forever; the zero
/// value of every field is a valid `Free` slot.
#[derive(Clone, Debug, Default, PartialEq)]
struct Slot {
    state: SlotState,
    expert: Option<u16>,
    generation: u64,
    copy_readers: u32,
    cpu_readers: u32,
    event: Option<EventId>,
    deadline_ns: Option<u64>,
}

impl Slot {
    /// The total reader count of both kinds.
    fn readers(&self) -> u32 {
        self.copy_readers + self.cpu_readers
    }
}

/// Why a residency operation failed. Every error leaves the table unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidencyError {
    /// `begin_fill` found no Free slot in `layer`; a victim must first be
    /// unpublished and drained.
    NoFreeSlot { layer: usize },
    /// The expert already owns a slot of `layer` (filling, ready or published).
    AlreadyResident { layer: usize, expert: u16 },
    /// `unpublish` found no published slot holding `expert` in `layer`.
    NotPublished { layer: usize, expert: u16 },
    /// `layer` is outside `0..layers` of the config.
    BadLayer { layer: usize },
    /// `slot` is outside `0..bank_slots_per_layer` of the config.
    BadSlot { layer: usize, slot: usize },
    /// `expert` is outside `0..experts` of the config.
    BadExpert { layer: usize, expert: u16 },
    /// The slot is in a state that forbids the operation.
    BadState {
        layer: usize,
        slot: usize,
        state: SlotState,
    },
    /// `release` would take the `kind` reader count of the slot below zero.
    ReaderUnderflow {
        layer: usize,
        slot: usize,
        kind: ReaderKind,
    },
    /// No filling slot waits on the event: unknown, completed, or poisoned.
    UnknownEvent(EventId),
    /// `check_partition` found a broken invariant; the string names it.
    Partition(String),
}

/// The residency ledger: one table of `bank_slots_per_layer` slots per layer plus
/// the per-layer expert index of the owned slots. All scans run in layer and slot
/// order, so every decision is deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidencyTable {
    cfg: TieredConfig,
    /// The slots, `[layer][slot]`.
    slots: Vec<Vec<Slot>>,
    /// The slot of each expert a layer owns, `[layer][expert]`; set for Filling,
    /// Ready and Published slots, cleared at unpublish and poison time.
    index: Vec<Vec<Option<u16>>>,
    /// The highest generation ever published on a slot, `[layer][slot]`; a
    /// generation that ever falls below it is a partition bug.
    max_generation: Vec<Vec<u64>>,
}

impl ResidencyTable {
    /// Builds an all-Free table for `cfg`; the config should have passed `check`.
    pub fn new(cfg: &TieredConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            slots: vec![vec![Slot::default(); cfg.bank_slots_per_layer]; cfg.layers],
            index: vec![vec![None; cfg.experts]; cfg.layers],
            max_generation: vec![vec![0; cfg.bank_slots_per_layer]; cfg.layers],
        }
    }

    /// Takes the lowest Free slot of `layer` for `expert` and starts a fill under
    /// `event` with the absolute deadline `deadline_ns` (nanoseconds of the
    /// injected clock). WHY: the reservation runs before the copy, so two experts
    /// can never share one slot.
    pub fn begin_fill(
        &mut self,
        layer: usize,
        expert: u16,
        event: EventId,
        deadline_ns: u64,
    ) -> Result<usize, ResidencyError> {
        self.check_layer(layer)?;
        if usize::from(expert) >= self.cfg.experts {
            return Err(ResidencyError::BadExpert { layer, expert });
        }
        if self.index[layer][usize::from(expert)].is_some() {
            return Err(ResidencyError::AlreadyResident { layer, expert });
        }
        let Some(slot) = self.slots[layer]
            .iter()
            .position(|s| s.state == SlotState::Free)
        else {
            return Err(ResidencyError::NoFreeSlot { layer });
        };
        let s = &mut self.slots[layer][slot];
        s.state = SlotState::Filling;
        s.expert = Some(expert);
        s.event = Some(event);
        s.deadline_ns = Some(deadline_ns);
        self.index[layer][usize::from(expert)] = Some(slot as u16);
        Ok(slot)
    }

    /// Completes the fill that waits on `event`: Filling -> Ready. An event no
    /// filling slot waits on (unknown, completed, or poisoned) is refused without
    /// touching the table; with duplicate ids the lowest `(layer, slot)` wins.
    pub fn event_complete(&mut self, event: EventId) -> Result<(), ResidencyError> {
        for slots in &mut self.slots {
            for s in slots {
                if s.state == SlotState::Filling && s.event == Some(event) {
                    s.state = SlotState::Ready;
                    s.event = None;
                    s.deadline_ns = None;
                    return Ok(());
                }
            }
        }
        Err(ResidencyError::UnknownEvent(event))
    }

    /// Publishes every Ready slot (token boundary only): Published, generation
    /// raised by one. Returns the `(layer, expert, slot, generation)` rows in
    /// layer and slot order. WHY: readers must only see experts a whole boundary
    /// agreed on.
    pub fn publish_ready(&mut self) -> Vec<(usize, u16, usize, u64)> {
        let mut published = Vec::new();
        for (layer, slots) in self.slots.iter_mut().enumerate() {
            for (slot, s) in slots.iter_mut().enumerate() {
                if s.state == SlotState::Ready {
                    s.state = SlotState::Published;
                    s.generation += 1;
                    self.max_generation[layer][slot] = s.generation;
                    published.push((layer, s.expert.unwrap_or_default(), slot, s.generation));
                }
            }
        }
        published
    }

    /// Unpublishes the published slot of `expert`: the index entry is gone at
    /// once, so stale lookups miss immediately. The slot Drains until its readers
    /// release it, and is Free the moment the last one does.
    pub fn unpublish(&mut self, layer: usize, expert: u16) -> Result<(), ResidencyError> {
        self.check_layer(layer)?;
        let Some(slot) = self.index[layer]
            .get(usize::from(expert))
            .copied()
            .flatten()
        else {
            return Err(ResidencyError::NotPublished { layer, expert });
        };
        if self.slots[layer][usize::from(slot)].state != SlotState::Published {
            return Err(ResidencyError::NotPublished { layer, expert });
        }
        self.index[layer][usize::from(expert)] = None;
        self.retire(layer, usize::from(slot));
        Ok(())
    }

    /// The `(slot, generation)` of a published `expert` of `layer`; `None` for an
    /// unknown, unpinned or unpublished expert. A generation older than the
    /// returned one can never be a valid read again.
    pub fn lookup(&self, layer: usize, expert: u16) -> Option<(usize, u64)> {
        let slot = usize::from((*self.index.get(layer)?.get(usize::from(expert))?)?);
        let s = &self.slots[layer][slot];
        (s.state == SlotState::Published).then_some((slot, s.generation))
    }

    /// Registers a reader of `kind` on a Published slot. Only a published slot
    /// can be read: a filling or ready slot may still change hands, and a drained
    /// slot may already hold the bytes of the next expert.
    pub fn acquire(
        &mut self,
        layer: usize,
        slot: usize,
        kind: ReaderKind,
    ) -> Result<(), ResidencyError> {
        let s = self.slot_mut(layer, slot)?;
        if s.state != SlotState::Published {
            return Err(ResidencyError::BadState {
                layer,
                slot,
                state: s.state,
            });
        }
        match kind {
            ReaderKind::CopySource => s.copy_readers += 1,
            ReaderKind::CpuJob => s.cpu_readers += 1,
        }
        Ok(())
    }

    /// Drops one reader of `kind`; a Draining slot with no readers left is freed
    /// right here, which is what ends a drain without a timer.
    pub fn release(
        &mut self,
        layer: usize,
        slot: usize,
        kind: ReaderKind,
    ) -> Result<(), ResidencyError> {
        let s = self.slot_mut(layer, slot)?;
        if !matches!(s.state, SlotState::Published | SlotState::Draining) {
            return Err(ResidencyError::BadState {
                layer,
                slot,
                state: s.state,
            });
        }
        let count = match kind {
            ReaderKind::CopySource => &mut s.copy_readers,
            ReaderKind::CpuJob => &mut s.cpu_readers,
        };
        if *count == 0 {
            return Err(ResidencyError::ReaderUnderflow { layer, slot, kind });
        }
        *count -= 1;
        let drained = s.state == SlotState::Draining && s.readers() == 0;
        if drained {
            self.free(layer, slot);
        }
        Ok(())
    }

    /// Poisons every Filling slot whose deadline lies before `now_ns` and returns
    /// the `(layer, slot)` rows in layer and slot order. WHY: a fill that missed
    /// its deadline must stop blocking its slot and must never publish stale
    /// bytes; the poison drops the index entry at once and only
    /// `reclaim_poisoned` frees the slot.
    pub fn expire(&mut self, now_ns: u64) -> Vec<(usize, usize)> {
        let mut poisoned = Vec::new();
        for (layer, slots) in self.slots.iter_mut().enumerate() {
            for (slot, s) in slots.iter_mut().enumerate() {
                if s.state == SlotState::Filling && s.deadline_ns.is_some_and(|d| d < now_ns) {
                    if let Some(expert) = s.expert {
                        self.index[layer][usize::from(expert)] = None;
                    }
                    s.state = SlotState::Poisoned;
                    s.event = None;
                    s.deadline_ns = None;
                    poisoned.push((layer, slot));
                }
            }
        }
        poisoned
    }

    /// Frees a Poisoned slot. WHY: the reclamation is an explicit caller decision,
    /// never a timeout of its own, so a late fill event cannot race it.
    pub fn reclaim_poisoned(&mut self, layer: usize, slot: usize) -> Result<(), ResidencyError> {
        let s = self.slot_mut(layer, slot)?;
        if s.state != SlotState::Poisoned {
            return Err(ResidencyError::BadState {
                layer,
                slot,
                state: s.state,
            });
        }
        self.free(layer, slot);
        Ok(())
    }

    /// Verifies the partition invariants: every Filling/Ready/Published slot owns
    /// its expert and the index agrees both ways (one slot per expert per layer,
    /// no slot with two owners); Free holds no expert while Draining and Poisoned
    /// keep theirs but stay unindexed; only a Filling slot keeps a fill event;
    /// readers sit only on Published or Draining slots; and no generation sits
    /// below the slot's published maximum. `Err(Partition(..))` names the row.
    pub fn check_partition(&self) -> Result<(), ResidencyError> {
        let broken =
            |what: String| -> Result<(), ResidencyError> { Err(ResidencyError::Partition(what)) };
        for (layer, slots) in self.slots.iter().enumerate() {
            for (slot, s) in slots.iter().enumerate() {
                let owned = matches!(
                    s.state,
                    SlotState::Filling | SlotState::Ready | SlotState::Published
                );
                let place = format!("layer {layer} slot {slot} ({:?})", s.state);
                if s.expert.is_some() != (s.state != SlotState::Free) {
                    return broken(format!("{place} does not match its expert {:?}", s.expert));
                }
                if s.state == SlotState::Filling {
                    if s.event.is_none() || s.deadline_ns.is_none() {
                        return broken(format!("{place} lost its fill event"));
                    }
                } else if s.event.is_some() || s.deadline_ns.is_some() {
                    return broken(format!("{place} keeps a fill event"));
                }
                if s.readers() > 0 && !matches!(s.state, SlotState::Published | SlotState::Draining)
                {
                    return broken(format!("{place} holds readers"));
                }
                if s.generation != self.max_generation[layer][slot] {
                    return broken(format!("{place} generation left its maximum"));
                }
                if owned
                    && self.index[layer][usize::from(s.expert.unwrap_or_default())]
                        != Some(slot as u16)
                {
                    return broken(format!("{place} is missing from the index"));
                }
            }
            for (expert, entry) in self.index[layer].iter().enumerate() {
                if let Some(slot) = entry {
                    let s = &self.slots[layer][usize::from(*slot)];
                    let owned = matches!(
                        s.state,
                        SlotState::Filling | SlotState::Ready | SlotState::Published
                    );
                    if !owned || s.expert != Some(expert as u16) {
                        return broken(format!(
                            "layer {layer} expert {expert} is a stale index entry"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// The state of one slot, for the integration wiring and the tests.
    pub fn slot_state(&self, layer: usize, slot: usize) -> Result<SlotState, ResidencyError> {
        self.slot(layer, slot).map(|s| s.state)
    }

    /// The slot row of `(layer, slot)` after range checks.
    fn slot(&self, layer: usize, slot: usize) -> Result<&Slot, ResidencyError> {
        self.check_layer(layer)?;
        self.slots[layer]
            .get(slot)
            .ok_or(ResidencyError::BadSlot { layer, slot })
    }

    fn slot_mut(&mut self, layer: usize, slot: usize) -> Result<&mut Slot, ResidencyError> {
        self.check_layer(layer)?;
        self.slots[layer]
            .get_mut(slot)
            .ok_or(ResidencyError::BadSlot { layer, slot })
    }

    fn check_layer(&self, layer: usize) -> Result<(), ResidencyError> {
        if layer < self.cfg.layers {
            Ok(())
        } else {
            Err(ResidencyError::BadLayer { layer })
        }
    }

    /// Unpublished slot: Draining while readers hold it, Free the moment they are
    /// done. The index entry must already be gone.
    fn retire(&mut self, layer: usize, slot: usize) {
        if self.slots[layer][slot].readers() == 0 {
            self.free(layer, slot);
        } else {
            self.slots[layer][slot].state = SlotState::Draining;
        }
    }

    /// Clears a slot back to Free; the generation survives for the next expert.
    fn free(&mut self, layer: usize, slot: usize) {
        let s = &mut self.slots[layer][slot];
        s.state = SlotState::Free;
        s.expert = None;
        s.event = None;
        s.deadline_ns = None;
        s.copy_readers = 0;
        s.cpu_readers = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::PolicyMode;
    use super::{ReaderKind, ResidencyError, ResidencyTable, SlotState, TieredConfig};

    /// A small ledger config: 2 layers x 6 experts, banks of `slots` slots.
    fn table(slots: usize) -> ResidencyTable {
        let mut cfg = TieredConfig::qwen36(slots, PolicyMode::Adaptive);
        cfg.layers = 2;
        cfg.experts = 6;
        ResidencyTable::new(&cfg)
    }

    /// Fills, completes and publishes `expert` in `layer` with no readers.
    fn publish(table: &mut ResidencyTable, layer: usize, expert: u16, event: u64) -> usize {
        let slot = table.begin_fill(layer, expert, event, u64::MAX).unwrap();
        table.event_complete(event).unwrap();
        table.publish_ready();
        slot
    }

    #[test]
    fn publish_happens_only_at_the_boundary() {
        let mut table = table(2);
        table.begin_fill(0, 3, 1, u64::MAX).unwrap();
        table.event_complete(1).unwrap();
        // Ready, not live: a lookup misses until the boundary publishes it.
        assert_eq!(table.lookup(0, 3), None);
        assert_eq!(table.publish_ready(), &[(0, 3, 0, 1)]);
        assert_eq!(table.lookup(0, 3), Some((0, 1)));
        assert!(table.publish_ready().is_empty(), "nothing ready is left");
    }

    #[test]
    fn generations_increase_on_every_publication_of_a_slot() {
        let mut table = table(1);
        // The one slot cycles experts; each publication raises its generation.
        for (round, expert) in [2u16, 5, 1].into_iter().enumerate() {
            publish(&mut table, 0, expert, round as u64 + 1);
            assert_eq!(table.lookup(0, expert), Some((0, round as u64 + 1)));
            table.unpublish(0, expert).unwrap();
            // No readers, so the unpublish frees the slot at once.
            assert_eq!(table.slot_state(0, 0).unwrap(), SlotState::Free);
        }
    }

    #[test]
    fn a_slot_is_not_reused_while_a_cpu_job_reads_it() {
        let mut table = table(1);
        publish(&mut table, 0, 2, 1);
        table.acquire(0, 0, ReaderKind::CpuJob).unwrap();
        table.unpublish(0, 2).unwrap();
        // Draining under a CPU reader: not free, so no fill can take the slot.
        assert_eq!(table.slot_state(0, 0).unwrap(), SlotState::Draining);
        assert_eq!(
            table.begin_fill(0, 3, 2, u64::MAX),
            Err(ResidencyError::NoFreeSlot { layer: 0 })
        );
        table.release(0, 0, ReaderKind::CpuJob).unwrap();
        assert_eq!(table.slot_state(0, 0).unwrap(), SlotState::Free);
        assert_eq!(table.begin_fill(0, 3, 2, u64::MAX), Ok(0));
    }

    #[test]
    fn only_a_published_slot_can_be_acquired() {
        let mut table = table(1);
        table.begin_fill(0, 1, 1, u64::MAX).unwrap();
        assert_eq!(
            table.acquire(0, 0, ReaderKind::CopySource),
            Err(ResidencyError::BadState {
                layer: 0,
                slot: 0,
                state: SlotState::Filling
            })
        );
        table.event_complete(1).unwrap();
        assert_eq!(
            table.acquire(0, 0, ReaderKind::CopySource),
            Err(ResidencyError::BadState {
                layer: 0,
                slot: 0,
                state: SlotState::Ready
            })
        );
        table.publish_ready();
        table.acquire(0, 0, ReaderKind::CopySource).unwrap();
        // The copy reader keeps the slot draining, but a new reader is refused.
        table.unpublish(0, 1).unwrap();
        assert_eq!(
            table.acquire(0, 0, ReaderKind::CpuJob),
            Err(ResidencyError::BadState {
                layer: 0,
                slot: 0,
                state: SlotState::Draining
            })
        );
    }

    #[test]
    fn a_late_fill_poisons_the_slot_and_never_hangs() {
        let mut table = table(1);
        table.begin_fill(0, 4, 9, 100).unwrap();
        assert_eq!(table.expire(101), &[(0, 0)]);
        assert_eq!(table.slot_state(0, 0).unwrap(), SlotState::Poisoned);
        // Never published, never looked up, and the late event is refused.
        assert!(table.publish_ready().is_empty());
        assert_eq!(table.lookup(0, 4), None);
        assert_eq!(
            table.event_complete(9),
            Err(ResidencyError::UnknownEvent(9))
        );
        table.reclaim_poisoned(0, 0).unwrap();
        assert_eq!(table.slot_state(0, 0).unwrap(), SlotState::Free);
        assert_eq!(table.begin_fill(0, 5, 10, u64::MAX), Ok(0));
    }

    #[test]
    fn a_stale_lookup_is_impossible_after_unpublish() {
        let mut table = table(2);
        publish(&mut table, 1, 4, 1);
        table.acquire(1, 0, ReaderKind::CopySource).unwrap();
        assert_eq!(table.lookup(1, 4), Some((0, 1)));
        table.unpublish(1, 4).unwrap();
        // The map entry is gone at once, even while the slot still drains.
        assert_eq!(table.lookup(1, 4), None);
        assert_eq!(table.slot_state(1, 0).unwrap(), SlotState::Draining);
        table.release(1, 0, ReaderKind::CopySource).unwrap();
        assert_eq!(table.lookup(1, 4), None);
        assert_eq!(table.slot_state(1, 0).unwrap(), SlotState::Free);
    }

    /// The tiny deterministic LCG of the offline tests (rule 5): fixed seeds, no
    /// OS randomness.
    struct Lcg(u32);
    impl Lcg {
        fn below(&mut self, n: u32) -> u32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 8) % n
        }
    }

    #[test]
    fn partition_holds_after_every_operation() {
        let mut cfg = TieredConfig::qwen36(4, PolicyMode::Adaptive);
        cfg.layers = 3;
        cfg.experts = 16;
        let mut table = ResidencyTable::new(&cfg);
        let mut rng = Lcg(0xC0FFEE);
        let (mut now, mut next_event) = (0u64, 1u64);
        for _ in 0..5_000 {
            let before = table.clone();
            let kind = |i: u32| {
                if i.is_multiple_of(2) {
                    ReaderKind::CopySource
                } else {
                    ReaderKind::CpuJob
                }
            };
            let result = match rng.below(7) {
                0 => {
                    next_event += 1;
                    table
                        .begin_fill(
                            rng.below(4) as usize,
                            rng.below(18) as u16,
                            next_event,
                            now + u64::from(rng.below(50)),
                        )
                        .map(|_| ())
                }
                1 => table
                    .event_complete(u64::from(1 + rng.below(next_event as u32)))
                    .map(|_| ()),
                2 => {
                    for &(layer, expert, slot, generation) in &table.publish_ready() {
                        let was = &before.slots[layer][slot];
                        assert_eq!(was.state, SlotState::Ready, "publishes a Ready slot");
                        assert_eq!(was.expert, Some(expert));
                        assert_eq!(generation, was.generation + 1);
                    }
                    Ok(())
                }
                3 => table.unpublish(rng.below(4) as usize, rng.below(18) as u16),
                4 => table.acquire(
                    rng.below(4) as usize,
                    rng.below(5) as usize,
                    kind(rng.below(2)),
                ),
                5 => table.release(
                    rng.below(4) as usize,
                    rng.below(5) as usize,
                    kind(rng.below(2)),
                ),
                _ => {
                    now += 40;
                    for (layer, slot) in table.expire(now) {
                        assert_eq!(table.slot_state(layer, slot).unwrap(), SlotState::Poisoned);
                    }
                    Ok(())
                }
            };
            if let Err(error) = &result {
                assert_eq!(
                    table, before,
                    "an illegal operation changed state: {error:?}"
                );
            }
            table.check_partition().unwrap();
        }
    }
}
