//! The banked EXL3 expert load's family-neutral half: the host store that holds every
//! routed expert's bytes as one contiguous record per member in the exact device-bank
//! layout, so a later miss copy moves a member into a slot as a single DMA; the LRU slot
//! table with the sticky hot set; and the resolver that maps one call's picks to slots and
//! refills the misses. The store is page-locked after the load
//! ([`ExpertStore::pin_all`]) so refills DMA without staging; the pin is best effort, and
//! a host that refuses it serves staged but correct. The admission arithmetic that decides
//! banking stays with the family that owns its settings.

use crate::{
    EngError,
    family::project::cuda::exl3::{Exl3Bank, Exl3MemberStaging, exl3_member_layout},
    kernels::cuda::{
        DeviceBuffer, DeviceEvent, DeviceStream, device_synchronize, host_register, host_unregister,
    },
};

/// The refusal helper of this module: every admission error names its numbers.
fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("EXL3 expert bank: {detail}"))
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1 << 30) as f64)
}

pub(in crate::family) fn per_member_bytes(
    routed: u64,
    layers: usize,
    n_expert: usize,
) -> Result<u64, EngError> {
    if layers == 0 || n_expert == 0 {
        return Err(refuse("the model has no layers or no experts".into()));
    }
    let per_layer = routed / layers as u64;
    if per_layer * layers as u64 != routed {
        return Err(refuse(format!(
            "the routed figure {} does not divide over {layers} layers",
            gib(routed),
        )));
    }
    let per_member = per_layer / n_expert as u64;
    if per_member * n_expert as u64 != per_layer {
        return Err(refuse(
            "the routed figure does not divide evenly over the experts".into(),
        ));
    }
    Ok(per_member)
}

/// Every routed expert of one projection of one layer, one record per member in the exact
/// device-bank layout: member `i`'s `[trellis, scales]` bytes at `i * record`, so a slot
/// fill or miss copy is a single contiguous write.
pub(in crate::family) struct PartStore {
    records: Vec<u8>,
    pub(in crate::family) record: usize,
    trellis: usize,
    scale_stride: usize,
    pinned: bool,
}

impl PartStore {
    pub(in crate::family) fn new(shape: [usize; 2], bits_twice: u8, members: usize) -> Self {
        let (trellis, scale_stride) = exl3_member_layout(shape, bits_twice);
        let record = trellis + scale_stride;
        Self {
            records: vec![0; record * members],
            record,
            trellis,
            scale_stride,
            pinned: false,
        }
    }

    /// Writes member `member`'s staged bytes as its record. The length checks are the
    /// store's own rate guard: a member whose staged bytes disagree with the first member's
    /// strides is refused, not mis-laid.
    pub(in crate::family) fn put(
        &mut self,
        member: usize,
        staged: &Exl3MemberStaging,
    ) -> Result<(), EngError> {
        if staged.trellis.len() != self.trellis || staged.scales.len() != self.scale_stride {
            return Err(EngError::Checkpoint(format!(
                "EXL3 expert bank: member {member} stages {} trellis and {} scale bytes, \
                 the bank layout holds {} and {}",
                staged.trellis.len(),
                staged.scales.len(),
                self.trellis,
                self.scale_stride,
            )));
        }
        let at = member * self.record;
        self.records[at..at + self.trellis].copy_from_slice(&staged.trellis);
        self.records[at + self.trellis..at + self.record].copy_from_slice(&staged.scales);
        Ok(())
    }

    /// Member `member`'s record bytes, the source of a miss copy.
    pub(in crate::family) fn member(&self, member: usize) -> &[u8] {
        let at = member * self.record;
        &self.records[at..at + self.record]
    }

    pub(in crate::family) fn bytes(&self) -> usize {
        self.records.len()
    }

    /// Page-locks the record storage, best effort: a store that fails to pin stays
    /// pageable (the refills still work, only staged). The vector never moves after
    /// the load, so the registered pointer stays the buffer's own. Returns whether
    /// the pin took.
    pub(in crate::family) fn pin(&mut self) -> bool {
        if host_register(&self.records).is_ok() {
            self.pinned = true;
        }
        self.pinned
    }
}

impl Drop for PartStore {
    fn drop(&mut self) {
        // The page lock must go before the Vec free; a failure leaves the lock to the
        // process exit, which the CUDA runtime cleans up. Unregistering while a refill
        // DMA still reads the range is undefined, so drain the device first; live drop
        // paths reach this with the stream already drained, and the cost is irrelevant
        // on a drop path.
        if self.pinned {
            let _ = device_synchronize();
            let _ = host_unregister(&self.records);
        }
    }
}

/// Every routed expert of every layer: `layers` entries of the `[gate, up, down]` parts.
pub(crate) struct ExpertStore {
    layers: Vec<[PartStore; 3]>,
}

impl ExpertStore {
    pub(crate) fn with_layers(layers: usize) -> Self {
        Self {
            layers: Vec::with_capacity(layers),
        }
    }

    /// Appends one layer's three part stores; the load fills them layer by layer.
    pub(in crate::family) fn push_layer(&mut self, parts: [PartStore; 3]) {
        self.layers.push(parts);
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, layer: usize) -> &[PartStore; 3] {
        &self.layers[layer]
    }

    /// Page-locks every member's storage (see [`PartStore::pin`]); true when every
    /// member pinned.
    pub(crate) fn pin_all(&mut self) -> bool {
        self.layers
            .iter_mut()
            .flat_map(|parts| parts.iter_mut())
            .all(|part| part.pin())
    }

    /// The whole store's bytes, for the start line and the admission's echo.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|parts| parts.iter().map(PartStore::bytes).sum::<usize>())
            .sum::<usize>() as u64
    }
}

/// The least-recently-used slot table of one layer: which expert sits in which slot of the
/// layer's three banks. Pure — the device copies live one level up, so the policy is testable
/// without a GPU.
///
/// One slot index addresses member `i` of all three parts' device banks and of the store's
/// part stores alike, so the table is one array per layer, not per part.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerSlots {
    experts: usize,
    slots: usize,
    /// How many of the layer's hottest experts (by decayed use count) stay pinned: a miss
    /// evicts a non-hot slot first, so repeatedly routed experts stop refilling. Zero is
    /// plain LRU. Clamped to `slots - 1`; ties at the threshold can still mark more experts
    /// hot than that, and the victim's fallback — not the clamp — guarantees progress.
    sticky: usize,
    /// Expert -> slot, [`NO_SLOT`] when absent.
    pub(in crate::family) slot_of: Vec<i32>,
    /// Slot -> expert, [`EMPTY`] when the slot never filled.
    pub(in crate::family) expert_of: Vec<u16>,
    last_used: Vec<u64>,
    /// Decayed use count per expert; the sticky set is its top-`sticky`. Halved every
    /// [`DECAY_CALLS`] calls so the set tracks a drifting workload.
    counts: Vec<u32>,
    /// Cached top-`sticky` membership, recomputed at each decay.
    hot: Vec<bool>,
    pub(in crate::family) tick: u64,
}

/// Calls between use-count halvings. 64 calls ≈ 8 tokens of one layer's routing at top-8:
/// the replay study on a captured trace (384-token greedy decode) put the adaptive
/// policy's best miss reduction (9.5% fewer than LRU at k=48) at this interval; longer
/// intervals only lag a drifting workload, shorter ones thrash the hot set.
pub(in crate::family) const DECAY_CALLS: u64 = 64;

/// The absent marker of the slot -> expert direction.
pub(in crate::family) const EMPTY: u16 = u16::MAX;
/// The absent marker of the expert -> slot direction: the slot row the shared seam
/// publishes (`SlotTable::slot_row`) is exactly this array.
pub(in crate::family) const NO_SLOT: i32 = -1;

impl LayerSlots {
    pub(in crate::family) fn new(experts: usize, slots: usize, sticky: usize) -> Self {
        let sticky = sticky.min(slots.saturating_sub(1));
        // The decay reads `sorted[self.sticky - 1]`; `decide()` keeps slots <= experts,
        // pinned here so a future caller change cannot move the panic boundary.
        debug_assert!(sticky <= experts);
        Self {
            experts,
            slots,
            sticky,
            slot_of: vec![NO_SLOT; experts],
            expert_of: vec![EMPTY; slots],
            last_used: vec![0; slots],
            counts: vec![0; experts],
            hot: vec![false; experts],
            tick: 0,
        }
    }

    /// Halves every use count and rebuilds the hot set from the survivors: the `sticky`
    /// largest counts, ties included. Called on the decay boundary only; with no sticky
    /// set the counts are dead weight and the halving is skipped.
    fn decay(&mut self) {
        if self.sticky == 0 {
            return;
        }
        for count in &mut self.counts {
            *count >>= 1;
        }
        let mut sorted = self.counts.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let threshold = sorted[self.sticky - 1];
        for (expert, &count) in self.counts.iter().enumerate() {
            self.hot[expert] = count >= threshold && count > 0;
        }
    }

    /// Maps `picks` (this layer's routed experts of one token, distinct) to slot indices,
    /// evicting for every miss the least-recently-used slot that no pick of THIS call uses —
    /// all picks must be resident together, because the launches of one block read them all.
    /// Returns the slots in pick order and, for each miss, the slot whose device bytes the
    /// caller must refill from the store.
    pub(in crate::family) fn assign(
        &mut self,
        picks: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>), EngError> {
        // Atomic, the seam's rule: an Err changes nothing. The pass runs on a scratch
        // clone (the table is KiBs — noise next to the three record copies one miss
        // issues) and only lands on Ok, so a mid-stream refusal leaves the live table
        // exactly as the call found it.
        let mut scratch = self.clone();
        let outcome = scratch.assign_steps(picks)?;
        *self = scratch;
        Ok(outcome)
    }

    fn assign_steps(&mut self, picks: &[usize]) -> Result<(Vec<usize>, Vec<usize>), EngError> {
        self.tick += 1;
        if self.tick.is_multiple_of(DECAY_CALLS) {
            self.decay();
        }
        let mut slots = Vec::with_capacity(picks.len());
        let mut misses = Vec::new();
        for &expert in picks {
            if expert >= self.experts {
                return Err(EngError::BadRequest(format!(
                    "EXL3 expert bank: pick {expert} exceeds the {} experts of the layer",
                    self.experts
                )));
            }
            if self.sticky > 0 {
                self.counts[expert] = self.counts[expert].saturating_add(1);
            }
            let slot = match self.slot_of[expert] {
                s if (0..self.slots as i32).contains(&s) => s as usize,
                _ => self.victim(&slots)?,
            };
            let miss = self.slot_of[expert] == NO_SLOT;
            if miss {
                let evicted = self.expert_of[slot];
                if evicted != EMPTY {
                    self.slot_of[usize::from(evicted)] = NO_SLOT;
                }
                self.expert_of[slot] = u16::try_from(expert).map_err(|_| {
                    EngError::BadRequest(format!(
                        "EXL3 expert bank: expert {expert} does not fit the slot table"
                    ))
                })?;
                self.slot_of[expert] = i32::try_from(slot).expect("slots fit i32");
                misses.push(slot);
            }
            slots.push(slot);
        }
        for &slot in &slots {
            self.last_used[slot] = self.tick;
        }
        Ok((slots, misses))
    }

    /// The next slot for a miss: an empty one, else the least-recently-used slot no pick of
    /// this call took. A slot holding a hot (sticky) expert is skipped unless every
    /// untouched slot is hot — pinning yields to progress, never to a refusal.
    fn victim(&self, taken: &[usize]) -> Result<usize, EngError> {
        if let Some(slot) = (0..self.slots).find(|&slot| self.expert_of[slot] == EMPTY) {
            return Ok(slot);
        }
        let all = |hot: bool| {
            (0..self.slots)
                .filter(|&slot| {
                    !taken.contains(&slot) && self.hot[usize::from(self.expert_of[slot])] == hot
                })
                .min_by_key(|&slot| self.last_used[slot])
        };
        all(false).or_else(|| all(true)).ok_or_else(|| {
            EngError::BadRequest(format!(
                "EXL3 expert bank: {} slots cannot hold the {} picks of one call \
                     (a decode token, or a prefill sub-block's rows together)",
                self.slots,
                taken.len() + 1
            ))
        })
    }

    /// The expert a slot holds (for the store read of a miss copy).
    pub(in crate::family) fn expert_of(&self, slot: usize) -> usize {
        usize::from(self.expert_of[slot])
    }

    /// The torn-refill response: drop `slots` from both directions of the table so the
    /// next call treats them as misses and refills. A failed record write leaves a
    /// slot's bytes untrustworthy while the table still names its incoming expert, and
    /// restoring the pre-call table would be wrong — an evicted expert's bytes may be
    /// half overwritten — so the slot simply reads absent in both directions.
    /// The seam's read of the expert-to-slot row (`SlotTable::slot_row`): exactly the
    /// `slot_of` array, `NO_SLOT` where a expert is absent.
    pub(in crate::family) fn slot_row(&self) -> &[i32] {
        self.slot_of.as_slice()
    }

    /// The seam's read of the slot-to-expert direction: `None` where the slot never filled.
    pub(in crate::family) fn expert_in_slot(&self, slot: usize) -> Option<u16> {
        let expert = self.expert_of.get(slot).copied()?;
        (expert != EMPTY).then_some(expert)
    }

    pub(in crate::family) fn invalidate(&mut self, slots: &[usize]) {
        for &slot in slots {
            let expert = self.expert_of[slot];
            if expert != EMPTY {
                self.slot_of[usize::from(expert)] = NO_SLOT;
            }
            self.expert_of[slot] = EMPTY;
        }
    }
}

/// One routed bank's device upload target: the record buffer a slot's bytes land in, and
/// the record stride of [`exl3_member_layout`].
struct BankTarget<'w> {
    bank: &'w DeviceBuffer,
    pub(in crate::family) record: usize,
}

/// The upload target of one bank, by shared layout (a plain fn so the borrowed buffers keep
/// the resolver's lifetime).
fn target_of<'w>(bank: &'w Exl3Bank) -> BankTarget<'w> {
    BankTarget {
        bank: bank.bank_buffer(),
        record: bank.record_bytes(),
    }
}

/// The banked decode state: every layer's slot table with its three banks' upload targets,
/// and the host store the misses copy from. The banks the tables point into are the same
/// banks the block's views read, resolved once at the start.
pub(crate) struct Exl3SlotResolver<'w> {
    layers: Vec<(LayerSlots, [BankTarget<'w>; 3])>,
    store: ExpertStore,
    hits: u64,
    misses: u64,
    /// The overlap pair (`Q36_CUDA_BANK_OVERLAP=1`): a copy stream that carries miss refills
    /// beside the launch stream, and the event the launches reading the refilled slots wait.
    /// Only [`LayerResolver::resolve_split`] uses it; a plain [`LayerResolver::resolve`]
    /// never does, so a caller that cannot split its launches keeps today's ordering.
    overlap: Option<(DeviceStream, DeviceEvent)>,
}

impl<'w> Exl3SlotResolver<'w> {
    /// Builds the resolver from one resolved [`Exl3MoeWeights`](crate::family::exl3::moe::Exl3MoeWeights)
    /// per layer (banked: banks of `slots` members) and the matching store, so table, banks and
    /// store agree on the layer order and the part order (gate, up, down).
    pub(crate) fn new(
        store: ExpertStore,
        blocks: &[crate::family::exl3::moe::Exl3MoeWeights<'w>],
        experts: usize,
        sticky: usize,
    ) -> Result<Self, EngError> {
        if store.layers() != blocks.len() {
            return Err(EngError::MemoryPlan(format!(
                "EXL3 expert bank: the store holds {} layers, the step resolved {}",
                store.layers(),
                blocks.len()
            )));
        }
        let layers = blocks
            .iter()
            .map(|block| {
                let slots = block.gate.count();
                for bank in [block.gate, block.up, block.down] {
                    if bank.count() != slots {
                        return Err(EngError::MemoryPlan(format!(
                            "EXL3 expert bank: a layer's banks hold {} gate, {} up and {} down \
                             slots",
                            slots,
                            block.up.count(),
                            block.down.count()
                        )));
                    }
                }
                Ok((
                    LayerSlots::new(experts, slots, sticky),
                    [
                        target_of(block.gate),
                        target_of(block.up),
                        target_of(block.down),
                    ],
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            store,
            hits: 0,
            misses: 0,
            overlap: None,
        })
    }

    /// Arms the overlap pair of [`LayerResolver::resolve_split`]: one copy stream for miss
    /// refills, one event to gate the launches that read them. The start of a banked session
    /// whose environment asked for it arms this once, before the first token; a later arm
    /// stays safe (every resolve_split sits at the readback quiescence anchor), but the
    /// start never does it and callers should not rely on mid-session arming.
    pub(crate) fn enable_overlap(
        &mut self,
        context: &crate::kernels::cuda::CudaContext,
    ) -> Result<(), EngError> {
        if self.overlap.is_some() {
            return Ok(());
        }
        self.overlap = Some((context.stream()?, context.event()?));
        Ok(())
    }

    /// Whether [`LayerResolver::resolve_split`] refills on the copy stream (the start armed
    /// the pair from `Q36_CUDA_BANK_OVERLAP=1`); the MoE blocks read this to split their
    /// launches around the refills.
    pub(crate) fn overlap_armed(&self) -> bool {
        self.overlap.is_some()
    }

    /// Drains the overlap copy stream (see [`LayerResolver::fence_copy_stream`]); a no-op
    /// without the pair armed.
    pub(crate) fn fence_copy_stream(&mut self) {
        if let Some((copy, _)) = &self.overlap
            && let Err(error) = copy.synchronize()
        {
            eprintln!("q36 cuda: overlap fence failed: {error:?}");
        }
    }

    /// One layer's resolver view, what the MoE block hooks take.
    pub(crate) fn layer(&mut self, layer: usize) -> LayerResolver<'_, 'w> {
        LayerResolver {
            resolver: self,
            layer,
        }
    }

    /// The slot count of every layer's bank (the prefill sub-block arithmetic).
    pub(crate) fn slots(&self) -> usize {
        self.layers
            .first()
            .map(|(slots, _)| slots.slots)
            .unwrap_or(0)
    }

    /// The hit/miss counts of the session, for the start-line echo and the report.
    pub(crate) fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// One layer's slice of the resolver: the block hooks need the layer bound once, not threaded
/// through every launch builder.
pub(crate) struct LayerResolver<'r, 'w> {
    resolver: &'r mut Exl3SlotResolver<'w>,
    layer: usize,
}

/// The slot for every pick, the picks gated on a refill of their slot, and the refill's event
/// when one fired (absent when nothing refilled).
type ResolveOutcome<'a> = Result<(Vec<usize>, Vec<usize>, Option<&'a DeviceEvent>), EngError>;

impl LayerResolver<'_, '_> {
    /// Whether this resolver's calls refill on the copy stream (the arming of the resolver
    /// it borrows); the launch builders read this to split around the refills. Every reader
    /// lives in test or device-harness code today, so the passthrough itself is test-gated.
    #[cfg(test)]
    pub(crate) fn overlap_armed(&self) -> bool {
        self.resolver.overlap_armed()
    }

    /// Drains the overlap copy stream. The error path out of a split MoE block queues no
    /// wait, so its refills keep writing with no fence; a later token's hit on a refilling
    /// slot could read torn bytes, because only the launch stream is drained at the next
    /// readback. Every block that resolves with a split fences on its way out with an error.
    pub(crate) fn fence_copy_stream(&mut self) {
        self.resolver.fence_copy_stream();
    }

    /// Maps one call's picks (a decode token's, a prefill sub-block's, or a K-row window
    /// row's) to slots, refilling the missed slots' device bytes from the store. The copies
    /// are enqueued on `stream` (`write_at_enqueue`; the store is page-locked after the load):
    /// the readback that produced the picks has already waited for the stream, and the
    /// launches this return feeds queue behind the copies on the same stream.
    pub(crate) fn resolve(
        &mut self,
        stream: &DeviceStream,
        picks: &[usize],
    ) -> Result<Vec<usize>, EngError> {
        Ok(self.resolve_mapped(stream, picks, false)?.0)
    }

    /// [`Self::resolve`] for a caller that splits its launches around the refills
    /// (`Q36_CUDA_BANK_OVERLAP=1`): the missed slots refill on the resolver's copy stream
    /// instead, the event lands in the return, and the caller queues the launches that read
    /// only already-resident slots before the wait (their compute hides the refill DMA), the
    /// launches that read a refilled slot after it. Without the pair armed this is exactly
    /// [`Self::resolve`] with an absent event.
    ///
    /// Safety rests on the readback that produced the picks: it waited the launch stream,
    /// which had waited every earlier copy event before the launches it read back, so both
    /// streams are quiescent here and a copy-stream refill cannot overtake a queued reader.
    pub(crate) fn resolve_split(
        &mut self,
        stream: &DeviceStream,
        picks: &[usize],
        overlap: bool,
    ) -> ResolveOutcome<'_> {
        let (slots, refilled, event) = self.resolve_mapped(stream, picks, overlap)?;
        // A pick gates on the event when its SLOT refilled this call, not when the pick
        // itself reported the miss: a duplicate pick of a just-refilled expert reads the
        // same refilling bytes and must wait with the miss.
        let gated = (0..picks.len())
            .filter(|&p| refilled.contains(&slots[p]))
            .collect();
        Ok((slots, gated, event))
    }

    /// The shared body: `assign`, the refills, the stats. With `split`, the refills go on
    /// the overlap copy stream and the event is recorded on it (absent when nothing
    /// refilled — no wait orders anything); without, they ride `stream` as they always did.
    fn resolve_mapped(
        &mut self,
        stream: &DeviceStream,
        picks: &[usize],
        split: bool,
    ) -> ResolveOutcome<'_> {
        #[cfg(test)]
        let began = std::time::Instant::now();
        let (table, parts) = &mut self.resolver.layers[self.layer];
        let store = self.resolver.store.layer(self.layer);
        let (slots, misses) = table.assign(picks)?;
        let miss_count = misses.len();
        let copy = if split && miss_count > 0 {
            self.resolver.overlap.as_ref()
        } else {
            None
        };
        let refill = copy.map_or(stream, |(copy, _)| copy);
        for op in refill_plan(table, &misses, parts_strides(parts)) {
            let record = store[op.part].member(op.expert);
            // On the launch stream (no split): the copy races the launches queued
            // after it instead of a device-wide sync per miss, and the next layer's
            // readback (already a stream wait) covers its completion. On the copy
            // stream (split): the event below gates the launches that read the slot;
            // both streams were quiescent at the readback that produced the picks.
            // The store is pinned, so the record must stay untouched until the
            // streams drain; the store is immutable post-load and
            // process-long-lived, which is that invariant.
            if let Err(err) = parts[op.part]
                .bank
                .write_at_enqueue(refill, op.offset(), record)
            {
                // The write failed mid-plan: the enqueued copies of this call's misses
                // cannot be trusted to be complete, and the table still names their
                // incoming experts. Both directions of every missed slot read absent
                // again, so the next call re-misses them and refills over whatever
                // bytes landed.
                table.invalidate(&misses);
                return Err(err);
            }
        }
        let event = match (copy, miss_count > 0) {
            (Some((_, event)), true) => {
                if let Err(err) = refill.record(event) {
                    // The writes were enqueued, but the gate the launches wait on was
                    // not: treat the misses as torn, as above, so correctness does not
                    // depend on the caller's fence succeeding.
                    table.invalidate(&misses);
                    return Err(err);
                }
                Some(event)
            }
            _ => None,
        };
        self.resolver.hits += (picks.len() - miss_count) as u64;
        self.resolver.misses += miss_count as u64;
        #[cfg(test)]
        if trace::armed() {
            trace::record(self.layer, picks);
        }
        #[cfg(test)]
        probe::resolve(began.elapsed());
        Ok((slots, misses, event))
    }
}

/// One record copy of a layer refill: the slot that refills, the expert now in it, the
/// part's buffer index, and the record stride of that part; the write offset is the slot
/// times the stride ([`RefillOp::offset`]).
pub(in crate::family) struct RefillOp {
    slot: usize,
    pub(in crate::family) expert: usize,
    pub(in crate::family) part: usize,
    pub(in crate::family) record: usize,
}

impl RefillOp {
    pub(in crate::family) fn offset(&self) -> usize {
        self.slot * self.record
    }
}

/// The record strides of the three parts — the one place the plan's stride inputs are
/// derived. `resolve_mapped` passes exactly this to [`refill_plan`]; the pick-replay
/// harness mirrors the values (`PART_LEN`) against this derivation.
fn parts_strides(parts: &[BankTarget<'_>; 3]) -> [usize; 3] {
    [parts[0].record, parts[1].record, parts[2].record]
}

/// The refill plan of one layer's misses — in miss order, then part order — as the exact
/// copies [`LayerResolver::resolve_mapped`] issues, parameterized by the three parts'
/// record strides. Pure host data over the table: the pick-replay harness records this
/// instead of recomputing the refill semantics, so a change in the engine's refill order
/// fails the golden compare.
pub(in crate::family) fn refill_plan(
    table: &LayerSlots,
    misses: &[usize],
    strides: [usize; 3],
) -> Vec<RefillOp> {
    let mut ops = Vec::with_capacity(misses.len() * 3);
    for &slot in misses {
        let expert = table.expert_of(slot);
        for (part, &record) in strides.iter().enumerate() {
            ops.push(RefillOp {
                slot,
                expert,
                part,
                record,
            });
        }
    }
    ops
}

/// Test-only pick-trace capture: the admission-policy study replays real routing through
/// offline slot policies. `arm()` starts capture, `dump` writes it.
#[cfg(test)]
pub(crate) mod trace {
    use std::{
        cell::RefCell,
        sync::atomic::{AtomicBool, Ordering},
    };

    static ARMED: AtomicBool = AtomicBool::new(false);
    thread_local! {
        static CALLS: RefCell<Vec<(usize, Vec<u32>)>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn arm() {
        ARMED.store(true, Ordering::Relaxed);
        CALLS.with(|c| c.borrow_mut().clear());
    }

    pub(crate) fn armed() -> bool {
        ARMED.load(Ordering::Relaxed)
    }

    pub(crate) fn record(layer: usize, picks: &[usize]) {
        CALLS.with(|c| {
            c.borrow_mut()
                .push((layer, picks.iter().map(|&e| e as u32).collect()));
        });
    }

    /// One line per resolve call: `layer pick pick ...`.
    pub(crate) fn dump(path: &std::path::Path) -> std::io::Result<usize> {
        use std::io::Write;
        let mut lines = 0;
        CALLS.with(|c| {
            let calls = c.borrow();
            let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
            for (layer, picks) in calls.iter() {
                write!(out, "{layer}")?;
                for &e in picks {
                    write!(out, " {e}")?;
                }
                writeln!(out)?;
                lines += 1;
            }
            out.flush()
        })?;
        ARMED.store(false, Ordering::Relaxed);
        Ok(lines)
    }
}

#[cfg(test)]
#[cfg(test)]
pub(crate) mod probe {
    //! Test-only timing of the banked readback and resolve, so a decode probe can report
    //! what share of a token the host round trips take.
    use std::cell::Cell;

    thread_local! {
        static READBACK: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
        static RESOLVE: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    pub(crate) fn readback(spent: std::time::Duration) {
        READBACK.with(|cell| {
            let (calls, nanos) = cell.get();
            cell.set((calls + 1, nanos + spent.as_nanos() as u64));
        });
    }

    pub(crate) fn resolve(spent: std::time::Duration) {
        RESOLVE.with(|cell| {
            let (calls, nanos) = cell.get();
            cell.set((calls + 1, nanos + spent.as_nanos() as u64));
        });
    }

    /// The accumulated `(calls, nanos)` of each phase since the last take.
    pub(crate) fn take() -> ((u64, u64), (u64, u64)) {
        (
            READBACK.with(|cell| cell.replace((0, 0))),
            RESOLVE.with(|cell| cell.replace((0, 0))),
        )
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_store_lays_members_as_contiguous_records() {
        let shape = [64usize, 32];
        let (stride, scale_stride) = exl3_member_layout(shape, 8);
        let mut store = PartStore::new(shape, 8, 4);
        let member = |trellis_byte: u8| Exl3MemberStaging {
            trellis: vec![trellis_byte; stride],
            scales: vec![trellis_byte + 1; scale_stride],
        };
        for i in 0..4u8 {
            store.put(i as usize, &member(i)).unwrap();
        }
        for i in 0..4u8 {
            let record = store.member(i as usize);
            // One contiguous record per member: its trellis then its scales, adjacent.
            assert_eq!(record.len(), stride + scale_stride);
            assert!(record[..stride].iter().all(|&byte| byte == i));
            assert!(record[stride..].iter().all(|&byte| byte == i + 1));
        }
        // A member staged at another rate is refused, not mis-laid.
        let error = format!(
            "{:?}",
            store
                .put(
                    2,
                    &Exl3MemberStaging {
                        trellis: vec![0; stride - 1],
                        scales: vec![0; scale_stride],
                    },
                )
                .unwrap_err()
        );
        assert!(error.contains("member 2"), "{error}");
    }

    /// The first picks fill the empty slots in order; a repeat of the same picks is all hits
    /// with the same slots and no misses.
    #[test]
    fn slot_table_fills_in_order_then_hits() {
        let mut table = LayerSlots::new(64, 8, 0);
        let picks = [3usize, 41, 7, 19];
        let (slots, misses) = table.assign(&picks).unwrap();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert_eq!(misses, [0, 1, 2, 3]);
        let (slots, misses) = table.assign(&picks).unwrap();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert!(misses.is_empty());
    }

    /// A miss evicts the least-recently-used slot no pick of the call uses: touching expert 3
    /// again makes expert 41's slot the victim, not 3's.
    #[test]
    fn slot_table_evicts_least_recently_used() {
        let mut table = LayerSlots::new(64, 8, 0);
        table.assign(&[3, 41, 5, 6, 7, 8, 9, 10]).unwrap();
        table.assign(&[3]).unwrap(); // touch 3; 41 is now the least-recently-used
        let (slots, misses) = table.assign(&[50]).unwrap();
        assert_eq!(slots, [1], "the victim is 41's slot");
        assert_eq!(misses, [1]);
        // 41 must miss again; it lands in some other slot.
        let (slots, misses) = table.assign(&[41]).unwrap();
        assert_eq!(misses.len(), 1);
        assert_ne!(
            slots[0], 1,
            "41 cannot return to the slot it was evicted from"
        );
    }

    /// Sticky pinning: after a hot set emerges (enough calls to cross the decay boundary),
    /// a cold sweep cannot evict it, while plain LRU loses the same experts and re-misses.
    #[test]
    fn slot_table_sticky_keeps_the_hot_set_through_a_cold_sweep() {
        fn sweep_misses(sticky: usize) -> usize {
            let mut table = LayerSlots::new(64, 8, sticky);
            // The hot set: experts 0-3, routed twice per call, long enough that the
            // repeated halvings reach the count steady state and promote them past any
            // cold count.
            for _ in 0..600 {
                table.assign(&[0, 1, 2, 3]).unwrap();
                table.assign(&[0, 1, 2, 3]).unwrap();
            }
            // The cold sweep: distinct experts the hot calls never touch.
            let mut misses = 0;
            for cold in 20..44 {
                misses += table.assign(&[cold]).unwrap().1.len();
            }
            // The hot set returns: sticky keeps it resident.
            misses += table.assign(&[0, 1, 2, 3]).unwrap().1.len();
            misses
        }
        let lru = sweep_misses(0);
        let sticky = sweep_misses(4);
        assert_eq!(lru, 24 + 4, "plain LRU re-misses the whole hot set");
        assert_eq!(sticky, 24, "the sweep misses, the hot set does not");
    }

    /// Pinning yields to progress: when ties at the decay threshold mark every resident
    /// expert hot (sticky=1, eight equally-counted residents), a new pick still resolves —
    /// the victim falls back to plain LRU instead of refusing.
    #[test]
    fn slot_table_sticky_falls_back_to_lru_when_every_slot_is_hot() {
        let mut table = LayerSlots::new(64, 8, 1);
        // Eight equal-count residents: the tie marks them all hot at the first decay.
        for _ in 0..128 {
            table.assign(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        }
        assert!(
            table.hot[..8].iter().all(|&hot| hot),
            "the tie marks all hot"
        );
        // A cold pick must still seat: the fallback evicts a hot slot rather than refusing.
        let (slots, misses) = table.assign(&[40]).unwrap();
        assert_eq!(misses.len(), 1);
        assert_eq!(slots.len(), 1);
    }

    /// A token whose picks are all new may replace every resident: the victims hold old
    /// experts, which no new pick uses. A token with more picks than slots is refused.
    #[test]
    fn slot_table_replaces_everything_then_refuses_overload() {
        let mut table = LayerSlots::new(64, 4, 0);
        table.assign(&[0, 1, 2, 3]).unwrap();
        let (slots, misses) = table.assign(&[10, 11, 12, 13]).unwrap();
        assert_eq!(misses.len(), 4);
        assert_eq!(slots.len(), 4);
        assert!((0..4).all(|slot| table.expert_of(slot) >= 10));
        let error = format!("{:?}", table.assign(&[20, 21, 22, 23, 24]).unwrap_err());
        assert!(error.contains("cannot hold"), "{error}");
    }

    /// A pick outside the layer's experts is refused by name.
    #[test]
    fn slot_table_refuses_out_of_range_picks() {
        let mut table = LayerSlots::new(64, 8, 0);
        let error = format!("{:?}", table.assign(&[64]).unwrap_err());
        assert!(error.contains("pick 64"), "{error}");
    }
}
