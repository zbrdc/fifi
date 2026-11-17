//! The per-card execution owner of the multi-device step (v1 eager).
//!
//! [`CardStack`] owns the whole distributed step: one [`CardShard`] per
//! card (its arena, its full-size state, its ring status word), the
//! contexts, the ONE stream per card (the coordinator invariant of
//! `multi.rs`), and the [`PeerRing`] spanning them. Contexts and streams
//! sit at the STACK, not inside the shards, because the ring operations
//! take the participating contexts and streams as whole slices; the
//! shard's buffers hold their own device leases, so they stay valid
//! wherever the handles live.
//!
//! What each shard holds:
//!
//! * the arena built from `StepPlan::for_card`'s tensor list through the
//!   same planner, so arena tensor `i` IS plan slot `i` at the plan's
//!   offset;
//! * a FULL-SIZE `ResidentState` on the card's own context: the card
//!   touches its head slice at the same offsets a single-device state
//!   would, so no `StateLayout` surgery is owed (the waste is the unused
//!   tail of the delta buffer, accepted for v1);
//! * the card's ring status word, zeroed at construction, ORed by every
//!   ring wait of the session, folded at read-back (`fold_ring_status`).
//!
//! Weights are not owned here yet: they arrive with the shard's weight
//! views (the next slice), loaded through `shard_loads` + `load_sliced`.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device executor, the first production caller"
    )
)]

use crate::EngError;
use crate::family::qwen36::cuda::dims::{StepDims, StepOptions};
use crate::family::qwen36::cuda::shard::ShardTable;
use crate::family::qwen36::cuda::state::ResidentState;
use crate::family::qwen36::cuda::step_plan::{KernelScratch, StepPlan};
use crate::kernels::cuda::peer_ring::{PeerRing, RingMode};
use std::cell::Cell;

use crate::kernels::cuda::{
    Admission, CudaContext, DeviceArena, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream,
};

/// One card's device allocations. Field order is drop order; every buffer
/// holds its own device lease, so nothing here depends on the context
/// handle outliving the shard.
pub(crate) struct CardShard {
    arena: DeviceArena,
    ring_word: DeviceBuffer,
}

impl CardShard {
    /// The shard's allocations on `context` with `stream`: the arena over
    /// `plan`'s tensor list and a zeroed ring status word. The card's
    /// [`ResidentState`] is allocated beside it but returned separately:
    /// the shared stack keeps only what the step's views borrow, the way
    /// the single-device split leaks `StepResources` and keeps the state
    /// mutable in the step.
    pub(crate) fn new(
        context: &CudaContext,
        stream: &DeviceStream,
        plan: &StepPlan,
    ) -> Result<Self, EngError> {
        let arena = context.arena(&plan.arena_tensors())?;
        let ring_word = context.allocate(4)?;
        ring_word.write_at(stream, 0, &0u32.to_le_bytes())?;
        Ok(Self { arena, ring_word })
    }

    /// The card's arena (the card block driver resolves its op slots
    /// against it through `MixerViews::for_card`).
    pub(crate) fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    /// Arena tensor `index` of this card's plan: the same slot, at the
    /// plan's own offset and byte count.
    pub(crate) fn tensor(&self, index: usize) -> Result<DeviceRange<'_>, EngError> {
        self.arena.tensor(index)
    }

    /// The card's ring status word as the range the ring operations take.
    pub(crate) fn ring_status(&self) -> Result<DeviceRange<'_>, EngError> {
        self.ring_word.range(0, 4)
    }

    /// The status word's buffer, for the read-back that folds it.
    pub(crate) fn ring_word(&self) -> &DeviceBuffer {
        &self.ring_word
    }
}

/// The whole distributed step's owners. Field order is drop order: the
/// ring's mailboxes first, then the shards' allocations, then the plain
/// plans and table, and the context handles last - though every device
/// allocation above carries its own lease, the order keeps destruction
/// easy to reason about.
pub(crate) struct CardStack {
    ring: PeerRing,
    shards: Vec<CardShard>,
    table: ShardTable,
    plans: Vec<StepPlan>,
    streams: Vec<DeviceStream>,
    /// Reusable completion events, one owned by each consumer's context.
    broadcast_events: Vec<DeviceEvent>,
    contexts: Vec<CudaContext>,
    /// A broadcast has only one publisher, so its following broadcast cannot
    /// acknowledge that every peer has consumed a double-buffered slot.
    sync_broadcasts: bool,
    /// Experimental device-side consumer barrier for broadcast slot reuse.
    async_broadcasts: Cell<bool>,
}

impl CardStack {
    /// Builds the stack for `devices.len()` cards. Card `i` runs on
    /// `devices[i]` (one ordinal per physical card; the same ordinal twice
    /// gives two logical contexts, the rig configuration). The mailbox
    /// covers `dims.shape.hidden` values: every exchange payload of the
    /// plan - embed broadcast, mixer sum, MoE sum - is at most the hidden
    /// row; route picks and gathered K/V rows are narrower.
    pub(crate) fn new(
        dims: &StepDims,
        options: &StepOptions,
        devices: &[usize],
        admission: &Admission,
    ) -> Result<(Self, Vec<ResidentState>), EngError> {
        let table = ShardTable::new(&dims.shape, devices.len(), 32)?;
        Self::with_table(dims, options, devices, admission, &table)
    }

    /// The ring's mailbox payload cap: the default covers the eager step's
    /// exchanges (hidden*top_k for the fused MoE block); a chunked session
    /// passes m*hidden so every layer sum of a chunk is one sequence.
    pub(crate) fn with_payload(
        dims: &StepDims,
        options: &StepOptions,
        devices: &[usize],
        admission: &Admission,
        table: &ShardTable,
        payload: Option<usize>,
    ) -> Result<(Self, Vec<ResidentState>), EngError> {
        let ring_payload = payload.unwrap_or(dims.shape.hidden * dims.shape.moe.top_k.max(1));
        Self::with_table_payload(dims, options, devices, admission, table, ring_payload)
    }

    /// The stack over an explicit shard table: the placement planner's
    /// table, whose runs the loader must have used for the same cards.
    pub(crate) fn with_table(
        dims: &StepDims,
        options: &StepOptions,
        devices: &[usize],
        admission: &Admission,
        table: &ShardTable,
    ) -> Result<(Self, Vec<ResidentState>), EngError> {
        Self::with_table_payload(
            dims,
            options,
            devices,
            admission,
            table,
            dims.shape.hidden * dims.shape.moe.top_k.max(1),
        )
    }

    pub(crate) fn with_table_payload(
        dims: &StepDims,
        options: &StepOptions,
        devices: &[usize],
        admission: &Admission,
        table: &ShardTable,
        ring_payload: usize,
    ) -> Result<(Self, Vec<ResidentState>), EngError> {
        if devices.is_empty() {
            return Err(EngError::BadRequest(
                "the card stack needs at least one device".to_owned(),
            ));
        }
        let scratch = KernelScratch::for_dims(dims, 1, options.kv_mode)?;
        let mut contexts = Vec::with_capacity(devices.len());
        let mut streams = Vec::with_capacity(devices.len());
        for &device in devices {
            let context = CudaContext::with_admission_on(device, admission)?;
            streams.push(context.stream()?);
            contexts.push(context);
        }
        // The ring first: it captures each mailbox's base address from the
        // bare contexts before anything else borrows them. The payload cap
        // covers the MoE all-reduce, which exchanges every `expert_down`
        // pick row (`top_k` hidden rows) in one sequence.
        let ring = PeerRing::new(&contexts, ring_payload)?;
        let async_broadcasts =
            std::env::var_os("Q36_P2P_ASYNC_BROADCAST").is_some_and(|value| value == "1");
        let broadcast_events = if async_broadcasts {
            contexts
                .iter()
                .map(CudaContext::event)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let sync_broadcasts =
            std::env::var_os("Q36_P2P_SYNC_BROADCAST").is_some_and(|value| value == "1");
        let mut plans = Vec::with_capacity(devices.len());
        let mut shards = Vec::with_capacity(devices.len());
        let mut states = Vec::with_capacity(devices.len());
        for (card, context) in contexts.iter().enumerate() {
            let plan = StepPlan::for_card(dims, table, card, options, &scratch)?;
            shards.push(CardShard::new(context, &streams[card], &plan)?);
            // The card's cache planes hold only its own K/V head groups
            // (whole head multiples by the shard table's deal); the
            // gathered full row stays at the whole shape's width.
            let kv_heads = table.kv_rows(card)?.len() / dims.shape.attn.head_dim;
            states.push(ResidentState::allocate_with_kv_heads(
                context,
                &streams[card],
                dims,
                options,
                kv_heads,
            )?);
            plans.push(plan);
        }
        Ok((
            Self {
                ring,
                shards,
                table: table.clone(),
                plans,
                streams,
                broadcast_events,
                contexts,
                sync_broadcasts,
                async_broadcasts: Cell::new(async_broadcasts),
            },
            states,
        ))
    }

    /// The ring re-zero for a new turn: mailboxes and ring status words,
    /// folded into `reset_states_and_ring`.
    pub(crate) fn ring(&self) -> &PeerRing {
        &self.ring
    }

    pub(crate) fn shards(&self) -> &[CardShard] {
        &self.shards
    }

    pub(crate) fn table(&self) -> &ShardTable {
        &self.table
    }

    pub(crate) fn plans(&self) -> &[StepPlan] {
        &self.plans
    }

    /// The participating contexts, one per card, in card order.
    pub(crate) fn contexts(&self) -> &[CudaContext] {
        &self.contexts
    }

    /// The turn reset: every card's state cleared and the ring re-zeroed.
    /// Every stream is joined first - a reset never overtakes a running
    /// token - and the first token of the next turn writes fresh step
    /// params over the cleared word block. The states travel mutably by
    /// parameter: the stack itself stays shared.
    pub(crate) fn reset_states_and_ring(
        &self,
        states: &mut [ResidentState],
    ) -> Result<(), EngError> {
        for ((card, state), shard) in states.iter_mut().enumerate().zip(self.shards.iter()) {
            self.streams[card].synchronize()?;
            state.reset(&self.streams[card])?;
            shard
                .ring_word
                .write_at(&self.streams[card], 0, &0u32.to_le_bytes())?;
        }
        self.ring.reset(&self.contexts, &self.streams)
    }

    pub(crate) fn streams(&self) -> &[DeviceStream] {
        &self.streams
    }

    #[cfg(test)]
    pub(super) fn enable_broadcast_sync_for_test(&mut self) {
        self.sync_broadcasts = true;
    }

    #[cfg(test)]
    pub(super) fn prepare_async_broadcasts_for_test(&mut self) -> Result<(), EngError> {
        if self.broadcast_events.is_empty() {
            self.broadcast_events = self
                .contexts
                .iter()
                .map(CudaContext::event)
                .collect::<Result<Vec<_>, _>>()?;
        }
        self.async_broadcasts.set(false);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn new_for_async_broadcast_test(
        dims: &StepDims,
        options: &StepOptions,
        devices: &[usize],
        admission: &Admission,
    ) -> Result<(Self, Vec<ResidentState>), EngError> {
        let (mut stack, states) = Self::new(dims, options, devices, admission)?;
        stack.prepare_async_broadcasts_for_test()?;
        Ok((stack, states))
    }

    #[cfg(test)]
    pub(super) fn set_async_broadcasts_for_test(&self, enabled: bool) {
        assert!(!enabled || self.broadcast_events.len() == self.contexts.len());
        self.async_broadcasts.set(enabled);
    }

    /// One ring sum over every card: `partials[card]` in, `outs[card]`
    /// out, `statuses[card]` ORed. The executor's per-op entry point; the
    /// caller keeps the gather-pair gate and the end-of-token fold.
    pub(crate) fn exchange_sum(
        &self,
        partials: &[DeviceRange<'_>],
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        count: usize,
        mode: RingMode,
    ) -> Result<(), EngError> {
        self.ring.exchange_sum(
            &self.contexts,
            &self.streams,
            partials,
            outs,
            statuses,
            count,
            mode,
        )
    }

    /// One ring broadcast from `source_card`, the same argument shape as
    /// [`Self::exchange_sum`].
    pub(crate) fn broadcast(
        &self,
        source_card: usize,
        src: DeviceRange<'_>,
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        count: usize,
    ) -> Result<(), EngError> {
        self.ring.broadcast(
            &self.contexts,
            &self.streams,
            source_card,
            src,
            outs,
            statuses,
            count,
        )?;
        if self.async_broadcasts.get() {
            // A publisher waits on each peer's post-copy event before it can
            // reuse a mailbox parity slot for a later broadcast.
            for (card, (stream, event)) in
                self.streams.iter().zip(&self.broadcast_events).enumerate()
            {
                if card != source_card {
                    stream.record(event)?;
                    self.streams[source_card].wait_peer_event(event)?;
                }
            }
        } else if self.sync_broadcasts {
            // A producer can otherwise reuse its parity slot while a slower
            // peer is still waiting for the preceding broadcast. Wait until
            // every consumer has copied the payload before any later publish.
            for stream in &self.streams {
                stream.synchronize()?;
            }
        }
        Ok(())
    }

    /// A broadcast followed by a consumer barrier, regardless of the eager
    /// path's optional synchronization setting. Use this when another
    /// broadcast can reuse the same mailbox parity before a sum or gather.
    pub(crate) fn broadcast_synchronized(
        &self,
        source_card: usize,
        src: DeviceRange<'_>,
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        count: usize,
    ) -> Result<(), EngError> {
        self.broadcast(source_card, src, outs, statuses, count)?;
        if !self.sync_broadcasts && !self.async_broadcasts.get() {
            for stream in &self.streams {
                stream.synchronize()?;
            }
        }
        Ok(())
    }

    /// One all-gather of a row dealt across the cards: card `i` computed
    /// `ranges[i]` of the `total` row in `parts[i]`; afterwards every
    /// `outs[i]` holds the full row. The K/V gathers of the full-attention
    /// core run through here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn all_gather_row(
        &self,
        parts: &[DeviceRange<'_>],
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        ranges: &[std::ops::Range<usize>],
        total: usize,
    ) -> Result<(), EngError> {
        self.ring.all_gather_row(
            &self.contexts,
            &self.streams,
            parts,
            outs,
            statuses,
            ranges,
            total,
        )
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use crate::family::project::cuda::served_admission;
    use crate::family::qwen36::cuda::common_tests::dims;
    use crate::family::qwen36::cuda::dims::KvMode;
    use crate::kernels::cuda::device_present;

    fn options() -> StepOptions {
        StepOptions {
            kv_mode: KvMode::Int8G32,
            context_tokens: 1024,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: crate::family::project::cuda::WeightLayout::Native,
        }
    }

    #[test]
    #[ignore = "requires two peer-enabled CUDA devices; run under the GPU lease"]
    fn synchronized_broadcasts_keep_three_messages_intact_on_two_gpus() {
        assert!(device_present(
            "synchronized_broadcasts_keep_three_messages_intact_on_two_gpus"
        ));
        let devices = [0, 1];
        let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe devices");
        crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer-capable devices");
        crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peers");
        let (mut stack, _states) =
            CardStack::new(&dims(), &options(), &devices, &served_admission()).expect("stack");
        stack.prepare_async_broadcasts_for_test().expect("events");
        let mut inputs = Vec::new();
        let mut outputs = [Vec::new(), Vec::new()];
        let mut expected = Vec::new();
        for message in 0..3 {
            let bytes: Vec<u8> = (0..4)
                .flat_map(|i| ((message * 10 + i) as f32).to_le_bytes())
                .collect();
            let input = stack.contexts()[0].allocate(bytes.len()).expect("input");
            input
                .write_at(&stack.streams()[0], 0, &bytes)
                .expect("seed");
            inputs.push(input);
            for (card, card_outputs) in outputs.iter_mut().enumerate() {
                card_outputs.push(
                    stack.contexts()[card]
                        .allocate(bytes.len())
                        .expect("output"),
                );
            }
            expected.push(bytes);
        }
        for (sync_broadcasts, async_broadcasts) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            stack.sync_broadcasts = sync_broadcasts;
            stack.set_async_broadcasts_for_test(async_broadcasts);
            for _ in 0..32 {
                for message in 0..3 {
                    let outs = [
                        outputs[0][message].range(0, 16).unwrap(),
                        outputs[1][message].range(0, 16).unwrap(),
                    ];
                    let statuses = [
                        stack.shards()[0].ring_status().unwrap(),
                        stack.shards()[1].ring_status().unwrap(),
                    ];
                    stack
                        .broadcast_synchronized(
                            0,
                            inputs[message].range(0, 16).unwrap(),
                            &outs,
                            &statuses,
                            4,
                        )
                        .expect("broadcast");
                }
                for (card, card_outputs) in outputs.iter().enumerate() {
                    let mut status = [0; 4];
                    stack.shards()[card]
                        .ring_word()
                        .read_at(&stack.streams()[card], 0, &mut status)
                        .unwrap();
                    assert_eq!(u32::from_le_bytes(status), 0, "card {card} ring status");
                    for message in 0..3 {
                        let mut actual = [0; 16];
                        card_outputs[message]
                            .read_at(&stack.streams()[card], 0, &mut actual)
                            .unwrap();
                        assert_eq!(
                            actual.as_slice(),
                            expected[message],
                            "card {card}, message {message}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires two peer-enabled CUDA devices; run under the GPU lease"]
    fn asynchronous_broadcasts_preserve_reused_slots_on_two_gpus() {
        const NAME: &str = "asynchronous_broadcasts_preserve_reused_slots_on_two_gpus";
        assert!(device_present(NAME));
        let devices = [0, 1];
        let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe devices");
        crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer-capable devices");
        crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peers");
        let (stack, _states) = CardStack::new_for_async_broadcast_test(
            &dims(),
            &options(),
            &devices,
            &served_admission(),
        )
        .expect("stack");
        let unrecorded = stack.contexts()[1].event().expect("unrecorded event");
        assert!(matches!(
            stack.streams()[0].wait_peer_event(&unrecorded),
            Err(EngError::BadRequest(message)) if message.contains("never recorded")
        ));
        stack.set_async_broadcasts_for_test(true);

        // The first half repeatedly uses one publisher; the second alternates
        // publishers. Queue every message before reading anything back, so
        // both mailbox slots and the reusable events are stressed in flight.
        const MESSAGES: usize = 128;
        let mut inputs = Vec::with_capacity(MESSAGES);
        let mut outputs = [Vec::new(), Vec::new()];
        let mut expected = Vec::with_capacity(MESSAGES);
        for message in 0..MESSAGES {
            let source = if message < MESSAGES / 2 {
                0
            } else {
                message % 2
            };
            let bytes: Vec<u8> = (0..4)
                .flat_map(|i| ((message * 10 + i) as f32).to_le_bytes())
                .collect();
            let input = stack.contexts()[source]
                .allocate(bytes.len())
                .expect("input");
            input
                .write_at(&stack.streams()[source], 0, &bytes)
                .expect("seed");
            inputs.push(input);
            for (card, card_outputs) in outputs.iter_mut().enumerate() {
                card_outputs.push(stack.contexts()[card].allocate(16).expect("output"));
            }
            expected.push(bytes);
        }
        let statuses = [
            stack.shards()[0].ring_status().unwrap(),
            stack.shards()[1].ring_status().unwrap(),
        ];
        for (message, input) in inputs.iter().enumerate() {
            let source = if message < MESSAGES / 2 {
                0
            } else {
                message % 2
            };
            let outs = [
                outputs[0][message].range(0, 16).unwrap(),
                outputs[1][message].range(0, 16).unwrap(),
            ];
            stack
                .broadcast(source, input.range(0, 16).unwrap(), &outs, &statuses, 4)
                .expect("async broadcast");
        }
        for (card, card_outputs) in outputs.iter().enumerate() {
            stack.streams()[card].synchronize().expect("drain stream");
            let mut status = [0; 4];
            stack.shards()[card]
                .ring_word()
                .read_at(&stack.streams()[card], 0, &mut status)
                .unwrap();
            assert_eq!(u32::from_le_bytes(status), 0, "card {card} ring status");
            for (message, output) in card_outputs.iter().enumerate() {
                let mut actual = [0; 16];
                output
                    .read_at(&stack.streams()[card], 0, &mut actual)
                    .unwrap();
                assert_eq!(
                    actual.as_slice(),
                    expected[message],
                    "card {card} message {message}"
                );
            }
        }
    }

    /// Two logical shards on the one physical device: every plan slot lands
    /// in its own arena at the plan's offset and size, both states are
    /// full-size and identical in shape, the ring spans both cards, and
    /// both status words start at zero - the ownership facts the
    /// distributed block tests build on.
    #[test]
    fn two_logical_shards_own_their_plan_arenas_and_states() {
        const NAME: &str = "two_logical_shards_own_their_plan_arenas_and_states";
        if !device_present(NAME) {
            return;
        }
        let d = dims();
        let admission = served_admission();
        let (stack, states) = CardStack::new(&d, &options(), &[0, 0], &admission).expect("shards");
        assert_eq!(stack.shards().len(), 2);
        assert_eq!(stack.plans().len(), 2);
        assert_eq!(stack.ring().cards(), 2);
        assert_eq!(stack.table().cards(), 2);
        assert_eq!(states.len(), 2);
        for (card, shard) in stack.shards().iter().enumerate() {
            let plan = &stack.plans()[card];
            let tensors = plan.tensors();
            for (index, slot) in tensors.iter().enumerate() {
                let range = shard.tensor(index).expect("arena tensor");
                assert_eq!(
                    range.offset(),
                    plan.plan().offsets[index],
                    "card {card} slot {:?}: offset",
                    slot.slot
                );
                assert_eq!(
                    range.bytes(),
                    slot.bytes,
                    "card {card} slot {:?}: bytes",
                    slot.slot
                );
            }
            // The delta state replicates; the KV pool is SLICED to the
            // card's own heads, and the cards' pools together hold exactly
            // one full replica's worth of cache.
            assert!(states[card].kv_bytes() > 0);
            assert_eq!(
                states[card].layout().delta.total,
                states[1 - card].layout().delta.total,
                "card {card}: delta state size"
            );
            assert_eq!(
                states[card].kv_bytes(),
                states[1 - card].kv_bytes(),
                "card {card}: kv state size"
            );
            assert_eq!(
                states[card].kv_bytes() + states[1 - card].kv_bytes(),
                crate::family::qwen36::cuda::state_layout::StateLayout::new(&d, &options())
                    .expect("full layout")
                    .kv
                    .total,
                "the sliced pools hold exactly one replica of KV"
            );
        }
        for (card, shard) in stack.shards().iter().enumerate() {
            let mut word = [0u8; 4];
            shard
                .ring_word()
                .read_at(&stack.streams()[card], 0, &mut word)
                .expect("read the status word");
            assert_eq!(u32::from_le_bytes(word), 0);
        }
    }

    /// The destruction path under load: queue eight dependent chained
    /// exchanges on a fresh stack, sync NOTHING, and drop the stack
    /// through its ordinary field order (auxiliary buffers first, ring,
    /// then the shards and handles). A healthy device afterwards - a
    /// fresh stack exchanges exactly - proves the drop left no pending
    /// reference or CUDA error behind; the test adds no synchronization
    /// production itself does not perform.
    #[test]
    fn a_stack_with_queued_exchanges_drops_clean_without_syncs() {
        const NAME: &str = "a_stack_with_queued_exchanges_drops_clean_without_syncs";
        if !device_present(NAME) {
            return;
        }
        let d = dims();
        let admission = served_admission();
        {
            let (stack, _states) =
                CardStack::new(&d, &options(), &[0, 0], &admission).expect("stack");
            let n = 64usize;
            let rounds = 8usize;
            // Auxiliary buffers after the stack, so they drop first.
            let mut partials = Vec::with_capacity(2);
            let mut outs = Vec::with_capacity(2 * rounds);
            for card in 0..2 {
                let seed: Vec<u8> = (0..n)
                    .map(|i| ((card * 17 + i) as f32).to_le_bytes())
                    .collect::<Vec<_>>()
                    .concat();
                let partial = stack.contexts()[card].allocate(n * 4).expect("partial");
                partial
                    .write_at(&stack.streams()[card], 0, &seed)
                    .expect("seed partial");
                partials.push(partial);
                for _ in 0..rounds {
                    outs.push(stack.contexts()[card].allocate(n * 4).expect("out"));
                }
            }
            // Round 0 from the seeded partials; every later round chains the
            // previous round's out on the same card stream.
            let mut previous = [
                partials[0].range(0, n * 4).expect("range"),
                partials[1].range(0, n * 4).expect("range"),
            ];
            for round in 0..rounds {
                let current = [
                    outs[round].range(0, n * 4).expect("range"),
                    outs[rounds + round].range(0, n * 4).expect("range"),
                ];
                let statuses = [
                    stack.shards()[0].ring_status().expect("status"),
                    stack.shards()[1].ring_status().expect("status"),
                ];
                stack
                    .exchange_sum(&previous, &current, &statuses, n, RingMode::F32)
                    .expect("queue the chained exchange");
                previous = current;
            }
            // No synchronize: the buffers and the stack drop as-is.
        }
        // The device must be healthy: a fresh stack exchanges exactly.
        let (stack, _states) =
            CardStack::new(&d, &options(), &[0, 0], &admission).expect("fresh stack");
        let n = 64usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
        let mut partials = Vec::with_capacity(2);
        let mut outs = Vec::with_capacity(2);
        for (card, values) in [a.clone(), b.clone()].iter().enumerate() {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let partial = stack.contexts()[card].allocate(n * 4).expect("partial");
            partial
                .write_at(&stack.streams()[card], 0, &bytes)
                .expect("seed");
            partials.push(partial);
            outs.push(stack.contexts()[card].allocate(n * 4).expect("out"));
        }
        stack
            .exchange_sum(
                &[
                    partials[0].range(0, n * 4).expect("range"),
                    partials[1].range(0, n * 4).expect("range"),
                ],
                &[
                    outs[0].range(0, n * 4).expect("range"),
                    outs[1].range(0, n * 4).expect("range"),
                ],
                &[
                    stack.shards()[0].ring_status().expect("status"),
                    stack.shards()[1].ring_status().expect("status"),
                ],
                n,
                RingMode::F32,
            )
            .expect("the fresh exchange");
        for (card, (stream, out)) in stack.streams().iter().zip(outs.iter()).enumerate().take(2) {
            stream.synchronize().expect("sync");
            let mut bytes = vec![0u8; n * 4];
            out.read_at(stream, 0, &mut bytes).expect("read");
            let got: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            for (i, (&g, (&x, &y))) in got.iter().zip(a.iter().zip(&b)).enumerate() {
                assert_eq!(g, x + y, "card {card} value {i}");
            }
        }
    }
}
