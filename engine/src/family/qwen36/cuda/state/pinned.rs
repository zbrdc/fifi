//! Fixed-capacity host backing; only committed ranges are transferred or imported.
use super::super::state_layout::StateLayout;
use super::{
    ResidentState, StateIdentity,
    device_tip::{DeviceTip, DeviceTipKey},
    export::IMPORT_REWRITES_WORDS,
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceEvent, DeviceStream, PinnedBuf},
};
use std::{cell::RefCell, rc::Rc};

fn invalid(message: &str) -> EngError {
    EngError::BadRequest(format!("pinned prefix: {message}"))
}

pub(crate) struct PinnedSnapshot {
    identity: StateIdentity,
    layout: StateLayout,
    tokens: usize,
    owner: u64,
    slot: usize,
    generation: u64,
    copied_kv_bytes: usize,
    data: Rc<RefCell<PinnedBuf>>,
    device_tip: Option<DeviceTipKey>,
}

impl PinnedSnapshot {
    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn copied_kv_bytes(&self) -> usize {
        self.copied_kv_bytes
    }
}

pub(crate) struct PendingPinned {
    saved: PinnedSnapshot,
    epoch: u64,
    instance: Rc<()>,
    inherited_tokens: usize,
}

impl PendingPinned {
    pub(crate) fn attach_device_tip(&mut self, key: DeviceTipKey) -> Result<(), EngError> {
        if key.positions() != self.saved.tokens {
            return Err(invalid("device tip position differs"));
        }
        self.saved.device_tip = Some(key);
        Ok(())
    }
}

/// Bounded persistent backing: retained checkpoints plus one independently owned staging slot.
pub(crate) struct PrefixPinPool {
    slots: Vec<Rc<RefCell<PinnedBuf>>>,
    event: DeviceEvent,
    layout: StateLayout,
    delta_bytes: usize,
    slab_bytes: usize,
    next: usize,
    epoch: u64,
    generations: Vec<u64>,
    live_source: Option<(usize, u64, Rc<()>)>,
}

impl PrefixPinPool {
    pub(crate) fn checkpoint_bytes(layout: &StateLayout) -> Result<usize, EngError> {
        layout
            .delta
            .total
            .checked_div(layout.delta.slots)
            .ok_or_else(|| invalid("zero Delta slots"))?
            .checked_add(layout.kv.total)
            .ok_or_else(|| EngError::MemoryPlan("pinned prefix size overflows".into()))
    }

    pub(crate) fn new(
        ctx: &CudaContext,
        layout: StateLayout,
        retained: usize,
    ) -> Result<Self, EngError> {
        if !(1..=2).contains(&retained) {
            return Err(invalid("pool needs one or two retained entries"));
        }
        let slab_bytes = Self::checkpoint_bytes(&layout)?;
        let delta_bytes = layout.delta.total / layout.delta.slots;
        slab_bytes
            .checked_mul(retained + 1)
            .ok_or_else(|| EngError::MemoryPlan("pinned prefix pool size overflows".into()))?;
        let mut slots = Vec::with_capacity(retained + 1);
        for _ in 0..=retained {
            slots.push(Rc::new(RefCell::new(ctx.pinned_buf(slab_bytes)?)));
        }
        Ok(Self {
            slots,
            event: ctx.event()?,
            layout,
            delta_bytes,
            slab_bytes,
            next: 0,
            epoch: 0,
            generations: vec![0; retained + 1],
            live_source: None,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.slab_bytes * self.slots.len()
    }

    pub(crate) fn retained_entries(&self) -> usize {
        self.slots.len() - 1
    }

    pub(crate) fn invalidate(&mut self) -> Result<(), EngError> {
        self.live_source = None;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| invalid("generation exhausted"))?;
        Ok(())
    }

    pub(crate) fn can_recycle(
        &self,
        stream: &DeviceStream,
        state: &ResidentState,
        saved: &PinnedSnapshot,
    ) -> bool {
        self.live_source
            .as_ref()
            .is_some_and(|(slot, generation, instance)| {
                (*slot, *generation) == (saved.slot, saved.generation)
                    && Rc::ptr_eq(instance, &state.instance)
            })
            && self.generations.get(saved.slot) == Some(&saved.generation)
            && saved.owner == stream.id()
            && saved.identity == state.identity
            && saved.layout == self.layout
            && saved.layout == state.layout
            && saved.tokens <= state.tokens()
            && state.book.begin_export(stream.id()).is_ok()
            && self
                .slots
                .get(saved.slot)
                .is_some_and(|slot| Rc::ptr_eq(slot, &saved.data) && Rc::strong_count(slot) == 2)
    }

    pub(crate) fn begin(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        recycled: Option<PinnedSnapshot>,
    ) -> Result<PendingPinned, EngError> {
        let source = recycled
            .as_ref()
            .filter(|saved| self.can_recycle(stream, state, saved))
            .map(|saved| (saved.slot, saved.tokens));
        self.live_source = None;
        state.resident_kv()?;
        state.book.begin_export(stream.id())?;
        if self.layout != state.layout {
            return Err(invalid("state layout changed"));
        }
        drop(recycled);
        // Dropping the retired Tip must leave no reader before its backing can change.
        let source = source.filter(|&(slot, _)| Rc::strong_count(&self.slots[slot]) == 1);
        let (slot, inherited_tokens) = match source {
            Some(source) => source,
            None => {
                let slot = (0..self.slots.len())
                    .map(|offset| (self.next + offset) % self.slots.len())
                    .find(|&i| Rc::strong_count(&self.slots[i]) == 1)
                    .ok_or_else(|| {
                        EngError::MemoryPlan("all pinned prefix slots are retained".into())
                    })?;
                (slot, 0)
            }
        };
        self.generations[slot] = self.generations[slot]
            .checked_add(1)
            .ok_or_else(|| invalid("slot generation exhausted"))?;
        ctx.copy_to_pinned(
            stream,
            state.delta.range(0, self.delta_bytes)?,
            &mut self.slots[slot].borrow_mut(),
            0,
        )?;
        self.next = (slot + 1) % self.slots.len();
        Ok(PendingPinned {
            saved: PinnedSnapshot {
                identity: state.identity.clone(),
                layout: self.layout,
                tokens: state.tokens(),
                owner: stream.id(),
                slot,
                generation: self.generations[slot],
                copied_kv_bytes: 0,
                data: Rc::clone(&self.slots[slot]),
                device_tip: None,
            },
            epoch: self.epoch,
            instance: Rc::clone(&state.instance),
            inherited_tokens,
        })
    }

    pub(crate) fn finish(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
        mut pending: PendingPinned,
    ) -> Result<PinnedSnapshot, EngError> {
        self.live_source = None;
        state.resident_kv()?;
        state.book.begin_export(stream.id())?;
        let saved = &pending.saved;
        if pending.epoch != self.epoch
            || !Rc::ptr_eq(&pending.instance, &state.instance)
            || self.generations.get(saved.slot) != Some(&saved.generation)
            || saved.owner != stream.id()
            || saved.layout != state.layout
            || saved.identity != state.identity
            || saved.tokens > state.tokens()
            || self
                .slots
                .get(saved.slot)
                .is_none_or(|slot| !Rc::ptr_eq(&saved.data, slot))
        {
            return Err(invalid("deferred save lost its live lineage"));
        }
        let mut buffer = self.slots[saved.slot].borrow_mut();
        let mut copied_kv_bytes = 0usize;
        for layer in 0..state.layout.kv.layers {
            let inherited = state
                .layout
                .kv
                .prefix_ranges(layer, pending.inherited_tokens)?;
            for (range, old) in state
                .layout
                .kv
                .prefix_ranges(layer, saved.tokens)?
                .into_iter()
                .zip(inherited)
            {
                let bytes = range
                    .bytes
                    .checked_sub(old.bytes)
                    .ok_or_else(|| invalid("inherited prefix exceeds save"))?;
                if bytes == 0 {
                    continue;
                }
                let device = range
                    .device_offset
                    .checked_add(old.bytes)
                    .ok_or_else(|| invalid("device offset overflows"))?;
                let host = self
                    .delta_bytes
                    .checked_add(device)
                    .ok_or_else(|| invalid("host offset overflows"))?;
                ctx.copy_to_pinned(
                    stream,
                    state.resident_kv()?.range(device, bytes)?,
                    &mut buffer,
                    host,
                )?;
                copied_kv_bytes = copied_kv_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("copied KV byte count overflows"))?;
            }
        }
        stream.record(&self.event)?;
        PinnedBuf::wait_all(&mut [&mut buffer], &self.event)?;
        self.live_source = Some((saved.slot, saved.generation, Rc::clone(&state.instance)));
        pending.saved.copied_kv_bytes = copied_kv_bytes;
        Ok(pending.saved)
    }

    pub(crate) fn restore(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &mut ResidentState,
        saved: &PinnedSnapshot,
        device: Option<&DeviceTip>,
    ) -> Result<bool, EngError> {
        self.live_source = None;
        state.resident_kv()?;
        let Some(slot) = self.slots.get(saved.slot) else {
            return Err(invalid("unknown slot"));
        };
        if self.generations.get(saved.slot) != Some(&saved.generation)
            || saved.owner != stream.id()
            || saved.layout != state.layout
            || saved.identity != state.identity
            || saved.tokens > state.layout.kv.capacity
            || !Rc::ptr_eq(slot, &saved.data)
        {
            return Err(invalid("snapshot identity, layout or owner differs"));
        }
        if let (Some(device), Some(key)) = (device, saved.device_tip.as_ref())
            && device.restore(ctx, stream, state, key)?
        {
            self.live_source = Some((saved.slot, saved.generation, Rc::clone(&state.instance)));
            return Ok(true);
        }
        let mut buffer = saved.data.borrow_mut();
        state
            .book
            .begin_import(stream.id(), IMPORT_REWRITES_WORDS)?;
        state.resident_tip = None;
        state.clear_words(stream)?;
        state.clear_delta(stream, 1)?;
        ctx.copy_from_pinned(
            stream,
            &mut buffer,
            0,
            state.delta.range(0, self.delta_bytes)?,
        )?;
        for layer in 0..state.layout.kv.layers {
            for range in state.layout.kv.prefix_ranges(layer, saved.tokens)? {
                if range.bytes == 0 {
                    continue;
                }
                ctx.copy_from_pinned(
                    stream,
                    &mut buffer,
                    self.delta_bytes + range.device_offset,
                    state
                        .resident_kv()?
                        .range(range.device_offset, range.bytes)?,
                )?;
            }
        }
        stream.record(&self.event)?;
        buffer.wait(&self.event)?;
        state.book.finish_overwrite(saved.tokens);
        self.live_source = Some((saved.slot, saved.generation, Rc::clone(&state.instance)));
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{
        project::cuda::WeightLayout,
        qwen36::cuda::{
            common_tests,
            dims::{KvMode, StepOptions},
            prefix::CudaPrefillPath,
        },
        session::{DecodeSession, PrefixSlot, SaveKind, StoreLimits, TokenStepper},
    };

    #[test]
    #[ignore = "requires native CUDA; run explicitly under the GPU lease"]
    fn retained_pinned_checkpoints_survive_a_foreign_branch() {
        struct SnapshotBytes {
            mode: KvMode,
            tokens: usize,
            delta: Vec<u8>,
            kv: Vec<u8>,
        }

        let observed = (|| -> Result<Vec<SnapshotBytes>, EngError> {
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let mut observed = Vec::new();
            for mode in [KvMode::Fp16, KvMode::Int8G32] {
                let options = StepOptions {
                    kv_mode: mode,
                    context_tokens: 16,
                    delta_slots: 1,
                    forked_experts: false,
                    tiered: None,
                    layout: WeightLayout::Native,
                };
                let mut state =
                    ResidentState::allocate(&ctx, &stream, &common_tests::dims(), &options)?;
                let mut pool = PrefixPinPool::new(&ctx, state.layout, 2)?;
                let mut snapshots = Vec::new();
                for (tokens, byte) in [(4, 7u8), (8, 29u8)] {
                    pool.invalidate()?;
                    state
                        .delta
                        .write_at(&stream, 0, &vec![byte; pool.delta_bytes])?;
                    let kv = state.resident_kv()?;
                    kv.write_at(&stream, 0, &vec![byte + 1; kv.bytes()])?;
                    state.book.finish_overwrite(tokens);
                    let pending = pool.begin(&ctx, &stream, &state, None)?;
                    snapshots.push(pool.finish(&ctx, &stream, &state, pending)?);
                }
                for saved in &snapshots {
                    state
                        .delta
                        .write_at(&stream, 0, &vec![61; pool.delta_bytes])?;
                    let kv = state.resident_kv()?;
                    kv.write_at(&stream, 0, &vec![63; kv.bytes()])?;
                    state.book.finish_overwrite(16);
                    pool.restore(&ctx, &stream, &mut state, saved, None)?;
                    let mut delta = vec![0; pool.delta_bytes];
                    state.delta.read_at(&stream, 0, &mut delta)?;
                    let mut kv = Vec::new();
                    for layer in 0..state.layout.kv.layers {
                        for range in state.layout.kv.prefix_ranges(layer, saved.tokens())? {
                            let mut bytes = vec![0; range.bytes];
                            state.resident_kv()?.read_at(
                                &stream,
                                range.device_offset,
                                &mut bytes,
                            )?;
                            kv.extend_from_slice(&bytes);
                        }
                    }
                    observed.push(SnapshotBytes {
                        mode,
                        tokens: state.tokens(),
                        delta,
                        kv,
                    });
                }
            }
            Ok(observed)
        })()
        .expect("independent checkpoints must remain usable after a foreign branch");

        assert_eq!(observed.len(), 4);
        for saved in observed {
            let byte = match saved.tokens {
                4 => 7,
                8 => 29,
                other => panic!("unexpected restored position {other}"),
            };
            assert!(!saved.delta.is_empty() && !saved.kv.is_empty());
            assert!(
                saved.delta.iter().all(|&value| value == byte),
                "{:?}: recurrent state changed at {}",
                saved.mode,
                saved.tokens
            );
            assert!(
                saved.kv.iter().all(|&value| value == byte + 1),
                "{:?}: retained KV changed at {}",
                saved.mode,
                saved.tokens
            );
        }
    }

    // The counter drives session metadata; native buffers supply the checkpoint bytes.
    struct Counter(usize);
    impl TokenStepper for Counter {
        const OPERATION: &'static str = "pinned-store fixture";
        fn feed(&mut self, _token: u32) -> Result<(), EngError> {
            self.0 += 1;
            Ok(())
        }
        fn logits(&mut self, _token: u32) -> Result<Vec<f32>, EngError> {
            Err(EngError::Checkpoint("this fixture never decodes".into()))
        }
        fn fed(&self) -> usize {
            self.0
        }
        fn reset(&mut self) -> Result<(), EngError> {
            self.0 = 0;
            Ok(())
        }
    }

    #[test]
    #[ignore = "requires native CUDA; run explicitly under the GPU lease"]
    fn shared_store_keeps_native_block_after_short_tip_and_failed_finish() {
        let outcome = (|| -> Result<(), EngError> {
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let main = [1, 2, 3, 4, 5];
            let side = [9, 8, 7];
            let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
            for mode in [KvMode::Fp16, KvMode::Int8G32] {
                let require = |valid: bool, detail: &str| {
                    if valid {
                        Ok(())
                    } else {
                        Err(EngError::Checkpoint(format!("{mode:?}: {detail}")))
                    }
                };
                let options = StepOptions {
                    kv_mode: mode,
                    context_tokens: 16,
                    delta_slots: 1,
                    forked_experts: false,
                    tiered: None,
                    layout: WeightLayout::Native,
                };
                let mut state =
                    ResidentState::allocate(&ctx, &stream, &common_tests::dims(), &options)?;
                let mut pool = PrefixPinPool::new(&ctx, state.layout, 2)?;
                let mut slot = PrefixSlot::with_limits(StoreLimits {
                    roots: 0,
                    entries: 2,
                });
                let mut session = DecodeSession::default();
                let mut counter = Counter(0);
                let path = CudaPrefillPath::Steps(mode);
                for (key, byte, kind) in [
                    (&main[..], 7u8, SaveKind::Block),
                    (&side[..], 29u8, SaveKind::Tip),
                ] {
                    session.reset(&mut counter)?;
                    session.prefill(&mut counter, key, 16)?;
                    pool.invalidate()?;
                    state
                        .delta
                        .write_at(&stream, 0, &vec![byte; pool.delta_bytes])?;
                    let kv = state.resident_kv()?;
                    kv.write_at(&stream, 0, &vec![byte + 1; kv.bytes()])?;
                    state.book.finish_overwrite(counter.fed());
                    let frozen = session.prepare_prefix_save(key, counter.fed())?;
                    let mut pending = Some(pool.begin(&ctx, &stream, &state, None)?);
                    let saved =
                        frozen.save_prefix(&mut slot, key, path, counter.fed(), kind, || {
                            let pending = pending.take().ok_or_else(|| {
                                EngError::Checkpoint("duplicate publication".into())
                            })?;
                            pool.finish(&ctx, &stream, &state, pending)
                        })?;
                    require(saved, "the main and short foreign save must both publish")?;
                }
                require(
                    slot.keys() == vec![(5, false), (3, false)],
                    "both native entries must coexist",
                )?;
                require(
                    slot.snapshots()
                        .map(PinnedSnapshot::tokens)
                        .collect::<Vec<_>>()
                        == vec![4, 2],
                    "native row counts differ from held keys",
                )?;

                let delta_bytes = pool.delta_bytes;
                require(delta_bytes > 0, "empty Delta fixture")?;
                let check_main = |state: &ResidentState| -> Result<(), EngError> {
                    require(state.tokens() == 4, "restored native row count")?;
                    let mut delta = vec![0; delta_bytes];
                    state.delta.read_at(&stream, 0, &mut delta)?;
                    require(
                        delta.iter().all(|&byte| byte == 7),
                        "main Delta bytes changed",
                    )?;
                    let mut kv_bytes = 0;
                    for layer in 0..state.layout.kv.layers {
                        for range in state.layout.kv.prefix_ranges(layer, 4)? {
                            let mut bytes = vec![0; range.bytes];
                            state.resident_kv()?.read_at(
                                &stream,
                                range.device_offset,
                                &mut bytes,
                            )?;
                            kv_bytes += bytes.len();
                            require(bytes.iter().all(|&byte| byte == 8), "main KV bytes changed")?;
                        }
                    }
                    require(kv_bytes > 0, "empty KV fixture")
                };
                session.reset(&mut counter)?;
                let covered = session.restore_prefix(&mut slot, &prompt, path, |saved| {
                    pool.restore(&ctx, &stream, &mut state, saved, None)?;
                    counter.0 = state.tokens();
                    Ok(())
                })?;
                require(
                    covered == 5 && counter.fed() == 4,
                    "main prompt must restore the intermediate boundary",
                )?;
                require(
                    slot.restored_kind() == Some(SaveKind::Block),
                    "main restore kind",
                )?;
                check_main(&state)?;

                let failed = [1, 2, 3, 4, 44, 45];
                session.reset(&mut counter)?;
                session.prefill(&mut counter, &failed, 16)?;
                state.delta.write_at(&stream, 0, &vec![31; delta_bytes])?;
                let kv = state.resident_kv()?;
                kv.write_at(&stream, 0, &vec![32; kv.bytes()])?;
                state.book.finish_overwrite(counter.fed());
                let frozen = session.prepare_prefix_save(&failed, counter.fed())?;
                let mut pending = Some(pool.begin(&ctx, &stream, &state, None)?);
                pool.invalidate()?;
                let mut called = false;
                let result = frozen.save_prefix(
                    &mut slot,
                    &failed,
                    path,
                    counter.fed(),
                    SaveKind::Tip,
                    || {
                        called = true;
                        let pending = pending.take().ok_or_else(|| {
                            EngError::Checkpoint("duplicate failed publication".into())
                        })?;
                        pool.finish(&ctx, &stream, &state, pending)
                    },
                );
                require(
                    called && pending.is_none(),
                    "failed finish must run and consume staging",
                )?;
                require(
                    matches!(result, Err(EngError::BadRequest(_))),
                    "stale finish must return BadRequest",
                )?;
                require(
                    !slot.keys().iter().any(|&(len, _)| len == failed.len()),
                    "failed key was published",
                )?;
                require(
                    slot.snapshots().any(|saved| saved.tokens() == 4),
                    "failed finish lost main snapshot",
                )?;
                // Making room may already have evicted the short tip before capture failed.
                session.reset(&mut counter)?;
                let covered = session.restore_prefix(&mut slot, &prompt, path, |saved| {
                    pool.restore(&ctx, &stream, &mut state, saved, None)?;
                    counter.0 = state.tokens();
                    Ok(())
                })?;
                require(
                    covered == 5 && counter.fed() == 4,
                    "main boundary after failed publication",
                )?;
                require(
                    slot.restored_kind() == Some(SaveKind::Block),
                    "main kind after failed publication",
                )?;
                check_main(&state)?;
            }
            Ok(())
        })();
        outcome
            .expect("native shared-store checkpoints must survive the side turn and failed finish");
    }

    #[test]
    #[ignore = "requires native CUDA; run explicitly under the GPU lease"]
    fn warm_tip_recycles_only_its_exclusive_live_source() {
        let outcome = (|| -> Result<(), EngError> {
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let key = [1, 2, 3, 4, 5];
            let extended = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            for mode in [KvMode::Fp16, KvMode::Int8G32] {
                for case in [
                    "warm",
                    "aliased",
                    "invalidated",
                    "live-generation",
                    "slot-generation",
                    "other-source",
                    "failed-finish",
                ] {
                    let options = StepOptions {
                        kv_mode: mode,
                        context_tokens: 16,
                        delta_slots: 1,
                        forked_experts: false,
                        tiered: None,
                        layout: WeightLayout::Native,
                    };
                    let mut state =
                        ResidentState::allocate(&ctx, &stream, &common_tests::dims(), &options)?;
                    let mut pool = PrefixPinPool::new(&ctx, state.layout, 2)?;
                    let mut store = PrefixSlot::with_limits(StoreLimits {
                        roots: 0,
                        entries: 2,
                    })
                    .with_recent_longer_protection();
                    let mut session = DecodeSession::default();
                    let mut counter = Counter(0);
                    session.prefill(&mut counter, &key, 16)?;
                    let mut expected: Vec<u8> = (0..state.resident_kv()?.bytes())
                        .map(|i| ((i % 251 * 37 + 11) % 251) as u8)
                        .collect();
                    state.resident_kv()?.write_at(&stream, 0, &expected)?;
                    state
                        .delta
                        .write_at(&stream, 0, &vec![17; pool.delta_bytes])?;
                    state.book.finish_overwrite(4);
                    let pending = pool.begin(&ctx, &stream, &state, None)?;
                    let first = pool.finish(&ctx, &stream, &state, pending)?;
                    let first_slot = first.slot;
                    let mut ready = Some(first);
                    assert!(session.save_prefix(
                        &mut store,
                        &key,
                        0u8,
                        4,
                        SaveKind::Tip,
                        || Ok(ready.take().expect("one publication"))
                    )?);
                    let alias = (case == "aliased")
                        .then(|| Rc::clone(&store.snapshots().next().unwrap().data));
                    if case == "invalidated" {
                        pool.invalidate()?;
                    }
                    if case == "live-generation" {
                        pool.live_source = Some((
                            first_slot,
                            pool.generations[first_slot] + 1,
                            Rc::clone(&state.instance),
                        ));
                    }
                    if case == "slot-generation" {
                        pool.generations[first_slot] += 1;
                    }
                    let other = if case == "other-source" {
                        let mut foreign = ResidentState::allocate(
                            &ctx,
                            &stream,
                            &common_tests::dims(),
                            &options,
                        )?;
                        foreign.book.finish_overwrite(4);
                        assert!(
                            !pool.can_recycle(&stream, &foreign, store.snapshots().next().unwrap()),
                            "same-layout state is not the live KV owner"
                        );
                        let pending = pool.begin(&ctx, &stream, &state, None)?;
                        assert!(matches!(
                            pool.finish(&ctx, &stream, &foreign, pending),
                            Err(EngError::BadRequest(_))
                        ));
                        pool.restore(
                            &ctx,
                            &stream,
                            &mut state,
                            store.snapshots().next().unwrap(),
                            None,
                        )?;
                        let pending = pool.begin(&ctx, &stream, &state, None)?;
                        Some(pool.finish(&ctx, &stream, &state, pending)?)
                    } else {
                        None
                    };
                    let mut full_bytes = 0;
                    let mut suffix_bytes = 0;
                    for layer in 0..state.layout.kv.layers {
                        let old = state.layout.kv.prefix_ranges(layer, 4)?;
                        for (new, old) in state
                            .layout
                            .kv
                            .prefix_ranges(layer, 8)?
                            .into_iter()
                            .zip(old)
                        {
                            let start = new.device_offset + old.bytes;
                            let end = new.device_offset + new.bytes;
                            for (offset, byte) in expected[start..end].iter_mut().enumerate() {
                                *byte = (((start + offset) % 251 * 19 + 83) % 251) as u8;
                            }
                            state
                                .resident_kv()?
                                .write_at(&stream, start, &expected[start..end])?;
                            full_bytes += new.bytes;
                            suffix_bytes += new.bytes - old.bytes;
                        }
                    }
                    assert!(
                        suffix_bytes > 0 && full_bytes > suffix_bytes,
                        "nonempty, distinct full and suffix transfers"
                    );
                    state
                        .delta
                        .write_at(&stream, 0, &vec![29; pool.delta_bytes])?;
                    state.book.finish_overwrite(8);
                    session.reset(&mut counter)?;
                    session.prefill(&mut counter, &extended, 16)?;
                    let recycled = store.take_tip_if(&extended, &0u8, |saved| {
                        pool.can_recycle(&stream, &state, saved)
                    });
                    let inherit = matches!(case, "warm" | "failed-finish");
                    assert_eq!(recycled.is_some(), inherit, "{mode:?}/{case}");
                    let started = std::time::Instant::now();
                    let pending = pool.begin(&ctx, &stream, &state, recycled)?;
                    assert_eq!(pending.inherited_tokens, if inherit { 4 } else { 0 });
                    if inherit {
                        assert_eq!(pending.saved.slot, first_slot);
                        assert_eq!(
                            Rc::strong_count(&pool.slots[first_slot]),
                            2,
                            "pool plus pending only"
                        );
                    }
                    if case == "failed-finish" {
                        pool.invalidate()?;
                        assert!(matches!(
                            pool.finish(&ctx, &stream, &state, pending),
                            Err(EngError::BadRequest(_))
                        ));
                        assert!(
                            store.keys().is_empty(),
                            "neither retired nor failed key is restorable"
                        );
                        assert!(pool.live_source.is_none());
                        eprintln!(
                            "PINNED_WARM_NATIVE {mode:?}/{case}: expected finish error; restorable keys 0"
                        );
                        continue;
                    }
                    let saved = pool.finish(&ctx, &stream, &state, pending)?;
                    let save_ms = started.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "PINNED_WARM_NATIVE {mode:?}/{case}: KV copied {} B, full {full_bytes} B, suffix {suffix_bytes} B, native_save_ms {save_ms:.3}",
                        saved.copied_kv_bytes()
                    );
                    assert_eq!(
                        saved.copied_kv_bytes(),
                        if inherit { suffix_bytes } else { full_bytes },
                        "{mode:?}/{case}"
                    );
                    // A reader that blocked recycling must still see the original prefix.
                    if let Some(first) = store
                        .snapshots()
                        .next()
                        .filter(|_| case != "slot-generation")
                    {
                        pool.restore(&ctx, &stream, &mut state, first, None)?;
                        for layer in 0..state.layout.kv.layers {
                            for range in state.layout.kv.prefix_ranges(layer, 4)? {
                                let mut got = vec![0; range.bytes];
                                state.resident_kv()?.read_at(
                                    &stream,
                                    range.device_offset,
                                    &mut got,
                                )?;
                                assert_eq!(
                                    got,
                                    expected
                                        [range.device_offset..range.device_offset + range.bytes]
                                );
                            }
                        }
                    }
                    if case == "slot-generation" {
                        let first = store.snapshots().next().unwrap();
                        assert!(matches!(
                            pool.restore(&ctx, &stream, &mut state, first, None),
                            Err(EngError::BadRequest(_))
                        ));
                    }
                    pool.restore(&ctx, &stream, &mut state, &saved, None)?;
                    assert!(
                        pool.can_recycle(&stream, &state, &saved),
                        "host restore establishes source"
                    );
                    let mut delta = vec![0; pool.delta_bytes];
                    state.delta.read_at(&stream, 0, &mut delta)?;
                    assert_eq!(delta, vec![29; pool.delta_bytes]);
                    for layer in 0..state.layout.kv.layers {
                        for range in state.layout.kv.prefix_ranges(layer, 8)? {
                            let mut got = vec![0; range.bytes];
                            state
                                .resident_kv()?
                                .read_at(&stream, range.device_offset, &mut got)?;
                            assert_eq!(
                                got,
                                expected[range.device_offset..range.device_offset + range.bytes],
                                "{mode:?}/{case}"
                            );
                        }
                    }
                    drop((alias, other));
                }
            }
            Ok(())
        })();
        outcome.expect("exclusive live-source suffix copies and failure quarantine");
    }

    #[test]
    #[ignore = "requires native CUDA; run explicitly under the GPU lease"]
    fn persistent_pool_freezes_delta_and_keeps_incremental_ancestor() {
        let ctx = CudaContext::new().expect("native CUDA context");
        let stream = ctx.stream().expect("stream");
        for mode in [KvMode::Fp16, KvMode::Int8G32] {
            let options = StepOptions {
                kv_mode: mode,
                context_tokens: 16,
                delta_slots: 1,
                forked_experts: false,
                tiered: None,
                layout: WeightLayout::Native,
            };
            let mut state = ResidentState::allocate(&ctx, &stream, &common_tests::dims(), &options)
                .expect("state");
            let mut pool = PrefixPinPool::new(&ctx, state.layout, 2).expect("pool");
            let original = vec![17u8; pool.delta_bytes];
            state.delta.write_at(&stream, 0, &original).expect("delta");
            state
                .resident_kv()
                .unwrap()
                .write_at(&stream, 0, &vec![7; state.resident_kv().unwrap().bytes()])
                .expect("KV");
            state.book.finish_overwrite(4);
            let pending = pool
                .begin(&ctx, &stream, &state, None)
                .expect("freeze point");
            state
                .delta
                .write_at(&stream, 0, &vec![29; pool.delta_bytes])
                .expect("advance delta");
            for layer in 0..state.layout.kv.layers {
                let old = state.layout.kv.prefix_ranges(layer, 4).unwrap();
                let new = state.layout.kv.prefix_ranges(layer, 8).unwrap();
                for (new, old) in new.iter().zip(old) {
                    state
                        .resident_kv()
                        .unwrap()
                        .write_at(
                            &stream,
                            new.device_offset + old.bytes,
                            &vec![9; new.bytes - old.bytes],
                        )
                        .expect("append suffix");
                }
            }
            state.book.finish_overwrite(8);
            let first = pool
                .finish(&ctx, &stream, &state, pending)
                .expect("finish old point");
            pool.restore(&ctx, &stream, &mut state, &first, None)
                .expect("restore first");
            let mut got = vec![0; pool.delta_bytes];
            state.delta.read_at(&stream, 0, &mut got).unwrap();
            assert!(
                got == original,
                "the prompt tail must not change the frozen Delta"
            );
            assert_eq!(state.tokens(), 4);
            let later = vec![33; pool.delta_bytes];
            state.delta.write_at(&stream, 0, &later).unwrap();
            state.book.finish_overwrite(8);
            let pending = pool
                .begin(&ctx, &stream, &state, None)
                .expect("incremental point");
            let second = pool
                .finish(&ctx, &stream, &state, pending)
                .expect("incremental finish");
            let pending = pool
                .begin(&ctx, &stream, &state, None)
                .expect("unused staging slot");
            let third = pool
                .finish(&ctx, &stream, &state, pending)
                .expect("complete third slot");
            assert!(matches!(
                pool.begin(&ctx, &stream, &state, None),
                Err(EngError::MemoryPlan(_))
            ));
            pool.restore(&ctx, &stream, &mut state, &first, None)
                .expect("retained ancestor");
            state.delta.read_at(&stream, 0, &mut got).unwrap();
            assert!(got == original, "retained ancestor Delta differs");
            for saved in [&second, &third] {
                pool.restore(&ctx, &stream, &mut state, saved, None)
                    .expect("independent full prefix");
                state.delta.read_at(&stream, 0, &mut got).unwrap();
                assert!(got == later, "independent slot Delta differs");
                for layer in 0..state.layout.kv.layers {
                    let old = state.layout.kv.prefix_ranges(layer, 4).unwrap();
                    let new = state.layout.kv.prefix_ranges(layer, 8).unwrap();
                    for (new, old) in new.iter().zip(old) {
                        let mut bytes = vec![0; new.bytes];
                        state
                            .resident_kv()
                            .unwrap()
                            .read_at(&stream, new.device_offset, &mut bytes)
                            .unwrap();
                        assert!(bytes[..old.bytes].iter().all(|&b| b == 7));
                        assert!(bytes[old.bytes..].iter().all(|&b| b == 9));
                    }
                }
            }
        }
        println!("PINNED_PREFIX_POOL_NATIVE_PASS");
    }
}
