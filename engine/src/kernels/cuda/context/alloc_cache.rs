//! Per-lease exact-size backing reuse; idle entries never own their cache.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{DeviceBuffer, DispatchGuard, EngError, LIVE_BYTES, Lease, RawDevice, TERMINAL};
use super::{device_error, graph, native};

/// Exact-size reuse is eligible at <=1 MiB, covering measured Flint decode transients.
pub(super) const MAX_BLOCK_BYTES: usize = 1_048_576;
/// Retain <=16 MiB requested bytes; CUDA padding is measured separately against headroom.
pub(super) const MAX_IDLE_BYTES: usize = 16 * 1_048_576;
/// Bound metadata and tiny allocations even when the requested-byte cap is not reached.
const MAX_IDLE_SLOTS: usize = 256;

#[cfg(all(test, feature = "cuda-fault-qualification"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TestFaultStage {
    Sync,
    BeforeOwnedCopy,
    BeforeNativeFree,
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct TestNativeCalls {
    sync_attempts: usize,
    copy_attempts: usize,
    free_attempts: usize,
    sync_status: Option<std::ffi::c_int>,
    copy_status: Option<std::ffi::c_int>,
    free_status: Option<std::ffi::c_int>,
}

pub(super) struct AllocCache {
    lease: Rc<Lease>,
    idle: RefCell<Vec<RawDevice>>,
    idle_bytes: Cell<usize>,
    terminal: AtomicBool,
    #[cfg(test)]
    hits: Cell<usize>,
    #[cfg(test)]
    misses: Cell<usize>,
    #[cfg(test)]
    quarantined_bytes: Cell<usize>,
    #[cfg(test)]
    poisoned_pending: RefCell<Vec<u64>>,
    #[cfg(test)]
    poisoned_hits: Cell<usize>,
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    test_fault_target: Cell<Option<TestFaultStage>>,
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    test_fault_fired: Cell<Option<TestFaultStage>>,
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    test_sync_successes: Cell<usize>,
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    test_sync_failure: Cell<Option<std::ffi::c_int>>,
    // The observer outlives teardown without keeping the cache or its lease alive.
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    test_native_calls: Rc<Cell<TestNativeCalls>>,
}

impl AllocCache {
    pub(super) fn new(lease: Rc<Lease>) -> Result<Self, EngError> {
        Self::with_metadata_capacity(lease, MAX_IDLE_SLOTS)
    }

    fn with_metadata_capacity(lease: Rc<Lease>, capacity: usize) -> Result<Self, EngError> {
        let mut idle = Vec::new();
        idle.try_reserve_exact(capacity).map_err(|error| {
            EngError::MemoryPlan(format!("CUDA allocation-cache metadata: {error}"))
        })?;
        Ok(Self {
            lease,
            idle: RefCell::new(idle),
            idle_bytes: Cell::new(0),
            terminal: AtomicBool::new(false),
            #[cfg(test)]
            hits: Cell::new(0),
            #[cfg(test)]
            misses: Cell::new(0),
            #[cfg(test)]
            quarantined_bytes: Cell::new(0),
            #[cfg(test)]
            poisoned_pending: RefCell::new(Vec::new()),
            #[cfg(test)]
            poisoned_hits: Cell::new(0),
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            test_fault_target: Cell::new(None),
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            test_fault_fired: Cell::new(None),
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            test_sync_successes: Cell::new(0),
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            test_sync_failure: Cell::new(None),
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            test_native_calls: Rc::new(Cell::new(TestNativeCalls::default())),
        })
    }

    fn enter(&self) -> Result<DispatchGuard<'static>, EngError> {
        // A terminal destructor may run while capture is abandoning its RefCell.
        if self.is_terminal() {
            self.mark_terminal();
            super::super::ensure_usable(&self.terminal)?;
        }
        graph::ensure_closed("allocation-cache access")?;
        let dispatch = super::enter_dispatch();
        if dispatch.is_err() {
            self.mark_terminal();
        }
        dispatch
    }

    pub(super) fn allocate(self: &Rc<Self>, bytes: usize) -> Result<DeviceBuffer, EngError> {
        let _dispatch = self.enter()?;
        let mut idle = self.idle.borrow_mut();
        let raw = if let Some(index) = idle.iter().position(|raw| raw.bytes() == bytes) {
            let raw = idle.swap_remove(index);
            self.idle_bytes.set(self.idle_bytes.get() - bytes);
            #[cfg(test)]
            {
                self.hits.set(self.hits.get().saturating_add(1));
                let mut pending = self.poisoned_pending.borrow_mut();
                if let Some(index) = pending.iter().position(|&address| address == raw.address()) {
                    pending.swap_remove(index);
                    self.poisoned_hits.set(self.poisoned_hits.get() + 1);
                }
            }
            raw
        } else {
            drop(idle);
            #[cfg(test)]
            self.misses.set(self.misses.get().saturating_add(1));
            let raw = native::alloc(bytes).map_err(|failure| {
                let error = device_error(failure);
                if TERMINAL.load(Ordering::Acquire) {
                    self.mark_terminal();
                }
                error
            })?;
            LIVE_BYTES.fetch_add(bytes, Ordering::AcqRel);
            return Ok(self.wrap(raw));
        };
        Ok(self.wrap(raw))
    }

    fn wrap(self: &Rc<Self>, raw: RawDevice) -> DeviceBuffer {
        DeviceBuffer {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            _thread_confined: std::marker::PhantomData,
            alloc_cache: Some(Rc::clone(self)),
        }
    }

    pub(super) fn copy(&self, raw: &RawDevice, bytes: &[u8]) -> Result<(), EngError> {
        let _dispatch = self.enter()?;
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        self.inject(TestFaultStage::BeforeOwnedCopy);
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        {
            let mut calls = self.test_native_calls.get();
            calls.copy_attempts += 1;
            self.test_native_calls.set(calls);
        }
        let result = native::upload_at(raw, 0, bytes);
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        {
            let mut calls = self.test_native_calls.get();
            calls.copy_status = Some(result.as_ref().err().map_or(0, |error| error.status));
            self.test_native_calls.set(calls);
        }
        result.map_err(|failure| {
            self.mark_terminal();
            device_error(failure)
        })
    }

    fn synchronize(&self) -> Result<(), EngError> {
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        self.inject(TestFaultStage::Sync);
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        {
            let mut calls = self.test_native_calls.get();
            calls.sync_attempts += 1;
            self.test_native_calls.set(calls);
        }
        let result = native::device_synchronize();
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        {
            let mut calls = self.test_native_calls.get();
            calls.sync_status = Some(result.as_ref().err().map_or(0, |error| error.status));
            self.test_native_calls.set(calls);
            match &result {
                Ok(()) => self
                    .test_sync_successes
                    .set(self.test_sync_successes.get() + 1),
                Err(failure) => self.test_sync_failure.set(Some(failure.status)),
            }
        }
        result.map_err(|failure| {
            self.mark_terminal();
            device_error(failure)
        })
    }

    pub(super) fn return_block(&self, raw: RawDevice) {
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        if self.is_terminal() {
            self.quarantine(raw);
            return;
        }
        if self.synchronize().is_err() {
            self.quarantine(raw);
            return;
        }
        let bytes = raw.bytes();
        let mut idle = self.idle.borrow_mut();
        if idle.len() < MAX_IDLE_SLOTS
            && idle.len() < idle.capacity()
            && bytes <= MAX_IDLE_BYTES - self.idle_bytes.get()
        {
            self.idle_bytes.set(self.idle_bytes.get() + bytes);
            idle.push(raw);
        } else {
            drop(idle);
            let _ = self.free(raw);
        }
    }

    pub(super) fn free(&self, raw: RawDevice) -> Result<(), EngError> {
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        if self.is_terminal() {
            self.quarantine(raw);
            return super::super::ensure_usable(&self.terminal);
        }
        #[cfg(all(test, feature = "cuda-fault-qualification"))]
        self.inject(TestFaultStage::BeforeNativeFree);
        #[cfg(test)]
        let bytes = raw.bytes();
        #[cfg(test)]
        self.poisoned_pending
            .borrow_mut()
            .retain(|&address| address != raw.address());
        let result = free_backing(
            raw,
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            &self.test_native_calls,
        );
        if result.is_err() {
            self.mark_terminal();
            #[cfg(test)]
            self.quarantined_bytes
                .set(self.quarantined_bytes.get().saturating_add(bytes));
        }
        result
    }

    pub(super) fn drain(&self) -> Result<(), EngError> {
        let _dispatch = self.enter()?;
        if self.idle.borrow().is_empty() {
            return Ok(());
        }
        self.synchronize()?;
        loop {
            let raw = self.idle.borrow_mut().pop();
            let Some(raw) = raw else { return Ok(()) };
            self.idle_bytes.set(self.idle_bytes.get() - raw.bytes());
            self.free(raw)?;
        }
    }

    pub(super) fn quarantine(&self, _block: RawDevice) {
        self.mark_terminal();
        #[cfg(test)]
        self.quarantined_bytes
            .set(self.quarantined_bytes.get().saturating_add(_block.bytes()));
    }

    pub(super) fn mark_terminal(&self) {
        self.terminal.store(true, Ordering::Release);
        #[cfg(test)]
        self.quarantined_bytes.set(
            self.quarantined_bytes
                .get()
                .saturating_add(self.idle_bytes.get()),
        );
        self.idle_bytes.set(0);
        self.idle.borrow_mut().clear();
    }

    pub(super) fn is_terminal(&self) -> bool {
        self.terminal.load(Ordering::Acquire) || TERMINAL.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn hits(&self) -> usize {
        self.hits.get()
    }
    #[cfg(test)]
    pub(super) fn misses(&self) -> usize {
        self.misses.get()
    }
    #[cfg(test)]
    pub(super) fn idle_bytes(&self) -> usize {
        self.idle_bytes.get()
    }
    #[cfg(test)]
    pub(super) fn idle_slots(&self) -> usize {
        self.idle.borrow().len()
    }
    #[cfg(test)]
    pub(super) fn quarantined_bytes(&self) -> usize {
        self.quarantined_bytes.get()
    }

    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    fn arm_fail_next_sync(&self) {
        self.arm_test_fault(TestFaultStage::Sync);
    }

    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    fn arm_test_fault(&self, target: TestFaultStage) {
        self.test_fault_target.set(Some(target));
    }

    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    fn inject(&self, target: TestFaultStage) {
        if self.test_fault_target.get() == Some(target) {
            self.test_fault_target.set(None);
            self.test_fault_fired.set(Some(target));
            assert_eq!(crate::kernels::cuda::test_inject_unmapped_fault(), Err(700));
        }
    }
}

impl Drop for AllocCache {
    fn drop(&mut self) {
        if self.is_terminal() {
            self.mark_terminal();
        } else if graph::window_open() {
            let blocks = std::mem::take(self.idle.get_mut());
            let lease = Rc::clone(&self.lease);
            #[cfg(all(test, feature = "cuda-fault-qualification"))]
            let observer = Rc::clone(&self.test_native_calls);
            graph::run_or_defer(move || {
                let _lease = lease;
                let _dispatch = DispatchGuard::enter(&TERMINAL);
                for raw in blocks {
                    if TERMINAL.load(Ordering::Acquire) {
                        break;
                    }
                    if free_backing(
                        raw,
                        #[cfg(all(test, feature = "cuda-fault-qualification"))]
                        &observer,
                    )
                    .is_err()
                    {
                        break;
                    }
                }
            });
        } else {
            let _ = self.drain();
        }
    }
}

fn free_error(failure: native::DeviceFailure) -> EngError {
    // Any failed free is terminal, even a nonsticky allocation-stage OOM.
    TERMINAL.store(true, Ordering::Release);
    device_error(failure)
}

fn free_backing(
    raw: RawDevice,
    #[cfg(all(test, feature = "cuda-fault-qualification"))] observer: &Cell<TestNativeCalls>,
) -> Result<(), EngError> {
    let bytes = raw.bytes();
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    {
        let mut calls = observer.get();
        calls.free_attempts += 1;
        observer.set(calls);
    }
    let result = native::free(raw);
    #[cfg(all(test, feature = "cuda-fault-qualification"))]
    {
        let mut calls = observer.get();
        calls.free_status = Some(result.as_ref().err().map_or(0, |error| error.status));
        observer.set(calls);
    }
    result.map_err(free_error)?;
    LIVE_BYTES.fetch_sub(bytes, Ordering::AcqRel);
    Ok(())
}

#[cfg(test)]
impl super::CudaContext {
    pub(crate) fn test_watch_poisoned_cache_block(&self, buffer: &DeviceBuffer) {
        let cache = self.alloc_cache.as_ref().expect("cache enabled");
        assert!(Rc::ptr_eq(
            cache,
            buffer.alloc_cache.as_ref().expect("cached backing")
        ));
        let address = buffer
            .raw
            .as_ref()
            .expect("live poisoned backing")
            .address();
        let mut pending = cache.poisoned_pending.borrow_mut();
        assert!(!pending.contains(&address), "distinct poisoned backing");
        pending.push(address);
    }

    pub(crate) fn test_poisoned_cache_hits(&self) -> usize {
        self.alloc_cache
            .as_ref()
            .map_or(0, |cache| cache.poisoned_hits.get())
    }
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
impl super::CudaContext {
    pub(crate) fn test_cache_fail_next_sync(&self) {
        self.alloc_cache
            .as_ref()
            .expect("cache sync hook requires an opted-in lease")
            .arm_fail_next_sync();
    }

    pub(crate) fn test_cache_sync_fault_observed(&self) -> bool {
        let cache = self.alloc_cache.as_ref().expect("opted-in cache");
        cache.test_fault_fired.get() == Some(TestFaultStage::Sync)
            && cache.test_sync_failure.get() == Some(700)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::{cell::Cell, rc::Rc};

    use super::super::{CudaContext, SLOTS, TERMINAL, native};
    #[cfg(feature = "cuda-fault-qualification")]
    use super::TestFaultStage;
    use super::{AllocCache, Lease, MAX_BLOCK_BYTES, MAX_IDLE_BYTES, MAX_IDLE_SLOTS};
    use crate::EngError;
    use crate::kernels::cuda::{CapturePlan, guard::DispatchGuard, native::Activations};

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn capture_deferred_buffer_drop_retains_its_lease_until_close() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let ctx = CudaContext::with_workspace(caps).expect("capture context");
        let stream = ctx.stream().expect("capture stream");
        // The capture's own context must not mask early release of the second lease.
        let second = CudaContext::with_workspace(caps).expect("second lease");
        assert!(
            ctx.alloc_cache.is_none() && second.alloc_cache.is_none(),
            "OFF-arm regression"
        );
        let buffer = second.allocate(16).expect("second lease buffer");
        let inside = Cell::new(0);
        let outcome = ctx.capture(
            &stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || -> Result<(), EngError> {
                drop(buffer);
                drop(second);
                inside.set(SLOTS.load(Ordering::Acquire));
                Err(EngError::BadRequest("deferred lease probe".into()))
            },
        );
        let expected_error = matches!(
            &outcome,
            Err(EngError::BadRequest(message)) if message == "deferred lease probe"
        );
        drop(outcome);
        let after_close = SLOTS.load(Ordering::Acquire);
        let healthy = ctx.ensure_usable().is_ok();
        drop(stream);
        drop(ctx);
        // A RED assertion must not unwind through capture or poison teardown.
        assert!(
            expected_error,
            "capture must close through the intentional error"
        );
        assert!(
            healthy,
            "the intentional body error must remain non-terminal"
        );
        assert_eq!(after_close, 1, "deferred frees release the second lease");
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "healthy teardown");
        assert_eq!(inside.get(), 2, "deferred backing must retain its lease");
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn prefill_sized_idle_pool_keeps_warm_decode_backing() {
        fn allocate_classes(
            ctx: &CudaContext,
            classes: &[(usize, usize)],
        ) -> Vec<super::super::DeviceBuffer> {
            let mut blocks = Vec::new();
            for &(bytes, count) in classes {
                for _ in 0..count {
                    blocks.push(ctx.allocate(bytes).expect("working-set backing"));
                }
            }
            blocks
        }

        // Measured post-prefill inventory nearly fills the old 8 MiB byte cap.
        let prefill = [
            (4, 2),
            (24, 3),
            (28, 1),
            (32, 1),
            (144, 5),
            (148, 1),
            (168, 5),
            (172, 1),
            (192, 5),
            (196, 1),
            (1024, 2),
            (2048, 3),
            (4096, 4),
            (6144, 1),
            (7168, 2),
            (8192, 1),
            (14336, 4),
            (16384, 1),
            (28672, 8),
            (32768, 8),
            (57344, 11),
            (65536, 6),
            (114688, 4),
            (131072, 2),
            (172032, 8),
            (196608, 3),
            (344064, 4),
            (393216, 4),
            (516096, 1),
            (589824, 1),
        ];
        let decode = [
            (4, 2),
            (24, 3),
            (28, 1),
            (1024, 2),
            (2048, 4),
            (4096, 8),
            (8192, 11),
            (16384, 4),
            (24576, 8),
            (49152, 3),
            (73728, 1),
            (600576, 2),
        ];
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let ctx = CudaContext::with_workspace_cache(caps, true).expect("cache context");
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        drop(allocate_classes(&ctx, &prefill));
        assert_eq!((cache.idle_bytes(), cache.idle_slots()), (8_387_688, 103));
        drop(allocate_classes(&ctx, &decode));
        let before = (cache.hits(), cache.misses());
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        for _ in 0..2 {
            drop(allocate_classes(&ctx, &decode));
        }
        let after = (cache.hits(), cache.misses());
        #[cfg(feature = "cuda-fault-qualification")]
        let warm_physical = native::allocation_calls();
        let idle = (cache.idle_bytes(), cache.idle_slots());
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "healthy lease teardown");
        assert!(idle.0 <= MAX_IDLE_BYTES && idle.1 <= MAX_IDLE_SLOTS);
        assert_eq!(
            after.1, before.1,
            "prefill must leave room for warm decode backing"
        );
        assert_eq!(after.0 - before.0, 98, "both decode batches must hit");
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            warm_physical, physical,
            "warm decode has no backing traffic"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn two_cache_leases_keep_eight_mib_each_without_cross_reuse() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let a = CudaContext::with_workspace_cache(caps, true).expect("lease A");
        let b = CudaContext::with_workspace_cache(caps, true).expect("lease B");
        let ca = Rc::clone(a.alloc_cache.as_ref().unwrap());
        let cb = Rc::clone(b.alloc_cache.as_ref().unwrap());
        assert!(!Rc::ptr_eq(&ca, &cb), "per-lease metadata");
        assert_eq!(SLOTS.load(Ordering::Acquire), 2);
        for ctx in [&a, &b] {
            let blocks: Vec<_> = (0..=MAX_IDLE_BYTES / MAX_BLOCK_BYTES)
                .map(|_| ctx.allocate(MAX_BLOCK_BYTES).expect("1 MiB backing"))
                .collect();
            drop(blocks);
        }
        let idle_before = (ca.idle_bytes(), cb.idle_bytes());
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let left = a.upload(&vec![0xA5; MAX_BLOCK_BYTES]).expect("A cache hit");
        let b_while_a_borrowed = cb.idle_bytes();
        let right = b.upload(&vec![0x5A; MAX_BLOCK_BYTES]).expect("B cache hit");
        let mut left_bytes = vec![0; MAX_BLOCK_BYTES];
        let mut right_bytes = vec![0; MAX_BLOCK_BYTES];
        a.download(&left, &mut left_bytes).unwrap();
        b.download(&right, &mut right_bytes).unwrap();
        drop(left);
        drop(right);
        #[cfg(feature = "cuda-fault-qualification")]
        let warm_physical = native::allocation_calls();
        let hits = (ca.hits(), cb.hits());
        let idle_after = (ca.idle_bytes(), cb.idle_bytes());
        a.drain_alloc_cache().unwrap();
        b.drain_alloc_cache().unwrap();
        drop(a);
        drop(b);
        drop(ca);
        drop(cb);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "healthy lease teardown");
        assert!(left_bytes.iter().all(|&byte| byte == 0xA5));
        assert!(right_bytes.iter().all(|&byte| byte == 0x5A));
        assert_eq!(
            idle_before,
            (MAX_IDLE_BYTES, MAX_IDLE_BYTES),
            "each lease must retain its own idle byte cap"
        );
        assert_eq!(
            b_while_a_borrowed, MAX_IDLE_BYTES,
            "A cannot take B idle backing"
        );
        assert_eq!(
            idle_after, idle_before,
            "each lease remains within its requested idle byte cap"
        );
        assert_eq!(hits, (1, 1), "each lease records its own hit");
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            warm_physical, physical,
            "warm leases do not allocate or free backing"
        );
    }

    #[test]
    fn metadata_capacity_failure_is_memory_plan_without_leaking_its_lease() {
        static SLOTS: AtomicUsize = AtomicUsize::new(0);
        static FAILED: AtomicBool = AtomicBool::new(false);
        let lease = Lease::acquire(&SLOTS, &FAILED).expect("metadata lease");
        let result = AllocCache::with_metadata_capacity(lease, usize::MAX);
        let memory_plan = matches!(&result, Err(EngError::MemoryPlan(_)));
        drop(result);
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            0,
            "failed metadata owns no lease"
        );
        assert!(
            !FAILED.load(Ordering::Acquire),
            "metadata failure is non-terminal"
        );
        assert!(
            memory_plan,
            "try_reserve_exact capacity failure must return MemoryPlan"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn exactly_one_mib_is_eligible_and_stays_live_while_idle() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let before = ctx.live_bytes();
        drop(
            ctx.allocate(MAX_BLOCK_BYTES)
                .expect("inclusive 1 MiB boundary"),
        );
        let idle = cache.idle_bytes();
        let held = ctx.live_bytes();
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "healthy teardown");
        assert_eq!(
            idle, MAX_BLOCK_BYTES,
            "exactly 1 MiB must become idle backing"
        );
        assert_eq!(
            held,
            before + MAX_BLOCK_BYTES,
            "idle backing remains in LIVE_BYTES"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn control_one_mib_plus_sixteen_bypasses_the_cache() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let before = ctx.live_bytes();
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        drop(
            ctx.allocate(MAX_BLOCK_BYTES + 16)
                .expect("oversize boundary"),
        );
        assert_eq!((cache.idle_bytes(), cache.idle_slots()), (0, 0));
        assert_eq!(
            ctx.live_bytes(),
            before,
            "oversize backing is not held idle"
        );
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(native::allocation_calls(), (physical.0 + 1, physical.1 + 1));

        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn metadata_entry_limit_frees_the_257th_eligible_block() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let blocks: Vec<_> = (0..=MAX_IDLE_SLOTS)
            .map(|_| {
                ctx.allocate(16)
                    .expect("simultaneously live eligible backing")
            })
            .collect();
        drop(blocks);
        let entries = cache.idle_slots();
        let bytes = cache.idle_bytes();
        #[cfg(feature = "cuda-fault-qualification")]
        let after = native::allocation_calls();
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "healthy teardown");
        assert_eq!(
            entries, MAX_IDLE_SLOTS,
            "metadata must retain exactly 256 entries"
        );
        assert_eq!(bytes, MAX_IDLE_SLOTS * 16);
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            after,
            (physical.0 + MAX_IDLE_SLOTS as u64 + 1, physical.1 + 1),
            "one cap-overflow free"
        );
    }

    #[test]
    fn control_empty_cache_has_reviewed_caps_and_releases_its_lease() {
        static SLOTS: AtomicUsize = AtomicUsize::new(0);
        static FAILED: AtomicBool = AtomicBool::new(false);
        assert_eq!(
            (MAX_BLOCK_BYTES, MAX_IDLE_BYTES, MAX_IDLE_SLOTS),
            (1_048_576, 16_777_216, 256)
        );
        let lease = Lease::acquire(&SLOTS, &FAILED).expect("empty cache lease");
        let cache = AllocCache::new(lease).expect("empty cache metadata");
        assert_eq!(
            (
                cache.hits(),
                cache.misses(),
                cache.idle_bytes(),
                cache.idle_slots()
            ),
            (0, 0, 0, 0)
        );
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn terminal_metadata_refuses_entry_and_releases_its_lease() {
        static SLOTS: AtomicUsize = AtomicUsize::new(0);
        static FAILED: AtomicBool = AtomicBool::new(false);
        let lease = Lease::acquire(&SLOTS, &FAILED).expect("metadata lease");
        let cache = AllocCache::new(lease).unwrap();
        cache.mark_terminal();
        assert!(matches!(cache.enter(), Err(EngError::Backend { .. })));
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "irreversible terminal latch; CPU-only, run alone with --exact"]
    fn metadata_tests_remain_safe_after_a_process_terminal_probe() {
        TERMINAL.store(true, Ordering::Release);
        control_empty_cache_has_reviewed_caps_and_releases_its_lease();
        terminal_metadata_refuses_entry_and_releases_its_lease();
    }

    #[test]
    #[ignore = "irreversible terminal latch; CPU-only, run alone with --exact"]
    fn failed_free_latches_backend_even_for_allocation_oom() {
        assert!(
            !TERMINAL.load(Ordering::Acquire),
            "healthy exclusive process"
        );
        let mapped = super::free_error(native::DeviceFailure {
            status: 2,
            stage: native::STAGE_ALLOCATE,
        });
        assert!(
            matches!(mapped, EngError::MemoryPlan(_)),
            "preserve failure mapping"
        );
        assert!(
            TERMINAL.load(Ordering::Acquire),
            "any failed free is terminal"
        );
        assert!(
            Lease::acquire(&SLOTS, &TERMINAL).is_err(),
            "failed free refuses new leases"
        );
    }

    #[test]
    #[ignore = "irreversible dispatch poison; run alone with --exact --test-threads=1"]
    fn dispatch_poison_refuses_cache_drain() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        assert!(!TERMINAL.load(Ordering::Acquire), "healthy process");
        let lease = Lease::acquire(&SLOTS, &TERMINAL).expect("real adapter lease");
        let cache = AllocCache::new(lease).expect("cache metadata");
        let uploads = native::upload_entry_calls();
        let unwound = std::panic::catch_unwind(|| {
            let _dispatch = DispatchGuard::enter(&TERMINAL);
            panic!("intentional dispatch-poison probe");
        })
        .is_err();
        let terminal = TERMINAL.load(Ordering::Acquire);
        let drained = cache.drain();
        drop(cache);
        assert!(unwound, "the dispatch unwind must be caught");
        assert!(terminal, "the real dispatch guard must latch terminal");
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            1,
            "terminal keeps its lease slot"
        );
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "terminal drain must not upload"
        );
        assert!(
            matches!(drained, Err(EngError::Backend { .. })),
            "dispatch poison must refuse cache drain"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "irreversible native CUDA fault; run alone with --exact --test-threads=1"]
    fn failed_native_sync_quarantines_idle_and_borrowed_backing() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx = CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true)
            .expect("opted-in cache context");
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        drop(ctx.allocate(16).expect("idle backing before fault"));
        let borrowed = ctx.allocate(32).expect("borrowed backing before fault");
        let charged = ctx.live_bytes();
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let observer = Rc::clone(&cache.test_native_calls);
        let calls_before = observer.get();
        let fault = crate::kernels::cuda::test_inject_unmapped_fault();
        let unmapped = !TERMINAL.load(Ordering::Acquire);
        let drained = ctx.drain_alloc_cache();
        let sync_failure = cache.test_sync_failure.get();
        let calls_after = observer.get();
        let latched = TERMINAL.load(Ordering::Acquire);
        let cache_terminal = cache.is_terminal();
        let idle_slots = cache.idle_slots();
        let idle_quarantine = cache.quarantined_bytes();
        let refused = matches!(ctx.upload(&[0xA5; 16]), Err(EngError::Backend { .. }));
        // The RED drain can miss the sticky fault; force safe teardown after recording its latch.
        TERMINAL.store(true, Ordering::Release);
        drop(borrowed);
        drop(ctx);
        let quarantined = cache.quarantined_bytes();
        drop(cache);
        assert_eq!(fault, Err(700), "existing real illegal-address fixture");
        assert!(
            unmapped,
            "the cache must map the previously unmapped native fault"
        );
        assert!(
            matches!(&drained, Err(EngError::Backend { operation: "cuda synchronize", message, .. }) if message.contains("status 700")),
            "failed native cache sync must return Backend"
        );
        assert_eq!(calls_after.sync_attempts, calls_before.sync_attempts + 1);
        assert_eq!(calls_after.sync_status, Some(700));
        assert_eq!(calls_after.free_attempts, calls_before.free_attempts);
        assert_eq!(calls_after.copy_attempts, calls_before.copy_attempts);
        assert_eq!(
            observer.get(),
            calls_after,
            "terminal refusal and teardown cannot retry native boundaries"
        );
        assert_eq!(
            sync_failure,
            Some(700),
            "the cache's own sync reports the failure"
        );
        assert!(
            latched && cache_terminal,
            "failed sync latches existing guard and cache"
        );
        assert!(refused, "failed sync refuses buffer reuse");
        assert_eq!(idle_slots, 0, "failed sync returns nothing to idle");
        assert_eq!(
            idle_quarantine, 16,
            "failed sync quarantines existing idle backing"
        );
        assert_eq!(quarantined, 48, "later borrowed drop joins quarantine");
        assert_eq!(
            native::allocation_calls(),
            physical,
            "no physical allocation/free after fault"
        );
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "no upload entry after failed sync"
        );
        assert_eq!(super::super::LIVE_BYTES.load(Ordering::Acquire), charged);
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            1,
            "terminal backing keeps its slot"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "irreversible native CUDA fault; run alone with --exact --test-threads=1"]
    fn cache_local_sync_hook_does_not_arm_the_other_healthy_lease() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let a = CudaContext::with_workspace_cache(caps, true).unwrap();
        let b = CudaContext::with_workspace_cache(caps, true).unwrap();
        let cache = Rc::clone(a.alloc_cache.as_ref().unwrap());
        let other_cache = Rc::clone(b.alloc_cache.as_ref().unwrap());
        let observer = Rc::clone(&cache.test_native_calls);
        let other_observer = Rc::clone(&other_cache.test_native_calls);
        drop(a.allocate(16).unwrap());
        let borrowed = a.allocate(32).unwrap();
        let other_buffer = b.upload(&[0x5A; 16]).unwrap();
        a.test_cache_fail_next_sync();
        let other_drained = b.drain_alloc_cache();
        let mut other_bytes = [0; 16];
        let other_read = b.download(&other_buffer, &mut other_bytes);
        let other_usable = b.ensure_usable().is_ok();
        drop(other_buffer);
        drop(b.allocate(48).unwrap());
        let two_leases = SLOTS.load(Ordering::Acquire);
        let armed_after_other = cache.test_fault_target.get() == Some(super::TestFaultStage::Sync);
        let charged = a.live_bytes();
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let calls_before = observer.get();
        let other_calls = other_observer.get();
        let drained = a.drain_alloc_cache();
        let calls_after = observer.get();
        let other_refused = b.ensure_usable().is_err() && b.drain_alloc_cache().is_err();
        drop(b);
        let other_quarantined = other_cache.quarantined_bytes();
        let other_terminal = other_cache.is_terminal();
        drop(other_cache);
        let armed_after_drain = cache.test_fault_target.get() == Some(super::TestFaultStage::Sync);
        let sync_observed = a.test_cache_sync_fault_observed();
        let latched = TERMINAL.load(Ordering::Acquire);
        let cache_terminal = cache.is_terminal();
        let idle_slots = cache.idle_slots();
        let idle_quarantine = cache.quarantined_bytes();
        let refused = matches!(a.upload(&[0xA5; 16]), Err(EngError::Backend { .. }));
        drop(borrowed);
        drop(a);
        let quarantined = cache.quarantined_bytes();
        drop(cache);
        assert!(other_drained.is_ok() && other_read.is_ok() && other_usable);
        assert_eq!(
            other_bytes, [0x5A; 16],
            "arming A leaves B healthy before A faults"
        );
        assert_eq!(two_leases, 2, "populated B stays alive across A's fault");
        assert!(
            matches!(&drained, Err(EngError::Backend { operation: "cuda synchronize", message, .. }) if message.contains("status 700")),
            "cache-local injected sync must return Backend"
        );
        assert!(
            other_refused && other_terminal,
            "A's native fault refuses the second populated lease"
        );
        assert_eq!(
            other_quarantined, 64,
            "global terminal quarantines B's idle backing"
        );
        assert_eq!(
            other_observer.get(),
            other_calls,
            "global terminal B cannot enter sync/copy/free during refusal or teardown"
        );
        assert!(
            sync_observed,
            "only the selected cache-own sync may fire and fail"
        );
        assert_eq!(calls_after.sync_attempts, calls_before.sync_attempts + 1);
        assert_eq!(calls_after.sync_status, Some(700));
        assert_eq!(calls_after.free_attempts, calls_before.free_attempts);
        assert_eq!(calls_after.copy_attempts, calls_before.copy_attempts);
        assert_eq!(
            observer.get(),
            calls_after,
            "failed lease never retries native boundaries"
        );
        assert!(
            latched && cache_terminal,
            "real sync failure latches the unchanged guard"
        );
        assert_eq!(
            (armed_after_other, armed_after_drain),
            (true, false),
            "cache-local hook consumes once"
        );
        assert!(refused, "no reuse after the process-wide fault");
        assert_eq!(idle_slots, 0);
        assert_eq!(idle_quarantine, 16);
        assert_eq!(quarantined, 48);
        assert_eq!(
            native::allocation_calls(),
            physical,
            "no physical free/reuse after failed sync"
        );
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "failed lease cannot enter upload"
        );
        assert_eq!(super::super::LIVE_BYTES.load(Ordering::Acquire), charged);
        assert_eq!(SLOTS.load(Ordering::Acquire), 2);
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn cache_enabled_quarantine_accounts_for_real_backing() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx = CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true)
            .expect("opted-in cache context");
        let cache = Rc::clone(ctx.alloc_cache.as_ref().expect("constructor opt-in"));
        let buffer = ctx.upload(&[0xA5; 16]).expect("real backing");
        let charged = ctx.live_bytes();
        let uploads = native::upload_entry_calls();
        #[cfg(feature = "cuda-fault-qualification")]
        let observer = Rc::clone(&cache.test_native_calls);
        #[cfg(feature = "cuda-fault-qualification")]
        let calls = observer.get();
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        TERMINAL.store(true, Ordering::Release);
        drop(buffer);
        let refused = matches!(ctx.allocate(16), Err(EngError::Backend { .. }));
        drop(ctx);
        let quarantined = cache.quarantined_bytes();
        drop(cache);
        assert!(refused, "terminal cache must refuse reuse");
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            observer.get(),
            calls,
            "terminal quarantine/refusal/teardown permits no sync/copy/free attempts"
        );
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            native::allocation_calls(),
            physical,
            "terminal teardown must not allocate or free"
        );
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "terminal teardown must not upload"
        );
        assert_eq!(
            super::super::LIVE_BYTES.load(Ordering::Acquire),
            charged,
            "quarantined backing stays charged"
        );
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            1,
            "terminal quarantine retains its lease slot"
        );
        assert_eq!(
            quarantined, 16,
            "cache quarantine must account for real backing"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn cache_upload_miss_uses_owned_acquisition_not_native_upload() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx = CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true)
            .expect("cache-enabled lease");
        let entries = native::upload_entry_calls();
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let buffer = ctx.upload(&[0xA5; 16]).expect("initialized miss");
        let mut got = [0; 16];
        ctx.download(&buffer, &mut got).unwrap();
        let after_entries = native::upload_entry_calls();
        #[cfg(feature = "cuda-fault-qualification")]
        let after_physical = native::allocation_calls();
        drop(buffer);
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "healthy teardown");
        assert_eq!(got, [0xA5; 16], "miss initializes every byte");
        assert_eq!(
            after_entries, entries,
            "cache ON upload miss must use owned acquisition and upload_at"
        );
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            after_physical,
            (physical.0 + 1, physical.1),
            "one fresh backing, no early free"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn different_size_upload_cannot_checkout_a_smaller_idle_block() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx = CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true)
            .expect("cache context");
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        drop(ctx.upload(&[0xA5; 16]).unwrap());
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let larger = ctx
            .upload(&[0x5A; 32])
            .expect("different-size fresh upload");
        let mut got = [0; 32];
        ctx.download(&larger, &mut got).unwrap();
        #[cfg(feature = "cuda-fault-qualification")]
        let acquired = native::allocation_calls();
        let while_borrowed = (cache.idle_bytes(), cache.hits(), cache.misses());
        drop(larger);
        let returned = (cache.idle_bytes(), cache.idle_slots());
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
        assert_eq!(got, [0x5A; 32]);
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            acquired,
            (physical.0 + 1, physical.1),
            "size mismatch allocates fresh backing"
        );
        assert_eq!(
            while_borrowed,
            (16, 0, 2),
            "different-size miss must leave the smaller idle block untouched"
        );
        assert_eq!(
            returned,
            (48, 2),
            "exact sizes occupy separate idle entries"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn smaller_upload_cannot_checkout_a_larger_idle_block() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let baseline = ctx.live_bytes();
        drop(ctx.upload(&[0xA5; 32]).unwrap());
        let physical = native::allocation_calls();
        let smaller = ctx.upload(&[0x5A; 16]).unwrap();
        let requested = smaller.bytes();
        let mut got = [0; 16];
        ctx.download(&smaller, &mut got).unwrap();
        let acquired = native::allocation_calls();
        let borrowed = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        drop(smaller);
        let returned = (cache.idle_bytes(), cache.idle_slots());
        ctx.drain_alloc_cache().unwrap();
        let drained = (cache.idle_bytes(), cache.idle_slots(), ctx.live_bytes());
        let freed = native::allocation_calls();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
        assert_eq!((requested, got), (16, [0x5A; 16]));
        assert_eq!(
            acquired,
            (physical.0 + 1, physical.1),
            "reverse exact-size exclusion allocates fresh backing"
        );
        assert_eq!(
            borrowed,
            (32, 1, 0, 2),
            "smaller miss must leave larger idle backing untouched"
        );
        assert_eq!(returned, (48, 2), "different exact sizes remain separate");
        assert_eq!(
            drained,
            (0, 0, baseline),
            "reverse exact-size drain closes accounting"
        );
        assert_eq!(
            freed,
            (acquired.0, acquired.1 + 2),
            "drain frees both size classes once"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn healthy_drain_frees_all_idle_backing_once_and_is_idempotent() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let baseline = ctx.live_bytes();
        let small = ctx.allocate(16).unwrap();
        let large = ctx.allocate(32).unwrap();
        drop((small, large));
        let retained = (cache.idle_bytes(), cache.idle_slots(), ctx.live_bytes());
        let physical = native::allocation_calls();
        ctx.drain_alloc_cache().unwrap();
        let first = (cache.idle_bytes(), cache.idle_slots(), ctx.live_bytes());
        let after_first = native::allocation_calls();
        ctx.drain_alloc_cache().unwrap();
        let second = (cache.idle_bytes(), cache.idle_slots(), ctx.live_bytes());
        let after_second = native::allocation_calls();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
        assert_eq!(
            retained,
            (48, 2, baseline + 48),
            "healthy drain starts with two retained blocks"
        );
        assert_eq!(
            first,
            (0, 0, baseline),
            "healthy drain empties metadata and live charges"
        );
        assert_eq!(
            after_first,
            (physical.0, physical.1 + 2),
            "healthy drain frees each backing once"
        );
        assert_eq!(
            second, first,
            "repeat drain cannot subtract live bytes twice"
        );
        assert_eq!(
            after_second, after_first,
            "repeat drain cannot free backing twice"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn populated_cache_destructor_frees_backing_without_explicit_drain() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let baseline = super::super::LIVE_BYTES.load(Ordering::Acquire);
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let weak = Rc::downgrade(&cache);
        let small = ctx.allocate(16).unwrap();
        let large = ctx.allocate(32).unwrap();
        drop((small, large));
        let retained = (cache.idle_bytes(), cache.idle_slots());
        drop(ctx);
        let live_before = super::super::LIVE_BYTES.load(Ordering::Acquire);
        let physical = native::allocation_calls();
        let slots_before = SLOTS.load(Ordering::Acquire);
        drop(cache);
        assert_eq!(retained, (48, 2), "destructor starts with populated cache");
        assert_eq!(
            live_before,
            baseline + 48,
            "cache alone owns retained backing after context drop"
        );
        assert_eq!(slots_before, 1, "cache retains its adapter lease");
        assert!(
            weak.upgrade().is_none(),
            "healthy cache destruction has no ownership cycle"
        );
        assert_eq!(
            native::allocation_calls(),
            (physical.0, physical.1 + 2),
            "destructor frees each retained backing once"
        );
        assert_eq!(super::super::LIVE_BYTES.load(Ordering::Acquire), baseline);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn eligible_buffer_keeps_cache_alive_after_context_drop() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let baseline = super::super::LIVE_BYTES.load(Ordering::Acquire);
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let weak = Rc::downgrade(ctx.alloc_cache.as_ref().unwrap());
        let buffer = ctx.allocate(16).unwrap();
        drop(ctx);
        let alive = weak.strong_count() > 0;
        let slots = SLOTS.load(Ordering::Acquire);
        let live = super::super::LIVE_BYTES.load(Ordering::Acquire);
        let physical = native::allocation_calls();
        drop(buffer);
        assert!(
            alive,
            "eligible buffer must keep its cache alive after context drop"
        );
        assert_eq!((slots, live), (1, baseline + 16));
        assert!(
            weak.upgrade().is_none(),
            "final buffer release must destroy the cache without a cycle"
        );
        assert_eq!(native::allocation_calls(), (physical.0, physical.1 + 1));
        assert_eq!(super::super::LIVE_BYTES.load(Ordering::Acquire), baseline);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn healthy_cached_capture_drop_bypasses_idle_and_frees_after_close() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let observer = Rc::clone(&cache.test_native_calls);
        let stream = ctx.stream().unwrap();
        drop(ctx.allocate(32).unwrap());
        let buffer = ctx.allocate(16).unwrap();
        let idle = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        let live = ctx.live_bytes();
        let physical = native::allocation_calls();
        let calls = observer.get();
        let inside = Cell::new(None);
        let outcome = ctx.capture(
            &stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || -> Result<(), EngError> {
                drop(buffer);
                inside.set(Some((
                    (
                        cache.idle_bytes(),
                        cache.idle_slots(),
                        cache.hits(),
                        cache.misses(),
                    ),
                    ctx.live_bytes(),
                    native::allocation_calls(),
                    observer.get(),
                )));
                Err(EngError::BadRequest("healthy cached deferred drop".into()))
            },
        );
        let closed = matches!(&outcome, Err(EngError::BadRequest(message)) if message == "healthy cached deferred drop");
        drop(outcome);
        let after = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        let after_live = ctx.live_bytes();
        let freed = native::allocation_calls();
        let after_calls = observer.get();
        let healthy = ctx.ensure_usable().is_ok();
        ctx.drain_alloc_cache().unwrap();
        drop(stream);
        drop(ctx);
        drop(cache);
        assert!(
            closed && healthy,
            "cached deferred drop closes through a nonterminal body error"
        );
        assert_eq!(
            idle.0, 32,
            "cached capture starts with independent idle backing"
        );
        assert_eq!(
            inside.get(),
            Some((idle, live, physical, calls)),
            "capture drop makes no native attempt or cache mutation inside capture"
        );
        assert_eq!(
            after, idle,
            "healthy deferred drop bypasses idle cache insertion"
        );
        assert_eq!(
            after_live,
            live - 16,
            "healthy deferred drop subtracts its backing after close"
        );
        assert_eq!(
            freed,
            (physical.0, physical.1 + 1),
            "healthy capture closure frees deferred backing once"
        );
        assert_eq!(after_calls.free_attempts, calls.free_attempts + 1);
        assert_eq!(after_calls.free_status, Some(0));
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn healthy_cached_deferred_drop_retains_cache_and_lease_until_close() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let ctx = CudaContext::with_workspace(caps).unwrap();
        let baseline = ctx.live_bytes();
        let stream = ctx.stream().unwrap();
        let second = CudaContext::with_workspace_cache(caps, true).unwrap();
        let weak = Rc::downgrade(second.alloc_cache.as_ref().unwrap());
        let observer = Rc::clone(&second.alloc_cache.as_ref().unwrap().test_native_calls);
        let buffer = second.allocate(16).unwrap();
        let charged = ctx.live_bytes();
        let physical = native::allocation_calls();
        let calls = observer.get();
        let inside = Cell::new(None);
        let result = ctx.capture(
            &stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || -> Result<(), EngError> {
                drop(buffer);
                drop(second);
                inside.set(Some((
                    SLOTS.load(Ordering::Acquire),
                    weak.strong_count() > 0,
                    ctx.live_bytes(),
                    native::allocation_calls(),
                    observer.get(),
                )));
                Err(EngError::BadRequest("healthy cached owner deferral".into()))
            },
        );
        let closed = matches!(&result, Err(EngError::BadRequest(message)) if message == "healthy cached owner deferral");
        drop(result);
        let gone = weak.upgrade().is_none();
        let remaining = (SLOTS.load(Ordering::Acquire), ctx.live_bytes());
        let after_calls = observer.get();
        let healthy = ctx.ensure_usable().is_ok();
        drop(stream);
        drop(ctx);
        assert!(closed && healthy);
        assert_eq!(
            inside.get(),
            Some((2, true, charged, physical, calls)),
            "healthy cached deferred backing must retain its cache and lease until close without native attempts"
        );
        assert!(
            gone,
            "deferred cleanup must release its final cache owner without a cycle"
        );
        assert_eq!(
            remaining,
            (1, baseline),
            "healthy cached deferral closes both backing and lease charges"
        );
        assert_eq!(after_calls.free_attempts, calls.free_attempts + 1);
        assert_eq!(after_calls.free_status, Some(0));
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn warmed_capture_hits_refuse_before_any_cache_mutation() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx = CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true)
            .expect("cache context");
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let stream = ctx.stream().unwrap();
        drop(ctx.upload(&[0xA5; 16]).unwrap());
        let before = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let refused = Cell::new((false, false));
        let outcome = ctx.capture(
            &stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || -> Result<(), EngError> {
                refused.set((
                    matches!(ctx.allocate(16), Err(EngError::BadRequest(_))),
                    matches!(ctx.upload(&[0x5A; 16]), Err(EngError::BadRequest(_))),
                ));
                Err(EngError::BadRequest("warm capture probe".into()))
            },
        );
        let body_error = matches!(&outcome, Err(EngError::BadRequest(message)) if message == "warm capture probe");
        drop(outcome);
        let after = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        let healthy = ctx.ensure_usable().is_ok();
        let upload_after = native::upload_entry_calls();
        #[cfg(feature = "cuda-fault-qualification")]
        let physical_after = native::allocation_calls();
        ctx.drain_alloc_cache().unwrap();
        drop(stream);
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
        assert!(
            body_error && healthy,
            "intentional capture refusal stays healthy"
        );
        assert_eq!(
            refused.get(),
            (true, true),
            "even cache hits cannot allocate or upload in capture"
        );
        assert_eq!(
            after, before,
            "capture refusal must precede cache counters and checkout"
        );
        assert_eq!(
            upload_after, uploads,
            "refusal makes no legacy upload entry"
        );
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            physical_after, physical,
            "refusal makes no allocation/free call"
        );
        assert_eq!(
            (before.0, before.1),
            (16, 1),
            "the capture probe must start with real warmed idle backing"
        );
    }

    #[test]
    #[ignore = "irreversible capture unwind; run alone with --exact --test-threads=1"]
    fn captured_cache_backing_keeps_its_lease_when_unwind_abandons_frees() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let ctx = CudaContext::with_workspace_cache(caps, true).unwrap();
        let stream = ctx.stream().unwrap();
        let second = CudaContext::with_workspace_cache(caps, true).unwrap();
        let buffer = second.allocate(16).unwrap();
        let charged = ctx.live_bytes();
        let inside = Cell::new(0);
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.capture(
                &stream,
                CapturePlan {
                    pins: &[],
                    holds: Vec::new(),
                },
                || -> Result<(), EngError> {
                    drop(buffer);
                    drop(second);
                    inside.set(SLOTS.load(Ordering::Acquire));
                    panic!("intentional cache capture unwind");
                },
            )
        }))
        .is_err();
        let terminal = TERMINAL.load(Ordering::Acquire);
        let refused = matches!(ctx.upload(&[0; 16]), Err(EngError::Backend { .. }));
        drop(stream);
        drop(ctx);
        assert!(
            unwound && terminal && refused,
            "capture unwind latches the unchanged guard"
        );
        assert_eq!(
            inside.get(),
            2,
            "captured cache-backed drop must retain the second lease before unwind"
        );
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            2,
            "abandoned backing keeps both slots charged"
        );
        assert_eq!(super::super::LIVE_BYTES.load(Ordering::Acquire), charged);
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "unwind permits no upload entry"
        );
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            native::allocation_calls(),
            physical,
            "abandonment permits no native free"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    fn assert_owned_copy_fault(prime_hit: bool) {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        if prime_hit {
            drop(ctx.upload(&[0xA5; 16]).unwrap());
        }
        let before_bytes = ctx.live_bytes();
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let observer = Rc::clone(&cache.test_native_calls);
        let calls_before = observer.get();
        cache.arm_test_fault(TestFaultStage::BeforeOwnedCopy);
        let result = ctx.upload(&[0x5A; 16]);
        let backend = matches!(&result, Err(EngError::Backend { operation: "cuda copy", message, .. }) if message.contains("status 700"));
        let acquired = native::allocation_calls();
        let calls_after = observer.get();
        let fired = cache.test_fault_fired.get();
        let consumed = cache.test_fault_target.get().is_none();
        let terminal = TERMINAL.load(Ordering::Acquire) && cache.is_terminal();
        drop(result);
        let charged = ctx.live_bytes();
        let hits = cache.hits();
        let quarantined = cache.quarantined_bytes();
        let idle = (cache.idle_bytes(), cache.idle_slots());
        let refused = matches!(ctx.allocate(16), Err(EngError::Backend { .. }));
        drop(ctx);
        drop(cache);
        let path = if prime_hit { "hit" } else { "miss" };
        assert!(
            backend,
            "cache upload {path} must fail at owned-copy stage with real status 700"
        );
        assert_eq!(
            fired,
            Some(TestFaultStage::BeforeOwnedCopy),
            "only the selected copy boundary fires"
        );
        assert_eq!(calls_after.copy_attempts, calls_before.copy_attempts + 1);
        assert_eq!(calls_after.copy_status, Some(700));
        assert_eq!(calls_after.free_attempts, calls_before.free_attempts);
        assert_eq!(calls_after.sync_attempts, calls_before.sync_attempts);
        assert_eq!(
            observer.get(),
            calls_after,
            "failed owned-copy path never retries native boundaries"
        );
        assert!(
            consumed && terminal && refused,
            "copy failure consumes once and refuses reuse"
        );
        assert_eq!(
            acquired,
            (physical.0 + u64::from(!prime_hit), physical.1),
            "MISS owns fresh backing before copy; HIT makes no allocation"
        );
        assert_eq!(charged, before_bytes + if prime_hit { 0 } else { 16 });
        assert_eq!(
            quarantined, 16,
            "failed copy never frees or loses owned backing"
        );
        assert_eq!(idle, (0, 0), "failed copy returns nothing to idle");
        assert_eq!(
            hits,
            usize::from(prime_hit),
            "HIT must be a real checkout, not a fresh fallback"
        );
        assert_eq!(
            native::allocation_calls(),
            acquired,
            "terminal cleanup makes no native allocation/free"
        );
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "cache copy never enters legacy upload"
        );
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            1,
            "quarantined backing keeps its lease"
        );
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "irreversible native copy fault; run alone with --exact --test-threads=1"]
    fn upload_miss_copy_fault_happens_after_owned_acquisition() {
        assert_owned_copy_fault(false);
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "irreversible native copy fault; run alone with --exact --test-threads=1"]
    fn upload_hit_copy_fault_happens_after_owned_checkout() {
        assert_owned_copy_fault(true);
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "irreversible native return fault; run alone with --exact --test-threads=1"]
    fn return_sync_failure_quarantines_instead_of_inserting_or_freeing() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let buffer = ctx.allocate(16).unwrap();
        let charged = ctx.live_bytes();
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let observer = Rc::clone(&cache.test_native_calls);
        let calls_before = observer.get();
        cache.arm_test_fault(TestFaultStage::Sync);
        drop(buffer);
        let calls_after = observer.get();
        let terminal = TERMINAL.load(Ordering::Acquire) && cache.is_terminal();
        let fired = cache.test_fault_fired.get();
        let sync_failure = cache.test_sync_failure.get();
        let idle = (cache.idle_bytes(), cache.idle_slots());
        let quarantined = cache.quarantined_bytes();
        let after_return = ctx.live_bytes();
        let refused = matches!(ctx.upload(&[0; 16]), Err(EngError::Backend { .. }));
        drop(ctx);
        drop(cache);
        assert!(
            terminal,
            "return sync failure must latch backend before reuse"
        );
        assert_eq!(
            fired,
            Some(TestFaultStage::Sync),
            "fault belongs to return sync"
        );
        assert_eq!(
            sync_failure,
            Some(700),
            "the cache's own return sync reports real status 700"
        );
        assert!(refused, "failed return refuses new uploads");
        assert_eq!(calls_after.sync_attempts, calls_before.sync_attempts + 1);
        assert_eq!(calls_after.sync_status, Some(700));
        assert_eq!(calls_after.free_attempts, calls_before.free_attempts);
        assert_eq!(calls_after.copy_attempts, calls_before.copy_attempts);
        assert_eq!(
            observer.get(),
            calls_after,
            "failed return never retries native boundaries"
        );
        assert_eq!(idle, (0, 0));
        assert_eq!(quarantined, 16);
        assert_eq!(after_return, charged, "quarantine remains in LIVE_BYTES");
        assert_eq!(
            native::allocation_calls(),
            physical,
            "failed completion never frees backing"
        );
        assert_eq!(
            native::upload_entry_calls(),
            uploads,
            "no native entry after terminal"
        );
        assert_eq!(SLOTS.load(Ordering::Acquire), 1);
    }

    #[cfg(feature = "cuda-fault-qualification")]
    #[test]
    #[ignore = "irreversible native free fault; run alone with --exact --test-threads=1"]
    fn drain_free_failure_follows_successful_drain_sync_and_stops_cleanup() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let small = ctx.allocate(16).unwrap();
        let larger = ctx.allocate(32).unwrap();
        drop(small);
        drop(larger);
        let charged = ctx.live_bytes();
        let syncs_before = cache.test_sync_successes.get();
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let observer = Rc::clone(&cache.test_native_calls);
        let calls_before = observer.get();
        cache.arm_test_fault(TestFaultStage::BeforeNativeFree);
        let drained = ctx.drain_alloc_cache();
        let calls_after = observer.get();
        let refused = ctx.allocate(16).is_err() && ctx.drain_alloc_cache().is_err();
        let backend = matches!(&drained, Err(EngError::Backend { message, .. }) if message.contains("status 700"));
        let fired = cache.test_fault_fired.get();
        let syncs_after = cache.test_sync_successes.get();
        let terminal = TERMINAL.load(Ordering::Acquire) && cache.is_terminal();
        let quarantined = cache.quarantined_bytes();
        let idle = (cache.idle_bytes(), cache.idle_slots());
        let after_drain = ctx.live_bytes();
        drop(ctx);
        drop(cache);
        assert!(
            backend,
            "native free failure after drain sync must return real Backend"
        );
        assert_eq!(
            fired,
            Some(TestFaultStage::BeforeNativeFree),
            "failure belongs to free, not prior sync"
        );
        assert!(
            syncs_after > syncs_before,
            "the drain's own device sync succeeded before the free fault"
        );
        assert!(terminal && refused, "failed free latches guard and cache");
        assert_eq!(
            calls_after.sync_attempts,
            calls_before.sync_attempts + 1,
            "exactly one successful drain sync before selected native free"
        );
        assert_eq!(calls_after.sync_status, Some(0));
        assert_eq!(calls_after.copy_attempts, calls_before.copy_attempts);
        assert_eq!(
            calls_after.free_attempts,
            calls_before.free_attempts + 1,
            "selected synchronized native free wrapper must actually be entered once"
        );
        assert_eq!(
            calls_after.free_status,
            Some(700),
            "actual synchronized-free boundary reports sticky 700"
        );
        assert_eq!(
            observer.get(),
            calls_after,
            "no later sync/copy/free attempt after selected free failure, including teardown"
        );
        assert_eq!(idle, (0, 0));
        assert_eq!(
            quarantined, 48,
            "failed and unattempted frees remain quarantined"
        );
        assert_eq!(
            after_drain, charged,
            "failed free cannot subtract live bytes"
        );
        assert_eq!(
            native::allocation_calls(),
            physical,
            "no successful free or retry after the error"
        );
        assert_eq!(native::upload_entry_calls(), uploads);
        assert_eq!(SLOTS.load(Ordering::Acquire), 1);
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn upload_reinitializes_a_poisoned_cached_status_word() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        drop(ctx.upload(&f32::NAN.to_bits().to_le_bytes()).unwrap());
        let hits_before = cache.hits();
        let buffer = ctx.upload(&0u32.to_le_bytes()).unwrap();
        let mut got = [0; 4];
        ctx.download(&buffer, &mut got).unwrap();
        let hits = cache.hits() - hits_before;
        drop(buffer);
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
        assert_eq!(
            u32::from_le_bytes(got),
            0,
            "a cached flag must not retain a nonzero poison word"
        );
        assert_eq!(
            hits, 1,
            "the zero status flag must use a real cached checkout"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn upload_fully_initializes_nan_poisoned_returned_backing() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let poison = f32::NAN.to_bits().to_le_bytes().repeat(512);
        drop(ctx.upload(&poison).unwrap());
        let payload: Vec<u8> = (0..512)
            .flat_map(|i| (i as f32 / 4.0).to_bits().to_le_bytes())
            .collect();
        let hits_before = cache.hits();
        let buffer = ctx.upload(&payload).unwrap();
        let mut got = vec![0; payload.len()];
        ctx.download(&buffer, &mut got).unwrap();
        let hits = cache.hits() - hits_before;
        drop(buffer);
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
        assert_eq!(
            got, payload,
            "full payload overwrites every returned NaN word"
        );
        assert_eq!(
            hits, 1,
            "NaN identity probe must use a real cached checkout"
        );
    }

    #[test]
    #[ignore = "irreversible terminal capture; run alone with --exact --test-threads=1"]
    fn terminal_capture_abandonment_quarantines_deferred_cache_backing() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        let stream = ctx.stream().unwrap();
        let buffer = ctx.allocate(16).unwrap();
        let charged = ctx.live_bytes();
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        let uploads = native::upload_entry_calls();
        let outcome = ctx.capture(
            &stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || -> Result<(), EngError> {
                drop(buffer);
                // Terminal is a state-machine input; separate fault tests prove real status700.
                TERMINAL.store(true, Ordering::Release);
                Err(EngError::BadRequest("abandon deferred cache free".into()))
            },
        );
        let closed_with_error = outcome.is_err();
        drop(outcome);
        let quarantined = cache.quarantined_bytes();
        let idle = (cache.idle_bytes(), cache.idle_slots());
        let charged_after = ctx.live_bytes();
        let refused = matches!(ctx.allocate(16), Err(EngError::Backend { .. }));
        drop(stream);
        drop(ctx);
        drop(cache);
        assert!(
            closed_with_error && refused,
            "terminal capture closes without native cleanup"
        );
        assert_eq!(
            quarantined, 16,
            "abandoned deferred cache backing must be accounted as quarantine"
        );
        assert_eq!(idle, (0, 0), "deferred capture drops bypass idle cache");
        assert_eq!(charged_after, charged);
        assert_eq!(SLOTS.load(Ordering::Acquire), 1);
        assert_eq!(native::upload_entry_calls(), uploads);
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            native::allocation_calls(),
            physical,
            "abandonment cannot invoke a native free"
        );
    }

    #[test]
    #[ignore = "native OOM control; request total-device-bytes+1, run alone under coordinator lease"]
    fn control_oversize_oom_bypasses_cache_and_preserves_idle() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let ctx =
            CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true).unwrap();
        let cache = Rc::clone(ctx.alloc_cache.as_ref().unwrap());
        drop(ctx.upload(&[0xA5; 16]).unwrap());
        let before = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        let requests = ctx.successful_allocations();
        let impossible = ctx.memory_info().unwrap().total.checked_add(1).unwrap();
        let result = ctx.allocate(impossible);
        let oom = matches!(&result, Err(EngError::MemoryPlan(_)));
        drop(result);
        let after = (
            cache.idle_bytes(),
            cache.idle_slots(),
            cache.hits(),
            cache.misses(),
        );
        let failed_counted = ctx.successful_allocations() != requests;
        let healthy = ctx.ensure_usable().is_ok() && !TERMINAL.load(Ordering::Acquire);
        let buffer = ctx
            .upload(&[0x5A; 16])
            .expect("healthy upload after real allocation OOM");
        let mut got = [0; 16];
        ctx.download(&buffer, &mut got).unwrap();
        drop(buffer);
        ctx.drain_alloc_cache().unwrap();
        drop(ctx);
        drop(cache);
        assert!(
            oom && healthy,
            "real allocation OOM stays nonterminal with cache enabled"
        );
        assert_eq!(
            after, before,
            "oversize OOM neither checks out nor mutates idle metadata"
        );
        assert!(
            !failed_counted,
            "failed OOM is not a successful logical allocation"
        );
        assert_eq!(got, [0x5A; 16]);
        assert_eq!(
            SLOTS.load(Ordering::Acquire),
            0,
            "healthy OOM recovery releases its lease"
        );
    }

    #[test]
    #[ignore = "requires native CUDA and an exclusive test process (--test-threads=1)"]
    fn control_non_flint_upload_keeps_single_native_entry() {
        assert_eq!(SLOTS.load(Ordering::Acquire), 0, "exclusive process");
        let opted_in = CudaContext::with_workspace_cache(Activations { x: 4, xf: 4, y: 4 }, true)
            .expect("separate opted-in lease");
        let (admission, _) =
            CudaContext::fixed_projection_plan([4, 128]).expect("actual admission");
        let ctx = CudaContext::with_admission(&admission).expect("default-OFF constructor");
        assert!(
            ctx.alloc_cache.is_none(),
            "non-Flint constructor must stay OFF"
        );
        let bytes = ctx.live_bytes();
        let entries = native::upload_entry_calls();
        #[cfg(feature = "cuda-fault-qualification")]
        let physical = native::allocation_calls();
        for value in [0xA5, 0x5A] {
            let buffer = ctx.upload(&[value; 16]).expect("single-call upload");
            let mut got = [0; 16];
            ctx.download(&buffer, &mut got).expect("initialized bytes");
            drop(buffer);
            assert_eq!(got, [value; 16]);
        }
        assert_eq!(
            native::upload_entry_calls(),
            entries + 2,
            "OFF upload uses one native::upload entry each time"
        );
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(
            native::allocation_calls(),
            (physical.0 + 2, physical.1 + 2),
            "OFF uploads keep physical malloc/free counts"
        );

        assert_eq!(ctx.live_bytes(), bytes, "OFF retains no idle backing");
        let cache = opted_in.alloc_cache.as_ref().unwrap();
        assert_eq!(
            (
                cache.hits(),
                cache.misses(),
                cache.idle_bytes(),
                cache.idle_slots()
            ),
            (0, 0, 0, 0),
            "default-OFF uploads cannot touch the opted-in lease"
        );
        drop(ctx);
        drop(opted_in);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }
}
