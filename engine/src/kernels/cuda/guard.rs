//! Process-wide serialization and adapter leases of the CUDA backend.
//!
//! Decision 0040 item 4 applies the rules of decisions 0023 and 0024 to CUDA:
//! one dispatch guard serializes every native call across threads, and the
//! process admits at most two adapter leases at once. Device buffers share
//! their adapter's lease, so a slot stays held until the last buffer is freed.
//!
//! The failure flag is passed in, not global here, so the tests can use their
//! own flag and counter while other tests run native calls in parallel.

use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::EngError;

/// The most adapter leases that the process admits at once (decision 0024).
pub(super) const MAX_LEASES: usize = 2;

static DISPATCH: Mutex<()> = Mutex::new(());

thread_local! {
    static DISPATCH_HELD: Cell<bool> = const { Cell::new(false) };
}

/// Serializes native work across threads (decision 0023).
///
/// The guard nests on one thread. An inner guard holds no lock and still runs
/// its cleanup. Mutex poison and unwinding latch `failed` before any further
/// native call, because the interrupted call's device state is unknown.
pub(super) struct DispatchGuard<'a> {
    lock: Option<MutexGuard<'static, ()>>,
    failed: &'a AtomicBool,
}

impl<'a> DispatchGuard<'a> {
    pub(super) fn enter(failed: &'a AtomicBool) -> Self {
        let lock = if DISPATCH_HELD.with(Cell::get) {
            None
        } else {
            let lock = DISPATCH.lock().unwrap_or_else(|poisoned| {
                failed.store(true, Ordering::Release);
                poisoned.into_inner()
            });
            DISPATCH_HELD.with(|held| held.set(true));
            Some(lock)
        };
        if std::thread::panicking() {
            failed.store(true, Ordering::Release);
        }
        Self { lock, failed }
    }

    #[cfg(test)]
    fn holds_lock(&self) -> bool {
        self.lock.is_some()
    }
}

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        if self.lock.is_some() {
            if std::thread::panicking() {
                self.failed.store(true, Ordering::Release);
            }
            DISPATCH_HELD.with(|held| held.set(false));
        }
    }
}

/// One admitted adapter slot, shared by the adapter and its device buffers.
pub(super) struct Lease {
    slots: &'static AtomicUsize,
    failed: &'static AtomicBool,
}

impl Lease {
    /// Admits one more adapter against `slots`, or reports why it cannot.
    ///
    /// A terminal backend admits nothing, even when a slot is free.
    pub(super) fn acquire(
        slots: &'static AtomicUsize,
        failed: &'static AtomicBool,
    ) -> Result<Rc<Self>, EngError> {
        let _dispatch = DispatchGuard::enter(failed);
        super::ensure_usable(failed)?;
        slots
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, |count| {
                count.checked_add(1).filter(|&next| next <= MAX_LEASES)
            })
            .map_err(|_| EngError::Backend {
                operation: super::INITIALIZE_OPERATION,
                message: "this process already holds two CUDA adapter leases".into(),
            })?;
        Ok(Rc::new(Self { slots, failed }))
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter(self.failed);
        // A terminal backend may still own device memory of this lease, so the
        // slot stays reserved for the rest of the process (decision 0024).
        if !self.failed.load(Ordering::Acquire) {
            self.slots.fetch_sub(1, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_guards_nest_on_one_thread() {
        static FAILED: AtomicBool = AtomicBool::new(false);
        let outer = DispatchGuard::enter(&FAILED);
        let inner = DispatchGuard::enter(&FAILED);
        assert!(outer.holds_lock());
        assert!(!inner.holds_lock());
        drop(inner);
        assert!(DISPATCH_HELD.with(Cell::get));
        drop(outer);
        assert!(!DISPATCH_HELD.with(Cell::get));
        assert!(!FAILED.load(Ordering::Acquire));
    }

    #[test]
    fn a_lease_acquired_inside_a_dispatch_guard_does_not_deadlock() {
        static SLOTS: AtomicUsize = AtomicUsize::new(0);
        static FAILED: AtomicBool = AtomicBool::new(false);
        let _outer = DispatchGuard::enter(&FAILED);
        let lease = Lease::acquire(&SLOTS, &FAILED).expect("first lease");
        drop(lease);
        assert_eq!(SLOTS.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_third_lease_is_rejected_until_one_is_released() {
        static SLOTS: AtomicUsize = AtomicUsize::new(0);
        static FAILED: AtomicBool = AtomicBool::new(false);
        let first = Lease::acquire(&SLOTS, &FAILED).expect("first lease");
        let _second = Lease::acquire(&SLOTS, &FAILED).expect("second lease");
        let error = Lease::acquire(&SLOTS, &FAILED)
            .err()
            .expect("third lease must fail");
        assert_eq!(
            error,
            EngError::Backend {
                operation: "cuda initialize",
                message: "this process already holds two CUDA adapter leases".into(),
            }
        );
        let shared = Rc::clone(&first);
        drop(first);
        assert!(
            Lease::acquire(&SLOTS, &FAILED).is_err(),
            "a shared lease keeps its slot"
        );
        drop(shared);
        let _replacement = Lease::acquire(&SLOTS, &FAILED).expect("released slot");
    }

    #[test]
    fn a_terminal_backend_keeps_its_slots_and_admits_no_lease() {
        static SLOTS: AtomicUsize = AtomicUsize::new(0);
        static FAILED: AtomicBool = AtomicBool::new(false);
        let lease = Lease::acquire(&SLOTS, &FAILED).expect("first lease");
        FAILED.store(true, Ordering::Release);
        drop(lease);
        assert_eq!(SLOTS.load(Ordering::Acquire), 1);
        let error = Lease::acquire(&SLOTS, &FAILED)
            .err()
            .expect("terminal backend");
        assert!(
            matches!(
                error,
                EngError::Backend {
                    operation: "execute",
                    ..
                }
            ),
            "{error:?}"
        );
    }
}
