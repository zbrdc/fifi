//! Process-wide serialization, ownership, and failure state of the native adapter.
//!
//! Failure policy. See "Native failure policy" in the kernels design.
//!
//! - A failure with unknown completion is terminal. The adapter calls [`mark_failed`] before any
//!   temporary array can drop, so `Array::drop` retains the array instead of freeing it
//!   (decisions 0022 and 0023).
//! - A completed-host method recovers when its synchronization succeeds (decision 0011).
//! - A resident method makes every native failure terminal, including a failure before submission.
//!
//! The flag is sticky for the whole process. Later calls return `Backend`, and adapter replacement
//! fails, until the process exits.

use std::{
    cell::{Cell, RefCell},
    ffi::{CStr, c_char, c_void},
    num::NonZeroU64,
    ptr,
    rc::Rc,
    sync::{
        Mutex, MutexGuard, Once,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

use super::ffi;
use crate::{EngError, tensor::OwnerKey};

/// The most adapter leases that the process admits at once (decision 0024).
const MAX_LEASES: usize = 2;

static LIVE_OWNERS: AtomicUsize = AtomicUsize::new(0);
static RESIDENT_FAILED: AtomicBool = AtomicBool::new(false);
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
static HANDLER: Once = Once::new();
static DISPATCH: Mutex<()> = Mutex::new(());

thread_local! {
    static MESSAGE: RefCell<([u8; 2048], usize)> = const { RefCell::new(([0; 2048], 0)) };
    static DISPATCH_HELD: Cell<bool> = const { Cell::new(false) };
}

/// Makes the native owner terminal for the rest of the process.
pub(super) fn mark_failed() {
    RESIDENT_FAILED.store(true, Ordering::Release);
}

/// Whether the native owner is terminal. A terminal owner dispatches and releases nothing.
pub(super) fn has_failed() -> bool {
    RESIDENT_FAILED.load(Ordering::Acquire)
}

/// Serializes native work across threads (decision 0023).
///
/// The guard nests on one thread. An inner guard holds no lock and still runs its cleanup.
/// Mutex poison and unwinding mark the owner terminal before any further native call.
pub(super) struct DispatchGuard {
    lock: Option<MutexGuard<'static, ()>>,
}

impl DispatchGuard {
    pub(super) fn enter() -> Self {
        let lock = if DISPATCH_HELD.with(Cell::get) {
            None
        } else {
            let lock = DISPATCH.lock().unwrap_or_else(|poisoned| {
                mark_failed();
                poisoned.into_inner()
            });
            DISPATCH_HELD.with(|held| held.set(true));
            Some(lock)
        };
        if std::thread::panicking() {
            mark_failed();
        }
        Self { lock }
    }
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        if self.lock.is_some() {
            if std::thread::panicking() {
                mark_failed();
            }
            DISPATCH_HELD.with(|held| held.set(false));
        }
    }
}

unsafe extern "C" fn capture_error(message: *const c_char, _: *mut c_void) {
    if message.is_null() {
        return;
    }
    let bytes = unsafe { CStr::from_ptr(message) }.to_bytes();
    let _ = MESSAGE.try_with(|slot| {
        if let Ok(mut saved) = slot.try_borrow_mut() {
            let length = bytes.len().min(saved.0.len());
            saved.0[..length].copy_from_slice(&bytes[..length]);
            saved.1 = length;
        }
    });
}

/// Registers the process error handler once. The handler stores the last native message per thread.
pub(super) fn install_error_handler() {
    HANDLER.call_once(|| unsafe {
        ffi::mlx_set_error_handler(Some(capture_error), ptr::null_mut(), None);
    });
}

/// Forgets the saved native message, so the next failure reports only its own text.
pub(super) fn clear_message() {
    MESSAGE.with(|slot| slot.borrow_mut().1 = 0);
}

/// Builds the `Backend` error for a failed native call and consumes the saved message.
pub(super) fn failure(operation: &'static str) -> EngError {
    let message = MESSAGE.with(|slot| {
        let mut saved = slot.borrow_mut();
        let message = String::from_utf8_lossy(&saved.0[..saved.1]).into_owned();
        saved.1 = 0;
        message
    });
    EngError::Backend {
        operation,
        message: if message.is_empty() {
            "native operation failed".into()
        } else {
            message
        },
    }
}

pub(super) fn check(status: i32, operation: &'static str) -> Result<(), EngError> {
    if status == 0 {
        Ok(())
    } else {
        Err(failure(operation))
    }
}

#[cfg(feature = "native-fault-qualification")]
pub(super) fn qualification_sync_count() -> u64 {
    unsafe { ffi::mlx_qualification_sync_count() }
}

#[cfg(feature = "native-fault-qualification")]
pub(super) fn selected_operation_count() -> u64 {
    unsafe { ffi::mlx_selected_operation_count() }
}

/// One admitted adapter slot. Residents share it with their adapter, so it outlives the adapter.
pub(super) struct Lease;

impl Lease {
    /// Admits one more adapter, or reports why the process cannot.
    pub(super) fn acquire() -> Result<Rc<Self>, EngError> {
        if has_failed() {
            return Err(EngError::Backend {
                operation: "initialize",
                message: "the native resident owner has failed".into(),
            });
        }
        LIVE_OWNERS
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, |count| {
                count.checked_add(1).filter(|&next| next <= MAX_LEASES)
            })
            .map_err(|_| EngError::Backend {
                operation: "initialize",
                message: "this process already holds two native adapter leases".into(),
            })?;
        Ok(Rc::new(Self))
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        if !has_failed() {
            LIVE_OWNERS.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Issues the identity of the next adapter generation.
pub(super) fn next_owner() -> Result<OwnerKey, EngError> {
    NEXT_OWNER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map(|value| OwnerKey(NonZeroU64::new(value).unwrap()))
        .map_err(|_| EngError::Backend {
            operation: "initialize",
            message: "native owner identifiers exhausted".into(),
        })
}
