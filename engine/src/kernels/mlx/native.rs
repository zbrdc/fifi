//! The private native adapter over the MLX C interface.
//!
//! This module and its children hold all `unsafe` code of the MLX adapter. The public
//! `MlxKernels` in the parent module validates input, then calls the `Context` methods here.
//!
//! | File | Responsibility |
//! |---|---|
//! | `ffi.rs` | Foreign declarations and the empty array handle. |
//! | `guard.rs` | Dispatch lock, terminal-failure flag, error capture, and the adapter lease. |
//! | `handles.rs` | Owning wrappers for devices, streams, and arrays, and the resident storage. |
//! | `transfer.rs` | Import of host tensors and readback of resident tensors. |
//! | `host.rs` | Completed-host operations that return host bytes. |
//! | `resident_ops.rs` | Resident `qmm`, `mm`, elementwise operations, `rms_norm`, `rope`, and the gated delta kernel (decision 0050). |
//! | `attention.rs` | The resident dense-attention caller. |
//! | `pages.rs` | Caller-owned host pages and their in-place import (decision 0041). |
//!
//! Failure policy. See "Native failure policy" in the kernels design and `guard.rs`.

#![allow(unsafe_code)]

use std::{marker::PhantomData, rc::Rc};

mod attention;
mod ffi;
mod guard;
mod handles;
mod host;
mod pages;
mod resident_ops;
mod transfer;

use crate::{
    EngError,
    tensor::{Dtype, ExecutionTensor, OwnerKey},
};
use guard::{
    DispatchGuard, Lease, clear_message, has_failed, install_error_handler, mark_failed, next_owner,
};
use handles::{Device, RetireCore, Stream};

pub(crate) use handles::Resident;
pub use pages::HostPages;

/// The adapter state behind one `Context`.
///
/// Field order is drop order. The retire core drops first, so retired arrays free before the stream
/// and the device. Pending work drops next, and the lease drops after the stream and the device.
/// At most one pending field holds work, because each method checks readiness first.
struct State {
    retire: Rc<RetireCore>,
    pending_host: Option<host::HostWork>,
    pending_import: Option<transfer::PendingImport>,
    pending_imports: Option<Vec<transfer::PendingImport>>,
    pending_attention: Option<attention::PendingAttention>,
    stream: Stream,
    _device: Device,
    /// Held only for drop order: the adapter slot outlives the stream and the device (decision 0024).
    _lease: Rc<Lease>,
    owner: OwnerKey,
    /// Whether the resident float operations take BF16 operands (packet Q36-BF16-a). `false` until
    /// `Context::allow_bf16` sets it, so a caller that never calls it rejects BF16 exactly as before.
    allow_bf16: bool,
}

impl State {
    /// Whether `dtype` is a float operand of the resident operations: F32, and BF16 only after
    /// `Context::allow_bf16` sets the switch.
    fn accepts_float(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F32 || (dtype == Dtype::Bf16 && self.allow_bf16)
    }

    /// Wraps an output as a resident tensor that holds this adapter's retire core, which holds the lease.
    ///
    /// The output can still be lazy. `readback` evaluates it (decision 0032).
    fn resident(
        &self,
        array: handles::Array,
        dtype: Dtype,
        shape: Vec<u32>,
        byte_len: u64,
    ) -> ExecutionTensor {
        ExecutionTensor::new(Resident::new(
            array,
            dtype,
            shape,
            byte_len,
            self.owner,
            &self.retire,
        ))
    }
}

pub(super) struct Context {
    state: Option<State>,
    _thread: PhantomData<Rc<()>>,
}

impl Context {
    pub(super) fn owner_key(&self) -> OwnerKey {
        self.state.as_ref().unwrap().owner
    }

    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn qualification_sync_count(&self) -> u64 {
        guard::qualification_sync_count()
    }

    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn qualification_retire_stats(&self) -> (u64, u64) {
        handles::qualification_retire_stats()
    }

    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn qualification_reset_retire_peak(&mut self) {
        handles::qualification_reset_retire_peak();
    }

    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn qualification_flush_stats(&self) -> crate::kernels::QualificationFlushStats {
        handles::qualification_flush_stats()
    }

    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn qualification_reset_flush_stats(&mut self) {
        handles::qualification_reset_flush_stats();
    }

    /// Lets the resident float operations take BF16 operands and return BF16 results (packet
    /// Q36-BF16-a). It makes no native call. Mixed F32 and BF16 operands promote to F32 in MLX.
    pub(super) fn allow_bf16(&mut self, allow: bool) {
        self.state.as_mut().unwrap().allow_bf16 = allow;
    }

    /// Reads the process-wide MLX allocator counters. A failure is reported and recoverable,
    /// because the query starts no device work.
    pub(super) fn memory_stats(&self) -> Result<(usize, usize, usize), EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let (mut active, mut cache, mut peak) = (0usize, 0usize, 0usize);
        let status = unsafe { ffi::mlx_memory_stats(&mut active, &mut cache, &mut peak) };
        if status != 0 {
            return Err(EngError::Backend {
                operation: "memory statistics",
                message: "native memory query failed".into(),
            });
        }
        Ok((active, cache, peak))
    }

    pub(super) fn reset_peak_memory(&mut self) -> Result<(), EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        if unsafe { ffi::mlx_reset_peak_memory() } != 0 {
            return Err(EngError::Backend {
                operation: "memory statistics",
                message: "native peak reset failed".into(),
            });
        }
        Ok(())
    }

    pub(super) fn set_cache_limit(&mut self, limit: usize) -> Result<usize, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let mut previous = 0usize;
        if unsafe { ffi::mlx_set_cache_limit(&mut previous, limit) } != 0 {
            return Err(EngError::Backend {
                operation: "cache limit",
                message: "native cache limit update failed".into(),
            });
        }
        Ok(previous)
    }

    /// Sets the retire cap of this adapter and returns the previous cap (decision 0044 item 8).
    pub(super) fn set_retire_cap(&mut self, bytes: u64) -> Result<u64, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_ref().ok_or_else(|| EngError::Backend {
            operation: "retire cap",
            message: "adapter state is gone".into(),
        })?;
        Ok(state.retire.set_cap(bytes))
    }

    pub(super) fn try_new() -> Result<Self, EngError> {
        let _dispatch = DispatchGuard::enter();
        let lease = Lease::acquire()?;
        let retire = RetireCore::new(&lease);
        let owner = next_owner()?;
        install_error_handler();
        clear_message();
        let device = Device::try_new()?;
        let stream = Stream::for_device(&device)?;
        Ok(Self {
            state: Some(State {
                retire,
                pending_host: None,
                pending_import: None,
                pending_imports: None,
                pending_attention: None,
                stream,
                _device: device,
                _lease: lease,
                owner,
                allow_bf16: false,
            }),
            _thread: PhantomData,
        })
    }

    pub(super) fn check_ready(&self) -> Result<(), EngError> {
        if has_failed() {
            return Err(EngError::Backend {
                operation: "execute",
                message: "the native resident owner has failed".into(),
            });
        }
        let state = self.state.as_ref().unwrap();
        if state.pending_host.is_some()
            || state.pending_import.is_some()
            || state.pending_imports.is_some()
            || state.pending_attention.is_some()
        {
            return Err(EngError::Backend {
                operation: "execute",
                message: "the native adapter has incomplete work".into(),
            });
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        if let Some(state) = self.state.take() {
            if has_failed() || unsafe { ffi::mlx_synchronize(state.stream.0) } != 0 {
                mark_failed();
                // Failed completion retains arrays, staging, stream, device, and the process lease together.
                std::mem::forget(state);
                return;
            }
            state.retire.flush_after_sync();
        }
    }
}
