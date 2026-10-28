//! Arena stage 1: the pinned staging slot for dense-weight uploads.
//!
//! Scoped projections keep the private upload guard through the synchronous
//! consumer. The retained allocation grows; the valid payload remains exact.
//! WHY: the pageable path allocates, copies and synchronizes per call at a
//! measured 0.6 GB/s; a reused page-locked slot lets the DMA engine read
//! host memory directly and pays the allocation only on a grow.
//! HOW: the native entry copies the payload into the pinned buffer, issues
//! the async device copy on the ABI default stream and waits for that stream
//! only, so the pinned bytes are never read by a DMA while being written.

use std::{cell::Ref, sync::atomic::Ordering};

use super::graph;
use super::pinned_buf::PinnedBuf;
use super::{CudaContext, DeviceBuffer, MemoryInfo, device_error};
use crate::EngError;
use crate::kernels::cuda::guard::DispatchGuard;
use crate::kernels::cuda::native::{self, resident};
use crate::kernels::cuda::{TERMINAL, device_is_integrated};

/// Eligibility ceiling for pinned uploads, not a resource admission.
/// Larger payloads keep the separately admitted pageable upload path.
pub(crate) const PINNED_STAGE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One pinned host staging slot and its paired device slot.
pub(super) struct Stage {
    host: PinnedBuf,
    device: DeviceBuffer,
    capacity: usize,
}

/// Failed initialization must not run the eventual context cleanup callback.
struct PendingStageAdmission {
    max_payload_bytes: usize,
    after_cleanup: Option<Box<dyn FnOnce()>>,
}

impl PendingStageAdmission {
    fn installed(mut self) -> StageAdmission {
        StageAdmission {
            max_payload_bytes: self.max_payload_bytes,
            after_cleanup: self.after_cleanup.take(),
        }
    }
}

impl Drop for PendingStageAdmission {
    fn drop(&mut self) {
        if let Some(release) = self.after_cleanup.take() {
            std::mem::forget(release);
        }
    }
}

/// The release obligation stays after all native owners in the context.
pub(super) struct StageAdmission {
    max_payload_bytes: usize,
    after_cleanup: Option<Box<dyn FnOnce()>>,
}

impl Drop for StageAdmission {
    fn drop(&mut self) {
        let mut release = self.after_cleanup.take();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _dispatch = DispatchGuard::enter(&TERMINAL);
            if std::thread::panicking() || TERMINAL.load(Ordering::Acquire) || graph::window_open()
            {
                return;
            }
            if let Some(release) = release.take() {
                release();
            }
        }));
        if result.is_err() {
            TERMINAL.store(true, Ordering::Release);
        }
        // Uncertain or deferred cleanup cannot return reservation credit.
        if let Some(release) = release {
            std::mem::forget(release);
        }
    }
}

/// Pure slot plan: Some of the new capacity when the payload forces a grow.
fn plan(capacity: Option<usize>, bytes: usize) -> Option<usize> {
    match capacity {
        Some(c) if bytes <= c => None,
        _ => Some(bytes),
    }
}

fn check_pinned_projection(
    weight: &[u8],
    shape: [usize; 2],
    input: &[f32],
) -> Result<(), EngError> {
    let (required, _) = native::f32_projection_bytes(shape).ok_or_else(|| {
        EngError::BadRequest("cuda pinned projection shape is empty or overflows".to_owned())
    })?;
    if weight.len() != required {
        return Err(EngError::BadRequest(
            "cuda pinned projection weight byte length differs from its shape".to_owned(),
        ));
    }
    if input.len() != shape[1] || !input.iter().all(|value| value.is_finite()) {
        return Err(EngError::BadRequest(
            "cuda pinned projection requires one finite input row of the weight width".to_owned(),
        ));
    }
    if weight.is_empty() || weight.len() > PINNED_STAGE_LIMIT_BYTES {
        return Err(EngError::BadRequest(
            "cuda pinned projection payload is outside the stage limit".to_owned(),
        ));
    }
    Ok(())
}

impl CudaContext {
    /// Reserves native resources before the first workspace allocation.
    ///
    /// The caller reserves the complete host/device growth envelope through
    /// its admission authority before returning the release callback.
    pub(crate) fn with_pinned_stage(
        admission: &super::Admission,
        max_payload_bytes: usize,
        admit: impl FnOnce(MemoryInfo, bool) -> Result<Box<dyn FnOnce()>, EngError>,
    ) -> Result<Self, EngError> {
        crate::kernels::cuda::ensure_usable(&TERMINAL)?;
        graph::ensure_closed("a pinned staging admission")?;
        if max_payload_bytes == 0 || max_payload_bytes > PINNED_STAGE_LIMIT_BYTES {
            return Err(EngError::BadRequest(
                "cuda pinned staging admission is outside the stage limit".to_owned(),
            ));
        }
        let _dispatch = super::enter_dispatch()?;
        let caps = admission.caps()?;
        crate::kernels::cuda::require_device(crate::kernels::cuda::device_count()?)?;
        let lease = super::Lease::acquire(&super::SLOTS, &TERMINAL)?;
        let (free, total) = native::memory_info().map_err(|status| {
            crate::kernels::cuda::native_error(
                &TERMINAL,
                crate::kernels::cuda::memory_info_operation(status),
                status,
            )
        })?;
        let integrated = device_is_integrated()?;
        let after_cleanup = admit(MemoryInfo { free, total }, integrated)?;
        let pending = PendingStageAdmission {
            max_payload_bytes,
            after_cleanup: Some(after_cleanup),
        };
        crate::kernels::cuda::ensure_usable(&TERMINAL)?;
        graph::ensure_closed("a pinned staging allocation")?;
        let workspace = super::workspace::Workspace::allocate(&lease, caps)?;
        let mut context = Self::from_workspace(lease, workspace, admission.growth);
        context.pinned_stage_admission = Some(pending.installed());
        Ok(context)
    }

    /// Projects exactly the supplied F32 payload through reusable staging.
    pub(crate) fn linear_pinned_f32(
        &self,
        weight: &[u8],
        shape: [usize; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("a pinned projection")?;
        check_pinned_projection(weight, shape, input)?;
        let staged = self.upload_pinned(weight)?;
        self.linear_f32(&staged, shape, input)
    }

    /// Returns the winning logit from exactly the supplied F32 payload.
    pub(crate) fn linear_argmax_pinned_f32(
        &self,
        weight: &[u8],
        shape: [usize; 2],
        input: &[f32],
    ) -> Result<(u32, f32), EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("a pinned projection")?;
        check_pinned_projection(weight, shape, input)?;
        let staged = self.upload_pinned(weight)?;
        self.linear_argmax_f32(&staged, shape, input)
    }

    /// Dense-weight upload through the pinned staging slot.
    ///
    /// Empty input and payloads above PINNED_STAGE_LIMIT_BYTES are
    /// BadRequest; those callers keep the pageable upload path.
    fn upload_pinned(&self, bytes: &[u8]) -> Result<Ref<'_, DeviceBuffer>, EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("a pinned upload")?;
        if bytes.is_empty() {
            return Err(EngError::BadRequest(
                "cuda pinned upload of zero bytes".to_owned(),
            ));
        }
        if bytes.len() > PINNED_STAGE_LIMIT_BYTES {
            return Err(EngError::BadRequest(format!(
                "cuda pinned upload of {} bytes above the {} byte stage limit",
                bytes.len(),
                PINNED_STAGE_LIMIT_BYTES
            )));
        }
        {
            let mut slot = self.pinned_stage.try_borrow_mut().map_err(|_| {
                EngError::BadRequest("cuda pinned upload stage is already in use".to_owned())
            })?;
            let admission = self.pinned_stage_admission.as_ref().ok_or_else(|| {
                EngError::MemoryPlan(
                    "cuda pinned staging has no owned resource admission".to_owned(),
                )
            })?;
            if bytes.len() > admission.max_payload_bytes {
                return Err(EngError::MemoryPlan(format!(
                    "cuda pinned staging payload of {} bytes exceeds admitted maximum {}",
                    bytes.len(),
                    admission.max_payload_bytes
                )));
            }
            let _dispatch = self.enter()?;
            if plan(slot.as_ref().map(|s| s.capacity), bytes.len()).is_some() {
                let host = self.pinned_buf(bytes.len())?;
                let device = DeviceBuffer::allocate(&self.lease, bytes.len())?;
                // The swap frees the old slots only after the new pair exists,
                // so a failed grow leaves the old slot serving smaller weights.
                *slot = Some(Stage {
                    host,
                    device,
                    capacity: bytes.len(),
                });
            }
            // Releasing the old pair can latch a terminal cleanup failure.
            self.ensure_usable()?;
            let stage = slot
                .as_ref()
                .expect("the grow above installed the stage slot");
            let pinned = stage.host.raw()?;
            let device = self.owned(&stage.device)?;
            resident::upload_pinned(device, pinned, bytes).map_err(device_error)?;
        }
        Ok(Ref::map(self.pinned_stage.borrow(), |slot| {
            &slot
                .as_ref()
                .expect("the pinned stage was just staged")
                .device
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_stage_slot_is_grow_only_and_exact() {
        assert_eq!(plan(Some(4096), 4096), None);
        assert_eq!(plan(Some(4096), 1024), None);
        assert_eq!(plan(Some(1024), 4096), Some(4096));
        assert_eq!(plan(None, 512), Some(512));
    }

    #[test]
    fn pinned_stage_plan_never_rounds_up() {
        assert_eq!(plan(Some(100), 101), Some(101));
        assert_eq!(plan(None, 3), Some(3));
    }
}
