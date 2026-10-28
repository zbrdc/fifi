//! Pending two-bank transport reservations, charged through the existing placement store.

use super::{ExpertId, ExpertPayload, GpuExpertStore, VRAM_HEADROOM_BYTES};
use crate::EngError;

/// Two pinned banks, two device banks and the caller's bounded auxiliary storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamBankPlan {
    bank_bytes: u64,
    device_bytes: u64,
    host_bytes: u64,
}

impl StreamBankPlan {
    /// Auxiliary bytes cover descriptors/status/scratch not already reserved.
    /// Host scratch includes any temporary owned payload from the source.
    pub fn new(
        bank_bytes: u64,
        device_aux_bytes: u64,
        host_scratch_bytes: u64,
    ) -> Result<Self, EngError> {
        if bank_bytes == 0 {
            return Err(EngError::BadRequest(
                "streaming bank bytes must be positive".into(),
            ));
        }
        let pair = bank_bytes.checked_mul(2).ok_or_else(overflow)?;
        Ok(Self {
            bank_bytes,
            device_bytes: pair.checked_add(device_aux_bytes).ok_or_else(overflow)?,
            host_bytes: pair.checked_add(host_scratch_bytes).ok_or_else(overflow)?,
        })
    }

    pub fn bank_bytes(&self) -> u64 {
        self.bank_bytes
    }

    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    pub fn host_bytes(&self) -> u64 {
        self.host_bytes
    }

    /// Reserve before hot placement; allocate the banks after this wrapper is released.
    /// The caller's host budget must already exclude model memory and desktop headroom.
    pub fn reserve<'a, G: GpuExpertStore>(
        self,
        gpu: &'a mut G,
        host_budget: u64,
    ) -> Result<impl GpuExpertStore + 'a, EngError> {
        if self.host_bytes > host_budget {
            return Err(EngError::MemoryPlan(format!(
                "streaming banks need {} host bytes, budget is {host_budget}",
                self.host_bytes
            )));
        }
        check_free(gpu.free_bytes()?, gpu.reserved_bytes(), self.device_bytes)?;
        Ok(ReservedStore {
            gpu,
            pending: self.device_bytes,
        })
    }
}

fn overflow() -> EngError {
    EngError::MemoryPlan("streaming bank reservation byte count overflow".into())
}

fn check_free(free: u64, reserved: u64, pending: u64) -> Result<(), EngError> {
    let required = reserved
        .checked_add(pending)
        .and_then(|bytes| bytes.checked_add(VRAM_HEADROOM_BYTES))
        .ok_or_else(overflow)?;
    if free < required {
        return Err(EngError::MemoryPlan(format!(
            "device reserves need {required} bytes including headroom, only {free} free"
        )));
    }
    Ok(())
}

struct ReservedStore<'a, G: GpuExpertStore> {
    gpu: &'a mut G,
    pending: u64,
}

impl<G: GpuExpertStore> GpuExpertStore for ReservedStore<'_, G> {
    fn upload(
        &mut self,
        id: ExpertId,
        part: usize,
        payload: &ExpertPayload,
    ) -> Result<(), EngError> {
        self.gpu.upload(id, part, payload)
    }

    fn free_bytes(&self) -> Result<u64, EngError> {
        self.gpu.free_bytes()
    }

    fn reserved_bytes(&self) -> u64 {
        self.gpu.reserved_bytes().saturating_add(self.pending)
    }

    fn finish_placement(&mut self, free_bytes: u64) -> Result<(), EngError> {
        check_free(free_bytes, self.gpu.reserved_bytes(), self.pending)?;
        // The dense cache must not consume the banks' still-unallocated reservation.
        self.gpu.finish_placement(free_bytes - self.pending)
    }
}
