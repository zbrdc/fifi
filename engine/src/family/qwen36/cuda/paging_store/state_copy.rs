//! State checkpoint bytes pass through an owned pinned slot, never a shared host vector.
use super::{PageSnapshot, PagedKv, backing};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceRange, DeviceStream},
};

pub(in super::super) struct HostCharge(backing::Charge);

impl HostCharge {
    pub(in super::super) fn grow(&mut self, bytes: usize) -> Result<(), EngError> {
        self.0.grow(bytes)
    }
}

impl PageSnapshot {
    pub(in super::super) fn charge_host(&self, bytes: usize) -> Result<HostCharge, EngError> {
        self.pages.charge(bytes).map(HostCharge)
    }

    pub(in super::super) fn tokens(&self) -> usize {
        self.visible_positions
    }
}

impl PagedKv {
    fn state_copy_ready(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<(), EngError> {
        context.ensure_uncaptured("paged state copy")?;
        context.check_launch_stream(stream)?;
        self.ensure_stream(stream)?;
        if self.slots.iter().any(|slot| slot.page.is_some()) {
            return Err(EngError::BadRequest(
                "state copy needs drained KV slots".to_owned(),
            ));
        }
        Ok(())
    }

    pub(in super::super) fn read_state(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        device: DeviceRange<'_>,
        host: &mut [u8],
    ) -> Result<(), EngError> {
        self.state_copy_ready(context, stream)?;
        if device.bytes() != host.len() {
            return Err(EngError::BadRequest("state read lengths differ".to_owned()));
        }
        let slot = &mut self.slots[0];
        let chunk = slot.pinned.len();
        for (i, bytes) in host.chunks_mut(chunk).enumerate() {
            let range = device
                .buffer()
                .range(device.offset() + i * chunk, bytes.len())?;
            context.copy_to_pinned(stream, range, &mut slot.pinned, 0)?;
            stream.record(&slot.event)?;
            slot.pinned.wait(&slot.event)?;
            slot.pinned.read(0, bytes)?;
        }
        Ok(())
    }

    pub(in super::super) fn write_state(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        host: &[u8],
        device: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.state_copy_ready(context, stream)?;
        if device.bytes() != host.len() {
            return Err(EngError::BadRequest(
                "state write lengths differ".to_owned(),
            ));
        }
        let slot = &mut self.slots[0];
        let chunk = slot.pinned.len();
        for (i, bytes) in host.chunks(chunk).enumerate() {
            let range = device
                .buffer()
                .range(device.offset() + i * chunk, bytes.len())?;
            slot.pinned.write(0, bytes)?;
            context.copy_from_pinned(stream, &mut slot.pinned, 0, range)?;
            stream.record(&slot.event)?;
            slot.pinned.wait(&slot.event)?;
        }
        Ok(())
    }
}
