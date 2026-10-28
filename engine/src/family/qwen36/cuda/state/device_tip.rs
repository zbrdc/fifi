//! One fixed device-side Delta snapshot; its key leases the unchanged live KV prefix.
use super::super::book::Health;
use super::super::state_layout::StateLayout;
use super::{ResidentState, StateIdentity, export::IMPORT_REWRITES_WORDS};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream},
};
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct DeviceTipKey(Rc<usize>);

impl DeviceTipKey {
    pub(crate) fn positions(&self) -> usize {
        *self.0
    }

    fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

pub(crate) struct DeviceTip {
    delta: DeviceBuffer,
    layout: StateLayout,
    identity: StateIdentity,
    owner: u64,
    current: Option<DeviceTipKey>,
}

impl DeviceTip {
    pub(crate) fn new(
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &ResidentState,
    ) -> Result<Self, EngError> {
        state.book.begin_export(stream.id())?;
        let bytes = state
            .layout
            .delta
            .total
            .checked_div(state.layout.delta.slots)
            .ok_or_else(|| EngError::BadRequest("device tip has zero Delta slots".into()))?;
        Ok(Self {
            delta: ctx.allocate(bytes)?,
            layout: state.layout,
            identity: state.identity.clone(),
            owner: stream.id(),
            current: None,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.delta.bytes()
    }

    fn check(&self, stream: &DeviceStream, state: &ResidentState) -> Result<(), EngError> {
        if self.owner != stream.id()
            || self.layout != state.layout
            || self.identity != state.identity
        {
            return Err(EngError::BadRequest(
                "device tip identity, layout or owner differs".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn capture(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &mut ResidentState,
    ) -> Result<DeviceTipKey, EngError> {
        self.check(stream, state)?;
        state.book.begin_export(stream.id())?;
        if let Some(current) = self.current.as_ref()
            && current.positions() == state.tokens()
            && state
                .resident_tip
                .as_ref()
                .is_some_and(|live| live.same(current))
        {
            return Ok(current.clone());
        }
        self.current = None;
        state.resident_tip = None;
        ctx.copy_device(
            stream,
            state.delta.range(0, self.bytes())?,
            self.delta.range(0, self.bytes())?,
        )?;
        let key = DeviceTipKey(Rc::new(state.tokens()));
        self.current = Some(key.clone());
        state.resident_tip = Some(key.clone());
        Ok(key)
    }

    /// A stale key is an ordinary miss: the independently fenced host snapshot remains usable.
    pub(crate) fn restore(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        state: &mut ResidentState,
        key: &DeviceTipKey,
    ) -> Result<bool, EngError> {
        self.check(stream, state)?;
        let matches = state.health() == Health::Valid
            && self
                .current
                .as_ref()
                .is_some_and(|current| current.same(key))
            && state
                .resident_tip
                .as_ref()
                .is_some_and(|live| live.same(key));
        if !matches {
            return Ok(false);
        }
        state
            .book
            .begin_import(stream.id(), IMPORT_REWRITES_WORDS)?;
        state.clear_words(stream)?;
        state.clear_delta(stream, 1)?;
        ctx.copy_device(
            stream,
            self.delta.range(0, self.bytes())?,
            state.delta.range(0, self.bytes())?,
        )?;
        stream.synchronize()?;
        state.book.finish_overwrite(key.positions());
        Ok(true)
    }
}

#[cfg(test)]
#[path = "device_tip_tests.rs"]
mod tests;

impl ResidentState {
    pub(super) fn invalidate_tip_before_write(&mut self) {
        if self
            .resident_tip
            .as_ref()
            .is_some_and(|tip| self.tokens() < tip.positions())
        {
            self.resident_tip = None;
        }
    }
}
