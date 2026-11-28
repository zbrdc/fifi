//! What a step carries from one token to the next.
//!
//! A delta-rule visit keeps a convolution ring and a record, an attention visit a plane of keys
//! and one of values. Each visit of a kind owns a slot. A new sequence starts from zeros.

use super::attend::Cache;
use super::delta::{Delta, DeltaState};
use super::scratch::Alloc;
use crate::EngError;
use crate::kernels::cuda::{DeviceBuffer, DeviceStream};

/// How much the visits of a model carry.
#[derive(Clone, Copy)]
pub(crate) struct Slots {
    /// Delta-rule slots, and the floats of one ring and of one record.
    pub(crate) delta: usize,
    pub(crate) ring: usize,
    pub(crate) record: usize,
    /// Attention slots, the bytes of one plane and the rows it holds.
    pub(crate) attend: usize,
    pub(crate) plane: usize,
    pub(crate) rows: usize,
}

pub(crate) struct State<B = DeviceBuffer> {
    slots: Slots,
    ring: Option<B>,
    record: Option<B>,
    keys: Vec<B>,
    values: Vec<B>,
}

impl<B> State<B> {
    /// The zeroed state of `slots`.
    pub(crate) fn new<A: Alloc<Buf = B>>(a: &A, slots: Slots) -> Result<Self, EngError> {
        let delta = |floats: usize| -> Result<Option<B>, EngError> {
            if slots.delta == 0 {
                return Ok(None);
            }
            a.zeroed(slots.delta * floats * 4).map(Some)
        };
        let planes = || -> Result<Vec<B>, EngError> {
            (0..slots.attend).map(|_| a.zeroed(slots.plane)).collect()
        };
        Ok(Self {
            slots,
            ring: delta(slots.ring)?,
            record: delta(slots.record)?,
            keys: planes()?,
            values: planes()?,
        })
    }
}

impl State {
    fn missing(what: &str) -> EngError {
        EngError::BadRequest(format!("the step carries no {what} for this visit"))
    }

    /// The ring and the record of delta-rule slot `slot`, sized by `layer`.
    pub(crate) fn delta(&self, slot: usize, layer: &Delta<'_>) -> Result<DeltaState<'_>, EngError> {
        let (ring, record) = (layer.ring_floats() * 4, layer.record_floats() * 4);
        let (Some(rings), Some(records)) = (&self.ring, &self.record) else {
            return Err(Self::missing("delta-rule state"));
        };
        Ok(DeltaState {
            ring: rings.range(slot * ring, ring)?,
            record: records.range(slot * record, record)?,
            context: self.slots.rows,
        })
    }

    /// The key and value planes of attention slot `slot`.
    pub(crate) fn cache(&self, slot: usize) -> Result<Cache<'_>, EngError> {
        match (self.keys.get(slot), self.values.get(slot)) {
            (Some(k), Some(v)) => Ok(Cache {
                k,
                v,
                rows: self.slots.rows,
            }),
            _ => Err(Self::missing("attention cache")),
        }
    }

    /// The buffer of every ring and the buffer of every record.
    #[cfg(test)]
    pub(crate) fn delta_buffers(&self) -> Result<(&DeviceBuffer, &DeviceBuffer), EngError> {
        match (&self.ring, &self.record) {
            (Some(rings), Some(records)) => Ok((rings, records)),
            _ => Err(Self::missing("delta-rule state")),
        }
    }

    /// The buffers a sequence cannot cut back to an earlier position: the rings and the records.
    pub(crate) fn running(&self) -> impl Iterator<Item = &DeviceBuffer> {
        self.ring.iter().chain(self.record.iter())
    }

    /// Every carried buffer.
    pub(crate) fn buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        let delta = self.ring.iter().chain(self.record.iter());
        delta.chain(self.keys.iter()).chain(self.values.iter())
    }

    /// Queues the zeros of a new sequence on `stream`.
    pub(crate) fn reset(&self, stream: &DeviceStream) -> Result<(), EngError> {
        self.buffers()
            .try_for_each(|buffer| buffer.zero_at(stream, 0, buffer.bytes()))
    }
}
