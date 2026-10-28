//! Immutable canonical expert payloads retained in pageable host memory.

use super::{EXPERTS, PART_BYTES, source::GgufSource};
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayload, ExpertPayloadSource, ExpertPayloadView},
    weights::GgufHeader,
};
use std::{fs::File, mem::size_of};

struct PreparedExpert {
    id: ExpertId,
    parts: [ExpertPayload; 3],
}

pub(crate) struct PreparedSource {
    entries: Vec<PreparedExpert>,
    retained_host_bytes: u64,
}

fn memory_error() -> EngError {
    EngError::MemoryPlan("Flint prepared expert allocation or host budget exceeded".into())
}

fn reserve<T>(count: usize) -> Result<Vec<T>, EngError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| memory_error())?;
    Ok(values)
}

fn charge(total: u64, count: usize, bytes: usize) -> Result<u64, EngError> {
    count
        .checked_mul(bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .and_then(|bytes| total.checked_add(bytes))
        .ok_or_else(memory_error)
}

fn check_budget(bytes: u64, budget: u64) -> Result<(), EngError> {
    if bytes > budget {
        return Err(memory_error());
    }
    Ok(())
}

fn ordered_ids(ids: &[ExpertId]) -> Result<Vec<ExpertId>, EngError> {
    let mut ordered = reserve(ids.len())?;
    for &id in ids {
        if !(2..49).contains(&id.layer) || id.expert >= EXPERTS as u32 {
            return Err(EngError::BadRequest(
                "invalid Flint prepared expert ID".into(),
            ));
        }
        ordered.push(id);
    }
    ordered.sort_unstable();
    if ordered.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(EngError::BadRequest(
            "duplicate Flint prepared expert ID".into(),
        ));
    }
    Ok(ordered)
}

impl PreparedSource {
    /// The budget covers retained capacities/index only, not transient copies or file cache.
    /// The caller must admit those separately; this is not a global host-memory permit.
    pub(crate) fn new(
        file: &File,
        header: &GgufHeader,
        ids: &[ExpertId],
        host_budget: u64,
    ) -> Result<Self, EngError> {
        let metadata = charge(
            size_of::<Self>() as u64,
            ids.len(),
            size_of::<PreparedExpert>(),
        )?;
        check_budget(
            charge(metadata, ids.len(), 3 * PART_BYTES as usize)?,
            host_budget,
        )?;
        let ids = ordered_ids(ids)?;
        let mut source = GgufSource::new(file, header)?;
        let mut entries = reserve::<PreparedExpert>(ids.len())?;
        let mut bytes = charge(
            size_of::<Self>() as u64,
            entries.capacity(),
            size_of::<PreparedExpert>(),
        )?;
        check_budget(bytes, host_budget)?;
        for id in ids {
            let parts = [
                source.payload(id, 0)?,
                source.payload(id, 1)?,
                source.payload(id, 2)?,
            ];
            for part in &parts {
                bytes = charge(bytes, part.data.capacity(), 1)?;
                bytes = charge(bytes, part.scales.capacity(), 1)?;
            }
            check_budget(bytes, host_budget)?;
            entries.push(PreparedExpert { id, parts });
        }
        Ok(Self {
            entries,
            retained_host_bytes: bytes,
        })
    }

    pub(crate) fn retained_host_bytes(&self) -> u64 {
        self.retained_host_bytes
    }

    fn part(&self, id: ExpertId, part: usize) -> Result<&ExpertPayload, EngError> {
        let index = self
            .entries
            .binary_search_by_key(&id, |entry| entry.id)
            .map_err(|_| EngError::BadRequest("Flint expert was not prepared".into()))?;
        self.entries[index]
            .parts
            .get(part)
            .ok_or_else(|| EngError::BadRequest("Flint expert part must be gate/up/down".into()))
    }
}

fn copy(bytes: &[u8]) -> Result<Vec<u8>, EngError> {
    let mut owned = reserve(bytes.len())?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

impl ExpertPayloadSource for PreparedSource {
    fn parts(&self) -> usize {
        3
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        let view = self.part(id, part)?;
        Ok(ExpertPayload {
            data: copy(&view.data)?,
            scales: copy(&view.scales)?,
            tensor_scale: view.tensor_scale,
            shape: view.shape,
        })
    }

    fn with_payload<R>(
        &mut self,
        id: ExpertId,
        part: usize,
        consume: impl FnOnce(ExpertPayloadView<'_>) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        consume(self.part(id, part)?.as_view())
    }

    fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
        Err(EngError::BadRequest(
            "Flint prepared experts are retained until drop".into(),
        ))
    }

    fn release_experts(&mut self, _: &[ExpertId]) -> Result<(), EngError> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod tests;
