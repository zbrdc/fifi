//! Host allocation and physical storage policy for the CUDA state.
//! Physical capacity is separate from the causal window and compact checkpoint rows.

use super::{config::FULL_LAYERS, reference::ChunkWindow};
use crate::EngError;

const LAYERS: usize = 49;
pub(super) const WINDOW: usize = 513;
const MODEL_CONTEXT_LIMIT: usize = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvStorage {
    F16,
    Q8G32,
}

#[allow(
    dead_code,
    reason = "Portable serving does not use the CUDA KV adapter"
)]
impl KvStorage {
    /// Payload and F16 scale bytes per row of one K or V plane.
    pub(crate) fn plane_row_bytes(self) -> (usize, usize) {
        match self {
            Self::F16 => (4 * 128 * 2, 0),
            Self::Q8G32 => (4 * 128, 4 * (128 / 32) * 2),
        }
    }

    pub(crate) fn row_bytes(self) -> usize {
        let (data, scales) = self.plane_row_bytes();
        2 * (data + scales)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlan {
    context_rows: usize,
    storage: KvStorage,
    cache_rows: [usize; LAYERS],
    snapshots: Option<SnapshotPlan>,
}

impl KvPlan {
    /// The caller admits backend context/chunk limits first. False retains full capacities
    /// and disables these checkpoints; no environment variable or native state is changed.
    pub fn new(context: usize, chunk_rows: usize, ring_enabled: bool) -> Result<Self, EngError> {
        if context == 0 || chunk_rows == 0 {
            return Err(EngError::BadRequest(
                "Flint KV context and chunk rows must be positive".into(),
            ));
        }
        if context > MODEL_CONTEXT_LIMIT {
            return Err(EngError::ContextLimit(format!(
                "Flint KV context exceeds {MODEL_CONTEXT_LIMIT} positions"
            )));
        }
        let required = ChunkWindow::new(WINDOW, WINDOW - 1, chunk_rows)?.required_slots();
        let mut cache_rows = [context; LAYERS];
        if ring_enabled {
            for (layer, capacity) in cache_rows.iter_mut().enumerate() {
                if !FULL_LAYERS.contains(&(layer as u64)) {
                    *capacity = context.min(required);
                }
            }
        }
        Ok(Self {
            context_rows: context,
            storage: KvStorage::F16,
            cache_rows,
            snapshots: ring_enabled.then_some(SnapshotPlan {
                rows: context.min(WINDOW),
            }),
        })
    }

    pub(crate) fn with_storage(mut self, storage: KvStorage) -> Self {
        self.storage = storage;
        self
    }

    #[allow(
        dead_code,
        reason = "Portable serving does not use the CUDA KV adapter"
    )]
    pub(crate) fn storage(&self) -> KvStorage {
        self.storage
    }

    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn context_rows(&self) -> usize {
        self.context_rows
    }

    pub fn cache_rows(&self, layer: usize) -> Result<usize, EngError> {
        self.cache_rows
            .get(layer)
            .copied()
            .ok_or_else(|| EngError::BadRequest("Flint layer is outside 0..49".into()))
    }

    /// Encoded K/V bytes per layer-position include codec scales. Non-row padding,
    /// allocator overhead, and other owners must be charged separately by the caller.
    pub fn bytes(&self, row_bytes: usize) -> Result<usize, EngError> {
        if row_bytes == 0 {
            return Err(byte_error());
        }
        self.cache_rows.iter().try_fold(0usize, |total, rows| {
            rows.checked_mul(row_bytes)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(byte_error)
        })
    }

    pub fn snapshots(&self) -> Option<SnapshotPlan> {
        self.snapshots
    }
}

/// Maximum compact payload shape. Copy ownership/publication belongs to the later state slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotPlan {
    rows: usize,
}

impl SnapshotPlan {
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Two reusable slots, including an in-progress replacement, never a third staging slot.
    pub fn keep(&self) -> usize {
        2
    }

    /// Charge both maximum payloads, with last logits in their actual encoded format.
    pub fn bytes(&self, row_bytes: usize, logits_bytes: usize) -> Result<usize, EngError> {
        if row_bytes == 0 {
            return Err(byte_error());
        }
        self.rows
            .checked_mul(LAYERS - FULL_LAYERS.len())
            .and_then(|rows| rows.checked_mul(row_bytes))
            .and_then(|bytes| bytes.checked_add(logits_bytes))
            .and_then(|bytes| bytes.checked_mul(self.keep()))
            .ok_or_else(byte_error)
    }
}

fn byte_error() -> EngError {
    EngError::MemoryPlan("Flint KV row byte charge is zero or the total overflows".into())
}
