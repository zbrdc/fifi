//! Acquisition and completion boundary for stored tensors.
//!
//! A caller submits an [`AcquireRequest`], receives a [`Ticket`], and waits on
//! it for a [`Completion`]. See `acquisition-contract.md`, in the ledger.

use std::collections::{BTreeMap, BTreeSet};
use std::{io::SeekFrom, sync::Mutex};

use super::indexed::{check_shard_set, parse_weight_map};
use super::{
    shard::{HeaderEntry, ShardHeader},
    tiered::ShardIo,
};
use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

/// Which part of a stored tensor a request selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// The complete stored tensor.
    Whole,
    /// One slice of the stored leading dimension, such as one expert or one table row.
    LeadingSlice {
        /// Index into the stored leading dimension.
        index: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Scheduling class of a request.
pub enum Priority {
    Demand,
    Prefetch,
}

/// Cancellation state for one request. T1 supplies only an inactive handle.
#[derive(Debug, Clone, Default)]
pub struct CancelHandle {
    _inactive: (),
}

impl CancelHandle {
    /// Creates an inactive handle.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct AcquireRequest {
    key: String,
    selection: Selection,
    #[expect(
        dead_code,
        reason = "T2 to T6 read this field; acquisition-contract.md section 10"
    )]
    priority: Priority,
    #[expect(
        dead_code,
        reason = "T2 to T6 read this field; acquisition-contract.md section 10"
    )]
    cancel: CancelHandle,
}

impl AcquireRequest {
    /// Builds a request for the exact stored tensor `key`.
    pub fn new(
        key: String,
        selection: Selection,
        priority: Priority,
        cancel: CancelHandle,
    ) -> Self {
        Self {
            key,
            selection,
            priority,
            cancel,
        }
    }
}

#[derive(Debug)]
pub enum Completion {
    /// Complete checked portable bytes with stored dtype and shape.
    Ready(Tensor),
    /// The key is absent from the checked index.
    Missing,
}

/// One caller's handle on a pending acquisition.
#[derive(Debug)]
pub struct Ticket {
    outcome: Result<Completion, EngError>,
}

/// A store that reads tensors of one shard on demand through a seekable source.
///
/// One store covers one shard. [`Completion::Missing`] means the key is absent
/// from this shard's checked header. [`StreamedCheckpoint`] routes across the
/// shards of one indexed checkpoint.
///
/// The read buffer is bounded only by the shard byte length that the header
/// check enforces. A memory reservation arrives with a later slice.
pub struct StreamedWeights {
    header: ShardHeader,
    reader: Mutex<Box<dyn ShardIo + Send>>,
}

impl StreamedWeights {
    /// Reads and checks the header from `reader`, which holds `byte_len` bytes.
    pub fn open(
        mut reader: Box<dyn ShardIo + Send>,
        byte_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| EngError::Checkpoint(format!("shard seek failed: {error}")))?;
        let header = ShardHeader::read(&mut reader, byte_len, max_header_bytes)?;
        Ok(Self {
            header,
            reader: Mutex::new(reader),
        })
    }

    /// Accepts the request and starts the read.
    pub fn acquire(&self, request: AcquireRequest) -> Result<Ticket, EngError> {
        let outcome = match request.selection {
            Selection::Whole => self.read_whole(&request.key),
            Selection::LeadingSlice { index } => self.read_slice(&request.key, index),
        };
        Ok(Ticket { outcome })
    }

    /// Returns `Ok(Missing)` for an absent key and `Err(Backend)` for a failed read.
    pub fn wait(&self, ticket: Ticket) -> Result<Completion, EngError> {
        ticket.outcome
    }

    fn read_whole(&self, key: &str) -> Result<Completion, EngError> {
        let Some(entry) = self.header.entry(key) else {
            return Ok(Completion::Missing);
        };
        let offsets = entry.offsets();
        let len = offsets.end.checked_sub(offsets.start).ok_or_else(|| {
            EngError::Checkpoint(format!("tensor {key:?}: offsets end before they start"))
        })?;
        let bytes = self.read_range(key, offsets.start, len)?;
        Tensor::try_from_host(entry.dtype(), entry.shape().to_vec(), bytes).map(Completion::Ready)
    }

    fn read_slice(&self, key: &str, index: u32) -> Result<Completion, EngError> {
        let Some((entry, start, len)) = self.slice_span(key, index)? else {
            return Ok(Completion::Missing);
        };
        let bytes = self.read_range(key, start, len)?;
        Tensor::try_from_host(entry.dtype(), entry.shape()[1..].to_vec(), bytes)
            .map(Completion::Ready)
    }

    /// Reads leading slice `index` of `key` into `dst` without an intermediate buffer.
    ///
    /// The checks of `Selection::LeadingSlice` apply. A destination whose length differs from the
    /// slice length is `BadRequest`, before any read, and leaves `dst` unchanged. An absent key is
    /// `Checkpoint`. A failed read is `Backend` and can leave `dst` partly written.
    pub fn read_slice_into(&self, key: &str, index: u32, dst: &mut [u8]) -> Result<(), EngError> {
        let Some((_, start, len)) = self.slice_span(key, index)? else {
            return Err(EngError::Checkpoint(format!("missing tensor {key}")));
        };
        if u64::try_from(dst.len()).ok() != Some(len) {
            return Err(EngError::BadRequest(format!(
                "tensor {key:?}: slice {index} has {len} bytes, the destination holds {}",
                dst.len()
            )));
        }
        self.read_range_into(key, start, dst)
    }

    /// The entry, data offset, and byte length of leading slice `index` of `key`.
    /// `None` when the key is absent.
    fn slice_span(
        &self,
        key: &str,
        index: u32,
    ) -> Result<Option<(&HeaderEntry, u64, u64)>, EngError> {
        let Some(entry) = self.header.entry(key) else {
            return Ok(None);
        };
        let shape = entry.shape();
        if shape.is_empty() {
            return Err(EngError::BadRequest(format!(
                "tensor {key:?}: a scalar has no leading slice"
            )));
        }
        let lead = u64::from(shape[0]);
        if u64::from(index) >= lead {
            return Err(EngError::BadRequest(format!(
                "tensor {key:?}: slice {index} is outside leading dimension {lead}"
            )));
        }
        let offsets = entry.offsets();
        let len = offsets.end.checked_sub(offsets.start).ok_or_else(|| {
            EngError::Checkpoint(format!("tensor {key:?}: offsets end before they start"))
        })?;
        if len % lead != 0 {
            return Err(EngError::Checkpoint(format!(
                "tensor {key:?}: byte length does not divide by the leading dimension"
            )));
        }
        let slice_len = len / lead;
        let start = u64::from(index)
            .checked_mul(slice_len)
            .and_then(|offset| offsets.start.checked_add(offset))
            .ok_or_else(|| {
                EngError::Checkpoint(format!("tensor {key:?}: slice offset overflows u64"))
            })?;
        Ok(Some((entry, start, slice_len)))
    }

    fn read_range(&self, key: &str, start: u64, len: u64) -> Result<Vec<u8>, EngError> {
        let len = usize::try_from(len).map_err(|_| {
            EngError::MemoryPlan(format!("tensor {key:?}: byte length exceeds host limits"))
        })?;
        let mut bytes = vec![0u8; len];
        self.read_range_into(key, start, &mut bytes)?;
        Ok(bytes)
    }

    /// Fills `dst` from data offset `start` with one seek and one `read_exact` under the reader lock.
    fn read_range_into(&self, key: &str, start: u64, dst: &mut [u8]) -> Result<(), EngError> {
        let position =
            self.header.data_start().checked_add(start).ok_or_else(|| {
                EngError::Checkpoint(format!("tensor {key:?}: offset overflows u64"))
            })?;
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| read_failure(key, "reader lock poisoned"))?;
        reader
            .seek(SeekFrom::Start(position))
            .and_then(|_| reader.read_exact(dst))
            .map_err(|error| read_failure(key, &error.to_string()))
    }
}

/// Streamed stores for every shard of one indexed checkpoint.
///
/// `open` checks that the index and the shard headers name the same keys in the same shards.
/// Each request goes to the store of the shard that holds its key.
pub struct StreamedCheckpoint {
    stores: Vec<StreamedWeights>,
    route: BTreeMap<String, usize>,
}

impl StreamedCheckpoint {
    /// Opens one store per `(shard name, reader, byte length)` and checks them against `index`.
    pub fn open(
        index: &[u8],
        shards: Vec<(String, Box<dyn ShardIo + Send>, u64)>,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        let weight_map = parse_weight_map(index)?;
        let mut names = BTreeSet::new();
        for (name, _, _) in &shards {
            if !names.insert(name.as_str()) {
                return Err(EngError::Checkpoint(format!(
                    "shard {name:?} is supplied twice"
                )));
            }
        }
        check_shard_set(&weight_map, &names)?;

        let mut stores = Vec::with_capacity(shards.len());
        let mut positions = BTreeMap::new();
        for (name, reader, byte_len) in shards {
            let store =
                StreamedWeights::open(reader, byte_len, max_header_bytes).map_err(|error| {
                    match error {
                        EngError::Checkpoint(message) => {
                            EngError::Checkpoint(format!("shard {name:?}: {message}"))
                        }
                        EngError::MemoryPlan(message) => {
                            EngError::MemoryPlan(format!("shard {name:?}: {message}"))
                        }
                        other => other,
                    }
                })?;
            positions.insert(name, stores.len());
            stores.push(store);
        }

        let mut route = BTreeMap::new();
        for (key, shard) in &weight_map {
            let position = *positions.get(shard).ok_or_else(invalid_route)?;
            let store = stores.get(position).ok_or_else(invalid_route)?;
            if store.header.entry(key).is_none() {
                return Err(EngError::Checkpoint(format!(
                    "tensor {key:?} is missing from shard {shard:?}"
                )));
            }
            route.insert(key.clone(), position);
        }

        for (name, &position) in &positions {
            let store = stores.get(position).ok_or_else(invalid_route)?;
            for (key, _) in store.header.iter() {
                if route.get(key) != Some(&position) {
                    return Err(EngError::Checkpoint(format!(
                        "shard {name:?} holds unindexed tensor {key:?}"
                    )));
                }
            }
        }
        Ok(Self { stores, route })
    }

    /// Returns true when the index names `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.route.contains_key(key)
    }

    /// Returns the stored dtype and shape of `key`, or `None` when the index does not name it.
    pub fn stored(&self, key: &str) -> Option<(Dtype, Vec<u32>)> {
        let index = *self.route.get(key)?;
        let entry = self.stores.get(index)?.header.entry(key)?;
        Some((entry.dtype(), entry.shape().to_vec()))
    }

    /// Accepts the request and starts the read in the shard that holds the key.
    pub fn acquire(&self, request: AcquireRequest) -> Result<Ticket, EngError> {
        match self.route.get(&request.key) {
            None => Ok(Ticket {
                outcome: Ok(Completion::Missing),
            }),
            Some(&index) => self
                .stores
                .get(index)
                .ok_or_else(invalid_route)?
                .acquire(request),
        }
    }

    /// Reads leading slice `index` of `key` into `dst` in the shard that holds the key.
    /// See [`StreamedWeights::read_slice_into`].
    pub fn read_slice_into(&self, key: &str, index: u32, dst: &mut [u8]) -> Result<(), EngError> {
        let &store = self
            .route
            .get(key)
            .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {key}")))?;
        self.stores
            .get(store)
            .ok_or_else(invalid_route)?
            .read_slice_into(key, index, dst)
    }

    /// Returns the outcome of a ticket from this checkpoint.
    pub fn wait(&self, ticket: Ticket) -> Result<Completion, EngError> {
        ticket.outcome
    }
}

/// Reports a route that names no store. `open` builds the route from the checked shard set.
fn invalid_route() -> EngError {
    EngError::Checkpoint("checkpoint route is invalid".into())
}

fn read_failure(key: &str, message: &str) -> EngError {
    EngError::Backend {
        operation: "weight_read",
        message: format!("tensor {key:?}: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One shard holding `tensors`, with its index.
    fn shard(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> (Vec<u8>, Vec<u8>) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name.to_string(), serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        (file, index)
    }

    fn open(file: Vec<u8>, index: &[u8]) -> StreamedCheckpoint {
        let len = u64::try_from(file.len()).unwrap();
        let reader: Box<dyn ShardIo + Send> = Box::new(std::io::Cursor::new(file));
        StreamedCheckpoint::open(
            index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }

    fn slice_by_ticket(ckpt: &StreamedCheckpoint, key: &str, index: u32) -> Vec<u8> {
        let request = AcquireRequest::new(
            key.to_string(),
            Selection::LeadingSlice { index },
            Priority::Demand,
            CancelHandle::new(),
        );
        let ticket = ckpt.acquire(request).unwrap();
        match ckpt.wait(ticket).unwrap() {
            Completion::Ready(tensor) => tensor.host_bytes().unwrap().to_vec(),
            Completion::Missing => panic!("{key} is missing"),
        }
    }

    #[test]
    fn read_slice_into_writes_the_bytes_of_each_slice() {
        // Three slices of 24 bytes and two slices of 6 bytes, with distinct bytes everywhere.
        let (file, index) = shard(&[
            ("a", "U32", vec![3, 2, 3], (0..72u8).collect()),
            ("b", "BF16", vec![2, 3], (100..112u8).collect()),
        ]);
        let ckpt = open(file, &index);
        for (key, slices, len) in [("a", 3u32, 24usize), ("b", 2, 6)] {
            for slice in 0..slices {
                let mut dst = vec![0xEE; len];
                ckpt.read_slice_into(key, slice, &mut dst).unwrap();
                assert_eq!(
                    dst,
                    slice_by_ticket(&ckpt, key, slice),
                    "{key} slice {slice}"
                );
            }
        }
    }

    #[test]
    fn read_slice_into_rejects_a_wrong_destination_before_reading() {
        let (file, index) = shard(&[("a", "U32", vec![3, 2, 3], (0..72u8).collect())]);
        let ckpt = open(file, &index);
        for len in [23usize, 25, 0] {
            let mut dst = vec![0xEE; len];
            let result = ckpt.read_slice_into("a", 1, &mut dst);
            assert!(
                matches!(result, Err(EngError::BadRequest(_))),
                "length {len}: {result:?}"
            );
            assert!(
                dst.iter().all(|&byte| byte == 0xEE),
                "length {len}: unchanged"
            );
        }
        let mut dst = vec![0xEE; 24];
        assert!(matches!(
            ckpt.read_slice_into("a", 3, &mut dst),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            ckpt.read_slice_into("absent", 0, &mut dst),
            Err(EngError::Checkpoint(_))
        ));
        assert!(
            dst.iter().all(|&byte| byte == 0xEE),
            "rejections write nothing"
        );
    }
}
