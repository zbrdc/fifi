//! Acquisition and completion boundary for stored tensors.
//!
//! A caller submits an [`AcquireRequest`], receives a [`Ticket`], and waits on
//! it for a [`Completion`].

use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use std::{fs::File, io::IoSliceMut, os::unix::fs::FileExt};
use std::{
    io::{Read, SeekFrom},
    sync::Mutex,
};

use super::indexed::{check_shard_set, parse_weight_map};
use super::{
    ResidentWeights,
    shard::{HeaderEntry, ShardHeader},
    tiered::ShardIo,
};
use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Whole,
    /// One slice of the stored leading dimension, such as one expert or one table row.
    LeadingSlice {
        index: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Demand,
    Prefetch,
}

/// T1 supplies only an inactive handle.
#[derive(Debug, Clone, Default)]
pub struct CancelHandle {
    _inactive: (),
}

impl CancelHandle {
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
    Ready(Tensor),
    Missing,
}

#[derive(Debug)]
pub struct Ticket {
    outcome: Result<Completion, EngError>,
}

/// A store that reads tensors of one shard on demand through a seekable source.
///
/// [`Completion::Missing`] means the key is absent
/// from this shard's checked header. [`StreamedCheckpoint`] routes across the
/// shards of one indexed checkpoint.
///
/// The read buffer is bounded only by the shard byte length that the header
/// check enforces. A memory reservation arrives with a later slice.
pub struct StreamedWeights {
    header: ShardHeader,
    reader: Reader,
}

enum Reader {
    Seek(Mutex<Box<dyn ShardIo + Send>>),
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    File(File),
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
struct AtOffset<'a> {
    file: &'a File,
    offset: u64,
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl Read for AtOffset<'_> {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read_at(dst, self.offset)?;
        self.offset += count as u64;
        Ok(count)
    }

    fn read_vectored(&mut self, dst: &mut [IoSliceMut<'_>]) -> std::io::Result<usize> {
        let count = rustix::io::preadv(self.file, dst, self.offset)?;
        self.offset += count as u64;
        Ok(count)
    }
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
fn read_all_vectored(
    reader: &mut dyn Read,
    mut remaining: &mut [std::io::IoSliceMut<'_>],
) -> std::io::Result<()> {
    use std::io::{ErrorKind, IoSliceMut};
    while !remaining.is_empty() {
        match reader.read_vectored(remaining) {
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(n) => IoSliceMut::advance_slices(&mut remaining, n),
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) fn read_all_vectored_at(
    file: &File,
    offset: u64,
    dst: &mut [IoSliceMut<'_>],
) -> std::io::Result<()> {
    read_all_vectored(&mut AtOffset { file, offset }, dst)
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn positioned_reads() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_MLX_POSITIONED_READS").is_some_and(|value| value == "1")
    })
}

impl StreamedWeights {
    pub fn open(
        mut reader: Box<dyn ShardIo + Send>,
        byte_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        let header = Self::read_header(reader.as_mut(), byte_len, max_header_bytes)?;
        Ok(Self {
            header,
            reader: Reader::Seek(Mutex::new(reader)),
        })
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    fn open_file(
        file: std::fs::File,
        byte_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
        if positioned_reads() {
            return Self::open_positioned(file, byte_len, max_header_bytes);
        }
        Self::open(Box::new(file), byte_len, max_header_bytes)
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    fn open_positioned(
        mut file: File,
        byte_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        let header = Self::read_header(&mut file, byte_len, max_header_bytes)?;
        Ok(Self {
            header,
            reader: Reader::File(file),
        })
    }

    fn read_header(
        reader: &mut dyn ShardIo,
        byte_len: u64,
        max_header_bytes: u64,
    ) -> Result<ShardHeader, EngError> {
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| EngError::Checkpoint(format!("shard seek failed: {error}")))?;
        ShardHeader::read(reader, byte_len, max_header_bytes)
    }

    /// The absolute byte position of `key`'s data within this shard's file
    /// and its length, for callers that re-open the shard for direct reads:
    /// the streaming reads seek to this same position. `None` when the key
    /// is absent from this shard's header.
    pub fn extent(&self, key: &str) -> Option<(u64, u64)> {
        let entry = self.header.entry(key)?;
        let offsets = entry.offsets();
        if offsets.end < offsets.start {
            return None;
        }
        let start = self.header.data_start().checked_add(offsets.start)?;
        Some((start, offsets.end - offsets.start))
    }

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

    /// Reads consecutive leading slices into their contiguous destination.
    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    fn read_slice_run_into(
        &self,
        key: &str,
        first: u32,
        count: u32,
        dst: &mut [u8],
    ) -> Result<(), EngError> {
        let start = self.slice_run_start(key, first, count, dst.len())?;
        self.read_range_into(key, start, dst)
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    fn slice_run_start(
        &self,
        key: &str,
        first: u32,
        count: u32,
        bytes: usize,
    ) -> Result<u64, EngError> {
        let Some((entry, start, slice_len)) = self.slice_span(key, first)? else {
            return Err(EngError::Checkpoint(format!("missing tensor {key}")));
        };
        if count == 0 || count > entry.shape()[0] - first {
            return Err(EngError::BadRequest(format!(
                "tensor {key:?}: invalid run of {count} slices starting at {first}"
            )));
        }
        let expected = u64::from(count).checked_mul(slice_len);
        if expected != u64::try_from(bytes).ok() {
            return Err(EngError::BadRequest(format!(
                "tensor {key:?}: slice run does not fill the {}-byte destination",
                bytes
            )));
        }
        Ok(start)
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    fn read_slice_run_scattered(
        &self,
        key: &str,
        first: u32,
        dst: &mut [std::io::IoSliceMut<'_>],
    ) -> Result<(), EngError> {
        let count =
            u32::try_from(dst.len()).map_err(|_| read_failure(key, "too many destinations"))?;
        let bytes = dst
            .iter()
            .try_fold(0usize, |total, slice| total.checked_add(slice.len()))
            .ok_or_else(|| read_failure(key, "scatter byte count overflows"))?;
        if dst.windows(2).any(|pair| pair[0].len() != pair[1].len()) {
            return Err(EngError::BadRequest(format!(
                "tensor {key:?}: scatter slices have different lengths"
            )));
        }
        let start = self.slice_run_start(key, first, count, bytes)?;
        self.read_range_with(key, start, bytes, |reader| read_all_vectored(reader, dst))
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

    fn read_range_into(&self, key: &str, start: u64, dst: &mut [u8]) -> Result<(), EngError> {
        self.read_range_with(key, start, dst.len(), |reader| reader.read_exact(dst))
    }

    fn read_range_with(
        &self,
        key: &str,
        start: u64,
        bytes: usize,
        read: impl FnOnce(&mut dyn Read) -> std::io::Result<()>,
    ) -> Result<(), EngError> {
        let position =
            self.header.data_start().checked_add(start).ok_or_else(|| {
                EngError::Checkpoint(format!("tensor {key:?}: offset overflows u64"))
            })?;
        let (clock, outcome) = match &self.reader {
            Reader::Seek(reader) => {
                let mut reader = reader
                    .lock()
                    .map_err(|_| read_failure(key, "reader lock poisoned"))?;
                // The original timing excludes lock acquisition.
                let clock = read_stats::now();
                let outcome = reader
                    .seek(SeekFrom::Start(position))
                    .and_then(|_| read(reader.as_mut()));
                (clock, outcome)
            }
            #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
            Reader::File(file) => (
                read_stats::now(),
                read(&mut AtOffset {
                    file,
                    offset: position,
                }),
            ),
        };
        if let Some(clock) = clock {
            read_stats::record(bytes as u64, clock.elapsed());
        }
        outcome.map_err(|error| read_failure(key, &error.to_string()))
    }
}

/// The cold-load read accounting, live only in test builds (the load harness is a test): a
/// production build compiles `now` to a constant `None` and `record` to nothing, so the
/// production read path is behaviorally equivalent to what it was. When switched on around
/// a load phase, every positioned tensor read accumulates its bytes, wall time, and call
/// count, giving the phase's effective read rate without touching the read itself.
mod read_stats {
    use std::time::{Duration, Instant};

    #[cfg(test)]
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[cfg(test)]
    static ENABLED: AtomicBool = AtomicBool::new(false);
    #[cfg(test)]
    static NS: AtomicU64 = AtomicU64::new(0);
    #[cfg(test)]
    static BYTES: AtomicU64 = AtomicU64::new(0);
    #[cfg(test)]
    static CALLS: AtomicU64 = AtomicU64::new(0);

    #[cfg(test)]
    pub(super) fn now() -> Option<Instant> {
        ENABLED.load(Ordering::Relaxed).then(Instant::now)
    }

    #[cfg(not(test))]
    pub(super) fn now() -> Option<Instant> {
        None
    }

    #[cfg(test)]
    pub(super) fn record(bytes: u64, elapsed: Duration) {
        NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(not(test))]
    pub(super) fn record(_bytes: u64, _elapsed: Duration) {}

    /// The load harness brackets exactly its load phases.
    #[cfg(all(
        test,
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub fn set(on: bool) {
        ENABLED.store(on, Ordering::Relaxed);
    }

    /// The accumulated bytes, wall nanoseconds, and call count since the last reset.
    #[cfg(all(
        test,
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub fn snapshot() -> (u64, u64, u64) {
        (
            BYTES.load(Ordering::Relaxed),
            NS.load(Ordering::Relaxed),
            CALLS.load(Ordering::Relaxed),
        )
    }

    #[cfg(all(
        test,
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub fn reset() {
        NS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        CALLS.store(0, Ordering::Relaxed);
    }
}

/// The harness-facing surface of the accounting, visible to the load-timing test only.
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) use read_stats::{reset, set, snapshot};

/// `open` checks that the index and the shard headers name the same keys in the same shards.
pub struct StreamedCheckpoint {
    stores: Vec<StreamedWeights>,
    route: BTreeMap<String, usize>,
    /// Each store's shard file name, parallel to `stores`, so an extent can
    /// name the file its bytes live in.
    shard_names: Vec<String>,
}

impl StreamedCheckpoint {
    /// The shard file name, the absolute data start, and the byte length of
    /// `key`'s data inside that shard, for callers that re-open the shard
    /// file for direct reads: the streaming reads seek to these same file
    /// positions. `None` when the key is absent.
    pub fn tensor_extent(&self, key: &str) -> Option<(String, u64, u64)> {
        let idx = *self.route.get(key)?;
        let (start, len) = self.stores[idx].extent(key)?;
        Some((self.shard_names[idx].clone(), start, len))
    }

    pub fn open(
        index: &[u8],
        shards: Vec<(String, Box<dyn ShardIo + Send>, u64)>,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        Self::open_shards(
            parse_weight_map(index)?,
            shards,
            max_header_bytes,
            StreamedWeights::open,
        )
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(crate) fn open_files(
        index: &[u8],
        shards: Vec<(String, std::fs::File, u64)>,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        Self::open_shards(
            parse_weight_map(index)?,
            shards,
            max_header_bytes,
            StreamedWeights::open_file,
        )
    }

    /// Opens the local indexed shards for the resident MLX model loaders.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn open_local(
        dir: &std::path::Path,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        let guard = crate::storage_guard::LocalGuard::new()?;
        guard.check(dir, "model weights")?;
        let index_path = dir.join("model.safetensors.index.json");
        guard.check(&index_path, "model weights")?;
        let index = std::fs::read(&index_path)
            .map_err(|error| EngError::Checkpoint(format!("{}: {error}", index_path.display())))?;
        let weight_map = parse_weight_map(&index)?;
        let names: BTreeSet<_> = weight_map.values().collect();
        let mut shards = Vec::with_capacity(names.len());
        for name in names {
            let path = dir.join(name);
            guard.check(&path, "model weights")?;
            let error = |cause| EngError::Checkpoint(format!("{}: {cause}", path.display()));
            let file = File::open(&path).map_err(error)?;
            let bytes = file.metadata().map_err(error)?.len();
            shards.push((name.clone(), file, bytes));
        }
        Self::open_shards(
            weight_map,
            shards,
            max_header_bytes,
            StreamedWeights::open_file,
        )
    }

    fn open_shards<R>(
        weight_map: BTreeMap<String, String>,
        shards: Vec<(String, R, u64)>,
        max_header_bytes: u64,
        open: impl Fn(R, u64, u64) -> Result<StreamedWeights, EngError>,
    ) -> Result<Self, EngError> {
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
            let store = open(reader, byte_len, max_header_bytes).map_err(|error| match error {
                EngError::Checkpoint(message) => {
                    EngError::Checkpoint(format!("shard {name:?}: {message}"))
                }
                EngError::MemoryPlan(message) => {
                    EngError::MemoryPlan(format!("shard {name:?}: {message}"))
                }
                other => other,
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
        // `shard_names[i]` names `stores[i]`, from the inverted positions map.
        let mut shard_names = vec![String::new(); stores.len()];
        for (name, &position) in &positions {
            shard_names[position] = name.clone();
        }
        Ok(Self {
            stores,
            route,
            shard_names,
        })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.route.contains_key(key)
    }

    /// Every tensor name the checkpoint's index routes, in sorted name order (the route is a
    /// sorted map). A coverage survey diffs this list against the names a load reads.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.route.keys().map(String::as_str)
    }

    pub fn stored(&self, key: &str) -> Option<(Dtype, Vec<u32>)> {
        let index = *self.route.get(key)?;
        let entry = self.stores.get(index)?.header.entry(key)?;
        Some((entry.dtype(), entry.shape().to_vec()))
    }

    /// Payload length from the checked header, without reading tensor data.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn stored_byte_len(&self, key: &str) -> Option<u64> {
        let shard = *self.route.get(key)?;
        let offsets = self.stores.get(shard)?.header.entry(key)?.offsets();
        offsets.end.checked_sub(offsets.start)
    }

    /// Reads selected whole tensors into a resident source; absent keys are omitted.
    ///
    /// Duplicate keys count once. The aggregate stored payload must fit `max_bytes` and host
    /// limits before any payload read. A zero budget is `BadRequest`; an excess is `MemoryPlan`.
    pub fn read_selected(
        &self,
        keys: &[&str],
        max_bytes: u64,
    ) -> Result<ResidentWeights, EngError> {
        if max_bytes == 0 {
            return Err(EngError::BadRequest("input budget must be nonzero".into()));
        }
        let keys: BTreeSet<_> = keys.iter().copied().collect();
        let budget = max_bytes.min(isize::MAX as u64);
        let mut resident_bytes = 0u64;
        for key in &keys {
            let Some(&index) = self.route.get(*key) else {
                continue;
            };
            let entry = self
                .stores
                .get(index)
                .and_then(|store| store.header.entry(key))
                .ok_or_else(invalid_route)?;
            let offsets = entry.offsets();
            resident_bytes = offsets
                .end
                .checked_sub(offsets.start)
                .and_then(|len| resident_bytes.checked_add(len))
                .filter(|&bytes| bytes <= budget)
                .ok_or_else(|| {
                    EngError::MemoryPlan(
                        "selected payload exceeds its budget or host limits".into(),
                    )
                })?;
        }
        let mut tensors = HashMap::new();
        for key in keys {
            let request = AcquireRequest::new(
                key.to_string(),
                Selection::Whole,
                Priority::Demand,
                CancelHandle::new(),
            );
            if let Completion::Ready(tensor) = self.wait(self.acquire(request)?)? {
                tensors.insert(key.to_string(), tensor);
            }
        }
        Ok(ResidentWeights {
            tensors,
            resident_bytes,
        })
    }

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

    /// See [`StreamedWeights::read_slice_run_into`].
    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(crate) fn read_slice_run_into(
        &self,
        key: &str,
        first: u32,
        count: u32,
        dst: &mut [u8],
    ) -> Result<(), EngError> {
        let &store = self
            .route
            .get(key)
            .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {key}")))?;
        self.stores
            .get(store)
            .ok_or_else(invalid_route)?
            .read_slice_run_into(key, first, count, dst)
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    pub(crate) fn read_slice_run_scattered(
        &self,
        key: &str,
        first: u32,
        dst: &mut [std::io::IoSliceMut<'_>],
    ) -> Result<(), EngError> {
        let shard = *self
            .route
            .get(key)
            .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {key}")))?;
        self.stores
            .get(shard)
            .ok_or_else(invalid_route)?
            .read_slice_run_scattered(key, first, dst)
    }

    pub fn wait(&self, ticket: Ticket) -> Result<Completion, EngError> {
        ticket.outcome
    }
}

/// `open` builds the route from the checked shard set.
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
    use crate::weights::WeightSource;

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

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn positioned_shard_handles_concurrent_ranges_and_a_partial_vectored_read() {
        use std::io::{Seek, Write};

        let payload: Vec<u8> = (0..24).collect();
        let (bytes, _) = shard(&[("experts", "U32", vec![6, 1], payload.clone())]);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&bytes).unwrap();
        let store = StreamedWeights::open_positioned(
            file.try_clone().unwrap(),
            bytes.len() as u64,
            1 << 20,
        )
        .unwrap();
        let cursor = file.seek(SeekFrom::Start(9)).unwrap();
        std::thread::scope(|scope| {
            for first in [0u32, 2, 4] {
                let store = &store;
                let payload = &payload;
                scope.spawn(move || {
                    for _ in 0..32 {
                        let mut contiguous = [0; 8];
                        store
                            .read_slice_run_into("experts", first, 2, &mut contiguous)
                            .unwrap();
                        let start = first as usize * 4;
                        assert_eq!(contiguous, payload[start..start + 8]);
                        let mut a = [0; 4];
                        let mut b = [0; 4];
                        store
                            .read_slice_run_scattered(
                                "experts",
                                first,
                                &mut [IoSliceMut::new(&mut a), IoSliceMut::new(&mut b)],
                            )
                            .unwrap();
                        assert_eq!(a, contiguous[..4]);
                        assert_eq!(b, contiguous[4..]);
                    }
                });
            }
        });
        assert_eq!(file.stream_position().unwrap(), cursor);

        file.set_len(bytes.len() as u64 - 2).unwrap();
        let mut a = [0xAB; 4];
        let mut b = [0xAB; 4];
        let result = store.read_slice_run_scattered(
            "experts",
            4,
            &mut [IoSliceMut::new(&mut a), IoSliceMut::new(&mut b)],
        );
        assert!(matches!(
            result,
            Err(EngError::Backend {
                operation: "weight_read",
                ..
            })
        ));
        assert_eq!(a, payload[16..20]);
        assert_eq!(b, [20, 21, 0xAB, 0xAB]);
        assert_eq!(file.stream_position().unwrap(), cursor);
    }

    #[test]
    fn selected_whole_tensors_omit_unrequested_payload() {
        let (mut file, index) = shard(&[
            ("a", "F16", vec![2], vec![0, 60, 0, 64]),
            ("b", "U32", vec![1], vec![9, 8, 7, 6]),
        ]);
        let byte_len = u64::try_from(file.len()).unwrap();
        file.truncate(file.len() - 4);
        let checkpoint = StreamedCheckpoint::open(
            &index,
            vec![(
                "m.safetensors".to_string(),
                Box::new(std::io::Cursor::new(file)),
                byte_len,
            )],
            1 << 20,
        )
        .unwrap();

        let selected = checkpoint.read_selected(&["a", "absent", "a"], 4).unwrap();
        let tensor = selected.tensor("a").unwrap();
        assert_eq!(tensor.dtype(), Dtype::F16);
        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(tensor.host_bytes().unwrap(), &[0, 60, 0, 64]);
        assert_eq!(selected.resident_bytes(), 4);
        assert!(selected.tensor("b").is_none());
        assert!(selected.tensor("absent").is_none());
    }

    #[test]
    fn selected_budget_is_aggregate_and_checked_before_reads() {
        let (mut file, index) = shard(&[
            ("a", "U32", vec![1], vec![1, 2, 3, 4]),
            ("b", "U32", vec![1], vec![5, 6, 7, 8]),
        ]);
        let byte_len = u64::try_from(file.len()).unwrap();
        file.truncate(file.len() - 8);
        let checkpoint = StreamedCheckpoint::open(
            &index,
            vec![(
                "m.safetensors".to_string(),
                Box::new(std::io::Cursor::new(file)),
                byte_len,
            )],
            1 << 20,
        )
        .unwrap();

        assert!(matches!(
            checkpoint.read_selected(&["a", "b"], 7),
            Err(EngError::MemoryPlan(_))
        ));
        assert!(matches!(
            checkpoint.read_selected(&["a"], 0),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            checkpoint.read_selected(&["a"], 4),
            Err(EngError::Backend { .. })
        ));
    }

    #[test]
    fn read_slice_into_writes_the_bytes_of_each_slice() {
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
    fn read_slice_run_checks_extent_and_destination_before_writing() {
        let (file, index) = shard(&[("a", "U32", vec![3, 2, 3], (0..72u8).collect())]);
        let ckpt = open(file, &index);
        let mut dst = [0xEE; 48];
        for (first, count) in [(0, 0), (0, 1), (2, 2), (3, 2), (1, u32::MAX)] {
            assert!(matches!(
                ckpt.read_slice_run_into("a", first, count, &mut dst),
                Err(EngError::BadRequest(_))
            ));
            assert_eq!(dst, [0xEE; 48]);
        }
        ckpt.read_slice_run_into("a", 1, 2, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), (24..72u8).collect::<Vec<_>>());
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
