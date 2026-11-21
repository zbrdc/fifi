//! Bounded expert-major sidecars for the native qualification runner.

use std::{
    cell::Cell,
    collections::BTreeMap,
    fs::{self, File, Metadata},
    io::{IoSliceMut, Read},
    os::unix::fs::MetadataExt,
    path::Path,
    time::Duration,
};

use crate::{EngError, family::stored::counted_read, weights::read_all_vectored_at};

use super::ExpertPageLayout;

const MAX_LAYERS: usize = 4;
const EXPERTS: u32 = 256;
const RECORD_BYTES: usize = 1_769_472;

#[derive(serde::Deserialize, PartialEq, Eq)]
pub(super) struct ShardStamp {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: i64,
    mtime_ns: i64,
}

impl From<&Metadata> for ShardStamp {
    fn from(meta: &Metadata) -> Self {
        Self {
            dev: meta.dev(),
            ino: meta.ino(),
            size: meta.len(),
            mtime: meta.mtime(),
            mtime_ns: meta.mtime_nsec(),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    format: String,
    layer: usize,
    shards: Vec<ShardStamp>,
    role_bytes: [usize; 9],
}

pub(super) struct RecordSet {
    layers: BTreeMap<usize, ExpertRecords>,
    enabled: bool,
}

impl RecordSet {
    pub(super) fn open(
        path: &Path,
        shards: &[ShardStamp],
        enabled: bool,
    ) -> Result<Self, EngError> {
        let guard = crate::storage_guard::LocalGuard::new()?;
        guard.check(path, "Flint expert records")?;
        let mut layers = BTreeMap::new();
        if fs::metadata(path).map_err(io)?.is_dir() {
            for entry in fs::read_dir(path).map_err(io)? {
                let file = entry.map_err(io)?.path();
                if !file.extension().is_some_and(|ext| ext == "records") {
                    continue;
                }
                if layers.len() == MAX_LAYERS {
                    return Err(invalid("a sidecar set is limited to four layers"));
                }
                guard.check(&file, "Flint expert records")?;
                let layer = ExpertRecords::open(&file, shards)?;
                if layers.insert(layer.layer, layer).is_some() {
                    return Err(invalid("duplicate layer in sidecar directory"));
                }
            }
        } else {
            let layer = ExpertRecords::open(path, shards)?;
            layers.insert(layer.layer, layer);
        }
        if layers.is_empty() {
            return Err(invalid("no .records files in sidecar directory"));
        }
        Ok(Self { layers, enabled })
    }

    pub(super) fn for_stem(&self, stem: &str) -> Option<(&ExpertRecords, bool)> {
        let layer = stem
            .strip_prefix("model.layers.")?
            .strip_suffix(".mlp.switch_mlp")?
            .parse::<usize>()
            .ok()?;
        self.layers.get(&layer).map(|file| (file, self.enabled))
    }

    pub(super) fn describe(&self) -> String {
        let layers = self.layers.keys().map(usize::to_string).collect::<Vec<_>>();
        let mode = if self.enabled { "records" } else { "shards" };
        format!("layers={} mode={mode}", layers.join(","))
    }
}

pub(super) struct ExpertRecords {
    layer: usize,
    file: File,
    data_start: u64,
    role_bytes: [usize; 9],
}

impl ExpertRecords {
    pub(super) fn open(path: &Path, shards: &[ShardStamp]) -> Result<Self, EngError> {
        if !fs::metadata(path).map_err(io)?.is_file() {
            return Err(invalid("sidecar must be a regular file"));
        }
        let mut file = File::open(path).map_err(io)?;
        let mut length = [0; 8];
        file.read_exact(&mut length).map_err(io)?;
        let length = u64::from_le_bytes(length);
        if !(1..=4096).contains(&length) {
            return Err(invalid("invalid header length"));
        }
        let mut header = vec![0; length as usize];
        file.read_exact(&mut header).map_err(io)?;
        let header: Header =
            serde_json::from_slice(&header).map_err(|cause| invalid(&cause.to_string()))?;
        if header.format != "fifi-flint-expert-records-v1"
            || !(2..super::super::config::LAYERS).contains(&header.layer)
            || header.shards != shards
        {
            return Err(invalid("layout or source shards differ"));
        }
        if header
            .role_bytes
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            != Some(RECORD_BYTES)
        {
            return Err(invalid("invalid record width"));
        }
        let data_start = 8 + length;
        if file.metadata().map_err(io)?.len()
            != data_start + u64::from(EXPERTS) * RECORD_BYTES as u64
        {
            return Err(invalid("file length differs from the declared layout"));
        }
        Ok(Self {
            layer: header.layer,
            file,
            data_start,
            role_bytes: header.role_bytes,
        })
    }

    pub(super) fn read(
        &self,
        layout: &ExpertPageLayout,
        ids: &[u32],
        dst: &mut [u8],
    ) -> Result<(), EngError> {
        let roles: Vec<_> = layout
            .stacks
            .iter()
            .flat_map(|(_, stack)| {
                [&stack.payload, &stack.scales]
                    .into_iter()
                    .chain(stack.biases.as_ref())
            })
            .collect();
        if !roles.iter().map(|role| role.slice_len).eq(self.role_bytes)
            || ids.len() != 6
            || ids.iter().any(|&id| id >= EXPERTS)
        {
            return Err(invalid("page layout or selected experts differ"));
        }
        for (rank, &id) in ids.iter().enumerate() {
            let mut end = 0;
            let mut unused = &mut *dst;
            let mut views = Vec::with_capacity(self.role_bytes.len());
            for role in &roles {
                let start = role.offset + rank * role.slice_len;
                let skip = start
                    .checked_sub(end)
                    .ok_or_else(|| invalid("overlapping page ranges"))?;
                let target = unused
                    .get_mut(skip..)
                    .ok_or_else(|| invalid("page range is out of bounds"))?;
                let (slot, rest) = target
                    .split_at_mut_checked(role.slice_len)
                    .ok_or_else(|| invalid("page slot is out of bounds"))?;
                unused = rest;
                end = start + role.slice_len;
                views.push(IoSliceMut::new(slot));
            }
            let offset = self.data_start + u64::from(id) * RECORD_BYTES as u64;
            counted_read(RECORD_BYTES, || {
                read_all_vectored_at(&self.file, offset, &mut views)
            })
            .map_err(io)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
pub(in crate::family::flint) struct LayerReads {
    pub(in crate::family::flint) stages: u64,
    pub(in crate::family::flint) record_stages: u64,
    pub(in crate::family::flint) read_ns: u64,
}

thread_local! { static READS: Cell<LayerReads> = Cell::new(LayerReads::default()); }

pub(in crate::family::flint) fn layer_reads() -> LayerReads {
    READS.get()
}

pub(super) fn record_layer_read(elapsed: Duration, sidecar: bool) {
    let prior = layer_reads();
    READS.set(LayerReads {
        stages: prior.stages + 1,
        record_stages: prior.record_stages + u64::from(sidecar),
        read_ns: prior.read_ns + elapsed.as_nanos() as u64,
    });
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("Flint expert records: {message}"))
}

fn io(cause: std::io::Error) -> EngError {
    EngError::Backend {
        operation: "Flint expert records",
        message: cause.to_string(),
    }
}
