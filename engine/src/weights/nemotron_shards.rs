use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::Path,
};

use safetensors::{
    Dtype as StoredDtype,
    tensor::{Metadata, TensorInfo},
};
use serde_json::Value;

use super::header::parse_header_json;
use crate::EngError;

const PREFIX_BYTES: u64 = 8;
const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronShardTensor {
    dtype: &'static str,
    shape: Vec<u64>,
    offsets: Range<u64>,
    byte_len: u64,
}

impl NemotronShardTensor {
    pub fn dtype(&self) -> &'static str {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn offsets(&self) -> &Range<u64> {
        &self.offsets
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

#[derive(Debug)]
pub struct NemotronShardReader {
    file: File,
    data_start: u64,
    file_len: u64,
    tensors: BTreeMap<String, NemotronShardTensor>,
}

impl NemotronShardReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, EngError> {
        let mut file = File::open(path.as_ref())
            .map_err(|error| EngError::Checkpoint(format!("open shard: {error}")))?;
        let file_len = file
            .metadata()
            .map_err(|error| EngError::Checkpoint(format!("read shard length: {error}")))?
            .len();
        if file_len < PREFIX_BYTES {
            return Err(EngError::Checkpoint(
                "shard is shorter than the safetensors prefix".into(),
            ));
        }

        let mut prefix = [0u8; PREFIX_BYTES as usize];
        file.read_exact(&mut prefix)
            .map_err(|error| EngError::Checkpoint(format!("read shard prefix: {error}")))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(EngError::Checkpoint(format!(
                "shard header length {header_len} exceeds the limit {MAX_HEADER_BYTES}"
            )));
        }
        let data_start = PREFIX_BYTES
            .checked_add(header_len)
            .ok_or_else(|| EngError::Checkpoint("shard header length overflows".into()))?;
        if data_start > file_len {
            return Err(EngError::Checkpoint(format!(
                "shard header ends at {data_start}, beyond file length {file_len}"
            )));
        }
        let header_size = usize::try_from(header_len)
            .map_err(|_| EngError::Checkpoint("shard header does not fit host size".into()))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_size)
            .map_err(|error| EngError::MemoryPlan(format!("allocate shard header: {error}")))?;
        header.resize(header_size, 0);
        file.read_exact(&mut header)
            .map_err(|error| EngError::Checkpoint(format!("read shard header: {error}")))?;

        let value = parse_header_json(&header, "Nemotron shard header")?;
        let tensors = parse_tensors(&value, data_start, file_len)?;
        Ok(Self {
            file,
            data_start,
            file_len,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&NemotronShardTensor> {
        self.tensors.get(name)
    }

    /// Length of the shard file; `open` proved the header-declared data section ends exactly here.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Absolute byte range of one tensor payload inside the shard file, for a caller that maps the
    /// file instead of copying the tensor out.
    pub fn absolute_range(&self, name: &str) -> Option<Range<usize>> {
        let tensor = self.tensors.get(name)?;
        let start = usize::try_from(self.data_start.checked_add(tensor.offsets.start)?).ok()?;
        let end = usize::try_from(self.data_start.checked_add(tensor.offsets.end)?).ok()?;
        Some(start..end)
    }

    /// Evicts the clean page-cache pages of the given absolute byte ranges of the shard file, for
    /// bytes whose only remaining home should be somewhere else (the device).
    pub fn drop_cache(&self, ranges: &[Range<usize>]) -> Result<(), EngError> {
        for range in ranges {
            let length = range.end.saturating_sub(range.start);
            super::page_cache::drop_file_range(&self.file, range.start as u64, length as u64)
                .map_err(|error| EngError::Backend {
                    operation: "nemotron_shard_drop_cache",
                    message: format!("posix_fadvise: {error}"),
                })?;
        }
        Ok(())
    }

    /// Reads `length` bytes starting `offset` bytes into one tensor payload.
    pub fn read_tensor_range(
        &mut self,
        name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, EngError> {
        let tensor = self.tensors.get(name).ok_or_else(|| {
            EngError::Checkpoint(format!("Nemotron shard has no tensor {name:?}"))
        })?;
        offset
            .checked_add(length as u64)
            .filter(|&end| end <= tensor.byte_len)
            .ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "range {offset}+{length} exceeds tensor {name:?} of {} bytes",
                    tensor.byte_len
                ))
            })?;
        let start = self
            .data_start
            .checked_add(tensor.offsets.start)
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(|| EngError::Checkpoint("tensor file offset overflows".into()))?;
        let mut bytes = vec![0u8; length];
        self.file
            .seek(SeekFrom::Start(start))
            .map_err(|error| EngError::Checkpoint(format!("seek to tensor {name:?}: {error}")))?;
        self.file.read_exact(&mut bytes).map_err(|error| {
            EngError::Checkpoint(format!("read tensor range of {name:?}: {error}"))
        })?;
        Ok(bytes)
    }

    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>, EngError> {
        let tensor = self.tensors.get(name).ok_or_else(|| {
            EngError::Checkpoint(format!("Nemotron shard has no tensor {name:?}"))
        })?;
        let start = self
            .data_start
            .checked_add(tensor.offsets.start)
            .ok_or_else(|| EngError::Checkpoint("tensor file offset overflows".into()))?;
        let end = self
            .data_start
            .checked_add(tensor.offsets.end)
            .ok_or_else(|| EngError::Checkpoint("tensor file end offset overflows".into()))?;
        let file_len = self
            .file
            .metadata()
            .map_err(|error| EngError::Checkpoint(format!("read shard length: {error}")))?
            .len();
        if end > file_len {
            return Err(EngError::Checkpoint(format!(
                "tensor {name:?} ends at {end}, beyond file length {file_len}"
            )));
        }

        let size = usize::try_from(tensor.byte_len)
            .map_err(|_| EngError::MemoryPlan(format!("tensor {name:?} is too large to read")))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|error| EngError::MemoryPlan(format!("allocate tensor {name:?}: {error}")))?;
        bytes.resize(size, 0);
        self.file
            .seek(SeekFrom::Start(start))
            .map_err(|error| EngError::Checkpoint(format!("seek to tensor {name:?}: {error}")))?;
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| EngError::Checkpoint(format!("read tensor {name:?}: {error}")))?;
        Ok(bytes)
    }
}

fn parse_tensors(
    value: &Value,
    data_start: u64,
    file_len: u64,
) -> Result<BTreeMap<String, NemotronShardTensor>, EngError> {
    let object = value
        .as_object()
        .ok_or_else(|| EngError::Checkpoint("Nemotron shard header is not an object".into()))?;
    let mut entries = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = serde_json::from_value::<TensorInfo>(value.clone()).map_err(|error| {
            EngError::Checkpoint(format!("shard tensor {name:?} metadata: {error}"))
        })?;
        let (dtype, width) = dtype_info(info.dtype).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "shard tensor {name:?} has unsupported dtype {:?}",
                info.dtype
            ))
        })?;
        let shape = info
            .shape
            .iter()
            .map(|&dimension| {
                u64::try_from(dimension).map_err(|_| {
                    EngError::Checkpoint(format!(
                        "shard tensor {name:?} dimension does not fit u64"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let start = u64::try_from(info.data_offsets.0).map_err(|_| {
            EngError::Checkpoint(format!(
                "shard tensor {name:?} start offset does not fit u64"
            ))
        })?;
        let end = u64::try_from(info.data_offsets.1).map_err(|_| {
            EngError::Checkpoint(format!("shard tensor {name:?} end offset does not fit u64"))
        })?;
        let range_len = end.checked_sub(start).ok_or_else(|| {
            EngError::Checkpoint(format!("shard tensor {name:?} has reversed data_offsets"))
        })?;
        let elements = shape.iter().try_fold(1u64, |count, &dimension| {
            count.checked_mul(dimension).ok_or_else(|| {
                EngError::Checkpoint(format!("shard tensor {name:?} shape size overflows"))
            })
        })?;
        let expected_len = elements.checked_mul(width).ok_or_else(|| {
            EngError::Checkpoint(format!("shard tensor {name:?} byte size overflows"))
        })?;
        if range_len != expected_len {
            return Err(EngError::Checkpoint(format!(
                "shard tensor {name:?} range is {range_len} bytes, shape and dtype require {expected_len}"
            )));
        }
        let absolute_end = data_start.checked_add(end).ok_or_else(|| {
            EngError::Checkpoint(format!("shard tensor {name:?} file offset overflows"))
        })?;
        if absolute_end > file_len {
            return Err(EngError::Checkpoint(format!(
                "shard tensor {name:?} ends at {absolute_end}, beyond file length {file_len}"
            )));
        }
        entries.push((
            name.clone(),
            info,
            NemotronShardTensor {
                dtype,
                shape,
                offsets: start..end,
                byte_len: expected_len,
            },
        ));
    }

    let mut metadata_entries = entries
        .iter()
        .map(|(name, info, _)| (name.clone(), info.clone()))
        .collect::<Vec<_>>();
    metadata_entries.sort_by_key(|(_, info)| info.data_offsets);
    let metadata = Metadata::new(None, metadata_entries)
        .map_err(|error| EngError::Checkpoint(format!("invalid safetensors metadata: {error}")))?;
    let data_len = u64::try_from(metadata.data_len())
        .map_err(|_| EngError::Checkpoint("shard data length does not fit u64".into()))?;
    let data_end = data_start
        .checked_add(data_len)
        .ok_or_else(|| EngError::Checkpoint("shard data section end overflows".into()))?;
    if data_end != file_len {
        return Err(EngError::Checkpoint(format!(
            "shard data section ends at {data_end}, but file length is {file_len}"
        )));
    }
    Ok(entries
        .into_iter()
        .map(|(name, _, tensor)| (name, tensor))
        .collect())
}

fn dtype_info(dtype: StoredDtype) -> Option<(&'static str, u64)> {
    match dtype {
        StoredDtype::BF16 => Some(("BF16", 2)),
        StoredDtype::F32 => Some(("F32", 4)),
        StoredDtype::F8_E4M3 => Some(("F8_E4M3", 1)),
        StoredDtype::U8 => Some(("U8", 1)),
        StoredDtype::I64 => Some(("I64", 8)),
        _ => None,
    }
}
