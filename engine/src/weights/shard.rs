use std::{collections::BTreeMap, io::Read, ops::Range};

use safetensors::tensor::{Metadata, TensorInfo};

use super::header::{parse_header_json, stored_dtype};
use crate::{EngError, tensor::Dtype};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderEntry {
    dtype: Dtype,
    shape: Vec<u32>,
    offsets: Range<u64>,
}

impl HeaderEntry {
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    pub fn offsets(&self) -> &Range<u64> {
        &self.offsets
    }
}

#[derive(Debug, Clone)]
pub struct ShardHeader {
    data_start: u64,
    entries: BTreeMap<String, HeaderEntry>,
}

impl ShardHeader {
    pub fn read<R: Read + ?Sized>(
        reader: &mut R,
        byte_len: u64,
        max_header_bytes: u64,
    ) -> Result<Self, EngError> {
        if byte_len < 8 {
            return Err(EngError::Checkpoint(
                "shard is shorter than the safetensors prefix".into(),
            ));
        }
        let header_len = read_prefix(reader)?;
        if header_len > max_header_bytes {
            return Err(EngError::MemoryPlan(format!(
                "shard header length {header_len} exceeds the header budget {max_header_bytes}"
            )));
        }
        let header_len = usize::try_from(header_len).map_err(|_| {
            EngError::MemoryPlan(format!(
                "shard header length {header_len} does not fit the platform word size"
            ))
        })?;
        let data_start = (8u64)
            .checked_add(header_len as u64)
            .ok_or_else(|| EngError::Checkpoint("shard header length overflows".into()))?;
        if data_start > byte_len {
            return Err(EngError::Checkpoint(format!(
                "shard header length {header_len} exceeds the shard byte length {byte_len}"
            )));
        }
        let json = read_header(reader, header_len)?;
        let value = parse_header_json(&json, "shard header")?;
        let entries = parse_entries(&value)?;
        from_entries(entries, data_start, byte_len)
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn entry(&self, key: &str) -> Option<&HeaderEntry> {
        self.entries.get(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &HeaderEntry)> {
        self.entries
            .iter()
            .map(|(key, entry)| (key.as_str(), entry))
    }
}

fn read_prefix<R: Read + ?Sized>(reader: &mut R) -> Result<u64, EngError> {
    let mut prefix = [0u8; 8];
    reader
        .read_exact(&mut prefix)
        .map_err(|e| EngError::Checkpoint(format!("shard prefix read: {e}")))?;
    Ok(u64::from_le_bytes(prefix))
}

fn read_header<R: Read + ?Sized>(reader: &mut R, header_len: usize) -> Result<Vec<u8>, EngError> {
    let mut json = vec![0u8; header_len];
    reader
        .read_exact(&mut json)
        .map_err(|e| EngError::Checkpoint(format!("shard header read: {e}")))?;
    Ok(json)
}

fn parse_entries(value: &serde_json::Value) -> Result<Vec<(String, TensorInfo)>, EngError> {
    let object = value
        .as_object()
        .ok_or_else(|| EngError::Checkpoint("shard header is not a JSON object".into()))?;
    let mut entries = Vec::with_capacity(object.len());
    for (name, raw) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = serde_json::from_value::<TensorInfo>(raw.clone())
            .map_err(|e| EngError::Checkpoint(format!("shard header entry {name:?}: {e}")))?;
        entries.push((name.clone(), info));
    }
    entries.sort_by_key(|(_, info)| info.data_offsets);
    Ok(entries)
}

fn from_entries(
    entries: Vec<(String, TensorInfo)>,
    data_start: u64,
    byte_len: u64,
) -> Result<ShardHeader, EngError> {
    let metadata = Metadata::new(None, entries.clone())
        .map_err(|e| EngError::Checkpoint(format!("shard header validation failed: {e}")))?;
    let data_len = u64::try_from(metadata.data_len())
        .map_err(|_| EngError::MemoryPlan("shard data length does not fit u64".into()))?;
    let data_end = data_start
        .checked_add(data_len)
        .ok_or_else(|| EngError::Checkpoint("shard data section overflows".into()))?;
    if data_end != byte_len {
        return Err(EngError::Checkpoint(format!(
            "shard byte length {byte_len} does not match the data section end {data_end}"
        )));
    }
    let mut map = BTreeMap::new();
    for (name, info) in &entries {
        let converted = convert_entry(name, info)?;
        map.insert(name.clone(), converted);
    }
    Ok(ShardHeader {
        data_start,
        entries: map,
    })
}

fn convert_entry(name: &str, info: &TensorInfo) -> Result<HeaderEntry, EngError> {
    let dtype = stored_dtype(info.dtype).ok_or_else(|| {
        EngError::Checkpoint(format!(
            "shard header entry {name:?} has unsupported dtype {:?}",
            info.dtype
        ))
    })?;
    let shape = info
        .shape
        .iter()
        .map(|&dim| u32::try_from(dim))
        .collect::<Result<Vec<u32>, _>>()
        .map_err(|_| {
            EngError::Checkpoint(format!(
                "shard header entry {name:?} has a dimension beyond u32"
            ))
        })?;
    let start = u64::try_from(info.data_offsets.0)
        .map_err(|_| EngError::MemoryPlan("shard offset does not fit u64".into()))?;
    let end = u64::try_from(info.data_offsets.1)
        .map_err(|_| EngError::MemoryPlan("shard offset does not fit u64".into()))?;
    Ok(HeaderEntry {
        dtype,
        shape,
        offsets: start..end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{self, Cursor};

    struct Entry<'a> {
        name: &'a str,
        dtype: &'a str,
        shape: &'a [usize],
        offsets: (usize, usize),
    }

    fn entry<'a>(
        name: &'a str,
        dtype: &'a str,
        shape: &'a [usize],
        offsets: (usize, usize),
    ) -> Entry<'a> {
        Entry {
            name,
            dtype,
            shape,
            offsets,
        }
    }

    fn header_value(entries: &[Entry], metadata: bool) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if metadata {
            map.insert("__metadata__".into(), json!({"format": "pt"}));
        }
        for e in entries {
            map.insert(
                e.name.into(),
                json!({
                    "dtype": e.dtype,
                    "shape": e.shape,
                    "data_offsets": e.offsets,
                }),
            );
        }
        serde_json::Value::Object(map)
    }

    fn shard_bytes(header: &serde_json::Value, payload: usize) -> Vec<u8> {
        let json = serde_json::to_string(header).unwrap();
        let n = json.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(8 + n + payload, 0x5a);
        bytes
    }

    fn built(entries: &[Entry]) -> Vec<u8> {
        shard_bytes(&header_value(entries, false), payload_len(entries))
    }

    fn payload_len(entries: &[Entry]) -> usize {
        entries.iter().map(|e| e.offsets.1).max().unwrap_or(0)
    }

    struct HeaderOnly<'a> {
        data: &'a [u8],
        pos: usize,
        limit: usize,
    }

    impl Read for HeaderOnly<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos + buf.len() > self.limit {
                return Err(io::Error::other("payload read attempted"));
            }
            let count = buf.len().min(self.data.len() - self.pos);
            buf[..count].copy_from_slice(&self.data[self.pos..self.pos + count]);
            self.pos += count;
            Ok(count)
        }
    }

    fn header_only(data: &[u8], limit: usize) -> HeaderOnly<'_> {
        HeaderOnly {
            data,
            pos: 0,
            limit,
        }
    }

    #[test]
    fn reads_entries_with_checked_ranges() {
        let bytes = shard_bytes(
            &header_value(
                &[
                    entry("a", "F32", &[2, 2], (0, 16)),
                    entry("b", "BF16", &[3], (16, 22)),
                    entry("c", "U32", &[2, 1, 2], (22, 38)),
                ],
                true,
            ),
            38,
        );
        let header =
            ShardHeader::read(&mut Cursor::new(bytes.clone()), bytes.len() as u64, 1 << 20)
                .unwrap();
        assert_eq!(header.iter().count(), 3);
        let expected_start = 8 + (bytes.len() - 8 - 38);
        assert_eq!(header.data_start(), expected_start as u64);
        assert_eq!(header.data_start() as usize + 38, bytes.len());
        let a = header.entry("a").unwrap();
        assert_eq!(a.dtype(), Dtype::F32);
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.offsets(), &(0..16));
        let b = header.entry("b").unwrap();
        assert_eq!(b.dtype(), Dtype::Bf16);
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.offsets(), &(16..22));
        let c = header.entry("c").unwrap();
        assert_eq!(c.dtype(), Dtype::U32);
        assert_eq!(c.shape(), &[2, 1, 2]);
        assert_eq!(c.offsets(), &(22..38));
        let abs = header.data_start() + 16..header.data_start() + 22;
        assert_eq!(b.offsets().start + header.data_start(), abs.start);
        assert_eq!(b.offsets().end + header.data_start(), abs.end);
        assert!(header.entry("__metadata__").is_none());
        assert!(header.entry("missing").is_none());
    }

    #[test]
    fn header_only_never_reads_payload() {
        let bytes = built(&[
            entry("a", "F32", &[2, 2], (0, 16)),
            entry("b", "U8", &[4], (16, 20)),
        ]);
        let limit = bytes.len() - 20;
        let mut reader = header_only(&bytes, limit);
        let header = ShardHeader::read(&mut reader, bytes.len() as u64, 1 << 20).unwrap();
        assert_eq!(header.iter().count(), 2);
        assert_eq!(header.data_start() as usize, limit);
    }

    #[test]
    fn rejects_files_shorter_than_the_prefix() {
        let err = ShardHeader::read(&mut Cursor::new(Vec::new()), 4, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint("shard is shorter than the safetensors prefix".into())
        );
        let mut short = Cursor::new(vec![0u8; 4]);
        let err = ShardHeader::read(&mut short, 8, 1 << 20).unwrap_err();
        assert!(
            matches!(err, EngError::Checkpoint(message) if message.starts_with("shard prefix read: "))
        );
    }

    #[test]
    fn rejects_headers_over_the_budget() {
        let bytes = built(&[entry("a", "F32", &[2], (0, 8))]);
        let n = (bytes.len() - 8) as u64 - 8;
        let err = ShardHeader::read(&mut Cursor::new(bytes), 8 + n + 8, n - 1).unwrap_err();
        assert_eq!(
            err,
            EngError::MemoryPlan(format!(
                "shard header length {n} exceeds the header budget {}",
                n - 1
            ))
        );
    }

    #[test]
    fn rejects_headers_longer_than_the_file() {
        let bytes = built(&[entry("a", "F32", &[2], (0, 8))]);
        let n = (bytes.len() - 8 - 8) as u64;
        let err = ShardHeader::read(&mut Cursor::new(bytes), 8 + n - 1, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint(format!(
                "shard header length {n} exceeds the shard byte length {}",
                8 + n - 1
            ))
        );
    }

    #[test]
    fn rejects_duplicate_json_keys() {
        let json = format!(
            "{{\"a\":{},\"a\":{}}}",
            json!({"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}),
            json!({"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}),
        );
        let n = json.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(8 + n + 8, 0);
        let err = ShardHeader::read(&mut Cursor::new(bytes), (16 + n) as u64, 1 << 20).unwrap_err();
        assert!(
            matches!(err, EngError::Checkpoint(message) if message.contains("duplicate JSON member"))
        );
    }

    #[test]
    fn rejects_non_contiguous_offsets() {
        let bytes = built(&[
            entry("a", "U8", &[4], (0, 4)),
            entry("b", "U8", &[4], (8, 12)),
        ]);
        let byte_len = bytes.len() as u64;
        let err = ShardHeader::read(&mut Cursor::new(bytes), byte_len, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint(
                "shard header validation failed: invalid offset for tensor `b`".into()
            )
        );
    }

    #[test]
    fn rejects_byte_length_mismatches() {
        let bytes = built(&[entry("a", "U8", &[4], (0, 4))]);
        let end = bytes.len() as u64;
        let err = ShardHeader::read(&mut Cursor::new(bytes.clone()), end + 1, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint(format!(
                "shard byte length {} does not match the data section end {end}",
                end + 1
            ))
        );
        let err = ShardHeader::read(&mut Cursor::new(bytes), end - 1, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint(format!(
                "shard byte length {} does not match the data section end {end}",
                end - 1
            ))
        );
    }

    #[test]
    fn rejects_unsupported_dtypes() {
        let bytes = built(&[
            entry("a", "U8", &[4], (0, 4)),
            entry("c", "I8", &[1], (4, 5)),
        ]);
        let byte_len = bytes.len() as u64;
        let err = ShardHeader::read(&mut Cursor::new(bytes), byte_len, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint("shard header entry \"c\" has unsupported dtype I8".into())
        );
    }

    #[test]
    fn rejects_non_object_headers() {
        let json = "[1,2]";
        let n = json.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        let err = ShardHeader::read(&mut Cursor::new(bytes), 8 + n as u64, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint("shard header is not a JSON object".into())
        );
    }

    fn shard_with_json(json: &str, payload: usize) -> Vec<u8> {
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(8 + json.len() + payload, 0);
        bytes
    }

    #[test]
    fn rejects_metadata_that_is_not_a_string_map() {
        let entry = r#""a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}"#;
        for metadata in [r#"{"format":1}"#, r#"["pt"]"#, r#""pt""#, "7"] {
            let json = format!(r#"{{"__metadata__":{metadata},{entry}}}"#);
            let bytes = shard_with_json(&json, 4);
            let byte_len = bytes.len() as u64;
            let err = ShardHeader::read(&mut Cursor::new(bytes), byte_len, 1 << 20).unwrap_err();
            assert_eq!(
                err,
                EngError::Checkpoint("shard header metadata is not a string map".into()),
                "{metadata}"
            );
        }
    }

    #[test]
    fn accepts_null_metadata_as_absent() {
        let json = r#"{"__metadata__":null,"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let bytes = shard_with_json(json, 4);
        let byte_len = bytes.len() as u64;
        let header = ShardHeader::read(&mut Cursor::new(bytes), byte_len, 1 << 20).unwrap();
        assert_eq!(header.iter().count(), 1);
        assert_eq!(header.entry("a").unwrap().offsets(), &(0..4));
    }

    #[test]
    fn rejects_dimensions_beyond_u32() {
        let json = r#"{"wide":{"dtype":"U8","shape":[4294967296,0],"data_offsets":[0,0]}}"#;
        let bytes = shard_with_json(json, 0);
        let byte_len = bytes.len() as u64;
        let err = ShardHeader::read(&mut Cursor::new(bytes), byte_len, 1 << 20).unwrap_err();
        assert_eq!(
            err,
            EngError::Checkpoint("shard header entry \"wide\" has a dimension beyond u32".into())
        );
    }
}
