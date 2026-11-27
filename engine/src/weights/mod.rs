mod acquire;
pub(crate) mod dspark_gguf;
#[cfg(all(
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) use acquire::read_all_vectored_at;
pub mod exl3;

// No caller yet; kept compiling and tested standalone.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod expert_store;
mod gguf;
pub(crate) mod gguf_dequant;
mod gguf_encode;
pub(crate) mod gguf_experts;
mod gguf_split;
mod gguf_tables;
/// The lookup tables of the lattice and non-linear GGUF types, re-exported for the CUDA
/// matvec, which passes them to its kernels as launch arguments, and for the table pin test
/// of the IQ4_NL header (kernels/cuda/resident/matvec_q/tests.rs).
#[cfg(feature = "native-cuda")]
pub(crate) use gguf_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL,
};
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) use gguf_tables::{IQ1S_GRID_U16, inkling_codebook_words};
/// The same lattice tables and the packed IQ1_S grid for the Metal expert path, which
/// uploads them once per model.
#[cfg(all(
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64",
    not(feature = "native-cuda")
))]
pub(crate) use gguf_tables::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID};
#[cfg(all(test, feature = "native-cuda"))]
pub(crate) use gguf_tables::{KSIGNS_IQ2XS, KVALUES_FP4};
mod gguf_identity;
mod gguf_weights;
mod header;
mod indexed;
pub mod nemotron_index;
pub mod nemotron_shards;
mod pack_digest;
pub(crate) use pack_digest::{Hidden, pack_digest};
pub mod page_cache;
mod shard;
mod tiered;

pub use acquire::{
    AcquireRequest, CancelHandle, Completion, Priority, Selection, StreamedCheckpoint,
    StreamedWeights, Ticket,
};
/// The cold-load read accounting for the load-timing harness: acquire's `read_stats` module
/// under a stable name, Mac test builds only.
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) mod acquire_read {
    pub(crate) use super::acquire::{reset, set, snapshot};
}
pub(crate) use gguf::shared_header;
pub use gguf::{
    ArrayMeta, GgufHeader, GgufTensor, GgufType, MetaValue, SMALL_VECTOR_BYTES, SplitInfo,
    check_split_set, hash_sampled_edges, sha256_hex, sha256_stream, verify_sha256,
};
pub(crate) use gguf_dequant::project_f32_rows;
pub use gguf_dequant::{
    dequantize, dequantize_into, dequantize_nvfp4_expert_into, repack_nvfp4_into,
    widen_to_f32_bytes,
};
pub(crate) use gguf_dequant::{dot_q6_k, dot_q8_0};
pub use gguf_encode::{encode_plain_float_rows, encode_q8_0, is_plain_float, q8_0_len};
pub use gguf_experts::{
    ExpertSlice, ExpertSlices, ExpertTensorNames, GgufExpertIndex, LocatorError,
};
pub use gguf_identity::{TENSOR_SAMPLE, gguf_identity_digest};
#[cfg(test)]
pub(crate) use gguf_split::write_two_shard_set;
pub(crate) use gguf_split::{ShardName, shard_name_of};
pub(crate) use gguf_weights::{GgufLimits, GgufWeights};
pub use indexed::IndexedWeights;
pub(crate) use shard::read_stored_header;
pub use shard::{HeaderEntry, ShardHeader};
pub use tiered::{LoadLimits, ShardIo, TieredWeights};

use std::{collections::HashMap, io::Read, ops::Range, sync::Arc};

use safetensors::SafeTensors;

use self::header::{stored_dtype, validate_header_json};
use crate::{
    EngError,
    tensor::{Backing, Tensor},
};

/// Length of the little-endian header size that starts every safetensors file.
const PREFIX_LEN: usize = 8;

pub trait WeightSource: Send + Sync {
    fn tensor(&self, key: &str) -> Option<Tensor>;
    fn resident_bytes(&self) -> u64;
    fn tier(&self) -> Tier;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Resident,
    Ssd { budget: u64 },
}

#[derive(Debug)]
pub struct ResidentWeights {
    tensors: HashMap<String, Tensor>,
    resident_bytes: u64,
}

impl ResidentWeights {
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, EngError> {
        Self::from_backing(Backing::Bytes(bytes))
    }

    pub fn from_reader<R: Read + ?Sized>(
        reader: &mut R,
        byte_len: u64,
        max_input_bytes: u64,
    ) -> Result<Self, EngError> {
        if max_input_bytes == 0 {
            return Err(EngError::BadRequest("input budget must be nonzero".into()));
        }
        if byte_len < 8 {
            return Err(EngError::Checkpoint(
                "input is shorter than its length header".into(),
            ));
        }
        if byte_len > max_input_bytes || byte_len > isize::MAX as u64 {
            return Err(EngError::MemoryPlan(
                "input exceeds its budget or host slice limit".into(),
            ));
        }
        let length = usize::try_from(byte_len)
            .map_err(|_| EngError::MemoryPlan("input exceeds host limits".into()))?;
        let mut mapped = memmap2::MmapOptions::new()
            .len(length)
            .map_anon()
            .map_err(|error| EngError::Backend {
                operation: "host_map_create",
                message: error.to_string(),
            })?;
        reader
            .read_exact(&mut mapped)
            .map_err(|error| EngError::Checkpoint(format!("checkpoint read: {error}")))?;
        let mut extra = [0];
        loop {
            match reader.read(&mut extra) {
                Ok(0) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Ok(_) => return Err(EngError::Checkpoint("checkpoint has trailing input".into())),
                Err(error) => {
                    return Err(EngError::Checkpoint(format!(
                        "checkpoint end read: {error}"
                    )));
                }
            }
        }
        let mapped = mapped.make_read_only().map_err(|error| EngError::Backend {
            operation: "host_map_read_only",
            message: error.to_string(),
        })?;
        Self::from_backing(Backing::Mapped(Arc::new(mapped)))
    }

    fn from_backing(bytes: Backing) -> Result<Self, EngError> {
        check_header_json(&bytes)?;
        let (header_len, metadata) = SafeTensors::read_metadata(&bytes)
            .map_err(|error| EngError::Checkpoint(format!("safetensors buffer: {error}")))?;
        let data_start = header_len
            .checked_add(PREFIX_LEN)
            .ok_or_else(|| EngError::Checkpoint("safetensors data offset overflow".into()))?;
        let mut tensors = HashMap::new();
        let mut resident_bytes = 0u64;
        for (name, info) in metadata.tensors() {
            let dtype = stored_dtype(info.dtype).ok_or_else(|| {
                EngError::Checkpoint(format!("tensor {name:?}: unsupported dtype {}", info.dtype))
            })?;
            let shape = info
                .shape
                .iter()
                .map(|&dim| {
                    u32::try_from(dim).map_err(|_| {
                        EngError::Checkpoint(format!("tensor {name:?}: dimension exceeds u32"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let range = tensor_range(&name, data_start, info.data_offsets)?;
            let tensor = Tensor::from_shared(dtype, shape, bytes.clone(), range)
                .map_err(|error| tensor_error(&name, error))?;
            resident_bytes = resident_bytes
                .checked_add(tensor.byte_len())
                .ok_or_else(|| {
                    EngError::Checkpoint("stored payload byte count overflows u64".into())
                })?;
            tensors.insert(name, tensor);
        }
        Ok(Self {
            tensors,
            resident_bytes,
        })
    }
}

/// Longest header that `read_metadata` accepts. A longer header is left to its own error.
///
/// This value mirrors `MAX_HEADER_SIZE` at `engine/vendor/safetensors/src/tensor.rs:10`.
/// Change it together with the vendored crate.
const MAX_HEADER_BYTES: u64 = 100_000_000;

fn header_bytes(bytes: &[u8]) -> Option<&[u8]> {
    let prefix: [u8; PREFIX_LEN] = bytes.get(..PREFIX_LEN)?.try_into().ok()?;
    let length = u64::from_le_bytes(prefix);
    if length > MAX_HEADER_BYTES {
        return None;
    }
    let end = PREFIX_LEN.checked_add(usize::try_from(length).ok()?)?;
    bytes.get(PREFIX_LEN..end)
}

/// Applies the shared header rules before the crate parses the header.
///
/// `read_metadata` parses into a map, keeps the last duplicate member, and reports its own
/// text for other defects. Running the shared parser first gives every load path the same
/// documented text for a repeated member name and for bad `__metadata__`.
/// A prefix that does not fit the buffer is left to `read_metadata`.
fn check_header_json(bytes: &[u8]) -> Result<(), EngError> {
    match header_bytes(bytes) {
        Some(header) => validate_header_json(header, "safetensors header"),
        None => Ok(()),
    }
}

fn tensor_range(
    name: &str,
    data_start: usize,
    offsets: (usize, usize),
) -> Result<Range<usize>, EngError> {
    let start = data_start
        .checked_add(offsets.0)
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {name:?}: start offset overflow")))?;
    let end = data_start
        .checked_add(offsets.1)
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {name:?}: end offset overflow")))?;
    Ok(start..end)
}

/// Adds the tensor name to a storage error and keeps the error variant.
///
/// This branch is defensive. `read_metadata` already checks each shape against its byte range
/// and each range against the buffer, so `Tensor::from_shared` cannot fail after it.
/// The unit tests beside this module cover the branch, because no public input reaches it.
fn tensor_error(name: &str, error: EngError) -> EngError {
    match error {
        EngError::Checkpoint(message) => {
            EngError::Checkpoint(format!("tensor {name:?}: {message}"))
        }
        other => other,
    }
}

impl WeightSource for ResidentWeights {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn tier(&self) -> Tier {
        Tier::Resident
    }

    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.tensors.get(key).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_message(error: EngError) -> String {
        match error {
            EngError::Checkpoint(message) => message,
            other => panic!("expected a checkpoint error, found {other:?}"),
        }
    }

    #[test]
    fn a_tensor_range_adds_the_data_start_to_both_offsets() {
        assert_eq!(tensor_range("t", 10, (2, 6)).unwrap(), 12..16);
        assert_eq!(
            tensor_range("t", usize::MAX, (0, 0)).unwrap(),
            usize::MAX..usize::MAX
        );
    }

    #[test]
    fn tensor_range_overflow_names_the_tensor_and_the_offset() {
        let message = checkpoint_message(tensor_range("wide", usize::MAX, (1, 2)).unwrap_err());
        assert_eq!(message, "tensor \"wide\": start offset overflow");
        let message = checkpoint_message(tensor_range("wide", usize::MAX, (0, 1)).unwrap_err());
        assert_eq!(message, "tensor \"wide\": end offset overflow");
    }

    #[test]
    fn a_storage_error_keeps_its_message_without_the_debug_form() {
        let error = EngError::Checkpoint("stored shape requires 8 bytes, received 4".into());
        let message = checkpoint_message(tensor_error("x", error));
        assert_eq!(
            message,
            "tensor \"x\": stored shape requires 8 bytes, received 4"
        );
        assert!(!message.contains("Checkpoint("), "{message}");
    }

    #[test]
    fn a_storage_error_of_another_kind_passes_through_unchanged() {
        let error = EngError::MemoryPlan("over budget".into());
        assert_eq!(
            tensor_error("x", error),
            EngError::MemoryPlan("over budget".into())
        );
    }

    fn file(header: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn header_bytes_returns_exactly_the_declared_header() {
        let bytes = file(b"{}", b"payload");
        assert_eq!(header_bytes(&bytes), Some(b"{}".as_slice()));
        assert_eq!(header_bytes(&file(b"", b"x")), Some(b"".as_slice()));
    }

    #[test]
    fn header_bytes_rejects_prefixes_that_do_not_fit() {
        assert_eq!(header_bytes(&[0; 7]), None);
        let mut short = file(b"{}", b"");
        short.pop();
        assert_eq!(header_bytes(&short), None);
        let over_limit = (MAX_HEADER_BYTES + 1).to_le_bytes();
        assert_eq!(header_bytes(&over_limit), None);
        assert_eq!(header_bytes(&u64::MAX.to_le_bytes()), None);
    }

    #[test]
    fn a_prefix_that_does_not_fit_is_left_to_the_crate() {
        assert_eq!(check_header_json(&[0; 7]), Ok(()));
        assert_eq!(check_header_json(&u64::MAX.to_le_bytes()), Ok(()));
    }

    #[test]
    fn the_header_check_reads_only_the_header_bytes() {
        let bytes = file(br#"{"a":{"b":1}}"#, b"\xff\xff payload is not JSON");
        assert_eq!(check_header_json(&bytes), Ok(()));
        let bad = file(br#"{"a":1,"a":2}"#, b"");
        let message = checkpoint_message(check_header_json(&bad).unwrap_err());
        assert!(
            message
                .starts_with("safetensors header is not valid JSON: duplicate JSON member \"a\""),
            "{message}"
        );
    }
}
