//! Per-tensor reader for a single-file GGUF checkpoint (decision 0040 item 4).
//!
//! `GgufWeights` is portable: it parses the header once, checks every tensor's
//! type up front, and then reads tensors on request. It never maps the file.
//! `read_tensor` reads one tensor at a time into a transient owned buffer, so
//! its peak host memory is the largest tensor, not the file. Device upload
//! belongs to the CUDA adapter, not to this module.
//!
//! `read_range` reads any byte range of one tensor into the caller's buffer,
//! so a caller that cannot hold a whole tensor streams it in chunks and holds
//! only one chunk at a time.

use std::{
    fs::File,
    io::{BufReader, ErrorKind},
    os::unix::fs::FileExt,
    path::Path,
};

use super::gguf::{GgufHeader, GgufTensor, GgufType};
use crate::EngError;

/// Tensor types the CUDA path loads. Q4_K, Q6_K and Q8_0 hold the matrices; F32
/// and F16 hold norms and small tables.
const SUPPORTED_TYPES: [GgufType; 5] = [
    GgufType::F32,
    GgufType::F16,
    GgufType::Q4K,
    GgufType::Q6K,
    GgufType::Q80,
];

/// Size limits checked before anything large is allocated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GgufLimits {
    /// Upper bound on the header and tensor table, passed to the header parser.
    pub(crate) max_header_bytes: u64,
    /// Upper bound on one tensor's payload, and so on one read buffer.
    pub(crate) max_tensor_bytes: u64,
}

/// An open GGUF file whose header and tensor types have been checked.
#[derive(Debug)]
pub(crate) struct GgufWeights {
    file: File,
    header: GgufHeader,
}

impl GgufWeights {
    /// Opens `path` and checks the header, the recorded header hash, the split
    /// metadata, every tensor's type, and every tensor's size.
    ///
    /// The header parser already checks that every tensor range lies inside
    /// the file. A tensor of an unsupported type, a split file, or a hash
    /// mismatch fails with `EngError::Checkpoint`. A tensor above
    /// `limits.max_tensor_bytes` fails with `EngError::MemoryPlan`.
    pub(crate) fn open(
        path: &Path,
        limits: GgufLimits,
        header_sha256: Option<&str>,
    ) -> Result<Self, EngError> {
        let file = File::open(path).map_err(|e| io_error(path, &e))?;
        let file_len = current_len(&file)?;
        let header = GgufHeader::parse(
            &mut BufReader::new(&file),
            file_len,
            limits.max_header_bytes,
        )?;
        if let Some(expected) = header_sha256 {
            header.check_header_sha256(expected)?;
        }
        if let Some(split) = header.split().filter(|split| split.count > 1) {
            return Err(checkpoint(format!(
                "split file (shard {} of {}) is not supported",
                split.no, split.count
            )));
        }
        for tensor in header.tensors() {
            check_tensor(tensor, limits.max_tensor_bytes)?;
        }
        Ok(Self { file, header })
    }

    pub(crate) fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Reads one tensor's raw payload into a new buffer of exactly its length.
    ///
    /// The file length is checked against the parsed header before and after
    /// the read, so a file replaced or truncated after `open` fails with
    /// `EngError::Checkpoint` instead of returning mixed bytes.
    pub(crate) fn read_tensor(&self, name: &str) -> Result<(GgufType, Vec<u8>), EngError> {
        let tensor = self.tensor(name)?;
        let len = usize::try_from(tensor.byte_len()).map_err(|_| {
            EngError::MemoryPlan(format!("gguf: tensor {name} does not fit in memory"))
        })?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(len).map_err(|e| {
            EngError::MemoryPlan(format!("gguf: tensor {name} needs {len} bytes: {e}"))
        })?;
        buf.resize(len, 0);
        self.read_range(name, 0, &mut buf)?;
        Ok((tensor.ggml_type(), buf))
    }

    /// Fills `buf` with the payload bytes of tensor `name` that start `offset`
    /// bytes into the tensor, so a caller can stream a tensor in chunks.
    ///
    /// The range must lie inside the tensor; one that does not (including an
    /// offset that overflows) fails with `EngError::BadRequest` before any
    /// read. An unknown name and a file length that differs from the parsed
    /// header, checked before and after the read as in [`Self::read_tensor`],
    /// fail with `EngError::Checkpoint`.
    pub(crate) fn read_range(
        &self,
        name: &str,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<(), EngError> {
        let tensor = self.tensor(name)?;
        let len = buf.len() as u64;
        let end = offset.checked_add(len);
        if end.is_none_or(|end| end > tensor.byte_len()) {
            return Err(EngError::BadRequest(format!(
                "gguf: range of {len} bytes at offset {offset} lies outside tensor {name} of {} bytes",
                tensor.byte_len()
            )));
        }
        // The header parser checked that the whole tensor lies inside the
        // file, so this sum stays below the file length.
        let at = tensor.abs_offset().checked_add(offset).ok_or_else(|| {
            checkpoint(format!("tensor {name} offset {offset} overflows the file"))
        })?;
        self.check_len()?;
        // A positioned read shares no file cursor, so concurrent callers on
        // one `GgufWeights` cannot receive each other's bytes.
        self.file
            .read_exact_at(buf, at)
            .map_err(|e| read_error(name, &e))?;
        self.check_len()
    }

    fn tensor(&self, name: &str) -> Result<&GgufTensor, EngError> {
        self.header
            .tensor(name)
            .ok_or_else(|| checkpoint(format!("tensor {name} is not in the file")))
    }

    fn check_len(&self) -> Result<(), EngError> {
        let now = current_len(&self.file)?;
        if now != self.header.file_len() {
            return Err(checkpoint(format!(
                "file length changed from {} to {now} after open",
                self.header.file_len()
            )));
        }
        Ok(())
    }
}

fn check_tensor(tensor: &GgufTensor, max_tensor_bytes: u64) -> Result<(), EngError> {
    let ty = tensor.ggml_type();
    if !SUPPORTED_TYPES.contains(&ty) {
        return Err(checkpoint(format!(
            "tensor {} has unsupported type {}",
            tensor.name(),
            ty.name()
        )));
    }
    if tensor.byte_len() > max_tensor_bytes {
        return Err(EngError::MemoryPlan(format!(
            "gguf: tensor {} has {} bytes, above the limit of {max_tensor_bytes}",
            tensor.name(),
            tensor.byte_len()
        )));
    }
    Ok(())
}

fn current_len(file: &File) -> Result<u64, EngError> {
    file.metadata()
        .map(|meta| meta.len())
        .map_err(|e| checkpoint(format!("cannot read file metadata: {e}")))
}

fn checkpoint(message: String) -> EngError {
    EngError::Checkpoint(format!("gguf: {message}"))
}

fn io_error(path: &Path, error: &std::io::Error) -> EngError {
    checkpoint(format!("cannot open {}: {error}", path.display()))
}

fn read_error(name: &str, error: &std::io::Error) -> EngError {
    if error.kind() == ErrorKind::UnexpectedEof {
        return checkpoint(format!("tensor {name} ends past the end of the file"));
    }
    checkpoint(format!("cannot read tensor {name}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Write, path::PathBuf};

    use super::*;
    use crate::weights::{dequantize, sha256_hex};

    const LIMITS: GgufLimits = GgufLimits {
        max_header_bytes: 1 << 20,
        max_tensor_bytes: 1 << 20,
    };
    const ALIGN: usize = 32;

    fn string(text: &str) -> Vec<u8> {
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out
    }

    fn kv_u16(key: &str, value: u16) -> Vec<u8> {
        let mut out = string(key);
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    /// One tensor: name, element count (one dimension), type, and payload.
    struct Entry<'a> {
        name: &'a str,
        elements: u64,
        ty: GgufType,
        payload: &'a [u8],
    }

    /// Builds a GGUF v3 file with one-dimensional tensors at aligned offsets.
    fn gguf(kvs: &[Vec<u8>], entries: &[Entry<'_>]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in kvs {
            out.extend_from_slice(kv);
        }
        let mut offset = 0usize;
        for entry in entries {
            out.extend_from_slice(&string(entry.name));
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&entry.elements.to_le_bytes());
            out.extend_from_slice(&entry.ty.id().to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset = (offset + entry.payload.len()).div_ceil(ALIGN) * ALIGN;
        }
        for entry in entries {
            out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
            out.extend_from_slice(entry.payload);
        }
        out
    }

    fn write(dir: &tempfile::TempDir, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    fn fixture(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gguf-dequant")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn elements(ty: GgufType, payload: &[u8]) -> u64 {
        payload.len() as u64 / ty.block_bytes() * ty.block_elems()
    }

    fn entry<'a>(name: &'a str, ty: GgufType, payload: &'a [u8]) -> Entry<'a> {
        Entry {
            name,
            elements: elements(ty, payload),
            ty,
            payload,
        }
    }

    fn expect_checkpoint<T: std::fmt::Debug>(result: Result<T, EngError>, expect: &str) {
        match result {
            Err(EngError::Checkpoint(message)) => {
                assert!(message.contains(expect), "{expect:?} not in {message:?}");
            }
            other => panic!("expected Checkpoint with {expect:?}, got {other:?}"),
        }
    }

    #[test]
    fn tensors_read_back_and_match_the_cpu_dequantizer_fixtures() {
        let cases = [
            ("f32", GgufType::F32),
            ("f16", GgufType::F16),
            ("q4_k", GgufType::Q4K),
            ("q6_k", GgufType::Q6K),
            ("q8_0", GgufType::Q80),
        ];
        let payloads: Vec<Vec<u8>> = cases
            .iter()
            .map(|(name, _)| fixture(&format!("{name}.blocks")))
            .collect();
        let entries: Vec<Entry<'_>> = cases
            .iter()
            .zip(&payloads)
            .map(|((name, ty), payload)| entry(name, *ty, payload))
            .collect();
        let dir = tempfile::tempdir().expect("tempdir");
        let weights =
            GgufWeights::open(&write(&dir, &gguf(&[], &entries)), LIMITS, None).expect("open");
        for ((name, ty), payload) in cases.iter().zip(&payloads) {
            let (read_ty, bytes) = weights.read_tensor(name).expect("read");
            assert_eq!(read_ty, *ty, "{name}");
            assert_eq!(&bytes, payload, "{name}");
            let values = dequantize(read_ty, &bytes).expect("dequantize");
            let want = fixture(&format!("{name}.f32"));
            let (want, rest) = want.as_chunks::<4>();
            assert!(rest.is_empty(), "{name} oracle length");
            assert_eq!(values.len(), want.len(), "{name} values");
            for (i, (got, want)) in values.iter().zip(want).enumerate() {
                let want = f32::from_le_bytes(*want);
                // x86 and arm64 default NaNs differ in sign; tests/gguf_dequant.rs
                // pins the exact bits per architecture.
                let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
                assert!(same, "{name}[{i}]: {got:?} != {want:?}");
            }
        }
    }

    #[test]
    fn unsupported_tensor_types_fail_the_open_naming_the_tensor() {
        let q5k = fixture("q5_k.blocks");
        let f32s = fixture("f32.blocks");
        let entries = [
            entry("norm", GgufType::F32, &f32s),
            entry("blk.0.ffn_down", GgufType::Q5K, &q5k),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &entries));
        expect_checkpoint(
            GgufWeights::open(&path, LIMITS, None),
            "tensor blk.0.ffn_down has unsupported type Q5_K",
        );
    }

    #[test]
    fn split_files_are_rejected() {
        let kvs = [
            kv_u16("split.no", 0),
            kv_u16("split.count", 2),
            kv_u16("split.tensors.count", 2),
        ];
        let payload = [0u8; 8];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&kvs, &[entry("a", GgufType::F32, &payload)]));
        expect_checkpoint(GgufWeights::open(&path, LIMITS, None), "split file");
    }

    #[test]
    fn single_shard_split_metadata_is_accepted() {
        let kvs = [kv_u16("split.no", 0), kv_u16("split.count", 1)];
        let payload = [0u8; 8];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&kvs, &[entry("a", GgufType::F32, &payload)]));
        GgufWeights::open(&path, LIMITS, None).expect("one shard is a whole file");
    }

    #[test]
    fn tensors_above_the_size_limit_are_memory_plan_errors() {
        let payload = [0u8; 64];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("big", GgufType::F32, &payload)]));
        let limits = GgufLimits {
            max_tensor_bytes: 63,
            ..LIMITS
        };
        match GgufWeights::open(&path, limits, None) {
            Err(EngError::MemoryPlan(message)) => {
                assert!(message.contains("tensor big has 64 bytes"), "{message}");
            }
            other => panic!("expected MemoryPlan, got {other:?}"),
        }
        let limits = GgufLimits {
            max_tensor_bytes: 64,
            ..LIMITS
        };
        GgufWeights::open(&path, limits, None).expect("a tensor at the limit fits");
    }

    #[test]
    fn the_recorded_header_hash_is_checked() {
        let payload = [0u8; 8];
        let bytes = gguf(&[], &[entry("a", GgufType::F32, &payload)]);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &bytes);
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        let recorded = sha256_hex(&weights.header().header_sha256());
        GgufWeights::open(&path, LIMITS, Some(&recorded)).expect("matching hash");
        let wrong = "0".repeat(64);
        expect_checkpoint(
            GgufWeights::open(&path, LIMITS, Some(&wrong)),
            "header sha256",
        );
    }

    #[test]
    fn tensor_ranges_past_the_file_end_fail_the_open() {
        let payload = [0u8; 64];
        let mut bytes = gguf(&[], &[entry("a", GgufType::F32, &payload)]);
        bytes.truncate(bytes.len() - 1);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &bytes);
        assert!(matches!(
            GgufWeights::open(&path, LIMITS, None),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn a_file_length_change_after_open_fails_the_read() {
        let payload = [1u8; 16];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("a", GgufType::F32, &payload)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        let mut file = OpenOptions::new().append(true).open(&path).expect("reopen");
        file.write_all(&[0]).expect("append");
        expect_checkpoint(weights.read_tensor("a"), "file length changed");
    }

    #[test]
    fn a_truncation_after_open_fails_before_the_read() {
        let payload = [1u8; 16];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("a", GgufType::F32, &payload)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        let len = std::fs::metadata(&path).expect("metadata").len();
        let file = OpenOptions::new().write(true).open(&path).expect("reopen");
        file.set_len(len - 8).expect("truncate");
        expect_checkpoint(weights.read_tensor("a"), "file length changed");
    }

    #[test]
    fn two_threads_read_their_own_tensors() {
        let first = fixture("q4_k.blocks");
        let second = fixture("q6_k.blocks");
        let entries = [
            entry("first", GgufType::Q4K, &first),
            entry("second", GgufType::Q6K, &second),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let weights =
            GgufWeights::open(&write(&dir, &gguf(&[], &entries)), LIMITS, None).expect("open");
        let read_many = |name: &str, want: &[u8]| {
            for _ in 0..200 {
                let (_, bytes) = weights.read_tensor(name).expect("read");
                assert_eq!(bytes, want, "{name}");
            }
        };
        std::thread::scope(|scope| {
            scope.spawn(|| read_many("first", &first));
            scope.spawn(|| read_many("second", &second));
        });
    }

    /// Two F32 tensors with distinct byte patterns, so a range read at the wrong
    /// tensor offset or from the wrong tensor shows up as a byte mismatch.
    fn range_fixture() -> (tempfile::TempDir, PathBuf, GgufWeights, Vec<u8>, Vec<u8>) {
        let first: Vec<u8> = (0..64u8).collect();
        let second: Vec<u8> = (0..40u8).map(|byte| 200 - byte).collect();
        let entries = [
            entry("first", GgufType::F32, &first),
            entry("second", GgufType::F32, &second),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &entries));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        (dir, path, weights, first, second)
    }

    fn expect_bad_request<T: std::fmt::Debug>(result: Result<T, EngError>, expect: &str) {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(expect), "{expect:?} not in {message:?}");
            }
            other => panic!("expected BadRequest with {expect:?}, got {other:?}"),
        }
    }

    #[test]
    fn read_range_returns_the_bytes_at_each_offset_of_each_tensor() {
        let (_dir, _path, weights, first, second) = range_fixture();
        for (name, payload) in [("first", &first), ("second", &second)] {
            for (offset, len) in [(0, payload.len()), (0, 1), (7, 13), (payload.len() - 4, 4)] {
                let mut buf = vec![0xAAu8; len];
                weights
                    .read_range(name, offset as u64, &mut buf)
                    .expect("read the range");
                assert_eq!(buf, payload[offset..offset + len], "{name} {offset}+{len}");
            }
        }
    }

    #[test]
    fn read_range_chunks_concatenate_to_the_whole_tensor() {
        let (_dir, _path, weights, first, _second) = range_fixture();
        let mut joined = Vec::new();
        for offset in (0..first.len()).step_by(24) {
            let mut buf = vec![0u8; 24.min(first.len() - offset)];
            weights
                .read_range("first", offset as u64, &mut buf)
                .expect("read a chunk");
            joined.extend_from_slice(&buf);
        }
        let (_, whole) = weights.read_tensor("first").expect("read_tensor");
        assert_eq!(joined, first);
        assert_eq!(whole, first);
    }

    #[test]
    fn read_range_accepts_an_empty_range_at_the_tensor_end() {
        let (_dir, _path, weights, first, _second) = range_fixture();
        weights
            .read_range("first", first.len() as u64, &mut [])
            .expect("an empty range at the end lies inside the tensor");
    }

    #[test]
    fn read_range_rejects_ranges_outside_the_tensor_before_reading() {
        let (_dir, _path, weights, first, second) = range_fixture();
        let mut buf = [0x55u8; 8];
        // One byte past the end of `second`.
        let past_end = second.len() as u64 - 7;
        expect_bad_request(
            weights.read_range("second", past_end, &mut buf),
            "range of 8 bytes at offset 33 lies outside tensor second of 40 bytes",
        );
        expect_bad_request(
            weights.read_range("first", first.len() as u64 + 1, &mut []),
            "outside tensor first",
        );
        expect_bad_request(
            weights.read_range("first", u64::MAX - 3, &mut buf),
            "outside tensor first",
        );
        // `first` is followed by `second` in the file, so only the tensor bound
        // rejects this range.
        let mut long = vec![0u8; first.len() + 1];
        expect_bad_request(
            weights.read_range("first", 0, &mut long),
            "outside tensor first",
        );
        assert_eq!(buf, [0x55; 8], "a rejected range wrote into the buffer");
        assert!(long.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn read_range_names_a_missing_tensor() {
        let (_dir, _path, weights, _first, _second) = range_fixture();
        expect_checkpoint(
            weights.read_range("absent", 0, &mut [0u8; 4]),
            "tensor absent is not in the file",
        );
    }

    #[test]
    fn read_range_fails_when_the_file_length_changed_after_open() {
        let (_dir, path, weights, _first, _second) = range_fixture();
        let mut file = OpenOptions::new().append(true).open(&path).expect("reopen");
        file.write_all(&[0]).expect("append");
        expect_checkpoint(
            weights.read_range("first", 0, &mut [0u8; 4]),
            "file length changed",
        );
    }

    #[test]
    fn read_range_fails_before_reading_when_the_file_shrank() {
        let (_dir, path, weights, _first, _second) = range_fixture();
        let len = std::fs::metadata(&path).expect("metadata").len();
        let file = OpenOptions::new().write(true).open(&path).expect("reopen");
        file.set_len(len - 8).expect("truncate");
        // The range lies in the cut-off tail: without the length check before
        // the read, the failure would be a short read, not a length change.
        expect_checkpoint(
            weights.read_range("second", 32, &mut [0u8; 8]),
            "file length changed",
        );
    }

    #[test]
    fn missing_tensors_are_checkpoint_errors() {
        let payload = [0u8; 8];
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, &gguf(&[], &[entry("a", GgufType::F32, &payload)]));
        let weights = GgufWeights::open(&path, LIMITS, None).expect("open");
        expect_checkpoint(weights.read_tensor("b"), "tensor b is not in the file");
    }

    #[test]
    fn a_missing_file_is_a_checkpoint_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        expect_checkpoint(
            GgufWeights::open(&dir.path().join("absent.gguf"), LIMITS, None),
            "cannot open",
        );
    }

    #[test]
    fn short_reads_report_the_tensor() {
        let error = std::io::Error::from(ErrorKind::UnexpectedEof);
        expect_checkpoint::<()>(Err(read_error("w", &error)), "tensor w ends past");
        let error = std::io::Error::other("disk");
        expect_checkpoint::<()>(Err(read_error("w", &error)), "cannot read tensor w");
    }
}
