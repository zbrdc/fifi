use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use inference_engine::weights::{GgufHeader, GgufType, MetaValue, check_split_set, sha256_hex};

const HEADER_LIMIT: u64 = 64 << 20;
const SHARD1_LENGTH: u64 = 47_039_860_096;
const SHARD1_HEADER_SHA256: &str =
    "37fd1796405af7d5394736fbf9b01125b36758506473143ecc840ff47a9191bb";
const SHARD2_LENGTH: u64 = 28_800_138_432;
const SHARD2_HEADER_SHA256: &str =
    "49f992d94de7d687ab0042f54d72fbabb4fec16e6860181f1b5bd91e46c89a96";
const TABLE_TENSOR: &str = "per_layer_token_embd.weight";

struct Shard {
    path: String,
    header: GgufHeader,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "metadata" {
        return Err(format!(
            "usage: {} metadata <mac-g1-prefix-directory>",
            args.first().map_or("ngram-trace", String::as_str)
        ));
    }
    let prefix_dir = Path::new(&args[2]);
    let shard1_path = prefix_dir.join("shard1.prefix");
    let shard2_path = prefix_dir.join("shard2.prefix");
    let shard1 = load_shard(&shard1_path, SHARD1_LENGTH, SHARD1_HEADER_SHA256)?;
    let shard2 = load_shard(&shard2_path, SHARD2_LENGTH, SHARD2_HEADER_SHA256)?;
    check_split_set(&[&shard1.header, &shard2.header])
        .map_err(|error| format!("GGUF shard split set: {error:?}"))?;

    let mut metadata_file =
        File::open(&shard1.path).map_err(|error| format!("open {}: {error}", shard1.path))?;
    let metadata = read_ple_metadata(&mut metadata_file, &shard1.header)?;

    let tensor = shard2
        .header
        .tensor(TABLE_TENSOR)
        .ok_or_else(|| format!("shard 2 is missing tensor {TABLE_TENSOR:?}"))?;
    let dims = tensor.dims();
    if tensor.ggml_type() != GgufType::Iq4Nl || dims.len() != 2 || dims[0] != 160 {
        return Err(format!(
            "unexpected n-gram tensor type or shape: {:?}, {dims:?}",
            tensor.ggml_type()
        ));
    }
    let table_rows = dims[1];
    if table_rows == 0 || tensor.byte_len() % table_rows != 0 {
        return Err("n-gram tensor byte length is not divisible by its row count".into());
    }
    let row_bytes = tensor.byte_len() / table_rows;
    if tensor.abs_offset() != 192
        || table_rows != 320_001_536
        || row_bytes != 90
        || tensor.byte_len() != 28_800_138_240
        || tensor.abs_range().end != SHARD2_LENGTH
    {
        return Err("n-gram tensor geometry differs from the recorded G1 header facts".into());
    }
    if metadata.layers != [1] || metadata.ngram_size != 3 || metadata.heads_per_ngram != 8 {
        return Err(
            "PLE layer, n-gram size, or head count differs from the recorded model layout".into(),
        );
    }
    if metadata.eos > u64::from(u32::MAX) {
        return Err("PLE EOS token id does not fit u32".into());
    }

    println!("metadata_source=bounded GgufHeader parsing and header metadata ranges");
    println!("payload_bytes_read=0");
    println!("shard1_header_len={}", shard1.header.header_len());
    println!("shard1_data_start={}", shard1.header.data_start());
    println!(
        "shard1_header_sha256={}",
        sha256_hex(&shard1.header.header_sha256())
    );
    println!("shard1_declared_file_len={SHARD1_LENGTH}");
    println!("shard2_header_len={}", shard2.header.header_len());
    println!("shard2_data_start={}", shard2.header.data_start());
    println!(
        "shard2_header_sha256={}",
        sha256_hex(&shard2.header.header_sha256())
    );
    println!("shard2_declared_file_len={SHARD2_LENGTH}");
    println!("ple.layers={:?}", metadata.layers);
    println!("ple.ngram_size={}", metadata.ngram_size);
    println!("ple.heads_per_ngram={}", metadata.heads_per_ngram);
    println!("ple.eos_token_id={}", metadata.eos);
    println!("ple.layer_multipliers={:?}", metadata.multipliers);
    println!("ple.head_offsets={:?}", metadata.offsets);
    println!("ple.head_vocab_sizes={:?}", metadata.vocab_sizes);
    println!("table.tensor={TABLE_TENSOR}");
    println!("table.rows={table_rows}");
    println!("table.row_bytes={row_bytes}");
    println!("table.tensor_offset={}", tensor.abs_offset());
    println!("payload_identity=not established by header hashes");
    Ok(())
}

fn load_shard(path: &Path, file_len: u64, expected_header_sha256: &str) -> Result<Shard, String> {
    let path_text = path.display().to_string();
    let mut reader = File::open(path).map_err(|error| format!("open {path_text}: {error}"))?;
    let header = load_shard_reader(&mut reader, file_len, expected_header_sha256)
        .map_err(|error| format!("{path_text}: {error}"))?;
    Ok(Shard {
        path: path_text,
        header,
    })
}

fn load_shard_reader<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    expected_header_sha256: &str,
) -> Result<GgufHeader, String> {
    let header = GgufHeader::parse(reader, file_len, HEADER_LIMIT)
        .map_err(|error| format!("parse GGUF header: {error:?}"))?;
    header
        .check_header_sha256(expected_header_sha256)
        .map_err(|error| format!("header hash: {error:?}"))?;
    Ok(header)
}

#[derive(Debug, PartialEq, Eq)]
struct PleMetadata {
    layers: Vec<i32>,
    multipliers: Vec<u64>,
    offsets: Vec<u64>,
    vocab_sizes: Vec<u64>,
    eos: u64,
    ngram_size: u64,
    heads_per_ngram: u64,
}

fn read_ple_metadata<R: Read + Seek>(
    reader: &mut R,
    header: &GgufHeader,
) -> Result<PleMetadata, String> {
    Ok(PleMetadata {
        layers: read_i32_array(reader, header, "qwen4exp.ple.layers", 1)?,
        multipliers: read_u64_array(reader, header, "qwen4exp.ple.layer_multipliers", 3)?,
        offsets: read_u64_array(reader, header, "qwen4exp.ple.head_offsets", 16)?,
        vocab_sizes: read_u64_array(reader, header, "qwen4exp.ple.head_vocab_sizes", 16)?,
        eos: read_integer(header, "qwen4exp.ple.eos_token_id")?,
        ngram_size: read_integer(header, "qwen4exp.ple.ngram_size")?,
        heads_per_ngram: read_integer(header, "qwen4exp.ple.heads_per_ngram")?,
    })
}

fn array_bytes<R: Read + Seek>(
    reader: &mut R,
    header: &GgufHeader,
    key: &str,
    element_type: u32,
    elements: u64,
    element_bytes: u64,
) -> Result<Vec<u8>, String> {
    let array = match header.meta(key) {
        Some(MetaValue::Array(array)) => array,
        Some(_) => return Err(format!("metadata {key:?} is not an array")),
        None => return Err(format!("metadata {key:?} is missing")),
    };
    if array.elem_type != element_type || array.len != elements {
        return Err(format!(
            "metadata {key:?} has type {} and length {}, expected type {element_type} and length {elements}",
            array.elem_type, array.len
        ));
    }
    let expected_bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| format!("metadata {key:?} byte length overflows"))?;
    if array.byte_range.end.checked_sub(array.byte_range.start) != Some(expected_bytes) {
        return Err(format!(
            "metadata {key:?} checked byte range has the wrong length"
        ));
    }
    let buffer_len = usize::try_from(expected_bytes)
        .map_err(|_| format!("metadata {key:?} does not fit this host"))?;
    reader
        .seek(SeekFrom::Start(array.byte_range.start))
        .map_err(|error| format!("seek to metadata {key:?}: {error}"))?;
    let mut bytes = vec![0; buffer_len];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("read metadata {key:?}: {error}"))?;
    Ok(bytes)
}

fn read_u64_array<R: Read + Seek>(
    reader: &mut R,
    header: &GgufHeader,
    key: &str,
    elements: u64,
) -> Result<Vec<u64>, String> {
    let bytes = array_bytes(reader, header, key, 10, elements, 8)?;
    let capacity = usize::try_from(elements)
        .map_err(|_| format!("metadata {key:?} element count does not fit this host"))?;
    let (chunks, remainder) = bytes.as_chunks::<8>();
    if !remainder.is_empty() {
        return Err(format!("metadata {key:?} has an incomplete u64 element"));
    }
    let mut values = Vec::with_capacity(capacity);
    values.extend(chunks.iter().map(|chunk| u64::from_le_bytes(*chunk)));
    Ok(values)
}

fn read_i32_array<R: Read + Seek>(
    reader: &mut R,
    header: &GgufHeader,
    key: &str,
    elements: u64,
) -> Result<Vec<i32>, String> {
    let bytes = array_bytes(reader, header, key, 5, elements, 4)?;
    let capacity = usize::try_from(elements)
        .map_err(|_| format!("metadata {key:?} element count does not fit this host"))?;
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(format!("metadata {key:?} has an incomplete i32 element"));
    }
    let mut values = Vec::with_capacity(capacity);
    values.extend(chunks.iter().map(|chunk| i32::from_le_bytes(*chunk)));
    Ok(values)
}

fn read_integer(header: &GgufHeader, key: &str) -> Result<u64, String> {
    let value = header
        .meta(key)
        .ok_or_else(|| format!("metadata {key:?} is missing"))?;
    match value {
        MetaValue::U8(value) => Ok(u64::from(*value)),
        MetaValue::U16(value) => Ok(u64::from(*value)),
        MetaValue::U32(value) => Ok(u64::from(*value)),
        MetaValue::U64(value) => Ok(*value),
        MetaValue::I8(value) => {
            u64::try_from(*value).map_err(|_| format!("metadata {key:?} is negative"))
        }
        MetaValue::I16(value) => {
            u64::try_from(*value).map_err(|_| format!("metadata {key:?} is negative"))
        }
        MetaValue::I32(value) => {
            u64::try_from(*value).map_err(|_| format!("metadata {key:?} is negative"))
        }
        MetaValue::I64(value) => {
            u64::try_from(*value).map_err(|_| format!("metadata {key:?} is negative"))
        }
        _ => Err(format!("metadata {key:?} is not an integer")),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Read, Seek, SeekFrom};

    use super::*;

    const OVERSIZED_INPUT_LEN: u64 = 1 << 40;

    struct HeaderOnlyReader {
        header: Vec<u8>,
        position: u64,
        virtual_len: u64,
        non_header_read_attempts: usize,
        max_read_end: u64,
    }

    impl HeaderOnlyReader {
        fn new(header: Vec<u8>, virtual_len: u64) -> Self {
            Self {
                header,
                position: 0,
                virtual_len,
                non_header_read_attempts: 0,
                max_read_end: 0,
            }
        }
    }

    impl Read for HeaderOnlyReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if buffer.is_empty() {
                return Ok(0);
            }
            let header_len = self.header.len() as u64;
            if self.position >= header_len {
                self.non_header_read_attempts += 1;
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "synthetic non-header read denied",
                ));
            }
            let start = self.position as usize;
            let count = buffer.len().min(self.header.len() - start);
            buffer[..count].copy_from_slice(&self.header[start..start + count]);
            self.position += count as u64;
            self.max_read_end = self.max_read_end.max(self.position);
            Ok(count)
        }
    }

    impl Seek for HeaderOnlyReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            let target = match position {
                SeekFrom::Start(offset) => i128::from(offset),
                SeekFrom::Current(offset) => i128::from(self.position) + i128::from(offset),
                SeekFrom::End(offset) => i128::from(self.virtual_len) + i128::from(offset),
            };
            if target < 0 || target > i128::from(self.virtual_len) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "synthetic seek is outside the virtual input",
                ));
            }
            self.position = target as u64;
            Ok(self.position)
        }
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        push_u32(bytes, 4);
        push_u32(bytes, value);
    }

    fn push_array_metadata(
        bytes: &mut Vec<u8>,
        key: &str,
        element_type: u32,
        element_count: u64,
        values: &[u8],
    ) {
        push_string(bytes, key);
        push_u32(bytes, 9);
        push_u32(bytes, element_type);
        push_u64(bytes, element_count);
        bytes.extend_from_slice(values);
    }

    fn encoded_i32(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn encoded_u64(values: &[u64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn synthetic_metadata_header() -> Vec<u8> {
        let mut entries = Vec::new();
        push_u32_metadata(&mut entries, "general.alignment", 4096);
        push_array_metadata(
            &mut entries,
            "qwen4exp.ple.layers",
            5,
            1,
            &encoded_i32(&[1]),
        );
        push_array_metadata(
            &mut entries,
            "qwen4exp.ple.layer_multipliers",
            10,
            3,
            &encoded_u64(&[3, 5, 7]),
        );
        push_array_metadata(
            &mut entries,
            "qwen4exp.ple.head_offsets",
            10,
            16,
            &encoded_u64(&(0..16).collect::<Vec<_>>()),
        );
        push_array_metadata(
            &mut entries,
            "qwen4exp.ple.head_vocab_sizes",
            10,
            16,
            &encoded_u64(&[101; 16]),
        );
        push_u32_metadata(&mut entries, "qwen4exp.ple.eos_token_id", 97);
        push_u32_metadata(&mut entries, "qwen4exp.ple.ngram_size", 3);
        push_u32_metadata(&mut entries, "qwen4exp.ple.heads_per_ngram", 8);

        let mut header = b"GGUF".to_vec();
        push_u32(&mut header, 3);
        push_u64(&mut header, 0);
        push_u64(&mut header, 8);
        header.extend_from_slice(&entries);
        header
    }

    #[test]
    fn metadata_extraction_does_not_read_past_header_in_oversized_prefix() {
        let header_bytes = synthetic_metadata_header();
        let mut hash_reader = Cursor::new(header_bytes.as_slice());
        let known_header =
            GgufHeader::parse(&mut hash_reader, OVERSIZED_INPUT_LEN, HEADER_LIMIT).unwrap();
        let expected_header_sha256 = sha256_hex(&known_header.header_sha256());
        let mut reader = HeaderOnlyReader::new(header_bytes, OVERSIZED_INPUT_LEN);

        let loaded = load_shard_reader(&mut reader, OVERSIZED_INPUT_LEN, &expected_header_sha256);
        assert_eq!(
            reader.non_header_read_attempts, 0,
            "metadata loading attempted a read past the synthetic header"
        );
        let header = loaded.expect("the synthetic header must load");
        assert!(header.data_start() < OVERSIZED_INPUT_LEN);
        assert!(header.header_len() < header.data_start());
        let metadata = read_ple_metadata(&mut reader, &header).unwrap();

        assert_eq!(metadata.layers, [1]);
        assert_eq!(metadata.multipliers, [3, 5, 7]);
        assert_eq!(metadata.offsets, (0..16).collect::<Vec<_>>());
        assert_eq!(metadata.vocab_sizes, [101; 16]);
        assert_eq!(metadata.eos, 97);
        assert_eq!(metadata.ngram_size, 3);
        assert_eq!(metadata.heads_per_ngram, 8);
        assert_eq!(reader.non_header_read_attempts, 0);
        assert!(reader.max_read_end <= header.header_len());
    }
}
