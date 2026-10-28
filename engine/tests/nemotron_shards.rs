use std::{fs, path::Path};

use inference_engine::{EngError, weights::nemotron_shards::NemotronShardReader};

const HEADER: &[u8] = include_bytes!("fixtures/nemotron35/shard-header.json");

fn write_shard(directory: &Path, name: &str, header: &[u8], payload: &[u8]) -> std::path::PathBuf {
    let path = directory.join(name);
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(payload);
    fs::write(&path, bytes).expect("write safetensors fixture");
    path
}

#[test]
fn reads_exact_tensor_bytes_from_a_tiny_safetensors_file() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let payload = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00, 0x00, 0x80, 0x3f,
    ];
    let path = write_shard(directory.path(), "tiny.safetensors", HEADER, &payload);
    let mut shard = NemotronShardReader::open(&path).expect("open tiny shard");

    let tensor = shard.tensor("sample.weight").expect("tensor metadata");
    assert_eq!(tensor.dtype(), "BF16");
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.offsets(), &(0..8));
    assert_eq!(tensor.byte_len(), 8);
    assert_eq!(
        shard.read_tensor("sample.weight").expect("read tensor"),
        payload[..8]
    );
    assert_eq!(
        shard.read_tensor("sample.input_scale").expect("read scale"),
        payload[8..]
    );
}

#[test]
fn maps_absolute_ranges_and_reads_a_sub_range_of_a_tensor() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let payload = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00, 0x00, 0x80, 0x3f,
    ];
    let path = write_shard(directory.path(), "ranges.safetensors", HEADER, &payload);
    let mut shard = NemotronShardReader::open(&path).expect("open tiny shard");
    let file = fs::read(&path).expect("whole shard");

    assert_eq!(shard.file_len(), file.len() as u64);
    let range = shard
        .absolute_range("sample.weight")
        .expect("range of a listed tensor");
    assert_eq!(&file[range], &payload[..8]);
    let scale = shard.absolute_range("sample.input_scale").expect("range");
    assert_eq!(&file[scale], &payload[8..]);
    assert_eq!(shard.absolute_range("missing"), None);

    assert_eq!(
        shard
            .read_tensor_range("sample.weight", 2, 4)
            .expect("inner range"),
        payload[2..6]
    );
    assert_eq!(
        shard
            .read_tensor_range("sample.weight", 8, 0)
            .expect("empty range at the end"),
        Vec::<u8>::new()
    );
    for (offset, length) in [(5, 4), (9, 0), (u64::MAX, 2)] {
        assert!(matches!(
            shard.read_tensor_range("sample.weight", offset, length),
            Err(EngError::Checkpoint(_))
        ));
    }
    assert!(shard.read_tensor_range("missing", 0, 1).is_err());

    // Evicting a range from the page cache changes nothing about what the file reads back.
    let ranges = [
        shard.absolute_range("sample.weight").expect("range"),
        shard.absolute_range("sample.input_scale").expect("range"),
    ];
    shard.drop_cache(&ranges).expect("drop the cached pages");
    assert_eq!(
        shard.read_tensor("sample.weight").expect("read again"),
        payload[..8]
    );
}

#[test]
fn rejects_duplicate_tensor_names_in_the_header() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let path = write_shard(directory.path(), "duplicate.safetensors", header, &[0]);

    assert!(
        matches!(NemotronShardReader::open(path), Err(EngError::Checkpoint(message)) if message.contains("duplicate JSON member"))
    );
}

#[test]
fn rejects_a_file_truncated_inside_its_header() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("truncated.safetensors");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&64u64.to_le_bytes());
    bytes.extend_from_slice(b"{}");
    fs::write(&path, bytes).expect("write truncated shard");

    assert!(matches!(
        NemotronShardReader::open(path),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn rejects_tensor_offsets_outside_the_real_file() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let header = br#"{"x":{"dtype":"U8","shape":[5],"data_offsets":[0,5]}}"#;
    let path = write_shard(directory.path(), "out-of-bounds.safetensors", header, &[0]);

    assert!(matches!(
        NemotronShardReader::open(path),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn rejects_shape_and_dtype_byte_size_mismatches() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let header = br#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    let path = write_shard(directory.path(), "mismatch.safetensors", header, &[0; 4]);

    assert!(matches!(
        NemotronShardReader::open(path),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn rejects_unrecognized_dtype_strings() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let header = br#"{"x":{"dtype":"F4_UNKNOWN","shape":[1],"data_offsets":[0,1]}}"#;
    let path = write_shard(directory.path(), "unknown-dtype.safetensors", header, &[0]);

    assert!(matches!(
        NemotronShardReader::open(path),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn reads_pinned_nemotron_shard_entries_when_configured() {
    let Some(path) = std::env::var_os("NEM35_SHARD") else {
        return;
    };
    let mut shard = NemotronShardReader::open(path).expect("pinned Nemotron shard parses");

    let embedding = shard
        .tensor("language_model.backbone.embeddings.weight")
        .expect("embedding metadata");
    assert_eq!(embedding.dtype(), "BF16");
    assert_eq!(embedding.shape(), [131_072, 4_096]);
    assert_eq!(embedding.byte_len(), 1_073_741_824);

    let projection = shard
        .tensor("language_model.backbone.layers.0.mixer.in_proj.weight")
        .expect("projection metadata");
    assert_eq!(projection.dtype(), "F8_E4M3");
    assert_eq!(projection.shape(), [18_560, 4_096]);
    assert_eq!(projection.byte_len(), 76_021_760);

    let input_scale_metadata = shard
        .tensor("language_model.backbone.layers.0.mixer.in_proj.input_scale")
        .expect("input scale metadata");
    assert_eq!(input_scale_metadata.dtype(), "F32");
    assert_eq!(input_scale_metadata.shape(), [1]);

    let input_scale = shard
        .read_tensor("language_model.backbone.layers.0.mixer.in_proj.input_scale")
        .expect("read input scale");
    assert_eq!(input_scale.len(), 4);
    let value = f32::from_le_bytes(input_scale.try_into().expect("four-byte F32"));
    assert!(
        value.is_finite() && value > 0.0,
        "invalid input scale: {value}"
    );
}
