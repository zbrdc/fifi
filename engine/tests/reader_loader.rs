use inference_engine::{
    tensor::Dtype,
    weights::{ResidentWeights, WeightSource},
};

#[test]
fn reader_loads_a_retained_scalar() {
    let mut reader = include_bytes!("fixtures/scalar.safetensors").as_slice();
    let length = reader.len() as u64;
    let weights = ResidentWeights::from_reader(&mut reader, length, length).unwrap();
    let tensor = weights.tensor("answer").unwrap();
    drop(weights);
    assert_eq!(tensor.dtype(), Dtype::U8);
    assert!(tensor.shape().is_empty());
    assert_eq!(tensor.byte_len(), 1);
    assert_eq!(tensor.host_bytes(), Some([42].as_slice()));
}

#[test]
fn zero_budget_rejects_input_before_reading() {
    let bytes = include_bytes!("fixtures/scalar.safetensors");
    let mut reader = std::io::Cursor::new(bytes);
    assert!(matches!(
        ResidentWeights::from_reader(&mut reader, bytes.len() as u64, 0),
        Err(inference_engine::EngError::BadRequest(_))
    ));
    assert_eq!(reader.position(), 0);
}

#[test]
fn inputs_shorter_than_a_header_fail_before_reading() {
    for length in 0..8 {
        let mut reader = std::io::Cursor::new([0; 8]);
        assert!(matches!(
            ResidentWeights::from_reader(&mut reader, length, 8),
            Err(inference_engine::EngError::Checkpoint(_))
        ));
        assert_eq!(reader.position(), 0);
    }
}

#[test]
fn oversized_inputs_fail_before_reading_or_allocation() {
    for (length, budget) in [
        (65, 64),
        (isize::MAX as u64 + 1, u64::MAX),
        (u64::MAX, u64::MAX),
    ] {
        let mut reader = std::io::Cursor::new([0; 8]);
        assert!(matches!(
            ResidentWeights::from_reader(&mut reader, length, budget),
            Err(inference_engine::EngError::MemoryPlan(_))
        ));
        assert_eq!(reader.position(), 0);
    }
}

#[test]
fn trailing_input_rejects_the_source_after_one_extra_byte() {
    let bytes = include_bytes!("fixtures/scalar.safetensors");
    let mut extended = bytes.to_vec();
    extended.extend_from_slice(&[8, 9, 10]);
    let mut reader = std::io::Cursor::new(extended);
    assert!(matches!(
        ResidentWeights::from_reader(&mut reader, bytes.len() as u64, 1024),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
    assert_eq!(reader.position(), bytes.len() as u64 + 1);
}

struct ReadAdapter<F>(F);

impl<F: FnMut(&mut [u8]) -> std::io::Result<usize>> std::io::Read for ReadAdapter<F> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        (self.0)(output)
    }
}

#[test]
fn short_and_interrupted_reads_include_the_end_probe() {
    use std::io::{ErrorKind, Read};
    let mut bytes = include_bytes!("fixtures/scalar.safetensors").as_slice();
    let length = bytes.len() as u64;
    let mut interrupt = true;
    let mut reader = ReadAdapter(|output: &mut [u8]| {
        interrupt = !interrupt;
        if !interrupt {
            return Err(ErrorKind::Interrupted.into());
        }
        bytes.read(&mut output[..1])
    });
    let weights = ResidentWeights::from_reader(&mut reader, length, length).unwrap();
    assert_eq!(
        weights.tensor("answer").unwrap().host_bytes(),
        Some([42].as_slice())
    );
}

#[test]
fn mixed_storage_preserves_metadata_and_payload_accounting() {
    let mut bytes = include_bytes!("fixtures/mixed.safetensors").as_slice();
    let length = bytes.len() as u64;
    let source = ResidentWeights::from_reader(&mut bytes, length, length).unwrap();
    assert_eq!(source.resident_bytes(), 81);
    assert_eq!(source.tier(), inference_engine::weights::Tier::Resident);
    assert!(source.tensor("missing").is_none());
    for (key, dtype, expected) in [
        ("u8", Dtype::U8, &[42][..]),
        ("f16", Dtype::F16, &[5, 6][..]),
        ("bf16", Dtype::Bf16, &[7, 8][..]),
        ("f32", Dtype::F32, &[1, 2, 3, 4][..]),
        ("i64", Dtype::I64, &[9, 10, 11, 12, 13, 14, 15, 16][..]),
    ] {
        let tensor = source.tensor(key).unwrap();
        assert_eq!(tensor.dtype(), dtype);
        assert!(tensor.shape().is_empty());
        assert_eq!(tensor.host_bytes(), Some(expected));
    }
    let packed = source.tensor("packed").unwrap();
    assert_eq!(packed.dtype(), Dtype::U32);
    assert_eq!(packed.shape(), &[2, 8]);
    assert_eq!(packed.byte_len(), 64);
    assert_eq!(packed.host_bytes().unwrap(), &(0..64).collect::<Vec<u8>>());
    assert_eq!(source.tensor("empty").unwrap().shape(), &[2, 0]);
}

#[test]
fn early_end_and_reader_failures_reject_the_source() {
    use std::io::{Error, Read};
    let fixture = include_bytes!("fixtures/scalar.safetensors");
    let length = fixture.len() as u64;
    let mut short = &fixture[..fixture.len() - 1];
    assert!(matches!(
        ResidentWeights::from_reader(&mut short, length, length),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
    for fail_at in [0, 5, fixture.len()] {
        let mut bytes = fixture.as_slice();
        let mut consumed = 0;
        let mut reader = ReadAdapter(|output: &mut [u8]| {
            if consumed == fail_at {
                return Err(Error::other("fixture reader failure"));
            }
            let limit = output.len().min(fail_at - consumed);
            let count = bytes.read(&mut output[..limit])?;
            consumed += count;
            Ok(count)
        });
        let error = ResidentWeights::from_reader(&mut reader, length, length).unwrap_err();
        assert!(
            matches!(error, inference_engine::EngError::Checkpoint(message)
            if message.contains("fixture reader failure"))
        );
    }
}

fn file_bytes(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn malformed_storage_rejects_the_complete_source() {
    for bytes in [
        file_bytes("{", &[]),
        file_bytes(
            r#"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#,
            &[],
        ),
        file_bytes(
            r#"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            &[0, 1],
        ),
        file_bytes(
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            &[0],
        ),
        file_bytes(
            r#"{"bad":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}"#,
            &[0],
        ),
        file_bytes(
            r#"{"bad":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
            &[0],
        ),
        file_bytes(
            r#"{"bad":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            &[0, 1],
        ),
        file_bytes(
            r#"{"valid":{"dtype":"U8","shape":[],"data_offsets":[0,1]},"bad":{"dtype":"BOOL","shape":[],"data_offsets":[1,2]}}"#,
            &[42, 1],
        ),
        file_bytes(
            r#"{"wide":{"dtype":"U8","shape":[4294967296,0],"data_offsets":[0,0]}}"#,
            &[],
        ),
        file_bytes(
            r#"{"huge":{"dtype":"U8","shape":[2147483648,2147483648,4,0],"data_offsets":[0,0]}}"#,
            &[],
        ),
    ] {
        let length = bytes.len() as u64;
        assert!(matches!(
            ResidentWeights::from_reader(&mut bytes.as_slice(), length, length),
            Err(inference_engine::EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn mapped_views_share_immutable_storage_across_threads() {
    let bytes = include_bytes!("fixtures/scalar.safetensors");
    let length = bytes.len() as u64;
    let weights = ResidentWeights::from_reader(&mut bytes.as_slice(), length, length).unwrap();
    let first = weights.tensor("answer").unwrap();
    let second = weights.tensor("answer").unwrap();
    assert_eq!(
        first.host_bytes().unwrap().as_ptr(),
        second.host_bytes().unwrap().as_ptr()
    );
    drop(weights);
    drop(first);
    std::thread::spawn(move || assert_eq!(second.host_bytes(), Some([42].as_slice())))
        .join()
        .unwrap();
}

#[test]
fn an_empty_checkpoint_has_no_payload() {
    let bytes = file_bytes("{}", &[]);
    let length = bytes.len() as u64;
    let weights = ResidentWeights::from_reader(&mut bytes.as_slice(), length, length).unwrap();
    assert_eq!(weights.resident_bytes(), 0);
    assert!(weights.tensor("missing").is_none());
}

#[test]
fn source_file_changes_do_not_change_retained_tensor_bytes() {
    use std::io::{Seek, Write};
    let mut file = tempfile::tempfile().unwrap();
    let bytes = include_bytes!("fixtures/scalar.safetensors");
    file.write_all(bytes).unwrap();
    file.rewind().unwrap();
    let length = bytes.len() as u64;
    let weights = ResidentWeights::from_reader(&mut file, length, length).unwrap();
    let tensor = weights.tensor("answer").unwrap();
    drop(weights);
    file.set_len(0).unwrap();
    file.rewind().unwrap();
    file.write_all(b"changed input").unwrap();
    drop(file);
    assert_eq!(tensor.host_bytes(), Some([42].as_slice()));
}

#[test]
fn duplicate_json_names_reject_the_reader_source() {
    for (header, name) in [
        (
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            "a",
        ),
        (
            r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            "a",
        ),
        (
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[4,8]},"a":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#,
            "a",
        ),
        (
            r#"{"a":{"dtype":"U8","dtype":"F32","shape":[8],"data_offsets":[0,8]}}"#,
            "dtype",
        ),
        (
            r#"{"__metadata__":{"format":"pt","format":"mlx"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            "format",
        ),
        (
            r#"{"__metadata__":{"format":"pt"},"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
            "__metadata__",
        ),
    ] {
        let bytes = file_bytes(header, &[0; 8]);
        let length = bytes.len() as u64;
        let error =
            ResidentWeights::from_reader(&mut bytes.as_slice(), length, length).unwrap_err();
        let inference_engine::EngError::Checkpoint(message) = error else {
            panic!("expected a checkpoint error");
        };
        assert!(
            message.contains(&format!("duplicate JSON member {name:?}")),
            "{message}"
        );
    }
}

#[test]
fn an_unmappable_declared_length_reports_the_mapping_failure() {
    let mut reader = std::io::Cursor::new([0; 8]);
    let error = ResidentWeights::from_reader(&mut reader, 1 << 56, u64::MAX).unwrap_err();
    assert!(
        matches!(
            error,
            inference_engine::EngError::Backend {
                operation: "host_map_create",
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(reader.position(), 0);
}
