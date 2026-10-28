use std::sync::Arc;

use inference_engine::{
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, Tier, WeightSource},
};

#[test]
fn scalar_loads_through_the_weight_source() {
    let bytes: Arc<[u8]> = Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice());
    let weights = ResidentWeights::from_bytes(bytes).expect("valid scalar checkpoint");
    let tensor = weights.tensor("answer").expect("named scalar");

    assert_eq!(tensor.dtype(), Dtype::U8);
    assert!(tensor.shape().is_empty());
    assert_eq!(tensor.byte_len(), 1);
    assert_eq!(tensor.host_bytes(), Some([42].as_slice()));
}

#[test]
fn host_tensor_preserves_stored_shape_and_bytes() {
    let bytes = vec![0; 24];
    let tensor =
        Tensor::try_from_host(Dtype::F32, vec![2, 3], bytes).expect("six stored float32 elements");

    assert_eq!(tensor.dtype(), Dtype::F32);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.byte_len(), 24);
    assert_eq!(tensor.host_bytes(), Some([0; 24].as_slice()));
}

#[test]
fn host_storage_rejects_short_and_long_payloads() {
    for bytes in [vec![0; 23], vec![0; 25]] {
        assert!(matches!(
            Tensor::try_from_host(Dtype::F32, vec![2, 3], bytes),
            Err(inference_engine::EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn shape_product_overflow_returns_a_checkpoint_error() {
    assert!(matches!(
        Tensor::try_from_host(Dtype::U8, vec![2147483648, 2147483648, 4], vec![]),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
}

#[test]
fn byte_product_overflow_returns_a_checkpoint_error() {
    assert!(matches!(
        Tensor::try_from_host(Dtype::F32, vec![2147483648, 2147483648], vec![]),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
}

#[test]
fn a_later_zero_dimension_does_not_hide_overflow() {
    assert!(matches!(
        Tensor::try_from_host(Dtype::U8, vec![2147483648, 2147483648, 4, 0], vec![]),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
}

#[test]
fn zero_length_storage_is_not_a_scalar() {
    let tensor = Tensor::try_from_host(Dtype::F32, vec![2, 0], vec![]).unwrap();
    assert_eq!(tensor.shape(), &[2, 0]);
    assert_eq!(tensor.byte_len(), 0);
    assert_eq!(tensor.host_bytes(), Some([].as_slice()));
    assert!(Tensor::try_from_host(Dtype::F32, vec![], vec![]).is_err());
}

#[test]
fn tensor_views_keep_the_original_bytes_alive_without_payload_copies() {
    let bytes: Arc<[u8]> = Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice());
    let weights = ResidentWeights::from_bytes(Arc::clone(&bytes)).unwrap();
    let first = weights.tensor("answer").unwrap();
    let second = weights.tensor("answer").unwrap();
    let first_bytes = first.host_bytes().unwrap();

    assert!(bytes.as_ptr_range().contains(&first_bytes.as_ptr()));
    assert_eq!(first_bytes.as_ptr(), second.host_bytes().unwrap().as_ptr());
    drop(weights);
    drop(bytes);
    drop(first);
    assert_eq!(second.host_bytes(), Some([42].as_slice()));
}

#[test]
fn missing_keys_return_none() {
    let bytes: Arc<[u8]> = Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice());
    let weights = ResidentWeights::from_bytes(bytes).unwrap();
    assert!(weights.tensor("missing").is_none());
    assert!(weights.tensor("answer").is_some());
}

fn file_bytes(header: &str, payload: &[u8]) -> Arc<[u8]> {
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    Arc::from(bytes)
}

#[test]
fn malformed_files_fail_before_a_weight_source_is_published() {
    let cases = [
        Arc::from([0u8; 7]),
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
    ];
    for bytes in cases {
        assert!(matches!(
            ResidentWeights::from_bytes(bytes),
            Err(inference_engine::EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn unsupported_dtype_rejects_the_entire_checkpoint_with_key_context() {
    let bytes = file_bytes(
        r#"{"valid":{"dtype":"U8","shape":[],"data_offsets":[0,1]},"unsupported":{"dtype":"BOOL","shape":[],"data_offsets":[1,2]}}"#,
        &[42, 1],
    );
    let error = ResidentWeights::from_bytes(bytes).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("unsupported"));
    assert!(message.contains("BOOL"));
}

#[test]
fn header_dimensions_cannot_truncate_to_u32() {
    let bytes = file_bytes(
        r#"{"wide":{"dtype":"U8","shape":[4294967296,0],"data_offsets":[0,0]}}"#,
        &[],
    );
    assert!(matches!(
        ResidentWeights::from_bytes(bytes),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
}

#[test]
fn packed_payload_keeps_its_stored_shape_and_storage_accounting() {
    let bytes: Arc<[u8]> = Arc::from(include_bytes!("fixtures/mixed.safetensors").as_slice());
    let weights = ResidentWeights::from_bytes(bytes).unwrap();
    let packed = weights.tensor("packed").unwrap();

    assert_eq!(packed.dtype(), Dtype::U32);
    assert_eq!(packed.shape(), &[2, 8]);
    assert_eq!(packed.byte_len(), 64);
    assert_eq!(packed.host_bytes().unwrap(), &(0..64).collect::<Vec<u8>>());
    assert_eq!(weights.resident_bytes(), 81);
    assert_eq!(weights.tier(), Tier::Resident);
    let _second_view = weights.tensor("packed").unwrap();
    assert_eq!(weights.resident_bytes(), 81);
}

#[test]
fn supported_scalar_dtypes_keep_their_metadata_and_bytes() {
    let bytes: Arc<[u8]> = Arc::from(include_bytes!("fixtures/mixed.safetensors").as_slice());
    let weights = ResidentWeights::from_bytes(bytes).unwrap();
    let cases: &[(&str, Dtype, &[u8])] = &[
        ("u8", Dtype::U8, &[42]),
        ("f16", Dtype::F16, &[5, 6]),
        ("bf16", Dtype::Bf16, &[7, 8]),
        ("f32", Dtype::F32, &[1, 2, 3, 4]),
        ("i64", Dtype::I64, &[9, 10, 11, 12, 13, 14, 15, 16]),
    ];
    for &(key, dtype, expected) in cases {
        let tensor = weights.tensor(key).unwrap();
        assert_eq!(tensor.dtype(), dtype);
        assert!(tensor.shape().is_empty());
        assert_eq!(tensor.host_bytes(), Some(expected));
        let host = Tensor::try_from_host(dtype, vec![], expected.to_vec()).unwrap();
        assert_eq!(host.host_bytes(), Some(expected));
    }
    let empty = weights.tensor("empty").unwrap();
    assert_eq!(empty.shape(), &[2, 0]);
    assert_eq!(empty.byte_len(), 0);
    assert_eq!(empty.host_bytes(), Some([].as_slice()));
}

#[test]
fn empty_checkpoint_is_a_valid_empty_weight_source() {
    let weights = ResidentWeights::from_bytes(file_bytes("{}", &[])).unwrap();
    assert_eq!(weights.resident_bytes(), 0);
    assert!(weights.tensor("missing").is_none());
}

#[test]
fn cumulative_tensor_ranges_return_a_checkpoint_error_without_overflow() {
    let mut entries = Vec::new();
    let mut start = 0u64;
    for index in 0..16 {
        let count = if index == 15 {
            (1u64 << 60) - 1
        } else {
            1u64 << 60
        };
        let end = start.checked_add(count).unwrap();
        entries.push(format!(
            r#""t{index}":{{"dtype":"U8","shape":[{count}],"data_offsets":[{start},{end}]}}"#
        ));
        start = end;
    }
    let header = format!("{{{}}}", entries.join(","));
    assert!(matches!(
        ResidentWeights::from_bytes(file_bytes(&header, &[])),
        Err(inference_engine::EngError::Checkpoint(_))
    ));
}

/// Each case: label, header JSON, and the repeated member name. Every payload is 8 bytes.
const DUPLICATE_CASES: [(&str, &str, &str); 6] = [
    (
        "tensor name with conflicting entries",
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
        "a",
    ),
    (
        "tensor name with equal entries",
        r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
        "a",
    ),
    (
        "tensor name whose last entry leaves an offset gap",
        r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[4,8]},"a":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#,
        "a",
    ),
    (
        "field inside one tensor entry",
        r#"{"a":{"dtype":"U8","dtype":"F32","shape":[8],"data_offsets":[0,8]}}"#,
        "dtype",
    ),
    (
        "metadata member",
        r#"{"__metadata__":{"format":"pt","format":"mlx"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
        "format",
    ),
    (
        "metadata object",
        r#"{"__metadata__":{"format":"pt"},"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
        "__metadata__",
    ),
];

fn checkpoint_message(error: inference_engine::EngError) -> String {
    match error {
        inference_engine::EngError::Checkpoint(message) => message,
        other => panic!("expected a checkpoint error, found {other:?}"),
    }
}

#[test]
fn duplicate_json_names_reject_the_checkpoint_with_the_documented_text() {
    for (label, header, name) in DUPLICATE_CASES {
        let message = checkpoint_message(
            ResidentWeights::from_bytes(file_bytes(header, &[0; 8])).unwrap_err(),
        );
        assert!(
            message.contains(&format!("duplicate JSON member {name:?}")),
            "{label}: {message}"
        );
    }
}

#[test]
fn metadata_and_whitespace_padding_remain_valid_header_content() {
    let header = format!(
        r#"{{"__metadata__":{{"format":"pt"}},"a":{{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}}}{}"#,
        " ".repeat(5)
    );
    let weights = ResidentWeights::from_bytes(file_bytes(&header, &[7, 9])).unwrap();
    assert_eq!(
        weights.tensor("a").unwrap().host_bytes(),
        Some([7, 9].as_slice())
    );
    assert!(weights.tensor("__metadata__").is_none());
    assert_eq!(weights.resident_bytes(), 2);
}

#[test]
fn the_resident_loader_and_the_shard_header_return_the_same_duplicate_text() {
    use inference_engine::weights::ShardHeader;
    for (label, header, name) in DUPLICATE_CASES {
        let bytes = file_bytes(header, &[0; 8]);
        let length = bytes.len() as u64;
        let shard = ShardHeader::read(&mut std::io::Cursor::new(bytes.to_vec()), length, 1 << 20)
            .map(drop)
            .unwrap_err();
        let resident = ResidentWeights::from_bytes(bytes).map(drop).unwrap_err();
        let expected = format!("duplicate JSON member {name:?}");
        let shard = checkpoint_message(shard);
        let resident = checkpoint_message(resident);
        assert!(shard.contains(&expected), "{label}: {shard}");
        assert!(resident.contains(&expected), "{label}: {resident}");
        assert!(
            shard.starts_with("shard header is not valid JSON: "),
            "{label}: {shard}"
        );
        assert!(
            resident.starts_with("safetensors header is not valid JSON: "),
            "{label}: {resident}"
        );
    }
}

#[test]
fn an_unsupported_dtype_error_names_the_tensor_and_the_stored_dtype() {
    let bytes = file_bytes(
        r#"{"flag":{"dtype":"BOOL","shape":[],"data_offsets":[0,1]}}"#,
        &[1],
    );
    assert_eq!(
        ResidentWeights::from_bytes(bytes).unwrap_err(),
        inference_engine::EngError::Checkpoint(r#"tensor "flag": unsupported dtype BOOL"#.into())
    );
}

const METADATA_ENTRY: &str = r#""a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}"#;
const INVALID_METADATA: [&str; 7] = [
    "1",
    "true",
    r#""pt""#,
    r#"["pt"]"#,
    r#"{"format":1}"#,
    r#"{"format":null}"#,
    r#"{"format":{"nested":"pt"}}"#,
];

fn load_on_every_path(metadata: &str) -> [Result<(), inference_engine::EngError>; 4] {
    use inference_engine::weights::{IndexedWeights, ShardHeader};
    let header = format!(r#"{{"__metadata__":{metadata},{METADATA_ENTRY}}}"#);
    let bytes = file_bytes(&header, &[0]);
    let length = bytes.len() as u64;
    let shard = ShardHeader::read(&mut std::io::Cursor::new(bytes.to_vec()), length, 1 << 20);
    let resident = ResidentWeights::from_bytes(Arc::clone(&bytes));
    let mut reader = bytes.as_ref();
    let reader = ResidentWeights::from_reader(&mut reader, length, length);
    let indexed = IndexedWeights::from_shards(
        br#"{"weight_map":{"a":"one.safetensors"}}"#,
        std::collections::BTreeMap::from([("one.safetensors".to_owned(), bytes)]),
    );
    [
        shard.map(drop),
        resident.map(drop),
        reader.map(drop),
        indexed.map(drop),
    ]
}

#[test]
fn null_metadata_loads_on_every_path() {
    for (path, result) in load_on_every_path("null").into_iter().enumerate() {
        assert_eq!(result, Ok(()), "path {path} rejected null metadata");
    }
}

#[test]
fn metadata_that_is_not_a_string_map_is_rejected_on_every_path() {
    for metadata in INVALID_METADATA {
        for (path, result) in load_on_every_path(metadata).into_iter().enumerate() {
            assert!(
                matches!(result, Err(inference_engine::EngError::Checkpoint(_))),
                "path {path} accepted metadata {metadata}"
            );
        }
    }
}

#[test]
fn the_metadata_error_has_the_same_text_on_every_path() {
    for metadata in INVALID_METADATA {
        for (path, result) in load_on_every_path(metadata).into_iter().enumerate() {
            let message = checkpoint_message(result.unwrap_err());
            assert!(
                message.ends_with(" metadata is not a string map"),
                "path {path}, metadata {metadata}: {message}"
            );
        }
    }
}

#[test]
fn a_string_map_loads_on_every_path() {
    for metadata in ["{}", r#"{"format":"mlx"}"#] {
        for (path, result) in load_on_every_path(metadata).into_iter().enumerate() {
            assert_eq!(result, Ok(()), "path {path} rejected metadata {metadata}");
        }
    }
}
