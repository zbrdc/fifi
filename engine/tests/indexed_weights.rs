use std::{collections::BTreeMap, sync::Arc};

use inference_engine::{
    tensor::Dtype,
    weights::{IndexedWeights, Tier, WeightSource},
};

fn shards() -> BTreeMap<String, Arc<[u8]>> {
    BTreeMap::from([
        (
            "scalar.safetensors".into(),
            Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice()),
        ),
        (
            "mixed.safetensors".into(),
            Arc::from(include_bytes!("fixtures/mixed.safetensors").as_slice()),
        ),
    ])
}

const INDEX: &[u8] = br#"{"weight_map":{
    "answer":"scalar.safetensors",
    "packed":"mixed.safetensors",
    "u8":"mixed.safetensors",
    "f16":"mixed.safetensors",
    "bf16":"mixed.safetensors",
    "f32":"mixed.safetensors",
    "i64":"mixed.safetensors",
    "empty":"mixed.safetensors"
}}"#;

#[test]
fn indexed_shards_expose_the_complete_checkpoint_through_weight_source() {
    let weights = IndexedWeights::from_shards(INDEX, shards()).unwrap();
    let source: &dyn WeightSource = &weights;
    let scalar = source.tensor("answer").unwrap();
    let packed = source.tensor("packed").unwrap();

    assert_eq!(scalar.dtype(), Dtype::U8);
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.host_bytes(), Some([42].as_slice()));
    assert_eq!(packed.dtype(), Dtype::U32);
    assert_eq!(packed.shape(), &[2, 8]);
    assert_eq!(&packed.host_bytes().unwrap()[..4], &[0, 1, 2, 3]);
    assert_eq!(source.tier(), Tier::Resident);
    assert_eq!(source.resident_bytes(), 82);
    assert!(source.tensor("missing").is_none());
}

#[test]
fn index_requires_a_nonempty_weight_map_object() {
    for index in [
        "",
        "{",
        "[]",
        "null",
        "false",
        "0",
        "{}",
        r#"{"weight_map":null}"#,
        r#"{"weight_map":[]}"#,
        r#"{"weight_map":{}}"#,
    ] {
        assert!(
            matches!(
                IndexedWeights::from_shards(index.as_bytes(), shards()),
                Err(inference_engine::EngError::Checkpoint(_))
            ),
            "accepted {index}"
        );
    }
}

#[test]
fn index_rejects_duplicate_json_names_including_ignored_metadata() {
    let map = std::str::from_utf8(INDEX).unwrap();
    let cases = [
        map.replace("\"weight_map\":", "\"weight_map\":{},\"weight_map\":"),
        map.replace(
            "\"answer\":",
            "\"answer\":\"scalar.safetensors\",\"answer\":",
        ),
        map.replacen('{', r#"{"metadata":{"total_size":82,"total_size":82},"#, 1),
        map.replacen('{', r#"{"extra":[{"same":null,"same":true}],"#, 1),
    ];
    for index in cases {
        let error = IndexedWeights::from_shards(index.as_bytes(), shards()).unwrap_err();
        let inference_engine::EngError::Checkpoint(message) = error else {
            panic!("expected a checkpoint error");
        };
        assert!(message.contains("duplicate"), "{message}");
    }
}

#[test]
fn every_index_entry_requires_a_nonempty_tensor_name_and_string_shard() {
    for index in [
        br#"{"weight_map":{"":"scalar.safetensors"}}"#.as_slice(),
        br#"{"weight_map":{"answer":null}}"#.as_slice(),
        br#"{"weight_map":{"answer":42}}"#.as_slice(),
        br#"{"weight_map":{"answer":["scalar.safetensors"]}}"#.as_slice(),
    ] {
        assert!(matches!(
            IndexedWeights::from_shards(index, shards()),
            Err(inference_engine::EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn shard_names_must_be_ascii_safetensors_basenames() {
    for name in [
        "",
        "../scalar.safetensors",
        "/scalar.safetensors",
        "dir/scalar.safetensors",
        "dir\\scalar.safetensors",
        ".",
        "..",
        "scalar.bin",
        "café.safetensors",
        "line\nbreak.safetensors",
        "nul\0.safetensors",
        "delete\u{7f}.safetensors",
    ] {
        let index = serde_json::to_vec(&serde_json::json!({"weight_map":{"answer":name}})).unwrap();
        let bytes = Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice());
        assert!(
            matches!(
                IndexedWeights::from_shards(&index, BTreeMap::from([(name.into(), bytes)])),
                Err(inference_engine::EngError::Checkpoint(_))
            ),
            "accepted {name:?}"
        );
    }
}

#[test]
fn supplied_shards_must_equal_the_index_shard_set() {
    let mut missing = shards();
    missing.remove("scalar.safetensors");
    let mut extra = shards();
    extra.insert(
        "extra.safetensors".into(),
        Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice()),
    );
    for inputs in [missing, extra, BTreeMap::new()] {
        assert!(matches!(
            IndexedWeights::from_shards(INDEX, inputs),
            Err(inference_engine::EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn tensor_keys_cannot_appear_in_more_than_one_shard() {
    let mut inputs = shards();
    inputs.insert(
        "extra.safetensors".into(),
        Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice()),
    );
    let index = std::str::from_utf8(INDEX)
        .unwrap()
        .replace("\"answer\":", "\"other\":\"extra.safetensors\",\"answer\":");
    let error = IndexedWeights::from_shards(index.as_bytes(), inputs).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("answer"), "{message}");
    assert!(message.contains("duplicate"), "{message}");
}

#[test]
fn index_assignment_must_identify_the_shard_that_contains_the_tensor() {
    let index = std::str::from_utf8(INDEX)
        .unwrap()
        .replace(
            "\"answer\":\"scalar.safetensors\"",
            "\"answer\":\"mixed.safetensors\"",
        )
        .replace(
            "\"packed\":\"mixed.safetensors\"",
            "\"packed\":\"scalar.safetensors\"",
        );
    let error = IndexedWeights::from_shards(index.as_bytes(), shards()).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(
        message.contains("answer") || message.contains("packed"),
        "{message}"
    );
    assert!(message.contains("safetensors"), "{message}");
}

#[test]
fn every_stored_tensor_requires_an_index_entry() {
    let index = std::str::from_utf8(INDEX).unwrap().replace(
        "\"empty\":\"mixed.safetensors\"",
        "\"ghost\":\"mixed.safetensors\"",
    );
    let error = IndexedWeights::from_shards(index.as_bytes(), shards()).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("empty"), "{message}");
    assert!(message.contains("mixed.safetensors"), "{message}");
}

#[test]
fn every_indexed_tensor_must_exist_in_the_named_shard() {
    let index = std::str::from_utf8(INDEX)
        .unwrap()
        .replace("\"answer\":", "\"ghost\":\"mixed.safetensors\",\"answer\":");
    let error = IndexedWeights::from_shards(index.as_bytes(), shards()).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("ghost"), "{message}");
    assert!(message.contains("mixed.safetensors"), "{message}");
}

#[test]
fn each_shard_must_pass_complete_buffer_validation_before_publication() {
    for broken in [vec![0; 7], {
        let mut bytes = include_bytes!("fixtures/scalar.safetensors").to_vec();
        bytes.push(0);
        bytes
    }] {
        let mut inputs = shards();
        inputs.insert("scalar.safetensors".into(), Arc::from(broken));
        let error = IndexedWeights::from_shards(INDEX, inputs).unwrap_err();
        let inference_engine::EngError::Checkpoint(message) = error else {
            panic!("expected a checkpoint error");
        };
        assert!(message.contains("scalar.safetensors"), "{message}");
    }
}

#[test]
fn returned_tensors_retain_original_shard_buffers_after_the_source_drops() {
    let inputs = shards();
    let scalar_bytes = inputs["scalar.safetensors"].clone();
    let mixed_bytes = inputs["mixed.safetensors"].clone();
    let weights = IndexedWeights::from_shards(INDEX, inputs).unwrap();
    let scalar = weights.tensor("answer").unwrap();
    let packed = weights.tensor("packed").unwrap();
    let repeated = weights.tensor("packed").unwrap();
    assert!(
        scalar_bytes
            .as_ptr_range()
            .contains(&scalar.host_bytes().unwrap().as_ptr())
    );
    assert!(
        mixed_bytes
            .as_ptr_range()
            .contains(&packed.host_bytes().unwrap().as_ptr())
    );
    assert_eq!(
        packed.host_bytes().unwrap().as_ptr(),
        repeated.host_bytes().unwrap().as_ptr()
    );
    drop(weights);
    drop(scalar_bytes);
    drop(mixed_bytes);
    drop(packed);
    assert_eq!(scalar.host_bytes(), Some([42].as_slice()));
    assert_eq!(&repeated.host_bytes().unwrap()[..4], &[0, 1, 2, 3]);
}

#[test]
fn rejected_indexes_and_late_shard_failures_release_all_unpublished_owners() {
    for index in [b"{}".as_slice(), INDEX] {
        let mut inputs = shards();
        inputs.insert("scalar.safetensors".into(), Arc::from([0u8; 7]));
        let owners: Vec<_> = inputs.values().map(Arc::downgrade).collect();
        assert!(IndexedWeights::from_shards(index, inputs).is_err());
        assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    }
}

#[test]
fn publisher_totals_and_repeated_lookups_do_not_change_payload_accounting() {
    let index = std::str::from_utf8(INDEX).unwrap().replacen('{',
        r#"{"metadata":{"total_size":18446744073709551615},"extra":[null,true,-1,1.25,{"label":"kept as provenance"}],"#,
        1);
    let weights = IndexedWeights::from_shards(index.as_bytes(), shards()).unwrap();
    let first = weights.tensor("packed").unwrap();
    let second = weights.tensor("packed").unwrap();
    assert_eq!(weights.resident_bytes(), 82);
    assert_eq!(first.byte_len(), 64);
    assert_eq!(second.byte_len(), 64);
    assert_eq!(weights.tensor("empty").unwrap().byte_len(), 0);
}

#[test]
fn invalid_scalar_storage_and_empty_tensor_names_fail_with_shard_context() {
    for header in [
        r#"{"answer":{"dtype":"BOOL","shape":[],"data_offsets":[0,1]}}"#,
        r#"{"":{"dtype":"U8","shape":[],"data_offsets":[0,1]}}"#,
        r#"{"answer":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}"#,
    ] {
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.push(42);
        let mut inputs = shards();
        inputs.insert("scalar.safetensors".into(), Arc::from(bytes));
        let error = IndexedWeights::from_shards(INDEX, inputs).unwrap_err();
        let inference_engine::EngError::Checkpoint(message) = error else {
            panic!("expected a checkpoint error");
        };
        assert!(message.contains("scalar.safetensors"), "{message}");
    }
}

fn shard_with_header(header: &str, payload: &[u8]) -> Arc<[u8]> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    Arc::from(bytes)
}

#[test]
fn a_shard_header_with_a_repeated_tensor_name_names_the_shard() {
    let index = br#"{"weight_map":{"a":"dup.safetensors"}}"#;
    let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
    let shards = BTreeMap::from([(
        "dup.safetensors".to_owned(),
        shard_with_header(header, &[0; 8]),
    )]);
    let error = IndexedWeights::from_shards(index, shards).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("dup.safetensors"), "{message}");
    assert!(
        message.contains(r#"duplicate JSON member "a""#),
        "{message}"
    );
}

#[test]
fn multi_shard_escape_rejection_releases_every_supplied_owner() {
    let escape = "../outside.safetensors";
    let index = serde_json::to_vec(&serde_json::json!({
        "weight_map": {
            "answer": "scalar.safetensors",
            "packed": "mixed.safetensors",
            "leak": escape
        }
    }))
    .unwrap();
    let mut inputs = shards();
    inputs.insert(
        escape.into(),
        Arc::from(include_bytes!("fixtures/scalar.safetensors").as_slice()),
    );
    let owners: Vec<_> = inputs.values().map(Arc::downgrade).collect();
    let error = IndexedWeights::from_shards(&index, inputs).unwrap_err();
    let inference_engine::EngError::Checkpoint(message) = error else {
        panic!("expected a checkpoint error");
    };
    assert!(message.contains("leak"), "{message}");
    assert!(message.contains(escape), "{message}");
    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
}
