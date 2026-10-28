use inference_engine::{
    EngError,
    weights::nemotron_index::{NemotronComponent, NemotronIndex},
};
use serde_json::Value;

const INDEX_EXCERPT: &[u8] = include_bytes!("fixtures/nemotron35/index-excerpt.json");

#[test]
fn hand_distilled_index_excerpt_covers_component_prefixes() {
    let excerpt: Value = serde_json::from_slice(INDEX_EXCERPT).expect("excerpt JSON");
    let weight_map = excerpt["weight_map"]
        .as_object()
        .expect("excerpt weight_map");
    assert_eq!(weight_map.len(), 8);

    let mut counts = [0u64; 5];
    for tensor_name in weight_map.keys() {
        let index = match NemotronComponent::classify_tensor_name(tensor_name) {
            NemotronComponent::Text => 0,
            NemotronComponent::Mtp => 1,
            NemotronComponent::Vision => 2,
            NemotronComponent::Connector => 3,
            NemotronComponent::Ambiguous => 4,
        };
        counts[index] = counts[index]
            .checked_add(1)
            .expect("fixture component count");
    }
    assert_eq!(counts, [2, 1, 1, 1, 3]);
}

#[test]
fn rejects_duplicate_json_member_names_at_root_and_nested_levels() {
    for index in [
        br#"{"metadata":{"total_size":1},"metadata":{"total_size":2},"weight_map":{"key":"part.safetensors"}}"#.as_slice(),
        br#"{"metadata":{"total_size":1,"total_size":2},"weight_map":{"key":"part.safetensors"}}"#.as_slice(),
        br#"{"metadata":{"total_size":1},"weight_map":{"key":"part.safetensors","key":"other.safetensors"}}"#.as_slice(),
    ] {
        assert!(
            matches!(NemotronIndex::from_json(index), Err(EngError::Checkpoint(message)) if message.contains("duplicate JSON member")),
            "duplicate member was accepted"
        );
    }
}

#[test]
fn rejects_empty_weight_map_and_malformed_shard_names() {
    let empty = br#"{"metadata":{"total_size":1},"weight_map":{}}"#;
    assert!(matches!(
        NemotronIndex::from_json(empty),
        Err(EngError::Checkpoint(_))
    ));

    for shard in [
        "",
        "nested/part.safetensors",
        "nested\\part.safetensors",
        "part.bin",
    ] {
        let index = serde_json::json!({
            "metadata": { "total_size": 1 },
            "weight_map": { "tensor": shard }
        });
        let bytes = serde_json::to_vec(&index).expect("index JSON");
        assert!(
            matches!(
                NemotronIndex::from_json(&bytes),
                Err(EngError::Checkpoint(_))
            ),
            "malformed shard name {shard:?} was accepted"
        );
    }
}

#[test]
fn parses_pinned_nemotron_nvfp4_index_and_writes_inventory_when_configured() {
    let Some(path) = std::env::var_os("NEM35_INDEX") else {
        return;
    };
    let bytes = std::fs::read(path).expect("pinned Nemotron index");
    let index = NemotronIndex::from_json(&bytes).expect("pinned Nemotron index parses");

    assert_eq!(index.tensor_count(), 169_505);
    assert_eq!(index.shard_key_counts().len(), 88);
    assert_eq!(
        index.shard_key_counts()["model-00001-of-00088.safetensors"],
        410
    );
    assert_eq!(
        index
            .shard_key_counts()
            .values()
            .try_fold(0u64, |total, count| total.checked_add(*count)),
        Some(169_505)
    );
    assert_eq!(index.component_counts().text, 164_986);
    assert_eq!(index.component_counts().vision, 391);
    assert_eq!(index.component_counts().connector, 2);
    assert_eq!(index.component_counts().mtp, 4_120);
    assert_eq!(index.component_counts().ambiguous, 6);

    let directory = tempfile::tempdir().expect("inventory directory");
    let output = directory.path().join("inventory.json");
    index.write_inventory(&output).expect("write inventory");
    let inventory: Vec<Value> =
        serde_json::from_slice(&std::fs::read(output).expect("read inventory"))
            .expect("inventory JSON");
    assert_eq!(inventory.len(), 169_505);
    assert_eq!(
        inventory[0]["tensor_name"],
        "language_model.backbone.embeddings.weight"
    );
    assert_eq!(inventory[0]["shard"], "model-00001-of-00088.safetensors");
    assert_eq!(inventory[0]["component"], "text");
}
