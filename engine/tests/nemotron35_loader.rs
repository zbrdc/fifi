use std::{collections::BTreeMap, fs, path::Path};

use inference_engine::{
    EngError,
    family::nemotron35::{
        exec::WeightSource,
        loader::{ExpertIndex, ExpertProjection, NemotronLoader},
        runner::LoaderWeightSource,
    },
    tensor::{Dtype, Tensor},
    weights::nemotron_shards::NemotronShardReader,
};
use serde_json::{Value, json};

const NORM: &str = "language_model.backbone.norm_f.weight";
const PROJ: &str = "language_model.backbone.layers.0.mixer.in_proj.weight";
const EXPERT: &str = "language_model.backbone.layers.1.mixer.experts.0";
const SHARED: &str = "language_model.backbone.layers.1.mixer.shared_experts";
const CORRECTION: &str = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias";

#[derive(Default)]
struct Shard {
    header: BTreeMap<String, Value>,
    bytes: Vec<u8>,
}

impl Shard {
    fn add(&mut self, name: &str, dtype: &str, shape: &[u32], bytes: &[u8]) {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        self.header.insert(
            name.into(),
            json!({
                "dtype": dtype, "shape": shape, "data_offsets": [start, self.bytes.len()]
            }),
        );
    }

    fn write(&self, dir: &Path, name: &str, index: &mut BTreeMap<String, String>) {
        let header = serde_json::to_vec(&self.header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(&self.bytes);
        fs::write(dir.join(name), bytes).unwrap();
        for key in self.header.keys() {
            index.insert(key.clone(), name.into());
        }
    }
}

fn fixture(change: impl FnOnce(&mut Shard)) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        include_bytes!("fixtures/nemotron35/config.json"),
    )
    .unwrap();
    let mut dense = Shard::default();
    dense.add(NORM, "BF16", &[2], &[0x80, 0x3f, 0x00, 0xc0]);
    dense.add(PROJ, "F8_E4M3", &[1, 4], &[0x38, 0x40, 0xb8, 0x30]);
    dense.add(
        &PROJ.replace(".weight", ".weight_scale"),
        "F32",
        &[1],
        &2.0f32.to_le_bytes(),
    );
    dense.add(
        &PROJ.replace(".weight", ".input_scale"),
        "F32",
        &[],
        &0.25f32.to_le_bytes(),
    );
    let mut experts = Shard::default();
    for base in [EXPERT, SHARED] {
        for which in ["up_proj", "down_proj"] {
            let base = format!("{base}.{which}");
            experts.add(
                &format!("{base}.weight"),
                "U8",
                &[1, 8],
                &[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe],
            );
            experts.add(&format!("{base}.weight_scale"), "F8_E4M3", &[1, 1], &[0x40]);
            experts.add(
                &format!("{base}.weight_scale_2"),
                "F32",
                &[],
                &0.5f32.to_le_bytes(),
            );
            experts.add(
                &format!("{base}.input_scale"),
                "F32",
                &[1],
                &0.125f32.to_le_bytes(),
            );
        }
    }
    change(&mut experts);
    let mut index = BTreeMap::new();
    dense.write(dir.path(), "dense.safetensors", &mut index);
    experts.write(dir.path(), "experts.safetensors", &mut index);
    // An unaccessed shard must never be opened.
    index.insert(
        "language_model.backbone.embeddings.weight".into(),
        "absent.safetensors".into(),
    );
    fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": index})).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn loader_weight_source_exposes_routed_mapping() {
    let mut payload_bytes = 0usize;
    let dir = fixture(|shard| {
        *shard = Shard::default();
        for expert in 0..512 {
            for (which, rows, columns) in [
                ("up_proj", 2688usize, 1024usize),
                ("down_proj", 1024usize, 2688usize),
            ] {
                let base =
                    format!("language_model.backbone.layers.1.mixer.experts.{expert}.{which}");
                for (suffix, dtype, shape, length) in [
                    ("weight", "U8", vec![rows, columns / 2], rows * columns / 2),
                    (
                        "weight_scale",
                        "F8_E4M3",
                        vec![rows, columns / 16],
                        rows * columns / 16,
                    ),
                    ("weight_scale_2", "F32", vec![], 4),
                ] {
                    let start = payload_bytes;
                    payload_bytes += length;
                    shard.header.insert(
                        format!("{base}.{suffix}"),
                        json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload_bytes]}),
                    );
                }
            }
        }
    });
    // Mapping reads metadata only; a sparse payload keeps the pinned geometry without allocating it.
    let file = fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("experts.safetensors"))
        .unwrap();
    file.set_len(file.metadata().unwrap().len() + payload_bytes as u64)
        .unwrap();
    let expected = NemotronLoader::open(dir.path(), 1)
        .unwrap()
        .expert_layer_mapping(1)
        .expect("fixture has complete routed ranges");
    assert_eq!(expected.experts.len(), 512);
    assert_eq!(
        (expected.experts[0].up.rows, expected.experts[0].up.columns),
        (2688, 1024)
    );
    let mut source = LoaderWeightSource::open(dir.path(), 1).unwrap();
    let actual = source
        .expert_layer_mapping(1)
        .expect("real WeightSource exposes routed mapping");
    assert_eq!(actual, expected);
    source
        .release_routed_experts(1, &[511, 1, 1])
        .expect("real mapped adapter releases selected expert IDs, including duplicates");
    source
        .release_expert_layer(1)
        .expect("whole-layer release remains available");
    assert!(matches!(
        source.release_routed_experts(1, &[0, 512]),
        Err(EngError::Checkpoint(_))
    ));
    assert!(source.release_routed_experts(0, &[0]).is_err());
    assert!(source.release_routed_experts(u32::MAX, &[0]).is_err());
    source
        .release_routed_experts(u32::MAX, &[])
        .expect("empty selection is a no-op");
    assert!(source.expert_layer_mapping(u32::MAX).is_err());
}

fn floats(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::F32);
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn decodes_tiny_bf16_fp8_and_nvfp4_from_two_lazy_shards() {
    let dir = fixture(|_| {});
    let mut loader = NemotronLoader::open(dir.path(), 2).unwrap();
    let (norm, scale) = loader.dense_weight(NORM).unwrap();
    assert_eq!(norm.shape(), [2]);
    assert_eq!(floats(&norm), [1.0, -2.0]);
    assert_eq!(scale, None);
    let (projection, scale) = loader.dense_weight(PROJ).unwrap();
    assert_eq!(projection.shape(), [1, 4]);
    assert_eq!(floats(&projection), [2.0, 4.0, -2.0, 1.0]);
    assert_eq!(scale, Some(0.25));
    for expert in [ExpertIndex::Routed(0), ExpertIndex::Shared] {
        let weight = loader
            .expert_block(1, expert, ExpertProjection::Up)
            .unwrap();
        assert_eq!(weight.shape(), [1, 16]);
        assert_eq!(
            floats(&weight),
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0
            ]
        );
    }
    assert!(loader.index().shard_for_tensor("missing").is_none());
    assert!(matches!(
        loader.dense_weight("missing"),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn cache_reuses_storage_refreshes_lru_and_evicts_before_decode() {
    let dir = fixture(|_| {});
    let mut loader = NemotronLoader::open(dir.path(), 2).unwrap();
    let up = loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up)
        .unwrap();
    let down = loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Down)
        .unwrap();
    let hit = loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up)
        .unwrap();
    assert_eq!(
        up.host_bytes().unwrap().as_ptr(),
        hit.host_bytes().unwrap().as_ptr()
    );
    assert_eq!(loader.eviction_count(), 0);
    loader
        .expert_block(1, ExpertIndex::Shared, ExpertProjection::Up)
        .unwrap();
    assert_eq!(loader.eviction_count(), 1);
    let up_hit = loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up)
        .unwrap();
    assert_eq!(
        up.host_bytes().unwrap().as_ptr(),
        up_hit.host_bytes().unwrap().as_ptr()
    );
    let down_new = loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Down)
        .unwrap();
    assert_ne!(
        down.host_bytes().unwrap().as_ptr(),
        down_new.host_bytes().unwrap().as_ptr()
    );
    assert_eq!(loader.eviction_count(), 2);
    assert!(matches!(
        NemotronLoader::open(dir.path(), 0),
        Err(EngError::Checkpoint(_))
    ));
    assert!(matches!(
        NemotronLoader::open(dir.path(), usize::MAX),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn retains_open_shards_across_cache_misses() {
    let dir = fixture(|_| {});
    let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
    loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up)
        .unwrap();
    fs::rename(
        dir.path().join("experts.safetensors"),
        dir.path().join("moved.safetensors"),
    )
    .unwrap();
    loader
        .expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Down)
        .unwrap();
}

#[test]
fn rejects_malformed_expert_metadata_and_scales_as_checkpoint_errors() {
    for (part, member, value) in [
        ("weight", "dtype", json!("F8_E4M3")),
        ("weight", "shape", json!([8])),
        ("weight_scale", "dtype", json!("U8")),
        ("weight_scale", "shape", json!([1])),
        ("weight_scale_2", "dtype", json!("U32")),
        ("input_scale", "shape", json!([1, 1])),
    ] {
        let dir = fixture(|s| {
            s.header
                .get_mut(&format!("{EXPERT}.up_proj.{part}"))
                .unwrap()[member] = value
        });
        let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
        assert!(
            matches!(
                loader.expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up),
                Err(EngError::Checkpoint(_))
            ),
            "{part}.{member}"
        );
    }
    let dir = fixture(|s| {
        let info = &s.header[&format!("{EXPERT}.up_proj.weight_scale_2")];
        let start = info["data_offsets"][0].as_u64().unwrap() as usize;
        s.bytes[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    });
    let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
    assert!(matches!(
        loader.expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up),
        Err(EngError::Checkpoint(_))
    ));
    assert!(matches!(
        loader.expert_block(0, ExpertIndex::Routed(0), ExpertProjection::Up),
        Err(EngError::Checkpoint(_))
    ));
    assert!(matches!(
        loader.expert_block(1, ExpertIndex::Routed(512), ExpertProjection::Up),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn rejects_missing_companions_and_nonfinite_group_scales() {
    for part in ["weight_scale_2", "input_scale"] {
        let dir = fixture(|_| {});
        let path = dir.path().join("model.safetensors.index.json");
        let mut index: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        index["weight_map"]
            .as_object_mut()
            .unwrap()
            .remove(&format!("{EXPERT}.up_proj.{part}"));
        fs::write(path, serde_json::to_vec(&index).unwrap()).unwrap();
        let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
        assert!(matches!(
            loader.expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up),
            Err(EngError::Checkpoint(_))
        ));
    }
    let dir = fixture(|s| {
        let offset = s.header[&format!("{EXPERT}.up_proj.weight_scale")]["data_offsets"][0]
            .as_u64()
            .unwrap() as usize;
        s.bytes[offset] = 0x7f;
    });
    let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
    assert!(matches!(
        loader.expert_block(1, ExpertIndex::Routed(0), ExpertProjection::Up),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn gathers_all_routed_and_shared_activation_scales_without_weights() {
    let dir = fixture(|s| {
        for expert in 1..512 {
            for which in ["up_proj", "down_proj"] {
                s.add(&format!("language_model.backbone.layers.1.mixer.experts.{expert}.{which}.input_scale"), "F32", &[], &0.25f32.to_le_bytes());
            }
        }
    });
    let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
    let facts = loader.expert_scale_facts(1).unwrap();
    assert_eq!(facts.routed.len(), 512);
    assert_eq!(facts.routed[0].up_input_scale, 0.125);
    assert_eq!(facts.routed[0].down_input_scale, 0.125);
    assert_eq!(facts.routed[511].up_input_scale, 0.25);
    assert_eq!(facts.routed[511].down_input_scale, 0.25);
    assert_eq!(facts.shared.up_input_scale, 0.125);
    assert_eq!(facts.shared.down_input_scale, 0.125);
    assert_eq!(loader.eviction_count(), 0);
}

#[test]
fn loads_pinned_real_model_when_configured() {
    let Some(dir) = std::env::var_os("NEM35_MODEL_DIR") else {
        return;
    };
    let mut loader = NemotronLoader::open(&dir, 1).unwrap();
    assert_eq!(loader.index().tensor_count(), 169_505);
    assert_eq!(
        loader
            .index()
            .shard_for_tensor("language_model.backbone.embeddings.weight"),
        Some("model-00001-of-00088.safetensors")
    );
    assert_eq!(loader.index().shard_for_tensor("missing"), None);
    let (embedding, scale) = loader
        .dense_weight("language_model.backbone.embeddings.weight")
        .unwrap();
    assert_eq!(embedding.dtype(), Dtype::F32);
    assert_eq!(embedding.shape(), [131_072, 4096]);
    assert_eq!(scale, None);
    drop(embedding);
    let (projection, scale) = loader.dense_weight(PROJ).unwrap();
    assert_eq!(projection.dtype(), Dtype::F32);
    assert_eq!(projection.shape(), [18_560, 4096]);
    assert!(scale.unwrap().is_finite() && scale.unwrap() > 0.0);
    drop(projection);
    for (expert, up_shape, down_shape) in [
        (ExpertIndex::Routed(0), [2688, 1024], [1024, 2688]),
        (ExpertIndex::Shared, [5376, 4096], [4096, 5376]),
    ] {
        let base = if expert == ExpertIndex::Shared {
            SHARED
        } else {
            EXPERT
        };
        for (which, name, shape) in [
            (ExpertProjection::Up, "up_proj", up_shape),
            (ExpertProjection::Down, "down_proj", down_shape),
        ] {
            let key = format!("{base}.{name}.weight");
            let shard_name = loader.index().shard_for_tensor(&key).unwrap();
            let shard = NemotronShardReader::open(Path::new(&dir).join(shard_name)).unwrap();
            let metadata = shard.tensor(&key).unwrap();
            assert_eq!(metadata.dtype(), "U8");
            assert_eq!(
                metadata.shape(),
                [u64::from(shape[0]), u64::from(shape[1]) / 2]
            );
            let decoded = loader.expert_block(1, expert, which).unwrap();
            assert_eq!(decoded.dtype(), Dtype::F32);
            assert_eq!(decoded.shape(), shape);
            let numel = metadata.byte_len() * 2;
            assert_eq!(u64::from(decoded.shape()[0]), numel / u64::from(shape[1]));
            assert_eq!(decoded.byte_len(), numel * 4);
            eprintln!(
                "{key}: stored {:?}, decoded {:?}, F32 bytes {}",
                metadata.shape(),
                decoded.shape(),
                decoded.byte_len()
            );
        }
    }
    assert_eq!(loader.eviction_count(), 3);
    let facts = loader.expert_scale_facts(1).unwrap();
    assert_eq!(facts.routed.len(), 512);
    assert!(
        facts
            .routed
            .iter()
            .chain(std::iter::once(&facts.shared))
            .all(|s| s.up_input_scale.is_finite()
                && s.up_input_scale > 0.0
                && s.down_input_scale.is_finite()
                && s.down_input_scale > 0.0)
    );
}

#[test]
fn decodes_f32_dense_weights_and_rejects_nonfinite_values() {
    for (case, values) in [("finite", [-1.5f32, 2.5]), ("nonfinite", [f32::NAN, 2.5])] {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            include_bytes!("fixtures/nemotron35/config.json"),
        )
        .unwrap();
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut dense = Shard::default();
        dense.add(CORRECTION, "F32", &[2], &payload);
        let mut index = BTreeMap::new();
        dense.write(dir.path(), "dense.safetensors", &mut index);
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": index})).unwrap(),
        )
        .unwrap();
        let mut loader = NemotronLoader::open(dir.path(), 1).unwrap();
        match loader.dense_weight(CORRECTION) {
            Ok((tensor, scale)) => {
                assert_eq!(case, "finite");
                assert_eq!(tensor.shape(), [2]);
                assert_eq!(floats(&tensor), [-1.5, 2.5]);
                assert_eq!(scale, None);
            }
            Err(EngError::Checkpoint(_)) => assert_eq!(case, "nonfinite"),
            Err(other) => panic!("{case}: unexpected error {other:?}"),
        }
    }
}
