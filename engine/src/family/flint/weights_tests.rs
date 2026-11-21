use std::{collections::BTreeMap, fs::File, io::Write};

use super::*;
use serde_json::json;

fn shard(entries: Vec<(&str, Dtype, Vec<u32>, Vec<u8>)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        let start = data.len();
        data.extend(bytes);
        header.insert(
            name.into(),
            json!({
                "dtype": match dtype { Dtype::U8 => "U8", Dtype::Bf16 => "BF16", _ => "F32" },
                "shape": shape, "data_offsets": [start, data.len()]
            }),
        );
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(data);
    bytes
}

fn bf16(value: f32, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|_| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn tiny(logical: Vec<u32>, bits: u8) -> (FlintCheckpoint, Vec<u8>) {
    let mut payload_shape = logical.clone();
    let last = payload_shape.len() - 1;
    payload_shape[last] = logical[last] * u32::from(bits) / 8;
    let mut scale_shape = logical.clone();
    scale_shape[last] /= 64;
    let count = payload_shape.iter().product::<u32>() as usize;
    let payload: Vec<_> = (0..count)
        .map(|i| ((i + i / payload_shape[last] as usize) % 256) as u8)
        .collect();
    let scales = scale_shape.iter().product::<u32>() as usize;
    let scale_bytes = (0..scales)
        .flat_map(|i| bf16(0.5 + (i / scale_shape[last] as usize) as f32 * 0.125, 1))
        .collect();
    let bias_bytes = (0..scales)
        .flat_map(|i| bf16(-2.0 + (i / scale_shape[last] as usize) as f32 * 0.25, 1))
        .collect();
    let bytes = shard(vec![
        (
            "model.embed_tokens.weight",
            Dtype::U8,
            payload_shape,
            payload.clone(),
        ),
        (
            "model.embed_tokens.scales",
            Dtype::Bf16,
            scale_shape.clone(),
            scale_bytes,
        ),
        (
            "model.embed_tokens.biases",
            Dtype::Bf16,
            scale_shape,
            bias_bytes,
        ),
        ("norm", Dtype::Bf16, vec![2], bf16(1.0, 2)),
    ]);
    let routes: BTreeMap<_, _> = [
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
        "norm",
    ]
    .into_iter()
    .map(|name| (name, "tiny.safetensors"))
    .collect();
    let index = serde_json::to_vec(&json!({"weight_map": routes})).unwrap();
    let weights = StreamedCheckpoint::open(
        &index,
        vec![(
            "tiny.safetensors".into(),
            Box::new(std::io::Cursor::new(bytes.clone())),
            bytes.len() as u64,
        )],
        HEADER_LIMIT,
    )
    .unwrap();
    let mut profile = Profile::default();
    profile
        .packed
        .insert("model.embed_tokens".into(), (logical, bits));
    (
        FlintCheckpoint {
            weights,
            profile,
            #[cfg(all(
                feature = "native-fault-qualification",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            records: None,
        },
        payload,
    )
}

#[test]
fn packed_api_preserves_2d_and_3d_bytes_and_affine_metadata() {
    for shape in [vec![2, 64], vec![2, 3, 128]] {
        let (checkpoint, bytes) = tiny(shape.clone(), 4);
        let packed = checkpoint.packed("model.embed_tokens").unwrap();
        assert_eq!(packed.logical_shape(), shape);
        assert_eq!(packed.payload().dtype(), Dtype::U32);
        assert_eq!(packed.payload().host_bytes().unwrap(), bytes);
        assert_eq!(packed.quantization().bits, 4);
        assert_eq!(packed.quantization().group_size, 64);
        assert_eq!(packed.scales().dtype(), Dtype::Bf16);
        assert_eq!(packed.scales().host_bytes().unwrap()[..2], [0, 0x3f]);
        assert_eq!(
            packed.biases().unwrap().host_bytes().unwrap()[..2],
            [0, 0xc0]
        );
        // The first bytes 00 01 02 03 are the little-endian word 0x03020100.
        let word = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!(word, 0x03020100);
        let values: Vec<_> = (0..8)
            .map(|i| ((word >> (4 * i)) & 15) as f32 * 0.5 - 2.0)
            .collect();
        assert_eq!(values, [-2.0, -2.0, -1.5, -2.0, -1.0, -2.0, -0.5, -2.0]);
    }
}

#[test]
fn embedding_api_gathers_int8_rows_in_request_order() {
    let (checkpoint, bytes) = tiny(vec![4, 2048], 8);
    let rows = checkpoint.embedding_rows(&[3, 0, 3]).unwrap();
    assert_eq!(rows.logical_shape(), [3, 2048]);
    assert_eq!(rows.payload().shape(), [3, 512]);
    assert_eq!(rows.scales().shape(), [3, 32]);
    assert_eq!(rows.quantization().bits, 8);
    let expected: Vec<_> = [3, 0, 3]
        .into_iter()
        .flat_map(|id| bytes[id * 2048..(id + 1) * 2048].iter().copied())
        .collect();
    assert_eq!(rows.payload().host_bytes().unwrap(), expected);
    let expected_scales: Vec<_> = [0.875, 0.5, 0.875]
        .into_iter()
        .flat_map(|scale| bf16(scale, 32))
        .collect();
    let expected_biases: Vec<_> = [-1.25, -2.0, -1.25]
        .into_iter()
        .flat_map(|bias| bf16(bias, 32))
        .collect();
    assert_eq!(rows.scales().host_bytes().unwrap(), expected_scales);
    assert_eq!(
        rows.biases().unwrap().host_bytes().unwrap(),
        expected_biases
    );
    // Unsigned INT8 row 3 starts with q=3, scale=0.875, additive bias=-1.25.
    let first = rows.payload().host_bytes().unwrap()[0] as f32 * 0.875 - 1.25;
    assert_eq!(first, 1.375);
    assert!(checkpoint.embedding_rows(&[]).is_err());
    assert!(checkpoint.embedding_rows(&[4]).is_err());
    assert!(checkpoint.embedding_rows(&[u32::MAX]).is_err());
    assert_eq!(
        checkpoint.tensor("norm").unwrap().host_bytes().unwrap(),
        bf16(1.0, 2)
    );
    assert!(checkpoint.tensor("absent").is_err());
    assert!(checkpoint.packed("absent").is_err());
}

#[test]
fn inventory_matches_all_1751_publisher_header_records() {
    use sha2::{Digest, Sha256};
    let profile = Profile::pinned();
    assert_eq!(profile.tensors.len(), 1751);
    let mut manifest = String::new();
    for (name, (dtype, shape)) in profile.tensors {
        let dtype = match dtype {
            Dtype::U8 => "U8",
            Dtype::Bf16 => "BF16",
            _ => panic!("unexpected dtype"),
        };
        let dims = shape
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        manifest.push_str(&format!("{name}\t{dtype}\t{dims}\n"));
    }
    // Frozen independently from the five real headers at revision 3b93f905.
    assert_eq!(
        format!("{:x}", Sha256::digest(manifest.as_bytes())),
        "71faa325abcaadef1551935b181ff67baffc76995e066f3d10acc412a379515b"
    );
}

type Entry = (String, Dtype, Vec<u32>);

fn sparse_checkpoint(change: impl FnOnce(&mut [Vec<Entry>])) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("config.json"),
        include_bytes!("../../../testdata/flint/mlx-config.json"),
    )
    .unwrap();
    let mut shards: Vec<Vec<Entry>> = vec![Vec::new(); 5];
    for (index, (name, (dtype, shape))) in Profile::pinned().tensors.into_iter().enumerate() {
        shards[index % 5].push((name, dtype, shape));
    }
    change(&mut shards);
    for (index, entries) in shards.into_iter().enumerate() {
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, dtype, shape) in entries {
            let len = shape.iter().map(|&dim| u64::from(dim)).product::<u64>() * dtype.byte_width();
            let dtype = match dtype {
                Dtype::U8 => "U8",
                Dtype::Bf16 => "BF16",
                _ => "F32",
            };
            header.insert(
                name,
                json!({"dtype": dtype, "shape": shape, "data_offsets": [end, end + len]}),
            );
            end += len;
        }
        let json = serde_json::to_vec(&header).unwrap();
        let mut file =
            File::create(root.path().join(format!("model-{index:05}.safetensors"))).unwrap();
        file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&json).unwrap();
        // Real architecture extents, sparse zero payload: no model materialization.
        file.set_len(8 + json.len() as u64 + end).unwrap();
    }
    root
}

// Fails if `FlintCheckpoint::open` stops guarding the root directory.
#[cfg(target_os = "linux")]
#[test]
fn open_refuses_a_root_on_a_network_mount() {
    let root = sparse_checkpoint(|_| {});
    let refused = crate::storage_guard::seam::with_nfs_at(root.path(), || {
        FlintCheckpoint::open(root.path()).err().expect("refused")
    });
    assert!(
        refused.describe().contains("must be on local storage"),
        "{}",
        refused.describe()
    );
    assert!(FlintCheckpoint::open(root.path()).is_ok());
}

// Fails if `FlintCheckpoint::open` checks only the root and not each shard:
// the root is local and one shard is a link to a file on the network mount.
#[cfg(target_os = "linux")]
#[test]
fn open_refuses_a_shard_linked_into_a_network_mount() {
    let root = sparse_checkpoint(|_| {});
    let share = tempfile::tempdir().unwrap();
    let shard = root.path().join("model-00002.safetensors");
    let moved = share.path().join("model-00002.safetensors");
    std::fs::rename(&shard, &moved).unwrap();
    std::os::unix::fs::symlink(&moved, &shard).unwrap();
    let refused = crate::storage_guard::seam::with_nfs_at(share.path(), || {
        FlintCheckpoint::open(root.path()).err().expect("refused")
    });
    assert!(
        refused.describe().contains("must be on local storage"),
        "{}",
        refused.describe()
    );
    assert!(FlintCheckpoint::open(root.path()).is_ok());
}

#[test]
fn open_checks_complete_profile_without_reading_payloads() {
    let root = sparse_checkpoint(|_| {});
    let checkpoint = FlintCheckpoint::open(root.path()).unwrap();
    let norm = checkpoint.tensor("model.norm.weight").unwrap();
    assert_eq!(norm.shape(), [2048]);
    assert_eq!(norm.dtype(), Dtype::Bf16);
    let k = checkpoint
        .packed("model.layers.0.self_attn.k_proj")
        .unwrap();
    assert_eq!(k.logical_shape(), [512, 2048]);
    let embedding = checkpoint.embedding_rows(&[150143, 0]).unwrap();
    assert_eq!(embedding.logical_shape(), [2, 2048]);
    assert_eq!(embedding.quantization().bits, 8);
    assert!(checkpoint.tensor("model.embed_tokens.weight").is_err());
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn disk_experts_match_resident_grouped_predictions() {
    use super::super::{
        mlx_expert_cache::LayerExpertCache,
        mlx_forward::{disk_experts, grouped_experts},
        mlx_model::SwiGlu,
    };
    use crate::kernels::MlxKernels;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.safetensors");
    let mut profile = Profile::default();
    let mut entries = Vec::new();
    for (projection, shift) in [("gate_proj", 1), ("up_proj", 3), ("down_proj", 5)] {
        let stem = format!("experts.{projection}");
        profile.packed.insert(stem.clone(), (vec![256, 64, 64], 4));
        let payload = (0..256 * 64 * 32)
            .map(|i| ((i + i / 2048 + shift) % 256) as u8)
            .collect();
        entries.push((
            format!("{stem}.weight"),
            Dtype::U8,
            vec![256, 64, 32],
            payload,
        ));
        entries.push((
            format!("{stem}.scales"),
            Dtype::Bf16,
            vec![256, 64, 1],
            bf16(0.125, 256 * 64),
        ));
        entries.push((
            format!("{stem}.biases"),
            Dtype::Bf16,
            vec![256, 64, 1],
            bf16(-0.5, 256 * 64),
        ));
    }
    let index = serde_json::to_vec(&json!({"weight_map": entries.iter()
        .map(|(name, _, _, _)| (name.as_str(), "experts.safetensors")).collect::<BTreeMap<_, _>>() }))
        .unwrap();
    let bytes = shard(
        entries
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (name.as_str(), *dtype, shape.clone(), bytes.clone())
            })
            .collect(),
    );
    std::fs::write(&path, &bytes).unwrap();
    let file = File::open(&path).unwrap();
    let weights = StreamedCheckpoint::open(
        &index,
        vec![(
            "experts.safetensors".into(),
            Box::new(file),
            bytes.len() as u64,
        )],
        HEADER_LIMIT,
    )
    .unwrap();
    let checkpoint = FlintCheckpoint {
        profile,
        weights,
        #[cfg(feature = "native-fault-qualification")]
        records: None,
    };
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let mut load = |projection| {
        k.import_packed(&checkpoint.packed(&format!("experts.{projection}")).unwrap())
            .unwrap()
    };
    let full = SwiGlu {
        gate: load("gate_proj"),
        up: load("up_proj"),
        down: load("down_proj"),
    };
    for rows in [1, 2] {
        let x = k
            .import_tensor(
                &Tensor::try_from_host(
                    Dtype::Bf16,
                    vec![rows, 64],
                    bf16(0.375, rows as usize * 64),
                )
                .unwrap(),
            )
            .unwrap();
        let ids = [19_u32, 255, 8, 0, 3, 42, 255, 91, 2, 42, 17, 19];
        let ids = Tensor::try_from_host(
            Dtype::U32,
            vec![6 * rows],
            ids[..6 * rows as usize]
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect(),
        )
        .unwrap();
        let ids = k.import_tensor(&ids).unwrap();
        let ids = k.bound_indices(&ids, 256).unwrap();
        let expected = grouped_experts(&mut k, &full, &x, &ids).unwrap();
        let expected = k.readback(&expected).unwrap();
        let streamed = disk_experts(&mut k, &checkpoint, "experts", &x, &ids, false).unwrap();
        let streamed = k.readback(&streamed).unwrap();
        assert_eq!(
            streamed.host_bytes().unwrap(),
            expected.host_bytes().unwrap()
        );
        assert!(streamed.host_bytes().unwrap().iter().any(|&byte| byte != 0));
    }

    let x = k
        .import_tensor(&Tensor::try_from_host(Dtype::Bf16, vec![1, 64], bf16(0.375, 64)).unwrap())
        .unwrap();
    let mut cache = LayerExpertCache::new(&checkpoint, "experts", 6).unwrap();
    // First fill, a repeat with different ranks, an eviction, and a read of the old victim.
    for experts in [
        [19_u32, 255, 8, 0, 3, 42],
        [42, 3, 0, 8, 255, 19],
        [91, 3, 0, 8, 255, 19],
        [42, 3, 0, 8, 255, 19],
    ] {
        let ids = Tensor::try_from_host(
            Dtype::U32,
            vec![6],
            experts.iter().flat_map(|id| id.to_le_bytes()).collect(),
        )
        .unwrap();
        let ids = k.import_tensor(&ids).unwrap();
        let ids = k.bound_indices(&ids, 256).unwrap();
        let expected = grouped_experts(&mut k, &full, &x, &ids).unwrap();
        let expected = k.readback(&expected).unwrap();
        let cached = cache.evaluate(&mut k, &checkpoint, &x, &experts).unwrap();
        let cached = k.readback(&cached).unwrap();
        assert_eq!(cached.host_bytes(), expected.host_bytes());
    }
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn sparse_oversized_flint_checkpoint_streams_and_decodes_on_mlx() {
    use super::super::{mlx_model::Model, mlx_state::State};
    let root = sparse_checkpoint(|_| {});
    let source = FlintCheckpoint::open(root.path()).unwrap();
    let weights_budget = 2 << 30;
    assert!(source.mlx_weight_bytes(false) > weights_budget);
    assert!(
        super::super::mlx_model::expert_staging_bytes(source.bytes_per_routed_expert(), 1)
            + source.mlx_weight_bytes(true)
            < weights_budget
    );
    drop(source);
    let _serial = crate::native_test_lock();
    let mut model = Model::open_for_budget(root.path(), Some(weights_budget)).unwrap();
    assert_eq!(model.streamed_expert_bytes, 1_769_472);
    model.kernels.reset_peak_memory().unwrap();
    let mut state = State::new();
    assert!(
        model
            .prefill_rows(&mut state, &[1, 5], false)
            .unwrap()
            .is_none()
    );
    let logits = model.step(&mut state, 8, true).unwrap().unwrap();
    assert_eq!(logits.len(), 150_144);
    assert!(logits.iter().all(|&value| value == 0.0));
    let memory = model.kernels.memory_stats().unwrap();
    assert!(memory.peak < weights_budget, "{memory:?}");
    eprintln!(
        "Flint sparse fixture prefill and decode allocator peak {} bytes",
        memory.peak
    );
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn planner_flash_price_uses_the_forwards_geometry() {
    use super::super::{config, mlx_model::Model};
    let root = sparse_checkpoint(|_| {});
    drop(FlintCheckpoint::open(root.path()).unwrap());
    let _serial = crate::native_test_lock();
    // The load check pins the checkpoint's head_dim to PREFILL_FLASH_HEAD_DIM,
    // so the pair the planner prices flash by is the pair the forward's
    // request tensors carry; assert the pair opens the route on this device.
    let model = Model::open_for_budget(root.path(), Some(2 << 30)).unwrap();
    assert!(
        model
            .kernels
            .float_flash_prefill_available(
                super::super::mlx_executor::PREFILL_FLASH_DTYPE,
                config::PREFILL_FLASH_HEAD_DIM
            )
            .unwrap()
    );
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn flash_prefill_peak_stays_within_the_ladder_charge() {
    use super::super::{mlx_model::Model, mlx_state::State};
    let root = sparse_checkpoint(|_| {});
    drop(FlintCheckpoint::open(root.path()).unwrap());
    // The resident fixture plus the prefill charge needs ~24 GB of device
    // working set; skip on a host without room instead of driving it into
    // swap, unless the repository's require-data variable is set.
    let recommended = crate::kernels::MlxKernels::recommended_working_set_bytes();
    let need = 24 << 30;
    if recommended < need && std::env::var_os("FIFI_TEST_REQUIRE_DATA").is_none() {
        eprintln!(
            "skipped: resident fixture needs ~{need} bytes, device working set {recommended}"
        );
        return;
    }
    let _serial = crate::native_test_lock();
    // All experts resident: the bound carries no staging slack, so the
    // measured flash charge alone must cover the prefill.
    let mut model = Model::open_for_budget(root.path(), None).unwrap();
    assert_eq!(model.streamed_expert_bytes, 0);
    model.kernels.reset_peak_memory().unwrap();
    let active = model.kernels.memory_stats().unwrap().active;
    // ~29K tokens: the last chunks run near the family's 32K context limit,
    // so a composite dispatch there (rows*32*32K*4 bytes, 12+ GiB at 1,024
    // rows) cannot hide under the bound. This pins the route the planner
    // priced to the route the executor dispatches.
    let rows = 29 * 1024;
    let tokens: Vec<u32> = (0..rows).map(|i| (i % 140_000) as u32).collect();
    let mut state = State::new();
    for piece in tokens.chunks(1024) {
        assert!(
            model
                .prefill_rows(&mut state, piece, false)
                .unwrap()
                .is_none()
        );
    }
    let growth = model.kernels.memory_stats().unwrap().peak - active;
    // The KV planes this prefill materializes are growth the allocator sees
    // but the session charges, not the ladder, price; attribute them here so
    // the bound still pins the flash route alone.
    use crate::family::kv_format::{KvFormat, KvRowShape};
    let row_bytes = KvFormat::Bf16
        .layer_position_bytes(KvRowShape {
            heads_kv: 4,
            head_dim: 128,
        })
        .unwrap();
    let kv_state = 14 * rows as u64 * row_bytes + 35 * 513 * row_bytes;
    let charge = super::super::mlx_executor::FLASH_PREFILL_BASE_BYTES
        + super::super::mlx_executor::FLASH_PREFILL_ROW_BYTES * 1024
        + kv_state;
    assert!(
        growth <= charge,
        "flash prefill allocator growth {growth} exceeds the ladder charge plus KV planes {charge}: \
         the executor left the priced flash route or the kernel outgrew the measured terms"
    );
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn expert_page_staging_matches_the_packed_experts_path() {
    use super::super::{
        mlx_forward::{disk_experts, grouped_experts},
        mlx_model::SwiGlu,
    };
    use crate::kernels::MlxKernels;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.safetensors");
    let mut profile = Profile::default();
    let mut entries = Vec::new();
    for (projection, shift) in [("gate_proj", 1), ("up_proj", 3), ("down_proj", 5)] {
        let stem = format!("experts.{projection}");
        profile.packed.insert(stem.clone(), (vec![256, 64, 64], 4));
        let payload = (0..256 * 64 * 32)
            .map(|i| ((i + i / 2048 + shift) % 256) as u8)
            .collect();
        entries.push((
            format!("{stem}.weight"),
            Dtype::U8,
            vec![256, 64, 32],
            payload,
        ));
        entries.push((
            format!("{stem}.scales"),
            Dtype::Bf16,
            vec![256, 64, 1],
            bf16(0.125, 256 * 64),
        ));
        entries.push((
            format!("{stem}.biases"),
            Dtype::Bf16,
            vec![256, 64, 1],
            bf16(-0.5, 256 * 64),
        ));
    }
    // Only the width is wrong: the bad stem carries well-formed scales and
    // biases, so the layout reaches the word check and refuses 33 there.
    profile
        .packed
        .insert("experts.bad_proj".into(), (vec![256, 64, 33], 4));
    entries.push((
        "experts.bad_proj.weight".to_string(),
        Dtype::U8,
        vec![256, 64, 33],
        vec![0_u8; 256 * 64 * 33],
    ));
    entries.push((
        "experts.bad_proj.scales".to_string(),
        Dtype::Bf16,
        vec![256, 64, 1],
        bf16(0.125, 256 * 64),
    ));
    entries.push((
        "experts.bad_proj.biases".to_string(),
        Dtype::Bf16,
        vec![256, 64, 1],
        bf16(-0.5, 256 * 64),
    ));
    let index = serde_json::to_vec(&json!({"weight_map": entries.iter()
        .map(|(name, _, _, _)| (name.as_str(), "experts.safetensors")).collect::<BTreeMap<_, _>>() }))
        .unwrap();
    let bytes = shard(
        entries
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (name.as_str(), *dtype, shape.clone(), bytes.clone())
            })
            .collect(),
    );
    std::fs::write(&path, &bytes).unwrap();
    let file = File::open(&path).unwrap();
    let weights = StreamedCheckpoint::open(
        &index,
        vec![(
            "experts.safetensors".into(),
            Box::new(file),
            bytes.len() as u64,
        )],
        HEADER_LIMIT,
    )
    .unwrap();
    let checkpoint = FlintCheckpoint {
        profile,
        weights,
        #[cfg(feature = "native-fault-qualification")]
        records: None,
    };
    let page = crate::kernels::HostPages::page_size();
    let refused = crate::family::stored::StackLayout::for_byte_payload(
        &checkpoint.weights,
        "experts.bad_proj",
        affine(4),
        &[256, 64, 33],
        3,
        0,
        page,
    );
    let message = format!("{:?}", refused.expect_err("width 33 must be refused"));
    assert!(
        message.contains("33") && message.contains("not a multiple of four"),
        "{message}"
    );

    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let selected = [19_u32, 3, 40, 2, 41, 3];
    let mut experts = selected.to_vec();
    experts.sort_unstable();
    experts.dedup();
    let remapped: Vec<u8> = selected
        .iter()
        .map(|&id| experts.binary_search(&id).unwrap() as u32)
        .flat_map(|slot| slot.to_le_bytes())
        .collect();
    let remapped =
        Tensor::try_from_host(Dtype::U32, vec![selected.len() as u32], remapped).unwrap();
    let (paged, staged_ids, _pages) =
        SwiGlu::load_expert_pages(&mut k, &checkpoint, "experts", &experts, &remapped, false)
            .unwrap();
    let (whole, packed_ids) =
        SwiGlu::load_experts(&mut k, &checkpoint, "experts", &experts, &remapped).unwrap();
    let x = k
        .import_tensor(&Tensor::try_from_host(Dtype::Bf16, vec![1, 64], bf16(0.375, 64)).unwrap())
        .unwrap();
    let staged_ids = k.bound_indices(&staged_ids, experts.len() as u32).unwrap();
    let rows = grouped_experts(&mut k, &paged, &x, &staged_ids).unwrap();
    let staged = k.readback(&rows).unwrap().host_bytes().unwrap().to_vec();
    let packed_ids = k.bound_indices(&packed_ids, experts.len() as u32).unwrap();
    let rows = grouped_experts(&mut k, &whole, &x, &packed_ids).unwrap();
    let packed_path = k.readback(&rows).unwrap().host_bytes().unwrap().to_vec();
    assert_eq!(staged, packed_path);
    assert!(staged.iter().any(|&byte| byte != 0));
    let ids = Tensor::try_from_host(
        Dtype::U32,
        vec![selected.len() as u32],
        selected.iter().flat_map(|id| id.to_le_bytes()).collect(),
    )
    .unwrap();
    let ids = k.import_tensor(&ids).unwrap();
    let ids = k.bound_indices(&ids, 256).unwrap();
    let streamed = disk_experts(&mut k, &checkpoint, "experts", &x, &ids, false).unwrap();
    let streamed = k
        .readback(&streamed)
        .unwrap()
        .host_bytes()
        .unwrap()
        .to_vec();
    assert_eq!(streamed, staged);
}

#[test]
fn open_rejects_missing_duplicate_unknown_and_wrong_shape_or_dtype() {
    for case in 0..5 {
        let root = sparse_checkpoint(|shards| match case {
            0 => {
                shards[0].pop();
            }
            1 => {
                let copy = shards[0][0].clone();
                shards[1].push(copy);
            }
            2 => {
                shards[0][0].0 = "unrecognized.weight".into();
            }
            3 => {
                shards[0][0].2[0] += 1;
            }
            4 => {
                shards[0][0].1 = Dtype::F32;
            }
            _ => unreachable!(),
        });
        assert!(
            matches!(
                FlintCheckpoint::open(root.path()),
                Err(EngError::Checkpoint(_))
            ),
            "case {case}"
        );
    }
}

#[test]
fn open_requires_qualified_apple_quantization_not_bf16_reference() {
    let root = sparse_checkpoint(|_| {});
    let base = include_bytes!("../../../testdata/flint/config.json");
    std::fs::write(root.path().join("config.json"), base).unwrap();
    assert!(FlintCheckpoint::open(root.path()).is_err());
    let mlx = include_bytes!("../../../testdata/flint/mlx-config.json");
    for field in ["quantization", "mlx_export"] {
        let mut config: serde_json::Value = serde_json::from_slice(mlx).unwrap();
        config.as_object_mut().unwrap().remove(field);
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        assert!(FlintCheckpoint::open(root.path()).is_err());
    }
    let mut config: serde_json::Value = serde_json::from_slice(mlx).unwrap();
    config["quantization"]["model.embed_tokens"] = json!({"bits": 4});
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    assert!(FlintCheckpoint::open(root.path()).is_err());
}

#[test]
fn packed_api_rejects_incomplete_words_and_wrong_storage() {
    for (dtype, shape, data) in [
        (Dtype::U8, vec![2, 3], vec![0; 6]),
        (Dtype::U8, vec![3], vec![0; 3]),
        (Dtype::U8, vec![1, 1, 1, 4], vec![0; 4]),
        (Dtype::F32, vec![1, 1], vec![0; 4]),
    ] {
        let bytes = shard(vec![("x.weight", dtype, shape, data)]);
        let index = br#"{"weight_map":{"x.weight":"x.safetensors"}}"#;
        let weights = StreamedCheckpoint::open(
            index,
            vec![(
                "x.safetensors".into(),
                Box::new(std::io::Cursor::new(bytes.clone())),
                bytes.len() as u64,
            )],
            HEADER_LIMIT,
        )
        .unwrap();
        let mut profile = Profile::default();
        profile.packed.insert("x".into(), (vec![2, 64], 4));
        let checkpoint = FlintCheckpoint {
            weights,
            profile,
            #[cfg(all(
                feature = "native-fault-qualification",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            records: None,
        };
        let error = checkpoint.packed("x").unwrap_err();
        assert!(
            matches!(error, EngError::Checkpoint(message) if message.contains("payload") || message.contains("words"))
        );
    }
}

#[test]
fn open_rejects_malformed_ranges_duplicate_members_and_truncated_files() {
    let root = sparse_checkpoint(|_| {});
    let path = root.path().join("model-00000.safetensors");
    for (header, payload_len) in [
        (
            r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            1,
        ),
        (
            r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#,
            8,
        ),
        (
            r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,8]}}"#,
            8,
        ),
        (
            r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
            3,
        ),
        (
            r#"{"x":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
            0,
        ),
    ] {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.resize(bytes.len() + payload_len, 0);
        std::fs::write(&path, bytes).unwrap();
        let error = FlintCheckpoint::open(root.path())
            .err()
            .expect("invalid header accepted");
        assert!(
            matches!(&error, EngError::Checkpoint(message) if message.contains("shard ")),
            "unexpected refusal: {error:?}"
        );
    }
    std::fs::write(&path, [0u8; 4]).unwrap();
    assert!(FlintCheckpoint::open(root.path()).is_err());
}

#[cfg(all(
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[test]
fn flint_records_fill_exact_expert_slots_and_check_source_identity() {
    use std::os::unix::fs::FileExt;

    let root = sparse_checkpoint(|_| {});
    let source = FlintCheckpoint::open(root.path()).unwrap();
    let layout = source
        .expert_page_layout("model.layers.24.mlp.switch_mlp", 6)
        .unwrap();
    let roles: Vec<_> = layout
        .stacks
        .iter()
        .flat_map(|(_, stack)| {
            [
                &stack.payload,
                &stack.scales,
                stack.biases.as_ref().unwrap(),
            ]
        })
        .collect();
    let sizes: Vec<_> = roles.iter().map(|role| role.slice_len).collect();
    let width: usize = sizes.iter().sum();
    let header = serde_json::to_vec(&json!({
        "format": "fifi-flint-expert-records-v1", "layer": 24, "shards": [], "role_bytes": sizes
    }))
    .unwrap();
    let path = root.path().join("sidecar.records");
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    let start = 8 + header.len() as u64;
    let file_len = start + 256 * width as u64;
    file.set_len(file_len).unwrap();
    let ids = [0, 4, 30, 88, 100, 255];
    let mut expected = vec![0xEF; layout.index_offset];
    for (rank, &id) in ids.iter().enumerate() {
        let mut offset = start + u64::from(id) * width as u64;
        for (index, role) in roles.iter().enumerate() {
            let marker = 1 + (rank * roles.len() + index) as u8;
            let slot = role.offset + rank * role.slice_len;
            expected[slot..slot + role.slice_len].fill(marker);
            file.write_all_at(&expected[slot..slot + role.slice_len], offset)
                .unwrap();
            offset += role.slice_len as u64;
        }
    }
    let mut got = vec![0xEF; layout.index_offset];
    records::ExpertRecords::open(&path, &[])
        .unwrap()
        .read(&layout, &ids, &mut got)
        .unwrap();
    assert!(
        got == expected,
        "record reads must preserve the selected ranges and all padding"
    );
    let unrelated = records::ShardStamp::from(&file.metadata().unwrap());
    assert!(records::ExpertRecords::open(&path, &[unrelated]).is_err());
    file.set_len(file_len - 1).unwrap();
    assert!(records::ExpertRecords::open(&path, &[]).is_err());
}

#[cfg(all(
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[test]
fn flint_record_set_selects_only_declared_layers_and_limits_files() {
    let root = tempfile::tempdir().unwrap();
    let write_sidecar = |name: &str, layer| {
        let header = serde_json::to_vec(&json!({
            "format": "fifi-flint-expert-records-v1", "layer": layer, "shards": [],
            "role_bytes": vec![196608; 9]
        }))
        .unwrap();
        let mut file = File::create(root.path().join(name)).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + 256 * 1_769_472)
            .unwrap();
    };
    assert!(records::RecordSet::open(root.path(), &[], true).is_err());
    write_sidecar("a.records", 47);
    write_sidecar("b.records", 2);
    std::fs::write(root.path().join("ignore.tmp"), []).unwrap();
    let control = records::RecordSet::open(root.path(), &[], false).unwrap();
    assert_eq!(control.describe(), "layers=2,47 mode=shards");
    assert!(matches!(
        control.for_stem("model.layers.2.mlp.switch_mlp"),
        Some((_, false))
    ));
    assert!(control.for_stem("model.layers.24.mlp.switch_mlp").is_none());
    assert!(matches!(
        records::RecordSet::open(root.path(), &[], true)
            .unwrap()
            .for_stem("model.layers.47.mlp.switch_mlp"),
        Some((_, true))
    ));
    write_sidecar("c.records", 47);
    assert!(records::RecordSet::open(root.path(), &[], true).is_err());
    write_sidecar("c.records", 1);
    assert!(records::RecordSet::open(root.path(), &[], true).is_err());
    write_sidecar("c.records", 24);
    write_sidecar("d.records", 30);
    assert!(records::RecordSet::open(root.path(), &[], true).is_ok());
    write_sidecar("e.records", 31);
    assert!(records::RecordSet::open(root.path(), &[], true).is_err());
}

#[test]
fn reads_propagate_truncation_after_header_validation() {
    let root = sparse_checkpoint(|_| {});
    let checkpoint = FlintCheckpoint::open(root.path()).unwrap();
    for index in 0..5 {
        std::fs::OpenOptions::new()
            .write(true)
            .open(root.path().join(format!("model-{index:05}.safetensors")))
            .unwrap()
            .set_len(8)
            .unwrap();
    }
    assert!(matches!(
        checkpoint.tensor("model.norm.weight"),
        Err(EngError::Backend {
            operation: "weight_read",
            ..
        })
    ));
    assert!(checkpoint.embedding_rows(&[0]).is_err());
}

fn fixture_bytes(root: &Path, record: &serde_json::Value) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(root.join(record["file"].as_str().unwrap())).unwrap();
    assert_eq!(bytes.len() as u64, record["bytes"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        record["sha256"].as_str().unwrap()
    );
    bytes
}

fn inject_fixture_tensors(root: &Path, fixture: &Path, records: &serde_json::Value) {
    use std::io::{Seek, SeekFrom};
    for (name, record) in records.as_object().unwrap() {
        let bytes = fixture_bytes(fixture, record);
        let mut found = false;
        for index in 0..5 {
            let path = root.join(format!("model-{index:05}.safetensors"));
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .unwrap();
            let file_len = file.metadata().unwrap().len();
            let header = ShardHeader::read(&mut file, file_len, HEADER_LIMIT).unwrap();
            let Some(entry) = header.entry(name) else {
                continue;
            };
            let start = header.data_start() + entry.offsets().start;
            if let Some(ids) = record["row_ids"].as_array() {
                let row_bytes =
                    (entry.offsets().end - entry.offsets().start) / u64::from(entry.shape()[0]);
                assert_eq!(bytes.len() as u64, row_bytes * ids.len() as u64);
                for (id, row) in ids.iter().zip(bytes.chunks_exact(row_bytes as usize)) {
                    file.seek(SeekFrom::Start(start + id.as_u64().unwrap() * row_bytes))
                        .unwrap();
                    file.write_all(row).unwrap();
                }
            } else {
                assert_eq!(
                    bytes.len() as u64,
                    entry.offsets().end - entry.offsets().start
                );
                file.seek(SeekFrom::Start(start)).unwrap();
                file.write_all(&bytes).unwrap();
            }
            found = true;
        }
        assert!(found, "missing fixture tensor {name}");
    }
}

fn cpu_bf16_dequantized_bytes(packed: &crate::kernels::PackedWeight) -> Vec<u8> {
    fn bf16(bytes: &[u8], index: usize) -> f32 {
        let bits = u16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().unwrap());
        f32::from_bits(u32::from(bits) << 16)
    }
    fn rounded(value: f32) -> f32 {
        let bits = value.to_bits();
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
    }
    let payload = packed.payload().host_bytes().unwrap();
    let scales = packed.scales().host_bytes().unwrap();
    let biases = packed.biases().unwrap().host_bytes().unwrap();
    let bits = packed.quantization().bits;
    let elements = packed
        .logical_shape()
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
        .unwrap();
    let mut bytes = Vec::with_capacity(elements.checked_mul(4).unwrap());
    for index in 0..elements {
        let code = if bits == 4 {
            (payload[index / 2] >> (4 * (index % 2))) & 15
        } else {
            payload[index]
        };
        // The pinned MLX CPU fallback has separate BF16 multiply/add operations.
        let product = rounded(f32::from(code) * bf16(scales, index / 64));
        bytes.extend(rounded(product + bf16(biases, index / 64)).to_le_bytes());
    }
    bytes
}

fn assert_publisher_bytes(actual: &[u8], fixture: &Path, record: &serde_json::Value) {
    let expected = fixture_bytes(fixture, record);
    assert!(
        actual == expected.as_slice(),
        "publisher byte mismatch: {}",
        record["file"]
    );
}

#[test]
#[ignore = "requires the pinned publisher CPU-stream fixture in FLINT_PUBLISHER_CPU_PROOF"]
fn publisher_cpu_fixture_matches_checkpoint_api() {
    let fixture = std::path::PathBuf::from(
        std::env::var_os("FLINT_PUBLISHER_CPU_PROOF").expect("publisher fixture path"),
    );
    let proof: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("result.json")).unwrap()).unwrap();
    assert_eq!(proof["exit"], 0);
    assert_eq!(
        proof["revision"],
        "3b93f9053b530488405ae9fca51d4900d6ce21f0"
    );
    assert_eq!(
        proof["publisher_source_commit"],
        "e53663b262af4cb1ca498e596ca8b1a2d79e0800"
    );
    assert_eq!(proof["versions"]["mlx"], "0.32.0");
    assert_eq!(
        proof["default_device_and_stream"],
        "cpu, selected before arrays"
    );
    assert_eq!(proof["full_model_loaded"], false);
    let root = sparse_checkpoint(|_| {});
    inject_fixture_tensors(root.path(), &fixture, &proof["tensors"]);
    let checkpoint = FlintCheckpoint::open(root.path()).unwrap();
    let attention = checkpoint
        .packed("model.layers.0.self_attn.k_proj")
        .unwrap();
    assert_eq!(attention.logical_shape(), [512, 2048]);
    assert_publisher_bytes(
        attention.payload().host_bytes().unwrap(),
        &fixture,
        &proof["outputs"]["attention_u32"],
    );
    assert_publisher_bytes(
        &cpu_bf16_dequantized_bytes(&attention),
        &fixture,
        &proof["outputs"]["attention_dequant"],
    );
    let embedding = checkpoint.embedding_rows(&[150000, 0, 150000]).unwrap();
    assert_eq!(embedding.logical_shape(), [3, 2048]);
    assert_eq!(embedding.quantization().bits, 8);
    let publisher_rows = fixture_bytes(&fixture, &proof["outputs"]["embedding_u32"]);
    let expected: Vec<_> = [1, 0, 1]
        .into_iter()
        .flat_map(|id| publisher_rows[id * 2048..(id + 1) * 2048].iter().copied())
        .collect();
    assert!(
        embedding.payload().host_bytes().unwrap() == expected.as_slice(),
        "publisher embedding byte order differs"
    );
    assert_publisher_bytes(
        &cpu_bf16_dequantized_bytes(&embedding),
        &fixture,
        &proof["outputs"]["embedding_gather"],
    );
}
