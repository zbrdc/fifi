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
    (FlintCheckpoint { weights, profile }, payload)
}

#[test]
fn packed_api_preserves_2d_and_3d_bytes_and_affine_metadata() {
    for shape in [vec![2, 64], vec![2, 3, 128]] {
        let (mut checkpoint, bytes) = tiny(shape.clone(), 4);
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
    let (mut checkpoint, bytes) = tiny(vec![4, 2048], 8);
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

#[test]
fn open_checks_complete_profile_without_reading_payloads() {
    let root = sparse_checkpoint(|_| {});
    let mut checkpoint = FlintCheckpoint::open(root.path()).unwrap();
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
        let mut checkpoint = FlintCheckpoint { weights, profile };
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

#[test]
fn reads_propagate_truncation_after_header_validation() {
    let root = sparse_checkpoint(|_| {});
    let mut checkpoint = FlintCheckpoint::open(root.path()).unwrap();
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
    let mut checkpoint = FlintCheckpoint::open(root.path()).unwrap();
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
