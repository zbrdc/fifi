use std::path::Path;

use super::*;
use crate::weights::{WeightSource, sha256_hex, sha256_stream};
use serde_json::{Value, json};

fn assert_hash(mut bytes: &[u8], expected: &str) {
    let (digest, _) = sha256_stream(&mut bytes).unwrap();
    assert_eq!(sha256_hex(&digest), expected);
}

fn open_checkpoint(root: &Path) -> Checkpoint {
    let index = std::fs::read(root.join("model.safetensors.index.json")).unwrap();
    assert_hash(
        &index,
        "60deaae44fb80909f694fdd256c0edc10790829cac775d4f01e84eb107ec2754",
    );
    Checkpoint::open(root).unwrap()
}

#[test]
#[ignore = "requires admitted GPU, pinned checkpoint, hash-bound reference cases and scratch output"]
fn real_qwen_projection_uses_shared_loader_and_checked_weight_facts() {
    let root = std::env::var("FIFI_EXL3_CHECKPOINT").unwrap();
    let cases = std::env::var("FIFI_EXL3_REAL_CASES").unwrap();
    let output = std::env::var("FIFI_EXL3_PROJECTION_OUT").unwrap();
    let root = Path::new(&root);
    let metadata_bytes = std::fs::read(cases).unwrap();
    assert_hash(
        &metadata_bytes,
        "8bdc376dbbfdc2ec155c06e1a22e2c9084756e59e16de4b28dc6444ba0396d9b",
    );
    let metadata: Value = serde_json::from_slice(&metadata_bytes).unwrap();
    let prefix = "model.language_model.layers.3.self_attn.q_proj";
    let case = metadata["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"].as_str().unwrap() == format!("{prefix}.trellis"))
        .unwrap();
    let checkpoint = open_checkpoint(root);
    let names: Vec<_> = crate::weights::exl3::COMPONENT_SUFFIXES
        .into_iter()
        .map(|suffix| format!("{prefix}.{suffix}"))
        .collect();
    let keys: Vec<_> = names.iter().map(String::as_str).collect();
    let selected = checkpoint.source.read_selected(&keys, 64 << 20).unwrap();
    let selected_bytes = selected.resident_bytes();
    let matrix = Matrix::load(&selected, prefix, &checkpoint.config).unwrap();
    drop(selected);
    let [k, n] = matrix.shape();
    assert_eq!([k, n], [2048, 8192]);
    assert_eq!(matrix.bitrate_twice(), 10);
    assert_hash(
        matrix.trellis().host_bytes().unwrap(),
        case["trellis"]["sha256"].as_str().unwrap(),
    );
    assert_hash(
        &float_bytes(matrix.suh()).unwrap(),
        case["suh"]["sha256"].as_str().unwrap(),
    );
    assert_hash(
        &float_bytes(matrix.svh()).unwrap(),
        case["svh"]["sha256"].as_str().unwrap(),
    );
    let input = std::fs::read(case["x"]["path"].as_str().unwrap()).unwrap();
    let reference = std::fs::read(case["projection_f64"]["path"].as_str().unwrap()).unwrap();
    assert_hash(&input, case["x"]["sha256"].as_str().unwrap());
    assert_hash(
        &reference,
        case["projection_f64"]["sha256"].as_str().unwrap(),
    );
    assert_eq!(input.len(), 3 * k * 4);
    assert_eq!(reference.len(), 3 * n * 8);
    let reference: Vec<_> = reference
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect();
    let projector = CudaProjector::new().unwrap();
    let before = projector.live_bytes();
    assert!(matches!(
        projector.upload_exl3(
            &matrix,
            Reserve {
                headroom: u64::MAX,
                ..Reserve::default()
            }
        ),
        Err(EngError::MemoryPlan(_))
    ));
    assert_eq!(projector.live_bytes(), before);
    let weight = projector
        .load_exl3(&checkpoint, prefix, Reserve::default())
        .unwrap();
    assert_eq!(weight.shape, matrix.shape());
    assert_eq!(weight.bits_twice, matrix.bitrate_twice());
    assert_eq!(weight.book, matrix.codebook());
    assert_eq!(weight.device_bytes(), 10_526_720);
    assert_eq!(projector.live_bytes() - before, weight.device_bytes());
    let ctx = projector.context();
    let stream = ctx.stream().unwrap();
    for rows in [1usize, 8] {
        let x = ctx
            .upload(
                &input
                    .iter()
                    .copied()
                    .cycle()
                    .take(rows * k * 4)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let xh = ctx.upload(&vec![0xff; rows * k * 4]).unwrap();
        let z = ctx.upload(&vec![0xff; rows * n * 4]).unwrap();
        let out = ctx.upload(&vec![0xff; rows * n * 4]).unwrap();
        let flag = ctx.upload(&0u32.to_le_bytes()).unwrap();
        assert!(matches!(
            weight.project(
                ctx,
                &stream,
                Exl3Io {
                    x: x.range(0, x.bytes()).unwrap(),
                    xh: xh.range(0, xh.bytes()).unwrap(),
                    z: z.range(0, z.bytes()).unwrap(),
                    out: out.range(0, out.bytes()).unwrap(),
                    flag: flag.range(0, 1).unwrap(),
                    rows,
                }
            ),
            Err(EngError::BadRequest(_))
        ));
        weight
            .project(
                ctx,
                &stream,
                Exl3Io {
                    x: x.range(0, x.bytes()).unwrap(),
                    xh: xh.range(0, xh.bytes()).unwrap(),
                    z: z.range(0, z.bytes()).unwrap(),
                    out: out.range(0, out.bytes()).unwrap(),
                    flag: flag.range(0, 4).unwrap(),
                    rows,
                },
            )
            .unwrap();
        stream.synchronize().unwrap();
        let mut status = [0u8; 4];
        ctx.download(&flag, &mut status).unwrap();
        assert_eq!(status, 0u32.to_le_bytes());
        let mut actual = vec![0u8; rows * n * 4];
        ctx.download(&out, &mut actual).unwrap();
        let mut error = 0.0;
        let mut norm = 0.0;
        for (index, bytes) in actual.as_chunks::<4>().0.iter().enumerate() {
            let got = f64::from(f32::from_le_bytes(*bytes));
            let want = reference[index % reference.len()];
            assert!(got.is_finite() && want.is_finite());
            error += (got - want).powi(2);
            norm += want.powi(2);
        }
        let relative = (error / norm).sqrt();
        assert!(
            relative < 1e-5,
            "unchanged FP64 stage bound: rows{rows} error{relative}"
        );
        std::fs::write(
            Path::new(&output).join(format!("fifi-row{rows}-f32.bin")),
            &actual,
        )
        .unwrap();
        println!(
            "{}",
            json!({"rows": rows, "shape": weight.shape, "bits_twice": weight.bits_twice, "fp64_relative": relative, "selected_bytes": selected_bytes, "weight_device_bytes": weight.device_bytes(), "scope": "projection only, not family/model qualification"})
        );
    }
}
