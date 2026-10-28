//! Explicit, model-free export for the independent D128 producer qualification.

use std::{fs::File, io::Write, path::Path};

use sha2::{Digest, Sha256};

fn write_values(path: &Path, values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    format!("{:x}", Sha256::digest(&bytes))
}

#[test]
#[ignore = "explicit model-free Q/K golden export into a fresh scratch directory"]
fn export_requested_qk_goldens() {
    let output = std::path::PathBuf::from(
        std::env::var("FIFI_FLINT_QK_ORACLE_DIR").expect("fresh Q/K golden directory"),
    );
    assert!(output.starts_with("/media/dan/nvme/fifi/scrch/"));
    std::fs::create_dir(&output).unwrap();
    let gamma_q: [f32; 128] =
        std::array::from_fn(|i| f32::from_bits(0x3f700001 + ((i as u32 % 17) << 12)));
    let gamma_k: [f32; 128] =
        std::array::from_fn(|i| f32::from_bits(0x3f800003 + ((i as u32 % 11) << 10)));
    let mut manifest = Vec::new();
    let cases = [1_usize, 2, 4, 8, 4096]
        .into_iter()
        .flat_map(|rows| [(rows, None), (rows, Some(0))])
        .chain([(4, Some(37))]);
    for (rows, position) in cases {
        let q = (0..rows * 4096)
            .map(|i| {
                if i % 67 == 0 {
                    -0.0
                } else {
                    ((i % 29) as i32 - 14) as f32 / 8.0
                }
            })
            .collect::<Vec<_>>();
        let k = (0..rows * 512)
            .map(|i| {
                if i % 71 == 0 {
                    -0.0
                } else {
                    ((i % 23) as i32 - 11) as f32 / 16.0
                }
            })
            .collect::<Vec<_>>();
        let values = super::qk_producer_oracle(&q, &k, [&gamma_q, &gamma_k], position).unwrap();
        let mode = position.map_or_else(|| "no-rope".to_string(), |pos| format!("rope-{pos}"));
        let mut artifacts = Vec::new();
        for (name, values) in ["q", "k"].into_iter().zip(&values) {
            let filename = format!("{name}-{rows}-{mode}.f32le");
            let digest = write_values(&output.join(&filename), values);
            artifacts.push(serde_json::json!({
                "name": name, "file": filename, "f32_values": values.len(), "sha256": digest,
            }));
        }
        manifest
            .push(serde_json::json!({"rows": rows, "position": position, "artifacts": artifacts}));
    }
    assert_eq!(manifest.len(), 11);
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(output.join("manifest.json"))
        .unwrap();
    serde_json::to_writer(&mut file, &manifest).unwrap();
    file.sync_all().unwrap();
}
