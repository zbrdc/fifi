use std::{io::Cursor, sync::Arc};

use inference_engine::{
    EngError,
    tensor::{Dtype, Tensor},
    weights::{
        ResidentWeights, ShardHeader, WeightSource,
        exl3::{Codebook, Config, Matrix},
    },
};
use serde_json::json;

const TRELLIS_PATTERN: [u8; 8] = [0x01, 0x80, 0xfe, 0x7f, 0x00, 0xff, 0xab, 0x12];

fn storage_fixture() -> Vec<u8> {
    let mut header = serde_json::to_vec(&json!({
        "matrix.trellis": {"dtype": "I16", "shape": [8, 8, 48], "data_offsets": [0, 6144]},
        "matrix.suh": {"dtype": "F16", "shape": [128], "data_offsets": [6144, 6400]},
        "matrix.svh": {"dtype": "F16", "shape": [128], "data_offsets": [6400, 6656]},
        "matrix.mul1": {"dtype": "I32", "shape": [], "data_offsets": [6656, 6660]}
    }))
    .unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend(TRELLIS_PATTERN.into_iter().cycle().take(6144));
    bytes.resize(bytes.len() + 512, 0);
    bytes.extend_from_slice(&0x83dc_d12du32.to_le_bytes());
    bytes
}

#[test]
fn matrix_loader_uses_physical_rate_not_config_average() {
    let config = Config::from_json(br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1","bits":3.08,"out_scales":"always"}}"#).unwrap();
    let source = ResidentWeights::from_bytes(Arc::from(storage_fixture())).unwrap();
    let matrix = Matrix::load(&source, "matrix", &config).unwrap();
    assert_eq!(matrix.shape(), [128, 128]);
    assert_eq!(matrix.bitrate_twice(), 6);
    assert_eq!(matrix.codebook(), Codebook::Mul1);
    assert_eq!(config.expert_interm_div(), 1);
}

#[test]
fn shared_header_preserves_exl3_storage_types_and_ranges() {
    let bytes = storage_fixture();
    let header = ShardHeader::read(&mut Cursor::new(&bytes), bytes.len() as u64, 1024)
        .expect("shared safetensors headers must accept EXL3 storage types");
    for (name, dtype, shape, range) in [
        ("matrix.trellis", Dtype::I16, vec![8, 8, 48], 0..6144),
        ("matrix.suh", Dtype::F16, vec![128], 6144..6400),
        ("matrix.svh", Dtype::F16, vec![128], 6400..6656),
        ("matrix.mul1", Dtype::I32, vec![], 6656..6660),
    ] {
        let entry = header.entry(name).expect("component is retained");
        assert_eq!(entry.dtype(), dtype, "{name}");
        assert_eq!(entry.shape(), shape, "{name}");
        assert_eq!(entry.offsets(), &range, "{name}");
    }
}

#[test]
fn resident_exl3_components_keep_their_raw_bytes() {
    let weights = ResidentWeights::from_bytes(Arc::from(storage_fixture())).unwrap();
    let trellis = weights.tensor("matrix.trellis").unwrap();
    assert_eq!(trellis.dtype(), Dtype::I16);
    let expected = TRELLIS_PATTERN.repeat(6144 / TRELLIS_PATTERN.len());
    assert!(trellis.host_bytes().unwrap() == expected.as_slice());
    let marker = weights.tensor("matrix.mul1").unwrap();
    assert_eq!(marker.dtype(), Dtype::I32);
    assert!(marker.shape().is_empty());
    assert_eq!(marker.host_bytes().unwrap(), 0x83dc_d12du32.to_le_bytes());
    assert_eq!(weights.resident_bytes(), 6660);
}

#[test]
fn integer_storage_widths_reject_short_or_long_payloads() {
    for (dtype, stored, width) in [(Dtype::I16, "I16", 2), (Dtype::I32, "I32", 4)] {
        let tensor = Tensor::try_from_host(dtype, vec![2], vec![0xff; 2 * width]).unwrap();
        assert_eq!(tensor.byte_len(), (2 * width) as u64);
        assert_eq!(tensor.dtype(), dtype);
        for length in [2 * width - 1, 2 * width + 1] {
            assert!(Tensor::try_from_host(dtype, vec![2], vec![0; length]).is_err());
            let mut header = serde_json::to_vec(&json!({
                "bad": {"dtype": stored, "shape": [2], "data_offsets": [0, length]}
            }))
            .unwrap();
            header.resize(header.len().div_ceil(8) * 8, b' ');
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.resize(bytes.len() + length, 0);
            assert!(matches!(
                ShardHeader::read(&mut Cursor::new(&bytes), bytes.len() as u64, 1024),
                Err(EngError::Checkpoint(_))
            ));
            assert!(matches!(
                ResidentWeights::from_bytes(Arc::from(bytes)),
                Err(EngError::Checkpoint(_))
            ));
        }
    }
}
