//! Pin-bound sequential oracle transport; independent from the native execution path.

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::family::flint::{config::FULL_LAYERS, cuda_ffn::reference::real::gguf};

pub(super) fn rows_from_bytes(bytes: usize, columns: usize) -> Result<usize, String> {
    let row_bytes = columns
        .checked_mul(size_of::<f32>())
        .filter(|&width| width != 0)
        .ok_or_else(|| "real-model row width is zero or overflows".to_string())?;
    if !bytes.is_multiple_of(row_bytes) {
        return Err("real-model backing contains a partial row".into());
    }
    let rows = bytes / row_bytes;
    if !(1..=4096).contains(&rows) {
        return Err("real-model row count must be in 1..=4096".into());
    }
    Ok(rows)
}

#[derive(Deserialize)]
pub(super) struct Layer {
    pub(super) layer: u32,
    pub(super) sliding: bool,
    pub(super) input_bits: Vec<u32>,
    pub(super) output_bits: Vec<u32>,
}

#[derive(Deserialize)]
pub(super) struct Oracle {
    pub(super) schema_version: u32,
    pub(super) model_sha256: String,
    pub(super) physical_kv: String,
    pub(super) rows: usize,
    pub(super) token_ids: Vec<u32>,
    pub(super) input_bits: Vec<u32>,
    pub(super) layers: Vec<Layer>,
}

impl Oracle {
    pub(super) fn parse(bytes: &[u8], sha256: &str, layers: usize) -> Result<Self, String> {
        if format!("{:x}", Sha256::digest(bytes)) != sha256 {
            return Err("sequential oracle SHA-256 mismatch".into());
        }
        let oracle: Self = serde_json::from_slice(bytes).map_err(|e| format!("{e:?}"))?;
        oracle.validate(layers)?;
        Ok(oracle)
    }

    pub(super) fn validate(&self, expected_layers: usize) -> Result<(), String> {
        if self.schema_version != 1
            || self.model_sha256 != gguf::MODEL_SHA256
            || self.physical_kv != "F16"
            || self.rows != 4
            || self.token_ids != [0, 1, 2, 3]
            || !(1..=49).contains(&expected_layers)
            || self.layers.len() != expected_layers
        {
            return Err("wrong sequential oracle identity or dimensions".into());
        }
        let valid = |bits: &[u32]| {
            bits.len() == 4 * 2048
                && bits
                    .iter()
                    .all(|&v| v & 0xffff == 0 && f32::from_bits(v).is_finite())
        };
        if !valid(&self.input_bits) {
            return Err("sequential oracle input must be finite BF16 rows".into());
        }
        let mut previous = &self.input_bits;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.layer != index as u32
                || layer.sliding == FULL_LAYERS.contains(&(index as u64))
                || layer.input_bits != *previous
                || !valid(&layer.output_bits)
            {
                return Err(format!(
                    "invalid sequential oracle linkage or output at layer {index}"
                ));
            }
            previous = &layer.output_bits;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Oracle {
        let input = vec![1.0_f32.to_bits(); 4 * 2048];
        let mut previous = input.clone();
        let layers = (0..6)
            .map(|index| {
                let output = vec![(index as f32 + 2.0).to_bits(); previous.len()];
                let layer = Layer {
                    layer: index,
                    sliding: !FULL_LAYERS.contains(&u64::from(index)),
                    input_bits: previous.clone(),
                    output_bits: output.clone(),
                };
                previous = output;
                layer
            })
            .collect();
        Oracle {
            schema_version: 1,
            model_sha256: gguf::MODEL_SHA256.to_string(),
            physical_kv: "F16".into(),
            rows: 4,
            token_ids: vec![0, 1, 2, 3],
            input_bits: input,
            layers,
        }
    }

    #[test]
    fn input_rows_follow_buffer_geometry_not_four_row_fixture() {
        for rows in [1, 3, 4, 5, 4096] {
            assert_eq!(rows_from_bytes(rows * 2048 * 4, 2048).unwrap(), rows);
        }
        for (bytes, columns) in [
            (0, 2048),
            (3 * 2048 * 4 + 1, 2048),
            (4, 0),
            (4097 * 2048 * 4, 2048),
            (usize::MAX, usize::MAX),
        ] {
            assert!(rows_from_bytes(bytes, columns).is_err());
        }
    }

    #[test]
    fn linked_bf16_chain_requires_previous_layer_output() {
        let mut oracle = fixture();
        oracle.validate(6).unwrap();
        oracle.layers[3].input_bits[0] = (-0.0_f32).to_bits();
        assert!(oracle.validate(6).is_err());
        oracle.layers[3].input_bits[0] = oracle.layers[2].output_bits[0];
        oracle.validate(6).unwrap();
        oracle.layers[5].output_bits[0] = f32::NAN.to_bits();
        assert!(oracle.validate(6).is_err());
    }

    #[test]
    fn rejects_wrong_shape_topology_and_non_bf16_output() {
        let mut oracle = fixture();
        assert!(oracle.validate(0).is_err());
        assert!(oracle.validate(49).is_err());
        oracle.layers[1].sliding = true;
        assert!(oracle.validate(6).is_err());
        oracle.layers[1].sliding = false;
        oracle.layers[2].output_bits[0] |= 1;
        assert!(oracle.validate(6).is_err());
        oracle.layers[2].output_bits[0] &= !0xffff;
        oracle.layers[2].output_bits.pop();
        assert!(oracle.validate(6).is_err());
    }

    #[test]
    fn parser_binds_exact_bytes_not_only_parsed_values() {
        let oracle = fixture();
        let value = serde_json::json!({
            "schema_version": oracle.schema_version, "model_sha256": oracle.model_sha256,
            "physical_kv": oracle.physical_kv, "rows": oracle.rows, "token_ids": oracle.token_ids,
            "input_bits": oracle.input_bits,
            "layers": oracle.layers.iter().map(|layer| serde_json::json!({
                "layer": layer.layer, "sliding": layer.sliding,
                "input_bits": layer.input_bits, "output_bits": layer.output_bits,
            })).collect::<Vec<_>>(),
        });
        let mut bytes = serde_json::to_vec(&value).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        assert!(Oracle::parse(&bytes, &sha, 6).is_ok());
        bytes.push(b' ');
        assert!(Oracle::parse(&bytes, &sha, 6).is_err());
    }
}
