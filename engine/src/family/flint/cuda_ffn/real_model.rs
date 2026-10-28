//! Frozen eight-prefix input and independent full-model oracle transport.

use super::reference::real::gguf;
use crate::{
    family::flint::gguf_weights,
    weights::{GgufHeader, GgufType, dequantize},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

pub(super) const PREFIX_SHA256: &str =
    "2b6f2c43d5a66a85c5feb53d541e2ceda086e2563a8e9ab00e4a64fbada1e814";
pub(super) const GOLDEN_SHA256: &str =
    "0f21860e004ce1a66ca7b09b9eb35774d1b0143b38889110be37605c1e2abe42";
pub(super) const FIRST_TOKENS: [usize; 8] = [12834, 220, 489, 489, 11729, 652, 2865, 332];

#[derive(Deserialize)]
pub(super) struct Prefix {
    pub(super) case: String,
    pub(super) input_ids: Vec<u32>,
}

pub(super) fn prefixes() -> Vec<Prefix> {
    let bytes = include_bytes!("fixtures/first-token-prefixes.json");
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), PREFIX_SHA256);
    let prefixes: Vec<Prefix> = serde_json::from_slice(bytes).unwrap();
    assert_eq!(prefixes.len(), 8);
    for (index, prefix) in prefixes.iter().enumerate() {
        assert_eq!(prefix.case, format!("p{:02}", index + 1));
        assert!((3..=5).contains(&prefix.input_ids.len()));
        assert!(prefix.input_ids.iter().all(|&id| id < 150144));
    }
    prefixes
}

pub(super) fn model() -> (File, GgufHeader, serde_json::Value) {
    let identity = std::env::var("FIFI_FLINT_MODEL_IDENTITY").expect("host identity receipt");
    let identity = serde_json::from_reader(File::open(identity).unwrap()).unwrap();
    let file = File::open(std::env::var("FIFI_FLINT_GGUF").expect("pinned GGUF")).unwrap();
    gguf::verify_prepared_identity(&file, &identity).unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    (file, header, identity)
}

pub(super) fn embedding(file: &File, header: &GgufHeader, tokens: &[u32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(tokens.len() * 2048);
    for &token in tokens {
        let bytes = gguf::raw(file, gguf_weights::embedding_row(header, token).unwrap()).unwrap();
        values.extend(dequantize(GgufType::Bf16, &bytes).unwrap());
    }
    values
}

pub(super) fn write_new(path: &Path, data: &impl Serialize) {
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(&mut file, data).unwrap();
    file.flush().unwrap();
    file.sync_all().unwrap();
}

pub(super) fn fresh_directory(name: &str, sources: &[PathBuf]) -> PathBuf {
    let output = PathBuf::from(std::env::var(name).expect("fresh evidence directory"));
    checked_directory(&output, sources).unwrap()
}

fn checked_directory(output: &Path, sources: &[PathBuf]) -> Result<PathBuf, String> {
    let canonical = |path: &Path| path.canonicalize().map_err(|e| e.to_string());
    let output = canonical(output)?;
    if std::fs::read_dir(&output)
        .map_err(|e| e.to_string())?
        .next()
        .is_some()
    {
        return Err("evidence output directory is not empty".into());
    }
    let mut roots = vec![output.clone()];
    for source in sources {
        let source = canonical(source)?;
        let source = if source.is_file() {
            source.parent().unwrap().to_path_buf()
        } else {
            source
        };
        if roots
            .iter()
            .any(|root| root.starts_with(&source) || source.starts_with(root))
        {
            return Err("evidence roots overlap or alias".into());
        }
        roots.push(source);
    }
    Ok(output)
}

#[derive(Deserialize, Serialize)]
pub(super) struct State {
    pub(super) input_bits: Vec<u32>,
    pub(super) output_bits: Vec<u32>,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Oracle {
    pub(super) schema_version: u32,
    pub(super) model_sha256: String,
    pub(super) prefixes_sha256: String,
    pub(super) physical_kv: String,
    pub(super) token_ids: Vec<Vec<u32>>,
    pub(super) layers: Vec<Vec<State>>,
}

impl Oracle {
    pub(super) fn parse(bytes: &[u8], sha: &str) -> Result<Self, String> {
        if format!("{:x}", Sha256::digest(bytes)) != sha {
            return Err("full-model oracle SHA-256 mismatch".into());
        }
        let oracle: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        oracle.validate()?;
        Ok(oracle)
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        let prefixes = prefixes();
        if self.schema_version != 1
            || self.model_sha256 != gguf::MODEL_SHA256
            || self.prefixes_sha256 != PREFIX_SHA256
            || self.physical_kv != "F16"
            || self.token_ids
                != prefixes
                    .iter()
                    .map(|p| p.input_ids.clone())
                    .collect::<Vec<_>>()
            || self.layers.len() != 49
        {
            return Err("wrong full-model oracle identity or dimensions".into());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.len() != prefixes.len() {
                return Err("wrong full-model case count".into());
            }
            for (case, state) in layer.iter().enumerate() {
                let valid = |bits: &[u32]| {
                    bits.len() == prefixes[case].input_ids.len() * 2048
                        && bits
                            .iter()
                            .all(|&v| v & 0xffff == 0 && f32::from_bits(v).is_finite())
                };
                if !valid(&state.input_bits)
                    || !valid(&state.output_bits)
                    || (index != 0 && state.input_bits != self.layers[index - 1][case].output_bits)
                {
                    return Err(format!(
                        "invalid full-model linkage/publication at layer {index} case {case}"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Oracle {
        let prefixes = prefixes();
        Oracle {
            schema_version: 1,
            model_sha256: gguf::MODEL_SHA256.into(),
            prefixes_sha256: PREFIX_SHA256.into(),
            physical_kv: "F16".into(),
            token_ids: prefixes.iter().map(|p| p.input_ids.clone()).collect(),
            layers: (0..49)
                .map(|_| {
                    prefixes
                        .iter()
                        .map(|p| State {
                            input_bits: vec![0; p.input_ids.len() * 2048],
                            output_bits: vec![0; p.input_ids.len() * 2048],
                        })
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn pinned_prefixes_have_exact_rows_and_no_added_bos() {
        let prefixes = prefixes();
        assert_eq!(
            prefixes
                .iter()
                .map(|p| p.input_ids.len())
                .collect::<Vec<_>>(),
            [5, 5, 5, 3, 4, 3, 3, 3]
        );
        assert!(prefixes.iter().all(|p| p.input_ids[0] != 150000));
        assert_eq!(FIRST_TOKENS.len(), prefixes.len());
        assert_eq!(GOLDEN_SHA256.len(), 64);
    }

    #[test]
    fn evidence_roots_reject_stale_nested_and_aliased_directories() {
        let root = std::env::temp_dir().join(format!(
            "flint-roots-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let output = root.join("output");
        let source = root.join("source");
        std::fs::create_dir(&output).unwrap();
        std::fs::create_dir(&source).unwrap();
        assert!(checked_directory(&output, std::slice::from_ref(&source)).is_ok());
        assert!(checked_directory(&output, &[output.join(".")]).is_err());
        assert!(checked_directory(&output, std::slice::from_ref(&root)).is_err());
        assert!(checked_directory(&output, &[source.clone(), source.join(".")]).is_err());
        std::fs::write(output.join("stale"), b"old").unwrap();
        assert!(checked_directory(&output, &[source]).is_err());
        assert!(root.is_absolute() && root.starts_with(std::env::temp_dir()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn full_model_transport_rejects_identity_linkage_and_non_bf16() {
        let mut oracle = fixture();
        oracle.validate().unwrap();
        oracle.layers[48][7].input_bits[0] = (-0.0_f32).to_bits();
        assert!(oracle.validate().is_err());
        oracle.layers[48][7].input_bits[0] = 1;
        assert!(oracle.validate().is_err());
        oracle.layers[48][7].input_bits[0] = f32::INFINITY.to_bits();
        assert!(oracle.validate().is_err());
        oracle.layers[48][7].input_bits[0] = 0;
        oracle.token_ids[0][0] += 1;
        assert!(oracle.validate().is_err());
        assert!(Oracle::parse(b"{}", &"0".repeat(64)).is_err());
    }
}
