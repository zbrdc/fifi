//! Bounded metadata opening for an indexed EXL3 checkpoint.

use std::{collections::BTreeSet, fs::File, io::Read, path::Path};

use super::Config;
use crate::{
    EngError,
    weights::{ShardIo, StreamedCheckpoint, indexed::parse_weight_map},
};

pub(crate) const CONFIG_LIMIT: u64 = 1024 * 1024;
const INDEX_LIMIT: u64 = 16 * 1024 * 1024;
const HEADER_LIMIT: u64 = 64 * 1024 * 1024;

pub(crate) struct Checkpoint {
    pub(crate) source: StreamedCheckpoint,
    pub(crate) config: Config,
    pub(crate) model_config: Vec<u8>,
}

impl Checkpoint {
    /// Checks bounded configuration, index, and shard headers without reading payloads.
    pub(crate) fn open(dir: &Path) -> Result<Self, EngError> {
        let model_config = read_metadata(&dir.join("config.json"), CONFIG_LIMIT)?;
        let config = Config::from_json(&model_config)?;
        let index = read_metadata(&dir.join("model.safetensors.index.json"), INDEX_LIMIT)?;
        // Validate every index-controlled name before opening any shard path.
        let weight_map = parse_weight_map(&index)?;
        let names: BTreeSet<_> = weight_map.into_values().collect();
        let mut shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = Vec::new();
        for name in names {
            let path = dir.join(&name);
            let file = File::open(&path).map_err(|error| checkpoint_io(&path, error))?;
            let len = file
                .metadata()
                .map_err(|error| checkpoint_io(&path, error))?
                .len();
            shards.push((name, Box::new(file), len));
        }
        let source = StreamedCheckpoint::open(&index, shards, HEADER_LIMIT)?;
        Ok(Self {
            source,
            config,
            model_config,
        })
    }
}

fn read_metadata(path: &Path, limit: u64) -> Result<Vec<u8>, EngError> {
    let file = File::open(path).map_err(|error| checkpoint_io(path, error))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| checkpoint_io(path, error))?;
    if bytes.len() as u64 > limit {
        return Err(EngError::MemoryPlan(format!(
            "EXL3 checkpoint metadata {} exceeds {limit} bytes",
            path.display()
        )));
    }
    Ok(bytes)
}

fn checkpoint_io(path: &Path, error: std::io::Error) -> EngError {
    EngError::Checkpoint(format!("EXL3 checkpoint {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::Dtype, weights::WeightSource};
    use std::io::{Seek, SeekFrom, Write};

    const CONFIG: &[u8] =
        br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9"},"hidden_size":2048}"#;
    const INDEX: &[u8] = br#"{"weight_map":{"a":"m.safetensors","b":"m.safetensors"}}"#;

    fn fixture() -> (tempfile::TempDir, u64) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), CONFIG).unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"), INDEX).unwrap();
        let header = serde_json::to_vec(&serde_json::json!({
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}
        }))
        .unwrap();
        let data_start = 8 + header.len() as u64;
        let mut file = File::create(dir.path().join("m.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&1f32.to_le_bytes()).unwrap();
        file.write_all(&2f32.to_le_bytes()).unwrap();
        (dir, data_start)
    }

    #[test]
    fn opens_metadata_and_reads_selected_payload_later() {
        let (dir, data_start) = fixture();
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(checkpoint.model_config, CONFIG);
        assert_eq!(checkpoint.config.version(), "1.4.9");
        assert_eq!(checkpoint.source.stored("a"), Some((Dtype::F32, vec![1])));
        // A later read must use the retained file, not an eager payload copy.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("m.safetensors"))
            .unwrap();
        file.seek(SeekFrom::Start(data_start)).unwrap();
        file.write_all(&3f32.to_le_bytes()).unwrap();
        let selected = checkpoint.source.read_selected(&["a", "a"], 4).unwrap();
        assert_eq!(selected.resident_bytes(), 4);
        assert_eq!(
            selected.tensor("a").unwrap().host_bytes().unwrap(),
            &3f32.to_le_bytes()
        );
        assert!(selected.tensor("b").is_none());
        assert!(matches!(
            checkpoint.source.read_selected(&["a", "b"], 4),
            Err(EngError::MemoryPlan(_))
        ));
        // Payload I/O errors belong to the later selected read, not opening.
        file.set_len(data_start).unwrap();
        assert!(matches!(
            checkpoint.source.read_selected(&["b"], 4),
            Err(EngError::Backend { .. })
        ));
    }

    #[test]
    fn refuses_invalid_index_before_opening_shards() {
        for (index, expected) in [
            (
                r#"{"weight_map":{"a":"missing.safetensors","a":"m.safetensors"}}"#,
                "duplicate",
            ),
            (
                r#"{"weight_map":{"a":"missing.safetensors","b":"../outside.safetensors"}}"#,
                "invalid shard basename",
            ),
            (
                r#"{"weight_map":{"a":"missing.safetensors","b":"dir\\outside.safetensors"}}"#,
                "invalid shard basename",
            ),
            (
                r#"{"weight_map":{"a":"missing.safetensors","b":12}}"#,
                "requires a string shard",
            ),
        ] {
            let (dir, _) = fixture();
            std::fs::write(dir.path().join("model.safetensors.index.json"), index).unwrap();
            let error = Checkpoint::open(dir.path()).err().unwrap();
            assert!(matches!(&error, EngError::Checkpoint(message) if message.contains(expected)));
        }
    }

    #[test]
    fn enforces_config_and_index_caps() {
        for (name, limit, contents) in [
            ("config.json", CONFIG_LIMIT, CONFIG),
            ("model.safetensors.index.json", INDEX_LIMIT, INDEX),
        ] {
            let (dir, _) = fixture();
            let path = dir.path().join(name);
            let mut padded = contents.to_vec();
            padded.resize(limit as usize, b' ');
            std::fs::write(&path, padded).unwrap();
            assert!(Checkpoint::open(dir.path()).is_ok());
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(limit + 1)
                .unwrap();
            assert!(matches!(
                Checkpoint::open(dir.path()),
                Err(EngError::MemoryPlan(_))
            ));
        }
    }

    #[test]
    fn enforces_header_cap_before_reading_header() {
        let (dir, _) = fixture();
        std::fs::write(
            dir.path().join("m.safetensors"),
            (HEADER_LIMIT + 1).to_le_bytes(),
        )
        .unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path()),
            Err(EngError::MemoryPlan(_))
        ));
    }

    #[test]
    fn refuses_missing_metadata_or_shard() {
        for name in [
            "config.json",
            "model.safetensors.index.json",
            "m.safetensors",
        ] {
            let (dir, _) = fixture();
            std::fs::remove_file(dir.path().join(name)).unwrap();
            assert!(matches!(
                Checkpoint::open(dir.path()),
                Err(EngError::Checkpoint(_))
            ));
        }
    }

    #[test]
    fn refuses_malformed_metadata_and_inconsistent_shards() {
        for (name, bytes) in [
            ("config.json", &b"{"[..]),
            ("config.json", &b"{}"[..]),
            (
                "config.json",
                &br#"{"quantization_config":{"quant_method":"exl3","version":"2.0.0"}}"#[..],
            ),
            ("model.safetensors.index.json", &b"{"[..]),
            ("model.safetensors.index.json", &b"{}"[..]),
            (
                "model.safetensors.index.json",
                &br#"{"weight_map":{"missing":"m.safetensors"}}"#[..],
            ),
            ("m.safetensors", &b"bad"[..]),
            ("m.safetensors", &b"\x01\0\0\0\0\0\0\0{"[..]),
        ] {
            let (dir, _) = fixture();
            std::fs::write(dir.path().join(name), bytes).unwrap();
            assert!(matches!(
                Checkpoint::open(dir.path()),
                Err(EngError::Checkpoint(_))
            ));
        }
        let (dir, data_start) = fixture();
        File::options()
            .write(true)
            .open(dir.path().join("m.safetensors"))
            .unwrap()
            .set_len(data_start + 7)
            .unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path()),
            Err(EngError::Checkpoint(_))
        ));
    }
}
