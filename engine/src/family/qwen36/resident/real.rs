//! Test support for the ignored runs on the real checkpoint: open the pinned directory and read its files.

use std::{
    collections::BTreeSet,
    fs::File,
    path::{Path, PathBuf},
};

use crate::weights::{ShardIo, StreamedCheckpoint};

/// The directory named by the environment variable `name`. Panics when it is not set.
pub(super) fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

/// The sorted shard file names of a `model.safetensors.index.json`.
fn shard_names(index: &[u8]) -> Vec<String> {
    let value: serde_json::Value = serde_json::from_slice(index).unwrap();
    let map = value["weight_map"].as_object().unwrap();
    map.values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// A checkpoint directory opened for streaming, with the bytes of its configuration.
pub(super) struct RealCheckpoint {
    pub(super) ckpt: StreamedCheckpoint,
    pub(super) config: Vec<u8>,
}

/// Opens `dir` through the same reader as the server: one file reader per shard.
pub(super) fn open(dir: &Path) -> RealCheckpoint {
    let config = std::fs::read(dir.join("config.json")).unwrap();
    let index = std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let shards = shard_names(&index)
        .into_iter()
        .map(|name| {
            let file = File::open(dir.join(&name)).unwrap();
            let len = file.metadata().unwrap().len();
            let reader: Box<dyn ShardIo + Send> = Box::new(file);
            (name, reader, len)
        })
        .collect();
    let ckpt = StreamedCheckpoint::open(&index, shards, 1 << 20).unwrap();
    RealCheckpoint { ckpt, config }
}

/// The resident set size of this process in kilobytes, from `ps` (Mac and Linux).
pub(super) fn rss_kb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}
