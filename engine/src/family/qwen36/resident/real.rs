//! Test support for the ignored runs on the real checkpoint: open the pinned directory and read its files.

use std::{
    collections::BTreeSet,
    fs::File,
    path::{Path, PathBuf},
};

use super::super::settings::{Activation, KvDtype};
use crate::weights::{ShardIo, StreamedCheckpoint};

/// The directory named by the environment variable `name`. Panics when it is not set.
pub(super) fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

/// The decode activation dtype named by `value`, the raw `Q36_ACTIVATION` text:
/// the words are the setting's own `f32` | `bf16`, parsed by `settings::parse_activation`, so the
/// knob and the `activation_dtype` config member cannot drift apart. Unset is NOT the config
/// default here: the knob's unset means the F32 control, so a plain run of the ignored tests is
/// the frozen F32 gate run of before. Any other text is an error that names the variable.
pub(super) fn parse_activation(value: Option<&str>) -> Result<Activation, String> {
    match value {
        None => Ok(Activation::F32),
        Some(text) => super::super::settings::parse_activation(text)
            .map_err(|_| format!("Q36_ACTIVATION must be unset, f32, or bf16, got {text:?}")),
    }
}

/// The cache form named by `value`, the raw `Q36_KV_DTYPE` text: the
/// words are the setting's own `f32` | `q8`, parsed by `settings::parse_kv_dtype`, so the knob
/// and the `kv_dtype` config member cannot drift apart. Unset is the config default, the F32
/// control. Any other text is an error that names the variable.
pub(super) fn parse_kv_dtype(value: Option<&str>) -> Result<KvDtype, String> {
    match value {
        None => Ok(KvDtype::F32),
        Some(text) => super::super::settings::parse_kv_dtype(text)
            .map_err(|_| format!("Q36_KV_DTYPE must be unset, f32, bf16, or q8, got {text:?}")),
    }
}

/// Reads `Q36_KV_DTYPE`, the knob of the ignored runs that selects the product `kv_dtype` lever.
/// Unset gives the F32 control, so an unset run is unchanged by the knob. An invalid value panics.
pub(super) fn kv_dtype_from_env() -> KvDtype {
    parse_kv_dtype(env_text("Q36_KV_DTYPE").as_deref()).unwrap_or_else(|error| panic!("{error}"))
}

/// Reads `Q36_ACTIVATION`, the knob of the ignored runs that selects the product
/// `activation_dtype` lever. Unset gives F32, the qualification control of the frozen gates, so an
/// unset run is the F32 run of before. An invalid value panics.
pub(super) fn activation_from_env() -> Activation {
    parse_activation(env_text("Q36_ACTIVATION").as_deref())
        .unwrap_or_else(|error| panic!("{error}"))
}

/// The text of the environment variable `name`, or `None` when it is unset. A value that is not
/// UTF-8 panics with the name of the variable.
fn env_text(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("{name}: {error}"),
    }
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

/// The current resident set size of this process in KiB.
pub(super) fn rss_kb() -> u64 {
    crate::kernels::mlx::native::process_rss_bytes().expect("native process RSS query must succeed")
        / 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache-form knob parses the settings words, unset is the F32 control, and a bad word
    /// names the variable and the accepted words.
    #[test]
    fn the_kv_dtype_knob_reuses_the_settings_words_and_unset_means_the_f32_control() {
        assert_eq!(parse_kv_dtype(None), Ok(KvDtype::F32));
        assert_eq!(parse_kv_dtype(Some("f32")), Ok(KvDtype::F32));
        assert_eq!(parse_kv_dtype(Some("q8")), Ok(KvDtype::Q8));
        assert_eq!(
            parse_kv_dtype(Some("q4")),
            Err("Q36_KV_DTYPE must be unset, f32, bf16, or q8, got \"q4\"".to_string())
        );
    }

    /// The knob parses its words with the settings parser (one authority)
    /// and keeps its own unset rule: unset means the F32 control, not the config default `bf16`,
    /// so a plain run of the ignored tests is the frozen F32 gate run.
    #[test]
    fn the_activation_knob_reuses_the_settings_words_and_unset_means_the_f32_control() {
        assert_eq!(parse_activation(None), Ok(Activation::F32));
        assert_eq!(parse_activation(Some("f32")), Ok(Activation::F32));
        assert_eq!(parse_activation(Some("bf16")), Ok(Activation::Bf16));
        assert_eq!(
            parse_activation(Some("f64")),
            Err("Q36_ACTIVATION must be unset, f32, or bf16, got \"f64\"".to_string()),
            "a bad word names the knob"
        );
    }

    #[test]
    #[ignore = "needs the Mac; run alone with --ignored"]
    fn rss_tracks_current_mapped_pages_in_kib() {
        let _serial = crate::native_test_lock();
        let before = rss_kb();
        assert!(before > 0, "initial RSS must be positive: {before} KiB");

        let mut mapping = memmap2::MmapMut::map_anon(67_108_864).unwrap();
        mapping.fill(0xa5);
        std::hint::black_box(&mapping[..]);
        let mapped = rss_kb();
        drop(mapping);
        let after_drop = rss_kb();

        let growth = mapped
            .checked_sub(before)
            .expect("RSS must not decrease while the mapping is resident");
        let release = mapped
            .checked_sub(after_drop)
            .expect("RSS must not increase after dropping the mapping");
        let diagnostic = format!(
            "rss_before_kb={before}, rss_mapped_kb={mapped}, rss_after_drop_kb={after_drop}"
        );
        assert!(
            (64_512..=66_560).contains(&growth),
            "mapped RSS growth was {growth} KiB; {diagnostic}"
        );
        assert!(
            (64_512..=66_560).contains(&release),
            "RSS release was {release} KiB; {diagnostic}"
        );
    }
}
