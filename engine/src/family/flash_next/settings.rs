//! Settings of one Flash-Next engine entry in the `enginesrv` config (decision 0026; packet H8c).
//!
//! The entry carries `id`, `family`, `model_dir`, `tokenizer_dir`, and `model` (server config) plus
//! the `flash_next` object, which `FlashOptions` reads. A key that is absent takes its default or
//! its environment fallback. This file only reads and checks values. It opens no file and starts
//! no thread.

use crate::EngError;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Vocabulary size of the pinned Flash-Next `tokenizer.json` with its added tokens.
pub(crate) const DEFAULT_VOCAB_SIZE: u32 = 248_077;
/// Prompt tokens per `prefill` call. At 0.246 s per token a chunk of 16 takes about 4 s, which
/// bounds how long a cancellation waits.
pub(crate) const DEFAULT_PREFILL_CHUNK: u32 = 16;
/// The largest accepted prefill chunk.
pub(crate) const MAX_PREFILL_CHUNK: u32 = 512;

/// The `flash_next` object of an engine entry. Every member is optional.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlashOptions {
    /// The artifact manifest (`download-manifest.json`) that pins the tokenizer files.
    /// Fallback: `FLASH_NEXT_ARTIFACT_MANIFEST`.
    pub(crate) manifest: Option<PathBuf>,
    /// The MLX allocator cache limit in bytes. Fallback: `FLASH_NEXT_CACHE_LIMIT`. There is no
    /// default.
    pub(crate) cache_limit_bytes: Option<u64>,
    /// The vocabulary size of `tokenizer.json`. The default is 248,077.
    pub(crate) tokenizer_vocab_size: Option<u32>,
    /// The prompt tokens per `prefill` call, from 1 to 512. The default is 16.
    pub(crate) prefill_chunk: Option<u32>,
}

/// The environment fallbacks, read by the caller so that this file stays pure.
#[derive(Debug, Default)]
pub(crate) struct FlashEnv {
    /// `FLASH_NEXT_ARTIFACT_MANIFEST`.
    pub(crate) manifest: Option<String>,
    /// `FLASH_NEXT_CACHE_LIMIT`, a byte count.
    pub(crate) cache_limit: Option<String>,
}

/// The checked settings of one Flash-Next engine.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FlashSettings {
    /// The artifact directory with the safetensors shards and `config.json`.
    pub(crate) artifact: PathBuf,
    /// The directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and
    /// `generation_config.json`.
    pub(crate) tokenizer_dir: PathBuf,
    /// The artifact manifest that pins the tokenizer files.
    pub(crate) manifest: PathBuf,
    /// The MLX allocator cache limit in bytes.
    pub(crate) cache_limit: u64,
    /// The vocabulary size of `tokenizer.json`.
    pub(crate) tokenizer_vocab_size: u32,
    /// The prompt tokens per `prefill` call.
    pub(crate) prefill_chunk: u32,
}

fn bad(message: String) -> EngError {
    EngError::BadRequest(format!("engine config: {message}"))
}

fn ranged(
    key: &str,
    value: Option<u32>,
    default: u32,
    low: u32,
    high: u32,
) -> Result<u32, EngError> {
    let value = value.unwrap_or(default);
    if (low..=high).contains(&value) {
        Ok(value)
    } else {
        Err(bad(format!(
            "{key} must be between {low} and {high}, found {value}"
        )))
    }
}

fn manifest_path(options: &FlashOptions, env: &FlashEnv) -> Result<PathBuf, EngError> {
    if let Some(path) = &options.manifest {
        if path.as_os_str().is_empty() {
            return Err(bad("manifest must not be empty".into()));
        }
        return Ok(path.clone());
    }
    env.manifest
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            bad("manifest is missing and FLASH_NEXT_ARTIFACT_MANIFEST is not set".into())
        })
}

fn cache_limit(options: &FlashOptions, env: &FlashEnv) -> Result<u64, EngError> {
    let limit = match options.cache_limit_bytes {
        Some(value) => value,
        None => env
            .cache_limit
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                bad("cache_limit_bytes is missing and FLASH_NEXT_CACHE_LIMIT is not set".into())
            })?
            .parse::<u64>()
            .map_err(|_| bad("FLASH_NEXT_CACHE_LIMIT is not a byte count".into()))?,
    };
    if limit == 0 {
        return Err(bad("the cache limit must be above 0".into()));
    }
    Ok(limit)
}

impl FlashSettings {
    /// Checks the directories of the entry and its `flash_next` object (absent means all members
    /// absent). The config value wins over the environment.
    pub(crate) fn new(
        model_dir: &Path,
        tokenizer_dir: &Path,
        options: Option<&FlashOptions>,
        env: &FlashEnv,
    ) -> Result<Self, EngError> {
        let default = FlashOptions::default();
        let options = options.unwrap_or(&default);
        Ok(Self {
            artifact: model_dir.to_path_buf(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            manifest: manifest_path(options, env)?,
            cache_limit: cache_limit(options, env)?,
            tokenizer_vocab_size: ranged(
                "tokenizer_vocab_size",
                options.tokenizer_vocab_size,
                DEFAULT_VOCAB_SIZE,
                1,
                u32::MAX,
            )?,
            prefill_chunk: ranged(
                "prefill_chunk",
                options.prefill_chunk,
                DEFAULT_PREFILL_CHUNK,
                1,
                MAX_PREFILL_CHUNK,
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{FlashEnv, FlashOptions, FlashSettings};
    use crate::EngError;
    use std::path::{Path, PathBuf};

    fn full() -> FlashOptions {
        FlashOptions {
            manifest: Some(PathBuf::from("/art/download-manifest.json")),
            cache_limit_bytes: Some(1_073_741_824),
            tokenizer_vocab_size: Some(1000),
            prefill_chunk: Some(8),
        }
    }

    fn settings(options: Option<&FlashOptions>, env: &FlashEnv) -> Result<FlashSettings, EngError> {
        FlashSettings::new(Path::new("/art"), Path::new("/tok"), options, env)
    }

    fn message(result: Result<FlashSettings, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_full_object_is_read_as_written() {
        let got = settings(Some(&full()), &FlashEnv::default()).unwrap();
        assert_eq!(
            got,
            FlashSettings {
                artifact: PathBuf::from("/art"),
                tokenizer_dir: PathBuf::from("/tok"),
                manifest: PathBuf::from("/art/download-manifest.json"),
                cache_limit: 1_073_741_824,
                tokenizer_vocab_size: 1000,
                prefill_chunk: 8,
            }
        );
    }

    #[test]
    fn the_environment_fills_the_manifest_and_the_limit_and_the_defaults_fill_the_rest() {
        let env = FlashEnv {
            manifest: Some("/env-art/download-manifest.json".into()),
            cache_limit: Some("2048".into()),
        };
        let got = settings(None, &env).unwrap();
        assert_eq!(
            got,
            FlashSettings {
                artifact: PathBuf::from("/art"),
                tokenizer_dir: PathBuf::from("/tok"),
                manifest: PathBuf::from("/env-art/download-manifest.json"),
                cache_limit: 2048,
                tokenizer_vocab_size: 248_077,
                prefill_chunk: 16,
            }
        );
    }

    #[test]
    fn the_config_wins_over_the_environment() {
        let env = FlashEnv {
            manifest: Some("/env-manifest".into()),
            cache_limit: Some("5".into()),
        };
        let got = settings(Some(&full()), &env).unwrap();
        assert_eq!(got.manifest, PathBuf::from("/art/download-manifest.json"));
        assert_eq!(got.cache_limit, 1_073_741_824);
    }

    #[test]
    fn a_missing_manifest_or_limit_names_the_key_and_the_variable() {
        let env = FlashEnv::default();
        let no_manifest = FlashOptions {
            manifest: None,
            ..full()
        };
        assert_eq!(
            message(settings(Some(&no_manifest), &env)),
            "engine config: manifest is missing and FLASH_NEXT_ARTIFACT_MANIFEST is not set"
        );
        let no_limit = FlashOptions {
            cache_limit_bytes: None,
            ..full()
        };
        assert_eq!(
            message(settings(Some(&no_limit), &env)),
            "engine config: cache_limit_bytes is missing and FLASH_NEXT_CACHE_LIMIT is not set"
        );
    }

    #[test]
    fn an_empty_environment_variable_counts_as_not_set() {
        let env = FlashEnv {
            manifest: Some(String::new()),
            cache_limit: Some(String::new()),
        };
        let no_manifest = FlashOptions {
            manifest: None,
            ..full()
        };
        assert!(
            message(settings(Some(&no_manifest), &env)).contains("manifest is missing"),
            "an empty manifest variable must not become the path \"\""
        );
        let no_limit = FlashOptions {
            cache_limit_bytes: None,
            ..full()
        };
        assert!(message(settings(Some(&no_limit), &env)).contains("cache_limit_bytes is missing"));
    }

    #[test]
    fn a_value_out_of_range_is_rejected_with_its_key() {
        let env = FlashEnv::default();
        let cases = [
            (
                FlashOptions {
                    manifest: Some(PathBuf::new()),
                    ..full()
                },
                "manifest must not be empty",
            ),
            (
                FlashOptions {
                    cache_limit_bytes: Some(0),
                    ..full()
                },
                "the cache limit must be above 0",
            ),
            (
                FlashOptions {
                    prefill_chunk: Some(0),
                    ..full()
                },
                "prefill_chunk must be between 1 and 512, found 0",
            ),
            (
                FlashOptions {
                    prefill_chunk: Some(513),
                    ..full()
                },
                "prefill_chunk must be between 1 and 512, found 513",
            ),
            (
                FlashOptions {
                    tokenizer_vocab_size: Some(0),
                    ..full()
                },
                "tokenizer_vocab_size must be between 1 and 4294967295, found 0",
            ),
        ];
        for (options, want) in cases {
            let got = message(settings(Some(&options), &env));
            assert_eq!(got, format!("engine config: {want}"), "options {options:?}");
        }
    }

    #[test]
    fn an_environment_limit_that_is_not_a_number_is_rejected() {
        let env = FlashEnv {
            cache_limit: Some("1GiB".into()),
            ..FlashEnv::default()
        };
        let options = FlashOptions {
            cache_limit_bytes: None,
            ..full()
        };
        assert_eq!(
            message(settings(Some(&options), &env)),
            "engine config: FLASH_NEXT_CACHE_LIMIT is not a byte count"
        );
    }

    #[test]
    fn the_prefill_chunk_limits_include_their_bounds() {
        let env = FlashEnv::default();
        for chunk in [1, 512] {
            let options = FlashOptions {
                prefill_chunk: Some(chunk),
                ..full()
            };
            assert_eq!(settings(Some(&options), &env).unwrap().prefill_chunk, chunk);
        }
    }

    #[test]
    fn the_object_reads_from_json_and_rejects_an_unknown_member() {
        let text = r#"{"manifest": "/m.json", "cache_limit_bytes": 7, "prefill_chunk": 4}"#;
        let got: FlashOptions = serde_json::from_str(text).unwrap();
        assert_eq!(
            got,
            FlashOptions {
                manifest: Some(PathBuf::from("/m.json")),
                cache_limit_bytes: Some(7),
                tokenizer_vocab_size: None,
                prefill_chunk: Some(4),
            }
        );
        let error = serde_json::from_str::<FlashOptions>(r#"{"context_limit": 9}"#).unwrap_err();
        assert!(
            error.to_string().contains("unknown field `context_limit`"),
            "{error}"
        );
    }
}
