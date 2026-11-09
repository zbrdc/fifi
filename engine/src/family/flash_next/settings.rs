//! Settings of one Flash-Next engine entry in the `enginesrv` config.
//!
//! The entry carries `id`, `family`, `model_dir`, `tokenizer_dir`, and `model` (server config) plus
//! the `flash_next` object, which `FlashOptions` reads. A key that is absent takes its default or
//! its environment fallback. This file only reads and checks values. It opens no file and starts
//! no thread.

use super::moe::context_budget::KvFormat;
use crate::EngError;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Vocabulary size of the pinned Flash-Next `tokenizer.json` with its added tokens.
pub(crate) const DEFAULT_VOCAB_SIZE: u32 = 248_077;
/// Prompt tokens per `prefill` call. At 0.246 s per token a chunk of 16 takes about 4 s, which
/// bounds how long a cancellation waits.
pub(crate) const DEFAULT_PREFILL_CHUNK: u32 = 16;
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
    pub(crate) tokenizer_vocab_size: Option<u32>,
    /// From 1 to 512.
    pub(crate) prefill_chunk: Option<u32>,
    /// Records speculative next-layer expert hints in one-token steps. The default is false.
    /// Hints do not stage storage or change routing.
    pub(crate) expert_prefetch: Option<bool>,
    /// Serves the model on a CUDA card (a `native-cuda` Linux build). The default is false: the
    /// CUDA decoder is experimental and a CUDA build refuses to serve Flash-Next without it.
    /// When true, `cache_limit_bytes` (an MLX allocator key) is not required and is refused if set.
    pub(crate) cuda_decode: Option<bool>,
    /// CUDA only: the positions the K/V caches hold, a multiple of 4 from 4 to 262144. Default
    /// 4096. The load refuses a context the card cannot hold next to the trunk.
    pub(crate) context: Option<u32>,
    /// CUDA only: how the K/V is stored on the card. Default `f16`; `int8` has no attention
    /// kernel yet and the settings refuse it.
    pub(crate) kv_format: Option<KvFormatName>,
    /// CUDA only: routed-expert slots per layer in video memory, from 11 to 512. Default: what the
    /// load plan leaves at the context.
    pub(crate) slots: Option<u32>,
    /// CUDA only: rows of one device step during prefill, from 1 to 8. The default, 1, is the
    /// decode shape: one row per device call, exactly as before. A value above one runs the
    /// prompt as device chunks of that many rows (one PLE host roundtrip per chunk), with decode
    /// unchanged at one row.
    pub(crate) prefill_rows: Option<u32>,
    /// CUDA only: the popularity profile file of the expert bank. A load reads it for a
    /// rank-ordered initial fill, and the bank rewrites it with its own usage counts on shutdown,
    /// so the next start of the same entry serves warm. A file that does not exist yet is a cold
    /// start (identity fill); a file that exists but does not parse is refused. Needs
    /// `cuda_decode`.
    pub(crate) expert_stats: Option<PathBuf>,
}

/// The K/V storage formats a `kv_format` value names.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum KvFormatName {
    F16,
    Int8,
}

impl KvFormatName {
    pub(in crate::family::flash_next) fn format(self) -> KvFormat {
        match self {
            Self::F16 => KvFormat::F16,
            Self::Int8 => KvFormat::Int8,
        }
    }
}

/// The CUDA decoder's own settings, present only when `cuda_decode` is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaSettings {
    pub(crate) context: u32,
    pub(crate) kv_format: KvFormatName,
    /// `None`: the load plan decides.
    pub(crate) slots: Option<u32>,
    /// The popularity profile path of `flash_next.expert_stats`.
    pub(crate) expert_stats: Option<PathBuf>,
    /// Rows of one device prefill step (1 = the decode shape). From `prefill_rows`.
    pub(crate) prefill_rows: u32,
}

/// Default positions of the CUDA K/V caches.
pub(crate) const DEFAULT_CUDA_CONTEXT: u32 = 4096;
/// Largest `context` value of the CUDA decoder.
pub(crate) const MAX_CUDA_CONTEXT: u32 = 262_144;
/// The attention indexer's block ratio: the caches hold whole blocks.
const CUDA_CONTEXT_MULTIPLE: u32 = 4;
/// Fewest expert slots: ten picks and the filler slot.
pub(crate) const MIN_CUDA_SLOTS: u32 = 11;
const MAX_CUDA_SLOTS: u32 = 512;

/// The environment fallbacks, read by the caller so that this file stays pure.
#[derive(Debug, Default)]
pub(crate) struct FlashEnv {
    /// `FLASH_NEXT_ARTIFACT_MANIFEST`.
    pub(crate) manifest: Option<String>,
    /// `FLASH_NEXT_CACHE_LIMIT`, a byte count.
    pub(crate) cache_limit: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FlashSettings {
    /// The artifact directory with the safetensors shards and `config.json`.
    pub(crate) artifact: PathBuf,
    /// The directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and
    /// `generation_config.json`.
    pub(crate) tokenizer_dir: PathBuf,
    pub(crate) manifest: PathBuf,
    /// The MLX allocator cache limit in bytes (0 with the CUDA decoder, which ignores it).
    pub(crate) cache_limit: u64,
    pub(crate) tokenizer_vocab_size: u32,
    pub(crate) prefill_chunk: u32,
    /// The default-off expert hint seam. It performs no prefetch I/O in this slice.
    pub(crate) expert_prefetch: bool,
    /// The CUDA decoder, when `cuda_decode` is on.
    pub(crate) cuda: Option<CudaSettings>,
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
    bounded(key, value.unwrap_or(default), low, high)
}

fn bounded(key: &str, value: u32, low: u32, high: u32) -> Result<u32, EngError> {
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

/// The CUDA keys. Absent when `cuda_decode` is off; the keys are then refused if present, so a
/// misspelt switch cannot leave an operator believing the card is in use.
fn cuda_settings(options: &FlashOptions) -> Result<Option<CudaSettings>, EngError> {
    if !options.cuda_decode.unwrap_or(false) {
        let stray = [
            ("context", options.context.is_some()),
            ("kv_format", options.kv_format.is_some()),
            ("slots", options.slots.is_some()),
            ("expert_stats", options.expert_stats.is_some()),
            ("prefill_rows", options.prefill_rows.is_some()),
        ];
        return match stray.iter().find(|(_, set)| *set) {
            Some((key, _)) => Err(bad(format!("{key} needs cuda_decode set to true"))),
            None => Ok(None),
        };
    }
    if options.cache_limit_bytes.is_some() {
        return Err(bad(
            "cache_limit_bytes is an MLX allocator key and cannot be combined with cuda_decode"
                .into(),
        ));
    }
    let kv_format = options.kv_format.unwrap_or(KvFormatName::F16);
    if kv_format == KvFormatName::Int8 {
        return Err(bad(
            "kv_format int8 has no attention kernel yet; use f16".into()
        ));
    }
    let context = ranged(
        "context",
        options.context,
        DEFAULT_CUDA_CONTEXT,
        CUDA_CONTEXT_MULTIPLE,
        MAX_CUDA_CONTEXT,
    )?;
    if !context.is_multiple_of(CUDA_CONTEXT_MULTIPLE) {
        return Err(bad(format!(
            "context must be a multiple of {CUDA_CONTEXT_MULTIPLE}, found {context}"
        )));
    }
    let slots = options
        .slots
        .map(|slots| bounded("slots", slots, MIN_CUDA_SLOTS, MAX_CUDA_SLOTS))
        .transpose()?;
    if options
        .expert_stats
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(bad("expert_stats must not be empty".into()));
    }
    // The step's rows are capped by the merged expert launches; `MAX_STEP_ROWS` in
    // `cuda/step_plan.rs` is the same 8.
    let prefill_rows = ranged("prefill_rows", options.prefill_rows, 1, 1, 8)?;
    Ok(Some(CudaSettings {
        context,
        kv_format,
        slots,
        prefill_rows,
        expert_stats: options.expert_stats.clone(),
    }))
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
        let cuda = cuda_settings(options)?;
        Ok(Self {
            artifact: model_dir.to_path_buf(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            manifest: manifest_path(options, env)?,
            // An MLX allocator key: the CUDA decoder has no use for it.
            cache_limit: if cuda.is_some() {
                0
            } else {
                cache_limit(options, env)?
            },
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
            expert_prefetch: options.expert_prefetch.unwrap_or(false),
            cuda,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaSettings, FlashEnv, FlashOptions, FlashSettings, KvFormatName};
    use crate::EngError;
    use crate::family::flash_next::moe::context_budget::KvFormat;
    use std::path::{Path, PathBuf};

    fn full() -> FlashOptions {
        FlashOptions {
            manifest: Some(PathBuf::from("/art/download-manifest.json")),
            cache_limit_bytes: Some(1_073_741_824),
            tokenizer_vocab_size: Some(1000),
            prefill_chunk: Some(8),
            expert_prefetch: Some(true),
            ..FlashOptions::default()
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
                expert_prefetch: true,
                cuda: None,
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
                expert_prefetch: false,
                cuda: None,
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
    fn expert_prefetch_defaults_off_and_parses_both_booleans() {
        let env = FlashEnv {
            manifest: Some("/manifest.json".into()),
            cache_limit: Some("1024".into()),
        };
        assert!(!settings(None, &env).unwrap().expert_prefetch);
        for (json, expected) in [
            ("{}", false),
            (r#"{"expert_prefetch": false}"#, false),
            (r#"{"expert_prefetch": true}"#, true),
        ] {
            let options: FlashOptions = serde_json::from_str(json).unwrap();
            assert_eq!(
                settings(Some(&options), &env).unwrap().expert_prefetch,
                expected,
                "{json}"
            );
        }
        assert!(serde_json::from_str::<FlashOptions>(r#"{"expert_prefetch": 1}"#).is_err());
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
                ..FlashOptions::default()
            }
        );
        let error = serde_json::from_str::<FlashOptions>(r#"{"context_limit": 9}"#).unwrap_err();
        assert!(
            error.to_string().contains("unknown field `context_limit`"),
            "{error}"
        );
    }

    fn cuda_on() -> FlashOptions {
        FlashOptions {
            manifest: Some(PathBuf::from("/art/download-manifest.json")),
            cuda_decode: Some(true),
            ..FlashOptions::default()
        }
    }

    #[test]
    fn the_cuda_decoder_is_off_by_default_and_its_keys_need_the_switch() {
        let env = FlashEnv::default();
        assert_eq!(settings(Some(&full()), &env).unwrap().cuda, None);
        for (options, key) in [
            (
                FlashOptions {
                    context: Some(8192),
                    ..full()
                },
                "context",
            ),
            (
                FlashOptions {
                    kv_format: Some(KvFormatName::F16),
                    ..full()
                },
                "kv_format",
            ),
            (
                FlashOptions {
                    slots: Some(64),
                    ..full()
                },
                "slots",
            ),
        ] {
            assert_eq!(
                message(settings(Some(&options), &env)),
                format!("engine config: {key} needs cuda_decode set to true")
            );
        }
        let off = FlashOptions {
            cuda_decode: Some(false),
            ..full()
        };
        assert_eq!(settings(Some(&off), &env).unwrap().cuda, None);
    }

    #[test]
    fn the_cuda_decoder_takes_defaults_and_needs_no_cache_limit() {
        let got = settings(Some(&cuda_on()), &FlashEnv::default()).unwrap();
        assert_eq!(got.cache_limit, 0);
        assert_eq!(
            got.cuda,
            Some(CudaSettings {
                context: 4096,
                kv_format: KvFormatName::F16,
                slots: None,
                expert_stats: None,
                prefill_rows: 1,
            })
        );
        // The manifest is still required: it pins the tokenizer.
        let no_manifest = FlashOptions {
            manifest: None,
            ..cuda_on()
        };
        assert!(
            message(settings(Some(&no_manifest), &FlashEnv::default()))
                .contains("manifest is missing")
        );
    }

    #[test]
    fn the_cuda_keys_are_read_checked_and_bounded() {
        let env = FlashEnv::default();
        let options: FlashOptions = serde_json::from_str(
            r#"{"manifest": "/m.json", "cuda_decode": true, "context": 65536,
                "kv_format": "f16", "slots": 48}"#,
        )
        .unwrap();
        assert_eq!(
            settings(Some(&options), &env).unwrap().cuda,
            Some(CudaSettings {
                context: 65536,
                kv_format: KvFormatName::F16,
                slots: Some(48),
                expert_stats: None,
                prefill_rows: 1,
            })
        );
        assert_eq!(KvFormatName::Int8.format(), KvFormat::Int8);
        // expert_stats is a CUDA key: carried with cuda_decode, refused without it.
        let warm = serde_json::from_str::<FlashOptions>(
            r#"{"manifest": "/m.json", "cuda_decode": true, "expert_stats": "/var/lib/fn/stats.json"}"#,
        )
        .unwrap();
        assert_eq!(
            settings(Some(&warm), &env)
                .unwrap()
                .cuda
                .map(|c| c.expert_stats),
            Some(Some(std::path::PathBuf::from("/var/lib/fn/stats.json")))
        );
        let off = serde_json::from_str::<FlashOptions>(
            r#"{"manifest": "/m.json", "expert_stats": "/var/lib/fn/stats.json"}"#,
        )
        .unwrap();
        let refused = settings(Some(&off), &env).unwrap_err();
        assert!(
            format!("{refused:?}").contains("expert_stats needs cuda_decode"),
            "{refused:?}"
        );
        for (options, want) in [
            (
                FlashOptions {
                    kv_format: Some(KvFormatName::Int8),
                    ..cuda_on()
                },
                "kv_format int8 has no attention kernel yet; use f16",
            ),
            (
                FlashOptions {
                    cache_limit_bytes: Some(1024),
                    ..cuda_on()
                },
                "cache_limit_bytes is an MLX allocator key and cannot be combined with cuda_decode",
            ),
            (
                FlashOptions {
                    context: Some(3),
                    ..cuda_on()
                },
                "context must be between 4 and 262144, found 3",
            ),
            (
                FlashOptions {
                    context: Some(4098),
                    ..cuda_on()
                },
                "context must be a multiple of 4, found 4098",
            ),
            (
                FlashOptions {
                    context: Some(262_148),
                    ..cuda_on()
                },
                "context must be between 4 and 262144, found 262148",
            ),
            (
                FlashOptions {
                    slots: Some(10),
                    ..cuda_on()
                },
                "slots must be between 11 and 512, found 10",
            ),
            (
                FlashOptions {
                    slots: Some(513),
                    ..cuda_on()
                },
                "slots must be between 11 and 512, found 513",
            ),
        ] {
            assert_eq!(
                message(settings(Some(&options), &env)),
                format!("engine config: {want}")
            );
        }
        assert!(serde_json::from_str::<FlashOptions>(r#"{"kv_format": "fp8"}"#).is_err());
    }
}
