//! Settings of one Q36 engine entry in the `enginesrv` config (decision 0046; packet Q36-F3a).
//!
//! The entry carries `id`, `family`, `model_dir`, `tokenizer_dir`, and `model` (server config) plus
//! the `qwen36` object, which `Q36Options` reads. A key that is absent takes its default or its
//! environment fallback. This file only reads and checks values. It opens no file and starts no
//! thread. Q36 results never satisfy Phase 0B or Phase 1B.

use crate::family::spec::{AttentionKind, FamilySpec, QWEN3_6_35B_A3B};
use crate::{EngError, tokenizer::TokenizerPin};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Vocabulary size of the pinned `tokenizer.json` with its added tokens.
const TOKENIZER_VOCAB_SIZE: u32 = 248_077;
/// Positions of the default context limit (decision 0046 item 6).
pub(crate) const DEFAULT_CONTEXT_LIMIT: u32 = 32_768;
/// The default cache cap: 2 GiB admits 32,768 positions at 40,960 bytes each (1.25 GiB).
pub(crate) const DEFAULT_SESSION_CAP: u64 = 2 << 30;
/// The largest chunk that the prefill plan may pick when `prefill_budget_bytes` is set. One chunk
/// takes about 0.9 s at 290 tok/s, so a cancellation waits that long at most. The plan result sets
/// the worker piece (`ExecutorOps::prefill_piece`). Without a plan, or when the plan admits no
/// chunk, the piece is `NO_BUDGET_PIECE`.
pub(crate) const DEFAULT_PREFILL_CHUNK: u32 = 256;
/// The fallback worker piece for an executor that plans no chunk: no budget, or a plan that admits
/// none. The executor then runs one token per step.
/// The worker checks the cancel flag only between pieces. At about 62 tok/s, a piece of 16 waits
/// about 0.26 s, and a piece of 256 would wait about 4.1 s.
pub(crate) const NO_BUDGET_PIECE: u32 = 16;
/// The largest accepted prefill chunk.
pub(crate) const MAX_PREFILL_CHUNK: u32 = 512;

/// SHA-256 of the pinned files of `/Users/dan/edge0/models/edge0-35b`
/// (`docs/environment/mac-phase0a.json`).
const TOKENIZER_JSON_SHA256: &str =
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4";
const TOKENIZER_CONFIG_SHA256: &str =
    "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182";
const CHAT_TEMPLATE_SHA256: &str =
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
const GENERATION_CONFIG_SHA256: &str =
    "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e";

/// The `qwen36` object of an engine entry. Every member is optional.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Q36Options {
    /// The MLX allocator cache limit in bytes. Fallback: `Q36_CACHE_LIMIT`. There is no default.
    pub(crate) cache_limit_bytes: Option<u64>,
    /// The first position that the engine refuses, from 2. The default is 32,768. The executor
    /// rejects a limit whose cache does not fit `session_cap_bytes`.
    pub(crate) context_limit: Option<u32>,
    /// The bytes of cache growth that the engine admits. The default is 2 GiB.
    pub(crate) session_cap_bytes: Option<u64>,
    /// The prompt tokens per `prefill` call, from 1 to 512. The default is 256. It is the largest
    /// chunk that the planner may choose; the worker piece is the planned chunk, or 16
    /// (`NO_BUDGET_PIECE`) when no chunk is planned: no budget, or a budget that admits none.
    pub(crate) prefill_chunk: Option<u32>,
    /// The engine memory budget in bytes for the admission of the chunked prefill (decision 0044
    /// item 6), from 1. Absent means off: every prompt token is one step. The executor picks the
    /// largest chunk of at most `prefill_chunk` rows whose workspace fits the budget after the
    /// weights and the reserve.
    pub(crate) prefill_budget_bytes: Option<u64>,
    /// The bytes that the planner keeps free for the session state, the MLX cache, and the
    /// pressure margin. It has no default: it is required with `prefill_budget_bytes` and refused
    /// without it. A value of 0 is allowed and admits a larger chunk than the memory plan allows.
    pub(crate) prefill_session_reserve_bytes: Option<u64>,
}

/// The environment fallbacks, read by the caller so that this file stays pure.
#[derive(Debug, Default)]
pub(crate) struct Q36Env {
    /// `Q36_CACHE_LIMIT`, a byte count.
    pub(crate) cache_limit: Option<String>,
}

/// The checked settings of one Q36 engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Q36Settings {
    /// The checkpoint directory with the safetensors shards and `config.json`.
    pub(crate) artifact: PathBuf,
    /// The directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and
    /// `generation_config.json`.
    pub(crate) tokenizer_dir: PathBuf,
    /// The MLX allocator cache limit in bytes.
    pub(crate) cache_limit: u64,
    /// The first position that the engine refuses.
    pub(crate) context_limit: u32,
    /// The bytes of cache growth that the engine admits.
    pub(crate) session_cap: u64,
    /// The prompt tokens per `prefill` call. With a budget it is also the largest planned chunk.
    pub(crate) prefill_chunk: u32,
    /// The budget of the chunked prefill, or `None` when the chunked prefill is off.
    pub(crate) prefill_budget: Option<u64>,
    /// The bytes that the planner keeps free. It is used only when `prefill_budget` is set.
    pub(crate) prefill_reserve: u64,
}

fn bad(message: String) -> EngError {
    EngError::BadRequest(format!("engine config: {message}"))
}

/// `value` or `default`, which must lie in `low..=high`.
fn ranged<T: Copy + PartialOrd + std::fmt::Display>(
    key: &str,
    value: Option<T>,
    default: T,
    low: T,
    high: T,
) -> Result<T, EngError> {
    let value = value.unwrap_or(default);
    if value >= low && value <= high {
        Ok(value)
    } else {
        Err(bad(format!(
            "{key} must be between {low} and {high}, found {value}"
        )))
    }
}

fn cache_limit(options: &Q36Options, env: &Q36Env) -> Result<u64, EngError> {
    let limit = match options.cache_limit_bytes {
        Some(value) => value,
        None => env
            .cache_limit
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                bad("cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set".into())
            })?
            .parse::<u64>()
            .map_err(|_| bad("Q36_CACHE_LIMIT is not a byte count".into()))?,
    };
    if limit == 0 {
        return Err(bad("the cache limit must be above 0".into()));
    }
    Ok(limit)
}

/// `value` when it is set, which must be at least 1.
fn at_least_one(key: &str, value: Option<u64>) -> Result<Option<u64>, EngError> {
    value
        .map(|value| ranged(key, Some(value), 1, 1, u64::MAX))
        .transpose()
}

/// The reserve of the chunked prefill. A budget needs the member, because no default covers the
/// session state, the MLX cache, and the pressure margin of the memory plan. A reserve without a
/// budget is a mistake, because nothing would read it. With neither, the prefill runs one token
/// at a time, nothing reads the reserve, and it stays 0.
fn prefill_reserve(options: &Q36Options, budget: Option<u64>) -> Result<u64, EngError> {
    match (options.prefill_session_reserve_bytes, budget) {
        (Some(_), None) => Err(bad(
            "prefill_session_reserve_bytes needs prefill_budget_bytes".into(),
        )),
        (None, Some(_)) => Err(bad(
            "prefill_budget_bytes needs prefill_session_reserve_bytes".into(),
        )),
        (Some(reserve), Some(_)) => Ok(reserve),
        (None, None) => Ok(0),
    }
}

/// The DeltaNet state bytes of one model: per DeltaNet layer the recurrent state
/// `[value_heads, value_head_dim, key_head_dim]` and the convolution ring
/// `[conv_dim, conv_kernel - 1]`, F32. It is derived from the family spec, never a literal:
/// `QWEN3_6_35B_A3B` holds 65,863,680 bytes in its 30 DeltaNet layers.
fn delta_state_bytes(spec: &FamilySpec) -> Result<u64, EngError> {
    let Some(linear) = spec.linear else {
        return Err(bad(
            "the model has no linear-attention state to reserve".into()
        ));
    };
    let rec = u64::from(linear.value_heads)
        .checked_mul(u64::from(linear.value_head_dim))
        .and_then(|elems| elems.checked_mul(u64::from(linear.key_head_dim)))
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))?;
    let ring = u64::from(linear.conv_dim()?)
        .checked_mul(u64::from(linear.conv_kernel - 1))
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))?;
    let per_layer = rec
        .checked_add(ring)
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))?;
    let layers = spec
        .layers_attention
        .iter()
        .filter(|kind| **kind == AttentionKind::DeltaNet)
        .count();
    u64::try_from(layers)
        .ok()
        .and_then(|layers| per_layer.checked_mul(layers))
        .ok_or_else(|| bad("the DeltaNet state bytes overflow u64".into()))
}

/// The cache bytes that one position adds over every full-attention layer: keys and values
/// `[1, heads_kv, 1, head_dim]`, F32. It is derived from the family spec, never a literal:
/// `QWEN3_6_35B_A3B` adds 40,960 bytes per position in its 10 attention layers.
fn cache_growth_per_position(spec: &FamilySpec) -> Result<u64, EngError> {
    let per_layer = u64::from(spec.heads_kv)
        .checked_mul(u64::from(spec.head_dim))
        .and_then(|row| row.checked_mul(2))
        .and_then(|bytes| bytes.checked_mul(4))
        .ok_or_else(|| bad("the cache growth per position overflows u64".into()))?;
    let layers = spec
        .layers_attention
        .iter()
        .filter(|kind| **kind == AttentionKind::Full)
        .count();
    u64::try_from(layers)
        .ok()
        .and_then(|layers| per_layer.checked_mul(layers))
        .ok_or_else(|| bad("the cache growth per position overflows u64".into()))
}

impl Q36Settings {
    /// Checks the directories of the entry and its `qwen36` object (absent means all members
    /// absent). The config value wins over the environment.
    pub(crate) fn new(
        model_dir: &Path,
        tokenizer_dir: &Path,
        options: Option<&Q36Options>,
        env: &Q36Env,
    ) -> Result<Self, EngError> {
        let default = Q36Options::default();
        let options = options.unwrap_or(&default);
        let prefill_budget = at_least_one("prefill_budget_bytes", options.prefill_budget_bytes)?;
        let prefill_reserve = prefill_reserve(options, prefill_budget)?;
        Ok(Self {
            artifact: model_dir.to_path_buf(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            cache_limit: cache_limit(options, env)?,
            context_limit: ranged(
                "context_limit",
                options.context_limit,
                DEFAULT_CONTEXT_LIMIT,
                2,
                u32::MAX,
            )?,
            session_cap: ranged(
                "session_cap_bytes",
                options.session_cap_bytes,
                DEFAULT_SESSION_CAP,
                1,
                u64::MAX,
            )?,
            prefill_chunk: ranged(
                "prefill_chunk",
                options.prefill_chunk,
                DEFAULT_PREFILL_CHUNK,
                1,
                MAX_PREFILL_CHUNK,
            )?,
            prefill_budget,
            prefill_reserve,
        })
    }

    /// The `prefix_reuse` cap rule (decision 0036, addendum 1 item 7; fix round Q36-PR-1 item 1).
    /// With the process-wide `prefix_reuse` flag on, one saved entry and the live session share
    /// this engine's `SessionBudget`: the entry charges its attention caches plus the fixed
    /// DeltaNet state, and the live session can hold the same caches again up to the context
    /// limit. The cap must therefore cover `2 * growth_per_position * context_limit +
    /// fixed_bytes` (2,750,218,240 bytes at the default context limit of 32,768), or a later job
    /// fails with `MemoryPlan` although the same job succeeds with `prefix_reuse` off. Engine
    /// start refuses a smaller cap and answers with the cap, the requirement, the context limit,
    /// and the hint `raise session_cap_bytes`. With the flag off the rule does not apply.
    pub(crate) fn check_prefix_reuse_cap(&self, prefix_reuse: bool) -> Result<(), EngError> {
        if !prefix_reuse {
            return Ok(());
        }
        let growth = cache_growth_per_position(&QWEN3_6_35B_A3B)?;
        let fixed = delta_state_bytes(&QWEN3_6_35B_A3B)?;
        let required = growth
            .checked_mul(2)
            .and_then(|twice| twice.checked_mul(u64::from(self.context_limit)))
            .and_then(|cache| cache.checked_add(fixed))
            .ok_or_else(|| bad("the prefix_reuse cap requirement overflows u64".into()))?;
        if self.session_cap < required {
            let (cap, limit) = (self.session_cap, self.context_limit);
            let message = format!(
                "prefix_reuse needs session_cap_bytes of at least {required} bytes for context limit {limit}: 2 * {growth} cache bytes per position plus {fixed} DeltaNet bytes; the cap is {cap} bytes; raise session_cap_bytes"
            );
            return Err(bad(message));
        }
        Ok(())
    }
}

fn digest(text: &str) -> Result<[u8; 32], EngError> {
    let invalid = || EngError::Checkpoint(format!("tokenizer pin {text} is not 64 hex digits"));
    let mut out = [0u8; 32];
    if text.len() != 64 {
        return Err(invalid());
    }
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        *slot = std::str::from_utf8(pair.as_slice())
            .ok()
            .and_then(|digits| u8::from_str_radix(digits, 16).ok())
            .ok_or_else(invalid)?;
    }
    Ok(out)
}

/// The pin of the tokenizer files of the pinned checkpoint. `TokenizerBundle::load` rejects any
/// other `tokenizer.json`, `tokenizer_config.json`, template, or `generation_config.json`.
pub(crate) fn tokenizer_pin() -> Result<TokenizerPin, EngError> {
    Ok(TokenizerPin {
        tokenizer_json: digest(TOKENIZER_JSON_SHA256)?,
        tokenizer_config_json: digest(TOKENIZER_CONFIG_SHA256)?,
        chat_template_jinja: digest(CHAT_TEMPLATE_SHA256)?,
        generation_config_json: digest(GENERATION_CONFIG_SHA256)?,
        vocab_size: TOKENIZER_VOCAB_SIZE,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CONTEXT_LIMIT, DEFAULT_PREFILL_CHUNK, DEFAULT_SESSION_CAP, NO_BUDGET_PIECE, Q36Env,
        Q36Options, Q36Settings, cache_growth_per_position, delta_state_bytes, digest,
        tokenizer_pin,
    };
    use crate::EngError;
    use crate::family::spec::QWEN3_6_35B_A3B;
    use std::path::{Path, PathBuf};

    fn with_limit() -> Q36Options {
        Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        }
    }

    fn settings(options: Option<&Q36Options>, env: &Q36Env) -> Result<Q36Settings, EngError> {
        Q36Settings::new(Path::new("/m"), Path::new("/t"), options, env)
    }

    fn message(result: Result<Q36Settings, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    /// The message of a refused call of any kind.
    fn error_text<T: std::fmt::Debug>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    /// The rule derives its byte counts from the family spec, never from a literal.
    #[test]
    fn the_prefix_reuse_bytes_are_derived_from_the_family_spec() {
        // 30 DeltaNet layers of a `[32, 128, 128]` state and an `[8192, 3]` ring, F32.
        assert_eq!(
            delta_state_bytes(&QWEN3_6_35B_A3B).unwrap(),
            65_863_680,
            "DeltaNet state bytes"
        );
        // 10 attention layers of keys and values `[1, 2, 1, 256]`, F32.
        assert_eq!(
            cache_growth_per_position(&QWEN3_6_35B_A3B).unwrap(),
            40_960,
            "cache bytes per position"
        );
    }

    /// The rule accepts exactly `2 * growth_per_position * context_limit + fixed_bytes` and
    /// refuses one byte below it.
    #[test]
    fn the_prefix_reuse_cap_rule_accepts_the_required_cap_and_refuses_one_byte_less() {
        let required = 2 * 40_960 * u64::from(DEFAULT_CONTEXT_LIMIT) + 65_863_680;
        assert_eq!(required, 2_750_218_240, "the required cap");
        let enough = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(required),
            ..Q36Options::default()
        };
        let passed = settings(Some(&enough), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(true);
        assert_eq!(passed, Ok(()), "the required cap passes");
        let one_less = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(required - 1),
            ..Q36Options::default()
        };
        let refused = settings(Some(&one_less), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(true);
        assert!(
            refused.is_err(),
            "one byte below the requirement: {refused:?}"
        );
    }

    /// The refusal names the cap, the requirement, the context limit, and the hint.
    #[test]
    fn the_prefix_reuse_cap_error_names_the_cap_the_requirement_the_limit_and_the_hint() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(2_750_218_239),
            ..Q36Options::default()
        };
        let refused = settings(Some(&options), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(true);
        assert_eq!(
            error_text(refused),
            "engine config: prefix_reuse needs session_cap_bytes of at least 2750218240 bytes for \
             context limit 32768: 2 * 40960 cache bytes per position plus 65863680 DeltaNet bytes; \
             the cap is 2750218239 bytes; raise session_cap_bytes"
        );
    }

    /// Without the flag the rule does not apply, whatever the cap is.
    #[test]
    fn the_prefix_reuse_cap_rule_is_off_when_the_flag_is_off() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(1),
            ..Q36Options::default()
        };
        let passed = settings(Some(&options), &Q36Env::default())
            .unwrap()
            .check_prefix_reuse_cap(false);
        assert_eq!(passed, Ok(()), "the flag is off");
    }

    #[test]
    fn an_object_with_a_cache_limit_takes_the_defaults() {
        let got = settings(Some(&with_limit()), &Q36Env::default()).unwrap();
        assert_eq!(
            got,
            Q36Settings {
                artifact: PathBuf::from("/m"),
                tokenizer_dir: PathBuf::from("/t"),
                cache_limit: 1024,
                context_limit: 32_768,
                session_cap: 2 << 30,
                prefill_chunk: 256,
                prefill_budget: None,
                prefill_reserve: 0,
            }
        );
        assert_eq!(DEFAULT_CONTEXT_LIMIT, 32_768);
        assert_eq!(DEFAULT_SESSION_CAP, 2_147_483_648);
        assert_eq!(DEFAULT_PREFILL_CHUNK, 256);
        assert_eq!(NO_BUDGET_PIECE, 16);
        // The reserve of `config/qwen36-mac-studio-36gb.json` is a plan sum (see its README).
        assert_eq!(32_768u64 * 40_960, 1_342_177_280);
        assert_eq!(
            1_342_177_280u64 + 65_863_680 + 4_294_967_296 + (1 << 30),
            6_776_750_080
        );
    }

    #[test]
    fn every_member_of_the_object_overrides_its_default() {
        let options = Q36Options {
            cache_limit_bytes: Some(5),
            context_limit: Some(65_536),
            session_cap_bytes: Some(3_000_000_000),
            prefill_chunk: Some(64),
            prefill_budget_bytes: Some(30_000_000_000),
            prefill_session_reserve_bytes: Some(7),
        };
        let got = settings(Some(&options), &Q36Env::default()).unwrap();
        assert_eq!(got.cache_limit, 5);
        assert_eq!(got.context_limit, 65_536);
        assert_eq!(got.session_cap, 3_000_000_000);
        assert_eq!(got.prefill_chunk, 64);
        assert_eq!(got.prefill_budget, Some(30_000_000_000));
        assert_eq!(got.prefill_reserve, 7);
        assert_eq!(got.artifact, PathBuf::from("/m"));
        assert_eq!(got.tokenizer_dir, PathBuf::from("/t"));
    }

    #[test]
    fn the_environment_fills_the_cache_limit_and_the_object_wins() {
        let env = Q36Env {
            cache_limit: Some("77".to_string()),
        };
        assert_eq!(settings(None, &env).unwrap().cache_limit, 77);
        assert_eq!(
            settings(Some(&with_limit()), &env).unwrap().cache_limit,
            1024,
            "the object wins"
        );
    }

    #[test]
    fn a_missing_or_bad_cache_limit_names_the_key_and_the_variable() {
        assert_eq!(
            message(settings(None, &Q36Env::default())),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
        let empty = Q36Env {
            cache_limit: Some(String::new()),
        };
        assert_eq!(
            message(settings(None, &empty)),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
        let env = Q36Env {
            cache_limit: Some("many".to_string()),
        };
        assert_eq!(
            message(settings(None, &env)),
            "engine config: Q36_CACHE_LIMIT is not a byte count"
        );
    }

    #[test]
    fn a_cache_limit_of_zero_is_refused() {
        let options = Q36Options {
            cache_limit_bytes: Some(0),
            ..Q36Options::default()
        };
        assert_eq!(
            message(settings(Some(&options), &Q36Env::default())),
            "engine config: the cache limit must be above 0"
        );
    }

    #[test]
    fn the_numeric_members_are_range_checked_and_name_the_key() {
        let cases = [
            (
                Q36Options {
                    context_limit: Some(1),
                    ..with_limit()
                },
                "context_limit must be between 2 and 4294967295, found 1",
            ),
            (
                Q36Options {
                    prefill_chunk: Some(0),
                    ..with_limit()
                },
                "prefill_chunk must be between 1 and 512, found 0",
            ),
            (
                Q36Options {
                    prefill_chunk: Some(513),
                    ..with_limit()
                },
                "prefill_chunk must be between 1 and 512, found 513",
            ),
            (
                Q36Options {
                    session_cap_bytes: Some(0),
                    ..with_limit()
                },
                "session_cap_bytes must be between 1 and 18446744073709551615, found 0",
            ),
        ];
        for (options, want) in cases {
            let got = message(settings(Some(&options), &Q36Env::default()));
            assert_eq!(got, format!("engine config: {want}"));
        }
    }

    #[test]
    fn the_object_reads_from_json_and_rejects_an_unknown_member() {
        let text = r#"{"cache_limit_bytes": 7, "context_limit": 4096, "session_cap_bytes": 9, "prefill_chunk": 3}"#;
        let options: Q36Options = serde_json::from_str(text).unwrap();
        assert_eq!(
            options,
            Q36Options {
                cache_limit_bytes: Some(7),
                context_limit: Some(4096),
                session_cap_bytes: Some(9),
                prefill_chunk: Some(3),
                ..Q36Options::default()
            }
        );
        let error = serde_json::from_str::<Q36Options>(r#"{"manifest": "/m"}"#).unwrap_err();
        assert!(
            error.to_string().contains("unknown field `manifest`"),
            "{error}"
        );
    }

    #[test]
    fn a_prefill_budget_must_be_positive_and_a_reserve_needs_a_budget() {
        let zero = Q36Options {
            prefill_budget_bytes: Some(0),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&zero), &Q36Env::default())),
            "engine config: prefill_budget_bytes must be between 1 and 18446744073709551615, found 0"
        );
        let alone = Q36Options {
            prefill_session_reserve_bytes: Some(5),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&alone), &Q36Env::default())),
            "engine config: prefill_session_reserve_bytes needs prefill_budget_bytes"
        );
        let unreserved = Q36Options {
            prefill_budget_bytes: Some(5),
            ..with_limit()
        };
        assert_eq!(
            message(settings(Some(&unreserved), &Q36Env::default())),
            "engine config: prefill_budget_bytes needs prefill_session_reserve_bytes"
        );
        let both = Q36Options {
            prefill_budget_bytes: Some(1),
            prefill_session_reserve_bytes: Some(0),
            ..with_limit()
        };
        let got = settings(Some(&both), &Q36Env::default()).unwrap();
        assert_eq!((got.prefill_budget, got.prefill_reserve), (Some(1), 0));
    }

    #[test]
    fn the_prefill_reserve_is_the_member_and_never_a_default() {
        let off = settings(Some(&with_limit()), &Q36Env::default()).unwrap();
        assert_eq!((off.prefill_budget, off.prefill_reserve), (None, 0));
        let options = Q36Options {
            cache_limit_bytes: Some(24),
            session_cap_bytes: Some(1000),
            prefill_budget_bytes: Some(5000),
            prefill_session_reserve_bytes: Some(77),
            ..Q36Options::default()
        };
        let got = settings(Some(&options), &Q36Env::default()).unwrap();
        assert_eq!(got.prefill_reserve, 77, "not the cap plus the cache limit");
        let max = Q36Options {
            prefill_session_reserve_bytes: Some(u64::MAX),
            ..options
        };
        let got = settings(Some(&max), &Q36Env::default()).unwrap();
        assert_eq!(got.prefill_reserve, u64::MAX);
    }

    #[test]
    fn the_prefill_members_read_from_json() {
        let text =
            r#"{"prefill_budget_bytes": 32212254720, "prefill_session_reserve_bytes": 6776750080}"#;
        let options: Q36Options = serde_json::from_str(text).unwrap();
        assert_eq!(
            options,
            Q36Options {
                prefill_budget_bytes: Some(32_212_254_720),
                prefill_session_reserve_bytes: Some(6_776_750_080),
                ..Q36Options::default()
            }
        );
    }

    #[test]
    fn the_pin_holds_the_digest_of_each_tokenizer_file() {
        let pin = tokenizer_pin().unwrap();
        assert_eq!(pin.vocab_size, 248_077);
        assert_eq!(pin.tokenizer_json[..4], [0x87, 0xa7, 0x83, 0x0d]);
        assert_eq!(pin.tokenizer_json[28..], [0x3d, 0xb7, 0x2d, 0xe4]);
        assert_eq!(pin.tokenizer_config_json[..4], [0xe9, 0x8f, 0x19, 0x01]);
        assert_eq!(pin.chat_template_jinja[..4], [0xe8, 0x4f, 0x32, 0xa2]);
        assert_eq!(pin.generation_config_json[..4], [0xe7, 0x0c, 0x13, 0x6c]);
    }

    #[test]
    fn a_digest_that_is_not_64_hex_digits_is_refused() {
        assert!(digest(&"a".repeat(64)).is_ok());
        for text in ["", &"a".repeat(63), &"a".repeat(65), &"g".repeat(64)] {
            assert!(
                matches!(digest(text), Err(EngError::Checkpoint(_))),
                "{text:?}"
            );
        }
    }
}
