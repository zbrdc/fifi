//! Settings of one Flash-Next engine entry in the `enginesrv` config.
//!
//! The entry carries `id`, `family`, `model_dir`, `tokenizer_dir`, and `model` (server config) plus
//! the `flash_next` object, which `FlashOptions` reads. A key that is absent takes its default or
//! its environment fallback. This file only reads and checks values. It opens no file and starts
//! no thread.

use super::moe::context_budget::KvFormat;
use crate::EngError;
use crate::family::project::exl3_jobs_plan::exl3_rows_ceiling;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Vocabulary size of the pinned Flash-Next `tokenizer.json` with its added tokens.
pub(crate) const DEFAULT_VOCAB_SIZE: u32 = 248_077;
/// Prompt tokens per `prefill` call. At 0.246 s per token a chunk of 16 takes about 4 s, which
/// bounds how long a cancellation waits.
pub(crate) const DEFAULT_PREFILL_CHUNK: u32 = 16;
pub(crate) const MAX_PREFILL_CHUNK: u32 = 2048;

/// The `flash_next` object of an engine entry. Every member is optional.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlashOptions {
    /// The artifact manifest (`download-manifest.json`) that names the tokenizer files.
    /// Fallbacks: `FLASH_NEXT_ARTIFACT_MANIFEST`, then `download-manifest.json` in the model
    /// directory (or the tokenizer directory).
    pub(crate) manifest: Option<PathBuf>,
    /// The MLX allocator cache limit in bytes. Fallback: `FLASH_NEXT_CACHE_LIMIT`. There is no
    /// default.
    pub(crate) cache_limit_bytes: Option<u64>,
    /// MLX only. Total byte budget for selecting a batched prompt path at startup.
    pub(crate) prefill_budget_bytes: Option<u64>,
    pub(crate) tokenizer_vocab_size: Option<u32>,
    /// From 1 to 2,048.
    pub(crate) prefill_chunk: Option<u32>,
    /// Records speculative next-layer expert hints in one-token steps. The default is false.
    /// Hints do not stage storage or change routing.
    pub(crate) expert_prefetch: Option<bool>,
    /// Serves the model on a CUDA card (a `native-cuda` Linux build). The default is true in a
    /// build that has the CUDA decoder (when neither `cache_limit_bytes` nor `prefill_budget_bytes`
    /// names the MLX arm) and false elsewhere; `false` on a CUDA build refuses to
    /// serve Flash-Next. When true, `cache_limit_bytes` (an MLX allocator key) is not required and is refused if set.
    pub(crate) cuda_decode: Option<bool>,
    /// CUDA only: the positions the K/V caches hold, a multiple of 4 from 4 to 262144. Unset, the
    /// memory plan chooses it at start. The load refuses a context the card cannot hold next to
    /// the trunk.
    pub(crate) context: Option<u32>,
    /// CUDA only: how the K/V is stored on the card. Default `f16`; `int8` has no attention
    /// kernel yet and the settings refuse it.
    pub(crate) kv_format: Option<KvFormatName>,
    /// CUDA only: routed-expert slots per layer in video memory, from 11 to 512. Default: what the
    /// load plan leaves at the context.
    pub(crate) slots: Option<u32>,
    /// CUDA only: rows of one device step during prefill, 1 to 2,048. The unset default is
    /// 256, the qualified shared width; an explicit `1` selects the per-token prefill
    /// (the decode shape, served by the single-row path).
    pub(crate) prefill_rows: Option<u32>,
    /// CUDA only, shared prefill path only: the chunked-prefill row count at and above which
    /// an active prefill ring serves the whole chunk; below it the bank's union-admit path
    /// serves. The legacy (non-shared) path ignores this key and always streams the ring.
    /// A value above `prefill_rows` never engages the ring (every chunk is below it). Default 32.
    pub(crate) ring_rows: Option<u32>,
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
    /// The row threshold of the prefill ring (from `ring_rows`, default 32); the shared
    /// prefill path only, the legacy path always streams the ring.
    pub(crate) ring_rows: u32,
}

/// Default positions of the CUDA K/V caches.
pub(crate) const DEFAULT_CUDA_CONTEXT: u32 = 4096;
/// Largest `context` value of the CUDA decoder.
pub(crate) const MAX_CUDA_CONTEXT: u32 = 262_144;
/// The share of the expert bank, in percent of the slots it has at [`DEFAULT_CUDA_CONTEXT`], that
/// the default context may give up to the K/V cache. A longer context costs bank slots, and the
/// slots decide decode speed, so the default stops where the bank would shrink past this.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
const BANK_SHARE_GIVEN_PERCENT: usize = 10;

/// The context the CUDA decoder serves when the config sets none, and what it was chosen from.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::family::flash_next) struct DefaultContext {
    pub(in crate::family::flash_next) context: u32,
    /// Expert slots per layer the plan leaves at `context`.
    pub(in crate::family::flash_next) slots: usize,
    /// Expert slots per layer at the smaller of [`DEFAULT_CUDA_CONTEXT`] and the cap.
    pub(in crate::family::flash_next) baseline_slots: usize,
}

/// The longest context, a multiple of 1024 (or `cap` itself), at which the bank keeps at least
/// `BANK_SHARE_GIVEN_PERCENT` percent fewer slots than at [`DEFAULT_CUDA_CONTEXT`] and not
/// fewer than `min_slots`. `slots_at` is the memory plan: the slots per layer a context leaves,
/// `None` when the plan refuses it; a longer context never leaves more. `cap` is the model's
/// maximum. `None` when not even the baseline context fits, so the load then reports the plan's
/// own refusal for it.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family::flash_next) fn default_context(
    slots_at: &dyn Fn(u32) -> Option<usize>,
    min_slots: usize,
    cap: u32,
) -> Option<DefaultContext> {
    const GRANULE: u32 = 1024;
    let cap = cap.min(MAX_CUDA_CONTEXT) / CUDA_CONTEXT_MULTIPLE * CUDA_CONTEXT_MULTIPLE;
    let base_context = DEFAULT_CUDA_CONTEXT.min(cap);
    if base_context == 0 {
        return None;
    }
    let baseline_slots = slots_at(base_context)?;
    let floor = min_slots.max(baseline_slots - baseline_slots * BANK_SHARE_GIVEN_PERCENT / 100);
    let fits = |context: u32| slots_at(context).filter(|&slots| slots >= floor);
    let choice = |context: u32, slots: usize| {
        Some(DefaultContext {
            context,
            slots,
            baseline_slots,
        })
    };
    if let Some(slots) = fits(cap) {
        return choice(cap, slots);
    }
    // The baseline is known to fit; bisect over multiples of the granule above it.
    let mut low = base_context.div_ceil(GRANULE).min(cap / GRANULE);
    let mut high = cap / GRANULE;
    while low < high {
        let mid = (low + high).div_ceil(2);
        if fits(mid * GRANULE).is_some() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let context = (low * GRANULE).max(base_context);
    let slots = slots_at(context)?;
    choice(context, slots)
}

/// The model's maximum positions from its `config.json` (`max_position_embeddings`, in the
/// text configuration when there is one).
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family::flash_next) fn max_positions(config: &[u8]) -> Option<u32> {
    let root = serde_json::from_slice::<serde_json::Value>(config).ok()?;
    let text = root.get("text_config").unwrap_or(&root);
    let value = text.get("max_position_embeddings")?.as_u64()?;
    u32::try_from(value).ok().filter(|&value| value > 0)
}
/// Default row threshold of the prefill ring (`ring_rows`): on the shared prefill path,
/// chunks at or above it take the ring's whole-chunk path, chunks below it the bank's
/// union-admit path.
pub(crate) const DEFAULT_RING_ROWS: u32 = 32;

/// Whether a chunked-prefill chunk takes the ring's whole-chunk path: the ring is active and
/// the chunk reaches `ring_min_rows`, or the window already latched onto the ring
/// (`ring_engaged`). Below the threshold and not latched, the chunk takes the bank's
/// union-admit path. Pure so the threshold crossing is host-testable without a device.
pub(crate) fn ring_whole_chunk(
    ring_active: bool,
    rows: usize,
    ring_min_rows: usize,
    ring_engaged: bool,
) -> bool {
    ring_active && (rows >= ring_min_rows || ring_engaged)
}
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

impl FlashEnv {
    pub(crate) fn capture() -> Self {
        Self {
            manifest: std::env::var("FLASH_NEXT_ARTIFACT_MANIFEST").ok(),
            cache_limit: std::env::var("FLASH_NEXT_CACHE_LIMIT").ok(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlashSettings {
    /// The artifact directory with the safetensors shards and `config.json`.
    pub(crate) artifact: PathBuf,
    /// The directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and
    /// `generation_config.json`.
    pub(crate) tokenizer_dir: PathBuf,
    pub(crate) manifest: PathBuf,
    /// True when neither the config nor the environment named the manifest and it was found
    /// as `download-manifest.json` in the model or tokenizer directory.
    pub(crate) manifest_found_in_pack: bool,
    /// True when the CUDA decoder is on and the config sets no `context`: the memory plan
    /// chooses it at start (`default_context`) and `cuda.context` holds the placeholder.
    pub(crate) context_auto: bool,
    /// The MLX allocator cache limit in bytes (0 with the CUDA decoder, which ignores it).
    pub(crate) cache_limit: u64,
    pub(crate) prefill_budget: Option<u64>,
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

/// The file name a pack ships its manifest under.
const PACK_MANIFEST: &str = "download-manifest.json";

/// The manifest path: the config key, then `FLASH_NEXT_ARTIFACT_MANIFEST`, then
/// `download-manifest.json` in the model directory or the tokenizer directory. The flag says the
/// pack's own file was used.
fn manifest_path(
    options: &FlashOptions,
    env: &FlashEnv,
    model_dir: &Path,
    tokenizer_dir: &Path,
) -> Result<(PathBuf, bool), EngError> {
    if let Some(path) = &options.manifest {
        if path.as_os_str().is_empty() {
            return Err(bad("manifest must not be empty".into()));
        }
        return Ok((path.clone(), false));
    }
    if let Some(value) = env.manifest.as_deref().filter(|value| !value.is_empty()) {
        return Ok((PathBuf::from(value), false));
    }
    let mut looked = Vec::new();
    for dir in [model_dir, tokenizer_dir] {
        let path = dir.join(PACK_MANIFEST);
        if looked.contains(&path) {
            continue;
        }
        if path.is_file() {
            return Ok((path, true));
        }
        looked.push(path);
    }
    let names = looked
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(" or ");
    Err(bad(format!(
        "manifest is missing: set the manifest key or FLASH_NEXT_ARTIFACT_MANIFEST, or put \
         {PACK_MANIFEST} in the model directory (looked for {names})"
    )))
}

/// The ring budget from its environment variable's value, shared by the
/// loader and the preflight rows: unset is off (0) and a value that is not
/// a byte count refuses with the loader's own text, byte for byte. The
/// value is the raw budget — the plan ring and the reserved-bytes line
/// consume it as-is; only the free-memory carve saturates against it.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family::flash_next) fn ring_budget_bytes(
    spec: Option<&str>,
) -> Result<u64, EngError> {
    match spec {
        None => Ok(0),
        Some(text) => text.trim().parse::<u64>().map_err(|_| {
            EngError::BadRequest("INFERENCE_FLASH_NEXT_RING_BUDGET must be a byte count".into())
        }),
    }
}

/// The one whole-record rule both the loader's ring build and the tuner's
/// ring candidate read: how many whole expert records a byte budget buys,
/// capped at `cap_records` — the loader's twice-the-streamed horizon. The
/// loader sizes the ring's slots with it; the tuner shows the same capped
/// rule over the free VRAM, so the plan and the start cannot disagree on
/// the arithmetic.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family::flash_next) fn ring_record_count(
    budget_bytes: u64,
    record_bytes: u64,
    cap_records: u64,
) -> u64 {
    budget_bytes
        .checked_div(record_bytes)
        .unwrap_or(0)
        .min(cap_records)
}

/// The CUDA keys. Absent when `cuda_decode` is off; the keys are then refused if present, so a
/// misspelt switch cannot leave an operator believing the card is in use.
pub(in crate::family::flash_next) fn cuda_settings(
    options: &FlashOptions,
) -> Result<Option<CudaSettings>, EngError> {
    if !options.cuda_decode.unwrap_or(false) {
        let stray = [
            ("context", options.context.is_some()),
            ("kv_format", options.kv_format.is_some()),
            ("slots", options.slots.is_some()),
            ("expert_stats", options.expert_stats.is_some()),
            ("prefill_rows", options.prefill_rows.is_some()),
            ("ring_rows", options.ring_rows.is_some()),
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
    // The step plan refuses widths above the rows ceiling (`cuda/step_plan.rs`), which
    // floors at `SUPPORTED_EXL3_ROWS` of `project/exl3_jobs_plan.rs`; refusing here names
    // `prefill_rows` at parse time instead of a plan refusal at boot. The ceiling may be
    // raised above the supported width by `INFERENCE_EXL3_ROWS_MAX` (a measurement width
    // above the supported ceiling).
    let rows_ceiling = exl3_rows_ceiling().map_err(EngError::BadRequest)?;
    let prefill_rows = ranged(
        "prefill_rows",
        options.prefill_rows,
        256,
        1,
        rows_ceiling as u32,
    )?;
    // The stock threshold is 32: at 32 rows a top-10 union already outgrows a 200-slot
    // bank, so smaller chunks usually fit the bank without admission while larger ones
    // want the ring's whole-chunk grouping. A ring_rows above prefill_rows is accepted but
    // never engages the ring: every chunk is below the threshold, so the window stays on
    // the union-admit path (the ring's reservation is still carved at start).
    let ring_rows = bounded(
        "ring_rows",
        options.ring_rows.unwrap_or(DEFAULT_RING_ROWS),
        1,
        rows_ceiling as u32,
    )?;
    Ok(Some(CudaSettings {
        context,
        kv_format,
        slots,
        prefill_rows,
        ring_rows,
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
        Self::with_cuda_default(
            model_dir,
            tokenizer_dir,
            options,
            env,
            cfg!(all(feature = "native-cuda", target_os = "linux")),
        )
    }

    /// `new` with the default of `cuda_decode` given: a build with the CUDA decoder serves on
    /// the card unless the entry says `cuda_decode: false` or sets an MLX allocator key.
    fn with_cuda_default(
        model_dir: &Path,
        tokenizer_dir: &Path,
        options: Option<&FlashOptions>,
        env: &FlashEnv,
        cuda_default: bool,
    ) -> Result<Self, EngError> {
        let default = FlashOptions::default();
        let options = options.unwrap_or(&default);
        let resolved;
        // The MLX allocator keys name the MLX arm, so they keep an entry off the card.
        let mlx_keys =
            options.cache_limit_bytes.is_some() || options.prefill_budget_bytes.is_some();
        let options = if cuda_default && options.cuda_decode.is_none() && !mlx_keys {
            resolved = FlashOptions {
                cuda_decode: Some(true),
                ..options.clone()
            };
            &resolved
        } else {
            options
        };
        let cuda = cuda_settings(options)?;
        let (manifest, manifest_found_in_pack) =
            manifest_path(options, env, model_dir, tokenizer_dir)?;
        if let Some(budget) = options.prefill_budget_bytes {
            if cuda.is_some() {
                return Err(bad(
                    "prefill_budget_bytes is an MLX key and cannot be combined with cuda_decode"
                        .into(),
                ));
            }
            if budget == 0 {
                return Err(bad("prefill_budget_bytes must be above 0".into()));
            }
        }
        Ok(Self {
            artifact: model_dir.to_path_buf(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            manifest,
            manifest_found_in_pack,
            context_auto: cuda.is_some() && options.context.is_none(),
            // An MLX allocator key: the CUDA decoder has no use for it.
            cache_limit: if cuda.is_some() {
                0
            } else {
                cache_limit(options, env)?
            },
            prefill_budget: options.prefill_budget_bytes,
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

/// Whether CUDA prefill reads each chunk's distinct n-gram rows through the row pool. On by
/// default; `INFERENCE_FLASH_NEXT_NGRAM_BULK=0` reads them one token at a time and `1` names the
/// default. Strict: any other set value, a non-text value included, is refused.
pub(in crate::family::flash_next) fn ngram_bulk() -> Result<bool, EngError> {
    let raw = std::env::var_os("INFERENCE_FLASH_NEXT_NGRAM_BULK");
    ngram_bulk_text(raw.as_ref().map(|value| value.to_string_lossy()).as_deref())
}

/// The text core of [`ngram_bulk`], split out so the strict rule is testable: this crate denies
/// `unsafe_code`, so a test cannot carry the variable into the env reader itself.
pub(in crate::family::flash_next) fn ngram_bulk_text(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(_) => Err(EngError::BadRequest(
            "INFERENCE_FLASH_NEXT_NGRAM_BULK must be unset, \"0\" or \"1\"".into(),
        )),
    }
}

/// Whether the chunked prefill runs its two PLE projections on the card (the
/// `dense_gemv` pair): `INFERENCE_FLASH_NEXT_PLE_DEVICE=1` opts in, the default keeps the host
/// matvecs. The card uses the trunk's stored F16/BF16 weights and reserves workspace for the
/// token rows. The rest of the PLE block stays on the host. Strict: any set
/// value other than `1` is refused.
pub(in crate::family::flash_next) fn ple_device() -> Result<bool, EngError> {
    ple_device_text(
        std::env::var("INFERENCE_FLASH_NEXT_PLE_DEVICE")
            .ok()
            .as_deref(),
    )
}

/// The text core of [`ple_device`], split out so the strict rule is
/// testable: this crate denies `unsafe_code`, so a test cannot carry the
/// variable into the env reader itself.
pub(in crate::family::flash_next) fn ple_device_text(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        Some("1") => Ok(true),
        Some(_) => Err(EngError::BadRequest(
            "INFERENCE_FLASH_NEXT_PLE_DEVICE must be unset or \"1\"".into(),
        )),
        None => Ok(false),
    }
}

/// Whether chunked prefill batches its two PLE host projections into one blocked product per
/// chunk (`INFERENCE_FLASH_NEXT_PLE_BATCH=1` opts in): the chunk's embeddings project at once —
/// the 131 MB of weight matrices stream once per token block instead of once per token — and
/// each row's pair then feeds the unchanged host step through the fixed projection, so the
/// numbers stay bit-identical to the per-token path. Strict: any set value other than `1` is
/// refused.
pub(in crate::family::flash_next) fn ple_batch() -> Result<bool, EngError> {
    ple_batch_text(
        std::env::var("INFERENCE_FLASH_NEXT_PLE_BATCH")
            .ok()
            .as_deref(),
    )
}

/// The text core of [`ple_batch`], split out so the strict rule is
/// testable: this crate denies `unsafe_code`, so a test cannot carry the
/// variable into the env reader itself.
pub(in crate::family::flash_next) fn ple_batch_text(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        Some("1") => Ok(true),
        Some(_) => Err(EngError::BadRequest(
            "INFERENCE_FLASH_NEXT_PLE_BATCH must be unset or \"1\"".into(),
        )),
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CudaSettings, FlashEnv, FlashOptions, FlashSettings, KvFormatName, ring_whole_chunk,
    };
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

    /// The settings of an entry with the build's default of `cuda_decode` pinned off, so the
    /// expectations hold in a CUDA build too.
    fn settings(options: Option<&FlashOptions>, env: &FlashEnv) -> Result<FlashSettings, EngError> {
        FlashSettings::with_cuda_default(Path::new("/art"), Path::new("/tok"), options, env, false)
    }

    fn message(result: Result<FlashSettings, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn ring_record_count_buys_whole_records_on_real_geometry() {
        use super::ring_record_count;
        // The pack's real geometry: a 1,881,600-byte expert record, the
        // free VRAM of a 24 GB card, and the loader's horizon of twice the
        // 512 streamed experts per layer.
        const RECORD: u64 = 1_881_600;
        let records = ring_record_count(22_000_000_000, RECORD, 1_024);
        assert!(records > 0, "a 24 GB card budgets whole records: {records}");
        // The pack's 512 streamed experts all fit under the cap with room.
        assert!(records >= 512, "512 experts of this record fit: {records}");
        // The cap is the loader's horizon: twice the streamed experts.
        assert_eq!(ring_record_count(22_000_000_000, RECORD, 8), 8);
        // A budget short of one record buys none, and no record size buys none.
        assert_eq!(ring_record_count(RECORD - 1, RECORD, 1_024), 0);
        assert_eq!(ring_record_count(22_000_000_000, 0, 1_024), 0);
        assert_eq!(ring_record_count(0, RECORD, 1_024), 0);
    }

    #[test]
    fn ring_budget_bytes_parses_the_raw_value_and_refuses_with_start_text() {
        use super::ring_budget_bytes;
        assert_eq!(ring_budget_bytes(None).unwrap(), 0);
        assert_eq!(ring_budget_bytes(Some("0")).unwrap(), 0);
        assert_eq!(ring_budget_bytes(Some(" 1073741824 ")).unwrap(), 1 << 30);
        // The raw budget survives above the free card: the plan ring's slot
        // count and the reserved-bytes line consume it as-is, and only the
        // carve saturates (free_vram - budget), so a budget above the card
        // funds nothing at the carve and changes no other line.
        assert_eq!(
            ring_budget_bytes(Some("99999999999")).unwrap(),
            99_999_999_999
        );
        // Start's own refusal text, byte for byte.
        assert_eq!(
            ring_budget_bytes(Some("1 GiB")).unwrap_err().describe(),
            "INFERENCE_FLASH_NEXT_RING_BUDGET must be a byte count"
        );
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
                manifest_found_in_pack: false,
                context_auto: false,
                cache_limit: 1_073_741_824,
                prefill_budget: None,
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
                manifest_found_in_pack: false,
                context_auto: false,
                cache_limit: 2048,
                prefill_budget: None,
                tokenizer_vocab_size: 248_077,
                prefill_chunk: 16,
                expert_prefetch: false,
                cuda: None,
            }
        );
    }

    #[test]
    fn a_cuda_build_serves_on_the_card_unless_the_entry_turns_it_off() {
        let pack = tempfile::tempdir().expect("tempdir");
        std::fs::write(pack.path().join("download-manifest.json"), b"{}").expect("write");
        let build = |options: Option<&FlashOptions>, cuda_default: bool| {
            FlashSettings::with_cuda_default(
                pack.path(),
                pack.path(),
                options,
                &FlashEnv::default(),
                cuda_default,
            )
        };
        // Nothing but the directories: the card, with every default of the decoder.
        let on = build(None, true).unwrap();
        assert_eq!(on.cache_limit, 0);
        let cuda = on.cuda.expect("the CUDA decoder is on by default");
        assert_eq!(cuda.context, super::DEFAULT_CUDA_CONTEXT);
        // The same entry without the decoder in the build is the MLX arm, which needs its limit.
        assert!(
            message(build(None, false)).contains("cache_limit_bytes is missing"),
            "an MLX-only build keeps the MLX settings"
        );
        // An explicit false keeps the MLX arm on a CUDA build.
        let off = FlashOptions {
            cuda_decode: Some(false),
            cache_limit_bytes: Some(1 << 30),
            ..FlashOptions::default()
        };
        assert_eq!(build(Some(&off), true).unwrap().cuda, None);
        // An MLX allocator key with no `cuda_decode` names the MLX arm, so it stays off the card.
        let mlx_key = FlashOptions {
            cache_limit_bytes: Some(1 << 30),
            ..FlashOptions::default()
        };
        assert_eq!(build(Some(&mlx_key), true).unwrap().cuda, None);
        // The same key beside an explicit `cuda_decode: true` is the conflict, named.
        let both = FlashOptions {
            cuda_decode: Some(true),
            ..mlx_key
        };
        assert!(message(build(Some(&both), true)).contains("cannot be combined with cuda_decode"));
    }

    /// A plan whose slots fall by one per 2,048 positions from `top`.
    fn falling(top: usize) -> impl Fn(u32) -> Option<usize> {
        move |context| top.checked_sub(context as usize / 2048)
    }

    #[test]
    fn the_default_context_stops_where_the_bank_would_lose_a_tenth() {
        use super::default_context;
        // 198 slots at 4,096: a tenth is 19 slots, so the longest context keeps 179 or more.
        let chosen = default_context(&falling(200), 11, 262_144).expect("a plan");
        assert_eq!(chosen.baseline_slots, 198);
        assert_eq!(chosen.context, 44_032);
        assert_eq!(chosen.slots, 179);
        assert!(falling(200)(chosen.context + 1024).unwrap() < 179);
        // The cap is the model maximum.
        let capped = default_context(&falling(200), 11, 16_384).expect("a plan");
        assert_eq!((capped.context, capped.slots), (16_384, 192));
        // A model smaller than the baseline context serves its own maximum.
        let tiny = default_context(&falling(200), 11, 2_048).expect("a plan");
        assert_eq!(tiny.context, 2_048);
        assert_eq!(tiny.baseline_slots, falling(200)(2_048).unwrap());
        // The minimum slots of the step are a floor of their own.
        let floor = default_context(&falling(200), 190, 262_144).expect("a plan");
        assert_eq!(floor.context, 21_504);
        assert!(floor.slots >= 190);
        // A plan that refuses the baseline gives no default, so the load reports its refusal.
        assert_eq!(default_context(&|_| None, 11, 262_144), None);
        // A plan that refuses long contexts only: the longest admitted one wins.
        let short = |context: u32| (context <= 24_576).then_some(100);
        assert_eq!(
            default_context(&short, 11, 262_144).map(|chosen| chosen.context),
            Some(24_576)
        );
        // A flat plan serves the whole model.
        assert_eq!(
            default_context(&|_| Some(50), 11, 262_144).map(|chosen| chosen.context),
            Some(262_144)
        );
    }

    #[test]
    fn the_model_maximum_is_read_from_the_text_configuration_or_the_top_level() {
        use super::max_positions;
        assert_eq!(
            max_positions(br#"{"text_config":{"max_position_embeddings":262144}}"#),
            Some(262_144)
        );
        assert_eq!(
            max_positions(br#"{"max_position_embeddings":32768}"#),
            Some(32_768)
        );
        assert_eq!(max_positions(br#"{"text_config":{}}"#), None);
        assert_eq!(max_positions(b"not json"), None);
        assert_eq!(
            max_positions(include_bytes!(
                "../../../tests/fixtures/qwen3_8_flash_next_config.json"
            )),
            Some(262_144)
        );
    }

    #[test]
    fn a_context_in_the_config_is_not_chosen_by_the_plan() {
        let pack = tempfile::tempdir().expect("tempdir");
        std::fs::write(pack.path().join("download-manifest.json"), b"{}").expect("write");
        let build = |context: Option<u32>| {
            let options = FlashOptions {
                context,
                ..FlashOptions::default()
            };
            FlashSettings::with_cuda_default(
                pack.path(),
                pack.path(),
                Some(&options),
                &FlashEnv::default(),
                true,
            )
            .unwrap()
        };
        let auto = build(None);
        assert!(auto.context_auto);
        let set = build(Some(8192));
        assert!(!set.context_auto);
        assert_eq!(set.cuda.unwrap().context, 8192);
    }

    #[test]
    fn a_manifest_in_the_pack_directory_is_used_when_nothing_names_one() {
        let pack = tempfile::tempdir().expect("tempdir");
        let tokens = tempfile::tempdir().expect("tempdir");
        let options = FlashOptions {
            manifest: None,
            ..full()
        };
        let build =
            |env: &FlashEnv| FlashSettings::new(pack.path(), tokens.path(), Some(&options), env);
        // Not shipped anywhere: the refusal names both files it looked for.
        let text = message(build(&FlashEnv::default()));
        assert!(
            text.starts_with("engine config: manifest is missing"),
            "{text}"
        );
        assert!(
            text.contains(
                &pack
                    .path()
                    .join("download-manifest.json")
                    .display()
                    .to_string()
            ) && text.contains(
                &tokens
                    .path()
                    .join("download-manifest.json")
                    .display()
                    .to_string()
            ),
            "{text}"
        );
        // Shipped with the tokenizer files only.
        std::fs::write(tokens.path().join("download-manifest.json"), b"{}").expect("write");
        let got = build(&FlashEnv::default()).unwrap();
        assert_eq!(got.manifest, tokens.path().join("download-manifest.json"));
        assert!(got.manifest_found_in_pack);
        // Shipped with the model: the model directory wins.
        std::fs::write(pack.path().join("download-manifest.json"), b"{}").expect("write");
        let got = build(&FlashEnv::default()).unwrap();
        assert_eq!(got.manifest, pack.path().join("download-manifest.json"));
        assert!(got.manifest_found_in_pack);
        // The environment variable and the config key both win over the pack's file.
        let env = FlashEnv {
            manifest: Some("/env/m.json".into()),
            cache_limit: Some("1".into()),
        };
        let got = build(&env).unwrap();
        assert_eq!(got.manifest, PathBuf::from("/env/m.json"));
        assert!(!got.manifest_found_in_pack);
        let named = FlashOptions {
            manifest: Some(PathBuf::from("/cfg/m.json")),
            ..options.clone()
        };
        let got = FlashSettings::new(pack.path(), tokens.path(), Some(&named), &env).unwrap();
        assert_eq!(got.manifest, PathBuf::from("/cfg/m.json"));
        assert!(!got.manifest_found_in_pack);
    }

    #[test]
    fn prefill_budget_is_an_optional_positive_mlx_byte_count() {
        let options = serde_json::from_str::<FlashOptions>(
            r#"{"manifest":"/m","cache_limit_bytes":1,"prefill_budget_bytes":42}"#,
        )
        .unwrap();
        let env = FlashEnv::default();
        assert_eq!(
            settings(Some(&options), &env).unwrap().prefill_budget,
            Some(42)
        );
        let zero = FlashOptions {
            prefill_budget_bytes: Some(0),
            ..options
        };
        assert!(
            message(settings(Some(&zero), &env)).contains("prefill_budget_bytes must be above 0")
        );
        let cuda = FlashOptions {
            prefill_budget_bytes: Some(42),
            ..cuda_on()
        };
        assert!(
            message(settings(Some(&cuda), &env)).contains("cannot be combined with cuda_decode")
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
            "engine config: manifest is missing: set the manifest key or \
             FLASH_NEXT_ARTIFACT_MANIFEST, or put download-manifest.json in the model directory \
             (looked for /art/download-manifest.json or /tok/download-manifest.json)"
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
                "prefill_chunk must be between 1 and 2048, found 0",
            ),
            (
                FlashOptions {
                    prefill_chunk: Some(2049),
                    ..full()
                },
                "prefill_chunk must be between 1 and 2048, found 2049",
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

    /// `prefill_rows` takes the supported width (`SUPPORTED_EXL3_ROWS`, no measurement
    /// variable) and refuses one row above it at parse time, naming the key. The
    /// measurement variable still widens past it (pinned in `exl3_jobs_plan`).
    #[test]
    fn prefill_rows_is_bounded_at_the_supported_width() {
        let env = FlashEnv::default();
        let at = |rows: u32| {
            let options = FlashOptions {
                prefill_rows: Some(rows),
                ..cuda_on()
            };
            settings(Some(&options), &env)
        };
        assert_eq!(at(256).unwrap().cuda.unwrap().prefill_rows, 256);
        assert_eq!(at(1).unwrap().cuda.unwrap().prefill_rows, 1);
        // The widened window is a supported setting now: 1,024 and 2,048 rows parse with
        // no measurement variable; one row above the supported ceiling still refuses.
        assert_eq!(at(1024).unwrap().cuda.unwrap().prefill_rows, 1024);
        assert_eq!(at(2048).unwrap().cuda.unwrap().prefill_rows, 2048);
        assert!(message(at(2049)).contains("prefill_rows"));
        assert!(message(at(0)).contains("prefill_rows"));
    }

    /// `ring_rows` defaults to the stock 32, takes any value up to the rows ceiling (the
    /// shared supported width), refuses values outside it at parse time naming the key, and
    /// stays refused without
    /// `cuda_decode`.
    #[test]
    fn ring_rows_defaults_to_32_and_is_bounded_at_the_supported_width() {
        let env = FlashEnv::default();
        let at = |rows: Option<u32>| {
            let options = FlashOptions {
                ring_rows: rows,
                ..cuda_on()
            };
            settings(Some(&options), &env)
        };
        assert_eq!(at(None).unwrap().cuda.unwrap().ring_rows, 32);
        assert_eq!(at(Some(256)).unwrap().cuda.unwrap().ring_rows, 256);
        assert_eq!(at(Some(1)).unwrap().cuda.unwrap().ring_rows, 1);
        // The bound shares the rows ceiling: a ring threshold at the supported width
        // parses with no measurement variable; one above the ceiling still refuses.
        assert_eq!(at(Some(2048)).unwrap().cuda.unwrap().ring_rows, 2048);
        assert!(message(at(Some(2049))).contains("ring_rows"));
        assert!(message(at(Some(0))).contains("ring_rows"));
        let stray = FlashOptions {
            ring_rows: Some(32),
            ..FlashOptions::default()
        };
        assert!(message(settings(Some(&stray), &env)).contains("ring_rows"));
    }

    #[test]
    fn the_ring_whole_chunk_predicate_crosses_at_the_threshold_and_latches() {
        // At the threshold the ring serves; one row below it the union admits.
        assert!(ring_whole_chunk(true, 32, 32, false));
        assert!(!ring_whole_chunk(true, 31, 32, false));
        // Once latched, even a 1-row tail stays on the ring.
        assert!(ring_whole_chunk(true, 1, 32, true));
        // No active ring: always the union path, latched or not.
        assert!(!ring_whole_chunk(false, 256, 32, false));
        assert!(!ring_whole_chunk(false, 1, 32, true));
        // Threshold 1: every chunk is whole.
        assert!(ring_whole_chunk(true, 1, 1, false));
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
        // Pin the shared mode: the default-rows expectation must hold whatever
        // the shell exported.
        let got = settings(Some(&cuda_on()), &FlashEnv::default()).unwrap();
        assert_eq!(got.cache_limit, 0);
        assert_eq!(
            got.cuda,
            Some(CudaSettings {
                context: 4096,
                kv_format: KvFormatName::F16,
                slots: None,
                expert_stats: None,
                // Unset rows resolve the mode-dependent default: the shared
                // path (env unset in tests) yields 256.
                prefill_rows: 256,
                ring_rows: 32,
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
                // No prefill_rows in the options: the shared-path default.
                prefill_rows: 256,
                ring_rows: 32,
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

    #[test]
    fn the_ngram_bulk_text_is_on_by_default_and_off_only_for_zero() {
        assert!(super::ngram_bulk_text(None).unwrap());
        assert!(super::ngram_bulk_text(Some("1")).unwrap());
        assert!(!super::ngram_bulk_text(Some("0")).unwrap());
        for bad in ["", "2", "yes", "true", "off", " 0", "00", "\u{fffd}"] {
            assert_eq!(
                super::ngram_bulk_text(Some(bad)).unwrap_err().describe(),
                "INFERENCE_FLASH_NEXT_NGRAM_BULK must be unset, \"0\" or \"1\"",
                "{bad:?}"
            );
        }
    }

    #[test]
    fn the_ple_switch_texts_refuse_anything_but_one() {
        assert!(!super::ple_device_text(None).unwrap());
        assert!(super::ple_device_text(Some("1")).unwrap());
        assert_eq!(
            super::ple_device_text(Some("2")).unwrap_err().describe(),
            "INFERENCE_FLASH_NEXT_PLE_DEVICE must be unset or \"1\""
        );
        assert!(!super::ple_batch_text(None).unwrap());
        assert!(super::ple_batch_text(Some("1")).unwrap());
        assert_eq!(
            super::ple_batch_text(Some("yes")).unwrap_err().describe(),
            "INFERENCE_FLASH_NEXT_PLE_BATCH must be unset or \"1\""
        );
    }
}
