//! Starts the Q36 CUDA engine of one `enginesrv` config entry: the tokenizer bundle and the
//! rules on the calling thread, then the engine thread builds everything CUDA inside `make` (no
//! CUDA value crosses a thread boundary).
//!
//! Lifetime: `spawn_engine` needs an executor without a lifetime and `QwenResidentStep` borrows
//! its owners, so the projector, the weights and the step resources are `Box::leak`ed once per
//! engine, giving `QwenResidentStep<'static>` (an owning `Arc` variant would rewrite the
//! qualified step, >= 1,000 lines). ONE CUDA engine is built per process: `BuildGuard` pins the
//! flag before `spawn_engine`, commits it right after the leaks, and every retry in the same
//! process is refused.

use super::super::settings::{
    DEFAULT_CUDA_PREFILL_CHUNK, KvDtype, NO_BUDGET_PIECE, Q36Settings, tokenizer_pin,
};
use super::{
    dims::{KvMode, StepDims, StepOptions, TieredPlan},
    pinned::{MIN_FREE_AFTER_LOAD, ROTARY_DIM, head_rows_of},
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    prefill_plan::Extra,
    resources::{PagedResourcesPlan, STEP_ROWS, StepResources, check_paged_modes},
    rules::QwenRules,
    state_layout::{KvPool, StateLayout},
    step::{CaptureConfig, ChunkVerdict, QwenResidentStep},
    step_plan::{KernelScratch, StepPlan},
    tiered_load::{attach_tiered_parts, names_without_experts},
};
use crate::{
    EngError,
    family::{
        QWEN3_6_35B_A3B,
        gguf_model::{Inventory, open_qwen3_6_35b_a3b},
        project::cuda::{CudaProjector, CudaWeights, LoadOptions, Reserve, WeightLayout},
        qwen36::host::HostShape,
        session::DecodeSession,
    },
    sched::worker::{Engine, ReadyNote, spawn_engine_noted},
    tokenizer::TokenizerBundle,
};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrefixMemory {
    Disabled,
    Pinned,
    DeviceTip,
}

impl PrefixMemory {
    fn from_flags(
        reuse: bool,
        pinned: Option<&std::ffi::OsStr>,
        device: Option<&std::ffi::OsStr>,
        default_on: bool,
    ) -> Self {
        let device = device.map_or(default_on, |value| value == "1");
        // Explicit device requests retain their pinned parent unless the pinned kill switch wins.
        let pinned = pinned.map_or(default_on || device, |value| value == "1");
        match (reuse && pinned, device) {
            (false, _) => Self::Disabled,
            (true, true) => Self::DeviceTip,
            (true, false) => Self::Pinned,
        }
    }
}

struct RuntimeModes {
    chunked: bool,
    chunk_rows: u32,
    context_staging: bool,
    fixed_admission: bool,
    wide_stage: bool,
    prefix: PrefixMemory,
}

struct StartupPlan {
    reserve: u64,
    paged: Option<PagedResourcesPlan>,
    device_first_native: Option<usize>,
    verify: bool,
    modes: RuntimeModes,
}

impl StartupPlan {
    /// Reserve only fixed storage before weights; exact native fit wins after the load.
    fn new_device_first(
        settings: &Q36Settings,
        verify: bool,
        modes: RuntimeModes,
    ) -> Result<Self, EngError> {
        let Some(limit) = settings.kv_offload_host_bytes else {
            return Self::new(settings, verify, modes);
        };
        let kv_mode = serve_kv_mode(settings)?;
        if kv_mode != KvMode::Int8G32 || settings.tiered.is_some() {
            return Err(EngError::BadRequest(
                "device-first KV requires Native Int8G32 without tiered experts".into(),
            ));
        }
        if limit == 0 || usize::try_from(limit).is_err() {
            return Err(EngError::BadRequest(
                "kv_offload_host_bytes must be positive and fit usize".into(),
            ));
        }
        settings.check_prefix_reuse_cap(settings.prefix_reuse)?;
        check_paged_modes(true, settings.capture, modes.chunked)?;
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
        let dims = StepDims::new(&shape, &inventory)?;
        let options = step_options(settings.context_limit, WeightLayout::Native, kv_mode)?;
        let state = StateLayout::new(&dims, &options)?;
        let scratch = KernelScratch::for_dims(&dims, STEP_ROWS, kv_mode)?;
        let step = StepPlan::new(&dims, STEP_ROWS, &options, &scratch)?;
        let reserve = state
            .bytes()?
            .checked_add(step.arena_bytes())
            .ok_or_else(|| {
                EngError::MemoryPlan("device-first resident reserve size overflows".into())
            })?;
        let native = reserve;
        let fixed = PagedResourcesPlan::fixed_device_bytes(&dims, &options)?;
        let reserve = u64::try_from(fixed.min(native))
            .map_err(|_| EngError::MemoryPlan("device-first fixed reserve exceeds u64".into()))?;
        Ok(Self {
            reserve,
            paged: None,
            device_first_native: Some(native),
            verify,
            modes,
        })
    }

    fn after_load(mut self, settings: &Q36Settings, free: usize) -> Result<Self, EngError> {
        let Some(native) = self.device_first_native else {
            return Ok(self);
        };
        let available = free.checked_sub(MIN_FREE_AFTER_LOAD).ok_or_else(|| {
            EngError::MemoryPlan("device-first load is below the free-memory floor".into())
        })?;
        if native <= available {
            self.reserve = u64::try_from(native)
                .map_err(|_| EngError::MemoryPlan("native reserve exceeds u64".into()))?;
            return Ok(self);
        }
        let fixed = usize::try_from(self.reserve)
            .map_err(|_| EngError::MemoryPlan("fixed cache reserve exceeds usize".into()))?;
        let budget = available.checked_sub(fixed).ok_or_else(|| {
            EngError::MemoryPlan("device-first fixed resources exceed free VRAM".into())
        })?;
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
        let dims = StepDims::new(&shape, &inventory)?;
        let options = step_options(
            settings.context_limit,
            WeightLayout::Native,
            KvMode::Int8G32,
        )?;
        let host_limit = settings
            .kv_offload_host_bytes
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                EngError::BadRequest("device-first host quota is absent or invalid".into())
            })?;
        let cache = super::paging_store::device_first::DeviceCachePlan::new(
            &dims,
            options.context_tokens,
            budget,
            host_limit,
        )?;
        let paged = PagedResourcesPlan::new_cached(dims, &inventory, &options, cache)?;
        let device_bytes = paged.device_bytes()?;
        if device_bytes > available {
            return Err(EngError::MemoryPlan(
                "device cache resources exceed post-load VRAM".into(),
            ));
        }
        self.reserve = u64::try_from(device_bytes)
            .map_err(|_| EngError::MemoryPlan("device cache reserve exceeds u64".into()))?;
        self.paged = Some(paged);
        Ok(self)
    }

    fn new(settings: &Q36Settings, verify: bool, modes: RuntimeModes) -> Result<Self, EngError> {
        let kv_mode = serve_kv_mode(settings)?;
        let capacity = context_tokens(settings.context_limit, kv_mode)?;
        let paged = match settings.kv_offload_host_bytes {
            None => None,
            Some(limit) => {
                settings.check_prefix_reuse_cap(settings.prefix_reuse)?;
                check_paged_modes(true, settings.capture, modes.chunked)?;
                let limit = usize::try_from(limit).map_err(|_| {
                    EngError::BadRequest("kv_offload_host_bytes exceeds usize".into())
                })?;
                let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
                let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
                let dims = StepDims::new(&shape, &inventory)?;
                let mut options =
                    step_options(settings.context_limit, WeightLayout::Native, kv_mode)?;
                options.tiered = tiered_plan_of(settings);
                Some(PagedResourcesPlan::new(dims, &inventory, &options, limit)?)
            }
        };
        let reserve = match &paged {
            Some(plan) => u64::try_from(plan.device_bytes()?)
                .map_err(|_| EngError::MemoryPlan("paged resource reserve exceeds u64".into()))?,
            None => kv_cache_bytes(capacity, kv_mode)?,
        };
        Ok(Self {
            reserve,
            paged,
            device_first_native: None,
            verify,
            modes,
        })
    }
}

/// The pinned checkpoint of the CUDA arm: the file NAME inside `model_dir` and its exact SIZE,
/// both matching the pinned identity the qualification tool checks before every gated run. The
/// full-file SHA-256 beside them is checked only when `Q36_VERIFY_GGUF_SHA256` is `1`: hashing
/// 20 GB costs about 15-40 s per start (ESTIMATE) and the qualification script checks it.
pub(super) const GGUF_NAME: &str = "Qwen3.6-35B-A3B-Q4_K_M.gguf";
pub(super) const GGUF_BYTES: u64 = 20_419_565_568;
const GGUF_SHA256: &str = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7";
/// The full-file SHA-256 is checked only when this variable is EXACTLY `1`: unset or `0`
/// leaves the check off, and every OTHER spelling — `true`, `yes`, an empty value — is a
/// typed refusal (raised in `make`, after the CUDA context exists), so a misspelled promise
/// cannot silently skip the check. A value Rust cannot read as UTF-8 reads as unset (std's
/// `env::var`), so it leaves the check off.
const VERIFY_SHA256_VAR: &str = "Q36_VERIFY_GGUF_SHA256";
const CHUNK_PREFILL_VAR: &str = "Q36_CUDA_CHUNK_PREFILL";
const CHUNK_ROWS_VAR: &str = "Q36_CUDA_CHUNK_ROWS";
const CONTEXT_STAGING_VAR: &str = "Q36_CUDA_KV8_STAGE_CONTEXT";
const FIXED_ADMISSION_VAR: &str = "Q36_CUDA_KV8_FIXED_ADMISSION";
const DEVICE_FIRST_VAR: &str = "Q36_CUDA_PAGED_DEVICE_FIRST";
pub(super) const WIDE_STAGE_VAR: &str = "Q36_CUDA_CPF_STAGE_128K";

fn device_first_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err(EngError::BadRequest(format!(
            "{DEVICE_FIRST_VAR} must be 0 or 1"
        ))),
    }
}

fn require_device_first_room(free: usize, reserve: u64) -> Result<(), EngError> {
    let needed = usize::try_from(reserve)
        .ok()
        .and_then(|bytes| bytes.checked_add(MIN_FREE_AFTER_LOAD))
        .ok_or_else(|| {
            EngError::MemoryPlan("device-first resident reserve plus headroom exceeds usize".into())
        })?;
    if free < needed {
        return Err(EngError::MemoryPlan(format!(
            "device-first fully-resident KV needs {needed} free bytes, found {free}"
        )));
    }
    Ok(())
}

fn chunk_prefill_rows(value: Option<&std::ffi::OsStr>) -> Result<u32, EngError> {
    match value {
        None => Ok(1024),
        Some(value) if value == "16" => Ok(16),
        Some(value) if value == "128" => Ok(128),
        Some(value) if value == "512" => Ok(512),
        Some(value) if value == "1024" => Ok(1024),
        Some(_) => Err(EngError::BadRequest(format!(
            "{CHUNK_ROWS_VAR} must be 16, 128, 512 or 1024"
        ))),
    }
}

fn chunk_prefill_wanted(kv: KvDtype, value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(CHUNK_PREFILL_VAR, matches!(kv, KvDtype::Q8), value)
}

fn context_staging_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(CONTEXT_STAGING_VAR, true, value)
}

/// The wide 131072-row staged-split candidate is DEFAULT ON (against the 32768-row stage:
/// -7.4% TTFT at 128K, token-identical at 149731 and 127788 tokens, same binary, cold boots).
/// `Q36_CUDA_CPF_STAGE_128K=0` rolls it back to the 32768-row default stage.
fn wide_stage_wanted(value: Option<&std::ffi::OsStr>) -> Result<bool, EngError> {
    prefill_flag(WIDE_STAGE_VAR, true, value)
}

fn fixed_admission_wanted(
    kv: KvDtype,
    chunked: bool,
    context_staging: bool,
    value: Option<&std::ffi::OsStr>,
) -> Result<bool, EngError> {
    Ok(prefill_flag(FIXED_ADMISSION_VAR, true, value)?
        && matches!(kv, KvDtype::Q8)
        && chunked
        && context_staging)
}

/// Why the wide CPF stage flag is inert, when it is: the wide candidate only
/// exists inside fixed admission, so name the mode that dropped fixed admission
/// (or that fixed admission itself is off) instead of failing silently.
fn wide_stage_inert_reason(
    fixed_admission: bool,
    kv: KvDtype,
    chunked: bool,
    context_staging: bool,
    tiered: bool,
    host_offload: bool,
) -> Option<&'static str> {
    if fixed_admission {
        return None;
    }
    Some(if host_offload {
        "host KV offload"
    } else if tiered {
        "tiered mode"
    } else if !matches!(kv, KvDtype::Q8) {
        "non-Q8 KV dtype"
    } else if !chunked {
        "chunked prefill off (Q36_CUDA_CHUNK_PREFILL=0)"
    } else if !context_staging {
        "context staging off (Q36_CUDA_KV8_STAGE_CONTEXT=0)"
    } else {
        "fixed admission off (Q36_CUDA_KV8_FIXED_ADMISSION=0)"
    })
}

pub(super) fn prefill_headroom(fixed_admission: bool) -> usize {
    if fixed_admission {
        FIXED_ADMISSION_MARGIN
    } else {
        HEADROOM as usize
    }
}

fn prefill_flag(
    name: &str,
    default: bool,
    value: Option<&std::ffi::OsStr>,
) -> Result<bool, EngError> {
    match value {
        None => Ok(default),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err(EngError::BadRequest(format!("{name} must be 0 or 1"))),
    }
}
/// Legacy admission headroom. The global free-after-load floor stays at 1 GiB even when
/// fixed admission uses its measured margin; `work: 0` is explained by `load_options`.
const HEADROOM: u64 = 1 << 30;

/// The process-wide one-engine flag: `true` from the first leak of a build onwards.
pub(super) static CUDA_ENGINE_BUILT: AtomicBool = AtomicBool::new(false);

/// The refusal of a second build in one process (the documented text).
pub(super) fn already_built() -> EngError {
    EngError::BadRequest("engine config: the CUDA Q36 backend builds one engine per process: its weights are leaked for the process lifetime and an in-process restart would leak 20 GB again; configure a single qwen3_6_35b_a3b entry and restart the process to reload".to_string())
}

/// The one-engine guard: `acquire` takes the flag before `spawn_engine`, `commit` pins it
/// right after the first leak, and the `Drop` releases it only for a failure BEFORE the first
/// leak, when nothing is on the device and a retry may build.
pub(super) struct BuildGuard {
    committed: bool,
}

impl BuildGuard {
    pub(super) fn acquire() -> Result<Self, EngError> {
        CUDA_ENGINE_BUILT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| already_built())?;
        Ok(Self { committed: false })
    }

    /// Pins the flag: a failure from here on leaves the ~20 GB on the device, so a retry in
    /// the same process must be refused, not run.
    pub(super) fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for BuildGuard {
    /// Releases the flag only uncommitted: a failure before the first leak owns nothing.
    fn drop(&mut self) {
        if !self.committed {
            CUDA_ENGINE_BUILT.store(false, Ordering::Release);
        }
    }
}

/// The start line of one MLX-only key the CUDA arm ignores (the settings record the names in the
/// order of `Q36Options`). `prefill_chunk` keeps its line until chunked CUDA prefill accepts the
/// key as the cap of `M_sess`.
pub(crate) fn ignored_mlx_line(key: &str) -> &'static str {
    match key {
        "cache_limit_bytes" => "q36 cuda: cache_limit_bytes is an MLX allocator key and is ignored",
        "session_cap_bytes" => "q36 cuda: session_cap_bytes is an MLX session key and is ignored",
        "prefill_chunk" => "q36 cuda: prefill_chunk is an MLX chunked-prefill key and is ignored",
        "prefix_store_entries" => {
            "q36 cuda: prefix_store_entries is an MLX prefix-store key and is ignored"
        }
        _ => "q36 cuda: an unknown key of the object is ignored",
    }
}

/// The start line of the session's prefill path: one line per session build, printed after the
/// arena admission — the accept names the ADMITTED `M_sess`, every refusal names its reason, and
/// the match is exhaustive (no `_` arm), so a new verdict cannot go unlabelled. `make` prints it
/// once `admit_prefill` returns; the text is pinned by the host rows of `prefill_driver_tests`.
/// `pub(super)` like the enum it takes: a `pub(crate)` fn cannot expose the `pub(super)` verdict
/// type, and every caller — `make`, the host row, the device row — is in `cuda`.
pub(super) fn prefill_start_line(verdict: &ChunkVerdict) -> String {
    match verdict {
        ChunkVerdict::Ready { m_sess } => format!("q36 cuda: prefill chunk M_sess {m_sess}"),
        ChunkVerdict::TiledLayout => "q36 cuda: prefill per token: tiled layout".to_owned(),
        ChunkVerdict::Kv8 => "q36 cuda: prefill per token: 8-bit cache".to_owned(),
        ChunkVerdict::Tiered => "q36 cuda: prefill per token: tiered mode".to_owned(),
        ChunkVerdict::Q6KWeight => "q36 cuda: prefill per token: Q6_K weight".to_owned(),
        ChunkVerdict::NoArena => "q36 cuda: prefill per token: no arena".to_owned(),
    }
}

fn kv8_prefill_start_line(verdict: &ChunkVerdict) -> String {
    let mode = if matches!(verdict, ChunkVerdict::Ready { .. }) {
        "chunked"
    } else {
        "per-token"
    };
    format!(
        "{} | KV8 mode={mode} | {CHUNK_PREFILL_VAR}=0 disables chunking",
        prefill_start_line(verdict)
    )
}

/// The pinned GGUF inside the artifact directory: the NAME is part of the identity, so the
/// join itself is the check.
pub(super) fn pinned_gguf(artifact: &Path) -> PathBuf {
    artifact.join(GGUF_NAME)
}

/// `true` when the environment asks for the full-file hash (`=1`); any other value is a typed
/// refusal, so a misspelled `yes` cannot promise an unchecked check.
pub(super) fn sha256_wanted(raw: Option<&str>) -> Result<bool, EngError> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(EngError::BadRequest(format!(
            "{VERIFY_SHA256_VAR} must be 0 or 1, found {other:?}"
        ))),
    }
}

/// The SHA-256 of a file, one 4 MiB pass.
pub(super) fn file_sha256(path: &Path) -> Result<String, EngError> {
    let read = |error: std::io::Error| {
        EngError::Checkpoint(format!("cuda checkpoint: {}: {error}", path.display()))
    };
    let mut file = std::fs::File::open(path).map_err(read)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let count = file.read(&mut buffer).map_err(read)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The full-file digest check, with the digest-and-elapsed log line printed only AFTER the
/// compare: a mismatch logs nothing here — its refusal names both digests.
pub(super) fn check_sha256(path: &Path, expected: &str) -> Result<(), EngError> {
    let began = Instant::now();
    let digest = file_sha256(path)?;
    if digest != expected {
        return Err(EngError::Checkpoint(format!(
            "cuda checkpoint: {} sha256 {digest} does not match the pinned file ({expected})",
            path.display()
        )));
    }
    eprintln!(
        "q36 cuda: gguf sha256 {digest} verified in {} s",
        began.elapsed().as_secs()
    );
    Ok(())
}

/// Checks the pinned identity of the GGUF: the NAME by construction (`pinned_gguf`), the exact
/// SIZE always (the metadata alone), the full-file SHA-256 only when `verify` asks for it. The
/// toggle value itself is resolved by the caller (`start()`), so a bad spelling is refused at
/// startup, before any device work; unset stays OFF with no error.
pub(super) fn check_gguf_identity(path: &Path, verify: bool) -> Result<(), EngError> {
    let metadata = std::fs::metadata(path).map_err(|_| {
        EngError::Checkpoint(format!(
            "cuda checkpoint: {} is missing (pinned: {GGUF_NAME}, {GGUF_BYTES} bytes)",
            path.display()
        ))
    })?;
    let size = metadata.len();
    if size != GGUF_BYTES {
        return Err(EngError::Checkpoint(format!(
            "cuda checkpoint: {} has {size} bytes; the pinned {GGUF_NAME} has {GGUF_BYTES}",
            path.display()
        )));
    }
    if verify {
        check_sha256(path, GGUF_SHA256)?;
    }
    Ok(())
}

/// The tiered plan the settings ask for, or `None` for the plain engine: tiered is never on without
/// the member of the config.
pub(super) fn tiered_plan_of(settings: &Q36Settings) -> Option<TieredPlan> {
    settings.tiered.map(|tiered| TieredPlan {
        timeout_ms: tiered.timeout_ms,
        preload: tiered.preload,
        ..TieredPlan::new(tiered.slots, tiered.threads)
    })
}

/// The refusal of a tiered start with graph capture, before any work (the settings refuse it too;
/// this is the start's own check): the tiered step waits for the host inside every layer.
pub(super) fn check_tiered_start(settings: &Q36Settings) -> Result<(), EngError> {
    if settings.tiered.is_some() && settings.capture {
        return Err(EngError::BadRequest(
            "engine config: tiered and capture exclude each other: the tiered step is eager only"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The checked KV mode of the parsed `kv_dtype` member and the layout the step runs: absent or
/// `f32` is `Fp16` — the shared "f32" names the CONTROL cache, and on the CUDA arm the control
/// cache is the fp16 cache — and `q8` is `Int8G32`, which exists as Native op lists only, so
/// the Tiled layout is a typed refusal here (the plan refuses it again, `check_layout_rows`).
pub(super) fn kv_mode_of(kv: KvDtype, layout: WeightLayout) -> Result<KvMode, EngError> {
    match (kv, layout) {
        (_, WeightLayout::Exl3) => Err(EngError::BadRequest(
            "engine config: EXL3 serving is not admitted".into(),
        )),
        (KvDtype::F32, _) => Ok(KvMode::Fp16),
        (KvDtype::Q8, WeightLayout::Native) => Ok(KvMode::Int8G32),
        (KvDtype::Q8, WeightLayout::Tiled) => Err(EngError::BadRequest(
            "engine config: kv_dtype q8 builds the Int8G32 KV mode, which takes the native \
             layout; Tiled is the fp16 control path"
                .to_owned(),
        )),
        // The BF16 KV form exists on the MLX arm only.
        (KvDtype::Bf16, _) => Err(EngError::BadRequest(
            "engine config: kv_dtype bf16 is an MLX-arm KV form; the CUDA arm takes f32 (the \
             fp16 control cache) or q8"
                .to_owned(),
        )),
    }
}

/// The KV mode of the config for the serve admission: resolved against the layout the load
/// PINS (`load_options` writes `WeightLayout::Native` outright), so the admission needs no
/// device work. `make` derives the mode again from the loaded weights' OWN layout, so the
/// step's mode always agrees with the admission's.
pub(super) fn serve_kv_mode(settings: &Q36Settings) -> Result<KvMode, EngError> {
    kv_mode_of(settings.kv, WeightLayout::Native)
}

/// The session options: one committed DeltaNet slot, no forked experts, the fixed KV
/// capacity of `context_limit` positions (the step's context limit IS this number), the KV
/// mode of the config, and the step layout. `make` calls it AFTER the load with the loaded
/// weights' own layout (`device.layout()?`), so the step always runs the layout the device
/// holds.
pub(super) fn step_options(
    context_limit: u32,
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Result<StepOptions, EngError> {
    Ok(StepOptions {
        kv_mode,
        context_tokens: context_tokens(context_limit, kv_mode)?,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    })
}

/// The KV capacity in positions of `context_limit`, with its KV bytes checked: below 2 is
/// refused. `start` validates the limit with this before any device work.
pub(super) fn context_tokens(context_limit: u32, kv_mode: KvMode) -> Result<usize, EngError> {
    if context_limit < 2 {
        return Err(EngError::BadRequest(format!(
            "engine config: context_limit must be at least 2, found {context_limit}"
        )));
    }
    let context_tokens = usize::try_from(context_limit)
        .map_err(|_| EngError::BadRequest("engine config: context_limit exceeds usize".into()))?;
    kv_cache_bytes(context_tokens, kv_mode)?;
    Ok(context_tokens)
}

/// The KV bytes of ONE position under `kv_mode`, straight from the pool's own layout
/// arithmetic (`KvPool::bytes_per_token`, 20,480 under Fp16 and 10,880 under Int8G32, pinned
/// by the start tests), never a constant: the admission must follow the KV mode the config
/// picked.
pub(super) fn kv_bytes_per_position(kv_mode: KvMode) -> Result<u64, EngError> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    u64::try_from(KvPool::bytes_per_token(&dims, kv_mode)?).map_err(|_| {
        EngError::MemoryPlan("cuda admission: the KV bytes per position exceed u64".into())
    })
}

/// The KV cache bytes of `context_tokens` positions of `kv_mode`, in checked arithmetic: the
/// number is the loader's admission term, so an impossible product is refused at planning
/// time. A u32 limit cannot overflow u64; the largest limits are refused by the admission
/// itself.
pub(super) fn kv_cache_bytes(context_tokens: usize, kv_mode: KvMode) -> Result<u64, EngError> {
    let positions = u64::try_from(context_tokens)
        .map_err(|_| EngError::MemoryPlan("cuda admission: KV positions exceed u64".into()))?;
    kv_bytes_per_position(kv_mode)?
        .checked_mul(positions)
        .ok_or_else(|| EngError::MemoryPlan("cuda admission: the KV bytes overflow u64".into()))
}

/// The load options of the engine: THE ONE PLACE the reserve terms and the load layout live
/// (the layout is written outright, never through a derived default). The layout is `Native`,
/// qualified on the 3090; the step follows the loaded weights, so this one field is the whole
/// switch. `work: 0` is right: `CudaProjector::new` allocates the matmul workspace BEFORE the
/// admission reads the free figure, so counting it in `work` too would double-count it.
pub(super) fn load_options(kv_cache: u64, fixed_admission: bool) -> LoadOptions<'static> {
    LoadOptions {
        reserve: Reserve {
            kv_cache,
            work: 0,
            headroom: prefill_headroom(fixed_admission) as u64,
        },
        drop_page_cache: None,
        layout: WeightLayout::Native,
    }
}

/// Loads and pins the tokenizer, builds the rules from the SAME loaded bundle BEFORE any device
/// work (a missing tokenizer file fails in seconds, not after a 20 GB upload), takes the
/// one-engine guard, and starts the engine thread. The state is `Loading` until `make` returns;
/// a load failure leaves `Error` with the message and releases the guard.
pub(crate) fn start(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    // The toggle is resolved HERE, before any work: a bad `Q36_VERIFY_GGUF_SHA256` spelling is
    // refused at startup, not on the engine thread after the projector exists; unset stays OFF
    // with no error (`sha256_wanted(None)` is `Ok(false)`, pinned in start_tests).
    let verify = sha256_wanted(std::env::var(VERIFY_SHA256_VAR).ok().as_deref())?;
    let chunked =
        chunk_prefill_wanted(settings.kv, std::env::var_os(CHUNK_PREFILL_VAR).as_deref())?;
    let chunk_rows = if chunked {
        chunk_prefill_rows(std::env::var_os(CHUNK_ROWS_VAR).as_deref())?
    } else {
        512
    };
    check_tiered_start(settings)?;
    let context_staging = context_staging_wanted(std::env::var_os(CONTEXT_STAGING_VAR).as_deref())?;
    let wide_stage = wide_stage_wanted(std::env::var_os(WIDE_STAGE_VAR).as_deref())?;
    let fixed_admission = fixed_admission_wanted(
        settings.kv,
        chunked,
        context_staging,
        std::env::var_os(FIXED_ADMISSION_VAR).as_deref(),
    )? && settings.tiered.is_none()
        && settings.kv_offload_host_bytes.is_none();
    if let Some(reason) = wide_stage
        .then(|| {
            wide_stage_inert_reason(
                fixed_admission,
                settings.kv,
                chunked,
                context_staging,
                settings.tiered.is_some(),
                settings.kv_offload_host_bytes.is_some(),
            )
        })
        .flatten()
    {
        // Default ON: the inert note is operator-facing only when the flag
        // was explicitly set (unset+inert is the normal 131K/FP16/tiered
        // boot, not worth a line).
        if std::env::var_os(WIDE_STAGE_VAR).is_some() {
            eprintln!("q36 cuda: wide CPF stage flag has no effect: {reason}");
        }
    }
    let modes = RuntimeModes {
        chunked,
        chunk_rows,
        context_staging,
        fixed_admission,
        wide_stage,
        prefix: PrefixMemory::from_flags(
            settings.prefix_reuse,
            std::env::var_os("Q36_CUDA_PINNED_PREFIX").as_deref(),
            std::env::var_os("Q36_CUDA_STATE_ONLY_PREFIX").as_deref(),
            settings.tiered.is_none(),
        ),
    };
    let device_first = device_first_wanted(std::env::var_os(DEVICE_FIRST_VAR).as_deref())?;
    let plan = if device_first {
        StartupPlan::new_device_first(settings, verify, modes)?
    } else {
        StartupPlan::new(settings, verify, modes)?
    };
    for key in &settings.ignored_mlx_keys {
        eprintln!("{}", ignored_mlx_line(key));
    }
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &tokenizer_pin()?)?;
    let eos = bundle.eos_token_ids().to_vec();
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".into()))?;
    let head_rows = head_rows_of()?;
    let rules = QwenRules::new(eos, sampler_vocab, head_rows)?;
    // The CUDA arm refuses the prefill budget, so its worker piece is the no-budget piece.
    let piece = usize::try_from(NO_BUDGET_PIECE).map_err(|_| {
        EngError::BadRequest("engine config: the worker piece exceeds usize".into())
    })?;
    let guard = BuildGuard::acquire()?;
    let owned = settings.clone();
    let engine = spawn_engine_noted(piece, ready_note, move || make(owned, rules, plan, guard))?;
    Ok((engine, bundle))
}

/// The engine-thread half: everything CUDA is built here, in the gate's own order — projector,
/// memory figure, GGUF identity, open, load with admission, free-after-load floor, the step
/// options from the loaded layout, step resources, then the leaks, the step and the capture
/// setting. The SHA-256 toggle arrives already resolved (`start()` validated it), so this
/// thread never reads the environment.
fn make(
    settings: Q36Settings,
    rules: QwenRules,
    plan: StartupPlan,
    mut guard: BuildGuard,
) -> Result<(QwenResidentStep<'static>, DecodeSession), EngError> {
    let kv_cache = plan.reserve;
    let verify = plan.verify;
    let fixed_admission = plan.modes.fixed_admission;
    let projector = CudaProjector::new()?;
    let before = projector.memory_info()?;
    let gguf = pinned_gguf(&settings.artifact);
    check_gguf_identity(&gguf, verify)?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let tiered = tiered_plan_of(&settings);
    // A tiered load leaves the routed expert stacks in the file: the banks hold the hot ones.
    let device = match tiered {
        None => {
            projector
                .load_model(&model, load_options(kv_cache, fixed_admission))?
                .device
        }
        Some(_) => {
            let names = model.plan().device_names();
            let names = names_without_experts(&names);
            projector.load(
                &model.weights,
                &names,
                load_options(kv_cache, fixed_admission),
            )?
        }
    };
    let after = projector.memory_info()?;
    if after.free < MIN_FREE_AFTER_LOAD {
        return Err(EngError::MemoryPlan(format!(
            "cuda admission: {} bytes free after the load is below the {}-byte floor",
            after.free, MIN_FREE_AFTER_LOAD
        )));
    }
    let plan = plan.after_load(&settings, after.free)?;
    let StartupPlan {
        reserve: kv_cache,
        paged,
        device_first_native,
        verify: _,
        modes,
    } = plan;
    let device_first = device_first_native.is_some();
    if device_first && paged.is_none() {
        require_device_first_room(after.free, kv_cache)?;
    }
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    // The step layout is the loaded weights' own, never a constant or the load options: the
    // step's matvecs read the blocks the device holds (a mixed load is refused by layout()).
    // The KV mode is derived from the SAME parsed form against that layout (`kv_mode_of`),
    // so the step's mode always agrees with the admission's.
    let layout = device.layout()?;
    let kv_mode = kv_mode_of(settings.kv, layout)?;
    let mut options = step_options(settings.context_limit, layout, kv_mode)?;
    options.tiered = tiered;
    let res = match paged {
        Some(plan) => StepResources::allocate_paged(projector.context(), plan)?,
        None => StepResources::new(projector.context(), dims, &inventory, &options)?,
    };
    // LEAK: the three owners are leaked for the process lifetime, giving
    // `QwenResidentStep<'static>`; an owning `Arc` variant would rewrite the qualified step
    // (>= 1,000 lines). `guard.commit()` right below pins the one-engine flag, so a retry after
    // ANY failure from here on — a `QwenResidentStep::new` or `set_capture_config` error leaves
    // the ~20 GB on the device — is refused with the documented text. A process whose engine
    // ended in Error or NativeFailed is restarted, not rebuilt.
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(device));
    let res: &'static StepResources = Box::leak(Box::new(res));
    guard.commit();
    let mut step = QwenResidentStep::new(projector.context(), res, weights, &options, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: settings.capture,
        // The section-timing marks stay off on the serve path: the timer is a measurement harness,
        // not a serving feature.
        section_marks: false,
    })?;
    // Q8 chunking defaults on at 1024 rows; overrides use the supported row ladder.
    // Other sessions keep DEFAULT_CUDA_PREFILL_CHUNK; MLX `prefill_chunk` is ignored.
    let kv8_chunked = modes.chunked && kv_mode == KvMode::Int8G32;
    step.set_kv8_chunked_prefill(kv8_chunked)?;
    step.set_m_sess_cap(if kv8_chunked {
        modes.chunk_rows
    } else {
        DEFAULT_CUDA_PREFILL_CHUNK
    });

    // The prefill arena: ONE admission after the load and the step build, from the post-load
    // snapshot less the selected headroom. Fixed mode also checks live free VRAM and refuses an
    // undersized plan; legacy mode may keep the per-token arm.
    if let Some(host_limit) = settings.kv_offload_host_bytes {
        if device_first {
            if let Some(rows) = res
                .paged
                .as_ref()
                .and_then(|paged| paged.state.cache_rows())
            {
                eprintln!(
                    "q36 cuda: device-first KV, cached rows {rows}, host overflow limit {host_limit} B, physical state/arena/attention reserve {kv_cache} B"
                );
            } else {
                eprintln!(
                    "q36 cuda: device-first KV, fully resident; native state/arena reserve {kv_cache} B, host payloads absent"
                );
            }
        } else {
            eprintln!(
                "q36 cuda: host-paged KV, host limit {host_limit} B, physical state/arena/attention reserve {kv_cache} B; prefix reuse, capture and chunked prefill disabled"
            );
        }
    } else {
        let mut extras = Vec::new();
        if modes.prefix != PrefixMemory::Disabled {
            step.enable_pinned_prefix()?;
        }
        if modes.prefix == PrefixMemory::DeviceTip && step.pinned_prefix_enabled() {
            extras.push(Extra {
                name: "device-prefix-delta",
                fixed: step.enable_device_tip()?,
                bytes_per_token: 0,
            });
        }
        let device_free = after
            .free
            .saturating_sub(prefill_headroom(modes.fixed_admission));
        let verdict = step.admit_prefill(
            PrefillBudget {
                device_free,
                extras,
            },
            modes.context_staging,
            modes.fixed_admission,
            modes.wide_stage,
        )?;
        let line = if kv_mode == KvMode::Int8G32 {
            kv8_prefill_start_line(&verdict)
        } else {
            prefill_start_line(&verdict)
        };
        eprintln!("{line}");
    }
    if let Some(plan) = tiered {
        // The banks come last: they take what the weights, the KV cache and the step state left.
        attach_tiered_parts(
            &mut step,
            projector.context(),
            ((model.weights.header(), &gguf), &res.dims),
            &plan,
        )?;
        eprintln!(
            "q36 cuda: tiered, {} slots per layer, {} cpu threads, timeout {} ms",
            plan.slots_per_layer, plan.threads, plan.timeout_ms
        );
    }
    eprintln!(
        "q36 cuda: weights resident ({} B), free after load {} B, context_limit {}, kv {}, \
         capture {}",
        before.free.saturating_sub(after.free),
        after.free,
        options.context_tokens,
        match options.kv_mode {
            KvMode::Fp16 => "fp16",
            KvMode::Int8G32 => "int8g32",
        },
        if settings.capture { "on" } else { "off" }
    );
    Ok((step, DecodeSession::default()))
}

#[cfg(test)]
mod chunk_prefill_flag_tests {
    use super::{
        ChunkVerdict, KvDtype, PrefixMemory, chunk_prefill_rows, chunk_prefill_wanted,
        context_staging_wanted, kv8_prefill_start_line,
    };
    use std::ffi::OsStr;

    #[test]
    fn wide_stage_defaults_on_and_zero_rolls_it_back() {
        use super::wide_stage_wanted;
        // The flip: unset means ON; =0 is the documented rollback; the parse
        // rules match every prefill flag (only 0 or 1, else a typed refusal).
        assert!(wide_stage_wanted(None).unwrap());
        assert!(!wide_stage_wanted(Some(OsStr::new("0"))).unwrap());
        assert!(wide_stage_wanted(Some(OsStr::new("1"))).unwrap());
        assert!(wide_stage_wanted(Some(OsStr::new("x"))).is_err());
    }

    #[test]
    fn wide_stage_inert_reason_names_every_silent_mode() {
        use super::wide_stage_inert_reason as reason;
        // Fixed admission holds: the flag is live, no reason.
        assert_eq!(reason(true, KvDtype::Q8, true, true, false, false), None);
        // Each mode that drops fixed admission names itself, most specific first.
        assert_eq!(
            reason(false, KvDtype::Q8, true, true, false, true),
            Some("host KV offload")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, true, true, true, false),
            Some("tiered mode")
        );
        assert_eq!(
            reason(false, KvDtype::F32, true, true, false, false),
            Some("non-Q8 KV dtype")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, false, true, false, false),
            Some("chunked prefill off (Q36_CUDA_CHUNK_PREFILL=0)")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, true, false, false, false),
            Some("context staging off (Q36_CUDA_KV8_STAGE_CONTEXT=0)")
        );
        assert_eq!(
            reason(false, KvDtype::Q8, true, true, false, false),
            Some("fixed admission off (Q36_CUDA_KV8_FIXED_ADMISSION=0)")
        );
    }

    #[test]
    fn prefix_memory_never_enables_prefix_reuse() {
        for pinned in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
            for device in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
                assert!(
                    PrefixMemory::from_flags(false, pinned, device, true) == PrefixMemory::Disabled
                );
            }
        }
    }

    #[test]
    fn prefix_memory_defaults_to_device_tip_when_reuse_is_on() {
        assert!(PrefixMemory::from_flags(true, None, None, true) == PrefixMemory::DeviceTip);
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("1")), None, true)
                == PrefixMemory::DeviceTip
        );
    }

    #[test]
    fn pinned_zero_restores_pageable_even_when_device_is_requested() {
        for device in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
            assert!(
                PrefixMemory::from_flags(true, Some(OsStr::new("0")), device, true)
                    == PrefixMemory::Disabled
            );
        }
    }

    #[test]
    fn state_only_zero_keeps_the_pinned_host_path() {
        assert!(
            PrefixMemory::from_flags(true, None, Some(OsStr::new("0")), true)
                == PrefixMemory::Pinned
        );
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("1")), Some(OsStr::new("0")), true)
                == PrefixMemory::Pinned
        );
    }

    #[test]
    fn invalid_prefix_flag_values_do_not_enable_the_memory_mode() {
        for value in ["", "true", "yes", "2", " 1"] {
            assert!(
                PrefixMemory::from_flags(true, Some(OsStr::new(value)), None, true)
                    == PrefixMemory::Disabled
            );
            assert!(
                PrefixMemory::from_flags(
                    true,
                    Some(OsStr::new("1")),
                    Some(OsStr::new(value)),
                    true
                ) == PrefixMemory::Pinned
            );
        }
    }

    #[test]
    fn unsupported_tiered_backend_ignores_implicit_prefix_memory() {
        assert!(PrefixMemory::from_flags(true, None, None, false) == PrefixMemory::Disabled);
        assert!(
            PrefixMemory::from_flags(true, None, Some(OsStr::new("0")), false)
                == PrefixMemory::Disabled
        );
    }

    #[test]
    fn explicit_unsupported_requests_still_reach_mode_validation() {
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("1")), None, false)
                == PrefixMemory::Pinned
        );
        assert!(
            PrefixMemory::from_flags(true, None, Some(OsStr::new("1")), false)
                == PrefixMemory::DeviceTip
        );
        assert!(
            PrefixMemory::from_flags(true, Some(OsStr::new("0")), Some(OsStr::new("1")), false)
                == PrefixMemory::Disabled
        );
    }

    #[cfg(unix)]
    #[test]
    fn invalid_unicode_cannot_enable_prefix_memory() {
        use std::os::unix::ffi::OsStrExt;
        let invalid = OsStr::from_bytes(&[0xff]);
        assert!(
            PrefixMemory::from_flags(true, Some(invalid), None, true) == PrefixMemory::Disabled
        );
        assert!(PrefixMemory::from_flags(true, None, Some(invalid), true) == PrefixMemory::Pinned);
    }

    #[test]
    fn context_staging_defaults_on_and_zero_restores_legacy() {
        assert!(context_staging_wanted(None).unwrap());
        assert!(!context_staging_wanted(Some(OsStr::new("0"))).unwrap());
        assert!(context_staging_wanted(Some(OsStr::new("1"))).unwrap());
    }

    #[test]
    fn fixed_admission_defaults_on_and_all_kill_switches_win() {
        assert!(
            !super::fixed_admission_wanted(KvDtype::Q8, true, true, Some(OsStr::new("0"))).unwrap()
        );
        for (kv, chunked, context, wanted) in [
            (KvDtype::Q8, true, true, true),
            (KvDtype::Q8, false, true, false),
            (KvDtype::Q8, true, false, false),
            (KvDtype::F32, true, true, false),
        ] {
            for flag in [None, Some(OsStr::new("1"))] {
                assert_eq!(
                    super::fixed_admission_wanted(kv, chunked, context, flag).unwrap(),
                    wanted
                );
            }
        }
        assert!(
            super::fixed_admission_wanted(KvDtype::Q8, false, false, Some(OsStr::new("invalid")),)
                .is_err()
        );
        assert_eq!(super::prefill_headroom(false), 1usize << 30);
        assert_eq!(super::prefill_headroom(true), 256usize << 20);
        assert_eq!(super::MIN_FREE_AFTER_LOAD, 1usize << 30);
    }

    #[test]
    fn q8_chunking_defaults_on_and_accepts_only_zero_or_one() {
        assert!(chunk_prefill_wanted(KvDtype::Q8, None).unwrap());
        assert!(!chunk_prefill_wanted(KvDtype::Q8, Some(OsStr::new("0"))).unwrap());
        assert!(chunk_prefill_wanted(KvDtype::Q8, Some(OsStr::new("1"))).unwrap());
        for kv in [KvDtype::F32, KvDtype::Q8] {
            for invalid in ["", "true", "yes", "2", " 1"] {
                assert!(chunk_prefill_wanted(kv, Some(OsStr::new(invalid))).is_err());
            }
        }
    }

    #[test]
    fn fp16_default_is_unchanged() {
        assert!(!chunk_prefill_wanted(KvDtype::F32, None).unwrap());
        assert!(!chunk_prefill_wanted(KvDtype::F32, Some(OsStr::new("0"))).unwrap());
        assert!(chunk_prefill_wanted(KvDtype::F32, Some(OsStr::new("1"))).unwrap());
    }

    #[test]
    fn kv8_start_line_names_actual_mode_and_kill_switch() {
        let on = kv8_prefill_start_line(&ChunkVerdict::Ready { m_sess: 1024 });
        assert!(on.contains("prefill chunk M_sess 1024 | KV8 mode=chunked"));
        assert!(on.contains("Q36_CUDA_CHUNK_PREFILL=0 disables chunking"));
        for verdict in [
            ChunkVerdict::Kv8,
            ChunkVerdict::NoArena,
            ChunkVerdict::Tiered,
        ] {
            let off = kv8_prefill_start_line(&verdict);
            assert!(off.contains("KV8 mode=per-token"));
            assert!(off.contains("Q36_CUDA_CHUNK_PREFILL=0 disables chunking"));
        }
    }

    #[test]
    fn requested_bulk_sizes_are_explicit_and_default_to_1024() {
        assert_eq!(chunk_prefill_rows(None).unwrap(), 1024);
        for rows in ["16", "128", "512", "1024"] {
            assert_eq!(
                chunk_prefill_rows(Some(OsStr::new(rows))).unwrap(),
                rows.parse::<u32>().unwrap()
            );
        }
        for bad in ["", "0", "1", "64", "1025", "2048", " 512"] {
            assert!(chunk_prefill_rows(Some(OsStr::new(bad))).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn invalid_unicode_cannot_silently_enable_chunking() {
        use std::os::unix::ffi::OsStrExt;
        for kv in [KvDtype::F32, KvDtype::Q8] {
            assert!(chunk_prefill_wanted(kv, Some(OsStr::from_bytes(&[0xff]))).is_err());
        }
    }

    include!("prefill_ttft_tests.rs");
}

#[cfg(test)]
mod paged_start_tests {
    use super::*;
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options};

    fn settings(body: &str) -> Q36Settings {
        let options: Q36Options = serde_json::from_str(body).unwrap();
        Q36Settings::new(
            Path::new("/not-opened"),
            Path::new("/not-opened"),
            Some(&options),
            &Q36Env::default(),
            Backend::Cuda,
        )
        .unwrap()
    }

    fn modes(chunked: bool) -> RuntimeModes {
        RuntimeModes {
            chunked,
            chunk_rows: 512,
            context_staging: false,
            fixed_admission: false,
            wide_stage: false,
            prefix: PrefixMemory::Disabled,
        }
    }

    #[test]
    fn absent_offload_preserves_the_resident_loader_reserve() {
        let settings = settings("{}");
        let plan = StartupPlan::new(&settings, false, modes(false)).unwrap();
        assert!(plan.paged.is_none());
        assert_eq!(
            plan.reserve,
            kv_cache_bytes(
                settings.context_limit as usize,
                serve_kv_mode(&settings).unwrap()
            )
            .unwrap(),
        );
    }

    #[test]
    fn paged_startup_retains_the_exact_physical_resource_plan() {
        let settings = settings(
            r#"{"kv_dtype":"q8","context_limit":131072,"kv_offload_host_bytes":2000000000}"#,
        );
        let plan = StartupPlan::new(&settings, false, modes(false)).unwrap();
        assert_eq!(
            plan.reserve,
            u64::try_from(plan.paged.as_ref().unwrap().device_bytes().unwrap()).unwrap()
        );
        assert!(plan.reserve < kv_cache_bytes(131072, KvMode::Int8G32).unwrap());
    }

    #[test]
    fn device_first_start_uses_the_complete_resident_reserve_without_host_payloads() {
        let configured =
            settings(r#"{"kv_dtype":"q8","context_limit":33280,"kv_offload_host_bytes":1}"#);
        let plan = StartupPlan::new_device_first(&configured, false, modes(false)).unwrap();
        assert!(plan.paged.is_none());
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM).unwrap();
        let dims = StepDims::new(&shape, &inventory).unwrap();
        let options = step_options(33280, WeightLayout::Native, KvMode::Int8G32).unwrap();
        let state = super::super::state_layout::StateLayout::new(&dims, &options).unwrap();
        let scratch =
            super::super::step_plan::KernelScratch::for_dims(&dims, 1, KvMode::Int8G32).unwrap();
        let step = super::super::step_plan::StepPlan::new(&dims, 1, &options, &scratch).unwrap();
        let native = state.bytes().unwrap() + step.arena_bytes();
        assert!(
            plan.reserve < native as u64,
            "pre-load reserve must exclude the full KV pool"
        );
        let plan = plan
            .after_load(&configured, MIN_FREE_AFTER_LOAD + native)
            .unwrap();
        assert!(plan.paged.is_none());
        assert_eq!(plan.reserve, native as u64);

        let absent = settings("{}");
        let plan = StartupPlan::new_device_first(&absent, false, modes(false)).unwrap();
        assert_eq!(
            plan.reserve,
            StartupPlan::new(&absent, false, modes(false))
                .unwrap()
                .reserve
        );
    }

    #[test]
    fn tiny_native_fit_is_not_rejected_by_rounded_page_reservation() {
        let configured =
            settings(r#"{"kv_dtype":"q8","context_limit":2,"kv_offload_host_bytes":1}"#);
        let plan = StartupPlan::new_device_first(&configured, false, modes(false)).unwrap();
        let native = plan.device_first_native.unwrap();
        assert!(plan.reserve as usize <= native);
        let plan = plan
            .after_load(&configured, MIN_FREE_AFTER_LOAD + native)
            .unwrap();
        assert!(plan.paged.is_none());
        assert_eq!(plan.reserve as usize, native);
    }

    #[test]
    fn device_first_selects_the_measured_cache_budget_after_loading_weights() {
        let configured =
            settings(r#"{"kv_dtype":"q8","context_limit":1280,"kv_offload_host_bytes":33554432}"#);
        let dims = super::super::common_tests::dims();
        let paging = super::super::paging_plan::PagingPlan::new(&dims, 1280).unwrap();
        let group = paging.group.total * paging.full_layers;
        let plan = StartupPlan::new_device_first(&configured, false, modes(false)).unwrap();
        let fixed = plan.reserve as usize;
        let plan = plan
            .after_load(&configured, MIN_FREE_AFTER_LOAD + fixed + 2 * group)
            .unwrap();
        assert_eq!(
            plan.paged.as_ref().unwrap().cache_rows(),
            Some(2 * paging.group_cells)
        );
        assert_eq!(plan.reserve as usize, fixed + 2 * group);

        let plan = StartupPlan::new_device_first(&configured, false, modes(false)).unwrap();
        assert!(
            plan.after_load(&configured, MIN_FREE_AFTER_LOAD + fixed + group - 1)
                .is_err()
        );
        let mut insufficient_host = configured;
        insufficient_host.kv_offload_host_bytes = Some(1);
        let plan = StartupPlan::new_device_first(&insufficient_host, false, modes(false)).unwrap();
        assert!(
            plan.after_load(&insufficient_host, MIN_FREE_AFTER_LOAD + fixed + 2 * group)
                .is_err()
        );
    }

    #[test]
    fn device_first_flag_is_default_off_and_strict() {
        use std::ffi::OsStr;
        assert!(!device_first_wanted(None).unwrap());
        assert!(!device_first_wanted(Some(OsStr::new("0"))).unwrap());
        assert!(device_first_wanted(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "yes", "2", " 1"] {
            assert!(device_first_wanted(Some(OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn device_first_resident_admission_retains_the_complete_headroom() {
        assert!(require_device_first_room(MIN_FREE_AFTER_LOAD + 256, 256).is_ok());
        assert!(require_device_first_room(MIN_FREE_AFTER_LOAD + 255, 256).is_err());
        assert!(require_device_first_room(usize::MAX, u64::MAX).is_err());
    }

    #[test]
    fn device_first_retains_the_capacity_mode_refusals() {
        let base = settings(r#"{"kv_dtype":"q8","context_limit":320,"kv_offload_host_bytes":1}"#);
        assert!(StartupPlan::new_device_first(&base, false, modes(true)).is_err());
        let mut bad = base.clone();
        bad.capture = true;
        assert!(StartupPlan::new_device_first(&bad, false, modes(false)).is_err());
        let mut bad = base.clone();
        bad.prefix_reuse = true;
        assert!(StartupPlan::new_device_first(&bad, false, modes(false)).is_err());
        let mut bad = base.clone();
        bad.kv_offload_host_bytes = Some(0);
        assert!(StartupPlan::new_device_first(&bad, false, modes(false)).is_err());
        let mut bad = base;
        bad.kv = KvDtype::F32;
        assert!(StartupPlan::new_device_first(&bad, false, modes(false)).is_err());
    }

    #[test]
    fn unsupported_paged_starts_fail_without_a_device_or_checkpoint() {
        let settings =
            settings(r#"{"kv_dtype":"q8","context_limit":320,"kv_offload_host_bytes":8388608}"#);
        assert!(StartupPlan::new(&settings, false, modes(true)).is_err());
        let mut bad = settings.clone();
        bad.capture = true;
        assert!(StartupPlan::new(&bad, false, modes(false)).is_err());
        let mut bad = settings.clone();
        bad.prefix_reuse = true;
        assert!(StartupPlan::new(&bad, false, modes(false)).is_err());
        let mut bad = settings;
        bad.kv_offload_host_bytes = Some(1);
        assert!(StartupPlan::new(&bad, false, modes(false)).is_err());
    }
}
