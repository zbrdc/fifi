//! Starts the Q36 CUDA engine of one `enginesrv` config entry (CUDA-SERVE-1): the tokenizer
//! bundle and the rules on the calling thread, then the engine thread builds everything CUDA
//! inside `make` (no CUDA value crosses a thread boundary, decisions 0023/0024).
//!
//! Lifetime (CUDA-SERVE-1 2.1, git-db's design condition): `spawn_engine` needs an executor
//! without a lifetime and `QwenResidentStep` borrows its owners, so the projector, the weights
//! and the step resources are `Box::leak`ed once per engine, giving `QwenResidentStep<'static>`
//! (an owning `Arc` variant would rewrite the qualified step, >= 1,000 lines). ONE CUDA engine
//! is built per process: `BuildGuard` pins the flag before `spawn_engine`, commits it right
//! after the leaks, and every retry in the same process is refused. No code copied.

use super::super::settings::{NO_BUDGET_PIECE, Q36Settings, tokenizer_pin};
use super::{
    dims::{KvMode, StepDims, StepOptions},
    pinned::{MIN_FREE_AFTER_LOAD, ROTARY_DIM, head_rows_of},
    resources::StepResources,
    rules::QwenRules,
    step::{CaptureConfig, QwenResidentStep},
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
    sched::worker::{Engine, spawn_engine},
    tokenizer::TokenizerBundle,
};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// The pinned checkpoint of the CUDA arm: the file NAME inside `model_dir` and its exact SIZE,
/// both matching `qualification/cuda-rd/a5-gated-steps.json`. The full-file SHA-256 beside them
/// is checked only when `Q36_VERIFY_GGUF_SHA256` is `1` (KUBDAN-MSG-69 condition 3): hashing
/// 20 GB costs about 15-40 s per start (ESTIMATE) and the qualification script checks it.
pub(super) const GGUF_NAME: &str = "Qwen3.6-35B-A3B-Q4_K_M.gguf";
pub(super) const GGUF_BYTES: u64 = 20_419_565_568;
const GGUF_SHA256: &str = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7";
/// The full-file SHA-256 is checked only when this variable is EXACTLY `1`: unset or `0`
/// leaves the check off, and every OTHER spelling — `true`, `yes`, an empty value — is a
/// typed refusal (raised in `make`, after the CUDA context exists), so a misspelled promise
/// cannot silently skip the check. A value Rust cannot read as UTF-8 reads as unset (std's
/// `env::var`), so it leaves the check off (KUBDAN-MSG-72 NIT).
const VERIFY_SHA256_VAR: &str = "Q36_VERIFY_GGUF_SHA256";
/// KV bytes per position of the pinned fp16 cache (`cuda/tests.rs` pins 20,480 B/position;
/// `start_tests` ties the figure to `KvPool::bytes_per_token`, KUBDAN-MSG-72 item 6).
pub(super) const KV_BYTES_PER_POSITION: u64 = 20_480;
/// The loader's admission headroom beside weights and KV (0040 A2 keeps the 1 GiB floor for
/// the free-after-load check; `work: 0` is right — see `load_options`).
const HEADROOM: u64 = 1 << 30;

/// The process-wide one-engine flag: `true` from the first leak of a build onwards.
pub(super) static CUDA_ENGINE_BUILT: AtomicBool = AtomicBool::new(false);

/// The refusal of a second build in one process (the packet's documented text, 2.1).
pub(super) fn already_built() -> EngError {
    EngError::BadRequest("engine config: the CUDA Q36 backend builds one engine per process: its weights are leaked for the process lifetime and an in-process restart would leak 20 GB again; configure a single qwen3_6_35b_a3b entry and restart the process to reload".to_string())
}

/// The one-engine guard (condition 2): `acquire` takes the flag before `spawn_engine`,
/// `commit` pins it right after the first leak, and the `Drop` releases it only for a failure
/// BEFORE the first leak, when nothing is on the device and a retry may build.
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

/// The start line of one MLX-only key the CUDA arm ignores (the settings record the names in
/// the order of `Q36Options`).
pub(crate) fn ignored_mlx_line(key: &str) -> &'static str {
    match key {
        "cache_limit_bytes" => "q36 cuda: cache_limit_bytes is an MLX allocator key and is ignored",
        "session_cap_bytes" => "q36 cuda: session_cap_bytes is an MLX session key and is ignored",
        "prefill_chunk" => "q36 cuda: prefill_chunk is an MLX chunked-prefill key and is ignored",
        _ => "q36 cuda: an unknown key of the object is ignored",
    }
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

/// The full-file digest check, with the digest-and-elapsed log line of condition 3, printed
/// only AFTER the compare (KUBDAN-MSG-72 NIT): a mismatch logs nothing here — its refusal
/// names both digests.
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
/// SIZE always (the metadata alone), the full-file SHA-256 only when `verify` asks for it.
/// The toggle value itself is resolved by the caller (`start()`, SERVE-1-F F4), so a bad
/// spelling is refused at startup, before any device work; unset stays OFF with no error.
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

/// The session options: fp16 KV, one committed DeltaNet slot, no forked experts, the fixed KV
/// capacity of `context_limit` positions (the step's context limit IS this number), and the
/// step layout. `make` calls it AFTER the load with the loaded weights' own layout
/// (`loaded.device.layout()?`, KUBDAN-MSG-72 item 1), so the step always runs the layout the
/// device holds.
pub(super) fn step_options(
    context_limit: u32,
    layout: WeightLayout,
) -> Result<StepOptions, EngError> {
    Ok(StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: context_tokens(context_limit)?,
        delta_slots: 1,
        forked_experts: false,
        layout,
    })
}

/// The KV capacity in positions of `context_limit`, with its KV bytes checked: below 2 is
/// refused. `start` validates the limit with this before any device work.
pub(super) fn context_tokens(context_limit: u32) -> Result<usize, EngError> {
    if context_limit < 2 {
        return Err(EngError::BadRequest(format!(
            "engine config: context_limit must be at least 2, found {context_limit}"
        )));
    }
    let context_tokens = usize::try_from(context_limit)
        .map_err(|_| EngError::BadRequest("engine config: context_limit exceeds usize".into()))?;
    kv_cache_bytes(context_tokens)?;
    Ok(context_tokens)
}

/// The KV cache bytes of `context_tokens` fp16 positions, in checked arithmetic: the number is
/// the loader's admission term, so an impossible product is refused at planning time. A u32
/// limit cannot overflow u64; the largest limits are refused by the admission itself.
pub(super) fn kv_cache_bytes(context_tokens: usize) -> Result<u64, EngError> {
    let positions = u64::try_from(context_tokens)
        .map_err(|_| EngError::MemoryPlan("cuda admission: KV positions exceed u64".into()))?;
    KV_BYTES_PER_POSITION
        .checked_mul(positions)
        .ok_or_else(|| EngError::MemoryPlan("cuda admission: the KV bytes overflow u64".into()))
}

/// The load options of the engine: THE ONE PLACE the reserve terms and the load layout live
/// (MSG-70 condition 8: the layout is written outright, never through a derived default). The
/// layout is `Native` since the S3c qualification of MATVEC-PAR S3b on the 3090 (lane C's
/// ruling); the step follows the loaded weights, so this one field is the whole switch. `work: 0` is right: `CudaProjector::new`
/// allocates the matmul workspace BEFORE the admission reads the free figure, so counting it
/// in `work` too would double-count it.
pub(super) fn load_options(kv_cache: u64) -> LoadOptions<'static> {
    LoadOptions {
        reserve: Reserve {
            kv_cache,
            work: 0,
            headroom: HEADROOM,
        },
        drop_page_cache: None,
        layout: WeightLayout::Native,
    }
}

/// Loads and pins the tokenizer, builds the rules from the SAME loaded bundle BEFORE any device
/// work (a missing tokenizer file fails in seconds, not after a 20 GB upload), takes the
/// one-engine guard, and starts the engine thread. The state is `Loading` until `make` returns;
/// a load failure leaves `Error` with the message and releases the guard.
pub(crate) fn start(settings: &Q36Settings) -> Result<(Engine, TokenizerBundle), EngError> {
    // The toggle is resolved HERE, before any work (SERVE-1-F F4): a bad
    // `Q36_VERIFY_GGUF_SHA256` spelling is refused at startup, not on the engine thread after
    // the projector exists; unset stays OFF with no error (`sha256_wanted(None)` is
    // `Ok(false)`, pinned in start_tests).
    let verify = sha256_wanted(std::env::var(VERIFY_SHA256_VAR).ok().as_deref())?;
    for key in &settings.ignored_mlx_keys {
        eprintln!("{}", ignored_mlx_line(key));
    }
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &tokenizer_pin()?)?;
    let eos = bundle.eos_token_ids().to_vec();
    let sampler_vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".into()))?;
    let head_rows = head_rows_of()?;
    let rules = QwenRules::new(eos, sampler_vocab, head_rows)?;
    let kv_cache = kv_cache_bytes(context_tokens(settings.context_limit)?)?;
    // The CUDA arm refuses the prefill budget, so its worker piece is the no-budget piece.
    let piece = usize::try_from(NO_BUDGET_PIECE).map_err(|_| {
        EngError::BadRequest("engine config: the worker piece exceeds usize".into())
    })?;
    let guard = BuildGuard::acquire()?;
    let owned = settings.clone();
    let engine = spawn_engine(piece, move || make(owned, rules, kv_cache, verify, guard))?;
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
    kv_cache: u64,
    verify: bool,
    mut guard: BuildGuard,
) -> Result<(QwenResidentStep<'static>, DecodeSession), EngError> {
    let projector = CudaProjector::new()?;
    let before = projector.memory_info()?;
    let gguf = pinned_gguf(&settings.artifact);
    check_gguf_identity(&gguf, verify)?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let loaded = projector.load_model(&model, load_options(kv_cache))?;
    let after = projector.memory_info()?;
    if after.free < MIN_FREE_AFTER_LOAD {
        return Err(EngError::MemoryPlan(format!(
            "cuda admission: {} bytes free after the load is below the {}-byte floor",
            after.free, MIN_FREE_AFTER_LOAD
        )));
    }
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    // The step layout is the loaded weights' own, never a constant or the load options: the
    // step's matvecs read the blocks the device holds (a mixed load is refused by layout()).
    let options = step_options(settings.context_limit, loaded.device.layout()?)?;
    let res = StepResources::new(projector.context(), dims, &inventory, &options)?;
    // LEAK (CUDA-SERVE-1 2.1, condition 2): the three owners are leaked for the process
    // lifetime, giving `QwenResidentStep<'static>`; an owning `Arc` variant would rewrite the
    // qualified step (>= 1,000 lines). `guard.commit()` right below pins the one-engine flag,
    // so a retry after ANY failure from here on — a `QwenResidentStep::new` or
    // `set_capture_config` error leaves the ~20 GB on the device — is refused with the
    // documented text. A process whose engine ended in Error or NativeFailed is restarted,
    // not rebuilt.
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(loaded.device));
    let res: &'static StepResources = Box::leak(Box::new(res));
    guard.commit();
    let mut step = QwenResidentStep::new(projector.context(), res, weights, &options, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: settings.capture,
    })?;
    eprintln!(
        "q36 cuda: weights resident ({} B), free after load {} B, context_limit {}, capture {}",
        before.free.saturating_sub(after.free),
        after.free,
        options.context_tokens,
        if settings.capture { "on" } else { "off" }
    );
    Ok((step, DecodeSession::default()))
}
