//! Starts the Flash-Next CUDA engine of one `enginesrv` config entry: the tokenizer bundle and the
//! rules on the calling thread, then the engine thread builds everything CUDA inside `make` (no
//! CUDA value crosses a thread boundary). It follows `qwen36/cuda/start.rs`.
//!
//! Load order inside `make`, each stage announced to the engine state:
//! 1. the CUDA context, and the card's free memory before any weight is resident;
//! 2. the pack, opened through one local-storage guard (`FlashPackFiles`);
//! 3. the memory plan for the configured context (`FlashLoadPlan::at_context`): the expert slots
//!    per layer that fit next to the trunk and the K/V, or a refusal that names the longest context;
//! 4. the trunk, uploaded in the order of the walk (`trunk_weights`);
//! 5. the expert bank: device slots per layer filled with the first experts of each layer, and the
//!    rest pinned in host RAM (`DeviceBank::build`), after the host plan and the cgroup check;
//! 6. the decode step and the split MoE block over the bank;
//! 7. the host feeds: the embedding table and the PLE block (host projections), the n-gram table.
//!
//! Lifetime: `spawn_engine` needs an executor without a lifetime and the step borrows its owners, so
//! the projector, the stream, the checkpoint and the weights are `Box::leak`ed once per engine.
//! ONE CUDA engine is built per process (the same rule as the Qwen3.6 CUDA engine): a guard pins
//! the flag before the first leak, and a retry in the same process is refused.

use std::{
    cell::Cell,
    path::Path,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use super::{
    bank::{
        BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
        ReplaceOnMissLru, device::DeviceBank,
    },
    bank_source::PackRecordSource,
    device::{CardChunkDevice, CardDevice},
    driver::chunk::ChunkDriver,
    driver::{DriverShape, StepperPositions, TokenDriver, TokenTimes},
    executor::{CudaRules, FlashCudaExecutor},
    feeds::StartShape,
    feeds::device::{DevicePle, PleUpload},
    load::{TrunkList, upload::trunk_weights},
    moe_block::{SplitMoe, SplitMoeTrunk},
    pack_feeds::PackFeeds,
    ple_feed::{FixedProjection, NgramRows, PleFeed, PleVectors},
    step::{FlashStep, SplitMoeAt},
};
use crate::{
    EngError,
    family::{
        exl3_pack::layer_name,
        flash_next::{
            artifact::tokenizer_pin,
            load::{
                model_shape,
                pack_files::{FlashPackFiles, NGRAM_FILE},
            },
            moe::{
                load_plan::FlashLoadPlan,
                tier_plan::{TierInputs, TierShape},
            },
            settings::{CudaSettings, FlashSettings},
        },
        flint::exl3_resident::HostBudget,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve, exl3::Exl3Weight},
        qwen36::{cuda::exl3_source::PackReader, tiered::budget::KvPlan},
        session::DecodeSession,
    },
    kernels::cuda::{
        CudaContext, DeviceBuffer,
        resident::exl3::{IndexedOptions, SERVED_SINGLE_ROW},
    },
    sched::worker::{Engine, ReadyNote, Stages, spawn_engine_noted},
    tokenizer::TokenizerBundle,
    weights::exl3::Checkpoint,
};

/// Threads of the host PLE projections (rows split over them). The two projection matvecs are
/// DRAM-bandwidth-bound (131 MB per token), so more threads helps until the socket saturates;
/// `INFERENCE_FLASH_NEXT_PLE_THREADS` overrides the default.
fn ple_threads() -> usize {
    std::env::var("INFERENCE_FLASH_NEXT_PLE_THREADS")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .map(|n| n.clamp(1, 128))
        .unwrap_or(8)
}

/// Whether the chunked prefill runs its two PLE projections on the card (the staged
/// `dense_gemv` pair): `INFERENCE_FLASH_NEXT_PLE_DEVICE=1` opts in, the default keeps the host
/// matvecs. The card reads the stored F16/BF16 weights once per chunk instead of the host's
/// 131 MB DRAM sweep per token; the rest of the PLE block stays on the host. Strict: any set
/// value other than `1` is refused.
fn ple_device() -> Result<bool, EngError> {
    match std::env::var("INFERENCE_FLASH_NEXT_PLE_DEVICE") {
        Ok(text) if text == "1" => Ok(true),
        Ok(_) => Err(EngError::BadRequest(
            "INFERENCE_FLASH_NEXT_PLE_DEVICE must be unset or \"1\"".into(),
        )),
        Err(_) => Ok(false),
    }
}

/// The id-indexed options Flash-Next serves its routed experts with: the DP4A decode of codebook 2
/// (the codebook of this pack), one launch for the 20 gate and up pairs of ten picks, and the
/// pipelined tile loop on: it writes the bytes of the default launches and the serve A/B
/// (alternating arms, two rounds, server metrics) took 8-11% off the time to first token at
/// every depth and added 5-8% decode tokens per second. A set `FIFI_EXL3_DP4A`,
/// `FIFI_EXL3_INDEXED_PAIRS` or `FIFI_EXL3_PIPELINE` still wins for its own option.
const SERVED_INDEXED: IndexedOptions = IndexedOptions {
    dp4a: true,
    pairs20: true,
    pipeline: true,
};

/// Registers the kernels the Flash-Next decode runs on its context: the wide single-row kernel for
/// every dense projection and the id-indexed options of the routed experts. Qwen3.6 and Flint
/// register their own; a bad launch switch or an option the kernel cannot combine with is refused
/// here, before any weight is uploaded.
fn register_served_kernels(ctx: &CudaContext) -> Result<(), EngError> {
    ctx.set_single_row_default(SERVED_SINGLE_ROW);
    ctx.set_indexed_default(SERVED_INDEXED);
    ctx.set_odd_rate_kernel(true);
    // The fused-input launch of the dense projections (exl3_decode_fused.h): the input rotation
    // runs inside the projection kernel, two launches per set instead of three, bit-identical
    // bytes (64/64 receipt logits match). Isolated it saves ~1.27 us per launch, but measured on
    // the full 64-step decode it REGRESSES the step: median 39.25 vs 38.28 ms and paired mean
    // +2.25 ms (steps 9-64 of the adjacent off/on pair; REPORT.md cites the artifacts) - the
    // larger shared footprint and the per-column-block rotation recompute reduce the overlap the
    // 253 small launches had with the attention and expert kernels. Kept off for re-measuring
    // after the launch floor changes (e.g. graph capture); `FIFI_FLASH_NEXT_FUSED_IN=1` opts in,
    // any other non-empty value is refused, never a silent fallback. Read once here, per process.
    let fused_in = match std::env::var("FIFI_FLASH_NEXT_FUSED_IN") {
        Err(_) => false,
        Ok(v) if v == "1" => true,
        Ok(v) if v.is_empty() || v == "0" => false,
        Ok(v) => {
            return Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_FUSED_IN={v}: only 0 and 1 are valid"
            )));
        }
    };
    ctx.set_fused_in_kernel(fused_in);
    ctx.check_single_gather()?;
    eprintln!(
        "flash-next cuda: single-row kernel {}{}",
        ctx.single_row_kernel_name()?,
        ctx.single_gather_suffix()
    );
    if fused_in {
        eprintln!("flash-next cuda: fused-input launch on");
    }
    Ok(())
}

static CUDA_ENGINE_BUILT: AtomicBool = AtomicBool::new(false);

fn already_built() -> EngError {
    EngError::BadRequest(
        "engine config: the Flash-Next CUDA backend builds one engine per process: its weights are leaked for the process lifetime; configure a single qwen3_8_flash_next entry and restart the process to reload"
            .to_string(),
    )
}

/// Takes the one-engine flag; released on drop unless committed (a failure before the first leak
/// owns nothing on the card).
pub(super) struct BuildGuard {
    committed: bool,
}

impl BuildGuard {
    fn acquire() -> Result<Self, EngError> {
        CUDA_ENGINE_BUILT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| already_built())?;
        Ok(Self { committed: false })
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        if !self.committed {
            CUDA_ENGINE_BUILT.store(false, Ordering::Release);
        }
    }
}

/// Loads and pins the tokenizer, then starts the engine thread, which loads the pack. The engine
/// state stays `Loading` until the executor exists.
pub(in crate::family::flash_next) fn start(
    settings: &FlashSettings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle), EngError> {
    let cuda = settings.cuda.as_ref().ok_or_else(|| {
        EngError::BadRequest("engine config: the CUDA decoder is not enabled".to_string())
    })?;
    let manifest = std::fs::read(&settings.manifest).map_err(|error| {
        EngError::Checkpoint(format!("{}: {error}", settings.manifest.display()))
    })?;
    let pin = tokenizer_pin(&manifest, settings.tokenizer_vocab_size)?;
    let bundle = TokenizerBundle::load(&settings.tokenizer_dir, &pin)?;
    let rules = CudaRules {
        eos: Arc::from(bundle.eos_token_ids()),
        sampler_vocab: usize::try_from(bundle.vocab_size())
            .map_err(|_| EngError::Checkpoint("tokenizer vocabulary exceeds usize".to_string()))?,
    };
    let prefill_chunk = usize::try_from(settings.prefill_chunk)
        .map_err(|_| EngError::BadRequest("engine config: prefill_chunk exceeds usize".into()))?;
    let artifact = settings.artifact.clone();
    let wanted = cuda.clone();
    let guard = BuildGuard::acquire()?;
    let engine = spawn_engine_noted(prefill_chunk, ready_note, move |stages| {
        make(stages, &artifact, &wanted, rules, guard)
    })?;
    Ok((engine, bundle))
}

fn raw<'w>(weights: &'w CudaWeights, name: &str) -> Result<&'w DeviceBuffer, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => Ok(buffer),
        _ => Err(EngError::Checkpoint(format!(
            "Flash-Next start: {name} is not an uploaded raw tensor"
        ))),
    }
}

fn exl3<'w>(weights: &'w CudaWeights, name: &str) -> Result<&'w Exl3Weight, EngError> {
    match weights.get(name) {
        Some(CudaWeight::Exl3 { weight, .. }) => Ok(weight),
        _ => Err(EngError::Checkpoint(format!(
            "Flash-Next start: {name} is not an uploaded EXL3 matrix"
        ))),
    }
}

/// The trunk of the split MoE block of every layer, borrowed from the uploaded weights.
fn moe_trunks(
    weights: &'static CudaWeights,
    layers: usize,
) -> Result<Vec<SplitMoeTrunk<'static>>, EngError> {
    (0..layers)
        .map(|layer| {
            let at = |suffix: &str| layer_name(layer, suffix);
            Ok(SplitMoeTrunk {
                router: raw(weights, &at("mlp.gate.weight"))?,
                router_shared: raw(weights, &at("mlp.shared_expert_gate.weight"))?,
                shared_gate: exl3(weights, &at("mlp.shared_expert.gate_proj"))?,
                shared_up: exl3(weights, &at("mlp.shared_expert.up_proj"))?,
                shared_down: exl3(weights, &at("mlp.shared_expert.down_proj"))?,
            })
        })
        .collect()
}

fn read_meminfo() -> Result<(u64, u64), EngError> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| EngError::MemoryPlan(format!("/proc/meminfo: {error}")))?;
    let memory = HostMemory::from_meminfo(&text)?;
    Ok((memory.total, memory.available))
}

/// A record source that adds up the time spent reading and packing records from the pack, so the
/// load log can split the bank fill into the pack read and the pin and upload.
struct TimedSource<'a> {
    inner: &'a PackRecordSource<'a>,
    spent: Cell<Duration>,
    records: Cell<u64>,
}

impl RecordSource for TimedSource<'_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        let opened = Instant::now();
        let result = self.inner.record(layer, expert, out);
        self.spent.set(self.spent.get() + opened.elapsed());
        self.records.set(self.records.get() + 1);
        result
    }
}

/// Reads an exllamav3 `expert_stats` JSON into per-layer expert use weights, in layer order.
/// Refuses a missing file, bad JSON, any layer key that is not `model.language_model.layers.<i>.mlp`,
/// or a layer index outside `0..layers`. A duplicate layer key cannot be detected here: the JSON
/// parser keeps the last value of a repeated object key, so the layer reads as missing instead.
/// The weights themselves are validated against the bank geometry by [`fill_from_profile`].
fn read_expert_stats(path: &str, layers: usize) -> Result<Vec<Vec<f64>>, EngError> {
    let bad =
        |detail: String| EngError::BadRequest(format!("expert_stats profile {path}: {detail}"));
    let text = std::fs::read_to_string(path).map_err(|e| bad(format!("unreadable: {e}")))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| bad(format!("not valid JSON: {e}")))?;
    let layer_map = parsed
        .get("layers")
        .and_then(|v| v.as_object())
        .ok_or_else(|| bad("no layers object".into()))?;
    let mut out: Vec<Option<Vec<f64>>> = vec![None; layers];
    for (key, weights) in layer_map {
        let index: Option<usize> = key
            .strip_prefix("model.language_model.layers.")
            .and_then(|rest| rest.strip_suffix(".mlp"))
            .and_then(|n| n.parse().ok());
        let index = index.ok_or_else(|| bad(format!("layer key {key} is not expected")))?;
        if index >= layers {
            return Err(bad(format!("layer index {index} outside 0..{layers}")));
        }
        let row: Vec<f64> = serde_json::from_value(weights.clone())
            .map_err(|e| bad(format!("layer {index}: not a list of numbers: {e}")))?;
        if out[index].is_some() {
            return Err(bad(format!("layer {index} appears twice")));
        }
        out[index] = Some(row);
    }
    out.into_iter()
        .enumerate()
        .map(|(index, row)| row.ok_or_else(|| bad(format!("layer {index} is missing"))))
        .collect()
}

/// The per-layer slot fill: expert ids in slot order and their LRU counts.
type FillFromProfile = (Vec<Vec<i32>>, Vec<Vec<u64>>);

/// The slot rows and LRU counts of the initial fill. With a profile: per layer, the experts sorted
/// by use weight (highest first, id breaking ties) take the slots in rank order and the weights
/// scale to `u64` counts. Without: the first experts by id take the slots and lower ids count as
/// hotter. Pure; both arms keep every invariant `ReplaceOnMissLru::from_fill` checks.
fn fill_from_profile(
    geometry: BankGeometry,
    profile: Option<&[Vec<f64>]>,
) -> Result<FillFromProfile, EngError> {
    if let Some(profile) = profile
        && profile.len() != geometry.layers
    {
        return Err(EngError::BadRequest(format!(
            "Flash-Next expert bank: the popularity profile has {} layers, the bank has {}",
            profile.len(),
            geometry.layers
        )));
    }
    let mut rows = Vec::with_capacity(geometry.layers);
    let mut counts = Vec::with_capacity(geometry.layers);
    for layer in 0..geometry.layers {
        let Some(weights) = profile.and_then(|p| p.get(layer)) else {
            rows.push(
                (0..geometry.experts)
                    .map(|e| if e < geometry.slots { e as i32 } else { -1 })
                    .collect(),
            );
            counts.push((0..geometry.experts as u64).rev().collect());
            continue;
        };
        if weights.len() != geometry.experts || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(EngError::BadRequest(format!(
                "Flash-Next expert bank: layer {layer} of the popularity profile has {} weights \
                 (need {}, each finite and non-negative)",
                weights.len(),
                geometry.experts
            )));
        }
        let mut by_weight: Vec<usize> = (0..geometry.experts).collect();
        by_weight.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
        let mut row = vec![-1; geometry.experts];
        for (slot, &expert) in by_weight.iter().take(geometry.slots).enumerate() {
            row[expert] = slot as i32;
        }
        rows.push(row);
        counts.push(weights.iter().map(|w| (*w * 1000.0) as u64).collect());
    }
    Ok((rows, counts))
}

/// The bank: under the host plan the caller admitted. The plan's mode picks the store (an inclusive
/// plan holds every expert, so a miss copies one way only). `profile`, when the caller read one, is
/// the per-layer expert use weights of a serving run: the hottest `slots` experts fill the slots in
/// rank order and the counts keep them the last evicted; without one the first experts by id fill
/// the slots and lower ids count as hotter.
fn build_bank(
    projector: &CudaProjector,
    geometry: BankGeometry,
    plan: &HostPinPlan,
    profile: Option<&[Vec<f64>]>,
    usage_out: Option<&std::path::Path>,
    source: &PackRecordSource<'_>,
) -> Result<DeviceBank, EngError> {
    let (rows, counts) = fill_from_profile(geometry, profile)?;
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, usage_out)?;

    let state = ExpertBankState::with_policy(geometry, plan.mode, policy)?;
    let timed = TimedSource {
        inner: source,
        spent: Cell::new(Duration::ZERO),
        records: Cell::new(0),
    };
    let opened = Instant::now();
    let bank = DeviceBank::build(projector.context(), state, &timed, plan)?;
    let total = opened.elapsed();
    eprintln!(
        "flash-next cuda: expert bank: {} records, pack read and pack {:.1} s, pin and upload {:.1} s, total {:.1} s",
        timed.records.get(),
        timed.spent.get().as_secs_f64(),
        total.saturating_sub(timed.spent.get()).as_secs_f64(),
        total.as_secs_f64()
    );
    Ok(bank)
}

fn make(
    stages: &Stages<'_>,
    artifact: &Path,
    wanted: &CudaSettings,
    rules: CudaRules,
    guard: BuildGuard,
) -> Result<(FlashCudaExecutor<StepDriver>, DecodeSession), EngError> {
    let driver = build_driver(
        &|name| stages.begin(name),
        artifact,
        wanted,
        rules.sampler_vocab,
        Some(guard),
    )?;
    FlashCudaExecutor::new(driver, rules)
}

/// Loads the pack and builds the token driver. `guard` pins the one-engine flag at the first leak;
/// a device test passes none.
pub(super) fn build_driver(
    stage: &dyn Fn(&'static str),
    artifact: &Path,
    wanted: &CudaSettings,
    sampler_vocab: usize,
    mut guard: Option<BuildGuard>,
) -> Result<StepDriver, EngError> {
    let began = Instant::now();
    stage("initializing CUDA");
    let projector = CudaProjector::new()?;
    register_served_kernels(projector.context())?;
    let free_vram = projector.memory_info()?.free as u64;

    stage("opening the pack");
    let files = FlashPackFiles::open(artifact)?;
    let FlashPackFiles {
        checkpoint,
        ngram,
        report,
    } = files;
    let (model, kinds) = model_shape(&checkpoint.model_config)?;
    let layers = kinds.len();
    let ngram_bytes = std::fs::metadata(artifact.join(NGRAM_FILE))
        .map_err(|error| EngError::Checkpoint(format!("{NGRAM_FILE}: {error}")))?
        .len();

    stage("planning memory");
    let tier_shape = TierShape::from_pack(
        layers,
        model.moe.experts(),
        model.moe.top_k(),
        report.expert_bytes,
    )?;
    let (mem_total, mem_available) = read_meminfo()?;
    // The prefill ring's budget, carved out before the load plan so the bank sizes itself
    // into what remains: at most 1.5 GiB and 30% of the free card, only for chunked prefill.
    // A ring below one streamed layer is not built (the multi-row path then keeps its
    // union-admit shape, correct and slower).
    let record_bytes = PackRecordSource::new(&checkpoint)?.layout().record_bytes();
    // INFERENCE_FLASH_NEXT_RING_BUDGET sets the ring's bytes. The ring measures a serve
    // regression today (it re-issues its copies every chunk while the union-admit path
    // copies once), so the default is off; the flag keeps the lane's experiments open.
    // A budget above the free VRAM is clamped, not wrapped: the plan then fits the card.
    let ring_budget = match std::env::var("INFERENCE_FLASH_NEXT_RING_BUDGET") {
        Ok(bytes) => bytes.trim().parse::<u64>().map_err(|_| {
            EngError::BadRequest("INFERENCE_FLASH_NEXT_RING_BUDGET must be a byte count".into())
        })?,
        Err(_) => 0,
    };
    // `FlashLoadPlan::new` replaces the trunk and state bytes (the `(0, 0)`) and the K/V (the
    // placeholder positions) with the numbers the pack and the context give.
    let card = TierInputs::measured(
        tier_shape,
        (0, 0),
        KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
        free_vram.saturating_sub(ring_budget),
        (mem_total, mem_available),
    );
    let load_plan = FlashLoadPlan::new(&report, &checkpoint.model_config, ngram_bytes, card)?;
    let load = load_plan.at_context(
        u64::from(wanted.context),
        wanted.kv_format.format(),
        model.moe.top_k() + 1,
    )?;
    let planned_slots = load.context.plan.slots_per_layer();
    let slots = match wanted.slots {
        None => planned_slots,
        Some(asked) if (asked as usize) <= planned_slots => asked as usize,
        Some(asked) => {
            return Err(EngError::MemoryPlan(format!(
                "Flash-Next start: slots {asked} per layer asked, the memory plan at context {} leaves {planned_slots}",
                wanted.context
            )));
        }
    };
    eprintln!("{load}; serving {slots} slots per layer");
    if sampler_vocab == 0 || sampler_vocab > model.vocab {
        return Err(EngError::Checkpoint(format!(
            "sampler vocabulary {} is outside 1..={}, the width of the model head",
            sampler_vocab, model.vocab
        )));
    }

    // The host side is admitted before anything is uploaded: the pinned store of the bank, the
    // engine heap and margin of `HostTerms`, and the two PLE projection matrices that
    // `PleVectors::read` holds in F32 (`HostTerms::engine_heap` does not count them).
    let geometry = BankGeometry::new(
        layers,
        model.moe.experts(),
        model.moe.top_k(),
        slots,
        record_bytes,
    )?;
    // The store mode: an inclusive store holds every expert pinned, so a miss copies one way
    // only (the D2H write-back of the evicted expert disappears); it needs every record's bytes
    // of host RAM (43.07 GiB on this pack), so `auto` takes it only when the plan admits it and
    // falls back to the exclusive store of spare places. `FIFI_FLASH_NEXT_BANK_MODE` forces a
    // mode (`inclusive` or `exclusive`); anything else is refused, never a silent fallback.
    let memory = HostMemory {
        total: mem_total,
        available: mem_available,
    };
    let host_plan = match std::env::var("FIFI_FLASH_NEXT_BANK_MODE").as_deref() {
        Err(_) | Ok("") | Ok("auto") => {
            HostPinPlan::new(&geometry, HostMode::Inclusive, &memory, &HostTerms::DEFAULT).or_else(
                |inclusive| {
                    eprintln!(
                        "flash-next cuda: inclusive store refused ({inclusive:?}); exclusive store"
                    );
                    HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &HostTerms::DEFAULT)
                },
            )?
        }
        Ok("inclusive") => {
            HostPinPlan::new(&geometry, HostMode::Inclusive, &memory, &HostTerms::DEFAULT)?
        }
        Ok("exclusive") => {
            HostPinPlan::new(&geometry, HostMode::Exclusive, &memory, &HostTerms::DEFAULT)?
        }
        Ok(other) => {
            return Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_BANK_MODE={other}: only auto, inclusive and exclusive are valid"
            )));
        }
    };
    eprintln!(
        "flash-next cuda: {} store, {} pinned host bytes",
        match host_plan.mode {
            HostMode::Inclusive => "inclusive",
            HostMode::Exclusive => "exclusive",
        },
        host_plan.store_bytes
    );
    let ple_width = (model.hyper.hidden() * model.hyper.branches() + model.hyper.hidden()) as u64;
    let ple_bytes = ple_width * model.ple.embed() as u64 * 4;
    HostBudget::probe()?.admit_for(
        host_plan.process_bytes + ple_bytes,
        "Flash-Next expert bank and PLE projections",
    )?;

    // LEAK: the owners are leaked for the process lifetime, giving the step `'static` borrows.
    // From the first leak a failure leaves the card loaded, so the one-engine flag is pinned here.
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let checkpoint: &'static Checkpoint = Box::leak(Box::new(checkpoint));
    if let Some(guard) = guard.as_mut() {
        guard.commit();
    }
    let ctx = projector.context();

    stage("loading weights");
    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config)?;
    let reader = PackReader::new(projector, checkpoint, Reserve::default());
    let weights: &'static CudaWeights =
        Box::leak(Box::new(trunk_weights(&reader, &list, &load_plan)?));
    eprintln!(
        "flash-next cuda: trunk uploaded in {:.1} s",
        began.elapsed().as_secs_f64()
    );

    stage("filling the expert bank");
    let source = PackRecordSource::new(checkpoint)?;
    // Optional popularity profile, an exllamav3 expert_stats JSON:
    // {"model_dir": .., "layers": {"model.language_model.layers.<i>.mlp": [weights]}}. Sources,
    // in order: the engine's `expert_stats` key (which also takes the usage dump, so the next
    // start of the same entry serves warm) and `FIFI_FLASH_NEXT_EXPERT_STATS`. A set env value
    // must read and parse; a malformed file is refused, never a silent identity fill. A
    // not-yet-written engine-key file is a cold start (the dump creates it on shutdown).
    let env_stats = std::env::var("FIFI_FLASH_NEXT_EXPERT_STATS")
        .ok()
        .filter(|path| !path.is_empty());
    let stats_path = wanted
        .expert_stats
        .as_deref()
        .map(std::path::Path::new)
        .or_else(|| env_stats.as_deref().map(std::path::Path::new));
    let profile = match &stats_path {
        None => None,
        Some(path) if wanted.expert_stats.is_some() && !path.exists() => {
            eprintln!(
                "flash-next cuda: expert bank: no profile at {} yet; identity fill, the dump writes it on shutdown",
                path.display()
            );
            None
        }
        Some(path) => Some(read_expert_stats(&path.display().to_string(), layers)?),
    };
    let usage_out = wanted.expert_stats.clone().or_else(|| {
        std::env::var("FIFI_FLASH_NEXT_EXPERT_USAGE_OUT")
            .ok()
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from)
    });
    if profile.is_some() {
        eprintln!(
            "flash-next cuda: expert bank: popularity profile {}",
            stats_path.map_or(String::new(), |p| p.display().to_string())
        );
    }
    let bank = build_bank(
        projector,
        geometry,
        &host_plan,
        profile.as_deref(),
        usage_out.as_deref(),
        &source,
    )?;
    eprintln!(
        "flash-next cuda: expert bank filled at {:.1} s",
        began.elapsed().as_secs_f64()
    );

    stage("building the step");
    // `prefill_rows` above one builds the step's plans at that width, so a prompt chunk runs as
    // device chunks of that many rows; one (the default) is the decode shape of before.
    let rows = wanted.prefill_rows as usize;
    let stream = Box::leak(Box::new(ctx.stream()?));
    let step = FlashStep::new(
        ctx,
        stream,
        weights,
        &checkpoint.model_config,
        wanted.context as usize,
        rows,
        0..layers,
    )?;
    let mut block = SplitMoe::new(
        ctx,
        moe_trunks(weights, layers)?,
        bank,
        source.layout(),
        &step.moe_dims(rows),
    )?;
    if rows > 1 && ring_budget > 0 {
        let slots = block.plan_ring(
            ctx,
            (ring_budget / u64::try_from(record_bytes).unwrap_or(u64::MAX)) as usize,
        )?;
        if slots == 0 {
            eprintln!(
                "flash-next cuda: prefill ring not funded ({} bytes reserved); the union-admit \
             path serves prefill",
                ring_budget
            );
        } else {
            eprintln!(
                "flash-next cuda: prefill ring of {slots} slots ({:.2} GiB) behind the bank",
                slots as f64 * record_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    let model_layers: Vec<usize> = (0..layers).collect();
    let moe = SplitMoeAt::new(block, &model_layers)?;

    stage("opening the n-gram table");
    let (vectors, projection) = PleVectors::read(
        &checkpoint.source,
        model.ple_layer,
        &model.ple,
        ple_threads(),
    )?;
    let ngram_rows = NgramRows::new(ngram)?;
    // The device route (chunked prefill only): the projections upload to the card and the feed
    // takes precomputed pairs; the host projection still backs the flag-off path.
    let device_ple = match ple_device()?.then_some(rows > 1) {
        Some(true) => {
            stage("uploading the PLE projections");
            let dims = model.ple.dims();
            let (dtype, key, value) =
                PleVectors::raw_projections(&checkpoint.source, model.ple_layer, &model.ple)?;
            Some(DevicePle::upload(
                ctx,
                dtype,
                &key,
                &value,
                PleUpload {
                    width: (dims.hidden * dims.branches) as usize,
                    hidden: dims.hidden as usize,
                    embed: model.ple.embed(),
                    rows_max: rows,
                },
            )?)
        }
        Some(false) => {
            eprintln!(
                "flash-next cuda: INFERENCE_FLASH_NEXT_PLE_DEVICE ignored (prefill rows = 1)"
            );
            None
        }
        None => None,
    };
    let ple = if device_ple.is_some() {
        PleFeed::with_fixed(
            model.ple,
            ngram_rows,
            vectors,
            Rc::new(FixedProjection::new()),
        )
    } else {
        PleFeed::new(model.ple, ngram_rows, vectors, Box::new(projection))
    };
    let hidden = model.hyper.hidden();
    let branches = model.hyper.branches();
    let ple_embed = model.ple.embed();
    let feeds = PackFeeds::new(&checkpoint.source, ple, hidden, branches);
    let shape = DriverShape {
        layers,
        ple_layer: model.ple_layer,
        positions: wanted.context as usize,
        start: StartShape {
            sampler_vocab,
            stream: branches * hidden,
            ple_embed,
        },
    };
    let driver = if rows > 1 {
        let ple_route = device_ple;
        let ple_on_card = ple_route.is_some();
        let mut chunked = ChunkDriver::new(
            CardChunkDevice::with_ple(step, moe, ple_route),
            feeds,
            shape,
        )?;
        chunked.chunk_cap = rows;
        chunked.ple_device = ple_on_card;
        eprintln!(
            "flash-next cuda: ready in {:.1} s, prefill chunks of {rows} rows{}",
            began.elapsed().as_secs_f64(),
            if ple_on_card {
                ", PLE projections on the card"
            } else {
                ""
            }
        );
        StepDriver::Chunk(chunked)
    } else {
        eprintln!(
            "flash-next cuda: ready in {:.1} s",
            began.elapsed().as_secs_f64()
        );
        StepDriver::Token(TokenDriver::new(CardDevice::new(step, moe), feeds, shape)?)
    };
    Ok(driver)
}

/// The one driver type at the engine boundary: the plain token driver of decode, or the chunk
/// driver `prefill_rows` builds. Forwards the stepper and the device-test surface.
pub(in crate::family::flash_next::cuda) enum StepDriver {
    Token(TokenDriver<CardDevice, PackFeeds>),
    Chunk(ChunkDriver<CardChunkDevice, PackFeeds>),
}

impl crate::family::session::TokenStepper for StepDriver {
    const OPERATION: &'static str = "flash_next_cuda";

    fn feed(&mut self, token: u32) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.feed(token),
            Self::Chunk(driver) => driver.feed(token),
        }
    }

    /// Forwarded, not defaulted: the chunk driver's whole point is the chunked `feed_rows`.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.feed_rows(tokens),
            Self::Chunk(driver) => driver.feed_rows(tokens),
        }
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Token(driver) => driver.logits(token),
            Self::Chunk(driver) => driver.logits(token),
        }
    }

    fn fed(&self) -> usize {
        match self {
            Self::Token(driver) => driver.fed(),
            Self::Chunk(driver) => driver.fed(),
        }
    }

    fn reset(&mut self) -> Result<(), EngError> {
        match self {
            Self::Token(driver) => driver.reset(),
            Self::Chunk(driver) => driver.reset(),
        }
    }
}

impl StepperPositions for StepDriver {
    fn positions(&self) -> usize {
        match self {
            Self::Token(driver) => driver.positions(),
            Self::Chunk(driver) => driver.positions(),
        }
    }
}

impl StepDriver {
    /// The bank's miss count for the per-job report.
    pub(super) fn bank_misses(&self) -> u64 {
        match self {
            Self::Token(driver) => driver.device().bank_misses(),
            Self::Chunk(driver) => driver.device().bank_misses(),
        }
    }

    /// The single-row kernel counters, for the device tests.
    #[cfg(test)]
    pub(super) fn kernel_launches(&self) -> crate::kernels::single_row::SingleRowLaunches {
        match self {
            Self::Token(driver) => driver.device().kernel_launches(),
            Self::Chunk(driver) => driver.device().kernel_launches(),
        }
    }

    /// The single-row kernel name, for the device tests (both devices expose it).
    #[cfg(test)]
    pub(super) fn single_row_kernel_name(&self) -> Result<&'static str, EngError> {
        match self {
            Self::Token(driver) => driver.device().single_row_kernel_name(),
            Self::Chunk(driver) => driver.device().single_row_kernel_name(),
        }
    }

    /// The last step's stage times, for the driver device tests. The chunk driver does not
    /// track stage times; the stage table runs on the token driver.
    pub(super) fn last_times(&self) -> TokenTimes {
        match self {
            Self::Token(driver) => driver.last_times(),
            Self::Chunk(_) => TokenTimes::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BankGeometry, fill_from_profile, read_expert_stats};
    use std::io::Write as _;

    /// A profile fills the slots in weight rank order (ties by id) and scales the weights to
    /// counts; the same bank without one keeps the id-ordered fill and the reversed-id counts.
    #[test]
    fn a_profile_ranks_the_initial_fill() {
        let geometry = BankGeometry::new(1, 6, 2, 3, 8).expect("geometry");
        let profile = [vec![1.0, 9.0, 4.0, 9.0, 0.0, 2.0]];
        let (rows, counts) = fill_from_profile(geometry, Some(&profile)).expect("fill");
        // experts 1 and 3 tie at the top weight; id breaks it, so 1 takes slot 0, 3 slot 1, 2 slot 2.
        assert_eq!(rows[0], [-1, 0, 2, 1, -1, -1]);
        assert_eq!(counts[0], vec![1000, 9000, 4000, 9000, 0, 2000]);
        let (rows, counts) = fill_from_profile(geometry, None).expect("identity fill");
        assert_eq!(rows[0], [0, 1, 2, -1, -1, -1]);
        assert_eq!(counts[0], (0..6u64).rev().collect::<Vec<u64>>());
    }

    /// A profile with the wrong layer count or non-finite weights is refused, not folded into
    /// the identity fill.
    #[test]
    fn a_bad_profile_is_refused() {
        let geometry = BankGeometry::new(2, 4, 2, 3, 8).expect("geometry");
        assert!(fill_from_profile(geometry, Some(&[vec![1.0; 4]])).is_err());
        assert!(
            fill_from_profile(geometry, Some(&[vec![1.0; 4], vec![-1.0, 0.0, 0.0, 0.0]])).is_err()
        );
        assert!(
            fill_from_profile(
                geometry,
                Some(&[vec![1.0; 4], vec![f64::NAN, 0.0, 0.0, 0.0]])
            )
            .is_err()
        );
    }

    /// The stats reader takes the stock layer keys in any object order, refuses a bad key, a
    /// duplicate, an out-of-range index and a missing layer, and returns weights in layer order.
    #[test]
    fn the_usage_dump_round_trips_through_the_reader() {
        let dir = std::env::temp_dir().join(format!("fn-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("dump.json");
        let geometry = BankGeometry::new(1, 4, 2, 3, 8).expect("geometry");
        let (rows, counts) = fill_from_profile(geometry, None).expect("identity fill");
        let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
        {
            let _policy =
                super::ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, Some(&path))
                    .expect("policy");
        } // drop writes the dump
        assert!(path.exists(), "the drop wrote the dump");
        let read = read_expert_stats(path.to_str().unwrap(), 1).expect("read back");
        assert_eq!(read, vec![vec![0.0; 4]], "zero picks dump zero weights");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_stats_reader_parses_and_refuses() {
        let dir = std::env::temp_dir().join(format!("fn-stats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let write = |name: &str, body: &str| {
            let path = dir.join(name);
            let mut file = std::fs::File::create(&path).expect("file");
            file.write_all(body.as_bytes()).expect("write");
            path
        };
        let good = write(
            "good.json",
            r#"{"model_dir": "x", "layers": {
                "model.language_model.layers.1.mlp": [2.0, 1.0],
                "model.language_model.layers.0.mlp": [0.5, 4.0]}}"#,
        );
        assert_eq!(
            read_expert_stats(good.to_str().unwrap(), 2).expect("read"),
            vec![vec![0.5, 4.0], vec![2.0, 1.0]]
        );
        for (name, body) in [
            ("badkey.json", "{\"layers\": {\"layers.0.mlp\": [1.0]}}"),
            (
                "dup.json",
                r#"{"layers": {"model.language_model.layers.0.mlp": [1.0], "model.language_model.layers.0.mlp": [2.0]}}"#,
            ),
            (
                "range.json",
                "{\"layers\": {\"model.language_model.layers.2.mlp\": [1.0]}}",
            ),
            (
                "missing.json",
                "{\"layers\": {\"model.language_model.layers.0.mlp\": [1.0]}}",
            ),
            ("notjson.json", "nope"),
        ] {
            let path = write(name, body);
            assert!(
                read_expert_stats(path.to_str().unwrap(), 2).is_err(),
                "{name} must be refused"
            );
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// The served kernels are registered and checked before the pack is opened and before any
    /// stream or weight exists, in the order set default, set options, check.
    #[test]
    fn the_served_kernels_are_registered_and_checked_before_the_first_upload() {
        let source = include_str!("start.rs");
        let register = &source[source
            .find("fn register_served_kernels(")
            .expect("register_served_kernels")..];
        let single = register
            .find("ctx.set_single_row_default(")
            .expect("the single-row default");
        let indexed = register
            .find("ctx.set_indexed_default(")
            .expect("the indexed default");
        let odd = register
            .find("ctx.set_odd_rate_kernel(true)")
            .expect("the odd-rate kernel switch");
        let check = register
            .find("ctx.check_single_gather()?")
            .expect("the switch check");
        assert!(
            single < indexed && indexed < odd && odd < check,
            "the defaults must be registered before check_single_gather()"
        );
        let build = &source[source
            .find("pub(super) fn build_driver(")
            .expect("build_driver")..];
        let call = build
            .find("register_served_kernels(projector.context())?")
            .expect("the registration in build_driver");
        let pack = build
            .find("stage(\"opening the pack\")")
            .expect("the pack open");
        let stream = build.find("ctx.stream()").expect("the stream");
        assert!(
            call < pack && call < stream,
            "register_served_kernels() must precede the pack open and the first stream"
        );
    }
}
