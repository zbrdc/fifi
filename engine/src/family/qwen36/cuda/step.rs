//! `QwenResidentStep`: the resident Qwen3.6 decode step on the CUDA backend (decision 0047,
//! packet RD-9d). One token is the launch list of `step_ops.rs` queued on ONE owned stream:
//! `step_begin`, the embedding, the 40 layers (the RD-9b mixer block, then the RD-9c MoE block),
//! the final norm, the `lm_head` projection, `argmax`, `step_advance`. Then the host reads the
//! result pair (and, when asked, the 993 KB logits row) with ONE blocking wait.
//!
//! # Structure of a token
//!
//! 1. a host start-block write (`write_start`: one blocking host-to-device copy);
//! 2. the private `run_body`, which ONLY enqueues launches (its doc states the contract);
//! 3. the read-back, the event and ONE blocking wait (`read_back`).
//!
//! Since RD-11a the body of a token runs in one of three modes (the pure `next_mode` rule
//! below): eager (the warm-up token, every fallback and every plain session), captured (one
//! capture, then replays), or eager-after-refusal. A refused capture is STICKY for the whole
//! step construction: later tokens stay eager and keep reporting
//! `RefusedFellBackToEager`, across `reset_state` and `import_state`, which are
//! `ResidentState` methods and cannot touch the step's own flags.
//!
//! # Call order and errors (RD-9b, binding)
//!
//! Per token: `write_step_params`, `enqueue_rows`, the body (`run_block` for each layer), then
//! `commit_step(accepted, status)` with the pair's status. A failure of the first op of the body
//! takes the step back (`cancel_enqueue`); any later failure, any error of the read-back and any
//! non-zero status bit (REPLY-76) poisons the state. Poisoned means: only `reset` or a full
//! reset-equivalent `import_state` leaves it.
//!
//! One row only: `m = 1` everywhere (the MoE block takes one row).
//!
//! Drop order: the captured graph first (it keeps the holds of its capture and covers the state
//! buffers by drop order), then the borrowed weights and views, the state, the pinned word, the
//! event and the stream; `Drop` synchronizes the stream first.
//!
//! Design guidance only, no code copied.

use super::{
    dims::{KvMode, StepOptions},
    layer_table::LayerTable,
    mixer::{BlockIo, MixerParts, MixerViews, Mixers},
    mixer_geometry::MixerGeometry,
    mixer_weights::MixerWeights,
    moe::{
        ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, Packs, RouteBufs, run_moe_block,
        tiered::TieredScratch,
    },
    prefill_driver::{
        AdmitFacts, PrefillArena, PrefillBudget, admit_guard, admit_m_sess, needs_arena,
    },
    prefill_plan::{chunk_costs, state_bytes},
    prefix::CudaPrefillPath,
    readback::{
        ReadbackPlan, SYNC_EACH_VAR, captured_report, check_mode, eager_report, parse_sync_each,
        refused_report, with_host_syncs,
    },
    resources::{
        EmbedWeight, GlobalSpecs, GlobalWeights, HeadWeight, MoeSpecs, STEP_ROWS, StepResources,
        f32_bytes, moe_dims, moe_layer_parts, moe_layer_weights,
    },
    rules::{QwenRules, StepWidths, check_token_id, validate_step_inputs},
    run_report::{CaptureOutcome, HostSyncs, PrefillReport, RunReport, StepKind, StepRecord},
    state::{ResidentState, StepField, export::ExportedState},
    state_layout::{StateLayout, Word},
    step_ops::{StepOp, final_resid, resid_in, resid_out, token_ops},
    step_plan::Slot,
    tiered_step::{LayerParts, TieredRuntime},
    verdict::{
        Decision, FailureAction, action_for, bad_token_error, decide, entered_after_failed_sync,
    },
};
use crate::{
    EngError,
    family::gguf_model::Role,
    family::project::cuda::{CudaWeight, CudaWeights, WeightLayout},
    family::qwen36::settings::DEFAULT_CUDA_PREFILL_CHUNK,
    family::qwen36::tiered::{host::TieredHost, report::TokenReport},
    family::session::PrefixSlot,
    kernels::cuda::{
        CapturePlan, CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, ExpertBank,
        GraphExec, MatmulIo, PinnedWord, Projection,
        resident::{
            add_norm_pack::AddNormCall,
            add_rmsnorm_q8::AddNormQ8Call,
            embed::{EmbedCall, EmbedNativeCall},
            matvec_q::{MatvecCall, QuantType},
            step_params::{ArgmaxCall, ResultPair, StepParams, StepStatus},
        },
    },
};

const F32: usize = size_of::<f32>();

/// Capture settings of the step; distinct from `dims::StepOptions` (the dimensions options of
/// `new`).
pub(crate) struct CaptureConfig {
    /// Whether the step should capture its body into a replayable graph.
    pub(crate) capture: bool,
}

/// How the body of this token will run (the dispatch discriminator of `run_token`).
pub(super) enum Mode {
    /// The stored graph replays.
    Replay,
    /// The first capture is attempted (a warmed step that asked for it and never refused one).
    Capture,
    /// The launches run one by one.
    Eager,
}

/// The dispatch rule: replay once a graph exists; attempt the capture only on a warmed step
/// that asked for it and has not already refused one; otherwise run eagerly.
pub(super) fn next_mode(graph_ready: bool, warmed: bool, capture_on: bool, refused: bool) -> Mode {
    if graph_ready {
        Mode::Replay
    } else if warmed && capture_on && !refused {
        Mode::Capture
    } else {
        Mode::Eager
    }
}

/// The pure part of `capture_once`: what to do with the result of the capture attempt.
/// `Ok(Some(graph))`: the capture succeeded, the caller stores and replays the graph. `Ok(None)`:
/// the device refused the capture, so the attempt's op count is discarded (`entered` back to 0,
/// the failure rule must judge the rerun alone), the sticky refusal is set (`refused`) and the
/// caller reruns the body eagerly. `Err`: any other error comes back unchanged, with `entered`
/// and `refused` untouched, for the caller to fail on.
pub(super) fn fold_capture<G>(
    attempt: Result<(G, ()), EngError>,
    is_refusal: fn(&EngError) -> bool,
    entered: &mut usize,
    refused: &mut bool,
) -> Result<Option<G>, EngError> {
    match attempt {
        Ok((graph, ())) => Ok(Some(graph)),
        Err(error) if is_refusal(&error) => {
            *entered = 0;
            *refused = true;
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// How the body of one token ran (the discriminator `reported_outcome` maps to the outcome).
#[derive(Debug, Clone, Copy)]
pub(super) enum HowRan {
    /// The launches ran one by one (warm-up, fallback or plain session).
    Eager,
    /// A capture succeeded and the graph ran this token.
    Captured,
    /// A capture was refused and the body ran eagerly in the same token.
    Refused,
}

/// The outcome one token reports: after a refusal every later token keeps reporting the
/// fallback (sticky outcome), so the one-read gate FAILs the silent fallback with no new field.
pub(super) fn reported_outcome(ran: HowRan, refused: bool) -> CaptureOutcome {
    match ran {
        HowRan::Eager if refused => CaptureOutcome::RefusedFellBackToEager,
        HowRan::Eager => CaptureOutcome::Eager,
        HowRan::Captured => CaptureOutcome::Captured,
        HowRan::Refused => CaptureOutcome::RefusedFellBackToEager,
    }
}

/// The one typed refusal of a layout mismatch (MATVEC-PAR S3b): the step runs the layout of the
/// loaded weights and the options must agree with it. Pure, so the CPU tests pin it directly.
pub(super) fn agree_layout(weights: WeightLayout, options: WeightLayout) -> Result<(), EngError> {
    if weights != options {
        return Err(EngError::Checkpoint(format!(
            "the step options run {options:?} over {weights:?} weights: the layout follows the \
             loaded model"
        )));
    }
    Ok(())
}

/// The tiered mode is eager only: it waits for the host inside every layer, which a captured
/// graph cannot do (decision 0047, amendment of 2026-10-13). Pure, so the CPU tests pin it.
pub(super) fn tiered_capture_refusal(tiered_mode: bool, capture: bool) -> Result<(), EngError> {
    if tiered_mode && capture {
        return Err(EngError::BadRequest(
            "step: the tiered mode is eager only: it waits for the host inside every layer, \
             which a captured graph cannot do (decision 0047, amendment of 2026-10-13)"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The environment value of `INFERENCE_CUDA_SYNC_EACH`, `None` when unset.
fn sync_each_from_env() -> Result<bool, EngError> {
    match std::env::var(SYNC_EACH_VAR) {
        Ok(value) => parse_sync_each(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_sync_each(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(EngError::BadRequest(format!(
            "{SYNC_EACH_VAR} is not valid text"
        ))),
    }
}

/// The resident decode step of one session. See the module doc.
pub(crate) struct QwenResidentStep<'a> {
    // Declaration order is drop order: the graph first, then the borrows, the stream last.
    /// The leaked pin slice of the weights and the arena slot: one leaked slice per new step
    /// (per session in stage 1); the graph bakes their addresses in.
    pins: &'a [&'a DeviceBuffer],
    /// The replayable body, once a capture has succeeded. Declared BEFORE the state field: the
    /// graph (which keeps the arena holds and borrows the pin slice) drops first, and that
    /// covers the state buffers by drop order -- they live inside the step and never move
    /// after they are built, so the pin list need not name them (the one documented deviation
    /// from the plan's pins doc, accepted for stage 1).
    graph: Option<GraphExec<'a>>,
    context: &'a CudaContext,
    res: &'a StepResources,
    mixers: Mixers<'a>,
    moe: Vec<MoeLayerWeights<'a>>,
    /// The tiered mode (OFF-2 D5): the options asked for it. The routed expert stacks are then not
    /// loaded, `moe` is empty and every layer's MoE block runs through `tiered`. Cost: each layer of
    /// each token syncs with the host: about 0.8 ms per token over the plain step when every expert is
    /// resident (3090, 256 slots, measured), and about 18 ms per step with most picks on the CPU (slots 40
    /// with preload, measured). A prompt is prefilled one token per step, so a prompt of n tokens costs n
    /// of these steps until the batched tiered prefill (OFF-7) exists.
    tiered_mode: bool,
    /// The router and shared weights of each layer, held until `attach_tiered` builds the runtime.
    tiered_parts: Option<Vec<LayerParts<'a>>>,
    /// The tiered runtime; interior mutability because the token body takes `&self`.
    tiered: Option<std::cell::RefCell<TieredRuntime<'a>>>,
    /// The counters of the last token's boundary.
    tiered_report: TokenReport,
    globals: GlobalWeights<'a>,
    /// The layout the step runs (MATVEC-PAR S3b): the loaded weights' own layout, checked
    /// against the options at construction.
    layout: WeightLayout,
    moe_dims: MoeDims,
    /// The prefill arena of the session (CPF-7b-C B.1/B.3): admitted ONCE by
    /// `admit_prefill` after the step is built and the cap is set, and only for a session
    /// whose pre-arena verdict passes (`needs_arena`). Declared BEFORE the state: the
    /// arena drops before the state's buffers by drop order.
    prefill: Option<PrefillArena>,
    state: ResidentState,
    word: PinnedWord,
    event: DeviceEvent,
    stream: DeviceStream,
    sync_each: bool,
    /// Set once a token has passed the host checks (token id, read-back plan) and is about to
    /// write its start block, succeeded or not: the capture mode is fixed from then on. A token
    /// the host refuses before that has no device effect and does not freeze the mode.
    started: bool,
    /// Set only at the end of a fully successful eager token (a failed first token does not
    /// warm; the warm-up retries).
    warmed: bool,
    /// The capture setting of `set_capture_config` (default false).
    capture_on: bool,
    /// The sticky refusal flag: set only by a refused capture, never cleared -- `reset_state`
    /// and `import_state` are `ResidentState` methods and cannot touch it.
    refused: bool,
    pub(super) rules: QwenRules,
    /// The sampler widths the step wires in (see `rules::StepWidths`): the argmax `n` and the
    /// verdict width, both the sampler vocabulary, never the padded head width.
    widths: StepWidths,
    /// True when any MMQ weight of the chunk is Q6_K (`any_chunk_weight_q6k` at build), the
    /// refusal of decision 0049 amendment A6 item 4.
    chunk_q6k: bool,
    /// The cap of `M_sess` (decision 0049, A6 item 2), set by `set_m_sess_cap`; the value is
    /// the no-cap default until lane M's CPF-7a integration commit wires the config member
    /// `qwen36.prefill_chunk` into it (lane C, 2026-10-14).
    m_sess_cap: u32,
    /// The bytes of the last logits row read back (little-endian f32).
    logits_bytes: Vec<u8>,
    report: RunReport,
    /// The prefill report of this session (CPF-7b-C B.8): the chunk and feed steps and the
    /// counts the serve surface prints. Host values only.
    prefill_report: PrefillReport,
    /// The engine's one saved prefix entry (decision 0036, addendum 1): host memory only
    /// (`ExportedState`), so its place in the drop order is free.
    pub(super) prefix: PrefixSlot<ExportedState, CudaPrefillPath>,
}

/// The typed verdict of the chunk gate (plan CPF-V4 rev 3 §C.4): whether `feed_rows` may run
/// the chunk path for this session. Every refusal keeps the per-token prompt path; the run
/// report records the reason when its fields land (CPF-7b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkVerdict {
    /// The session may use the chunk path. `m_sess` is the ADMITTED `M_sess`, one of
    /// `M_LADDER` (`PrefillArena::m_sess`), never the raw cap: the arena admission of
    /// CPF-7b-C narrows the cap of `set_m_sess_cap` to the largest ladder size whose
    /// device sum fits the budget (CPF-7b-C B.2).
    Ready {
        /// The first reader is the chunk list of the prefill driver (CPF-1, decision 0049).
        m_sess: usize,
    },
    /// The session runs the Tiled layout: the chunk path is defined for Native sessions only
    /// (decision 0049, amendment A4 item 2).
    TiledLayout,
    /// The KV cache is Int8G32: v1 has no KV8 prefill (only the default serve path selects
    /// Fp16; `kv_dtype` q8 selects Int8G32 on the Native layout).
    Kv8,
    /// The session is tiered: `chunk_ready` sends tiered sessions to the per-token loop
    /// (plan §C.4); the tiered prefill stage is OFF-7's seam I5.
    Tiered,
    /// An MMQ weight of the chunk is Q6_K: no Q6_K chunk row exists, so the session keeps the
    /// per-token path (decision 0049, A6 item 4).
    Q6KWeight,
    /// The session admitted no prefill arena (no `M` of the ladder fits the budget, or
    /// `admit_prefill` was not called); the session keeps the per-token path (CPF-7b-C
    /// B.2, carry item 4 of the merge note).
    NoArena,
}

/// The chunk gate's inputs, fixed at step build (plan §C.4): the session's layout, KV mode
/// and tier flag, the Q6_K scan of the chunk's MMQ weights, the cap of `M_sess`, and the
/// admitted arena (CPF-7b-C B.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkFacts {
    pub(super) layout: WeightLayout,
    pub(super) kv_mode: KvMode,
    pub(super) tiered: bool,
    /// True when any MMQ weight of the chunk is Q6_K (`any_chunk_weight_q6k`).
    pub(super) mmq_q6k: bool,
    /// The cap of `M_sess` (`qwen36.prefill_chunk` on the CUDA arm). It stays beside the
    /// admitted size: the run report needs both (CPF-8a §E.2).
    pub(super) m_sess_cap: u32,
    /// The admitted `M_sess` of the prefill arena (`PrefillArena::m_sess`); `None` until
    /// `admit_prefill` admits one, so the gate answers `NoArena` (CPF-7b-C B.2).
    pub(super) arena: Option<usize>,
}

/// The chunk gate itself, in the order of plan §C.4: Native first, then Fp16 KV, then the
/// tier flag, then the Q6_K scan, then the admitted arena. The first refusal wins, so a
/// session that fails more than one condition reports the earliest one; `NoArena` is
/// checked after the Q6_K scan, and the accept verdict carries the ADMITTED size.
pub(super) fn chunk_verdict(facts: &ChunkFacts) -> ChunkVerdict {
    if facts.layout != WeightLayout::Native {
        return ChunkVerdict::TiledLayout;
    }
    if facts.kv_mode != KvMode::Fp16 {
        return ChunkVerdict::Kv8;
    }
    if facts.tiered {
        return ChunkVerdict::Tiered;
    }
    if facts.mmq_q6k {
        return ChunkVerdict::Q6KWeight;
    }
    let Some(m_sess) = facts.arena else {
        return ChunkVerdict::NoArena;
    };
    ChunkVerdict::Ready { m_sess }
}

/// The MoE roles whose weights the chunk's MMQ launches read beside every dense projection of
/// `MixerGeometry`: the routed expert stacks and the shared expert. The router matrices are
/// raw f32 (not an MMQ input), the embedding is a Q4_K gather and the head runs in the decode
/// step only (plan §A.4, §C.1), so none of the three is scanned.
pub(super) const CHUNK_MOE_ROLES: [Role; 6] = [
    Role::FfnGateExps,
    Role::FfnUpExps,
    Role::FfnDownExps,
    Role::FfnGateShexp,
    Role::FfnUpShexp,
    Role::FfnDownShexp,
];

/// True when any MMQ weight of the chunk is Q6_K (decision 0049, A6 item 4): every dense
/// projection of every layer plus the roles of `CHUNK_MOE_ROLES`. `quant_of` answers the
/// quant type of the loaded Native tensor that the layer table names; `None` means the
/// tensor was not loaded — legitimate only for the six MoE roles under a tiered load (the
/// stacks stay in the file, and the `Tiered` verdict refuses such a session anyway), so an
/// absent dense projection is a `Checkpoint` refusal naming the tensor: dense projections
/// load in every mode (review B-3, review MINOR-2). Tiled loads carry no quant type and the
/// layout refusal covers them.
pub(super) fn any_chunk_weight_q6k(
    table: &LayerTable,
    geometry: &MixerGeometry,
    quant_of: impl Fn(&str) -> Option<QuantType>,
) -> Result<bool, EngError> {
    for layer in 0..table.len() {
        let kind = table.slot(layer)?.kind;
        for spec in geometry.dense_specs(kind)? {
            let name = table.name(layer, spec.role)?;
            match quant_of(name) {
                Some(QuantType::Q6K) => return Ok(true),
                Some(_) => {}
                None => {
                    return Err(EngError::Checkpoint(format!(
                        "chunk gate: {name} is not loaded; dense projections load in every mode"
                    )));
                }
            }
        }
        for role in CHUNK_MOE_ROLES {
            if quant_of(table.name(layer, role)?) == Some(QuantType::Q6K) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

impl<'a> QwenResidentStep<'a> {
    /// Builds the step: stream, event, pinned word, resident state, and every weight lookup. All
    /// allocation of a session happens here.
    ///
    /// Lifetime. The step borrows the context, the resources and the weights, and `spawn_engine`
    /// needs an executor without a lifetime. Stage 1 takes the pattern of the Flash-Next
    /// executor (`flash_next/resident/executor.rs`): the engine thread leaks what the step
    /// borrows (`Box::leak` of the projector, the loaded weights and the resources) inside the
    /// `make` closure. Its documented limit applies unchanged: every engine open leaks one handle,
    /// so every reload leaks one more. An owning `Arc` of the three is the clean fix; the types
    /// allow it (nothing here needs a borrow longer than the step), and it is a later packet.
    ///
    /// The pin slice of the graph is built here too, beside that limit: one leaked slice of
    /// the weight buffers and the arena slot per new step (per session in stage 1), stored for
    /// the whole life of the step and of any graph it captures.
    pub(crate) fn new(
        context: &'a CudaContext,
        res: &'a StepResources,
        weights: &'a CudaWeights,
        options: &StepOptions,
        rules: QwenRules,
    ) -> Result<Self, EngError> {
        let widths = validate_step_inputs(options, res.dims.shape.vocab, &rules)?;
        // The mode follows the loaded weights (MATVEC-PAR S3b): the options carry the same
        // layout, derived from `CudaWeights::layout()` by every production caller, and a
        // mismatch is a typed error before any context work.
        agree_layout(weights.layout()?, options.layout)?;
        let sync_each = sync_each_from_env()?;
        let stream = context.stream()?;
        let state = ResidentState::allocate(context, &stream, &res.dims, options)?;
        let (event, word) = (context.event()?, context.pinned_word()?);
        let layers: Vec<usize> = (0..res.table.len()).collect();
        let parts = MixerParts {
            geometry: res.geometry,
            table: &res.table,
            weights: MixerWeights::new(
                &res.geometry,
                &res.table,
                weights,
                &layers,
                options.layout,
            )?,
            views: MixerViews::new(
                &res.plan,
                &res.arena,
                &res.geometry,
                options.layout,
                options.kv_mode,
            )?,
            layout: options.layout,
            kv_mode: options.kv_mode,
        };
        let specs = MoeSpecs::new(&res.dims)?;
        let tiered_mode = options.tiered.is_some();
        let (moe, tiered_parts) = if tiered_mode {
            let parts = layers
                .iter()
                .map(|&layer| moe_layer_parts(&res.table, weights, &specs, layer, options.layout))
                .collect::<Result<Vec<_>, _>>()?;
            (Vec::new(), Some(parts))
        } else {
            let moe = layers
                .iter()
                .map(|&layer| moe_layer_weights(&res.table, weights, &specs, layer, options.layout))
                .collect::<Result<Vec<_>, _>>()?;
            (moe, None)
        };
        let vocab = res.dims.shape.vocab;
        let mut pinned: Vec<&'a DeviceBuffer> = weights.device_buffers().collect();
        pinned.push(res.arena.buffer());
        let pins: &'a [&'a DeviceBuffer] = Box::leak(pinned.into_boxed_slice());
        // The chunk gate's Q6_K scan (decision 0049, A6 item 4), beside the other build-time
        // facts: only Native weights carry a quant type; a Tiled session is refused by its
        // layout before the scan matters. The lookup answers `None` for a tensor the load
        // left in the file (the tiered stacks), which the scan takes as not Q6_K.
        let chunk_q6k = options.layout == WeightLayout::Native
            && any_chunk_weight_q6k(&res.table, &res.geometry, |name| match weights.get(name) {
                Some(CudaWeight::Native { weight, .. }) => Some(weight.quant()),
                _ => None,
            })?;
        let step = Self {
            pins,
            graph: None,
            context,
            res,
            mixers: Mixers::new(context, parts)?,
            moe,
            tiered_mode,
            tiered_parts,
            tiered: None,
            tiered_report: TokenReport::default(),
            globals: GlobalWeights::new(
                &res.names,
                &GlobalSpecs::new(&res.dims),
                weights,
                options.layout,
            )?,
            layout: options.layout,
            moe_dims: moe_dims(&res.dims),
            prefill: None,
            state,
            word,
            event,
            stream,
            sync_each,
            started: false,
            warmed: false,
            capture_on: false,
            refused: false,
            rules,
            widths,
            chunk_q6k,
            m_sess_cap: DEFAULT_CUDA_PREFILL_CHUNK,
            logits_bytes: vec![0; f32_bytes(vocab)?],
            report: RunReport::default(),
            prefill_report: PrefillReport::default(),
            prefix: PrefixSlot::default(),
        };
        let mut step = step;
        step.refill_prefill_report();
        Ok(step)
    }

    /// Sets the capture settings; refuses after the first token and refuses
    /// `INFERENCE_CUDA_SYNC_EACH=1` together with a capture (`check_mode`'s first caller).
    pub(crate) fn set_capture_config(&mut self, config: CaptureConfig) -> Result<(), EngError> {
        check_mode(self.sync_each, config.capture)?;
        tiered_capture_refusal(self.tiered_mode, config.capture)?;
        if self.started {
            return Err(EngError::BadRequest(
                "step: the capture mode is fixed once the session runs".to_owned(),
            ));
        }
        self.capture_on = config.capture;
        Ok(())
    }

    /// Sets the cap of `M_sess` (decision 0049, A6 item 2: one of 128, 256, 512, 1,024 and
    /// 2,048 once the config is wired). The serve start calls this once at session build,
    /// beside `set_capture_config`; until lane M's CPF-7a integration commit accepts
    /// `qwen36.prefill_chunk` (lane C, 2026-10-14: the CUDA arm ignores the key), the call
    /// passes the no-cap default, and `chunk_ready` carries the cap only in its accept
    /// verdict.
    pub(crate) fn set_m_sess_cap(&mut self, m_sess_cap: u32) {
        self.m_sess_cap = m_sess_cap;
    }

    /// Admits the session's prefill arena (CPF-7b-C B.1): the two guards first (before the
    /// first token, at most once — `admit_guard`), then the pre-arena verdict — a session
    /// whose layout, KV mode, tier flag or Q6_K scan refuses the chunk path gets its
    /// refusal verdict back with NOTHING allocated and no device fact even queried — then
    /// the MMQ device facts, the ladder costs and [`admit_m_sess`] with the REAL state
    /// term. An admitted `M` allocates the [`PrefillArena`] (the whole-arena zero fill of
    /// B.3 runs here, ONCE per session); `None` allocates nothing. Both outcomes return
    /// `chunk_ready()`: `Ready` with the admitted `M_sess`, or `NoArena`.
    ///
    /// A `MemoryPlan` from `PrefillArena::allocate` after a passing admission fails the
    /// start: the budget snapshot is stale (another process took VRAM between the load and
    /// here), and failing loud matches the free-after-load floor of `make`. The partly
    /// built arena is dropped before the error returns — `allocate` fills the arena before
    /// it takes `part`, and nothing of it survives a refusal.
    ///
    /// The tiered attach of `make` never meets an arena: `needs_arena` is false for a
    /// tiered session, so the refusal returns before any allocation.
    pub(crate) fn admit_prefill(
        &mut self,
        budget: PrefillBudget,
    ) -> Result<ChunkVerdict, EngError> {
        admit_guard(self.started, self.prefill.is_some())?;
        let layout = self.state.layout();
        if !needs_arena(
            self.layout,
            layout.kv.mode,
            self.tiered_mode,
            self.chunk_q6k,
        ) {
            return Ok(self.chunk_ready());
        }
        let admit = AdmitFacts {
            budget: budget.device_free,
            decode_arena: self.res.arena.bytes(),
            state_planned: state_bytes(
                &self.res.dims,
                layout.kv.mode,
                layout.kv.capacity,
                layout.delta.slots,
            )?,
            state_allocated: layout.bytes()?,
            cap: self.m_sess_cap as usize,
        };
        let facts = self.context.mmq_device_facts()?;
        let costs = chunk_costs(&self.res.dims, facts.nsm)?;
        let Some(m) = admit_m_sess(&admit, &budget.extras, &costs)? else {
            return Ok(self.chunk_ready());
        };
        self.prefill = Some(PrefillArena::allocate(
            self.context,
            &self.stream,
            &self.res.dims,
            m,
            facts,
            budget.device_free,
        )?);
        // The report static fields come from the arena (CPF-7b-C B.8): refill them now,
        // because the arena did not exist when the step was built. The session has no
        // steps yet, so the refill cannot drop a record.
        self.refill_prefill_report();
        Ok(self.chunk_ready())
    }

    /// The chunk-gate verdict of this session (plan §C.4): whether `feed_rows` may run the
    /// chunk path. The conditions, in order: the Native layout (A4 item 2), `Fp16` KV, a
    /// non-tiered session, no Q6_K MMQ weight of the chunk (A6 item 4), and an ADMITTED
    /// prefill arena (CPF-7b-C B.2) — until `admit_prefill` admits one, an otherwise-ready
    /// session answers `NoArena` and keeps the per-token path. The accept verdict carries
    /// the admitted `M_sess` of the arena, never the raw cap.
    pub(super) fn chunk_ready(&self) -> ChunkVerdict {
        chunk_verdict(&ChunkFacts {
            layout: self.layout,
            kv_mode: self.state.layout().kv.mode,
            tiered: self.tiered_mode,
            mmq_q6k: self.chunk_q6k,
            m_sess_cap: self.m_sess_cap,
            arena: self.prefill.as_ref().map(PrefillArena::m_sess),
        })
    }

    /// The report of the last SUCCESSFUL token: after a failed token it still holds the previous
    /// token's numbers.
    pub(crate) fn last_report(&self) -> RunReport {
        self.report
    }

    /// The resident state of the step, for the device tests' direct writes and reads (the
    /// poison NaN of DV-B3 through `delta_rec`, the stale word of DV-B4 through `word`):
    /// the production surface keeps the state private, so the rows reach it through this
    /// test-only borrow. Test builds only.
    #[cfg(test)]
    pub(super) fn state_for_tests(&self) -> &ResidentState {
        &self.state
    }

    /// The step own stream, for the device tests direct writes (DV-B4 stale word). Test
    /// builds only.
    #[cfg(test)]
    pub(super) fn stream_for_tests(&self) -> &DeviceStream {
        &self.stream
    }

    /// Builds the tiered runtime of a step made with `StepOptions.tiered`: one bank per layer
    /// and the host. Refuses a plain step, a second attach and a bank count other than the layer
    /// count. The scratch buffers are allocated here.
    pub(crate) fn attach_tiered(
        &mut self,
        banks: Vec<ExpertBank>,
        host: TieredHost,
    ) -> Result<(), EngError> {
        let parts = self.tiered_parts.take().ok_or_else(|| {
            EngError::BadRequest(
                "step: no tiered weights to attach to (not a tiered step, or already attached)"
                    .to_owned(),
            )
        })?;
        let moe = &self.moe_dims;
        let scratch = TieredScratch::new(self.context, moe.hidden, moe.top_k)?;
        self.tiered = Some(std::cell::RefCell::new(TieredRuntime::new(
            parts, banks, scratch, host,
        )?));
        Ok(())
    }

    /// Fills the tiered banks before the first token; returns the number of fills. Refuses a
    /// step without an attached tiered runtime.
    pub(crate) fn preload_tiered(&mut self) -> Result<usize, EngError> {
        let runtime = self.tiered.as_ref().ok_or_else(|| {
            EngError::BadRequest("step: the tiered runtime is not attached".to_owned())
        })?;
        runtime.borrow_mut().preload(&self.stream)
    }

    /// The counters of the last token's tiered boundary (all zero on a plain step).
    pub(crate) fn last_tiered_report(&self) -> TokenReport {
        self.tiered_report
    }

    /// KV positions the session holds.
    pub(crate) fn context_limit(&self) -> usize {
        self.state.layout().kv.capacity
    }

    /// Committed rows.
    pub(crate) fn tokens(&self) -> usize {
        self.state.tokens()
    }

    /// Feeds prompt rows through the chunk plan (CPF-7b-C C4a): a multi-row step runs as one
    /// chunk, a one-row step through the per-token feed. The chunk protocol records its own
    /// step; this closure records only the feed steps. start_rows is taken at the first
    /// chunked call of a reset or import epoch.
    pub(super) fn feed_rows_chunked(
        &mut self,
        tokens: &[u32],
        m_sess: usize,
    ) -> Result<(), EngError> {
        use super::prefill_driver::drive_chunks;
        use super::prefill_plan::PrefillStep;
        use crate::family::session::TokenStepper;

        let start = self.tokens();
        self.prefill_report.note_chunked_epoch_start(start);
        drive_chunks(start, tokens, m_sess, |step, rows| match step {
            PrefillStep::Chunk { p0, .. } => self.run_chunk(p0, rows),
            PrefillStep::Feed { p0 } => {
                let [token] = rows else {
                    return Err(EngError::BadRequest(
                        "prefill chunk: a feed step needs exactly one row".to_owned(),
                    ));
                };
                let opened = std::time::Instant::now();
                TokenStepper::feed(self, *token)?;
                let ns = u64::try_from(opened.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.record_feed(p0, ns);
                Ok(())
            }
        })
        .map(|_| ())
    }

    /// Records one one-row feed step of the prefill report.
    pub(super) fn record_feed(&mut self, p0: usize, ns: u64) {
        self.prefill_report.push_step(StepRecord {
            p0,
            rows: 1,
            kind: StepKind::Feed,
            ns,
            status: 0,
        });
    }

    /// Runs ONE prefill chunk of rows at p0 (CPF-7b-C B.4, decision 0049): the open, clear,
    /// body, status-read, record and publish protocol. The one host wait of a chunk is the status
    /// read (protection 2). Every error after the chunk opens poisons the state.
    pub(super) fn run_chunk(&mut self, p0: usize, rows: &[u32]) -> Result<(), EngError> {
        use super::moe::Launch;
        use super::prefill::{
            ChunkBufs, ChunkModel, ChunkRows, ChunkRun, ChunkViews, ChunkWeights, run_chunk_body,
        };
        use super::prefill_driver::{check_chunk_start, chunk_ids};
        use super::prefill_plan::ChunkSlot;

        let Self {
            state,
            prefill,
            mixers,
            moe,
            globals,
            context,
            stream,
            res,
            widths,
            prefill_report,
            started,
            ..
        } = self;
        // 1. Host checks. Nothing touches the device before all of them pass.
        check_chunk_start(p0, state.tokens())?;
        let arena = prefill
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("prefill chunk: no prefill arena".to_owned()))?;
        let m = rows.len();
        let chunk_rows = ChunkRows::new(p0, m, state.layout().kv.capacity, arena.m_sess())?;
        let ids = chunk_ids(rows, widths.verdict)?;
        let embed = match globals.embed {
            super::resources::GlobalWeight::Native(weight) => weight,
            super::resources::GlobalWeight::Tiled(_) => {
                return Err(EngError::BadRequest(
                    "prefill chunk: the embedding is not native".to_owned(),
                ));
            }
        };
        // 2. The chunk parts, still before any device state change: a refusal leaves the state
        //    untouched.
        let views = ChunkViews::new(arena.list(), arena.arena())?;
        let ids_range = views.get(ChunkSlot::IdsTok)?;
        let bufs = ChunkBufs::new(
            views,
            arena.part().range(0, arena.part_bytes())?,
            arena.facts(),
            arena.m_sess(),
        )?;
        let model = ChunkModel::new(
            &res.table,
            &res.geometry,
            &res.dims,
            ChunkWeights::new(mixers.weights(), moe, embed),
        )?;
        // 3. From here every error poisons.
        *started = true;
        let opened = std::time::Instant::now();
        let body = || -> Result<(), EngError> {
            state.begin_prefill()?;
            state.clear_prefill_status(stream)?;
            let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            ids_range
                .buffer()
                .write_at(stream, ids_range.offset(), &bytes)?;
            let run = ChunkRun::new(&*state, &model, &bufs, chunk_rows)?;
            run_chunk_body(
                &Launch {
                    ctx: context,
                    stream,
                },
                &run,
            )
        };
        body().inspect_err(|_| state.poison())?;
        let status = state
            .read_prefill_status(stream)
            .inspect_err(|_| state.poison())?;
        let ns = u64::try_from(opened.elapsed().as_nanos()).unwrap_or(u64::MAX);
        prefill_report.push_step(StepRecord {
            p0,
            rows: m,
            kind: StepKind::Chunk,
            ns,
            status,
        });
        state
            .revalidate(stream, p0 + m, status)
            .inspect_err(|_| state.poison())?;
        Ok(())
    }

    /// Whether the step runs the tiered MoE path: a tiered step never saves a prefix.
    pub(super) fn is_tiered(&self) -> bool {
        self.tiered_mode
    }

    /// The prefill report of this session (CPF-7b-C B.8).
    pub(crate) fn last_prefill_report(&self) -> &PrefillReport {
        &self.prefill_report
    }

    /// Records one feed_rows call and the verdict it routed on.
    pub(super) fn note_feed_rows(&mut self, verdict: &ChunkVerdict, rows: usize) {
        self.prefill_report.note_feed_rows(verdict, rows);
    }

    /// Takes the once-per-job prefill line.
    pub(super) fn take_job_line(&mut self) -> Option<String> {
        self.prefill_report.take_job_line()
    }

    /// Records one ExecutorOps::prefill call with its host wall time.
    pub(super) fn record_prefill_piece(&mut self, tokens: usize, began: std::time::Instant) {
        let ns = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.prefill_report.record_piece(tokens, ns);
    }

    /// The static fields of the prefill report from the session as it stands: the cap, the
    /// arena and its part buffer (absent before the admission), the budget, the cache mode, the
    /// layout and the capture setting.
    fn refill_prefill_report(&mut self) {
        let (arena_bytes, part_bytes, budget_bytes) = match self.prefill.as_ref() {
            Some(arena) => (arena.bytes(), arena.part_bytes(), arena.budget()),
            None => (0, 0, 0),
        };
        self.prefill_report.refill(
            usize::try_from(self.m_sess_cap).unwrap_or(usize::MAX),
            arena_bytes,
            part_bytes,
            budget_bytes,
            self.state.layout().kv.mode,
            self.layout,
            self.capture_on,
        );
    }

    /// The state layout, for the entry-byte charge of a prefix save.
    pub(super) fn state_layout(&self) -> &StateLayout {
        self.state.layout()
    }

    /// Zeroes the session state (see `ResidentState::reset`) and leaves it `Valid`; only a device
    /// write that failed in the middle of the reset leaves it `Poisoned`.
    pub(crate) fn reset_state(&mut self) -> Result<(), EngError> {
        self.state.reset(&self.stream)?;
        self.refill_prefill_report();
        Ok(())
    }

    /// The committed state as a host value (checkpoint store, decision 0036).
    pub(crate) fn export_state(&self) -> Result<ExportedState, EngError> {
        self.state.export_state(&self.stream)
    }

    /// Replaces the committed state.
    pub(crate) fn import_state(&mut self, exported: &ExportedState) -> Result<(), EngError> {
        self.state.import_state(&self.stream, exported)?;
        self.refill_prefill_report();
        Ok(())
    }

    fn view(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        let res: &'a StepResources = self.res;
        let index = res
            .plan
            .index(slot)
            .ok_or_else(|| EngError::BadRequest(format!("step: the plan has no slot {slot:?}")))?;
        res.arena.tensor(index)
    }

    /// The buffers of the MoE block of the layers whose residual input is buffer `parity` (0 is
    /// `ResidA`, 1 is `ResidB`).
    fn moe_buffers(&self, parity: usize) -> Result<MoeBuffers<'_>, EngError> {
        let view = |slot| self.view(slot);
        Ok(MoeBuffers {
            route: RouteBufs {
                resid_in: view(resid_in(parity))?,
                resid_out: view(resid_out(parity))?,
                norm_row: view(Slot::NormRow)?,
                logits: view(Slot::RouterLogits)?,
                ids: view(Slot::TopkId)?,
                weights: view(Slot::TopkWeight)?,
                shared_gate: view(Slot::SharedGate)?,
                status: self.state.step_status()?,
            },
            expert: ExpertBufs {
                // The packs follow the plan: the fp16 packs of the Tiled plan or the padded
                // q8_1 packs of the Native plan (MATVEC-PAR S3b).
                packs: if self.res.plan.index(Slot::Q8Moe).is_some() {
                    Packs::Q8 {
                        moe: view(Slot::Q8Moe)?,
                        down: view(Slot::Q8Down)?,
                        shared: view(Slot::Q8Shared)?,
                    }
                } else {
                    Packs::Xf {
                        moe: view(Slot::XfMoe)?,
                        down: view(Slot::XfDown(0))?,
                        shared: view(Slot::XfShared)?,
                    }
                },
                expert_gu: view(Slot::ExpertGu)?,
                expert_down: view(Slot::ExpertDown)?,
                shared_gu: view(Slot::SharedGu)?,
                shared_out: view(Slot::SharedOut)?,
            },
        })
    }

    /// Runs one token: feeds `token` at the next position, optionally reads the logits row back,
    /// and returns the token the device argmax chose. The row is in [`Self::row_bytes`] afterwards
    /// when `want_logits` was true. A `token` outside the sampler vocabulary (a padding row of
    /// the padded head) is refused before any device write, so the state stays idle.
    pub(crate) fn run_token(&mut self, token: u32, want_logits: bool) -> Result<u32, EngError> {
        let ran = self.run_token_inner(token, want_logits);
        if ran.is_err() {
            self.abort_tiered_token();
        }
        ran
    }

    /// Drops what a failed token left in the tiered host (a layer that never reached its
    /// `finish_layer` would refuse every later token). A plain step has no runtime: nothing to do.
    fn abort_tiered_token(&self) {
        if let Some(runtime) = self.tiered.as_ref() {
            runtime.borrow_mut().abort_token();
        }
    }

    /// The host waits the tiered runtime made in the token that just ended (none on a plain step).
    fn take_tiered_syncs(&self) -> HostSyncs {
        self.tiered
            .as_ref()
            .map_or_else(HostSyncs::default, |runtime| {
                runtime.borrow_mut().take_syncs()
            })
    }

    /// The token of [`Self::run_token`], whose failure the caller cleans up after.
    fn run_token_inner(&mut self, token: u32, want_logits: bool) -> Result<u32, EngError> {
        check_token_id(token, self.widths.verdict)?;
        if self.tiered_mode && self.tiered.is_none() {
            return Err(EngError::BadRequest(
                "step: a tiered step needs attach_tiered before its first token".to_owned(),
            ));
        }
        let plan = self.readback_plan(want_logits)?;
        // The start block is written at a committed boundary (the state is idle), then the step
        // is admitted; a refused write queues nothing.
        self.started = true;
        self.write_start(token)?;
        let admitted = self.state.enqueue_rows(STEP_ROWS)?;
        let mut entered = 0;
        let mode = next_mode(
            self.graph.is_some(),
            self.warmed,
            self.capture_on,
            self.refused,
        );
        let ran = match mode {
            Mode::Replay => self.replay().map(|()| HowRan::Captured),
            Mode::Capture => self.capture_once(admitted.admit_pos, &mut entered),
            Mode::Eager => self
                .run_body(admitted.admit_pos, &mut entered)
                .map(|()| HowRan::Eager),
        };
        let how = match ran {
            Ok(how) => how,
            Err(error) => return Err(self.fail(entered, error)),
        };
        let pair = self.read_back(&plan).inspect_err(|_| self.state.poison())?;
        // The post-read-back decision is `decide`'s: a flagged row keeps a filler token and
        // hands its exact status to `commit_step`, a clean row commits and yields its token,
        // and a clean row with a token outside the sampler vocabulary poisons.
        let token = match decide(pair, self.widths.verdict) {
            Decision::BadToken(token) => {
                self.state.poison();
                return Err(bad_token_error(token));
            }
            Decision::Commit { token, status } => {
                // A non-zero status is refused here and poisons the state; the error has the
                // class of the wrapper (4, 8, 1, 2 in that order).
                self.state.commit_step(STEP_ROWS, status)?;
                token
            }
        };
        if matches!(how, HowRan::Eager) {
            self.warmed = true;
        }
        if self.tiered_mode {
            self.end_tiered_token()?;
        }
        let report = match reported_outcome(how, self.refused) {
            CaptureOutcome::Eager => eager_report(&plan),
            CaptureOutcome::Captured => captured_report(&plan),
            CaptureOutcome::RefusedFellBackToEager => refused_report(&plan),
        };
        self.report = with_host_syncs(report, self.take_tiered_syncs());
        Ok(token)
    }

    /// The token boundary of the tiered mode, after the token committed: policy P, the bank
    /// fills, the counters. A failure poisons the state: the bank and the policy may disagree.
    fn end_tiered_token(&mut self) -> Result<(), EngError> {
        let ended = match self.tiered.as_ref() {
            Some(runtime) => runtime.borrow_mut().end_token(&self.stream),
            None => Err(EngError::BadRequest(
                "step: the tiered runtime is not attached".to_owned(),
            )),
        };
        match ended {
            Ok(report) => {
                self.tiered_report = report;
                Ok(())
            }
            Err(error) => {
                self.state.poison();
                Err(error)
            }
        }
    }

    /// The read-back plan of one token. The row buffer is head-wide (the `lm_head` writes the
    /// padded head), so the plan takes the head width, not the sampler width of the verdict.
    fn readback_plan(&self, want_logits: bool) -> Result<ReadbackPlan, EngError> {
        ReadbackPlan::new(self.res.dims.shape.vocab, want_logits)
    }

    /// The start block of the step: the token, at the position of the committed rows. A token
    /// outside the sampler vocabulary is refused before `write_step_params`, so nothing is
    /// written and the state is not poisoned by the refusal.
    fn write_start(&mut self, token: u32) -> Result<(), EngError> {
        check_token_id(token, self.widths.verdict)?;
        let too_big = |what| EngError::BadRequest(format!("step: the {what} does not fit an i32"));
        let token = i32::try_from(token).map_err(|_| too_big("token"))?;
        let pos = i32::try_from(self.state.tokens()).map_err(|_| too_big("position"))?;
        self.state
            .write_step_params(&self.stream, &StepParams::new(token, pos)?)
    }

    /// Applies the failure rule of `verdict.rs`: take the step back, or poison.
    fn fail(&mut self, entered: usize, error: EngError) -> EngError {
        match action_for(entered) {
            FailureAction::Cancel => match self.state.cancel_enqueue() {
                Ok(()) => error,
                Err(cancel) => cancel,
            },
            FailureAction::Poison => {
                self.state.poison();
                error
            }
        }
    }

    /// The body of a token. CONTRACT: it only ENQUEUES launches of `step_ops::token_ops` on the
    /// owned stream, through the launch-only wrappers. It makes no device allocation, copies
    /// nothing to or from the host, reads no device word, records no event and waits for nothing
    /// (it does allocate on the HOST: the op list and the two sets of views are built per call,
    /// which does not touch the stream and is not captured); the single
    /// exception is the `INFERENCE_CUDA_SYNC_EACH` branch (eager debug mode, refused together with
    /// a capture by `readback::check_mode`). The start block is written before it and the
    /// read-back follows it. `entered` counts the ops that were entered, the failing one included.
    fn run_body(&self, admit_pos: usize, entered: &mut usize) -> Result<(), EngError> {
        let bufs = [self.moe_buffers(0)?, self.moe_buffers(1)?];
        let params = self.state.word(Word::StepParams)?;
        for op in token_ops(self.res.table.len()) {
            *entered += 1;
            self.run_op(op, admit_pos, &bufs, params)?;
            if self.sync_each
                && let Err(error) = self.stream.synchronize()
            {
                // The op was queued and the sync error is a device fault, not a refused
                // launch: `entered == 1` here would make `fail` treat the step as "nothing
                // ran" and take it back, so the failed sync counts as having entered the
                // next op and the failure rule poisons instead.
                *entered = entered_after_failed_sync(*entered);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Attempts the one capture of the body: the holds are a local here (the returned graph
    /// keeps them for its life; eager tokens hold nothing), the body runs inside the window
    /// through the launch-only wrappers, and a succeeded capture stores the graph and replays
    /// it in the same token (a capture records but does not run). A refused capture (the one
    /// non-terminal refusal of the device capture) discards the attempt's op count, reruns the
    /// body eagerly in the same token (which re-records any events) and sets the sticky
    /// refusal; every other error is the caller's to fail on.
    fn capture_once(&mut self, admit_pos: usize, entered: &mut usize) -> Result<HowRan, EngError> {
        let holds = vec![self.res.arena.hold(), self.context.workspace_hold()?];
        let attempt = self.context.capture(
            &self.stream,
            CapturePlan {
                pins: self.pins,
                holds,
            },
            || self.run_body(admit_pos, entered),
        );
        match fold_capture(
            attempt,
            crate::kernels::cuda::is_capture_refusal,
            entered,
            &mut self.refused,
        )? {
            Some(graph) => {
                self.graph = Some(graph);
                self.replay()?;
                Ok(HowRan::Captured)
            }
            None => {
                self.run_body(admit_pos, entered)?;
                Ok(HowRan::Refused)
            }
        }
    }

    /// Replays the stored graph on the owned stream (the graph is borrowed in its own scope).
    fn replay(&mut self) -> Result<(), EngError> {
        let Self { graph, stream, .. } = self;
        match graph {
            Some(graph) => graph.launch(stream),
            None => Err(EngError::BadRequest("step: no graph to replay".to_owned())),
        }
    }

    fn run_op(
        &self,
        op: StepOp,
        admit_pos: usize,
        bufs: &[MoeBuffers<'_>; 2],
        params: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let (ctx, s) = (self.context, &self.stream);
        match op {
            StepOp::Begin => ctx.step_begin(s, params),
            StepOp::Embed => {
                let ids = self.state.step_field(StepField::TokenId)?;
                let x = self.view(resid_in(0))?;
                let status = self.state.step_status()?;
                match self.globals.embed {
                    EmbedWeight::Tiled(weight) => ctx.embed(
                        s,
                        EmbedCall {
                            weight,
                            ids,
                            x,
                            status,
                            m: STEP_ROWS,
                        },
                    ),
                    // The native gather reads the file's own Q4_K blocks (MATVEC-PAR S3b).
                    EmbedWeight::Native(weight) => ctx.embed_native(
                        s,
                        EmbedNativeCall {
                            weight,
                            ids,
                            x,
                            status,
                            m: STEP_ROWS,
                        },
                    ),
                }
            }
            StepOp::Mixer(layer) => {
                let io = BlockIo {
                    resid: self.view(resid_in(layer))?,
                    admit_pos,
                };
                self.mixers.run_block(s, &self.state, layer, io)
            }
            StepOp::Moe(layer) if self.tiered_mode => {
                let runtime = self.tiered.as_ref().ok_or_else(|| {
                    EngError::BadRequest("step: the tiered runtime is not attached".to_owned())
                })?;
                let launch = Launch { ctx, stream: s };
                runtime
                    .borrow_mut()
                    .run_layer(&launch, layer, &bufs[layer % 2], &self.moe_dims)
            }
            StepOp::Moe(layer) => {
                let weights = self.moe.get(layer).ok_or_else(|| {
                    EngError::BadRequest(format!("step: layer {layer} has no MoE weights"))
                })?;
                let launch = Launch { ctx, stream: s };
                run_moe_block(&launch, weights, &bufs[layer % 2], &self.moe_dims)
            }
            StepOp::FinalNorm => self.final_norm(),
            StepOp::LmHead => self.lm_head(),
            StepOp::Argmax => ctx.argmax(
                s,
                ArgmaxCall {
                    n: self.widths.argmax_n,
                    logits: self.view(Slot::Logits)?,
                    part: self.view(Slot::ArgmaxPart)?,
                    params,
                    result: Some(self.state.word(Word::ResultPair)?),
                },
            ),
            StepOp::Advance => ctx.step_advance(s, params, STEP_ROWS as u32),
        }
    }

    fn final_norm(&self) -> Result<(), EngError> {
        let layers = self.res.table.len();
        let norm = self.globals.output_norm;
        let h = self.view(final_resid(layers))?;
        let weight = norm.range(0, norm.bytes())?;
        match self.layout {
            WeightLayout::Tiled => self.context.add_norm_pack(
                &self.stream,
                AddNormCall {
                    m: STEP_ROWS,
                    n: self.res.geometry.hidden,
                    eps: self.res.geometry.eps,
                    h,
                    delta: None,
                    weight,
                    xf: self.view(Slot::XfFinal)?,
                    norm_out: None,
                },
            ),
            // The fused producer must write an f32 row beside the pack: the native plan's OWN
            // FinalNormRow slot (MSG-74 item 2 / C4-F3's true reading): writing NormRow here
            // would have broken no lifetime (its native lifetime covers FINAL_NORM and live
            // slots never share bytes) - it silently repurposed the router-input value slot as
            // scratch, a latent bug the moment NormRow's lifetime tightens to its true need.
            WeightLayout::Native => self.context.add_rmsnorm_q8(
                &self.stream,
                AddNormQ8Call {
                    n: self.res.geometry.hidden,
                    eps: self.res.geometry.eps,
                    h,
                    delta: None,
                    weight,
                    norm_out: self.view(Slot::FinalNormRow)?,
                    q8: self.view(Slot::Q8Final)?,
                },
            ),
        }
    }

    fn lm_head(&self) -> Result<(), EngError> {
        // The pack slot is resolved INSIDE each arm (the S3c scratch defect): the Native plan
        // holds no XfFinal slot, so a lookup before the match refused every native run.
        let y = self.view(Slot::Logits)?;
        match self.globals.head {
            HeadWeight::Tiled(head) => self.context.matmul(
                &self.stream,
                Projection {
                    weight: head,
                    first_row: 0,
                    rows: head.n_rows(),
                },
                MatmulIo {
                    xf: self.view(Slot::XfFinal)?,
                    y,
                    m: STEP_ROWS,
                },
            ),
            HeadWeight::Native(head) => {
                let ty = head.quant();
                self.context.matvec_q(
                    &self.stream,
                    MatvecCall {
                        ty,
                        w: head.buffer().range(0, head.buffer().bytes())?,
                        x_q8: self.view(Slot::Q8Final)?,
                        y,
                        k: head.k(),
                        rows: head.n_rows(),
                        ncols: STEP_ROWS,
                    },
                )
            }
        }
    }

    /// The defined read-back point: queue the pair copy and its event, read the logits row when
    /// wanted (this call waits for the stream), then read the pair (its event is complete).
    fn read_back(&mut self, plan: &ReadbackPlan) -> Result<ResultPair, EngError> {
        let pair_range = self.state.word(Word::ResultPair)?;
        self.context
            .copy_word_to_host(&self.stream, pair_range, &self.word)?;
        self.stream.record(&self.event)?;
        if let Some(bytes) = plan.logits_bytes {
            let row = self.view(Slot::Logits)?;
            let out = &mut self.logits_bytes[..bytes];
            row.buffer().read_at(&self.stream, row.offset(), out)?;
        }
        let value = self.word.read_after(&self.event)?;
        Ok(ResultPair {
            token: value.token,
            status: StepStatus(value.status),
        })
    }

    /// The last logits row as values (a new vector). Needs a run with `want_logits`; otherwise
    /// it holds the bytes of an earlier run.
    pub(super) fn logits_values(&self) -> Vec<f32> {
        let (values, _) = self.logits_bytes.as_chunks::<F32>();
        values
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect()
    }

    /// The last logits row as little-endian f32 bytes, for hashing without a conversion.
    pub(crate) fn row_bytes(&self) -> &[u8] {
        &self.logits_bytes
    }
}

impl Drop for QwenResidentStep<'_> {
    fn drop(&mut self) {
        // Work may still be queued behind the last token. A failed synchronize latches the
        // backend (decision 0022), and every owner below checks the latch before it frees, so
        // the error needs no other handling here and `drop` must not panic.
        let _synced = self.stream.synchronize();
    }
}
