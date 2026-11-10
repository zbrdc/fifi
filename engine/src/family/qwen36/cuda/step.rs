//! `QwenResidentStep`: the resident Qwen3.6 decode step on the CUDA backend. One token is the
//! launch list of `step_ops.rs` queued on ONE owned stream: `step_begin`, the embedding, the 40
//! layers (the mixer block, then the MoE block), the final norm, the `lm_head` projection,
//! `argmax`, `step_advance`. Then the host reads the result pair (and, when asked, the 993 KB
//! logits row) with ONE blocking wait.
//!
//! # Structure of a token
//!
//! 1. a host start-block write (`write_start`: one blocking host-to-device copy);
//! 2. the private `run_body`, which only enqueues launches and the section marks' event
//!    records (its doc states the contract);
//! 3. the read-back, the event and ONE blocking wait (`read_back`).
//!
//! The body of a token runs in one of three modes (the pure `next_mode` rule below): eager (the
//! warm-up token, every fallback and every plain session), captured (one capture, then
//! replays), or eager-after-refusal. A refused capture is STICKY for the whole step
//! construction: later tokens stay eager and keep reporting `RefusedFellBackToEager`, across
//! `reset_state` and `import_state`, which are `ResidentState` methods and cannot touch the
//! step's own flags.
//!
//! # Call order and errors
//!
//! Per token: `write_step_params`, `enqueue_rows`, the body (`run_block` for each layer), then
//! `commit_step(accepted, status)` with the pair's status. A failure of the first op of the body
//! takes the step back (`cancel_enqueue`); any later failure, any error of the read-back and any
//! non-zero status bit poisons the state. Poisoned means: only `reset` or a full reset-equivalent
//! `import_state` leaves it.
//!
//! One row only: `m = 1` everywhere (the MoE block takes one row).
//!
//! Drop order: the captured graph first (it keeps the holds of its capture and covers the state
//! buffers by drop order), then the borrowed weights and views, the state, the pinned word, the
//! event and the stream; `Drop` synchronizes the stream first.

use std::cell::RefCell;

use super::{
    dims::{KvMode, StepOptions},
    exl3_bank::Exl3SlotResolver,
    exl3_moe::{
        Exl3MoeBufs, Exl3MoeWeights, Scratch,
        merge::{SharedMerge, run_exl3_moe_merged},
        run_exl3_moe,
    },
    layer_table::LayerTable,
    mixer::{BlockIo, MixerParts, MixerViews, Mixers},
    mixer_geometry::MixerGeometry,
    mixer_weights::MixerWeights,
    moe::{
        ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, Packs, RouteBufs, StackWeights,
        run_moe_block, tiered::TieredScratch,
    },
    prefill_driver::{
        AdmitFacts, FIXED_ADMISSION_MARGIN, FULL_STAGE_TIER_ROWS, Kv8StagingPlan, LedgerFigures,
        PrefillArena, PrefillBudget, admit_fixed_staged, admit_guard, admit_with_staging,
        fixed_chunk_rows, fixed_staging_rows, ledger_line, needs_arena, stage_label,
        wide_stage_candidate,
    },
    prefill_plan::{chunk_costs, exl3_chunk_costs, state_bytes},
    prefix::{CudaPrefillPath, CudaSavedPrefix, PendingPrefix},
    readback::{
        ReadbackPlan, SHORTLIST_VAR, SYNC_EACH_VAR, captured_report, check_mode,
        device_sampling_supported, device_shortlist_enabled, eager_report, parse_sync_each,
        refused_report, with_host_syncs,
    },
    resources::{
        EmbedWeight, GlobalSpecs, GlobalWeights, HeadWeight, MoeSpecs, STEP_ROWS, StepResources,
        check_paged_modes, f32_bytes, moe_dims, moe_layer_parts, moe_layer_weights,
    },
    rules::{QwenRules, StepWidths, check_token_id, validate_step_inputs},
    run_report::{CaptureOutcome, HostSyncs, PrefillReport, RunReport, StepKind, StepRecord},
    section_timing::{FineSplit, MarkedGroup, SectionPlan, SectionTimes, section_plan},
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
    family::project::cuda::{CudaWeight, CudaWeights, WeightLayout, exl3::Exl3Io},
    family::qwen36::exl3_merge_plan::{
        Intent, Source, load_suffix, resolve_switch, shared_in_routed_explicit,
    },
    family::qwen36::settings::{DEFAULT_CUDA_PREFILL_CHUNK, speculator_of},
    family::qwen36::tiered::{host::TieredHost, report::TokenReport},
    family::qwen36::window_logic,
    family::session::PrefixSlot,
    family::speculate::{
        config::SpecConfig,
        driver::{RepeatDrafter, Speculator},
    },
    family::{AttentionKind, gguf_model::Role, sampler::Sampler},
    kernels::cuda::{
        CapturePlan, CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, ExpertBank,
        GraphExec, MatmulIo, PinnedWord, Projection, TimingEvent,
        resident::{
            add_norm_pack::AddNormCall,
            add_rmsnorm_q8::AddNormQ8Call,
            embed::{EmbedCall, EmbedNativeCall},
            embed_bf16::EmbedBf16Call,
            matvec_q::{MatvecCall, QuantType},
            shortlist::{self, ShortlistCall},
            step_params::{ArgmaxCall, ResultPair, STATUS_EXL3_ARITHMETIC, StepParams, StepStatus},
        },
    },
};

#[cfg(feature = "research-activation")]
#[path = "step_activation.rs"]
mod activation;

const F32: usize = size_of::<f32>();

#[cfg(all(test, feature = "native-cuda"))]
#[path = "prefill_kv8_target_device_tests.rs"]
mod kv8_target;

/// The routed expert ids of one MoE block: the layer and its `top_k` ids (test builds only).
#[cfg(test)]
pub(super) type MoePicks = (usize, Vec<i32>);

/// Capture settings of the step; distinct from `dims::StepOptions` (the dimensions options of
/// `new`).
pub(crate) struct CaptureConfig {
    pub(crate) capture: bool,
    /// Whether the step records in-graph section-timing marks: a timing-event pair per section of
    /// the token body, recorded with the external-record flag so a capture turns them into
    /// event-record nodes that every replay re-fires. Off by default; the default path queues
    /// nothing extra.
    pub(crate) section_marks: bool,
}

/// The timing events of the section marks, created by `set_capture_config` before the first token
/// so their addresses are stable across the capture and every replay. Pairs: the whole token, the
/// embedding, one per marked run of identical layers (`section_timing::section_plan`), and the
/// tail; the tail's end is the whole token's end, so the last two share one event. The section
/// algebra and the DeepInfer citation live in `section_timing`.
struct SectionMarks {
    plan: SectionPlan,
    whole_start: TimingEvent,
    embed_start: TimingEvent,
    embed_end: TimingEvent,
    groups: Vec<(TimingEvent, TimingEvent)>,
    /// The finer split (`FineSplit`): one event per marked group, recorded at the `Moe` start
    /// (= the mixer half's end), the `LmHead` pair, and the attention-core pair inside the
    /// mixer half (the `Conv`/`AttnPrep` start to the `Delta`/`AttnDecode` end).
    mixer_end: Vec<TimingEvent>,
    core: Vec<(TimingEvent, TimingEvent)>,
    head_start: TimingEvent,
    head_end: TimingEvent,
    tail_start: TimingEvent,
    end: TimingEvent,
    /// Whether the last token's marks were all recorded and its read-back completed: the
    /// elapsed queries are legal only then (a query on unfinished or never-recorded events
    /// fails with 600/400 and latches the backend terminal), so `section_times` refuses by
    /// this host flag before any native call. Every write happens under `&mut self`.
    readable: bool,
}

impl SectionMarks {
    fn new(context: &CudaContext, kinds: &[AttentionKind]) -> Result<Self, EngError> {
        let plan = section_plan(kinds);
        let mut groups = Vec::with_capacity(plan.groups.len());
        let mut mixer_end = Vec::with_capacity(plan.groups.len());
        let mut core = Vec::with_capacity(plan.groups.len());
        for _ in &plan.groups {
            groups.push((context.event_timing()?, context.event_timing()?));
            mixer_end.push(context.event_timing()?);
            core.push((context.event_timing()?, context.event_timing()?));
        }
        Ok(Self {
            whole_start: context.event_timing()?,
            embed_start: context.event_timing()?,
            embed_end: context.event_timing()?,
            groups,
            mixer_end,
            core,
            head_start: context.event_timing()?,
            head_end: context.event_timing()?,
            tail_start: context.event_timing()?,
            end: context.event_timing()?,
            plan,
            readable: false,
        })
    }

    /// Records the opening event of the section `op` starts, if it starts one. Inside a capture
    /// window the external record becomes an event-record node; outside one it records eagerly
    /// (the eager fallback of a refused capture runs the same body).
    fn record_before(&self, op: StepOp, stream: &DeviceStream) -> Result<(), EngError> {
        let event = match op {
            StepOp::Begin => &self.whole_start,
            StepOp::Embed => &self.embed_start,
            StepOp::Mixer(layer) => {
                return match self.group_index(layer) {
                    Some(at) => stream.record_timing(&self.groups[at].0),
                    None => Ok(()),
                };
            }
            StepOp::FinalNorm => &self.tail_start,
            StepOp::LmHead => &self.head_start,
            StepOp::Moe(layer) => {
                // The MoE half starts where the mixer half ends: one extra event per marked
                // group splits the pair (see `FineSplit`).
                if let Some(at) = self.group_index(layer) {
                    stream.record_timing(&self.mixer_end[at])?;
                }
                return Ok(());
            }
            StepOp::Argmax | StepOp::Advance => return Ok(()),
        };
        stream.record_timing(event)
    }

    fn record_after(&self, op: StepOp, stream: &DeviceStream) -> Result<(), EngError> {
        let event = match op {
            StepOp::Embed => &self.embed_end,
            StepOp::Moe(layer) => {
                return match self.group_index(layer) {
                    Some(at) => stream.record_timing(&self.groups[at].1),
                    None => Ok(()),
                };
            }
            StepOp::Advance => &self.end,
            StepOp::LmHead => return stream.record_timing(&self.head_end),
            StepOp::Begin | StepOp::Mixer(_) | StepOp::FinalNorm | StepOp::Argmax => return Ok(()),
        };
        stream.record_timing(event)
    }

    fn group_index(&self, layer: usize) -> Option<usize> {
        self.plan
            .groups
            .iter()
            .position(|group| group.layer == layer)
    }
}

pub(super) enum Mode {
    Replay,
    Capture,
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

#[derive(Debug, Clone, Copy)]
pub(super) enum HowRan {
    Eager,
    Captured,
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

/// The one typed refusal of a layout mismatch: the step runs the layout of the loaded weights
/// and the options must agree with it. Pure, so the CPU tests pin it directly.
pub(super) fn agree_layout(weights: WeightLayout, options: WeightLayout) -> Result<(), EngError> {
    if weights != options {
        return Err(EngError::Checkpoint(format!(
            "the step options run {options:?} over {weights:?} weights: the layout follows the \
             loaded model"
        )));
    }
    Ok(())
}

/// The tiered mode is eager only: it waits for the host inside every layer, which a captured graph
/// cannot do. Pure, so the CPU tests pin it.
pub(super) fn tiered_capture_refusal(tiered_mode: bool, capture: bool) -> Result<(), EngError> {
    if tiered_mode && capture {
        return Err(EngError::BadRequest(
            "step: the tiered mode is eager only: it waits for the host inside every layer, \
             which a captured graph cannot do"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The id-indexed launch holds 16 pairs (gate and up of every pick), so a captured EXL3 step
/// supports at most 8 picks per token.
pub(super) fn exl3_device_pick_refusal(
    layout: WeightLayout,
    capture: bool,
    top_k: usize,
) -> Result<(), EngError> {
    if layout == WeightLayout::Exl3 && capture && 2 * top_k > 16 {
        return Err(EngError::BadRequest(format!(
            "step: a captured EXL3 step reads at most 8 picks per token on the device, got {top_k}"
        )));
    }
    Ok(())
}

pub(super) fn exl3_flag_error() -> EngError {
    EngError::ModelOutput("EXL3 projection flagged invalid arithmetic".to_owned())
}

/// The refusal of a result pair whose status carries the EXL3 arithmetic verdict (mask 16). It is
/// taken before `decide` sees the pair, so a flagged EXL3 token never commits, whatever other
/// bits the status holds. Only an EXL3 step folds the flag into the status.
pub(super) fn exl3_pair_refusal(layout: WeightLayout, pair: ResultPair) -> Option<EngError> {
    (layout == WeightLayout::Exl3 && pair.status.0 & STATUS_EXL3_ARITHMETIC != 0)
        .then(exl3_flag_error)
}

fn sync_each_from_env() -> Result<bool, EngError> {
    match std::env::var(SYNC_EACH_VAR) {
        Ok(value) => parse_sync_each(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_sync_each(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(EngError::BadRequest(format!(
            "{SYNC_EACH_VAR} is not valid text"
        ))),
    }
}

pub(crate) struct QwenResidentStep<'a> {
    // Declaration order is drop order: the graph first, then the borrows, the stream last.
    /// The leaked pin slice of the weights and the arena slot: one leaked slice per new step (one
    /// step per session); the graph bakes their addresses in.
    pins: &'a [&'a DeviceBuffer],
    /// The pins of a K-row graph: the pins above and the K-row arena. Leaked at `enable_krow`
    /// (one slice per armed step); empty until then.
    krow_pins: &'a [&'a DeviceBuffer],
    /// The replayable body, once a capture has succeeded. Declared BEFORE the state field: the
    /// graph (which keeps the arena holds and borrows the pin slice) drops first, and that
    /// covers the state buffers by drop order -- they live inside the step and never move
    /// after they are built, so the pin list need not name them.
    graph: Option<GraphExec<'a>>,
    /// The graphs of the K-row body, one per window length (`krow_graph`). Declared right after
    /// `graph` and before `krow`, `state` and `rollback`: a graph bakes in the addresses of the
    /// K-row buffer, the state and the rollback slots, so every graph must drop first. A test
    /// pins this order (`krow_graph_tests::the_k_row_graphs_are_declared_before_the_buffers_they_bake_in`).
    krow_graphs: krow_graph::GraphTable<GraphExec<'a>>,
    /// The K-row verify window (`draft_verify = krow`), once armed. Declared after the graphs and
    /// before the state: a captured K-row body bakes in the addresses of its small buffer, so
    /// any graph of it must drop first.
    krow: Option<krow_step::KrowRuntime>,
    #[cfg(feature = "research-activation")]
    activation: std::cell::RefCell<Option<crate::family::qwen36::activation::Capture>>,
    /// The section-timing marks: `None` unless the capture config asked for them. Declared after
    /// the graph (which drops first, synchronizing the device) and before the stream: the timing
    /// events drop while the stream is still alive.
    marks: Option<SectionMarks>,
    context: &'a CudaContext,
    res: &'a StepResources,
    mixers: Mixers<'a>,
    moe: Vec<MoeLayerWeights<'a>>,
    /// The MoE weights of each layer of an EXL3 load, whose experts the grouped EXL3 launches
    /// read; `moe` is empty then.
    exl3_moe: Vec<Exl3MoeWeights<'a>>,
    /// The banked EXL3 decode state: every layer's slot tables with the host store, behind a
    /// `RefCell` because the decode runs through a shared step view. `None` is the resident
    /// step; when set, the eager MoE blocks map their read picks through the tables and
    /// refill missed slots before the launches (the engine thread is the only borrower).
    exl3_bank: Option<&'a std::cell::RefCell<Exl3SlotResolver<'a>>>,
    /// Tokens decoded since the last banked hits/misses log line (a banked step only).
    banked_echo: u32,
    /// The working buffer of the MoE block that queues the shared expert inside the routed launches
    /// (the default of an EXL3 load; `FIFI_EXL3_SHARED_IN_ROUTED=0` or a load that cannot run it
    /// leaves `None`, which keeps the reference block).
    ///
    /// Declared after `graph`, so the graph drops first: a captured body bakes in this buffer's
    /// address and its replay must never outlive the buffer.
    exl3_merge: Option<SharedMerge>,
    /// What `FIFI_EXL3_SHARED_IN_ROUTED` asked of this load, as resolved at the build
    /// (`Intent::Off` for every layout but EXL3). `request_krow_verify` reads it to tell a merged
    /// block built by the default (which yields to krow) from an explicit `1` (which krow refuses).
    merge_intent: Intent,
    /// Where the mode of the merged block came from, for the load line.
    merge_source: Source,
    /// The tiered mode: the options asked for it. The routed expert stacks are then not loaded,
    /// `moe` is empty and every layer's MoE block runs through `tiered`. Cost: each layer of each
    /// token syncs with the host: about 0.8 ms per token over the plain step when every expert is
    /// resident (3090, 256 slots, measured), and about 18 ms per step with most picks on the CPU
    /// (slots 40 with preload, measured). A prompt is prefilled one token per step, so a prompt
    /// of n tokens costs n of these steps until a batched tiered prefill exists.
    tiered_mode: bool,
    /// The router and shared weights of each layer, held until `attach_tiered` builds the runtime.
    tiered_parts: Option<Vec<LayerParts<'a>>>,
    /// The tiered runtime; interior mutability because the token body takes `&self`.
    tiered: Option<std::cell::RefCell<TieredRuntime<'a>>>,
    tiered_report: TokenReport,
    globals: GlobalWeights<'a>,
    /// The layout the step runs: the loaded weights' own layout, checked against the options at
    /// construction.
    layout: WeightLayout,
    moe_dims: MoeDims,
    /// The prefill arena of the session: admitted ONCE by `admit_prefill` after the step is built
    /// and the cap is set, and only for a session whose pre-arena verdict passes (`needs_arena`).
    /// Declared BEFORE the state: the arena drops before the state's buffers by drop order.
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
    capture_on: bool,
    /// An EXL3 step reads the router's picks on the device (no host read in the MoE block); set
    /// with the capture config, which needs a body that makes no host read.
    exl3_device_ids: bool,
    /// The sticky refusal flag: set only by a refused capture, never cleared -- `reset_state`
    /// and `import_state` are `ResidentState` methods and cannot touch it.
    refused: bool,
    pub(super) rules: QwenRules,
    /// The sampler widths the step wires in (see `rules::StepWidths`): the argmax `n` and the
    /// verdict width, both the sampler vocabulary, never the padded head width.
    widths: StepWidths,
    /// True when any MMQ weight of the chunk is Q6_K (`any_chunk_weight_q6k` at build): the
    /// chunk gate then refuses with `Q6KWeight`.
    chunk_q6k: bool,
    /// The cap of `M_sess`, set by `set_m_sess_cap`; the value is the no-cap default until a
    /// later change wires the config member `qwen36.prefill_chunk` into it.
    m_sess_cap: u32,
    kv8_chunked_prefill: bool,
    /// The window that has run and awaits its verdict (see `eager_window`).
    pub(super) window: Option<super::eager_window::EagerWindow>,
    /// A fault armed for the next window or commit: test builds only, so a serving build
    /// carries neither the field nor the check (see `eager_window::inject_fault`).
    #[cfg(test)]
    pub(super) window_fault: Option<crate::family::speculate::d1_oracle::Fault>,
    /// The host and device time of the last window, by part (test builds only).
    #[cfg(test)]
    pub(super) window_trace: super::eager_window::WindowTrace,
    /// The routed expert ids of each EXL3 MoE block, `(layer, top_k ids)` in launch order, filled
    /// only while the measurement tests switch it on (`None` is off). It reads the ids back after
    /// each block, so it is valid for the eager body only (test builds only).
    #[cfg(test)]
    pub(super) moe_pick_tap: std::cell::RefCell<Option<Vec<MoePicks>>>,
    /// The per-layer residual taps of the next eager body, when a test turned them on (test
    /// builds only; the bisect aid of the K-row checks, absent from the serving path).
    #[cfg(test)]
    taps: std::cell::RefCell<Option<Vec<krow_step::LayerTap>>>,
    /// Drain the stream after each device-rollback capture, so the copy's time shows in the trace
    /// (test builds only; the copy otherwise overlaps the host's next work).
    #[cfg(test)]
    pub(super) trace_sync_capture: bool,
    /// The speculative decode step, when the settings asked for one.
    pub(super) speculator: Option<Speculator<RepeatDrafter>>,
    #[cfg(test)]
    chunk_kernel_class: super::prefill::ChunkKernelClass,
    /// The bytes of the last logits row read back (little-endian f32).
    logits_bytes: Vec<u8>,
    shortlist_on: bool,
    /// The routing readback spike (`Q36_ROUTE_CAPTURE=1`, `route_capture`): when on, each
    /// decode step runs EAGER and reads the router picks back after every MoE block. A
    /// diagnostic measurement opt-in, not a serving feature (see `route_capture`).
    route_capture: bool,
    /// The env flag of the capture-preserving stage (Q36_ROUTE_STAGE=1): when on, the plain-MoE
    /// layers' router picks are D2D-staged inside the step's enqueued work (graph-capture legal)
    /// and read back once per token. EXL3 and tiered layouts are refused at step time.
    route_stage_on: bool,
    /// The stage itself, allocated for the plain layout when `route_stage_on` (see `RouteStage`).
    route_stage: Option<RouteStage>,
    shortlist_bytes: [u8; shortlist::RESULT_BYTES],
    report: RunReport,
    prefill_report: PrefillReport,
    /// The engine's one host-backed prefix entry.
    /// The pinned pool keeps its backing alive across entry eviction.
    pub(super) prefix: PrefixSlot<CudaSavedPrefix, CudaPrefillPath>,
    prefix_pins: Option<super::state::pinned::PrefixPinPool>,
    device_tip: Option<super::state::device_tip::DeviceTip>,
    /// `qwen36.draft_rollback = device` asked for: a verify window rolls back through `rollback`.
    /// A window on a step that wants it and holds no slots is refused before any state change.
    rollback_wanted: bool,
    /// `qwen36.draft_verify = krow` asked for (`rollback_runtime`): the merged MoE block is refused.
    krow_wanted: bool,
    /// The options asked for experts on branch streams (refused in stage 1 at resource creation,
    /// and refused again when the device rollback is armed, which orders its copies on one stream).
    forked_experts: bool,
    /// The device rollback slots (see `rollback_runtime`), armed before the first token. Armed means
    /// `Some`; the shared `Speculator` knows nothing of it.
    rollback: Option<super::state::rollback::RollbackSlots>,
    pending_prefix: Option<PendingPrefix>,
    /// The recovery gate: a failed or indeterminate window commit set it. Only `reset_state`
    /// clears it (see `window_gate`).
    needs_reset: bool,
}

/// The typed verdict of the chunk gate: whether `feed_rows` may run the chunk path for this
/// session. Every refusal keeps the per-token prompt path; the run report records the reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChunkVerdict {
    /// The session may use the chunk path. `m_sess` is the ADMITTED `M_sess`, one of
    /// `M_LADDER` (`PrefillArena::m_sess`), never the raw cap: the arena admission narrows the
    /// cap of `set_m_sess_cap` to the largest ladder size whose device sum fits the budget.
    Ready {
        /// The first reader is the chunk list of the prefill driver.
        m_sess: usize,
    },
    /// The session runs the Tiled layout: the chunk path is defined for Native sessions only.
    TiledLayout,
    /// The `Int8G32` session has no explicit chunk-prefill opt-in and keeps per-token prefill.
    Kv8,
    /// The session is tiered: `chunk_ready` sends tiered sessions to the per-token loop; no
    /// batched tiered prefill exists yet.
    Tiered,
    /// An MMQ weight of the chunk is Q6_K: no Q6_K chunk row exists, so the session keeps the
    /// per-token path.
    Q6KWeight,
    /// The session admitted no prefill arena (no `M` of the ladder fits the budget, or
    /// `admit_prefill` was not called); the session keeps the per-token path.
    NoArena,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkFacts {
    pub(super) layout: WeightLayout,
    pub(super) kv_mode: KvMode,
    pub(super) kv8_chunked_prefill: bool,
    pub(super) tiered: bool,
    pub(super) mmq_q6k: bool,
    /// The cap of `M_sess` (`qwen36.prefill_chunk` on the CUDA arm). It stays beside the
    /// admitted size: the run report needs both.
    pub(super) m_sess_cap: u32,
    /// The admitted `M_sess` of the prefill arena (`PrefillArena::m_sess`); `None` until
    /// `admit_prefill` admits one, so the gate answers `NoArena`.
    pub(super) arena: Option<usize>,
}

/// The chunk gate itself, in this order: a Native or EXL3 layout first, then Fp16 KV, then the
/// tier flag, then the Q6_K scan, then the admitted arena. The first refusal wins, so a session that fails more than
/// one condition reports the earliest one; `NoArena` is checked after the Q6_K scan, and the accept
/// verdict carries the ADMITTED size.
pub(super) fn chunk_verdict(facts: &ChunkFacts) -> ChunkVerdict {
    if facts.layout == WeightLayout::Tiled {
        return ChunkVerdict::TiledLayout;
    }
    if facts.kv_mode == KvMode::Int8G32 && !facts.kv8_chunked_prefill {
        return ChunkVerdict::Kv8;
    }
    if facts.tiered
        && !super::prefill::tiered_chunk_enabled(
            std::env::var_os(super::prefill::TIERED_CHUNK_VAR).as_deref(),
        )
    {
        return ChunkVerdict::Tiered;
    }
    if facts.mmq_q6k {
        return ChunkVerdict::Q6KWeight;
    }
    let Some(m_sess) = facts.arena else {
        return ChunkVerdict::NoArena;
    };
    let m_sess = if facts.kv_mode == KvMode::Int8G32
        && facts.kv8_chunked_prefill
        && facts.m_sess_cap == 16
    {
        m_sess.min(16)
    } else {
        m_sess
    };
    ChunkVerdict::Ready { m_sess }
}

/// The MoE roles whose weights the chunk's MMQ launches read beside every dense projection of
/// `MixerGeometry`: the routed expert stacks and the shared expert. The router matrices are raw f32
/// (not an MMQ input), the embedding is a Q4_K gather and the head runs in the decode step only, so
/// none of the three is scanned.
pub(super) const CHUNK_MOE_ROLES: [Role; 6] = [
    Role::FfnGateExps,
    Role::FfnUpExps,
    Role::FfnDownExps,
    Role::FfnGateShexp,
    Role::FfnUpShexp,
    Role::FfnDownShexp,
];

/// True when any MMQ weight of the chunk is Q6_K: every dense projection of every layer plus the
/// roles of `CHUNK_MOE_ROLES`. `quant_of` answers the quant type of the loaded Native tensor that
/// the layer table names; `None` means the tensor was not loaded — legitimate only for the six MoE
/// roles under a tiered load (the stacks stay in the file, and the `Tiered` verdict refuses such a
/// session anyway), so an absent dense projection is a `Checkpoint` refusal naming the tensor:
/// dense projections load in every mode. Tiled loads carry no quant type and the layout refusal
/// covers them.
pub(super) fn any_chunk_weight_q6k(
    table: &LayerTable,
    geometry: &MixerGeometry,
    quant_of: impl Fn(&str) -> Option<QuantType>,
) -> Result<bool, EngError> {
    for layer in 0..table.len() {
        let kind = table.slot(layer)?.kind;
        for spec in geometry.dense_specs(WeightLayout::Native, kind)? {
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

/// The capture-preserving routing stage (`Q36_ROUTE_STAGE`, `route_stage`): a staging buffer
/// beside the arena whose address a captured graph bakes in, so the per-layer D2D copies that
/// fill it — launch-only, enqueued in the plain-MoE arm — become graph nodes a replay refreshes
/// every step. The host reads it back once per token, after the step's own read-back (the stream
/// is drained by then, so the copy rides that sync). Leaked like `pins`: the buffer is pinned
/// for the graph's life, 40 layers x 68 bytes, once per engine.
struct RouteStage {
    buffer: &'static DeviceBuffer,
    /// Bytes per layer: ids `[top_k]` i32, weights `[top_k]` f32, gate f32.
    stride: usize,
    top_k: usize,
    /// Host scratch of the whole buffer, refilled by the per-token readback.
    bytes: Vec<u8>,
}

/// The tiled-experts hybrid must be uniform: one packs kind serves every layer, so a mixed
/// override (tiled stacks in some layers, native in others) refuses. Pure so the CPU tests pin
/// the refusal without loaded weights.
pub(super) fn moe_override_uniform(tiled: &[bool]) -> Result<bool, EngError> {
    let Some(&first) = tiled.first() else {
        return Ok(false);
    };
    if tiled.iter().any(|&t| t != first) {
        return Err(EngError::Checkpoint(
            "step weights: the INFERENCE_CUDA_TILED_HEAD override must cover the expert \
             stacks of every layer, not some"
                .to_owned(),
        ));
    }
    Ok(first)
}

fn new_route_stage(
    context: &CudaContext,
    layers: usize,
    top_k: usize,
) -> Result<RouteStage, EngError> {
    let stride = (2 * top_k + 1) * size_of::<f32>();
    let buffer = context.allocate(stride * layers)?;
    Ok(RouteStage {
        buffer: Box::leak(Box::new(buffer)),
        stride,
        top_k,
        bytes: vec![0; stride * layers],
    })
}

/// The staging copies of one layer: three launch-only D2D enqueues from the layer's router
/// ranges into the stage at `layer * stride` (ids, then weights, then gate). Inside a capture
/// window each becomes a memcpy node of the graph.
fn stage_route_layer(
    ctx: &CudaContext,
    stream: &DeviceStream,
    stage: &RouteStage,
    layer: usize,
    route: &super::moe::RouteBufs<'_>,
) -> Result<(), EngError> {
    let at = layer * stage.stride;
    let ids = stage.buffer.range(at, stage.top_k * size_of::<i32>())?;
    let weights = stage.buffer.range(
        at + stage.top_k * size_of::<f32>(),
        stage.top_k * size_of::<f32>(),
    )?;
    let shared_gate = stage
        .buffer
        .range(at + 2 * stage.top_k * size_of::<f32>(), size_of::<f32>())?;
    ctx.copy_device_pinned(stream, route.ids, ids)?;
    ctx.copy_device_pinned(stream, route.weights, weights)?;
    ctx.copy_device_pinned(stream, route.shared_gate, shared_gate)
}

/// The per-token host readback of the whole stage, decoded into one `LayerPicks` per layer in
/// launch order. The caller runs it after the step's own read-back, so the single `read_at` is
/// hot (the stream is already drained).
fn read_staged_route(
    stage: &mut RouteStage,
    stream: &DeviceStream,
) -> Result<Vec<crate::family::qwen36::route_capture::LayerPicks>, EngError> {
    stage.buffer.read_at(stream, 0, &mut stage.bytes)?;
    let layer = stage.bytes.chunks(stage.stride).map(|row| {
        let word = |at: usize| {
            u32::from_le_bytes(row[4 * at..4 * at + 4].try_into().expect("stage row word"))
        };
        crate::family::qwen36::route_capture::LayerPicks {
            ids: (0..stage.top_k).map(|i| word(i) as i32).collect(),
            weights: (0..stage.top_k)
                .map(|i| f32::from_bits(word(stage.top_k + i)))
                .collect(),
            shared_gate: f32::from_bits(word(2 * stage.top_k)),
        }
    });
    Ok(layer.collect())
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
    /// allow it (nothing here needs a borrow longer than the step), and it is left for a later
    /// change.
    ///
    /// The pin slice of the graph is built here too, beside that limit: one leaked slice of
    /// the weight buffers and the arena slot per new step (one step per session), stored for
    /// the whole life of the step and of any graph it captures.
    pub(crate) fn new(
        context: &'a CudaContext,
        res: &'a StepResources,
        weights: &'a CudaWeights,
        options: &StepOptions,
        rules: QwenRules,
    ) -> Result<Self, EngError> {
        Self::build(context, res, weights, options, rules, None)
    }

    /// The build shared by both constructors; `banked` names the slot count and the resolver
    /// of a banked EXL3 load (`None` is the resident step, whose banks must hold every
    /// expert).
    fn build(
        context: &'a CudaContext,
        res: &'a StepResources,
        weights: &'a CudaWeights,
        options: &StepOptions,
        rules: QwenRules,
        banked: Option<(usize, &'a RefCell<Exl3SlotResolver<'a>>)>,
    ) -> Result<Self, EngError> {
        let widths = validate_step_inputs(options, res.dims.shape.vocab, &rules)?;
        // The mode follows the loaded weights: the options carry the same layout, derived from
        // `CudaWeights::layout()` by every production caller, and a mismatch is a typed error
        // before any context work.
        agree_layout(weights.layout()?, options.layout)?;
        if let Some(paged) = &res.paged
            && paged.options != *options
        {
            return Err(EngError::BadRequest(
                "paged resources and step options differ".into(),
            ));
        }
        let sync_each = sync_each_from_env()?;
        if options.layout == WeightLayout::Exl3 {
            // A bad FIFI_EXL3_SINGLE_GATHER stops the start, not the first token.
            context.check_single_gather()?;
        }
        let stream = context.stream()?;
        let state = match &res.paged {
            Some(paged) => ResidentState::allocate_from_plan(context, &stream, &paged.state)?,
            None => ResidentState::allocate(context, &stream, &res.dims, options)?,
        };
        let (event, word) = (context.event()?, context.pinned_word()?);
        let layers: Vec<usize> = (0..res.table.len()).collect();
        // The tiled-experts hybrid: layer 0's gate stack loaded as tiles beside the native
        // session flips the MoE epilogue to the fp16 pack form. Uniformity across layers is
        // enforced when the MoE weights resolve below; layer 0 is enough to pick the op list.
        let moe_tiled = options.layout == WeightLayout::Native
            && res
                .table
                .name(0, Role::FfnGateExps)
                .ok()
                .and_then(|name| weights.get(name))
                .is_some_and(|weight| matches!(weight, CudaWeight::Tiled { .. }));
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
                moe_tiled,
            )?,
            layout: options.layout,
            kv_mode: options.kv_mode,
            moe_tiled,
            paged: res.paged.as_ref(),
            sparse: res.sparse.as_ref(),
        };
        let specs = MoeSpecs::new(&res.dims)?;
        let tiered_mode = options.tiered.is_some();
        let exl3_moe = if options.layout == WeightLayout::Exl3 {
            let dims = moe_dims(&res.dims);
            layers
                .iter()
                .map(|&layer| match banked.as_ref().map(|(slots, _)| *slots) {
                    Some(slots) => {
                        Exl3MoeWeights::resolve_banked(weights, &res.table, layer, &dims, slots)
                    }
                    None => Exl3MoeWeights::resolve(weights, &res.table, layer, &dims),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        // The merged shared-expert block: on by default for an EXL3 load (a load that cannot run it
        // falls back to the reference block with a logged reason), off with
        // `FIFI_EXL3_SHARED_IN_ROUTED=0`, refused if `=1` cannot be honoured. A bad value of the
        // switch stops the start, not the first token.
        let (exl3_merge, merge_intent, merge_source) = if options.layout == WeightLayout::Exl3 {
            let intent = resolve_switch(shared_in_routed_explicit(), true)?;
            let (merge, source) =
                SharedMerge::choose(context, &moe_dims(&res.dims), &exl3_moe, intent)?;
            (merge, intent, source)
        } else {
            (None, Intent::Off, Source::Explicit)
        };
        let (moe, tiered_parts) = if tiered_mode {
            let parts = layers
                .iter()
                .map(|&layer| moe_layer_parts(&res.table, weights, &specs, layer, options.layout))
                .collect::<Result<Vec<_>, _>>()?;
            (Vec::new(), Some(parts))
        } else if options.layout == WeightLayout::Exl3 {
            (Vec::new(), None)
        } else {
            let moe = layers
                .iter()
                .map(|&layer| moe_layer_weights(&res.table, weights, &specs, layer, options.layout))
                .collect::<Result<Vec<_>, _>>()?;
            // The tiled-experts hybrid must be uniform: one packs kind serves every layer, so a
            // mixed override (tiled stacks in some layers, native in others) refuses here
            // instead of failing per layer inside the block's own check.
            moe_override_uniform(
                &moe.iter()
                    .map(|w| matches!(w.gate, StackWeights::Tiled(_)))
                    .collect::<Vec<_>>(),
            )?;
            (moe, None)
        };
        let vocab = res.dims.shape.vocab;
        let route_stage_on = crate::family::qwen36::route_stage::enabled(
            std::env::var(crate::family::qwen36::route_stage::VAR)
                .ok()
                .as_deref(),
        );
        // EXL3 and tiered are refused at step time (their MoE arms are not staged); the stage
        // is not even allocated for them.
        let route_stage = if route_stage_on && options.layout != WeightLayout::Exl3 && !tiered_mode
        {
            let dims = moe_dims(&res.dims);
            let stage = new_route_stage(context, layers.len(), dims.top_k)?;
            crate::family::qwen36::route_stage::set_facts(layers.len(), dims.top_k, dims.n_expert);
            Some(stage)
        } else {
            None
        };
        let mut pinned: Vec<&'a DeviceBuffer> = weights.device_buffers().collect();
        pinned.push(res.arena.buffer());
        if let Some(stage) = &route_stage {
            pinned.push(stage.buffer);
        }
        let pins: &'a [&'a DeviceBuffer] = Box::leak(pinned.into_boxed_slice());
        // The chunk gate's Q6_K scan, beside the other build-time facts: only Native weights carry
        // a quant type; a Tiled session is refused by its layout before the scan matters. The
        // lookup answers `None` for a tensor the load left in the file (the tiered stacks), which
        // the scan takes as not Q6_K.
        let chunk_q6k = options.layout == WeightLayout::Native
            && any_chunk_weight_q6k(&res.table, &res.geometry, |name| match weights.get(name) {
                Some(CudaWeight::Native { weight, .. }) => Some(weight.quant()),
                _ => None,
            })?;
        let step = Self {
            pins,
            krow_pins: &[],
            graph: None,
            krow_graphs: krow_graph::GraphTable::default(),
            krow: None,
            #[cfg(feature = "research-activation")]
            activation: std::cell::RefCell::new(None),
            context,
            res,
            mixers: Mixers::new(context, parts)?,
            moe,
            exl3_moe,
            exl3_bank: banked.map(|(_, bank)| bank),
            banked_echo: 0,
            exl3_merge,
            merge_intent,
            merge_source,
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
            #[cfg(test)]
            chunk_kernel_class: super::prefill::ChunkKernelClass::ProductionMmq,
            state,
            word,
            event,
            stream,
            sync_each,
            started: false,
            warmed: false,
            capture_on: false,
            exl3_device_ids: false,
            refused: false,
            marks: None,
            rules,
            widths,
            chunk_q6k,
            kv8_chunked_prefill: false,
            window: None,
            #[cfg(test)]
            window_fault: None,
            #[cfg(test)]
            window_trace: Default::default(),
            #[cfg(test)]
            moe_pick_tap: Default::default(),
            #[cfg(test)]
            taps: Default::default(),
            speculator: None,
            m_sess_cap: DEFAULT_CUDA_PREFILL_CHUNK,
            logits_bytes: vec![0; f32_bytes(vocab)?],
            shortlist_on: device_shortlist_enabled(std::env::var(SHORTLIST_VAR).ok().as_deref()),
            route_capture: crate::family::qwen36::route_capture::enabled(
                std::env::var(crate::family::qwen36::route_capture::VAR)
                    .ok()
                    .as_deref(),
            ),
            route_stage_on,
            route_stage,
            shortlist_bytes: [0; shortlist::RESULT_BYTES],
            report: RunReport::default(),
            prefill_report: PrefillReport::default(),
            prefix: PrefixSlot::default(),
            prefix_pins: None,
            device_tip: None,
            #[cfg(test)]
            trace_sync_capture: false,
            rollback_wanted: false,
            krow_wanted: false,
            forked_experts: options.forked_experts,
            rollback: None,
            pending_prefix: None,
            needs_reset: false,
        };
        let mut step = step;
        step.refill_prefill_report();
        Ok(step)
    }

    /// [`Self::new`] for a banked EXL3 load: the routed banks hold `slots` members (resolved
    /// against that count) and the decode maps every layer's picks through `bank`'s slot
    /// tables, refilling missed slots from its host store.
    pub(crate) fn new_banked(
        context: &'a CudaContext,
        res: &'a StepResources,
        weights: &'a CudaWeights,
        options: &StepOptions,
        rules: QwenRules,
        slots: usize,
        bank: &'a RefCell<Exl3SlotResolver<'a>>,
    ) -> Result<Self, EngError> {
        Self::build(context, res, weights, options, rules, Some((slots, bank)))
    }

    /// The load-line suffix of the merged MoE block: the resolved mode and its source on an EXL3
    /// load (` exl3_shared_in_routed=on source=default`), nothing on any other layout.
    pub(crate) fn shared_in_routed_suffix(&self) -> String {
        if self.layout == WeightLayout::Exl3 {
            load_suffix(self.exl3_merge.is_some(), self.merge_source)
        } else {
            String::new()
        }
    }

    /// Test builds: puts the shared expert in the routed launches (or back) on this step, which
    /// the process-wide switch cannot do twice in one process. Before the first token only.
    #[cfg(test)]
    pub(crate) fn set_shared_in_routed(&mut self, on: bool) -> Result<(), EngError> {
        if self.started {
            return Err(EngError::BadRequest(
                "step: the MoE block is fixed once the session runs".to_owned(),
            ));
        }
        if on && self.layout != WeightLayout::Exl3 {
            return Err(EngError::BadRequest(
                "step: the shared expert rides in the routed launches of an EXL3 step only"
                    .to_owned(),
            ));
        }
        if on && self.krow_wanted {
            // The same refusal as the variable: the K-row path is qualified without the block.
            crate::family::qwen36::krow::check_krow_switches(None, true)?;
        }
        self.merge_source = Source::Explicit;
        self.merge_intent = if on { Intent::Explicit } else { Intent::Off };
        self.exl3_merge = if on {
            Some(SharedMerge::new(
                self.context,
                &self.moe_dims,
                &self.exl3_moe,
            )?)
        } else {
            None
        };
        Ok(())
    }

    /// Sets the capture settings; refuses after the first token and refuses
    /// `INFERENCE_CUDA_SYNC_EACH=1` together with a capture (`check_mode`'s first caller).
    pub(crate) fn set_capture_config(&mut self, config: CaptureConfig) -> Result<(), EngError> {
        check_mode(self.sync_each, config.capture)?;
        tiered_capture_refusal(self.tiered_mode, config.capture)?;
        check_paged_modes(
            self.res.paged.is_some(),
            config.capture,
            self.kv8_chunked_prefill,
        )?;
        if self.started {
            return Err(EngError::BadRequest(
                "step: the capture mode is fixed once the session runs".to_owned(),
            ));
        }
        exl3_device_pick_refusal(self.layout, config.capture, self.moe_dims.top_k)?;
        // B2: the sparse chain clears and reads its status word with blocking, stream-
        // ordered copies — a captured body forbids both. While sparse is admitted the
        // session stays EAGER, so the chain is never silently absent from a replay graph.
        if config.capture && self.res.sparse.is_some() {
            eprintln!("q36 cuda: capture off — the sparse attention chain is admitted");
        }
        // A banked engine never captures: the slot resolver works on the host between the
        // readback and the launches, which a graph cannot do (the settings force capture off;
        // this guard holds for any caller).
        self.capture_on = config.capture && self.res.sparse.is_none() && self.exl3_bank.is_none();
        // A captured body cannot wait for the host inside a layer: the EXL3 MoE blocks read the
        // picks from device memory then.
        self.exl3_device_ids =
            config.capture && self.layout == WeightLayout::Exl3 && self.exl3_bank.is_none();
        // Created here, before the first token: the events must exist before the capture that
        // records them, and their addresses stay stable for the session's life.
        self.marks = if config.section_marks {
            Some(SectionMarks::new(self.context, &self.res.dims.shape.kinds)?)
        } else {
            None
        };
        Ok(())
    }

    /// Arms the speculative decode step the settings ask for. Call it before the first token. The
    /// n-gram mode drafts from the committed tokens; the MTP mode has no draft step yet and is
    /// refused. A window of more than 8 rows is refused (`check_window_rows`).
    pub(crate) fn set_speculation(&mut self, config: &SpecConfig) -> Result<(), EngError> {
        self.speculator = speculator_of(config)?;
        Ok(())
    }

    /// Sets the cap of `M_sess` (one of 128, 256, 512, 1,024 and 2,048 once the config is
    /// wired). The serve start calls this once at session build, beside `set_capture_config`;
    /// until the CUDA arm accepts `qwen36.prefill_chunk` (it ignores the key), the call passes
    /// the no-cap default, and `chunk_ready` carries the cap only in its accept verdict.
    pub(crate) fn set_m_sess_cap(&mut self, m_sess_cap: u32) {
        self.m_sess_cap = if self.kv8_chunked_prefill {
            // The KV8 chunk domain (KV8_BULK_ROWS in prefill.rs); raising the env's
            // accepted rows past this needs that domain extended first.
            m_sess_cap.min(2048)
        } else {
            m_sess_cap
        };
    }

    pub(super) fn set_kv8_chunked_prefill(&mut self, enabled: bool) -> Result<(), EngError> {
        admit_guard(self.started, self.prefill.is_some())?;
        check_paged_modes(self.res.paged.is_some(), self.capture_on, enabled)?;
        self.kv8_chunked_prefill = enabled;
        if enabled {
            self.m_sess_cap = self.m_sess_cap.min(512);
        }
        Ok(())
    }

    /// Admits the session's prefill arena: the two guards first (before the first token, at most
    /// once — `admit_guard`), then the pre-arena verdict — a session whose layout, KV mode, tier
    /// flag or Q6_K scan refuses the chunk path gets its refusal verdict back with NOTHING
    /// allocated and no device fact even queried — then the MMQ device facts, the ladder costs and
    /// [`admit_m_sess`] with the REAL state term. An admitted `M` allocates the [`PrefillArena`]
    /// (the whole-arena zero fill runs here, ONCE per session); `None` allocates nothing. Both
    /// outcomes return `chunk_ready()`: `Ready` with the admitted `M_sess`, or `NoArena`.
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
        context_staging: bool,
        fixed_admission: bool,
        wide_stage: bool,
    ) -> Result<ChunkVerdict, EngError> {
        admit_guard(self.started, self.prefill.is_some())?;
        let layout = self.state.layout();
        let kv8 = layout.kv.mode == KvMode::Int8G32 && self.kv8_chunked_prefill;
        let fixed = fixed_admission && context_staging && kv8;
        if !needs_arena(
            self.layout,
            layout.kv.mode,
            self.tiered_mode,
            self.chunk_q6k,
            self.kv8_chunked_prefill,
        ) {
            if fixed {
                return Err(EngError::BadRequest(
                    "fixed prefill admission: this step cannot use a chunk arena".into(),
                ));
            }
            return Ok(self.chunk_ready());
        }
        let mut admit = AdmitFacts {
            budget: budget.device_free,
            decode_arena: self.res.arena.bytes() + self.res.krow_arena_bytes(),
            state_planned: state_bytes(
                &self.res.dims,
                layout.kv.mode,
                layout.kv.capacity,
                layout.delta.slots,
            )?,
            state_allocated: layout.bytes()?,
            // shortcut: 16-row KV8 dispatch uses the minimum 128-row arena; specialize if its VRAM cost matters.
            cap: if layout.kv.mode == KvMode::Int8G32
                && self.kv8_chunked_prefill
                && self.m_sess_cap == 16
            {
                128
            } else {
                self.m_sess_cap as usize
            },
        };
        if fixed {
            admit.cap = fixed_chunk_rows(layout.kv.capacity, admit.cap);
        }
        let facts = self.context.mmq_device_facts()?;
        let costs = if self.layout == WeightLayout::Exl3 {
            exl3_chunk_costs(&self.res.dims, facts.nsm)?
        } else {
            chunk_costs(&self.res.dims, facts.nsm)?
        };
        let staging = if kv8 {
            let a = &self.res.geometry.attn;
            Some(if fixed {
                Kv8StagingPlan::for_context(
                    fixed_staging_rows(layout.kv.capacity),
                    a.heads_kv,
                    a.head_dim,
                )?
            } else if context_staging {
                Kv8StagingPlan::for_context(layout.kv.capacity, a.heads_kv, a.head_dim)?
            } else {
                Kv8StagingPlan::new(layout.kv.capacity, a.heads_kv, a.head_dim)?
            })
        } else {
            None
        };
        let (admitted, staging) = if fixed {
            let plan = staging.ok_or_else(|| {
                EngError::MemoryPlan("fixed prefill admission: staging is absent".into())
            })?;
            // The wide flag offers the FULL_STAGE_TIER_ROWS full-split candidate; it is
            // only a candidate where it is actually wider than today's stage.
            if std::env::var_os(super::start::WIDE_STAGE_VAR).is_some()
                && wide_stage
                && !wide_stage_candidate(layout.kv.capacity, wide_stage)
            {
                eprintln!(
                    "q36 cuda: wide CPF stage flag has no effect at {} context rows (needs more than {})",
                    layout.kv.capacity, FULL_STAGE_TIER_ROWS
                );
            }
            let wide = wide_stage_candidate(layout.kv.capacity, wide_stage)
                .then(|| {
                    let a = &self.res.geometry.attn;
                    Kv8StagingPlan::for_context(FULL_STAGE_TIER_ROWS, a.heads_kv, a.head_dim)
                })
                .transpose()?;
            // The plan construction above runs BEFORE the selector, so a MemoryPlan
            // from it would refuse the boot rather than fall back — unreachable in
            // practice: the tier geometry is fixed and bounded, so the constructor
            // cannot overflow; only the selector's budget checks can refuse.
            // One live snapshot gates BOTH candidates inside the selection, so a wide
            // plan can never refuse a boot the default stage would have kept.
            let free = self.context.memory_info()?.free;
            // Boot ledger: the first attempt's exact admission inputs, so a
            // refusal can be closed by subtraction (which charge grew) instead
            // of guessed from the outside.
            let ledger_cost = costs.iter().find(|cost| cost.m == admit.cap);
            // Every figure the admission gates actually charge: the part term is
            // the max of the flat charge and the chunk's own, the budget is the
            // post-margin device free, and the staging figures are named per
            // candidate. The spare minimum is the last owed charge.
            eprintln!(
                "{}",
                ledger_line(&LedgerFigures {
                    budget_post_margin: budget.device_free,
                    live_free: free,
                    state_allocated: admit.state_allocated,
                    decode_arena: admit.decode_arena,
                    chunk_rows: admit.cap,
                    part: ledger_cost.map_or(0, |cost| cost.part),
                    arena: ledger_cost.map_or(0, |cost| cost.arena),
                    extras: &budget.extras,
                    staging: &plan,
                    wide: wide.as_ref(),
                })
            );
            let full_capacity = plan.capacity;
            let (cost, plan, choice) = admit_fixed_staged(
                &admit,
                &budget.extras,
                &costs,
                plan,
                wide,
                free,
                layout.kv.capacity,
            )?;
            // A tight live snapshot degrades the stage rung by rung instead of
            // refusing the boot. Above 131,072 rows the ladder adds up to 48 MiB
            // (64 MiB stage -> 16 MiB floor) of headroom over the 32,768-row default
            // stage, plus 192 MiB when a wide stage was offered and refused; at or
            // below 131,072 rows it runs from the full-context stage down to the
            // floor instead. A snapshot too tight for every rung down to the floor
            // still refuses, with the original full-stage error.
            if plan.capacity < full_capacity {
                eprintln!(
                    "q36 cuda: fixed admission refused at {full_capacity} stage rows, shrank stage to {} stage rows",
                    plan.capacity
                );
            }
            super::prefill::kv8_segment_query_rows(
                super::prefill::chunk_attn_geometry(&self.res.geometry, cost.m),
                layout.kv.capacity.div_ceil(plan.capacity),
            )?;
            let label = stage_label(choice, plan.capacity, full_capacity);
            eprintln!(
                "q36 cuda: fixed KV8 admission, {} rows, {} stage rows ({}), {} B margin, {} B live free; segmented attention enabled",
                cost.m, plan.capacity, label, FIXED_ADMISSION_MARGIN, free
            );
            (Some(cost.m), Some(plan))
        } else {
            admit_with_staging(&admit, &budget.extras, &costs, staging)?
        };
        let Some(m) = admitted else {
            if kv8 {
                eprintln!("q36 cuda: KV8 per-token prefill (no chunk arena admitted)");
            }
            return Ok(self.chunk_ready());
        };
        let allocate = if self.layout == WeightLayout::Exl3 {
            PrefillArena::allocate_exl3
        } else {
            PrefillArena::allocate
        };
        let mut arena = allocate(
            self.context,
            &self.stream,
            &self.res.dims,
            m,
            facts,
            budget.device_free,
        )?;
        if let Some(plan) = staging {
            arena.attach_kv8_staging(self.context, plan, fixed)?;
            eprintln!(
                "q36 cuda: KV8 FP16 attention staging: {} rows, {} bytes",
                plan.capacity, plan.bytes
            );
        } else if kv8 {
            eprintln!("q36 cuda: KV8 16-row attention (staging not admitted)");
        }
        self.prefill = Some(arena);
        // The report static fields come from the arena: refill them now, because the arena did not
        // exist when the step was built. The session has no steps yet, so the refill cannot drop a
        // record.
        self.refill_prefill_report();
        Ok(self.chunk_ready())
    }

    /// The chunk-gate verdict of this session: whether `feed_rows` may run the chunk path. The
    /// conditions, in order: the Native layout, `Fp16` KV, a non-tiered session, no Q6_K MMQ weight
    /// of the chunk, and an ADMITTED prefill arena — until `admit_prefill` admits one, an
    /// otherwise-ready session answers `NoArena` and keeps the per-token path. The accept verdict
    /// carries the admitted `M_sess` of the arena, never the raw cap.
    pub(super) fn chunk_ready(&self) -> ChunkVerdict {
        chunk_verdict(&ChunkFacts {
            layout: self.layout,
            kv_mode: self.state.layout().kv.mode,
            kv8_chunked_prefill: self.kv8_chunked_prefill,
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

    /// The resident state of the step, for the device tests' direct writes and reads (the poison
    /// NaN through `delta_rec`, the stale word through `word`): the production surface keeps the
    /// state private, so the rows reach it through this test-only borrow. Test builds only.
    #[cfg(test)]
    pub(super) fn state_for_tests(&self) -> &ResidentState {
        &self.state
    }

    /// The step own stream, for the device tests direct writes (the stale word). Test builds only.
    #[cfg(test)]
    pub(super) fn stream_for_tests(&self) -> &DeviceStream {
        &self.stream
    }

    /// The attached KV8 staging plan, for the device tests' stage-provenance pins. Test
    /// builds only.
    #[cfg(test)]
    pub(super) fn kv8_staging_plan_for_tests(&self) -> Option<Kv8StagingPlan> {
        Some(self.prefill.as_ref()?.kv8_staging().ok()??.plan)
    }

    /// Builds the tiered runtime of a step made with `StepOptions.tiered`: one bank per layer,
    /// the host and the optional cold staging bank of the chunked prefill. Refuses a plain
    /// step, a second attach and a bank count other than the layer count. The scratch buffers
    /// are allocated here.
    pub(crate) fn attach_tiered(
        &mut self,
        banks: Vec<ExpertBank>,
        host: TieredHost,
        cold: Option<super::tiered_step::ColdStaging>,
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
            parts, banks, scratch, host, cold,
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

    pub(crate) fn context_limit(&self) -> usize {
        self.state.layout().kv.capacity
    }

    pub(crate) fn tokens(&self) -> usize {
        self.state.tokens()
    }

    /// Feeds prompt rows through the chunk plan: a multi-row step runs as one chunk, a one-row step
    /// through the per-token feed. The chunk protocol records its own step; this closure records
    /// only the feed steps. start_rows is taken at the first chunked call of a reset or import
    /// epoch.
    pub(super) fn feed_rows_chunked(
        &mut self,
        tokens: &[u32],
        m_sess: usize,
    ) -> Result<(), EngError> {
        use super::prefill_driver::drive_chunks;
        use super::prefill_plan::PrefillStep;
        use crate::family::session::TokenStepper;

        let start = self.tokens();
        let fixed_depth = self.fixed_prefill_depth();
        self.prefill_report.note_chunked_epoch_start(start);
        drive_chunks(
            start,
            tokens,
            m_sess,
            fixed_depth,
            |step, rows| match step {
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
            },
        )
        .map(|_| ())
    }

    pub(super) fn fixed_prefill_depth(&self) -> bool {
        self.prefill.as_ref().is_some_and(PrefillArena::segmented)
    }

    pub(super) fn record_feed(&mut self, p0: usize, ns: u64) {
        self.prefill_report.push_step(StepRecord {
            p0,
            rows: 1,
            kind: StepKind::Feed,
            ns,
            status: 0,
        });
    }

    /// Runs one prefill chunk at p0: open, clear, body, status-read, record and publish. The final
    /// status read gates publication. KV8 also uses blocking causal-length writes before each
    /// attention tile. Every error after the chunk opens poisons the state.
    pub(super) fn run_chunk(&mut self, p0: usize, rows: &[u32]) -> Result<(), EngError> {
        use super::moe::Launch;
        use super::prefill::{
            ChunkBufs, ChunkModel, ChunkRows, ChunkRun, ChunkSparse, ChunkViews, ChunkWeights,
            run_chunk_body,
        };
        use super::prefill_driver::{check_chunk_start, chunk_ids};
        use super::prefill_plan::ChunkSlot;

        // A chunk advances the live state: refused through a gated state or an awaiting window.
        self.state_usable()?;
        #[cfg(test)]
        let kernel_class = self.chunk_kernel_class;
        let Self {
            state,
            prefill,
            mixers,
            moe,
            exl3_moe,
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
        // A banked chunk runs its MoE stage in sub-blocks (see prefill_exl3::moe_block), so
        // any admitted chunk row count serves.
        check_chunk_start(p0, state.tokens())?;
        let arena = prefill
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("prefill chunk: no prefill arena".to_owned()))?;
        let m = rows.len();
        let chunk_rows = ChunkRows::new(p0, m, state.layout().kv.capacity, arena.m_sess())?;
        let ids = chunk_ids(rows, widths.verdict)?;
        let weights = match globals.embed {
            super::resources::GlobalWeight::Native(embed) => {
                ChunkWeights::new(mixers.weights(), moe, embed)
            }
            super::resources::GlobalWeight::Bf16(table) => {
                ChunkWeights::exl3(mixers.weights(), exl3_moe, table)
            }
            super::resources::GlobalWeight::Tiled(_) | super::resources::GlobalWeight::Exl3(_) => {
                return Err(EngError::BadRequest(
                    "prefill chunk: the embedding is neither native nor an EXL3 table".to_owned(),
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
        )?
        .with_kv8_staging(arena.kv8_staging()?);
        let model = ChunkModel::new(&res.table, &res.geometry, &res.dims, weights)?;
        // 3. From here every error poisons.
        *started = true;
        let opened = std::time::Instant::now();
        let body = || -> Result<usize, EngError> {
            state.begin_prefill()?;
            state.clear_prefill_status(stream)?;
            let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            ids_range
                .buffer()
                .write_at(stream, ids_range.offset(), &bytes)?;
            let run = ChunkRun::new(&*state, &model, &bufs, chunk_rows)?;
            let run = run.with_bank(self.exl3_bank).with_sparse(
                res.sparse
                    .as_ref()
                    .map(|sp| ChunkSparse::new(sp, &res.dims.shape.kinds)),
            );
            // The tiered chunk arm: the runtime is present when the session is tiered
            // AND the chunk flag is on (the gate ships dark until device-qualified).
            let tiered_chunk = self.tiered.as_ref().filter(|_| {
                super::prefill::tiered_chunk_enabled(
                    std::env::var_os(super::prefill::TIERED_CHUNK_VAR).as_deref(),
                )
            });
            let run = run.with_tiered(tiered_chunk);
            #[cfg(test)]
            let run = run.with_kernel_class(kernel_class);
            run_chunk_body(
                &Launch {
                    ctx: context,
                    stream,
                },
                &run,
            )?;
            Ok(run.segmented_parts())
        };
        let segmented_parts = body().inspect_err(|_| state.poison())?;
        prefill_report.note_segmented_parts(segmented_parts);
        let t_status = std::time::Instant::now();
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
        let t_reval = std::time::Instant::now();
        state
            .revalidate(stream, p0 + m, status)
            .inspect_err(|_| state.poison())?;
        if std::env::var_os("Q36_TIERED_TRACE").is_some_and(|v| v == "1") {
            eprintln!(
                "chunk-close p0={p0} m={m}: body {:.0} ms, status-read {:.0} ms, revalidate {:.0} ms",
                t_status.duration_since(opened).as_secs_f64() * 1e3,
                t_reval.duration_since(t_status).as_secs_f64() * 1e3,
                t_reval.elapsed().as_secs_f64() * 1e3,
            );
        }
        // R3: the chunk committed [p0, p0 + m) and its sketch cells are queued; the
        // selection-eligible prefix grows with it, but only when this chunk continues
        // it exactly (interrupted or gapped prefill keeps the smaller prefix).
        if let Some(sp) = res.sparse.as_ref() {
            sp.note_prefill(p0, m);
        }
        Ok(())
    }

    /// Whether the step runs the tiered MoE path, whose prefix reuse is experimental.
    pub(super) fn is_tiered(&self) -> bool {
        self.tiered_mode
    }

    pub(crate) fn last_prefill_report(&self) -> &PrefillReport {
        &self.prefill_report
    }

    pub(super) fn note_feed_rows(&mut self, verdict: &ChunkVerdict, rows: usize) {
        self.prefill_report.note_feed_rows(verdict, rows);
    }

    pub(super) fn take_job_line(&mut self) -> Option<String> {
        self.prefill_report.take_job_line()
    }

    pub(super) fn record_prefill_piece(&mut self, tokens: usize, began: std::time::Instant) {
        let ns = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.prefill_report.record_piece(tokens, ns);
    }

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
        // Exempt from `state_usable`: the one recovery of a failed commit must not be refused by
        // the state it recovers.
        window_logic::reset_prologue(self)?;
        self.state.reset(&self.stream)?;
        // A reset is the one recovery of a failed window commit: the gate opens only when the
        // state really was zeroed.
        window_logic::reset_epilogue(self);
        // R3: the position restarts at zero; the sketch-valid prefix follows (the
        // re-appends rewrite every consumed cell, but the binding stays honest).
        if let Some(sp) = self.res.sparse.as_ref() {
            sp.note_restore(0);
        }
        if let Some(pool) = self.prefix_pins.as_mut() {
            pool.invalidate()?;
        }
        self.refill_prefill_report();
        Ok(())
    }

    /// Poisons the state as a failed step does. Test builds only: the fault injection of the
    /// eager window uses it.
    #[cfg(test)]
    pub(super) fn poison_state(&mut self) {
        self.state.poison();
    }

    /// The committed state as a host value, for the checkpoint store.
    pub(crate) fn export_state(&self) -> Result<ExportedState, EngError> {
        window_logic::export_gate(self)?;
        self.state.export_state(&self.stream)
    }

    /// Replaces the committed state with `exported`: every restore goes through here. The order
    /// and the refusals live in `window_logic::import_state`.
    pub(crate) fn import_state(&mut self, exported: &ExportedState) -> Result<(), EngError> {
        window_logic::import_state(self, exported)
    }

    /// The device writes of an import (see `window_gate`).
    pub(super) fn overwrite_state(&mut self, exported: &ExportedState) -> Result<(), EngError> {
        self.state.import_state(&self.stream, exported)?;
        // R3: the checkpoint restored the KV planes and the position, but NOT the
        // sparse sketch (it belongs to this session's SparseExec): the sketch-valid
        // prefix rewinds with it, so a stale sketch cell is never selectable.
        if let Some(sp) = self.res.sparse.as_ref() {
            sp.note_restore(exported.tokens);
        }
        if let Some(pool) = self.prefix_pins.as_mut() {
            pool.invalidate()?;
        }
        self.refill_prefill_report();
        Ok(())
    }

    /// Whether this load overrode the head to tiles (the decode-only hybrid): the final norm
    /// then emits the tiled pack and the vendored head matmul reads it.
    fn head_is_tiled(&self) -> bool {
        matches!(self.globals.head, HeadWeight::Tiled(_))
    }

    fn view(&self, slot: Slot) -> Result<DeviceRange<'a>, EngError> {
        let res: &'a StepResources = self.res;
        let index = res
            .plan
            .index(slot)
            .ok_or_else(|| EngError::BadRequest(format!("step: the plan has no slot {slot:?}")))?;
        res.arena.tensor(index)
    }

    /// The routing ranges of the MoE block of the layers whose residual input is buffer `parity`
    /// (0 is `ResidA`, 1 is `ResidB`).
    fn route_bufs(&self, parity: usize) -> Result<RouteBufs<'_>, EngError> {
        let view = |slot| self.view(slot);
        Ok(RouteBufs {
            resid_in: view(resid_in(parity))?,
            resid_out: view(resid_out(parity))?,
            norm_row: view(Slot::NormRow)?,
            logits: view(Slot::RouterLogits)?,
            ids: view(Slot::TopkId)?,
            weights: view(Slot::TopkWeight)?,
            shared_gate: view(Slot::SharedGate)?,
            status: self.state.step_status()?,
        })
    }

    /// The ranges of the EXL3 MoE block of the layers whose residual input is buffer `parity`.
    fn exl3_moe_buffers(&self, parity: usize) -> Result<Exl3MoeBufs<'_>, EngError> {
        Ok(Exl3MoeBufs {
            route: self.route_bufs(parity)?,
            expert_gu: self.view(Slot::ExpertGu)?,
            expert_act: self.view(Slot::ExpertAct)?,
            expert_down: self.view(Slot::ExpertDown)?,
            shared_gu: self.view(Slot::SharedGu)?,
            shared_act: self.view(Slot::SharedAct)?,
            shared_out: self.view(Slot::SharedOut)?,
            scratch: self.exl3_scratch()?,
        })
    }

    fn exl3_scratch(&self) -> Result<Scratch<'_>, EngError> {
        Ok(Scratch {
            xh: self.view(Slot::Exl3Xh)?,
            z: self.view(Slot::Exl3Z)?,
            flag: self.view(Slot::Exl3Flag)?,
        })
    }

    /// Queues the clearing of the sticky arithmetic flag of the EXL3 projections, right after
    /// `step_begin`.
    fn clear_exl3_flag(&self) -> Result<(), EngError> {
        self.context
            .exl3_flag_clear(&self.stream, self.view(Slot::Exl3Flag)?)
    }

    /// Queues the fold of the flag into the status word, after the last EXL3 projection and
    /// before `argmax`, so the result pair carries the verdict (mask 16).
    fn fold_exl3_flag(&self, params: DeviceRange<'_>) -> Result<(), EngError> {
        self.context
            .exl3_flag_fold(&self.stream, params, self.view(Slot::Exl3Flag)?)
    }

    /// Whether this load overrode the expert stacks to tiles (the tiled-experts hybrid): the
    /// MoE blocks then run the Tiled per-slot form over the plan's fp16 packs.
    fn moe_is_tiled(&self) -> bool {
        self.moe
            .first()
            .is_some_and(|w| matches!(w.gate, StackWeights::Tiled(_)))
    }

    /// The buffers of the MoE block of the layers whose residual input is buffer `parity` (0 is
    /// `ResidA`, 1 is `ResidB`).
    fn moe_buffers(&self, parity: usize) -> Result<MoeBuffers<'_>, EngError> {
        let view = |slot| self.view(slot);
        Ok(MoeBuffers {
            route: self.route_bufs(parity)?,
            expert: ExpertBufs {
                // The packs follow the plan: the fp16 packs of the Tiled plan, the padded q8_1
                // packs of the Native plan - or, in the tiled-experts hybrid, the plan's dead
                // fp16 slots beside the native ones, because the overridden stacks read them.
                packs: if self.res.plan.index(Slot::Q8Moe).is_some() && !self.moe_is_tiled() {
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
        self.run_token_output(token, want_logits, false)
    }

    fn run_token_output(
        &mut self,
        token: u32,
        want_logits: bool,
        want_shortlist: bool,
    ) -> Result<u32, EngError> {
        if let Some(bank) = self.exl3_bank {
            // The banked session's pulse: one line every 256 tokens, the cumulative hit/miss
            // counts that explain its speed (a miss is a host-store refill).
            self.banked_echo += 1;
            if self.banked_echo.is_multiple_of(256) {
                let (hits, misses) = bank.borrow().stats();
                eprintln!(
                    "q36 cuda: EXL3 expert bank after {} tokens: hits {hits}, misses {misses}",
                    self.banked_echo
                );
            }
        }
        let ran = self.run_token_inner(token, want_logits, want_shortlist);
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
    fn run_token_inner(
        &mut self,
        token: u32,
        want_logits: bool,
        want_shortlist: bool,
    ) -> Result<u32, EngError> {
        check_token_id(token, self.widths.verdict)?;
        // A window that waits for its verdict leaves the state ahead of the lineage, and a failed
        // commit leaves the recovery gate set: a decode step through either is refused.
        window_logic::step_gate(self)?;
        if self.tiered_mode && self.tiered.is_none() {
            return Err(EngError::BadRequest(
                "step: a tiered step needs attach_tiered before its first token".to_owned(),
            ));
        }
        // The spike instruments the plain gguf MoE arm only, so it refuses the layouts it
        // cannot record (silently partial picks would poison the measurement). The stage is
        // refused with it: it stages the same arm.
        if (self.route_capture || self.route_stage_on)
            && (self.layout == WeightLayout::Exl3 || self.tiered_mode)
        {
            return Err(EngError::BadRequest(
                "step: Q36_ROUTE_CAPTURE and Q36_ROUTE_STAGE cover the plain MoE arm only, not \
                 EXL3 or tiered"
                    .to_owned(),
            ));
        }
        let mut plan = self.readback_plan(want_logits)?;
        if want_shortlist {
            plan = ReadbackPlan::shortlist(self.res.dims.shape.vocab)?;
        }
        // The start block is written at a committed boundary (the state is idle), then the step
        // is admitted; a refused write queues nothing. The marks stop reading meanwhile: the
        // elapsed queries are legal only after this token's read-back.
        self.started = true;
        if let Some(marks) = self.marks.as_mut() {
            marks.readable = false;
        }
        self.write_start(token)?;
        let admitted = self.state.enqueue_rows(STEP_ROWS)?;
        let mut entered = 0;
        // The layer trace needs per-layer readbacks, which cannot live inside a captured
        // graph: a traced run stays EAGER and claims one step index for its layers. The
        // routing spike readback is eager for the same reason.
        let trace_step = if crate::family::qwen36::trace::trace_enabled() {
            crate::family::qwen36::trace::begin_step()
        } else {
            None
        };
        if self.route_capture {
            crate::family::qwen36::route_capture::begin_step();
        }
        let mode = if trace_step.is_some() || self.route_capture {
            Mode::Eager
        } else {
            next_mode(
                self.graph.is_some(),
                self.warmed,
                self.capture_on,
                self.refused,
            )
        };
        let ran = match mode {
            Mode::Replay => self.replay().map(|()| HowRan::Captured),
            Mode::Capture => self.capture_once(admitted.admit_pos, &mut entered),
            Mode::Eager => self
                .run_body(admitted.admit_pos, &mut entered, trace_step)
                .map(|()| HowRan::Eager),
        };
        let how = match ran {
            Ok(how) => how,
            Err(error) => return Err(self.fail(entered, error)),
        };
        // The first two staged steps log their picks once: a bench log then carries the sanity
        // evidence that the readback held real picks (top-k ids below n_expert, no -1 fault,
        // renormalized weights) without any extra surface.
        if self.route_capture
            && let Some((step, layers)) = crate::family::qwen36::route_capture::staged()
            && step < 2
            && let (Some(first), Some(last)) = (layers.first(), layers.last())
        {
            eprintln!(
                "q36 route capture: step {step} staged {} layers; first ids {:?} weights {:?} gate {:.4}; last ids {:?}",
                layers.len(),
                first.ids,
                first.weights,
                first.shared_gate,
                last.ids
            );
        }
        let pair = self.read_back(&plan).inspect_err(|_| self.state.poison())?;
        if let Some(stage) = self.route_stage.as_mut() {
            // The one host readback of the stage (see `RouteStage`): the stream is drained by
            // `read_back`, so this single `read_at` rides that sync and adds no wait of its own.
            // A failure poisons like `read_back`'s own does, and the step index is claimed only
            // after a successful read so the ring never shows a gap.
            let picks =
                read_staged_route(stage, &self.stream).inspect_err(|_| self.state.poison())?;
            let step = crate::family::qwen36::route_stage::begin_step();
            if step < 2 {
                let first = picks.first().expect("stage readback has layer 0");
                let last = picks.last().expect("stage readback has the last layer");
                eprintln!(
                    "q36 route stage: step {step}: {} layers; layer 0 ids {:?} weights {:?} \
                     gate {:.4}; layer {} ids {:?}",
                    picks.len(),
                    first.ids,
                    first.weights,
                    first.shared_gate,
                    picks.len() - 1,
                    last.ids
                );
            }
            crate::family::qwen36::route_stage::store(step, picks);
        }
        if let Some(error) = exl3_pair_refusal(self.layout, pair) {
            self.state.poison();
            return Err(error);
        }
        if let Some(marks) = self.marks.as_mut() {
            marks.readable = true;
        }
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

    /// The marked groups of the section plan, for the report lines of the device rows; empty when
    /// the step runs without marks.
    pub(super) fn marked_groups(&self) -> &[MarkedGroup] {
        self.marks
            .as_ref()
            .map(|marks| marks.plan.groups.as_slice())
            .unwrap_or(&[])
    }

    /// The section readings of the last completed token, or `None` when the step runs without
    /// marks. Valid once a token's read-back has completed: the replay (or the eager body) that
    /// recorded the timing events has then finished, so the elapsed queries find completed records.
    pub(super) fn section_times(&self) -> Result<Option<SectionTimes>, EngError> {
        let Some(marks) = &self.marks else {
            return Ok(None);
        };
        if !marks.readable {
            // Refused by host state, not by a native query: a query on unfinished or
            // never-recorded events (600/400) would latch the backend terminal.
            return Err(EngError::BadRequest(
                "section timing: no completed marked token to read".to_owned(),
            ));
        }
        let embed = marks.embed_start.elapsed_ms_since(&marks.embed_end)?;
        let tail = marks.tail_start.elapsed_ms_since(&marks.end)?;
        let whole = marks.whole_start.elapsed_ms_since(&marks.end)?;
        let groups = marks
            .groups
            .iter()
            .map(|(start, end)| start.elapsed_ms_since(end))
            .collect::<Result<Vec<_>, _>>()?;
        let mixer_ms = marks
            .groups
            .iter()
            .zip(&marks.mixer_end)
            .map(|((start, _), mixer_end)| start.elapsed_ms_since(mixer_end))
            .collect::<Result<Vec<_>, _>>()?;
        let core_ms = if crate::family::qwen36::section_core_marks() {
            marks
                .core
                .iter()
                .map(|(start, end)| start.elapsed_ms_since(end))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // The core pair was never recorded: zeros keep the readings shaped (the split is
            // an opt-in attribution run; a never-recorded event must never be queried).
            vec![0.0; marks.core.len()]
        };
        let head_ms = marks.head_start.elapsed_ms_since(&marks.head_end)?;
        marks
            .plan
            .reconcile(
                embed,
                tail,
                &groups,
                FineSplit {
                    mixer_ms,
                    core_ms,
                    head_ms,
                },
                whole,
            )
            .map(Some)
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
    /// nothing to or from the host, reads no device word and waits for nothing (it does allocate on
    /// the HOST: the op list and the two sets of views are built per call, which does not touch the
    /// stream and is not captured). The named exceptions are the `INFERENCE_CUDA_SYNC_EACH` branch
    /// (eager debug mode, refused together with a capture by `readback::check_mode`) and the
    /// section marks: `record_timing` and its `is_capturing` query run in the body — inside a
    /// capture window the record becomes an event-record node of the graph, outside one an eager
    /// record. The start block is written before it and the read-back follows it. `entered` counts
    /// the ops that were entered, the failing one included. Opt-in research activation capture is
    /// another eager-only exception: one synchronous read after Moe(19), before arena reuse. The
    /// routing readback spike (`Q36_ROUTE_CAPTURE`, `route_capture`) is the last one: three
    /// synchronous reads after every plain-MoE block, so a step with it on stays EAGER. The
    /// capture-preserving stage (`Q36_ROUTE_STAGE`, `route_stage`) is launch-only like the rest
    /// of the body: its per-layer D2D staging copies enqueue on the owned stream and become
    /// memcpy nodes of a captured graph (the host readback runs after the body, beside
    /// `read_back`), so it does NOT force eager and is the production shape.
    fn run_body(
        &self,
        admit_pos: usize,
        entered: &mut usize,
        trace_step: Option<u32>,
    ) -> Result<(), EngError> {
        // An EXL3 plan has no packs: its MoE blocks build their own ranges.
        let bufs = if self.layout == WeightLayout::Exl3 {
            None
        } else {
            Some([self.moe_buffers(0)?, self.moe_buffers(1)?])
        };
        let params = self.state.word(Word::StepParams)?;
        for op in token_ops(self.res.table.len()) {
            *entered += 1;
            // The mark record comes after the count, so a failed record counts as the op it
            // precedes (only the first op's failure still finds nothing launched, which a
            // count of one reports as cancelable).
            if let Some(marks) = &self.marks {
                marks.record_before(op, &self.stream)?;
            }
            self.run_op(op, admit_pos, bufs.as_ref(), params, trace_step)?;
            #[cfg(test)]
            self.tap_after(op, 1, false)?;
            #[cfg(feature = "research-activation")]
            if let StepOp::Moe(crate::family::qwen36::activation::LAYER) = op {
                self.capture_activation(admit_pos)?;
            }
            if let Some(marks) = &self.marks {
                marks.record_after(op, &self.stream)?;
            }
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
            || self.run_body(admit_pos, entered, None),
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
                self.run_body(admit_pos, entered, None)?;
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

    /// The trace dump of one decode layer: the post-mixer, pre-MoE residual row
    /// (`resid_in` of the MoE buffers, f32 `[hidden]`) as `step{S}_layer{L}.f32`. Runs only
    /// under `Q36_LAYER_TRACE_DIR` (the caller holds the `Option`); `read_at` orders the copy
    /// on the stream, so the row is final when it returns.
    fn trace_moe_row(
        &self,
        launch: &Launch<'_>,
        bufs: &MoeBuffers<'_>,
        step: u32,
        layer: usize,
    ) -> Result<(), EngError> {
        self.trace_row(launch, &bufs.route.resid_in, step, layer)
    }

    fn trace_row(
        &self,
        launch: &Launch<'_>,
        resid: &DeviceRange<'_>,
        step: u32,
        layer: usize,
    ) -> Result<(), EngError> {
        let row = super::trace::f32_range(launch, resid)?;
        crate::family::qwen36::trace::write_step_row(step, layer, &row)
    }

    /// The routing spike readback of one MoE block: the pick ids, pick weights and shared gate
    /// of `route`, each copied out with one stream-ordered `read_at` (so the first syncs the
    /// block, the next two read final data). The caller gates on `route_capture` and a block
    /// that ran clean; the picks land in the staged slot in launch order.
    fn capture_route_picks(
        &self,
        stream: &DeviceStream,
        route: &super::moe::RouteBufs<'_>,
        top_k: usize,
    ) -> Result<(), EngError> {
        let read = |range: &DeviceRange<'_>, bytes: usize| -> Result<Vec<u8>, EngError> {
            let mut raw = vec![0u8; bytes];
            range.buffer().read_at(stream, range.offset(), &mut raw)?;
            Ok(raw)
        };
        let ids = read(&route.ids, top_k * size_of::<i32>())?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        let weights = read(&route.weights, top_k * size_of::<f32>())?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let shared_gate = f32::from_le_bytes(
            read(&route.shared_gate, size_of::<f32>())?
                .as_chunks::<4>()
                .0[0],
        );
        crate::family::qwen36::route_capture::record(
            crate::family::qwen36::route_capture::LayerPicks {
                ids,
                weights,
                shared_gate,
            },
        );
        Ok(())
    }

    fn run_op(
        &self,
        op: StepOp,
        admit_pos: usize,
        bufs: Option<&[MoeBuffers<'_>; 2]>,
        params: DeviceRange<'_>,
        trace_step: Option<u32>,
    ) -> Result<(), EngError> {
        let (ctx, s) = (self.context, &self.stream);
        let packs = || {
            bufs.ok_or_else(|| {
                EngError::BadRequest("step: the EXL3 plan holds no pack buffers".to_owned())
            })
        };
        match op {
            StepOp::Begin => {
                ctx.step_begin(s, params)?;
                if self.layout == WeightLayout::Exl3 {
                    self.clear_exl3_flag()?;
                }
                Ok(())
            }
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
                    EmbedWeight::Bf16(table) => ctx.embed_bf16(
                        s,
                        EmbedBf16Call {
                            table: table.range(0, table.bytes())?,
                            ids,
                            x,
                            status,
                            m: STEP_ROWS,
                            n_rows: self.res.dims.shape.vocab,
                            k: self.res.geometry.hidden,
                        },
                    ),
                    EmbedWeight::Exl3(_) => Err(EngError::Checkpoint(
                        "step weights: an EXL3 matrix is not an embedding table".to_owned(),
                    )),
                }
            }
            StepOp::Mixer(layer) => {
                let io = BlockIo {
                    resid: self.view(resid_in(layer))?,
                    admit_pos,
                };
                // The attention-core pair of the finer split: `None` on unmarked layers (the
                // pair is one extra event node per marked group per replay). Opt-in via
                // `Q36_SECTION_CORE_MARKS=1`: an event node BETWEEN the block's dependent
                // kernels costs real replay time on small GPUs (measured +30ms on the Xavier
                // over 20 groups), unlike the boundary events, so the default rig stays clean
                // and the core/proj split is an explicit attribution run.
                let core_marks = if crate::family::qwen36::section_core_marks() {
                    self.marks.as_ref().and_then(|marks| {
                        marks
                            .group_index(layer)
                            .map(|at| (&marks.core[at].0, &marks.core[at].1))
                    })
                } else {
                    None
                };
                self.mixers.run_block(s, &self.state, layer, io, core_marks)
            }
            StepOp::Moe(layer) if self.layout == WeightLayout::Exl3 => {
                let weights = self.exl3_moe.get(layer).ok_or_else(|| {
                    EngError::BadRequest(format!("step: layer {layer} has no EXL3 MoE weights"))
                })?;
                let launch = Launch { ctx, stream: s };
                let bufs = self.exl3_moe_buffers(layer % 2)?;
                if let Some(step) = trace_step {
                    self.trace_row(&launch, &bufs.route.resid_in, step, layer)?;
                }
                let mut bank_guard = self.exl3_bank.map(|bank| bank.borrow_mut());
                // The overlap fact travels with the resolver: the start armed it from the
                // environment, the block splits its launches around the copy-stream refills.
                let overlap = bank_guard
                    .as_ref()
                    .is_some_and(|guard| guard.overlap_armed());
                let mut bank = bank_guard.as_mut().map(|guard| guard.layer(layer));
                let block = if let Some(merge) = &self.exl3_merge {
                    run_exl3_moe_merged(
                        &launch,
                        (weights, &bufs, merge),
                        &self.moe_dims,
                        self.exl3_device_ids,
                        bank.as_mut(),
                        overlap,
                    )
                } else {
                    run_exl3_moe(
                        &launch,
                        weights,
                        &bufs,
                        &self.moe_dims,
                        self.exl3_device_ids,
                        bank.as_mut(),
                        overlap,
                    )
                };
                #[cfg(test)]
                if block.is_ok()
                    && let Some(taps) = self.moe_pick_tap.borrow_mut().as_mut()
                {
                    let ids = bufs.route.ids;
                    let mut raw = vec![0u8; self.moe_dims.top_k * size_of::<i32>()];
                    ids.buffer().read_at(s, ids.offset(), &mut raw)?;
                    taps.push((
                        layer,
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|word| i32::from_le_bytes(*word))
                            .collect(),
                    ));
                }
                block
            }
            StepOp::Moe(layer) if self.tiered_mode => {
                let runtime = self.tiered.as_ref().ok_or_else(|| {
                    EngError::BadRequest("step: the tiered runtime is not attached".to_owned())
                })?;
                let launch = Launch { ctx, stream: s };
                let bufs = &packs()?[layer % 2];
                if let Some(step) = trace_step {
                    self.trace_moe_row(&launch, bufs, step, layer)?;
                }
                runtime
                    .borrow_mut()
                    .run_layer(&launch, layer, bufs, &self.moe_dims)
            }
            StepOp::Moe(layer) => {
                let weights = self.moe.get(layer).ok_or_else(|| {
                    EngError::BadRequest(format!("step: layer {layer} has no MoE weights"))
                })?;
                let launch = Launch { ctx, stream: s };
                let bufs = &packs()?[layer % 2];
                if let Some(step) = trace_step {
                    self.trace_moe_row(&launch, bufs, step, layer)?;
                }
                let block = run_moe_block(&launch, weights, bufs, &self.moe_dims);
                if block.is_ok() {
                    if let Some(stage) = &self.route_stage {
                        stage_route_layer(ctx, s, stage, layer, &bufs.route)?;
                    }
                    if self.route_capture {
                        self.capture_route_picks(s, &bufs.route, self.moe_dims.top_k)?;
                    }
                }
                block
            }
            StepOp::FinalNorm => self.final_norm(),
            StepOp::LmHead => self.lm_head(),
            StepOp::Argmax => {
                if self.layout == WeightLayout::Exl3 {
                    self.fold_exl3_flag(params)?;
                }
                // The logits row of a traced step feeds the top-2 margin check (the near-tie
                // question of the chunk-split diagnostic); `read_at` orders on the stream.
                if let Some(step) = trace_step {
                    let launch = Launch { ctx, stream: s };
                    let row = super::trace::f32_range(&launch, &self.view(Slot::Logits)?)?;
                    crate::family::qwen36::trace::write_step_logits(step, &row)?;
                }
                ctx.argmax(
                    s,
                    ArgmaxCall {
                        n: self.widths.argmax_n,
                        logits: self.view(Slot::Logits)?,
                        part: self.view(Slot::ArgmaxPart)?,
                        params,
                        result: Some(self.state.word(Word::ResultPair)?),
                    },
                )
            }
            StepOp::Advance => ctx.step_advance(s, params, STEP_ROWS as u32),
        }
    }

    fn final_norm(&self) -> Result<(), EngError> {
        let layers = self.res.table.len();
        let norm = self.globals.output_norm;
        let h = self.view(final_resid(layers))?;
        let weight = norm.range(0, norm.bytes())?;
        match self.layout {
            // The pack is discarded; the EXL3 head reads the F32 row beside it.
            WeightLayout::Exl3 => self.context.add_norm_pack(
                &self.stream,
                AddNormCall {
                    m: STEP_ROWS,
                    n: self.res.geometry.hidden,
                    eps: self.res.geometry.eps,
                    h,
                    delta: None,
                    weight,
                    xf: self.view(Slot::XfFinal)?,
                    norm_out: Some(self.view(Slot::FinalNormRow)?),
                },
            ),
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
            // FinalNormRow slot. Writing NormRow here would have broken no lifetime (its native
            // lifetime covers FINAL_NORM and live slots never share bytes) - it silently repurposed
            // the router-input value slot as scratch, a latent bug the moment NormRow's lifetime
            // tightens to its true need.
            //
            // The decode-only tiled-head hybrid: when the load overrode the head to tiles, the
            // final norm emits the tiled pack beside the f32 row (the vendored head matmul reads
            // the pack, not the row), exactly the Exl3 producer shape.
            WeightLayout::Native if self.head_is_tiled() => self.context.add_norm_pack(
                &self.stream,
                AddNormCall {
                    m: STEP_ROWS,
                    n: self.res.geometry.hidden,
                    eps: self.res.geometry.eps,
                    h,
                    delta: None,
                    weight,
                    xf: self.view(Slot::XfFinal)?,
                    norm_out: Some(self.view(Slot::FinalNormRow)?),
                },
            ),
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
        // The pack slot is resolved INSIDE each arm: the plain native plan's xf slot is dead
        // bytes (the matvec head never reads it), and a lookup before the match is only needed
        // by the arms that use it.
        let y = self.view(Slot::Logits)?;
        if let HeadWeight::Exl3(head) = self.globals.head {
            let io = Exl3Io {
                x: self.view(Slot::FinalNormRow)?,
                xh: self.view(Slot::Exl3Xh)?,
                z: self.view(Slot::Exl3Z)?,
                out: y,
                flag: self.view(Slot::Exl3Flag)?,
                rows: STEP_ROWS,
                fused: false,
            };
            return self
                .context
                .exl3_linear(&self.stream, head.view()?.linear_call(&io));
        }
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
            HeadWeight::Bf16(_) | HeadWeight::Exl3(_) => Err(EngError::Checkpoint(
                "step weights: the head of this load is not a quantized matrix".to_owned(),
            )),
        }
    }

    /// The defined read-back point: queue the pair copy and its event, read the logits row when
    /// wanted (this call waits for the stream), then read the pair (its event is complete).
    fn read_back(&mut self, plan: &ReadbackPlan) -> Result<ResultPair, EngError> {
        if plan.shortlist {
            self.context.shortlist(
                &self.stream,
                ShortlistCall {
                    logits: self.view(Slot::Logits)?,
                    scratch: self.view(Slot::ArgmaxPart)?,
                    n: self.widths.verdict,
                },
            )?;
        }
        let pair_range = self.state.word(Word::ResultPair)?;
        self.context
            .copy_word_to_host(&self.stream, pair_range, &self.word)?;
        self.stream.record(&self.event)?;
        if let Some(bytes) = plan.logits_bytes {
            let row = self.view(Slot::Logits)?;
            let out = &mut self.logits_bytes[..bytes];
            row.buffer().read_at(&self.stream, row.offset(), out)?;
        }
        if plan.shortlist {
            let part = self.view(Slot::ArgmaxPart)?;
            let offset = part
                .offset()
                .checked_add(shortlist::RESULT_OFFSET)
                .ok_or_else(|| {
                    EngError::MemoryPlan("shortlist readback offset overflows".to_owned())
                })?;
            part.buffer()
                .read_at(&self.stream, offset, &mut self.shortlist_bytes)?;
        }
        let value = self.word.read_after(&self.event)?;
        Ok(ResultPair {
            token: value.token,
            status: StepStatus(value.status),
        })
    }

    /// Returns None before any step when the request needs the full host sampler.
    pub(super) fn try_device_shortlist_token(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        sampler_vocab: usize,
    ) -> Result<Option<u32>, EngError> {
        if !self.shortlist_on
            || !device_sampling_supported(
                sampler.params(),
                self.sync_each,
                sampler_vocab == self.widths.verdict,
            )
        {
            return Ok(None);
        }
        if sampler.params().is_greedy() {
            let selected = self.run_token(token, false)?;
            sampler.record_device_greedy(selected);
            return Ok(Some(selected));
        }
        self.run_token_output(token, false, true)?;
        let candidates = self.shortlist_values()?;
        sampler
            .sample_shortlist(&candidates, sampler_vocab)
            .map(Some)
    }

    fn shortlist_values(&self) -> Result<Vec<(u32, f32)>, EngError> {
        let bytes = &self.shortlist_bytes;
        if bytes[160..164] != [0; 4] {
            return Err(EngError::Backend {
                operation: "sample",
                message: "device shortlist found non-finite logits".to_owned(),
            });
        }
        Ok(bytes[..self.widths.verdict.min(shortlist::K) * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|entry| {
                (
                    u32::from_le_bytes(entry[4..8].try_into().expect("candidate id word")),
                    f32::from_le_bytes(entry[..4].try_into().expect("candidate logit word")),
                )
            })
            .collect())
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

#[cfg(test)]
#[path = "shortlist_bench.rs"]
mod shortlist_bench;

#[path = "prefix_runtime.rs"]
mod prefix_runtime;

#[path = "krow_graph.rs"]
mod krow_graph;

#[path = "krow_step.rs"]
mod krow_step;

#[cfg(test)]
pub(super) use krow_graph::GraphStats;
#[cfg(test)]
pub(super) use krow_step::{Injection, LayerTap};

#[path = "rollback_runtime.rs"]
mod rollback_runtime;

#[path = "window_gate.rs"]
mod window_gate;

#[cfg(test)]
#[path = "window_gate_device_tests.rs"]
mod window_gate_device_tests;

#[cfg(test)]
#[path = "prefix_quality_device_tests.rs"]
mod prefix_quality;

impl Drop for QwenResidentStep<'_> {
    fn drop(&mut self) {
        // Work may still be queued behind the last token. A failed synchronize latches the backend,
        // and every owner below checks the latch before it frees, so the error needs no other
        // handling here and `drop` must not panic.
        let _synced = self.stream.synchronize();
    }
}
