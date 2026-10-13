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
    dims::StepOptions,
    mixer::{BlockIo, MixerParts, MixerViews, Mixers},
    mixer_weights::MixerWeights,
    moe::{
        ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, Packs, RouteBufs, run_moe_block,
        tiered::TieredScratch,
    },
    readback::{
        ReadbackPlan, SYNC_EACH_VAR, captured_report, check_mode, eager_report, parse_sync_each,
        refused_report,
    },
    resources::{
        EmbedWeight, GlobalSpecs, GlobalWeights, HeadWeight, MoeSpecs, STEP_ROWS, StepResources,
        f32_bytes, moe_dims, moe_layer_parts, moe_layer_weights,
    },
    rules::{QwenRules, StepWidths, check_token_id, validate_step_inputs},
    run_report::{CaptureOutcome, RunReport},
    state::{ResidentState, StepField, export::ExportedState},
    state_layout::Word,
    step_ops::{StepOp, final_resid, resid_in, resid_out, token_ops},
    step_plan::Slot,
    tiered_step::{LayerParts, TieredRuntime},
    verdict::{
        Decision, FailureAction, action_for, bad_token_error, decide, entered_after_failed_sync,
    },
};
use crate::{
    EngError,
    family::project::cuda::{CudaWeights, WeightLayout},
    family::qwen36::tiered::{host::TieredHost, report::TokenReport},
    kernels::cuda::{
        CapturePlan, CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, ExpertBank,
        GraphExec, MatmulIo, PinnedWord, Projection,
        resident::{
            add_norm_pack::AddNormCall,
            add_rmsnorm_q8::AddNormQ8Call,
            embed::{EmbedCall, EmbedNativeCall},
            matvec_q::MatvecCall,
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
    /// The bytes of the last logits row read back (little-endian f32).
    logits_bytes: Vec<u8>,
    report: RunReport,
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
            views: MixerViews::new(&res.plan, &res.arena, &res.geometry, options.layout)?,
            layout: options.layout,
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
        Ok(Self {
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
            logits_bytes: vec![0; f32_bytes(vocab)?],
            report: RunReport::default(),
        })
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

    /// The report of the last SUCCESSFUL token: after a failed token it still holds the previous
    /// token's numbers.
    pub(crate) fn last_report(&self) -> RunReport {
        self.report
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

    /// Zeroes the session state (see `ResidentState::reset`) and leaves it `Valid`; only a device
    /// write that failed in the middle of the reset leaves it `Poisoned`.
    pub(crate) fn reset_state(&mut self) -> Result<(), EngError> {
        self.state.reset(&self.stream)
    }

    /// The committed state as a host value (checkpoint store, decision 0036).
    pub(crate) fn export_state(&self) -> Result<ExportedState, EngError> {
        self.state.export_state(&self.stream)
    }

    /// Replaces the committed state.
    pub(crate) fn import_state(&mut self, exported: &ExportedState) -> Result<(), EngError> {
        self.state.import_state(&self.stream, exported)
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
        self.report = match reported_outcome(how, self.refused) {
            CaptureOutcome::Eager => eager_report(&plan),
            CaptureOutcome::Captured => captured_report(&plan),
            CaptureOutcome::RefusedFellBackToEager => refused_report(&plan),
        };
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
