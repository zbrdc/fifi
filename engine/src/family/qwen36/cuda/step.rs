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
//! The pin accessors, the warm-up token and the setting of `CaptureOutcome` belong to the slim
//! RD-11a (lane C); this packet always runs the body eagerly.
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
//! Drop order: the borrowed weights and views first, then the state, the pinned word, the event
//! and the stream; `Drop` synchronizes the stream first.
//!
//! Design guidance only, no code copied.

use super::{
    dims::StepOptions,
    mixer::{BlockIo, MixerParts, MixerViews, Mixers},
    mixer_weights::MixerWeights,
    moe::{ExpertBufs, Launch, MoeBuffers, MoeDims, MoeLayerWeights, RouteBufs, run_moe_block},
    readback::{ReadbackPlan, SYNC_EACH_VAR, eager_report, parse_sync_each},
    resources::{
        GlobalSpecs, GlobalWeights, MoeSpecs, STEP_ROWS, StepResources, f32_bytes, moe_dims,
        moe_layer_weights,
    },
    rules::{QwenRules, StepWidths, validate_step_inputs},
    run_report::RunReport,
    state::{ResidentState, StepField, export::ExportedState},
    state_layout::Word,
    step_ops::{StepOp, final_resid, resid_in, resid_out, token_ops},
    step_plan::Slot,
    verdict::{Decision, FailureAction, action_for, bad_token_error, decide},
};
use crate::{
    EngError,
    family::project::cuda::CudaWeights,
    kernels::cuda::{
        CudaContext, DeviceEvent, DeviceRange, DeviceStream, MatmulIo, PinnedWord, Projection,
        resident::{
            add_norm_pack::AddNormCall,
            embed::EmbedCall,
            step_params::{ArgmaxCall, ResultPair, StepParams, StepStatus},
        },
    },
};

const F32: usize = size_of::<f32>();

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
    // Declaration order is drop order: borrows first, the stream last.
    context: &'a CudaContext,
    res: &'a StepResources,
    mixers: Mixers<'a>,
    moe: Vec<MoeLayerWeights<'a>>,
    globals: GlobalWeights<'a>,
    moe_dims: MoeDims,
    state: ResidentState,
    word: PinnedWord,
    event: DeviceEvent,
    stream: DeviceStream,
    sync_each: bool,
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
    pub(crate) fn new(
        context: &'a CudaContext,
        res: &'a StepResources,
        weights: &'a CudaWeights,
        options: &StepOptions,
        rules: QwenRules,
    ) -> Result<Self, EngError> {
        let widths = validate_step_inputs(options, res.dims.shape.vocab, &rules)?;
        let sync_each = sync_each_from_env()?;
        let stream = context.stream()?;
        let state = ResidentState::allocate(context, &stream, &res.dims, options)?;
        let (event, word) = (context.event()?, context.pinned_word()?);
        let layers: Vec<usize> = (0..res.table.len()).collect();
        let parts = MixerParts {
            geometry: res.geometry,
            table: &res.table,
            weights: MixerWeights::new(&res.geometry, &res.table, weights, &layers)?,
            views: MixerViews::new(&res.plan, &res.arena, &res.geometry)?,
        };
        let specs = MoeSpecs::new(&res.dims)?;
        let moe = layers
            .iter()
            .map(|&layer| moe_layer_weights(&res.table, weights, &specs, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let vocab = res.dims.shape.vocab;
        Ok(Self {
            context,
            res,
            mixers: Mixers::new(context, parts)?,
            moe,
            globals: GlobalWeights::new(&res.names, &GlobalSpecs::new(&res.dims), weights)?,
            moe_dims: moe_dims(&res.dims),
            state,
            word,
            event,
            stream,
            sync_each,
            rules,
            widths,
            logits_bytes: vec![0; f32_bytes(vocab)?],
            report: RunReport::default(),
        })
    }

    /// The report of the last SUCCESSFUL token: after a failed token it still holds the previous
    /// token's numbers.
    pub(crate) fn last_report(&self) -> RunReport {
        self.report
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
                xf_moe: view(Slot::XfMoe)?,
                expert_gu: view(Slot::ExpertGu)?,
                xf_down: view(Slot::XfDown(0))?,
                expert_down: view(Slot::ExpertDown)?,
                shared_gu: view(Slot::SharedGu)?,
                xf_shared: view(Slot::XfShared)?,
                shared_out: view(Slot::SharedOut)?,
            },
        })
    }

    /// Runs one token: feeds `token` at the next position, optionally reads the logits row back,
    /// and returns the token the device argmax chose. The row is in [`Self::row_bytes`] afterwards
    /// when `want_logits` was true.
    pub(crate) fn run_token(&mut self, token: u32, want_logits: bool) -> Result<u32, EngError> {
        let plan = self.readback_plan(want_logits)?;
        // The start block is written at a committed boundary (the state is idle), then the step
        // is admitted; a refused write queues nothing.
        self.write_start(token)?;
        let admitted = self.state.enqueue_rows(STEP_ROWS)?;
        let mut entered = 0;
        if let Err(error) = self.run_body(admitted.admit_pos, &mut entered) {
            return Err(self.fail(entered, error));
        }
        let pair = self.read_back(&plan).inspect_err(|_| self.state.poison())?;
        let token = match decide(pair, self.widths.verdict) {
            Decision::BadToken(token) => {
                self.state.poison();
                return Err(bad_token_error(token));
            }
            Decision::Commit { token } => token,
        };
        // A non-zero status is refused here and poisons the state; the error has the class of
        // the wrapper (4, 8, 1, 2 in that order).
        self.state.commit_step(STEP_ROWS, pair.status.0)?;
        self.report = eager_report(&plan);
        Ok(token)
    }

    /// The read-back plan of one token. The row buffer is head-wide (the `lm_head` writes the
    /// padded head), so the plan takes the head width, not the sampler width of the verdict.
    fn readback_plan(&self, want_logits: bool) -> Result<ReadbackPlan, EngError> {
        ReadbackPlan::new(self.res.dims.shape.vocab, want_logits)
    }

    /// The start block of the step: the token, at the position of the committed rows.
    fn write_start(&mut self, token: u32) -> Result<(), EngError> {
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
            if self.sync_each {
                self.stream.synchronize()?;
            }
        }
        Ok(())
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
            StepOp::Embed => ctx.embed(
                s,
                EmbedCall {
                    weight: self.globals.embed,
                    ids: self.state.step_field(StepField::TokenId)?,
                    x: self.view(resid_in(0))?,
                    status: self.state.step_status()?,
                    m: STEP_ROWS,
                },
            ),
            StepOp::Mixer(layer) => {
                let io = BlockIo {
                    resid: self.view(resid_in(layer))?,
                    admit_pos,
                };
                self.mixers.run_block(s, &self.state, layer, io)
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
        self.context.add_norm_pack(
            &self.stream,
            AddNormCall {
                m: STEP_ROWS,
                n: self.res.geometry.hidden,
                eps: self.res.geometry.eps,
                h: self.view(final_resid(layers))?,
                delta: None,
                weight: norm.range(0, norm.bytes())?,
                xf: self.view(Slot::XfFinal)?,
                norm_out: None,
            },
        )
    }

    fn lm_head(&self) -> Result<(), EngError> {
        let head = self.globals.head;
        self.context.matmul(
            &self.stream,
            Projection {
                weight: head,
                first_row: 0,
                rows: head.n_rows(),
            },
            MatmulIo {
                xf: self.view(Slot::XfFinal)?,
                y: self.view(Slot::Logits)?,
                m: STEP_ROWS,
            },
        )
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
