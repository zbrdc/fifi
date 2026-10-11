//! The resident Q36 decoder (packet Q36-D): weights, session budget, and the step entry point.
//!
//! `ResidentDecoder` runs one token per `step` in a `StepMode`. It owns the adapter and the weights.
//! It holds no session: the caller owns each `ResidentState` and passes it to `step`.

use super::moe::Routing;
use super::prefill;
use super::state::{ResidentSnapshot, ResidentState};
use super::step;
use super::weights::ResidentWeights;
use crate::{
    EngError, family::qwen36::host_load::HostWeights, kernels::MlxKernels, sched::SessionBudget,
    weights::StreamedCheckpoint,
};

/// What one step computes and reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StepMode {
    /// Greedy decoding: the head runs and the token id is read back.
    Generate,
    /// A prompt token whose logits are unused: no head, no readback after `evaluate`.
    Prompt,
    /// `Generate` plus the logits, for a sampler.
    Logits,
    /// Parity runs only: `Logits` plus the residual stream after every layer.
    Record,
}

/// Host results of one step.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StepResult {
    /// The greedy next token, lowest index on ties. `None` in `Prompt`.
    pub(super) token: Option<u32>,
    /// Next-token logits `[vocab]`. `Logits` and `Record` only.
    pub(super) logits: Option<Vec<f32>>,
    /// The `[hidden]` residual stream after each layer, as the host reference records it. `Record` only.
    pub(super) layer_out: Option<Vec<Vec<f32>>>,
    /// The experts of each layer in rank order, `top_k` per layer. `Record` only.
    pub(super) routes: Option<Vec<Vec<u32>>>,
}

/// The default context limit of decision 0046 item 6.
pub(super) const DEFAULT_CONTEXT_LIMIT: u32 = 32_768;

/// The retire cap of the decoder's adapter: 4 GiB. With device routing a token builds one graph
/// and every dropped temporary stays retired until the token's `evaluate`. The default cap of
/// 64 MiB would force a flush synchronization inside the token (decision 0039 retire list). The
/// retired arrays are inputs of the pending graph, so the flush would free nothing.
pub(super) const DECODE_RETIRE_CAP: u64 = 4 << 30;

/// The default rows of one prefill chunk (plan item Q36-E: 256 to 512 first).
pub(super) const DEFAULT_CHUNK_ROWS: u32 = 512;

/// The most rows that `set_chunk_rows` accepts.
pub(super) const MAX_CHUNK_ROWS: u32 = 2_048;

/// One resident Q36 model on its engine thread.
///
/// Field order is drop order: the adapter drops first, then the weights and the pages under them.
pub(super) struct ResidentDecoder<'c> {
    pub(super) kernels: MlxKernels,
    pub(super) ckpt: &'c StreamedCheckpoint,
    pub(super) weights: ResidentWeights,
    /// The engine's session budget. Each step charges its cache growth here. The cap is `u64::MAX`
    /// until the executor sets the admitted value.
    pub(super) budget: SessionBudget,
    /// The first position that a step refuses with `ContextLimit`.
    pub(super) context_limit: u32,
    /// The most rows of one prefill chunk. `prefill` splits a longer prompt.
    pub(super) chunk_rows: u32,
    /// How the layers pick their experts. `Device` unless a parity run asks for `Host`.
    pub(super) routing: Routing,
}

impl<'c> ResidentDecoder<'c> {
    /// Sets the retire cap to `DECODE_RETIRE_CAP`, imports the weights (see
    /// `ResidentWeights::import`, with F32 scales and biases), and consumes `host`. The routing is
    /// `Routing::Device`.
    pub(super) fn new(
        mut kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
    ) -> Result<Self, EngError> {
        kernels.set_retire_cap(DECODE_RETIRE_CAP)?;
        let weights = ResidentWeights::import(&mut kernels, ckpt, host)?;
        Ok(Self::with_weights(kernels, ckpt, weights, context_limit))
    }

    /// Like `new`, with every scale and bias kept as stored (BF16; see
    /// `ResidentWeights::import_bf16_scales`). Qualification tests only.
    #[cfg(all(test, feature = "native-fault-qualification"))]
    pub(super) fn new_bf16_scales(
        mut kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        context_limit: u32,
    ) -> Result<Self, EngError> {
        kernels.set_retire_cap(DECODE_RETIRE_CAP)?;
        let weights = ResidentWeights::import_bf16_scales(&mut kernels, ckpt, host)?;
        Ok(Self::with_weights(kernels, ckpt, weights, context_limit))
    }

    fn with_weights(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        weights: ResidentWeights,
        context_limit: u32,
    ) -> Self {
        Self {
            kernels,
            ckpt,
            weights,
            budget: SessionBudget::new(u64::MAX),
            context_limit,
            chunk_rows: DEFAULT_CHUNK_ROWS,
            routing: Routing::default(),
        }
    }

    /// Selects how the next steps route. The parity gate runs `Host` as the reference. The choice
    /// applies to every state of this decoder.
    pub(super) fn set_routing(&mut self, routing: Routing) {
        self.routing = routing;
    }

    /// Replaces the session budget. Tests use small caps so that a step can be denied. Call it before
    /// any state exists: permits of earlier steps charge the old budget.
    #[cfg(test)]
    pub(super) fn set_session_budget(&mut self, budget: SessionBudget) {
        self.budget = budget;
    }

    /// Bytes charged to the session budget by the committed steps.
    #[cfg(test)]
    pub(super) fn session_used(&self) -> u64 {
        self.budget.used()
    }

    /// Sets the most rows of one prefill chunk. A value of zero or above `MAX_CHUNK_ROWS` returns
    /// `BadRequest` and changes nothing.
    pub(super) fn set_chunk_rows(&mut self, rows: u32) -> Result<(), EngError> {
        if rows == 0 || rows > MAX_CHUNK_ROWS {
            return Err(EngError::BadRequest(format!(
                "a prefill chunk holds between 1 and {MAX_CHUNK_ROWS} rows"
            )));
        }
        self.chunk_rows = rows;
        Ok(())
    }

    /// Imports a zero state with one synchronization.
    pub(super) fn zero_state(&mut self) -> Result<ResidentState, EngError> {
        ResidentState::zero(&mut self.kernels, &self.weights)
    }

    /// Replaces `state` with a zero state. On error `state` is unchanged.
    pub(super) fn reset(&mut self, state: &mut ResidentState) -> Result<(), EngError> {
        state.reset(&mut self.kernels, &self.weights)
    }

    /// Saves `state` as a snapshot that shares its arrays and charges the session budget. See
    /// `ResidentState::snapshot`.
    pub(super) fn snapshot(&mut self, state: &ResidentState) -> Result<ResidentSnapshot, EngError> {
        state.snapshot(&mut self.budget)
    }

    /// Replaces `state` with `snapshot`. See `ResidentState::restore`.
    pub(super) fn restore(
        &mut self,
        state: &mut ResidentState,
        snapshot: &ResidentSnapshot,
    ) -> Result<(), EngError> {
        state.restore(snapshot, &self.weights, &mut self.budget)
    }

    /// Runs one token. See `step::step`.
    pub(super) fn step(
        &mut self,
        state: &mut ResidentState,
        token: u32,
        mode: StepMode,
    ) -> Result<StepResult, EngError> {
        step::step(self, state, token, mode)
    }

    /// Runs a prompt in chunks. See `prefill::step::prefill`.
    ///
    /// The chunks run in `Prompt` mode except the last, which runs in `mode`. The result is that of
    /// the last chunk: the head reads the last row of the prompt.
    ///
    /// A prompt of one token runs as a chunk of one row, not as `step`. Every chunk routes on the
    /// host whatever `Routing` the decoder holds, so the result has `routes: None` in every mode.
    /// `step` returns `Some(routes)` in `Record` mode.
    ///
    /// The host checks of the whole prompt (a poisoned state, an empty prompt, and the context
    /// limit) run first and change nothing on error. Each chunk repeats the host checks and takes
    /// its growth permit before it marks the state poisoned. A failed check or a denied permit
    /// (`MemoryPlan`) in a later chunk leaves the state clean: the chunks before it stay
    /// published, `state.pos()` is the position after the last of them, and the session budget
    /// holds their charge. A caller that retries must continue from `state.pos()` or `reset` the
    /// state. An error after the state is marked poisoned (a native failure, or a token outside
    /// the vocabulary, which the read of the embedding rows finds) poisons the state, drops its
    /// layer states, and releases the permit of that chunk. Only `reset` recovers a poisoned state.
    pub(super) fn prefill(
        &mut self,
        state: &mut ResidentState,
        tokens: &[u32],
        mode: StepMode,
    ) -> Result<StepResult, EngError> {
        prefill::step::prefill(self, state, tokens, mode)
    }
}
