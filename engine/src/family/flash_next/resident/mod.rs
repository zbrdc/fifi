//! Resident Flash-Next decode on MLX (decision 0039; `s1c-resident-decode.md`, in the ledger).
//!
//! Weights, constants, and session state stay resident on the engine thread. `ResidentDecoder` runs
//! one token per `step` in a `StepMode`. `fixtures` is portable so that Linux tests check the synthetic
//! checkpoint.

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod deltanet;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) mod executor;
#[cfg(test)]
mod fixtures;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod hyper;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod moe;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod ops;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod parity;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod ple;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod prefill;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod qsa;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod slots;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod state;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod step;
#[cfg(test)]
mod trace;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod weights;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use self::slots::{ExpertSlots, POOL_SETS, SetLayout};
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use self::state::ResidentState;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use self::weights::ResidentWeights;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use super::load::FlashWeights;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::{
    EngError,
    kernels::{HostPages, MlxKernels},
    sched::SessionBudget,
    weights::StreamedCheckpoint,
};

/// What one step computes and reads back (plan section 2.6, step 3).
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepMode {
    /// Greedy decoding: the head runs, and the token id is read back.
    Generate,
    /// Sampled decoding: the head runs and the next-token logits are read back. There is no
    /// `argmax` and no token. The host sampler chooses the token.
    Logits,
    /// A prompt token whose logits are unused: no head and no readback after `evaluate`.
    Prompt,
    /// Parity tests only: `Generate` plus the logits and every layer's post-MoE residual.
    Record,
}

/// Host results of one step.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, PartialEq)]
struct StepResult {
    /// The greedy next token, lowest index on ties. `None` in `Prompt` and `Logits`.
    token: Option<u32>,
    /// Next-token logits `[vocab]`. `Record` and `Logits` only.
    logits: Option<Vec<f32>>,
    /// The `[C * H]` residual after each layer's MoE, as the host `layer_out`. `Record` only.
    layer_out: Option<Vec<Vec<f32>>>,
}

/// One resident Flash-Next model on its engine thread.
///
/// Field order is drop order: the expert page pool drops after the adapter has completed its work.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
struct ResidentDecoder<'c> {
    kernels: MlxKernels,
    ckpt: &'c StreamedCheckpoint,
    weights: ResidentWeights,
    /// The fixed expert page pool: `POOL_SETS` sets sized for the largest layer (decision 0041).
    slots: ExpertSlots,
    /// The prefill page sets (decision 0044 item 5), open from `open_prefill` to `close_prefill`.
    /// They drop after the adapter like the decode pool.
    prefill_pages: Option<prefill::step::Open>,
    /// The engine's session budget. Each step charges its QSA growth here (S1-C decision 5).
    /// The cap is `u64::MAX` until the executor sets the admitted value.
    budget: SessionBudget,
    /// Tests only: the route recorder of the S3 trace (packet T0). `None` records nothing.
    #[cfg(test)]
    route_trace: Option<trace::RouteTrace>,
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl<'c> ResidentDecoder<'c> {
    /// Imports the weights with `layers + 2` synchronizations and maps the expert page pool.
    /// The caller may drop `flash` afterwards.
    fn new(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        flash: &FlashWeights,
    ) -> Result<Self, EngError> {
        Self::build(kernels, ckpt, flash, POOL_SETS)
    }

    /// `new` with `sets` page sets. Tests use smaller pools so that every release matters.
    #[cfg(test)]
    fn with_pool_sets(
        kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        flash: &FlashWeights,
        sets: usize,
    ) -> Result<Self, EngError> {
        Self::build(kernels, ckpt, flash, sets)
    }

    fn build(
        mut kernels: MlxKernels,
        ckpt: &'c StreamedCheckpoint,
        flash: &FlashWeights,
        sets: usize,
    ) -> Result<Self, EngError> {
        let weights = ResidentWeights::import(&mut kernels, flash)?;
        let top_k = u32::try_from(weights.shape.moe.top_k())
            .map_err(|_| EngError::BadRequest("top-k exceeds u32".into()))?;
        let page = HostPages::page_size();
        let mut set_bytes = 0;
        for layer in &weights.layers {
            set_bytes = set_bytes.max(SetLayout::new(ckpt, &layer.moe, top_k, page)?.end);
        }
        let slots = ExpertSlots::new(sets, set_bytes)?;
        Ok(Self {
            kernels,
            ckpt,
            weights,
            slots,
            prefill_pages: None,
            budget: SessionBudget::new(u64::MAX),
            #[cfg(test)]
            route_trace: None,
        })
    }

    /// Mapped bytes of the expert page pool, fixed at `new` (decision 0041 Memory). MLX active
    /// memory also counts the lent pages, so a report must not add both.
    fn pool_bytes(&self) -> u64 {
        self.slots.byte_len()
    }

    /// Replaces the session budget. Tests use small caps so that a step can be denied. Call it
    /// before any state exists: permits of earlier steps charge the old budget.
    #[cfg(test)]
    fn set_session_budget(&mut self, budget: SessionBudget) {
        self.budget = budget;
    }

    /// Bytes charged to the session budget by the committed steps.
    #[cfg(test)]
    fn session_used(&self) -> u64 {
        self.budget.used()
    }

    /// Imports a zero state with one synchronization.
    fn zero_state(&mut self) -> Result<ResidentState, EngError> {
        ResidentState::zero(&mut self.kernels, &self.weights)
    }

    /// Replaces `state` with a zero state. On error `state` is unchanged.
    fn reset(&mut self, state: &mut ResidentState) -> Result<(), EngError> {
        state.reset(&mut self.kernels, &self.weights)
    }

    /// Saves `state` as a snapshot that shares its arrays and charges the session budget. See
    /// `ResidentState::snapshot`.
    fn snapshot(&mut self, state: &ResidentState) -> Result<state::ResidentSnapshot, EngError> {
        state.snapshot(&mut self.budget)
    }

    /// Replaces `state` with `snapshot`. See `ResidentState::restore`.
    fn restore(
        &mut self,
        state: &mut ResidentState,
        snapshot: &state::ResidentSnapshot,
    ) -> Result<(), EngError> {
        state.restore(snapshot, &self.weights, &mut self.budget)
    }

    /// Runs one token. See `step::step`.
    fn step(
        &mut self,
        state: &mut ResidentState,
        token: u32,
        mode: StepMode,
    ) -> Result<StepResult, EngError> {
        step::step(self, state, token, mode)
    }

    /// Maps the prefill page sets for chunks of up to `max_tokens` tokens, in one wave per layer.
    /// Call it once per prompt, before its first `prefill`. See `prefill::step::open`.
    fn open_prefill(&mut self, max_tokens: u32) -> Result<(), EngError> {
        prefill::step::open(self, max_tokens, None)
    }

    /// Like `open_prefill`, with expert waves of at most `wave_cap` experts per layer.
    fn open_prefill_waves(&mut self, max_tokens: u32, wave_cap: u32) -> Result<(), EngError> {
        prefill::step::open(self, max_tokens, Some(wave_cap))
    }

    /// Releases and drops the prefill page sets. Call it after the last `prefill` of a prompt.
    fn close_prefill(&mut self) {
        prefill::step::close(self);
    }

    /// Runs `tokens` as a prefill and publishes the state after them. Up to the dense QSA limit
    /// the tokens run as one chunk, and the rest run as one-token `Prompt` steps on the sparse
    /// path. The last token runs in `mode`: `Prompt` runs no head, `Generate` also returns the
    /// greedy token, and `Record` adds the logits and the post-MoE residual of every layer, all
    /// for the last row only. See `prefill::step::prefill`.
    fn prefill(
        &mut self,
        state: &mut ResidentState,
        tokens: &[u32],
        mode: StepMode,
    ) -> Result<StepResult, EngError> {
        prefill::step::prefill(self, state, tokens, mode)
    }
}
