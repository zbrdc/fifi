//! Resident Q36 session state (packet Q36-D), with the step's check, poison, and commit.
//!
//! DeltaNet layers keep a recurrent state `[VH, VD, KD]` and a convolution ring `[CD, CK - 1]`.
//! Attention layers keep a key-value cache in preallocated capacity buffers written in place
//! (packets Q36-KV-INTEGRATE and KV8-INTEGRATE): in the F32 control `[1, KVH, cap, D]` per key
//! and value tensor, in the Q8 form the codes `[1, KVH, cap, D/4]` beside scales and biases
//! `[1, KVH, cap, D/64]` of the activation dtype, `len` valid positions, growth in `KV_STEP`
//! quanta. The growth charge covers the cache only, as in the Flash-Next decoder (decision
//! 0039 amendment), and stays LOGICAL — `len` positions of growth bytes per layer, from the
//! cache form and the activation dtype — so the capacity slack, bounded by one `KV_STEP` of
//! growth bytes per attention layer (256 * 4,096 * 10 = 10,485,760 bytes on the real model in
//! the F32 form; 256 * 1,088 * 10 = 2,785,280 at Q8 with BF16 scales), never enters a permit;
//! `kv_storage_bytes` reports the stored bytes. The slack bound excludes the growth TRANSIENT —
//! the old buffer, its slice copy, and the new buffer alive together at a growth, about 100 MB
//! per layer at depth 8,192 in the F32 form — which the session reserve margin and the retire
//! cap cover. The fixed DeltaNet state of 65,863,680 bytes is not charged by the steps; a
//! snapshot charges it once with its entry (decision 0036, addendum 1).

use super::attn::{KvCache, growth_bytes};
use super::kv_plan::{KV_STEP, session_bytes};
use super::ops::{Bytes, dim, short_import, zeros};
use super::weights::{RMixer, ResidentWeights};
use crate::{
    EngError,
    kernels::MlxKernels,
    sched::{GrowthPermit, SessionBudget},
    tensor::{ExecutionTensor, Tensor},
};

/// Resident state of one layer.
pub(super) enum LayerState {
    /// DeltaNet state `[VH, VD, KD]` and convolution ring `[CD, CK - 1]`.
    Delta {
        rec: ExecutionTensor,
        ring: ExecutionTensor,
    },
    /// Dense attention cache.
    Attn(KvCache),
}

impl LayerState {
    /// Returns a layer state that shares every array with `self`. It copies no array.
    fn share(&self) -> Self {
        match self {
            Self::Delta { rec, ring } => Self::Delta {
                rec: rec.share(),
                ring: ring.share(),
            },
            Self::Attn(cache) => Self::Attn(cache.share()),
        }
    }
}

/// Bytes that one more token adds to the stored caches of all attention layers, from the cache
/// form and the activation dtype (packet KV8-INTEGRATE item 5): the F32 control charges
/// `layers * KVH * D * 2 * 4` and the Q8 form the codes plus the activation-dtype scales and
/// biases.
pub(super) fn kv_growth_bytes(weights: &ResidentWeights) -> Result<u64, EngError> {
    let layers = weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.mixer, RMixer::Attn(_)))
        .count();
    let per_layer = kv_growth_per_layer_bytes(weights)?;
    u64::try_from(layers)
        .ok()
        .and_then(|layers| per_layer.checked_mul(layers))
        .ok_or_else(|| EngError::MemoryPlan("kv growth bytes overflow u64".into()))
}

/// Bytes that one more token adds to the cache of ONE attention layer, from the cache form and
/// the activation dtype: the logical charge multiplies it by the attention layer count
/// (`kv_bytes`), so the state keeps this per-layer figure and never the all-layer sum.
pub(super) fn kv_growth_per_layer_bytes(weights: &ResidentWeights) -> Result<u64, EngError> {
    let activation = crate::family::qwen36::settings::Activation::of_dtype(weights.consts.act)
        .ok_or_else(|| EngError::Backend {
            operation: "qwen36",
            message: "the decoder holds no known activation dtype".into(),
        })?;
    growth_bytes(&weights.shape, weights.kv, activation)
}

/// Bytes that a chunk of `rows` tokens adds to the stored caches: `rows` positions of
/// `kv_growth_bytes`. It makes no native call.
pub(super) fn chunk_growth_bytes(weights: &ResidentWeights, rows: u32) -> Result<u64, EngError> {
    kv_growth_bytes(weights)?
        .checked_mul(u64::from(rows))
        .ok_or_else(|| EngError::MemoryPlan("kv chunk growth bytes overflow u64".into()))
}

/// The LOGICAL bytes of the attention caches in `layers` (packet Q36-KV-INTEGRATE): one
/// `kv_plan::session_bytes` charge — the attention layer count, the valid length they hold, and
/// the PER-LAYER growth — because every attention layer of a committed, snapshotted, or restored
/// state holds the same valid length, the position (`check_step` before a step,
/// `check_saved_layer` before a restore). The charge of a state at `pos` is therefore exactly
/// `pos * kv_growth_bytes(weights)`, the formula the step and chunk permits hold; the capacity
/// slack of the buffers never enters it, and the slack of a whole session stays below one
/// `KV_STEP` of growth bytes per attention layer (256 * 4,096 * 10 = 10,485,760 bytes on the
/// real model). Attention layers of UNEQUAL valid length give `u64::MAX`, so a state whose layers
/// disagree fails the permit equality check instead of passing on the last layer's length. Use
/// `kv_storage_bytes` for the stored bytes.
fn kv_bytes(layers: &[LayerState], per_layer: u64) -> u64 {
    let mut attn_layers = 0;
    let mut len: Option<usize> = None;
    for layer in layers {
        if let LayerState::Attn(cache) = layer {
            attn_layers += 1;
            match len {
                None => len = Some(cache.len),
                // Attention layers of unequal valid length cannot be a committed state: the
                // charge saturates, so the permit equality check of `commit_rows` fails.
                Some(first) if first != cache.len => return u64::MAX,
                Some(_) => {}
            }
        }
    }
    session_bytes(attn_layers, len.unwrap_or(0), per_layer).unwrap_or(u64::MAX)
}

/// Bytes that the attention cache tensors of `layers` store: the capacity buffers summed from
/// the tensors' own sizes, slack included (packet Q36-KV-INTEGRATE). Reports and tests read this;
/// the admission charge is `kv_bytes`.
pub(super) fn kv_storage_bytes(layers: &[LayerState]) -> u64 {
    layers
        .iter()
        .map(|layer| match layer {
            LayerState::Attn(cache) => cache.byte_len(),
            LayerState::Delta { .. } => 0,
        })
        .fold(0, u64::saturating_add)
}

/// Bytes of the DeltaNet states and convolution rings: the part of a state that does not grow with
/// the position. On the real model this is 65,863,680 bytes (30 layers of `[32, 128, 128]` states
/// and `[8192, 3]` rings).
fn fixed_bytes(layers: &[LayerState]) -> Result<u64, EngError> {
    let mut total = Bytes::default();
    for layer in layers {
        if let LayerState::Delta { rec, ring } = layer {
            total.tensor(rec)?;
            total.tensor(ring)?;
        }
    }
    Ok(total.total())
}

/// Checks one saved layer against the weights' layout at `pos`: the DeltaNet state and ring shapes,
/// and the attention cache length and shapes. It makes no native call. A mismatch returns
/// `BadRequest`.
fn check_saved_layer(
    weights: &ResidentWeights,
    mixer: &RMixer,
    saved: &LayerState,
    pos: u32,
) -> Result<(), EngError> {
    let d = &weights.shape.delta;
    let a = &weights.shape.attn;
    match (mixer, saved) {
        (RMixer::Delta(_), LayerState::Delta { rec, ring }) => {
            let rec_shape = [dim(d.value_heads)?, dim(d.value_dim)?, dim(d.key_dim)?];
            let ring_shape = [dim(d.conv_dim())?, dim(d.conv_kernel - 1)?];
            if rec.shape() == rec_shape.as_slice() && ring.shape() == ring_shape.as_slice() {
                Ok(())
            } else {
                Err(EngError::BadRequest(
                    "snapshot state shape does not match the model layout".into(),
                ))
            }
        }
        (RMixer::Attn(_), LayerState::Attn(cache)) => {
            // The form and the shapes of every buffer, with no native call (packet
            // KV8-INTEGRATE: a saved cache of another form or shape is refused).
            let form_ok = cache
                .check_buffers(dim(a.heads_kv)?, dim(a.head_dim)?, weights.kv)
                .is_ok();
            if usize::try_from(pos).ok() == Some(cache.len) && form_ok {
                Ok(())
            } else {
                Err(EngError::BadRequest(
                    "snapshot cache does not hold the snapshot position in the model layout".into(),
                ))
            }
        }
        _ => Err(EngError::BadRequest(
            "snapshot layer does not match its weights".into(),
        )),
    }
}

/// A committed `ResidentState` held by shared handles (decision 0036, addendum 1). Its arrays
/// stay unchanged: a step on a cache buffer this snapshot shares keeps the handle count above
/// one, so `kv_append` copies the buffer once instead of writing it in place, and a step on a
/// buffer nothing else holds writes in place without touching this snapshot. One growth permit
/// charges all its bytes to the session budget and releases them when the snapshot drops. It
/// holds no logits (addendum 1 item 3).
pub(super) struct ResidentSnapshot {
    layers: Vec<LayerState>,
    pos: u32,
    charge: GrowthPermit,
}

impl ResidentSnapshot {
    /// Position of the next token after the saved prefix.
    pub(super) fn pos(&self) -> u32 {
        self.pos
    }

    /// Bytes that the snapshot charges to the session budget.
    pub(super) fn charged_bytes(&self) -> u64 {
        self.charge.retained_bytes()
    }
}

/// Resident state of one sequence.
pub(super) struct ResidentState {
    layers: Vec<LayerState>,
    pos: u32,
    poisoned: bool,
    /// One growth permit per committed step. Together they hold the logical cache bytes.
    /// `reset` and drop release them. A failed step keeps the earlier permits until `reset`.
    permits: Vec<GrowthPermit>,
    /// The LOGICAL growth bytes of ONE position of ONE attention layer
    /// (`attn::growth_bytes(&weights.shape, weights.kv, activation)`), fixed at `zero`. `kv_bytes`
    /// multiplies it by the
    /// attention layer count and the valid length, so a state at `pos` charges exactly
    /// `pos * kv_growth_bytes(weights)` — the all-layer growth — which the permits hold (packet
    /// Q36-KV-INTEGRATE fix1 item 1).
    kv_growth_per_layer: u64,
    /// The growth quantum of the attention caches, in positions: `KV_STEP` for `zero` and the
    /// test step for `zero_with_step`. `reset` keeps it (fix1 item 3.5).
    kv_step: usize,
}

impl ResidentState {
    /// Imports a zero state with one `import_many` (one synchronization, outside the per-token
    /// budget). The attention caches grow in `KV_STEP` quanta.
    pub(super) fn zero(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
    ) -> Result<Self, EngError> {
        Self::zero_with(kernels, weights, KV_STEP)
    }

    /// Imports a zero state whose attention caches grow in `kv_step`-position quanta. Tests use
    /// a small step so a run crosses a growth boundary within a few tokens.
    #[cfg(test)]
    pub(super) fn zero_with_step(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        kv_step: usize,
    ) -> Result<Self, EngError> {
        Self::zero_with(kernels, weights, kv_step)
    }

    /// The one constructor body of `zero` and `zero_with_step`.
    fn zero_with(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        kv_step: usize,
    ) -> Result<Self, EngError> {
        let d = &weights.shape.delta;
        let rec = vec![dim(d.value_heads)?, dim(d.value_dim)?, dim(d.key_dim)?];
        let ring = vec![dim(d.conv_dim())?, dim(d.conv_kernel - 1)?];
        let mut sources: Vec<Tensor> = Vec::new();
        for layer in &weights.layers {
            if matches!(layer.mixer, RMixer::Delta(_)) {
                sources.push(zeros(rec.clone())?);
                sources.push(zeros(ring.clone())?);
            }
        }
        let refs: Vec<&Tensor> = sources.iter().collect();
        let mut got = if refs.is_empty() {
            Vec::new().into_iter()
        } else {
            kernels.import_many(&refs)?.into_iter()
        };
        let mut next = || got.next().ok_or_else(short_import);
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(match layer.mixer {
                RMixer::Delta(_) => LayerState::Delta {
                    rec: next()?,
                    ring: next()?,
                },
                RMixer::Attn(_) => LayerState::Attn(KvCache::with_form(weights.kv, kv_step)),
            });
        }
        Ok(Self {
            layers,
            pos: 0,
            poisoned: false,
            permits: Vec::new(),
            kv_growth_per_layer: kv_growth_per_layer_bytes(weights)?,
            kv_step,
        })
    }

    /// Position of the next token.
    pub(super) fn pos(&self) -> u32 {
        self.pos
    }

    /// True after a failed step, until `reset`.
    pub(super) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Test-only: leaves the state as a failed step leaves it — poisoned, its layer states taken
    /// out, and the permits of its committed steps kept — without a native failure, which decision
    /// 0032 makes terminal for the whole process.
    #[cfg(test)]
    pub(super) fn poison_for_test(&mut self) {
        self.poisoned = true;
        self.layers.clear();
    }

    /// Layer states in model order (test observer).
    #[cfg(test)]
    pub(super) fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// Host checks before a step. They make no native call. On error the state is unchanged and not
    /// poisoned. A poisoned state returns `BadRequest`. A position at `context_limit` returns
    /// `ContextLimit`. A layer kind, cache length, or attention buffer shape — the `[1, KVH, cap, D]`
    /// pair with `cap >= len`, the same check as `KvCache::check_buffers` — that differs from the
    /// weights or the position returns `BadRequest`, so every host validation of the step runs
    /// before the first native call and before any buffer is consumed (fix1 item 3.3).
    pub(super) fn check_step(
        &self,
        weights: &ResidentWeights,
        context_limit: u32,
    ) -> Result<(), EngError> {
        if self.poisoned {
            return Err(EngError::BadRequest(
                "resident state needs a reset after a failed step".into(),
            ));
        }
        if self.pos >= context_limit {
            return Err(EngError::ContextLimit(format!(
                "the context holds {context_limit} positions"
            )));
        }
        if weights.layers.len() != self.layers.len() {
            return Err(EngError::BadRequest(
                "model layer count does not match the state".into(),
            ));
        }
        for (layer, state) in weights.layers.iter().zip(&self.layers) {
            match (&layer.mixer, state) {
                (RMixer::Delta(_), LayerState::Delta { .. }) => {}
                (RMixer::Attn(_), LayerState::Attn(cache)) => {
                    if usize::try_from(self.pos).ok() != Some(cache.len) {
                        return Err(EngError::BadRequest(
                            "layer state does not match its weights or the position".into(),
                        ));
                    }
                    cache.check_buffers(
                        dim(weights.shape.attn.heads_kv)?,
                        dim(weights.shape.attn.head_dim)?,
                        weights.kv,
                    )?;
                }
                _ => {
                    return Err(EngError::BadRequest(
                        "layer state does not match its weights or the position".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Host checks before a chunk of `rows` tokens. They make no native call. On error the state is
    /// unchanged and not poisoned. The checks of `check_step` apply. An empty chunk returns
    /// `BadRequest`. A chunk whose last row passes `context_limit` returns `ContextLimit`, also when
    /// the one-token path allows the first position.
    pub(super) fn check_chunk(
        &self,
        weights: &ResidentWeights,
        context_limit: u32,
        rows: u32,
    ) -> Result<(), EngError> {
        self.check_step(weights, context_limit)?;
        if rows == 0 {
            return Err(EngError::BadRequest(
                "a prefill chunk needs at least one token".into(),
            ));
        }
        match self.pos.checked_add(rows) {
            Some(end) if end <= context_limit => Ok(()),
            _ => Err(EngError::ContextLimit(format!(
                "a chunk of {rows} tokens passes the context of {context_limit} positions"
            ))),
        }
    }

    /// Poisons the state and takes its layer states out. The step drops each old layer as soon as its
    /// replacement graph exists. Only `commit` or `reset` clears the poison.
    pub(super) fn begin(&mut self) -> Vec<LayerState> {
        self.poisoned = true;
        std::mem::take(&mut self.layers)
    }

    /// Bytes held by the permits of the committed steps.
    fn charged_bytes(&self) -> Result<u64, EngError> {
        self.permits
            .iter()
            .try_fold(0u64, |total, permit| {
                total.checked_add(permit.retained_bytes())
            })
            .ok_or_else(|| EngError::MemoryPlan("session charge overflows u64".into()))
    }

    /// Publishes a one-token step. See `commit_rows`.
    pub(super) fn commit(
        &mut self,
        layers: Vec<LayerState>,
        permit: GrowthPermit,
    ) -> Result<(), EngError> {
        self.commit_rows(1, layers, permit)
    }

    /// Publishes a step of `rows` tokens after its `evaluate` and readbacks succeeded. It keeps
    /// `permit` only if the total charge equals the LOGICAL cache bytes (`len` positions of
    /// growth per layer; packet Q36-KV-INTEGRATE). Otherwise it returns `MemoryPlan`, drops
    /// `permit`, and changes nothing.
    pub(super) fn commit_rows(
        &mut self,
        rows: u32,
        layers: Vec<LayerState>,
        permit: GrowthPermit,
    ) -> Result<(), EngError> {
        let pos = self
            .pos
            .checked_add(rows)
            .ok_or_else(|| EngError::ContextLimit("position overflows u32".into()))?;
        let charged = self
            .charged_bytes()?
            .checked_add(permit.retained_bytes())
            .ok_or_else(|| EngError::MemoryPlan("session charge overflows u64".into()))?;
        let logical = kv_bytes(&layers, self.kv_growth_per_layer);
        if logical != charged {
            return Err(EngError::MemoryPlan(format!(
                "session charge {charged} bytes differs from the logical cache bytes {logical}"
            )));
        }
        self.permits.push(permit);
        self.layers = layers;
        self.pos = pos;
        self.poisoned = false;
        Ok(())
    }

    /// Shares the committed state as a snapshot. It makes no native call and copies no array. One
    /// permit charges the cache bytes and the fixed DeltaNet bytes to `budget`. A poisoned state
    /// returns `BadRequest`. A denied permit returns `MemoryPlan`. On error the state and the budget
    /// are unchanged.
    pub(super) fn snapshot(
        &self,
        budget: &mut SessionBudget,
    ) -> Result<ResidentSnapshot, EngError> {
        if self.is_poisoned() {
            return Err(EngError::BadRequest(
                "a poisoned resident state cannot be saved".into(),
            ));
        }
        let bytes = kv_bytes(&self.layers, self.kv_growth_per_layer)
            .checked_add(fixed_bytes(&self.layers)?)
            .ok_or_else(|| EngError::MemoryPlan("snapshot bytes overflow u64".into()))?;
        let charge = budget.acquire_growth(bytes)?;
        Ok(ResidentSnapshot {
            layers: self.layers.iter().map(LayerState::share).collect(),
            pos: self.pos,
            charge,
        })
    }

    /// Replaces this state with the snapshot's state. It makes no native call and copies no array.
    /// The state releases its old permits first and then keeps one permit for the restored cache
    /// bytes, so the next `commit` check holds and a tight cap cannot deny a restore that fits in
    /// net. The snapshot keeps its own permit, so the shared arrays count twice. A snapshot of
    /// another layer layout, shapes, or cache length returns `BadRequest`. A denied permit returns
    /// `MemoryPlan`. On error the state is unchanged: its permits, layers, position, and poison
    /// flag are as they were.
    pub(super) fn restore(
        &mut self,
        snapshot: &ResidentSnapshot,
        weights: &ResidentWeights,
        budget: &mut SessionBudget,
    ) -> Result<(), EngError> {
        let same_kinds = weights
            .layers
            .iter()
            .zip(&snapshot.layers)
            .all(|(layer, saved)| {
                matches!(
                    (&layer.mixer, saved),
                    (RMixer::Delta(_), LayerState::Delta { .. })
                        | (RMixer::Attn(_), LayerState::Attn(_))
                )
            });
        if weights.layers.len() != snapshot.layers.len() || !same_kinds {
            return Err(EngError::BadRequest(
                "snapshot layers do not match the model weights".into(),
            ));
        }
        for (layer, saved) in weights.layers.iter().zip(&snapshot.layers) {
            check_saved_layer(weights, &layer.mixer, saved, snapshot.pos)?;
        }
        // The permits must hold exactly the restored cache bytes for the next `commit` check.
        // Release the target's old permits first and then acquire the restored bytes: a tight cap
        // must not deny a restore that fits in net. A denied acquire re-acquires the released
        // bytes first, so the state keeps its old charge and returns `MemoryPlan` unchanged.
        let held = self.charged_bytes()?;
        drop(std::mem::take(&mut self.permits));
        let permit =
            match budget.acquire_growth(kv_bytes(&snapshot.layers, self.kv_growth_per_layer)) {
                Ok(permit) => permit,
                Err(denied) => {
                    self.permits = budget
                        .acquire_growth(held)
                        .map(|permit| vec![permit])
                        .unwrap_or_default();
                    return Err(denied);
                }
            };
        self.permits = vec![permit];
        self.layers = snapshot.layers.iter().map(LayerState::share).collect();
        self.pos = snapshot.pos;
        self.poisoned = false; // a snapshot holds committed state
        Ok(())
    }

    /// Replaces this state with a fresh zero state. On error the state is unchanged. The state
    /// keeps its growth quantum: a state made with `zero_with_step` resets to the same step, not
    /// to `KV_STEP` (fix1 item 3.5).
    pub(super) fn reset(
        &mut self,
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
    ) -> Result<(), EngError> {
        let kv_step = self.kv_step;
        *self = Self::zero_with(kernels, weights, kv_step)?;
        Ok(())
    }

    /// Resident bytes of the layer states.
    pub(super) fn byte_len(&self) -> Result<u64, EngError> {
        let mut total = Bytes::default();
        for layer in &self.layers {
            match layer {
                LayerState::Delta { rec, ring } => {
                    total.tensor(rec)?;
                    total.tensor(ring)?;
                }
                LayerState::Attn(cache) => total.add(cache.byte_len())?,
            }
        }
        Ok(total.total())
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::super::decoder::{ResidentDecoder, StepMode};
    use super::super::ops::values;
    use super::{KV_STEP, LayerState, ResidentState, kv_storage_bytes};
    use crate::family::project::Projector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::{
        EngError, kernels::MlxKernels, sched::SessionBudget, tensor::ExecutionTensor,
        weights::StreamedCheckpoint,
    };

    /// Bytes that one step adds: two attention layers, 2 key-value heads of 32, keys and values, F32.
    const KV_PER_TOKEN: u64 = 2 * 2 * 2 * 32 * 4;

    /// Bytes of the tiny state that do not grow: per DeltaNet layer the recurrent state `[2, 32, 32]`
    /// is 8,192 bytes and the convolution ring `[128, 3]` is 1,536; two layers hold 19,456.
    const FIXED_BYTES: u64 = 2 * (2 * 32 * 32 * 4 + 128 * 3 * 4);

    /// The tiny GQA checkpoint and its host weights.
    fn load() -> (StreamedCheckpoint, HostWeights) {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let host = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
        (ckpt, host)
    }

    /// The tiny KV8 checkpoint and its host weights (packet KV8-INTEGRATE): head dimension 64.
    fn load_kv8() -> (StreamedCheckpoint, HostWeights) {
        let ckpt = tiny::checkpoint(&tiny::TINY_KV8);
        let host =
            HostWeights::load(&ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap();
        (ckpt, host)
    }

    /// A decoder over the KV8 fixture with an explicit cache form and activation, whose session
    /// budget holds `cap` bytes.
    fn decoder_kv8<'c>(
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        cap: u64,
        kv: crate::family::qwen36::settings::KvDtype,
        activation: crate::family::qwen36::settings::Activation,
    ) -> ResidentDecoder<'c> {
        let mut decoder = ResidentDecoder::new_with_kv(
            MlxKernels::try_new().unwrap(),
            ckpt,
            host,
            32,
            activation,
            kv,
        )
        .unwrap();
        decoder.set_session_budget(SessionBudget::new(cap));
        decoder
    }

    /// The F32 control and Q8 decoders of the KV8 logits tests: two decoders over one host load
    /// each, differing only by the cache form.
    fn control_and_q8(
        ckpt: &StreamedCheckpoint,
        activation: crate::family::qwen36::settings::Activation,
    ) -> (ResidentDecoder<'_>, ResidentDecoder<'_>) {
        let (control, host) = (
            HostWeights::load(ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap(),
            HostWeights::load(ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap(),
        );
        (
            decoder_kv8(
                ckpt,
                control,
                1_000_000,
                crate::family::qwen36::settings::KvDtype::F32,
                activation,
            ),
            decoder_kv8(
                ckpt,
                host,
                1_000_000,
                crate::family::qwen36::settings::KvDtype::Q8,
                activation,
            ),
        )
    }

    /// The largest row sum of |W| of a host projection, computed exactly from the basis rows
    /// (the host linear of the unit vector e_i is column i of the weight): the elementwise
    /// amplification of the projection.
    fn row_amp(
        cpu: &mut crate::family::qwen36::host_cpu::CpuProjector<'_>,
        weight: &crate::family::flash_next::stored::StoredWeight,
        cols: usize,
    ) -> f32 {
        let mut sums = vec![0f32; 1 << 6];
        for i in 0..cols {
            let basis = (0..cols)
                .map(|j| if j == i { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>();
            let column = cpu.linear(weight, &basis).unwrap();
            for (j, &value) in column.iter().enumerate() {
                sums[j] += value.abs();
            }
        }
        sums.iter().copied().fold(0f32, f32::max)
    }

    /// The per-element 8-bit affine error bound and the largest magnitude of the valid F32 cache
    /// rows of one state (packet KV8-INTEGRATE): per group of 64 lanes the error is about
    /// `(max - min) / 510`, measured over the valid positions of both key-value heads of both
    /// attention caches. A buffer is `[1, KVH, cap, D]`, so element (g, t, d) sits at
    /// `(g * cap + t) * D + d`, and the slack above `len` does not count.
    fn q8_error_of(
        state: &ResidentState,
        decoder: &mut ResidentDecoder<'_>,
        width: usize,
    ) -> (f32, f32) {
        let hkv = decoder.weights.shape.attn.heads_kv;
        let groups = width / crate::family::qwen36::settings::Q8_GROUP as usize;
        let mut range = 0f32;
        let mut top = 0f32;
        for layer in state.layers() {
            let LayerState::Attn(cache) = layer else {
                continue;
            };
            for tensor in cache.tensors() {
                if tensor.dtype() != crate::tensor::Dtype::F32
                    || tensor.shape()[3] as usize != width
                {
                    continue;
                }
                let cap = tensor.shape()[2] as usize;
                let rows = values(&mut decoder.kernels, tensor).unwrap();
                for head in 0..hkv {
                    for position in 0..cache.len {
                        let lanes = &rows[(head * cap + position) * width..][..width];
                        for lane in lanes {
                            top = top.max(lane.abs());
                        }
                        for group in 0..groups {
                            let g = &lanes[group * 64..(group + 1) * 64];
                            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                            for &lane in g {
                                lo = lo.min(lane);
                                hi = hi.max(lane);
                            }
                            range = range.max(hi - lo);
                        }
                    }
                }
            }
        }
        assert!(
            range > 0.0 && top > 0.0,
            "the control's cache rows are all zero"
        );
        (range / 510.0, top)
    }

    /// A decoder on `ckpt` whose session budget holds `cap` bytes.
    fn decoder_with_cap<'c>(
        ckpt: &'c StreamedCheckpoint,
        host: HostWeights,
        cap: u64,
    ) -> ResidentDecoder<'c> {
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), ckpt, host, 32).unwrap();
        decoder.set_session_budget(SessionBudget::new(cap));
        decoder
    }

    /// Runs `tokens` as one-token `Prompt` steps: the token-by-token revision that reuses bitwise.
    fn prefill(decoder: &mut ResidentDecoder<'_>, state: &mut ResidentState, tokens: &[u32]) {
        for &token in tokens {
            decoder.step(state, token, StepMode::Prompt).unwrap();
        }
    }

    /// Every array of `layers` in model order: the DeltaNet state and ring, then the cache
    /// buffers in `KvCache::tensors` order.
    fn arrays(layers: &[LayerState]) -> Vec<&ExecutionTensor> {
        let mut out = Vec::new();
        for layer in layers {
            match layer {
                LayerState::Delta { rec, ring } => out.extend([rec, ring]),
                LayerState::Attn(cache) => out.extend(cache.tensors()),
            }
        }
        out
    }

    /// The `len` of every attention cache in `layers`, in model order.
    fn kv_lens(layers: &[LayerState]) -> Vec<usize> {
        layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Attn(cache) => Some(cache.len),
                LayerState::Delta { .. } => None,
            })
            .collect()
    }

    /// The capacity of every attention cache in `layers`, in model order.
    fn kv_caps(layers: &[LayerState]) -> Vec<usize> {
        layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Attn(cache) => Some(cache.cap),
                LayerState::Delta { .. } => None,
            })
            .collect()
    }

    /// Reads back the state that `layers` holds, in the order of `arrays`. An attention cache
    /// reads only its first `len` positions: the rows beyond `len` are capacity slack with no
    /// value (packet Q36-KV-INTEGRATE).
    fn read_state(decoder: &mut ResidentDecoder<'_>, layers: &[LayerState]) -> Vec<Vec<f32>> {
        let shape = &decoder.weights.shape;
        let hkv = shape.attn.heads_kv as u32;
        let mut out = Vec::new();
        for layer in layers {
            match layer {
                LayerState::Delta { rec, ring } => {
                    out.push(values(&mut decoder.kernels, rec).unwrap());
                    out.push(values(&mut decoder.kernels, ring).unwrap());
                }
                LayerState::Attn(cache) => {
                    // The F32 buffers read their prefix directly; the U32 codes of the Q8 form
                    // convert first and BF16 scales cast to F32 (the values of a quantized cache
                    // never come back by hand).
                    for tensor in cache.tensors() {
                        let last = tensor.shape()[3];
                        let prefix = decoder
                            .kernels
                            .slice(tensor, &[0, 0, 0, 0], &[1, hkv, cache.len as u32, last])
                            .unwrap();
                        let prefix = if tensor.dtype() == crate::tensor::Dtype::U32 {
                            decoder.kernels.u32_to_f32(&prefix).unwrap()
                        } else if tensor.dtype() != crate::tensor::Dtype::F32 {
                            decoder
                                .kernels
                                .cast(&prefix, crate::tensor::Dtype::F32)
                                .unwrap()
                        } else {
                            prefix
                        };
                        out.push(values(&mut decoder.kernels, &prefix).unwrap());
                    }
                }
            }
        }
        out
    }

    #[test]
    fn a_snapshot_charges_its_bytes_in_one_permit_and_releases_them_on_drop() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&state).unwrap();
        // The snapshot charges the caches 3 * 1,024 = 3,072 and the fixed 19,456 in one permit.
        assert_eq!(
            snapshot.charged_bytes(),
            FIXED_BYTES + 3 * KV_PER_TOKEN,
            "snapshot charge"
        );
        assert_eq!(snapshot.pos(), 3, "snapshot position");
        // The state keeps its 3,072; the snapshot adds 19,456 + 3,072 = 22,528.
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 6 * KV_PER_TOKEN,
            "state plus snapshot"
        );
        drop(snapshot);
        assert_eq!(decoder.session_used(), 3 * KV_PER_TOKEN, "after the drop");
    }

    #[test]
    fn a_snapshot_denied_by_the_budget_returns_memory_plan_and_charges_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        // Cap 10,000: after three steps 3,072 are used and 6,928 are free. The snapshot needs 22,528.
        let mut decoder = decoder_with_cap(&ckpt, host, 10_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let denied = decoder.snapshot(&state);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m.contains("22528") && m.contains("6928")),
            "{:?}",
            denied.err()
        );
        assert_eq!(
            decoder.session_used(),
            3 * KV_PER_TOKEN,
            "a denial charges nothing"
        );
        assert_eq!(state.pos(), 3, "the state is unchanged");
        assert!(!state.is_poisoned(), "a denial does not poison");
    }

    #[test]
    fn a_poisoned_state_cannot_be_saved() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1]);
        // A poisoned state without a failed step: a real native failure marks the process-wide
        // native owner terminal (decision 0032), so the test sets the poison directly.
        state.poison_for_test();
        assert!(state.is_poisoned(), "the state is poisoned for the test");
        let saved = decoder.snapshot(&state);
        assert!(
            matches!(&saved, Err(EngError::BadRequest(_))),
            "{:?}",
            saved.err()
        );
        assert_eq!(
            decoder.session_used(),
            KV_PER_TOKEN,
            "only the first step is charged"
        );
    }

    #[test]
    fn a_restore_shares_the_arrays_and_gives_position_cache_lengths_and_a_passing_check() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut restored = decoder.zero_state().unwrap();
        decoder.restore(&mut restored, &snapshot).unwrap();
        assert_eq!(restored.pos(), 3, "position");
        assert_eq!(
            kv_lens(restored.layers()),
            vec![3, 3],
            "restored cache lengths"
        );
        assert!(!restored.is_poisoned(), "poison");
        assert_eq!(
            restored.byte_len().unwrap(),
            live.byte_len().unwrap(),
            "state bytes"
        );
        // Every restored array is a handle of the snapshot array, not a copy.
        let restored_arrays = arrays(restored.layers());
        let saved_arrays = arrays(&snapshot.layers);
        assert_eq!(restored_arrays.len(), 8, "array count");
        for (i, (restored_array, saved)) in restored_arrays.iter().zip(&saved_arrays).enumerate() {
            assert!(
                restored_array.shares_storage_with(saved),
                "array {i} is shared"
            );
        }
        // Every attention cache holds `pos` positions, so the next `check_step` passes.
        assert!(
            restored.check_step(&decoder.weights, 32).is_ok(),
            "check_step after a restore"
        );
        // Live 3,072 + snapshot 22,528 + the restored state's own cache permit 3,072.
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 9 * KV_PER_TOKEN,
            "charges while all three live"
        );
        drop(snapshot);
        assert_eq!(
            decoder.session_used(),
            6 * KV_PER_TOKEN,
            "after the snapshot drops"
        );
        decoder.reset(&mut restored).unwrap();
        assert_eq!(decoder.session_used(), 3 * KV_PER_TOKEN, "after the reset");
    }

    #[test]
    fn a_restored_state_and_a_full_recompute_agree_bitwise_on_eight_suffix_tokens() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let prefix = [1u32, 2, 3];
        let suffix = [4u32, 5, 6, 7, 1, 2, 3, 4];
        // `live` continues after the snapshot. `restored` starts from the snapshot.
        // `recomputed` prefills the prefix again, as a request without reuse does.
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &prefix);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut restored = decoder.zero_state().unwrap();
        decoder.restore(&mut restored, &snapshot).unwrap();
        let mut recomputed = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut recomputed, &prefix);
        for (t, &token) in suffix.iter().enumerate() {
            let from_live = decoder.step(&mut live, token, StepMode::Record).unwrap();
            let from_restore = decoder
                .step(&mut restored, token, StepMode::Record)
                .unwrap();
            let from_recompute = decoder
                .step(&mut recomputed, token, StepMode::Record)
                .unwrap();
            assert_eq!(
                from_live, from_recompute,
                "suffix token {t}: two runs of the same steps differ (run-to-run nondeterminism, not a restore defect)"
            );
            assert_eq!(
                from_restore, from_recompute,
                "suffix token {t}: the restored run differs from the full recompute"
            );
        }
        assert_eq!(restored.pos(), 11, "position after the suffix");
    }

    #[test]
    fn live_steps_after_a_snapshot_leave_its_arrays_unchanged() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let live_before = read_state(&mut decoder, live.layers());
        let saved_before = read_state(&mut decoder, &snapshot.layers);
        assert_eq!(
            saved_before, live_before,
            "the snapshot holds the live values"
        );
        // Layout: two DeltaNet layers of state and ring, then two caches of keys and values.
        assert_eq!(saved_before.len(), 8, "array count");
        assert_eq!(
            saved_before[2].len(),
            3 * 2 * 32,
            "key values of three positions"
        );
        assert_eq!(
            saved_before[3].len(),
            3 * 2 * 32,
            "value values of three positions"
        );
        assert!(
            saved_before[0].iter().any(|v| *v != 0.0),
            "the DeltaNet state is not all zero, so the test can see a change"
        );
        prefill(&mut decoder, &mut live, &[4, 5, 6, 7, 1]);
        let saved_after = read_state(&mut decoder, &snapshot.layers);
        assert_eq!(
            saved_after, saved_before,
            "live steps changed the snapshot arrays"
        );
        let live_after = read_state(&mut decoder, live.layers());
        assert_ne!(
            live_after[0], saved_before[0],
            "the live DeltaNet state moved on"
        );
    }

    /// Packet Q36-KV-INTEGRATE item 4: a snapshot taken before further steps keeps its own
    /// values. The caches grow in quanta of 3, so the snapshot at position 2 shares a buffer
    /// whose next append on the live state would write in place (the operation copies a shared
    /// buffer once), the fourth token grows the buffers to 6, and the seventh to 9. The rows
    /// below the snapshot's `len` stay bitwise unchanged, and restoring the snapshot and running
    /// the same suffix gives the same tokens and logits as the straight run.
    #[test]
    fn a_snapshot_keeps_its_rows_across_growth_and_restores_to_the_same_tokens() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut live =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved = read_state(&mut decoder, &snapshot.layers);
        let suffix = [4u32, 5, 6, 7, 1, 2, 3];
        let mut straight = Vec::new();
        for &token in &suffix {
            straight.push(decoder.step(&mut live, token, StepMode::Record).unwrap());
        }
        let after = read_state(&mut decoder, &snapshot.layers);
        assert_eq!(
            after, saved,
            "live steps (an in-place append on the shared buffer and two growths) changed the snapshot"
        );
        // The live state really crossed the growth boundaries: caps 3 -> 6 -> 9.
        assert_eq!(kv_caps(live.layers()), vec![9, 9], "both caches grew twice");
        // Restoring the snapshot and running the same suffix gives the straight run.
        let mut restored = decoder.zero_state().unwrap();
        decoder.restore(&mut restored, &snapshot).unwrap();
        for (index, &token) in suffix.iter().enumerate() {
            let got = decoder
                .step(&mut restored, token, StepMode::Record)
                .unwrap();
            assert_eq!(
                got.token, straight[index].token,
                "suffix token {index} differs after the restore"
            );
            assert_eq!(
                got.logits, straight[index].logits,
                "suffix token {index} logits differ after the restore"
            );
        }
        assert_eq!(restored.pos(), 9, "position after the suffix");
    }

    /// Packet Q36-KV-INTEGRATE item 6(d): after 30 steps with the default quantum (the test decoder's context limit is 32 positions) the actual
    /// storage (the sum of the tensors' `byte_len`) is exactly one quantum of capacity per
    /// attention layer, that is the logical bytes plus the unused `KV_STEP - 30` positions.
    #[test]
    fn the_cache_storage_exceeds_the_logical_bytes_by_exactly_the_unused_quantum() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut state = decoder.zero_state().unwrap();
        let tokens: Vec<u32> = (0..30).map(|t| (t % 7 + 1) as u32).collect();
        prefill(&mut decoder, &mut state, &tokens);
        let logical = u64::from(state.pos()) * KV_PER_TOKEN;
        let stored = kv_storage_bytes(state.layers());
        assert_eq!(
            kv_lens(state.layers()),
            vec![30, 30],
            "both caches hold the run"
        );
        // The exact hand-derived storage (review B finding 4): 30 positions of the default
        // quantum 256 round up to one step of capacity per layer, so the buffers hold exactly
        // `KV_STEP * KV_PER_TOKEN` bytes, not any capacity in [30, 30 + 256].
        assert_eq!(
            kv_caps(state.layers()),
            vec![KV_STEP, KV_STEP],
            "thirty positions of the default quantum hold exactly one step of capacity"
        );
        assert_eq!(
            state.byte_len().unwrap(),
            FIXED_BYTES + stored,
            "the state reports the stored bytes"
        );
        assert_eq!(stored, KV_STEP as u64 * KV_PER_TOKEN, "the exact storage");
        assert!(
            stored >= logical,
            "the buffers hold at least the valid rows: {stored} < {logical}"
        );
    }

    #[test]
    fn a_restore_replaces_a_poisoned_state_and_the_next_step_commits() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        // The target holds one committed step (1,024 bytes) and a poisoned failed step.
        let mut target = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut target, &[5]);
        // A poisoned target without a failed step: a real native failure marks the process-wide
        // native owner terminal (decision 0032), so the test poisons the target directly, with
        // its layer states taken out as a failed step leaves them.
        target.poison_for_test();
        assert!(target.is_poisoned(), "the target is poisoned for the test");
        assert_eq!(target.pos(), 1, "the poison does not advance the position");
        assert!(
            arrays(target.layers()).is_empty(),
            "the poison took the layer states out"
        );
        decoder.restore(&mut target, &snapshot).unwrap();
        assert!(!target.is_poisoned(), "restore clears the poison");
        assert_eq!(target.pos(), 3, "position");
        assert_eq!(kv_lens(target.layers()), vec![3, 3], "cache lengths");
        // Every restored array is a handle of the snapshot array, not of the old state.
        let restored_arrays = arrays(target.layers());
        let saved_arrays = arrays(&snapshot.layers);
        for (i, (restored_array, saved)) in restored_arrays.iter().zip(&saved_arrays).enumerate() {
            assert!(
                restored_array.shares_storage_with(saved),
                "array {i} is shared"
            );
        }
        // Live 3,072 + snapshot 22,528 + target 3,072. The old 1,024-byte permit is released.
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 9 * KV_PER_TOKEN,
            "charges after the restore"
        );
        // The next commit checks the charge against the stored cache bytes, so this step commits
        // only when the restored state holds exactly one permit of the restored cache bytes.
        decoder.step(&mut target, 4, StepMode::Prompt).unwrap();
        assert_eq!(target.pos(), 4, "position after one more step");
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 10 * KV_PER_TOKEN,
            "one more step charges 1,024"
        );
    }

    #[test]
    fn a_restore_rejects_a_snapshot_with_another_layer_layout() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        // Layer 0 is DeltaNet and layer 1 is attention in the model. Swap them in the snapshot.
        let mut swapped = decoder.snapshot(&live).unwrap();
        swapped.layers.swap(0, 1);
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &swapped);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged after a kind swap");
        // A snapshot with one layer fewer than the model is refused the same way.
        let mut short = decoder.snapshot(&live).unwrap();
        short.layers.pop();
        let rejected = decoder.restore(&mut target, &short);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged after a pop");
        // A rejection charges no permit: the live 3,072 and the two snapshots of 22,528.
        assert_eq!(
            decoder.session_used(),
            3 * KV_PER_TOKEN + 2 * (FIXED_BYTES + 3 * KV_PER_TOKEN),
            "charges while the rejected snapshots live"
        );
    }

    /// Packet KV8-INTEGRATE test (a) at the decoder level: eight decode steps of a Q8-cache
    /// engine against the F32-cache control, both in F32 activation. The tolerance derivation
    /// (stated): the 8-bit affine error of one cached element is `e = R / 510` (`R` the largest
    /// group range over the valid rows of both heads of the control's caches, measured in the
    /// test); one score moves by at most `sqrt(D) * Q_max * e` with the
    /// RMS-normed query magnitude bounded by `Q_max` 16; the softmax moves total variation by at
    /// most twice that; one attention output moves by at most `e + V_max * 2 * sqrt(D) * Q_max *
    /// e`; the o-projection amplifies elementwise by `amp` (its exact largest row sum), and the
    /// head by `head_amp` (the exact largest row sum of `lm_head`). The factor 16 stands for the
    /// compounding through the 2 attention layers and the 8 steps; it is a safety factor, not a
    /// derivation, because the RMS norms, the DeltaNet recurrence, and the discrete top-2 routing
    /// are not bounded here. The bound is therefore a worst-case guard that catches only a gross
    /// error; the corrupted-cache test of `attn.rs` supplies the sensitivity to a wrong scale or
    /// bias. Every logit is finite, and every greedy token of the run is equal (a gate on top of
    /// the bound: it assumes the tiny vocabulary keeps a top-two gap above the Q8 error, which
    /// the passing Mac run confirms for this fixture only).
    #[test]
    fn eight_q8_decode_steps_match_the_f32_cache_logits_within_the_derived_tolerance() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut cpu = crate::family::qwen36::host_cpu::CpuProjector::new(&ckpt, 1);
        let view = host.view();
        let mut amp = 0f32;
        for layer in &view.layers {
            if let crate::family::qwen36::host::MixerW::Attn(w) = &layer.mixer {
                amp = amp.max(row_amp(&mut cpu, w.o, 256));
            }
        }
        let head_amp = row_amp(&mut cpu, view.lm_head, 64);
        let (mut control, mut quantized) =
            control_and_q8(&ckpt, crate::family::qwen36::settings::Activation::F32);
        let mut control_state = control.zero_state().unwrap();
        let mut q8_state = quantized.zero_state().unwrap();
        let mut worst = 0f32;
        for step in 0..8u32 {
            let token = (step + 1) % 8;
            let from_control = control
                .step(&mut control_state, token, StepMode::Logits)
                .unwrap();
            let from_q8 = quantized
                .step(&mut q8_state, token, StepMode::Logits)
                .unwrap();
            let (control_logits, q8_logits) = (
                from_control.logits.as_ref().unwrap(),
                from_q8.logits.as_ref().unwrap(),
            );
            for value in q8_logits {
                assert!(value.is_finite(), "step {step}: a Q8 logit is not finite");
            }
            assert_eq!(
                from_control.token, from_q8.token,
                "step {step}: the greedy token differs"
            );
            let (e, v_max) = q8_error_of(&control_state, &mut control, 64);
            let tol = 2.0 * 8.0 * amp * (e + v_max * 2.0 * 8.0 * 16.0 * e) * head_amp;
            let step_worst = control_logits
                .iter()
                .zip(q8_logits)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                step_worst <= tol,
                "step {step}: the Q8 logits differ by {step_worst}, the derived tolerance is                  {tol} (e {e}, v_max {v_max}, amp {amp}, head_amp {head_amp})"
            );
            worst = worst.max(step_worst);
        }
        println!("q8 decoder logits: worst difference {worst}");
        assert_eq!(q8_state.pos(), 8, "the Q8 run holds 8 positions");
    }

    /// Packet KV8-INTEGRATE test (b): the same comparison under BF16 activation, where the Q8
    /// scales and biases are BF16. Rounding a scale `s` and a bias `b` to BF16 moves a
    /// reconstructed element `q * s + b` by at most `(255 * |s| + |b|) / 256`, which is
    /// `(R + |b|) / 256` for a group of range `R` (`255 * |s|` is about `R`) and at most
    /// `(R + V_max) / 256` because a bias is no larger than the largest magnitude; this term is
    /// added to `e`. The scores, the softmax, and the attention output of the Q8 path also run in
    /// BF16 and add a relative 2^-8 to every score; the factor 16 and the `V_max` term of the
    /// bound swallow it, so the bound stays a worst-case guard.
    #[test]
    fn eight_q8_decode_steps_match_the_f32_cache_logits_within_the_derived_tolerance_in_bf16() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut cpu = crate::family::qwen36::host_cpu::CpuProjector::new(&ckpt, 1);
        let view = host.view();
        let mut amp = 0f32;
        for layer in &view.layers {
            if let crate::family::qwen36::host::MixerW::Attn(w) = &layer.mixer {
                amp = amp.max(row_amp(&mut cpu, w.o, 256));
            }
        }
        let head_amp = row_amp(&mut cpu, view.lm_head, 64);
        let activation = crate::family::qwen36::settings::Activation::Bf16;
        let (mut control, mut quantized) = control_and_q8(&ckpt, activation);
        let mut control_state = control.zero_state().unwrap();
        let mut q8_state = quantized.zero_state().unwrap();
        for step in 0..8u32 {
            let token = (step + 1) % 8;
            let from_control = control
                .step(&mut control_state, token, StepMode::Logits)
                .unwrap();
            let from_q8 = quantized
                .step(&mut q8_state, token, StepMode::Logits)
                .unwrap();
            let (control_logits, q8_logits) = (
                from_control.logits.as_ref().unwrap(),
                from_q8.logits.as_ref().unwrap(),
            );
            for value in q8_logits {
                assert!(
                    value.is_finite(),
                    "step {step}: a BF16 Q8 logit is not finite"
                );
            }
            assert_eq!(
                from_control.token, from_q8.token,
                "step {step}: the greedy token differs under BF16"
            );
            let (e, v_max) = q8_error_of(&control_state, &mut control, 64);
            let e = e + (e * 510.0 + v_max) / 256.0;
            let tol = 2.0 * 8.0 * amp * (e + v_max * 2.0 * 8.0 * 16.0 * e) * head_amp;
            let step_worst = control_logits
                .iter()
                .zip(q8_logits)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                step_worst <= tol,
                "step {step}: the BF16 Q8 logits differ by {step_worst}, the derived tolerance                  is {tol}"
            );
        }
    }

    /// Packet KV8-INTEGRATE test (c) at the decoder level: a Q8 run that crosses two capacity
    /// growth boundaries (the step quantum 3 against the default 256) is BIT-IDENTICAL to the
    /// same run that never grows: the quantization of a row and the prefix views never depend on
    /// the capacity.
    #[test]
    fn a_q8_run_across_growth_boundaries_is_bit_identical_to_a_large_step_run() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut small = decoder_kv8(
            &ckpt,
            host,
            1_000_000,
            crate::family::qwen36::settings::KvDtype::Q8,
            crate::family::qwen36::settings::Activation::F32,
        );
        let mut small_state =
            ResidentState::zero_with_step(&mut small.kernels, &small.weights, 3).unwrap();
        let mut large = decoder_kv8(
            &ckpt,
            HostWeights::load(&ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap(),
            1_000_000,
            crate::family::qwen36::settings::KvDtype::Q8,
            crate::family::qwen36::settings::Activation::F32,
        );
        let mut large_state = large.zero_state().unwrap();
        for step in 0..8u32 {
            let token = (step + 1) % 8;
            let from_small = small
                .step(&mut small_state, token, StepMode::Logits)
                .unwrap();
            let from_large = large
                .step(&mut large_state, token, StepMode::Logits)
                .unwrap();
            assert_eq!(
                from_small.token, from_large.token,
                "step {step}: the token differs across the growth"
            );
            assert_eq!(
                from_small.logits, from_large.logits,
                "step {step}: the logits differ across the growth"
            );
        }
        assert_eq!(kv_caps(small_state.layers()), vec![9, 9], "two growths");
        assert_eq!(
            kv_caps(large_state.layers()),
            vec![KV_STEP, KV_STEP],
            "the default quantum never grows in 8 steps"
        );
    }

    /// Packet KV8-INTEGRATE test (e), first half: a snapshot taken before further Q8 steps keeps
    /// its own rows across an in-place append and two growths, and restoring it and running the
    /// same suffix is bit-identical to the first continuation.
    #[test]
    fn a_q8_snapshot_keeps_its_rows_and_restores_to_the_same_tokens() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut decoder = decoder_kv8(
            &ckpt,
            host,
            1_000_000,
            crate::family::qwen36::settings::KvDtype::Q8,
            crate::family::qwen36::settings::Activation::F32,
        );
        let mut live =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved = read_state(&mut decoder, &snapshot.layers);
        // The snapshot holds the live arrays, part by part. A snapshot that swapped two parts
        // (scales for biases) would still round-trip through a restore that swaps them back, so
        // the round trip alone cannot see it.
        assert_eq!(
            read_state(&mut decoder, live.layers()),
            saved,
            "the Q8 snapshot differs from the live state it was taken from"
        );
        let suffix = [4u32, 5, 6, 7, 1, 2, 3];
        let mut straight = Vec::new();
        for &token in &suffix {
            straight.push(decoder.step(&mut live, token, StepMode::Record).unwrap());
        }
        let after = read_state(&mut decoder, &snapshot.layers);
        assert_eq!(
            after, saved,
            "live Q8 steps (an in-place append on the shared buffers and two growths) changed the snapshot"
        );
        assert_eq!(kv_caps(live.layers()), vec![9, 9], "the buffers grew twice");
        let mut restored = decoder.zero_state().unwrap();
        decoder.restore(&mut restored, &snapshot).unwrap();
        for (index, &token) in suffix.iter().enumerate() {
            let got = decoder
                .step(&mut restored, token, StepMode::Record)
                .unwrap();
            assert_eq!(
                got.token, straight[index].token,
                "suffix token {index} differs after the Q8 restore"
            );
            assert_eq!(
                got.logits, straight[index].logits,
                "suffix token {index} logits differ after the Q8 restore"
            );
        }
        assert_eq!(restored.pos(), 9, "position after the suffix");
    }

    /// Packet KV8-INTEGRATE test (e), second half: the session A, B, A isolation of the F32 path
    /// repeated for Q8. Two states restored from ONE Q8 snapshot step DIFFERENT tokens at the
    /// SAME position; every sharer's append writes the first slack row of the shared buffers, so
    /// a missed copy makes B's token overwrite A's. Each state's logits are compared bit for bit
    /// with an independent straight run.
    #[test]
    fn two_q8_restores_that_step_different_tokens_match_independent_straight_runs() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut decoder = decoder_kv8(
            &ckpt,
            host,
            1_000_000,
            crate::family::qwen36::settings::KvDtype::Q8,
            crate::family::qwen36::settings::Activation::F32,
        );
        let mut live =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut a = decoder.zero_state().unwrap();
        decoder.restore(&mut a, &snapshot).unwrap();
        let mut b = decoder.zero_state().unwrap();
        decoder.restore(&mut b, &snapshot).unwrap();
        let a_four = decoder.step(&mut a, 4, StepMode::Record).unwrap();
        let b_five = decoder.step(&mut b, 5, StepMode::Record).unwrap();
        let a_six = decoder.step(&mut a, 6, StepMode::Record).unwrap();
        let mut straight_a =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut straight_a, &[1, 2]);
        let want_four = decoder.step(&mut straight_a, 4, StepMode::Record).unwrap();
        let want_six = decoder.step(&mut straight_a, 6, StepMode::Record).unwrap();
        let mut straight_b =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut straight_b, &[1, 2]);
        let want_five = decoder.step(&mut straight_b, 5, StepMode::Record).unwrap();
        assert_eq!(
            a_four.token, want_four.token,
            "A: the token of the first step"
        );
        assert_eq!(
            a_four.logits, want_four.logits,
            "A: the logits of the first step differ from the straight run"
        );
        assert_eq!(
            a_six.token, want_six.token,
            "A: the token after the aliased row"
        );
        assert_eq!(
            a_six.logits, want_six.logits,
            "A: the logits after stepping past the aliased row differ from the straight run"
        );
        assert_eq!(b_five.token, want_five.token, "B: the token");
        assert_eq!(
            b_five.logits, want_five.logits,
            "B: the logits differ from the straight run"
        );
        assert_eq!((a.pos(), b.pos()), (4, 3), "A moved on and B stayed");
    }

    /// Packet KV8-INTEGRATE test (f): the stored bytes of a whole Q8 state equal the logical
    /// charge of its positions plus at most one `KV_STEP` of slack per attention layer. `per_layer`
    /// is the hand-derived charge of one layer and position on the fixture (two key-value heads of
    /// width 64, one group per head): per side the codes take 2 * 64 bytes and the scales and
    /// biases take 2 arrays * 2 heads * 1 group * the scale width, so F32 scales give
    /// 2 * (128 + 16) = 288 and BF16 scales give 2 * (128 + 8) = 272. The fixture has two
    /// attention layers.
    fn q8_state_stores_the_logical_charge(
        activation: crate::family::qwen36::settings::Activation,
        per_layer: u64,
    ) {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut decoder = decoder_kv8(
            &ckpt,
            host,
            1_000_000,
            crate::family::qwen36::settings::KvDtype::Q8,
            activation,
        );
        let mut state = decoder.zero_state().unwrap();
        let tokens: Vec<u32> = (0..30).map(|t| (t % 7 + 1) as u32).collect();
        prefill(&mut decoder, &mut state, &tokens);
        let growth = 2 * per_layer;
        let logical = u64::from(state.pos()) * growth;
        // The charge the code holds against the session budget is the hand-derived figure, so a
        // wrong scale width in the charge (F32 bytes for BF16 scales) cannot hide behind a
        // charge and a permit that are both wrong the same way.
        assert_eq!(
            decoder.session_used(),
            logical,
            "the logical charge of the committed positions"
        );
        let stored = kv_storage_bytes(state.layers());
        assert_eq!(
            kv_lens(state.layers()),
            vec![30, 30],
            "both Q8 caches hold the run"
        );
        assert_eq!(
            kv_caps(state.layers()),
            vec![KV_STEP, KV_STEP],
            "thirty positions of the default quantum hold one step of capacity"
        );
        assert_eq!(stored, KV_STEP as u64 * growth, "the exact Q8 storage");
        assert!(
            stored <= logical + KV_STEP as u64 * growth,
            "the storage {stored} exceeds the logical {logical} plus one step of slack per layer"
        );
        assert!(
            stored >= logical,
            "the buffers hold at least the valid rows: {stored} < {logical}"
        );
    }

    #[test]
    fn the_q8_state_stores_the_logical_charge_plus_the_slack_bound() {
        q8_state_stores_the_logical_charge(crate::family::qwen36::settings::Activation::F32, 288);
    }

    #[test]
    fn the_q8_state_with_bf16_scales_stores_the_logical_charge_plus_the_slack_bound() {
        q8_state_stores_the_logical_charge(crate::family::qwen36::settings::Activation::Bf16, 272);
    }

    /// Packet KV8-INTEGRATE test (g): a Q8 decoder refuses a state whose attention cache holds
    /// the F32 form at `check_step`, with no native call (the host validation of the step, which
    /// runs before the first native call and before any buffer is consumed).
    #[test]
    fn a_q8_decoder_refuses_a_state_of_the_f32_form_before_any_native_call() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load_kv8();
        let mut decoder = decoder_kv8(
            &ckpt,
            host,
            1_000_000,
            crate::family::qwen36::settings::KvDtype::Q8,
            crate::family::qwen36::settings::Activation::F32,
        );
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1]);
        if let LayerState::Attn(cache) = &mut state.layers[1] {
            cache.kv = crate::family::qwen36::settings::KvDtype::F32;
        }
        let refused = state.check_step(&decoder.weights, 32);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("form")),
            "an F32-form cache under a Q8 engine: {:?}",
            refused.err()
        );
    }

    /// A saved layer whose recurrent state shape differs from the model layout is refused with
    /// `BadRequest` before the restore shares anything.
    #[test]
    fn a_restore_rejects_a_saved_layer_whose_recurrent_state_shape_differs() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut wrong = decoder.snapshot(&live).unwrap();
        // Layer 0 is DeltaNet: give its recurrent state the shape of its convolution ring.
        if let LayerState::Delta { rec, ring } = &mut wrong.layers[0] {
            *rec = ring.share();
        }
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("state shape")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    /// A saved layer whose convolution ring shape differs from the model layout is refused with
    /// `BadRequest` before the restore shares anything.
    #[test]
    fn a_restore_rejects_a_saved_layer_whose_convolution_ring_shape_differs() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut wrong = decoder.snapshot(&live).unwrap();
        // Layer 0 is DeltaNet: give its convolution ring the shape of its recurrent state.
        if let LayerState::Delta { rec, ring } = &mut wrong.layers[0] {
            *ring = rec.share();
        }
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("state shape")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    /// A saved attention cache whose length differs from the snapshot position is refused with
    /// `BadRequest`: the restored state would fail the next `check_step`.
    #[test]
    fn a_restore_rejects_a_saved_cache_whose_length_differs_from_the_snapshot_position() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut wrong = decoder.snapshot(&live).unwrap();
        // The caches hold three positions and their shapes say so; the snapshot claims two.
        wrong.pos = 2;
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("snapshot position")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    /// A saved attention cache whose declared form does not match its buffers is refused with
    /// `BadRequest` (packet KV8-INTEGRATE item 7g). The half-filled pair of the old
    /// `Option`-per-side layout is now a TYPE error - the `KvBuffers` enum cannot express it -
    /// so the runtime refusal that stays is the form mismatch: a cache that claims the Q8 form
    /// while holding the F32 buffers.
    #[test]
    fn a_restore_rejects_a_saved_cache_whose_form_does_not_match_its_buffers() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut target = decoder.zero_state().unwrap();
        let mut wrong_form = decoder.snapshot(&live).unwrap();
        if let LayerState::Attn(cache) = &mut wrong_form.layers[1] {
            cache.kv = crate::family::qwen36::settings::KvDtype::Q8;
        }
        let rejected = decoder.restore(&mut target, &wrong_form);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "a cache whose form does not match its buffers: {:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    /// A saved attention cache that holds no arrays but claims a position is refused with
    /// `BadRequest`.
    #[test]
    fn a_restore_rejects_a_saved_cache_that_holds_no_pair_at_a_position() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut wrong = decoder.snapshot(&live).unwrap();
        // Layer 1 is attention: its cache claims three positions and holds no arrays.
        if let LayerState::Attn(cache) = &mut wrong.layers[1] {
            cache.buffers = None;
        }
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("snapshot position")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    #[test]
    fn a_denied_restore_returns_memory_plan_and_leaves_the_target_untouched() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        // Cap 27,000: live 3,072 + snapshot 22,528 + target 1,024 = 26,624 are used and 376 are
        // free. The restore needs 3,072 restored cache bytes and the net of them over the target's
        // 1,024 is 2,048, which does not fit.
        let mut decoder = decoder_with_cap(&ckpt, host, 27_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut target = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut target, &[5]);
        // A poisoned target without a failed step: the tight cap below denies a step's growth
        // permit before it poisons anything, and a real native failure marks the process-wide
        // native owner terminal (decision 0032), so the test poisons the target directly.
        target.poison_for_test();
        assert!(target.is_poisoned(), "the target is poisoned for the test");
        let denied = decoder.restore(&mut target, &snapshot);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(_))),
            "{:?}",
            denied.err()
        );
        assert_eq!(target.pos(), 1, "a denied restore keeps the position");
        assert!(target.is_poisoned(), "a denied restore keeps the poison");
        assert!(
            arrays(target.layers()).is_empty(),
            "a denied restore keeps the taken-out layer states"
        );
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 7 * KV_PER_TOKEN,
            "a denied restore keeps the target's 1,024-byte charge"
        );
    }

    #[test]
    fn a_restore_that_fits_in_net_with_a_full_budget_succeeds() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        // Cap = the snapshot charge (fixed plus one position) plus the two positions that the
        // live and the target states hold: the budget is exactly full when the restore runs.
        let mut decoder = decoder_with_cap(&ckpt, host, FIXED_BYTES + 3 * KV_PER_TOKEN);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut target = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut target, &[5]);
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 3 * KV_PER_TOKEN,
            "live plus snapshot plus target fill the budget"
        );
        // The target holds 1,024 and the restore takes 1,024: it fits in net (a release and an
        // acquire of the same bytes), although not one byte is free.
        decoder.restore(&mut target, &snapshot).unwrap();
        assert_eq!(target.pos(), 1, "position");
        assert_eq!(kv_lens(target.layers()), vec![1, 1], "cache lengths");
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 3 * KV_PER_TOKEN,
            "the restore fits in net and charges nothing more"
        );
        let target_arrays = arrays(target.layers());
        let saved_arrays = arrays(&snapshot.layers);
        assert_eq!(target_arrays.len(), 8, "array count");
        for (i, (target_array, saved)) in target_arrays.iter().zip(&saved_arrays).enumerate() {
            assert!(
                target_array.shares_storage_with(saved),
                "array {i} is shared"
            );
        }
        assert!(
            target.check_step(&decoder.weights, 32).is_ok(),
            "check_step after the restore"
        );
    }

    #[test]
    fn a_restore_at_position_zero_gives_an_empty_state_whose_next_step_commits() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        // A snapshot of a zero state: position 0, empty caches, and only the fixed bytes charged.
        let live = decoder.zero_state().unwrap();
        let snapshot = decoder.snapshot(&live).unwrap();
        assert_eq!(snapshot.pos(), 0, "snapshot position");
        assert_eq!(snapshot.charged_bytes(), FIXED_BYTES, "snapshot charge");
        let mut target = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut target, &[5, 6]);
        decoder.restore(&mut target, &snapshot).unwrap();
        assert_eq!(target.pos(), 0, "restored position");
        assert_eq!(kv_lens(target.layers()), vec![0, 0], "cache lengths");
        assert!(!target.is_poisoned(), "poison");
        // The restore releases the target's two positions and keeps no cache permit.
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES,
            "charges after the restore"
        );
        assert!(
            target.check_step(&decoder.weights, 32).is_ok(),
            "check_step at position 0"
        );
        // One step from the restored position 0 commits and equals the same step from a zero state.
        let from_restore = decoder.step(&mut target, 3, StepMode::Record).unwrap();
        let mut fresh = decoder.zero_state().unwrap();
        let from_zero = decoder.step(&mut fresh, 3, StepMode::Record).unwrap();
        assert_eq!(
            from_restore, from_zero,
            "a step from the restored position 0 differs from a fresh zero state"
        );
        assert_eq!(target.pos(), 1, "position after one step");
        // One step charges one position of KV growth. Two states stepped here (the restored
        // target and the fresh comparison state), so the session holds the snapshot's fixed bytes
        // plus one position for each of them.
        assert_eq!(
            decoder.session_used(),
            FIXED_BYTES + 2 * KV_PER_TOKEN,
            "the snapshot's fixed bytes plus one position per stepped state"
        );
    }

    #[test]
    fn two_restores_of_one_snapshot_stay_independent_when_one_state_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut first = decoder.zero_state().unwrap();
        decoder.restore(&mut first, &snapshot).unwrap();
        let mut second = decoder.zero_state().unwrap();
        decoder.restore(&mut second, &snapshot).unwrap();
        let saved = read_state(&mut decoder, &snapshot.layers);
        let first_before = read_state(&mut decoder, first.layers());
        let second_before = read_state(&mut decoder, second.layers());
        assert_eq!(
            first_before, saved,
            "the first restore shares the saved arrays"
        );
        assert_eq!(
            second_before, saved,
            "the second restore shares the saved arrays"
        );
        decoder.step(&mut first, 4, StepMode::Prompt).unwrap();
        let first_after = read_state(&mut decoder, first.layers());
        let second_after = read_state(&mut decoder, second.layers());
        assert_eq!(
            second_after, second_before,
            "a step on one state changed the other bitwise"
        );
        assert_ne!(
            first_after[0], first_before[0],
            "the step moved the first state on"
        );
        assert_eq!(first.pos(), 4, "the first state advanced");
        assert_eq!(second.pos(), 3, "the second state stayed");
    }

    /// Fix1 item 2 (review A finding 2, review B finding 1): two states restored from ONE
    /// snapshot step DIFFERENT tokens at the SAME position. Every sharer's append writes
    /// position 2, the first slack row of the shared buffer, so a missed copy — `kv_append`
    /// writing in place through the snapshot's handle — makes B's token 5 overwrite A's token 4,
    /// and A's next step, whose growth copies the valid prefix `[0, 3)`, reads B's row. Each
    /// state's logits are compared bit for bit with an independent straight run of the same
    /// token sequence on a fresh state.
    #[test]
    fn two_restores_that_step_different_tokens_match_independent_straight_runs() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut live =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        // The snapshot at position 2 of capacity 3 shares its buffers with every restore.
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut a = decoder.zero_state().unwrap();
        decoder.restore(&mut a, &snapshot).unwrap();
        let mut b = decoder.zero_state().unwrap();
        decoder.restore(&mut b, &snapshot).unwrap();
        // Both states write position 2, A with token 4 and B with token 5.
        let a_four = decoder.step(&mut a, 4, StepMode::Record).unwrap();
        let b_five = decoder.step(&mut b, 5, StepMode::Record).unwrap();
        // A steps again: the growth to 6 copies the prefix `[0, 3)`, which under the missed-copy
        // bug holds B's row 2 instead of A's.
        let a_six = decoder.step(&mut a, 6, StepMode::Record).unwrap();
        // The independent straight runs of the same token sequences, on fresh states.
        let mut straight_a =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut straight_a, &[1, 2]);
        let want_four = decoder.step(&mut straight_a, 4, StepMode::Record).unwrap();
        let want_six = decoder.step(&mut straight_a, 6, StepMode::Record).unwrap();
        let mut straight_b =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut straight_b, &[1, 2]);
        let want_five = decoder.step(&mut straight_b, 5, StepMode::Record).unwrap();
        assert_eq!(
            a_four.token, want_four.token,
            "A: the token of the first step"
        );
        assert_eq!(
            a_four.logits, want_four.logits,
            "A: the logits of the first step differ from the straight run [1, 2, 4]"
        );
        assert_eq!(
            a_six.token, want_six.token,
            "A: the token after the aliased row"
        );
        assert_eq!(
            a_six.logits, want_six.logits,
            "A: the logits after stepping past the aliased row differ from the straight run [1, 2, 4, 6]"
        );
        assert_eq!(b_five.token, want_five.token, "B: the token");
        assert_eq!(
            b_five.logits, want_five.logits,
            "B: the logits differ from the straight run [1, 2, 5]"
        );
        assert_eq!(a.pos(), 4, "A holds four positions");
        assert_eq!(b.pos(), 3, "B holds three positions");
    }

    /// Review B finding 5: the `cap >= len` clause of `check_saved_layer` is reached by a cache
    /// whose buffer shapes MATCH the claimed capacity. The snapshot claims 300 positions while
    /// its buffers hold the 256 of one default quantum, and the restore refuses it.
    #[test]
    fn a_restore_rejects_a_saved_cache_whose_capacity_is_below_its_length() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut wrong = decoder.snapshot(&live).unwrap();
        // The attention layers (1 and 3 of TINY_GQA) have buffers `[1, 2, 256, 32]` (one
        // quantum) and now ALL claim 300 valid positions at the same capacity, so the layer
        // check reaches its `cap >= len` clause (one layer left at 3 positions would reject on
        // the position before the capacity is looked at).
        for layer in &mut wrong.layers {
            if let LayerState::Attn(cache) = layer {
                cache.len = 300;
            }
        }
        wrong.pos = 300;
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("snapshot position")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    #[test]
    fn a_restore_followed_by_a_chunked_prefill_agrees_bitwise_with_a_full_recompute() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        decoder.set_chunk_rows(2).unwrap();
        let prefix = [1u32, 2, 3];
        let suffix = [4u32, 5, 6, 7, 1];
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &prefix);
        let snapshot = decoder.snapshot(&live).unwrap();
        // The restored run continues from the snapshot with the chunked path (five rows in
        // chunks of two). The recompute run runs every token again from a zero state.
        let mut restored = decoder.zero_state().unwrap();
        decoder.restore(&mut restored, &snapshot).unwrap();
        let from_restore = decoder
            .prefill(&mut restored, &suffix, StepMode::Logits)
            .unwrap();
        let mut recomputed = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut recomputed, &prefix);
        let from_recompute = decoder
            .prefill(&mut recomputed, &suffix, StepMode::Logits)
            .unwrap();
        assert_eq!(
            from_restore, from_recompute,
            "the restored run differs from the full recompute"
        );
        assert_eq!(
            read_state(&mut decoder, restored.layers()),
            read_state(&mut decoder, recomputed.layers()),
            "the restored state differs from the recomputed state bitwise"
        );
        assert_eq!(restored.pos(), 8, "position after prefix and suffix");
    }
}
