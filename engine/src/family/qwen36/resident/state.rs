//! Resident Q36 session state, with the step's check, poison, and commit.
//!
//! DeltaNet layers keep a recurrent state `[VH, VD, KD]` and a convolution ring `[CD, CK - 1]`.
//! Attention layers keep a key-value cache in preallocated capacity buffers written in place: in
//! the F32 control `[1, KVH, cap, D]` per key and value tensor, in the Q8 form the codes
//! `[1, KVH, cap, D/4]` beside scales and biases `[1, KVH, cap, D/64]` of the activation dtype,
//! `len` valid positions, growth in `KV_STEP` quanta. The growth charge covers the cache only, as
//! in the Flash-Next decoder, and stays LOGICAL — `len` positions of growth bytes per layer, from
//! the cache form and the activation dtype — so the capacity slack, bounded by one `KV_STEP` of
//! growth bytes per attention layer (256 * 4,096 * 10 = 10,485,760 bytes on the real model in the
//! F32 form; 256 * 1,088 * 10 = 2,785,280 at Q8 with BF16 scales), never enters a permit;
//! `kv_storage_bytes` reports the stored bytes. The slack bound excludes the growth TRANSIENT — the
//! old buffer, its slice copy, and the new buffer alive together at a growth, about 100 MB per
//! layer at depth 8,192 in the F32 form — which the session reserve margin and the retire cap
//! cover. The fixed DeltaNet state of 65,863,680 bytes is not charged by the steps; a snapshot
//! charges it once with its entry.

use super::attn::{KvCache, growth_bytes};
use super::kv_plan::{KV_STEP, session_bytes};
use super::ops::{Bytes, dim, short_import, zeros};
use super::prefix_pages::{Page, allowance_pages};
use super::weights::{RMixer, ResidentWeights};
use crate::{
    EngError,
    kernels::MlxKernels,
    sched::{GrowthPermit, SessionBudget},
    tensor::{ExecutionTensor, Tensor},
};
use std::sync::atomic::{AtomicU64, Ordering};

/// The global lineage counter of attention-buffer sets: every `zero_with` draws a
/// fresh unique generation, so a snapshot restores only into the state whose buffers it
/// aliased - a separately-zeroed state never matches, and a reset draws a new generation for
/// its fresh buffers.
static BUFFER_GENERATION: AtomicU64 = AtomicU64::new(0);

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
/// form and the activation dtype: the F32 control charges
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

/// The LOGICAL bytes of the attention caches in `layers`: one
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
    kv_bytes_at_len(layers, None, per_layer)
}

/// The admission charge with an optional valid-length override: the rewind path acquires
/// for the REWOUND extent before any mutation, so it passes the snapshot position while the
/// caches still hold the longer valid length.
fn kv_bytes_at_len(layers: &[LayerState], len_override: Option<usize>, per_layer: u64) -> u64 {
    let mut attn_layers = 0;
    let mut len: Option<usize> = None;
    for layer in layers {
        if let LayerState::Attn(cache) = layer {
            attn_layers += 1;
            let valid = len_override.unwrap_or(cache.len);
            match len {
                None => len = Some(valid),
                // Attention layers of unequal valid length cannot be a committed state: the
                // charge saturates, so the permit equality check of `commit_rows` fails.
                Some(first) if first != valid => return u64::MAX,
                Some(_) => {}
            }
        }
    }
    session_bytes(attn_layers, len.unwrap_or(0), per_layer).unwrap_or(u64::MAX)
}

/// Bytes that the attention cache tensors of `layers` store: the capacity buffers summed from
/// the tensors' own sizes, slack included. Reports and tests read this;
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
            // The form and the shapes of every buffer, with no native call: a saved cache of
            // another form or shape is refused.
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

/// A committed `ResidentState` shared by a prefix entry without copying KV payloads.
/// The snapshot shares layer buffers and co-owns their KV growth permits.
/// A step replaces DeltaNet arrays, so saved handles retain the committed values.
/// A restore rewinds compatible live state or adopts the saved buffers.
/// Its own growth permit charges fixed DeltaNet bytes plus a bounded page-table allowance.
/// The snapshot holds no logits.
pub(super) struct ResidentSnapshot {
    /// Every layer state shared (Delta tensors AND the attention caches' capacity buffers,
    /// zero-copy handles): an adoption restore takes them wholesale, so a snapshot of one
    /// conversation restores into another whose state was reset.
    layers: Vec<LayerState>,
    pos: u32,
    /// The entry's own charge: the fixed DeltaNet bytes plus the page-table allowance.
    charge: GrowthPermit,
    /// Co-owned shares of the source state's KV growth permits:
    /// the budget counts the KV bytes once while the source state or this snapshot lives,
    /// and the last holder's drop releases them.
    kv_permits: Vec<GrowthPermit>,
    generation: u64,
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

    /// Every shared layer state in model order (the test observer reads them).
    pub(super) fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// The Delta layer states, in model order (the restore path swaps them in).
    pub(super) fn delta(&self) -> impl Iterator<Item = &LayerState> {
        self.layers
            .iter()
            .filter(|layer| matches!(layer, LayerState::Delta { .. }))
    }
}

/// Resident state of one sequence.
pub(super) struct ResidentState {
    layers: Vec<LayerState>,
    pos: u32,
    poisoned: bool,
    /// The identity of this state's attention-cache buffers for the generation guard: a
    /// snapshot restores only into a state whose buffers still hold its positions, so `reset`
    /// and `restore` bump this counter and a stale snapshot is refused instead of silently
    /// aliasing a rewritten cache.
    generation: u64,
    /// The highest position this state's buffers have ever held: a commit
    /// that writes below it rewrites previously-published ground (a rewind followed by a
    /// divergent step), so it draws a fresh generation and every older snapshot of the
    /// rewritten positions stops restoring. A commit at or above it extends new ground and
    /// keeps the generation.
    high_water: u32,
    /// One growth permit per committed step. Together they hold the logical cache bytes.
    /// `reset` and drop release them. A failed step keeps the earlier permits until `reset`.
    permits: Vec<GrowthPermit>,
    /// The LOGICAL growth bytes of ONE position of ONE attention layer
    /// (`attn::growth_bytes(&weights.shape, weights.kv, activation)`), fixed at `zero`. `kv_bytes`
    /// multiplies it by the attention layer count and the valid length, so a state at `pos` charges
    /// exactly `pos * kv_growth_bytes(weights)` — the all-layer growth — which the permits hold.
    kv_growth_per_layer: u64,
    /// The growth quantum of the attention caches, in positions: `KV_STEP` for `zero` and the
    /// test step for `zero_with_step`. `reset` keeps it.
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
            generation: BUFFER_GENERATION.fetch_add(1, Ordering::Relaxed),
            high_water: 0,
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
    /// out, and the permits of its committed steps kept — without a native failure, which is
    /// terminal for the whole process.
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
    /// before the first native call and before any buffer is consumed.
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
                (RMixer::Delta(_), LayerState::Delta { rec, ring }) => {
                    super::delta::check_state_shapes(&weights.shape, rec, ring)?;
                }
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

    /// Bytes held by the permits of the committed steps. Public to the executor for the
    /// pool-derived prompt ceiling: the bytes the session's next reset releases.
    pub(super) fn charged_bytes(&self) -> Result<u64, EngError> {
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
    /// growth per layer). Otherwise it returns `MemoryPlan`, drops
    /// `permit`, and changes nothing.
    pub(super) fn commit_rows(
        &mut self,
        rows: u32,
        layers: Vec<LayerState>,
        permit: GrowthPermit,
    ) -> Result<(), EngError> {
        let pos = self.check_commit_rows(rows, &layers, &permit)?;
        // Writing below the high-water mark rewrites published ground - draw
        // a fresh generation so snapshots of the rewritten positions stop restoring.
        if self.pos < self.high_water {
            self.generation = BUFFER_GENERATION.fetch_add(1, Ordering::Relaxed);
        }
        self.high_water = self.high_water.max(pos);
        self.permits.push(permit);
        self.layers = layers;
        self.pos = pos;
        self.poisoned = false;
        Ok(())
    }

    /// Validates publication without changing state, so a batch can check every slot first.
    pub(super) fn check_commit_rows(
        &self,
        rows: u32,
        layers: &[LayerState],
        permit: &GrowthPermit,
    ) -> Result<u32, EngError> {
        let pos = self
            .pos
            .checked_add(rows)
            .ok_or_else(|| EngError::ContextLimit("position overflows u32".into()))?;
        let charged = self
            .charged_bytes()?
            .checked_add(permit.retained_bytes())
            .ok_or_else(|| EngError::MemoryPlan("session charge overflows u64".into()))?;
        let logical = kv_bytes(layers, self.kv_growth_per_layer);
        if logical != charged {
            return Err(EngError::MemoryPlan(format!(
                "session charge {charged} bytes differs from the logical cache bytes {logical}"
            )));
        }
        Ok(pos)
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
        // The entry charges the fixed DeltaNet bytes plus the bounded page-table
        // allowance of its alias - never the KV of its positions.
        let pages = allowance_pages(self.pos);
        let allowance = pages
            .checked_mul(std::mem::size_of::<Page>() as u64)
            .ok_or_else(|| EngError::MemoryPlan("alias allowance overflows u64".into()))?;
        let bytes = fixed_bytes(&self.layers)?
            .checked_add(allowance)
            .ok_or_else(|| EngError::MemoryPlan("snapshot bytes overflow u64".into()))?;
        let charge = budget.acquire_growth(bytes)?;
        let layers = self.layers.iter().map(LayerState::share).collect();
        let kv_permits = self.permits.iter().map(GrowthPermit::share).collect();
        Ok(ResidentSnapshot {
            layers,
            pos: self.pos,
            charge,
            kv_permits,
            generation: self.generation,
        })
    }

    /// Rewinds this live state to the snapshot's position. It makes no native call and
    /// copies no array: the attention caches stay exactly as they are (the generation guard
    /// proved this state's buffers still hold the snapshot's positions, and the cache only
    /// appends), the DeltaNet tensors swap to the snapshot's shared handles (a step on a
    /// Delta state always builds new arrays, so the shared ones are immutable), and the
    /// position rewinds. The permits stay: the capacity buffers and their charge are
    /// unchanged. The snapshot keeps its own permit. A snapshot of another generation, of a
    /// position the live state no longer extends, or of another Delta layout returns
    /// `BadRequest`. On error the state is unchanged.
    pub(super) fn restore(
        &mut self,
        snapshot: &ResidentSnapshot,
        weights: &ResidentWeights,
        budget: &mut SessionBudget,
    ) -> Result<(), EngError> {
        if snapshot.generation != self.generation {
            // A mismatched generation means the live state is NOT the one the snapshot came from
            // (the executor's prefix reuse restores into a RESET session, a fresh generation). The
            // snapshot shares EVERY layer - the attention caches' capacity buffers included - so
            // the live state ADOPTS them wholesale: its own layers, position, and generation become
            // the snapshot's, and the permits rebuild to the adopted extent. A refusal here would
            // break prefix reuse across turns. The layer layout must still match the model weights
            // before adoption.
            if snapshot.layers.len() != weights.layers.len() {
                return Err(EngError::BadRequest(
                    "snapshot layers do not match the model weights".into(),
                ));
            }
            // Acquire BEFORE release: a MemoryPlan denial must leave the
            // state unchanged (the documented error contract). The adopted extent's
            // permit is acquired first; only on success are the old layers replaced and
            // the old permits dropped.
            let adopted_layers: Vec<LayerState> =
                snapshot.layers.iter().map(LayerState::share).collect();
            let adopted_pos = snapshot.pos;
            let adopted_bytes = kv_bytes(&adopted_layers, self.kv_growth_per_layer);
            let new_permit = budget.acquire_growth(adopted_bytes)?;
            // Success: replace the state (the old permits drop, releasing their charge).
            self.layers = adopted_layers;
            self.pos = adopted_pos;
            // Adoption MINTS a fresh generation instead of rejoining the snapshot's old one. A
            // rejoined generation would let a later restore of a larger same-lineage snapshot take
            // the rewind fast path over the adoptee's own rewritten caches - fabricated history
            // with no error. A minted generation is held by no existing snapshot, so every later
            // restore adopts wholesale.
            self.generation = BUFFER_GENERATION.fetch_add(1, Ordering::Relaxed);
            drop(std::mem::take(&mut self.permits));
            self.permits = vec![new_permit];
            self.poisoned = false;
            return Ok(());
        }
        if self.is_poisoned() {
            // A poisoned state holds no layer states (a failed step took them out), so there
            // is nothing to rewind: refuse until `reset` builds a fresh generation.
            return Err(EngError::BadRequest(
                "a poisoned resident state cannot be restored".into(),
            ));
        }
        if self.pos < snapshot.pos {
            return Err(EngError::BadRequest(
                "the live state no longer extends the snapshot position".into(),
            ));
        }
        // The total layer count must match too (the rewind iterates Delta layers only, so
        // a popped Attn layer would pass the Delta check but leave a mismatched layout).
        if snapshot.layers.len() != weights.layers.len() {
            return Err(EngError::BadRequest(
                "snapshot layers do not match the model weights".into(),
            ));
        }
        let mut saved = snapshot.delta();
        for layer in &weights.layers {
            match &layer.mixer {
                RMixer::Delta(_) => {
                    let entry = saved.next().ok_or_else(|| {
                        EngError::BadRequest(
                            "snapshot layers do not match the model weights".into(),
                        )
                    })?;
                    check_saved_layer(weights, &layer.mixer, entry, snapshot.pos)?;
                }
                RMixer::Attn(_) => {}
            }
        }
        if saved.next().is_some() {
            return Err(EngError::BadRequest(
                "snapshot layers do not match the model weights".into(),
            ));
        }
        // Acquire the rewound permit BEFORE any mutation. The opposite order (swap, rewind pos and
        // lens, then drop and acquire) could be denied mid-restore under shared-charge pins -
        // permits pinned by surviving snapshots release nothing on drop - leaving a half-rewound
        // state. A denial returns before the first mutation, with pos, layers, and permits exactly
        // as they were.
        let rewind_len = usize::try_from(snapshot.pos)
            .map_err(|_| EngError::BadRequest("the rewind position overflows usize".into()))?;
        let rewound = kv_bytes_at_len(&self.layers, Some(rewind_len), self.kv_growth_per_layer);
        let new_permit = budget.acquire_growth(rewound)?;
        let mut next_saved = snapshot.delta();
        for layer in &mut self.layers {
            if matches!(layer, LayerState::Delta { .. })
                && let Some(LayerState::Delta { rec, ring }) = next_saved.next()
            {
                *layer = LayerState::Delta {
                    rec: rec.share(),
                    ring: ring.share(),
                };
            }
        }
        self.pos = snapshot.pos;
        // The attention caches keep their capacity buffers but their VALID length rewinds
        // with the position (the rows beyond the rewind point become slack again), so the
        // `cache.len == pos` invariant of the next `check_step` holds. A rewind to position
        // zero returns the caches to their pristine lazy form: the append path treats
        // present buffers with zero valid rows as a shape mismatch.
        for layer in &mut self.layers {
            if let LayerState::Attn(cache) = layer {
                if rewind_len == 0 {
                    *cache = KvCache::with_form(weights.kv, self.kv_step);
                } else {
                    cache.len = rewind_len;
                }
            }
        }
        // Success: the permit is already acquired; swap it in (the old permits drop,
        // releasing their charge).
        drop(std::mem::take(&mut self.permits));
        self.permits = vec![new_permit];
        self.poisoned = false;
        Ok(())
    }

    /// Replaces this state with a fresh zero state. On error the state is unchanged. The state
    /// keeps its growth quantum: a state made with `zero_with_step` resets to the same step, not to
    /// `KV_STEP`. A reset REWRITES the attention buffers (a fresh zero holds new arrays), so it
    /// bumps the generation and every snapshot of the old generation stops restoring.
    pub(super) fn reset(
        &mut self,
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
    ) -> Result<(), EngError> {
        let kv_step = self.kv_step;
        // A reset builds fresh zero buffers, and `zero_with` draws a fresh unique generation,
        // so every snapshot of the old buffers stops restoring.
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
    use super::{KV_STEP, LayerState, Page, ResidentState, kv_storage_bytes};
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

    /// The tiny KV8 checkpoint and its host weights: head dimension 64.
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
    /// rows of one state: per group of 64 lanes the error is about
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
    /// value.
    /// The DeltaNet arrays of a state selected by kind: the layer layout interleaves Delta
    /// and attention layers, so a positional slice cannot pick them.
    fn live_delta(dec: &mut ResidentDecoder<'_>, state: &ResidentState) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for layer in state.layers() {
            if let LayerState::Delta { rec, ring } = layer {
                out.push(values(&mut dec.kernels, rec).unwrap());
                out.push(values(&mut dec.kernels, ring).unwrap());
            }
        }
        out
    }
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
    fn a_state_only_snapshot_charges_the_delta_state_and_aliases_the_kv() {
        // A snapshot adds DeltaNet state and a bounded page-table allowance.
        // Shared KV payloads retain their existing charge.
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&state).unwrap();
        let alias_allowance = 3 * KV_PER_TOKEN / 100;
        assert!(
            snapshot.charged_bytes() <= FIXED_BYTES + alias_allowance,
            "snapshot charged {} bytes; expected at most fixed {} plus alias allowance {}",
            snapshot.charged_bytes(),
            FIXED_BYTES,
            alias_allowance
        );
    }

    #[test]
    fn a_snapshot_charges_its_bytes_in_one_permit_and_releases_them_on_drop() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&state).unwrap();
        // The snapshot charges the fixed 19,456 plus the one-page alias allowance of
        // `size_of::<Page>()` (16) = 19,472 in one permit - never the KV of its positions.
        assert_eq!(
            snapshot.charged_bytes(),
            FIXED_BYTES + std::mem::size_of::<Page>() as u64,
            "snapshot charge"
        );
        assert_eq!(snapshot.pos(), 3, "snapshot position");
        // The state keeps its 3,072; the snapshot adds its own 19,472.
        assert_eq!(
            decoder.session_used(),
            3 * KV_PER_TOKEN + FIXED_BYTES + std::mem::size_of::<Page>() as u64,
            "state plus snapshot"
        );
        drop(snapshot);
        assert_eq!(decoder.session_used(), 3 * KV_PER_TOKEN, "after the drop");
    }

    #[test]
    fn a_snapshot_denied_by_the_budget_returns_memory_plan_and_charges_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        // Cap 10,000: after three steps 3,072 are used and 6,928 are free. The snapshot needs
        // 19,472.
        let mut decoder = decoder_with_cap(&ckpt, host, 10_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let denied = decoder.snapshot(&state);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m.contains("19472") && m.contains("6928")),
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
        // native owner terminal, so the test sets the poison directly.
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
        let delta_layers: Vec<LayerState> = snapshot.delta().map(LayerState::share).collect();
        let saved = read_state(&mut decoder, &delta_layers);
        // Three more steps move the live state past the snapshot; the restore rewinds it.
        prefill(&mut decoder, &mut live, &[4, 5, 6]);
        let used_before = decoder.session_used();
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(live.pos(), 3, "the position rewinds");
        assert!(!live.is_poisoned(), "poison");
        let after = live_delta(&mut decoder, &live);
        assert_eq!(
            after, saved,
            "the DeltaNet state returns to the saved values"
        );
        assert!(
            live.check_step(&decoder.weights, 32).is_ok(),
            "check_step after the rewind"
        );
        // The rewind under shared-charge: the live state drops its permits
        // and re-acquires the rewound extent, but the snapshot's co-owned shares of [0, 3)
        // keep the base charged, so the net budget change is zero - the growth above the
        // rewind point releases, and the snapshot's base re-charges through the new permit.
        assert_eq!(
            decoder.session_used(),
            used_before,
            "the rewind nets zero under shared-charge: the snapshot keeps the base alive"
        );
    }

    /// Suffix decoding after a snapshot restore matches a full recompute bitwise.
    #[test]
    fn a_restored_state_and_a_full_recompute_agree_bitwise_on_eight_suffix_tokens() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let prefix = [1u32, 2, 3];
        let suffix = [4u32, 5, 6, 7, 1, 2, 3, 4];
        // `live` steps past the snapshot, rewinds to it, then decodes the suffix; `recomputed`
        // prefills the prefix again and decodes the same suffix, as a request without reuse.
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &prefix);
        let snapshot = decoder.snapshot(&live).unwrap();
        prefill(&mut decoder, &mut live, &[4]);
        decoder.restore(&mut live, &snapshot).unwrap();
        let mut recomputed = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut recomputed, &prefix);
        for (t, &token) in suffix.iter().enumerate() {
            let from_rewind = decoder.step(&mut live, token, StepMode::Record).unwrap();
            let from_recompute = decoder
                .step(&mut recomputed, token, StepMode::Record)
                .unwrap();
            assert_eq!(
                from_rewind, from_recompute,
                "suffix token {t}: the rewound run differs from the full recompute"
            );
        }
        assert_eq!(live.pos(), 11, "position after the suffix");
    }

    #[test]
    fn live_steps_after_a_snapshot_leave_its_arrays_unchanged() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved_before = read_state(
            &mut decoder,
            &snapshot.delta().map(LayerState::share).collect::<Vec<_>>(),
        );
        // The entry holds the DeltaNet arrays only: two layers of recurrent state and ring.
        assert_eq!(saved_before.len(), 4, "array count");
        // Live steps past the snapshot; the shared Delta arrays are immutable, so the entry
        // keeps its values bitwise.
        prefill(&mut decoder, &mut live, &[4, 5]);
        let saved_after = read_state(
            &mut decoder,
            &snapshot.delta().map(LayerState::share).collect::<Vec<_>>(),
        );
        assert_eq!(
            saved_after, saved_before,
            "live steps changed the saved Delta arrays"
        );
        // A rewind restore brings the live Delta back to the saved values. The layer layout
        // interleaves Delta and attention layers, so the Delta arrays are selected by kind.
        decoder.restore(&mut live, &snapshot).unwrap();
        let mut after = Vec::new();
        for layer in live.layers() {
            if let LayerState::Delta { rec, ring } = layer {
                after.push(values(&mut decoder.kernels, rec).unwrap());
                after.push(values(&mut decoder.kernels, ring).unwrap());
            }
        }
        assert_eq!(after, saved_before, "the rewind returns the saved values");
    }

    /// A snapshot taken before further steps keeps its own
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
        let delta_layers: Vec<LayerState> = snapshot.delta().map(LayerState::share).collect();
        let saved = read_state(&mut decoder, &delta_layers);
        let suffix = [4u32, 5, 6, 7, 1, 2, 3];
        let mut straight = Vec::new();
        for &token in &suffix {
            straight.push(decoder.step(&mut live, token, StepMode::Record).unwrap());
        }
        let after = read_state(&mut decoder, &delta_layers);
        assert_eq!(
            after, saved,
            "live steps (an in-place append on the shared buffer and two growths) changed the snapshot"
        );
        // The live state really crossed the growth boundaries: caps 3 -> 6 -> 9.
        assert_eq!(kv_caps(live.layers()), vec![9, 9], "both caches grew twice");
        // Rewinding the live state to the snapshot and running the same suffix gives the
        // straight run (the restore rewinds this state, it does not build a new one).
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(live.pos(), 2, "the rewind returns to the snapshot position");
        for (index, &token) in suffix.iter().enumerate() {
            let got = decoder.step(&mut live, token, StepMode::Record).unwrap();
            assert_eq!(
                got.token, straight[index].token,
                "suffix token {index} differs after the rewind"
            );
            assert_eq!(
                got.logits, straight[index].logits,
                "suffix token {index} logits differ after the rewind"
            );
        }
        assert_eq!(live.pos(), 9, "position after the suffix");
    }

    /// After 30 steps with the default quantum (the test decoder's context limit is 32 positions)
    /// the actual storage (the sum of the tensors' `byte_len`) is exactly one quantum of capacity
    /// per attention layer, that is the logical bytes plus the unused `KV_STEP - 30` positions.
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
        // The exact hand-derived storage: 30 positions of the default
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
    fn a_poisoned_state_cannot_be_restored_until_reset() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        // A poisoned live state without a failed step: a real native failure marks the
        // process-wide native owner terminal, so the test poisons directly,
        // with the layer states taken out as a failed step leaves them.
        live.poison_for_test();
        assert!(live.is_poisoned(), "the state is poisoned for the test");
        let rejected = decoder.restore(&mut live, &snapshot);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("poisoned")),
            "{:?}",
            rejected.err()
        );
        assert!(live.is_poisoned(), "the refusal keeps the poison");
        // A reset builds a fresh generation; the restore ADOPTS the snapshot wholesale: layers,
        // position, and generation become the snapshot's, so the state is live at the snapshot's
        // position again.
        decoder.reset(&mut live).unwrap();
        let adopted = decoder.restore(&mut live, &snapshot);
        assert!(
            adopted.is_ok(),
            "the post-reset restore adopts the snapshot: {:?}",
            adopted.err()
        );
        assert_eq!(
            live.pos(),
            snapshot.pos(),
            "the adoption rewinds to the snapshot position"
        );
        assert!(!live.is_poisoned(), "the adoption clears the poison");
    }

    #[test]
    fn a_restore_adopts_a_foreign_snapshot_and_rejects_a_short_layout() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        // A snapshot of a DIFFERENT state is ADOPTED: the live state
        // takes its layers, position, and generation wholesale.
        let mut foreign_state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut foreign_state, &[1, 2, 3]);
        let foreign = decoder.snapshot(&foreign_state).unwrap();
        let adopted = decoder.restore(&mut live, &foreign);
        assert!(
            adopted.is_ok(),
            "the foreign snapshot restores by adoption: {:?}",
            adopted.err()
        );
        assert_eq!(
            live.pos(),
            3,
            "the adoption rewinds to the foreign snapshot position"
        );
        drop(foreign_state);
        // An entry with one Delta layer fewer than the model is refused the same way.
        let mut short = decoder.snapshot(&live).unwrap();
        short.layers.pop();
        let rejected = decoder.restore(&mut live, &short);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "{:?}",
            rejected.err()
        );
        assert_eq!(live.pos(), 3, "the live state is unchanged after a pop");
        // A rejection charges no permit: the live 3,072 and the two snapshots of 19,472 each, PLUS
        // the foreign snapshot's co-owned KV permits (shared-charge: the foreign state was dropped,
        // but its snapshot's shared permits keep its 3,072 KV bytes alive in the budget).
        assert_eq!(
            decoder.session_used(),
            3 * KV_PER_TOKEN
                + 2 * (FIXED_BYTES + std::mem::size_of::<Page>() as u64)
                + 3 * KV_PER_TOKEN,
            "charges while the rejected snapshots live"
        );
    }

    /// At the decoder level: eight decode steps of a Q8-cache
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

    /// The same comparison under BF16 activation, where the Q8
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

    /// At the decoder level, a Q8 run that crosses two capacity
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

    /// A snapshot taken before further Q8 steps keeps
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
        let delta_layers: Vec<LayerState> = snapshot.delta().map(LayerState::share).collect();
        let saved = read_state(&mut decoder, &delta_layers);
        // The snapshot holds the live arrays, part by part. A snapshot that swapped two parts
        // (scales for biases) would still round-trip through a restore that swaps them back, so
        // the round trip alone cannot see it.
        assert_eq!(
            live_delta(&mut decoder, &live),
            saved,
            "the Q8 snapshot differs from the live state it was taken from"
        );
        let suffix = [4u32, 5, 6, 7, 1, 2, 3];
        let mut straight = Vec::new();
        for &token in &suffix {
            straight.push(decoder.step(&mut live, token, StepMode::Record).unwrap());
        }
        let after = read_state(&mut decoder, &delta_layers);
        assert_eq!(
            after, saved,
            "live Q8 steps (an in-place append on the shared buffers and two growths) changed the snapshot"
        );
        assert_eq!(kv_caps(live.layers()), vec![9, 9], "the buffers grew twice");
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(
            live.pos(),
            2,
            "the Q8 rewind returns to the snapshot position"
        );
        for (index, &token) in suffix.iter().enumerate() {
            let got = decoder.step(&mut live, token, StepMode::Record).unwrap();
            assert_eq!(
                got.token, straight[index].token,
                "suffix token {index} differs after the Q8 rewind"
            );
            assert_eq!(
                got.logits, straight[index].logits,
                "suffix token {index} logits differ after the Q8 rewind"
            );
        }
        assert_eq!(live.pos(), 9, "position after the suffix");
    }

    /// The A, B, A isolation of the Q8
    /// path. The ONE live state rewinds to one snapshot, steps divergent tokens, rewinds again,
    /// and steps a different continuation; every rewound run is compared bit for bit with an
    /// independent straight run of the same tokens.
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
        // Run A: rewind, then 4 and 6.
        decoder.restore(&mut live, &snapshot).unwrap();
        let a_four = decoder.step(&mut live, 4, StepMode::Record).unwrap();
        let a_six = decoder.step(&mut live, 6, StepMode::Record).unwrap();
        // Run B: rewind again (over the divergent rows), then 5.
        decoder.restore(&mut live, &snapshot).unwrap();
        let b_five = decoder.step(&mut live, 5, StepMode::Record).unwrap();
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
            (a_four.token, a_four.logits),
            (want_four.token, want_four.logits),
            "run A token 4"
        );
        assert_eq!(
            (a_six.token, a_six.logits),
            (want_six.token, want_six.logits),
            "run A token 6"
        );
        assert_eq!(
            (b_five.token, b_five.logits),
            (want_five.token, want_five.logits),
            "run B token 5 after the divergent run A"
        );
        assert_eq!(live.pos(), 3, "run B ends one step past the rewind");
    }

    /// The stored bytes of a whole Q8 state equal the logical
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

    /// A Q8 decoder refuses a state whose attention cache holds
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
        let rejected = decoder.restore(&mut live, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("state shape")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(live.pos(), 3, "the live state is unchanged");
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
        let rejected = decoder.restore(&mut live, &wrong);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("state shape")),
            "{:?}",
            rejected.err()
        );
        assert_eq!(live.pos(), 3, "the live state is unchanged");
    }

    #[test]
    fn a_restore_at_position_zero_gives_an_empty_state_whose_next_step_commits() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        // A snapshot of the empty state aliases nothing yet; after one step the rewind returns
        // to position 0 and the next step matches a fresh zero state bitwise.
        let snapshot = decoder.snapshot(&live).unwrap();
        prefill(&mut decoder, &mut live, &[4]);
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(live.pos(), 0, "the rewind returns to position 0");
        let from_rewind = decoder.step(&mut live, 3, StepMode::Record).unwrap();
        let mut fresh = decoder.zero_state().unwrap();
        let from_zero = decoder.step(&mut fresh, 3, StepMode::Record).unwrap();
        assert_eq!(
            from_rewind, from_zero,
            "a step from the rewound position 0 differs from a fresh zero state"
        );
        assert_eq!(live.pos(), 1, "position after one step");
    }

    /// The A/B/A isolation gate: decode away from a snapshot, rewind, decode differently, rewind
    /// again - the Delta values return to the saved set after every cycle. These tests pass
    /// deterministically on the Mac (three runs).
    #[test]
    fn two_restores_of_one_snapshot_stay_independent_when_one_state_steps() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved = read_state(
            &mut decoder,
            &snapshot.delta().map(LayerState::share).collect::<Vec<_>>(),
        );
        // A: decode away from the snapshot, then rewind to it - twice. The values must return
        // to the saved set after every cycle (the A/B/A isolation gate of the assignment).
        for cycle in 0..2 {
            prefill(&mut decoder, &mut live, &[5 + cycle as u32, 6]);
            assert_ne!(live.pos(), 3, "the live state moved past the snapshot");
            decoder.restore(&mut live, &snapshot).unwrap();
            assert_eq!(live.pos(), 3, "cycle {cycle}: the position rewinds");
            let after = live_delta(&mut decoder, &live);
            assert_eq!(after, saved, "cycle {cycle}: the saved values return");
        }
        // The snapshot itself never moved.
        assert_eq!(
            read_state(
                &mut decoder,
                &snapshot.delta().map(LayerState::share).collect::<Vec<_>>()
            ),
            saved,
            "the entry changed"
        );
    }

    /// Two entries of one live state rewind to their own saved Delta values
    /// however the live state stepped between the saves.
    #[test]
    fn two_restores_that_step_different_tokens_match_independent_straight_runs() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        let early = decoder.snapshot(&live).unwrap();
        let early_saved = read_state(
            &mut decoder,
            &early.delta().map(LayerState::share).collect::<Vec<_>>(),
        );
        prefill(&mut decoder, &mut live, &[3]);
        let late = decoder.snapshot(&live).unwrap();
        let late_saved = read_state(
            &mut decoder,
            &late.delta().map(LayerState::share).collect::<Vec<_>>(),
        );
        // Step divergent tokens, then rewind to each entry in turn: every entry returns its
        // own saved Delta values, not the other entry's and not the divergent steps'.
        prefill(&mut decoder, &mut live, &[5, 5]);
        decoder.restore(&mut live, &late).unwrap();
        assert_eq!(live.pos(), 3, "the late rewind");
        let after_late = live_delta(&mut decoder, &live);
        assert_eq!(
            after_late,
            &late_saved[..],
            "the late entry returns its values"
        );
        decoder.restore(&mut live, &early).unwrap();
        assert_eq!(live.pos(), 2, "the early rewind");
        let after_early = live_delta(&mut decoder, &live);
        assert_eq!(
            after_early,
            &early_saved[..],
            "the early entry returns its values"
        );
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
        // The rewound run continues from the snapshot with the chunked path (five rows in
        // chunks of two). The recompute run runs every token again from a zero state.
        decoder.restore(&mut live, &snapshot).unwrap();
        let from_restore = decoder
            .prefill(&mut live, &suffix, StepMode::Logits)
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
            live_delta(&mut decoder, &live),
            live_delta(&mut decoder, &recomputed),
            "the restored state differs from the recomputed state bitwise"
        );
        assert_eq!(live.pos(), 8, "position after prefix and suffix");
    }
    /// Cache GROWTH relocates the buffer and preserves positions [0, n), so a snapshot of length n
    /// survives the growth: restoring it gives bit-identical Delta values to a fresh prefill of the
    /// same tokens. The growth never bumps the generation.
    #[test]
    fn growth_then_restore_is_bit_identical_to_a_fresh_prefill() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut live =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved = live_delta(&mut decoder, &live);
        // Two growths cross the step-3 boundary.
        prefill(&mut decoder, &mut live, &[4, 5, 6]);
        assert!(kv_caps(live.layers())[0] >= 6, "the cache grew");
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(live.pos(), 3, "the growth-crossing restore rewinds");
        let after = live_delta(&mut decoder, &live);
        assert_eq!(after, saved, "the growth-crossing restore is bit-identical");
        // Review (d): the reference is a FRESH prefill of the same tokens, and the
        // comparison is the full state - attention caches included - not the saved
        // Delta set of the same lineage.
        let mut fresh =
            ResidentState::zero_with_step(&mut decoder.kernels, &decoder.weights, 3).unwrap();
        prefill(&mut decoder, &mut fresh, &[1, 2, 3]);
        assert_eq!(
            read_state(&mut decoder, live.layers()),
            read_state(&mut decoder, fresh.layers()),
            "the restored state equals a fresh prefill bitwise, attention included"
        );
    }

    /// Denial is atomic: a rewind whose permit acquire is denied leaves pos, cache lens, and
    /// permits exactly as they were (the documented on-error contract) and the next step still
    /// commits. An acquire-after-mutation ordering would half-rewind the state on this denial - pos
    /// and lens rewound, permits dropped - and wedge every later commit on the charge-equality
    /// check, so this test fails on that ordering.
    #[test]
    fn a_rewind_denied_by_the_budget_changes_nothing_and_the_next_step_commits() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        // Measure on a throwaway decoder: the charge at pos 6 and the rewound extent the
        // rewind acquire will need (the byte math is deterministic for the same weights).
        let (probe_ckpt, probe_host) = load();
        let mut probe = decoder_with_cap(&probe_ckpt, probe_host, 1_000_000);
        let mut p = probe.zero_state().unwrap();
        prefill(&mut probe, &mut p, &[1, 2]);
        let snap = probe.snapshot(&p).unwrap();
        prefill(&mut probe, &mut p, &[3, 4, 5, 6]);
        // WITH the pos-6 snapshot alive: a snapshot charges its own entry (the fixed
        // bytes plus the page allowance), and the denial scenario keeps one alive.
        let snap6_probe = probe.snapshot(&p).unwrap();
        let used_at_six = probe.session_used();
        drop(snap6_probe);
        // The rewind acquire charges the extent at the snapshot position: the same
        // single-formula count the restore itself uses (this test module is a child of
        // state.rs, so the private items are in reach).
        let rewound_extent = super::kv_bytes_at_len(p.layers(), Some(2), p.kv_growth_per_layer);
        drop(snap);
        drop(p);
        drop(probe);
        drop(probe_ckpt);
        // The cap holds the pos-6 charge WITH the live pos-6 snapshot's own entry charge
        // and everything except the rewound extent, so the pinned rewind acquire is
        // denied by one byte while the next decode step still fits.
        let cap = used_at_six + rewound_extent - 1;
        let mut decoder = decoder_with_cap(&ckpt, host, cap);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        let snap2 = decoder.snapshot(&live).unwrap();
        prefill(&mut decoder, &mut live, &[3, 4, 5, 6]);
        // A live pos-6 snapshot PINS all six permits: a drop-then-acquire order would release
        // nothing, so the rewound acquire is genuinely denied, and the checks below would catch the
        // half-rewound state that order left.
        let snap6 = decoder.snapshot(&live).unwrap();
        let before = decoder.session_used();
        let denied = decoder.restore(&mut live, &snap2);
        assert!(
            matches!(denied, Err(EngError::MemoryPlan { .. })),
            "the rewind acquire is denied: {denied:?}"
        );
        assert_eq!(live.pos(), 6, "the denial leaves the position unchanged");
        assert_eq!(
            decoder.session_used(),
            before,
            "the denial leaves the session charge unchanged"
        );
        prefill(&mut decoder, &mut live, &[1]);
        assert_eq!(live.pos(), 7, "the next step still commits");
        drop(snap6);
    }

    /// The adoption read: snapshot@4, rewind to 2, two divergent
    /// steps below high_water, then the pos-4 snapshot ADOPTS (a minted generation can
    /// never take the rewind fast path). The adopted rows [0,4) - attention caches
    /// included - equal the snapshot-time values bitwise, and a suffix run equals an
    /// independent straight run bitwise.
    #[test]
    fn an_adoption_after_divergent_rewrite_reads_the_snapshot_rows_and_matches_a_straight_run() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        // The independent straight run: [1, 2, 3, 4] then the suffix [5, 6, 7].
        let mut straight = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut straight, &[1, 2, 3, 4]);
        let straight_at_four = read_state(&mut decoder, straight.layers());
        prefill(&mut decoder, &mut straight, &[5, 6, 7]);
        let straight_final = read_state(&mut decoder, straight.layers());

        // The live lineage: snapshot@2, grow to 4, snapshot@4.
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2]);
        let snap2 = decoder.snapshot(&live).unwrap();
        prefill(&mut decoder, &mut live, &[3, 4]);
        let snap4 = decoder.snapshot(&live).unwrap();
        // Rewind to 2, then two divergent steps: rows 2-3 rewrite below high_water 4,
        // so the generation bumps.
        decoder.restore(&mut live, &snap2).unwrap();
        prefill(&mut decoder, &mut live, &[6, 7]);
        assert_eq!(live.pos(), 4, "the divergent rewrite reached pos 4");
        // The pos-4 snapshot restores by ADOPT: the pre-rewrite rows come back wholesale.
        decoder.restore(&mut live, &snap4).unwrap();
        assert_eq!(live.pos(), 4);
        let adopted = read_state(&mut decoder, live.layers());
        assert_eq!(
            adopted, straight_at_four,
            "the adopted rows are the pre-rewrite rows, attention included"
        );
        // The suffix run equals the straight run bitwise.
        prefill(&mut decoder, &mut live, &[5, 6, 7]);
        assert_eq!(
            read_state(&mut decoder, live.layers()),
            straight_final,
            "the suffix after adoption equals the independent straight run"
        );

        // The DISCRIMINATING form: the adoption lands in a FRESH state
        // whose high_water is zero, so the divergent steps bump nothing on the old
        // gen-rejoining code and the big same-lineage snapshot then reached it through
        // the REWIND fast path over fabricated rows. The mint makes that restore adopt
        // wholesale instead - the full-state assertion below fails on the old code.
        let mut fresh = decoder.zero_state().unwrap();
        decoder.restore(&mut fresh, &snap2).unwrap();
        assert_eq!(fresh.pos(), 2, "the fresh state adopts the small snapshot");
        prefill(&mut decoder, &mut fresh, &[6, 7]);
        assert_eq!(fresh.pos(), 4);
        decoder.restore(&mut fresh, &snap4).unwrap();
        assert_eq!(
            read_state(&mut decoder, fresh.layers()),
            straight_at_four,
            "the big same-lineage snapshot adopts wholesale, never rewinds over the rewrite"
        );
    }

    /// A commit that writes BELOW the snapshot's length (a rewind followed by a divergent step)
    /// rewrites published ground and draws a fresh generation; restoring the older snapshot then
    /// ADOPTS: the snapshot's pre-rewrite shared handles come back wholesale, discarding the
    /// divergent writes.
    #[test]
    fn a_write_below_the_snapshot_length_adopts_the_pre_rewrite_state() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved = live_delta(&mut decoder, &live);
        // Step past the snapshot (pos 6, high_water 6).
        prefill(&mut decoder, &mut live, &[4, 5, 6]);
        // Rewind to the snapshot (pos 3, high_water stays 6, generation stays).
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(live.pos(), 3);
        // One divergent step at pos 3 writes BELOW high_water 6: the generation bumps.
        prefill(&mut decoder, &mut live, &[7]);
        assert_eq!(live.pos(), 4);
        // The original snapshot (old generation) restores by ADOPT: the pre-rewrite
        // shared handles come back wholesale, discarding the divergent [42] write.
        let result = decoder.restore(&mut live, &snapshot);
        assert!(
            result.is_ok(),
            "the pre-rewrite snapshot restores by adoption: {:?}",
            result.err()
        );
        assert_eq!(
            live.pos(),
            3,
            "the adoption rewinds to the snapshot position"
        );
        let after = live_delta(&mut decoder, &live);
        assert_eq!(
            after, saved,
            "the adoption brings back the pre-rewrite Delta values"
        );
        // Review (d): the divergent attention-row write is discarded too - the adopted
        // FULL state equals a fresh prefill of the same tokens bitwise, so the stale
        // row at the adopted len boundary cannot hide behind a Delta-only comparison.
        let mut fresh = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut fresh, &[1, 2, 3]);
        assert_eq!(
            read_state(&mut decoder, live.layers()),
            read_state(&mut decoder, fresh.layers()),
            "the adopted state equals a fresh prefill bitwise, attention included"
        );
    }

    /// A commit that writes AT or ABOVE the snapshot's
    /// length extends new ground (no generation bump), so the snapshot still restores
    /// through the fast rewind path with bit-identical values.
    #[test]
    fn a_write_at_or_above_the_snapshot_length_preserves_the_rewind() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 1_000_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let saved = live_delta(&mut decoder, &live);
        // Steps at positions [3, 6) are AT or ABOVE the snapshot's length 3.
        prefill(&mut decoder, &mut live, &[4, 5, 6]);
        // The restore takes the rewind path (no generation bump from the above-snapshot
        // writes) and returns the snapshot's values.
        decoder.restore(&mut live, &snapshot).unwrap();
        assert_eq!(live.pos(), 3, "the at-or-above write preserves the rewind");
        let after = live_delta(&mut decoder, &live);
        assert_eq!(after, saved, "the rewind returns the saved values");
        // Review (d): values, not just is_ok - the rewound FULL state equals a fresh
        // prefill of the same tokens bitwise, attention included.
        let mut fresh = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut fresh, &[1, 2, 3]);
        assert_eq!(
            read_state(&mut decoder, live.layers()),
            read_state(&mut decoder, fresh.layers()),
            "the rewound state equals a fresh prefill bitwise, attention included"
        );
    }
}
