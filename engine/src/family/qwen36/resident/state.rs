//! Resident Q36 session state (packet Q36-D), with the step's check, poison, and commit.
//!
//! DeltaNet layers keep a recurrent state `[VH, VD, KD]` and a convolution ring `[CD, CK - 1]`.
//! Attention layers keep an F32 key-value cache that grows by one position per step. The growth
//! charge covers the cache only, as in the Flash-Next decoder (decision 0039 amendment). The fixed
//! DeltaNet state of 65,863,680 bytes is not charged by the steps; a snapshot charges it once with
//! its entry (decision 0036, addendum 1).

use super::attn::{KvCache, growth_bytes};
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
            Self::Attn(cache) => Self::Attn(KvCache {
                k: cache.k.as_ref().map(ExecutionTensor::share),
                v: cache.v.as_ref().map(ExecutionTensor::share),
                len: cache.len,
            }),
        }
    }
}

/// Bytes that one more token adds to the stored caches of all attention layers.
pub(super) fn kv_growth_bytes(weights: &ResidentWeights) -> Result<u64, EngError> {
    let layers = weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.mixer, RMixer::Attn(_)))
        .count();
    u64::try_from(layers)
        .ok()
        .and_then(|layers| growth_bytes(&weights.shape).ok()?.checked_mul(layers))
        .ok_or_else(|| EngError::MemoryPlan("kv growth bytes overflow u64".into()))
}

/// Bytes that a chunk of `rows` tokens adds to the stored caches: `rows` positions of
/// `kv_growth_bytes`. It makes no native call.
pub(super) fn chunk_growth_bytes(weights: &ResidentWeights, rows: u32) -> Result<u64, EngError> {
    kv_growth_bytes(weights)?
        .checked_mul(u64::from(rows))
        .ok_or_else(|| EngError::MemoryPlan("kv chunk growth bytes overflow u64".into()))
}

/// Bytes of the attention caches in `layers`, summed from the tensors' own sizes.
fn kv_bytes(layers: &[LayerState]) -> u64 {
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
            let kv_shape = [1, dim(a.heads_kv)?, dim(cache.len)?, dim(a.head_dim)?];
            let parts_ok = match (&cache.k, &cache.v) {
                (None, None) => cache.len == 0,
                (Some(k), Some(v)) => {
                    k.shape() == kv_shape.as_slice() && v.shape() == kv_shape.as_slice()
                }
                _ => false,
            };
            if usize::try_from(pos).ok() == Some(cache.len) && parts_ok {
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

/// A committed `ResidentState` held by shared handles (decision 0036, addendum 1). Its arrays are
/// immutable: a step builds new arrays and never writes into an existing one. One growth permit
/// charges all its bytes to the session budget and releases them when the snapshot drops. It holds
/// no logits (addendum 1 item 3).
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
    /// One growth permit per committed step. Together they hold the stored cache bytes.
    /// `reset` and drop release them. A failed step keeps the earlier permits until `reset`.
    permits: Vec<GrowthPermit>,
}

impl ResidentState {
    /// Imports a zero state with one `import_many` (one synchronization, outside the per-token budget).
    pub(super) fn zero(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
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
                RMixer::Attn(_) => LayerState::Attn(KvCache::empty()),
            });
        }
        Ok(Self {
            layers,
            pos: 0,
            poisoned: false,
            permits: Vec::new(),
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
    /// `ContextLimit`. A layer kind or cache length that differs from the weights or the position
    /// returns `BadRequest`.
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
                (RMixer::Attn(_), LayerState::Attn(cache))
                    if usize::try_from(self.pos).ok() == Some(cache.len) => {}
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
    /// `permit` only if the total charge equals the stored cache bytes. Otherwise it returns
    /// `MemoryPlan`, drops `permit`, and changes nothing.
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
        let stored = kv_bytes(&layers);
        if stored != charged {
            return Err(EngError::MemoryPlan(format!(
                "session charge {charged} bytes differs from stored cache bytes {stored}"
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
        let bytes = kv_bytes(&self.layers)
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
        let permit = match budget.acquire_growth(kv_bytes(&snapshot.layers)) {
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

    /// Replaces this state with a fresh zero state. On error the state is unchanged.
    pub(super) fn reset(
        &mut self,
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
    ) -> Result<(), EngError> {
        *self = Self::zero(kernels, weights)?;
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
    use super::{LayerState, ResidentState};
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

    /// Every array of `layers` in model order: the DeltaNet state and ring, then keys and values.
    fn arrays(layers: &[LayerState]) -> Vec<&ExecutionTensor> {
        let mut out = Vec::new();
        for layer in layers {
            match layer {
                LayerState::Delta { rec, ring } => out.extend([rec, ring]),
                LayerState::Attn(cache) => {
                    out.extend([cache.k.as_ref(), cache.v.as_ref()].into_iter().flatten())
                }
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

    /// Reads back every array in `list`.
    fn read(decoder: &mut ResidentDecoder<'_>, list: Vec<&ExecutionTensor>) -> Vec<Vec<f32>> {
        list.into_iter()
            .map(|tensor| values(&mut decoder.kernels, tensor).unwrap())
            .collect()
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
        let live_before = read(&mut decoder, arrays(live.layers()));
        let saved_before = read(&mut decoder, arrays(&snapshot.layers));
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
        let saved_after = read(&mut decoder, arrays(&snapshot.layers));
        assert_eq!(
            saved_after, saved_before,
            "live steps changed the snapshot arrays"
        );
        let live_after = read(&mut decoder, arrays(live.layers()));
        assert_ne!(
            live_after[0], saved_before[0],
            "the live DeltaNet state moved on"
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

    /// A saved attention cache that holds half a key-value pair is refused with `BadRequest`, in
    /// both directions.
    #[test]
    fn a_restore_rejects_a_saved_cache_that_holds_half_a_key_value_pair() {
        let _serial = crate::native_test_lock();
        let (ckpt, host) = load();
        let mut decoder = decoder_with_cap(&ckpt, host, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut target = decoder.zero_state().unwrap();
        let mut missing_v = decoder.snapshot(&live).unwrap();
        if let LayerState::Attn(cache) = &mut missing_v.layers[1] {
            cache.v = None;
        }
        let rejected = decoder.restore(&mut target, &missing_v);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "a cache without its values: {:?}",
            rejected.err()
        );
        let mut missing_k = decoder.snapshot(&live).unwrap();
        if let LayerState::Attn(cache) = &mut missing_k.layers[1] {
            cache.k = None;
        }
        let rejected = decoder.restore(&mut target, &missing_k);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "a cache without its keys: {:?}",
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
            cache.k = None;
            cache.v = None;
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
        let saved = read(&mut decoder, arrays(&snapshot.layers));
        let first_before = read(&mut decoder, arrays(first.layers()));
        let second_before = read(&mut decoder, arrays(second.layers()));
        assert_eq!(
            first_before, saved,
            "the first restore shares the saved arrays"
        );
        assert_eq!(
            second_before, saved,
            "the second restore shares the saved arrays"
        );
        decoder.step(&mut first, 4, StepMode::Prompt).unwrap();
        let first_after = read(&mut decoder, arrays(first.layers()));
        let second_after = read(&mut decoder, arrays(second.layers()));
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
            read(&mut decoder, arrays(restored.layers())),
            read(&mut decoder, arrays(recomputed.layers())),
            "the restored state differs from the recomputed state bitwise"
        );
        assert_eq!(restored.pos(), 8, "position after prefix and suffix");
    }
}
