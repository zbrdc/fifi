//! Resident Flash-Next session state, with the step's check, poison, and commit.

use super::ops::{Bytes, dim, short_import, zeros};
use super::prefill::qsa::check_chunk_dense;
use super::qsa::check_cache;
use super::weights::{RAttn, ResidentWeights};
use crate::{
    EngError,
    kernels::MlxKernels,
    sched::{GrowthPermit, SessionBudget},
    tensor::{ExecutionTensor, Tensor},
};

/// Bytes of one F32 element, the resident QSA cache dtype.
const F32_BYTES: u64 = 4;

/// Bytes that one more token adds to the stored QSA caches. Each QSA layer adds `KVH * D` F32
/// values to K, the same to V, and `ID` to the raw indexer keys. The sparse step keeps the whole
/// cache and appends the same rows at every position, so the growth does not depend on the
/// position. It makes no native call. `commit` checks the charge against the stored bytes.
pub(super) fn qsa_growth_bytes(weights: &ResidentWeights) -> Result<u64, EngError> {
    let d = weights.shape.qsa.dims();
    let layers = weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.attn, RAttn::Qsa(_)))
        .count();
    u64::try_from(layers)
        .ok()
        .zip(
            u64::from(d.heads_kv)
                .checked_mul(u64::from(d.head_dim))
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(u64::from(d.idx_dim))),
        )
        .and_then(|(layers, values)| values.checked_mul(F32_BYTES)?.checked_mul(layers))
        .ok_or_else(|| EngError::MemoryPlan("qsa growth bytes overflow u64".into()))
}

/// Bytes that a chunk of `rows` tokens adds to the stored QSA caches: `rows` positions of
/// `qsa_growth_bytes`. The growth per position is constant while the QSA path keeps its whole
/// cache. It makes no native call.
pub(super) fn chunk_growth_bytes(weights: &ResidentWeights, rows: u32) -> Result<u64, EngError> {
    qsa_growth_bytes(weights)?
        .checked_mul(u64::from(rows))
        .ok_or_else(|| EngError::MemoryPlan("qsa chunk growth bytes overflow u64".into()))
}

/// Bytes of the QSA tensors in `layers`, summed from the tensors' own sizes.
fn qsa_bytes(layers: &[LayerState]) -> Result<u64, EngError> {
    let mut total = Bytes::default();
    for layer in layers {
        if let LayerState::Qsa { k, v, ik, len: _ } = layer {
            for tensor in [k, v, ik].into_iter().flatten() {
                total.tensor(tensor)?;
            }
        }
    }
    Ok(total.total())
}

/// Resident state of one layer, in the host layouts.
pub(super) enum LayerState {
    /// DeltaNet state `[VH, KD, VD]` and convolution ring `[CD, CK - 1]`.
    Delta {
        rec: ExecutionTensor,
        ring: ExecutionTensor,
    },
    /// QSA keys `[1, KVH, T, D]`, values, and raw indexer keys `[T, ID]`. `None` while `len` is 0.
    Qsa {
        k: Option<ExecutionTensor>,
        v: Option<ExecutionTensor>,
        ik: Option<ExecutionTensor>,
        len: usize,
    },
}

impl LayerState {
    /// Returns a layer state that shares every array with `self`. It copies no array.
    fn share(&self) -> Self {
        match self {
            Self::Delta { rec, ring } => Self::Delta {
                rec: rec.share(),
                ring: ring.share(),
            },
            Self::Qsa { k, v, ik, len } => Self::Qsa {
                k: k.as_ref().map(ExecutionTensor::share),
                v: v.as_ref().map(ExecutionTensor::share),
                ik: ik.as_ref().map(ExecutionTensor::share),
                len: *len,
            },
        }
    }
}

/// Bytes of the DeltaNet states and the PLE ring: the part of a state that does not grow with the position.
fn fixed_bytes(layers: &[LayerState], ple_ring: &ExecutionTensor) -> Result<u64, EngError> {
    let mut total = Bytes::default();
    for layer in layers {
        if let LayerState::Delta { rec, ring } = layer {
            total.tensor(rec)?;
            total.tensor(ring)?;
        }
    }
    total.tensor(ple_ring)?;
    Ok(total.total())
}

/// A committed `ResidentState` held by shared handles. Its arrays are immutable: a step builds new
/// arrays and never writes into an existing one. One growth permit charges all its bytes to the
/// session budget and releases them when the snapshot drops.
pub(super) struct ResidentSnapshot {
    layers: Vec<LayerState>,
    ple_ring: ExecutionTensor,
    history: Vec<u32>,
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

/// Resident state of one sequence. It mirrors the host `ModelState`; the PLE ring is `[W, (PCK - 1) * ngram]`.
pub(super) struct ResidentState {
    layers: Vec<LayerState>,
    ple_ring: ExecutionTensor,
    history: Vec<u32>,
    pos: u32,
    poisoned: bool,
    /// One growth permit per committed step. Together they hold the stored QSA bytes.
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
        let rec = vec![
            dim(d.value_heads())?,
            dim(d.key_dim())?,
            dim(d.value_dim())?,
        ];
        let ring = vec![dim(d.conv_dim())?, dim(d.conv_kernel() - 1)?];
        let ple = weights.shape.ple.dims();
        let cols = (ple.conv_kernel - 1)
            .checked_mul(ple.ngram)
            .ok_or_else(|| EngError::MemoryPlan("ple ring width overflows".into()))?;
        let mut sources: Vec<Tensor> = Vec::new();
        for layer in &weights.layers {
            if matches!(layer.attn, RAttn::Delta(_)) {
                sources.push(zeros(rec.clone())?);
                sources.push(zeros(ring.clone())?);
            }
        }
        sources.push(zeros(vec![dim(weights.shape.hyper.width())?, cols])?);
        let refs: Vec<&Tensor> = sources.iter().collect();
        let mut got = kernels.import_many(&refs)?.into_iter();
        let mut next = || got.next().ok_or_else(short_import);
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(match layer.attn {
                RAttn::Delta(_) => LayerState::Delta {
                    rec: next()?,
                    ring: next()?,
                },
                RAttn::Qsa(_) => LayerState::Qsa {
                    k: None,
                    v: None,
                    ik: None,
                    len: 0,
                },
            });
        }
        Ok(Self {
            layers,
            ple_ring: next()?,
            history: Vec::new(),
            pos: 0,
            poisoned: false,
            permits: Vec::new(),
        })
    }

    /// Position of the next token.
    pub(super) fn pos(&self) -> u32 {
        self.pos
    }

    /// Layer states in model order (test observer).
    #[cfg(test)]
    pub(super) fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// True after a failed step, until `reset` or `commit`.
    pub(super) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// The n-gram history after the last committed token.
    pub(super) fn history(&self) -> &[u32] {
        &self.history
    }

    /// The committed PLE ring `[W, (PCK - 1) * ngram]`.
    pub(super) fn ple_ring(&self) -> &ExecutionTensor {
        &self.ple_ring
    }

    /// Host checks before a step. They make no native call. On error the state is unchanged and not
    /// poisoned. A QSA cache that differs from the position returns `BadRequest`, and a position at
    /// `MAX_POSITIONS` returns `ContextLimit`.
    pub(super) fn check_step(&self, weights: &ResidentWeights) -> Result<(), EngError> {
        if self.poisoned {
            return Err(EngError::BadRequest(
                "resident state needs a reset after a failed step".into(),
            ));
        }
        if self.pos == u32::MAX {
            return Err(EngError::ContextLimit("position limit reached".into()));
        }
        if weights.layers.len() != self.layers.len()
            || weights.shape.ple_layer >= weights.layers.len()
        {
            return Err(EngError::BadRequest(
                "model layer count does not match the state or the PLE layer".into(),
            ));
        }
        for (layer, state) in weights.layers.iter().zip(&self.layers) {
            match (&layer.attn, state) {
                (RAttn::Delta(_), LayerState::Delta { .. }) => {}
                (RAttn::Qsa(_), LayerState::Qsa { len, .. }) => {
                    check_cache(*len, self.pos)?;
                }
                _ => {
                    return Err(EngError::BadRequest(
                        "layer state kind does not match its weights".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Host checks before a chunk of `rows` tokens. They make no native call. On error the state is
    /// unchanged and not poisoned. A chunk whose last row passes the dense QSA limit returns
    /// `ContextLimit`, also when the one-token path allows that position.
    pub(super) fn check_chunk(&self, weights: &ResidentWeights, rows: u32) -> Result<(), EngError> {
        self.check_step(weights)?;
        if rows == 0 {
            return Err(EngError::BadRequest(
                "a prefill chunk needs at least one token".into(),
            ));
        }
        if self.pos.checked_add(rows).is_none() {
            return Err(EngError::ContextLimit("position overflows u32".into()));
        }
        for state in &self.layers {
            if let LayerState::Qsa { len, .. } = state {
                check_chunk_dense(&weights.shape.qsa, *len, self.pos, rows)?;
            }
        }
        Ok(())
    }

    /// Poisons the state and takes its layer states out. The step drops each old layer as soon as
    /// its replacement graph exists. Only `commit` or `reset` clears it.
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
        ple_ring: ExecutionTensor,
        history: Vec<u32>,
        permit: GrowthPermit,
    ) -> Result<(), EngError> {
        self.commit_rows(1, layers, ple_ring, history, permit)
    }

    /// Publishes a step of `rows` tokens after its `evaluate` and readbacks succeeded (plan
    /// section 1, step 3). It keeps `permit` only if the total charge equals the stored QSA
    /// bytes. Otherwise it returns `MemoryPlan`, drops `permit`, and changes nothing.
    pub(super) fn commit_rows(
        &mut self,
        rows: u32,
        layers: Vec<LayerState>,
        ple_ring: ExecutionTensor,
        history: Vec<u32>,
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
        let stored = qsa_bytes(&layers)?;
        if stored != charged {
            return Err(EngError::MemoryPlan(format!(
                "session charge {charged} bytes differs from stored QSA bytes {stored}"
            )));
        }
        self.permits.push(permit);
        self.layers = layers;
        self.ple_ring = ple_ring;
        self.history = history;
        self.pos = pos;
        self.poisoned = false;
        Ok(())
    }

    /// Shares the committed state as a snapshot. It makes no native call and copies no array. One
    /// permit charges the QSA bytes and the fixed bytes to `budget`. A poisoned state returns
    /// `BadRequest`. A denied permit returns `MemoryPlan`. On error the state and the budget are unchanged.
    pub(super) fn snapshot(
        &self,
        budget: &mut SessionBudget,
    ) -> Result<ResidentSnapshot, EngError> {
        if self.is_poisoned() {
            return Err(EngError::BadRequest(
                "a poisoned resident state cannot be saved".into(),
            ));
        }
        let bytes = qsa_bytes(&self.layers)?
            .checked_add(fixed_bytes(&self.layers, &self.ple_ring)?)
            .ok_or_else(|| EngError::MemoryPlan("snapshot bytes overflow u64".into()))?;
        let charge = budget.acquire_growth(bytes)?;
        Ok(ResidentSnapshot {
            layers: self.layers.iter().map(LayerState::share).collect(),
            ple_ring: self.ple_ring.share(),
            history: self.history.clone(),
            pos: self.pos,
            charge,
        })
    }

    /// Replaces this state with the snapshot's state. It makes no native call and copies no array.
    /// The state keeps one permit for the restored QSA bytes, so the next `commit` check holds. The
    /// snapshot keeps its own permit, so the shared arrays count twice. A snapshot of another layer
    /// layout returns `BadRequest`. A denied permit returns `MemoryPlan`. On error the state is unchanged.
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
            .all(|(layer, state)| {
                matches!(
                    (&layer.attn, state),
                    (RAttn::Delta(_), LayerState::Delta { .. })
                        | (RAttn::Qsa(_), LayerState::Qsa { .. })
                )
            });
        if weights.layers.len() != snapshot.layers.len() || !same_kinds {
            return Err(EngError::BadRequest(
                "snapshot layers do not match the model weights".into(),
            ));
        }
        let permit = budget.acquire_growth(qsa_bytes(&snapshot.layers)?)?;
        self.layers = snapshot.layers.iter().map(LayerState::share).collect();
        self.ple_ring = snapshot.ple_ring.share();
        self.history = snapshot.history.clone();
        self.pos = snapshot.pos;
        self.poisoned = false; // a snapshot holds committed state
        self.permits = vec![permit];
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

    /// Resident bytes of the layer states and the PLE ring, plus 4 bytes per history token.
    pub(super) fn byte_len(&self) -> Result<u64, EngError> {
        let mut total = Bytes::default();
        for layer in &self.layers {
            match layer {
                LayerState::Delta { rec, ring } => {
                    total.tensor(rec)?;
                    total.tensor(ring)?;
                }
                LayerState::Qsa { k, v, ik, len: _ } => {
                    for t in [k, v, ik].into_iter().flatten() {
                        total.tensor(t)?;
                    }
                }
            }
        }
        total.tensor(&self.ple_ring)?;
        let history = u64::try_from(self.history.len())
            .ok()
            .and_then(|n| n.checked_mul(4));
        total.add(
            history.ok_or_else(|| EngError::MemoryPlan("history byte count overflows".into()))?,
        )?;
        Ok(total.total())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::load::FlashWeights;
    use super::super::fixtures;
    use super::super::ops::{assert_close, values, zeros};
    use super::super::qsa::MAX_POSITIONS;
    use super::super::{ResidentDecoder, StepMode};
    use super::*;
    use crate::sched::SessionBudget;

    #[test]
    fn zero_state_uses_the_host_layouts() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let state = ResidentState::zero(&mut kernels, &weights).unwrap();
        let LayerState::Delta { rec, ring } = &state.layers[0] else {
            panic!("layer 0 is not DeltaNet");
        };
        let LayerState::Qsa { k, v, ik, len } = &state.layers[1] else {
            panic!("layer 1 is not QSA");
        };
        assert_eq!(rec.shape(), &[2, 32, 32][..], "delta rec shape");
        assert_eq!(ring.shape(), &[128, 3][..], "delta ring shape");
        assert!(
            k.is_none() && v.is_none() && ik.is_none(),
            "qsa caches are empty"
        );
        assert_eq!(*len, 0, "qsa length");
        assert_eq!(state.ple_ring.shape(), &[128, 9][..], "ple ring shape");
        assert_eq!(state.pos(), 0, "position");
        assert!(state.history.is_empty(), "history");
        assert!(!state.poisoned, "poison");
        assert_close(
            &values(&mut kernels, rec).unwrap(),
            &vec![0.0; 2048],
            "delta rec zeros",
        );
        assert_eq!(state.byte_len().unwrap(), 14_336, "state bytes");
    }

    #[test]
    fn reset_clears_position_history_and_poison() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let mut state = ResidentState::zero(&mut kernels, &weights).unwrap();
        state.pos = 3;
        state.history = vec![1, 2];
        state.poisoned = true;
        assert_eq!(
            state.byte_len().unwrap(),
            14_344,
            "history adds 4 bytes per token"
        );
        state.reset(&mut kernels, &weights).unwrap();
        assert_eq!(state.pos(), 0, "position after reset");
        assert!(state.history.is_empty(), "history after reset");
        assert!(!state.poisoned, "poison after reset");
        assert_eq!(state.byte_len().unwrap(), 14_336, "state bytes after reset");
    }

    /// A QSA layer state of `len` zero positions in the fixture layout: KVH 1, D 64, ID 32, F32.
    fn qsa_layer(kernels: &mut MlxKernels, len: u32) -> LayerState {
        let kv = zeros(vec![1, 1, len, 64]).unwrap();
        let ik = zeros(vec![len, 32]).unwrap();
        LayerState::Qsa {
            k: Some(kernels.import_tensor(&kv).unwrap()),
            v: Some(kernels.import_tensor(&kv).unwrap()),
            ik: Some(kernels.import_tensor(&ik).unwrap()),
            len: len as usize,
        }
    }

    #[test]
    fn the_qsa_layer_helper_holds_640_bytes_per_position() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        // One position: K and V hold 1 * 64 F32 values each, the indexer key holds 32:
        // (64 + 64 + 32) * 4 = 640 bytes. Three positions: 1,920 bytes.
        let layer = qsa_layer(&mut kernels, 3);
        assert_eq!(
            qsa_bytes(&[layer]).unwrap(),
            1_920,
            "bytes of three positions"
        );
    }

    #[test]
    fn growth_bytes_counts_one_position_of_every_qsa_layer() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        // The fixture has one QSA layer (layer 1). Per token: (2 * 64 + 32) values of 4 bytes.
        assert_eq!(qsa_growth_bytes(&weights).unwrap(), 640, "growth bytes");
    }

    #[test]
    fn a_commit_whose_charge_differs_from_the_stored_bytes_is_rejected() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let mut state = ResidentState::zero(&mut kernels, &weights).unwrap();
        let mut budget = SessionBudget::new(10_000);
        // The new layer 1 stores one position (640 bytes). The permit charges 639.
        let permit = budget.acquire_growth(639).unwrap();
        let mut layers = state.begin();
        layers[1] = qsa_layer(&mut kernels, 1);
        let ring = state.ple_ring().share();
        let rejected = state.commit(layers, ring, vec![1], permit);
        assert!(
            matches!(&rejected, Err(EngError::MemoryPlan(m)) if m.contains("639") && m.contains("640")),
            "{rejected:?}"
        );
        assert_eq!(budget.used(), 0, "the rejected permit is released");
        assert_eq!(state.pos(), 0, "nothing is published");
        assert!(state.history().is_empty(), "history is unchanged");
        assert!(state.is_poisoned(), "a rejected commit leaves the poison");
    }

    #[test]
    fn a_committed_permit_stays_charged_until_reset() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let mut state = ResidentState::zero(&mut kernels, &weights).unwrap();
        let mut budget = SessionBudget::new(10_000);
        let permit = budget.acquire_growth(640).unwrap();
        let mut layers = state.begin();
        layers[1] = qsa_layer(&mut kernels, 1);
        let ring = state.ple_ring().share();
        state.commit(layers, ring, vec![1], permit).unwrap();
        assert_eq!(state.pos(), 1, "position after commit");
        assert!(!state.is_poisoned(), "commit clears the poison");
        assert_eq!(budget.used(), 640, "the state holds its permit");
        state.reset(&mut kernels, &weights).unwrap();
        assert_eq!(budget.used(), 0, "reset releases the permit");
    }

    #[test]
    fn check_step_rejects_the_context_limit_and_a_wrong_cache_length() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let mut state = ResidentState::zero(&mut kernels, &weights).unwrap();
        // The check reads only the position and the cache lengths, so empty caches with a length
        // field stand in for caches of that many positions.
        let qsa = |len: u32| LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: len as usize,
        };
        state.pos = MAX_POSITIONS - 1;
        state.layers[1] = qsa(MAX_POSITIONS - 1);
        assert!(
            state.check_step(&weights).is_ok(),
            "the last position was rejected"
        );
        state.pos = MAX_POSITIONS;
        state.layers[1] = qsa(MAX_POSITIONS);
        assert!(
            matches!(state.check_step(&weights), Err(EngError::ContextLimit(_))),
            "position {MAX_POSITIONS} was not rejected as a context limit"
        );
        assert!(!state.is_poisoned(), "a context limit does not poison");
        state.pos = 5;
        state.layers[1] = qsa(4);
        assert!(
            matches!(state.check_step(&weights), Err(EngError::BadRequest(_))),
            "a cache of 4 positions at position 5 was not rejected as a bad request"
        );
    }

    #[test]
    fn the_context_limit_is_checked_before_the_growth_permit() {
        let _serial = crate::native_test_lock();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        // Cap 639: a step needs 640 bytes, so every step that reaches the permit is denied.
        decoder.set_session_budget(SessionBudget::new(639));
        let mut state = decoder.zero_state().unwrap();
        let denied = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m == "session growth requires 640 bytes, available 639"),
            "{denied:?}"
        );
        // The context check reads only the position and the cache lengths (see the test above).
        state.pos = MAX_POSITIONS;
        state.layers[1] = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: MAX_POSITIONS as usize,
        };
        let past = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(matches!(past, Err(EngError::ContextLimit(_))), "{past:?}");
        assert!(!state.is_poisoned(), "a context limit does not poison");
        assert_eq!(decoder.session_used(), 0, "no step holds a permit");
    }

    /// A prefill whose last token would sit at `MAX_POSITIONS` returns `ContextLimit` before any
    /// native call. The decode step allows one more position than this prefill reaches, so only the
    /// whole-prompt check rejects it. The state keeps its position and is not poisoned.
    #[test]
    fn a_prefill_past_the_position_limit_returns_context_limit_and_changes_nothing() {
        let _serial = crate::native_test_lock();
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        decoder.open_prefill(2).unwrap();
        let mut state = decoder.zero_state().unwrap();
        // The checks read only the position and the cache lengths (see the tests above).
        state.pos = MAX_POSITIONS - 1;
        state.layers[1] = LayerState::Qsa {
            k: None,
            v: None,
            ik: None,
            len: (MAX_POSITIONS - 1) as usize,
        };
        let past = decoder.prefill(&mut state, &[1, 2], StepMode::Prompt);
        assert!(matches!(past, Err(EngError::ContextLimit(_))), "{past:?}");
        assert_eq!(state.pos(), MAX_POSITIONS - 1, "position");
        assert!(!state.is_poisoned(), "a context limit does not poison");
        assert_eq!(decoder.session_used(), 0, "no charge");
    }

    #[test]
    fn a_commit_that_stores_fewer_bytes_than_the_permits_hold_is_rejected() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        let mut state = ResidentState::zero(&mut kernels, &weights).unwrap();
        let mut budget = SessionBudget::new(10_000);
        // Two committed steps hold 2 * 640 = 1,280 bytes and store two positions.
        for len in 1..=2u32 {
            let permit = budget.acquire_growth(640).unwrap();
            let mut layers = state.begin();
            layers[1] = qsa_layer(&mut kernels, len);
            let ring = state.ple_ring().share();
            state
                .commit(layers, ring, vec![1; len as usize], permit)
                .unwrap();
        }
        assert_eq!(budget.used(), 1_280, "two permits are held");
        // A trim to one position stores 640 bytes. No call releases part of a charge, so the new
        // permit of 0 bytes leaves 1,280 bytes charged against 640 stored, and the commit refuses.
        let permit = budget.acquire_growth(0).unwrap();
        let mut layers = state.begin();
        layers[1] = qsa_layer(&mut kernels, 1);
        let ring = state.ple_ring().share();
        let rejected = state.commit(layers, ring, vec![1], permit);
        assert!(
            matches!(&rejected, Err(EngError::MemoryPlan(m)) if m == "session charge 1280 bytes differs from stored QSA bytes 640"),
            "{rejected:?}"
        );
        assert_eq!(budget.used(), 1_280, "the earlier permits stay held");
        assert_eq!(state.pos(), 2, "nothing is published");
        assert_eq!(state.history().len(), 2, "the history is unchanged");
        state.reset(&mut kernels, &weights).unwrap();
        assert_eq!(budget.used(), 0, "reset releases every permit");
    }

    /// A chunk of `rows` tokens adds `rows` positions of 640 bytes (the fixture has one QSA layer).
    #[test]
    fn a_chunk_charge_is_rows_times_one_position() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = super::super::weights::synthetic(&mut kernels);
        assert_eq!(
            chunk_growth_bytes(&weights, 3).unwrap(),
            1_920,
            "three positions"
        );
        assert_eq!(
            chunk_growth_bytes(&weights, 1).unwrap(),
            qsa_growth_bytes(&weights).unwrap(),
            "one position"
        );
        assert_eq!(chunk_growth_bytes(&weights, 0).unwrap(), 0, "no rows");
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::super::super::load::FlashWeights;
    use super::super::ops::values;
    use super::super::{ResidentDecoder, StepMode, fixtures};
    use super::{LayerState, ResidentState};
    use crate::{
        EngError, kernels::MlxKernels, sched::SessionBudget, tensor::ExecutionTensor,
        weights::StreamedCheckpoint,
    };

    /// Bytes that one token adds to the QSA caches of the fixture: (2 * 64 + 32) values of 4 bytes.
    const QSA_PER_TOKEN: u64 = 640;
    /// Bytes of the fixture state that do not grow: DeltaNet `rec` 2 * 32 * 32 * 4 = 8,192,
    /// the convolution ring 128 * 3 * 4 = 1,536, and the PLE ring 128 * 9 * 4 = 4,608.
    const FIXED_BYTES: u64 = 14_336;

    fn load() -> (StreamedCheckpoint, FlashWeights) {
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        (ckpt, flash)
    }

    /// A decoder on the synthetic checkpoint whose session budget holds `cap` bytes.
    fn decoder_with_cap<'c>(
        ckpt: &'c StreamedCheckpoint,
        flash: &FlashWeights,
        cap: u64,
    ) -> ResidentDecoder<'c> {
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), ckpt, flash).unwrap();
        decoder.set_session_budget(SessionBudget::new(cap));
        decoder
    }

    /// Runs `tokens` as `Prompt` steps.
    fn prefill(decoder: &mut ResidentDecoder<'_>, state: &mut ResidentState, tokens: &[u32]) {
        for &token in tokens {
            decoder.step(state, token, StepMode::Prompt).unwrap();
        }
    }

    /// Every array of `layers`, then `ple_ring`, in model order.
    fn arrays<'a>(
        layers: &'a [LayerState],
        ple_ring: &'a ExecutionTensor,
    ) -> Vec<&'a ExecutionTensor> {
        let mut out = Vec::new();
        for layer in layers {
            match layer {
                LayerState::Delta { rec, ring } => out.extend([rec, ring]),
                LayerState::Qsa { k, v, ik, len: _ } => {
                    out.extend([k, v, ik].into_iter().flatten());
                }
            }
        }
        out.push(ple_ring);
        out
    }

    /// The `len` of every QSA layer in `layers`, in model order.
    fn qsa_lens(layers: &[LayerState]) -> Vec<usize> {
        layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Qsa { len, .. } => Some(*len),
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
    fn three_prompt_steps_leave_the_hand_derived_position_history_and_bytes() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        assert_eq!(state.pos(), 3, "position");
        // The PLE history keeps the last two tokens.
        assert_eq!(state.history(), &[2, 3][..], "history");
        // QSA 3 * 640 = 1,920 bytes, fixed 14,336 bytes, and 2 history tokens of 4 bytes.
        assert_eq!(
            state.byte_len().unwrap(),
            FIXED_BYTES + 3 * QSA_PER_TOKEN + 8,
            "state bytes"
        );
        assert_eq!(decoder.session_used(), 3 * QSA_PER_TOKEN, "step charges");
    }

    #[test]
    fn a_snapshot_charges_its_bytes_in_one_permit_and_releases_them_on_drop() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&state).unwrap();
        // The snapshot charges QSA 1,920 + fixed 14,336 = 16,256 bytes. The state keeps its 1,920.
        assert_eq!(snapshot.charged_bytes(), 16_256, "snapshot charge");
        assert_eq!(snapshot.pos(), 3, "snapshot position");
        assert_eq!(decoder.session_used(), 18_176, "state plus snapshot");
        drop(snapshot);
        assert_eq!(decoder.session_used(), 1_920, "after the drop");
    }

    #[test]
    fn a_snapshot_denied_by_the_budget_returns_memory_plan_and_charges_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        // Cap 10,000: after three steps 1,920 are used and 8,080 are free. The snapshot needs 16,256.
        let mut decoder = decoder_with_cap(&ckpt, &flash, 10_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1, 2, 3]);
        let denied = decoder.snapshot(&state);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m.contains("16256") && m.contains("8080")),
            "{:?}",
            denied.err()
        );
        assert_eq!(decoder.session_used(), 1_920, "a denial charges nothing");
        assert_eq!(state.pos(), 3, "the state is unchanged");
        assert!(!state.is_poisoned(), "a denial does not poison");
    }

    #[test]
    fn a_poisoned_state_cannot_be_saved() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut state = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut state, &[1]);
        // Token 8 is outside the 8-row embedding: the step fails after `begin` and poisons the state.
        assert!(decoder.step(&mut state, 8, StepMode::Prompt).is_err());
        assert!(state.is_poisoned(), "the failed step poisons the state");
        let saved = decoder.snapshot(&state);
        assert!(
            matches!(&saved, Err(EngError::BadRequest(_))),
            "{:?}",
            saved.err()
        );
        assert_eq!(
            decoder.session_used(),
            640,
            "only the first step is charged"
        );
    }

    #[test]
    fn restore_gives_a_fresh_state_the_snapshot_position_history_and_shared_arrays() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut restored = decoder.zero_state().unwrap();
        decoder.restore(&mut restored, &snapshot).unwrap();
        assert_eq!(restored.pos(), 3, "position");
        assert_eq!(restored.history(), &[2, 3][..], "history");
        // Each of the three steps appends one position, so the one QSA layer holds three.
        assert_eq!(qsa_lens(live.layers()), vec![3], "live QSA length");
        assert_eq!(qsa_lens(restored.layers()), vec![3], "restored QSA length");
        assert!(!restored.is_poisoned(), "poison");
        assert_eq!(
            restored.byte_len().unwrap(),
            live.byte_len().unwrap(),
            "state bytes"
        );
        assert!(
            restored.ple_ring().shares_storage_with(live.ple_ring()),
            "the restored ring is the shared array, not a copy"
        );
        // Live 1,920 + snapshot 16,256 + the restored state's own QSA permit 1,920.
        assert_eq!(
            decoder.session_used(),
            20_096,
            "charges while all three live"
        );
        drop(snapshot);
        assert_eq!(decoder.session_used(), 3_840, "after the snapshot drops");
        decoder.reset(&mut restored).unwrap();
        assert_eq!(decoder.session_used(), 1_920, "after the reset");
    }

    #[test]
    fn a_restored_state_and_a_full_recompute_agree_bitwise_on_eight_suffix_tokens() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
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
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let live_before = read(&mut decoder, arrays(live.layers(), live.ple_ring()));
        let saved_before = read(&mut decoder, arrays(&snapshot.layers, &snapshot.ple_ring));
        assert_eq!(
            saved_before, live_before,
            "the snapshot holds the live values"
        );
        // Layout: DeltaNet rec and ring, QSA keys, values, and indexer keys, then the PLE ring.
        assert_eq!(saved_before.len(), 6, "array count");
        assert_eq!(qsa_lens(&snapshot.layers), vec![3], "snapshot QSA length");
        assert_eq!(
            saved_before[2].len(),
            3 * 64,
            "QSA key values of three positions"
        );
        assert_eq!(
            saved_before[4].len(),
            3 * 32,
            "indexer key values of three positions"
        );
        assert!(
            saved_before[0].iter().any(|v| *v != 0.0),
            "the DeltaNet state is not all zero, so the test can see a change"
        );
        prefill(&mut decoder, &mut live, &[4, 5, 6, 7, 1]);
        let saved_after = read(&mut decoder, arrays(&snapshot.layers, &snapshot.ple_ring));
        assert_eq!(
            saved_after, saved_before,
            "live steps changed the snapshot arrays"
        );
        let live_after = read(&mut decoder, arrays(live.layers(), live.ple_ring()));
        assert_ne!(
            live_after[0], live_before[0],
            "the live DeltaNet state moved on"
        );
    }

    #[test]
    fn a_restore_denied_by_the_budget_returns_memory_plan_and_leaves_the_state() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        // Cap 19,000: live 1,920 + snapshot 16,256 = 18,176 are used and 824 are free.
        // The restore needs 1,920 bytes for the QSA permit of the restored state.
        let mut decoder = decoder_with_cap(&ckpt, &flash, 19_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        let mut target = decoder.zero_state().unwrap();
        let denied = decoder.restore(&mut target, &snapshot);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m.contains("1920") && m.contains("824")),
            "{:?}",
            denied.err()
        );
        assert_eq!(target.pos(), 0, "position");
        assert!(target.history().is_empty(), "history");
        assert_eq!(decoder.session_used(), 18_176, "a denial charges nothing");
    }

    #[test]
    fn restore_replaces_a_poisoned_state_and_the_next_step_commits() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let snapshot = decoder.snapshot(&live).unwrap();
        // The target holds one step (640 bytes) and a poisoned failed step.
        let mut target = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut target, &[5]);
        assert!(decoder.step(&mut target, 8, StepMode::Prompt).is_err());
        assert!(target.is_poisoned(), "the failed step poisons the target");
        // The failed step leaves the history of the one committed step, and the old PLE ring.
        assert_eq!(target.history(), &[5][..], "history before the restore");
        let old_ring = target.ple_ring().share();
        decoder.restore(&mut target, &snapshot).unwrap();
        assert!(!target.is_poisoned(), "restore clears the poison");
        assert_eq!(target.pos(), 3, "position");
        assert_eq!(
            target.history(),
            snapshot.history.as_slice(),
            "the restored history replaces the old one"
        );
        assert_eq!(target.history(), &[2, 3][..], "history");
        assert_eq!(qsa_lens(target.layers()), vec![3], "QSA length");
        // Every restored array is a handle of the snapshot array, not of the old state.
        let restored_arrays = arrays(target.layers(), target.ple_ring());
        let saved_arrays = arrays(&snapshot.layers, &snapshot.ple_ring);
        assert_eq!(restored_arrays.len(), saved_arrays.len(), "array count");
        for (i, (restored, saved)) in restored_arrays.iter().zip(&saved_arrays).enumerate() {
            assert!(restored.shares_storage_with(saved), "array {i} is shared");
        }
        assert!(
            !target.ple_ring().shares_storage_with(&old_ring),
            "the old PLE ring is gone"
        );
        // Live 1,920 + snapshot 16,256 + target 1,920. The old 640-byte permit is released.
        assert_eq!(decoder.session_used(), 20_096, "charges after the restore");
        decoder.step(&mut target, 4, StepMode::Prompt).unwrap();
        assert_eq!(target.pos(), 4, "position after one more step");
        assert_eq!(decoder.session_used(), 20_736, "one more step charges 640");
    }

    #[test]
    fn restore_rejects_a_snapshot_whose_layer_kinds_differ_from_the_model() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut snapshot = decoder.snapshot(&live).unwrap();
        // Layer 0 is DeltaNet and layer 1 is QSA in the model. Swap them in the snapshot.
        snapshot.layers.swap(0, 1);
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &snapshot);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }

    #[test]
    fn restore_rejects_a_snapshot_with_a_different_layer_count() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder = decoder_with_cap(&ckpt, &flash, 100_000);
        let mut live = decoder.zero_state().unwrap();
        prefill(&mut decoder, &mut live, &[1, 2, 3]);
        let mut snapshot = decoder.snapshot(&live).unwrap();
        snapshot.layers.pop();
        let mut target = decoder.zero_state().unwrap();
        let rejected = decoder.restore(&mut target, &snapshot);
        assert!(
            matches!(&rejected, Err(EngError::BadRequest(_))),
            "{:?}",
            rejected.err()
        );
        assert_eq!(target.pos(), 0, "the target is unchanged");
    }
}
