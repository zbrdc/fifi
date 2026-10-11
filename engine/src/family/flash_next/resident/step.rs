//! Resident Flash-Next decode step and head (packet C4; S1-C plan sections 2.6 and 3).
//!
//! A `Prompt` token synchronizes `layers + 2` times (50 at 48 layers): the token-start import, the
//! router readback of every layer, and one `evaluate`. The expert page imports do not synchronize
//! (decision 0041). `Generate` adds the token readback (51). `Logits` adds the logits readback and
//! no token readback (51). `Record` reads the token, the logits, and the layer rows (53). Each
//! layer's old state drops as soon as its replacement graph exists. The state stays poisoned from
//! the first native call until `commit` (decision 0039, amendment
//! 2026-10-09).
//!
//! QSA growth charge: after the host checks and before `begin`, `step` acquires one `GrowthPermit`
//! for the step's new QSA bytes. A denial returns `MemoryPlan` and changes nothing. The permit
//! moves into `run` and then into `commit`, which keeps it. Any earlier return drops it. The
//! permit makes no native call, so the synchronization budget does not change.
//!
//! Expert page sets: `moe::step_paged` releases the previous layer's set after its router readback.
//! `step` releases the last set when `run` returns, with `run`'s graphs already dropped. In
//! `Generate` and `Record` the `evaluate` has consumed it. In `Prompt` the last layer's MoE output
//! is never evaluated, and after an error the failed step's graphs are never evaluated either, so
//! no evaluation can read or write those pages any more.

use super::ops::{Ctx, linear, values};
use super::state::{LayerState, ResidentState, qsa_growth_bytes};
use super::weights::RAttn;
use super::{ResidentDecoder, StepMode, StepResult, deltanet, hyper, moe, ple, qsa};
use crate::{
    EngError,
    kernels::MlxKernels,
    sched::GrowthPermit,
    tensor::{Dtype, ExecutionTensor},
};

/// Runs one token (host step.rs:137-165). The host checks and the growth permit run first and
/// change nothing on error: a denied permit returns `MemoryPlan`. After them, any error leaves the
/// state poisoned until `reset` and releases the step's permit.
pub(super) fn step(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    token: u32,
    mode: StepMode,
) -> Result<StepResult, EngError> {
    state.check_step(&decoder.weights)?;
    let growth = qsa_growth_bytes(&decoder.weights)?;
    let permit = decoder.budget.acquire_growth(growth)?;
    let old = state.begin();
    let result = run(decoder, state, old, permit, token, mode);
    decoder.slots.release_lent();
    result
}

/// Builds the token graph, evaluates it once, reads back, and commits (host step.rs:168-254).
/// `permit` moves into the commit. Every earlier return drops it.
fn run(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    old: Vec<LayerState>,
    permit: GrowthPermit,
    token: u32,
    mode: StepMode,
) -> Result<StepResult, EngError> {
    let ckpt = decoder.ckpt;
    let weights = &decoder.weights;
    let shape = &weights.shape;
    let pos = state.pos();
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &weights.consts,
    };
    // Plan 2.5 steps 1-4: the embedding and PLE rows, one synchronization.
    let start = ple::import_token(&mut ctx, ckpt, weights, state.history(), token)?;
    let mut h = start.hyper;
    let mut ple_ring = None;
    let mut layers = Vec::with_capacity(old.len());
    let mut records = Vec::new();
    for (index, (layer, old_layer)) in weights.layers.iter().zip(old).enumerate() {
        if index == shape.ple_layer {
            let out = ple::block(
                &mut ctx,
                &weights.ple,
                shape,
                &start.emb,
                &h,
                state.ple_ring(),
            )?;
            h = out.hyper;
            ple_ring = Some(out.ring);
        }
        let read = hyper::read(&mut ctx, &shape.hyper, &layer.attn_hyper, &h)?;
        let gate = read.gate.ok_or_else(|| {
            EngError::BadRequest("attention hyper-connection needs inject".into())
        })?;
        // The old layer state moves into its arm and drops at the end of the arm, after the
        // replacement graph exists. The next synchronization frees it (plan finding 0.2).
        let (block, next) = match (&layer.attn, old_layer) {
            (RAttn::Delta(w), LayerState::Delta { rec, ring }) => {
                let s = deltanet::step(&mut ctx, &shape.delta, w, &rec, &ring, &read.mixed)?;
                (
                    s.out,
                    LayerState::Delta {
                        rec: s.rec,
                        ring: s.ring,
                    },
                )
            }
            (RAttn::Qsa(w), cache @ LayerState::Qsa { .. }) => {
                let s = qsa::step(&mut ctx, &shape.qsa, w, &cache, pos, &read.mixed)?;
                (s.out, s.state)
            }
            _ => {
                return Err(EngError::BadRequest(
                    "layer state kind does not match its weights".into(),
                ));
            }
        };
        layers.push(next);
        h = hyper::apply(&mut ctx, &shape.hyper, &h, &gate, &block)?;
        let read = hyper::read(&mut ctx, &shape.hyper, &layer.mlp_hyper, &h)?;
        let gate = read
            .gate
            .ok_or_else(|| EngError::BadRequest("mlp hyper-connection needs inject".into()))?;
        // Plan 2.4 and decision 0041: the router readback, one synchronization.
        let (out, selection) = moe::step_paged(
            &mut ctx,
            ckpt,
            &shape.moe,
            &layer.moe,
            &mut decoder.slots,
            &read.mixed,
        )?;
        // Tests only: the S3 route recorder (packet T0) keeps the host selection of each layer.
        #[cfg(test)]
        if let Some(trace) = decoder.route_trace.as_mut() {
            let (routes, probs) = selection;
            trace.push(index, routes, probs)?;
        }
        #[cfg(not(test))]
        drop(selection);
        h = hyper::apply(&mut ctx, &shape.hyper, &h, &gate, &out)?;
        if mode == StepMode::Record {
            records.push(h.share());
        }
    }
    let ple_ring =
        ple_ring.ok_or_else(|| EngError::BadRequest("the PLE layer did not run".into()))?;
    // Plan 2.6 step 3: the head. `Prompt` skips the mixer and `lm_head`.
    let head = match mode {
        StepMode::Prompt => None,
        StepMode::Generate | StepMode::Record | StepMode::Logits => {
            let mixed = hyper::mixer_read(&mut ctx, &shape.hyper, &weights.mixer, &h)?;
            let logits = linear(&mut ctx, &weights.lm_head, &mixed)?;
            // `Logits` builds no `argmax`: the host sampler chooses the token.
            let next = if mode == StepMode::Logits {
                None
            } else {
                Some(ctx.kernels.argmax(&logits, 1, false)?)
            };
            Some((logits, next))
        }
    };
    let rows = if mode == StepMode::Record {
        let parts: Vec<&ExecutionTensor> = records.iter().collect();
        Some(ctx.kernels.concatenate(&parts, 0)?)
    } else {
        None
    };
    // Plan 2.6 step 4: one evaluate of the new state and the head outputs.
    let mut outputs: Vec<&ExecutionTensor> = Vec::new();
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
            LayerState::Qsa { k, v, ik, len: _ } => {
                outputs.extend([k, v, ik].into_iter().flatten());
            }
        }
    }
    outputs.push(&ple_ring);
    if let Some((logits, next)) = &head {
        if let Some(next) = next {
            outputs.push(next);
        }
        if matches!(mode, StepMode::Record | StepMode::Logits) {
            outputs.push(logits);
        }
    }
    if let Some(rows) = &rows {
        outputs.push(rows);
    }
    ctx.kernels.evaluate(&outputs)?;
    let token_out = match &head {
        Some((_, Some(next))) => Some(read_token(ctx.kernels, next, shape.vocab)?),
        _ => None,
    };
    let logits_out = match (&head, mode) {
        (Some((logits, _)), StepMode::Record | StepMode::Logits) => {
            Some(values(ctx.kernels, logits)?)
        }
        _ => None,
    };
    if mode == StepMode::Logits && logits_out.as_ref().map(Vec::len) != Some(shape.vocab) {
        return Err(EngError::Backend {
            operation: "flash_next",
            message: "logits length differs from the vocabulary".into(),
        });
    }
    let layer_out = match &rows {
        Some(rows) => Some(split_rows(values(ctx.kernels, rows)?, records.len())?),
        None => None,
    };
    state.commit(layers, ple_ring, start.history, permit)?;
    Ok(StepResult {
        token: token_out,
        logits: logits_out,
        layer_out,
    })
}

/// Reads the U32 `[1]` argmax back as one token inside the vocabulary. One synchronization.
pub(super) fn read_token(
    kernels: &mut MlxKernels,
    next: &ExecutionTensor,
    vocab: usize,
) -> Result<u32, EngError> {
    let host = kernels.readback(next)?;
    host.host_bytes()
        .filter(|_| host.dtype() == Dtype::U32)
        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
        .map(u32::from_le_bytes)
        .filter(|&t| usize::try_from(t).is_ok_and(|t| t < vocab))
        .ok_or_else(|| EngError::Backend {
            operation: "flash_next",
            message: "argmax did not return one token inside the vocabulary".into(),
        })
}

/// Splits the `[layers * C, H]` readback into one `[C * H]` row per layer.
pub(super) fn split_rows(flat: Vec<f32>, layers: usize) -> Result<Vec<Vec<f32>>, EngError> {
    if layers == 0 || !flat.len().is_multiple_of(layers) {
        return Err(EngError::Backend {
            operation: "flash_next",
            message: "recorded rows do not split by layer".into(),
        });
    }
    let width = flat.len() / layers;
    Ok(flat.chunks_exact(width).map(<[f32]>::to_vec).collect())
}

#[cfg(test)]
mod tests {
    use super::super::super::load::FlashWeights;
    use super::super::super::mlx::MlxProjector;
    use super::super::super::step::{LayerState as HostLayer, ModelState, step as host_step};
    use super::super::fixtures;
    use super::*;
    use crate::sched::SessionBudget;
    use crate::weights::StreamedCheckpoint;

    fn load() -> (StreamedCheckpoint, FlashWeights) {
        let ckpt = fixtures::checkpoint();
        let flash = FlashWeights::load(&ckpt, &fixtures::config()).unwrap();
        (ckpt, flash)
    }

    /// The process-wide synchronization count in qualification builds (weights.rs:689-699).
    fn syncs(kernels: &MlxKernels) -> Option<u64> {
        #[cfg(feature = "native-fault-qualification")]
        {
            Some(kernels.qualification_sync_count())
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = kernels;
            None
        }
    }

    /// Asserts cos >= 0.99999 and max |a - e| / max |e| <= 1e-4 (plan section 4, option B).
    fn assert_parity(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        let (mut dot, mut aa, mut ee, mut diff, mut top) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, &e) in actual.iter().zip(expected) {
            let (a, e) = (f64::from(a), f64::from(e));
            dot += a * e;
            aa += a * a;
            ee += e * e;
            diff = diff.max((a - e).abs());
            top = top.max(e.abs());
        }
        assert!(top > 0.0, "{what}: expected values are all zero");
        let cos = dot / (aa.sqrt() * ee.sqrt());
        assert!(cos >= 0.99999, "{what}: cos {cos}");
        assert!(diff / top <= 1e-4, "{what}: relative error {}", diff / top);
    }

    /// The lowest index of the largest logit: the device `argmax` tie rule.
    fn host_argmax(logits: &[f32]) -> u32 {
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    #[test]
    fn three_tokens_match_the_host_step_in_every_mode() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let shape = *flash.shape();
        let view = flash.view();
        let mut proj = MlxProjector::new(&ckpt).unwrap();
        let mut host = ModelState::new(&shape, &view);
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        // One state per mode sequence. The last token runs `Record` in all three, so the
        // `Generate` and `Prompt` commits are checked against the host through the next step.
        let mut record = decoder.zero_state().unwrap();
        let mut generate = decoder.zero_state().unwrap();
        let mut prompt = decoder.zero_state().unwrap();
        let tokens = [1u32, 2, 3];
        for (t, &token) in tokens.iter().enumerate() {
            let want = host_step(&mut proj, &shape, &view, &mut host, token, true).unwrap();
            let want_rows = want.layer_out.unwrap();
            let want_token = host_argmax(&want.logits);
            let last = t + 1 == tokens.len();
            let runs = [
                (&mut record, StepMode::Record),
                (
                    &mut generate,
                    if last {
                        StepMode::Record
                    } else {
                        StepMode::Generate
                    },
                ),
                (
                    &mut prompt,
                    if last {
                        StepMode::Record
                    } else {
                        StepMode::Prompt
                    },
                ),
            ];
            for (state, mode) in runs {
                let what = format!("token {t} {mode:?}");
                let got = decoder.step(state, token, mode).unwrap();
                assert_eq!(state.pos(), t as u32 + 1, "{what}: position");
                assert!(!state.is_poisoned(), "{what}: poison");
                match mode {
                    StepMode::Prompt => {
                        assert_eq!(
                            got,
                            StepResult {
                                token: None,
                                logits: None,
                                layer_out: None
                            },
                            "{what}"
                        );
                    }
                    StepMode::Generate => {
                        assert_eq!(
                            got,
                            StepResult {
                                token: Some(want_token),
                                logits: None,
                                layer_out: None
                            },
                            "{what}"
                        );
                    }
                    StepMode::Record => {
                        assert_eq!(got.token, Some(want_token), "{what}: argmax");
                        assert_parity(
                            &got.logits.unwrap(),
                            &want.logits,
                            &format!("{what} logits"),
                        );
                        let rows = got.layer_out.unwrap();
                        assert_eq!(rows.len(), want_rows.len(), "{what}: recorded layers");
                        for (l, (g, w)) in rows.iter().zip(&want_rows).enumerate() {
                            assert_parity(g, w, &format!("{what} layer {l}"));
                        }
                        // The committed DeltaNet state, compared directly: its effect on the rows and
                        // logits of a later token can stay inside the parity tolerance.
                        let mut compared = 0;
                        for (l, pair) in host.layers().iter().zip(state.layers()).enumerate() {
                            if let (HostLayer::Delta(h), LayerState::Delta { rec, ring }) = pair {
                                let rec = values(&mut decoder.kernels, rec).unwrap();
                                assert_parity(&rec, h.rec(), &format!("{what} layer {l} rec"));
                                let ring = values(&mut decoder.kernels, ring).unwrap();
                                assert_parity(&ring, h.ring(), &format!("{what} layer {l} ring"));
                                compared += 1;
                            }
                        }
                        assert!(compared > 0, "{what}: no DeltaNet layer compared");
                    }
                    StepMode::Logits => panic!("{what}: this test does not run Logits"),
                }
            }
        }
    }

    #[test]
    fn a_logits_step_returns_the_logits_of_a_record_step_and_no_token() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        let mut logits_state = decoder.zero_state().unwrap();
        let mut record_state = decoder.zero_state().unwrap();
        let mut generate_state = decoder.zero_state().unwrap();
        for (t, token) in [1u32, 2, 3].into_iter().enumerate() {
            let what = format!("token {t}");
            let want = decoder
                .step(&mut record_state, token, StepMode::Record)
                .unwrap();
            let got = decoder
                .step(&mut logits_state, token, StepMode::Logits)
                .unwrap();
            assert_eq!(got.token, None, "{what}: token");
            assert_eq!(got.layer_out, None, "{what}: layer rows");
            let logits = got.logits.unwrap();
            assert_eq!(logits.len(), 8, "{what}: the synthetic vocabulary");
            assert_parity(&logits, &want.logits.unwrap(), &format!("{what} logits"));
            let device = decoder
                .step(&mut generate_state, token, StepMode::Generate)
                .unwrap();
            assert_eq!(device.token, Some(host_argmax(&logits)), "{what}: argmax");
            assert_eq!(logits_state.pos(), t as u32 + 1, "{what}: position");
            assert!(!logits_state.is_poisoned(), "{what}: poison");
        }
    }

    #[test]
    fn a_denied_growth_permit_at_the_first_sparse_step_changes_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        // Ratio 4, budget 16: positions 0 to 18 are dense, position 19 is the first sparse step.
        // The 19 dense steps charge 19 * 640 = 12,160 bytes, so this cap denies position 19.
        decoder.set_session_budget(SessionBudget::new(12_160));
        let mut state = decoder.zero_state().unwrap();
        for pos in 0..19u32 {
            let ran = decoder.step(&mut state, pos % 7, StepMode::Prompt);
            assert!(ran.is_ok(), "position {pos}: {:?}", ran.err());
        }
        assert_eq!(state.pos(), 19, "position before the sparse step");
        assert_eq!(
            decoder.session_used(),
            12_160,
            "charge before the sparse step"
        );
        let bytes = state.byte_len().unwrap();
        let before = syncs(&decoder.kernels);
        let denied = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m == "session growth requires 640 bytes, available 0"),
            "{denied:?}"
        );
        assert_eq!(syncs(&decoder.kernels), before, "no synchronization");
        assert_eq!(state.pos(), 19, "position after the denial");
        assert!(!state.is_poisoned(), "a denial does not poison");
        assert_eq!(
            state.byte_len().unwrap(),
            bytes,
            "the layer states stay in place"
        );
        assert_eq!(decoder.session_used(), 12_160, "a denial charges nothing");
    }

    #[test]
    fn the_charge_and_the_stored_bytes_grow_640_per_position_through_the_sparse_switch() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        decoder.set_session_budget(SessionBudget::new(20_000));
        let mut state = decoder.zero_state().unwrap();
        // Ratio 4, budget 16: positions 0 to 18 are dense and positions 19 to 23 are sparse. The
        // complete-block count rises at positions 3, 7, 11, 15, 19, and 23. The fixture has one QSA
        // layer, and each position stores (2 * 64 + 32) values of 4 bytes: 640 bytes.
        for pos in 0..24u32 {
            let ran = decoder.step(&mut state, pos % 7, StepMode::Prompt);
            assert!(ran.is_ok(), "position {pos}: {:?}", ran.err());
            let want = 640 * u64::from(pos + 1);
            assert_eq!(decoder.session_used(), want, "charge after position {pos}");
            assert_eq!(
                stored_qsa_bytes(&state),
                want,
                "stored bytes after position {pos}"
            );
        }
    }

    #[test]
    fn a_failed_step_poisons_the_state_until_reset() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        let mut state = decoder.zero_state().unwrap();
        let mut control = decoder.zero_state().unwrap();
        decoder.step(&mut state, 1, StepMode::Prompt).unwrap();
        // Token 8 is outside the 8-row embedding. `read_slice` (stored.rs:289-294) fails on the
        // host after `begin` has poisoned the state, so the adapter stays usable.
        let failed = decoder.step(&mut state, 8, StepMode::Generate);
        assert!(
            matches!(
                failed,
                Err(EngError::BadRequest(_) | EngError::Checkpoint(_))
            ),
            "{failed:?}"
        );
        assert!(state.is_poisoned(), "a failed step poisons the state");
        assert_eq!(state.pos(), 1, "a failed step does not advance");
        let before = syncs(&decoder.kernels);
        let again = decoder.step(&mut state, 1, StepMode::Generate);
        assert!(
            matches!(&again, Err(EngError::BadRequest(m)) if m.contains("reset")),
            "{again:?}"
        );
        assert_eq!(
            syncs(&decoder.kernels),
            before,
            "a poisoned step makes no synchronization"
        );
        decoder.reset(&mut state).unwrap();
        assert!(!state.is_poisoned(), "reset clears the poison");
        assert_eq!(state.pos(), 0, "reset clears the position");
        let got = decoder.step(&mut state, 1, StepMode::Record).unwrap();
        let want = decoder.step(&mut control, 1, StepMode::Record).unwrap();
        assert_parity(
            &got.logits.unwrap(),
            &want.logits.unwrap(),
            "logits after reset",
        );
    }

    /// Bytes of the stored QSA tensors, summed from the tensors themselves.
    fn stored_qsa_bytes(state: &ResidentState) -> u64 {
        state
            .layers()
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Qsa { k, v, ik, len: _ } => Some([k, v, ik]),
                LayerState::Delta { .. } => None,
            })
            .flatten()
            .flatten()
            .map(ExecutionTensor::byte_len)
            .sum()
    }

    #[test]
    fn each_committed_step_charges_the_stored_qsa_growth() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        decoder.set_session_budget(SessionBudget::new(10_000));
        let mut state = decoder.zero_state().unwrap();
        assert_eq!(decoder.session_used(), 0, "a zero state is not charged");
        for t in 1..=3u32 {
            let ran = decoder.step(&mut state, t, StepMode::Prompt);
            assert!(ran.is_ok(), "step {t}: {:?}", ran.err());
            // The fixture has one QSA layer: (2 * 64 + 32) values of 4 bytes, 640 bytes per token.
            let want = 640 * u64::from(t);
            assert_eq!(decoder.session_used(), want, "charge after step {t}");
            assert_eq!(
                stored_qsa_bytes(&state),
                want,
                "stored bytes after step {t}"
            );
        }
    }

    #[test]
    fn a_denied_growth_permit_returns_memory_plan_and_changes_nothing() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        // Cap 1,000: one step charges 640 and fits; a second step would reach 1,280 and does not.
        decoder.set_session_budget(SessionBudget::new(1_000));
        let mut state = decoder.zero_state().unwrap();
        let first = decoder.step(&mut state, 1, StepMode::Prompt);
        assert!(first.is_ok(), "first step: {:?}", first.err());
        assert_eq!(decoder.session_used(), 640, "charge after the first step");
        let bytes = state.byte_len().unwrap();
        let before = syncs(&decoder.kernels);
        let denied = decoder.step(&mut state, 2, StepMode::Generate);
        assert!(
            matches!(&denied, Err(EngError::MemoryPlan(m)) if m == "session growth requires 640 bytes, available 360"),
            "{denied:?}"
        );
        assert_eq!(syncs(&decoder.kernels), before, "no synchronization");
        assert_eq!(state.pos(), 1, "position after the denial");
        assert!(!state.is_poisoned(), "a denial does not poison");
        assert_eq!(
            state.byte_len().unwrap(),
            bytes,
            "the layer states stay in place"
        );
        assert_eq!(decoder.session_used(), 640, "a denial charges nothing");
        decoder.reset(&mut state).unwrap();
        assert_eq!(decoder.session_used(), 0, "reset releases the charge");
        let again = decoder.step(&mut state, 2, StepMode::Generate);
        assert!(again.is_ok(), "step after reset: {:?}", again.err());
        assert_eq!(decoder.session_used(), 640, "charge after the reset step");
    }

    #[test]
    fn a_failed_step_releases_its_growth_permit() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        decoder.set_session_budget(SessionBudget::new(10_000));
        let mut state = decoder.zero_state().unwrap();
        let first = decoder.step(&mut state, 1, StepMode::Prompt);
        assert!(first.is_ok(), "first step: {:?}", first.err());
        assert_eq!(decoder.session_used(), 640, "charge after the first step");
        // Token 8 is outside the 8-row embedding: the step fails after `begin` (see
        // `a_failed_step_poisons_the_state_until_reset`).
        let failed = decoder.step(&mut state, 8, StepMode::Generate);
        assert!(
            matches!(
                failed,
                Err(EngError::BadRequest(_) | EngError::Checkpoint(_))
            ),
            "{failed:?}"
        );
        assert!(state.is_poisoned(), "the failed step poisons the state");
        assert_eq!(
            decoder.session_used(),
            640,
            "only the failed step's permit is released"
        );
        decoder.reset(&mut state).unwrap();
        assert_eq!(decoder.session_used(), 0, "reset releases the old permits");
    }

    #[test]
    fn the_pool_maps_three_sets_of_ten_pages() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let decoder = ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        // Each fixture stack role holds 2 experts of at most 2,048 B, so it fills one page: 3 roles
        // of 3 projections and the gates page make 10 pages per set.
        let page = u64::try_from(crate::kernels::HostPages::page_size()).unwrap();
        assert_eq!(decoder.pool_bytes(), 3 * 10 * page, "pool bytes");
    }

    #[test]
    fn every_step_returns_its_page_sets() {
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::with_pool_sets(MlxKernels::try_new().unwrap(), &ckpt, &flash, 2)
                .unwrap();
        let mut state = decoder.zero_state().unwrap();
        let modes = [
            StepMode::Prompt,
            StepMode::Generate,
            StepMode::Record,
            StepMode::Logits,
            StepMode::Generate,
        ];
        for (token, mode) in (1u32..).zip(modes) {
            let ran = decoder.step(&mut state, token, mode);
            assert!(ran.is_ok(), "{mode:?} token {token}: {:?}", ran.err());
            // One more synchronization frees the residents of every released set.
            let probe = f32_probe(&mut decoder.kernels);
            drop(probe);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            let lent = loop {
                let lent: Vec<usize> = (0..2)
                    .filter(|&index| decoder.slots.pages(index).unwrap().in_use())
                    .collect();
                if lent.is_empty() || std::time::Instant::now() >= deadline {
                    break lent;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            assert!(
                lent.is_empty(),
                "{mode:?} token {token}: sets still lent after the step: {lent:?}"
            );
        }
    }

    /// Imports one F32 value: one synchronization and one retire flush.
    fn f32_probe(kernels: &mut MlxKernels) -> ExecutionTensor {
        let zero = crate::tensor::Tensor::try_from_host(Dtype::F32, vec![1], vec![0; 4]).unwrap();
        kernels.import_tensor(&zero).unwrap()
    }

    /// Synchronizations per token (S2 plan section 4.5): the token-start import, one router
    /// readback per layer, one `evaluate`, and the readbacks (the token; `Logits` reads the logits
    /// instead of the token; `Record` reads the token, the logits, and the rows).
    #[cfg(feature = "native-fault-qualification")]
    fn budget(layers: u64, mode: StepMode) -> u64 {
        let base = layers + 2;
        match mode {
            StepMode::Prompt => base,
            StepMode::Generate => base + 1,
            StepMode::Logits => base + 1,
            StepMode::Record => base + 3,
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn each_mode_uses_its_synchronization_budget() {
        let real = [
            StepMode::Generate,
            StepMode::Prompt,
            StepMode::Record,
            StepMode::Logits,
        ]
        .map(|m| budget(48, m));
        assert_eq!(
            real,
            [51, 50, 53, 51],
            "decision 0041 item 3 and user decision Q2"
        );
        let _serial = crate::native_test_lock();
        let (ckpt, flash) = load();
        let mut decoder =
            ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
        assert_eq!(decoder.weights.layers.len(), 2, "synthetic layer count");
        let mut state = decoder.zero_state().unwrap();
        let runs = [
            (1, StepMode::Generate, 5),
            (2, StepMode::Prompt, 4),
            (3, StepMode::Record, 7),
            (4, StepMode::Generate, 5),
            (5, StepMode::Logits, 5),
        ];
        for (token, mode, want) in runs {
            let before = decoder.kernels.qualification_sync_count();
            decoder.step(&mut state, token, mode).unwrap();
            let used = decoder.kernels.qualification_sync_count() - before;
            assert_eq!(
                used,
                budget(2, mode),
                "{mode:?} token {token}: plan formula"
            );
            assert_eq!(used, want, "{mode:?} token {token}: synthetic count");
        }
    }

    /// Fault tests. A native failure is terminal for the whole process, so each runs in a child.
    #[cfg(feature = "native-fault-qualification")]
    mod faults {
        use super::*;
        use std::{
            env,
            io::Read,
            process::{Command, Stdio},
            thread,
            time::{Duration, Instant},
        };

        const CHILD: &str = "INFERENCE_RESIDENT_STEP_CHILD";
        const DEADLINE: Duration = Duration::from_secs(180);

        fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = pipe.read_to_end(&mut bytes);
                String::from_utf8_lossy(&bytes).into_owned()
            })
        }

        /// In the parent, runs `name` in a child with the `fault` variable (resident_ops.rs:1492-1526).
        fn delegated_to_child(name: &str, fault: (&str, &str)) -> bool {
            if env::var_os(CHILD).is_some() {
                return false;
            }
            let path = module_path!().split_once("::").unwrap().1;
            let mut child = Command::new(env::current_exe().unwrap())
                .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
                .env(CHILD, "1")
                .env(fault.0, fault.1)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let stdout = drain(child.stdout.take().unwrap());
            let stderr = drain(child.stderr.take().unwrap());
            let started = Instant::now();
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break Some(status);
                }
                if started.elapsed() >= DEADLINE {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    break None;
                }
                thread::sleep(Duration::from_millis(20));
            };
            let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
            assert!(
                status.is_some_and(|status| status.success()) && captured.contains("1 passed"),
                "child did not pass {name}: {captured}"
            );
            true
        }

        #[test]
        fn an_injected_evaluation_failure_poisons_the_state() {
            let name = "an_injected_evaluation_failure_poisons_the_state";
            if delegated_to_child(name, ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "evaluate")) {
                return;
            }
            let (ckpt, flash) = load();
            let mut decoder =
                ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
            let mut state = decoder.zero_state().unwrap();
            // The fault arms the first single-array evaluation (guard.cpp:93-115, 641-648): the
            // layer-0 router readback. Layer 0's old state is already dropped at that point.
            let failed = decoder.step(&mut state, 1, StepMode::Generate);
            assert!(
                matches!(
                    failed,
                    Err(EngError::Backend {
                        operation: "evaluate",
                        ..
                    })
                ),
                "{failed:?}"
            );
            assert!(state.is_poisoned(), "a failed evaluation poisons the state");
            assert_eq!(state.pos(), 0, "nothing is published");
            let before = decoder.kernels.qualification_sync_count();
            let again = decoder.step(&mut state, 1, StepMode::Generate);
            assert!(
                matches!(&again, Err(EngError::BadRequest(m)) if m.contains("reset")),
                "{again:?}"
            );
            assert_eq!(
                decoder.kernels.qualification_sync_count(),
                before,
                "no synchronization"
            );
            // The failure is terminal (decision 0039 item 7), so reset cannot import a zero state.
            let reset = decoder.reset(&mut state);
            assert!(matches!(reset, Err(EngError::Backend { .. })), "{reset:?}");
            assert!(
                state.is_poisoned(),
                "a failed reset leaves the state poisoned"
            );
        }

        /// Fails synchronization `n` of the child. Load is 4 and the zero state is 1, so `Generate`
        /// makes 6 to 10: token import 6, the layer 0 router readback 7, the layer 1 router
        /// readback 8, `evaluate` 9, and the token readback 10.
        fn assert_sync_failure_publishes_nothing(name: &str, n: &str) {
            if delegated_to_child(name, ("INFERENCE_MLX_FAIL_NEXT_SYNC", n)) {
                return;
            }
            let (ckpt, flash) = load();
            let mut decoder =
                ResidentDecoder::new(MlxKernels::try_new().unwrap(), &ckpt, &flash).unwrap();
            let mut state = decoder.zero_state().unwrap();
            assert_eq!(
                decoder.kernels.qualification_sync_count(),
                5,
                "syncs before the step"
            );
            let failed = decoder.step(&mut state, 1, StepMode::Generate);
            assert!(
                matches!(failed, Err(EngError::Backend { .. })),
                "{failed:?}"
            );
            assert!(state.is_poisoned(), "the state stays poisoned");
            assert_eq!(state.pos(), 0, "nothing is published");
        }

        #[test]
        fn a_failed_evaluate_synchronization_publishes_nothing() {
            assert_sync_failure_publishes_nothing(
                "a_failed_evaluate_synchronization_publishes_nothing",
                "9",
            );
        }

        #[test]
        fn a_failed_token_readback_publishes_nothing() {
            assert_sync_failure_publishes_nothing(
                "a_failed_token_readback_publishes_nothing",
                "10",
            );
        }
    }
}
