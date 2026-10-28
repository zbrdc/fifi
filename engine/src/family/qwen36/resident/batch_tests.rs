#[path = "batch_compiled_tests.rs"]
mod compiled;

use super::super::{
    decoder::{ResidentDecoder, StepMode},
    moe::Routing,
    state::{LayerState, ResidentState, kv_growth_bytes},
};
use crate::{
    EngError,
    family::qwen36::{
        host_load::HostWeights,
        settings::{Activation, KvDtype},
        tiny::{self, TINY_KV8},
    },
    kernels::MlxKernels,
    sched::SessionBudget,
    weights::StreamedCheckpoint,
};

fn decoder(ckpt: &StreamedCheckpoint, activation: Activation, kv: KvDtype) -> ResidentDecoder<'_> {
    let host = HostWeights::load(ckpt, &TINY_KV8, &tiny::config(&TINY_KV8)).unwrap();
    let mut decoder = ResidentDecoder::new_with_kv(
        MlxKernels::try_new().unwrap(),
        ckpt,
        host,
        128,
        activation,
        kv,
    )
    .unwrap();
    decoder.routing = Routing::Host;
    assert!(!super::super::compiled::step_requested(decoder.compile));
    decoder
}

#[test]
fn distinct_tokens_and_staggered_positions_match_solo_logits() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    // F32 decoders do not enable BF16 casts, so F32-activation/BF16-cache is not a valid cell.
    for (activation, kv) in [
        (Activation::F32, KvDtype::F32),
        (Activation::F32, KvDtype::Q8),
        (Activation::Bf16, KvDtype::F32),
        (Activation::Bf16, KvDtype::Bf16),
        (Activation::Bf16, KvDtype::Q8),
    ] {
        let mut dec = decoder(&ckpt, activation, kv);
        for rows in [2, 4, 8] {
            let mut solo = Vec::new();
            let mut batch = Vec::new();
            for slot in 0..rows {
                let mut a = dec.zero_state().unwrap();
                let mut b = dec.zero_state().unwrap();
                for pos in 0..slot {
                    let token = ((slot + pos) % 8) as u32;
                    dec.step(&mut a, token, StepMode::Generate).unwrap();
                    dec.step(&mut b, token, StepMode::Generate).unwrap();
                }
                solo.push(a);
                batch.push(b);
            }
            for step in 0..3 {
                let tokens: Vec<_> = (0..rows).map(|slot| ((slot + step) % 8) as u32).collect();
                let expected: Vec<_> = solo
                    .iter_mut()
                    .zip(&tokens)
                    .map(|(state, &token)| dec.step(state, token, StepMode::Logits).unwrap())
                    .collect();
                let mut window: Vec<_> = batch.iter_mut().collect();
                let before = super::BATCH_CALLS.with(std::cell::Cell::get);
                let actual = dec
                    .decode_rows(&mut window, &tokens, StepMode::Logits)
                    .unwrap();
                assert_eq!(super::BATCH_CALLS.with(std::cell::Cell::get), before + 1);
                for (slot, (got, want)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        got.token, want.token,
                        "{activation:?} {kv:?} B={rows} slot={slot}"
                    );
                    let (got, want) = (got.logits.as_ref().unwrap(), want.logits.as_ref().unwrap());
                    assert!(got.iter().chain(want).all(|v| v.is_finite()));
                    let diff = got
                        .iter()
                        .zip(want)
                        .map(|(g, w)| f64::from((g - w).abs()))
                        .fold(0.0_f64, f64::max);
                    let scale = want
                        .iter()
                        .map(|v| f64::from(v.abs()))
                        .fold(0.0_f64, f64::max);
                    let relative = diff / scale;
                    let cosine = super::super::parity::cosine(got, want);
                    // Use the existing within-mode chunk/solo logit bounds, not bit identity.
                    assert!(
                        super::super::within_mode::within_mode_logits_passes(
                            activation, cosine, relative
                        ),
                        "logits: {activation:?} {kv:?} B={rows} slot={slot} step={step}: cosine={cosine}, relative={relative}"
                    );
                    assert_eq!(batch[slot].pos(), solo[slot].pos());
                }
            }
        }
    }
}

#[test]
fn later_permit_refusal_preserves_all_states_and_budget() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    let mut dec = decoder(&ckpt, Activation::Bf16, KvDtype::Q8);
    let mut a = dec.zero_state().unwrap();
    let mut b = dec.zero_state().unwrap();
    let growth = kv_growth_bytes(&dec.weights).unwrap();
    // Exactly one provisional permit fits; the second must fail and release the first.
    dec.budget = SessionBudget::new(growth);
    let used = dec.budget.used();
    let error = dec
        .decode_rows(&mut [&mut a, &mut b], &[1, 2], StepMode::Generate)
        .unwrap_err();
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    assert_eq!(dec.budget.used(), used);
    assert_eq!((a.pos(), b.pos()), (0, 0));
    a.check_step(&dec.weights, dec.context_limit).unwrap();
    b.check_step(&dec.weights, dec.context_limit).unwrap();
    dec.budget = SessionBudget::new(u64::MAX);
    dec.step(&mut a, 1, StepMode::Generate).unwrap();
    dec.step(&mut b, 2, StepMode::Generate).unwrap();
}

#[test]
fn fallback_rejects_shape_errors_and_stops_at_first_failure() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    let mut dec = decoder(&ckpt, Activation::F32, KvDtype::F32);
    dec.routing = Routing::Device;
    let mut a = dec.zero_state().unwrap();
    let mut b = dec.zero_state().unwrap();
    assert!(
        dec.decode_rows(&mut [&mut a, &mut b], &[1], StepMode::Generate)
            .is_err()
    );
    assert!(
        dec.decode_rows(&mut [&mut a], &[1, 2], StepMode::Generate)
            .is_err()
    );
    let mut empty: [&mut ResidentState; 0] = [];
    assert!(
        dec.decode_rows(&mut empty, &[], StepMode::Generate)
            .is_err()
    );
    assert_eq!((a.pos(), b.pos()), (0, 0));
    a.poison_for_test();
    assert!(
        dec.decode_rows(&mut [&mut a, &mut b], &[1, 2], StepMode::Generate)
            .is_err()
    );
    assert_eq!(b.pos(), 0);
    dec.step(&mut b, 2, StepMode::Generate).unwrap();
}

#[test]
fn independent_dispatch_guards_use_the_solo_fallback() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    for (routing, compile, mode) in [
        (Routing::Device, false, StepMode::Logits),
        (Routing::Host, true, StepMode::Logits),
        (Routing::Device, true, StepMode::Logits),
        (Routing::Host, false, StepMode::Prompt),
        (Routing::Host, false, StepMode::Record),
    ] {
        let mut dec = decoder(&ckpt, Activation::Bf16, KvDtype::Q8);
        dec.routing = routing;
        dec.set_compile(compile).unwrap();
        let mut a = dec.zero_state().unwrap();
        let mut b = dec.zero_state().unwrap();
        let mut reference_a = dec.zero_state().unwrap();
        let mut reference_b = dec.zero_state().unwrap();
        let want_a = dec.step(&mut reference_a, 1, mode).unwrap();
        let want_b = dec.step(&mut reference_b, 2, mode).unwrap();
        let before = super::BATCH_CALLS.with(std::cell::Cell::get);
        let got = dec
            .decode_rows(&mut [&mut a, &mut b], &[1, 2], mode)
            .unwrap();
        assert_eq!(super::BATCH_CALLS.with(std::cell::Cell::get), before);
        for (got, want) in got.iter().zip([want_a, want_b]) {
            assert_eq!(got.token, want.token);
            assert_eq!(got.logits, want.logits);
            assert_eq!(got.layer_out.is_some(), want.layer_out.is_some());
            assert_eq!(got.routes.is_some(), want.routes.is_some());
        }
        assert_eq!((a.pos(), b.pos()), (1, 1));
    }
}

#[test]
fn invalid_token_preserves_all_states_for_batch_and_fallback() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    for routing in [Routing::Host, Routing::Device] {
        let mut dec = decoder(&ckpt, Activation::F32, KvDtype::F32);
        dec.routing = routing;
        let mut a = dec.zero_state().unwrap();
        let mut b = dec.zero_state().unwrap();
        let invalid = dec.weights.shape.vocab as u32;
        let used = dec.budget.used();
        let error = dec
            .decode_rows(&mut [&mut a, &mut b], &[1, invalid], StepMode::Generate)
            .unwrap_err();
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        assert_eq!((a.pos(), b.pos(), dec.budget.used()), (0, 0, used));
        a.check_step(&dec.weights, dec.context_limit).unwrap();
        b.check_step(&dec.weights, dec.context_limit).unwrap();
        dec.decode_rows(&mut [&mut a, &mut b], &[1, 2], StepMode::Generate)
            .unwrap();
    }
}

#[test]
fn fallback_late_context_refusal_preserves_earlier_slot() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    let mut dec = decoder(&ckpt, Activation::F32, KvDtype::F32);
    dec.routing = Routing::Device;
    let mut a = dec.zero_state().unwrap();
    let mut b = dec.zero_state().unwrap();
    dec.step(&mut b, 2, StepMode::Generate).unwrap();
    dec.context_limit = 1;
    let used = dec.budget.used();
    let error = dec
        .decode_rows(&mut [&mut a, &mut b], &[1, 2], StepMode::Generate)
        .unwrap_err();
    assert!(matches!(error, EngError::ContextLimit(_)), "{error:?}");
    assert_eq!((a.pos(), b.pos(), dec.budget.used()), (0, 1, used));
    assert!(!a.is_poisoned() && !b.is_poisoned());
}

#[test]
fn fallback_failure_after_first_commit_poisons_the_entire_window() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    let mut dec = decoder(&ckpt, Activation::F32, KvDtype::F32);
    dec.routing = Routing::Device;
    let mut a = dec.zero_state().unwrap();
    let mut b = dec.zero_state().unwrap();
    // A valid-shape NaN ring passes structural checks; device router health fails later.
    let mut layers = b.begin();
    let ring = layers
        .iter_mut()
        .find_map(|layer| match layer {
            LayerState::Delta { ring, .. } => Some(ring),
            _ => None,
        })
        .unwrap();
    *ring = dec
        .kernels
        .binary(crate::kernels::BinaryOp::Div, ring, ring)
        .unwrap();
    b.commit_rows(0, layers, dec.budget.acquire_growth(0).unwrap())
        .unwrap();
    b.check_step(&dec.weights, dec.context_limit).unwrap();
    let error = dec
        .decode_rows(&mut [&mut a, &mut b], &[1, 2], StepMode::Generate)
        .unwrap_err();
    assert!(matches!(error, EngError::Backend { .. }), "{error:?}");
    assert_eq!((a.pos(), b.pos()), (1, 0));
    assert!(a.is_poisoned() && b.is_poisoned());
    assert!(dec.step(&mut a, 1, StepMode::Generate).is_err());
    assert!(dec.step(&mut b, 2, StepMode::Generate).is_err());
}

#[test]
fn malformed_delta_shapes_are_refused_before_begin_for_both_routes() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    for routing in [Routing::Host, Routing::Device] {
        for flatten_rec in [true, false] {
            let mut dec = decoder(&ckpt, Activation::F32, KvDtype::F32);
            dec.routing = routing;
            let mut a = dec.zero_state().unwrap();
            let mut b = dec.zero_state().unwrap();
            let mut layers = b.begin();
            let tensor = layers
                .iter_mut()
                .find_map(|layer| match layer {
                    LayerState::Delta { rec, ring } => Some(if flatten_rec { rec } else { ring }),
                    _ => None,
                })
                .unwrap();
            let count = tensor.shape().iter().copied().product();
            *tensor = dec.kernels.reshape(tensor, &[count]).unwrap();
            b.commit_rows(0, layers, dec.budget.acquire_growth(0).unwrap())
                .unwrap();
            let used = dec.budget.used();
            let error = dec
                .decode_rows(&mut [&mut a, &mut b], &[1, 2], StepMode::Generate)
                .unwrap_err();
            assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
            assert_eq!((a.pos(), b.pos(), dec.budget.used()), (0, 0, used));
            assert!(!a.is_poisoned() && !b.is_poisoned());
            dec.step(&mut a, 1, StepMode::Generate).unwrap();
        }
    }
}
