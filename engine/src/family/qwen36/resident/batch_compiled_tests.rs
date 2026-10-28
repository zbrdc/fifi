use super::super::super::{decoder::StepMode, moe::Routing, parity, weights::RMixer, within_mode};
use crate::family::qwen36::{
    settings::{Activation, KvDtype},
    tiny::{self, TINY_KV8},
};

#[test]
fn compiled_rows_use_warmed_blocks_and_match_independent_solo_states() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_KV8);
    for activation in [Activation::F32, Activation::Bf16] {
        let mut decoder = super::decoder(&ckpt, activation, KvDtype::F32);
        decoder.routing = Routing::Device;
        decoder.set_compile(true).unwrap();
        decoder.prepare_compiled_rows(&[2, 4, 8]).unwrap();
        let delta_layers = decoder
            .weights
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, RMixer::Delta(_)))
            .count() as u64;
        let moe_layers = decoder.weights.layers.len() as u64;
        assert!(delta_layers > 0);
        for batch in [2usize, 4, 8] {
            let (mut solo, mut rows) = (Vec::new(), Vec::new());
            for slot in 0..batch {
                let mut a = decoder.zero_state().unwrap();
                let mut b = decoder.zero_state().unwrap();
                for position in 0..slot {
                    let token = ((slot + position) % 8) as u32;
                    decoder.step(&mut a, token, StepMode::Generate).unwrap();
                    decoder.step(&mut b, token, StepMode::Generate).unwrap();
                }
                solo.push(a);
                rows.push(b);
            }
            for step in 0..3 {
                let tokens: Vec<_> = (0..batch).map(|slot| ((slot + step) % 8) as u32).collect();
                let expected: Vec<_> = solo
                    .iter_mut()
                    .zip(&tokens)
                    .map(|(state, &token)| decoder.step(state, token, StepMode::Logits).unwrap())
                    .collect();
                let before = decoder.kernels.compile_counters();
                let dispatched = super::super::BATCH_CALLS.with(std::cell::Cell::get);
                let mut window: Vec<_> = rows.iter_mut().collect();
                let actual = decoder
                    .decode_rows(&mut window, &tokens, StepMode::Logits)
                    .unwrap();
                assert_eq!(
                    super::super::BATCH_CALLS.with(std::cell::Cell::get),
                    dispatched + 1,
                    "compiled B={batch} silently took the solo fallback"
                );
                let after = decoder.kernels.compile_counters();
                assert_eq!(after.delta.calls - before.delta.calls, delta_layers);
                assert_eq!(after.moe.calls - before.moe.calls, moe_layers);
                assert_eq!(
                    after.delta.traces, before.delta.traces,
                    "decode traced DeltaNet"
                );
                assert_eq!(after.moe.traces, before.moe.traces, "decode traced MoE");
                for (slot, (got, want)) in actual.iter().zip(&expected).enumerate() {
                    let (got, want) = (got.logits.as_ref().unwrap(), want.logits.as_ref().unwrap());
                    assert_eq!(got.len(), want.len());
                    assert!(got.iter().chain(want).all(|value| value.is_finite()));
                    let difference = got
                        .iter()
                        .zip(want)
                        .map(|(a, b)| f64::from((*a - *b).abs()))
                        .fold(0.0_f64, f64::max);
                    let scale = want
                        .iter()
                        .map(|value| f64::from(value.abs()))
                        .fold(f64::MIN_POSITIVE, f64::max);
                    let cosine = parity::cosine(got, want);
                    assert!(
                        within_mode::within_mode_logits_passes(
                            activation,
                            cosine,
                            difference / scale
                        ),
                        "{activation:?} B={batch} slot={slot} step={step}: cosine={cosine} relative={}",
                        difference / scale
                    );
                    assert_eq!(rows[slot].pos(), solo[slot].pos());
                }
            }
        }
    }
}
