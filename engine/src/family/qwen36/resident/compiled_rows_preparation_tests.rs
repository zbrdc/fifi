//! Explicit preparation lifecycle checks. These require the native tiny-model fixture.

use super::*;
use crate::family::qwen36::tiny::{self, TINY_GQA};

fn decoder(ckpt: &StreamedCheckpoint, activation: Activation) -> ResidentDecoder<'_> {
    ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        ckpt,
        HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap(),
        1024,
        activation,
    )
    .unwrap()
}

#[test]
fn preparation_deduplicates_sizes_and_keeps_the_b1_warmup_separate() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_GQA);
    for activation in [Activation::F32, Activation::Bf16] {
        let mut decoder = decoder(&ckpt, activation);
        decoder.set_compile(true).unwrap();
        let b1 = decoder.compile_warmup.clone();
        assert!(!decoder.compiled_rows_ready(2));
        let before = decoder.kernels.compile_counters();
        decoder.prepare_compiled_rows(&[8, 2, 4, 2]).unwrap();
        let after = decoder.kernels.compile_counters();
        assert_eq!(
            after.delta.calls - before.delta.calls,
            3 * b1.delta.len() as u64
        );
        assert_eq!(after.moe.calls - before.moe.calls, 3 * b1.moe.len() as u64);
        assert_eq!(decoder.compile_warmup, b1);
        assert_eq!(decoder.compiled_rows_warmup.len(), 3);
        for rows in [2, 4, 8] {
            assert!(decoder.compiled_rows_ready(rows));
        }
        for rows in [0, 1, 3, 16] {
            assert!(!decoder.compiled_rows_ready(rows));
        }
        let timings = decoder.compiled_rows_warmup.clone();
        decoder.prepare_compiled_rows(&[4, 8, 2, 4]).unwrap();
        let repeated = decoder.kernels.compile_counters();
        assert_eq!(repeated.delta.calls, after.delta.calls);
        assert_eq!(repeated.moe.calls, after.moe.calls);
        assert_eq!(decoder.compiled_rows_warmup, timings);
    }
}

#[test]
fn preparation_refuses_the_whole_request_before_any_trace_or_readiness_change() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_GQA);
    let mut decoder = decoder(&ckpt, Activation::F32);
    assert!(matches!(
        decoder.prepare_compiled_rows(&[2]),
        Err(EngError::BadRequest(_))
    ));
    decoder.set_compile(true).unwrap();
    decoder.prepare_compiled_rows(&[2]).unwrap();
    let before = decoder.kernels.compile_counters();
    let timings = decoder.compiled_rows_warmup.clone();
    assert!(matches!(
        decoder.prepare_compiled_rows(&[4, 3]),
        Err(EngError::BadRequest(_))
    ));
    assert!(decoder.compiled_rows_ready(2));
    assert!(!decoder.compiled_rows_ready(4));
    let kernel = decoder.weights.shape.delta.conv_kernel;
    decoder.weights.shape.delta.conv_kernel = 2;
    assert!(matches!(
        decoder.prepare_compiled_rows(&[4]),
        Err(EngError::BadRequest(_))
    ));
    decoder.weights.shape.delta.conv_kernel = kernel;
    let after = decoder.kernels.compile_counters();
    assert_eq!(after.delta.calls, before.delta.calls);
    assert_eq!(after.moe.calls, before.moe.calls);
    assert_eq!(decoder.compiled_rows_warmup, timings);
    decoder.set_routing(Routing::Host);
    assert!(!decoder.compiled_rows_ready(2));
    assert!(matches!(
        decoder.prepare_compiled_rows(&[2]),
        Err(EngError::BadRequest(_))
    ));
    decoder.set_routing(Routing::Device);
    assert!(!decoder.compiled_rows_ready(2));
    decoder.prepare_compiled_rows(&[2]).unwrap();
    decoder.set_compile(false).unwrap();
    assert!(!decoder.compiled_rows_ready(2));
    decoder.set_compile(true).unwrap();
    assert!(!decoder.compiled_rows_ready(2));
}

#[test]
fn preparation_tracks_thread_local_clears_and_drops_but_not_counter_resets() {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&TINY_GQA);
    let mut decoder = decoder(&ckpt, Activation::F32);
    decoder.set_compile(true).unwrap();
    decoder.prepare_compiled_rows(&[2]).unwrap();
    decoder.kernels.reset_compile_counters();
    assert!(decoder.compiled_rows_ready(2));
    std::thread::spawn(|| {
        let mut other = MlxKernels::try_new().unwrap();
        other.clear_compile_cache().unwrap();
        // Both this clear and this drop belong to a different thread's cache.
    })
    .join()
    .unwrap();
    assert!(decoder.compiled_rows_ready(2));
    decoder.kernels.clear_compile_cache().unwrap();
    assert!(!decoder.compiled_rows_ready(2));
    decoder.prepare_compiled_rows(&[2]).unwrap();
    assert!(decoder.compiled_rows_ready(2));
    let other = MlxKernels::try_new().unwrap();
    drop(other);
    assert!(!decoder.compiled_rows_ready(2));
    decoder.prepare_compiled_rows(&[2]).unwrap();
    assert!(decoder.compiled_rows_ready(2));
}
