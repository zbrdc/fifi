//! Public-API smoke tests for the CUDA adapter.
//!
//! Tests that need a device print `SKIPPED` and return when no CUDA device is
//! visible. A skipped run is not evidence that the GPU path works, so GPU gate
//! runs set `INFERENCE_CUDA_REQUIRE_DEVICE=1`, which turns a missing device
//! into a failure. Any value except empty or `0` counts as set.

#![cfg(all(feature = "native-cuda", target_os = "linux"))]

use inference_engine::EngError;
use inference_engine::kernels::cuda::{device_count, vector_add_f32};

#[test]
fn mismatched_lengths_are_bad_requests() {
    let error = vector_add_f32(&[1.0, 2.0], &[1.0]).expect_err("length mismatch must fail");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
}

#[test]
fn empty_inputs_return_empty_output_without_a_device_call() {
    let sum = vector_add_f32(&[], &[]).expect("empty inputs are valid");
    assert!(sum.is_empty());
}

/// Returns whether a device is visible, failing when the gate requires one.
fn device_present(test: &str) -> bool {
    let count = device_count().expect("CUDA device count query");
    println!("CUDA devices visible: {count}");
    if count > 0 {
        return true;
    }
    let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
        .is_ok_and(|value| !value.is_empty() && value != "0");
    assert!(
        !required,
        "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible"
    );
    println!("SKIPPED {test}: no CUDA device");
    false
}

#[test]
fn device_count_query_succeeds() {
    device_present("device_count_query_succeeds");
}

#[test]
fn vector_add_matches_host_sum() {
    if !device_present("vector_add_matches_host_sum") {
        return;
    }
    // 1_000_003 is not a multiple of the block size. 16_781_059 exceeds one
    // launch's maximum of 65535 blocks * 256 threads, so the kernel must loop
    // with a grid stride. Its inputs pass 2^24, where `i as f32` rounds; host
    // and device both add with IEEE-754 round-to-nearest, so results still
    // match exactly. Each length needs about 200 MB of host and device memory,
    // which the Xavier's shared memory holds under the gpu.lock run.
    for len in [1_000_003_usize, 16_781_059] {
        check_vector_add(len);
    }
}

fn check_vector_add(len: usize) {
    let lhs: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
    let rhs: Vec<f32> = (0..len).map(|i| (i % 7) as f32).collect();
    let sum = vector_add_f32(&lhs, &rhs).expect("GPU vector add");
    assert_eq!(sum.len(), len);
    for (index, value) in sum.iter().enumerate() {
        assert_eq!(
            *value,
            lhs[index] + rhs[index],
            "mismatch at index {index} for len {len}"
        );
    }
}
