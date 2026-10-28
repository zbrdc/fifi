//! Proposed S1 diagnostic API, not an existing or qualified CUDA executor.
//! Register under cuda_ffn only for native-cuda Linux tests; no serving/default change.

use super::{decode::CudaDecode, real_model, reference::real::gguf};
use crate::{EngError, kernels::cuda::CudaContext};

#[test]
#[ignore = "requires fresh admitted GPU, pinned GGUF and prepared identity; selected-only N=0 pilot"]
fn selected_cold_decode_fetches_only_six_experts_per_layer() {
    let (file, header, identity) = real_model::model();
    let prefixes = real_model::prefixes();
    let ctx = CudaContext::new().expect("required CUDA device");
    let stream = ctx.stream().unwrap();
    let live_before = ctx.live_bytes();
    let pinned_before = ctx.live_owners().pinned;
    let construction = std::time::Instant::now();
    let mut decode = CudaDecode::new_selected(&ctx, &stream, (&file, &header), 8, 0).unwrap();
    println!(
        "SELECTED_CONSTRUCTOR seconds={} includes_kv_base_weights_banks=true excludes_model_open_context_init=true shared_correctness_only=true",
        construction.elapsed().as_secs_f64()
    );
    let initial = decode.fetch_stats();
    assert_eq!(initial.resident_experts, 0);
    assert_eq!(
        ctx.live_bytes() - live_before,
        initial.kv_device_bytes
            + initial.retained_device_bytes
            + initial.bank_device_bytes as usize
    );
    assert_eq!(
        ctx.live_owners().pinned - pinned_before,
        initial.bank_device_bytes as usize
    );
    let mut next = None;
    for (position, &token) in prefixes[0].input_ids.iter().enumerate() {
        let started = std::time::Instant::now();
        next = Some(decode.step(token).unwrap());
        let wall = started.elapsed().as_secs_f64();
        let stats = decode.fetch_stats();
        assert_eq!(stats.cold_experts.len(), 47 * 6);
        assert_eq!(stats.payload_bytes, 47 * 6 * 1_769_472);
        println!(
            "SELECTED_STEP position={position} seconds={wall} cold_experts={} canonical_payload_bytes={} includes_head=true constructor_separate=true shared_correctness_only=true",
            stats.cold_experts.len(),
            stats.payload_bytes
        );
    }
    assert_eq!(next.unwrap() as usize, real_model::FIRST_TOKENS[0]);
    assert_eq!(decode.tokens(), prefixes[0].input_ids.len());
    let stats = decode.fetch_stats();
    assert_eq!(stats.resident_experts, 0);
    assert_eq!(stats.cold_experts.len(), 47 * 6);
    assert_eq!(stats.payload_bytes, 47 * 6 * 1_769_472);
    assert!(stats.kv_device_bytes > 0 && stats.retained_device_bytes > 0);
    assert_eq!(stats.bank_device_bytes, 2 * 1_769_472);
    println!(
        "SELECTED_LEDGER kv={} retained_weights={} banks={} workspace_reservation={} host_stream_reservation={} free_before={} free_after={} cold_payload={} head_and_allocator_not_in_retained=true file_cache_not_credited=true",
        stats.kv_device_bytes,
        stats.retained_device_bytes,
        stats.bank_device_bytes,
        stats.workspace_device_reservation,
        stats.host_stream_reservation,
        stats.device_free_before,
        stats.device_free_after,
        stats.payload_bytes,
    );
    for layer in 2..49 {
        let ids = stats
            .cold_experts
            .iter()
            .filter(|id| id.layer == layer)
            .map(|id| id.expert)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 6, "layer {layer}: selected IDs must be distinct");
        assert!(ids.iter().all(|&id| id < 256));
    }
    drop(decode);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}

#[test]
#[ignore = "requires admitted GPU, pinned GGUF and prepared identity; source-only RED draft"]
fn eager_m1_retains_kv_across_all_eight_frozen_prefixes() {
    let (file, header, identity) = real_model::model();
    let ctx = CudaContext::new().expect("required CUDA device");
    let stream = ctx.stream().unwrap();
    let mut decode = CudaDecode::new(&ctx, &stream, (&file, &header), 8).unwrap();
    for (case, prefix) in real_model::prefixes().iter().enumerate() {
        decode.reset().unwrap();
        assert_eq!(decode.tokens(), 0);
        let mut actual = None;
        for (position, &token) in prefix.input_ids.iter().enumerate() {
            actual = Some(decode.step(token).unwrap());
            assert_eq!(decode.tokens(), position + 1);
        }
        assert_eq!(actual.unwrap() as usize, real_model::FIRST_TOKENS[case]);
    }
    drop(decode);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}

#[test]
#[ignore = "requires admitted GPU, pinned GGUF and prepared identity; source-only RED draft"]
fn invalid_token_does_not_commit_or_break_the_pending_valid_step() {
    let (file, header, identity) = real_model::model();
    let prefixes = real_model::prefixes();
    let input = &prefixes[0].input_ids;
    let ctx = CudaContext::new().expect("required CUDA device");
    let stream = ctx.stream().unwrap();
    let mut decode = CudaDecode::new(&ctx, &stream, (&file, &header), 8).unwrap();
    for &token in &input[..input.len() - 1] {
        decode.step(token).unwrap();
    }
    let committed = decode.tokens();
    assert!(matches!(decode.step(150_144), Err(EngError::BadRequest(_))));
    assert_eq!(decode.tokens(), committed);
    assert_eq!(
        decode.step(*input.last().unwrap()).unwrap() as usize,
        real_model::FIRST_TOKENS[0]
    );
    assert_eq!(decode.tokens(), input.len());
    drop(decode);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}

#[test]
#[ignore = "requires admitted GPU, pinned GGUF and prepared identity; source-only RED draft"]
fn full_context_refuses_without_commit_and_reset_reuses_the_session() {
    let (file, header, identity) = real_model::model();
    let prefixes = real_model::prefixes();
    let input = &prefixes[0].input_ids;
    let ctx = CudaContext::new().expect("required CUDA device");
    let stream = ctx.stream().unwrap();
    let mut decode = CudaDecode::new(&ctx, &stream, (&file, &header), input.len()).unwrap();
    for &token in input {
        decode.step(token).unwrap();
    }
    assert_eq!(decode.tokens(), input.len());
    assert!(matches!(
        decode.step(input[0]),
        Err(EngError::ContextLimit(_))
    ));
    assert_eq!(decode.tokens(), input.len());
    decode.reset().unwrap();
    assert_eq!(decode.tokens(), 0);
    let mut actual = None;
    for &token in input {
        actual = Some(decode.step(token).unwrap());
    }
    assert_eq!(actual.unwrap() as usize, real_model::FIRST_TOKENS[0]);
    assert_eq!(decode.tokens(), input.len());
    drop(decode);
    gguf::verify_prepared_identity(&file, &identity).unwrap();
}
