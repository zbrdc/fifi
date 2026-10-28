use super::*;
use crate::family::flint::{
    config::FULL_LAYERS,
    cuda_ffn::device::runtime::{test_commit_ends, test_counts, test_head_ranges, test_head_reads},
    kv_plan::KvPlan,
};
use crate::kernels::cuda::DeviceBuffer;
use sha2::{Digest, Sha256};
use std::{ops::Range, path::PathBuf};

#[path = "cuda_executor_tests/prefill_controls.rs"]
mod prefill_controls;

const CONTEXT: usize = 576;
const ROW_BYTES: usize = 4 * 128 * 2;

fn fixture_tokens(length: usize) -> Vec<u32> {
    let bytes = include_bytes!("cuda_ffn/fixtures/first-token-prefixes.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "2b6f2c43d5a66a85c5feb53d541e2ceda086e2563a8e9ab00e4a64fbada1e814"
    );
    let cases: Vec<serde_json::Value> = serde_json::from_slice(bytes).unwrap();
    let tokens: Vec<u32> = cases
        .iter()
        .flat_map(|case| serde_json::from_value::<Vec<u32>>(case["input_ids"].clone()).unwrap())
        .collect();
    assert_eq!(tokens.len(), 31);
    // Repeat the frozen, nonuniform real prefixes to reach the ring; do not invent vocabulary IDs.
    tokens.into_iter().cycle().take(length).collect()
}

fn real_executor() -> (FlintExecutor, FlintSession, FlintSession) {
    real_executor_at_context(CONTEXT, false)
}

fn pinned_late_health_arm() -> bool {
    match std::env::var("FIFI_FLINT_LATE_HEALTH").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        other => panic!("GPU runner must pin FIFI_FLINT_LATE_HEALTH=0/1, got {other:?}"),
    }
}

fn fixture_work_reserve(
    serial: &KvPlan,
    batch: &KvPlan,
    cache_bytes: u64,
    production_b8: bool,
) -> u64 {
    let scratch = State::scratch_bytes(serial)
        .unwrap()
        .checked_add(State::scratch_bytes(batch).unwrap())
        .unwrap();
    if production_b8 {
        work_reserve(batch, 8, cache_bytes)
            .unwrap()
            .checked_add(State::scratch_bytes(serial).unwrap())
            .unwrap()
    } else {
        (512_u64 << 20)
            .checked_add(scratch)
            .and_then(|bytes| bytes.checked_add(cache_bytes))
            .unwrap()
    }
}

fn real_executor_at_context(
    context: usize,
    production_b8: bool,
) -> (FlintExecutor, FlintSession, FlintSession) {
    let late_health = pinned_late_health_arm();
    let root = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in real EXL3 pack"));
    let options: FlintOptions = serde_json::from_value(serde_json::json!({
        "context_limit": context, "kv_profile": "f16", "prefill_chunk_size": 8
    }))
    .expect("production B8 options; no serial fallback");
    let arm = std::env::var("FIFI_FLINT_CUDA_ALLOC_CACHE").unwrap_or_else(|error| {
        assert!(matches!(error, std::env::VarError::NotPresent));
        "1".into()
    });
    assert!(
        matches!(arm.as_str(), "0" | "1"),
        "explicit cache0/cache1 test arm"
    );
    let (executor, initial) = FlintExecutor::open(&root, &options, vec![2], VOCAB).unwrap();
    assert_eq!(executor.ctx.alloc_cache_enabled(), arm == "1");
    assert_eq!(executor.prefill_chunk_size, 8);
    drop(initial);
    let serial = KvPlan::new(context, 1, true).unwrap();
    let batch = KvPlan::new(context, 8, true).unwrap();
    assert_eq!(serial.cache_rows(2).unwrap(), 513);
    assert_eq!(batch.cache_rows(2).unwrap(), 520);
    // Open has charged the model/workspace; admit both additional test states afterward.
    let work = fixture_work_reserve(
        &serial,
        &batch,
        CudaContext::alloc_cache_reserve(executor.ctx.alloc_cache_enabled()),
        production_b8,
    );
    plan::admit(
        &plan::Demand {
            tensors: 0,
            kv_cache: (serial.bytes(2 * ROW_BYTES).unwrap() + batch.bytes(2 * ROW_BYTES).unwrap())
                as u64,
            work,
            headroom: 512 << 20,
        },
        executor.ctx.memory_info().unwrap().free as u64,
    )
    .unwrap();
    let session = |plan| FlintSession {
        state: State::new(&executor.ctx, plan).unwrap(),
        core: DecodeSession::default(),
    };
    let serial = session(serial);
    let batch = session(batch);
    assert_eq!(serial.state.test_late_health_enabled(), late_health);
    assert_eq!(batch.state.test_late_health_enabled(), late_health);
    (executor, serial, batch)
}

#[test]
fn cache_switch_defaults_on_and_supports_explicit_rollback() {
    if std::env::var_os(ALLOC_CACHE_VAR).is_none() {
        assert!(
            alloc_cache_selected(),
            "unset Flint selector must enable reuse"
        );
    }
    assert!(crate::tokenizer::env_allows_value(None));
    for value in ["1", "true", "on", "  TrUe\n"] {
        assert!(crate::tokenizer::env_allows_value(Some(value)));
    }
    for value in ["0", "off", "", "2"] {
        assert!(!crate::tokenizer::env_allows_value(Some(value)));
    }
}

#[test]
fn cache_fixture_reserve_covers_b8_and_both_states() {
    for context in [576, 2048, 262_144] {
        let serial = KvPlan::new(context, 1, true).unwrap();
        let batch = KvPlan::new(context, 8, true).unwrap();
        let scratch =
            State::scratch_bytes(&serial).unwrap() + State::scratch_bytes(&batch).unwrap();
        for (production_b8, workspace_mib) in [(false, 512_u64), (true, 624)] {
            let expected = (workspace_mib << 20) + scratch;
            let off = fixture_work_reserve(&serial, &batch, 0, production_b8);
            let on = fixture_work_reserve(
                &serial,
                &batch,
                CudaContext::alloc_cache_reserve(true),
                production_b8,
            );
            assert_eq!(
                off, expected,
                "workspace and both actual State scratch charges"
            );
            assert_eq!(
                on,
                expected + CudaContext::alloc_cache_reserve(true),
                "one cache reserve per lease"
            );
        }
    }
}

#[test]
#[ignore = "requires admitted GPU, real EXL3 weights and allocation reuse enabled (unset or =1)"]
fn control_cache_session_poison_resets_without_backend_terminal() {
    let (executor, mut session, _spare) = real_executor_at_context(CONTEXT, true);
    assert!(
        executor.ctx.alloc_cache_enabled(),
        "allocation reuse must be enabled"
    );
    let token = fixture_tokens(1)[0];
    let decoder = executor.decoder().unwrap();
    let pristine: Vec<u32> = decoder
        .step(&mut session.state, token, |_, logits| {
            Ok(logits.iter().map(|value| value.to_bits()).collect())
        })
        .unwrap();
    Step {
        executor: &executor,
        state: &mut session.state,
    }
    .reset()
    .unwrap();
    let failure = EngError::ModelOutput("cache control post-KV acceptance refusal".into());
    let refused: Result<()> = decoder.step(&mut session.state, token, |_, _| Err(failure.clone()));
    assert_eq!(refused, Err(failure));
    assert_eq!(session.state.fed(), 0);
    assert!(session.state.test_poisoned());
    assert!(
        session
            .state
            .test_kv()
            .iter()
            .all(|kv| kv.cursor.test_poisoned())
    );
    executor
        .ctx
        .ensure_usable()
        .expect("session failure is not backend terminal");
    Step {
        executor: &executor,
        state: &mut session.state,
    }
    .reset()
    .unwrap();
    assert!(!session.state.test_poisoned());
    assert!(
        session
            .state
            .test_kv()
            .iter()
            .all(|kv| !kv.cursor.test_poisoned())
    );
    let replay: Vec<u32> = decoder
        .step(&mut session.state, token, |_, logits| {
            Ok(logits.iter().map(|value| value.to_bits()).collect())
        })
        .unwrap();
    assert_eq!(
        replay, pristine,
        "healthy cache reset replays the full head bitwise"
    );
    executor.ctx.ensure_usable().unwrap();
}

#[cfg(feature = "cuda-fault-qualification")]
#[test]
#[ignore = "irreversible native fault; requires real EXL3 weights and allocation reuse enabled (unset or =1)"]
fn control_cache_reset_propagates_prior_native_sync_backend() {
    let (executor, mut session, _spare) = real_executor_at_context(CONTEXT, true);
    assert!(
        executor.ctx.alloc_cache_enabled(),
        "allocation reuse must be enabled"
    );
    executor
        .decoder()
        .unwrap()
        .step(&mut session.state, fixture_tokens(1)[0], |_, _| Ok(()))
        .unwrap();
    let fed = session.state.fed();
    let fault = crate::kernels::cuda::test_inject_unmapped_fault();
    let reset = Step {
        executor: &executor,
        state: &mut session.state,
    }
    .reset();
    let backend_terminal = executor.ctx.ensure_usable().is_err();
    assert_eq!(fault, Err(700));
    assert!(
        matches!(&reset, Err(EngError::Backend { message, .. }) if message.contains("status 700")),
        "existing State reset propagates the prior native sync failure"
    );
    assert!(backend_terminal);
    assert_eq!(
        session.state.fed(),
        fed,
        "failed State sync did not reset the session"
    );
}

#[cfg(feature = "cuda-fault-qualification")]
#[test]
#[ignore = "irreversible native fault; requires real EXL3 weights and allocation reuse enabled (unset or =1)"]
fn reset_propagates_cache_sync_backend_after_successful_state_reset() {
    let (executor, mut session, _spare) = real_executor_at_context(CONTEXT, true);
    assert!(
        executor.ctx.alloc_cache_enabled(),
        "allocation reuse must be enabled"
    );
    executor
        .decoder()
        .unwrap()
        .step(&mut session.state, fixture_tokens(1)[0], |_, _| Ok(()))
        .unwrap();
    assert_eq!(session.state.fed(), 1);
    executor.ctx.test_cache_fail_next_sync();
    let reset = Step {
        executor: &executor,
        state: &mut session.state,
    }
    .reset();
    let backend_terminal = executor.ctx.ensure_usable().is_err();
    assert_eq!(
        session.state.fed(),
        0,
        "State reset completed before the cache fault"
    );
    assert!(
        matches!(&reset, Err(EngError::Backend { operation: "cuda synchronize", message, .. }) if message.contains("status 700")),
        "Step reset must propagate cache sync Backend after State reset"
    );
    assert!(
        executor.ctx.test_cache_sync_fault_observed(),
        "reset fault must fire at the cache's own sync, not State reset"
    );
    assert!(
        backend_terminal,
        "cache sync failure is process-wide terminal"
    );
}

#[test]
#[ignore = "whole-card real EXL3 gate; run the same ELF with explicit cache0 and cache1 arms"]
fn control_populated_long_context_real_executor_cache_identity() {
    let requested =
        std::env::var("FIFI_FLINT_CUDA_ALLOC_CACHE").expect("explicit cache0/cache1 arm");
    assert!(matches!(requested.as_str(), "0" | "1"));
    let (mut executor, _serial, mut session) = real_executor_at_context(2080, true);
    assert_eq!(executor.context_limit, 2080);
    assert_eq!(
        executor.prefill_chunk_size, 8,
        "the long gate must exercise B8"
    );
    assert_eq!(executor.ctx.alloc_cache_enabled(), requested == "1");
    let prompt = fixture_tokens(2048);
    assert_eq!(
        (
            (prompt.len() - 1) / executor.prefill_chunk_size,
            (prompt.len() - 1) % executor.prefill_chunk_size,
        ),
        (255, 7),
        "the poison set requires 255 B8 chunks and one seven-row tail before single-row decode"
    );
    let mut traces = Vec::new();
    let mut reset_us = 0;
    // Scale uploads use 8/7/1 rows; the 16-byte four-row class is never replayed.
    let poisoned_sizes = [
        4, 24, 28, 1024, 2048, 4096, 8192, 16384, 24576, 49152, 73728, 600576,
    ];
    for run in 0..2 {
        if run == 1 {
            let start = std::time::Instant::now();
            executor.reset(&mut session).unwrap();
            reset_us = start.elapsed().as_micros();
            assert_eq!(session.state.fed(), 0);
            assert!(session.core.history().is_empty());
            session
                .state
                .test_dirty_hot_scratch(&executor.stream)
                .unwrap();
            if executor.ctx.alloc_cache_enabled() {
                // Watch these exact poisoned allocations, not unrelated same-size hits.
                for bytes in poisoned_sizes {
                    let poison = f32::NAN.to_bits().to_le_bytes().repeat(bytes / 4);
                    let buffer = executor.ctx.upload(&poison).unwrap();
                    executor.ctx.test_watch_poisoned_cache_block(&buffer);
                    drop(buffer);
                }
            }
        }
        executor.prefill(&mut session, &prompt).unwrap();
        assert_eq!(
            session.state.fed(),
            2047,
            "the pending prompt token is not fed yet"
        );
        let mut pending = session.core.checked_pending().unwrap();
        let mut sampler = Sampler::greedy();
        let mut trace = Vec::new();
        for _ in 0..17 {
            let fingerprint =
                decode_fingerprint(&executor, &mut session.state, pending, &mut sampler);
            pending = fingerprint.0;
            trace.push(fingerprint);
        }
        assert_eq!(
            session.state.fed(),
            2064,
            "the context was populated, not just admitted"
        );
        traces.push(trace);
    }
    assert_eq!(
        traces[0], traces[1],
        "reset replay changes full F32-head fingerprints"
    );
    let poisoned_return_replay_hits = executor.ctx.test_poisoned_cache_hits();
    println!(
        "CACHE_GOLDEN_V1 {}",
        serde_json::json!({
            "cache_on": executor.ctx.alloc_cache_enabled(),
            "context_limit": 2080,
            "prompt_tokens": prompt.len(),
            "fed": session.state.fed(),
            "chunk_rows": executor.prefill_chunk_size,
            "trace": traces[0],
            "reset_us": reset_us,
            "poisoned_return_replay_hits": poisoned_return_replay_hits,
        })
    );
    if executor.ctx.alloc_cache_enabled() {
        assert_eq!(
            poisoned_return_replay_hits,
            poisoned_sizes.len(),
            "every exact poisoned block must be checked out during the populated replay"
        );
    }
}

fn count_delta(before: [usize; 4]) -> [usize; 4] {
    let after = test_counts();
    std::array::from_fn(|index| after[index].checked_sub(before[index]).unwrap())
}

fn serial_prefill(executor: &mut FlintExecutor, session: &mut FlintSession, tokens: &[u32]) {
    for token in tokens {
        executor
            .prefill(session, std::slice::from_ref(token))
            .unwrap();
    }
}

fn canonical_plane(stream: &DeviceStream, plane: &DeviceBuffer, rows: Range<usize>) -> Vec<u8> {
    let cache_rows = plane.bytes() / ROW_BYTES;
    let count = rows.end - rows.start;
    assert!(count <= cache_rows);
    let mut bytes = vec![0; count * ROW_BYTES];
    if count == 0 {
        return bytes;
    }
    let slot = rows.start % cache_rows;
    let first = count.min(cache_rows - slot) * ROW_BYTES;
    plane
        .read_at(stream, slot * ROW_BYTES, &mut bytes[..first])
        .unwrap();
    if first != bytes.len() {
        plane.read_at(stream, 0, &mut bytes[first..]).unwrap();
    }
    bytes
}

fn assert_same_kv(executor: &FlintExecutor, serial: &FlintSession, batch: &FlintSession) {
    let fed = serial.state.fed();
    assert_eq!(batch.state.fed(), fed);
    assert_eq!(serial.state.test_kv().len(), 49);
    assert_eq!(batch.state.test_kv().len(), 49);
    for (layer, (left, right)) in serial
        .state
        .test_kv()
        .iter()
        .zip(batch.state.test_kv())
        .enumerate()
    {
        assert_eq!(left.cursor.committed_prefix(), fed, "serial layer {layer}");
        assert_eq!(right.cursor.committed_prefix(), fed, "B8 layer {layer}");
        assert!(!left.cursor.test_poisoned());
        assert!(!right.cursor.test_poisoned());
        let start = if FULL_LAYERS.contains(&(layer as u64)) {
            0
        } else {
            fed.saturating_sub(513)
        };
        for (name, left, right) in [
            ("K", &left.keys, &right.keys),
            ("V", &left.values, &right.values),
        ] {
            let expected = canonical_plane(&executor.stream, left, start..fed);
            let actual = canonical_plane(&executor.stream, right, start..fed);
            let mismatch = expected.iter().zip(&actual).position(|(a, b)| a != b);
            assert_eq!(expected.len(), actual.len());
            assert!(
                mismatch.is_none(),
                "layer {layer} {name}, fed={fed}, logical_start={start}, differing byte={mismatch:?}"
            );
        }
    }
}

fn greedy_trajectory(executor: &FlintExecutor, session: &mut FlintSession) -> (Vec<u32>, Vec<u32>) {
    let pending = session.core.checked_pending().unwrap();
    let before = test_head_reads();
    let mut sampler = Sampler::greedy();
    let (head_bits, first) = executor
        .decoder()
        .unwrap()
        .step(&mut session.state, pending, |head, logits| {
            assert_eq!(logits.len(), VOCAB);
            assert!(logits.iter().all(|value| value.is_finite()));
            let first = executor.sampling.sample(
                &executor.ctx,
                &executor.stream,
                head,
                logits,
                &mut sampler,
                executor.vocab,
            )?;
            Ok((logits.iter().map(|value| value.to_bits()).collect(), first))
        })
        .unwrap();
    assert_eq!(
        test_head_reads() - before,
        1,
        "first sampling must read the held prompt head exactly once"
    );
    let mut ids = vec![first];
    let mut step = Step {
        executor,
        state: &mut session.state,
    };
    for _ in 0..16 {
        ids.push(
            step.next_token(*ids.last().unwrap(), &mut sampler, executor.vocab)
                .unwrap(),
        );
    }
    (head_bits, ids)
}

fn assert_same_trajectory(
    executor: &FlintExecutor,
    serial: &mut FlintSession,
    batch: &mut FlintSession,
) {
    let prompt_length = serial.core.history().len();
    let expected = greedy_trajectory(executor, serial);
    let actual = greedy_trajectory(executor, batch);
    let mismatch = expected.0.iter().zip(&actual.0).position(|(a, b)| a != b);
    assert_eq!(expected.0.len(), actual.0.len());
    assert!(
        mismatch.is_none(),
        "last-prompt-row logits differ at {mismatch:?}"
    );
    assert_eq!(actual.1.len(), 17);
    assert_eq!(
        actual.1, expected.1,
        "first sampled ID and sixteen continuations"
    );
    assert_eq!(serial.state.fed(), prompt_length + 16);
    assert_eq!(batch.state.fed(), prompt_length + 16);
    assert_same_kv(executor, serial, batch);
}

fn decode_fingerprint(
    executor: &FlintExecutor,
    state: &mut State,
    pending: u32,
    sampler: &mut Sampler,
) -> (u32, String) {
    executor
        .decoder()
        .unwrap()
        .step(state, pending, |head, logits| {
            assert_eq!(logits.len(), VOCAB);
            assert!(logits.iter().all(|value| value.is_finite()));
            let id = executor.sampling.sample(
                &executor.ctx,
                &executor.stream,
                head,
                logits,
                sampler,
                executor.vocab,
            )?;
            let mut hash = Sha256::new();
            for value in logits {
                hash.update(value.to_bits().to_le_bytes());
            }
            Ok((id, format!("{:x}", hash.finalize())))
        })
        .unwrap()
}

#[test]
#[ignore = "requires admitted CUDA and full EXL3 pack; allocation and independent bitwise golden gate"]
fn decode_reuses_large_scratch_on_real_dispatch() {
    assert!(
        !pinned_late_health_arm(),
        "legacy allocation golden requires FIFI_FLINT_LATE_HEALTH=0"
    );
    let (mut executor, mut serial, _batch) = real_executor();
    let observer = executor.ctx.buffer_requests();
    let sentinel = executor.ctx.allocate(31).unwrap();
    assert_eq!(
        executor
            .ctx
            .buffer_requests()
            .allocations
            .get(&31)
            .copied()
            .unwrap_or(0),
        observer
            .allocations
            .get(&31)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .unwrap(),
        "allocation observer must be live before the measured dispatch"
    );
    drop(sentinel);
    let prompt = fixture_tokens(22);
    serial_prefill(&mut executor, &mut serial, &prompt);
    let pending = serial.core.checked_pending().unwrap();
    let mut sampler = Sampler::greedy();
    let seed = Step {
        executor: &executor,
        state: &mut serial.state,
    }
    .next_token(pending, &mut sampler, executor.vocab)
    .unwrap();
    serial
        .state
        .test_dirty_hot_scratch(&executor.stream)
        .unwrap();
    let before = executor.ctx.buffer_requests();
    let before_alloc_requests = executor.ctx.successful_allocations();
    let model_before = test_counts();
    let (next, head_hash) = decode_fingerprint(&executor, &mut serial.state, seed, &mut sampler);
    let after = executor.ctx.buffer_requests();
    let alloc_requests = executor
        .ctx
        .successful_allocations()
        .checked_sub(before_alloc_requests)
        .unwrap();
    let public_requests = after
        .allocations
        .values()
        .sum::<usize>()
        .checked_sub(before.allocations.values().sum::<usize>())
        .unwrap();
    let private_requests = alloc_requests.checked_sub(public_requests).unwrap();
    println!(
        "successful ctx.alloc requests={alloc_requests}, public={public_requests}, private={private_requests} (excludes native upload/workspace allocations; not driver call counts)"
    );
    assert_eq!(serial.state.fed(), 23);
    assert_eq!(count_delta(model_before), [49, 98, 49, 49]);
    let sizes = [1_048_592, 5_457_920, 2_097_152];
    let hot = sizes.map(|bytes| {
        after
            .allocations
            .get(&bytes)
            .copied()
            .unwrap_or(0)
            .checked_sub(before.allocations.get(&bytes).copied().unwrap_or(0))
            .unwrap()
    });
    let hot_calls: usize = hot.iter().sum();
    let hot_bytes: usize = sizes
        .into_iter()
        .zip(hot)
        .map(|(bytes, calls)| bytes * calls)
        .sum();
    let sum_bytes = |requests: &std::collections::BTreeMap<usize, usize>| {
        requests
            .iter()
            .map(|(bytes, calls)| bytes * calls)
            .sum::<usize>()
    };
    println!(
        "successful allocate requests: hot [norm,CPF,router-publication]={hot:?}, count={hot_calls}, bytes={hot_bytes}; source baseline=[99,49,47]/195/469814832"
    );
    let h2d_calls =
        after.ranged_h2d.values().sum::<usize>() - before.ranged_h2d.values().sum::<usize>();
    let h2d_bytes = sum_bytes(&after.ranged_h2d) - sum_bytes(&before.ranged_h2d);
    println!(
        "successful write_at-only H2D requests={h2d_calls}, bytes={h2d_bytes} (excludes ctx.upload and other copy paths)"
    );
    // Keep the independent receipt ahead of the allocation gate, including on failure.
    let mut ids = vec![next];
    let mut heads = vec![head_hash];
    for _ in 1..16 {
        let (id, hash) = decode_fingerprint(
            &executor,
            &mut serial.state,
            *ids.last().unwrap(),
            &mut sampler,
        );
        ids.push(id);
        heads.push(hash);
    }
    assert_eq!(serial.state.fed(), 38);
    let mut kv = Sha256::new();
    for layer in serial.state.test_kv() {
        for plane in [&layer.keys, &layer.values] {
            kv.update(canonical_plane(&executor.stream, plane, 0..38));
        }
    }
    let golden = serde_json::json!({
        "start_fed": 22, "end_fed": 38, "seed": seed, "ids": ids,
        "logits_sha256": heads, "layer_major_k_v_f16_sha256": format!("{:x}", kv.finalize()),
    });
    println!("ALLOC_REUSE_GOLDEN={golden}");
    // Frozen from independently qualified b275, never regenerated from this candidate.
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("cuda_executor_alloc_golden.json")).unwrap();
    assert_eq!(
        golden, expected,
        "entire independent allocation baseline receipt"
    );
    assert_eq!(
        (h2d_calls, h2d_bytes),
        (99, 100_384),
        "unchanged ranged KV/sampler writes"
    );
    // The 49 per-layer EXL3 projection arenas remain transient; State scratch must not be.
    assert_eq!(
        private_requests, 49,
        "only the known per-layer private arenas"
    );
    assert_eq!(
        (hot_calls, hot_bytes),
        (0, 0),
        "warm real decode must reuse normalization, CPF, and router-publication scratch"
    );
}

#[test]
#[ignore = "requires admitted CUDA and full EXL3 pack; real B8 assertion RED"]
fn b8_direct_prefill_batches_8_8_5_on_real_pack() {
    let (mut executor, _serial, mut batch) = real_executor();
    let prompt = fixture_tokens(22);
    let before = test_counts();
    let commits = test_commit_ends().len();
    let heads = test_head_reads();
    executor.prefill(&mut batch, &prompt).unwrap();
    let counts = count_delta(before);
    println!(
        "direct22 actual [producer,KV writes,CPF queries,commits]={counts:?}; head reads={}",
        test_head_reads() - heads
    );
    assert_eq!(batch.state.fed(), 21);
    assert_eq!(batch.core.pending(), Some(prompt[21]));
    assert_eq!(
        counts,
        [147, 294, 1029, 147],
        "bounded B8 prefill must issue three layer batches, not 21 serial bodies"
    );
    let ends: Vec<usize> = [8, 16, 21]
        .into_iter()
        .flat_map(|end| std::iter::repeat_n(end, 49))
        .collect();
    assert_eq!(
        &test_commit_ends()[commits..],
        ends.as_slice(),
        "actual 8/8/5 commit ranges across all 49 layers"
    );
    assert_eq!(
        test_head_reads() - heads,
        0,
        "feed-only prefill must not read vocabulary heads"
    );
    assert!(
        batch
            .state
            .test_kv()
            .iter()
            .all(|cache| cache.cursor.committed_prefix() == 21)
    );
}

#[test]
#[ignore = "requires admitted CUDA and full EXL3 pack; real worker-piece assertion RED"]
fn b8_worker_prefill_batches_7_8_6_on_real_pack() {
    let (mut executor, _serial, mut batch) = real_executor();
    let prompt = fixture_tokens(22);
    let mut offset = 0;
    for (input_rows, body_rows) in [(8, 7), (8, 8), (6, 6)] {
        let before = test_counts();
        let commits = test_commit_ends().len();
        let heads = test_head_reads();
        executor
            .prefill(&mut batch, &prompt[offset..offset + input_rows])
            .unwrap();
        let counts = count_delta(before);
        println!("worker piece input={input_rows} body={body_rows} actual={counts:?}");
        assert_eq!(
            counts,
            [49, 98, body_rows * 49, 49],
            "one worker piece must use one body batch, including its earlier pending token"
        );
        assert_eq!(test_head_reads() - heads, 0);
        offset += input_rows;
        assert_eq!(
            &test_commit_ends()[commits..],
            [offset - 1; 49].as_slice(),
            "actual worker 7/8/6 commit ranges"
        );
        assert_eq!(batch.state.fed(), offset - 1);
        assert_eq!(executor.history(&batch), &prompt[..offset]);
        assert_eq!(batch.core.pending(), Some(prompt[offset - 1]));
    }
    assert_eq!(
        executor.prefill_piece(),
        Some(8),
        "the real worker must receive the bounded B8 hint"
    );
}

#[test]
#[ignore = "requires admitted CUDA and full EXL3 pack; bitwise same-pack tails and ring proof"]
fn b8_real_prefill_bitwise_tails_and_ring_boundaries() {
    let (mut executor, mut serial, mut batch) = real_executor();
    executor.prefill(&mut batch, &fixture_tokens(9)).unwrap();
    // Keep the full 8-row case immediately before a short tail, after warming the arena.
    for length in std::iter::once(9).chain(2..=8).chain(std::iter::once(22)) {
        let prompt = fixture_tokens(length);
        executor.reset(&mut serial).unwrap();
        executor.reset(&mut batch).unwrap();
        serial_prefill(&mut executor, &mut serial, &prompt);
        batch
            .state
            .test_dirty_hot_scratch(&executor.stream)
            .unwrap();
        executor.prefill(&mut batch, &prompt).unwrap();
        assert_same_kv(&executor, &serial, &batch);
        assert_same_trajectory(&executor, &mut serial, &mut batch);
    }
    executor.reset(&mut serial).unwrap();
    executor.reset(&mut batch).unwrap();
    let prompt = fixture_tokens(529);
    let mut offset = 0;
    // These calls start their bodies at 0, 512, 513 and 520; the last wraps the 520-row ring.
    for input_rows in [513, 1, 7, 8] {
        let part = &prompt[offset..offset + input_rows];
        serial_prefill(&mut executor, &mut serial, part);
        batch
            .state
            .test_dirty_hot_scratch(&executor.stream)
            .unwrap();
        executor.prefill(&mut batch, part).unwrap();
        offset += input_rows;
        assert_eq!(batch.state.fed(), offset - 1);
        assert_same_kv(&executor, &serial, &batch);
        println!(
            "real fixture logical F16 KV bitwise at prefix {} (physical 513 vs 520)",
            offset - 1
        );
    }
    assert_same_trajectory(&executor, &mut serial, &mut batch);
    executor.stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires admitted CUDA and full EXL3 pack; one append split across the physical ring"]
fn b8_real_prefill_splits_ring_write() {
    let (mut executor, mut serial, mut batch) = real_executor();
    let prompt = fixture_tokens(528);
    serial_prefill(&mut executor, &mut serial, &prompt[..520]);
    executor.prefill(&mut batch, &prompt[..520]).unwrap();
    assert_eq!(batch.state.fed(), 519);
    assert_same_kv(&executor, &serial, &batch);
    serial_prefill(&mut executor, &mut serial, &prompt[520..]);
    batch
        .state
        .test_dirty_hot_scratch(&executor.stream)
        .unwrap();
    let before = test_counts();
    executor.prefill(&mut batch, &prompt[520..]).unwrap();
    assert_eq!(batch.state.fed(), 527);
    // The 35 sliding layers write 1+7 rows in each plane; the 14 full layers write once.
    assert_eq!(count_delta(before), [49, 168, 392, 49]);
    assert_same_kv(&executor, &serial, &batch);
    assert_same_trajectory(&executor, &mut serial, &mut batch);
    executor.stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires admitted CUDA and full EXL3 pack; late all49 failure and reset replay"]
fn b8_failed_chunk_poison_all49_and_reset() {
    let (mut executor, mut serial, mut batch) = real_executor();
    let prompt = fixture_tokens(30);
    executor.prefill(&mut batch, &prompt[..22]).unwrap();
    let prefix = batch.state.fed();
    assert_eq!(prefix, 21);
    let saved = {
        let norm = executor.model.layer(48).unwrap().post_attention_norm;
        let mut saved = [0; 4];
        norm.buffer()
            .read_at(&executor.stream, norm.offset(), &mut saved)
            .unwrap();
        norm.buffer()
            .write_at(
                &executor.stream,
                norm.offset(),
                &f32::INFINITY.to_ne_bytes(),
            )
            .unwrap();
        saved
    };
    let before = test_counts();
    let failed = executor.prefill(&mut batch, &prompt[22..]);
    {
        let norm = executor.model.layer(48).unwrap().post_attention_norm;
        norm.buffer()
            .write_at(&executor.stream, norm.offset(), &saved)
            .unwrap();
    }
    assert!(
        matches!(failed, Err(EngError::ModelOutput(_))),
        "last-layer numeric failure must remain recoverable: {failed:?}"
    );
    assert_eq!(
        count_delta(before)[3],
        0,
        "a failed chunk must commit no layer cursor"
    );
    assert!(batch.state.test_poisoned());
    assert_eq!(batch.state.fed(), prefix);
    assert_eq!(batch.state.test_kv().len(), 49);
    for (layer, cache) in batch.state.test_kv().iter().enumerate() {
        assert_eq!(
            cache.cursor.committed_prefix(),
            prefix,
            "failed layer {layer}"
        );
        assert!(
            cache.cursor.test_poisoned(),
            "unpoisoned failed layer {layer}"
        );
    }
    assert!(executor.prefill(&mut batch, &prompt[22..]).is_err());
    assert_eq!(batch.state.fed(), prefix);
    executor.ctx.ensure_usable().unwrap();
    executor.reset(&mut batch).unwrap();
    assert!(!batch.state.test_poisoned());
    assert!(
        batch
            .state
            .test_kv()
            .iter()
            .all(|cache| cache.cursor.committed_prefix() == 0 && !cache.cursor.test_poisoned())
    );
    serial_prefill(&mut executor, &mut serial, &prompt[..22]);
    executor.prefill(&mut batch, &prompt[..22]).unwrap();
    assert_same_kv(&executor, &serial, &batch);
    assert_same_trajectory(&executor, &mut serial, &mut batch);
}

fn cache_single_state(
    context: usize,
    bundle: &crate::tokenizer::TokenizerBundle,
) -> (FlintExecutor, FlintSession, bool) {
    let late_health = pinned_late_health_arm();
    let expected = match std::env::var(ALLOC_CACHE_VAR) {
        Err(std::env::VarError::NotPresent) => true,
        Ok(value) if matches!(value.as_str(), "0" | "1") => value == "1",
        other => panic!("cache qualification requires unset/1 ON or explicit 0 OFF: {other:?}"),
    };
    let root = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in real EXL3 pack"));
    let options: FlintOptions = serde_json::from_value(serde_json::json!({
        "context_limit": context, "kv_profile": "f16", "prefill_chunk_size": 8,
    }))
    .unwrap();
    let (executor, session) = FlintExecutor::open(
        &root,
        &options,
        bundle.eos_token_ids().to_vec(),
        bundle.vocab_size() as usize,
    )
    .unwrap();
    assert_eq!(executor.ctx.alloc_cache_enabled(), expected);
    assert_eq!(session.state.test_late_health_enabled(), late_health);
    assert_eq!(executor.prefill_chunk_size, 8);
    assert_eq!(session.state.fed(), 0);
    assert!(session.core.history().is_empty());
    assert_eq!(session.core.pending(), None);
    (executor, session, expected)
}

fn cache_long_single_state(prompt_len: usize) {
    // Compare OFF/ON full-head fingerprints offline; each process emits one arm only.
    // This existing loader verifies the factory's a263cce0... tokenizer SHA and companion pins.
    let bundle = crate::tokenizer::flint_tests::load_bundle();
    let context = prompt_len + 64;
    let (mut executor, mut session, enabled) = cache_single_state(context, &bundle);
    let prompt = fixture_tokens(prompt_len);
    executor.prefill(&mut session, &prompt).unwrap();
    assert_eq!(session.state.fed(), prompt_len - 1);
    assert_eq!(executor.history(&session), prompt.as_slice());
    assert_eq!(session.core.pending(), prompt.last().copied());
    let mut pending = session.core.checked_pending().unwrap();
    let mut sampler = Sampler::greedy();
    let mut ids = Vec::new();
    let mut heads = Vec::new();
    for _ in 0..17 {
        let (id, hash) = decode_fingerprint(&executor, &mut session.state, pending, &mut sampler);
        ids.push(id);
        heads.push(hash);
        pending = id;
    }
    assert_eq!(session.state.fed(), prompt_len + 16);
    // Raw fingerprint steps intentionally do not publish into DecodeSession.
    assert_eq!(executor.history(&session), prompt.as_slice());
    assert_eq!(session.core.pending(), prompt.last().copied());
    executor.ctx.ensure_usable().unwrap();
    let receipt = serde_json::json!({
        "schema": "flint-cache-long-smoke-v1",
        "arm": {"alloc_cache_enabled": enabled},
        "context_limit": context, "prompt_tokens": prompt_len,
        "prefill_fed": prompt_len - 1, "end_fed": session.state.fed(),
        "prefill_chunk_size": 8, "kv_profile": "f16", "decode_steps": 17,
        "tokenizer_sha256": "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68",
        "fingerprint": {
            "ids": ids, "logits_sha256": heads,
            "text": bundle.tokenizer().decode(&ids).unwrap(),
        },
    });
    println!("FLINT_CACHE_LONG_SMOKE={receipt}");
}

#[test]
#[ignore = "requires admitted CUDA, real EXL3 pack and pinned FLINT_MODEL_DIR; one fresh process per arm"]
fn control_cache_16k_single_state_identity() {
    cache_long_single_state(16_384);
}

#[test]
#[ignore = "requires admitted CUDA, real EXL3 pack and pinned FLINT_MODEL_DIR; one fresh process per arm"]
fn control_cache_32k_single_state_identity() {
    cache_long_single_state(32_768);
}

#[test]
#[ignore = "requires admitted CUDA, real EXL3 pack and pinned FLINT_MODEL_DIR; one fresh process per arm"]
fn control_cache_three_turn_append_only() {
    use crate::tokenizer::IncrementalDecoder;

    let bundle = crate::tokenizer::flint_tests::load_bundle();
    let (mut executor, mut session, enabled) = cache_single_state(4096, &bundle);
    assert_eq!(executor.prefix_limits().roots, 0);
    assert_eq!(executor.prefix_limits().entries, 0);
    let mut expected_history = Vec::new();
    let mut turns = Vec::new();
    for prompt_len in [512, 768, 1024] {
        assert_eq!(executor.history(&session), expected_history.as_slice());
        let before_fed = session.state.fed();
        let before_history = expected_history.len();
        let suffix = fixture_tokens(prompt_len - before_history);
        expected_history.extend_from_slice(&suffix);
        // Only the new suffix enters prefill; earlier generated tokens remain in the same state.
        executor.prefill(&mut session, &suffix).unwrap();
        assert_eq!(session.state.fed(), prompt_len - 1);
        assert!(session.state.fed() > before_fed);
        assert_eq!(executor.history(&session), expected_history.as_slice());
        assert_eq!(session.core.prompt_len(), prompt_len);
        assert_eq!(session.core.pending(), expected_history.last().copied());
        assert!(!session.core.is_finished());
        let mut generation =
            GenerationState::new(IncrementalDecoder::new(bundle.tokenizer())).with_max_tokens(18);
        let mut ids = Vec::new();
        let mut published_text = String::new();
        // Stop before the length finish so the next suffix can extend this live session.
        for index in 0..17 {
            let batch = executor.decode(&mut session, &mut generation).unwrap();
            assert_eq!(batch.tokens.len(), 1);
            assert_eq!(
                batch.finish, None,
                "unexpected EOS or context/length finish"
            );
            expected_history.extend_from_slice(&batch.tokens);
            ids.extend_from_slice(&batch.tokens);
            published_text.push_str(&batch.text);
            assert_eq!(executor.history(&session), expected_history.as_slice());
            assert_eq!(session.state.fed(), prompt_len + index);
            assert_eq!(session.core.pending(), batch.tokens.last().copied());
            assert!(!session.core.is_finished());
            executor.after_emit().unwrap();
        }
        published_text.push_str(&generation.decoder.finish().unwrap());
        executor.boundary(&mut session);
        assert_eq!(session.state.fed() + 1, expected_history.len());
        assert!(expected_history.len() < 4096);
        turns.push(serde_json::json!({
            "turn": turns.len() + 1, "prompt_tokens": prompt_len,
            "prior_history_tokens": before_history, "appended_tokens": suffix.len(),
            "prefill_fed": prompt_len - 1, "end_fed": session.state.fed(),
            "history_tokens": expected_history.len(), "pending": session.core.pending(),
            "ids": ids, "text": bundle.tokenizer().decode(&ids).unwrap(),
            "published_text": published_text,
        }));
    }
    executor.ctx.ensure_usable().unwrap();
    let receipt = serde_json::json!({
        "schema": "flint-cache-three-turn-v1",
        "arm": {"alloc_cache_enabled": enabled},
        "context_limit": 4096, "prefill_chunk_size": 8, "kv_profile": "f16",
        "prefix_limits": {"roots": 0, "entries": 0},
        "tokenizer_sha256": "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68",
        "fingerprint": {"turns": turns},
    });
    println!("FLINT_CACHE_THREE_TURN_SMOKE={receipt}");
}

#[test]
#[ignore = "requires admitted CUDA, full EXL3 pack and independent same-base RED receipt"]
fn context_2048_same_pack_bitwise_golden() {
    let late_health = pinned_late_health_arm();
    let root = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("opt-in real EXL3 pack"));
    let prompt = fixture_tokens(2049);
    let mut cases = Vec::new();
    for chunk in [1_usize, 8] {
        let options: FlintOptions = serde_json::from_value(serde_json::json!({
            "context_limit": 2112, "kv_profile": "f16", "prefill_chunk_size": chunk,
        }))
        .unwrap();
        // Sequential opens charge the actual production plan and keep only one State live.
        let (mut executor, mut session) =
            FlintExecutor::open(&root, &options, vec![2], VOCAB).unwrap();
        assert_eq!(session.state.test_late_health_enabled(), late_health);
        let before = test_counts();
        let heads = test_head_reads();
        executor.prefill(&mut session, &prompt).unwrap();
        let prefill_counts = count_delta(before);
        let batches = 2048 / chunk;
        assert_eq!(
            prefill_counts,
            [49 * batches, 98 * batches, 49 * 2048, 49 * batches]
        );
        // The scalar adapter evaluates/discards each fed row's head; B8 is feed-only.
        let expected_heads = if chunk == 1 { prompt.len() - 1 } else { 0 };
        assert_eq!(test_head_reads() - heads, expected_heads);
        assert_eq!(session.state.fed(), 2048);
        assert_eq!(session.core.history(), prompt.as_slice());
        assert_eq!(session.core.pending(), Some(prompt[2048]));
        let mut pending = session.core.checked_pending().unwrap();
        let mut sampler = Sampler::greedy();
        let mut ids = Vec::new();
        let mut heads = Vec::new();
        for _ in 0..17 {
            let (id, hash) =
                decode_fingerprint(&executor, &mut session.state, pending, &mut sampler);
            ids.push(id);
            heads.push(hash);
            pending = id;
        }
        assert_eq!(session.state.fed(), 2065);
        assert_eq!(session.state.test_kv().len(), 49);
        let mut kv = Sha256::new();
        for (layer, cache) in session.state.test_kv().iter().enumerate() {
            assert_eq!(cache.cursor.committed_prefix(), 2065);
            assert!(!cache.cursor.test_poisoned());
            let start = if FULL_LAYERS.contains(&(layer as u64)) {
                0
            } else {
                1552
            };
            for plane in [&cache.keys, &cache.values] {
                kv.update(canonical_plane(&executor.stream, plane, start..2065));
            }
        }
        cases.push(serde_json::json!({
            "prefill_chunk_size": chunk,
            "prefill_counts": prefill_counts,
            "fingerprint": {
                "ids": ids, "logits_sha256": heads,
                "layer_major_live_k_v_f16_sha256": format!("{:x}", kv.finalize()),
            },
        }));
    }
    let receipt = serde_json::json!({
        "schema": "flint-context2048-v1", "context_limit": 2112,
        "prompt_tokens": 2049, "prefill_fed": 2048, "end_fed": 2065,
        "continuations": 16, "chunks": cases,
    });
    // RED prints both modes before the intentional missing-fixture failure, after GPU owners drop.
    println!("CONTEXT2048_GOLDEN={receipt}");
    assert_eq!(
        receipt["chunks"][0]["fingerprint"], receipt["chunks"][1]["fingerprint"],
        "production chunk 1 and chunk 8 must agree bitwise"
    );
    let path = std::env::var_os("FLINT_CONTEXT2048_GOLDEN")
        .expect("set FLINT_CONTEXT2048_GOLDEN to the independently captured RED receipt");
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(
        receipt, expected,
        "entire independent same-base RED receipt"
    );
}

#[test]
#[ignore = "requires admitted GPU and real EXL3 weights; explicit late-health/cache arms"]
fn late_health_attention_status_budget_and_bitwise_probe() {
    use crate::family::flint::cuda_ffn::device::runtime::test_health_observations;
    let enabled = match std::env::var("FIFI_FLINT_LATE_HEALTH").as_deref() {
        Ok("1") | Err(std::env::VarError::NotPresent) => true,
        Ok("0") => false,
        other => panic!("explicit late-health 0/1 test arm required, got {other:?}"),
    };
    let (executor, mut session, spare) = real_executor_at_context(CONTEXT, true);
    let decoder = executor.decoder().unwrap();
    let tokens = fixture_tokens(4);
    let mut digest = Sha256::new();
    let mut head_probe_ids = Vec::new();
    let before = executor.ctx.scalar_read_requests();
    let bulk_before = test_health_observations();
    for &token in &tokens {
        let next = decoder
            .step(&mut session.state, token, |_, logits| {
                for value in logits {
                    digest.update(value.to_bits().to_le_bytes());
                }
                Ok(logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0 as u32)
            })
            .unwrap();
        head_probe_ids.push(next);
    }
    let reads = executor.ctx.scalar_read_requests() - before;
    let expected_per_step = if enabled { 103 } else { 1132 };
    let bulk = test_health_observations() - bulk_before;
    assert_eq!(session.state.fed(), tokens.len());
    assert!(!session.state.test_poisoned());
    assert!(
        session
            .state
            .test_kv()
            .iter()
            .all(|kv| !kv.cursor.test_poisoned())
    );
    executor.stream.synchronize().unwrap();
    println!(
        "LATE_HEALTH_PROBE {}",
        serde_json::json!({
            "enabled": enabled, "alloc_cache_enabled": executor.ctx.alloc_cache_enabled(),
            "steps": tokens.len(), "scalar_reads": reads,
            "expected_scalar_reads_per_step": expected_per_step,
            "head_bits_sha256": format!("{:x}", digest.finalize()),
            "head_probe_ids": head_probe_ids,
            "bulk_health_reads": bulk,
            "bulk_health_reads_per_step_expected": if enabled { 49 } else { 0 }
        })
    );
    drop(spare);
    drop(session);
    drop(executor);
    assert_eq!(bulk, tokens.len() * if enabled { 49 } else { 0 });
    assert_eq!(
        reads,
        tokens.len() * expected_per_step,
        "late mode must replace attention's 1029 scalar reads by 49 typed bulk observations"
    );
}

#[test]
#[ignore = "requires admitted GPU and real EXL3 weights; same-binary B8 OFF/ON control pair"]
fn late_health_b8_commits_and_head_bitwise_probe() {
    use crate::family::flint::cuda_ffn::device::runtime::test_health_observations;
    let enabled = match std::env::var("FIFI_FLINT_LATE_HEALTH").as_deref() {
        Ok("1") | Err(std::env::VarError::NotPresent) => true,
        Ok("0") => false,
        other => panic!("explicit late-health 0/1 test arm required, got {other:?}"),
    };
    let (executor, spare, mut session) = real_executor_at_context(CONTEXT, true);
    let decoder = executor.decoder().unwrap();
    let tokens = fixture_tokens(9);
    let before_scalar = executor.ctx.scalar_read_requests();
    let before_bulk = test_health_observations();
    let before_counts = test_counts();
    let before_ends = test_commit_ends().len();
    let before_heads = test_head_reads();
    decoder.feed_rows(&mut session.state, &tokens[..8]).unwrap();
    assert_eq!(session.state.fed(), 8);
    assert_eq!(test_head_reads(), before_heads);
    assert_eq!(
        test_health_observations() - before_bulk,
        if enabled { 49 } else { 0 }
    );
    assert_eq!(
        executor.ctx.scalar_read_requests() - before_scalar,
        if enabled { 99 } else { 1128 }
    );
    let head = decoder
        .step(&mut session.state, tokens[8], |_, logits| {
            let mut digest = Sha256::new();
            for value in logits {
                digest.update(value.to_bits().to_le_bytes());
            }
            Ok(format!("{:x}", digest.finalize()))
        })
        .unwrap();
    let counts = std::array::from_fn::<_, 4, _>(|i| test_counts()[i] - before_counts[i]);
    let ends = test_commit_ends()[before_ends..].to_vec();
    assert_eq!(counts, [98, 196, 441, 98]);
    assert_eq!(ends, [vec![8; 49], vec![9; 49]].concat());
    assert_eq!(test_head_reads() - before_heads, 1);
    assert_eq!(session.state.fed(), 9);
    assert!(!session.state.test_poisoned());
    assert!(
        session
            .state
            .test_kv()
            .iter()
            .all(|kv| kv.cursor.committed_prefix() == 9 && !kv.cursor.test_poisoned())
    );
    let scalar = executor.ctx.scalar_read_requests() - before_scalar;
    let bulk = test_health_observations() - before_bulk;
    assert_eq!(scalar, if enabled { 202 } else { 2260 });
    assert_eq!(bulk, if enabled { 98 } else { 0 });
    executor.stream.synchronize().unwrap();
    println!(
        "LATE_HEALTH_B8_PROBE {}",
        serde_json::json!({
            "enabled": enabled, "alloc_cache_enabled": executor.ctx.alloc_cache_enabled(),
            "control": "same-binary OFF; not a pre-feature receipt", "prefill_rows": 8,
            "fed": session.state.fed(), "head_bits_sha256": head,
            "commit_ends": ends, "counts": counts, "head_reads": test_head_reads() - before_heads,
            "scalar_reads": scalar, "bulk_health_reads": bulk,
        })
    );
    drop(session);
    drop(spare);
    drop(executor);
}

#[test]
#[ignore = "requires admitted GPU and real EXL3 weights; B8 ring-wrap OFF/ON/ON+cache arms"]
fn late_health_b8_ring_wrap_commits_and_head_bitwise_probe() {
    use crate::family::flint::cuda_ffn::device::runtime::test_health_observations;
    let enabled = match std::env::var("FIFI_FLINT_LATE_HEALTH").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        other => panic!("explicit late-health 0/1 ring-wrap arm required, got {other:?}"),
    };
    assert!(
        matches!(
            std::env::var("FIFI_FLINT_CUDA_ALLOC_CACHE").as_deref(),
            Ok("0" | "1")
        ),
        "explicit cache0/cache1 ring-wrap arm required"
    );
    let (executor, spare, mut session) = real_executor_at_context(2048, true);
    assert_eq!(session.state.test_kv().len(), 49);
    let ring_rows = session.state.test_kv()[2].keys.bytes() / ROW_BYTES;
    assert_eq!(ring_rows, 520);
    for plane in [
        &session.state.test_kv()[2].keys,
        &session.state.test_kv()[2].values,
    ] {
        assert_eq!(plane.bytes(), ring_rows * ROW_BYTES);
    }
    let bulk_rows = 1048;
    assert!(bulk_rows >= 2 * ring_rows + 8);
    let tokens = fixture_tokens(bulk_rows + 1);
    let decoder = executor.decoder().unwrap();
    let before_scalar = executor.ctx.scalar_read_requests();
    let before_bulk = test_health_observations();
    let before_counts = test_counts();
    let before_ends = test_commit_ends().len();
    let before_heads = test_head_reads();
    for (group, rows) in tokens[..bulk_rows].as_chunks::<8>().0.iter().enumerate() {
        decoder.feed_rows(&mut session.state, rows).unwrap();
        assert_eq!(session.state.fed(), (group + 1) * 8);
    }
    assert_eq!(session.state.fed(), bulk_rows);
    assert_eq!(test_head_reads(), before_heads);
    let head = decoder
        .step(&mut session.state, tokens[bulk_rows], |_, logits| {
            let mut digest = Sha256::new();
            for value in logits {
                digest.update(value.to_bits().to_le_bytes());
            }
            Ok(format!("{:x}", digest.finalize()))
        })
        .unwrap();
    let groups = bulk_rows / 8 + 1;
    let counts = count_delta(before_counts);
    let ends = test_commit_ends()[before_ends..].to_vec();
    let expected_ends: Vec<_> = (8..=bulk_rows)
        .step_by(8)
        .chain(std::iter::once(tokens.len()))
        .flat_map(|end| [end; 49])
        .collect();
    assert_eq!(ends, expected_ends);
    assert_eq!(
        counts,
        [49 * groups, 98 * groups, 49 * tokens.len(), 49 * groups]
    );
    assert_eq!(test_head_reads() - before_heads, 1);
    assert_eq!(session.state.fed(), tokens.len());
    assert!(!session.state.test_poisoned());
    assert!(
        session
            .state
            .test_kv()
            .iter()
            .all(|kv| kv.cursor.committed_prefix() == tokens.len() && !kv.cursor.test_poisoned())
    );
    let scalar = executor.ctx.scalar_read_requests() - before_scalar;
    let bulk = test_health_observations() - before_bulk;
    assert_eq!(
        scalar,
        if enabled {
            99 * (groups - 1) + 103
        } else {
            1128 * (groups - 1) + 1132
        }
    );
    assert_eq!(bulk, if enabled { 49 * groups } else { 0 });
    executor.stream.synchronize().unwrap();
    println!(
        "LATE_HEALTH_B8_RING_WRAP_PROBE {}",
        serde_json::json!({
            "case": "B8 ring-wrap", "enabled": enabled,
            "alloc_cache_enabled": executor.ctx.alloc_cache_enabled(),
            "control": "same-binary OFF; not a pre-feature receipt",
            "context_limit": 2048, "prefill_chunk_size": 8, "kv_profile": "f16",
            "ring_layer": 2, "ring_rows": ring_rows,
            "bulk_capacity_crossings": bulk_rows / ring_rows, "prefill_rows": bulk_rows,
            "fed": session.state.fed(), "head_bits_sha256": head,
            "commit_ends": ends, "counts": counts, "head_reads": test_head_reads() - before_heads,
            "scalar_reads": scalar, "bulk_health_reads": bulk,
        })
    );
    drop(session);
    drop(spare);
    drop(executor);
}

struct HealthFaultCase {
    checkpoint: usize,
    rows: usize,
    status: u32,
}

fn exercise_health_checkpoint(
    executor: &FlintExecutor,
    session: &mut FlintSession,
    fault: HealthFaultCase,
    pristine: &[u32],
) {
    let HealthFaultCase {
        checkpoint,
        rows,
        status,
    } = fault;
    let decoder = executor.decoder().unwrap();
    let tokens = fixture_tokens(rows);
    let before_head = test_head_reads();
    let before_commits = test_counts()[3];
    let before_ends = test_commit_ends().len();
    executor
        .ctx
        .test_poison_scalar_read_after(1 + checkpoint, status);
    let mut accepted = false;
    let result = if rows == 1 {
        decoder.step(&mut session.state, tokens[0], |_, _| {
            accepted = true;
            Ok(())
        })
    } else {
        decoder.feed_rows(&mut session.state, &tokens)
    };
    let error = result.unwrap_err();
    let expected = match checkpoint {
        1 | 2 | 5 | 6 | 9 | 10 | 15 | 16 if status & 1 != 0 => {
            EngError::ModelOutput("EXL3 projection flagged invalid arithmetic".into())
        }
        13 => EngError::Backend {
            operation: "flint_qk_producer",
            message: format!("Q/K producer numeric flag {status:#x}; output invalid"),
        },
        _ => EngError::ModelOutput(
            "Flint CUDA FFN: late numeric/index health failed; output invalid".into(),
        ),
    };
    assert_eq!(
        error, expected,
        "rows{rows}, checkpoint{checkpoint}, status{status}"
    );
    assert!(
        !executor.ctx.test_health_fault_pending(),
        "physical fault did not fire"
    );
    assert!(!accepted);
    assert_eq!(test_head_reads(), before_head);
    assert_eq!(test_counts()[3], before_commits);
    assert_eq!(test_commit_ends().len(), before_ends);
    assert_eq!(session.state.fed(), 0);
    assert!(session.state.test_poisoned());
    for (layer, kv) in session.state.test_kv().iter().enumerate() {
        assert_eq!(kv.cursor.committed_prefix(), 0);
        assert_eq!(kv.cursor.test_poisoned(), layer == 0, "visited layer only");
    }
    let before_reads = executor.ctx.scalar_read_requests();
    assert!(matches!(
        decoder.step(&mut session.state, tokens[0], |_, _| -> Result<()> {
            panic!("poisoned session must not accept output")
        }),
        Err(EngError::BadRequest(_))
    ));
    assert_eq!(executor.ctx.scalar_read_requests(), before_reads);
    executor.ctx.ensure_usable().unwrap();
    session.state.reset(&executor.stream).unwrap();
    decoder
        .step(&mut session.state, tokens[0], |_, logits| {
            assert!(
                logits
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(pristine.iter().copied()),
                "reset replay differs at rows{rows}, checkpoint{checkpoint}, status{status}"
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(session.state.fed(), 1);
    assert!(!session.state.test_poisoned());
    assert!(
        session
            .state
            .test_kv()
            .iter()
            .all(|kv| kv.cursor.committed_prefix() == 1 && !kv.cursor.test_poisoned())
    );
    session.state.reset(&executor.stream).unwrap();
    println!(
        "LATE_HEALTH_CHECKPOINT rows={rows} checkpoint={checkpoint} status={status:#x} physical/refusal/poison/reset=PASS"
    );
}

#[test]
#[ignore = "requires admitted GPU and real EXL3 weights; pair late-health0/1 cacheOFF"]
fn late_health_all_checkpoints_refuse_before_output_and_reset_bitwise() {
    let (executor, mut session, spare) = real_executor_at_context(CONTEXT, true);
    assert!(!executor.ctx.alloc_cache_enabled());
    let token = fixture_tokens(1)[0];
    let pristine = executor
        .decoder()
        .unwrap()
        .step(&mut session.state, token, |_, logits| {
            Ok(logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>())
        })
        .unwrap();
    session.state.reset(&executor.stream).unwrap();
    for rows in [1, 8] {
        for checkpoint in 0..21 {
            let numeric = match checkpoint {
                3 | 7 | 11 | 17 => 2,
                13 => 8,
                _ => 1,
            };
            for status in [numeric, 1 << 31] {
                exercise_health_checkpoint(
                    &executor,
                    &mut session,
                    HealthFaultCase {
                        checkpoint,
                        rows,
                        status,
                    },
                    &pristine,
                );
            }
        }
    }
    drop(spare);
}

#[test]
fn selected_row_offsets_validate_chunk_and_selection() {
    let row_bytes = 2048 * std::mem::size_of::<f32>();
    for rows in 1..=8 {
        assert_eq!(
            Decoder::selected_row_offsets(rows, &[0, rows - 1]).unwrap(),
            [0, (rows - 1) * row_bytes]
        );
        assert!(Decoder::selected_row_offsets(rows, &[rows]).is_err());
        assert!(Decoder::selected_row_offsets(rows, &[usize::MAX]).is_err());
    }
    for rows in [0, 9, usize::MAX] {
        assert!(Decoder::selected_row_offsets(rows, &[]).is_err());
    }
    assert!(Decoder::selected_row_offsets(8, &[]).unwrap().is_empty());
    assert_eq!(
        Decoder::selected_row_offsets(8, &[6, 0, 7, 6]).unwrap(),
        [6 * row_bytes, 0, 7 * row_bytes, 6 * row_bytes]
    );
}

fn selected_head_forward(
    executor: &FlintExecutor,
    state: &mut State,
    tokens: &[u32],
    selected: &[usize],
) -> Vec<Vec<f32>> {
    let before = test_counts();
    let before_heads = test_head_reads();
    let before_ends = test_commit_ends().len();
    let end = state.fed() + tokens.len();
    let heads = executor
        .decoder()
        .unwrap()
        .feed_rows_selected_logits(state, tokens, selected)
        .unwrap();
    assert_eq!(heads.len(), selected.len());
    assert!(heads.iter().all(|row| row.len() == VOCAB));
    assert_eq!(count_delta(before), [49, 98, 49 * tokens.len(), 49]);
    assert_eq!(test_head_reads() - before_heads, selected.len());
    assert_eq!(&test_commit_ends()[before_ends..], &[end; 49]);
    assert_eq!(state.fed(), end);
    assert!(!state.test_poisoned());
    assert!(
        state
            .test_kv()
            .iter()
            .all(|kv| { kv.cursor.committed_prefix() == end && !kv.cursor.test_poisoned() })
    );
    heads
}

#[test]
#[ignore = "requires admitted GPU and real EXL3 weights; F16 observation contract, not Q8 quality"]
fn selected_heads_preserve_c8_rows_and_commit_coverage() {
    let (executor, spare, mut session) = real_executor_at_context(CONTEXT, true);
    let decoder = executor.decoder().unwrap();
    let tokens = fixture_tokens(9);
    let before_ranges = test_head_ranges().len();
    let before = test_counts();
    let before_heads = test_head_reads();
    let before_ends = test_commit_ends().len();
    for (input, selected) in [
        (&tokens[..0], &[][..]),
        (&tokens[..9], &[0][..]),
        (&tokens[..8], &[8][..]),
        (&tokens[..8], &[usize::MAX][..]),
    ] {
        assert!(matches!(
            decoder.feed_rows_selected_logits(&mut session.state, input, selected),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(test_counts(), before);
        assert_eq!(test_head_reads(), before_heads);
        assert_eq!(test_commit_ends().len(), before_ends);
        assert_eq!(session.state.fed(), 0);
        assert!(!session.state.test_poisoned());
        assert!(
            session
                .state
                .test_kv()
                .iter()
                .all(|kv| { kv.cursor.committed_prefix() == 0 && !kv.cursor.test_poisoned() })
        );
    }
    assert_eq!(test_head_ranges().len(), before_ranges);
    let all = selected_head_forward(
        &executor,
        &mut session.state,
        &tokens[..8],
        &[0, 1, 2, 3, 4, 5, 6, 7],
    );
    session.state.reset(&executor.stream).unwrap();
    let before = test_counts();
    let before_heads = test_head_reads();
    let before_ends = test_commit_ends().len();
    let last = decoder
        .feed_rows_logits(&mut session.state, &tokens[..8])
        .unwrap();
    assert_eq!(last.len(), VOCAB);
    assert!(
        all[7]
            .iter()
            .zip(&last)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(count_delta(before), [49, 98, 49 * 8, 49]);
    assert_eq!(test_head_reads() - before_heads, 1);
    assert_eq!(&test_commit_ends()[before_ends..], &[8; 49]);
    session.state.reset(&executor.stream).unwrap();
    let selected = [6, 0, 7, 6];
    let sparse = selected_head_forward(&executor, &mut session.state, &tokens[..8], &selected);
    assert_eq!(
        &test_head_ranges()[before_ranges..],
        &[
            [0, 8192, 65536],
            [8192, 8192, 65536],
            [16384, 8192, 65536],
            [24576, 8192, 65536],
            [32768, 8192, 65536],
            [40960, 8192, 65536],
            [49152, 8192, 65536],
            [57344, 8192, 65536],
            [57344, 8192, 65536],
            [49152, 8192, 65536],
            [0, 8192, 65536],
            [57344, 8192, 65536],
            [49152, 8192, 65536],
        ]
    );
    for (row, &index) in sparse.iter().zip(&selected) {
        assert!(
            row.iter()
                .zip(&all[index])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "selected row {index} differs from its full-chunk head"
        );
    }
    assert!(
        all[0]
            .iter()
            .zip(&all[7])
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "fixture must distinguish an earlier head from the last-row head"
    );
    drop(spare);
}

fn interim_case_index(case: &str, profile: &str) -> Option<usize> {
    if !matches!(profile, "f16" | "q8") {
        return None;
    }
    match case {
        "P1" => Some(0),
        "P2" => Some(1),
        "P3" => Some(2),
        _ => None,
    }
}

#[test]
fn interim_selection_accepts_only_the_six_declared_arms() {
    for (case, index) in [("P1", 0), ("P2", 1), ("P3", 2)] {
        for profile in ["f16", "q8"] {
            assert_eq!(interim_case_index(case, profile), Some(index));
        }
    }
    for (case, profile) in [("P0", "f16"), ("P4", "q8"), ("P1", "F16"), ("P1", "q4")] {
        assert_eq!(interim_case_index(case, profile), None);
    }
}

#[test]
#[ignore = "requires an admitted report-only CUDA lease, sealed inputs, and one fresh process per arm"]
fn control_q8_interim_prompt_generation() {
    use crate::tokenizer::{DecodeFinish, IncrementalDecoder, PromptMessage, Role, Thinking};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    let stamp = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    };
    let selected = std::env::var("FLINT_INTERIM_CASE").unwrap();
    let profile = std::env::var("FLINT_INTERIM_KV_PROFILE").unwrap();
    let index = interim_case_index(&selected, &profile).expect("select P1/P2/P3 and f16/q8");
    assert_eq!(
        std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE").as_deref(),
        Ok("1")
    );
    assert!(std::env::var_os("FIFI_FLINT_CUDA_EXPERT_CACHE").is_none());
    assert!(pinned_late_health_arm(), "late-health must be ON");
    let seal = fs::read(std::env::var_os("FLINT_INTERIM_INPUT_SEAL").unwrap()).unwrap();
    let seal_sha = format!("{:x}", Sha256::digest(&seal));
    assert_eq!(
        seal_sha,
        "1655f56e24a5b3ccb110a7356f4dd0f3f60a57fec4e2063a8bfaab4d40697e72"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&seal).unwrap();
    let case = &manifest["cases"][index];
    let sealed_bytes = |entry: &serde_json::Value| {
        let bytes = fs::read(entry["path"].as_str().unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            entry["sha256"].as_str().unwrap()
        );
        bytes
    };
    let bundle = crate::tokenizer::flint_tests::load_bundle();
    let messages = [PromptMessage {
        role: Role::User,
        content: Some(String::from_utf8(sealed_bytes(&case["raw_prompt"])).unwrap()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }];
    let ids = bundle
        .encode_prompt(&messages, &[], Thinking::Default)
        .unwrap();
    let frozen: Vec<u32> = serde_json::from_slice(&sealed_bytes(&case["prompt_ids"])).unwrap();
    assert!(ids == frozen, "native prompt IDs differ from frozen IDs");
    assert_eq!(
        ids.len() as u64,
        case["native_rendered_prompt_ids_count"].as_u64().unwrap()
    );
    assert!(ids.len() + 256 < 4096);
    let receipt_path = PathBuf::from(std::env::var_os("FLINT_INTERIM_RECEIPT").unwrap());
    assert!(receipt_path.is_absolute());
    assert!(
        matches!(fs::symlink_metadata(&receipt_path), Err(error) if error.kind() == std::io::ErrorKind::NotFound)
    );
    let mut output = tempfile::NamedTempFile::new_in(receipt_path.parent().unwrap()).unwrap();
    assert_eq!(
        output.as_file().metadata().unwrap().permissions().mode() & 0o777,
        0o600
    );
    let started_utc_unix_ms = stamp();
    let load_started = Instant::now();
    let options: FlintOptions = serde_json::from_value(serde_json::json!({
        "context_limit": 4096, "kv_profile": profile, "prefill_chunk_size": 8,
    }))
    .unwrap();
    let root = PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").unwrap());
    let (mut executor, mut session) = FlintExecutor::open(
        &root,
        &options,
        bundle.eos_token_ids().to_vec(),
        bundle.vocab_size() as usize,
    )
    .unwrap();
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    assert!(executor.ctx.alloc_cache_enabled());
    executor.check_prompt(ids.len()).unwrap();
    executor.reset(&mut session).unwrap();
    executor.boundary(&mut session);
    assert!(session.state.test_late_health_enabled());
    assert_eq!(executor.prefill_piece(), Some(8));
    let prefill_started = Instant::now();
    for piece in ids.chunks(8) {
        executor.prefill(&mut session, piece).unwrap();
    }
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut generation = GenerationState::with_sampler(
        IncrementalDecoder::new(bundle.tokenizer()),
        Sampler::greedy(),
    )
    .with_max_tokens(256);
    let decode_started_utc_unix_ms = stamp();
    let decode_started = Instant::now();
    let mut emitted = Vec::new();
    let mut text = String::new();
    let mut batches = Vec::new();
    let finish = loop {
        let batch = executor.decode(&mut session, &mut generation).unwrap();
        batches.push(serde_json::json!({
            "utc_unix_ms": stamp(), "decode_elapsed_ms": decode_started.elapsed().as_secs_f64() * 1000.0,
            "tokens": batch.tokens,
        }));
        emitted.extend_from_slice(&batch.tokens);
        text.push_str(&batch.text);
        executor.after_emit().unwrap();
        if let Some(reason) = batch.finish {
            break reason;
        }
    };
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let ended_utc_unix_ms = stamp();
    assert!(matches!(finish, DecodeFinish::Stop | DecodeFinish::Length));
    assert!(!emitted.is_empty() && emitted.len() <= 256);
    executor.boundary(&mut session);
    executor.ctx.ensure_usable().unwrap();
    let receipt = serde_json::json!({
        "schema": "flint-q8-interim-report-only-v1", "complete": true, "quality_PASS": false, "N": 1,
        "case": selected, "kv_profile": profile, "input_seal_sha256": seal_sha,
        "prompt_sha256": case["raw_prompt"]["sha256"], "input_ids_sha256": case["prompt_ids"]["sha256"],
        "runtime_prompt_ids_count": ids.len(), "context_limit": 4096, "max_new_tokens": 256,
        "prefill_chunk_size": 8, "sampler": "greedy", "alloc_cache_enabled": true, "expert_cache": "UNSET",
        "late_health_enabled": session.state.test_late_health_enabled(),
        "started_utc_unix_ms": started_utc_unix_ms, "decode_started_utc_unix_ms": decode_started_utc_unix_ms,
        "ended_utc_unix_ms": ended_utc_unix_ms, "load_ms": load_ms, "prefill_ms": prefill_ms,
        "decode_ms": decode_ms, "batches": batches, "ids": emitted, "emitted_length": emitted.len(),
        "text": text, "stop_reason": format!("{finish:?}"),
    });
    serde_json::to_writer(output.as_file_mut(), &receipt).unwrap();
    output.as_file().sync_all().unwrap();
    output.persist_noclobber(&receipt_path).unwrap();
}
