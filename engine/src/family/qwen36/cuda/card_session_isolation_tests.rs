//! Per-sequence state on one loaded card stack, with shared weights, scratch and prefix index.
use super::super::{
    card_session_device_tests::{build_rig_session, rig_options, rig_spec},
    native_device_rig::device_present,
};
use super::*;
use crate::family::qwen36::host::HostShape;
use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};

#[test]
fn peer_prefix_requires_one_position_across_a_nonempty_card_set() {
    let dims = super::super::common_tests::dims();
    let shard = |tokens| ExportedState {
        identity: super::super::state::StateIdentity::new(&dims, super::super::dims::KvMode::Fp16),
        tokens,
        delta: vec![],
        kv: vec![],
    };
    assert!(prefix_shard_depth(&[], 0).is_err());
    assert!(prefix_shard_depth(&[shard(1)], 2).is_err());
    assert!(matches!(
        prefix_shard_depth(&[shard(1), shard(2)], 2),
        Err(EngError::Checkpoint(_))
    ));
    assert_eq!(prefix_shard_depth(&[shard(0), shard(0)], 2).unwrap(), 0);
    assert_eq!(prefix_shard_depth(&[shard(9), shard(9)], 2).unwrap(), 9);
}

fn other_sequence(stack: &CardStack, capacity: usize) -> Result<CardSequence, EngError> {
    let spec = rig_spec();
    let dims = StepDims::new(
        &HostShape::from_spec(&spec, 64)?,
        &Inventory::for_spec(&spec)?,
    )?;
    let options = rig_options(capacity);
    let states = stack
        .contexts()
        .iter()
        .zip(stack.streams())
        .enumerate()
        .map(|(card, (context, stream))| {
            let heads = stack.table().kv_rows(card)?.len() / dims.shape.attn.head_dim;
            ResidentState::allocate_with_kv_heads(context, stream, &dims, &options, heads)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CardSequence::new(states))
}

const SHARED: &[u32] = &[11, 12, 13, 14];
const PROMPT_B: &[u32] = &[11, 12, 13, 14, 31, 17, 19];

fn decode_three(
    exec: &mut impl ExecutorOps<Session = DecodeSession>,
    row: &mut DecodeSession,
    tokenizer: &Tokenizer,
) -> Result<Vec<DecodeBatch>, EngError> {
    let mut generation =
        GenerationState::new(IncrementalDecoder::new(tokenizer)).with_max_tokens(3);
    let mut batches = Vec::new();
    for _ in 0..3 {
        let batch = exec.decode(row, &mut generation)?;
        let finished = batch.finish.is_some();
        batches.push(batch);
        if finished {
            break;
        }
    }
    Ok(batches)
}

fn cancel(
    exec: &mut impl ExecutorOps<Session = DecodeSession>,
    row: &mut DecodeSession,
    tokenizer: &Tokenizer,
) -> Result<(), EngError> {
    let mut generation = GenerationState::new(IncrementalDecoder::new(tokenizer));
    generation
        .cancel
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        exec.decode(row, &mut generation)?.finish,
        Some(DecodeFinish::Cancelled)
    );
    Ok(())
}

fn check_batches(got: &[DecodeBatch], cold: &[DecodeBatch]) {
    assert_eq!(got, cold);
    assert!(got.iter().all(|batch| batch.tokens.len() == 1));
    assert!(matches!(
        got.last().and_then(|batch| batch.finish),
        Some(DecodeFinish::Length | DecodeFinish::Stop)
    ));
}

fn interleave(devices: [usize; 2]) -> Result<(), EngError> {
    const CAP: usize = 32;
    let (mut exec, stack, _, _, _) =
        build_rig_session(devices, CAP, StoreLimits::default(), None, false, false)?;
    let mut parked = other_sequence(stack, CAP)?;
    let tokenizer = Tokenizer::from_bytes(include_bytes!(
        "../../../server/testdata/tokenizer/tokenizer.json"
    ))?;
    exec.sampler_vocab = 128; // Every sampled id belongs to the byte tokenizer.
    exec.eos = Arc::from([127]);
    let (mut a, mut b) = (DecodeSession::default(), DecodeSession::default());

    ExecutorOps::reset(&mut exec, &mut a)?;
    exec.prefill(&mut a, SHARED)?;
    assert!(exec.save_prefix(&a, SHARED, SaveKind::Tip)?);
    exec.prefill(&mut a, &[22, 23])?;
    std::mem::swap(&mut parked, &mut exec.sequence);
    ExecutorOps::reset(&mut exec, &mut b)?;
    assert_eq!(exec.restore_prefix(&mut b, PROMPT_B)?, SHARED.len());
    exec.prefill(&mut b, &PROMPT_B[SHARED.len()..])?;
    assert!(exec.prefix.evict_lru_one(None));
    assert_eq!(exec.prefix.key_len(), None);

    // A malformed final shard must leave the earlier shard in B's state.
    let before = exec.export_states_for_test()?;
    let mut invalid = before.clone();
    let first = &mut invalid[0].delta[0].recurrent[0];
    *first = if *first == 0.0 { 1.0 } else { 0.0 };
    invalid[1].kv[0].planes[0].pop();
    assert!(matches!(
        exec.import_prefix_shards(&invalid),
        Err(EngError::Checkpoint(_))
    ));
    assert!(
        exec.export_states_for_test()? == before,
        "a refused tail shard changed live state"
    );
    assert!(!exec.poisoned);

    std::mem::swap(&mut parked, &mut exec.sequence);
    cancel(&mut exec, &mut a, &tokenizer)?;
    assert_eq!(exec.fed(), 0);
    std::mem::swap(&mut parked, &mut exec.sequence);

    let got = decode_three(&mut exec, &mut b, &tokenizer)?;
    let got_state = exec.export_states_for_test()?;
    ExecutorOps::reset(&mut exec, &mut b)?;
    exec.prefill(&mut b, PROMPT_B)?;
    let cold = decode_three(&mut exec, &mut b, &tokenizer)?;
    check_batches(&got, &cold);
    assert!(
        got_state == exec.export_states_for_test()?,
        "resumed state differs from cold recomputation"
    );
    // A partially copied group can only be used after the shared ring and all shards reset.
    exec.poisoned = true;
    assert!(matches!(
        exec.import_prefix_shards(&before),
        Err(EngError::Backend { .. })
    ));
    assert!(exec.export_states_for_test()? == got_state);
    ExecutorOps::reset(&mut exec, &mut b)?;
    assert_eq!(exec.fed(), 0);
    assert!(!exec.poisoned);
    Ok(())
}

#[test]
#[ignore = "requires one CUDA device with two logical shards; run alone"]
fn interleaved_rows_survive_eviction_and_cancellation_on_one_gpu() {
    assert!(device_present(
        "interleaved_rows_survive_eviction_and_cancellation_on_one_gpu"
    ));
    interleave([0, 0]).unwrap();
}

#[test]
#[ignore = "requires two peer-enabled CUDA devices; run alone"]
fn interleaved_rows_survive_eviction_and_cancellation_on_peer_gpus() {
    assert!(device_present(
        "interleaved_rows_survive_eviction_and_cancellation_on_peer_gpus"
    ));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).unwrap();
    crate::kernels::cuda::peer::check_peer_facts(&facts).unwrap();
    crate::kernels::cuda::peer::enable_peer_access(&devices).unwrap();
    interleave(devices).unwrap();
}

#[test]
#[ignore = "runs the standalone resident CUDA executor; run alone"]
fn interleaved_rows_survive_eviction_and_cancellation_on_standalone_cuda() {
    assert!(device_present(
        "interleaved_rows_survive_eviction_and_cancellation_on_standalone_cuda"
    ));
    standalone().unwrap();
}

fn standalone() -> Result<(), EngError> {
    use super::super::{
        card_session_device_tests::rig_weights, resources::StepResources, rules::QwenRules,
        step::QwenResidentStep,
    };
    use crate::family::project::cuda::{CudaProjector, LoadOptions, WeightLayout};
    let spec = rig_spec();
    let inv = Inventory::for_spec(&spec)?;
    let dims = StepDims::new(&HostShape::from_spec(&spec, 64)?, &inv)?;
    let table = LayerTable::new(&dims, &inv)?;
    let (_dir, gguf, _) = rig_weights(&dims, &inv, &table, false)?;
    let names = gguf
        .header()
        .tensors()
        .iter()
        .map(|tensor| tensor.name())
        .collect::<Vec<_>>();
    let projector = CudaProjector::new()?;
    let weights = projector.load(
        &gguf,
        &names,
        LoadOptions {
            layout: WeightLayout::Native,
            ..Default::default()
        },
    )?;
    let ctx = projector.context();
    let options = rig_options(32);
    let make_rules = || QwenRules::new(vec![127], 128, spec.vocab as usize);
    let arena_a = StepResources::new(ctx, dims.clone(), &inv, &options)?;
    let arena_b = StepResources::new(ctx, dims, &inv, &options)?;
    let mut a = QwenResidentStep::new(ctx, &arena_a, &weights, &options, make_rules()?)?;
    let mut b = QwenResidentStep::new(ctx, &arena_b, &weights, &options, make_rules()?)?;
    let tok = Tokenizer::from_bytes(include_bytes!(
        "../../../server/testdata/tokenizer/tokenizer.json"
    ))?;
    let (mut row_a, mut row_b) = (DecodeSession::default(), DecodeSession::default());
    ExecutorOps::reset(&mut a, &mut row_a)?;
    a.prefill(&mut row_a, SHARED)?;
    assert!(a.save_prefix(&row_a, SHARED, SaveKind::Tip)?);
    a.prefill(&mut row_a, &[22, 23])?;
    ExecutorOps::reset(&mut b, &mut row_b)?;
    // Hand over only the pageable prefix index. Each step keeps its own stream, KV and DeltaNet state.
    std::mem::swap(&mut a.prefix, &mut b.prefix);
    assert_eq!(b.restore_prefix(&mut row_b, PROMPT_B)?, SHARED.len());
    b.prefill(&mut row_b, &PROMPT_B[SHARED.len()..])?;
    assert!(b.prefix.evict_lru_one(None));
    cancel(&mut a, &mut row_a, &tok)?;
    assert_eq!(a.tokens(), 0);
    let got = decode_three(&mut b, &mut row_b, &tok)?;
    let state = b.export_state()?;
    ExecutorOps::reset(&mut b, &mut row_b)?;
    b.prefill(&mut row_b, PROMPT_B)?;
    check_batches(&got, &decode_three(&mut b, &mut row_b, &tok)?);
    assert!(
        state == b.export_state()?,
        "standalone state differs from cold recomputation"
    );
    Ok(())
}
