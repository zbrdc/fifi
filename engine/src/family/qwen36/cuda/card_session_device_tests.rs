//! The session-level device rig: a REAL GGUF file holding every tensor of
//! the two-layer rig spec, loaded through the production per-card loaders
//! into a [`CardStack`], wrapped in the [`CardSession`], and driven as the
//! executor would drive it - prompt feeds, logits reads, the fold, the
//! reset, and the wrapper's refusals. The block-level numerics are the
//! card token rig's proof; this rig proves the session wrapper: the step
//! params of the first token, the token-word writes of the later ones, the
//! status fold, the vocabulary-shard concat, and the turn reset.
use std::sync::Arc;

use super::card_block_device_tests::{FileEntry, write_gguf};
use super::card_session::CardSession;
use super::card_step::CardStack;
use super::dims::{KvMode, StepDims, StepOptions};
use super::layer_table::LayerTable;
use super::mixer_geometry::MixerGeometry;
use super::native_device_rig::{Rng, device_present, q4k_rows, raw_values};
use super::resources::{GlobalNames, MoeSpecs};
use super::shard::ShardTable;

use crate::EngError;
use crate::family::AttentionKind;
use crate::family::gguf_model::Inventory;
use crate::family::project::cuda::{BorrowedLoader, LoadOptions, WeightLayout, served_admission};
use crate::family::qwen36::cuda::card_load::card_loads;
use crate::family::qwen36::cuda::card_weights::load_card_weights;
use crate::family::qwen36::host::HostShape;
use crate::family::session::TokenStepper;
use crate::family::spec::{FamilySpec, QWEN3_6_35B_A3B};
use crate::weights::GgufType;
use crate::weights::{GgufLimits, GgufWeights};

/// DeltaNet then Full, the model's head pattern: the two kinds the card
/// paths cover, at a vocabulary the file can afford twice.
static TWO_LAYERS: [AttentionKind; 2] = [AttentionKind::DeltaNet, AttentionKind::Full];

pub(super) fn rig_spec() -> FamilySpec {
    FamilySpec {
        layers: 2,
        vocab: 4096,
        layers_attention: &TWO_LAYERS,
        ..QWEN3_6_35B_A3B
    }
}

/// Best value and best runner-up of a logits row, for the greedy-separation
/// print.
fn top2_of(row: &[f32]) -> (f32, f32) {
    let (mut best, mut runner) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in row {
        if v > best {
            runner = best;
            best = v;
        } else if v > runner {
            runner = v;
        }
    }
    (best, runner)
}

/// One q4_k stack of `experts` experts of `rows` rows of `k` values, in
/// GGUF rank-3 order `[k, rows, experts]`.
fn q4k_stack(experts: usize, rows: usize, k: usize, rng: &mut Rng) -> FileEntry {
    FileEntry {
        name: String::new(),
        ty: GgufType::Q4K,
        dims: vec![k as u64, rows as u64, experts as u64],
        payload: (0..experts).flat_map(|_| q4k_rows(rows, k, rng)).collect(),
    }
}

// One-per-process by design: the session keeps its stack, weights and states
// leaked for its whole life (the 'static owner pattern), so its two CUDA
// adapter leases never release and every later CUDA test in this process
// would fail on lease admission. Run it with --ignored, alone.
#[test]
#[ignore = "holds its two CUDA adapter leases for the process lifetime; run with --ignored, one per process"]
fn a_card_session_feeds_logits_folds_and_resets_over_a_real_gguf() {
    const NAME: &str = "a_card_session_feeds_logits_folds_and_resets_over_a_real_gguf";
    if !device_present(NAME) {
        return;
    }
    let result = run_session_rig(NAME, [0, 0], false);
    result.expect("the card session rig");
}

#[test]
#[ignore = "requires two peer-enabled GPUs and holds their adapter leases for the process lifetime"]
fn prefill_without_head_matches_normal_logits_on_peer_gpus() {
    const NAME: &str = "prefill_without_head_matches_normal_logits_on_peer_gpus";
    assert!(device_present(NAME));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    run_session_rig(NAME, devices, false).expect("two-GPU session parity");
}

#[test]
#[ignore = "requires two peer-enabled GPUs and holds their adapter leases for the process lifetime"]
fn async_broadcasts_preserve_logits_on_peer_gpus() {
    const NAME: &str = "async_broadcasts_preserve_logits_on_peer_gpus";
    assert!(device_present(NAME));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    run_session_rig(NAME, devices, true).expect("async two-GPU session parity");
}

pub(super) fn build_rig_session_for_chunk(
    devices: [usize; 2],
    context_tokens: usize,
    prefix_limits: crate::family::session::StoreLimits,
    chunk_rows: usize,
    prepare_async: bool,
) -> Result<
    (
        CardSession,
        &'static CardStack,
        &'static [&'static crate::family::project::cuda::CudaWeights],
        GlobalNames,
        usize,
    ),
    EngError,
> {
    build_rig_session(
        devices,
        context_tokens,
        prefix_limits,
        Some(chunk_rows),
        prepare_async,
    )
}

fn build_rig_session(
    devices: [usize; 2],
    context_tokens: usize,
    prefix_limits: crate::family::session::StoreLimits,
    chunk_rows: Option<usize>,
    prepare_async: bool,
) -> Result<
    (
        CardSession,
        &'static CardStack,
        &'static [&'static crate::family::project::cuda::CudaWeights],
        GlobalNames,
        usize,
    ),
    EngError,
> {
    let spec = rig_spec();
    let inv = Inventory::for_spec(&spec)?;
    let shape = HostShape::from_spec(&spec, 64)?;
    let d = StepDims::new(&shape, &inv)?;
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let table_layers = LayerTable::new(&d, &inv)?;

    // The file: every mixer, MoE and global tensor of the spec, full-width.
    let geometry = MixerGeometry::new(&d)?;
    let moe_specs = MoeSpecs::new(&d)?;
    let mut rng = Rng::new(0x5E55);
    let mut entries: Vec<FileEntry> = Vec::new();
    for layer in 0..table_layers.len() {
        let kind = table_layers.slot(layer)?.kind;
        for spec in geometry.dense_specs(WeightLayout::Native, kind)? {
            entries.push(FileEntry {
                name: table_layers.name(layer, spec.role)?.to_owned(),
                ty: GgufType::Q4K,
                dims: vec![spec.k as u64, spec.rows as u64],
                payload: q4k_rows(spec.rows, spec.k, &mut rng),
            });
        }
        for spec in geometry.raw_specs(kind)? {
            // The conv1d is the ONE raw tensor the card loads slice by
            // CHANNEL rows, so the file carries it rank 2 - kernel values
            // wide, one row per channel - like the shipped checkpoint.
            let dims = if spec.role == crate::family::gguf_model::Role::SsmConv1d {
                vec![
                    d.shape.delta.conv_kernel as u64,
                    d.shape.delta.conv_dim() as u64,
                ]
            } else {
                vec![spec.values as u64]
            };
            entries.push(FileEntry {
                name: table_layers.name(layer, spec.role)?.to_owned(),
                ty: GgufType::F32,
                dims,
                payload: raw_values(&spec.role, spec.values)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            });
        }
        let f32_entry = |role, values: usize| -> Result<FileEntry, EngError> {
            Ok(FileEntry {
                name: table_layers.name(layer, role)?.to_owned(),
                ty: GgufType::F32,
                dims: vec![values as u64],
                payload: raw_values(&role, values)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            })
        };
        let (h, m) = (d.shape.hidden, &d.shape.moe);
        entries.push(f32_entry(
            crate::family::gguf_model::Role::FfnGateInp,
            moe_specs.router_values,
        )?);
        entries.push(f32_entry(
            crate::family::gguf_model::Role::FfnGateInpShexp,
            moe_specs.shexp_values,
        )?);
        for (role, experts, rows, k) in [
            (
                crate::family::gguf_model::Role::FfnGateExps,
                m.experts,
                m.inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnUpExps,
                m.experts,
                m.inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnDownExps,
                m.experts,
                h,
                m.inter,
            ),
        ] {
            let mut entry = q4k_stack(experts, rows, k, &mut rng);
            entry.name = table_layers.name(layer, role)?.to_owned();
            entries.push(entry);
        }
        for (role, rows, k) in [
            (
                crate::family::gguf_model::Role::FfnGateShexp,
                m.shared_inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnUpShexp,
                m.shared_inter,
                h,
            ),
            (
                crate::family::gguf_model::Role::FfnDownShexp,
                h,
                m.shared_inter,
            ),
        ] {
            entries.push(FileEntry {
                name: table_layers.name(layer, role)?.to_owned(),
                ty: GgufType::Q4K,
                dims: vec![k as u64, rows as u64],
                payload: q4k_rows(rows, k, &mut rng),
            });
        }
    }
    // The globals: the q4_k embedding (card 0), the full head (sliced per
    // card at load), and the replicated final norm.
    let names = GlobalNames::new(&inv)?;
    let (h, vocab) = (d.shape.hidden, d.shape.vocab);
    entries.push(FileEntry {
        name: names.embed.clone(),
        ty: GgufType::Q4K,
        dims: vec![h as u64, vocab as u64],
        payload: q4k_rows(vocab, h, &mut rng),
    });
    entries.push(FileEntry {
        name: names.head.clone(),
        ty: GgufType::Q4K,
        dims: vec![h as u64, vocab as u64],
        payload: q4k_rows(vocab, h, &mut rng),
    });
    entries.push(FileEntry {
        name: names.norm.clone(),
        ty: GgufType::F32,
        dims: vec![h as u64],
        payload: raw_values(&crate::family::gguf_model::Role::OutputNorm, h)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    });

    let dir = tempfile::tempdir().map_err(|e| EngError::Checkpoint(e.to_string()))?;
    let path = write_gguf(dir.path(), "session.gguf", &entries);
    let weights = GgufWeights::open(
        &path,
        GgufLimits {
            max_header_bytes: 1 << 22,
            max_tensor_bytes: 1 << 30,
        },
        None,
    )?;

    // The stack and the per-card loads, through the production loaders on
    // the cards' own contexts.
    let admission = served_admission();
    let shard = ShardTable::new(&d.shape, 2, 32)?;
    let loads = card_loads(&shard, &inv, GgufType::Q4K)?;
    let payload = chunk_rows
        .map(|rows| {
            super::card_prefill::chunk_mailbox_values(rows, d.shape.hidden, d.shape.moe.top_k)
        })
        .transpose()?;
    let (mut stack, states) =
        CardStack::with_payload(&d, &options, &devices, &admission, &shard, payload)?;
    stack.enable_broadcast_sync_for_test();
    if prepare_async {
        stack.prepare_async_broadcasts_for_test()?;
    }
    let load_options = LoadOptions {
        layout: WeightLayout::Native,
        ..LoadOptions::default()
    };
    let mut owned: Vec<&'static mut crate::family::project::cuda::CudaWeights> = Vec::new();
    for (card, load) in loads.iter().enumerate() {
        let loader = BorrowedLoader::new(&stack.contexts()[card])?;
        let weights = load_card_weights(&loader, &weights, load, card, load_options)?;
        owned.push(Box::leak(Box::new(weights)));
    }
    // The leaked slice of shared refs: one per engine, like the start's
    // leaks. The mut handles convert by consuming them.
    let loaded: &'static [&'static crate::family::project::cuda::CudaWeights] = Box::leak(
        owned
            .into_iter()
            .map(|weights| &*weights)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let stack: &'static CardStack = Box::leak(Box::new(stack));
    let table_static: &'static LayerTable = Box::leak(Box::new(table_layers));
    let eos: Arc<[u32]> = Arc::from(vec![7u32]);
    let mut session = CardSession::new(
        stack,
        states,
        loaded,
        table_static,
        &inv,
        &d,
        &options,
        eos,
        vocab,
        prefix_limits,
        None,
    )?;
    session.set_skip_prefill_head_for_test(false);

    Ok((session, stack, loaded, names, vocab))
}

fn run_session_rig(
    name: &'static str,
    devices: [usize; 2],
    compare_async: bool,
) -> Result<(), EngError> {
    let (mut session, stack, loaded, names, vocab) = build_rig_session(
        devices,
        8,
        crate::family::session::StoreLimits::default(),
        None,
        compare_async,
    )?;
    let limit = session.context_limit();
    if limit < 4 {
        return Err(EngError::Checkpoint(format!(
            "{name}: the fixture context limit {limit} is too small for the rig"
        )));
    }

    // The prompt feeds: no read-back, the fed book advances.
    session.feed(0)?;
    session.feed(1)?;
    let row = session.logits(2)?;
    assert_eq!(session.fed(), 3);
    assert_eq!(row.len(), vocab, "{name}: the head shards rebuild the row");
    assert!(row.iter().all(|v| v.is_finite()), "{name}: finite logits");
    let (best, runner) = top2_of(&row);
    println!(
        "{name}: greedy gap after 3 tokens: {:.3} (runner {runner:.3}, best {best:.3})",
        best - runner
    );

    // The out-of-vocabulary token refuses by name and changes nothing.
    let refused = session
        .logits(vocab as u32)
        .expect_err("out-of-vocab token");
    assert!(
        matches!(refused, EngError::BadRequest(_)),
        "{name}: the wrapper refuses the id, got {refused:?}"
    );
    assert_eq!(session.fed(), 3);

    // Past the limit the session refuses: feed to the limit then one more.
    for token in 3..limit as u32 {
        session.feed(token)?;
    }
    let refused = session
        .feed(limit as u32)
        .expect_err("past the context limit");
    assert!(
        matches!(refused, EngError::BadRequest(_)),
        "{name}: the limit refusal, got {refused:?}"
    );

    // The turn reset: same tokens again, the same logits row bit for bit.
    TokenStepper::reset(&mut session)?;
    assert_eq!(session.fed(), 0);
    session.feed(0)?;
    session.feed(1)?;
    let row_after = session.logits(2)?;
    assert_eq!(
        row, row_after,
        "{name}: the reset run reproduces the logits row exactly"
    );

    // Record a later decode row after another prompt feed. The optimized
    // path must preserve state both before and after a logits-producing step.
    session.feed(3)?;
    let next_row = session.logits(4)?;
    TokenStepper::reset(&mut session)?;
    session.set_skip_prefill_head_for_test(true);
    session.feed(0)?;
    session.feed(1)?;
    let optimized_row = session.logits(2)?;
    assert_eq!(row, optimized_row, "{name}: first decoded logits");
    session.feed(3)?;
    let optimized_next = session.logits(4)?;
    assert_eq!(next_row, optimized_next, "{name}: later decoded logits");

    if compare_async {
        stack.set_async_broadcasts_for_test(true);
        TokenStepper::reset(&mut session)?;
        session.feed(0)?;
        session.feed(1)?;
        let async_row = session.logits(2)?;
        assert_eq!(row, async_row, "{name}: async first decoded logits");
        session.feed(3)?;
        let async_next = session.logits(4)?;
        assert_eq!(next_row, async_next, "{name}: async later decoded logits");
        stack.set_async_broadcasts_for_test(false);
    }

    // A corrupt head makes a logits-producing step fail. A prompt-only feed
    // must still succeed with that head, proving that it was not launched.
    let head = match loaded[0].get(&names.head) {
        Some(crate::family::project::cuda::CudaWeight::Native { weight, .. }) => weight.buffer(),
        _ => {
            return Err(EngError::Checkpoint(
                "test fixture has no native head".into(),
            ));
        }
    };
    let stream = &stack.streams()[0];
    let mut saved = vec![0u8; head.bytes()];
    head.read_at(stream, 0, &mut saved)?;
    head.write_at(stream, 0, &vec![0xff; saved.len()])?;
    TokenStepper::reset(&mut session)?;
    let feed_result = session.feed(0);
    let logits_result = if feed_result.is_ok() {
        Some(session.logits(1))
    } else {
        None
    };
    head.write_at(stream, 0, &saved)?;
    feed_result?;
    assert!(
        matches!(logits_result, Some(Err(EngError::Backend { .. }))),
        "{name}: the decode must still reject a non-finite head"
    );
    Ok(())
}

/// Each session rig holds its CUDA leases for the process lifetime. Run each
/// ignored test alone with a device runner and one test thread.
#[test]
#[ignore = "holds CUDA adapter leases for the process lifetime; run alone"]
fn card_prefix_turn_chain_restores_bitwise() {
    const NAME: &str = "card_prefix_turn_chain_restores_bitwise";
    assert!(device_present(NAME));
    run_card_prefix_turn_chain(NAME, [0, 0]);
}

#[test]
#[ignore = "requires two peer-enabled GPUs and holds their adapter leases for the process lifetime"]
fn card_prefix_turn_chain_restores_bitwise_on_peer_gpus() {
    const NAME: &str = "card_prefix_turn_chain_restores_bitwise_on_peer_gpus";
    assert!(device_present(NAME));
    let devices = [0, 1];
    let facts = crate::kernels::cuda::peer::probe_device_set(&devices).expect("probe GPUs");
    crate::kernels::cuda::peer::check_peer_facts(&facts).expect("peer access");
    crate::kernels::cuda::peer::enable_peer_access(&devices).expect("enable peer access");
    run_card_prefix_turn_chain(NAME, devices);
}

/// Exercise the worker's save/restore sequence and compare its resulting
/// logits and committed state to a run from scratch.
fn run_card_prefix_turn_chain(name: &'static str, devices: [usize; 2]) {
    use crate::family::executor::ExecutorOps;
    use crate::family::session::{DecodeSession, SaveKind, TokenStepper};
    let (mut session, _stack, _loaded, _names, vocab) = build_rig_session(
        devices,
        256,
        crate::family::session::StoreLimits {
            roots: 0,
            entries: 2,
        },
        None,
        false,
    )
    .expect("the rig session");

    // t1 is a strict prefix of t2 (the growing-conversation shape).
    let t1: Vec<u32> = (0..60).map(|i| (i * 7 + 1) % 900).collect();
    let mut t2 = t1.clone();
    t2.extend((0..40).map(|i| (i * 11 + 3) % 900));

    // Turn 1: prefill t1 piecewise and save at the boundary, exactly like
    // the worker's save points.
    let mut st = DecodeSession::default();
    ExecutorOps::reset(&mut session, &mut st).expect("reset 1");
    for piece in t1[..59].chunks(16) {
        ExecutorOps::prefill(&mut session, &mut st, piece).expect("prefill t1");
    }
    let saved = ExecutorOps::save_prefix(&mut session, &st, &t1[..59], SaveKind::Tip)
        .expect("the save call itself");
    assert!(saved, "{name}: the boundary save must succeed");

    // Turn 2 through the restore: same tokens, covered prefix.
    ExecutorOps::reset(&mut session, &mut st).expect("reset 2");
    let covered =
        ExecutorOps::restore_prefix(&mut session, &mut st, &t2).expect("the restore call itself");
    assert_eq!(covered, 59, "{name}: the restore covers t1's saved prefix");
    for piece in t2[covered..].chunks(16) {
        ExecutorOps::prefill(&mut session, &mut st, piece).expect("prefill t2 tail");
    }
    assert_eq!(
        st.pending(),
        t2.last().copied(),
        "{name}: restored pending token"
    );
    assert_eq!(
        TokenStepper::fed(&session),
        t2.len() - 1,
        "{name}: restored prefill count"
    );
    let restored_row = TokenStepper::logits(&mut session, t2[t2.len() - 1]).expect("logits");
    let restored_export = session.export_states_for_test().expect("export restored");

    // Control: the same tokens fed from scratch, no restore.
    ExecutorOps::reset(&mut session, &mut st).expect("reset 3");
    for piece in t2.chunks(16) {
        ExecutorOps::prefill(&mut session, &mut st, piece).expect("prefill control");
    }
    assert_eq!(
        st.pending(),
        t2.last().copied(),
        "{name}: control pending token"
    );
    assert_eq!(
        TokenStepper::fed(&session),
        t2.len() - 1,
        "{name}: control prefill count"
    );
    let control_row = TokenStepper::logits(&mut session, t2[t2.len() - 1]).expect("logits");
    let control_export = session.export_states_for_test().expect("export control");

    assert_eq!(restored_row.len(), vocab, "{name}: full vocabulary row");
    assert!(
        restored_row.iter().all(|v| v.is_finite()),
        "{name}: finite logits"
    );
    assert_eq!(
        restored_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        control_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{name}: the restored logits match bit for bit"
    );
    assert_eq!(
        restored_export.len(),
        control_export.len(),
        "{name}: one export per card"
    );
    assert_eq!(restored_export.len(), 2, "{name}: both card exports exist");
    for (card, (got, want)) in restored_export
        .iter()
        .zip(control_export.iter())
        .enumerate()
    {
        assert_eq!(got.identity, want.identity, "{name}: card {card} identity");
        assert_eq!(got.tokens, want.tokens, "{name}: card {card} token count");
        assert_eq!(
            got.tokens,
            t2.len(),
            "{name}: card {card} consumed every token"
        );
        assert_eq!(got.delta.len(), 1, "{name}: one DeltaNet layer");
        assert_eq!(got.kv.len(), 1, "{name}: one attention layer");
        assert_eq!(
            got.delta.len(),
            want.delta.len(),
            "{name}: card {card} DeltaNet layers"
        );
        for (layer, (got, want)) in got.delta.iter().zip(&want.delta).enumerate() {
            for (field, got, want) in [
                ("recurrent", &got.recurrent, &want.recurrent),
                ("conv", &got.conv, &want.conv),
            ] {
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{name}: card {card} layer {layer} {field} bits"
                );
            }
        }
        assert_eq!(got.kv, want.kv, "{name}: card {card} KV plane bytes");
    }
}

/// The chunk-driver keystone: the card session's per-card weight views
/// construct the single-device chunk machinery's ChunkModel under the CARD
/// geometry - the constructor is where card/single-device compatibility is
/// decided (every weight check runs against the geometry's specs, and the
/// card's views satisfy them by construction). If this holds, the card
/// chunk driver reuses the stage functions unchanged and only adds the
/// per-layer ring sums.
#[test]
fn card_views_construct_chunk_models() {
    const NAME: &str = "card_views_construct_chunk_models";
    if !device_present(NAME) {
        return;
    }
    let _ = NAME;
    let (session, stack, _loaded, _names, _vocab) = build_rig_session(
        [0, 0],
        256,
        crate::family::session::StoreLimits::default(),
        None,
        false,
    )
    .expect("the rig session");
    let spec = rig_spec();
    let inv = Inventory::for_spec(&spec).expect("inventory");
    let shape = HostShape::from_spec(&spec, 64).expect("shape");
    let d = StepDims::new(&shape, &inv).expect("dims");
    let table = LayerTable::new(&d, &inv).expect("layers");
    for card in 0..2 {
        let geometry = MixerGeometry::for_card(&d, stack.table(), card).expect("card geometry");
        let (mixer, moe) = session.chunk_views_for_test(card);
        let weights = super::prefill::ChunkWeights::new(mixer, &moe, session.embed_for_test());
        let inter = stack.table().expert_inter(card).expect("card inter").len();
        let shared_inter = stack
            .table()
            .shared_inter(card)
            .expect("card shared inter")
            .len();
        let model =
            super::prefill::ChunkModel::card(&table, &geometry, &d, weights, inter, shared_inter);
        assert!(
            model.is_ok(),
            "{NAME}: card {card} views must construct the CARD chunk model: {:?}",
            model.err()
        );
        assert_eq!(
            model.expect("checked above").moe_dims().inter,
            inter,
            "{NAME}: card {card} dims carry its inter slice"
        );
    }
}
