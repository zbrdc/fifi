//! Device tests of the prefix store of the Qwen3.6 CUDA executor (decision 0036, addendum
//! 1; packet IMPL-CUDA-SERVE-2, C3).
//!
//! D0 rounds a seeded `ExportedState` through `import_state`/`export_state` on a real state
//! pool in both KV modes and refuses a wrong-mode value (no model load). The model tests load
//! the pinned model with the gate's pattern (`gate/run.rs`, never `start.rs`, which leaks) and
//! drive `ExecutorOps` in the WORKER order (`sched/worker.rs::start`): reset, restore when the
//! job has a boundary, `boundary`, the head in pieces of 16 with the save at the boundary, the
//! tail in pieces of 16, then greedy decode. A restored turn must equal a cold one bitwise:
//! the same ids, the same sha256 of every logits row, the same first-row bytes and the same
//! finish reason. D1/D1c pin the mid-piece restore eager and captured, D2 a chain of restores,
//! D3 the fallback for a prompt that does not extend the key, D4 that a reset leaves no
//! residue, D5 is the Int8G32 twin of D2, and D6 the tiered refusal. An unrelated turn
//! Z = P5[300..380] runs between the save and the restore of D1/D1c and D2/D5, so every
//! restore must overwrite device KV rows that hold different data (a skipped or misplaced KV
//! import cannot pass on the rows the earlier turns happened to leave there), and every
//! capture-on turn asserts its last token really replayed the captured graph.
//!
//! Isolation (the MSG-73 latch rule): the device phase of every test is a
//! `fn ... -> Result<Obs, EngError>` that builds every owner locally, returns plain host data
//! through `?` and drops every owner at its return; only the caller, with no device owner
//! alive, asserts. The seeded values of D0 are finite by construction (they come from small
//! integers), so the float comparison of the round trip is safe; the model tests compare ids,
//! hashes and bytes, never floats. One test per process through the baton.

use super::super::host::HostShape;
use super::book::Health;
use super::dims::{KvMode, StepDims, StepOptions, TieredPlan};
use super::native_device_rig::{Rng, device_present};
use super::pinned::{ROTARY_DIM, head_rows_of};
use super::resources::StepResources;
use super::rules::QwenRules;
use super::start::GGUF_NAME;
use super::state::{
    ResidentState, StateIdentity,
    export::{DeltaExport, ExportedState, KvExport, plane_sizes},
};
use super::state_layout::StateLayout;
use super::step::{CaptureConfig, QwenResidentStep};
use super::tiered_load::{attach_tiered_parts, names_without_experts};
use crate::{
    EngError,
    family::{
        QWEN3_6_35B_A3B,
        executor::{ExecutorOps, GenerationState},
        gguf_model::{Inventory, open_qwen3_6_35b_a3b},
        project::cuda::{CudaProjector, LoadOptions, Reserve, WeightLayout},
        qwen36::settings::tokenizer_pin,
        session::{DecodeSession, SaveKind},
    },
    kernels::cuda::CudaContext,
    tokenizer::{DecodeFinish, IncrementalDecoder, TokenizerBundle},
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The worker's prefill piece (A5).
const PIECE: usize = 16;
/// The greedy decode budget; a reference chain must reach 16 ids (the packet's rule).
const DECODE: u32 = 24;
/// The context of the model tests: 1,024 positions, Native layout, one DeltaNet slot.
const CONTEXT: usize = 1_024;
/// The D0 pool: 64 positions.
const POOL: usize = 64;

fn dims() -> Result<StepDims, EngError> {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    StepDims::new(&shape, &inventory)
}

fn options(kv_mode: KvMode, ctx: usize, tiered: Option<TieredPlan>) -> StepOptions {
    StepOptions {
        kv_mode,
        context_tokens: ctx,
        delta_slots: 1,
        forked_experts: false,
        tiered,
        layout: WeightLayout::Native,
    }
}

/// `ids[..n]` inside a device phase: a fixture shorter than the slice is a returned error,
/// never a panic while a device owner is still alive.
fn head(ids: &[u32], n: usize) -> Result<&[u32], EngError> {
    ids.get(..n)
        .ok_or_else(|| EngError::Checkpoint(format!("the prefix fixture is shorter than {n} ids")))
}

/// `ids[a..b]` inside a device phase, for the same reason.
fn range_of(ids: &[u32], range: std::ops::Range<usize>) -> Result<&[u32], EngError> {
    ids.get(range).ok_or_else(|| {
        EngError::Checkpoint("the prefix fixture is shorter than the slice needs".to_owned())
    })
}

/// P5's prompt ids from the qualification fixtures (the oracle's own input, 612 ids).
fn p5_ids() -> Result<Vec<u32>, EngError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../qualification/cuda-rd/oracle/prompts/P5.ids.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| EngError::Checkpoint(format!("{}: {error}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| EngError::Checkpoint(format!("P5 ids json: {error}")))?;
    let ids = value["ids"]
        .as_array()
        .ok_or_else(|| EngError::Checkpoint("P5 ids json: no ids array".into()))?;
    ids.iter()
        .map(|id| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| EngError::Checkpoint("P5 ids json: a bad id".into()))
        })
        .collect()
}

/// The pinned model's GGUF: the directory from `Q36_CUDA_MODEL_DIR`, the pinned NAME (the name
/// is part of the identity, `start::pinned_gguf`).
fn gguf_path() -> Result<PathBuf, EngError> {
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    Ok(Path::new(&dir).join(GGUF_NAME))
}

fn load_options() -> LoadOptions<'static> {
    LoadOptions {
        reserve: Reserve {
            kv_cache: 0,
            work: 0,
            headroom: 0,
        },
        drop_page_cache: None,
        layout: WeightLayout::Native,
    }
}

/// The rules of the loaded bundle (the gate's seam): eos ids and the sampler vocabulary from
/// the SAME bundle, never the padded head width.
fn rules_for(bundle: &TokenizerBundle) -> Result<QwenRules, EngError> {
    let vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("the vocabulary exceeds usize".into()))?;
    // The eos set is a PROBE CHOICE, not the bundle's: every truncation of P5 greedily emits
    // the bundle's eos after one or two ids (probed at 77, 100, 140, 200, 340 and 560 tokens),
    // so no turn would reach the 16-id rule, and only the full 612-token prompt continues (the
    // G1 health test). An empty set is refused by `QwenRules::new`, and 612-token prompts push
    // the debug runs past the 280-s window. Id 0 is never the greedy argmax of these chains
    // (the passing turns are the probe), so every finish is `Length` at the token limit and
    // the bitwise comparison carries 24 rows instead of one.
    QwenRules::new(vec![0], vocab, head_rows_of()?)
}

/// One turn's observations, all plain host data (asserted only after every owner dropped).
#[derive(Debug)]
struct Turn {
    covered: usize,
    saved: Option<bool>,
    /// history/pending/committed rows observed right after the restore (D1's helper check).
    restored: Option<Restored>,
    key_len: Option<usize>,
    ids: Vec<u32>,
    row_hashes: Vec<Vec<u8>>,
    first_row: Vec<u8>,
    finish: Option<DecodeFinish>,
    /// Whether the turn's LAST token replayed the captured graph (`RunReport::capture`): the
    /// capture-on tests assert it, so a refused capture that fell back to eager fails them.
    captured: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Restored {
    history: Vec<u32>,
    pending: Option<u32>,
    tokens: usize,
}

/// One reset-plus-restore attempt (the worker's start without a prefill).
#[derive(Debug)]
struct Probe {
    covered: usize,
    history: Vec<u32>,
    pending: Option<u32>,
    tokens: usize,
    key_len: Option<usize>,
    no_export: bool,
    health: Health,
}

/// One turn through `ExecutorOps` in the worker order (`sched/worker.rs::start`): reset, the
/// restore when the job has a boundary, `boundary`, the head in pieces of 16 with the save at
/// the boundary, the tail in pieces of 16, then greedy decode until a finish. The sha256 of
/// `row_bytes()` is taken after every decode; the first row's bytes are kept whole.
fn turn(
    step: &mut QwenResidentStep<'_>,
    session: &mut DecodeSession,
    bundle: &TokenizerBundle,
    prompt: &[u32],
    boundary: Option<usize>,
) -> Result<Turn, EngError> {
    ExecutorOps::reset(step, session)?;
    let covered = match boundary {
        Some(_) => ExecutorOps::restore_prefix(step, session, prompt)?,
        None => 0,
    };
    if covered >= prompt.len() {
        return Err(EngError::BadRequest(
            "restored prefix covers the whole prompt".to_owned(),
        ));
    }
    let restored = (covered > 0).then(|| Restored {
        history: session.history().to_vec(),
        pending: session.pending(),
        tokens: step.tokens(),
    });
    ExecutorOps::boundary(step, session);
    let mut next = covered;
    let mut saved = None;
    if let Some(boundary) = boundary.filter(|&boundary| boundary > covered) {
        for piece in prompt[covered..boundary].chunks(PIECE) {
            ExecutorOps::prefill(step, session, piece)?;
        }
        saved = Some(ExecutorOps::save_prefix(
            step,
            session,
            &prompt[..boundary],
            SaveKind::Tip,
        )?);
        next = boundary;
    }
    for piece in prompt[next..].chunks(PIECE) {
        ExecutorOps::prefill(step, session, piece)?;
    }
    let mut generation =
        GenerationState::new(IncrementalDecoder::new(bundle.tokenizer())).with_max_tokens(DECODE);
    let mut ids = Vec::new();
    let mut row_hashes = Vec::new();
    let mut first_row = Vec::new();
    // The first iteration is an explicit flag, not `first_row.is_empty()`: the row's emptiness
    // is a property of the row, not of the iteration.
    let mut first = true;
    let finish;
    loop {
        let batch = ExecutorOps::decode(step, session, &mut generation)?;
        ids.extend_from_slice(&batch.tokens);
        row_hashes.push(Sha256::digest(step.row_bytes()).to_vec());
        if first {
            first_row = step.row_bytes().to_vec();
            first = false;
        }
        if let Some(reason) = batch.finish {
            finish = Some(reason);
            break;
        }
    }
    Ok(Turn {
        covered,
        saved,
        restored,
        key_len: step.prefix.key_len(),
        ids,
        row_hashes,
        first_row,
        finish,
        captured: step.last_report().capture.captured(),
    })
}

/// Reset, then one restore attempt, with the session and slot observed after it.
fn restore_probe(
    step: &mut QwenResidentStep<'_>,
    session: &mut DecodeSession,
    prompt: &[u32],
) -> Result<Probe, EngError> {
    ExecutorOps::reset(step, session)?;
    let sequence = step.stream_for_tests().enqueued();
    let covered = ExecutorOps::restore_prefix(step, session, prompt)?;
    Ok(Probe {
        covered,
        history: session.history().to_vec(),
        pending: session.pending(),
        tokens: step.tokens(),
        key_len: step.prefix.key_len(),
        no_export: step.stream_for_tests().enqueued() == sequence,
        health: step.state_for_tests().health(),
    })
}

/// Bitwise equality of two turns (the ids, every row hash, the first row and the finish).
fn assert_same_bits(what: &str, got: &Turn, want: &Turn) {
    assert_eq!(got.ids, want.ids, "{what}: the ids differ");
    assert_eq!(
        got.row_hashes, want.row_hashes,
        "{what}: a logits row hash differs"
    );
    assert_eq!(
        got.first_row, want.first_row,
        "{what}: the first row bytes differ"
    );
    assert_eq!(got.finish, want.finish, "{what}: the finish differs");
}

/// The packet's chain rule: every reference turn reaches 16 ids.
fn assert_chain_length(what: &str, got: &Turn) {
    assert!(
        got.ids.len() >= PIECE,
        "{what}: the turn made only {} ids",
        got.ids.len()
    );
}

/// MINOR-2: a turn that ran with capture on really replayed the captured graph; a capture
/// refused to eager (`RefusedFellBackToEager`) fails the test instead of passing silently.
fn assert_captured(what: &str, got: &Turn) {
    assert!(
        got.captured,
        "{what}: the turn's last token did not replay the captured graph"
    );
}

/// A seeded export for `layout` at `tokens`: finite f32 DeltaNet values (small integers over
/// 64, so the round-trip comparison of the floats is safe) and arbitrary plane bytes.
fn seeded_export(
    layout: &StateLayout,
    identity: &StateIdentity,
    tokens: usize,
    rng: &mut Rng,
) -> Result<ExportedState, EngError> {
    let mut f32s = |n: usize| {
        (0..n)
            .map(|_| (rng.next_u64() % 8192) as f32 / 64.0)
            .collect()
    };
    let delta = (0..layout.delta.layers)
        .map(|_| DeltaExport {
            recurrent: f32s(layout.delta.rec_values),
            conv: f32s(layout.delta.conv_values),
        })
        .collect();
    // D0 calls this with a live `ResidentState`, so a size error is returned, never unwrapped.
    let kv = (0..layout.kv.layers)
        .map(|ordinal| {
            let planes = plane_sizes(&layout.kv, ordinal, tokens)?
                .into_iter()
                .map(|n| (0..n).map(|_| rng.next_u64() as u8).collect())
                .collect();
            Ok(KvExport { planes })
        })
        .collect::<Result<_, EngError>>()?;
    Ok(ExportedState {
        identity: identity.clone(),
        tokens,
        delta,
        kv,
    })
}

struct D0Round {
    mode: KvMode,
    tokens: usize,
    round_trips: bool,
    committed: bool,
    valid: bool,
}

struct D0Refusal {
    mode: KvMode,
    checkpoint: bool,
    committed: usize,
    valid: bool,
    unchanged: bool,
}

/// D0's device phase: one `ResidentState` per mode, seeded imports at {1, 37, 64} positions
/// re-exported and compared, then a wrong-`kv_mode` import refused.
fn d0_round_trips() -> Result<(Vec<D0Round>, Vec<D0Refusal>), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let d = dims()?;
    let mut rounds = Vec::new();
    let mut refusals = Vec::new();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let opts = options(mode, POOL, None);
        let layout = StateLayout::new(&d, &opts)?;
        let identity = StateIdentity::new(&d, mode);
        let mut state = ResidentState::allocate(&ctx, &stream, &d, &opts)?;
        let mut rng = Rng::new(match mode {
            KvMode::Fp16 => 0xD0,
            KvMode::Int8G32 => 0xD5,
        });
        let mut last = None;
        for tokens in [1_usize, 37, 64] {
            let src = seeded_export(&layout, &identity, tokens, &mut rng)?;
            state.import_state(&stream, &src)?;
            let out = state.export_state(&stream)?;
            rounds.push(D0Round {
                mode,
                tokens,
                round_trips: out == src,
                committed: state.tokens() == tokens,
                valid: state.health() == Health::Valid,
            });
            last = Some(out);
        }
        // A value of the OTHER kv_mode: refused as Checkpoint by the identity, before any
        // device write; the health, the committed rows and the next export stay unchanged.
        let other = match mode {
            KvMode::Fp16 => KvMode::Int8G32,
            KvMode::Int8G32 => KvMode::Fp16,
        };
        let wrong = seeded_export(&layout, &StateIdentity::new(&d, other), 37, &mut rng)?;
        let refused = state.import_state(&stream, &wrong);
        let next = state.export_state(&stream)?;
        refusals.push(D0Refusal {
            mode,
            checkpoint: matches!(refused, Err(EngError::Checkpoint(_))),
            committed: state.tokens(),
            valid: state.health() == Health::Valid,
            unchanged: Some(next) == last,
        });
    }
    Ok((rounds, refusals))
}

#[test]
fn export_import_round_trips_on_device() {
    if !device_present("export_import_round_trips_on_device") {
        return;
    }
    let (rounds, refusals) =
        d0_round_trips().unwrap_or_else(|error| panic!("D0 device phase: {error:?}"));
    assert_eq!(rounds.len(), 6, "three depths in each mode");
    for round in &rounds {
        assert!(round.round_trips, "{:?}", round.mode);
        assert!(round.committed, "{:?} at {}", round.mode, round.tokens);
        assert!(round.valid, "{:?}", round.mode);
    }
    assert_eq!(refusals.len(), 2, "one refusal per mode");
    for refusal in &refusals {
        assert!(refusal.checkpoint, "{:?}", refusal.mode);
        assert_eq!(refusal.committed, 64, "the rows are unchanged");
        assert!(refusal.valid, "{:?}", refusal.mode);
        assert!(refusal.unchanged, "{:?}", refusal.mode);
    }
}

/// D1/D1c's device phase: T = P5[..77] with the boundary 37 (37 = 2*16 + 5, so the key ends
/// inside a worker piece). A cold turn with the save, the unrelated turn Z between the save
/// and the restore (so the KV rows the restore overwrites hold different data), the restoring
/// turn, and the boundary-less reference.
fn mid_piece_phase(capture: bool) -> Result<(Vec<u32>, Vec<Turn>), EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    let rules = rules_for(&bundle)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let device = projector.load_model(&model, load_options())?.device;
    let opts = options(KvMode::Fp16, CONTEXT, None);
    let res = StepResources::new(
        projector.context(),
        dims()?,
        &Inventory::for_spec(&QWEN3_6_35B_A3B)?,
        &opts,
    )?;
    let mut step = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture,
        section_marks: false,
    })?;
    let mut session = DecodeSession::default();
    let p5 = p5_ids()?;
    let prompt = head(&p5, 77)?.to_vec();
    // The unrelated turn Z = P5[300..380] runs between the save and the restore (MINOR-1): the
    // restore must overwrite KV rows that hold Z's data, not the rows T's own cold turn left.
    let z = range_of(&p5, 300..380)?.to_vec();
    let turns = vec![
        turn(&mut step, &mut session, &bundle, &prompt, Some(37))?,
        turn(&mut step, &mut session, &bundle, &z, None)?,
        turn(&mut step, &mut session, &bundle, &prompt, Some(37))?,
        turn(&mut step, &mut session, &bundle, &prompt, None)?,
    ];
    Ok((prompt, turns))
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn restore_mid_piece_is_bitwise_eager() {
    if !device_present("restore_mid_piece_is_bitwise_eager") {
        return;
    }
    let (prompt, turns) =
        mid_piece_phase(false).unwrap_or_else(|e| panic!("D1 device phase: {e:?}"));
    let (cold, z, restored, reference) = (&turns[0], &turns[1], &turns[2], &turns[3]);
    assert_eq!(cold.covered, 0, "the cold turn restores nothing");
    assert_eq!(
        cold.saved,
        Some(true),
        "the cold turn saves at the boundary"
    );
    assert_eq!(cold.key_len, Some(37), "the entry holds the boundary key");
    assert_eq!(z.covered, 0, "Z does not extend the key");
    assert_eq!(z.saved, None, "Z has no boundary, so it never saves");
    assert_eq!(
        z.key_len,
        Some(37),
        "the unrelated turn keeps the entry (A8)"
    );
    assert_eq!(restored.covered, 37);
    assert_eq!(
        restored.saved, None,
        "the worker skips the save when the boundary is already restored (A5)"
    );
    assert_eq!(
        restored.key_len,
        Some(37),
        "the refused-save restore leaves the entry in place"
    );
    let after = restored
        .restored
        .as_ref()
        .expect("the restore was observed");
    assert_eq!(after.history, prompt[..37], "the history is the key");
    assert_eq!(
        after.pending,
        Some(prompt[36]),
        "the pending token is the key's last"
    );
    assert_eq!(after.tokens, 36, "the import committed key.len() - 1 rows");
    assert_chain_length("cold", cold);
    assert_chain_length("Z", z);
    assert_chain_length("restored", restored);
    assert_chain_length("reference", reference);
    assert_same_bits("a save does not perturb the turn", cold, reference);
    assert_same_bits("the restored turn is bitwise eager", restored, reference);
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn restore_mid_piece_is_bitwise_captured() {
    if !device_present("restore_mid_piece_is_bitwise_captured") {
        return;
    }
    let (prompt, turns) =
        mid_piece_phase(true).unwrap_or_else(|e| panic!("D1c device phase: {e:?}"));
    let (cold, z, restored, reference) = (&turns[0], &turns[1], &turns[2], &turns[3]);
    assert_eq!(cold.covered, 0);
    assert_eq!(cold.saved, Some(true));
    assert_eq!(z.covered, 0);
    assert_eq!(z.saved, None);
    assert_eq!(restored.covered, 37);
    assert_eq!(restored.key_len, Some(37));
    let after = restored
        .restored
        .as_ref()
        .expect("the restore was observed");
    assert_eq!(after.history, prompt[..37]);
    assert_eq!(after.pending, Some(prompt[36]));
    assert_eq!(after.tokens, 36);
    for (what, got) in [("cold", cold), ("Z", z), ("restored", restored)] {
        assert_captured(what, got);
        assert_chain_length(what, got);
    }
    assert_captured("reference", reference);
    assert_chain_length("reference", reference);
    assert_same_bits("a save does not perturb the captured turn", cold, reference);
    assert_same_bits("the captured restore replays bitwise", restored, reference);
}

/// D2/D5's device phase: T1 = P5[..340] with the boundary 300, then T2 = P5[..560] with the
/// boundary 520: a first restore of T1's entry, a save taken after that restore, a second
/// restore of the new entry, and the boundary-less reference of T2. The unrelated turn Z =
/// P5[300..380] runs between T1's save and the first restore, so that restore must overwrite
/// foreign KV rows, not the rows T1's own prefill left there.
fn chain_phase(kv_mode: KvMode) -> Result<Vec<Turn>, EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    let rules = rules_for(&bundle)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let device = projector.load_model(&model, load_options())?.device;
    let opts = options(kv_mode, CONTEXT, None);
    let res = StepResources::new(
        projector.context(),
        dims()?,
        &Inventory::for_spec(&QWEN3_6_35B_A3B)?,
        &opts,
    )?;
    let mut step = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: true,
        section_marks: false,
    })?;
    let mut session = DecodeSession::default();
    let p5 = p5_ids()?;
    let t1: Vec<u32> = head(&p5, 340)?.to_vec();
    let t2: Vec<u32> = head(&p5, 560)?.to_vec();
    let z: Vec<u32> = range_of(&p5, 300..380)?.to_vec();
    let turns = vec![
        turn(&mut step, &mut session, &bundle, &t1, Some(300))?,
        turn(&mut step, &mut session, &bundle, &z, None)?,
        turn(&mut step, &mut session, &bundle, &t2, Some(520))?,
        turn(&mut step, &mut session, &bundle, &t2, Some(520))?,
        turn(&mut step, &mut session, &bundle, &t2, None)?,
    ];
    Ok(turns)
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn a_turn_chain_restores_a_restored_entry() {
    if !device_present("a_turn_chain_restores_a_restored_entry") {
        return;
    }
    let turns = chain_phase(KvMode::Fp16).unwrap_or_else(|e| panic!("D2 device phase: {e:?}"));
    let (t1, z, first, again, reference) = (&turns[0], &turns[1], &turns[2], &turns[3], &turns[4]);
    assert_eq!(t1.covered, 0);
    assert_eq!(t1.saved, Some(true), "T1 saves at 300");
    assert_eq!(z.covered, 0, "Z does not extend T1's key");
    assert_eq!(z.saved, None, "Z has no boundary, so it never saves");
    assert_eq!(
        z.key_len,
        Some(300),
        "the unrelated turn keeps T1's entry (A8)"
    );
    assert_eq!(first.covered, 300, "T2 restores T1's entry");
    assert_eq!(
        first.saved,
        Some(true),
        "a save after a restore replaces the entry"
    );
    assert_eq!(again.covered, 520, "the restored entry restores in turn");
    assert_eq!(again.key_len, Some(520));
    for (what, got) in [
        ("T1", t1),
        ("Z", z),
        ("T2 first", first),
        ("T2 again", again),
    ] {
        assert_chain_length(what, got);
        assert_captured(what, got);
    }
    assert_chain_length("reference", reference);
    assert_captured("reference", reference);
    assert_same_bits("the first restored T2 turn", first, reference);
    assert_same_bits("the twice-restored T2 turn", again, reference);
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn int8g32_turn_chain_restores_bitwise() {
    if !device_present("int8g32_turn_chain_restores_bitwise") {
        return;
    }
    let turns = chain_phase(KvMode::Int8G32).unwrap_or_else(|e| panic!("D5 device phase: {e:?}"));
    let (t1, z, first, again, reference) = (&turns[0], &turns[1], &turns[2], &turns[3], &turns[4]);
    assert_eq!(t1.covered, 0);
    assert_eq!(z.covered, 0);
    assert_eq!(first.covered, 300);
    assert_eq!(again.covered, 520);
    for (what, got) in [
        ("T1", t1),
        ("Z", z),
        ("T2 first", first),
        ("T2 again", again),
    ] {
        assert_chain_length(what, got);
        assert_captured(what, got);
    }
    assert_chain_length("reference", reference);
    assert_captured("reference", reference);
    assert_same_bits("the first restored Int8G32 T2 turn", first, reference);
    assert_same_bits("the twice-restored Int8G32 T2 turn", again, reference);
}

/// D3's device phase. K = P5[..100] is saved by the P5[..140] turn; the equal, the shorter and
/// the divergent prompts each refuse the restore with the session reset and the entry kept.
/// The divergent prompt then takes its full worker turn (the refused restore falls back to a
/// full prefill, and the save at its boundary replaces the entry), which must equal its
/// boundary-less reference bitwise; the K turn runs once more, and the final probe restores
/// the fresh K. (A divergent worker turn cannot avoid the save: the worker saves at every
/// boundary it reaches, so K is re-established the same way it was made.)
fn fallback_phase() -> Result<FallbackObs, EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    let rules = rules_for(&bundle)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let device = projector.load_model(&model, load_options())?.device;
    let opts = options(KvMode::Fp16, CONTEXT, None);
    let res = StepResources::new(
        projector.context(),
        dims()?,
        &Inventory::for_spec(&QWEN3_6_35B_A3B)?,
        &opts,
    )?;
    let mut step = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    let mut session = DecodeSession::default();
    let p5 = p5_ids()?;
    let k: Vec<u32> = head(&p5, 140)?.to_vec();
    let divergent: Vec<u32> = head(&p5, 60)?
        .iter()
        .chain(range_of(&p5, 200..260)?)
        .copied()
        .collect();
    let save = turn(&mut step, &mut session, &bundle, &k, Some(100))?;
    let equal = restore_probe(&mut step, &mut session, head(&p5, 100)?)?;
    let shorter = restore_probe(&mut step, &mut session, head(&p5, 80)?)?;
    let refused = restore_probe(&mut step, &mut session, &divergent)?;
    let full = turn(&mut step, &mut session, &bundle, &divergent, Some(100))?;
    let reference = turn(&mut step, &mut session, &bundle, &divergent, None)?;
    let resave = turn(&mut step, &mut session, &bundle, &k, Some(100))?;
    let final_probe = restore_probe(&mut step, &mut session, &k)?;
    Ok(FallbackObs {
        save,
        equal,
        shorter,
        refused,
        full,
        reference,
        resave,
        final_probe,
    })
}

struct FallbackObs {
    save: Turn,
    equal: Probe,
    shorter: Probe,
    refused: Probe,
    full: Turn,
    reference: Turn,
    resave: Turn,
    final_probe: Probe,
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn a_prompt_that_does_not_extend_the_key_falls_back() {
    if !device_present("a_prompt_that_does_not_extend_the_key_falls_back") {
        return;
    }
    let obs = fallback_phase().unwrap_or_else(|e| panic!("D3 device phase: {e:?}"));
    assert_eq!(obs.save.covered, 0);
    assert_eq!(obs.save.saved, Some(true), "K = P5[..100] is saved");
    for (what, probe) in [
        ("equal", &obs.equal),
        ("shorter", &obs.shorter),
        ("divergent", &obs.refused),
    ] {
        assert_eq!(probe.covered, 0, "{what}: the restore is refused");
        assert_eq!(probe.tokens, 0, "{what}: the step is at zero rows");
        assert!(probe.history.is_empty(), "{what}: the session is reset");
        assert_eq!(probe.pending, None, "{what}: no pending token");
        assert_eq!(probe.key_len, Some(100), "{what}: the entry stays");
    }
    assert_chain_length("the divergent full turn", &obs.full);
    assert_chain_length("the divergent reference", &obs.reference);
    assert_same_bits(
        "the refused restore falls back to the full prefill",
        &obs.full,
        &obs.reference,
    );
    assert_eq!(
        obs.resave.covered, 0,
        "the divergent turn's entry does not cover P5[..140]"
    );
    assert_eq!(obs.resave.saved, Some(true), "K is re-established");
    assert_eq!(obs.final_probe.covered, 100, "P5[..140] still restores 100");
    assert_eq!(obs.final_probe.key_len, Some(100));
}

/// D4's device phase: an unrelated prompt Z, its reference, a save, a restore plus a 10-token
/// prefill, then a reset and Z again — bitwise equal — with the entry still alive.
fn residue_phase() -> Result<ResidueObs, EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    let rules = rules_for(&bundle)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let device = projector.load_model(&model, load_options())?.device;
    let opts = options(KvMode::Fp16, CONTEXT, None);
    let res = StepResources::new(
        projector.context(),
        dims()?,
        &Inventory::for_spec(&QWEN3_6_35B_A3B)?,
        &opts,
    )?;
    let mut step = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: true,
        section_marks: false,
    })?;
    let mut session = DecodeSession::default();
    let p5 = p5_ids()?;
    let z: Vec<u32> = range_of(&p5, 300..380)?.to_vec();
    let k: Vec<u32> = head(&p5, 140)?.to_vec();
    let reference = turn(&mut step, &mut session, &bundle, &z, None)?;
    let save = turn(&mut step, &mut session, &bundle, &k, Some(100))?;
    // The restore plus ten prefilled tokens, never decoded.
    ExecutorOps::reset(&mut step, &mut session)?;
    let covered = ExecutorOps::restore_prefix(&mut step, &mut session, &k)?;
    ExecutorOps::boundary(&mut step, &mut session);
    ExecutorOps::prefill(&mut step, &mut session, &k[100..110])?;
    let mid = Restored {
        history: session.history().to_vec(),
        pending: session.pending(),
        tokens: step.tokens(),
    };
    ExecutorOps::reset(&mut step, &mut session)?;
    let after = turn(&mut step, &mut session, &bundle, &z, None)?;
    let final_probe = restore_probe(&mut step, &mut session, &k)?;
    Ok(ResidueObs {
        reference,
        save,
        covered,
        mid,
        after,
        final_probe,
    })
}

struct ResidueObs {
    reference: Turn,
    save: Turn,
    covered: usize,
    mid: Restored,
    after: Turn,
    final_probe: Probe,
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn reset_after_restore_leaves_no_residue() {
    if !device_present("reset_after_restore_leaves_no_residue") {
        return;
    }
    let obs = residue_phase().unwrap_or_else(|e| panic!("D4 device phase: {e:?}"));
    assert_eq!(obs.save.covered, 0);
    assert_eq!(obs.save.saved, Some(true), "K = P5[..100] is saved");
    assert_eq!(obs.covered, 100, "the mid-turn restore covers the key");
    assert_eq!(
        obs.mid.history.len(),
        110,
        "ten prefilled tokens extend the hundred-token key (the last stays pending)"
    );
    assert_eq!(obs.mid.tokens, 109, "the step holds the prefilled rows");
    assert_chain_length("the reference Z turn", &obs.reference);
    assert_captured("the reference Z turn", &obs.reference);
    assert_captured("the save turn", &obs.save);
    assert_chain_length("the Z turn after the reset", &obs.after);
    assert_captured("the Z turn after the reset", &obs.after);
    assert_same_bits("a reset leaves no residue", &obs.after, &obs.reference);
    assert_eq!(
        obs.final_probe.covered, 100,
        "the reset keeps the entry (A8)"
    );
    assert_eq!(obs.final_probe.key_len, Some(100));
}

/// D6's device phase: a tiered step of the pinned model (the banks preload the lowest experts),
/// whose save must refuse with `Ok(false)` and leave the slot empty.
fn tiered_phase() -> Result<Turn, EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    let rules = rules_for(&bundle)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let names = names_without_experts(&model.plan().device_names());
    let device = projector.load(&model.weights, &names, load_options())?;
    let mut plan = TieredPlan::new(40, 4);
    plan.preload = true;
    let opts = options(KvMode::Fp16, CONTEXT, Some(plan));
    let res = StepResources::new(
        projector.context(),
        dims()?,
        &Inventory::for_spec(&QWEN3_6_35B_A3B)?,
        &opts,
    )?;
    let mut step = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    attach_tiered_parts(
        &mut step,
        projector.context(),
        ((model.weights.header(), gguf.as_path()), &res.dims),
        &plan,
    )?;
    let mut session = DecodeSession::default();
    let prompt = head(&p5_ids()?, 45)?.to_vec();
    turn(&mut step, &mut session, &bundle, &prompt, Some(37))
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model"]
fn a_tiered_step_never_saves() {
    if !device_present("a_tiered_step_never_saves") {
        return;
    }
    let turn = tiered_phase().unwrap_or_else(|e| panic!("D6 device phase: {e:?}"));
    assert_eq!(turn.covered, 0, "the empty slot restores nothing");
    assert_eq!(
        turn.saved,
        Some(false),
        "a tiered step answers Ok(false): it cannot save"
    );
    assert_eq!(turn.key_len, None, "the slot stays empty");
}

#[derive(Debug)]
struct PinnedSaveObs {
    accepted: bool,
    before_publish: usize,
    ready: usize,
    pinned_ready: usize,
    no_export: bool,
    pinned_bytes: usize,
}

#[derive(Debug)]
struct PinnedReadyObs {
    mode: KvMode,
    capacity: usize,
    checkpoint: usize,
    pinned_before: usize,
    pinned_enabled: usize,
    initial_ready: usize,
    cold: Probe,
    saves: Vec<PinnedSaveObs>,
    late_refused: bool,
    late_unchanged: bool,
    restored: Probe,
    released: bool,
}

/// Observe the real adapter, not a separately constructed pool or a planned-entry count.
fn pinned_ready_phase() -> Result<(Vec<u32>, Vec<PinnedReadyObs>), EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    // The old P5 oracle asset is not required for this capacity probe.
    let ids = bundle.tokenizer().encode(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
    )?;
    head(&ids, 13)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let device = projector.load_model(&model, load_options())?.device;
    let ctx = projector.context();
    let mut observations = Vec::new();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let opts = options(mode, CONTEXT, None);
        let res = StepResources::new(ctx, dims()?, &Inventory::for_spec(&QWEN3_6_35B_A3B)?, &opts)?;
        let before_step = ctx.live_owners();
        let mut step = QwenResidentStep::new(ctx, &res, &device, &opts, rules_for(&bundle)?)?;
        let layout = step.state_layout();
        let checkpoint = layout
            .delta
            .total
            .checked_div(layout.delta.slots)
            .and_then(|bytes| bytes.checked_add(layout.kv.total))
            .ok_or_else(|| EngError::Checkpoint("pinned-ready layout size".into()))?;
        let pinned_before = ctx.live_owners().pinned;
        step.enable_pinned_prefix()?;
        let pinned_enabled = ctx.live_owners().pinned;
        let capacity = step.prefix.limits().entries;
        let initial_ready = step.prefix.entries_len() + step.prefix.roots_len();
        let mut session = DecodeSession::default();
        let cold = restore_probe(&mut step, &mut session, head(&ids, 13)?)?;
        ExecutorOps::boundary(&mut step, &mut session);
        let mut saves = Vec::new();
        for (start, end, kind) in [(0, 5, SaveKind::Block), (5, 9, SaveKind::Tip)] {
            ExecutorOps::prefill(&mut step, &mut session, range_of(&ids, start..end)?)?;
            let sequence = step.stream_for_tests().enqueued();
            let accepted = ExecutorOps::save_prefix(&mut step, &session, head(&ids, end)?, kind)?;
            let before_publish = step.prefix.entries_len() + step.prefix.roots_len();
            ExecutorOps::after_emit(&mut step)?;
            saves.push(PinnedSaveObs {
                accepted,
                before_publish,
                ready: step.prefix.entries_len() + step.prefix.roots_len(),
                pinned_ready: step
                    .prefix
                    .snapshots()
                    .filter(|snapshot| {
                        matches!(snapshot, super::prefix::CudaSavedPrefix::Pinned(_))
                    })
                    .count(),
                no_export: step.stream_for_tests().enqueued() == sequence,
                pinned_bytes: ctx.live_owners().pinned,
            });
        }
        let late_owners = ctx.live_owners();
        let late_entries = step.prefix.entries_len() + step.prefix.roots_len();
        let late_sequence = step.stream_for_tests().enqueued();
        let late_refused = match step.enable_pinned_prefix() {
            Err(EngError::BadRequest(_)) => true,
            Ok(()) => false,
            Err(error) => return Err(error),
        };
        let late_unchanged = ctx.live_owners() == late_owners
            && step.prefix.entries_len() + step.prefix.roots_len() == late_entries
            && step.stream_for_tests().enqueued() == late_sequence;
        let restored = restore_probe(&mut step, &mut session, head(&ids, 13)?)?;
        drop(step);
        let released = ctx.live_owners() == before_step;
        observations.push(PinnedReadyObs {
            mode,
            capacity,
            checkpoint,
            pinned_before,
            pinned_enabled,
            initial_ready,
            cold,
            saves,
            late_refused,
            late_unchanged,
            restored,
            released,
        });
    }
    Ok((ids, observations))
}

fn assert_empty_probe(probe: &Probe) {
    assert_eq!(probe.covered, 0);
    assert!(probe.history.is_empty());
    assert_eq!(probe.pending, None);
    assert_eq!(probe.tokens, 0);
    assert_eq!(probe.key_len, None);
    assert!(probe.no_export);
    assert_eq!(probe.health, Health::Valid);
}

fn assert_pinned_ready(expected: usize) {
    let (ids, observations) =
        pinned_ready_phase().unwrap_or_else(|error| panic!("pinned-ready device phase: {error:?}"));
    assert_eq!(observations.len(), 2);
    for (obs, mode) in observations.iter().zip([KvMode::Fp16, KvMode::Int8G32]) {
        eprintln!("PINNED_READY_NATIVE expected={expected} {obs:?}");
        assert_eq!(obs.mode, mode);
        assert_eq!(obs.capacity, expected);
        let charge = if expected == 0 {
            0
        } else {
            (expected + 1) * obs.checkpoint
        };
        assert_eq!(
            obs.pinned_enabled.checked_sub(obs.pinned_before),
            Some(charge)
        );
        assert_eq!(obs.initial_ready, 0);
        assert_empty_probe(&obs.cold);
        assert_eq!(obs.saves.len(), 2);
        for (index, save) in obs.saves.iter().enumerate() {
            let ready = (index + 1).min(expected);
            assert_eq!(save.accepted, expected > 0);
            assert!(save.before_publish <= index);
            assert_eq!(save.ready, ready);
            assert_eq!(save.pinned_ready, ready);
            assert_eq!(save.no_export, expected == 0);
            assert_eq!(save.pinned_bytes, obs.pinned_enabled);
        }
        assert!(obs.late_refused);
        assert!(obs.late_unchanged);
        if expected == 0 {
            assert_empty_probe(&obs.restored);
        } else {
            assert_eq!(obs.restored.covered, 9);
            assert_eq!(obs.restored.history, ids[..9]);
            assert_eq!(obs.restored.pending, Some(ids[8]));
            assert_eq!(obs.restored.tokens, 8);
            assert_eq!(obs.restored.health, Health::Valid);
            assert!(!obs.restored.no_export);
        }
        assert!(obs.released, "dropping the step releases its owners");
    }
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model; zero-byte pinned cap"]
fn pinned_ready_zero() {
    assert_pinned_ready(0);
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model; one retained pinned entry"]
fn pinned_ready_one() {
    assert_pinned_ready(1);
}

#[test]
#[ignore = "the 3090 through the baton: loads the real model; two retained pinned entries"]
fn pinned_ready_two() {
    assert_pinned_ready(2);
}
