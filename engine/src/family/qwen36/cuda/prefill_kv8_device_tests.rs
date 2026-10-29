//! Structural KV8 coverage of the five-layer synthetic chunk body, not KV8 acceptance.
//!
//! The reference is sixteen one-row CHUNK bodies, not production decode. The existing
//! `QwenResidentStep::new` requires complete `CudaWeights`, every table layer and global
//! norm/head weights. `Rig` supplies only five layers and an embedding. No constructor or
//! production option is widened here. Cross-batching differences are diagnostics only;
//! no kernel-local E3 bound is transferred to the whole mixed body.

use super::*;
use crate::family::qwen36::cuda::{prefill_plan::ChunkSlot, state_layout::Plane};

const M: usize = 16;
const M_SESS: usize = 128;
const PARAM_CANARY: u8 = 0xA5;

/// Host-only data. Each DeltaNet entry holds recurrent then convolution bytes; each KV entry
/// holds K codes, K scales, V codes and V scales, in native full-capacity head-major order.
struct Snapshot {
    hidden: Vec<u8>,
    delta: Vec<[Vec<u8>; 2]>,
    kv: Vec<[Vec<u8>; 4]>,
    params: Vec<u8>,
    statuses: Vec<i32>,
    tokens: usize,
}

fn read_range(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0; range.bytes()];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes)
}

fn checked_view<'a>(
    tensors: &[(ChunkSlot, ArenaTensor)],
    arena: &'a DeviceArena,
    slot: ChunkSlot,
) -> Result<DeviceRange<'a>, EngError> {
    ChunkViews::new(tensors, arena)?.get(slot)
}

fn snapshot(
    rig: &Rig,
    state: &ResidentState,
    hidden: Vec<u8>,
    statuses: Vec<i32>,
) -> Result<Snapshot, EngError> {
    let mut delta = Vec::new();
    for ordinal in 0..state.layout().delta.layers {
        delta.push([
            read_range(&rig.stream, state.delta_rec(0, ordinal)?)?,
            read_range(&rig.stream, state.delta_conv(0, ordinal)?)?,
        ]);
    }
    let mut kv = Vec::new();
    for ordinal in 0..state.layout().kv.layers {
        kv.push([
            read_range(&rig.stream, state.kv_plane(ordinal, Plane::K)?)?,
            read_range(&rig.stream, state.kv_plane(ordinal, Plane::KScales)?)?,
            read_range(&rig.stream, state.kv_plane(ordinal, Plane::V)?)?,
            read_range(&rig.stream, state.kv_plane(ordinal, Plane::VScales)?)?,
        ]);
    }
    Ok(Snapshot {
        hidden,
        delta,
        kv,
        params: read_range(&rig.stream, state.word(Word::StepParams)?)?,
        statuses,
        tokens: state.tokens(),
    })
}

fn reset_with_canary(rig: &Rig, state: &mut ResidentState) -> Result<(), EngError> {
    state.reset(&rig.stream)?;
    let params = state.word(Word::StepParams)?;
    params.buffer().write_at(
        &rig.stream,
        params.offset(),
        &vec![PARAM_CANARY; params.bytes()],
    )
}

/// Uploads explicit ids and poisons the independent KV-length carrier before the body.
/// The attention stage must set its own carrier without consuming deliberately invalid StepParams.
fn run_ids(
    rig: &Rig,
    state: &mut ResidentState,
    model: &ChunkModel<'_, '_>,
    session: &RigSession,
    ids: &[i32],
    chunk_rows: usize,
) -> Result<Snapshot, EngError> {
    let (tensors, arena, part) = session;
    let bufs = rig_bufs(rig, tensors, arena, part, M_SESS)?;
    let mut hidden = Vec::new();
    let mut statuses = Vec::new();
    for ids in ids.chunks(chunk_rows) {
        let p0 = state.tokens();
        let rows = ChunkRows::new(p0, ids.len(), RIG_CAP, M_SESS)?;
        let end = rows.end();
        let carrier_at = CPF_ATTN_PART_CAP
            .checked_sub(I32)
            .ok_or_else(|| EngError::BadRequest("rig: missing KV length carrier".to_owned()))?;
        let ids_view = checked_view(tensors, arena, ChunkSlot::IdsTok)?;
        ids_view.buffer().write_at(
            &rig.stream,
            ids_view.offset(),
            &ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let h = checked_view(tensors, arena, ChunkSlot::Hidden)?;
        let h_bytes = ids.len() * rig.dims.shape.hidden * F32;
        ff_fill(&rig.stream, &h, h_bytes)?;
        part.write_at(&rig.stream, carrier_at, &(-1_i32).to_le_bytes())?;
        open_chunk(state, &rig.stream)?;
        let run = ChunkRun::new(state, model, &bufs, rows)?;
        run_chunk_body(
            &Launch {
                ctx: &rig.ctx,
                stream: &rig.stream,
            },
            &run,
        )?;
        let status = state.read_prefill_status(&rig.stream)?;
        statuses.push(status);
        state.revalidate(&rig.stream, end, status)?;
        let mut bytes = vec![0; h_bytes];
        h.buffer().read_at(&rig.stream, h.offset(), &mut bytes)?;
        hidden.extend(bytes);
    }
    snapshot(rig, state, hidden, statuses)
}

/// All device owners stay inside this Result-only phase. One arena and one owning stream
/// serve both batching schedules, their continuation and the dirty-state reset repeat.
fn mixed_body_phase() -> Result<[Snapshot; 5], EngError> {
    let rig = build_rig()?;
    let mut state = rig_state(&rig, KvMode::Int8G32)?;
    for ordinal in 0..state.layout().kv.layers {
        for &plane in state.layout().kv.planes() {
            let range = state.kv_plane(ordinal, plane)?;
            ff_fill(&rig.stream, &range, range.bytes())?;
        }
    }
    let nsm = rig.ctx.mmq_device_facts()?.nsm;
    let session = rig_session(&rig, M_SESS, nsm, CPF_ATTN_PART_CAP)?;
    let mixer = rig_mixer(&rig, false)?;
    let moe = rig_moe_slice(&rig, false)?;
    let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
    let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, RIG_LAYERS)?;
    let ids = clean_ids(M + 1, &mut Rng::new(0x4B56_3800_0010));
    reset_with_canary(&rig, &mut state)?;
    let batch = run_ids(&rig, &mut state, &model, &session, &ids[..M], M)?;
    let batch_next = run_ids(&rig, &mut state, &model, &session, &ids[M..], 1)?;
    reset_with_canary(&rig, &mut state)?;
    let singles = run_ids(&rig, &mut state, &model, &session, &ids[..M], 1)?;
    let singles_next = run_ids(&rig, &mut state, &model, &session, &ids[M..], 1)?;
    reset_with_canary(&rig, &mut state)?;
    let repeat = run_ids(&rig, &mut state, &model, &session, &ids[..M], M)?;
    Ok([batch, batch_next, singles, singles_next, repeat])
}

/// Checks every committed KV group, all DeltaNet states and untouched layer ordinals.
fn hard_gate(name: &str, snap: &Snapshot, hidden_rows: usize) {
    let dims = common_tests::dims();
    assert_eq!(snap.hidden.len(), hidden_rows * dims.shape.hidden * F32);
    assert_finite(name, &floats(&snap.hidden));
    assert!(
        snap.hidden.iter().any(|&byte| byte != 0),
        "{name}: empty hidden signal"
    );
    assert!(
        snap.statuses.iter().all(|&status| status == 0),
        "{name}: status"
    );
    assert!(
        snap.params.iter().all(|&byte| byte == PARAM_CANARY),
        "{name}: StepParams changed"
    );
    for (ordinal, states) in snap.delta.iter().enumerate() {
        for (which, bytes) in states.iter().enumerate() {
            assert_finite(&format!("{name}: delta {ordinal}/{which}"), &floats(bytes));
            if ordinal >= 4 {
                assert!(
                    bytes.iter().all(|&byte| byte == 0),
                    "{name}: untouched delta {ordinal}/{which}"
                );
            }
        }
        if ordinal < 4 {
            assert!(
                states.iter().flatten().any(|&byte| byte != 0),
                "{name}: unwritten delta {ordinal}"
            );
        }
    }
    for (ordinal, planes) in snap.kv.iter().enumerate() {
        for (plane, bytes) in planes.iter().enumerate() {
            if ordinal != 0 {
                assert!(
                    bytes.iter().all(|&byte| byte == NAN_FILL),
                    "{name}: untouched KV {ordinal}/{plane}"
                );
                continue;
            }
            let width = if plane.is_multiple_of(2) {
                dims.shape.attn.head_dim
            } else {
                dims.shape.attn.head_dim / 32 * 2
            };
            for head in 0..dims.shape.attn.heads_kv {
                let at = head * RIG_CAP * width;
                let prefix = &bytes[at..at + snap.tokens * width];
                if plane.is_multiple_of(2) {
                    assert!(
                        prefix.iter().all(|&code| code != 0x80),
                        "{name}: invalid code"
                    );
                    assert!(
                        prefix.iter().any(|&code| code != NAN_FILL),
                        "{name}: unwritten codes"
                    );
                } else {
                    for half in prefix.as_chunks::<2>().0 {
                        let scale = f16(u16::from_le_bytes(*half));
                        assert!(scale.is_finite() && scale >= 0.0, "{name}: invalid scale");
                    }
                    assert!(
                        prefix.iter().any(|&byte| byte != 0),
                        "{name}: zero scale plane"
                    );
                }
            }
        }
    }
}

/// Compares committed KV only: reset intentionally retains stale, uncommitted cells.
fn assert_committed_equal(name: &str, got: &Snapshot, want: &Snapshot) {
    assert_eq!(got.tokens, want.tokens, "{name}: committed rows");
    let tokens = got.tokens;
    assert_eq!(got.hidden, want.hidden, "{name}: hidden");
    assert_eq!(got.delta, want.delta, "{name}: DeltaNet");
    let dims = common_tests::dims();
    for (ordinal, (got, want)) in got.kv.iter().zip(&want.kv).enumerate() {
        for (plane, (got, want)) in got.iter().zip(want).enumerate() {
            let width = if plane.is_multiple_of(2) {
                dims.shape.attn.head_dim
            } else {
                dims.shape.attn.head_dim / 32 * 2
            };
            for head in 0..dims.shape.attn.heads_kv {
                let at = head * RIG_CAP * width;
                let end = at + tokens * width;
                assert_eq!(
                    &got[at..end],
                    &want[at..end],
                    "{name}: KV {ordinal}/{plane}/{head}"
                );
            }
        }
    }
}

/// Checks preserved prefixes/tails across append and reset. Only attention ordinal zero
/// belongs to the five-layer body; the hard gate checks every other ordinal separately.
fn assert_outside_write(before: &Snapshot, after: &Snapshot, start: usize, end: usize) {
    let dims = common_tests::dims();
    for (plane, (before, after)) in before.kv[0].iter().zip(&after.kv[0]).enumerate() {
        let width = if plane.is_multiple_of(2) {
            dims.shape.attn.head_dim
        } else {
            dims.shape.attn.head_dim / 32 * 2
        };
        for head in 0..dims.shape.attn.heads_kv {
            let at = head * RIG_CAP * width;
            assert_eq!(
                &before[at..at + start * width],
                &after[at..at + start * width],
                "KV {plane}/{head}: committed prefix changed"
            );
            assert_eq!(
                &before[at + end * width..at + RIG_CAP * width],
                &after[at + end * width..at + RIG_CAP * width],
                "KV {plane}/{head}: unwritten tail changed"
            );
        }
    }
}

fn f32_differences(got: &[u8], want: &[u8]) -> (usize, f64) {
    assert_eq!(got.len(), want.len());
    let mismatches = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let worst = floats(got)
        .iter()
        .zip(floats(want))
        .map(|(&a, b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max);
    (mismatches, worst)
}

/// Reports differences without asserting a cross-batching numerical acceptance threshold.
fn report_chunk_differences(name: &str, got: &Snapshot, want: &Snapshot) {
    assert_eq!(got.tokens, want.tokens);
    let (hidden_bytes, hidden_abs) = f32_differences(&got.hidden, &want.hidden);
    let mut delta_bytes = 0;
    let mut delta_abs = 0.0f64;
    for (got, want) in got.delta.iter().flatten().zip(want.delta.iter().flatten()) {
        let (bytes, abs) = f32_differences(got, want);
        delta_bytes += bytes;
        delta_abs = delta_abs.max(abs);
    }
    let dims = common_tests::dims();
    let mut kv_bytes = 0;
    for (got_planes, want_planes) in got.kv.iter().zip(&want.kv) {
        for (plane, (got_plane, want_plane)) in got_planes.iter().zip(want_planes).enumerate() {
            let width = if plane.is_multiple_of(2) {
                dims.shape.attn.head_dim
            } else {
                dims.shape.attn.head_dim / 32 * 2
            };
            for head in 0..dims.shape.attn.heads_kv {
                let at = head * RIG_CAP * width;
                let end = at + got.tokens * width;
                kv_bytes += got_plane[at..end]
                    .iter()
                    .zip(&want_plane[at..end])
                    .filter(|(a, b)| a != b)
                    .count();
            }
        }
    }
    println!(
        "STRUCTURAL DIAGNOSTIC ONLY {name}: hidden differing bytes {hidden_bytes}, max abs \
         {hidden_abs:.6e}; DeltaNet differing bytes {delta_bytes}, max abs {delta_abs:.6e}; \
         committed raw KV differing bytes {kv_bytes}; not production-step equivalence"
    );
}

/// A structural scheduling comparison, not a production-decode or target-model quality gate.
#[test]
fn chunk_kv8_mini_preserves_state_and_continues() {
    if !device_present("chunk_kv8_mini_preserves_state_and_continues") {
        return;
    }
    let [batch, batch_next, singles, singles_next, repeat] =
        mixed_body_phase().expect("the Result-only device phase");
    for (name, snap, rows, tokens) in [
        ("m16", &batch, M, M),
        ("m16 continuation", &batch_next, 1, M + 1),
        ("16m1", &singles, M, M),
        ("16m1 continuation", &singles_next, 1, M + 1),
        ("dirty reset repeat", &repeat, M, M),
    ] {
        hard_gate(name, snap, rows);
        assert_eq!(snap.tokens, tokens, "{name}: token publication");
    }
    assert_eq!(batch.statuses.len(), 1);
    assert_eq!(singles.statuses.len(), M);
    assert_eq!(batch_next.statuses.len(), 1);
    assert_eq!(singles_next.statuses.len(), 1);
    assert_eq!(repeat.statuses.len(), 1);
    let dims = common_tests::dims();
    for (plane, bytes) in batch.kv[0].iter().enumerate() {
        let width = if plane.is_multiple_of(2) {
            dims.shape.attn.head_dim
        } else {
            dims.shape.attn.head_dim / 32 * 2
        };
        for head in 0..dims.shape.attn.heads_kv {
            let at = head * RIG_CAP * width;
            assert!(
                bytes[at + M * width..at + RIG_CAP * width]
                    .iter()
                    .all(|&byte| byte == NAN_FILL),
                "m16: unwritten KV tail changed"
            );
        }
    }
    assert_outside_write(&batch, &batch_next, M, M + 1);
    assert_outside_write(&batch_next, &singles, 0, M);
    assert_outside_write(&singles, &singles_next, M, M + 1);
    assert_outside_write(&singles_next, &repeat, 0, M);
    assert_committed_equal("same-batch dirty-state reset repeat", &repeat, &batch);
    report_chunk_differences("m16 versus sixteen m1 CHUNK bodies", &batch, &singles);
    report_chunk_differences(
        "one-row continuation after each schedule",
        &batch_next,
        &singles_next,
    );
}
