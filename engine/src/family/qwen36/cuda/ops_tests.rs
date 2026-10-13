//! Unit tests of the op lists of the mixer blocks against the real Qwen3.6 plan (test seam U13).
//! They check the list the executor runs, not a copy of it. Lifetimes are read from `StepPlan`.
//! CPU only.
use super::{
    common_tests::{dims, plan},
    mixer_geometry::MixerGeometry,
    mixer_ops::{BlockOp, block_ops},
    state::StepField,
    step_plan::{Phase, Slot},
};
use crate::family::{AttentionKind, gguf_model::Role};

fn ops_of(kind: AttentionKind) -> Vec<BlockOp> {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    block_ops(kind, &g).expect("ops")
}

fn kinds() -> Vec<AttentionKind> {
    vec![AttentionKind::DeltaNet, AttentionKind::Full]
}

#[test]
fn projection_columns_tile_the_shared_row_exactly() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    let expected = [
        (AttentionKind::DeltaNet, Slot::DeltaProj, g.delta.ld),
        (AttentionKind::Full, Slot::AttnProj, g.attn.ld),
    ];
    for (kind, row, ld) in expected {
        let specs = g.dense_specs(kind).expect("specs");
        let mut cols: Vec<(usize, usize)> = ops_of(kind)
            .iter()
            .filter_map(|op| match *op {
                BlockOp::Project { role, y, col, .. } if y == row => {
                    let rows = specs.iter().find(|s| s.role == role).expect("spec").rows;
                    Some((col, rows))
                }
                _ => None,
            })
            .collect();
        cols.sort_unstable();
        let mut next = 0;
        for (col, rows) in cols {
            assert_eq!(col, next, "{kind:?}: a gap or an overlap in the shared row");
            next = col + rows;
        }
        assert_eq!(next, ld, "{kind:?}");
    }
    let delta_cols: Vec<usize> = ops_of(AttentionKind::DeltaNet)
        .iter()
        .filter_map(|op| match *op {
            BlockOp::Project {
                y: Slot::DeltaProj,
                col,
                ..
            } => Some(col),
            _ => None,
        })
        .collect();
    assert_eq!(delta_cols, [0, 8192, 12288, 12320]);
}

#[test]
fn launches_ascend_and_every_slot_is_written_before_it_is_read() {
    for kind in kinds() {
        let ops = ops_of(kind);
        let mut last = 0;
        let mut written: Vec<Slot> = Vec::new();
        for op in &ops {
            assert!(
                op.phase() as usize >= last,
                "{kind:?}: phases must not descend at {op:?}"
            );
            last = op.phase() as usize;
            for slot in op.reads() {
                assert!(
                    written.contains(&slot),
                    "{kind:?}: {slot:?} is read before it is written"
                );
            }
            written.extend(op.writes());
        }
    }
    let attn = ops_of(AttentionKind::Full);
    let at = |want: fn(&BlockOp) -> bool| attn.iter().position(want).expect("op");
    let prep = at(|op| matches!(op, BlockOp::AttnPrep { .. }));
    let decode = at(|op| matches!(op, BlockOp::AttnDecode { .. }));
    assert!(prep < decode, "prep appends the KV row that decode reads");
}

#[test]
fn launches_stay_inside_their_slot_lifetimes() {
    let d = dims();
    let p = plan(&d);
    let life = |slot| {
        let planned = p.tensors()[p.index(slot).expect("slot in plan")];
        (planned.first, planned.last)
    };
    for kind in kinds() {
        let mut touches: Vec<(Slot, usize)> = Vec::new();
        for op in &ops_of(kind) {
            let phase = op.phase() as usize;
            for slot in op.reads().into_iter().chain(op.writes()) {
                let (first, last) = life(slot);
                assert!(
                    first <= phase && phase <= last,
                    "{kind:?}: {slot:?} at {:?} outside {first}..={last}",
                    op.phase()
                );
                touches.push((slot, phase));
            }
        }
        // A slot is live from its first writer to its last reader or writer.
        for &(slot, _) in &touches {
            let phases: Vec<usize> = touches
                .iter()
                .filter(|t| t.0 == slot)
                .map(|t| t.1)
                .collect();
            let (first, last) = life(slot);
            assert!(
                first <= *phases.iter().min().expect("one"),
                "{slot:?} starts late"
            );
            assert!(
                last >= *phases.iter().max().expect("one"),
                "{slot:?} ends early"
            );
        }
    }
}

#[test]
fn a_block_starts_with_the_input_norm_and_hands_the_moe_input_over() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    for kind in kinds() {
        let ops = block_ops(kind, &g).expect("ops");
        assert_eq!(
            ops.first(),
            Some(&BlockOp::NormPack {
                phase: Phase::MixPack,
                weight: Role::AttnNorm,
                delta: None,
                xf: Slot::XfMix,
                norm_out: None,
            })
        );
        let last = ops.last().expect("ops");
        assert_eq!(last.reads(), [Slot::MixOut]);
        assert_eq!(last.writes(), [Slot::XfMoe, Slot::NormRow]);
        assert_eq!(last.phase(), Phase::AddNorm);
        // The mixer output reaches MixOut exactly once.
        let into_mix_out = ops
            .iter()
            .filter(|op| op.writes().contains(&Slot::MixOut))
            .count();
        assert_eq!(into_mix_out, 1, "{kind:?}");
    }
}

#[test]
fn every_role_an_op_names_has_a_weight_spec() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    for kind in kinds() {
        let dense: Vec<Role> = g
            .dense_specs(kind)
            .expect("dense")
            .iter()
            .map(|s| s.role)
            .collect();
        let raw: Vec<Role> = g
            .raw_specs(kind)
            .expect("raw")
            .iter()
            .map(|s| s.role)
            .collect();
        for op in ops_of(kind) {
            let (needs_dense, needs_raw): (Vec<Role>, Vec<Role>) = match op {
                BlockOp::Project { role, .. } => (vec![role], vec![]),
                BlockOp::NormPack { weight, .. } => (vec![], vec![weight]),
                BlockOp::Conv { weights, .. } => (vec![], vec![weights]),
                BlockOp::Delta {
                    a, dt_bias, norm, ..
                } => (vec![], vec![a, dt_bias, norm]),
                BlockOp::AttnPrep { q_norm, k_norm, .. } => (vec![], vec![q_norm, k_norm]),
                _ => (vec![], vec![]),
            };
            for role in needs_dense {
                assert!(
                    dense.contains(&role),
                    "{kind:?}: no dense spec for {role:?}"
                );
            }
            for role in needs_raw {
                assert!(raw.contains(&role), "{kind:?}: no raw spec for {role:?}");
            }
        }
    }
}

#[test]
fn the_deltanet_op_list_matches_the_packet_table() {
    // Packet RD-9b3a section 2, launch sequence: every weight role, projection column and delta
    // argument the DeltaNet block names, in queue order (`Conv` and `Pack` name none of them).
    #[derive(Debug, PartialEq, Eq)]
    enum Expect {
        NormPack(Role),
        Project(Role, usize),
        Delta(Role, Role, Role),
    }
    let expected = vec![
        Expect::NormPack(Role::AttnNorm),
        Expect::Project(Role::AttnQkv, 0),
        Expect::Project(Role::AttnGate, 8192),
        Expect::Project(Role::SsmBeta, 12288),
        Expect::Project(Role::SsmAlpha, 12320),
        Expect::Delta(Role::SsmA, Role::SsmDtBias, Role::SsmNorm),
        Expect::Project(Role::SsmOut, 0),
        Expect::NormPack(Role::PostAttentionNorm),
    ];
    let got: Vec<Expect> = ops_of(AttentionKind::DeltaNet)
        .iter()
        .filter_map(|op| match op {
            BlockOp::NormPack { weight, .. } => Some(Expect::NormPack(*weight)),
            BlockOp::Project { role, col, .. } => Some(Expect::Project(*role, *col)),
            BlockOp::Delta {
                a, dt_bias, norm, ..
            } => Some(Expect::Delta(*a, *dt_bias, *norm)),
            _ => None,
        })
        .collect();
    assert_eq!(
        got, expected,
        "every role must stay bound to its column and op"
    );
}

/// The width the delta-rule output packs (packet RD-9b3a section 2): `k` of the `PackCall` is
/// `value_width`, pinned here as a number so the op cannot drift to the row length `ld`.
const PACKET_DELTA_VALUE_WIDTH: usize = 4096;

#[test]
fn the_deltanet_pack_op_reads_the_value_width_of_the_packet() {
    // Packet RD-9b3a section 2, D7: `pack(S, PackCall{x: DeltaCore, xf: XfDeltaOut, m:1,
    // k: value_width})`. Section 2 pins the projection columns of D1-D4 (`attn_gate` (z) at 8192,
    // `ssm_beta` at 12288) and `n_v_heads 32`, and the kernel's head size is 128 (RD-7b), so
    // `value_width` = the z gate block 12288 - 8192 = 32 x 128 = 4096 f32 values: `DeltaCore`
    // and `XfDeltaOut` hold one value-width row, never a whole `ld` projection row.
    let pack = ops_of(AttentionKind::DeltaNet)
        .into_iter()
        .find(|op| {
            matches!(
                op,
                BlockOp::Pack {
                    phase: Phase::DeltaPack,
                    ..
                }
            )
        })
        .expect("the DeltaNet list packs the delta-rule output");
    assert_eq!(
        pack,
        BlockOp::Pack {
            phase: Phase::DeltaPack,
            x: Slot::DeltaCore,
            xf: Slot::XfDeltaOut,
            k: PACKET_DELTA_VALUE_WIDTH,
        },
        "k must stay the packet's value width, not the projection row length"
    );
}

#[test]
fn the_attention_ops_name_the_step_words_they_take() {
    let ops = ops_of(AttentionKind::Full);
    let prep = ops
        .iter()
        .find(|op| matches!(op, BlockOp::AttnPrep { .. }))
        .expect("prep");
    let BlockOp::AttnPrep { pos, status, .. } = *prep else {
        unreachable!()
    };
    assert_eq!((pos, status), (StepField::Pos, StepField::Status));
    let decode = ops
        .iter()
        .find(|op| matches!(op, BlockOp::AttnDecode { .. }))
        .expect("decode");
    let BlockOp::AttnDecode { kv_len, part, .. } = *decode else {
        unreachable!()
    };
    assert_eq!((kv_len, part), (StepField::KvLen, Slot::AttnPart));
}

/// Packet RD-9b3b section 2, A1-A3: the role -> column association of the three `AttnIn`
/// projections. `attn_q` writes col 0 with 8192 columns (per head `[q 256 | gate 256]` over
/// `n_head` 16), `attn_k` col 8192 and `attn_v` col 8704 with 512 columns each (`n_kv_head` 2 of
/// `head_dim` 256; `k_off` 8192, `v_off` 8704). k and v are the same width, so a role swap
/// between them leaves the column tiling exact: only this pin binds each role to its column.
#[test]
fn the_attnin_projections_bind_each_role_to_its_column() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    let specs = g.dense_specs(AttentionKind::Full).expect("specs");
    let width = |role: Role| {
        specs
            .iter()
            .find(|s| s.role == role)
            .expect("dense spec")
            .rows
    };
    let expected = [
        (Role::AttnQ, 0usize, 8192usize),
        (Role::AttnK, 8192, 512),
        (Role::AttnV, 8704, 512),
    ];
    let got: Vec<(Role, usize, usize)> = ops_of(AttentionKind::Full)
        .iter()
        .filter_map(|op| match *op {
            BlockOp::Project {
                phase: Phase::AttnIn,
                role,
                xf: _,
                y: Slot::AttnProj,
                col,
            } => Some((role, col, width(role))),
            _ => None,
        })
        .collect();
    assert_eq!(
        got, expected,
        "each AttnIn role must stay on the column the packet names"
    );
}

/// Packet RD-9b3b section 2, A4: the prep op carries its two slots, its two norm weights and its
/// two step words exactly. The wrapper norms q with `q_norm` and k with `k_norm` (RD-8a module
/// doc: per-head RMSNorm of q and k, the weight of each head applied to its own half), so the
/// role each field names is pinned, not just that both roles appear somewhere in the list.
#[test]
fn the_attention_prep_op_pins_its_slots_and_norm_weights() {
    let prep = ops_of(AttentionKind::Full)
        .into_iter()
        .find(|op| matches!(op, BlockOp::AttnPrep { .. }))
        .expect("prep");
    assert_eq!(
        prep,
        BlockOp::AttnPrep {
            proj: Slot::AttnProj,
            q: Slot::AttnQ,
            q_norm: Role::AttnQNorm,
            k_norm: Role::AttnKNorm,
            pos: StepField::Pos,
            status: StepField::Status,
        },
        "the q norm belongs to q and the k norm to k (packet A4)"
    );
}

/// Packet RD-9b3b section 2, A5: the decode op carries its four slots and its step word exactly:
/// `q` is the normed and roped query of prep, `proj` the gated projection row (the wrapper reads
/// head `h`'s gate from it), `part` the partials scratch and `out` the pre-projection core. Both
/// `q` and `proj` are written and live at `AttnCore`, so a swap between them passes every
/// lifetime test: only this pin tells them apart.
#[test]
fn the_attention_decode_op_pins_its_slots() {
    let decode = ops_of(AttentionKind::Full)
        .into_iter()
        .find(|op| matches!(op, BlockOp::AttnDecode { .. }))
        .expect("decode");
    assert_eq!(
        decode,
        BlockOp::AttnDecode {
            q: Slot::AttnQ,
            proj: Slot::AttnProj,
            part: Slot::AttnPart,
            out: Slot::AttnCore,
            kv_len: StepField::KvLen,
        },
        "q is the normed query, proj the gate row (packet A5)"
    );
}

/// The width the attention output packs (packet RD-9b3b section 2): `k` of the attention `Pack`
/// is `q_width`, pinned here as a number so the op cannot drift to the row length `ld`.
const PACKET_ATTN_Q_WIDTH: usize = 4096;

#[test]
fn the_attention_pack_op_reads_the_query_width_of_the_packet() {
    // Packet RD-9b3b section 2, A6: `pack(.. x: AttnCore, xf: XfAttnOut, k: q_width)`. A1 makes
    // `attn_q` col 0 of the shared row, 8192 columns per head `[q 256 | gate 256]` over `n_head`
    // 16, and decode's `out` is f32 `[m][n_head * head_dim]` (RD-8b), so one q half is
    // 16 x 256 = 4096 f32 values: `AttnCore` and `XfAttnOut` hold one q-width row, never the
    // 9216-float projection row `ld`.
    let pack = ops_of(AttentionKind::Full)
        .into_iter()
        .find(|op| {
            matches!(
                op,
                BlockOp::Pack {
                    phase: Phase::AttnPack,
                    ..
                }
            )
        })
        .expect("the attention list packs the decode output");
    assert_eq!(
        pack,
        BlockOp::Pack {
            phase: Phase::AttnPack,
            x: Slot::AttnCore,
            xf: Slot::XfAttnOut,
            k: PACKET_ATTN_Q_WIDTH,
        },
        "k must stay the packet's q width, not the projection row length"
    );
}

#[test]
fn a_kind_without_a_block_is_refused() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    assert!(block_ops(AttentionKind::Qsa, &g).is_err());
}
