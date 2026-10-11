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

#[test]
fn a_kind_without_a_block_is_refused() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    assert!(block_ops(AttentionKind::Qsa, &g).is_err());
}
