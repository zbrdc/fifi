//! CPU tests of the NATIVE mixer op lists of MATVEC-PAR S3b (test seam U13): the Tiled lists
//! pinned by LITERAL fixtures captured from a4ca4075's code (never the function under edit),
//! and the same list-vs-plan checks the Tiled lists have. The construction guards live in
//! `native_guard_tests.rs`. CPU only.
use super::common_tests::{dims, plan_of};
use super::mixer_geometry::MixerGeometry;
use super::mixer_ops::{BlockOp, block_ops};
use super::step_plan::{Phase, Slot};
use crate::family::AttentionKind;
use crate::family::gguf_model::Role;
use crate::family::project::cuda::WeightLayout;

fn geometry() -> MixerGeometry {
    MixerGeometry::new(&dims()).expect("geometry")
}

fn ops_of(kind: AttentionKind, layout: WeightLayout) -> Vec<BlockOp> {
    block_ops(kind, &geometry(), layout).expect("ops")
}

/// The literal a4ca4075 lists: the Tiled `block_ops` outputs for the two kinds, written out once
/// from the code at the integration base and never derived from the function under edit, so any
/// behaviour change of the Tiled path fails here by name.
#[test]
fn native_lists_equal_the_tiled_lists_when_tiled() {
    let want_delta: Vec<BlockOp> = vec![
        BlockOp::NormPack {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            xf: Slot::XfMix,
            norm_out: None,
        },
        BlockOp::Project {
            phase: Phase::DeltaIn,
            role: Role::AttnQkv,
            xf: Slot::XfMix,
            y: Slot::DeltaProj,
            col: 0,
        },
        BlockOp::Project {
            phase: Phase::DeltaIn,
            role: Role::AttnGate,
            xf: Slot::XfMix,
            y: Slot::DeltaProj,
            col: 8192,
        },
        BlockOp::Project {
            phase: Phase::DeltaIn,
            role: Role::SsmBeta,
            xf: Slot::XfMix,
            y: Slot::DeltaProj,
            col: 12288,
        },
        BlockOp::Project {
            phase: Phase::DeltaIn,
            role: Role::SsmAlpha,
            xf: Slot::XfMix,
            y: Slot::DeltaProj,
            col: 12320,
        },
        BlockOp::Conv {
            proj: Slot::DeltaProj,
            out: Slot::DeltaConv,
            weights: Role::SsmConv1d,
        },
        BlockOp::Delta {
            conv: Slot::DeltaConv,
            proj: Slot::DeltaProj,
            out: Slot::DeltaCore,
            a: Role::SsmA,
            dt_bias: Role::SsmDtBias,
            norm: Role::SsmNorm,
        },
        BlockOp::Pack {
            phase: Phase::DeltaPack,
            x: Slot::DeltaCore,
            xf: Slot::XfDeltaOut,
            k: 4096,
        },
        BlockOp::Project {
            phase: Phase::DeltaOut,
            role: Role::SsmOut,
            xf: Slot::XfDeltaOut,
            y: Slot::MixOut,
            col: 0,
        },
        BlockOp::NormPack {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            xf: Slot::XfMoe,
            norm_out: Some(Slot::NormRow),
        },
    ];
    let want_attn: Vec<BlockOp> = vec![
        BlockOp::NormPack {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            xf: Slot::XfMix,
            norm_out: None,
        },
        BlockOp::Project {
            phase: Phase::AttnIn,
            role: Role::AttnQ,
            xf: Slot::XfMix,
            y: Slot::AttnProj,
            col: 0,
        },
        BlockOp::Project {
            phase: Phase::AttnIn,
            role: Role::AttnK,
            xf: Slot::XfMix,
            y: Slot::AttnProj,
            col: 8192,
        },
        BlockOp::Project {
            phase: Phase::AttnIn,
            role: Role::AttnV,
            xf: Slot::XfMix,
            y: Slot::AttnProj,
            col: 8704,
        },
        BlockOp::AttnPrep {
            proj: Slot::AttnProj,
            q: Slot::AttnQ,
            q_norm: Role::AttnQNorm,
            k_norm: Role::AttnKNorm,
            pos: super::state::StepField::Pos,
            status: super::state::StepField::Status,
        },
        BlockOp::AttnDecode {
            q: Slot::AttnQ,
            proj: Slot::AttnProj,
            part: Slot::AttnPart,
            out: Slot::AttnCore,
            kv_len: super::state::StepField::KvLen,
        },
        BlockOp::Pack {
            phase: Phase::AttnPack,
            x: Slot::AttnCore,
            xf: Slot::XfAttnOut,
            k: 4096,
        },
        BlockOp::Project {
            phase: Phase::AttnOut,
            role: Role::AttnOutput,
            xf: Slot::XfAttnOut,
            y: Slot::MixOut,
            col: 0,
        },
        BlockOp::NormPack {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            xf: Slot::XfMoe,
            norm_out: Some(Slot::NormRow),
        },
    ];
    assert_eq!(
        ops_of(AttentionKind::DeltaNet, WeightLayout::Tiled),
        want_delta
    );
    assert_eq!(ops_of(AttentionKind::Full, WeightLayout::Tiled), want_attn);
}

/// The Native lists run the `ops_tests.rs` rules: phases ascend, every slot is written before
/// it is read, every touch sits inside the slot's lifetime in the NATIVE plan.
#[test]
fn native_lists_ascend_and_stay_inside_their_slot_lifetimes() {
    let p = plan_of(&dims(), WeightLayout::Native);
    let life = |slot| {
        let planned = p.tensors()[p.index(slot).expect("slot in plan")];
        (planned.first, planned.last)
    };
    for kind in [AttentionKind::DeltaNet, AttentionKind::Full] {
        let ops = ops_of(kind, WeightLayout::Native);
        assert_eq!(
            ops.first(),
            Some(&BlockOp::NormQ8 {
                phase: Phase::MixPack,
                weight: Role::AttnNorm,
                delta: None,
                q8: Slot::Q8Mix,
                norm_out: Slot::NormRow,
            }),
            "{kind:?}"
        );
        assert_eq!(
            ops.last(),
            Some(&BlockOp::NormQ8 {
                phase: Phase::AddNorm,
                weight: Role::PostAttentionNorm,
                delta: Some(Slot::MixOut),
                q8: Slot::Q8Moe,
                norm_out: Slot::NormRow,
            }),
            "{kind:?}"
        );
        let mut last = 0;
        let mut written: Vec<Slot> = Vec::new();
        for op in &ops {
            let phase = op.phase() as usize;
            assert!(phase >= last, "{kind:?}: descent at {op:?}");
            last = phase;
            for slot in op.reads() {
                assert!(
                    written.contains(&slot),
                    "{kind:?}: {slot:?} read before write"
                );
            }
            written.extend(op.writes());
            for slot in op.reads().into_iter().chain(op.writes()) {
                let (first, end) = life(slot);
                assert!(
                    first <= phase && phase <= end,
                    "{kind:?}: {slot:?} outside {first}..={end}"
                );
            }
        }
    }
}
