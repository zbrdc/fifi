//! CPU tests of the NATIVE mixer op lists: the Tiled lists pinned by LITERAL fixtures captured from
//! the code before the native layout (never the function under edit), and the same list-vs-plan
//! checks the Tiled lists have. The construction guards live in `native_guard_tests.rs`. CPU only.
use super::common_tests::{dims, plan_of};
use super::dims::{KvMode, StepOptions};
use super::mixer_geometry::MixerGeometry;
use super::mixer_ops::{BlockOp, block_ops, block_ops_with};
use super::state::StepField;
use super::state_layout::Plane;
use super::step_plan::{KernelScratch, Phase, Slot, StepPlan};
use crate::family::AttentionKind;
use crate::family::gguf_model::Role;
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::resident::attn_decode::AttnSplit;

fn geometry() -> MixerGeometry {
    MixerGeometry::new(&dims()).expect("geometry")
}

fn ops_of(kind: AttentionKind, layout: WeightLayout) -> Vec<BlockOp> {
    block_ops(kind, &geometry(), layout, KvMode::Fp16).expect("ops")
}

/// The NATIVE Int8G32 attention list, whose slots the Int8G32 plan of `native_plan_tests.rs`
/// carries.
fn int8_ops() -> Vec<BlockOp> {
    block_ops(
        AttentionKind::Full,
        &geometry(),
        WeightLayout::Native,
        KvMode::Int8G32,
    )
    .expect("ops")
}

/// The literal Tiled lists: the Tiled `block_ops` outputs for the two kinds, written out once
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

// ---- the Int8G32 attention op list --------------------------------------------------------

/// The Int8G32 attention list, written out literally like the fixtures above: the NATIVE list with
/// the fp16 attention core swapped for the KV8 triple (the rows entry, the append, the q8 decode at
/// the Q8 split). Every field is pinned, so a wrong slot, word, plane, split or order fails here by
/// name.
#[test]
fn the_int8_attention_list_is_pinned() {
    let want: Vec<BlockOp> = vec![
        BlockOp::NormQ8 {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            q8: Slot::Q8Mix,
            norm_out: Slot::NormRow,
        },
        BlockOp::ProjectQ {
            phase: Phase::AttnIn,
            role: Role::AttnQ,
            x_q8: Slot::Q8Mix,
            y: Slot::AttnProj,
            col: 0,
        },
        BlockOp::ProjectQ {
            phase: Phase::AttnIn,
            role: Role::AttnK,
            x_q8: Slot::Q8Mix,
            y: Slot::AttnProj,
            col: 8192,
        },
        BlockOp::ProjectQ {
            phase: Phase::AttnIn,
            role: Role::AttnV,
            x_q8: Slot::Q8Mix,
            y: Slot::AttnProj,
            col: 8704,
        },
        BlockOp::AttnPrepRows {
            proj: Slot::AttnProj,
            q: Slot::AttnQ,
            q_norm: Role::AttnQNorm,
            k_norm: Role::AttnKNorm,
            k_rows: Slot::KRows,
            v_rows: Slot::VRows,
            pos: StepField::Pos,
            status: StepField::Status,
        },
        BlockOp::Kv8Append {
            k: Slot::KRows,
            v: Slot::VRows,
            k_codes: Plane::K,
            k_scales: Plane::KScales,
            v_codes: Plane::V,
            v_scales: Plane::VScales,
            pos: StepField::Pos,
            status: StepField::Status,
        },
        BlockOp::AttnDecodeQ8 {
            q: Slot::AttnQ,
            proj: Slot::AttnProj,
            k_codes: Plane::K,
            k_scales: Plane::KScales,
            v_codes: Plane::V,
            v_scales: Plane::VScales,
            kv_len: StepField::KvLen,
            part: Slot::AttnPart,
            out: Slot::AttnCore,
            split: AttnSplit::Q8,
        },
        BlockOp::QuantizeQ8 {
            phase: Phase::AttnPack,
            x: Slot::AttnCore,
            q8: Slot::Q8AttnOut,
            k: 4096,
        },
        BlockOp::ProjectQ {
            phase: Phase::AttnOut,
            role: Role::AttnOutput,
            x_q8: Slot::Q8AttnOut,
            y: Slot::MixOut,
            col: 0,
        },
        BlockOp::NormQ8 {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            q8: Slot::Q8Moe,
            norm_out: Slot::NormRow,
        },
    ];
    assert_eq!(int8_ops(), want);
    // The Int8G32 list is the Fp16 NATIVE list with exactly the KV core swapped: prologue, the
    // three in-projections, ONE extra core op (the fp16 pair becomes a triple) and the tail.
    let fp16 = ops_of(AttentionKind::Full, WeightLayout::Native);
    assert_eq!(int8_ops().len(), fp16.len() + 1);
    assert_eq!(&int8_ops()[..4], &fp16[..4]);
    assert_eq!(&int8_ops()[7..], &fp16[6..]);
    // The DeltaNet list is the KV mode's twin: no attention core, nothing changes.
    assert_eq!(
        block_ops(
            AttentionKind::DeltaNet,
            &geometry(),
            WeightLayout::Native,
            KvMode::Int8G32
        )
        .expect("ops"),
        ops_of(AttentionKind::DeltaNet, WeightLayout::Native)
    );
    // The typed refusal the plan owns: the builder refuses the Tiled Int8G32 attention list itself,
    // naming the mode and the layout, with every earlier check satisfied (the geometry and the
    // prologue build fine).
    match block_ops(
        AttentionKind::Full,
        &geometry(),
        WeightLayout::Tiled,
        KvMode::Int8G32,
    ) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(
                message.contains("Int8G32") && message.contains("Tiled"),
                "{message}"
            );
        }
        other => panic!("a Tiled Int8G32 attention list must be refused: {other:?}"),
    }
}

/// The dataflow order of the KV8 core: the rows entry writes the f32 rows, the append reads them
/// and fills the code and scale planes, and only then does the q8 decode read the planes and the
/// kv_len word. Pinned on its own so a reorder dies here even if a field edit keeps the list
/// length.
#[test]
fn the_int8_kv_core_is_ordered_prep_append_decode() {
    let ops = int8_ops();
    let at = |want: fn(&BlockOp) -> bool| ops.iter().position(want).expect("op");
    let prep = at(|op| matches!(op, BlockOp::AttnPrepRows { .. }));
    let append = at(|op| matches!(op, BlockOp::Kv8Append { .. }));
    let decode = at(|op| matches!(op, BlockOp::AttnDecodeQ8 { .. }));
    assert!(prep < append, "the append reads the rows the prep writes");
    assert!(
        append < decode,
        "the decode reads the planes the append fills"
    );
    assert_eq!(ops[prep].writes(), [Slot::AttnQ, Slot::KRows, Slot::VRows]);
    assert_eq!(ops[append].reads(), [Slot::KRows, Slot::VRows]);
    assert!(
        ops[append].writes().is_empty(),
        "the planes are state, not slots"
    );
    assert_eq!(ops[decode].reads(), [Slot::AttnQ, Slot::AttnProj]);
}

/// The Int8G32 list obeys the `ops_tests.rs` rules against the Int8G32 plan: phases ascend,
/// every slot is written before it is read, and every touch sits inside the slot's lifetime
/// (KRows/VRows live ATTN_PREP..KV8_APPEND in the Int8G32 phase table).
#[test]
fn the_int8_list_ascends_and_stays_inside_the_int8_slots() {
    let d = dims();
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let scratch = KernelScratch::for_dims(&d, 1, KvMode::Int8G32).expect("scratch");
    let p = StepPlan::new(&d, 1, &options, &scratch).expect("plan");
    let life = |slot| {
        let planned = p.tensors()[p.index(slot).expect("slot in plan")];
        (planned.first, planned.last)
    };
    assert_eq!(
        life(Slot::KRows),
        (Phase::AttnPrep as usize, Phase::Kv8Append as usize)
    );
    let ops = int8_ops();
    let mut last = 0;
    let mut written: Vec<Slot> = Vec::new();
    for op in &ops {
        let phase = op.phase() as usize;
        assert!(phase >= last, "descent at {op:?}");
        last = phase;
        for slot in op.reads() {
            assert!(written.contains(&slot), "{slot:?} read before write");
        }
        written.extend(op.writes());
        for slot in op.reads().into_iter().chain(op.writes()) {
            let (first, end) = life(slot);
            assert!(
                first <= phase && phase <= end,
                "{slot:?} outside {first}..={end}"
            );
        }
    }
}

/// The fused GDN input-projection arm (flag ON): the four same-input sibling launches collapse
/// to ONE projection of `AttnQkv` at column 0, the DeltaNet list keeps every other op, and the
/// Full-attention list is untouched (its q/k/v fusion is a banked follow-up, not this slice).
#[test]
fn fused_gdn_inproj_collapses_the_sibling_launches_only() {
    let g = geometry();
    for layout in [WeightLayout::Native, WeightLayout::Tiled] {
        let unfused =
            block_ops(AttentionKind::DeltaNet, &g, layout, KvMode::Fp16).expect("unfused ops");
        let fused = block_ops_with(true, AttentionKind::DeltaNet, &g, layout, KvMode::Fp16)
            .expect("fused ops");
        let projects = |ops: &[BlockOp], role: Role| {
            ops.iter()
                .filter(|op| {
                    matches!(
                        op,
                        BlockOp::Project { role: r, .. }
                        | BlockOp::ProjectQ { role: r, .. }
                        | BlockOp::ProjectExl3 { role: r, .. }
                            if *r == role
                    )
                })
                .count()
        };
        for role in [Role::AttnQkv, Role::AttnGate, Role::SsmBeta, Role::SsmAlpha] {
            assert_eq!(
                projects(&unfused, role),
                1,
                "{role:?} in {layout:?} unfused"
            );
        }
        assert_eq!(projects(&fused, Role::AttnQkv), 1);
        for gone in [Role::AttnGate, Role::SsmBeta, Role::SsmAlpha] {
            assert_eq!(projects(&fused, gone), 0, "{gone:?} fused away");
        }
        // Everything except the three removed sibling projects is unchanged, in count and order.
        let kept = |ops: &[BlockOp]| {
            ops.iter()
                .filter(|op| {
                    !matches!(
                        op,
                        BlockOp::Project {
                            role: Role::AttnGate | Role::SsmBeta | Role::SsmAlpha,
                            ..
                        } | BlockOp::ProjectQ {
                            role: Role::AttnGate | Role::SsmBeta | Role::SsmAlpha,
                            ..
                        } | BlockOp::ProjectExl3 {
                            role: Role::AttnGate | Role::SsmBeta | Role::SsmAlpha,
                            ..
                        }
                    )
                })
                .count()
        };
        assert_eq!(kept(&unfused) + 3, unfused.len());
        assert_eq!(kept(&fused), kept(&unfused));
    }
    // The Full list never fuses in this slice.
    let full_unfused =
        block_ops(AttentionKind::Full, &g, WeightLayout::Native, KvMode::Fp16).expect("full ops");
    let full_fused = block_ops_with(
        true,
        AttentionKind::Full,
        &g,
        WeightLayout::Native,
        KvMode::Fp16,
    )
    .expect("full fused ops");
    assert_eq!(full_unfused.len(), full_fused.len());
}

/// The fused spec arm of `dense_specs_with`: one `AttnQkv` spec of the full fused row plus the
/// untouched output projection, and the unfused arm keeps the four siblings.
#[test]
fn fused_gdn_inproj_spec_is_one_row_wide_entry() {
    let g = geometry();
    let unfused = g
        .dense_specs_with(false, AttentionKind::DeltaNet)
        .expect("specs");
    let fused = g
        .dense_specs_with(true, AttentionKind::DeltaNet)
        .expect("fused specs");
    assert_eq!(unfused.len(), 5);
    assert_eq!(fused.len(), 2);
    let d = &g.delta;
    let want = d.channels + d.z_rows + d.beta_rows + d.alpha_rows;
    assert_eq!(want, d.ld, "the fixture geometry must stay contiguous");
    assert_eq!(fused[0].role, Role::AttnQkv);
    assert_eq!(fused[0].rows, want);
    assert_eq!(fused[1].role, Role::SsmOut);
    assert_eq!(fused[1].rows, unfused[4].rows);
}
