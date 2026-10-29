//! CPU tests of the NATIVE construction guards: the pinned fused-row column offsets, the layout
//! accessor's mixed refusal and the rows % 4 / column refusals. CPU only.
use super::common_tests::dims;
use super::dims::KvMode;
use super::mixer_geometry::{DenseSpec, MixerGeometry, check_dense, check_native_dense};
use super::mixer_ops::{block_ops, check_native_column};
use crate::EngError;
use crate::family::AttentionKind;
use crate::family::gguf_model::Role;
use crate::family::project::cuda::{TensorKind, WeightLayout, load_layout};

fn geometry() -> MixerGeometry {
    MixerGeometry::new(&dims()).expect("geometry")
}

/// The Qwen3.6 fused-row column offsets pinned as the matvec needs them: the geometry's offsets,
/// and the construction of the Native lists (which runs `check_native_column` over exactly these
/// pairs) passing both kinds. There is no alignment line: `col * 4 % 4 == 0` holds for every input
/// by construction. The native embed guard: only Q4_K passes, every other type the loader can
/// produce for the embedding table is refused naming the tensor and Q4_K.
#[test]
fn the_native_embed_takes_q4_k_blocks_only() {
    use super::resources::check_native_embed;
    use crate::kernels::cuda::resident::matvec_q::QuantType;
    assert!(check_native_embed("token_embd.weight", QuantType::Q4K).is_ok());
    for quant in [QuantType::Q6K, QuantType::Q50, QuantType::Q80] {
        match check_native_embed("token_embd.weight", quant) {
            Err(EngError::Checkpoint(message)) => {
                assert!(
                    message.contains("Q4_K") && message.contains("token_embd.weight"),
                    "{message}"
                )
            }
            other => panic!("a {quant:?} embedding table must be refused: {other:?}"),
        }
    }
}

#[test]
fn the_native_projection_columns_are_aligned() {
    let g = geometry();
    let (d, a) = (&g.delta, &g.attn);
    assert_eq!(
        (d.z_off, d.beta_off, d.alpha_off, d.ld),
        (8192, 12288, 12320, 12352)
    );
    assert_eq!((a.k_off, a.v_off, a.ld), (8192, 8704, 9216));
    for kind in [AttentionKind::DeltaNet, AttentionKind::Full] {
        assert!(
            block_ops(kind, &g, WeightLayout::Native, KvMode::Fp16).is_ok(),
            "{kind:?}"
        );
    }
}

/// A load that mixes the two layouts is refused naming the tensor that broke the one-layout rule;
/// a raw-only load has no quantized reader and answers Tiled. Pure: names and kinds only.
#[test]
fn a_mixed_layout_is_refused_naming_the_tensor() {
    let ok = |tensors: &[(&str, TensorKind)]| load_layout(tensors.iter().copied());
    assert_eq!(
        ok(&[("a", TensorKind::Tiled), ("b", TensorKind::Raw)]),
        Ok(WeightLayout::Tiled)
    );
    assert_eq!(
        ok(&[("a", TensorKind::Raw), ("b", TensorKind::Native)]),
        Ok(WeightLayout::Native)
    );
    assert_eq!(ok(&[("a", TensorKind::Raw)]), Ok(WeightLayout::Tiled));
    match ok(&[("a", TensorKind::Tiled), ("b", TensorKind::Native)]) {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("tensor b"), "{message}");
            assert!(
                message.contains("Native") && message.contains("Tiled"),
                "{message}"
            );
        }
        other => panic!("a mixed load must be refused: {other:?}"),
    }
    // The native tensor first names the tiled one as the one that broke it.
    match ok(&[("a", TensorKind::Native), ("b", TensorKind::Tiled)]) {
        Err(EngError::Checkpoint(message)) => assert!(message.contains("tensor b"), "{message}"),
        other => panic!("a mixed load must be refused: {other:?}"),
    }
}

#[test]
fn exl3_layout_is_distinct_from_both_gguf_layouts() {
    assert!(matches!(
        crate::family::project::cuda::CudaWeights::from_exl3(Vec::new()),
        Err(EngError::Checkpoint(_))
    ));
    let layout = |tensors: &[(&str, TensorKind)]| load_layout(tensors.iter().copied());
    assert_eq!(
        layout(&[("norm", TensorKind::Raw), ("q_proj", TensorKind::Exl3)]),
        Ok(WeightLayout::Exl3)
    );
    for other in [TensorKind::Tiled, TensorKind::Native] {
        for kinds in [[TensorKind::Exl3, other], [other, TensorKind::Exl3]] {
            match layout(&[("first", kinds[0]), ("second", kinds[1])]) {
                Err(EngError::Checkpoint(message)) => {
                    assert!(message.contains("tensor second"), "{message}");
                    assert!(message.contains("Exl3"), "{message}");
                }
                other => panic!("EXL3 must not masquerade as a GGUF layout: {other:?}"),
            }
        }
    }
}

#[test]
fn exl3_mixers_read_f32_and_keep_projection_inputs_live() {
    use super::dims::StepOptions;
    use super::mixer_ops::BlockOp;
    use super::step_plan::{KernelScratch, StepPlan};
    let d = dims();
    let g = geometry();
    for kv_mode in [KvMode::Fp16, KvMode::Int8G32] {
        let options = StepOptions {
            kv_mode,
            context_tokens: 1024,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: WeightLayout::Exl3,
        };
        let scratch = KernelScratch::for_dims(&d, 1, kv_mode).unwrap();
        let plan = StepPlan::exl3_mixers(&d, &options, &scratch).unwrap();
        assert!(StepPlan::new(&d, 1, &options, &scratch).is_err());
        for kind in [AttentionKind::DeltaNet, AttentionKind::Full] {
            let ops = block_ops(kind, &g, WeightLayout::Exl3, kv_mode).unwrap();
            assert!(
                ops.iter()
                    .any(|op| matches!(op, BlockOp::ProjectExl3 { .. }))
            );
            assert!(!ops.iter().any(|op| matches!(
                op,
                BlockOp::Project { .. }
                    | BlockOp::ProjectQ { .. }
                    | BlockOp::Pack { .. }
                    | BlockOp::QuantizeQ8 { .. }
                    | BlockOp::NormQ8 { .. }
            )));
            assert_exl3_live_slots(&plan, &ops);
        }
    }
}

fn assert_exl3_live_slots(plan: &super::step_plan::StepPlan, ops: &[super::mixer_ops::BlockOp]) {
    let mut written = Vec::new();
    for op in ops {
        for slot in op.reads() {
            assert!(written.contains(&slot), "uninitialized {slot:?} in {op:?}");
        }
        for slot in op.reads().into_iter().chain(op.writes()) {
            let span = plan.tensors()[plan.index(slot).expect("planned slot")];
            let phase = op.phase() as usize;
            assert!(
                span.first <= phase && phase <= span.last,
                "dead {slot:?} in {op:?}"
            );
        }
        written.extend(op.writes());
    }
}

/// The typed refusals of the NATIVE construction (an unsupported type or row count; the type half
/// is the embed's Q4_K rule and the mixed-layout rule above): a native dense weight of
/// rows % 4 != 0, a wrong shape, and the two reachable column guards, each case reaching its own
/// guard with every earlier check satisfied.
#[test]
fn an_unsupported_type_or_row_count_is_refused_at_construction() {
    let spec = super::mixer_geometry::DenseSpec {
        role: Role::AttnQkv,
        rows: 8192,
        k: 2048,
    };
    // The valid native shape passes; a wrong k or row count is the shape guard's; the rows%4
    // case carries a spec of EXACTLY that ragged row count, so only the row-group guard can
    // refuse it. The two column guards follow, each reaching its own arm.
    assert!(check_native_dense("w", (8192, 2048), &spec).is_ok());
    assert!(check_dense("w", (8192, 4096, None), &spec).is_err());
    assert!(check_dense("w", (8196, 2048, None), &spec).is_err());
    let ragged = DenseSpec {
        role: Role::SsmBeta,
        rows: 8190,
        k: 2048,
    };
    let refused = |result: Result<(), EngError>, needle: &str| match result {
        Err(EngError::Checkpoint(message)) => assert!(message.contains(needle), "{message}"),
        other => panic!("the native construction must refuse: {other:?}"),
    };
    refused(
        check_native_dense("w", (8190, 2048), &ragged),
        "multiple of 4",
    );
    refused(
        check_native_column(Role::AttnQkv, usize::MAX / 2, 4, usize::MAX),
        "overflows",
    );
    refused(
        check_native_column(Role::SsmBeta, 12288, 33, 12320),
        "past the fused row",
    );
    // The same pair one column shorter fits (the offset arm is a pinned statement: col * 4 is a
    // multiple of 4 for every col, so no input reaches it).
    assert!(check_native_column(Role::SsmBeta, 12288, 32, 12320).is_ok());
}

/// The head and the final norm name only slots their layout's plan holds, and the native final
/// norm's f32 scratch row is its own slot live exactly at FINAL_NORM (NormRow's bytes may belong to
/// a slot live at FINAL_NORM, so the fused producer's mandatory row needs a home that cannot alias
/// anything live).
#[test]
fn the_head_and_final_norm_name_only_live_slots_of_their_layout() {
    use super::common_tests::plan_of;
    use super::step_plan::{Phase, Slot};
    let native = plan_of(&super::common_tests::dims(), WeightLayout::Native);
    let tiled = plan_of(&super::common_tests::dims(), WeightLayout::Tiled);
    let has = |plan: &super::step_plan::StepPlan, slot: Slot| plan.index(slot).is_some();
    // final_norm: Tiled reads the last residual and writes XfFinal; Native writes Q8Final and
    // the FinalNormRow scratch beside reading the residual (not a plan slot).
    assert!(has(&tiled, Slot::XfFinal), "the tiled plan holds XfFinal");
    for slot in [Slot::Q8Final, Slot::FinalNormRow] {
        assert!(has(&native, slot), "the native plan holds {slot:?}");
    }
    // lm_head: the tiled arm reads XfFinal; the native arm reads Q8Final; both write Logits.
    assert!(has(&tiled, Slot::Logits) && has(&native, Slot::Logits));
    assert!(
        !has(&native, Slot::XfFinal),
        "the defect: the native plan has no XfFinal"
    );
    // The scratch row lives exactly at FINAL_NORM, dead before the head reads its pack.
    let at = native.index(Slot::FinalNormRow).expect("the scratch row");
    let planned = native.tensors()[at];
    assert_eq!(
        (planned.first, planned.last),
        (Phase::FinalNorm as usize, Phase::FinalNorm as usize)
    );
}
