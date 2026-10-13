//! CPU tests of the NATIVE construction guards of MATVEC-PAR S3b (test seam U13): the pinned
//! fused-row column offsets, the layout accessor's mixed refusal and the rows % 4 / column
//! refusals. CPU only.
use super::common_tests::dims;
use super::mixer_geometry::{DenseSpec, MixerGeometry, check_dense, check_native_dense};
use super::mixer_ops::{block_ops, check_native_column};
use crate::EngError;
use crate::family::AttentionKind;
use crate::family::gguf_model::Role;
use crate::family::project::cuda::{TensorKind, WeightLayout, load_layout};

fn geometry() -> MixerGeometry {
    MixerGeometry::new(&dims()).expect("geometry")
}

/// The Qwen3.6 fused-row column offsets pinned as the matvec needs them (open point 2): the
/// geometry's offsets, and the construction of the Native lists (which runs `check_native_column`
/// over exactly these pairs) passing both kinds. The alignment line is gone with the arm
/// (MSG-74 item 3): `col * 4 % 4 == 0` held for every input by construction - a tautology.
/// The native embed guard (MSG-70 condition 5; C4's F2): only Q4_K passes, every other type
/// the loader can produce for the embedding table is refused naming the tensor and Q4_K.
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
            block_ops(kind, &g, WeightLayout::Native).is_ok(),
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

/// The typed refusals of the NATIVE construction (the packet's "unsupported type or row count",
/// of which the type half is the embed's Q4_K rule in C4 and the mixed-layout rule above): a
/// native dense weight of rows % 4 != 0, a wrong shape, and the two reachable column guards, each
/// case reaching its own guard with every earlier check satisfied.
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
/// norm's f32 scratch row is its own slot live exactly at FINAL_NORM (the S3c scratch defect 2:
/// NormRow's bytes may belong to a slot live at FINAL_NORM, so the fused producer's mandatory
/// row needed a home that cannot alias anything live).
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
