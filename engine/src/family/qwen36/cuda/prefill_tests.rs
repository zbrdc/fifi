//! Host tests of the prefill chunk loop's pure layer (CPF-7a part 2, D.1). Real dims from
//! `common_tests::dims()`; every refusal row reaches its own guard with every other guard
//! satisfied and asserts its own text.

use super::{
    common_tests,
    prefill::{
        ALL_SLOTS, ChunkRows, EMBED_PHASE, EMBED_SLOTS, MoeStage, Stage, check_q8_layouts,
        check_view_layout, check_view_list, kv8_segment_query_rows, kv8_use_segments, layer_stages,
        moe_stage_phase, moe_stage_slots, resident_moe_stages, routed_swiglu, routed_up_view,
        slot_position, stage_phase, stage_slots,
    },
    prefill_plan::{M_LADDER, chunk_tensors},
};
use crate::family::project::cuda::WeightLayout;
use crate::{
    EngError,
    family::spec::AttentionKind,
    kernels::cuda::{
        plan_arena,
        prefill::{mmq_mmid::mmq_type_of, mmq_size::Q8_1Layout},
        resident::matvec_q::QuantType,
    },
};

#[test]
fn a_single_q8_tile_avoids_deep_segment_staging() {
    for rows in [0, 1, 5, 7, 15, 16] {
        assert!(!kv8_use_segments(rows), "{rows} rows fit one Q8 tile");
    }
    for rows in [17, 41, 64, 128, 256, 512, 1024] {
        assert!(kv8_use_segments(rows), "stage the larger {rows}-row chunk");
    }
}

#[test]
fn segmented_query_slices_preserve_session_rows_within_part_cap() {
    use crate::kernels::cuda::prefill::attn_prefill::{
        CPF_ATTN_MIN_KEYS, CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE, CpfAttnGeometry, CpfAttnPlan,
        cpf_attn_part_bytes,
    };
    let geometry = CpfAttnGeometry {
        m: 512,
        n_head: 16,
        n_kv_head: 2,
        head_dim: 256,
    };
    let plan = CpfAttnPlan {
        q_tile: CPF_ATTN_Q_TILE,
        max_splits: 8,
        min_keys: CPF_ATTN_MIN_KEYS,
    };
    assert_eq!(cpf_attn_part_bytes(geometry, plan).unwrap(), 67_633_152);
    let query_rows = kv8_segment_query_rows(geometry, 8).expect("bounded query subtiles");
    assert_eq!(query_rows, 256);
    assert_eq!(geometry.m, 512, "the admitted logical chunk is unchanged");
    let bytes = cpf_attn_part_bytes(
        CpfAttnGeometry {
            m: query_rows,
            ..geometry
        },
        plan,
    )
    .unwrap();
    assert_eq!(bytes, 33_816_576);
    assert!(bytes <= CPF_ATTN_PART_CAP);
    assert_eq!(kv8_segment_query_rows(geometry, 7).unwrap(), 512);
}

#[test]
fn segmented_query_plan_checks_bounds_and_covers_each_row() {
    use crate::kernels::cuda::prefill::attn_prefill::{
        CPF_ATTN_MIN_KEYS, CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE, CpfAttnGeometry, CpfAttnPlan,
        cpf_attn_part_bytes,
    };
    let geometry = CpfAttnGeometry {
        m: 512,
        n_head: 16,
        n_kv_head: 2,
        head_dim: 256,
    };
    for bad in [0, 129] {
        assert!(matches!(
            kv8_segment_query_rows(geometry, bad),
            Err(EngError::BadRequest(_))
        ));
    }
    assert!(kv8_segment_query_rows(CpfAttnGeometry { m: 0, ..geometry }, 8).is_err());
    assert!(matches!(
        kv8_segment_query_rows(
            CpfAttnGeometry {
                m: 8,
                n_head: 512,
                n_kv_head: 64,
                ..geometry
            },
            128
        ),
        Err(EngError::MemoryPlan(_))
    ));
    for m in [1, 7, 8, 16, 128, 512, 1024] {
        for segments in [1, 2, 7, 8, 128] {
            let geometry = CpfAttnGeometry { m, ..geometry };
            let query = kv8_segment_query_rows(geometry, segments).unwrap();
            assert!((1..=m).contains(&query));
            let plan = CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: segments,
                min_keys: CPF_ATTN_MIN_KEYS,
            };
            assert!(
                cpf_attn_part_bytes(
                    CpfAttnGeometry {
                        m: query,
                        ..geometry
                    },
                    plan
                )
                .unwrap()
                    <= CPF_ATTN_PART_CAP
            );
            let covered: usize = (0..m).step_by(query).map(|row| query.min(m - row)).sum();
            assert_eq!(covered, m);
        }
    }
}

#[test]
fn segmented_launch_counts_are_visible_and_reset_between_jobs() {
    let mut report = super::run_report::PrefillReport::default();
    assert_eq!(report.segmented_parts(), 0);
    assert!(!report.job_line().contains("segmented_parts"));
    report.note_segmented_parts(5);
    report.note_segmented_parts(7);
    assert_eq!(report.segmented_parts(), 12);
    assert!(report.job_line().ends_with(" segmented_parts 12"));
    report.clear();
    assert_eq!(report.segmented_parts(), 0);
    assert!(!report.job_line().contains("segmented_parts"));
}

/// H5: the slot column of `chunk_tensors` is a permutation of `ALL_SLOTS` at every ladder
/// size, so a slot added to the plan without this module fails here.
#[test]
fn prefill_views_cover_every_slot_once() {
    for m_sess in M_LADDER {
        let list = chunk_tensors(&common_tests::dims(), m_sess, 82).expect("chunk_tensors");
        let slots: Vec<_> = list.iter().map(|(slot, _)| *slot).collect();
        assert_eq!(slots.len(), ALL_SLOTS.len(), "m_sess {m_sess}: extra slots");
        for slot in ALL_SLOTS {
            let count = slots.iter().filter(|known| **known == slot).count();
            assert_eq!(count, 1, "m_sess {m_sess}: {slot:?} appears {count} times");
        }
    }
}

/// H6: one row per guard of `ChunkRows::new`, each with the others satisfied, plus the
/// accepting edges. Kills 7a-M7.
#[test]
fn chunk_rows_refuse_each_bound() {
    let (cap, m_sess) = (4_096usize, 128usize);
    let bad_request = |p0: usize, m: usize, want: &str| {
        let error = ChunkRows::new(p0, m, cap, m_sess).expect_err("refused");
        assert!(
            matches!(&error, EngError::BadRequest(text) if text.contains(want)),
            "{p0}+{m}: {error:?} does not contain {want:?}"
        );
    };
    // The accepting edges: m = m_sess, and p0 + m = cap.
    assert!(ChunkRows::new(972, m_sess, cap, m_sess).is_ok());
    assert!(ChunkRows::new(cap - m_sess, m_sess, cap, m_sess).is_ok());
    assert_eq!(
        ChunkRows::new(972, m_sess, cap, m_sess).unwrap().end(),
        1_100
    );
    // R1, both edges.
    bad_request(0, 0, "prefill chunk: rows");
    bad_request(0, m_sess + 1, "prefill chunk: rows");
    // R2 (R1 satisfied: m = 1).
    bad_request(usize::MAX, 1, "p0 + m overflows");
    // R3 (R1, R2 satisfied: p0 + m = cap + 1).
    let error = ChunkRows::new(cap + 1 - m_sess, m_sess, cap, m_sess).expect_err("refused");
    assert!(
        matches!(&error, EngError::ContextLimit(text) if text.contains("prefill chunk: rows")),
        "{error:?} is not the ContextLimit refusal"
    );
}

/// H9: one row per guard of the view checks, each with every earlier guard satisfied; then
/// `slot_position`. Kills 7a-V1, V2, V3, V5, V6, V7.
#[test]
fn view_list_refuses_each_guard() {
    let slots: Vec<_> = ALL_SLOTS.to_vec();
    let text = |arena_len: usize, slots: &[super::prefill_plan::ChunkSlot]| {
        let error = check_view_list(arena_len, slots).expect_err("refused");
        let EngError::BadRequest(text) = &error else {
            panic!("{error:?} is not a BadRequest");
        };
        text.clone()
    };
    // The valid list (V1-V3 accepted).
    assert!(check_view_list(35, &slots).is_ok());
    // V1: the arena one short of the full list.
    let v1 = text(34, &slots);
    assert!(
        v1.contains("the arena holds 34 tensors, the list 35"),
        "{v1}"
    );
    // V2: V1 satisfied (36 and 36), one slot twice.
    let mut twice = slots.clone();
    twice.push(super::prefill_plan::ChunkSlot::Fixup);
    assert!(text(36, &twice).contains("Fixup appears twice"));
    // V3: V1 satisfied (34 and 34), one slot missing.
    let short: Vec<_> = slots
        .iter()
        .copied()
        .filter(|slot| *slot != super::prefill_plan::ChunkSlot::Gu)
        .collect();
    assert!(text(34, &short).contains("Gu is missing"));

    // V4 against the real plan of the real list.
    let list = chunk_tensors(&common_tests::dims(), 128, 82).expect("chunk_tensors");
    let tensors: Vec<_> = list.iter().map(|(_, tensor)| *tensor).collect();
    let plan = plan_arena(&tensors).expect("plan_arena");
    let actual: Vec<_> = plan
        .offsets
        .iter()
        .copied()
        .zip(tensors.iter().map(|tensor| tensor.bytes))
        .collect();
    assert!(check_view_layout(&list, &actual).is_ok());

    // V4a: `Hidden` and `X` swapped (same size), against the unswapped actual. The two
    // planned offsets differ on the host first, so the row cannot pass by accident.
    let at = |slot: super::prefill_plan::ChunkSlot| {
        list.iter()
            .position(|(known, _)| *known == slot)
            .expect("slot in list")
    };
    let (hidden, x) = (
        at(super::prefill_plan::ChunkSlot::Hidden),
        at(super::prefill_plan::ChunkSlot::X),
    );
    assert_eq!(
        list[hidden].1.bytes, list[x].1.bytes,
        "Hidden and X must be the same size for the swap row"
    );
    assert_ne!(plan.offsets[hidden], plan.offsets[x]);
    let mut swapped = list.clone();
    swapped.swap(hidden, x);
    let error = check_view_layout(&swapped, &actual).expect_err("refused");
    // The first mismatching position is the swapped list's first entry, `X`.
    assert!(
        matches!(&error, EngError::BadRequest(text) if text.contains("X is planned at")),
        "{error:?} does not name X"
    );

    // V4b: one view's bytes one 256-byte step short, its offset unchanged.
    let mut shrunk = actual.clone();
    let gu = at(super::prefill_plan::ChunkSlot::Gu);
    shrunk[gu].1 -= 256;
    let error = check_view_layout(&list, &shrunk).expect_err("refused");
    assert!(
        matches!(&error, EngError::BadRequest(text) if text.contains("Gu is planned at")),
        "{error:?} does not name Gu"
    );

    // `slot_position` on a list without `Gu`.
    let error = slot_position(&short, super::prefill_plan::ChunkSlot::Gu).expect_err("refused");
    assert!(
        matches!(&error, EngError::BadRequest(text) if text.contains("Gu is not a view")),
        "{error:?} does not refuse Gu"
    );
    assert_eq!(
        slot_position(&short, super::prefill_plan::ChunkSlot::X).unwrap(),
        x
    );
}

/// H2: the stage lists equal the plan table, and a third kind refuses.
#[test]
fn prefill_stage_order_is_the_plan_table() {
    assert_eq!(
        layer_stages(AttentionKind::DeltaNet).unwrap(),
        vec![
            Stage::NormA,
            Stage::QuantA,
            Stage::DeltaProj,
            Stage::Conv,
            Stage::Scan,
            Stage::QuantDOut,
            Stage::SsmOut,
            Stage::NormF,
            Stage::Route,
            Stage::Moe,
        ]
    );
    assert_eq!(
        layer_stages(AttentionKind::Full).unwrap(),
        vec![
            Stage::NormA,
            Stage::QuantA,
            Stage::AttnProj,
            Stage::Prep,
            Stage::Flash,
            Stage::QuantAOut,
            Stage::AttnOut,
            Stage::NormF,
            Stage::Route,
            Stage::Moe,
        ]
    );
    assert_eq!(
        resident_moe_stages(),
        [
            MoeStage::SharedQuant,
            MoeStage::SharedGate,
            MoeStage::SharedUp,
            MoeStage::SharedSwiglu,
            MoeStage::SharedDownQuant,
            MoeStage::SharedDown,
            MoeStage::Ids,
            MoeStage::Scatter,
            MoeStage::Gate,
            MoeStage::Up,
            MoeStage::Swiglu,
            MoeStage::GatherQuant,
            MoeStage::Down,
            MoeStage::Combine,
        ]
    );
    let error = layer_stages(AttentionKind::Qsa).expect_err("refused");
    assert!(
        matches!(&error, EngError::BadRequest(text) if text.contains("prefill chunk: kind")),
        "{error:?} does not refuse the kind"
    );
}

/// H3: at every ladder size, each stage's phase lies inside the lifetime of every slot it
/// reads or writes, and the phases never decrease along the embedding plus each layer list.
#[test]
fn prefill_stages_touch_only_live_slots() {
    for m_sess in M_LADDER {
        let list = chunk_tensors(&common_tests::dims(), m_sess, 82).expect("chunk_tensors");
        let lifetime = |slot: super::prefill_plan::ChunkSlot| {
            let (_, tensor) = list
                .iter()
                .find(|(known, _)| *known == slot)
                .expect("slot in list");
            (tensor.first, tensor.last)
        };
        let check = |phase: super::prefill_plan::Phase,
                     (reads, writes): (
            &[super::prefill_plan::ChunkSlot],
            &[super::prefill_plan::ChunkSlot],
        )| {
            for slot in reads.iter().chain(writes) {
                let (first, last) = lifetime(*slot);
                assert!(
                    first <= phase as usize && phase as usize <= last,
                    "m_sess {m_sess}: {slot:?} lives {first}..={last}, the stage runs at {}",
                    phase as usize
                );
            }
        };
        check(EMBED_PHASE, EMBED_SLOTS);
        for kind in [AttentionKind::DeltaNet, AttentionKind::Full] {
            let mut previous = EMBED_PHASE as usize;
            for stage in layer_stages(kind).unwrap() {
                let phase = stage_phase(stage);
                assert!(
                    phase as usize >= previous,
                    "m_sess {m_sess}: {stage:?} runs at {} after {previous}",
                    phase as usize
                );
                previous = phase as usize;
                check(phase, stage_slots(stage));
            }
        }
        for stage in resident_moe_stages() {
            check(moe_stage_phase(stage), moe_stage_slots(stage));
        }
    }
}

/// H7: the layout map and the mixed-buffer refusal. Kills 7a-M9 and 7a-Q1.
#[test]
fn q8_layouts_refuse_a_mixed_buffer() {
    for (quant, want) in [
        (QuantType::Q80, Q8_1Layout::D4),
        (QuantType::Q50, Q8_1Layout::D4),
        (QuantType::Q6K, Q8_1Layout::D4),
        (QuantType::Q4K, Q8_1Layout::Ds4),
    ] {
        assert_eq!(mmq_type_of(quant).q8_1_layout(), want, "{quant:?}");
    }
    use super::prefill_plan::ChunkSlot;
    let q8x = ChunkSlot::Q8X;
    assert!(check_q8_layouts(&[(q8x, &[QuantType::Q80; 4])]).is_ok());
    assert!(check_q8_layouts(&[(q8x, &[QuantType::Q80, QuantType::Q6K])]).is_ok());
    let error = check_q8_layouts(&[(q8x, &[QuantType::Q80, QuantType::Q4K])]).expect_err("refused");
    assert!(
        matches!(&error, EngError::BadRequest(text) if text.contains("prefill chunk: q8 layout")),
        "{error:?} does not refuse the mixed buffer"
    );
    assert!(check_q8_layouts(&[(ChunkSlot::Q8S, &[QuantType::Q4K, QuantType::Q4K])]).is_ok());
}

/// H8: the up-half sub-view follows `dims.inter`, never a literal column (MAJOR-1). Kills
/// 7a-M12.
#[test]
fn routed_up_view_follows_the_moe_inter() {
    use super::moe::MoeDims;
    for dims in [
        MoeDims::new(&common_tests::dims(), 1),
        super::native_device_rig::DIMS,
    ] {
        let bytes = dims.hidden * 2 * dims.inter * 4;
        let (offset, length) = routed_up_view(bytes, &dims).expect("view");
        assert_eq!(offset, dims.inter * 4, "inter {}", dims.inter);
        assert_eq!(length, bytes - offset);
        let swiglu = routed_swiglu(7, &dims);
        assert_eq!(offset, swiglu.up_off * 4);
        assert_eq!(offset, swiglu.ld_in * 2);
        // A slot that ends at the offset holds no up half.
        assert!(routed_up_view(offset, &dims).is_err());
    }
    // The literal values of the two rigs: Qwen inter 512, the small rig inter 256.
    let qwen = MoeDims::new(&common_tests::dims(), 1);
    assert_eq!(routed_up_view(8 * 1024, &qwen).unwrap().0, 2_048);
    assert_eq!(
        routed_up_view(8 * 1024, &super::native_device_rig::DIMS)
            .unwrap()
            .0,
        1_024
    );
}

/// H1: every builder at M = 130, p0 = 1,000, cap = 4,096 equals the plan's literal values
/// field by field. Kills 7a-M1 and 7a-M1b.
#[test]
fn prefill_call_shapes_are_the_plan_values() {
    use super::mixer_geometry::MixerGeometry;
    use super::prefill::{
        attn_proj_cols, chunk_attn_geometry, chunk_attn_shape, chunk_combine_shape,
        chunk_conv_shape, chunk_mmid_shape, chunk_prep_shape, chunk_router_shape, chunk_scan_shape,
        delta_proj_cols, shared_cols, shared_swiglu,
    };
    use crate::family::gguf_model::Role;
    use crate::kernels::cuda::prefill::attn_prefill::{
        CPF_ATTN_PART_CAP, cpf_attn_launch_plan, cpf_attn_part_bytes,
    };

    let dims = common_tests::dims();
    let g = MixerGeometry::new(&dims).expect("geometry");
    let rows = ChunkRows::new(1_000, 130, 4_096, 256).expect("rows");
    assert_eq!(rows.p0(), 1_000);
    assert_eq!(rows.m(), 130);
    for (role, n, off) in delta_proj_cols(&g) {
        let want = match role {
            Role::AttnQkv => (8_192, 0),
            Role::AttnGate => (4_096, 8_192),
            Role::SsmBeta => (32, 12_288),
            Role::SsmAlpha => (32, 12_320),
            other => panic!("unexpected role {other:?}"),
        };
        assert_eq!((n, off), want, "{role:?}");
    }
    for (role, n, off) in attn_proj_cols(&g) {
        let want = match role {
            Role::AttnQ => (8_192, 0),
            Role::AttnK => (512, 8_192),
            Role::AttnV => (512, 8_704),
            other => panic!("unexpected role {other:?}"),
        };
        assert_eq!((n, off), want, "{role:?}");
    }
    let conv = chunk_conv_shape(&g, 130);
    assert_eq!((conv.m, conv.channels, conv.ld), (130, 8_192, 12_352));
    let scan = chunk_scan_shape(&g, &rows, 4_096);
    assert_eq!(
        (
            scan.m,
            scan.p0,
            scan.ctx_tokens,
            scan.n_k_heads,
            scan.n_v_heads,
            scan.head_dim,
            scan.conv_ld,
            scan.ld,
            scan.z_off,
            scan.beta_off,
            scan.alpha_off,
        ),
        (
            130, 1_000, 4_096, 16, 32, 128, 8_192, 12_352, 8_192, 12_288, 12_320
        )
    );
    let prep = chunk_prep_shape(&g, &rows, 4_096);
    assert_eq!(
        (
            prep.m,
            prep.n_head,
            prep.n_kv_head,
            prep.head_dim,
            prep.n_rot,
            prep.rows,
            prep.ld,
            prep.p0
        ),
        (130, 16, 2, 256, 64, 4_096, 9_216, 1_000)
    );
    assert!(g.attn.rope_base.is_finite() && g.attn.rope_base > 0.0);
    let attn = chunk_attn_shape(&g, &rows, 4_096);
    assert_eq!(
        (
            attn.m,
            attn.n_head,
            attn.n_kv_head,
            attn.head_dim,
            attn.cache_rows,
            attn.ld,
            attn.p0
        ),
        (130, 16, 2, 256, 4_096, 9_216, 1_000)
    );
    assert_eq!(g.attn.scale, 0.0625);
    let geometry = chunk_attn_geometry(&g, 130);
    assert_eq!(
        (
            geometry.m,
            geometry.n_head,
            geometry.n_kv_head,
            geometry.head_dim
        ),
        (130, 16, 2, 256)
    );
    // The launch plan of stage 5a takes kv_len = p0 + m = 1,130.
    let plan = cpf_attn_launch_plan(geometry, 1_130, 0, 82).expect("launch plan");
    assert!(plan.max_splits >= 1);
    assert!(
        cpf_attn_part_bytes(geometry, plan).unwrap() <= CPF_ATTN_PART_CAP,
        "the launch plan's part bytes pass the flat cap"
    );

    let moe = super::moe::MoeDims::new(&dims, 130);
    let router = chunk_router_shape(&moe, 130);
    assert_eq!(
        (router.m, router.hidden, router.n_expert, router.top_k),
        (130, 2_048, 256, 8)
    );
    let combine = chunk_combine_shape(&moe, 130);
    assert_eq!(
        (combine.m, combine.n, combine.top_k, combine.ld_down),
        (130, 2_048, 8, 2_048)
    );
    let mmid = chunk_mmid_shape(&moe, 130);
    assert_eq!((mmid.n_experts, mmid.m, mmid.top_k), (256, 130, 8));
    assert_eq!(shared_cols(&moe), [(1_024, 0), (1_024, 512)]);
    let shared = shared_swiglu(130, &moe);
    assert_eq!(
        (
            shared.rows,
            shared.n_ff,
            shared.up_off,
            shared.ld_in,
            shared.ld_out
        ),
        (130, 512, 512, 1_024, 512)
    );
    let routed = super::prefill::routed_swiglu(130, &moe);
    assert_eq!(
        (
            routed.rows,
            routed.n_ff,
            routed.up_off,
            routed.ld_in,
            routed.ld_out
        ),
        (1_040, 512, 512, 1_024, 512)
    );
    // The Q4_K per-expert strides of the routed stacks.
    assert_eq!(QuantType::Q4K.weight_bytes(512, 2_048), Some(589_824));
    assert_eq!(QuantType::Q4K.weight_bytes(2_048, 512), Some(589_824));
}

/// H4: at every ladder size and a spread of chunk rows, every call's byte need fits the
/// session slot of the stage that takes it (M1's prefix rule), from the wrappers' own pure
/// sizing. The 13b row uses the derived offset, never a literal column.
#[test]
fn prefill_stage_needs_fit_the_session_slots() {
    use super::mixer_geometry::MixerGeometry;
    use super::prefill::{
        attn_proj_cols, chunk_attn_geometry, chunk_conv_shape, delta_proj_cols, routed_swiglu,
        routed_up_view, shared_swiglu,
    };
    use crate::kernels::cuda::prefill::attn_prefill::{
        CPF_ATTN_PART_CAP, cpf_attn_launch_plan, cpf_attn_part_bytes,
    };
    use crate::kernels::cuda::prefill::gdn_scan_seq::scan_bytes;
    use crate::kernels::cuda::prefill::mmq_dense::dense_needs;
    use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes as q8_1_mmq_bytes;
    use crate::kernels::cuda::prefill::mmq_size::{
        expert_bounds_ints, ids_dst_ints, ids_src1_ints,
    };
    use crate::kernels::cuda::prefill::swiglu_rows::swiglu_rows_needs;
    use crate::kernels::cuda::resident::add_norm_pack::row_bytes;
    use crate::kernels::cuda::resident::attn_prep::q_bytes;
    use crate::kernels::cuda::resident::gdn_conv::conv_bytes;
    use crate::kernels::cuda::resident::moe_combine::combine_down_bytes;
    use crate::kernels::cuda::resident::router::router_logits_bytes;
    use crate::kernels::cuda::xf_bytes;

    let dims = common_tests::dims();
    let g = MixerGeometry::new(&dims).expect("geometry");
    let cap = 4_096;
    for m_sess in M_LADDER {
        let planned = |slot: super::prefill_plan::ChunkSlot| {
            chunk_tensors(&dims, m_sess, 82)
                .expect("chunk_tensors")
                .into_iter()
                .find(|(known, _)| *known == slot)
                .expect("slot")
                .1
                .bytes
        };
        use super::prefill_plan::ChunkSlot as S;
        for m in [2, 7, 16, 17, 49, 128, 130, m_sess - 1, m_sess] {
            if m > m_sess {
                continue;
            }
            let rows = ChunkRows::new(0, m, cap, m_sess).expect("rows");
            let (h, d, a) = (g.hidden, &g.delta, &g.attn);
            let moe = super::moe::MoeDims::new(&dims, m);
            let routed = m * moe.top_k;
            let row = |n| row_bytes(m, n).expect("row bytes");
            let q8 = |k, cols| q8_1_mmq_bytes(k, cols).expect("q8 bytes");
            let fits = |need: usize, slot: super::prefill_plan::ChunkSlot, what: &str| {
                assert!(
                    need <= planned(slot),
                    "m_sess {m_sess} m {m}: {what} needs {need}, {slot:?} holds {}",
                    planned(slot)
                );
            };
            // P1.
            fits(m * 4, S::IdsTok, "embed ids");
            fits(row(h), S::Hidden, "embed x");
            // 1 NormA and 8 NormF.
            fits(row(h), S::Hidden, "norm h");
            fits(xf_bytes(m, h).expect("xf"), S::XfA, "NormA xf");
            fits(xf_bytes(m, h).expect("xf"), S::XfF, "NormF xf");
            fits(row(h), S::X, "NormA norm_out");
            fits(row(h), S::X2, "NormF norm_out");
            // 2 QuantA (both kinds share the buffer).
            fits(row(h), S::X, "quant x");
            fits(q8(h, m), S::Q8X, "QuantA y");
            // 3d DeltaProj.
            for (_role, n, off) in delta_proj_cols(&g) {
                let needs = dense_needs(QuantType::Q80, h, n, m, d.ld, off).expect("dense");
                fits(needs.y, S::Q8X, "DeltaProj q8");
                fits(needs.dst, S::ProjD, "DeltaProj dst");
            }
            // 4d Conv, 5d Scan, 6d, 7d.
            let conv = conv_bytes(chunk_conv_shape(&g, m)).expect("conv");
            fits(conv.proj, S::ProjD, "conv proj");
            fits(conv.out, S::ConvOut, "conv out");
            let scan = scan_bytes(super::prefill::chunk_scan_shape(&g, &rows, cap)).expect("scan");
            fits(scan.conv, S::ConvOut, "scan conv");
            fits(scan.proj, S::ProjD, "scan proj");
            fits(scan.o_out, S::OOut, "scan o_out");
            fits(row(d.value_width), S::OOut, "QuantDOut x");
            fits(q8(d.value_width, m), S::Q8OD, "QuantDOut y");
            let ssm_out = dense_needs(QuantType::Q80, d.value_width, h, m, h, 0).expect("SsmOut");
            fits(ssm_out.y, S::Q8OD, "SsmOut q8");
            fits(ssm_out.dst, S::Delta, "SsmOut dst");
            // 3a, 4a, 5a, 6a, 7a.
            for (_role, n, off) in attn_proj_cols(&g) {
                let needs = dense_needs(QuantType::Q80, h, n, m, a.ld, off).expect("dense");
                fits(needs.y, S::Q8X, "AttnProj q8");
                fits(needs.dst, S::ProjA, "AttnProj dst");
            }
            fits(row(a.ld), S::ProjA, "prep proj");
            fits(q_bytes(m, a.heads_q), S::Q, "prep q");
            let geometry = chunk_attn_geometry(&g, m);
            let plan = cpf_attn_launch_plan(geometry, m, 0, 82).expect("launch plan");
            assert!(plan.max_splits >= 1);
            assert!(
                cpf_attn_part_bytes(geometry, plan).unwrap() <= CPF_ATTN_PART_CAP,
                "m_sess {m_sess} m {m}: the launch plan's part passes the flat cap"
            );
            fits(row(a.q_width), S::AttnOut, "flash out");
            fits(q_bytes(m, a.heads_q), S::Q, "flash q");
            fits(row(a.ld), S::ProjA, "flash proj");
            fits(row(a.q_width), S::AttnOut, "QuantAOut x");
            fits(q8(a.q_width, m), S::Q8OA, "QuantAOut y");
            let attn_out = dense_needs(QuantType::Q80, a.q_width, h, m, h, 0).expect("AttnOut");
            fits(attn_out.y, S::Q8OA, "AttnOut q8");
            fits(attn_out.dst, S::Delta, "AttnOut dst");
            // 9 Route.
            let router =
                router_logits_bytes(super::prefill::chunk_router_shape(&moe, m)).expect("router");
            fits(router, S::Logits, "router logits");
            fits(m * moe.top_k * 4, S::TopkId, "router ids");
            fits(m * moe.top_k * 4, S::TopkWeight, "router weights");
            fits(m * 4, S::SharedGate, "router shared gate");
            // 10a-f Shared.
            fits(q8(h, m), S::Q8X2, "shared quant y");
            fits(row(h), S::X2, "shared quant x");
            for (_, off) in super::prefill::shared_cols(&moe) {
                let needs = dense_needs(
                    QuantType::Q80,
                    h,
                    moe.shared_inter,
                    m,
                    2 * moe.shared_inter,
                    off,
                )
                .expect("shared gemm");
                fits(needs.dst, S::GuSh, "shared dst");
            }
            let shared = swiglu_rows_needs(&shared_swiglu(m, &moe)).expect("shared swiglu");
            fits(shared.gu, S::GuSh, "shared swiglu gu");
            fits(shared.act, S::ActSh, "shared swiglu act");
            fits(q8(moe.shared_inter, m), S::Q8Ash, "shared down quant y");
            let shared_down =
                dense_needs(QuantType::Q80, moe.shared_inter, h, m, h, 0).expect("shared down");
            fits(shared_down.dst, S::ShOut, "shared down dst");
            // 11 Ids.
            fits(m * moe.top_k * 4, S::IdsG, "ids guard");
            fits(ids_src1_ints(routed) * 4, S::IdsSrc1, "ids src1");
            fits(
                ids_dst_ints(routed, m).expect("ids dst") * 4,
                S::IdsDst,
                "ids dst",
            );
            fits(
                expert_bounds_ints(moe.n_expert).expect("bounds") * 4,
                S::ExpertBounds,
                "bounds",
            );
            // 12 Scatter.
            fits(q8(h, routed), S::Q8S, "scatter y");
            fits(row(h), S::X2, "scatter x");
            // 13a/13b Gate and Up (13b through the derived view).
            let gate_dst = (routed - 1) * (2 * moe.inter) + moe.inter;
            fits(gate_dst * 4, S::Gu, "gate dst");
            let (off, len) = routed_up_view(planned(S::Gu), &moe).expect("up view");
            let up_dst = (routed - 1) * (2 * moe.inter) + moe.inter;
            assert!(
                up_dst * 4 <= len,
                "m_sess {m_sess} m {m}: the up view of {off} bytes is short"
            );
            fits(q8(h, routed), S::Q8S, "gate/up y");
            // 14 Swiglu, 15 GatherQuant, 16 Down, 17 Combine.
            let act = swiglu_rows_needs(&routed_swiglu(m, &moe)).expect("routed swiglu");
            fits(act.gu, S::Gu, "routed swiglu gu");
            fits(act.act, S::Act, "routed swiglu act");
            fits(q8(moe.inter, routed), S::Q8Act, "gather y");
            fits(
                row_bytes(routed, moe.inter).expect("rows"),
                S::Act,
                "gather x rows",
            );
            fits((routed * h) * 4, S::Down, "down dst");
            let combine =
                combine_down_bytes(&super::prefill::chunk_combine_shape(&moe, m)).expect("combine");
            fits(combine, S::Down, "combine down");
        }
    }
}

/// H10: one row per guard of `check_bufs`, each with the others satisfied, and the accepting
/// edges. Kills 7a-B1..B5.
#[test]
fn bufs_refuse_each_guard() {
    use super::prefill::{BufFacts, check_bufs};
    use crate::kernels::cuda::prefill::{
        attn_prefill::CPF_ATTN_PART_CAP, mmq_size::fixup_floats_max,
    };
    let floor = fixup_floats_max(82).unwrap() * 4;
    let facts =
        |m_sess: usize, part_bytes: usize, part_offset: usize, nsm: usize, fixup: usize| BufFacts {
            m_sess,
            part_bytes,
            part_offset,
            nsm,
            fixup_bytes: fixup,
        };
    let valid = facts(256, CPF_ATTN_PART_CAP, 0, 82, floor);
    assert!(check_bufs(&valid).is_ok());
    // The accepting edges of B1.
    assert!(check_bufs(&facts(1, CPF_ATTN_PART_CAP, 0, 82, floor)).is_ok());
    assert!(check_bufs(&facts(2_048, CPF_ATTN_PART_CAP, 0, 82, floor)).is_ok());
    let refuse = |f: &BufFacts, want: &str| {
        let error = check_bufs(f).expect_err("refused");
        assert!(
            matches!(&error, EngError::BadRequest(text) if text.contains(want)),
            "{error:?} does not contain {want:?}"
        );
    };
    // B1, both edges.
    refuse(
        &facts(0, CPF_ATTN_PART_CAP, 0, 82, floor),
        "prefill chunk: m_sess",
    );
    refuse(
        &facts(2_049, CPF_ATTN_PART_CAP, 0, 82, floor),
        "prefill chunk: m_sess",
    );
    // B2.
    refuse(
        &facts(256, CPF_ATTN_PART_CAP - 1, 0, 82, floor),
        "below the",
    );
    // B3.
    refuse(
        &facts(256, CPF_ATTN_PART_CAP, 8, 82, floor),
        "not 16-byte aligned",
    );
    // B4.
    refuse(
        &facts(256, CPF_ATTN_PART_CAP, 0, 0, floor),
        "prefill chunk: nsm",
    );
    // B5.
    refuse(
        &facts(256, CPF_ATTN_PART_CAP, 0, 82, floor - 4),
        "prefill chunk: fixup",
    );
}

/// The valid 40-layer `ModelFacts` fixture of H11 and the A1 rows: kinds and roles from the
/// real table and geometry, every weight Native.
fn model_fixture() -> super::prefill::ModelFacts {
    use super::layer_table::LayerTable;
    use super::mixer_geometry::MixerGeometry;
    use super::prefill::LayerFacts;
    use crate::kernels::cuda::prefill::gdn_scan_seq::GDN_SCAN_EPS;

    let dims = common_tests::dims();
    let geometry = MixerGeometry::new(&dims).expect("geometry");
    let table = LayerTable::new(&dims, &common_tests::inventory()).expect("table");
    let layer = |l: usize| {
        let kind = table.slot(l).expect("slot").kind;
        let dense = geometry
            .dense_specs(WeightLayout::Native, kind)
            .expect("specs")
            .into_iter()
            .map(|spec| (spec.role, Some(QuantType::Q80)))
            .collect();
        let stack = |role| super::prefill::StackFacts {
            role,
            quant: Some(QuantType::Q4K),
            experts: Some(256),
            n_rows: match role {
                crate::family::gguf_model::Role::FfnDownExps => 256 * 2_048,
                _ => 256 * 512,
            },
            k: match role {
                crate::family::gguf_model::Role::FfnDownExps => 512,
                _ => 2_048,
            },
        };
        LayerFacts {
            kind,
            dense,
            stacks: [
                stack(crate::family::gguf_model::Role::FfnGateExps),
                stack(crate::family::gguf_model::Role::FfnUpExps),
                stack(crate::family::gguf_model::Role::FfnDownExps),
            ],
            shared: [
                (
                    crate::family::gguf_model::Role::FfnGateShexp,
                    Some(QuantType::Q80),
                ),
                (
                    crate::family::gguf_model::Role::FfnUpShexp,
                    Some(QuantType::Q80),
                ),
                (
                    crate::family::gguf_model::Role::FfnDownShexp,
                    Some(QuantType::Q80),
                ),
            ],
        }
    };
    super::prefill::ModelFacts {
        eps_bits: GDN_SCAN_EPS.to_bits(),
        hidden: 2_048,
        inter: 512,
        experts: 256,
        layers: (0..40).map(layer).collect(),
    }
}

/// H11: the fixture is accepted, then one row per guard of `check_model`, each breaking one
/// fact of layer 3 unless the guard is global. Kills 7a-D1..D9.
#[test]
fn model_refuses_each_guard() {
    use super::prefill::check_model;
    use crate::kernels::cuda::prefill::gdn_scan_seq::GDN_SCAN_EPS;

    let refuse = |facts: &super::prefill::ModelFacts, layers: usize, want: &str| {
        let error = check_model(facts, layers).expect_err("refused");
        assert!(
            matches!(&error, EngError::BadRequest(text) if text.contains(want)),
            "{error:?} does not contain {want:?}"
        );
    };
    let mut valid = model_fixture();
    assert!(check_model(&valid, 40).is_ok());

    // D1: the expected layer count.
    refuse(&valid, 39, "prefill chunk: layers");
    // D2: the epsilon.
    valid.eps_bits = 1.0e-5_f32.to_bits();
    refuse(&valid, 40, "prefill chunk: eps");
    valid.eps_bits = GDN_SCAN_EPS.to_bits();
    // Layer-3 rows (the attention layer of the real table).
    let mut facts = valid.clone();
    facts.layers[3].kind = AttentionKind::Qsa;
    refuse(&facts, 40, "prefill chunk: kind");
    let mut facts = valid.clone();
    facts.layers[3].dense[0].1 = None; // AttnQ of the attention layer.
    refuse(&facts, 40, "layout: layer 3 AttnQ");
    let mut facts = valid.clone();
    facts.layers[0].dense[3].1 = None; // SsmAlpha of the DeltaNet layer.
    refuse(&facts, 40, "layer 0 SsmAlpha");
    let mut facts = valid.clone();
    facts.layers[3].stacks[2].quant = None; // FfnDownExps.
    refuse(&facts, 40, "stack FfnDownExps");
    let mut facts = valid.clone();
    facts.layers[3].shared[1].1 = None; // FfnUpShexp.
    refuse(&facts, 40, "shared FfnUpShexp");
    let mut facts = valid.clone();
    facts.layers[3].stacks[0].experts = Some(255); // FfnGateExps.
    refuse(&facts, 40, "prefill chunk: experts");
    let mut facts = valid.clone();
    facts.layers[3].stacks[2].n_rows -= 2_048; // one matrix short.
    refuse(&facts, 40, "prefill chunk: stack shape");
    let mut facts = valid.clone();
    facts.layers[3].dense[1].1 = Some(QuantType::Q4K); // AttnK beside Q8_0 siblings.
    refuse(&facts, 40, "prefill chunk: q8 layout");
}
/// The A1 rows: one row per refusal of `ChunkRun::new`'s check.
#[test]
fn run_refuses_each_bound() {
    use super::prefill::{RunFacts, check_run};
    assert!(
        check_run(&RunFacts {
            m: 130,
            m_sess: 256,
            model_layers: 40,
            state_layers: 40
        })
        .is_ok()
    );
    let refuse = |f: &RunFacts, want: &str| {
        let error = check_run(f).expect_err("refused");
        assert!(
            matches!(&error, EngError::BadRequest(text) if text.contains(want)),
            "{error:?} does not contain {want:?}"
        );
    };
    // The rows do not fit the buffers.
    refuse(
        &RunFacts {
            m: 257,
            m_sess: 256,
            model_layers: 40,
            state_layers: 40,
        },
        "prefill chunk: rows",
    );
    refuse(
        &RunFacts {
            m: 0,
            m_sess: 256,
            model_layers: 40,
            state_layers: 40,
        },
        "prefill chunk: rows",
    );
    // The model holds more layers than the state.
    refuse(
        &RunFacts {
            m: 130,
            m_sess: 256,
            model_layers: 41,
            state_layers: 40,
        },
        "prefill chunk: layers",
    );
    // A `leading(n)` prefix model of the state is accepted.
    assert!(
        check_run(&RunFacts {
            m: 130,
            m_sess: 256,
            model_layers: 5,
            state_layers: 40
        })
        .is_ok()
    );
}

/// Native attention tiles cover every bulk row once and retain absolute causal positions.
#[test]
fn kv8_attention_tiles_keep_bounds_and_absolute_offsets() {
    use super::prefill::kv8_attention_tile;
    for m in 1..=1024 {
        let mut covered = 0;
        for row in (0..m).step_by(16) {
            let (rows, length) = kv8_attention_tile(972, m, row).unwrap();
            assert_eq!(rows, (m - row).min(16));
            assert_eq!(length, (973 + row) as i32);
            assert!((1..=16).contains(&rows));
            assert!(row + rows <= m);
            covered += rows;
        }
        assert_eq!(covered, m);
    }
    assert_eq!(kv8_attention_tile(0, 16, 0).unwrap(), (16, 1));
    assert_eq!(kv8_attention_tile(32, 17, 16).unwrap(), (1, 49));
    for (p0, m, row) in [
        (0, 0, 0),
        (0, 1025, 0),
        (0, 1024, 1024),
        (0, 16, 16),
        (usize::MAX, 1, 0),
        (i32::MAX as usize, 1, 0),
        (i32::MAX as usize - 15, 16, 0),
        (i32::MAX as usize - 1023, 1024, 0),
    ] {
        assert!(kv8_attention_tile(p0, m, row).is_err(), "{p0}/{m}/{row}");
    }
    assert_eq!(
        kv8_attention_tile(i32::MAX as usize - 16, 16, 0).unwrap(),
        (16, i32::MAX - 15),
    );
    assert_eq!(
        kv8_attention_tile(i32::MAX as usize - 1024, 1024, 1023).unwrap(),
        (1, i32::MAX),
    );
}

/// Sliced projection gates retain ld; packed Q/output rows use their own smaller stride.
#[test]
fn kv8_attention_row_extents_stay_inside_live_slots() {
    use super::prefill::kv8_row_extent;
    let (q_stride, proj_stride, gate) = (4096 * 4, 9216 * 4, 8192 * 4);
    assert_eq!(
        kv8_row_extent(512 * q_stride, 496, 16, q_stride, q_stride).unwrap(),
        (496 * q_stride, 16 * q_stride),
    );
    let end = 511 * proj_stride + gate;
    assert_eq!(
        kv8_row_extent(end, 496, 16, proj_stride, gate).unwrap(),
        (496 * proj_stride, 15 * proj_stride + gate),
    );
    assert_eq!(
        kv8_row_extent(17 * q_stride, 16, 1, q_stride, q_stride).unwrap(),
        (16 * q_stride, q_stride),
    );
    for (capacity, row, rows, stride, live) in [
        (end - 1, 496, 16, proj_stride, gate),
        (usize::MAX, usize::MAX, 1, 4, 4),
        (usize::MAX, 0, usize::MAX, 4, 4),
        (64, 0, 0, 4, 4),
        (64, 0, 1, 4, 5),
        (64, 0, 1, 4, 0),
    ] {
        assert!(kv8_row_extent(capacity, row, rows, stride, live).is_err());
    }
}

/// The fixed arena holds max-16 attention scratch plus disjoint all-M temporary K/V and word.
#[test]
fn kv8_attention_part_capacity_is_bounded_and_disjoint() {
    use super::prefill::kv8_scratch_layout;
    use crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_PART_CAP;
    for m in 1..=1024 {
        let (part, k, v, rows) = kv8_scratch_layout(m, CPF_ATTN_PART_CAP).unwrap();
        assert_eq!(part, m.min(16) * 2 * 64 * 8 * (2 + 256) * 4);
        assert_eq!(rows, m * 2 * 256 * 4);
        assert_eq!(k, part);
        assert_eq!(v, k + rows);
        assert!(v + rows <= CPF_ATTN_PART_CAP - 4);
    }
    let (part, _, v, rows) = kv8_scratch_layout(512, CPF_ATTN_PART_CAP).unwrap();
    assert_eq!(part, 16_908_288);
    assert_eq!(v + rows, 19_005_440);
    let (part, _, v, rows) = kv8_scratch_layout(1024, CPF_ATTN_PART_CAP).unwrap();
    assert_eq!(part, 16_908_288);
    assert_eq!(v + rows, 21_102_592);
    assert!(kv8_scratch_layout(1024, CPF_ATTN_PART_CAP - 1).is_err());
    assert!(kv8_scratch_layout(0, CPF_ATTN_PART_CAP).is_err());
    assert!(kv8_scratch_layout(1025, CPF_ATTN_PART_CAP).is_err());
}

/// H12: each stage passes the prefill word it was given (revision 3, check 6). Mask 8 is
/// unreachable through the body — the wrappers' own host checks refuse a bad p0/M first —
/// and a dropped router or guard range hides behind the other's mask 1 in DV3, so the wiring
/// is pinned on the source text of `prefill.rs` (the `start_pins` style): comments stripped,
/// whitespace collapsed, each status argument exactly once in its own stage function.
/// Kills 7a-S1..S4.
#[test]
fn prefill_stages_pass_the_prefill_word() {
    use super::start_pins::{pinned, section_of};
    let source = include_str!("prefill.rs");
    let body = |start: &str| pinned(section_of(source, "prefill.rs", start));
    let count = |text: &str, needle: &str| text.matches(needle).count();
    // The scan, the preparation and the flash take the resolved word.
    for stage in ["fn delta_scan(", "fn attn_prep(", "fn attn_flash_fp16("] {
        let text = body(stage);
        assert_eq!(
            count(&text, "status:Some(status)"),
            1,
            "{stage} does not take the resolved word exactly once"
        );
    }
    // The ids guard takes the block's word; the router writes into it.
    for stage in ["fn run_chunk_routed(", "fn diagnostic_routed("] {
        let moe = body(stage);
        assert_eq!(
            count(&moe, "status:Some(io.status)"),
            1,
            "{stage}: the ids guard"
        );
    }
    let route = body("fn run_chunk_route(");
    assert_eq!(count(&route, "status:io.status"), 1, "the router word");
    // The embedding takes the resolved word by shorthand.
    let embed = body("fn chunk_embed(");
    assert_eq!(count(&embed, "status,"), 1, "the embedding");
}
