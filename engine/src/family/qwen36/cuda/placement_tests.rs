//! Host-only planner qualification: asymmetric device profiles (the
//! 10 GB RTX 3080 and 12 GB RTX 3060 pair), overcommitment refusal, and
//! conservative tie-breaks. No GPU needed: profiles are injected.

use super::card_load::ModelWeights;
use super::dims::{KvMode, StepDims};
use super::start::step_options;
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::qwen36::cuda::placement::{
    DeviceProfile, PlacementPlan, PlacementPolicy, PlacementRequest, plan_placement,
};
use crate::family::qwen36::cuda::shard::{ShardRuns, ShardTable, VALUE_BLOCK};
use crate::family::qwen36::host::HostShape;
use crate::family::{AttentionKind, FamilySpec, QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::weights::GgufType;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

const EIGHT_LAYERS: &[AttentionKind] = &[
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
];

/// The real 35B geometry with eight layers and a rig vocabulary, experts
/// at the full 256: about 7.5 GiB at Q4_K. The real expert width matters
/// because the 256-value inter grain is what bounds any unequal split.
fn scaled_spec() -> FamilySpec {
    FamilySpec {
        layers: 8,
        vocab: 4096,
        layers_attention: EIGHT_LAYERS,
        ..QWEN3_6_35B_A3B
    }
}

fn rtx_3080_10g(free: u64) -> DeviceProfile {
    DeviceProfile {
        device: 0,
        name: "RTX 3080 10 GB".to_owned(),
        total_bytes: 10 * GIB,
        free_bytes: free,
        multiprocessors: 68,
        memory_bandwidth_bytes: 760_000_000_000,
        can_access: vec![true, true],
    }
}

fn rtx_3060_12g(free: u64) -> DeviceProfile {
    DeviceProfile {
        device: 1,
        name: "RTX 3060 12 GB".to_owned(),
        total_bytes: 12 * GIB,
        free_bytes: free,
        multiprocessors: 28,
        memory_bandwidth_bytes: 360_000_000_000,
        can_access: vec![true, true],
    }
}

fn plan_for(
    spec: &FamilySpec,
    profiles: &[DeviceProfile],
    quant: GgufType,
) -> Result<PlacementPlan, EngError> {
    plan_at(spec, profiles, quant, 4096, KvMode::Int8G32, 4)
}

fn plan_at(
    spec: &FamilySpec,
    profiles: &[DeviceProfile],
    quant: GgufType,
    context: u32,
    mode: KvMode,
    slots: usize,
) -> Result<PlacementPlan, EngError> {
    let shape = HostShape::from_spec(spec, 64)?;
    let inventory = Inventory::for_spec(spec)?;
    let policy = PlacementPolicy::default();
    let dims = StepDims::new(&shape, &inventory)?;
    let mut options = step_options(context, WeightLayout::Native, mode)?;
    options.delta_slots = slots;
    let weights = ModelWeights::uniform(&inventory, quant)?;
    let request = PlacementRequest {
        dims: &dims,
        options: &options,
        weights: &weights,
        profiles,
        policy: &policy,
    };
    plan_placement(&request)
}

fn assert_no_overcommit(plan: &PlacementPlan) {
    for card in &plan.receipt.cards {
        assert!(
            card.planned_bytes + card.safety_reserve_bytes <= card.free_bytes,
            "{} overcommits: planned {} + reserve {} > free {}",
            card.name,
            card.planned_bytes,
            card.safety_reserve_bytes,
            card.free_bytes
        );
    }
}

/// The headline qualification: on a 10 GB 3080 + 12 GB 3060 pair, the
/// fast small card carries MORE of the sliceable mass than the slow big
/// card. Capacity alone would say the opposite; bandwidth is the only
/// distinguishing evidence, and the receipt says which class it is.
/// Qualified at Q8_0: its 32-value grain leaves the expert width 16
/// legal grains, so a bandwidth split is expressible at all (at Q4_K the
/// 256-value superblock pins the experts even; see the parity test).
#[test]
fn asymmetric_pair_splits_by_bandwidth_not_by_capacity() {
    let spec = scaled_spec();
    let profiles = [rtx_3080_10g(9_500 * MIB), rtx_3060_12g(11_000 * MIB)];
    let plan = plan_for(&spec, &profiles, GgufType::Q80).expect("the scaled model fits the pair");
    let shares: Vec<f64> = plan
        .receipt
        .cards
        .iter()
        .map(|c| c.share.unwrap())
        .collect();
    assert!(
        shares[0] > 0.55 && shares[0] > shares[1] + 0.05,
        "expected a clearly unequal split favoring the 3080, got {shares:?}; receipt {}",
        plan.receipt.to_json()
    );
    assert_no_overcommit(&plan);
    for run in plan.runs.kv_rows.iter().chain(&plan.runs.inter) {
        assert!(run.len().is_multiple_of(VALUE_BLOCK));
    }
    assert_eq!(plan.receipt.chosen, "bandwidth-waterfill");
    assert_eq!(plan.receipt.evidence, "device-properties");
}

/// The honest geometric limit, recorded: at Q4_K the 256-value superblock
/// gives the real 512-wide expert intermediate only two legal grains, so
/// experts pin 50/50 on any two-card plan. The planner still tilts every
/// freely splittable dim toward the faster card and still wins on makespan.
#[test]
fn at_q4k_the_real_expert_width_pins_experts_near_even() {
    let spec = scaled_spec();
    let profiles = [rtx_3080_10g(9_500 * MIB), rtx_3060_12g(11_000 * MIB)];
    let plan = plan_for(&spec, &profiles, GgufType::Q4K).expect("the Q4_K model fits the pair");
    let shares: Vec<f64> = plan
        .receipt
        .cards
        .iter()
        .map(|c| c.share.unwrap())
        .collect();
    assert!(
        (0.5..0.53).contains(&shares[0]),
        "Q4_K pins experts even: {shares:?}"
    );
    assert_eq!(plan.receipt.chosen, "bandwidth-waterfill");
    assert_eq!(plan.runs.inter, vec![0..256, 256..512]);
}

/// When the fast card is short on VRAM, the waterfill clamps it to its
/// budget and hands the remainder to the slow card — still no
/// overcommitment, and the receipt records the clamp. 3.5 GiB free leaves
/// room for exactly one inter grain past the fixed costs.
#[test]
fn a_fast_card_short_on_vram_is_clamped_to_its_budget() {
    let spec = scaled_spec();
    let profiles = [rtx_3080_10g(3_500 * MIB), rtx_3060_12g(11_000 * MIB)];
    let plan = plan_for(&spec, &profiles, GgufType::Q80).expect("the pair still fits overall");
    let shares: Vec<f64> = plan
        .receipt
        .cards
        .iter()
        .map(|c| c.share.unwrap())
        .collect();
    assert!(
        shares[0] < 0.5,
        "the 3.5 GiB-free 3080 must be clamped below half, got {shares:?}"
    );
    assert!(
        plan.receipt.cards[0].clamped,
        "card 0 must be marked clamped"
    );
    assert_no_overcommit(&plan);
}

/// The real 35B model does not fit a 10+12 GB pair at Q4_K; the planner
/// must refuse with numbers, not squeeze or split unevenly to fake fit.
#[test]
fn the_full_35b_is_refused_on_a_10_and_12_gib_pair() {
    let profiles = [rtx_3080_10g(9_500 * MIB), rtx_3060_12g(11_000 * MIB)];
    let err = plan_for(&QWEN3_6_35B_A3B, &profiles, GgufType::Q4K)
        .expect_err("the full model cannot fit 20.5 GiB of free VRAM");
    let message = format!("{err:?}");
    assert!(
        message.contains("placement")
            && (message.contains("no room") || message.contains("no feasible")),
        "refusal must name the cause: {message}"
    );
}

/// Identical devices: the evidence cannot distinguish any split, so the
/// plan is today's conservative even deal and the receipt says why.
#[test]
fn equal_cards_fall_back_to_the_conservative_even_deal() {
    let spec = scaled_spec();
    let profiles = [rtx_3060_12g(11_000 * MIB), rtx_3060_12g(11_000 * MIB)];
    let plan = plan_for(&spec, &profiles, GgufType::Q80).expect("identical cards fit");
    assert_eq!(plan.receipt.chosen, "conservative-even");
    assert_eq!(plan.receipt.evidence, "conservative");
    assert!(plan.receipt.reason.contains("cannot distinguish"));
    let kv_lens: Vec<usize> = plan.runs.kv_rows.iter().map(|r| r.len()).collect();
    assert!(
        kv_lens.iter().max().unwrap() - kv_lens.iter().min().unwrap() <= VALUE_BLOCK,
        "even deal runs differ by at most one block: {kv_lens:?}"
    );
    assert_no_overcommit(&plan);
}

/// One card: whole ranges everywhere, no placement decision to make.
#[test]
fn a_single_card_keeps_the_whole_ranges() {
    let spec = scaled_spec();
    let mut solo = rtx_3080_10g(9_500 * MIB);
    solo.can_access = vec![true];
    let profiles = vec![solo];
    let plan = plan_for(&spec, &profiles, GgufType::Q80).expect("one card loads whole");
    assert_eq!(plan.receipt.chosen, "single-card");
    assert_eq!(plan.runs.value_heads, vec![0..32]);
    assert!(plan.receipt.cards[0].share.is_none());
}

/// The qualified exchange needs direct peer access; a pair without it is
/// refused instead of silently assuming a transport that does not exist.
#[test]
fn a_pair_without_peer_access_is_refused() {
    let spec = scaled_spec();
    let mut slow = rtx_3060_12g(11_000 * MIB);
    slow.can_access = vec![false, true];
    let profiles = [rtx_3080_10g(9_500 * MIB), slow];
    let err = plan_for(&spec, &profiles, GgufType::Q80).expect_err("no peer access must refuse");
    assert!(
        format!("{err:?}").contains("peer"),
        "refusal must name the peer requirement: {err:?}"
    );
}

/// Planning is deterministic: the same inputs give the same plan.
#[test]
fn planning_is_deterministic() {
    let spec = scaled_spec();
    let profiles = [rtx_3080_10g(9_500 * MIB), rtx_3060_12g(11_000 * MIB)];
    assert_eq!(
        plan_for(&spec, &profiles, GgufType::Q80).unwrap(),
        plan_for(&spec, &profiles, GgufType::Q80).unwrap()
    );
}

/// The receipt is machine-readable JSON with the commitments inside.
#[test]
fn the_receipt_is_json_with_per_card_commitments() {
    let spec = scaled_spec();
    let profiles = [rtx_3080_10g(9_500 * MIB), rtx_3060_12g(11_000 * MIB)];
    let json = plan_for(&spec, &profiles, GgufType::Q80)
        .unwrap()
        .receipt
        .to_json();
    assert!(json.starts_with('{') && json.ends_with('}'));
    assert!(json.contains("\"evidence\":\"device-properties\""));
    assert!(json.contains("\"chosen\":\"bandwidth-waterfill\""));
    assert!(json.contains("\"sliced_weights_bytes\":"));
    assert!(json.contains("\"safety_reserve_bytes\":"));
    assert!(json.contains("\"peer_access\":[true,true]"));
    assert!(json.contains("\"device\":0") && json.contains("\"device\":1"));
}

/// The planner's runs validate under `from_runs`, and hand-broken runs
/// are refused by name: coverage gaps and mid-block cuts. The baseline
/// runs come from the table's own accessors, not hand-guessed dims.
#[test]
fn from_runs_refuses_broken_runs_by_name() {
    let spec = scaled_spec();
    let shape = HostShape::from_spec(&spec, 64).unwrap();
    let good = ShardTable::new(&shape, 2, 256).unwrap();
    let runs = ShardRuns {
        value_heads: (0..2).map(|c| good.value_heads(c).unwrap()).collect(),
        query_heads: (0..2).map(|c| good.query_heads(c).unwrap()).collect(),
        kv_rows: (0..2).map(|c| good.kv_rows(c).unwrap()).collect(),
        inter: (0..2).map(|c| good.expert_inter(c).unwrap()).collect(),
        shared_inter: (0..2).map(|c| good.shared_inter(c).unwrap()).collect(),
        vocab: (0..2).map(|c| good.head_rows(c).unwrap()).collect(),
    };
    assert_eq!(
        ShardTable::from_runs(&shape, 256, &runs).unwrap().cards(),
        good.cards()
    );
    let mut gap = runs.clone();
    gap.value_heads = vec![gap.value_heads[0].clone(), 17..32];
    let err = ShardTable::from_runs(&shape, 256, &gap).unwrap_err();
    assert!(format!("{err:?}").contains("not contiguous"), "{err:?}");
    let mut cut = runs.clone();
    cut.kv_rows = vec![0..30, 30..cut.kv_rows[1].end];
    let err = ShardTable::from_runs(&shape, 256, &cut).unwrap_err();
    assert!(format!("{err:?}").contains("splits"), "{err:?}");
}

/// The exchange machinery tops out at four cards; five refuse.
#[test]
fn five_devices_are_refused() {
    let spec = scaled_spec();
    let profiles: Vec<DeviceProfile> = (0..5)
        .map(|i| DeviceProfile {
            device: i,
            name: format!("card {i}"),
            total_bytes: 12 * GIB,
            free_bytes: 11_000 * MIB,
            multiprocessors: 28,
            memory_bandwidth_bytes: 360_000_000_000,
            can_access: vec![true; 5],
        })
        .collect();
    let err = plan_for(&spec, &profiles, GgufType::Q80).expect_err("five devices must refuse");
    assert!(format!("{err:?}").contains("four"), "{err:?}");
}

/// The decode kernel serves exactly 8 query heads per K/V head, so runs
/// that deal query heads finely and K/V rows independently (both whole,
/// contiguous, full coverage) still refuse: the card holding a query
/// group must own that group's K/V head.
#[test]
fn from_runs_refuses_decoupled_kv() {
    let spec = FamilySpec {
        heads_q: 32,
        heads_kv: 4,
        ..scaled_spec()
    };
    let shape = HostShape::from_spec(&spec, 64).unwrap();
    let good = ShardTable::new(&shape, 2, 256).unwrap();
    assert_eq!(
        good.query_heads(0).unwrap(),
        0..16,
        "32 query heads deal 16/16 in whole 8-groups"
    );
    let runs = ShardRuns {
        value_heads: (0..2).map(|c| good.value_heads(c).unwrap()).collect(),
        query_heads: (0..2).map(|c| good.query_heads(c).unwrap()).collect(),
        // Whole heads, contiguous, full coverage — but 1 head on card 0
        // against its 16 query heads (2 groups).
        kv_rows: vec![
            0..shape.attn.head_dim,
            shape.attn.head_dim..4 * shape.attn.head_dim,
        ],
        inter: (0..2).map(|c| good.expert_inter(c).unwrap()).collect(),
        shared_inter: (0..2).map(|c| good.shared_inter(c).unwrap()).collect(),
        vocab: (0..2).map(|c| good.head_rows(c).unwrap()).collect(),
    };
    let err = ShardTable::from_runs(&shape, 256, &runs).unwrap_err();
    assert!(
        format!("{err:?}").contains("groups 8 query heads per K/V head"),
        "{err:?}"
    );
}

/// A shape whose attention is not the kernel's 8-per-head grouping
/// refuses at table construction, single card included.
#[test]
fn non_kernel_head_ratio_refuses() {
    let spec = FamilySpec {
        heads_q: 32,
        heads_kv: 8,
        ..scaled_spec()
    };
    let shape = HostShape::from_spec(&spec, 64).unwrap();
    let err = ShardTable::new(&shape, 1, 256).unwrap_err();
    assert!(
        format!("{err:?}").contains("8-per-head grouping"),
        "{err:?}"
    );
}

/// Qwen3.6 has 16 query heads = two whole 8-groups: three cards cannot
/// each own a group, so the plan refuses by name instead of dealing
/// fractional heads the kernel cannot run.
#[test]
fn three_cards_refuse_on_two_query_groups() {
    let spec = scaled_spec();
    let profiles: Vec<DeviceProfile> = (0..3)
        .map(|i| DeviceProfile {
            device: i,
            name: format!("card {i}"),
            total_bytes: 12 * GIB,
            free_bytes: 11_000 * MIB,
            multiprocessors: 28,
            memory_bandwidth_bytes: 360_000_000_000,
            can_access: vec![true; 3],
        })
        .collect();
    let err = plan_for(&spec, &profiles, GgufType::Q80).expect_err("3 cards must refuse");
    assert!(
        format!("{err:?}").contains("attention query heads has 2 blocks for 3 cards"),
        "{err:?}"
    );
}

/// With only two query groups, the q/KV split is 50/50 no matter what
/// bandwidth says: the receipt must say so instead of presenting a
/// bandwidth-proportional deal the kernels cannot realize.
#[test]
fn unequal_bandwidth_reports_geometry_cap() {
    let spec = scaled_spec();
    let profiles = vec![rtx_3080_10g(9_800 * MIB), rtx_3060_12g(11_600 * MIB)];
    let plan = plan_for(&spec, &profiles, GgufType::Q4K).expect("the pair fits");
    assert_eq!(plan.receipt.chosen, "bandwidth-waterfill");
    assert!(
        plan.receipt
            .geometry_capped
            .iter()
            .any(|d| d == "attention query heads"),
        "capped dims: {:?}",
        plan.receipt.geometry_capped
    );
    assert!(plan.receipt.reason.contains("geometry-capped"));
    assert!(
        !plan
            .receipt
            .geometry_capped
            .iter()
            .any(|d| d == "vocabulary"),
        "the fine-grained vocab dim must not be flagged"
    );
    // The realized q split is the forced 8/8, kernels-legal.
    assert_eq!(plan.table.query_heads(0).unwrap(), 0..8);
    assert_eq!(plan.table.query_heads(1).unwrap(), 8..16);
}

#[test]
fn the_real_shape_deals_rows_the_native_matvec_accepts() {
    use crate::family::qwen36::cuda::pinned::ROTARY_DIM;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM).expect("shape");
    // The deployed pair with tonight's host-expert relief folded in.
    let mut p0 = rtx_3060_12g(11_292 * MIB);
    p0.free_bytes += 1_342_177_280;
    let mut p1 = rtx_3080_10g(9_060 * MIB);
    p1.free_bytes += 3_489_660_928;
    let profiles = [p0, p1];
    let PlacementPlan { runs, table, .. } = plan_at(
        &QWEN3_6_35B_A3B,
        &profiles,
        GgufType::Q4K,
        131072,
        KvMode::Int8G32,
        1,
    )
    .expect("a feasible plan");
    // Every dealt dimension covers its whole total exactly, and the two
    // dimensions whose rows resolve through the native matvec move in
    // multiples of four rows on every card.
    let lens =
        |runs: &Vec<std::ops::Range<usize>>| runs.iter().map(|r| r.len()).collect::<Vec<_>>();
    for (what, got, total) in [
        ("value heads", &runs.value_heads, shape.delta.value_heads),
        ("query heads", &runs.query_heads, shape.attn.heads_q),
        (
            "kv rows",
            &runs.kv_rows,
            shape.attn.heads_kv * shape.attn.head_dim,
        ),
        ("vocab", &runs.vocab, shape.vocab),
    ] {
        let per_card = lens(got);
        assert_eq!(per_card.iter().sum::<usize>(), total, "{what} coverage");
        if what == "value heads" || what == "vocab" {
            for (card, len) in per_card.iter().enumerate() {
                assert_eq!(
                    len % 4,
                    0,
                    "{what} card {card} rows {len} break the native matvec"
                );
            }
        }
    }
    for card in 0..2 {
        let vh = table.value_heads(card).expect("value heads").len();
        assert_eq!(vh % 4, 0, "card {card} beta/alpha rows {vh}");
    }
}

#[test]
fn long_context_admission_matches_local_heads_format_and_replicated_delta() {
    let mut spec = FamilySpec {
        heads_q: 32,
        heads_kv: 4,
        ..scaled_spec()
    };
    // Four 8-head DeltaNet groups give every card a legal nonempty shard.
    spec.linear.as_mut().unwrap().key_heads = 8;
    let shape = HostShape::from_spec(&spec, 64).unwrap();
    let inventory = Inventory::for_spec(&spec).unwrap();
    let dims = StepDims::new(&shape, &inventory).unwrap();
    for n in 1..=4 {
        let profiles: Vec<_> = (0..n)
            .map(|device| DeviceProfile {
                device,
                free_bytes: 32 * GIB,
                total_bytes: 32 * GIB,
                can_access: vec![true; n],
                ..rtx_3080_10g(32 * GIB)
            })
            .collect();
        for context in [
            2048, 4096, 8192, 16_384, 16_385, 32_768, 65_536, 131_072, 262_144,
        ] {
            for mode in [KvMode::Int8G32, KvMode::Fp16] {
                let mut options = step_options(context, WeightLayout::Native, mode).unwrap();
                options.delta_slots = 2;
                let plan = plan_at(&spec, &profiles, GgufType::Q80, context, mode, 2).unwrap();
                assert_no_overcommit(&plan);
                for (i, card) in plan.receipt.cards.iter().enumerate() {
                    let heads = plan.table.query_heads(i).unwrap().len() / 8;
                    let layout =
                        super::state_layout::StateLayout::new_with_kv_heads(&dims, &options, heads)
                            .unwrap();
                    assert_eq!(card.kv_cache_bytes, layout.kv.total as u64);
                    assert_eq!(card.delta_state_bytes, layout.delta.total as u64);
                }
            }
        }
    }
}

#[test]
fn admission_honors_an_exact_two_card_cache_budget_and_refuses_a_larger_format() {
    let spec = scaled_spec();
    let mut profiles = [rtx_3080_10g(10 * GIB), rtx_3080_10g(10 * GIB)];
    let plan = plan_at(&spec, &profiles, GgufType::Q80, 262144, KvMode::Int8G32, 1).unwrap();
    for (profile, card) in profiles.iter_mut().zip(plan.receipt.cards) {
        profile.free_bytes = card.planned_bytes + card.safety_reserve_bytes;
    }
    let fit = plan_at(&spec, &profiles, GgufType::Q80, 262144, KvMode::Int8G32, 1).unwrap();
    assert_no_overcommit(&fit);
    assert!(plan_at(&spec, &profiles, GgufType::Q80, 262144, KvMode::Fp16, 1).is_err());
}

#[test]
fn mixed_tensor_types_determine_cut_alignment_and_match_the_upload_manifest() {
    let spec = scaled_spec();
    let inventory = Inventory::for_spec(&spec).unwrap();
    let shape = HostShape::from_spec(&spec, 64).unwrap();
    let dims = StepDims::new(&shape, &inventory).unwrap();
    let weights = ModelWeights::from_types(&inventory, |name, allowed| {
        let ty = if name.ends_with("ffn_down_exps.weight") {
            GgufType::Q6K
        } else {
            GgufType::Q80
        };
        Ok(if allowed.contains(&ty) {
            ty
        } else {
            allowed[0]
        })
    })
    .unwrap();
    assert_eq!(
        weights.get("blk.0.ffn_gate_exps.weight").unwrap().ty,
        GgufType::Q80
    );
    assert_eq!(
        weights.get("blk.0.ffn_down_exps.weight").unwrap().ty,
        GgufType::Q6K
    );
    assert_eq!(weights.column_block, 256);
    let plan = plan_placement(&PlacementRequest {
        dims: &dims,
        options: &step_options(2048, WeightLayout::Native, KvMode::Int8G32).unwrap(),
        weights: &weights,
        profiles: &[rtx_3080_10g(10 * GIB), rtx_3060_12g(12 * GIB)],
        policy: &PlacementPolicy::default(),
    })
    .unwrap();
    let uploads = super::card_load::checkpoint_card_loads(&plan.table, &weights).unwrap();
    for (card, manifest) in plan.receipt.cards.iter().zip(uploads) {
        assert_eq!(card.sliced_weights_bytes, manifest.sliced_bytes);
        assert_eq!(card.replicated_weights_bytes, manifest.whole_bytes);
        assert_eq!(card.embedding_bytes, manifest.solo_bytes);
    }
}

#[test]
#[ignore = "requires the Qwen3.6 35B GGUF checkpoint on disk; CPU only"]
fn checkpoint_placement_matches_the_two_card_manifest_at_256k() {
    let dir = std::env::var_os("FIFI_QWEN36_MODEL_DIR").expect("set the model directory");
    let (model, _) =
        super::start::open_gguf(std::path::Path::new(&dir), false, false, None).unwrap();
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
    let dims = StepDims::new(&shape, &inventory).unwrap();
    let weights = ModelWeights::from_checkpoint(&inventory, model.weights.header()).unwrap();
    let profiles = [
        DeviceProfile {
            total_bytes: 24 * GIB,
            ..rtx_3080_10g(22 * GIB)
        },
        DeviceProfile {
            total_bytes: 24 * GIB,
            ..rtx_3060_12g(22 * GIB)
        },
    ];
    let options = step_options(262_144, WeightLayout::Native, KvMode::Int8G32).unwrap();
    let plan = plan_placement(&PlacementRequest {
        dims: &dims,
        options: &options,
        weights: &weights,
        profiles: &profiles,
        policy: &PlacementPolicy::default(),
    })
    .unwrap();
    let manifests = super::card_load::checkpoint_card_loads(&plan.table, &weights).unwrap();
    for (i, (card, manifest)) in plan.receipt.cards.iter().zip(manifests).enumerate() {
        assert_eq!(card.kv_cache_bytes, 1_426_063_360);
        assert_eq!(card.sliced_weights_bytes, manifest.sliced_bytes);
        assert_eq!(card.replicated_weights_bytes, manifest.whole_bytes);
        assert_eq!(card.embedding_bytes, manifest.solo_bytes);
        println!(
            "card {i} checkpoint upload {} bytes, KV {} bytes",
            manifest.upload_bytes(),
            card.kv_cache_bytes
        );
    }
}
