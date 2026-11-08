//! CPU-only unit tests of the pure MoE planning: the documented op order, the slot lifetimes of
//! the real step plan, aliasing over the launch positions, and the pure refusals of
//! `check_block`. No device, no kernel.

use super::super::super::host::HostShape;
use super::super::dims::{KvMode, StepDims, StepOptions};
use super::super::shard::ShardTable;
use super::super::step_plan::{KernelScratch, Slot, StepPlan};
use super::{
    BufferFacts, CardMoeOp, ExpertFacts, MoeDims, MoeOp, OpSlots, PackFacts, PackKind, ProjFacts,
    RangeFacts, RouteFacts, StackFacts, WeightFacts, check_block, moe_sequence, moe_sequence_card,
    op_slots,
};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::resident::router::{RouterShape, router_logits_bytes};
use crate::kernels::cuda::xf_bytes;

/// The Qwen3.6-35B-A3B fixture the step-plan tests use, as checked step dims.
fn step_dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("layout record");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(&shape, &inventory).expect("checked dims")
}

pub(super) fn moe_dims(rows: usize) -> MoeDims {
    MoeDims::new(&step_dims(), rows)
}

/// The real step plan of the fixture at `rows` rows, single-expert-pack layout.
fn plan(rows: usize) -> StepPlan {
    let dims = step_dims();
    let router = RouterShape {
        m: rows,
        hidden: dims.shape.hidden,
        n_expert: dims.shape.moe.experts,
        top_k: dims.shape.moe.top_k,
    };
    let scratch = KernelScratch::new(
        &dims,
        rows,
        router_logits_bytes(router).expect("router bytes"),
        KvMode::Fp16,
    )
    .expect("scratch");
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Tiled,
    };
    StepPlan::new(&dims, rows, &options, &scratch).expect("plan")
}

/// The real step plan of the fixture at one row in the given layout.
fn plan_of(layout: WeightLayout) -> StepPlan {
    let dims = step_dims();
    let router = RouterShape {
        m: 1,
        hidden: dims.shape.hidden,
        n_expert: dims.shape.moe.experts,
        top_k: dims.shape.moe.top_k,
    };
    let scratch = KernelScratch::new(
        &dims,
        1,
        router_logits_bytes(router).expect("router bytes"),
        KvMode::Fp16,
    )
    .expect("scratch");
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    };
    StepPlan::new(&dims, 1, &options, &scratch).expect("plan")
}

/// The first and last launch position at which the sequence touches `slot`.
fn span_of(seq: &[MoeOp], slot: Slot) -> (usize, usize) {
    let touches = |slots: &OpSlots| slots.reads.contains(&slot) || slots.writes.contains(&slot);
    let at: Vec<usize> = (0..seq.len())
        .filter(|&at| touches(&op_slots(seq[at])))
        .collect();
    (
        *at.first().expect("a touched slot"),
        *at.last().expect("a touched slot"),
    )
}

/// The valid weight facts of `dims`: raw router allocations and the tiled stack geometry.
pub(super) fn weight_facts(dims: &MoeDims) -> WeightFacts {
    WeightFacts {
        router: RangeFacts {
            offset: 0,
            bytes: dims.n_expert * dims.hidden * 4,
            buffer: 0x100,
        },
        router_shared: RangeFacts {
            offset: 0,
            bytes: dims.hidden * 4,
            buffer: 0x101,
        },
        gate: StackFacts {
            rows: dims.n_expert * dims.inter,
            k: dims.hidden,
            experts: dims.n_expert,
        },
        up: StackFacts {
            rows: dims.n_expert * dims.inter,
            k: dims.hidden,
            experts: dims.n_expert,
        },
        down: StackFacts {
            rows: dims.n_expert * dims.hidden,
            k: dims.inter,
            experts: dims.n_expert,
        },
        shared_gate: ProjFacts {
            rows: dims.shared_inter,
            k: dims.hidden,
        },
        shared_up: ProjFacts {
            rows: dims.shared_inter,
            k: dims.hidden,
        },
        shared_down: ProjFacts {
            rows: dims.hidden,
            k: dims.shared_inter,
        },
    }
}

/// The valid buffer facts of `dims`: the fifteen ranges placed one behind the other in one
/// allocation, the way the arena places them apart.
pub(super) fn buffer_facts(dims: &MoeDims) -> BufferFacts {
    let mut next = 0;
    let mut at = |bytes: usize| {
        let at = RangeFacts {
            offset: next,
            bytes,
            buffer: 1,
        };
        next += bytes;
        at
    };
    let pack = |k: usize| xf_bytes(1, k).expect("a pack size");
    BufferFacts {
        route: RouteFacts {
            resid_in: at(dims.hidden * 4),
            resid_out: at(dims.hidden * 4),
            norm_row: at(dims.hidden * 4),
            logits: at(4 * (dims.n_expert + 1)),
            ids: at(4 * dims.top_k),
            weights: at(4 * dims.top_k),
            shared_gate: at(4),
            status: at(4),
        },
        expert: ExpertFacts {
            packs: PackFacts {
                kind: PackKind::Xf,
                moe: at(pack(dims.hidden)),
                down: at(pack(dims.inter)),
                shared: at(pack(dims.shared_inter)),
            },
            expert_gu: at(dims.top_k * dims.inter * 2 * 4),
            expert_down: at(dims.top_k * dims.hidden * 4),
            shared_gu: at(dims.shared_inter * 2 * 4),
            shared_out: at(dims.hidden * 4),
        },
    }
}

/// The byte span of `slot` in `plan`: offset and offset-plus-bytes.
fn bytes_of(plan: &StepPlan, slot: Slot) -> (usize, usize) {
    let start = plan.offset(slot).expect("a plan slot");
    (
        start,
        start + plan.tensors()[plan.index(slot).expect("a plan slot")].bytes,
    )
}

pub(super) fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &str) {
    let Err(error) = result else {
        panic!("{case}: expected a BadRequest, got Ok");
    };
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected a BadRequest containing {needle:?}, got {error:?}"
    );
}

/// T1: the literal documented order; the length is pinned.
#[test]
fn sequence_has_the_documented_order() {
    let want: Vec<MoeOp> = vec![
        MoeOp::Router,
        MoeOp::ExpertGate(0),
        MoeOp::ExpertUp(0),
        MoeOp::ExpertGate(1),
        MoeOp::ExpertUp(1),
        MoeOp::ExpertGate(2),
        MoeOp::ExpertUp(2),
        MoeOp::ExpertGate(3),
        MoeOp::ExpertUp(3),
        MoeOp::ExpertGate(4),
        MoeOp::ExpertUp(4),
        MoeOp::ExpertGate(5),
        MoeOp::ExpertUp(5),
        MoeOp::ExpertGate(6),
        MoeOp::ExpertUp(6),
        MoeOp::ExpertGate(7),
        MoeOp::ExpertUp(7),
        MoeOp::SharedGate,
        MoeOp::SharedUp,
        MoeOp::SwigluShared,
        MoeOp::SwigluExpert(0),
        MoeOp::ExpertDown(0),
        MoeOp::SwigluExpert(1),
        MoeOp::ExpertDown(1),
        MoeOp::SwigluExpert(2),
        MoeOp::ExpertDown(2),
        MoeOp::SwigluExpert(3),
        MoeOp::ExpertDown(3),
        MoeOp::SwigluExpert(4),
        MoeOp::ExpertDown(4),
        MoeOp::SwigluExpert(5),
        MoeOp::ExpertDown(5),
        MoeOp::SwigluExpert(6),
        MoeOp::ExpertDown(6),
        MoeOp::SwigluExpert(7),
        MoeOp::ExpertDown(7),
        MoeOp::SharedDown,
        MoeOp::Combine,
    ];
    let got = moe_sequence(moe_dims(1).top_k, WeightLayout::Tiled);
    assert_eq!(got, want);
    assert_eq!(
        got.len(),
        38,
        "1 router + 16 gate/up + 2 shared + 1 pack + 16 pack/down + 1 + 1"
    );
}

/// Every slot an op touches is live in its phase, in the real plan at one row.
#[test]
fn ops_stay_inside_their_slot_lifetimes() {
    let plan = plan(1);
    for op in moe_sequence(moe_dims(1).top_k, WeightLayout::Tiled) {
        let slots = op_slots(op);
        for slot in slots.reads.iter().chain(slots.writes.iter()) {
            let planned = plan.tensors()[plan.index(*slot).expect("a plan slot")];
            assert!(
                planned.first <= slots.phase,
                "{op:?} touches {slot:?} at phase {} before its first {}",
                slots.phase,
                planned.first
            );
            assert!(
                slots.phase <= planned.last,
                "{op:?} touches {slot:?} at phase {} after its last {}",
                slots.phase,
                planned.last
            );
        }
    }
}

/// Slots whose planned byte ranges share bytes are never live at the same launch position. The
/// span of a slot runs from its first touch to its last; two spans that intersect while the
/// byte ranges overlap is the write-then-still-needed hazard.
#[test]
fn aliased_slots_are_never_live_together() {
    let plan = plan(1);
    let seq = moe_sequence(moe_dims(1).top_k, WeightLayout::Tiled);
    let mut touched: Vec<Slot> = Vec::new();
    for op in &seq {
        let slots = op_slots(*op);
        for slot in slots.reads.iter().chain(slots.writes.iter()) {
            if !touched.contains(slot) {
                touched.push(*slot);
            }
        }
    }
    for (index, &a) in touched.iter().enumerate() {
        for &b in &touched[index + 1..] {
            let (span_a, span_b) = (span_of(&seq, a), span_of(&seq, b));
            if span_a.0 <= span_b.1 && span_b.0 <= span_a.1 {
                let (bytes_a, bytes_b) = (bytes_of(&plan, a), bytes_of(&plan, b));
                assert!(
                    bytes_a.1 <= bytes_b.0 || bytes_b.1 <= bytes_a.0,
                    "{a:?} and {b:?} share bytes and are live together at {:?}/{:?}",
                    span_a,
                    span_b
                );
            }
        }
    }
}

/// T4: the block is one row; anything else is refused before the first launch.
#[test]
fn block_refuses_rows_other_than_one() {
    for rows in [2, 0] {
        let dims = moe_dims(rows);
        let result = check_block(&dims, &weight_facts(&dims), &buffer_facts(&dims));
        assert_bad_request(result, "{rows} rows", "one row");
    }
}

/// T5: each broken fact is refused with its own text, before any launch.
#[test]
fn block_refuses_shape_mismatches_before_launch() {
    let base = moe_dims(1);
    assert!(
        check_block(&base, &weight_facts(&base), &buffer_facts(&base)).is_ok(),
        "the valid facts of the fixture pass"
    );
    let grown = MoeDims {
        top_k: base.n_expert + 1,
        ..base
    };
    let ragged = MoeDims {
        inter: base.inter - 128,
        ..base
    };
    let mut deep = weight_facts(&base);
    deep.gate.experts += 1;
    let mut short = weight_facts(&base);
    short.router.bytes -= 4;
    let mut packed = buffer_facts(&base);
    packed.expert.packs.moe.bytes -= 256;
    let mut shared = buffer_facts(&base);
    shared.route.ids.offset = shared.route.norm_row.offset;
    let cases = [
        (
            "top_k above the expert count",
            grown,
            weight_facts(&grown),
            buffer_facts(&grown),
            "top_k",
        ),
        (
            "inter not a multiple of the pack width",
            ragged,
            weight_facts(&ragged),
            buffer_facts(&ragged),
            "multiple",
        ),
        (
            "gate stack depth above the experts",
            base,
            deep,
            buffer_facts(&base),
            "depth",
        ),
        (
            "router weight too short",
            base,
            short,
            buffer_facts(&base),
            "router weight",
        ),
        (
            "xf_moe range of the wrong size",
            base,
            weight_facts(&base),
            packed,
            "xf_moe",
        ),
        (
            "router ranges overlapping",
            base,
            weight_facts(&base),
            shared,
            "overlap",
        ),
    ];
    for (case, dims, weights, bufs, needle) in cases {
        assert_bad_request(check_block(&dims, &weights, &bufs), case, needle);
    }
}

/// The NATIVE sequence as a literal: the router first and the combine last (their places and
/// bytes are the tiered hook's), the per-slot triples collapsed to one merged launch each.
#[test]
fn the_native_moe_sequence_is_the_ten_merged_ops() {
    let want: Vec<MoeOp> = vec![
        MoeOp::Router,
        MoeOp::ExpertGateQ,
        MoeOp::ExpertUpQ,
        MoeOp::SharedGateQ,
        MoeOp::SharedUpQ,
        MoeOp::SwigluSharedQ,
        MoeOp::SwigluExperts,
        MoeOp::ExpertDownQ,
        MoeOp::SharedDownQ,
        MoeOp::Combine,
    ];
    assert_eq!(moe_sequence(8, WeightLayout::Native), want);
    // Any top_k gives the same ten ops: the merged launches take the slots together.
    assert_eq!(moe_sequence(4, WeightLayout::Native), want);
}

/// With `INFERENCE_CUDA_MOE_GLU_FUSE=1` the gate+up pair collapses to the fused GLU launch and
/// the expert pack becomes the quantize-only form, so the Native block is nine ops. The flag is
/// process-global, so the parent drives one child with the env set and pins the printed order
/// (the GDN `FUSED_AB_ARM` pattern, CPU-only).
#[test]
fn the_fused_glu_native_moe_sequence_is_the_nine_ops() {
    const NAME: &str = "the_fused_glu_native_moe_sequence_is_the_nine_ops";
    if std::env::var("GLU_SEQ_ARM").is_ok() {
        for op in moe_sequence(8, WeightLayout::Native) {
            println!("GLU_SEQ_OP {op:?}");
        }
        return;
    }
    let exe = std::env::current_exe().expect("the test binary path");
    // module_path!() carries the crate name, which libtest's --exact does not match; strip it
    // (the GDN pattern).
    let path = std::module_path!();
    let stripped = path.split_once("::").map_or(path, |(_, rest)| rest);
    let output = std::process::Command::new(exe)
        .args(["--exact", &format!("{stripped}::{NAME}"), "--nocapture"])
        .env("GLU_SEQ_ARM", "on")
        .env("INFERENCE_CUDA_MOE_GLU_FUSE", "1")
        .output()
        .expect("spawn the child arm");
    assert!(
        output.status.success(),
        "{NAME}: the child exited with {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let got: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("GLU_SEQ_OP "))
        .collect();
    let want = [
        "Router",
        "ExpertGluQ",
        "SharedGateQ",
        "SharedUpQ",
        "SwigluSharedQ",
        "SwigluExpertsGlu",
        "ExpertDownQ",
        "SharedDownQ",
        "Combine",
    ];
    assert_eq!(got, want, "{NAME}: the fused sequence order");
    assert_eq!(got.len(), 9, "{NAME}: nine ops, not ten");
}

/// The card sequence keeps the Native order and inserts the two ring markers where the
/// token plan of `multi.rs` places them: the picks broadcast right after the router on
/// card 0 (and at the same place as a receive on every other card), the partial sum
/// directly before the combine. Both cards run the SAME marker sequence, and the local
/// ops between the markers are identical.
#[test]
fn the_card_moe_sequence_aligns_its_markers_on_every_card() {
    let card0 = moe_sequence_card(true);
    let card1 = moe_sequence_card(false);
    let markers = |ops: &[CardMoeOp]| -> Vec<CardMoeOp> {
        ops.iter()
            .copied()
            .filter(|op| !matches!(op, CardMoeOp::Local(_)))
            .collect()
    };
    assert_eq!(
        markers(&card0),
        vec![CardMoeOp::RouteBroadcast, CardMoeOp::MoeSum],
        "card 0 marker order"
    );
    assert_eq!(markers(&card1), markers(&card0), "both cards align");
    // Card 0 runs the router, then the broadcast; the other card receives instead.
    assert_eq!(
        (&card0[0], &card0[1]),
        (&CardMoeOp::Local(MoeOp::Router), &CardMoeOp::RouteBroadcast)
    );
    assert_eq!(card1[0], CardMoeOp::RouteBroadcast);
    // The sum sits directly before the combine on both cards.
    for card in [&card0, &card1] {
        let at = card.len() - 2;
        assert_eq!(
            (card[at], card[at + 1]),
            (CardMoeOp::MoeSum, CardMoeOp::Local(MoeOp::Combine))
        );
    }
    // The local ops are the single-device list minus the router, in order, on every card.
    let locals = |ops: &[CardMoeOp]| -> Vec<MoeOp> {
        ops.iter()
            .filter_map(|op| match op {
                CardMoeOp::Local(op) => Some(*op),
                _ => None,
            })
            .collect()
    };
    let single = moe_sequence(8, WeightLayout::Native);
    assert_eq!(locals(&card0), single, "card 0 keeps the whole list");
    assert_eq!(
        locals(&card1),
        single[1..],
        "the other card skips the router"
    );
}

/// On the native list: every slot an op touches is live in its phase in the NATIVE plan, and
/// two slots that share bytes are never live at the same launch position (the aliasing rule,
/// spelled over the merged ops).
#[test]
fn native_moe_ops_stay_inside_their_slot_lifetimes() {
    let plan = plan_of(WeightLayout::Native);
    let seq = moe_sequence(8, WeightLayout::Native);
    for op in &seq {
        let slots = op_slots(*op);
        for slot in slots.reads.iter().chain(slots.writes.iter()) {
            let planned = plan.tensors()[plan.index(*slot).expect("a plan slot")];
            assert!(
                planned.first <= slots.phase && slots.phase <= planned.last,
                "{op:?} touches {slot:?} at phase {} outside {}..={}",
                slots.phase,
                planned.first,
                planned.last
            );
        }
    }
    let mut touched: Vec<Slot> = Vec::new();
    for op in &seq {
        let slots = op_slots(*op);
        for slot in slots.reads.iter().chain(slots.writes.iter()) {
            if !touched.contains(slot) {
                touched.push(*slot);
            }
        }
    }
    for (index, &a) in touched.iter().enumerate() {
        for &b in &touched[index + 1..] {
            let (span_a, span_b) = (span_of(&seq, a), span_of(&seq, b));
            if span_a.0 <= span_b.1 && span_b.0 <= span_a.1 {
                let (bytes_a, bytes_b) = (bytes_of(&plan, a), bytes_of(&plan, b));
                assert!(
                    bytes_a.1 <= bytes_b.0 || bytes_b.1 <= bytes_a.0,
                    "{a:?} and {b:?} share bytes and are live together at {span_a:?}/{span_b:?}"
                );
            }
        }
    }
}

/// The q8_1 pack sizes of the native block: a Q8 buffer_facts set passes check_block, a wrong
/// moe pack size is refused, and the Tiled fact set (fp16 sizes) is refused against the Q8 kind.
#[test]
fn the_native_block_needs_the_q8_1_pack_sizes() {
    let dims = moe_dims(1);
    let mut facts = buffer_facts(&dims);
    facts.expert.packs = PackFacts {
        kind: PackKind::Q8,
        moe: facts.expert.packs.moe,
        down: facts.expert.packs.down,
        shared: facts.expert.packs.shared,
    };
    assert_bad_request(
        check_block(&dims, &weight_facts(&dims), &facts),
        "fp16 sizes under the q8 kind",
        "xf_moe",
    );
    let q8 = |k: usize, rows: usize| {
        crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes(k, rows).expect("size")
    };
    facts.expert.packs.moe.bytes = q8(dims.hidden, 1);
    facts.expert.packs.down.bytes = q8(dims.inter, dims.top_k);
    facts.expert.packs.shared.bytes = q8(dims.shared_inter, 1);
    assert!(
        check_block(&dims, &weight_facts(&dims), &facts).is_ok(),
        "the q8_1 pack sizes pass the pure checks"
    );
    facts.expert.packs.down.bytes -= 36;
    assert_bad_request(
        check_block(&dims, &weight_facts(&dims), &facts),
        "one block short",
        "xf_down",
    );
}

#[test]
fn card_moe_dims_slice_the_intermediate_widths_and_keep_the_rest_full() {
    let d = step_dims();
    let full = MoeDims::new(&d, 1);
    let table = ShardTable::new(&d.shape, 2, 32).expect("table");
    let mut inter = 0;
    let mut shared = 0;
    for card in 0..2 {
        let dims = MoeDims::for_card(&d, &table, card, 1).expect("card dims");
        inter += dims.inter;
        shared += dims.shared_inter;
        assert_eq!(dims.hidden, full.hidden, "the residual row is replicated");
        assert_eq!(dims.n_expert, full.n_expert, "every expert on every card");
        assert_eq!(dims.top_k, full.top_k, "routing is replicated");
    }
    assert_eq!(inter, full.inter, "expert inter slices tile the width");
    assert_eq!(
        shared, full.shared_inter,
        "shared inter slices tile the width"
    );
    // A one-card table is the degenerate case: the single-device dims.
    let one = ShardTable::new(&d.shape, 1, 32).expect("table");
    assert_eq!(
        MoeDims::for_card(&d, &one, 0, 1).expect("card dims"),
        full,
        "a one-card table must give the single-device dims"
    );
}
