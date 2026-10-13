//! CPU-only unit tests of the pure MoE planning (decision 0047 item 9, seam U13): the documented
//! op order, the slot lifetimes of the real RD-9a plan, aliasing over the launch positions, and
//! the pure refusals of `check_block`. No device, no kernel.

use super::super::super::host::HostShape;
use super::super::dims::{KvMode, StepDims, StepOptions};
use super::super::step_plan::{KernelScratch, Slot, StepPlan};
use super::{
    BufferFacts, ExpertFacts, MoeDims, MoeOp, OpSlots, PackFacts, PackKind, ProjFacts, RangeFacts,
    RouteFacts, StackFacts, WeightFacts, check_block, moe_sequence, op_slots,
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

/// The MoE block dims of the fixture at `rows` rows.
fn moe_dims(rows: usize) -> MoeDims {
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
    )
    .expect("scratch");
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
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
    let scratch = KernelScratch::new(&dims, 1, router_logits_bytes(router).expect("router bytes"))
        .expect("scratch");
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
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
fn weight_facts(dims: &MoeDims) -> WeightFacts {
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
fn buffer_facts(dims: &MoeDims) -> BufferFacts {
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

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &str) {
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

/// T2 (H1): every slot an op touches is live in its phase, in the real plan at one row.
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

/// T3 (H2, H3): slots whose planned byte ranges share bytes are never live at the same launch
/// position. The span of a slot runs from its first touch to its last; two spans that intersect
/// while the byte ranges overlap is the write-then-still-needed hazard.
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
/// bytes are the OFF-T hook's), the per-slot triples collapsed to one merged launch each.
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

/// T2 of the packet, on the native list: every slot an op touches is live in its phase in the
/// NATIVE plan, and two slots that share bytes are never live at the same launch position
/// (T3's rule, spelled over the merged ops).
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
    // The Q8 kind over the Tiled fp16 sizes is refused before any launch.
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
