//! CPU tests of the NATIVE step plan: the q8_1 slot sizes, the pinned arena figures, the lifetimes
//! of the swapped packs, the aliasing freedom of the plan and the pure refusals of the one-row
//! rule. CPU only, no device call. The TILED plan stays pinned by `tests.rs`'s own figures; the
//! Int8G32 rows slots are pinned here beside them.
use super::super::host::HostShape;
use super::{
    dims::{KvMode, StepDims, StepOptions},
    step_plan::{KernelScratch, Phase, Slot, StepPlan},
};
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::plan_arena;
use crate::kernels::cuda::resident::attn_decode::AttnSplit;
use crate::kernels::cuda::resident::attn_decode_q8::attn_q8_part_bytes;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

fn dims() -> StepDims {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(
        &HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape"),
        &inventory,
    )
    .expect("dims")
}

fn options(layout: WeightLayout) -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    }
}

#[test]
fn host_read_volume_pins_the_current_two_slot_reload_policy() {
    use super::paging_plan::PagingPlan;

    for (capacity, cells, read_8k, read_32k) in [
        (33280, 768, 100_270_080, 367_656_960),
        (131072, 2048, 111_411_200, 378_798_080),
    ] {
        let p = PagingPlan::new(&dims(), capacity).expect("production paging geometry");
        assert_eq!((p.full_layers, p.kv_heads, p.head_dim), (10, 2, 256));
        assert_eq!(p.group_cells, cells);
        assert_eq!(
            p.group.total,
            2 * p.kv_heads * cells * (p.head_dim + p.head_dim / 32 * 2)
        );
        assert_eq!(p.slot_bytes, p.group.total + 4);
        for (visible, expected) in [(8192_usize, read_8k), (32768, read_32k)] {
            // Append loads its current page before attention reloads every visible page.
            let loads = p.full_layers * (visible.div_ceil(cells) + 1);
            assert_eq!(loads * p.group.total, expected);
            assert_eq!(loads * p.slot_bytes, expected + loads * 4);
        }
    }
}

#[test]
fn device_first_budget_keeps_only_whole_page_groups_on_the_device() {
    use super::paging_plan::PagingPlan;

    let p = PagingPlan::new(&dims(), 33280).expect("production paging geometry");
    let group_bytes = p.group.total * p.full_layers;
    for (budget, groups) in [
        (0, 0),
        (group_bytes - 1, 0),
        (group_bytes, 1),
        (3 * group_bytes - 1, 2),
        (p.backing_bytes, p.groups),
        (usize::MAX, p.groups),
    ] {
        let residency = p
            .device_residency(budget)
            .expect("bounded device residency");
        assert_eq!(residency.device_groups, groups);
        assert_eq!(residency.device_rows, groups * p.group_cells);
        assert_eq!(residency.device_bytes, groups * group_bytes);
        assert!(residency.device_bytes <= budget);
        assert_eq!(
            residency.host_payload_bytes,
            p.backing_bytes - residency.device_bytes
        );
    }
}

#[test]
fn cached_state_keeps_logical_geometry_and_charges_only_physical_storage() {
    use super::{
        paging_plan::PagingPlan, paging_store::device_first::DeviceCachePlan,
        state::PagedStatePlan, state_layout::StateLayout,
    };
    let d = dims();
    let opts = StepOptions {
        context_tokens: 1280,
        kv_mode: KvMode::Int8G32,
        ..options(WeightLayout::Native)
    };
    let logical = StateLayout::new(&d, &opts).unwrap();
    let paging = PagingPlan::new(&d, opts.context_tokens).unwrap();
    let group = paging.group.total * paging.full_layers;
    let cache = DeviceCachePlan::new(&d, opts.context_tokens, 2 * group, 32 << 20).unwrap();
    let planned = PagedStatePlan::new_cached(&d, &opts, cache).unwrap();
    assert_eq!(planned.capacity(), opts.context_tokens);
    assert_eq!(planned.cache_rows(), Some(2 * paging.group_cells));
    assert_eq!(
        planned.device_bytes().unwrap(),
        logical.delta.total + logical.words.total + paging.device_bytes + 2 * group
    );
    let different = DeviceCachePlan::new(&d, 640, 2 * group, 32 << 20).unwrap();
    assert!(PagedStatePlan::new_cached(&d, &opts, different).is_err());
}

fn plan(layout: WeightLayout) -> StepPlan {
    let d = dims();
    StepPlan::new(
        &d,
        1,
        &options(layout),
        &KernelScratch::for_dims(&d, 1, KvMode::Fp16).expect("scratch"),
    )
    .expect("plan")
}

#[test]
fn paged_native_plan_omits_the_resident_attention_partial() {
    let d = dims();
    let o = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 33280,
        ..options(WeightLayout::Native)
    };
    let scratch = KernelScratch::for_dims(&d, 1, o.kv_mode).expect("scratch");
    let resident = StepPlan::new(&d, 1, &o, &scratch).expect("resident plan");
    let paged = StepPlan::new_paged(&d, &o, &scratch).expect("paged plan");
    assert!(resident.index(Slot::AttnPart).is_some());
    assert!(paged.index(Slot::AttnPart).is_none());
    let other_slots: Vec<_> = resident
        .tensors()
        .iter()
        .filter(|slot| slot.slot != Slot::AttnPart)
        .copied()
        .collect();
    assert_eq!(paged.tensors(), other_slots.as_slice());
    assert_eq!(paged.plan(), &plan_arena(&paged.arena_tensors()).unwrap());
    let removed = resident.tensors()[resident.index(Slot::AttnPart).unwrap()]
        .bytes
        .next_multiple_of(256);
    assert_eq!(
        paged.unaliased_bytes(),
        resident.unaliased_bytes() - removed
    );
}

#[test]
fn paged_native_plan_refuses_unsupported_kv_or_weight_layout() {
    let d = dims();
    for (kv_mode, layout) in [
        (KvMode::Fp16, WeightLayout::Native),
        (KvMode::Int8G32, WeightLayout::Tiled),
    ] {
        let o = StepOptions {
            kv_mode,
            ..options(layout)
        };
        let scratch = KernelScratch::for_dims(&d, 1, o.kv_mode).expect("scratch");
        assert!(StepPlan::new_paged(&d, &o, &scratch).is_err());
    }
}

/// Every Native pack slot holds exactly the padded `block_q8_1` bytes of its rows, and the eight
/// rows of `Q8Down` sit at a stride of whole 36-byte blocks.
#[test]
fn native_slot_sizes_are_the_q8_1_sizes() {
    let (d, p) = (dims(), plan(WeightLayout::Native));
    let size = |slot| p.tensors()[p.index(slot).expect("slot")].bytes;
    let (h, m) = (d.shape.hidden, &d.shape.moe);
    let q_width = d.shape.attn.heads_q * d.shape.attn.head_dim;
    let value_width = d.shape.delta.value_heads * d.shape.delta.value_dim;
    // The expected literals: 2,304 B for k 2048, 4,608 B for k 4096, 576 B for k 512.
    let want = [
        (Slot::Q8Mix, 2304, h, 1),
        (Slot::Q8DeltaOut, 4608, value_width, 1),
        (Slot::Q8AttnOut, 4608, q_width, 1),
        (Slot::Q8Moe, 2304, h, 1),
        (Slot::Q8Down, 4608, m.inter, m.top_k),
        (Slot::Q8Shared, 576, m.shared_inter, 1),
        (Slot::Q8Final, 2304, h, 1),
    ];
    for (slot, pinned, k, rows) in want {
        assert_eq!(size(slot), pinned, "{slot:?}");
        assert_eq!(
            size(slot),
            q8_1_bytes(k, rows).expect("q8 size"),
            "{slot:?}"
        );
    }
    let stride = size(Slot::Q8Down) / m.top_k;
    assert_eq!(
        (stride, stride % 36),
        (576, 0),
        "a whole block per slot row"
    );
}

/// The Native plan's figures, pinned like the Tiled ones of `tests.rs` (`arena_check.py`
/// recomputes them independently at qualification), plus the planner's invariants: 256-byte
/// offsets, no two live-together slots share bytes, `plan_arena` reproduces the plan, and
/// `ArgmaxPart` is live in ARGMAX alone (it may reuse a dead q8 slot's bytes, never a live one's).
#[test]
fn the_native_plan_figures_are_pinned_and_do_not_overlap() {
    let p = plan(WeightLayout::Native);
    assert_eq!(
        (p.arena_bytes(), p.unaliased_bytes()),
        (1_159_168, 2_396_416),
        "the swapped q8_1 sizes must move these figures or nothing"
    );
    assert_eq!(p.tensors().len(), 29);
    let t = p.tensors();
    for (i, a) in t.iter().enumerate() {
        assert_eq!(p.plan().offsets[i] % 256, 0, "{:?}", a.slot);
        for (j, b) in t.iter().enumerate().skip(i + 1) {
            let (oa, ob) = (p.plan().offsets[i], p.plan().offsets[j]);
            let together = a.first <= b.last && b.first <= a.last;
            let apart = oa + a.bytes <= ob || ob + b.bytes <= oa;
            assert!(!together || apart, "{:?} and {:?} overlap", a.slot, b.slot);
        }
    }
    assert_eq!(&plan_arena(&p.arena_tensors()).expect("arena"), p.plan());
    let part = p.tensors()[p.index(Slot::ArgmaxPart).expect("argmax")];
    assert_eq!(
        (part.first, part.last),
        (Phase::Argmax as usize, Phase::Argmax as usize)
    );
}

/// Every Native slot takes the lifetime and size of its Tiled twin, with TWO spelled-out
/// exceptions: the packs swap names, and `NormRow` — dead scratch that the fused `add_rmsnorm_q8`
/// writes at the NATIVE mixer prologue beside its pack (the producer always writes the f32 row) —
/// spans `MIX_PACK..ARGMAX` instead of `ADD_NORM..ROUTE`; the final norm writes its OWN
/// FinalNormRow (`tests.rs` pins the Tiled table literally; the mapping here is the only new fact).
#[test]
fn native_pack_slots_take_the_tiled_lifetimes_and_sizes() {
    let swap = |slot: Slot| match slot {
        Slot::XfMix => Slot::Q8Mix,
        Slot::XfDeltaOut => Slot::Q8DeltaOut,
        Slot::XfAttnOut => Slot::Q8AttnOut,
        Slot::XfMoe => Slot::Q8Moe,
        Slot::XfDown(_) => Slot::Q8Down,
        Slot::XfShared => Slot::Q8Shared,
        Slot::XfFinal => Slot::Q8Final,
        other => other,
    };
    let (p, tiled) = (plan(WeightLayout::Native), plan(WeightLayout::Tiled));
    let want: Vec<(Slot, usize, usize)> = tiled
        .tensors()
        .iter()
        .map(|t| {
            let (first, last) = if t.slot == Slot::NormRow {
                (Phase::MixPack as usize, Phase::Argmax as usize)
            } else {
                (t.first, t.last)
            };
            (swap(t.slot), first, last)
        })
        .collect();
    let got: Vec<(Slot, usize, usize)> = p
        .tensors()
        .iter()
        .map(|t| (t.slot, t.first, t.last))
        .collect();
    // Every swapped Tiled slot appears with its lifetime, beside the plan's own FinalNormRow
    // (the final norm's dead-scratch row, FINAL_NORM only - NormRow's bytes may be live here).
    for want_slot in &want {
        assert!(
            got.contains(want_slot),
            "{want_slot:?} missing from the native plan"
        );
    }
    assert!(got.contains(&(
        Slot::FinalNormRow,
        Phase::FinalNorm as usize,
        Phase::FinalNorm as usize
    )));
    assert_eq!(got.len(), want.len() + 1);
    // Non-pack slots keep their Tiled byte counts (a pack slot's q8 bytes are test 1's literals).
    for planned in tiled.tensors() {
        if swap(planned.slot) == planned.slot {
            let at = p.index(planned.slot).expect("the shared slot");
            assert_eq!(p.tensors()[at].bytes, planned.bytes, "{:?}", planned.slot);
        }
    }
    // And the swap is exactly seven packs: no Tiled pack survives in the Native plan.
    assert!(p.index(Slot::XfMix).is_none() && p.index(Slot::XfDown(0)).is_none());
    assert!(tiled.index(Slot::Q8Mix).is_none() && tiled.index(Slot::Q8Down).is_none());
}

/// The pure refusals of the Native plan: more than one row (the q8_1 producers are single-row),
/// and the forked per-expert packs (the Native MoE runs one merged down launch). Each case has
/// every earlier check satisfied (row count in range, matching scratch), so only the layout rule
/// can refuse.
#[test]
fn the_native_plan_refuses_extra_rows_and_forked_packs() {
    let d = dims();
    let scratch2 = KernelScratch::for_dims(&d, 2, KvMode::Fp16).expect("scratch");
    match StepPlan::new(&d, 2, &options(WeightLayout::Native), &scratch2) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(message.contains("one row"), "{message}");
        }
        other => panic!("two Native rows must be refused: {other:?}"),
    }
    let forked = StepOptions {
        forked_experts: true,
        tiered: None,
        ..options(WeightLayout::Native)
    };
    match StepPlan::new(
        &d,
        1,
        &forked,
        &KernelScratch::for_dims(&d, 1, KvMode::Fp16).expect("scratch"),
    ) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(message.contains("merged expert down pack"), "{message}");
        }
        other => panic!("forked Native packs must be refused: {other:?}"),
    }
}

// ---- the Int8G32 rows slots of the NATIVE plan -----------------------------------------------

fn int8_options(layout: WeightLayout) -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    }
}

/// The one-row NATIVE Int8G32 plan: the native plan plus the two rows slots of the KV handoff.
fn int8_plan() -> StepPlan {
    let d = dims();
    StepPlan::new(
        &d,
        1,
        &int8_options(WeightLayout::Native),
        &KernelScratch::for_dims(&d, 1, KvMode::Int8G32).expect("scratch"),
    )
    .expect("plan")
}

/// The Int8G32 plan's figures, pinned like the Fp16 ones (`arena_check.py` recomputes them at
/// qualification), plus the two rows slots: their size is one f32 row per kv head, their
/// lifetime is `ATTN_PREP..KV8_APPEND` (written by the rows entry of `attn_prep`, consumed by
/// `kv8_append`), and no other slot or byte of the native plan changes.
#[test]
fn the_int8_native_plan_figures_are_pinned_and_do_not_overlap() {
    let (d, p) = (dims(), int8_plan());
    assert_eq!(
        (p.arena_bytes(), p.unaliased_bytes()),
        (1_159_168, 2_400_512),
        "the two rows slots must move these figures or nothing"
    );
    assert_eq!(p.tensors().len(), 31);
    let kv_width = d.shape.attn.heads_kv * d.shape.attn.head_dim;
    for slot in [Slot::KRows, Slot::VRows] {
        let planned = p.tensors()[p.index(slot).expect("rows slot")];
        assert_eq!(planned.bytes, kv_width * 4, "{slot:?}");
        assert_eq!(
            (planned.first, planned.last),
            (Phase::AttnPrep as usize, Phase::Kv8Append as usize),
            "{slot:?}"
        );
    }
    // The rows slots are the ONLY difference from the Fp16 native plan.
    let fp16 = plan(WeightLayout::Native);
    let int8: Vec<(Slot, usize, usize, usize)> = p
        .tensors()
        .iter()
        .map(|t| (t.slot, t.bytes, t.first, t.last))
        .collect();
    let mut want: Vec<(Slot, usize, usize, usize)> = fp16
        .tensors()
        .iter()
        .map(|t| (t.slot, t.bytes, t.first, t.last))
        .collect();
    want.push((
        Slot::KRows,
        kv_width * 4,
        Phase::AttnPrep as usize,
        Phase::Kv8Append as usize,
    ));
    want.push((
        Slot::VRows,
        kv_width * 4,
        Phase::AttnPrep as usize,
        Phase::Kv8Append as usize,
    ));
    want.sort_by_key(|(slot, _, _, _)| format!("{slot:?}"));
    let mut got = int8.clone();
    got.sort_by_key(|(slot, _, _, _)| format!("{slot:?}"));
    assert_eq!(got, want);
    // The planner invariants of the Fp16 pins hold here too.
    for (i, a) in p.tensors().iter().enumerate() {
        assert_eq!(p.plan().offsets[i] % 256, 0, "{:?}", a.slot);
        for (j, b) in p.tensors().iter().enumerate().skip(i + 1) {
            let (oa, ob) = (p.plan().offsets[i], p.plan().offsets[j]);
            let together = a.first <= b.last && b.first <= a.last;
            let apart = oa + a.bytes <= ob || ob + b.bytes <= oa;
            assert!(!together || apart, "{:?} and {:?} overlap", a.slot, b.slot);
        }
    }
    assert_eq!(&plan_arena(&p.arena_tensors()).expect("arena"), p.plan());
}

/// The KV8 sizing rule: under Int8G32 the scratch sizes `AttnPart` with the KV8 kernel's own
/// `attn_q8_part_bytes` at `AttnSplit::Q8`, and the plan refuses a scratch of the other mode.
/// STAGE1 and Q8 share `max_splits` 64, so the byte counts COINCIDE: the split is the observable,
/// and a scratch sized with STAGE1 under Int8G32 is refused by the plan, not by a size.
#[test]
fn int8_attn_part_is_sized_with_the_q8_split() {
    let d = dims();
    let a = &d.shape.attn;
    let scratch = KernelScratch::for_dims(&d, 1, KvMode::Int8G32).expect("scratch");
    assert_eq!(scratch.attn_split(), AttnSplit::Q8);
    let p = int8_plan();
    assert_eq!(
        p.tensors()[p.index(Slot::AttnPart).expect("attn part")].bytes,
        attn_q8_part_bytes(1, a.heads_kv, a.head_dim, AttnSplit::Q8.max_splits).unwrap()
    );
    // The Fp16 scratch stays on STAGE1 and its plan refuses the Q8 scratch, and the Int8G32 plan
    // refuses the STAGE1 scratch: each mode takes its own split, and the refusal names the split it
    // found and the one it needs.
    let fp16 = KernelScratch::for_dims(&d, 1, KvMode::Fp16).expect("scratch");
    assert_eq!(fp16.attn_split(), AttnSplit::STAGE1);
    match StepPlan::new(&d, 1, &int8_options(WeightLayout::Native), &fp16) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(
                message.contains("split") && message.contains("min_keys: 32"),
                "{message}"
            );
        }
        other => panic!("an Int8G32 plan must refuse the STAGE1 scratch: {other:?}"),
    }
    match StepPlan::new(
        &d,
        1,
        &options(WeightLayout::Native),
        &KernelScratch::for_dims(&d, 1, KvMode::Int8G32).expect("scratch"),
    ) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(
                message.contains("split") && message.contains("min_keys: 64"),
                "{message}"
            );
        }
        other => panic!("an Fp16 plan must refuse the Q8 scratch: {other:?}"),
    }
}

/// The typed refusal of the control path: Int8G32 has Native op lists only, so a Tiled Int8G32 plan
/// is refused before any slot is sized.
#[test]
fn the_int8_plan_refuses_the_tiled_layout() {
    let d = dims();
    match StepPlan::new(
        &d,
        1,
        &int8_options(WeightLayout::Tiled),
        &KernelScratch::for_dims(&d, 1, KvMode::Int8G32).expect("scratch"),
    ) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(message.contains("Int8G32"), "{message}");
            assert!(message.contains("Tiled"), "{message}");
        }
        other => panic!("a Tiled Int8G32 plan must be refused: {other:?}"),
    }
}
