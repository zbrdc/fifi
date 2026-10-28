//! Unit tests of the pure host planning (decision 0047, packet RD-9a; test seam approved by the
//! user as U13, 2026-10-10): dims, state layout and step plan. CPU only, no device call.
use super::super::host::HostShape;
use super::{
    dims::{KvMode, KvRequest, StepDims, StepOptions},
    state_layout::{KvPool, Plane, StateLayout, Word},
    step_plan::{KernelScratch, MAX_STEP_ROWS, PHASES, Phase, Slot, StepPlan, writes_status},
};
use crate::family::project::cuda::WeightLayout;
use crate::family::{AttentionKind, QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::{ArenaTensor, plan_arena};

/// Router scratch the way RD-5 states it: `4 * rows * (experts + 1)` (1,028 bytes at one row).
fn router_bytes(d: &StepDims, rows: usize) -> usize {
    4 * rows * (d.shape.moe.experts + 1)
}

fn dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(&shape, &inventory).expect("dims")
}

fn scratch(d: &StepDims, rows: usize) -> KernelScratch {
    KernelScratch::new(d, rows, router_bytes(d, rows), KvMode::Fp16).expect("scratch")
}

fn options(mode: KvMode, ctx: usize) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens: ctx,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Tiled,
    }
}

fn plan(d: &StepDims, rows: usize, opts: &StepOptions) -> StepPlan {
    StepPlan::new(d, rows, opts, &scratch(d, rows)).expect("plan")
}

// Pinned figures of the Qwen3.6-35B-A3B plan at FP16 KV (rows, arena_bytes, unaliased_bytes).
// `arena_check.py` recomputes them independently.
const PINNED: [(usize, usize, usize); 4] = [
    (1, 2_225_156, 9_708_544),
    (2, 3_068_160, 12_071_168),
    (4, 5_087_488, 16_796_416),
    (8, 9_207_808, 26_246_912),
];

#[test]
fn plan_figures_are_pinned_and_do_not_overlap() {
    let d = dims();
    for (rows, arena, unaliased) in PINNED {
        let p = plan(&d, rows, &options(KvMode::Fp16, 1024));
        assert_eq!(
            (p.arena_bytes(), p.unaliased_bytes()),
            (arena, unaliased),
            "{rows} rows"
        );
        assert_eq!(p.tensors().len(), 28);
        let t = p.tensors();
        for (i, a) in t.iter().enumerate() {
            let oa = p.plan().offsets[i];
            assert_eq!(oa % 256, 0);
            for (j, b) in t.iter().enumerate().skip(i + 1) {
                let ob = p.plan().offsets[j];
                let together = a.first <= b.last && b.first <= a.last;
                let apart = oa + a.bytes <= ob || ob + b.bytes <= oa;
                assert!(!together || apart, "{:?} and {:?} overlap", a.slot, b.slot);
            }
        }
        // Determinism and list order: `plan_arena` over `arena_tensors()` gives the plan's own
        // offsets. This compares `plan_arena` with `plan()`; it does NOT run `CudaContext::arena`
        // (that it calls the same planner on the same list is read from `device_arena.rs:80`).
        assert_eq!(&plan_arena(&p.arena_tensors()).expect("arena"), p.plan());
        assert_eq!(p, plan(&d, rows, &options(KvMode::Fp16, 1024)));
    }
}

#[test]
fn forked_experts_add_one_pack_per_routed_expert() {
    let d = dims();
    let fork_opts = StepOptions {
        forked_experts: true,
        tiered: None,
        ..options(KvMode::Fp16, 1024)
    };
    let fork = plan(&d, 1, &fork_opts);
    let seq = plan(&d, 1, &options(KvMode::Fp16, 1024));
    assert_eq!(
        (fork.arena_bytes(), fork.unaliased_bytes()),
        (9_566_980, 17_050_368)
    );
    assert_eq!(fork.tensors().len(), seq.tensors().len() + 7);
    assert!(fork.offset(Slot::XfDown(7)).is_some() && seq.offset(Slot::XfDown(1)).is_none());
    // Every forked branch pack covers the whole Down phase, like the single pack of the plain plan.
    for branch in 0..8 {
        let at = fork.index(Slot::XfDown(branch)).expect("branch pack");
        let t = fork.tensors()[at];
        assert_eq!(
            (t.first, t.last),
            (Phase::Swiglu as usize, Phase::Down as usize),
            "{branch}"
        );
    }
}

/// The lifetime of every tensor, producer to last consumer, written out as a specification.
/// An under-extended lifetime lets the planner alias a live buffer, and no size figure notices.
#[test]
fn plan_lifetimes_are_pinned() {
    use Phase as P;
    use Slot as S;
    let table: [(Slot, Phase, Phase); 28] = [
        (S::ResidA, P::Embed, P::Argmax),
        (S::ResidB, P::Embed, P::Argmax),
        (S::XfMix, P::MixPack, P::AttnIn), // read by the in-projections of either mixer
        (S::DeltaProj, P::DeltaIn, P::DeltaRule), // z, beta, alpha are read by the delta rule
        (S::DeltaConv, P::DeltaConv, P::DeltaRule),
        (S::DeltaCore, P::DeltaRule, P::DeltaPack),
        (S::XfDeltaOut, P::DeltaPack, P::DeltaOut),
        (S::AttnProj, P::AttnIn, P::AttnCore), // the output gate is read by attention
        (S::AttnQ, P::AttnPrep, P::AttnCore),
        (S::AttnPart, P::AttnCore, P::AttnCore),
        (S::AttnCore, P::AttnCore, P::AttnPack),
        (S::XfAttnOut, P::AttnPack, P::AttnOut),
        (S::MixOut, P::DeltaOut, P::AddNorm),
        (S::XfMoe, P::AddNorm, P::GateUp), // read by the gate/up matmuls only
        (S::NormRow, P::AddNorm, P::Route),
        (S::RouterLogits, P::Route, P::Route),
        (S::TopkId, P::Route, P::Combine),
        (S::TopkWeight, P::Route, P::Combine),
        (S::SharedGate, P::Route, P::Combine),
        (S::ExpertGu, P::GateUp, P::Down), // every slot's swiglu reads it, inside the Down phase
        (S::XfDown(0), P::Swiglu, P::Down),
        (S::ExpertDown, P::Down, P::Combine),
        (S::SharedGu, P::GateUp, P::Down),
        (S::XfShared, P::Swiglu, P::Down),
        (S::SharedOut, P::Down, P::Combine),
        (S::XfFinal, P::FinalNorm, P::LmHead),
        (S::Logits, P::LmHead, P::Argmax),
        (S::ArgmaxPart, P::Argmax, P::Argmax),
    ];
    let p = plan(&dims(), 1, &options(KvMode::Fp16, 8));
    let got: Vec<(Slot, usize, usize)> = p
        .tensors()
        .iter()
        .map(|t| (t.slot, t.first, t.last))
        .collect();
    let want: Vec<(Slot, usize, usize)> = table
        .iter()
        .map(|&(slot, first, last)| (slot, first as usize, last as usize))
        .collect();
    assert_eq!(got, want);
}

/// RD-9c's contract, phase ordered on one stream: (1) router; (2) all expert gate/up matmuls and
/// the shared gate/up; (3) shared swiglu; (4) per slot s: swiglu_pack(s) into the one `XfDown`, then
/// down(s); (5) shared down; (6) combine. Every buffer's real first and last use is derived from
/// this list; the plan must cover it, and buffers whose uses overlap in time must not share bytes.
#[test]
fn phase_ordered_expert_buffers_never_alias() {
    use Phase as P;
    use Slot as S;
    let mut ops: Vec<(Phase, Vec<Slot>)> = vec![
        (P::AddNorm, vec![S::XfMoe, S::NormRow]),
        (
            P::Route,
            vec![
                S::NormRow,
                S::RouterLogits,
                S::TopkId,
                S::TopkWeight,
                S::SharedGate,
            ],
        ),
        (P::GateUp, vec![S::XfMoe, S::ExpertGu]),
        (P::GateUp, vec![S::XfMoe, S::SharedGu]),
        (P::Swiglu, vec![S::SharedGu, S::XfShared]),
    ];
    for _slot in 0..8 {
        ops.push((P::Down, vec![S::ExpertGu, S::XfDown(0)])); // swiglu_pack(s)
        ops.push((P::Down, vec![S::XfDown(0), S::ExpertDown, S::TopkId])); // down(s)
    }
    // `SharedGu` is last read by the shared swiglu (phase Swiglu); its `..Down` end is lane C's
    // conservative rule, pinned by `plan_lifetimes_are_pinned` (m29) and by no op of this list.
    ops.push((P::Down, vec![S::XfShared, S::SharedOut])); // shared down
    ops.push((
        P::Combine,
        vec![
            S::ExpertDown,
            S::SharedOut,
            S::TopkId,
            S::TopkWeight,
            S::SharedGate,
        ],
    ));
    // The list is launch order on one stream: its phases never go back.
    assert!(
        ops.windows(2)
            .all(|w| (w[0].0 as usize) <= (w[1].0 as usize))
    );
    let d = dims();
    for rows in [1, 8] {
        let p = plan(&d, rows, &options(KvMode::Fp16, 8));
        let slots: Vec<Slot> =
            ops.iter()
                .flat_map(|(_, used)| used.clone())
                .fold(Vec::new(), |mut all, slot| {
                    if !all.contains(&slot) {
                        all.push(slot);
                    }
                    all
                });
        let uses = |slot: Slot| {
            let at: Vec<usize> = (0..ops.len())
                .filter(|&i| ops[i].1.contains(&slot))
                .collect();
            (at[0], at[at.len() - 1])
        };
        for &slot in &slots {
            let (first, last) = uses(slot);
            let planned = p.tensors()[p.index(slot).expect("slot")];
            assert!(
                planned.first <= ops[first].0 as usize,
                "{slot:?} starts late"
            );
            assert!(planned.last >= ops[last].0 as usize, "{slot:?} dies early");
        }
        let range = |slot| {
            let at = p.index(slot).expect("slot");
            let start = p.plan().offsets[at];
            (start, start + p.tensors()[at].bytes)
        };
        for (i, &a) in slots.iter().enumerate() {
            for &b in &slots[i + 1..] {
                let ((fa, la), (fb, lb)) = (uses(a), uses(b));
                if fa <= lb && fb <= la {
                    let (ra, rb) = (range(a), range(b));
                    assert!(
                        ra.1 <= rb.0 || rb.1 <= ra.0,
                        "{a:?} aliases {b:?} at {rows} rows"
                    );
                }
            }
        }
    }
}

#[test]
fn plan_refuses_bad_rows_and_scratch() {
    let d = dims();
    let opts = options(KvMode::Fp16, 8);
    for rows in [0, MAX_STEP_ROWS + 1, 16, 4096] {
        // The scratch matches the row count, so only the row bound can refuse the plan.
        let own =
            KernelScratch::new(&d, rows, router_bytes(&d, rows), KvMode::Fp16).expect("scratch");
        assert!(StepPlan::new(&d, rows, &opts, &own).is_err(), "{rows}");
    }
    // A scratch made for one row cannot size a plan of eight rows, even when its router bytes
    // are big enough: the row counts must agree.
    assert!(StepPlan::new(&d, 8, &opts, &scratch(&d, 1)).is_err());
    let wrong_rows = KernelScratch::new(&d, 2, router_bytes(&d, 8), KvMode::Fp16).expect("scratch");
    assert!(StepPlan::new(&d, 8, &opts, &wrong_rows).is_err());
    // A router scratch below 4 * rows * (experts + 1) is refused, one byte short included.
    let need = router_bytes(&d, 2);
    for short in [0, need - 1] {
        let low = KernelScratch::new(&d, 2, short, KvMode::Fp16).expect("scratch");
        assert!(StepPlan::new(&d, 2, &opts, &low).is_err(), "{short}");
    }
    let exact = KernelScratch::new(&d, 2, need, KvMode::Fp16).expect("scratch");
    assert!(StepPlan::new(&d, 2, &opts, &exact).is_ok());
}

#[test]
fn kernel_scratch_takes_the_merged_wrapper_sizes() {
    let d = dims();
    assert_eq!(router_bytes(&d, 1), 1_028);
    for (rows, part) in [(1, 1_056_768), (8, 8_454_144)] {
        let p = plan(&d, rows, &options(KvMode::Fp16, 8));
        let at = p.index(Slot::AttnPart).expect("attn part");
        assert_eq!(p.tensors()[at].bytes, part, "{rows} rows");
        let at = p.index(Slot::ArgmaxPart).expect("argmax part");
        assert_eq!(p.tensors()[at].bytes, 3_072);
        let at = p.index(Slot::Logits).expect("logits");
        assert_eq!(p.tensors()[at].bytes, rows * 993_280);
    }
}

#[test]
fn only_the_documented_phases_write_status_and_argmax_is_last() {
    let writers: Vec<Phase> = PHASES
        .iter()
        .copied()
        .filter(|&phase| writes_status(phase))
        .collect();
    // step_begin first, the embedding (RD-4c ORs mask 4 for a bad id), attn_prep (RD-8a ORs mask 4),
    // kv8_append (KV8-B ORs mask 4 on a bad pos word; redundant, see `writes_status`),
    // the router (mask 1), argmax (mask 2).
    assert_eq!(
        writers,
        [
            Phase::StepBegin,
            Phase::Embed,
            Phase::AttnPrep,
            Phase::Kv8Append,
            Phase::Route,
            Phase::Argmax
        ]
    );
    assert_eq!(PHASES[0], Phase::StepBegin);
    assert!(PHASES.windows(2).all(|w| (w[0] as usize) < (w[1] as usize)));
    assert_eq!(writers.last().copied(), Some(Phase::Argmax));
    assert!((Phase::Advance as usize) > Phase::Argmax as usize);
    assert!(!writes_status(Phase::Advance));
}

#[test]
fn kv_and_delta_sizes() {
    let d = dims();
    let f = KvPool::bytes_per_token(&d, KvMode::Fp16).unwrap();
    let q = KvPool::bytes_per_token(&d, KvMode::Int8G32).unwrap();
    assert_eq!((f, q), (20_480, 10_880));
    let ctx = 131_072;
    let lf = StateLayout::new(&d, &options(KvMode::Fp16, ctx)).unwrap();
    let lq = StateLayout::new(&d, &options(KvMode::Int8G32, ctx)).unwrap();
    assert_eq!((lf.kv.total, lq.kv.total), (f * ctx, q * ctx));
    assert_eq!(lf.delta.layers, 30);
    assert_eq!(lf.delta.total, 65_863_680);
    assert_eq!(
        lf.delta.rec_offset(0, 1).unwrap() - lf.delta.rec_offset(0, 0).unwrap(),
        2_097_152 + 98_304
    );
    assert!(lf.delta.rec_offset(1, 0).is_err() && lf.delta.rec_offset(0, 30).is_err());
    // The persistent words: one aligned slot each, appended in order, inside the state term.
    assert_eq!(
        (
            Word::StepParams.offset(),
            Word::ResultPair.offset(),
            Word::PrefillStatus.offset()
        ),
        (0, 256, 512)
    );
    assert_eq!(
        (
            Word::StepParams.bytes(),
            Word::ResultPair.bytes(),
            Word::PrefillStatus.bytes()
        ),
        (32, 8, 4)
    );
    assert_eq!(lf.words.total, 768);
    assert_eq!(lf.bytes().unwrap(), 2_750_219_008);
    // One whole K plane of the first attention layer, then the V plane right behind it.
    let (k, k_bytes) = lf.kv.plane_extent(0, Plane::K).unwrap();
    let (v, _) = lf.kv.plane_extent(0, Plane::V).unwrap();
    assert_eq!((k, v - k), (0, k_bytes));
    assert!(lf.kv.plane_extent(10, Plane::K).is_err());
}

#[test]
fn ranges_cover_exactly_the_prefix() {
    let d = dims();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let cap = 1024;
        let layout = StateLayout::new(&d, &options(mode, cap)).unwrap();
        for tokens in [0, 1, 5, cap] {
            let mut all = Vec::new();
            for layer in 0..layout.kv.layers {
                for r in layout.kv.prefix_ranges(layer, tokens).unwrap() {
                    assert!(r.device_offset + r.bytes <= layout.kv.total);
                    all.push((r.device_offset, r.device_offset + r.bytes));
                }
            }
            all.sort();
            assert!(all.windows(2).all(|w| w[0].1 <= w[1].0), "ranges overlap");
            let sum: usize = all.iter().map(|r| r.1 - r.0).sum();
            let want = KvPool::bytes_per_token(&d, mode).unwrap() * tokens;
            assert_eq!(sum, want, "{mode:?} tokens {tokens}");
        }
        assert!(layout.kv.prefix_ranges(0, cap + 1).is_err());
        assert!(layout.kv.prefix_ranges(layout.kv.layers, 1).is_err());
    }
}

#[test]
fn pinned_offsets() {
    let d = dims();
    let f = StateLayout::new(&d, &options(KvMode::Fp16, 1024)).unwrap();
    let ranges = |l: &StateLayout, layer, tokens| l.kv.prefix_ranges(layer, tokens).unwrap();
    let r: Vec<(usize, usize)> = ranges(&f, 1, 5)
        .iter()
        .map(|r| (r.device_offset, r.bytes))
        .collect();
    assert_eq!(r, vec![(2_097_152, 5_120), (2_097_152 + 1_048_576, 5_120)]);
    let q = StateLayout::new(&d, &options(KvMode::Int8G32, 1024)).unwrap();
    let r: Vec<(usize, usize)> = ranges(&q, 0, 5)
        .iter()
        .map(|r| (r.device_offset, r.bytes))
        .collect();
    // K codes of head 0 and head 1 (capacity cells apart), K scales, V codes, V scales.
    assert_eq!(
        r,
        vec![
            (0, 1_280),
            (262_144, 1_280),
            (524_288, 80),
            (524_288 + 16_384, 80),
            (557_056, 1_280),
            (557_056 + 262_144, 1_280),
            (1_081_344, 80),
            (1_081_344 + 16_384, 80),
        ]
    );
    assert_eq!(f.delta.conv_offset(0, 0).unwrap(), 2_097_152);
}

#[test]
fn kv_request_resolves() {
    assert_eq!(KvRequest::Auto.resolve(10, Some(10)).unwrap(), KvMode::Fp16);
    assert_eq!(
        KvRequest::Auto.resolve(11, Some(10)).unwrap(),
        KvMode::Int8G32
    );
    assert_eq!(
        KvRequest::Int8.resolve(1, Some(10)).unwrap(),
        KvMode::Int8G32
    );
    assert_eq!(KvRequest::Fp16.resolve(11, Some(10)).unwrap(), KvMode::Fp16);
    // No frozen budget: Auto is an error, an explicit request still works (REPLY-43 item 6).
    assert!(KvRequest::Auto.resolve(1, None).is_err());
    assert_eq!(KvRequest::Fp16.resolve(1, None).unwrap(), KvMode::Fp16);
    assert_eq!(dims().layers_of(AttentionKind::Full), 10);
}

#[test]
fn hostile_shapes_are_refused() {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    let mut shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
    shape.moe.inter += 256;
    assert!(StepDims::new(&shape, &inventory).is_err());
    let mut shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
    shape.delta.value_heads *= 2;
    assert!(StepDims::new(&shape, &inventory).is_err());
    let d = dims();
    assert!(
        StateLayout::new(
            &d,
            &StepOptions {
                delta_slots: 0,
                ..options(KvMode::Fp16, 8)
            }
        )
        .is_err()
    );
    assert!(StateLayout::new(&d, &options(KvMode::Fp16, 0)).is_err());
}

#[test]
fn arena_tensor_list_equals_the_slots() {
    let p = plan(&dims(), 1, &options(KvMode::Fp16, 8));
    let list: Vec<ArenaTensor> = p.arena_tensors();
    assert_eq!(list.len(), p.tensors().len());
    assert!(
        list.iter()
            .zip(p.tensors())
            .all(|(a, s)| (a.bytes, a.first, a.last) == (s.bytes, s.first, s.last))
    );
}

#[test]
fn delta_slots_address_distinct_state() {
    let d = dims();
    let mut opts = options(KvMode::Fp16, 8);
    opts.delta_slots = 3;
    let layout = StateLayout::new(&d, &opts).unwrap();
    let delta = &layout.delta;
    let stride = 2_097_152 + 98_304;
    assert_eq!(delta.total, 3 * 65_863_680);
    // The layout is [slot][layer]: slot 2, layer 29 is the last of 90 (slot, layer) pairs.
    assert_eq!(delta.rec_offset(2, 29).unwrap(), (2 * 30 + 29) * stride);
    assert_eq!(delta.conv_offset(1, 0).unwrap(), (30 * stride) + 2_097_152);
    let starts: Vec<usize> = (0..3)
        .flat_map(|slot| (0..30).map(move |layer| (slot, layer)))
        .map(|(slot, layer)| delta.rec_offset(slot, layer).unwrap())
        .collect();
    assert!(starts.windows(2).all(|w| w[1] - w[0] == stride));
    assert_eq!(starts[89] + stride, delta.total);
    assert!(delta.rec_offset(3, 0).is_err());
}

#[test]
fn int8_planes_round_up_to_the_alignment() {
    let d = dims();
    let layout = StateLayout::new(&d, &options(KvMode::Int8G32, 1001)).unwrap();
    let kv = &layout.kv;
    // 1001 cells x 32 bytes of scales is 32,032 bytes, rounded up to 32,256; the codes are
    // 512,512 bytes, already a multiple of 256, so only the scale planes need the rounding.
    assert_eq!(kv.plane_extent(0, Plane::K).unwrap(), (0, 512_512));
    assert_eq!(
        kv.plane_extent(0, Plane::KScales).unwrap(),
        (512_512, 32_256)
    );
    assert_eq!(kv.plane_extent(0, Plane::V).unwrap(), (544_768, 512_512));
    assert_eq!(
        kv.plane_extent(0, Plane::VScales).unwrap(),
        (1_057_280, 32_256)
    );
    assert_eq!(kv.plane_extent(1, Plane::K).unwrap().0, 1_089_536);
    assert_eq!(kv.total, kv.layers * 1_089_536);
    let mut runs = Vec::new();
    for layer in 0..kv.layers {
        for &plane in kv.planes() {
            assert_eq!(kv.plane_extent(layer, plane).unwrap().0 % 256, 0);
        }
        for r in kv.prefix_ranges(layer, 7).unwrap() {
            runs.push((r.device_offset, r.device_offset + r.bytes));
        }
    }
    runs.sort();
    assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "ranges overlap");
    assert!(runs.last().is_some_and(|r| r.1 <= kv.total));
}
