//! Unit tests of the pure part of the export (test seam U13): `check_import` and the little-endian
//! helpers. CPU only. The device copies are covered by the scratch device run of the packet.
use super::super::host::HostShape;
use super::{
    dims::{KvMode, StepDims, StepOptions},
    state::{
        StateIdentity,
        export::{
            DeltaExport, ExportedState, KvExport, check_import, f32_from_le, f32_to_le, plane_sizes,
        },
    },
    state_layout::StateLayout,
};
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};

fn dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(&shape, &inventory).expect("dims")
}

fn options(mode: KvMode, ctx: usize) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens: ctx,
        delta_slots: 1,
        forked_experts: false,
        layout: WeightLayout::Tiled,
    }
}

fn export_for(
    d: &StepDims,
    mode: KvMode,
    cap: usize,
    tokens: usize,
) -> (StateLayout, StateIdentity, ExportedState) {
    let layout = StateLayout::new(d, &options(mode, cap)).unwrap();
    let identity = StateIdentity::new(d, mode);
    let delta = (0..layout.delta.layers)
        .map(|_| DeltaExport {
            recurrent: vec![0.5; layout.delta.rec_values],
            conv: vec![0.25; layout.delta.conv_values],
        })
        .collect();
    let kv = (0..layout.kv.layers)
        .map(|o| KvExport {
            planes: plane_sizes(&layout.kv, o, tokens)
                .unwrap()
                .into_iter()
                .map(|n| vec![7u8; n])
                .collect(),
        })
        .collect();
    let exported = ExportedState {
        identity: identity.clone(),
        tokens,
        delta,
        kv,
    };
    (layout, identity, exported)
}

#[test]
fn import_checks() {
    let d = dims();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let (layout, id, good) = export_for(&d, mode, 64, 10);
        check_import(&layout, &id, &good).unwrap();
        let mut m = good.clone();
        m.tokens = 65;
        assert!(matches!(
            check_import(&layout, &id, &m),
            Err(crate::EngError::ContextLimit(_))
        ));
        let mut m = good.clone();
        m.delta[3].recurrent.pop();
        refused_with(
            check_import(&layout, &id, &m),
            "a DeltaNet layer has the wrong length",
        );
        let mut m = good.clone();
        m.delta.pop();
        refused_with(
            check_import(&layout, &id, &m),
            "a DeltaNet layer has the wrong length",
        );
        let mut m = good.clone();
        m.kv[2].planes[0].push(0);
        refused_with(
            check_import(&layout, &id, &m),
            "a KV plane has the wrong length",
        );
        let mut m = good.clone();
        m.kv.pop();
        refused_with(
            check_import(&layout, &id, &m),
            "the attention layer count differs",
        );
        let other = if mode == KvMode::Fp16 {
            KvMode::Int8G32
        } else {
            KvMode::Fp16
        };
        let mut m = good.clone();
        m.identity = StateIdentity::new(&d, other);
        refused_with(check_import(&layout, &id, &m), "the identity");
        let mut m = good.clone();
        m.identity.layout = "other";
        refused_with(check_import(&layout, &id, &m), "the identity");
    }
}

#[test]
fn le_round_trip() {
    let v = [
        0.0f32,
        -0.0,
        1.5,
        f32::NAN,
        f32::INFINITY,
        f32::MIN_POSITIVE / 2.0,
    ];
    let back = f32_from_le(&f32_to_le(&v));
    assert!(v.iter().zip(&back).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(f32_to_le(&[1.0]), vec![0, 0, 0x80, 0x3f]);
}

/// A snapshot of `tokens` positions whose KV planes are zero-filled (lazily allocated, so a deep
/// synthetic snapshot costs no resident memory) and whose DeltaNet vectors hold `delta_fill`.
fn zero_export(
    d: &StepDims,
    layout: &StateLayout,
    tokens: usize,
    delta_fill: f32,
) -> ExportedState {
    ExportedState {
        identity: StateIdentity::new(d, KvMode::Fp16),
        tokens,
        delta: (0..layout.delta.layers)
            .map(|_| DeltaExport {
                recurrent: vec![delta_fill; layout.delta.rec_values],
                conv: vec![delta_fill; layout.delta.conv_values],
            })
            .collect(),
        kv: (0..layout.kv.layers)
            .map(|o| KvExport {
                planes: plane_sizes(&layout.kv, o, tokens)
                    .expect("sizes")
                    .into_iter()
                    .map(|n| vec![0u8; n])
                    .collect(),
            })
            .collect(),
    }
}

#[test]
fn a_synthetic_snapshot_at_any_depth_up_to_the_capacity_passes_the_structure_check() {
    let d = dims();
    let depth = 32_768;
    let layout = StateLayout::new(&d, &options(KvMode::Fp16, depth)).unwrap();
    let id = StateIdentity::new(&d, KvMode::Fp16);
    for tokens in [1, 4_096, depth - 1, depth] {
        check_import(&layout, &id, &zero_export(&d, &layout, tokens, 0.5))
            .unwrap_or_else(|e| panic!("N = {tokens}: {e:?}"));
    }
    // Structure only: values are never inspected, so non-finite values pass too (no finiteness check is promised).
    check_import(&layout, &id, &zero_export(&d, &layout, depth, f32::NAN)).unwrap();
    // A larger allocation takes a shallower snapshot as well.
    let roomy = StateLayout::new(&d, &options(KvMode::Fp16, depth + 4_096)).unwrap();
    check_import(&roomy, &id, &zero_export(&d, &roomy, depth, 0.25)).unwrap();
}

#[test]
fn one_position_past_the_capacity_is_a_context_limit() {
    let d = dims();
    let capacity = 32_768;
    let layout = StateLayout::new(&d, &options(KvMode::Fp16, capacity)).unwrap();
    let id = StateIdentity::new(&d, KvMode::Fp16);
    let roomy = StateLayout::new(&d, &options(KvMode::Fp16, capacity + 1)).unwrap();
    let too_deep = zero_export(&d, &roomy, capacity + 1, 0.5);
    let error = check_import(&layout, &id, &too_deep).expect_err("N = capacity + 1");
    assert!(
        matches!(error, crate::EngError::ContextLimit(_)),
        "{error:?}"
    );
}

#[test]
fn the_import_flag_lets_a_poisoned_state_in() {
    use super::{book::StateBook, state::export::IMPORT_REWRITES_WORDS};
    // `import_state` does all of reset's work, so the book must take it from a `Poisoned` state
    // (REPLY-76 Q1); the flag is what `import_state` passes, so flipping it fails here.
    let mut book = StateBook::new(8, 7);
    book.poison();
    book.begin_import(7, IMPORT_REWRITES_WORDS)
        .expect("a poisoned state is imported with reset's work");
    assert!(StateBook::new(8, 7).begin_import(7, false).is_err());
}

#[test]
fn an_export_with_too_many_layers_is_refused_by_name() {
    let d = dims();
    for mode in [KvMode::Fp16, KvMode::Int8G32] {
        let (layout, id, good) = export_for(&d, mode, 64, 10);
        let mut m = good.clone();
        let extra = m.delta[0].clone();
        m.delta.push(extra);
        refused_with(
            check_import(&layout, &id, &m),
            "a DeltaNet layer has the wrong length",
        );
        let mut m = good.clone();
        let extra = m.kv[0].clone();
        m.kv.push(extra);
        refused_with(
            check_import(&layout, &id, &m),
            "the attention layer count differs",
        );
    }
}

/// The refusal is a `Checkpoint` error that names the rule.
fn refused_with(result: Result<(), crate::EngError>, needle: &str) {
    match result {
        Err(crate::EngError::Checkpoint(text)) => assert!(text.contains(needle), "{text}"),
        other => panic!("expected a Checkpoint refusal naming {needle:?}, got {other:?}"),
    }
}
