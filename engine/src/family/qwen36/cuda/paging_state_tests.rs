//! Paged state construction chooses storage before any full resident KV allocation.
use super::{
    book::Health,
    common_tests::dims,
    dims::{KvMode, StepOptions},
    paging_pipeline_tests::read,
    paging_plan::PagingPlan,
    state::ResidentState,
    state_layout::{Plane, StateLayout, Word},
};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{CudaContext, device_present},
};

fn options(mode: KvMode, layout: WeightLayout) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens: 320,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    }
}

fn counts(ctx: &CudaContext) -> [usize; 5] {
    let owners = ctx.live_owners();
    [
        ctx.live_bytes(),
        owners.pinned,
        owners.events,
        owners.streams,
        owners.graphs,
    ]
}

struct Observed {
    before: [usize; 5],
    allocated: [usize; 5],
    released: [usize; 5],
    expected_device: usize,
    expected_pinned: usize,
    full_resident_kv_bytes: usize,
    actual_paged_kv_bytes: usize,
    no_resident_plane: bool,
    logical_layout_preserved: bool,
    reset_valid: bool,
    zero_words: bool,
    rejected: usize,
    invalid_owners_unchanged: bool,
}

fn observe() -> Result<Observed, EngError> {
    let dimensions = dims();
    let opts = options(KvMode::Int8G32, WeightLayout::Native);
    let logical = StateLayout::new(&dimensions, &opts)?;
    let plan = PagingPlan::new(&dimensions, opts.context_tokens)?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let before = counts(&ctx);
    // Existing real state construction is the control, not a fake context or dummy allocation.
    let resident = ResidentState::allocate(&ctx, &stream, &dimensions, &opts)?;
    let control = resident.kv_bytes() == logical.kv.total
        && read(&ctx, &stream, resident.word(Word::StepParams)?)?
            .iter()
            .all(|b| *b == 0);
    drop(resident);
    if !control || counts(&ctx) != before {
        return Err(EngError::BadRequest(
            "default resident state control failed".to_owned(),
        ));
    }
    eprintln!("PAGED_STATE_DEFAULT_RESIDENT_CONTROL_PASS");
    let mut state = ResidentState::allocate_paged(&ctx, &stream, &dimensions, &opts, 8 << 20)?;
    let allocated = counts(&ctx);
    let actual_paged_kv_bytes = state.kv_bytes();
    let no_resident_plane = matches!(state.kv_plane(0, Plane::K), Err(EngError::BadRequest(_)));
    let logical_layout_preserved = *state.layout() == logical
        && state.delta_rec(0, 0)?.bytes() > 0
        && state.delta_conv(0, 0)?.bytes() > 0;
    state.poison();
    state.reset(&stream)?;
    let reset_valid = state.health() == Health::Valid && state.tokens() == 0;
    let zero_words = read(&ctx, &stream, state.word(Word::StepParams)?)?
        .iter()
        .all(|b| *b == 0);
    drop(state);
    let released = counts(&ctx);
    let mut rejected = 0;
    let mut invalid_owners_unchanged = true;
    for (mode, layout, limit) in [
        (KvMode::Fp16, WeightLayout::Native, 8 << 20),
        (KvMode::Int8G32, WeightLayout::Tiled, 8 << 20),
        (KvMode::Int8G32, WeightLayout::Native, 0),
        (KvMode::Int8G32, WeightLayout::Native, 1),
        (
            KvMode::Int8G32,
            WeightLayout::Native,
            plan.backing_bytes + plan.pinned_bytes - 1,
        ),
    ] {
        let prior = counts(&ctx);
        let invalid = options(mode, layout);
        let result = ResidentState::allocate_paged(&ctx, &stream, &dimensions, &invalid, limit);
        if matches!(
            &result,
            Err(EngError::BadRequest(_)) | Err(EngError::MemoryPlan(_))
        ) {
            rejected += 1;
        }
        drop(result);
        invalid_owners_unchanged &= counts(&ctx) == prior;
    }
    Ok(Observed {
        before,
        allocated,
        released,
        expected_device: logical.delta.total + logical.words.total + plan.device_bytes,
        expected_pinned: plan.pinned_bytes,
        full_resident_kv_bytes: logical.kv.total,
        actual_paged_kv_bytes,
        no_resident_plane,
        logical_layout_preserved,
        reset_valid,
        zero_words,
        rejected,
        invalid_owners_unchanged,
    })
}

#[test]
fn paged_state_owns_only_two_kv_slots_preserves_logical_layout_and_reset() {
    if !device_present("paged_state_owns_only_two_kv_slots_preserves_logical_layout_and_reset") {
        return;
    }
    let x = observe().expect("real paged state constructor");
    assert_eq!(
        x.allocated,
        [
            x.before[0] + x.expected_device,
            x.before[1] + x.expected_pinned,
            x.before[2] + 2,
            x.before[3],
            x.before[4],
        ]
    );
    assert_eq!(x.released, x.before);
    assert!(x.actual_paged_kv_bytes < x.full_resident_kv_bytes);
    assert!(x.no_resident_plane && x.logical_layout_preserved && x.reset_valid && x.zero_words);
    assert_eq!(x.rejected, 5);
    assert!(x.invalid_owners_unchanged);
    eprintln!("PAGED_STATE slots_only=true logical=true reset=true released=true rejected=5");
}

include!("cache_model_route_tests.rs");
