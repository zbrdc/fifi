//! Native state-only prefix roundtrip at a committed nonzero position, without a model.
use std::cell::RefMut;

use super::super::{
    book::Health,
    common_tests::dims,
    dims::{KvMode, StepDims, StepOptions},
    paging_cow_tests::{read_page, write_page},
    paging_plan::PagingPlan,
    paging_store::PagedKv,
    state_layout::{StateLayout, Word},
};
use super::{KvStorage, ResidentState};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{CudaContext, DeviceStream, device_present, resident::step_params::StepParams},
};

pub(super) fn pages(state: &ResidentState) -> Result<RefMut<'_, PagedKv>, EngError> {
    match &state.kv {
        KvStorage::Paged { store, .. } => store
            .try_borrow_mut()
            .map_err(|_| EngError::BadRequest("fixture page store is already borrowed".to_owned())),
        KvStorage::Resident(_) | KvStorage::Cached { .. } => Err(EngError::BadRequest(
            "fixture needs paged storage".to_owned(),
        )),
    }
}

pub(super) fn fill_delta(
    state: &ResidentState,
    stream: &DeviceStream,
    slot: usize,
    value: u8,
) -> Result<(), EngError> {
    let pool = state.layout().delta;
    let bytes = vec![value; pool.rec_values.max(pool.conv_values) * 4];
    for layer in 0..pool.layers {
        state.delta.write_at(
            stream,
            pool.rec_offset(slot, layer)?,
            &bytes[..pool.rec_values * 4],
        )?;
        state.delta.write_at(
            stream,
            pool.conv_offset(slot, layer)?,
            &bytes[..pool.conv_values * 4],
        )?;
    }
    Ok(())
}

pub(super) fn delta_matches(
    state: &ResidentState,
    stream: &DeviceStream,
    slot: usize,
    value: u8,
) -> Result<bool, EngError> {
    let pool = state.layout().delta;
    let mut bytes = vec![0; pool.rec_values.max(pool.conv_values) * 4];
    let mut exact = true;
    for layer in 0..pool.layers {
        for (offset, len) in [
            (pool.rec_offset(slot, layer)?, pool.rec_values * 4),
            (pool.conv_offset(slot, layer)?, pool.conv_values * 4),
        ] {
            state.delta.read_at(stream, offset, &mut bytes[..len])?;
            exact &= bytes[..len].iter().all(|b| *b == value);
        }
    }
    Ok(exact)
}

pub(super) fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 320,
        delta_slots: 2,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

pub(super) fn snapshot_budget(
    dims: &StepDims,
    opts: &StepOptions,
) -> Result<(usize, usize), EngError> {
    let layout = StateLayout::new(dims, opts)?;
    let bytes = layout.delta.layers * (layout.delta.rec_values + layout.delta.conv_values) * 4;
    let plan = PagingPlan::new(dims, opts.context_tokens)?;
    let limit = plan.backing_bytes + plan.pinned_bytes + bytes + plan.group.total + (64 << 10);
    Ok((bytes, limit))
}

pub(super) fn seeded_state(
    ctx: &CudaContext,
    stream: &DeviceStream,
    limit: usize,
    tokens: usize,
) -> Result<(ResidentState, Vec<u8>), EngError> {
    let mut state = ResidentState::allocate_paged(ctx, stream, &dims(), &options(), limit)?;
    state.begin_prefill()?;
    state.clear_prefill_status(stream)?;
    fill_delta(&state, stream, 0, 1)?;
    fill_delta(&state, stream, 1, 9)?;
    let original = write_page(ctx, stream, &mut *pages(&state)?, 1)?;
    let status = state.read_prefill_status(stream)?;
    state.revalidate(stream, tokens, status)?;
    Ok((state, original))
}

fn send_clone<T: Send + Clone>(_: &T) {}

struct Observed {
    state_bytes: usize,
    snapshot_charge: usize,
    tokens: bool,
    delta: bool,
    kv: bool,
    other_slot: bool,
    words: bool,
    start_block: bool,
    quota_released: bool,
    device_released: bool,
    pins_released: bool,
}

fn observe() -> Result<Observed, EngError> {
    let dimensions = dims();
    let (state_bytes, limit) = snapshot_budget(&dimensions, &options())?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let before_device = ctx.live_bytes();
    let before_pins = ctx.live_owners().pinned;
    let (mut state, original) = seeded_state(&ctx, &stream, limit, 37)?;
    let base = pages(&state)?.owned_host_bytes();
    eprintln!(
        "PKV_STATE_PREFIX_REAL_DEVICE_READY tokens={}",
        state.tokens()
    );
    let saved = state.export_prefix(&ctx, &stream)?;
    send_clone(&saved);
    let retained = saved.clone();
    drop(saved);
    let snapshot_charge = pages(&state)?.owned_host_bytes() - base;
    fill_delta(&state, &stream, 0, 2)?;
    fill_delta(&state, &stream, 1, 3)?;
    write_page(&ctx, &stream, &mut *pages(&state)?, 2)?;
    state.poison();
    state.import_prefix(&ctx, &stream, &retained)?;
    let tokens = state.tokens() == 37 && state.health() == Health::Valid;
    let delta = delta_matches(&state, &stream, 0, 1)?;
    let other_slot = delta_matches(&state, &stream, 1, 0)?;
    let kv = read_page(&ctx, &stream, &mut *pages(&state)?)? == original;
    let mut words = vec![0xff; state.layout().words.total];
    state.words.read_at(&stream, 0, &mut words)?;
    let words = words.iter().all(|b| *b == 0);
    state.write_step_params(&stream, &StepParams::new(0, 37)?)?;
    let mut params = [0; 32];
    state
        .words
        .read_at(&stream, Word::StepParams.offset(), &mut params)?;
    let start_block = StepParams::from_bytes(params) == StepParams::new(0, 37)?;
    drop(retained);
    let quota_released = pages(&state)?.owned_host_bytes() == base;
    drop(state);
    Ok(Observed {
        state_bytes,
        snapshot_charge,
        tokens,
        delta,
        kv,
        other_slot,
        words,
        start_block,
        quota_released,
        device_released: ctx.live_bytes() == before_device,
        pins_released: ctx.live_owners().pinned == before_pins,
    })
}

#[test]
fn state_only_prefix_restores_deltanet_and_aliased_kv_without_whole_copy() {
    if !device_present("state_only_prefix_restores_deltanet_and_aliased_kv_without_whole_copy") {
        return;
    }
    let x = observe().expect("native state-only prefix roundtrip");
    assert!(x.snapshot_charge >= x.state_bytes);
    assert!(x.snapshot_charge < x.state_bytes + (64 << 10));
    assert!(x.tokens && x.delta && x.kv && x.other_slot);
    assert!(x.words && x.start_block && x.quota_released);
    assert!(x.device_released && x.pins_released);
}
