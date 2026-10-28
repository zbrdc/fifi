//! Native page snapshot divergence through normal transfer/store APIs.
//! Covers the native RED-to-GREEN sharing slice and shared transfer fixtures.
use super::{
    common_tests::dims,
    paging_plan::PagingPlan,
    paging_store::{PageId, PagedKv},
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceStream, device_present},
};

const METADATA_ROOM: usize = 64 << 10;

pub(super) fn write_page(
    ctx: &CudaContext,
    stream: &DeviceStream,
    store: &mut PagedKv,
    value: u8,
) -> Result<Vec<u8>, EngError> {
    store.load(
        ctx,
        stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let bytes = write_loaded(ctx, stream, store, value)?;
    store.writeback(ctx, stream, 0)?;
    Ok(bytes)
}

pub(super) fn write_loaded(
    ctx: &CudaContext,
    stream: &DeviceStream,
    store: &PagedKv,
    value: u8,
) -> Result<Vec<u8>, EngError> {
    let range = store.resident(0)?;
    let payload_bytes = range.bytes() - size_of::<i32>();
    let mut bytes = vec![value; range.bytes()];
    bytes[payload_bytes..].copy_from_slice(&0_i32.to_le_bytes());
    let mut pin = ctx.pinned_buf(bytes.len())?;
    let event = ctx.event()?;
    pin.write(0, &bytes)?;
    ctx.copy_from_pinned(stream, &mut pin, 0, range)?;
    stream.record(&event)?;
    pin.wait(&event)?;
    bytes.truncate(payload_bytes);
    Ok(bytes)
}

pub(super) fn read_page(
    ctx: &CudaContext,
    stream: &DeviceStream,
    store: &mut PagedKv,
) -> Result<Vec<u8>, EngError> {
    store.load(
        ctx,
        stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let bytes = read_loaded(ctx, stream, store)?;
    store.writeback(ctx, stream, 0)?;
    Ok(bytes)
}

pub(super) fn read_loaded(
    ctx: &CudaContext,
    stream: &DeviceStream,
    store: &PagedKv,
) -> Result<Vec<u8>, EngError> {
    let range = store.resident(0)?;
    let mut pin = ctx.pinned_buf(range.bytes())?;
    let event = ctx.event()?;
    ctx.copy_to_pinned(stream, range, &mut pin, 0)?;
    stream.record(&event)?;
    pin.wait(&event)?;
    let mut bytes = vec![0; range.bytes()];
    pin.read(0, &mut bytes)?;
    bytes.truncate(bytes.len() - size_of::<i32>());
    Ok(bytes)
}

fn send_clone<T: Send + Clone>(_: &T) {}

struct Observed {
    save_bytes: usize,
    detached_bytes: usize,
    group_bytes: usize,
    live_diverged: bool,
    restored_original: bool,
    diverged_again: bool,
    clone_holds_charge: bool,
    last_drop_releases: bool,
    device_released: bool,
    pinned_released: bool,
}

fn observe() -> Result<Observed, EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    // Room for metadata and one detached group, but not a second whole backing.
    let limit = plan.backing_bytes + plan.pinned_bytes + plan.group.total + METADATA_ROOM;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let before_device = ctx.live_bytes();
    let before_pinned = ctx.live_owners().pinned;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, limit)?;
    let original = write_page(&ctx, &stream, &mut store, 1)?;
    let base = store.owned_host_bytes();
    eprintln!("PKV_COW_REAL_DEVICE_READY");
    let saved = store.snapshot(&ctx, &stream, 37)?;
    send_clone(&saved);
    let saved_clone = saved.clone();
    let after_save = store.owned_host_bytes();
    let changed = write_page(&ctx, &stream, &mut store, 2)?;
    let after_divergence = store.owned_host_bytes();
    let live_diverged = read_page(&ctx, &stream, &mut store)? == changed;
    store.restore(&stream, &saved)?;
    let restored_original = read_page(&ctx, &stream, &mut store)? == original;
    let changed_again = write_page(&ctx, &stream, &mut store, 3)?;
    let diverged_again = read_page(&ctx, &stream, &mut store)? == changed_again;
    let held = store.owned_host_bytes();
    drop(saved);
    let clone_holds_charge = store.owned_host_bytes() == held;
    drop(saved_clone);
    let last_drop_releases = store.owned_host_bytes() == base;
    drop(store);
    Ok(Observed {
        save_bytes: after_save.saturating_sub(base),
        detached_bytes: after_divergence.saturating_sub(after_save),
        group_bytes: plan.group.total,
        live_diverged,
        restored_original,
        diverged_again,
        clone_holds_charge,
        last_drop_releases,
        device_released: ctx.live_bytes() == before_device,
        pinned_released: ctx.live_owners().pinned == before_pinned,
    })
}

#[test]
fn paged_snapshot_shares_backing_detaches_one_group_and_releases_last_hold() {
    if !device_present("paged_snapshot_shares_backing_detaches_one_group_and_releases_last_hold") {
        return;
    }
    let x = observe().expect("native shared page snapshot");
    assert!(x.save_bytes > 0 && x.save_bytes < METADATA_ROOM);
    assert!(x.detached_bytes >= x.group_bytes);
    assert!(x.detached_bytes < x.group_bytes + METADATA_ROOM);
    assert!(x.live_diverged && x.restored_original && x.diverged_again);
    assert!(x.clone_holds_charge && x.last_drop_releases);
    assert!(x.device_released && x.pinned_released);
}
