//! Real page-store ownership; assertions run only after device owners have dropped.

use super::{
    common_tests::dims,
    paging_plan::PagingPlan,
    paging_store::{PageId, PagedKv},
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceRange, DeviceStream, PackCall, device_present},
};

// Requested device bytes, pinned bytes, events, streams, graphs.
fn counts(ctx: &CudaContext) -> [usize; 5] {
    let owned = ctx.live_owners();
    [
        ctx.live_bytes(),
        owned.pinned,
        owned.events,
        owned.streams,
        owned.graphs,
    ]
}

fn ownership_counts() -> Result<(usize, [[usize; 5]; 3]), EngError> {
    let dimensions = dims();
    let ctx = CudaContext::new()?;
    let before = counts(&ctx);
    eprintln!("PKV_PAGE_STORE_REAL_DEVICE_READY");
    let store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    let backing = store.backing_bytes();
    let allocated = counts(&ctx);
    drop(store);
    let released = counts(&ctx);
    Ok((backing, [before, allocated, released]))
}

#[test]
fn paged_store_owns_two_bounded_slots_and_releases_them() {
    if !device_present("paged_store_owns_two_bounded_slots_and_releases_them") {
        return;
    }
    let (backing, [before, allocated, released]) =
        ownership_counts().expect("allocate native paged KV store");
    assert_eq!(backing, 5_570_560);
    assert_eq!(
        allocated,
        [
            before[0] + 557_064,
            before[1] + 557_064,
            before[2] + 2,
            before[3],
            before[4]
        ]
    );
    assert_eq!(released, before);
}

fn read_owned(
    ctx: &CudaContext,
    stream: &DeviceStream,
    range: DeviceRange<'_>,
) -> Result<Vec<u8>, EngError> {
    let mut pinned = ctx.pinned_buf(range.bytes())?;
    let event = ctx.event()?;
    ctx.copy_to_pinned(stream, range, &mut pinned, 0)?;
    stream.record(&event)?;
    pinned.wait(&event)?;
    let mut bytes = vec![0; range.bytes()];
    pinned.read(0, &mut bytes)?;
    Ok(bytes)
}

fn zero_slot_observations() -> Result<[Vec<u8>; 4], EngError> {
    let dimensions = dims();
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    eprintln!("PKV_PAGE_LOAD_REAL_DEVICE_READY");
    store.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let first = read_owned(&ctx, &stream, store.resident(0)?)?;
    let pattern: Vec<u8> = (0..first.len())
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let mut pinned = ctx.pinned_buf(pattern.len())?;
    let event = ctx.event()?;
    pinned.write(0, &pattern)?;
    ctx.copy_from_pinned(&stream, &mut pinned, 0, store.resident(0)?)?;
    stream.record(&event)?;
    pinned.wait(&event)?;
    store.load(
        &ctx,
        &stream,
        1,
        PageId {
            full_layer: 9,
            group: 1,
        },
    )?;
    let second = read_owned(&ctx, &stream, store.resident(1)?)?;
    let first_after = read_owned(&ctx, &stream, store.resident(0)?)?;
    Ok([first, second, first_after, pattern])
}

#[test]
fn paged_store_loads_zero_groups_into_independent_slots() {
    if !device_present("paged_store_loads_zero_groups_into_independent_slots") {
        return;
    }
    let [first, second, first_after, pattern] =
        zero_slot_observations().expect("load real paged KV groups");
    assert_eq!(first.len(), 278_532);
    assert_eq!(second.len(), 278_532);
    assert!(first.iter().all(|byte| *byte == 0));
    assert!(second.iter().all(|byte| *byte == 0));
    assert_eq!(
        first_after, pattern,
        "loading the second slot must not overwrite the first"
    );
}

struct BudgetObservations {
    before: [usize; 5],
    results: Vec<(bool, [usize; 5])>,
}

fn budget_observations() -> Result<BudgetObservations, EngError> {
    let dimensions = dims();
    let ctx = CudaContext::new()?;
    let before = counts(&ctx);
    let floor = 5_570_560 + 557_064;
    let mut observations = Vec::new();
    for limit in [0, floor, floor + size_of::<PagedKv>() - 1] {
        let error = PagedKv::allocate(&ctx, &dimensions, 320, limit).err();
        observations.push((matches!(error, Some(EngError::MemoryPlan(_))), counts(&ctx)));
    }
    Ok(BudgetObservations {
        before,
        results: observations,
    })
}

#[test]
fn paged_store_refuses_host_limits_without_retaining_owners() {
    if !device_present("paged_store_refuses_host_limits_without_retaining_owners") {
        return;
    }
    let observed = budget_observations().expect("observe host budget refusals");
    for (refused, after) in observed.results {
        assert!(refused, "backing, pins and metadata must all fit");
        assert_eq!(
            after, observed.before,
            "refusal must leave no live allocation"
        );
    }
}

struct SlotObservations {
    refused: Vec<bool>,
    before: Vec<u8>,
    after: Vec<u8>,
    owners_unchanged: bool,
}

fn invalid_slot_observations() -> Result<SlotObservations, EngError> {
    let dimensions = dims();
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    let before = counts(&ctx);
    let mut refused = vec![store.resident(0).is_err(), store.resident(2).is_err()];
    for (slot, full_layer, group) in [
        (2, 0, 0),
        (usize::MAX, 0, 0),
        (0, 10, 0),
        (0, 0, 2),
        (0, usize::MAX, usize::MAX),
    ] {
        refused.push(matches!(
            store.load(&ctx, &stream, slot, PageId { full_layer, group }),
            Err(EngError::BadRequest(_))
        ));
    }
    store.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let bytes_before = read_owned(&ctx, &stream, store.resident(0)?)?;
    refused.push(matches!(
        store.load(
            &ctx,
            &stream,
            0,
            PageId {
                full_layer: 9,
                group: 1
            }
        ),
        Err(EngError::BadRequest(_))
    ));
    let bytes_after = read_owned(&ctx, &stream, store.resident(0)?)?;
    Ok(SlotObservations {
        refused,
        before: bytes_before,
        after: bytes_after,
        owners_unchanged: counts(&ctx) == before,
    })
}

#[test]
fn paged_store_refuses_invalid_and_occupied_slots_without_overwrite() {
    if !device_present("paged_store_refuses_invalid_and_occupied_slots_without_overwrite") {
        return;
    }
    let observed = invalid_slot_observations().expect("observe invalid page and slot refusals");
    assert!(observed.refused.into_iter().all(|value| value));
    assert_eq!(observed.before, observed.after);
    assert!(observed.owners_unchanged);
}

fn upload_owned(
    ctx: &CudaContext,
    stream: &DeviceStream,
    range: DeviceRange<'_>,
    bytes: &[u8],
) -> Result<(), EngError> {
    let mut pinned = ctx.pinned_buf(bytes.len())?;
    let event = ctx.event()?;
    pinned.write(0, bytes)?;
    ctx.copy_from_pinned(stream, &mut pinned, 0, range)?;
    stream.record(&event)?;
    pinned.wait(&event)
}

fn plane_seed(plan: &PagingPlan) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0; plan.slot_bytes];
    for (index, plane) in plan.group.planes().iter().enumerate() {
        let (offset, len) = plan.group.plane_extent(0, *plane)?;
        for (i, byte) in bytes[offset..offset + len].iter_mut().enumerate() {
            *byte = (i as u8)
                .wrapping_mul(29)
                .wrapping_add(17 + index as u8 * 41);
        }
    }
    Ok(bytes)
}

fn writeback_observations() -> Result<([Vec<u8>; 3], bool), EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    store.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 9,
            group: 1,
        },
    )?;
    let bytes = store.resident(0)?.bytes();
    let input = ctx.allocate(17 * 256 * size_of::<f32>())?;
    let reference = ctx.allocate(bytes)?;
    let row: Vec<u8> = (0..17 * 256)
        .flat_map(|i| ((i as f32 + 1.0) / 16.0).to_le_bytes())
        .collect();
    upload_owned(&ctx, &stream, input.range(0, row.len())?, &row)?;
    let seed = plane_seed(&plan)?;
    upload_owned(&ctx, &stream, reference.range(0, bytes)?, &seed)?;
    upload_owned(&ctx, &stream, store.resident(0)?, &seed)?;
    ctx.pack(
        &stream,
        PackCall {
            x: input.range(0, row.len())?,
            xf: reference.range(0, bytes)?,
            m: 17,
            k: 256,
        },
    )?;
    let expected = read_owned(&ctx, &stream, reference.range(0, bytes)?)?;
    ctx.pack(
        &stream,
        PackCall {
            x: input.range(0, row.len())?,
            xf: store.resident(0)?,
            m: 17,
            k: 256,
        },
    )?;
    // No synchronize or readback of slot0 may hide a missing consumer fence here.
    eprintln!("PKV_WRITEBACK_QUEUED_REAL_KERNEL");
    store.writeback(&ctx, &stream, 0)?;
    let unloaded = store.resident(0).is_err();
    store.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    store.load(
        &ctx,
        &stream,
        1,
        PageId {
            full_layer: 9,
            group: 1,
        },
    )?;
    let actual = read_owned(&ctx, &stream, store.resident(1)?)?;
    let other = read_owned(&ctx, &stream, store.resident(0)?)?;
    Ok(([expected, actual, other], unloaded))
}

#[test]
fn paged_store_writeback_preserves_queued_kernel_bytes_across_reload() {
    if !device_present("paged_store_writeback_preserves_queued_kernel_bytes_across_reload") {
        return;
    }
    let ([expected, actual, other], unloaded) =
        writeback_observations().expect("write back real paged KV group");
    assert!(unloaded);
    assert!(
        expected.iter().any(|byte| *byte != 0),
        "queued kernel must produce nonzero bytes"
    );
    let plan = PagingPlan::new(&dims(), 320).expect("physical fixture layout");
    for plane in plan.group.planes() {
        let (offset, len) = plan.group.plane_extent(0, *plane).expect("fixture plane");
        assert!(
            expected[offset..offset + len].iter().any(|byte| *byte != 0),
            "each payload plane must contain nonzero bytes: {plane:?}"
        );
    }
    assert_eq!(
        actual, expected,
        "writeback and reload must preserve every payload byte"
    );
    assert!(
        other.iter().all(|byte| *byte == 0),
        "other page must remain isolated"
    );
    assert_eq!(
        &actual[actual.len() - 4..],
        &[0; 4],
        "local control is not persisted"
    );
}

struct TransferRefusals {
    refused: Vec<bool>,
    before: Vec<u8>,
    after: Vec<u8>,
    unloaded: bool,
    owners_unchanged: bool,
}

fn transfer_refusals() -> Result<TransferRefusals, EngError> {
    let dimensions = dims();
    let ctx = CudaContext::new()?;
    let foreign = CudaContext::new()?;
    let stream = ctx.stream()?;
    let other_stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    store.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let owners = counts(&ctx);
    let before = read_owned(&ctx, &stream, store.resident(0)?)?;
    let mut refused = vec![
        matches!(
            store.load(
                &ctx,
                &other_stream,
                1,
                PageId {
                    full_layer: 0,
                    group: 1
                }
            ),
            Err(EngError::BadRequest(_))
        ),
        matches!(
            store.writeback(&ctx, &other_stream, 0),
            Err(EngError::BadRequest(_))
        ),
        matches!(
            store.load(
                &foreign,
                &stream,
                1,
                PageId {
                    full_layer: 0,
                    group: 1
                }
            ),
            Err(EngError::BadRequest(_))
        ),
        matches!(
            store.writeback(&foreign, &stream, 0),
            Err(EngError::BadRequest(_))
        ),
    ];
    for slot in [1, 2, usize::MAX] {
        refused.push(matches!(
            store.writeback(&ctx, &stream, slot),
            Err(EngError::BadRequest(_))
        ));
    }
    let after = read_owned(&ctx, &stream, store.resident(0)?)?;
    let unloaded = store.resident(1).is_err();
    store.load(
        &ctx,
        &stream,
        1,
        PageId {
            full_layer: 0,
            group: 1,
        },
    )?;
    store.writeback(&ctx, &stream, 0)?;
    refused.push(matches!(
        store.writeback(&ctx, &stream, 0),
        Err(EngError::BadRequest(_))
    ));
    Ok(TransferRefusals {
        refused,
        before,
        after,
        unloaded,
        owners_unchanged: owners == counts(&ctx),
    })
}

#[test]
fn paged_store_refuses_foreign_transfers_without_losing_resident_state() {
    if !device_present("paged_store_refuses_foreign_transfers_without_losing_resident_state") {
        return;
    }
    let observed = transfer_refusals().expect("observe normal transfer refusals");
    assert!(observed.refused.into_iter().all(|refused| refused));
    assert_eq!(observed.before, observed.after);
    assert!(observed.unloaded);
    assert!(observed.owners_unchanged);
}

struct DuplicateObservation {
    refused: bool,
    before: Vec<u8>,
    after: Vec<u8>,
    owners_unchanged: bool,
}

fn duplicate_observation() -> Result<DuplicateObservation, EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    let page = PageId {
        full_layer: 9,
        group: 1,
    };
    store.load(&ctx, &stream, 0, page)?;
    let owners = counts(&ctx);
    upload_owned(&ctx, &stream, store.resident(0)?, &plane_seed(&plan)?)?;
    let before = read_owned(&ctx, &stream, store.resident(0)?)?;
    eprintln!("PKV_DUPLICATE_PAGE_REAL_DEVICE_READY");
    let refused = matches!(
        store.load(&ctx, &stream, 1, page),
        Err(EngError::BadRequest(_))
    );
    store.writeback(&ctx, &stream, 0)?;
    if !refused {
        // Demonstrate the lost update without replaying a dangling pointer or corrupting CUDA.
        store.writeback(&ctx, &stream, 1)?;
    }
    store.load(&ctx, &stream, 0, page)?;
    let after = read_owned(&ctx, &stream, store.resident(0)?)?;
    Ok(DuplicateObservation {
        refused,
        before,
        after,
        owners_unchanged: owners == counts(&ctx),
    })
}

#[test]
fn paged_store_refuses_duplicate_residency_without_losing_updates() {
    if !device_present("paged_store_refuses_duplicate_residency_without_losing_updates") {
        return;
    }
    let observed = duplicate_observation().expect("observe real duplicate-page behavior");
    eprintln!(
        "PKV_DUPLICATE_OBSERVATION refused={} bytes_equal={}",
        observed.refused,
        observed.before == observed.after
    );
    assert!(observed.refused, "a page must not have two resident copies");
    assert!(
        observed.before == observed.after,
        "duplicate residency must not lose a modified page"
    );
    assert!(observed.owners_unchanged);
}

fn control_observations() -> Result<[Vec<u8>; 3], EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    let page = PageId {
        full_layer: 9,
        group: 1,
    };
    store.load(&ctx, &stream, 0, page)?;
    let mut seed = plane_seed(&plan)?;
    seed[plan.group.total..].copy_from_slice(&37_i32.to_le_bytes());
    upload_owned(&ctx, &stream, store.resident(0)?, &seed)?;
    let advanced = read_owned(&ctx, &stream, store.resident(0)?)?;
    store.writeback(&ctx, &stream, 0)?;
    store.load(&ctx, &stream, 1, page)?;
    let alternate = read_owned(&ctx, &stream, store.resident(1)?)?;
    store.writeback(&ctx, &stream, 1)?;
    store.load(&ctx, &stream, 0, page)?;
    let original = read_owned(&ctx, &stream, store.resident(0)?)?;
    Ok([advanced, alternate, original])
}

#[test]
fn paged_store_resets_advanced_local_control_on_both_reload_slots() {
    if !device_present("paged_store_resets_advanced_local_control_on_both_reload_slots") {
        return;
    }
    let [advanced, alternate, original] =
        control_observations().expect("reload actual advanced control");
    let payload = advanced.len() - 4;
    assert_eq!(&advanced[payload..], &37_i32.to_le_bytes());
    for reloaded in [alternate, original] {
        assert_eq!(
            &reloaded[payload..],
            &[0; 4],
            "append control must not persist"
        );
        assert!(
            advanced[..payload] == reloaded[..payload],
            "payload must survive control reset"
        );
    }
}
