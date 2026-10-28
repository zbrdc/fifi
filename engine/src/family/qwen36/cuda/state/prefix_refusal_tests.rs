//! Prefix admission and validation must refuse before state mutation or native enqueue.
use super::super::{
    book::Health,
    common_tests::dims,
    dims::KvMode,
    paging_cow_tests::{read_loaded, write_loaded},
    paging_store::PageId,
};
use super::{
    ResidentState,
    paged_prefix_tests::{
        delta_matches, fill_delta, options, pages, seeded_state, snapshot_budget,
    },
};
use crate::{
    EngError,
    kernels::cuda::{CapturePlan, CudaContext, device_present},
};

struct Observed {
    quota: bool,
    wrong_stream: bool,
    wrong_domain: bool,
    wrong_mode: bool,
    capture: bool,
    unchanged: bool,
    slot: bool,
    delta: bool,
    released: bool,
}

fn observe() -> Result<Observed, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let other_stream = ctx.stream()?;
    let (_, limit) = snapshot_budget(&dims(), &options())?;
    let (mut source, _) = seeded_state(&ctx, &stream, limit, 37)?;
    let base = pages(&source)?.owned_host_bytes();
    let saved = source.export_prefix(&ctx, &stream)?;
    let held = pages(&source)?.owned_host_bytes();
    let queued = (stream.enqueued(), other_stream.enqueued());
    eprintln!(
        "PKV_PREFIX_REFUSAL_REAL_DEVICE_READY tokens={}",
        source.tokens()
    );
    let quota = matches!(
        source.export_prefix(&ctx, &stream),
        Err(EngError::MemoryPlan(_))
    );
    let wrong_stream = matches!(
        source.import_prefix(&ctx, &other_stream, &saved),
        Err(EngError::BadRequest(_))
    );
    let mut unchanged = source.health() == Health::Valid
        && source.tokens() == 37
        && pages(&source)?.owned_host_bytes() == held
        && (stream.enqueued(), other_stream.enqueued()) == queued;

    let (mut target, _) = seeded_state(&ctx, &stream, limit, 3)?;
    fill_delta(&target, &stream, 0, 9)?;
    pages(&target)?.load(
        &ctx,
        &stream,
        0,
        PageId {
            full_layer: 0,
            group: 0,
        },
    )?;
    let expected = write_loaded(&ctx, &stream, &*pages(&target)?, 2)?;
    let before = stream.enqueued();
    let target_bytes = pages(&target)?.owned_host_bytes();
    let wrong_domain = matches!(
        target.import_prefix(&ctx, &stream, &saved),
        Err(EngError::BadRequest(_))
    );
    unchanged &= target.health() == Health::Valid
        && target.tokens() == 3
        && pages(&target)?.owned_host_bytes() == target_bytes
        && stream.enqueued() == before;
    let slot = read_loaded(&ctx, &stream, &*pages(&target)?)? == expected;
    let delta = delta_matches(&target, &stream, 0, 9)?;

    let mut fp16 = options();
    fp16.kv_mode = KvMode::Fp16;
    let mut resident = ResidentState::allocate(&ctx, &stream, &dims(), &fp16)?;
    let resident_saved = resident.export_prefix(&ctx, &stream)?;
    let before = stream.enqueued();
    let wrong_mode = matches!(
        resident.import_prefix(&ctx, &stream, &saved),
        Err(EngError::Checkpoint(_))
    ) && matches!(
        source.import_prefix(&ctx, &stream, &resident_saved),
        Err(EngError::Checkpoint(_))
    );
    unchanged &= stream.enqueued() == before && source.tokens() == 37 && resident.tokens() == 0;

    let before = stream.enqueued();
    // No captured operation is allowed, so there is no buffer to pin into a graph.
    let captured = ctx.capture(
        &stream,
        CapturePlan {
            pins: &[],
            holds: Vec::new(),
        },
        || {
            let export = matches!(
                source.export_prefix(&ctx, &stream),
                Err(EngError::BadRequest(_))
            );
            let import = matches!(
                source.import_prefix(&ctx, &stream, &saved),
                Err(EngError::BadRequest(_))
            );
            if export && import {
                Err::<(), _>(EngError::BadRequest(
                    "end own prefix guard probe".to_owned(),
                ))
            } else {
                Err(EngError::BadRequest(
                    "prefix capture guard failed".to_owned(),
                ))
            }
        },
    );
    let capture = matches!(captured, Err(EngError::BadRequest(ref text)) if text == "end own prefix guard probe")
        && !stream.is_capturing()?
        && stream.enqueued() == before;
    unchanged &= source.health() == Health::Valid
        && source.tokens() == 37
        && pages(&source)?.owned_host_bytes() == held;
    drop(saved);
    let released = pages(&source)?.owned_host_bytes() == base;
    Ok(Observed {
        quota,
        wrong_stream,
        wrong_domain,
        wrong_mode,
        capture,
        unchanged,
        slot,
        delta,
        released,
    })
}

#[test]
fn prefix_quota_identity_stream_and_capture_refusals_leave_live_state_unchanged() {
    if !device_present(
        "prefix_quota_identity_stream_and_capture_refusals_leave_live_state_unchanged",
    ) {
        return;
    }
    let x = observe().expect("real native prefix admission and validation");
    assert!(x.quota && x.wrong_stream && x.wrong_domain && x.wrong_mode && x.capture);
    assert!(x.unchanged && x.slot && x.delta && x.released);
}
