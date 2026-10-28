use super::*;
#[cfg(feature = "cuda-fault-qualification")]
use crate::kernels::cuda::native;
use crate::kernels::cuda::{CapturePlan, testkit};

// Keep every assertion outside the scope owning CUDA resources. In particular a failed
// assertion during capture would latch terminal and poison the rest of this process.
fn check_refusals(cases: Vec<(Result<(), EngError>, &'static str)>) {
    for (answer, needle) in cases {
        assert!(
            matches!(&answer, Err(EngError::BadRequest(text)) if text.contains(needle)),
            "expected BadRequest containing {needle:?}, got {answer:?}"
        );
    }
}

#[test]
fn h2d_pinned_ranges_round_trip_and_same_stream_directions() {
    if !testkit::device_present("h2d_pinned_ranges_round_trip_and_same_stream_directions") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let event = ctx.event()?;
        let device = ctx.upload(&[0u8; 64])?;
        let mut source = ctx.pinned_buf(80)?;
        let mut output = ctx.pinned_buf(64)?;
        let data: Vec<u8> = (0..64)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        source.write(8, &data)?;
        for (offset, len) in [(0, 17), (17, 31), (48, 16)] {
            ctx.copy_from_pinned(&stream, &mut source, 8 + offset, device.range(offset, len)?)?;
        }
        let mut refused = vec![(source.write(0, &[1]), "outstanding")];
        let mut byte = [0];
        refused.push((source.read(0, &mut byte), "outstanding"));
        // A queued D2H into the H2D source is ordered after all prior DMA reads.
        ctx.copy_to_pinned(&stream, device.range(0, 64)?, &mut source, 0)?;
        // And a subsequent H2D sees that D2H, not the original shifted host bytes.
        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 64)?)?;
        ctx.copy_to_pinned(&stream, device.range(0, 64)?, &mut output, 0)?;
        stream.record(&event)?;
        source.wait(&event)?;
        output.wait(&event)?;
        let mut got = [0; 64];
        output.read(0, &mut got)?;
        source.write(0, &[0xA5])?;
        let mut changed = [0];
        source.read(0, &mut changed)?;
        Ok((got.to_vec(), data, changed, source.is_pending(), refused))
    })();
    let (got, expected, changed, pending, refused) = result.expect("owned H2D round trip");
    check_refusals(refused);
    assert_eq!(got, expected);
    assert_eq!(changed, [0xA5]);
    assert!(!pending);
}

#[test]
fn h2d_pinned_requires_event_covering_last_enqueue() {
    if !testkit::device_present("h2d_pinned_requires_event_covering_last_enqueue") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let foreign = CudaContext::new()?;
        let stream = ctx.stream()?;
        let side = ctx.stream()?;
        let foreign_stream = foreign.stream()?;
        let early = ctx.event()?;
        let middle = ctx.event()?;
        let never = ctx.event()?;
        let wrong = ctx.event()?;
        let alien = foreign.event()?;
        let late = ctx.event()?;
        let device = ctx.upload(&[0u8; 16])?;
        let mut source = ctx.pinned_buf(16)?;
        source.write(0, &[0x5A; 16])?;
        stream.record(&early)?;
        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 8)?)?;
        stream.record(&middle)?;
        ctx.copy_from_pinned(&stream, &mut source, 8, device.range(8, 8)?)?;
        for _ in 0..8 {
            side.record(&wrong)?;
        }
        foreign_stream.record(&alien)?;
        let mut refused = vec![
            (source.wait(&early), "before the pinned copy"),
            (source.wait(&middle), "before the pinned copy"),
            (source.wait(&never), "never recorded"),
            (source.wait(&wrong), "another stream than the pinned copy"),
            (source.wait(&alien), "belongs to another"),
        ];
        let unchanged = stream.enqueued();
        refused.push((
            ctx.copy_from_pinned(&side, &mut source, 0, device.range(0, 8)?),
            "another stream",
        ));
        refused.push((
            ctx.copy_to_pinned(&side, device.range(0, 8)?, &mut source, 0),
            "another stream",
        ));
        stream.synchronize()?;
        refused.push((source.write(0, &[1]), "outstanding"));
        let pending = source.is_pending();
        let after_refusals = stream.enqueued();
        let side_sequence = wrong.mark();
        stream.record(&late)?;
        source.wait(&late)?;
        source.write(0, &[2])?;
        refused.push((source.wait(&late), "no copy to wait for"));
        Ok((refused, pending, unchanged, after_refusals, side_sequence))
    })();
    let (refused, pending, before, after, side) = result.expect("owned event checks");
    check_refusals(refused);
    assert!(pending);
    assert_eq!(before, after);
    assert!(side.expect("recorded").1 > before);
}

#[test]
fn h2d_pinned_bounds_and_foreign_owners_refuse_before_enqueue() {
    if !testkit::device_present("h2d_pinned_bounds_and_foreign_owners_refuse_before_enqueue") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let foreign = CudaContext::new()?;
        let stream = ctx.stream()?;
        let alien_stream = foreign.stream()?;
        let device = ctx.upload(&[0u8; 32])?;
        let alien_device = foreign.upload(&[0u8; 32])?;
        let mut source = ctx.pinned_buf(16)?;
        let mut alien_source = foreign.pinned_buf(16)?;
        let before = stream.enqueued();
        let refused = vec![
            (source.write(0, &[]), "empty slice"),
            (source.write(16, &[1]), "lies outside"),
            (source.write(usize::MAX, &[1, 2]), "lies outside"),
            (
                ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 17)?),
                "lies outside",
            ),
            (
                ctx.copy_from_pinned(&stream, &mut source, 9, device.range(0, 8)?),
                "lies outside",
            ),
            (
                ctx.copy_from_pinned(&stream, &mut source, usize::MAX, device.range(0, 2)?),
                "lies outside",
            ),
            (
                ctx.copy_from_pinned(&alien_stream, &mut source, 0, device.range(0, 8)?),
                "stream belongs",
            ),
            (
                ctx.copy_from_pinned(&stream, &mut alien_source, 0, device.range(0, 8)?),
                "pinned buffer belongs",
            ),
            (
                ctx.copy_from_pinned(&stream, &mut source, 0, alien_device.range(0, 8)?),
                "buffer belongs",
            ),
            (device.range(0, 0).map(|_| ()), "does not fit"),
            (device.range(usize::MAX, 2).map(|_| ()), "does not fit"),
        ];
        // Exercise the raw wrapper's independent overflow/bounds proof without mapping
        // its expected validation error into a terminal backend failure.
        let raw = RawRange::new(ctx.owned(&device)?, 0, 8).ok_or_else(|| refuse("range"))?;
        let native_bounds =
            resident::copy_h2d_async_at(raw, source.raw()?, 9, stream.raw()?).is_err();
        let native_overflow =
            resident::copy_h2d_async_at(raw, source.raw()?, usize::MAX, stream.raw()?).is_err();
        let native_write = resident::pinned_write_from(source.raw()?, usize::MAX, &[1]).is_err();
        let after = stream.enqueued();
        let pending = source.is_pending();
        let mut zero = [1; 16];
        source.read(0, &mut zero)?;
        source.write(15, &[0xFF])?; // exact endpoint remains valid after all refusals
        ctx.ensure_usable()?;
        Ok((
            refused,
            before,
            after,
            pending,
            zero,
            native_bounds,
            native_overflow,
            native_write,
        ))
    })();
    let (refused, before, after, pending, zero, bounds, overflow, write) =
        result.expect("owned bounds checks");
    check_refusals(refused);
    assert_eq!(before, after);
    assert!(!pending);
    assert_eq!(zero, [0; 16]);
    assert!(bounds && overflow && write);
}

#[test]
fn h2d_pinned_capture_refuses_copy_write_and_captured_completion() {
    if !testkit::device_present("h2d_pinned_capture_refuses_copy_write_and_captured_completion") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let event = ctx.event()?;
        let device = ctx.upload(&[0u8; 8])?;
        let mut source = ctx.pinned_buf(8)?;
        let spare = ctx.pinned_buf(8)?;
        source.write(0, &[7; 8])?;
        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 8)?)?;
        stream.synchronize()?;
        let pinned_before = ctx.live_owners().pinned;
        let destroys = ctx.destroy_calls();
        let (graph, (mut refused, during, during_destroy)) = ctx.capture(
            &stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || {
                stream.record(&event)?;
                let refused = vec![
                    (source.write(0, &[1]), "not allowed inside a capture window"),
                    (
                        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 8)?),
                        "not allowed inside a capture window",
                    ),
                    (
                        ctx.pinned_buf(8).map(|_| ()),
                        "not allowed inside a capture window",
                    ),
                ];
                drop(spare); // must defer the synchronizing native free
                Ok((refused, ctx.live_owners().pinned, ctx.destroy_calls()))
            },
        )?;
        let after = ctx.live_owners().pinned;
        let after_destroy = ctx.destroy_calls();
        refused.push((source.wait(&event), "last recorded inside a capture"));
        let pending = source.is_pending();
        drop(graph);
        stream.record(&event)?;
        source.wait(&event)?;
        source.write(0, &[2])?;
        Ok((
            refused,
            pending,
            pinned_before,
            during,
            after,
            destroys,
            during_destroy,
            after_destroy,
        ))
    })();
    let (refused, pending, before, during, after, destroys, during_destroy, after_destroy) =
        result.expect("owned capture checks");
    check_refusals(refused);
    assert!(pending);
    assert_eq!(during, before);
    assert_eq!(after, before - 8);
    assert_eq!(during_destroy, destroys);
    assert_eq!(after_destroy, destroys + 1);
}

#[test]
fn h2d_pinned_pending_drop_waits_and_releases_exactly_once() {
    if !testkit::device_present("h2d_pinned_pending_drop_waits_and_releases_exactly_once") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let event = ctx.event()?;
        let device = ctx.upload(&[0u8; 64])?;
        let mut output = ctx.pinned_buf(64)?;
        let before = ctx.live_owners();
        let destroys = ctx.destroy_calls();
        let mut source = ctx.pinned_buf(64)?;
        source.write(0, &[0xD3; 64])?;
        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 64)?)?;
        let held = ctx.live_owners().pinned;
        let pending = source.is_pending();
        drop(source);
        let released = ctx.live_owners();
        let after_destroy = ctx.destroy_calls();
        ctx.copy_to_pinned(&stream, device.range(0, 64)?, &mut output, 0)?;
        stream.record(&event)?;
        output.wait(&event)?;
        let mut got = [0; 64];
        output.read(0, &mut got)?;
        Ok((
            before,
            held,
            pending,
            released,
            destroys,
            after_destroy,
            got,
        ))
    })();
    let (before, held, pending, released, destroys, after_destroy, got) =
        result.expect("owned teardown checks");
    assert!(pending);
    assert_eq!(held, before.pinned + 64);
    assert_eq!(released, before);
    assert_eq!(after_destroy, destroys + 1);
    assert_eq!(got, [0xD3; 64]);
}

#[test]
#[cfg(feature = "cuda-fault-qualification")]
fn h2d_pinned_failed_drop_leaks_and_latches_in_child() {
    const PATH: &str = "kernels::cuda::context::pinned_buf::h2d_tests::h2d_pinned_failed_drop_leaks_and_latches_in_child";
    if !testkit::latching_test_in_child(PATH, &["h2d failed drop policy checked"]) {
        return;
    }
    if !testkit::device_present("h2d_pinned_failed_drop_leaks_and_latches_in_child") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let device = ctx.upload(&[0u8; 8])?;
        let mut source = ctx.pinned_buf(8)?;
        source.write(0, &[0x5A; 8])?;
        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 8)?)?;
        let before = ctx.live_owners();
        let destroys = ctx.destroy_calls();
        let fault = {
            let _dispatch = ctx.enter()?;
            native::fault_illegal_address()
        };
        // Leave the native fault unmapped: pinned_free must itself wait, fail, leak,
        // and latch. Every subsequent owner drop must avoid a native destroy.
        drop(source);
        let after = ctx.live_owners();
        let attempts = ctx.destroy_calls();
        let terminal = TERMINAL.load(Ordering::Acquire);
        let refused = ctx.pinned_buf(8).map(|_| ());
        drop(device);
        drop(stream);
        let final_attempts = ctx.destroy_calls();
        Ok((
            fault.map_err(|e| e.status),
            before,
            after,
            destroys,
            attempts,
            final_attempts,
            terminal,
            refused,
        ))
    })();
    let (fault, before, after, destroys, attempts, final_attempts, terminal, refused) =
        result.expect("owned failed drop checks");
    assert_eq!(fault, Err(700));
    assert_eq!(
        before, after,
        "failed synchronize must not release pinned pages"
    );
    assert_eq!(attempts, destroys + 1);
    assert_eq!(final_attempts, attempts);
    assert!(terminal);
    assert!(matches!(refused, Err(EngError::Backend { .. })));
    println!("h2d failed drop policy checked");
}

#[test]
#[cfg(feature = "cuda-fault-qualification")]
fn h2d_pinned_terminal_refuses_access_and_skips_drop_in_child() {
    const PATH: &str = "kernels::cuda::context::pinned_buf::h2d_tests::h2d_pinned_terminal_refuses_access_and_skips_drop_in_child";
    if !testkit::latching_test_in_child(PATH, &["h2d terminal policy checked"]) {
        return;
    }
    if !testkit::device_present("h2d_pinned_terminal_refuses_access_and_skips_drop_in_child") {
        return;
    }
    let result = (|| -> Result<_, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let event = ctx.event()?;
        let device = ctx.upload(&[0u8; 8])?;
        let mut source = ctx.pinned_buf(8)?;
        source.write(0, &[9; 8])?;
        ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 8)?)?;
        stream.record(&event)?;
        let before = ctx.live_owners();
        let destroys = ctx.destroy_calls();
        let fault = {
            let _dispatch = ctx.enter()?;
            native::fault_illegal_address()
        };
        let mapped = fault.map_err(device_error);
        let mut byte = [0];
        let refused = vec![
            source.write(0, &[1]),
            source.read(0, &mut byte),
            source.wait(&event),
            ctx.copy_from_pinned(&stream, &mut source, 0, device.range(0, 8)?),
        ];
        let pending = source.is_pending();
        drop(source);
        drop(device);
        drop(event);
        drop(stream);
        Ok((
            mapped,
            refused,
            pending,
            before,
            ctx.live_owners(),
            destroys,
            ctx.destroy_calls(),
        ))
    })();
    let (mapped, refused, pending, before, after, destroys, attempts) =
        result.expect("owned terminal checks");
    assert!(matches!(mapped, Err(EngError::Backend { .. })));
    for error in refused {
        assert!(matches!(error, Err(EngError::Backend { .. })));
    }
    assert!(pending);
    assert_eq!(before, after);
    assert_eq!(destroys, attempts);
    println!("h2d terminal policy checked");
}
