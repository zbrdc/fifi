//! Device tests (3090) of `allocate`, `read_at` and `write_at` (seam 1: the Rust
//! wrapper against the C entries against the driver).
//!
//! The byte counts and the queued work are process-wide, so these tests assume
//! `--test-threads=1`, as the device suite is always run (decision 0040).

use crate::EngError;
use crate::kernels::cuda::native;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, PackCall};

#[test]
fn request_counters_keep_size_direction_and_lease_separate_without_a_device() {
    let first = usize::MAX;
    let other = usize::MAX - 1;
    super::BUFFER_REQUESTS.with_borrow(|counts| {
        assert!(!counts.contains_key(&first));
        assert!(!counts.contains_key(&other));
    });
    super::record_allocation(first, 1_048_592);
    super::record_allocation(first, 1_048_592);
    super::record_allocation(other, 5_457_920);
    super::record_ranged_h2d(first, 2_097_152);
    super::BUFFER_REQUESTS.with_borrow_mut(|counts| {
        let first = counts.remove(&first).unwrap();
        let other = counts.remove(&other).unwrap();
        assert_eq!(first.allocations, [(1_048_592, 2)].into());
        assert_eq!(first.ranged_h2d, [(2_097_152, 1)].into());
        assert_eq!(other.allocations, [(5_457_920, 1)].into());
        assert!(other.ranged_h2d.is_empty());
        assert_eq!(
            first
                .allocations
                .iter()
                .map(|(bytes, calls)| bytes * calls)
                .sum::<usize>(),
            2_097_184,
            "request bytes retain the sixteen-byte tail, not a rounded MiB"
        );
    });
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires leased CUDA; public request-hook qualification, not physical allocation stats"]
fn public_request_hooks_count_success_refuse_failures_and_exclude_upload() {
    let ctx = CudaContext::new().expect("leased CUDA context");
    let stream = ctx.stream().unwrap();
    let before = ctx.buffer_requests();
    assert!(matches!(ctx.allocate(0), Err(EngError::BadRequest(_))));
    let _uploaded = ctx.upload(&[7; 7]).unwrap();
    let unchanged = ctx.buffer_requests();
    assert_eq!(unchanged.allocations, before.allocations);
    assert_eq!(unchanged.ranged_h2d, before.ranged_h2d);

    let buffer = ctx.allocate(64).unwrap();
    let mut allocations = before.allocations.clone();
    *allocations.entry(64).or_default() += 1;
    let allocated = ctx.buffer_requests();
    assert_eq!(allocated.allocations, allocations);
    assert_eq!(allocated.ranged_h2d, before.ranged_h2d);
    for (offset, bytes) in [(0, &[][..]), (64, &[1][..]), (usize::MAX, &[1][..])] {
        assert!(matches!(
            buffer.write_at(&stream, offset, bytes),
            Err(EngError::BadRequest(_))
        ));
    }
    let rejected = ctx.buffer_requests();
    assert_eq!(rejected.allocations, allocations);
    assert_eq!(rejected.ranged_h2d, before.ranged_h2d);

    buffer.write_at(&stream, 4, &[1; 5]).unwrap();
    let mut actual = [0; 5];
    buffer.read_at(&stream, 4, &mut actual).unwrap();
    assert_eq!(actual, [1; 5]);
    let mut h2d = before.ranged_h2d.clone();
    *h2d.entry(5).or_default() += 1;
    let written = ctx.buffer_requests();
    assert_eq!(written.allocations, allocations);
    assert_eq!(written.ranged_h2d, h2d);

    let other = CudaContext::new().unwrap();
    let other_stream = other.stream().unwrap();
    other
        .allocate(17)
        .unwrap()
        .write_at(&other_stream, 0, &[2; 3])
        .unwrap();
    let isolated = ctx.buffer_requests();
    assert_eq!(isolated.allocations, allocations);
    assert_eq!(isolated.ranged_h2d, h2d);
    ctx.ensure_usable().unwrap();
    println!("PUBLIC_REQUEST_HOOKS_NATIVE_PASS");
}

/// Bytes of the fixed fragment-order `xf` buffer.
const XF_FRAGMENT: usize = 1_048_592;
/// Packs queued ahead of the call under test in the ordering test. A pack of 256
/// tokens of 32768 values moves about 50 MB, so the backlog keeps the stream busy
/// for far longer than the host needs to queue the call behind it.
const BACKLOG_LAUNCHES: usize = 1000;

/// A deterministic byte pattern: byte `i` is `i % 251`, so it never repeats within a page.
fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| u8::try_from(i % 251).expect("below 251"))
        .collect()
}

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &str) {
    let Err(error) = result else {
        panic!("{case}: expected a BadRequest, got Ok");
    };
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected BadRequest containing {needle:?}, got {error:?}"
    );
}

/// The whole buffer, read through `stream`.
fn read_all(buffer: &DeviceBuffer, stream: &DeviceStream) -> Vec<u8> {
    let mut bytes = vec![0u8; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes).expect("read_at");
    bytes
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn device_copy_keeps_a_frozen_range_and_checks_owners() {
    let ctx = CudaContext::new().expect("native CUDA context");
    let stream = ctx.stream().expect("stream");
    let source = ctx.allocate(96).expect("source");
    let saved = ctx.allocate(96).expect("saved");
    let bytes = pattern(96);
    source.write_at(&stream, 0, &bytes).unwrap();
    saved.write_at(&stream, 0, &[0; 96]).unwrap();
    ctx.copy_device(
        &stream,
        source.range(7, 33).unwrap(),
        saved.range(9, 33).unwrap(),
    )
    .expect("D2D copy");
    source.write_at(&stream, 7, &[255; 33]).unwrap();
    let mut expected = vec![0; 96];
    expected[9..42].copy_from_slice(&bytes[7..40]);
    assert_eq!(read_all(&saved, &stream), expected);
    assert_bad_request(
        ctx.copy_device(
            &stream,
            source.range(0, 32).unwrap(),
            saved.range(0, 31).unwrap(),
        ),
        "unequal ranges",
        "equal",
    );
    assert_bad_request(
        ctx.copy_device(
            &stream,
            source.range(0, 32).unwrap(),
            source.range(32, 32).unwrap(),
        ),
        "same allocation",
        "distinct",
    );
    let other = CudaContext::new().expect("other context");
    let foreign = other.allocate(32).expect("foreign allocation");
    assert!(matches!(
        ctx.copy_device(
            &stream,
            source.range(0, 32).unwrap(),
            foreign.range(0, 32).unwrap()
        ),
        Err(EngError::BadRequest(_))
    ));
    ctx.ensure_usable().unwrap();
    println!("STATE_TIP_D2D_COPY_NATIVE_PASS");
}

#[test]
fn allocate_returns_a_buffer_of_the_asked_size_and_refuses_zero_bytes() {
    if !testkit::device_present(
        "allocate_returns_a_buffer_of_the_asked_size_and_refuses_zero_bytes",
    ) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let before = ctx.live_bytes();

    // A zero-byte request is a caller mistake: it must not reach the adapter,
    // which would answer with a status that latches the backend.
    let zero = ctx.allocate(0).map(|_| ());
    assert_bad_request(zero, "zero bytes", "zero bytes");
    ctx.ensure_usable().expect("the backend is still usable");
    assert_eq!(
        ctx.live_bytes(),
        before,
        "a refused allocation holds nothing"
    );

    let buffer = ctx.allocate(4096).expect("allocate 4096");
    assert_eq!(buffer.bytes(), 4096);
    assert_eq!(ctx.live_bytes(), before + 4096);
    // The allocation is usable as a buffer: bytes go in and the same bytes come out.
    let data = pattern(4096);
    buffer.write_at(&stream, 0, &data).expect("write_at");
    assert_eq!(read_all(&buffer, &stream), data);
    drop(buffer);
    assert_eq!(ctx.live_bytes(), before, "dropping gives the bytes back");

    // More than the device holds is the non-terminal "does not fit" error.
    let total = ctx.memory_info().expect("memory info").total;
    let too_big = ctx.allocate(total + (1 << 30));
    assert!(
        matches!(too_big, Err(EngError::MemoryPlan(_))),
        "an allocation larger than the device must be MemoryPlan"
    );
    assert_eq!(
        ctx.live_bytes(),
        before,
        "a failed allocation holds nothing"
    );
    ctx.ensure_usable()
        .expect("MemoryPlan leaves the backend usable");
}

#[test]
fn ranged_io_round_trips_bytes_at_nonzero_offsets() {
    if !testkit::device_present("ranged_io_round_trips_bytes_at_nonzero_offsets") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let buffer = ctx.allocate(4096).expect("allocate");
    let before = ctx.live_bytes();

    let whole = pattern(4096);
    buffer.write_at(&stream, 0, &whole).expect("write all");
    let splice: Vec<u8> = (0..24).map(|i| 0xE0 + i).collect();
    buffer.write_at(&stream, 1000, &splice).expect("write 24");

    let mut got = vec![0u8; 24];
    buffer.read_at(&stream, 1000, &mut got).expect("read 24");
    assert_eq!(got, splice, "the spliced bytes");
    let mut expected = whole.clone();
    expected[1000..1024].copy_from_slice(&splice);
    assert_eq!(read_all(&buffer, &stream), expected, "the whole buffer");

    // An odd offset and an odd length: no alignment is required of the caller.
    buffer.write_at(&stream, 5, &[1, 2, 3]).expect("write odd");
    let mut odd = [0u8; 3];
    buffer.read_at(&stream, 5, &mut odd).expect("read odd");
    assert_eq!(odd, [1, 2, 3]);

    // The last byte of the buffer is inside it.
    buffer.write_at(&stream, 4095, &[0x7E]).expect("write last");
    let mut last = [0u8; 1];
    buffer.read_at(&stream, 4095, &mut last).expect("read last");
    assert_eq!(last, [0x7E]);

    assert_eq!(ctx.live_bytes(), before, "ranged io allocates nothing");
}

#[test]
fn ranged_io_refuses_bad_bounds_and_empty_slices_without_latching() {
    if !testkit::device_present("ranged_io_refuses_bad_bounds_and_empty_slices_without_latching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let buffer = ctx.allocate(4096).expect("allocate");
    buffer
        .write_at(&stream, 0, &[0x5A; 4096])
        .expect("write the sentinel");

    // (offset, length, text of the refusal)
    let cases = [
        (0, 0, "empty"),
        (4096, 1, "outside a buffer of"),
        (4095, 2, "outside a buffer of"),
        (0, 4097, "outside a buffer of"),
        (1, 4096, "outside a buffer of"),
        (usize::MAX, 1, "outside a buffer of"),
        (usize::MAX - 3, 8, "outside a buffer of"),
    ];
    for (offset, len, needle) in cases {
        let case = format!("offset {offset}, length {len}");
        let mut out = vec![0u8; len];
        assert_bad_request(
            buffer.read_at(&stream, offset, &mut out),
            &format!("read, {case}"),
            needle,
        );
        assert_bad_request(
            buffer.write_at(&stream, offset, &out),
            &format!("write, {case}"),
            needle,
        );
    }

    // A stream of another context is refused before any native call.
    let other = CudaContext::new().expect("a second context");
    let foreign = other.stream().expect("foreign stream");
    assert_bad_request(
        buffer.read_at(&foreign, 0, &mut [0u8; 8]),
        "read through a foreign stream",
        "belongs to another adapter",
    );
    assert_bad_request(
        buffer.write_at(&foreign, 0, &[0u8; 8]),
        "write through a foreign stream",
        "belongs to another adapter",
    );

    assert_eq!(
        read_all(&buffer, &stream),
        vec![0x5A; 4096],
        "no refused call changed the buffer"
    );
    ctx.ensure_usable()
        .expect("refusals leave the backend usable");
}

/// The pack of one token of 256 values from `x` into the fragment-order `xf`.
fn one_token_pack<'b>(x: &'b DeviceBuffer, xf: &'b DeviceBuffer) -> PackCall<'b> {
    PackCall {
        x: x.range(0, 1024).expect("x range"),
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        m: 1,
        k: 256,
    }
}

/// Queues `BACKLOG_LAUNCHES` large packs on `stream` and returns without waiting.
fn queue_backlog(ctx: &CudaContext, stream: &DeviceStream, x: &DeviceBuffer, xf: &DeviceBuffer) {
    for _ in 0..BACKLOG_LAUNCHES {
        let pack = PackCall {
            x: x.range(0, x.bytes()).expect("backlog x"),
            xf: xf.range(0, xf.bytes()).expect("backlog xf"),
            m: 256,
            k: 32768,
        };
        ctx.pack(stream, pack).expect("queue a backlog pack");
    }
}

#[test]
fn ranged_io_runs_behind_the_work_queued_on_its_stream() {
    if !testkit::device_present("ranged_io_runs_behind_the_work_queued_on_its_stream") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (x_bytes, xf_bytes) = native::pack_bytes(256, 32768).expect("backlog sizes");
    let backlog_x = testkit::upload_filled(&ctx, 0, x_bytes);
    let backlog_xf = testkit::upload_filled(&ctx, 0, xf_bytes);
    let x = testkit::upload_f32(&ctx, &testkit::input(1, 256));

    // The reference: the same pack with nothing ahead of it, drained before it is read.
    let reference = testkit::upload_filled(&ctx, 0xAA, XF_FRAGMENT);
    ctx.pack(&stream, one_token_pack(&x, &reference))
        .expect("reference pack");
    stream.synchronize().expect("drain the reference");
    let packed = testkit::read_bytes(&ctx, &reference, XF_FRAGMENT);
    assert!(
        packed.iter().any(|&byte| byte != 0 && byte != 0xAA) && packed.contains(&0),
        "the pack must change the sentinel, or the comparison proves nothing"
    );

    // Read: the pack is queued behind the backlog and nothing waits for it. The
    // read returns the packed bytes only if the copy itself ran behind them.
    let target = testkit::upload_filled(&ctx, 0xAA, XF_FRAGMENT);
    queue_backlog(&ctx, &stream, &backlog_x, &backlog_xf);
    ctx.pack(&stream, one_token_pack(&x, &target))
        .expect("queue the pack");
    let mut seen = vec![0u8; XF_FRAGMENT];
    target
        .read_at(&stream, 0, &mut seen)
        .expect("read behind the pack");
    assert!(
        seen == packed,
        "read_at returned bytes from before the queued pack"
    );

    // Write: the pack zeroes the whole buffer first, so a write that landed before
    // it would be erased. The write must come after the pack.
    let target = testkit::upload_filled(&ctx, 0xAA, XF_FRAGMENT);
    queue_backlog(&ctx, &stream, &backlog_x, &backlog_xf);
    ctx.pack(&stream, one_token_pack(&x, &target))
        .expect("queue the pack");
    let mark = [0x5C; 64];
    target
        .write_at(&stream, 8192, &mark)
        .expect("write behind the pack");
    let mut expected = packed.clone();
    expected[8192..8192 + 64].copy_from_slice(&mark);
    stream.synchronize().expect("drain");
    let after = testkit::read_bytes(&ctx, &target, XF_FRAGMENT);
    assert!(after == expected, "write_at landed before the queued pack");
}

#[test]
fn pageable_io_crosses_two_chunk_boundaries_and_an_odd_tail() {
    if !testkit::device_present("pageable_io_crosses_two_chunk_boundaries_and_an_odd_tail") {
        return;
    }
    let ctx = CudaContext::new().expect("chunk context");
    let stream = ctx.stream().expect("chunk stream");
    let len = 2 * super::CHUNK_LIMIT + 17;
    let offset = 11;
    let buffer = ctx.allocate(offset + len + 13).expect("chunk allocation");
    let allocated = ctx.live_bytes();
    buffer
        .write_at(&stream, 0, &[0xD5; 11])
        .expect("prefix guard");
    buffer
        .write_at(&stream, offset + len, &[0x5D; 13])
        .expect("suffix guard");
    let source = pattern(len);
    buffer
        .write_at(&stream, offset, &source)
        .expect("three H2D chunks");
    let mut output = vec![0; len];
    buffer
        .read_at(&stream, offset, &mut output)
        .expect("three D2H chunks");
    assert_eq!(output.iter().zip(&source).position(|(a, b)| a != b), None);
    let mut prefix = [0; 11];
    let mut suffix = [0; 13];
    buffer
        .read_at(&stream, 0, &mut prefix)
        .expect("read prefix");
    buffer
        .read_at(&stream, offset + len, &mut suffix)
        .expect("read suffix");
    assert_eq!(prefix, [0xD5; 11]);
    assert_eq!(suffix, [0x5D; 13]);
    assert_eq!(ctx.live_bytes(), allocated, "no extra device allocation");
    println!("PKV_PAGEABLE_CHUNKS_PASS bytes={len} chunks=3 offset={offset}");
}

#[test]
fn pageable_staging_is_bounded_even_for_a_maximum_size_request() {
    let staging = super::pageable_staging(usize::MAX).expect("bounded staging");
    assert_eq!(staging.len(), super::CHUNK_LIMIT);
    assert!(staging.capacity() <= super::CHUNK_LIMIT);
    assert!(staging.iter().all(|byte| *byte == 0));
}
