//! Device tests of the streams, events and the pinned word (3090). The pinned
//! word has no tests file of its own, so its tests live here.
//!
//! The owner counts (`live_owners`, `destroy_calls`) are process-global, like
//! `LIVE_BYTES`, and every test makes its own contexts, so these tests assume
//! `--test-threads=1`, as the device suite is always run (decision 0040).
//!
//! Each refusal is checked by the text of its error as well as its variant, so
//! a test fails when another check refuses the call instead of the one under
//! test.

use crate::EngError;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, PinnedWord, WordValue,
};

/// The 8 bytes a test copies into the word: token `0x04030201`, status `0x0A0B0C0D`.
const WORD_BYTES: [u8; 8] = [0x01, 0x02, 0x03, 0x04, 0x0D, 0x0C, 0x0B, 0x0A];

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request<T>(result: Result<T, EngError>, case: &str, needle: &str) {
    let Err(error) = result else {
        panic!("{case}: expected a BadRequest, got Ok");
    };
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected BadRequest containing {needle:?}, got {error:?}"
    );
}

/// The first 8 bytes of a buffer, as the word copy reads them.
fn word_source(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, 8).expect("the first 8 bytes of the buffer")
}

/// Copies the word behind `stream`, records `event` after it and reads it.
fn copy_and_read(
    ctx: &CudaContext,
    stream: &DeviceStream,
    source: &DeviceBuffer,
    word: &PinnedWord,
) -> WordValue {
    let event = ctx.event().expect("event behind the copy");
    ctx.copy_word_to_host(stream, word_source(source), word)
        .expect("queue the copy");
    stream.record(&event).expect("record behind the copy");
    word.read_after(&event).expect("read the word")
}

const EXPECTED: WordValue = WordValue {
    token: 0x0403_0201,
    status: 0x0A0B_0C0D,
};

#[test]
fn streams_events_and_the_pinned_word_release_exactly_once() {
    if !testkit::device_present("streams_events_and_the_pinned_word_release_exactly_once") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let before = ctx.live_owners();
    let destroys = ctx.destroy_calls();
    let streams: Vec<DeviceStream> = (0..3).map(|_| ctx.stream().expect("stream")).collect();
    let events: Vec<DeviceEvent> = (0..2).map(|_| ctx.event().expect("event")).collect();
    let word: PinnedWord = ctx.pinned_word().expect("pinned word");
    let held = ctx.live_owners();
    assert_eq!(held.streams, before.streams + 3);
    assert_eq!(held.events, before.events + 2);
    assert_eq!(held.pinned, before.pinned + 8);
    drop(word);
    drop(events);
    drop(streams);
    assert_eq!(ctx.live_owners(), before, "every owner released its count");
    assert_eq!(ctx.destroy_calls(), destroys + 6, "one destroy per owner");
}

#[test]
fn an_event_before_the_copy_cannot_release_the_pinned_word() {
    if !testkit::device_present("an_event_before_the_copy_cannot_release_the_pinned_word") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let source = testkit::upload_bytes(&ctx, &WORD_BYTES);
    let stream = ctx.stream().expect("stream");
    let other_stream = ctx.stream().expect("second stream");
    let (early, late, unrecorded, elsewhere) = (
        ctx.event().expect("early event"),
        ctx.event().expect("late event"),
        ctx.event().expect("unrecorded event"),
        ctx.event().expect("event of the other stream"),
    );
    let word = ctx.pinned_word().expect("pinned word");

    stream.record(&early).expect("record early");
    ctx.copy_word_to_host(&stream, word_source(&source), &word)
        .expect("queue the copy");
    stream.record(&late).expect("record late");
    // The copy is the 2nd operation of its stream. Three records first give
    // `elsewhere` a count above that, so only the stream id can refuse it.
    let spares: Vec<DeviceEvent> = (0..3).map(|_| ctx.event().expect("spare")).collect();
    for spare in &spares {
        other_stream.record(spare).expect("record a spare");
    }
    other_stream.record(&elsewhere).expect("record elsewhere");

    assert_bad_request(
        word.read_after(&early),
        "an event before the copy",
        "before the word copy",
    );
    assert_bad_request(
        word.read_after(&unrecorded),
        "an event never recorded",
        "never recorded",
    );
    assert_bad_request(
        word.read_after(&elsewhere),
        "an event of another stream",
        "another stream",
    );
    assert_bad_request(
        ctx.copy_word_to_host(&stream, word_source(&source), &word),
        "a second copy before the first was read",
        "still holds",
    );
    // The refusals above left the copy pending, so the late event still reads it.
    let value = word
        .read_after(&late)
        .expect("the late event releases the word");
    assert_eq!(value, EXPECTED);
    assert_bad_request(
        word.read_after(&late),
        "a second read of one copy",
        "no copy",
    );
    // Fork and join: the second stream waits for the first, then both drain.
    other_stream.wait(&late).expect("wait for the late event");
    other_stream.synchronize().expect("drain the second stream");
    stream.synchronize().expect("drain the first stream");
}

#[test]
fn a_wait_on_an_event_that_was_never_recorded_is_refused() {
    if !testkit::device_present("a_wait_on_an_event_that_was_never_recorded_is_refused") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let waiter = ctx.stream().expect("waiting stream");
    let event = ctx.event().expect("event");

    // CUDA would accept this wait and order nothing.
    assert_bad_request(
        waiter.wait(&event),
        "a wait before any record",
        "never recorded",
    );
    ctx.ensure_usable().expect("the refusal is not terminal");
    stream.record(&event).expect("record");
    waiter.wait(&event).expect("a wait on a recorded event");
    waiter.synchronize().expect("drain the waiting stream");
}

#[test]
fn a_stream_or_event_of_another_context_is_refused() {
    if !testkit::device_present("a_stream_or_event_of_another_context_is_refused") {
        return;
    }
    let ctx = CudaContext::new().expect("first context");
    let other = CudaContext::new().expect("second context");
    let stream = ctx.stream().expect("stream");
    let source = testkit::upload_bytes(&ctx, &WORD_BYTES);
    let word = ctx.pinned_word().expect("word");
    let foreign_stream = other.stream().expect("stream of the other context");
    let foreign_event = other.event().expect("event of the other context");
    let foreign_source = testkit::upload_bytes(&other, &WORD_BYTES);
    let foreign_word = other.pinned_word().expect("word of the other context");

    // One foreign owner per case, everything else belongs to `ctx`.
    assert_bad_request(
        stream.record(&foreign_event),
        "record a foreign event",
        "event belongs",
    );
    assert!(
        foreign_event.mark().is_none(),
        "a refused record leaves the event unrecorded"
    );
    assert_bad_request(
        stream.wait(&foreign_event),
        "wait for a foreign event",
        "event belongs",
    );
    assert_bad_request(
        ctx.copy_word_to_host(&foreign_stream, word_source(&source), &word),
        "a foreign stream",
        "stream belongs",
    );
    assert_bad_request(
        ctx.copy_word_to_host(&stream, word_source(&foreign_source), &word),
        "a foreign buffer",
        "buffer belongs",
    );
    assert_bad_request(
        ctx.copy_word_to_host(&stream, word_source(&source), &foreign_word),
        "a foreign word",
        "pinned word belongs",
    );
    // A copy is pending now, so only the adapter check can refuse the foreign event.
    let event = ctx.event().expect("own event");
    ctx.copy_word_to_host(&stream, word_source(&source), &word)
        .expect("queue the copy");
    stream.record(&event).expect("record behind the copy");
    assert_bad_request(
        word.read_after(&foreign_event),
        "read_after with a foreign event",
        "event belongs",
    );
    assert_eq!(word.read_after(&event).expect("own event"), EXPECTED);
    ctx.ensure_usable()
        .expect("the refusals leave the backend usable");
}

#[test]
fn device_ranges_reject_empty_and_out_of_bounds_requests() {
    if !testkit::device_present("device_ranges_reject_empty_and_out_of_bounds_requests") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let buffer = testkit::upload_bytes(&ctx, &WORD_BYTES);

    assert_bad_request(buffer.range(0, 0), "an empty range", "does not fit");
    assert_bad_request(buffer.range(1, 8), "one byte past the end", "does not fit");
    assert_bad_request(
        buffer.range(usize::MAX, 2),
        "offset + bytes overflows",
        "does not fit",
    );
    let whole = buffer.range(0, 8).expect("the whole buffer");
    assert_eq!((whole.offset(), whole.bytes()), (0, 8));
    let tail = buffer.range(7, 1).expect("the last byte");
    assert_eq!((tail.offset(), tail.bytes()), (7, 1));
    ctx.ensure_usable()
        .expect("the refusals leave the backend usable");
}

#[test]
fn the_word_copy_rejects_a_short_or_misaligned_source_and_stays_free() {
    if !testkit::device_present("the_word_copy_rejects_a_short_or_misaligned_source_and_stays_free")
    {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let word = ctx.pinned_word().expect("word");
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&WORD_BYTES);
    let source = testkit::upload_bytes(&ctx, &bytes);

    let short = source.range(0, 4).expect("a 4-byte range");
    assert_bad_request(
        ctx.copy_word_to_host(&stream, short, &word),
        "fewer than 8 bytes",
        "8 bytes at a 4-byte aligned offset",
    );
    let unaligned = source.range(2, 8).expect("an 8-byte range at offset 2");
    assert_bad_request(
        ctx.copy_word_to_host(&stream, unaligned, &word),
        "an offset that is not a multiple of 4",
        "8 bytes at a 4-byte aligned offset",
    );
    // Neither refusal left a copy pending, so a valid copy still goes through.
    assert_eq!(copy_and_read(&ctx, &stream, &source, &word), EXPECTED);
    ctx.ensure_usable()
        .expect("the refusals leave the backend usable");
}
