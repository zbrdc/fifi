//! Tests of the raw mailbox memory (TCAP C1a). The host-only tests run the accessors over an ordinary
//! allocation standing in for the mapping; the device test allocates the real thing.

use super::super::{
    RawRange, copy_d2h_async, pinned_alloc, pinned_free, pinned_read_into, stream_create,
    stream_destroy, stream_synchronize,
};
use super::*;
use crate::kernels::cuda::testkit::device_present;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// The accessors over ordinary host memory, for the host-only tests here and in the context layer: the result must
/// never reach `mapped_free`, and its device view is never used.
pub(in crate::kernels::cuda) fn over_host_memory(words: &mut [u64]) -> RawMapped {
    let ptr = NonNull::new(words.as_mut_ptr().cast::<c_void>()).expect("a slice is not null");
    RawMapped {
        host: ptr,
        device: RawDevice {
            ptr,
            bytes: size_of_val(words),
        },
    }
}

#[test]
fn the_accessors_round_trip_and_refuse_ranges_outside_the_mapping_or_off_the_word() {
    let mut backing = [0u64; 8];
    let mapped = over_host_memory(&mut backing);
    let row = [1u32, 2, 3, 4];
    assert!(mapped.write_words(16, &row));
    let mut back = [0u32; 4];
    assert!(mapped.read_words(16, &mut back));
    assert_eq!(back, row);
    assert!(mapped.store_release(8, 0xDEAD_BEEF_0000_0001));
    assert_eq!(mapped.load_acquire(8), Some(0xDEAD_BEEF_0000_0001));
    // The 4-byte words at 16 are the low and high half of the u64 at 16 in memory order.
    assert!(mapped.write_words(24, &[7, 9]));
    assert_eq!(
        mapped.load_acquire(24),
        Some(u64::from(7u32) | (u64::from(9u32) << 32))
    );
    let mut untouched = [0xAAAA_AAAAu32; 2];
    let refused: [(&str, bool); 6] = [
        ("a read past the end", mapped.read_words(60, &mut untouched)),
        (
            "a read of a late offset",
            mapped.read_words(72, &mut untouched),
        ),
        (
            "a read off the 4-byte word",
            mapped.read_words(2, &mut untouched),
        ),
        ("a write past the end", mapped.write_words(60, &untouched)),
        (
            "an offset that overflows",
            mapped.read_words(usize::MAX - 3, &mut untouched),
        ),
        (
            "a release store off the 8-byte word",
            mapped.store_release(12, 1),
        ),
    ];
    for (what, accepted) in refused {
        assert!(!accepted, "{what} was accepted");
    }
    assert_eq!(
        untouched, [0xAAAA_AAAA; 2],
        "a refused read leaves the output alone"
    );
    assert_eq!(
        mapped.load_acquire(64),
        None,
        "an acquire load past the end"
    );
    assert_eq!(
        mapped.load_acquire(4),
        None,
        "an acquire load off the 8-byte word"
    );
    assert!(
        mapped.read_words(0, &mut []),
        "an empty read inside the mapping is fine"
    );
    let floats = [1.5f32, -0.0, f32::MIN_POSITIVE / 2.0, f32::INFINITY];
    assert!(mapped.write_f32s(32, &floats));
    let mut round = [0f32; 4];
    assert!(mapped.read_f32s(32, &mut round));
    assert_eq!(
        round.map(f32::to_bits),
        floats.map(f32::to_bits),
        "float bits survive"
    );
}

/// The host half of the protocol on a real second thread: a payload written before a release store is seen
/// whole by a reader whose acquire load saw that store. (x86 gives the order anyway; the aarch64 Xavier is
/// where a weaker implementation would show, which is why the accessors are atomics and not plain stores.)
/// Two words: the sequence the writer releases at 0 and the acknowledgement the reader releases at 8.
#[test]
fn a_release_store_publishes_the_words_written_before_it() {
    const ROUNDS: u64 = 20_000;
    const WORDS: usize = 64;
    let mut backing = vec![0u64; 2 + WORDS / 2];
    let mapped = Arc::new(over_host_memory(&mut backing));
    let deadline = Instant::now() + Duration::from_secs(20);
    let reader = {
        let mapped = Arc::clone(&mapped);
        thread::spawn(move || {
            let mut payload = [0u32; WORDS];
            let mut bad = 0u64;
            for round in 1..=ROUNDS {
                while mapped.load_acquire(0).expect("in bounds") < round {
                    if Instant::now() > deadline {
                        return u64::MAX;
                    }
                    std::hint::spin_loop();
                }
                assert!(mapped.read_words(16, &mut payload));
                bad += payload.iter().filter(|&&w| u64::from(w) != round).count() as u64;
                assert!(mapped.store_release(8, round));
            }
            bad
        })
    };
    for round in 1..=ROUNDS {
        assert!(mapped.write_words(16, &[u32::try_from(round).expect("fits"); WORDS]));
        assert!(mapped.store_release(0, round));
        while mapped.load_acquire(8).expect("in bounds") < round {
            if Instant::now() > deadline {
                break;
            }
            std::hint::spin_loop();
        }
    }
    let bad = reader.join().expect("the reader thread");
    assert_ne!(
        bad,
        u64::MAX,
        "a sequence never arrived within the deadline"
    );
    assert_eq!(bad, 0, "words of an older round were read");
}

/// What the device phase of the mapping test saw, as plain data.
#[derive(Debug)]
struct Aliased {
    caps: MailboxCaps,
    empty_refused: bool,
    zeroed: bool,
    written: bool,
    seen: Vec<u32>,
    freed: bool,
}

/// The device phase: allocate, check the zeroing, write words, copy the device view back, free everything. Only `?`
/// inside up to the copy; the frees run whatever the copy returned, and nothing asserts while a handle is alive.
fn alias_phase(pattern: &[u32]) -> Result<Aliased, DeviceFailure> {
    let empty_refused = mapped_alloc(0).is_err();
    let caps = mailbox_caps()?;
    let mapped = mapped_alloc(512)?;
    let mut zeros = [0xFFFF_FFFFu32; 128];
    let zeroed = mapped.read_words(0, &mut zeros) && zeros == [0u32; 128];
    let written = mapped.write_words(0, pattern);
    let pinned = pinned_alloc(512);
    let stream = stream_create();
    let copied = match (&pinned, &stream, RawRange::new(mapped.device(), 0, 512)) {
        (Ok(pinned), Ok(stream), Some(range)) => {
            let mut bytes = [0u8; 512];
            copy_d2h_async(pinned, range, stream)
                .and_then(|()| stream_synchronize(stream))
                .and_then(|()| pinned_read_into(pinned, 0, &mut bytes))
                .map(|()| bytes)
        }
        _ => Err(invalid_argument()),
    };
    let freed = [
        stream.map_or(true, |s| stream_destroy(s).is_ok()),
        pinned.map_or(true, |p| pinned_free(p).is_ok()),
        mapped_free(mapped).is_ok(),
    ];
    let seen = copied?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_ne_bytes(*b))
        .collect();
    Ok(Aliased {
        caps,
        empty_refused,
        zeroed,
        written,
        seen,
        freed: freed.iter().all(|ok| *ok),
    })
}

/// Device test: the allocation is zeroed, the pointer the device uses aliases the host memory (a device-to-host
/// copy of the device view returns what the host wrote through the atomic accessors), and the free works.
#[test]
fn the_mapping_is_zeroed_aliased_by_the_device_view_and_freed() {
    if !device_present("the_mapping_is_zeroed_aliased_by_the_device_view_and_freed") {
        return;
    }
    let pattern: Vec<u32> = (0..128u32)
        .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(1))
        .collect();
    let seen = alias_phase(&pattern).expect("the device phase");
    eprintln!("mailbox caps: {:?}", seen.caps);
    assert!(seen.empty_refused, "an empty mapping is refused");
    assert!(seen.caps.can_map_host, "this device cannot map host memory");
    assert!(seen.zeroed, "the mapping is zeroed");
    assert!(seen.written, "the pattern was written");
    assert_eq!(seen.seen, pattern, "the device read what the host wrote");
    assert!(seen.freed, "the handles were freed");
}
