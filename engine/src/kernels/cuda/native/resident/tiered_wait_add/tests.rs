//! Host-only tests of the raw wait-add entry (TCAP C3): the C entry refuses every bad argument
//! before any CUDA call, and `check_wait_add` refuses the same terms on the Rust side. No device is
//! needed.

use super::super::super::{RawDevice, STATUS_INVALID_ARGUMENT};
use super::*;
use std::ptr::{self, NonNull};

/// The addresses of the five ranges of one call, in the order of [`regions`], then the stream.
#[derive(Clone, Copy)]
struct Numbers {
    at: [usize; 5],
    hidden: c_int,
    timeout_ns: u64,
    stream: usize,
}

const CTRL: usize = 0;
const WORD: usize = 1;
const MBOX_ANS: usize = 2;
const MBOX_PARTIAL: usize = 3;
const RESID: usize = 4;

/// A Qwen3.6-sized call (2048 hidden, a 2-second wait budget) on made-up addresses 4 GiB apart;
/// each case changes one thing so that the entry refuses before it reaches the device, and a valid
/// call is never made.
const BASE: Numbers = Numbers {
    at: [1 << 32, 2 << 32, 3 << 32, 4 << 32, 5 << 32],
    hidden: 2048,
    timeout_ns: 2_000_000_000,
    stream: 0x1,
};

type Case = (&'static str, fn(&mut Numbers));
fn run_entry(numbers: Numbers, null_args: bool) -> (c_int, c_int) {
    let p = |i: usize| ptr::without_provenance_mut::<c_void>(numbers.at[i]);
    let args = WaitAddArgsRaw {
        ctrl: p(CTRL).cast_const(),
        word: p(WORD),
        mbox_ans: p(MBOX_ANS).cast_const(),
        mbox_partial: p(MBOX_PARTIAL).cast_const(),
        resid: p(RESID),
        hidden: numbers.hidden,
        timeout_ns: numbers.timeout_ns,
    };
    let args_ptr = if null_args {
        ptr::null()
    } else {
        &raw const args
    };
    let mut stage = STAGE_NONE;
    // SAFETY: only called with arguments the entry refuses before it uses any pointer or makes a
    // CUDA call (see `BASE`); `args` and `stage` are live for the call.
    let status = unsafe {
        tiered_wait_add_launch_raw(
            args_ptr,
            ptr::without_provenance_mut(numbers.stream),
            &mut stage,
        )
    };
    (status, stage)
}

/// Host only: every case is refused by the entry's own checks before any CUDA call. It covers the C
/// half of the defence in depth: a NULL pointer or stream, every limit, every alignment and every
/// overlap of a written range, all of which the Rust wrapper already refuses.
#[test]
fn the_native_wait_add_entry_refuses_bad_arguments_before_any_device_call() {
    let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
    let cases: &[Case] = &[
        ("NULL stream", |n| n.stream = 0),
        ("NULL ctrl", |n| n.at[CTRL] = 0),
        ("NULL word", |n| n.at[WORD] = 0),
        ("NULL mbox_ans", |n| n.at[MBOX_ANS] = 0),
        ("NULL mbox_partial", |n| n.at[MBOX_PARTIAL] = 0),
        ("NULL resid", |n| n.at[RESID] = 0),
        ("hidden 0", |n| n.hidden = 0),
        ("hidden 3", |n| n.hidden = 3),
        ("hidden 65540", |n| n.hidden = 65540),
        ("timeout 0", |n| n.timeout_ns = 0),
        ("timeout past the limit", |n| {
            n.timeout_ns = MAX_TIMEOUT_NS + 1
        }),
        ("ctrl + 4", |n| n.at[CTRL] += 4),
        ("word + 2", |n| n.at[WORD] += 2),
        ("mbox_ans + 4", |n| n.at[MBOX_ANS] += 4),
        ("mbox_partial + 8", |n| n.at[MBOX_PARTIAL] += 8),
        ("resid + 8", |n| n.at[RESID] += 8),
        ("resid over word", |n| n.at[RESID] = n.at[WORD]),
        ("resid over ctrl", |n| n.at[RESID] = n.at[CTRL] + 16),
        ("resid over mbox_ans", |n| n.at[RESID] = n.at[MBOX_ANS] + 16),
        ("resid over mbox_partial", |n| {
            n.at[RESID] = n.at[MBOX_PARTIAL] + 16
        }),
        ("word over ctrl", |n| n.at[WORD] = n.at[CTRL] + 8),
        ("word over mbox_ans", |n| n.at[WORD] = n.at[MBOX_ANS] + 8),
        ("word over resid", |n| n.at[WORD] = n.at[RESID] + 16),
    ];
    assert_eq!(run_entry(BASE, true), refused, "wait-add: NULL args");
    for (name, edit) in cases {
        let mut numbers = BASE;
        edit(&mut numbers);
        assert_eq!(run_entry(numbers, false), refused, "wait-add: {name}");
    }
}

/// Five unbacked allocations, each far larger than any range below, so only the term under test can
/// refuse a call.
fn devices() -> [RawDevice; 5] {
    [(); 5].map(|()| RawDevice::unbacked(1 << 16))
}

fn range(device: &RawDevice, offset: usize, bytes: usize) -> RawRange<'_> {
    RawRange::new(device, offset, bytes).expect("a range inside the fixture")
}

/// The Qwen3.6-sized call with every range exact and aligned, each in its own allocation.
fn raw_base(d: &[RawDevice; 5]) -> WaitAddRaw<'_> {
    WaitAddRaw {
        ctrl: range(&d[CTRL], 0, 64),
        word: range(&d[WORD], 0, 4),
        mbox_ans: range(&d[MBOX_ANS], 0, 128),
        mbox_partial: range(&d[MBOX_PARTIAL], 0, 8192),
        resid: range(&d[RESID], 0, 8192),
        hidden: 2048,
        timeout_ns: 2_000_000_000,
    }
}

/// `check_wait_add` itself, on the host: the Rust half of the defence, which the C entry test above
/// does not reach.
#[test]
fn check_wait_add_refuses_limits_misalignment_and_overlap() {
    let d = devices();
    assert!(check_wait_add(&raw_base(&d)).is_ok(), "the base call");
    let same_input = WaitAddRaw {
        mbox_ans: range(&d[CTRL], 0, 128),
        ..raw_base(&d)
    };
    assert!(check_wait_add(&same_input).is_ok(), "two reads may alias");
    let cases: Vec<(&str, WaitAddRaw<'_>)> = vec![
        (
            "hidden 0",
            WaitAddRaw {
                hidden: 0,
                ..raw_base(&d)
            },
        ),
        (
            "hidden 3",
            WaitAddRaw {
                hidden: 3,
                ..raw_base(&d)
            },
        ),
        (
            "hidden 65540",
            WaitAddRaw {
                hidden: 65540,
                ..raw_base(&d)
            },
        ),
        (
            "timeout 0",
            WaitAddRaw {
                timeout_ns: 0,
                ..raw_base(&d)
            },
        ),
        (
            "timeout past the limit",
            WaitAddRaw {
                timeout_ns: MAX_TIMEOUT_NS + 1,
                ..raw_base(&d)
            },
        ),
        (
            "short ctrl",
            WaitAddRaw {
                ctrl: range(&d[CTRL], 0, 56),
                ..raw_base(&d)
            },
        ),
        (
            "short word",
            WaitAddRaw {
                word: range(&d[WORD], 0, 2),
                ..raw_base(&d)
            },
        ),
        (
            "short mbox_ans",
            WaitAddRaw {
                mbox_ans: range(&d[MBOX_ANS], 0, 120),
                ..raw_base(&d)
            },
        ),
        (
            "short mbox_partial",
            WaitAddRaw {
                mbox_partial: range(&d[MBOX_PARTIAL], 0, 8188),
                ..raw_base(&d)
            },
        ),
        (
            "short resid",
            WaitAddRaw {
                resid: range(&d[RESID], 0, 8188),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned ctrl",
            WaitAddRaw {
                ctrl: range(&d[CTRL], 4, 64),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned word",
            WaitAddRaw {
                word: range(&d[WORD], 2, 4),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned mbox_ans",
            WaitAddRaw {
                mbox_ans: range(&d[MBOX_ANS], 4, 128),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned mbox_partial",
            WaitAddRaw {
                mbox_partial: range(&d[MBOX_PARTIAL], 8, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned resid",
            WaitAddRaw {
                resid: range(&d[RESID], 8, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "resid over word",
            WaitAddRaw {
                resid: range(&d[WORD], 0, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "resid over ctrl",
            WaitAddRaw {
                resid: range(&d[CTRL], 16, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "word over ctrl",
            WaitAddRaw {
                word: range(&d[CTRL], 8, 4),
                ..raw_base(&d)
            },
        ),
        (
            "word over resid",
            WaitAddRaw {
                word: range(&d[RESID], 16, 4),
                ..raw_base(&d)
            },
        ),
        (
            "resid over mbox_partial",
            WaitAddRaw {
                resid: range(&d[MBOX_PARTIAL], 16, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "two writes over each other",
            WaitAddRaw {
                word: range(&d[RESID], 0, 4),
                ..raw_base(&d)
            },
        ),
    ];
    for (name, call) in cases {
        assert!(check_wait_add(&call).is_err(), "refuses {name}");
    }
    // The safe wrapper refuses before any native call, so a dangling stream is never used.
    let refused = WaitAddRaw {
        hidden: 0,
        ..raw_base(&d)
    };
    let stream = RawStream(NonNull::dangling());
    assert_eq!(
        tiered_wait_add_launch(&stream, &refused),
        Err(invalid_argument())
    );
}
