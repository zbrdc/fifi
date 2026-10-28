//! Host-only tests of the raw classification entry (TCAP C2): the C entry refuses every bad argument before any
//! CUDA call, and `check_classify` refuses the same terms on the Rust side. No device is needed.

use super::super::super::{RawDevice, STATUS_INVALID_ARGUMENT};
use super::*;
use std::ptr;

/// The addresses of the eleven ranges of one call, in the order of [`regions`], then the stream.
#[derive(Clone, Copy)]
struct Numbers {
    at: [usize; 11],
    n_expert: c_int,
    slots: c_int,
    top_k: c_int,
    hidden: c_int,
    layer: c_int,
    stream: usize,
}

const IDS: usize = 0;
const WEIGHTS: usize = 1;
const SLOT_ROW: usize = 2;
const NORM_ROW: usize = 3;
const SLOT_IDS_OUT: usize = 4;
const WEIGHTS_OUT: usize = 5;
const LOG_OUT: usize = 6;
const CTRL: usize = 7;
const WORD: usize = 8;
const MBOX_REQ: usize = 9;
const MBOX_ROW: usize = 10;

/// A Qwen3.6-sized call (256 experts, 31 slots, 8 picks, 2048 hidden) on made-up addresses 4 GiB
/// apart; each case changes one thing so that the entry refuses before it reaches the device, and a
/// valid call is never made.
const BASE: Numbers = Numbers {
    at: [
        1 << 32,
        2 << 32,
        3 << 32,
        4 << 32,
        5 << 32,
        6 << 32,
        7 << 32,
        8 << 32,
        9 << 32,
        10 << 32,
        11 << 32,
    ],
    n_expert: 256,
    slots: 31,
    top_k: 8,
    hidden: 2048,
    layer: 3,
    stream: 0x1,
};

type Case = (&'static str, fn(&mut Numbers));

fn run_entry(numbers: Numbers, null_args: bool) -> (c_int, c_int) {
    let p = |i: usize| ptr::without_provenance_mut::<c_void>(numbers.at[i]);
    let args = ClassifyArgsRaw {
        ids: p(IDS).cast_const(),
        weights: p(WEIGHTS).cast_const(),
        slot_row: p(SLOT_ROW).cast_const(),
        norm_row: p(NORM_ROW).cast_const(),
        slot_ids_out: p(SLOT_IDS_OUT),
        weights_out: p(WEIGHTS_OUT),
        log_out: p(LOG_OUT),
        ctrl: p(CTRL),
        word: p(WORD),
        mbox_req: p(MBOX_REQ),
        mbox_row: p(MBOX_ROW),
        n_expert: numbers.n_expert,
        slots: numbers.slots,
        top_k: numbers.top_k,
        hidden: numbers.hidden,
        layer: numbers.layer,
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
        tiered_classify_launch_raw(
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
fn the_native_classify_entry_refuses_bad_arguments_before_any_device_call() {
    let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
    let cases: &[Case] = &[
        ("NULL stream", |n| n.stream = 0),
        ("NULL ids", |n| n.at[IDS] = 0),
        ("NULL weights", |n| n.at[WEIGHTS] = 0),
        ("NULL slot_row", |n| n.at[SLOT_ROW] = 0),
        ("NULL norm_row", |n| n.at[NORM_ROW] = 0),
        ("NULL slot_ids_out", |n| n.at[SLOT_IDS_OUT] = 0),
        ("NULL weights_out", |n| n.at[WEIGHTS_OUT] = 0),
        ("NULL log_out", |n| n.at[LOG_OUT] = 0),
        ("NULL ctrl", |n| n.at[CTRL] = 0),
        ("NULL word", |n| n.at[WORD] = 0),
        ("NULL mbox_req", |n| n.at[MBOX_REQ] = 0),
        ("NULL mbox_row", |n| n.at[MBOX_ROW] = 0),
        ("n_expert 0", |n| n.n_expert = 0),
        ("n_expert 4097", |n| n.n_expert = 4097),
        ("slots 0", |n| n.slots = 0),
        ("slots past n_expert", |n| n.slots = 257),
        ("top_k 0", |n| n.top_k = 0),
        ("top_k 9", |n| n.top_k = 9),
        ("hidden 0", |n| n.hidden = 0),
        ("hidden 6", |n| n.hidden = 6),
        ("hidden 65540", |n| n.hidden = 65540),
        ("layer -1", |n| n.layer = -1),
        ("layer 1024", |n| n.layer = 1024),
        ("ids + 2", |n| n.at[IDS] += 2),
        ("weights + 2", |n| n.at[WEIGHTS] += 2),
        ("slot_row + 2", |n| n.at[SLOT_ROW] += 2),
        ("norm_row + 8", |n| n.at[NORM_ROW] += 8),
        ("slot_ids_out + 2", |n| n.at[SLOT_IDS_OUT] += 2),
        ("weights_out + 2", |n| n.at[WEIGHTS_OUT] += 2),
        ("log_out + 4", |n| n.at[LOG_OUT] += 4),
        ("ctrl + 4", |n| n.at[CTRL] += 4),
        ("word + 2", |n| n.at[WORD] += 2),
        ("mbox_req + 4", |n| n.at[MBOX_REQ] += 4),
        ("mbox_row + 8", |n| n.at[MBOX_ROW] += 8),
        ("slot_ids_out over ids", |n| n.at[SLOT_IDS_OUT] = n.at[IDS]),
        ("weights_out over weights", |n| {
            n.at[WEIGHTS_OUT] = n.at[WEIGHTS] + 16
        }),
        ("log_out over slot_row", |n| {
            n.at[LOG_OUT] = n.at[SLOT_ROW] + 8
        }),
        ("ctrl over norm_row", |n| n.at[CTRL] = n.at[NORM_ROW] + 16),
        ("word over ids", |n| n.at[WORD] = n.at[IDS] + 16),
        ("mbox_req over log_out", |n| n.at[MBOX_REQ] = n.at[LOG_OUT]),
        ("mbox_row over norm_row", |n| {
            n.at[MBOX_ROW] = n.at[NORM_ROW] + 16
        }),
        ("slot_ids_out over weights_out", |n| {
            n.at[SLOT_IDS_OUT] = n.at[WEIGHTS_OUT]
        }),
        ("ctrl over word", |n| n.at[CTRL] = n.at[WORD] - 8),
        ("mbox_req over mbox_row", |n| {
            n.at[MBOX_REQ] = n.at[MBOX_ROW] + 16
        }),
        ("mbox_row over ctrl", |n| n.at[MBOX_ROW] = n.at[CTRL] - 16),
        ("log_out over weights_out", |n| {
            n.at[LOG_OUT] = n.at[WEIGHTS_OUT]
        }),
        ("word over mbox_req", |n| n.at[WORD] = n.at[MBOX_REQ] + 8),
        ("weights_out over mbox_row", |n| {
            n.at[WEIGHTS_OUT] = n.at[MBOX_ROW] + 16
        }),
        ("slot_ids_out over norm_row", |n| {
            n.at[SLOT_IDS_OUT] = n.at[NORM_ROW] + 16
        }),
        ("ctrl over slot_row", |n| n.at[CTRL] = n.at[SLOT_ROW] + 8),
        ("word over weights", |n| n.at[WORD] = n.at[WEIGHTS] + 4),
        ("mbox_row over slot_row", |n| {
            n.at[MBOX_ROW] = n.at[SLOT_ROW] + 16
        }),
    ];
    assert_eq!(run_entry(BASE, true), refused, "classify: NULL args");
    for (name, edit) in cases {
        let mut numbers = BASE;
        edit(&mut numbers);
        assert_eq!(run_entry(numbers, false), refused, "classify: {name}");
    }
}

/// Eleven unbacked allocations, each far larger than any range below, so only the term under test can
/// refuse a call.
fn devices() -> [RawDevice; 11] {
    [(); 11].map(|()| RawDevice::unbacked(1 << 16))
}

fn range(device: &RawDevice, offset: usize, bytes: usize) -> RawRange<'_> {
    RawRange::new(device, offset, bytes).expect("a range inside the fixture")
}

/// The Qwen3.6-sized call with every range exact and aligned, each in its own allocation.
fn raw_base(d: &[RawDevice; 11]) -> ClassifyRaw<'_> {
    ClassifyRaw {
        ids: range(&d[IDS], 0, 32),
        weights: range(&d[WEIGHTS], 0, 32),
        slot_row: range(&d[SLOT_ROW], 0, 1024),
        norm_row: range(&d[NORM_ROW], 0, 8192),
        slot_ids_out: range(&d[SLOT_IDS_OUT], 0, 32),
        weights_out: range(&d[WEIGHTS_OUT], 0, 32),
        log_out: range(&d[LOG_OUT], 0, 64),
        ctrl: range(&d[CTRL], 0, 64),
        word: range(&d[WORD], 0, 4),
        mbox_req: range(&d[MBOX_REQ], 0, 128),
        mbox_row: range(&d[MBOX_ROW], 0, 8192),
        n_expert: 256,
        slots: 31,
        top_k: 8,
        hidden: 2048,
        layer: 3,
    }
}

/// `check_classify` itself, on the host: the Rust half of the defence, which the C entry test
/// above does not reach.
#[test]
fn check_classify_refuses_limits_short_ranges_misalignment_and_overlap() {
    let d = devices();
    assert!(check_classify(&raw_base(&d)).is_ok(), "the base call");
    let same_input = ClassifyRaw {
        weights: range(&d[IDS], 0, 32),
        ..raw_base(&d)
    };
    assert!(check_classify(&same_input).is_ok(), "two inputs may alias");
    let cases: Vec<(&str, ClassifyRaw<'_>)> = vec![
        (
            "n_expert 0",
            ClassifyRaw {
                n_expert: 0,
                ..raw_base(&d)
            },
        ),
        (
            "slots past n_expert",
            ClassifyRaw {
                slots: 257,
                ..raw_base(&d)
            },
        ),
        (
            "top_k 9",
            ClassifyRaw {
                top_k: 9,
                ..raw_base(&d)
            },
        ),
        (
            "hidden 6",
            ClassifyRaw {
                hidden: 6,
                ..raw_base(&d)
            },
        ),
        (
            "layer 1024",
            ClassifyRaw {
                layer: 1024,
                ..raw_base(&d)
            },
        ),
        (
            "negative layer",
            ClassifyRaw {
                layer: -1,
                ..raw_base(&d)
            },
        ),
        (
            "short slot_row",
            ClassifyRaw {
                slot_row: range(&d[SLOT_ROW], 0, 1020),
                ..raw_base(&d)
            },
        ),
        (
            "short mbox_row",
            ClassifyRaw {
                mbox_row: range(&d[MBOX_ROW], 0, 8188),
                ..raw_base(&d)
            },
        ),
        (
            "short log_out",
            ClassifyRaw {
                log_out: range(&d[LOG_OUT], 0, 56),
                ..raw_base(&d)
            },
        ),
        (
            "short mbox_req",
            ClassifyRaw {
                mbox_req: range(&d[MBOX_REQ], 0, 120),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned norm_row",
            ClassifyRaw {
                norm_row: range(&d[NORM_ROW], 8, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned log_out",
            ClassifyRaw {
                log_out: range(&d[LOG_OUT], 4, 64),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned ctrl",
            ClassifyRaw {
                ctrl: range(&d[CTRL], 4, 64),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned word",
            ClassifyRaw {
                word: range(&d[WORD], 2, 4),
                ..raw_base(&d)
            },
        ),
        (
            "slot_ids_out over ids",
            ClassifyRaw {
                slot_ids_out: range(&d[IDS], 16, 32),
                ..raw_base(&d)
            },
        ),
        (
            "ctrl over norm_row",
            ClassifyRaw {
                ctrl: range(&d[NORM_ROW], 16, 64),
                ..raw_base(&d)
            },
        ),
        (
            "mbox_row over slot_row",
            ClassifyRaw {
                mbox_row: range(&d[SLOT_ROW], 16, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "two outputs over each other",
            ClassifyRaw {
                word: range(&d[CTRL], 8, 4),
                ..raw_base(&d)
            },
        ),
        (
            "mbox_req over log_out",
            ClassifyRaw {
                mbox_req: range(&d[LOG_OUT], 0, 128),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned ids",
            ClassifyRaw {
                ids: range(&d[IDS], 2, 32),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned mbox_req",
            ClassifyRaw {
                mbox_req: range(&d[MBOX_REQ], 4, 128),
                ..raw_base(&d)
            },
        ),
        (
            "misaligned mbox_row",
            ClassifyRaw {
                mbox_row: range(&d[MBOX_ROW], 8, 8192),
                ..raw_base(&d)
            },
        ),
        (
            "short ids",
            ClassifyRaw {
                ids: range(&d[IDS], 0, 28),
                ..raw_base(&d)
            },
        ),
        (
            "short ctrl",
            ClassifyRaw {
                ctrl: range(&d[CTRL], 0, 56),
                ..raw_base(&d)
            },
        ),
        (
            "short word",
            ClassifyRaw {
                word: range(&d[WORD], 0, 2),
                ..raw_base(&d)
            },
        ),
        (
            "mbox_req over an input",
            ClassifyRaw {
                mbox_req: range(&d[IDS], 16, 128),
                ..raw_base(&d)
            },
        ),
    ];
    for (name, call) in cases {
        assert!(check_classify(&call).is_err(), "refuses {name}");
    }
}
