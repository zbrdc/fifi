//! Host tests of the C side of the dense MMQ entry: the field-by-field pin of the vendored
//! `mmq_args` fill (`fill_dense_args`) and the refusal table of the entry's validation. Both go
//! through `inference_cuda_mmq_dense_fill`, which validates exactly like the launching entry
//! and then writes the fill, with no CUDA call: it compares addresses and never dereferences
//! them, so the fake pointers below are safe, and a case that is refused differs from the valid
//! base call in exactly one field, so it reaches its own guard.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};
use std::ptr;

use super::MmqDenseArgsRaw;
use crate::kernels::cuda::native::STATUS_INVALID_ARGUMENT;
use crate::kernels::cuda::prefill::mmq_size::{MmqConfig, MmqType, fixup_floats_exact, j_best};

/// Mirror of `inference_cuda_mmq_args_view`, field for field.
#[repr(C)]
pub(super) struct ArgsViewRaw {
    x: *const c_void,
    y: *const c_void,
    ids_dst: *const c_void,
    expert_bounds: *const c_void,
    dst: *mut c_void,
    y_scale: *const c_void,
    fixup: *mut c_void,
    fixup_bytes: usize,
    // ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst, nchannels_x,
    // nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x,
    // nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, ncols_max, ncols_opt
    ints: [i64; 18],
    type_x: c_int,
    reserved: c_int,
}

const _: () = assert!(
    size_of::<ArgsViewRaw>() == 216
        && offset_of!(ArgsViewRaw, fixup_bytes) == 56
        && offset_of!(ArgsViewRaw, ints) == 64
        && offset_of!(ArgsViewRaw, type_x) == 208
);

/// Mirror of `inference_cuda_mmq_facts` (mmq_facts.h), field for field.
#[repr(C)]
struct FactsRaw {
    device: c_int,
    cc: c_int,
    nsm: c_int,
    reserved: c_int,
    smpbo: usize,
}

const _: () = assert!(size_of::<FactsRaw>() == 24 && offset_of!(FactsRaw, smpbo) == 16);

unsafe extern "C" {
    fn inference_cuda_mmq_dense_fill(args: *const MmqDenseArgsRaw, out: *mut ArgsViewRaw) -> c_int;
    fn inference_cuda_mmq_dense_fixup_need(
        args: *const MmqDenseArgsRaw,
        facts: *const FactsRaw,
    ) -> usize;
}

/// A fake device address; the entry compares addresses and never dereferences them.
fn at(address: usize) -> *mut c_void {
    ptr::without_provenance_mut(address)
}

/// A valid call: Q8_0, K = 2,048, 512 rows, 64 columns, tight stride, ranges 256 MiB apart.
fn base() -> MmqDenseArgsRaw {
    MmqDenseArgsRaw {
        w: at(0x1000_0000),
        y: at(0x2000_0000),
        dst: at(0x3000_0000),
        fixup: at(0x4000_0000),
        fixup_bytes: 1 << 20,
        ty: 8,
        k: 2048,
        n: 512,
        m: 64,
        ld: 512,
        col_off: 0,
    }
}

/// The fill of `args`, or the entry's status.
pub(super) fn fill(args: &MmqDenseArgsRaw) -> Result<ArgsViewRaw, c_int> {
    let mut view = ArgsViewRaw {
        x: ptr::null(),
        y: ptr::null(),
        ids_dst: ptr::null(),
        expert_bounds: ptr::null(),
        dst: ptr::null_mut(),
        y_scale: ptr::null(),
        fixup: ptr::null_mut(),
        fixup_bytes: 0,
        ints: [0; 18],
        type_x: 0,
        reserved: 0,
    };
    // SAFETY: both pointers address live structs of the C layout for the whole call (the
    // layout is asserted above and by the header); the entry reads the addresses inside
    // `args` as numbers only and writes `view` only on success.
    let status = unsafe { inference_cuda_mmq_dense_fill(args, &mut view) };
    if status == 0 { Ok(view) } else { Err(status) }
}

#[test]
fn the_dense_fill_pins_every_mmq_args_field() {
    // Field order of `ints`: ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst,
    // nchannels_x, nchannels_y, stride_channel_x/y/dst, nsamples_x, nsamples_y,
    // stride_sample_x/y/dst, ncols_max, ncols_opt.
    let a = MmqDenseArgsRaw {
        n: 8192,
        m: 49,
        ld: 8192,
        ..base()
    };
    let v = fill(&a).expect("the Q8_0 qkv shape");
    assert_eq!(
        (
            v.x as usize,
            v.y as usize,
            v.dst as usize,
            v.fixup as usize,
            v.fixup_bytes
        ),
        (0x1000_0000, 0x2000_0000, 0x3000_0000, 0x4000_0000, 1 << 20)
    );
    assert!(v.ids_dst.is_null() && v.expert_bounds.is_null() && v.y_scale.is_null());
    assert_eq!(
        v.ints,
        [
            2048, 8192, 49, 64, 49, 8192, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 49, 49
        ]
    );
    assert_eq!((v.type_x, v.reserved), (8, 0));

    // A Q4_K slice of a fused destination: ld wider than n, a column offset (floats), the
    // stride in 256-value blocks.
    let a = MmqDenseArgsRaw {
        ty: 12,
        k: 4096,
        n: 2048,
        m: 1000,
        ld: 6144,
        col_off: 2048,
        ..base()
    };
    let v = fill(&a).expect("the fused Q4_K slice");
    assert_eq!(v.dst as usize, 0x3000_0000 + 2048 * 4);
    assert!(v.ids_dst.is_null() && v.expert_bounds.is_null() && v.y_scale.is_null());
    assert_eq!(
        v.ints,
        [
            4096, 2048, 1000, 16, 1000, 6144, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1000, 1000
        ]
    );
    assert_eq!(v.type_x, 12);

    let a = MmqDenseArgsRaw {
        ty: 14,
        k: 512,
        n: 32,
        m: 24,
        ld: 32,
        ..base()
    };
    let v = fill(&a).expect("the Q6_K ssm shape");
    assert_eq!(
        v.ints,
        [512, 32, 24, 2, 24, 32, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 24, 24]
    );
    assert_eq!(v.type_x, 14);
}

#[test]
fn the_dense_entry_refuses_each_bad_argument_at_its_own_guard() {
    fill(&base()).expect("the base call is valid");
    // The grid-limit pair: 2^20 row tiles of Q8_0 at K = 32,768 reach 2^30 exactly.
    // The weight is 4.7 TB there, so the other ranges sit beyond its end.
    let tiles = |row_tiles: usize| {
        let n = c_int::try_from(row_tiles * 128).expect("fits");
        MmqDenseArgsRaw {
            y: at(0x500_0000_0000),
            dst: at(0x600_0000_0000),
            fixup: at(0x700_0000_0000),
            k: 32_768,
            n,
            m: 8,
            ld: n,
            ..base()
        }
    };
    fill(&tiles((1 << 20) - 1)).expect("one row tile below the grid limit");
    // The destination reach pair: (m - 1) * ld + n is exactly INT_MAX, then one more. The
    // destination is 8.6 GB there, so the other ranges sit far below and above it.
    let far = |n: usize, ld: usize| MmqDenseArgsRaw {
        y: at(0x500_0000_0000),
        dst: at(0x600_0000_0000),
        fixup: at(0x700_0000_0000),
        k: 256,
        n: c_int::try_from(n).expect("fits"),
        m: 2048,
        ld: c_int::try_from(ld).expect("fits"),
        ..base()
    };
    fill(&far((1 << 20) - 1, 1 << 20)).expect("a reach of exactly INT_MAX");
    let w = 0x1000_0000usize;
    let cases: Vec<(&str, MmqDenseArgsRaw)> = vec![
        (
            "w null",
            MmqDenseArgsRaw {
                w: ptr::null(),
                ..base()
            },
        ),
        (
            "y null",
            MmqDenseArgsRaw {
                y: ptr::null(),
                ..base()
            },
        ),
        (
            "dst null",
            MmqDenseArgsRaw {
                dst: ptr::null_mut(),
                ..base()
            },
        ),
        ("Q5_0 has no case", MmqDenseArgsRaw { ty: 6, ..base() }),
        ("F32 has no case", MmqDenseArgsRaw { ty: 0, ..base() }),
        ("K off the k grid", MmqDenseArgsRaw { k: 384, ..base() }),
        ("K zero", MmqDenseArgsRaw { k: 0, ..base() }),
        ("K negative", MmqDenseArgsRaw { k: -256, ..base() }),
        (
            "K above the extent",
            MmqDenseArgsRaw {
                k: 33_024,
                ..base()
            },
        ),
        ("M zero", MmqDenseArgsRaw { m: 0, ..base() }),
        (
            "M above the chunk cap",
            MmqDenseArgsRaw { m: 2049, ..base() },
        ),
        ("N zero", MmqDenseArgsRaw { n: 0, ..base() }),
        ("ld below n", MmqDenseArgsRaw { ld: 511, ..base() }),
        (
            "col_off negative",
            MmqDenseArgsRaw {
                col_off: -1,
                ..base()
            },
        ),
        (
            "ld below col_off + n",
            MmqDenseArgsRaw {
                col_off: 1,
                ..base()
            },
        ),
        ("tile grid at 2^30", tiles(1 << 20)),
        ("dst reach one past INT_MAX", far(1 << 20, 1 << 20)),
        ("dst reach past INT_MAX, n 128", far(128, 1_118_482)),
        (
            "w misaligned",
            MmqDenseArgsRaw {
                w: at(w + 8),
                ..base()
            },
        ),
        (
            "y misaligned",
            MmqDenseArgsRaw {
                y: at(0x2000_0008),
                ..base()
            },
        ),
        (
            "dst misaligned",
            MmqDenseArgsRaw {
                dst: at(0x3000_0002),
                ..base()
            },
        ),
        (
            "fixup misaligned",
            MmqDenseArgsRaw {
                fixup: at(0x4000_0008),
                ..base()
            },
        ),
        (
            "fixup null with bytes",
            MmqDenseArgsRaw {
                fixup: ptr::null_mut(),
                ..base()
            },
        ),
        (
            "fixup without bytes",
            MmqDenseArgsRaw {
                fixup_bytes: 0,
                ..base()
            },
        ),
        (
            "dst overlaps w",
            MmqDenseArgsRaw {
                dst: at(w + 16),
                ..base()
            },
        ),
        (
            "dst overlaps y",
            MmqDenseArgsRaw {
                dst: at(0x2000_0010),
                ..base()
            },
        ),
        (
            "fixup overlaps dst",
            MmqDenseArgsRaw {
                fixup: at(0x3000_0010),
                ..base()
            },
        ),
        (
            "fixup overlaps w",
            MmqDenseArgsRaw {
                fixup: at(w + 16),
                ..base()
            },
        ),
        (
            "fixup overlaps y",
            MmqDenseArgsRaw {
                fixup: at(0x2000_0010),
                ..base()
            },
        ),
    ];
    for (name, args) in cases {
        assert_eq!(fill(&args).err(), Some(STATUS_INVALID_ARGUMENT), "{name}");
    }
    // A NULL out pointer is refused before anything is written.
    // SAFETY: the entry checks `out` before it writes and reads `args` as a live struct.
    let status = unsafe { inference_cuda_mmq_dense_fill(&base(), ptr::null_mut()) };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

/// The C replica's fixup need for a dense call of `n` rows and `m` columns of `ggml_type`
/// `ty`, on the device `facts`.
fn c_fixup_need(ty: c_int, n: usize, m: usize, facts: &FactsRaw) -> usize {
    let n = c_int::try_from(n).expect("n fits");
    let args = MmqDenseArgsRaw {
        ty,
        n,
        m: c_int::try_from(m).expect("m fits"),
        ld: n,
        ..base()
    };
    // SAFETY: both pointers address live structs of the C layout for the whole call; the
    // function is pure: it reads integers and calls no CUDA function.
    unsafe { inference_cuda_mmq_dense_fixup_need(&args, facts) }
}

/// The Rust side of the cross-pin: the tile width from the Rust replica `j_best`, then the
/// vendored launch's tile count, tiling-mode test and block count (mmq.cuh:1395-1408), then
/// `fixup_floats_exact`. Every kept config row has I = 128 and stream-k on.
fn rust_fixup_need(ty: MmqType, n: usize, m: usize, facts: &FactsRaw) -> usize {
    let nsm = usize::try_from(facts.nsm).expect("nsm");
    let Some(j) = j_best(ty, !n.is_multiple_of(128), facts.cc, facts.smpbo, m) else {
        return 0;
    };
    // A tiling config needs no fixup: patch V-5 makes the Q4_K rows at or below 64 tiles.
    if !MmqConfig::select(facts.cc).stream_k(ty, j) {
        return 0;
    }
    let tiles = m.div_ceil(j) * n.div_ceil(128);
    let waves = tiles.div_ceil(nsm);
    let blocks = if 100 * tiles / (nsm * waves) >= 90 {
        tiles
    } else {
        nsm
    };
    if tiles.is_multiple_of(blocks) {
        return 0;
    }
    fixup_floats_exact(blocks, j, 128).expect("exact fixup") * 4
}

/// The architectures this build compiled (the list build.rs wrote into its generated
/// `__CUDA_ARCH_LIST__` header, the one `ggml_cuda_highest_compiled_arch` reads): the C
/// replica selects its config table from it, so a device-facts set whose built architecture
/// is not in it is not a state this build can be in.
fn compiled_archs() -> Vec<i32> {
    // Embedded at build time, not read at run time: a test exe copied away from its target
    // directory (a mutant's, a sanitizer's) still has the list.
    let generated = include_str!(concat!(env!("OUT_DIR"), "/fifi_cuda_arch_list.h"));
    let list = generated
        .lines()
        .find_map(|line| line.strip_prefix("#define __CUDA_ARCH_LIST__ "))
        .expect("the __CUDA_ARCH_LIST__ define");
    list.split(',')
        .map(|arch| arch.trim().parse().expect("a decimal arch"))
        .collect()
}

#[test]
fn the_c_fixup_need_agrees_with_the_rust_sizing_everywhere() {
    // The cross-pin of the two J-loop replicas: the C function that sizes the fixup refusal
    // and the Rust replica must give the same bytes, including the "0 when the launch tiles"
    // and "no tile width fits" cases, for every M in 1..=2048, the Qwen dense row counts
    // (fallback off) and the partial-tile ones (fallback on), all three weight types, and
    // device facts of the sm_86, sm_89 and sm_72 classes plus a one-SM device and a device
    // whose shared memory fits no tile width.
    let sets: [(&str, i32, i32, usize); 6] = [
        ("sm_86, 82 SMs", 860, 82, 101_376),
        ("sm_89, 20 SMs", 860, 20, 101_376),
        ("sm_72, 8 SMs", 720, 8, 98_304),
        ("sm_86 class, 1 SM", 860, 1, 101_376),
        ("sm_86 class, shared memory skips J >= 120", 860, 82, 56_000),
        ("no tile width fits", 860, 82, 1_000),
    ];
    let types = [(8, MmqType::Q80), (12, MmqType::Q4K), (14, MmqType::Q6K)];
    let rows = [8192usize, 4096, 2048, 512, 4160, 160, 32, 8];
    let (mut zero, mut nonzero, mut compared) = (0usize, 0usize, 0usize);
    let built = compiled_archs();
    for (name, cc, nsm, smpbo) in sets {
        if !built.contains(&cc) {
            println!("fixup-need cross-pin: skipped {name}: {cc} is not in this build {built:?}");
            continue;
        }
        let facts = FactsRaw {
            device: 0,
            cc,
            nsm,
            reserved: 0,
            smpbo,
        };
        for (ggml, ty) in types {
            for n in rows {
                for m in 1..=2048usize {
                    let (c, r) = (
                        c_fixup_need(ggml, n, m, &facts),
                        rust_fixup_need(ty, n, m, &facts),
                    );
                    assert_eq!(c, r, "{name}, {ty:?}, n {n}, m {m}: C {c} B, Rust {r} B");
                    compared += 1;
                    if c == 0 {
                        zero += 1;
                    } else {
                        nonzero += 1;
                    }
                }
            }
        }
    }
    println!(
        "fixup-need cross-pin: {compared} cases equal, {nonzero} with a fixup, {zero} without"
    );
    // The sweep is not vacuous: both outcomes occur in volume (the sm_86 sets are in every
    // build).
    assert!(
        zero > 10_000 && nonzero > 10_000,
        "zero {zero}, nonzero {nonzero}"
    );
}

#[test]
fn the_c_fixup_need_is_zero_for_arguments_the_entry_refuses() {
    // A pure function must not divide by an SM count of zero or read a type it has no row
    // for.
    let good = FactsRaw {
        device: 0,
        cc: 860,
        nsm: 82,
        reserved: 0,
        smpbo: 101_376,
    };
    assert_ne!(
        c_fixup_need(8, 32, 24, &good),
        0,
        "the control shape needs a fixup"
    );
    let no_sms = FactsRaw { nsm: 0, ..good };
    assert_eq!(c_fixup_need(8, 32, 24, &no_sms), 0);
    assert_eq!(c_fixup_need(6, 32, 24, &good), 0, "Q5_0 has no dense case");
    assert_eq!(c_fixup_need(8, 32, 0, &good), 0, "no columns");
    assert_eq!(c_fixup_need(8, 32, 2049, &good), 0, "above the chunk cap");
}
