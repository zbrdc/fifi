//! Raw FFI to `inference_cuda_embed_launch` of `native/resident/embed.cu` (declared in
//! `native/resident/embed.h`): the embedding row gather of the resident step. Child of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that the tiles hold the whole table, that `ids`
//!   holds `m` ints, that `x` holds `m` rows of `k` floats, that `status` holds one int, that every
//!   range is aligned, and that `x` and `status`, which the kernel writes, share no byte with
//!   `ids`, the tiles or each other. The C entry repeats these checks on the addresses.
//! - The ids are not read by the host: the kernel reads them when it runs. An id outside
//!   `[0, n_rows)` makes it write NaN to its row of `x`, read no tile, and OR status bit 4 into
//!   `status`; it never reads or writes outside the ranges above.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The launch entry only queues work on the stream it is given and returns the launch status;
//!   a fault in the kernel surfaces at the next synchronize.
//! - Every call runs under the process dispatch guard and on the thread that owns the context.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::native::{TileVec, Tiled, TiledType};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// Mirror of `inference_cuda_embed_args` in `embed.h`.
#[repr(C)]
struct EmbedArgsRaw {
    tiles: *const c_void,
    ids: *const c_int,
    x: *mut f32,
    status: *mut c_int,
    ggml_type: c_int,
    m: c_int,
    n_rows: c_int,
    k: c_int,
}

const _: () = assert!(
    size_of::<EmbedArgsRaw>() == 48
        && offset_of!(EmbedArgsRaw, tiles) == 0
        && offset_of!(EmbedArgsRaw, ids) == 8
        && offset_of!(EmbedArgsRaw, x) == 16
        && offset_of!(EmbedArgsRaw, status) == 24
        && offset_of!(EmbedArgsRaw, ggml_type) == 32
        && offset_of!(EmbedArgsRaw, m) == 36
        && offset_of!(EmbedArgsRaw, n_rows) == 40
        && offset_of!(EmbedArgsRaw, k) == 44,
    "EmbedArgsRaw must equal inference_cuda_embed_args in embed.h"
);

unsafe extern "C" {
    #[link_name = "inference_cuda_embed_launch"]
    fn embed_launch_raw(
        args: *const EmbedArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The most tokens one gather takes (`kMaxTokens` in `embed.cu`).
const MAX_TOKENS: usize = 4096;
/// The most rows and values per row of the table (`kMaxRows`, `kMaxK`).
const MAX_ROWS: usize = 1 << 28;
const MAX_K: usize = 32768;

/// What one gather reads and writes: `tiles` holds the tiled table of `n_rows` x `k` values of
/// type `ty`, `ids` the `m` token ids (read when the kernel runs), `x` receives `m` rows of `k`
/// floats, and `status` is the step status word that a bad id ORs bit 4 into.
pub(in crate::kernels::cuda) struct EmbedRaw<'a> {
    pub(in crate::kernels::cuda) ty: TiledType,
    pub(in crate::kernels::cuda) tiles: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_rows: c_int,
    pub(in crate::kernels::cuda) k: c_int,
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// Bytes each range of a gather must hold.
struct Needs {
    tiles: usize,
    ids: usize,
    x: usize,
}

/// The sizes of a gather, or `None` for a type, shape or token count the kernel refuses.
fn needs(call: &EmbedRaw<'_>) -> Option<Needs> {
    let m = usize::try_from(call.m).ok()?;
    let rows = usize::try_from(call.n_rows).ok()?;
    let k = usize::try_from(call.k).ok()?;
    let table_ok = matches!(call.ty, TiledType::Q4K | TiledType::Q80)
        && (8..=MAX_ROWS).contains(&rows)
        && rows.is_multiple_of(8)
        && (256..=MAX_K).contains(&k)
        && k.is_multiple_of(256)
        && (1..=MAX_TOKENS).contains(&m);
    if !table_ok {
        return None;
    }
    let table = Tiled {
        ty: call.ty,
        n_rows: rows,
        k,
    };
    Some(Needs {
        tiles: table.tile_vecs()?.checked_mul(size_of::<TileVec>())?,
        ids: m.checked_mul(size_of::<c_int>())?,
        x: m.checked_mul(k)?.checked_mul(size_of::<f32>())?,
    })
}

/// Proves a gather before the native call: type, shape, `m`, sizes, alignment (tiles and `x` 16,
/// ids and status 4) and that `x` and `status`, which the kernel writes, overlap none of the other
/// ranges.
fn check_embed(call: &EmbedRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(call).ok_or_else(invalid_argument)?;
    let sized = fits(&call.tiles, need.tiles, 16)
        && fits(&call.ids, need.ids, 4)
        && fits(&call.x, need.x, 16)
        && fits(&call.status, size_of::<c_int>(), 4);
    let (tiles, ids) = (
        Span::of(&call.tiles, need.tiles),
        Span::of(&call.ids, need.ids),
    );
    let x = Span::of(&call.x, need.x);
    let status = Span::of(&call.status, size_of::<c_int>());
    let apart = x.apart(&ids)
        && x.apart(&tiles)
        && x.apart(&status)
        && status.apart(&ids)
        && status.apart(&tiles);
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the gather of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn embed_launch(
    stream: &RawStream,
    call: &EmbedRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_embed(call)?;
    let args = EmbedArgsRaw {
        tiles: call.tiles.address().cast_const(),
        ids: call.ids.address().cast::<c_int>().cast_const(),
        x: call.x.address().cast::<f32>(),
        status: call.status.address().cast::<c_int>(),
        ggml_type: call.ty.ggml(),
        m: call.m,
        n_rows: call.n_rows,
        k: call.k,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_embed` proved the type and shape, the tiles, ids, x and status ranges inside
    // their live allocations, large enough for the call, aligned, and `x` and `status` disjoint
    // from every other range; the adapter repeats the checks on the addresses. The allocations stay
    // valid until the queued launch finishes because `inference_cuda_free` synchronizes the device
    // first. The kernel reads the ids when it runs and the adapter keeps no pointer. `args` and
    // `stage` are live for the call, and `stream` is live for the call.
    let status = unsafe { embed_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Mirror of `inference_cuda_embed_native_args` in `embed.h` (the native Q4_K gather of the same
/// contract, against the file's own blocks).
#[repr(C)]
struct EmbedNativeArgsRaw {
    blocks: *const c_void,
    ids: *const c_int,
    x: *mut f32,
    status: *mut c_int,
    m: c_int,
    n_rows: c_int,
    k: c_int,
}

const _: () = assert!(
    size_of::<EmbedNativeArgsRaw>() == 48
        && offset_of!(EmbedNativeArgsRaw, blocks) == 0
        && offset_of!(EmbedNativeArgsRaw, ids) == 8
        && offset_of!(EmbedNativeArgsRaw, x) == 16
        && offset_of!(EmbedNativeArgsRaw, status) == 24
        && offset_of!(EmbedNativeArgsRaw, m) == 32
        && offset_of!(EmbedNativeArgsRaw, n_rows) == 36
        && offset_of!(EmbedNativeArgsRaw, k) == 40,
    "EmbedNativeArgsRaw must equal inference_cuda_embed_native_args in embed.h"
);

unsafe extern "C" {
    #[link_name = "inference_cuda_embed_native_launch"]
    fn embed_native_launch_raw(
        args: *const EmbedNativeArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one native gather reads and writes: `blocks` holds the file's Q4_K blocks of an
/// `n_rows` x `k` table (row-major, 144 bytes per 256 values), `ids` the `m` token ids (read
/// when the kernel runs), `x` receives `m` rows of `k` floats, and `status` is the step status
/// word that a bad id ORs bit 4 into. Bit equal to the tiled gather on the same weights.
pub(in crate::kernels::cuda) struct EmbedNativeRaw<'a> {
    pub(in crate::kernels::cuda) blocks: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_rows: c_int,
    pub(in crate::kernels::cuda) k: c_int,
}

/// Bytes each range of a native gather must hold: Q4_K only, any row count from 1 up (no
/// 8-row rule), `k` a multiple of 256. `None` for a shape or token count the kernel refuses.
struct NativeNeeds {
    blocks: usize,
    ids: usize,
    x: usize,
}

fn native_needs(call: &EmbedNativeRaw<'_>) -> Option<NativeNeeds> {
    let m = usize::try_from(call.m).ok()?;
    let rows = usize::try_from(call.n_rows).ok()?;
    let k = usize::try_from(call.k).ok()?;
    let table_ok = (1..=MAX_ROWS).contains(&rows)
        && (256..=MAX_K).contains(&k)
        && k.is_multiple_of(256)
        && (1..=MAX_TOKENS).contains(&m);
    if !table_ok {
        return None;
    }
    Some(NativeNeeds {
        // The Q4_K block count of the table (144 bytes per 256 values).
        blocks: QuantType::Q4K.weight_bytes(rows, k)?,
        ids: m.checked_mul(size_of::<c_int>())?,
        x: m.checked_mul(k)?.checked_mul(size_of::<f32>())?,
    })
}

/// Proves a native gather before the native call: shape, `m`, sizes, alignment (blocks and `x`
/// 16, ids and status 4) and that `x` and `status`, which the kernel writes, overlap none of the
/// other ranges.
fn check_embed_native(call: &EmbedNativeRaw<'_>) -> Result<(), DeviceFailure> {
    let need = native_needs(call).ok_or_else(invalid_argument)?;
    let sized = fits(&call.blocks, need.blocks, 16)
        && fits(&call.ids, need.ids, 4)
        && fits(&call.x, need.x, 16)
        && fits(&call.status, size_of::<c_int>(), 4);
    let blocks = Span::of(&call.blocks, need.blocks);
    let ids = Span::of(&call.ids, need.ids);
    let x = Span::of(&call.x, need.x);
    let status = Span::of(&call.status, size_of::<c_int>());
    let apart = x.apart(&ids)
        && x.apart(&blocks)
        && x.apart(&status)
        && status.apart(&ids)
        && status.apart(&blocks);
    if !(sized && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the native Q4_K gather of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn embed_native_launch(
    stream: &RawStream,
    call: &EmbedNativeRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_embed_native(call)?;
    let args = EmbedNativeArgsRaw {
        blocks: call.blocks.address().cast_const(),
        ids: call.ids.address().cast::<c_int>().cast_const(),
        x: call.x.address().cast::<f32>(),
        status: call.status.address().cast::<c_int>(),
        m: call.m,
        n_rows: call.n_rows,
        k: call.k,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_embed_native` proved the shape, the blocks, ids, x and status ranges inside
    // their live allocations, large enough for the call, aligned, and `x` and `status` disjoint
    // from every other range; the adapter repeats the checks on the addresses. The allocations
    // stay valid until the queued launch finishes because `inference_cuda_free` synchronizes the
    // device first. The kernel reads the ids when it runs and the adapter keeps no pointer.
    // `args` and `stage` are live for the call, and `stream` is live for the call.
    let status = unsafe { embed_native_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::{RawDevice, STATUS_INVALID_ARGUMENT};
    use super::*;

    const TILES: usize = 0x10_0000;
    const IDS: usize = 0x20_0000;
    const X: usize = 0x30_0000;
    const STATUS: usize = 0x40_0000;

    fn pointer(address: usize) -> *mut c_void {
        std::ptr::without_provenance_mut::<c_void>(address)
    }

    /// A valid-looking Q4_K call of one token on the smallest table, at made-up addresses.
    fn args() -> EmbedArgsRaw {
        EmbedArgsRaw {
            tiles: pointer(TILES).cast_const(),
            ids: pointer(IDS).cast::<c_int>().cast_const(),
            x: pointer(X).cast::<f32>(),
            status: pointer(STATUS).cast::<c_int>(),
            ggml_type: 12,
            m: 1,
            n_rows: 8,
            k: 256,
        }
    }

    /// The `(status, stage)` the entry answers for `args` on the made-up `stream`.
    fn launch(args: *const EmbedArgsRaw, stream: usize) -> (c_int, c_int) {
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it reads a pointer or
        // makes a CUDA call; `stage` is a live, writable c_int.
        let status = unsafe { embed_launch_raw(args, pointer(stream), &mut stage) };
        (status, stage)
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call, so the
    /// test needs no device. The rows are the terms of `valid()` in `embed.cu`.
    #[test]
    fn the_native_embed_entry_refuses_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        assert_eq!(launch(std::ptr::null(), 1), refused, "NULL args");
        assert_eq!(launch(&args(), 0), refused, "NULL stream");
        type Change = fn(&mut EmbedArgsRaw);
        let cases: [(&str, Change); 22] = [
            ("NULL tiles", |a| a.tiles = std::ptr::null()),
            ("NULL ids", |a| a.ids = std::ptr::null()),
            ("NULL x", |a| a.x = std::ptr::null_mut()),
            ("NULL status", |a| a.status = std::ptr::null_mut()),
            ("type 14", |a| a.ggml_type = 14),
            ("type 0", |a| a.ggml_type = 0),
            ("m 0", |a| a.m = 0),
            ("m 4097", |a| a.m = 4097),
            ("n_rows 0", |a| a.n_rows = 0),
            ("n_rows 12", |a| a.n_rows = 12),
            ("n_rows 2^28 + 8", |a| a.n_rows = (1 << 28) + 8),
            ("k 0", |a| a.k = 0),
            ("k 100", |a| a.k = 100),
            ("k 300", |a| a.k = 300),
            ("k 33024", |a| a.k = 33024),
            ("tiles misaligned", |a| {
                a.tiles = pointer(TILES + 8).cast_const()
            }),
            ("x misaligned", |a| a.x = pointer(X + 4).cast::<f32>()),
            ("ids misaligned", |a| {
                a.ids = pointer(IDS + 2).cast::<c_int>().cast_const()
            }),
            ("status misaligned", |a| {
                a.status = pointer(STATUS + 2).cast::<c_int>()
            }),
            ("x over ids", |a| a.x = pointer(IDS).cast::<f32>()),
            ("x over the tiles", |a| a.x = pointer(TILES).cast::<f32>()),
            ("status over x", |a| a.status = pointer(X).cast::<c_int>()),
        ];
        for (why, change) in cases {
            let mut bad = args();
            change(&mut bad);
            assert_eq!(launch(&bad, 1), refused, "refuses {why}");
        }
        let mut bad = args();
        bad.ids = pointer(X + 4).cast::<c_int>().cast_const();
        assert_eq!(launch(&bad, 1), refused, "refuses ids inside x");
        let mut bad = args();
        bad.status = pointer(IDS).cast::<c_int>();
        assert_eq!(launch(&bad, 1), refused, "refuses status over ids");
        let mut bad = args();
        bad.status = pointer(TILES + 64).cast::<c_int>();
        assert_eq!(launch(&bad, 1), refused, "refuses status inside the tiles");
        check_the_rust_rechecks();
    }

    /// The four fixture allocations of `check_embed`: tiles, ids, x and status, none backed.
    fn devices() -> [RawDevice; 4] {
        [(); 4].map(|()| RawDevice::unbacked(4096))
    }

    fn range(device: &RawDevice, offset: usize, bytes: usize) -> RawRange<'_> {
        RawRange::new(device, offset, bytes).expect("a range inside the fixture")
    }

    /// One token of 256 values against the smallest Q4_K table, each range in its own allocation.
    fn base(d: &[RawDevice; 4]) -> EmbedRaw<'_> {
        EmbedRaw {
            ty: TiledType::Q4K,
            tiles: range(&d[0], 0, 1152),
            ids: range(&d[1], 0, 4),
            x: range(&d[2], 0, 1024),
            status: range(&d[3], 0, 4),
            m: 1,
            n_rows: 8,
            k: 256,
        }
    }

    /// `check_embed` refuses, on the host and before the C entry sees them, the cases the C
    /// entry cannot see (a range too short for its use) and the ones it would also refuse.
    fn check_the_rust_rechecks() {
        let d = devices();
        assert!(check_embed(&base(&d)).is_ok(), "the base call");
        let cases: [(&str, EmbedRaw<'_>); 19] = [
            (
                "k 300 with room for its 1200 bytes of x",
                EmbedRaw {
                    k: 300,
                    x: range(&d[2], 0, 1280),
                    ..base(&d)
                },
            ),
            ("k 0", EmbedRaw { k: 0, ..base(&d) }),
            (
                "n_rows 12",
                EmbedRaw {
                    n_rows: 12,
                    ..base(&d)
                },
            ),
            ("m 0", EmbedRaw { m: 0, ..base(&d) }),
            (
                "m 4097",
                EmbedRaw {
                    m: 4097,
                    ..base(&d)
                },
            ),
            (
                "Q6_K",
                EmbedRaw {
                    ty: TiledType::Q6K,
                    ..base(&d)
                },
            ),
            (
                "tiles short",
                EmbedRaw {
                    tiles: range(&d[0], 0, 1136),
                    ..base(&d)
                },
            ),
            (
                "ids short",
                EmbedRaw {
                    ids: range(&d[1], 0, 2),
                    ..base(&d)
                },
            ),
            (
                "x short",
                EmbedRaw {
                    x: range(&d[2], 0, 1020),
                    ..base(&d)
                },
            ),
            (
                "status short",
                EmbedRaw {
                    status: range(&d[3], 0, 2),
                    ..base(&d)
                },
            ),
            (
                "tiles misaligned",
                EmbedRaw {
                    tiles: range(&d[0], 8, 1152),
                    ..base(&d)
                },
            ),
            (
                "ids misaligned",
                EmbedRaw {
                    ids: range(&d[1], 2, 4),
                    ..base(&d)
                },
            ),
            (
                "x misaligned",
                EmbedRaw {
                    x: range(&d[2], 4, 1024),
                    ..base(&d)
                },
            ),
            (
                "status misaligned",
                EmbedRaw {
                    status: range(&d[3], 2, 4),
                    ..base(&d)
                },
            ),
            (
                "x over ids",
                EmbedRaw {
                    ids: range(&d[2], 0, 4),
                    ..base(&d)
                },
            ),
            (
                "x over the tiles",
                EmbedRaw {
                    tiles: range(&d[2], 0, 1152),
                    ..base(&d)
                },
            ),
            (
                "status over x",
                EmbedRaw {
                    status: range(&d[2], 8, 4),
                    ..base(&d)
                },
            ),
            (
                "status over ids",
                EmbedRaw {
                    ids: range(&d[3], 0, 4),
                    ..base(&d)
                },
            ),
            (
                "status over the tiles",
                EmbedRaw {
                    tiles: range(&d[3], 0, 1152),
                    ..base(&d)
                },
            ),
        ];
        for (why, call) in cases {
            assert!(check_embed(&call).is_err(), "check_embed refuses {why}");
        }
    }

    // --- The native twin ----------------------------------------------------------------

    const BLOCKS: usize = 0x50_0000;

    /// A valid-looking native Q4_K call of one token on the smallest table, at made-up
    /// addresses one MiB apart.
    fn native_args() -> EmbedNativeArgsRaw {
        EmbedNativeArgsRaw {
            blocks: pointer(BLOCKS).cast_const(),
            ids: pointer(IDS).cast::<c_int>().cast_const(),
            x: pointer(X).cast::<f32>(),
            status: pointer(STATUS).cast::<c_int>(),
            m: 1,
            n_rows: 8,
            k: 256,
        }
    }

    /// The `(status, stage)` the native entry answers for `args` on the made-up `stream`.
    fn launch_native(args: *const EmbedNativeArgsRaw, stream: usize) -> (c_int, c_int) {
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it reads a pointer or
        // makes a CUDA call; `stage` is a live, writable c_int.
        let status = unsafe { embed_native_launch_raw(args, pointer(stream), &mut stage) };
        (status, stage)
    }

    /// Host only: every case is refused by the entry's own `valid_native` before any CUDA
    /// call, so the test needs no device. The rows are the terms of `valid_native` in
    /// embed.cu: no type term (the entry is Q4_K by construction), no 8-row rule, and the
    /// row bound is inclusive at 2^28. Each case flips exactly one later term while every
    /// earlier term of `valid_native` still holds (a case refused by an earlier guard proves
    /// nothing).
    #[test]
    fn the_embed_native_entry_refuses_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        assert_eq!(launch_native(std::ptr::null(), 1), refused, "NULL args");
        assert_eq!(launch_native(&native_args(), 0), refused, "NULL stream");
        type Change = fn(&mut EmbedNativeArgsRaw);
        let cases: [(&str, Change); 17] = [
            ("NULL blocks", |a| a.blocks = std::ptr::null()),
            ("NULL ids", |a| a.ids = std::ptr::null()),
            ("NULL x", |a| a.x = std::ptr::null_mut()),
            ("NULL status", |a| a.status = std::ptr::null_mut()),
            ("m 0", |a| a.m = 0),
            ("m 4097", |a| a.m = 4097),
            ("n_rows 0", |a| a.n_rows = 0),
            ("n_rows 2^28 + 1", |a| a.n_rows = (1 << 28) + 1),
            ("k 0", |a| a.k = 0),
            ("k 100", |a| a.k = 100),
            ("k 300", |a| a.k = 300),
            ("k 33024", |a| a.k = 33024),
            ("blocks misaligned", |a| {
                a.blocks = pointer(BLOCKS + 8).cast_const()
            }),
            ("x misaligned", |a| a.x = pointer(X + 4).cast::<f32>()),
            ("ids misaligned", |a| {
                a.ids = pointer(IDS + 2).cast::<c_int>().cast_const()
            }),
            ("status misaligned", |a| {
                a.status = pointer(STATUS + 2).cast::<c_int>()
            }),
            ("x over ids", |a| a.x = pointer(IDS).cast::<f32>()),
        ];
        for (why, change) in cases {
            let mut bad = native_args();
            change(&mut bad);
            assert_eq!(launch_native(&bad, 1), refused, "refuses {why}");
        }
        let mut bad = native_args();
        bad.x = pointer(BLOCKS).cast::<f32>();
        assert_eq!(launch_native(&bad, 1), refused, "refuses x over the blocks");
        let mut bad = native_args();
        bad.status = pointer(X).cast::<c_int>();
        assert_eq!(launch_native(&bad, 1), refused, "refuses status over x");
        let mut bad = native_args();
        bad.ids = pointer(X + 4).cast::<c_int>().cast_const();
        assert_eq!(launch_native(&bad, 1), refused, "refuses ids inside x");
        let mut bad = native_args();
        bad.status = pointer(IDS).cast::<c_int>();
        assert_eq!(launch_native(&bad, 1), refused, "refuses status over ids");
        let mut bad = native_args();
        bad.status = pointer(BLOCKS + 64).cast::<c_int>();
        assert_eq!(
            launch_native(&bad, 1),
            refused,
            "refuses status inside the blocks"
        );
        check_the_native_rust_rechecks();
    }

    /// One token of 256 values against the smallest native table, each range in its own
    /// allocation.
    fn native_base(d: &[RawDevice; 4]) -> EmbedNativeRaw<'_> {
        EmbedNativeRaw {
            blocks: range(&d[0], 0, 1152),
            ids: range(&d[1], 0, 4),
            x: range(&d[2], 0, 1024),
            status: range(&d[3], 0, 4),
            m: 1,
            n_rows: 8,
            k: 256,
        }
    }

    /// `check_embed_native` refuses, on the host and before the C entry sees them, the cases
    /// the C entry cannot see (a range too short for its use) and the ones it would also
    /// refuse; 12 rows is its POSITIVE control, the row count the tiled twin refuses.
    fn check_the_native_rust_rechecks() {
        let d = devices();
        assert!(
            check_embed_native(&native_base(&d)).is_ok(),
            "the base call"
        );
        assert!(
            check_embed_native(&EmbedNativeRaw {
                blocks: range(&d[0], 0, 12 * 144),
                n_rows: 12,
                ..native_base(&d)
            })
            .is_ok(),
            "12 rows: no 8-row rule on the native path"
        );
        let cases: [(&str, EmbedNativeRaw<'_>); 18] = [
            (
                "k 300 with room for its 1200 bytes of x",
                EmbedNativeRaw {
                    k: 300,
                    x: range(&d[2], 0, 1280),
                    ..native_base(&d)
                },
            ),
            (
                "k 0",
                EmbedNativeRaw {
                    k: 0,
                    ..native_base(&d)
                },
            ),
            (
                "n_rows 0",
                EmbedNativeRaw {
                    n_rows: 0,
                    ..native_base(&d)
                },
            ),
            (
                "m 0",
                EmbedNativeRaw {
                    m: 0,
                    ..native_base(&d)
                },
            ),
            (
                "m 4097",
                EmbedNativeRaw {
                    m: 4097,
                    ..native_base(&d)
                },
            ),
            (
                "blocks short",
                EmbedNativeRaw {
                    blocks: range(&d[0], 0, 1136),
                    ..native_base(&d)
                },
            ),
            (
                "ids short",
                EmbedNativeRaw {
                    ids: range(&d[1], 0, 2),
                    ..native_base(&d)
                },
            ),
            (
                "x short",
                EmbedNativeRaw {
                    x: range(&d[2], 0, 1020),
                    ..native_base(&d)
                },
            ),
            (
                "status short",
                EmbedNativeRaw {
                    status: range(&d[3], 0, 2),
                    ..native_base(&d)
                },
            ),
            (
                "blocks misaligned",
                EmbedNativeRaw {
                    blocks: range(&d[0], 8, 1152),
                    ..native_base(&d)
                },
            ),
            (
                "ids misaligned",
                EmbedNativeRaw {
                    ids: range(&d[1], 2, 4),
                    ..native_base(&d)
                },
            ),
            (
                "x misaligned",
                EmbedNativeRaw {
                    x: range(&d[2], 4, 1024),
                    ..native_base(&d)
                },
            ),
            (
                "status misaligned",
                EmbedNativeRaw {
                    status: range(&d[3], 2, 4),
                    ..native_base(&d)
                },
            ),
            (
                "x over ids",
                EmbedNativeRaw {
                    ids: range(&d[2], 0, 4),
                    ..native_base(&d)
                },
            ),
            (
                "x over the blocks",
                EmbedNativeRaw {
                    blocks: range(&d[2], 0, 1152),
                    ..native_base(&d)
                },
            ),
            (
                "status over x",
                EmbedNativeRaw {
                    status: range(&d[2], 8, 4),
                    ..native_base(&d)
                },
            ),
            (
                "status over ids",
                EmbedNativeRaw {
                    ids: range(&d[3], 0, 4),
                    ..native_base(&d)
                },
            ),
            (
                "status over the blocks",
                EmbedNativeRaw {
                    blocks: range(&d[3], 0, 1152),
                    ..native_base(&d)
                },
            ),
        ];
        for (why, call) in cases {
            assert!(check_embed_native(&call).is_err(), "refuses {why}");
        }
    }
}
