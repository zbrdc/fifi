//! Device-memory admission for the CUDA loader (decision 0040 item 5).
//!
//! Pure arithmetic over byte counts: nothing here calls CUDA, so the admission
//! test also runs on hosts without a driver. The loader computes the device
//! size of every tensor, adds the KV cache, the work buffers, and the headroom,
//! and compares the sum with the free device memory before the first upload.

use crate::EngError;
use crate::weights::GgufType;

/// Rows and columns of one tile of the vendored `mma884` layout.
pub(crate) const TILE_ROWS: u64 = 8;
pub(crate) const TILE_COLS: u64 = 256;
/// `kern::kMaxK`: the longest row the vendored matmul accepts.
pub(crate) const MAX_COLS: u64 = 32768;
/// Bytes of one tile: `kTileVecs` of `mma884::Q4_K`, `mma884::Q6_K` and
/// `mma884::Q8_0` times 16. Q4_K, Q8_0 and Q5_0 tiles hold the same bytes as
/// their ggml blocks; Q6_K tiles add precomputed fp16 scales and are larger
/// than the ggml blocks. A Q5_0 tile is 8 rows of 128 values (4 blocks of 22
/// bytes per row) and is built on the host, not by the vendored repack.
const Q4K_TILE_BYTES: u64 = 72 * 16;
const Q6K_TILE_BYTES: u64 = 112 * 16;
const Q8_0_TILE_BYTES: u64 = 136 * 16;
const Q5_0_TILE_BYTES: u64 = 44 * 16;
/// Values per row of one Q5_0 tile.
const Q5_0_TILE_COLS: u64 = 128;

/// Values per row of one tile of `ty`.
fn tile_cols(ty: GgufType) -> u64 {
    match ty {
        GgufType::Q50 => Q5_0_TILE_COLS,
        _ => TILE_COLS,
    }
}

/// Device memory one load needs, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Demand {
    pub(crate) tensors: u64,
    pub(crate) kv_cache: u64,
    pub(crate) work: u64,
    pub(crate) headroom: u64,
}

/// Device bytes of the tensor `name` after upload.
///
/// `dims` is in GGUF order: `dims[0]` is the row length, the product of the
/// rest is the row count. Q4_K, Q6_K, Q8_0 and Q5_0 are stored as tiles, so
/// their row length must be a multiple of the tile columns (256, or 128 for
/// Q5_0) and at most 32768, and their row count a multiple of 8. F32 and F16
/// are stored as read. Any other type, or a shape the tiles cannot hold, fails
/// with `EngError::Checkpoint` naming the tensor.
pub(crate) fn device_bytes(name: &str, ty: GgufType, dims: &[u64]) -> Result<u64, EngError> {
    let overflow = || EngError::MemoryPlan(format!("tensor {name}: device size overflows u64"));
    let (&cols, rest) = dims
        .split_first()
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {name} has no dimensions")))?;
    let rows = rest
        .iter()
        .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(overflow)?;
    let elements = rows.checked_mul(cols).ok_or_else(overflow)?;
    let tile_bytes = match ty {
        GgufType::F32 => return elements.checked_mul(4).ok_or_else(overflow),
        GgufType::F16 => return elements.checked_mul(2).ok_or_else(overflow),
        GgufType::Q4K => Q4K_TILE_BYTES,
        GgufType::Q6K => Q6K_TILE_BYTES,
        GgufType::Q80 => Q8_0_TILE_BYTES,
        GgufType::Q50 => Q5_0_TILE_BYTES,
        other => {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} has type {other:?}, which the CUDA backend does not load"
            )));
        }
    };
    check_tile_shape(name, ty, rows, cols)?;
    (rows / TILE_ROWS)
        .checked_mul(cols / tile_cols(ty))
        .and_then(|tiles| tiles.checked_mul(tile_bytes))
        .ok_or_else(overflow)
}

/// Values per block and bytes per block of the four native-layout types.
const fn native_block(ty: GgufType) -> Option<(u64, u64)> {
    match ty {
        GgufType::Q4K => Some((256, 144)),
        GgufType::Q6K => Some((256, 210)),
        GgufType::Q80 => Some((32, 34)),
        GgufType::Q50 => Some((32, 22)),
        _ => None,
    }
}

/// Device bytes of the tensor `name` after a native upload (MATVEC-PAR S2): the file's own
/// blocks, row-major, exactly the source bytes. Q4_K, Q5_0, Q6_K and Q8_0 only. There is no
/// 8-row rule and no tile-column rule: any row count from one up loads, and `dims[0]` must be
/// a positive multiple of the type's block size up to `MAX_COLS`. The count differs from the
/// tiles only for Q6_K, which needs 1680 native against 1792 tile bytes per 8x256 rows.
/// F32 and F16 count as in [`device_bytes`]: both layouts load them as raw copies. Any other
/// type fails with `EngError::Checkpoint` naming the tensor.
pub(crate) fn native_device_bytes(name: &str, ty: GgufType, dims: &[u64]) -> Result<u64, EngError> {
    if matches!(ty, GgufType::F32 | GgufType::F16) {
        return device_bytes(name, ty, dims);
    }
    let overflow = || EngError::MemoryPlan(format!("tensor {name}: device size overflows u64"));
    let (&cols, rest) = dims
        .split_first()
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {name} has no dimensions")))?;
    let rows = rest
        .iter()
        .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(overflow)?;
    let Some((block_size, block_bytes)) = native_block(ty) else {
        return Err(EngError::Checkpoint(format!(
            "tensor {name} has type {ty:?}, which the CUDA native weights do not take"
        )));
    };
    if rows == 0 || cols == 0 || !cols.is_multiple_of(block_size) || cols > MAX_COLS {
        return Err(EngError::Checkpoint(format!(
            "tensor {name} ({ty:?}) has {rows} rows of {cols}; the native layout needs rows \
             above zero and a row length that is a multiple of {block_size} up to {MAX_COLS}"
        )));
    }
    rows.checked_mul(cols / block_size)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(overflow)
}

fn check_tile_shape(name: &str, ty: GgufType, rows: u64, cols: u64) -> Result<(), EngError> {
    let tile_cols = tile_cols(ty);
    if rows == 0
        || !rows.is_multiple_of(TILE_ROWS)
        || cols == 0
        || !cols.is_multiple_of(tile_cols)
        || cols > MAX_COLS
    {
        return Err(EngError::Checkpoint(format!(
            "tensor {name} ({ty:?}) has {rows} rows of {cols}; the CUDA tiles need rows a \
             multiple of {TILE_ROWS} and a row length that is a multiple of {tile_cols} up \
             to {MAX_COLS}"
        )));
    }
    Ok(())
}

/// Admits `demand` against `free` device bytes and returns the bytes required.
///
/// Fails with `EngError::MemoryPlan` when the sum exceeds `free` or overflows.
pub(crate) fn admit(demand: &Demand, free: u64) -> Result<u64, EngError> {
    let required = [demand.kv_cache, demand.work, demand.headroom]
        .into_iter()
        .try_fold(demand.tensors, u64::checked_add)
        .ok_or_else(|| EngError::MemoryPlan("cuda admission: demand overflows u64".to_owned()))?;
    if required > free {
        return Err(EngError::MemoryPlan(format!(
            "cuda admission: need {required} bytes (tensors {}, KV cache {}, work {}, \
             headroom {}), device has {free} free",
            demand.tensors, demand.kv_cache, demand.work, demand.headroom
        )));
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEMAND: Demand = Demand {
        tensors: 600,
        kv_cache: 200,
        work: 100,
        headroom: 100,
    };

    #[test]
    fn admission_accepts_a_plan_that_fits_exactly() {
        assert_eq!(admit(&DEMAND, 1000), Ok(1000));
    }

    #[test]
    fn admission_rejects_an_oversized_plan_with_memory_plan() {
        let error = admit(&DEMAND, 999).expect_err("one byte short must fail");
        let EngError::MemoryPlan(message) = error else {
            panic!("expected MemoryPlan, got {error:?}");
        };
        assert!(message.contains("need 1000 bytes"), "{message}");
        assert!(message.contains("headroom 100"), "{message}");
        assert!(message.ends_with("device has 999 free"), "{message}");
    }

    #[test]
    fn admission_rejects_an_overflowing_demand() {
        let demand = Demand {
            headroom: u64::MAX,
            ..DEMAND
        };
        assert!(matches!(
            admit(&demand, u64::MAX),
            Err(EngError::MemoryPlan(_))
        ));
    }

    #[test]
    fn dense_tensors_keep_their_file_size() {
        assert_eq!(device_bytes("norm", GgufType::F32, &[5]), Ok(20));
        assert_eq!(device_bytes("table", GgufType::F16, &[3, 7]), Ok(42));
    }

    #[test]
    fn tiled_tensors_use_the_tile_size() {
        // 16 rows of 512: 2 x 2 tiles.
        assert_eq!(device_bytes("w", GgufType::Q4K, &[512, 16]), Ok(4 * 1152));
        assert_eq!(device_bytes("w", GgufType::Q6K, &[512, 16]), Ok(4 * 1792));
        // Q8_0 tiles hold the 64 ggml blocks of 34 bytes each.
        assert_eq!(device_bytes("w", GgufType::Q80, &[512, 16]), Ok(4 * 2176));
        // An expert stack [k, n, experts] counts all rows.
        assert_eq!(device_bytes("e", GgufType::Q4K, &[256, 8, 3]), Ok(3 * 1152));
    }

    #[test]
    fn device_bytes_of_a_q5_0_tensor_equal_its_source_bytes_and_have_no_padding() {
        // Mellum ffn_down_exps: K = 896 is 28 blocks of 22 bytes per row.
        assert_eq!(
            device_bytes("e", GgufType::Q50, &[896, 2304, 64]),
            Ok(28 * 22 * 2304 * 64)
        );
        assert_eq!(
            device_bytes("e", GgufType::Q50, &[896, 2304, 64]),
            Ok(90_832_896)
        );
        assert_eq!(device_bytes("w", GgufType::Q50, &[128, 8]), Ok(704));
        // The 14 Mellum ffn_down_exps tensors.
        let total: u64 = (0..14)
            .map(|_| device_bytes("e", GgufType::Q50, &[896, 2304, 64]).expect("fits"))
            .sum();
        assert_eq!(total, 1_271_660_544);
        for dims in [[864, 8], [128, 12]] {
            assert!(matches!(
                device_bytes("w", GgufType::Q50, &dims),
                Err(EngError::Checkpoint(message)) if message.contains("multiple of 128")
            ));
        }
    }

    #[test]
    fn tiled_tensors_reject_shapes_the_tiles_cannot_hold() {
        for dims in [[256, 12], [128, 8], [0, 8], [256, 0], [MAX_COLS + 256, 8]] {
            let error = device_bytes("w", GgufType::Q4K, &dims).expect_err("bad shape");
            assert!(
                matches!(&error, EngError::Checkpoint(message) if message.starts_with("tensor w (Q4K)")),
                "{dims:?}: {error:?}"
            );
        }
    }

    #[test]
    fn other_types_fail_the_load_with_checkpoint() {
        for ty in [GgufType::Q5K, GgufType::Q40, GgufType::Bf16] {
            assert!(matches!(
                device_bytes("w", ty, &[256, 8]),
                Err(EngError::Checkpoint(_))
            ));
        }
        assert!(matches!(
            device_bytes("w", GgufType::F32, &[]),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn native_bytes_equal_tile_bytes_except_q6_k_for_every_tensor_class() {
        // The tensor classes of the pinned checkpoint (Annex A dims): the Q4_K expert stacks
        // [2048, 512, 256], the Q8_0 dense shapes, the Q4_K token embedding [2048, 248320] and
        // the Q6_K output head. Native equals the tiles everywhere but Q6_K.
        for (ty, dims) in [
            (GgufType::Q4K, &[2048, 512, 256][..]),
            (GgufType::Q80, &[2048, 8192][..]),
            (GgufType::Q80, &[512, 2048][..]),
            (GgufType::Q4K, &[2048, 248_320][..]),
        ] {
            assert_eq!(
                native_device_bytes("w", ty, dims),
                device_bytes("w", ty, dims),
                "{ty:?} {dims:?}"
            );
        }
        // Q6_K: 1680 native against 1792 tile bytes per 8x256 rows; the lm_head of the pinned
        // checkpoint is 27,811,840 bytes smaller natively.
        assert_eq!(native_device_bytes("w", GgufType::Q6K, &[256, 8]), Ok(1680));
        assert_eq!(device_bytes("w", GgufType::Q6K, &[256, 8]), Ok(1792));
        assert_eq!(
            native_device_bytes("out", GgufType::Q6K, &[2048, 248_320]),
            Ok(417_177_600)
        );
        assert_eq!(
            device_bytes("out", GgufType::Q6K, &[2048, 248_320]),
            Ok(444_989_440)
        );
        assert_eq!(444_989_440 - 417_177_600, 27_811_840);
    }

    #[test]
    fn native_bytes_drop_the_tile_shape_rules_and_refuse_other_types() {
        // 12 rows and a 224-value row: off every tile rule, on the Q8_0 block grid.
        assert_eq!(
            native_device_bytes("w", GgufType::Q80, &[224, 12]),
            Ok(12 * 7 * 34)
        );
        assert!(device_bytes("w", GgufType::Q80, &[224, 12]).is_err());
        for dims in [[100, 8], [0, 8], [MAX_COLS + 256, 8], [256, 0]] {
            let error =
                native_device_bytes("embd", GgufType::Q4K, &dims).expect_err("off the native grid");
            assert!(
                matches!(&error, EngError::Checkpoint(message) if message.contains("embd")),
                "{dims:?}: {error:?}"
            );
        }
        assert!(matches!(
            native_device_bytes("w", GgufType::Q5K, &[256, 8]),
            Err(EngError::Checkpoint(message)) if message.contains("tensor w has type Q5K")
        ));
        // F32 and F16 count as in the tiled plan: raw copies in both layouts.
        assert_eq!(native_device_bytes("norm", GgufType::F32, &[5]), Ok(20));
        assert_eq!(native_device_bytes("t", GgufType::F16, &[3, 7]), Ok(42));
        assert!(matches!(
            native_device_bytes("w", GgufType::F32, &[]),
            Err(EngError::Checkpoint(_))
        ));
        // Q5_0 pins its own block, 22 bytes per 32 values (not Q8_0's 34): one row of one
        // block, and the Mellum ffn_down_exps stack. Rows below the tile rule load natively.
        assert_eq!(native_device_bytes("q5", GgufType::Q50, &[32, 1]), Ok(22));
        assert_eq!(
            native_device_bytes("e", GgufType::Q50, &[896, 2304, 64]),
            Ok(28 * 22 * 2304 * 64)
        );
    }

    #[test]
    fn oversized_dense_tensors_fail_with_memory_plan() {
        assert!(matches!(
            device_bytes("w", GgufType::F32, &[u64::MAX, 2]),
            Err(EngError::MemoryPlan(_))
        ));
    }
}
