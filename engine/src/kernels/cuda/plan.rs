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
/// `mma884::Q8_0` times 16. Q4_K and Q8_0 tiles hold the same bytes as their
/// ggml blocks; Q6_K tiles add precomputed fp16 scales and are larger than the
/// ggml blocks.
const Q4K_TILE_BYTES: u64 = 72 * 16;
const Q6K_TILE_BYTES: u64 = 112 * 16;
const Q8_0_TILE_BYTES: u64 = 136 * 16;

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
/// rest is the row count. Q4_K, Q6_K and Q8_0 are stored as tiles, so their row
/// length must be a multiple of 256 and at most 32768, and their row count a
/// multiple of 8. F32 and F16 are stored as read. Any other type, or a shape
/// the tiles cannot hold, fails with `EngError::Checkpoint` naming the tensor.
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
        other => {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} has type {other:?}, which the CUDA backend does not load"
            )));
        }
    };
    check_tile_shape(name, ty, rows, cols)?;
    (rows / TILE_ROWS)
        .checked_mul(cols / TILE_COLS)
        .and_then(|tiles| tiles.checked_mul(tile_bytes))
        .ok_or_else(overflow)
}

fn check_tile_shape(name: &str, ty: GgufType, rows: u64, cols: u64) -> Result<(), EngError> {
    if rows == 0
        || !rows.is_multiple_of(TILE_ROWS)
        || cols == 0
        || !cols.is_multiple_of(TILE_COLS)
        || cols > MAX_COLS
    {
        return Err(EngError::Checkpoint(format!(
            "tensor {name} ({ty:?}) has {rows} rows of {cols}; the CUDA tiles need rows a \
             multiple of {TILE_ROWS} and a row length that is a multiple of {TILE_COLS} up \
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
    fn oversized_dense_tensors_fail_with_memory_plan() {
        assert!(matches!(
            device_bytes("w", GgufType::F32, &[u64::MAX, 2]),
            Err(EngError::MemoryPlan(_))
        ));
    }
}
