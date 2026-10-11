//! Host repack of GGUF Q5_0 weights into the device tile (MEL-C1a).
//!
//! A tile is 8 rows by 128 values: 4 blocks per row, 32 ggml `block_q5_0`
//! of 22 bytes (`d`: 2 bytes, `qh`: 4, `qs`: 16), 704 bytes in all. Block
//! `b = 4 * row + j` (`j` is the block's index inside the tile's 128 columns)
//! has its three fields stored planar:
//!
//! ```text
//! tile byte 0          64                192                       704
//!           | d[0..32] | qh[0..32]       | qs[0..32]               |
//!           | 2 B each | 4 B each        | 16 B each               |
//! offsets:    2 * b      64 + 4 * b        192 + 16 * b
//! ```
//!
//! This is a pure byte permutation: every output byte is one input byte,
//! copied unchanged, and the output is as long as the input. There is no
//! arithmetic, no rescaling of `d`, no sign change (the `-16` offset of the
//! quants stays in the interpretation, not in the bytes). Tiles are
//! group-major like the vendored ones: group `g` of 8 rows starts at
//! `g * (k / 128) * 704`, which is also where its source blocks start.
//!
//! There is no padding. A row length is accepted only when it is a multiple
//! of 128, so for K = 896 (Mellum `ffn_down_exps`) the tiles end exactly at
//! block 28 of each row and no guard bytes exist after it.

use crate::EngError;

/// Rows of a tile; a group is one tile row.
const ROWS: usize = 8;
/// Blocks (32 values) per row of one tile.
const BLOCKS_PER_TILE_ROW: usize = 4;
const BLOCK_BYTES: usize = 22;
const D_BYTES: usize = 2;
const QH_BYTES: usize = 4;
const QS_BYTES: usize = 16;
/// Start of each field region inside a tile.
const D_AT: usize = 0;
const QH_AT: usize = 64;
const QS_AT: usize = 192;
const TILE_BYTES: usize = 704;
/// Values per row of one tile.
const TILE_COLS: usize = 128;

fn bad(message: String) -> EngError {
    EngError::Checkpoint(format!("Q5_0 tile repack: {message}"))
}

/// Permutes the ggml Q5_0 blocks `src` of rows of `k` values into tiles in
/// `dst`. `src` holds whole 8-row groups and `dst` is exactly as long.
///
/// Returns `Checkpoint` when `k` is not a positive multiple of 128, when
/// `src` is not whole groups, or when `dst` has another length; `dst` is
/// untouched then.
pub(super) fn pack_q5_0(src: &[u8], k: usize, dst: &mut [u8]) -> Result<(), EngError> {
    if k == 0 || !k.is_multiple_of(TILE_COLS) {
        return Err(bad(format!(
            "row length {k} is not a multiple of {TILE_COLS}"
        )));
    }
    let group_bytes = ROWS * (k / 32) * BLOCK_BYTES;
    if !src.len().is_multiple_of(group_bytes) {
        return Err(bad(format!(
            "{} source bytes are not whole groups of {group_bytes}",
            src.len()
        )));
    }
    if dst.len() != src.len() {
        return Err(bad(format!(
            "{} tile bytes for {} source bytes",
            dst.len(),
            src.len()
        )));
    }
    for (group_src, group_dst) in src
        .chunks_exact(group_bytes)
        .zip(dst.chunks_exact_mut(group_bytes))
    {
        pack_group(group_src, k / 32, group_dst);
    }
    Ok(())
}

/// One group: 8 rows of `blocks` blocks each, into `blocks / 4` tiles.
fn pack_group(src: &[u8], blocks: usize, dst: &mut [u8]) {
    for (row, row_src) in src.chunks_exact(blocks * BLOCK_BYTES).enumerate() {
        for (kb, block) in row_src.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            let tile = &mut dst[(kb / BLOCKS_PER_TILE_ROW) * TILE_BYTES..][..TILE_BYTES];
            let b = BLOCKS_PER_TILE_ROW * row + kb % BLOCKS_PER_TILE_ROW;
            tile[D_AT + D_BYTES * b..][..D_BYTES].copy_from_slice(&block[..D_BYTES]);
            tile[QH_AT + QH_BYTES * b..][..QH_BYTES]
                .copy_from_slice(&block[D_BYTES..D_BYTES + QH_BYTES]);
            tile[QS_AT + QS_BYTES * b..][..QS_BYTES].copy_from_slice(&block[D_BYTES + QH_BYTES..]);
        }
    }
}
