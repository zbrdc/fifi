//! Device tests of the CUDA projector (decision 0040 packet 3 gates).
//!
//! The gates decode the device tiles back to the GGUF block layout and compare
//! them with the source bytes, so the repack is checked against the file rather
//! than against itself. Blocks come from a fixed xorshift stream so a failure
//! reproduces exactly. Each device test is skipped when no device is visible;
//! INFERENCE_CUDA_REQUIRE_DEVICE turns that skip into a failure.

use super::load_policy::{OOM_RETRIES, is_device_out_of_memory, tests::resident_bytes};
use super::*;
use crate::family::project::{DenseProjector, DenseWeight};
use crate::kernels::cuda::device_count;
use crate::weights::{GgufHeader, GgufLimits, dequantize};
use std::{
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    ops::Range,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const TILE_ROWS: usize = 8;
const N_ROWS: usize = 64;
const K: usize = 512;
const BLOCK_VALUES: usize = 256;
const BLOCKS_PER_ROW: usize = K / BLOCK_VALUES;
const Q4K_BLOCK_BYTES: usize = 144;
const Q6K_BLOCK_BYTES: usize = 210;
/// One Q8_0 block: fp16 `d` and 32 int8 quants. Eight of them cover 256 values.
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCKS_PER_256: usize = 8;
/// 72 and 112 sixteen-byte vectors per 8 x 256 tile (`mma884::kTileVecs`).
const Q4K_TILE_BYTES: usize = 72 * 16;
const Q6K_TILE_BYTES: usize = 112 * 16;
const Q8_0_TILE_BYTES: usize = 136 * 16;
/// The Q6_K tile stores its fp16 scale words after the 96 vectors of quants.
const Q6K_SCALE_BASE: usize = 96 * 16;
pub(super) const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
pub(super) const NMSE_BOUND: f64 = 1e-5;
const ALIGN: usize = 32;
/// Two loads of the 1 GiB token_embd may raise the peak resident set by less than
/// this: a chunk plus its tile buffer is 128 MiB at most, and a whole-tensor read
/// would add at least the 1 GiB tensor.
const LARGE_RSS_BOUND_KB: u64 = 512 * 1024;
/// Mirrors `context::CHUNK_LIMIT` (64 MiB of tile bytes per upload chunk).
const CHUNK_BYTES: usize = 64 << 20;
/// 3648 rows of 32768 values: 456 groups of 147456 tile bytes, so the upload
/// splits into a chunk of 455 groups and a chunk of one group.
const CHUNK_ROWS: usize = 3648;
const CHUNK_K: usize = 32768;
const CHUNK_BLOCKS_PER_ROW: usize = CHUNK_K / BLOCK_VALUES;
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 1 << 20,
    max_tensor_bytes: 1 << 24,
};
/// The open cap of decision 0040 item 4: a tensor up to 1 GiB opens, and `load`
/// streams it in chunks whatever its size.
const LARGE_LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 1 << 20,
    max_tensor_bytes: 1 << 30,
};
/// The measured file is named by the operator, so the reader does not second-guess
/// its tensor sizes.
const MEASURE_LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 1 << 26,
    max_tensor_bytes: u64::MAX,
};

/// xorshift64 with Marsaglia's 13/7/17 shifts. Only repeatability matters here.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 32) as u8
    }

    /// A value in `[-1, 1)` from the top 24 bits, exact in f32.
    pub(super) fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// An fp16 pattern of magnitude in `[2^-8, 2^-4)` with random sign and mantissa.
/// Small scales keep `d * scale` well inside the fp16 range that the Q6_K check
/// accepts, so the fixtures exercise the decode and not the overflow guard.
fn small_f16(rng: &mut Rng) -> [u8; 2] {
    let exponent = 7 + (rng.next_u64() % 4) as u16;
    let mantissa = (rng.next_u64() as u16) & 0x3ff;
    let sign = (rng.next_u64() as u16) & 1;
    ((sign << 15) | (exponent << 10) | mantissa).to_le_bytes()
}

/// Q4_K blocks: fp16 `d` and `dmin`, then 140 random bytes of scales and quants.
fn q4k_blocks(rng: &mut Rng, blocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks * Q4K_BLOCK_BYTES);
    for _ in 0..blocks {
        out.extend_from_slice(&small_f16(rng));
        out.extend_from_slice(&small_f16(rng));
        out.extend((0..Q4K_BLOCK_BYTES - 4).map(|_| rng.byte()));
    }
    out
}

/// Q6_K blocks: 128 ql bytes, 64 qh bytes and 16 int8 scales (all random), then fp16 `d`.
fn q6k_blocks(rng: &mut Rng, blocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks * Q6K_BLOCK_BYTES);
    for _ in 0..blocks {
        out.extend((0..Q6K_BLOCK_BYTES - 2).map(|_| rng.byte()));
        out.extend_from_slice(&small_f16(rng));
    }
    out
}

/// Q8_0 blocks for `groups` runs of 256 values: eight blocks per run, each a
/// small fp16 `d` and 32 random int8 quants (including -128).
fn q8_0_blocks(rng: &mut Rng, groups: usize) -> Vec<u8> {
    let blocks = groups * Q8_0_BLOCKS_PER_256;
    let mut out = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
    for _ in 0..blocks {
        out.extend_from_slice(&small_f16(rng));
        out.extend((0..Q8_0_BLOCK_BYTES - 2).map(|_| rng.byte()));
    }
    out
}

/// One tensor for the fixture file.
struct Entry {
    name: &'static str,
    ty: GgufType,
    dims: Vec<u64>,
    payload: Vec<u8>,
}

/// A rank-2 tensor of `shape = (rows, k)`, stored in GGUF order as `dims = [k, rows]`.
fn weight_entry(
    name: &'static str,
    ty: GgufType,
    shape: (usize, usize),
    payload: Vec<u8>,
) -> Entry {
    let (rows, k) = shape;
    Entry {
        name,
        ty,
        dims: vec![k as u64, rows as u64],
        payload,
    }
}

/// A rank-3 stack of `shape = (experts, out, k)`, stored in GGUF order as
/// `dims = [k, out, experts]`: expert `e` is rows `e * out..(e + 1) * out`.
fn stack_entry(
    name: &'static str,
    ty: GgufType,
    shape: (usize, usize, usize),
    payload: Vec<u8>,
) -> Entry {
    let (experts, out, k) = shape;
    Entry {
        name,
        ty,
        dims: vec![k as u64, out as u64, experts as u64],
        payload,
    }
}

/// A GGUF v3 file with no metadata. Each payload starts at a 32-byte-aligned
/// offset of the data section, and the data section starts 32-byte aligned too.
fn gguf_bytes(entries: &[Entry]) -> Vec<u8> {
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    let mut data: Vec<u8> = Vec::new();
    for entry in entries {
        header.extend_from_slice(&(entry.name.len() as u64).to_le_bytes());
        header.extend_from_slice(entry.name.as_bytes());
        header.extend_from_slice(&(entry.dims.len() as u32).to_le_bytes());
        for &dim in &entry.dims {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&entry.ty.id().to_le_bytes());
        data.resize(data.len().next_multiple_of(ALIGN), 0);
        header.extend_from_slice(&(data.len() as u64).to_le_bytes());
        data.extend_from_slice(&entry.payload);
    }
    header.resize(header.len().next_multiple_of(ALIGN), 0);
    header.extend_from_slice(&data);
    header
}

fn open_fixture(entries: &[Entry]) -> (tempfile::TempDir, GgufWeights) {
    open_with(entries, LIMITS)
}

/// Writes the entries to a GGUF file in a temp dir and opens it. The dir is
/// returned too, because the weights read from that file on load.
fn open_with(entries: &[Entry], limits: GgufLimits) -> (tempfile::TempDir, GgufWeights) {
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path = dir.path().join("fixture.gguf");
    std::fs::write(&path, gguf_bytes(entries)).expect("write the GGUF fixture");
    let weights = GgufWeights::open(&path, limits, None).expect("open the GGUF fixture");
    (dir, weights)
}

/// A Q4_K or Q6_K weight written as tensor `w`, with the dense reference the
/// device output is compared against.
struct Fixture {
    _dir: tempfile::TempDir,
    weights: GgufWeights,
    dense: DenseWeight,
}

impl Fixture {
    fn new(ty: GgufType, source: Vec<u8>) -> Self {
        let dense = DenseWeight::matrix(
            N_ROWS,
            K,
            dequantize(ty, &source).expect("dequantize the source"),
        );
        let (dir, weights) = open_fixture(&[weight_entry("w", ty, (N_ROWS, K), source)]);
        Self {
            _dir: dir,
            weights,
            dense,
        }
    }
}

/// Host error text by its Debug form, so the helpers need no `T: Debug`.
fn error_text<T>(result: Result<T, EngError>) -> String {
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => format!("{error:?}"),
    }
}

/// Asserts the error variant and that its message contains `needle`.
fn assert_error<T>(result: Result<T, EngError>, kind: &str, needle: &str) {
    let text = error_text(result);
    assert!(
        text.starts_with(kind) && text.contains(needle),
        "expected {kind} containing {needle:?}, got {text}"
    );
}

/// Runs a device gate only where a device is visible. INFERENCE_CUDA_REQUIRE_DEVICE
/// turns a missing device into a failure, so a GPU job cannot pass by skipping.
pub(super) fn device_present(test: &str) -> bool {
    let count = device_count().expect("CUDA device count query");
    println!("CUDA devices visible: {count}");
    if count > 0 {
        return true;
    }
    let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
        .is_ok_and(|value| !value.is_empty() && value != "0");
    assert!(
        !required,
        "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible"
    );
    println!("SKIPPED {test}: no CUDA device");
    false
}

/// fp16 bits -> f32 from the fields, so the scale checks do not rely on the
/// production decoder they are meant to check. Every fp16 value is exact in f32.
pub(super) fn f16_value(bits: u16) -> f32 {
    let sign = if (bits & 0x8000) != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = bits & 0x3ff;
    match exponent {
        0 => sign * f32::from(fraction) * 2f32.powi(-24),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + f32::from(fraction) / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Round-to-nearest-even f32 -> fp16, the host oracle for the scale words. It is
/// written from the bit fields and shares no code with the device path.
fn f32_to_f16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return if mantissa != 0 { 0x7e00 } else { sign | 0x7c00 };
    }
    // f32 zero or subnormal: far below half of the smallest fp16 subnormal.
    if exponent == 0 {
        return sign;
    }
    let e = exponent as i32 - 127;
    let full = (1u32 << 23) | mantissa;
    if e >= 16 {
        sign | 0x7c00
    } else if e >= -14 {
        normal_f16(sign, e, full)
    } else {
        subnormal_f16(sign, e, full)
    }
}

/// Keeps 11 significant bits of the 24-bit significand, ties to even.
fn normal_f16(sign: u16, e: i32, full: u32) -> u16 {
    let mut m = full >> 13;
    let rem = full & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m % 2 == 1) {
        m += 1;
    }
    let mut e = e;
    if m == 0x800 {
        m = 0x400;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7c00;
    }
    sign | (((e + 15) as u16) << 10) | (m as u16 - 0x400)
}

/// Subnormal fp16 counts units of 2^-24, so the shift is `-1 - e` (14 at e = -15).
fn subnormal_f16(sign: u16, e: i32, full: u32) -> u16 {
    let shift = (-1 - e) as u32;
    // Below 2^-25 the value rounds to zero; at exactly 2^-25 the tie rounds to even.
    if shift > 24 {
        return sign;
    }
    let m = full >> shift;
    let rem = full & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let m = if rem > half || (rem == half && m % 2 == 1) {
        m + 1
    } else {
        m
    };
    sign | m as u16
}

/// Geometry of a tiled weight. The blocks are in source order: row-major, then
/// block-major. The tiles are group-major: the tile of rows `8g..8g+8` and block
/// `b` starts at `(g * blocks_per_row + b) * tile_bytes`.
#[derive(Clone, Copy)]
pub(super) struct Layout {
    block_bytes: usize,
    tile_bytes: usize,
    n_rows: usize,
    blocks_per_row: usize,
}

impl Layout {
    pub(super) fn q4k(n_rows: usize, blocks_per_row: usize) -> Self {
        Self {
            block_bytes: Q4K_BLOCK_BYTES,
            tile_bytes: Q4K_TILE_BYTES,
            n_rows,
            blocks_per_row,
        }
    }

    pub(super) fn q6k(n_rows: usize, blocks_per_row: usize) -> Self {
        Self {
            block_bytes: Q6K_BLOCK_BYTES,
            tile_bytes: Q6K_TILE_BYTES,
            n_rows,
            blocks_per_row,
        }
    }

    /// Q8_0 in 256-value units: the eight 34-byte blocks of one row and tile
    /// column are contiguous in the source, so they act as one 272-byte block.
    pub(super) fn q8_0(n_rows: usize, blocks_per_row: usize) -> Self {
        Self {
            block_bytes: Q8_0_BLOCKS_PER_256 * Q8_0_BLOCK_BYTES,
            tile_bytes: Q8_0_TILE_BYTES,
            n_rows,
            blocks_per_row,
        }
    }

    fn block_count(&self) -> usize {
        self.n_rows * self.blocks_per_row
    }

    fn tile_total(&self) -> usize {
        self.n_rows / TILE_ROWS * self.blocks_per_row * self.tile_bytes
    }

    fn block_range(&self, row: usize, block: usize) -> Range<usize> {
        let start = (row * self.blocks_per_row + block) * self.block_bytes;
        start..start + self.block_bytes
    }

    fn tile_range(&self, row: usize, block: usize) -> Range<usize> {
        let start = ((row / TILE_ROWS) * self.blocks_per_row + block) * self.tile_bytes;
        start..start + self.tile_bytes
    }
}

/// The lane that holds row `row` (of an 8-row group) for quad `q` of a 256-wide
/// tile. This is the mma fragment layout of `mma884.cuh`.
fn lane_of(row: usize, q: usize) -> usize {
    16 * (row / 4) + 4 * q + row % 4
}

/// Sixteen-byte vector `index` of a tile.
fn vector(tile: &[u8], index: usize) -> &[u8] {
    &tile[16 * index..16 * index + 16]
}

/// Writes one Q4_K block from its tile. Vector `row` holds the 16-byte header,
/// and chunk `c = 2q + h` sits in vector `8 + 32h + lane_of(row, q)`.
fn unpack_q4k_block(tile: &[u8], row: usize, block: &mut [u8]) {
    block[..16].copy_from_slice(vector(tile, row));
    for chunk in 0..8 {
        let (q, half) = (chunk / 2, chunk % 2);
        let at = 16 + 16 * chunk;
        block[at..at + 16].copy_from_slice(vector(tile, 8 + 32 * half + lane_of(row, q)));
    }
}

/// Writes the eight Q8_0 blocks of row `row` (of an 8-row group) from their
/// tile, after `pack_q8_0` in `repack.cu`. The tile opens with 64 fp16 `d`
/// words, row-major by block. Quant bytes `16j..16j+16` of quad `q` (values
/// `64q..64q+64` of the row) sit in vector `8 + 32j + lane_of(row, q)`.
fn unpack_q8_0_blocks(tile: &[u8], row: usize, blocks: &mut [u8]) {
    for (index, block) in blocks
        .as_chunks_mut::<Q8_0_BLOCK_BYTES>()
        .0
        .iter_mut()
        .enumerate()
    {
        let d = 2 * (row * Q8_0_BLOCKS_PER_256 + index);
        block[..2].copy_from_slice(&tile[d..d + 2]);
    }
    for q in 0..4 {
        for j in 0..4 {
            let value = 64 * q + 16 * j;
            let at = (value / 32) * Q8_0_BLOCK_BYTES + 2 + value % 32;
            blocks[at..at + 16].copy_from_slice(vector(tile, 8 + 32 * j + lane_of(row, q)));
        }
    }
}

/// The Q8_0 source bytes rebuilt from the device tiles.
pub(super) fn decode_q8_0(tiles: &[u8], layout: Layout) -> Vec<u8> {
    let mut source = vec![0u8; layout.block_count() * layout.block_bytes];
    for row in 0..layout.n_rows {
        for block in 0..layout.blocks_per_row {
            let tile = &tiles[layout.tile_range(row, block)];
            let out = &mut source[layout.block_range(row, block)];
            unpack_q8_0_blocks(tile, row % TILE_ROWS, out);
        }
    }
    source
}

/// The source bytes rebuilt from the device tiles.
pub(super) fn decode_q4k(tiles: &[u8], layout: Layout) -> Vec<u8> {
    let mut source = vec![0u8; layout.block_count() * layout.block_bytes];
    for row in 0..layout.n_rows {
        for block in 0..layout.blocks_per_row {
            let tile = &tiles[layout.tile_range(row, block)];
            let out = &mut source[layout.block_range(row, block)];
            unpack_q4k_block(tile, row % TILE_ROWS, out);
        }
    }
    source
}

/// The 6-bit quant of element `element` (0..256) of a block in row `row` (of an
/// 8-row group), read from the tile as the device lays it out.
fn q6_from_tile(tile: &[u8], row: usize, element: usize) -> u8 {
    let lane = lane_of(row, element / 64);
    let j = element % 64;
    let m = j % 32;
    let (wi, v) = (m / 4, m % 4);
    let ql = if wi < 4 {
        tile[16 * lane + 4 * wi + v]
    } else {
        tile[16 * (32 + lane) + 4 * (wi - 4) + v]
    };
    let qh = tile[16 * (64 + lane) + 4 * (wi / 2) + v];
    let shift = 4 * (wi % 2);
    if j < 32 {
        (ql & 0xf) | (((qh >> shift) & 3) << 4)
    } else {
        (ql >> 4) | (((qh >> (shift + 2)) & 3) << 4)
    }
}

/// The reference quant of element `i` of a Q6_K block, from the ql/qh indexing
/// of `repack.cu`. The device tile must reproduce it exactly.
fn q6_reference(block: &[u8], i: usize) -> u8 {
    let (half, quarter, l) = (i / 128, (i % 128) / 32, i % 32);
    let ql = block[64 * half + l + 32 * (quarter & 1)];
    let qh = block[128 + 32 * half + l];
    let lo4 = if quarter < 2 { ql & 0xf } else { ql >> 4 };
    lo4 | (((qh >> (2 * quarter)) & 3) << 4)
}

/// The fp16 scale word the tile stores for the 16-element group that holds `element`.
fn stored_scale_bits(tile: &[u8], row: usize, element: usize) -> u16 {
    let lane = lane_of(row, element / 64);
    let at = Q6K_SCALE_BASE + 8 * lane + 2 * ((element % 64) / 16);
    u16::from_le_bytes([tile[at], tile[at + 1]])
}

/// The 6-bit quants of every block, in source order, rebuilt from the tiles.
pub(super) fn q6_quants_from_tiles(tiles: &[u8], layout: Layout) -> Vec<u8> {
    let mut quants = Vec::with_capacity(layout.block_count() * BLOCK_VALUES);
    for row in 0..layout.n_rows {
        for block in 0..layout.blocks_per_row {
            let tile = &tiles[layout.tile_range(row, block)];
            quants.extend((0..BLOCK_VALUES).map(|e| q6_from_tile(tile, row % TILE_ROWS, e)));
        }
    }
    quants
}

/// Splits a slice into whole `N`-element chunks. A remainder is a test bug, so it
/// fails loudly instead of being dropped the way `chunks_exact` would drop it.
fn whole_chunks<T, const N: usize>(values: &[T]) -> &[[T; N]] {
    let (chunks, rest) = values.as_chunks::<N>();
    assert!(
        rest.is_empty(),
        "{} values do not split into chunks of {N}",
        values.len()
    );
    chunks
}

/// The reference 6-bit quants of every block, in source order.
pub(super) fn q6_quants_of_source(source: &[u8]) -> Vec<u8> {
    whole_chunks::<u8, Q6K_BLOCK_BYTES>(source)
        .iter()
        .flat_map(|block| (0..BLOCK_VALUES).map(move |i| q6_reference(block, i)))
        .collect()
}

/// Every stored scale word must equal `RNE_f16(d * scale)` computed from the
/// source block. The product is exact in f32 (an fp16 significand times an
/// 8-bit integer), so a single rounding decides the bits.
pub(super) fn assert_scale_words(tiles: &[u8], source: &[u8], layout: Layout) {
    let blocks = whole_chunks::<u8, Q6K_BLOCK_BYTES>(source);
    for (index, block) in blocks.iter().enumerate() {
        let (row, b) = (index / layout.blocks_per_row, index % layout.blocks_per_row);
        let tile = &tiles[layout.tile_range(row, b)];
        let d = f16_value(u16::from_le_bytes([block[208], block[209]]));
        for group in 0..16 {
            let scale = f32::from(block[192 + group].cast_signed());
            let want = f32_to_f16_rne(d * scale);
            let got = stored_scale_bits(tile, row % TILE_ROWS, 16 * group);
            assert_eq!(got, want, "scale word of row {row} block {b} group {group}");
        }
    }
}

/// Info only, not a gate: the largest |decoded - cpu| / max(|cpu|, 1), where
/// decoded is `(quant - 32) * fp16 stored scale` and cpu is the dequantized source.
fn q6_decode_error(tiles: &[u8], layout: Layout, cpu: &[f32]) -> f64 {
    let mut worst = 0.0f64;
    for (index, value) in cpu.iter().enumerate() {
        let (block, element) = (index / BLOCK_VALUES, index % BLOCK_VALUES);
        let (row, b) = (block / layout.blocks_per_row, block % layout.blocks_per_row);
        let tile = &tiles[layout.tile_range(row, b)];
        let quant = f32::from(q6_from_tile(tile, row % TILE_ROWS, element)) - 32.0;
        let scale = f16_value(stored_scale_bits(tile, row % TILE_ROWS, element));
        let error = f64::from((quant * scale - *value).abs());
        worst = worst.max(error / f64::from(value.abs().max(1.0)));
    }
    worst
}

pub(super) fn assert_same_bytes(label: &str, device: &[u8], expected: &[u8]) {
    assert_eq!(device.len(), expected.len(), "{label}: length");
    if let Some(at) = device
        .iter()
        .zip(expected)
        .position(|(got, want)| got != want)
    {
        panic!(
            "{label}: byte {at} is {:#04x}, expected {:#04x}",
            device[at], expected[at]
        );
    }
}

fn assert_bits_equal(label: &str, device: &[f32], expected: &[f32]) {
    assert_eq!(device.len(), expected.len(), "{label}: length");
    let mismatch = device
        .iter()
        .zip(expected)
        .position(|(got, want)| got.to_bits() != want.to_bits());
    if let Some(at) = mismatch {
        panic!(
            "{label}: value {at} is {:?}, expected {:?}",
            device[at], expected[at]
        );
    }
}

/// The tile bytes of a weight, downloaded through the context.
pub(super) fn download_tiles(projector: &CudaProjector, weight: &TiledWeight) -> Vec<u8> {
    let mut tiles = vec![0u8; weight.buffer().bytes()];
    projector
        .context
        .download(weight.buffer(), &mut tiles)
        .expect("download the tiles");
    tiles
}

pub(super) fn weight_of<'a>(loaded: &'a CudaWeights, name: &str) -> &'a CudaWeight {
    loaded
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not loaded"))
}

pub(super) fn tiled_of<'a>(loaded: &'a CudaWeights, name: &str) -> &'a TiledWeight {
    match weight_of(loaded, name) {
        CudaWeight::Tiled { weight, .. } => weight,
        CudaWeight::Raw { ty, .. } => panic!("{name} loaded raw as {ty:?}, not tiled"),
    }
}

/// Dense reference, one token at a time through the dense projector.
fn dense_reference(weight: &DenseWeight, x: &[f32]) -> Vec<f32> {
    let mut reference = DenseProjector::default();
    whole_chunks::<f32, K>(x)
        .iter()
        .flat_map(|token| {
            reference
                .linear(weight, token)
                .expect("dense reference projection")
        })
        .collect()
}

/// `sum((y - ref)^2) / sum(ref^2)`, summed in f64 so the sums carry no f32 error.
pub(super) fn normalized_mse(device: &[f32], reference: &[f32]) -> f64 {
    let error: f64 = device
        .iter()
        .zip(reference)
        .map(|(y, r)| (f64::from(*y) - f64::from(*r)).powi(2))
        .sum();
    let energy: f64 = reference.iter().map(|r| f64::from(*r).powi(2)).sum();
    assert!(energy > 0.0, "the reference token has no energy");
    error / energy
}

pub(super) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .expect("a token has outputs")
}

/// Per-token gate: the argmax must agree and the NMSE must stay within NMSE_BOUND.
/// The worst token is printed next to the bound.
fn assert_tokens_match(label: &str, device: &[f32], reference: &[f32]) {
    assert_eq!(device.len(), reference.len(), "{label}: output length");
    let mut worst = 0.0f64;
    let tokens = whole_chunks::<f32, N_ROWS>(device)
        .iter()
        .zip(whole_chunks::<f32, N_ROWS>(reference));
    for (token, (got, want)) in tokens.enumerate() {
        let nmse = normalized_mse(got, want);
        assert_eq!(argmax(got), argmax(want), "{label}: token {token} argmax");
        assert!(
            nmse <= NMSE_BOUND,
            "{label}: token {token} NMSE {nmse:e} exceeds the bound {NMSE_BOUND:e}"
        );
        worst = worst.max(nmse);
    }
    let count = device.len() / N_ROWS;
    println!("T3 {label}: max NMSE {worst:.3e} over {count} tokens (bound {NMSE_BOUND:e})");
}

/// One token goes through `Projector::linear`. Several tokens go through the
/// context, because the trait method is single-token by contract. Every caller
/// loads its weight as `w`.
fn device_linear(projector: &mut CudaProjector, loaded: &CudaWeights, x: &[f32]) -> Vec<f32> {
    if x.len() == K {
        projector
            .linear(weight_of(loaded, "w"), x)
            .expect("device linear, one token")
    } else {
        projector
            .context
            .linear(tiled_of(loaded, "w"), x)
            .expect("device linear, several tokens")
    }
}

/// Linear on the device against the dense projector on the dequantized source,
/// for one token and for seventeen.
fn parity_case(test: &str, ty: GgufType, make_blocks: fn(&mut Rng, usize) -> Vec<u8>) {
    if !device_present(test) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = make_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let fixture = Fixture::new(ty, source);
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the weight");
    for tokens in [1, 17] {
        let x: Vec<f32> = (0..tokens * K).map(|_| rng.unit()).collect();
        let device = device_linear(&mut projector, &loaded, &x);
        let reference = dense_reference(&fixture.dense, &x);
        assert_tokens_match(&format!("{ty:?} tokens={tokens}"), &device, &reference);
    }
}

#[test]
fn q4k_tiles_decode_to_source_bytes_and_dequantize_bit_exact() {
    const NAME: &str = "q4k_tiles_decode_to_source_bytes_and_dequantize_bit_exact";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let (_dir, weights) = open_fixture(&[weight_entry(
        "w",
        GgufType::Q4K,
        (N_ROWS, K),
        source.clone(),
    )]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&weights, &["w"], LoadOptions::default())
        .expect("load the Q4_K weight");
    let weight = tiled_of(&loaded, "w");
    let layout = Layout::q4k(N_ROWS, BLOCKS_PER_ROW);
    let tiles = download_tiles(&projector, weight);
    assert_eq!(weight.n_rows(), N_ROWS);
    assert_eq!(tiles.len(), layout.tile_total());
    let decoded = decode_q4k(&tiles, layout);
    assert_same_bytes("q4k decoded bytes", &decoded, &source);
    let from_device = dequantize(GgufType::Q4K, &decoded).expect("dequantize the decoded tiles");
    let from_source = dequantize(GgufType::Q4K, &source).expect("dequantize the source");
    assert_bits_equal("q4k dequantized values", &from_device, &from_source);
    println!(
        "T1 q4k: {} source bytes equal; {} dequantized values bit-equal",
        source.len(),
        from_source.len()
    );
}

#[test]
fn q6k_tiles_hold_every_quant_and_fp16_scale_word() {
    const NAME: &str = "q6k_tiles_hold_every_quant_and_fp16_scale_word";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = q6k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let fixture = Fixture::new(GgufType::Q6K, source.clone());
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the Q6_K weight");
    let weight = tiled_of(&loaded, "w");
    let layout = Layout::q6k(N_ROWS, BLOCKS_PER_ROW);
    let tiles = download_tiles(&projector, weight);
    assert_eq!(tiles.len(), layout.tile_total());
    let device_quants = q6_quants_from_tiles(&tiles, layout);
    assert_same_bytes("q6k quants", &device_quants, &q6_quants_of_source(&source));
    assert_scale_words(&tiles, &source, layout);
    let cpu = dequantize(GgufType::Q6K, &source).expect("dequantize the Q6_K source");
    let error = q6_decode_error(&tiles, layout, &cpu);
    let reference = 2f64.powi(-11) * 32.0;
    println!(
        "T1 q6k info: max |decode-cpu|/max(|cpu|,1) = {error:.6e} (reference 2^-11*32 = {reference:.6e}; not a gate)"
    );
    println!(
        "T1 q6k: {} quants and {} scale words match the source",
        device_quants.len(),
        source.len() / Q6K_BLOCK_BYTES * 16
    );
}

#[test]
fn q8_0_tiles_decode_to_source_bytes_and_dequantize_bit_exact() {
    const NAME: &str = "q8_0_tiles_decode_to_source_bytes_and_dequantize_bit_exact";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = q8_0_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let fixture = Fixture::new(GgufType::Q80, source.clone());
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the Q8_0 weight");
    let weight = tiled_of(&loaded, "w");
    let layout = Layout::q8_0(N_ROWS, BLOCKS_PER_ROW);
    let tiles = download_tiles(&projector, weight);
    assert_eq!(weight.n_rows(), N_ROWS);
    assert_eq!(tiles.len(), layout.tile_total());
    let decoded = decode_q8_0(&tiles, layout);
    assert_same_bytes("q8_0 decoded bytes", &decoded, &source);
    let from_device = dequantize(GgufType::Q80, &decoded).expect("dequantize the decoded tiles");
    let from_source = dequantize(GgufType::Q80, &source).expect("dequantize the source");
    assert_bits_equal("q8_0 dequantized values", &from_device, &from_source);
    assert_eq!(projector.live_bytes() - before, tiles.len());
    drop(loaded);
    assert_eq!(
        projector.live_bytes(),
        before,
        "the dropped weight kept device memory"
    );
    println!(
        "T1 q8_0: {} source bytes equal; {} dequantized values bit-equal",
        source.len(),
        from_source.len()
    );
}

#[test]
fn q4k_tiles_spanning_two_upload_chunks_decode_bit_exactly() {
    const NAME: &str = "q4k_tiles_spanning_two_upload_chunks_decode_bit_exactly";
    if !device_present(NAME) {
        return;
    }
    let layout = Layout::q4k(CHUNK_ROWS, CHUNK_BLOCKS_PER_ROW);
    let groups = CHUNK_ROWS / TILE_ROWS;
    let per_chunk = CHUNK_BYTES / (layout.tile_bytes * layout.blocks_per_row);
    assert!(
        layout.tile_total() > CHUNK_BYTES,
        "tiles of {} bytes must exceed one chunk of {CHUNK_BYTES}",
        layout.tile_total()
    );
    assert_eq!(
        groups.div_ceil(per_chunk),
        2,
        "the tensor must span two chunks"
    );
    let mut rng = Rng(SEED ^ 1);
    let source = q4k_blocks(&mut rng, layout.block_count());
    let (_dir, weights) = open_with(
        &[weight_entry(
            "big",
            GgufType::Q4K,
            (CHUNK_ROWS, CHUNK_K),
            source.clone(),
        )],
        LARGE_LIMITS,
    );
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&weights, &["big"], LoadOptions::default())
        .expect("load the multi-chunk Q4_K weight");
    let weight = tiled_of(&loaded, "big");
    assert_eq!(weight.n_rows(), CHUNK_ROWS);
    let tiles = download_tiles(&projector, weight);
    assert_eq!(tiles.len(), layout.tile_total());
    assert_same_bytes("multi-chunk q4k", &decode_q4k(&tiles, layout), &source);
    println!(
        "T1 q4k multi-chunk: {} tile bytes in 2 upload chunks ({per_chunk} and {}), source bytes equal",
        tiles.len(),
        groups - per_chunk
    );
}

#[test]
fn q4k_linear_matches_the_dense_projector_for_one_and_seventeen_tokens() {
    parity_case(
        "q4k_linear_matches_the_dense_projector_for_one_and_seventeen_tokens",
        GgufType::Q4K,
        q4k_blocks,
    );
}

#[test]
fn q6k_linear_matches_the_dense_projector_for_one_and_seventeen_tokens() {
    parity_case(
        "q6k_linear_matches_the_dense_projector_for_one_and_seventeen_tokens",
        GgufType::Q6K,
        q6k_blocks,
    );
}

#[test]
fn q8_0_linear_matches_the_dense_projector_for_one_and_seventeen_tokens() {
    parity_case(
        "q8_0_linear_matches_the_dense_projector_for_one_and_seventeen_tokens",
        GgufType::Q80,
        q8_0_blocks,
    );
}

/// Q8_0 linear at `(n_rows, k)` shapes of the surveyed Q8_0 tensors, for one
/// token (mma884) and seventeen (gemm DQ8), against the dense projector. The
/// argmax gate applies from 64 outputs (REPLY-20 item 1).
fn q8_0_shape_case(test: &str, shapes: &[(usize, usize)], limits: GgufLimits) {
    if !device_present(test) {
        return;
    }
    let mut rng = Rng(SEED);
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    for &(n_rows, k) in shapes {
        let source = q8_0_blocks(&mut rng, n_rows * k / BLOCK_VALUES);
        let values = dequantize(GgufType::Q80, &source).expect("dequantize the source");
        let dense = DenseWeight::matrix(n_rows, k, values);
        let entry = weight_entry("w", GgufType::Q80, (n_rows, k), source);
        let (_dir, weights) = open_with(&[entry], limits);
        let loaded = projector
            .load(&weights, &["w"], LoadOptions::default())
            .expect("load the weight");
        for tokens in [1, 17] {
            let x: Vec<f32> = (0..tokens * k).map(|_| rng.unit()).collect();
            let device = if tokens == 1 {
                projector.linear(weight_of(&loaded, "w"), &x)
            } else {
                projector.context.linear(tiled_of(&loaded, "w"), &x)
            }
            .expect("device linear");
            assert_eq!(device.len(), tokens * n_rows, "output length");
            let label = format!("q8_0 {n_rows}x{k} tokens={tokens}");
            let worst = gate_shape_tokens(&label, &dense, &device, &x);
            println!("T3 {label}: max NMSE {worst:.3e} (bound {NMSE_BOUND:e})");
        }
    }
}

/// The per-token NMSE and argmax gates of [`q8_0_shape_case`]; returns the
/// worst NMSE.
fn gate_shape_tokens(label: &str, dense: &DenseWeight, device: &[f32], x: &[f32]) -> f64 {
    let (n_rows, k) = (dense.rows, dense.cols);
    let mut reference = DenseProjector::default();
    let mut worst = 0.0f64;
    for (token, (got, input)) in device
        .chunks_exact(n_rows)
        .zip(x.chunks_exact(k))
        .enumerate()
    {
        let want = reference
            .linear(dense, input)
            .expect("dense reference projection");
        let nmse = normalized_mse(got, &want);
        assert!(
            nmse <= NMSE_BOUND,
            "{label}: token {token} NMSE {nmse:e} exceeds the bound {NMSE_BOUND:e}"
        );
        if n_rows >= 64 {
            assert_eq!(argmax(got), argmax(&want), "{label}: token {token} argmax");
        }
        worst = worst.max(nmse);
    }
    worst
}

/// Reduced shapes for the default native-cuda run: `k / 256` = 2, 8 and 20
/// blocks per row, a 32-row GDN gate, and 48 rows at 20 blocks, the one
/// split-K case (two K halves added with atomicAdd).
#[test]
fn q8_0_linear_matches_the_dense_projector_at_reduced_surveyed_shapes() {
    q8_0_shape_case(
        "q8_0_linear_matches_the_dense_projector_at_reduced_surveyed_shapes",
        &[(2048, 512), (512, 2048), (32, 2048), (48, 5120), (64, 5120)],
        LIMITS,
    );
}

/// Full-size surveyed shapes (50 to 60 MB of Q8_0 each). Ignored by default;
/// the packet 4 gates run it on the 3090 and the Xavier with
/// `cargo test --features native-cuda -- --ignored q8_0_linear_matches_the_dense_projector_at_full_surveyed_shapes`.
#[test]
#[ignore = "large Q8_0 shapes, run by the packet 4 device gates"]
fn q8_0_linear_matches_the_dense_projector_at_full_surveyed_shapes() {
    q8_0_shape_case(
        "q8_0_linear_matches_the_dense_projector_at_full_surveyed_shapes",
        &[(8192, 2048), (2048, 4096), (10240, 5120), (12288, 5120)],
        LARGE_LIMITS,
    );
}

#[test]
fn load_rejects_a_missing_tensor() {
    const NAME: &str = "load_rejects_a_missing_tensor";
    if !device_present(NAME) {
        return;
    }
    let fixture = Fixture::new(
        GgufType::Q4K,
        q4k_blocks(&mut Rng(SEED), N_ROWS * BLOCKS_PER_ROW),
    );
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let result = projector.load(&fixture.weights, &["w", "absent"], LoadOptions::default());
    assert_error(result, "Checkpoint", "tensor absent is not in the file");
}

#[test]
fn load_rejects_a_repeated_tensor_name() {
    const NAME: &str = "load_rejects_a_repeated_tensor_name";
    if !device_present(NAME) {
        return;
    }
    let fixture = Fixture::new(
        GgufType::Q4K,
        q4k_blocks(&mut Rng(SEED), N_ROWS * BLOCKS_PER_ROW),
    );
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let result = projector.load(&fixture.weights, &["w", "w"], LoadOptions::default());
    assert_error(result, "BadRequest", "cuda load names tensor w twice");
}

#[test]
fn headroom_beyond_free_memory_fails_admission_and_the_backend_still_loads() {
    const NAME: &str = "headroom_beyond_free_memory_fails_admission_and_the_backend_still_loads";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let fixture = Fixture::new(GgufType::Q4K, q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW));
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let reserve = Reserve {
        headroom: u64::MAX / 2,
        ..Reserve::default()
    };
    let refused = LoadOptions {
        reserve,
        ..LoadOptions::default()
    };
    let result = projector.load(&fixture.weights, &["w"], refused);
    assert_error(result, "MemoryPlan", "cuda admission");
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load after a refused admission");
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    let device = projector
        .linear(weight_of(&loaded, "w"), &x)
        .expect("linear after a refused admission");
    assert_tokens_match(
        "q4k after a refused admission",
        &device,
        &dense_reference(&fixture.dense, &x),
    );
}

/// The F32 payload of the tensor `norm` of the fixtures: distinct values, so a shifted or
/// zeroed upload differs.
fn norm_bytes() -> Vec<u8> {
    (0..N_ROWS * K)
        .flat_map(|i| (i as f32 / 64.0).to_le_bytes())
        .collect()
}

/// Requires `name` to be a raw F32 tensor whose device bytes equal `want`.
pub(super) fn assert_raw_f32_bytes(
    projector: &CudaProjector,
    loaded: &CudaWeights,
    name: &str,
    want: &[u8],
) {
    let CudaWeight::Raw {
        ty: GgufType::F32,
        buffer,
        ..
    } = weight_of(loaded, name)
    else {
        panic!("{name} did not load as a raw F32 tensor");
    };
    let mut device = vec![0u8; buffer.bytes()];
    projector
        .context
        .download(buffer, &mut device)
        .expect("download the raw tensor");
    assert_same_bytes(name, &device, want);
}

#[test]
fn raw_tensors_load_as_raw_and_linear_rejects_them_without_breaking_the_backend() {
    const NAME: &str =
        "raw_tensors_load_as_raw_and_linear_rejects_them_without_breaking_the_backend";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let dense = DenseWeight::matrix(
        N_ROWS,
        K,
        dequantize(GgufType::Q4K, &source).expect("dequantize"),
    );
    let norm = norm_bytes();
    let half: Vec<u8> = (0..N_ROWS * K)
        .flat_map(|_| 0x3c00u16.to_le_bytes())
        .collect();
    let (_dir, weights) = open_fixture(&[
        weight_entry("w", GgufType::Q4K, (N_ROWS, K), source),
        weight_entry("norm", GgufType::F32, (N_ROWS, K), norm.clone()),
        weight_entry("half", GgufType::F16, (N_ROWS, K), half),
    ]);
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&weights, &["w", "norm", "half"], LoadOptions::default())
        .expect("load the mixed weights");
    assert_raw_f32_bytes(&projector, &loaded, "norm", &norm);
    assert!(matches!(
        loaded.get("half"),
        Some(CudaWeight::Raw {
            ty: GgufType::F16,
            ..
        })
    ));
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    assert_error(
        projector.linear(weight_of(&loaded, "norm"), &x),
        "Checkpoint",
        "tensor norm has type F32",
    );
    assert_error(
        projector.linear(weight_of(&loaded, "half"), &x),
        "Checkpoint",
        "tensor half has type F16",
    );
    let device = projector
        .linear(weight_of(&loaded, "w"), &x)
        .expect("tiled linear after the raw rejections");
    assert_tokens_match(
        "q4k after raw rejections",
        &device,
        &dense_reference(&dense, &x),
    );
}

#[test]
fn expert_linear_on_a_matrix_and_rows_are_bad_requests_naming_the_op() {
    const NAME: &str = "expert_linear_on_a_matrix_and_rows_are_bad_requests_naming_the_op";
    if !device_present(NAME) {
        return;
    }
    let fixture = Fixture::new(
        GgufType::Q4K,
        q4k_blocks(&mut Rng(SEED), N_ROWS * BLOCKS_PER_ROW),
    );
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the weight");
    let x = [0.0f32; K];
    assert_error(
        projector.expert_linear(weight_of(&loaded, "w"), 0, &x),
        "BadRequest",
        "tensor w is not a 3-D expert stack",
    );
    assert_error(
        projector.rows(weight_of(&loaded, "w"), &[0]),
        "BadRequest",
        "cuda rows is not implemented",
    );
}

const EXPERTS: usize = 4;

/// A stack of [`EXPERTS`] experts of [`N_ROWS`] rows, written as tensor `w`,
/// with the dense matrix of every expert's slice of the dequantized source.
struct StackFixture {
    _dir: tempfile::TempDir,
    weights: GgufWeights,
    experts: Vec<DenseWeight>,
}

impl StackFixture {
    fn new(ty: GgufType, source: Vec<u8>) -> Self {
        let values = dequantize(ty, &source).expect("dequantize the source");
        let experts = values
            .as_chunks::<{ N_ROWS * K }>()
            .0
            .iter()
            .map(|slice| DenseWeight::matrix(N_ROWS, K, slice.to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(experts.len(), EXPERTS, "one dense matrix per expert");
        let (dir, weights) = open_fixture(&[stack_entry("w", ty, (EXPERTS, N_ROWS, K), source)]);
        Self {
            _dir: dir,
            weights,
            experts,
        }
    }
}

/// The device output of expert `expert` of the stack `w` for the tokens in `x`.
/// One token goes through `Projector::expert_linear`; several go through the
/// context's row-range matmul, since the trait method is single-token by
/// contract.
fn expert_output(
    projector: &mut CudaProjector,
    loaded: &CudaWeights,
    expert: usize,
    x: &[f32],
) -> Vec<f32> {
    if x.len() == K {
        projector
            .expert_linear(weight_of(loaded, "w"), expert as u32, x)
            .expect("expert_linear, one token")
    } else {
        projector
            .context
            .linear_rows(tiled_of(loaded, "w"), expert * N_ROWS, N_ROWS, x)
            .expect("expert rows, several tokens")
    }
}

/// Every expert of a 3-D stack against the dense projector on its own slice, for
/// one token and seventeen (see [`expert_output`]). The live device bytes must
/// return to their start when the load drops.
fn expert_case(test: &str, ty: GgufType, make_blocks: fn(&mut Rng, usize) -> Vec<u8>) {
    if !device_present(test) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = make_blocks(&mut rng, EXPERTS * N_ROWS * BLOCKS_PER_ROW);
    let fixture = StackFixture::new(ty, source);
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the stack");
    assert_eq!(projector.live_bytes() - before, loaded.device_bytes());
    for (expert, dense) in fixture.experts.iter().enumerate() {
        for tokens in [1, 17] {
            let x: Vec<f32> = (0..tokens * K).map(|_| rng.unit()).collect();
            let device = expert_output(&mut projector, &loaded, expert, &x);
            assert_tokens_match(
                &format!("{ty:?} expert={expert} tokens={tokens}"),
                &device,
                &dense_reference(dense, &x),
            );
        }
    }
    drop(loaded);
    assert_eq!(
        projector.live_bytes(),
        before,
        "the stack leaked device bytes"
    );
}

#[test]
fn q4k_expert_linear_matches_the_dense_projector_per_expert() {
    expert_case(
        "q4k_expert_linear_matches_the_dense_projector_per_expert",
        GgufType::Q4K,
        q4k_blocks,
    );
}

/// A Q4_K stack uploaded in several chunks, with chunk boundaries inside an
/// expert, against the dense projector for the first, a middle and the last
/// expert (one token and seventeen). Neither the chunk-size tests (one matrix)
/// nor the per-expert tests (one chunk) cover a stack that spans chunks. A
/// chunk written at the wrong device offset puts one expert's tiles in another's
/// place. The live device bytes must return to their start when the load drops.
#[test]
fn a_q4k_expert_stack_uploaded_in_chunks_matches_the_dense_projector() {
    const NAME: &str = "a_q4k_expert_stack_uploaded_in_chunks_matches_the_dense_projector";
    // 3 groups per chunk: the 4 experts of 8 groups upload as 11 chunks, and the
    // boundaries at groups 3, 6, 9, 12, ... fall inside every expert.
    const GROUPS_PER_CHUNK: usize = 3;
    const GROUPS_PER_EXPERT: usize = N_ROWS / TILE_ROWS;
    const _: () = assert!(!GROUPS_PER_EXPERT.is_multiple_of(GROUPS_PER_CHUNK));
    const _: () = assert!(EXPERTS * GROUPS_PER_EXPERT >= 3 * GROUPS_PER_CHUNK);
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED ^ 3);
    let source = q4k_blocks(&mut rng, EXPERTS * N_ROWS * BLOCKS_PER_ROW);
    let fixture = StackFixture::new(GgufType::Q4K, source);
    let chunk_limit = GROUPS_PER_CHUNK * Q4K_TILE_BYTES * BLOCKS_PER_ROW;
    let mut projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_chunk_limit(chunk_limit);
    let before = projector.live_bytes();
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the chunked stack");
    assert_eq!(projector.live_bytes() - before, loaded.device_bytes());
    for expert in [0, EXPERTS / 2, EXPERTS - 1] {
        for tokens in [1, 17] {
            let x: Vec<f32> = (0..tokens * K).map(|_| rng.unit()).collect();
            let device = expert_output(&mut projector, &loaded, expert, &x);
            assert_tokens_match(
                &format!("chunked Q4K expert={expert} tokens={tokens}"),
                &device,
                &dense_reference(&fixture.experts[expert], &x),
            );
        }
    }
    drop(loaded);
    assert_eq!(
        projector.live_bytes(),
        before,
        "the chunked stack leaked device bytes"
    );
}

#[test]
fn q6k_expert_linear_matches_the_dense_projector_per_expert() {
    expert_case(
        "q6k_expert_linear_matches_the_dense_projector_per_expert",
        GgufType::Q6K,
        q6k_blocks,
    );
}

#[test]
fn q8_0_expert_linear_matches_the_dense_projector_per_expert() {
    expert_case(
        "q8_0_expert_linear_matches_the_dense_projector_per_expert",
        GgufType::Q80,
        q8_0_blocks,
    );
}

#[test]
fn expert_linear_rejects_bad_requests_and_the_backend_stays_usable() {
    const NAME: &str = "expert_linear_rejects_bad_requests_and_the_backend_stays_usable";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let stack = StackFixture::new(
        GgufType::Q4K,
        q4k_blocks(&mut rng, EXPERTS * N_ROWS * BLOCKS_PER_ROW),
    );
    let flat = Fixture::new(GgufType::Q4K, q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW));
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let stack_load = projector
        .load(&stack.weights, &["w"], LoadOptions::default())
        .expect("load the stack");
    let flat_load = projector
        .load(&flat.weights, &["w"], LoadOptions::default())
        .expect("load the matrix");
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    let stack_weight = weight_of(&stack_load, "w");
    for expert in [EXPERTS as u32, u32::MAX] {
        assert_error(
            projector.expert_linear(stack_weight, expert, &x),
            "BadRequest",
            &format!("expert {expert} is outside the {EXPERTS} experts of tensor w"),
        );
    }
    assert_error(
        projector.expert_linear(stack_weight, 0, &x[1..]),
        "BadRequest",
        "cuda expert_linear input has 511 values, expected 512",
    );
    assert_error(
        projector.expert_linear(weight_of(&flat_load, "w"), 0, &x),
        "BadRequest",
        "tensor w is not a 3-D expert stack",
    );
    let last = EXPERTS - 1;
    let device = projector
        .expert_linear(stack_weight, last as u32, &x)
        .expect("expert_linear after rejected requests");
    assert_tokens_match(
        "q4k last expert after rejected requests",
        &device,
        &dense_reference(&stack.experts[last], &x),
    );
}

#[test]
fn linear_rejects_a_3_d_expert_stack_naming_the_tensor_and_the_backend_stays_usable() {
    const NAME: &str =
        "linear_rejects_a_3_d_expert_stack_naming_the_tensor_and_the_backend_stays_usable";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let stack = StackFixture::new(
        GgufType::Q4K,
        q4k_blocks(&mut rng, EXPERTS * N_ROWS * BLOCKS_PER_ROW),
    );
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&stack.weights, &["w"], LoadOptions::default())
        .expect("load the stack");
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    assert_error(
        projector.linear(weight_of(&loaded, "w"), &x),
        "BadRequest",
        "tensor w is a 3-D expert stack; use expert_linear",
    );
    let device = projector
        .expert_linear(weight_of(&loaded, "w"), 0, &x)
        .expect("expert_linear after the rejected linear");
    assert_tokens_match(
        "q4k expert 0 after a rejected linear",
        &device,
        &dense_reference(&stack.experts[0], &x),
    );
}

#[test]
fn expert_linear_on_a_raw_tensor_is_a_checkpoint_error_naming_the_op_and_the_tensor() {
    const NAME: &str =
        "expert_linear_on_a_raw_tensor_is_a_checkpoint_error_naming_the_op_and_the_tensor";
    if !device_present(NAME) {
        return;
    }
    let (_dir, weights) = open_fixture(&[
        weight_entry("norm", GgufType::F32, (N_ROWS, K), vec![0; N_ROWS * K * 4]),
        stack_entry(
            "raw_stack",
            GgufType::F32,
            (EXPERTS, N_ROWS, K),
            vec![0; EXPERTS * N_ROWS * K * 4],
        ),
    ]);
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&weights, &["norm", "raw_stack"], LoadOptions::default())
        .expect("load the raw tensors");
    let x = [0.0f32; K];
    for name in ["norm", "raw_stack"] {
        assert_error(
            projector.expert_linear(weight_of(&loaded, name), 0, &x),
            "Checkpoint",
            &format!(
                "tensor {name} has type F32; cuda expert_linear executes Q4_K, Q6_K and Q8_0 only"
            ),
        );
    }
}

#[test]
fn planned_rejects_a_tiled_stack_whose_experts_miss_the_tile_groups_on_every_host() {
    // 8 experts of 4 rows: 32 rows in all fit the tiles, but an expert is half a group.
    let mut rng = Rng(SEED);
    for (ty, make_blocks) in [
        (GgufType::Q4K, q4k_blocks as fn(&mut Rng, usize) -> Vec<u8>),
        (GgufType::Q6K, q6k_blocks),
        (GgufType::Q80, q8_0_blocks),
    ] {
        let (_dir, weights) = open_fixture(&[stack_entry(
            "w",
            ty,
            (8, 4, K),
            make_blocks(&mut rng, 8 * 4 * BLOCKS_PER_ROW),
        )]);
        assert_error(
            planned(&weights, "w"),
            "Checkpoint",
            "tensor w stacks experts of 4 rows, which do not start on 8-row tile groups",
        );
    }
}

#[test]
fn planned_accepts_whole_group_stacks_and_raw_stacks_of_any_expert_height() {
    let mut rng = Rng(SEED);
    let (_dir, weights) = open_fixture(&[
        stack_entry(
            "tiled",
            GgufType::Q4K,
            (2, 8, K),
            q4k_blocks(&mut rng, 2 * 8 * BLOCKS_PER_ROW),
        ),
        stack_entry("raw", GgufType::F32, (8, 4, K), vec![0; 8 * 4 * K * 4]),
    ]);
    let (_, bytes) = planned(&weights, "tiled").expect("two whole groups");
    // 2 groups of 2 tiles of 72 vectors of 16 bytes.
    assert_eq!(bytes, 2 * 2 * 72 * 16);
    let (_, bytes) = planned(&weights, "raw").expect("a raw stack is copied as read");
    assert_eq!(bytes, 8 * 4 * K as u64 * 4);
}

#[test]
fn a_stack_of_experts_that_miss_the_tile_groups_fails_the_load() {
    const NAME: &str = "a_stack_of_experts_that_miss_the_tile_groups_fails_the_load";
    if !device_present(NAME) {
        return;
    }
    // 8 experts of 4 rows: 32 rows in all fit the tiles, but an expert is half a group.
    let mut rng = Rng(SEED);
    let (_dir, weights) = open_fixture(&[stack_entry(
        "w",
        GgufType::Q4K,
        (8, 4, K),
        q4k_blocks(&mut rng, 8 * 4 * BLOCKS_PER_ROW),
    )]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    assert_error(
        projector.load(&weights, &["w"], LoadOptions::default()),
        "Checkpoint",
        "tensor w stacks experts of 4 rows",
    );
    assert_eq!(projector.live_bytes(), before);
}

#[test]
fn linear_rejects_a_wrong_input_length_before_the_device_call() {
    const NAME: &str = "linear_rejects_a_wrong_input_length_before_the_device_call";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let fixture = Fixture::new(GgufType::Q4K, q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW));
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load the weight");
    let short = [0.0f32; K - 1];
    assert_error(
        projector.linear(weight_of(&loaded, "w"), &short),
        "BadRequest",
        "cuda linear input has 511 values, expected 512",
    );
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    let device = projector
        .linear(weight_of(&loaded, "w"), &x)
        .expect("linear after a rejected length");
    assert_tokens_match(
        "q4k after a rejected length",
        &device,
        &dense_reference(&fixture.dense, &x),
    );
}

/// Compile-time probes: `probe` resolves only when `T` is neither `Send` nor
/// `Sync`. For a `Send` type both impls apply and the call is ambiguous.
trait AmbiguousIfSend<A> {
    fn probe() {}
}
impl<T: ?Sized> AmbiguousIfSend<()> for T {}
impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}

trait AmbiguousIfSync<A> {
    fn probe() {}
}
impl<T: ?Sized> AmbiguousIfSync<()> for T {}
impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

/// The context, its buffers and the projector hold a thread-confined lease, so
/// the compiler must reject moving or sharing them across threads. Adding
/// `Send` or `Sync` to any of them breaks this test's build.
#[test]
fn cuda_handles_are_neither_send_nor_sync() {
    fn confined<T>() {
        <T as AmbiguousIfSend<_>>::probe();
        <T as AmbiguousIfSync<_>>::probe();
    }
    confined::<CudaContext>();
    confined::<DeviceBuffer>();
    confined::<TiledWeight>();
    confined::<CudaWeights>();
    confined::<CudaProjector>();
}

/// 3072 rows of 8192 values: 14 MiB of tiles, a load large enough to span
/// several upload chunks before the failure.
const UNWIND_ROWS: usize = 3072;
const UNWIND_K: usize = 8192;

/// A read end of a pipe as a `File`. `posix_fadvise` on it fails with ESPIPE,
/// so a load that drops page cache fails right after its first upload.
fn pipe_file() -> File {
    let (reader, _writer) = std::io::pipe().expect("create a pipe");
    File::from(std::os::fd::OwnedFd::from(reader))
}

/// A Q4_K tensor `big` of [`UNWIND_ROWS`] rows, a small Q4_K tensor `w` and a small F32
/// tensor `norm`, in that file order.
fn big_and_small_fixture() -> Fixture {
    let mut rng = Rng(SEED);
    let big = q4k_blocks(&mut rng, UNWIND_ROWS * UNWIND_K / BLOCK_VALUES);
    let small = q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let dense = DenseWeight::matrix(
        N_ROWS,
        K,
        dequantize(GgufType::Q4K, &small).expect("dequantize the source"),
    );
    let (dir, weights) = open_fixture(&[
        weight_entry("big", GgufType::Q4K, (UNWIND_ROWS, UNWIND_K), big),
        weight_entry("w", GgufType::Q4K, (N_ROWS, K), small),
        weight_entry("norm", GgufType::F32, (N_ROWS, K), norm_bytes()),
    ]);
    Fixture {
        _dir: dir,
        weights,
        dense,
    }
}

/// A load, a failure, and the checks that the backend still works.
fn assert_loads_and_runs_after(projector: &mut CudaProjector, fixture: &Fixture, label: &str) {
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load after the failed load");
    let x: Vec<f32> = (0..K).map(|index| (index % 7) as f32 - 3.0).collect();
    let device = projector
        .linear(weight_of(&loaded, "w"), &x)
        .expect("linear after the failed load");
    assert_tokens_match(label, &device, &dense_reference(&fixture.dense, &x));
}

#[test]
fn the_live_byte_count_follows_an_upload_and_its_drop() {
    const NAME: &str = "the_live_byte_count_follows_an_upload_and_its_drop";
    if !device_present(NAME) {
        return;
    }
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let buffer = projector
        .context
        .upload(&[7u8; 4096])
        .expect("upload one buffer");
    assert_eq!(projector.live_bytes() - before, buffer.bytes());
    drop(buffer);
    assert_eq!(projector.live_bytes(), before);
}

#[test]
fn a_load_that_fails_after_an_upload_frees_it_and_stays_usable() {
    const NAME: &str = "a_load_that_fails_after_an_upload_frees_it_and_stays_usable";
    if !device_present(NAME) {
        return;
    }
    let fixture = big_and_small_fixture();
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    {
        let loaded = projector
            .load(&fixture.weights, &["big"], LoadOptions::default())
            .expect("load big alone");
        assert!(loaded.device_bytes() > 12 << 20);
        assert_eq!(
            projector.live_bytes() - before,
            loaded.device_bytes(),
            "the live count must hold exactly the loaded tensor"
        );
    }
    assert_eq!(projector.live_bytes(), before, "dropping the load freed it");
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    let text = error_text(projector.load(&fixture.weights, &["big", "w"], options));
    assert!(
        text.starts_with("Backend") && text.contains("cuda adapter"),
        "expected a non-terminal Backend cuda adapter error, got {text}"
    );
    assert!(
        text.contains("posix_fadvise") && text.contains("errno 29"),
        "the error must carry the status and ESPIPE, got {text}"
    );
    let after = projector.live_bytes();
    assert_eq!(
        after,
        before,
        "the failed load leaked {} device bytes",
        after.saturating_sub(before)
    );
    assert_loads_and_runs_after(&mut projector, &fixture, "q4k after a failed load");
}

#[test]
fn a_load_that_fails_after_a_raw_upload_frees_it_and_stays_usable() {
    const NAME: &str = "a_load_that_fails_after_a_raw_upload_frees_it_and_stays_usable";
    if !device_present(NAME) {
        return;
    }
    let fixture = big_and_small_fixture();
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    {
        let loaded = projector
            .load(&fixture.weights, &["norm"], LoadOptions::default())
            .expect("load norm alone");
        assert_eq!(loaded.device_bytes(), N_ROWS * K * 4, "raw F32 bytes");
        assert_eq!(
            projector.live_bytes() - before,
            loaded.device_bytes(),
            "the live count must hold exactly the raw tensor"
        );
    }
    assert_eq!(projector.live_bytes(), before, "dropping the load freed it");
    // The pipe fails the page-cache drop right after the first upload, which
    // is the raw tensor here: it is already owned by the load when that fails.
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    let text = error_text(projector.load(&fixture.weights, &["norm", "big", "w"], options));
    assert!(
        text.starts_with("Backend") && text.contains("posix_fadvise") && text.contains("errno 29"),
        "expected the ESPIPE page-cache failure, got {text}"
    );
    let after = projector.live_bytes();
    assert_eq!(
        after,
        before,
        "the failed load leaked {} device bytes",
        after.saturating_sub(before)
    );
    assert_loads_and_runs_after(&mut projector, &fixture, "q4k after a failed raw load");
}

#[test]
fn the_admission_sum_of_a_load_includes_its_raw_f32_bytes() {
    const NAME: &str = "the_admission_sum_of_a_load_includes_its_raw_f32_bytes";
    if !device_present(NAME) {
        return;
    }
    let fixture = big_and_small_fixture();
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let tiled = (UNWIND_ROWS / TILE_ROWS) * (UNWIND_K / BLOCK_VALUES) * Q4K_TILE_BYTES
        + (N_ROWS / TILE_ROWS) * BLOCKS_PER_ROW * Q4K_TILE_BYTES;
    let raw = N_ROWS * K * 4;
    let refused = LoadOptions {
        reserve: Reserve {
            headroom: u64::MAX / 2,
            ..Reserve::default()
        },
        ..LoadOptions::default()
    };
    let names = ["big", "w", "norm"];
    let text = error_text(projector.load(&fixture.weights, &names, refused));
    assert!(
        text.contains(&format!("(tensors {},", tiled + raw)),
        "the sum must be the tiles plus the raw bytes ({tiled} + {raw}), got {text}"
    );
    assert_eq!(projector.live_bytes(), before, "a refusal uploads nothing");
}

#[test]
fn a_q6k_scale_overflow_in_a_later_tensor_fails_before_any_upload() {
    const NAME: &str = "a_q6k_scale_overflow_in_a_later_tensor_fails_before_any_upload";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let big = q4k_blocks(&mut rng, UNWIND_ROWS * UNWIND_K / BLOCK_VALUES);
    let mut bad = q6k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    // d = 65504 (0x7bff) and a scale of 2 give a product of 131008, past fp16.
    for block in bad.as_chunks_mut::<Q6K_BLOCK_BYTES>().0 {
        block[192] = 2;
        block[208..].copy_from_slice(&0x7bffu16.to_le_bytes());
    }
    let (_dir, weights) = open_fixture(&[
        weight_entry("big", GgufType::Q4K, (UNWIND_ROWS, UNWIND_K), big),
        weight_entry("bad", GgufType::Q6K, (N_ROWS, K), bad),
    ]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    // After an upload of `big` the load would drop its page cache and fail with
    // ESPIPE. The scale error instead proves the check ran before the upload.
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    let result = projector.load(&weights, &["big", "bad"], options);
    assert_error(
        result,
        "Checkpoint",
        "tensor bad block 0: Q6_K scale product",
    );
    assert_eq!(
        projector.live_bytes(),
        before,
        "the rejected load allocated device memory"
    );
}

#[test]
fn a_gguf_with_a_q5k_tensor_fails_the_open_before_any_device_call() {
    const NAME: &str = "a_gguf_with_a_q5k_tensor_fails_the_open_before_any_device_call";
    if !device_present(NAME) {
        return;
    }
    // `GgufWeights::open` rejects Q5_K (no CPU reference decoder), so the plan's
    // own Q5_K rejection is not reachable through `load`; `plan` tests cover it.
    // The open names the tensor, and the device is untouched.
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path = dir.path().join("q5.gguf");
    let payload = vec![0u8; N_ROWS * BLOCKS_PER_ROW * 176];
    let entries = [weight_entry("q5", GgufType::Q5K, (N_ROWS, K), payload)];
    std::fs::write(&path, gguf_bytes(&entries)).expect("write the GGUF fixture");
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let result = GgufWeights::open(&path, LIMITS, None);
    assert_error(result, "Checkpoint", "tensor q5 has unsupported type Q5_K");
    assert_eq!(
        projector.live_bytes(),
        before,
        "the open touched the device"
    );
    let fixture = big_and_small_fixture();
    assert_loads_and_runs_after(&mut projector, &fixture, "q4k after a rejected q5_k");
}

/// Q8_0 blocks whose every `d` is `d_bits` and whose quant 0 is `q`, the rest
/// random, for `N_ROWS` rows of `K` values.
fn q8_0_blocks_with_scale(d_bits: u16, q: i8) -> Vec<u8> {
    let mut rng = Rng(SEED);
    let mut blocks = q8_0_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    for block in blocks.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0 {
        block[..2].copy_from_slice(&d_bits.to_le_bytes());
        block[2] = q.cast_unsigned();
        // Keep the other quants below |q| so quant 0 sets the block's maximum.
        for quant in &mut block[3..] {
            *quant = (quant.cast_signed() / 2).cast_unsigned();
        }
    }
    blocks
}

#[test]
fn a_q8_0_scale_product_of_exactly_the_fp16_maximum_loads_unchanged() {
    const NAME: &str = "a_q8_0_scale_product_of_exactly_the_fp16_maximum_loads_unchanged";
    if !device_present(NAME) {
        return;
    }
    // 0x5fff is 511.75, and 511.75 * 128 = 65504 (q = -128 counts as 128).
    let source = q8_0_blocks_with_scale(0x5fff, -128);
    let (_dir, weights) = open_fixture(&[weight_entry(
        "w",
        GgufType::Q80,
        (N_ROWS, K),
        source.clone(),
    )]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&weights, &["w"], LoadOptions::default())
        .expect("a product of exactly 65504 is within the fp16 range");
    let tiles = download_tiles(&projector, tiled_of(&loaded, "w"));
    let decoded = decode_q8_0(&tiles, Layout::q8_0(N_ROWS, BLOCKS_PER_ROW));
    assert_same_bytes("q8_0 boundary bytes", &decoded, &source);
}

#[test]
fn a_q8_0_scale_overflow_in_a_later_tensor_fails_before_any_upload() {
    const NAME: &str = "a_q8_0_scale_overflow_in_a_later_tensor_fails_before_any_upload";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let big = q4k_blocks(&mut rng, UNWIND_ROWS * UNWIND_K / BLOCK_VALUES);
    // 0x6000 is 512, and 512 * 128 = 65536: one fp16 step of d past 65504.
    let bad = q8_0_blocks_with_scale(0x6000, -128);
    let (_dir, weights) = open_fixture(&[
        weight_entry("big", GgufType::Q4K, (UNWIND_ROWS, UNWIND_K), big),
        weight_entry("bad", GgufType::Q80, (N_ROWS, K), bad),
    ]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    // As in the Q6_K test: an upload of `big` would fail on the pipe with
    // ESPIPE, so the scale error proves the check ran before the upload.
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    let result = projector.load(&weights, &["big", "bad"], options);
    assert_error(
        result,
        "Checkpoint",
        "tensor bad block 0: Q8_0 scale 512 times quant 128 exceeds the fp16 range",
    );
    assert_eq!(
        projector.live_bytes(),
        before,
        "the rejected load allocated device memory"
    );
}

#[test]
fn a_nan_or_infinite_q8_0_scale_fails_the_load_before_any_upload() {
    const NAME: &str = "a_nan_or_infinite_q8_0_scale_fails_the_load_before_any_upload";
    if !device_present(NAME) {
        return;
    }
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    for (d_bits, shown) in [(0x7e00u16, "NaN"), (0x7c00, "inf"), (0xfc00, "-inf")] {
        let (_dir, weights) = open_fixture(&[weight_entry(
            "w",
            GgufType::Q80,
            (N_ROWS, K),
            q8_0_blocks_with_scale(d_bits, 0),
        )]);
        let before = projector.live_bytes();
        let result = projector.load(&weights, &["w"], LoadOptions::default());
        assert_error(
            result,
            "Checkpoint",
            &format!("tensor w block 0: Q8_0 scale {shown} times quant"),
        );
        assert_eq!(
            projector.live_bytes(),
            before,
            "the rejected load allocated device memory"
        );
    }
    let fixture = big_and_small_fixture();
    assert_loads_and_runs_after(&mut projector, &fixture, "q4k after a rejected q8_0 scale");
}

/// Checks the tiles of `n_rows` rows against their source blocks: byte-exact
/// for Q4_K and Q8_0, and every quant and fp16 scale word for Q6_K.
fn assert_tiles_match_source(ty: GgufType, tiles: &[u8], source: &[u8], n_rows: usize, k: usize) {
    let blocks_per_row = k / BLOCK_VALUES;
    match ty {
        GgufType::Q4K => {
            let layout = Layout::q4k(n_rows, blocks_per_row);
            assert_eq!(tiles.len(), layout.tile_total());
            assert_same_bytes("q4k decoded bytes", &decode_q4k(tiles, layout), source);
        }
        GgufType::Q80 => {
            let layout = Layout::q8_0(n_rows, blocks_per_row);
            assert_eq!(tiles.len(), layout.tile_total());
            assert_same_bytes("q8_0 decoded bytes", &decode_q8_0(tiles, layout), source);
        }
        GgufType::Q6K => {
            let layout = Layout::q6k(n_rows, blocks_per_row);
            assert_eq!(tiles.len(), layout.tile_total());
            let quants = q6_quants_from_tiles(tiles, layout);
            assert_same_bytes("q6k quants", &quants, &q6_quants_of_source(source));
            assert_scale_words(tiles, source, layout);
        }
        other => panic!("no tile decoder for {other:?}"),
    }
}

/// Loads `source` as tensor `w` with upload and scan chunks of `chunk_limit`
/// bytes and returns the device tiles.
fn tiles_at_chunk_limit(ty: GgufType, source: &[u8], chunk_limit: usize) -> Vec<u8> {
    let (_dir, weights) = open_fixture(&[weight_entry("w", ty, (N_ROWS, K), source.to_vec())]);
    let projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_chunk_limit(chunk_limit);
    let loaded = projector
        .load(&weights, &["w"], LoadOptions::default())
        .expect("load the chunked weight");
    download_tiles(&projector, tiled_of(&loaded, "w"))
}

/// The same tensor must give the same tiles whatever the chunk size, and those
/// tiles must decode to the source (T1). The default limit holds the whole
/// 8-group tensor in one chunk. The small limits give chunks of 1 group
/// (8 chunks), 3 groups (3, 3, 2), 7 groups (7, 1) and a limit that rounds
/// down to 3 groups.
fn chunked_tiles_case(
    test: &str,
    ty: GgufType,
    make_blocks: fn(&mut Rng, usize) -> Vec<u8>,
    tile_bytes: usize,
) {
    if !device_present(test) {
        return;
    }
    let mut rng = Rng(SEED ^ 2);
    let source = make_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let group_tiles = tile_bytes * BLOCKS_PER_ROW;
    let whole = tiles_at_chunk_limit(ty, &source, CHUNK_BYTES);
    assert_tiles_match_source(ty, &whole, &source, N_ROWS, K);
    for limit in [1, 3 * group_tiles, 7 * group_tiles, 4 * group_tiles - 1] {
        let chunked = tiles_at_chunk_limit(ty, &source, limit);
        assert_same_bytes(&format!("{ty:?} at chunk limit {limit}"), &chunked, &whole);
    }
    println!(
        "T1 {ty:?} chunked: {} tile bytes equal the whole-tensor upload at 4 chunk limits and decode to the source",
        whole.len()
    );
}

#[test]
fn q4k_tiles_are_identical_at_every_chunk_size() {
    chunked_tiles_case(
        "q4k_tiles_are_identical_at_every_chunk_size",
        GgufType::Q4K,
        q4k_blocks,
        Q4K_TILE_BYTES,
    );
}

#[test]
fn q6k_tiles_are_identical_at_every_chunk_size() {
    chunked_tiles_case(
        "q6k_tiles_are_identical_at_every_chunk_size",
        GgufType::Q6K,
        q6k_blocks,
        Q6K_TILE_BYTES,
    );
}

#[test]
fn q8_0_tiles_are_identical_at_every_chunk_size() {
    chunked_tiles_case(
        "q8_0_tiles_are_identical_at_every_chunk_size",
        GgufType::Q80,
        q8_0_blocks,
        Q8_0_TILE_BYTES,
    );
}

/// Uploads `ty` with a limit of three groups plus five bytes and checks the read
/// pattern of the source: chunks of 3, 3 and 2 whole groups, in order, at the
/// source offset of each. `group_src` is the source bytes of one 8-row group.
fn read_pattern_case(
    ty: GgufType,
    make_blocks: fn(&mut Rng, usize) -> Vec<u8>,
    group_src: usize,
    group_tiles: usize,
) {
    let mut rng = Rng(SEED ^ 3);
    let source = make_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    assert_eq!(
        source.len(),
        N_ROWS / TILE_ROWS * group_src,
        "{ty:?} fixture"
    );
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let mut reads = Vec::new();
    let weight = projector
        .context
        .upload_tiled(
            QuantTensor {
                name: "w",
                ty,
                n_rows: N_ROWS,
                k: K,
                source_len: source.len(),
            },
            3 * group_tiles + 5,
            &mut |at, buf| {
                reads.push((at, buf.len()));
                buf.copy_from_slice(&source[at..at + buf.len()]);
                Ok(())
            },
        )
        .expect("upload with a limit of three groups");
    assert_eq!(
        reads,
        [
            (0, 3 * group_src),
            (3 * group_src, 3 * group_src),
            (6 * group_src, 2 * group_src)
        ],
        "{ty:?}: chunks of 3, 3 and 2 whole groups, in order"
    );
    assert!(reads.iter().all(|&(_, len)| len <= 3 * group_tiles + 5));
    let tiles = download_tiles(&projector, &weight);
    assert_tiles_match_source(ty, &tiles, &source, N_ROWS, K);
}

/// Q4_K and Q8_0 have as many source bytes as tile bytes in a group (2304 and
/// 4352). Q6_K does not (3360 and 3584), so only it fails if the plan
/// takes the source offsets or lengths from the tile size.
#[test]
fn the_source_is_read_in_whole_group_chunks_in_order() {
    const NAME: &str = "the_source_is_read_in_whole_group_chunks_in_order";
    if !device_present(NAME) {
        return;
    }
    let q8_0_group = BLOCKS_PER_ROW * Q8_0_BLOCKS_PER_256 * Q8_0_BLOCK_BYTES * TILE_ROWS;
    let q6k_group = BLOCKS_PER_ROW * Q6K_BLOCK_BYTES * TILE_ROWS;
    assert_ne!(
        q6k_group,
        BLOCKS_PER_ROW * Q6K_TILE_BYTES,
        "Q6_K tells the sizes apart"
    );
    read_pattern_case(
        GgufType::Q4K,
        q4k_blocks,
        BLOCKS_PER_ROW * Q4K_BLOCK_BYTES * TILE_ROWS,
        BLOCKS_PER_ROW * Q4K_TILE_BYTES,
    );
    read_pattern_case(
        GgufType::Q6K,
        q6k_blocks,
        q6k_group,
        BLOCKS_PER_ROW * Q6K_TILE_BYTES,
    );
    read_pattern_case(
        GgufType::Q80,
        q8_0_blocks,
        q8_0_group,
        BLOCKS_PER_ROW * Q8_0_TILE_BYTES,
    );
}

#[test]
fn a_failed_source_read_returns_its_error_and_frees_the_device_tiles() {
    const NAME: &str = "a_failed_source_read_returns_its_error_and_frees_the_device_tiles";
    if !device_present(NAME) {
        return;
    }
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let mut calls = 0;
    let result = projector.context.upload_tiled(
        QuantTensor {
            name: "w",
            ty: GgufType::Q4K,
            n_rows: N_ROWS,
            k: K,
            source_len: N_ROWS * BLOCKS_PER_ROW * Q4K_BLOCK_BYTES,
        },
        BLOCKS_PER_ROW * Q4K_TILE_BYTES,
        &mut |_, buf| {
            calls += 1;
            buf.fill(0);
            if calls == 3 {
                return Err(EngError::Checkpoint("disk gone".to_owned()));
            }
            Ok(())
        },
    );
    assert_error(result, "Checkpoint", "disk gone");
    assert_eq!(calls, 3, "the upload stopped at the failed read");
    assert_eq!(
        projector.live_bytes(),
        before,
        "the failed upload kept device memory"
    );
    assert_backend_still_loads_and_projects(&mut projector);
}

/// A failed read is not a device fault: a following load and `linear` succeed
/// and match the dense reference.
fn assert_backend_still_loads_and_projects(projector: &mut CudaProjector) {
    let mut rng = Rng(SEED ^ 4);
    let fixture = Fixture::new(GgufType::Q4K, q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW));
    let loaded = projector
        .load(&fixture.weights, &["w"], LoadOptions::default())
        .expect("load after a failed source read");
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    let device = device_linear(projector, &loaded, &x);
    assert_tokens_match(
        "q4k after a failed source read",
        &device,
        &dense_reference(&fixture.dense, &x),
    );
}

/// A tensor `bad` after a small valid one, scanned in chunks of five blocks, with
/// the overflowing scale in the last block (a late chunk, at a local index that is
/// not zero). The error must carry the index from the tensor start, and no upload
/// may run: the page-cache drop after `w` would fail on the pipe with ESPIPE.
fn late_chunk_overflow_case(test: &str, bad: Entry, block_bytes: usize, want: &str) {
    if !device_present(test) {
        return;
    }
    let mut rng = Rng(SEED);
    let small = q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let (_dir, weights) =
        open_fixture(&[weight_entry("w", GgufType::Q4K, (N_ROWS, K), small), bad]);
    let projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_chunk_limit(5 * block_bytes);
    let before = projector.live_bytes();
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    assert_error(
        projector.load(&weights, &["w", "bad"], options),
        "Checkpoint",
        want,
    );
    assert_eq!(
        projector.live_bytes(),
        before,
        "the rejected load allocated device memory"
    );
}

#[test]
fn a_q6k_scale_overflow_in_a_late_chunk_names_the_absolute_block_and_fails_before_any_upload() {
    let mut rng = Rng(SEED);
    let mut source = q6k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let blocks = source.len() / Q6K_BLOCK_BYTES;
    // d = 65504 (0x7bff) and a scale of 2 give a product of 131008, past fp16.
    let last = &mut source[(blocks - 1) * Q6K_BLOCK_BYTES..];
    last[192] = 2;
    last[208..].copy_from_slice(&0x7bffu16.to_le_bytes());
    late_chunk_overflow_case(
        "a_q6k_scale_overflow_in_a_late_chunk_names_the_absolute_block_and_fails_before_any_upload",
        weight_entry("bad", GgufType::Q6K, (N_ROWS, K), source),
        Q6K_BLOCK_BYTES,
        &format!("tensor bad block {}: Q6_K scale product", blocks - 1),
    );
}

#[test]
fn a_q8_0_scale_overflow_in_a_late_chunk_names_the_absolute_block_and_fails_before_any_upload() {
    let mut rng = Rng(SEED);
    let mut source = q8_0_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let blocks = source.len() / Q8_0_BLOCK_BYTES;
    // 0x6000 is 512, and 512 * 128 = 65536: one fp16 step of d past 65504.
    let last = &mut source[(blocks - 1) * Q8_0_BLOCK_BYTES..];
    last[..2].copy_from_slice(&0x6000u16.to_le_bytes());
    last[2] = (-128i8).cast_unsigned();
    late_chunk_overflow_case(
        "a_q8_0_scale_overflow_in_a_late_chunk_names_the_absolute_block_and_fails_before_any_upload",
        weight_entry("bad", GgufType::Q80, (N_ROWS, K), source),
        Q8_0_BLOCK_BYTES,
        &format!(
            "tensor bad block {}: Q8_0 scale 512 times quant 128 exceeds the fp16 range",
            blocks - 1
        ),
    );
}

/// The peak resident set of this process so far, in kB, from /proc/self/status.
fn peak_rss_kb() -> u64 {
    let text = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    meminfo_kb(&text, "VmHWM")
}

/// Asserts that the loads since `rss_before_kb` raised the peak resident set
/// by less than `LARGE_RSS_BOUND_KB`, and prints both figures.
///
/// `VmHWM` never shrinks, so the difference is not negative; `saturating_sub`
/// only avoids a debug panic if a reader returns 0.
fn assert_peak_rss_bounded(rss_before_kb: u64, tensor_bytes: u64) {
    let rss_after_kb = peak_rss_kb();
    let rss_delta_kb = rss_after_kb.saturating_sub(rss_before_kb);
    println!(
        "LARGE peak_rss_kb_before={rss_before_kb} peak_rss_kb_after_loads={rss_after_kb} \
         delta_kb={rss_delta_kb} bound_kb={LARGE_RSS_BOUND_KB}"
    );
    assert!(
        rss_delta_kb < LARGE_RSS_BOUND_KB,
        "two loads of a {tensor_bytes}-byte tensor raised the peak resident set by {rss_delta_kb} kB, \
         not below the bound of {LARGE_RSS_BOUND_KB} kB: the source is no longer chunked"
    );
}

/// Largest tensor of the GGUF named by INFERENCE_CUDA_LARGE_TENSOR_GGUF that the
/// loader can upload. Loads it twice (the first time cold: its pages are dropped
/// from the page cache first), prints both times and the peak resident set of
/// the process, then checks the tiles of a sample of 8-row groups against the
/// source blocks read back from the file. The 27B token_embd is 1,042,944,000
/// bytes of Q6_K, above the old 960 MiB staging limit. Nothing but the sampled
/// groups of the source is held beside the downloaded tiles.
///
/// `VmHWM` is process-wide, so run this test alone with `--exact`. A peak
/// from an earlier test in the same process would hide a regression.
#[test]
#[ignore = "uploads the largest tensor of the GGUF named by INFERENCE_CUDA_LARGE_TENSOR_GGUF"]
fn the_largest_tensor_of_a_named_gguf_uploads_in_chunks_and_matches_the_source() {
    const NAME: &str =
        "the_largest_tensor_of_a_named_gguf_uploads_in_chunks_and_matches_the_source";
    let Some(path) = std::env::var_os("INFERENCE_CUDA_LARGE_TENSOR_GGUF")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        println!("SKIPPED {NAME}: INFERENCE_CUDA_LARGE_TENSOR_GGUF unset");
        return;
    };
    if !device_present(NAME) {
        return;
    }
    let limits = GgufLimits {
        max_header_bytes: MEASURE_LIMITS.max_header_bytes,
        max_tensor_bytes: LARGE_LIMITS.max_tensor_bytes,
    };
    let weights = GgufWeights::open(&path, limits, None)
        .unwrap_or_else(|error| panic!("open {}: {error:?}", path.display()));
    let tensor = weights
        .header()
        .tensors()
        .iter()
        .filter(|tensor| uploadable(tensor))
        .max_by_key(|tensor| tensor.byte_len())
        .expect("the GGUF holds an uploadable tensor");
    let (n_rows, k) = matrix_shape(tensor).expect("the largest tensor is a matrix");
    let (name, ty) = (tensor.name(), tensor.ggml_type());
    println!(
        "LARGE tensor={name} type={ty:?} rows={n_rows} k={k} source_bytes={}",
        tensor.byte_len()
    );
    let file = File::open(&path).expect("reopen the GGUF for the page-cache drop");
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let rss_before_kb = peak_rss_kb();
    cuda::drop_page_cache(&file, tensor.abs_offset(), tensor.byte_len())
        .expect("drop the tensor from the page cache");
    let mut loaded = None;
    for pass in ["cold", "warm"] {
        drop(loaded.take());
        let start = Instant::now();
        let weights_loaded = projector
            .load(&weights, &[name], LoadOptions::default())
            .expect("load the largest tensor");
        println!(
            "LARGE pass={pass} load_s={:.3} device_bytes={}",
            start.elapsed().as_secs_f64(),
            weights_loaded.device_bytes()
        );
        loaded = Some(weights_loaded);
    }
    assert_peak_rss_bounded(rss_before_kb, tensor.byte_len());
    let loaded = loaded.expect("two loads ran");
    let tiles = download_tiles(&projector, tiled_of(&loaded, name));
    let oracle = File::open(&path).expect("open the GGUF a second time for the oracle");
    assert_sampled_groups_match_source(tensor, &oracle, &tiles, (n_rows, k));
}

/// Checks the first and last three groups, about 60 evenly spaced ones and 64
/// pseudo-random ones against their source bytes. The expected bytes come from
/// positioned reads of `file`, a handle the loader never touched, at the tensor
/// offset the header recorded; `GgufWeights::read_range`, which the upload
/// uses, is not the oracle.
fn assert_sampled_groups_match_source(
    tensor: &GgufTensor,
    file: &File,
    tiles: &[u8],
    shape: (usize, usize),
) {
    let (n_rows, k) = shape;
    let ty = tensor.ggml_type();
    let groups = n_rows / TILE_ROWS;
    let group_tiles = tiles.len() / groups;
    let group_src = tensor.byte_len() as usize / groups;
    let mut rng = Rng(SEED);
    let mut sample: Vec<usize> = (0..3).chain(groups - 3..groups).collect();
    sample.extend((0..60).map(|i| i * groups / 60));
    sample.extend((0..64).map(|_| rng.next_u64() as usize % groups));
    sample.sort_unstable();
    sample.dedup();
    let mut source = vec![0u8; group_src];
    for &group in &sample {
        file.read_exact_at(
            &mut source,
            tensor.abs_offset() + (group * group_src) as u64,
        )
        .expect("read a source group from the file");
        let tile = &tiles[group * group_tiles..(group + 1) * group_tiles];
        assert_tiles_match_source(ty, tile, &source, TILE_ROWS, k);
    }
    println!(
        "T1 large: {} of {groups} groups ({} tile bytes each) decode to the source",
        sample.len(),
        group_tiles
    );
}

/// The GGUF to measure, when the operator names one with INFERENCE_CUDA_MEASURE_GGUF.
/// Only the trim step reads it. The measured runs open the trimmed copy instead.
fn measure_path() -> Option<PathBuf> {
    std::env::var_os("INFERENCE_CUDA_MEASURE_GGUF")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The directory for the trimmed copy, from INFERENCE_CUDA_MEASURE_TRIM_DIR. It must be
/// named whenever INFERENCE_CUDA_MEASURE_GGUF is set.
fn trim_dir() -> PathBuf {
    std::env::var_os("INFERENCE_CUDA_MEASURE_TRIM_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .expect("INFERENCE_CUDA_MEASURE_TRIM_DIR must name a directory when INFERENCE_CUDA_MEASURE_GGUF is set")
}

/// INFERENCE_CUDA_MEASURE_TRIM_ONLY=1 writes the trimmed copy and returns before any
/// device work. The operator runs it once in its own process, so the copy's page-cache
/// traffic stays out of every measured baseline. Any other value than 0 or unset fails.
fn trim_only() -> bool {
    match std::env::var("INFERENCE_CUDA_MEASURE_TRIM_ONLY")
        .ok()
        .as_deref()
    {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        other => panic!(
            "INFERENCE_CUDA_MEASURE_TRIM_ONLY must be 1 (trim only) or 0 (measure), got {other:?}"
        ),
    }
}

/// Whether a run drops the page cache of each uploaded tensor before it is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageCacheMode {
    Drop,
    Keep,
}

impl PageCacheMode {
    fn label(self) -> &'static str {
        match self {
            Self::Drop => "drop",
            Self::Keep => "keep",
        }
    }
}

/// INFERENCE_CUDA_MEASURE_DROP_PAGE_CACHE picks the variant. The operator runs one
/// variant per process, so pool state left by one variant cannot reach the other's
/// baseline. Any value other than 1 or 0 fails the run instead of guessing.
fn measure_mode() -> PageCacheMode {
    match std::env::var("INFERENCE_CUDA_MEASURE_DROP_PAGE_CACHE")
        .ok()
        .as_deref()
    {
        Some("1") => PageCacheMode::Drop,
        Some("0") => PageCacheMode::Keep,
        other => panic!(
            "INFERENCE_CUDA_MEASURE_DROP_PAGE_CACHE must be 1 (drop) or 0 (keep) when INFERENCE_CUDA_MEASURE_GGUF is set, got {other:?}"
        ),
    }
}

/// INFERENCE_CUDA_MEASURE_MAX_BYTES caps the tensor bytes one run uploads. Unset or
/// empty means no cap.
fn measure_cap() -> u64 {
    let Some(text) = std::env::var("INFERENCE_CUDA_MEASURE_MAX_BYTES")
        .ok()
        .filter(|text| !text.is_empty())
    else {
        return u64::MAX;
    };
    text.parse().unwrap_or_else(|error| {
        panic!("INFERENCE_CUDA_MEASURE_MAX_BYTES must be a u64 byte count, got {text:?}: {error}")
    })
}

/// Whether the loader can upload a tensor: the plan rejects any type or shape the
/// CUDA backend cannot hold (`device_bytes` fails for them), and `load` has no other
/// per-tensor limit. Every uploadable name loads.
fn uploadable(tensor: &GgufTensor) -> bool {
    plan::device_bytes(tensor.name(), tensor.ggml_type(), tensor.dims()).is_ok()
}

/// Tensors of the measured file that one run leaves out: those the loader cannot
/// upload (the file may hold types such as Q5_K), and uploadable ones past the cap.
struct Skipped {
    unsupported: usize,
    capped: usize,
}

/// The uploadable tensors of `all` in header order, cut before the first one that
/// would take the total past `max_bytes`. Returns the kept tensors and what was left out.
fn select_tensors(all: &[GgufTensor], max_bytes: u64) -> (Vec<&GgufTensor>, Skipped) {
    let (candidates, unsupported): (Vec<&GgufTensor>, Vec<&GgufTensor>) =
        all.iter().partition(|tensor| uploadable(tensor));
    let byte_lens: Vec<u64> = candidates.iter().map(|tensor| tensor.byte_len()).collect();
    let kept = within_cap(&byte_lens, max_bytes);
    let skipped = Skipped {
        unsupported: unsupported.len(),
        capped: candidates.len() - kept,
    };
    (candidates.into_iter().take(kept).collect(), skipped)
}

/// How many leading entries of `byte_lens` fit within `max_bytes` in total. Counting
/// stops at the first entry that would overflow, so the kept set is a header prefix.
fn within_cap(byte_lens: &[u64], max_bytes: u64) -> usize {
    let mut total = 0u64;
    for (index, len) in byte_lens.iter().enumerate() {
        total = total.saturating_add(*len);
        if total > max_bytes {
            return index;
        }
    }
    byte_lens.len()
}

/// Host memory in kB from /proc/meminfo. On Xavier the GPU shares this DRAM, so these
/// are the figures that page-cache drops act on.
struct HostMemory {
    mem_free_kb: u64,
    mem_available_kb: u64,
    cached_kb: u64,
}

fn host_memory() -> HostMemory {
    let text = std::fs::read_to_string("/proc/meminfo").expect("read /proc/meminfo");
    HostMemory {
        mem_free_kb: meminfo_kb(&text, "MemFree"),
        mem_available_kb: meminfo_kb(&text, "MemAvailable"),
        cached_kb: meminfo_kb(&text, "Cached"),
    }
}

/// The number on one `Key: <n> kB` line. A missing or unparsable line fails the run,
/// so a report never shows a zero for a field it could not read.
pub(super) fn meminfo_kb(text: &str, key: &str) -> u64 {
    let label = format!("{key}:");
    text.lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            if fields.next() != Some(label.as_str()) {
                return None;
            }
            fields.next()?.parse::<u64>().ok()
        })
        .unwrap_or_else(|| panic!("/proc/meminfo has no numeric {key} line"))
}

/// Prints device and host memory for one stage of one variant, one line per stage.
fn report_memory(mode: PageCacheMode, stage: &str, projector: &CudaProjector) {
    let device = projector.memory_info().expect("query device memory");
    let host = host_memory();
    println!(
        "MEASURE variant={} stage={stage} device_free={} device_total={} mem_free_kb={} mem_available_kb={} cached_kb={}",
        mode.label(),
        device.free,
        device.total,
        host.mem_free_kb,
        host.mem_available_kb,
        host.cached_kb
    );
}

/// The trimmed copy's file name inside INFERENCE_CUDA_MEASURE_TRIM_DIR.
const TRIMMED_NAME: &str = "measure-trimmed.gguf";

/// Bytes before the metadata section: magic, version, tensor count, metadata count.
const PREFIX_BYTES: u64 = 24;

/// What one trim kept and left out, with the sizes of both files.
struct Trimmed {
    kept: usize,
    skipped: Skipped,
    source_bytes: u64,
    trimmed_bytes: u64,
}

/// The kept tensors and their offsets in the data section, in header order.
struct Plan<'a> {
    kept: Vec<&'a GgufTensor>,
    offsets: Vec<u64>,
}

impl<'a> Plan<'a> {
    /// Packs the kept tensors back to back, each starting at the next `align` boundary.
    fn packed(kept: Vec<&'a GgufTensor>, align: u64) -> Self {
        let mut end = 0u64;
        let offsets = kept
            .iter()
            .map(|tensor| {
                let offset = end.next_multiple_of(align);
                end = offset + tensor.byte_len();
                offset
            })
            .collect();
        Self { kept, offsets }
    }
}

/// Writes a GGUF v3 copy of `source` with its metadata and only the tensors that
/// `select_tensors` keeps, in header order. Payloads are byte copies at new aligned
/// offsets. The copy is written under a temporary name, synced, then renamed, so a
/// failed write never leaves a truncated file at `dest`.
fn write_trimmed(source: &Path, dest: &Path, max_bytes: u64) -> Trimmed {
    let mut file = File::open(source).expect("open the GGUF to trim");
    let source_bytes = file.metadata().expect("stat the GGUF to trim").len();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        source_bytes,
        MEASURE_LIMITS.max_header_bytes,
    )
    .expect("parse the header of the GGUF to trim");
    assert!(
        header.split().is_none_or(|split| split.count <= 1),
        "the trim does not handle split GGUF files"
    );
    let (kept, skipped) = select_tensors(header.tensors(), max_bytes);
    let plan = Plan::packed(kept, header.alignment());
    let partial = dest.with_extension("partial");
    let mut out = BufWriter::new(File::create(&partial).expect("create the trimmed GGUF"));
    write_header(&mut out, &mut file, &header, &plan);
    write_payloads(&mut out, &mut file, &plan);
    out.flush().expect("flush the trimmed GGUF");
    let out = out
        .into_inner()
        .map_err(|error| error.into_error())
        .expect("finish the trimmed GGUF");
    out.sync_all().expect("sync the trimmed GGUF");
    drop(out);
    std::fs::rename(&partial, dest).expect("move the trimmed GGUF into place");
    check_trimmed(dest, &header, &plan.kept);
    Trimmed {
        kept: plan.kept.len(),
        skipped,
        source_bytes,
        trimmed_bytes: std::fs::metadata(dest)
            .expect("stat the trimmed GGUF")
            .len(),
    }
}

/// Writes the GGUF prefix, the source metadata bytes, the kept tensor table, and zero
/// padding up to the aligned data section.
fn write_header(out: &mut impl Write, source: &mut File, header: &GgufHeader, plan: &Plan<'_>) {
    out.write_all(b"GGUF").expect("write the GGUF magic");
    out.write_all(&header.version().to_le_bytes())
        .expect("write the GGUF version");
    out.write_all(&(plan.kept.len() as u64).to_le_bytes())
        .expect("write the tensor count");
    out.write_all(&(header.metadata().len() as u64).to_le_bytes())
        .expect("write the metadata count");
    let meta_end = metadata_end(header);
    copy_range(source, PREFIX_BYTES..meta_end, out);
    for (tensor, offset) in plan.kept.iter().zip(&plan.offsets) {
        write_info(out, tensor, *offset);
    }
    let header_len = meta_end + plan.kept.iter().map(|tensor| info_len(tensor)).sum::<u64>();
    write_zeros(
        out,
        header_len.next_multiple_of(header.alignment()) - header_len,
    );
}

/// Writes each kept tensor's bytes at its offset, zero-filling the gaps between them.
fn write_payloads(out: &mut impl Write, source: &mut File, plan: &Plan<'_>) {
    let mut written = 0u64;
    for (tensor, offset) in plan.kept.iter().zip(&plan.offsets) {
        write_zeros(out, offset - written);
        copy_range(source, tensor.abs_range(), out);
        written = offset + tensor.byte_len();
    }
}

/// Where the metadata section ends. The tensor table is the last part of the header,
/// so it starts as many bytes before the header end as the table entries take.
fn metadata_end(header: &GgufHeader) -> u64 {
    let table: u64 = header.tensors().iter().map(info_len).sum();
    header.header_len() - table
}

/// Serialized size of one tensor-table entry: name length and bytes, dimension count,
/// dimensions, type, and offset.
fn info_len(tensor: &GgufTensor) -> u64 {
    8 + tensor.name().len() as u64 + 4 + 8 * tensor.dims().len() as u64 + 4 + 8
}

/// Writes one tensor-table entry for `tensor` at data offset `offset`.
fn write_info(out: &mut impl Write, tensor: &GgufTensor, offset: u64) {
    let name = tensor.name().as_bytes();
    out.write_all(&(name.len() as u64).to_le_bytes())
        .expect("write a tensor name length");
    out.write_all(name).expect("write a tensor name");
    out.write_all(&(tensor.dims().len() as u32).to_le_bytes())
        .expect("write a dimension count");
    for dim in tensor.dims() {
        out.write_all(&dim.to_le_bytes())
            .expect("write a dimension");
    }
    out.write_all(&tensor.ggml_type().id().to_le_bytes())
        .expect("write a tensor type");
    out.write_all(&offset.to_le_bytes())
        .expect("write a tensor offset");
}

/// Copies `range` of `source` to `out`. A short read fails the trim, so a truncated
/// source never produces a copy that looks complete.
fn copy_range(source: &mut File, range: Range<u64>, out: &mut impl Write) {
    source
        .seek(SeekFrom::Start(range.start))
        .expect("seek in the GGUF to trim");
    let len = range.end - range.start;
    let copied = io::copy(&mut (&mut *source).take(len), out).expect("copy a tensor payload");
    assert_eq!(copied, len, "the GGUF to trim ended inside a tensor");
}

fn write_zeros(out: &mut impl Write, count: u64) {
    out.write_all(&vec![0u8; count as usize])
        .expect("write padding");
}

/// Parses the finished copy and compares it with the source: the same metadata keys,
/// then the kept tensors in the same order with the same type, dimensions, and size.
fn check_trimmed(dest: &Path, source: &GgufHeader, kept: &[&GgufTensor]) {
    let file = File::open(dest).expect("reopen the trimmed GGUF");
    let len = file.metadata().expect("stat the trimmed GGUF").len();
    let copy = GgufHeader::parse(
        &mut BufReader::new(&file),
        len,
        MEASURE_LIMITS.max_header_bytes,
    )
    .expect("parse the trimmed GGUF header");
    let keys = |header: &GgufHeader| -> Vec<String> {
        header
            .metadata()
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    };
    assert_eq!(keys(&copy), keys(source), "trimmed metadata keys differ");
    assert_eq!(
        copy.alignment(),
        source.alignment(),
        "trimmed alignment differs"
    );
    assert_eq!(
        copy.tensors().len(),
        kept.len(),
        "trimmed tensor count differs"
    );
    for (got, want) in copy.tensors().iter().zip(kept) {
        assert_eq!(got.name(), want.name(), "trimmed tensor order differs");
        assert_eq!(got.ggml_type(), want.ggml_type(), "type of {}", got.name());
        assert_eq!(got.dims(), want.dims(), "dims of {}", got.name());
        assert_eq!(got.byte_len(), want.byte_len(), "size of {}", got.name());
        assert_eq!(
            got.data_offset() % copy.alignment(),
            0,
            "offset of {} is misaligned",
            got.name()
        );
    }
}

/// Writes the trimmed copy for the measurement runs, then opens it once so a bad copy
/// fails here instead of inside a measured run.
fn trim_and_report(source: &Path, dir: &Path, max_bytes: u64) {
    let dest = dir.join(TRIMMED_NAME);
    let trim = write_trimmed(source, &dest, max_bytes);
    let weights = GgufWeights::open(&dest, MEASURE_LIMITS, None).expect("open the trimmed GGUF");
    println!(
        "MEASURE stage=trim source={} trimmed={} kept={} tensors_opened={} skipped_unsupported={} skipped_cap={} max_bytes={max_bytes} source_bytes={} trimmed_bytes={}",
        source.display(),
        dest.display(),
        trim.kept,
        weights.header().tensors().len(),
        trim.skipped.unsupported,
        trim.skipped.capped,
        trim.source_bytes,
        trim.trimmed_bytes
    );
}

/// Loads one variant's tensors in this process and reports memory before the load,
/// after it, and after release. The operator compares the two variants across
/// separate processes, so each run records its own baseline. The run reads only the
/// trimmed copy, which the trim step wrote before any measured run.
#[test]
fn measure_load_memory_for_one_page_cache_mode() {
    const NAME: &str = "measure_load_memory_for_one_page_cache_mode";
    let Some(source) = measure_path() else {
        println!("SKIPPED {NAME}: INFERENCE_CUDA_MEASURE_GGUF unset");
        return;
    };
    println!("loader page-cache measurement, not a model-load result");
    let dir = trim_dir();
    let max_bytes = measure_cap();
    if trim_only() {
        trim_and_report(&source, &dir, max_bytes);
        return;
    }
    let mode = measure_mode();
    if !device_present(NAME) {
        return;
    }
    let path = dir.join(TRIMMED_NAME);
    let weights = GgufWeights::open(&path, MEASURE_LIMITS, None).unwrap_or_else(|error| {
        panic!(
            "open {} (run with INFERENCE_CUDA_MEASURE_TRIM_ONLY=1 first): {error:?}",
            path.display()
        )
    });
    let (tensors, skipped) = select_tensors(weights.header().tensors(), max_bytes);
    let names: Vec<&str> = tensors.iter().map(|tensor| tensor.name()).collect();
    let file = File::open(&path).expect("reopen the trimmed GGUF for page-cache drops");
    let projector = CudaProjector::new().expect("create the CUDA projector");
    println!(
        "MEASURE variant={} gguf={} measured={} skipped_unsupported={} skipped_cap={} max_bytes={max_bytes}",
        mode.label(),
        path.display(),
        names.len(),
        skipped.unsupported,
        skipped.capped
    );
    report_memory(mode, "before_load", &projector);
    let options = LoadOptions {
        drop_page_cache: (mode == PageCacheMode::Drop).then_some(&file),
        ..LoadOptions::default()
    };
    let loaded = projector
        .load(&weights, &names, options)
        .expect("load the measured tensors");
    println!(
        "MEASURE variant={} stage=loaded tensors={} device_bytes={}",
        mode.label(),
        loaded.len(),
        loaded.device_bytes()
    );
    report_memory(mode, "after_load", &projector);
    drop(loaded);
    report_memory(mode, "after_release", &projector);
}

/// REPLY-20 item 4: the time the pre-upload Q8_0 scale scan adds to a load. It
/// reads every Q8_0 tensor of INFERENCE_CUDA_MEASURE_SCAN_GGUF the way
/// `check_scale_ranges` does, once after dropping the file from the page cache
/// (cold) and once more (warm), and prints read and check time apart. Nothing
/// is uploaded.
#[test]
fn measure_q8_0_scale_scan_time() {
    const NAME: &str = "measure_q8_0_scale_scan_time";
    let Some(path) = std::env::var_os("INFERENCE_CUDA_MEASURE_SCAN_GGUF").map(PathBuf::from) else {
        println!("SKIPPED {NAME}: INFERENCE_CUDA_MEASURE_SCAN_GGUF unset");
        return;
    };
    if !device_present(NAME) {
        return;
    }
    let weights = GgufWeights::open(&path, MEASURE_LIMITS, None)
        .unwrap_or_else(|error| panic!("open {}: {error:?}", path.display()));
    let q8_0: Vec<&GgufTensor> = weights
        .header()
        .tensors()
        .iter()
        .filter(|tensor| tensor.ggml_type() == GgufType::Q80)
        .collect();
    let file = File::open(&path).expect("reopen the GGUF for the page-cache drop");
    let file_len = file.metadata().expect("GGUF metadata").len();
    cuda::drop_page_cache(&file, 0, file_len).expect("drop the GGUF from the page cache");
    for pass in ["cold", "warm"] {
        let (mut read, mut check, mut bytes) = (Duration::ZERO, Duration::ZERO, 0u64);
        for tensor in &q8_0 {
            let start = Instant::now();
            let (_, data) = weights
                .read_tensor(tensor.name())
                .expect("read a Q8_0 tensor");
            let read_done = Instant::now();
            cuda::check_q8_0_scales(tensor.name(), 0, &data).expect("Q8_0 scales in range");
            check += read_done.elapsed();
            read += read_done - start;
            bytes += data.len() as u64;
        }
        let seconds = (read + check).as_secs_f64();
        println!(
            "SCAN pass={pass} tensors={} bytes={bytes} read_s={:.3} check_s={:.3} total_s={seconds:.3} mb_per_s={:.1}",
            q8_0.len(),
            read.as_secs_f64(),
            check.as_secs_f64(),
            bytes as f64 / 1e6 / seconds
        );
    }
}

/// Four tensors around the Q5_K one that the loader rejects. The Q4_K and Q6_K shapes
/// have 8 rows and 256 columns, the smallest the device tiles accept, so `fit`
/// keeps them. Payload bytes differ per tensor, so a misplaced copy shows up as a byte
/// mismatch.
fn trim_fixture() -> Vec<Entry> {
    vec![
        weight_entry("blk.0.q4k", GgufType::Q4K, (8, 256), filled(8 * 144, 1)),
        weight_entry("blk.0.q5k", GgufType::Q5K, (2, 256), filled(2 * 176, 2)),
        weight_entry("blk.0.q6k", GgufType::Q6K, (8, 256), filled(8 * 210, 4)),
        weight_entry("blk.0.f32", GgufType::F32, (2, 4), filled(32, 3)),
    ]
}

fn filled(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

/// Every kept tensor reads back with its source name, type, and payload bytes, in
/// header order, and the file keeps the fixture's alignment.
fn assert_payloads(weights: &GgufWeights, kept: &[&Entry]) {
    let names: Vec<&str> = weights
        .header()
        .tensors()
        .iter()
        .map(|tensor| tensor.name())
        .collect();
    let want: Vec<&str> = kept.iter().map(|entry| entry.name).collect();
    assert_eq!(names, want);
    assert_eq!(weights.header().alignment(), 32);
    for entry in kept {
        let (ty, bytes) = weights
            .read_tensor(entry.name)
            .expect("read a trimmed tensor");
        assert_eq!(ty, entry.ty, "type of {}", entry.name);
        assert_eq!(bytes, entry.payload, "payload of {}", entry.name);
    }
}

#[test]
fn trimmed_copy_drops_the_q5k_tensor_and_keeps_the_rest_byte_for_byte() {
    let dir = tempfile::tempdir().expect("create the trim directory");
    let source = dir.path().join("source.gguf");
    let entries = trim_fixture();
    std::fs::write(&source, gguf_bytes(&entries)).expect("write the source fixture");
    assert!(
        GgufWeights::open(&source, MEASURE_LIMITS, None).is_err(),
        "the source must fail the open because of its Q5_K tensor"
    );
    let dest = dir.path().join("trimmed.gguf");
    let trim = write_trimmed(&source, &dest, u64::MAX);
    assert_eq!(
        (trim.kept, trim.skipped.unsupported, trim.skipped.capped),
        (3, 1, 0)
    );
    let weights = GgufWeights::open(&dest, MEASURE_LIMITS, None).expect("open the trimmed fixture");
    let kept: Vec<&Entry> = entries.iter().filter(|e| e.ty != GgufType::Q5K).collect();
    assert_payloads(&weights, &kept);
}

#[test]
fn trimmed_copy_stops_at_the_cap_in_header_order() {
    let dir = tempfile::tempdir().expect("create the trim directory");
    let source = dir.path().join("source.gguf");
    let entries = trim_fixture();
    std::fs::write(&source, gguf_bytes(&entries)).expect("write the source fixture");
    let dest = dir.path().join("trimmed.gguf");
    // Q4_K is 1152 bytes and fits under 1200. The Q6_K tensor (1680 bytes) then
    // overflows, so the kept prefix ends at Q4_K even though the F32 tensor after it
    // (32 bytes) would fit on its own.
    let trim = write_trimmed(&source, &dest, 1200);
    assert_eq!((trim.kept, trim.skipped.capped), (1, 2));
    let weights = GgufWeights::open(&dest, MEASURE_LIMITS, None).expect("open the trimmed fixture");
    let kept: Vec<&Entry> = entries.iter().take(1).collect();
    assert_payloads(&weights, &kept);
}

#[test]
fn within_cap_keeps_the_prefix_before_the_first_overflow() {
    assert_eq!(within_cap(&[4, 4, 4], 8), 2);
    assert_eq!(within_cap(&[4, 9, 1], 8), 1);
    assert_eq!(within_cap(&[4, 4], u64::MAX), 2);
    assert_eq!(within_cap(&[], 0), 0);
}

#[test]
fn meminfo_kb_reads_only_the_named_line() {
    let text = "MemTotal:       30000 kB\nMemFree:         1000 kB\nMemAvailable:    2000 kB\nCached:           300 kB\nSwapCached:         9 kB\n";
    assert_eq!(meminfo_kb(text, "MemFree"), 1000);
    assert_eq!(meminfo_kb(text, "MemAvailable"), 2000);
    assert_eq!(meminfo_kb(text, "Cached"), 300);
}

// LD-1: the load policy of an integrated GPU.

/// Residency at or below which a file counts as evicted. `posix_fadvise(DONTNEED)`
/// drops whole pages inside the range only, so a page shared by two tensors, or
/// holding the header, survives a per-tensor drop.
const EVICTED_MAX: u64 = 8 * 4096;
/// Residency allowed after a load that read tensors and dropped each range.
/// Sequential reads leave a readahead tail past the last range: 512 KiB were
/// measured on the Xavier (`read_ahead_kb` 128). A load that skipped the drop
/// leaves the whole tensor data, 14 MiB or more in these fixtures.
const LOAD_RESIDUAL_MAX: u64 = 2 << 20;
/// Rows of the Q6_K tensor of the scan test: 3.4 MB of source, well above any
/// readahead tail, so only the second eviction can clear it.
const SCAN_Q6_ROWS: usize = 8192;

fn fixture_path(fixture: &Fixture) -> PathBuf {
    fixture._dir.path().join("fixture.gguf")
}

/// Reads the whole file so the page cache holds it.
fn warm(path: &Path) {
    std::fs::read(path).expect("read the fixture file");
}

/// True when the filesystem of `path` honours `posix_fadvise(DONTNEED)` and
/// `fincore` can see it. Otherwise prints why the test cannot judge residency
/// and returns false: a skip is never a pass (see `SKIPPED` in the logs).
fn droppable(path: &Path, name: &str) -> bool {
    let file = File::open(path).expect("open the fixture file");
    // Dirty pages survive the call, so write the fixture out first.
    file.sync_all().expect("sync the fixture file");
    warm(path);
    match resident_bytes(path) {
        Some(bytes) if bytes > EVICTED_MAX => {}
        other => {
            println!("SKIPPED {name}: fincore shows {other:?} bytes after reading the file");
            return false;
        }
    }
    cuda::drop_page_cache(&file, 0, 0).expect("drop the fixture from the page cache");
    match resident_bytes(path) {
        Some(bytes) if bytes <= EVICTED_MAX => true,
        other => {
            println!("SKIPPED {name}: the filesystem keeps {other:?} bytes after posix_fadvise");
            false
        }
    }
}

/// Prints `SKIPPED` and returns true on an integrated device, where a request
/// above the device total is shared RAM and is not run.
fn skip_if_integrated(name: &str) -> bool {
    let integrated = cuda::device_is_integrated().expect("query the integrated attribute");
    if integrated {
        println!("SKIPPED {name}: integrated device, an oversized request is shared RAM");
    }
    integrated
}

/// Rows of 32768 Q4_K values whose tiles need more than `device_total` plus
/// 1 GiB, a whole number of 8-row groups.
fn oversized_rows(device_total: usize) -> usize {
    let row_tile_bytes = CHUNK_BLOCKS_PER_ROW * Q4K_TILE_BYTES / TILE_ROWS;
    ((device_total + (1 << 30)) / row_tile_bytes + 1).next_multiple_of(TILE_ROWS)
}

fn oversized_quant(name: &str, rows: usize) -> QuantTensor<'_> {
    QuantTensor {
        name,
        ty: GgufType::Q4K,
        n_rows: rows,
        k: CHUNK_K,
        source_len: rows * CHUNK_BLOCKS_PER_ROW * Q4K_BLOCK_BYTES,
    }
}

/// A GGUF with a small Q4_K `w` and, last, a Q4_K `huge` whose tiles exceed the
/// device. `huge` is a sparse file hole: the file is extended to the end of its
/// range and nothing is ever read from it.
fn sparse_oom_fixture(device_total: usize) -> (tempfile::TempDir, GgufWeights) {
    let rows = oversized_rows(device_total);
    let mut rng = Rng(SEED);
    let small = q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let entries = [
        weight_entry("w", GgufType::Q4K, (N_ROWS, K), small),
        weight_entry("huge", GgufType::Q4K, (rows, CHUNK_K), Vec::new()),
    ];
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path = dir.path().join("fixture.gguf");
    let bytes = gguf_bytes(&entries);
    std::fs::write(&path, &bytes).expect("write the sparse fixture");
    let huge_bytes = oversized_quant("huge", rows).source_len as u64;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("reopen the sparse fixture")
        .set_len(bytes.len() as u64 + huge_bytes)
        .expect("extend the sparse fixture");
    let weights = GgufWeights::open(&path, MEASURE_LIMITS, None).expect("open the sparse fixture");
    (dir, weights)
}

fn expect_error<T>(result: Result<T, EngError>) -> EngError {
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => error,
    }
}

/// `names` as the plan `load` hands to `transfer`, without admission.
fn plan_of<'a>(
    weights: &'a GgufWeights,
    names: &[&str],
    policy: LoadPolicy,
    drop_file: Option<&'a File>,
) -> LoadPlan<'a> {
    let tensors = names
        .iter()
        .map(|name| planned(weights, name))
        .collect::<Result<Vec<_>, _>>()
        .expect("plan the tensors");
    LoadPlan {
        tensors,
        index: load_index(names).expect("index the names"),
        policy,
        drop_file,
    }
}

#[test]
fn a_real_device_allocation_failure_is_retried_and_the_next_attempt_can_succeed() {
    const NAME: &str =
        "a_real_device_allocation_failure_is_retried_and_the_next_attempt_can_succeed";
    if !device_present(NAME) || skip_if_integrated(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let source = q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW);
    let fixture = Fixture::new(GgufType::Q4K, source.clone());
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let total = projector.memory_info().expect("query device memory").total;
    let huge = oversized_quant("huge", oversized_rows(total));
    let small = QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows: N_ROWS,
        k: K,
        source_len: source.len(),
    };
    let policy = LoadPolicy {
        retry_pause: Duration::ZERO,
        ..LoadPolicy::INTEGRATED
    };
    let retries = Cell::new(0);
    let before = projector.live_bytes();
    let (mut calls, mut betweens) = (0, 0);
    let mut live_at_entry = Vec::new();
    let mut first_was_retriable = false;
    let uploaded = retry_on_oom(
        &policy,
        &retries,
        || {
            live_at_entry.push(projector.live_bytes());
            calls += 1;
            if calls > 1 {
                let mut read = |at: usize, buf: &mut [u8]| -> Result<(), EngError> {
                    buf.copy_from_slice(&source[at..at + buf.len()]);
                    Ok(())
                };
                return projector
                    .context
                    .upload_tiled(small, CHUNK_LIMIT, &mut read);
            }
            let mut never = |_: usize, _: &mut [u8]| -> Result<(), EngError> {
                panic!("the reader of the oversized tensor must not run")
            };
            let outcome = projector
                .context
                .upload_tiled(huge, CHUNK_LIMIT, &mut never);
            let error = expect_error(outcome);
            first_was_retriable = is_device_out_of_memory(&error);
            Err(error)
        },
        || {
            betweens += 1;
            Ok(())
        },
    )
    .expect("the second attempt uploads the small tensor");
    // The classifier is pinned to the real error text of the driver refusal.
    assert!(
        first_was_retriable,
        "the real refusal is not classified as retriable"
    );
    assert_eq!((calls, betweens, retries.get()), (2, 1, 1));
    assert_eq!(
        live_at_entry,
        [before, before],
        "the failed attempt left device bytes"
    );
    let tile_bytes = N_ROWS / TILE_ROWS * BLOCKS_PER_ROW * Q4K_TILE_BYTES;
    assert_eq!(uploaded.buffer().bytes(), tile_bytes);
    assert_eq!(projector.live_bytes() - before, tile_bytes);
    drop(uploaded);
    assert_eq!(projector.live_bytes(), before);
    assert_loads_and_runs_after(&mut projector, &fixture, "q4k after a retried attempt");
}

/// Shared body of L2 and L3: `w` uploads, `huge` is refused by the device with
/// `retries` retries first, and everything is freed again.
fn assert_persistent_refusal(name: &str, integrated: bool, retries: u32) {
    if !device_present(name) || skip_if_integrated(name) {
        return;
    }
    let mut projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_integrated(integrated);
    let total = projector.memory_info().expect("query device memory").total;
    let (_dir, weights) = sparse_oom_fixture(total);
    let policy = LoadPolicy {
        retry_pause: Duration::ZERO,
        ..LoadPolicy::for_device(integrated)
    };
    let before = projector.live_bytes();
    let plan = plan_of(&weights, &["w", "huge"], policy, None);
    let error = expect_error(projector.transfer(&weights, plan));
    assert!(is_device_out_of_memory(&error), "got {error:?}");
    assert_eq!(
        projector.live_bytes(),
        before,
        "w was not freed by the unwind"
    );
    assert_eq!(projector.oom_retries(), retries);
    let fixture = Fixture::new(
        GgufType::Q4K,
        q4k_blocks(&mut Rng(SEED), N_ROWS * BLOCKS_PER_ROW),
    );
    assert_loads_and_runs_after(&mut projector, &fixture, "q4k after a refused tensor");
}

#[test]
fn a_persistent_allocation_failure_on_an_integrated_policy_retries_twice_and_frees_everything() {
    assert_persistent_refusal(
        "a_persistent_allocation_failure_on_an_integrated_policy_retries_twice_and_frees_everything",
        true,
        OOM_RETRIES,
    );
}

#[test]
fn a_persistent_allocation_failure_on_a_discrete_policy_is_not_retried() {
    assert_persistent_refusal(
        "a_persistent_allocation_failure_on_a_discrete_policy_is_not_retried",
        false,
        0,
    );
}

#[test]
fn an_integrated_load_evicts_the_file_before_admission() {
    const NAME: &str = "an_integrated_load_evicts_the_file_before_admission";
    if !device_present(NAME) {
        return;
    }
    for integrated in [true, false] {
        let fixture = big_and_small_fixture();
        let path = fixture_path(&fixture);
        if !droppable(&path, NAME) {
            return;
        }
        warm(&path);
        assert!(resident_bytes(&path).is_some_and(|bytes| bytes > EVICTED_MAX));
        let projector = CudaProjector::new()
            .expect("create the CUDA projector")
            .with_integrated(integrated);
        let options = LoadOptions {
            reserve: Reserve {
                headroom: u64::MAX / 2,
                ..Reserve::default()
            },
            drop_page_cache: None,
        };
        assert_error(
            projector.load(&fixture.weights, &["w"], options),
            "MemoryPlan",
            "cuda admission",
        );
        let resident = resident_bytes(&path).expect("fincore after the refused load");
        assert_eq!(
            resident <= EVICTED_MAX,
            integrated,
            "integrated={integrated}: {resident} bytes resident after the refusal"
        );
    }
}

#[test]
fn an_integrated_load_drops_each_tensor_when_the_caller_passes_no_file() {
    const NAME: &str = "an_integrated_load_drops_each_tensor_when_the_caller_passes_no_file";
    if !device_present(NAME) {
        return;
    }
    for integrated in [true, false] {
        let fixture = big_and_small_fixture();
        let path = fixture_path(&fixture);
        if !droppable(&path, NAME) {
            return;
        }
        warm(&path);
        let projector = CudaProjector::new()
            .expect("create the CUDA projector")
            .with_integrated(integrated);
        let loaded = projector
            .load(&fixture.weights, &["big", "w"], LoadOptions::default())
            .expect("load big and w");
        drop(loaded);
        let resident = resident_bytes(&path).expect("fincore after the load");
        let length = std::fs::metadata(&path).expect("fixture length").len();
        if integrated {
            assert!(
                resident <= LOAD_RESIDUAL_MAX,
                "{resident} bytes resident after the load"
            );
            assert_eq!(projector.oom_retries(), 0);
        } else {
            // Control: a discrete policy keeps the caller's choice, here no drop.
            assert!(
                resident >= length / 2,
                "{resident} of {length} bytes resident"
            );
        }
    }
}

#[test]
fn a_callers_file_wins_over_the_forced_drop() {
    const NAME: &str = "a_callers_file_wins_over_the_forced_drop";
    if !device_present(NAME) {
        return;
    }
    let fixture = big_and_small_fixture();
    let projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_integrated(true);
    let before = projector.live_bytes();
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    let text = error_text(projector.load(&fixture.weights, &["big", "w"], options));
    assert!(
        text.starts_with("Backend") && text.contains("cuda adapter"),
        "expected a non-terminal Backend cuda adapter error, got {text}"
    );
    assert!(
        text.contains("posix_fadvise") && text.contains("errno 29"),
        "the error must carry the status and ESPIPE, got {text}"
    );
    assert_eq!(
        projector.live_bytes(),
        before,
        "the failed load leaked device bytes"
    );
}

#[test]
fn an_integrated_load_evicts_again_after_the_scale_scan() {
    const NAME: &str = "an_integrated_load_evicts_again_after_the_scale_scan";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED);
    let (dir, weights) = open_fixture(&[
        weight_entry(
            "w",
            GgufType::Q4K,
            (N_ROWS, K),
            q4k_blocks(&mut rng, N_ROWS * BLOCKS_PER_ROW),
        ),
        weight_entry(
            "q6",
            GgufType::Q6K,
            (SCAN_Q6_ROWS, K),
            q6k_blocks(&mut rng, SCAN_Q6_ROWS * BLOCKS_PER_ROW),
        ),
    ]);
    let path = dir.path().join("fixture.gguf");
    if !droppable(&path, NAME) {
        return;
    }
    warm(&path);
    let projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_integrated(true);
    // The pipe fails the drop right after `w`: the failure point is the first
    // upload, so the scan of `q6` and both evictions have run by then.
    let pipe = pipe_file();
    let options = LoadOptions {
        drop_page_cache: Some(&pipe),
        ..LoadOptions::default()
    };
    let text = error_text(projector.load(&weights, &["w", "q6"], options));
    assert!(
        text.contains("posix_fadvise") && text.contains("errno 29"),
        "the load must fail at the pipe drop, got {text}"
    );
    // The scan read the 3.4 MB of `q6` back into the cache. Only the second
    // eviction removes them; what is left is `w` and a readahead tail.
    let resident = resident_bytes(&path).expect("fincore after the failed load");
    assert!(
        resident < 1 << 20,
        "{resident} bytes resident after the failed load"
    );
}

const Q5_0_BLOCK_BYTES: usize = 22;
/// One Q5_0 tile: 8 rows of 128 values, 44 sixteen-byte vectors.
const Q5_0_TILE_BYTES: usize = 44 * 16;
/// Values per row of a Q5_0 tile, and Mellum's `ffn_down_exps` row length.
const Q5_0_TILE_COLS: usize = 128;
const Q5_0_K: usize = 896;

impl Layout {
    /// Q5_0 in 128-value units: the four 22-byte blocks of one row and tile
    /// column are contiguous in the source, so they act as one 88-byte block.
    pub(super) fn q5_0(n_rows: usize, tiles_per_row: usize) -> Self {
        Self {
            block_bytes: 4 * Q5_0_BLOCK_BYTES,
            tile_bytes: Q5_0_TILE_BYTES,
            n_rows,
            blocks_per_row: tiles_per_row,
        }
    }
}

/// The Q5_0 source bytes rebuilt from the device tiles. Written from the tile
/// diagram of `tile_q5_0.rs`, walking the destination (source) blocks.
pub(super) fn decode_q5_0(tiles: &[u8], layout: Layout) -> Vec<u8> {
    let mut source = vec![0u8; layout.block_count() * layout.block_bytes];
    for row in 0..layout.n_rows {
        for column in 0..layout.blocks_per_row {
            let tile = &tiles[layout.tile_range(row, column)];
            let out = &mut source[layout.block_range(row, column)];
            for (j, block) in out
                .as_chunks_mut::<Q5_0_BLOCK_BYTES>()
                .0
                .iter_mut()
                .enumerate()
            {
                let b = 4 * (row % TILE_ROWS) + j;
                block[..2].copy_from_slice(&tile[2 * b..2 * b + 2]);
                block[2..6].copy_from_slice(&tile[64 + 4 * b..64 + 4 * b + 4]);
                block[6..].copy_from_slice(&tile[192 + 16 * b..192 + 16 * b + 16]);
            }
        }
    }
    source
}

/// Q5_0 blocks: a small fp16 `d`, then random `qh` and `qs` bytes.
fn q5_0_blocks(rng: &mut Rng, blocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks * Q5_0_BLOCK_BYTES);
    for _ in 0..blocks {
        out.extend_from_slice(&small_f16(rng));
        out.extend((0..Q5_0_BLOCK_BYTES - 2).map(|_| rng.byte()));
    }
    out
}

/// Loads a Q5_0 tensor `w` of `n_rows` x 896 under `chunk_limit` and checks
/// that its tiles decode to the source bytes and dequantize to the same bits.
fn q5_0_round_trip(n_rows: usize, chunk_limit: usize, label: &str) {
    let mut rng = Rng(SEED ^ 5);
    let blocks_per_row = Q5_0_K / 32;
    let source = q5_0_blocks(&mut rng, n_rows * blocks_per_row);
    let (_dir, weights) = open_fixture(&[weight_entry(
        "w",
        GgufType::Q50,
        (n_rows, Q5_0_K),
        source.clone(),
    )]);
    let projector = CudaProjector::new()
        .expect("create the CUDA projector")
        .with_chunk_limit(chunk_limit);
    let before = projector.live_bytes();
    let loaded = projector
        .load(&weights, &["w"], LoadOptions::default())
        .expect("load the Q5_0 weight");
    let weight = tiled_of(&loaded, "w");
    let tiles = download_tiles(&projector, weight);
    assert_eq!(weight.n_rows(), n_rows);
    assert_eq!(tiles.len(), source.len(), "{label}: no padding bytes");
    assert_eq!(projector.live_bytes() - before, source.len(), "{label}");
    let layout = Layout::q5_0(n_rows, Q5_0_K / Q5_0_TILE_COLS);
    let decoded = decode_q5_0(&tiles, layout);
    assert_same_bytes(label, &decoded, &source);
    let from_device = dequantize(GgufType::Q50, &decoded).expect("dequantize the decoded tiles");
    let from_source = dequantize(GgufType::Q50, &source).expect("dequantize the source");
    assert_bits_equal(label, &from_device, &from_source);
    println!("T1 {label}: {} tile bytes equal the source", tiles.len());
}

#[test]
fn q5_0_tiles_decode_to_source_bytes_and_dequantize_bit_exact() {
    const NAME: &str = "q5_0_tiles_decode_to_source_bytes_and_dequantize_bit_exact";
    if !device_present(NAME) {
        return;
    }
    q5_0_round_trip(16, CHUNK_BYTES, "q5_0 16 x 896");
}

#[test]
fn q5_0_tiles_spanning_two_upload_chunks_decode_bit_exactly() {
    const NAME: &str = "q5_0_tiles_spanning_two_upload_chunks_decode_bit_exactly";
    if !device_present(NAME) {
        return;
    }
    // One group of K = 896 is 7 tiles of 704 bytes: a limit of one group gives
    // three chunks for 24 rows.
    q5_0_round_trip(24, 7 * Q5_0_TILE_BYTES, "q5_0 24 x 896 in 3 chunks");
}

/// `x · Wᵀ` of the Q5_0 `source` rows (`k` values each) per token of `x`, summed in f64 from the
/// exactly dequantized weights. The activations of the tests below are exact in fp16, so the
/// device's fp16 conversion of x changes nothing in the reference.
fn q5_0_reference(source: &[u8], k: usize, x: &[f32]) -> Vec<f32> {
    let weights = dequantize(GgufType::Q50, source).expect("dequantize the Q5_0 source");
    let mut out = Vec::new();
    for token in x.chunks_exact(k) {
        for row in weights.chunks_exact(k) {
            let sum: f64 = row
                .iter()
                .zip(token)
                .map(|(w, x)| f64::from(*w) * f64::from(*x))
                .sum();
            out.push(sum as f32);
        }
    }
    out
}

/// Per-token T3 gate for `rows` outputs: the NMSE within the bound, and the argmax from 64 rows.
fn assert_rows_match(label: &str, device: &[f32], reference: &[f32], rows: usize) {
    assert_eq!(device.len(), reference.len(), "{label}: output length");
    let mut worst = 0.0f64;
    for (token, (got, want)) in device
        .chunks_exact(rows)
        .zip(reference.chunks_exact(rows))
        .enumerate()
    {
        let nmse = normalized_mse(got, want);
        assert!(
            nmse <= NMSE_BOUND,
            "{label}: token {token} NMSE {nmse:e} exceeds the bound {NMSE_BOUND:e}"
        );
        if rows >= 64 {
            assert_eq!(argmax(got), argmax(want), "{label}: token {token} argmax");
        }
        worst = worst.max(nmse);
    }
    println!(
        "T3 {label}: max NMSE {worst:.3e} over {} tokens (bound {NMSE_BOUND:e})",
        device.len() / rows
    );
}

/// Tokens of integers in [-4, 4] over eighths: exact in fp16.
fn exact_fp16_input(tokens: usize, k: usize) -> Vec<f32> {
    (0..tokens * k)
        .map(|i| ((i * 7 + i / k) % 17) as f32 / 4.0 - 2.0)
        .collect()
}

#[test]
fn a_q5_0_weight_runs_through_linear_and_the_guard_is_gone() {
    const NAME: &str = "a_q5_0_weight_runs_through_linear_and_the_guard_is_gone";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED ^ 6);
    let source = q5_0_blocks(&mut rng, N_ROWS * (Q5_0_K / 32));
    let (_dir, weights) = open_fixture(&[weight_entry(
        "q5",
        GgufType::Q50,
        (N_ROWS, Q5_0_K),
        source.clone(),
    )]);
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let loaded = projector
        .load(&weights, &["q5"], LoadOptions::default())
        .expect("load the Q5_0 weight");
    // One token through the trait: the whole weight.
    let x = exact_fp16_input(1, Q5_0_K);
    let device = projector
        .linear(weight_of(&loaded, "q5"), &x)
        .expect("linear, one token");
    assert_rows_match(
        "q5_0 linear 64 x 896 m=1",
        &device,
        &q5_0_reference(&source, Q5_0_K, &x),
        N_ROWS,
    );
    // Nine tokens against rows 8..16 through the context, as an expert read does.
    let x9 = exact_fp16_input(9, Q5_0_K);
    let tiled = tiled_of(&loaded, "q5");
    let device = projector
        .context
        .linear_rows(tiled, 8, 8, &x9)
        .expect("linear_rows, nine tokens");
    let row_bytes = Q5_0_K / 32 * Q5_0_BLOCK_BYTES;
    let reference = q5_0_reference(&source[8 * row_bytes..16 * row_bytes], Q5_0_K, &x9);
    assert_rows_match("q5_0 linear_rows 8..16 m=9", &device, &reference, 8);
    // Seventeen tokens are refused in Rust before any native call.
    let x17 = exact_fp16_input(17, Q5_0_K);
    assert_error(
        projector.context.linear_rows(tiled, 0, N_ROWS, &x17),
        "BadRequest",
        "16 tokens",
    );
    drop(loaded);
    let fixture = big_and_small_fixture();
    assert_loads_and_runs_after(&mut projector, &fixture, "after a refused Q5_0 matmul");
}

#[test]
fn a_q5_0_tensor_with_a_scale_past_the_fp16_range_fails_the_load_before_any_upload() {
    const NAME: &str =
        "a_q5_0_tensor_with_a_scale_past_the_fp16_range_fails_the_load_before_any_upload";
    if !device_present(NAME) {
        return;
    }
    let mut rng = Rng(SEED ^ 7);
    let mut source = q5_0_blocks(&mut rng, 8 * (Q5_0_K / 32));
    // Block 3: d = 4096 (0x6c00), and 16 * 4096 is past the fp16 maximum 65504.
    source[3 * Q5_0_BLOCK_BYTES..3 * Q5_0_BLOCK_BYTES + 2]
        .copy_from_slice(&0x6c00u16.to_le_bytes());
    let (_dir, weights) = open_fixture(&[weight_entry("q5", GgufType::Q50, (8, Q5_0_K), source)]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let result = projector.load(&weights, &["q5"], LoadOptions::default());
    assert_eq!(projector.live_bytes(), before, "the refused load uploaded");
    assert_error(
        result,
        "Checkpoint",
        "tensor q5 block 3: Q5_0 scale 4096 times quant 16 exceeds the fp16 range",
    );
}

#[test]
fn a_q5_0_tensor_with_k_off_the_grid_fails_the_load_before_any_upload() {
    const NAME: &str = "a_q5_0_tensor_with_k_off_the_grid_fails_the_load_before_any_upload";
    if !device_present(NAME) {
        return;
    }
    // 864 = 27 blocks: a whole number of blocks but not of 128-value tiles.
    let payload = vec![0u8; 8 * 27 * Q5_0_BLOCK_BYTES];
    let (_dir, weights) = open_fixture(&[weight_entry("q5", GgufType::Q50, (8, 864), payload)]);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let before = projector.live_bytes();
    let result = projector.load(&weights, &["q5"], LoadOptions::default());
    assert_eq!(projector.live_bytes(), before, "the refused load uploaded");
    assert_error(result, "Checkpoint", "tensor q5 (Q50) has 8 rows of 864");
}
