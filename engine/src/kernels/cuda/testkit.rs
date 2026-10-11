//! Helpers shared by the tests of the resident-decode owners (RD-2). RD-2a-1 wrote
//! the device check and the upload helper; RD-2a-2 added the weight, input and
//! read-back helpers that its launch tests and the later RD-2 tests share. Helpers
//! that only one test file uses stay in that file.

use super::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, QuantTensor, TiledWeight,
};
use crate::weights::GgufType;

/// Mirrors `fault.rs`: a host with no device skips the test, unless
/// `INFERENCE_CUDA_REQUIRE_DEVICE` turns the missing device into a failure.
pub(super) fn device_present(test: &str) -> bool {
    let count = super::device_count().expect("CUDA device count query");
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

/// Deterministic xorshift64* generator for the test data of the resident kernels (RD-2p). The
/// kernel test modules draw from it instead of keeping local copies; one seed always gives one
/// sequence, so a failing case replays.
pub(crate) struct Rng(u64);

impl Rng {
    /// A generator for `seed`. The all-zero state is a fixed point of xorshift, so seed 0 maps to
    /// a fixed non-zero state.
    pub(crate) fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A value in `0..limit`. The modulo bias is negligible for test data. Panics for `limit` 0.
    pub(crate) fn below(&mut self, limit: u64) -> u64 {
        assert!(limit > 0, "the Rng limit must be positive");
        self.next_u64() % limit
    }
}

/// Draws of the `Rng` that the kernel tests share (stage 1 wave 1). They sit in a second `impl`
/// block, after the one that holds `new`, `next_u64` and `below`, so the staged `expect` of that
/// block stays independent of them. Each test file uses a subset, hence the `allow`.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
impl Rng {
    /// A uniform value in [0, 1): the top 53 bits of `next_u64` over 2^53. Every result is an
    /// exact f64 and 1.0 is never drawn.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// An approximately standard normal value: the sum of 12 `unit` draws minus 6 (mean 0,
    /// variance 1, support [-6, 6]). The Irwin-Hall sum is enough for test inputs and keeps the
    /// sequence a pure function of the seed.
    pub(crate) fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}

/// `value` as IEEE binary16 bits, rounded to nearest with ties to the even significand. It is
/// written from the bit fields, like the oracle `f32_to_f16_rne` in `context/tests.rs`
/// (bc7f391, lines 26 to 85), whose logic it repeats because that test module is private.
/// NaN gives 0x7e00, overflow gives infinity, and values below half of the smallest fp16
/// subnormal give a signed zero.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn f16_bits(value: f32) -> u16 {
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
        f16_normal(sign, e, full)
    } else {
        f16_subnormal(sign, e, full)
    }
}

/// Keeps 11 significant bits of the 24-bit significand, ties to even.
fn f16_normal(sign: u16, e: i32, full: u32) -> u16 {
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
fn f16_subnormal(sign: u16, e: i32, full: u32) -> u16 {
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

/// The f32 value of the binary16 `bits`: the decoder `f16_to_f32` of `context.rs` that the
/// production scale checks use, so the tests and those checks read fp16 the same way.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn f16_to_f32(bits: u16) -> f32 {
    super::context::f16_to_f32(bits)
}

/// `x / (1 + exp(-x))` in f64: the SiLU reference of the DeltaNet kernels (RD-7a, RD-7b).
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `ln(1 + exp(x))` in f64 with the vendored kernel's cut: above 20 the value is `x`.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn softplus_f64(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// The little-endian f32 values of the file at `path`, for fixtures and oracle dumps. An error
/// names the path; a length that is not a multiple of 4 is an error.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn read_f32_file(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "{}: {} bytes is not a whole number of f32",
            path.display(),
            bytes.len()
        ));
    }
    Ok(floats(&bytes))
}

/// FNV-1a over `bytes` (64 bit): a cheap hash to compare whole device buffers before and after a
/// refused call.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Binary16 bits of `value` rounded to nearest, ties to even, straight from the f64 (no f32 step, so
/// no double rounding). Overflow (a magnitude of 65520 or more, and infinity) gives the signed
/// infinity, subnormals are kept, NaN gives 0x7e00 and the sign of zero is kept.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn f64_to_f16_bits(value: f64) -> u16 {
    if value.is_nan() {
        return 0x7e00;
    }
    let sign = if value.is_sign_negative() {
        0x8000_u16
    } else {
        0
    };
    let magnitude = value.abs();
    if magnitude >= 65520.0 {
        return sign | 0x7c00;
    }
    // Below 2^-14 the spacing is the fixed 2^-24; the scaling is by a power of two, so it is exact.
    if magnitude < 6.103_515_625e-5 {
        return sign | (magnitude * 16_777_216.0).round_ties_even() as u16;
    }
    let mut exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let scale = f64::from_bits(((1023 + 10 - exponent) as u64) << 52);
    let mut significand = (magnitude * scale).round_ties_even() as u32;
    if significand >= 2048 {
        significand = 1024;
        exponent += 1;
    }
    if exponent > 15 {
        return sign | 0x7c00;
    }
    sign | (((exponent + 15) as u16) << 10) | (significand as u16 - 1024)
}

/// The value of binary16 `bits` as an ordered f64; an infinity counts as 65536 (the next binade),
/// so that the midpoint of 65504 and infinity is 65520, the overflow threshold of round to nearest.
fn f16_ordered(bits: u16) -> f64 {
    if bits & 0x7fff == 0x7c00 {
        return if bits & 0x8000 != 0 {
            -65536.0
        } else {
            65536.0
        };
    }
    f64::from(f16_to_f32(bits))
}

/// The E2 test of lane M (REPLY-44): `bits` equals `f64_to_f16_bits(reference)`, or it is the
/// neighbouring binary16 value and `reference` lies within 2^-20 (relative) of the rounding
/// midpoint between the two. A reference out of range needs the same signed infinity (the
/// neighbour rule gives 65504 against infinity a window of 2^-20 around 65520, nothing more).
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn e2_holds(bits: u16, reference: f64) -> bool {
    let want = f64_to_f16_bits(reference);
    // The two zeros are one value.
    if bits == want || (bits | want) & 0x7fff == 0 {
        return true;
    }
    if reference.is_nan() || bits & 0x7fff > 0x7c00 {
        return false;
    }
    let neighbours = (i32::from(bits & 0x7fff) - i32::from(want & 0x7fff)).abs() == 1;
    if !(bits & 0x8000 == want & 0x8000 && neighbours) {
        return false;
    }
    let midpoint = 0.5 * (f16_ordered(bits) + f16_ordered(want));
    (reference - midpoint).abs() <= reference.abs() * 2f64.powi(-20)
}

/// The binary16 bits of the matmul input `xf` as a row-major `[m][k]` table. `xf` holds the bytes
/// of a pack (little-endian uint4 words): fragment order for `m <= 16` (token `t` of tile `t / 8`,
/// lane `(t % 8 & 3) + 4 q + 16 (t % 8 >> 2)`, one uint4 per (tile, block of 256, wi, lane) with
/// the values `base .. base + 3` and `base + 32 .. base + 35`, `base = 256 ib + 64 q + 4 wi`) and
/// row-major fp16 `[m][k]` above 16 tokens, exactly as `write_xf` in the vendored
/// `frag_layout.cuh` writes them. `k` is a multiple of 256.
#[allow(
    dead_code,
    reason = "shared by the stage 1 kernel tests; each test file draws with a subset"
)]
pub(super) fn unpack_xf(xf: &[u8], m: usize, k: usize) -> Vec<u16> {
    let half = |vec: usize, j: usize| {
        let at = vec * 16 + 2 * j;
        u16::from_le_bytes([xf[at], xf[at + 1]])
    };
    let mut out = vec![0_u16; m * k];
    if m > 16 {
        for token in 0..m {
            for e in 0..k / 8 {
                for j in 0..8 {
                    out[token * k + 8 * e + j] = half(token * (k / 8) + e, j);
                }
            }
        }
        return out;
    }
    let blocks = k / 256;
    for token in 0..m {
        let (tile, r) = (token / 8, token % 8);
        for ib in 0..blocks {
            for q in 0..4 {
                for wi in 0..8 {
                    let lane = (r & 3) + 4 * q + 16 * (r >> 2);
                    let vec = ((tile * blocks + ib) * 8 + wi) * 32 + lane;
                    let base = ib * 256 + 64 * q + 4 * wi;
                    for j in 0..4 {
                        out[token * k + base + j] = half(vec, j);
                        out[token * k + base + 32 + j] = half(vec, 4 + j);
                    }
                }
            }
        }
    }
    out
}

/// A device buffer that holds `bytes`.
pub(super) fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> DeviceBuffer {
    ctx.upload(bytes).expect("upload of the test bytes")
}

/// Bytes of one ggml block of `ty` and the blocks that hold 256 values.
fn block_shape(ty: GgufType) -> (usize, usize) {
    match ty {
        GgufType::Q4K => (144, 1),
        GgufType::Q6K => (210, 1),
        GgufType::Q80 => (34, 8),
        other => panic!("the test weights are Q4_K, Q6_K or Q8_0, not {other:?}"),
    }
}

/// Sets the fp16 scale fields of one block: `d` = 1.0, and `dmin` = 0.5 for Q4_K.
fn set_scales(ty: GgufType, block: &mut [u8]) {
    // Q6_K keeps `d` in the last two bytes of its block; the others in the first two.
    let d_at = if ty == GgufType::Q6K { 208 } else { 0 };
    block[d_at..d_at + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
    if ty == GgufType::Q4K {
        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
    }
}

/// ggml blocks of an `n_rows` x `k` weight: deterministic xorshift bytes with the
/// scale fields set, so every dequantized value is finite and small.
pub(super) fn weight_source(ty: GgufType, n_rows: usize, k: usize) -> Vec<u8> {
    let (block, per_256) = block_shape(ty);
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ ((n_rows * 31 + k) as u64);
    let mut bytes: Vec<u8> = (0..n_rows * (k / 256) * per_256 * block)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        })
        .collect();
    for chunk in bytes.chunks_exact_mut(block) {
        set_scales(ty, chunk);
    }
    bytes
}

/// A tiled weight of the `weight_source` blocks.
pub(super) fn upload_weight(
    ctx: &CudaContext,
    ty: GgufType,
    n_rows: usize,
    k: usize,
) -> TiledWeight {
    let source = weight_source(ty, n_rows, k);
    let tensor = QuantTensor {
        name: "test weight",
        ty,
        n_rows,
        k,
        source_len: source.len(),
    };
    ctx.upload_tiled(tensor, CHUNK_LIMIT, &mut |offset, buffer| {
        buffer.copy_from_slice(&source[offset..offset + buffer.len()]);
        Ok(())
    })
    .expect("upload of the test weight")
}

/// `tokens` rows of `k` small integers in [-4, 4], exact in fp16.
pub(super) fn input(tokens: usize, k: usize) -> Vec<f32> {
    (0..tokens * k)
        .map(|i| f32::from(u8::try_from((i * 5 + i / k) % 9).expect("below 9")) - 4.0)
        .collect()
}

/// A device buffer that holds `values`.
pub(super) fn upload_f32(ctx: &CudaContext, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    upload_bytes(ctx, &bytes)
}

/// A device buffer of `len` bytes that all equal `byte`, so a byte a launch did
/// not write is visible afterwards.
pub(super) fn upload_filled(ctx: &CudaContext, byte: u8, len: usize) -> DeviceBuffer {
    upload_bytes(ctx, &vec![byte; len])
}

/// The little-endian f32 values of `bytes`.
pub(super) fn floats(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The first `len` bytes of a device buffer. The download waits for the device.
pub(super) fn read_bytes(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    ctx.download(buffer, &mut bytes).expect("download");
    bytes
}

/// Rows `first_row..first_row + rows` of `x · Wᵀ` by the blocking path: the
/// default stream, a host copy in and out, and a device wait.
pub(super) fn blocking(
    ctx: &CudaContext,
    weight: &TiledWeight,
    rows: (usize, usize),
    x: &[f32],
) -> Vec<f32> {
    ctx.linear_rows(weight, rows.0, rows.1, x)
        .expect("blocking linear")
}

/// Repeats of one projection in the seam-1 determinism tests (decision 0047 item 7).
pub(super) const REPEATS: usize = 1_000;
/// Repeats queued, synchronized and compared together; bounds the host and device memory of a run.
const REPEAT_CHUNK: usize = 100;

/// The first run that differs from repeat 0, and the first differing byte of its output.
#[derive(Debug, PartialEq)]
pub(super) struct Mismatch {
    pub(super) repeat: usize,
    pub(super) index: usize,
}

/// Index of the first byte where the slices differ; a length difference counts at the shorter
/// length. Bytes are compared, not floats, so NaN payloads and signed zeros count.
pub(super) fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) {
        return Some(at);
    }
    (a.len() != b.len()).then(|| a.len().min(b.len()))
}

/// Queues `REPEATS` runs through `queue(repeat, y)`, `REPEAT_CHUNK` at a time, on `stream`.
///
/// One buffer holds a chunk of slots (a slot is `out_bytes` rounded up to 256). Per chunk the
/// slots are first written with a stale pattern, f32 `repeat + 1` repeated, so an output that
/// depends on the old content of `y` differs between repeats. Then each repeat is queued into
/// the first `out_bytes` bytes of its slot, the stream is synchronized, the chunk is read back
/// and every slot is compared with repeat 0. Padding bytes are never compared. Returns the bytes
/// of repeat 0, or the first mismatch. `queue` must `expect` its own launches.
pub(super) fn run_repeats(
    ctx: &CudaContext,
    stream: &DeviceStream,
    out_bytes: usize,
    mut queue: impl FnMut(usize, DeviceRange<'_>),
) -> Result<Vec<u8>, Mismatch> {
    let slot = out_bytes.next_multiple_of(256);
    let buffer = ctx
        .allocate(slot * REPEAT_CHUNK)
        .expect("allocate the repeat chunk");
    let mut want: Option<Vec<u8>> = None;
    for start in (0..REPEATS).step_by(REPEAT_CHUNK) {
        let count = REPEAT_CHUNK.min(REPEATS - start);
        let stale: Vec<u8> = (start..start + count)
            .flat_map(|repeat| {
                let pattern = (repeat as f32 + 1.0).to_le_bytes();
                pattern.into_iter().cycle().take(slot).collect::<Vec<u8>>()
            })
            .collect();
        buffer
            .write_at(stream, 0, &stale)
            .expect("write the stale y");
        for r in 0..count {
            let y = buffer.range(r * slot, out_bytes).expect("slot range");
            queue(start + r, y);
        }
        stream.synchronize().expect("drain the repeats");
        let mut chunk = vec![0u8; count * slot];
        buffer
            .read_at(stream, 0, &mut chunk)
            .expect("read the chunk");
        compare_chunk(&chunk, slot, out_bytes, start, &mut want)?;
    }
    Ok(want.expect("at least one repeat ran"))
}

/// Compares the slots of one chunk with repeat 0, which the first chunk records in `want`.
fn compare_chunk(
    chunk: &[u8],
    slot: usize,
    out_bytes: usize,
    start: usize,
    want: &mut Option<Vec<u8>>,
) -> Result<(), Mismatch> {
    for (r, bytes) in chunk.chunks_exact(slot).enumerate() {
        let got = &bytes[..out_bytes];
        let reference = want.get_or_insert_with(|| got.to_vec());
        if let Some(index) = first_difference(reference, got) {
            return Err(Mismatch {
                repeat: start + r,
                index,
            });
        }
    }
    Ok(())
}
