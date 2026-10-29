//! Helpers shared by the tests of the resident-decode owners: the device check, the
//! upload helper, and the weight, input and read-back helpers that the launch tests and
//! the other owner tests share. Helpers that only one test file uses stay in that file.

use std::sync::OnceLock;

use super::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, QuantTensor, TiledWeight,
};
use crate::EngError;
use crate::weights::GgufType;

/// `cudaErrorInsufficientDriver` of `cuda_runtime_api.h` (the build image's CUDA 12.2): the
/// runtime cannot use the host's driver library, the status a host with no CUDA driver at
/// all reports from `cudaDeviceCount`.
const CUDA_ERROR_INSUFFICIENT_DRIVER: i32 = 35;

/// `cudaErrorNoDevice` of `cuda_runtime_api.h`: a usable driver but no device. The adapter
/// already maps this status to a count of zero (`native/abi.h`), so reading it here is
/// defensive; like 35 it means "no usable CUDA driver or device".
const CUDA_ERROR_NO_DEVICE: i32 = 100;

/// The panic text of a run that demands a device on a host without one: the message the
/// `assert!` of the old per-file copies printed, kept so logs read the same.
const REQUIRE_DEVICE_MESSAGE: &str =
    "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible";

/// What a device test does with the result of [`super::device_count`].
#[derive(Debug, Clone, PartialEq)]
enum Presence {
    /// A device is visible; the test runs.
    Present,
    /// No driver or device, and none is required; the test prints SKIPPED.
    Skip,
    /// The probe failed a way that is not a missing driver or device, or a device is
    /// required and missing; the test panics with the message.
    Fail(String),
}

/// The decision of [`device_present`], pure so each of its branches is testable on a host
/// without a driver: a visible device is
/// [`Presence::Present`] whatever `required` says; a count of zero, or an error whose
/// status means "no usable CUDA driver or device" (35 and 100), skips unless a device is
/// required; every other error fails, required or not, keeping its text.
fn presence(count: Result<i32, EngError>, required: bool) -> Presence {
    let missing = match &count {
        Ok(visible) => *visible == 0,
        Err(error) => matches!(
            status_of(error),
            Some(CUDA_ERROR_INSUFFICIENT_DRIVER | CUDA_ERROR_NO_DEVICE)
        ),
    };
    if missing {
        return if required {
            Presence::Fail(REQUIRE_DEVICE_MESSAGE.to_owned())
        } else {
            Presence::Skip
        };
    }
    match count {
        Ok(_) => Presence::Present,
        Err(error) => Presence::Fail(format!("CUDA device count query: {error:?}")),
    }
}

/// The runtime status of a failed count query. `EngError` has no structured status field,
/// so the "(status N)" suffix that `native_error` writes into every `Backend` message is
/// the only place it survives; other variants, and a message without the suffix, give
/// `None`.
fn status_of(error: &EngError) -> Option<i32> {
    let EngError::Backend { message, .. } = error else {
        return None;
    };
    let (_, digits) = message.rsplit_once("(status ")?;
    digits
        .strip_suffix(')')
        .and_then(|digits| digits.parse().ok())
}

/// The count of [`super::device_count`], widened back: it returned a `usize` converted
/// from a non-negative `c_int`, so the value always fits an `i32`.
fn probe_count() -> Result<i32, EngError> {
    super::device_count().map(|visible| visible as i32)
}

/// What the process's one device probe saw: the count, and the decision [`presence`]
/// made of it with `required` false (`Present` and `Skip` do not depend on `required`).
#[derive(Debug)]
struct Probed {
    /// The visible count, `None` when the probe failed.
    visible: Option<i32>,
    /// The decision for `required` false; a cached `Skip` still fails every test of a
    /// run that sets `INFERENCE_CUDA_REQUIRE_DEVICE`.
    decision: Presence,
}

impl Probed {
    /// The probe outcome of `count`.
    fn of(count: Result<i32, EngError>) -> Self {
        let visible = count.as_ref().copied().ok();
        Self {
            visible,
            decision: presence(count, false),
        }
    }
}

/// Caches one probe decision for a whole process: the no-driver
/// count error latches the backend TERMINAL inside `native_error`, so only the FIRST
/// probe sees status 35; every later `device_count()` of the same process would return
/// the terminal refusal instead. The gate therefore probes once and every test shares
/// the outcome, because device presence cannot change inside a process. The production
/// `device_count()` and its latch stay unchanged.
struct ProbeCache {
    probed: OnceLock<Probed>,
}

impl ProbeCache {
    /// An empty cache.
    const fn new() -> Self {
        Self {
            probed: OnceLock::new(),
        }
    }

    /// The cached decision, running `probe` only the first time.
    fn decision(&self, probe: impl FnOnce() -> Result<i32, EngError>) -> &Probed {
        self.probed.get_or_init(|| Probed::of(probe()))
    }
}

/// The process's one device probe.
static DEVICE_PROBE: ProbeCache = ProbeCache::new();

/// A host with no driver or device skips the test, unless `INFERENCE_CUDA_REQUIRE_DEVICE`
/// (non-empty and not "0") turns the missing device into a failure; a count-query error
/// that is not a missing driver still panics, with its text. Every CUDA device test of the
/// crate shares this gate: the kernels tests and `fault.rs` through `super::testkit`, the
/// family rigs through the `device_present` re-export of [`super`]. The count comes from
/// the one cached probe ([`DEVICE_PROBE`]): a second probe would see the TERMINAL latch
/// the first no-driver error set, not status 35.
pub(crate) fn device_present(test: &str) -> bool {
    let probed = DEVICE_PROBE.decision(probe_count);
    // The count line prints for every `Ok` on record, as before the cache: a runtime
    // that reports zero devices still prints "CUDA devices visible: 0".
    if let Some(visible) = probed.visible {
        println!("CUDA devices visible: {visible}");
    }
    match &probed.decision {
        Presence::Present => true,
        Presence::Skip => {
            let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
                .is_ok_and(|value| !value.is_empty() && value != "0");
            assert!(!required, "{REQUIRE_DEVICE_MESSAGE}");
            println!("SKIPPED {test}: no CUDA device");
            false
        }
        Presence::Fail(message) => panic!("{message}"),
    }
}

/// The environment marker of a latching test's child run (see `latching_test_in_child`).
const LATCHING_CHILD: &str = "INFERENCE_CUDA_LATCHING_CHILD";

/// A latching test's child must finish within this long; on expiry the helper kills the child
/// and fails the test. It is the same bound the other child helpers use (`resident_ops.rs:1612`,
/// `step.rs:769`).
const LATCHING_DEADLINE: std::time::Duration = std::time::Duration::from_secs(180);

/// Drains a child's pipe on its own thread, so a full pipe buffer cannot deadlock the parent
/// (the pattern of `resident_ops.rs:1614`).
fn drain<R: std::io::Read + Send + 'static>(mut pipe: R) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = pipe.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Runs a test whose body deliberately triggers a native refusal in a child process of the test
/// executable. A native refusal latches the CUDA backend terminal for the whole process
/// (`native.rs`), so in a one-process suite run every later device test would fail on the
/// latch. The pattern is `fault.rs:105`'s `run_child`
/// (CUDA) and `resident_ops.rs:1624`'s `delegated_to_child` (MLX): re-run the test executable on
/// this one test.
///
/// `path` is the test's full path below the crate. Returns `true` in the child, where the caller
/// runs its body. In the parent it runs the child with `--exact path` under a deadline, checks
/// that the child passed and printed every string of `expect` (the refusal text), prints the
/// child's output so the text reaches the suite log, and returns `false`.
pub(super) fn latching_test_in_child(path: &str, expect: &[&str]) -> bool {
    if std::env::var(LATCHING_CHILD).is_ok_and(|name| name == path) {
        return true;
    }
    let exe = std::env::current_exe().expect("the test executable");
    let mut child = std::process::Command::new(exe)
        .args(["--exact", path, "--test-threads=1", "--nocapture"])
        .env(LATCHING_CHILD, path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the latching test's child process");
    let stdout = drain(child.stdout.take().expect("the child's stdout pipe"));
    let stderr = drain(child.stderr.take().expect("the child's stderr pipe"));
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the child's status") {
            break Some(status);
        }
        if started.elapsed() >= LATCHING_DEADLINE {
            child.kill().expect("kill the latching test's child");
            child.wait().expect("reap the latching test's child");
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let captured = format!(
        "{} {}",
        stdout.join().expect("the child's stdout reader"),
        stderr.join().expect("the child's stderr reader")
    );
    let Some(status) = status else {
        panic!(
            "the latching test's child did not finish within {LATCHING_DEADLINE:?} and was \
             killed:\n{captured}"
        );
    };
    // A renamed test makes the child run nothing and exit 0, so require libtest's summary of
    // exactly one passed test (the MLX helpers' "1 passed" check, narrowed).
    assert!(
        status.success() && captured.contains("test result: ok. 1 passed"),
        "the latching test must pass in its child: {captured}"
    );
    for text in expect {
        assert!(
            captured.contains(text),
            "the child must print {text:?}: {captured}"
        );
    }
    println!("latching test child {path}:\n{captured}");
    false
}

/// Deterministic xorshift64* generator for the test data of the resident kernels. The
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

/// Draws of the `Rng` that the kernel tests share. They sit in a second `impl`
/// block, after the one that holds `new`, `next_u64` and `below`, so the staged `expect` of that
/// block stays independent of them. Each test file uses a subset, hence the `allow`.
#[allow(
    dead_code,
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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
    reason = "shared by the resident kernel tests; each test file draws with a subset"
)]
pub(super) fn f16_to_f32(bits: u16) -> f32 {
    super::context::f16_to_f32(bits)
}

/// `x / (1 + exp(-x))` in f64: the SiLU reference of the DeltaNet conv and gated-delta kernels.
#[allow(
    dead_code,
    reason = "shared by the resident kernel tests; each test file draws with a subset"
)]
pub(super) fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `ln(1 + exp(x))` in f64 with the vendored kernel's cut: above 20 the value is `x`.
#[allow(
    dead_code,
    reason = "shared by the resident kernel tests; each test file draws with a subset"
)]
pub(super) fn softplus_f64(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// The little-endian f32 values of the file at `path`, for fixtures and oracle dumps. An error
/// names the path; a length that is not a multiple of 4 is an error.
#[allow(
    dead_code,
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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

/// The binary16 rounding test: `bits` equals `f64_to_f16_bits(reference)`, or it is the
/// neighbouring binary16 value and `reference` lies within 2^-20 (relative) of the rounding
/// midpoint between the two. A reference out of range needs the same signed infinity (the
/// neighbour rule gives 65504 against infinity a window of 2^-20 around 65520, nothing more).
#[allow(
    dead_code,
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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
    reason = "shared by the resident kernel tests; each test file draws with a subset"
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

/// Repeats of one projection in the determinism tests.
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

/// `source` without its `//` and `/* */` comments: commenting a statement out must not satisfy a source pin. (Limits: a
/// backslash-continued `//` comment and `#if 0` blocks are not recognised; the compiled pins see both.)
pub(super) fn without_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, chars.peek()) {
            ('/', Some('/')) => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                chars.next();
                let mut previous = ' ';
                for d in chars.by_ref() {
                    if previous == '*' && d == '/' {
                        break;
                    }
                    previous = d;
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// The code of `source` with every run of whitespace collapsed to one space and no comments.
pub(super) fn squeezed(source: &str) -> String {
    without_comments(source)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Walks `steps` through `text` in order, each one after the end of the previous.
pub(super) fn assert_in_order(text: &str, steps: &[&str], what: &str) {
    let mut at = 0;
    for step in steps {
        match text[at..].find(step) {
            Some(found) => at += found + step.len(),
            None => panic!("{what}: `{step}` is missing or out of order"),
        }
    }
}

/// The trimmed PTX lines of the kernel whose entry name contains `kernel`, compiled by nvcc (`-arch=sm_80 -O2`) from the
/// embedded `files` (path under a scratch directory, text; `main` is the one to compile). `None`, with a note on stderr,
/// when nvcc is not found, unless `INFERENCE_CUDA_REQUIRE_DEVICE` is set (then it fails). `NVCC` names the compiler like
/// build.rs does. The scratch directory is always removed.
pub(super) fn kernel_ptx(kernel: &str, main: &str, files: &[(&str, &str)]) -> Option<Vec<String>> {
    let dir = std::env::temp_dir().join(format!("tcap-ptx-{kernel}-{}", std::process::id()));
    for (name, text) in files {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().expect("a file has a directory"))
            .expect("the scratch directory");
        std::fs::write(path, text).expect("a source file of the pin");
    }
    let ptx_path = dir.join("kernel.ptx");
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let compiled = std::process::Command::new(&nvcc)
        .args(["-ptx", "-arch=sm_80", "-O2", "-I"])
        .arg(&dir)
        .arg(dir.join(main))
        .arg("-o")
        .arg(&ptx_path)
        .output();
    let ptx = match compiled {
        Ok(output) if output.status.success() => std::fs::read_to_string(&ptx_path),
        Ok(output) => Err(std::io::Error::other(format!(
            "nvcc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let _ = std::fs::remove_dir_all(&dir);
            let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
                .is_ok_and(|v| !v.is_empty() && v != "0");
            assert!(!required, "nvcc is required here but not found: {error}");
            eprintln!("SKIPPED the compiled pin of {kernel}: `{nvcc}` not found ({error})");
            return None;
        }
        Err(error) => Err(error),
    };
    let _ = std::fs::remove_dir_all(&dir);
    let ptx = ptx.expect("the PTX of the kernel");
    let entry = ptx
        .match_indices(".entry")
        .map(|(at, _)| at)
        .find(|&at| ptx[at..].lines().next().is_some_and(|l| l.contains(kernel)))
        .expect("the kernel entry");
    let body = &ptx[entry..entry + ptx[entry..].find("\n}\n").expect("the end of the kernel")];
    Some(body.lines().map(|l| l.trim().to_owned()).collect())
}

/// The pure cases of the device-presence decision; they run on any host, driver or not.
mod presence_tests {
    use super::*;

    /// A `Backend` error exactly as `native_error` writes it for a failed count query.
    fn backend_error(status: i32) -> EngError {
        EngError::Backend {
            operation: "cuda initialize",
            message: format!("the runtime refused the count (status {status})"),
        }
    }

    /// The failure of any error that is not a missing driver or device.
    fn query_failure(error: &EngError) -> Presence {
        Presence::Fail(format!("CUDA device count query: {error:?}"))
    }

    #[test]
    fn a_visible_device_is_present_required_or_not() {
        assert_eq!(presence(Ok(1), false), Presence::Present);
        assert_eq!(presence(Ok(1), true), Presence::Present);
    }

    #[test]
    fn zero_devices_skip_unless_a_device_is_required() {
        assert_eq!(presence(Ok(0), false), Presence::Skip);
        assert_eq!(
            presence(Ok(0), true),
            Presence::Fail(REQUIRE_DEVICE_MESSAGE.to_owned())
        );
    }

    #[test]
    fn an_insufficient_driver_skips_unless_a_device_is_required() {
        assert_eq!(presence(Err(backend_error(35)), false), Presence::Skip);
        assert_eq!(
            presence(Err(backend_error(35)), true),
            Presence::Fail(REQUIRE_DEVICE_MESSAGE.to_owned())
        );
    }

    #[test]
    fn the_no_device_status_behaves_like_an_insufficient_driver() {
        assert_eq!(presence(Err(backend_error(100)), false), Presence::Skip);
        assert_eq!(
            presence(Err(backend_error(100)), true),
            Presence::Fail(REQUIRE_DEVICE_MESSAGE.to_owned())
        );
    }

    /// An unknown runtime status, a negative adapter status and a `Backend` message with
    /// no "(status N)" suffix are all failures, with the error's text, `required` or not.
    #[test]
    fn any_other_error_fails_required_or_not() {
        let other_error = |case: i32| match case {
            0 => backend_error(999),
            1 => EngError::Backend {
                operation: "cuda adapter",
                message: "the adapter entry is missing (status -3)".to_owned(),
            },
            _ => EngError::Backend {
                operation: "cuda initialize",
                message: "the adapter could not be loaded".to_owned(),
            },
        };
        for case in 0..3 {
            let want = query_failure(&other_error(case));
            assert_eq!(presence(Err(other_error(case)), false), want.clone());
            assert_eq!(presence(Err(other_error(case)), true), want);
        }
    }

    /// The probe runs once and every call shares its decision:
    /// the later probes would answer differently, but the cache never runs them, and a
    /// fresh cache probes again on its own.
    #[test]
    fn the_probe_runs_once_and_every_call_shares_its_decision() {
        use std::cell::Cell;
        let cache = ProbeCache::new();
        let calls = Cell::new(0);
        let run = |calls: &Cell<i32>| {
            calls.set(calls.get() + 1);
        };
        let first = cache.decision(|| {
            run(&calls);
            Err(backend_error(35))
        });
        assert_eq!(first.visible, None);
        assert_eq!(first.decision, Presence::Skip);
        for _ in 0..3 {
            let again = cache.decision(|| {
                run(&calls);
                Ok(1)
            });
            assert_eq!(again.visible, None, "the cached outcome of the first probe");
            assert_eq!(again.decision, Presence::Skip);
        }
        assert_eq!(calls.get(), 1, "the probe ran once across several calls");
        let second = ProbeCache::new();
        assert_eq!(second.decision(|| Ok(1)).visible, Some(1));
        assert_eq!(
            second.decision(|| Err(backend_error(35))).decision,
            Presence::Present
        );
    }
}
